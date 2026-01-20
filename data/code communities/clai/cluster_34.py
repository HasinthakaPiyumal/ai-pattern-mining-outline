# Cluster 34

class DemoAgent(Agent):

    def get_next_action(self, state: State) -> Union[Action, List[Action]]:
        logger.info('This is my agent')
        if state.command == 'ls':
            return Action(suggested_command='ls -la', description='This is a demo sample that helps to execute the command in better way.', confidence=1)
        if state.command == 'pwd':
            return [Action(suggested_command='ls -la', description='This is a demo sample that helps to execute the command in better way.', confidence=1), Action(suggested_command='pwd -P', description='This is a demo sample that helps to execute the command in better way.', confidence=1)]
        if state.previous_execution and state.previous_execution.command == 'ls -4':
            return Action(suggested_command='ls -a', execute=True, confidence=1)
        return Action(suggested_command=state.command)

    def post_execute(self, state: State) -> Action:
        if state.command.startswith('ls') and state.result_code != '0':
            return Action(description=Colorize().append(f'Are you sure that this command is correct?({state.result_code})\n').warning().append(f'Try man ls for more info ').to_console(), confidence=1)
        return Action(suggested_command=state.command)

class SocketClientConnector(ClientConnector):

    def __init__(self, host: str, port: int):
        self.sel = selectors.DefaultSelector()
        self.host = host
        self.port = port
        self.uuid = uuid.uuid4()

    def send(self, message: StateDTO) -> Action:
        try:
            return self._internal_send(message)
        except Exception as error:
            logger.info(f'error {error}')
            logger.info(traceback.format_exc())
            return Action(origin_command=message.command, suggested_command=message.command)
        finally:
            self.close()

    def _internal_send(self, command_to_send):
        self.start_connections(self.host, int(self.port))
        self.write(command_to_send)
        action = self.read()
        if action:
            return action
        return Action(origin_command=command_to_send.command, suggested_command=command_to_send.command)

    def start_connections(self, host, port):
        server_address = (host, port)
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setblocking(False)
        client_socket.connect_ex(server_address)
        events = selectors.EVENT_READ | selectors.EVENT_WRITE
        data = types.SimpleNamespace(connid=self.uuid, outb=b'')
        self.sel.register(client_socket, events, data=data)

    def write(self, message: StateDTO):
        events = self.sel.select(timeout=5)
        key = events[0][0]
        client_socket = key.fileobj
        data = key.data
        self.sel.modify(client_socket, selectors.EVENT_WRITE, data)
        logger.info(f'echoing ${data}')
        data.outb = str(message.json())
        sent = client_socket.send(data.outb.encode('utf-8'))
        data.outb = data.outb[sent:]
        self.sel.modify(client_socket, selectors.EVENT_READ, data)

    def read(self) -> Optional[Action]:
        events = self.sel.select(timeout=6)
        if events and events[0]:
            key = events[0][0]
            client_socket = key.fileobj
            received_data = client_socket.recv(4024)
            if received_data:
                message = process_message(received_data)
                return message
        return None

    def close(self):
        self.sel.close()

class SocketServerConnector(ServerConnector):
    BUFFER_SIZE = 4024

    def __init__(self, server_status_datasource: ServerStatusDatasource):
        self.server_status_datasource = server_status_datasource
        self.sel = selectors.DefaultSelector()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def create_socket(self, host: str, port: int):
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen()
        logger.info(f'Listening {host} {port}')
        self.server_socket.setblocking(False)

    def loop(self, process_message: Callable[[bytes], Action]):
        self.sel.register(self.server_socket, selectors.EVENT_READ, data=None)
        try:
            while self.server_status_datasource.running:
                events = self.sel.select(timeout=None)
                for key, mask in events:
                    if key.data is None:
                        self.__accept_wrapper(key.fileobj)
                    else:
                        self.__service_connection(key, mask, process_message)
            self.sel.unregister(self.server_socket)
            self.server_socket.close()
        except KeyboardInterrupt:
            logger.info('caught keyboard interrupt, exiting')
        finally:
            logger.info('server closed')
            self.sel.close()

    def __accept_wrapper(self, server_socket):
        connection, address = server_socket.accept()
        connection.setblocking(False)
        data = types.SimpleNamespace(addr=address, inb=b'', outb=b'')
        events = selectors.EVENT_READ | selectors.EVENT_WRITE
        self.sel.register(connection, events, data=data)

    def __service_connection(self, key, mask, process_message):
        fileobj = key.fileobj
        data = key.data
        if mask & selectors.EVENT_READ:
            data = self.__read(data, fileobj, process_message)
        if mask & selectors.EVENT_WRITE:
            self.__write(data, fileobj)

    @staticmethod
    def __write(data, server_socket):
        if data.outb:
            logger.info(f'sending from client ${data.outb}')
            server_socket.send(data.outb)
            data.outb = b''

    def __read(self, data, server_socket, process_message):
        recv_data = b''
        chewing = True
        logger.info(f'receiving from client')
        while chewing:
            part = server_socket.recv(self.BUFFER_SIZE)
            recv_data += part
            if len(part) < self.BUFFER_SIZE:
                chewing = False
        if recv_data:
            logger.info(f'receiving from client ${recv_data}')
            action = process_message(recv_data)
            data.outb = str(action.json()).encode('utf8')
        else:
            self.sel.unregister(server_socket)
            server_socket.close()
        return data

class MessageHandler:

    def __init__(self, server_status_datasource: ServerStatusDatasource, agent_datasource: AgentDatasource):
        self.agent_datasource = agent_datasource
        orchestrator_provider = OrchestratorProvider(agent_datasource)
        self.agent_runner = AgentRunner(self.agent_datasource, orchestrator_provider)
        self.server_status_datasource = server_status_datasource
        self.server_pending_actions_datasource = ServerPendingActionsDatasource()
        self.command_runner_factory = CommandRunnerFactory(self.agent_datasource, config_storage, self.server_status_datasource, orchestrator_provider)

    def init_server(self):
        self.agent_datasource.preload_plugins()

    def process_post_command(self, message: State) -> Action:
        message = self.complete_history(message)
        command_runner = self.command_runner_factory.provide_post_command_runner(message.command, self.agent_runner)
        action = command_runner.execute_post(message)
        if not action:
            action = Action()
        action.origin_command = message.command
        return action

    def __process_command_ai(self, message) -> List[Action]:
        if message.command == STOP_COMMAND:
            self.server_status_datasource.running = False
            return [Action()]
        command_runner = self.command_runner_factory.provide_command_runner(message.command, self.agent_runner)
        action = command_runner.execute(message)
        if isinstance(action, Action):
            return [action]
        return action

    def __process_command(self, message: State) -> Action:
        if not message.is_already_processed():
            message.previous_execution = self.server_status_datasource.get_last_message(message.user_name)
            actions = self.__process_command_ai(message)
            message.mark_as_processed()
            logger.info(f'after setting info: {message.is_already_processed()}')
            self.server_status_datasource.store_info(message)
            action = self.server_pending_actions_datasource.store_pending_actions(message.command_id, actions, message.user_name)
        else:
            logger.info(f'we have pending action')
            action = self.server_pending_actions_datasource.get_next_action(message.command_id, message.user_name)
        if action is None:
            action = Action(suggested_command=message.command, origin_command=message.command, execute=False)
        action.origin_command = message.command
        if message.is_post_process():
            message.action_post_suggested = action
        else:
            message.action_suggested = action
        self.server_status_datasource.store_info(message)
        action.execute = action.execute or self.server_status_datasource.is_power()
        return action

    def process_message(self, message: State) -> Action:
        try:
            message = self.server_status_datasource.store_info(message)
            if message.is_post_process():
                message = self.server_status_datasource.find_message_stored(message.command_id, message.user_name)
                return self.process_post_command(message)
            if message.is_command():
                return self.__process_command(message)
        except Exception as ex:
            logger.info(f'error processing message {ex}')
            logger.info(traceback.format_exc())
        return Action(origin_command=message.command)

    def find_value(self, lines, message: State) -> Optional[int]:
        for i in reversed(range(len(lines))):
            if self.message_executed(lines[i], message):
                return i
        return None

    def complete_history(self, message: State):
        lines = read_history()
        index = self.find_value(lines, message)
        if index:
            last_values = lines[index:]
            message.values_executed = last_values
            if message.action_suggested.suggested_command and message.action_suggested.suggested_command in last_values[0]:
                message.suggested_executed = True
        else:
            message.values_executed = []
            message.suggested_executed = False
        return self.server_status_datasource.store_info(message)

    @staticmethod
    def message_executed(command_executed: str, message):
        if message is None or message.action_suggested is None:
            return True
        action_suggested = message.action_suggested.suggested_command
        return message.command in command_executed or (action_suggested and action_suggested in command_executed)

class ClaiClient:

    def __init__(self, host: str=LOCALHOST, port: int=DEFAULT_PORT, connector: ClientConnector=None):
        self.connector = connector
        if not connector:
            self.connector = SocketClientConnector(host=host, port=port)
        self.port = port
        self.host = host

    def send(self, message: StateDTO) -> Action:
        try:
            return self.connector.send(message)
        except Exception as exception:
            logger.info(f'error: {exception}')
            return Action(origin_command=message.command, suggested_command=message.command)

class AgentDatasource:

    def __init__(self, config_storage: ConfigStorage=config):
        self.__selected_plugin: Dict[str, Optional[List[pkg.ModuleInfo]]] = {}
        self.__plugins: Dict[str, Agent] = {}
        self.num_workers = 4
        self.config_storage = config_storage
        self.current_orchestrator = None

    @staticmethod
    def get_path():
        return clai.server.plugins.__path__

    def preload_plugins(self):
        all_descriptors = self.all_plugins()
        installed_agents = list(filter(lambda value: value.installed, all_descriptors))
        for descriptor in installed_agents:
            self.start_agent(descriptor)
        logger.info('finish init')

    def start_agent(self, descriptor):
        agent_datasource_executor.execute(self.load_agent, descriptor.pkg_name)

    def load_agent(self, name: str):
        try:
            plugin = importlib.import_module(f'clai.server.plugins.{name}.{name}', package=name)
            importlib.invalidate_caches()
            plugin = importlib.reload(plugin)
            for _, class_member in inspect.getmembers(plugin, inspect.isclass):
                if issubclass(class_member, Agent) and class_member is not Agent:
                    member = class_member()
                    if not member:
                        member = ClaiIdentity()
                    self.__plugins[name] = member
                    member.init_agent()
                    member.ready = True
                    logger.info(f'{name} is ready')
        except Exception as ex:
            logger.info(f'load agent exception: {ex}')

    def get_instances(self, user_name: str, agent_to_select: str=None) -> List[Agent]:
        select_plugins_by_user = self.__get_selected_by_user(user_name)
        if select_plugins_by_user is None:
            self.init_plugin_config(user_name)
            select_plugins_by_user = self.__get_selected_by_user(user_name)
        if agent_to_select:
            agent_descriptor_selected = self.get_agent_descriptor(agent_to_select)
            if agent_descriptor_selected is None:
                return [ClaiIdentity()]
            select_plugins_by_user = list(filter(lambda value: value.name == agent_descriptor_selected.pkg_name, select_plugins_by_user))
        if agent_to_select:
            agent_descriptor_selected = self.get_agent_descriptor(agent_to_select)
            if agent_descriptor_selected is None:
                return [ClaiIdentity()]
            select_plugins_by_user = list(filter(lambda value: value.name == agent_descriptor_selected.pkg_name, select_plugins_by_user))
        agents = []
        for select_plugin_by_user in select_plugins_by_user:
            if select_plugin_by_user.name in self.__plugins:
                agent = self.__plugins[select_plugin_by_user.name]
                if agent.ready:
                    agents.append(agent)
        if not agents:
            agents.append(ClaiIdentity())
        return agents

    @staticmethod
    def load_descriptors(path, name) -> AgentDescriptor:
        file_path = os.path.join(path, 'manifest.properties')
        if os.path.exists(file_path):
            config_parser = configparser.ConfigParser()
            config_parser.read(file_path)
            default = z_default = False
            if config_parser.has_option('DEFAULT', 'default'):
                default = config_parser.getboolean('DEFAULT', 'default')
            if config_parser.has_option('DEFAULT', 'z_default'):
                z_default = config_parser.getboolean('DEFAULT', 'z_default')
            exclude = []
            if config_parser.has_option('DEFAULT', 'exclude'):
                exclude = config_parser.get('DEFAULT', 'exclude').lower().split()
            return AgentDescriptor(pkg_name=name, name=config_parser['DEFAULT']['name'], description=config_parser['DEFAULT']['description'], exclude=exclude, default=default, z_default=z_default)
        return AgentDescriptor(pkg_name=name, name=name)

    def get_report_enable(self) -> bool:
        plugins_config = self.config_storage.read_config(None)
        return plugins_config.report_enable

    def mark_report_enable(self, report_enable):
        plugins_config = self.config_storage.read_config(None)
        plugins_config.report_enable = report_enable
        self.config_storage.store_config(plugins_config, None)

    def mark_plugins_as_installed(self, name_plugin: str, user_name: Optional[str]):
        plugins_config = self.config_storage.read_config(user_name)
        if name_plugin not in plugins_config.installed:
            plugins_config.installed.append(name_plugin)
        self.config_storage.store_config(plugins_config, user_name)

    @staticmethod
    def filter_by_platform(agent_descriptors: List[AgentDescriptor]) -> List[AgentDescriptor]:
        os_name = os.uname().sysname.lower()
        return list(filter(lambda agent: os_name not in agent.exclude, agent_descriptors))

    def all_plugins(self) -> List[AgentDescriptor]:
        agent_descriptors = list((self.load_descriptors(os.path.join(importer.path, name), name) for importer, name, _ in pkg.iter_modules(self.get_path())))
        agent_descriptors = self.filter_by_platform(agent_descriptors)
        plugins_installed = self.config_storage.load_installed()
        for agent_descriptor in agent_descriptors:
            agent_descriptor.installed = agent_descriptor.name in plugins_installed
        logger.info(f'agents runned: {self.__plugins}')
        for agent_descriptor in agent_descriptors:
            if agent_descriptor.pkg_name in self.__plugins:
                logger.info(f'{agent_descriptor.pkg_name} is {self.__plugins[agent_descriptor.pkg_name].ready}')
                agent_descriptor.ready = self.__plugins[agent_descriptor.pkg_name].ready
            else:
                logger.info(f'{agent_descriptor.pkg_name} not iniciate.')
                agent_descriptor.ready = False
        return agent_descriptors

    def get_current_orchestrator(self) -> str:
        if not self.current_orchestrator:
            plugin_config = self.config_storage.read_config()
            self.current_orchestrator = plugin_config.default_orchestrator
            plugin_config.orchestrator = self.current_orchestrator
            self.config_storage.store_config(plugin_config)
        return self.current_orchestrator

    def select_orchestrator(self, orchestrator_name: str):
        plugin_config = self.config_storage.read_config()
        self.current_orchestrator = orchestrator_name
        plugin_config.orchestrator = self.current_orchestrator
        self.config_storage.store_config(config)

    def get_current_plugin_name(self, user_name: str) -> List[str]:
        selected_plugin = self.__get_selected_by_user(user_name)
        if selected_plugin is None:
            self.init_plugin_config(user_name)
            selected_plugin = self.__get_selected_by_user(user_name)
        if not selected_plugin:
            return []
        return list(map(lambda plugin: plugin.name, selected_plugin))

    def init_plugin_config(self, user_name: str) -> PluginConfig:
        plugin_config = self.config_storage.read_config(user_name)
        plugin_name = plugin_config.selected
        if not plugin_name:
            plugin_name = plugin_config.default
        for plugin_to_select in plugin_name:
            self.select_plugin(plugin_to_select, user_name)
        return plugin_config

    def get_agent_descriptor(self, plugin_to_select) -> Optional[AgentDescriptor]:
        all_plugins = self.all_plugins()
        for plugin in all_plugins:
            if plugin_to_select in (plugin.name, plugin.pkg_name):
                return plugin
        return None

    def select_plugin(self, plugin_to_select: str, user_name: str) -> Optional[pkg.ModuleInfo]:
        agent_descriptor_selected = self.get_agent_descriptor(plugin_to_select)
        if agent_descriptor_selected is None:
            return None
        for module in pkg.iter_modules(self.get_path()):
            if module.name == agent_descriptor_selected.pkg_name:
                self.__select_plugin_for_user(module, user_name)
                if agent_descriptor_selected.pkg_name not in self.__plugins:
                    self.start_agent(agent_descriptor_selected)
                return module
        return None

    def unselect_plugin(self, plugin_to_select: str, user_name: str) -> Optional[pkg.ModuleInfo]:
        agent_descriptor_selected = self.get_agent_descriptor(plugin_to_select)
        if agent_descriptor_selected is None:
            return None
        for module in pkg.iter_modules(self.get_path()):
            if module.name == agent_descriptor_selected.pkg_name:
                self.__unselect_plugin_for_user(module, user_name)
                return module
        return None

    def __select_plugin_for_user(self, plugin_to_select, user_name):
        if user_name in self.__selected_plugin:
            if plugin_to_select not in self.__selected_plugin[user_name]:
                self.__selected_plugin[user_name].append(plugin_to_select)
        else:
            self.__selected_plugin[user_name] = [plugin_to_select]

    def __unselect_plugin_for_user(self, plugin_to_select, user_name):
        if user_name in self.__selected_plugin and plugin_to_select in self.__selected_plugin[user_name]:
            self.__selected_plugin[user_name].remove(plugin_to_select)

    def __get_selected_by_user(self, user_name) -> Optional[List[pkg.ModuleInfo]]:
        if user_name in self.__selected_plugin:
            return self.__selected_plugin[user_name]
        return None

    def reload(self):
        self.__plugins.clear()
        self.preload_plugins()

class ClaiInstallCommandRunner(CommandRunner, PostCommandRunner):
    INSTALL_PLUGIN_DIRECTIVE = 'clai install'

    def __init__(self, agent_datasource: AgentDatasource):
        self.agent_datasource = agent_datasource

    @staticmethod
    def __move_plugin__(dir_to_install: str) -> Action:
        if dir_to_install.startswith('http') or dir_to_install.startswith('https'):
            cmd = f'cd $CLAI_PATH/clai/server/plugins && curl -O {dir_to_install}'
            return Action(suggested_command=cmd, execute=True)
        if not os.path.exists(dir_to_install) or not os.path.isdir(dir_to_install):
            return create_error_install(dir_to_install)
        plugin_name = dir_to_install.split('/')[-1]
        logger.info(f'installing plugin name {plugin_name}')
        cmd = f'cp -R {dir_to_install} $CLAI_PATH/clai/server/plugins'
        return Action(suggested_command=cmd, execute=True)

    def execute(self, state: State) -> Action:
        dir_to_install = state.command.replace(f'{self.INSTALL_PLUGIN_DIRECTIVE}', '').strip()
        logger.info(f'trying to install ${dir_to_install}')
        return self.__move_plugin__(dir_to_install)

    def execute_post(self, state: State) -> Action:
        if state.result_code == '0' and state.action_suggested.suggested_command != ':':
            return create_message_list(self.agent_datasource.get_current_plugin_name(state.user_name), self.agent_datasource.all_plugins())
        return Action()

class Provider:
    base_uri: str = ''
    excludes: list = []

    def __init__(self, name: str, description: str, section: dict):
        self.name = name
        self.description = description
        api_value: str = section.get('api')
        self.base_uri = api_value if api_value.endswith('/') else api_value + '/'
        if 'exclude' in section.keys():
            self.excludes = [excludeTarget.lower() for excludeTarget in section.get('exclude').split()]
        else:
            self.excludes = []
        if 'variants' in section.keys():
            self.variants = [variant.upper() for variant in section.get('variants').split()]
        else:
            self.variants = []

    def __str__(self) -> str:
        return self.description

    def __log_info__(self, message):
        logger.info(f'{self.name}: {message}')

    def __log_warning__(self, message):
        logger.warning(f'{self.name}: {message}')

    def __log_debug__(self, message):
        logger.debug(f'{self.name}: {message}')

    def __log_json__(self, data):
        output: List[str] = ['']
        getters: List[str] = []
        is_json_data: bool = False
        if isinstance(data, Request):
            output.append(f'{data.method} --> {data.url}')
            getters = ['files', 'data', 'json', 'params', 'auth', 'cookies', 'hooks']
        elif isinstance(data, Response):
            output.append(f'RESPONSE[{data.status_code}] <-- {data.url}')
            getters = ['apparent_encoding', 'cookies', 'elapsed', 'encoding', 'ok', 'status_code', 'reason', 'content']
        if data.headers.keys():
            output.append('.--[Headers]'.ljust(80, '-'))
            for key in data.headers.keys():
                if key == 'Content-Type':
                    if '/json' in data.headers[key]:
                        is_json_data = True
                line_header = f'|    {key}: '
                print_data: str = str(data.headers[key])
                if len(print_data) > 64:
                    print_data = f'{print_data[:50]} ... {print_data[-10:]}'
                output.append(f'{line_header}{print_data}')
            output.append('`'.ljust(80, '-'))
        for method in getters:
            result = getattr(data, method)
            if method == 'content' and is_json_data:
                outstr = json.dumps(json.loads(result), indent=2)
                output.append(f'{method}: ' + outstr.replace('\n', '\n\t'))
            else:
                print_data: str = str(result)
                line_header = f'{method}: '
                if len(print_data) > 64:
                    print_data = f'{print_data[:50]} ... {print_data[-10:]}'
                output.append(f'{line_header}{print_data}')
        self.__log_info__('\n\t'.join(output))

    def __set_default_values__(self, args: dict, **kwargs) -> dict:
        for key in kwargs.keys():
            if key not in args:
                args[key] = kwargs[key]
        return args

    def __send_request__(self, method: str, uri: str, **kwargs) -> Response:
        kwargs = self.__set_default_values__(kwargs, headers={'Content-Type': 'application/json', 'Accept': '*/*'}, data=None, params=None)
        request: Request = Request(method, uri.rstrip('/'), headers=kwargs['headers'], data=kwargs['data'], params=kwargs['params'])
        self.__log_json__(request)
        session: Session = Session()
        response: Response = session.send(request=request.prepare())
        self.__log_json__(response)
        return response

    def __send_get_request__(self, uri: str, **kwargs):
        return self.__send_request__(method='GET', uri=uri, **kwargs)

    def __send_post_request__(self, uri: str, **kwargs):
        return self.__send_request__(method='POST', uri=uri, **kwargs)

    def get_excludes(self) -> list:
        """Returns a list of operating systems that the search provider cannot
          be run from
        """
        return self.excludes

    def has_variants(self) -> bool:
        """Returns `True` if this search provider has more than one search
          variant
        """
        return len(self.variants) > 0

    def get_variants(self) -> list:
        """Returns a list of search variants supported by this search provider
        """
        return self.variants

    def can_run_on_this_os(self) -> bool:
        """Returns True if this search provider can be used on the client OS
        """
        if not self.excludes:
            return True
        os_name: str = os.uname().sysname.lower()
        return os_name not in self.excludes

    @abc.abstractclassmethod
    def extract_search_result(self, data: List[Dict]) -> str:
        """Given the result of an API search, extract the search result from it
        """
        pass

    @abc.abstractclassmethod
    def get_printable_output(self, data: List[Dict]) -> str:
        """Extract the result string from the data returned by an API search
        """
        pass

    @abc.abstractclassmethod
    def call(self, query: str, limit: int=1, **kwargs):
        """Perform a query on the API
        """
        pass

class ActionRemoteStorage:
    _instance = None
    manager = None
    queue = None
    consumer_task = None
    pool = None
    report_enable = None
    anonymizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ActionRemoteStorage, cls).__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        self.manager = mp.Manager()
        self.queue = self.manager.Queue()
        self.consumer_task = None
        self.pool = mp.Pool(1)
        self.report_enable = False
        self.anonymizer = Anonymizer()

    @staticmethod
    def consumer(queue):
        while True:
            data = queue.get()
            if data is None:
                break
            logger.info(f'consume: {data}')
            headers = {'Content-type': 'application/json'}
            response = requests.post(url=URL_SERVER, data=data, headers=headers)
            logger.info(f'[Sent] {response.status_code} message {response.text}')
            queue.task_done()
        logger.info('----STOP CONSUMING-----')

    def start(self, agent_datasource: AgentDatasource):
        logger.info(f'-Start sender-')
        self.report_enable = agent_datasource.get_report_enable()
        if self.report_enable:
            self.consumer_task = self.pool.map_async(self.consumer, (self.queue,))

    def wait(self):
        if self.report_enable:
            self.queue.put(None)
            self.consumer_task.wait(timeout=3)

    def store(self, message: TerminalReplayMemory):
        if not self.report_enable:
            return
        try:
            command = message.command
            message_as_json = TerminalReplayMemoryApi(command=self.__parse_state__(command, self.anonymizer), agent_names=message.agent_names, candidate_actions=self.__parse_actions__(message.candidate_actions), force_response=str(message.force_response), suggested_command=self.__parse_actions__(message.suggested_command))
            logger.info(f'store -> {message.command.command_id}')
            message_to_send = RecordToSendApi(bashbot_info=message_as_json)
            self.queue.put(message_to_send.json())
        except Exception as err:
            logger.info(f'error sending: {err}')

    @staticmethod
    def __parse_state__(command: State, anonymizer: Anonymizer):
        command_api = StateApi()
        command_api.command_id = command.command_id
        command_api.user_name = anonymizer.anonymize(command.user_name)
        command_api.command = command.command
        command_api.root = command.root
        command_api.processes = command.processes
        command_api.file_changes = command.file_changes
        command_api.network = command.network
        command_api.result_code = command.result_code
        command_api.stderr = command.stderr
        return command_api

    @staticmethod
    def __parse_actions__(candidate_actions: Optional[List[Union[Action, List[Action]]]]):
        if not candidate_actions:
            return []
        if isinstance(candidate_actions, Action):
            return [candidate_actions]
        return candidate_actions

def __send_event__(event: StatEvent):
    try:
        amplitude_logger = amplitude.AmplitudeLogger(api_key='cc826565c91ab899168235a2845db189')
        event_args = {'device_id': event.user, 'event_type': event.event_type, 'event_properties': event.data}
        event = amplitude_logger.create_event(**event_args)
        amplitude_logger.log_event(event)
    except Exception as ex:
        logger.info(f'error tracking event {ex}')

class StatsTracker:
    _instance = None
    manager = None
    queue = None
    pool = None
    consumer_stats = None
    anonymizer = None
    report_enable = None

    def __new__(cls, sync=False, anonymizer: Anonymizer=Anonymizer()):
        if cls._instance is None:
            cls._instance = super(StatsTracker, cls).__new__(cls)
            cls._instance.init(sync, anonymizer)
        return cls._instance

    def init(self, sync, anonymizer):
        if not sync:
            self.manager = mp.Manager()
            self.queue = self.manager.Queue()
            self.pool = mp.Pool(1)
            self.consumer_stats = None
        self.anonymizer = anonymizer
        self.report_enable = False

    @staticmethod
    def consumer(queue):
        while True:
            event = queue.get()
            if event is None:
                break
            logger.info(f'send_event: {event}')
            __send_event__(event)
            queue.task_done()
        logger.info('----STOP CONSUMING-----')

    def start(self, agent_datasource: AgentDatasource):
        logger.info(f'-Start tracker-')
        self.report_enable = agent_datasource.get_report_enable()
        if self.report_enable:
            self.consumer_stats = self.pool.map_async(self.consumer, (self.queue,))

    def wait(self):
        if self.report_enable:
            self.__store__(None)
            self.consumer_stats.wait(timeout=3)

    def log_activate_skills(self, user: str, skill_name: str):
        event = StatEvent(event_type='activate', user=self.anonymizer.anonymize(user), data={'skill': f'{skill_name}'})
        self.__store__(event)

    def log_deactivate_skills(self, user: str, skill_name: str):
        event = StatEvent(event_type='deactivate', user=self.anonymizer.anonymize(user), data={'skill': f'{skill_name}'})
        self.__store__(event)

    def log_install(self, user: str):
        if not self.report_enable:
            return
        event = StatEvent(event_type='install', user=self.anonymizer.anonymize(user), data={})
        __send_event__(event)

    def log_uninstall(self, user: str):
        if not self.report_enable:
            return
        event = StatEvent(event_type='uninstall', user=self.anonymizer.anonymize(user), data={})
        __send_event__(event)

    def __store__(self, event: StatEvent):
        if not self.report_enable:
            return
        try:
            logger.info(f'record stats -> {event}')
            self.queue.put(event)
        except Exception as err:
            logger.info(f'error sending: {err}')

