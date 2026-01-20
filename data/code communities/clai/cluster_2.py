# Cluster 2

def stat_uninstall(bin_path):
    agent_datasource = AgentDatasource(config_storage=ConfigStorage(alternate_path=f'{bin_path}/configPlugins.json'))
    report_enable = agent_datasource.get_report_enable()
    stats_tracker = StatsTracker(sync=True, anonymizer=Anonymizer(alternate_path=f'{bin_path}/anonymize.json'))
    stats_tracker.report_enable = report_enable
    login = getpass.getuser()
    stats_tracker.log_uninstall(login)
    print('record uninstall')

def ask_to_user(text):
    print(Colorize().info().append(f'{text} ').append('(y/n)').to_console())
    while True:
        command_input = input()
        if command_input in ('y', 'yes'):
            return True
        if command_input in ('n', 'no'):
            return False
        print(Colorize().info().append('choose yes[y] or no[n]').to_console())

def save_report_info(unassisted, agent_datasource, bin_path, demo_mode):
    enable_report = True
    if demo_mode:
        enable_report = False
    elif not unassisted:
        enable_report = ask_to_user('Would you like anonymously send debugging and usage informationto the CLAI team in order to help improve it?')
    agent_datasource.mark_report_enable(enable_report)
    stats_tracker = StatsTracker(sync=True, anonymizer=Anonymizer(alternate_path=f'{bin_path}/anonymize.json'))
    stats_tracker.report_enable = enable_report
    stats_tracker.log_install(getpass.getuser())

class AgentRunner:

    def __init__(self, agent_datasource: AgentDatasource, orchestrator_provider: OrchestratorProvider):
        self.agent_datasource = agent_datasource
        self.orchestrator_provider = orchestrator_provider
        self.remote_storage = ActionRemoteStorage()
        self.orchestrator_storage = OrchestratorStorage(orchestrator_provider, self.remote_storage)
        self._pre_exec_id = 'pre'
        self._post_exec_id = 'post'

    def store_pre_orchestrator_memory(self, command: State, agent_list: List[Agent], candidate_actions: Optional[List[Union[Action, List[Action]]]], force_response: bool, suggested_command: Optional[Action]):
        agent_names = [agent.agent_name for agent in agent_list]
        state = TerminalReplayMemory(command, agent_names, candidate_actions, force_response, suggested_command)
        self.orchestrator_storage.store_pre(state)

    def store_post_orchestrator_memory(self, command: State, agent_list: List[Agent], candidate_actions: Optional[List[Union[Action, List[Action]]]], force_response: bool, suggested_command: Optional[Action]):
        agent_names = [agent.agent_name for agent in agent_list]
        state = TerminalReplayMemory(command, agent_names, candidate_actions, force_response, suggested_command)
        self.orchestrator_storage.store_post(state)

    def select_best_candidate(self, command: State, agent_list: List[Agent], candidate_actions: Optional[List[Union[Action, List[Action]]]], force_response: bool, pre_post_state: str) -> Optional[Union[Action, List[Action]]]:
        agent_names = [agent.agent_name for agent in agent_list]
        orchestrator = self.orchestrator_provider.get_current_orchestrator()
        suggested_command = orchestrator.choose_action(command=command, agent_names=agent_names, candidate_actions=candidate_actions, force_response=force_response, pre_post_state=pre_post_state)
        if not suggested_command:
            suggested_command = Action()
        return suggested_command

    def process(self, command: State, ignore_threshold: bool, force_agent: str=None) -> Optional[Union[Action, List[Action]]]:
        if force_agent:
            plugin_instances = self.agent_datasource.get_instances(command.user_name, force_agent)
            ignore_threshold = True
        else:
            plugin_instances = self.agent_datasource.get_instances(command.user_name)
        candidate_actions = agent_executor.execute_agents(command, plugin_instances)
        suggested_command = self.select_best_candidate(command, plugin_instances, candidate_actions, ignore_threshold, self._pre_exec_id)
        if not suggested_command:
            suggested_command = Action()
        self.store_pre_orchestrator_memory(command, plugin_instances, candidate_actions, ignore_threshold, suggested_command)
        if isinstance(suggested_command, Action):
            if not suggested_command.suggested_command:
                suggested_command.suggested_command = command.command
        else:
            for action in suggested_command:
                if not action.suggested_command:
                    action.suggested_command = command.command
        return suggested_command

    def process_post(self, command: State, ignore_threshold: bool) -> Optional[Action]:
        plugin_instances = self.agent_datasource.get_instances(command.user_name)
        candidate_actions = []
        for plugin_instance in plugin_instances:
            action_post_executed = plugin_instance.post_execute(command)
            action_post_executed.agent_owner = plugin_instance.agent_name
            if action_post_executed:
                candidate_actions.append(action_post_executed)
        suggested_command = self.select_best_candidate(command, plugin_instances, candidate_actions, ignore_threshold, self._post_exec_id)
        self.store_post_orchestrator_memory(command, plugin_instances, candidate_actions, ignore_threshold, suggested_command)
        if not suggested_command:
            suggested_command = Action()
        if not suggested_command.suggested_command:
            suggested_command.suggested_command = command.command
        return suggested_command

class ClaiServer:

    def __init__(self, server_status_datasource: ServerStatusDatasource=current_status_datasource, connector: ServerConnector=SocketServerConnector(current_status_datasource), agent_datasource=AgentDatasource()):
        self.connector = connector
        self.agent_datasource = agent_datasource
        self.server_status_datasource = server_status_datasource
        self.remote_storage = ActionRemoteStorage()
        self.message_handler = MessageHandler(server_status_datasource, agent_datasource=agent_datasource)
        self.stats_tracker = StatsTracker()

    def init_server(self):
        self.message_handler.init_server()
        self.server_status_datasource.running = True
        self.remote_storage.start(self.agent_datasource)
        self.stats_tracker.start(self.agent_datasource)

    @staticmethod
    def serialize_message(data) -> State:
        StateDTO.update_forward_refs()
        dto = StateDTO(**json.loads(data))
        return State(command_id=dto.command_id, user_name=dto.user_name, command=dto.command, root=dto.root, processes=dto.processes, file_changes=dto.file_changes, network=dto.network, result_code=dto.result_code, stderr=dto.stderr)

    def create_socket(self, host, port):
        self.connector.create_socket(host, port)

    def listen_client_sockets(self):
        self.connector.loop(self.process_message)
        self.remote_storage.wait()
        self.stats_tracker.wait()

    def process_message(self, received_data: bytes) -> Action:
        message = self.serialize_message(received_data)
        return self.message_handler.process_message(message)

class ClaiUnselectCommandRunner(CommandRunner):
    UNSELECT_DIRECTIVE = 'clai deactivate'

    def __init__(self, config_storage: ConfigStorage, agent_datasource: AgentDatasource):
        self.config_storage = config_storage
        self.agent_datasource = agent_datasource
        self.stats_tracker = StatsTracker()

    def execute(self, state: State) -> Action:
        plugin_to_select = state.command.replace(f'{self.UNSELECT_DIRECTIVE}', '').strip()
        plugin_to_select = extract_quoted_agent_name(plugin_to_select)
        selected = self.agent_datasource.unselect_plugin(plugin_to_select, state.user_name)
        if selected:
            self.stats_tracker.log_deactivate_skills(state.user_name, plugin_to_select)
            plugins_config = self.config_storage.read_config(state.user_name)
            if plugins_config.selected is not None and selected.name in plugins_config.selected:
                plugins_config.selected.remove(selected.name)
            self.config_storage.store_config(plugins_config, state.user_name)
        action_to_return = create_message_list(self.agent_datasource.get_current_plugin_name(state.user_name), self.agent_datasource.all_plugins())
        action_to_return.origin_command = state.command
        return action_to_return

class ClaiSelectCommandRunner(CommandRunner, PostCommandRunner):
    SELECT_DIRECTIVE = 'clai activate'

    def __init__(self, config_storage: ConfigStorage, agent_datasource: AgentDatasource):
        self.agent_datasource = agent_datasource
        self.config_storage = config_storage
        self.stats_tracker = StatsTracker()

    def execute(self, state: State) -> Action:
        plugin_to_select = state.command.replace(f'{self.SELECT_DIRECTIVE}', '').strip()
        plugin_to_select = extract_quoted_agent_name(plugin_to_select)
        agent_descriptor = self.agent_datasource.get_agent_descriptor(plugin_to_select)
        plugins_config = self.config_storage.read_config(None)
        if not agent_descriptor:
            return create_error_select(plugin_to_select)
        if agent_descriptor and (not agent_descriptor.installed):
            logger.info(f'installing dependencies of plugin {agent_descriptor.name}')
            command = f'$CLAI_PATH/fileExist.sh {agent_descriptor.pkg_name} $CLAI_PATH{(' --user' if plugins_config.user_install else '')}'
            action_selected_to_return = Action(suggested_command=command, execute=True)
        else:
            self.select_plugin(plugin_to_select, state)
            action_selected_to_return = Action(suggested_command=':', execute=True)
        action_selected_to_return.origin_command = state.command
        return action_selected_to_return

    def select_plugin(self, plugin_to_select, state):
        selected_plugin = self.agent_datasource.select_plugin(plugin_to_select, state.user_name)
        if selected_plugin:
            self.stats_tracker.log_activate_skills(state.user_name, plugin_to_select)
            plugins_config = self.config_storage.read_config(state.user_name)
            if plugins_config.selected is None:
                plugins_config.selected = [selected_plugin.name]
            elif selected_plugin.name not in plugins_config.selected:
                plugins_config.selected.append(selected_plugin.name)
            self.config_storage.store_config(plugins_config, state.user_name)
        return create_message_list(self.agent_datasource.get_current_plugin_name(state.user_name), self.agent_datasource.all_plugins())

    def execute_post(self, state: State) -> Action:
        plugin_to_select = state.command.replace(f'{self.SELECT_DIRECTIVE}', '').strip()
        plugin_to_select = extract_quoted_agent_name(plugin_to_select)
        agent_descriptor = self.agent_datasource.get_agent_descriptor(plugin_to_select)
        if not agent_descriptor:
            return Action()
        if state.result_code == '0':
            self.agent_datasource.mark_plugins_as_installed(plugin_to_select, state.user_name)
            return self.select_plugin(plugin_to_select, state)
        return create_message_list(self.agent_datasource.get_current_plugin_name(state.user_name), self.agent_datasource.all_plugins())

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

