# Cluster 15

class MockExecutor(AgentExecutor):

    def execute_agents(self, command: State, agents: List[Agent]) -> List[Action]:
        return list(map(lambda agent: self.execute(command, agent), agents))

    @staticmethod
    def execute(command: State, agent: Agent) -> Action:
        action = agent.execute(command)
        if not action:
            action = Action()
        return action

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

class Agent(ABC):

    def __init__(self):
        self.agent_name = self.__class__.__name__
        self.ready = False
        self._save_basedir = os.path.join(BASEDIR, 'saved_agents')
        self._save_dirpath = os.path.join(self._save_basedir, self.agent_name)

    def execute(self, state: State) -> Union[Action, List[Action]]:
        try:
            action_to_return = self.get_next_action(state)
        except:
            action_to_return = Action()
        if isinstance(action_to_return, list):
            for action in action_to_return:
                action.agent_owner = self.agent_name
        else:
            action_to_return.agent_owner = self.agent_name
        return action_to_return

    def post_execute(self, state: State) -> Action:
        """Provide a post execution"""
        return Action(origin_command=state.command)

    @abstractmethod
    def get_next_action(self, state: State) -> Union[Action, List[Action]]:
        """Provide next action to execute in the bash console"""

    def init_agent(self):
        """Add here all heavy task for initialize the """

    def get_agent_state(self) -> dict:
        """Returns the agent state to be saved for future loading"""
        return {}

    def __prepare_state_folder__(self):
        os.makedirs(self._save_dirpath, exist_ok=True)
        for file in os.listdir(self._save_dirpath):
            path = os.path.join(self._save_dirpath, file)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)

    def save_agent(self) -> bool:
        """Saves agent state into persisting memory"""
        try:
            state = self.get_agent_state()
            if not state:
                return True
            self.__prepare_state_folder__()
            for var_name, var_val in state.items():
                with open('{}/{}.p'.format(self._save_dirpath, var_name), 'wb') as file:
                    pickle.dump(var_val, file)
        except Exception:
            return False
        else:
            return True

    def load_saved_state(self) -> dict:
        agent_state = {}
        filenames = []
        if os.path.exists(self._save_dirpath):
            filenames = [file for file in os.listdir(self._save_dirpath) if os.path.isfile(os.path.join(self._save_dirpath, file))]
        for filename in filenames:
            try:
                key = os.path.splitext(filename)[0]
                with open(os.path.join(self._save_dirpath, filename), 'rb') as file:
                    agent_state[key] = pickle.load(file)
            except:
                pass
        return agent_state

    def extract_name_without_extension(self, filename):
        return os.path.splitext(filename)[0]

    def __del__(self):
        self.save_agent()

class ClaiIdentity(Agent):

    def get_next_action(self, state: State) -> Action:
        return Action(suggested_command=state.command)

class AgentCommandRunner(CommandRunner, PostCommandRunner):

    def __init__(self, agent_runner: AgentRunner, server_status_datasource: ServerStatusDatasource, ignore_threshold: bool=False):
        self.agent_runner = agent_runner
        self.server_status_datasource = server_status_datasource
        self.ignore_threshold = ignore_threshold
        self.force_agent = None

    def execute(self, state: State) -> Union[Action, List[Action]]:
        action = self.agent_runner.process(state, self.ignore_threshold, self.force_agent)
        if not action:
            action = Action()
        return action

    def execute_post(self, state: State) -> Action:
        action = self.agent_runner.process_post(state, self.ignore_threshold)
        if not action:
            action = Action()
        action.origin_command = state.command
        return action

class ClaiLastInfoCommandRunner(CommandRunner, PostCommandRunner):
    LAST_DIRECTIVE_DIRECTIVE = 'clai last-info'

    def __init__(self, server_status_datasource: ServerStatusDatasource):
        self.server_status_datasource = server_status_datasource

    def execute(self, state: State) -> Action:
        return Action(suggested_command=':', execute=True, origin_command=state.command)

    def execute_post(self, state: State) -> Action:
        offset_last = state.command.replace(f'{self.LAST_DIRECTIVE_DIRECTIVE}', '').strip()
        if not offset_last:
            offset_last = '0'
        if not offset_last.isdigit():
            offset_last = '0'
        offset_last_as_int = int(offset_last)
        last_message = self.server_status_datasource.get_last_message(state.user_name, offset=offset_last_as_int)
        if not last_message:
            return Action()
        info_to_show = InfoDebug(command_id=last_message.command_id, user_name=last_message.user_name, command=last_message.command, root=last_message.root, processes=last_message.processes, file_changes=last_message.file_changes, network=last_message.network, result_code=last_message.result_code, stderr=last_message.stderr, already_processed=last_message.already_processed, action_suggested=last_message.action_suggested)
        return Action(description=str(info_to_show.json()))

class ClaiPowerCommandRunner(CommandRunner):

    def __init__(self, server_status_datasource: ServerStatusDatasource):
        self.server_status_datasource = server_status_datasource

    def execute(self, state: State) -> Action:
        if self.server_status_datasource.is_power():
            text = 'You have the auto mode already enable, use clai manual to deactivate it'
        else:
            self.server_status_datasource.set_power(True)
            text = 'You have enabled the auto mode'
        return Action(origin_command=state.command, suggested_command=NOOP_COMMAND, description=text, execute=True)

class ClaiPowerDisableCommandRunner(CommandRunner):

    def __init__(self, server_status_datasource: ServerStatusDatasource):
        self.server_status_datasource = server_status_datasource

    def execute(self, state: State) -> Action:
        if not self.server_status_datasource.is_power():
            text = 'You have manual mode already enable, use clai auto to activate it'
        else:
            self.server_status_datasource.set_power(False)
            text = 'You have enable the manual mode'
        return Action(suggested_command=NOOP_COMMAND, origin_command=state.command, description=text, execute=True)

class Linuss(Agent):

    def __init__(self):
        super(Linuss, self).__init__()
        self._config_path = os.path.join(Path(__file__).parent.absolute(), 'equivalencies.json')
        self.equivalencies = self.__read_equivalencies()

    def __read_equivalencies(self):
        logger.info('####### read equivalencies inside linuss ########')
        try:
            with open(self._config_path, 'r') as json_file:
                equivalencies = json.load(json_file)
        except Exception as e:
            logger.debug(f'Linuss Error: {e}')
            equivalencies = json.load({})
        return equivalencies

    def __build_suggestion(self, command, options, cmd_key) -> any:
        params: list[Tuple(str)] = []
        suggestion: str = None
        explanations: list[str] = []
        tokens: list[str] = command.split()
        idx = 0
        max_idx = len(tokens) - 1
        while idx <= max_idx:
            token: str = tokens[idx]
            if re.match('-([\\w_-]+)', token):
                if idx < max_idx and (not re.match('-([\\w_-]+)', tokens[idx + 1])):
                    next_token = tokens[idx + 1]
                    params.append((token[1:], next_token))
                    idx = idx + 1
                else:
                    params.append((token[1:], None))
            elif token != cmd_key:
                params.append((None, token))
            idx = idx + 1
        if '' in options:
            suggestion = self.equivalencies[cmd_key]['']['equivalent']
        else:
            suggestion = cmd_key
        for opt, arg in params:
            if opt is not None:
                if opt[0] == '-' or len(opt) == 1:
                    equivalency = self.__get_equavalency(opt, options, cmd_key)
                    if equivalency['equivalent']:
                        suggestion = f'{suggestion} {equivalency['equivalent']}'
                        if 'explanation' in equivalency:
                            explanations.append(equivalency['explanation'])
                    else:
                        explanations.append(f'The -{opt} flag is not available on USS')
                else:
                    for char in opt:
                        equivalency = self.__get_equavalency(char, options, cmd_key)
                        if equivalency['equivalent']:
                            suggestion = f'{suggestion} {equivalency['equivalent']}'
                            if 'explanation' in equivalency:
                                explanations.append(equivalency['explanation'])
                        else:
                            explanations.append(f'The -{char} flag is not available on USS')
            if arg is not None:
                suggestion = f'{suggestion} {arg}'
        if suggestion is not None:
            return Action(suggested_command=suggestion, confidence=1, description='\n'.join(explanations))
        else:
            return Action(suggested_command=NOOP_COMMAND, description=None)

    def __get_equavalency(self, target, options, cmd_key) -> dict:
        target = f'-{target}'
        for option in options:
            if option == '':
                pass
            elif re.search('{}'.format(option), target):
                return self.equivalencies[cmd_key][option]
        return {'equivalent': target}

    def get_next_action(self, state: State) -> Action:
        command = state.command
        for cmd in self.equivalencies:
            if command.startswith(cmd):
                return self.__build_suggestion(command, self.equivalencies[cmd], cmd)
        return Action(suggested_command=NOOP_COMMAND)

class MsgCodeAgent(Agent):

    def __init__(self):
        super(MsgCodeAgent, self).__init__()
        inifile_path = os.path.join(str(Path(__file__).parent.absolute()), 'config.ini')
        self.store = Datastore(inifile_path)

    def get_next_action(self, state: State) -> Action:
        return Action(suggested_command=state.command)

    def post_execute(self, state: State) -> Action:
        logger.info('==================== In zMsgCode Bot:post_execute ============================')
        logger.info('State:\n\tCommand: {}\n\tError Code: {}\n\tStderr: {}'.format(state.command, state.result_code, state.stderr))
        logger.info('============================================================================')
        if state.result_code == '0':
            return Action(suggested_command=state.command)
        stderr = state.stderr.strip()
        matches = re.compile(REGEX_ZMSG).match(stderr)
        if matches is None:
            logger.info(f"No Z message ID found in '{stderr}'")
            return Action(suggested_command=state.command)
        logger.info(f"Analyzing error message '{matches[0]}'")
        msgid: str = matches[2]
        helpWasFound = False
        bpx_matches: List[str] = self.__search(matches[0], REGEX_BPX)
        if bpx_matches is not None:
            reason_code: str = bpx_matches[1]
            logger.info(f'==> Reason Code: {reason_code}')
            result: CompletedProcess = subprocess.run(['bpxmtext', reason_code], stdout=subprocess.PIPE)
            if result.returncode == 0:
                messageText = result.stdout.decode('UTF8')
                if self.__search(messageText, REGEX_BPX_BADANSWER) is None:
                    suggested_command = state.command
                    description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f'I asked bpxmtext about that message:\n').info().append(messageText).warning().to_console()
                    helpWasFound = True
        if not helpWasFound:
            kc_api: Provider = self.store.get_apis()['ibm_kc']
            if kc_api is not None and kc_api.can_run_on_this_os():
                data = self.store.search(msgid, service='ibm_kc', size=1)
                if data:
                    logger.info(f'==> Success!!! Found information for msgid {msgid}')
                    suggested_command = state.command
                    description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f'I looked up {msgid} in the IBM KnowledgeCenter for you:\n').info().append(kc_api.get_printable_output(data)).warning().to_console()
                    helpWasFound = True
        if not helpWasFound:
            logger.info('Failure: Unable to be helpful')
            logger.info('============================================================================')
            suggested_command = NOOP_COMMAND
            description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f"I couldn't find any help for message code '{msgid}'\n").info().to_console()
        return Action(suggested_command=suggested_command, description=description, confidence=1.0)

    def __search(self, target: str, regex_list: List[str]) -> List[str]:
        """Check all possible regexes in a list, return the first match encountered"""
        for regex in regex_list:
            this_match = re.compile(regex).match(target)
            if this_match is not None:
                return this_match
        return None

class GITBOT(Agent):

    def __init__(self):
        super(GITBOT, self).__init__()
        self.service = Service()
    ' pre execution processing '

    def get_next_action(self, state: State) -> Action:
        command = state.command
        return self.service(command)
    ' pre execution processing '

    def post_execute(self, state: State) -> Action:
        if state.result_code == '0':
            self.service.parse_command(state.command, stdout='')
        return Action(suggested_command=NOOP_COMMAND)

    def save_agent(self) -> bool:
        os.system('lsof -t -i tcp:{} | xargs kill'.format(_rasa_port_number))
        super().save_agent()

class GPT3(Agent):

    def __init__(self):
        super(GPT3, self).__init__()
        self._gpt3_api = self.__init_gpt3_api__()

    def __init_gpt3_api__(self):
        current_directory = str(Path(__file__).parent.absolute())
        path_to_gpt3_key = os.path.join(current_directory, 'openai_api.key')
        path_to_gpt3_prompts = os.path.join(current_directory, 'prompt.json')
        gpt3_key = open(path_to_gpt3_key, 'r').read()
        gpt3_prompts = json.load(open(path_to_gpt3_prompts, 'r'))
        gpt3_api = GPT(temperature=0)
        gpt3_api.set_api_key(gpt3_key)
        for prompt in gpt3_prompts:
            ip, op = (prompt['input'], prompt['output'])
            example = Example(ip, op)
            gpt3_api.add_example(example)
        return gpt3_api

    def get_next_action(self, state: State) -> Action:
        command = state.command
        command = command[:1000]
        try:
            response = self._gpt3_api.get_top_reply(command, strip_output_suffix=True)
            response = response.strip()
            return Action(suggested_command=response, execute=False, description='Currently the GPT-3 skill does not provide an explanation. Got an idea? Contribute to CLAI!', confidence=0.0)
        except Exception as ex:
            return [{'text': 'Method failed with status ' + str(ex)}, 0.0]

class IBMCloud(Agent):

    def __init__(self):
        super(IBMCloud, self).__init__()
        self.exe = KubeExe()
        self.intents = ['deploy to kube', 'build yaml', 'run Dockerfile']

    def get_next_action(self, state: State) -> Action:
        if state.command in self.intents:
            self.exe.set_goal(state.command)
            plan = self.exe.get_plan()
            if plan:
                logger.info('####### log plan inside ibmcloud ########')
                logger.info(plan)
                action_list = []
                for action in plan:
                    action_object = self.exe.execute_action(action)
                    if action_object:
                        action_list.append(action_object)
                return action_list
            else:
                return Action(suggested_command=NOOP_COMMAND, execute=True, description=Colorize().info().append('Sorry could not find a plan to help! :-(').to_console(), confidence=1.0)
        else:
            return Action(suggested_command=NOOP_COMMAND)

    def post_execute(self, state: State) -> Action:
        if state.result_code == '0':
            self.exe.parse_command(state.command, stdout='')
        return Action(suggested_command=NOOP_COMMAND)

class KubeExe(KubeTracker):

    def __init__(self):
        super().__init__()
        self.refresh_observations()

    def get_plan(self) -> list:
        problem = copy.deepcopy(self.template).replace('<GOAL>', self.goal)
        plan = get_plan_from_pr2plan(domain=self.domain, problem=problem, obs=self.get_observations())
        return plan

    def execute_action(self, action: str) -> str:
        try:
            command = getattr(self, action.replace('-', '_'))()
            if command:
                return command
        except Exception as e:
            print(e)

    def set_state(self):
        if not self.host_port:
            self.host_port = '8085'
        if not self.name:
            self.name = 'app'
        if not self.tag:
            self.tag = 'v1'
        return None

    def docker_build(self):
        command = 'docker build -t {}:{} .'.format(self.name, self.tag)
        return Action(suggested_command=command, confidence=1.0, execute=False)

    def dockerfile_read(self):
        with open('Dockerfile', 'r') as f:
            dockerfile_contents = f.read()
        self.local_port = re.findall('EXPOSE\\s[0-9]+', dockerfile_contents)[0].strip().split(' ')[-1].strip()
        return None

    def docker_run(self):
        command = 'docker run -i -p {}:{} -d {}:{}'.format(self.host_port, self.local_port, self.name, self.tag)
        return Action(suggested_command=command, confidence=1.0)

    def ibmcloud_login(self):
        command = 'ibmcloud login'
        return Action(suggested_command=command, confidence=1.0)

    def ibmcloud_cr_login(self):
        command = 'ibmcloud cr login'
        return Action(suggested_command=command, confidence=1.0, execute=False)

    def get_namespace(self):
        command = 'ibmcloud cr namespaces'
        return None

    def docker_tag_for_ibmcloud(self):
        if not self.namespace:
            self.namespace = '<enter-namespace>'
        command = 'docker tag {}:{} us.icr.io/{}/{}:{}'.format(self.name, self.tag, self.namespace, self.name, self.tag)
        return Action(suggested_command=command, confidence=1.0, execute=False)

    def docker_push(self):
        if not self.namespace:
            self.namespace = '<enter-namespace>'
        command = 'docker push us.icr.io/{}/{}:{}'.format(self.namespace, self.name, self.tag)
        return Action(suggested_command=command, confidence=1.0, execute=False)

    def list_images(self):
        command = 'ibmcloud cr image-list'
        return Action(suggested_command=command, confidence=1.0, execute=False)

    def get_image_name_to_delete(self):
        return None

    def ibmcloud_delete_image(self):
        command = 'ibmcloud cr image-rm us.icr.io/{}/'.format(self.namespace, self.image_to_remove)
        return Action(suggested_command=command, confidence=1.0)

    def check_account_free(self):
        command = 'ibmcloud account show'
        return Action(suggested_command=command, confidence=1.0, execute=False)

    def check_account_paid(self):
        command = 'ibmcloud account show'
        return Action(suggested_command=command, confidence=1.0, execute=False)

    def set_protocol(self):
        return None

    def ask_protocol(self):
        description = 'Do you want to use NodePort protocol?'
        return Action(suggested_command=NOOP_COMMAND, description=description, confidence=1.0, execute=False)

    def build_yaml(self):
        app_yaml = open(_path_to_yaml_temnplate, 'r').read()
        app_yaml = app_yaml.replace('{name}', self.name).replace('{tag}', self.tag).replace('{namespace}', self.namespace).replace('{protocol}', self.protocol).replace('{host_port}', self.host_port).replace('{local_port}', self.local_port)
        with open(_real_path + '/app.yaml', 'w') as f:
            f.write(app_yaml)
        self.yaml = app_yaml
        return Action(suggested_command=NOOP_COMMAND, description=self.yaml)

    def get_set_cluster_config(self):
        if not self.cluster_name:
            self.cluster_name = '<enter-cluster-name>'
        command = 'ibmcloud ks cluster-config {} | grep -e "export" | echo'.format(self.cluster_name)
        return Action(suggested_command=command, confidence=1.0)

    def kube_deploy(self):
        command = 'kubectl apply -f {}'.format(_path_to_yaml_temnplate)
        return Action(suggested_command=command, confidence=1.0)

class HelpMeAgent(Agent):

    def __init__(self):
        super(HelpMeAgent, self).__init__()
        inifile_path = os.path.join(str(Path(__file__).parent.absolute()), 'config.ini')
        self.store = Datastore(inifile_path)

    def compute_simple_token_similarity(self, src_sequence, tgt_sequence):
        src_tokens = set([x.lower().strip() for x in src_sequence.split()])
        tgt_tokens = set([x.lower().strip() for x in tgt_sequence.split()])
        return len(src_tokens & tgt_tokens) / len(src_tokens)

    def compute_confidence(self, query, forum, manpage):
        """
        Computes the confidence based on query, stack-exchange post answer and manpage

        Algorithm:
            1. Compute token-wise similarity b/w query and forum text
            2. Compute token-wise similarity b/w forum text and manpage description
            3. Return product of two similarities


        Args:
            query (str): standard error captured in state variable
            forum (str): answer text from most relevant stack exchange post w.r.t query
            manpage (str): manpage description for most relevant manpage w.r.t. forum

        Returns:
             confidence (float): confidence on the returned manpage w.r.t. query
        """
        query_forum_similarity = self.compute_simple_token_similarity(query, forum[0]['Content'])
        forum_manpage_similarity = self.compute_simple_token_similarity(forum[0]['Answer'], manpage)
        confidence = query_forum_similarity * forum_manpage_similarity
        return confidence

    def get_next_action(self, state: State) -> Action:
        return Action(suggested_command=state.command)

    def post_execute(self, state: State) -> Action:
        logger.info('==================== In Helpme Bot:post_execute ============================')
        logger.info('State:\n\tCommand: {}\n\tError Code: {}\n\tStderr: {}'.format(state.command, state.result_code, state.stderr))
        logger.info('============================================================================')
        if state.result_code == '0':
            return Action(suggested_command=state.command)
        apis: OrderedDict = self.store.get_apis()
        helpWasFound = False
        for provider in apis:
            if provider == 'manpages':
                logger.info(f"Skipping search provider 'manpages'")
                continue
            thisAPI: Provider = apis[provider]
            if not thisAPI.can_run_on_this_os():
                logger.info(f"Skipping search provider '{provider}'")
                logger.info(f'==> Excluded on platforms: {str(thisAPI.get_excludes())}')
                continue
            logger.info(f"Processing search provider '{provider}'")
            if thisAPI.has_variants():
                logger.info(f'==> Has search variants: {str(thisAPI.get_variants())}')
                variants: List = thisAPI.get_variants()
            else:
                logger.info(f'==> Has no search variants')
                variants: List = [None]
            for variant in variants:
                if variant is not None:
                    logger.info(f"==> Searching variant '{variant}'")
                    data = self.store.search(state.stderr, service=provider, size=1, searchType=variant)
                else:
                    data = self.store.search(state.stderr, service=provider, size=1)
                if data:
                    apiString = str(thisAPI)
                    if variant is not None:
                        apiString = f"{apiString} '{variant}' variant"
                    logger.info(f'==> Success!!! Found a result in the {apiString}')
                    searchResult = thisAPI.extract_search_result(data)
                    manpages = self.store.search(searchResult, service='manpages', size=5)
                    if manpages:
                        logger.info('==> Success!!! found relevant manpages.')
                        command = manpages['commands'][-1]
                        confidence = manpages['dists'][-1]
                        confidence = 1.0
                        logger.info('==> Command: {} \t Confidence:{}'.format(command, confidence))
                        suggested_command = 'man {}'.format(command)
                        description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f'I did little bit of Internet searching for you, ').append(f'and found this in the {thisAPI}:\n').info().append(thisAPI.get_printable_output(data)).warning().append('Do you want to try: man {}'.format(command)).to_console()
                        helpWasFound = True
                        break
            if helpWasFound:
                break
        if not helpWasFound:
            logger.info('Failure: Unable to be helpful')
            logger.info('============================================================================')
            suggested_command = NOOP_COMMAND
            description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f"Sorry. It looks like you have stumbled across a problem that even the Internet doesn't have answer to.\n").info().append(f'Have you tried turning it OFF and ON again. ;)').to_console()
            confidence = 0.0
        return Action(suggested_command=suggested_command, description=description, confidence=confidence)

class Voice(Agent):

    def __init__(self):
        super(Voice, self).__init__()
        self._api_filename = 'openai_api.key'
        self._priming_filename = 'priming.json'
        self._tmp_filepath = os.path.join(tempfile.gettempdir(), 'tts.mp3')
        self._gpt_api = self.__init_gpt_api__()
        self.__prime_gpt_model__()

    def __init_gpt_api__(self):
        curdir = str(Path(__file__).parent.absolute())
        key_filepath = os.path.join(curdir, self._api_filename)
        with open(key_filepath, 'r') as f:
            key = f.read()
        gpt_api = GPT()
        gpt_api.set_api_key(key)
        return gpt_api

    def __prime_gpt_model__(self):
        curdir = str(Path(__file__).parent.absolute())
        priming_filepath = os.path.join(curdir, self._priming_filename)
        with open(priming_filepath, 'r') as f:
            priming_examples = json.load(f)
        for priming_set in priming_examples:
            ip, op = (priming_set['input'], priming_set['output'])
            example = Example(ip, op)
            self._gpt_api.add_example(example)

    def summarize_output(self, state):
        stderr = str(state.stderr)
        prompt = stderr.split('\n')[0]
        gpt_summary = self._gpt_api.get_top_reply(prompt, strip_output_suffix=True)
        summary = f'error. {gpt_summary}'
        return summary

    def synthesize(self, text):
        """ Converts text to audio and saves to temp file """
        tts = gTTS(text, lang='en', lang_check=False)
        tts.save(self._tmp_filepath)

    def speak(self):
        subprocess.Popen(['nohup', 'ffplay', '-nodisp', '-autoexit', '-nostats', '-hide_banner', '-loglevel', 'warning', self._tmp_filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def post_execute(self, state: State) -> Action:
        if state.result_code == '0':
            return Action(suggested_command=state.command)
        text_to_speak = self.summarize_output(state)
        self.synthesize(text_to_speak)
        self.speak()
        return Action(suggested_command=NOOP_COMMAND, confidence=0.01)

    def get_next_action(self, state: State) -> Action:
        return Action(suggested_command=state.command)

class FixBot(Agent):
    """
    Fixes the last executed command by running it through the `thefuck` plugin
    """

    def __init__(self):
        super(FixBot, self).__init__()
        pass

    def get_next_action(self, state: State) -> Action:
        return Action(suggested_command=state.command)

    def post_execute(self, state: State) -> Action:
        if state.result_code == '0':
            return Action(suggested_command=state.command)
        cmd = str(state.command)
        stderr = str(state.stderr)
        try:
            settings.init()
            cmd = Command(cmd, stderr)
            cmd_corrected = get_corrected_commands(cmd)
            cmd_to_run = next(cmd_corrected).script
        except Exception:
            return Action(suggested_command=state.command, confidence=0.1)
        else:
            return Action(description=Colorize().info().append('Maybe you want to try: {}'.format(cmd_to_run)).to_console(), confidence=0.8)

class ManPageAgent(Agent):

    def __init__(self):
        super(ManPageAgent, self).__init__()
        self.question_detection_mod = QuestionDetection()
        self._config = None
        self._API_URL = None
        self.read_config()

    def read_config(self):
        curdir = str(Path(__file__).parent.absolute())
        config_path = os.path.join(curdir, 'config.json')
        with open(config_path, 'r') as f:
            self._config = json.load(f)
        self._API_URL = self._config['API_URL']

    def __call_api__(self, search_text):
        payload = {'text': search_text, 'result_count': 1}
        headers = {'Content-Type': 'application/json'}
        r = requests.post(self._API_URL, params=payload, headers=headers)
        if r.status_code == 200:
            return r.json()
        return None

    def get_next_action(self, state: State) -> Action:
        cmd = state.command
        is_question = self.question_detection_mod.is_question(cmd)
        if not is_question:
            return Action(suggested_command=state.command, confidence=0.0)
        response = None
        try:
            response = self.__call_api__(cmd)
        except Exception:
            pass
        if response is None or response['status'] != 'success':
            return Action(suggested_command=state.command, confidence=0.0)
        command = response['commands'][-1]
        confidence = response['dists'][-1]
        try:
            cmd_tldr = tldr_wrapper.get_command_tldr(command)
        except Exception as err:
            print('Exception: ' + str(err))
            cmd_tldr = ''
        return Action(suggested_command='man {}'.format(command), confidence=confidence, description=cmd_tldr)

class RLTKBandit(Orchestrator):

    def __init__(self):
        super(RLTKBandit, self).__init__()
        self._config_filepath = os.path.join(Path(__file__).parent.absolute(), 'config.yml')
        self._bandit_config_filepath = os.path.join(Path(__file__).parent.absolute(), 'bandit_config.json')
        self._noop_confidence = None
        self._agent = None
        self._n_actions = None
        self._action_order = None
        self._warm_start = None
        self._warm_start_type = None
        self._warm_start_kwargs = None
        self._reward_match_threshold = None
        self.load_bandit_state()
        self.load_state()
        self.warm_start_orchestrator()

    def load_bandit_state(self):
        with open(self._bandit_config_filepath, 'r') as conf_file:
            bandit_config = json.load(conf_file)
        self._noop_confidence = bandit_config['noop_confidence']
        self._warm_start = bandit_config['warm_start']
        self._warm_start_type = bandit_config['warm_start_config']['type']
        self._warm_start_kwargs = bandit_config['warm_start_config']['kwargs']
        self._reward_match_threshold = bandit_config.get('reward_match_threshold', 0.7)

    def get_orchestrator_state(self):
        state = {'agent': self._agent, 'action_order': self._action_order, 'warm_start': self._warm_start}
        return state

    def load_state(self):
        state = self.load()
        default_action_order = {self.noop_command: 0}
        self._agent = state.get('agent', None)
        if self._agent is None:
            self._agent = instantiate_from_file(self._config_filepath)
        self._action_order = state.get('action_order', None)
        if self._action_order is None:
            self._action_order = default_action_order
        self._n_actions = self._agent.num_actions
        self._warm_start = state.get('warm_start', self._warm_start)

    def warm_start_orchestrator(self):
        """
        Warm starts the orchestrator (pre-trains the weights) to suit a
        particular profile
        """

        def noop_setup():
            profile = 'noop-always'
            kwargs = {'n_points': 1000, 'context_size': self._n_actions, 'noop_position': 0}
            return (profile, kwargs)

        def ignore_skill_setup(skill_name):
            self.__add_to_action_order__(skill_name)
            profile = 'ignore-skill'
            kwargs = {'n_points': 1000, 'context_size': self._n_actions, 'skill_idx': self._action_order[skill_name]}
            return (profile, kwargs)

        def max_orchestrator_setup():
            profile = 'max-orchestrator'
            kwargs = {'n_points': 1000, 'context_size': self._n_actions}
            return (profile, kwargs)

        def preferred_skill_orchestrator_setup(advantage_skill, disadvantage_skill):
            self.__add_to_action_order__(advantage_skill)
            self.__add_to_action_order__(disadvantage_skill)
            profile = 'preferred-skill'
            kwargs = {'n_points': 1000, 'context_size': self._n_actions, 'advantage_skillidx': self._action_order[advantage_skill], 'disadvantage_skillidx': self._action_order[disadvantage_skill]}
            return (profile, kwargs)
        try:
            warm_start_methods = {'noop': noop_setup, 'ignore-skill': ignore_skill_setup, 'max-orchestrator': max_orchestrator_setup, 'preferred-skill': preferred_skill_orchestrator_setup}
            method = warm_start_methods[self._warm_start_type.lower()]
            profile, kwargs = method(**self._warm_start_kwargs)
            tids, contexts, arm_rewards = warm_start_datagen.get_warmstart_data(profile, **kwargs)
            self._agent.warm_start(tids, arm_rewards, contexts=contexts)
            self._warm_start = False
            self.save()
        except Exception as err:
            logger.warning('Exception in warm starting orchestrator. Error: ' + str(err))
            raise err

    def choose_action(self, command: State, agent_names: List[str], candidate_actions: Optional[List[Union[Action, List[Action]]]], force_response: bool, pre_post_state: str):
        if not candidate_actions:
            return None
        if isinstance(candidate_actions, Action):
            candidate_actions = [candidate_actions]
        context = self.__build_context__(candidate_actions)
        action_idx = self._agent.choose(t_id=command.command_id, context=context, num_arms=1)
        suggested_action = self.__choose_action__(action_idx[0], candidate_actions)
        if suggested_action is None:
            suggested_action = Action(suggested_command=command.command)
        return suggested_action

    def __build_context__(self, candidate_actions: Optional[List[Union[Action, List[Action]]]]) -> np.array:
        context = [0.0] * self._n_actions
        noop_pos = self._action_order[self.noop_command]
        context[noop_pos] = self._noop_confidence
        for action in candidate_actions:
            self.__add_to_action_order__(action.agent_owner)
            pos = self._action_order[action.agent_owner]
            conf = self.__calculate_confidence__(action)
            context[pos] = conf
        return np.array(context, dtype=np.float)

    def __add_to_action_order__(self, agent_name):
        if agent_name in self._action_order:
            return
        max_action_order = max(self._action_order.values())
        self._action_order[agent_name] = max_action_order + 1

    def __choose_action__(self, action_idx: int, candidate_actions: Optional[List[Union[Action, List[Action]]]]):
        suggested_agent = None
        for agent_name, agent_idx in self._action_order.items():
            if agent_idx == action_idx:
                suggested_agent = agent_name
                break
        if suggested_agent == self.noop_command or suggested_agent is None:
            return None
        for action in candidate_actions:
            if action.agent_owner == suggested_agent:
                return action
        return None

    def record_transition(self, prev_state: TerminalReplayMemoryComplete, current_state_pre: TerminalReplayMemory):
        try:
            prev_state_pre = prev_state.pre_replay
            prev_state_post = prev_state.post_replay
            if prev_state_pre.command.action_suggested is None or prev_state_post.command.action_suggested is None or prev_state_post.command.action_suggested.suggested_command == self.noop_command:
                return
            reward = float(prev_state_post.command.suggested_executed)
            currently_executed_command = current_state_pre.command.command
            all_the_stuff_from_last_execution = prev_state_pre.command.action_suggested.suggested_command + prev_state_pre.command.action_suggested.description + prev_state_post.command.action_suggested.description
            base = set(currently_executed_command.split())
            reference = set(all_the_stuff_from_last_execution.split())
            match_score = len(base & reference) / len(base)
            reward += float(match_score > self._reward_match_threshold)
            self._agent.observe(prev_state.post_replay.command.command_id, reward)
        except Exception as err:
            logger.warning(f'Error in record_transition of bandit orchestrator. Error: {err}')

