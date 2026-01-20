# Cluster 26

def override_last_command(last_command):
    lines = read_history()
    if lines:
        new_last_line = lines[-1].rstrip() + last_command
        lines[-1] = new_last_line
    else:
        lines.append(last_command)
    io.open(get_history_file_name(), 'w').writelines(lines)

def read_history():
    history_file_name = get_history_file_name()
    if os.path.isfile(history_file_name):
        return io.open(history_file_name, 'r', encoding='utf-8', errors='ignore').readlines()
    return []

def get_history_file_name():
    return os.environ.get('HISTFILE', os.path.expanduser('~/.bash_history'))

def restore_history(original_command: str):
    lines = read_history()
    new_lines = __remove_clai_history__(lines, original_command + '\n')
    io.open(get_history_file_name(), 'w').writelines(new_lines)

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

