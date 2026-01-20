# Cluster 13

def test_should_return_the_command_to_get_the_plugin_from_url_and_move_it_into_plugin_folder_server():
    agent_datasource = AgentDatasource()
    clai_install_command_runner = ClaiInstallCommandRunner(agent_datasource)
    command_to_execute = clai_install_command_runner.execute(CLAI_INSTALL_STATE_URL)
    dir_to_install = CLAI_INSTALL_STATE_URL.command.replace(f'{'clai install'}', '').strip()
    assert command_to_execute.suggested_command == f'cd $CLAI_PATH/clai/server/plugins && curl -O {dir_to_install}'

def test_should_return_the_command_to_get_the_plugin_from_route_and_move_it_into_plugin_folder_server(mocker):
    mock_exisit = mocker.patch('os.path.exists')
    mock_exisit.return_value = True
    mock_exisit = mocker.patch('os.path.isdir')
    mock_exisit.return_value = True
    agent_datasource = AgentDatasource()
    clai_install_command_runner = ClaiInstallCommandRunner(agent_datasource)
    command_to_execute = clai_install_command_runner.execute(CLAI_INSTALL_STATE_FOLDER)
    dir_to_install = CLAI_INSTALL_STATE_FOLDER.command.replace(f'{'clai install'}', '').strip()
    assert command_to_execute.suggested_command == f'cp -R {dir_to_install} $CLAI_PATH/clai/server/plugins'

def test_should_return_the_message_error_when_the_folder_doesnt_exist(mocker):
    mock_exisit = mocker.patch('os.path.exists')
    mock_exisit.return_value = False
    mock_exisit = mocker.patch('os.path.isdir')
    mock_exisit.return_value = False
    agent_datasource = AgentDatasource()
    clai_install_command_runner = ClaiInstallCommandRunner(agent_datasource)
    command_to_execute = clai_install_command_runner.execute(CLAI_INSTALL_STATE_FOLDER)
    dir_to_install = CLAI_INSTALL_STATE_FOLDER.command.replace(f'{'clai install'}', '').strip()
    assert command_to_execute.suggested_command == ':'
    assert command_to_execute.description == create_error_install(dir_to_install).description

def create_error_install(name: str) -> Action:
    text = Colorize().warning().append(f'{name} is not a valid skill to add to the catalog. You need to write a folder or a valid url. \n').append('Example: clai install ./my_new_agent').to_console()
    return Action(suggested_command=':', description=text, execute=True)

class CommandRunnerFactory:

    def __init__(self, agent_datasource: AgentDatasource, config_storage: ConfigStorage, server_status_datasource: ServerStatusDatasource, orchestrator_provider: OrchestratorProvider):
        self.server_status_datasource = server_status_datasource
        self.clai_commands: Dict[str, CommandRunner] = {'skills': ClaiPluginsCommandRunner(agent_datasource), 'orchestrate': ClaiOrchestrateCommandRunner(orchestrator_provider), 'activate': ClaiSelectCommandRunner(config_storage, agent_datasource), 'deactivate': ClaiUnselectCommandRunner(config_storage, agent_datasource), 'manual': ClaiPowerDisableCommandRunner(server_status_datasource), 'auto': ClaiPowerCommandRunner(server_status_datasource), 'install': ClaiInstallCommandRunner(agent_datasource), 'last-info': ClaiLastInfoCommandRunner(server_status_datasource), 'reload': ClaiReloadCommandRunner(agent_datasource), 'help': ClaiHelpCommandRunner()}
        self.clai_post_commands: Dict[str, PostCommandRunner] = {'activate': ClaiSelectCommandRunner(config_storage, agent_datasource), 'last-info': ClaiLastInfoCommandRunner(server_status_datasource), 'install': ClaiInstallCommandRunner(agent_datasource)}

    def provide_command_runner(self, command: str, selected_agent: AgentRunner) -> CommandRunner:
        if command.startswith(CLAI_COMMAND_NAME):
            clai_command_name = command.replace(CLAI_COMMAND_NAME, '', 1).strip()
            return self.__get_clai_command_runner(clai_command_name, selected_agent)
        return AgentCommandRunner(selected_agent, self.server_status_datasource)

    def provide_post_command_runner(self, command: str, selected_agent: AgentRunner) -> PostCommandRunner:
        if command.startswith(CLAI_COMMAND_NAME):
            clai_command_name = command.replace(CLAI_COMMAND_NAME, '', 1).strip()
            return self.__get_clai_post_command_runner(clai_command_name, selected_agent)
        return AgentCommandRunner(selected_agent, self.server_status_datasource)

    def __get_clai_post_command_runner(self, clai_command_name: str, selected_agent: AgentRunner) -> PostCommandRunner:
        clai_post_commands_names = self.clai_post_commands.keys()
        commands_filtered = filter(clai_command_name.startswith, clai_post_commands_names)
        command_found = next(commands_filtered, None)
        if command_found:
            return self.clai_post_commands[command_found]
        return ClaiDelegateToAgentCommandRunner(AgentCommandRunner(selected_agent, self.server_status_datasource))

    def __get_clai_command_runner(self, clai_command_name: str, selected_agent: AgentRunner) -> CommandRunner:
        clai_commands_names = self.clai_commands.keys()
        commands_filtered = filter(clai_command_name.startswith, clai_commands_names)
        command_found = next(commands_filtered, None)
        if command_found:
            return self.clai_commands[command_found]
        return ClaiDelegateToAgentCommandRunner(AgentCommandRunner(selected_agent, self.server_status_datasource, ignore_threshold=True))

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

class ClaiDelegateToAgentCommandRunner(CommandRunner, PostCommandRunner):

    def __init__(self, agent: AgentCommandRunner):
        self.agent = agent

    def execute(self, state: State) -> Action:
        command_to_check = state.command.replace(CLAI_COMMAND_NAME, '', 1).strip()
        state.command = command_to_check
        if not command_to_check.strip():
            return ClaiHelpCommandRunner().execute(state)
        if command_to_check.startswith('"'):
            possible_agents = command_to_check.split('"')[1::2]
            if possible_agents:
                agent_name = possible_agents[0]
                self.agent.force_agent = agent_name
                state.command = command_to_check.replace(f'"{agent_name}"', '', 1).strip()
        action = self.agent.execute(state)
        if not action:
            return ClaiHelpCommandRunner().execute(state)
        if isinstance(action, Action):
            if action.is_same_command() and (not action.description):
                return ClaiHelpCommandRunner().execute(state)
        if isinstance(action, List):
            diffent_actions = list(filter(lambda value: not value.is_same_action(), action))
            if not diffent_actions:
                return ClaiHelpCommandRunner().execute(state)
        return action

    def execute_post(self, state: State) -> Action:
        return self.agent.execute_post(state)

