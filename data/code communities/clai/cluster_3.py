# Cluster 3

def test_should_active_the_power_mode_when_use_the_command_clai_power(mocker):
    mock_agent = create_mock_agent()
    mocker.patch.object(AgentDatasource, 'get_instances', return_value=[mock_agent], autospec=True)
    message_handler = MessageHandler(ServerStatusDatasource(), mock_agent)
    action = message_handler.process_message(clai_power_state())
    assert action.suggested_command == NOOP_COMMAND
    assert action.description == 'You have enabled the auto mode'
    assert action.origin_command == 'clai auto'
    assert action.execute
    assert message_handler.server_status_datasource.is_power()

def create_mock_agent() -> Agent:
    agent = Mock(spec=Agent)
    agent.agent_name = 'demo_agent'
    return agent

def clai_power_state():
    return State(command_id=ANY_ID, user_name=ANY_NAME, command='clai auto')

def test_should_desactive_the_power_mode_when_use_the_command_clai_power_disable(mocker):
    mock_agent = create_mock_agent()
    mocker.patch.object(AgentDatasource, 'get_instances', return_value=[mock_agent], autospec=True)
    message_handler = MessageHandler(ServerStatusDatasource(), mock_agent)
    message_handler.server_status_datasource.set_power(True)
    action = message_handler.process_message(clai_power_disabled_state())
    assert action.suggested_command == NOOP_COMMAND
    assert action.description == 'You have enable the manual mode'
    assert not message_handler.server_status_datasource.is_power()

def test_should_not_change_power_variable_when_active_power_mode_and_it_already_active(mocker):
    mock_agent = create_mock_agent()
    mocker.patch.object(AgentDatasource, 'get_instances', return_value=[mock_agent], autospec=True)
    message_handler = MessageHandler(ServerStatusDatasource(), mock_agent)
    message_handler.server_status_datasource.set_power(True)
    action = message_handler.process_message(clai_power_state())
    assert action.suggested_command == NOOP_COMMAND
    assert action.description == 'You have the auto mode already enable, use clai manual to deactivate it'
    assert message_handler.server_status_datasource.is_power()

def test_should_not_change_power_variable_when_active_power_mode_and_it_already_disable(mocker):
    mock_agent = create_mock_agent()
    mocker.patch.object(AgentDatasource, 'get_instances', return_value=[mock_agent], autospec=True)
    message_handler = MessageHandler(ServerStatusDatasource(), mock_agent)
    action = message_handler.process_message(clai_power_disabled_state())
    assert action.suggested_command == NOOP_COMMAND
    assert action.description == 'You have manual mode already enable, use clai auto to activate it'
    assert not message_handler.server_status_datasource.is_power()

def test_should_have_action_execute_true_when_power_mode_is_active(mocker):
    mock_agent = create_mock_agent()
    mocker.patch.object(AgentDatasource, 'get_instances', return_value=[mock_agent], autospec=True)
    message_handler = MessageHandler(ServerStatusDatasource(), mock_agent)
    message_handler.process_message(clai_power_state())
    action = message_handler.process_message(ANY_COMMAND_MESSAGE)
    assert action.origin_command == ANY_COMMAND_MESSAGE.command
    assert action.execute

def expected_description(all_plugins, selected) -> str:
    text = 'Available Skills:\n'
    for plugin in all_plugins:
        if plugin.pkg_name in selected:
            text += Colorize().emoji(Colorize.EMOJI_CHECK).complete().append(f' {get_printable_name(plugin)}\n').to_console()
        else:
            text += Colorize().emoji(Colorize.EMOJI_BOX).append(f' {get_printable_name(plugin)}\n').to_console()
    return text

def test_should_return_the_list_of_plugins_with_default_selected_when_the_server_received_plugins_no_selected(mocker):
    mock_agent = create_mock_agent()
    mocker.patch.object(AgentDatasource, 'get_instances', return_value=[mock_agent], autospec=True)
    mocker.patch.object(ConfigStorage, 'read_config', return_value=NO_SELECTED, autospec=True)
    mocker.patch.object(AgentDatasource, 'all_plugins', return_value=ALL_PLUGINS, autospec=True)
    message_handler = MessageHandler(ServerStatusDatasource(), AgentDatasource())
    action = message_handler.process_message(clai_plugins_state())
    assert action.suggested_command == NOOP_COMMAND
    assert action.origin_command == 'clai skills'
    assert action.execute
    assert action.description == expected_description(ALL_PLUGINS, NO_SELECTED.default)

def clai_plugins_state():
    return State(command_id=ANY_ID, user_name=ANY_NAME, command='clai skills')

def test_should_return_the_list_of_plugins_with_selected_when_the_server_received_plugins(mocker):
    mock_agent = create_mock_agent()
    mocker.patch.object(AgentDatasource, 'get_instances', return_value=[mock_agent], autospec=True)
    agent_selected = 'nlc2cmd'
    mocker.patch.object(AgentDatasource, 'all_plugins', return_value=ALL_PLUGINS, autospec=True)
    mocker.patch.object(ConfigStorage, 'read_all_user_config', return_value=None, autospec=True)
    mocker.patch.object(ConfigStorage, 'read_config', return_value=PluginConfig(selected=[agent_selected], default_orchestrator='max_orchestrator'), autospec=True)
    message_handler = MessageHandler(ServerStatusDatasource(), AgentDatasource())
    action = message_handler.process_message(clai_plugins_state())
    assert action.suggested_command == NOOP_COMMAND
    assert action.origin_command == 'clai skills'
    assert action.execute
    assert action.description == expected_description(ALL_PLUGINS, agent_selected)

def test_should_return_the_list_without_any_selected_plugin_when_default_doesnt_exist(mocker):
    mock_agent = create_mock_agent()
    mocker.patch.object(AgentDatasource, 'get_instances', return_value=[mock_agent], autospec=True)
    mocker.patch.object(AgentDatasource, 'all_plugins', return_value=ALL_PLUGINS, autospec=True)
    mocker.patch.object(ConfigStorage, 'read_config', return_value=PluginConfig(default='', default_orchestrator='max_orchestrator'), autospec=True)
    message_handler = MessageHandler(ServerStatusDatasource(), AgentDatasource())
    action = message_handler.process_message(clai_plugins_state())
    assert action.suggested_command == NOOP_COMMAND
    assert action.origin_command == 'clai skills'
    assert action.execute
    assert action.description == expected_description(ALL_PLUGINS, '')

def test_should_return_the_install_command_when_the_new_plugin_is_not_installed_yet(mocker):
    mock_agent = create_mock_agent()
    mocker.patch.object(AgentDatasource, 'get_instances', return_value=[mock_agent], autospec=True)
    mocker.patch.object(ConfigStorage, 'read_config', return_value=PluginConfig(selected=['nlc2cmd'], default_orchestrator='max_orchestrator'), autospec=True)
    mocker.patch.object(ConfigStorage, 'store_config', return_value=None, autospec=True)
    mocker.patch.object(AgentDatasource, 'all_plugins', return_value=ALL_PLUGINS, autospec=True)
    message_handler = MessageHandler(ServerStatusDatasource(), AgentDatasource())
    select_agent = clai_select_state('nlc2cmd')
    action = message_handler.process_message(select_agent)
    assert action.suggested_command == '$CLAI_PATH/fileExist.sh nlc2cmd $CLAI_PATH'
    assert action.origin_command == select_agent.command
    assert message_handler.agent_datasource.get_current_plugin_name(select_agent.user_name) == ['nlc2cmd']

def clai_select_state(plugin_to_select):
    return State(command_id=ANY_ID, user_name=ANY_NAME, command=f'clai activate {plugin_to_select}')

def test_should_return_the_list_with_the_new_selected_values_if_exists_and_is_installed(mocker):
    mock_agent = create_mock_agent()
    mocker.patch.object(AgentDatasource, 'get_instances', return_value=[mock_agent], autospec=True)
    mocker.patch.object(ConfigStorage, 'read_config', return_value=PluginConfig(selected=['demo_agent'], default_orchestrator='max_orchestrator'), autospec=True)
    mocker.patch.object(ConfigStorage, 'store_config', return_value=None, autospec=True)
    mocker.patch.object(AgentDatasource, 'all_plugins', return_value=ALL_PLUGINS_WITH_TAR_INSTALLED, autospec=True)
    message_handler = MessageHandler(ServerStatusDatasource(), AgentDatasource())
    select_agent = clai_select_state('nlc2cmd')
    action = message_handler.process_message(select_agent)
    assert action.suggested_command == NOOP_COMMAND
    assert action.origin_command == select_agent.command
    assert action.execute
    assert message_handler.agent_datasource.get_current_plugin_name(select_agent.user_name) == ['nlc2cmd']

def test_should_return_an_error_when_agent_doesnt_exist(mocker):
    mock_agent = create_mock_agent()
    mocker.patch.object(AgentDatasource, 'get_instances', return_value=[mock_agent], autospec=True)
    mocker.patch.object(AgentDatasource, 'all_plugins', return_value=ALL_PLUGINS, autospec=True)
    mocker.patch.object(ConfigStorage, 'read_config', return_value=PluginConfig(selected=['nlc2cmd'], default_orchestrator='max_orchestrator'), autospec=True)
    mocker.patch.object(ConfigStorage, 'store_config', return_value=None, autospec=True)
    message_handler = MessageHandler(ServerStatusDatasource(), AgentDatasource())
    select_agent = clai_select_state('wrong_agent')
    action = message_handler.process_message(select_agent)
    assert action.suggested_command == NOOP_COMMAND
    assert action.origin_command == select_agent.command
    assert action.execute
    assert action.description == create_error_select('wrong_agent').description
    assert message_handler.agent_datasource.get_current_plugin_name(select_agent.user_name) == ['nlc2cmd']

def test_should_return_an_error_when_selected_is_empty(mocker):
    mock_agent = create_mock_agent()
    mocker.patch.object(AgentDatasource, 'get_instances', return_value=[mock_agent], autospec=True)
    mocker.patch.object(AgentDatasource, 'all_plugins', return_value=ALL_PLUGINS, autospec=True)
    mocker.patch.object(ConfigStorage, 'read_config', return_value=PluginConfig(selected=['nlc2cmd'], default_orchestrator='max_orchestrator'), autospec=True)
    mocker.patch.object(ConfigStorage, 'store_config', return_value=None, autospec=True)
    message_handler = MessageHandler(ServerStatusDatasource(), AgentDatasource())
    select_agent = clai_select_state('')
    action = message_handler.process_message(select_agent)
    assert action.suggested_command == NOOP_COMMAND
    assert action.origin_command == select_agent.command
    assert action.execute
    assert action.description == create_error_select('').description
    assert message_handler.agent_datasource.get_current_plugin_name(select_agent.user_name) == ['nlc2cmd']

def test_should_return_the_action_from_selected_agent_when_the_command_goes_to_the_agent_and_threshold_is_ok(mocker):
    mock_agent = create_mock_agent()
    mocker.patch.object(AgentDatasource, 'get_instances', return_value=[mock_agent], autospec=True)
    action_to_execute = Action(suggested_command='command', confidence=1.0)
    mock_agent.execute.return_value = action_to_execute
    mocker.patch.object(ConfigStorage, 'read_config', return_value=PluginConfig(selected=['demo_agent'], default_orchestrator='max_orchestrator'), autospec=True)
    message_handler = MessageHandler(ServerStatusDatasource(), AgentDatasource())
    action = message_handler.process_message(command_state())
    assert action.suggested_command == action_to_execute.suggested_command
    assert action.origin_command == command_state().command
    assert not action.execute
    assert not action.description

def command_state():
    return State(command_id=ANY_ID, user_name=ANY_NAME, command='command')

def test_should_return_empty_action_from_selected_agent_when_the_command_goes_to_the_agent_and_not_confidence(mocker):
    mock_agent = create_mock_agent()
    mocker.patch.object(AgentDatasource, 'get_instances', return_value=[mock_agent], autospec=True)
    action_to_execute = Action(suggested_command='command', confidence=0.1)
    mock_agent.execute.return_value = action_to_execute
    mocker.patch.object(ConfigStorage, 'read_config', return_value=PluginConfig(selected=['demo_agent'], default_orchestrator='max_orchestrator'), autospec=True)
    message_handler = MessageHandler(ServerStatusDatasource(), AgentDatasource())
    action = message_handler.process_message(command_state())
    assert action.suggested_command is action.origin_command
    assert action.origin_command == command_state().command
    assert not action.execute
    assert not action.description

def test_should_return_the_suggestion_from_agent_ignoring_confidence_if_is_clai_command(mocker):
    mock_agent = create_mock_agent()
    mocker.patch.object(AgentDatasource, 'get_instances', return_value=[mock_agent], autospec=True)
    action_to_execute = Action(suggested_command='command', confidence=0.0)
    mock_agent.execute.return_value = action_to_execute
    mocker.patch.object(ConfigStorage, 'read_config', return_value=PluginConfig(selected=['demo_agent'], default_orchestrator='max_orchestrator'), autospec=True)
    message_handler = MessageHandler(ServerStatusDatasource(), AgentDatasource())
    action = message_handler.process_message(COMMAND_AGENT_STATE)
    assert action.suggested_command == action_to_execute.suggested_command
    assert action.origin_command == command_state().command
    assert not action.execute
    assert not action.description

def test_should_return_the_suggestion_from_agent_ignoring_confidence_if_is_name_agent_command(mocker):
    mock_agent = create_mock_agent()
    mocker.patch.object(AgentDatasource, 'get_instances', return_value=[mock_agent], autospec=True)
    action_to_execute = Action(suggested_command='command', confidence=0.0)
    mock_agent.execute.return_value = action_to_execute
    mocker.patch.object(ConfigStorage, 'read_config', return_value=PluginConfig(selected=['demo_agent'], default_orchestrator='max_orchestrator'), autospec=True)
    message_handler = MessageHandler(ServerStatusDatasource(), AgentDatasource())
    action = message_handler.process_message(COMMAND_NAME_AGENT_STATE)
    assert action.suggested_command == action_to_execute.suggested_command
    assert action.origin_command == command_state().command
    assert not action.execute
    assert not action.description

def test_should_return_valid_action_if_the_select_agent_return_none(mocker):
    mock_agent = create_mock_agent()
    mocker.patch.object(AgentDatasource, 'get_instances', return_value=[mock_agent], autospec=True)
    mock_agent.execute.return_value = None
    mocker.patch.object(ConfigStorage, 'read_config', return_value=PluginConfig(selected=['demo_agent'], default_orchestrator='max_orchestrator'), autospec=True)
    message_handler = MessageHandler(ServerStatusDatasource(), AgentDatasource())
    action = message_handler.process_message(command_state())
    assert action.suggested_command is action.origin_command
    assert action.origin_command == command_state().command
    assert not action.execute
    assert not action.description

class ConfigStorage:

    def __init__(self, alternate_path: Optional[str]=None):
        self.alternate_path = alternate_path

    def get_config_path(self):
        if self.alternate_path:
            return self.alternate_path
        base_dir = os.path.dirname(clai.datasource.__file__)
        filename = os.path.join(base_dir, '../../configPlugins.json')
        return filename

    def read_all_user_config(self) -> PluginConfigJson:
        with open(self.get_config_path(), 'r') as json_file:
            loaded = json.load(json_file)
            config_for_all_users = PluginConfigJson(**loaded)
            return config_for_all_users

    def read_config(self, user_name: Optional[str]=None) -> PluginConfig:
        selected = None
        config_for_all_users = self.read_all_user_config()
        if user_name in config_for_all_users.selected:
            selected = config_for_all_users.selected[user_name]
        if not selected:
            if isinstance(config_for_all_users.default, str):
                selected = [config_for_all_users.default]
            else:
                selected = config_for_all_users.default
        return PluginConfig(selected=selected, default=config_for_all_users.default, default_orchestrator=config_for_all_users.default_orchestrator, installed=config_for_all_users.installed, report_enable=config_for_all_users.report_enable, user_install=config_for_all_users.user_install)

    def store_config(self, config: PluginConfig, user_name: str=None):
        current_config = self.read_all_user_config()
        with open(self.get_config_path(), 'w') as json_file:
            if user_name:
                current_config.selected[user_name] = config.selected
            current_config.installed = config.installed
            current_config.report_enable = config.report_enable
            current_config.orchestrator = config.orchestrator
            current_config.user_install = config.user_install
            json_as_string = str(current_config.json())
            json_file.write(json_as_string)

    def load_installed(self) -> List[str]:
        current_config = self.read_all_user_config()
        return current_config.installed

