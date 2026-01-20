# Cluster 19

def get_printable_name(plugin: AgentDescriptor):
    composed_name = f'{plugin.name} '
    if plugin.installed:
        composed_name = composed_name + '(Installed)'
    else:
        composed_name = composed_name + '(Not Installed)'
    return composed_name

def create_error_select(selected_plugin: str) -> Action:
    text = Colorize().warning().append(f"'{selected_plugin}' is not a valid skill name. ").append('To check available skills, issue:\n >> clai skills\n').append('Example:\n >> clai activate nlc2cmd').to_console()
    return Action(suggested_command=':', description=text, execute=True)

def create_orchestrator_list(selected_orchestrator: str, all_orchestrator: List[OrchestratorDescriptor], verbose_mode=False) -> Action:
    text = 'Available Orchestrators:\n'
    for orchestrator in all_orchestrator:
        if selected_orchestrator == orchestrator.name:
            text += Colorize().emoji(Colorize.EMOJI_CHECK).complete().append(f' {orchestrator.name}\n').to_console()
            if verbose_mode:
                text += Colorize().complete().append(f' {orchestrator.description}\n').to_console()
        else:
            text += Colorize().emoji(Colorize.EMOJI_BOX).append(f' {orchestrator.name}\n').to_console()
            if verbose_mode:
                text += Colorize().append(f' {orchestrator.description}\n').to_console()
    return Action(suggested_command=':', description=text, execute=True)

def create_message_list(selected_plugin: List[str], all_plugins: List[AgentDescriptor], verbose_mode=False) -> Action:
    text = 'Available Skills:\n'
    for plugin in all_plugins:
        if plugin.pkg_name in selected_plugin:
            text += Colorize().emoji(Colorize.EMOJI_CHECK).complete().append(f' {get_printable_name(plugin)}\n').to_console()
            if verbose_mode:
                text += Colorize().complete().append(f'{plugin.description}\n').to_console()
        else:
            text += Colorize().emoji(Colorize.EMOJI_BOX).append(f' {get_printable_name(plugin)}\n').to_console()
            if verbose_mode:
                text += Colorize().append(f'{plugin.description}\n').to_console()
    return Action(suggested_command=':', description=text, execute=True)

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

def extract_quoted_agent_name(plugin_to_select):
    if plugin_to_select.startswith('"'):
        names = plugin_to_select.split('"')[1::2]
        if names:
            return names[0]
    return plugin_to_select

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

class ClaiPluginsCommandRunner(CommandRunner):
    __VERBOSE_MODE = '-v'

    def __init__(self, agent_datasource: AgentDatasource):
        self.agent_datasource = agent_datasource

    def execute(self, state: State) -> Action:
        action_to_return = create_message_list(self.agent_datasource.get_current_plugin_name(state.user_name), self.agent_datasource.all_plugins(), self.__VERBOSE_MODE in state.command)
        action_to_return.origin_command = state.command
        return action_to_return

class ClaiOrchestrateCommandRunner(CommandRunner):
    SELECT_DIRECTIVE = 'clai orchestrate'
    __VERBOSE_MODE = '-v'

    def __init__(self, orchestrator_provider: OrchestratorProvider):
        self.orchestrator_provider = orchestrator_provider

    def execute(self, state: State) -> Action:
        orchestrator_to_select = state.command.replace(f'{self.SELECT_DIRECTIVE}', '').strip()
        verbose = False
        if self.__VERBOSE_MODE in orchestrator_to_select:
            verbose = True
            orchestrator_to_select = ''
        else:
            orchestrator_to_select = extract_quoted_agent_name(orchestrator_to_select)
        if orchestrator_to_select:
            self.orchestrator_provider.select_orchestrator(orchestrator_to_select)
        return create_orchestrator_list(self.orchestrator_provider.get_current_orchestrator_name(), self.orchestrator_provider.all_orchestrator(), verbose)

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

