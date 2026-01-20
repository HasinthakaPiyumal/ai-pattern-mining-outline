# Cluster 38

class FactorioInstance:
    namespace_class = FactorioNamespace
    _cleanup_registered = False

    def __init__(self, address='localhost', fast=True, tcp_port=START_RCON_PORT, inventory: Dict={}, cache_scripts=True, all_technologies_researched=True, clear_entities=True, peaceful=True, num_agents=1, reset_speed=10, reset_paused=False, **kwargs):
        self.id = str(uuid.uuid4())[:8]
        self.num_agents = num_agents
        self.persistent_vars = {}
        self.tcp_port = tcp_port
        self.rcon_client, self.address = self.connect_to_server(address, tcp_port)
        self.fast = fast
        self._ticks_elapsed = 0
        self._is_initialised = False
        self.peaceful = peaceful
        self.namespaces = [self.namespace_class(self, i) for i in range(num_agents)]
        render_message_tool = None
        if hasattr(self.first_namespace, '_render_message'):
            render_message_tool = self.first_namespace._render_message
        self.game_control = GameControl(self.rcon_client, render_message_tool, reset_speed, reset_paused)
        self.game_control.reset_to_defaults()
        self.lua_script_manager = LuaScriptManager(self.rcon_client, cache_scripts)
        self.script_dict = {**self.lua_script_manager.lib_scripts, **self.lua_script_manager.tool_scripts}
        self.pre_tool_hooks = {}
        self.post_tool_hooks = {}
        self.lua_script_manager.load_init_into_game('initialise')
        self.lua_script_manager.setup_tools(self)
        if inventory is None:
            inventory = {}
        self.initial_inventory = inventory
        self.initialise(fast, all_technologies_researched, clear_entities)
        self.initial_score = 0
        try:
            self.first_namespace.score()
        except Exception:
            self.lua_script_manager = LuaScriptManager(self.rcon_client, False)
            self.script_dict = {**self.lua_script_manager.lib_scripts, **self.lua_script_manager.tool_scripts}
            self.lua_script_manager.setup_tools(self)
            self.initialise(fast, all_technologies_researched, clear_entities)
        self.initial_score, goal = self.first_namespace.score()
        if not FactorioInstance._cleanup_registered:
            atexit.register(self.cleanup)
            FactorioInstance._cleanup_registered = True
        self._executor = ThreadPoolExecutor(max_workers=2)

    @property
    def namespace(self):
        if len(self.namespaces) == 1:
            return self.namespaces[0]
        else:
            raise ValueError('Can only use .namespace for single-agent instances')

    @property
    def first_namespace(self) -> Optional[FactorioNamespace]:
        return self.namespaces[0] if self.namespaces else None

    @property
    def is_multiagent(self):
        return self.num_agents > 1

    def reset(self, game_state: Optional[GameState]=None, reset_position: bool=False, all_technologies_researched: bool=True, clear_entities: bool=True):
        assert not game_state or len(game_state.inventories) == self.num_agents, 'Game state must have the same number of inventories as num_agents'
        for namespace in self.namespaces:
            namespace.reset()
        if not game_state:
            inventories = [self.initial_inventory] * self.num_agents
            self.first_namespace._reset(inventories, reset_position, all_technologies_researched, clear_entities)
            if not all_technologies_researched:
                self.first_namespace._load_research_state(ResearchState(technologies={}, research_progress=0, current_research=None, research_queue=[], progress={}))
        else:
            self.first_namespace._reset(game_state.inventories, reset_position, all_technologies_researched, clear_entities)
            self.first_namespace._load_entity_state(game_state.entities, decompress=True)
            self.first_namespace._load_research_state(game_state.research)
            for i in range(min(self.num_agents, len(game_state.agent_messages))):
                self.namespaces[i].load_messages(game_state.agent_messages[i])
            for i in range(self.num_agents):
                self.namespaces[i].load(game_state.namespaces[i])
        self.game_control.reset_to_defaults()
        try:
            self.initial_score, _ = self.first_namespace.score()
        except Exception:
            self.initial_score = 0

    def set_speed(self, speed: float):
        """Set game speed (only affects speed when unpaused)"""
        self.game_control.set_speed(speed)

    def get_speed(self) -> float:
        """Get current speed setting (regardless of pause state)"""
        return self.game_control.get_speed()

    def get_elapsed_ticks(self):
        """Get the number of ticks elapsed since the game started"""
        return self.game_control.get_elapsed_ticks()

    def pause(self):
        """Pause the game (preserves speed setting)"""
        self.game_control.pause()

    def unpause(self):
        """Unpause the game (restores previous speed)"""
        self.game_control.unpause()

    def set_speed_and_unpause(self, speed: float):
        """Set speed and ensure game is unpaused - common use case"""
        self.game_control.set_speed_and_unpause(speed)

    def get_system_prompt(self, agent_idx: int=0) -> str:
        """
        Get the system prompt for the Factorio environment.
        This includes all the available actions, objects, and entities that the agent can interact with.
        We get the system prompt by loading the schema, definitions, and entity definitions from their source files.
        These are converted to their signatures - leaving out the implementations.
        :return:
        """
        execution_path = Path(os.path.dirname(os.path.realpath(__file__)))
        generator = SystemPromptGenerator(str(execution_path))
        return generator.generate_for_agent(agent_idx=agent_idx, num_agents=self.num_agents)

    @staticmethod
    def connect_to_server(address, tcp_port):
        try:
            rcon_client = RCONClient(address, tcp_port, RCON_PASSWORD)
            address = address
        except ConnectionError as e:
            print(e)
            rcon_client = RCONClient('localhost', tcp_port, RCON_PASSWORD)
            address = 'localhost'
        try:
            rcon_client.connect()
            player_count = rcon_client.send_command('/sc rcon.print(#game.players)')
            if int(player_count) == 0:
                print("WARNING: LuaPlayer hasn't been initialised into the game. Entity placement behavior _may_ be incorrect for boilers and pumps.")
        except Exception as e:
            raise ConnectionError(f'Could not connect to {address} at tcp/{tcp_port}: \n{e.args[0]}')
        print(f'Connected to {address} client at tcp/{tcp_port}.')
        return (rcon_client, address)

    def __eval_with_error(self, expr, agent_idx=0, timeout=60):
        """Evaluate an expression with a timeout, and return the result without error handling"""

        def handler(signum, frame):
            raise TimeoutError()
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        try:
            return self.namespaces[agent_idx].eval_with_timeout(expr)
        finally:
            signal.alarm(0)

    def eval_with_error(self, expr, agent_idx=0, timeout=60):
        """Evaluate an expression with a timeout, and return the result without error handling"""
        future = self._executor.submit(self.namespaces[agent_idx].eval_with_timeout, expr)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            future.cancel()
            raise TimeoutError()
        except Exception:
            raise

    def eval(self, expr, agent_idx=0, timeout=60):
        """Evaluate several lines of input, returning the result of the last line with a timeout"""
        try:
            return self.eval_with_error(expr, agent_idx, timeout)
        except TimeoutError:
            return (-1, '', 'Error: Evaluation timed out')
        except Exception as e:
            message = e.args[0].replace('\\n', '')
            return (-1, '', f'{message}'.strip())

    def initialise(self, fast=True, all_technologies_researched=True, clear_entities=True):
        self.rcon_client.send_command(f'/sc global.fast = {str(fast).lower()}')
        self.first_namespace._create_agent_characters(self.num_agents)
        init_scripts = ['lualib_util', 'utils', 'alerts', 'connection_points', 'recipe_fluid_connection_mappings', 'serialize']
        for script_name in init_scripts:
            self.lua_script_manager.load_init_into_game(script_name)
        if self.peaceful:
            self.rcon_client.send_command('/sc global.remove_enemies()')
        inventories = [self.initial_inventory] * self.num_agents
        self.first_namespace._reset(inventories, reset_position=False, all_technologies_researched=all_technologies_researched, clear_entities=clear_entities)
        self.first_namespace._clear_collision_boxes()

    def get_warnings(self, seconds=10):
        """
        Get all alerts that have been raised before the last n seconds
        :param seconds: The number of seconds to look back
        :return:
        """
        start = timer()
        lua_response = self.rcon_client.send_command(f'/sc rcon.print(dump(global.get_alerts({seconds})))')
        alert_dict, duration = _lua2python('alerts', lua_response, start=start)
        if isinstance(alert_dict, dict):
            alerts = list(alert_dict.values())
            alert_strings = []
            for alert in alerts:
                issues = ', '.join([al.replace('_', ' ') for al in list(alert['issues'].values())])
                alert_strings.append(f'{alert['entity_name']} at {tuple(alert['position'].values())}: {issues}')
            return alert_strings
        else:
            return []

    def cleanup(self):
        if hasattr(self, 'rcon_client') and self.rcon_client:
            self.rcon_client.close()
        self.post_tool_hooks = {}
        self.pre_tool_hooks = {}
        for thread in threading.enumerate():
            if thread != threading.current_thread() and thread.is_alive() and (not thread.daemon):
                try:
                    thread.join(timeout=5)
                except Exception as e:
                    print(f'Error joining thread {thread.name}: {e}')
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True, cancel_futures=True)

