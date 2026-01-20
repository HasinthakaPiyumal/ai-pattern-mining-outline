# Cluster 21

@dataclass
class GameState:
    """Serializable Factorio game state"""
    entities: str
    inventories: List[Any]
    research: Optional[ResearchState] = field()
    timestamp: float = field(default_factory=time.time)
    namespaces: List[bytes] = field(default_factory=list)
    agent_messages: List[Any] = field(default_factory=list)

    @property
    def is_multiagent(self) -> bool:
        return len(self.inventories) > 1

    @property
    def num_agents(self) -> int:
        return len(self.inventories)

    def parse_agent_messages(data: dict) -> List[Any]:
        agent_messages = data.get('agent_messages', [])
        if not isinstance(agent_messages, list):
            raise ValueError('agent_messages must be a list')
        if agent_messages and (not all((isinstance(msg, (dict, list)) for msg in agent_messages))):
            for idx, message in enumerate(agent_messages):
                if isinstance(message, dict):
                    continue
                elif isinstance(message, list):
                    if len(message) > 0:
                        if not all((isinstance(msg, dict) for msg in message)):
                            raise ValueError(f'agent_messages[{idx}] contains non-dictionary elements')
                else:
                    raise ValueError(f'agent_messages[{idx}] must be a dictionary or a list of dictionaries, but got {type(message)}')
        return agent_messages

    @classmethod
    def from_instance(cls, instance) -> 'GameState':
        """Capture current game state from Factorio instances"""
        entities = instance.first_namespace._save_entity_state(compress=True, encode=True)
        research_state = instance.first_namespace._save_research_state()
        namespaces = []
        for namespace in instance.namespaces:
            if hasattr(namespace, 'persistent_vars'):
                serializable_vars = filter_serializable_vars(namespace.persistent_vars)
                namespaces.append(pickle.dumps(serializable_vars))
            else:
                namespaces.append(bytes())
        inventories = [namespace.inspect_inventory() for namespace in instance.namespaces]
        agent_messages = [namespace.get_messages() for namespace in instance.namespaces]
        return cls(entities=entities, inventories=inventories, namespaces=namespaces, research=research_state, agent_messages=agent_messages)

    def __repr__(self):
        readable_namespaces = [pickle.loads(namespace) for namespace in self.namespaces]
        return f'GameState(entities={self.entities}, inventories={self.inventories}, timestamp={self.timestamp}, namespace={{{readable_namespaces}}}, agent_messages={self.agent_messages})'

    @classmethod
    def parse_raw(cls, json_str: str) -> 'GameState':
        data = json.loads(json_str)
        namespaces = []
        if 'namespaces' in data:
            namespaces = [bytes.fromhex(ns) if ns else bytes() for ns in data['namespaces']]
        research = None
        if 'research' in data:
            research = ResearchState(technologies={name: TechnologyState(**tech) for name, tech in data['research']['technologies'].items()}, current_research=data['research']['current_research'], research_progress=data['research']['research_progress'], research_queue=data['research']['research_queue'], progress=data['research']['progress'] if 'progress' in data['research'] else {})
        return cls(entities=data['entities'], inventories=data['inventories'], timestamp=data['timestamp'] if 'timestamp' in data else time.time(), namespaces=namespaces, research=research, agent_messages=cls.parse_agent_messages(data))

    @classmethod
    def parse(cls, data) -> 'GameState':
        if 'namespace' in data:
            data['namespaces'] = [data['namespace']]
            data['inventories'] = [data['inventory']]
            data['agent_messages'] = []
        namespaces = []
        if 'namespaces' in data:
            namespaces = [bytes.fromhex(ns) if ns else bytes() for ns in data['namespaces']]
        research = None
        if 'research' in data:
            research = ResearchState(technologies={name: TechnologyState(**tech) for name, tech in data['research']['technologies'].items()}, current_research=data['research']['current_research'], research_progress=data['research']['research_progress'], research_queue=data['research']['research_queue'], progress=data['research']['progress'] if 'progress' in data['research'] else {})
        return cls(entities=data['entities'], inventories=data['inventories'], timestamp=data['timestamp'] if 'timestamp' in data else time.time(), namespaces=namespaces, research=research, agent_messages=cls.parse_agent_messages(data))

    def to_raw(self) -> str:
        """Convert state to JSON string"""
        data = {'entities': self.entities, 'inventories': [inventory.__dict__ if hasattr(inventory, '__dict__') else inventory for inventory in self.inventories], 'timestamp': self.timestamp, 'namespaces': [ns.hex() if ns else '' for ns in self.namespaces], 'agent_messages': self.agent_messages}
        if self.research:
            data['research'] = {'technologies': {name: asdict(tech) for name, tech in self.research.technologies.items()}, 'current_research': self.research.current_research, 'research_progress': self.research.research_progress, 'research_queue': self.research.research_queue, 'progress': self.research.progress}
        return json.dumps(data)

    def to_instance(self, instance):
        """Restore game state to a Factorio instance"""
        assert instance.num_agents == self.num_agents, f'GameState can only be restored to a multiagent instance with the same number of agents (num_agents={self.num_agents})'
        instance.first_namespace._load_entity_state(self.entities, decompress=True)
        if self.inventories:
            for namespace, inventory in zip(instance.namespaces, self.inventories):
                namespace._set_inventory(inventory)
        if self.research:
            instance.first_namespace._load_research_state(self.research)
        if self.agent_messages:
            for i in range(instance.num_agents):
                if i < len(self.agent_messages):
                    instance.namespaces[i].load_messages(self.agent_messages[i])
        if self.namespaces:
            for namespace in instance.namespaces:
                if namespace:
                    restored_vars = pickle.loads(namespace)
                if not hasattr(instance, 'persistent_vars') or instance.persistent_vars is None:
                    instance.persistent_vars = {}
                instance.persistent_vars.update(restored_vars)

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

def eval_program_with_achievements(instance: Any, program: str) -> Tuple[List[str], str, bool, Dict[str, Dict[str, float]]]:
    """Evaluate a program and calculate achievements."""
    pre_flows = ProductionFlows.from_dict(instance.first_namespace._get_production_stats())
    try:
        score, goal, result = instance.eval_with_error(program, timeout=300)
        error = False
    except Exception as e:
        result = str(e)
        error = True
    post_flows = ProductionFlows.from_dict(instance.first_namespace._get_production_stats())
    achievements = calculate_achievements(pre_flows, post_flows)
    return (result.splitlines(), result, error, achievements)

def calculate_achievements(pre: ProductionFlows, post: ProductionFlows) -> Dict[str, Dict[str, float]]:
    """Calculate achievements between two production states."""
    achievements = {'static': {}, 'dynamic': {}}
    if not pre.is_valid() or not post.is_valid():
        print('Warning: Invalid production flows')
        return achievements
    post = deepcopy(post)
    new_flows = pre.get_new_flows(post)
    static_items = deepcopy(new_flows.harvested)
    for craft in new_flows.crafted:
        for item, value in craft['outputs'].items():
            static_items[item] = static_items.get(item, 0) + value
    for item in post.output:
        post_value = post.output[item]
        pre_value = pre.output.get(item, 0)
        if post_value > pre_value:
            created = post_value - pre_value
            static = static_items.get(item, 0)
            if static > 0:
                achievements['static'][item] = static
            if created > static:
                achievements['dynamic'][item] = created - static
    return achievements

class SaveResearchState(Tool):

    def __init__(self, connection, game_state):
        super().__init__(connection, game_state)

    def __call__(self) -> ResearchState:
        """
        Save the current research state of the force

        Returns:
            ResearchState: Complete research state including all technologies
        """
        state, _ = self.execute(self.player_index)
        if not isinstance(state, dict):
            raise Exception(f'Could not save research state: {state}')
        try:
            technologies = {}
            if 'technologies' in state:
                technologies = {name: TechnologyState(name=tech['name'], researched=tech['researched'], enabled=tech['enabled'], level=tech['level'], research_unit_count=tech['research_unit_count'], research_unit_energy=tech['research_unit_energy'], prerequisites=[x for x in tech['prerequisites'].values()], ingredients=[{x['name']: x['amount']} for x in tech['ingredients'].values()]) for name, tech in state['technologies'].items()}
            return ResearchState(technologies=technologies, current_research=state['current_research'] if 'current_research' in state else None, research_progress=state['research_progress'] if 'research_progress' in state else None, research_queue=[x for x in state['research_queue'].values()] if 'research_queue' in state else [], progress=state['progress'] if 'progress' in state else None)
        except Exception as e:
            print(f'Could not save technologies: {e}')
            raise e

@dataclass
class Observation:
    """Complete observation of the game state"""
    raw_text: str
    entities: List[str]
    inventory: Inventory
    research: ResearchState
    game_info: GameInfo
    score: float
    flows: ProductionFlows
    task_verification: Optional[TaskResponse]
    messages: List[AgentMessage]
    serialized_functions: List[Dict[str, Any]]
    task_info: Optional[TaskInfo]
    map_image: str

    @classmethod
    def from_dict(cls, obs_dict: Dict[str, Any]) -> 'Observation':
        """Create an Observation from a dictionary matching the gym observation space"""
        entities = obs_dict.get('entities', [])
        inventory = Inventory()
        for item in obs_dict.get('inventory', []):
            inventory[item['type']] = item['quantity']
        research = ResearchState(technologies={tech['name']: TechnologyState(name=tech['name'], researched=bool(tech['researched']), enabled=bool(tech['enabled']), level=tech['level'], research_unit_count=tech['research_unit_count'], research_unit_energy=tech['research_unit_energy'], prerequisites=tech['prerequisites'], ingredients={item['item']: item['amount'] for item in tech['ingredients']}) for tech in obs_dict.get('research', {}).get('technologies', [])}, current_research=obs_dict.get('research', {}).get('current_research'), research_progress=obs_dict.get('research', {}).get('research_progress', 0.0), research_queue=obs_dict.get('research', {}).get('research_queue', []), progress={item['name']: item['value'] for item in obs_dict.get('research', {}).get('progress', [])})
        game_info = GameInfo(tick=obs_dict.get('game_info', {}).get('tick', 0), time=obs_dict.get('game_info', {}).get('time', 0.0), speed=obs_dict.get('game_info', {}).get('speed', 0.0))
        flows_dict = obs_dict.get('flows', {})
        transformed_flows = {'input': {item['type']: item['rate'] for item in flows_dict.get('input', [])}, 'output': {item['type']: item['rate'] for item in flows_dict.get('output', [])}, 'crafted': flows_dict.get('crafted', []), 'harvested': {item['type']: item['amount'] for item in flows_dict.get('harvested', [])}, 'price_list': {item['type']: item['price'] for item in flows_dict.get('price_list', [])} if flows_dict.get('price_list') else None, 'static_items': {item['type']: item['value'] for item in flows_dict.get('static_items', [])} if flows_dict.get('static_items') else None}
        flows = ProductionFlows.from_dict(transformed_flows)
        task_verification = None
        if obs_dict.get('task_verification'):
            task_verification = TaskResponse(success=bool(obs_dict['task_verification']['success']), meta={item['key']: json.loads(item['value']) for item in obs_dict['task_verification'].get('meta', [])})
        messages = [AgentMessage(sender=msg['sender'], content=msg['content'], timestamp=msg['timestamp']) for msg in obs_dict.get('messages', [])]
        serialized_functions = obs_dict.get('serialized_functions', [])
        task_info = None
        task_dict = obs_dict.get('task_info', {})
        if task_dict.get('goal_description'):
            agent_instructions = task_dict.get('agent_instructions')
            if agent_instructions == '':
                agent_instructions = None
            task_info = TaskInfo(goal_description=task_dict.get('goal_description', ''), agent_instructions=agent_instructions, task_key=task_dict.get('task_key', ''), trajectory_length=task_dict.get('trajectory_length', 0))
        map_image = obs_dict.get('map_image', '')
        return cls(raw_text=obs_dict.get('raw_text', ''), entities=entities, inventory=inventory, research=research, game_info=game_info, score=obs_dict.get('score', 0.0), flows=flows, task_verification=task_verification, messages=messages, serialized_functions=serialized_functions, task_info=task_info, map_image=map_image)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Observation to a dictionary matching the gym observation space"""
        flows_dict = self.flows.to_dict()
        transformed_flows = {'input': [{'type': k, 'rate': v} for k, v in flows_dict['input'].items()], 'output': [{'type': k, 'rate': v} for k, v in flows_dict['output'].items()], 'crafted': flows_dict['crafted'], 'harvested': [{'type': k, 'amount': v} for k, v in flows_dict['harvested'].items()], 'price_list': [{'type': k, 'price': v} for k, v in (flows_dict['price_list'] or {}).items()], 'static_items': [{'type': k, 'value': v} for k, v in (flows_dict['static_items'] or {}).items()]}
        return {'raw_text': self.raw_text, 'map_image': self.map_image, 'entities': self.entities, 'inventory': [{'quantity': np.int32(v), 'type': k} for k, v in self.inventory.items() if v > 0], 'research': {'technologies': [{'name': tech.name, 'researched': int(tech.researched), 'enabled': int(tech.enabled), 'level': tech.level, 'research_unit_count': tech.research_unit_count, 'research_unit_energy': tech.research_unit_energy, 'prerequisites': tech.prerequisites, 'ingredients': []} for tech in self.research.technologies.values()], 'current_research': self.research.current_research if self.research.current_research is not None else 'None', 'research_progress': self.research.research_progress, 'research_queue': self.research.research_queue, 'progress': [{'name': name, 'value': value} for name, value in self.research.progress.items()]}, 'game_info': {'tick': self.game_info.tick, 'time': self.game_info.time, 'speed': self.game_info.speed}, 'score': self.score, 'flows': transformed_flows, 'task_verification': {'success': int(self.task_verification.success), 'meta': [{'key': k, 'value': json.dumps(v)} for k, v in self.task_verification.meta.items()]} if self.task_verification else {'success': 0, 'meta': []}, 'messages': [{'sender': msg.sender, 'content': msg.content, 'timestamp': msg.timestamp} for msg in self.messages], 'serialized_functions': self.serialized_functions, 'task_info': {'goal_description': self.task_info.goal_description if self.task_info else '', 'agent_instructions': self.task_info.agent_instructions or '' if self.task_info else '', 'task_key': self.task_info.task_key if self.task_info else '', 'trajectory_length': self.task_info.trajectory_length if self.task_info else 0}}

class FactorioGymEnv(gym.Env):
    """OpenAI Gym environment for Factorio"""

    def __init__(self, instance: FactorioInstance, task: Optional[TaskABC]=None, error_penalty: float=0.0, pause_after_action: bool=True, enable_vision: bool=False):
        super().__init__()
        self.instance = instance
        self.task = task
        self.error_penalty = error_penalty
        self.instance_speed = instance.get_speed()
        self.pause_after_action = pause_after_action
        self.enable_vision = enable_vision
        self.action_space = spaces.Dict({'agent_idx': spaces.Discrete(instance.num_agents), 'game_state': ObsSpaces.VERY_LONG_TEXT, 'code': ObsSpaces.LONG_TEXT})
        self.observation_space = spaces.Dict({'raw_text': ObsSpaces.LONG_TEXT, 'map_image': ObsSpaces.VERY_LONG_TEXT, 'entities': spaces.Sequence(ObsSpaces.LONG_TEXT), 'inventory': spaces.Sequence(ObsSpaces.ITEM_WITH_QUANTITY), 'research': ObsSpaces.RESEARCH, 'game_info': ObsSpaces.GAME_INFO, 'score': ObsSpaces.SCORE_FLOAT, 'flows': ObsSpaces.FLOWS, 'task_verification': ObsSpaces.TASK_VERIFICATION, 'messages': spaces.Sequence(ObsSpaces.MESSAGE), 'serialized_functions': spaces.Sequence(ObsSpaces.SERIALIZED_FUNCTION), 'task_info': ObsSpaces.TASK_INFO})
        self.current_state = None
        self.initial_score = 0
        self.last_observation = None
        self.last_message_timestamps = {i: 0.0 for i in range(instance.num_agents)}

    def get_observation(self, agent_idx: int=0, response: Optional[Response]=None) -> Observation:
        """Convert the current game state into a gym observation"""
        namespace = self.instance.namespaces[agent_idx]
        map_image = ''
        if self.enable_vision:
            map_image = namespace._render_simple().to_base64()
        entities = namespace.get_entities()
        entity_obs = [str(e) for e in entities]
        inventory_obs = namespace.inspect_inventory()
        research_obs = namespace._save_research_state()
        game_info = GameInfo(tick=self.instance.get_elapsed_ticks(), time=self.instance.get_elapsed_ticks() / 60, speed=self.instance.get_speed())
        if response:
            flows_obs = response.flows
        else:
            flows = namespace._get_production_stats()
            flows_obs = ProductionFlows.from_dict(flows)
        messages = namespace.get_messages()
        messages_obs = []
        latest_timestamp = self.last_message_timestamps[agent_idx]
        for msg in messages:
            if msg['timestamp'] > self.last_message_timestamps[agent_idx]:
                messages_obs.append(AgentMessage(sender=msg['sender'], content=msg['message'], timestamp=msg['timestamp']))
                latest_timestamp = max(latest_timestamp, msg['timestamp'])
        if messages_obs:
            self.last_message_timestamps[agent_idx] = latest_timestamp
        task_verification = None
        if response and hasattr(response, 'task'):
            task_verification = TaskResponse(success=response.task.success, meta=response.task.meta if hasattr(response.task, 'meta') else {})
        serialized_functions = []
        for func in namespace.get_functions():
            serialized_functions.append({'name': func.name, 'pickled_function': pickle.dumps(func).hex()})
        task_info = None
        if self.task:
            agent_instructions = None
            if self.task.agent_instructions:
                try:
                    agent_instructions = self.task.get_agent_instructions(agent_idx)
                except (IndexError, AttributeError):
                    agent_instructions = None
            task_info = TaskInfo(goal_description=self.task.goal_description, agent_instructions=agent_instructions, task_key=self.task.task_key, trajectory_length=self.task.trajectory_length)
        observation = Observation(raw_text=response.response if response else '', map_image=map_image, entities=entity_obs, inventory=inventory_obs, research=research_obs, game_info=game_info, score=response.score if response else 0.0, flows=flows_obs, task_verification=task_verification, messages=messages_obs, serialized_functions=serialized_functions, task_info=task_info)
        self.last_observation = observation
        return observation

    def step(self, action: Action) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment

        Args:
            action: Action object

        Returns:
            observation: The new observation as a dictionary matching the observation space
            reward: The reward for this step
            terminated: Whether the episode is done
            truncated: Whether the episode is truncated
            info: Additional information
        """
        assert isinstance(action, Action)
        agent_idx = action.agent_idx
        self.instance.set_speed_and_unpause(self.instance_speed)
        if action.game_state:
            self.reset_instance(GameState.parse_raw(action.game_state.to_raw()))
        namespace = self.instance.namespaces[agent_idx]
        start_production_flows = ProductionFlows.from_dict(namespace._get_production_stats())
        initial_score, eval_time, result = self.instance.eval(action.code, agent_idx=agent_idx, timeout=60)
        error_occurred = 'error' in result.lower() or 'exception: ' in result.lower()
        task_response = task_success = None
        terminated = truncated = False
        if self.task:
            task_success = self.task.verify(initial_score, self.instance, step_statistics={})
            task_response = self.task.enhance_response_with_task_output(result, task_success)
            terminated = task_success.success
        production_score, _ = namespace.score()
        if task_success and REWARD_OVERRIDE_KEY in task_success.meta:
            reward = task_success.meta[REWARD_OVERRIDE_KEY]
        else:
            reward = production_score - initial_score
        reward = float(reward) - self.error_penalty
        output_game_state = GameState.from_instance(self.instance)
        current_flows = ProductionFlows.from_dict(namespace._get_production_stats())
        achievements = calculate_achievements(start_production_flows, current_flows)
        response = Response(code=f'```python\n{action.code}\n```', created_at=datetime.datetime.now(), score=reward, achievements=achievements, step=0, ticks=self.instance.get_elapsed_ticks(), flows=start_production_flows.get_new_flows(current_flows), response=task_response if task_response else result, task=task_success if task_success else TaskResponse(success=False, meta={}), error=error_occurred, program_id=None)
        observation = self.get_observation(action.agent_idx, response)
        info = {'error_occurred': error_occurred, 'result': result, 'ticks': self.instance.get_elapsed_ticks(), 'flows': response.flows, 'agent_idx': agent_idx, 'last_message_timestamp': self.last_message_timestamps[agent_idx], 'task_verification': task_response, 'output_game_state': output_game_state, 'achievements': achievements, 'production_score': production_score}
        if self.pause_after_action:
            self.instance.pause()
        return (observation.to_dict(), reward, terminated, truncated, info)

    def reset_instance(self, state: Optional[GameState]=None) -> None:
        """Reset the Factorio instance to a given state or initial state.

        Args:
            state: Optional[GameState] to reset to. If None, resets to initial state.
        """
        self.instance.reset(state)

    def reset(self, options: Optional[Dict[str, Any]]=None, seed: Optional[int]=None) -> Dict[str, Any]:
        """Reset the environment to initial state

        Args:
            options: dict containing 'game_state' key with Optional[GameState] value to reset to
            seed: Not used
        """
        if options is None:
            options = {}
        game_state = options.get('game_state')
        self.reset_instance(game_state)
        self.initial_score, _ = self.instance.namespaces[0].score()
        self.last_observation = None
        self.last_message_timestamps = {i: 0.0 for i in range(self.instance.num_agents)}
        observation = self.get_observation(0).to_dict()
        return observation

    def close(self):
        """Clean up resources"""
        self.instance.cleanup()

class FactorioMCPRepository:
    """
    Version control system for Factorio game states and project files using Dulwich.

    Uses .claude-code as the working directory:
    - If current directory is named .claude-code, uses it directly
    - If .claude-code exists as subdirectory, uses that
    - Otherwise creates .claude-code as a new subdirectory
    """

    def __init__(self, instance: FactorioInstance):
        current_dir = os.getcwd()
        current_dirname = os.path.basename(current_dir)
        if current_dirname == '.claude-code':
            self.repo_dir = current_dir
        else:
            claude_code_path = os.path.join(current_dir, '.claude-code')
            if os.path.exists(claude_code_path) and os.path.isdir(claude_code_path):
                self.repo_dir = os.path.abspath(claude_code_path)
            else:
                self.repo_dir = os.path.abspath(claude_code_path)
                os.makedirs(self.repo_dir, exist_ok=True)
        instance_id = instance.tcp_port
        self.instance_repo_dir = os.path.join(self.repo_dir, f'instance_{instance_id}')
        os.makedirs(self.instance_repo_dir, exist_ok=True)
        if os.path.exists(os.path.join(self.instance_repo_dir, '.git')):
            self.repo = Repo(self.instance_repo_dir)
        else:
            self.repo = Repo.init(self.instance_repo_dir)
        self.branch = b'refs/heads/main'
        self.current_branch = 'main'
        self.instance = instance
        self.branches = {'main': None}
        self.tags = {}
        self.undo_stack = []
        if not self._has_commits():
            self._init_repo()
        else:
            self._load_existing_state()

    def _has_commits(self):
        """Check if the repository has any commits"""
        try:
            refs_dict = self.repo.refs.as_dict()
            return self.branch in refs_dict
        except Exception as e:
            print(f'Error checking for commits: {str(e)}')
            return False

    def _load_existing_state(self):
        """Load existing state from disk repository"""
        try:
            refs_dict = self.repo.refs.as_dict()
            for ref_name, ref_value in refs_dict.items():
                if ref_name.startswith(b'refs/heads/'):
                    branch_name = ref_name.decode('utf-8').replace('refs/heads/', '')
                    self.branches[branch_name] = ref_value.decode('utf-8')
                elif ref_name.startswith(b'refs/tags/'):
                    tag_name = ref_name.decode('utf-8').replace('refs/tags/', '')
                    self.tags[tag_name] = ref_value.decode('utf-8')
            if self.branch in refs_dict:
                history = self.get_history(max_count=100)
                self.undo_stack = [commit['id'] for commit in history]
        except Exception as e:
            print(f'Error loading existing repository state: {str(e)}')
            if not self.branches:
                self.branches = {'main': None}

    def _init_repo(self):
        """Initialize the repository with an empty commit"""
        initial_state = GameState.from_instance(self.instance)
        self.commit(initial_state, 'Initial state', None)

    def _make_blob(self, data: str) -> Tuple[bytes, Blob]:
        """Create a blob object from string data"""
        blob = Blob.from_string(data.encode('utf-8'))
        self.repo.object_store.add_object(blob)
        return (blob.id, blob)

    def _make_blob_from_bytes(self, data: bytes) -> Tuple[bytes, Blob]:
        """Create a blob object from bytes data"""
        blob = Blob.from_string(data)
        self.repo.object_store.add_object(blob)
        return (blob.id, blob)

    def _make_tree(self, entries: Dict[str, Tuple[int, bytes]]) -> Tuple[bytes, Tree]:
        """Create a tree object from a dictionary of entries"""
        tree = Tree()
        for name, (mode, blob_id) in entries.items():
            tree.add(name.encode('utf-8'), mode, blob_id)
        self.repo.object_store.add_object(tree)
        return (tree.id, tree)

    def _scan_working_directory(self) -> Dict[str, bytes]:
        """
        Scan the instance repo directory for files to include in commits.
        Returns a dict of relative_path -> file_content_bytes
        """
        files_to_track = {}
        ignore_patterns = ['.git', '__pycache__', '*.pyc', '.DS_Store', '*.swp', '*.swo', '.factorio_mcp_repo']
        for root, dirs, files in os.walk(self.instance_repo_dir):
            dirs[:] = [d for d in dirs if d not in ignore_patterns]
            for file_name in files:
                if any((file_name.endswith(pattern.replace('*', '')) for pattern in ignore_patterns if '*' in pattern)):
                    continue
                if file_name in ignore_patterns:
                    continue
                file_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(file_path, self.instance_repo_dir)
                if '.git' in rel_path:
                    continue
                try:
                    with open(file_path, 'rb') as f:
                        files_to_track[rel_path] = f.read()
                except Exception as e:
                    print(f'Warning: Could not read file {rel_path}: {str(e)}')
        return files_to_track

    def commit(self, state: GameState, message: str, policy: Optional[str]=None, include_files: bool=True) -> str:
        """
        Create a commit with the given state, message, and optionally tracked files

        Args:
            state: Game state to commit
            message: Commit message
            policy: Optional Python code that was executed
            include_files: Whether to include working directory files in the commit
        """
        state_id, state_blob = self._make_blob(state.to_raw())
        entries = {'gamestate.json': (33188, state_id)}
        if policy:
            policy_id, policy_blob = self._make_blob(policy)
            entries['policy.py'] = (33188, policy_id)
        if include_files:
            tracked_files = self._scan_working_directory()
            for file_path, file_content in tracked_files.items():
                if file_path in ['gamestate.json', 'policy.py']:
                    continue
                file_id, file_blob = self._make_blob_from_bytes(file_content)
                git_path = file_path.replace('\\', '/')
                entries[git_path] = (33188, file_id)
        tree_id, tree = self._make_tree(entries)
        commit = Commit()
        commit.tree = tree_id
        commit.author = commit.committer = b'FLE-Agent <agent@fle.local>'
        commit.commit_time = commit.author_time = int(time.time())
        commit.commit_timezone = commit.author_timezone = 0
        commit.message = message.encode('utf-8')
        try:
            refs_dict = self.repo.refs.as_dict()
            if self.branch in refs_dict:
                commit.parents = [refs_dict[self.branch]]
        except Exception as e:
            print(f'No parent commit found (this may be normal for first commit): {str(e)}')
            pass
        self.repo.object_store.add_object(commit)
        commit_id = commit.id.decode('utf-8')
        old_ref = None
        try:
            if self.branch in self.repo.refs:
                old_ref = self.repo.refs[self.branch]
        except Exception:
            pass
        self.repo.refs.set_if_equals(self.branch, old_ref, commit.id)
        self.branches[self.current_branch] = commit_id
        self.undo_stack.append(commit_id)
        try:
            self.repo.refs.pack_refs()
        except Exception as e:
            print(f'Warning: Could not pack refs: {str(e)}')
        return commit_id

    def restore_files_from_commit(self, commit_id: str) -> Dict[str, str]:
        """
        Restore files from a specific commit to the working directory.
        Returns a dict of files that were restored.
        """
        commit_id = commit_id.encode('utf-8') if isinstance(commit_id, str) else commit_id
        restored_files = {}
        try:
            try:
                commit = self.repo.object_store[commit_id]
            except KeyError:
                self.repo.object_store.add_objects_from_pack(commit_id)
                commit = self.repo.object_store[commit_id]
            tree = self.repo.object_store[commit.tree]
            for name, entry in tree.items():
                name = name.decode('utf-8')
                if name == 'gamestate.json':
                    continue
                mode, blob_id = entry
                blob = self.repo.object_store[blob_id]
                file_path = os.path.join(self.instance_repo_dir, name)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(blob.data)
                restored_files[name] = f'Restored ({len(blob.data)} bytes)'
        except Exception as e:
            print(f'Error restoring files from commit {commit_id}: {str(e)}')
        return restored_files

    def tag_commit(self, name: str, commit_id: Optional[str]=None) -> str:
        """Create a named tag for a commit (default: current HEAD)"""
        if commit_id is None:
            try:
                refs_dict = self.repo.refs.as_dict()
                if self.branch in refs_dict:
                    commit_id = refs_dict[self.branch].decode('utf-8')
                else:
                    raise ValueError('No current HEAD to tag')
            except Exception as e:
                raise ValueError(f'Error getting current HEAD: {str(e)}')
        self.tags[name] = commit_id
        tag_ref = f'refs/tags/{name}'.encode('utf-8')
        commit_id_bytes = commit_id.encode('utf-8')
        self.repo.refs.add_if_new(tag_ref, commit_id_bytes)
        try:
            self.repo.refs.pack_refs()
        except Exception as e:
            print(f'Warning: Could not pack refs for tag: {str(e)}')
        return commit_id

    def get_tag(self, name: str) -> Optional[str]:
        """Get commit ID for a named tag"""
        return self.tags.get(name)

    def list_tags(self) -> Dict[str, str]:
        """List all tags and their commit IDs"""
        return self.tags

    def checkout(self, ref: str) -> str:
        """
        Checkout a specific commit, branch, or tag.
        This changes internal state AND restores files to working directory.
        """
        refs_dict = self.repo.refs.as_dict()
        if ref in self.tags:
            ref = self.tags[ref]
            tag_ref = f'refs/tags/{ref}'.encode('utf-8')
            if tag_ref in refs_dict:
                commit_id = refs_dict[tag_ref]
            else:
                commit_id = ref.encode('utf-8') if isinstance(ref, str) else ref
        elif ref in self.branches:
            self.current_branch = ref
            self.branch = f'refs/heads/{ref}'.encode('utf-8')
            if self.branch in refs_dict:
                commit_id = refs_dict[self.branch]
            else:
                print(f'Warning: Branch {ref} not found in refs')
                return None
        else:
            commit_id = ref.encode('utf-8') if isinstance(ref, str) else ref
            self.current_branch = None
        try:
            if self.current_branch:
                self.repo.refs.set_symbolic_ref(b'HEAD', self.branch)
            else:
                self.repo.refs[b'HEAD'] = commit_id
            self.repo.refs.pack_refs()
            restored = self.restore_files_from_commit(commit_id)
            if restored:
                print(f'Restored {len(restored)} files from commit')
        except Exception as e:
            print(f'Warning: Error updating HEAD reference: {str(e)}')
        return commit_id.decode('utf-8') if isinstance(commit_id, bytes) else commit_id

    def apply_to_instance(self, commit_id: Optional[str]=None) -> bool:
        """Apply a specific commit to the game instance and restore files"""
        if commit_id is None:
            try:
                refs_dict = self.repo.refs.as_dict()
                head_value = refs_dict.get(b'HEAD')
                if head_value and head_value.startswith(b'ref: '):
                    ref_name = head_value[5:]
                    if ref_name in refs_dict:
                        commit_id = refs_dict[ref_name]
                    else:
                        raise ValueError(f'Symbolic ref {ref_name.decode('utf-8')} not found')
                else:
                    commit_id = head_value
                if not commit_id:
                    raise ValueError('No current commit to apply')
            except Exception as e:
                raise ValueError(f'Error getting current HEAD: {str(e)}')
        else:
            commit_id = commit_id.encode('utf-8') if isinstance(commit_id, str) else commit_id
        try:
            try:
                commit = self.repo.object_store[commit_id]
            except KeyError:
                self.repo.object_store.add_objects_from_pack(commit_id)
                commit = self.repo.object_store[commit_id]
            tree = self.repo.object_store[commit.tree]
            if b'gamestate.json' in tree:
                state_id = tree[b'gamestate.json'][1]
                state_blob = self.repo.object_store[state_id]
                state_json = state_blob.data.decode('utf-8')
                state = GameState.parse_raw(state_json)
                self.instance.reset(game_state=state)
                self.restore_files_from_commit(commit_id)
                print('Instance reset and files restored')
                return True
        except Exception as e:
            print(f'Error applying commit {commit_id}: {str(e)}')
        return False

    def undo(self) -> Optional[str]:
        """
        Undo to the previous commit.
        Returns the commit ID that was restored, or None if no more history.
        """
        if len(self.undo_stack) <= 1:
            return None
        self.undo_stack.pop()
        if not self.undo_stack:
            return None
        prev_commit_id = self.undo_stack[-1]
        commit_id_bytes = prev_commit_id.encode('utf-8')
        try:
            old_ref = None
            if self.branch in self.repo.refs:
                old_ref = self.repo.refs[self.branch]
            self.repo.refs.set_if_equals(self.branch, old_ref, commit_id_bytes)
            if self.current_branch:
                self.branches[self.current_branch] = prev_commit_id
            if self.current_branch:
                self.repo.refs.set_symbolic_ref(b'HEAD', self.branch)
            else:
                self.repo.refs[b'HEAD'] = commit_id_bytes
            self.repo.refs.pack_refs()
        except Exception as e:
            print(f'Warning: Error updating references during undo: {str(e)}')
        return prev_commit_id

    def get_policy(self, commit_id: str) -> Optional[str]:
        """Get the policy associated with a commit"""
        commit_id = commit_id.encode('utf-8') if isinstance(commit_id, str) else commit_id
        try:
            try:
                commit = self.repo.object_store[commit_id]
            except KeyError:
                self.repo.do_pack()
                commit = self.repo.object_store[commit_id]
            try:
                tree = self.repo.object_store[commit.tree]
            except KeyError:
                self.repo.do_pack()
                tree = self.repo.object_store[commit.tree]
            if b'policy.py' in tree:
                policy_id = tree[b'policy.py'][1]
                try:
                    policy_blob = self.repo.object_store[policy_id]
                except KeyError:
                    self.repo.do_pack()
                    policy_blob = self.repo.object_store[policy_id]
                return policy_blob.data.decode('utf-8')
        except KeyError:
            pass
        except Exception as e:
            print(f'Error getting policy: {str(e)}')
        return None

    def get_file_from_commit(self, commit_id: str, file_path: str) -> Optional[str]:
        """Get a specific file from a commit"""
        commit_id = commit_id.encode('utf-8') if isinstance(commit_id, str) else commit_id
        file_path = file_path.replace('\\', '/')
        try:
            commit = self.repo.object_store[commit_id]
            tree = self.repo.object_store[commit.tree]
            file_path_bytes = file_path.encode('utf-8')
            if file_path_bytes in tree:
                mode, blob_id = tree[file_path_bytes]
                blob = self.repo.object_store[blob_id]
                return blob.data.decode('utf-8')
        except Exception as e:
            print(f'Error getting file {file_path} from commit: {str(e)}')
        return None

    def list_files_in_commit(self, commit_id: str) -> List[str]:
        """List all files tracked in a specific commit"""
        commit_id = commit_id.encode('utf-8') if isinstance(commit_id, str) else commit_id
        files = []
        try:
            commit = self.repo.object_store[commit_id]
            tree = self.repo.object_store[commit.tree]
            for name, entry in tree.items():
                files.append(name.decode('utf-8'))
        except Exception as e:
            print(f'Error listing files in commit: {str(e)}')
        return files

    def get_history(self, max_count=10) -> List[Dict[str, Any]]:
        """Get commit history"""
        history = []
        try:
            if self.branch not in self.repo.refs:
                return history
            commit_id = self.repo.refs[self.branch]
            while commit_id and len(history) < max_count:
                try:
                    try:
                        commit = self.repo.object_store[commit_id]
                    except KeyError:
                        self.repo.do_pack()
                        commit = self.repo.object_store[commit_id]
                    tree = self.repo.object_store[commit.tree]
                    file_count = len(list(tree.items()))
                    history.append({'id': commit_id.decode('utf-8') if isinstance(commit_id, bytes) else commit_id, 'message': commit.message.decode('utf-8'), 'timestamp': commit.commit_time, 'has_policy': self._has_policy(commit.tree), 'file_count': file_count})
                    if commit.parents:
                        commit_id = commit.parents[0]
                    else:
                        break
                except KeyError:
                    break
                except Exception as e:
                    print(f'Error processing commit {commit_id}: {str(e)}')
                    break
        except Exception as e:
            print(f'Error getting history: {str(e)}')
        return history

    def _has_policy(self, tree_id):
        """Check if a tree contains a policy file"""
        try:
            try:
                tree = self.repo.object_store[tree_id]
            except KeyError:
                self.repo.do_pack()
                tree = self.repo.object_store[tree_id]
            return b'policy.py' in tree
        except Exception:
            return False

    def diff_policies(self, commit_id1: str, commit_id2: str) -> Dict[str, Any]:
        """
        Compare policies between two commits.
        Returns information about the differences.
        """
        try:
            policy1 = self.get_policy(commit_id1)
            policy2 = self.get_policy(commit_id2)
            if policy1 is None and policy2 is None:
                return {'status': 'no_policies', 'message': 'Neither commit has a policy'}
            if policy1 is None:
                return {'status': 'added', 'message': 'Policy added in second commit', 'policy': policy2}
            if policy2 is None:
                return {'status': 'removed', 'message': 'Policy removed in second commit', 'policy': policy1}
            import difflib
            diff = list(difflib.unified_diff(policy1.splitlines(keepends=True), policy2.splitlines(keepends=True), fromfile=f'policy-{commit_id1[:8]}.py', tofile=f'policy-{commit_id2[:8]}.py'))
            return {'status': 'modified', 'message': 'Policy modified between commits', 'diff': ''.join(diff), 'policy1': policy1, 'policy2': policy2}
        except Exception as e:
            return {'status': 'error', 'message': f'Error comparing policies: {str(e)}'}

class UnboundedThroughputTask(TaskABC):

    def __init__(self, trajectory_length, goal_description: str, task_key: str, throughput_entity: Entity, holdout_wait_period: int, pre_holdout_wait_period: int=0, show_number_of_steps_left_in_prompt=False, use_populated_inventory=True, unlock_all_research=True, agent_instructions: Optional[List[str]]=None) -> None:
        goal_description += f'\n{INSTRUCTIONS}'
        if show_number_of_steps_left_in_prompt:
            goal_description += f'\n\nIn total you have {trajectory_length} steps to build your factory'
        starting_inventory = LAB_PLAY_POPULATED_STARTING_INVENTORY if use_populated_inventory else {}
        super().__init__(trajectory_length, starting_inventory=starting_inventory, goal_description=goal_description, task_key=task_key, all_technology_reserached=unlock_all_research, agent_instructions=agent_instructions)
        self.throughput_entity = throughput_entity
        self.holdout_wait_period = holdout_wait_period
        self.starting_game_state = None
        self.pre_holdout_wait_period = pre_holdout_wait_period
        self.show_number_of_steps_left_in_prompt = show_number_of_steps_left_in_prompt

    def verify(self, score: float, instance: FactorioInstance, step_statistics: Dict) -> TaskResponse:
        max_achieved_throughput = 0
        max_achievements = None
        while True:
            result_list, result, error, achievements = eval_program_with_achievements(program=f'sleep({self.holdout_wait_period})', instance=instance)
            if max_achievements is None:
                max_achievements = achievements
            dynamic_achievements = achievements['dynamic']
            target_throughput = dynamic_achievements.get(self.throughput_entity, 0)
            if target_throughput > max_achieved_throughput:
                max_achieved_throughput = target_throughput
                max_achievements = achievements
            else:
                break
        return TaskResponse(success=False, meta={'achievements': max_achievements})

    def _to_dict(self) -> Dict[str, Any]:
        return {'task': self.goal_description, 'throughput_entity': self.throughput_entity, 'trajectory_length': self.trajectory_length, 'starting_inventory': self.starting_inventory, 'initial_state': self.starting_game_state.to_raw() if self.starting_game_state else None, 'show_number_of_steps_left_in_prompt': self.show_number_of_steps_left_in_prompt}

    def setup_instance(self, instance):
        """Code to provision the task environment"""
        pass

    def enhance_response_with_task_output(self, response: str, task_response: TaskResponse) -> str:
        task_throughputs = task_response.meta.get('achievements', None)
        number_of_steps_left = task_response.meta.get('nr_of_steps_left', None)
        if task_throughputs:
            response += f'\n\nHere is the current throughput of your factory: {task_throughputs['dynamic']} created per 60 seconds'
        if self.show_number_of_steps_left_in_prompt and number_of_steps_left:
            response += f'\n\nYou have: {number_of_steps_left} steps left to build or expand your factory'
        return response

class ThroughputTask(TaskABC):

    def __init__(self, trajectory_length, goal_description: str, task_key: str, throughput_entity: Entity, quota: int, holdout_wait_period: int, pre_holdout_wait_period: int=0, agent_instructions: Optional[List[str]]=None):
        goal_description += f'\n{INSTRUCTIONS}'
        super().__init__(trajectory_length, starting_inventory=LAB_PLAY_POPULATED_STARTING_INVENTORY, goal_description=goal_description, task_key=task_key, all_technology_reserached=True, agent_instructions=agent_instructions)
        self.throughput_entity = throughput_entity
        self.quota = quota
        self.holdout_wait_period = holdout_wait_period
        self.starting_game_state = None
        self.pre_holdout_wait_period = pre_holdout_wait_period
        self.throughput_key = f'{throughput_entity} achieved throughput per {holdout_wait_period} seconds'

    def verify(self, score: float, instance: FactorioInstance, step_statistics: Dict) -> TaskResponse:
        max_achieved_throughput = 0
        max_achievements = None
        while True:
            result_list, result, error, achievements = eval_program_with_achievements(program=f'sleep({self.holdout_wait_period})', instance=instance)
            if max_achievements is None:
                max_achievements = achievements
            dynamic_achievements = achievements['dynamic']
            target_throughput = dynamic_achievements.get(self.throughput_entity, 0)
            if target_throughput > max_achieved_throughput:
                max_achieved_throughput = target_throughput
                max_achievements = achievements
            else:
                break
        return TaskResponse(success=max_achieved_throughput >= self.quota, meta={self.throughput_key: max_achieved_throughput, REWARD_OVERRIDE_KEY: max_achieved_throughput})

    def _to_dict(self) -> Dict[str, Any]:
        return {'task': self.goal_description, 'throughput_entity': self.throughput_entity, 'quota': self.quota, 'trajectory_length': self.trajectory_length, 'starting_inventory': self.starting_inventory, 'initial_state': self.starting_game_state.to_raw() if self.starting_game_state else None}

    def setup_instance(self, instance):
        """Code to provision the task environment"""
        pass

    def enhance_response_with_task_output(self, response: str, task_response: TaskResponse) -> str:
        task_throughput = task_response.meta.get(self.throughput_key, None)
        if task_throughput:
            response += f'\n\nThe current throughput of your factory is {task_throughput} of {self.throughput_entity} created per 60 seconds'
        return response

def make_minimal_observation(**kwargs):
    return Observation(raw_text=kwargs.get('raw_text', 'Test output'), entities=kwargs.get('entities', ['Entity(name=burner-mining-drill, position=(10,5))']), inventory=kwargs.get('inventory', {'iron-ore': 100, 'coal': 50}), research=kwargs.get('research', ResearchState(technologies={'automation': TechnologyState(name='automation', researched=True, enabled=True, level=1, research_unit_count=10, research_unit_energy=30.0, prerequisites=['logistics'], ingredients=[{'item': 'iron-gear-wheel', 'amount': 1}])}, current_research='automation', research_progress=0.5, research_queue=['automation'], progress={'automation': 0.5})), game_info=kwargs.get('game_info', GameInfo(tick=123, time=2.0, speed=1.0)), score=kwargs.get('score', 42.0), flows=kwargs.get('flows', ProductionFlows(input={'coal': 1.5}, output={'iron-ore': 0.75}, crafted=[{'type': 'iron-gear-wheel', 'count': 2}], harvested={'iron-ore': 10.0}, price_list={'iron-ore': 3.0}, static_items={'iron-ore': 100.0})), task_verification=kwargs.get('task_verification', TaskResponse(success=False, meta={'criteria': 'Place mining drill'})), messages=kwargs.get('messages', [AgentMessage(sender='1', content='Need more iron plates', timestamp=1.0)]), serialized_functions=kwargs.get('serialized_functions', [{'name': 'dummy_func', 'pickled_function': ''}]))

def test_inventory_formatting():
    make_minimal_observation()
    formatter = BasicObservationFormatter()
    formatted = formatter.format_inventory([{'type': 'iron-ore', 'quantity': 100}, {'type': 'coal', 'quantity': 50}])
    assert 'iron-ore' in formatted and 'coal' in formatted
    assert formatted.startswith('### Inventory')

def test_entities_formatting():
    obs = make_minimal_observation()
    formatter = BasicObservationFormatter()
    formatted = formatter.format_entities(obs.entities)
    assert 'burner-mining-drill' in formatted or 'Entity' in formatted
    assert formatted.startswith('### Entities')

def test_flows_formatting():
    obs = make_minimal_observation()
    formatter = BasicObservationFormatter()
    flows_dict = obs.flows.to_dict()
    flows_obs = {'input': [{'type': k, 'rate': v} for k, v in flows_dict['input'].items()], 'output': [{'type': k, 'rate': v} for k, v in flows_dict['output'].items()], 'crafted': flows_dict['crafted'], 'harvested': [{'type': k, 'amount': v} for k, v in flows_dict['harvested'].items()], 'price_list': [{'type': k, 'price': v} for k, v in (flows_dict['price_list'] or {}).items()], 'static_items': [{'type': k, 'value': v} for k, v in (flows_dict['static_items'] or {}).items()]}
    formatted = formatter.format_flows(flows_obs)
    assert 'Production Flows' in formatted
    assert 'Inputs' in formatted and 'Outputs' in formatted

def test_research_formatting():
    make_minimal_observation()
    formatter = BasicObservationFormatter()
    research_dict = {'technologies': {'automation': {'name': 'automation', 'researched': 1, 'enabled': 1, 'level': 1, 'research_unit_count': 10, 'research_unit_energy': 30.0, 'prerequisites': ['logistics'], 'ingredients': [{'item': 'iron-gear-wheel', 'amount': 1}]}}, 'current_research': 'automation', 'research_progress': 0.5, 'research_queue': ['automation'], 'progress': [{'name': 'automation', 'value': 0.5}]}
    formatted = formatter.format_research(research_dict)
    assert 'Research' in formatted
    assert 'automation' in formatted

def test_task_formatting():
    formatter = BasicObservationFormatter()
    task = {'success': False, 'meta': [{'key': 'criteria', 'value': 'Place mining drill'}]}
    formatted = formatter.format_task(task)
    assert 'Task Status' in formatted
    assert 'IN PROGRESS' in formatted

def test_messages_formatting():
    formatter = BasicObservationFormatter()
    messages = [{'sender': '1', 'content': 'Need more iron plates', 'timestamp': 1.0}, {'sender': '2', 'content': "I'll help with that", 'timestamp': 2.0}]
    formatted = formatter.format_messages(messages, last_timestamp=0.0)
    assert 'Messages' in formatted
    assert 'Agent 1' in formatted and 'Agent 2' in formatted

def test_functions_formatting():
    formatter = BasicObservationFormatter()
    functions = [{'name': 'dummy_func', 'pickled_function': ''}]
    formatted = formatter.format_functions(functions)
    assert 'Available Functions' in formatted
    assert 'dummy_func' in formatted

def test_raw_text_formatting():
    formatter = BasicObservationFormatter()
    formatted = formatter.format_raw_text('Some output')
    assert 'Raw Output' in formatted
    assert 'Some output' in formatted

def test_full_format():
    obs = make_minimal_observation()
    formatter = BasicObservationFormatter()
    formatted = formatter.format(obs)
    assert 'Inventory' in formatted.raw_str
    assert 'Entities' in formatted.raw_str
    assert 'Production Flows' in formatted.raw_str
    assert 'Research' in formatted.raw_str
    assert 'Task Status' in formatted.raw_str or formatted.task_str == ''
    assert 'Messages' in formatted.raw_str
    assert 'Available Functions' in formatted.raw_str
    assert 'Raw Output' in formatted.raw_str

class TestAchievements(unittest.TestCase):

    def test_achievements(self):
        instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, fast=True, inventory={})
        instance.set_speed(10)
        test_string_1 = 'pos = nearest(Resource.Stone)\nmove_to(pos)\nharvest_resource(pos, 10)\ncraft_item(Prototype.StoneFurnace, 1)\npos = nearest(Resource.Coal)\nmove_to(pos)\nharvest_resource(pos, 10)\npos = nearest(Resource.IronOre)\nmove_to(pos)\nharvest_resource(pos, 10)\npos = Position(x = 0, y = 0)\nmove_to(pos)\nfurnace = place_entity(Prototype.StoneFurnace, position = pos)\ninsert_item(Prototype.IronOre, furnace, 5)\ninsert_item(Prototype.Coal, furnace, 5)\nsleep(16)\nextract_item(Prototype.IronPlate, furnace.position, 10)'
        _, _, _, achievements = eval_program_with_achievements(instance, test_string_1)
        ground_truth_achievement = {'static': {'stone-furnace': 1, 'coal': 10, 'stone': 10, 'iron-ore': 10}, 'dynamic': {'iron-plate': 5}}
        assert achievements == ground_truth_achievement
        instance.reset()
        test_string = 'pos = nearest(Resource.Stone)\nmove_to(pos)\nharvest_resource(pos, 10)\ncraft_item(Prototype.StoneFurnace, 1)\npos = nearest(Resource.Coal)\nmove_to(pos)\nharvest_resource(pos, 10)\npos = nearest(Resource.CopperOre)\nmove_to(pos)\nharvest_resource(pos, 10)\npos = Position(x = 0, y = 0)\nmove_to(pos)\nfurnace = place_entity(Prototype.StoneFurnace, position = pos)\ninsert_item(Prototype.CopperOre, furnace, 5)\ninsert_item(Prototype.Coal, furnace, 5)\nsleep(16)'
        _, _, _, achievements = eval_program_with_achievements(instance, test_string)
        ground_truth_achievement = {'static': {'stone-furnace': 1, 'coal': 10, 'stone': 10, 'copper-ore': 10}, 'dynamic': {'copper-plate': 5}}
        assert achievements == ground_truth_achievement
        instance.reset()
        test_string = 'pos = nearest(Resource.Stone)\nmove_to(pos)\nharvest_resource(pos, 10)\ncraft_item(Prototype.StoneFurnace, 1)\npos = nearest(Resource.Coal)\nmove_to(pos)\nharvest_resource(pos, 10)\npos = nearest(Resource.CopperOre)\nmove_to(pos)\nharvest_resource(pos, 10)\npos = Position(x = 0, y = 0)\nmove_to(pos)\nfurnace = place_entity(Prototype.StoneFurnace, position = pos)\ninsert_item(Prototype.CopperOre, furnace, 5)\ninsert_item(Prototype.Coal, furnace, 5)\nsleep(16)'
        _, _, _, achievements = eval_program_with_achievements(instance, test_string)
        ground_truth_achievement = {'static': {'stone-furnace': 1, 'coal': 10, 'stone': 10, 'copper-ore': 10}, 'dynamic': {'copper-plate': 5}}
        assert achievements == ground_truth_achievement

