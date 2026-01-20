# Cluster 39

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

def _lua2python(command, response, *parameters, trace=False, start=0):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        if not response:
            return (None, timer() - start)
        try:
            if response.strip().startswith('{') and response.strip().endswith('}'):
                output = lua.decode(response)
            else:
                splitted = response.split('\n')[-1]
                if '[string' in splitted:
                    splitted = re.sub('\\[string[^\\]]*\\]', '', splitted)
                output = lua.decode(splitted)
            if isinstance(output, dict) and 'b' in output:
                output['b'] = _remove_numerical_keys(output['b'])
            return (output, timer() - start)
        except Exception as e:
            if trace:
                print(f'Parsing error: {str(e)}')
            return (None, timer() - start)

def _remove_numerical_keys(dictionary):
    pruned = {}
    if not isinstance(dictionary, dict):
        return dictionary
    parts = []
    for key, value in dictionary.items():
        if isinstance(key, int):
            if isinstance(value, dict):
                parts.append(_remove_numerical_keys(value))
            elif isinstance(value, str):
                parts.append(value.replace('!!', '"').strip())
            else:
                parts.append(value)
        else:
            pruned[key] = value
    if parts:
        pruned = parts
    return pruned

@deprecated("Doesn't handle nested structures that well")
def _lua2python_old(command, response, *parameters, trace=False, start=0):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        if trace:
            print(command, parameters, response)
        if response:
            if trace:
                print(f'success: {command}')
            end = timer()
            if response[0] != '{':
                splitted = response.split('\n')[-1]
                if '[string' in splitted:
                    a, b = splitted.split('[string')
                    splitted = a + '["' + b.replace('"', '!!')
                    splitted = re.sub(',\\s*}\\s*$', '', splitted) + '"]}'
                try:
                    output = lua.decode(splitted)
                except Exception:
                    output = None
            else:
                try:
                    output = lua.decode(response)
                except Exception:
                    output = None
            if trace:
                print('{hbar}\nCOMMAND: {command}\nPARAMETERS: {parameters}\n\n{response}\n\nOUTPUT:{output}'.format(hbar='-' * 100, command=command, parameters=parameters, response=response, output=output))
            captured_output = stdout.getvalue()
            _check_output_for_errors(command, response, captured_output)
            if isinstance(output, dict) and 'b' in output:
                pruned = _remove_numerical_keys(output['b'])
                output['b'] = pruned
            return (output, end - start)
        else:
            if trace:
                print(f'failure: {command} \t')
            end = timer()
            try:
                return (lua.decode(response), end - start)
            except Exception:
                captured_output = stdout.getvalue()
                _check_output_for_errors(command, response, captured_output)
                return (None, end - start)

def _check_output_for_errors(command, response, output):
    """Check captured stdout for known Lua parsing errors"""
    ERRORS = {'unexp_end_string': 'Unexpected end of string while parsing Lua string.', 'unexp_end_table': 'Unexpected end of table while parsing Lua string.', 'mfnumber_minus': 'Malformed number (no digits after initial minus).', 'mfnumber_dec_point': 'Malformed number (no digits after decimal point).', 'mfnumber_sci': 'Malformed number (bad scientific format).'}
    for error_key, error_msg in ERRORS.items():
        if error_msg in output:
            raise LuaConversionError(f"Lua parsing error: {error_msg} for command:\n'{command}' with response:\n'{response}'")

class Controller:

    def __init__(self, lua_script_manager: 'LuaScriptManager', game_state: 'FactorioNamespace', *args, **kwargs):
        self.connection = lua_script_manager
        self.game_state = game_state
        self.name = self.camel_to_snake(self.__class__.__name__)
        self.lua_script_manager = lua_script_manager
        self.player_index = game_state.agent_index + 1

    def clean_response(self, response):

        def is_lua_list(d):
            """Check if dictionary represents a Lua-style list (keys are consecutive numbers from 1)"""
            if not isinstance(d, dict) or not d:
                return False
            keys = set((str(k) for k in d.keys()))
            return all((str(i) in keys for i in range(1, len(d) + 1)))

        def clean_value(value):
            """Recursively clean a value"""
            if isinstance(value, dict):
                if is_lua_list(value):
                    sorted_items = sorted(value.items(), key=lambda x: int(str(x[0])))
                    return [clean_value(v) for k, v in sorted_items]
                if any((isinstance(k, int) for k in value.keys())) and all((isinstance(v, dict) and 'name' in v and ('count' in v) for v in value.values())):
                    cleaned_dict = {}
                    for v in value.values():
                        cleaned_dict[v['name']] = v['count']
                    return cleaned_dict
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [clean_value(v) for v in value]
            return value
        cleaned_response = {}
        if not hasattr(response, 'items'):
            pass
        for key, value in response.items():
            if key == 'direction' and isinstance(value, str):
                cleaned_response[key] = Direction.from_string(value)
            elif not value and key in ('warnings', 'input_connection_points', 'output_connection_points'):
                cleaned_response[key] = []
            else:
                cleaned_response[key] = clean_value(value)
        return cleaned_response

    def parse_lua_dict(self, d):
        if isinstance(d, (int, str, float)):
            return d
        if isinstance(d, list):
            return [self.parse_lua_dict(item) for item in d]
        if isinstance(d, dict) and all((isinstance(k, int) for k in d.keys())):
            return [self.parse_lua_dict(d[k]) for k in sorted(d.keys())]
        else:
            new_dict = {}
            last_key = None
            for key in d.keys():
                if isinstance(key, int):
                    if last_key is not None and isinstance(d[key], str):
                        new_dict[last_key] += '-' + d[key]
                else:
                    last_key = key
                    if isinstance(d[key], dict):
                        new_dict[key] = self.parse_lua_dict(d[key])
                    else:
                        new_dict[key] = d[key]
            return new_dict

    def camel_to_snake(self, camel_str):
        snake_str = ''
        for index, char in enumerate(camel_str):
            if char.isupper():
                if index != 0:
                    snake_str += '_'
                snake_str += char.lower()
            else:
                snake_str += char
        return snake_str

    def _get_command(self, command, parameters=[], measured=True):
        if command in self.script_dict:
            script = f'{COMMAND} ' + self.script_dict[command]
            for index in range(len(parameters)):
                script = script.replace(f'arg{index + 1}', lua.encode(parameters[index]))
        else:
            script = command
        return script

    def execute(self, *args) -> Tuple[Dict, Any]:
        try:
            start = time.time()
            parameters = [lua.encode(arg) for arg in args]
            invocation = f'pcall(global.actions.{self.name}{(', ' if parameters else '') + ','.join(parameters)})'
            wrapped = f'{COMMAND} a, b = {invocation}; rcon.print(dump({{a=a, b=b}}))'
            lua_response = self.connection.rcon_client.send_command(wrapped)
            parsed, elapsed = _lua2python(invocation, lua_response, start=start)
            if parsed is None:
                return ({}, lua_response)
            if not parsed.get('a') and 'b' in parsed and isinstance(parsed['b'], str):
                parts = lua_response.split('["b"] = ')
                if len(parts) > 1:
                    msg = parts[1]
                    msg = msg.rstrip()
                    if msg.endswith('}'):
                        msg = msg[:-2] if len(msg) >= 2 else msg
                    msg = msg.replace('!!', '"').strip()
                    return (msg, lua_response)
                return (parsed['b'], lua_response)
            return (parsed.get('b', {}), lua_response)
        except Exception:
            return ({}, -1)

    def execute2(self, *args) -> Tuple[Dict, Any]:
        lua_response = ''
        try:
            start = time.time()
            parameters = [lua.encode(arg) for arg in args]
            invocation = f'pcall(global.actions.{self.name}{(', ' if parameters else '') + ','.join(parameters)})'
            wrapped = f'{COMMAND} a, b = {invocation}; rcon.print(dump({{a=a, b=b}}))'
            lua_response = self.connection.rcon_client.send_command(wrapped)
            parsed, elapsed = _lua2python(invocation, lua_response, start=start)
            if not parsed['a'] and 'b' in parsed and isinstance(parsed['b'], str):
                parts = lua_response.split('["b"] = ')
                parts[1] = f'{parts[1][:-2]}' if parts[1][-1] == '}' else parts[1]
                parsed['b'] = parts[1].replace('!!', '"')
            if 'b' not in parsed:
                return ({}, elapsed)
        except ParseError as e:
            try:
                parts = lua_response.split('["b"] = ')
                return (parts[1][:-2], -1)
            except IndexError:
                return (e.args[0], -1)
        except TypeError:
            return (lua_response, -1)
        except Exception:
            return (lua_response, -1)
        return (parsed['b'], elapsed)

    def send(self, command, *parameters, trace=False) -> List[str]:
        start = timer()
        script = self._get_command(command, parameters=list(parameters), measured=False)
        lua_response = self.connection.send_command(script)
        return _lua2python(command, lua_response, start=start)

class TestProductionStats(unittest.TestCase):

    def test_production_stats(self):
        inventory = {'iron-plate': 50, 'coal': 50, 'copper-plate': 50, 'iron-chest': 2, 'burner-mining-drill': 3, 'electric-mining-drill': 1, 'assembling-machine-1': 1, 'stone-furnace': 9, 'transport-belt': 50, 'boiler': 1, 'burner-inserter': 32, 'pipe': 15, 'steam-engine': 1, 'small-electric-pole': 10}
        instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, fast=True, inventory=inventory)
        instance.namespace.move_to(instance.namespace.nearest(Resource.IronOre))
        instance.namespace.harvest_resource(instance.namespace.nearest(Resource.IronOre), quantity=10)
        result = instance.namespace._production_stats()
        assert result['input']['iron-ore'] == 10
        result = instance.namespace._production_stats()
        assert result['input']['iron-ore'] == 0

    def test_lua2python(self):
        result = _lua2python('pcall(global.actions.get_production_stats, 1)', '{ ["a"] = true,["b"] = {\n ["iron-ore"] = 10\n},}')
        assert result

def test_lua_2_python():
    lua_response = '{ ["a"] = false,["b"] = ["string global"],}'
    command = 'pcall(global.actions.move_to,1,11.5,20)'
    response, timing = _lua2python(command, lua_response)
    assert response == {'a': False, 'b': 'string global', 2: ']'}

