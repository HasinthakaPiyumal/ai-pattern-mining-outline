# Cluster 15

def register_all_environments() -> None:
    """Register all discovered environments with gym"""
    _registry.discover_tasks()

def get_environment_info(task_key: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific environment"""
    spec = _registry.get_environment_spec(task_key)
    if spec is None:
        return None
    return asdict(spec)

def list_available_environments() -> List[str]:
    """List all available gym environment IDs"""
    return _registry.list_environments()

def print_environment_list(env_ids: List[str], detailed: bool=False):
    """Print a formatted list of environments"""
    if not env_ids:
        print('No environments found.')
        return
    print(f'Found {len(env_ids)} Factorio gym environments:\n')
    for i, env_id in enumerate(env_ids, 1):
        info = get_environment_info(env_id)
        if not info:
            continue
        print(f'{i:2d}. {env_id}')
        print(f'     Description: {info['description']}')
        if detailed:
            print(f'     Task Key: {info['task_key']}')
            print(f'     Config Path: {info['task_config_path']}')
            print(f'     Agents: {info['num_agents']}')
        print()

def search_environments(search_term: str, detailed: bool=False) -> List[str]:
    """Search for environments containing the given term"""
    all_env_ids = list_available_environments()
    matching_env_ids = []
    search_term_lower = search_term.lower()
    for env_id in all_env_ids:
        info = get_environment_info(env_id)
        if not info:
            continue
        if search_term_lower in env_id.lower() or search_term_lower in info['task_key'].lower() or search_term_lower in info['description'].lower():
            matching_env_ids.append(env_id)
    return matching_env_ids

def run_interactive_examples():
    """Run interactive examples demonstrating the gym registry functionality"""
    print('=== Factorio Gym Registry Interactive Examples ===\n')
    print('1. Available Environments:')
    env_ids = list_available_environments()
    for env_id in env_ids[:3]:
        info = get_environment_info(env_id)
        print(f'   {env_id}')
        print(f'     Description: {info['description']}')
        print(f'     Task Key: {info['task_key']}')
        print(f'     Agents: {info['num_agents']}')
        print()
    if len(env_ids) > 3:
        print(f'   ... and {len(env_ids) - 3} more environments')
        print()
    if env_ids:
        example_env_id = env_ids[0]
        print(f'2. Creating environment: {example_env_id}')
        try:
            env = gym.make(example_env_id)
            print('   Resetting environment...')
            obs, info = env.reset(options={'game_state': None})
            print(f'   Initial observation keys: {list(obs.keys())}')
            print('   Taking a simple action...')
            current_game_state = GameState.from_instance(env.instance)
            action = Action(agent_idx=0, game_state=current_game_state, code='print("Hello from Factorio Gym!")')
            obs, reward, terminated, truncated, info = env.step(action)
            print(f'   Reward: {reward}')
            print(f'   Terminated: {terminated}')
            print(f'   Truncated: {truncated}')
            print(f'   Info keys: {list(info.keys())}')
            env.close()
            print('   Environment closed successfully')
        except Exception as e:
            print(f'   Error creating/using environment: {e}')
            print('   Note: This might be due to missing Factorio containers or other setup requirements')
    print('\n3. Usage Examples:')
    print('   # List all environments')
    print('   from gym_env.registry import list_available_environments')
    print('   env_ids = list_available_environments()')
    print()
    print('   # Create an environment')
    print('   import gym')
    print("   env = gym.make('Factorio-iron_ore_throughput_16-v0')")
    print()
    print('   # Use the environment')
    print("   obs, info = env.reset(options={'game_state': None})")
    print('   action = Action(agent_idx=0, game_state=\'\', code=\'print("Hello Factorio!")\')')
    print('   obs, reward, terminated, truncated, info = env.step(action)')
    print('   env.close()')

def run_command_line_mode():
    """Run in command-line mode for listing and searching environments"""
    parser = argparse.ArgumentParser(description='Factorio Gym Registry - Environment Explorer and Examples', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  python example_usage.py                    # Run interactive examples\n  python example_usage.py --list            # List all environments\n  python example_usage.py --detail          # Show detailed information\n  python example_usage.py --search iron     # Search for iron-related tasks\n  python example_usage.py --search science  # Search for science pack tasks\n  python example_usage.py --gym-format      # Output in gym.make() format\n        ')
    parser.add_argument('--list', '-l', action='store_true', help='List all available environments')
    parser.add_argument('--detail', '-d', action='store_true', help='Show detailed information for each environment')
    parser.add_argument('--search', '-s', type=str, help='Search for environments containing the given term')
    parser.add_argument('--gym-format', '-g', action='store_true', help='Output in gym.make() format for easy copy-paste')
    args = parser.parse_args()
    if not any([args.list, args.detail, args.search, args.gym_format]):
        run_interactive_examples()
        return
    if args.search:
        env_ids = search_environments(args.search, args.detail)
        if not env_ids:
            print(f"No environments found matching '{args.search}'")
            return
        print(f"Environments matching '{args.search}':\n")
    else:
        env_ids = list_available_environments()
    if args.gym_format:
        print('# Available Factorio gym environments:')
        print('# Copy and paste these lines to create environments:')
        print()
        for env_id in env_ids:
            print(f"env = gym.make('{env_id}')")
        return
    print_environment_list(env_ids, args.detail)
    if env_ids:
        example_env = env_ids[0]
        print('Example usage:')
        print('```python')
        print('import gym')
        print(f"env = gym.make('{example_env}')")
        print("obs, info = env.reset(options={'game_state': None})")
        print('action = Action(agent_idx=0, game_state=\'\', code=\'print("Hello Factorio!")\')')
        print('obs, reward, terminated, truncated, info = env.step(action)')
        print('env.close()')
        print('```')

def main():
    """Main function - determines whether to run interactive examples or command-line mode"""
    if len(sys.argv) > 1:
        run_command_line_mode()
    else:
        run_interactive_examples()

class FactorioMCPState:
    """Manages the state of the Factorio MCP server"""

    def __init__(self):
        self.available_servers: Dict[int, FactorioServer] = {}
        self.active_server: Optional[FactorioInstance] = None
        self.server_entities: Dict[int, Dict[str, Any]] = {}
        self.server_resources: Dict[int, Dict[str, ResourcePatch]] = {}
        self.recipes: Dict[str, Recipe] = {}
        self.recipes_loaded = False
        self.checkpoints: Dict[int, Dict[str, str]] = {}
        self.current_task: Optional[str] = None
        self.last_entity_update = 0
        self.vcs_repos: Dict[int, 'FactorioMCPRepository'] = {}
        try:
            env_ids = list_available_environments()
            if not env_ids:
                raise Exception('No environments found')
            for id in env_ids:
                if 'open' in id:
                    print(f'DEBUG: Using open environment: {id}')
                    self.gym_env = gym.make(id, run_idx=0)
                    self.gym_env.reset()
                    return
            self.gym_env = gym.make(env_ids[0], run_idx=0)
        except IndexError as e:
            print(f'IndexError in __init__: {e}')
            print(f'env_ids length: {(len(env_ids) if 'env_ids' in locals() else 'Not available')}')
            print('Falling back to steel_plate_throughput environment')
            self.gym_env = gym.make('steel_plate_throughput', run_idx=0)
        except Exception as e:
            print(f'Error in __init__: {e}')
            print(f'Error type: {type(e)}')
            print('Falling back to steel_plate_throughput environment')
            self.gym_env = gym.make('steel_plate_throughput', run_idx=0)
        self.gym_env.reset()

    def create_factorio_instance(self, instance_id: int) -> FactorioInstance:
        """Create a single Factorio instance"""
        try:
            ips, udp_ports, tcp_ports = get_local_container_ips()
            if instance_id >= len(ips):
                raise IndexError(f'instance_id {instance_id} out of range for ips list of length {len(ips)}')
            if instance_id >= len(tcp_ports):
                raise IndexError(f'instance_id {instance_id} out of range for tcp_ports list of length {len(tcp_ports)}')
            instance = FactorioInstance(address=ips[instance_id], tcp_port=tcp_ports[instance_id], bounding_box=200, fast=True, cache_scripts=True, inventory={'stone-furnace': 1, 'burner-mining-drill': 1, 'wood': 5, 'iron-plate': 8}, all_technologies_researched=False)
            char_check = instance.rcon_client.send_command('/c rcon.print(global.agent_characters and #global.agent_characters or 0)')
            if int(char_check) == 0:
                instance.first_namespace._create_agent_characters(1)
            instance.set_speed(10)
            return instance
        except IndexError as e:
            print(f'IndexError in create_factorio_instance: {e}')
            try:
                print(f'Available IPs: {ips}')
                print(f'Available TCP ports: {tcp_ports}')
            except NameError:
                print('ERROR: Could not retrieve container IPs/ports')
            raise e
        except Exception as e:
            print(f'Error creating Factorio instance: {e}')
            print(f'Error type: {type(e)}')
            raise e

    async def scan_for_servers(self, ctx=None) -> List[FactorioServer]:
        """Scan for running Factorio servers"""
        try:
            ips, udp_ports, tcp_ports = get_local_container_ips()
            new_servers = {}
            for i in range(len(ips)):
                if ctx:
                    await ctx.report_progress(i, len(ips))
                instance_id = i
                if instance_id in self.available_servers:
                    server = self.available_servers[instance_id]
                    server.last_checked = time.time()
                    server.address = ips[i]
                    server.tcp_port = tcp_ports[i]
                    if not server.is_active:
                        try:
                            self.create_factorio_instance(i)
                            server.is_active = True
                        except Exception as e:
                            server.is_active = False
                            server.system_response = str(e)
                            print(str(e))
                    new_servers[instance_id] = server
                else:
                    server = FactorioServer(address=ips[i], tcp_port=int(tcp_ports[i]), instance_id=instance_id, name=f'Factorio Server {i + 1}', last_checked=time.time())
                    try:
                        self.create_factorio_instance(i)
                        server.is_active = True
                    except Exception as e:
                        server.is_active = False
                        server.system_response = str(e)
                    new_servers[instance_id] = server
                    if instance_id not in self.checkpoints:
                        self.checkpoints[instance_id] = {}
            self.available_servers = new_servers
            return list(self.available_servers.values())
        except Exception as e:
            raise e

    async def connect_to_server(self, instance_id: int) -> bool:
        """Connect to a Factorio server by instance ID"""
        if instance_id not in self.available_servers:
            return False
        server = self.available_servers[instance_id]
        if not server.is_active:
            return False
        try:
            instance = self.create_factorio_instance(instance_id)
            server.connected = True
            self.active_server = instance
            await self.refresh_game_data(instance_id)
            if not self.recipes:
                self.recipes = self.load_recipes_from_file()
            if instance_id not in self.vcs_repos:
                print('Initializing repo')
                self.vcs_repos[instance_id] = FactorioMCPRepository(instance)
            return True
        except Exception as e:
            print(f'Error connecting to Factorio server: {e}')
            return False

    def get_vcs(self):
        """Get the VCS repository for the active server"""
        if not self.active_server:
            return None
        instance_id = self.active_server.tcp_port
        if instance_id not in self.vcs_repos:
            self.vcs_repos[instance_id] = FactorioMCPRepository(self.active_server)
        return self.vcs_repos[instance_id]

    async def refresh_game_data(self, instance_id: int):
        """Refresh game data for a specific server instance"""
        if instance_id not in self.available_servers:
            return False
        self.last_entity_update = time.time()
        return True

    def load_recipes_from_file(self) -> Dict[str, Recipe]:
        """Load recipes from the jsonl file"""
        if self.recipes_loaded:
            return self.recipes
        recipes_path = Path(__file__).parent.parent / 'data' / 'recipes' / 'recipes.jsonl'
        if not recipes_path.exists():
            recipes_path = Path('/Users/jackhopkins/PycharmProjects/PaperclipMaximiser/data/recipes/recipes.jsonl')
        try:
            recipes = {}
            with open(recipes_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            recipe_data = json.loads(line)
                            ingredients = recipe_data.get('ingredients', [])
                            simplified_ingredients = []
                            for ingredient in ingredients:
                                simplified_ingredients.append({'name': ingredient.get('name', ''), 'amount': ingredient.get('amount', 1)})
                            results = [{'name': recipe_data.get('name', ''), 'amount': 1}]
                            recipes[recipe_data['name']] = Recipe(name=recipe_data['name'], ingredients=simplified_ingredients, results=results, energy_required=1.0)
                        except json.JSONDecodeError:
                            print(f'Warning: Could not parse recipe line: {line}')
                        except KeyError as e:
                            print(f'Warning: Missing key in recipe: {e}')
                        except Exception as e:
                            print(f'Warning: Error processing recipe: {e}')
            self.recipes_loaded = True
            return recipes
        except Exception as e:
            print(f'Error loading recipes from file: {e}')
            raise e

def get_validated_run_configs(run_config_location: str) -> list[GymRunConfig]:
    """Read and validate run configurations from file"""
    with open(run_config_location, 'r') as f:
        run_configs_raw = json.load(f)
        run_configs = [GymRunConfig(**config) for config in run_configs_raw]
    available_envs = list_available_environments()
    for run_config in run_configs:
        if run_config.env_id not in available_envs:
            raise ValueError(f"Environment ID '{run_config.env_id}' not found in registry. Available environments: {available_envs}")
    return run_configs

def test_registry_discovery():
    """Test that the registry can discover and list environments"""
    print('=== Testing Registry Discovery ===')
    env_ids = list_available_environments()
    print(f'Found {len(env_ids)} environments:')
    for env_id in env_ids[:5]:
        info = get_environment_info(env_id)
        print(f'  {env_id}')
        print(f'    Description: {info['description'][:60]}...')
        print(f'    Task Key: {info['task_key']}')
        print(f'    Agents: {info['num_agents']}')
    if len(env_ids) > 5:
        print(f'  ... and {len(env_ids) - 5} more environments')
    print()
    return env_ids

def test_environment_creation():
    """Test creating an environment (without actually running it)"""
    print('=== Testing Environment Creation ===')
    env_ids = list_available_environments()
    if not env_ids:
        print('No environments available to test')
        return
    test_env_id = env_ids[0]
    print(f'Attempting to create environment: {test_env_id}')
    try:
        env = gym.make(test_env_id)
        print('✓ Environment created successfully!')
        print(f'  Action space: {env.action_space}')
        print(f'  Observation space: {type(env.observation_space)}')
        print(f'  Number of agents: {env.instance.num_agents}')
        print('  Testing reset...')
        obs, info = env.reset()
        print(f'  Reset successful, observation keys: {list(obs.keys())}')
        print('  Testing simple action...')
        action = Action(agent_idx=0, code='print("Hello from Factorio Gym Registry!")', game_state=None)
        obs, reward, terminated, truncated, info = env.step(action)
        print('  Action successful!')
        print(f'    Reward: {reward}')
        print(f'    Done: {terminated}')
        print(f'    Truncated: {truncated}')
        print(f'    Info keys: {list(info.keys())}')
        env.close()
        print('  Environment closed successfully')
    except Exception as e:
        print(f'✗ Environment creation failed (expected if containers not running): {e}')
        print("  This is normal if Factorio containers aren't available")
    print()

def test_registry_functions():
    """Test the registry utility functions"""
    print('=== Testing Registry Functions ===')
    env_ids = list_available_environments()
    assert isinstance(env_ids, list)
    assert len(env_ids) > 0
    print(f'✓ list_available_environments() returned {len(env_ids)} environments')
    if env_ids:
        info = get_environment_info(env_ids[0])
        assert info is not None
        assert 'env_id' in info
        assert 'description' in info
        assert 'task_key' in info
        print(f'✓ get_environment_info() returned valid info for {env_ids[0]}')
    print()

def main():
    """Run all tests"""
    print('Factorio Gym Registry Test Suite')
    print('=' * 40)
    print()
    env_ids = test_registry_discovery()
    test_registry_functions()
    test_gym_integration()
    test_environment_creation(env_ids)
    print('Test suite completed!')
    print('\nTo use the registry in your code:')
    print('```python')
    print('import gym')
    print('from gym_env.registry import list_available_environments')
    print()
    print('# List available environments')
    print('env_ids = list_available_environments()')
    print()
    print('# Create an environment')
    print("env = gym.make('Factorio-iron_ore_throughput_16-v0')")
    print('```')

def test_gym_integration():
    """Test that environments are properly registered with gym"""
    print('=== Testing Gym Integration ===')
    from gym.envs.registration import registry
    factorio_envs = [env_id for env_id in registry.keys() if env_id.startswith('Factorio-')]
    print(f'Found {len(factorio_envs)} Factorio environments in gym registry:')
    for env_id in factorio_envs[:3]:
        print(f'  {env_id}')
    if len(factorio_envs) > 3:
        print(f'  ... and {len(factorio_envs) - 3} more')
    print()

