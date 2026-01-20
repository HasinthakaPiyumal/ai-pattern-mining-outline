# Cluster 36

class Program(BaseModel):
    id: Optional[int] = None
    code: str
    conversation: Conversation
    value: float = 0.0
    visits: int = 0
    parent_id: Optional[int] = None
    state: Optional[GameState] = None
    raw_reward: Optional[float] = None
    holdout_value: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)
    prompt_token_usage: Optional[int] = None
    completion_token_usage: Optional[int] = None
    token_usage: Optional[int] = None
    response: Optional[str] = None
    version: int = 1
    version_description: Optional[str] = ''
    model: str = 'gpt-4o'
    meta: dict = {}
    achievements: dict = {}
    instance: int = -1
    depth: int = 0
    advantage: float = 0
    ticks: int = 0
    flows: Optional[ProductionFlows] = None
    timing_metrics: List[TimingMetrics] = Field(default_factory=list)

    def __repr__(self):
        return self.code

    def get_step(self):
        return int((self.depth - 1) / 2 + 1)

    def get_uct(self, parent_visits: int, exploration_constant: float=1.41) -> float:
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + exploration_constant * np.sqrt(np.log(parent_visits) / self.visits)
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    @classmethod
    def from_row(cls, row: Dict):
        return cls(id=row['id'], code=row['code'], conversation=Conversation.parse_raw(row['conversation_json']), value=row['value'], visits=row['visits'], parent_id=row['parent_id'], state=GameState.parse(row['state_json']) if row['state_json'] else None, raw_reward=row['raw_reward'], holdout_value=row['holdout_value'], created_at=row['created_at'], prompt_token_usage=row['prompt_token_usage'], completion_token_usage=row['completion_token_usage'], token_usage=row['token_usage'], response=row['response'], version=row['version'], version_description=row['version_description'], meta=row['meta'] if row['meta'] else {}, achievements=row['achievements_json'] if row['achievements_json'] else {}, instance=row['instance'], depth=row['depth'], advantage=row['advantage'], ticks=row['ticks'], timing_metrics=[TimingMetrics.parse_raw(m) for m in row['timing_metrics_json']] if row.get('timing_metrics_json') else [])

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

class PlanSampler:

    def __init__(self, model: str, system_prompt_path: str, starting_scenarios_folder: str):
        self.model = model
        self.system_prompt_path = system_prompt_path
        self.llm_factory = APIFactory(model)
        self.starting_scenarios_folder = starting_scenarios_folder
        self.planning_addition_for_prompt = '\nFirst bring out a thorough step-by-step plan how you can achieve this task and then create the python script to achieve the task.\nFor your plan, follow this structure:\n1) What entities are needed for the task\n2) What entities do we have on the map, in different entity inventories or in our inventory\n3) What entities are we missing for the task\n4) Execution -- Taking into account 1,2 and 3, what steps do we need to take to successfully carry out the task\n\nCreate the python script based on your plan.\n'

    def get_game_state(self, instance: FactorioInstance, scenario: str) -> Optional[GameState]:
        """Load a scenario and return the corresponding game state"""
        try:
            instance.reset()
            scenario_path = os.path.join(self.starting_scenarios_folder, scenario)
            if not os.path.exists(scenario_path):
                return None
            config_file = os.path.join(scenario_path, 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                if 'inventory' in config:
                    instance.set_inventory(config['inventory'])
                if 'setup_script' in config:
                    setup_script_path = os.path.join(scenario_path, config['setup_script'])
                    if os.path.exists(setup_script_path):
                        with open(setup_script_path, 'r') as f:
                            setup_code = f.read()
                        instance.eval(setup_code, timeout=30)
            return GameState.from_instance(instance)
        except Exception as e:
            print(f'Error loading scenario {scenario}: {str(e)}')
            return None

    async def __call__(self, instance: FactorioInstance, game_state: GameState) -> Tuple[str, Any]:
        """Generate an objective/plan for the given game state"""
        try:
            with open(self.system_prompt_path, 'r') as f:
                system_prompt = f.read()
            inventory_info = f'Current inventory: {game_state.inventories[0]}'
            messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': f'{inventory_info}\n\n{self.planning_addition_for_prompt}'}]
            response = await self.llm_factory.acall(messages=messages, temperature=0.7, max_tokens=1024)
            if hasattr(response, 'choices'):
                objective = response.choices[0].message.content
            elif hasattr(response, 'content'):
                objective = response.content[0].text if hasattr(response.content[0], 'text') else response.content
            else:
                objective = str(response)
            return (objective.strip(), response)
        except Exception as e:
            print(f'Error generating plan: {str(e)}')
            return ('', None)

class MCTSFactory:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not MCTSFactory._initialized:
            self.db_client = None
            self.api_factory = None
            self.instances = None
            self.sampler = None
            MCTSFactory._initialized = True

    def initialize(self, instances, db_client, config: Union[BaseConfig, PlanningConfig, ChunkedConfig], sampler_config: SamplerConfig):
        self.instances = instances
        self.db_client = db_client
        self.config = config
        self.api_factory = APIFactory(model=config.model)
        self.sampler = _get_sampler(config.sampler_type, db_client, **sampler_config.__dict__)

    def create_mcts(self, config: Union[BaseConfig, PlanningConfig, ChunkedConfig, ObjectiveConfig]):
        if not all([self.instances, self.db_client, self.api_factory, self.sampler]):
            raise ValueError('Factory not initialized. Call initialize() first.')
        if config.mcts_type == MCTSType.CHUNKED:
            return self._create_chunked_mcts(config)
        elif config.mcts_type == MCTSType.PLANNING:
            return self._create_planning_mcts(config)
        elif config.mcts_type == MCTSType.OBJECTIVE:
            return self._create_objective_mcts(config)
        elif config.mcts_type == MCTSType.NORMAL:
            return self._create_mcts(config)
        raise ValueError(f'Unknown MCTS type: {config.mcts_type}')

    def _create_mcts(self, config: BaseConfig):
        from eval.algorithms.mcts import MCTS
        from eval.algorithms.mcts import ParallelMCTS
        from eval.algorithms.mcts import ParallelMCTSConfig
        mcts_config = ParallelMCTSConfig(n_parallel=config.n_parallel, system_prompt=config.system_prompt, initial_state=GameState.from_instance(self.instances[0]), mcts_class=MCTS, sampler=self.sampler, mcts_kwargs={'version': config.version, 'version_description': config.version_description, 'error_penalty': config.error_penalty})
        return ParallelMCTS(instances=self.instances, db_client=self.db_client, api_factory=self.api_factory, config=mcts_config, version=config.version, version_description=config.version_description)

    def _create_chunked_mcts(self, config: ChunkedConfig):
        from .mcts import ChunkedMCTS
        from .parallel_mcts import ParallelMCTS
        from .parallel_mcts_config import ParallelMCTSConfig
        from fle.agents.formatters.conversation_formatter_abc import StructurePreservingFormatter
        mcts_config = ParallelMCTSConfig(n_parallel=config.n_parallel, system_prompt=config.system_prompt, initial_state=GameState.from_instance(self.instances[0]), mcts_class=ChunkedMCTS, sampler=self.sampler, mcts_kwargs={'logit_bias': config.logit_bias, 'version': config.version, 'version_description': config.version_description, 'formatter': StructurePreservingFormatter(planning=True), 'presence_penalty': config.presence_penalty, 'frequency_penalty': config.frequency_penalty, 'error_penalty': config.error_penalty})
        return ParallelMCTS(instances=self.instances, db_client=self.db_client, api_factory=self.api_factory, config=mcts_config, version=config.version, version_description=config.version_description)

    def _create_objective_mcts(self, config: ObjectiveConfig):
        from eval.algorithms.mcts import ObjectiveMCTS
        from eval.algorithms.mcts import ParallelMCTS
        from eval.algorithms.mcts import ParallelMCTSConfig
        from fle.agents.formatters.conversation_formatter_abc import StructurePreservingFormatter
        mcts_config = ParallelMCTSConfig(n_parallel=config.n_parallel, system_prompt=config.system_prompt, initial_state=GameState.from_instance(self.instances[0]), mcts_class=ObjectiveMCTS, sampler=self.sampler, mcts_kwargs={'objective_model': config.objective_model, 'logit_bias': config.logit_bias, 'version': config.version, 'version_description': config.version_description, 'formatter': StructurePreservingFormatter(planning=True), 'presence_penalty': config.presence_penalty, 'frequency_penalty': config.frequency_penalty, 'error_penalty': config.error_penalty})
        return ParallelMCTS(instances=self.instances, db_client=self.db_client, api_factory=self.api_factory, config=mcts_config, version=config.version, version_description=config.version_description)

    def _create_planning_mcts(self, config: PlanningConfig):
        from eval.algorithms.mcts import PlanningMCTS
        from eval.algorithms.mcts import ParallelPlanningMCTS
        from eval.algorithms.mcts import ParallelMCTSConfig
        game_state = GameState.from_instance(self.instances[0])
        mcts_config = ParallelMCTSConfig(n_parallel=config.n_parallel, mcts_class=PlanningMCTS, sampler=self.sampler, system_prompt=config.system_prompt, initial_state=game_state, max_steps_per_objective=config.max_steps_per_objective, number_of_steps_for_judge=config.number_of_steps_for_judge, mcts_kwargs={'planning_model': config.planning_model, 'executor_model': config.executor_model, 'objective_model': config.objective_model, 'step_executor_prompt_path': config.step_executor_prompt_path, 'step_generator_prompt_path': config.step_generator_prompt_path, 'step_judge_prompt_path': config.step_judge_prompt_path, 'example_plan_prompt_path': config.example_plan_prompt_path, 'system_prompt': config.system_prompt, 'initial_state': game_state, 'error_penalty': config.error_penalty})
        return ParallelPlanningMCTS(instances=self.instances, db_client=self.db_client, api_factory=self.api_factory, config=mcts_config, version=config.version, version_description=config.version_description)

    @staticmethod
    def get_config_from_cli(default_version=42) -> Tuple[Union[BaseConfig, PlanningConfig, ChunkedConfig], SamplerConfig]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--type', choices=['chunked', 'planning', 'normal', 'objective'], help='MCTS type')
        parser.add_argument('--no-interactive', action='store_true', help='Skip interactive prompts')
        args, _ = parser.parse_known_args()
        if args.no_interactive:
            config, sampler_config = MCTSFactory._get_config_from_args(parser)
        else:
            config, sampler_config = MCTSFactory._get_config_interactive(args.type, default_version)
        MCTSFactory._save_config(config, sampler_config)
        return (config, sampler_config)

    @staticmethod
    def _get_config_from_args(parser) -> Tuple[Union[BaseConfig, PlanningConfig, ChunkedConfig], SamplerConfig]:
        parser.add_argument('--model', required=True)
        parser.add_argument('--version', type=int, required=True)
        parser.add_argument('--version-description', required=True)
        parser.add_argument('--n-parallel', type=int, default=4)
        parser.add_argument('--error-penalty', type=float, default=-10)
        parser.add_argument('--temperature', type=float, default=0.7)
        parser.add_argument('--compression-strength', type=float, default=None)
        parser.add_argument('--max-conversation-length', type=int, default=30)
        parser.add_argument('--adaptive-period', type=int, default=200)
        parser.add_argument('--window-size', type=int, default=200)
        parser.add_argument('--planning-model', default='claude-3-5-sonnet-20241022')
        parser.add_argument('--executor-model', default='ft:gpt-4o-2024-08-06:paperplane-ai:fact-instruct-1:ATSVGf4d:ckpt-step-214')
        parser.add_argument('--objective-model', default='ft:gpt-4o-2024-08-06:paperplane-ai:fact-self-gen-planning:AQzcPI91')
        parser.add_argument('--step-executor-prompt-path', default='../../prompts/bottoms_up_prompts/finetuning_prompts/step_supervised')
        parser.add_argument('--step-generator-prompt-path', default='../../prompts/bottoms_up_prompts/finetuning_prompts/step_generator')
        parser.add_argument('--step-judge-prompt-path', default='../../prompts/bottoms_up_prompts/finetuning_prompts/step_judge')
        parser.add_argument('--example-plan-prompt-path', default='../../prompts/bottoms_up_prompts/finetuning_prompts/executor_plan')
        args = parser.parse_args()
        mcts_type = MCTSType(args.type)
        if mcts_type == MCTSType.PLANNING:
            mcts_config = PlanningConfig(mcts_type=mcts_type, model=args.model, version=args.version, version_description=args.version_description, n_parallel=args.n_parallel, system_prompt='', planning_model=args.planning_model, executor_model=args.executor_model, objective_model=args.objective_model, step_executor_prompt_path=Path(args.step_executor_prompt_path), step_generator_prompt_path=Path(args.step_generator_prompt_path), step_judge_prompt_path=Path(args.step_judge_prompt_path), example_plan_prompt_path=Path(args.example_plan_prompt_path), error_penalty=args.error_penalty)
        elif mcts_type == MCTSType.CHUNKED:
            mcts_config = ChunkedConfig(mcts_type=mcts_type, model=args.model, version=args.version, version_description=args.version_description, n_parallel=args.n_parallel, system_prompt='', error_penalty=args.error_penalty)
        elif mcts_type == MCTSType.OBJECTIVE:
            mcts_config = ObjectiveConfig(objective_model=args.objective_model, mcts_type=mcts_type, model=args.model, version=args.version, version_description=args.version_description, n_parallel=args.n_parallel, system_prompt='', error_penalty=args.error_penalty)
        else:
            mcts_config = BaseConfig(mcts_type=mcts_type, model=args.model, version=args.version, version_description=args.version_description, n_parallel=args.n_parallel, system_prompt='', error_penalty=args.error_penalty)
        sampler_config = SamplerConfig(temperature=args.temperature, compression_strength=args.compression_strength, max_conversation_length=args.max_conversation_length, adaptive_period=args.adaptive_period, window_size=args.window_size)
        return (mcts_config, sampler_config)

    @staticmethod
    def _get_config_interactive(default_type=None, default_version=42) -> Tuple[Union[BaseConfig, PlanningConfig, ChunkedConfig], SamplerConfig]:
        mcts_type = default_type or questionary.select('Select MCTS type:', choices=['chunked', 'normal', 'planning', 'objective'], instruction='Choose MCTS algorithm variant. Planning is recommended for complex tasks.').ask()
        model = 'gpt-4o'
        if mcts_type != 'planning':
            model = questionary.select('Model name:', choices=['gemini-2.0-flash-exp', 'gemini-2.0-flash-thinking-exp-1219', 'gemini-exp-1206', 'deepseek-chat', 'gpt-4o', 'claude-3-5-sonnet-20241022', 'meta-llama/Llama-3.3-70B-Instruct-Turbo', 'meta-llama/Meta-Llama-3.3-8B-Instruct-Turbo', 'Qwen/Qwen2.5-7B-Instruct-Turbo', 'Qwen/Qwen2.5-72B-Instruct-Turbo', 'ft:gpt-4o-mini-2024-07-18:paperplane-ai:mcts-pruned-masked:AYIViDdb'], instruction='Model to use for program synthesis.').ask()
        base_config = {'mcts_type': MCTSType(mcts_type), 'model': model, 'version': int(questionary.text('Version:', default=str(default_version), instruction='The run version number. Higher versions may include bug fixes or improvements.').ask()), 'n_parallel': int(questionary.text('Number of parallel instances:', default='4').ask()), 'presence_penalty': float(questionary.text('Fixed presence penalty applied across previously sampled logits. -2 to 2.', default='0').ask()), 'frequency_penalty': float(questionary.text('Dynamic frequency penalty applied across previously sampled logits. -2 to 2.', default='0').ask()), 'error_penalty': float(questionary.text('Penalty applied when there is an execution error(e.g. syntax error).', default='-10').ask()), 'system_prompt': ''}
        if mcts_type == 'planning':
            mcts_config = PlanningConfig(**base_config, planning_model=questionary.text('Planning model:', default='claude-3-5-sonnet-20241022', instruction='The model that samples plans by reasoning over objectives and game states.').ask(), executor_model=questionary.text('Executor model:', default='ft:gpt-4o-2024-08-06:paperplane-ai:fact-instruct-1:ATSVGf4d:ckpt-step-214', instruction='The model that samples programs.').ask(), objective_model=questionary.text('Objective model:', default='ft:gpt-4o-2024-08-06:paperplane-ai:fact-self-gen-planning:AQzcPI91', instruction='The model that generates new objectives.').ask(), max_steps_per_objective=int(questionary.text('Maximum steps per objective:', default='12').ask()), number_of_steps_for_judge=int(questionary.text('Number of steps for judge:', default='3', instruction='The branching factor for the planning tree. Higher values increase quality but use more tokens.').ask()))
        elif mcts_type == 'objective':
            mcts_config = ObjectiveConfig(**base_config, objective_model=questionary.text('Objective model:', default='ft:gpt-4o-mini-2024-07-18:paperplane-ai:plans-tree:AcZ8gHSo', instruction='The model that samples objectives.').ask())
        elif mcts_type == 'chunked':
            mcts_config = ChunkedConfig(**base_config)
        else:
            mcts_config = BaseConfig(**base_config)
        mcts_config.sampler_type = SamplerType(questionary.select('Select MCTS node sampler type:', choices=['weighted reward', 'kld', 'beam'], instruction='Choose the sampling method for selecting actions. KLD priorities varied game states. Weighted reward prioritizes high-reward states.').ask())
        skip_failures = questionary.select('Skip failures?', choices=['no', 'yes'], instruction='Shall we skip nodes that trigger an exception/error?').ask()
        mcts_config.skip_failures = skip_failures == 'yes'
        if mcts_config.sampler_type == SamplerType.KLD:
            sampler_config = SamplerConfig(temperature=float(questionary.text('Temperature:', default='1', instruction='Higher values are closer to uniform sampling. Zero means greedy sampling from reward.').ask()), window_size=int(questionary.text('Window size:', default='100', instruction='The number of recent programs to consider when sampling the next node').ask()), maximum_lookback=int(questionary.text('Maximum lookback steps', default='20').ask()))
        elif mcts_config.sampler_type == SamplerType.BEAM:
            sampler_config = SamplerConfig(beam_width=int(questionary.text('Beam width:', default='8', instruction='The number of nodes to keep in the beam for sampling subsequent nodes').ask()), exploration_prob=float(questionary.text('Exploration probability:', default='0.1', instruction='The probability to sample outside of the beam (for exploration)').ask()), maximum_lookback=int(questionary.text('Maximum lookback steps', default='20').ask()))
        else:
            compression_strength = float(questionary.text('Compression strength:', instruction='Between 0-1. Higher values mean more exploration. Lower means more exploitation. -1 means adaptively cycle', default='-1').ask())
            if compression_strength < 0:
                compression_strength = None
            sampler_config = SamplerConfig(compression_strength=compression_strength, max_conversation_length=int(questionary.text('Maximum conversation length:', instruction='The maximum number of assistant actions in the dialogue', default='100').ask()))
            if compression_strength is not None:
                sampler_config.adaptive_period = int(questionary.text('Adaptive period:', instruction='The period for cycling exploration and exploitation', default='50').ask())
            sampler_config.maximum_lookback = int(questionary.text('Maximum lookback steps', default='20').ask())
        version_description = ''
        for key, value in mcts_config.__dict__.items():
            if isinstance(value, Path):
                value = str(value)
            version_description += f'{key}:{value}\n'
        for key, value in sampler_config.__dict__.items():
            if isinstance(value, Path):
                value = str(value)
            version_description += f'{key}:{value}\n'
        mcts_config.version_description = version_description
        return (mcts_config, sampler_config)

    @staticmethod
    def _save_config(config: Union[BaseConfig, PlanningConfig, ChunkedConfig], sampler_config: SamplerConfig):
        """Save the run configuration to a JSON file"""
        runs_dir = Path(f'runs/{config.version}')
        runs_dir.mkdir(exist_ok=True)
        config_dict = {k: str(v) if isinstance(v, (Path, MCTSType, SamplerType)) else v for k, v in asdict(config).items() if not k.endswith('_model') and (not isinstance(v, (Path, type(None))))}
        sampler_dict = {k: v for k, v in dataclasses.asdict(sampler_config).items() if v is not None}
        save_data = {'mcts_config': config_dict, 'sampler_config': sampler_dict, 'timestamp': datetime.now().isoformat()}
        config_file = runs_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(save_data, f, indent=2)

class Evaluator:

    def __init__(self, db_client: DBClient, instances: List[FactorioInstance], value_accrual_time=10, error_penalty=10, logger=None):
        self.db = db_client
        self.instances = instances
        self.value_accrual_time = value_accrual_time
        self.error_penalty = error_penalty
        if not logger:
            self.logger = FactorioLogger(len(instances))
            self.logger.start()
        else:
            self.logger = logger
        self.instance_to_port = {i: instance.tcp_port for i, instance in enumerate(self.instances)}
        if logger:
            self.port_to_group = logger.port_to_group

    def set_status(self, status):
        for instance in self.instances:
            self.logger.update_instance(instance.tcp_port, status=status)

    def set_sampling_status(self):
        """Update status for all instances in this evaluator's group"""
        if self.logger:
            for instance in self.instances:
                self.logger.update_instance(instance.tcp_port, status='sampling')

    def set_iteration(self, iteration, n_iterations):
        """Update iteration number for all instances in this evaluator's group"""
        if self.logger:
            for instance in self.instances:
                self.logger.update_instance(instance.tcp_port, iteration=iteration, n_iterations=n_iterations)

    async def evaluate_batch(self, programs: List[Program], start_state: GameState) -> List[Program]:
        try:
            eval_futures = []
            for i, (prog, inst) in enumerate(zip(programs, self.instances)):
                inst.reset(start_state)
                if self.logger:
                    self.logger.update_instance(inst.tcp_port, program_id=prog.id, status='resetting')
                eval_futures.append(self._evaluate_single(inst.tcp_port, prog, inst))
            eval_results = await asyncio.gather(*eval_futures)
            for i, (program, (raw_reward, state, response, entities, achievements, ticks)) in enumerate(zip(programs, eval_results)):
                relative_reward = raw_reward
                if self.logger:
                    self.logger.update_instance(self.instances[i].tcp_port, status='completed', raw_reward=raw_reward, holdout_value=raw_reward, relative_reward=relative_reward, total_programs=self.logger.groups[self.port_to_group[self.instances[i].tcp_port]].instances[self.instances[i].tcp_port].total_programs + 1)
                program.value = relative_reward
                program.state = state
                program.raw_reward = raw_reward
                program.ticks = ticks
                conversation = copy.deepcopy(program.conversation)
                conversation.add_result(program.code, response, score=raw_reward, advantage=relative_reward, objectives=program.meta['objectives'] if 'objectives' in program.meta else [])
                program.conversation = conversation
                program.response = response
                program.achievements = achievements
            return programs
        except Exception as e:
            if self.logger:
                for instance in self.instances:
                    self.logger.update_instance(instance.tcp_port, status='error', error_count=self.logger.groups[self.port_to_group[instance.tcp_port]].instances[instance.tcp_port].error_count + 1)
            raise e

    def _evaluate_for_achievements(self, code: str, instance: FactorioInstance) -> Tuple[float, GameState, str, List[Union[Entity, EntityGroup]], Dict[str, Dict[str, int]]]:
        start_production_flows = instance.namespace._get_production_stats()
        reward, time, result = instance.eval(code, timeout=120)
        post_production_flows = instance.namespace._get_production_stats()
        achievements = get_achievements(start_production_flows, copy.deepcopy(post_production_flows))
        return (result, achievements, post_production_flows)

    async def _evaluate_single(self, instance_id: int, program: Program, instance: FactorioInstance) -> Tuple[float, GameState, str, List[Union[Entity, EntityGroup]], Dict[str, Dict[str, int]], int]:
        try:
            tcp_port = self.instance_to_port[instance_id]
        except:
            tcp_port = instance_id
        try:
            self.logger.update_instance(tcp_port, status='starting value')
            start_entities = instance.namespace.get_entities()
            start_inventory = instance.namespace.inspect_inventory()
            start_production_flows = instance.namespace._get_production_stats()
            initial_value, start_time = instance.namespace.score()
            self.logger.update_instance(tcp_port, status='executing')
            reward, time, result = instance.eval(program.code, timeout=60)
            self.logger.update_instance(tcp_port, status='capturing state')
            state = GameState.from_instance(instance)
            self.logger.update_instance(tcp_port, status=f'accruing value ({self.value_accrual_time}s)')
            await asyncio.sleep(self.value_accrual_time)
            entities = instance.namespace.get_entities()
            final_inventory = instance.namespace.inspect_inventory()
            get_inventory_code = 'print(f"Current inventory {inspect_inventory()}")'
            if start_inventory.__dict__ != final_inventory.__dict__ and 'error' not in result.lower() and (get_inventory_code not in program.code) and ('inspect_inventory()' not in program.code):
                program.code += f'\n{get_inventory_code}'
                result += '\n' + str(len(program.code.split('\n'))) + f": ('Current inventory {final_inventory}',)"
            get_entities_code = 'print(f"Entities on the map: {get_entities()}")'
            if start_entities != entities and 'error' not in result.lower() and (get_entities_code not in program.code) and ('get_entities()' not in program.code):
                program.code += f'\n{get_entities_code}\n'
                result += '\n' + str(len(program.code.split('\n'))) + f": ('Entities on the map: {entities}',)"
            result = result.rstrip() + '\n'
            if 'error' in result.lower():
                result += f"('Current inventory: {final_inventory}',)\n"
                result += f"('Entities on the map after the current step: {entities}',)"
            score, _ = instance.namespace.score()
            final_reward = score - initial_value
            ticks = instance.get_elapsed_ticks()
            post_production_flows = instance.namespace._get_production_stats()
            achievements = get_achievements(start_production_flows, post_production_flows)
            group_id = self.port_to_group[tcp_port]
            group = self.logger.groups[group_id]
            instance_metrics = group.instances[tcp_port]
            self.logger.update_instance(tcp_port, status='accrued value', current_reward=final_reward, raw_reward=final_reward, final_entities=len(entities), start_entities=len(start_entities), total_programs=instance_metrics.total_programs + 1, start_inventory_count=sum([v for k, v in start_inventory.__dict__.items() if v > 0]), final_inventory_count=sum([v for k, v in final_inventory.__dict__.items() if v > 0]))
            if 'error' in result.lower() and self.logger:
                group_id = self.port_to_group[tcp_port]
                group = self.logger.groups[group_id]
                instance_metrics = group.instances[tcp_port]
                self.logger.update_instance(tcp_port, status='error', error_count=instance_metrics.error_count + 1)
            return (final_reward, state, result, entities, achievements, ticks)
        except Exception as e:
            print('Error in _evaluate_single:')
            print(f'Instance ID: {instance_id}')
            print(f'TCP Port: {self.instance_to_port.get(instance_id, 'Unknown')}')
            print(f'Error: {str(e)}')
            import traceback
            traceback.print_exc()
            if self.logger:
                tcp_port = self.instance_to_port[instance_id]
                group_id = self.port_to_group[tcp_port]
                group = self.logger.groups[group_id]
                instance_metrics = group.instances[tcp_port]
                self.logger.update_instance(tcp_port, status='error', error_count=instance_metrics.error_count + 1)
            raise e

    def __del__(self):
        """Clean up logger on deletion"""
        self.logger.stop()

def get_achievements(pre_production_flows, post_production_flows):
    """
    Calculate the dynamic production flows between two states
    """
    achievements = {'static': {}, 'dynamic': {}}
    if not isinstance(pre_production_flows, dict) or not isinstance(post_production_flows, dict):
        return achievements
    if 'output' not in pre_production_flows or 'output' not in post_production_flows:
        return achievements
    post_production_flows['static_items'] = get_updated_static_items(pre_production_flows, post_production_flows)
    achievements = process_achievements(pre_production_flows, post_production_flows, achievements)
    return achievements

class BlueprintScenarioSampler:
    """Samples scenarios from existing Factorio blueprint implementations in the `skills` table."""

    def __init__(self, db_config: Dict[str, str], system_prompt: str):
        """
        Initialize the blueprint sampler

        Args:
            db_config: Database configuration for skills DB
            system_prompt: System prompt for the conversation
        """
        self.db_config = db_config
        self.system_prompt = system_prompt
        self.conn = psycopg2.connect(**db_config)

    def _get_blueprint_scenarios(self, limit: int=100, version: str='v1.4') -> List[Dict]:
        """Get blueprint scenarios from skills database"""
        query = '\n            SELECT description, implementation, score, complexity, dependencies\n            FROM skills\n            WHERE implementation IS NOT NULL \n            AND description IS NOT NULL\n            AND version = %s\n            ORDER BY RANDOM()\n            LIMIT %s\n        '
        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (version, limit))
            return [dict(row) for row in cur.fetchall()]

    def _prepare_scenario(self, instance: FactorioInstance, blueprint: Dict, inventory: Dict) -> Tuple[Optional[GameState], str, str]:
        """
        Prepare a scenario from a blueprint implementation

        Returns:
            Tuple of (game_state, objective, implementation)
        """
        try:
            instance.reset()
            instance.set_inventory(inventory)
            reward, _, result = instance.eval(blueprint['implementation'], timeout=30)
            if 'error' in result.lower():
                raise Exception('Could not initialise scenario')
            game_state = GameState.from_instance(instance)
            objective = blueprint['description']
            if not objective.startswith('"""'):
                objective = f'"""\n{objective}\n"""'
            return (game_state, objective, blueprint['implementation'])
        except Exception as e:
            print(f'Error preparing scenario: {str(e)}')
            raise e

    async def sample_scenarios(self, instance: FactorioInstance, n_samples: int=10, version: int=17, skill_version: str='v1.4') -> List[Program]:
        """
        Sample scenarios and create seed programs

        Args:
            instance: Factorio instance for state setup
            n_samples: Number of scenarios to sample

        Returns:
            List of Program objects ready for seeding
        """
        blueprints = self._get_blueprint_scenarios(limit=n_samples, version=skill_version)
        programs = []
        for blueprint in blueprints:
            dependencies = blueprint['dependencies']
            inventory = {}
            for dependency in dependencies:
                item, count = dependency.strip().split(':')
                inventory[item.strip("'")] = int(count)
            try:
                game_state, objective, implementation = self._prepare_scenario(instance, blueprint, inventory)
            except Exception:
                continue
            if not game_state:
                continue
            conversation = Conversation(messages=[Message(role='system', content=self.system_prompt), Message(role='user', content=f'Starting Inventory: {inventory}'), Message(role='assistant', content=objective), Message(role='user', content='Execution result: \n'), Message(role='assistant', content=implementation.strip())])
            program = Program(id=hash((objective, str(conversation.messages))), code=implementation.strip(), conversation=conversation, value=float(blueprint['score']), state=game_state, version=version, version_description='Blueprint-based scenario seed')
            programs.append(program)
        return programs

    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()

class TaskABC:

    def __init__(self, trajectory_length, starting_inventory: Inventory, goal_description: str, task_key: str, all_technology_reserached: bool=False, agent_instructions: Optional[List[str]]=None):
        self.trajectory_length = trajectory_length
        self.starting_inventory = starting_inventory
        self.goal_description = goal_description
        self.task_key = task_key
        self.all_technology_reserached = all_technology_reserached
        self.agent_instructions = agent_instructions

    def get_agent_instructions(self, agent_idx: int) -> Optional[str]:
        if self.agent_instructions is None:
            return None
        elif agent_idx >= len(self.agent_instructions):
            raise IndexError(f'Agent index {agent_idx} is out of bounds for agent instructions')
        else:
            return self.agent_instructions[agent_idx]

    def verify(self, score: float, step: int, instance: FactorioInstance, step_statistics: Dict) -> TaskResponse:
        """Verify if the task is completed based on the current state.

        Args:
            score (float): The current score/reward value
            step (int): The current step number
            instance (FactorioInstance): The Factorio game instance
            step_statistics (Dict): Dictionary containing statistics about the current step

        Returns:
            TaskResponse: Response object indicating task completion status and metadata
        """
        pass

    def setup_instance(self, instance):
        """Code to provision the task environment"""
        pass

    def enhance_response_with_task_output(self, response: str, task_response: TaskResponse) -> str:
        """Add task specific information to the environment response"""
        return response

    def setup(self, instance):
        """setup function"""
        instance.initial_inventory = self.starting_inventory
        instance.reset(all_technologies_researched=self.all_technology_reserached)
        self.setup_instance(instance)
        self.starting_game_state = GameState.from_instance(instance)

class TestSaveLoadPythonNamespace(unittest.TestCase):
    """
    FactorioInstance exposes a Python namespace for the agent to persist variables.
    These tests verify that the namespace can be saved and loaded correctly into new instances.
    """

    def setUp(self):
        self.instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, fast=True, inventory={'boiler': 1, 'burner-mining-drill': 1, 'coal': 50, 'transport-belt': 10})

    def test_save_load_simple_variable_namespace(self):
        self.instance.eval('x=2')
        game_state = GameState.from_instance(self.instance)
        self.instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, fast=True, inventory={})
        self.instance.reset(game_state)
        self.instance.eval('assert x == 2')

    def test_load_in_all_variable_values(self):
        game_state_raw = {'entities': 'eJyrrgUAAXUA+Q==', 'inventory': {'coal': 50, 'iron-ore': 50}, 'namespace': '800495d6090000000000007d94288c046576616c948c086275696c74696e73948c046576616c9493948c08456c6c6970736973948c086275696c74696e73948c08456c6c69707369739493948c0546616c736594898c044e6f6e65944e8c0e4e6f74496d706c656d656e7465649468068c0e4e6f74496d706c656d656e7465649493948c045472756594888c036162739468028c036162739493948c03616c6c9468028c03616c6c9493948c03616e799468028c03616e799493948c0561736369699468028c0561736369699493948c0362696e9468028c0362696e9493948c0a627265616b706f696e749468028c0a627265616b706f696e749493948c0863616c6c61626c659468028c0863616c6c61626c659493948c036368729468028c036368729493948c07636f6d70696c659468028c07636f6d70696c659493948c09636f70797269676874948c0d5f736974656275696c74696e73948c085f5072696e7465729493942981947d94288c0e5f5072696e7465725f5f6e616d6594682a8c0e5f5072696e7465725f5f64617461945833010000436f707972696768742028632920323030312d3230323120507974686f6e20536f66747761726520466f756e646174696f6e2e0a416c6c205269676874732052657365727665642e0a0a436f707972696768742028632920323030302042654f70656e2e636f6d2e0a416c6c205269676874732052657365727665642e0a0a436f707972696768742028632920313939352d3230303120436f72706f726174696f6e20666f72204e6174696f6e616c20526573656172636820496e6974696174697665732e0a416c6c205269676874732052657365727665642e0a0a436f707972696768742028632920313939312d3139393520537469636874696e67204d617468656d6174697363682043656e7472756d2c20416d7374657264616d2e0a416c6c205269676874732052657365727665642e948c0f5f5072696e7465725f5f6c696e6573944e8c135f5072696e7465725f5f66696c656e616d6573945d9475628c076372656469747394682d2981947d94286830683668318c9e202020205468616e6b7320746f204357492c20434e52492c2042654f70656e2e636f6d2c205a6f706520436f72706f726174696f6e20616e6420612063617374206f662074686f7573616e64730a20202020666f7220737570706f7274696e6720507974686f6e20646576656c6f706d656e742e2020536565207777772e707974686f6e2e6f726720666f72206d6f726520696e666f726d6174696f6e2e9468334e68345d9475628c0764656c617474729468028c0764656c617474729493948c036469729468028c036469729493948c066469766d6f649468028c066469766d6f649493948c04657865639468028c04657865639493948c046578697494682b8c07517569747465729493942981947d94288c046e616d659468478c03656f66948c114374726c2d442028692e652e20454f46299475628c06666f726d61749468028c06666f726d61749493948c07676574617474729468028c07676574617474729493948c07676c6f62616c739468028c07676c6f62616c739493948c07686173617474729468028c07686173617474729493948c04686173689468028c04686173689493948c0468656c7094682b8c075f48656c7065729493942981948c036865789468028c036865789493948c0269649468028c0269649493948c05696e7075749468028c05696e7075749493948c0a6973696e7374616e63659468028c0a6973696e7374616e63659493948c0a6973737562636c6173739468028c0a6973737562636c6173739493948c04697465729468028c04697465729493948c036c656e9468028c036c656e9493948c076c6963656e736594682d2981947d94286830687768318c275365652068747470733a2f2f7777772e707974686f6e2e6f72672f7073662f6c6963656e73652f9468334e68345d94288c722f4c6962726172792f446576656c6f7065722f436f6d6d616e644c696e65546f6f6c732f4c6962726172792f4672616d65776f726b732f507974686f6e332e6672616d65776f726b2f56657273696f6e732f332e392f6c69622f707974686f6e332e392f2e2e2f4c4943454e53452e747874948c6e2f4c6962726172792f446576656c6f7065722f436f6d6d616e644c696e65546f6f6c732f4c6962726172792f4672616d65776f726b732f507974686f6e332e6672616d65776f726b2f56657273696f6e732f332e392f6c69622f707974686f6e332e392f2e2e2f4c4943454e5345948c6f2f4c6962726172792f446576656c6f7065722f436f6d6d616e644c696e65546f6f6c732f4c6962726172792f4672616d65776f726b732f507974686f6e332e6672616d65776f726b2f56657273696f6e732f332e392f6c69622f707974686f6e332e392f4c4943454e53452e747874948c6b2f4c6962726172792f446576656c6f7065722f436f6d6d616e644c696e65546f6f6c732f4c6962726172792f4672616d65776f726b732f507974686f6e332e6672616d65776f726b2f56657273696f6e732f332e392f6c69622f707974686f6e332e392f4c4943454e5345948c0d2e2f4c4943454e53452e747874948c092e2f4c4943454e5345946575628c066c6f63616c739468028c066c6f63616c739493948c036d61789468028c036d61789493948c036d696e9468028c036d696e9493948c046e6578749468028c046e6578749493948c036f63749468028c036f63749493948c046f70656e948c02696f948c046f70656e9493948c036f72649468028c036f72649493948c03706f779468028c03706f779493948c057072696e749468028c057072696e749493948c04717569749468492981947d9428684c689e684d684e75628c04726570729468028c04726570729493948c05726f756e649468028c05726f756e649493948c07736574617474729468028c07736574617474729493948c06736f727465649468028c06736f727465649493948c0373756d9468028c0373756d9493948c04766172739468028c04766172739493948c0d636f616c5f706f736974696f6e948c11666163746f72696f5f656e746974696573948c08506f736974696f6e9493942981947d94288c085f5f646963745f5f947d94288c0178944740338000000000008c01799447c027000000000000758c125f5f707964616e7469635f65787472615f5f944e8c175f5f707964616e7469635f6669656c64735f7365745f5f948f942868bc68bb908c145f5f707964616e7469635f707269766174655f5f944e75628c0e636f616c5f686172766573746564944b328c0e69726f6e5f686172766573746564944b328c0d69726f6e5f706f736974696f6e9468b62981947d942868b97d942868bb47c02700000000000068bc4740338000000000007568bd4e68be8f942868bc68bb9068c04e7562752e', 'timestamp': 1735481767.4378161}
        game_state = GameState.parse(game_state_raw)
        self.instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, fast=True, inventory={})
        self.instance.reset(game_state)
        ngame_state = GameState.from_instance(self.instance)
        nvars = pickle.loads(ngame_state.namespaces[0])
        assert 'coal_position' in nvars and nvars['coal_position']

    def test_save_load_items_on_ground(self):
        MINER = '\nmove_to(nearest(Resource.IronOre))\nminer = place_entity(Prototype.BurnerMiningDrill, Direction.UP, nearest(Resource.IronOre))\ninsert_item(Prototype.Coal, miner)\nsleep(15)\npickup_entity(miner)\n'
        self.instance.eval(MINER)
        game_state = GameState.from_instance(self.instance)
        self.instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, fast=True, inventory={})
        self.instance.reset(game_state)
        response = self.instance.eval('pickup_entity(Prototype.IronOre, miner.drop_position)')
        assert 'Error' not in response

    def test_save_load_items_on_belt(self):
        MINER = '\nmove_to(nearest(Resource.IronOre))\nminer = place_entity(Prototype.BurnerMiningDrill, Direction.UP, nearest(Resource.IronOre))\ninsert_item(Prototype.Coal, miner)\nbelt = place_entity(Prototype.TransportBelt, Direction.UP, miner.drop_position)\nsleep(15)\npickup_entity(miner)\n'
        response = self.instance.eval(MINER)
        game_state = GameState.from_instance(self.instance)
        self.instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, fast=True, inventory={})
        self.instance.reset(game_state)
        response = self.instance.eval('pickup_entity(Prototype.TransportBelt, belt.position)')
        assert 'Error' not in response
        assert self.instance.namespace.inspect_inventory()[Prototype.IronOre] == 4

    def test_save_load_simple_variable_namespace_with_exception(self):
        self.instance.eval('boiler = place_entity(Prototype.Boiler, Direction.UP, Position(x=0, y=0))')
        game_state = GameState.from_instance(self.instance)
        self.instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, fast=True, inventory={})
        self.instance.reset(game_state)
        _, _, response = self.instance.eval('print(boiler.position)')
        assert 'Error' not in response
        pass

    def test_save_load_simple_variable_namespace2(self):
        self.instance.eval('boiler = place_entity(Prototype.Boiler, Direction.UP, Position(x=0, y=0))')
        game_state = GameState.from_instance(self.instance)
        self.instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, fast=True, inventory={})
        self.instance.reset()
        self.instance.reset(game_state)
        self.instance.eval('assert boiler')
        resp2 = self.instance.eval('print(boiler)')
        assert 'error' not in resp2

    def test_declare_load_function_definition(self):
        self.instance.eval('def myfunc():\n  return "hello world"')
        _, _, resp2 = self.instance.eval('print(myfunc())')
        assert 'hello world' in resp2
        game_state = GameState.from_instance(self.instance)
        self.instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, fast=True, inventory={})
        self.instance.reset()
        self.instance.reset(game_state)
        _, _, resp3 = self.instance.eval('print(myfunc())')
        assert 'hello world' in resp3

    def test_full_program(self):
        splitter = ChunkedMCTS(None, None, None, '', None, initial_state=None)
        parts = splitter._split_into_chunks(FULL_PROGRAM)
        for part in parts:
            resp = self.instance.eval(part.code)
            game_state = GameState.from_instance(self.instance)
            self.instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, fast=True, inventory={})
            self.instance.reset()
            self.instance.reset(game_state)
            print(resp)

    def test_drill_inventory(self):
        resp = self.instance.eval(FULL_PROGRAM_2)
        game_state = GameState.from_instance(self.instance)
        self.instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, fast=True, inventory={'coal': 50})
        self.instance.reset()
        self.instance.reset(game_state)
        print(resp)

    def test_save_load_research_new_instance(self):
        game_state = GameState.from_instance(self.instance)
        self.instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, fast=True, all_technologies_researched=False, inventory={})
        self.instance.reset()
        n_game_state = GameState.from_instance(self.instance)
        game_state_techs = game_state.research.technologies.values()
        n_game_state_techs = n_game_state.research.technologies.values()
        for i, j in zip(game_state_techs, n_game_state_techs):
            assert i.name == j.name
            skip = False
            for k in i.prerequisites:
                if k == 'space-science-pack':
                    skip = True
            if not skip:
                assert i.researched, f'Technology {i.name} should be researched'
                assert not j.researched, f'Technology {j.name} should not be researched'
        self.instance.reset(game_state)
        k_game_state = GameState.from_instance(self.instance)
        k_game_state_techs = k_game_state.research.technologies.values()
        for tech in k_game_state_techs:
            skip = False
            for k in tech.prerequisites:
                if k == 'space-science-pack':
                    skip = True
            if not skip:
                assert tech.researched

def test_game_state_reserach():

    class DummyObject(BaseModel):
        game_state: GameState = None
    instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27019, fast=True, inventory={}, all_technologies_researched=True)
    zero_state = GameState.from_instance(instance)
    new_object = DummyObject(game_state=zero_state)

def test_craft_automation_packs_and_research(game):
    game.inspect_inventory()
    game.move_to(game.nearest(Resource.Water))
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=game.nearest(Resource.Water), direction=Direction.LEFT)
    boiler = game.place_entity_next_to(Prototype.Boiler, reference_position=offshore_pump.position, direction=offshore_pump.direction, spacing=3)
    game.insert_item(Prototype.Coal, boiler, quantity=10)
    steam_engine = game.place_entity_next_to(Prototype.SteamEngine, reference_position=boiler.position, direction=boiler.direction, spacing=2)
    lab = game.place_entity_next_to(Prototype.Lab, reference_position=steam_engine.position, direction=steam_engine.direction, spacing=2)
    assert lab, 'Failed to place Lab'
    game.connect_entities(steam_engine, lab, connection_type=Prototype.SmallElectricPole)
    game.connect_entities(boiler, steam_engine, connection_type=Prototype.Pipe)
    game.connect_entities(boiler, offshore_pump, connection_type=Prototype.Pipe)
    game.insert_item(Prototype.AutomationSciencePack, lab, quantity=10)
    lab_inventory = game.inspect_inventory(lab)
    assert lab_inventory.get(Prototype.AutomationSciencePack) == 10, f'Failed to insert science packs into Lab. Current count: {lab_inventory.get(Prototype.AutomationSciencePack)}'
    ingredients1 = game.set_research(Technology.Automation)
    game.get_entities()
    game.sleep(10)
    ingredients2 = game.get_research_progress(Technology.Automation)
    assert ingredients1[0].count > ingredients2[0].count, f'Research did not progress. Initial: {ingredients1[0].count}, Current: {ingredients2[0].count}'
    game.set_research(Technology.Logistics)
    game.sleep(10)
    n_game_state = GameState.from_instance(game.instance)
    game.instance.reset(n_game_state)
    game.sleep(5)
    n_game_state = GameState.from_instance(game.instance)
    game.instance.reset(n_game_state)
    pass

