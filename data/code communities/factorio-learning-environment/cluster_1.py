# Cluster 1

def main():
    SERPER_KEYS = os.getenv('SERPER_KEYS')
    SERPER_KEYS = SERPER_KEYS.split(',')
    num_workers_per_key = 2
    total_workers = len(SERPER_KEYS) * num_workers_per_key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    search_queries = ['beginner guide first steps', 'getting started tutorial', 'first hour guide', 'basic controls tutorial', 'survival basics guide', 'early game tips tricks', 'coal mining guide early', 'iron mining setup', 'copper mining basics', 'stone mining early game', 'manual resource gathering', 'starting resource management', 'burner inserter setup', 'first automation steps', 'basic mining drill layout', 'steam power setup', 'coal power generation', 'basic inserter patterns', 'early smelting setup', 'red science automation', 'green science setup', 'early research guide', 'science pack production', 'research priorities guide', 'laboratory setup guide', 'belt layout guide', 'early game logistics', 'transport belt basics', 'inserter mechanics guide', 'chest usage tutorial', 'underground belt guide', 'splitter tutorial basic', 'basic power setup', 'steam engine layout', 'boiler configuration', 'power management early', 'electricity network guide', 'power pole placement', 'early game defense', 'basic military setup', 'turret placement guide', 'ammunition production', 'wall construction guide', 'biter defense basics', 'early game military research', 'basic factory layout', 'early production lines', 'assembling machine setup', 'manufacturing priorities', 'component production guide', 'intermediate products guide', 'basic ore processing', 'early smelting layouts', 'furnace setup guide', 'plate production tutorial', 'smelting column design', 'basic fluid handling', 'pipe systems tutorial', 'offshore pump setup', 'early oil processing', 'fluid storage basics', 'pump mechanics guide', 'main bus concept', 'early base organization', 'factory planning guide', 'basic ratios tutorial', 'production efficiency', 'base layout principles', 'base expansion guide', 'resource outpost setup', 'early trains tutorial', 'expanding production guide', 'scaling up basics', 'electronic circuit production', 'iron gear wheel automation', 'copper cable manufacturing', 'steel production setup', 'basic materials flow', 'bottleneck solutions early', 'production backup fixes', 'power shortage guide', 'resource management tips', 'common mistakes avoid', 'troubleshooting guide early', 'early game optimization', 'basic factory ratios', 'production efficiency guide', 'resource balancing tips', 'throughput optimization', 'early automation tips', 'manual crafting guide', 'inventory management', 'quickbar setup guide', 'hotkey optimization', 'tech tree progression', 'research order guide', 'milestone planning', 'advancement strategy', 'progress benchmarks', 'first automation priority', 'automating coal mining', 'basic inserter chains', 'burner phase automation', 'crafting queue efficiency', 'manual to automated transition', 'iron plate automation', 'copper plate production line', 'basic materials flow', 'starter base layout', 'initial factory setup', 'early game ratios', 'automated mining setup', 'coal power sustainability', 'ore patch efficiency', 'starting miners layout', 'resource field optimization', 'initial mining outpost', 'early power scaling', 'coal supply automation', 'power consumption planning', 'backup power systems', 'power grid layout', 'electricity management tips', 'initial belt systems', 'early sorting methods', 'basic item routing', 'starter bus design', 'item balancing early', 'material distribution', 'automated red science', 'science pack scaling', 'research automation', 'lab feeding setup', 'science production ratio', 'research facility layout', 'automated ammo production', 'turret feeding systems', 'military supply chain', 'defensive production', 'automated wall building', 'military automation priority', 'gear wheel automation', 'circuit production line', 'inserter manufacturing', 'belt production setup', 'automated components', 'intermediate products flow', 'starter factory layout', 'production block design', 'early game spacing', 'factory organization', 'modular design basics', 'scalable layouts early', 'coal distribution system', 'ore processing layout', 'plate balancing setup', 'resource priority system', 'material overflow handling', 'resource buffer design', 'assembly line basics', 'production cell design', 'manufacturing blocks', 'automated crafting setup', 'production flow design', 'assembly machine layout', 'belt balancing early', 'underground belt usage', 'splitter arrangements', 'belt compression tips', 'belt priority system', 'logistics optimization', 'steam engine array', 'boiler automation', 'coal power layout', 'power pole coverage', 'early grid design', 'power distribution', 'initial base planning', 'expansion preparation', 'growth bottlenecks', 'factory scaling tips', 'base organization', 'design principles early', 'early efficiency tips', 'production bottlenecks', 'throughput optimization', 'resource efficiency', 'automation priorities', 'system bottlenecks', 'common automation issues', 'production line fixes', 'power system problems', 'belt backup solutions', 'inserter timing fixes', 'resource starvation', 'factory efficiency', 'automation upgrades', 'production speed tips', 'system improvements', 'optimization methods', 'performance enhancement', 'automation milestones', 'production goals', 'development stages', 'expansion timing', 'upgrade priorities', 'advancement planning']
    search_queries = random.sample(search_queries, len(search_queries))
    query_chunks = split_queries(search_queries, total_workers)
    worker_args = []
    for key_idx, serper_key in enumerate(SERPER_KEYS):
        for worker_num in range(num_workers_per_key):
            worker_id = key_idx * num_workers_per_key + worker_num
            if worker_id < len(query_chunks):
                worker_args.append((query_chunks[worker_id], serper_key, openai_api_key, worker_id))
    with Pool(processes=total_workers) as pool:
        pool.map(crawler_worker, worker_args)

def split_queries(queries: List[str], num_splits: int) -> List[List[str]]:
    """Split queries into approximately equal chunks"""
    chunk_size = math.ceil(len(queries) / num_splits)
    return [queries[i:i + chunk_size] for i in range(0, len(queries), chunk_size)]

def create_factorio_instances() -> List[FactorioInstance]:
    """Create Factorio instances in parallel from local servers"""

    def init_instance(params: Tuple[str, int, int]) -> FactorioInstance:
        ip, udp_port, tcp_port = params
        instance = FactorioInstance(address=ip, tcp_port=tcp_port, bounding_box=200, fast=True, cache_scripts=False, inventory={})
        instance.speed(100)
        return instance
    ips, udp_ports, tcp_ports = get_local_container_ips()
    with futures.ThreadPoolExecutor() as executor:
        return list(executor.map(init_instance, zip(ips, udp_ports, tcp_ports)))

def game_instance():
    try:
        instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, cache_scripts=False, fast=True, inventory={})
        instance.set_speed(20)
        return instance
    except Exception as e:
        raise e

def initiate_task_configs(input_task):
    if input_task['task_type'] == 'populated_lab_play':
        input_task['config']['starting_inventory'] = LAB_PLAY_POPULATED_STARTING_INVENTORY
        return ThroughputTask(**input_task['config'])
    task_config = ThroughputTask(**input_task['config'])
    return task_config

def initiate_executor(config, instances, version, db_client, version_description, api_factory, formatter):
    executor = config['executor'](instances=instances, version=version, db_client=db_client, version_description=version_description, api_factory=api_factory, config=config['config'], formatter=formatter)
    return executor

def initialise_starting_state(instance, task, reset_game_state):
    instance.reset(reset_game_state)
    task.setup(instance)
    return task

def plot_throughput_timeseries(data, file_path, task):
    throughput_entity = task.throughput_entity
    data = data['results'][0]
    data_to_plot = {}
    for key, value in data.items():
        values = [x['holdout_achievements']['dynamic'].get(throughput_entity, 0) for x in value]
        data_to_plot[key] = values
    plt.figure(figsize=(10, 6))
    for series_name, values in data_to_plot.items():
        plt.plot(values, label=series_name, marker='o')
    plt.title(f'{throughput_entity} throughput ')
    plt.xlabel('Steps')
    plt.ylabel(f'{throughput_entity}/20s')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)

def plot_throughput_timeseries_mean(input_data, file_path, task):
    throughput_entity = task.throughput_entity
    input_data = input_data['results'][0]
    data = {}
    for key, value in input_data.items():
        values = [x['holdout_achievements']['dynamic'].get(f'{throughput_entity}', 0) for x in value]
        data[key] = values
    values = np.array(list(data.values()))
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    time_points = np.arange(len(mean))
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, mean, 'k-', linewidth=2, label='Mean')
    plt.fill_between(time_points, mean - 2 * std, mean + 2 * std, color='gray', alpha=0.2, label='±2 STD')
    plt.title(f'{throughput_entity} throughput ')
    plt.xlabel('Steps')
    plt.ylabel(f'{throughput_entity}/20s')
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(file_path)

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

