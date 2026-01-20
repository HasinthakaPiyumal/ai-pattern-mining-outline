# Cluster 167

class SingleMachineParallelExecutor(WorkerPool):
    """
    This worker distributes all tasks across multiple threads on this machine.
    """

    def __init__(self, use_process_pool: bool=False, max_workers: Optional[int]=None):
        """
        Create worker with limited threads.
        :param use_process_pool: if true, ProcessPoolExecutor will be used as executor, otherwise ThreadPoolExecutor.
        :param max_workers: if available, use this number as used number of threads.
        """
        number_of_cpus_per_node = max_workers if max_workers else WorkerResources.current_node_cpu_count()
        super().__init__(WorkerResources(number_of_nodes=1, number_of_cpus_per_node=number_of_cpus_per_node, number_of_gpus_per_node=0))
        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) if use_process_pool else concurrent.futures.ThreadPoolExecutor(max_workers=number_of_cpus_per_node)

    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool=False) -> List[Any]:
        """Inherited, see superclass."""
        return list(tqdm(self._executor.map(task.fn, *item_lists), leave=False, total=get_max_size_of_arguments(*item_lists), desc='SingleMachineParallelExecutor', disable=not verbose))

    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future[Any]:
        """Inherited, see superclass."""
        return self._executor.submit(task.fn, *args, **kwargs)

def initialize_ray(master_node_ip: Optional[str]=None, threads_per_node: Optional[int]=None, local_mode: bool=False, log_to_driver: bool=True, use_distributed: bool=False) -> WorkerResources:
    """
    Initialize ray worker.
    ENV_VAR_MASTER_NODE_IP="master node IP".
    ENV_VAR_MASTER_NODE_PASSWORD="password to the master node".
    ENV_VAR_NUM_NODES="number of nodes available".
    :param master_node_ip: if available, ray will connect to remote cluster.
    :param threads_per_node: Number of threads to use per node.
    :param log_to_driver: If true, the output from all of the worker
            processes on all nodes will be directed to the driver.
    :param local_mode: If true, the code will be executed serially. This
            is useful for debugging.
    :param use_distributed: If true, and the env vars are available,
            ray will launch in distributed mode
    :return: created WorkerResources.
    """
    env_var_master_node_ip = 'ip_head'
    env_var_master_node_password = 'redis_password'
    env_var_num_nodes = 'num_nodes'
    number_of_cpus_per_node = threads_per_node if threads_per_node else cpu_count(logical=True)
    number_of_gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if not number_of_gpus_per_node:
        logger.info('Not using GPU in ray')
    if master_node_ip and use_distributed:
        logger.info(f'Connecting to cluster at: {master_node_ip}!')
        ray.init(address=f'ray://{master_node_ip}:10001', local_mode=local_mode, log_to_driver=log_to_driver)
        number_of_nodes = 1
    elif env_var_master_node_ip in os.environ and use_distributed:
        number_of_nodes = int(os.environ[env_var_num_nodes])
        master_node_ip = os.environ[env_var_master_node_ip].split(':')[0]
        redis_password = os.environ[env_var_master_node_password].split(':')[0]
        logger.info(f'Connecting as part of a cluster at: {master_node_ip} with password: {redis_password}!')
        ray.init(address='auto', _node_ip_address=master_node_ip, _redis_password=redis_password, log_to_driver=log_to_driver, local_mode=local_mode)
    else:
        number_of_nodes = 1
        logger.info('Starting ray local!')
        ray.init(num_cpus=number_of_cpus_per_node, dashboard_host='0.0.0.0', local_mode=local_mode, log_to_driver=log_to_driver)
    return WorkerResources(number_of_nodes=number_of_nodes, number_of_cpus_per_node=number_of_cpus_per_node, number_of_gpus_per_node=number_of_gpus_per_node)

class RayDistributed(WorkerPool):
    """
    This worker uses ray to distribute work across all available threads.
    """

    def __init__(self, master_node_ip: Optional[str]=None, threads_per_node: Optional[int]=None, debug_mode: bool=False, log_to_driver: bool=True, output_dir: Optional[Union[str, Path]]=None, logs_subdir: Optional[str]='logs', use_distributed: bool=False):
        """
        Initialize ray worker.
        :param master_node_ip: if available, ray will connect to remote cluster.
        :param threads_per_node: Number of threads to use per node.
        :param debug_mode: If true, the code will be executed serially. This
            is useful for debugging.
        :param log_to_driver: If true, the output from all of the worker
                processes on all nodes will be directed to the driver.
        :param output_dir: Experiment output directory.
        :param logs_subdir: Subdirectory inside experiment dir to store worker logs.
        :param use_distributed: Boolean flag to explicitly enable/disable distributed computation
        """
        self._master_node_ip = master_node_ip
        self._threads_per_node = threads_per_node
        self._local_mode = debug_mode
        self._log_to_driver = log_to_driver
        self._log_dir: Optional[Path] = Path(output_dir) / (logs_subdir or '') if output_dir is not None else None
        self._use_distributed = use_distributed
        super().__init__(self.initialize())

    def initialize(self) -> WorkerResources:
        """
        Initialize ray.
        :return: created WorkerResources.
        """
        if ray.is_initialized():
            logger.warning('Ray is running, we will shut it down before starting again!')
            ray.shutdown()
        return initialize_ray(master_node_ip=self._master_node_ip, threads_per_node=self._threads_per_node, local_mode=self._local_mode, log_to_driver=self._log_to_driver, use_distributed=self._use_distributed)

    def shutdown(self) -> None:
        """
        Shutdown the worker and clear memory.
        """
        ray.shutdown()

    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool=False) -> List[Any]:
        """Inherited, see superclass."""
        del verbose
        return ray_map(task, *item_lists, log_dir=self._log_dir)

    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future[Any]:
        """Inherited, see superclass."""
        remote_fn = ray.remote(task.fn).options(num_gpus=task.num_gpus, num_cpus=task.num_cpus)
        object_ids: ray._raylet.ObjectRef = remote_fn.remote(*args, **kwargs)
        return object_ids.future()

class Sequential(WorkerPool):
    """
    This function does execute all functions sequentially.
    """

    def __init__(self) -> None:
        """
        Initialize simple sequential worker.
        """
        super().__init__(WorkerResources(number_of_nodes=1, number_of_cpus_per_node=1, number_of_gpus_per_node=0))

    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool=False) -> List[Any]:
        """Inherited, see superclass."""
        if task.num_cpus not in [None, 1]:
            raise ValueError(f'Expected num_cpus to be 1 or unset for Sequential worker, got {task.num_cpus}')
        output = [task.fn(*args) for args in tqdm(zip(*item_lists), leave=False, total=get_max_size_of_arguments(*item_lists), desc='Sequential', disable=not verbose)]
        return output

    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future[Any]:
        """Inherited, see superclass."""
        raise NotImplementedError

class TestWorkerPool(unittest.TestCase):
    """Unittest class for WorkerPool"""

    def setUp(self) -> None:
        """
        Setup worker
        """
        self.worker = RayDistributed(debug_mode=True)

    def test_ray(self) -> None:
        """
        Test ray GPU allocation
        """
        num_calls = 3
        num_gpus = 1
        output = self.worker.map(Task(fn=function_to_load_model, num_gpus=num_gpus), num_calls * [1])
        for gpu_available, num_threads in output:
            self.assertTrue(gpu_available)
            self.assertGreater(num_threads, 0)

class TestWorkerPool(unittest.TestCase):
    """Unittest class for WorkerPool"""

    def setUp(self) -> None:
        """Set up basic config."""
        self.lhs_matrix: npt.NDArray[np.float32] = np.array([[1, 2, 4], [2, 3, 4]])
        self.rhs_matrix: npt.NDArray[np.float32] = np.array([[2, 3, 4], [2, 5, 4]]).T
        self.target: npt.NDArray[np.float32] = np.array([[24, 28], [29, 35]])
        self.workers = [Sequential(), RayDistributed(debug_mode=True), SingleMachineParallelExecutor(), SingleMachineParallelExecutor(use_process_pool=True)]

    def test_task(self) -> None:
        """Test Task whether a function can be called"""

        def add_inputs(input1: float, input2: float) -> float:
            """
            :return: input1 + input2 + 1
            """
            return input1 + input2 + 1
        task = Task(fn=add_inputs)
        self.assertEqual(task(10, 20), 31)

    def test_workers(self) -> None:
        """Tests the sequential worker."""
        for worker in self.workers:
            if not isinstance(worker, Sequential):
                self.check_worker_submit(worker)
            self.check_worker_map(worker)

    def check_worker_map(self, worker: WorkerPool) -> None:
        """
        Check whether worker.map passes all checks.
        :param worker: to be tested.
        """
        task = Task(fn=matrix_multiplication)
        result = worker.map(task, self.lhs_matrix, self.rhs_matrix)
        self.assertEqual(len(result), 1)
        self.validate_result(result)
        number_of_functions = 10
        result = worker.map(task, [self.lhs_matrix] * number_of_functions, self.rhs_matrix)
        self.assertEqual(len(result), number_of_functions)
        self.validate_result(result)
        result = worker.map(task, self.lhs_matrix, [self.rhs_matrix] * number_of_functions)
        self.assertEqual(len(result), number_of_functions)
        self.validate_result(result)
        result = worker.map(task, [self.lhs_matrix] * number_of_functions, [self.rhs_matrix] * number_of_functions)
        self.assertEqual(len(result), number_of_functions)
        self.validate_result(result)

    def check_worker_submit(self, worker: WorkerPool) -> None:
        """
        Check whether worker.submit passes all checks
        :param worker: to be tested
        """
        task = Task(fn=matrix_multiplication)
        result = worker.submit(task, self.lhs_matrix, self.rhs_matrix).result()
        self.assertTrue((result == self.target).all())

    def validate_result(self, results: List[npt.NDArray[np.float32]]) -> None:
        """
        Validate that result from np.dot matched expectations
        :param results: List of results from worker
        """
        for result in results:
            self.assertTrue((result == self.target).all())

    def test_splitter(self) -> None:
        """
        Test chunk splitter
        """
        num_chunks = 10
        chunks = chunk_list([1] * num_chunks, num_chunks)
        self.assertEqual(len(chunks), num_chunks)
        chunks = chunk_list([1, 2, 3, 4, 5], 2)
        self.assertEqual(len(chunks), 2)

def scale_cfg_for_distributed_training(cfg: DictConfig, datamodule: pl.LightningDataModule, worker: WorkerPool) -> DictConfig:
    """
    Adjusts parameters in cfg for ddp.
    :param cfg: Config with parameters for instantiation.
    :param datamodule: Datamodule which will be used for updating the lr_scheduler parameters.
    :return cfg: Updated config.
    """
    OmegaConf.set_struct(cfg, False)
    cfg = update_distributed_optimizer_config(cfg)
    if 'lr_scheduler' in cfg:
        num_train_samples = int(len(datamodule._splitter.get_train_samples(datamodule._all_samples, worker)) * datamodule._train_fraction)
        cfg = update_distributed_lr_scheduler_config(cfg=cfg, num_train_batches=num_train_samples // cfg.data_loader.params.batch_size)
    OmegaConf.set_struct(cfg, True)
    logger.info('Optimizer and LR Scheduler configs updated according to ddp strategy.')
    return cfg

@lru_cache(maxsize=1)
def get_test_nuplan_scenario_builder() -> NuPlanScenarioBuilder:
    """Get a nuPlan scenario builder object with default settings to be used in testing."""
    return NuPlanScenarioBuilder(data_root=NUPLAN_DATA_ROOT, map_root=NUPLAN_MAPS_ROOT, sensor_root=NUPLAN_SENSOR_ROOT, db_files=NUPLAN_DB_FILES, map_version=NUPLAN_MAP_VERSION)

class TestDataloaderSequential(SkeletonTestDataloader):
    """
    Tests data loading functionality in a sequential manner.
    """

    def test_dataloader_nuplan_sequential(self) -> None:
        """
        Test dataloader using nuPlan DB using a sequential worker.
        """
        self._test_dataloader(Sequential())

class TestDataloaderRay(SkeletonTestDataloader):
    """
    Tests data loading functionality in ray.
    """

    def test_dataloader_nuplan_ray(self) -> None:
        """
        Test dataloader using nuPlan DB.
        """
        self._test_dataloader(RayDistributed())

def run_simulation(cfg: DictConfig, planners: Optional[Union[AbstractPlanner, List[AbstractPlanner]]]=None) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Helper function for main to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    :param planners: Pre-built planner(s) to run in simulation. Can either be a single planner or list of planners.
    """
    pl.seed_everything(cfg.seed, workers=True)
    profiler_name = 'building_simulation'
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)
    callbacks_worker_pool = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(cfg=cfg, output_dir=common_builder.output_dir, worker=callbacks_worker_pool)
    if planners and 'planner' in cfg.keys():
        logger.info('Using pre-instantiated planner. Ignoring planner in config')
        OmegaConf.set_struct(cfg, False)
        cfg.pop('planner')
        OmegaConf.set_struct(cfg, True)
    if isinstance(planners, AbstractPlanner):
        planners = [planners]
    runners = build_simulations(cfg=cfg, callbacks=callbacks, worker=common_builder.worker, pre_built_planners=planners, callbacks_worker=callbacks_worker_pool)
    if common_builder.profiler:
        common_builder.profiler.save_profiler(profiler_name)
    logger.info('Running simulation...')
    run_runners(runners=runners, common_builder=common_builder, cfg=cfg, profiler_name='running_simulation')
    logger.info('Finished running simulation!')

def build_callbacks_worker(cfg: DictConfig) -> Optional[WorkerPool]:
    """
    Builds workerpool for callbacks.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Workerpool, or None if we'll run without one.
    """
    if not is_target_type(cfg.worker, Sequential) or cfg.disable_callback_parallelization:
        return None
    if cfg.number_of_cpus_allocated_per_simulation not in [None, 1]:
        raise ValueError('Expected `number_of_cpus_allocated_per_simulation` to be set to 1 with Sequential worker.')
    max_workers = min(WorkerResources.current_node_cpu_count() - (cfg.number_of_cpus_allocated_per_simulation or 1), cfg.max_callback_workers)
    callbacks_worker_pool = SingleMachineParallelExecutor(use_process_pool=True, max_workers=max_workers)
    return callbacks_worker_pool

def build_simulation_callbacks(cfg: DictConfig, output_dir: pathlib.Path, worker: Optional[WorkerPool]=None) -> List[AbstractCallback]:
    """
    Builds callback.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param output_dir: directory for all experiment results.
    :param worker: to run certain callbacks in the background (everything runs in main process if None).
    :return: List of callbacks.
    """
    logger.info('Building AbstractCallback...')
    callbacks = []
    for config in cfg.callback.values():
        if is_target_type(config, SerializationCallback):
            callback: SerializationCallback = instantiate(config, output_directory=output_dir)
        elif is_target_type(config, TimingCallback):
            tensorboard = torch.utils.tensorboard.SummaryWriter(log_dir=output_dir)
            callback = instantiate(config, writer=tensorboard)
        elif is_target_type(config, SimulationLogCallback) or is_target_type(config, MetricCallback):
            continue
        else:
            callback = instantiate(config)
        validate_type(callback, AbstractCallback)
        callbacks.append(callback)
    logger.info(f'Building AbstractCallback: {len(callbacks)}...DONE!')
    return callbacks

def _build_planner(planner_cfg: DictConfig, scenario: Optional[AbstractScenario]) -> AbstractPlanner:
    """
    Instantiate planner
    :param planner_cfg: config of a planner
    :param scenario: scenario
    :return AbstractPlanner
    """
    config = planner_cfg.copy()
    if is_target_type(planner_cfg, MLPlanner):
        torch_module_wrapper = build_torch_module_wrapper(planner_cfg.model_config)
        model = LightningModuleWrapper.load_from_checkpoint(planner_cfg.checkpoint_path, model=torch_module_wrapper).model
        OmegaConf.set_struct(config, False)
        config.pop('model_config')
        config.pop('checkpoint_path')
        OmegaConf.set_struct(config, True)
        planner: AbstractPlanner = instantiate(config, model=model)
    else:
        planner_cls: Type[AbstractPlanner] = _locate(config._target_)
        if planner_cls.requires_scenario:
            assert scenario is not None, f'Scenario was not provided to build the planner. Planner {config} can not be build!'
            planner = cast(AbstractPlanner, instantiate(config, scenario=scenario))
        else:
            planner = cast(AbstractPlanner, instantiate(config))
    return planner

def is_target_type(cfg: DictConfig, target_type: Union[Type[Any], Callable[..., Any]]) -> bool:
    """
    Check whether the config's resolved type matches the target type or callable.
    :param cfg: config
    :param target_type: Type or callable to check against.
    :return: Whether cfg._target_ matches the target_type.
    """
    return bool(_locate(cfg._target_) == target_type)

def build_planners(planner_cfg: DictConfig, scenario: Optional[AbstractScenario]) -> List[AbstractPlanner]:
    """
    Instantiate multiple planners by calling build_planner
    :param planners_cfg: planners config
    :param scenario: scenario
    :return planners: List of AbstractPlanners
    """
    return [_build_planner(planner, scenario) for planner in planner_cfg.values()]

def _instantiate_main_lr_scheduler(optimizer: Optimizer, lr_scheduler_cfg: DictConfig, lr: float) -> Dict[str, Any]:
    """
    Instantiates the main learning rate scheduler to be used during training.
    :param optimizer: Optimizer used for training.
    :param lr_scheduler_cfg: Learning rate scheduler config
    :param lr: Learning rate to be used. If using OneCycleLR, then this is the maximum learning rate to be reached during training.
    :return: Learning rate scheduler and associated parameters
    """
    lr_scheduler_params: Dict[str, Any] = {}
    if is_target_type(lr_scheduler_cfg, OneCycleLR):
        lr_scheduler_cfg.max_lr = lr
        lr_scheduler_params['interval'] = 'step'
        frequency_of_lr_scheduler_step = lr_scheduler_cfg.epochs
        lr_scheduler_params['frequency'] = frequency_of_lr_scheduler_step
        logger.info(f'lr_scheduler.step() will be called every {frequency_of_lr_scheduler_step} batches')
    lr_scheduler: _LRScheduler = instantiate(config=lr_scheduler_cfg, optimizer=optimizer)
    lr_scheduler_params['scheduler'] = lr_scheduler
    return lr_scheduler_params

def build_observations(observation_cfg: DictConfig, scenario: AbstractScenario) -> AbstractObservation:
    """
    Instantiate observations
    :param observation_cfg: config of a planner
    :param scenario: scenario
    :return AbstractObservation
    """
    if is_TorchModuleWrapper_config(observation_cfg):
        torch_module_wrapper = build_torch_module_wrapper(observation_cfg.model_config)
        model = LightningModuleWrapper.load_from_checkpoint(observation_cfg.checkpoint_path, model=torch_module_wrapper).model
        config = observation_cfg.copy()
        OmegaConf.set_struct(config, False)
        config.pop('model_config')
        config.pop('checkpoint_path')
        OmegaConf.set_struct(config, True)
        observation: AbstractObservation = instantiate(config, model=model, scenario=scenario)
    else:
        observation = cast(AbstractObservation, instantiate(observation_cfg, scenario=scenario))
    return observation

def is_TorchModuleWrapper_config(cfg: DictConfig) -> bool:
    """
    Check whether the config is meant for a TorchModuleWrapper
    :param cfg: config
    :return: True if model_config and checkpoint_path is in the cfg, False otherwise
    """
    return 'model_config' in cfg and 'checkpoint_path' in cfg

def find_builder_in_config(cfg: DictConfig, desired_type: Type[Any]) -> DictConfig:
    """
    Find the corresponding config for the desired builder
    :param cfg: config structured as a dictionary
    :param desired_type: desired builder type
    :return: found config
    @raise ValueError if the config cannot be found for the builder
    """
    for cfg_builder in cfg.values():
        if is_target_type(cfg_builder, desired_type):
            return cast(DictConfig, cfg_builder)
    raise ValueError(f'Config does not exist for builder type: {desired_type}!')

def update_distributed_optimizer_config(cfg: DictConfig) -> DictConfig:
    """
    Scale the learning rate according to scaling method provided in distributed setting with ddp strategy.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return cfg: DictConfig. Updated configuration that is used to run the experiment.
    """
    lr_scale = get_num_gpus_used(cfg)
    logger.info(f'World size: {lr_scale}')
    logger.info(f'Learning rate before: {cfg.optimizer.lr}')
    scaling_method = 'Equal Variance' if cfg.lightning.distributed_training.equal_variance_scaling_strategy else 'Linearly'
    logger.info(f'Scaling method: {scaling_method}')
    cfg.optimizer.lr = scale_parameter(parameter=cfg.optimizer.lr, world_size=lr_scale, equal_variance_scaling_strategy=cfg.lightning.distributed_training.equal_variance_scaling_strategy)
    if is_target_type(cfg.optimizer, torch.optim.Adam) or is_target_type(cfg.optimizer, torch.optim.AdamW):
        cfg.optimizer.betas[0] = scale_parameter(parameter=cfg.optimizer.betas[0], world_size=lr_scale, equal_variance_scaling_strategy=cfg.lightning.distributed_training.equal_variance_scaling_strategy, raise_power=True)
        cfg.optimizer.betas[1] = scale_parameter(parameter=cfg.optimizer.betas[1], world_size=lr_scale, equal_variance_scaling_strategy=cfg.lightning.distributed_training.equal_variance_scaling_strategy, raise_power=True)
        logger.info(f'Betas after scaling: {cfg.optimizer.betas}')
    logger.info(f'Learning rate after scaling: {cfg.optimizer.lr}')
    return cfg

def update_distributed_lr_scheduler_config(cfg: DictConfig, num_train_batches: int) -> DictConfig:
    """
    Updates the learning rate scheduler config that modifies optimizer parameters over time.
    Optimizer and LR Scheduler is built in configure_optimizers() methods of the model.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param num_train_batches: Number of batches in train dataloader.
    :return cfg: Configuration with the updated lr_scheduler key.
    """
    logger.info('Updating Learning Rate Scheduler Config...')
    number_gpus = get_num_gpus_used(cfg)
    if is_target_type(cfg.lr_scheduler, OneCycleLR):
        enable_overfitting = cfg.lightning.trainer.overfitting.enable
        overfit_batches = cfg.lightning.trainer.overfitting.params.overfit_batches
        if enable_overfitting and overfit_batches != 0:
            if overfit_batches >= 1.0:
                num_train_batches = overfit_batches
            else:
                num_train_batches = math.ceil(num_train_batches * overfit_batches)
        cfg.lr_scheduler.steps_per_epoch = scale_oclr_steps_per_epoch(num_train_batches=num_train_batches, world_size=number_gpus, epochs=cfg.lightning.trainer.params.max_epochs, warm_up_steps=cfg.warm_up_lr_scheduler.lr_lambda.warm_up_steps if 'warm_up_lr_scheduler' in cfg else 0)
        logger.info(f'Updating learning rate scheduler config Completed. Using {cfg.lr_scheduler._target_}.')
    else:
        logger.info(f'Updating {cfg.lr_scheduler._target_} in ddp setting is not yet supported. Learning rate scheduler config will not be updated.')
    return cfg

class TestUtilsType(unittest.TestCase):
    """Test utils_type functions."""

    def test_is_TorchModuleWrapper_config(self) -> None:
        """Tests that is_TorchModuleWrapper_config works as expected."""
        mock_config = DictConfig({'model_config': 'some_value', 'checkpoint_path': 'some_value', 'some_other_key': 'some_value'})
        expect_true = is_TorchModuleWrapper_config(mock_config)
        self.assertTrue(expect_true)
        mock_config.pop('some_other_key')
        expect_true = is_TorchModuleWrapper_config(mock_config)
        self.assertTrue(expect_true)
        mock_config.pop('model_config')
        expect_false = is_TorchModuleWrapper_config(mock_config)
        self.assertFalse(expect_false)
        mock_config.pop('checkpoint_path')
        expect_false = is_TorchModuleWrapper_config(mock_config)
        self.assertFalse(expect_false)

    def test_is_target_type(self) -> None:
        """Tests that is_target_type works as expected."""
        mock_config_test_utils_mock_type = DictConfig({'_target_': f'{__name__}.TestUtilsTypeMockType'})
        mock_config_test_utils_another_mock_type = DictConfig({'_target_': f'{__name__}.TestUtilsTypeAnotherMockType'})
        expect_true = is_target_type(mock_config_test_utils_mock_type, TestUtilsTypeMockType)
        self.assertTrue(expect_true)
        expect_true = is_target_type(mock_config_test_utils_another_mock_type, TestUtilsTypeAnotherMockType)
        self.assertTrue(expect_true)
        expect_false = is_target_type(mock_config_test_utils_mock_type, TestUtilsTypeAnotherMockType)
        self.assertFalse(expect_false)
        expect_false = is_target_type(mock_config_test_utils_another_mock_type, TestUtilsTypeMockType)
        self.assertFalse(expect_false)

    def test_validate_type(self) -> None:
        """Tests that validate_type works as expected."""
        test_utils_type_mock_type = TestUtilsTypeMockType()
        validate_type(test_utils_type_mock_type, TestUtilsTypeMockType)
        with self.assertRaises(AssertionError):
            validate_type(test_utils_type_mock_type, TestUtilsTypeAnotherMockType)

    def test_are_the_same_type(self) -> None:
        """Tests that are_the_same_type works as expected."""
        test_utils_type_mock_type = TestUtilsTypeMockType()
        another_test_utils_type_mock_type = TestUtilsTypeMockType()
        test_utils_type_another_mock_type = TestUtilsTypeAnotherMockType()
        are_the_same_type(test_utils_type_mock_type, another_test_utils_type_mock_type)
        with self.assertRaises(AssertionError):
            are_the_same_type(test_utils_type_mock_type, test_utils_type_another_mock_type)

    def test_validate_dict_type(self) -> None:
        """Tests that validate_dict_type works as expected."""
        mock_config = DictConfig({'_convert_': 'all', 'correct_object': {'_target_': f'{__name__}.TestUtilsTypeMockType', 'a': 1, 'b': 2.5}, 'correct_object_2': {'_target_': f'{__name__}.TestUtilsTypeMockType', 'a': 1, 'b': 2.5}})
        instantiated_config = hydra.utils.instantiate(mock_config)
        validate_dict_type(instantiated_config, TestUtilsTypeMockType)
        mock_config.other_object = {'_target_': f'{__name__}.TestUtilsTypeAnotherMockType', 'c': 1}
        instantiated_config = hydra.utils.instantiate(mock_config)
        with self.assertRaises(AssertionError):
            validate_dict_type(instantiated_config, TestUtilsTypeMockType)

    def test_find_builder_in_config(self) -> None:
        """Tests that find_builder_in_config works as expected."""
        mock_config = DictConfig({'correct_object': {'_target_': f'{__name__}.TestUtilsTypeMockType', 'a': 1, 'b': 2.5}, 'other_object': {'_target_': f'{__name__}.TestUtilsTypeAnotherMockType', 'c': 1}})
        test_utils_mock_type = find_builder_in_config(mock_config, TestUtilsTypeMockType)
        self.assertTrue(is_target_type(test_utils_mock_type, TestUtilsTypeMockType))
        test_utils_another_mock_type = find_builder_in_config(mock_config, TestUtilsTypeAnotherMockType)
        self.assertTrue(is_target_type(test_utils_another_mock_type, TestUtilsTypeAnotherMockType))
        del mock_config.other_object
        with self.assertRaises(ValueError):
            find_builder_in_config(mock_config, TestUtilsTypeAnotherMockType)

class TestUpdateDistributedTrainingCfg(unittest.TestCase):
    """Test update_distributed_optimizer_config function."""
    world_size = 4

    def setUp(self) -> None:
        """Setup test attributes."""
        self.lr = 1e-05
        self.num_train_batches = 12
        self.batch_size = 2
        self.div_factor = 2
        self.max_lr = 0.01
        self.betas = [0.9, 0.999]
        self.max_epochs = 2
        self.exponential_lr_scheduler_cfg = {'_target_': 'torch.optim.lr_scheduler.ExponentialLR', 'gamma': 0.9, 'steps_per_epoch': None}
        self.one_cycle_lr_scheduler_cfg = {'_target_': 'torch.optim.lr_scheduler.OneCycleLR', 'max_lr': self.max_lr, 'steps_per_epoch': None, 'div_factor': self.div_factor}
        self.cfg_mock = DictConfig({'optimizer': {'_target_': 'torch.optim.Adam', 'lr': self.lr, 'betas': self.betas.copy()}, 'lightning': {'trainer': {'overfitting': {'enable': False, 'params': {'overfit_batches': 1}}, 'params': {'max_epochs': self.max_epochs}}, 'distributed_training': {'equal_variance_scaling_strategy': False}}, 'dataloader': {'params': {'batch_size': self.batch_size}}, 'warm_up_scheduler': {'lr_lambda': {'warm_up_steps': 0.0}}})

    @patch.dict(os.environ, {'WORLD_SIZE': str(world_size)}, clear=True)
    def test_update_distributed_optimizer_config_equal_variance(self) -> None:
        """Test default setting where the lr is scaled to maintain equal variance."""
        cfg_mock = self.cfg_mock.copy()
        cfg_mock.lightning.distributed_training.equal_variance_scaling_strategy = True
        cfg_mock = update_distributed_optimizer_config(cfg_mock)
        msg = f'Expected {self.world_size ** 0.5 * self.lr} but got {cfg_mock.optimizer.lr}'
        msg_beta_1 = f'Expected {self.betas[0]}, {self.world_size ** 0.5}, {self.betas[0] ** self.world_size ** 0.5} but got {cfg_mock.optimizer.betas[0]}'
        msg_beta_2 = f'Expected {self.betas[1] ** self.world_size ** 0.5} but got {cfg_mock.optimizer.betas[1]}'
        self.assertAlmostEqual(float(cfg_mock.optimizer.lr), self.world_size ** 0.5 * self.lr, msg=msg)
        self.assertAlmostEqual(float(cfg_mock.optimizer.betas[0]), self.betas[0] ** self.world_size ** 0.5, msg=msg_beta_1)
        self.assertAlmostEqual(float(cfg_mock.optimizer.betas[1]), self.betas[1] ** self.world_size ** 0.5, msg=msg_beta_2)

    @patch.dict(os.environ, {'WORLD_SIZE': str(world_size)}, clear=True)
    def test_update_distributed_optimizer_config_linearly(self) -> None:
        """Test default setting where the lr is scaled linearly."""
        cfg_mock = self.cfg_mock.copy()
        cfg_mock = update_distributed_optimizer_config(cfg_mock)
        msg = f'Expected {self.world_size * self.lr} but got {cfg_mock.optimizer.lr}'
        msg_beta_1 = f'Expected {self.betas[0] ** self.world_size} but got {cfg_mock.optimizer.betas[0]}'
        msg_beta_2 = f'Expected {self.betas[1] ** self.world_size} but got {cfg_mock.optimizer.betas[1]}'
        self.assertAlmostEqual(float(cfg_mock.optimizer.lr), self.world_size * self.lr, msg=msg)
        self.assertAlmostEqual(float(cfg_mock.optimizer.betas[0]), self.betas[0] ** self.world_size, msg=msg_beta_1)
        self.assertAlmostEqual(float(cfg_mock.optimizer.betas[1]), self.betas[1] ** self.world_size, msg=msg_beta_2)

    @patch.dict(os.environ, {'WORLD_SIZE': str(world_size)}, clear=True)
    def test_update_distributed_lr_scheduler_config_not_one_cycle_lr(self) -> None:
        """
        Test default setting where the lr_scheduler is not supported.
        Currently, anything other than OneCycleLR is not supported.
        """
        cfg_mock = self.cfg_mock.copy()
        cfg_mock.lr_scheduler = self.exponential_lr_scheduler_cfg.copy()
        cfg_mock.lightning.trainer.overfitting.enable = True
        cfg_mock.lightning.trainer.overfitting.params.overfit_batches = 1
        cfg_mock = update_distributed_lr_scheduler_config(cfg_mock, num_train_batches=self.num_train_batches)
        msg_steps_per_epoch = f'Expected Mock to not be edited, but steps_per_epoch was edited: steps_per_epoch is {cfg_mock.lr_scheduler.steps_per_epoch}'
        self.assertIsNone(cfg_mock.lr_scheduler.steps_per_epoch, msg=msg_steps_per_epoch)

    @patch.dict(os.environ, {'WORLD_SIZE': str(world_size)}, clear=True)
    def test_update_distributed_lr_scheduler_config_oclr_overfit_zero_batches(self) -> None:
        """Test default setting where the overfit_batches parameter is set to 0."""
        cfg_mock = self.cfg_mock.copy()
        cfg_mock.lr_scheduler = self.one_cycle_lr_scheduler_cfg.copy()
        cfg_mock.lightning.trainer.overfitting.enable = True
        cfg_mock.lightning.trainer.overfitting.params.overfit_batches = 0
        cfg_mock = update_distributed_lr_scheduler_config(cfg_mock, num_train_batches=self.num_train_batches)
        expected_steps_per_epoch = math.ceil(math.ceil(self.num_train_batches / self.world_size) / self.max_epochs)
        msg_steps_per_epoch = f'Expected steps per epoch to be {expected_steps_per_epoch} but got {cfg_mock.lr_scheduler.steps_per_epoch}'
        self.assertEqual(cfg_mock.lr_scheduler.steps_per_epoch, expected_steps_per_epoch, msg=msg_steps_per_epoch)

    @patch.dict(os.environ, {'WORLD_SIZE': str(world_size)}, clear=True)
    def test_update_distributed_lr_scheduler_config_overfit_one_batches(self) -> None:
        """Test default setting where the overfit_batches parameter is set to 1."""
        cfg_mock = self.cfg_mock.copy()
        cfg_mock.lr_scheduler = self.one_cycle_lr_scheduler_cfg.copy()
        cfg_mock.lightning.trainer.overfitting.enable = True
        cfg_mock.lightning.trainer.overfitting.params.overfit_batches = 1
        cfg_mock = update_distributed_lr_scheduler_config(cfg_mock, num_train_batches=self.num_train_batches)
        expected_steps_per_epoch = math.ceil(cfg_mock.lightning.trainer.overfitting.params.overfit_batches / self.world_size / self.max_epochs)
        msg_steps_per_epoch = f'Expected steps per epoch to be {expected_steps_per_epoch} but got {cfg_mock.lr_scheduler.steps_per_epoch}'
        self.assertEqual(cfg_mock.lr_scheduler.steps_per_epoch, expected_steps_per_epoch, msg=msg_steps_per_epoch)

    @patch.dict(os.environ, {'WORLD_SIZE': str(world_size)}, clear=True)
    def test_update_distributed_lr_scheduler_config_overfit_batches_fractional(self) -> None:
        """Test default setting where the overfit_batches parameter is set to 1."""
        cfg_mock = self.cfg_mock.copy()
        cfg_mock.lr_scheduler = self.one_cycle_lr_scheduler_cfg.copy()
        cfg_mock.lightning.trainer.overfitting.enable = True
        cfg_mock.lightning.trainer.overfitting.params.overfit_batches = 0.5
        cfg_mock = update_distributed_lr_scheduler_config(cfg_mock, num_train_batches=self.num_train_batches)
        batches_to_overfit = math.ceil(self.num_train_batches * cfg_mock.lightning.trainer.overfitting.params.overfit_batches)
        expected_steps_per_epoch = math.ceil(math.ceil(batches_to_overfit / self.world_size) / self.max_epochs)
        msg_steps_per_epoch = f'Expected steps per epoch to be {expected_steps_per_epoch} but got {cfg_mock.lr_scheduler.steps_per_epoch}'
        self.assertEqual(cfg_mock.lr_scheduler.steps_per_epoch, expected_steps_per_epoch, msg=msg_steps_per_epoch)

class TestSimulationCallbackBuilder(unittest.TestCase):
    """Unit tests for functions in simulation_callback_builder.py."""
    mock_cpu_node_count = 4

    @staticmethod
    def _generate_mock_build_callbacks_worker_config(number_of_cpus_allocated_per_simulation: int=1, max_callback_workers: int=1, disable_callback_parallelization: bool=False) -> DictConfig:
        """
        Utility function to generate a mocked callback worker configuration with Sequential worker type. Parameters are
        used directly as the config values.
        """
        return DictConfig({'worker': {'_target_': 'nuplan.planning.utils.multithreading.worker_sequential.Sequential'}, 'number_of_cpus_allocated_per_simulation': number_of_cpus_allocated_per_simulation, 'max_callback_workers': max_callback_workers, 'disable_callback_parallelization': disable_callback_parallelization})

    @staticmethod
    def _calculate_expected_number_of_threads(max_callback_workers: int) -> int:
        """
        Utility function to calculate the expected number of threads available to the workers. The calculation is based on
        the current build_callbacks_worker implementation.
        :param max_callback_workers: Config value passed from Sequential worker config.
        """
        return min(TestSimulationCallbackBuilder.mock_cpu_node_count - 1, max_callback_workers)

    @given(number_of_cpus_allocated_per_simulation=st.one_of(st.none(), st.just(1)), max_callback_workers=st.integers(min_value=1))
    def test_build_callbacks_worker_nominal(self, number_of_cpus_allocated_per_simulation: int, max_callback_workers: int) -> None:
        """Tests the nominal case of build_callbacks_worker."""
        with mock.patch('nuplan.planning.utils.multithreading.worker_pool.WorkerResources.current_node_cpu_count', return_value=self.mock_cpu_node_count):
            mock_config = TestSimulationCallbackBuilder._generate_mock_build_callbacks_worker_config(number_of_cpus_allocated_per_simulation=number_of_cpus_allocated_per_simulation, max_callback_workers=max_callback_workers)
            worker_pool = build_callbacks_worker(mock_config)
            expected_number_of_threads = TestSimulationCallbackBuilder._calculate_expected_number_of_threads(max_callback_workers)
            self.assertEqual(worker_pool.number_of_threads, expected_number_of_threads)
            self.assertTrue(isinstance(worker_pool, SingleMachineParallelExecutor))

    @given(number_of_cpus_allocated_per_simulation=st.one_of(st.integers(max_value=0), st.integers(min_value=2)), max_callback_workers=st.integers(min_value=1))
    def test_build_callbacks_worker_edge_case_invalid_cpus_allocated(self, number_of_cpus_allocated_per_simulation: int, max_callback_workers: int) -> None:
        """Tests an edge case of build_callbacks_worker, where an invalid cpu allocation setting is passed."""
        mock_config = TestSimulationCallbackBuilder._generate_mock_build_callbacks_worker_config(number_of_cpus_allocated_per_simulation=number_of_cpus_allocated_per_simulation, max_callback_workers=max_callback_workers, disable_callback_parallelization=False)
        with self.assertRaises(ValueError):
            build_callbacks_worker(mock_config)

    @given(number_of_cpus_allocated_per_simulation=st.one_of(st.none(), st.just(1)), max_callback_workers=st.integers(min_value=1))
    def test_build_callbacks_worker_edge_cases(self, number_of_cpus_allocated_per_simulation: int, max_callback_workers: int) -> None:
        """Tests other edge cases of build_callbacks_worker."""
        mock_config = TestSimulationCallbackBuilder._generate_mock_build_callbacks_worker_config(number_of_cpus_allocated_per_simulation=number_of_cpus_allocated_per_simulation, max_callback_workers=max_callback_workers, disable_callback_parallelization=False)
        mock_config.worker._target_ = 'nuplan.planning.utils.multithreading.worker_parallel.SingleMachineParallelExecutor'
        worker_pool = build_callbacks_worker(mock_config)
        self.assertIsNone(worker_pool)
        mock_config = TestSimulationCallbackBuilder._generate_mock_build_callbacks_worker_config(number_of_cpus_allocated_per_simulation=number_of_cpus_allocated_per_simulation, max_callback_workers=max_callback_workers, disable_callback_parallelization=True)
        worker_pool = build_callbacks_worker(mock_config)
        self.assertIsNone(worker_pool)

    def test_build_simulation_callbacks_serialization_callback(self) -> None:
        """
        Tests that build_simulation_callbacks returns the expected result when passed SerializationCallback config.
        """
        mock_config = DictConfig({'callback': {'serialization_callback': {'_target_': 'nuplan.planning.simulation.callback.serialization_callback.SerializationCallback', 'folder_name': 'mock_folder', 'serialization_type': 'pickle', 'serialize_into_single_file': False}}})
        callbacks = build_simulation_callbacks(mock_config, Path('/tmp/mock_dir'))
        expected_serialization_callback, *_ = callbacks
        self.assertEqual(1, len(callbacks))
        self.assertTrue(isinstance(expected_serialization_callback, SerializationCallback))

    def test_build_simulation_callbacks_timing_callback(self) -> None:
        """
        Tests that build_simulation_callbacks returns the expected result when passed TimingCallback config.
        """
        mock_config = DictConfig({'callback': {'timing_callback': {'_target_': 'nuplan.planning.simulation.callback.timing_callback.TimingCallback'}}})
        callbacks = build_simulation_callbacks(mock_config, Path('/tmp/mock_dir'))
        expected_timing_callback, *_ = callbacks
        self.assertEqual(1, len(callbacks))
        self.assertTrue(isinstance(expected_timing_callback, TimingCallback))

    def test_build_simulation_callbacks_simulation_log_metric_callbacks(self) -> None:
        """
        Tests that build_simulation_callbacks returns the expected result when passed SimulationLogCallback
        & MetricCallback configurations.
        """
        mock_config = DictConfig({'callback': {'simulation_log_callback': {'_target_': 'nuplan.planning.simulation.callback.simulation_log_callback.SimulationLogCallback'}, 'metric_callback': {'_target_': 'nuplan.planning.simulation.callback.metric_callback.MetricCallback'}}})
        callbacks = build_simulation_callbacks(mock_config, Path('/tmp/mock_dir'))
        self.assertEqual(0, len(callbacks))

    def test_build_simulation_callbacks_multi_callback(self) -> None:
        """
        Tests that build_simulation_callbacks returns the expected result when passed MultiCallback config.
        """
        mock_config = DictConfig({'callback': {'multi_callback': {'_target_': 'nuplan.planning.simulation.callback.multi_callback.MultiCallback', 'callbacks': []}}})
        callbacks = build_simulation_callbacks(mock_config, Path('/tmp/mock_dir'))
        expected_multi_callback, *_ = callbacks
        self.assertEqual(1, len(callbacks))
        self.assertTrue(isinstance(expected_multi_callback, MultiCallback))

    def test_build_simulation_callbacks_visualization_callback(self) -> None:
        """
        Tests that build_simulation_callbacks returns the expected result when passed MultiCallback config.
        """
        mock_config = DictConfig({'callback': {'visualization_callback': {'_target_': 'nuplan.planning.simulation.callback.visualization_callback.VisualizationCallback', 'renderer': {}}}})
        callbacks = build_simulation_callbacks(mock_config, Path('/tmp/mock_dir'))
        expected_visualization_callback, *_ = callbacks
        self.assertEqual(1, len(callbacks))
        self.assertTrue(isinstance(expected_visualization_callback, VisualizationCallback))

class TestRunSimulation(SkeletonTestSimulation):
    """Test running main simulation."""

    def test_run_simulation(self) -> None:
        """
        Sanity test for passing planner as argument to run_simulation
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, 'observation=box_observation', 'ego_controller=log_play_back_controller', 'experiment_name=simulation_test'])
            planner_cfg = cfg.planner
            planner = build_planners(planner_cfg, MockAbstractScenario())
            OmegaConf.set_struct(cfg, False)
            cfg.pop('planner')
            OmegaConf.set_struct(cfg, True)
            run_simulation(cfg, planner)

class TestRunMetricAggregator(SkeletonTestSimulation):
    """Test the run_metric_aggregator script."""

    def test_run_metric_aggregator_without_challenges(self) -> None:
        """Sanity test to run metric_aggregator script without any challenges."""
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, '+simulation=open_loop_boxes', 'experiment_name=simulation_metric_aggregator_test'])
            run_simulation(cfg)
            exp_output_dir = deepcopy(cfg.output_dir)
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=METRIC_AGGREGATOR_CONFIG_NAME, overrides=[f'output_dir={exp_output_dir}', 'scenario_metric_paths=[]', 'metric_aggregator=[default_weighted_average]', 'challenges=[]'])
            run_metric_aggregator(cfg)
            metric_aggregator_output = Path(cfg.aggregator_save_path)
            aggregator_output_file_length = len(list(metric_aggregator_output.rglob('*')))
            self.assertEqual(aggregator_output_file_length, 1)

class TestRunMetric(SkeletonTestSimulation):
    """Test running metrics only."""

    def test_run_simulation_fails_with_no_logs(self) -> None:
        """Sanity test to test that metric_runner fails to run when there is no simulation logs."""
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, '+simulation=open_loop_boxes', f'simulation_log_main_path={self.tmp_dir.name}', 'experiment_name=simulation_no_metric_test'])
            with self.assertRaises(FileNotFoundError):
                run_metric(cfg)

    def test_run_simulation_logs(self) -> None:
        """Sanity test to run simulation logs by computing metrics only."""
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, '+simulation=open_loop_boxes', 'run_metric=false', 'experiment_name=open_loop_boxes/simulation_metric_test', 'worker=sequential', 'main_callback=[time_callback]'])
            run_simulation(cfg)
            exp_output_dir = deepcopy(cfg.output_dir)
            OmegaConf.set_struct(cfg, False)
            cfg.simulation_log_main_path = exp_output_dir
            OmegaConf.set_struct(cfg, True)
            run_metric(cfg)
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=METRIC_AGGREGATOR_CONFIG_NAME, overrides=[f'output_dir={exp_output_dir}', "challenges=['open_loop_boxes']"])
            run_metric_aggregator(cfg)
            metric_aggregator_output = Path(cfg.aggregator_save_path)
            aggregator_output_file_length = len(list(metric_aggregator_output.rglob('*')))
            self.assertEqual(aggregator_output_file_length, 1)

class TestNuPlanScenarioBuilder(unittest.TestCase):
    """
    Tests scenario filtering and construction functionality.
    """

    def test_nuplan_scenario_builder_implements_abstract_scenario_builder(self) -> None:
        """
        Tests that the NuPlanScenarioBuilder implements the AbstractScenarioBuilder interface.
        """
        assert_class_properly_implements_interface(AbstractScenarioBuilder, NuPlanScenarioBuilder)

    def test_get_scenarios_no_filters(self) -> None:
        """
        Tests that the get_scenarios() method functions properly
        With no additional filters applied.
        """

        def db_file_patch(params: GetScenariosFromDbFileParams) -> ScenarioDict:
            """
            A patch for the get_scenarios_from_db_file method that validates the input args.
            """
            self.assertIsNone(params.filter_tokens)
            self.assertIsNone(params.filter_types)
            self.assertIsNone(params.filter_map_names)
            self.assertFalse(params.include_cameras)
            m1 = MockNuPlanScenario(token='a', scenario_type='type1')
            m2 = MockNuPlanScenario(token='b', scenario_type='type1')
            m3 = MockNuPlanScenario(token='c', scenario_type='type2')
            return {'type1': [m1, m2], 'type2': [m3]}

        def discover_log_dbs_patch(load_path: Union[List[str], str]) -> List[str]:
            """
            A patch for the discover_log_dbs method.
            """
            return ['filename']
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.get_scenarios_from_db_file', db_file_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder.discover_log_dbs', discover_log_dbs_patch):
            scenario_builder = NuPlanScenarioBuilder(data_root='foo', map_root='bar', sensor_root='qux', db_files=None, map_version='baz', max_workers=None, verbose=False, scenario_mapping=None, vehicle_parameters=None, include_cameras=False)
            scenario_filter = ScenarioFilter(scenario_types=None, scenario_tokens=None, log_names=None, map_names=None, num_scenarios_per_type=None, limit_total_scenarios=None, expand_scenarios=False, remove_invalid_goals=False, shuffle=False, timestamp_threshold_s=None, ego_displacement_minimum_m=None, ego_start_speed_threshold=None, ego_stop_speed_threshold=None, speed_noise_tolerance=None, token_set_path=None, fraction_in_token_set_threshold=None)
            result = scenario_builder.get_scenarios(scenario_filter, Sequential())
            self.assertEqual(3, len(result))
            result.sort(key=lambda s: s.token)
            self.assertEqual('a', result[0].token)
            self.assertEqual('b', result[1].token)
            self.assertEqual('c', result[2].token)

    def test_get_scenarios_db_filters(self) -> None:
        """
        Tests that the get_scenarios() method functions properly with db filters applied.
        """

        def db_file_patch(params: GetScenariosFromDbFileParams) -> ScenarioDict:
            """
            A patch for the get_scenarios_from_db_file method.
            """
            self.assertEqual(params.filter_tokens, ['a', 'b', 'c', 'd', 'e', 'f'])
            self.assertEqual(params.filter_types, ['type1', 'type2', 'type3'])
            self.assertEqual(params.filter_map_names, ['map1', 'map2'])
            self.assertTrue(params.include_cameras)
            self.assertTrue(params.log_file_absolute_path in ['filename1', 'filename2'])
            m1 = MockNuPlanScenario(token='a', scenario_type='type1')
            m2 = MockNuPlanScenario(token='b', scenario_type='type1')
            m3 = MockNuPlanScenario(token='c', scenario_type='type2')
            return {'type1': [m1, m2], 'type2': [m3]}

        def discover_log_dbs_patch(load_path: Union[List[str], str]) -> List[str]:
            """
            A patch for the discover_log_dbs method.
            """
            return ['filename1', 'filename2', 'filename3']
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.get_scenarios_from_db_file', db_file_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder.discover_log_dbs', discover_log_dbs_patch):
            scenario_builder = NuPlanScenarioBuilder(data_root='foo', map_root='bar', sensor_root='qux', db_files=None, map_version='baz', max_workers=None, verbose=False, scenario_mapping=None, vehicle_parameters=None, include_cameras=True)
            scenario_filter = ScenarioFilter(scenario_types=['type1', 'type2', 'type3'], scenario_tokens=['a', 'b', 'c', 'd', 'e', 'f'], log_names=['filename1', 'filename2'], map_names=['map1', 'map2'], num_scenarios_per_type=None, limit_total_scenarios=None, expand_scenarios=False, remove_invalid_goals=False, shuffle=False, timestamp_threshold_s=None, ego_displacement_minimum_m=None, ego_start_speed_threshold=None, ego_stop_speed_threshold=None, speed_noise_tolerance=None, token_set_path=None, fraction_in_token_set_threshold=None)
            result = scenario_builder.get_scenarios(scenario_filter, Sequential())
            self.assertEqual(6, len(result))
            result.sort(key=lambda s: s.token)
            self.assertEqual('a', result[0].token)
            self.assertEqual('a', result[1].token)
            self.assertEqual('b', result[2].token)
            self.assertEqual('b', result[3].token)
            self.assertEqual('c', result[4].token)
            self.assertEqual('c', result[5].token)

    def test_get_scenarios_num_scenarios_per_type_filter(self) -> None:
        """
        Tests that the get_scenarios() method functions properly
        With a num_scenarios_per_type filter applied.
        """

        def db_file_patch(params: GetScenariosFromDbFileParams) -> ScenarioDict:
            """
            A patch for the get_scenarios_from_db_file method
            """
            self.assertEqual(params.filter_tokens, ['a', 'b', 'c', 'd', 'e', 'f'])
            self.assertEqual(params.filter_types, ['type1', 'type2', 'type3'])
            self.assertEqual(params.filter_map_names, ['map1', 'map2'])
            self.assertEqual(params.include_cameras, False)
            self.assertTrue(params.log_file_absolute_path in ['filename1', 'filename2'])
            m1 = MockNuPlanScenario(token='a', scenario_type='type1')
            m2 = MockNuPlanScenario(token='b', scenario_type='type1')
            m3 = MockNuPlanScenario(token='c', scenario_type='type2')
            return {'type1': [m1, m2], 'type2': [m3]}

        def discover_log_dbs_patch(load_path: Union[List[str], str]) -> List[str]:
            """
            A patch for the discover_log_dbs method
            """
            return ['filename1', 'filename2', 'filename3']
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.get_scenarios_from_db_file', db_file_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder.discover_log_dbs', discover_log_dbs_patch):
            scenario_builder = NuPlanScenarioBuilder(data_root='foo', map_root='bar', sensor_root='qux', db_files=None, map_version='baz', max_workers=None, verbose=False, scenario_mapping=None, vehicle_parameters=None, include_cameras=False)
            scenario_filter = ScenarioFilter(scenario_types=['type1', 'type2', 'type3'], scenario_tokens=['a', 'b', 'c', 'd', 'e', 'f'], log_names=['filename1', 'filename2'], map_names=['map1', 'map2'], num_scenarios_per_type=2, limit_total_scenarios=None, expand_scenarios=False, remove_invalid_goals=False, shuffle=False, timestamp_threshold_s=None, ego_displacement_minimum_m=None, ego_start_speed_threshold=None, ego_stop_speed_threshold=None, speed_noise_tolerance=None, token_set_path=None, fraction_in_token_set_threshold=None)
            result = scenario_builder.get_scenarios(scenario_filter, Sequential())
            self.assertEqual(4, len(result))
            self.assertEqual(2, sum((1 if s.scenario_type == 'type1' else 0 for s in result)))
            self.assertEqual(2, sum((1 if s.scenario_type == 'type2' else 0 for s in result)))

    def test_get_scenarios_total_num_scenarios_filter(self) -> None:
        """
        Tests that the get_scenarios() method functions properly
        With a total_num_scenarios filter.
        """

        def db_file_patch(params: GetScenariosFromDbFileParams) -> ScenarioDict:
            """
            A patch for the get_scenarios_from_db_file method
            """
            self.assertEqual(params.filter_tokens, ['a', 'b', 'c', 'd', 'e', 'f'])
            self.assertEqual(params.filter_types, ['type1', 'type2', 'type3'])
            self.assertEqual(params.filter_map_names, ['map1', 'map2'])
            self.assertFalse(params.include_cameras)
            self.assertTrue(params.log_file_absolute_path in ['filename1', 'filename2'])
            m1 = MockNuPlanScenario(token='a', scenario_type='type1')
            m2 = MockNuPlanScenario(token='b', scenario_type='type1')
            m3 = MockNuPlanScenario(token='c', scenario_type='type2')
            return {'type1': [m1, m2], 'type2': [m3]}

        def discover_log_dbs_patch(load_path: Union[List[str], str]) -> List[str]:
            """
            A patch for the discover_log_dbs method
            """
            return ['filename1', 'filename2', 'filename3']
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.get_scenarios_from_db_file', db_file_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder.discover_log_dbs', discover_log_dbs_patch):
            scenario_builder = NuPlanScenarioBuilder(data_root='foo', map_root='bar', sensor_root='qux', db_files=None, map_version='baz', max_workers=None, verbose=False, scenario_mapping=None, vehicle_parameters=None, include_cameras=False)
            scenario_filter = ScenarioFilter(scenario_types=['type1', 'type2', 'type3'], scenario_tokens=['a', 'b', 'c', 'd', 'e', 'f'], log_names=['filename1', 'filename2'], map_names=['map1', 'map2'], num_scenarios_per_type=None, limit_total_scenarios=5, expand_scenarios=False, remove_invalid_goals=False, shuffle=False, timestamp_threshold_s=None, ego_displacement_minimum_m=None, ego_start_speed_threshold=None, ego_stop_speed_threshold=None, speed_noise_tolerance=None, token_set_path=None, fraction_in_token_set_threshold=None)
            result = scenario_builder.get_scenarios(scenario_filter, Sequential())
            self.assertEqual(5, len(result))

class DetectionTracksChallengeServicer(chpb_grpc.DetectionTracksChallengeServicer):
    """
    Servicer for exposing initialization and trajectory computation services to the client.
    It keeps a rolling history buffer to avoid unnecessary serialization/deserialization.
    """

    def __init__(self, planner_config: DictConfig, map_manager: MapManager):
        """
        :param planner_config: The planner configuration to instantiate the planner.
        :param map_manager: The map manager.
        """
        self.planner: Optional[AbstractPlanner] = None
        self._planner_config = planner_config
        self.map_manager = map_manager
        self.simulation_history_buffer: Optional[SimulationHistoryBuffer] = None
        self._initialized = False

    @staticmethod
    def _extract_simulation_iteration(planner_input_message: chpb.PlannerInput) -> SimulationIteration:
        return SimulationIteration(TimePoint(planner_input_message.simulation_iteration.time_us), planner_input_message.simulation_iteration.index)

    def _build_planner_input(self, planner_input_message: chpb.PlannerInput, buffer: Optional[SimulationHistoryBuffer]) -> PlannerInput:
        """
        Builds a PlannerInput from a serialized PlannerInput message and an existing data buffer
        :param planner_input_message: the serialized message
        :param buffer: The history buffer
        :return: PlannerInput object
        """
        simulation_iteration = self._extract_simulation_iteration(planner_input_message)
        new_data = planner_input_message.simulation_history_buffer
        states = []
        observations = []
        for serialized_state, serialized_observation in zip(new_data.ego_states, new_data.observations):
            states.append(pickle.loads(serialized_state))
            observations.append(pickle.loads(serialized_observation))
        if buffer is not None:
            buffer.extend(states, observations)
        else:
            buffer = SimulationHistoryBuffer.initialize_from_list(len(states), states, observations, new_data.sample_interval)
            self.simulation_history_buffer = buffer
        tl_data_messages = planner_input_message.traffic_light_data
        tl_data = [tl_status_data_from_proto_tl_status_data(tl_data_message) for tl_data_message in tl_data_messages]
        return PlannerInput(iteration=simulation_iteration, history=buffer, traffic_light_data=tl_data)

    def InitializePlanner(self, planner_initialization_message: chpb.PlannerInitializationLight, context: Any) -> chpb.Empty:
        """
        Service to initialize the planner given the initialization request.
        :param planner_initialization_message: Message containing initialization details
        :param context
        """
        planners = build_planners(self._planner_config, None)
        assert len(planners) == 1, f'Configuration should build exactly 1 planner, got {len(planners)} instead!'
        self.planner = planners[0]
        logger.info('Initialization request received..')
        route_roadblock_ids = planner_initialization_message.route_roadblock_ids
        mission_goal = se2_from_proto_se2(planner_initialization_message.mission_goal)
        map_api = self.map_manager.get_map(planner_initialization_message.map_name)
        map_api.initialize_all_layers()
        planner_initialization = PlannerInitialization(route_roadblock_ids=route_roadblock_ids, mission_goal=mission_goal, map_api=map_api)
        self.simulation_history_buffer = None
        self.planner.initialize(planner_initialization)
        logging.info('Planner initialized!')
        self._initialized = True
        return chpb.Empty()

    def ComputeTrajectory(self, planner_input_message: chpb.PlannerInput, context: Any) -> chpb.Trajectory:
        """
        Service to compute a trajectory given a planner input message
        :param planner_input_message: Message containing the input to the planner
        :param context
        :return Message containing the computed trajectories
        """
        assert self._initialized, 'Planner has not been initialized. Please call InitializePlanner'
        planner_inputs = self._build_planner_input(planner_input_message, self.simulation_history_buffer)
        if isinstance(self.planner, AbstractPlanner):
            trajectory = self.planner.compute_trajectory(planner_inputs)
            return proto_traj_from_inter_traj(trajectory)
        raise RuntimeError('The planner was not initialized correctly!')

def se2_from_proto_se2(se2: chpb.StateSE2) -> StateSE2:
    """
    Deserializes StateSE2 message to a StateSE2 object
    :param se2: The proto StateSE2 message
    :return: The corresponding StateSE2 object
    """
    return StateSE2(x=se2.x, y=se2.y, heading=se2.heading)

