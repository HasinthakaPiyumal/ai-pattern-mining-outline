# Cluster 8

class TestPlannerTutorialHydra(unittest.TestCase):
    """
    Test planner tutorial Jupyter notebook hydra configuration.
    """

    def setUp(self) -> None:
        """Setup."""
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        """Clean up."""
        if Path(self.tmp_dir.name).exists():
            self.tmp_dir.cleanup()
        if ray.is_initialized():
            ray.shutdown()

    def test_hydra_paths_utils(self) -> None:
        """
        Test HydraConfigPaths utility functions for storing config paths for simulation and visualization.
        """
        simulation_hydra_paths = construct_simulation_hydra_paths(BASE_CONFIG_PATH)
        with hydra.initialize_config_dir(config_dir=simulation_hydra_paths.config_path):
            cfg = hydra.compose(config_name=simulation_hydra_paths.config_name, overrides=[f'hydra.searchpath=[{simulation_hydra_paths.common_dir}, {simulation_hydra_paths.experiment_dir}]', '+simulation=open_loop_boxes', 'log_config=false', 'scenario_builder=nuplan_mini', 'planner=simple_planner', 'scenario_filter=one_of_each_scenario_type', 'scenario_filter.limit_total_scenarios=2', 'exit_on_failure=true', "selected_simulation_metrics='[ego_acceleration_statistics, ego_jerk_statistics]'", f'group={self.tmp_dir.name}', 'experiment_name=hydra_paths_utils_test', 'output_dir=${group}/${experiment}'])
            main_simulation(cfg)
        results_dir = Path(cfg.output_dir)
        simulation_file = [str(file) for file in results_dir.iterdir() if file.is_file() and file.suffix == '.nuboard'][0]
        nuboard_hydra_paths = construct_nuboard_hydra_paths(BASE_CONFIG_PATH)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TEST_TIMEOUT)
        try:
            with hydra.initialize_config_dir(config_dir=nuboard_hydra_paths.config_path):
                cfg = hydra.compose(config_name=nuboard_hydra_paths.config_name, overrides=['scenario_builder=nuplan_mini', f'simulation_path={simulation_file}', f'hydra.searchpath=[{nuboard_hydra_paths.common_dir}, {nuboard_hydra_paths.experiment_dir}]', 'port_number=4555'])
                main_nuboard(cfg)
        except Exception as exc:
            signal.alarm(0)
            self.assertTrue(isinstance(exc, TimeoutError))

def construct_simulation_hydra_paths(base_config_path: str) -> HydraConfigPaths:
    """
    Specifies relative paths to simulation configs to pass to hydra to declutter tutorial.
    :param base_config_path: Base config path.
    :return: Hydra config path.
    """
    common_dir = 'file://' + join(base_config_path, 'config', 'common')
    config_name = 'default_simulation'
    config_path = join(base_config_path, 'config', 'simulation')
    experiment_dir = 'file://' + join(base_config_path, 'experiments')
    return HydraConfigPaths(common_dir, config_name, config_path, experiment_dir)

def construct_nuboard_hydra_paths(base_config_path: str) -> HydraConfigPaths:
    """
    Specifies relative paths to nuBoard configs to pass to hydra to declutter tutorial.
    :param base_config_path: Base config path.
    :return: Hydra config path.
    """
    common_dir = 'file://' + join(base_config_path, 'config', 'common')
    config_name = 'default_nuboard'
    config_path = join(base_config_path, 'config/nuboard')
    experiment_dir = 'file://' + join(base_config_path, 'experiments')
    return HydraConfigPaths(common_dir, config_name, config_path, experiment_dir)

class TestNuPlanDBWrapper(unittest.TestCase):
    """Test NuPlanDB wrapper which supports loading/accessing multiple log databases."""

    def setUp(self) -> None:
        """Set up test case."""
        self.db_wrapper = get_test_nuplan_db_wrapper_nocache()

    def test_serialization(self) -> None:
        """Test whether the wrapper object can be serialized/deserialized correctly."""
        serialized_binary = pickle.dumps(self.db_wrapper)
        re_db_wrapper: NuPlanDBWrapper = pickle.loads(serialized_binary)
        self.assertEqual(self.db_wrapper.data_root, re_db_wrapper.data_root)

    def test_maps_db(self) -> None:
        """Test that maps DB has been loaded."""
        self.db_wrapper.maps_db.load_vector_layer('us-nv-las-vegas-strip', 'lane_connectors')

    def test_nuplandb_wrapper_memory_usage(self) -> None:
        """
        Test that repeatedly creating and destroying nuplan DB wrapper objects does not cause memory leaks.
        """

        def spin_up_db_wrapper() -> None:
            db_wrapper = get_test_nuplan_db_wrapper_nocache()
            del db_wrapper
        starting_usage = 0
        ending_usage = 0
        num_iterations = 5
        hpy = guppy.hpy()
        hpy.setrelheap()
        for i in range(0, num_iterations, 1):
            spin_up_db_wrapper()
            gc.collect()
            heap = hpy.heap()
            _ = heap.size
            if i == num_iterations - 2:
                starting_usage = heap.size
            if i == num_iterations - 1:
                ending_usage = heap.size
        memory_difference_in_mb = (ending_usage - starting_usage) / (1024 * 1024)
        max_allowable_growth_mb = max(0.1, 0.1 * starting_usage / (1024 * 1024))
        self.assertGreater(max_allowable_growth_mb, memory_difference_in_mb)

def get_test_nuplan_db_wrapper_nocache() -> NuPlanDBWrapper:
    """
    Gets a nuPlan DB wrapper object with default settings to be used in testing.
    This object will not be cached.
    """
    return NuPlanDBWrapper(data_root=NUPLAN_DATA_ROOT, map_root=NUPLAN_MAPS_ROOT, db_files=NUPLAN_DB_FILES, map_version=NUPLAN_MAP_VERSION)

def initialize_nuboard(cfg: DictConfig) -> NuBoard:
    """
    Sets up dependencies and instantiates a NuBoard object.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: NuBoard object.
    """
    update_config_for_nuboard(cfg=cfg)
    scenario_builder = build_scenario_builder(cfg)
    vehicle_parameters: VehicleParameters = instantiate(cfg.scenario_builder.vehicle_parameters)
    profiler_path = None
    if cfg.profiler_path:
        profiler_path = Path(cfg.profiler_path)
    nuboard = NuBoard(profiler_path=profiler_path, nuboard_paths=cfg.simulation_path, scenario_builder=scenario_builder, port_number=cfg.port_number, resource_prefix=cfg.resource_prefix, vehicle_parameters=vehicle_parameters, async_scenario_rendering=cfg.async_scenario_rendering, scenario_rendering_frame_rate_cap_hz=cfg.scenario_rendering_frame_rate_cap_hz)
    return nuboard

def update_config_for_nuboard(cfg: DictConfig) -> None:
    """
    Updates the config based on some conditions.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    """
    OmegaConf.set_struct(cfg, False)
    if cfg.simulation_path is None:
        cfg.simulation_path = []
    elif not (isinstance(cfg.simulation_path, list) or isinstance(cfg.simulation_path, ListConfig)):
        cfg.simulation_path = [cfg.simulation_path]
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)
    if cfg.log_config:
        logger.info('\n' + OmegaConf.to_yaml(cfg))

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Execute all available challenges simultaneously on the same scenario.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    """
    nuboard = initialize_nuboard(cfg)
    nuboard.run()

class TestUtilsConfig(unittest.TestCase):
    """Tests for the non-distributed training functions in utils_config.py."""
    specific_world_size = 4

    @staticmethod
    def _generate_mock_training_config() -> DictConfig:
        """
        Returns a mock training configuration with sensible default values.
        :return: DictConfig representing the training configuration.
        """
        return DictConfig({'log_config': True, 'experiment': 'mock_experiment_name', 'group': 'mock_group_name', 'cache': {'cleanup_cache': False, 'cache_path': None}, 'data_loader': {'params': {'num_workers': None}}, 'lightning': {'trainer': {'params': {'gpus': None, 'accelerator': None, 'precision': None}, 'overfitting': {'enable': False}}}, 'gpu': False})

    @staticmethod
    def _generate_mock_simulation_config() -> DictConfig:
        """
        Returns a mock simulation configuration with sensible default values.
        :return: DictConfig representing the simulation configuration.
        """
        return DictConfig({'log_config': True, 'experiment': 'mock_experiment_name', 'group': 'mock_group_name', 'callback': {'timing_callback': {'_target_': 'nuplan.planning.simulation.callback.timing_callback.TimingCallback'}, 'simulation_log_callback': {'_target_': 'nuplan.planning.simulation.callback.simulation_log_callback.SimulationLogCallback'}, 'metric_callback': {'_target_': 'nuplan.planning.simulation.callback.metric_callback.MetricCallback'}}})

    @staticmethod
    def _patch_return_false() -> bool:
        """A patch function that will always return False."""
        return False

    @staticmethod
    def _patch_return_true() -> bool:
        """A patch function that will always return True."""
        return True

    @given(cache_path=st.one_of(st.none(), st.just('s3://bucket/key')))
    @settings(deadline=None)
    def test_update_config_for_training_cache_path_none_or_s3(self, cache_path: Optional[str]) -> None:
        """
        Tests the behavior of update_config_for_training when the supplied cfg.cache.cache_path is either
        None or an S3 path.
        """
        mock_config = TestUtilsConfig._generate_mock_training_config()
        mock_config.cache.cache_path = cache_path
        with TemporaryDirectory() as tmp_dir:
            with patch_with_validation('torch.cuda.is_available', TestUtilsConfig._patch_return_false):
                update_config_for_training(mock_config)
            self.assertTrue(Path(tmp_dir).exists())

    def test_update_config_for_training_cache_path_local_non_existing(self) -> None:
        """
        Tests the behavior of update_config_for_training when the supplied cfg.cache.cache_path doesn't exist yet.
        """
        mock_config = TestUtilsConfig._generate_mock_training_config()
        with TemporaryDirectory() as tmp_dir:
            mock_config.cache.cache_path = tmp_dir
            rmtree(tmp_dir)
            with patch_with_validation('torch.cuda.is_available', TestUtilsConfig._patch_return_false):
                update_config_for_training(mock_config)
            self.assertTrue(Path(tmp_dir).exists())

    def test_update_config_for_training_cache_path_local_cleanup(self) -> None:
        """
        Tests the behavior of update_config_for_training when the supplied cfg.cache.cache_path exists and
        cleanup_cache is requested.
        """
        mock_config = TestUtilsConfig._generate_mock_training_config()
        with TemporaryDirectory() as tmp_dir:
            _, tmp_file = mkstemp(dir=tmp_dir)
            mock_config.cache.cache_path = tmp_dir
            mock_config.cache.cleanup_cache = True
            self.assertTrue(Path(tmp_file).exists())
            with patch_with_validation('torch.cuda.is_available', TestUtilsConfig._patch_return_false):
                update_config_for_training(mock_config)
            self.assertFalse(Path(tmp_file).exists())
            self.assertTrue(Path(tmp_dir).exists())
        self.assertFalse(Path(tmp_dir).exists())

    def test_update_config_for_training_overfitting(self) -> None:
        """
        Tests the behavior of update_config_for_training in regard to overfitting configurations.
        """
        mock_config = TestUtilsConfig._generate_mock_training_config()
        num_workers = 32
        mock_config.data_loader.params.num_workers = num_workers
        mock_config.lightning.trainer.overfitting.enable = False
        with patch_with_validation('torch.cuda.is_available', TestUtilsConfig._patch_return_false):
            update_config_for_training(mock_config)
        self.assertEqual(num_workers, mock_config.data_loader.params.num_workers)
        mock_config.lightning.trainer.overfitting.enable = True
        with patch_with_validation('torch.cuda.is_available', TestUtilsConfig._patch_return_false):
            update_config_for_training(mock_config)
        self.assertEqual(0, mock_config.data_loader.params.num_workers)

    @given(is_gpu_enabled=st.booleans(), is_cuda_available=st.booleans())
    @settings(deadline=None)
    def test_update_config_for_training_gpu(self, is_gpu_enabled: bool, is_cuda_available: bool) -> None:
        """
        Tests the behavior of update_config_for_training in regard to gpu configurations.
        """
        invalid_value = -99
        cuda_patch = TestUtilsConfig._patch_return_true if is_cuda_available else TestUtilsConfig._patch_return_false

        def get_expected_gpu_config(gpu_enabled: bool, cuda_available: bool) -> Optional[int]:
            return -1 if gpu_enabled and cuda_available else None

        def get_expected_accelerator_config(gpu_enabled: bool, cuda_available: bool) -> Optional[int]:
            return invalid_value if gpu_enabled and cuda_available else None

        def get_expected_precision_config(gpu_enabled: bool, cuda_available: bool) -> Optional[int]:
            return invalid_value if gpu_enabled and cuda_available else 32
        mock_config = TestUtilsConfig._generate_mock_training_config()
        mock_config.gpu = is_gpu_enabled
        mock_config.lightning.trainer.params.gpus = invalid_value
        mock_config.lightning.trainer.params.accelerator = invalid_value
        mock_config.lightning.trainer.params.precision = invalid_value
        with patch_with_validation('torch.cuda.is_available', cuda_patch):
            update_config_for_training(mock_config)
        self.assertEqual(get_expected_gpu_config(is_gpu_enabled, is_cuda_available), mock_config.lightning.trainer.params.gpus)
        self.assertEqual(get_expected_accelerator_config(is_gpu_enabled, is_cuda_available), mock_config.lightning.trainer.params.accelerator)
        self.assertEqual(get_expected_precision_config(is_gpu_enabled, is_cuda_available), mock_config.lightning.trainer.params.precision)

    @given(max_number_of_workers=st.one_of(st.none(), st.just(0)))
    @settings(deadline=None)
    def test_update_config_for_simulation_falsy_max_number_of_workers(self, max_number_of_workers: int) -> None:
        """
        Tests that update_config_for_simulation works as expected.
        When max number of workers is falsy, timing_callback won't be removed
        """
        mock_config = TestUtilsConfig._generate_mock_simulation_config()
        mock_config.max_number_of_workers = max_number_of_workers
        update_config_for_simulation(mock_config)
        self.assertEqual(3, len(mock_config.callback))

    @given(max_number_of_workers=st.integers(min_value=1))
    @settings(deadline=None)
    def test_update_config_for_simulation_truthy_max_number_of_workers(self, max_number_of_workers: int) -> None:
        """
        Tests that update_config_for_simulation works as expected. When max number of workers is truthy, a new
        `callbacks` entry will be added. The values are taken from `callback` with timing_callback target removed.
        """
        mock_config = TestUtilsConfig._generate_mock_simulation_config()
        mock_config.max_number_of_workers = max_number_of_workers
        update_config_for_simulation(mock_config)
        self.assertEqual(3, len(mock_config.callback))
        self.assertEqual(2, len(mock_config.callbacks))
        callbacks_targets = [callback['_target_'] for callback in mock_config.callbacks]
        self.assertNotIn('nuplan.planning.simulation.callback.timing_callback.TimingCallback', callbacks_targets)

    def test_update_config_for_nuboard(self) -> None:
        """Tests that update_config_for_nuboard works as expected."""
        mock_config = DictConfig({'log_config': True})
        mock_config.simulation_path = None
        update_config_for_nuboard(mock_config)
        self.assertIsNotNone(mock_config.simulation_path)
        self.assertEqual(0, len(mock_config.simulation_path))
        simulation_path_list = ['/mock/path', '/to/somewhere']
        mock_config.simulation_path = simulation_path_list
        update_config_for_nuboard(mock_config)
        self.assertEqual(simulation_path_list, mock_config.simulation_path)
        simulation_path_list_config = ListConfig(element_type=str, content=['/mock/path', '/to/somewhere'])
        mock_config.simulation_path = simulation_path_list_config
        update_config_for_nuboard(mock_config)
        self.assertEqual(simulation_path_list_config, mock_config.simulation_path)
        simulation_path = '/mock/path'
        mock_config.simulation_path = simulation_path
        update_config_for_nuboard(mock_config)
        expected_simulation_path_list = [simulation_path]
        self.assertEqual(expected_simulation_path_list, mock_config.simulation_path)

    @patch.dict(os.environ, {'WORLD_SIZE': str(specific_world_size)}, clear=True)
    def test_get_num_gpus_used_from_world_size(self) -> None:
        """
        Tests that that get_num_gpus_used works as expected. When WORLD_SIZE is set to a specific value, the function
        will simply return that value.
        """
        mock_config = DictConfig({})
        num_gpus = get_num_gpus_used(mock_config)
        self.assertEqual(self.specific_world_size, num_gpus)

    @given(num_gpus_config=st.integers(min_value=-1), cuda_device_count=st.integers(min_value=0), num_nodes=st.integers(min_value=1))
    @example(num_gpus_config=-1, cuda_device_count=2, num_nodes=2)
    @settings(deadline=None)
    def test_get_num_gpus_used_from_config(self, num_gpus_config: int, cuda_device_count: int, num_nodes: int) -> None:
        """
        Tests that that get_num_gpus_used works as expected when WORLD_SIZE environment variable is not set.
        """

        def patch_get_cuda_device_count() -> int:
            return cuda_device_count
        with patch.dict(os.environ, {'NUM_NODES': str(num_nodes)}, clear=True), patch_with_validation('torch.cuda.device_count', patch_get_cuda_device_count):
            mock_config = TestUtilsConfig._generate_mock_training_config()
            mock_config.lightning.trainer.params.gpus = num_gpus_config
            num_gpus = get_num_gpus_used(mock_config)
            expected_num_gpus = num_gpus_config if num_gpus_config != -1 else cuda_device_count * num_nodes
            self.assertEqual(expected_num_gpus, num_gpus)

    def test_get_num_gpus_used_invalid_config(self) -> None:
        """
        Tests that that get_num_gpus_used raises a RuntimeError when WORLD_SIZE environment variable is not set and
        a string is passed as the value of mock_config.lightning.trainer.params.gpus.
        """
        mock_config = TestUtilsConfig._generate_mock_training_config()
        mock_config.lightning.trainer.params.gpus = '1'
        with self.assertRaises(RuntimeError):
            get_num_gpus_used(mock_config)

def setup_notebook() -> None:
    """
    Code that must be run at the start of every tutorial notebook to:
        - patch the event loop to allow nesting, eg. so we can run asyncio.run from
          within a notebook.
    """
    nest_asyncio.apply()

