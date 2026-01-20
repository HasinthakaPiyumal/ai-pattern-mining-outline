# Cluster 171

class TestScenarioBuilder(unittest.TestCase):
    """Test update_distributed_optimizer_config function."""

    def setUp(self) -> None:
        """Setup test attributes."""
        self.num_scenarios = 5
        self.specified_feature_names = ['agents', 'trajectory', 'vector_map']
        self.mock_cache_path = 's3://mock_path'
        self.expected_s3_paths = sorted((Path(f'mock_vehicle_log_123/mock_scenario_type_A/mock_token_{i}') for i in range(5)))

    def _get_mock_get_s3_scenario_cache_with_scenario_type_patch(self) -> Callable[..., List[Any]]:
        """
        Gets mock get_s3_scenario_cache_patch function with scenario types.
        """

        def mock_get_s3_scenario_cache_with_scenario_type(cache_path: str, feature_names: List[Any], worker: WorkerPool, load_from_metadata: bool=True) -> List[Path]:
            """
            Mock function for get_s3_scenario_cache
            :param cache_path: Parent of cache path
            :param feature_names: List of feature names
            :return: Mock cache paths
            """
            return [Path('s3://mock_vehicle_log_123/mock_scenario_type_A/mock_token') for _ in range(5)] + [Path('s3://mock_vehicle_log_123/mock_scenario_type_B/mock_token') for _ in range(5)]
        return mock_get_s3_scenario_cache_with_scenario_type

    def _get_mock_get_s3_scenario_cache_without_scenario_type_patch(self) -> Callable[..., List[Any]]:
        """
        Gets mock get_s3_scenario_cache_patch function without scenario types.
        """

        def mock_get_s3_scenario_cache_without_scenario_type(cache_path: str, feature_names: List[Any], worker: WorkerPool) -> List[Path]:
            """
            Mock function for get_s3_scenario_cache
            :param cache_path: Parent of cache path
            :param feature_names: List of feature names
            :return: Mock cache paths
            """
            return [Path('s3://mock_vehicle_log_123/mock_token') for _ in range(5)] + [Path('s3://mock_vehicle_log_123/mock_token') for _ in range(5)]
        return mock_get_s3_scenario_cache_without_scenario_type

    def _get_mock_check_s3_path_exists_patch(self) -> Callable[[str], bool]:
        """
        Gets mock get_s3_scenario_cache_patch function without scenario types.
        """

        def mock_check_s3_path_exists(cache_path: str) -> bool:
            """
            Mock function for check_s3_path_exists
            :param cache_path: Parent of cache path
            :return: True
            """
            return True
        return mock_check_s3_path_exists

    def _get_mock_expand_s3_dir(self) -> Callable[[str], List[str]]:
        """
        Gets mock expand_s3_dir function.
        """

        def mock_expand_s3_dir(cache_path: str) -> List[str]:
            """
            Mock function for expand_s3_dir.
            :param cache_path: S3 cache path.
            :return: List of mock s3 file paths fetched directly from s3 cache path provided.
            """
            return [f'{cache_path}/mock_vehicle_log_123/mock_scenario_type_A/mock_token_{i}/{feature_name}.bin' for i in range(5) for feature_name in ['agents', 'trajectory', 'vector_map']]
        return mock_expand_s3_dir

    def _get_mock_fail_to_get_cache_metadata_paths(self) -> Callable[[Path, str], List[str]]:
        """
        Gets mock get_cache_metadata_paths function.
        """

        def mock_fail_to_get_cache_metadata_paths(s3_key: Path, s3_bucket: str) -> List[str]:
            """
            Mock function for get_cache_metadata_paths.
            :param s3_key: S3 cache key.
            :param s3_bucket: S3 cache bucket.
            :return: List of mock s3 metadata file paths fetched from s3 cache path provided.
            """
            return []
        return mock_fail_to_get_cache_metadata_paths

    def _get_mock_worker_map(self) -> Callable[..., List[Any]]:
        """
        Gets mock worker_map function.
        """

        def mock_worker_map(worker: WorkerPool, fn: Callable[..., List[Any]], input_objects: List[Any]) -> List[Any]:
            """
            Mock function for worker_map
            :param worker: Worker pool
            :param fn: Callable function
            :param input_objects: List of objects to be used as input
            :return: List of output objects
            """
            return fn(input_objects)
        return mock_worker_map

    def _get_mock_read_cache_metadata(self) -> Callable[..., List[CacheMetadataEntry]]:
        """
        Gets mock read_cache_metadata function.
        """

        def mock_read_cache_metadata(cache_path: Path, metadata_filenames: List[str], worker: WorkerPool) -> List[CacheMetadataEntry]:
            """
            Mock function for read_cache_metadata
            :param cache_path: Path to s3 cache.
            :param metadata_filenames: Filenames of the metadata csv files.
            :return: List of CacheMetadataEntry
            """
            return [CacheMetadataEntry(f'{cache_path}/mock_vehicle_log_123/mock_scenario_type_A/mock_token_{i}/{feature_name}.bin') for i in range(5) for feature_name in ['agents', 'trajectory', 'vector_map']]
        return mock_read_cache_metadata

    def test_is_valid_token(self) -> None:
        """
        Test that scenario token validation works.
        """
        self.assertFalse(is_valid_token('a'))
        self.assertFalse(is_valid_token(3))
        self.assertTrue(is_valid_token('48681125850853e4'))

    def test_extract_and_filter_scenarios_from_cache(self) -> None:
        """
        Test extracting the scenarios from cache and filtering by scenario type
        """
        mock_cfg = Mock(DictConfig)
        cache = Mock()
        cache.cache_path = 's3://mock_path'
        scenario_filter = Mock()
        scenario_filter.scenario_types = ['mock_scenario_type_A']
        mock_cfg.cache = cache
        mock_cfg.scenario_filter = scenario_filter
        mock_worker = Mock(WorkerPool)
        mock_model = MockModel()
        mock_model = cast(TorchModuleWrapper, mock_model)
        mock_worker_map = self._get_mock_worker_map()
        mock_get_s3_scenario_cache = self._get_mock_get_s3_scenario_cache_with_scenario_type_patch()
        with mock.patch('nuplan.planning.script.builders.scenario_builder.worker_map', mock_worker_map), mock.patch('nuplan.planning.script.builders.scenario_builder.get_s3_scenario_cache', mock_get_s3_scenario_cache):
            scenarios = extract_scenarios_from_cache(mock_cfg, mock_worker, mock_model)
            msg = f'Expected number of scenarios to be {self.num_scenarios} but got {len(scenarios)}'
            self.assertEqual(len(scenarios), self.num_scenarios, msg=msg)

    def test_extract_and_filter_scenarios_from_cache_when_cache_path_has_no_scenario_type(self) -> None:
        """
        Test extracting the scenarios from cache and filtering by scenario type when it doesn't exist in the cache path.
        """
        mock_cfg = Mock(DictConfig)
        cache = Mock()
        cache.cache_path = 's3://mock_path'
        scenario_filter = Mock()
        scenario_filter.scenario_types = ['mock_scenario_type_A']
        mock_cfg.cache = cache
        mock_cfg.scenario_filter = scenario_filter
        mock_worker = Mock(WorkerPool)
        mock_model = MockModel()
        mock_model = cast(TorchModuleWrapper, mock_model)
        mock_worker_map = self._get_mock_worker_map()
        mock_get_s3_scenario_cache = self._get_mock_get_s3_scenario_cache_without_scenario_type_patch()
        with mock.patch('nuplan.planning.script.builders.scenario_builder.worker_map', mock_worker_map), mock.patch('nuplan.planning.script.builders.scenario_builder.get_s3_scenario_cache', mock_get_s3_scenario_cache):
            with self.assertRaises(AssertionError):
                extract_scenarios_from_cache(mock_cfg, mock_worker, mock_model)

    def test_extract_and_filter_scenarios_from_cache_when_specified_scenario_type_does_not_exist(self) -> None:
        """
        Test extracting the scenarios from cache and filtering by scenario type when specified scenario type does not exist.
        """
        mock_cfg = Mock(DictConfig)
        cache = Mock()
        cache.cache_path = 's3://mock_path'
        scenario_filter = Mock()
        scenario_filter.scenario_types = ['nonexistent_scenario_type']
        mock_cfg.cache = cache
        mock_cfg.scenario_filter = scenario_filter
        mock_worker = Mock(WorkerPool)
        mock_model = MockModel()
        mock_model = cast(TorchModuleWrapper, mock_model)
        mock_worker_map = self._get_mock_worker_map()
        mock_get_s3_scenario_cache = self._get_mock_get_s3_scenario_cache_with_scenario_type_patch()
        with mock.patch('nuplan.planning.script.builders.scenario_builder.worker_map', mock_worker_map), mock.patch('nuplan.planning.script.builders.scenario_builder.get_s3_scenario_cache', mock_get_s3_scenario_cache):
            with self.assertRaises(AssertionError):
                extract_scenarios_from_cache(mock_cfg, mock_worker, mock_model)

    def test_get_s3_scenario_cache(self) -> None:
        """
        Test get_s3_scenario_cache and ensure that it returns the correct format of cache paths.
        """
        mock_cache_path = self.mock_cache_path
        mock_feature_names = set(self.specified_feature_names)
        mock_worker = Mock(WorkerPool)
        mock_expand_s3_dir = self._get_mock_expand_s3_dir()
        mock_check_s3_path_exists = self._get_mock_check_s3_path_exists_patch()
        mock_read_cache_metadata = self._get_mock_read_cache_metadata()
        mock_fail_to_get_cache_metadata_paths = self._get_mock_fail_to_get_cache_metadata_paths()
        with mock.patch('nuplan.planning.script.builders.scenario_builder.expand_s3_dir', mock_expand_s3_dir), mock.patch('nuplan.planning.script.builders.scenario_builder.check_s3_path_exists', mock_check_s3_path_exists), mock.patch('nuplan.planning.script.builders.scenario_builder.read_cache_metadata', mock_read_cache_metadata), mock.patch('nuplan.planning.script.builders.scenario_builder.get_cache_metadata_paths', mock_fail_to_get_cache_metadata_paths):
            scenario_cache_paths = get_s3_scenario_cache(mock_cache_path, mock_feature_names, mock_worker)
            msg = f'Expected S3 cache paths to be {self.expected_s3_paths} but got {scenario_cache_paths}'
            self.assertEqual(scenario_cache_paths, self.expected_s3_paths, msg=msg)

def is_valid_token(token: Any) -> bool:
    """
    Basic check that a scenario token is the right type/length.
    :token: parsed by hydra.
    :return: true if it looks valid, otherwise false.
    """
    if not isinstance(token, str) or len(token) != 16:
        return False
    try:
        return bytearray.fromhex(token).hex() == token
    except (TypeError, ValueError):
        return False

