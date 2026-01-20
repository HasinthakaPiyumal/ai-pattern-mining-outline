# Cluster 140

class FeaturePreprocessor:
    """
    Compute features and targets for a scenario. This class also manages cache. If a feature/target
    is not present in a cache, it is computed, otherwise it is loaded
    """

    def __init__(self, cache_path: Optional[str], force_feature_computation: bool, feature_builders: List[AbstractFeatureBuilder], target_builders: List[AbstractTargetBuilder]):
        """
        Initialize class.
        :param cache_path: Whether to cache features.
        :param force_feature_computation: If true, even if cache exists, it will be overwritten.
        :param feature_builders: List of feature builders.
        :param target_builders: List of target builders.
        """
        self._cache_path = pathlib.Path(cache_path) if cache_path else None
        self._force_feature_computation = force_feature_computation
        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._storing_mechanism = FeatureCacheS3(cache_path) if str(cache_path).startswith('s3://') else FeatureCachePickle()
        assert len(feature_builders) != 0, 'Number of feature builders has to be grater than 0!'

    @property
    def feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        :return: all feature builders
        """
        return self._feature_builders

    @property
    def target_builders(self) -> List[AbstractTargetBuilder]:
        """
        :return: all target builders
        """
        return self._target_builders

    def get_list_of_feature_types(self) -> List[Type[AbstractModelFeature]]:
        """
        :return all features that are computed by the builders
        """
        return [builder.get_feature_type() for builder in self._feature_builders]

    def get_list_of_target_types(self) -> List[Type[AbstractModelFeature]]:
        """
        :return all targets that are computed by the builders
        """
        return [builder.get_feature_type() for builder in self._target_builders]

    def compute_features(self, scenario: AbstractScenario) -> Tuple[FeaturesType, TargetsType, List[CacheMetadataEntry]]:
        """
        Compute features for a scenario, in case cache_path is set, features will be stored in cache,
        otherwise just recomputed
        :param scenario for which features and targets should be computed
        :return: model features and targets and cache metadata
        """
        try:
            all_features: FeaturesType
            all_feature_cache_metadata: List[CacheMetadataEntry]
            all_targets: TargetsType
            all_targets_cache_metadata: List[CacheMetadataEntry]
            all_features, all_feature_cache_metadata = self._compute_all_features(scenario, self._feature_builders)
            all_targets, all_targets_cache_metadata = self._compute_all_features(scenario, self._target_builders)
            all_cache_metadata = all_feature_cache_metadata + all_targets_cache_metadata
            return (all_features, all_targets, all_cache_metadata)
        except Exception as error:
            msg = f'Failed to compute features for scenario token {scenario.token} in log {scenario.log_name}\nError: {error}'
            logger.error(msg)
            traceback.print_exc()
            raise RuntimeError(msg)

    def _compute_all_features(self, scenario: AbstractScenario, builders: List[Union[AbstractFeatureBuilder, AbstractTargetBuilder]]) -> Tuple[Union[FeaturesType, TargetsType], List[Optional[CacheMetadataEntry]]]:
        """
        Compute all features/targets from builders for scenario
        :param scenario: for which features should be computed
        :param builders: to use for feature computation
        :return: computed features/targets and the metadata entries for the computed features/targets
        """
        all_features: FeaturesType = {}
        all_features_metadata_entries: List[CacheMetadataEntry] = []
        for builder in builders:
            feature, feature_metadata_entry = compute_or_load_feature(scenario, self._cache_path, builder, self._storing_mechanism, self._force_feature_computation)
            all_features[builder.get_feature_unique_name()] = feature
            all_features_metadata_entries.append(feature_metadata_entry)
        return (all_features, all_features_metadata_entries)

def compute_or_load_feature(scenario: AbstractScenario, cache_path: Optional[pathlib.Path], builder: Union[AbstractFeatureBuilder, AbstractTargetBuilder], storing_mechanism: FeatureCache, force_feature_computation: bool) -> Tuple[AbstractModelFeature, Optional[CacheMetadataEntry]]:
    """
    Compute features if non existent in cache, otherwise load them from cache
    :param scenario: for which features should be computed
    :param cache_path: location of cached features
    :param builder: which builder should compute the features
    :param storing_mechanism: a way to store features
    :param force_feature_computation: if true, even if cache exists, it will be overwritten
    :return features computed with builder and the metadata entry for the computed feature if feature is valid.
    """
    cache_path_available = cache_path is not None
    file_name = cache_path / scenario.log_name / scenario.scenario_type / scenario.token / builder.get_feature_unique_name() if cache_path_available else None
    need_to_compute_feature = force_feature_computation or not cache_path_available or (not storing_mechanism.exists_feature_cache(file_name))
    feature_stored_sucessfully = False
    if need_to_compute_feature:
        logger.debug('Computing feature...')
        if isinstance(scenario, CachedScenario):
            raise ValueError(textwrap.dedent(f'\n                Attempting to recompute scenario with CachedScenario.\n                This should typically never happen, and usually means that the scenario is missing from the cache.\n                Check the cache to ensure that the scenario is present.\n\n                If it was intended to re-compute the feature on the fly, re-run with `cache.use_cache_without_dataset=False`.\n\n                Debug information:\n                Scenario type: {scenario.scenario_type}. Scenario log name: {scenario.log_name}. Scenario token: {scenario.token}.\n                '))
        if isinstance(builder, AbstractFeatureBuilder):
            feature = builder.get_features_from_scenario(scenario)
        elif isinstance(builder, AbstractTargetBuilder):
            feature = builder.get_targets(scenario)
        else:
            raise ValueError(f'Unknown builder type: {type(builder)}')
        if feature.is_valid and cache_path_available:
            logger.debug(f'Saving feature: {file_name} to a file...')
            file_name.parent.mkdir(parents=True, exist_ok=True)
            feature_stored_sucessfully = storing_mechanism.store_computed_feature_to_folder(file_name, feature)
    else:
        logger.debug(f'Loading feature: {file_name} from a file...')
        feature = storing_mechanism.load_computed_feature_from_folder(file_name, builder.get_feature_type())
        assert feature.is_valid, 'Invalid feature loaded from cache!'
    return (feature, CacheMetadataEntry(file_name=file_name) if need_to_compute_feature and feature_stored_sucessfully else None)

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

