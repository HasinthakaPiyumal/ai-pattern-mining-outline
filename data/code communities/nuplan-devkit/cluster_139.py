# Cluster 139

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

class TestUtilsCache(unittest.TestCase):
    """Test caching utilities."""

    def setUp(self) -> None:
        """Set up test case."""
        local_cache_path = '/tmp/cache'
        s3_cache_path = 's3://tmp/cache'
        self.cache_paths = [local_cache_path, s3_cache_path]
        local_store = FeatureCachePickle()
        s3_store = FeatureCacheS3(s3_cache_path)
        s3_store._store = MockS3Store()
        self.cache_engines = [local_store, s3_store]

    def test_storing_to_cache_vector_map(self) -> None:
        """
        Test storing feature to cache
        """
        dim = 50
        feature = VectorMap(coords=[np.zeros((dim, 2, 2)).astype(np.float32)], lane_groupings=[[np.zeros(dim).astype(np.float32)]], multi_scale_connections=[{1: np.zeros((dim, 2)).astype(np.float32)}], on_route_status=[np.zeros((dim, 2)).astype(np.float32)], traffic_light_data=[np.zeros((dim, 4)).astype(np.float32)])
        for cache_path, cache in zip(self.cache_paths, self.cache_engines):
            folder = pathlib.Path(cache_path) / 'tmp_log_name' / 'tmp_scenario_token' / 'vector_map'
            if not str(folder).startswith('s3:/'):
                folder.parent.mkdir(parents=True, exist_ok=True)
            time_now = time.time()
            loaded_feature: VectorMap = self.store_and_load(cache, folder, feature)
            time_later = time.time()
            logger.debug(f'Cache: {type(cache)} = {time_later - time_now}')
            self.assertEqual(feature.num_of_batches, loaded_feature.num_of_batches)
            self.assertEqual(1, loaded_feature.num_of_batches)
            self.assertEqual(feature.coords[0].shape, loaded_feature.coords[0].shape)
            self.assertEqual(feature.lane_groupings[0][0].shape, loaded_feature.lane_groupings[0][0].shape)
            self.assertEqual(feature.multi_scale_connections[0][1].shape, loaded_feature.multi_scale_connections[0][1].shape)

    def test_storing_to_cache_raster(self) -> None:
        """
        Test storing feature to cache
        """
        feature = Raster(data=np.zeros((244, 244, 3)))
        for cache_path, cache in zip(self.cache_paths, self.cache_engines):
            folder = pathlib.Path(cache_path) / 'tmp_log_name' / 'tmp_scenario_token' / 'raster'
            if not str(folder).startswith('s3:/'):
                folder.parent.mkdir(parents=True, exist_ok=True)
            loaded_feature = self.store_and_load(cache, folder, feature)
            self.assertEqual(feature.data.shape, loaded_feature.data.shape)

    def store_and_load(self, cache: FeatureCache, folder: pathlib.Path, feature: AbstractModelFeature) -> AbstractModelFeature:
        """
        Store feature and load it back.
        :param cache: Caching mechanism to use.
        :param folder: Folder to store feature.
        :param feature: Feature to store.
        :return: Loaded feature.
        """
        time_now = time.time()
        cache.store_computed_feature_to_folder(folder, feature)
        logger.debug(f'store_computed_feature_to_folder: {type(cache)} = {time.time() - time_now}')
        time_now = time.time()
        out = cache.load_computed_feature_from_folder(folder, feature)
        logger.debug(f'load_computed_feature_from_folder: {type(cache)} = {time.time() - time_now}')
        self.assertIsInstance(out, type(feature))
        return out

