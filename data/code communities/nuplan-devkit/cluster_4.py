# Cluster 4

@lru_cache(maxsize=2)
def get_maps_db(map_root: str, map_version: str) -> GPKGMapsDB:
    """
    Get a maps_db from disk.
    :param map_root: The root folder for the map data.
    :param map_version: The version of the map to load.
    :return; The loaded MapsDB object.
    """
    return GPKGMapsDB(map_root=map_root, map_version=map_version)

def load_log_db_mapping(data_root: str, db_files: Optional[Union[List[str], str]], maps_db: GPKGMapsDB, max_workers: Optional[int]=None, verbose: bool=True) -> Dict[str, NuPlanDB]:
    """
    Load all log database objects and hash them based on their log name.
    Log databases will be discovered based on the input path and downloaded if needed.
    All discovered databases will be loaded/downloaded concurrently in multiple threads.
    :param data_root: Local data root for loading/storing the log databases.
                      If `db_files` is not None, all downloaded databases will be stored to this data root.
    :param db_files: Local/remote filename or list of filenames to be loaded.
                     If None, all database filenames found under `data_root` will be used.
    :param maps_db: Instantiated map database object to be passed to each log database.
    :param max_workers: Maximum number of workers to use when loading the databases concurrently.
                        Only used when the number of databases to load is larger than this parameter.
    :param verbose: Whether to print progress and details during the database loading process.
    :return: Mapping from log name to loaded log database object.
    """
    load_path = data_root if db_files is None else db_files
    db_filenames = discover_log_dbs(load_path)
    num_workers = min(len(db_filenames), MAX_DB_LOADING_THREADS)
    num_workers = num_workers if max_workers is None else min(num_workers, max_workers)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(NuPlanDB, data_root, path, maps_db, verbose) for path in db_filenames]
        futures_iterable = as_completed(futures)
        wrapped_iterable = tqdm(futures_iterable, total=len(futures), leave=False) if verbose else futures_iterable
        log_dbs = [future.result() for future in wrapped_iterable]
    log_dbs = sorted(log_dbs, key=lambda log_db: str(log_db.log_name))
    log_db_mapping = {log_db.log_name: log_db for log_db in log_dbs}
    return log_db_mapping

def discover_log_dbs(load_path: Union[List[str], str]) -> List[str]:
    """
    Discover all log dbs from the input load path.
    If the path is a filename, expand the path and return the list of filenames in that path.
    Else, if the path is already a list, expand each path in the list and return the flattened list.
    :param load_path: Load path, it can be a filename or list of filenames of a database and/or dirs of databases.
    :return: A list with all discovered log database filenames.
    """
    if isinstance(load_path, list):
        nested_db_filenames = [get_db_filenames_from_load_path(path) for path in sorted(load_path)]
        db_filenames = [filename for filenames in nested_db_filenames for filename in filenames]
    else:
        db_filenames = get_db_filenames_from_load_path(load_path)
    return db_filenames

class NuPlanDBWrapper:
    """Wrapper for NuPlanDB that allows loading and accessing mutliple log database."""

    def __init__(self, data_root: str, map_root: str, db_files: Optional[Union[List[str], str]], map_version: str, max_workers: Optional[int]=None, verbose: bool=True):
        """
        Initialize the database wrapper.
        :param data_root: Local data root for loading (or storing downloaded) the log databases.
                        If `db_files` is not None, all downloaded databases will be stored to this data root.
        :param map_root: Local map root for loading (or storing downloaded) the map database.
        :param db_files: Path to load the log databases from, which can be:
                         * filename path to single database:
                            - locally - e.g. /data/sets/nuplan/v1.0/log_1.db
                            - remotely (S3) - e.g. s3://bucket/nuplan/v1.0/log_1.db
                         * directory path of databases to load:
                            - locally - e.g. /data/sets/nuplan/v1.0/
                            - remotely (S3) - e.g. s3://bucket/nuplan/v1.0/
                         * list of database filenames:
                            - locally - e.g. [/data/sets/nuplan/v1.0/log_1.db, /data/sets/nuplan/v1.0/log_2.db]
                            - remotely (S3) - e.g. [s3://bucket/nuplan/v1.0/log_1.db, s3://bucket/nuplan/v1.0/log_2.db]
                         * list of database directories:
                            - locally - e.g. [/data/sets/nuplan/v1.0_split_1/, /data/sets/nuplan/v1.0_split_2/]
                            - remotely (S3) - e.g. [s3://bucket/nuplan/v1.0_split_1/, s3://bucket/nuplan/v1.0_split_2/]
                         Note: Regex expansion is not yet supported.
                         Note: If None, all database filenames found under `data_root` will be used.
        :param map_version: Version of map database to load. The map database is passed to each loaded log database.
        :param max_workers: Maximum number of workers to use when loading the databases concurrently.
                            Only used when the number of databases to load is larger than this parameter.
        :param verbose: Whether to print progress and details during the database loading process.
        """
        self._data_root = data_root
        self._map_root = map_root
        self._db_files = db_files
        self._map_version = map_version
        self._max_workers = max_workers
        self._verbose = verbose
        assert not self._data_root.startswith('s3://'), f'Data root cannot be an S3 path, got {self._data_root}'
        assert not self._map_root.startswith('s3://'), f'Map root cannot be an S3 path, got {self._map_root}'
        self._data_root = self._data_root.replace('//', '/')
        self._map_root = self._map_root.replace('//', '/')
        self._maps_db = GPKGMapsDB(map_root=self._map_root, map_version=self._map_version)
        logger.info('Loaded maps DB')
        self._log_db_mapping = load_log_db_mapping(data_root=self._data_root, db_files=self._db_files, maps_db=self._maps_db, max_workers=self._max_workers, verbose=self._verbose)
        logger.info(f'Loaded {len(self.log_dbs)} log DBs')

    def __reduce__(self) -> Tuple[Type[NuPlanDBWrapper], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return (self.__class__, (self._data_root, self._map_root, self._db_files, self._map_version, self._max_workers, self._verbose))

    def __del__(self) -> None:
        """
        Called when the object is being garbage collected.
        """
        for log_name in self.log_names:
            self._log_db_mapping[log_name].remove_ref()

    @property
    def data_root(self) -> str:
        """Get the data root."""
        return self._data_root

    @property
    def map_root(self) -> str:
        """Get the map root."""
        return self._map_root

    @property
    def map_version(self) -> str:
        """Get the map version."""
        return self._map_version

    @property
    def maps_db(self) -> GPKGMapsDB:
        """Get the map database object."""
        return self._maps_db

    @property
    def log_db_mapping(self) -> Dict[str, NuPlanDB]:
        """Get the dictionary that maps log names to log database objects."""
        return self._log_db_mapping

    @property
    def log_names(self) -> List[str]:
        """Get the list of log names of all loaded log databases."""
        return list(self._log_db_mapping.keys())

    @property
    def log_dbs(self) -> List[NuPlanDB]:
        """Get the list of all loaded log databases."""
        return list(self._log_db_mapping.values())

    def get_log_db(self, log_name: str) -> NuPlanDB:
        """
        Retrieve a log database by log name.
        :param log_name: Log name to access the database hash table.
        :return: Retrieve database object.
        """
        return self._log_db_mapping[log_name]

    def get_all_scenes(self) -> Iterable[Scene]:
        """
        Retrieve and yield all scenes across all loaded log databases.
        :yield: Next scene from all scenes in the loaded databases.
        """
        for db in self._log_db_mapping.values():
            for scene in db.scene:
                yield scene

    def get_all_scenario_types(self) -> List[str]:
        """Retrieve all unique scenario tags in the collection of databases."""
        return sorted({tag for log_db in self.log_dbs for tag in log_db.get_unique_scenario_tags()})

class TestNuPlanDB(unittest.TestCase):
    """Test main nuPlan database class."""

    def setUp(self) -> None:
        """Set up test case."""
        self.db = get_test_nuplan_db()
        self.db.add_ref()

    def test_pickle(self) -> None:
        """Test dumping and loading the object through pickle."""
        db_binary = pickle.dumps(self.db)
        re_db: NuPlanDB = pickle.loads(db_binary)
        self.assertEqual(self.db.data_root, re_db.data_root)
        self.assertEqual(self.db.name, re_db.name)
        self.assertEqual(self.db._verbose, re_db._verbose)

    def test_table_getters(self) -> None:
        """Test the table getters."""
        self.assertTrue(isinstance(self.db.category, Table))
        self.assertTrue(isinstance(self.db.camera, Table))
        self.assertTrue(isinstance(self.db.lidar, Table))
        self.assertTrue(isinstance(self.db.image, Table))
        self.assertTrue(isinstance(self.db.lidar_pc, Table))
        self.assertTrue(isinstance(self.db.lidar_box, Table))
        self.assertTrue(isinstance(self.db.track, Table))
        self.assertTrue(isinstance(self.db.scene, Table))
        self.assertTrue(isinstance(self.db.scenario_tag, Table))
        self.assertTrue(isinstance(self.db.traffic_light_status, Table))
        self.assertSetEqual(self.db.cam_channels, {'CAM_R2', 'CAM_R1', 'CAM_R0', 'CAM_F0', 'CAM_L2', 'CAM_L1', 'CAM_B0', 'CAM_L0'})
        self.assertSetEqual(self.db.lidar_channels, {'MergedPointCloud'})

    def test_nuplan_memory_usage(self) -> None:
        """
        Test that repeatedly creating and destroying nuplan DB objects does not cause memory leaks.
        """

        def spin_up_db() -> None:
            db = get_test_nuplan_db_nocache()
            db.remove_ref()
        starting_usage = 0
        ending_usage = 0
        num_iterations = 5
        hpy = guppy.hpy()
        hpy.setrelheap()
        for i in range(0, num_iterations, 1):
            spin_up_db()
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

def get_test_nuplan_db_nocache() -> NuPlanDB:
    """
    Get a nuPlan DB object with default settings to be used in testing.
    Forces the data to be read from disk.
    """
    load_path = get_test_nuplan_db_path()
    maps_db = get_test_maps_db()
    return NuPlanDB(data_root=NUPLAN_DATA_ROOT, load_path=load_path, maps_db=maps_db)

class TestMapApi(unittest.TestCase):
    """Test NuPlanMapWrapper class."""

    def setUp(self) -> None:
        """
        Initialize the map for each location.
        """
        self.maps_db = get_test_maps_db()
        self.locations = ['sg-one-north', 'us-ma-boston', 'us-nv-las-vegas-strip', 'us-pa-pittsburgh-hazelwood']
        self.available_locations = self.maps_db.get_locations()
        self.nuplan_maps = dict()
        for location in self.available_locations:
            self.nuplan_maps[location] = NuPlanMapWrapper(maps_db=self.maps_db, map_name=location)

    def test_version_names(self) -> None:
        """Tests the locations map version are correct."""
        assert len(self.maps_db.version_names) == len(self.available_locations), 'Incorrect number of version names'

    def test_locations(self) -> None:
        """
        Checks if maps for all locations are available.
        """
        assert len(self.locations) == len(self.available_locations), 'Incorrect number of locations'
        assert sorted(self.locations) == sorted(self.available_locations), 'Missing Locations'

    def test_patch_coord(self) -> None:
        """
        Checks the function to get patch coordinates without rotation.
        """
        path_center = [0, 0]
        path_dimension = [10, 10]
        polygon_coords = self.nuplan_maps[self.locations[0]].get_patch_coord((path_center[0], path_center[1], path_dimension[0], path_dimension[1]), 0.0)
        expected_polygon_coords = Polygon([[5, -5], [5, 5], [-5, 5], [-5, -5], [5, -5]])
        self.assertEqual(polygon_coords, expected_polygon_coords)

    def test_patch_coord_rotated(self) -> None:
        """
        Checks the function to get patch coordinates with rotation.
        """
        path_center = [0, 0]
        path_dimension = [10, 20]
        polygon_coords = self.nuplan_maps[self.locations[0]].get_patch_coord((path_center[0], path_center[1], path_dimension[0], path_dimension[1]), 90.0)
        expected_polygon_coords = Polygon([[5, 10], [-5, 10], [-5, -10], [5, -10], [5, 10]])
        self.assertEqual(polygon_coords, expected_polygon_coords)

    def test_vector_dimensions(self) -> None:
        """
        Checks dimensions of vector layer. It must be less than or equal to size of map.
        """
        for location in self.locations:
            vector_layer_bounds = self.nuplan_maps[location].get_bounds('lanes_polygons')
            map_shape = self.nuplan_maps[location].get_map_dimension()
            self.assertLess(vector_layer_bounds[0], vector_layer_bounds[2])
            self.assertLess(vector_layer_bounds[1], vector_layer_bounds[3])
            self.assertLess(vector_layer_bounds[2] - vector_layer_bounds[0], map_shape[0])
            self.assertLess(vector_layer_bounds[3] - vector_layer_bounds[1], map_shape[1])

    def test_line_in_patch(self) -> None:
        """
        Checks if the line inside patch.
        """
        line_coords = LineString([(1.0, 1.0), (10.0, 10.0)])
        box_coords = [0.0, 0.0, 11.0, 11.0]
        self.assertTrue(self.nuplan_maps[self.locations[0]]._is_line_record_in_patch(line_coords, box_coords))
        box_coords = [0.0, 0.0, 8.0, 8.0]
        self.assertFalse(self.nuplan_maps[self.locations[0]]._is_line_record_in_patch(line_coords, box_coords))

    def test_line_intersects_patch(self) -> None:
        """
        Checks if line intersects the patch.
        """
        line_coords = LineString([(0.0, 0.0), (10.0, 10.0)])
        box_coords = [0.0, 0.0, 11.0, 11.0]
        self.assertTrue(self.nuplan_maps[self.locations[0]]._is_line_record_in_patch(line_coords, box_coords, 'intersect'))
        box_coords = [11.0, 11.0, 16.0, 16.0]
        self.assertFalse(self.nuplan_maps[self.locations[0]]._is_line_record_in_patch(line_coords, box_coords, 'intersect'))

    def test_polygon_in_patch(self) -> None:
        """
        Checks if polygon is inside patch.
        """
        polygon_coords = Polygon([(1.0, 1.0), (1.0, 10.0), (10.0, 10.0), (1.0, 1.0)])
        box_coords = [0.0, 0.0, 11.0, 11.0]
        self.assertTrue(self.nuplan_maps[self.locations[0]]._is_polygon_record_in_patch(polygon_coords, box_coords))
        box_coords = [0.0, 0.0, 8.0, 8.0]
        self.assertFalse(self.nuplan_maps[self.locations[0]]._is_polygon_record_in_patch(polygon_coords, box_coords))

    def test_polygon_intersects_patch(self) -> None:
        """
        Check if polygon intersects patch.
        """
        polygon_coords = Polygon([(1.0, 1.0), (1.0, 10.0), (10.0, 10.0), (1.0, 1.0)])
        box_coords = [1.0, 1.0, 11.0, 11.0]
        self.assertTrue(self.nuplan_maps[self.locations[0]]._is_polygon_record_in_patch(polygon_coords, box_coords, 'intersect'))
        box_coords = [12.0, 14.0, 15.0, 15.0]
        self.assertFalse(self.nuplan_maps[self.locations[0]]._is_polygon_record_in_patch(polygon_coords, box_coords, 'intersect'))

    def test_mask_for_polygons(self) -> None:
        """
        Checks the mask generated using polygons.
        """
        polygon_coords = MultiPolygon([Polygon([(0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0), (0.0, 0.0)])])
        mask = np.zeros((10, 10))
        map_explorer = NuPlanMapExplorer(self.nuplan_maps[self.locations[0]])
        predicted_mask = map_explorer.mask_for_polygons(polygon_coords, mask)
        expected_mask = np.zeros((10, 10))
        expected_mask[0:3, 0:3] = 1
        np.testing.assert_array_equal(predicted_mask, expected_mask)

    def test_mask_for_lines(self) -> None:
        """Checks the mask generated using lines."""
        line_coords = LineString([(0, 0), (0, 5), (5, 5), (5, 0), (0, 0)])
        mask = np.zeros((10, 10))
        map_explorer = NuPlanMapExplorer(self.nuplan_maps[self.locations[0]])
        predicted_mask = map_explorer.mask_for_lines(line_coords, mask)
        expected_mask = np.zeros((10, 10))
        expected_mask[0:7, 0:7] = 1
        expected_mask[2:4, 2:4] = 0
        expected_mask[6, 6] = 0
        np.testing.assert_array_equal(predicted_mask, expected_mask)

    def test_layers_on_points(self) -> None:
        """
        Checks if returns correct layers given a point.
        """
        with self.assertRaises(Exception):
            self.nuplan_maps[self.locations[3]].layers_on_point(0, 0, ['lane_connectors'])
        self.assertFalse(self.nuplan_maps[self.locations[3]].layers_on_point(0, 0, []))
        layer = self.nuplan_maps[self.locations[2]].layers_on_point(664777.776, 3999698.364, ['lanes_polygons'])
        self.assertEqual(layer['lanes_polygons'], ['63085'])
        layer = self.nuplan_maps[self.locations[3]].layers_on_point(87488.0, 43600.0, ['lanes_polygons'])
        self.assertFalse(layer['lanes_polygons'])

    def test_get_records_in_patch(self) -> None:
        """
        Checks the function of getting all the record token that intersects or within a particular rectangular patch.
        """
        with self.assertRaises(Exception):
            self.nuplan_maps[self.locations[3]].get_records_in_patch([0, 0, 0, 0], ['drivable_area'])
        tokens = self.nuplan_maps[self.locations[3]].get_records_in_patch([0, 0, 0, 0], ['lanes_polygons'])
        self.assertFalse(tokens['lanes_polygons'])
        xmin, ymin, xmax, ymax = self.nuplan_maps[self.locations[3]].get_bounds('lanes_polygons')
        tokens = self.nuplan_maps[self.locations[3]].get_records_in_patch([xmin, ymin, xmax, ymax], ['lanes_polygons'])
        self.assertTrue(tokens['lanes_polygons'])

    def test_get_layer_polygon(self) -> None:
        """Checks the function of retrieving the polygons of a particular layer within the specified patch."""
        with self.assertRaises(Exception):
            self.nuplan_maps[self.locations[3]].get_layer_polygon((0, 0, 0, 0), 0.0, 'drivable_area')
        self.assertFalse(self.nuplan_maps[self.locations[3]].get_layer_polygon((0, 0, 0, 0), 0.0, 'lanes_polygons'))
        xmin, ymin, xmax, ymax = self.nuplan_maps[self.locations[0]].get_bounds('lanes_polygons')
        width = xmax - xmin
        height = ymax - ymin
        patch_box = (xmin + width / 2, ymin + height / 2, height, width)
        patch_angle = 0.0
        self.assertTrue(self.nuplan_maps[self.locations[0]].get_layer_polygon(patch_box, patch_angle, 'lanes_polygons'))

    def test_get_layer_line(self) -> None:
        """Checks the function of retrieving the lines of a particular layer within the specified patch."""
        with self.assertRaises(Exception):
            self.nuplan_maps[self.locations[3]].get_layer_line((0, 0, 0, 0), 0.0, 'drivable_area')
        self.assertFalse(self.nuplan_maps[self.locations[3]].get_layer_line((0, 0, 0, 0), 0.0, 'lanes_polygons'))
        xmin, ymin, xmax, ymax = self.nuplan_maps[self.locations[0]].get_bounds('lanes_polygons')
        width = xmax - xmin
        height = ymax - ymin
        patch_box = (xmin + width / 2, ymin + height / 2, height, width)
        patch_angle = 0.0
        self.assertTrue(self.nuplan_maps[self.locations[0]].get_layer_line(patch_box, patch_angle, 'lanes_polygons'))

@lru_cache(maxsize=1)
def get_test_maps_db() -> IMapsDB:
    """Get a nuPlan maps DB object with default settings to be used in testing."""
    return GPKGMapsDB(map_version=NUPLAN_MAP_VERSION, map_root=NUPLAN_MAPS_ROOT)

class TestMapExplorer(unittest.TestCase):
    """Test NuPlanMapExplorer class."""

    def setUp(self) -> None:
        """
        Initialize the map.
        """
        self.maps_db = get_test_maps_db()
        self.location = 'us-nv-las-vegas-strip'
        self.nuplan_map = NuPlanMapWrapper(maps_db=self.maps_db, map_name=self.location)
        self.nuplan_explore = NuPlanMapExplorer(self.nuplan_map)

    def test_render_layers(self) -> None:
        """
        Checks the function to render layers.
        """
        try:
            self.nuplan_explore.render_layers(self.nuplan_map.vector_layers, alpha=0.5)
        except RuntimeError:
            self.fail('render_layers() raised RuntimeError unexpectedly!')

    def test_render_map_mask(self) -> None:
        """
        Checks the function to render map mask.
        """
        xmin, ymin, xmax, ymax = self.nuplan_map.get_bounds('lanes_polygons')
        width = xmax - xmin
        height = ymax - ymin
        try:
            self.nuplan_explore.render_map_mask((xmin + width / 2, ymin + height / 2, height, width), 0.0, ['lanes_polygons', 'intersections'], (500, 500), (50, 50), 2)
        except RuntimeError:
            self.fail('render_map_mask() raised RuntimeError unexpectedly!')

    def test_render_nearby_roads(self) -> None:
        """
        Checks the function to render nearby roads.
        """
        xmin, ymin, xmax, ymax = self.nuplan_map.get_bounds('lanes_polygons')
        width = xmax - xmin
        height = ymax - ymin
        x = xmin + width / 2 - 921
        y = ymin + height / 2 + 1540
        try:
            self.nuplan_explore.render_nearby_roads(x, y)
        except RuntimeError:
            self.fail('render_nearby_roads() raised RuntimeError unexpectedly!')

@lru_cache(maxsize=1)
def get_test_nuplan_db_path() -> str:
    """Get a single nuPlan DB path to be used in testing."""
    paths = discover_log_dbs(NUPLAN_DB_FILES)
    return cast(str, paths[DEFAULT_TEST_DB_INDEX])

class TestMapManager(unittest.TestCase):
    """
    MapManager test suite.
    """

    @patch('nuplan.common.maps.map_manager.AbstractMapFactory')
    def setUp(self, mock_map_factory: Mock) -> None:
        """
        Initializes the map manager.
        """
        self.map_manager = MapManager(mock_map_factory)

    @patch('nuplan.common.maps.map_manager.AbstractMapFactory')
    def test_initialization(self, mock_map_factory: Mock) -> None:
        """Tests that objects are initialized correctly."""
        map_manager = MapManager(mock_map_factory)
        self.assertEqual(mock_map_factory, map_manager.map_factory)
        self.assertEqual({}, map_manager.maps)

    def test_get_map(self) -> None:
        """Tests that maps are retrieved from cache, if not present created and added to it."""
        map_name = 'map_name'
        self.map_manager.map_factory.build_map_from_name.return_value = 'built_map'
        _map = self.map_manager.get_map(map_name)
        self.map_manager.map_factory.build_map_from_name.assert_called_once_with(map_name)
        self.assertTrue(map_name in self.map_manager.maps)
        self.assertEqual('built_map', _map)
        _ = self.map_manager.get_map(map_name)
        self.map_manager.map_factory.build_map_from_name.assert_called_once_with(map_name)

class NuPlanMapFactory(AbstractMapFactory):
    """
    Factory creating maps from an IMapsDB interface.
    """

    def __init__(self, maps_db: IMapsDB):
        """
        :param maps_db: An IMapsDB instance e.g. GPKGMapsDB.
        """
        self._maps_db = maps_db

    def __reduce__(self) -> Tuple[Type[NuPlanMapFactory], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return (self.__class__, (self._maps_db,))

    def build_map_from_name(self, map_name: str) -> NuPlanMap:
        """
        Builds a map interface given a map name.
        Examples of names: 'sg-one-north', 'us-ma-boston', 'us-nv-las-vegas-strip', 'us-pa-pittsburgh-hazelwood'
        :param map_name: Name of the map.
        :return: The constructed map interface.
        """
        return NuPlanMap(self._maps_db, map_name.replace('.gpkg', ''))

@lru_cache(maxsize=32)
def get_maps_api(map_root: str, map_version: str, map_name: str) -> NuPlanMap:
    """
    Get a NuPlanMap object corresponding to a particular set of parameters.
    :param map_root: The root folder for the map data.
    :param map_version: The map version to load.
    :param map_name: The map name to load.
    :return: The loaded NuPlanMap object.
    """
    maps_db = get_maps_db(map_root, map_version)
    return NuPlanMap(maps_db, map_name.replace('.gpkg', ''))

@pytest.fixture()
def map_factory() -> NuPlanMapFactory:
    """Fixture loading ta returning a map factory"""
    maps_db = get_test_maps_db()
    return NuPlanMapFactory(maps_db)

class NuPlanScenario(AbstractScenario):
    """Scenario implementation for the nuPlan dataset that is used in training and simulation."""

    def __init__(self, data_root: str, log_file_load_path: str, initial_lidar_token: str, initial_lidar_timestamp: int, scenario_type: str, map_root: str, map_version: str, map_name: str, scenario_extraction_info: Optional[ScenarioExtractionInfo], ego_vehicle_parameters: VehicleParameters, sensor_root: Optional[str]=None) -> None:
        """
        Initialize the nuPlan scenario.
        :param data_root: The prefix for the log file. e.g. "/data/root/nuplan". For remote paths, this is where the file will be downloaded if necessary.
        :param log_file_load_path: Name of the log that this scenario belongs to. e.g. "/data/sets/nuplan-v1.1/splits/mini/2021.07.16.20.45.29_veh-35_01095_01486.db", "s3://path/to/db.db"
        :param initial_lidar_token: Token of the scenario's initial lidarpc.
        :param initial_lidar_timestamp: The timestamp of the initial lidarpc.
        :param scenario_type: Type of scenario (e.g. ego overtaking).
        :param map_root: The root path for the map db
        :param map_version: The version of maps to load
        :param map_name: The map name to use for the scenario
        :param scenario_extraction_info: Structure containing information used to extract the scenario.
            None means the scenario has no length and it is comprised only by the initial lidarpc.
        :param ego_vehicle_parameters: Structure containing the vehicle parameters.
        :param sensor_root: The root path for the sensor blobs.
        """
        self._local_store: Optional[LocalStore] = None
        self._remote_store: Optional[S3Store] = None
        self._data_root = data_root
        self._log_file_load_path = log_file_load_path
        self._initial_lidar_token = initial_lidar_token
        self._initial_lidar_timestamp = initial_lidar_timestamp
        self._scenario_type = scenario_type
        self._map_root = map_root
        self._map_version = map_version
        self._map_name = map_name
        self._scenario_extraction_info = scenario_extraction_info
        self._ego_vehicle_parameters = ego_vehicle_parameters
        self._sensor_root = sensor_root
        if self._scenario_extraction_info is not None:
            skip_rows = 1.0 / self._scenario_extraction_info.subsample_ratio
            if abs(int(skip_rows) - skip_rows) > 0.001:
                raise ValueError(f'Subsample ratio is not valid. Must resolve to an integer number of skipping rows, instead received {self._scenario_extraction_info.subsample_ratio}, which would skip {skip_rows} rows.')
        self._database_row_interval = 0.05
        self._log_file = download_file_if_necessary(self._data_root, self._log_file_load_path)
        self._log_name: str = absolute_path_to_log_name(self._log_file)

    def __reduce__(self) -> Tuple[Type[NuPlanScenario], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return (self.__class__, (self._data_root, self._log_file_load_path, self._initial_lidar_token, self._initial_lidar_timestamp, self._scenario_type, self._map_root, self._map_version, self._map_name, self._scenario_extraction_info, self._ego_vehicle_parameters, self._sensor_root))

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """Inherited, see superclass."""
        return self._ego_vehicle_parameters

    @cached_property
    def _lidarpc_tokens(self) -> List[str]:
        """
        :return: list of lidarpc tokens in the scenario
        """
        if self._scenario_extraction_info is None:
            return [self._initial_lidar_token]
        lidarpc_tokens = list(extract_sensor_tokens_as_scenario(self._log_file, get_lidarpc_sensor_data(), self._initial_lidar_timestamp, self._scenario_extraction_info))
        return cast(List[str], lidarpc_tokens)

    @cached_property
    def _route_roadblock_ids(self) -> List[str]:
        """
        return: Route roadblock ids extracted from expert trajectory.
        """
        expert_trajectory = list(self._extract_expert_trajectory())
        return get_roadblock_ids_from_trajectory(self.map_api, expert_trajectory)

    @property
    def token(self) -> str:
        """Inherited, see superclass."""
        return self._initial_lidar_token

    @property
    def log_name(self) -> str:
        """Inherited, see superclass."""
        return self._log_name

    @property
    def scenario_name(self) -> str:
        """Inherited, see superclass."""
        return self.token

    @property
    def scenario_type(self) -> str:
        """Inherited, see superclass."""
        return self._scenario_type

    @property
    def map_api(self) -> AbstractMap:
        """Inherited, see superclass."""
        return get_maps_api(self._map_root, self._map_version, self._map_name)

    @property
    def map_root(self) -> str:
        """Get the map root folder."""
        return self._map_root

    @property
    def map_version(self) -> str:
        """Get the map version."""
        return self._map_version

    @property
    def database_interval(self) -> float:
        """Inherited, see superclass."""
        if self._scenario_extraction_info is None:
            return 0.05
        return float(0.05 / self._scenario_extraction_info.subsample_ratio)

    def get_number_of_iterations(self) -> int:
        """Inherited, see superclass."""
        return len(self._lidarpc_tokens)

    def get_lidar_to_ego_transform(self) -> Transform:
        """Inherited, see superclass."""
        return get_sensor_transform_matrix_for_sensor_data_token_from_db(self._log_file, get_lidarpc_sensor_data(), self._initial_lidar_token)

    def get_mission_goal(self) -> Optional[StateSE2]:
        """Inherited, see superclass."""
        return get_mission_goal_for_sensor_data_token_from_db(self._log_file, get_lidarpc_sensor_data(), self._initial_lidar_token)

    def get_route_roadblock_ids(self) -> List[str]:
        """Inherited, see superclass."""
        roadblock_ids = get_roadblock_ids_for_lidarpc_token_from_db(self._log_file, self._initial_lidar_token)
        assert roadblock_ids is not None, 'Unable to find Roadblock ids for current scenario'
        return cast(List[str], roadblock_ids)

    def get_expert_goal_state(self) -> StateSE2:
        """Inherited, see superclass."""
        return get_statese2_for_lidarpc_token_from_db(self._log_file, self._lidarpc_tokens[-1])

    def get_time_point(self, iteration: int) -> TimePoint:
        """Inherited, see superclass."""
        return TimePoint(time_us=get_sensor_data_token_timestamp_from_db(self._log_file, get_lidarpc_sensor_data(), self._lidarpc_tokens[iteration]))

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """Inherited, see superclass."""
        return get_ego_state_for_lidarpc_token_from_db(self._log_file, self._lidarpc_tokens[iteration])

    def get_tracked_objects_at_iteration(self, iteration: int, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations(), f'Iteration is out of scenario: {iteration}!'
        return DetectionsTracks(extract_tracked_objects(self._lidarpc_tokens[iteration], self._log_file, future_trajectory_sampling))

    def get_tracked_objects_within_time_window_at_iteration(self, iteration: int, past_time_horizon: float, future_time_horizon: float, filter_track_tokens: Optional[Set[str]]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations(), f'Iteration is out of scenario: {iteration}!'
        return DetectionsTracks(extract_tracked_objects_within_time_window(self._lidarpc_tokens[iteration], self._log_file, past_time_horizon, future_time_horizon, filter_track_tokens, future_trajectory_sampling))

    def get_sensors_at_iteration(self, iteration: int, channels: Optional[List[SensorChannel]]=None) -> Sensors:
        """Inherited, see superclass."""
        channels = [LidarChannel.MERGED_PC] if channels is None else channels
        lidar_pc = next(get_sensor_data_from_sensor_data_tokens_from_db(self._log_file, get_lidarpc_sensor_data(), LidarPc, [self._lidarpc_tokens[iteration]]))
        return self._get_sensor_data_from_lidar_pc(cast(LidarPc, lidar_pc), channels)

    def get_future_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True):
            yield TimePoint(lidar_pc.timestamp)

    def get_past_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield TimePoint(lidar_pc.timestamp)

    def get_ego_past_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._database_row_interval)
        return cast(Generator[EgoState, None, None], get_sampled_ego_states_from_db(self._log_file, self._lidarpc_tokens[iteration], get_lidarpc_sensor_data(), indices, future=False))

    def get_ego_future_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._database_row_interval)
        return cast(Generator[EgoState, None, None], get_sampled_ego_states_from_db(self._log_file, self._lidarpc_tokens[iteration], get_lidarpc_sensor_data(), indices, future=True))

    def get_past_tracked_objects(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield DetectionsTracks(extract_tracked_objects(lidar_pc.token, self._log_file, future_trajectory_sampling))

    def get_future_tracked_objects(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True):
            yield DetectionsTracks(extract_tracked_objects(lidar_pc.token, self._log_file, future_trajectory_sampling))

    def get_past_sensors(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, channels: Optional[List[SensorChannel]]=None) -> Generator[Sensors, None, None]:
        """Inherited, see superclass."""
        channels = [LidarChannel.MERGED_PC] if channels is None else channels
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield self._get_sensor_data_from_lidar_pc(lidar_pc, channels)

    def get_traffic_light_status_at_iteration(self, iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """Inherited, see superclass."""
        token = self._lidarpc_tokens[iteration]
        return cast(Generator[TrafficLightStatusData, None, None], get_traffic_light_status_for_lidarpc_token_from_db(self._log_file, token))

    def get_past_traffic_light_status_history(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TrafficLightStatuses, None, None]:
        """
        Gets past traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the past.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the past.
        """
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield TrafficLightStatuses(list(get_traffic_light_status_for_lidarpc_token_from_db(self._log_file, lidar_pc.token)))

    def get_future_traffic_light_status_history(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TrafficLightStatuses, None, None]:
        """
        Gets future traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the future.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the future.
        """
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True):
            yield TrafficLightStatuses(list(get_traffic_light_status_for_lidarpc_token_from_db(self._log_file, lidar_pc.token)))

    def get_scenario_tokens(self) -> List[str]:
        """Return the list of lidarpc tokens from the DB that are contained in the scenario."""
        return self._lidarpc_tokens

    def _find_matching_lidar_pcs(self, iteration: int, num_samples: Optional[int], time_horizon: float, look_into_future: bool) -> Generator[LidarPc, None, None]:
        """
        Find the best matching lidar_pcs to the desired samples and time horizon
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future, if None it will be deduced from the DB
        :param time_horizon: the desired horizon to the future
        :param look_into_future: if True, we will iterate into next lidar_pc otherwise we will iterate through prev
        :return: lidar_pcs matching to database indices
        """
        num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._database_row_interval)
        return cast(Generator[LidarPc, None, None], get_sampled_lidarpcs_from_db(self._log_file, self._lidarpc_tokens[iteration], get_lidarpc_sensor_data(), indices, look_into_future))

    def _extract_expert_trajectory(self, max_future_seconds: int=60) -> Generator[EgoState, None, None]:
        """
        Extract expert trajectory with specified time parameters. If initial lidar pc does not have enough history/future
            only available time will be extracted
        :param max_future_seconds: time to future which should be considered for route extraction [s]
        :return: list of expert ego states
        """
        minimal_required_future_time_available = 0.5
        end_log_time_us = get_end_sensor_time_from_db(self._log_file, get_lidarpc_sensor_data())
        max_future_time = min((end_log_time_us - self._initial_lidar_timestamp) * 1e-06, max_future_seconds)
        if max_future_time < minimal_required_future_time_available:
            return
        for traj in self.get_ego_future_trajectory(0, max_future_time):
            yield traj

    def _create_blob_store_if_needed(self) -> Tuple[LocalStore, Optional[S3Store]]:
        """
        A convenience method that creates the blob stores if it's not already created.
        :return: The created or cached LocalStore and S3Store objects.
        """
        if self._local_store is not None and self._remote_store is not None:
            return (self._local_store, self._remote_store)
        if self._sensor_root is None:
            raise ValueError('sensor_root is not set. Please set the sensor_root to access sensor data.')
        Path(self._sensor_root).mkdir(exist_ok=True)
        self._local_store = LocalStore(self._sensor_root)
        if os.getenv('NUPLAN_DATA_STORE', '') == 's3':
            s3_url = os.getenv('NUPLAN_DATA_ROOT_S3_URL', '')
            self._remote_store = S3Store(os.path.join(s3_url, 'sensor_blobs'), show_progress=True)
        return (self._local_store, self._remote_store)

    def _get_sensor_data_from_lidar_pc(self, lidar_pc: LidarPc, channels: List[SensorChannel]) -> Sensors:
        """
        Loads Sensor data given a database LidarPC object.
        :param lidar_pc: The lidar_pc for which to grab the point cloud.
        :param channels: The sensor channels to return.
        :return: The corresponding sensor data.
        """
        local_store, remote_store = self._create_blob_store_if_needed()
        retrieved_images = get_images_from_lidar_tokens(self._log_file, [lidar_pc.token], [cast(str, channel.value) for channel in channels])
        lidar_pcs = {LidarChannel.MERGED_PC: load_point_cloud(cast(LidarPc, lidar_pc), local_store, remote_store)} if LidarChannel.MERGED_PC in channels else None
        images = {CameraChannel[image.channel]: load_image(image, local_store, remote_store) for image in retrieved_images}
        return Sensors(pointcloud=lidar_pcs, images=images if images else None)

class NuPlanScenarioBuilder(AbstractScenarioBuilder):
    """Builder class for constructing nuPlan scenarios for training and simulation."""

    def __init__(self, data_root: str, map_root: str, sensor_root: str, db_files: Optional[Union[List[str], str]], map_version: str, include_cameras: bool=False, max_workers: Optional[int]=None, verbose: bool=True, scenario_mapping: Optional[ScenarioMapping]=None, vehicle_parameters: Optional[VehicleParameters]=None):
        """
        Initialize scenario builder that filters and retrieves scenarios from the nuPlan dataset.
        :param data_root: Local data root for loading (or storing downloaded) the log databases.
                          If `db_files` is not None, all downloaded databases will be stored to this data root.
                          E.g.: /data/sets/nuplan
        :param map_root: Local map root for loading (or storing downloaded) the map database.
        :param sensor_root: Local map root for loading (or storing downloaded) the sensor blobs.
        :param db_files: Path to load the log database(s) from.
                         It can be a local/remote path to a single database, list of databases or dir of databases.
                         If None, all database filenames found under `data_root` will be used.
                         E.g.: /data/sets/nuplan/nuplan-v1.1/splits/mini/2021.10.11.08.31.07_veh-50_01750_01948.db
        :param map_version: Version of map database to load. The map database is passed to each loaded log database.
        :param include_cameras: If true, make camera data available in scenarios.
        :param max_workers: Maximum number of workers to use when loading the databases concurrently.
                            Only used when the number of databases to load is larger than this parameter.
        :param verbose: Whether to print progress and details during the database loading and scenario building.
        :param scenario_mapping: Mapping of scenario types to extraction information.
        :param vehicle_parameters: Vehicle parameters for this db.
        """
        self._data_root = data_root
        self._map_root = map_root
        self._sensor_root = sensor_root
        self._db_files = discover_log_dbs(data_root if db_files is None else db_files)
        self._map_version = map_version
        self._include_cameras = include_cameras
        self._max_workers = max_workers
        self._verbose = verbose
        self._scenario_mapping = scenario_mapping if scenario_mapping is not None else ScenarioMapping({}, None)
        self._vehicle_parameters = vehicle_parameters if vehicle_parameters is not None else get_pacifica_parameters()

    def __reduce__(self) -> Tuple[Type[NuPlanScenarioBuilder], Tuple[Any, ...]]:
        """
        :return: tuple of class and its constructor parameters, this is used to pickle the class
        """
        return (self.__class__, (self._data_root, self._map_root, self._sensor_root, self._db_files, self._map_version, self._include_cameras, self._max_workers, self._verbose, self._scenario_mapping, self._vehicle_parameters))

    @classmethod
    def get_scenario_type(cls) -> Type[AbstractScenario]:
        """Inherited. See superclass."""
        return cast(Type[AbstractScenario], NuPlanScenario)

    def get_map_factory(self) -> AbstractMapFactory:
        """Inherited. See superclass."""
        return NuPlanMapFactory(get_maps_db(self._map_root, self._map_version))

    def _aggregate_dicts(self, dicts: List[ScenarioDict]) -> ScenarioDict:
        """
        Combines multiple scenario dicts into a single dictionary by concatenating lists of matching scenario names.
        Sample input:
            [{"a": [1, 2, 3], "b": [2, 3, 4]}, {"b": [3, 4, 5], "c": [4, 5]}]
        Sample output:
            {"a": [1, 2, 3], "b": [2, 3, 4, 3, 4, 5], "c": [4, 5]}
        :param dicts: The list of dictionaries to concatenate.
        :return: The concatenated dictionaries.
        """
        output_dict = dicts[0]
        for merge_dict in dicts[1:]:
            for key in merge_dict:
                if key not in output_dict:
                    output_dict[key] = merge_dict[key]
                else:
                    output_dict[key] += merge_dict[key]
        return output_dict

    def _create_scenarios(self, scenario_filter: ScenarioFilter, worker: WorkerPool) -> ScenarioDict:
        """
        Creates a scenario dictionary with scenario type as key and list of scenarios for each type.
        :param scenario_filter: Structure that contains scenario filtering instructions.
        :param worker: Worker pool for concurrent scenario processing.
        :return: Constructed scenario dictionary.
        """
        allowable_log_names = set(scenario_filter.log_names) if scenario_filter.log_names is not None else None
        map_parameters = [GetScenariosFromDbFileParams(data_root=self._data_root, log_file_absolute_path=log_file, expand_scenarios=scenario_filter.expand_scenarios, map_root=self._map_root, map_version=self._map_version, scenario_mapping=self._scenario_mapping, vehicle_parameters=self._vehicle_parameters, filter_tokens=scenario_filter.scenario_tokens, filter_types=scenario_filter.scenario_types, filter_map_names=scenario_filter.map_names, remove_invalid_goals=scenario_filter.remove_invalid_goals, sensor_root=self._sensor_root, include_cameras=self._include_cameras, verbose=self._verbose) for log_file in self._db_files if allowable_log_names is None or absolute_path_to_log_name(log_file) in allowable_log_names]
        if len(map_parameters) == 0:
            logger.warning('No log files found! This may mean that you need to set your environment, or that all of your log files got filtered out on this worker.')
            return {}
        dicts = worker_map(worker, get_scenarios_from_log_file, map_parameters)
        return self._aggregate_dicts(dicts)

    def _create_filter_wrappers(self, scenario_filter: ScenarioFilter, worker: WorkerPool) -> List[FilterWrapper]:
        """
        Creates a series of filter wrappers that will be applied sequentially to construct the list of scenarios.
        :param scenario_filter: Structure that contains scenario filtering instructions.
        :param worker: Worker pool for concurrent scenario processing.
        :return: Series of filter wrappers.
        """
        filters = [FilterWrapper(fn=partial(filter_num_scenarios_per_type, num_scenarios_per_type=scenario_filter.num_scenarios_per_type, randomize=scenario_filter.shuffle), enable=scenario_filter.num_scenarios_per_type is not None, name='num_scenarios_per_type'), FilterWrapper(fn=partial(filter_total_num_scenarios, limit_total_scenarios=scenario_filter.limit_total_scenarios, randomize=scenario_filter.shuffle), enable=scenario_filter.limit_total_scenarios is not None, name='limit_total_scenarios'), FilterWrapper(fn=partial(filter_scenarios_by_timestamp, timestamp_threshold_s=scenario_filter.timestamp_threshold_s), enable=scenario_filter.timestamp_threshold_s is not None, name='filter_scenarios_by_timestamp'), FilterWrapper(fn=partial(filter_non_stationary_ego, minimum_threshold=scenario_filter.ego_displacement_minimum_m), enable=scenario_filter.ego_displacement_minimum_m is not None, name='filter_non_stationary_ego'), FilterWrapper(fn=partial(filter_ego_starts, speed_threshold=scenario_filter.ego_start_speed_threshold, speed_noise_tolerance=scenario_filter.speed_noise_tolerance), enable=scenario_filter.ego_start_speed_threshold is not None, name='filter_ego_starts'), FilterWrapper(fn=partial(filter_ego_stops, speed_threshold=scenario_filter.ego_stop_speed_threshold, speed_noise_tolerance=scenario_filter.speed_noise_tolerance), enable=scenario_filter.ego_stop_speed_threshold is not None, name='filter_ego_stops'), FilterWrapper(fn=partial(filter_fraction_lidarpc_tokens_in_set, token_set_path=scenario_filter.token_set_path, fraction_threshold=scenario_filter.fraction_in_token_set_threshold), enable=scenario_filter.token_set_path is not None and scenario_filter.fraction_in_token_set_threshold is not None, name='filter_fraction_lidarpc_tokens_in_set'), FilterWrapper(fn=partial(filter_ego_has_route, map_radius=scenario_filter.ego_route_radius), enable=scenario_filter.ego_route_radius is not None, name='filter_ego_has_route')]
        return filters

    def get_scenarios(self, scenario_filter: ScenarioFilter, worker: WorkerPool) -> List[AbstractScenario]:
        """Implemented. See interface."""
        scenario_dict = self._create_scenarios(scenario_filter, worker)
        filter_wrappers = self._create_filter_wrappers(scenario_filter, worker)
        for filter_wrapper in filter_wrappers:
            scenario_dict = filter_wrapper.run(scenario_dict)
        return scenario_dict_to_list(scenario_dict, shuffle=scenario_filter.shuffle)

    @property
    def repartition_strategy(self) -> RepartitionStrategy:
        """Implemented. See interface."""
        return RepartitionStrategy.REPARTITION_FILE_DISK

class SubmissionPlanner:
    """
    Class holding a planner and exposing functionalities as a server. The services are planner initialization and
    trajectory computation.
    """

    def __init__(self, planner_config: DictConfig):
        """
        Prepares the planner and the server. The communication port is read from an environmental variable.
        :param planner_config: The planner configuration to instantiate the planner
        """
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        map_version = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')
        map_factory = NuPlanMapFactory(GPKGMapsDB(map_version=map_version, map_root=os.path.join(os.getenv('NUPLAN_DATA_ROOT', '~/nuplan/dataset'), 'maps')))
        map_manager = MapManager(map_factory)
        chpb_grpc.add_DetectionTracksChallengeServicer_to_server(DetectionTracksChallengeServicer(planner_config, map_manager), self.server)
        port = os.getenv('SUBMISSION_CONTAINER_PORT', 50051)
        logger.info(f'Submission container starting with port {port}')
        if not port:
            raise RuntimeError("Environment variable not specified: 'SUBMISSION_CONTAINER_PORT'")
        self.server.add_insecure_port(f'[::]:{port}')

    def serve(self) -> None:
        """Starts the server."""
        logger.info('Server starting...')
        self.server.start()
        logger.info('Server started!')
        self.server.wait_for_termination()
        logger.info('Server terminated!')

class TestDetectionTracksChallengeServicer(TestCase):
    """Tests the DetectionTracksChallengeServicer class"""

    @patch('nuplan.submission.challenge_servicers.MapManager', return_value='map')
    def setUp(self, mock_map_manager: Mock) -> None:
        """Sets variables for testing"""
        mock_planner_cfg = {'planner1': Mock()}
        self.servicer = DetectionTracksChallengeServicer(mock_planner_cfg, mock_map_manager)

    @patch('nuplan.submission.challenge_servicers.MapManager', return_value='map')
    def test_initialization(self, mock_map_manager: Mock) -> None:
        """Tests that the class is initialized as intended."""
        mock_planner_cfg = Mock()
        mock_servicer = DetectionTracksChallengeServicer(mock_planner_cfg, mock_map_manager)
        self.assertEqual(mock_servicer.planner, None)
        self.assertEqual(mock_servicer._planner_config, mock_planner_cfg)
        self.assertEqual(mock_servicer.map_manager, mock_map_manager)

    @patch('nuplan.submission.challenge_servicers.MapManager')
    @patch('nuplan.submission.challenge_servicers.PlannerInitialization', autospec=True)
    @patch('nuplan.submission.challenge_servicers.se2_from_proto_se2')
    @patch('nuplan.submission.challenge_servicers.build_planners')
    def test_InitializePlanner(self, builder: Mock, mock_s2_conversion: Mock, mock_planner_initialization: Mock, mock_map_manager: Mock) -> None:
        """Tests the client call to InitializePlanner."""
        mock_input = Mock()
        mock_context = Mock()
        mock_map_api = Mock()
        mock_planner_initialization.return_value = 'planner_initialization'
        mock_map_manager.return_value = mock_map_api
        builder.return_value = [Mock()]
        self.servicer.InitializePlanner(mock_input, mock_context)
        calls = [call(mock_input.mission_goal)]
        mock_s2_conversion.assert_has_calls(calls)
        map_calls = [call(mock_input.map_name), call().initialize_all_layers()]
        self.servicer.map_manager.get_map.assert_has_calls(map_calls)
        self.servicer.planner.initialize.assert_called_once_with('planner_initialization')

    def test_ComputeTrajectory_uninitialized(self) -> None:
        """Tests the client call to ComputeTrajectory fails if the planner wasn't initialized."""
        with self.assertRaises(AssertionError, msg='Planner has not been initialized. Please call InitializePlanner'):
            self.servicer.simulation_history_buffers = []
            self.servicer.ComputeTrajectory(Mock(), Mock())

    @patch('nuplan.submission.challenge_servicers.proto_traj_from_inter_traj')
    def test_ComputeTrajectory(self, proto_traj_from_inter_traj: Mock) -> None:
        """Tests the client call to ComputeTrajectory."""
        mock_context = Mock()
        self.servicer.planner = Mock(spec=AbstractPlanner)
        self.servicer.planner.compute_trajectory.return_value = 'trajectory'
        self.servicer.simulation_history_buffer = 'buffer_1'
        self.servicer._initialized = True
        history_buffer = MagicMock(ego_states=['ego_state_1'], observations=['observation_1'])
        simulation_iteration = MagicMock(time_us=123, index=234)
        mock_serialized_input = MagicMock(simulation_history_buffer=history_buffer, simulation_iteration=simulation_iteration)
        with patch.object(self.servicer, '_build_planner_input', autospec=True) as build_planner_input:
            result = self.servicer.ComputeTrajectory(mock_serialized_input, mock_context)
            build_planner_input.assert_called_with(mock_serialized_input, 'buffer_1')
            self.servicer.planner.compute_trajectory.assert_called_with(build_planner_input.return_value)
            proto_traj_from_inter_traj.assert_called_with(self.servicer.planner.compute_trajectory.return_value)
            self.assertEqual(proto_traj_from_inter_traj.return_value, result)

    @patch('nuplan.submission.challenge_servicers.SimulationIteration', autospec=True)
    @patch('nuplan.submission.challenge_servicers.TimePoint', autospec=True)
    def test__extract_simulation_iteration(self, time_point: Mock, simulation_iteration: Mock) -> None:
        """Tests extraction of simulation iteration data from serialized message"""
        mock_iteration = Mock(time_us=123, index=456, spec_set=SerializedSimulationIteration)
        mock_message = Mock(simulation_iteration=mock_iteration, spec_set=SerializedPlannerInput)
        result = self.servicer._extract_simulation_iteration(mock_message)
        time_point.assert_called_once_with(123)
        simulation_iteration.assert_called_once_with(time_point.return_value, 456)
        self.assertEqual(simulation_iteration.return_value, result)

    @patch('pickle.loads')
    @patch('nuplan.submission.challenge_servicers.PlannerInput', autospec=True)
    @patch('nuplan.submission.challenge_servicers.SimulationHistoryBuffer', autospec=True)
    def test__build_planner_input(self, buffer: MagicMock, planner_input: Mock, loads: Mock) -> None:
        """Tests that planner input is correctly deserialized"""
        mock_serialized_buffer = Mock(ego_states=['ego_state'], observations=['observations'], sample_interval=['sample_interval'], spec_set=SerializedHistoryBuffer)
        mock_message = MagicMock(simulation_history_buffer=mock_serialized_buffer, spec_set=SerializedPlannerInput)
        loads.side_effect = ['deserialized_ego_state', 'deserialized_observations']
        with patch.object(self.servicer, '_extract_simulation_iteration', autospec=True) as extract_iteration:
            result = self.servicer._build_planner_input(mock_message, buffer)
            extract_iteration.assert_called_with(mock_message)
            loads.assert_has_calls([call('ego_state'), call('observations')])
            buffer.extend.assert_called_once_with(['deserialized_ego_state'], ['deserialized_observations'])
            self.assertEqual(planner_input.return_value, result)

    @patch('pickle.loads')
    @patch('nuplan.submission.challenge_servicers.PlannerInput', autospec=True)
    @patch('nuplan.submission.challenge_servicers.SimulationHistoryBuffer', autospec=True)
    def test__build_planner_input_no_buffer(self, buffer: MagicMock, planner_input: Mock, loads: Mock) -> None:
        """Tests that planner input is correctly deserialized"""
        mock_serialized_buffer = Mock(ego_states=['ego_state'], observations=['observations'], sample_interval=['sample_interval'], spec_set=SerializedHistoryBuffer)
        mock_message = MagicMock(simulation_history_buffer=mock_serialized_buffer, spec_set=SerializedPlannerInput)
        loads.side_effect = ['deserialized_ego_state', 'deserialized_observations']
        with patch.object(self.servicer, '_extract_simulation_iteration', autospec=True):
            self.servicer.simulation_history_buffers = [mock_serialized_buffer]
            result = self.servicer._build_planner_input(mock_message, None)
            buffer.initialize_from_list.assert_called_once_with(1, ['deserialized_ego_state'], ['deserialized_observations'], ['sample_interval'])
            self.assertEqual(planner_input.return_value, result)

