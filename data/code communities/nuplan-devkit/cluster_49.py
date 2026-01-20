# Cluster 49

def create_binary_masks(array: npt.NDArray[np.uint8], map_layer: MapLayerMeta, layer_dir: str) -> None:
    """
    Creates the binary mask for a given map layer in a given map version and
    stores it in the cache.
    :param array: Map array to write to binary.
    :param map_layer: Map layer to create the masks for.
    :param layer_dir: Directory where binary masks will be stored.
    """
    if len(array.shape) == 3:
        array = array[:, :, 0]
    if map_layer.is_binary:
        array[array < 255] = 0
        array[array == 255] = 1
    destination = os.path.join(layer_dir, '{}')
    logger.debug('Writing binary mask to {}...'.format(destination.format(map_layer.binary_mask_name)))
    with open(destination.format(map_layer.binary_mask_name), 'wb') as f:
        f.write(array.tobytes())
    logger.debug('Writing binary mask to {} done.'.format(destination.format(map_layer.binary_mask_name)))
    if map_layer.can_dilate:
        logger.debug('Writing joint distance mask to {}...'.format(destination.format(map_layer.binary_joint_dist_name)))
        joint_distances = compute_joint_distance_matrix(array, map_layer.precision)
        with open(destination.format(map_layer.binary_joint_dist_name), 'wb') as f:
            f.write(joint_distances.tobytes())
        del joint_distances
        del array
        logger.debug('Writing joint distance mask to {} done.'.format(destination.format(map_layer.binary_joint_dist_name)))

def compute_joint_distance_matrix(array: npt.NDArray[np.uint8], precision: float) -> npt.NDArray[np.float64]:
    """
    For each pixel in `array`, computes the physical distance to the nearest
    mask boundary. Distances from a 0 to the boundary are returned as positive
    values, and distances from a 1 to the boundary are returned as negative
    values.
    :param array: Binary array of pixel values.
    :param precision: Meters per pixel.
    :return: The physical distance to the nearest mask boundary.
    """
    distances_0_to_boundary = cv2.distanceTransform((1.0 - array).astype(np.uint8), cv2.DIST_L2, 5)
    distances_0_to_boundary[distances_0_to_boundary > 0] -= 0.5
    distances_0_to_boundary = (distances_0_to_boundary * precision).astype(np.float32)
    distances_1_to_boundary = cv2.distanceTransform(array.astype(np.uint8), cv2.DIST_L2, 5)
    distances_1_to_boundary[distances_1_to_boundary > 0] -= 0.5
    distances_1_to_boundary = (distances_1_to_boundary * precision).astype(np.float32)
    return distances_0_to_boundary - distances_1_to_boundary

class GPKGMapsDB(IMapsDB):
    """GPKG MapsDB implementation."""

    def __init__(self, map_version: str, map_root: str) -> None:
        """
        Constructor.
        :param map_version: Version of map.
        :param map_root: Root folder of the maps.
        """
        self._map_version = map_version
        self._map_root = map_root
        self._blob_store = BlobStoreCreator.create_mapsdb(map_root=self._map_root)
        version_file = self._blob_store.get(f'{self._map_version}.json')
        self._metadata = json.load(version_file)
        self._map_dimensions = MAP_DIMENSIONS
        self._max_attempts = MAX_ATTEMPTS
        self._seconds_between_attempts = SECONDS_BETWEEN_ATTEMPTS
        self._map_lock_dir = os.path.join(self._map_root, '.maplocks')
        os.makedirs(self._map_lock_dir, exist_ok=True)
        self._load_map_data()

    def __reduce__(self) -> Tuple[Type['GPKGMapsDB'], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        This object is reconstructed by pickle to avoid serializing potentially large state/caches.
        :return: Object type and constructor arguments to be used.
        """
        return (self.__class__, (self._map_version, self._map_root))

    def _load_map_data(self) -> None:
        """Load all available maps once to trigger automatic downloading if the maps are loaded for the first time."""
        for location in MAP_LOCATIONS:
            self.load_vector_layer(location, DUMMY_LOAD_LAYER)

    @property
    def version_names(self) -> List[str]:
        """
        Lists the map version names for all valid map locations, e.g.
        ['9.17.1964', '9.12.1817', '9.15.1915', '9.17.1937']
        """
        return [self._metadata[location]['version'] for location in self.get_locations()]

    def get_map_version(self) -> str:
        """Inherited, see superclass."""
        return self._map_version

    def get_version(self, location: str) -> str:
        """Inherited, see superclass."""
        return str(self._metadata[location]['version'])

    def _get_shape(self, location: str, layer_name: str) -> List[int]:
        """
        Gets the shape of a layer given the map location and layer name.
        :param location: Name of map location, e.g. "sg-one-north". See `self.get_locations()`.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        """
        if layer_name == 'intensity':
            return self._metadata[location]['layers']['Intensity']['shape']
        else:
            return list(self._map_dimensions[location])

    def _get_transform_matrix(self, location: str, layer_name: str) -> npt.NDArray[np.float64]:
        """
        Get transformation matrix of a layer given location and layer name.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        """
        return np.array(self._metadata[location]['layers'][layer_name]['transform_matrix'])

    @staticmethod
    def is_binary(layer_name: str) -> bool:
        """
        Checks if the layer is binary.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        """
        return layer_name in ['drivable_area', 'intersection', 'pedestrian_crossing', 'walkway', 'walk_way']

    @staticmethod
    def _can_dilate(layer_name: str) -> bool:
        """
        If the layer can be dilated.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        """
        return layer_name in ['drivable_area']

    def get_locations(self) -> Sequence[str]:
        """
        Gets the list of available location in this GPKGMapsDB version.
        """
        return self._metadata.keys()

    def layer_names(self, location: str) -> Sequence[str]:
        """Inherited, see superclass."""
        gpkg_layers = self._metadata[location]['layers'].keys()
        return list(filter(lambda x: '_distance_px' not in x, gpkg_layers))

    def load_layer(self, location: str, layer_name: str) -> MapLayer:
        """Inherited, see superclass."""
        if layer_name == 'intensity':
            layer_name = 'Intensity'
        is_bin = self.is_binary(layer_name)
        can_dilate = self._can_dilate(layer_name)
        layer_data = self._get_layer_matrix(location, layer_name)
        transform_matrix = self._get_transform_matrix(location, layer_name)
        precision = 1 / transform_matrix[0, 0]
        layer_meta = MapLayerMeta(name=layer_name, md5_hash='not_used_for_gpkg_mapsdb', can_dilate=can_dilate, is_binary=is_bin, precision=precision)
        distance_matrix = None
        return MapLayer(data=layer_data, metadata=layer_meta, joint_distance=distance_matrix, transform_matrix=transform_matrix)

    def _wait_for_expected_filesize(self, path_on_disk: str, location: str) -> None:
        """
        Waits until the file at `path_on_disk` is exactly `expected_size` bytes.
        :param path_on_disk: Path of the file being downloaded.
        :param location: Location to which the file belongs.
        """
        if isinstance(self._blob_store, LocalStore):
            return
        s3_bucket = self._blob_store._remote._bucket
        s3_key = os.path.join(self._blob_store._remote._prefix, self._get_gpkg_file_path(location))
        client = get_s3_client()
        map_file_size = client.head_object(Bucket=s3_bucket, Key=s3_key).get('ContentLength', 0)
        for _ in range(self._max_attempts):
            if os.path.getsize(path_on_disk) == map_file_size:
                break
            time.sleep(self._seconds_between_attempts)
        if os.path.getsize(path_on_disk) != map_file_size:
            raise GPKGMapsDBException(f'Waited {self._max_attempts * self._seconds_between_attempts} seconds for file {path_on_disk} to reach {map_file_size}, but size is now {os.path.getsize(path_on_disk)}')

    def _safe_save_layer(self, layer_lock_file: str, file_path: str) -> None:
        """
        Safely download the file.
        :param layer_lock_file: Path to lock file.
        :param file_path: Path of the file being downloaded.
        """
        fd = open(layer_lock_file, 'w')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            _ = self._blob_store.save_to_disk(file_path, check_for_compressed=True)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()

    @lru_cache(maxsize=16)
    def load_vector_layer(self, location: str, layer_name: str) -> gpd.geodataframe:
        """Inherited, see superclass."""
        location = location.replace('.gpkg', '')
        rel_path = self._get_gpkg_file_path(location)
        path_on_disk = os.path.join(self._map_root, rel_path)
        if not os.path.exists(path_on_disk):
            layer_lock_file = f'{self._map_lock_dir}/{location}_{layer_name}.lock'
            self._safe_save_layer(layer_lock_file, rel_path)
        self._wait_for_expected_filesize(path_on_disk, location)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            map_meta = gpd.read_file(path_on_disk, layer='meta', engine='pyogrio')
            projection_system = map_meta[map_meta['key'] == 'projectedCoordSystem']['value'].iloc[0]
            gdf_in_pixel_coords = pyogrio.read_dataframe(path_on_disk, layer=layer_name, fid_as_index=True)
            gdf_in_utm_coords = gdf_in_pixel_coords.to_crs(projection_system)
            gdf_in_utm_coords.index = gdf_in_utm_coords.index.map(str)
            gdf_in_utm_coords['fid'] = gdf_in_utm_coords.index
        return gdf_in_utm_coords

    def vector_layer_names(self, location: str) -> Sequence[str]:
        """Inherited, see superclass."""
        location = location.replace('.gpkg', '')
        rel_path = self._get_gpkg_file_path(location)
        path_on_disk = os.path.join(self._map_root, rel_path)
        self._blob_store.save_to_disk(rel_path)
        return pyogrio.list_layers(path_on_disk)

    def purge_cache(self) -> None:
        """Inherited, see superclass."""
        logger.debug('Purging cache...')
        for f in glob.glob(os.path.join(self._map_root, 'gpkg', '*')):
            os.remove(f)
        logger.debug('Done purging cache.')

    def _get_map_dataset(self, location: str) -> rasterio.DatasetReader:
        """
        Returns a *context manager* for the map dataset (includes all the layers).
        Extract the result in a "with ... as ...:" line.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        :return: A *context manager* for the map dataset (includes all the layers).
        """
        rel_path = self._get_gpkg_file_path(location)
        path_on_disk = os.path.join(self._map_root, rel_path)
        self._blob_store.save_to_disk(rel_path)
        return rasterio.open(path_on_disk)

    def get_layer_dataset(self, location: str, layer_name: str) -> rasterio.DatasetReader:
        """
        Returns a *context manager* for the layer dataset.
        Extract the result in a "with ... as ...:" line.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        :return: A *context manager* for the layer dataset.
        """
        with self._get_map_dataset(location) as map_dataset:
            layer_dataset_path = next((path for path in map_dataset.subdatasets if path.endswith(':' + layer_name)), None)
            if layer_dataset_path is None:
                raise ValueError(f"Layer '{layer_name}' not found in map '{location}', version '{self.get_version(location)}'")
            return rasterio.open(layer_dataset_path)

    def get_raster_layer_names(self, location: str) -> Sequence[str]:
        """
        Gets the list of available layers for a given map location.
        :param location: The layers name for this map location will be returned.
        :return: List of available raster layers.
        """
        all_layers_dataset = self._get_map_dataset(location)
        fully_qualified_layer_names = all_layers_dataset.subdatasets
        return [name.split(':')[-1] for name in fully_qualified_layer_names]

    def get_gpkg_path_and_store_on_disk(self, location: str) -> str:
        """
        Saves a gpkg map from a location to disk.
        :param location: The layers name for this map location will be returned.
        :return: Path on disk to save a gpkg file.
        """
        rel_path = self._get_gpkg_file_path(location)
        path_on_disk = os.path.join(self._map_root, rel_path)
        self._blob_store.save_to_disk(rel_path)
        return path_on_disk

    def get_metadata_json_path_and_store_on_disk(self, location: str) -> str:
        """
        Saves a metadata.json for a location to disk.
        :param location: The layers name for this map location will be returned.
        :return: Path on disk to save metadata.json.
        """
        rel_path = self._get_metadata_json_path(location)
        path_on_disk = os.path.join(self._map_root, rel_path)
        self._blob_store.save_to_disk(rel_path)
        return path_on_disk

    def _get_gpkg_file_path(self, location: str) -> str:
        """
        Gets path to the gpkg map file.
        :param location: Location for which gpkg needs to be loaded.
        :return: Path to the gpkg file.
        """
        version = self.get_version(location)
        return f'{location}/{version}/map.gpkg'

    def _get_metadata_json_path(self, location: str) -> str:
        """
        Gets path to the metadata json file.
        :param location: Location for which json needs to be loaded.
        :return: Path to the meta json file.
        """
        version = self.get_version(location)
        return f'{location}/{version}/metadata.json'

    def _get_layer_matrix_npy_path(self, location: str, layer_name: str) -> str:
        """
        Gets path to the numpy file for the layer.
        :param location: Location for which layer needs to be loaded.
        :param layer_name: Which layer to load.
        :return: Path to the numpy file.
        """
        version = self.get_version(location)
        return f'{location}/{version}/{layer_name}.npy.npz'

    @staticmethod
    def _get_np_array(path_on_disk: str) -> np.ndarray:
        """
        Gets numpy array from file.
        :param path_on_disk: Path to numpy file.
        :return: Numpy array containing the layer.
        """
        np_data = np.load(path_on_disk)
        return np_data['data']

    def _get_expected_file_size(self, path: str, shape: List[int]) -> int:
        """
        Gets the expected file size.
        :param path: Path to the file.
        :param shape: The shape of the map file.
        :return: The expected file size.
        """
        if path.endswith('_dist.npy'):
            return shape[0] * shape[1] * 4
        return shape[0] * shape[1]

    def _get_layer_matrix(self, location: str, layer_name: str) -> npt.NDArray[np.uint8]:
        """
        Returns the map layer for `location` and `layer_name` as a numpy array.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        :return: Numpy representation of layer.
        """
        rel_path = self._get_layer_matrix_npy_path(location, layer_name)
        path_on_disk = os.path.join(self._map_root, rel_path)
        if not os.path.exists(path_on_disk):
            self._save_layer_matrix(location=location, layer_name=layer_name)
        return self._get_np_array(path_on_disk)

    def _save_layer_matrix(self, location: str, layer_name: str) -> None:
        """
        Extracts the data for `layer_name` from the GPKG file for `location`,
        and saves it on disk so it can be retrieved with `_get_layer_matrix`.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        """
        is_bin = self.is_binary(layer_name)
        with self.get_layer_dataset(location, layer_name) as layer_dataset:
            layer_data = layer_dataset_ops.load_layer_as_numpy(layer_dataset, is_bin)
        if '_distance_px' in layer_name:
            transform_matrix = self._get_transform_matrix(location, layer_name)
            precision = 1 / transform_matrix[0, 0]
            layer_data = np.negative(layer_data / precision).astype('float32')
        npy_file_path = os.path.join(self._map_root, f'{location}/{self.get_version(location)}/{layer_name}.npy')
        np.savez_compressed(npy_file_path, data=layer_data)

    def _save_all_layers(self, location: str) -> None:
        """
        Saves data on disk for all layers in the GPKG file for `location`.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        """
        rasterio_layers = self.get_raster_layer_names(location)
        for layer_name in rasterio_layers:
            logger.debug('Working on layer: ', layer_name)
            self._save_layer_matrix(location, layer_name)

def make_dilatable_map_layer(mask: npt.NDArray[np.uint8], precision: float) -> MapLayer:
    """
    Convenience method for constructing a dilatable map with the appropriate pre-computed distances.
    :param mask: Pixel values.
    :param precision: Meters per pixel.
    :return: A MapLayer Object.
    """
    joint_distance = compute_joint_distance_matrix(mask, precision)
    layer = MapLayer(mask, make_meta(True, precision), joint_distance=joint_distance)
    return layer

def make_meta(can_dilate: bool, precision: float, is_binary: bool=True) -> MapLayerMeta:
    """
    Helper method to initialize a MapLayerMeta instance.
    :param can_dilate: whether to can dilate or not.
    :param precision: Meters per pixel.
    :param is_binary: Flag to indicate if is binary.
    :return: A MapLayerMeta object.
    """
    return MapLayerMeta(name='test_fixture', md5_hash='not used here', can_dilate=can_dilate, is_binary=is_binary, precision=precision)

class TestMask(unittest.TestCase):
    """Test Mask."""
    data = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])

    def test_negative_dilation(self) -> None:
        """Checks if it raises with negative dilation values."""
        layer = make_dilatable_map_layer(TestMask.data, ONE_PIXEL_PER_METER)
        for negative_number in [-0.01, -1, -200]:
            with self.subTest(negative_number=negative_number):
                self.assertRaises(AssertionError, layer.mask, dilation=negative_number)

    def test_dilate_undilatable_layer(self) -> None:
        """Checks if it raises with dilating on unlidatable layer."""
        meta = make_meta(can_dilate=False, precision=ONE_PIXEL_PER_METER)
        layer = MapLayer(TestMask.data, meta)
        self.assertRaises(AssertionError, layer.mask, dilation=1)

    def test_no_dilation(self) -> None:
        """Tests layer with no dilation."""
        layer = make_dilatable_map_layer(TestMask.data, ONE_PIXEL_PER_METER)
        self.assertTrue(np.array_equal(layer.mask(dilation=0), TestMask.data))

    def test_dilation(self) -> None:
        """Tests dilation with different dilation values."""
        layer = make_dilatable_map_layer(TestMask.data, ONE_PIXEL_PER_METER)
        test_cases = [(0.1, np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])), (0.5, np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]])), (1, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]])), (2, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))]
        for dilation, expected_mask in test_cases:
            with self.subTest(dilation=dilation, expected_mask=expected_mask):
                self.assertTrue(np.array_equal(layer.mask(dilation=dilation), expected_mask))

class TestCrop(unittest.TestCase):
    """Test class for Cropping layer."""
    data = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])

    def test_empty_slice(self) -> None:
        """Checks empty slice, and size of layer after cropping should be zero."""
        layer = make_dilatable_map_layer(TestCrop.data, ONE_PIXEL_PER_METER)
        crop = layer.crop(slice(0), slice(0))
        self.assertEqual(crop.shape, (0, 0))

    def test_full_slices(self) -> None:
        """Tests with full slice and various out-of-bounds slices."""
        layer = make_dilatable_map_layer(TestCrop.data, ONE_PIXEL_PER_METER)
        test_cases = [(slice(0, 3), slice(0, 3)), (slice(0, 3), slice(0, 5)), (slice(0, 5), slice(0, 3)), (slice(0, 5), slice(0, 5))]
        for row_slice, col_slice in test_cases:
            with self.subTest(row_slice=row_slice, col_slice=col_slice):
                crop = layer.crop(row_slice, col_slice)
                self.assertTrue(np.array_equal(crop, TestCrop.data))

    def test_negative_dilation(self) -> None:
        """Tests to dilate with negative dilation value."""
        layer = make_dilatable_map_layer(TestCrop.data, ONE_PIXEL_PER_METER)
        for negative_number in [-0.01, -1, -200]:
            with self.subTest(negative_number=negative_number):
                self.assertRaises(AssertionError, layer.crop, rows=slice(0, 2), cols=slice(0, 2), dilation=negative_number)

    def test_dilate_undilatable_layer(self) -> None:
        """Tests to dilate an undilatable layer in crop function."""
        meta = make_meta(can_dilate=False, precision=ONE_PIXEL_PER_METER)
        layer = MapLayer(TestCrop.data, meta)
        self.assertRaises(AssertionError, layer.crop, rows=slice(0, 2), cols=slice(0, 2), dilation=1)

    def test_no_dilation(self) -> None:
        """Test no dilation with crop function."""
        layer = make_dilatable_map_layer(TestCrop.data, ONE_PIXEL_PER_METER)
        upper_left_crop = layer.crop(rows=slice(0, 2), cols=slice(0, 2), dilation=0)
        self.assertTrue(np.array_equal(upper_left_crop, np.array([[1, 1], [1, 0]])))
        lower_right_crop = layer.crop(rows=slice(1, 3), cols=slice(1, 3), dilation=0)
        self.assertTrue(np.array_equal(lower_right_crop, np.array([[0, 0], [0, 0]])))

    def test_dilation(self) -> None:
        """Tests dilation in crop function."""
        layer = make_dilatable_map_layer(TestCrop.data, ONE_PIXEL_PER_METER)
        test_cases = [(0.1, np.array([[1, 0, 0], [0, 0, 0]])), (0.5, np.array([[1, 1, 0], [1, 0, 0]])), (1, np.array([[1, 1, 1], [1, 1, 0]])), (2, np.array([[1, 1, 1], [1, 1, 1]]))]
        for dilation, expected_lower_crop in test_cases:
            with self.subTest(dilation=dilation, expected_lower_crop=expected_lower_crop):
                crop = layer.crop(rows=slice(1, 3), cols=slice(0, 3), dilation=dilation)
                self.assertTrue(np.array_equal(crop, expected_lower_crop))

class TestTransformMatrix(unittest.TestCase):
    """Test Class for Transform matrix."""

    def test_transform_matrix(self) -> None:
        """Tests transform matrix for MapLayers with different precisions and different size."""
        test_cases = [(101, 101, 1, np.array([[1, 0, 0, 0], [0, -1, 0, 100], [0, 0, 1, 0], [0, 0, 0, 1]])), (101, 101, 0.1, np.array([[10, 0, 0, 0], [0, -10, 0, 100], [0, 0, 1, 0], [0, 0, 0, 1]])), (51, 51, 1, np.array([[1, 0, 0, 0], [0, -1, 0, 50], [0, 0, 1, 0], [0, 0, 0, 1]])), (51, 51, 10, np.array([[0.1, 0, 0, 0], [0, -0.1, 0, 50], [0, 0, 1, 0], [0, 0, 0, 1]]))]
        for nrows, ncols, precision, expected_matrix in test_cases:
            with self.subTest(nrows=nrows, ncols=ncols, precision=precision, expected_matrix=expected_matrix):
                layer = MapLayer(np.ones((nrows, ncols)), make_meta(False, precision))
                self.assertTrue(np.array_equal(layer.transform_matrix, expected_matrix))

class TestToPixelCoords(unittest.TestCase):
    """Test Class of converting to pixel coordinates."""

    def test_without_precision_scale(self) -> None:
        """Tests to_pixel_coords function on a map layer of 1 pixel per meter."""
        layer = MapLayer(DUMMY_DATA_101_BY_101, make_meta(False, ONE_PIXEL_PER_METER))
        test_cases = [[(0, 0), (0, 100)], [(0, 100), (0, 0)], [(100, 0), (100, 100)], [(100, 100), (100, 0)], [(0, 40), (0, 60)], [(40, 0), (40, 100)], [(40, 100), (40, 0)], [(100, 40), (100, 60)], [(50, 50), (50, 50)], [(36, 42), (36, 58)], [(99, 37), (99, 63)], [(7, 99), (7, 1)]]
        for input_point, expected_output in test_cases:
            with self.subTest(input_point=input_point, expected_output=expected_output):
                self.assertEqual(layer.to_pixel_coords(*input_point), expected_output)

    def test_with_precision_scale(self) -> None:
        """Test to_pixel_coords function on a map layer of 0.1 precision."""
        layer = MapLayer(DUMMY_DATA_101_BY_101, make_meta(False, TEN_PIXELS_PER_METER))
        test_cases = [[(0, 0), (0, 100)], [(0, 10), (0, 0)], [(10, 0), (100, 100)], [(10, 10), (100, 0)], [(0, 4), (0, 60)], [(4, 0), (40, 100)], [(4, 10), (40, 0)], [(10, 4), (100, 60)], [(5, 5), (50, 50)], [(3.6, 4.2), (36, 58)], [(9.9, 3.7), (99, 63)], [(0.7, 9.9), (7, 1)]]
        for input_point, expected_output in test_cases:
            with self.subTest(input_point=input_point, expected_output=expected_output):
                self.assertEqual(layer.to_pixel_coords(*input_point), expected_output)

    def test_multiple_inputs(self) -> None:
        """Test with multiple inputs."""
        layer = MapLayer(DUMMY_DATA_101_BY_101, make_meta(False, ONE_PIXEL_PER_METER))
        input_x = np.array([5, 60])
        input_y = np.array([20, 77])
        expected_x = np.array([5, 60])
        expected_y = np.array([80, 23])
        output_x, output_y = layer.to_pixel_coords(input_x, input_y)
        self.assertTrue(np.array_equal(output_x, expected_x))
        self.assertTrue(np.array_equal(output_y, expected_y))

class TestIsOnMask(unittest.TestCase):
    """Test Class of is_on_mask function."""
    small_number = 1e-05
    half_gt = TEN_PIXELS_PER_METER / 2 + small_number
    half_lt = TEN_PIXELS_PER_METER / 2 - small_number

    def test_out_of_bounds_without_dilation(self) -> None:
        """This checks the boundary conditions for is_on_mask."""
        mask = np.ones((10, 10))
        layer = MapLayer(mask, make_meta(False, TEN_PIXELS_PER_METER))
        x, y = OutOfBoundsData.in_bounds_for_10_by_10_layer_with_10px_per_m
        self.assertTrue(np.all(layer.is_on_mask(x, y)))
        x, y = OutOfBoundsData.out_of_bounds_for_10_by_10_layer_with_10px_per_m
        self.assertFalse(np.any(layer.is_on_mask(x, y)))

    def test_out_of_bounds_with_dilation(self) -> None:
        """This checks the boundary conditions for is_on_mask."""
        mask = np.ones((10, 10))
        layer = make_dilatable_map_layer(mask, TEN_PIXELS_PER_METER)
        x, y = OutOfBoundsData.in_bounds_for_10_by_10_layer_with_10px_per_m
        self.assertTrue(np.all(layer.is_on_mask(x, y, dilation=0.3)))
        x, y = OutOfBoundsData.out_of_bounds_for_10_by_10_layer_with_10px_per_m
        self.assertFalse(np.any(layer.is_on_mask(x, y, dilation=0.3)))

    def test_native_resolution(self) -> None:
        """Test map resolution."""
        mask = np.zeros((51, 40))
        mask[30, 20] = 1
        layer = MapLayer(mask, make_meta(False, TEN_PIXELS_PER_METER))
        self.assertTrue(layer.is_on_mask(2, 2))
        on_mask = [(2 + self.half_lt, 2), (2 - self.half_lt, 2), (2, 2 + self.half_lt), (2, 2 - self.half_lt)]
        off_mask = [(2 + self.half_gt, 2), (2 - self.half_gt, 2), (2, 2 + self.half_gt), (2, 2 - self.half_gt)]
        for point in on_mask:
            with self.subTest(point=point):
                self.assertTrue(layer.is_on_mask(*point)[0])
        for point in off_mask:
            with self.subTest(point=point):
                self.assertFalse(layer.is_on_mask(*point)[0])

    def test_edges(self) -> None:
        """Test map edges."""
        mask = np.ones((51, 40))
        layer = MapLayer(mask, make_meta(False, TEN_PIXELS_PER_METER))
        self.assertTrue(layer.is_on_mask(0, 0.1))
        self.assertTrue(layer.is_on_mask(0, 5))
        self.assertTrue(layer.is_on_mask(3.9, 0.1))
        self.assertTrue(layer.is_on_mask(3.9, 5))
        self.assertFalse(layer.is_on_mask(3.9 + self.half_gt, 0.1))
        self.assertFalse(layer.is_on_mask(3.9 + self.half_gt, 5))
        self.assertFalse(layer.is_on_mask(0 - self.half_gt, 0.1))
        self.assertFalse(layer.is_on_mask(0 - self.half_gt, 5))

class TestDistToMask(unittest.TestCase):
    """This Class to test dist_to_mask function."""

    def test_out_of_bounds(self) -> None:
        """This checks the boundary conditions for dist_to_mask."""
        mask = np.ones((10, 10))
        layer = make_dilatable_map_layer(mask, TEN_PIXELS_PER_METER)
        x, y = OutOfBoundsData.in_bounds_for_10_by_10_layer_with_10px_per_m
        self.assertFalse(np.any(np.isnan(layer.dist_to_mask(x, y))))
        x, y = OutOfBoundsData.out_of_bounds_for_10_by_10_layer_with_10px_per_m
        self.assertTrue(np.all(np.isnan(layer.dist_to_mask(x, y))))

    def test_linear_edge_low_precision(self) -> None:
        """Tests linear edges with low precision of 1."""
        mask = np.array([[0, 0, 1, 1]])
        layer = make_dilatable_map_layer(mask, ONE_PIXEL_PER_METER)
        test_cases = [(0, np.array([[1.5, 0.5, -0.5, -1.5]])), (1, np.array([[0.5, -0.5, -1.5, -2.5]])), (2, np.array([[-0.5, -1.5, -2.5, -3.5]]))]
        for dilation, expected_dist_to_mask in test_cases:
            for x in range(0, 4):
                with self.subTest(dilation=dilation, expected_dist_to_mask=expected_dist_to_mask, x=x):
                    actual = layer.dist_to_mask(x, 0, dilation)
                    expected = expected_dist_to_mask[0, x]
                    self.assertTrue(abs(actual - expected) < 0.001)

    def test_linear_edge_high_precision(self) -> None:
        """Test linear edge with high precision of 0.1."""
        mask = np.array([[0, 0, 1, 1]])
        layer = make_dilatable_map_layer(mask, TEN_PIXELS_PER_METER)
        test_cases = [(0, np.array([[0.15, 0.05, -0.05, -0.15]])), (0.1, np.array([[0.05, -0.05, -0.15, -0.25]])), (0.2, np.array([[-0.05, -0.15, -0.25, -0.35]]))]
        x_in_meters = np.array([0, 0.1, 0.2, 0.3])
        y_in_meters = np.array([0, 0, 0, 0])
        for dilation, expected_dist_to_mask in test_cases:
            actual = layer.dist_to_mask(x_in_meters, y_in_meters, dilation)
            with self.subTest(dilation=dilation, expected_dist_to_mask=expected_dist_to_mask, actual=actual):
                self.assertTrue(np.allclose(actual, expected_dist_to_mask))

    def _test_non_linear_edge_helper(self, mask: npt.NDArray[np.uint8], expected_matrix: npt.NDArray[np.float64], test_name: str) -> None:
        """
        Helper function to test nonlinear edge cases.
        :param mask: Pixel values.
        :param expected_matrix: The expected distance matrix of points on mask.
        :param test_name: A string of test name.
        """
        layer = make_dilatable_map_layer(mask, TEN_PIXELS_PER_METER)
        for dilation in [0, 0.1, 0.2, 0.3]:
            for x_in_pixels in range(0, 5):
                for y_in_pixels in range(0, 5):
                    x_in_meters = x_in_pixels * 0.1
                    y_in_meters = y_in_pixels * 0.1
                    matrix_row = mask.shape[1] - 1 - y_in_pixels
                    matrix_col = x_in_pixels
                    actual = layer.dist_to_mask(x_in_meters, y_in_meters, dilation=dilation)
                    expected = expected_matrix[matrix_row, matrix_col] - dilation
                    with self.subTest(x_in_meters=x_in_meters, y_in_meters=y_in_meters, actual=actual, expected=expected, test_name=test_name):
                        self.assertTrue(abs(actual - expected) < 0.005)

    def test_round_edge(self) -> None:
        """Test case of round edge."""
        mask = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        expected_matrix = np.array([[0.266, 0.23, 0.15, 0.05, -0.05], [0.174, 0.15, 0.09, 0.05, -0.05], [0.09, 0.05, 0.05, -0.05, -0.09], [0.05, -0.05, -0.05, -0.09, -0.174], [-0.05, -0.09, -0.15, -0.174, -0.23]])
        self._test_non_linear_edge_helper(mask, expected_matrix, 'test_round_edge')

    def test_hole(self) -> None:
        """Test case of a hole mask."""
        mask = np.array([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]])
        expected_matrix = np.array([[-0.09, -0.05, -0.05, -0.05, -0.09], [-0.05, 0.05, 0.05, 0.05, -0.05], [-0.05, 0.05, 0.15, 0.09, 0.05], [-0.05, 0.05, 0.05, 0.05, -0.05], [-0.09, -0.05, -0.05, -0.05, -0.09]])
        self._test_non_linear_edge_helper(mask, expected_matrix, 'test_hole')

class TestConsistentIsOnMaskDistToMask(unittest.TestCase):
    """This Class to test the consistency of is_on_mask and dist_to_mask function."""

    def assert_on_mask_equals(self, layer: MapLayer, x: Any, y: Any, dilation: float, expected: bool) -> None:
        """
        Asserts when point is on mask, return True, otherwise False, and
        dist_to_mask gives a negative value, otherwise positive.
        :param layer: A MapLayer Object.
        :param x: Global x coordinates. Can be a scalar, list or a numpy array of x coordinates.
        :param y: Global y coordinates. Can be a scalar, list or a numpy array of x coordinates.
        :param dilation: Specifies the threshold on the distance from the drivable_area mask.
        The drivable_area mask is dilated to include points which are within this distance from itself.
        :param expected: The expected boolean value.
        """
        on_mask_result = layer.is_on_mask(x, y, dilation=dilation)
        self.assertEqual(on_mask_result, expected, f'expected is_on_mask({x}, {y}, {dilation}) to be {expected}, got {on_mask_result}')
        dist_to_mask_result = layer.dist_to_mask(x, y, dilation=dilation)
        self.assertEqual(dist_to_mask_result <= 0, expected, f'expected dist_to_mask({x}, {y}, {dilation}), to be {('<= 0' if expected else '> 0')}, got {dist_to_mask_result}')

    def test_dilation_with_foreground_point(self) -> None:
        """Test dilation with foreground point."""
        mask = np.zeros((51, 40))
        mask[30, 20] = 1
        layer = make_dilatable_map_layer(mask, TEN_PIXELS_PER_METER)
        self.assertTrue(layer.is_on_mask(2, 2))
        self.assertFalse(layer.is_on_mask(2, 3))
        self.assert_on_mask_equals(layer, 2, 3, dilation=1, expected=True)
        self.assert_on_mask_equals(layer, 3, 2, dilation=1, expected=True)
        self.assert_on_mask_equals(layer, 2 + np.sqrt(1 / 2), 2 + np.sqrt(1 / 2), dilation=1, expected=True)
        self.assert_on_mask_equals(layer, 2, 3, dilation=0.9, expected=False)

    def test_dilation_with_curved_line(self) -> None:
        """Test Dilation over Curved Line."""
        mask = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        layer = make_dilatable_map_layer(mask, ONE_PIXEL_PER_METER)
        off_mask_points_in_physical_space = [(0, 4), (1, 4)]
        for x in range(0, 5):
            for y in range(0, 5):
                with self.subTest(x=x, y=y):
                    expected_on_mask = (x, y) not in off_mask_points_in_physical_space
                    self.assert_on_mask_equals(layer, x, y, dilation=2, expected=expected_on_mask)

    def test_coarse_resolution(self) -> None:
        """Tests with normal and low resolution."""
        mask = np.zeros((51, 40))
        mask[30, 20] = 1
        mask[31, 20] = 1
        mask[30, 21] = 1
        mask[31, 21] = 1
        normal_res_layer = make_dilatable_map_layer(mask, TEN_PIXELS_PER_METER)
        low_res_layer = make_dilatable_map_layer(mask, FIVE_PIXELS_PER_METER)
        self.assert_on_mask_equals(normal_res_layer, 2, 2, dilation=0, expected=True)
        self.assert_on_mask_equals(low_res_layer, 2, 4, dilation=0, expected=False)
        self.assert_on_mask_equals(low_res_layer, 2, 4, dilation=2.0, expected=True)
        self.assert_on_mask_equals(low_res_layer, 2, 4, dilation=1.9001, expected=True)
        self.assert_on_mask_equals(low_res_layer, 2, 4, dilation=1.8, expected=False)

