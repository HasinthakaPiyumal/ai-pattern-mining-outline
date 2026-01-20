# Cluster 48

class S3Store(BlobStore):
    """
    S3 blob store. Load blobs from AWS S3.
    """

    def __init__(self, s3_prefix: str, profile_name: Optional[str]=None, show_progress: bool=True) -> None:
        """
        Initialize S3Store.
        :param s3_prefix: S3 path
        :param profile_name: Profile name.
        :param show_progress: Whether to show download progress.
        """
        assert s3_prefix.startswith('s3://')
        self._s3_prefix = s3_prefix
        if not self._s3_prefix.endswith('/'):
            self._s3_prefix += '/'
        self._profile_name = profile_name
        url = parse.urlparse(self._s3_prefix)
        self._bucket = url.netloc
        self._prefix = url.path.lstrip('/')
        self._client = get_s3_client(self._profile_name)
        self._show_progress = show_progress

    def __reduce__(self) -> Tuple[Type[S3Store], Tuple[Any, ...]]:
        """
        :return: tuple of class and its constructor parameters, this is used to pickle the class.
        """
        return (self.__class__, (self._s3_prefix, self._profile_name))

    def _get_s3_location(self, key: str) -> Tuple[str, str, str]:
        """
        Get s3 location information.
        :param key: Full S3 path or bucket key of blob.
        :return: Full S3 path, bucket and key.
        """
        s3_path = key if key.startswith('s3://') else f's3://{self._bucket}/{self._prefix}{key}'
        url = parse.urlparse(s3_path)
        bucket = url.netloc
        parsed_key = url.path.lstrip('/')
        return (s3_path, bucket, parsed_key)

    def get(self, key: str, check_for_compressed: bool=False) -> BinaryIO:
        """
        Get blob content.
        :param key: Full S3 path or bucket key of blob.
        :param check_for_compressed: Flag that check for a "<key>+.gzip" file and extracts the <key> file.
        :return: A file-like object, use read() to get raw bytes.
        """
        path, _, _ = self._get_s3_location(key)
        gzip_path = path + '.gzip'
        if check_for_compressed and self.exists(gzip_path):
            gzip_stream = self._get(key=gzip_path)
            content: BinaryIO = self._extract_gzip_content(gzip_stream)
        else:
            content = self._get(key=key)
        return content

    def _get(self, key: str, num_tries: int=7) -> BinaryIO:
        """
        Get blob content from path/key.

        Note: Occasionally S3 give a ConnectionResetError or http.client.IncompleteRead
              exception. urllib3 wraps both of these in a ProtocolError. Sometimes S3 also
              gives an "ssl.SSLError: [SSL: WRONG_VERSION_NUMBER]" error. Unfortunately the
              boto3 retrying ("max_attempts") gives up when it sees any of these exceptions,
              and we have to handle retrying them ourselves. Starting with version 1.26.0,
              urllib3 wraps the ssl.SSLError into a urllib3.exceptions.SSLError.

        Note: Pytorch uses an ExceptionWrapper class that tries to "reconstruct" its wrapped
              exception, but if a new exception gets thrown *while calling the constructor* of
              the wrapped exception's type, then that new exception is raised instead of an
              instance of the wrapped exception's type. Long story short, this means some
              retryable AWS exceptions get turned into KeyErrors, so we have to catch KeyError too.

        :param key: Full S3 path or bucket key of blob.
        :param num_tries: Number of download tries.
        :return: Blob binary stream.
        """
        s3_path, bucket, key = self._get_s3_location(key)
        disable_progress = not self._show_progress
        for try_number in range(0, num_tries):
            try:
                total_length = int(self._client.head_object(Bucket=bucket, Key=key).get('ContentLength', 0))
                bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                with tqdm(total=total_length, desc=f'Downloading {s3_path}...', bar_format=bar_format, unit='B', unit_scale=True, unit_divisor=1024, disable=disable_progress) as pbar:
                    stream: BinaryIO = io.BytesIO()
                    self._client.download_fileobj(bucket, key, stream, Callback=pbar.update)
                    stream.seek(0)
                break
            except (urllib3.exceptions.ProtocolError, ssl.SSLError, urllib3.exceptions.SSLError, KeyError, BotoCoreError, NoCredentialsError) as e:
                if isinstance(e, KeyError):
                    logger.warning(f'Caught KeyError: {e}. Retrying S3 read.')
                was_last_try = try_number == num_tries - 1
                if was_last_try:
                    raise e
                else:
                    logger.debug(f'Retrying S3 fetch due to exception {e}')
                    time.sleep(2 ** try_number)
            except botocore.exceptions.ClientError as error:
                if error.response['Error']['Code'] == 'NoSuchKey':
                    message = f'{str(error)}\nS3 path not found: {s3_path}'
                    raise BlobStoreKeyNotFound(message)
                else:
                    raise RuntimeError(f'{error} Key: {s3_path}')
        return stream

    async def get_async(self, key: str) -> BinaryIO:
        """Inherited, see superclass."""
        raise NotImplementedError('Not today.')

    def save_to_disk(self, key: str, check_for_compressed: bool=False) -> None:
        """Inherited, see superclass."""
        super().save_to_disk(key, check_for_compressed=check_for_compressed)

    def exists(self, key: str) -> bool:
        """
        Tell if the blob exists.
        :param key: blob path or token.
        :return: True if the blob exists else False.
        """
        _, bucket, key = self._get_s3_location(key)
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response['ResponseMetadata']['HTTPStatusCode'] == 404:
                return False
            raise
        except BotoCoreError as e:
            logger.debug(e)
            return False

    def put(self, key: str, value: BinaryIO, ignore_if_client_error: bool=False) -> bool:
        """
        Writes content to the blobstore.
        :param key: Blob path or token.
        :param value: Data to save.
        :param ignore_if_client_error: Set to true if we want to ignore botocore client error
        """
        _, bucket, key = self._get_s3_location(key)
        successfully_stored_object = False
        try:
            response = self._client.put_object(Body=value, Bucket=bucket, Key=key)
            successfully_stored_object = response is not None
            if not successfully_stored_object:
                raise RuntimeError(f'Failed to store object to blobstore. Key : {key}')
        except botocore.exceptions.ClientError as error:
            logger.info(f'{error}')
            if not ignore_if_client_error:
                raise RuntimeError(f'{error} Key: {key}')
        return successfully_stored_object

def get_s3_client(profile_name: Optional[str]=None, max_attempts: int=10, aws_access_key_id: Optional[str]=None, aws_secret_access_key: Optional[str]=None) -> boto3.client:
    """
    Start a Boto3 session and retrieve the client.
    :param profile_name: S3 profile name to use when creating the session.
    :param aws_access_key_id: Aws access key id.
    :param aws_secret_access_key: Aws secret access key.
    :param max_attempts: Maximum number of attempts in loading the client.
    :return: The instantiated client object.
    """
    session = _get_sync_session(profile_name, aws_access_key_id, aws_secret_access_key)
    config = Config(retries={'max_attempts': max_attempts})
    client = session.client('s3', config=config)
    return client

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

class FileBackedBarrier:
    """
    A file-based synchronization barrier.
    This class can be used to synchronize activies across multiple machines.
    """

    def __init__(self, barrier_directory: Path) -> None:
        """
        Initializes a FileBackedBarrier.
        :param barrier_directory: The path that the barrier files will use for synchronization.
          This can be a local or S3 path.
        """
        self._barrier_directory = barrier_directory
        self._is_s3 = str(barrier_directory).startswith('s3:')
        self._activity_file_content = 'x'

    def wait_barrier(self, activity_id: str, expected_activity_ids: Set[str], timeout_s: Optional[float]=None, poll_interval_s: float=1) -> None:
        """
        Registers that `activity_id` has completed.
        Waits until all activities in `expected_activity_ids` have completed.
        If timeout_s has been provided, the operation will raise a TimeoutError after
          the supplied number of seconds has passed.

        :param activity_id: The activity ID that will be registered as completed.
        :param expected_activity_ids: The list of activity IDs that are expected to be completed.
          The function will block until these are done.
        :param timeout_s: If provided, the timeout for the wait operation.
          If the operation does not complete within this amount of time, then a TimeoutError will be raised.
        :param poll_interval_s: The elapsed time before polling for new files.
        """
        logger.info('Writing completion of activity id %s to directory %s...', activity_id, self._barrier_directory)
        self._register_activity_id_complete(activity_id)
        logger.info('Waiting for all processes to finish processing')
        self._wait(expected_activity_ids, timeout_s, poll_interval_s)
        logger.info(f'Sleeping for {poll_interval_s * SLEEP_MULTIPLIER_BEFORE_CLEANUP} seconds so that the other processes catch up before moving on')
        time.sleep(poll_interval_s * SLEEP_MULTIPLIER_BEFORE_CLEANUP)
        logger.info('All Processes Synced, clearing activity file')
        self._remove_activity_after_processing(activity_id)
        logger.info('Waiting for all processes to clean up barrier files')
        self._wait(set(), timeout_s, poll_interval_s)

    def _wait(self, expected_activity_ids: Set[str], timeout_s: Optional[float]=None, poll_interval_s: float=1) -> None:
        start_wait_time = time.time()
        logger.info('Beginning barrier wait at time %f', start_wait_time)
        while True:
            next_wait_time = time.time() + poll_interval_s
            logger.debug('The next wait time is %f. Getting completed activity ids...', next_wait_time)
            completed_activity_ids = self._get_completed_activity_ids()
            logger.debug('There are %d completed activities.', len(completed_activity_ids))
            if expected_activity_ids == completed_activity_ids:
                logger.debug('All activities completed! Ending wait.')
                return
            total_wait_time = time.time() - start_wait_time
            logger.debug('All tasks not finished. Total elapsed wait time is %f.', total_wait_time)
            if timeout_s is not None and total_wait_time > timeout_s:
                raise TimeoutError(f'Waited {total_wait_time} sec for barrier {self._barrier_directory}, which is longer than configured timeout of {timeout_s}.')
            sleep_time = max(0.0, next_wait_time - time.time())
            logger.debug('Sleeping for %f seconds.', sleep_time)
            time.sleep(sleep_time)

    def _register_activity_id_complete(self, activity_id: str) -> None:
        """
        Registers an activity_id as completed by creating a file in the configured directory.
        :param activity_id: The activity ID to register as completed.
        """
        activity_id_file_path = self._barrier_directory / activity_id
        if self._is_s3:
            s3_bucket, s3_key = self._split_s3_path(activity_id_file_path)
            self._create_activity_file_in_s3(s3_key, s3_bucket)
        else:
            activity_id_file_path.parent.mkdir(exist_ok=True, parents=True)
            with open(activity_id_file_path, 'w') as f:
                f.write(self._activity_file_content)

    def _get_completed_activity_ids(self) -> Set[str]:
        """
        Gets the activity IDs from the filesystem that have been marked as completed.
        :return: The completed file system activity ids.
        """
        if self._is_s3:
            s3_bucket, s3_key = self._split_s3_path(self._barrier_directory)
            files = [Path(p) for p in self._list_files_in_s3_directory(s3_key, s3_bucket)]
        else:
            files = [x for x in self._barrier_directory.iterdir() if x.is_file()]
        unique_activity_ids = {f.stem for f in files}
        return unique_activity_ids

    def _remove_activity_after_processing(self, activity_id: str) -> None:
        """
        Removes the activity file so that we can reuse the same directory in future calls to sync
        """
        activity_id_file_path = self._barrier_directory / activity_id
        if self._is_s3:
            s3_bucket, s3_key = self._split_s3_path(activity_id_file_path)
            self._remove_activity_file_from_s3(s3_key, s3_bucket)
        else:
            activity_id_file_path.unlink()

    @retry(RETRYABLE_EXCEPTIONS, backoff=1, tries=3, delay=0.5)
    def _create_activity_file_in_s3(self, s3_key: Path, s3_bucket: str) -> None:
        """
        Creates an activity file in S3
        :param s3_key: The S3 path for the file, without the bucket.
        :param s3_bucket: The name of the bucket to write to.
        """
        with closing(get_s3_client()) as s3_client:
            logger.info(f'Creating activity file at {s3_key} in bucket {s3_bucket}...')
            s3_client.put_object(Body=self._activity_file_content.encode('utf-8'), Bucket=s3_bucket, Key=str(s3_key))

    @retry(RETRYABLE_EXCEPTIONS, backoff=1, tries=3, delay=0.5)
    def _remove_activity_file_from_s3(self, s3_key: Path, s3_bucket: str) -> None:
        """
        Creates an activity file in S3
        :param s3_key: The S3 path for the file, without the bucket.
        :param s3_bucket: The name of the bucket to write to.
        """
        with closing(get_s3_client()) as s3_client:
            logger.info(f'Removing activity file at {s3_key} in bucket {s3_bucket}...')
            s3_client.delete_object(Bucket=s3_bucket, Key=str(s3_key))

    @retry(RETRYABLE_EXCEPTIONS, backoff=1, tries=3, delay=0.5)
    def _list_files_in_s3_directory(self, s3_key: Path, s3_bucket: str) -> List[Path]:
        """
        Lists the files available in a particular S3 directory.
        :param s3_key: The path to list, without the bucket.
        :param s3_bucket: The bucket to list.
        :return: The files in the folder.
        """
        with closing(get_s3_client()) as s3_client:
            key = str(s3_key)
            if not key.endswith('/'):
                key += '/'
            objects = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=key)
            if 'Contents' in objects:
                return [Path(k['Key']) for k in objects['Contents']]
            return []

    def _split_s3_path(self, s3_path: Path) -> Tuple[str, Path]:
        """
        Splits a S3 path into a (bucket, path) set of identifiers.
        :param s3_path: The full S3 path.
        :return: A tuple of (bucket, path).
        """
        chunks = [v.strip() for v in str(s3_path).split('/') if len(v.strip()) > 0]
        bucket = chunks[1]
        path = Path('/'.join(chunks[2:]))
        return (bucket, path)

class PublisherCallback(AbstractMainCallback):
    """Callback publishing data to S3"""

    def __init__(self, uploads: Dict[str, Any], s3_client: Optional[boto3.client], s3_bucket: str, remote_prefix: Optional[List[str]]):
        """
        Construct publisher callback, responsible to publish results of simulation, image validation and result aggregation
        :param uploads: dict containing information on which directories to publish
        """
        self._s3_client = s3_client
        if self._s3_client is None:
            self._s3_client = get_s3_client()
        self._s3_bucket = s3_bucket.strip('s3://') if s3_bucket.startswith('s3://') else s3_bucket
        self._remote_prefix: List[str] = remote_prefix or ['/']
        self._upload_targets: List[UploadConfig] = []
        for name, upload_data in uploads.items():
            if upload_data['upload']:
                save_path = pathlib.Path(upload_data['save_path'])
                remote_path = pathlib.Path(upload_data.get('remote_path') or '')
                self._upload_targets.append(UploadConfig(name=name, local_path=save_path, remote_path=pathlib.Path(*self._remote_prefix) / remote_path))

    def on_run_simulation_end(self) -> None:
        """
        On reached_end push results to S3 bucket.
        """
        logger.info('Publishing results on S3...')
        for upload_target in self._upload_targets:
            paths = list_files(upload_target.local_path)
            for path in paths:
                key = str(upload_target.remote_path / path)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'Pushing to S3 bucket: {self._s3_bucket}\n\t file: {str(upload_target.local_path.joinpath(path))}\n\t on destination: {key}')
                local_target = upload_target.local_path
                if not local_target.is_file():
                    local_target = local_target.joinpath(path)
                self._s3_client.upload_file(str(local_target), self._s3_bucket, key)
        logger.info('Publishing results on S3... DONE')

class TestPublisherCallback(TestCase):
    """
    Tests PublisherCallback.
    """

    def setUp(self) -> None:
        """Setup mocks for the tests"""
        fake_targets = {'metrics': {'upload': True, 'save_path': 'some/path/to/save', 'remote_path': 'path/save'}, 'pictures': {'upload': True, 'save_path': 'some/path/to/pictures', 'remote_path': 'path/pictures'}}
        self.fake_uploads = [UploadConfig('metrics', pathlib.Path('some/path/to/save'), pathlib.Path('user/image/path/save')), UploadConfig('pictures', pathlib.Path('some/path/to/pictures'), pathlib.Path('user/image/path/pictures'))]
        self.mock_client = Mock()
        self.publisher_callback = PublisherCallback(fake_targets, self.mock_client, 'bucket', ['user', 'image'])

    def test_publisher_callback_init(self) -> None:
        """
        Tests if all the properties are set to the expected values in constructor.
        """
        self.assertEqual(self.fake_uploads, self.publisher_callback._upload_targets)

    @patch('nuplan.planning.simulation.main_callback.publisher_callback.pathlib')
    @patch('nuplan.planning.simulation.main_callback.publisher_callback.list_files')
    def test_on_run_simulation_end_push_to_s3(self, mock_files: Mock, mock_pathlib: Mock) -> None:
        """
        Tests if the callback is called with the correct parameters.
        """
        fake_path = Mock()
        fake_path.iterdir.return_value = [True]
        fake_path.__truediv__ = lambda name, x: f'bucket/{x}'
        mock_pathlib.Path.return_value = fake_path
        mock_files.return_value = ['a', 'b']
        self.publisher_callback.on_run_simulation_end()
        expected_calls = [call('some/path/to/save/a', 'bucket', 'user/image/path/save/a'), call('some/path/to/save/b', 'bucket', 'user/image/path/save/b'), call('some/path/to/pictures/a', 'bucket', 'user/image/path/pictures/a'), call('some/path/to/pictures/b', 'bucket', 'user/image/path/pictures/b')]
        self.mock_client.upload_file.assert_has_calls(expected_calls)

    @patch('nuplan.planning.simulation.main_callback.publisher_callback.pathlib', MagicMock())
    @patch('nuplan.planning.simulation.main_callback.publisher_callback.boto3')
    def test_no_push_without_results(self, mock_boto3: Mock) -> None:
        """
        Tests if the callback is called with the correct parameters.
        """
        empty_publisher_callback = PublisherCallback({}, self.mock_client, 'bucket', ['user', 'image'])
        empty_publisher_callback.on_run_simulation_end()
        mock_boto3.client.return_value.assert_not_called()

def get_s3_file_contents(s3_path: str, client: Optional[boto3.client]=None, delimiter: str='/', include_previous_folder: bool=False) -> S3FileResultMessage:
    """
    Get folders and files contents in the provided s3 path provided.
    :param s3_path: S3 path dir to expand.
    :param client: Boto3 client to use, if None create a new one.
    :param delimiter: Delimiter for path.
    :param include_previous_folder: Set True to include '..' as previous folder.
    :return: Dict of file contents.
    """
    return_message = 'Connect successfully'
    file_contents: Dict[str, S3FileContent] = {}
    try:
        client = get_s3_client() if client is None else client
        if not s3_path.endswith('/'):
            s3_path = s3_path + '/'
        url = parse.urlparse(s3_path)
        paginator = client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=url.netloc, Prefix=url.path.lstrip('/'), Delimiter=delimiter)
        previous_folder = os.path.join(url.path.lstrip('/'), '..')
        if previous_folder != '..' and include_previous_folder:
            file_contents[previous_folder] = S3FileContent(filename=previous_folder)
        for page in page_iterator:
            for obj in page.get('CommonPrefixes', []):
                file_contents[obj['Prefix']] = S3FileContent(filename=obj['Prefix'])
            for content in page.get('Contents', []):
                file_name = str(content['Key'])
                if file_name == url.path.lstrip('/'):
                    continue
                file_contents[file_name] = S3FileContent(filename=file_name, last_modified=content['LastModified'], size=content['Size'])
        success = True
    except Exception as err:
        logger.info('Error: {}'.format(err))
        return_message = f'{err}'
        success = False
    s3_connection_status = S3ConnectionStatus(return_message=return_message, success=success)
    s3_file_result_message = S3FileResultMessage(s3_connection_status=s3_connection_status, file_contents=file_contents)
    return s3_file_result_message

class TestNuBoardCloudUtil(unittest.TestCase):
    """Unit tests for cloud utils in nuboard."""

    def setUp(self) -> None:
        """Set up a list of nuboard files."""
        self.tmp_dir = tempfile.TemporaryDirectory()

    def test_check_s3_nuboard_files_fail(self) -> None:
        """Test if check_s3_nuboard_files fails when there is no nuboard file."""
        s3_client = boto3.Session().client('s3')
        stubber = Stubber(s3_client)
        dummy_file_result_message = {'dummy_a': S3FileContent(filename='dummy_a', size=10, last_modified=datetime(day=2, month=7, year=1992, tzinfo=timezone.utc)), 'dummy_b': S3FileContent(filename='dummy_b', size=10, last_modified=datetime(day=3, month=8, year=1992, tzinfo=timezone.utc))}
        encoded_expected_messages = {'dummy_a': json.dumps(dummy_file_result_message['dummy_a'].serialize()).encode(), 'dummy_b': json.dumps(dummy_file_result_message['dummy_b'].serialize()).encode()}
        dummy_streaming_io_message_response = {}
        expected_params = {}
        for s3_key, result_message in dummy_file_result_message.items():
            dummy_streaming_io_message_response[s3_key] = {'Body': io.BytesIO(encoded_expected_messages[s3_key])}
            expected_params[s3_key] = {'Bucket': 'test-bucket', 'Key': s3_key}
        for s3_key, expected_param in expected_params.items():
            response = dummy_streaming_io_message_response[s3_key]
            stubber.add_response('get_object', response, expected_param)
        with stubber:
            s3_nuboard_file_result_message = check_s3_nuboard_files(s3_file_contents=dummy_file_result_message, s3_client=s3_client, s3_path='s3://test-bucket')
        self.assertIsNone(s3_nuboard_file_result_message.nuboard_file)
        self.assertFalse(s3_nuboard_file_result_message.s3_connection_status.success)

    def test_check_s3_nuboard_files_success(self) -> None:
        """Test if check_s3_nuboard_files success when there is a nuboard file."""
        s3_client = boto3.Session().client('s3')
        stubber = Stubber(s3_client)
        nuboard_file = NuBoardFile(simulation_main_path=self.tmp_dir.name, metric_folder='metrics', simulation_folder='simulations', metric_main_path=self.tmp_dir.name, aggregator_metric_folder='aggregator_metric')
        nuboard_file_name = 'dummy_a' + NuBoardFile.extension()
        dummy_file_result_message = {nuboard_file_name: S3FileContent(filename=nuboard_file_name, size=12, last_modified=datetime(day=4, month=5, year=1992, tzinfo=timezone.utc))}
        encoded_expected_messages = {nuboard_file_name: pickle.dumps(nuboard_file.serialize())}
        dummy_streaming_io_message_response = {}
        expected_params = {}
        for s3_key, result_message in dummy_file_result_message.items():
            dummy_streaming_io_message_response[s3_key] = {'Body': io.BytesIO(encoded_expected_messages[s3_key])}
            expected_params[s3_key] = {'Bucket': 'test-bucket', 'Key': s3_key}
        for s3_key, expected_param in expected_params.items():
            response = dummy_streaming_io_message_response[s3_key]
            stubber.add_response('get_object', response, expected_param)
        with stubber:
            s3_nuboard_file_result_message = check_s3_nuboard_files(s3_file_contents=dummy_file_result_message, s3_client=s3_client, s3_path='s3://test-bucket')
        self.assertTrue(s3_nuboard_file_result_message.s3_connection_status.success)
        self.assertIsNotNone(s3_nuboard_file_result_message.nuboard_file)
        self.assertEqual(nuboard_file.simulation_main_path, s3_nuboard_file_result_message.nuboard_file.simulation_main_path)
        self.assertEqual(nuboard_file.metric_main_path, s3_nuboard_file_result_message.nuboard_file.metric_main_path)

    def test_get_s3_file_content(self) -> None:
        """Test if download_s3_file works."""
        s3_client = boto3.Session().client('s3')
        stubber = Stubber(s3_client)
        expected_response = {'CommonPrefixes': [{'Prefix': 'dummy_folder_a/log.txt'}, {'Prefix': 'dummy_folder_b/log_2.txt'}], 'Contents': [{'Key': 'dummy_a', 'Size': 15, 'LastModified': datetime(day=2, month=7, year=1992, tzinfo=timezone.utc)}, {'Key': 'dummy_b', 'Size': 45, 'LastModified': datetime(day=6, month=7, year=1992, tzinfo=timezone.utc)}]}
        expected_params = {'Bucket': 'test-bucket', 'Prefix': '', 'Delimiter': '/'}
        s3_path = 's3://test-bucket'
        stubber.add_response('list_objects_v2', expected_response, expected_params)
        with stubber:
            s3_file_contents = get_s3_file_contents(s3_path=s3_path, client=s3_client, include_previous_folder=True)
            self.assertTrue(s3_file_contents.s3_connection_status.success)
            expected_file_names = ['dummy_folder_a/log.txt', 'dummy_folder_b/log_2.txt', 'dummy_a', 'dummy_b']
            for index, (file_name, _) in enumerate(s3_file_contents.file_contents.items()):
                self.assertEqual(file_name, expected_file_names[index])

    def test_s3_download_file(self) -> None:
        """Test s3_download_file in utils."""
        s3_client = boto3.Session().client('s3')
        dummy_s3_file_content = S3FileContent(filename='dummy_a', size=10, last_modified=datetime(day=2, month=7, year=1992, tzinfo=timezone.utc))
        s3_path = 's3://test-bucket'
        save_path = self.tmp_dir.name
        with self.assertRaises(Boto3Error):
            download_s3_file(s3_path=s3_path, s3_client=s3_client, save_path=save_path, file_content=dummy_s3_file_content)

    def test_s3_download_path(self) -> None:
        """Test s3_download_path in utils."""
        s3_client = boto3.Session().client('s3')
        stubber = Stubber(s3_client)
        expected_response = {'CommonPrefixes': [{'Prefix': 'dummy_folder_a/log.txt'}, {'Prefix': 'dummy_folder_b/log_2.txt'}], 'Contents': [{'Key': 'dummy_a', 'Size': 15, 'LastModified': datetime(day=2, month=7, year=1992, tzinfo=timezone.utc)}, {'Key': 'dummy_b', 'Size': 45, 'LastModified': datetime(day=6, month=7, year=1992, tzinfo=timezone.utc)}]}
        expected_params = {'Bucket': 'test-bucket', 'Prefix': '', 'Delimiter': '/'}
        s3_path = 's3://test-bucket'
        stubber.add_response('list_objects_v2', expected_response, expected_params)
        save_path = self.tmp_dir.name
        with stubber:
            with self.assertRaises(Boto3Error):
                download_s3_path(s3_path=s3_path, s3_client=s3_client, save_path=save_path)

    def tearDown(self) -> None:
        """Remove and clean up the tmp folder."""
        self.tmp_dir.cleanup()

class CloudTab:
    """Cloud tab in nuboard."""

    def __init__(self, doc: Document, configuration_tab: ConfigurationTab, s3_bucket: Optional[str]=''):
        """
        Cloud tab for remote connection features.
        :param doc: Bokeh HTML document.
        :param configuration_tab: Configuration tab.
        :param s3_bucket: Aws s3 bucket name.
        """
        self._doc = doc
        self._configuration_tab = configuration_tab
        self._nuplan_exp_root = os.getenv('NUPLAN_EXP_ROOT', None)
        assert self._nuplan_exp_root is not None, 'Please set environment variable: NUPLAN_EXP_ROOT!'
        download_path = Path(self._nuplan_exp_root)
        download_path.mkdir(parents=True, exist_ok=True)
        self._default_datasource_dict = dict(object=['-'], last_modified=['-'], timestamp=['-'], size=['-'])
        self._s3_content_datasource = ColumnDataSource(data=self._default_datasource_dict)
        self._selected_column = TextInput()
        self._selected_row = TextInput()
        self.s3_bucket_name = Div(**S3TabBucketNameConfig.get_config())
        self.s3_bucket_name.js_on_change('text', S3TabDataTableUpdateJSCode.get_js_code())
        self.s3_error_text = Div(**S3TabErrorTextConfig.get_config())
        self.s3_download_text_input = TextInput(**S3TabDownloadTextInputConfig.get_config())
        self.s3_download_button = Button(**S3TabDownloadButtonConfig.get_config())
        self.s3_download_button.on_click(self._s3_download_button_on_click)
        self.s3_download_button.js_on_click(S3TabLoadingJSCode.get_js_code())
        self.s3_download_button.js_on_change('disabled', S3TabDownloadUpdateJSCode.get_js_code())
        self.s3_bucket_text_input = TextInput(**S3TabBucketTextInputConfig.get_config(), value=s3_bucket)
        self.s3_access_key_id_text_input = TextInput(**S3TabS3AccessKeyIDTextInputConfig.get_config())
        self.s3_secret_access_key_password_input = PasswordInput(**S3TabS3SecretAccessKeyPasswordTextInputConfig.get_config())
        self.s3_bucket_prefix_text_input = TextInput(**S3TabS3BucketPrefixTextInputConfig.get_config())
        self.s3_modal_query_btn = Button(**S3TabS3ModalQueryButtonConfig.get_config())
        self.s3_modal_query_btn.on_click(self._s3_modal_query_on_click)
        self.s3_modal_query_btn.js_on_click(S3TabLoadingJSCode.get_js_code())
        self._default_columns = [TableColumn(**S3TabObjectColumnConfig.get_config()), TableColumn(**S3TabLastModifiedColumnConfig.get_config()), TableColumn(**S3TabTimeStampColumnConfig.get_config()), TableColumn(**S3TabSizeColumnConfig.get_config())]
        self._s3_content_datasource = ColumnDataSource(data=self._default_datasource_dict)
        self._s3_content_datasource.js_on_change('data', S3TabDataTableUpdateJSCode.get_js_code())
        self._s3_content_datasource.selected.js_on_change('indices', S3TabContentDataSourceOnSelected.get_js_code(selected_column=self._selected_column, selected_row=self._selected_row))
        self._s3_content_datasource.selected.js_on_change('indices', S3TabContentDataSourceOnSelectedLoadingJSCode.get_js_code(source=self._s3_content_datasource, selected_column=self._selected_column))
        self._s3_content_datasource.selected.on_change('indices', self._s3_data_source_on_selected)
        self.data_table = DataTable(source=self._s3_content_datasource, columns=self._default_columns, **S3TabDataTableConfig.get_config())
        self._s3_client: Optional[boto3.client] = None
        if s3_bucket:
            self._update_blob_store(s3_bucket=s3_bucket, s3_prefix='')

    def _update_blob_store(self, s3_bucket: str, s3_prefix: str='') -> None:
        """
        :param s3_bucket:
        :param s3_prefix:
        """
        aws_profile_name = bytes(self.s3_access_key_id_text_input.value + self.s3_secret_access_key_password_input.value, encoding='utf-8')
        hash_md5 = hashlib.md5(aws_profile_name)
        profile = hash_md5.hexdigest()
        self._s3_client = get_s3_client(aws_access_key_id=self.s3_access_key_id_text_input.value, aws_secret_access_key=self.s3_secret_access_key_password_input.value, profile_name=profile)
        s3_path = os.path.join(s3_bucket, s3_prefix)
        s3_file_result_message = get_s3_file_contents(s3_path=s3_path, include_previous_folder=True, client=self._s3_client)
        self._load_s3_contents(s3_file_result_message=s3_file_result_message)
        self.s3_error_text.text = s3_file_result_message.s3_connection_status.return_message
        if s3_file_result_message.s3_connection_status.success:
            self.s3_bucket_name.text = s3_bucket

    def _s3_modal_query_on_click(self) -> None:
        """On click function for modal query button."""
        self._update_blob_store(s3_bucket=self.s3_bucket_text_input.value, s3_prefix=self.s3_bucket_prefix_text_input.value)

    def _s3_data_source_on_selected(self, attr: str, old: List[int], new: List[int]) -> None:
        """Helper function when select a row in data source."""
        if not new:
            return
        row_index = new[0]
        self._s3_content_datasource.selected.update(indices=[])
        column_index = int(self._selected_column.value)
        s3_prefix = self.data_table.source.data['object'][row_index]
        if column_index == 0:
            if not s3_prefix or s3_prefix == '-':
                return
            if '..' in s3_prefix:
                s3_prefix = Path(s3_prefix).parents[1].name
            self._update_blob_store(s3_bucket=self.s3_bucket_text_input.value, s3_prefix=s3_prefix)
        else:
            if '..' in s3_prefix or '-' == s3_prefix:
                return
            self.s3_download_text_input.value = s3_prefix

    def _update_data_table_source(self, data_sources: Dict[str, List[Any]]) -> None:
        """Update data table source."""
        self.data_table.source.data = data_sources

    def _load_s3_contents(self, s3_file_result_message: S3FileResultMessage) -> None:
        """
        Load s3 contents into a data table.
        :param s3_file_result_message: File content and return messages from s3 connection.
        """
        file_contents = s3_file_result_message.file_contents
        if not s3_file_result_message.s3_connection_status.success or len(s3_file_result_message.file_contents) <= 1:
            default_data_sources = self._default_datasource_dict
            self._doc.add_next_tick_callback(partial(self._update_data_table_source, data_sources=default_data_sources))
        else:
            data_sources: Dict[str, List[Any]] = {'object': [], 'last_modified': [], 'timestamp': [], 'size': []}
            for file_name, content in file_contents.items():
                data_sources['object'].append(file_name)
                data_sources['last_modified'].append(content.last_modified_day if content.last_modified is not None else '')
                data_sources['timestamp'].append(content.date_string if content.date_string is not None else '')
                data_sources['size'].append(content.kb_size() if content.kb_size() is not None else '')
            self._doc.add_next_tick_callback(partial(self._update_data_table_source, data_sources=data_sources))

    def _reset_s3_download_button(self) -> None:
        """Reset s3 download button."""
        self.s3_download_button.label = 'Download'
        self.s3_download_button.disabled = False
        self.s3_download_text_input.disabled = False

    def _update_error_text_label(self, text: str) -> None:
        """Update error text message in a sequential manner."""
        self.s3_error_text.text = text

    def _s3_download_prefixes(self) -> None:
        """Download s3 prefixes and update progress in a sequential manner."""
        try:
            start_time = time.perf_counter()
            if not self._s3_client:
                raise Boto3Error('No s3 connection!')
            selected_s3_bucket = str(self.s3_bucket_name.text).strip()
            selected_s3_prefix = str(self.s3_download_text_input.value).strip()
            selected_s3_path = os.path.join(selected_s3_bucket, selected_s3_prefix)
            s3_result_file_contents = get_s3_file_contents(s3_path=selected_s3_path, client=self._s3_client, include_previous_folder=False)
            s3_nuboard_file_result = check_s3_nuboard_files(s3_result_file_contents.file_contents, s3_client=self._s3_client, s3_path=selected_s3_path)
            if not s3_nuboard_file_result.s3_connection_status.success:
                raise Boto3Error(s3_nuboard_file_result.s3_connection_status.return_message)
            if not s3_result_file_contents.file_contents:
                raise Boto3Error(f'No objects exist in the path: {selected_s3_path}')
            self._download_s3_file_contents(s3_result_file_contents=s3_result_file_contents, selected_s3_bucket=selected_s3_bucket)
            self._update_s3_nuboard_file_main_path(s3_nuboard_file_result=s3_nuboard_file_result, selected_prefix=selected_s3_prefix)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            successful_message = f'Downloaded to {self._nuplan_exp_root} and took {elapsed_time:.4f} seconds'
            logger.info('Downloaded to {} and took {:.4f} seconds'.format(self._nuplan_exp_root, elapsed_time))
            self._doc.add_next_tick_callback(partial(self._update_error_text_label, text=successful_message))
        except Exception as e:
            logger.info(str(e))
            self.s3_error_text.text = str(e)
        self._doc.add_next_tick_callback(self._reset_s3_download_button)

    def _update_s3_nuboard_file_main_path(self, s3_nuboard_file_result: S3NuBoardFileResultMessage, selected_prefix: str) -> None:
        """
        Update nuboard file simulation and metric main path.
        :param s3_nuboard_file_result: S3 nuboard file result.
        :param selected_prefix: Selected prefix on s3.
        """
        nuboard_file = s3_nuboard_file_result.nuboard_file
        nuboard_filename = s3_nuboard_file_result.nuboard_filename
        if not nuboard_file or not nuboard_filename or (not self._nuplan_exp_root):
            return
        main_path = Path(self._nuplan_exp_root) / selected_prefix
        nuboard_file.simulation_main_path = str(main_path)
        nuboard_file.metric_main_path = str(main_path)
        metric_path = main_path / nuboard_file.metric_folder
        if not metric_path.exists():
            metric_path.mkdir(parents=True, exist_ok=True)
        simulation_path = main_path / nuboard_file.simulation_folder
        if not simulation_path.exists():
            simulation_path.mkdir(parents=True, exist_ok=True)
        aggregator_metric_path = main_path / nuboard_file.aggregator_metric_folder
        if not aggregator_metric_path.exists():
            aggregator_metric_path.mkdir(parents=True, exist_ok=True)
        save_path = main_path / nuboard_filename
        nuboard_file.save_nuboard_file(save_path)
        logger.info('Updated nubBard main path in {} to {}'.format(save_path, main_path))
        self._configuration_tab.add_nuboard_file_to_experiments(nuboard_file=s3_nuboard_file_result.nuboard_file)

    def _download_s3_file_contents(self, s3_result_file_contents: S3FileResultMessage, selected_s3_bucket: str) -> None:
        """
        Download s3 file contents.
        :param s3_result_file_contents: S3 file result contents.
        :param selected_s3_bucket: Selected s3 bucket name.
        """
        for index, (file_name, content) in enumerate(s3_result_file_contents.file_contents.items()):
            if '..' in file_name:
                continue
            s3_path = os.path.join(selected_s3_bucket, file_name)
            if not file_name.endswith('/'):
                s3_connection_message = download_s3_file(s3_path=s3_path, s3_client=self._s3_client, file_content=content, save_path=self._nuplan_exp_root)
            else:
                s3_connection_message = download_s3_path(s3_path=s3_path, s3_client=self._s3_client, save_path=self._nuplan_exp_root)
            if s3_connection_message.success:
                text_message = f'Downloaded {file_name} ({index + 1} / {len(s3_result_file_contents.file_contents)})'
                logger.info('Downloaded {} / ({}/{})'.format(file_name, index + 1, len(s3_result_file_contents.file_contents)))
                self._doc.add_next_tick_callback(partial(self._update_error_text_label, text=text_message))

    def _s3_download_button_on_click(self) -> None:
        """Function to call when the download button is click."""
        selected_s3_bucket = str(self.s3_bucket_name.text).strip()
        self.s3_download_button.label = 'Downloading...'
        self.s3_download_button.disabled = True
        self.s3_download_text_input.disabled = True
        if not selected_s3_bucket:
            self.s3_error_text.text = 'Please connect to a s3 bucket'
            self._doc.add_next_tick_callback(self._reset_s3_download_button)
            return
        selected_s3_prefix = str(self.s3_download_text_input.value).strip()
        if not selected_s3_prefix:
            self.s3_error_text.text = 'Please input a prefix'
            self._doc.add_next_tick_callback(self._reset_s3_download_button)
            return
        self._doc.add_next_tick_callback(self._s3_download_prefixes)

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Downloads evaluation results from S3, runs metric aggregator and re-uploads the results.
    :param cfg: Hydra config dict.
    """
    local_output_dir = Path(cfg.output_dir, cfg.contestant_id, cfg.submission_id)
    cfg.challenges = CHALLENGES
    Path(cfg.output_dir).mkdir(exist_ok=True, parents=True)
    s3_download(prefix='/'.join([cfg.contestant_id, cfg.submission_id]), local_path_name=cfg.output_dir, filters=None)
    simulation_successful = is_submission_successful(cfg.challenges, local_output_dir)
    cfg.output_dir = str(local_output_dir)
    cfg.scenario_metric_paths = list_subdirs_filtered(local_output_dir, re.compile(f'/{cfg.metric_folder_name}$'))
    logger.info('Found metric paths %s' % cfg.scenario_metric_paths)
    aggregated_metric_save_path = local_output_dir / cfg.aggregated_metric_folder_name
    leaderboard_writer = LeaderBoardWriter(cfg, str(local_output_dir))
    simulation_results = {}
    summary_results = {}
    try:
        if simulation_successful:
            shutil.rmtree(str(aggregated_metric_save_path), ignore_errors=True)
            aggregated_metric_save_path.mkdir(parents=True, exist_ok=True)
            aggregator_main(cfg)
            simulation_results['aggregated-metrics'] = {'upload': True, 'save_path': str(aggregated_metric_save_path), 'remote_path': 'aggregated_metrics'}
            simulation_results['metrics'] = {'upload': True, 'save_path': str(local_output_dir / cfg.metric_folder_name), 'remote_path': 'metrics'}
            summary_results['summary'] = {'upload': True, 'save_path': str(local_output_dir / 'summary'), 'remote_path': 'summary'}
    except Exception as e:
        submission_logger.error('Aggregation failed!')
        submission_logger.error(e)
        simulation_successful = False
    finally:
        simulation_results['submission_logs'] = {'upload': True, 'save_path': '/tmp/submission.log', 'remote_path': 'aggregated_metrics'}
        result_remote_prefix = [str(cfg.contestant_id), str(cfg.submission_id)]
        result_s3_client = get_s3_client()
        result_publisher_callback = PublisherCallback(simulation_results, remote_prefix=result_remote_prefix, s3_client=result_s3_client, s3_bucket=os.getenv('NUPLAN_SERVER_S3_ROOT_URL'))
        result_publisher_callback.on_run_simulation_end()
        summary_publisher_callback = PublisherCallback(summary_results, remote_prefix=['public/leaderboard/planning/2022', cfg.submission_id], s3_client=result_s3_client, s3_bucket=os.getenv('NUPLAN_SERVER_S3_ROOT_URL'))
        summary_publisher_callback.on_run_simulation_end()
    leaderboard_writer.write_to_leaderboard(simulation_successful=simulation_successful)
    shutil.rmtree(local_output_dir)

def s3_download(prefix: str, local_path_name: str, filters: Optional[List[str]]=None) -> None:
    """
    Downloads all files matching a pattern on s3 creating a client
    :param prefix: The pattern matching prefix
    :param local_path_name: The local destination
    :param filters: Keywords to filter paths, if empty no filtering is performed.
    """
    args = {'region_name': 'us-east-1'}
    if os.getenv('AWS_WEB_IDENTITY_TOKEN_FILE') is None and os.getenv('AWS_CONTAINER_CREDENTIALS_RELATIVE_URI') is None:
        args['aws_access_key_id'] = os.environ['NUPLAN_SERVER_AWS_ACCESS_KEY_ID']
        args['aws_secret_access_key'] = os.environ['NUPLAN_SERVER_AWS_SECRET_ACCESS_KEY']
    s3_client = boto3.client('s3', **args)
    s3_bucket = os.getenv('NUPLAN_SERVER_S3_ROOT_URL')
    assert s3_bucket, 'S3 bucket not specified!'
    s3_download_dir(s3_bucket, s3_client, prefix, local_path_name, filters)

def is_submission_successful(challenges: List[str], simulation_results_dir: Path) -> bool:
    """
    Checks if evaluation of one submission was successful, by checking that all instances for all challenges
    were completed.
    :param challenges: The list of challenges.
    :param simulation_results_dir: Path were the simulation results are saved locally.
    :return: True if the submission was evaluated successfully, False otherwise.
    """
    completed = list(simulation_results_dir.rglob('*completed.txt'))
    successful = True if len(completed) == len(challenges) * NUM_INSTANCES_PER_CHALLENGE else False
    logger.info('Found %s completed simulations' % len(completed))
    logger.info('Simulation was successful:  %s' % successful)
    return successful

def list_subdirs_filtered(root_dir: Path, regex_pattern: re.Pattern[str]) -> List[str]:
    """
    Lists the path of files present in a directory. Results are filtered by ending pattern.
    :param root_dir: The path to start the search.
    :param regex_pattern: Regex based Pattern for which paths to keep.
    :return: List of paths under root_dir which wnd with path_end_filter.
    """
    paths = [str(path) for path in root_dir.rglob('**/*') if regex_pattern.search(str(path))]
    return paths

class TestRunResultProcessor(unittest.TestCase):
    """Test ResultProcessor script."""

    def test_is_submission_successful(self) -> None:
        """Tests that is_submission_successful utility function is working as expected."""
        challenge_names = ['challenge_1', 'challenge_2']
        temp_dirs = []
        temp_files = []
        with TemporaryDirectory() as tmpdir:
            for _ in challenge_names:
                temp_files.append(NamedTemporaryFile(dir=tmpdir))
                sub_tmpdir = TemporaryDirectory(dir=tmpdir)
                temp_dirs.append(sub_tmpdir)
                for instance in range(NUM_INSTANCES_PER_CHALLENGE):
                    temp_files.append(NamedTemporaryFile(dir=sub_tmpdir.name, suffix='_completed.txt'))
                    temp_files.append(NamedTemporaryFile(dir=sub_tmpdir.name, suffix='_completed_not.txt'))
            self.assertTrue(is_submission_successful(challenge_names, Path(tmpdir)))
            extra_completed_at_root = NamedTemporaryFile(dir=tmpdir, suffix='_completed.txt')
            self.assertFalse(is_submission_successful(challenge_names, Path(tmpdir)))
            extra_completed_at_root.close()
            extra_completed_at_challenge_dirs = []
            for sub_tmpdir in temp_dirs:
                extra_completed_at_challenge_dirs.append(NamedTemporaryFile(dir=sub_tmpdir.name, suffix='_completed.txt'))
            self.assertFalse(is_submission_successful(challenge_names, Path(tmpdir)))
            for item in extra_completed_at_challenge_dirs:
                item.close()
            self.assertTrue(is_submission_successful(challenge_names, Path(tmpdir)))
            for item in temp_files:
                item.close()

    def test_list_subdirs_filtered(self) -> None:
        """Tests listing of filtered files in subdirectories."""
        expected_found = []
        temporary_files = []
        with TemporaryDirectory() as tmpdir:
            with TemporaryDirectory(dir=tmpdir) as sub_tmpdir:
                file1 = NamedTemporaryFile(dir=sub_tmpdir, suffix='.yes')
                file2 = NamedTemporaryFile(dir=tmpdir, suffix='.yes')
                rubbish1 = NamedTemporaryFile(dir=tmpdir, suffix='.no')
                rubbish2 = NamedTemporaryFile(dir=sub_tmpdir, suffix='.no')
                temporary_files.extend([file1, file2, rubbish1, rubbish2])
                expected_found.extend([file1.name, file2.name])
                paths = list_subdirs_filtered(Path(tmpdir), regex_pattern=re.compile('\\.yes'))
                self.assertEqual(set(expected_found), set(paths))
                for temp_file in temporary_files:
                    temp_file.close()

class TestLeaderboardWriter(unittest.TestCase):
    """Tests for the LeaderboardWriter class."""

    @patch(f'{TEST_FILE}.EvalaiInterface')
    def setUp(self, mock_interface: Mock) -> None:
        """Sets up variables for testing."""
        self.mock_interface = mock_interface
        main_path = os.path.dirname(os.path.realpath(__file__))
        common_dir = 'file://' + os.path.join(main_path, '../../../planning/script/config/common')
        self.search_path = f'hydra.searchpath=[{common_dir}]'
        with initialize_config_dir(config_dir=CONFIG_PATH):
            cfg = compose(config_name=CONFIG_NAME, overrides=[self.search_path, 'contestant_id=contestant', 'submission_id=submission'])
            self.tmpdir = tempfile.TemporaryDirectory()
            self.addCleanup(self.tmpdir.cleanup)
            metadata = {'challenge_phase': 'phase', 'submission_id': 'my_sub'}
            with open(f'{self.tmpdir.name}/submission_metadata.json', 'w') as fp:
                json.dump(metadata, fp)
            self.leaderboard_writer = LeaderBoardWriter(cfg, self.tmpdir.name)

    def test_write_to_leaderboard(self) -> None:
        """Tests that writing to leaderboard calls the correct callbacks an api."""
        with patch.object(self.leaderboard_writer, '_on_successful_submission'):
            self.leaderboard_writer.write_to_leaderboard(simulation_successful=True)
            self.leaderboard_writer._on_successful_submission.assert_called_once()
            self.leaderboard_writer.interface.update_submission_data.assert_called_once_with(self.leaderboard_writer._on_successful_submission.return_value)
        self.mock_interface.reset_mock()
        with patch.object(self.leaderboard_writer, '_on_failed_submission'):
            self.leaderboard_writer.write_to_leaderboard(simulation_successful=False)
            self.leaderboard_writer._on_failed_submission.assert_called_once()
            self.leaderboard_writer.interface.update_submission_data.assert_called_once_with(self.leaderboard_writer._on_failed_submission.return_value)

    def test__on_failed_submission(self) -> None:
        """Tests message creation on failes submission callback."""
        expected_data = {'challenge_phase': 'phase', 'submission': 'my_sub', 'stdout': '', 'stderr': '', 'submission_status': 'FAILED', 'metadata': ''}
        data = self.leaderboard_writer._on_failed_submission()
        self.assertEqual(expected_data, data)

    def test__on_successful_submission(self) -> None:
        """Tests message creation on successful submission callback."""
        expected_data = {'challenge_phase': 'phase', 'submission': 'my_sub', 'stdout': '', 'stderr': '', 'result': '[{"split": "data_split", "show_to_participant": true, "accuracies": "results"}]', 'submission_status': 'FINISHED', 'metadata': {'status': 'finished'}}
        with patch(f'{TEST_FILE}.read_metrics_from_results') as reader:
            reader.return_value = 'results'
            data = self.leaderboard_writer._on_successful_submission()
            self.assertEqual(expected_data, data)

    def test_read_metrics_from_results(self) -> None:
        """Tests parsing of dataframes."""
        dataframes = {'open_loop_boxes': pd.DataFrame.from_dict({'scenario': 'final_score', 'score': [0], 'planner_expert_average_l2_error_within_bound': [1], 'planner_expert_final_l2_error_within_bound': [2], 'planner_miss_rate_within_bound': [3], 'planner_expert_average_heading_error_within_bound': [4], 'planner_expert_final_heading_error_within_bound': [5]}), 'closed_loop_nonreactive_agents': pd.DataFrame.from_dict({'scenario': 'final_score', 'score': [10], 'ego_is_making_progress': [11], 'no_ego_at_fault_collisions': [12], 'drivable_area_compliance': [13], 'driving_direction_compliance': [14], 'ego_is_comfortable': [15], 'ego_progress_along_expert_route': [16], 'time_to_collision_within_bound': [17], 'speed_limit_compliance': [18]}), 'closed_loop_reactive_agents': pd.DataFrame.from_dict({'scenario': 'final_score', 'score': [110], 'ego_is_making_progress': [111], 'no_ego_at_fault_collisions': [112], 'drivable_area_compliance': [113], 'driving_direction_compliance': [114], 'ego_is_comfortable': [115], 'ego_progress_along_expert_route': [116], 'time_to_collision_within_bound': [117], 'speed_limit_compliance': [118]})}
        metrics = read_metrics_from_results(dataframes)
        expected_metrics = {'ch1_overall_score': 0, 'ch1_avg_displacement_error_within_bound': 1, 'ch1_final_displacement_error_within_bound': 2, 'ch1_miss_rate_within_bound': 3, 'ch1_avg_heading_error_within_bound': 4, 'ch1_final_heading_error_within_bound': 5, 'ch2_overall_score': 10, 'ch2_ego_is_making_progress': 11, 'ch2_no_ego_at_fault_collisions': 12, 'ch2_drivable_area_compliance': 13, 'ch2_driving_direction_compliance': 14, 'ch2_ego_is_comfortable': 15, 'ch2_ego_progress_along_expert_route': 16, 'ch2_time_to_collision_within_bound': 17, 'ch2_speed_limit_compliance': 18, 'ch3_overall_score': 110, 'ch3_ego_is_making_progress': 111, 'ch3_no_ego_at_fault_collisions': 112, 'ch3_drivable_area_compliance': 113, 'ch3_driving_direction_compliance': 114, 'ch3_ego_is_comfortable': 115, 'ch3_ego_progress_along_expert_route': 116, 'ch3_time_to_collision_within_bound': 117, 'ch3_speed_limit_compliance': 118, 'combined_overall_score': 40.0}
        self.assertEqual(metrics, expected_metrics)

def _download_files(bucket: str, client: boto3.client, local_path_name: str, keys: List[str], filters: Optional[List[str]]=None) -> None:
    """
    Downloads a list of objects from s3
    :param bucket: The s3 bucket
    :param client: The s3 client
    :param local_path_name: the base path
    :param keys: The name of the objects to download
    """
    local_path = pathlib.Path(local_path_name)
    filtered_keys = filter_paths(keys, filters)
    for key in filtered_keys:
        dest_file = local_path / key
        dest_file.parent.mkdir(exist_ok=True, parents=True)
        client.download_file(bucket, key, str(dest_file))

def filter_paths(paths: List[str], filters: Optional[List[str]]) -> List[str]:
    """Filters a list of paths according to a list of filters.
    :param paths: The input paths.
    :param filters: The filters of the elements to keep.
    :return: The subset of paths that contain at least one of the keywords defined in filters.
    """
    return [path for path in paths if not filters or any((_filter in path for _filter in filters))]

def s3_download_dir(bucket: str, client: boto3.client, prefix: str, local_path_name: str, filters: Optional[List[str]]=None) -> None:
    """
    Downloads targets matching a prefix from s3
    :param bucket: The s3 bucket
    :param client: The s3 client
    :param prefix: Prefix used to filer targets to download
    :param local_path_name: the base path
    :param filters: Keywords to filter paths, if empty no filtering is performed.
    """
    directories, keys = list_objects(bucket, client, prefix)
    _create_directories(local_path_name, directories)
    _download_files(bucket, client, local_path_name, keys, filters)

def list_objects(bucket: str, client: boto3.client, prefix: str) -> Tuple[List[str], List[str]]:
    """
    Returns files and directories in the bucket at the given prefix.
    :param bucket: The s3 bucket
    :param client: The s3 client
    :param prefix: Prefix used to filer targets to download
    :return: A list of directories and a list of files on the bucket matching the prefix.
    """
    keys: List[str] = []
    directories: List[str] = []
    next_token = 'InitialToken'
    list_request = {'Bucket': bucket, 'Prefix': prefix}
    while next_token:
        results = client.list_objects_v2(**list_request)
        contents = results.get('Contents')
        if not contents:
            break
        for content in contents:
            target: str = content.get('Key')
            if target.endswith('/'):
                directories.append(target)
            else:
                keys.append(target)
        next_token = results.get('NextContinuationToken')
        list_request['ContinuationToken'] = next_token
    return (directories, keys)

def _create_directories(local_path_name: str, directories: List[str]) -> None:
    """
    Creates directories from a list of directory names and a base path
    :param local_path_name: the base path
    :param directories: The name of the directories to create
    """
    local_path = pathlib.Path(local_path_name)
    for _dir in directories:
        (local_path / _dir).mkdir(exist_ok=True, parents=True)

class TestAWSUtils(unittest.TestCase):
    """Tests for AWS utils."""

    def test__create_directories(self) -> None:
        """Checks the right number of directories is created."""
        local_path_name = 'base_path'
        directories = ['foo', 'bar']
        with patch(f'{TEST_FILE_PATH}.pathlib.Path.mkdir') as mock_mkdir:
            _create_directories(local_path_name, directories)
            self.assertEqual(2, mock_mkdir.call_count)

    @patch(f'{TEST_FILE_PATH}.pathlib.Path.mkdir', Mock)
    def test__download_files(self) -> None:
        """Checks the S3 client is used correctly."""
        mock_client = Mock()
        files = ['file1', 'file2']
        expected_calls = (call('bucket', 'file1', 'dest/file1'), call('bucket', 'file2', 'dest/file2'))
        _download_files('bucket', mock_client, 'dest', files)
        mock_client.download_file.assert_has_calls(expected_calls)

    @patch(f'{TEST_FILE_PATH}._create_directories')
    @patch(f'{TEST_FILE_PATH}._download_files', Mock)
    def test_s3_download_dir_empty(self, mock_dir_create: Mock) -> None:
        """Tests S3 download dir downloads correct targets."""
        mock_client = StubS3Client(empty=True)
        bucket = 'bucket'
        prefix = 'prefix'
        local_path_name = '/my/path'
        s3_download_dir(bucket, mock_client, prefix, local_path_name)
        mock_dir_create.assert_called_once_with(local_path_name, [])

    @patch(f'{TEST_FILE_PATH}._create_directories')
    @patch(f'{TEST_FILE_PATH}._download_files')
    def test_s3_download_dir(self, mock_download: Mock, mock_dir_create: Mock) -> None:
        """Tests S3 download dir downloads correct targets."""
        mock_client = StubS3Client(empty=False)
        bucket = 'bucket'
        prefix = 'prefix'
        local_path_name = '/my/path'
        s3_download_dir(bucket, mock_client, prefix, local_path_name)
        mock_dir_create.assert_called_once_with(local_path_name, ['prefix/'])
        mock_download.assert_called_once_with('bucket', mock_client, local_path_name, ['prefix/file'], None)

    @patch(f'{TEST_FILE_PATH}.boto3.client')
    @patch(f'{TEST_FILE_PATH}.s3_download_dir')
    @patch.dict(os.environ, {'NUPLAN_SERVER_AWS_ACCESS_KEY_ID': 'key', 'NUPLAN_SERVER_AWS_SECRET_ACCESS_KEY': 'secret', 'NUPLAN_SERVER_S3_ROOT_URL': 'bucket'}, clear=True)
    def test_s3_download(self, mock_download_dir: Mock, mock_client: Mock) -> None:
        """Tests S3 download calls the correct api."""
        s3_download('prefix', 'path')
        mock_client.assert_called_once_with('s3', aws_access_key_id='key', aws_secret_access_key='secret', region_name='us-east-1')
        mock_download_dir.assert_called_once_with('bucket', mock_client.return_value, 'prefix', 'path', None)

