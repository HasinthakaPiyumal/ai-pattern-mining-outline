# Cluster 9

def download_file_if_necessary(data_root: str, potentially_remote_path: str, verbose: bool=False) -> str:
    """
    Downloads the db file if necessary.
    :param data_root: Path's data root.
    :param potentially_remote_path: The path from which to download the file.
    :param verbose: Verbosity level.
    :return: The local path for the file.
    """
    if os.path.exists(potentially_remote_path):
        return potentially_remote_path
    log_name = absolute_path_to_log_name(potentially_remote_path)
    download_name = log_name + '.db'
    os.makedirs(data_root, exist_ok=True)
    local_store = LocalStore(data_root)
    if not local_store.exists(download_name):
        blob_store = BlobStoreCreator.create_nuplandb(data_root, verbose=verbose)
        logger.info('DB path not found. Downloading to %s...' % download_name)
        start_time = time.time()
        remote_key = potentially_remote_path
        if not remote_key.startswith('s3://'):
            fixed_local_path = convert_legacy_nuplan_path_to_latest(potentially_remote_path)
            remote_key = infer_remote_key_from_local_path(fixed_local_path)
        content = blob_store.get(remote_key)
        local_store.put(download_name, content)
        logger.info('Downloading db file took %.2f seconds.' % (time.time() - start_time))
    return os.path.join(data_root, download_name)

class DB:
    """
    Base class for DB loaders. Inherited classes should implement property method for each table with type
    annotation, for example:
        class NuPlanDB(DB):
            @property
            def category(self) -> Table[nuplandb_model.Category]:
                return self.tables['category']

    It is not recommended to use db.get('category', some_token), use db.category.get(some_token) or
    db.category[some_token] instead, because we can't get any type hint from the former one.
    """

    def __init__(self, table_names: List[str], models: Any, data_root: str, db_path: str, verbose: bool, model_source_dict: Dict[str, str]={}):
        """
        Initialize database by loading from filesystem or downloading from S3, load json table and build token index.
        :param table_names: List of table names.
        :param models: Auto-generated model template.
        :param data_root: Path to load the database from; if the database is downloaded from S3
                          this is the path to store the downloaded database.
        :param db_path: Local or S3 path to the database file.
        :param verbose: Whether to print status messages when loading the database.
        """
        self._table_names = list(table_names)
        self._data_root = data_root
        self._blob_store = BlobStoreCreator.create_nuplandb(data_root)
        self._tables = {}
        self._tables_detached = False
        self._refcount = 1
        self._refcount_lock = threading.Lock()
        db_path = db_path if db_path.endswith('.db') else f'{db_path}.db'
        self._db_path = Path(db_path)
        self._filename = self._db_path if self._db_path.exists() else Path(self._data_root) / self._db_path.name
        if not self._filename.exists():
            logger.debug(f'DB path not found, downloading db file to {self._filename}...')
            start_time = time.time()
            cache_store = CacheStore(self._data_root, self._blob_store)
            cache_store.save_to_disk(self._db_path.name)
            logger.debug('Downloading db file took {:.1f} seconds'.format(time.time() - start_time))
        if verbose:
            logger.debug('\nLoading tables for database {}...'.format(self.name))
            start_time = time.time()
        self._session_manager = SessionManager(self._create_db_instance)
        for table_name in self._table_names:
            model_name = ''.join([s.capitalize() for s in table_name.split('_')])
            if len(model_source_dict) != 0:
                if model_name in model_source_dict:
                    model_pcls = getattr(models, model_source_dict[model_name])
                else:
                    model_pcls = getattr(models, model_source_dict['default'])
                model_cls = getattr(model_pcls, model_name)
            else:
                model_cls = getattr(models, model_name)
            self._tables[table_name] = Table[model_cls](model_cls, self)
        if verbose:
            for table_name in self._table_names:
                logger.debug('{} {},'.format(len(self._tables[table_name]), table_name))
            logger.debug('Done loading in {:.1f} seconds.\n'.format(time.time() - start_time))

    def __repr__(self) -> str:
        """
        Get the string representation.
        :return: The string representation.
        """
        return "{}('{}', data_root='{}')".format(self.__class__.__name__, self.name, self.data_root)

    def __str__(self) -> str:
        """
        Get the string representation.
        :return: The string representation.
        """
        _str = '{} {} with tables:\n{}'.format(self.__class__.__name__, self.name, '=' * 30)
        for table_name in self.table_names:
            if 'log' == table_name:
                continue
            _str += '\n{:20}: {}'.format(table_name, getattr(self, table_name).count())
        return _str

    @property
    def session(self) -> Session:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return self._session_manager.session

    @property
    def name(self) -> str:
        """
        Get the db name.
        :return: The db name.
        """
        return self._db_path.stem

    @property
    def data_root(self) -> str:
        """
        Get the data root.
        :return: The data root.
        """
        return self._data_root

    @property
    def table_root(self) -> str:
        """
        Get the table root.
        :return: The table root.
        """
        return str(self._filename)

    @property
    def table_names(self) -> List[str]:
        """
        Get the list of table names.
        :return: The list of table names.
        """
        self._assert_tables_attached()
        return self._table_names

    @property
    def tables(self) -> Dict[str, Table[Any]]:
        """
        Get the list of tables.
        :return: The list of tables.
        """
        self._assert_tables_attached()
        return self._tables

    def load_blob(self, path: str) -> BinaryIO:
        """
        Loads a blob.
        :param path: Path to the blob.
        :return: A binary stream to read the blob.
        """
        return self._blob_store.get(path)

    def get(self, table: str, token: str) -> Any:
        """
        Returns a record from table.
        :param table: Table name.
        :param token: Token of the record.
        :return: The record. See "templates.py" for details.
        """
        warnings.warn('deprecated', DeprecationWarning)
        self._assert_tables_attached()
        return getattr(self, table).get(token)

    def field2token(self, table: str, field: str, query: str) -> List[str]:
        """
        Function returns a list of tokens given a table and field of that table.
        :param table: Table name.
        :param field: Field name, see "template.py" for details.
        :param query: The same type as the field.
        :return: Return a list of record tokens.
        """
        warnings.warn('deprecated', DeprecationWarning)
        self._assert_tables_attached()
        return [rec.token for rec in getattr(self, table).search(**{field: query})]

    def are_tables_detached(self) -> bool:
        """
        Returns true if the tables have been detached, false otherwise.
        :returns: True if the tables have been detached, false otherwise.
        """
        return self._tables_detached

    def detach_tables(self) -> None:
        """
        Prepares all tables for destruction.
        This must be called when DB is ready to be released to reclaim used memory.
        After calling this method, no further queries should be run from the db.

        Placing this in __del__ is not sufficient, because without detaching tables,
          SQLAlchemy will keep references to the tables alive.
          Which contain references to the DB.
          Which means that __del__ will never be called.
        """
        if not self._tables_detached:
            for table_name in self.table_names:
                self.tables[table_name].detach()
            self._tables_detached = True

    def _assert_tables_attached(self) -> None:
        """
        Checks to ensure that the tables are attached. If not, raises an error.
        """
        if self.are_tables_detached():
            raise RuntimeError('Attempting to query from detached tables.')

    def add_ref(self) -> None:
        """
        Add an external reference to this class to prevent it from being reclaimed by the GC.
        This method should be called when any non-SqlAlchemy class takes a reference to the class.

        See the comments in __init__ for explanation
        """
        with self._refcount_lock:
            if self._refcount == 0:
                raise ValueError('Attempting to revive a database that has had its tables detached. This is likely due to a reference counting error.')
            self._refcount += 1

    def remove_ref(self) -> None:
        """
        Removes an external reference to this class.
        This should be called when any non-SqlAlchemy class is finished using the database (e.g. in their __del__ method).
        If the reference count gets to zero, it will be prepared for collection by the GC.
        """
        with self._refcount_lock:
            self._refcount -= 1
            if self._refcount == 0:
                self.detach_tables()

    def _create_db_instance(self) -> sqlite3.Connection:
        """
        Internal method, return sqlite3 connection for sqlalchemy.
        :return: Sqlite3 connection.
        """
        assert Path(self.table_root).exists(), 'DB file not found: {}'.format(self.table_root)
        db = sqlite3.connect('file:{}?mode=ro'.format(self.table_root), uri=True, check_same_thread=False)
        db.execute('PRAGMA main.journal_mode = OFF;')
        db.execute('PRAGMA main.cache_size=10240;')
        db.execute('PRAGMA main.page_size = 4096;')
        db.execute('PRAGMA main.journal_mode = OFF;')
        db.execute('PRAGMA query_only = 1;')
        return db

class BlobStoreCreator:
    """BlobStoreCreator Class."""

    @classmethod
    def create_nuplandb(cls, data_root: str, verbose: bool=False) -> BlobStore:
        """
        Create nuPlan DB blob storage.

        :param data_root: nuPlan database root.
        :param verbose: Verbose setting, defaults to False.
        :return: Blob storage created.
        """
        conf = RemoteConfig(http_root_url=os.getenv('NUPLAN_DATA_ROOT_HTTP_URL', ''), s3_root_url=os.getenv('NUPLAN_DATA_ROOT_S3_URL', ''))
        return cls.create(data_root, conf, verbose)

    @classmethod
    def create_mapsdb(cls, map_root: str, verbose: bool=False) -> BlobStore:
        """
        Create Maps DB blob storage.

        :param map_root: Maps database root.
        :param verbose: Verbose setting, defaults to False.
        :return: Blob storage created.
        """
        conf = RemoteConfig(http_root_url=os.getenv('NUPLAN_MAPS_ROOT_HTTP_URL', ''), s3_root_url=os.getenv('NUPLAN_MAPS_ROOT_S3_URL', ''))
        return cls.create(map_root, conf, verbose)

    @classmethod
    def create(cls, data_root: str, conf: RemoteConfig, verbose: bool=False) -> BlobStore:
        """
        Create blob storage.

        :param data_root: Data root.
        :param conf: Configuration to use.
        :param verbose: Verbose setting, defaults to False.
        :return: Blob storage created.
        """
        if NUPLAN_DATA_STORE == 'http':
            if not conf.http_root_url:
                raise ValueError('HTTP root url to be specified if using http storage.')
            requests.get(conf.http_root_url, timeout=2.0)
            logger.debug(f'Using HTTP blob store {conf.http_root_url} WITH local disk cache at {data_root}')
            return CacheStore(data_root, HttpStore(conf.http_root_url))
        elif NUPLAN_DATA_STORE == 'local':
            logger.debug(f'Using local disk store at {data_root} with no remote store')
            return LocalStore(data_root)
        elif NUPLAN_DATA_STORE == 's3':
            if not conf.s3_root_url:
                raise ValueError(f'S3 root url to be specified if using s3 storage. s3_root_url: {conf.s3_root_url}')
            store = S3Store(conf.s3_root_url, show_progress=verbose)
            if NUPLAN_CACHE_FROM_S3:
                logger.debug(f'Using s3 blob store for {conf.s3_root_url} WITH local disk cache at {data_root}')
                return CacheStore(data_root, store)
            else:
                logger.debug(f'Using s3 blob store for {conf.s3_root_url} WITHOUT local disk cache')
                return store
        else:
            raise ValueError(f"Environment variable NUPLAN_DATA_STORE was set to '{NUPLAN_DATA_STORE}'. Valid values are 'http', 'local', 's3'.")

class CacheStore(BlobStore):
    """
    Cache store, it combines a remote blob store and local store. The idea is to load blob
    from a remote store and cache it in local store so the next time we can load it from
    local.
    """

    def __init__(self, cache_dir: str, remote: BlobStore) -> None:
        """
        Initialize CacheStore.
        :param cache_dir: Path where to cache.
        :param remote: BlobStore instance.
        """
        os.makedirs(cache_dir, exist_ok=True)
        self._local = LocalStore(cache_dir)
        self._cache_dir = cache_dir
        self._remote = remote
        self._on_disk: Set[str] = set()

    def __reduce__(self) -> Tuple[Type[CacheStore], Tuple[str, BlobStore]]:
        """
        :return: tuple of class and its constructor parameters, this is used to pickle the class.
        """
        return (self.__class__, (self._cache_dir, self._remote))

    def get(self, key: str, check_for_compressed: bool=False) -> BinaryIO:
        """
        Get blob content if its present. Else download and then return.
        :param key: Blob path or token.
        :param check_for_compressed: Flag that check for a "<key>+.gzip" file and extracts the <key> file.
        :return: A file-like object, use read() to get raw bytes.
        """
        if self.exists(key):
            content: BinaryIO = self._local.get(key)
        else:
            content = self._remote.get(key, check_for_compressed)
            key_split = key.split('/')
            self.save(key_split[-1], content)
            content.seek(0)
        return content

    def save_to_disk(self, key: str, check_for_compressed: bool=False) -> None:
        """
        Save content to disk.
        :param key: Blob path or token.
        :param check_for_compressed: Flag that check for a "<key>+.gzip" file and extracts the <key> file.
        """
        if not self.exists(key):
            content = self._remote.get(key, check_for_compressed)
            self.save(key, content)

    async def get_async(self, key: str) -> BinaryIO:
        """Inherited, see superclass."""
        raise NotImplementedError('Not today.')

    def exists(self, key: str) -> bool:
        """
        Check if the blob exists.
        :param key: blob path or token.
        :return: True if the blob exists else False.
        """
        if key in self._on_disk:
            return True
        if self._local.exists(key):
            self._on_disk.add(key)
            return True
        return False

    def put(self, key: str, value: BinaryIO) -> None:
        """
        Write content.
        :param key: Blob path or token.
        :param value: Data to save.
        """
        self._remote.put(key, value)
        value.seek(0)
        self._local.put(key, value)
        self._on_disk.add(key)

    def save(self, key: str, content: BinaryIO) -> None:
        """
        Save to disk.
        :param key: Blob path or token.
        :param content: Data to save.
        """
        assert os.access(self._cache_dir, os.W_OK), 'Can not write to %s' % self._cache_dir
        path = os.path.join(self._cache_dir, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as fp:
            fp.write(content.read())

def _read_metadata_from_s3(inputs: List[ReadMetadataFromS3Input]) -> List[CacheMetadataEntry]:
    """
    Reads metadata csv from s3.
    :param inputs: The inputs to use for the function.
    :returns: The read metadata.
    """
    outputs: List[CacheMetadataEntry] = []
    if len(inputs) == 0:
        return outputs
    sanitized_cache_path = safe_path_to_string(inputs[0].cache_path)
    s3_store = S3Store(sanitized_cache_path)
    for input_value in inputs:
        df = pd.read_csv(s3_store.get(input_value.metadata_filename))
        metadata_dict_list = df.to_dict('records')
        for metadata_dict in metadata_dict_list:
            outputs.append(CacheMetadataEntry(**metadata_dict))
    return outputs

class FeatureCacheS3(FeatureCache):
    """
    Store features remotely in S3
    """

    def __init__(self, s3_path: str) -> None:
        """
        Initialize the S3 remote feature cache.
        :param s3_path: Path to S3 directory where features will be stored to or loaded from.
        """
        self._store = S3Store(s3_path, show_progress=False)

    def exists_feature_cache(self, feature_file: pathlib.Path) -> bool:
        """Inherited, see superclass."""
        return cast(bool, check_s3_path_exists(self.with_extension(feature_file)))

    def with_extension(self, feature_file: pathlib.Path) -> str:
        """Inherited, see superclass."""
        fixed_s3_filename = f's3://{str(feature_file).lstrip('s3:/')}'
        return f'{fixed_s3_filename}.bin'

    def store_computed_feature_to_folder(self, feature_file: pathlib.Path, feature: AbstractModelFeature) -> bool:
        """Inherited, see superclass."""
        serialized_feature = BytesIO()
        joblib.dump(feature, serialized_feature)
        serialized_feature.seek(os.SEEK_SET)
        storage_key = self.with_extension(feature_file)
        successfully_stored_feature = self._store.put(storage_key, serialized_feature, ignore_if_client_error=True)
        return cast(bool, successfully_stored_feature)

    def load_computed_feature_from_folder(self, feature_file: pathlib.Path, feature_type: Type[AbstractModelFeature]) -> AbstractModelFeature:
        """Inherited, see superclass."""
        storage_key = self.with_extension(feature_file)
        serialized_feature = self._store.get(storage_key)
        feature = joblib.load(serialized_feature)
        return feature

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

def get_scenarios_from_db_file(params: GetScenariosFromDbFileParams) -> ScenarioDict:
    """
    Gets all of the scenarios present in a single sqlite db file that match the provided filter parameters.
    :param params: The filter parameters to use.
    :return: A ScenarioDict containing the relevant scenarios.
    """
    local_log_file_absolute_path = download_file_if_necessary(params.data_root, params.log_file_absolute_path, params.verbose)
    scenario_dict: ScenarioDict = {}
    for row in get_scenarios_from_db(local_log_file_absolute_path, params.filter_tokens, params.filter_types, params.filter_map_names, not params.remove_invalid_goals, params.include_cameras):
        scenario_type = row['scenario_type']
        if scenario_type is None:
            scenario_type = DEFAULT_SCENARIO_NAME
        if scenario_type not in scenario_dict:
            scenario_dict[scenario_type] = []
        extraction_info = None if params.expand_scenarios else params.scenario_mapping.get_extraction_info(scenario_type)
        scenario_dict[scenario_type].append(NuPlanScenario(data_root=params.data_root, log_file_load_path=params.log_file_absolute_path, initial_lidar_token=row['token'].hex(), initial_lidar_timestamp=row['timestamp'], scenario_type=scenario_type, map_root=params.map_root, map_version=params.map_version, map_name=row['map_name'], scenario_extraction_info=extraction_info, ego_vehicle_parameters=params.vehicle_parameters, sensor_root=params.sensor_root))
    return scenario_dict

def get_scenarios_from_log_file(parameters: List[GetScenariosFromDbFileParams]) -> List[ScenarioDict]:
    """
    Gets all scenarios from a log file that match the provided parameters.
    :param parameters: The parameters to use for scenario extraction.
    :return: The extracted scenarios.
    """
    output_dict: ScenarioDict = {}
    for parameter in parameters:
        this_dict = get_scenarios_from_db_file(parameter)
        for key in this_dict:
            if key not in output_dict:
                output_dict[key] = this_dict[key]
            else:
                output_dict[key] += this_dict[key]
    return [output_dict]

class TestNuPlanScenarioUtilsIntegration(unittest.TestCase):
    """Test cases for nuplan_scenario_utils.py"""

    def setUp(self) -> None:
        """Will be run before every test."""
        self.data_root = Path('/data/sets/nuplan/nuplan-v1.1/splits/mini/')
        self.local_path = self.data_root / '2021.09.16.15.12.03_veh-42_01037_01434.db'

    def test_download_file_if_necessary_local_path(self) -> None:
        """
        Test that download_file_if_necessary works as expected with local path input.
        WARNING: This test will attempt to remove and re-download 2021.09.16.15.12.03_veh-42_01037_01434.db from
                 the local splits folder.
        """
        if os.path.exists(self.local_path):
            os.remove(self.local_path)
        self.assertFalse(os.path.exists(self.local_path))
        download_file_if_necessary(str(self.data_root), str(self.local_path))
        self.assertTrue(os.path.exists(self.local_path))

    def test_download_file_if_necessary_remote_path(self) -> None:
        """
        Test that download_file_if_necessary works as expected.
        WARNING: This test will attempt to remove and re-download 2021.09.16.15.12.03_veh-42_01037_01434.db from
                 the local splits folder.
        """
        if os.path.exists(self.local_path):
            os.remove(self.local_path)
        self.assertFalse(os.path.exists(self.local_path))
        remote_path = 's3://nuplan-production/nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db'
        download_file_if_necessary(str(self.data_root), remote_path)
        self.assertTrue(os.path.exists(self.local_path))

