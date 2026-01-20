# Cluster 6

def visualize_nuplan_scenarios(data_root: str, db_files: str, map_root: str, map_version: str, bokeh_port: int=8899) -> None:
    """
    Create a dropdown box populated with unique scenario types to visualize from a database.
    :param data_root: The root directory to use for looking for db files.
    :param db_files: List of db files to load.
    :param map_root: The root directory to use for looking for maps.
    :param map_version: The map version to use.
    :param bokeh_port: Port that the server bokeh starts to render the generate the visualization will run on.
    """
    from IPython.display import clear_output, display
    from ipywidgets import Dropdown, Output
    log_db_files = discover_log_dbs(db_files)
    scenario_type_token_map = get_scenario_type_token_map(log_db_files)
    out = Output()
    drop_down = Dropdown(description='Scenario', options=sorted(scenario_type_token_map.keys()))

    def scenario_dropdown_handler(change: Any) -> None:
        """
        Dropdown handler that randomly chooses a scenario from the selected scenario type and renders it.
        :param change: Object containing scenario selection.
        """
        with out:
            clear_output()
            logger.info('Randomly rendering a scenario...')
            scenario_type = str(change.new)
            log_db_file, token = random.choice(scenario_type_token_map[scenario_type])
            scenario = get_default_scenario_from_token(data_root, log_db_file, token, map_root, map_version)
            visualize_scenario(scenario, bokeh_port=bokeh_port)
    display(drop_down)
    display(out)
    drop_down.observe(scenario_dropdown_handler, names='value')

def discover_log_dbs(load_path: Union[List[str], str]) -> List[str]:
    """
    Discover all log dbs from the input load path.
    If the path is a filename, expand the path and return the list of filenames in that path.
    Else, if the path is already a list, expand each path in the list and return the flattened list.
    :param load_path: Load path, it can be a filename or list of filenames of a database and/or dirs of databases.
    :return: A list with all discovered log database filenames.
    """
    if isinstance(load_path, list):
        nested_db_filenames = [get_db_filenames_from_load_path(path) for path in sorted(set(load_path))]
        db_filenames = [filename for filenames in nested_db_filenames for filename in filenames]
    else:
        db_filenames = get_db_filenames_from_load_path(load_path)
    return db_filenames

class TestTutorialUtils(unittest.TestCase):
    """Unit tests for tutorial_utils.py."""

    def test_scenario_visualization_utils(self) -> None:
        """Test if scenario visualization utils work as expected."""
        visualize_nuplan_scenarios(data_root=NUPLAN_DATA_ROOT, db_files=NUPLAN_DB_FILES, map_root=NUPLAN_MAPS_ROOT, map_version=NUPLAN_MAP_VERSION)

    def test_scenario_rendering(self) -> None:
        """Test if scenario rendering works."""
        bokeh_port = 8999
        output_notebook()
        scenario_type_token_map = get_scenario_type_token_map(NUPLAN_DB_FILES)
        available_keys = list(scenario_type_token_map.keys())
        log_db, token = scenario_type_token_map[available_keys[0]][0]
        scenario = get_default_scenario_from_token(NUPLAN_DATA_ROOT, log_db, token, NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION)
        for _ in range(2):
            visualize_scenario(scenario, bokeh_port=bokeh_port)
        for server in curstate().uuid_to_server.values():
            self.assertEqual(bokeh_port, server.port)

    def test_start_event_loop_if_needed(self) -> None:
        """Tests if start_event_loop_if_needed works."""

        async def test_fn() -> int:
            """Minimal async function"""
            return 1
        start_event_loop_if_needed()
        _ = asyncio.get_event_loop()
        start_event_loop_if_needed()
        _ = asyncio.get_event_loop()
        asyncio.run(test_fn())
        with self.assertRaises(RuntimeError):
            _ = asyncio.get_event_loop()
        start_event_loop_if_needed()
        _ = asyncio.get_event_loop()

def get_db_filenames_from_load_path(load_path: str) -> List[str]:
    """
    Retrieve all log database filenames from a load path.
    The path can be either local or remote (S3).
    The path can represent either a single database filename (.db file) or a directory containing files.
    :param load_path: Load path, it can be a filename or list of filenames.
    :return: A list of all discovered log database filenames.
    """
    if load_path.endswith('.db'):
        if load_path.startswith('s3://'):
            assert check_s3_path_exists(load_path), f'S3 db path does not exist: {load_path}'
            os.environ['NUPLAN_DATA_ROOT_S3_URL'] = load_path.rstrip(Path(load_path).name)
        else:
            assert Path(load_path).is_file(), f'Local db path does not exist: {load_path}'
        db_filenames = [load_path]
    elif load_path.startswith('s3://'):
        db_filenames = expand_s3_dir(load_path, filter_suffix='.db')
        assert len(db_filenames) > 0, f'S3 dir does not contain any dbs: {load_path}'
        os.environ['NUPLAN_DATA_ROOT_S3_URL'] = load_path
    elif Path(load_path).expanduser().is_dir():
        db_filenames = [str(path) for path in sorted(Path(load_path).expanduser().iterdir()) if path.suffix == '.db']
    else:
        raise ValueError(f'Expected db load path to be file, dir or list of files/dirs, but got {load_path}')
    return db_filenames

@retry(RETRYABLE_EXCEPTIONS, backoff=2, tries=7, delay=0.5, jitter=(0.5, 3))
def check_s3_path_exists(s3_path: Optional[str]) -> bool:
    """
    Check whether the S3 path exists.
    If "None" is passed, then the return will be false, because a "None" path will never exist.
    :param s3_path: S3 path to check.
    :return: Whether the path exists or not.
    """
    if s3_path is None:
        return False
    result: bool = asyncio.run(check_s3_path_exists_async(s3_path))
    return result

@retry(RETRYABLE_EXCEPTIONS, backoff=2, tries=7, delay=0.5, jitter=(0.5, 3))
def expand_s3_dir(s3_path: str, client: Optional[boto3.client]=None, filter_suffix: str='') -> List[str]:
    """
    Expand S3 path dir to a list of S3 path files.
    :param s3_path: S3 path dir to expand.
    :param client: Boto3 client to use, if None create a new one.
    :param filter_suffix: Optional suffix to filter S3 filenames with.
    :return: List of S3 filenames discovered.
    """
    logger.warning('Function expand_s3_dir will soon be removed in favor of list_files_in_s3_directory')
    client = get_s3_client() if client is None else client
    url = parse.urlparse(s3_path)
    paginator = client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=url.netloc, Prefix=url.path.lstrip('/'))
    filenames = [str(content['Key']) for page in page_iterator for content in page['Contents']]
    filenames = [f's3://{url.netloc}/{path}' for path in filenames if path.endswith(filter_suffix)]
    return filenames

def get_async_s3_session(profile_name: Optional[str]=None, aws_access_key_id: Optional[str]=None, aws_secret_access_key: Optional[str]=None, force_new: bool=False) -> aioboto3.Session:
    """
    Get synchronous boto3 session.
    :param profile_name: Optional profile name to authenticate with.
    :param aws_access_key_id: Optional access key to authenticate with.
    :param aws_secret_access_key: Optional secret access key to authenticate with.
    :param force_new: If true, ignore any cached  session and get a new one.
                      Any existing cached session will be overwritten.
    :return: Session object.
    """
    global G_ASYNC_SESSION
    if not force_new and G_ASYNC_SESSION is not None:
        return G_ASYNC_SESSION

    def _set_async_session_func(session: aioboto3.Session) -> None:
        global G_ASYNC_SESSION
        G_ASYNC_SESSION = session

    def _create_session_func(**kwargs: Any) -> aioboto3.Session:
        return aioboto3.Session(**kwargs)
    return _get_session_internal(profile_name, aws_access_key_id, aws_secret_access_key, _create_session_func, _set_async_session_func)

def _get_session_internal(profile_name: Optional[str], aws_access_key_id: Optional[str], aws_secret_access_key: Optional[str], create_session_func: Callable[..., Union[boto3.Session, aioboto3.Session]], set_session_func: Callable[[Union[boto3.Session, aioboto3.Session]], None]) -> Union[boto3.Session, aioboto3.Session]:
    """
    Get synchronous boto3 session.
    :param profile_name: Optional profile name to authenticate with.
    :param aws_access_key_id: Optional access key to authenticate with.
    :param aws_secret_access_key: Optional secret access key to authenticate with.
    :param create_session_func: Session creation function.
    :param set_session_func: Session caching function.
    :return: Session object.
    """
    args: Dict[str, Any] = {}
    if os.getenv('AWS_WEB_IDENTITY_TOKEN_FILE') is not None:
        logger.debug('Using AWS_WEB_IDENTITY_TOKEN_FILE for credentials.')
    elif profile_name is None and aws_access_key_id is None and (aws_secret_access_key is None):
        logger.debug('Using default credentials for AWS session.')
    else:
        logger.debug('Attempting to use credentialed authentication for S3 client...')
        args = {'profile_name': os.getenv('NUPLAN_S3_PROFILE', '') if profile_name is None else profile_name}
        if aws_access_key_id and aws_secret_access_key:
            args['aws_access_key_id'] = aws_access_key_id
            args['aws_secret_access_key'] = aws_secret_access_key
    try:
        session = create_session_func(**args)
        set_session_func(session)
    except BotoCoreError as e:
        if 'profile_name' in args:
            logger.info(f'Trying default AWS credential chain, since we got this exception while trying to use AWS profile [{args['profile_name']}]: {e}')
        session = create_session_func()
        set_session_func(session)
    return session

def _get_sync_session(profile_name: Optional[str]=None, aws_access_key_id: Optional[str]=None, aws_secret_access_key: Optional[str]=None, force_new: bool=False) -> boto3.Session:
    """
    Get synchronous boto3 session.
    :param profile_name: Optional profile name to authenticate with.
    :param aws_access_key_id: Optional access key to authenticate with.
    :param aws_secret_access_key: Optional secret access key to authenticate with.
    :param force_new: If true, ignore any cached session and get a new one.
                      Any existing cached session will be overwritten.
    :return: Session object.
    """
    global G_SYNC_SESSION
    if not force_new and G_SYNC_SESSION is not None:
        return G_SYNC_SESSION

    def _set_sync_session_func(session: boto3.Session) -> None:
        global G_SYNC_SESSION
        G_SYNC_SESSION = session

    def _create_session_func(**kwargs: Any) -> aioboto3.Session:
        return boto3.Session(**kwargs)
    return _get_session_internal(profile_name, aws_access_key_id, aws_secret_access_key, _create_session_func, _set_sync_session_func)

def split_s3_path(s3_path: Path) -> Tuple[str, Path]:
    """
    Splits a S3 path into a (bucket, path) set of identifiers.
    :param s3_path: The full S3 path.
    :return: A tuple of (bucket, path).
    """
    if not is_s3_path(s3_path):
        raise ValueError(f'{str(s3_path)} is not an s3 path.')
    chunks = [v.strip() for v in str(s3_path).split('/') if len(v.strip()) > 0]
    bucket = chunks[1]
    path = Path('/'.join(chunks[2:]))
    return (bucket, path)

def _trim_leading_slash_if_exists(path: Union[str, Path]) -> Path:
    """
    Trims the leading slash in a path if it exists.
    :param path: The path to trim.
    :return: The trimmed path.
    """
    path_str = str(path)
    if path_str == '/':
        raise ValueError("Path is the root path '/'. This should never happen.")
    path_str = path_str[1:] if path_str.startswith('/') else path_str
    return Path(path_str)

class DistributedScenarioFilter:
    """
    Class to distribute the work to build / filter scenarios across workers, and to break up those scenarios in chunks to be
    handled on individual machines
    """

    def __init__(self, cfg: DictConfig, worker: WorkerPool, node_rank: int, num_nodes: int, synchronization_path: str, timeout_seconds: int=7200, distributed_mode: DistributedMode=DistributedMode.SCENARIO_BASED):
        """
        :param cfg: top level config for the job (used to build scenario builder / scenario_filter)
        :param worker: worker to use in each node to parallelize the work
        :param node_rank: number from (0, num_nodes -1) denoting "which" node we are on
        :param num_nodes: total number of nodes the job is running on
        :param synchronization_path: path that can be in s3 or on a shared file system that will be used to synchronize
                                     across workers
        :param timeout_seconds: how long to wait during sync operations
        :param distributed_mode: what distributed mode to use to distribute computation
        """
        self._cfg = cfg
        self._worker = worker
        self._node_rank = node_rank
        self._num_nodes = num_nodes
        self.synchronization_path = synchronization_path
        self._timeout_seconds = timeout_seconds
        self._distributed_mode = distributed_mode

    def get_scenarios(self) -> List[AbstractScenario]:
        """
        Get all the scenarios that the current node should process
        :returns: list of scenarios for the current node
        """
        if self._num_nodes == 1 or self._distributed_mode == DistributedMode.SINGLE_NODE:
            logger.info('Building Scenarios in mode %s', DistributedMode.SINGLE_NODE)
            scenario_builder = build_scenario_builder(cfg=self._cfg)
            scenario_filter = build_scenario_filter(cfg=self._cfg.scenario_filter)
        elif self._distributed_mode in (DistributedMode.LOG_FILE_BASED, DistributedMode.SCENARIO_BASED):
            logger.info('Getting Log Chunks')
            current_chunk = self._get_log_db_files_for_single_node()
            logger.info('Getting Scenarios From Log Chunk of size %d', len(current_chunk))
            scenarios = self._get_scenarios_from_list_of_log_files(current_chunk)
            if self._distributed_mode == DistributedMode.LOG_FILE_BASED:
                logger.info('Distributed mode is %s, so we are just returning the scenariosfound from log files on the current worker.  There are %d scenarios to processon node %d/%d', DistributedMode.LOG_FILE_BASED, len(scenarios), self._node_rank, self._num_nodes)
                return scenarios
            logger.info('Distributed mode is %s, so we are going to repartition the scenarios we got from the log files to better distribute the work', DistributedMode.SCENARIO_BASED)
            logger.info('Getting repartitioned scenario tokens')
            tokens, log_db_files = self._get_repartition_tokens(scenarios)
            OmegaConf.set_struct(self._cfg, False)
            self._cfg.scenario_filter.scenario_tokens = tokens
            self._cfg.scenario_builder.db_files = log_db_files
            OmegaConf.set_struct(self._cfg, True)
            logger.info('Building repartitioned scenarios')
            scenario_builder = build_scenario_builder(cfg=self._cfg)
            scenario_filter = build_scenario_filter(cfg=self._cfg.scenario_filter)
        else:
            raise ValueError(f'Distributed mode must be one of {[x.name for x in fields(DistributedMode)]}, got {self._distributed_mode} instead!')
        scenarios = scenario_builder.get_scenarios(scenario_filter, self._worker)
        return scenarios

    def _get_repartition_tokens(self, scenarios: List[AbstractScenario]) -> Tuple[List[str], List[str]]:
        """
        Submit list of scenarios found by the current node, sync up with other nodes to get the full list of tokens,
        and calculate the current node's set of tokens to process
        :param scenarios: Scenarios found by the current node
        :returns: (list of tokens, list of db files)
        """
        unique_job_id = get_unique_job_id()
        token_distribution_file_dir = Path(self.synchronization_path) / Path('tokens') / Path(unique_job_id)
        token_distribution_barrier_dir = Path(self.synchronization_path) / Path('barrier') / Path(unique_job_id)
        if self.synchronization_path.startswith('s3'):
            token_distribution_file_dir = safe_path_to_string(token_distribution_file_dir)
            token_distribution_barrier_dir = safe_path_to_string(token_distribution_barrier_dir)
        self._write_token_csv_file(scenarios, token_distribution_file_dir)
        distributed_sync(token_distribution_barrier_dir, timeout_seconds=self._timeout_seconds)
        token_distribution = self._get_all_generated_csv(token_distribution_file_dir)
        db_files_path = Path(self._cfg.scenario_builder.db_files[0]).parent if isinstance(self._cfg.scenario_builder.db_files, (list, ListConfig)) else Path(self._cfg.scenario_builder.db_files)
        return self._get_token_and_log_chunk_on_single_node(token_distribution, db_files_path)

    def _get_all_generated_csv(self, token_distribution_file_dir: Union[Path, str]) -> List[Tuple[str, str]]:
        """
        Read the csv files that every machine in the cluster generated and get the full list of (token, db_file) pairs
        :param token_distribution_file_dir: path where to the csv files are stored
        :returns: full list of (token, db_file) pairs
        """
        if self.synchronization_path.startswith('s3'):
            token_distribution_file_list = [el for el in expand_s3_dir(token_distribution_file_dir) if el.endswith('.csv')]
            token_distribution_list = []
            bucket, file_path = split_s3_path(Path(token_distribution_file_list[0]))
            s3_store = S3Store(s3_prefix=os.path.join('s3://', bucket))
            for token_distribution_file in token_distribution_file_list:
                with s3_store.get(token_distribution_file) as f:
                    try:
                        token_distribution_list.append(pd.read_csv(f, delimiter=','))
                    except EmptyDataError:
                        logger.warning('Token file for worker %s was empty, this may mean that something is wrong with yourconfiguration, or just that all of the data on that worker got filtered out.', token_distribution_file)
        else:
            token_distribution_list = []
            for file_name in os.listdir(token_distribution_file_dir):
                try:
                    token_distribution_list.append(pd.read_csv(os.path.join(token_distribution_file_dir, str(file_name))))
                except EmptyDataError:
                    logger.warning('Token file for worker %s was empty, this may mean that something is wrong with yourconfiguration, or just that all of the data on that worker got filtered out.', file_name)
        if not token_distribution_list:
            raise AssertionError('No scenarios found to simulate!')
        token_distribution_df = pd.concat(token_distribution_list, ignore_index=True)
        token_distribution = token_distribution_df.values.tolist()
        return cast(List[Tuple[str, str]], token_distribution)

    def _get_token_and_log_chunk_on_single_node(self, token_distribution: List[Tuple[str, str]], db_files_path: Path) -> Tuple[List[str], List[str]]:
        """
        Get the list of tokens and the list of logs those tokens are found in restricted to the current node
        :param token_distribution: Full list of all (token, log_file) pairs to be divided among the nodes
        :param db_files_path: Path to the actual db files
        """
        db_files_path_sanitized = safe_path_to_string(db_files_path)
        if not check_s3_path_exists(db_files_path_sanitized):
            raise AssertionError(f'Multinode caching only works in S3, but db_files path given was {db_files_path_sanitized}')
        token_distribution_chunk = chunk_list(token_distribution, self._num_nodes)
        current_chunk = token_distribution_chunk[self._node_rank]
        current_logs_chunk = list({os.path.join(db_files_path_sanitized, f'{pair[1]}.db') for pair in current_chunk})
        current_token_chunk = [pair[0] for pair in current_chunk]
        return (current_token_chunk, current_logs_chunk)

    def _write_token_csv_file(self, scenarios: List[AbstractScenario], token_distribution_file_dir: Union[str, Path]) -> None:
        """
        Writes a csv file of format token,log_name that stores the tokens associated with the given scenarios
        :param scenarios: Scenarios to take token/log pairs from
        :param token_distribution_file_dir: directory to write our csv file to
        """
        token_distribution_file = os.path.join(token_distribution_file_dir, f'{self._node_rank}.csv')
        token_log_pairs = [(scenario.token, scenario.log_name) for scenario in scenarios]
        os.makedirs(token_distribution_file_dir, exist_ok=True)
        token_log_pairs_df = pd.DataFrame(token_log_pairs)
        token_log_pairs_df.to_csv(token_distribution_file, index=False)

    def _get_scenarios_from_list_of_log_files(self, log_db_files: List[str]) -> List[AbstractScenario]:
        """
        Gets the scenarios based on self._cfg, restricted to a list of log files
        :param log_db_files: list of log db files to restrict our search to
        :returns: list of scenarios
        """
        OmegaConf.set_struct(self._cfg, False)
        self._cfg.scenario_builder.db_files = log_db_files
        OmegaConf.set_struct(self._cfg, True)
        scenario_builder = build_scenario_builder(self._cfg)
        scenario_filter = build_scenario_filter(self._cfg.scenario_filter)
        scenarios: List[AbstractScenario] = scenario_builder.get_scenarios(scenario_filter, self._worker)
        return scenarios

    def _get_log_db_files_for_single_node(self) -> List[str]:
        """
        Get the list of log db files to be run on the current node
        :returns: list of log db files
        """
        if self._num_nodes == 1:
            return cast(List[str], self._cfg.scenario_builder.db_files)
        if not check_s3_path_exists(self._cfg.scenario_builder.db_files):
            raise AssertionError(f'DistributedScenarioFilter with multiple nodes only works in S3, but db_files path given was {self._cfg.scenario_builder.db_files}')
        all_files = get_db_filenames_from_load_path(self._cfg.scenario_builder.db_files)
        file_chunks = chunk_list(all_files, self._num_nodes)
        current_chunk = file_chunks[self._node_rank]
        return cast(List[str], current_chunk)

def chunk_list(input_list: List[Any], num_chunks: Optional[int]=None) -> List[List[Any]]:
    """
    Chunks a list to equal sized lists. The size of the last list might be truncated.
    :param input_list: List to be chunked.
    :param num_chunks: Number of chunks, equals to the number of cores if set to None.
    :return: List of equal sized lists.
    """
    num_chunks = num_chunks if num_chunks else cpu_count(logical=True)
    chunks = np.array_split(input_list, num_chunks)
    return [chunk.tolist() for chunk in chunks if len(chunk) != 0]

def get_db_filenames_from_load_path(load_path: str) -> List[str]:
    """
    Retrieve all log database filenames from a load path.
    The path can be either local or remote (S3).
    The path can represent either a single database filename (.db file) or a directory containing files.
    :param load_path: Load path, it can be a filename or list of filenames.
    :return: A list of all discovered log database filenames.
    """
    if load_path.endswith('.db'):
        if load_path.startswith('s3://'):
            assert check_s3_path_exists(load_path), f'S3 db path does not exist: {load_path}'
            os.environ['NUPLAN_DATA_ROOT_S3_URL'] = load_path.rstrip(Path(load_path).name)
        else:
            assert Path(load_path).is_file(), f'Local db path does not exist: {load_path}'
        db_filenames = [load_path]
    elif load_path.startswith('s3://'):
        db_filenames = expand_s3_dir(load_path, filter_suffix='.db')
        assert len(db_filenames) > 0, f'S3 dir does not contain any dbs: {load_path}'
        os.environ['NUPLAN_DATA_ROOT_S3_URL'] = load_path
    elif Path(load_path).expanduser().is_dir():
        db_filenames = [str(path) for path in sorted(Path(load_path).expanduser().iterdir()) if path.suffix == '.db']
    else:
        raise ValueError(f'Expected db load path to be file, dir or list of files/dirs, but got {load_path}')
    return db_filenames

class TestS3Utils(unittest.TestCase):
    """
    A class to test that the S3 utilities function properly.
    """

    def test_is_s3_path(self) -> None:
        """
        Tests that the is_s3_path method works properly.
        """
        self.assertTrue(is_s3_path(Path('s3://foo/bar/baz.txt')))
        self.assertFalse(is_s3_path(Path('/foo/bar/baz')))
        self.assertFalse(is_s3_path(Path('foo/bar/baz')))
        self.assertTrue(is_s3_path('s3://foo/bar/baz.txt'))
        self.assertFalse(is_s3_path('/foo/bar/baz'))
        self.assertFalse(is_s3_path('foo/bar/baz'))

    def test_split_s3_path(self) -> None:
        """
        Tests that the split_s3_path method works properly.
        """
        sample_s3_path = Path('s3://test-bucket/foo/bar/baz.txt')
        expected_bucket = 'test-bucket'
        expected_path = Path('foo/bar/baz.txt')
        actual_bucket, actual_path = split_s3_path(sample_s3_path)
        self.assertEqual(expected_bucket, actual_bucket)
        self.assertEqual(expected_path, actual_path)

    @mock_async_s3()
    def test_get_async_s3_session(self) -> None:
        """
        Tests that getting a session works correctly.
        """
        sess_1 = get_async_s3_session()
        sess_2 = get_async_s3_session()
        self.assertEqual(sess_1, sess_2)
        sess_3 = get_async_s3_session(force_new=True)
        sess_4 = get_async_s3_session()
        self.assertNotEqual(sess_2, sess_3)
        self.assertEqual(sess_3, sess_4)

    @mock_async_s3()
    def test_download_directory_from_s3(self) -> None:
        """
        Tests that the download_directory_from_s3 method works properly while mocking AWS.
        Assumes that upload_file_to_s3_async works (used to setup test directory in mock bucket).
        """
        test_upload_directory = Path('test_download_directory_from_s3')
        test_bucket_name = 'test-bucket'
        expected_relative_path_and_contents = {'file1.txt': 'this is file1.', 'dir1/file2.txt': 'this is file2.', 'dir1/file3.txt': 'this is file3.'}
        asyncio.run(setup_mock_s3_directory(expected_relative_path_and_contents, test_upload_directory, test_bucket_name))
        with tempfile.TemporaryDirectory() as temp_dir:
            expected_directory_path_and_contents = {os.path.join(temp_dir, path): contents for path, contents in expected_relative_path_and_contents.items()}
            download_directory_from_s3(temp_dir, test_upload_directory, test_bucket_name)
            all_files = glob.glob(f'{temp_dir}/**/*.txt', recursive=True)
            self.assertEqual(len(all_files), len(expected_directory_path_and_contents))
            for key in expected_directory_path_and_contents:
                self.assertTrue(os.path.exists(key))
                with open(key, 'r') as f:
                    actual_text = f.read().strip()
                self.assertEqual(expected_directory_path_and_contents[key], actual_text)

    @mock_async_s3()
    def test_list_files_in_s3_directory(self) -> None:
        """
        Tests that the list_files_in_s3_directory method works properly while mocking AWS.
        Assumes that upload_file_to_s3_async works (used to setup test directory in mock bucket).
        """
        test_files_directory = Path('test_list_files_in_s3_directory')
        test_bucket_name = 'test-bucket'
        expected_relative_path_and_contents = {'file1.txt': 'this is file1.', 'dir1/file2.txt': 'this is file2.', 'dir1/file3.txt': 'this is file3.'}
        asyncio.run(setup_mock_s3_directory(expected_relative_path_and_contents, test_files_directory, test_bucket_name))
        expected_files = {test_files_directory / path for path in expected_relative_path_and_contents}
        actual_files = list_files_in_s3_directory(test_files_directory, test_bucket_name)
        self.assertEqual(len(expected_files), len(actual_files))
        for file_path in actual_files:
            self.assertTrue(file_path in expected_files)

    @mock_async_s3()
    def test_check_s3_exist_ops(self) -> None:
        """
        Tests that the check_s3_object_exists and check_s3_path_exists methods functions properly while mocking AWS.
        Assumes that upload_file_to_s3_async works (used to setup test directory in mock bucket).
        """

        def to_s3_path(key: Path, bucket: str) -> str:
            """
            Returns s3 path string from split path.
            :param key: s3 key.
            :param bucket: s3 bucket.
            :return: Unsplit s3 path.
            """
            return f's3://{bucket}/{key}'
        test_files_directory = Path('test_check_s3_object_exists')
        test_bucket_name = 'test-bucket'
        expected_relative_path_and_contents = {'existing.txt': 'this exists.'}
        asyncio.run(setup_mock_s3_directory(expected_relative_path_and_contents, test_files_directory, test_bucket_name))
        existing_key = test_files_directory / 'existing.txt'
        non_existing_key = test_files_directory / 'does_not_exist.txt'
        self.assertTrue(check_s3_object_exists(existing_key, test_bucket_name))
        self.assertTrue(check_s3_path_exists(to_s3_path(existing_key, test_bucket_name)))
        self.assertFalse(check_s3_object_exists(non_existing_key, test_bucket_name))
        self.assertFalse(check_s3_path_exists(to_s3_path(non_existing_key, test_bucket_name)))
        self.assertFalse(check_s3_object_exists(test_files_directory, test_bucket_name))
        self.assertTrue(check_s3_path_exists(to_s3_path(test_files_directory, test_bucket_name)))

    @mock_async_s3()
    def test_get_cache_metadata_paths(self) -> None:
        """
        Tests that the get_cache_metadata_paths method functions properly while mocking AWS.
        Assumes that upload_file_to_s3_async works (used to setup test directory in mock bucket).
        """
        test_files_directory = Path('test_get_cache_metadata_paths')
        test_bucket_name = 'test-bucket'
        expected_relative_path_and_contents = {'file1.csv': 'this is file1.', 'metadata/file2.csv': 'this is file2.', 'metadata/file3.csv': 'this is file3.'}
        asyncio.run(setup_mock_s3_directory(expected_relative_path_and_contents, test_files_directory, test_bucket_name))
        expected_metadata_files = [test_files_directory / 'metadata/file2.csv', test_files_directory / 'metadata/file3.csv']
        actual_metadata_files = get_cache_metadata_paths(test_files_directory, test_bucket_name)
        self.assertEqual(len(expected_metadata_files), len(actual_metadata_files))
        for s3_path in actual_metadata_files:
            bucket, file_path = split_s3_path(s3_path)
            self.assertTrue(file_path in expected_metadata_files)
        non_existing_files = get_cache_metadata_paths(test_files_directory, test_bucket_name, metadata_folder='non_existing')
        self.assertEqual(len(non_existing_files), 0)

    @mock_async_s3()
    def test_s3_single_file_ops(self) -> None:
        """
        Tests that the following methods work properly while mocking AWS:
        * Upload file to S3
        * Download file from S3
        * Read file from S3
        * Delete file from S3
        """
        upload_bucket_name = 'test-bucket'
        asyncio.run(create_mock_bucket(upload_bucket_name))
        test_id = str(uuid.uuid4())
        upload_bucket_folder = Path('test_upload_file_to_s3')
        upload_bucket_path = upload_bucket_folder / f'{test_id}.txt'
        expected_file_contents = f'A random identifier: {test_id}.'
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_file_path = Path(os.path.join(temp_dir, 'upload.txt'))
            with open(upload_file_path, 'w') as f:
                f.write(expected_file_contents)
            upload_file_to_s3(upload_file_path, upload_bucket_path, upload_bucket_name)
            self.assertEqual(1, len(list_files_in_s3_directory(upload_bucket_path, upload_bucket_name)))
            read_file_contents = read_text_file_contents_from_s3(upload_bucket_path, upload_bucket_name)
            self.assertEqual(expected_file_contents, read_file_contents)
            read_binary_contents = read_binary_file_contents_from_s3(upload_bucket_path, upload_bucket_name)
            self.assertEqual(expected_file_contents, read_binary_contents.decode('utf-8'))
            download_file_path = Path(os.path.join(temp_dir, 'download.txt'))
            download_file_from_s3(download_file_path, upload_bucket_path, upload_bucket_name)
            self.assertTrue(os.path.exists(download_file_path))
            with open(download_file_path, 'r') as f:
                downloaded_text = f.read()
            self.assertEqual(expected_file_contents, downloaded_text)
            delete_file_from_s3(upload_bucket_path, upload_bucket_name)
            self.assertEqual(0, len(list_files_in_s3_directory(upload_bucket_path, upload_bucket_name)))

def download_directory_from_s3(local_dir: Path, s3_key: Path, s3_bucket: str) -> None:
    """
    Downloads a directory to the local machine.
    :param local_dir: The directory to which to download.
    :param s3_key: The directory in S3 to download, without the bucket.
    :param s3_bucket: The bucket name to use.
    """
    asyncio.run(download_directory_from_s3_async(local_dir, s3_key, s3_bucket))

def list_files_in_s3_directory(s3_key: Path, s3_bucket: str, filter_suffix: str='') -> List[Path]:
    """
    Lists the files available in a particular S3 directory.
    :param s3_key: The path to list, without the bucket.
    :param s3_bucket: The bucket to list.
    :param filter_suffix: Optional suffix to filter S3 filenames with.
    :return: The s3 keys of files in the folder.
    """
    result: List[Path] = asyncio.run(list_files_in_s3_directory_async(s3_key, s3_bucket, filter_suffix))
    return result

def check_s3_object_exists(s3_key: Path, s3_bucket: str) -> bool:
    """
    Checks if an object in S3 exists.
    Returns False if the path is to a directory.
    :param s3_key: The path to list, without the bucket.
    :param s3_bucket: The bucket to list.
    :return: True if the object exists, false otherwise.
    """
    result: bool = asyncio.run(check_s3_object_exists_async(s3_key, s3_bucket))
    return result

def get_cache_metadata_paths(s3_key: Path, s3_bucket: str, metadata_folder: str='metadata', filter_suffix: str='csv') -> List[str]:
    """
    Find metadata file paths in S3 cache path provided.
    :param s3_key: The path of cache outputs.
    :param s3_bucket: The bucket of cache outputs.
    :param metadata_folder: Metadata folder name.
    :param filter_suffix: Optional suffix to filter S3 filenames with.
    :return: List of S3 filenames discovered.
    """
    result: List[str] = asyncio.run(get_cache_metadata_paths_async(s3_key, s3_bucket, metadata_folder, filter_suffix))
    return result

@retry(RETRYABLE_EXCEPTIONS, backoff=1, tries=3, delay=0.5)
def upload_file_to_s3(local_path: Path, s3_key: Path, s3_bucket: str) -> None:
    """
    Uploads a file from the local disk to S3.
    :param local_path: The local path to the file.
    :param s3_key: The S3 path for the file, without the bucket.
    :param s3_bucket: The name of the bucket to write to.
    """
    asyncio.run(upload_file_to_s3_async(local_path, s3_key, s3_bucket))

def read_text_file_contents_from_s3(s3_key: Path, s3_bucket: str) -> str:
    """
    Reads the entire contents of a text file from S3.
    :param s3_key: The path pointing to the file, without the bucket.
    :param s3_bucket: The name of the bucket.
    :return: The contents of the file, decoded as a UTF-8 string.
    """
    result: str = asyncio.run(read_text_file_contents_from_s3_async(s3_key, s3_bucket))
    return result

def read_binary_file_contents_from_s3(s3_key: Path, s3_bucket: str) -> bytes:
    """
    Reads the entire contents of a file from S3.
    :param s3_key: The path pointing to the file, without the bucket.
    :param s3_bucket: The name of the bucket.
    :return: The contents of the file.
    """
    result: bytes = asyncio.run(read_binary_file_contents_from_s3_async(s3_key, s3_bucket))
    return result

def download_file_from_s3(local_path: Path, s3_key: Path, s3_bucket: str) -> None:
    """
    Downloads a file to local disk from S3.
    :param local_path: The path to which to download.
    :param s3_key: The S3 path from which to download, without the bucket.
    :param s3_bucket: The bucket name to use.
    """
    asyncio.run(download_file_from_s3_async(local_path, s3_key, s3_bucket))

@retry(RETRYABLE_EXCEPTIONS, backoff=1, tries=3, delay=0.5)
def delete_file_from_s3(s3_key: Path, s3_bucket: str) -> None:
    """
    Deletes a single file from S3.
    :param s3_key: The path pointing to the file, without the bucket.
    :param s3_bucket: The name of the bucket.
    """
    asyncio.run(delete_file_from_s3_async(s3_key, s3_bucket))

def set_mock_object_from_aws(s3_key: Path, s3_bucket: str) -> None:
    """
    Retrieve an object from real S3 and upload it to mock S3.
    :param s3_key: The S3 key to retrieve and store.
    :param s3_bucket: The S3 bucket to retrieve from and store to. Created if it doesn't exist.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        dump_file = Path(tmp_dir) / f'{str(uuid.uuid4())}.dat'
        download_file_from_s3(dump_file, s3_key, s3_bucket)
        with mock_async_s3():
            _ = get_async_s3_session(force_new=True)
            asyncio.run(create_mock_bucket(s3_bucket))
            upload_file_to_s3(dump_file, s3_key, s3_bucket)

def worker_map(worker: WorkerPool, fn: Callable[..., List[Any]], input_objects: List[Any]) -> List[Any]:
    """
    Map a list of objects through a worker.
    :param worker: Worker pool to use for parallelization.
    :param fn: Function to use when mapping.
    :param input_objects: List of objects to map.
    :return: List of mapped objects.
    """
    if worker.number_of_threads == 0:
        return fn(input_objects)
    object_chunks = chunk_list(input_objects, worker.number_of_threads)
    scattered_objects = worker.map(Task(fn=fn), object_chunks)
    output_objects = [result for results in scattered_objects for result in results]
    return output_objects

class TestChunkSplitter(unittest.TestCase):
    """Unittest class for splitters to chunks"""

    def validate_chunks(self, chunks: List[List[Any]]) -> None:
        """Validate splitter chunks."""
        self.assertTrue(all([len(chunk) > 0 for chunk in chunks]))

    def test_chunk_splitter_more_data_than_number_of_chunks(self) -> None:
        """Test Chunk splitter where"""
        num_variables = 108
        num_chunks = 32
        data = list(range(1, num_variables + 1))
        chunks = chunk_list(data, num_chunks)
        self.validate_chunks(chunks)
        self.assertEqual(len(chunks), num_chunks)
        self.assertLessEqual(max(np.abs(np.diff([len(chunk) for chunk in chunks]))), 1)

    def test_chunk_splitter(self) -> None:
        """Test Chunk splitter where data size is smaller than number of chunks"""
        num_variables = 20
        num_chunks = 32
        data = list(range(1, num_variables + 1))
        chunks = chunk_list(data, num_chunks)
        self.validate_chunks(chunks)
        self.assertLessEqual(max(np.abs(np.diff([len(chunk) for chunk in chunks]))), 1)

    def test_chunk_splitter_same_size(self) -> None:
        """Test Chunk splitter where data and number chunks is the same"""
        num_chunks = 32
        num_variables = num_chunks
        data = list(range(1, num_variables + 1))
        chunks = chunk_list(data, num_chunks)
        self.validate_chunks(chunks)
        self.assertTrue(all([len(chunk) == 1 for chunk in chunks]))

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

def read_cache_metadata(cache_path: Path, metadata_filenames: List[str], worker: WorkerPool) -> List[CacheMetadataEntry]:
    """
    Reads csv file path into list of CacheMetadataEntry.
    :param cache_path: Path to s3 cache.
    :param metadata_filenames: Filenames of the metadata csv files.
    :return: List of CacheMetadataEntry.
    """
    parallel_inputs = [ReadMetadataFromS3Input(cache_path=cache_path, metadata_filename=mf) for mf in metadata_filenames]
    result = worker_map(worker, _read_metadata_from_s3, parallel_inputs)
    return cast(List[CacheMetadataEntry], result)

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

def get_s3_scenario_cache(cache_path: str, feature_names: Set[str], worker: WorkerPool) -> List[Path]:
    """
    Get a list of cached scenario paths from a remote (S3) cache.
    :param cache_path: Root path of the remote cache dir.
    :param feature_names: Set of required feature names to check when loading scenario paths from the cache.
    :return: List of discovered cached scenario paths.
    """
    assert check_s3_path_exists(cache_path), 'Remote cache {cache_path} does not exist!'
    s3_bucket, s3_key = split_s3_path(cache_path)
    metadata_files = get_cache_metadata_paths(s3_key, s3_bucket)
    if len(metadata_files) > 0:
        logger.info('Reading s3 directory from metadata.')
        cache_metadata_entries = read_cache_metadata(Path(cache_path), metadata_files, worker)
        s3_filenames = extract_field_from_cache_metadata_entries(cache_metadata_entries, 'file_name')
    else:
        logger.warning('Not using metadata! This will be slow...')
        s3_filenames = expand_s3_dir(cache_path)
    assert len(s3_filenames) > 0, f'No files found in the remote cache {cache_path}!'
    cache_map: Dict[str, Dict[str, Dict[str, Set[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for s3_filename in s3_filenames:
        path = Path(s3_filename)
        cache_map[path.parent.parent.parent.name][path.parent.parent.name][path.parent.name].add(path.stem)
    scenario_cache_paths = [Path(f'{log_name}/{scenario_type}/{scenario_token}') for log_name, scenario_types in cache_map.items() for scenario_type, scenarios in scenario_types.items() for scenario_token, features in scenarios.items() if not feature_names - features]
    return scenario_cache_paths

def extract_field_from_cache_metadata_entries(cache_metadata_entries: List[CacheMetadataEntry], desired_attribute: str) -> List[Any]:
    """
    Extracts specified field from cache metadata entries.
    :param cache_metadata_entries: List of CacheMetadataEntry
    :return: List of desired attributes in each CacheMetadataEntry
    """
    metadata = [getattr(entry, desired_attribute) for entry in cache_metadata_entries]
    return metadata

def extract_scenarios_from_cache(cfg: DictConfig, worker: WorkerPool, model: TorchModuleWrapper) -> List[AbstractScenario]:
    """
    Build the scenario objects that comprise the training dataset from cache.
    :param cfg: Omegaconf dictionary.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :param model: NN model used for training.
    :return: List of extracted scenarios.
    """
    cache_path = str(cfg.cache.cache_path)
    feature_builders = model.get_list_of_required_feature()
    target_builders = model.get_list_of_computed_target()
    feature_names = {builder.get_feature_unique_name() for builder in feature_builders + target_builders}
    scenario_cache_paths = get_s3_scenario_cache(cache_path, feature_names, worker) if cache_path.startswith('s3://') else get_local_scenario_cache(cache_path, feature_names)

    def filter_scenario_cache_paths_by_scenario_type(paths: List[Path]) -> List[Path]:
        """
        Filter the scenario cache paths by scenario type.
        :param paths: Scenario cache paths
        :return: Scenario cache paths filtered by desired scenario types
        """
        scenario_types_to_include = cfg.scenario_filter.scenario_types
        filtered_scenario_cache_paths = [path for path in paths if path.parent.name in scenario_types_to_include]
        return filtered_scenario_cache_paths
    if cfg.scenario_filter.scenario_types:
        validate_scenario_type_in_cache_path(scenario_cache_paths)
        logger.info('Filtering by desired scenario types')
        scenario_cache_paths = worker_map(worker, filter_scenario_cache_paths_by_scenario_type, scenario_cache_paths)
        assert len(scenario_cache_paths) > 0, f'Zero scenario cache paths after filtering by desired scenario types: {cfg.scenario_filter.scenario_types}. Please check if the cache contains the desired scenario type.'
    scenarios = worker_map(worker, create_scenario_from_paths, scenario_cache_paths)
    return cast(List[AbstractScenario], scenarios)

def get_local_scenario_cache(cache_path: str, feature_names: Set[str]) -> List[Path]:
    """
    Get a list of cached scenario paths from a local cache.
    :param cache_path: Root path of the local cache dir.
    :param feature_names: Set of required feature names to check when loading scenario paths from the cache.
    :return: List of discovered cached scenario paths.
    """
    cache_dir = Path(cache_path)
    assert cache_dir.exists(), f'Local cache {cache_dir} does not exist!'
    assert any(cache_dir.iterdir()), f'No files found in the local cache {cache_dir}!'
    candidate_scenario_dirs = {x.parent for x in cache_dir.rglob('*.gz')}
    scenario_cache_paths = [path for path in candidate_scenario_dirs if not feature_names - {feature_name.stem for feature_name in path.iterdir()}]
    return scenario_cache_paths

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

class TestCache(SkeletonTestTrain):
    """
    Test main training entry point using combinations of models, datasets, filters etc.
    """

    def setUp(self) -> None:
        """
        Set up test attributes.
        """
        super().setUp()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.cache_path = f'{self.tmp_dir.name}/cache'
        self.test_args = ['+training=training_raster_model', 'scenario_builder=mock_abstract_scenario_builder', f'group={self.tmp_dir.name}', f'cache.cache_path={self.cache_path}']

    def tearDown(self) -> None:
        """
        Cleanup after each test.
        """
        self.tmp_dir.cleanup()

    @patch('nuplan.planning.training.modeling.models.raster_model.RasterModel.get_list_of_required_feature')
    @patch('nuplan.planning.training.modeling.models.raster_model.RasterModel.get_list_of_computed_target')
    def test_cache_dataset(self, feature_builders_fn: Mock, target_builders_fn: Mock) -> None:
        """
        Tests dataset caching.
        """
        feature_builders_fn.return_value = [MockFeatureBuilder(torch.Tensor([0.0]))]
        target_builders_fn.return_value = [MockFeatureBuilder(torch.Tensor([0.0]))]
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, *self.test_args, 'py_func=cache'])
            main(cfg)
        all_feature_builders = feature_builders_fn.return_value + target_builders_fn.return_value
        all_feature_names = {builder.get_feature_unique_name() for builder in all_feature_builders}
        scenario_cache_paths = get_local_scenario_cache(self.cache_path, all_feature_names)
        self.assertTrue(len(scenario_cache_paths) == cfg.scenario_builder.num_scenarios)

class TestCache(SkeletonTestTrain):
    """
    Test main training entry point using combinations of models, datasets, filters etc.
    """

    def setUp(self) -> None:
        """
        Set up test attributes.
        """
        super().setUp()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.local_cache_path = f'{self.tmp_dir.name}/cache'
        self.s3_cache_path = 's3://test-bucket/nuplan_tests/test_cache_nuplandb'
        self.test_args = ['+training=training_raster_model', 'scenario_builder=nuplan_mini', 'splitter=nuplan', f'group={self.tmp_dir.name}']

    def tearDown(self) -> None:
        """
        Cleanup after each test.
        """
        self.tmp_dir.cleanup()

    @unittest.skip('Skip in CI until issue is resolved')
    def test_cache_dataset_s3(self) -> None:
        """
        Tests dataset caching with mocked S3.
        """
        s3_bucket, s3_key = split_s3_path(self.s3_cache_path)
        set_mock_object_from_aws(Path('nuplan-v1.1/maps/us-pa-pittsburgh-hazelwood/9.17.1937/map.gpkg'), 'nuplan-production')
        with mock_async_s3():
            asyncio.run(create_mock_bucket(s3_bucket))
            with initialize_config_dir(config_dir=self.config_path):
                cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, *self.test_args, 'scenario_filter.limit_total_scenarios=10', 'py_func=cache', f'cache.cache_path={self.s3_cache_path}', 'cache.force_feature_computation=True'])
                main(cfg)
            self.assertTrue(len(list_files_in_s3_directory(s3_key, s3_bucket)) > 0)
            with initialize_config_dir(config_dir=self.config_path):
                cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, *self.test_args, 'py_func=train', 'scenario_filter.limit_total_scenarios=10', 'cache.cleanup_cache=false', 'cache.use_cache_without_dataset=true', f'cache.cache_path={self.s3_cache_path}'])
                main(cfg)
            with initialize_config_dir(config_dir=self.config_path):
                cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, *self.test_args, 'py_func=train', 'scenario_filter.limit_total_scenarios=10', 'cache.cleanup_cache=false', 'cache.use_cache_without_dataset=false', f'cache.cache_path={self.s3_cache_path}'])
                main(cfg)

    def test_cache_dataset_local(self) -> None:
        """
        Tests local dataset caching.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, *self.test_args, 'py_func=cache', f'cache.cache_path={self.local_cache_path}'])
            main(cfg)
        self.assertTrue(any(Path(self.local_cache_path).iterdir()))
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, *self.test_args, 'py_func=train', 'cache.cleanup_cache=false', 'cache.use_cache_without_dataset=true', f'cache.cache_path={self.local_cache_path}'])
            main(cfg)
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, *self.test_args, 'py_func=train', 'cache.cleanup_cache=false', 'cache.use_cache_without_dataset=false', f'cache.cache_path={self.local_cache_path}'])
            main(cfg)

    def test_profiling(self) -> None:
        """Test that profiling gets generated."""
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, *self.test_args, 'py_func=cache', 'enable_profiling=True', f'cache.cache_path={self.local_cache_path}'])
            main(cfg)
        self.assertTrue(Path(self.local_cache_path).rglob('caching.html'))

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

def absolute_path_to_log_name(absolute_path: str) -> str:
    """
    Gets the log name from the absolute path to a log file.
    E.g.
        input: data/sets/nuplan/nuplan-v1.1/splits/mini/2021.10.11.02.57.41_veh-50_01522_02088.db
        output: 2021.10.11.02.57.41_veh-50_01522_02088

        input: /tmp/abcdef
        output: abcdef
    :param absolute_path: The absolute path to a log file.
    :return: The log name.
    """
    filename = os.path.basename(absolute_path)
    if filename.endswith('.db'):
        filename = os.path.splitext(filename)[0]
    return filename

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

def filter_invalid_goals(scenario_dict: ScenarioDict, worker: WorkerPool) -> ScenarioDict:
    """
    Filter the scenarios with invalid mission goals in a scenario dictionary.
    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type.
    :param worker: Worker pool for concurrent scenario processing.
    :return: Filtered scenario dictionary.
    """

    def _filter_goals(scenarios: List[NuPlanScenario]) -> List[NuPlanScenario]:
        """
        Filter scenarios that contain invalid mission goals.
        :param scenarios: List of scenarios to filter.
        :return: List of filtered scenarios.
        """
        return [scenario for scenario in scenarios if scenario.get_mission_goal()]
    for scenario_type in scenario_dict:
        scenario_dict[scenario_type] = worker_map(worker, _filter_goals, scenario_dict[scenario_type])
    return scenario_dict

