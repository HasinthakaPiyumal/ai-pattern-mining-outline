# Cluster 54

def is_s3_path(candidate: Union[Path, str]) -> bool:
    """
    Returns true if the path points to a location in S3, false otherwise.
    :param candidate: The candidate path.
    :return: True if the path points to a location in S3, false otherwise.
    """
    candidate_str = str(candidate)
    return candidate_str.startswith('s3:/')

def safe_path_to_string(path: Union[Path, str]) -> str:
    """
    Converts local/s3 paths from Path objects to string.
    It's not always safe to pass the path object to certain io functions.
    For example,
        pd.read_csv(Path("s3://foo/bar"))
    gets interpreted like
        pd.read_csv("s3:/foo/bar")  -- should be s3://, not s3:/
    which is not recognized as an s3 path and raises and error. This function takes a path
    and returns a string that can be passed to any of these functions.
    :param s3_path: Path object of path
    :return: path with the correct format as a string.
    """
    if is_s3_path(path):
        return f's3://{str(path).lstrip('s3:/')}'
    return str(path)

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

@functools.cache
def get_unique_job_id() -> str:
    """
    In the cluster, it generates a hash from the unique job ID called NUPLAN_JOB_ID.
    Locally, it generates a hash from a UUID.

    Note that the returned value is cached as soon as the function is called the first time.
    After that, it is going to return always the same value.
    If a new value is needed, use get_unique_job_id.cache_clear() first.
    """
    global_job_id_str = os.environ.get('NUPLAN_JOB_ID', str(uuid.uuid4())).encode('utf-8')
    return hashlib.sha256(global_job_id_str).hexdigest()

def distributed_sync(path: Union[Path, str], timeout_seconds: int=7200, poll_interval: float=0.5) -> None:
    """
    Use a FileBackendBarrier at "path" to sync across multiple workers
    (Note that it deletes the path after the sync is done to allow the same path to be reused)
    :param path: path to use for distributed sync (must be shared across workers)
    :param timeout_seconds: how long to wait for nodes to sync
    :param poll_interval: how long to sleep between poll times
    """
    if int(os.environ.get('NUM_NODES', 1)) > 1:
        barrier = FileBackedBarrier(Path(path))
        barrier.wait_barrier(activity_id='barrier_token_' + str(os.environ.get('NODE_RANK', 0)), expected_activity_ids={'barrier_token_' + str(el) for el in range(0, int(os.environ.get('NUM_NODES', 1)))}, timeout_s=timeout_seconds, poll_interval_s=poll_interval)

class TestIoUtils(unittest.TestCase):
    """
    A class to test that the I/O utilities in nuplan_devkit function properly.
    """

    def test_nupath(self) -> None:
        """
        Tests that converting NuPath to strings works properly.
        """
        example_s3_path = NuPath('s3://test-bucket/foo/bar/baz.txt')
        expected_s3_str = 's3://test-bucket/foo/bar/baz.txt'
        actual_s3_str = str(example_s3_path)
        self.assertEqual(expected_s3_str, actual_s3_str)
        example_local_path = NuPath('/foo/bar/baz')
        expected_local_str = '/foo/bar/baz'
        actual_local_str = str(example_local_path)
        self.assertEqual(expected_local_str, actual_local_str)

    def test_safe_path_to_string(self) -> None:
        """
        Tests that converting paths to strings safely works properly.
        """
        example_s3_path = Path('s3://test-bucket/foo/bar/baz.txt')
        expected_s3_str = 's3://test-bucket/foo/bar/baz.txt'
        actual_s3_str = safe_path_to_string(example_s3_path)
        self.assertEqual(expected_s3_str, actual_s3_str)
        example_local_path = Path('/foo/bar/baz')
        expected_local_str = '/foo/bar/baz'
        actual_local_str = safe_path_to_string(example_local_path)
        self.assertEqual(expected_local_str, actual_local_str)
        example_s3_str_path = 's3://test-bucket/foo/bar/baz.txt'
        expected_s3_str = 's3://test-bucket/foo/bar/baz.txt'
        actual_s3_str = safe_path_to_string(example_s3_str_path)
        self.assertEqual(expected_s3_str, actual_s3_str)
        example_local_str_path = '/foo/bar/baz'
        expected_local_str = '/foo/bar/baz'
        actual_local_str = safe_path_to_string(example_local_str_path)
        self.assertEqual(expected_local_str, actual_local_str)

    def test_save_buffer_locally(self) -> None:
        """
        Tests that saving a buffer locally works properly.
        """
        expected_buffer = b'test'
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / 'local_buffer.bin'
            save_buffer(output_file, expected_buffer)
            with open(output_file, 'rb') as f:
                reconstructed_buffer = f.read()
            self.assertEqual(expected_buffer, reconstructed_buffer)

    def test_save_buffer_s3(self) -> None:
        """
        Tests that saving a buffer to s3 works properly.
        """
        upload_bucket_name = 'ml-caches'
        upload_path = Path('foo/bar/baz.bin')
        uploaded_file_contents: Optional[bytes] = None

        async def patch_upload_file_to_s3_async(local_path: Path, s3_key: Path, s3_bucket: str) -> None:
            """
            Patch for upload_file_to_s3_async method.
            :param local_path: The passed local_path.
            :param s3_key: The passed s3_key.
            :param s3_bucket: The passed s3_bucket.
            """
            nonlocal uploaded_file_contents
            self.assertEqual(upload_bucket_name, s3_bucket)
            self.assertEqual(upload_path, s3_key)
            with open(local_path, 'rb') as f:
                uploaded_file_contents = f.read()
        expected_buffer = b'test'
        with patch_with_validation('nuplan.common.utils.io_utils.upload_file_to_s3_async', patch_upload_file_to_s3_async):
            output_file = Path(f's3://{upload_bucket_name}') / f'{upload_path}'
            save_buffer(output_file, expected_buffer)
            self.assertIsNotNone(uploaded_file_contents)
            assert uploaded_file_contents is not None
            self.assertEqual(expected_buffer, uploaded_file_contents)

    def test_save_object_as_pickle_locally(self) -> None:
        """
        Tests that saving a pickled object locally works properly.
        """
        expected_object = {'a': 1, 'b': 2}
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / 'local.pkl'
            save_object_as_pickle(output_file, expected_object)
            with open(output_file, 'rb') as f:
                reconstructed_object = pickle.load(f)
            self.assertEqual(expected_object, reconstructed_object)

    def test_save_object_as_pickle_s3(self) -> None:
        """
        Tests that saving a pickled object to s3 works properly.
        """
        upload_bucket_name = 'ml-caches'
        upload_path = Path('foo/bar/baz.pkl')
        uploaded_file_contents: Optional[bytes] = None

        async def patch_upload_file_to_s3_async(local_path: Path, s3_key: Path, s3_bucket: str) -> None:
            """
            Patch for upload_file_to_s3_async method.
            :param local_path: The passed local_path.
            :param s3_key: The passed s3_key.
            :param s3_bucket: The passed s3_bucket.
            """
            nonlocal uploaded_file_contents
            self.assertEqual(upload_bucket_name, s3_bucket)
            self.assertEqual(upload_path, s3_key)
            with open(local_path, 'rb') as f:
                uploaded_file_contents = f.read()
        expected_object = {'a': 1, 'b': 2}
        with patch_with_validation('nuplan.common.utils.io_utils.upload_file_to_s3_async', patch_upload_file_to_s3_async):
            output_file = Path(f's3://{upload_bucket_name}') / f'{upload_path}'
            save_object_as_pickle(output_file, expected_object)
            self.assertIsNotNone(uploaded_file_contents)
            assert uploaded_file_contents is not None
            reconstructed_object: Dict[str, int] = pickle.loads(uploaded_file_contents)
            self.assertEqual(expected_object, reconstructed_object)

    def test_save_text_locally(self) -> None:
        """
        Tests that saving a text file locally works properly.
        """
        expected_text = 'test_save_text_locally.'
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / 'local.txt'
            save_text(output_file, expected_text)
            with open(output_file, 'r') as f:
                reconstructed_text = f.read()
            self.assertEqual(expected_text, reconstructed_text)

    def test_save_text_s3(self) -> None:
        """
        Tests that saving a text file to s3 works properly.
        """
        upload_bucket_name = 'ml-caches'
        upload_path = Path('foo/bar/baz.pkl')
        uploaded_file_contents: Optional[str] = None

        async def patch_upload_file_to_s3_async(local_path: Path, s3_key: Path, s3_bucket: str) -> None:
            """
            Patch for upload_file_to_s3_async method.
            :param local_path: The passed local_path.
            :param s3_key: The passed s3_key.
            :param s3_bucket: The passed s3_bucket.
            """
            nonlocal uploaded_file_contents
            self.assertEqual(upload_bucket_name, s3_bucket)
            self.assertEqual(upload_path, s3_key)
            with open(local_path, 'r') as f:
                uploaded_file_contents = f.read()
        expected_text = 'test_save_text_s3.'
        with patch_with_validation('nuplan.common.utils.io_utils.upload_file_to_s3_async', patch_upload_file_to_s3_async):
            output_file = Path(f's3://{upload_bucket_name}') / f'{upload_path}'
            save_text(output_file, expected_text)
            self.assertIsNotNone(uploaded_file_contents)
            self.assertEqual(expected_text, uploaded_file_contents)

    def test_read_text_locally(self) -> None:
        """
        Tests that reading a text file locally works properly.
        """
        expected_text = 'some expected text.'
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / 'read_text_locally.txt'
            with open(output_file, 'w') as f:
                f.write(expected_text)
            reconstructed_text = read_text(output_file)
            self.assertEqual(expected_text, reconstructed_text)

    def test_read_text_from_s3(self) -> None:
        """
        Tests that reading a text file from S3 works properly.
        """
        download_bucket = 'ml-caches'
        download_key = 'my/file/path.txt'
        expected_text = 'some expected text.'
        full_filepath = Path(f's3://{download_bucket}') / download_key

        async def patch_read_binary_file_contents_from_s3_async(s3_key: Path, s3_bucket: str) -> bytes:
            """
            A patch for the read_binary_file_contents_from_s3_async method.
            :param s3_key: The passed key
            :param s3_bucket: The passed bucket.
            """
            self.assertEqual(Path(download_key), s3_key)
            self.assertEqual(download_bucket, s3_bucket)
            return expected_text.encode('utf-8')
        with patch_with_validation('nuplan.common.utils.io_utils.read_binary_file_contents_from_s3_async', patch_read_binary_file_contents_from_s3_async):
            reconstructed_text = read_text(full_filepath)
            self.assertEqual(expected_text, reconstructed_text)

    def test_read_pickle_locally(self) -> None:
        """
        Tests that reading a pickle file locally works properly.
        """
        expected_obj = {'foo': 'bar'}
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / 'read_text_locally.txt'
            with open(output_file, 'wb') as f:
                f.write(pickle.dumps(expected_obj))
            reconstructed_obj = read_pickle(output_file)
            self.assertEqual(expected_obj, reconstructed_obj)

    def test_read_pickle_from_s3(self) -> None:
        """
        Tests that reading a pickle file from S3 works properly.
        """
        download_bucket = 'ml-caches'
        download_key = 'my/file/path.txt'
        expected_obj = {'foo': 'bar'}
        full_filepath = Path(f's3://{download_bucket}') / download_key

        async def patch_read_binary_file_contents_from_s3_async(s3_key: Path, s3_bucket: str) -> bytes:
            """
            A patch for the read_binary_file_contents_from_s3_async method.
            :param s3_key: The passed key
            :param s3_bucket: The passed bucket.
            """
            self.assertEqual(Path(download_key), s3_key)
            self.assertEqual(download_bucket, s3_bucket)
            return pickle.dumps(expected_obj)
        with patch_with_validation('nuplan.common.utils.io_utils.read_binary_file_contents_from_s3_async', patch_read_binary_file_contents_from_s3_async):
            reconstructed_obj = read_pickle(full_filepath)
            self.assertEqual(expected_obj, reconstructed_obj)

    def test_read_binary_locally(self) -> None:
        """
        Tests that reading a binary file locally works properly.
        """
        expected_data = bytes([1, 2, 3, 4, 5])
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / 'read_text_locally.txt'
            with open(output_file, 'wb') as f:
                f.write(expected_data)
            reconstructed_data = read_binary(output_file)
            self.assertEqual(expected_data, reconstructed_data)

    def test_read_binary_from_s3(self) -> None:
        """
        Tests that reading a binary file from S3 works properly.
        """
        download_bucket = 'ml-caches'
        download_key = 'my/file/path.data'
        expected_data = bytes([1, 2, 3, 4, 5])
        full_filepath = Path(f's3://{download_bucket}') / download_key

        async def patch_read_binary_file_contents_from_s3_async(s3_key: Path, s3_bucket: str) -> bytes:
            """
            A patch for the read_binary_file_contents_from_s3_async method.
            :param s3_key: The passed key
            :param s3_bucket: The passed bucket.
            """
            self.assertEqual(Path(download_key), s3_key)
            self.assertEqual(download_bucket, s3_bucket)
            return expected_data
        with patch_with_validation('nuplan.common.utils.io_utils.read_binary_file_contents_from_s3_async', patch_read_binary_file_contents_from_s3_async):
            reconstructed_data = read_binary(full_filepath)
            self.assertEqual(expected_data, reconstructed_data)

    def test_path_exists_locally(self) -> None:
        """
        Tests that path_exists works for local files.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            file_to_create = tmp_dir_path / 'existing.txt'
            file_to_not_create = tmp_dir_path / 'not_existing.txt'
            with open(file_to_create, 'w') as f:
                f.write('some irrelevant text.')
            self.assertTrue(path_exists(file_to_create))
            self.assertFalse(path_exists(file_to_not_create))
            self.assertTrue(path_exists(tmp_dir_path, include_directories=True))
            self.assertFalse(path_exists(tmp_dir_path, include_directories=False))

    def test_path_exists_s3(self) -> None:
        """
        Tests that path_exists works for s3 files.
        """
        test_bucket = 'ml-caches'
        test_parent_dir = 'my/file/that'
        test_existing_file = f'{test_parent_dir}/exists.txt'
        test_non_existing_file = f'{test_parent_dir}/does_not_exist.txt'
        test_dir_path = Path(f's3://{test_bucket}') / test_parent_dir
        test_existing_path = Path(f's3://{test_bucket}') / test_existing_file
        test_non_existing_path = Path(f's3://{test_bucket}') / test_non_existing_file

        async def patch_check_s3_object_exists_async(s3_key: Path, s3_bucket: str) -> bool:
            """
            Patches the check_s3_object_exists_async method.
            :param key: The s3 key to check.
            :param bucket: The s3 bucket to check.
            :return: The mocked return value.
            """
            self.assertEqual(test_bucket, s3_bucket)
            if str(s3_key) == test_existing_file:
                return True
            elif str(s3_key) in [test_non_existing_file, test_parent_dir]:
                return False
            self.fail(f'Unexpected path passed to check_s3_object_exists patch: {s3_key}')

        async def patch_check_s3_path_exists_async(s3_path: str) -> bool:
            """
            Patches the check_s3_object_exists_async method.
            :param s3_path: The s3 path to check.
            :return: The mocked return value.
            """
            if s3_path in [safe_path_to_string(test_existing_path), safe_path_to_string(test_dir_path)]:
                return True
            elif s3_path == safe_path_to_string(test_non_existing_path):
                return False
            self.fail(f'Unexpected path passed to check_s3_path_exists patch: {s3_path}')
        with patch_with_validation('nuplan.common.utils.io_utils.check_s3_object_exists_async', patch_check_s3_object_exists_async), patch_with_validation('nuplan.common.utils.io_utils.check_s3_path_exists_async', patch_check_s3_path_exists_async):
            self.assertTrue(path_exists(test_existing_path))
            self.assertFalse(path_exists(test_non_existing_path))
            self.assertTrue(path_exists(test_dir_path, include_directories=True))
            self.assertFalse(path_exists(test_dir_path, include_directories=False))

    def test_list_files_in_directory_locally(self) -> None:
        """
        Tests that list_files_in_directory works for local files.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            self.assertEqual(list_files_in_directory(tmp_dir_path), [])
            test_file_contents = {'a.txt': 'test file a.', 'b.txt': 'test file b.'}
            for filename, contents in test_file_contents.items():
                with open(tmp_dir_path / filename, 'w') as f:
                    f.write(contents)
            output_files_in_directory = list_files_in_directory(tmp_dir_path)
            self.assertEqual(len(output_files_in_directory), len(test_file_contents))
            for output_filepath in output_files_in_directory:
                self.assertIn(output_filepath.name, test_file_contents)

    def test_list_files_in_directory_s3(self) -> None:
        """
        Tests that list_files_in_directory works for s3.
        """
        test_bucket = 'ml-caches'
        test_directory_key = Path('test_dir')
        test_directory_s3_path = Path(f's3://{test_bucket}/{test_directory_key}')
        test_files_in_s3 = ['a.txt', 'b.txt']
        expected_files = [Path(f'{test_directory_key}/{filename}') for filename in test_files_in_s3]
        expected_s3_paths = [Path(f's3://{test_bucket}') / filename for filename in expected_files]

        async def patch_list_files_in_s3_directory_async(s3_key: Path, s3_bucket: str, filter_suffix: str='') -> List[Path]:
            """
            Patches the list_files_in_s3_directory_async method.
            :param key: The s3 key of the directory.
            :param bucket: The s3 bucket of the directory.
            :param filter_suffix: Unused.
            :return: The mocked return value.
            """
            self.assertEqual(test_bucket, s3_bucket)
            self.assertEqual(test_directory_key, s3_key)
            return expected_files
        with patch_with_validation('nuplan.common.utils.io_utils.list_files_in_s3_directory_async', patch_list_files_in_s3_directory_async):
            output_filepaths = list_files_in_directory(test_directory_s3_path)
            self.assertEqual(output_filepaths, expected_s3_paths)

    def test_delete_file_locally(self) -> None:
        """
        Tests that delete_file works for local files.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            test_file_contents = {'a.txt': 'test file a.', 'b.txt': 'test file b.'}
            test_file_paths = [tmp_dir_path / filename for filename in test_file_contents]
            for filename, contents in test_file_contents.items():
                with open(tmp_dir_path / filename, 'w') as f:
                    f.write(contents)
            self.assertEqual(set(tmp_dir_path.iterdir()), set(test_file_paths))
            for filename in test_file_contents:
                filepath = tmp_dir_path / filename
                delete_file(filepath)
                self.assertNotIn(filepath, tmp_dir_path.iterdir())
            self.assertEqual(len(list(tmp_dir_path.iterdir())), 0)
            with self.assertRaises(ValueError):
                delete_file(tmp_dir_path)

    def test_delete_file_s3(self) -> None:
        """
        Tests that delete_file works for s3.
        """
        test_bucket = 'ml-caches'
        test_directory_key = Path('test_dir')
        test_directory_s3_path = Path(f's3://{test_bucket}/{test_directory_key}')
        test_files_in_s3 = {'a.txt', 'b.txt'}

        def get_s3_key(filename: str) -> Path:
            """
            Turns a filename into an s3 key.
            """
            return Path(f'{test_directory_key}/{filename}')

        def list_s3_keys() -> List[Path]:
            """
            Lists the keys in s3.
            :return: S3 keys in the mocked test directory.
            """
            return [get_s3_key(filename) for filename in test_files_in_s3]

        async def patch_list_files_in_s3_directory_async(s3_key: Path, s3_bucket: str, filter_suffix: str='') -> List[Path]:
            """
            Patches the list_files_in_s3_directory_async method.
            :param key: The s3 key of the directory.
            :param bucket: The s3 bucket of the directory.
            :param filter_suffix: Unused.
            :return: The mocked return value.
            """
            self.assertEqual(test_bucket, s3_bucket)
            self.assertEqual(test_directory_key, s3_key)
            return list_s3_keys()

        async def patch_delete_file_from_s3_async(s3_key: Path, s3_bucket: str) -> None:
            """
            Patches the delete_file_from_s3_async method.
            :param s3_key: The s3 key to delete.
            :param s3_bucket: The s3 bucket.
            """
            nonlocal test_files_in_s3
            self.assertEqual(test_bucket, s3_bucket)
            self.assertEqual(test_directory_key, s3_key.parent)
            self.assertIn(s3_key.name, test_files_in_s3)
            test_files_in_s3.remove(s3_key.name)
        with patch_with_validation('nuplan.common.utils.io_utils.list_files_in_s3_directory_async', patch_list_files_in_s3_directory_async), patch_with_validation('nuplan.common.utils.io_utils.delete_file_from_s3_async', patch_delete_file_from_s3_async):
            initial_s3_keys = list_s3_keys()
            for filename in test_files_in_s3:
                self.assertIn(get_s3_key(filename), initial_s3_keys)
            for filename in set(test_files_in_s3):
                s3_path = test_directory_s3_path / filename
                delete_file(s3_path)
                self.assertNotIn(get_s3_key(filename), list_s3_keys())

def read_pickle(path: Path) -> Any:
    """
    Reads an object as a pickle file from the provided path.
    The path can be a local path or an S3 path.
    :param path: The path to read.
    :return: The depickled object.
    """
    return asyncio.run(read_pickle_async(path))

def path_exists(path: Path, include_directories: bool=True) -> bool:
    """
    Checks to see if a path exists.
    The path can be a local path or an S3 path.
    This method does not examine the file contents.
        That is, a file that exists and empty will return True.
    :param path: The path to check for existance.
    :param include_directories: Whether or not directories count as paths.
    :return: True if the path exists, False otherwise.
    """
    result: bool = asyncio.run(path_exists_async(path, include_directories=include_directories))
    return result

def list_files_in_directory(path: Path) -> List[Path]:
    """
    Returns a list of the string file paths in a directory.
    The path can be a local path or an S3 path.
    :param path: The path to list.
    :return: List of file paths in the folder.
    """
    result: List[Path] = asyncio.run(list_files_in_directory_async(path))
    return result

class TestDistributedSyncWrapper(unittest.TestCase):
    """
    Test the function distributed_sync that wraps the FileBackendBarrier for easier use
    """

    def test_call_with_single_node(self) -> None:
        """
        Test that we don't call wait if we are on one node
        """
        with unittest.mock.patch.object(nuplan.common.utils.file_backed_barrier.FileBackedBarrier, 'wait_barrier', MagicMock()) as mock_wait, unittest.mock.patch.dict(os.environ, {'NUM_NODES': '1'}):
            distributed_sync('')
            mock_wait.assert_not_called()

    def test_call_with_multiple_nodes(self) -> None:
        """
        Test that we call wait with the correct params if we are on multiple nodes
        """
        with unittest.mock.patch.object(nuplan.common.utils.file_backed_barrier.FileBackedBarrier, 'wait_barrier', MagicMock()) as mock_wait, unittest.mock.patch.dict(os.environ, {'NUM_NODES': '2', 'NODE_RANK': '1'}):
            distributed_sync('')
            mock_wait.assert_called_with(activity_id='barrier_token_1', expected_activity_ids={'barrier_token_1', 'barrier_token_0'}, timeout_s=7200, poll_interval_s=0.5)

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

class TestGetUniqueJobId(unittest.TestCase):
    """Test suite for the generation of unique job IDs"""

    def test_unique_job_id_no_job_id(self) -> None:
        """
        Tests that if NUPLAN_JOB_ID is not set, the same job ID is returned.
        """
        if os.environ.get('NUPLAN_JOB_ID') is not None:
            del os.environ['NUPLAN_JOB_ID']
        get_unique_job_id.cache_clear()
        job_id_1 = get_unique_job_id()
        job_id_2 = get_unique_job_id()
        self.assertEqual(job_id_1, job_id_2)

    def test_unique_job_id_with_job_id(self) -> None:
        """
        Tests that if NUPLAN_JOB_ID is not set, the same job ID is returned.
        """
        os.environ['NUPLAN_JOB_ID'] = '12345'
        get_unique_job_id.cache_clear()
        job_id_1 = get_unique_job_id()
        job_id_2 = get_unique_job_id()
        self.assertEqual(job_id_1, job_id_2)
        del os.environ['NUPLAN_JOB_ID']

    def test_uniqueness_job_id(self) -> None:
        """
        Tests that the returned job ids are unique if they are not cached.
        """
        if os.environ.get('NUPLAN_JOB_ID') is not None:
            del os.environ['NUPLAN_JOB_ID']
        job_id_1 = get_unique_job_id()
        get_unique_job_id.cache_clear()
        job_id_2 = get_unique_job_id()
        self.assertNotEqual(job_id_1, job_id_2)

    def test_get_unique_incremental_track_id(self) -> None:
        """Tests creation of unique track ids."""
        track_token_and_expected_id_0 = ('track_0', 0)
        track_token_and_expected_id_1 = ('track_1', 1)
        self.assertEqual(get_unique_incremental_track_id(track_token_and_expected_id_0[0]), track_token_and_expected_id_0[1])
        self.assertEqual(get_unique_incremental_track_id(track_token_and_expected_id_1[0]), track_token_and_expected_id_1[1])
        self.assertEqual(get_unique_incremental_track_id(track_token_and_expected_id_0[0]), track_token_and_expected_id_0[1])

class TestWorkerSquareNumbersTask(unittest.TestCase):
    """Class to run all workers on a simple worker task, squaring a list of numbers."""

    def setUp(self) -> None:
        """
        Instantiate all workers we want to check, not just numerically but for correct bazel BUILD file setup.
        """
        self.worker_arg_tuples: List[Tuple[WorkerPool, Dict[str, Any]]] = [(SingleMachineParallelExecutor, {}), (SingleMachineParallelExecutor, {'use_process_pool': True}), (RayDistributed, {'threads_per_node': 2}), (Sequential, {})]
        self.number_of_tasks = 10

    def test_square_numbers_task(self) -> None:
        """Make sure all workers can correctly execute map to square a list of numbers."""
        task_list = [x for x in range(self.number_of_tasks)]
        expected_result = [x ** 2 for x in task_list]
        for worker_arg_tuple in self.worker_arg_tuples:
            worker = worker_arg_tuple[0](**worker_arg_tuple[1])
            worker_result = worker.map(Task(fn=square_fn), task_list)
            self.assertEqual(worker_result, expected_result)
            if isinstance(worker, RayDistributed):
                worker.shutdown()

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

class CompletionCallback(AbstractMainCallback):
    """Callback that creates a token file to mark that the simulation instance finished the job."""

    def __init__(self, output_dir: str, challenge_name: str):
        """
        :param output_dir: Root dir used to find the report file and as path to save results.
        :param challenge_name: Name of the challenge being run.
        """
        self._bucket = os.getenv('NUPLAN_SERVER_S3_ROOT_URL')
        assert self._bucket, 'Target bucket must be specified!'
        instance_id = os.getenv('SCENARIO_FILTER_ID', '0')
        task_id = '_'.join([challenge_name, instance_id])
        self._completion_dir = Path(output_dir, 'simulation-results', task_id)

    def on_run_simulation_end(self) -> None:
        """
        On reached_end mark the task as completed by creating the relative file.
        """
        self._write_empty_file(self._completion_dir, 'completed.txt')

    @staticmethod
    def _write_empty_file(path: Path, filename: str) -> None:
        """
        Creates an empty file with the specified name at the given location.
        :param path: The location where to create the file.
        :param filename: The name of the file to be created.
        """
        if not is_s3_path(path):
            path.mkdir(parents=True, exist_ok=True)
        logger.info(f'Writing file {path / filename}')
        with (path / filename).open('w'):
            pass

class MetricFileCallback(AbstractMainCallback):
    """Callback to handle metric files at the end of process."""

    def __init__(self, metric_file_output_path: str, scenario_metric_paths: List[str], delete_scenario_metric_files: bool=False):
        """
        Constructor of MetricFileCallback.
        Output path can be local or s3.
        :param metric_file_output_path: Path to save integrated metric files.
        :param scenario_metric_paths: A list of paths with scenario metric files.
        :param delete_scenario_metric_files: Set True to delete scenario metric files.
        """
        self._metric_file_output_path = pathlib.Path(metric_file_output_path)
        if not is_s3_path(self._metric_file_output_path):
            self._metric_file_output_path.mkdir(exist_ok=True, parents=True)
        self._scenario_metric_paths = [pathlib.Path(scenario_metric_path) for scenario_metric_path in scenario_metric_paths]
        self._delete_scenario_metric_files = delete_scenario_metric_files

    def on_run_simulation_end(self) -> None:
        """Callback before end of the main function."""
        start_time = time.perf_counter()
        metrics = defaultdict(list)
        for scenario_metric_path in self._scenario_metric_paths:
            if not is_s3_path(scenario_metric_path) and (not path_exists(scenario_metric_path)):
                continue
            for scenario_metric_file in list_files_in_directory(scenario_metric_path):
                if not scenario_metric_file.name.endswith(JSON_FILE_EXTENSION):
                    continue
                json_dataframe = read_pickle(scenario_metric_file)
                for dataframe in json_dataframe:
                    pandas_dataframe = pandas.DataFrame(dataframe)
                    metrics[dataframe['metric_statistics_name']].append(pandas_dataframe)
                if self._delete_scenario_metric_files:
                    delete_file(scenario_metric_file)
        for metric_statistics_name, dataframe in metrics.items():
            save_path = self._metric_file_output_path / (metric_statistics_name + '.parquet')
            concat_pandas = pandas.concat([*dataframe], ignore_index=True)
            concat_pandas.to_parquet(safe_path_to_string(save_path))
        end_time = time.perf_counter()
        elapsed_time_s = end_time - start_time
        time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time_s))
        logger.info(f'Metric files integration: {time_str} [HH:MM:SS]')

class MetricSummaryCallback(AbstractMainCallback):
    """Callback to render histograms for metrics and metric aggregator."""

    def __init__(self, metric_save_path: str, metric_aggregator_save_path: str, summary_output_path: str, pdf_file_name: str, num_bins: int=20):
        """Callback to handle metric files at the end of process."""
        self._metric_save_path = Path(metric_save_path)
        self._metric_aggregator_save_path = Path(metric_aggregator_save_path)
        self._summary_output_path = Path(summary_output_path)
        if not is_s3_path(self._summary_output_path):
            self._summary_output_path.mkdir(parents=True, exist_ok=True)
        self._pdf_file_name = pdf_file_name
        self._num_bins = num_bins
        self._color_index = 0
        color_palette = cmap.get_cmap('Set1').colors + cmap.get_cmap('Set2').colors + cmap.get_cmap('Set3').colors
        self._color_choices = [mcolors.rgb2hex(color) for color in color_palette]
        self._metric_aggregator_dataframes: Dict[str, pd.DataFrame] = {}
        self._metric_statistics_dataframes: Dict[str, MetricStatisticsDataFrame] = {}

    @staticmethod
    def _read_metric_parquet_files(metric_save_path: Path, metric_reader: Callable[[Path], Any]) -> METRIC_DATAFRAME_TYPE:
        """
        Read metric parquet files with different readers.
        :param metric_save_path: Metric save path.
        :param metric_reader: Metric reader to read metric parquet files.
        :return A dictionary of {file_index: {file_name: MetricStatisticsDataFrame or pandas dataframe}}.
        """
        metric_dataframes: Dict[str, Union[MetricStatisticsDataFrame, pd.DataFrame]] = defaultdict()
        metric_file = metric_save_path.rglob('*.parquet')
        for file_index, file in enumerate(metric_file):
            try:
                if file.is_dir():
                    continue
                data_frame = metric_reader(file)
                metric_dataframes[file.stem] = data_frame
            except (FileNotFoundError, Exception):
                pass
        return metric_dataframes

    def _aggregate_metric_statistic_histogram_data(self) -> HistogramConstantConfig.HistogramDataType:
        """
        Aggregate metric statistic histogram data.
        :return A dictionary of metric names and their aggregated data.
        """
        data: HistogramConstantConfig.HistogramDataType = defaultdict(list)
        for dataframe_filename, dataframe in self._metric_statistics_dataframes.items():
            histogram_data_list = aggregate_metric_statistics_dataframe_histogram_data(metric_statistics_dataframe=dataframe, metric_statistics_dataframe_index=0, metric_choices=[], scenario_types=None)
            if histogram_data_list:
                data[dataframe.metric_statistic_name] += histogram_data_list
        return data

    def _aggregate_scenario_type_score_histogram_data(self) -> HistogramConstantConfig.HistogramDataType:
        """
        Aggregate scenario type score histogram data.
        :return A dictionary of scenario type metric name and their scenario type scores.
        """
        data: HistogramConstantConfig.HistogramDataType = defaultdict(list)
        for index, (dataframe_filename, dataframe) in enumerate(self._metric_aggregator_dataframes.items()):
            histogram_data_list = aggregate_metric_aggregator_dataframe_histogram_data(metric_aggregator_dataframe=dataframe, metric_aggregator_dataframe_index=index, scenario_types=['all'], dataframe_file_name=dataframe_filename)
            if histogram_data_list:
                data[f'{HistogramConstantConfig.SCENARIO_TYPE_SCORE_HISTOGRAM_NAME}_{dataframe_filename}'] += histogram_data_list
        return data

    def _assign_planner_colors(self) -> Dict[str, Any]:
        """
        Assign colors to planners.
        :return A dictionary of planner and colors.
        """
        planner_color_maps = {}
        for dataframe_filename, dataframe in self._metric_statistics_dataframes.items():
            planner_names = dataframe.planner_names
            for planner_name in planner_names:
                if planner_name not in planner_color_maps:
                    planner_color_maps[planner_name] = self._color_choices[self._color_index % len(self._color_choices)]
                    self._color_index += 1
        return planner_color_maps

    def _save_to_pdf(self, matplotlib_plots: List[Any]) -> None:
        """
        Save a list of matplotlib plots to a pdf file.
        :param matplotlib_plots: A list of matplotlib plots.
        """
        file_name = safe_path_to_string(self._summary_output_path / self._pdf_file_name)
        pp = PdfPages(file_name)
        for fig in matplotlib_plots[::-1]:
            fig.savefig(pp, format='pdf')
        pp.close()
        plt.close()

    @staticmethod
    def _render_ax_hist(ax: Any, x_values: npt.NDArray[np.float64], x_axis_label: str, y_axis_label: str, bins: npt.NDArray[np.float64], label: str, color: str, ax_title: str) -> None:
        """
        Render axis with histogram bins.
        :param ax: Matplotlib axis.
        :param x_values: An array of histogram x-axis values.
        :param x_axis_label: Label in the x-axis.
        :param y_axis_label: Label in the y-axis.
        :param bins: An array of histogram bins.
        :param label: Legend name for the bins.
        :param color: Color for the bins.
        :param ax_title: Axis title.
        """
        ax.hist(x=x_values, bins=bins, label=label, color=color, weights=np.ones(len(x_values)) / len(x_values))
        ax.set_xlabel(x_axis_label, fontsize=HistogramTabMatPlotLibPlotStyleConfig.x_axis_label_size)
        ax.set_ylabel(y_axis_label, fontsize=HistogramTabMatPlotLibPlotStyleConfig.y_axis_label_size)
        ax.set_title(ax_title, fontsize=HistogramTabMatPlotLibPlotStyleConfig.axis_title_size)
        ax.set_ylim(ymin=0)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(axis='both', which='major', labelsize=HistogramTabMatPlotLibPlotStyleConfig.axis_ticker_size)
        ax.legend(fontsize=HistogramTabMatPlotLibPlotStyleConfig.legend_font_size)

    @staticmethod
    def _render_ax_bar_hist(ax: Any, x_values: Union[npt.NDArray[np.float64], List[str]], x_axis_label: str, y_axis_label: str, x_range: List[str], label: str, color: str, ax_title: str) -> None:
        """
        Render axis with bar histogram.
        :param ax: Matplotlib axis.
        :param x_values: An array of histogram x-axis values.
        :param x_axis_label: Label in the x-axis.
        :param y_axis_label: Label in the y-axis.
        :param x_range: A list of histogram category names.
        :param label: Legend name for the bins.
        :param color: Color for the bins.
        :param ax_title: Axis title.
        """
        value_categories = {key: 0.0 for key in x_range}
        for value in x_values:
            value_categories[str(value)] += 1.0
        category_names = list(value_categories.keys())
        category_values: List[float] = list(value_categories.values())
        num_scenarios = sum(category_values)
        if num_scenarios != 0:
            category_values = [value / num_scenarios * 100 for value in category_values]
            category_values = np.round(category_values, decimals=HistogramTabFigureStyleConfig.decimal_places)
        ax.bar(category_names, category_values, label=label, color=color)
        ax.set_xlabel(x_axis_label, fontsize=HistogramTabMatPlotLibPlotStyleConfig.x_axis_label_size)
        ax.set_ylabel(y_axis_label, fontsize=HistogramTabMatPlotLibPlotStyleConfig.y_axis_label_size)
        ax.set_title(ax_title, fontsize=HistogramTabMatPlotLibPlotStyleConfig.axis_title_size)
        ax.set_ylim(ymin=0)
        ax.tick_params(axis='both', which='major', labelsize=HistogramTabMatPlotLibPlotStyleConfig.axis_ticker_size)
        ax.legend(fontsize=HistogramTabMatPlotLibPlotStyleConfig.legend_font_size)

    def _draw_histogram_plots(self, planner_color_maps: Dict[str, Any], histogram_data_dict: HistogramConstantConfig.HistogramDataType, histogram_edges: HistogramConstantConfig.HistogramEdgesDataType, n_cols: int=2) -> None:
        """
        :param planner_color_maps: Color maps from planner names.
        :param histogram_data_dict: A dictionary of histogram data.
        :param histogram_edges: A dictionary of histogram edges (bins) data.
        :param n_cols: Number of columns in subplot.
        """
        matplotlib_plots = []
        for histogram_title, histogram_data_list in tqdm(histogram_data_dict.items(), desc='Rendering histograms'):
            for histogram_data in histogram_data_list:
                color = planner_color_maps.get(histogram_data.planner_name, None)
                if not color:
                    planner_color_maps[histogram_data.planner_name] = self._color_choices[self._color_index % len(self._color_choices)]
                    color = planner_color_maps.get(histogram_data.planner_name)
                    self._color_index += 1
                n_rows = math.ceil(len(histogram_data.statistics) / n_cols)
                fig_size = min(max(6, len(histogram_data.statistics) // 5 * 5), 24)
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_size, fig_size))
                flatten_axs = axs.flatten()
                fig.suptitle(histogram_title, fontsize=HistogramTabMatPlotLibPlotStyleConfig.main_title_size)
                for index, (statistic_name, statistic) in enumerate(histogram_data.statistics.items()):
                    unit = statistic.unit
                    bins: npt.NDArray[np.float64] = np.unique(histogram_edges[histogram_title].get(statistic_name, None))
                    assert bins is not None, f'Count edge data for {statistic_name} cannot be None!'
                    x_range = get_histogram_plot_x_range(unit=unit, data=bins)
                    values = np.round(statistic.values, HistogramTabFigureStyleConfig.decimal_places)
                    if unit in ['count']:
                        self._render_ax_bar_hist(ax=flatten_axs[index], x_values=values, x_range=x_range, x_axis_label=unit, y_axis_label='Frequency (%)', label=histogram_data.planner_name, color=color, ax_title=statistic_name)
                    elif unit in ['bool', 'boolean']:
                        values = ['True' if value else 'False' for value in values]
                        self._render_ax_bar_hist(ax=flatten_axs[index], x_values=values, x_range=x_range, x_axis_label=unit, y_axis_label='Frequency (%)', label=histogram_data.planner_name, color=color, ax_title=statistic_name)
                    else:
                        self._render_ax_hist(ax=flatten_axs[index], x_values=values, bins=bins, x_axis_label=unit, y_axis_label='Frequency (%)', label=histogram_data.planner_name, color=color, ax_title=statistic_name)
                if n_rows * n_cols != len(histogram_data.statistics.values()):
                    flatten_axs[-1].set_axis_off()
                plt.tight_layout()
                matplotlib_plots.append(fig)
        self._save_to_pdf(matplotlib_plots=matplotlib_plots)

    def on_run_simulation_end(self) -> None:
        """Callback before end of the main function."""
        start_time = time.perf_counter()
        if not self._metric_save_path.exists() and (not self._metric_aggregator_save_path.exists()):
            return
        self._metric_aggregator_dataframes = self._read_metric_parquet_files(metric_save_path=self._metric_aggregator_save_path, metric_reader=metric_aggregator_reader)
        self._metric_statistics_dataframes = self._read_metric_parquet_files(metric_save_path=self._metric_save_path, metric_reader=metric_statistics_reader)
        planner_color_maps = self._assign_planner_colors()
        histogram_data_dict = self._aggregate_metric_statistic_histogram_data()
        scenario_type_histogram_data_dict = self._aggregate_scenario_type_score_histogram_data()
        histogram_data_dict.update(scenario_type_histogram_data_dict)
        histogram_edge_data = compute_histogram_edges(bins=self._num_bins, aggregated_data=histogram_data_dict)
        self._draw_histogram_plots(planner_color_maps=planner_color_maps, histogram_data_dict=histogram_data_dict, histogram_edges=histogram_edge_data)
        end_time = time.perf_counter()
        elapsed_time_s = end_time - start_time
        time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time_s))
        logger.info('Metric summary: {} [HH:MM:SS]'.format(time_str))

class ValidationCallback(AbstractMainCallback):
    """Callback checking if a validation simulation was successful or not."""

    def __init__(self, output_dir: str, validation_dir_name: str):
        """
        :param output_dir: Root dir used to find the report file and as path to save results.
        :param validation_dir_name: Name of the directory where the validation file should be stored.
        """
        self.output_dir = Path(output_dir)
        self._validation_dir_name = validation_dir_name

    def on_run_simulation_end(self) -> None:
        """
        On reached_end push results to S3 bucket.
        """
        if _validation_succeeded(self.output_dir):
            filename = 'passed.txt'
        else:
            filename = 'failed.txt'
        logger.info('Validation filename: %s' % filename)
        validation_dir = self.output_dir / self._validation_dir_name
        if not is_s3_path(validation_dir):
            validation_dir.mkdir(parents=True, exist_ok=True)
        with (validation_dir / filename).open('w'):
            pass

def _validation_succeeded(source_folder_path: Path) -> bool:
    """
    Reads runners report and checks if the simulation was successful or not.
    :param source_folder_path:  Root folder to where runners report is stored.
    :return: True, if the simulation was successful, false otherwise.
    """
    try:
        df = pd.read_parquet(f'{source_folder_path}/runner_report.parquet')
    except FileNotFoundError:
        logger.warning('No runners report file found in %s!' % source_folder_path)
        return False
    return bool(np.all(df['succeeded'].values))

class MetricAggregatorCallback(AbstractMainCallback):
    """Callback to aggregate metrics after the simulation ends."""

    def __init__(self, metric_save_path: str, metric_aggregators: List[AbstractMetricAggregator]):
        """Callback to handle metric files at the end of process."""
        self._metric_save_path = Path(metric_save_path)
        self._metric_aggregators = metric_aggregators

    def on_run_simulation_end(self) -> None:
        """Callback before end of the main function."""
        start_time = time.perf_counter()
        if not is_s3_path(self._metric_save_path) and (not self._metric_save_path.exists()):
            return
        for metric_aggregator in self._metric_aggregators:
            metric_dataframes = {}
            if is_s3_path(self._metric_save_path):
                metrics = [path for path in list_files_in_directory(self._metric_save_path) if path.suffix == '.parquet']
            else:
                metrics = list(self._metric_save_path.rglob('*.parquet'))
            if not metric_aggregator.challenge:
                challenge_metrics = list(metrics)
            else:
                challenge_metrics = [path for path in metrics if metric_aggregator.challenge in str(path)]
            for file in challenge_metrics:
                try:
                    metric_statistic_dataframe = MetricStatisticsDataFrame.load_parquet(file)
                    metric_statistic_name = metric_statistic_dataframe.metric_statistic_name
                    metric_dataframes[metric_statistic_name] = metric_statistic_dataframe
                except (FileNotFoundError, Exception) as e:
                    logger.info(f'Cannot load the file: {file}, error: {e}')
            if metric_dataframes:
                logger.info(f'Running metric aggregator: {metric_aggregator.name}')
                metric_aggregator(metric_dataframes=metric_dataframes)
            else:
                logger.warning(f'{metric_aggregator.name}: No metric files found for aggregation!')
                logger.warning("If you didn't expect this, ensure that the challenge name is part of your submitted job name.")
        end_time = time.perf_counter()
        elapsed_time_s = end_time - start_time
        time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time_s))
        logger.info(f'Metric aggregator: {time_str} [HH:MM:SS]')

class TestValidationCallback(unittest.TestCase):
    """Tests for the ValidationCallback class"""

    def test_initialization(self) -> None:
        """Tests that the object is constructed correctly"""
        callback = ValidationCallback(output_dir='out', validation_dir_name='validation')
        self.assertEqual(str(callback.output_dir), 'out')
        self.assertEqual(callback._validation_dir_name, 'validation')

    def test_on_run_simulation_end(self) -> None:
        """Tests that the correct files are created in the callback."""
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        callback = ValidationCallback(output_dir=tmp_dir.name, validation_dir_name='validation')
        with patch(f'{TEST_FILE}._validation_succeeded', Mock(return_value=False)):
            callback.on_run_simulation_end()
            self.assertTrue(os.path.exists('/'.join([tmp_dir.name, 'validation', 'failed.txt'])))
        with patch(f'{TEST_FILE}._validation_succeeded', Mock(return_value=True)):
            callback.on_run_simulation_end()
            self.assertTrue(os.path.exists('/'.join([tmp_dir.name, 'validation', 'passed.txt'])))

    def test__validation_succeeded(self) -> None:
        """Tests that helper function reads the runners_report file correctly."""
        with patch(f'{TEST_FILE}.pd.read_parquet', Mock(side_effect=FileNotFoundError)):
            result = _validation_succeeded(Mock())
            self.assertFalse(result)
        failed_df = pd.DataFrame({'succeeded': [True, False]})
        with patch(f'{TEST_FILE}.pd.read_parquet', Mock(return_value=failed_df)):
            result = _validation_succeeded(Mock())
            self.assertFalse(result)
        passed_df = pd.DataFrame({'succeeded': [True, True]})
        with patch(f'{TEST_FILE}.pd.read_parquet', Mock(return_value=passed_df)):
            result = _validation_succeeded(Mock())
            self.assertTrue(result)

class TestMetricSummaryCallback(unittest.TestCase):
    """Test metric_summary callback functionality."""

    def set_up_dummy_metric(self, metric_path: Path, log_name: str, planner_name: str, scenario_type: str, scenario_name: str) -> None:
        """
        Set up dummy metric results.
        :param metric_path: Metric path.
        :param log_name: Log name.
        :param planner_name: Planner name.
        :param scenario_type: Scenario type.
        :param scenario_name: Scenario name.
        """
        statistics = [Statistic(name='ego_max_acceleration', unit='meters_per_second_squared', value=2.0, type=MetricStatisticsType.MAX), Statistic(name='ego_min_acceleration', unit='meters_per_second_squared', value=0.0, type=MetricStatisticsType.MIN), Statistic(name='ego_p90_acceleration', unit='meters_per_second_squared', value=1.0, type=MetricStatisticsType.P90)]
        time_stamps = [0, 1, 2]
        accel = [0.0, 1.0, 2.0]
        time_series = TimeSeries(unit='meters_per_second_squared', time_stamps=list(time_stamps), values=list(accel))
        result = MetricStatistics(metric_computator='ego_acceleration', name='ego_acceleration_statistics', statistics=statistics, time_series=time_series, metric_category='Dynamic', metric_score=1)
        key = MetricFileKey(metric_name='ego_acceleration', scenario_name=scenario_name, log_name=log_name, scenario_type=scenario_type, planner_name=planner_name)
        metric_engine = MetricsEngine(main_save_path=metric_path)
        metric_files = {'ego_acceleration': [MetricFile(key=key, metric_statistics=[result])]}
        metric_engine.write_to_files(metric_files=metric_files)
        metric_file_callback = MetricFileCallback(metric_file_output_path=str(metric_path), scenario_metric_paths=[str(metric_path)], delete_scenario_metric_files=True)
        metric_file_callback.on_run_simulation_end()

    def setUp(self) -> None:
        """Set up a nuboard base tab."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        log_name = 'dummy_log'
        planner_name = 'SimplePlanner'
        scenario_type = 'Test'
        scenario_name = 'Dummy_scene'
        metric_path = Path(self.tmp_dir.name) / 'metrics'
        metric_path.mkdir(exist_ok=True, parents=True)
        self.set_up_dummy_metric(metric_path=metric_path, log_name=log_name, planner_name=planner_name, scenario_name=scenario_name, scenario_type=scenario_type)
        self.aggregator_save_path = Path(self.tmp_dir.name) / 'aggregator_metric'
        self.weighted_average_metric_aggregator = WeightedAverageMetricAggregator(name='weighted_average_metric_aggregator', metric_weights={'default': 1.0, 'dummy_metric': 0.5}, file_name='test_weighted_average_metric_aggregator.parquet', aggregator_save_path=self.aggregator_save_path, multiple_metrics=[])
        self.metric_statistics_dataframes = {}
        for metric_parquet_file in metric_path.iterdir():
            print(metric_parquet_file)
            data_frame = MetricStatisticsDataFrame.load_parquet(metric_parquet_file)
            self.metric_statistics_dataframes[data_frame.metric_statistic_name] = data_frame
        self.metric_summary_output_path = Path(self.tmp_dir.name) / 'summary'
        self.metric_summary_callback = MetricSummaryCallback(metric_save_path=str(metric_path), metric_aggregator_save_path=str(self.aggregator_save_path), summary_output_path=str(self.metric_summary_output_path), pdf_file_name='summary.pdf')

    def test_metric_summary_callback_on_simulation_end(self) -> None:
        """Test on_simulation_end in metric summary callback."""
        self.weighted_average_metric_aggregator(metric_dataframes=self.metric_statistics_dataframes)
        self.metric_summary_callback.on_run_simulation_end()
        pdf_files = self.metric_summary_output_path.rglob('*.pdf')
        self.assertEqual(len(list(pdf_files)), 1)

    def tearDown(self) -> None:
        """Remove all temporary folders and files."""
        self.tmp_dir.cleanup()

class TestMetricAggregatorCallback(TestCase):
    """Test MetricAggregatorCallback."""

    def setUp(self) -> None:
        """Setup mocks for the tests"""
        self.mock_metric_aggregator_callback = Mock(spec=MetricAggregatorCallback)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp_dir.name)
        self.path.mkdir(parents=True, exist_ok=True)
        self.metric_aggregators = [MockAbstractMetricAggregator(self.path)]

    def tearDown(self) -> None:
        """Clean up tmp dir."""
        self.tmp_dir.cleanup()

    def test_metric_callback_init(self) -> None:
        """
        Tests if all the properties are set to the expected values in constructor.
        """
        metric_aggregator_callback = MetricAggregatorCallback(str(self.path), self.metric_aggregators)
        self.assertEqual(metric_aggregator_callback._metric_save_path, self.path)
        self.assertEqual(metric_aggregator_callback._metric_aggregators, self.metric_aggregators)

    @patch('nuplan.planning.simulation.main_callback.metric_aggregator_callback.logger')
    def test_on_run_simulation_end(self, logger: MagicMock) -> None:
        """
        Tests if the callback is called with the correct parameters.
        """
        metric_file_callback = MetricAggregatorCallback(str(self.path), self.metric_aggregators)
        metric_file_callback.on_run_simulation_end()
        logger.warning.assert_has_calls([call('dummy_metric_aggregator: No metric files found for aggregation!')])
        logger.info.assert_has_calls([call('Metric aggregator: 00:00:00 [HH:MM:SS]')])

class SimulationRunner(AbstractRunner):
    """
    Manager which executes multiple simulations with the same planner
    """

    def __init__(self, simulation: Simulation, planner: AbstractPlanner):
        """
        Initialize the simulations manager
        :param simulation: Simulation which will be executed
        :param planner: to be used to compute the desired ego's trajectory
        """
        self._simulation = simulation
        self._planner = planner

    def _initialize(self) -> None:
        """
        Initialize the planner
        """
        self._simulation.callback.on_initialization_start(self._simulation.setup, self.planner)
        self.planner.initialize(self._simulation.initialize())
        self._simulation.callback.on_initialization_end(self._simulation.setup, self.planner)

    @property
    def planner(self) -> AbstractPlanner:
        """
        :return: Planner used by the SimulationRunner
        """
        return self._planner

    @property
    def simulation(self) -> Simulation:
        """
        :return: Simulation used by the SimulationRunner
        """
        return self._simulation

    @property
    def scenario(self) -> AbstractScenario:
        """
        :return: Get the scenario relative to the simulation.
        """
        return self.simulation.scenario

    def run(self) -> RunnerReport:
        """
        Run through all simulations. The steps of execution follow:
         - Initialize all planners
         - Step through simulations until there no running simulation
        :return: List of SimulationReports containing the results of each simulation
        """
        start_time = time.perf_counter()
        report = RunnerReport(succeeded=True, error_message=None, start_time=start_time, end_time=None, planner_report=None, scenario_name=self._simulation.scenario.scenario_name, planner_name=self.planner.name(), log_name=self._simulation.scenario.log_name)
        self.simulation.callback.on_simulation_start(self.simulation.setup)
        self._initialize()
        while self.simulation.is_simulation_running():
            self.simulation.callback.on_step_start(self.simulation.setup, self.planner)
            planner_input = self._simulation.get_planner_input()
            logger.debug('Simulation iterations: %s' % planner_input.iteration.index)
            self._simulation.callback.on_planner_start(self.simulation.setup, self.planner)
            trajectory = self.planner.compute_trajectory(planner_input)
            self._simulation.callback.on_planner_end(self.simulation.setup, self.planner, trajectory)
            self.simulation.propagate(trajectory)
            self.simulation.callback.on_step_end(self.simulation.setup, self.planner, self.simulation.history.last())
            current_time = time.perf_counter()
            if not self.simulation.is_simulation_running():
                report.end_time = current_time
        self.simulation.callback.on_simulation_end(self.simulation.setup, self.planner, self.simulation.history)
        planner_report = self.planner.generate_planner_report()
        report.planner_report = planner_report
        return report

def run_simulation(sim_runner: AbstractRunner, exit_on_failure: bool=False) -> RunnerReport:
    """
    Proxy for calling simulation.
    :param sim_runner: A simulation runner which will execute all batched simulations.
    :param exit_on_failure: If true, raises an exception when the simulation fails.
    :return report for the simulation.
    """
    start_time = time.perf_counter()
    try:
        return sim_runner.run()
    except Exception as e:
        error = traceback.format_exc()
        logger.warning('----------- Simulation failed: with the following trace:')
        traceback.print_exc()
        logger.warning(f'Simulation failed with error:\n {e}')
        failed_scenarios = f'[{sim_runner.scenario.log_name}, {sim_runner.scenario.scenario_name}]\n'
        logger.warning(f'\nFailed simulation [log,token]:\n {failed_scenarios}')
        logger.warning('----------- Simulation failed!')
        if exit_on_failure:
            raise RuntimeError('Simulation failed')
        end_time = time.perf_counter()
        report = RunnerReport(succeeded=False, error_message=error, start_time=start_time, end_time=end_time, planner_report=None, scenario_name=sim_runner.scenario.scenario_name, planner_name=sim_runner.planner.name(), log_name=sim_runner.scenario.log_name)
        return report

def execute_runners(runners: List[AbstractRunner], worker: WorkerPool, num_gpus: Optional[Union[int, float]], num_cpus: Optional[int], exit_on_failure: bool=False, verbose: bool=False) -> List[RunnerReport]:
    """
    Execute multiple simulation runners or metric runners.
    :param runners: A list of simulations to be run.
    :param worker: for submitting tasks.
    :param num_gpus: if None, no GPU will be used, otherwise number (also fractional) of GPU used per simulation.
    :param num_cpus: if None, all available CPU threads are used, otherwise number of threads used.
    :param exit_on_failure: If true, raises an exception when the simulation fails.
    """
    assert len(runners) > 0, 'No scenarios found to simulate!'
    number_of_sims = len(runners)
    logger.info(f'Starting {number_of_sims} simulations using {worker.__class__.__name__}!')
    reports: List[RunnerReport] = worker.map(Task(fn=run_simulation, num_gpus=num_gpus, num_cpus=num_cpus), runners, exit_on_failure, verbose=verbose)
    results: Dict[Tuple[str, str, str], RunnerReport] = {(report.scenario_name, report.planner_name, report.log_name): report for report in reports}
    simulations_runners = (runner for runner in runners if isinstance(runner, SimulationRunner))
    relevant_simulations = ((runner.simulation, runner) for runner in simulations_runners)
    callback_futures_lists = ((callback.futures, simulation, runner) for simulation, runner in relevant_simulations for callback in simulation.callback.callbacks if isinstance(callback, MetricCallback) or isinstance(callback, SimulationLogCallback))
    callback_futures_map = {future: (simulation.scenario.scenario_name, runner.planner.name(), simulation.scenario.log_name) for futures, simulation, runner in callback_futures_lists for future in futures}
    for future in concurrent.futures.as_completed(callback_futures_map.keys()):
        try:
            future.result()
        except Exception:
            error_message = traceback.format_exc()
            runner_report = results[callback_futures_map[future]]
            runner_report.error_message = error_message
            runner_report.succeeded = False
            runner_report.end_time = time.perf_counter()
    failed_simulations = str()
    number_of_successful = 0
    runner_reports: List[RunnerReport] = list(results.values())
    for result in runner_reports:
        if result.succeeded:
            number_of_successful += 1
        else:
            logger.warning("Failed Simulation.\n '%s'", result.error_message)
            failed_simulations += f'[{result.log_name}, {result.scenario_name}] \n'
    number_of_failures = number_of_sims - number_of_successful
    logger.info(f'Number of successful simulations: {number_of_successful}')
    logger.info(f'Number of failed simulations: {number_of_failures}')
    if number_of_failures > 0:
        logger.info(f'Failed simulations [log, token]:\n{failed_simulations}')
    return runner_reports

class MetricRunner(AbstractRunner):
    """Manager which executes metrics with multiple simulation logs."""

    def __init__(self, simulation_log: SimulationLog, metric_callback: MetricCallback) -> None:
        """
        Initialize the metric manager.
        :param simulation_log: A simulation log.
        :param metric_callback: A metric callback.
        """
        self._simulation_log = simulation_log
        self._metric_callback = metric_callback

    def run(self) -> RunnerReport:
        """
        Run through all metric runners with simulation logs.
        :return A list of runner reports.
        """
        start_time = time.perf_counter()
        report = RunnerReport(succeeded=True, error_message=None, start_time=start_time, end_time=None, planner_report=None, scenario_name=self._simulation_log.scenario.scenario_name, planner_name=self._simulation_log.planner.name(), log_name=self._simulation_log.scenario.log_name)
        run_metric_engine(metric_engine=self._metric_callback.metric_engine, scenario=self._simulation_log.scenario, history=self._simulation_log.simulation_history, planner_name=self._simulation_log.planner.name())
        enc_time = time.perf_counter()
        report.end_time = enc_time
        return report

    @property
    def scenario(self) -> AbstractScenario:
        """
        :return: Get the scenario.
        """
        return self._simulation_log.scenario

    @property
    def planner(self) -> AbstractPlanner:
        """
        :return: Get a planner.
        """
        return self._simulation_log.planner

def run_metric_engine(metric_engine: MetricsEngine, scenario: AbstractScenario, planner_name: str, history: SimulationHistory) -> None:
    """
    Run the metric engine.
    """
    logger.debug('Starting metrics computation...')
    metric_files = metric_engine.compute(history, scenario=scenario, planner_name=planner_name)
    logger.debug('Finished metrics computation!')
    logger.debug('Saving metric statistics!')
    metric_engine.write_to_files(metric_files)
    logger.debug('Saved metrics!')

class SimulationLogCallback(AbstractCallback):
    """
    Callback for simulation logging/object serialization to disk.
    """

    def __init__(self, output_directory: Union[str, pathlib.Path], simulation_log_dir: Union[str, pathlib.Path], serialization_type: str, worker_pool: Optional[WorkerPool]=None):
        """
        Construct simulation log callback.
        :param output_directory: where scenes should be serialized.
        :param simulation_log_dir: Folder where to save simulation logs.
        :param serialization_type: A way to serialize output, options: ["json", "pickle", "msgpack"].
        """
        available_formats = ['pickle', 'msgpack']
        if serialization_type not in available_formats:
            raise ValueError(f'The simulation log callback will not store files anywhere!Choose at least one format from {available_formats} instead of {serialization_type}!')
        self._output_directory = pathlib.Path(output_directory) / simulation_log_dir
        self._serialization_type = serialization_type
        if serialization_type == 'pickle':
            file_suffix = '.pkl.xz'
        elif serialization_type == 'msgpack':
            file_suffix = '.msgpack.xz'
        else:
            raise ValueError(f'Unknown option: {serialization_type}')
        self._file_suffix = file_suffix
        self._pool = worker_pool
        self._futures: List[Future[None]] = []

    @property
    def futures(self) -> List[Future[None]]:
        """
        Returns a list of futures, eg. for the main process to block on.
        :return: any futures generated by running any part of the callback asynchronously.
        """
        return self._futures

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """
        Create directory at initialization
        :param setup: simulation setup
        :param planner: planner before initialization
        """
        scenario_directory = self._get_scenario_folder(planner.name(), setup.scenario)
        if not is_s3_path(scenario_directory):
            scenario_directory.mkdir(exist_ok=True, parents=True)

    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_end(self, setup: SimulationSetup, planner: AbstractPlanner, trajectory: AbstractTrajectory) -> None:
        """Inherited, see superclass."""
        pass

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """Inherited, see superclass."""
        pass

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """
        On reached_end validate that all steps were correctly serialized.
        :param setup: simulation setup.
        :param planner: planner when simulation ends.
        :param history: resulting from simulation.
        """
        number_of_scenes = len(history)
        if number_of_scenes == 0:
            raise RuntimeError('Number of scenes has to be greater than 0')
        scenario_directory = self._get_scenario_folder(planner.name(), setup.scenario)
        scenario = setup.scenario
        file_name = scenario_directory / (scenario.scenario_name + self._file_suffix)
        if self._pool is not None:
            self._futures = []
            self._futures.append(self._pool.submit(Task(_save_log_to_file, num_cpus=1, num_gpus=0), file_name, scenario, planner, history))
        else:
            _save_log_to_file(file_name, scenario, planner, history)

    def _get_scenario_folder(self, planner_name: str, scenario: AbstractScenario) -> pathlib.Path:
        """
        Compute scenario folder directory where all files will be stored.
        :param planner_name: planner name.
        :param scenario: for which to compute directory name.
        :return directory path.
        """
        return self._output_directory / planner_name / scenario.scenario_type / scenario.log_name / scenario.scenario_name

class MetricCallback(AbstractCallback):
    """Callback for computing metrics at the end of the simulation."""

    def __init__(self, metric_engine: MetricsEngine, worker_pool: Optional[WorkerPool]=None):
        """
        Build A metric callback.
        :param metric_engine: Metric Engine.
        """
        self._metric_engine = metric_engine
        self._pool = worker_pool
        self._futures: List[Future[None]] = []

    @property
    def metric_engine(self) -> MetricsEngine:
        """
        Returns metric engine.
        :return: metric engine
        """
        return self._metric_engine

    @property
    def futures(self) -> List[Future[None]]:
        """
        Returns a list of futures, eg. for the main process to block on.
        :return: any futures generated by running any part of the callback asynchronously.
        """
        return self._futures

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_end(self, setup: SimulationSetup, planner: AbstractPlanner, trajectory: AbstractTrajectory) -> None:
        """Inherited, see superclass."""
        pass

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """Inherited, see superclass."""
        pass

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """Inherited, see superclass."""
        if self._pool is not None:
            self._futures = []
            self._futures.append(self._pool.submit(Task(run_metric_engine, num_cpus=1, num_gpus=0), metric_engine=self._metric_engine, history=history, scenario=setup.scenario, planner_name=planner.name()))
        else:
            run_metric_engine(metric_engine=self._metric_engine, history=history, scenario=setup.scenario, planner_name=planner.name())

class ProfileCallback(pl.Callback):
    """Profiling callback that produces an html report."""

    def __init__(self, output_dir: pathlib.Path, interval: float=0.01):
        """
        Initialize callback.
        :param output_dir: directory where output should be stored. Note, "profiling" sub-dir will be added
        :param interval: of the profiler
        """
        self._output_dir = output_dir / 'profiling'
        if not is_s3_path(self._output_dir):
            self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Profiler will report into folder: {str(self._output_dir)}')
        self._profiler = Profiler(interval=interval)
        self._profiler_running = False

    def on_init_start(self, trainer: pl.Trainer) -> None:
        """
        Called during training initialization.
        :param trainer: Lightning trainer.
        """
        self.start_profiler('on_init_start')

    def on_init_end(self, trainer: pl.Trainer) -> None:
        """
        Called at the end of the training.
        :param trainer: Lightning trainer.
        """
        self.save_profiler('on_init_end')

    def on_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called at each epoch start.
        :param trainer: Lightning trainer.
        :param pl_module: lightning model.
        """
        self.start_profiler('on_epoch_start')

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called at each epoch end.
        :param trainer: Lightning trainer.
        :param pl_module: lightning model.
        """
        self.save_profiler('epoch_' + str(trainer.current_epoch) + '-on_epoch_end')

    def start_profiler(self, when: str) -> None:
        """
        Start the profiler.
        Raise: in case profiler is already running.
        :param when: Message to log when starting the profiler.
        """
        assert not self._profiler_running, 'Profiler can not be started twice!'
        logger.info(f'STARTING profiler: {when}')
        self._profiler_running = True
        self._profiler.start()

    def stop_profiler(self) -> None:
        """
        Start profiler
        Raise: in case profiler is not running
        """
        assert self._profiler_running, 'Profiler has to be running!!'
        self._profiler.stop()
        self._profiler_running = False

    def save_profiler(self, file_name: str) -> None:
        """
        Save profiling output to a html report
        :param file_name: File name to save report to.
        """
        self.stop_profiler()
        profiler_out_html = self._profiler.output_html()
        html_save_path = self._output_dir / file_name
        path = str(html_save_path.with_suffix('.html'))
        logger.info(f'Saving profiler output to: {path}')
        fp = open(path, 'w+')
        fp.write(profiler_out_html)
        fp.close()

def metric_statistics_reader(parquet_file: Path) -> MetricStatisticsDataFrame:
    """
    Reader for a metric statistic parquet file.
    :param parquet_file: Parquet file path to read.
    :return MetricStatisticsDataFrame.
    """
    data_frame = MetricStatisticsDataFrame.load_parquet(parquet_file)
    return data_frame

@dataclass
class ExperimentFileData:
    """Data for experiment files."""
    file_paths: List[NuBoardFile]
    color_palettes: List[str] = field(default_factory=list)
    expert_color_palettes: List[str] = field(default_factory=list)
    available_metric_statistics_names: List[str] = field(default_factory=list)
    metric_statistics_dataframes: List[List[MetricStatisticsDataFrame]] = field(default_factory=list)
    metric_aggregator_dataframes: List[Dict[str, pd.DataFrame]] = field(default_factory=list)
    simulation_files: Dict[str, Any] = field(default_factory=dict)
    simulation_scenario_keys: List[SimulationScenarioKey] = field(default_factory=list)
    available_scenario_types: List[str] = field(default_factory=list)
    available_scenarios: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    available_scenario_tokens: Dict[str, ScenarioTokenInfo] = field(default_factory=dict)
    file_path_colors: Dict[int, Dict[str, str]] = field(default_factory=dict)
    color_index: int = 0

    def __post_init__(self) -> None:
        """Post initialization."""
        if not self.simulation_files:
            self.simulation_files = defaultdict(set)
        if not self.available_scenario_tokens:
            self.available_scenario_tokens = defaultdict()
        if not self.color_palettes:
            self.color_palettes = Set1[9] + Set2[8] + Set3[12]
        if not self.expert_color_palettes:
            self.expert_color_palettes = Pastel2[8] + Pastel1[9] + Dark2[8]
        if not self.available_scenarios:
            self.available_scenarios = defaultdict(lambda: defaultdict(list))
        if self.file_paths:
            file_paths = self.file_paths
            self.file_paths = []
            self.update_data(file_paths=file_paths)

    def update_data(self, file_paths: List[NuBoardFile]) -> None:
        """
        Update experiment data with a new list of nuboard file paths.
        :param file_paths: A list of new nuboard file paths.
        """
        starting_file_path_index = len(self.file_paths)
        self._update_file_path_color(file_paths=file_paths, starting_file_path_index=starting_file_path_index)
        self._add_metric_files(file_paths=file_paths, starting_file_path_index=starting_file_path_index)
        self._add_metric_aggregator_files(file_paths=file_paths, starting_file_path_index=starting_file_path_index)
        self._add_simulation_files(file_paths=file_paths, starting_file_path_index=starting_file_path_index)
        self.file_paths += file_paths

    @staticmethod
    def _get_base_path(current_path: Path, base_path: Path, sub_folder: str) -> Path:
        """
        Get valid base path.
        :param current_path: Current nuboard file path.
        :Param base_path: Alternative base path.
        :param sub_folder: Sub folder.
        :return A base path.
        """
        default_path = base_path / sub_folder
        if current_path is None:
            return default_path
        base_folder = current_path / sub_folder
        if not base_folder.exists():
            base_folder = default_path
        return base_folder

    def _update_file_path_color(self, file_paths: List[NuBoardFile], starting_file_path_index: int) -> None:
        """
        Update file path colors.
        :param file_paths: A list of new nuboard file paths.
        :param starting_file_path_index: Starting file path index.
        """
        for index, file_path in enumerate(file_paths):
            file_path_index = starting_file_path_index + index
            self.file_path_colors[file_path_index] = defaultdict(str)
            metric_path = self._get_base_path(current_path=file_path.current_path, base_path=Path(file_path.metric_main_path), sub_folder=file_path.metric_folder)
            planner_names: List[str] = []
            if not metric_path.exists():
                continue
            for file in metric_path.iterdir():
                try:
                    data_frame = MetricStatisticsDataFrame.load_parquet(file)
                    planner_names += data_frame.planner_names
                except (FileNotFoundError, Exception) as e:
                    logger.info(e)
                    pass
            if not planner_names:
                simulation_path = self._get_base_path(current_path=file_path.current_path, base_path=Path(file_path.simulation_main_path), sub_folder=file_path.simulation_folder)
                if not simulation_path.exists():
                    continue
                planner_name_paths = simulation_path.iterdir()
                for planner_name_path in planner_name_paths:
                    planner_name = planner_name_path.name
                    planner_names.append(planner_name)
            planner_names = list(set(planner_names))
            for planner_name in planner_names:
                self.file_path_colors[file_path_index][planner_name] = self.color_palettes[self.color_index]
                self.color_index += 1

    def _add_metric_files(self, file_paths: List[NuBoardFile], starting_file_path_index: int) -> None:
        """
        Add and load metric files.
        Folder hierarchy: planner_name -> scenario_type -> metric result name -> scenario_name.pkl
        :param file_paths: A list of new nuboard files.
        :param starting_file_path_index: Starting file path index.
        """
        for index, file_path in enumerate(file_paths):
            file_path_index = starting_file_path_index + index
            self.metric_statistics_dataframes.append([])
            metric_path = self._get_base_path(current_path=file_path.current_path, base_path=Path(file_path.metric_main_path), sub_folder=file_path.metric_folder)
            if not metric_path.exists():
                continue
            for file in metric_path.iterdir():
                if file.is_dir():
                    continue
                try:
                    data_frame = MetricStatisticsDataFrame.load_parquet(file)
                    self.metric_statistics_dataframes[file_path_index].append(data_frame)
                    self.available_metric_statistics_names.append(data_frame.metric_statistic_name)
                except (FileNotFoundError, Exception):
                    pass
        self.available_metric_statistics_names = sorted(list(set(self.available_metric_statistics_names)), reverse=False)

    def _add_metric_aggregator_files(self, file_paths: List[NuBoardFile], starting_file_path_index: int) -> None:
        """
        Load metric aggregator files.
        :param file_paths: A list of new nuboard files.
        :param starting_file_path_index: Starting file path index.
        """
        for index, file_path in enumerate(file_paths):
            file_path_index = starting_file_path_index + index
            self.metric_aggregator_dataframes.append({})
            metric_aggregator_path = self._get_base_path(current_path=file_path.current_path, base_path=Path(file_path.metric_main_path), sub_folder=file_path.aggregator_metric_folder)
            if not metric_aggregator_path.exists():
                continue
            for file in metric_aggregator_path.iterdir():
                if file.is_dir():
                    continue
                try:
                    data_frame = pd.read_parquet(file)
                    self.metric_aggregator_dataframes[file_path_index][file.stem] = data_frame
                except (FileNotFoundError, Exception):
                    pass

    def _add_simulation_files(self, file_paths: List[NuBoardFile], starting_file_path_index: int) -> None:
        """
        Load simulation files.
        Folder hierarchy: planner_name -> scenario_type -> scenario_names -> iteration.pkl.
        :param file_paths: A list of new nuboard files.
        :param starting_file_path_index: Starting file path index.
        """
        for index, file_path in enumerate(file_paths):
            if file_path.simulation_folder is None:
                continue
            file_path_index = starting_file_path_index + index
            simulation_path = self._get_base_path(current_path=file_path.current_path, base_path=Path(file_path.simulation_main_path), sub_folder=file_path.simulation_folder)
            if not simulation_path.exists():
                continue
            planner_name_paths = simulation_path.iterdir()
            for planner_name_path in planner_name_paths:
                planner_name = planner_name_path.name
                scenario_type_paths = planner_name_path.iterdir()
                for scenario_type_path in scenario_type_paths:
                    log_name_paths = scenario_type_path.iterdir()
                    scenario_type = scenario_type_path.name
                    for log_name_path in log_name_paths:
                        scenario_name_paths = log_name_path.iterdir()
                        log_name = log_name_path.name
                        for scenario_name_path in scenario_name_paths:
                            scenario_name = scenario_name_path.name
                            scenario_key = f'{simulation_path.parents[0].name}/{planner_name}/{scenario_type}/{log_name}/{scenario_name}'
                            if scenario_key in self.simulation_files:
                                continue
                            files = scenario_name_path.iterdir()
                            for file in files:
                                self.simulation_files[scenario_key].add(file)
                            self.available_scenarios[scenario_type][log_name].append(scenario_name)
                            self.available_scenario_tokens[scenario_name] = ScenarioTokenInfo(scenario_name=scenario_name, scenario_token=scenario_name, scenario_type=scenario_type, log_name=log_name)
                            self.simulation_scenario_keys.append(SimulationScenarioKey(nuboard_file_index=file_path_index, log_name=log_name, planner_name=planner_name, scenario_type=scenario_type, scenario_name=scenario_name, files=list(self.simulation_files[scenario_key])))
        available_scenario_types = list(set(self.available_scenarios.keys()))
        self.available_scenario_types = sorted(available_scenario_types, reverse=False)

class MetricsEngine:
    """The metrics engine aggregates and manages the instantiated metrics for a scenario."""

    def __init__(self, main_save_path: Path, metrics: Optional[List[AbstractMetricBuilder]]=None) -> None:
        """
        Initializer for MetricsEngine class
        :param metrics: Metric objects.
        """
        self._main_save_path = main_save_path
        if not is_s3_path(self._main_save_path):
            self._main_save_path.mkdir(parents=True, exist_ok=True)
        if metrics is None:
            self._metrics: List[AbstractMetricBuilder] = []
        else:
            self._metrics = metrics

    @property
    def metrics(self) -> List[AbstractMetricBuilder]:
        """Retrieve a list of metric results."""
        return self._metrics

    def add_metric(self, metric_builder: AbstractMetricBuilder) -> None:
        """TODO: Create the list of types needed from the history"""
        self._metrics.append(metric_builder)

    def write_to_files(self, metric_files: Dict[str, List[MetricFile]]) -> None:
        """
        Write to a file by constructing a dataframe
        :param metric_files: A dictionary of scenario names and a list of their metric files.
        """
        for scenario_name, metric_files in metric_files.items():
            file_name = scenario_name + JSON_FILE_EXTENSION
            save_path = self._main_save_path / file_name
            dataframes = []
            for metric_file in metric_files:
                metric_file_key = metric_file.key
                for metric_statistic in metric_file.metric_statistics:
                    dataframe = construct_dataframe(log_name=metric_file_key.log_name, scenario_name=metric_file_key.scenario_name, scenario_type=metric_file_key.scenario_type, planner_name=metric_file_key.planner_name, metric_statistics=metric_statistic)
                    dataframes.append(dataframe)
            if len(dataframes):
                save_object_as_pickle(save_path, dataframes)

    def compute_metric_results(self, history: SimulationHistory, scenario: AbstractScenario) -> Dict[str, List[MetricStatistics]]:
        """
        Compute metrics in the engine
        :param history: History from simulation
        :param scenario: Scenario running this metric engine
        :return A list of metric statistics.
        """
        metric_results = {}
        for metric in self._metrics:
            try:
                start_time = time.perf_counter()
                metric_results[metric.name] = metric.compute(history, scenario=scenario)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                logger.debug(f'Metric: {metric.name} running time: {elapsed_time:.2f} seconds.')
            except (NotImplementedError, Exception) as e:
                logger.error(f'Running {metric.name} with error: {e}')
                raise RuntimeError(f'Metric Engine failed with: {e}')
        return metric_results

    def compute(self, history: SimulationHistory, scenario: AbstractScenario, planner_name: str) -> Dict[str, List[MetricFile]]:
        """
        Compute metrics and return in a format of MetricStorageResult for each metric computation
        :param history: History from simulation
        :param scenario: Scenario running this metric engine
        :param planner_name: name of the planner
        :return A dictionary of scenario name and list of MetricStorageResult.
        """
        all_metrics_results = self.compute_metric_results(history=history, scenario=scenario)
        metric_files = defaultdict(list)
        for metric_name, metric_statistics_results in all_metrics_results.items():
            metric_file_key = MetricFileKey(metric_name=metric_name, log_name=scenario.log_name, scenario_name=scenario.scenario_name, scenario_type=scenario.scenario_type, planner_name=planner_name)
            metric_file = MetricFile(key=metric_file_key, metric_statistics=metric_statistics_results)
            metric_file_name = scenario.scenario_type + '_' + scenario.scenario_name + '_' + planner_name
            metric_files[metric_file_name].append(metric_file)
        return metric_files

@dataclass
class MetricStatisticsDataFrame:
    """Metric statistics data frame class."""
    metric_statistic_name: str
    metric_statistics_dataframe: pandas.DataFrame
    time_series_unit_column: ClassVar[str] = 'time_series_unit'
    time_series_timestamp_column: ClassVar[str] = 'time_series_timestamps'
    time_series_values_column: ClassVar[str] = 'time_series_values'
    time_series_selected_frames_column: ClassVar[str] = 'time_series_selected_frames'

    def __eq__(self, other: object) -> bool:
        """Compare equality."""
        if not isinstance(other, MetricStatisticsDataFrame):
            return NotImplemented
        return self.metric_statistic_name == other.metric_statistic_name and self.metric_statistics_dataframe.equals(other.metric_statistics_dataframe)

    def __hash__(self) -> int:
        """Implement hash for caching."""
        return hash(self.metric_statistic_name) + id(self.metric_statistics_dataframe)

    @classmethod
    def load_parquet(cls, parquet_path: Path) -> MetricStatisticsDataFrame:
        """
        Load a parquet file to this class.
        The path can be local or s3.
        :param parquet_path: A path to a parquet file.
        """
        data_frame = pandas.read_parquet(path=safe_path_to_string(parquet_path))
        try:
            if not len(data_frame):
                raise IndexError
            metric_statistics_name = data_frame['metric_statistics_name'][0]
        except (IndexError, Exception):
            metric_statistics_name = parquet_path.stem
        return MetricStatisticsDataFrame(metric_statistic_name=metric_statistics_name, metric_statistics_dataframe=data_frame)

    @lru_cache
    def query_scenarios(self, scenario_names: Optional[Tuple[str]]=None, scenario_types: Optional[Tuple[str]]=None, planner_names: Optional[Tuple[str]]=None, log_names: Optional[Tuple[str]]=None) -> pandas.DataFrame:
        """
        Query scenarios with a list of scenario types and planner names.
        :param scenario_names: A tuple of scenario names.
        :param scenario_types: A tuple of scenario types.
        :param planner_names: A tuple of planner names.
        :param log_names: A tuple of log names.
        :return Pandas dataframe after filtering.
        """
        if not scenario_names and (not scenario_types) and (not planner_names):
            return self.metric_statistics_dataframe
        default_query: npt.NDArray[np.bool_] = np.asarray([True] * len(self.metric_statistics_dataframe.index))
        scenario_name_query = self.metric_statistics_dataframe['scenario_name'].isin(scenario_names) if scenario_names else default_query
        scenario_type_query = self.metric_statistics_dataframe['scenario_type'].isin(scenario_types) if scenario_types else default_query
        planner_name_query = self.metric_statistics_dataframe['planner_name'].isin(planner_names) if planner_names else default_query
        log_name_query = self.metric_statistics_dataframe['log_name'].isin(log_names) if log_names else default_query
        return self.metric_statistics_dataframe[scenario_name_query & scenario_type_query & planner_name_query & log_name_query]

    @cached_property
    def metric_statistics_names(self) -> List[str]:
        """Return metric statistic names."""
        return list(self.metric_statistics_dataframe['metric_statistics_name'].unique())

    @cached_property
    def metric_computator(self) -> str:
        """Return metric computator."""
        if len(self.metric_statistics_dataframe):
            return self.metric_statistics_dataframe['metric_computator'][0]
        else:
            raise IndexError('No available records found!')

    @cached_property
    def metric_category(self) -> str:
        """Return metric category."""
        if len(self.metric_statistics_dataframe):
            return self.metric_statistics_dataframe['metric_category'][0]
        else:
            raise IndexError('No available records found!')

    @cached_property
    def metric_score_unit(self) -> str:
        """Return metric score unit."""
        return self.metric_statistics_dataframe['metric_score_unit'][0]

    @cached_property
    def scenario_types(self) -> List[str]:
        """Return a list of scenario types."""
        return list(self.metric_statistics_dataframe['scenario_type'].unique())

    @cached_property
    def scenario_names(self) -> List[str]:
        """Return a list of scenario names."""
        return list(self.metric_statistics_dataframe['scenario_name'])

    @cached_property
    def column_names(self) -> List[str]:
        """Return a list of column names in a table."""
        return list(self.metric_statistics_dataframe.columns)

    @cached_property
    def statistic_names(self) -> List[str]:
        """Return a list of statistic names in a table."""
        return [col.split('_stat_type')[0] for col in self.column_names if '_stat_type' in col]

    @cached_property
    def time_series_headers(self) -> List[str]:
        """Return time series headers."""
        return [self.time_series_unit_column, self.time_series_timestamp_column, self.time_series_values_column]

    @cached_property
    def get_time_series_selected_frames(self) -> Optional[List[int]]:
        """Return selected frames in time series."""
        try:
            return self.metric_statistics_dataframe[self.time_series_selected_frames_column].iloc[0]
        except KeyError:
            return None

    @cached_property
    def time_series_dataframe(self) -> pandas.DataFrame:
        """Return time series dataframe."""
        return self.metric_statistics_dataframe.loc[:, self.time_series_headers]

    @lru_cache
    def statistics_dataframe(self, statistic_names: Optional[Tuple[str]]=None) -> pandas.DataFrame:
        """
        Return statistics columns
        :param statistic_names: A list of statistic names to query
        :return Pandas dataframe after querying.
        """
        if statistic_names:
            return self.metric_statistics_dataframe[statistic_names]
        statistic_headers = []
        for column_name in self.column_names:
            for statistic_name in self.statistic_names:
                if statistic_name in column_name:
                    statistic_headers.append(column_name)
                    continue
        return self.metric_statistics_dataframe[statistic_headers]

    @cached_property
    def planner_names(self) -> List[str]:
        """Return a list of planner names."""
        return list(self.metric_statistics_dataframe['planner_name'].unique())

class WeightedAverageMetricAggregator(AbstractMetricAggregator):
    """Metric aggregator by implementing weighted sum."""

    def __init__(self, name: str, metric_weights: Dict[str, float], file_name: str, aggregator_save_path: Path, multiple_metrics: List[str], challenge_name: Optional[str]=None):
        """
        Initializes the WeightedAverageMetricAggregator class.
        :param name: Metric aggregator name.
        :param metric_weights: Weights for each metric. Default would be 1.0.
        :param file_name: Saved file name.
        :param aggregator_save_path: Save path for this aggregated parquet file.
        :param multiple_metrics: A list if metric names used in multiple factor when computing scenario scores.
        :param challenge_name: Optional, name of the challenge the metrics refer to, if set will be part of the
        output file name and path.
        """
        self._name = name
        self._metric_weights = metric_weights
        self._file_name = file_name
        if not self._file_name.endswith('.parquet'):
            self._file_name += '.parquet'
        self._aggregator_save_path = aggregator_save_path
        self._challenge_name = challenge_name
        if not is_s3_path(self._aggregator_save_path):
            self._aggregator_save_path.mkdir(exist_ok=True, parents=True)
        self._aggregator_type = 'weighted_average'
        self._multiple_metrics = multiple_metrics
        self._parquet_file = self._aggregator_save_path / self._file_name
        self._aggregated_metric_dataframe: Optional[pandas.DataFrame] = None

    @property
    def aggregated_metric_dataframe(self) -> Optional[pandas.DataFrame]:
        """Return the aggregated metric dataframe."""
        return self._aggregated_metric_dataframe

    @property
    def name(self) -> str:
        """
        Return the metric aggregator name.
        :return the metric aggregator name.
        """
        return self._name

    @property
    def final_metric_score(self) -> Optional[float]:
        """Return the final metric score."""
        if self._aggregated_metric_dataframe is not None:
            return self._aggregated_metric_dataframe.iloc[-1, -1]
        else:
            logger.warning('The metric not yet aggregated.')
            return None

    def _get_metric_weight(self, metric_name: str) -> float:
        """
        Get metric weights.
        :param metric_name: The metric name.
        :return Weight for the metric.
        """
        weight: Optional[float] = self._metric_weights.get(metric_name, None)
        metric_weight = self._metric_weights.get('default', 1.0) if weight is None else weight
        return metric_weight

    def _compute_scenario_score(self, scenario_metric_columns: metric_aggregator_dict_column) -> None:
        """
        Compute scenario scores.
        :param scenario_metric_columns: Scenario metric column in the format of {scenario_names: {metric_column:
        value}}.
        """
        excluded_columns = ['log_name', 'planner_name', 'aggregator_type', 'scenario_type', 'num_scenarios', 'score']
        for scenario_name, columns in scenario_metric_columns.items():
            metric_scores = 0.0
            sum_weights = 0.0
            multiple_factor = 1.0
            for column_key, column_value in columns.items():
                if column_key in excluded_columns or column_value is None:
                    continue
                if self._multiple_metrics and column_key in self._multiple_metrics:
                    multiple_factor *= column_value
                else:
                    weight = self._get_metric_weight(metric_name=column_key)
                    assert column_value is not None, f'Metric: {column_key} value should not be None!'
                    assert weight is not None, f'Metric: {column_key} weight should not be None!'
                    sum_weights += weight
                    metric_scores += weight * column_value
            weighted_average_score = metric_scores / sum_weights if sum_weights else 0.0
            final_score = multiple_factor * weighted_average_score
            scenario_metric_columns[scenario_name]['score'] = final_score

    @staticmethod
    def _group_scenario_type_metric(scenario_metric_columns: metric_aggregator_dict_column) -> metric_aggregator_dict_column:
        """
        Group scenario type metric columns in the format of {scenario_type: {metric_columns: value}}.
        :param scenario_metric_columns: Scenario metric columns in the format of {scenario_name: {metric_columns:
        value}}.
        :return Metric columns based on scenario type.
        """
        scenario_type_dicts: metric_aggregator_dict_column = defaultdict(lambda: defaultdict(list))
        total_scenarios = len(scenario_metric_columns)
        for scenario_name, columns in scenario_metric_columns.items():
            scenario_type = columns['scenario_type']
            scenario_type_dicts[scenario_type]['scenario_name'].append(scenario_name)
            for column_key, column_value in columns.items():
                scenario_type_dicts[scenario_type][column_key].append(column_value)
        common_columns = ['planner_name', 'aggregator_type', 'scenario_type']
        excluded_columns = ['scenario_name']
        scenario_type_metric_columns: metric_aggregator_dict_column = defaultdict(lambda: defaultdict())
        for scenario_type, columns in scenario_type_dicts.items():
            for key, values in columns.items():
                if key in excluded_columns:
                    continue
                elif key in common_columns:
                    scenario_type_metric_columns[scenario_type][key] = values[0]
                elif key == 'log_name':
                    scenario_type_metric_columns[scenario_type][key] = None
                elif key == 'num_scenarios':
                    scenario_type_metric_columns[scenario_type]['num_scenarios'] = len(values)
                else:
                    available_values: npt.NDArray[np.float64] = np.asarray([value for value in values if value is not None])
                    value: Optional[float] = float(np.sum(available_values)) if available_values.size > 0 else None
                    if key == 'score' and value is not None:
                        score_value: float = value / len(values) if total_scenarios else 0.0
                        scenario_type_metric_columns[scenario_type][key] = score_value
                    else:
                        scenario_type_metric_columns[scenario_type][key] = value
        return scenario_type_metric_columns

    @staticmethod
    def _group_final_score_metric(scenario_type_metric_columns: metric_aggregator_dict_column) -> metric_aggregator_dict_column:
        """
        Compute a final score based on a group of scenario types.
        :param scenario_type_metric_columns: Scenario type metric columns in the format of {scenario_type:
        {metric_column: value}}.
        :return A dictionary of final score in the format of {'final_score': {metric_column: value}}.
        """
        final_score_dicts: metric_aggregator_dict_column = defaultdict(lambda: defaultdict(list))
        for scenario_type, columns in scenario_type_metric_columns.items():
            for column_key, column_value in columns.items():
                final_score_dicts['final_score'][column_key].append(column_value)
        final_score_metric_columns: metric_aggregator_dict_column = defaultdict(lambda: defaultdict())
        total_scenarios = sum(final_score_dicts['final_score']['num_scenarios'])
        common_columns = ['planner_name', 'aggregator_type']
        for final_score_column_name, columns in final_score_dicts.items():
            for key, values in columns.items():
                if key == 'scenario_type':
                    final_score_metric_columns[final_score_column_name][key] = 'final_score'
                elif key == 'log_name':
                    final_score_metric_columns[final_score_column_name][key] = None
                elif key in common_columns:
                    final_score_metric_columns[final_score_column_name][key] = values[0]
                elif key == 'num_scenarios':
                    final_score_metric_columns[final_score_column_name][key] = total_scenarios
                else:
                    available_values: List[float] = []
                    if key == 'score':
                        for value, num_scenario in zip(values, columns['num_scenarios']):
                            if value is not None:
                                available_values.append(value * num_scenario)
                    else:
                        available_values = [value for value in values if value is not None]
                    if not available_values:
                        total_values = None
                    else:
                        available_value_array: npt.NDArray[np.float64] = np.asarray(available_values)
                        total_values = np.sum(available_value_array) / total_scenarios
                    final_score_metric_columns[final_score_column_name][key] = total_values
        return final_score_metric_columns

    def _group_scenario_metrics(self, metric_dataframes: Dict[str, MetricStatisticsDataFrame], planner_name: str) -> metric_aggregator_dict_column:
        """
        Group scenario metrics in the format of {scenario_name: {metric_column: value}}.
        :param metric_dataframes: A dict of metric dataframes.
        :param planner_name: A planner name.
        :return Dictionary column format in metric aggregator in {scenario_name: {metric_column: value}}.
        """
        metric_names = sorted(list(metric_dataframes.keys()))
        columns = {column: None for column in ['log_name', 'planner_name', 'aggregator_type', 'scenario_type', 'num_scenarios'] + metric_names + ['score']}
        scenario_metric_columns: metric_aggregator_dict_column = {}
        for metric_name, metric_dataframe in metric_dataframes.items():
            dataframe = metric_dataframe.query_scenarios(planner_names=tuple([planner_name]))
            for _, data in dataframe.iterrows():
                scenario_name = data.get('scenario_name')
                if scenario_name not in scenario_metric_columns:
                    scenario_metric_columns[scenario_name] = deepcopy(columns)
                scenario_type = data['scenario_type']
                scenario_metric_columns[scenario_name]['log_name'] = data['log_name']
                scenario_metric_columns[scenario_name]['planner_name'] = data['planner_name']
                scenario_metric_columns[scenario_name]['scenario_type'] = scenario_type
                scenario_metric_columns[scenario_name]['aggregator_type'] = self._aggregator_type
                scenario_metric_columns[scenario_name][metric_name] = data['metric_score']
        return scenario_metric_columns

    def __call__(self, metric_dataframes: Dict[str, MetricStatisticsDataFrame]) -> None:
        """
        Run an aggregator to generate an aggregated parquet file.
        :param metric_dataframes: A dictionary of metric name and dataframe.
        """
        planner_names = sorted(list({planner_name for metric_statistic_dataframe in metric_dataframes.values() for planner_name in metric_statistic_dataframe.planner_names}))
        weighted_average_dataframe_columns: Dict[str, List[Any]] = dict()
        for planner_name in planner_names:
            metric_names = sorted(list(metric_dataframes.keys())) + ['score']
            dataframe_columns: Dict[str, List[Any]] = {'scenario': [], 'log_name': [], 'scenario_type': [], 'num_scenarios': [], 'planner_name': [], 'aggregator_type': []}
            metric_name_columns: Dict[str, List[float]] = {metric_name: [] for metric_name in metric_names}
            dataframe_columns.update(metric_name_columns)
            scenario_metric_columns = self._group_scenario_metrics(metric_dataframes=metric_dataframes, planner_name=planner_name)
            self._compute_scenario_score(scenario_metric_columns=scenario_metric_columns)
            scenario_type_metric_columns = self._group_scenario_type_metric(scenario_metric_columns=scenario_metric_columns)
            scenario_type_final_metric_columns = self._group_final_score_metric(scenario_type_metric_columns=scenario_type_metric_columns)
            scenario_metric_columns.update(scenario_type_metric_columns)
            scenario_metric_columns.update(scenario_type_final_metric_columns)
            for scenario_name, columns in scenario_metric_columns.items():
                dataframe_columns['scenario'].append(scenario_name)
                for key, value in columns.items():
                    dataframe_columns[key].append(value)
            if not weighted_average_dataframe_columns:
                weighted_average_dataframe_columns.update(dataframe_columns)
            else:
                for column_name, value in weighted_average_dataframe_columns.items():
                    value += dataframe_columns[column_name]
        self._aggregated_metric_dataframe = pandas.DataFrame(data=weighted_average_dataframe_columns)
        self._save_parquet(dataframe=self._aggregated_metric_dataframe, save_path=self._parquet_file)

    def read_parquet(self) -> None:
        """Read a parquet file."""
        self._aggregated_metric_dataframe = pandas.read_parquet(self._parquet_file)

    @property
    def parquet_file(self) -> Path:
        """Inherited, see superclass"""
        return self._parquet_file

    @property
    def challenge(self) -> Optional[str]:
        """Inherited, see superclass"""
        return self._challenge_name

class AbstractMetricAggregator(metaclass=ABCMeta):
    """Interface for metric aggregator"""

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the metric aggregator name
        :return the metric aggregator name.
        """
        pass

    @property
    @abstractmethod
    def final_metric_score(self) -> Optional[float]:
        """Returns the final metric score."""
        pass

    @abstractmethod
    def __call__(self, metric_dataframes: Dict[str, MetricStatisticsDataFrame]) -> None:
        """
        Run an aggregator to generate an aggregated parquet file
        :param metric_dataframes: A dictionary of metric name and dataframe.
        """
        pass

    @staticmethod
    def _save_with_metadata(dataframe: pandas.DataFrame, save_path: Path, metadata: Dict[str, str]) -> None:
        """
        Save to a parquet file with additional metadata using pyarrow
        :param dataframe: Pandas dataframe
        :param save_path: Path to save the dataframe.
        """
        pyarrow_table = pyarrow.Table.from_pandas(df=dataframe)
        schema_metadata = pyarrow_table.schema.metadata
        schema_metadata.update(metadata)
        updated_schema = pyarrow_table.schema.with_metadata(schema_metadata)
        pyarrow_table = pyarrow_table.cast(updated_schema)
        pq.write_table(pyarrow_table, str(save_path))

    @staticmethod
    def _save_parquet(dataframe: pandas.DataFrame, save_path: Path) -> None:
        """
        Save dataframe to a parquet file.
        The path can be local or s3.
        :param dataframe: Pandas dataframe.
        :param save_path: Path to save the dataframe.
        """
        dataframe.to_parquet(safe_path_to_string(save_path))

    @abstractmethod
    def read_parquet(self) -> None:
        """Read a parquet file, and update the dataframe."""
        pass

    @property
    @abstractmethod
    def parquet_file(self) -> Path:
        """Getter for the path to the generated parquet file."""
        pass

    @property
    @abstractmethod
    def challenge(self) -> Optional[str]:
        """Returns the name of the challenge, if applicable."""
        pass

class TestWeightedAverageMetricAggregator(unittest.TestCase):
    """Run weighted average metric aggregator unit tests."""

    def setUp(self) -> None:
        """Set up dummy data and folders."""
        self.metric_scores = [[1, 0.5, 0.8], [0.1, 0.2]]
        dummy_dataframes = [pandas.DataFrame({'scenario_name': ['test_1', 'test_2', 'test_3'], 'log_name': ['dummy', 'dummy', 'dummy_2'], 'scenario_type': ['unknown', 'ego_stop_at_stop_line', 'unknown'], 'planner_name': ['simple_planner', 'dummy_planner', 'dummy_planner'], 'metric_score': self.metric_scores[0], 'metric_score_unit': 'float'}), pandas.DataFrame({'scenario_name': ['test_1', 'test_3'], 'log_name': ['dummy', 'dummy_3'], 'scenario_type': ['unknown', 'unknown'], 'planner_name': ['simple_planner', 'dummy_planner'], 'metric_score': self.metric_scores[1], 'metric_score_unit': 'float'})]
        metric_statistic_names = ['dummy_metric', 'second_dummy_metric']
        self.metric_statistic_dataframes = []
        for dummy_dataframe, metric_statistic_name in zip(dummy_dataframes, metric_statistic_names):
            self.metric_statistic_dataframes.append(MetricStatisticsDataFrame(metric_statistic_name=metric_statistic_name, metric_statistics_dataframe=dummy_dataframe))
        self.tmpdir = tempfile.TemporaryDirectory()
        self.weighted_average_metric_aggregator = WeightedAverageMetricAggregator(name='weighted_average_metric_aggregator', metric_weights={'default': 1.0, 'dummy_metric': 0.5}, file_name='test_weighted_average_metric_aggregator.parquet', aggregator_save_path=Path(self.tmpdir.name), multiple_metrics=[])

    def tearDown(self) -> None:
        """Clean up when unittests end."""
        self.tmpdir.cleanup()

    def test_name(self) -> None:
        """Test if name is expected."""
        self.assertEqual('weighted_average_metric_aggregator', self.weighted_average_metric_aggregator.name)

    def test_final_metric_score(self) -> None:
        """Test if final metric score is expected."""
        self.assertEqual(None, self.weighted_average_metric_aggregator.final_metric_score)

    def test_aggregated_metric_dataframe(self) -> None:
        """Test if aggregated metric dataframe is expected."""
        self.assertEqual(None, self.weighted_average_metric_aggregator.aggregated_metric_dataframe)

    def test_aggregation(self) -> None:
        """Test running the aggregation."""
        metric_dataframes = {metric_statistic_dataframe.metric_statistic_name: metric_statistic_dataframe for metric_statistic_dataframe in self.metric_statistic_dataframes}
        self.weighted_average_metric_aggregator(metric_dataframes=metric_dataframes)
        parquet_file = Path(self.tmpdir.name) / 'test_weighted_average_metric_aggregator.parquet'
        self.assertTrue(parquet_file.exists())
        self.weighted_average_metric_aggregator.read_parquet()
        aggregated_metric_dataframe = self.weighted_average_metric_aggregator.aggregated_metric_dataframe
        self.assertIsNot(aggregated_metric_dataframe, None)
        self.assertTrue(len(aggregated_metric_dataframe))
        self.assertTrue(np.isnan(aggregated_metric_dataframe['second_dummy_metric'][0]))
        expected_planners = ['dummy_planner', 'simple_planner']
        self.assertEqual(expected_planners, sorted(aggregated_metric_dataframe['planner_name'].unique(), reverse=False))
        self.assertEqual(['weighted_average'], list(aggregated_metric_dataframe['aggregator_type'].unique()))
        expected_values = {'dummy_planner': {'dummy_metric': [0.5, 0.8, 0.5, 0.8, 0.65], 'second_dummy_metric': [-1.0, 0.2, -1.0, 0.2, 0.1], 'score': [0.5, 0.4, 0.5, 0.4, 0.45]}, 'simple_planner': {'dummy_metric': [1.0, 1.0, 1.0], 'second_dummy_metric': [0.1, 0.1, 0.1], 'score': [0.4, 0.4, 0.4]}}
        for planner in expected_planners:
            planner_metric = aggregated_metric_dataframe[aggregated_metric_dataframe['planner_name'].isin([planner])]
            for name, expected_value in expected_values[planner].items():
                planner_values = np.round(planner_metric[name].fillna(-1.0).to_numpy(), 2).tolist()
                self.assertEqual(expected_value, planner_values)

    def test_parquet(self) -> None:
        """Test property."""
        self.assertEqual(self.weighted_average_metric_aggregator.parquet_file, self.weighted_average_metric_aggregator._parquet_file)

def save_runner_reports(reports: List[RunnerReport], output_dir: Path, report_name: str) -> None:
    """
    Save runner reports to a parquet file in the output directory.
    Output directory can be local or s3.
    :param reports: Runner reports returned from each simulation.
    :param output_dir: Output directory to save the report.
    :param report_name: Report name.
    """
    report_dicts = []
    for report in map(lambda x: x.__dict__, reports):
        if (planner_report := report['planner_report']) is not None:
            planner_report_statistics = planner_report.compute_summary_statistics()
            del report['planner_report']
            report.update(planner_report_statistics)
        report_dicts.append(report)
    df = pd.DataFrame(report_dicts)
    df['duration'] = df['end_time'] - df['start_time']
    save_path = output_dir / report_name
    df.to_parquet(safe_path_to_string(save_path))
    logger.info(f'Saved runner reports to {save_path}')

def build_main_multi_callback(cfg: DictConfig) -> MultiMainCallback:
    """
    Build a multi main callback.
    :param cfg: Configuration that is used to run the experiment.
    """
    logger.info('Building MultiMainCallback...')
    main_callbacks = []
    for callback_name, config in cfg.main_callback.items():
        if is_target_type(config, MetricAggregatorCallback):
            metric_aggregators = build_metrics_aggregators(cfg)
            callback: MetricAggregatorCallback = instantiate(config, metric_aggregators=metric_aggregators)
        else:
            callback = instantiate(config)
        validate_type(callback, AbstractMainCallback)
        main_callbacks.append(callback)
    multi_main_callback = MultiMainCallback(main_callbacks)
    logger.info(f'Building MultiMainCallback: {len(multi_main_callback)}...DONE!')
    return multi_main_callback

def build_simulation_experiment_folder(cfg: DictConfig) -> str:
    """
    Builds the main experiment folder for simulation.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: The main experiment folder path.
    """
    logger.info('Building experiment folders...')
    main_exp_folder = pathlib.Path(cfg.output_dir)
    logger.info(f'\n\n\tFolder where all results are stored: {main_exp_folder}\n')
    if not is_s3_path(main_exp_folder):
        main_exp_folder.mkdir(parents=True, exist_ok=True)
    if 'simulation_log_main_path' in cfg and cfg.simulation_log_main_path is not None:
        exp_folder = pathlib.Path(cfg.simulation_log_main_path)
        logger.info(f'\n\n\tUsing previous simulation logs: {exp_folder}\n')
        if not path_exists(exp_folder):
            raise FileNotFoundError(f'{exp_folder} does not exist.')
    else:
        exp_folder = main_exp_folder
    if 'simulation_log_callback' in cfg.callback:
        simulation_folder = cfg.callback.simulation_log_callback.simulation_log_dir
    else:
        simulation_folder = None
    metric_main_path = main_exp_folder / cfg.metric_dir
    if not is_s3_path(metric_main_path):
        metric_main_path.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get('NODE_RANK', 0)) == 0:
        nuboard_filename = main_exp_folder / (f'nuboard_{int(time.time())}' + NuBoardFile.extension())
        nuboard_file = NuBoardFile(simulation_main_path=safe_path_to_string(exp_folder), simulation_folder=simulation_folder, metric_main_path=safe_path_to_string(exp_folder), metric_folder=cfg.metric_dir, aggregator_metric_folder=cfg.aggregator_metric_dir)
        nuboard_file.save_nuboard_file(nuboard_filename)
    logger.info('Building experiment folders...DONE!')
    return exp_folder.name

def run_runners(runners: List[AbstractRunner], common_builder: CommonBuilder, profiler_name: str, cfg: DictConfig) -> None:
    """
    Run a list of runners.
    :param runners: A list of runners.
    :param common_builder: Common builder.
    :param profiler_name: Profiler name.
    :param cfg: Hydra config.
    """
    assert len(runners) > 0, 'No scenarios found to simulate!'
    if common_builder.profiler:
        common_builder.profiler.start_profiler(profiler_name)
    logger.info('Executing runners...')
    reports = execute_runners(runners=runners, worker=common_builder.worker, num_gpus=cfg.number_of_gpus_allocated_per_simulation, num_cpus=cfg.number_of_cpus_allocated_per_simulation, exit_on_failure=cfg.exit_on_failure, verbose=cfg.verbose)
    logger.info('Finished executing runners!')
    save_runner_reports(reports, common_builder.output_dir, cfg.runner_report_file)
    distributed_sync(Path(cfg.output_dir / Path('barrier')), cfg.distributed_timeout_seconds)
    if int(os.environ.get('NODE_RANK', 0)) == 0:
        common_builder.multi_main_callback.on_run_simulation_end()
    if common_builder.profiler:
        common_builder.profiler.save_profiler(profiler_name)

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Calls run_simulation to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    """
    assert cfg.simulation_log_main_path is None, 'Simulation_log_main_path must not be set when running simulation.'
    run_simulation(cfg=cfg)
    if is_s3_path(Path(cfg.output_dir)):
        clean_up_s3_artifacts()

def clean_up_s3_artifacts() -> None:
    """
    Cleanup lingering s3 artifacts that are written locally.
    This happens because some minor write-to-s3 functionality isn't yet implemented.
    """
    working_path = os.getcwd()
    s3_dirname = 's3:'
    s3_ind = working_path.find(s3_dirname)
    if s3_ind != -1:
        local_s3_path = working_path[:working_path.find(s3_dirname) + len(s3_dirname)]
        rmtree(local_s3_path)

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Execute metric aggregators with the simulation path.
    :param cfg: Hydra config dict.
    """
    cfg.scenario_metric_paths = cfg.scenario_metric_paths or []
    metric_summary_callbacks = []
    challenge_metric_save_paths = []
    for challenge in cfg.challenges:
        challenge_save_path = Path(cfg.output_dir) / cfg.metric_folder_name / challenge
        challenge_metric_save_paths.append(challenge_save_path)
        if not challenge_save_path.exists():
            challenge_save_path.mkdir(exist_ok=True, parents=True)
        if cfg.scenario_metric_paths:
            challenge_metric_paths = [path for path in cfg.scenario_metric_paths if challenge in path]
            metric_file_callback = MetricFileCallback(scenario_metric_paths=challenge_metric_paths, metric_file_output_path=str(challenge_save_path), delete_scenario_metric_files=cfg.delete_scenario_metric_files)
            metric_file_callback.on_run_simulation_end()
    metric_output_path = Path(cfg.output_dir) / cfg.metric_folder_name
    metric_summary_output_path = str(Path(cfg.output_dir) / 'summary')
    if cfg.enable_metric_summary:
        if not challenge_metric_save_paths:
            challenge_metric_save_paths.append(metric_output_path)
        for challenge_metric_save_path in challenge_metric_save_paths:
            file_name = challenge_metric_save_path.stem if challenge_metric_save_path.stem in cfg.challenges else 'summary'
            pdf_file_name = file_name + '.pdf'
            metric_summary_callbacks.append(MetricSummaryCallback(metric_save_path=challenge_metric_save_path, metric_aggregator_save_path=cfg.aggregator_save_path, summary_output_path=metric_summary_output_path, pdf_file_name=pdf_file_name))
    metric_aggregators = build_metrics_aggregators(cfg)
    metric_aggregator_callback = MetricAggregatorCallback(metric_save_path=str(metric_output_path), metric_aggregators=metric_aggregators)
    metric_aggregator_callback.on_run_simulation_end()
    for metric_summary_callback in metric_summary_callbacks:
        metric_summary_callback.on_run_simulation_end()

def build_metrics_aggregators(cfg: DictConfig) -> List[AbstractMetricAggregator]:
    """
    Build a list of metric aggregators.
    :param cfg: Config
    :return A list of metric aggregators, and the path in which they will  save the results
    """
    metric_aggregators = []
    metric_aggregator_configs = cfg.metric_aggregator
    aggregator_save_path = Path(cfg.aggregator_save_path)
    if not is_s3_path(aggregator_save_path):
        aggregator_save_path.mkdir(exist_ok=True, parents=True)
    for metric_aggregator_config_name, metric_aggregator_config in metric_aggregator_configs.items():
        metric_aggregators.append(instantiate(metric_aggregator_config, aggregator_save_path=aggregator_save_path))
    return metric_aggregators

class TestRunParallelWorker(SkeletonTestSimulation):
    """Test running parallel workers in simulation."""

    def test_worker_parallel(self) -> None:
        """
        Sanity test parallel worker.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, 'worker=single_machine_thread_pool', 'scenario_filter.limit_total_scenarios=2', "selected_simulation_metrics='[ego_acceleration_statistics, ego_jerk_statistics]'", '+simulation=open_loop_boxes'])
            main(cfg)

class TestRunRayWorker(SkeletonTestSimulation):
    """Test running ray workers in simulation."""

    def test_ray_worker(self) -> None:
        """
        Sanity test for ray worker.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, 'worker=ray_distributed', 'worker.debug_mode=true', 'scenario_filter.limit_total_scenarios=2', "selected_simulation_metrics='[ego_acceleration_statistics, ego_jerk_statistics]'", '+simulation=open_loop_boxes'])
            main(cfg)

class TestRunSequentialWorker(SkeletonTestSimulation):
    """Test running sequential workers in simulation."""

    def test_worker_sequential(self) -> None:
        """
        Sanity test for sequential worker.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, 'worker=sequential', "selected_simulation_metrics='[ego_acceleration_statistics, ego_jerk_statistics]'", '+simulation=open_loop_boxes'])
            main(cfg)

class TestRunChallenge(SkeletonTestSimulation):
    """Test main simulation entry point across different challenges."""

    def test_simulation_challenge_1(self) -> None:
        """
        Sanity check for challenge 1 simulation.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, 'worker=single_machine_thread_pool', 'worker.use_process_pool=true', '+simulation=open_loop_boxes'])
            main(cfg)

