# Cluster 137

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

def build_scenario_builder(cfg: DictConfig) -> AbstractScenarioBuilder:
    """
    Builds scenario builder.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of scenario builder.
    """
    logger.info('Building AbstractScenarioBuilder...')
    scenario_builder = instantiate(cfg.scenario_builder)
    validate_type(scenario_builder, AbstractScenarioBuilder)
    logger.info('Building AbstractScenarioBuilder...DONE!')
    return scenario_builder

def build_scenario_filter(cfg: DictConfig) -> ScenarioFilter:
    """
    Builds the scenario filter.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param db: dabatase.
    :return: Instance of ScenarioFilter.
    """
    logger.info('Building ScenarioFilter...')
    if cfg.scenario_tokens and (not all(map(is_valid_token, cfg.scenario_tokens))):
        raise RuntimeError('Expected all scenario tokens to be 16-character strings. Your shell may strip quotes causing hydra to parse a token as a float, so consider passing them like scenario_filter.scenario_tokens=\'["595322e649225137", ...]\'')
    scenario_filter: ScenarioFilter = instantiate(cfg)
    validate_type(scenario_filter, ScenarioFilter)
    logger.info('Building ScenarioFilter...DONE!')
    return scenario_filter

class TestDistributedScenarioFilter(unittest.TestCase):
    """
    Test the distributed scenario filter that is intended to be used to split work across multiple nodes
    """

    def setUp(self) -> None:
        """
        Build some useful mocks to use in a variety of functions
        """
        self.scenario_builder_mock = MagicMock(AbstractScenarioBuilder)
        self.mock_scenarios = [MagicMock(AbstractScenario), MagicMock(AbstractScenario)]
        self.scenario_builder_mock.get_scenarios = MagicMock()
        self.scenario_builder_mock.get_scenarios.return_value = self.mock_scenarios
        self.build_scenario_builder_mock = MagicMock()
        self.build_scenario_builder_mock.return_value = self.scenario_builder_mock
        self.scenario_filter_mock = MagicMock(ScenarioFilter)
        self.build_scenario_filter_mock = MagicMock()
        self.build_scenario_filter_mock.return_value = self.scenario_filter_mock
        self.mock_dbs = ['file_1', 'file_2']
        self.cfg_mock = MagicMock()
        self.cfg_mock.scenario_builder = MagicMock()
        self.cfg_mock.scenario_builder.db_files = self.mock_dbs
        self.cfg_mock.scenario_filter = MagicMock()
        self.mock_scenarios[0].token = 'a'
        self.mock_scenarios[0].log_name = '1.log'
        self.mock_scenarios[1].token = 'b'
        self.mock_scenarios[1].log_name = '2.log'
        self.worker_mock = MagicMock(WorkerPool)
        self.dist_filter_get_scenarios = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 2, 'path')
        self.dist_filter_get_scenarios._get_log_db_files_for_single_node = MagicMock()
        self.dist_filter_get_scenarios._get_scenarios_from_list_of_log_files = MagicMock()
        self.dist_filter_get_scenarios._get_repartition_tokens = MagicMock()
        self.dist_filter_get_scenarios._get_repartition_tokens.return_value = (['a'], ['1.log'])

    def test_get_scenarios_scenario_based(self) -> None:
        """
        Test that get_scenarios does full repartitioning in this case
        """
        with unittest.mock.patch('nuplan.common.utils.distributed_scenario_filter.build_scenario_builder', self.build_scenario_builder_mock), unittest.mock.patch('nuplan.common.utils.distributed_scenario_filter.build_scenario_filter', self.build_scenario_filter_mock), unittest.mock.patch('nuplan.common.utils.distributed_scenario_filter.OmegaConf.set_struct'):
            self.dist_filter_get_scenarios._distributed_mode = DistributedMode.SCENARIO_BASED
            scenarios = self.dist_filter_get_scenarios.get_scenarios()
            self.assertEqual(self.mock_scenarios, scenarios)
            self.dist_filter_get_scenarios._get_log_db_files_for_single_node.assert_called()
            self.dist_filter_get_scenarios._get_scenarios_from_list_of_log_files.assert_called()
            self.dist_filter_get_scenarios._get_repartition_tokens.assert_called()
            self.assertListEqual(self.cfg_mock.scenario_filter.scenario_tokens, ['a'])
            self.assertListEqual(self.cfg_mock.scenario_builder.db_files, ['1.log'])
            self.build_scenario_builder_mock.assert_called_with(cfg=self.cfg_mock)
            self.build_scenario_filter_mock.assert_called_with(cfg=self.cfg_mock.scenario_filter)

    def test_get_scenarios_multiple_nodes_log_file_mode(self) -> None:
        """
        Test that get_scenarios we only call the methods that get a chunk of log files + gets the scenarios from that chunk
        """
        with unittest.mock.patch('nuplan.common.utils.distributed_scenario_filter.build_scenario_builder', self.build_scenario_builder_mock), unittest.mock.patch('nuplan.common.utils.distributed_scenario_filter.build_scenario_filter', self.build_scenario_filter_mock):
            self.dist_filter_get_scenarios._distributed_mode = DistributedMode.LOG_FILE_BASED
            mock_scenarios = [MagicMock()]
            self.dist_filter_get_scenarios._get_scenarios_from_list_of_log_files.return_value = mock_scenarios
            scenarios = self.dist_filter_get_scenarios.get_scenarios()
            self.assertEqual(mock_scenarios, scenarios)
            self.dist_filter_get_scenarios._get_log_db_files_for_single_node.assert_called()
            self.dist_filter_get_scenarios._get_scenarios_from_list_of_log_files.assert_called()
            self.dist_filter_get_scenarios._get_repartition_tokens.assert_not_called()
            self.build_scenario_builder_mock.assert_not_called()
            self.build_scenario_filter_mock.assert_not_called()

    def test_get_scenarios_single_node(self) -> None:
        """
        Test that get_scenarios just returns the scenarios built by the scenario builder in this case.
        """
        with unittest.mock.patch('nuplan.common.utils.distributed_scenario_filter.build_scenario_builder', self.build_scenario_builder_mock), unittest.mock.patch('nuplan.common.utils.distributed_scenario_filter.build_scenario_filter', self.build_scenario_filter_mock):
            self.dist_filter_get_scenarios._distributed_mode = DistributedMode.SINGLE_NODE
            scenarios = self.dist_filter_get_scenarios.get_scenarios()
            self.assertEqual(self.mock_scenarios, scenarios)
            self.dist_filter_get_scenarios._get_log_db_files_for_single_node.assert_not_called()
            self.dist_filter_get_scenarios._get_scenarios_from_list_of_log_files.assert_not_called()
            self.dist_filter_get_scenarios._get_repartition_tokens.assert_not_called()
            self.build_scenario_builder_mock.assert_called_with(cfg=self.cfg_mock)
            self.build_scenario_filter_mock.assert_called_with(cfg=self.cfg_mock.scenario_filter)

    def test_get_repartition_tokens(self) -> None:
        """
        Test that we make all of the expected calls, in the expected order, to repartition the tokens.
        """
        with unittest.mock.patch('nuplan.common.utils.distributed_scenario_filter.get_unique_job_id') as id, unittest.mock.patch('nuplan.common.utils.distributed_scenario_filter.distributed_sync') as dist:
            dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 1, 'path', timeout_seconds=5)
            dist_filter._write_token_csv_file = MagicMock()
            dist_filter._get_all_generated_csv = MagicMock()
            dist_filter._get_token_and_log_chunk_on_single_node = MagicMock()
            id.return_value = '1'
            dist_filter._get_all_generated_csv.return_value = [('a', '1'), ('b', '2')]
            dist_filter._get_token_and_log_chunk_on_single_node.return_value = (['a', 'b'], ['path/1.db', 'path/2.db'])
            manager = Mock()
            manager.attach_mock(dist_filter._write_token_csv_file, 'write_csv')
            manager.attach_mock(dist, 'sync')
            manager.attach_mock(dist_filter._get_all_generated_csv, 'get_csvs')
            manager.attach_mock(dist_filter._get_token_and_log_chunk_on_single_node, 'chunk')
            output = dist_filter._get_repartition_tokens(scenarios=self.mock_scenarios)
            self.assertEqual(output, (['a', 'b'], ['path/1.db', 'path/2.db']))
            expected_calls = [call.write_csv(self.mock_scenarios, Path('path/tokens/1')), call.sync(Path('path/barrier/1'), timeout_seconds=5), call.get_csvs(Path('path/tokens/1')), call.chunk([('a', '1'), ('b', '2')], Path('.'))]
            self.assertListEqual(manager.mock_calls, expected_calls)

    def test_get_all_generated_csv_s3(self) -> None:
        """
        Test that we get all of the tokens from the csv files we have created when running in mocked s3.
        """
        with unittest.mock.patch('nuplan.common.utils.distributed_scenario_filter.expand_s3_dir') as expand, unittest.mock.patch('nuplan.common.utils.distributed_scenario_filter.split_s3_path') as split, unittest.mock.patch('nuplan.common.utils.distributed_scenario_filter.S3Store') as store:
            with tempfile.TemporaryDirectory() as tmp_dir_str:
                dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 1, 's3://dummy/path')
                dist_filter._write_token_csv_file(self.mock_scenarios, tmp_dir_str)
                split.return_value = ('bucket', 'file')
                expand.return_value = [os.path.join(tmp_dir_str, '0.csv')]

                def mock_get(path: str) -> IO[str]:
                    """
                    Mock get for the s3 store we mock, just opens the file as a local file.
                    """
                    return open(path)
                store.return_value = MagicMock()
                store.return_value.get = mock_get
                filter_output = dist_filter._get_all_generated_csv('s3://dummy/path')
                self.assertEqual(filter_output, [['a', '1.log'], ['b', '2.log']])

    def test_get_all_generated_csv_local(self) -> None:
        """
        Test that we get all of the tokens from the csv files we have created when running locally.
        """
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 2, tmp_dir_str)
            dist_filter_2 = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 1, 2, tmp_dir_str)
            dist_filter._write_token_csv_file(self.mock_scenarios[:1], tmp_dir_str)
            dist_filter_2._write_token_csv_file(self.mock_scenarios[1:], tmp_dir_str)
            filter_1_output = dist_filter._get_all_generated_csv(tmp_dir_str)
            filter_2_output = dist_filter_2._get_all_generated_csv(tmp_dir_str)
            self.assertListEqual(filter_1_output, filter_2_output)
            expected_token_set = {('a', '1.log'), ('b', '2.log')}
            self.assertEqual(len(filter_1_output), len(expected_token_set))
            self.assertSetEqual({tuple(i) for i in filter_1_output}, expected_token_set)

    def test_get_token_and_log_chunk_on_single_node(self) -> None:
        """
        Test that we correctly chunk the tokens and associated log names on each node.
        """
        with unittest.mock.patch('nuplan.common.utils.distributed_scenario_filter.check_s3_path_exists'):
            db_files_path = Path('s3://dummy/path')
            token_distribution = [('a', '1'), ('b', '1'), ('c', '2'), ('d', '2')]
            dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 1, '')
            tokens, log_files = dist_filter._get_token_and_log_chunk_on_single_node(token_distribution, db_files_path)
            self.assertSetEqual(set(tokens), {'a', 'b', 'c', 'd'})
            self.assertSetEqual(set(log_files), {'s3://dummy/path/1.db', 's3://dummy/path/2.db'})
            dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 2, '')
            tokens, log_files = dist_filter._get_token_and_log_chunk_on_single_node(token_distribution, db_files_path)
            self.assertSetEqual(set(tokens), {'a', 'b'})
            self.assertSetEqual(set(log_files), {'s3://dummy/path/1.db'})

    def test_write_token_csv_file(self) -> None:
        """
        Test that we correctly write out a csv file for the current node for the list of scenarios provided
        """
        dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 1, '')
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            dist_filter._write_token_csv_file(self.mock_scenarios, tmp_dir_str)
            expected_path = os.path.join(tmp_dir_str, '0.csv')
            self.assertTrue(os.path.exists(expected_path))
            csv_out = pd.read_csv(expected_path).to_dict()
            self.assertEqual(csv_out, {'0': {0: 'a', 1: 'b'}, '1': {0: '1.log', 1: '2.log'}})

    def test_get_scenarios_from_list_of_log_files(self) -> None:
        """
        Test that we build a scenario builder with the proper db files updated, and successfully get scenarios from it
        """
        dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 1, '')
        with unittest.mock.patch('nuplan.common.utils.distributed_scenario_filter.build_scenario_builder', self.build_scenario_builder_mock), unittest.mock.patch('nuplan.common.utils.distributed_scenario_filter.build_scenario_filter', self.build_scenario_filter_mock):
            scenarios = dist_filter._get_scenarios_from_list_of_log_files(['file_3'])
            self.assertListEqual(self.cfg_mock.scenario_builder.db_files, ['file_3'])
            self.build_scenario_filter_mock.assert_called_with(self.cfg_mock.scenario_filter)
            self.build_scenario_builder_mock.assert_called_with(self.cfg_mock)
            self.scenario_builder_mock.get_scenarios.assert_called_with(self.scenario_filter_mock, self.worker_mock)
            self.assertEqual(scenarios, self.mock_scenarios)

    def test_get_log_db_files_for_single_node_non_distributed(self) -> None:
        """
        Test that in a non-distributed context we simply return all the db files in the config
        """
        dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 1, '')
        logs = dist_filter._get_log_db_files_for_single_node()
        self.assertListEqual(logs, self.mock_dbs)

    def test_get_log_db_files_for_single_node_distributed(self) -> None:
        """
        Test that in a distributed context we call the proper functions and chunk the data as expected
        """
        with unittest.mock.patch('nuplan.common.utils.distributed_scenario_filter.get_db_filenames_from_load_path') as get, unittest.mock.patch('nuplan.common.utils.distributed_scenario_filter.check_s3_path_exists') as check:
            get.side_effect = lambda x: x
            check.return_value = True
            dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 2, '')
            logs = dist_filter._get_log_db_files_for_single_node()
            self.assertListEqual(logs, self.mock_dbs[:1])

def save_cache_metadata(cache_metadata_entries: List[CacheMetadataEntry], cache_path: Path, node_id: int) -> None:
    """
    Saves list of CacheMetadataEntry to output csv file path.
    :param cache_metadata_entries: List of metadata objects for cached features.
    :param cache_path: Path to s3 cache.
    :param node_id: Node ID of a node used for differentiating between nodes in multi-node caching.
    """
    cache_metadata_entries_dicts = [asdict(entry) for entry in cache_metadata_entries]
    cache_name = cache_path.name
    using_s3_cache_path = str(cache_path).startswith('s3:/')
    sanitized_cache_path = safe_path_to_string(cache_path)
    cache_metadata_storage_path = os.path.join(sanitized_cache_path, 'metadata', f'{cache_name}_metadata_node_{node_id}.csv')
    if not using_s3_cache_path:
        Path(cache_metadata_storage_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f'Using cache_metadata_storage_path: {cache_metadata_storage_path}')
    pd.DataFrame(cache_metadata_entries_dicts).to_csv(cache_metadata_storage_path, index=False)

def build_torch_module_wrapper(cfg: DictConfig) -> TorchModuleWrapper:
    """
    Builds the NN module.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of TorchModuleWrapper.
    """
    logger.info('Building TorchModuleWrapper...')
    model = instantiate(cfg)
    validate_type(model, TorchModuleWrapper)
    logger.info('Building TorchModuleWrapper...DONE!')
    return model

def build_scenarios_from_config(cfg: DictConfig, scenario_builder: AbstractScenarioBuilder, worker: WorkerPool) -> List[AbstractScenario]:
    """
    Build scenarios from config file.
    :param cfg: Omegaconf dictionary
    :param scenario_builder: Scenario builder.
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: A list of scenarios
    """
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    return scenario_builder.get_scenarios(scenario_filter, worker)

def cache_data(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Build the lightning datamodule and cache all samples.
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    """
    assert cfg.cache.cache_path is not None, f'Cache path cannot be None when caching, got {cfg.cache.cache_path}'
    scenario_builder = build_scenario_builder(cfg)
    if int(os.environ.get('NUM_NODES', 1)) > 1 and cfg.distribute_by_scenario:
        repartition_strategy = scenario_builder.repartition_strategy
        if repartition_strategy == RepartitionStrategy.REPARTITION_FILE_DISK:
            scenario_filter = DistributedScenarioFilter(cfg=cfg, worker=worker, node_rank=int(os.environ.get('NODE_RANK', 0)), num_nodes=int(os.environ.get('NUM_NODES', 1)), synchronization_path=cfg.cache.cache_path, timeout_seconds=cfg.get('distributed_timeout_seconds', 3600), distributed_mode=cfg.get('distributed_mode', DistributedMode.LOG_FILE_BASED))
            scenarios = scenario_filter.get_scenarios()
        elif repartition_strategy == RepartitionStrategy.INLINE:
            scenarios = build_scenarios_from_config(cfg, scenario_builder, worker)
            num_nodes = int(os.environ.get('NUM_NODES', 1))
            node_id = int(os.environ.get('NODE_RANK', 0))
            scenarios = chunk_list(scenarios, num_nodes)[node_id]
        else:
            expected_repartition_strategies = [e.value for e in RepartitionStrategy]
            raise ValueError(f'Expected repartition strategy to be in {expected_repartition_strategies}, got {repartition_strategy}.')
    else:
        logger.debug("Building scenarios without distribution, if you're running on a multi-node system, make sure you aren'taccidentally caching each scenario multiple times!")
        scenarios = build_scenarios_from_config(cfg, scenario_builder, worker)
    data_points = [{'scenario': scenario, 'cfg': cfg} for scenario in scenarios]
    logger.info('Starting dataset caching of %s files...', str(len(data_points)))
    cache_results = worker_map(worker, cache_scenarios, data_points)
    num_success = sum((result.successes for result in cache_results))
    num_fail = sum((result.failures for result in cache_results))
    num_total = num_success + num_fail
    logger.info('Completed dataset caching! Failed features and targets: %s out of %s', str(num_fail), str(num_total))
    cached_metadata = [cache_metadata_entry for cache_result in cache_results for cache_metadata_entry in cache_result.cache_metadata if cache_metadata_entry is not None]
    node_id = int(os.environ.get('NODE_RANK', 0))
    logger.info(f'Node {node_id}: Storing metadata csv file containing cache paths for valid features and targets...')
    save_cache_metadata(cached_metadata, Path(cfg.cache.cache_path), node_id)
    logger.info('Done storing metadata csv file.')

def build_training_engine(cfg: DictConfig, worker: WorkerPool) -> TrainingEngine:
    """
    Build the three core lightning modules: LightningDataModule, LightningModule and Trainer
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: TrainingEngine
    """
    logger.info('Building training engine...')
    torch_module_wrapper = build_torch_module_wrapper(cfg.model)
    datamodule = build_lightning_datamodule(cfg, worker, torch_module_wrapper)
    if cfg.lightning.trainer.params.accelerator == 'ddp':
        cfg = scale_cfg_for_distributed_training(cfg, datamodule=datamodule, worker=worker)
    else:
        logger.info(f'Updating configs based on {cfg.lightning.trainer.params.accelerator} strategy is currently not supported. Optimizer and LR Scheduler configs will not be updated.')
    model = build_lightning_module(cfg, torch_module_wrapper)
    trainer = build_trainer(cfg)
    engine = TrainingEngine(trainer=trainer, datamodule=datamodule, model=model)
    return engine

def build_lightning_datamodule(cfg: DictConfig, worker: WorkerPool, model: TorchModuleWrapper) -> pl.LightningDataModule:
    """
    Build the lightning datamodule from the config.
    :param cfg: Omegaconf dictionary.
    :param model: NN model used for training.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :return: Instantiated datamodule object.
    """
    feature_builders = model.get_list_of_required_feature()
    target_builders = model.get_list_of_computed_target()
    splitter = build_splitter(cfg.splitter)
    feature_preprocessor = FeaturePreprocessor(cache_path=cfg.cache.cache_path, force_feature_computation=cfg.cache.force_feature_computation, feature_builders=feature_builders, target_builders=target_builders)
    augmentors = build_agent_augmentor(cfg.data_augmentation) if 'data_augmentation' in cfg else None
    scenarios = build_scenarios(cfg, worker, model)
    datamodule: pl.LightningDataModule = DataModule(feature_preprocessor=feature_preprocessor, splitter=splitter, all_scenarios=scenarios, dataloader_params=cfg.data_loader.params, augmentors=augmentors, worker=worker, scenario_type_sampling_weights=cfg.scenario_type_weights.scenario_type_sampling_weights, **cfg.data_loader.datamodule)
    return datamodule

def build_lightning_module(cfg: DictConfig, torch_module_wrapper: TorchModuleWrapper) -> pl.LightningModule:
    """
    Builds the lightning module from the config.
    :param cfg: omegaconf dictionary
    :param torch_module_wrapper: NN model used for training
    :return: built object.
    """
    objectives = build_objectives(cfg)
    metrics = build_training_metrics(cfg)
    model = LightningModuleWrapper(model=torch_module_wrapper, objectives=objectives, metrics=metrics, batch_size=cfg.data_loader.params.batch_size, optimizer=cfg.optimizer, lr_scheduler=cfg.lr_scheduler if 'lr_scheduler' in cfg else None, warm_up_lr_scheduler=cfg.warm_up_lr_scheduler if 'warm_up_lr_scheduler' in cfg else None, objective_aggregate_mode=cfg.objective_aggregate_mode)
    return cast(pl.LightningModule, model)

def build_trainer(cfg: DictConfig) -> pl.Trainer:
    """
    Builds the lightning trainer from the config.
    :param cfg: omegaconf dictionary
    :return: built object.
    """
    params = cfg.lightning.trainer.params
    callbacks = build_callbacks(cfg)
    plugins = [pl.plugins.DDPPlugin(find_unused_parameters=False, num_nodes=params.num_nodes)]
    loggers = [pl.loggers.TensorBoardLogger(save_dir=cfg.group, name=cfg.experiment, log_graph=False, version='', prefix='')]
    if cfg.lightning.trainer.overfitting.enable:
        OmegaConf.set_struct(cfg, False)
        params = OmegaConf.merge(params, cfg.lightning.trainer.overfitting.params)
        params.check_val_every_n_epoch = params.max_epochs + 1
        OmegaConf.set_struct(cfg, True)
        return pl.Trainer(plugins=plugins, **params)
    if cfg.lightning.trainer.checkpoint.resume_training:
        output_dir = Path(cfg.output_dir)
        date_format = cfg.date_format
        OmegaConf.set_struct(cfg, False)
        last_checkpoint = extract_last_checkpoint_from_experiment(output_dir, date_format)
        if not last_checkpoint:
            raise ValueError('Resume Training is enabled but no checkpoint was found!')
        params.resume_from_checkpoint = str(last_checkpoint)
        latest_epoch = torch.load(last_checkpoint)['epoch']
        params.max_epochs += latest_epoch
        logger.info(f'Resuming at epoch {latest_epoch} from checkpoint {last_checkpoint}')
        OmegaConf.set_struct(cfg, True)
    trainer = pl.Trainer(callbacks=callbacks, plugins=plugins, logger=loggers, **params)
    return trainer

class SkeletonTestDataloader(unittest.TestCase):
    """
    Skeleton with initialized dataloader used in testing.
    """

    def setUp(self) -> None:
        """
        Set up basic configs.
        """
        pl.seed_everything(2022, workers=True)
        self.splitter = LogSplitter(log_splits={'train': ['2021.07.16.20.45.29_veh-35_01095_01486'], 'val': ['2021.06.07.18.53.26_veh-26_00005_00427'], 'test': ['2021.10.06.07.26.10_veh-52_00006_00398']})
        feature_builders = [DummyVectorMapBuilder(), VectorMapFeatureBuilder(radius=20), AgentsFeatureBuilder(TrajectorySampling(num_poses=4, time_horizon=1.5)), RasterFeatureBuilder(map_features={'LANE': 1, 'INTERSECTION': 1.0, 'STOP_LINE': 0.5, 'CROSSWALK': 0.5}, num_input_channels=4, target_width=224, target_height=224, target_pixel_size=0.5, ego_width=2.297, ego_front_length=4.049, ego_rear_length=1.127, ego_longitudinal_offset=0.0, baseline_path_thickness=1)]
        target_builders = [EgoTrajectoryTargetBuilder(TrajectorySampling(num_poses=10, time_horizon=5.0))]
        self.feature_preprocessor = FeaturePreprocessor(cache_path=None, force_feature_computation=True, feature_builders=feature_builders, target_builders=target_builders)
        self.scenario_filter = ScenarioFilter(scenario_types=None, scenario_tokens=None, log_names=None, map_names=None, num_scenarios_per_type=None, limit_total_scenarios=150, expand_scenarios=True, remove_invalid_goals=False, shuffle=True, timestamp_threshold_s=None, ego_displacement_minimum_m=None, ego_start_speed_threshold=None, ego_stop_speed_threshold=None, speed_noise_tolerance=None, token_set_path=None, fraction_in_token_set_threshold=None)
        self.augmentors = [KinematicAgentAugmentor(trajectory_length=10, dt=0.1, mean=[0.3, 0.1, np.pi / 12], std=[0.5, 0.1, np.pi / 12], low=[-0.2, 0.0, 0.0], high=[0.8, 0.2, np.pi / 6], augment_prob=0.5)]
        self.scenario_builder = get_test_nuplan_scenario_builder()

    def _test_dataloader(self, worker: WorkerPool) -> None:
        """
        Tests that the training dataloader can be iterated without errors
        """
        scenarios = self.scenario_builder.get_scenarios(self.scenario_filter, worker)
        self.assertGreater(len(scenarios), 0)
        batch_size = 4
        num_workers = 4
        scenario_type_sampling_weights = DictConfig({'enable': False, 'scenario_type_weights': {'unknown': 1.0}})
        datamodule = DataModule(feature_preprocessor=self.feature_preprocessor, splitter=self.splitter, train_fraction=1.0, val_fraction=0.1, test_fraction=0.1, all_scenarios=scenarios, augmentors=self.augmentors, worker=worker, scenario_type_sampling_weights=scenario_type_sampling_weights, dataloader_params={'batch_size': batch_size, 'num_workers': num_workers, 'drop_last': True})
        datamodule.setup('fit')
        self.assertGreater(len(datamodule.train_dataloader()), 0)
        for features, targets, scenarios in datamodule.train_dataloader():
            self.assertTrue('raster' in features.keys())
            self.assertTrue('vector_map' in features.keys())
            self.assertTrue('trajectory' in targets.keys())
            scenario_features: Raster = features['raster']
            trajectory_target: Trajectory = targets['trajectory']
            self.assertEqual(scenario_features.num_batches, trajectory_target.num_batches)
            self.assertIsInstance(scenario_features, Raster)
            self.assertIsInstance(trajectory_target, Trajectory)
            self.assertEqual(scenario_features.num_batches, batch_size)

    def tearDown(self) -> None:
        """
        Clean up.
        """
        if ray.is_initialized():
            ray.shutdown()

@dataclass
class VectorMap(AbstractModelFeature):
    """
    Vector map data struture, including:
        coords: List[<np.ndarray: num_lane_segments, 2, 2>].
            The (x, y) coordinates of the start and end point of the lane segments.
        lane_groupings: List[List[<np.ndarray: num_lane_segments_in_lane>]].
            Each lane grouping or polyline is represented by an array of indices of lane segments
            in coords belonging to the given lane. Each batch contains a List of lane groupings.
        multi_scale_connections: List[Dict of {scale: connections_of_scale}].
            Each connections_of_scale is represented by an array of <np.ndarray: num_connections, 2>,
            and each column in the array is [from_lane_segment_idx, to_lane_segment_idx].
        on_route_status: List[<np.ndarray: num_lane_segments, 2>].
            Binary encoding of on route status for lane segment at given index.
            Encoding: off route [0, 1], on route [1, 0], unknown [0, 0]
        traffic_light_data: List[<np.ndarray: num_lane_segments, 4>]
            One-hot encoding of on traffic light status for lane segment at given index.
            Encoding: green [1, 0, 0, 0] yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1]

    In all cases, the top level List represent number of batches. This is a special feature where
    each batch entry can have different size. Similarly, each lane grouping within a batch can have
    a variable number of elements. For that reason, the feature can not be placed to a single tensor,
    and we batch the feature with a custom `collate` function
    """
    coords: List[FeatureDataType]
    lane_groupings: List[List[FeatureDataType]]
    multi_scale_connections: List[Dict[int, FeatureDataType]]
    on_route_status: List[FeatureDataType]
    traffic_light_data: List[FeatureDataType]
    _lane_coord_dim: int = 2
    _on_route_status_encoding_dim: int = LaneOnRouteStatusData.encoding_dim()

    def __post_init__(self) -> None:
        """Sanitize attributes of the dataclass."""
        if len(self.coords) != len(self.multi_scale_connections):
            raise RuntimeError(f'Not consistent length of batches! {len(self.coords)} != {len(self.multi_scale_connections)}')
        if len(self.coords) != len(self.lane_groupings):
            raise RuntimeError(f'Not consistent length of batches! {len(self.coords)} != {len(self.lane_groupings)}')
        if len(self.coords) != len(self.on_route_status):
            raise RuntimeError(f'Not consistent length of batches! {len(self.coords)} != {len(self.on_route_status)}')
        if len(self.coords) != len(self.traffic_light_data):
            raise RuntimeError(f'Not consistent length of batches! {len(self.coords)} != {len(self.traffic_light_data)}')
        if len(self.coords) == 0:
            raise RuntimeError('Batch size has to be > 0!')
        for coords in self.coords:
            if coords.shape[1] != 2 or coords.shape[2] != 2:
                raise RuntimeError('The dimension of coords is not correct!')
        for coords, traffic_lights in zip(self.coords, self.traffic_light_data):
            if coords.shape[0] != traffic_lights.shape[0]:
                raise RuntimeError('Number of segments are inconsistent')

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return len(self.coords) > 0 and len(self.coords[0]) > 0 and (len(self.lane_groupings) > 0) and (len(self.lane_groupings[0]) > 0) and (len(self.lane_groupings[0][0]) > 0) and (len(self.on_route_status) > 0) and (len(self.on_route_status[0]) > 0) and (len(self.traffic_light_data) > 0) and (len(self.traffic_light_data[0]) > 0) and (len(self.multi_scale_connections) > 0) and (len(list(self.multi_scale_connections[0].values())[0]) > 0)

    @property
    def num_of_batches(self) -> int:
        """
        :return: number of batches
        """
        return len(self.coords)

    def num_lanes_in_sample(self, sample_idx: int) -> int:
        """
        :param sample_idx: sample index in batch
        :return: number of lanes represented by lane_groupings in sample
        """
        return len(self.lane_groupings[sample_idx])

    @classmethod
    def lane_coord_dim(cls) -> int:
        """
        :return: dimension of coords, should be 2 (x, y)
        """
        return cls._lane_coord_dim

    @classmethod
    def on_route_status_encoding_dim(cls) -> int:
        """
        :return: dimension of route following status encoding
        """
        return cls._on_route_status_encoding_dim

    @classmethod
    def flatten_lane_coord_dim(cls) -> int:
        """
        :return: dimension of flattened start and end coords, should be 4 = 2 x (x, y)
        """
        return 2 * cls._lane_coord_dim

    def get_lane_coords(self, sample_idx: int) -> FeatureDataType:
        """
        Retrieve lane coordinates at given sample index.
        :param sample_idx: the batch index of interest.
        :return: lane coordinate features.
        """
        return self.coords[sample_idx]

    @classmethod
    def collate(cls, batch: List[VectorMap]) -> VectorMap:
        """Implemented. See interface."""
        return VectorMap(coords=[data for sample in batch for data in sample.coords], lane_groupings=[data for sample in batch for data in sample.lane_groupings], multi_scale_connections=[data for sample in batch for data in sample.multi_scale_connections], on_route_status=[data for sample in batch for data in sample.on_route_status], traffic_light_data=[data for sample in batch for data in sample.traffic_light_data])

    def to_feature_tensor(self) -> VectorMap:
        """Implemented. See interface."""
        return VectorMap(coords=[to_tensor(coords).contiguous() for coords in self.coords], lane_groupings=[[to_tensor(lane_grouping).contiguous() for lane_grouping in lane_groupings] for lane_groupings in self.lane_groupings], multi_scale_connections=[{scale: to_tensor(connection).contiguous() for scale, connection in multi_scale_connections.items()} for multi_scale_connections in self.multi_scale_connections], on_route_status=[to_tensor(status).contiguous() for status in self.on_route_status], traffic_light_data=[to_tensor(data).contiguous() for data in self.traffic_light_data])

    def to_device(self, device: torch.device) -> VectorMap:
        """Implemented. See interface."""
        return VectorMap(coords=[coords.to(device=device) for coords in self.coords], lane_groupings=[[lane_grouping.to(device=device) for lane_grouping in lane_groupings] for lane_groupings in self.lane_groupings], multi_scale_connections=[{scale: connection.to(device=device) for scale, connection in multi_scale_connections.items()} for multi_scale_connections in self.multi_scale_connections], on_route_status=[status.to(device=device) for status in self.on_route_status], traffic_light_data=[data.to(device=device) for data in self.traffic_light_data])

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> VectorMap:
        """Implemented. See interface."""
        return VectorMap(coords=data['coords'], lane_groupings=data['lane_groupings'], multi_scale_connections=data['multi_scale_connections'], on_route_status=data['on_route_status'], traffic_light_data=data['traffic_light_data'])

    def unpack(self) -> List[VectorMap]:
        """Implemented. See interface."""
        return [VectorMap([coords], [lane_groupings], [multi_scale_connections], [on_route_status], [traffic_light_data]) for coords, lane_groupings, multi_scale_connections, on_route_status, traffic_light_data in zip(self.coords, self.lane_groupings, self.multi_scale_connections, self.on_route_status, self.traffic_light_data)]

    def rotate(self, quaternion: Quaternion) -> VectorMap:
        """
        Rotate the vector map.
        :param quaternion: Rotation to apply.
        """
        for coord in self.coords:
            validate_type(coord, np.ndarray)
        return VectorMap(coords=[rotate_coords(data, quaternion) for data in self.coords], lane_groupings=self.lane_groupings, multi_scale_connections=self.multi_scale_connections, on_route_status=self.on_route_status, traffic_light_data=self.traffic_light_data)

    def translate(self, translation_value: FeatureDataType) -> VectorMap:
        """
        Translate the vector map.
        :param translation_value: Translation in x, y, z.
        """
        assert translation_value.size == 3, 'Translation value must have dimension of 3 (x, y, z)'
        are_the_same_type(translation_value, self.coords[0])
        return VectorMap(coords=[translate_coords(coords, translation_value) for coords in self.coords], lane_groupings=self.lane_groupings, multi_scale_connections=self.multi_scale_connections, on_route_status=self.on_route_status, traffic_light_data=self.traffic_light_data)

    def scale(self, scale_value: FeatureDataType) -> VectorMap:
        """
        Scale the vector map.
        :param scale_value: <np.float: 3,>. Scale in x, y, z.
        """
        assert scale_value.size == 3, f'Scale value has incorrect dimension: {scale_value.size}!'
        are_the_same_type(scale_value, self.coords[0])
        return VectorMap(coords=[scale_coords(coords, scale_value) for coords in self.coords], lane_groupings=self.lane_groupings, multi_scale_connections=self.multi_scale_connections, on_route_status=self.on_route_status, traffic_light_data=self.traffic_light_data)

    def xflip(self) -> VectorMap:
        """
        Flip the vector map along the X-axis.
        """
        return VectorMap(coords=[xflip_coords(coords) for coords in self.coords], lane_groupings=self.lane_groupings, multi_scale_connections=self.multi_scale_connections, on_route_status=self.on_route_status, traffic_light_data=self.traffic_light_data)

    def yflip(self) -> VectorMap:
        """
        Flip the vector map along the Y-axis.
        """
        return VectorMap(coords=[yflip_coords(coords) for coords in self.coords], lane_groupings=self.lane_groupings, multi_scale_connections=self.multi_scale_connections, on_route_status=self.on_route_status, traffic_light_data=self.traffic_light_data)

    def extract_lane_polyline(self, sample_idx: int, lane_idx: int) -> FeatureDataType:
        """
        Extract start points (first coordinate) for segments in lane, specified by segment indices
            in lane_groupings.
        :param sample_idx: sample index in batch
        :param lane_idx: lane index in sample
        :return: lane_polyline: <np.ndarray: num_lane_segments_in_lane, 2>. Array of start points
            for each segment in lane.
        """
        lane_grouping = self.lane_groupings[sample_idx][lane_idx]
        return self.coords[sample_idx][lane_grouping, 0]

def validate_type(instantiated_class: Any, desired_type: Type[Any]) -> None:
    """
    Validate that constructed type is indeed the desired one
    :param instantiated_class: class that was created
    :param desired_type: type that the created class should have
    """
    assert isinstance(instantiated_class, desired_type), f'Class to be of type {desired_type}, but is {type(instantiated_class)}!'

def rotate_coords(coords: npt.NDArray[np.float32], quaternion: Quaternion) -> npt.NDArray[np.float32]:
    """
    Rotate all vector coordinates within input tensor using input quaternion.
    :param coords: coordinates to translate: <num_map_elements, num_points_per_element, 2>.
    :param quaternion: Rotation to apply.
    :return rotated coords.
    """
    _validate_coords_shape(coords)
    validate_type(coords, np.ndarray)
    num_map_elements, num_points_per_element, _ = coords.shape
    coords = coords.reshape(num_map_elements * num_points_per_element, 2)
    coords = np.concatenate((coords, np.zeros_like(coords[:, 0:1])), axis=-1)
    coords = np.dot(quaternion.rotation_matrix.astype(coords.dtype), coords.T)
    return coords.T[:, :2].reshape(num_map_elements, num_points_per_element, 2)

@dataclass
class Raster(AbstractModelFeature):
    """
    Dataclass that holds map/environment signals in a raster (HxWxC) or (CxHxW) to be consumed by the model.

    :param ego_layer: raster layer that represents the ego's position and extent
    :param agents_layer: raster layer that represents the position and extent of agents surrounding the ego
    :param roadmap_layer: raster layer that represents map information around the ego
    """
    data: FeatureDataType

    def __post_init__(self) -> None:
        """Sanitize attributes of dataclass."""
        self.num_map_channels = 2
        self.ego_agent_sep_channel_num = int((self.num_channels() - self.num_map_channels) // 2)
        shape = self.data.shape
        array_dims = len(shape)
        if array_dims != 3 and array_dims != 4:
            raise RuntimeError(f'Invalid raster array. Expected 3 or 4 dims, got {array_dims}.')

    @property
    def num_batches(self) -> Optional[int]:
        """Number of batches in the feature."""
        return None if len(self.data.shape) < 4 else self.data.shape[0]

    def to_feature_tensor(self) -> AbstractModelFeature:
        """Implemented. See interface."""
        to_tensor_torchvision = torchvision.transforms.ToTensor()
        return Raster(data=to_tensor_torchvision(np.asarray(self.data)))

    def to_device(self, device: torch.device) -> Raster:
        """Implemented. See interface."""
        validate_type(self.data, torch.Tensor)
        return Raster(data=self.data.to(device=device))

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Raster:
        """Implemented. See interface."""
        return Raster(data=data['data'])

    def unpack(self) -> List[Raster]:
        """Implemented. See interface."""
        return [Raster(data[None]) for data in self.data]

    @staticmethod
    def from_feature_tensor(tensor: torch.Tensor) -> Raster:
        """Implemented. See interface."""
        array = tensor.numpy()
        if len(array.shape) == 4:
            array = array.transpose(0, 2, 3, 1)
        else:
            array = array.transpose(1, 2, 0)
        return Raster(array)

    @property
    def width(self) -> int:
        """
        :return: the width of a raster
        """
        return self.data.shape[-2] if self._is_channels_last() else self.data.shape[-1]

    @property
    def height(self) -> int:
        """
        :return: the height of a raster
        """
        return self.data.shape[-3] if self._is_channels_last() else self.data.shape[-2]

    def num_channels(self) -> int:
        """
        Number of raster channels.
        """
        return self.data.shape[-1] if self._is_channels_last() else self.data.shape[-3]

    @property
    def ego_layer(self) -> FeatureDataType:
        """
        Get the 2D grid representing the ego layer
        located at channel 0.
        """
        return self._get_data_channel(range(0, self.ego_agent_sep_channel_num))

    @property
    def agents_layer(self) -> FeatureDataType:
        """
        Get the 2D grid representing the agents layer
        located at channel 1.
        """
        start_channel = self.ego_agent_sep_channel_num
        end_channel = self.num_channels() - self.num_map_channels
        return self._get_data_channel(range(start_channel, end_channel))

    @property
    def roadmap_layer(self) -> FeatureDataType:
        """
        Get the 2D grid representing the map layer
        located at channel 2.
        """
        return self._get_data_channel(-2)

    @property
    def baseline_paths_layer(self) -> FeatureDataType:
        """
        Get the 2D grid representing the baseline paths layer
        located at channel 3.
        """
        return self._get_data_channel(-1)

    def _is_channels_last(self) -> bool:
        """
        Check location of channel dimension
        :return True if position [-1] is the number of channels
        """
        if isinstance(self.data, Tensor):
            return False
        elif isinstance(self.data, ndarray):
            return True
        else:
            raise RuntimeError(f'The data needs to be either numpy array or torch Tensor, but got type(data): {type(self.data)}')

    def _get_data_channel(self, index: Union[int, range]) -> FeatureDataType:
        """
        Extract channel data
        :param index: of layer
        :return: data corresponding to layer
        """
        if self._is_channels_last():
            return self.data[..., index]
        else:
            return self.data[..., index, :, :]

@dataclass
class Trajectory(AbstractModelFeature):
    """
    Dataclass that holds trajectory signals produced from the model or from the dataset for supervision.

    :param data: either a [num_batches, num_states, 3] or [num_states, 3] representing the trajectory
                 where se2_state is [x, y, heading] with units [meters, meters, radians].
    """
    data: FeatureDataType

    def __post_init__(self) -> None:
        """Sanitize attributes of the dataclass."""
        array_dims = self.num_dimensions
        state_size = self.data.shape[-1]
        if array_dims != 2 and array_dims != 3:
            raise RuntimeError(f'Invalid trajectory array. Expected 2 or 3 dims, got {array_dims}.')
        if state_size != self.state_size():
            raise RuntimeError(f'Invalid trajectory array. Expected {self.state_size()} variables per state, got {state_size}.')

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return len(self.data) > 0 and self.data.shape[-2] > 0 and (self.data.shape[-1] == self.state_size())

    def to_device(self, device: torch.device) -> Trajectory:
        """Implemented. See interface."""
        validate_type(self.data, torch.Tensor)
        return Trajectory(data=self.data.to(device=device))

    def to_feature_tensor(self) -> Trajectory:
        """Inherited, see superclass."""
        return Trajectory(data=to_tensor(self.data))

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Trajectory:
        """Implemented. See interface."""
        return Trajectory(data=data['data'])

    def unpack(self) -> List[Trajectory]:
        """Implemented. See interface."""
        return [Trajectory(data[None]) for data in self.data]

    @staticmethod
    def state_size() -> int:
        """
        Size of each SE2 state of the trajectory.
        """
        return 3

    @property
    def xy(self) -> FeatureDataType:
        """
        :return: tensor of positions [..., x, y]
        """
        return self.data[..., :2]

    @property
    def terminal_position(self) -> FeatureDataType:
        """
        :return: tensor of terminal position [..., x, y]
        """
        return self.data[..., -1, :2]

    @property
    def terminal_heading(self) -> FeatureDataType:
        """
        :return: tensor of terminal position [..., heading]
        """
        return self.data[..., -1, 2]

    @property
    def position_x(self) -> FeatureDataType:
        """
        Array of x positions of trajectory.
        """
        return self.data[..., 0]

    @property
    def numpy_position_x(self) -> FeatureDataType:
        """
        Array of x positions of trajectory.
        """
        return np.asarray(self.data[..., 0])

    @property
    def position_y(self) -> FeatureDataType:
        """
        Array of y positions of trajectory.
        """
        return self.data[..., 1]

    @property
    def numpy_position_y(self) -> FeatureDataType:
        """
        Array of y positions of trajectory.
        """
        return np.asarray(self.data[..., 1])

    @property
    def heading(self) -> FeatureDataType:
        """
        Array of heading positions of trajectory.
        """
        return self.data[..., 2]

    @property
    def num_dimensions(self) -> int:
        """
        :return: dimensions of underlying data
        """
        return len(self.data.shape)

    @property
    def num_of_iterations(self) -> int:
        """
        :return: number of states in a trajectory
        """
        return int(self.data.shape[-2])

    @property
    def num_batches(self) -> Optional[int]:
        """
        :return: number of batches in the trajectory, None if trajectory does not have batch dimension
        """
        return None if self.num_dimensions <= 2 else self.data.shape[0]

    def state_at_index(self, index: int) -> FeatureDataType:
        """
        Query state at index along trajectory horizon
        :param index: along horizon
        :return: state corresponding to the index along trajectory horizon
        @raise in case index is not within valid range: 0 < index <= num_of_iterations
        """
        assert 0 <= index < self.num_of_iterations, f'Index is out of bounds! 0 <= {index} < {self.num_of_iterations}!'
        return self.data[..., index, :]

    def extract_number_of_last_states(self, number_of_states: int) -> Trajectory:
        """
        Extract last number_of_states from a trajectory
        :param number_of_states: from last point
        :return: shorter trajectory containing number_of_states from end of trajectory
        @raise in case number_of_states is not within valid range: 0 < number_of_states <= length
        """
        assert number_of_states > 0, f'number_of_states has to be > 0, {number_of_states} > 0!'
        length = self.num_of_iterations
        assert number_of_states <= length, f'number_of_states has to be smaller than length, {number_of_states} <= {length}!'
        return self.extract_trajectory_between(length - number_of_states, length)

    def extract_trajectory_between(self, start_index: int, end_index: Optional[int]) -> Trajectory:
        """
        Extract partial trajectory based on [start_index, end_index]
        :param start_index: starting index
        :param end_index: ending index
        :return: Trajectory
        @raise in case the desired ranges are not valid
        """
        if not end_index:
            end_index = self.num_of_iterations
        assert 0 <= start_index < self.num_of_iterations, f'Start index is out of bounds! 0 <= {start_index} < {self.num_of_iterations}!'
        assert 0 <= end_index <= self.num_of_iterations, f'Start index is out of bounds! 0 <= {end_index} <= {self.num_of_iterations}!'
        assert start_index < end_index, f'Start Index has to be smaller then end, {start_index} < {end_index}!'
        return Trajectory(data=self.data[..., start_index:end_index, :])

    @classmethod
    def append_to_trajectory(cls, trajectory: Trajectory, new_state: torch.Tensor) -> Trajectory:
        """
        Extend trajectory with a new state, in this case we require that both trajectory and new_state has dimension
        of 3, that means that they both have batch dimension
        :param trajectory: to be extended
        :param new_state: state with which trajectory should be extended
        :return: extended trajectory
        """
        assert trajectory.num_dimensions == 3, f'Trajectory dimension {trajectory.num_dimensions} != 3!'
        assert len(new_state.shape) == 3, f'New state dimension {new_state.shape} != 3!'
        if new_state.shape[0] != trajectory.data.shape[0]:
            raise RuntimeError(f'Not compatible shapes {new_state.shape} != {trajectory.data.shape}!')
        if new_state.shape[-1] != trajectory.data.shape[-1]:
            raise RuntimeError(f'Not compatible shapes {new_state.shape} != {trajectory.data.shape}!')
        return Trajectory(data=torch.cat((trajectory.data, new_state.clone()), dim=1))

@dataclass
class VectorSetMap(AbstractModelFeature):
    """
    Vector set map data structure, including:
        coords: Dict[str, List[<np.ndarray: num_elements, num_points, 2>]].
            The (x, y) coordinates of each point in a map element across map elements per sample in batch,
                indexed by map feature.
        traffic_light_data: Dict[str, List[<np.ndarray: num_elements, num_points, 4>]].
            One-hot encoding of traffic light status for each point in a map element across map elements per sample
                in batch, indexed by map feature. Same indexing as coords.
            Encoding: green [1, 0, 0, 0] yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1]
        availabilities: Dict[str, List[<np.ndarray: num_elements, num_points>]].
            Boolean indicator of whether feature data (coords as well as traffic light status if it exists for feature)
                is available for point at given index or if it is zero-padded.

    Feature formulation as sets of vectors for each map element similar to that of VectorNet ("VectorNet: Encoding HD
    Maps and Agent Dynamics from Vectorized Representation"), except map elements are encoded as sets of singular x, y
    points instead of start, end point pairs.

    Coords, traffic light status, and availabilities data are each keyed by map feature name, with dimensionality
    (availabilities don't include feature dimension):
    B: number of samples per batch (variable)
    N: number of map elements (fixed for a given map feature)
    P: number of points (fixed for a given map feature)
    F: number of features (2 for coords, 4 for traffic light status)

    Data at the same index represent the same map element/point among coords, traffic_light_data, and availabilities,
    with traffic_light_data only optionally included. For each map feature, the top level List represents number of
    samples per batch. This is a special feature where each batch entry can have a different size. For that reason, the
    features can not be placed to a single tensor, and we batch the feature with a custom `collate` function.
    """
    coords: Dict[str, List[FeatureDataType]]
    traffic_light_data: Dict[str, List[FeatureDataType]]
    availabilities: Dict[str, List[FeatureDataType]]
    _polyline_coord_dim: int = 2
    _traffic_light_status_dim: int = LaneSegmentTrafficLightData.encoding_dim()

    def __post_init__(self) -> None:
        """
        Sanitize attributes of the dataclass.
        :raise RuntimeError if dimensions invalid.
        """
        if not len(self.coords) > 0:
            raise RuntimeError('Coords cannot be empty!')
        if not all([len(coords) > 0 for coords in self.coords.values()]):
            raise RuntimeError('Batch size has to be > 0!')
        self._sanitize_feature_consistency()
        self._sanitize_data_dimensionality()

    def _sanitize_feature_consistency(self) -> None:
        """
        Check data dimensionality consistent across and within map features.
        :raise RuntimeError if dimensions invalid.
        """
        if not all([len(coords) == len(list(self.coords.values())[0]) for coords in self.coords.values()]):
            raise RuntimeError('Batch size inconsistent across features!')
        for feature_name, feature_coords in self.coords.items():
            if feature_name not in self.availabilities:
                raise RuntimeError('No matching feature in coords for availabilities data!')
            feature_avails = self.availabilities[feature_name]
            if len(feature_avails) != len(feature_coords):
                raise RuntimeError(f'Batch size between coords and availabilities data inconsistent! {len(feature_coords)} != {len(feature_avails)}')
            feature_size = self.feature_size(feature_name)
            if feature_size[1] == 0:
                raise RuntimeError('Features cannot be empty!')
            for coords in feature_coords:
                if coords.shape[0:2] != feature_size:
                    raise RuntimeError(f"Coords for {feature_name} feature don't have consistent feature size! {coords.shape[0:2] != feature_size}")
            for avails in feature_avails:
                if avails.shape[0:2] != feature_size:
                    raise RuntimeError(f"Availabilities for {feature_name} feature don't have consistent feature size! {avails.shape[0:2] != feature_size}")
        for feature_name, feature_tl_data in self.traffic_light_data.items():
            if feature_name not in self.coords:
                raise RuntimeError('No matching feature in coords for traffic light data!')
            feature_coords = self.coords[feature_name]
            if len(feature_tl_data) != len(self.coords[feature_name]):
                raise RuntimeError(f'Batch size between coords and traffic light data inconsistent! {len(feature_coords)} != {len(feature_tl_data)}')
            feature_size = self.feature_size(feature_name)
            for tl_data in feature_tl_data:
                if tl_data.shape[0:2] != feature_size:
                    raise RuntimeError(f"Traffic light data for {feature_name} feature don't have consistent feature size! {tl_data.shape[0:2] != feature_size}")

    def _sanitize_data_dimensionality(self) -> None:
        """
        Check data dimensionality as expected.
        :raise RuntimeError if dimensions invalid.
        """
        for feature_coords in self.coords.values():
            for sample in feature_coords:
                if sample.shape[2] != self._polyline_coord_dim:
                    raise RuntimeError('The dimension of coords is not correct!')
        for feature_tl_data in self.traffic_light_data.values():
            for sample in feature_tl_data:
                if sample.shape[2] != self._traffic_light_status_dim:
                    raise RuntimeError('The dimension of traffic light data is not correct!')
        for feature_avails in self.availabilities.values():
            for sample in feature_avails:
                if len(sample.shape) != 2:
                    raise RuntimeError('The dimension of availabilities is not correct!')

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return all([len(feature_coords) > 0 for feature_coords in self.coords.values()]) and all([feature_coords[0].shape[0] > 0 for feature_coords in self.coords.values()]) and all([feature_coords[0].shape[1] > 0 for feature_coords in self.coords.values()]) and all([len(feature_tl_data) > 0 for feature_tl_data in self.traffic_light_data.values()]) and all([feature_tl_data[0].shape[0] > 0 for feature_tl_data in self.traffic_light_data.values()]) and all([feature_tl_data[0].shape[1] > 0 for feature_tl_data in self.traffic_light_data.values()]) and all([len(features_avails) > 0 for features_avails in self.availabilities.values()]) and all([features_avails[0].shape[0] > 0 for features_avails in self.availabilities.values()]) and all([features_avails[0].shape[1] > 0 for features_avails in self.availabilities.values()])

    @property
    def batch_size(self) -> int:
        """
        Batch size across features.
        :return: number of batches.
        """
        return len(list(self.coords.values())[0])

    def feature_size(self, feature_name: str) -> Tuple[int, int]:
        """
        Number of map elements for given feature, points per element.
        :param feature_name: name of map feature to access.
        :return: [num_elements, num_points]
        :raise: RuntimeError if empty feature.
        """
        map_feature = self.coords[feature_name][0]
        if map_feature.size == 0:
            raise RuntimeError('Feature is empty!')
        return (map_feature.shape[0], map_feature.shape[1])

    @classmethod
    def coord_dim(cls) -> int:
        """
        Coords dimensionality, should be 2 (x, y).
        :return: dimension of coords.
        """
        return cls._polyline_coord_dim

    @classmethod
    def traffic_light_status_dim(cls) -> int:
        """
        Traffic light status dimensionality, should be 4.
        :return: dimension of traffic light status.
        """
        return cls._traffic_light_status_dim

    def get_lane_coords(self, sample_idx: int) -> FeatureDataType:
        """
        Retrieve lane coordinates at given sample index.
        :param sample_idx: the batch index of interest.
        :return: lane coordinate features.
        """
        lane_coords = self.coords[VectorFeatureLayer.LANE.name][sample_idx]
        if lane_coords.size == 0:
            raise RuntimeError('Lane feature is empty!')
        return lane_coords

    @classmethod
    def collate(cls, batch: List[VectorSetMap]) -> VectorSetMap:
        """Implemented. See interface."""
        coords: Dict[str, List[FeatureDataType]] = defaultdict(list)
        traffic_light_data: Dict[str, List[FeatureDataType]] = defaultdict(list)
        availabilities: Dict[str, List[FeatureDataType]] = defaultdict(list)
        for sample in batch:
            for feature_name, feature_coords in sample.coords.items():
                coords[feature_name] += feature_coords
            for feature_name, feature_tl_data in sample.traffic_light_data.items():
                traffic_light_data[feature_name] += feature_tl_data
            for feature_name, feature_avails in sample.availabilities.items():
                availabilities[feature_name] += feature_avails
        return VectorSetMap(coords=coords, traffic_light_data=traffic_light_data, availabilities=availabilities)

    def to_feature_tensor(self) -> VectorSetMap:
        """Implemented. See interface."""
        return VectorSetMap(coords={feature_name: [to_tensor(sample).contiguous() for sample in feature_coords] for feature_name, feature_coords in self.coords.items()}, traffic_light_data={feature_name: [to_tensor(sample).contiguous() for sample in feature_tl_data] for feature_name, feature_tl_data in self.traffic_light_data.items()}, availabilities={feature_name: [to_tensor(sample).contiguous() for sample in feature_avails] for feature_name, feature_avails in self.availabilities.items()})

    def to_device(self, device: torch.device) -> VectorSetMap:
        """Implemented. See interface."""
        return VectorSetMap(coords={feature_name: [sample.to(device=device) for sample in feature_coords] for feature_name, feature_coords in self.coords.items()}, traffic_light_data={feature_name: [sample.to(device=device) for sample in feature_tl_data] for feature_name, feature_tl_data in self.traffic_light_data.items()}, availabilities={feature_name: [sample.to(device=device) for sample in feature_avails] for feature_name, feature_avails in self.availabilities.items()})

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> VectorSetMap:
        """Implemented. See interface."""
        return VectorSetMap(coords=data['coords'], traffic_light_data=data['traffic_light_data'], availabilities=data['availabilities'])

    def unpack(self) -> List[VectorSetMap]:
        """Implemented. See interface."""
        return [VectorSetMap({feature_name: [feature_coords[sample_idx]] for feature_name, feature_coords in self.coords.items()}, {feature_name: [feature_tl_data[sample_idx]] for feature_name, feature_tl_data in self.traffic_light_data.items()}, {feature_name: [feature_avails[sample_idx]] for feature_name, feature_avails in self.availabilities.items()}) for sample_idx in range(self.batch_size)]

    def rotate(self, quaternion: Quaternion) -> VectorSetMap:
        """
        Rotate the vector set map.
        :param quaternion: Rotation to apply.
        :return rotated VectorSetMap.
        """
        for feature_coords in self.coords.values():
            for sample in feature_coords:
                validate_type(sample, np.ndarray)
        return VectorSetMap(coords={feature_name: [rotate_coords(sample, quaternion) for sample in feature_coords] for feature_name, feature_coords in self.coords.items()}, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)

    def translate(self, translation_value: FeatureDataType) -> VectorSetMap:
        """
        Translate the vector set map.
        :param translation_value: Translation in x, y, z.
        :return translated VectorSetMap.
        :raise ValueError if translation_value dimensions invalid.
        """
        if translation_value.size != 3:
            raise ValueError(f'Translation value has incorrect dimensions: {translation_value.size}! Expected: 3 (x, y, z)')
        are_the_same_type(translation_value, list(self.coords.values())[0])
        return VectorSetMap(coords={feature_name: [translate_coords(sample_coords, translation_value, sample_avails) for sample_coords, sample_avails in zip(self.coords[feature_name], self.availabilities[feature_name])] for feature_name in self.coords}, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)

    def scale(self, scale_value: FeatureDataType) -> VectorSetMap:
        """
        Scale the vector set map.
        :param scale_value: <np.float: 3,>. Scale in x, y, z.
        :return scaled VectorSetMap.
        :raise ValueError if scale_value dimensions invalid.
        """
        if scale_value.size != 3:
            raise ValueError(f'Scale value has incorrect dimensions: {scale_value.size}! Expected: 3 (x, y, z)')
        are_the_same_type(scale_value, list(self.coords.values())[0])
        return VectorSetMap(coords={feature_name: [scale_coords(sample, scale_value) for sample in feature_coords] for feature_name, feature_coords in self.coords.items()}, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)

    def xflip(self) -> VectorSetMap:
        """
        Flip the vector set map along the X-axis.
        :return flipped VectorSetMap.
        """
        return VectorSetMap(coords={feature_name: [xflip_coords(sample) for sample in feature_coords] for feature_name, feature_coords in self.coords.items()}, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)

    def yflip(self) -> VectorSetMap:
        """
        Flip the vector set map along the Y-axis.
        :return flipped VectorSetMap.
        """
        return VectorSetMap(coords={feature_name: [yflip_coords(sample) for sample in feature_coords] for feature_name, feature_coords in self.coords.items()}, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)

class TestVectorUtils(unittest.TestCase):
    """Test vector-based feature utility functions."""

    def setUp(self) -> None:
        """Set up test case."""
        self.coords: npt.NDArray[np.float32] = np.array([[[0.0, 0.0], [-1.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [-1.0, -1.0], [1.0, -1.0]]])
        self.avails: npt.NDArray[np.bool_] = np.array([[False, True, True], [True, True, True]])

    def test_rotate_coords(self) -> None:
        """
        Test vector feature coordinate rotation.
        """
        quaternion = Quaternion(axis=[1, 0, 0], angle=3.14159265)
        expected_result: npt.NDArray[np.float32] = np.array([[[0.0, 0.0], [-1.0, -1.0], [1.0, -1.0]], [[1.0, 0.0], [-1.0, 1.0], [1.0, 1.0]]])
        result = rotate_coords(self.coords, quaternion)
        np.testing.assert_allclose(expected_result, result)

    def test_translate_coords(self) -> None:
        """
        Test vector feature coordinate translation.
        """
        translation_value: npt.NDArray[np.float32] = np.array([1.0, 0.0, -1.0])
        expected_result: npt.NDArray[np.float32] = np.array([[[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]], [[2.0, 0.0], [0.0, -1.0], [2.0, -1.0]]])
        result = translate_coords(self.coords, translation_value)
        np.testing.assert_allclose(expected_result, result)
        result = translate_coords(self.coords, translation_value, self.avails)
        expected_result[0][0] = [0.0, 0.0]
        np.testing.assert_allclose(expected_result, result)
        result = translate_coords(torch.from_numpy(self.coords), torch.from_numpy(translation_value), torch.from_numpy(self.avails))
        torch.testing.assert_allclose(torch.from_numpy(expected_result), result)

    def test_scale_coords(self) -> None:
        """
        Test vector feature coordinate scaling.
        """
        scale_value: npt.NDArray[np.float32] = np.array([-2.0, 0.0, -1.0])
        expected_result: npt.NDArray[np.float32] = np.array([[[0.0, 0.0], [2.0, 0.0], [-2.0, 0.0]], [[-2.0, 0.0], [2.0, 0.0], [-2.0, 0.0]]])
        result = scale_coords(self.coords, scale_value)
        np.testing.assert_allclose(expected_result, result)
        result = scale_coords(torch.from_numpy(self.coords), torch.from_numpy(scale_value))
        torch.testing.assert_allclose(torch.from_numpy(expected_result), result)

    def test_xflip_coords(self) -> None:
        """
        Test flipping vector feature coordinates about X-axis.
        """
        expected_result: npt.NDArray[np.float32] = np.array([[[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]], [[-1.0, 0.0], [1.0, -1.0], [-1.0, -1.0]]])
        result = xflip_coords(self.coords)
        np.testing.assert_allclose(expected_result, result)
        result = xflip_coords(torch.from_numpy(self.coords))
        torch.testing.assert_allclose(torch.from_numpy(expected_result), result)

    def test_yflip_coords(self) -> None:
        """
        Test flipping vector feature coordinates about Y-axis.
        """
        expected_result: npt.NDArray[np.float32] = np.array([[[0.0, 0.0], [-1.0, -1.0], [1.0, -1.0]], [[1.0, 0.0], [-1.0, 1.0], [1.0, 1.0]]])
        result = yflip_coords(self.coords)
        np.testing.assert_allclose(expected_result, result)
        result = yflip_coords(torch.from_numpy(self.coords))
        torch.testing.assert_allclose(torch.from_numpy(expected_result), result)

class NuBoard:
    """NuBoard application class."""

    def __init__(self, nuboard_paths: List[str], scenario_builder: AbstractScenarioBuilder, vehicle_parameters: VehicleParameters, port_number: int=5006, profiler_path: Optional[Path]=None, resource_prefix: Optional[str]=None, async_scenario_rendering: bool=True, scenario_rendering_frame_rate_cap_hz: int=60):
        """
        Nuboard main class.
        :param nuboard_paths: A list of paths to nuboard files.
        :param scenario_builder: Scenario builder instance.
        :param vehicle_parameters: vehicle parameters.
        :param port_number: Bokeh port number.
        :param profiler_path: Path to save the profiler.
        :param resource_prefix: Prefix to the resource path in HTML.
        :param async_scenario_rendering: Whether to use asynchronous scenario rendering in the scenario tab.
        :param scenario_rendering_frame_rate_cap_hz: Maximum frames to render in the scenario tab per second.
            Use lower values when running nuBoard in the cloud to prevent frame queues due to latency. The rule of thumb
            is to match the frame rate with the expected latency, e.g 5Hz for 200ms round-trip latency.
            Internally this value is capped at 60.
        """
        self._profiler_path = profiler_path
        self._nuboard_paths = check_nuboard_file_paths(nuboard_paths)
        self._scenario_builder = scenario_builder
        self._port_number = port_number
        self._vehicle_parameters = vehicle_parameters
        self._doc: Optional[Document] = None
        self._resource_prefix = resource_prefix if resource_prefix else ''
        self._resource_path = Path(__file__).parents[0] / 'resource'
        self._profiler_file_name = 'nuboard'
        self._profiler: Optional[ProfileCallback] = None
        self._async_scenario_rendering = async_scenario_rendering
        if scenario_rendering_frame_rate_cap_hz < 1 or scenario_rendering_frame_rate_cap_hz > 60:
            raise ValueError('scenario_rendering_frame_rate_cap_hz should be between 1 and 60')
        self._scenario_rendering_frame_rate_cap_hz = scenario_rendering_frame_rate_cap_hz

    def stop_handler(self, sig: Any, frame: Any) -> None:
        """Helper to handle stop signals."""
        logger.info('Stopping the Bokeh application.')
        if self._profiler:
            self._profiler.save_profiler(self._profiler_file_name)
        IOLoop.current().stop()

    def run(self) -> None:
        """Run nuBoard WebApp."""
        logger.info(f'Opening Bokeh application on http://localhost:{self._port_number}/')
        logger.info(f'Async rendering is set to: {self._async_scenario_rendering}')
        io_loop = IOLoop.current()
        if self._profiler_path is not None:
            signal.signal(signal.SIGTERM, self.stop_handler)
            signal.signal(signal.SIGINT, self.stop_handler)
            self._profiler = ProfileCallback(output_dir=self._profiler_path)
            self._profiler.start_profiler(self._profiler_file_name)
        bokeh_app = Application(FunctionHandler(self.main_page))
        server = Server({'/': bokeh_app}, io_loop=io_loop, port=self._port_number, allow_websocket_origin=['*'], extra_patterns=[('/resource/(.*)', StaticFileHandler, {'path': str(self._resource_path)})])
        server.start()
        io_loop.add_callback(server.show, '/')
        try:
            io_loop.start()
        except RuntimeError as e:
            logger.warning(f'{e}')

    def main_page(self, doc: Document) -> None:
        """
        Main nuBoard page.
        :param doc: HTML document.
        """
        self._doc = doc
        template_path = Path(os.path.dirname(os.path.realpath(__file__))) / 'templates'
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_path))
        self._doc.template = env.get_template('index.html')
        self._doc.title = 'nuBoard'
        nuboard_files = read_nuboard_file_paths(file_paths=self._nuboard_paths)
        experiment_file_data = ExperimentFileData(file_paths=nuboard_files)
        overview_tab = OverviewTab(doc=self._doc, experiment_file_data=experiment_file_data)
        histogram_tab = HistogramTab(doc=self._doc, experiment_file_data=experiment_file_data)
        scenario_tab = ScenarioTab(experiment_file_data=experiment_file_data, scenario_builder=self._scenario_builder, doc=self._doc, vehicle_parameters=self._vehicle_parameters, async_rendering=self._async_scenario_rendering, frame_rate_cap_hz=self._scenario_rendering_frame_rate_cap_hz)
        configuration_tab = ConfigurationTab(experiment_file_data=experiment_file_data, doc=self._doc, tabs=[overview_tab, histogram_tab, scenario_tab])
        s3_tab = CloudTab(doc=self._doc, configuration_tab=configuration_tab)
        self._doc.add_root(configuration_tab.file_path_input)
        self._doc.add_root(configuration_tab.experiment_file_path_checkbox_group)
        self._doc.add_root(s3_tab.s3_bucket_name)
        self._doc.add_root(s3_tab.s3_bucket_text_input)
        self._doc.add_root(s3_tab.s3_error_text)
        self._doc.add_root(s3_tab.s3_access_key_id_text_input)
        self._doc.add_root(s3_tab.s3_secret_access_key_password_input)
        self._doc.add_root(s3_tab.s3_bucket_prefix_text_input)
        self._doc.add_root(s3_tab.s3_modal_query_btn)
        self._doc.add_root(s3_tab.s3_download_text_input)
        self._doc.add_root(s3_tab.s3_download_button)
        self._doc.add_root(s3_tab.data_table)
        self._doc.add_root(overview_tab.table)
        self._doc.add_root(overview_tab.planner_checkbox_group)
        self._doc.add_root(histogram_tab.scenario_type_multi_choice)
        self._doc.add_root(histogram_tab.metric_name_multi_choice)
        self._doc.add_root(histogram_tab.planner_checkbox_group)
        self._doc.add_root(histogram_tab.histogram_plots)
        self._doc.add_root(histogram_tab.bin_spinner)
        self._doc.add_root(histogram_tab.histogram_modal_query_btn)
        self._doc.add_root(scenario_tab.planner_checkbox_group)
        self._doc.add_root(scenario_tab.scenario_title_div)
        self._doc.add_root(scenario_tab.object_checkbox_group)
        self._doc.add_root(scenario_tab.traj_checkbox_group)
        self._doc.add_root(scenario_tab.map_checkbox_group)
        self._doc.add_root(scenario_tab.scalar_scenario_type_select)
        self._doc.add_root(scenario_tab.scalar_log_name_select)
        self._doc.add_root(scenario_tab.scalar_scenario_name_select)
        self._doc.add_root(scenario_tab.scenario_token_multi_choice)
        self._doc.add_root(scenario_tab.scenario_modal_query_btn)
        self._doc.add_root(scenario_tab.time_series_layout)
        self._doc.add_root(scenario_tab.ego_expert_states_layout)
        self._doc.add_root(scenario_tab.scenario_score_layout)
        self._doc.add_root(scenario_tab.simulation_tile_layout)

def set_up_common_builder(cfg: DictConfig, profiler_name: str) -> CommonBuilder:
    """
    Set up a common builder when running simulations.
    :param cfg: Hydra configuration.
    :param profiler_name: Profiler name.
    :return A data classes with common builders.
    """
    multi_main_callback = build_main_multi_callback(cfg)
    multi_main_callback.on_run_simulation_start()
    update_config_for_simulation(cfg=cfg)
    build_logger(cfg)
    worker = build_worker(cfg)
    build_simulation_experiment_folder(cfg=cfg)
    output_dir = Path(cfg.output_dir)
    profiler = None
    if cfg.enable_profiling:
        logger.info('Profiler is enabled!')
        profiler = ProfileCallback(output_dir=output_dir)
    if profiler:
        profiler.start_profiler(profiler_name)
    return CommonBuilder(worker=worker, multi_main_callback=multi_main_callback, output_dir=output_dir, profiler=profiler)

def update_config_for_simulation(cfg: DictConfig) -> None:
    """
    Updates the config based on some conditions.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    """
    OmegaConf.set_struct(cfg, False)
    if cfg.max_number_of_workers:
        cfg.callbacks = [callback for callback in cfg.callback.values() if not is_target_type(callback, TimingCallback)]
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)
    if cfg.log_config:
        logger.info(f'Creating experiment: {cfg.experiment}')
        logger.info('\n' + OmegaConf.to_yaml(cfg))

def build_logger(cfg: DictConfig) -> logging.Logger:
    """
    Setup the standard logger, always log to sys.stdout and optionally log to disk.
    :param cfg: Input dict config.
    :return: Logger with associated handlers.
    """
    handler_configs = [LogHandlerConfig(level=cfg.logger_level)]
    if cfg.output_dir is not None:
        path = str(Path(cfg.output_dir) / 'log.txt')
        handler_configs.append(LogHandlerConfig(level=cfg.logger_level, path=path))
    format_string = '%(asctime)s %(levelname)-2s {%(pathname)s:%(lineno)d}  %(message)s' if not cfg.logger_format_string else cfg.logger_format_string
    logger = configure_logger(handler_configs, format_str=format_string)
    if cfg.gpu:
        logger.disabled = int(os.environ.get('LOCAL_RANK', 0)) != 0
    logger.setLevel(level=LOGGING_LEVEL_MAP[cfg.logger_level])
    return logger

def build_worker(cfg: DictConfig) -> WorkerPool:
    """
    Builds the worker.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of WorkerPool.
    """
    logger.info('Building WorkerPool...')
    worker: WorkerPool = instantiate(cfg.worker, output_dir=cfg.output_dir) if is_target_type(cfg.worker, RayDistributed) else instantiate(cfg.worker)
    validate_type(worker, WorkerPool)
    logger.info('Building WorkerPool...DONE!')
    return worker

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for training/validation experiments.
    :param cfg: omegaconf dictionary
    """
    pl.seed_everything(cfg.seed, workers=True)
    build_logger(cfg)
    update_config_for_training(cfg)
    build_training_experiment_folder(cfg=cfg)
    worker = build_worker(cfg)
    if cfg.py_func == 'train':
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, 'build_training_engine'):
            engine = build_training_engine(cfg, worker)
        logger.info('Starting training...')
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, 'training'):
            engine.trainer.fit(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == 'test':
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, 'build_training_engine'):
            engine = build_training_engine(cfg, worker)
        logger.info('Starting testing...')
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, 'testing'):
            engine.trainer.test(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == 'cache':
        logger.info('Starting caching...')
        if cfg.worker == 'ray_distributed' and cfg.worker.use_distributed:
            raise AssertionError('ray in distributed mode will not work with this job')
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, 'caching'):
            cache_data(cfg=cfg, worker=worker)
        return None
    else:
        raise NameError(f'Function {cfg.py_func} does not exist')

def build_training_experiment_folder(cfg: DictConfig) -> None:
    """
    Builds the main experiment folder for training.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    """
    logger.info('Building experiment folders...')
    main_exp_folder = pathlib.Path(cfg.output_dir)
    logger.info(f'Experimental folder: {main_exp_folder}')
    main_exp_folder.mkdir(parents=True, exist_ok=True)

class ProfilerContextManager:
    """
    Class to wrap calls with a profiler callback.
    """

    def __init__(self, output_dir: str, enable_profiling: bool, name: str):
        """
        Build a profiler context.
        :param output_dir: dir to save profiling results in
        :param enable_profiling: whether we have profiling enabled or not
        :param name: name of the code segment we are profiling
        """
        self.profiler = ProfileCallback(pathlib.Path(output_dir)) if enable_profiling else None
        self.name = name

    def __enter__(self) -> None:
        """Start the profiler context."""
        if self.profiler:
            self.profiler.start_profiler(self.name)

    def __exit__(self, exc_type: type[BaseException], exc_val: BaseException, exc_tb: TracebackType) -> None:
        """
        Stop the profiler context and save the results.
        :param exc_type: type of exception raised while context is active
        :param exc_val: value of exception raised while context is active
        :param exc_tb: traceback of exception raised while context is active
        """
        if self.profiler:
            self.profiler.save_profiler(self.name)

def build_objectives(cfg: DictConfig) -> List[AbstractObjective]:
    """
    Build objectives based on config
    :param cfg: config
    :return list of objectives.
    """
    instantiated_objectives = []
    scenario_type_loss_weighting = cfg.scenario_type_weights.scenario_type_loss_weights if 'scenario_type_weights' in cfg and 'scenario_type_loss_weights' in cfg.scenario_type_weights else {}
    for objective_name, objective_type in cfg.objective.items():
        new_objective: AbstractObjective = instantiate(objective_type, scenario_type_loss_weighting=scenario_type_loss_weighting)
        validate_type(new_objective, AbstractObjective)
        instantiated_objectives.append(new_objective)
    return instantiated_objectives

def build_splitter(cfg: DictConfig) -> AbstractSplitter:
    """
    Build the splitter.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of Splitter.
    """
    logger.info('Building Splitter...')
    splitter: AbstractSplitter = instantiate(cfg)
    validate_type(splitter, AbstractSplitter)
    logger.info('Building Splitter...DONE!')
    return splitter

def build_agent_augmentor(cfg: DictConfig) -> List[AbstractAugmentor]:
    """
    Build list of augmentors based on config.
    :param cfg: Dict config.
    :return List of augmentor objects.
    """
    logger.info('Building augmentors...')
    instantiated_augmentors = []
    for augmentor_type in cfg.values():
        augmentor: AbstractAugmentor = instantiate(augmentor_type)
        validate_type(augmentor, AbstractAugmentor)
        instantiated_augmentors.append(augmentor)
    logger.info('Building augmentors...DONE!')
    return instantiated_augmentors

def build_scenarios(cfg: DictConfig, worker: WorkerPool, model: TorchModuleWrapper) -> List[AbstractScenario]:
    """
    Build the scenario objects that comprise the training dataset.
    :param cfg: Omegaconf dictionary.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :param model: NN model used for training.
    :return: List of extracted scenarios.
    """
    scenarios = extract_scenarios_from_cache(cfg, worker, model) if cfg.cache.use_cache_without_dataset else extract_scenarios_from_dataset(cfg, worker)
    logger.info(f'Extracted {len(scenarios)} scenarios for training')
    assert len(scenarios) > 0, 'No scenarios were retrieved for training, check the scenario_filter parameters!'
    return scenarios

def build_training_metrics(cfg: DictConfig) -> List[AbstractTrainingMetric]:
    """
    Build metrics based on config
    :param cfg: config
    :return list of metrics.
    """
    instantiated_metrics = []
    for metric_name, cfg_metric in cfg.training_metric.items():
        new_metric: AbstractTrainingMetric = instantiate(cfg_metric)
        validate_type(new_metric, AbstractTrainingMetric)
        instantiated_metrics.append(new_metric)
    return instantiated_metrics

def build_callbacks(cfg: DictConfig) -> List[pl.Callback]:
    """
    Build callbacks based on config.
    :param cfg: Dict config.
    :return List of callbacks.
    """
    logger.info('Building callbacks...')
    instantiated_callbacks = []
    for callback_type in cfg.callbacks.values():
        callback: pl.Callback = instantiate(callback_type)
        validate_type(callback, pl.Callback)
        instantiated_callbacks.append(callback)
    if 'data_augmentation_scheduler' in cfg:
        instantiated_callbacks.extend([instantiate(scheduler) for scheduler in cfg.data_augmentation_scheduler.values()])
    if cfg.lightning.trainer.params.gpus:
        instantiated_callbacks.append(pl.callbacks.GPUStatsMonitor(intra_step_time=True, inter_step_time=True))
    logger.info('Building callbacks...DONE!')
    return instantiated_callbacks

def extract_last_checkpoint_from_experiment(output_dir: pathlib.Path, date_format: str) -> Optional[pathlib.Path]:
    """
    Extract last checkpoint from latest experiment
    :param output_dir: of the current experiment, we assume that parent folder has previous experiments of the same type
    :param date_format: format time used for folders
    :return path to latest checkpoint, return None in case no checkpoint was found
    """
    date_times = [datetime.strptime(dir.name, date_format) for dir in output_dir.parent.iterdir() if dir != output_dir]
    date_times.sort(reverse=True)
    for date_time in date_times:
        checkpoint = find_last_checkpoint_in_dir(output_dir.parent, pathlib.Path(date_time.strftime(date_format)))
        if checkpoint:
            return checkpoint
    return None

def configure_logger(handler_configs: List[LogHandlerConfig], format_str: str='%(asctime)s %(levelname)-2s {%(pathname)s:%(lineno)d}  %(message)s') -> logging.Logger:
    """
    Configures the python default logger.
    :param handler_configs: List of LogHandlerConfig objects specifying the logger handlers.
    :param format_str: Formats the log events.
    :return: A logger.
    """
    logger = logging.getLogger()
    for old_handler in logger.handlers:
        logger.removeHandler(old_handler)
    for config in handler_configs:
        if not config.path:
            handler = TqdmLoggingHandler()
        else:
            handler = logging.FileHandler(config.path)
        handler.setLevel(LOGGING_LEVEL_MAP[config.level])
        handler.setFormatter(logging.Formatter(format_str))
        handler.addFilter(PathKeywordMatch(config.filter_regexp))
        logger.addHandler(handler)
    return logger

def extract_scenarios_from_dataset(cfg: DictConfig, worker: WorkerPool) -> List[AbstractScenario]:
    """
    Extract and filter scenarios by loading a dataset using the scenario builder.
    :param cfg: Omegaconf dictionary.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :return: List of extracted scenarios.
    """
    scenario_builder = build_scenario_builder(cfg)
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    scenarios: List[AbstractScenario] = scenario_builder.get_scenarios(scenario_filter, worker)
    return scenarios

def validate_dict_type(instantiated_dict: Dict[str, Any], desired_type: Type[Any]) -> None:
    """
    Validate that all entries in dict is indeed the desired one
    :param instantiated_dict: dictionary that was created
    :param desired_type: type that the created class should have
    """
    for value in instantiated_dict.values():
        if isinstance(value, dict):
            validate_dict_type(value, desired_type)
        else:
            validate_type(value, desired_type)

def find_last_checkpoint_in_dir(group_dir: pathlib.Path, experiment_uid: pathlib.Path) -> Optional[pathlib.Path]:
    """
    Extract last checkpoint from a experiment
    :param group_dir: defined by ${group}/${experiment_name}/${job_name} from hydra
    :param experiment_uid: date time which will be used as ${group}/${experiment_name}/${job_name}/${experiment_uid}
    return checkpoint dir if existent, otherwise None
    """
    last_checkpoint_dir = group_dir / experiment_uid / 'checkpoints'
    if not last_checkpoint_dir.exists():
        return None
    checkpoints = list(last_checkpoint_dir.iterdir())
    last_epoch = max((int(path.stem[6:]) for path in checkpoints if path.stem.startswith('epoch')))
    return last_checkpoint_dir / f'epoch={last_epoch}.ckpt'

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

class TestUtilsCheckpoint(unittest.TestCase):
    """Test checkpoint utils methods."""

    def setUp(self) -> None:
        """Setup test attributes."""
        self.group = Path('exp')
        self.experiment_uid = Path('2023.01.01.00.00.00')
        self.experiment = Path('experiment_name/job_name') / self.experiment_uid

    @patch.object(Path, 'exists', autospec=True, return_value=False)
    def test_find_last_checkpoint_in_dir_dir_unavailable(self, path_exists_mock: Mock) -> None:
        """Test 'find_last_checkpoint_in_dir' method when directory does not exist."""
        group_dir = self.group / self.experiment.parent
        result = find_last_checkpoint_in_dir(group_dir, self.experiment_uid)
        self.assertIsNone(result)

    @patch.object(Path, 'exists', autospec=True, return_value=True)
    @patch.object(Path, 'iterdir', autospec=True, return_value=[Path('epoch=0.ckpt'), Path('epoch=1.ckpt')])
    def test_find_last_checkpoint_in_dir(self, path_iterdir_mock: Mock, path_exists_mock: Mock) -> None:
        """Test 'find_last_checkpoint_in_dir' method under typical use case."""
        group_dir = self.group / self.experiment.parent
        result = find_last_checkpoint_in_dir(group_dir, self.experiment_uid)
        expected = Path('exp/experiment_name/job_name/2023.01.01.00.00.00/checkpoints/epoch=1.ckpt')
        self.assertEqual(result, expected)

    @patch.object(Path, 'iterdir', autospec=True, return_value=[Path('2023.01.01.00.00.00'), Path('2023.01.01.00.00.01'), Path('2023.01.01.00.00.02')])
    @patch(f'{PATCH_PREFIX}.find_last_checkpoint_in_dir', autospec=True)
    def test_extract_last_checkpoint_from_experiment(self, find_last_checkpoint_in_dir_mock: Mock, path_iterdir_mock: Mock) -> None:
        """Test extract_last_checkpoint_from_experiment method."""
        output_dir = self.group / self.experiment
        date_format = '%Y.%m.%d.%H.%M.%S'
        _ = extract_last_checkpoint_from_experiment(output_dir, date_format)
        calls = [call(Path('exp/experiment_name/job_name'), Path('2023.01.01.00.00.02'))]
        find_last_checkpoint_in_dir_mock.assert_has_calls(calls)

class TestUtilsType(unittest.TestCase):
    """Test utils_type functions."""

    def test_is_TorchModuleWrapper_config(self) -> None:
        """Tests that is_TorchModuleWrapper_config works as expected."""
        mock_config = DictConfig({'model_config': 'some_value', 'checkpoint_path': 'some_value', 'some_other_key': 'some_value'})
        expect_true = is_TorchModuleWrapper_config(mock_config)
        self.assertTrue(expect_true)
        mock_config.pop('some_other_key')
        expect_true = is_TorchModuleWrapper_config(mock_config)
        self.assertTrue(expect_true)
        mock_config.pop('model_config')
        expect_false = is_TorchModuleWrapper_config(mock_config)
        self.assertFalse(expect_false)
        mock_config.pop('checkpoint_path')
        expect_false = is_TorchModuleWrapper_config(mock_config)
        self.assertFalse(expect_false)

    def test_is_target_type(self) -> None:
        """Tests that is_target_type works as expected."""
        mock_config_test_utils_mock_type = DictConfig({'_target_': f'{__name__}.TestUtilsTypeMockType'})
        mock_config_test_utils_another_mock_type = DictConfig({'_target_': f'{__name__}.TestUtilsTypeAnotherMockType'})
        expect_true = is_target_type(mock_config_test_utils_mock_type, TestUtilsTypeMockType)
        self.assertTrue(expect_true)
        expect_true = is_target_type(mock_config_test_utils_another_mock_type, TestUtilsTypeAnotherMockType)
        self.assertTrue(expect_true)
        expect_false = is_target_type(mock_config_test_utils_mock_type, TestUtilsTypeAnotherMockType)
        self.assertFalse(expect_false)
        expect_false = is_target_type(mock_config_test_utils_another_mock_type, TestUtilsTypeMockType)
        self.assertFalse(expect_false)

    def test_validate_type(self) -> None:
        """Tests that validate_type works as expected."""
        test_utils_type_mock_type = TestUtilsTypeMockType()
        validate_type(test_utils_type_mock_type, TestUtilsTypeMockType)
        with self.assertRaises(AssertionError):
            validate_type(test_utils_type_mock_type, TestUtilsTypeAnotherMockType)

    def test_are_the_same_type(self) -> None:
        """Tests that are_the_same_type works as expected."""
        test_utils_type_mock_type = TestUtilsTypeMockType()
        another_test_utils_type_mock_type = TestUtilsTypeMockType()
        test_utils_type_another_mock_type = TestUtilsTypeAnotherMockType()
        are_the_same_type(test_utils_type_mock_type, another_test_utils_type_mock_type)
        with self.assertRaises(AssertionError):
            are_the_same_type(test_utils_type_mock_type, test_utils_type_another_mock_type)

    def test_validate_dict_type(self) -> None:
        """Tests that validate_dict_type works as expected."""
        mock_config = DictConfig({'_convert_': 'all', 'correct_object': {'_target_': f'{__name__}.TestUtilsTypeMockType', 'a': 1, 'b': 2.5}, 'correct_object_2': {'_target_': f'{__name__}.TestUtilsTypeMockType', 'a': 1, 'b': 2.5}})
        instantiated_config = hydra.utils.instantiate(mock_config)
        validate_dict_type(instantiated_config, TestUtilsTypeMockType)
        mock_config.other_object = {'_target_': f'{__name__}.TestUtilsTypeAnotherMockType', 'c': 1}
        instantiated_config = hydra.utils.instantiate(mock_config)
        with self.assertRaises(AssertionError):
            validate_dict_type(instantiated_config, TestUtilsTypeMockType)

    def test_find_builder_in_config(self) -> None:
        """Tests that find_builder_in_config works as expected."""
        mock_config = DictConfig({'correct_object': {'_target_': f'{__name__}.TestUtilsTypeMockType', 'a': 1, 'b': 2.5}, 'other_object': {'_target_': f'{__name__}.TestUtilsTypeAnotherMockType', 'c': 1}})
        test_utils_mock_type = find_builder_in_config(mock_config, TestUtilsTypeMockType)
        self.assertTrue(is_target_type(test_utils_mock_type, TestUtilsTypeMockType))
        test_utils_another_mock_type = find_builder_in_config(mock_config, TestUtilsTypeAnotherMockType)
        self.assertTrue(is_target_type(test_utils_another_mock_type, TestUtilsTypeAnotherMockType))
        del mock_config.other_object
        with self.assertRaises(ValueError):
            find_builder_in_config(mock_config, TestUtilsTypeAnotherMockType)

class TestTrainVectorModel(SkeletonTestTrain):
    """
    Test experiments: simple_vector_model, vector_model
    """

    def test_open_loop_training_simple_vector_model(self) -> None:
        """
        Tests simple vector model training in open loop.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, 'py_func=train', '+training=training_simple_vector_model', 'scenario_builder=nuplan_mini', 'scenario_filter.limit_total_scenarios=16', 'splitter=nuplan', 'lightning.trainer.params.max_epochs=1'])
            main(cfg)

    def test_open_loop_training_vector_model(self) -> None:
        """
        Tests vector model training in open loop.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[self.search_path, *self.default_overrides, 'py_func=train', '+training=training_vector_model', 'scenario_builder=nuplan_mini', 'scenario_filter.limit_total_scenarios=16', 'splitter=nuplan', 'model.num_res_blocks=1', 'model.num_attention_layers=1', 'model.feature_dim=8', 'lightning.trainer.params.max_epochs=1'])
            main(cfg)

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

class TestTrain(SkeletonTestTrain):
    """
    Test main training entry point using combinations of models, datasets, filters etc.
    """

    def test_raster_model_overfitting(self) -> None:
        """
        Tests raster model overfitting in open loop.
        """
        loss_threshold = 2.0
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, 'log_config=false', 'py_func=train', '+training=training_raster_model', 'scenario_builder=nuplan_mini', 'scenario_filter.limit_total_scenarios=15', 'splitter=nuplan', 'optimizer.lr=0.01', 'lightning.trainer.overfitting.enable=true', 'lightning.trainer.overfitting.params.max_epochs=200', 'data_loader.params.batch_size=2', 'data_loader.params.num_workers=2'])
            engine = main(cfg)
            self.assertLessEqual(engine.trainer.callback_metrics['loss/train_loss'], loss_threshold)

    def test_urban_driver_open_loop_model_overfitting(self) -> None:
        """
        Tests urban_driver_open_loop model overfitting in open loop.
        """
        loss_threshold = 2.0
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, 'log_config=false', 'py_func=train', '+training=training_urban_driver_open_loop_model', 'data_augmentation=[]', 'scenario_builder=nuplan_mini', 'scenario_filter.limit_total_scenarios=15', 'splitter=nuplan', 'optimizer=adamw', 'optimizer.lr=1.25e-5', 'lightning.trainer.overfitting.enable=true', 'lightning.trainer.overfitting.params.max_epochs=300', 'data_loader.params.batch_size=1', 'data_loader.params.num_workers=2'])
            engine = main(cfg)
            self.assertLessEqual(engine.trainer.callback_metrics['loss/train_loss'], loss_threshold)

class TestTrainOptimizerOCLRScheduler(SkeletonTestTrain):
    """
    Test Optimizer and LR Scheduler instantiation.
    """
    world_size = 4

    def setUp(self) -> None:
        """Setup test attributes."""
        super().setUp()
        self.optimizer_initial_lr = 0.01
        self.div_factor = 20
        self.max_lr = 2
        self.steps_per_epoch = 20

    @patch.dict(os.environ, {'WORLD_SIZE': str(world_size)}, clear=False)
    def test_optimizer_oclr_scheduler_instantiation(self) -> None:
        """
        Tests that optimizer and lr_scheduler were instantiated correctly.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, 'py_func=train', '+training=training_simple_vector_model', 'scenario_builder=nuplan_mini', 'scenario_filter.limit_total_scenarios=30', 'splitter=nuplan', 'lightning.trainer.params.max_epochs=1', 'gpu=false', 'optimizer=adamw', f'optimizer.lr={str(self.optimizer_initial_lr)}', 'lr_scheduler=one_cycle_lr', f'lr_scheduler.div_factor={str(self.div_factor)}', f'lr_scheduler.max_lr={str(self.max_lr)}', f'lr_scheduler.steps_per_epoch={str(self.steps_per_epoch)}'])
            engine = main(cfg)
            self.assertTrue(isinstance(engine.model.optimizers(), torch.optim.AdamW), msg=f'Expected optimizer {torch.optim.AdamW} but got {engine.model.optimizers()}')
            self.assertTrue(isinstance(engine.model.lr_schedulers(), torch.optim.lr_scheduler.OneCycleLR), msg=f'Expected lr_scheduler {torch.optim.lr_scheduler.OneCycleLR} but got {engine.model.lr_schedulers()}')
            expected_base_lr = self.optimizer_initial_lr / self.div_factor
            result_base_lr = engine.model.lr_schedulers().state_dict()['base_lrs'][0]
            self.assertEqual(result_base_lr, expected_base_lr, msg=f'Expected base lr to be {expected_base_lr} but got {result_base_lr}')
            self.tearDown()

class TestDataLoader(unittest.TestCase):
    """
    Tests data loading functionality
    """

    def setUp(self) -> None:
        """Setup hydra config."""
        seed = 10
        pl.seed_everything(seed, workers=True)
        main_path = os.path.dirname(os.path.realpath(__file__))
        self.config_path = os.path.join(main_path, '../config/training/')
        self.group = tempfile.TemporaryDirectory()
        self.cache_path = os.path.join(self.group.name, 'cache_path')

    def tearDown(self) -> None:
        """Remove temporary folder."""
        self.group.cleanup()

    @staticmethod
    def validate_cfg(cfg: DictConfig) -> None:
        """Validate hydra config."""
        update_config_for_training(cfg)
        OmegaConf.set_struct(cfg, False)
        cfg.scenario_filter.limit_total_scenarios = 0.001
        cfg.data_loader.datamodule.train_fraction = 1.0
        cfg.data_loader.datamodule.val_fraction = 1.0
        cfg.data_loader.datamodule.test_fraction = 1.0
        cfg.data_loader.params.batch_size = 2
        cfg.data_loader.params.num_workers = 2
        cfg.data_loader.params.pin_memory = False
        OmegaConf.set_struct(cfg, True)

    @staticmethod
    def _iterate_dataloader(dataloader: torch.utils.data.DataLoader) -> None:
        """
        Iterate a fixed number of batches of the dataloader.
        :param dataloader: Data loader to iterate.
        """
        num_batches = 5
        dataloader_iter = iter(dataloader)
        iterations = min(len(dataloader), num_batches)
        for _ in range(iterations):
            next(dataloader_iter)

    def _run_dataloader(self, cfg: DictConfig) -> None:
        """
        Test that the training dataloader can be iterated without errors.
        :param cfg: Hydra config.
        """
        worker = build_worker(cfg)
        lightning_module_wrapper = build_torch_module_wrapper(cfg.model)
        datamodule = build_lightning_datamodule(cfg, worker, lightning_module_wrapper)
        datamodule.setup('fit')
        datamodule.setup('test')
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        test_dataloader = datamodule.test_dataloader()
        for dataloader in [train_dataloader, val_dataloader]:
            assert len(dataloader) > 0
            self._iterate_dataloader(dataloader)
        self._iterate_dataloader(test_dataloader)

    def test_dataloader(self) -> None:
        """Test dataloader on nuPlan DB."""
        log_names = ['2021.07.16.20.45.29_veh-35_01095_01486', '2021.08.17.18.54.02_veh-45_00665_01065', '2021.06.08.12.54.54_veh-26_04262_04732', '2021.10.06.07.26.10_veh-52_00006_00398']
        overrides = ['scenario_builder=nuplan_mini', 'worker=sequential', 'splitter=nuplan', f'scenario_filter.log_names={log_names}', f'group={self.group.name}', f'cache.cache_path={self.cache_path}', 'output_dir=${group}/${experiment}', 'scenario_type_weights=default_scenario_type_weights']
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*overrides, '+training=training_raster_model'])
            self.validate_cfg(cfg)
            self._run_dataloader(cfg)

class TestTrainRasterModel(SkeletonTestTrain):
    """
    Test experiments: raster_model
    """

    def test_open_loop_training_raster_model(self) -> None:
        """
        Tests raster model training in open loop.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, 'py_func=train', '+training=training_raster_model', 'scenario_builder=nuplan_mini', 'scenario_filter.limit_total_scenarios=16', 'splitter=nuplan', 'model.model_name=resnet18', 'model.pretrained=false', 'model.feature_builders.0.target_width=64', 'model.feature_builders.0.target_height=64', 'lightning.trainer.params.max_epochs=1', 'gpu=false'])
            main(cfg)

class TestModelBuild(unittest.TestCase):
    """Test building model."""

    def setUp(self) -> None:
        """Setup hydra config."""
        main_path = os.path.dirname(os.path.realpath(__file__))
        self.config_path = os.path.join(main_path, '../config/training/')
        self.group = tempfile.TemporaryDirectory()
        self.cache_path = os.path.join(self.group.name, 'cache_path')
        model_path = pathlib.Path(__file__).parent.parent / 'config' / 'common' / 'model'
        self.model_cfg = []
        for model_module in model_path.iterdir():
            model_name = model_module.stem
            with initialize_config_dir(config_dir=self.config_path):
                cfg = compose(config_name=CONFIG_NAME, overrides=['+training=training_raster_model', f'model={model_name}', f'group={self.group.name}', f'cache.cache_path={self.cache_path}'])
                self.model_cfg.append(cfg)

    def tearDown(self) -> None:
        """Remove temporary folder."""
        self.group.cleanup()

    def validate_cfg(self, cfg: DictConfig) -> None:
        """
        Validate that a model can be constructed
        :param cfg: config for model which should be constructed
        """
        lightning_module_wrapper = build_torch_module_wrapper(cfg.model)
        self.assertIsInstance(lightning_module_wrapper, TorchModuleWrapper)
        for builder in lightning_module_wrapper.get_list_of_required_feature():
            self.assertIsInstance(builder, AbstractFeatureBuilder)
        for builder in lightning_module_wrapper.get_list_of_computed_target():
            self.assertIsInstance(builder, AbstractTargetBuilder)

    def test_all_common_models(self) -> None:
        """
        Test construction of all available common models
        """
        for cfg in self.model_cfg:
            self.validate_cfg(cfg)

class TestTrainUrbanDriverOpenLoopModel(SkeletonTestTrain):
    """
    Test experiments: urban_driver_open_loop_model
    """

    def test_open_loop_training_urban_driver_open_loop_model(self) -> None:
        """
        Tests urban_driver_open_loop model training in open loop.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, 'py_func=train', '+training=training_urban_driver_open_loop_model', 'scenario_builder=nuplan_mini', 'scenario_filter.limit_total_scenarios=32', 'splitter=nuplan', 'lightning.trainer.params.max_epochs=1', 'cache.force_feature_computation=True'])
            main(cfg)

class TestTrainProfiling(SkeletonTestTrain):
    """
    Test that profiling gets generated
    """

    def test_simple_vector_model_profiling(self) -> None:
        """
        Tests that profiling file for training gets generated
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*self.default_overrides, 'enable_profiling=True', 'py_func=train', '+training=training_simple_vector_model', 'scenario_builder=nuplan_mini', 'scenario_filter.limit_total_scenarios=16', 'splitter=nuplan', 'lightning.trainer.params.max_epochs=1'])
            main(cfg)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'profiling', 'training.html')))

