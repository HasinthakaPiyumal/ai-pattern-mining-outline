# Cluster 70

class TimerTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @windows_skip_marker
    def test_timer(self):
        sleep_time = Timer()
        with sleep_time:
            time.sleep(5)
        self.assertTrue(sleep_time.total_elapsed_time < 5.2)
        self.assertTrue(sleep_time.total_elapsed_time > 4.9)

    @pytest.mark.notparallel
    def test_timer_with_query(self):
        evadb = get_evadb_for_testing()
        evadb.catalog().reset()
        video_file_path = create_sample_video(NUM_FRAMES)
        load_query = f"LOAD VIDEO '{video_file_path}' INTO MyVideo;"
        transport = MagicMock()
        transport.write = MagicMock(return_value='response_message')
        response = asyncio.run(handle_request(evadb, transport, load_query))
        self.assertTrue(response.error is None)
        self.assertTrue(response.query_time is not None)
        load_query = "LOAD INFILE 'dummy.avi' INTO MyVideo;"
        transport = MagicMock()
        transport.write = MagicMock(return_value='response_message')
        response = asyncio.run(handle_request(evadb, transport, load_query))
        self.assertTrue(response.error is not None)
        self.assertTrue(response.query_time is None)
        file_remove('dummy.avi')

class ReuseTest(unittest.TestCase):

    def _load_hf_model(self):
        function_name = 'HFObjectDetector'
        create_function_query = f"CREATE FUNCTION {function_name}\n            TYPE HuggingFace\n            TASK 'object-detection'\n            MODEL 'facebook/detr-resnet-50';\n        "
        execute_query_fetch_all(self.evadb, create_function_query)

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        ua_detrac = f'{EvaDB_ROOT_DIR}/data/ua_detrac/ua_detrac.mp4'
        execute_query_fetch_all(self.evadb, f"LOAD VIDEO '{ua_detrac}' INTO DETRAC;")
        execute_query_fetch_all(self.evadb, 'CREATE TABLE fruitTable (data TEXT(100))')
        data_list = ['The color of apple is red', 'The color of banana is yellow']
        for data in data_list:
            execute_query_fetch_all(self.evadb, f"INSERT INTO fruitTable (data) VALUES ('{data}')")
        load_functions_for_testing(self.evadb)
        self._load_hf_model()

    def tearDown(self):
        shutdown_ray()
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS DETRAC;')
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS fruitTable;')

    def _verify_reuse_correctness(self, query, reuse_batch):
        gc.collect()
        rules_manager = RulesManager()
        with disable_rules(rules_manager, [CacheFunctionExpressionInApply(), CacheFunctionExpressionInFilter(), CacheFunctionExpressionInProject()]):
            custom_plan_generator = PlanGenerator(self.evadb, rules_manager)
            without_reuse_batch = execute_query_fetch_all(self.evadb, query, plan_generator=custom_plan_generator)
        self.assertEqual(reuse_batch.columns, reuse_batch.columns)
        reuse_batch.sort_orderby(by=[reuse_batch.columns[0]])
        without_reuse_batch.sort_orderby(by=[reuse_batch.columns[0]])
        self.assertEqual(without_reuse_batch, reuse_batch, msg=f'Without reuse {without_reuse_batch} \n With reuse{reuse_batch}')

    def _reuse_experiment(self, queries):
        exec_times = []
        batches = []
        for query in queries:
            timer = Timer()
            with timer:
                batches.append(execute_query_fetch_all(self.evadb, query))
            exec_times.append(timer.total_elapsed_time)
        return (batches, exec_times)

    def _strict_reuse_experiment(self, queries):
        exec_times = []
        batches = []
        for i, query in enumerate(queries):
            timer = Timer()
            if i != 0:
                with timer, patch.object(Batch, 'apply_function_expression') as mock_batch_func:
                    mock_batch_func.side_effect = Exception('Results are not reused')
                    batches.append(execute_query_fetch_all(self.evadb, query))
            else:
                with timer:
                    batches.append(execute_query_fetch_all(self.evadb, query))
            exec_times.append(timer.total_elapsed_time)
        return (batches, exec_times)

    def test_reuse_chatgpt(self):
        from evadb.constants import CACHEABLE_FUNCTIONS
        CACHEABLE_FUNCTIONS += ['DummyLLM']
        select_query = "SELECT DummyLLM('What is the fruit described in this sentence', data)\n            FROM fruitTable"
        batches, exec_times = self._strict_reuse_experiment([select_query, select_query])
        self._verify_reuse_correctness(select_query, batches[1])
        self.assertTrue(exec_times[0] > exec_times[1])

    def test_reuse_when_query_is_duplicate(self):
        select_query = 'SELECT id, label FROM DETRAC JOIN\n            LATERAL HFObjectDetector(data) AS Obj(score, label, bbox) WHERE id < 15;'
        batches, exec_times = self._strict_reuse_experiment([select_query, select_query])
        self._verify_reuse_correctness(select_query, batches[1])
        self.assertTrue(exec_times[0] > exec_times[1])

    @gpu_skip_marker
    def test_reuse_partial(self):
        select_query1 = 'SELECT id, label FROM DETRAC JOIN\n            LATERAL HFObjectDetector(data) AS Obj(score, label, bbox) WHERE id < 5;'
        select_query2 = 'SELECT id, label FROM DETRAC JOIN\n            LATERAL HFObjectDetector(data) AS Obj(score, label, bbox) WHERE id < 15;'
        batches, exec_times = self._reuse_experiment([select_query1, select_query2])
        self._verify_reuse_correctness(select_query2, batches[1])

    @gpu_skip_marker
    def test_reuse_in_with_multiple_occurrences(self):
        select_query1 = 'SELECT id, label FROM DETRAC JOIN\n            LATERAL HFObjectDetector(data) AS Obj(score, label, bbox) WHERE id < 10;'
        select_query2 = 'SELECT id, HFObjectDetector(data).label FROM DETRAC JOIN\n            LATERAL HFObjectDetector(data) AS Obj(score, label, bbox) WHERE id < 5;'
        batches, exec_times = self._reuse_experiment([select_query1, select_query2])
        self._verify_reuse_correctness(select_query2, batches[1])
        select_query = 'SELECT id, HFObjectDetector(data).label FROM DETRAC WHERE id < 15;'
        reuse_batch = execute_query_fetch_all(self.evadb, select_query)
        self._verify_reuse_correctness(select_query, reuse_batch)
        select_query = "SELECT id, HFObjectDetector(data).label FROM DETRAC WHERE ['car'] <@ HFObjectDetector(data).label AND id < 20"
        reuse_batch = execute_query_fetch_all(self.evadb, select_query)
        self._verify_reuse_correctness(select_query, reuse_batch)

    @gpu_skip_marker
    def test_reuse_logical_project_with_duplicate_query(self):
        project_query = 'SELECT id, HFObjectDetector(data).label FROM DETRAC WHERE id < 10;'
        batches, exec_times = self._reuse_experiment([project_query, project_query])
        self._verify_reuse_correctness(project_query, batches[1])
        self.assertGreater(exec_times[0], exec_times[1])

    @gpu_skip_marker
    def test_reuse_with_function_in_predicate(self):
        select_query = "SELECT id FROM DETRAC WHERE ['car'] <@ HFObjectDetector(data).label AND id < 4"
        batches, exec_times = self._reuse_experiment([select_query, select_query])
        self._verify_reuse_correctness(select_query, batches[1])
        self.assertGreater(exec_times[0], exec_times[1])

    @gpu_skip_marker
    def test_reuse_across_different_predicate_using_same_function(self):
        query1 = "SELECT id FROM DETRAC WHERE ['car'] <@ HFObjectDetector(data).label AND id < 15"
        query2 = "SELECT id FROM DETRAC WHERE ArrayCount(HFObjectDetector(data).label, 'car') > 3 AND id < 12;"
        batches, exec_times = self._reuse_experiment([query1, query2])
        self._verify_reuse_correctness(query2, batches[1])
        self.assertGreater(exec_times[0], exec_times[1])

    @gpu_skip_marker
    def test_reuse_filter_with_project(self):
        project_query = '\n            SELECT id, Yolo(data).labels FROM DETRAC WHERE id < 5;'
        select_query = "\n            SELECT id FROM DETRAC\n            WHERE ArrayCount(Yolo(data).labels, 'car') > 3 AND id < 5;"
        batches, exec_times = self._reuse_experiment([project_query, select_query])
        self._verify_reuse_correctness(select_query, batches[1])
        self.assertGreater(exec_times[0], exec_times[1])

    @gpu_skip_marker
    def test_reuse_in_extract_object(self):
        select_query = '\n            SELECT id, T.iids, T.bboxes, T.scores, T.labels\n            FROM DETRAC JOIN LATERAL EXTRACT_OBJECT(data, Yolo, NorFairTracker)\n                AS T(iids, labels, bboxes, scores)\n            WHERE id < 30;\n            '
        batches, exec_times = self._reuse_experiment([select_query, select_query])
        self._verify_reuse_correctness(select_query, batches[1])
        self.assertGreater(exec_times[0], exec_times[1])

    @windows_skip_marker
    def test_reuse_after_server_shutdown(self):
        select_query = 'SELECT id, label FROM DETRAC JOIN\n            LATERAL Yolo(data) AS Obj(label, bbox, conf) WHERE id < 4;'
        execute_query_fetch_all(self.evadb, select_query)
        os.system('nohup evadb_server --stop')
        os.system('nohup evadb_server --start &')
        select_query = 'SELECT id, label FROM DETRAC JOIN\n            LATERAL Yolo(data) AS Obj(label, bbox, conf) WHERE id < 6;'
        reuse_batch = execute_query_fetch_all(self.evadb, select_query)
        self._verify_reuse_correctness(select_query, reuse_batch)
        os.system('nohup evadb_server --stop')

    def test_drop_function_should_remove_cache(self):
        select_query = 'SELECT id, label FROM DETRAC JOIN\n            LATERAL Yolo(data) AS Obj(label, bbox, conf) WHERE id < 5;'
        execute_query_fetch_all(self.evadb, select_query)
        plan = next(get_logical_query_plan(self.evadb, select_query).find_all(LogicalFunctionScan))
        cache_name = plan.func_expr.signature()
        function_cache = self.evadb.catalog().get_function_cache_catalog_entry_by_name(cache_name)
        cache_dir = Path(function_cache.cache_path)
        self.assertIsNotNone(function_cache)
        self.assertTrue(cache_dir.exists())
        execute_query_fetch_all(self.evadb, 'DROP FUNCTION Yolo;')
        function_cache = self.evadb.catalog().get_function_cache_catalog_entry_by_name(cache_name)
        self.assertIsNone(function_cache)
        self.assertFalse(cache_dir.exists())

    def test_drop_table_should_remove_cache(self):
        select_query = 'SELECT id, label FROM DETRAC JOIN\n            LATERAL Yolo(data) AS Obj(label, bbox, conf) WHERE id < 5;'
        execute_query_fetch_all(self.evadb, select_query)
        plan = next(get_logical_query_plan(self.evadb, select_query).find_all(LogicalFunctionScan))
        cache_name = plan.func_expr.signature()
        function_cache = self.evadb.catalog().get_function_cache_catalog_entry_by_name(cache_name)
        cache_dir = Path(function_cache.cache_path)
        self.assertIsNotNone(function_cache)
        self.assertTrue(cache_dir.exists())
        execute_query_fetch_all(self.evadb, 'DROP TABLE DETRAC;')
        function_cache = self.evadb.catalog().get_function_cache_catalog_entry_by_name(cache_name)
        self.assertIsNone(function_cache)
        self.assertFalse(cache_dir.exists())

class DBAPITests(unittest.IsolatedAsyncioTestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self) -> None:
        f = open(suffix_pytest_xdist_worker_id_to_dir('upload.txt'), 'w')
        f.write('dummy data')
        f.close()
        return super().setUp()

    def tearDown(self) -> None:
        os.remove(suffix_pytest_xdist_worker_id_to_dir('upload.txt'))
        return super().tearDown()

    def test_evadb_cursor_execute_async(self):
        connection = AsyncMock()
        evadb_cursor = EvaDBCursor(connection)
        query = 'test_query'
        asyncio.run(evadb_cursor.execute_async(query))
        self.assertEqual(evadb_cursor._pending_query, True)
        with self.assertRaises(SystemError):
            asyncio.run(evadb_cursor.execute_async(query))

    def test_evadb_cursor_fetch_all_async(self):
        connection = AsyncMock()
        evadb_cursor = EvaDBCursor(connection)
        message = 'test_response'
        serialized_message = Response.serialize('test_response')
        serialized_message_length = b'%d' % len(serialized_message)
        connection._reader.readline.side_effect = [serialized_message_length]
        connection._reader.readexactly.side_effect = [serialized_message]
        response = asyncio.run(evadb_cursor.fetch_all_async())
        self.assertEqual(evadb_cursor._pending_query, False)
        self.assertEqual(message, response)

    def test_evadb_cursor_fetch_one_sync(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        connection = AsyncMock()
        evadb_cursor = EvaDBCursor(connection)
        message = 'test_response'
        serialized_message = Response.serialize('test_response')
        serialized_message_length = b'%d' % len(serialized_message)
        connection._reader.readline.side_effect = [serialized_message_length]
        connection._reader.readexactly.side_effect = [serialized_message]
        response = evadb_cursor.fetch_one()
        self.assertEqual(evadb_cursor._pending_query, False)
        self.assertEqual(message, response)

    def test_evadb_connection(self):
        hostname = 'localhost'
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        connection = AsyncMock()
        evadb_cursor = EvaDBCursor(connection)
        with self.assertRaises(AttributeError):
            evadb_cursor.__getattr__('foo')
        with self.assertRaises(OSError):
            connect_remote(hostname, port=1)

    async def test_evadb_signal(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        connection = AsyncMock()
        evadb_cursor = EvaDBCursor(connection)
        query = 'test_query'
        await evadb_cursor.execute_async(query)

    def test_client_stop_query(self):
        connection = AsyncMock()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        connection.protocol.loop = loop
        evadb_cursor = EvaDBCursor(connection)
        evadb_cursor.execute('test_query')
        evadb_cursor.stop_query()
        self.assertEqual(evadb_cursor._pending_query, False)

    def test_get_attr(self):
        connection = AsyncMock()
        evadb_cursor = EvaDBCursor(connection)
        with self.assertRaises(AttributeError):
            evadb_cursor.missing_function()

    @patch('asyncio.open_connection')
    def test_get_connection(self, mock_open):
        server_reader = asyncio.StreamReader()
        server_writer = MagicMock()
        mock_open.return_value = (server_reader, server_writer)
        connection = connect_remote('localhost', port=1)
        self.assertNotEqual(connection, None)

def connect_remote(host: str, port: int) -> EvaDBConnection:
    connection = asyncio.run(get_connection(host, port))
    return connection

class ResponseTest(unittest.TestCase):

    def test_server_response_serialize_deserialize(self):
        batch = Batch(frames=create_dataframe())
        response = Response(status=ResponseStatus.SUCCESS, batch=batch)
        response2 = Response.deserialize(response.serialize())
        self.assertEqual(response, response2)

def execute_query(evadb: EvaDBDatabase, query, report_time: bool=False, do_not_raise_exceptions: bool=False, do_not_print_exceptions: bool=False, **kwargs) -> Iterator[Batch]:
    """
    Execute the query and return a result generator.
    """
    query_compile_time = Timer()
    with query_compile_time:
        stmt = Parser().parse(query)[0]
        res_batch = execute_statement(evadb, stmt, do_not_raise_exceptions, do_not_print_exceptions, **kwargs)
    if report_time is True:
        query_compile_time.log_elapsed_time('Query Compile Time')
    return res_batch

class JobScheduler:

    def __init__(self, evadb: EvaDBDatabase) -> None:
        self.poll_interval_seconds = 30
        self._evadb = evadb

    def _update_next_schedule_run(self, job_catalog_entry: JobCatalogEntry) -> bool:
        job_end_time = job_catalog_entry.end_time
        active_status = False
        if job_catalog_entry.repeat_interval and job_catalog_entry.repeat_interval > 0:
            next_trigger_time = datetime.datetime.now() + datetime.timedelta(seconds=job_catalog_entry.repeat_interval)
            if not job_end_time or next_trigger_time < job_end_time:
                active_status = True
        next_trigger_time = next_trigger_time if active_status else job_catalog_entry.next_scheduled_run
        self._evadb.catalog().update_job_catalog_entry(job_catalog_entry.name, next_trigger_time, active_status)
        return (active_status, next_trigger_time)

    def _get_sleep_time(self, next_job_entry: JobCatalogEntry) -> int:
        sleep_time = self.poll_interval_seconds
        if next_job_entry:
            sleep_time = min(sleep_time, (next_job_entry.next_scheduled_run - datetime.datetime.now()).total_seconds())
        sleep_time = max(0, sleep_time)
        return sleep_time

    def _scan_and_execute_jobs(self):
        while True:
            try:
                for next_executable_job in iter(lambda: self._evadb.catalog().get_next_executable_job(only_past_jobs=True), None):
                    execution_time = datetime.datetime.now()
                    self._evadb.catalog().insert_job_history_catalog_entry(next_executable_job.row_id, next_executable_job.name, execution_time, None)
                    execution_results = [execute_query(self._evadb, query) for query in next_executable_job.queries]
                    logger.debug(f'Exection result for job: {next_executable_job.name} results: {execution_results}')
                    self._update_next_schedule_run(next_executable_job)
                    self._evadb.catalog().update_job_history_end_time(next_executable_job.row_id, execution_time, datetime.datetime.now())
                next_executable_job = self._evadb.catalog().get_next_executable_job(only_past_jobs=False)
                sleep_time = self._get_sleep_time(next_executable_job)
                if sleep_time > 0:
                    logger.debug(f'Job scheduler process sleeping for {sleep_time} seconds')
                    time.sleep(sleep_time)
            except Exception as e:
                logger.error(f'Got an exception in job scheduler: {str(e)}')
                time.sleep(self.poll_interval_seconds * 0.2)

    def execute(self):
        try:
            self._scan_and_execute_jobs()
        except KeyboardInterrupt:
            logger.debug('Exiting the job scheduler process due to interrupt')
            sys.exit()

