# Cluster 39

def suffix_pytest_xdist_worker_id_to_dir(path: str):
    try:
        worker_id = os.environ['PYTEST_XDIST_WORKER']
        path = Path(str(worker_id) + '_' + path)
    except KeyError:
        pass
    return Path(path)

def s3_dir():
    db_dir = suffix_pytest_xdist_worker_id_to_dir(EvaDB_DATABASE_DIR)
    return db_dir / S3_DOWNLOAD_DIR

@pytest.mark.notparallel
class TextFilteringTests(unittest.TestCase):

    def setUp(self):
        self.db_dir = suffix_pytest_xdist_worker_id_to_dir(EvaDB_DATABASE_DIR)
        self.conn = connect(self.db_dir)
        self.evadb = self.conn._evadb
        self.evadb.catalog().reset()

    def tearDown(self):
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyPDFs;')

    def test_text_filter(self):
        pdf_path = f'{EvaDB_ROOT_DIR}/data/documents/layout-parser-paper.pdf'
        cursor = self.conn.cursor()
        cursor.load(pdf_path, 'MyPDFs', 'pdf').df()
        load_pdf_data = cursor.table('MyPDFs').df()
        cursor.create_function('TextFilterKeyword', True, f'{EvaDB_ROOT_DIR}/evadb/functions/text_filter_keyword.py').df()
        filtered_data = cursor.table('MyPDFs').cross_apply("TextFilterKeyword(data, ['References'])", 'objs(filtered)').df()
        filtered_data.dropna(inplace=True)
        import pandas as pd
        pd.set_option('display.max_colwidth', None)
        self.assertNotEqual(len(filtered_data), len(load_pdf_data))

def connect(evadb_dir: str=EvaDB_DATABASE_DIR, sql_backend: str=None) -> EvaDBConnection:
    """
    Connects to the EvaDB server and returns a connection object.

    Args:
        evadb_dir (str): The directory used by EvaDB to store database-related content. Default is "evadb".
        sql_backend (str): Custom database URI to be used. We follow the SQLAlchemy database URL format.
            Default is SQLite in the EvaDB directory. See https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls.

    Returns:
        EvaDBConnection: A connection object representing the connection to the EvaDB database.

    Examples:
        >>> from evadb import connect
        >>> conn = connect()
    """
    evadb = init_evadb_instance(evadb_dir, custom_db_uri=sql_backend)
    init_builtin_functions(evadb, mode='release')
    return EvaDBConnection(evadb, None, None)

@pytest.mark.notparallel
class SaliencyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_dir = suffix_pytest_xdist_worker_id_to_dir(EvaDB_DATABASE_DIR)
        cls.conn = connect(cls.db_dir)
        cls.evadb = cls.conn._evadb

    @classmethod
    def tearDownClass(cls):
        shutdown_ray()
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS SALIENCY;')

    @unittest.skip('Not supported in current version')
    def test_saliency(self):
        Saliency1 = f'{EvaDB_ROOT_DIR}/data/saliency/test1.jpeg'
        create_function_query = f"LOAD IMAGE '{Saliency1}' INTO SALIENCY;"
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS SALIENCY;')
        execute_query_fetch_all(self.evadb, create_function_query)
        execute_query_fetch_all(self.evadb, 'DROP FUNCTION IF EXISTS SaliencyFeatureExtractor')
        create_function_query = f"CREATE FUNCTION IF NOT EXISTS SaliencyFeatureExtractor\n                    IMPL  '{EvaDB_ROOT_DIR}/evadb/functions/saliency_feature_extractor.py';\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        select_query_saliency = 'SELECT data, SaliencyFeatureExtractor(data)\n                  FROM SALIENCY\n        '
        actual_batch_saliency = execute_query_fetch_all(self.evadb, select_query_saliency)
        self.assertEqual(len(actual_batch_saliency.columns), 2)

class RelationalAPI(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.db_dir = suffix_pytest_xdist_worker_id_to_dir(EvaDB_DATABASE_DIR)
        cls.conn = connect(cls.db_dir)
        cls.evadb = cls.conn._evadb

    def setUp(self):
        self.evadb.catalog().reset()
        self.mnist_path = f'{EvaDB_ROOT_DIR}/data/mnist/mnist.mp4'
        load_functions_for_testing(self.evadb)
        self.images = f'{EvaDB_ROOT_DIR}/data/detoxify/*.jpg'

    def tearDown(self):
        shutdown_ray()
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS mnist_video;')
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS meme_images;')
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS dummy_table;')

    def test_relation_apis(self):
        cursor = self.conn.cursor()
        rel = cursor.load(self.mnist_path, table_name='mnist_video', format='video')
        rel.execute()
        rel = cursor.table('mnist_video')
        assert_frame_equal(rel.df(), cursor.query('select * from mnist_video;').df())
        rel = rel.select('_row_id, id, data')
        assert_frame_equal(rel.df(), cursor.query('select _row_id, id, data from mnist_video;').df())
        rel = rel.filter('id < 10')
        assert_frame_equal(rel.df(), cursor.query('select _row_id, id, data from mnist_video where id < 10;').df())
        rel = rel.cross_apply('unnest(MnistImageClassifier(data))', 'mnist(label)').filter('mnist.label = 1').select('_row_id, id')
        query = ' select _row_id, id\n                    from mnist_video\n                        join lateral unnest(MnistImageClassifier(data)) AS mnist(label)\n                    where id < 10 AND mnist.label = 1;'
        assert_frame_equal(rel.df(), cursor.query(query).df())
        rel = cursor.load(self.images, table_name='meme_images', format='image')
        rel.execute()
        rel = cursor.table('meme_images').select('_row_id, name')
        assert_frame_equal(rel.df(), cursor.query('select _row_id, name from meme_images;').df())
        rel = rel.filter('_row_id < 3')
        assert_frame_equal(rel.df(), cursor.query('select _row_id, name from meme_images where _row_id < 3;').df())

    def test_relation_api_chaining(self):
        cursor = self.conn.cursor()
        rel = cursor.load(self.mnist_path, table_name='mnist_video', format='video')
        rel.execute()
        rel = cursor.table('mnist_video').select('id, data').filter('id > 10').filter('id < 20')
        assert_frame_equal(rel.df(), cursor.query('select id, data from mnist_video where id > 10 AND id < 20;').df())

    def test_interleaving_calls(self):
        cursor = self.conn.cursor()
        rel = cursor.load(self.mnist_path, table_name='mnist_video', format='video')
        rel.execute()
        rel = cursor.table('mnist_video')
        filtered_rel = rel.filter('id > 10')
        assert_frame_equal(rel.filter('id > 10').df(), cursor.query('select * from mnist_video where id > 10;').df())
        assert_frame_equal(filtered_rel.select('_row_id, id').df(), cursor.query('select _row_id, id from mnist_video where id > 10;').df())

    @qdrant_skip_marker
    def test_create_index(self):
        cursor = self.conn.cursor()
        rel = cursor.load(self.images, table_name='meme_images', format='image')
        rel.execute()
        cursor.query(f"CREATE FUNCTION IF NOT EXISTS SiftFeatureExtractor\n                IMPL  '{EvaDB_ROOT_DIR}/evadb/functions/sift_feature_extractor.py'").df()
        cursor.create_vector_index('faiss_index', table_name='meme_images', expr='SiftFeatureExtractor(data)', using='QDRANT').df()
        base_image = f'{EvaDB_ROOT_DIR}/data/detoxify/meme1.jpg'
        rel = cursor.table('meme_images').order(f"Similarity(SiftFeatureExtractor(Open('{base_image}')), SiftFeatureExtractor(data))").limit(1).select('name')
        similarity_sql = 'SELECT name FROM meme_images\n                            ORDER BY\n                                Similarity(SiftFeatureExtractor(Open("{}")), SiftFeatureExtractor(data))\n                            LIMIT 1;'.format(base_image)
        assert_frame_equal(rel.df(), cursor.query(similarity_sql).df())

    def test_create_function_with_relational_api(self):
        video_file_path = create_sample_video(10)
        cursor = self.conn.cursor()
        rel = cursor.load(video_file_path, table_name='dummy_video', format='video')
        rel.execute()
        create_dummy_object_detector_function = cursor.create_function('DummyObjectDetector', if_not_exists=True, impl_path='test/util.py')
        create_dummy_object_detector_function.execute()
        args = {'task': 'automatic-speech-recognition', 'model': 'openai/whisper-base'}
        create_speech_recognizer_function_if_not_exists = cursor.create_function('SpeechRecognizer', if_not_exists=True, type='HuggingFace', **args)
        query = create_speech_recognizer_function_if_not_exists.sql_query()
        self.assertEqual(query, "CREATE FUNCTION IF NOT EXISTS SpeechRecognizer TYPE HuggingFace TASK 'automatic-speech-recognition' MODEL 'openai/whisper-base'")
        create_speech_recognizer_function_if_not_exists.execute()
        create_speech_recognizer_function = cursor.create_function('SpeechRecognizer', if_not_exists=False, type='HuggingFace', **args)
        query = create_speech_recognizer_function.sql_query()
        self.assertEqual(query, "CREATE FUNCTION SpeechRecognizer TYPE HuggingFace TASK 'automatic-speech-recognition' MODEL 'openai/whisper-base'")
        with self.assertRaises(ExecutorError):
            create_speech_recognizer_function.execute()
        select_query_sql = 'SELECT id, DummyObjectDetector(data) FROM dummy_video ORDER BY id;'
        actual_batch = cursor.query(select_query_sql).execute()
        labels = DummyObjectDetector().labels
        expected = [{'id': i, 'label': np.array([labels[1 + i % 2]])} for i in range(10)]
        expected_batch = Batch(frames=pd.DataFrame(expected))
        self.assertEqual(actual_batch, expected_batch)
        actual_batch = cursor.query(select_query_sql).execute(drop_alias=False)
        expected = [{'dummy_video.id': i, 'dummyobjectdetector.label': np.array([labels[1 + i % 2]])} for i in range(10)]
        expected_batch = Batch(frames=pd.DataFrame(expected))
        self.assertEqual(actual_batch, expected_batch)

    def test_drop_with_relational_api(self):
        video_file_path = create_sample_video(10)
        cursor = self.conn.cursor()
        rel = cursor.load(video_file_path, table_name='dummy_video', format='video')
        rel.execute()
        create_dummy_object_detector_function = cursor.create_function('DummyObjectDetector', if_not_exists=True, impl_path='test/util.py')
        create_dummy_object_detector_function.execute()
        drop_dummy_object_detector_function = cursor.drop_function('DummyObjectDetector', if_exists=True)
        drop_dummy_object_detector_function.execute()
        select_query_sql = 'SELECT id, DummyObjectDetector(data) FROM dummy_video ORDER BY id;'
        with self.assertRaises(BinderError):
            cursor.query(select_query_sql).execute()
        drop_dummy_object_detector_function = cursor.drop_function('DummyObjectDetector', if_exists=True)
        drop_dummy_object_detector_function.execute()
        drop_dummy_object_detector_function = cursor.drop_function('DummyObjectDetector', if_exists=False)
        with self.assertRaises(ExecutorError):
            drop_dummy_object_detector_function.execute()
        drop_table = cursor.drop_table('dummy_video', if_exists=True)
        drop_table.execute()
        select_query_sql = 'SELECT id, data FROM dummy_video ORDER BY id;'
        with self.assertRaises(BinderError):
            cursor.query(select_query_sql).execute()
        drop_table = cursor.drop_table('dummy_video', if_exists=True)
        drop_table.execute()
        drop_table = cursor.drop_table('dummy_video', if_exists=False)
        with self.assertRaises(ExecutorError):
            drop_table.execute()

    def test_pdf_similarity_search(self):
        conn = connect()
        cursor = conn.cursor()
        pdf_path = f'{EvaDB_ROOT_DIR}/data/documents/state_of_the_union.pdf'
        load_pdf = cursor.load(file_regex=pdf_path, format='PDF', table_name='PDFs')
        load_pdf.execute()
        function_check = cursor.drop_function('SentenceFeatureExtractor')
        function_check.df()
        function = cursor.create_function('SentenceFeatureExtractor', True, f'{EvaDB_ROOT_DIR}/evadb/functions/sentence_feature_extractor.py')
        function.execute()
        cursor.create_vector_index('faiss_index', table_name='PDFs', expr='SentenceFeatureExtractor(data)', using='FAISS').df()
        query = cursor.table('PDFs').order("Similarity(\n                    SentenceFeatureExtractor('When was the NATO created?'), SentenceFeatureExtractor(data)\n                ) DESC").limit(3).select('data')
        output = query.df()
        self.assertEqual(len(output), 3)
        self.assertTrue('data' in output.columns)
        output = query.df(drop_alias=False)
        self.assertTrue('pdfs.data' in output.columns)
        cursor.drop_index('faiss_index').df()

    def test_langchain_split_doc(self):
        conn = connect()
        cursor = conn.cursor()
        pdf_path1 = f'{EvaDB_ROOT_DIR}/data/documents/state_of_the_union.pdf'
        load_pdf = cursor.load(file_regex=pdf_path1, format='DOCUMENT', table_name='docs')
        load_pdf.execute()
        result1 = cursor.table('docs', chunk_size=2000, chunk_overlap=DEFAULT_DOCUMENT_CHUNK_OVERLAP).select('data').df()
        result2 = cursor.table('docs', chunk_size=DEFAULT_DOCUMENT_CHUNK_SIZE, chunk_overlap=2000).select('data').df()
        result3 = cursor.table('docs', chunk_size=DEFAULT_DOCUMENT_CHUNK_SIZE, chunk_overlap=0).select('data').df()
        self.assertGreater(len(result1), len(result2))
        self.assertGreater(len(result2), len(result3))
        result5 = cursor.table('docs', chunk_size=2000).select('data').df()
        self.assertEqual(len(result5), len(result1))
        result4 = cursor.table('docs', chunk_overlap=0).select('data').df()
        self.assertEqual(len(result3), len(result4))
        result1 = cursor.table('docs').select('data').df()
        result2 = cursor.query(f'SELECT data from docs chunk_size {DEFAULT_DOCUMENT_CHUNK_SIZE} chunk_overlap {DEFAULT_DOCUMENT_CHUNK_OVERLAP}').df()
        self.assertEqual(len(result1), len(result2))

    def test_show_relational(self):
        video_file_path = create_sample_video(10)
        cursor = self.conn.cursor()
        rel = cursor.load(video_file_path, table_name='dummy_video', format='video')
        rel.execute()
        result = cursor.show('tables').df()
        self.assertEqual(len(result), 1)
        self.assertEqual(result['name'][0], 'dummy_video')

    def test_explain_relational(self):
        video_file_path = create_sample_video(10)
        cursor = self.conn.cursor()
        rel = cursor.load(video_file_path, table_name='dummy_video', format='video')
        rel.execute()
        result = cursor.explain('SELECT * FROM dummy_video').df()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], '|__ ProjectPlan\n    |__ SeqScanPlan\n        |__ StoragePlan\n')

    def test_rename_relational(self):
        video_file_path = create_sample_video(10)
        cursor = self.conn.cursor()
        rel = cursor.load(video_file_path, table_name='dummy_video', format='video')
        rel.execute()
        cursor.rename('dummy_video', 'dummy_video_renamed').df()
        result = cursor.show('tables').df()
        self.assertEqual(len(result), 1)
        self.assertEqual(result['name'][0], 'dummy_video_renamed')

    def test_create_table_relational(self):
        cursor = self.conn.cursor()
        cursor.create_table(table_name='dummy_table', if_not_exists=True, columns='id INTEGER, name text(30)').df()
        result = cursor.show('tables').df()
        self.assertEqual(len(result), 1)
        self.assertEqual(result['name'][0], 'dummy_table')
        rel = cursor.create_table(table_name='dummy_table', if_not_exists=False, columns='id INTEGER, name text(30)')
        with self.assertRaises(ExecutorError):
            rel.execute()

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

@pytest.mark.notparallel
class RulesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        video_file_path = create_sample_video()
        load_query = f"LOAD VIDEO '{video_file_path}' INTO MyVideo;"
        execute_query_fetch_all(cls.evadb, load_query)

    @classmethod
    def tearDownClass(cls):
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS MyVideo;')

    def test_rules_promises_order(self):
        rewrite_promises = [Promise.LOGICAL_INNER_JOIN_COMMUTATIVITY, Promise.EMBED_FILTER_INTO_GET, Promise.EMBED_SAMPLE_INTO_GET, Promise.XFORM_LATERAL_JOIN_TO_LINEAR_FLOW, Promise.PUSHDOWN_FILTER_THROUGH_JOIN, Promise.PUSHDOWN_FILTER_THROUGH_APPLY_AND_MERGE, Promise.COMBINE_SIMILARITY_ORDERBY_AND_LIMIT_TO_VECTOR_INDEX_SCAN, Promise.REORDER_PREDICATES, Promise.XFORM_EXTRACT_OBJECT_TO_LINEAR_FLOW]
        for promise in rewrite_promises:
            self.assertTrue(promise > Promise.IMPLEMENTATION_DELIMITER)
        implementation_promises = [Promise.LOGICAL_EXCHANGE_TO_PHYSICAL, Promise.LOGICAL_UNION_TO_PHYSICAL, Promise.LOGICAL_GROUPBY_TO_PHYSICAL, Promise.LOGICAL_ORDERBY_TO_PHYSICAL, Promise.LOGICAL_LIMIT_TO_PHYSICAL, Promise.LOGICAL_INSERT_TO_PHYSICAL, Promise.LOGICAL_DELETE_TO_PHYSICAL, Promise.LOGICAL_RENAME_TO_PHYSICAL, Promise.LOGICAL_DROP_OBJECT_TO_PHYSICAL, Promise.LOGICAL_LOAD_TO_PHYSICAL, Promise.LOGICAL_CREATE_TO_PHYSICAL, Promise.LOGICAL_CREATE_FROM_SELECT_TO_PHYSICAL, Promise.LOGICAL_CREATE_FUNCTION_TO_PHYSICAL, Promise.LOGICAL_CREATE_FUNCTION_FROM_SELECT_TO_PHYSICAL, Promise.LOGICAL_SAMPLE_TO_UNIFORMSAMPLE, Promise.LOGICAL_GET_TO_SEQSCAN, Promise.LOGICAL_DERIVED_GET_TO_PHYSICAL, Promise.LOGICAL_LATERAL_JOIN_TO_PHYSICAL, Promise.LOGICAL_JOIN_TO_PHYSICAL_HASH_JOIN, Promise.LOGICAL_JOIN_TO_PHYSICAL_NESTED_LOOP_JOIN, Promise.LOGICAL_FUNCTION_SCAN_TO_PHYSICAL, Promise.LOGICAL_FILTER_TO_PHYSICAL, Promise.LOGICAL_PROJECT_TO_PHYSICAL, Promise.LOGICAL_PROJECT_NO_TABLE_TO_PHYSICAL, Promise.LOGICAL_SHOW_TO_PHYSICAL, Promise.LOGICAL_EXPLAIN_TO_PHYSICAL, Promise.LOGICAL_CREATE_INDEX_TO_VECTOR_INDEX, Promise.LOGICAL_APPLY_AND_MERGE_TO_PHYSICAL, Promise.LOGICAL_VECTOR_INDEX_SCAN_TO_PHYSICAL]
        for promise in implementation_promises:
            self.assertTrue(promise < Promise.IMPLEMENTATION_DELIMITER)
        promise_count = len(Promise)
        rewrite_count = len(set(rewrite_promises))
        implementation_count = len(set(implementation_promises))
        self.assertEqual(rewrite_count + implementation_count + 4, promise_count)

    def test_supported_rules(self):
        supported_rewrite_rules = [EmbedFilterIntoGet(), EmbedSampleIntoGet(), XformLateralJoinToLinearFlow(), PushDownFilterThroughApplyAndMerge(), PushDownFilterThroughJoin(), CombineSimilarityOrderByAndLimitToVectorIndexScan(), ReorderPredicates(), XformExtractObjectToLinearFlow()]
        rewrite_rules = RulesManager().stage_one_rewrite_rules + RulesManager().stage_two_rewrite_rules
        self.assertEqual(len(supported_rewrite_rules), len(rewrite_rules))
        for rule in supported_rewrite_rules:
            self.assertTrue(any((isinstance(rule, type(x)) for x in rewrite_rules)))
        supported_logical_rules = [LogicalInnerJoinCommutativity(), CacheFunctionExpressionInApply(), CacheFunctionExpressionInFilter(), CacheFunctionExpressionInProject()]
        self.assertEqual(len(supported_logical_rules), len(RulesManager().logical_rules))
        for rule in supported_logical_rules:
            self.assertTrue(any((isinstance(rule, type(x)) for x in RulesManager().logical_rules)))
        ray_enabled = self.evadb.catalog().get_configuration_catalog_value('ray')
        ray_enabled_and_installed = is_ray_enabled_and_installed(ray_enabled)
        supported_implementation_rules = [LogicalCreateToPhysical(), LogicalCreateFromSelectToPhysical(), LogicalRenameToPhysical(), LogicalCreateFunctionToPhysical(), LogicalCreateFunctionFromSelectToPhysical(), LogicalDropObjectToPhysical(), LogicalInsertToPhysical(), LogicalDeleteToPhysical(), LogicalLoadToPhysical(), LogicalGetToSeqScan(), LogicalProjectToRayPhysical() if ray_enabled_and_installed else LogicalProjectToPhysical(), LogicalProjectNoTableToPhysical(), LogicalDerivedGetToPhysical(), LogicalUnionToPhysical(), LogicalGroupByToPhysical(), LogicalOrderByToPhysical(), LogicalLimitToPhysical(), LogicalJoinToPhysicalNestedLoopJoin(), LogicalLateralJoinToPhysical(), LogicalFunctionScanToPhysical(), LogicalJoinToPhysicalHashJoin(), LogicalFilterToPhysical(), LogicalApplyAndMergeToRayPhysical() if ray_enabled_and_installed else LogicalApplyAndMergeToPhysical(), LogicalShowToPhysical(), LogicalExplainToPhysical(), LogicalCreateIndexToVectorIndex(), LogicalVectorIndexScanToPhysical()]
        if ray_enabled_and_installed:
            supported_implementation_rules.append(LogicalExchangeToPhysical())
        self.assertEqual(len(supported_implementation_rules), len(RulesManager().implementation_rules))
        for rule in supported_implementation_rules:
            self.assertTrue(any((isinstance(rule, type(x)) for x in RulesManager().implementation_rules)))

    def test_simple_filter_into_get(self):
        rule = EmbedFilterIntoGet()
        predicate = MagicMock()
        logi_get = LogicalGet(MagicMock(), MagicMock(), MagicMock())
        logi_filter = LogicalFilter(predicate, [logi_get])
        rewrite_opr = next(rule.apply(logi_filter, MagicMock()))
        self.assertFalse(rewrite_opr is logi_get)
        self.assertEqual(rewrite_opr.predicate, predicate)

    def test_embed_sample_into_get_does_not_work_with_structured_data(self):
        rule = EmbedSampleIntoGet()
        table_obj = TableCatalogEntry(name='foo', table_type=TableType.STRUCTURED_DATA, file_url=MagicMock())
        logi_get = LogicalGet(MagicMock(), table_obj, MagicMock(), MagicMock())
        logi_sample = LogicalSample(MagicMock(), MagicMock(), children=[logi_get])
        self.assertFalse(rule.check(logi_sample, MagicMock()))

    def test_disable_rules(self):
        rules_manager = RulesManager()
        with disable_rules(rules_manager, [PushDownFilterThroughApplyAndMerge()]):
            self.assertFalse(any((isinstance(PushDownFilterThroughApplyAndMerge, type(x)) for x in rules_manager.stage_two_rewrite_rules)))

    def test_xform_lateral_join_does_not_work_with_other_join(self):
        rule = XformLateralJoinToLinearFlow()
        logi_join = LogicalJoin(JoinType.INNER_JOIN)
        self.assertFalse(rule.check(logi_join, MagicMock()))

    def test_rule_base_errors(self):
        with patch.object(Rule, '__abstractmethods__', set()):
            rule = Rule(rule_type=RuleType.INVALID_RULE)
            with self.assertRaises(NotImplementedError):
                rule.promise()
            with self.assertRaises(NotImplementedError):
                rule.check(MagicMock(), MagicMock())
            with self.assertRaises(NotImplementedError):
                rule.apply(MagicMock())

@pytest.mark.notparallel
class CatalogManagerTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls) -> None:
        cls.mocks = [mock.patch('evadb.catalog.catalog_manager.SQLConfig'), mock.patch('evadb.catalog.catalog_manager.init_db')]
        for single_mock in cls.mocks:
            single_mock.start()
            cls.addClassCleanup(single_mock.stop)

    @mock.patch('evadb.catalog.catalog_manager.init_db')
    def test_catalog_bootstrap(self, mocked_db):
        x = CatalogManager(MagicMock())
        x._bootstrap_catalog()
        mocked_db.assert_called()

    @mock.patch('evadb.catalog.catalog_manager.CatalogManager.create_and_insert_table_catalog_entry')
    def test_create_multimedia_table_catalog_entry(self, mock):
        x = CatalogManager(MagicMock())
        name = 'myvideo'
        x.create_and_insert_multimedia_table_catalog_entry(name=name, format_type=FileFormatType.VIDEO)
        columns = get_video_table_column_definitions()
        mock.assert_called_once_with(TableInfo(name), columns, table_type=TableType.VIDEO_DATA)

    @mock.patch('evadb.catalog.catalog_manager.init_db')
    @mock.patch('evadb.catalog.catalog_manager.TableCatalogService')
    def test_insert_table_catalog_entry_should_create_table_and_columns(self, ds_mock, initdb_mock):
        catalog = CatalogManager(MagicMock())
        file_url = 'file1'
        table_name = 'name'
        columns = [ColumnCatalogEntry('c1', ColumnType.INTEGER)]
        catalog.insert_table_catalog_entry(table_name, file_url, columns)
        ds_mock.return_value.insert_entry.assert_called_with(table_name, file_url, identifier_column='id', table_type=TableType.VIDEO_DATA, column_list=[ANY] + columns)

    @mock.patch('evadb.catalog.catalog_manager.init_db')
    @mock.patch('evadb.catalog.catalog_manager.TableCatalogService')
    def test_get_table_catalog_entry_when_table_exists(self, ds_mock, initdb_mock):
        catalog = CatalogManager(MagicMock())
        table_name = 'name'
        database_name = 'database'
        row_id = 1
        table_obj = MagicMock(row_id=row_id)
        ds_mock.return_value.get_entry_by_name.return_value = table_obj
        actual = catalog.get_table_catalog_entry(table_name, database_name)
        ds_mock.return_value.get_entry_by_name.assert_called_with(database_name, table_name)
        self.assertEqual(actual.row_id, row_id)

    @mock.patch('evadb.catalog.catalog_manager.init_db')
    @mock.patch('evadb.catalog.catalog_manager.TableCatalogService')
    @mock.patch('evadb.catalog.catalog_manager.ColumnCatalogService')
    def test_get_table_catalog_entry_when_table_doesnot_exists(self, dcs_mock, ds_mock, initdb_mock):
        catalog = CatalogManager(MagicMock())
        table_name = 'name'
        database_name = 'database'
        table_obj = None
        ds_mock.return_value.get_entry_by_name.return_value = table_obj
        actual = catalog.get_table_catalog_entry(table_name, database_name)
        ds_mock.return_value.get_entry_by_name.assert_called_with(database_name, table_name)
        dcs_mock.return_value.filter_entries_by_table_id.assert_not_called()
        self.assertEqual(actual, table_obj)

    @mock.patch('evadb.catalog.catalog_manager.FunctionCatalogService')
    @mock.patch('evadb.catalog.catalog_manager.FunctionIOCatalogService')
    @mock.patch('evadb.catalog.catalog_manager.FunctionMetadataCatalogService')
    @mock.patch('evadb.catalog.catalog_manager.get_file_checksum')
    def test_insert_function(self, checksum_mock, functionmetadata_mock, functionio_mock, function_mock):
        catalog = CatalogManager(MagicMock())
        function_io_list = [MagicMock()]
        function_metadata_list = [MagicMock()]
        actual = catalog.insert_function_catalog_entry('function', 'sample.py', 'classification', function_io_list, function_metadata_list)
        function_mock.return_value.insert_entry.assert_called_with('function', 'sample.py', 'classification', checksum_mock.return_value, function_io_list, function_metadata_list)
        checksum_mock.assert_called_with('sample.py')
        self.assertEqual(actual, function_mock.return_value.insert_entry.return_value)

    @mock.patch('evadb.catalog.catalog_manager.FunctionCatalogService')
    def test_get_function_catalog_entry_by_name(self, function_mock):
        catalog = CatalogManager(MagicMock())
        actual = catalog.get_function_catalog_entry_by_name('name')
        function_mock.return_value.get_entry_by_name.assert_called_with('name')
        self.assertEqual(actual, function_mock.return_value.get_entry_by_name.return_value)

    @mock.patch('evadb.catalog.catalog_manager.FunctionCatalogService')
    def test_delete_function(self, function_mock):
        CatalogManager(MagicMock()).delete_function_catalog_entry_by_name('name')
        function_mock.return_value.delete_entry_by_name.assert_called_with('name')

    @mock.patch('evadb.catalog.catalog_manager.FunctionIOCatalogService')
    def test_get_function_outputs(self, function_mock):
        mock_func = function_mock.return_value.get_output_entries_by_function_id
        function_obj = MagicMock(spec=FunctionCatalogEntry)
        CatalogManager(MagicMock()).get_function_io_catalog_output_entries(function_obj)
        mock_func.assert_called_once_with(function_obj.row_id)

    @mock.patch('evadb.catalog.catalog_manager.FunctionIOCatalogService')
    def test_get_function_inputs(self, function_mock):
        mock_func = function_mock.return_value.get_input_entries_by_function_id
        function_obj = MagicMock(spec=FunctionCatalogEntry)
        CatalogManager(MagicMock()).get_function_io_catalog_input_entries(function_obj)
        mock_func.assert_called_once_with(function_obj.row_id)

class CatalogModelsTest(unittest.TestCase):

    def test_df_column(self):
        df_col = ColumnCatalogEntry('name', ColumnType.TEXT, is_nullable=False)
        df_col.array_dimensions = [1, 2]
        df_col.table_id = 1
        self.assertEqual(df_col.array_type, None)
        self.assertEqual(df_col.array_dimensions, [1, 2])
        self.assertEqual(df_col.is_nullable, False)
        self.assertEqual(df_col.name, 'name')
        self.assertEqual(df_col.type, ColumnType.TEXT)
        self.assertEqual(df_col.table_id, 1)
        self.assertEqual(df_col.row_id, None)

    def test_df_equality(self):
        df_col = ColumnCatalogEntry('name', ColumnType.TEXT, is_nullable=False)
        self.assertEqual(df_col, df_col)
        df_col1 = ColumnCatalogEntry('name2', ColumnType.TEXT, is_nullable=False)
        self.assertNotEqual(df_col, df_col1)
        df_col1 = ColumnCatalogEntry('name', ColumnType.INTEGER, is_nullable=False)
        self.assertNotEqual(df_col, df_col1)
        df_col1 = ColumnCatalogEntry('name', ColumnType.INTEGER, is_nullable=True)
        self.assertNotEqual(df_col, df_col1)
        df_col1 = ColumnCatalogEntry('name', ColumnType.INTEGER, is_nullable=False)
        self.assertNotEqual(df_col, df_col1)
        df_col._array_dimensions = [2, 4]
        df_col1 = ColumnCatalogEntry('name', ColumnType.INTEGER, is_nullable=False, array_dimensions=[1, 2])
        self.assertNotEqual(df_col, df_col1)
        df_col._table_id = 1
        df_col1 = ColumnCatalogEntry('name', ColumnType.INTEGER, is_nullable=False, array_dimensions=[2, 4], table_id=2)
        self.assertNotEqual(df_col, df_col1)

    def test_table_catalog_entry_equality(self):
        column_1 = ColumnCatalogEntry('frame_id', ColumnType.INTEGER, False)
        column_2 = ColumnCatalogEntry('frame_label', ColumnType.INTEGER, False)
        col_list = [column_1, column_2]
        table_catalog_entry = TableCatalogEntry('name', 'evadb_dataset', table_type=TableType.VIDEO_DATA, columns=col_list)
        self.assertEqual(table_catalog_entry, table_catalog_entry)
        table_catalog_entry1 = TableCatalogEntry('name2', 'evadb_dataset', table_type=TableType.VIDEO_DATA, columns=col_list)
        self.assertNotEqual(table_catalog_entry, table_catalog_entry1)

    def test_function(self):
        function = FunctionCatalogEntry('function', 'fasterRCNN', 'ObjectDetection', 'checksum')
        self.assertEqual(function.row_id, None)
        self.assertEqual(function.impl_file_path, 'fasterRCNN')
        self.assertEqual(function.name, 'function')
        self.assertEqual(function.type, 'ObjectDetection')
        self.assertEqual(function.checksum, 'checksum')

    def test_function_hash(self):
        function1 = FunctionCatalogEntry('function', 'fasterRCNN', 'ObjectDetection', 'checksum')
        function2 = FunctionCatalogEntry('function', 'fasterRCNN', 'ObjectDetection', 'checksum')
        self.assertEqual(hash(function1), hash(function2))

    def test_function_equality(self):
        function = FunctionCatalogEntry('function', 'fasterRCNN', 'ObjectDetection', 'checksum')
        self.assertEqual(function, function)
        function2 = FunctionCatalogEntry('function2', 'fasterRCNN', 'ObjectDetection', 'checksum')
        self.assertNotEqual(function, function2)
        function3 = FunctionCatalogEntry('function', 'fasterRCNN2', 'ObjectDetection', 'checksum')
        self.assertNotEqual(function, function3)
        function4 = FunctionCatalogEntry('function2', 'fasterRCNN', 'ObjectDetection3', 'checksum')
        self.assertNotEqual(function, function4)

    def test_function_io(self):
        function_io = FunctionIOCatalogEntry('name', ColumnType.NDARRAY, True, NdArrayType.UINT8, [2, 3], True, 1)
        self.assertEqual(function_io.row_id, None)
        self.assertEqual(function_io.function_id, 1)
        self.assertEqual(function_io.is_input, True)
        self.assertEqual(function_io.is_nullable, True)
        self.assertEqual(function_io.array_type, NdArrayType.UINT8)
        self.assertEqual(function_io.array_dimensions, [2, 3])
        self.assertEqual(function_io.name, 'name')
        self.assertEqual(function_io.type, ColumnType.NDARRAY)

    def test_function_io_equality(self):
        function_io = FunctionIOCatalogEntry('name', ColumnType.FLOAT, True, None, [2, 3], True, 1)
        self.assertEqual(function_io, function_io)
        function_io2 = FunctionIOCatalogEntry('name2', ColumnType.FLOAT, True, None, [2, 3], True, 1)
        self.assertNotEqual(function_io, function_io2)
        function_io2 = FunctionIOCatalogEntry('name', ColumnType.INTEGER, True, None, [2, 3], True, 1)
        self.assertNotEqual(function_io, function_io2)
        function_io2 = FunctionIOCatalogEntry('name', ColumnType.FLOAT, False, None, [2, 3], True, 1)
        self.assertNotEqual(function_io, function_io2)
        function_io2 = FunctionIOCatalogEntry('name', ColumnType.FLOAT, True, None, [2, 3, 4], True, 1)
        self.assertNotEqual(function_io, function_io2)
        function_io2 = FunctionIOCatalogEntry('name', ColumnType.FLOAT, True, None, [2, 3], False, 1)
        self.assertNotEqual(function_io, function_io2)
        function_io2 = FunctionIOCatalogEntry('name', ColumnType.FLOAT, True, None, [2, 3], True, 2)
        self.assertNotEqual(function_io, function_io2)

    def test_index(self):
        index = IndexCatalogEntry('index', 'FaissSavePath', 'HNSW')
        self.assertEqual(index.row_id, None)
        self.assertEqual(index.name, 'index')
        self.assertEqual(index.save_file_path, 'FaissSavePath')
        self.assertEqual(index.type, 'HNSW')

    def test_index_hash(self):
        index1 = IndexCatalogEntry('index', 'FaissSavePath', 'HNSW')
        index2 = IndexCatalogEntry('index', 'FaissSavePath', 'HNSW')
        self.assertEqual(hash(index1), hash(index2))

    def test_index_equality(self):
        index = IndexCatalogEntry('index', 'FaissSavePath', 'HNSW')
        self.assertEqual(index, index)
        index2 = IndexCatalogEntry('index2', 'FaissSavePath', 'HNSW')
        self.assertNotEqual(index, index2)
        index3 = IndexCatalogEntry('index', 'FaissSavePath3', 'HNSW')
        self.assertNotEqual(index, index3)
        index4 = IndexCatalogEntry('index', 'FaissSavePath', 'HNSW4')
        self.assertNotEqual(index, index4)

@pytest.mark.notparallel
class SQLStorageEngineTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.table = None

    def create_sample_table(self):
        table_info = TableCatalogEntry(str(suffix_pytest_xdist_worker_id_to_dir('dataset')), str(suffix_pytest_xdist_worker_id_to_dir('dataset')), table_type=TableType.VIDEO_DATA)
        column_pk = ColumnCatalogEntry(IDENTIFIER_COLUMN, ColumnType.INTEGER, is_nullable=False)
        column_0 = ColumnCatalogEntry('name', ColumnType.TEXT, is_nullable=False)
        column_1 = ColumnCatalogEntry('id', ColumnType.INTEGER, is_nullable=False)
        column_2 = ColumnCatalogEntry('data', ColumnType.NDARRAY, False, NdArrayType.UINT8, [2, 2, 3])
        table_info.columns = [column_pk, column_0, column_1, column_2]
        return table_info

    def setUp(self):
        self.table = self.create_sample_table()

    def tearDown(self):
        try:
            shutil.rmtree(suffix_pytest_xdist_worker_id_to_dir('dataset'), ignore_errors=True)
        except ValueError:
            pass

    def test_should_create_empty_table(self):
        evadb = get_evadb_for_testing()
        sqlengine = SQLStorageEngine(evadb)
        sqlengine.create(self.table)
        records = list(sqlengine.read(self.table, batch_mem_size=3000))
        self.assertEqual(len(records), 0)
        sqlengine.drop(self.table)

    def test_should_write_rows_to_table(self):
        dummy_batches = list(create_dummy_batches())
        dummy_batches = [batch.project(batch.columns[1:]) for batch in dummy_batches]
        evadb = get_evadb_for_testing()
        sqlengine = SQLStorageEngine(evadb)
        sqlengine.create(self.table)
        for batch in dummy_batches:
            batch.drop_column_alias()
            sqlengine.write(self.table, batch)
        read_batch = list(sqlengine.read(self.table, batch_mem_size=3000))
        self.assertTrue(read_batch, dummy_batches)
        sqlengine.drop(self.table)

    def test_rename(self):
        table_info = TableCatalogEntry('new_name', 'new_name', table_type=TableType.VIDEO_DATA)
        evadb = get_evadb_for_testing()
        sqlengine = SQLStorageEngine(evadb)
        with pytest.raises(Exception):
            sqlengine.rename(self.table, table_info)

    def test_sqlite_storage_engine_exceptions(self):
        evadb = get_evadb_for_testing()
        sqlengine = SQLStorageEngine(evadb)
        missing_table_info = TableCatalogEntry('missing_table', None, table_type=TableType.VIDEO_DATA)
        with self.assertRaises(Exception):
            sqlengine.drop(missing_table_info)
        with self.assertRaises(Exception):
            sqlengine.write(missing_table_info, None)
        with self.assertRaises(Exception):
            read_batch = list(sqlengine.read(missing_table_info))
            self.assertEqual(read_batch, None)
        with self.assertRaises(Exception):
            sqlengine.delete(missing_table_info, None)

    def test_cannot_delete_missing_column(self):
        evadb = get_evadb_for_testing()
        sqlengine = SQLStorageEngine(evadb)
        sqlengine.create(self.table)
        incorrect_where_clause = {'foo': None}
        with self.assertRaises(Exception):
            sqlengine.delete(self.table, incorrect_where_clause)
        sqlengine.drop(self.table)

@pytest.mark.notparallel
class VideoStorageEngineTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.table = None

    def create_sample_table(self):
        table_info = TableCatalogEntry(str(suffix_pytest_xdist_worker_id_to_dir('dataset')), str(suffix_pytest_xdist_worker_id_to_dir('dataset')), table_type=TableType.VIDEO_DATA)
        column_1 = ColumnCatalogEntry('id', ColumnType.INTEGER, False)
        column_2 = ColumnCatalogEntry('data', ColumnType.NDARRAY, False, NdArrayType.UINT8, [2, 2, 3])
        table_info.schema = [column_1, column_2]
        return table_info

    def setUp(self):
        evadb = get_evadb_for_testing()
        mock.table_type = TableType.VIDEO_DATA
        self.video_engine = StorageEngine.factory(evadb, mock)
        self.table = self.create_sample_table()

    def tearDown(self):
        pass

    @mock.patch('pathlib.Path.mkdir')
    def test_should_raise_file_exist_error(self, m):
        m.side_effect = FileExistsError
        with self.assertRaises(FileExistsError):
            self.video_engine.create(self.table, if_not_exists=False)
        self.video_engine.create(self.table, if_not_exists=True)

    def test_write(self):
        batch = MagicMock()
        batch.frames = []
        table = MagicMock()
        table.file_url = Exception()
        with self.assertRaises(Exception):
            self.video_engine.write(table, batch)

    def test_delete(self):
        batch = MagicMock()
        batch.frames = []
        table = MagicMock()
        table.file_url = Exception()
        with self.assertRaises(Exception):
            self.video_engine.delete(table, batch)
        self.video_engine.create(self.table, if_not_exists=True)
        video_file_path = create_sample_video()
        df = pd.DataFrame([video_file_path], columns=['file_path'])
        batch = Batch(df)
        with self.assertRaises(Exception):
            self.video_engine.delete(self.table, batch)
        self.video_engine.drop(self.table)

    def test_rename(self):
        table_info = TableCatalogEntry('new_name', 'new_name', table_type=TableType.VIDEO_DATA)
        with pytest.raises(Exception):
            self.video_engine.rename(self.table, table_info)

    def test_media_storage_engine_exceptions(self):
        missing_table_info = TableCatalogEntry('missing_table', None, table_type=TableType.VIDEO_DATA)
        with self.assertRaises(Exception):
            self.video_engine.drop(missing_table_info)
        with self.assertRaises(Exception):
            self.video_engine.write(missing_table_info, None)
        with self.assertRaises(Exception):
            read_batch = list(self.video_engine.read(missing_table_info))
            self.assertEqual(read_batch, None)
        with self.assertRaises(Exception):
            self.video_engine.delete(missing_table_info, None)

class StatementToPlanConverter:

    def __init__(self):
        self._plan = None

    def visit_table_ref(self, table_ref: TableRef):
        """Bind table ref object and convert to LogicalGet, LogicalJoin,
            LogicalFunctionScan, or LogicalQueryDerivedGet

        Arguments:
            table {TableRef} - - [Input table ref object created by the parser]
        """
        if table_ref.is_table_atom():
            catalog_entry = table_ref.table.table_obj
            self._plan = LogicalGet(table_ref, catalog_entry, table_ref.alias, chunk_params=table_ref.chunk_params)
        elif table_ref.is_table_valued_expr():
            tve = table_ref.table_valued_expr
            if tve.func_expr.name.lower() == str(FunctionType.EXTRACT_OBJECT).lower():
                self._plan = LogicalExtractObject(detector=tve.func_expr.children[1], tracker=tve.func_expr.children[2], alias=table_ref.alias, do_unnest=tve.do_unnest)
            else:
                self._plan = LogicalFunctionScan(func_expr=tve.func_expr, alias=table_ref.alias, do_unnest=tve.do_unnest)
        elif table_ref.is_select():
            self.visit_select(table_ref.select_statement)
            child_plan = self._plan
            self._plan = LogicalQueryDerivedGet(table_ref.alias)
            self._plan.append_child(child_plan)
        elif table_ref.is_join():
            join_node = table_ref.join_node
            join_plan = LogicalJoin(join_type=join_node.join_type, join_predicate=join_node.predicate)
            self.visit_table_ref(join_node.left)
            join_plan.append_child(self._plan)
            self.visit_table_ref(join_node.right)
            join_plan.append_child(self._plan)
            self._plan = join_plan
        if table_ref.sample_freq:
            self._visit_sample(table_ref.sample_freq, table_ref.sample_type)

    def visit_select(self, statement: SelectStatement):
        """converter for select statement

        Arguments:
            statement {SelectStatement} - - [input select statement]
        """
        col_with_func_exprs = []
        if statement.orderby_list and statement.groupby_clause is None:
            projection_cols = []
            for col in statement.target_list:
                if isinstance(col, FunctionExpression):
                    col_with_func_exprs.append(col)
                    projection_cols.extend(get_bound_func_expr_outputs_as_tuple_value_expr(col))
                else:
                    projection_cols.append(col)
            statement.target_list = projection_cols
        table_ref = statement.from_table
        if not table_ref and col_with_func_exprs:
            self._visit_projection(col_with_func_exprs)
        else:
            for col in col_with_func_exprs:
                tve = TableValuedExpression(col)
                if table_ref:
                    table_ref = TableRef(JoinNode(table_ref, TableRef(tve, alias=col.alias), join_type=JoinType.LATERAL_JOIN))
            statement.from_table = table_ref
        if table_ref is not None:
            self.visit_table_ref(table_ref)
            predicate = statement.where_clause
            if predicate is not None:
                self._visit_select_predicate(predicate)
            if statement.groupby_clause is not None:
                self._visit_groupby(statement.groupby_clause)
        if statement.orderby_list is not None:
            self._visit_orderby(statement.orderby_list)
        if statement.limit_count is not None:
            self._visit_limit(statement.limit_count)
        if statement.target_list is not None:
            self._visit_projection(statement.target_list)
        if statement.union_link is not None:
            self._visit_union(statement.union_link, statement.union_all)

    def _visit_sample(self, sample_freq, sample_type):
        sample_opr = LogicalSample(sample_freq, sample_type)
        sample_opr.append_child(self._plan)
        self._plan = sample_opr

    def _visit_groupby(self, groupby_clause):
        groupby_opr = LogicalGroupBy(groupby_clause)
        groupby_opr.append_child(self._plan)
        self._plan = groupby_opr

    def _visit_orderby(self, orderby_list):
        orderby_opr = LogicalOrderBy(orderby_list)
        orderby_opr.append_child(self._plan)
        self._plan = orderby_opr

    def _visit_limit(self, limit_count):
        limit_opr = LogicalLimit(limit_count)
        limit_opr.append_child(self._plan)
        self._plan = limit_opr

    def _visit_union(self, target, all):
        left_child_plan = self._plan
        self.visit_select(target)
        right_child_plan = self._plan
        self._plan = LogicalUnion(all=all)
        self._plan.append_child(left_child_plan)
        self._plan.append_child(right_child_plan)

    def _visit_projection(self, select_columns):
        projection_opr = LogicalProject(select_columns)
        if self._plan is not None:
            projection_opr.append_child(self._plan)
        self._plan = projection_opr

    def _visit_select_predicate(self, predicate: AbstractExpression):
        filter_opr = LogicalFilter(predicate)
        filter_opr.append_child(self._plan)
        self._plan = filter_opr

    def visit_insert(self, statement: AbstractStatement):
        """Converter for parsed insert statement

        Arguments:
            statement {AbstractStatement} - - [input insert statement]
        """
        insert_data_opr = LogicalInsert(statement.table_ref, statement.column_list, statement.value_list)
        self._plan = insert_data_opr
        '\n        table_ref = statement.table\n        table_metainfo = bind_dataset(table_ref.table)\n        if table_metainfo is None:\n            # Create a new metadata object\n            table_metainfo = create_video_metadata(table_ref.table.table_name)\n\n        # populate self._column_map\n        self._populate_column_map(table_metainfo)\n\n        # Bind column_list\n        bind_columns_expr(statement.column_list, self._column_map)\n\n        # Nothing to be done for values as we add support for other variants of\n        # insert we will handle them\n        value_list = statement.value_list\n\n        # Ready to create Logical node\n        insert_opr = LogicalInsert(\n            table_ref, table_metainfo, statement.column_list, value_list)\n        self._plan = insert_opr\n        '

    def visit_create(self, statement: AbstractStatement):
        """Converter for parsed insert Statement

        Arguments:
            statement {AbstractStatement} - - [Create statement]
        """
        table_info = statement.table_info
        if table_info is None:
            logger.error('Missing Table Name In Create Statement')
        create_opr = LogicalCreate(table_info, statement.column_list, statement.if_not_exists)
        if statement.query is not None:
            self.visit_select(statement.query)
            create_opr.append_child(self._plan)
        self._plan = create_opr

    def visit_rename(self, statement: RenameTableStatement):
        """Converter for parsed rename statement
        Arguments:
            statement(RenameTableStatement): [Rename statement]
        """
        rename_opr = LogicalRename(statement.old_table_ref, statement.new_table_name)
        self._plan = rename_opr

    def visit_create_function(self, statement: CreateFunctionStatement):
        """Converter for parsed create function statement

        Arguments:
            statement {CreateFunctionStatement} - - CreateFunctionStatement
        """
        annotated_inputs = column_definition_to_function_io(statement.inputs, True)
        annotated_outputs = column_definition_to_function_io(statement.outputs, False)
        annotated_metadata = metadata_definition_to_function_metadata(statement.metadata)
        create_function_opr = LogicalCreateFunction(statement.name, statement.or_replace, statement.if_not_exists, annotated_inputs, annotated_outputs, statement.impl_path, statement.function_type, annotated_metadata)
        if statement.query is not None:
            self.visit_select(statement.query)
            create_function_opr.append_child(self._plan)
        self._plan = create_function_opr

    def visit_drop_object(self, statement: DropObjectStatement):
        self._plan = LogicalDropObject(statement.object_type, statement.name, statement.if_exists)

    def visit_load_data(self, statement: LoadDataStatement):
        """Converter for parsed load data statement
        Arguments:
            statement(LoadDataStatement): [Load data statement]
        """
        load_data_opr = LogicalLoadData(statement.table_info, statement.path, statement.column_list, statement.file_options)
        self._plan = load_data_opr

    def visit_show(self, statement: ShowStatement):
        show_opr = LogicalShow(statement.show_type, statement.show_val)
        self._plan = show_opr

    def visit_explain(self, statement: ExplainStatement):
        explain_opr = LogicalExplain([self.visit(statement.explainable_stmt)])
        self._plan = explain_opr

    def visit_create_index(self, statement: CreateIndexStatement):
        create_index_opr = LogicalCreateIndex(statement.name, statement.if_not_exists, statement.table_ref, statement.col_list, statement.vector_store_type, statement.project_expr_list, statement.index_def)
        self._plan = create_index_opr

    def visit_delete(self, statement: DeleteTableStatement):
        delete_opr = LogicalDelete(statement.table_ref, statement.where_clause)
        self._plan = delete_opr

    def visit(self, statement: AbstractStatement):
        """Based on the instance of the statement the corresponding
           visit is called.
           The logic is hidden from client.

        Arguments:
            statement {AbstractStatement} - - [Input statement]
        """
        if isinstance(statement, SelectStatement):
            self.visit_select(statement)
        elif isinstance(statement, InsertTableStatement):
            self.visit_insert(statement)
        elif isinstance(statement, CreateTableStatement):
            self.visit_create(statement)
        elif isinstance(statement, RenameTableStatement):
            self.visit_rename(statement)
        elif isinstance(statement, CreateFunctionStatement):
            self.visit_create_function(statement)
        elif isinstance(statement, DropObjectStatement):
            self.visit_drop_object(statement)
        elif isinstance(statement, LoadDataStatement):
            self.visit_load_data(statement)
        elif isinstance(statement, ShowStatement):
            self.visit_show(statement)
        elif isinstance(statement, ExplainStatement):
            self.visit_explain(statement)
        elif isinstance(statement, CreateIndexStatement):
            self.visit_create_index(statement)
        elif isinstance(statement, DeleteTableStatement):
            self.visit_delete(statement)
        return self._plan

    @property
    def plan(self):
        return self._plan

class StatementBinder:

    def __init__(self, binder_context: StatementBinderContext):
        self._binder_context = binder_context
        self._catalog: Callable = binder_context._catalog

    @singledispatchmethod
    def bind(self, node):
        raise NotImplementedError(f'Cannot bind {type(node)}')

    @bind.register(AbstractStatement)
    def _bind_abstract_statement(self, node: AbstractStatement):
        pass

    @bind.register(AbstractExpression)
    def _bind_abstract_expr(self, node: AbstractExpression):
        for child in node.children:
            self.bind(child)

    @bind.register(ExplainStatement)
    def _bind_explain_statement(self, node: ExplainStatement):
        self.bind(node.explainable_stmt)

    @bind.register(CreateFunctionStatement)
    def _bind_create_function_statement(self, node: CreateFunctionStatement):
        if node.query is not None:
            self.bind(node.query)
            node.query.target_list = drop_row_id_from_target_list(node.query.target_list)
            all_column_list = get_column_definition_from_select_target_list(node.query.target_list)
            arg_map = {key: value for key, value in node.metadata}
            inputs, outputs = ([], [])
            if string_comparison_case_insensitive(node.function_type, 'ludwig'):
                assert 'predict' in arg_map, f"Creating {node.function_type} functions expects 'predict' metadata."
                predict_columns = set([arg_map['predict']])
                for column in all_column_list:
                    if column.name in predict_columns:
                        column.name = column.name + '_predictions'
                        outputs.append(column)
                    else:
                        inputs.append(column)
            elif string_comparison_case_insensitive(node.function_type, 'sklearn') or string_comparison_case_insensitive(node.function_type, 'XGBoost'):
                assert 'predict' in arg_map, f"Creating {node.function_type} functions expects 'predict' metadata."
                predict_columns = set([arg_map['predict']])
                for column in all_column_list:
                    if column.name in predict_columns:
                        outputs.append(column)
                    else:
                        inputs.append(column)
            elif string_comparison_case_insensitive(node.function_type, 'forecasting'):
                inputs = [ColumnDefinition('horizon', ColumnType.INTEGER, None, None)]
                required_columns = set([arg_map.get('predict', 'y')])
                for column in all_column_list:
                    if column.name == arg_map.get('id', 'unique_id'):
                        outputs.append(column)
                    elif column.name == arg_map.get('time', 'ds'):
                        outputs.append(column)
                    elif column.name == arg_map.get('predict', 'y'):
                        outputs.append(column)
                        required_columns.remove(column.name)
                assert len(required_columns) == 0, f'Missing required {required_columns} columns for forecasting function.'
                outputs.extend([ColumnDefinition(arg_map.get('predict', 'y') + '-lo', ColumnType.FLOAT, None, None), ColumnDefinition(arg_map.get('predict', 'y') + '-hi', ColumnType.FLOAT, None, None)])
            else:
                raise BinderError(f'Unsupported type of function: {node.function_type}.')
            assert len(node.inputs) == 0 and len(node.outputs) == 0, f"{node.function_type} functions' input and output are auto assigned"
            node.inputs, node.outputs = (inputs, outputs)

    @bind.register(SelectStatement)
    def _bind_select_statement(self, node: SelectStatement):
        if node.from_table:
            self.bind(node.from_table)
        if node.where_clause:
            self.bind(node.where_clause)
            if node.where_clause.etype == ExpressionType.COMPARE_LIKE:
                check_column_name_is_string(node.where_clause.children[0])
        if node.target_list:
            if len(node.target_list) == 1 and isinstance(node.target_list[0], TupleValueExpression) and (node.target_list[0].name == '*'):
                node.target_list = extend_star(self._binder_context)
            for expr in node.target_list:
                self.bind(expr)
                if isinstance(expr, FunctionExpression):
                    output_cols = get_bound_func_expr_outputs_as_tuple_value_expr(expr)
                    self._binder_context.add_derived_table_alias(expr.alias.alias_name, output_cols)
        if node.groupby_clause:
            self.bind(node.groupby_clause)
            check_table_object_is_groupable(node.from_table)
            check_groupby_pattern(node.from_table, node.groupby_clause.value)
        if node.orderby_list:
            for expr in node.orderby_list:
                self.bind(expr[0])
        if node.union_link:
            current_context = self._binder_context
            self._binder_context = StatementBinderContext(self._catalog)
            self.bind(node.union_link)
            self._binder_context = current_context
        if node.from_table and node.from_table.chunk_params:
            assert is_document_table(node.from_table.table.table_obj), 'CHUNK related parameters only supported for DOCUMENT tables.'
        assert not (self._binder_context.is_retrieve_audio() and self._binder_context.is_retrieve_video()), 'Cannot query over both audio and video streams'
        if self._binder_context.is_retrieve_audio():
            node.from_table.get_audio = True
        if self._binder_context.is_retrieve_video():
            node.from_table.get_video = True

    @bind.register(DeleteTableStatement)
    def _bind_delete_statement(self, node: DeleteTableStatement):
        self.bind(node.table_ref)
        if node.where_clause:
            self.bind(node.where_clause)

    @bind.register(CreateTableStatement)
    def _bind_create_statement(self, node: CreateTableStatement):
        for col in node.column_list:
            assert col.name.lower() not in RESTRICTED_COL_NAMES, f'EvaDB does not allow to create a table with column name {col.name}'
        if node.query is not None:
            self.bind(node.query)
            node.column_list = get_column_definition_from_select_target_list(node.query.target_list)

    @bind.register(CreateIndexStatement)
    def _bind_create_index_statement(self, node: CreateIndexStatement):
        from evadb.binder.create_index_statement_binder import bind_create_index
        bind_create_index(self, node)

    @bind.register(RenameTableStatement)
    def _bind_rename_table_statement(self, node: RenameTableStatement):
        self.bind(node.old_table_ref)
        assert node.old_table_ref.table.table_obj.table_type != TableType.STRUCTURED_DATA, 'Rename not yet supported on structured data'

    @bind.register(TableRef)
    def _bind_tableref(self, node: TableRef):
        if node.is_table_atom():
            self._binder_context.add_table_alias(node.alias.alias_name, node.table.database_name, node.table.table_name)
            bind_table_info(self._catalog(), node.table)
        elif node.is_select():
            current_context = self._binder_context
            self._binder_context = StatementBinderContext(self._catalog)
            self.bind(node.select_statement)
            self._binder_context = current_context
            self._binder_context.add_derived_table_alias(node.alias.alias_name, node.select_statement.target_list)
        elif node.is_join():
            self.bind(node.join_node.left)
            self.bind(node.join_node.right)
            if node.join_node.predicate:
                self.bind(node.join_node.predicate)
        elif node.is_table_valued_expr():
            func_expr = node.table_valued_expr.func_expr
            func_expr.alias = node.alias
            self.bind(func_expr)
            output_cols = get_bound_func_expr_outputs_as_tuple_value_expr(func_expr)
            self._binder_context.add_derived_table_alias(func_expr.alias.alias_name, output_cols)
        else:
            raise BinderError(f'Unsupported node {type(node)}')

    @bind.register(TupleValueExpression)
    def _bind_tuple_expr(self, node: TupleValueExpression):
        from evadb.binder.tuple_value_expression_binder import bind_tuple_expr
        bind_tuple_expr(self, node)

    @bind.register(FunctionExpression)
    def _bind_func_expr(self, node: FunctionExpression):
        from evadb.binder.function_expression_binder import bind_func_expr
        bind_func_expr(self, node)

def bind_create_index(binder: StatementBinder, node: CreateIndexStatement):
    binder.bind(node.table_ref)
    func_project_expr = None
    for project_expr in node.project_expr_list:
        binder.bind(project_expr)
        if isinstance(project_expr, FunctionExpression):
            func_project_expr = project_expr
    node.project_expr_list += [create_row_num_tv_expr(node.table_ref.alias)]
    assert len(node.col_list) == 1, 'Index cannot be created on more than 1 column'
    assert node.table_ref.is_table_atom(), 'Index can only be created on an existing table'
    catalog = binder._catalog()
    if node.vector_store_type == VectorStoreType.PGVECTOR:
        db_catalog_entry = catalog.get_database_catalog_entry(node.table_ref.table.database_name)
        if db_catalog_entry.engine != 'postgres':
            raise BinderError('PGVECTOR index works only with Postgres data source.')
        with get_database_handler(db_catalog_entry.engine, **db_catalog_entry.params) as handler:
            df = handler.execute_native_query("SELECT * FROM pg_extension WHERE extname = 'vector'").data
            if len(df) == 0:
                raise BinderError('PGVECTOR extension is not enabled.')
        return
    assert len(node.col_list) == 1, f'Index can be only created on one column, but instead {len(node.col_list)} are provided'
    col_def = node.col_list[0]
    if func_project_expr is None:
        table_ref_obj = node.table_ref.table.table_obj
        col_list = [col for col in table_ref_obj.columns if col.name == col_def.name]
        assert len(col_list) == 1, f'Index is created on non-existent column {col_def.name}'
        col = col_list[0]
        assert len(col.array_dimensions) == 2
        if node.vector_store_type == VectorStoreType.FAISS:
            assert col.array_type == NdArrayType.FLOAT32, 'Index input needs to be float32.'
    else:
        function_obj = binder._catalog().get_function_catalog_entry_by_name(func_project_expr.name)
        for output in function_obj.outputs:
            assert len(output.array_dimensions) == 2, 'Index input needs to be 2 dimensional.'
            if node.vector_store_type == VectorStoreType.FAISS:
                assert output.array_type == NdArrayType.FLOAT32, 'Index input needs to be float32.'

class StatementBinderContext:
    """
    This context is used to store information that is useful during the process of binding a statement (such as a SELECT statement) to the catalog. It stores the following information:

    Args:
        `_table_alias_map`: Maintains a mapping from table_alias to corresponding
        catalog table entry
        `_derived_table_alias_map`: Maintains a mapping from derived table aliases,
        such as subqueries or function expressions, to column alias maps for all the
        corresponding projected columns. For example, in the following queries, the
        `_derived_table_alias_map` attribute would contain:

        `Select * FROM (SELECT id1, id2 FROM table) AS A` :
            `{A: {id1: table.col1, id2: table.col2}}`
        `Select * FROM video LATERAL JOIN func AS T(a, b)` :
            `{T: {a: func.obj1, b:func.obj2}}`
    """

    def __init__(self, catalog: Callable):
        self._catalog = catalog
        self._table_alias_map: Dict[str, TableCatalogEntry] = dict()
        self._derived_table_alias_map: Dict[str, Dict[str, CatalogColumnType]] = dict()
        self._retrieve_audio = False
        self._retrieve_video = False

    def _check_duplicate_alias(self, alias: str):
        """
        Sanity check: no duplicate alias in table and derived_table
        Arguments:
            alias (str): name of the alias

        Exception:
            Raise exception if found duplication
        """
        if alias in self._derived_table_alias_map or alias in self._table_alias_map:
            err_msg = f'Found duplicate alias {alias}'
            logger.error(err_msg)
            raise BinderError(err_msg)

    def add_table_alias(self, alias: str, database_name: str, table_name: str):
        """
        Add a alias -> table_name mapping
        Arguments:
            alias (str): name of alias
            table_name (str): name of the table
        """
        self._check_duplicate_alias(alias)
        if database_name is not None:
            check_data_source_and_table_are_valid(self._catalog(), database_name, table_name)
            db_catalog_entry = self._catalog().get_database_catalog_entry(database_name)
            with get_database_handler(db_catalog_entry.engine, **db_catalog_entry.params) as handler:
                response = handler.get_columns(table_name)
                if response.error is not None:
                    raise BinderError(response.error)
                column_df = response.data
                table_obj = create_table_catalog_entry_for_data_source(table_name, database_name, column_df)
        else:
            table_obj = self._catalog().get_table_catalog_entry(table_name)
        self._table_alias_map[alias] = table_obj

    def add_derived_table_alias(self, alias: str, target_list: List[Union[TupleValueExpression, FunctionExpression, FunctionIOCatalogEntry]]):
        """
        Add a alias -> derived table column mapping
        Arguments:
            alias (str): name of alias
            target_list: list of TupleValueExpression or FunctionExpression or FunctionIOCatalogEntry
        """
        self._check_duplicate_alias(alias)
        col_alias_map = {}
        for expr in target_list:
            if isinstance(expr, FunctionExpression):
                for obj in expr.output_objs:
                    col_alias_map[obj.name] = obj
            elif isinstance(expr, TupleValueExpression):
                col_alias_map[expr.name] = expr.col_object
            else:
                continue
        self._derived_table_alias_map[alias] = col_alias_map

    def get_binded_column(self, col_name: str, alias: str=None) -> Tuple[str, CatalogColumnType]:
        """
        Find the binded column object
        Arguments:
            col_name (str): column name
            alias (str): alias name

        Returns:
            A tuple of alias and column object
        """
        col_name = col_name.lower()

        def raise_error():
            all_columns = sorted(list(set([col for _, col in self._get_all_alias_and_col_name()])))
            res = process.extractOne(col_name, all_columns)
            if res is not None:
                guess_column, _ = res
                err_msg = f'Cannot find column {col_name}. Did you mean {guess_column}? The feasible columns are {all_columns}.'
            else:
                err_msg = f'Cannot find column {col_name}. There are no feasible columns.'
            logger.error(err_msg)
            raise BinderError(err_msg)
        if not alias:
            alias, col_obj = self._search_all_alias_maps(col_name)
        else:
            col_obj = self._check_table_alias_map(alias, col_name)
            if not col_obj:
                col_obj = self._check_derived_table_alias_map(alias, col_name)
        if col_obj:
            return (alias, col_obj)
        raise_error()

    def _check_table_alias_map(self, alias, col_name) -> ColumnCatalogEntry:
        """
        Find the column object in table alias map
        Arguments:
            col_name (str): column name
            alias (str): alias name

        Returns:
            column object
        """
        table_obj = self._table_alias_map.get(alias, None)
        if table_obj is not None:
            if table_obj.table_type == TableType.NATIVE_DATA:
                for column_catalog_entry in table_obj.columns:
                    if column_catalog_entry.name == col_name:
                        return column_catalog_entry
            else:
                return self._catalog().get_column_catalog_entry(table_obj, col_name)

    def _check_derived_table_alias_map(self, alias, col_name) -> CatalogColumnType:
        """
        Find the column object in derived table alias map
        Arguments:
            col_name (str): column name
            alias (str): alias name

        Returns:
            column object
        """
        col_objs_map = self._derived_table_alias_map.get(alias, None)
        if col_objs_map is None:
            return None
        for name, obj in col_objs_map.items():
            if name == col_name:
                return obj

    def _get_all_alias_and_col_name(self) -> List[Tuple[str, str]]:
        """
        Return all alias and column objects mapping in the current context
        Returns:
            a list of tuple of alias name, column name
        """
        alias_cols = []
        for alias, table_obj in self._table_alias_map.items():
            alias_cols += list([(alias, col.name) for col in table_obj.columns])
        for alias, col_objs_map in self._derived_table_alias_map.items():
            alias_cols += list([(alias, col_name) for col_name in col_objs_map])
        return alias_cols

    def _search_all_alias_maps(self, col_name: str) -> Tuple[str, CatalogColumnType]:
        """
        Search the alias and column object using column name
        Arguments:
            col_name (str): column name

        Returns:
            A tuple of alias and column object.
        """
        num_alias_matches = 0
        alias_match = None
        match_obj = None
        for alias in self._table_alias_map:
            col_obj = self._check_table_alias_map(alias, col_name)
            if col_obj:
                match_obj = col_obj
                num_alias_matches += 1
                alias_match = alias
        for alias in self._derived_table_alias_map:
            col_obj = self._check_derived_table_alias_map(alias, col_name)
            if col_obj:
                match_obj = col_obj
                num_alias_matches += 1
                alias_match = alias
        if num_alias_matches > 1:
            err_msg = f'Ambiguous Column name {col_name}'
            logger.error(err_msg)
            raise BinderError(err_msg)
        return (alias_match, match_obj)

    def enable_audio_retrieval(self):
        self._retrieve_audio = True

    def is_retrieve_audio(self):
        return self._retrieve_audio

    def enable_video_retrieval(self):
        self._retrieve_video = True

    def is_retrieve_video(self):
        return self._retrieve_video

def check_data_source_and_table_are_valid(catalog: CatalogManager, database_name: str, table_name: str):
    """
    Validate the database is valid and the requested table in database is
    also valid.
    """
    error = None
    if catalog.get_database_catalog_entry(database_name) is None:
        error = '{} data source does not exist. Create the new database source using CREATE DATABASE.'.format(database_name)
    if not catalog.check_table_exists(table_name, database_name):
        error = 'Table {} does not exist in data source {}. Create the table using native query.'.format(table_name, database_name)
    if error:
        logger.error(error)
        raise BinderError(error)

def create_table_catalog_entry_for_data_source(table_name: str, database_name: str, column_info: pd.DataFrame):
    column_name_list = list(column_info['name'])
    column_type_list = [ColumnType.python_type_to_evadb_type(dtype) for dtype in list(column_info['dtype'])]
    column_list = []
    for name, dtype in zip(column_name_list, column_type_list):
        column_list.append(ColumnCatalogEntry(name.lower(), dtype))
    table_catalog_entry = TableCatalogEntry(name=table_name, file_url=None, table_type=TableType.NATIVE_DATA, columns=column_list, database_name=database_name)
    return table_catalog_entry

def bind_native_table_info(catalog: CatalogManager, table_info: TableInfo):
    check_data_source_and_table_are_valid(catalog, table_info.database_name, table_info.table_name)
    db_catalog_entry = catalog.get_database_catalog_entry(table_info.database_name)
    with get_database_handler(db_catalog_entry.engine, **db_catalog_entry.params) as handler:
        column_df = handler.get_columns(table_info.table_name).data
        table_info.table_obj = create_table_catalog_entry_for_data_source(table_info.table_name, table_info.database_name, column_df)

def create_row_num_tv_expr(table_alias):
    tv_expr = TupleValueExpression(name=ROW_NUM_COLUMN)
    tv_expr.table_alias = table_alias
    tv_expr.col_alias = f'{table_alias}.{ROW_NUM_COLUMN.lower()}'
    tv_expr.col_object = ColumnCatalogEntry(name=ROW_NUM_COLUMN, type=ColumnType.INTEGER)
    return tv_expr

