# Cluster 11

def get_logical_query_plan(db, query: str) -> Operator:
    stmt = Parser().parse(query)[0]
    StatementBinder(StatementBinderContext(db.catalog)).bind(stmt)
    l_plan = StatementToPlanConverter().visit(stmt)
    return l_plan

def get_physical_query_plan(db, query: str, rule_manager=None, cost_model=None) -> AbstractPlan:
    l_plan = get_logical_query_plan(db, query)
    p_plan = PlanGenerator(db, rule_manager, cost_model).build(l_plan)
    return p_plan

def remove_function_cache(db, query):
    plan = next(get_logical_query_plan(db, query).find_all(LogicalFilter))
    func_exprs = plan.predicate.find_all(FunctionExpression)
    for expr in func_exprs:
        cache_name = expr.signature()
        function_cache = db.catalog.get_function_cache_catalog_entry_by_name(cache_name)
        if function_cache is not None:
            cache_dir = Path(function_cache.cache_path)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

@pytest.mark.notparallel
@gpu_skip_marker
class OptimizerRulesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        ua_detrac = f'{EvaDB_ROOT_DIR}/data/ua_detrac/ua_detrac.mp4'
        execute_query_fetch_all(cls.evadb, f"LOAD VIDEO '{ua_detrac}' INTO MyVideo;")
        execute_query_fetch_all(cls.evadb, f"LOAD VIDEO '{ua_detrac}' INTO MyVideo2;")
        load_functions_for_testing(cls.evadb, mode='debug')

    @classmethod
    def tearDownClass(cls):
        shutdown_ray()
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS MyVideo;')

    @patch('evadb.expression.function_expression.FunctionExpression.evaluate')
    @patch('evadb.models.storage.batch.Batch.merge_column_wise')
    def test_should_benefit_from_pushdown(self, merge_mock, evaluate_mock):
        evaluate_mock.return_value = Batch(pd.DataFrame({'obj.labels': ['car'], 'obj.bboxes': [np.array([1, 2, 3, 4])], 'obj.scores': [0.8]}))
        query = 'SELECT id, obj.labels\n                  FROM MyVideo JOIN LATERAL\n                    FastRCNNObjectDetector(data) AS obj(labels, bboxes, scores)\n                  WHERE id < 2;'
        time_with_rule = Timer()
        result_with_rule = None
        with time_with_rule:
            result_with_rule = execute_query_fetch_all(self.evadb, query)
        evaluate_count_with_rule = evaluate_mock.call_count
        time_without_rule = Timer()
        result_without_pushdown_rules = None
        with time_without_rule:
            rules_manager = RulesManager()
            with disable_rules(rules_manager, [PushDownFilterThroughApplyAndMerge(), PushDownFilterThroughJoin()]):
                custom_plan_generator = PlanGenerator(self.evadb, rules_manager)
                result_without_pushdown_rules = execute_query_fetch_all(self.evadb, query, plan_generator=custom_plan_generator)
        self.assertEqual(result_without_pushdown_rules, result_with_rule)
        evaluate_count_without_rule = evaluate_mock.call_count - evaluate_count_with_rule
        self.assertGreater(evaluate_count_without_rule, 3 * evaluate_count_with_rule)

    def test_should_pushdown_without_pushdown_join_rule(self):
        query = 'SELECT id, obj.labels\n                    FROM MyVideo JOIN LATERAL\n                    FastRCNNObjectDetector(data) AS obj(labels, bboxes, scores)\n                    WHERE id < 2;'
        time_with_rule = Timer()
        result_with_rule = None
        with time_with_rule:
            result_with_rule = execute_query_fetch_all(self.evadb, query)
            query_plan = execute_query_fetch_all(self.evadb, f'EXPLAIN {query}')
        time_without_rule = Timer()
        result_without_pushdown_join_rule = None
        with time_without_rule:
            rules_manager = RulesManager()
            with disable_rules(rules_manager, [PushDownFilterThroughJoin()]):
                custom_plan_generator = PlanGenerator(self.evadb, rules_manager)
                result_without_pushdown_join_rule = execute_query_fetch_all(self.evadb, query, plan_generator=custom_plan_generator)
                query_plan_without_pushdown_join_rule = execute_query_fetch_all(self.evadb, f'EXPLAIN {query}', plan_generator=custom_plan_generator)
        self.assertEqual(result_without_pushdown_join_rule, result_with_rule)
        self.assertEqual(query_plan, query_plan_without_pushdown_join_rule)

    @patch('evadb.catalog.catalog_manager.CatalogManager.get_function_cost_catalog_entry')
    def test_should_reorder_predicates(self, mock):

        def _check_reorder(cost_func):
            mock.side_effect = cost_func
            pred_1 = "DummyObjectDetector(data).label = ['person']"
            pred_2 = "DummyMultiObjectDetector(data).labels @> ['person']"
            query = f'SELECT id FROM MyVideo WHERE {pred_2} AND {pred_1};'
            plan = get_physical_query_plan(self.evadb, query)
            predicate_plans = list(plan.find_all(PredicatePlan))
            self.assertEqual(len(predicate_plans), 1)
            left: ComparisonExpression = predicate_plans[0].predicate.children[0]
            right: ComparisonExpression = predicate_plans[0].predicate.children[1]
            self.assertEqual(left.children[0].name, 'DummyObjectDetector')
            self.assertEqual(right.children[0].name, 'DummyMultiObjectDetector')
        _check_reorder(lambda name: MagicMock(cost=10) if name == 'DummyMultiObjectDetector' else MagicMock(cost=5))
        _check_reorder(lambda name: MagicMock(cost=5) if name == 'DummyObjectDetector' else None)

    @patch('evadb.catalog.catalog_manager.CatalogManager.get_function_cost_catalog_entry')
    def test_should_not_reorder_predicates(self, mock):

        def _check_no_reorder(cost_func):
            mock.side_effect = cost_func
            cheap_pred = "DummyObjectDetector(data).label = ['person']"
            costly_pred = "DummyMultiObjectDetector(data).labels @> ['person']"
            query = f'SELECT id FROM MyVideo WHERE {cheap_pred} AND {costly_pred};'
            plan = get_physical_query_plan(self.evadb, query)
            predicate_plans = list(plan.find_all(PredicatePlan))
            self.assertEqual(len(predicate_plans), 1)
            left: ComparisonExpression = predicate_plans[0].predicate.children[0]
            right: ComparisonExpression = predicate_plans[0].predicate.children[1]
            self.assertEqual(left.children[0].name, 'DummyObjectDetector')
            self.assertEqual(right.children[0].name, 'DummyMultiObjectDetector')
        _check_no_reorder(lambda name: MagicMock(cost=10) if name == 'DummyMultiObjectDetector' else MagicMock(cost=5))
        _check_no_reorder(lambda name: MagicMock(cost=5) if name == 'DummyMultiObjectDetector' else MagicMock(cost=5))
        _check_no_reorder(lambda name: MagicMock(cost=5) if name == 'DummyObjectDetector' else None)
        _check_no_reorder(lambda name: None)

    @patch('evadb.catalog.catalog_manager.CatalogManager.get_function_cost_catalog_entry')
    def test_should_reorder_multiple_predicates(self, mock):

        def side_effect_func(name):
            if name == 'DummyMultiObjectDetector':
                return MagicMock(cost=10)
            else:
                return MagicMock(cost=5)
        mock.side_effect = side_effect_func
        cheapest_pred = 'id<10'
        cheap_pred = "DummyObjectDetector(data).label = ['person']"
        costly_pred = "DummyMultiObjectDetector(data).labels @> ['person']"
        query = f'SELECT id FROM MyVideo WHERE {costly_pred} AND {cheap_pred} AND {cheapest_pred};'
        plan = get_physical_query_plan(self.evadb, query)
        predicate_plans = list(plan.find_all(PredicatePlan))
        self.assertEqual(len(predicate_plans), 1)
        left = predicate_plans[0].predicate.children[0]
        right = predicate_plans[0].predicate.children[1]
        self.assertIsInstance(left, ComparisonExpression)
        self.assertIsInstance(right, ComparisonExpression)
        self.assertEqual(left.children[0].name, 'DummyObjectDetector')
        self.assertEqual(right.children[0].name, 'DummyMultiObjectDetector')

    def test_reorder_rule_should_not_have_side_effects(self):
        query = 'SELECT id FROM MyVideo WHERE id < 20 AND id > 10;'
        result = execute_query_fetch_all(self.evadb, query)
        rules_manager = RulesManager()
        with disable_rules(rules_manager, [ReorderPredicates()]):
            custom_plan_generator = PlanGenerator(self.evadb, rules_manager)
            expected = execute_query_fetch_all(self.evadb, query, plan_generator=custom_plan_generator)
            self.assertEqual(result, expected)

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

@pytest.mark.notparallel
class SelectExecutorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        video_file_path = create_sample_video(NUM_FRAMES)
        load_query = f"LOAD VIDEO '{video_file_path}' INTO MyVideo;"
        execute_query_fetch_all(cls.evadb, load_query)
        load_functions_for_testing(cls.evadb)
        cls.table1 = create_table(cls.evadb, 'table1', 100, 3)
        cls.table2 = create_table(cls.evadb, 'table2', 500, 3)
        cls.table3 = create_table(cls.evadb, 'table3', 1000, 3)

    @classmethod
    def tearDownClass(cls):
        shutdown_ray()
        file_remove('dummy.avi')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS table1;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS table2;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS table3;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS MyVideo;')

    def test_should_load_and_sort_in_table(self):
        select_query = 'SELECT data, id FROM MyVideo ORDER BY id;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected_rows = [{'myvideo.id': i, 'myvideo.data': np.array(np.ones((32, 32, 3)) * i, dtype=np.uint8)} for i in range(NUM_FRAMES)]
        expected_batch = Batch(frames=pd.DataFrame(expected_rows))
        self.assertEqual(actual_batch, expected_batch)
        select_query = 'SELECT data, id FROM MyVideo ORDER BY id DESC;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected_batch.reverse()
        self.assertEqual(actual_batch, expected_batch)

    def test_should_load_and_select_in_table(self):
        select_query = 'SELECT id FROM MyVideo;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_rows = [{'myvideo.id': i} for i in range(NUM_FRAMES)]
        expected_batch = Batch(frames=pd.DataFrame(expected_rows))
        self.assertEqual(actual_batch, expected_batch)
        select_query = 'SELECT * FROM MyVideo;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches())
        self.assertEqual([actual_batch], expected_batch)

    def test_should_raise_binder_error_on_native_datasource(self):
        select_query = 'SELECT * FROM test.MyVideo'
        self.assertRaises(BinderError, execute_query_fetch_all, self.evadb, select_query)

    def test_should_raise_binder_error_on_non_existent_column(self):
        select_query = 'SELECT b1 FROM table1;'
        with self.assertRaises(BinderError) as ctx:
            execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual("Cannot find column b1. Did you mean a1? The feasible columns are ['_row_id', 'a0', 'a1', 'a2'].", str(ctx.exception))

    def test_should_select_star_in_nested_query(self):
        select_query = 'SELECT * FROM (SELECT * FROM MyVideo) AS T;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches())[0]
        expected_batch.modify_column_alias('T')
        self.assertEqual(actual_batch, expected_batch)
        select_query = 'SELECT * FROM (SELECT id FROM MyVideo) AS T;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_rows = [{'T.id': i} for i in range(NUM_FRAMES)]
        expected_batch = Batch(frames=pd.DataFrame(expected_rows))
        self.assertEqual(actual_batch, expected_batch)

    @unittest.skip('Not supported in current version')
    def test_select_star_in_lateral_join(self):
        select_query = 'SELECT * FROM MyVideo JOIN LATERAL\n                          Yolo(data);'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(actual_batch.frames.columns, ['myvideo.id'])

    def test_should_throw_error_when_both_audio_and_video_selected(self):
        query = "LOAD VIDEO 'data/sample_videos/touchdown.mp4'\n                   INTO TOUCHDOWN1;"
        execute_query_fetch_all(self.evadb, query)
        select_query = 'SELECT id, audio, data FROM TOUCHDOWN1;'
        try:
            execute_query_fetch_all(self.evadb, select_query)
            self.fail("Didn't raise AssertionError")
        except AssertionError as e:
            self.assertEquals('Cannot query over both audio and video streams', e.args[0])

    def test_select_and_limit(self):
        select_query = 'SELECT * FROM MyVideo ORDER BY id LIMIT 5;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches(num_frames=10, batch_size=5))
        self.assertEqual(len(actual_batch), len(expected_batch[0]))
        self.assertEqual(actual_batch, expected_batch[0])

    def test_select_and_aggregate(self):
        simple_aggregate_query = 'SELECT COUNT(*), AVG(id) FROM MyVideo;'
        actual_batch = execute_query_fetch_all(self.evadb, simple_aggregate_query)
        self.assertEqual(actual_batch.frames.iat[0, 0], 10)
        self.assertEqual(actual_batch.frames.iat[0, 1], 4.5)

    def test_select_and_iframe_sample(self):
        select_query = 'SELECT id FROM MyVideo SAMPLE IFRAMES 7 ORDER BY id;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches(filters=range(0, NUM_FRAMES, 7)))
        expected_batch[0] = expected_batch[0].project(['myvideo.id'])
        self.assertEqual(len(actual_batch), len(expected_batch[0]))
        self.assertEqual(actual_batch, expected_batch[0])

    def test_select_and_iframe_sample_without_sampling_rate(self):
        select_query = 'SELECT id FROM MyVideo SAMPLE IFRAMES ORDER BY id;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches(filters=range(0, NUM_FRAMES, 1)))
        expected_batch[0] = expected_batch[0].project(['myvideo.id'])
        self.assertEqual(len(actual_batch), len(expected_batch[0]))
        self.assertEqual(actual_batch, expected_batch[0])

    def test_select_and_groupby_first(self):
        segment_size = 3
        select_query = "SELECT FIRST(id), SEGMENT(data) FROM MyVideo GROUP BY '{} frames';".format(segment_size)
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        ids = np.arange(NUM_FRAMES)
        segments = [ids[i:i + segment_size] for i in range(0, len(ids), segment_size)]
        segments = [i for i in segments if len(i) == segment_size]
        expected_batch = list(create_dummy_4d_batches(filters=segments))[0]
        self.assertEqual(len(actual_batch), len(expected_batch))
        expected_batch.rename(columns={'myvideo.id': 'FIRST.id', 'myvideo.data': 'SEGMENT.data'})
        self.assertEqual(actual_batch, expected_batch.project(['FIRST.id', 'SEGMENT.data']))

    def test_select_and_groupby_with_last(self):
        segment_size = 3
        select_query = "SELECT LAST(id), SEGMENT(data) FROM MyVideo GROUP BY '{}frames';".format(segment_size)
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        ids = np.arange(NUM_FRAMES)
        segments = [ids[i:i + segment_size] for i in range(0, len(ids), segment_size)]
        segments = [i for i in segments if len(i) == segment_size]
        expected_batch = list(create_dummy_4d_batches(filters=segments, start_id=segment_size - 1))[0]
        self.assertEqual(len(actual_batch), len(expected_batch))
        expected_batch.rename(columns={'myvideo.id': 'LAST.id', 'myvideo.data': 'SEGMENT.data'})
        self.assertEqual(actual_batch, expected_batch.project(['LAST.id', 'SEGMENT.data']))

    def test_select_and_groupby_should_fail_with_incorrect_pattern(self):
        segment_size = '4a'
        select_query = "SELECT FIRST(id), SEGMENT(data) FROM MyVideo GROUP BY '{} frames';".format(segment_size)
        self.assertRaises(BinderError, execute_query_fetch_all, self.evadb, select_query)

    def test_select_and_groupby_should_fail_with_seconds(self):
        segment_size = 4
        select_query = "SELECT FIRST(id), SEGMENT(data) FROM MyVideo GROUP BY '{} seconds';".format(segment_size)
        self.assertRaises(BinderError, execute_query_fetch_all, self.evadb, select_query)

    def test_select_and_groupby_should_fail_with_non_video_table(self):
        segment_size = 4
        select_query = "SELECT FIRST(a1) FROM table1 GROUP BY '{} frames';".format(segment_size)
        self.assertRaises(BinderError, execute_query_fetch_all, self.evadb, select_query)

    def test_select_and_groupby_with_sample(self):
        segment_size = 2
        sampling_rate = 2
        select_query = "SELECT FIRST(id), SEGMENT(data) FROM MyVideo SAMPLE {} GROUP BY '{} frames';".format(sampling_rate, segment_size)
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        ids = np.arange(0, NUM_FRAMES, sampling_rate)
        segments = [ids[i:i + segment_size] for i in range(0, len(ids), segment_size)]
        segments = [i for i in segments if len(i) == segment_size]
        expected_batch = list(create_dummy_4d_batches(filters=segments))[0]
        self.assertEqual(len(actual_batch), len(expected_batch))
        expected_batch.rename(columns={'myvideo.id': 'FIRST.id', 'myvideo.data': 'SEGMENT.data'})
        self.assertEqual(actual_batch, expected_batch.project(['FIRST.id', 'SEGMENT.data']))

    def test_select_and_groupby_and_aggregate_with_pdf(self):
        GROUPBY_SIZE = 8
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyPDFs;')
        pdf_path = 'test/data/uadetrac/small-data/pdf_data/fall_2023_orientation_document.pdf'
        load_query = f"LOAD PDF '{pdf_path}' INTO MyPDFs;"
        execute_query_fetch_all(self.evadb, load_query)
        select_all_query = 'SELECT * FROM MyPDFs;'
        all_pdf_batch = execute_query_fetch_all(self.evadb, select_all_query)
        select_query = f"SELECT COUNT(*) FROM MyPDFs GROUP BY '{GROUPBY_SIZE} paragraphs';"
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertAlmostEqual(len(all_pdf_batch), len(actual_batch) * actual_batch.frames.iloc[0, 0], None, None, GROUPBY_SIZE)
        self.assertEqual(len(actual_batch), 99)
        n = len(actual_batch)
        for i in range(n):
            self.assertEqual(actual_batch.frames.iloc[i, 0], GROUPBY_SIZE)
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyPDFs;')

    def test_lateral_join_with_unnest_and_sample(self):
        query = 'SELECT id, label\n                  FROM MyVideo SAMPLE 2 JOIN LATERAL\n                    UNNEST(DummyMultiObjectDetector(data).labels) AS T(label)\n                  WHERE id < 10 ORDER BY id;'
        unnest_batch = execute_query_fetch_all(self.evadb, query)
        expected = Batch(pd.DataFrame({'myvideo.id': np.array([0, 0, 2, 2, 4, 4, 6, 6, 8, 8], dtype=np.intp), 'T.label': np.array(['person', 'person', 'car', 'car', 'bicycle', 'bicycle', 'person', 'person', 'car', 'car'])}))
        self.assertEqual(len(unnest_batch), 10)
        self.assertEqual(unnest_batch, expected)

    def test_select_without_from(self):
        query = 'SELECT 1;'
        batch = execute_query_fetch_all(self.evadb, query)
        expected = Batch(pd.DataFrame([{0: 1}]))
        self.assertEqual(batch, expected)
        query = 'SELECT 1>2;'
        batch = execute_query_fetch_all(self.evadb, query)
        expected = Batch(pd.DataFrame([{0: False}]))
        self.assertEqual(batch, expected)

    def test_should_raise_error_with_missing_alias_in_lateral_join(self):
        function_name = 'DummyMultiObjectDetector'
        query = 'SELECT id, labels\n                  FROM MyVideo JOIN LATERAL DummyMultiObjectDetector(data).labels;'
        with self.assertRaises(SyntaxError) as cm:
            execute_query_fetch_all(self.evadb, query, do_not_print_exceptions=True)
        self.assertEqual(str(cm.exception), f'TableValuedFunction {function_name} should have alias.')
        query = 'SELECT id, labels\n                  FROM MyVideo JOIN LATERAL\n                    UNNEST(DummyMultiObjectDetector(data).labels);'
        with self.assertRaises(SyntaxError) as cm:
            execute_query_fetch_all(self.evadb, query)
        self.assertEqual(str(cm.exception), f'TableValuedFunction {function_name} should have alias.')
        query = 'SELECT id, labels\n                  FROM MyVideo JOIN LATERAL DummyMultiObjectDetector(data);'
        with self.assertRaises(SyntaxError) as cm:
            execute_query_fetch_all(self.evadb, query)
        self.assertEqual(str(cm.exception), f'TableValuedFunction {function_name} should have alias.')

    def test_should_raise_error_with_invalid_number_of_aliases(self):
        function_name = 'DummyMultiObjectDetector'
        query = 'SELECT id, labels\n                  FROM MyVideo JOIN LATERAL\n                    DummyMultiObjectDetector(data).bboxes AS T;'
        with self.assertRaises(BinderError) as cm:
            execute_query_fetch_all(self.evadb, query)
        self.assertEqual(str(cm.exception), f'Output bboxes does not exist for {function_name}.')

    def test_should_raise_error_with_invalid_output_lateral_join(self):
        query = 'SELECT id, a\n                  FROM MyVideo JOIN LATERAL\n                    DummyMultiObjectDetector(data) AS T(a, b);\n                '
        with self.assertRaises(AssertionError) as cm:
            execute_query_fetch_all(self.evadb, query)
        self.assertEqual(str(cm.exception), 'Expected 1 output columns for T, got 2.')

    def test_hash_join_with_one_on(self):
        select_query = 'SELECT * FROM table1 JOIN\n                        table2 ON table1.a1 = table2.a1;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected = pd.merge(self.table1, self.table2, left_on=['table1.a1'], right_on=['table2.a1'], how='inner')
        if len(expected):
            expected_batch = Batch(expected)
            self.assertEqual(expected_batch.sort_orderby(['table1.a2']), actual_batch.sort_orderby(['table1.a2']))

    def test_hash_join_with_multiple_on(self):
        select_query = 'SELECT * FROM table1 JOIN\n                        table1 AS table2 ON table1.a1 = table2.a1 AND\n                        table1.a0 = table2.a0;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected = pd.merge(self.table1, self.table1, left_on=['table1.a1', 'table1.a0'], right_on=['table1.a1', 'table1.a0'], how='inner')
        if len(expected):
            expected_batch = Batch(expected)
            self.assertEqual(expected_batch.sort_orderby(['table1.a1']), actual_batch.sort_orderby(['table1.a1']))

    def test_expression_tree_signature(self):
        plan = get_logical_query_plan(self.evadb, "SELECT id FROM MyVideo WHERE DummyMultiObjectDetector(data).labels @> ['person'];")
        signature = next(plan.find_all(LogicalFilter)).predicate.children[0].signature()
        function_id = self.evadb.catalog().get_function_catalog_entry_by_name('DummyMultiObjectDetector').row_id
        table_entry = self.evadb.catalog().get_table_catalog_entry('MyVideo')
        col_id = self.evadb.catalog().get_column_catalog_entry(table_entry, 'data').row_id
        self.assertEqual(signature, f'DummyMultiObjectDetector[{function_id}](MyVideo.data[{col_id}])')

    def test_function_with_no_input_arguments(self):
        select_query = 'SELECT DummyNoInputFunction();'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected = Batch(pd.DataFrame([{'dummynoinputfunction.label': 'DummyNoInputFunction'}]))
        self.assertEqual(actual_batch, expected)

class StatementBinderTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_bind_tuple_value_expression(self):
        with patch.object(StatementBinderContext, 'get_binded_column') as mock:
            mock.return_value = ['table_alias', 'col_obj']
            binder = StatementBinder(StatementBinderContext(MagicMock()))
            tve = MagicMock()
            tve.name = 'col_name'
            binder._bind_tuple_expr(tve)
            col_alias = '{}.{}'.format('table_alias', 'col_name')
            mock.assert_called_once()
            self.assertEqual(tve.col_object, 'col_obj')
            self.assertEqual(tve.col_alias, col_alias)

    @patch('evadb.binder.statement_binder.bind_table_info')
    def test_bind_tableref(self, mock_bind_table_info):
        with patch.object(StatementBinderContext, 'add_table_alias') as mock:
            catalog = MagicMock()
            binder = StatementBinder(StatementBinderContext(catalog))
            tableref = MagicMock()
            tableref.is_table_atom.return_value = True
            binder._bind_tableref(tableref)
            mock.assert_called_with(tableref.alias.alias_name, tableref.table.database_name, tableref.table.table_name)
            mock_bind_table_info.assert_called_once_with(catalog(), tableref.table)
        with patch.object(StatementBinder, 'bind') as mock_binder:
            with patch.object(StatementBinderContext, 'add_derived_table_alias') as mock_context:
                binder = StatementBinder(StatementBinderContext(MagicMock()))
                tableref = MagicMock()
                tableref.is_table_atom.return_value = False
                tableref.is_select.return_value = True
                binder._bind_tableref(tableref)
                mock_context.assert_called_with(tableref.alias.alias_name, tableref.select_statement.target_list)
                mock_binder.assert_called_with(tableref.select_statement)

    def test_bind_tableref_with_func_expr(self):
        with patch.object(StatementBinder, 'bind') as mock_binder:
            binder = StatementBinder(StatementBinderContext(MagicMock()))
            tableref = MagicMock()
            tableref.is_table_atom.return_value = False
            tableref.is_select.return_value = False
            tableref.is_join.return_value = False
            binder._bind_tableref(tableref)
            mock_binder.assert_called_with(tableref.table_valued_expr.func_expr)

    def test_bind_tableref_with_join(self):
        with patch.object(StatementBinder, 'bind') as mock_binder:
            binder = StatementBinder(StatementBinderContext(MagicMock()))
            tableref = MagicMock()
            tableref.is_table_atom.return_value = False
            tableref.is_select.return_value = False
            tableref.is_join.return_value = True
            binder._bind_tableref(tableref)
            mock_binder.assert_any_call(tableref.join_node.left)
            mock_binder.assert_any_call(tableref.join_node.right)

    def test_bind_tableref_should_raise(self):
        with patch.object(StatementBinder, 'bind'):
            with self.assertRaises(BinderError):
                binder = StatementBinder(StatementBinderContext(MagicMock()))
                tableref = MagicMock()
                tableref.is_select.return_value = False
                tableref.is_table_valued_expr.return_value = False
                tableref.is_join.return_value = False
                tableref.is_table_atom.return_value = False
                binder._bind_tableref(tableref)

    @patch('evadb.binder.statement_binder.StatementBinderContext')
    def test_bind_tableref_starts_new_context(self, mock_ctx):
        with patch.object(StatementBinder, 'bind'):
            binder = StatementBinder(StatementBinderContext(MagicMock()))
            tableref = MagicMock()
            tableref.is_table_atom.return_value = False
            tableref.is_join.return_value = False
            tableref.is_select.return_value = True
            binder._bind_tableref(tableref)
            self.assertEqual(mock_ctx.call_count, 1)

    def test_bind_create_table_from_select_statement(self):
        with patch.object(StatementBinder, 'bind') as mock_binder:
            binder = StatementBinder(StatementBinderContext(MagicMock()))
            output_obj = MagicMock()
            output_obj.type = ColumnType.INTEGER
            output_obj.array_type = NdArrayType.UINT8
            output_obj.array_dimensions = (1, 1)
            create_statement = MagicMock()
            create_statement.column_list = []
            create_statement.query.target_list = [TupleValueExpression(name='id', col_object=output_obj), TupleValueExpression(name='label', col_object=output_obj)]
            binder._bind_create_statement(create_statement)
            mock_binder.assert_called_with(create_statement.query)
            self.assertEqual(2, len(create_statement.column_list))

    def test_bind_explain_statement(self):
        with patch.object(StatementBinder, 'bind') as mock_binder:
            binder = StatementBinder(StatementBinderContext(MagicMock()))
            stmt = MagicMock()
            binder._bind_explain_statement(stmt)
            mock_binder.assert_called_with(stmt.explainable_stmt)

    def test_bind_func_expr_with_star(self):
        func_expr = MagicMock(name='func_expr', alias=Alias('func_expr'), output_col_aliases=[])
        func_expr.name.lower.return_value = 'func_expr'
        func_expr.children = [TupleValueExpression(name='*')]
        binderContext = MagicMock()
        tvp1 = ('T', 'col1')
        tvp2 = ('T', 'col2')
        binderContext._catalog.return_value.get_function_catalog_entry_by_name.return_value = None
        binderContext._get_all_alias_and_col_name.return_value = [tvp1, tvp2]
        with patch.object(StatementBinder, 'bind') as mock_binder:
            binder = StatementBinder(binderContext)
            with self.assertRaises(BinderError):
                binder._bind_func_expr(func_expr)
            call1, call2 = mock_binder.call_args_list
            self.assertEqual(call1.args[0], TupleValueExpression(name=tvp1[1], table_alias=tvp1[0]))
            self.assertEqual(call2.args[0], TupleValueExpression(name=tvp2[1], table_alias=tvp2[0]))

    @patch('evadb.binder.function_expression_binder.load_function_class_from_file')
    def test_bind_func_expr(self, mock_load_function_class_from_file):
        func_expr = MagicMock(name='func_expr', alias=Alias('func_expr'), output_col_aliases=[])
        func_expr.name.lower.return_value = 'func_expr'
        obj1 = MagicMock()
        obj1.name.lower.return_value = 'out1'
        obj2 = MagicMock()
        obj2.name.lower.return_value = 'out2'
        func_output_objs = [obj1, obj2]
        function_obj = MagicMock()
        mock_catalog = MagicMock()
        mock_get_name = mock_catalog().get_function_catalog_entry_by_name = MagicMock()
        mock_get_name.return_value = function_obj
        mock_get_function_outputs = mock_catalog().get_function_io_catalog_output_entries = MagicMock()
        mock_get_function_outputs.return_value = func_output_objs
        mock_load_function_class_from_file.return_value.return_value = 'load_function_class_from_file'
        func_expr.output = 'out1'
        binder = StatementBinder(StatementBinderContext(mock_catalog))
        binder._bind_func_expr(func_expr)
        mock_get_name.assert_called_with(func_expr.name)
        mock_get_function_outputs.assert_called_with(function_obj)
        mock_load_function_class_from_file.assert_called_with(function_obj.impl_file_path, function_obj.name)
        self.assertEqual(func_expr.output_objs, [obj1])
        self.assertEqual(func_expr.alias, Alias('func_expr', ['out1']))
        self.assertEqual(func_expr.function(), 'load_function_class_from_file')
        func_expr.output = None
        func_expr.alias = Alias('func_expr')
        binder = StatementBinder(StatementBinderContext(mock_catalog))
        binder._bind_func_expr(func_expr)
        mock_get_name.assert_called_with(func_expr.name)
        mock_get_function_outputs.assert_called_with(function_obj)
        mock_load_function_class_from_file.assert_called_with(function_obj.impl_file_path, function_obj.name)
        self.assertEqual(func_expr.output_objs, func_output_objs)
        self.assertEqual(func_expr.alias, Alias('func_expr', ['out1', 'out2']))
        self.assertEqual(func_expr.function(), 'load_function_class_from_file')
        mock_load_function_class_from_file.reset_mock()
        mock_error_msg = 'mock_load_function_class_from_file_error'
        mock_load_function_class_from_file.side_effect = MagicMock(side_effect=RuntimeError(mock_error_msg))
        binder = StatementBinder(StatementBinderContext(mock_catalog))
        with self.assertRaises(BinderError) as cm:
            binder._bind_func_expr(func_expr)
        err_msg = f'{mock_error_msg}. Please verify that the function class name in the implementation file matches the function name.'
        self.assertEqual(str(cm.exception), err_msg)

    @patch('evadb.binder.statement_binder.check_table_object_is_groupable')
    @patch('evadb.binder.statement_binder.check_groupby_pattern')
    def test_bind_select_statement(self, is_groupable_mock, groupby_mock):
        with patch.object(StatementBinder, 'bind') as mock_binder:
            binder = StatementBinder(StatementBinderContext(MagicMock()))
            select_statement = MagicMock()
            mocks = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()]
            select_statement.target_list = mocks[:2]
            select_statement.orderby_list = [(mocks[2], 0), (mocks[3], 0)]
            select_statement.groupby_clause = mocks[4]
            select_statement.groupby_clause.value = '8 frames'
            select_statement.from_table.chunk_params = None
            binder._bind_select_statement(select_statement)
            mock_binder.assert_any_call(select_statement.from_table)
            mock_binder.assert_any_call(select_statement.where_clause)
            mock_binder.assert_any_call(select_statement.groupby_clause)
            mock_binder.assert_any_call(select_statement.union_link)
            is_groupable_mock.assert_called()
            for mock in mocks:
                mock_binder.assert_any_call(mock)

    def test_bind_select_statement_without_from(self):
        with patch.object(StatementBinder, 'bind') as mock_binder:
            binder = StatementBinder(StatementBinderContext(MagicMock()))
            expr = MagicMock()
            from evadb.parser.select_statement import SelectStatement
            select_statement = SelectStatement(target_list=[expr])
            binder._bind_select_statement(select_statement)
            mock_binder.assert_not_called_with(select_statement.from_table)
            mock_binder.assert_any_call(expr)

    @patch('evadb.binder.statement_binder.StatementBinderContext')
    def test_bind_select_statement_union_starts_new_context(self, mock_ctx):
        with patch.object(StatementBinder, 'bind'):
            binder = StatementBinder(StatementBinderContext(MagicMock()))
            select_statement = MagicMock()
            select_statement.union_link = None
            select_statement.groupby_clause = None
            select_statement.from_table.chunk_params = None
            binder._bind_select_statement(select_statement)
            self.assertEqual(mock_ctx.call_count, 0)
            binder = StatementBinder(StatementBinderContext(MagicMock()))
            select_statement = MagicMock()
            select_statement.from_table.chunk_params = None
            select_statement.groupby_clause = None
            binder._bind_select_statement(select_statement)
            self.assertEqual(mock_ctx.call_count, 1)

    def test_bind_unknown_object(self):

        class UnknownType:
            pass
        with self.assertRaises(NotImplementedError):
            binder = StatementBinder(StatementBinderContext(MagicMock()))
            binder.bind(UnknownType())

    def test_bind_create_index(self):
        with patch.object(StatementBinder, 'bind'):
            catalog = MagicMock()
            binder = StatementBinder(StatementBinderContext(catalog))
            create_index_statement = MagicMock()
            with self.assertRaises(AssertionError):
                binder._bind_create_index_statement(create_index_statement)
            col_def = MagicMock()
            col_def.name = 'a'
            create_index_statement.col_list = [col_def]
            col = MagicMock()
            col.name = 'a'
            create_index_statement.table_ref.table.table_obj.columns = [col]
            function_obj = MagicMock()
            output = MagicMock()
            function_obj.outputs = [output]
            create_index_statement.project_expr_list = [FunctionExpression(MagicMock(), name='a'), TupleValueExpression(name='*')]
            with patch.object(catalog(), 'get_function_catalog_entry_by_name', return_value=function_obj):
                with self.assertRaises(AssertionError):
                    binder._bind_create_index_statement(create_index_statement)
                output.array_type = NdArrayType.FLOAT32
                with self.assertRaises(AssertionError):
                    binder._bind_create_index_statement(create_index_statement)
                output.array_dimensions = [1, 100]
                binder._bind_create_index_statement(create_index_statement)
            create_index_statement.project_expr_list = [TupleValueExpression(name='*')]
            with self.assertRaises(AssertionError):
                binder._bind_create_index_statement(create_index_statement)
            col.array_type = NdArrayType.FLOAT32
            with self.assertRaises(AssertionError):
                binder._bind_create_index_statement(create_index_statement)
            col.array_dimensions = [1, 10]
            binder._bind_create_index_statement(create_index_statement)

    def test_bind_create_function_should_raise_without_predict_for_ludwig(self):
        with patch.object(StatementBinder, 'bind'):
            create_function_statement = MagicMock()
            create_function_statement.function_type = 'ludwig'
            create_function_statement.query.target_list = []
            create_function_statement.metadata = []
            binder = StatementBinder(StatementBinderContext(MagicMock()))
            with self.assertRaises(AssertionError):
                binder._bind_create_function_statement(create_function_statement)

    def test_bind_create_function_should_drop_row_id_for_select_star(self):
        with patch.object(StatementBinder, 'bind'):
            create_function_statement = MagicMock()
            create_function_statement.function_type = 'ludwig'
            row_id_col_obj = ColumnCatalogEntry(name=IDENTIFIER_COLUMN, type=MagicMock(), array_type=MagicMock(), array_dimensions=MagicMock())
            input_col_obj = ColumnCatalogEntry(name='input_column', type=MagicMock(), array_type=MagicMock(), array_dimensions=MagicMock())
            output_col_obj = ColumnCatalogEntry(name='predict_column', type=MagicMock(), array_type=MagicMock(), array_dimensions=MagicMock())
            create_function_statement.query.target_list = [TupleValueExpression(name=IDENTIFIER_COLUMN, table_alias='a', col_object=row_id_col_obj), TupleValueExpression(name='input_column', table_alias='a', col_object=input_col_obj), TupleValueExpression(name='predict_column', table_alias='a', col_object=output_col_obj)]
            create_function_statement.metadata = [('predict', 'predict_column')]
            binder = StatementBinder(StatementBinderContext(MagicMock()))
            binder._bind_create_function_statement(create_function_statement)
            self.assertEqual(create_function_statement.query.target_list, [TupleValueExpression(name='input_column', table_alias='a', col_object=input_col_obj), TupleValueExpression(name='predict_column', table_alias='a', col_object=output_col_obj)])
            expected_inputs = [ColumnDefinition('input_column', input_col_obj.type, input_col_obj.array_type, input_col_obj.array_dimensions)]
            expected_outputs = [ColumnDefinition('predict_column_predictions', output_col_obj.type, output_col_obj.array_type, output_col_obj.array_dimensions)]
            self.assertEqual(create_function_statement.inputs, expected_inputs)
            self.assertEqual(create_function_statement.outputs, expected_outputs)

    def test_bind_create_function_should_bind_forecast_with_default_columns(self):
        with patch.object(StatementBinder, 'bind'):
            create_function_statement = MagicMock()
            create_function_statement.function_type = 'forecasting'
            id_col_obj = ColumnCatalogEntry(name='unique_id', type=MagicMock(), array_type=MagicMock(), array_dimensions=MagicMock())
            ds_col_obj = ColumnCatalogEntry(name='ds', type=MagicMock(), array_type=MagicMock(), array_dimensions=MagicMock())
            y_col_obj = ColumnCatalogEntry(name='y', type=MagicMock(), array_type=MagicMock(), array_dimensions=MagicMock())
            y_lo_col_obj = ColumnCatalogEntry(name='y-lo', type=ColumnType.FLOAT, array_type=None)
            y_hi_col_obj = ColumnCatalogEntry(name='y-hi', type=ColumnType.FLOAT, array_type=None)
            create_function_statement.query.target_list = [TupleValueExpression(name=id_col_obj.name, table_alias='a', col_object=id_col_obj), TupleValueExpression(name=ds_col_obj.name, table_alias='a', col_object=ds_col_obj), TupleValueExpression(name=y_col_obj.name, table_alias='a', col_object=y_col_obj)]
            create_function_statement.metadata = []
            binder = StatementBinder(StatementBinderContext(MagicMock()))
            binder._bind_create_function_statement(create_function_statement)
            expected_inputs = [ColumnDefinition('horizon', ColumnType.INTEGER, None, None)]
            expected_outputs = list([ColumnDefinition(col_obj.name, col_obj.type, col_obj.array_type, col_obj.array_dimensions) for col_obj in (id_col_obj, ds_col_obj, y_col_obj, y_lo_col_obj, y_hi_col_obj)])
            self.assertEqual(create_function_statement.inputs, expected_inputs)
            self.assertEqual(create_function_statement.outputs, expected_outputs)

    def test_bind_create_function_should_bind_forecast_with_renaming_columns(self):
        with patch.object(StatementBinder, 'bind'):
            create_function_statement = MagicMock()
            create_function_statement.function_type = 'forecasting'
            id_col_obj = ColumnCatalogEntry(name='type', type=MagicMock(), array_type=MagicMock(), array_dimensions=MagicMock())
            ds_col_obj = ColumnCatalogEntry(name='saledate', type=MagicMock(), array_type=MagicMock(), array_dimensions=MagicMock())
            y_col_obj = ColumnCatalogEntry(name='ma', type=MagicMock(), array_type=MagicMock(), array_dimensions=MagicMock())
            y_lo_col_obj = ColumnCatalogEntry(name='ma-lo', type=ColumnType.FLOAT, array_type=None)
            y_hi_col_obj = ColumnCatalogEntry(name='ma-hi', type=ColumnType.FLOAT, array_type=None)
            create_function_statement.query.target_list = [TupleValueExpression(name=id_col_obj.name, table_alias='a', col_object=id_col_obj), TupleValueExpression(name=ds_col_obj.name, table_alias='a', col_object=ds_col_obj), TupleValueExpression(name=y_col_obj.name, table_alias='a', col_object=y_col_obj)]
            create_function_statement.metadata = [('predict', 'ma'), ('id', 'type'), ('time', 'saledate')]
            binder = StatementBinder(StatementBinderContext(MagicMock()))
            binder._bind_create_function_statement(create_function_statement)
            expected_inputs = [ColumnDefinition('horizon', ColumnType.INTEGER, None, None)]
            expected_outputs = list([ColumnDefinition(col_obj.name, col_obj.type, col_obj.array_type, col_obj.array_dimensions) for col_obj in (id_col_obj, ds_col_obj, y_col_obj, y_lo_col_obj, y_hi_col_obj)])
            self.assertEqual(create_function_statement.inputs, expected_inputs)
            self.assertEqual(create_function_statement.outputs, expected_outputs)

    def test_bind_create_function_should_raise_forecast_missing_required_columns(self):
        with patch.object(StatementBinder, 'bind'):
            create_function_statement = MagicMock()
            create_function_statement.function_type = 'forecasting'
            id_col_obj = ColumnCatalogEntry(name='type', type=MagicMock(), array_type=MagicMock(), array_dimensions=MagicMock())
            ds_col_obj = ColumnCatalogEntry(name='saledate', type=MagicMock(), array_type=MagicMock(), array_dimensions=MagicMock())
            create_function_statement.query.target_list = [TupleValueExpression(name=id_col_obj.name, table_alias='a', col_object=id_col_obj), TupleValueExpression(name=ds_col_obj.name, table_alias='a', col_object=ds_col_obj)]
            create_function_statement.metadata = [('id', 'type'), ('time', 'saledate'), ('predict', 'ma')]
            binder = StatementBinder(StatementBinderContext(MagicMock()))
            with self.assertRaises(AssertionError) as cm:
                binder._bind_create_function_statement(create_function_statement)
            err_msg = "Missing required {'ma'} columns for forecasting function."
            self.assertEqual(str(cm.exception), err_msg)

@pytest.mark.notparallel
class StatementBinderTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_check_duplicate_alias(self):
        with self.assertRaises(BinderError):
            ctx = StatementBinderContext(MagicMock())
            ctx._derived_table_alias_map['alias'] = MagicMock()
            ctx._check_duplicate_alias('alias')
        with self.assertRaises(BinderError):
            ctx = StatementBinderContext(MagicMock())
            ctx._table_alias_map['alias'] = MagicMock()
            ctx._check_duplicate_alias('alias')
        ctx = StatementBinderContext(MagicMock())
        ctx._check_duplicate_alias('alias')

    def test_add_table_alias(self):
        mock_catalog = MagicMock()
        mock_get = mock_catalog().get_table_catalog_entry = MagicMock()
        mock_get.return_value = 'table_obj'
        ctx = StatementBinderContext(mock_catalog)
        mock_check = ctx._check_duplicate_alias = MagicMock()
        ctx.add_table_alias('alias', None, 'table_name')
        mock_check.assert_called_with('alias')
        mock_get.assert_called_with('table_name')
        self.assertEqual(ctx._table_alias_map['alias'], 'table_obj')

    def test_add_derived_table_alias(self):
        objs = [MagicMock(), MagicMock()]
        attributes = {'name': 'A', 'col_object': 'A_obj'}
        mock = MagicMock(spec=TupleValueExpression)
        for attr, value in attributes.items():
            setattr(mock, attr, value)
        exprs = [mock, MagicMock(spec=FunctionExpression, output_objs=objs)]
        ctx = StatementBinderContext(MagicMock())
        mock_check = ctx._check_duplicate_alias = MagicMock()
        ctx.add_derived_table_alias('alias', exprs)
        mock_check.assert_called_with('alias')
        col_map = {'A': 'A_obj', objs[0].name: objs[0], objs[1].name: objs[1]}
        self.assertEqual(ctx._derived_table_alias_map['alias'], col_map)

    def test_get_binded_column_should_search_all(self):
        ctx = StatementBinderContext(MagicMock())
        mock_search_all = ctx._search_all_alias_maps = MagicMock()
        mock_search_all.return_value = ('alias', 'col_obj')
        result = ctx.get_binded_column('col_name')
        mock_search_all.assert_called_once_with('col_name')
        self.assertEqual(result, ('alias', 'col_obj'))

    def test_get_binded_column_check_table_alias_map(self):
        ctx = StatementBinderContext(MagicMock())
        mock_table_map = ctx._check_table_alias_map = MagicMock()
        mock_table_map.return_value = 'col_obj'
        result = ctx.get_binded_column('col_name', 'alias')
        mock_table_map.assert_called_once_with('alias', 'col_name')
        self.assertEqual(result, ('alias', 'col_obj'))

    def test_get_binded_column_check_derived_table_alias_map(self):
        ctx = StatementBinderContext(MagicMock())
        mock_table_map = ctx._check_table_alias_map = MagicMock()
        mock_table_map.return_value = None
        mock_derived_map = ctx._check_derived_table_alias_map = MagicMock()
        mock_derived_map.return_value = 'col_obj'
        result = ctx.get_binded_column('col_name', 'alias')
        mock_table_map.assert_called_once_with('alias', 'col_name')
        mock_derived_map.assert_called_once_with('alias', 'col_name')
        self.assertEqual(result, ('alias', 'col_obj'))

    def test_get_binded_column_raise_error(self):
        with self.assertRaises(BinderError):
            ctx = StatementBinderContext(MagicMock())
            mock_search_all = ctx._search_all_alias_maps = MagicMock()
            mock_search_all.return_value = (None, None)
            ctx.get_binded_column('col_name')
        with self.assertRaises(BinderError):
            ctx = StatementBinderContext(MagicMock())
            mock_table_map = ctx._check_table_alias_map = MagicMock()
            mock_table_map.return_value = None
            mock_derived_map = ctx._check_derived_table_alias_map = MagicMock()
            mock_derived_map.return_value = None
            ctx.get_binded_column('col_name', 'alias')

    def test_check_table_alias_map(self):
        mock_catalog = MagicMock()
        mock_get_column_object = mock_catalog().get_column_catalog_entry = MagicMock()
        mock_get_column_object.return_value = 'catalog_value'
        ctx = StatementBinderContext(mock_catalog)
        table_obj = MagicMock()
        ctx._table_alias_map['alias'] = table_obj
        result = ctx._check_table_alias_map('alias', 'col_name')
        mock_get_column_object.assert_called_once_with(table_obj, 'col_name')
        self.assertEqual(result, 'catalog_value')
        mock_get_column_object.reset_mock()
        ctx = StatementBinderContext(mock_catalog)
        result = ctx._check_table_alias_map('alias', 'col_name')
        mock_get_column_object.assert_not_called()
        self.assertEqual(result, None)

    def test_check_derived_table_alias_map(self):
        ctx = StatementBinderContext(MagicMock())
        obj1 = MagicMock()
        obj2 = MagicMock()
        col_map = {'col1': obj1, 'col2': obj2}
        ctx._derived_table_alias_map['alias'] = col_map
        result = ctx._check_derived_table_alias_map('alias', 'col1')
        self.assertEqual(result, obj1)
        result = ctx._check_derived_table_alias_map('alias', 'col2')
        self.assertEqual(result, obj2)
        ctx = StatementBinderContext(MagicMock())
        result = ctx._check_derived_table_alias_map('alias', 'col3')
        self.assertEqual(result, None)

    def test_search_all_alias_maps(self):
        ctx = StatementBinderContext(MagicMock())
        check_table_map = ctx._check_table_alias_map = MagicMock()
        check_derived_map = ctx._check_derived_table_alias_map = MagicMock()
        check_table_map.return_value = 'col_obj'
        ctx._table_alias_map['alias'] = 'col_name'
        ctx._derived_table_alias_map = {}
        result = ctx._search_all_alias_maps('col_name')
        check_table_map.assert_called_once_with('alias', 'col_name')
        check_derived_map.assert_not_called()
        self.assertEqual(result, ('alias', 'col_obj'))
        check_derived_map.return_value = 'derived_col_obj'
        ctx._table_alias_map = {}
        ctx._derived_table_alias_map['alias'] = 'col_name'
        result = ctx._search_all_alias_maps('col_name')
        check_table_map.assert_called_once_with('alias', 'col_name')
        check_table_map.assert_called_once_with('alias', 'col_name')
        self.assertEqual(result, ('alias', 'derived_col_obj'))

    def test_search_all_alias_raise_duplicate_error(self):
        with self.assertRaises(BinderError):
            ctx = StatementBinderContext(MagicMock())
            ctx._check_table_alias_map = MagicMock()
            ctx._check_derived_table_alias_map = MagicMock()
            ctx._table_alias_map['alias'] = 'col_name'
            ctx._derived_table_alias_map['alias'] = 'col_name'
            ctx._search_all_alias_maps('col_name')

def execute_statement(evadb: EvaDBDatabase, stmt: AbstractStatement, do_not_raise_exceptions: bool=False, do_not_print_exceptions: bool=False, **kwargs) -> Iterator[Batch]:
    plan_generator = kwargs.get('plan_generator', PlanGenerator(evadb))
    if not isinstance(stmt, SKIP_BINDER_AND_OPTIMIZER_STATEMENTS):
        StatementBinder(StatementBinderContext(evadb.catalog)).bind(stmt)
        logical_plan = StatementToPlanConverter().visit(stmt)
        physical_plan = plan_generator.build(logical_plan)
    else:
        physical_plan = stmt
    output = PlanExecutor(evadb, physical_plan).execute_plan(do_not_raise_exceptions, do_not_print_exceptions)
    if output:
        batch_list = list(output)
        return Batch.concat(batch_list, copy=False)

class CatalogManager(object):

    def __init__(self, db_uri: str):
        self._db_uri = db_uri
        self._sql_config = SQLConfig(db_uri)
        self._bootstrap_catalog()
        self._db_catalog_service = DatabaseCatalogService(self._sql_config.session)
        self._config_catalog_service = ConfigurationCatalogService(self._sql_config.session)
        self._job_catalog_service = JobCatalogService(self._sql_config.session)
        self._job_history_catalog_service = JobHistoryCatalogService(self._sql_config.session)
        self._table_catalog_service = TableCatalogService(self._sql_config.session)
        self._column_service = ColumnCatalogService(self._sql_config.session)
        self._function_service = FunctionCatalogService(self._sql_config.session)
        self._function_cost_catalog_service = FunctionCostCatalogService(self._sql_config.session)
        self._function_io_service = FunctionIOCatalogService(self._sql_config.session)
        self._function_metadata_service = FunctionMetadataCatalogService(self._sql_config.session)
        self._index_service = IndexCatalogService(self._sql_config.session)
        self._function_cache_service = FunctionCacheCatalogService(self._sql_config.session)

    @property
    def sql_config(self):
        return self._sql_config

    def reset(self):
        """
        This method resets the state of the singleton instance.
        It should clear the contents of the catalog tables and any storage data
        Used by testcases to reset the db state before
        """
        self._clear_catalog_contents()

    def close(self):
        """
        This method closes all the connections
        """
        if self.sql_config is not None:
            sqlalchemy_engine = self.sql_config.engine
            sqlalchemy_engine.dispose()

    def _bootstrap_catalog(self):
        """Bootstraps catalog.
        This method runs all tasks required for using catalog. Currently,
        it includes only one task ie. initializing database. It creates the
        catalog database and tables if they do not exist.
        """
        logger.info('Bootstrapping catalog')
        init_db(self._sql_config.engine)

    def _clear_catalog_contents(self):
        """
        This method is responsible for clearing the contents of the
        catalog. It clears the tuples in the catalog tables, indexes, and cached data.
        """
        logger.info('Clearing catalog')
        drop_all_tables_except_catalog(self._sql_config.engine)
        truncate_catalog_tables(self._sql_config.engine, tables_not_to_truncate=['configuration_catalog'])
        for folder in ['cache_dir', 'index_dir', 'datasets_dir']:
            remove_directory_contents(self.get_configuration_catalog_value(folder))
    'Database catalog services'

    def insert_database_catalog_entry(self, name: str, engine: str, params: dict):
        """A new entry is persisted in the database catalog."

        Args:
            name: database name
            engine: engine name
            params: required params as a dictionary for the database
        """
        self._db_catalog_service.insert_entry(name, engine, params)

    def get_database_catalog_entry(self, database_name: str) -> DatabaseCatalogEntry:
        """
        Returns the database catalog entry for the given database_name
        Arguments:
            database_name (str): name of the database

        Returns:
            DatabaseCatalogEntry
        """
        table_entry = self._db_catalog_service.get_entry_by_name(database_name)
        return table_entry

    def get_all_database_catalog_entries(self):
        return self._db_catalog_service.get_all_entries()

    def drop_database_catalog_entry(self, database_entry: DatabaseCatalogEntry) -> bool:
        """
        This method deletes the database from  catalog.

        Arguments:
           database_entry: database catalog entry to remove

        Returns:
           True if successfully deleted else False
        """
        return self._db_catalog_service.delete_entry(database_entry)

    def check_native_table_exists(self, table_name: str, database_name: str):
        """
        Validate the database is valid and the requested table in database is
        also valid.
        """
        db_catalog_entry = self.get_database_catalog_entry(database_name)
        if db_catalog_entry is None:
            return False
        with get_database_handler(db_catalog_entry.engine, **db_catalog_entry.params) as handler:
            resp = handler.get_tables()
            if resp.error is not None:
                raise Exception(resp.error)
            table_df = resp.data
            if table_name not in table_df['table_name'].values:
                return False
        return True
    'Job catalog services'

    def insert_job_catalog_entry(self, name: str, queries: str, start_time: datetime, end_time: datetime, repeat_interval: int, active: bool, next_schedule_run: datetime) -> JobCatalogEntry:
        """A new entry is persisted in the job catalog.

        Args:
            name: job name
            queries: job's queries
            start_time: job start time
            end_time: job end time
            repeat_interval: job repeat interval
            active: job status
            next_schedule_run: next run time as per schedule
        """
        job_entry = self._job_catalog_service.insert_entry(name, queries, start_time, end_time, repeat_interval, active, next_schedule_run)
        return job_entry

    def get_job_catalog_entry(self, job_name: str) -> JobCatalogEntry:
        """
        Returns the job catalog entry for the given database_name
        Arguments:
            job_name (str): name of the job

        Returns:
            JobCatalogEntry
        """
        table_entry = self._job_catalog_service.get_entry_by_name(job_name)
        return table_entry

    def drop_job_catalog_entry(self, job_entry: JobCatalogEntry) -> bool:
        """
        This method deletes the job from  catalog.

        Arguments:
           job_entry: job catalog entry to remove

        Returns:
           True if successfully deleted else False
        """
        return self._job_catalog_service.delete_entry(job_entry)

    def get_next_executable_job(self, only_past_jobs: bool=False) -> JobCatalogEntry:
        """Get the oldest job that is ready to be triggered by trigger time
        Arguments:
            only_past_jobs: boolean flag to denote if only jobs with trigger time in
                past should be considered
        Returns:
            Returns the first job to be triggered
        """
        return self._job_catalog_service.get_next_executable_job(only_past_jobs)

    def update_job_catalog_entry(self, job_name: str, next_scheduled_run: datetime, active: bool):
        """Update the next_scheduled_run and active column as per the provided values
        Arguments:
            job_name (str): job which should be updated

            next_run_time (datetime): the next trigger time for the job

            active (bool): the active status for the job
        """
        self._job_catalog_service.update_next_scheduled_run(job_name, next_scheduled_run, active)
    'Job history catalog services'

    def insert_job_history_catalog_entry(self, job_id: str, job_name: str, execution_start_time: datetime, execution_end_time: datetime) -> JobCatalogEntry:
        """A new entry is persisted in the job history catalog.

        Args:
            job_id: job id for the execution entry
            job_name: job name for the execution entry
            execution_start_time: job execution start time
            execution_end_time: job execution end time
        """
        job_history_entry = self._job_history_catalog_service.insert_entry(job_id, job_name, execution_start_time, execution_end_time)
        return job_history_entry

    def get_job_history_by_job_id(self, job_id: int) -> List[JobHistoryCatalogEntry]:
        """Returns all the entries present for this job_id on in the history.

        Args:
            job_id: the id of job whose history should be fetched
        """
        return self._job_history_catalog_service.get_entry_by_job_id(job_id)

    def update_job_history_end_time(self, job_id: int, execution_start_time: datetime, execution_end_time: datetime) -> List[JobHistoryCatalogEntry]:
        """Updates the execution_end_time for this job history matching job_id and execution_start_time.

        Args:
            job_id: id of the job whose history entry which should be updated
            execution_start_time: the start time for the job history entry
            execution_end_time: the end time for the job history entry
        """
        return self._job_history_catalog_service.update_entry_end_time(job_id, execution_start_time, execution_end_time)
    'Table catalog services'

    def insert_table_catalog_entry(self, name: str, file_url: str, column_list: List[ColumnCatalogEntry], identifier_column='id', table_type=TableType.VIDEO_DATA) -> TableCatalogEntry:
        """A new entry is added to the table catalog and persisted in the database.
        The schema field is set before the object is returned."

        Args:
            name: table name
            file_url: #todo
            column_list: list of columns
            identifier_column (str):  A unique identifier column for each row
            table_type (TableType): type of the table, video, images etc
        Returns:
            The persisted TableCatalogEntry object with the id field populated.
        """
        column_list = [ColumnCatalogEntry(name=IDENTIFIER_COLUMN, type=ColumnType.INTEGER)] + column_list
        table_entry = self._table_catalog_service.insert_entry(name, file_url, identifier_column=identifier_column, table_type=table_type, column_list=column_list)
        return table_entry

    def get_table_catalog_entry(self, table_name: str, database_name: str=None) -> TableCatalogEntry:
        """
        Returns the table catalog entry for the given table name
        Arguments:
            table_name (str): name of the table

        Returns:
            TableCatalogEntry
        """
        table_entry = self._table_catalog_service.get_entry_by_name(database_name, table_name)
        return table_entry

    def delete_table_catalog_entry(self, table_entry: TableCatalogEntry) -> bool:
        """
        This method deletes the table along with its columns from table catalog
        and column catalog respectively

        Arguments:
           table: table catalog entry to remove

        Returns:
           True if successfully deleted else False
        """
        return self._table_catalog_service.delete_entry(table_entry)

    def rename_table_catalog_entry(self, curr_table: TableCatalogEntry, new_name: TableInfo):
        return self._table_catalog_service.rename_entry(curr_table, new_name.table_name)

    def check_table_exists(self, table_name: str, database_name: str=None):
        is_native_table = database_name is not None
        if is_native_table:
            return self.check_native_table_exists(table_name, database_name)
        else:
            table_entry = self._table_catalog_service.get_entry_by_name(database_name, table_name)
            return table_entry is not None

    def get_all_table_catalog_entries(self):
        return self._table_catalog_service.get_all_entries()
    'Column catalog services'

    def get_column_catalog_entry(self, table_obj: TableCatalogEntry, col_name: str) -> ColumnCatalogEntry:
        col_obj = self._column_service.filter_entry_by_table_id_and_name(table_obj.row_id, col_name)
        if col_obj:
            return col_obj
        else:
            if col_name == VideoColumnName.audio:
                return ColumnCatalogEntry(col_name, ColumnType.NDARRAY, table_id=table_obj.row_id, table_name=table_obj.name)
            return None

    def get_column_catalog_entries_by_table(self, table_obj: TableCatalogEntry):
        col_entries = self._column_service.filter_entries_by_table(table_obj)
        return col_entries
    'function catalog services'

    def insert_function_catalog_entry(self, name: str, impl_file_path: str, type: str, function_io_list: List[FunctionIOCatalogEntry], function_metadata_list: List[FunctionMetadataCatalogEntry]) -> FunctionCatalogEntry:
        """Inserts a function catalog entry along with Function_IO entries.
        It persists the entry to the database.

        Arguments:
            name(str): name of the function
            impl_file_path(str): implementation path of the function
            type(str): what kind of function operator like classification,
                                                        detection etc
            function_io_list(List[FunctionIOCatalogEntry]): input/output function info list

        Returns:
            The persisted FunctionCatalogEntry object.
        """
        checksum = get_file_checksum(impl_file_path)
        function_entry = self._function_service.insert_entry(name, impl_file_path, type, checksum, function_io_list, function_metadata_list)
        return function_entry

    def get_function_catalog_entry_by_name(self, name: str) -> FunctionCatalogEntry:
        """
        Get the function information based on name.

        Arguments:
             name (str): name of the function

        Returns:
            FunctionCatalogEntry object
        """
        return self._function_service.get_entry_by_name(name)

    def delete_function_catalog_entry_by_name(self, function_name: str) -> bool:
        return self._function_service.delete_entry_by_name(function_name)

    def get_all_function_catalog_entries(self):
        return self._function_service.get_all_entries()
    'function cost catalog services'

    def upsert_function_cost_catalog_entry(self, function_id: int, name: str, cost: int) -> FunctionCostCatalogEntry:
        """Upserts function cost catalog entry.

        Arguments:
            function_id(int): unique function id
            name(str): the name of the function
            cost(int): cost of this function

        Returns:
            The persisted FunctionCostCatalogEntry object.
        """
        self._function_cost_catalog_service.upsert_entry(function_id, name, cost)

    def get_function_cost_catalog_entry(self, name: str):
        return self._function_cost_catalog_service.get_entry_by_name(name)
    'FunctionIO services'

    def get_function_io_catalog_input_entries(self, function_obj: FunctionCatalogEntry) -> List[FunctionIOCatalogEntry]:
        return self._function_io_service.get_input_entries_by_function_id(function_obj.row_id)

    def get_function_io_catalog_output_entries(self, function_obj: FunctionCatalogEntry) -> List[FunctionIOCatalogEntry]:
        return self._function_io_service.get_output_entries_by_function_id(function_obj.row_id)
    ' Index related services. '

    def insert_index_catalog_entry(self, name: str, save_file_path: str, vector_store_type: VectorStoreType, feat_column: ColumnCatalogEntry, function_signature: str, index_def: str) -> IndexCatalogEntry:
        index_catalog_entry = self._index_service.insert_entry(name, save_file_path, vector_store_type, feat_column, function_signature, index_def)
        return index_catalog_entry

    def get_index_catalog_entry_by_name(self, name: str) -> IndexCatalogEntry:
        return self._index_service.get_entry_by_name(name)

    def get_index_catalog_entry_by_column_and_function_signature(self, column: ColumnCatalogEntry, function_signature: str):
        return self._index_service.get_entry_by_column_and_function_signature(column, function_signature)

    def drop_index_catalog_entry(self, index_name: str) -> bool:
        return self._index_service.delete_entry_by_name(index_name)

    def get_all_index_catalog_entries(self):
        return self._index_service.get_all_entries()
    ' Function Cache related'

    def insert_function_cache_catalog_entry(self, func_expr: FunctionExpression):
        cache_dir = self.get_configuration_catalog_value('cache_dir')
        entry = construct_function_cache_catalog_entry(func_expr, cache_dir=cache_dir)
        return self._function_cache_service.insert_entry(entry)

    def get_function_cache_catalog_entry_by_name(self, name: str) -> FunctionCacheCatalogEntry:
        return self._function_cache_service.get_entry_by_name(name)

    def drop_function_cache_catalog_entry(self, entry: FunctionCacheCatalogEntry) -> bool:
        if entry:
            shutil.rmtree(entry.cache_path)
        return self._function_cache_service.delete_entry(entry)
    ' function Metadata Catalog'

    def get_function_metadata_entries_by_function_name(self, function_name: str) -> List[FunctionMetadataCatalogEntry]:
        """
        Get the function metadata information for the provided function.

        Arguments:
             function_name (str): name of the function

        Returns:
            FunctionMetadataCatalogEntry objects
        """
        function_entry = self.get_function_catalog_entry_by_name(function_name)
        if function_entry:
            entries = self._function_metadata_service.get_entries_by_function_id(function_entry.row_id)
            return entries
        else:
            return []
    ' Utils '

    def create_and_insert_table_catalog_entry(self, table_info: TableInfo, columns: List[ColumnDefinition], identifier_column: str=None, table_type: TableType=TableType.STRUCTURED_DATA) -> TableCatalogEntry:
        """Create a valid table catalog tuple and insert into the table

        Args:
            table_info (TableInfo): table info object
            columns (List[ColumnDefinition]): columns definitions of the table
            identifier_column (str, optional): Specify unique columns. Defaults to None.
            table_type (TableType, optional): table type. Defaults to TableType.STRUCTURED_DATA.

        Returns:
            TableCatalogEntry: entry that has been inserted into the table catalog
        """
        table_name = table_info.table_name
        column_catalog_entries = xform_column_definitions_to_catalog_entries(columns)
        dataset_location = self.get_configuration_catalog_value('datasets_dir')
        file_url = str(generate_file_path(dataset_location, table_name))
        table_catalog_entry = self.insert_table_catalog_entry(table_name, file_url, column_catalog_entries, identifier_column=identifier_column, table_type=table_type)
        return table_catalog_entry

    def create_and_insert_multimedia_table_catalog_entry(self, name: str, format_type: FileFormatType) -> TableCatalogEntry:
        """Create a table catalog entry for the multimedia table.
        Depending on the type of multimedia, the appropriate "create catalog entry" command is called.

        Args:
            name (str):  name of the table catalog entry
            format_type (FileFormatType): media type

        Raises:
            CatalogError: if format_type is not supported

        Returns:
            TableCatalogEntry: newly inserted table catalog entry
        """
        assert format_type in [FileFormatType.VIDEO, FileFormatType.IMAGE, FileFormatType.DOCUMENT, FileFormatType.PDF], f'Format Type {format_type} is not supported'
        if format_type is FileFormatType.VIDEO:
            columns = get_video_table_column_definitions()
            table_type = TableType.VIDEO_DATA
        elif format_type is FileFormatType.IMAGE:
            columns = get_image_table_column_definitions()
            table_type = TableType.IMAGE_DATA
        elif format_type is FileFormatType.DOCUMENT:
            columns = get_document_table_column_definitions()
            table_type = TableType.DOCUMENT_DATA
        elif format_type is FileFormatType.PDF:
            columns = get_pdf_table_column_definitions()
            table_type = TableType.PDF_DATA
        return self.create_and_insert_table_catalog_entry(TableInfo(name), columns, table_type=table_type)

    def get_multimedia_metadata_table_catalog_entry(self, input_table: TableCatalogEntry) -> TableCatalogEntry:
        """Get table catalog entry for multimedia metadata table.
        Raise if it does not exists
        Args:
            input_table (TableCatalogEntryEntryEntryEntry): input media table

        Returns:
            TableCatalogEntry: metainfo table entry which is maintained by the system
        """
        media_metadata_name = Path(input_table.file_url).stem
        obj = self.get_table_catalog_entry(media_metadata_name)
        assert obj is not None, f'Table with name {media_metadata_name} does not exist in catalog'
        return obj

    def create_and_insert_multimedia_metadata_table_catalog_entry(self, input_table: TableCatalogEntry) -> TableCatalogEntry:
        """Create and insert table catalog entry for multimedia metadata table.
         This table is used to store all media filenames and related information. In
         order to prevent direct access or modification by users, it should be
         designated as a SYSTEM_STRUCTURED_DATA type.
         **Note**: this table is managed by the storage engine, so it should not be
         called elsewhere.
        Args:
            input_table (TableCatalogEntry): input video table

        Returns:
            TableCatalogEntry: metainfo table entry which is maintained by the system
        """
        media_metadata_name = Path(input_table.file_url).stem
        obj = self.get_table_catalog_entry(media_metadata_name)
        assert obj is None, 'Table with name {media_metadata_name} already exists'
        columns = [ColumnDefinition('file_url', ColumnType.TEXT, None, None)]
        obj = self.create_and_insert_table_catalog_entry(TableInfo(media_metadata_name), columns, identifier_column=columns[0].name, table_type=TableType.SYSTEM_STRUCTURED_DATA)
        return obj
    'Configuration catalog services'

    def upsert_configuration_catalog_entry(self, key: str, value: any):
        """Upserts configuration catalog entry"

        Args:
            key: key name
            value: value name
        """
        self._config_catalog_service.upsert_entry(key, value)

    def get_configuration_catalog_value(self, key: str, default: Any=None) -> Any:
        """
        Returns the value entry for the given key
        Arguments:
            key (str): key name

        Returns:
            ConfigurationCatalogEntry
        """
        table_entry = self._config_catalog_service.get_entry_by_name(key)
        if table_entry:
            return table_entry.value
        return default

    def get_all_configuration_catalog_entries(self) -> List:
        return self._config_catalog_service.get_all_entries()

class ColumnCatalog(BaseModel):
    """The `ColumnCatalog` catalog stores information about the columns of the table.
    It maintains the following information for each column
    `_row_id:` an autogenerated identifier
    `_name: ` name of the column
    `_type:` the type of the column, refer `ColumnType`
    `_is_nullable:` which indicates whether the column is nullable
    `_array_type:` the type of array, as specified in `NdArrayType` (or `None` if the column is a primitive type)
    `_array_dimensions:` the dimensions of the array (if `_array_type` is not `None`)
    `_table_id:` the `_row_id` of the `TableCatalog` entry to which the column belongs
    `_dep_caches`: list of function caches associated with the column
    """
    __tablename__ = 'column_catalog'
    _name = Column('name', String(100))
    _type = Column('type', Enum(ColumnType), default=Enum)
    _is_nullable = Column('is_nullable', Boolean, default=False)
    _array_type = Column('array_type', Enum(NdArrayType), nullable=True)
    _array_dimensions = Column('array_dimensions', String(100))
    _table_id = Column('table_id', Integer, ForeignKey('table_catalog._row_id'))
    __table_args__ = (UniqueConstraint('name', 'table_id'), {})
    _table_catalog = relationship('TableCatalog', back_populates='_columns')
    _dep_caches = relationship('FunctionCacheCatalog', secondary=depend_column_and_function_cache, back_populates='_col_depends', cascade='all, delete')
    _index_column = relationship('IndexCatalog', back_populates='_feat_column', cascade='all, delete')

    def __init__(self, name: str, type: ColumnType, is_nullable: bool=False, array_type: NdArrayType=None, array_dimensions: Tuple[int]=(), table_id: int=None):
        self._name = name
        self._type = type
        self._is_nullable = is_nullable
        self._array_type = array_type
        self.array_dimensions = array_dimensions
        self._table_id = table_id

    @property
    def array_dimensions(self):
        return literal_eval(self._array_dimensions)

    @array_dimensions.setter
    def array_dimensions(self, value: Tuple[int]):
        dimensions = []
        for dim in value:
            if dim == Dimension.ANYDIM:
                dimensions.append(None)
            else:
                dimensions.append(dim)
        self._array_dimensions = str(tuple(dimensions))

    def as_dataclass(self) -> 'ColumnCatalogEntry':
        return ColumnCatalogEntry(row_id=self._row_id, name=self._name, type=self._type, is_nullable=self._is_nullable, array_type=self._array_type, array_dimensions=self.array_dimensions, table_id=self._table_id, table_name=self._table_catalog._name, dep_caches=[cache.as_dataclass() for cache in self._dep_caches])

def parse_expression(expr: str):
    mock_query = f'SELECT {expr} FROM DUMMY;'
    stmt = Parser().parse(mock_query)[0]
    assert isinstance(stmt, SelectStatement), 'Expected a select statement'
    return stmt.target_list

def try_binding(catalog: Callable, stmt: AbstractStatement):
    StatementBinder(StatementBinderContext(catalog)).bind(stmt.copy())

def sql_string_to_expresssion_list(expr: str) -> List[AbstractExpression]:
    """Converts the sql expression to list of evadb abstract expressions

    Args:
        expr (str): the expr to convert

    Returns:
        List[AbstractExpression]: list of evadb abstract expressions

    """
    return parse_expression(expr)

def create_limit_expression(num: int):
    return ConstantValueExpression(num)

class EvaDBQuery:

    def __init__(self, evadb: EvaDBDatabase, query_node: Union[AbstractStatement, TableRef], alias: Alias=None):
        self._evadb = evadb
        self._query_node = query_node
        self._alias = alias

    def alias(self, alias: str) -> 'EvaDBQuery':
        """Returns a new Relation with an alias set.

        Args:
            alias (str): an alias name to be set for the Relation.

        Returns:
            EvaDBQuery: Aliased Relation.

        Examples:
            >>> relation = cursor.table("sample_table")
            >>> relation.alias('table')
        """
        self._alias = Alias(alias)

    def cross_apply(self, expr: str, alias: str) -> 'EvaDBQuery':
        """Execute a expr on all the rows of the relation

        Args:
            expr (str): sql expression
            alias (str): alias of the output of the expr

        Returns:
            `EvaDBQuery`: relation

        Examples:

            Runs Yolo on all the frames of the input table

            >>> relation = cursor.table("videos")
            >>> relation.cross_apply("Yolo(data)", "objs(labels, bboxes, scores)")

            Runs Yolo on all the frames of the input table and unnest each object as separate row.

            >>> relation.cross_apply("unnest(Yolo(data))", "obj(label, bbox, score)")
        """
        assert self._query_node.from_table is not None
        table_ref = string_to_lateral_join(expr, alias=alias)
        join_table = TableRef(JoinNode(TableRef(self._query_node, alias=self._alias), table_ref, join_type=JoinType.LATERAL_JOIN))
        self._query_node = SelectStatement(target_list=create_star_expression(), from_table=join_table)
        self._alias = Alias('Relation')
        try_binding(self._evadb.catalog, self._query_node)
        return self

    def df(self, drop_alias: bool=True) -> pandas.DataFrame:
        """
        Execute and fetch all rows as a pandas DataFrame

        Args:
            drop_alias (bool): whether to drop the table name in the output dataframe. Default: True.

        Returns:
            pandas.DataFrame:

        Example:

        Runs a SQL query and get a panda Dataframe.
            >>> cursor.query("SELECT * FROM MyTable;").df()
                col1  col2
            0      1     2
            1      3     4
            2      5     6
        """
        batch = self.execute(drop_alias=drop_alias)
        assert batch.frames is not None, 'relation execute failed'
        return batch.frames

    def execute(self, drop_alias: bool=True) -> Batch:
        """Transform the relation into a result set

        Args:
            drop_alias (bool): whether to drop the table name in the output batch. Default: True.

        Returns:
            Batch: result as evadb Batch

        Example:

            Runs a SQL query and get a Batch
            >>> batch = cursor.query("SELECT * FROM MyTable;").execute()
        """
        result = execute_statement(self._evadb, self._query_node.copy())
        if drop_alias:
            result.drop_column_alias()
        assert result is not None
        return result

    def filter(self, expr: str) -> 'EvaDBQuery':
        """
        Filters rows using the given condition. Multiple filters can be chained using `AND`

        Parameters:
            expr (str): The filter expression.

        Returns:
            EvaDBQuery : Filtered EvaDBQuery.
        Examples:
            >>> relation = cursor.table("sample_table")
            >>> relation.filter("col1 > 10")

            Filter by sql string

            >>> relation.filter("col1 > 10 AND col1 < 20")

        """
        parsed_expr = sql_predicate_to_expresssion_tree(expr)
        self._query_node = handle_select_clause(self._query_node, self._alias, 'where_clause', parsed_expr)
        try_binding(self._evadb.catalog, self._query_node)
        return self

    def limit(self, num: int) -> 'EvaDBQuery':
        """Limits the result count to the number specified.

        Args:
            num (int): Number of records to return. Will return num records or all records if the Relation contains fewer records.

        Returns:
            EvaDBQuery: Relation with subset of records

        Examples:
            >>> relation = cursor.table("sample_table")
            >>> relation.limit(10)

        """
        limit_expr = create_limit_expression(num)
        self._query_node = handle_select_clause(self._query_node, self._alias, 'limit_count', limit_expr)
        try_binding(self._evadb.catalog, self._query_node)
        return self

    def order(self, order_expr: str) -> 'EvaDBQuery':
        """Reorder the relation based on the order_expr

        Args:
            order_expr (str): sql expression to order the relation

        Returns:
            EvaDBQuery: A EvaDBQuery ordered based on the order_expr.

        Examples:
            >>> relation = cursor.table("PDFs")
            >>> relation.order("Similarity(SentenceTransformerFeatureExtractor('When was the NATO created?'), SentenceTransformerFeatureExtractor(data) ) DESC")

        """
        parsed_expr = parse_sql_orderby_expr(order_expr)
        self._query_node = handle_select_clause(self._query_node, self._alias, 'orderby_list', parsed_expr)
        try_binding(self._evadb.catalog, self._query_node)
        return self

    def select(self, expr: str) -> 'EvaDBQuery':
        """
        Projects a set of expressions and returns a new EvaDBQuery.

        Parameters:
            exprs (Union[str, List[str]]): The expression(s) to be selected. If '*' is provided, it expands to all columns in the current EvaDBQuery.

        Returns:
            EvaDBQuery: A EvaDBQuery with subset (or all) of columns.

        Examples:
            >>> relation = cursor.table("sample_table")

            Select all columns in the EvaDBQuery.

            >>> relation.select("*")

            Select all subset of columns in the EvaDBQuery.

            >>> relation.select("col1")
            >>> relation.select("col1, col2")
        """
        parsed_exprs = sql_string_to_expresssion_list(expr)
        self._query_node = handle_select_clause(self._query_node, self._alias, 'target_list', parsed_exprs)
        try_binding(self._evadb.catalog, self._query_node)
        return self

    def show(self) -> pandas.DataFrame:
        """Execute and fetch all rows as a pandas DataFrame

        Returns:
            pandas.DataFrame:
        """
        batch = self.execute()
        assert batch is not None, 'relation execute failed'
        return batch.frames

    def sql_query(self) -> str:
        """Get the SQL query that is equivalent to the relation

        Returns:
            str: the sql query

        Examples:
            >>> relation = cursor.table("sample_table").project('i')
            >>> relation.sql_query()
        """
        return str(self._query_node)

