# Cluster 45

class StatementToOprTest(unittest.TestCase):

    @patch('evadb.optimizer.statement_to_opr_converter.LogicalGet')
    def test_visit_table_ref_should_create_logical_get_opr(self, mock_lget):
        converter = StatementToPlanConverter()
        table_ref = MagicMock(spec=TableRef, alias='alias', chunk_params={})
        table_ref.is_select.return_value = False
        table_ref.sample_freq = None
        converter.visit_table_ref(table_ref)
        mock_lget.assert_called_with(table_ref, table_ref.table.table_obj, 'alias', chunk_params=table_ref.chunk_params)
        self.assertEqual(mock_lget.return_value, converter._plan)

    @patch('evadb.optimizer.statement_to_opr_converter.LogicalFilter')
    def test_visit_select_predicate_should_add_logical_filter(self, mock_lfilter):
        converter = StatementToPlanConverter()
        select_predicate = MagicMock()
        converter._visit_select_predicate(select_predicate)
        mock_lfilter.assert_called_with(select_predicate)
        mock_lfilter.return_value.append_child.assert_called()
        self.assertEqual(mock_lfilter.return_value, converter._plan)

    @patch('evadb.optimizer.statement_to_opr_converter.LogicalProject')
    def test_visit_projection_should_add_logical_predicate(self, mock_lproject):
        converter = StatementToPlanConverter()
        projects = MagicMock()
        converter._plan = MagicMock()
        converter._visit_projection(projects)
        mock_lproject.assert_called_with(projects)
        mock_lproject.return_value.append_child.assert_called()
        self.assertEqual(mock_lproject.return_value, converter._plan)

    @patch('evadb.optimizer.statement_to_opr_converter.LogicalProject')
    def test_visit_projection_should_not_add_logical_predicate(self, mock_lproject):
        converter = StatementToPlanConverter()
        projects = MagicMock()
        converter._plan = None
        converter._visit_projection(projects)
        mock_lproject.assert_called_with(projects)
        mock_lproject.return_value.append_child.assert_not_called()
        self.assertEqual(mock_lproject.return_value, converter._plan)

    def test_visit_select_should_call_appropriate_visit_methods(self):
        converter = StatementToPlanConverter()
        converter.visit_table_ref = MagicMock()
        converter._visit_projection = MagicMock()
        converter._visit_select_predicate = MagicMock()
        converter._visit_union = MagicMock()
        statement = MagicMock()
        statement.from_table = MagicMock(spec=TableRef)
        converter.visit_select(statement)
        converter.visit_table_ref.assert_called_with(statement.from_table)
        converter._visit_projection.assert_called_with(statement.target_list)
        converter._visit_select_predicate.assert_called_with(statement.where_clause)

    def test_visit_select_should_not_call_visits_for_null_values(self):
        converter = StatementToPlanConverter()
        converter.visit_table_ref = MagicMock()
        converter._visit_projection = MagicMock()
        converter._visit_select_predicate = MagicMock()
        converter._visit_union = MagicMock()
        statement = SelectStatement()
        converter.visit_select(statement)
        converter.visit_table_ref.assert_not_called()
        converter._visit_projection.assert_not_called()
        converter._visit_select_predicate.assert_not_called()

    def test_visit_select_without_table_ref(self):
        converter = StatementToPlanConverter()
        converter.visit_table_ref = MagicMock()
        converter._visit_projection = MagicMock()
        converter._visit_select_predicate = MagicMock()
        converter._visit_union = MagicMock()
        converter._visit_groupby = MagicMock()
        converter._visit_orderby = MagicMock()
        converter._visit_limit = MagicMock()
        column_list = MagicMock()
        statement = SelectStatement(target_list=column_list)
        converter.visit_select(statement)
        converter.visit_table_ref.assert_not_called()
        converter._visit_projection.assert_called_once_with(column_list)
        converter._visit_select_predicate.assert_not_called()
        converter._visit_union.assert_not_called()
        converter._visit_groupby.assert_not_called()
        converter._visit_orderby.assert_not_called()
        converter._visit_limit.assert_not_called()

    @patch('evadb.optimizer.statement_to_opr_converter.LogicalCreateFunction')
    @patch('evadb.optimizer.statement_to_opr_converter.column_definition_to_function_io')
    @patch('evadb.optimizer.statement_to_opr_converter.metadata_definition_to_function_metadata')
    def test_visit_create_function(self, metadata_def_mock, col_def_mock, l_create_function_mock):
        converter = StatementToPlanConverter()
        stmt = MagicMock()
        stmt.name = 'name'
        stmt.or_replace = False
        stmt.if_not_exists = True
        stmt.inputs = ['inp']
        stmt.outputs = ['out']
        stmt.impl_path = 'tmp.py'
        stmt.function_type = 'classification'
        stmt.query = None
        stmt.metadata = [('key1', 'value1'), ('key2', 'value2')]
        col_def_mock.side_effect = ['inp', 'out']
        metadata_def_mock.side_effect = [{'key1': 'value1', 'key2': 'value2'}]
        converter.visit_create_function(stmt)
        col_def_mock.assert_any_call(stmt.inputs, True)
        col_def_mock.assert_any_call(stmt.outputs, False)
        metadata_def_mock.assert_any_call(stmt.metadata)
        l_create_function_mock.assert_called_once()
        l_create_function_mock.assert_called_with(stmt.name, stmt.or_replace, stmt.if_not_exists, 'inp', 'out', stmt.impl_path, stmt.function_type, {'key1': 'value1', 'key2': 'value2'})

    def test_visit_should_call_create_function(self):
        stmt = MagicMock(spec=CreateFunctionStatement)
        converter = StatementToPlanConverter()
        mock = MagicMock()
        converter.visit_create_function = mock
        converter.visit(stmt)
        mock.assert_called_once()
        mock.assert_called_with(stmt)

    @patch('evadb.optimizer.statement_to_opr_converter.LogicalDropObject')
    def test_visit_drop_object(self, l_drop_obj_mock):
        converter = StatementToPlanConverter()
        stmt = MagicMock()
        stmt.name = 'name'
        stmt.object_type = 'object_type'
        stmt.if_exists = True
        converter.visit_drop_object(stmt)
        l_drop_obj_mock.assert_called_once()
        l_drop_obj_mock.assert_called_with(stmt.object_type, stmt.name, stmt.if_exists)

    def test_visit_should_call_insert(self):
        stmt = MagicMock(spec=InsertTableStatement)
        converter = StatementToPlanConverter()
        mock = MagicMock()
        converter.visit_insert = mock
        converter.visit(stmt)
        mock.assert_called_once()
        mock.assert_called_with(stmt)

    def test_visit_should_call_create(self):
        stmt = MagicMock(spec=CreateTableStatement)
        converter = StatementToPlanConverter()
        mock = MagicMock()
        converter.visit_create = mock
        converter.visit(stmt)
        mock.assert_called_once()
        mock.assert_called_with(stmt)

    def test_visit_should_call_rename(self):
        stmt = MagicMock(spec=RenameTableStatement)
        converter = StatementToPlanConverter()
        mock = MagicMock()
        converter.visit_rename = mock
        converter.visit(stmt)
        mock.assert_called_once()
        mock.assert_called_with(stmt)

    def test_visit_should_call_explain(self):
        stmt = MagicMock(spec=ExplainStatement)
        converter = StatementToPlanConverter()
        mock = MagicMock()
        converter.visit_explain = mock
        converter.visit(stmt)
        mock.assert_called_once()
        mock.assert_called_once_with(stmt)

    def test_visit_should_call_drop(self):
        stmt = MagicMock(spec=DropObjectStatement)
        converter = StatementToPlanConverter()
        mock = MagicMock()
        converter.visit_drop_object = mock
        converter.visit(stmt)
        mock.assert_called_once()
        mock.assert_called_with(stmt)

    def test_visit_should_call_create_index(self):
        stmt = MagicMock(spec=CreateIndexStatement)
        converter = StatementToPlanConverter()
        mock = MagicMock()
        converter.visit_create_index = mock
        converter.visit(stmt)
        mock.assert_called_once()
        mock.assert_called_with(stmt)

    def test_inequality_in_operator(self):
        dummy_plan = Dummy(MagicMock(), MagicMock())
        object = MagicMock()
        self.assertNotEqual(dummy_plan, object)

    def test_check_plan_equality(self):
        plans = []
        dummy_plan = Dummy(MagicMock(), MagicMock())
        create_plan = LogicalCreate(MagicMock(), MagicMock())
        create_function_plan = LogicalCreateFunction(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        create_index_plan = LogicalCreateIndex(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        delete_plan = LogicalDelete(MagicMock())
        insert_plan = LogicalInsert(MagicMock(), MagicMock(), [MagicMock()], [MagicMock()])
        query_derived_plan = LogicalQueryDerivedGet(MagicMock())
        load_plan = LogicalLoadData(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        limit_plan = LogicalLimit(MagicMock())
        rename_plan = LogicalRename(MagicMock(), MagicMock())
        explain_plan = LogicalExplain([MagicMock()])
        exchange_plan = LogicalExchange(MagicMock())
        show_plan = LogicalShow(MagicMock())
        drop_plan = LogicalDropObject(MagicMock(), MagicMock(), MagicMock())
        get_plan = LogicalGet(MagicMock(), MagicMock(), MagicMock())
        sample_plan = LogicalSample(MagicMock(), MagicMock())
        filter_plan = LogicalFilter(MagicMock())
        faiss_plan = LogicalVectorIndexScan(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        groupby_plan = LogicalGroupBy(MagicMock())
        order_by_plan = LogicalOrderBy(MagicMock())
        union_plan = LogicalUnion(MagicMock())
        function_scan_plan = LogicalFunctionScan(MagicMock(), MagicMock())
        join_plan = LogicalJoin(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        project_plan = LogicalProject(MagicMock(), MagicMock())
        apply_and_merge_plan = LogicalApplyAndMerge(MagicMock(), MagicMock())
        extract_object_plan = LogicalExtractObject(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        create_plan.append_child(create_function_plan)
        plans.append(dummy_plan)
        plans.append(create_plan)
        plans.append(create_function_plan)
        plans.append(create_index_plan)
        plans.append(delete_plan)
        plans.append(insert_plan)
        plans.append(query_derived_plan)
        plans.append(load_plan)
        plans.append(limit_plan)
        plans.append(rename_plan)
        plans.append(drop_plan)
        plans.append(get_plan)
        plans.append(sample_plan)
        plans.append(filter_plan)
        plans.append(groupby_plan)
        plans.append(order_by_plan)
        plans.append(union_plan)
        plans.append(function_scan_plan)
        plans.append(join_plan)
        plans.append(apply_and_merge_plan)
        plans.append(show_plan)
        plans.append(explain_plan)
        plans.append(exchange_plan)
        plans.append(faiss_plan)
        plans.append(project_plan)
        plans.append(extract_object_plan)
        derived_operators = list(get_all_subclasses(Operator))
        plan_type_list = []
        for plan in plans:
            plan_type_list.append(type(plan))
        length = len(plans)
        self.assertEqual(length, len(derived_operators))
        self.assertEqual(len(list(set(derived_operators) - set(plan_type_list))), 0)
        for i in range(length):
            self.assertEqual(plans[i], plans[i])
            self.assertNotEqual(str(plans[i]), None)
            if plans[i] != dummy_plan:
                self.assertNotEqual(plans[i], dummy_plan)
            if i >= 1:
                self.assertNotEqual(plans[i - 1], plans[i])
        derived_operators = list(get_all_subclasses(Operator))
        for derived_operator in derived_operators:
            sig = signature(derived_operator.__init__)
            params = sig.parameters
            self.assertLess(len(params), 15)

class TestOptimizerTask(unittest.TestCase):

    def execute_task_stack(self, task_stack):
        while not task_stack.empty():
            task = task_stack.pop()
            task.execute()

    def test_abstract_optimizer_task(self):
        task = OptimizerTask(MagicMock(), MagicMock())
        with self.assertRaises(NotImplementedError):
            task.execute()

    def top_down_rewrite(self, opr):
        opt_cxt = OptimizerContext(MagicMock(), CostModel(), RulesManager())
        grp_expr = opt_cxt.add_opr_to_group(opr)
        root_grp_id = grp_expr.group_id
        opt_cxt.task_stack.push(TopDownRewrite(grp_expr, RulesManager().stage_one_rewrite_rules, opt_cxt))
        self.execute_task_stack(opt_cxt.task_stack)
        return (opt_cxt, root_grp_id)

    def bottom_up_rewrite(self, root_grp_id, opt_cxt):
        grp_expr = opt_cxt.memo.groups[root_grp_id].logical_exprs[0]
        opt_cxt.task_stack.push(BottomUpRewrite(grp_expr, RulesManager().stage_two_rewrite_rules, opt_cxt))
        self.execute_task_stack(opt_cxt.task_stack)
        return (opt_cxt, root_grp_id)

    def implement_group(self, root_grp_id, opt_cxt):
        grp = opt_cxt.memo.groups[root_grp_id]
        opt_cxt.task_stack.push(OptimizeGroup(grp, opt_cxt))
        self.execute_task_stack(opt_cxt.task_stack)
        return (opt_cxt, root_grp_id)

    def test_simple_implementation(self):
        predicate = MagicMock()
        child_opr = LogicalGet(MagicMock(), MagicMock(), MagicMock())
        root_opr = LogicalFilter(predicate, [child_opr])
        opt_cxt, root_grp_id = self.top_down_rewrite(root_opr)
        opt_cxt, root_grp_id = self.bottom_up_rewrite(root_grp_id, opt_cxt)
        opt_cxt, root_grp_id = self.implement_group(root_grp_id, opt_cxt)
        root_grp = opt_cxt.memo.groups[root_grp_id]
        best_root_grp_expr = root_grp.get_best_expr(PropertyType.DEFAULT)
        self.assertEqual(type(best_root_grp_expr.opr), PredicatePlan)

    def test_nested_implementation(self):
        child_predicate = MagicMock()
        root_predicate = MagicMock()
        with patch('evadb.optimizer.rules.rules.extract_pushdown_predicate') as mock:
            with patch('evadb.optimizer.rules.rules.is_video_table') as mock_vid:
                mock_vid.return_value = True
                mock.side_effect = [(child_predicate, None), (root_predicate, None)]
                child_get_opr = LogicalGet(MagicMock(), MagicMock(), MagicMock())
                child_filter_opr = LogicalFilter(child_predicate, children=[child_get_opr])
                child_project_opr = LogicalProject([MagicMock()], children=[child_filter_opr])
                root_derived_get_opr = LogicalQueryDerivedGet(MagicMock(), children=[child_project_opr])
                root_filter_opr = LogicalFilter(root_predicate, children=[root_derived_get_opr])
                root_project_opr = LogicalProject([MagicMock()], children=[root_filter_opr])
                opt_cxt, root_grp_id = self.top_down_rewrite(root_project_opr)
                opt_cxt, root_grp_id = self.bottom_up_rewrite(root_grp_id, opt_cxt)
                opt_cxt, root_grp_id = self.implement_group(root_grp_id, opt_cxt)
                expected_expr_order = [ProjectPlan, PredicatePlan, SeqScanPlan, ProjectPlan, SeqScanPlan]
                curr_grp_id = root_grp_id
                idx = 0
                while True:
                    root_grp = opt_cxt.memo.groups[curr_grp_id]
                    best_root_grp_expr = root_grp.get_best_expr(PropertyType.DEFAULT)
                    self.assertEqual(type(best_root_grp_expr.opr), expected_expr_order[idx])
                    idx += 1
                    if idx == len(expected_expr_order):
                        break
                    curr_grp_id = best_root_grp_expr.children[0]

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

class ExpressionUtilsTest(unittest.TestCase):

    def gen_cmp_expr(self, val, expr_type=ExpressionType.COMPARE_GREATER, name='id', const_first=False):
        constexpr = ConstantValueExpression(val)
        colname = TupleValueExpression(name=name, col_alias=f'T.{name}')
        if const_first:
            return ComparisonExpression(expr_type, constexpr, colname)
        return ComparisonExpression(expr_type, colname, constexpr)

    def test_extract_range_list_from_comparison_expr(self):
        expr_types = [ExpressionType.COMPARE_NEQ, ExpressionType.COMPARE_EQUAL, ExpressionType.COMPARE_GREATER, ExpressionType.COMPARE_LESSER, ExpressionType.COMPARE_GEQ, ExpressionType.COMPARE_LEQ]
        results = []
        for expr_type in expr_types:
            cmpr_exp = self.gen_cmp_expr(10, expr_type, const_first=True)
            results.append(extract_range_list_from_comparison_expr(cmpr_exp, 0, 100))
        expected = [[(0, 9), (11, 100)], [(10, 10)], [(0, 9)], [(11, 100)], [(0, 10)], [(10, 100)]]
        self.assertEqual(results, expected)
        results = []
        for expr_type in expr_types:
            cmpr_exp = self.gen_cmp_expr(10, expr_type)
            results.append(extract_range_list_from_comparison_expr(cmpr_exp, 0, 100))
        expected = [[(0, 9), (11, 100)], [(10, 10)], [(11, 100)], [(0, 9)], [(10, 100)], [(0, 10)]]
        self.assertEqual(results, expected)
        with self.assertRaises(RuntimeError):
            cmpr_exp = LogicalExpression(ExpressionType.LOGICAL_AND, Mock(), Mock())
            extract_range_list_from_comparison_expr(cmpr_exp, 0, 100)
        with self.assertRaises(RuntimeError):
            cmpr_exp = self.gen_cmp_expr(10, ExpressionType.COMPARE_CONTAINS)
            extract_range_list_from_comparison_expr(cmpr_exp, 0, 100)
        with self.assertRaises(RuntimeError):
            cmpr_exp = self.gen_cmp_expr(10, ExpressionType.COMPARE_IS_CONTAINED)
            extract_range_list_from_comparison_expr(cmpr_exp, 0, 100)

    def test_extract_range_list_from_predicate(self):
        expr = LogicalExpression(ExpressionType.LOGICAL_AND, self.gen_cmp_expr(10), self.gen_cmp_expr(20))
        self.assertEqual(extract_range_list_from_predicate(expr, 0, 100), [(21, 100)])
        expr = LogicalExpression(ExpressionType.LOGICAL_OR, self.gen_cmp_expr(10), self.gen_cmp_expr(20))
        self.assertEqual(extract_range_list_from_predicate(expr, 0, 100), [(11, 100)])
        expr1 = LogicalExpression(ExpressionType.LOGICAL_OR, self.gen_cmp_expr(10), self.gen_cmp_expr(20))
        expr2 = LogicalExpression(ExpressionType.LOGICAL_AND, self.gen_cmp_expr(10), self.gen_cmp_expr(5, ExpressionType.COMPARE_LESSER))
        expr = LogicalExpression(ExpressionType.LOGICAL_OR, expr1, expr2)
        self.assertEqual(extract_range_list_from_predicate(expr, 0, 100), [(11, 100)])
        expr = LogicalExpression(ExpressionType.LOGICAL_AND, expr1, expr2)
        self.assertEqual(extract_range_list_from_predicate(expr, 0, 100), [])
        expr1 = LogicalExpression(ExpressionType.LOGICAL_OR, self.gen_cmp_expr(10, ExpressionType.COMPARE_LESSER), self.gen_cmp_expr(20))
        expr2 = LogicalExpression(ExpressionType.LOGICAL_OR, self.gen_cmp_expr(25), self.gen_cmp_expr(5, ExpressionType.COMPARE_LESSER))
        expr = LogicalExpression(ExpressionType.LOGICAL_OR, expr1, expr2)
        self.assertEqual(extract_range_list_from_predicate(expr, 0, 100), [(0, 9), (21, 100)])
        with self.assertRaises(RuntimeError):
            expr = ArithmeticExpression(ExpressionType.AGGREGATION_COUNT, Mock(), Mock())
            extract_range_list_from_predicate(expr, 0, 100)

    def test_predicate_contains_single_column(self):
        self.assertTrue(contains_single_column(self.gen_cmp_expr(10)))
        expr1 = LogicalExpression(ExpressionType.LOGICAL_OR, self.gen_cmp_expr(10, ExpressionType.COMPARE_GREATER, 'x'), self.gen_cmp_expr(10, ExpressionType.COMPARE_GREATER, 'x'))
        self.assertTrue(contains_single_column(expr1))
        expr2 = LogicalExpression(ExpressionType.LOGICAL_OR, self.gen_cmp_expr(10, ExpressionType.COMPARE_GREATER, 'x'), self.gen_cmp_expr(10, ExpressionType.COMPARE_GREATER, 'y'))
        self.assertFalse(contains_single_column(expr2))
        expr = LogicalExpression(ExpressionType.LOGICAL_OR, expr1, expr2)
        self.assertFalse(contains_single_column(expr))

    def test_is_simple_predicate(self):
        self.assertTrue(is_simple_predicate(self.gen_cmp_expr(10)))
        expr = ArithmeticExpression(ExpressionType.AGGREGATION_COUNT, Mock(), Mock())
        self.assertFalse(is_simple_predicate(expr))
        expr = LogicalExpression(ExpressionType.LOGICAL_OR, self.gen_cmp_expr(10, ExpressionType.COMPARE_GREATER, 'x'), self.gen_cmp_expr(10, ExpressionType.COMPARE_GREATER, 'y'))
        self.assertFalse(is_simple_predicate(expr))

    def test_and_(self):
        expr1 = self.gen_cmp_expr(10)
        expr2 = self.gen_cmp_expr(20)
        new_expr = conjunction_list_to_expression_tree([expr1, expr2])
        self.assertEqual(new_expr.etype, ExpressionType.LOGICAL_AND)
        self.assertEqual(new_expr.children[0], expr1)
        self.assertEqual(new_expr.children[1], expr2)

def contains_single_column(predicate: AbstractExpression, column: str=None) -> bool:
    """Checks if predicate contains conditions on single predicate

    Args:
        predicate (AbstractExpression): predicate expression
        column_alias (str): check if the single column matches
            the input column_alias
    Returns:
        bool: True, if contains single predicate, else False
            if predicate is None, return False
    """
    if not predicate:
        return False
    cols = get_columns_in_predicate(predicate)
    if len(cols) == 1:
        if column is None:
            return True
        pred_col = cols.pop()
        if pred_col == column:
            return True
    return False

def is_simple_predicate(predicate: AbstractExpression) -> bool:
    """Checks if conditions in the predicate are on a single column and
        only contains LogicalExpression, ComparisonExpression,
        TupleValueExpression or ConstantValueExpression

    Args:
        predicate (AbstractExpression): predicate expression to check

    Returns:
        bool: True, if it is a simple predicate, else False
    """

    def _has_simple_expressions(expr):
        simple = type(expr) in simple_expressions
        for child in expr.children:
            simple = simple and _has_simple_expressions(child)
        return simple
    simple_expressions = [LogicalExpression, ComparisonExpression, TupleValueExpression, ConstantValueExpression]
    return _has_simple_expressions(predicate) and contains_single_column(predicate)

def conjunction_list_to_expression_tree(expression_list: List[AbstractExpression]) -> AbstractExpression:
    """Convert expression list to expression tree using conjunction connector

    [a, b, c] -> AND( AND(a, b), c)
    Args:
        expression_list (List[AbstractExpression]): list of conjunctives

    Returns:
        AbstractExpression: expression tree

    Example:
        conjunction_list_to_expression_tree([a, b, c] ): AND( AND(a, b), c)
    """
    if len(expression_list) == 0:
        return None
    prev_expr = expression_list[0]
    for expr in expression_list[1:]:
        if expr is not None:
            prev_expr = LogicalExpression(ExpressionType.LOGICAL_AND, prev_expr, expr)
    return prev_expr

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

class Binder:

    def __init__(self, grp_expr: GroupExpression, pattern: Pattern, memo: Memo):
        self._grp_expr = grp_expr
        self._pattern = pattern
        self._memo = memo

    @staticmethod
    def _grp_binder(idx: int, pattern: Pattern, memo: Memo):
        grp = memo.groups[idx]
        for expr in grp.logical_exprs:
            yield from Binder._binder(expr, pattern, memo)

    @staticmethod
    def _binder(expr: GroupExpression, pattern: Pattern, memo: Memo):
        assert isinstance(expr, GroupExpression)
        curr_iterator = iter(())
        child_binders = []
        if pattern.opr_type is not OperatorType.DUMMY:
            curr_iterator = iter([expr.opr])
            if expr.opr.opr_type != pattern.opr_type:
                return
            if len(pattern.children) != len(expr.children):
                return
            for child_grp, pattern_child in zip(expr.children, pattern.children):
                child_binders.append(Binder._grp_binder(child_grp, pattern_child, memo))
        else:
            curr_iterator = iter([Dummy(expr.group_id, expr.opr)])
        yield from itertools.product(curr_iterator, *child_binders)

    @staticmethod
    def build_opr_tree_from_pre_order_repr(pre_order_repr: tuple) -> Operator:
        opr_tree = pre_order_repr[0]
        assert isinstance(opr_tree, Operator), f'Unknown operator encountered, expected Operator found {type(opr_tree)}'
        opr_tree = copy.copy(opr_tree)
        opr_tree.children.clear()
        if len(pre_order_repr) > 1:
            for child in pre_order_repr[1:]:
                opr_tree.append_child(Binder.build_opr_tree_from_pre_order_repr(child))
        return opr_tree

    def __iter__(self):
        for match in Binder._binder(self._grp_expr, self._pattern, self._memo):
            x = Binder.build_opr_tree_from_pre_order_repr(match)
            yield x

def extract_equi_join_keys(join_predicate: AbstractExpression, left_table_aliases: List[Alias], right_table_aliases: List[Alias]) -> Tuple[List[AbstractExpression], List[AbstractExpression]]:
    pred_list = to_conjunction_list(join_predicate)
    left_join_keys = []
    right_join_keys = []
    left_table_alias_strs = [left_table_alias.alias_name for left_table_alias in left_table_aliases]
    right_table_alias_strs = [right_table_alias.alias_name for right_table_alias in right_table_aliases]
    for pred in pred_list:
        if pred.etype == ExpressionType.COMPARE_EQUAL:
            left_child = pred.children[0]
            right_child = pred.children[1]
            if left_child.etype == ExpressionType.TUPLE_VALUE and right_child.etype == ExpressionType.TUPLE_VALUE:
                if left_child.table_alias in left_table_alias_strs and right_child.table_alias in right_table_alias_strs:
                    left_join_keys.append(left_child)
                    right_join_keys.append(right_child)
                elif left_child.table_alias in right_table_alias_strs and right_child.table_alias in left_table_alias_strs:
                    left_join_keys.append(right_child)
                    right_join_keys.append(left_child)
    return (left_join_keys, right_join_keys)

def to_conjunction_list(expression_tree: AbstractExpression) -> List[AbstractExpression]:
    """Convert expression tree to list of conjunctives

    Note: It does not normalize the expression tree before extracting the conjunctives.

    Args:
        expression_tree (AbstractExpression): expression tree to transform

    Returns:
        List[AbstractExpression]: list of conjunctives

    Example:
        to_conjunction_list(AND(AND(a,b), OR(c,d))): [a, b, OR(c,d)]
        to_conjunction_list(OR(AND(a,b), c)): [OR(AND(a,b), c)]
            returns the original expression, does not normalize
    """
    expression_list = []
    if expression_tree.etype == ExpressionType.LOGICAL_AND:
        expression_list.extend(to_conjunction_list(expression_tree.children[0]))
        expression_list.extend(to_conjunction_list(expression_tree.children[1]))
    else:
        expression_list.append(expression_tree)
    return expression_list

def extract_pushdown_predicate(predicate: AbstractExpression, column_alias: str) -> Tuple[AbstractExpression, AbstractExpression]:
    """Decompose the predicate into pushdown predicate and remaining predicate

    Args:
        predicate (AbstractExpression): predicate that needs to be decomposed
        column (str): column_alias to extract predicate
    Returns:
        Tuple[AbstractExpression, AbstractExpression]: (pushdown predicate,
        remaining predicate)
    """
    if predicate is None:
        return (None, None)
    if contains_single_column(predicate, column_alias):
        if is_simple_predicate(predicate):
            return (predicate, None)
    pushdown_preds = []
    rem_pred = []
    pred_list = to_conjunction_list(predicate)
    for pred in pred_list:
        if contains_single_column(pred, column_alias) and is_simple_predicate(pred):
            pushdown_preds.append(pred)
        else:
            rem_pred.append(pred)
    return (conjunction_list_to_expression_tree(pushdown_preds), conjunction_list_to_expression_tree(rem_pred))

def extract_pushdown_predicate_for_alias(predicate: AbstractExpression, aliases: List[Alias]):
    """Extract predicate that can be pushed down based on the input aliases.

    Atomic predicates on the table columns that are the subset of the input aliases are
    considered as candidates for pushdown.

    Args:
        predicate (AbstractExpression): input predicate
        aliases (List[str]): aliases for which predicate can be pushed
    """
    if predicate is None:
        return (None, None)
    pred_list = to_conjunction_list(predicate)
    pushdown_preds = []
    rem_pred = []
    aliases = [alias.alias_name for alias in aliases]
    for pred in pred_list:
        column_aliases = get_columns_in_predicate(pred)
        table_aliases = set([col.split('.')[0] for col in column_aliases])
        if table_aliases.issubset(set(aliases)):
            pushdown_preds.append(pred)
        else:
            rem_pred.append(pred)
    return (conjunction_list_to_expression_tree(pushdown_preds), conjunction_list_to_expression_tree(rem_pred))

def get_columns_in_predicate(predicate: AbstractExpression) -> Set[str]:
    """Get columns accessed in the predicate

    Args:
        predicate (AbstractExpression): input predicate

    Returns:
        Set[str]: list of column aliases used in the predicate
    """
    if isinstance(predicate, TupleValueExpression):
        return set([predicate.col_alias])
    cols = set()
    for child in predicate.children:
        child_cols = get_columns_in_predicate(child)
        if len(child_cols):
            cols.update(child_cols)
    return cols

def enable_cache_init(context: 'OptimizerContext', func_expr: FunctionExpression) -> FunctionExpressionCache:
    optimized_key = optimize_cache_key(context, func_expr)
    if optimized_key == func_expr.children:
        optimized_key = [None]
    catalog = context.db.catalog()
    name = func_expr.signature()
    cache_entry = catalog.get_function_cache_catalog_entry_by_name(name)
    if not cache_entry:
        cache_entry = catalog.insert_function_cache_catalog_entry(func_expr)
    cache = FunctionExpressionCache(key=tuple(optimized_key), store=DiskKVCache(cache_entry.cache_path))
    return cache

def optimize_cache_key(context: 'OptimizerContext', expr: FunctionExpression):
    """Optimize the cache key

    It tries to reduce the caching overhead by replacing the caching key with
    logically equivalent key. For instance, frame data can be replaced with frame id.

    Args:
        expr (FunctionExpression): expression to optimize the caching key for.

    Example:
        Yolo(data) -> return id

    Todo: Optimize complex expression
        FaceDet(Crop(data, bbox)) -> return

    """
    keys = expr.children
    optimize_key_mapping_f = {TupleValueExpression: optimize_cache_key_for_tuple_value_expression, ConstantValueExpression: optimize_cache_key_for_constant_value_expression}
    optimized_keys = []
    for key in keys:
        if type(key) not in optimize_key_mapping_f:
            raise RuntimeError(f'Optimize cache key of {type(key)} is not implemented')
        optimized_keys += optimize_key_mapping_f[type(key)](context, key)
    return optimized_keys

def enable_cache(context: 'OptimizerContext', func_expr: FunctionExpression) -> FunctionExpression:
    """Enables cache for a function expression.

    The cache key is optimized by replacing it with logical equivalent expressions.
    A cache entry is inserted in the catalog corresponding to the expression.

    Args:
        context (OptimizerContext): associated optimizer context
        func_expr (FunctionExpression): The function expression to enable cache for.

    Returns:
        FunctionExpression: The function expression with cache enabled.
    """
    cache = enable_cache_init(context, func_expr)
    return func_expr.copy().enable_cache(cache)

def enable_cache_on_expression_tree(context: 'OptimizerContext', expr_tree: AbstractExpression):
    func_exprs = list(expr_tree.find_all(FunctionExpression))
    func_exprs = list(filter(lambda expr: check_expr_validity_for_cache(expr), func_exprs))
    for expr in func_exprs:
        cache = enable_cache_init(context, expr)
        expr.enable_cache(cache)

class EmbedFilterIntoGet(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALFILTER)
        pattern.append_child(Pattern(OperatorType.LOGICALGET))
        super().__init__(RuleType.EMBED_FILTER_INTO_GET, pattern)

    def promise(self):
        return Promise.EMBED_FILTER_INTO_GET

    def check(self, before: LogicalFilter, context: OptimizerContext):
        predicate = before.predicate
        lget: LogicalGet = before.children[0]
        if predicate and is_video_table(lget.table_obj):
            video_alias = lget.video.alias
            col_alias = f'{video_alias}.id'
            pushdown_pred, _ = extract_pushdown_predicate(predicate, col_alias)
            if pushdown_pred:
                return True
        return False

    def apply(self, before: LogicalFilter, context: OptimizerContext):
        predicate = before.predicate
        lget = before.children[0]
        video_alias = lget.video.alias
        col_alias = f'{video_alias}.id'
        pushdown_pred, unsupported_pred = extract_pushdown_predicate(predicate, col_alias)
        if pushdown_pred:
            new_get_opr = LogicalGet(lget.video, lget.table_obj, alias=lget.alias, predicate=pushdown_pred, target_list=lget.target_list, sampling_rate=lget.sampling_rate, sampling_type=lget.sampling_type, children=lget.children)
            if unsupported_pred:
                unsupported_opr = LogicalFilter(unsupported_pred)
                unsupported_opr.append_child(new_get_opr)
                new_get_opr = unsupported_opr
            yield new_get_opr
        else:
            yield before

class EmbedSampleIntoGet(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALSAMPLE)
        pattern.append_child(Pattern(OperatorType.LOGICALGET))
        super().__init__(RuleType.EMBED_SAMPLE_INTO_GET, pattern)

    def promise(self):
        return Promise.EMBED_SAMPLE_INTO_GET

    def check(self, before: LogicalSample, context: OptimizerContext):
        lget: LogicalGet = before.children[0]
        if lget.table_obj.table_type == TableType.VIDEO_DATA:
            return True
        return False

    def apply(self, before: LogicalSample, context: OptimizerContext):
        sample_freq = before.sample_freq.value
        sample_type = before.sample_type.value.value if before.sample_type else None
        lget: LogicalGet = before.children[0]
        new_get_opr = LogicalGet(lget.video, lget.table_obj, alias=lget.alias, predicate=lget.predicate, target_list=lget.target_list, sampling_rate=sample_freq, sampling_type=sample_type, children=lget.children)
        yield new_get_opr

class CacheFunctionExpressionInProject(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALPROJECT)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.CACHE_FUNCTION_EXPRESISON_IN_PROJECT, pattern)

    def promise(self):
        return Promise.CACHE_FUNCTION_EXPRESISON_IN_PROJECT

    def check(self, before: LogicalProject, context: OptimizerContext):
        valid_exprs = []
        for expr in before.target_list:
            if isinstance(expr, FunctionExpression):
                func_exprs = list(expr.find_all(FunctionExpression))
                valid_exprs.extend(filter(lambda expr: check_expr_validity_for_cache(expr), func_exprs))
        if len(valid_exprs) > 0:
            return True
        return False

    def apply(self, before: LogicalProject, context: OptimizerContext):
        new_target_list = [expr.copy() for expr in before.target_list]
        for expr in new_target_list:
            enable_cache_on_expression_tree(context, expr)
        after = LogicalProject(target_list=new_target_list, children=before.children)
        yield after

class CacheFunctionExpressionInFilter(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALFILTER)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.CACHE_FUNCTION_EXPRESISON_IN_FILTER, pattern)

    def promise(self):
        return Promise.CACHE_FUNCTION_EXPRESISON_IN_FILTER

    def check(self, before: LogicalFilter, context: OptimizerContext):
        func_exprs = list(before.predicate.find_all(FunctionExpression))
        valid_exprs = list(filter(lambda expr: check_expr_validity_for_cache(expr), func_exprs))
        if len(valid_exprs) > 0:
            return True
        return False

    def apply(self, before: LogicalFilter, context: OptimizerContext):
        after_predicate = before.predicate.copy()
        enable_cache_on_expression_tree(context, after_predicate)
        after_operator = LogicalFilter(predicate=after_predicate, children=before.children)
        yield after_operator

class CacheFunctionExpressionInApply(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICAL_APPLY_AND_MERGE)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.CACHE_FUNCTION_EXPRESISON_IN_APPLY, pattern)

    def promise(self):
        return Promise.CACHE_FUNCTION_EXPRESISON_IN_APPLY

    def check(self, before: LogicalApplyAndMerge, context: OptimizerContext):
        expr = before.func_expr
        if expr.has_cache() or expr.name not in CACHEABLE_FUNCTIONS:
            return False
        if len(expr.children) > 1 or not isinstance(expr.children[0], TupleValueExpression):
            return False
        return True

    def apply(self, before: LogicalApplyAndMerge, context: OptimizerContext):
        new_func_expr = enable_cache(context, before.func_expr)
        after = LogicalApplyAndMerge(func_expr=new_func_expr, alias=before.alias, do_unnest=before.do_unnest)
        after.append_child(before.children[0])
        yield after

class PushDownFilterThroughJoin(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALFILTER)
        pattern_join = Pattern(OperatorType.LOGICALJOIN)
        pattern_join.append_child(Pattern(OperatorType.DUMMY))
        pattern_join.append_child(Pattern(OperatorType.DUMMY))
        pattern.append_child(pattern_join)
        super().__init__(RuleType.PUSHDOWN_FILTER_THROUGH_JOIN, pattern)

    def promise(self):
        return Promise.PUSHDOWN_FILTER_THROUGH_JOIN

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalFilter, context: OptimizerContext):
        predicate = before.predicate
        join: LogicalJoin = before.children[0]
        left: Dummy = join.children[0]
        right: Dummy = join.children[1]
        new_join_node = LogicalJoin(join.join_type, join.join_predicate, join.left_keys, join.right_keys)
        left_group_aliases = context.memo.get_group_by_id(left.group_id).aliases
        right_group_aliases = context.memo.get_group_by_id(right.group_id).aliases
        left_pushdown_pred, rem_pred = extract_pushdown_predicate_for_alias(predicate, left_group_aliases)
        right_pushdown_pred, rem_pred = extract_pushdown_predicate_for_alias(rem_pred, right_group_aliases)
        if left_pushdown_pred:
            left_filter = LogicalFilter(predicate=left_pushdown_pred)
            left_filter.append_child(left)
            new_join_node.append_child(left_filter)
        else:
            new_join_node.append_child(left)
        if right_pushdown_pred:
            right_filter = LogicalFilter(predicate=right_pushdown_pred)
            right_filter.append_child(right)
            new_join_node.append_child(right_filter)
        else:
            new_join_node.append_child(right)
        if rem_pred:
            new_join_node._join_predicate = conjunction_list_to_expression_tree([rem_pred, new_join_node.join_predicate])
        yield new_join_node

class XformLateralJoinToLinearFlow(Rule):
    """If the inner node of a lateral join is a function-valued expression, we
    eliminate the join node and make the inner node the parent of the outer node. This
    produces a linear data flow path. Because this scenario is common in our system,
    we chose to explicitly convert it to a linear flow, which simplifies the
    implementation of other optimizations such as function reuse and parallelized plans by
    removing the join."""

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALJOIN)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        pattern.append_child(Pattern(OperatorType.LOGICALFUNCTIONSCAN))
        super().__init__(RuleType.XFORM_LATERAL_JOIN_TO_LINEAR_FLOW, pattern)

    def promise(self):
        return Promise.XFORM_LATERAL_JOIN_TO_LINEAR_FLOW

    def check(self, before: LogicalJoin, context: OptimizerContext):
        if before.join_type == JoinType.LATERAL_JOIN:
            if before.join_predicate is None and (not before.join_project):
                return True
        return False

    def apply(self, before: LogicalJoin, context: OptimizerContext):
        A: Dummy = before.children[0]
        logical_func_scan: LogicalFunctionScan = before.children[1]
        logical_apply_merge = LogicalApplyAndMerge(logical_func_scan.func_expr, logical_func_scan.alias, logical_func_scan.do_unnest)
        logical_apply_merge.append_child(A)
        yield logical_apply_merge

class PushDownFilterThroughApplyAndMerge(Rule):
    """If it is feasible to partially or fully push the predicate contained within the
    logical filter through the ApplyAndMerge operator, we should do so. This is often
    beneficial, for instance, in order to prevent decoding additional frames beyond
    those that satisfy the predicate.
    Eg:

    Filter(id < 10 and func.label = 'car')           Filter(func.label = 'car')
            |                                                   |
        ApplyAndMerge(func)                  ->          ApplyAndMerge(func)
            |                                                   |
            A                                            Filter(id < 10)
                                                                |
                                                                A

    """

    def __init__(self):
        appply_merge_pattern = Pattern(OperatorType.LOGICAL_APPLY_AND_MERGE)
        appply_merge_pattern.append_child(Pattern(OperatorType.DUMMY))
        pattern = Pattern(OperatorType.LOGICALFILTER)
        pattern.append_child(appply_merge_pattern)
        super().__init__(RuleType.PUSHDOWN_FILTER_THROUGH_APPLY_AND_MERGE, pattern)

    def promise(self):
        return Promise.PUSHDOWN_FILTER_THROUGH_APPLY_AND_MERGE

    def check(self, before: LogicalFilter, context: OptimizerContext):
        return True

    def apply(self, before: LogicalFilter, context: OptimizerContext):
        A: Dummy = before.children[0].children[0]
        apply_and_merge: LogicalApplyAndMerge = before.children[0]
        aliases = context.memo.get_group_by_id(A.group_id).aliases
        predicate = before.predicate
        pushdown_pred, rem_pred = extract_pushdown_predicate_for_alias(predicate, aliases)
        if pushdown_pred is None:
            return
        if pushdown_pred:
            pushdown_filter = LogicalFilter(predicate=pushdown_pred)
            pushdown_filter.append_child(A)
            apply_and_merge.children = [pushdown_filter]
        root_node = apply_and_merge
        if rem_pred:
            root_node = LogicalFilter(predicate=rem_pred)
            root_node.append_child(apply_and_merge)
        yield root_node

class XformExtractObjectToLinearFlow(Rule):
    """If the inner node of a lateral join is a Extract_Object function-valued
    expression, we eliminate the join node and make the inner node the parent of the
    outer node. This produces a linear data flow path.
    TODO: We need to add a sorting operation after detector to ensure we always provide tracker data in order.
    """

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALJOIN)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        pattern.append_child(Pattern(OperatorType.LOGICAL_EXTRACT_OBJECT))
        super().__init__(RuleType.XFORM_EXTRACT_OBJECT_TO_LINEAR_FLOW, pattern)

    def promise(self):
        return Promise.XFORM_EXTRACT_OBJECT_TO_LINEAR_FLOW

    def check(self, before: LogicalJoin, context: OptimizerContext):
        if before.join_type == JoinType.LATERAL_JOIN:
            return True
        return False

    def apply(self, before: LogicalJoin, context: OptimizerContext):
        A: Dummy = before.children[0]
        logical_extract_obj: LogicalExtractObject = before.children[1]
        detector = LogicalApplyAndMerge(logical_extract_obj.detector, alias=logical_extract_obj.detector.alias)
        tracker = LogicalApplyAndMerge(logical_extract_obj.tracker, alias=logical_extract_obj.alias, do_unnest=logical_extract_obj.do_unnest)
        detector.append_child(A)
        tracker.append_child(detector)
        yield tracker

class CombineSimilarityOrderByAndLimitToVectorIndexScan(Rule):
    """
    This rule currently rewrites Order By + Limit to a vector index scan.
    Because vector index only works for similarity search, the rule will
    only be applied when the Order By is on Similarity expression. For
    simplicity, we also only enable this rule when the Similarity expression
    applies to the full table. Predicated query will yield incorrect results
    if we use an index scan.

    Limit(10)
        |
    OrderBy(func)        ->        IndexScan(10)
        |                               |
        A                               A
    """

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALLIMIT)
        orderby_pattern = Pattern(OperatorType.LOGICALORDERBY)
        orderby_pattern.append_child(Pattern(OperatorType.DUMMY))
        pattern.append_child(orderby_pattern)
        super().__init__(RuleType.COMBINE_SIMILARITY_ORDERBY_AND_LIMIT_TO_VECTOR_INDEX_SCAN, pattern)
        self._index_catalog_entry = None
        self._query_func_expr = None

    def promise(self):
        return Promise.COMBINE_SIMILARITY_ORDERBY_AND_LIMIT_TO_VECTOR_INDEX_SCAN

    def check(self, before: LogicalLimit, context: OptimizerContext):
        return True

    def apply(self, before: LogicalLimit, context: OptimizerContext):
        catalog_manager = context.db.catalog
        limit_node = before
        orderby_node = before.children[0]
        sub_tree_root = orderby_node.children[0]

        def _exists_predicate(opr):
            if isinstance(opr, LogicalGet):
                return opr.predicate is not None
            return True
        if _exists_predicate(sub_tree_root.opr):
            return
        func_orderby_expr = None
        for column, sort_type in orderby_node.orderby_list:
            if isinstance(column, FunctionExpression) and sort_type == ParserOrderBySortType.ASC:
                func_orderby_expr = column
        if not func_orderby_expr or func_orderby_expr.name != 'Similarity':
            return
        tb_catalog_entry = list(sub_tree_root.opr.find_all(LogicalGet))[0].table_obj
        db_catalog_entry = catalog_manager().get_database_catalog_entry(tb_catalog_entry.database_name)
        is_postgres_data_source = db_catalog_entry is not None and db_catalog_entry.engine == 'postgres'
        query_func_expr, base_func_expr = func_orderby_expr.children
        tv_expr = base_func_expr
        while not isinstance(tv_expr, TupleValueExpression):
            tv_expr = tv_expr.children[0]
        column_catalog_entry = tv_expr.col_object
        if not is_postgres_data_source:
            function_signature = None if isinstance(base_func_expr, TupleValueExpression) else base_func_expr.signature()
            index_catalog_entry = catalog_manager().get_index_catalog_entry_by_column_and_function_signature(column_catalog_entry, function_signature)
            if not index_catalog_entry:
                return
        else:
            index_catalog_entry = IndexCatalogEntry(name='', save_file_path='', type=VectorStoreType.PGVECTOR, feat_column=column_catalog_entry)
        vector_index_scan_node = LogicalVectorIndexScan(index_catalog_entry, limit_node.limit_count, query_func_expr)
        for child in orderby_node.children:
            vector_index_scan_node.append_child(child)
        yield vector_index_scan_node

class LogicalInnerJoinCommutativity(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALJOIN)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_INNER_JOIN_COMMUTATIVITY, pattern)

    def promise(self):
        return Promise.LOGICAL_INNER_JOIN_COMMUTATIVITY

    def check(self, before: LogicalJoin, context: OptimizerContext):
        return before.join_type == JoinType.INNER_JOIN

    def apply(self, before: LogicalJoin, context: OptimizerContext):
        new_join = LogicalJoin(before.join_type, before.join_predicate)
        new_join.append_child(before.rhs())
        new_join.append_child(before.lhs())
        yield new_join

class ReorderPredicates(Rule):
    """
    The current implementation orders conjuncts based on their individual cost.
    The optimization for OR clauses has `not` been implemented yet. Additionally, we do
    not optimize predicates that are not user-defined functions since we assume that
    they will likely be pushed to the underlying relational database, which will handle
    the optimization process.
    """

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALFILTER)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.REORDER_PREDICATES, pattern)

    def promise(self):
        return Promise.REORDER_PREDICATES

    def check(self, before: LogicalFilter, context: OptimizerContext):
        return len(list(before.predicate.find_all(FunctionExpression))) > 0

    def apply(self, before: LogicalFilter, context: OptimizerContext):
        conjuncts = to_conjunction_list(before.predicate)
        contains_func_exprs = []
        simple_exprs = []
        for conjunct in conjuncts:
            if list(conjunct.find_all(FunctionExpression)):
                contains_func_exprs.append(conjunct)
            else:
                simple_exprs.append(conjunct)
        function_expr_cost_tuples = [(expr, get_expression_execution_cost(context, expr)) for expr in contains_func_exprs]
        function_expr_cost_tuples = sorted(function_expr_cost_tuples, key=lambda x: x[1])
        ordered_conjuncts = simple_exprs + [expr for expr, _ in function_expr_cost_tuples]
        if ordered_conjuncts != conjuncts:
            reordered_predicate = conjunction_list_to_expression_tree(ordered_conjuncts)
            reordered_filter_node = LogicalFilter(predicate=reordered_predicate)
            reordered_filter_node.append_child(before.children[0])
            yield reordered_filter_node

def get_expression_execution_cost(context: 'OptimizerContext', expr: AbstractExpression) -> float:
    """
    This function computes the estimated cost of executing the given abstract expression
    based on the statistics in the catalog. The function assumes that all the
    expression, except for the FunctionExpression, have a cost of zero.
    For FunctionExpression, it checks the catalog for relevant statistics; if none are
    available, it uses a default cost of DEFAULT_FUNCTION_EXPRESSION_COST.

    Args:
        context (OptimizerContext): the associated optimizer context
        expr (AbstractExpression): The AbstractExpression object whose cost
        needs to be computed.

    Returns:
        float: The estimated cost of executing the function expression.
    """
    total_cost = 0
    for child_expr in expr.find_all(FunctionExpression):
        cost_entry = context.db.catalog().get_function_cost_catalog_entry(child_expr.name)
        if cost_entry:
            total_cost += cost_entry.cost
        else:
            total_cost += DEFAULT_FUNCTION_EXPRESSION_COST
    return total_cost

class LogicalJoinToPhysicalHashJoin(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALJOIN)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_JOIN_TO_PHYSICAL_HASH_JOIN, pattern)

    def promise(self):
        return Promise.LOGICAL_JOIN_TO_PHYSICAL_HASH_JOIN

    def check(self, before: Operator, context: OptimizerContext):
        """
        We don't want to apply this rule to the join when FuzzDistance
        is being used, which implies that the join is a FuzzyJoin
        """
        if before.join_predicate is None:
            return False
        j_child: FunctionExpression = before.join_predicate.children[0]
        if isinstance(j_child, FunctionExpression):
            if j_child.name.startswith('FuzzDistance'):
                return before.join_type == JoinType.INNER_JOIN and (not j_child or not j_child.name.startswith('FuzzDistance'))
        else:
            return before.join_type == JoinType.INNER_JOIN

    def apply(self, join_node: LogicalJoin, context: OptimizerContext):
        a: Dummy = join_node.lhs()
        b: Dummy = join_node.rhs()
        a_table_aliases = context.memo.get_group_by_id(a.group_id).aliases
        b_table_aliases = context.memo.get_group_by_id(b.group_id).aliases
        join_predicates = join_node.join_predicate
        a_join_keys, b_join_keys = extract_equi_join_keys(join_predicates, a_table_aliases, b_table_aliases)
        build_plan = HashJoinBuildPlan(join_node.join_type, a_join_keys)
        build_plan.append_child(a)
        probe_side = HashJoinProbePlan(join_node.join_type, b_join_keys, join_predicates, join_node.join_project)
        probe_side.append_child(build_plan)
        probe_side.append_child(b)
        yield probe_side

def _has_simple_expressions(expr):
    simple = type(expr) in simple_expressions
    for child in expr.children:
        simple = simple and _has_simple_expressions(child)
    return simple

class IndexCatalog(BaseModel):
    """The `IndexCatalogEntry` catalog stores information about all the indexes in the system.
    `_row_id:` an autogenerated unique identifier.
    `_name:` the name of the index.
    `_save_file_path:` the path to the index file on disk
    `_type:` the type of the index (refer to `VectorStoreType`)
    `_feat_column_id:` the `_row_id` of the `ColumnCatalog` entry for the column on which the index is built.
    `_function_signature:` if the index is created by running function expression on input column, this will store
                      the function signature of the used function. Otherwise, this field is None.
    `_index_def:` the original SQL statement that is used to create this index. We record this to rerun create index
                on updated table.
    """
    __tablename__ = 'index_catalog'
    _name = Column('name', String(100), unique=True)
    _save_file_path = Column('save_file_path', String(128))
    _type = Column('type', Enum(VectorStoreType), default=Enum)
    _feat_column_id = Column('column_id', Integer, ForeignKey('column_catalog._row_id', ondelete='CASCADE'))
    _function_signature = Column('function', String, default=None)
    _index_def = Column('index_def', String, default=None)
    _feat_column = relationship('ColumnCatalog', back_populates='_index_column')

    def __init__(self, name: str, save_file_path: str, type: VectorStoreType, feat_column_id: int=None, function_signature: str=None, index_def: str=None):
        self._name = name
        self._save_file_path = save_file_path
        self._type = type
        self._feat_column_id = feat_column_id
        self._function_signature = function_signature
        self._index_def = index_def

    def as_dataclass(self) -> 'IndexCatalogEntry':
        feat_column = self._feat_column.as_dataclass() if self._feat_column else None
        return IndexCatalogEntry(row_id=self._row_id, name=self._name, save_file_path=self._save_file_path, type=self._type, feat_column_id=self._feat_column_id, function_signature=self._function_signature, index_def=self._index_def, feat_column=feat_column)

