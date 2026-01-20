# Cluster 37

@pytest.mark.notparallel
class PlanNodeTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_create_plan(self):
        dummy_info = TableInfo('dummy')
        columns = [ColumnCatalogEntry('id', ColumnType.INTEGER), ColumnCatalogEntry('name', ColumnType.TEXT, array_dimensions=[50])]
        dummy_plan_node = CreatePlan(dummy_info, columns, False)
        self.assertEqual(dummy_plan_node.opr_type, PlanOprType.CREATE)
        self.assertEqual(dummy_plan_node.if_not_exists, False)
        self.assertEqual(dummy_plan_node.table_info.table_name, 'dummy')
        self.assertEqual(dummy_plan_node.column_list[0].name, 'id')
        self.assertEqual(dummy_plan_node.column_list[1].name, 'name')

    def test_rename_plan(self):
        dummy_info = TableInfo('old')
        dummy_old = TableRef(dummy_info)
        dummy_new = TableInfo('new')
        dummy_plan_node = RenamePlan(dummy_old, dummy_new)
        self.assertEqual(dummy_plan_node.opr_type, PlanOprType.RENAME)
        self.assertEqual(dummy_plan_node.old_table.table.table_name, 'old')
        self.assertEqual(dummy_plan_node.new_name.table_name, 'new')

    def test_insert_plan(self):
        video_id = 0
        column_ids = [0, 1]
        expression = type('AbstractExpression', (), {'evaluate': lambda: 1})
        values = [expression, expression]
        dummy_plan_node = InsertPlan(video_id, column_ids, values)
        self.assertEqual(dummy_plan_node.opr_type, PlanOprType.INSERT)

    def test_create_function_plan(self):
        function_name = 'function'
        or_replace = False
        if_not_exists = True
        functionIO = 'functionIO'
        inputs = [functionIO, functionIO]
        outputs = [functionIO]
        impl_path = 'test'
        ty = 'classification'
        node = CreateFunctionPlan(function_name, or_replace, if_not_exists, inputs, outputs, impl_path, ty)
        self.assertEqual(node.opr_type, PlanOprType.CREATE_FUNCTION)
        self.assertEqual(node.or_replace, or_replace)
        self.assertEqual(node.if_not_exists, if_not_exists)
        self.assertEqual(node.inputs, [functionIO, functionIO])
        self.assertEqual(node.outputs, [functionIO])
        self.assertEqual(node.impl_path, impl_path)
        self.assertEqual(node.function_type, ty)

    def test_drop_object_plan(self):
        object_type = ObjectType.TABLE
        function_name = 'function'
        if_exists = True
        node = DropObjectPlan(object_type, function_name, if_exists)
        self.assertEqual(node.opr_type, PlanOprType.DROP_OBJECT)
        self.assertEqual(node.if_exists, True)
        self.assertEqual(node.object_type, ObjectType.TABLE)

    def test_load_data_plan(self):
        table_info = 'info'
        file_path = 'test.mp4'
        file_format = FileFormatType.VIDEO
        file_options = {}
        file_options['file_format'] = file_format
        column_list = None
        batch_mem_size = 3000
        plan_str = 'LoadDataPlan(table_id={}, file_path={},             column_list={},             file_options={},             batch_mem_size={})'.format(table_info, file_path, column_list, file_options, batch_mem_size)
        plan = LoadDataPlan(table_info, file_path, column_list, file_options, batch_mem_size)
        self.assertEqual(plan.opr_type, PlanOprType.LOAD_DATA)
        self.assertEqual(plan.table_info, table_info)
        self.assertEqual(plan.file_path, file_path)
        self.assertEqual(plan.batch_mem_size, batch_mem_size)
        self.assertEqual(str(plan), plan_str)

    def test_union_plan(self):
        all = True
        plan = UnionPlan(all)
        self.assertEqual(plan.opr_type, PlanOprType.UNION)
        self.assertEqual(plan.all, all)

    def test_abstract_plan_str(self):
        derived_plan_classes = list(get_all_subclasses(AbstractPlan))
        for derived_plan_class in derived_plan_classes:
            sig = signature(derived_plan_class.__init__)
            params = sig.parameters
            plan_dict = {}
            if isabstract(derived_plan_class) is False:
                obj = get_mock_object(derived_plan_class, len(params))
                plan_dict[obj] = obj

class PlanExecutorTest(unittest.TestCase):

    def test_tree_structure_for_build_execution_tree(self):
        """
            Build an Abstract Plan with nodes:
         ß               root
                      /  |                      c1   c2 c3
                    /
                   c1_1
        """
        predicate = None
        root_abs_plan = SeqScanPlan(predicate=predicate, columns=[])
        child_1_abs_plan = SeqScanPlan(predicate=predicate, columns=[])
        child_2_abs_plan = SeqScanPlan(predicate=predicate, columns=[])
        child_3_abs_plan = SeqScanPlan(predicate=predicate, columns=[])
        child_1_1_abs_plan = SeqScanPlan(predicate=predicate, columns=[])
        root_abs_plan.append_child(child_1_abs_plan)
        root_abs_plan.append_child(child_2_abs_plan)
        root_abs_plan.append_child(child_3_abs_plan)
        child_1_abs_plan.append_child(child_1_1_abs_plan)
        'Build Execution Tree and check the nodes\n            are of the same type'
        root_abs_executor = PlanExecutor(MagicMock(), plan=root_abs_plan)._build_execution_tree(plan=root_abs_plan)
        self.assertEqual(root_abs_plan.opr_type, root_abs_executor._node.opr_type)
        for child_abs, child_exec in zip(root_abs_plan.children, root_abs_executor.children):
            self.assertEqual(child_abs.opr_type, child_exec._node.opr_type)
            for gc_abs, gc_exec in zip(child_abs.children, child_exec.children):
                self.assertEqual(gc_abs.opr_type, gc_exec._node.opr_type)

    def test_build_execution_tree_should_create_correct_exec_node(self):
        plan = SeqScanPlan(MagicMock(), [])
        executor = PlanExecutor(MagicMock(), plan)._build_execution_tree(plan)
        self.assertIsInstance(executor, SequentialScanExecutor)
        plan = PPScanPlan(MagicMock())
        executor = PlanExecutor(MagicMock(), plan)._build_execution_tree(plan)
        self.assertIsInstance(executor, PPExecutor)
        plan = CreatePlan(MagicMock(), [], False)
        executor = PlanExecutor(MagicMock(), plan)._build_execution_tree(plan)
        self.assertIsInstance(executor, CreateExecutor)
        plan = InsertPlan(0, [], [])
        executor = PlanExecutor(MagicMock(), plan)._build_execution_tree(plan)
        self.assertIsInstance(executor, InsertExecutor)
        plan = CreateFunctionPlan('test', False, [], [], MagicMock(), None)
        executor = PlanExecutor(MagicMock(), plan)._build_execution_tree(plan)
        self.assertIsInstance(executor, CreateFunctionExecutor)
        plan = DropObjectPlan(MagicMock(), 'test', False)
        executor = PlanExecutor(MagicMock(), plan)._build_execution_tree(plan)
        self.assertIsInstance(executor, DropObjectExecutor)
        plan = LoadDataPlan(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        executor = PlanExecutor(MagicMock(), plan)._build_execution_tree(plan)
        self.assertIsInstance(executor, LoadDataExecutor)

    @patch('evadb.executor.plan_executor.PlanExecutor._build_execution_tree')
    def test_execute_plan_for_seq_scan_plan(self, mock_build):
        batch_list = [Batch(pd.DataFrame([1])), Batch(pd.DataFrame([2])), Batch(pd.DataFrame([3]))]
        tree = MagicMock(node=SeqScanPlan(None, []))
        tree.exec.return_value = batch_list
        mock_build.return_value = tree
        actual = list(PlanExecutor(MagicMock(), None).execute_plan())
        mock_build.assert_called_once_with(None)
        tree.exec.assert_called_once()
        self.assertEqual(actual, batch_list)

    @patch('evadb.executor.plan_executor.PlanExecutor._build_execution_tree')
    def test_execute_plan_for_pp_scan_plan(self, mock_build):
        batch_list = [Batch(pd.DataFrame([1])), Batch(pd.DataFrame([2])), Batch(pd.DataFrame([3]))]
        tree = MagicMock(node=PPScanPlan(None))
        tree.exec.return_value = batch_list
        mock_build.return_value = tree
        actual = list(PlanExecutor(MagicMock(), None).execute_plan())
        mock_build.assert_called_once_with(None)
        tree.exec.assert_called_once()
        self.assertEqual(actual, batch_list)

    @patch('evadb.executor.plan_executor.PlanExecutor._build_execution_tree')
    def test_execute_plan_for_create_insert_load_upload_plans(self, mock_build):
        tree = MagicMock(node=CreatePlan(None, [], False))
        mock_build.return_value = tree
        actual = list(PlanExecutor(MagicMock(), None).execute_plan())
        tree.exec.assert_called_once()
        mock_build.assert_called_once_with(None)
        self.assertEqual(actual, [])
        mock_build.reset_mock()
        tree = MagicMock(node=InsertPlan(0, [], []))
        mock_build.return_value = tree
        actual = list(PlanExecutor(MagicMock(), None).execute_plan())
        tree.exec.assert_called_once()
        mock_build.assert_called_once_with(None)
        self.assertEqual(actual, [])
        mock_build.reset_mock()
        tree = MagicMock(node=CreateFunctionPlan(None, False, False, [], [], None))
        mock_build.return_value = tree
        actual = list(PlanExecutor(MagicMock(), None).execute_plan())
        tree.exec.assert_called_once()
        mock_build.assert_called_once_with(None)
        self.assertEqual(actual, [])
        mock_build.reset_mock()
        tree = MagicMock(node=LoadDataPlan(None, None, None, None, None))
        mock_build.return_value = tree
        actual = list(PlanExecutor(MagicMock(), None).execute_plan())
        tree.exec.assert_called_once()
        mock_build.assert_called_once_with(None)
        self.assertEqual(actual, [])

    @patch('evadb.executor.plan_executor.PlanExecutor._build_execution_tree')
    def test_execute_plan_for_rename_plans(self, mock_build):
        tree = MagicMock(node=RenamePlan(None, None))
        mock_build.return_value = tree
        actual = list(PlanExecutor(MagicMock(), None).execute_plan())
        tree.exec.assert_called_once()
        mock_build.assert_called_once_with(None)
        self.assertEqual(actual, [])

    @patch('evadb.executor.plan_executor.PlanExecutor._build_execution_tree')
    def test_execute_plan_for_drop_plans(self, mock_build):
        tree = MagicMock(node=DropObjectPlan(None, None, None))
        mock_build.return_value = tree
        actual = list(PlanExecutor(MagicMock(), None).execute_plan())
        tree.exec.assert_called_once()
        mock_build.assert_called_once_with(None)
        self.assertEqual(actual, [])

    @unittest.skip('disk_based_storage_deprecated')
    @patch('evadb.executor.disk_based_storage_executor.Loader')
    def test_should_return_the_new_path_after_execution(self, mock_class):
        class_instance = mock_class.return_value
        dummy_expr = type('dummy_expr', (), {'evaluate': lambda x=None: [True, False, True]})
        video = TableCatalogEntry('dataset', 'dummy.avi', table_type=TableType.VIDEO)
        batch_1 = Batch(pd.DataFrame({'data': [1, 2, 3]}))
        batch_2 = Batch(pd.DataFrame({'data': [4, 5, 6]}))
        class_instance.load.return_value = map(lambda x: x, [batch_1, batch_2])
        storage_plan = StoragePlan(video, batch_mem_size=3000)
        seq_scan = SeqScanPlan(predicate=dummy_expr, column_ids=[])
        seq_scan.append_child(storage_plan)
        executor = PlanExecutor(seq_scan)
        actual = executor.execute_plan()
        expected = batch_1[::2] + batch_2[::2]
        mock_class.assert_called_once()
        self.assertEqual(expected, actual)

class LogicalCreateToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALCREATE)
        super().__init__(RuleType.LOGICAL_CREATE_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_CREATE_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalCreate, context: OptimizerContext):
        after = CreatePlan(before.video, before.column_list, before.if_not_exists)
        yield after

class LogicalRenameToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALRENAME)
        super().__init__(RuleType.LOGICAL_RENAME_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_RENAME_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalRename, context: OptimizerContext):
        after = RenamePlan(before.old_table_ref, before.new_name)
        yield after

class LogicalCreateFunctionToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALCREATEFUNCTION)
        super().__init__(RuleType.LOGICAL_CREATE_FUNCTION_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_CREATE_FUNCTION_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalCreateFunction, context: OptimizerContext):
        after = CreateFunctionPlan(before.name, before.or_replace, before.if_not_exists, before.inputs, before.outputs, before.impl_path, before.function_type, before.metadata)
        yield after

class LogicalCreateFunctionFromSelectToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALCREATEFUNCTION)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_CREATE_FUNCTION_FROM_SELECT_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_CREATE_FUNCTION_FROM_SELECT_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalCreateFunction, context: OptimizerContext):
        after = CreateFunctionPlan(before.name, before.or_replace, before.if_not_exists, before.inputs, before.outputs, before.impl_path, before.function_type, before.metadata)
        for child in before.children:
            after.append_child(child)
        yield after

class LogicalCreateIndexToVectorIndex(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALCREATEINDEX)
        super().__init__(RuleType.LOGICAL_CREATE_INDEX_TO_VECTOR_INDEX, pattern)

    def promise(self):
        return Promise.LOGICAL_CREATE_INDEX_TO_VECTOR_INDEX

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalCreateIndex, context: OptimizerContext):
        after = CreateIndexPlan(before.name, before.if_not_exists, before.table_ref, before.col_list, before.vector_store_type, before.project_expr_list, before.index_def)
        child = SeqScanPlan(None, before.project_expr_list, before.table_ref.alias)
        batch_mem_size = context.db.catalog().get_configuration_catalog_value('batch_mem_size')
        child.append_child(StoragePlan(before.table_ref.table.table_obj, before.table_ref, batch_mem_size=batch_mem_size))
        after.append_child(child)
        yield after

class LogicalInsertToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALINSERT)
        super().__init__(RuleType.LOGICAL_INSERT_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_INSERT_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalInsert, context: OptimizerContext):
        after = InsertPlan(before.table, before.column_list, before.value_list)
        yield after

class LogicalLoadToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALLOADDATA)
        super().__init__(RuleType.LOGICAL_LOAD_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_LOAD_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalLoadData, context: OptimizerContext):
        after = LoadDataPlan(before.table_info, before.path, before.column_list, before.file_options)
        yield after

class LogicalGetToSeqScan(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALGET)
        super().__init__(RuleType.LOGICAL_GET_TO_SEQSCAN, pattern)

    def promise(self):
        return Promise.LOGICAL_GET_TO_SEQSCAN

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalGet, context: OptimizerContext):
        after = SeqScanPlan(None, before.target_list, before.alias)
        batch_mem_size = context.db.catalog().get_configuration_catalog_value('batch_mem_size')
        after.append_child(StoragePlan(before.table_obj, before.video, predicate=before.predicate, sampling_rate=before.sampling_rate, sampling_type=before.sampling_type, chunk_params=before.chunk_params, batch_mem_size=batch_mem_size))
        yield after

class LogicalDerivedGetToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALQUERYDERIVEDGET)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_DERIVED_GET_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_DERIVED_GET_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalQueryDerivedGet, context: OptimizerContext):
        after = SeqScanPlan(before.predicate, before.target_list, before.alias)
        after.append_child(before.children[0])
        yield after

