# Cluster 62

class ProjectExecutorTest(unittest.TestCase):

    def test_should_return_expr_without_table_source(self):
        constant = Batch(pd.DataFrame([1]))
        expression = [type('AbstractExpression', (), {'evaluate': lambda x: constant, 'find_all': lambda expr: []})]
        plan = type('ProjectPlan', (), {'predicate': None, 'target_list': expression})
        proj_executor = ProjectExecutor(MagicMock(), plan)
        actual = list(proj_executor.exec())[0]
        self.assertEqual(constant, actual)

class CreateFunctionExecutorTest(unittest.TestCase):

    @patch('evadb.executor.create_function_executor.load_function_class_from_file')
    def test_should_create_function(self, load_function_class_from_file_mock):
        catalog_instance = MagicMock()
        catalog_instance().get_function_catalog_entry_by_name.return_value = None
        catalog_instance().insert_function_catalog_entry.return_value = 'function'
        impl_path = MagicMock()
        abs_path = impl_path.absolute.return_value = MagicMock()
        abs_path.as_posix.return_value = 'test.py'
        load_function_class_from_file_mock.return_value.return_value = 'mock_class'
        plan = type('CreateFunctionPlan', (), {'name': 'function', 'if_not_exists': False, 'inputs': ['inp'], 'outputs': ['out'], 'impl_path': impl_path, 'function_type': 'classification', 'metadata': {'key1': 'value1', 'key2': 'value2'}})
        evadb = MagicMock
        evadb.catalog = catalog_instance
        evadb.config = MagicMock()
        create_function_executor = CreateFunctionExecutor(evadb, plan)
        next(create_function_executor.exec())
        catalog_instance().insert_function_catalog_entry.assert_called_with('function', 'test.py', 'classification', ['inp', 'out'], {'key1': 'value1', 'key2': 'value2'})

    def test_should_raise_or_replace_if_not_exists(self):
        plan = type('CreateFunctionPlan', (), {'name': 'function', 'or_replace': True, 'if_not_exists': True})
        evadb = MagicMock()
        create_function_executor = CreateFunctionExecutor(evadb, plan)
        with self.assertRaises(AssertionError) as cm:
            next(create_function_executor.exec())
        self.assertEqual(str(cm.exception), 'OR REPLACE and IF NOT EXISTS can not be both set for CREATE FUNCTION.')

    @patch('evadb.executor.create_function_executor.load_function_class_from_file')
    def test_should_skip_if_not_exists(self, load_function_class_from_file_mock):
        catalog_instance = MagicMock()
        catalog_instance().get_function_catalog_entry_by_name.return_value = True
        catalog_instance().insert_function_catalog_entry.return_value = 'function'
        impl_path = MagicMock()
        abs_path = impl_path.absolute.return_value = MagicMock()
        abs_path.as_posix.return_value = 'test.py'
        load_function_class_from_file_mock.return_value.return_value = 'mock_class'
        plan = type('CreateFunctionPlan', (), {'name': 'function', 'or_replace': False, 'if_not_exists': True, 'inputs': ['inp'], 'outputs': ['out'], 'impl_path': impl_path, 'function_type': 'classification', 'metadata': {'key1': 'value1', 'key2': 'value2'}})
        evadb = MagicMock()
        evadb.catalog = catalog_instance
        evadb.config = MagicMock()
        create_function_executor = CreateFunctionExecutor(evadb, plan)
        actual_batch = next(create_function_executor.exec())
        catalog_instance().insert_function_catalog_entry.assert_not_called()
        self.assertEqual(actual_batch.frames[0][0], 'Function function already exists, nothing added.')

    @patch('evadb.executor.create_function_executor.load_function_class_from_file')
    def test_should_overwrite_or_replace(self, load_function_class_from_file_mock):
        catalog_instance = MagicMock()
        catalog_instance().get_function_catalog_entry_by_name.return_value = False
        catalog_instance().insert_function_catalog_entry.return_value = 'function'
        impl_path = MagicMock()
        abs_path = impl_path.absolute.return_value = MagicMock()
        abs_path.as_posix.return_value = 'test.py'
        load_function_class_from_file_mock.return_value.return_value = 'mock_class'
        plan = type('CreateFunctionPlan', (), {'name': 'function', 'or_replace': True, 'if_not_exists': False, 'inputs': ['inp'], 'outputs': ['out'], 'impl_path': impl_path, 'function_type': 'classification', 'metadata': {'key1': 'value1', 'key2': 'value2'}})
        evadb = MagicMock()
        evadb.catalog = catalog_instance
        evadb.config = MagicMock()
        create_function_executor = CreateFunctionExecutor(evadb, plan)
        actual_batch = next(create_function_executor.exec())
        catalog_instance().insert_function_catalog_entry.assert_called_with('function', 'test.py', 'classification', ['inp', 'out'], {'key1': 'value1', 'key2': 'value2'})
        self.assertEqual(actual_batch.frames[0][0], 'Function function added to the database.')
        function_entry = MagicMock()
        cache = MagicMock()
        function_entry.dep_caches = [cache]
        catalog_instance().get_function_catalog_entry_by_name.return_value = function_entry
        plan = type('CreateFunctionPlan', (), {'name': 'function', 'or_replace': True, 'if_not_exists': False, 'inputs': ['inp'], 'outputs': ['out'], 'impl_path': impl_path, 'function_type': 'prediction', 'metadata': {'key1': 'value3', 'key2': 'value4'}})
        create_function_executor = CreateFunctionExecutor(evadb, plan)
        actual_batch = next(create_function_executor.exec())
        catalog_instance().drop_function_cache_catalog_entry.assert_called_with(cache)
        catalog_instance().delete_function_catalog_entry_by_name.assert_called_with('function')
        catalog_instance().insert_function_catalog_entry.assert_called_with('function', 'test.py', 'prediction', ['inp', 'out'], {'key1': 'value3', 'key2': 'value4'})
        self.assertEqual(actual_batch.frames[0][0], 'Function function overwritten.')

    @patch('evadb.executor.create_function_executor.load_function_class_from_file')
    def test_should_raise_error_on_incorrect_io_definition(self, load_function_class_from_file_mock):
        catalog_instance = MagicMock()
        catalog_instance().get_function_catalog_entry_by_name.return_value = None
        catalog_instance().insert_function_catalog_entry.return_value = 'function'
        impl_path = MagicMock()
        abs_path = impl_path.absolute.return_value = MagicMock()
        abs_path.as_posix.return_value = 'test.py'
        load_function_class_from_file_mock.return_value.return_value = 'mock_class'
        incorrect_input_definition = PandasDataframe(columns=['Frame_Array', 'Frame_Array_2'], column_types=[NdArrayType.UINT8], column_shapes=[(3, 256, 256), (3, 256, 256)])
        load_function_class_from_file_mock.return_value.forward.tags = {'input': [incorrect_input_definition], 'output': []}
        plan = type('CreateFunctionPlan', (), {'name': 'function', 'if_not_exists': False, 'inputs': [], 'outputs': [], 'impl_path': impl_path, 'function_type': 'classification'})
        evadb = MagicMock
        evadb.catalog = catalog_instance
        evadb.config = MagicMock()
        create_function_executor = CreateFunctionExecutor(evadb, plan)
        with self.assertRaises(RuntimeError) as exc:
            next(create_function_executor.exec())
        self.assertIn('Error creating function, input/output definition incorrect:', str(exc.exception))
        catalog_instance().insert_function_catalog_entry.assert_not_called()

class PlanExecutor:
    """
    This is an interface between plan tree and execution tree.
    We traverse the plan tree and build execution tree from it

    Arguments:
        plan (AbstractPlan): Physical plan tree which needs to be executed
        evadb (EvaDBDatabase): database to execute the query on
    """

    def __init__(self, evadb: EvaDBDatabase, plan: AbstractPlan):
        self._db = evadb
        self._plan = plan

    def _build_execution_tree(self, plan: Union[AbstractPlan, AbstractStatement]) -> AbstractExecutor:
        """build the execution tree from plan tree

        Arguments:
            plan {AbstractPlan} -- Input Plan tree

        Returns:
            AbstractExecutor -- Compiled Execution tree
        """
        root = None
        if plan is None:
            return root
        if isinstance(plan, CreateDatabaseStatement):
            return CreateDatabaseExecutor(db=self._db, node=plan)
        elif isinstance(plan, UseStatement):
            return UseExecutor(db=self._db, node=plan)
        elif isinstance(plan, SetStatement):
            return SetExecutor(db=self._db, node=plan)
        elif isinstance(plan, CreateJobStatement):
            return CreateJobExecutor(db=self._db, node=plan)
        plan_opr_type = plan.opr_type
        if plan_opr_type == PlanOprType.SEQUENTIAL_SCAN:
            executor_node = SequentialScanExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.UNION:
            executor_node = UnionExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.STORAGE_PLAN:
            executor_node = StorageExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.PP_FILTER:
            executor_node = PPExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.CREATE:
            executor_node = CreateExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.RENAME:
            executor_node = RenameExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.DROP_OBJECT:
            executor_node = DropObjectExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.INSERT:
            executor_node = InsertExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.CREATE_FUNCTION:
            executor_node = CreateFunctionExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.LOAD_DATA:
            executor_node = LoadDataExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.GROUP_BY:
            executor_node = GroupByExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.ORDER_BY:
            executor_node = OrderByExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.LIMIT:
            executor_node = LimitExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.SAMPLE:
            executor_node = SampleExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.NESTED_LOOP_JOIN:
            executor_node = NestedLoopJoinExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.HASH_JOIN:
            executor_node = HashJoinExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.HASH_BUILD:
            executor_node = BuildJoinExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.FUNCTION_SCAN:
            executor_node = FunctionScanExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.EXCHANGE:
            executor_node = ExchangeExecutor(db=self._db, node=plan)
            inner_executor = self._build_execution_tree(plan.inner_plan)
            executor_node.build_inner_executor(inner_executor)
        elif plan_opr_type == PlanOprType.PROJECT:
            executor_node = ProjectExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.PREDICATE_FILTER:
            executor_node = PredicateExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.SHOW_INFO:
            executor_node = ShowInfoExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.EXPLAIN:
            executor_node = ExplainExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.CREATE_INDEX:
            executor_node = CreateIndexExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.APPLY_AND_MERGE:
            executor_node = ApplyAndMergeExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.VECTOR_INDEX_SCAN:
            executor_node = VectorIndexScanExecutor(db=self._db, node=plan)
        elif plan_opr_type == PlanOprType.DELETE:
            executor_node = DeleteExecutor(db=self._db, node=plan)
        if plan_opr_type != PlanOprType.EXPLAIN:
            for children in plan.children:
                executor_node.append_child(self._build_execution_tree(children))
        return executor_node

    def execute_plan(self, do_not_raise_exceptions: bool=False, do_not_print_exceptions: bool=False) -> Iterator[Batch]:
        """execute the plan tree"""
        try:
            execution_tree = self._build_execution_tree(self._plan)
            output = execution_tree.exec()
            if output is not None:
                yield from output
        except Exception as e:
            if do_not_raise_exceptions is False:
                if do_not_print_exceptions is False:
                    logger.exception(str(e))
                raise ExecutorError(e)

