# Cluster 38

@pytest.mark.notparallel
class DropObjectExecutorTest(unittest.TestCase):

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        self.video_file_path = create_sample_video()

    def tearDown(self):
        file_remove('dummy.avi')

    def _create_index(self, index_name):
        import numpy as np
        feat1 = np.array([[0, 0, 0]]).astype(np.float32)
        feat2 = np.array([[100, 100, 100]]).astype(np.float32)
        feat3 = np.array([[200, 200, 200]]).astype(np.float32)
        execute_query_fetch_all(self.evadb, 'create table if not exists testCreateIndexFeatTable (\n                feat NDARRAY FLOAT32(1,3)\n            );')
        feat_batch_data = Batch(pd.DataFrame(data={'feat': [feat1, feat2, feat3]}))
        feat_tb_entry = self.evadb.catalog().get_table_catalog_entry('testCreateIndexFeatTable')
        storage_engine = StorageEngine.factory(self.evadb, feat_tb_entry)
        storage_engine.write(feat_tb_entry, feat_batch_data)
        query = f'CREATE INDEX {index_name} ON testCreateIndexFeatTable (feat) USING FAISS;'
        execute_query_fetch_all(self.evadb, query)

    def test_should_drop_table(self):
        query = f"LOAD VIDEO '{self.video_file_path}' INTO MyVideo;"
        execute_query_fetch_all(self.evadb, query)
        table_catalog_entry = self.evadb.catalog().get_table_catalog_entry('MyVideo')
        video_dir = table_catalog_entry.file_url
        self.assertFalse(table_catalog_entry is None)
        column_objects = self.evadb.catalog().get_column_catalog_entries_by_table(table_catalog_entry)
        self.assertEqual(len(column_objects), len(get_video_table_column_definitions()) + 1)
        self.assertTrue(Path(video_dir).exists())
        video_metadata_table = self.evadb.catalog().get_multimedia_metadata_table_catalog_entry(table_catalog_entry)
        self.assertTrue(video_metadata_table is not None)
        drop_query = 'DROP TABLE IF EXISTS MyVideo;'
        execute_query_fetch_all(self.evadb, drop_query)
        self.assertTrue(self.evadb.catalog().get_table_catalog_entry('MyVideo') is None)
        column_objects = self.evadb.catalog().get_column_catalog_entries_by_table(table_catalog_entry)
        self.assertEqual(len(column_objects), 0)
        self.assertFalse(Path(video_dir).exists())
        drop_query = 'DROP TABLE MyVideo;'
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, drop_query, do_not_print_exceptions=True)
        execute_query_fetch_all(self.evadb, query)
        execute_query_fetch_all(self.evadb, drop_query)

    def run_create_function_query(self):
        create_function_query = "CREATE FUNCTION DummyObjectDetector\n            INPUT  (Frame_Array NDARRAY UINT8(3, 256, 256))\n            OUTPUT (label NDARRAY STR(10))\n            TYPE  Classification\n            IMPL  'test/util.py';"
        execute_query_fetch_all(self.evadb, create_function_query)

    def test_should_drop_function(self):
        self.run_create_function_query()
        function_name = 'DummyObjectDetector'
        function = self.evadb.catalog().get_function_catalog_entry_by_name(function_name)
        self.assertTrue(function is not None)
        drop_query = 'DROP FUNCTION IF EXISTS {};'.format(function_name)
        execute_query_fetch_all(self.evadb, drop_query)
        function = self.evadb.catalog().get_function_catalog_entry_by_name(function_name)
        self.assertTrue(function is None)
        self.run_create_function_query()
        execute_query_fetch_all(self.evadb, drop_query)

    def test_drop_wrong_function_name(self):
        self.run_create_function_query()
        right_function_name = 'DummyObjectDetector'
        wrong_function_name = 'FakeDummyObjectDetector'
        function = self.evadb.catalog().get_function_catalog_entry_by_name(right_function_name)
        self.assertTrue(function is not None)
        drop_query = 'DROP FUNCTION {};'.format(wrong_function_name)
        try:
            execute_query_fetch_all(self.evadb, drop_query, do_not_print_exceptions=True)
        except Exception as e:
            err_msg = 'Function {} does not exist, therefore cannot be dropped.'.format(wrong_function_name)
            self.assertTrue(str(e) == err_msg)
        function = self.evadb.catalog().get_function_catalog_entry_by_name(right_function_name)
        self.assertTrue(function is not None)

    def test_should_drop_index(self):
        index_name = 'index_name'
        self._create_index(index_name)
        index_obj = self.evadb.catalog().get_index_catalog_entry_by_name(index_name)
        self.assertTrue(index_obj is not None)
        wrong_function_name = 'wrong_function_name'
        drop_query = f'DROP INDEX {wrong_function_name};'
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, drop_query, do_not_print_exceptions=True)
        obj = self.evadb.catalog().get_index_catalog_entry_by_name(index_name)
        self.assertTrue(obj is not None)
        drop_query = f'DROP INDEX IF EXISTS {index_name};'
        execute_query_fetch_all(self.evadb, drop_query)
        index_obj = self.evadb.catalog().get_index_catalog_entry_by_name(index_name)
        self.assertTrue(index_obj is None)

    def test_should_drop_database(self):
        database_name = 'test_data_source'
        params = {'database': 'evadb.db'}
        query = f'CREATE DATABASE {database_name}\n                    WITH ENGINE = "sqlite",\n                    PARAMETERS = {params};'
        execute_query_fetch_all(self.evadb, query)
        self.assertIsNotNone(self.evadb.catalog().get_database_catalog_entry(database_name))
        execute_query_fetch_all(self.evadb, f'DROP DATABASE {database_name}')
        self.assertIsNone(self.evadb.catalog().get_database_catalog_entry(database_name))
        result = execute_query_fetch_all(self.evadb, f'DROP DATABASE IF EXISTS {database_name}')
        self.assertTrue('does not exist' in result.frames.to_string())
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, f'DROP DATABASE {database_name}', do_not_print_exceptions=True)
        execute_query_fetch_all(self.evadb, query)
        result = execute_query_fetch_all(self.evadb, f'DROP DATABASE IF EXISTS {database_name}')

    def test_should_drop_job(self):
        job_name = 'test_async_job'
        query = f"CREATE JOB {job_name} AS {{\n            SELECT * from job_catalog;\n        }}\n        START '2023-04-01'\n        END '2023-05-01'\n        EVERY 2 week;"
        execute_query_fetch_all(self.evadb, query)
        self.assertIsNotNone(self.evadb.catalog().get_job_catalog_entry(job_name))
        execute_query_fetch_all(self.evadb, f'DROP JOB {job_name}')
        self.assertIsNone(self.evadb.catalog().get_job_catalog_entry(job_name))
        result = execute_query_fetch_all(self.evadb, f'DROP JOB IF EXISTS {job_name}')
        self.assertTrue('does not exist' in result.frames.to_string())
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, f'DROP JOB {job_name}', do_not_print_exceptions=True)
        execute_query_fetch_all(self.evadb, query)
        result = execute_query_fetch_all(self.evadb, f'DROP JOB IF EXISTS {job_name}')

def get_video_table_column_definitions() -> List[ColumnDefinition]:
    """
    name: video path
    id: frame id
    data: frame data
    audio: frame audio
    """
    columns = [ColumnDefinition(VideoColumnName.name.name, ColumnType.TEXT, None, None, ColConstraintInfo(unique=True)), ColumnDefinition(VideoColumnName.id.name, ColumnType.INTEGER, None, None), ColumnDefinition(VideoColumnName.data.name, ColumnType.NDARRAY, NdArrayType.UINT8, (None, None, None)), ColumnDefinition(VideoColumnName.seconds.name, ColumnType.FLOAT, None, [])]
    return columns

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

class FunctionExpressionTest(unittest.TestCase):

    @patch('evadb.expression.function_expression.Context')
    def test_function_move_the_device_to_gpu_if_compatible(self, context):
        context_instance = context.return_value
        mock_function = MagicMock(spec=GPUCompatible)
        gpu_mock_function = Mock(return_value=pd.DataFrame())
        gpu_device_id = '2'
        mock_function.to_device.return_value = gpu_mock_function
        context_instance.gpu_device.return_value = gpu_device_id
        expression = FunctionExpression(lambda: mock_function, name='test', alias=Alias('func_expr'))
        input_batch = Batch(frames=pd.DataFrame())
        expression.evaluate(input_batch)
        mock_function.to_device.assert_called_with(gpu_device_id)
        gpu_mock_function.assert_called()

    def test_should_use_the_same_function_if_not_gpu_compatible(self):
        mock_function = MagicMock(return_value=pd.DataFrame())
        expression = FunctionExpression(lambda: mock_function, name='test', alias=Alias('func_expr'))
        input_batch = Batch(frames=pd.DataFrame())
        expression.evaluate(input_batch)
        mock_function.assert_called()

    @patch('evadb.expression.function_expression.Context')
    def test_should_execute_same_function_if_no_gpu(self, context):
        context_instance = context.return_value
        mock_function = MagicMock(spec=GPUCompatible, return_value=pd.DataFrame())
        context_instance.gpu_device.return_value = NO_GPU
        expression = FunctionExpression(lambda: mock_function, name='test', alias=Alias('func_expr'))
        input_batch = Batch(frames=pd.DataFrame())
        expression.evaluate(input_batch)
        mock_function.assert_called()

class ParserTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_select_from_data_source(self):
        parser = Parser()
        query = 'SELECT * FROM DemoDB.DemoTable'
        evadb_stmt_list = parser.parse(query)
        self.assertIsInstance(evadb_stmt_list, list)
        self.assertEqual(len(evadb_stmt_list), 1)
        self.assertEqual(evadb_stmt_list[0].stmt_type, StatementType.SELECT)
        select_stmt = evadb_stmt_list[0]
        self.assertIsNotNone(select_stmt.from_table)
        self.assertIsInstance(select_stmt.from_table, TableRef)
        self.assertEqual(select_stmt.from_table.table.table_name, 'DemoTable')
        self.assertEqual(select_stmt.from_table.table.database_name, 'DemoDB')

    def test_use_statement(self):
        parser = Parser()
        query_list = ['SELECT * FROM DemoTable', 'SELECT * FROM DemoTable WHERE col == "xxx"\n            ', "SELECT * FROM DemoTable WHERE col == 'xxx'\n            "]
        for query in query_list:
            use_query = f'USE DemoDB {{{query}}};'
            evadb_stmt_list = parser.parse(use_query)
            self.assertIsInstance(evadb_stmt_list, list)
            self.assertEqual(len(evadb_stmt_list), 1)
            self.assertEqual(evadb_stmt_list[0].stmt_type, StatementType.USE)
            expected_stmt = UseStatement('DemoDB', query)
            actual_stmt = evadb_stmt_list[0]
            self.assertEqual(actual_stmt, expected_stmt)

    def test_create_index_statement(self):
        parser = Parser()
        create_index_query = 'CREATE INDEX testindex ON MyVideo (featCol) USING FAISS;'
        evadb_stmt_list = parser.parse(create_index_query)
        self.assertIsInstance(evadb_stmt_list, list)
        self.assertEqual(len(evadb_stmt_list), 1)
        self.assertEqual(evadb_stmt_list[0].stmt_type, StatementType.CREATE_INDEX)
        expected_stmt = CreateIndexStatement('testindex', False, TableRef(TableInfo('MyVideo')), [ColumnDefinition('featCol', None, None, None)], VectorStoreType.FAISS, [TupleValueExpression(name='featCol')])
        actual_stmt = evadb_stmt_list[0]
        self.assertEqual(actual_stmt, expected_stmt)
        self.assertEqual(actual_stmt.index_def, create_index_query)
        expected_stmt = CreateIndexStatement('testindex', True, TableRef(TableInfo('MyVideo')), [ColumnDefinition('featCol', None, None, None)], VectorStoreType.FAISS, [TupleValueExpression(name='featCol')])
        create_index_query = 'CREATE INDEX IF NOT EXISTS testindex ON MyVideo (featCol) USING FAISS;'
        evadb_stmt_list = parser.parse(create_index_query)
        actual_stmt = evadb_stmt_list[0]
        expected_stmt._if_not_exists = True
        self.assertEqual(actual_stmt, expected_stmt)
        self.assertEqual(actual_stmt.index_def, create_index_query)
        create_index_query = 'CREATE INDEX testindex ON MyVideo (FeatureExtractor(featCol)) USING FAISS;'
        evadb_stmt_list = parser.parse(create_index_query)
        self.assertIsInstance(evadb_stmt_list, list)
        self.assertEqual(len(evadb_stmt_list), 1)
        self.assertEqual(evadb_stmt_list[0].stmt_type, StatementType.CREATE_INDEX)
        func_expr = FunctionExpression(None, 'FeatureExtractor')
        func_expr.append_child(TupleValueExpression('featCol'))
        expected_stmt = CreateIndexStatement('testindex', False, TableRef(TableInfo('MyVideo')), [ColumnDefinition('featCol', None, None, None)], VectorStoreType.FAISS, [func_expr])
        actual_stmt = evadb_stmt_list[0]
        self.assertEqual(actual_stmt, expected_stmt)
        self.assertEqual(actual_stmt.index_def, create_index_query)

    @unittest.skip('Skip parser exception handling testcase, moved to binder')
    def test_create_index_exception_statement(self):
        parser = Parser()
        create_index_query = 'CREATE INDEX testindex USING FAISS ON MyVideo (featCol1, featCol2);'
        with self.assertRaises(Exception):
            parser.parse(create_index_query)

    def test_explain_dml_statement(self):
        parser = Parser()
        explain_query = 'EXPLAIN SELECT CLASS FROM TAIPAI;'
        evadb_statement_list = parser.parse(explain_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.EXPLAIN)
        inner_stmt = evadb_statement_list[0].explainable_stmt
        self.assertEqual(inner_stmt.stmt_type, StatementType.SELECT)
        self.assertIsNotNone(inner_stmt.from_table)
        self.assertIsInstance(inner_stmt.from_table, TableRef)
        self.assertEqual(inner_stmt.from_table.table.table_name, 'TAIPAI')

    def test_explain_ddl_statement(self):
        parser = Parser()
        select_query = 'SELECT id, Yolo(frame).labels FROM MyVideo\n                        WHERE id<5; '
        explain_query = 'EXPLAIN CREATE TABLE uadtrac_fastRCNN AS {}'.format(select_query)
        evadb_statement_list = parser.parse(explain_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.EXPLAIN)
        inner_stmt = evadb_statement_list[0].explainable_stmt
        self.assertEqual(inner_stmt.stmt_type, StatementType.CREATE)
        self.assertIsNotNone(inner_stmt.table_info, TableRef(TableInfo('uadetrac_fastRCNN')))

    def test_create_table_statement(self):
        parser = Parser()
        single_queries = []
        single_queries.append('CREATE TABLE IF NOT EXISTS Persons (\n                  Frame_ID INTEGER UNIQUE,\n                  Frame_Data TEXT,\n                  Frame_Value FLOAT,\n                  Frame_Array NDARRAY UINT8(5, 100, 2432, 4324, 100)\n            );')
        expected_cci = ColConstraintInfo()
        expected_cci.nullable = True
        unique_cci = ColConstraintInfo()
        unique_cci.unique = True
        unique_cci.nullable = False
        expected_stmt = CreateTableStatement(TableInfo('Persons'), True, [ColumnDefinition('Frame_ID', ColumnType.INTEGER, None, (), unique_cci), ColumnDefinition('Frame_Data', ColumnType.TEXT, None, (), expected_cci), ColumnDefinition('Frame_Value', ColumnType.FLOAT, None, (), expected_cci), ColumnDefinition('Frame_Array', ColumnType.NDARRAY, NdArrayType.UINT8, (5, 100, 2432, 4324, 100), expected_cci)])
        for query in single_queries:
            evadb_statement_list = parser.parse(query)
            self.assertIsInstance(evadb_statement_list, list)
            self.assertEqual(len(evadb_statement_list), 1)
            self.assertIsInstance(evadb_statement_list[0], AbstractStatement)
            self.assertEqual(evadb_statement_list[0], expected_stmt)

    def test_create_table_with_dimension_statement(self):
        parser = Parser()
        single_queries = []
        single_queries.append('CREATE TABLE IF NOT EXISTS Persons (\n                  Frame_ID INTEGER UNIQUE,\n                  Frame_Data TEXT(10),\n                  Frame_Value FLOAT(1000, 201),\n                  Frame_Array NDARRAY UINT8(5, 100, 2432, 4324, 100)\n            );')
        expected_cci = ColConstraintInfo()
        expected_cci.nullable = True
        unique_cci = ColConstraintInfo()
        unique_cci.unique = True
        unique_cci.nullable = False
        expected_stmt = CreateTableStatement(TableInfo('Persons'), True, [ColumnDefinition('Frame_ID', ColumnType.INTEGER, None, (), unique_cci), ColumnDefinition('Frame_Data', ColumnType.TEXT, None, (10,), expected_cci), ColumnDefinition('Frame_Value', ColumnType.FLOAT, None, (1000, 201), expected_cci), ColumnDefinition('Frame_Array', ColumnType.NDARRAY, NdArrayType.UINT8, (5, 100, 2432, 4324, 100), expected_cci)])
        for query in single_queries:
            evadb_statement_list = parser.parse(query)
            self.assertIsInstance(evadb_statement_list, list)
            self.assertEqual(len(evadb_statement_list), 1)
            self.assertIsInstance(evadb_statement_list[0], AbstractStatement)
            self.assertEqual(evadb_statement_list[0], expected_stmt)

    def test_create_table_statement_with_rare_datatypes(self):
        parser = Parser()
        query = 'CREATE TABLE IF NOT EXISTS Dummy (\n                  C NDARRAY UINT8(5),\n                  D NDARRAY INT16(5),\n                  E NDARRAY INT32(5),\n                  F NDARRAY INT64(5),\n                  G NDARRAY UNICODE(5),\n                  H NDARRAY BOOLEAN(5),\n                  I NDARRAY FLOAT64(5),\n                  J NDARRAY DECIMAL(5),\n                  K NDARRAY DATETIME(5)\n            );'
        evadb_statement_list = parser.parse(query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertIsInstance(evadb_statement_list[0], AbstractStatement)

    def test_create_table_statement_without_proper_datatype(self):
        parser = Parser()
        query = 'CREATE TABLE IF NOT EXISTS Dummy (\n                  C NDARRAY INT(5)\n                );'
        with self.assertRaises(Exception):
            parser.parse(query)

    def test_create_table_exception_statement(self):
        parser = Parser()
        create_table_query = 'CREATE TABLE ();'
        with self.assertRaises(Exception):
            parser.parse(create_table_query)

    def test_rename_table_statement(self):
        parser = Parser()
        rename_queries = 'RENAME TABLE student TO student_info'
        expected_stmt = RenameTableStatement(TableRef(TableInfo('student')), TableInfo('student_info'))
        evadb_statement_list = parser.parse(rename_queries)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.RENAME)
        rename_stmt = evadb_statement_list[0]
        self.assertEqual(rename_stmt, expected_stmt)

    def test_drop_table_statement(self):
        parser = Parser()
        drop_queries = 'DROP TABLE student_info'
        expected_stmt = DropObjectStatement(ObjectType.TABLE, 'student_info', False)
        evadb_statement_list = parser.parse(drop_queries)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.DROP_OBJECT)
        drop_stmt = evadb_statement_list[0]
        self.assertEqual(drop_stmt, expected_stmt)

    def test_drop_function_statement_str(self):
        drop_func_query1 = 'DROP FUNCTION MyFunc;'
        drop_func_query2 = 'DROP FUNCTION IF EXISTS MyFunc;'
        expected_stmt1 = DropObjectStatement(ObjectType.FUNCTION, 'MyFunc', False)
        expected_stmt2 = DropObjectStatement(ObjectType.FUNCTION, 'MyFunc', True)
        self.assertEqual(str(expected_stmt1), drop_func_query1)
        self.assertEqual(str(expected_stmt2), drop_func_query2)

    def test_single_statement_queries(self):
        parser = Parser()
        single_queries = []
        single_queries.append('SELECT CLASS FROM TAIPAI;')
        single_queries.append("SELECT CLASS FROM TAIPAI WHERE CLASS = 'VAN';")
        single_queries.append("SELECT CLASS,REDNESS FROM TAIPAI             WHERE CLASS = 'VAN' AND REDNESS > 20.5;")
        single_queries.append("SELECT CLASS FROM TAIPAI             WHERE (CLASS = 'VAN' AND REDNESS < 300 ) OR REDNESS > 500;")
        single_queries.append("SELECT CLASS FROM TAIPAI             WHERE (CLASS = 'VAN' AND REDNESS < 300 ) OR REDNESS > 500;")
        for query in single_queries:
            evadb_statement_list = parser.parse(query)
            self.assertIsInstance(evadb_statement_list, list)
            self.assertEqual(len(evadb_statement_list), 1)
            self.assertIsInstance(evadb_statement_list[0], AbstractStatement)

    def test_multiple_statement_queries(self):
        parser = Parser()
        multiple_queries = []
        multiple_queries.append("SELECT CLASS FROM TAIPAI                 WHERE (CLASS != 'VAN' AND REDNESS < 300)  OR REDNESS > 500;                 SELECT REDNESS FROM TAIPAI                 WHERE (CLASS = 'VAN' AND REDNESS = 300)")
        for query in multiple_queries:
            evadb_statement_list = parser.parse(query)
            self.assertIsInstance(evadb_statement_list, list)
            self.assertEqual(len(evadb_statement_list), 2)
            self.assertIsInstance(evadb_statement_list[0], AbstractStatement)
            self.assertIsInstance(evadb_statement_list[1], AbstractStatement)

    def test_select_statement(self):
        parser = Parser()
        select_query = "SELECT CLASS, REDNESS FROM TAIPAI                 WHERE (CLASS = 'VAN' AND REDNESS < 300 ) OR REDNESS > 500;"
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)
        select_stmt = evadb_statement_list[0]
        self.assertIsNotNone(select_stmt.target_list)
        self.assertEqual(len(select_stmt.target_list), 2)
        self.assertEqual(select_stmt.target_list[0].etype, ExpressionType.TUPLE_VALUE)
        self.assertEqual(select_stmt.target_list[1].etype, ExpressionType.TUPLE_VALUE)
        self.assertIsNotNone(select_stmt.from_table)
        self.assertIsInstance(select_stmt.from_table, TableRef)
        self.assertEqual(select_stmt.from_table.table.table_name, 'TAIPAI')
        self.assertIsNotNone(select_stmt.where_clause)

    def test_select_with_empty_string_literal(self):
        parser = Parser()
        select_query = "SELECT '' FROM TAIPAI;"
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)

    def test_string_literal_with_escaped_single_quote(self):
        parser = Parser()
        select_query = "SELECT ChatGPT('Here\\'s a question', 'This is the context') FROM TAIPAI;"
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)

    def test_string_literal_with_semi_colon(self):
        parser = Parser()
        select_query = 'SELECT ChatGPT("Here\'s a; question", "This is the context") FROM TAIPAI;'
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)

    def test_string_literal_with_single_quotes_from_variable(self):
        parser = Parser()
        question = json.dumps("Here's a question")
        answer = json.dumps('This is "the" context')
        select_query = f'SELECT ChatGPT({question}, {answer}) FROM TAIPAI;'
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)

    def test_select_union_statement(self):
        parser = Parser()
        select_union_query = 'SELECT CLASS, REDNESS FROM TAIPAI             UNION ALL SELECT CLASS, REDNESS FROM SHANGHAI;'
        evadb_statement_list = parser.parse(select_union_query)
        select_stmt = evadb_statement_list[0]
        self.assertIsNotNone(select_stmt.union_link)
        self.assertEqual(select_stmt.union_all, True)
        second_select_stmt = select_stmt.union_link
        self.assertIsNone(second_select_stmt.union_link)

    def test_select_statement_class(self):
        """Testing setting different clauses for Select
        Statement class
        Class: SelectStatement"""
        select_stmt_new = SelectStatement()
        parser = Parser()
        select_query_new = "SELECT CLASS, REDNESS FROM TAIPAI             WHERE (CLASS = 'VAN' AND REDNESS < 400 ) OR REDNESS > 700;"
        evadb_statement_list = parser.parse(select_query_new)
        select_stmt = evadb_statement_list[0]
        select_stmt_new.where_clause = select_stmt.where_clause
        select_stmt_new.target_list = select_stmt.target_list
        select_stmt_new.from_table = select_stmt.from_table
        self.assertEqual(select_stmt_new.where_clause, select_stmt.where_clause)
        self.assertEqual(select_stmt_new.target_list, select_stmt.target_list)
        self.assertEqual(select_stmt_new.from_table, select_stmt.from_table)
        self.assertEqual(str(select_stmt_new), str(select_stmt))

    def test_select_statement_where_class(self):
        """
        Unit test for logical operators in the where clause.
        """

        def _verify_select_statement(evadb_statement_list):
            self.assertIsInstance(evadb_statement_list, list)
            self.assertEqual(len(evadb_statement_list), 1)
            self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)
            select_stmt = evadb_statement_list[0]
            self.assertIsNotNone(select_stmt.target_list)
            self.assertEqual(len(select_stmt.target_list), 2)
            self.assertEqual(select_stmt.target_list[0].etype, ExpressionType.TUPLE_VALUE)
            self.assertEqual(select_stmt.target_list[0].name, 'CLASS')
            self.assertEqual(select_stmt.target_list[1].etype, ExpressionType.TUPLE_VALUE)
            self.assertEqual(select_stmt.target_list[1].name, 'REDNESS')
            self.assertIsNotNone(select_stmt.from_table)
            self.assertIsInstance(select_stmt.from_table, TableRef)
            self.assertEqual(select_stmt.from_table.table.table_name, 'TAIPAI')
            self.assertIsNotNone(select_stmt.where_clause)
            self.assertIsInstance(select_stmt.where_clause, LogicalExpression)
            self.assertEqual(select_stmt.where_clause.etype, ExpressionType.LOGICAL_AND)
            self.assertEqual(len(select_stmt.where_clause.children), 2)
            left = select_stmt.where_clause.children[0]
            right = select_stmt.where_clause.children[1]
            self.assertEqual(left.etype, ExpressionType.COMPARE_EQUAL)
            self.assertEqual(right.etype, ExpressionType.COMPARE_LESSER)
            self.assertEqual(len(left.children), 2)
            self.assertEqual(left.children[0].etype, ExpressionType.TUPLE_VALUE)
            self.assertEqual(left.children[0].name, 'CLASS')
            self.assertEqual(left.children[1].etype, ExpressionType.CONSTANT_VALUE)
            self.assertEqual(left.children[1].value, 'VAN')
            self.assertEqual(len(right.children), 2)
            self.assertEqual(right.children[0].etype, ExpressionType.TUPLE_VALUE)
            self.assertEqual(right.children[0].name, 'REDNESS')
            self.assertEqual(right.children[1].etype, ExpressionType.CONSTANT_VALUE)
            self.assertEqual(right.children[1].value, 400)
        parser = Parser()
        select_query = "SELECT CLASS, REDNESS FROM TAIPAI WHERE CLASS = 'VAN' AND REDNESS < 400;"
        _verify_select_statement(parser.parse(select_query))
        select_query = "select CLASS, REDNESS from TAIPAI where CLASS = 'VAN' and REDNESS < 400;"
        _verify_select_statement(parser.parse(select_query))
        select_query = "SELECT CLASS, REDNESS FROM TAIPAI WHERE CLASS = 'VAN' XOR REDNESS < 400;"
        with self.assertRaises(NotImplementedError) as cm:
            parser.parse(select_query)
        self.assertEqual(str(cm.exception), 'Unsupported logical operator: XOR')

    def test_select_statement_groupby_class(self):
        """Testing sample frequency"""
        parser = Parser()
        select_query = "SELECT FIRST(id) FROM TAIPAI GROUP BY '8 frames';"
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)
        select_stmt = evadb_statement_list[0]
        self.assertIsNotNone(select_stmt.target_list)
        self.assertEqual(len(select_stmt.target_list), 1)
        self.assertEqual(select_stmt.target_list[0].etype, ExpressionType.AGGREGATION_FIRST)
        self.assertIsNotNone(select_stmt.from_table)
        self.assertIsInstance(select_stmt.from_table, TableRef)
        self.assertEqual(select_stmt.from_table.table.table_name, 'TAIPAI')
        self.assertEqual(select_stmt.groupby_clause, ConstantValueExpression('8 frames', v_type=ColumnType.TEXT))

    def test_select_statement_orderby_class(self):
        """Testing order by clause in select statement
        Class: SelectStatement"""
        parser = Parser()
        select_query = "SELECT CLASS, REDNESS FROM TAIPAI                     WHERE (CLASS = 'VAN' AND REDNESS < 400 ) OR REDNESS > 700                     ORDER BY CLASS, REDNESS DESC;"
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)
        select_stmt = evadb_statement_list[0]
        self.assertIsNotNone(select_stmt.target_list)
        self.assertEqual(len(select_stmt.target_list), 2)
        self.assertEqual(select_stmt.target_list[0].etype, ExpressionType.TUPLE_VALUE)
        self.assertEqual(select_stmt.target_list[1].etype, ExpressionType.TUPLE_VALUE)
        self.assertIsNotNone(select_stmt.from_table)
        self.assertIsInstance(select_stmt.from_table, TableRef)
        self.assertEqual(select_stmt.from_table.table.table_name, 'TAIPAI')
        self.assertIsNotNone(select_stmt.where_clause)
        self.assertIsNotNone(select_stmt.orderby_list)
        self.assertEqual(len(select_stmt.orderby_list), 2)
        self.assertEqual(select_stmt.orderby_list[0][0].name, 'CLASS')
        self.assertEqual(select_stmt.orderby_list[0][1], ParserOrderBySortType.ASC)
        self.assertEqual(select_stmt.orderby_list[1][0].name, 'REDNESS')
        self.assertEqual(select_stmt.orderby_list[1][1], ParserOrderBySortType.DESC)

    def test_select_statement_limit_class(self):
        """Testing limit clause in select statement
        Class: SelectStatement"""
        parser = Parser()
        select_query = "SELECT CLASS, REDNESS FROM TAIPAI                     WHERE (CLASS = 'VAN' AND REDNESS < 400 ) OR REDNESS > 700                     ORDER BY CLASS, REDNESS DESC LIMIT 3;"
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)
        select_stmt = evadb_statement_list[0]
        self.assertIsNotNone(select_stmt.target_list)
        self.assertEqual(len(select_stmt.target_list), 2)
        self.assertEqual(select_stmt.target_list[0].etype, ExpressionType.TUPLE_VALUE)
        self.assertEqual(select_stmt.target_list[1].etype, ExpressionType.TUPLE_VALUE)
        self.assertIsNotNone(select_stmt.from_table)
        self.assertIsInstance(select_stmt.from_table, TableRef)
        self.assertEqual(select_stmt.from_table.table.table_name, 'TAIPAI')
        self.assertIsNotNone(select_stmt.where_clause)
        self.assertIsNotNone(select_stmt.orderby_list)
        self.assertEqual(len(select_stmt.orderby_list), 2)
        self.assertEqual(select_stmt.orderby_list[0][0].name, 'CLASS')
        self.assertEqual(select_stmt.orderby_list[0][1], ParserOrderBySortType.ASC)
        self.assertEqual(select_stmt.orderby_list[1][0].name, 'REDNESS')
        self.assertEqual(select_stmt.orderby_list[1][1], ParserOrderBySortType.DESC)
        self.assertIsNotNone(select_stmt.limit_count)
        self.assertEqual(select_stmt.limit_count, ConstantValueExpression(3))

    def test_select_statement_sample_class(self):
        """Testing sample frequency"""
        parser = Parser()
        select_query = 'SELECT CLASS, REDNESS FROM TAIPAI SAMPLE 5;'
        evadb_statement_list = parser.parse(select_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SELECT)
        select_stmt = evadb_statement_list[0]
        self.assertIsNotNone(select_stmt.target_list)
        self.assertEqual(len(select_stmt.target_list), 2)
        self.assertEqual(select_stmt.target_list[0].etype, ExpressionType.TUPLE_VALUE)
        self.assertEqual(select_stmt.target_list[1].etype, ExpressionType.TUPLE_VALUE)
        self.assertIsNotNone(select_stmt.from_table)
        self.assertIsInstance(select_stmt.from_table, TableRef)
        self.assertEqual(select_stmt.from_table.table.table_name, 'TAIPAI')
        self.assertEqual(select_stmt.from_table.sample_freq, ConstantValueExpression(5))

    def test_select_function_star(self):
        parser = Parser()
        query = 'SELECT DemoFunc(*) FROM DemoDB.DemoTable;'
        evadb_stmt_list = parser.parse(query)
        self.assertIsInstance(evadb_stmt_list, list)
        self.assertEqual(len(evadb_stmt_list), 1)
        self.assertEqual(evadb_stmt_list[0].stmt_type, StatementType.SELECT)
        select_stmt = evadb_stmt_list[0]
        self.assertIsNotNone(select_stmt.target_list)
        self.assertEqual(len(select_stmt.target_list), 1)
        self.assertEqual(select_stmt.target_list[0].etype, ExpressionType.FUNCTION_EXPRESSION)
        self.assertEqual(len(select_stmt.target_list[0].children), 1)
        self.assertEqual(select_stmt.target_list[0].children[0].etype, ExpressionType.TUPLE_VALUE)
        self.assertEqual(select_stmt.target_list[0].children[0].name, '*')
        self.assertIsNotNone(select_stmt.from_table)
        self.assertIsInstance(select_stmt.from_table, TableRef)
        self.assertEqual(select_stmt.from_table.table.table_name, 'DemoTable')
        self.assertEqual(select_stmt.from_table.table.database_name, 'DemoDB')

    def test_select_without_table_source(self):
        parser = Parser()
        query = 'SELECT DemoFunc(12);'
        evadb_stmt_list = parser.parse(query)
        self.assertIsInstance(evadb_stmt_list, list)
        self.assertEqual(len(evadb_stmt_list), 1)
        self.assertEqual(evadb_stmt_list[0].stmt_type, StatementType.SELECT)
        select_stmt = evadb_stmt_list[0]
        self.assertIsNotNone(select_stmt.target_list)
        self.assertEqual(len(select_stmt.target_list), 1)
        self.assertEqual(select_stmt.target_list[0].etype, ExpressionType.FUNCTION_EXPRESSION)
        self.assertEqual(len(select_stmt.target_list[0].children), 1)
        self.assertEqual(select_stmt.target_list[0].children[0].etype, ExpressionType.CONSTANT_VALUE)
        self.assertEqual(select_stmt.target_list[0].children[0].value, 12)
        self.assertIsNone(select_stmt.from_table)

    def test_table_ref(self):
        """Testing table info in TableRef
        Class: TableInfo
        """
        table_info = TableInfo('TAIPAI', 'Schema', 'Database')
        table_ref_obj = TableRef(table_info)
        select_stmt_new = SelectStatement()
        select_stmt_new.from_table = table_ref_obj
        self.assertEqual(select_stmt_new.from_table.table.table_name, 'TAIPAI')
        self.assertEqual(select_stmt_new.from_table.table.schema_name, 'Schema')
        self.assertEqual(select_stmt_new.from_table.table.database_name, 'Database')

    def test_insert_statement(self):
        parser = Parser()
        insert_query = "INSERT INTO MyVideo (Frame_ID, Frame_Path)\n                                    VALUES    (1, '/mnt/frames/1.png');\n                        "
        expected_stmt = InsertTableStatement(TableRef(TableInfo('MyVideo')), [TupleValueExpression('Frame_ID'), TupleValueExpression('Frame_Path')], [[ConstantValueExpression(1), ConstantValueExpression('/mnt/frames/1.png', ColumnType.TEXT)]])
        evadb_statement_list = parser.parse(insert_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.INSERT)
        insert_stmt = evadb_statement_list[0]
        self.assertEqual(insert_stmt, expected_stmt)

    def test_delete_statement(self):
        parser = Parser()
        delete_statement = 'DELETE FROM Foo WHERE id > 5'
        evadb_statement_list = parser.parse(delete_statement)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.DELETE)
        delete_stmt = evadb_statement_list[0]
        expected_stmt = DeleteTableStatement(TableRef(TableInfo('Foo')), ComparisonExpression(ExpressionType.COMPARE_GREATER, TupleValueExpression('id'), ConstantValueExpression(5)))
        self.assertEqual(delete_stmt, expected_stmt)

    def test_set_statement(self):
        parser = Parser()
        set_statement = "SET OPENAIKEY = 'ABCD'"
        evadb_statement_list = parser.parse(set_statement)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SET)
        set_stmt = evadb_statement_list[0]
        expected_stmt = SetStatement('OPENAIKEY', ConstantValueExpression('ABCD', ColumnType.TEXT))
        self.assertEqual(set_stmt, expected_stmt)
        set_statement = "SET OPENAIKEY TO 'ABCD'"
        evadb_statement_list = parser.parse(set_statement)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SET)
        set_stmt = evadb_statement_list[0]
        expected_stmt = SetStatement('OPENAIKEY', ConstantValueExpression('ABCD', ColumnType.TEXT))
        self.assertEqual(set_stmt, expected_stmt)

    def test_show_config_statement(self):
        parser = Parser()
        show_config_statement = 'SHOW OPENAIKEY'
        evadb_statement_list = parser.parse(show_config_statement)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.SHOW)
        show_config_stmt = evadb_statement_list[0]
        expected_stmt = ShowStatement(show_type=ShowType.CONFIGS, show_val='OPENAIKEY')
        self.assertEqual(show_config_stmt, expected_stmt)

    def test_create_predict_function_statement(self):
        parser = Parser()
        create_func_query = "\n            CREATE OR REPLACE FUNCTION HomeSalesForecast FROM\n            ( SELECT * FROM postgres_data.home_sales )\n            TYPE Forecasting\n            PREDICT 'price';\n        "
        evadb_statement_list = parser.parse(create_func_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.CREATE_FUNCTION)
        create_func_stmt = evadb_statement_list[0]
        self.assertEqual(create_func_stmt.name, 'HomeSalesForecast')
        self.assertEqual(create_func_stmt.or_replace, True)
        self.assertEqual(create_func_stmt.if_not_exists, False)
        self.assertEqual(create_func_stmt.impl_path, None)
        self.assertEqual(create_func_stmt.inputs, [])
        self.assertEqual(create_func_stmt.outputs, [])
        self.assertEqual(create_func_stmt.function_type, 'Forecasting')
        self.assertEqual(create_func_stmt.metadata, [('predict', 'price')])
        nested_select_stmt = create_func_stmt.query
        self.assertEqual(nested_select_stmt.stmt_type, StatementType.SELECT)
        self.assertEqual(len(nested_select_stmt.target_list), 1)
        self.assertEqual(nested_select_stmt.target_list[0].etype, ExpressionType.TUPLE_VALUE)
        self.assertEqual(nested_select_stmt.target_list[0].name, '*')
        self.assertIsInstance(nested_select_stmt.from_table, TableRef)
        self.assertIsInstance(nested_select_stmt.from_table.table, TableInfo)
        self.assertEqual(nested_select_stmt.from_table.table.table_name, 'home_sales')
        self.assertEqual(nested_select_stmt.from_table.table.database_name, 'postgres_data')

    def test_create_function_statement(self):
        parser = Parser()
        create_func_query = 'CREATE FUNCTION IF NOT EXISTS FastRCNN\n                  INPUT  (Frame_Array NDARRAY UINT8(3, 256, 256))\n                  OUTPUT (Labels NDARRAY STR(10), Bbox NDARRAY UINT8(10, 4))\n                  TYPE  Classification\n                  IMPL  \'data/fastrcnn.py\'\n                  PREDICT "VALUE";\n        '
        expected_cci = ColConstraintInfo()
        expected_cci.nullable = True
        expected_stmt = CreateFunctionStatement('FastRCNN', False, True, Path('data/fastrcnn.py'), [ColumnDefinition('Frame_Array', ColumnType.NDARRAY, NdArrayType.UINT8, (3, 256, 256), expected_cci)], [ColumnDefinition('Labels', ColumnType.NDARRAY, NdArrayType.STR, (10,), expected_cci), ColumnDefinition('Bbox', ColumnType.NDARRAY, NdArrayType.UINT8, (10, 4), expected_cci)], 'Classification', None, [('predict', 'VALUE')])
        evadb_statement_list = parser.parse(create_func_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.CREATE_FUNCTION)
        self.assertEqual(str(evadb_statement_list[0]), str(expected_stmt))
        create_func_stmt = evadb_statement_list[0]
        self.assertEqual(create_func_stmt, expected_stmt)

    def test_load_video_data_statement(self):
        parser = Parser()
        load_data_query = "LOAD VIDEO 'data/video.mp4'\n                             INTO MyVideo"
        file_options = {}
        file_options['file_format'] = FileFormatType.VIDEO
        column_list = None
        expected_stmt = LoadDataStatement(TableInfo('MyVideo'), Path('data/video.mp4'), column_list, file_options)
        evadb_statement_list = parser.parse(load_data_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.LOAD_DATA)
        load_data_stmt = evadb_statement_list[0]
        self.assertEqual(load_data_stmt, expected_stmt)

    def test_load_csv_data_statement(self):
        parser = Parser()
        load_data_query = "LOAD CSV 'data/meta.csv'\n                             INTO\n                             MyMeta (id, frame_id, video_id, label);"
        file_options = {}
        file_options['file_format'] = FileFormatType.CSV
        expected_stmt = LoadDataStatement(TableInfo('MyMeta'), Path('data/meta.csv'), [TupleValueExpression('id'), TupleValueExpression('frame_id'), TupleValueExpression('video_id'), TupleValueExpression('label')], file_options)
        evadb_statement_list = parser.parse(load_data_query)
        self.assertIsInstance(evadb_statement_list, list)
        self.assertEqual(len(evadb_statement_list), 1)
        self.assertEqual(evadb_statement_list[0].stmt_type, StatementType.LOAD_DATA)
        load_data_stmt = evadb_statement_list[0]
        self.assertEqual(load_data_stmt, expected_stmt)

    def test_nested_select_statement(self):
        parser = Parser()
        sub_query = "SELECT CLASS FROM TAIPAI WHERE CLASS = 'VAN'"
        nested_query = 'SELECT ID FROM ({}) AS T;'.format(sub_query)
        parsed_sub_query = parser.parse(sub_query)[0]
        actual_stmt = parser.parse(nested_query)[0]
        self.assertEqual(actual_stmt.stmt_type, StatementType.SELECT)
        self.assertEqual(actual_stmt.target_list[0].name, 'ID')
        self.assertEqual(actual_stmt.from_table, TableRef(parsed_sub_query, alias=Alias('T')))
        sub_query = "SELECT Yolo(frame).bbox FROM autonomous_vehicle_1\n                              WHERE Yolo(frame).label = 'vehicle'"
        nested_query = "SELECT Licence_plate(bbox) FROM\n                            ({}) AS T\n                          WHERE Is_suspicious(bbox) = 1 AND\n                                Licence_plate(bbox) = '12345';\n                      ".format(sub_query)
        query = "SELECT Licence_plate(bbox) FROM TAIPAI\n                    WHERE Is_suspicious(bbox) = 1 AND\n                        Licence_plate(bbox) = '12345';\n                "
        query_stmt = parser.parse(query)[0]
        actual_stmt = parser.parse(nested_query)[0]
        sub_query_stmt = parser.parse(sub_query)[0]
        self.assertEqual(actual_stmt.from_table, TableRef(sub_query_stmt, alias=Alias('T')))
        self.assertEqual(actual_stmt.where_clause, query_stmt.where_clause)
        self.assertEqual(actual_stmt.target_list, query_stmt.target_list)

    def test_should_return_false_for_unequal_expression(self):
        table = TableRef(TableInfo('MyVideo'))
        load_stmt = LoadDataStatement(table, Path('data/video.mp4'), FileFormatType.VIDEO)
        insert_stmt = InsertTableStatement(table)
        create_func = CreateFunctionStatement('func', False, False, Path('data/fastrcnn.py'), [ColumnDefinition('frame', ColumnType.NDARRAY, NdArrayType.UINT8, (3, 256, 256))], [ColumnDefinition('labels', ColumnType.NDARRAY, NdArrayType.STR, 10)], 'Classification')
        select_stmt = SelectStatement()
        self.assertNotEqual(load_stmt, insert_stmt)
        self.assertNotEqual(insert_stmt, load_stmt)
        self.assertNotEqual(create_func, insert_stmt)
        self.assertNotEqual(select_stmt, create_func)

    def test_create_table_from_select(self):
        select_query = 'SELECT id, Yolo(frame).labels FROM MyVideo\n                        WHERE id<5; '
        query = 'CREATE TABLE uadtrac_fastRCNN AS {}'.format(select_query)
        parser = Parser()
        mat_view_stmt = parser.parse(query)
        select_stmt = parser.parse(select_query)
        expected_stmt = CreateTableStatement(TableInfo('uadtrac_fastRCNN'), False, [], select_stmt[0])
        self.assertEqual(mat_view_stmt[0], expected_stmt)

    def test_join(self):
        select_query = 'SELECT table1.a FROM table1 JOIN table2\n                    ON table1.a = table2.a; '
        parser = Parser()
        select_stmt = parser.parse(select_query)[0]
        table1_col_a = TupleValueExpression('a', 'table1')
        table2_col_a = TupleValueExpression('a', 'table2')
        select_list = [table1_col_a]
        from_table = TableRef(JoinNode(TableRef(TableInfo('table1')), TableRef(TableInfo('table2')), predicate=ComparisonExpression(ExpressionType.COMPARE_EQUAL, table1_col_a, table2_col_a), join_type=JoinType.INNER_JOIN))
        expected_stmt = SelectStatement(select_list, from_table)
        self.assertEqual(select_stmt, expected_stmt)

    def test_join_with_where(self):
        select_query = 'SELECT table1.a FROM table1 JOIN table2\n            ON table1.a = table2.a WHERE table1.a <= 5'
        parser = Parser()
        select_stmt = parser.parse(select_query)[0]
        table1_col_a = TupleValueExpression('a', 'table1')
        table2_col_a = TupleValueExpression('a', 'table2')
        select_list = [table1_col_a]
        from_table = TableRef(JoinNode(TableRef(TableInfo('table1')), TableRef(TableInfo('table2')), predicate=ComparisonExpression(ExpressionType.COMPARE_EQUAL, table1_col_a, table2_col_a), join_type=JoinType.INNER_JOIN))
        where_clause = ComparisonExpression(ExpressionType.COMPARE_LEQ, table1_col_a, ConstantValueExpression(5))
        expected_stmt = SelectStatement(select_list, from_table, where_clause)
        self.assertEqual(select_stmt, expected_stmt)

    def test_multiple_join_with_multiple_ON(self):
        select_query = 'SELECT table1.a FROM table1 JOIN table2\n            ON table1.a = table2.a JOIN table3\n            ON table3.a = table1.a WHERE table1.a <= 5'
        parser = Parser()
        select_stmt = parser.parse(select_query)[0]
        table1_col_a = TupleValueExpression('a', 'table1')
        table2_col_a = TupleValueExpression('a', 'table2')
        table3_col_a = TupleValueExpression('a', 'table3')
        select_list = [table1_col_a]
        child_join = TableRef(JoinNode(TableRef(TableInfo('table1')), TableRef(TableInfo('table2')), predicate=ComparisonExpression(ExpressionType.COMPARE_EQUAL, table1_col_a, table2_col_a), join_type=JoinType.INNER_JOIN))
        from_table = TableRef(JoinNode(child_join, TableRef(TableInfo('table3')), predicate=ComparisonExpression(ExpressionType.COMPARE_EQUAL, table3_col_a, table1_col_a), join_type=JoinType.INNER_JOIN))
        where_clause = ComparisonExpression(ExpressionType.COMPARE_LEQ, table1_col_a, ConstantValueExpression(5))
        expected_stmt = SelectStatement(select_list, from_table, where_clause)
        self.assertEqual(select_stmt, expected_stmt)

    def test_lateral_join(self):
        select_query = 'SELECT frame FROM MyVideo JOIN LATERAL\n                            ObjectDet(frame) AS OD;'
        parser = Parser()
        select_stmt = parser.parse(select_query)[0]
        tuple_frame = TupleValueExpression('frame')
        func_expr = FunctionExpression(func=None, name='ObjectDet', children=[tuple_frame])
        from_table = TableRef(JoinNode(TableRef(TableInfo('MyVideo')), TableRef(TableValuedExpression(func_expr), alias=Alias('OD')), join_type=JoinType.LATERAL_JOIN))
        expected_stmt = SelectStatement([tuple_frame], from_table)
        self.assertEqual(select_stmt, expected_stmt)

    def test_class_equality(self):
        table_info = TableInfo('MyVideo')
        table_ref = TableRef(TableInfo('MyVideo'))
        tuple_frame = TupleValueExpression('frame')
        func_expr = FunctionExpression(func=None, name='ObjectDet', children=[tuple_frame])
        join_node = JoinNode(TableRef(TableInfo('MyVideo')), TableRef(TableValuedExpression(func_expr), alias=Alias('OD')), join_type=JoinType.LATERAL_JOIN)
        self.assertNotEqual(table_info, table_ref)
        self.assertNotEqual(tuple_frame, table_ref)
        self.assertNotEqual(join_node, table_ref)
        self.assertNotEqual(table_ref, table_info)

    def test_create_job(self):
        queries = ["CREATE OR REPLACE FUNCTION HomeSalesForecast FROM\n                ( SELECT * FROM postgres_data.home_sales )\n                TYPE Forecasting\n                PREDICT 'price';", 'Select HomeSalesForecast(10);']
        job_query = f"CREATE JOB my_job AS {{\n            {''.join(queries)}\n        }}\n        START '2023-04-01'\n        END '2023-05-01'\n        EVERY 2 hour\n        "
        parser = Parser()
        job_stmt = parser.parse(job_query)[0]
        self.assertEqual(job_stmt.job_name, 'my_job')
        self.assertEqual(len(job_stmt.queries), 2)
        self.assertTrue(queries[0].rstrip(';') == str(job_stmt.queries[0]))
        self.assertTrue(queries[1].rstrip(';') == str(job_stmt.queries[1]))
        self.assertEqual(job_stmt.start_time, '2023-04-01')
        self.assertEqual(job_stmt.end_time, '2023-05-01')
        self.assertEqual(job_stmt.repeat_interval, 2)
        self.assertEqual(job_stmt.repeat_period, 'hour')

class BatchTest(unittest.TestCase):

    def test_batch_serialize_deserialize(self):
        batch = Batch(frames=create_dataframe())
        batch2 = Batch.deserialize(batch.serialize())
        self.assertEqual(batch, batch2)

    def test_frames_as_numpy_array_should_frames_as_numpy_array(self):
        batch = Batch(frames=create_dataframe_same(2))
        expected = list(np.ones((2, 1, 1)))
        actual = list(batch.column_as_numpy_array(batch.columns[0]))
        self.assertEqual(expected, actual)

    def test_return_only_frames_specified_in_the_indices(self):
        batch = Batch(frames=create_dataframe(2))
        expected = Batch(frames=create_dataframe())
        output = batch[[0]]
        self.assertEqual(expected, output)

    def test_fetching_frames_by_index(self):
        batch = Batch(frames=create_dataframe_same(2))
        expected = Batch(frames=create_dataframe())
        self.assertEqual(expected, batch[0])

    def test_fetching_frames_by_index_should_raise(self):
        batch = Batch(frames=create_dataframe_same(2))
        with self.assertRaises(TypeError):
            batch[1.0]

    def test_slicing_on_batched_should_return_new_batch_frame(self):
        batch = Batch(frames=create_dataframe(2))
        expected = Batch(frames=create_dataframe())
        self.assertEqual(batch, batch[:])
        self.assertEqual(expected, batch[:-1])

    def test_slicing_should_word_for_negative_stop_value(self):
        batch = Batch(frames=create_dataframe(2))
        expected = Batch(frames=create_dataframe())
        self.assertEqual(expected, batch[:-1])

    def test_slicing_should_work_with_skip_value(self):
        batch = Batch(frames=create_dataframe(3))
        expected = Batch(frames=create_dataframe(3).iloc[[0, 2], :])
        self.assertEqual(expected, batch[::2])

    def test_add_should_raise_error_for_incompatible_type(self):
        batch = Batch(frames=create_dataframe())
        with self.assertRaises(TypeError):
            batch + 1

    def test_adding_to_empty_frame_batch_returns_itself(self):
        batch_1 = Batch(frames=pd.DataFrame())
        batch_2 = Batch(frames=create_dataframe())
        self.assertEqual(batch_2, batch_1 + batch_2)
        self.assertEqual(batch_2, batch_2 + batch_1)

    def test_adding_batch_frame_with_outcomes_returns_new_batch_frame(self):
        batch_1 = Batch(frames=create_dataframe())
        batch_2 = Batch(frames=create_dataframe())
        batch_3 = Batch(frames=create_dataframe_same(2))
        self.assertEqual(batch_3, batch_1 + batch_2)

    def test_concat_batch(self):
        batch_1 = Batch(frames=create_dataframe())
        batch_2 = Batch(frames=create_dataframe())
        batch_3 = Batch(frames=create_dataframe_same(2))
        self.assertEqual(batch_3, Batch.concat([batch_1, batch_2], copy=False))

    def test_concat_empty_batch_list_raise_exception(self):
        self.assertEqual(Batch(), Batch.concat([]))

    def test_project_batch_frame(self):
        batch_1 = Batch(frames=pd.DataFrame([{'id': 1, 'data': 2, 'info': 3}]))
        batch_2 = batch_1.project(['id', 'info'])
        batch_3 = Batch(frames=pd.DataFrame([{'id': 1, 'info': 3}]))
        self.assertEqual(batch_2, batch_3)

    def test_merge_column_wise_batch_frame(self):
        batch_1 = Batch(frames=pd.DataFrame([{'id': 0}]))
        batch_2 = Batch(frames=pd.DataFrame([{'data': 1}]))
        batch_3 = Batch.merge_column_wise([batch_1, batch_2])
        batch_4 = Batch(frames=pd.DataFrame([{'id': 0, 'data': 1}]))
        self.assertEqual(batch_3, batch_4)
        self.assertEqual(Batch.merge_column_wise([]), Batch())
        batch_1 = Batch(frames=pd.DataFrame({'id': [0, None, 1]}))
        batch_2 = Batch(frames=pd.DataFrame({'data': [None, 0, None]}))
        batch_res = Batch(frames=pd.DataFrame({'id': [0, None, 1], 'data': [None, 0, None]}))
        self.assertEqual(Batch.merge_column_wise([batch_1, batch_2]), batch_res)
        df_1 = pd.DataFrame({'id': [-10, 1, 2]})
        df_2 = pd.DataFrame({'data': [-20, 2, 3]})
        df_1 = df_1[df_1 < 0].dropna()
        df_1.reset_index(drop=True, inplace=True)
        df_2 = df_2[df_2 < 0].dropna()
        df_2.reset_index(drop=True, inplace=True)
        batch_1 = Batch(frames=df_1)
        batch_2 = Batch(frames=df_2)
        df_res = pd.DataFrame({'id': [-10, 1, 2], 'data': [-20, 2, 3]})
        df_res = df_res[df_res < 0].dropna()
        df_res.reset_index(drop=True, inplace=True)
        batch_res = Batch(frames=df_res)
        self.assertEqual(Batch.merge_column_wise([batch_1, batch_2]), batch_res)

    def test_should_fail_for_list(self):
        frames = [{'id': 0, 'data': [1, 2]}, {'id': 1, 'data': [1, 2]}]
        self.assertRaises(ValueError, Batch, frames)

    def test_should_fail_for_dict(self):
        frames = {'id': 0, 'data': [1, 2]}
        self.assertRaises(ValueError, Batch, frames)

    def test_should_return_correct_length(self):
        batch = Batch(create_dataframe(5))
        self.assertEqual(5, len(batch))

    def test_should_return_empty_dataframe(self):
        batch = Batch()
        self.assertEqual(batch, Batch(create_dataframe(0)))

    def test_stack_batch_more_than_one_column_should_raise_exception(self):
        batch = Batch(create_dataframe_same(2))
        self.assertRaises(ValueError, Batch.stack, batch)

    def test_modify_column_alias_should_raise_exception(self):
        batch = Batch(create_dataframe(5))
        dummy_alias = Alias('dummy', col_names=['t1'])
        with self.assertRaises(RuntimeError):
            batch.modify_column_alias(dummy_alias)

    def test_drop_column_alias_should_work_on_frame_without_alias(self):
        batch = Batch(create_dataframe(5))
        batch.drop_column_alias()

    def test_sort_orderby_should_raise_exception_on_missing_column(self):
        batch = Batch(create_dataframe(5))
        with self.assertRaises(AssertionError):
            batch.sort_orderby(by=['foo'])

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

def optimize_cache_key_for_tuple_value_expression(context: 'OptimizerContext', tv_expr: TupleValueExpression):
    catalog = context.db.catalog()
    col_catalog_obj = tv_expr.col_object
    new_keys = []
    if isinstance(col_catalog_obj, ColumnCatalogEntry):
        table_obj = catalog.get_table_catalog_entry(col_catalog_obj.table_name)
        for col in get_table_primary_columns(table_obj):
            new_obj = catalog.get_column_catalog_entry(table_obj, col.name)
            new_keys.append(TupleValueExpression(name=col.name, table_alias=tv_expr.table_alias, col_object=new_obj, col_alias=f'{tv_expr.table_alias}.{col.name}'))
        return new_keys
    return [tv_expr]

def get_table_primary_columns(table_catalog_obj: TableCatalogEntry) -> List[ColumnDefinition]:
    """
    Get the primary columns for a table based on its table type.

    Args:
        table_catalog_obj (TableCatalogEntry): The table catalog object.

    Returns:
        List[ColumnDefinition]: The list of primary columns for the table.
    """
    primary_columns = [ColumnDefinition(IDENTIFIER_COLUMN, ColumnType.INTEGER, None, None)]
    if table_catalog_obj.table_type == TableType.VIDEO_DATA:
        primary_columns.append(ColumnDefinition(VideoColumnName.id.name, ColumnType.INTEGER, None, None))
    elif table_catalog_obj.table_type == TableType.PDF_DATA:
        primary_columns.append(ColumnDefinition(PDFColumnName.paragraph.name, ColumnType.INTEGER, None, None))
    elif table_catalog_obj.table_type == TableType.DOCUMENT_DATA:
        primary_columns.append(ColumnDefinition(DocumentColumnName.chunk_id.name, ColumnType.INTEGER, None, None))
    return primary_columns

def resolve_alias_table_value_expression(node: FunctionExpression):
    default_alias_name = node.name.lower()
    default_output_col_aliases = [str(obj.name.lower()) for obj in node.output_objs]
    if not node.alias:
        node.alias = Alias(default_alias_name, default_output_col_aliases)
    elif not len(node.alias.col_names):
        node.alias = Alias(node.alias.alias_name, default_output_col_aliases)
    else:
        output_aliases = [str(col_name.lower()) for col_name in node.alias.col_names]
        node.alias = Alias(node.alias.alias_name, output_aliases)
    assert len(node.alias.col_names) == len(node.output_objs), f'Expected {len(node.output_objs)} output columns for {node.alias.alias_name}, got {len(node.alias.col_names)}.'

def handle_bind_extract_object_function(node: FunctionExpression, binder_context: StatementBinderContext):
    """Handles the binding of extract_object function.
        1. Bind the source video data
        2. Create and bind the detector function expression using the provided name.
        3. Create and bind the tracker function expression.
            Its inputs are id, data, output of detector.
        4. Bind the EXTRACT_OBJECT function expression and append the new children.
        5. Handle the alias and populate the outputs of the EXTRACT_OBJECT function

    Args:
        node (FunctionExpression): The function expression representing the extract object operation.
        binder_context (StatementBinderContext): The context object used to bind expressions in the statement.

    Raises:
        AssertionError: If the number of children in the `node` is not equal to 3.
    """
    assert len(node.children) == 3, f'Invalid arguments provided to {node}. Example correct usage, (data, Detector, Tracker)'
    video_data = node.children[0]
    binder_context.bind(video_data)
    detector = FunctionExpression(None, node.children[1].name)
    detector.append_child(video_data.copy())
    binder_context.bind(detector)
    tracker = FunctionExpression(None, node.children[2].name)
    columns = get_video_table_column_definitions()
    tracker.append_child(TupleValueExpression(name=columns[1].name, table_alias=video_data.table_alias))
    tracker.append_child(video_data.copy())
    binder_context.bind(tracker)
    for obj in detector.output_objs:
        col_alias = '{}.{}'.format(obj.function_name.lower(), obj.name.lower())
        child = TupleValueExpression(obj.name, table_alias=obj.function_name.lower(), col_object=obj, col_alias=col_alias)
        tracker.append_child(child)
    node.children = []
    node.children = [video_data, detector, tracker]
    node.output_objs = tracker.output_objs
    node.projection_columns = [obj.name.lower() for obj in node.output_objs]
    resolve_alias_table_value_expression(node)
    tracker.alias = node.alias

def handle_bind_extract_object_function(node: FunctionExpression, binder_context: StatementBinder):
    """Handles the binding of extract_object function.
        1. Bind the source video data
        2. Create and bind the detector function expression using the provided name.
        3. Create and bind the tracker function expression.
            Its inputs are id, data, output of detector.
        4. Bind the EXTRACT_OBJECT function expression and append the new children.
        5. Handle the alias and populate the outputs of the EXTRACT_OBJECT function
    Args:
        node (FunctionExpression): The function expression representing the extract object operation.
        binder_context (StatementBinder): The binder object used to bind expressions in the statement.
    Raises:
        AssertionError: If the number of children in the `node` is not equal to 3.
    """
    assert len(node.children) == 3, f'Invalid arguments provided to {node}. Example correct usage, (data, Detector, Tracker)'
    video_data = node.children[0]
    binder_context.bind(video_data)
    detector = FunctionExpression(None, node.children[1].name)
    detector.append_child(video_data.copy())
    binder_context.bind(detector)
    tracker = FunctionExpression(None, node.children[2].name)
    columns = get_video_table_column_definitions()
    tracker.append_child(TupleValueExpression(name=columns[1].name, table_alias=video_data.table_alias))
    tracker.append_child(video_data.copy())
    binder_context.bind(tracker)
    for obj in detector.output_objs:
        col_alias = '{}.{}'.format(obj.function_name.lower(), obj.name.lower())
        child = TupleValueExpression(obj.name, table_alias=obj.function_name.lower(), col_object=obj, col_alias=col_alias)
        tracker.append_child(child)
    node.children = []
    node.children = [video_data, detector, tracker]
    node.output_objs = tracker.output_objs
    node.projection_columns = [obj.name.lower() for obj in node.output_objs]
    resolve_alias_table_value_expression(node)
    tracker.alias = node.alias

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

def get_image_table_column_definitions() -> List[ColumnDefinition]:
    """
    name: image path
    data: image decoded data
    """
    columns = [ColumnDefinition(ImageColumnName.name.name, ColumnType.TEXT, None, None, ColConstraintInfo(unique=True)), ColumnDefinition(ImageColumnName.data.name, ColumnType.NDARRAY, NdArrayType.UINT8, (None, None, None))]
    return columns

def get_document_table_column_definitions() -> List[ColumnDefinition]:
    """
    name: file path
    chunk_id: chunk id (0-indexed for each file)
    data: text data associated with the chunk
    """
    columns = [ColumnDefinition(DocumentColumnName.name.name, ColumnType.TEXT, None, None, ColConstraintInfo(unique=True)), ColumnDefinition(DocumentColumnName.chunk_id.name, ColumnType.INTEGER, None, None), ColumnDefinition(DocumentColumnName.data.name, ColumnType.TEXT, None, None)]
    return columns

def get_pdf_table_column_definitions() -> List[ColumnDefinition]:
    """
    name: pdf name
    page: page no
    paragraph: paragraph no
    data: pdf paragraph data
    """
    columns = [ColumnDefinition(PDFColumnName.name.name, ColumnType.TEXT, None, None), ColumnDefinition(PDFColumnName.page.name, ColumnType.INTEGER, None, None), ColumnDefinition(PDFColumnName.paragraph.name, ColumnType.INTEGER, None, None), ColumnDefinition(PDFColumnName.data.name, ColumnType.TEXT, None, None)]
    return columns

class TableRef:
    """
    Attributes:
        : can be one of the following based on the query type:
            TableInfo: expression of table name and database name,
            TableValuedExpression: lateral function calls
            SelectStatement: select statement in case of nested queries,
            JoinNode: join node in case of join queries
        sample_freq: sampling frequency for the table reference
    """

    def __init__(self, table: Union[TableInfo, TableValuedExpression, SelectStatement, JoinNode], alias: Alias=None, sample_freq: float=None, sample_type: str=None, get_audio: bool=False, get_video: bool=False, chunk_params: dict={}):
        self._ref_handle = table
        self._sample_freq = sample_freq
        self._sample_type = sample_type
        self._get_audio = get_audio
        self._get_video = get_video
        self.chunk_params = chunk_params
        self.alias = alias or self.generate_alias()

    @property
    def sample_freq(self):
        return self._sample_freq

    @property
    def sample_type(self):
        return self._sample_type

    @property
    def get_audio(self):
        return self._get_audio

    @property
    def get_video(self):
        return self._get_video

    @get_audio.setter
    def get_audio(self, get_audio):
        self._get_audio = get_audio

    @get_video.setter
    def get_video(self, get_video):
        self._get_video = get_video

    def is_table_atom(self) -> bool:
        return isinstance(self._ref_handle, TableInfo)

    def is_table_valued_expr(self) -> bool:
        return isinstance(self._ref_handle, TableValuedExpression)

    def is_select(self) -> bool:
        return isinstance(self._ref_handle, SelectStatement)

    def is_join(self) -> bool:
        return isinstance(self._ref_handle, JoinNode)

    @property
    def ref_handle(self) -> Union[TableInfo, TableValuedExpression, SelectStatement, JoinNode]:
        return self._ref_handle

    @property
    def table(self) -> TableInfo:
        assert isinstance(self._ref_handle, TableInfo), 'Expected                 TableInfo, got {}'.format(type(self._ref_handle))
        return self._ref_handle

    @property
    def table_valued_expr(self) -> TableValuedExpression:
        assert isinstance(self._ref_handle, TableValuedExpression), 'Expected                 TableValuedExpression, got {}'.format(type(self._ref_handle))
        return self._ref_handle

    @property
    def join_node(self) -> JoinNode:
        assert isinstance(self._ref_handle, JoinNode), 'Expected                 JoinNode, got {}'.format(type(self._ref_handle))
        return self._ref_handle

    @property
    def select_statement(self) -> SelectStatement:
        assert isinstance(self._ref_handle, SelectStatement), 'Expected                 SelectStatement, got{}'.format(type(self._ref_handle))
        return self._ref_handle

    def generate_alias(self) -> Alias:
        if isinstance(self._ref_handle, TableInfo):
            return Alias(self._ref_handle.table_name.lower())

    def __str__(self):
        parts = []
        if self.is_select():
            parts.append(f'( {str(self._ref_handle)} ) AS {self.alias}')
        else:
            parts.append(str(self._ref_handle))
        if self.sample_freq is not None:
            parts.append(str(self.sample_freq))
        if self.sample_type is not None:
            parts.append(str(self.sample_type))
        if self.chunk_params is not None:
            parts.append(' '.join([f'{key}: {value}' for key, value in self.chunk_params.items()]))
        return ' '.join(parts)

    def __eq__(self, other):
        if not isinstance(other, TableRef):
            return False
        return self._ref_handle == other._ref_handle and self.alias == other.alias and (self.sample_freq == other.sample_freq) and (self.sample_type == other.sample_type) and (self.get_video == other.get_video) and (self.get_audio == other.get_audio) and (self.chunk_params == other.chunk_params)

    def __hash__(self) -> int:
        return hash((self._ref_handle, self.alias, self.sample_freq, self.sample_type, self.get_video, self.get_audio, frozenset(self.chunk_params.items())))

class Functions:

    def function(self, tree):
        function_name = None
        function_output = None
        function_args = []
        for child in tree.children:
            if isinstance(child, Token):
                if child.value == '*':
                    function_args = [TupleValueExpression(name='*')]
            if isinstance(child, Tree):
                if child.data == 'simple_id':
                    function_name = self.visit(child)
                elif child.data == 'dotted_id':
                    function_output = self.visit(child)
                elif child.data == 'function_args':
                    function_args = self.visit(child)
        func_expr = FunctionExpression(None, name=function_name, output=function_output)
        for arg in function_args:
            func_expr.append_child(arg)
        return func_expr

    def function_args(self, tree):
        args = []
        for child in tree.children:
            if isinstance(child, Tree):
                args.append(self.visit(child))
        return args

    def create_function(self, tree):
        function_name = None
        or_replace = False
        if_not_exists = False
        input_definitions = []
        output_definitions = []
        impl_path = None
        function_type = None
        query = None
        metadata = []
        create_definitions_index = 0
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'function_name':
                    function_name = self.visit(child)
                elif child.data == 'or_replace':
                    or_replace = True
                elif child.data == 'if_not_exists':
                    if_not_exists = True
                elif child.data == 'create_definitions':
                    if create_definitions_index == 0:
                        input_definitions = self.visit(child)
                        create_definitions_index += 1
                    elif create_definitions_index == 1:
                        output_definitions = self.visit(child)
                elif child.data == 'function_type':
                    function_type = self.visit(child)
                elif child.data == 'function_impl':
                    impl_path = self.visit(child).value
                elif child.data == 'simple_select':
                    query = self.visit(child)
                elif child.data == 'function_metadata':
                    key_value_pair = self.visit(child)
                    value = key_value_pair[1]
                    if isinstance(value, ConstantValueExpression):
                        value = value.value
                    (metadata.append((key_value_pair[0].lower(), value)),)
        return CreateFunctionStatement(function_name, or_replace, if_not_exists, impl_path, input_definitions, output_definitions, function_type, query, metadata)

    def get_aggregate_function_type(self, agg_func_name):
        agg_func_type = None
        if agg_func_name == 'COUNT':
            agg_func_type = ExpressionType.AGGREGATION_COUNT
        elif agg_func_name == 'MIN':
            agg_func_type = ExpressionType.AGGREGATION_MIN
        elif agg_func_name == 'MAX':
            agg_func_type = ExpressionType.AGGREGATION_MAX
        elif agg_func_name == 'SUM':
            agg_func_type = ExpressionType.AGGREGATION_SUM
        elif agg_func_name == 'AVG':
            agg_func_type = ExpressionType.AGGREGATION_AVG
        elif agg_func_name == 'FIRST':
            agg_func_type = ExpressionType.AGGREGATION_FIRST
        elif agg_func_name == 'LAST':
            agg_func_type = ExpressionType.AGGREGATION_LAST
        elif agg_func_name == 'SEGMENT':
            agg_func_type = ExpressionType.AGGREGATION_SEGMENT
        return agg_func_type

    def aggregate_windowed_function(self, tree):
        agg_func_arg = None
        agg_func_name = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'function_arg':
                    agg_func_arg = self.visit(child)
                elif child.data == 'aggregate_function_name':
                    agg_func_name = self.visit(child).value
            elif isinstance(child, Token):
                token = child.value
                if token != '*':
                    agg_func_name = token
                elif token == '*':
                    agg_func_arg = TupleValueExpression(name='_row_id')
                else:
                    agg_func_arg = TupleValueExpression(name='id')
        agg_func_type = self.get_aggregate_function_type(agg_func_name)
        agg_expr = AggregationExpression(agg_func_type, None, agg_func_arg)
        return agg_expr

class TableSources:

    def select_elements(self, tree):
        kind = tree.children[0]
        if kind == '*':
            select_list = [TupleValueExpression(name='*')]
        else:
            select_list = []
            for child in tree.children:
                element = self.visit(child)
                select_list.append(element)
        return select_list

    def table_sources(self, tree):
        return self.visit(tree.children[0])

    def table_source(self, tree):
        left_node = None
        join_nodes = []
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'table_source_item_with_param':
                    left_node = self.visit(child)
                    join_nodes = [left_node]
                elif child.data.endswith('join'):
                    table = self.visit(child)
                    join_nodes.append(table)
        num_table_joins = len(join_nodes)
        if num_table_joins > 1:
            for i in range(num_table_joins - 1):
                join_nodes[i + 1].join_node.left = join_nodes[i]
            return join_nodes[-1]
        else:
            return join_nodes[0]

    def table_source_item_with_param(self, tree):
        sample_freq = None
        sample_type = None
        alias = None
        table = None
        chunk_params = {}
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'table_source_item':
                    table = self.visit(child)
                elif child.data == 'sample_params':
                    sample_type, sample_freq = self.visit(child)
                elif child.data == 'chunk_params':
                    chunk_params = self.visit(child)
                elif child.data == 'alias_clause':
                    alias = self.visit(child)
        return TableRef(table=table, alias=alias, sample_freq=sample_freq, sample_type=sample_type, chunk_params=chunk_params)

    def table_source_item(self, tree):
        return self.visit(tree.children[0])

    def query_specification(self, tree):
        target_list = None
        from_clause = None
        where_clause = None
        groupby_clause = None
        orderby_clause = None
        limit_count = None
        for child in tree.children[1:]:
            try:
                if child.data == 'select_elements':
                    target_list = self.visit(child)
                elif child.data == 'from_clause':
                    clause = self.visit(child)
                    from_clause = clause.get('from', None)
                    where_clause = clause.get('where', None)
                    groupby_clause = clause.get('groupby', None)
                elif child.data == 'order_by_clause':
                    orderby_clause = self.visit(child)
                elif child.data == 'limit_clause':
                    limit_count = self.visit(child)
            except BaseException as e:
                logger.error('Error while parsing                                 QuerySpecification')
                raise e
        select_stmt = SelectStatement(target_list, from_clause, where_clause, groupby_clause=groupby_clause, orderby_list=orderby_clause, limit_count=limit_count)
        return select_stmt

    def from_clause(self, tree):
        from_table = None
        where_clause = None
        groupby_clause = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'table_sources':
                    from_table = self.visit(child)
                elif child.data == 'where_expr':
                    where_clause = self.visit(child)
                elif child.data == 'group_by_item':
                    groupby_item = self.visit(child)
                    groupby_clause = groupby_item
        return {'from': from_table, 'where': where_clause, 'groupby': groupby_clause}

    def inner_join(self, tree):
        table = None
        join_predicate = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'table_source_item_with_param':
                    table = self.visit(child)
                elif child.data.endswith('expression'):
                    join_predicate = self.visit(child)
        return TableRef(JoinNode(None, table, predicate=join_predicate, join_type=JoinType.INNER_JOIN))

    def lateral_join(self, tree):
        tve = None
        alias = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'table_valued_function':
                    tve = self.visit(child)
                elif child.data == 'alias_clause':
                    alias = self.visit(child)
        if alias is None:
            err_msg = f'TableValuedFunction {tve.func_expr.name} should have alias.'
            logger.error(err_msg)
            raise SyntaxError(err_msg)
        join_type = JoinType.LATERAL_JOIN
        return TableRef(JoinNode(None, TableRef(tve, alias=alias), join_type=join_type))

    def table_valued_function(self, tree):
        func_expr = None
        has_unnest = False
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data.endswith('function_call'):
                    func_expr = self.visit(child)
            elif child.lower() == 'unnest':
                has_unnest = True
        return TableValuedExpression(func_expr, do_unnest=has_unnest)

    def subquery_table_source_item(self, tree):
        subquery_table_source_item = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'simple_select':
                    subquery_table_source_item = self.visit(child)
        return subquery_table_source_item

    def union_select(self, tree):
        right_select_statement = None
        union_all = False
        statement_id = 0
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data.endswith('select'):
                    if statement_id == 0:
                        left_select_statement = self.visit(child)
                    elif statement_id == 1:
                        right_select_statement = self.visit(child)
                    statement_id += 1
            elif isinstance(child, Token):
                if child.value == 'ALL':
                    union_all = True
        if left_select_statement is not None:
            assert left_select_statement.union_link is None, 'Checking for the correctness of the operator'
            left_select_statement.union_link = right_select_statement
            if union_all is False:
                left_select_statement.union_all = False
            else:
                left_select_statement.union_all = True
        return left_select_statement

    def group_by_item(self, tree):
        expr = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data.endswith('expression'):
                    expr = self.visit(child)
        return expr

    def alias_clause(self, tree):
        alias_name = None
        column_list = []
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'uid':
                    alias_name = self.visit(child)
                elif child.data == 'uid_list':
                    column_list = self.visit(child)
                    column_list = [col.name for col in column_list]
        return Alias(alias_name, column_list)

class Load:

    def load_statement(self, tree):
        file_format = FileFormatType.VIDEO
        file_format = self.visit(tree.children[1])
        file_options = {}
        file_options['file_format'] = file_format
        file_path = self.visit(tree.children[2]).value
        table = self.visit(tree.children[4])
        column_list = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'uid_list':
                    column_list = self.visit(child)
        stmt = LoadDataStatement(table, file_path, column_list, file_options)
        return stmt

    def file_format(self, tree):
        file_format = None
        file_format_string = tree.children[0]
        if file_format_string == 'VIDEO':
            file_format = FileFormatType.VIDEO
        elif file_format_string == 'CSV':
            file_format = FileFormatType.CSV
        elif file_format_string == 'IMAGE':
            file_format = FileFormatType.IMAGE
        elif file_format_string == 'DOCUMENT':
            file_format = FileFormatType.DOCUMENT
        elif file_format_string == 'PDF':
            file_format = FileFormatType.PDF
        return file_format

    def file_options(self, tree):
        file_options = {}
        file_format = self.visit(tree.children[1])
        file_options['file_format'] = file_format
        return file_options

class CreateTable:

    def create_table(self, tree):
        table_info = None
        if_not_exists = False
        create_definitions = []
        query = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'if_not_exists':
                    if_not_exists = True
                elif child.data == 'table_name':
                    table_info = self.visit(child)
                elif child.data == 'create_definitions':
                    create_definitions = self.visit(child)
                elif child.data == 'simple_select':
                    query = self.visit(child)
        create_stmt = CreateTableStatement(table_info, if_not_exists, create_definitions, query=query)
        return create_stmt

    def create_definitions(self, tree):
        column_definitions = []
        for child in tree.children:
            if isinstance(child, Tree):
                create_definition = None
                if child.data == 'column_declaration':
                    create_definition = self.visit(child)
                column_definitions.append(create_definition)
        return column_definitions

    def column_declaration(self, tree):
        column_name = None
        data_type = None
        array_type = None
        dimensions = None
        column_constraint_information = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'uid':
                    column_name = self.visit(child)
                elif child.data == 'column_definition':
                    data_type, array_type, dimensions, column_constraint_information = self.visit(child)
        if column_name is not None:
            return ColumnDefinition(column_name, data_type, array_type, dimensions, column_constraint_information)

    def column_definition(self, tree):
        data_type = None
        array_type = None
        dimensions = None
        column_constraint_information = ColConstraintInfo()
        not_null_set = False
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data.endswith('data_type'):
                    data_type, array_type, dimensions = self.visit(child)
                elif child.data.endswith('column_constraint'):
                    return_type = self.visit(child)
                    if return_type == ColumnConstraintEnum.UNIQUE:
                        column_constraint_information.unique = True
                        column_constraint_information.nullable = False
                        not_null_set = True
                    elif return_type == ColumnConstraintEnum.NOTNULL:
                        column_constraint_information.nullable = False
                        not_null_set = True
        if not not_null_set:
            column_constraint_information.nullable = True
        return (data_type, array_type, dimensions, column_constraint_information)

    def unique_key_column_constraint(self, tree):
        return ColumnConstraintEnum.UNIQUE

    def null_column_constraint(self, tree):
        return ColumnConstraintEnum.NOTNULL

    def simple_data_type(self, tree):
        data_type = None
        array_type = None
        dimensions = []
        token = tree.children[0]
        if str.upper(token) == 'BOOLEAN':
            data_type = ColumnType.BOOLEAN
        return (data_type, array_type, dimensions)

    def integer_data_type(self, tree):
        data_type = None
        array_type = None
        dimensions = []
        token = tree.children[0]
        if str.upper(token) == 'INTEGER':
            data_type = ColumnType.INTEGER
        return (data_type, array_type, dimensions)

    def dimension_data_type(self, tree):
        data_type = None
        array_type = None
        dimensions = []
        token = tree.children[0]
        if str.upper(token) == 'FLOAT':
            data_type = ColumnType.FLOAT
        elif str.upper(token) == 'TEXT':
            data_type = ColumnType.TEXT
        if len(tree.children) > 1:
            dimensions = self.visit(tree.children[1])
        return (data_type, array_type, dimensions)

    def array_data_type(self, tree):
        data_type = ColumnType.NDARRAY
        array_type = NdArrayType.ANYTYPE
        dimensions = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'array_type':
                    array_type = self.visit(child)
                elif child.data == 'length_dimension_list':
                    dimensions = self.visit(child)
        return (data_type, array_type, dimensions)

    def any_data_type(self, tree):
        return (ColumnType.ANY, None, [])

    def array_type(self, tree):
        array_type = None
        token = tree.children[0]
        if str.upper(token) == 'INT8':
            array_type = NdArrayType.INT8
        elif str.upper(token) == 'UINT8':
            array_type = NdArrayType.UINT8
        elif str.upper(token) == 'INT16':
            array_type = NdArrayType.INT16
        elif str.upper(token) == 'INT32':
            array_type = NdArrayType.INT32
        elif str.upper(token) == 'INT64':
            array_type = NdArrayType.INT64
        elif str.upper(token) == 'UNICODE':
            array_type = NdArrayType.UNICODE
        elif str.upper(token) == 'BOOLEAN':
            array_type = NdArrayType.BOOL
        elif str.upper(token) == 'FLOAT32':
            array_type = NdArrayType.FLOAT32
        elif str.upper(token) == 'FLOAT64':
            array_type = NdArrayType.FLOAT64
        elif str.upper(token) == 'DECIMAL':
            array_type = NdArrayType.DECIMAL
        elif str.upper(token) == 'STR':
            array_type = NdArrayType.STR
        elif str.upper(token) == 'DATETIME':
            array_type = NdArrayType.DATETIME
        elif str.upper(token) == 'ANYTYPE':
            array_type = NdArrayType.ANYTYPE
        return array_type

    def dimension_helper(self, tree):
        dimensions = []
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'decimal_literal':
                    decimal = self.visit(child)
                    dimensions.append(decimal)
        return tuple(dimensions)

    def length_one_dimension(self, tree):
        dimensions = self.dimension_helper(tree)
        return dimensions

    def length_two_dimension(self, tree):
        dimensions = self.dimension_helper(tree)
        return dimensions

    def length_dimension_list(self, tree):
        dimensions = self.dimension_helper(tree)
        return dimensions

class CreateIndex:

    def create_index(self, tree):
        index_name = None
        if_not_exists = False
        table_name = None
        vector_store_type = None
        index_elem = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'uid':
                    index_name = self.visit(child)
                if child.data == 'if_not_exists':
                    if_not_exists = True
                elif child.data == 'table_name':
                    table_name = self.visit(child)
                    table_ref = TableRef(table_name)
                elif child.data == 'vector_store_type':
                    vector_store_type = self.visit(child)
                elif child.data == 'index_elem':
                    index_elem = self.visit(child)
        project_expr_list = []
        if not isinstance(index_elem, list):
            project_expr_list += [index_elem]
            while not isinstance(index_elem, TupleValueExpression):
                index_elem = index_elem.children[0]
            index_elem = [index_elem]
        else:
            project_expr_list += index_elem
        col_list = []
        for tv_expr in index_elem:
            col_list += [ColumnDefinition(tv_expr.name, None, None, None)]
        return CreateIndexStatement(index_name, if_not_exists, table_ref, col_list, vector_store_type, project_expr_list)

    def vector_store_type(self, tree):
        vector_store_type = None
        token = tree.children[1]
        if str.upper(token) == 'FAISS':
            vector_store_type = VectorStoreType.FAISS
        elif str.upper(token) == 'QDRANT':
            vector_store_type = VectorStoreType.QDRANT
        elif str.upper(token) == 'PINECONE':
            vector_store_type = VectorStoreType.PINECONE
        elif str.upper(token) == 'PGVECTOR':
            vector_store_type = VectorStoreType.PGVECTOR
        elif str.upper(token) == 'CHROMADB':
            vector_store_type = VectorStoreType.CHROMADB
        elif str.upper(token) == 'WEAVIATE':
            vector_store_type = VectorStoreType.WEAVIATE
        elif str.upper(token) == 'MILVUS':
            vector_store_type = VectorStoreType.MILVUS
        return vector_store_type

class RenameTable:

    def rename_table(self, tree):
        old_table_info = self.visit(tree.children[2])
        new_table_info = self.visit(tree.children[4])
        return RenameTableStatement(TableRef(old_table_info), new_table_info)

class Delete:

    def delete_statement(self, tree):
        table_ref = None
        where_clause = None
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'table_name':
                    table_name = self.visit(child)
                    table_ref = TableRef(table_name)
                elif child.data == 'where_expr':
                    where_clause = self.visit(child)
        delete_stmt = DeleteTableStatement(table_ref, where_clause)
        return delete_stmt

class Insert:

    def insert_statement(self, tree):
        table_ref = None
        column_list = []
        value_list = []
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'table_name':
                    table_name = self.visit(child)
                    table_ref = TableRef(table_name)
                elif child.data == 'uid_list':
                    column_list = self.visit(child)
                elif child.data == 'insert_statement_value':
                    value_list = self.visit(child)
        insert_stmt = InsertTableStatement(table_ref, column_list, value_list)
        return insert_stmt

    def uid_list(self, tree):
        uid_expr_list = []
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'uid':
                    uid = self.visit(child)
                    uid_expr = TupleValueExpression(uid)
                    uid_expr_list.append(uid_expr)
        return uid_expr_list

    def insert_statement_value(self, tree):
        insert_stmt_value = []
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'expressions_with_defaults':
                    expr = self.visit(child)
                    insert_stmt_value.append(expr)
        return insert_stmt_value

def create_star_expression():
    return [TupleValueExpression(name='*')]

def handle_select_clause(query: SelectStatement, alias: str, clause: str, value: Union[str, int, list]) -> SelectStatement:
    """
    Modifies a SELECT statement object by adding or modifying a specific clause.

    Args:
        query (SelectStatement): The SELECT statement object.
        alias (str): Alias for the table reference.
        clause (str): The clause to be handled.
        value (str, int, list): The value to be set for the clause.

    Returns:
        SelectStatement: The modified SELECT statement object.

    Raises:
        AssertionError: If the query is not an instance of SelectStatement class.
        AssertionError: If the clause is not in the accepted clauses list.
    """
    assert isinstance(query, SelectStatement), 'query must be an instance of SelectStatement'
    accepted_clauses = ['where_clause', 'target_list', 'groupby_clause', 'orderby_list', 'limit_count']
    assert clause in accepted_clauses, f'Unknown clause: {clause}'
    if clause == 'target_list' and getattr(query, clause) == create_star_expression():
        setattr(query, clause, None)
    if getattr(query, clause) is None:
        setattr(query, clause, value)
    else:
        query = SelectStatement(target_list=create_star_expression(), from_table=TableRef(query, alias=alias))
        setattr(query, clause, value)
    return query

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

class Batch:
    """
    Data model used for storing a batch of frames.
    Internally stored as a pandas DataFrame with columns
    "id" and "data".
    id: integer index of frame
    data: frame as np.array

    Arguments:
        frames (DataFrame): pandas Dataframe holding frames data
    """

    def __init__(self, frames=None):
        self._frames = pd.DataFrame() if frames is None else frames
        if not isinstance(self._frames, pd.DataFrame):
            raise ValueError(f'Batch constructor not properly called.\nExpected pandas.DataFrame, got {type(self._frames)}')

    @property
    def frames(self) -> pd.DataFrame:
        return self._frames

    def __len__(self):
        return len(self._frames)

    @property
    def columns(self):
        return list(self._frames.columns)

    def column_as_numpy_array(self, column_name: str) -> np.ndarray:
        """Return a column as numpy array

        Args:
            column_name (str): the name of the required column

        Returns:
            numpy.ndarray: the column data as a numpy array
        """
        return self._frames[column_name].to_numpy()

    def serialize(self):
        obj = {'frames': self._frames, 'batch_size': len(self)}
        return PickleSerializer.serialize(obj)

    @classmethod
    def deserialize(cls, data):
        obj = PickleSerializer.deserialize(data)
        return cls(frames=obj['frames'])

    @classmethod
    def from_eq(cls, batch1: Batch, batch2: Batch) -> Batch:
        return Batch(pd.DataFrame(batch1.to_numpy() == batch2.to_numpy()))

    @classmethod
    def from_greater(cls, batch1: Batch, batch2: Batch) -> Batch:
        return Batch(pd.DataFrame(batch1.to_numpy() > batch2.to_numpy()))

    @classmethod
    def from_lesser(cls, batch1: Batch, batch2: Batch) -> Batch:
        return Batch(pd.DataFrame(batch1.to_numpy() < batch2.to_numpy()))

    @classmethod
    def from_greater_eq(cls, batch1: Batch, batch2: Batch) -> Batch:
        return Batch(pd.DataFrame(batch1.to_numpy() >= batch2.to_numpy()))

    @classmethod
    def from_lesser_eq(cls, batch1: Batch, batch2: Batch) -> Batch:
        return Batch(pd.DataFrame(batch1.to_numpy() <= batch2.to_numpy()))

    @classmethod
    def from_not_eq(cls, batch1: Batch, batch2: Batch) -> Batch:
        return Batch(pd.DataFrame(batch1.to_numpy() != batch2.to_numpy()))

    @classmethod
    def compare_contains(cls, batch1: Batch, batch2: Batch) -> None:
        return cls(pd.DataFrame(([all((x in p for x in q)) for p, q in zip(left, right)] for left, right in zip(batch1.to_numpy(), batch2.to_numpy()))))

    @classmethod
    def compare_is_contained(cls, batch1: Batch, batch2: Batch) -> None:
        return cls(pd.DataFrame(([all((x in q for x in p)) for p, q in zip(left, right)] for left, right in zip(batch1.to_numpy(), batch2.to_numpy()))))

    @classmethod
    def compare_like(cls, batch1: Batch, batch2: Batch) -> None:
        col = batch1._frames.iloc[:, 0]
        regex = batch2._frames.iloc[:, 0][0]
        return cls(pd.DataFrame(col.astype('str').str.match(pat=regex)))

    def __str__(self) -> str:
        with pd.option_context('display.pprint_nest_depth', 1, 'display.max_colwidth', 100):
            return f'{self._frames}'

    def __eq__(self, other: Batch):
        return self._frames[sorted(self.columns)].equals(other.frames[sorted(other.columns)])

    def __getitem__(self, indices) -> Batch:
        """
        Returns a batch with the desired frames

        Arguments:
            indices (list, slice or mask): list must be
            a list of indices; mask is boolean array-like
            (i.e. list, NumPy array, DataFrame, etc.)
            of appropriate size with True for desired frames.
        """
        if isinstance(indices, list):
            return self._get_frames_from_indices(indices)
        elif isinstance(indices, slice):
            start = indices.start if indices.start else 0
            end = indices.stop if indices.stop else len(self.frames)
            if end < 0:
                end = len(self._frames) + end
            step = indices.step if indices.step else 1
            return self._get_frames_from_indices(range(start, end, step))
        elif isinstance(indices, int):
            return self._get_frames_from_indices([indices])
        else:
            raise TypeError('Invalid argument type: {}'.format(type(indices)))

    def _get_frames_from_indices(self, required_frame_ids):
        new_frames = self._frames.iloc[required_frame_ids, :]
        new_batch = Batch(new_frames)
        return new_batch

    def apply_function_expression(self, expr: Callable) -> Batch:
        """
        Execute function expression on frames.
        """
        self.drop_column_alias()
        return Batch(expr(self._frames))

    def iterrows(self):
        return self._frames.iterrows()

    def sort(self, by=None) -> None:
        """
        in_place sort
        """
        if self.empty():
            return
        if by is None:
            by = self.columns[0]
        self._frames.sort_values(by=by, ignore_index=True, inplace=True)

    def sort_orderby(self, by, sort_type=None) -> None:
        """
        in_place sort for order_by

        Args:
            by: list of column names
            sort_type: list of True/False if ASC for each column name in 'by'
                i.e [True, False] means [ASC, DESC]
        """
        if sort_type is None:
            sort_type = [True]
        assert by is not None
        for column in by:
            assert column in self._frames.columns, 'Can not orderby non-projected column: {}'.format(column)
        self._frames.sort_values(by, ascending=sort_type, ignore_index=True, inplace=True)

    def invert(self) -> None:
        self._frames = ~self._frames

    def all_true(self) -> bool:
        return self._frames.all().bool()

    def all_false(self) -> bool:
        inverted = ~self._frames
        return inverted.all().bool()

    def create_mask(self) -> List:
        """
        Return list of indices of first row.
        """
        return self._frames[self._frames[0]].index.tolist()

    def create_inverted_mask(self) -> List:
        return self._frames[~self._frames[0]].index.tolist()

    def update_indices(self, indices: List, other: Batch):
        self._frames.iloc[indices] = other._frames
        self._frames = pd.DataFrame(self._frames)

    def file_paths(self) -> Iterable:
        yield from self._frames['file_path']

    def project(self, cols: None) -> Batch:
        """
        Takes as input the column list, returns the projection.
        We do a copy for now.
        """
        cols = cols or []
        verified_cols = [c for c in cols if c in self._frames]
        unknown_cols = list(set(cols) - set(verified_cols))
        assert len(unknown_cols) == 0, unknown_cols
        return Batch(self._frames[verified_cols])

    @classmethod
    def merge_column_wise(cls, batches: List[Batch], auto_renaming=False) -> Batch:
        """
        Merge list of batch frames column_wise and return a new batch frame
        Arguments:
            batches: List[Batch]: list of batch objects to be merged
            auto_renaming: if true rename column names if required

        Returns:
            Batch: Merged batch object
        """
        if not len(batches):
            return Batch()
        frames = [batch.frames for batch in batches]
        frames_index = [list(frame.index) for frame in frames]
        for i, frame_index in enumerate(frames_index):
            assert frame_index == frames_index[i - 1], 'Merging of DataFrames with unmatched indices can cause undefined behavior'
        new_frames = pd.concat(frames, axis=1, copy=False, ignore_index=False)
        if new_frames.columns.duplicated().any():
            logger.debug('Duplicated column name detected {}'.format(new_frames))
        return Batch(new_frames)

    def __add__(self, other: Batch) -> Batch:
        """
        Adds two batch frames and return a new batch frame
        Arguments:
            other (Batch): other framebatch to add

        Returns:
            Batch
        """
        if not isinstance(other, Batch):
            raise TypeError('Input should be of type Batch')
        if self.empty():
            return other
        if other.empty():
            return self
        return Batch.concat([self, other], copy=False)

    @classmethod
    def concat(cls, batch_list: Iterable[Batch], copy=True) -> Batch:
        """Concat a list of batches.
        Notice: only frames are considered.
        """
        frame_list = list([batch.frames for batch in batch_list])
        if len(frame_list) == 0:
            return Batch()
        frame = pd.concat(frame_list, ignore_index=True, copy=copy)
        return Batch(frame)

    @classmethod
    def stack(cls, batch: Batch, copy=True) -> Batch:
        """Stack a given batch along the 0th dimension.
        Notice: input assumed to contain only one column with video frames

        Returns:
            Batch (always of length 1)
        """
        if len(batch.columns) > 1:
            raise ValueError('Stack can only be called on single-column batches')
        frame_data_col = batch.columns[0]
        data_to_stack = batch.frames[frame_data_col].values.tolist()
        if isinstance(data_to_stack[0], np.ndarray) and len(data_to_stack[0].shape) > 1:
            stacked_array = np.array(batch.frames[frame_data_col].values.tolist())
        else:
            stacked_array = np.hstack(batch.frames[frame_data_col].values)
        stacked_frame = pd.DataFrame([{frame_data_col: stacked_array}])
        return Batch(stacked_frame)

    @classmethod
    def join(cls, first: Batch, second: Batch, how='inner') -> Batch:
        return cls(first._frames.merge(second._frames, left_index=True, right_index=True, how=how))

    @classmethod
    def combine_batches(cls, first: Batch, second: Batch, expression: ExpressionType) -> Batch:
        """
        Creates Batch by combining two batches using some arithmetic expression.
        """
        if expression == ExpressionType.ARITHMETIC_ADD:
            return Batch(pd.DataFrame(first._frames + second._frames))
        elif expression == ExpressionType.ARITHMETIC_SUBTRACT:
            return Batch(pd.DataFrame(first._frames - second._frames))
        elif expression == ExpressionType.ARITHMETIC_MULTIPLY:
            return Batch(pd.DataFrame(first._frames * second._frames))
        elif expression == ExpressionType.ARITHMETIC_DIVIDE:
            return Batch(pd.DataFrame(first._frames / second._frames))

    def reassign_indices_to_hash(self, indices) -> None:
        """
        Hash indices and replace the indices with those hash values.
        """
        self._frames.index = self._frames[indices].apply(lambda x: hash(tuple(x)), axis=1)

    def aggregate(self, method: str) -> None:
        """
        Aggregate batch based on method.
        Methods can be sum, count, min, max, mean

        Arguments:
            method: string with one of the five above options
        """
        self._frames = self._frames.agg([method])

    def empty(self):
        """Checks if the batch is empty
        Returns:
            True if the batch_size == 0
        """
        return len(self) == 0

    def unnest(self, cols: List[str]=None) -> None:
        """
        Unnest columns and drop columns with no data
        """
        if cols is None:
            cols = list(self.columns)
        self._frames = self._frames.explode(cols)
        self._frames.dropna(inplace=True)

    def reverse(self) -> None:
        """Reverses dataframe"""
        self._frames = self._frames[::-1]
        self._frames.reset_index(drop=True, inplace=True)

    def drop_zero(self, outcomes: Batch) -> None:
        """Drop all columns with corresponding outcomes containing zero."""
        self._frames = self._frames[(outcomes._frames > 0).to_numpy()]

    def reset_index(self):
        """Resets the index of the data frame in the batch"""
        self._frames.reset_index(drop=True, inplace=True)

    def modify_column_alias(self, alias: Union[Alias, str]) -> None:
        if isinstance(alias, str):
            alias = Alias(alias)
        new_col_names = []
        if len(alias.col_names):
            if len(self.columns) != len(alias.col_names):
                err_msg = f'Expected {len(alias.col_names)} columns {alias.col_names},got {len(self.columns)} columns {self.columns}.'
                raise RuntimeError(err_msg)
            new_col_names = ['{}.{}'.format(alias.alias_name, col_name) for col_name in alias.col_names]
        else:
            for col_name in self.columns:
                if '.' in str(col_name):
                    new_col_names.append('{}.{}'.format(alias.alias_name, str(col_name).split('.')[1]))
                else:
                    new_col_names.append('{}.{}'.format(alias.alias_name, col_name))
        self._frames.columns = new_col_names

    def drop_column_alias(self) -> None:
        new_col_names = []
        for col_name in self.columns:
            if isinstance(col_name, str) and '.' in col_name:
                new_col_names.append(col_name.split('.')[1])
            else:
                new_col_names.append(col_name)
        self._frames.columns = new_col_names

    def to_numpy(self):
        return self._frames.to_numpy()

    def rename(self, columns) -> None:
        """Rename column names"""
        self._frames.rename(columns=columns, inplace=True)

