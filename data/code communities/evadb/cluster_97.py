# Cluster 97

class JobSchedulerIntegrationTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        cls.job_name_1 = 'test_async_job_1'
        cls.job_name_2 = 'test_async_job_2'

    def setUp(self):
        execute_query_fetch_all(self.evadb, f'DROP JOB IF EXISTS {self.job_name_1};')
        execute_query_fetch_all(self.evadb, f'DROP JOB IF EXISTS {self.job_name_2};')

    @classmethod
    def tearDownClass(cls):
        shutdown_ray()
        execute_query_fetch_all(cls.evadb, f'DROP JOB IF EXISTS {cls.job_name_1};')
        execute_query_fetch_all(cls.evadb, f'DROP JOB IF EXISTS {cls.job_name_2};')

    def create_jobs(self):
        datetime_format = '%Y-%m-%d %H:%M:%S'
        start_time = (datetime.now() - timedelta(seconds=10)).strftime(datetime_format)
        end_time = (datetime.now() + timedelta(seconds=60)).strftime(datetime_format)
        create_csv_query = 'CREATE TABLE IF NOT EXISTS MyCSV (\n                                    id INTEGER UNIQUE,\n                                    frame_id INTEGER,\n                                    video_id INTEGER\n                                );\n                            '
        job_1_query = f"CREATE JOB IF NOT EXISTS {self.job_name_1} AS {{\n                                SELECT * FROM MyCSV;\n                            }}\n                            START '{start_time}'\n                            END '{end_time}'\n                            EVERY 4 seconds;\n                        "
        job_2_query = f"CREATE JOB IF NOT EXISTS {self.job_name_2} AS {{\n                            SHOW FUNCTIONS;\n                        }}\n                        START '{start_time}'\n                        END '{end_time}'\n                        EVERY 2 seconds;\n                    "
        execute_query_fetch_all(self.evadb, create_csv_query)
        execute_query_fetch_all(self.evadb, job_1_query)
        execute_query_fetch_all(self.evadb, job_2_query)

    def test_should_execute_the_scheduled_jobs(self):
        self.create_jobs()
        connection = EvaDBConnection(self.evadb, MagicMock(), MagicMock())
        connection.start_jobs()
        time.sleep(15)
        connection.stop_jobs()
        job_1_execution_count = len(self.evadb.catalog().get_job_history_by_job_id(1))
        job_2_execution_count = len(self.evadb.catalog().get_job_history_by_job_id(2))
        self.assertGreater(job_2_execution_count, job_1_execution_count)
        self.assertGreater(job_2_execution_count, 2)
        self.assertGreater(job_1_execution_count, 2)

class ModulePathTest(unittest.TestCase):

    def test_helper_validates_kwargs(self):
        with self.assertRaises(TypeError):
            validate_kwargs({'a': 1, 'b': 2}, ['a'], 'Invalid keyword argument:')

    def test_should_return_correct_class_for_string(self):
        vl = str_to_class('evadb.readers.decord_reader.DecordReader')
        self.assertEqual(vl, DecordReader)

    def test_should_return_correct_class_for_path(self):
        vl = load_function_class_from_file('evadb/readers/decord_reader.py', 'DecordReader')
        assert vl.__qualname__ == DecordReader.__qualname__

    def test_should_return_correct_class_for_path_without_classname(self):
        vl = load_function_class_from_file('evadb/readers/decord_reader.py')
        assert vl.__qualname__ == DecordReader.__qualname__

    def test_should_raise_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_function_class_from_file('evadb/readers/opencv_reader_abdfdsfds.py')

    def test_should_raise_on_empty_file(self):
        Path('/tmp/empty_file.py').touch()
        with self.assertRaises(ImportError):
            load_function_class_from_file('/tmp/empty_file.py')
        Path('/tmp/empty_file.py').unlink()

    def test_should_raise_if_class_does_not_exists(self):
        with self.assertRaises(ImportError):
            load_function_class_from_file('evadb/utils/s3_utils.py')

    def test_should_raise_if_multiple_classes_exist_and_no_class_mentioned(self):
        with self.assertRaises(ImportError):
            load_function_class_from_file('evadb/utils/generic_utils.py')

    def test_should_use_torch_to_check_if_gpu_is_available(self):
        try:
            import builtins
        except ImportError:
            import __builtin__ as builtins
        realimport = builtins.__import__

        def missing_import(name, globals, locals, fromlist, level):
            if name == 'torch':
                raise ImportError
            return realimport(name, globals, locals, fromlist, level)
        builtins.__import__ = missing_import
        self.assertFalse(is_gpu_available())
        builtins.__import__ = realimport
        is_gpu_available()

    @windows_skip_marker
    def test_should_return_a_random_full_path(self):
        actual = generate_file_path(EvaDB_DATASET_DIR, 'test')
        self.assertTrue(actual.is_absolute())
        self.assertTrue(EvaDB_DATASET_DIR in str(actual.parent))

def load_function_class_from_file(filepath, classname=None):
    """
    Load a class from a Python file. If the classname is not specified, the function will check if there is only one class in the file and load that. If there are multiple classes, it will raise an error.

    Args:
        filepath (str): The path to the Python file.
        classname (str, optional): The name of the class to load. If not specified, the function will try to load a class with the same name as the file. Defaults to None.

    Returns:
        The class instance.

    Raises:
        ImportError: If the module cannot be loaded.
        FileNotFoundError: If the file cannot be found.
        RuntimeError: Any othe type of runtime error.
    """
    try:
        abs_path = Path(filepath).resolve()
        spec = importlib.util.spec_from_file_location(abs_path.stem, abs_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as e:
        err_msg = f"ImportError : Couldn't load function from {filepath} : {str(e)}. Not able to load the code provided in the file {abs_path}. Please ensure that the file contains the implementation code for the function."
        raise ImportError(err_msg)
    except FileNotFoundError as e:
        err_msg = f"FileNotFoundError : Couldn't load function from {filepath} : {str(e)}. This might be because the function implementation file does not exist. Please ensure the file exists at {abs_path}"
        raise FileNotFoundError(err_msg)
    except Exception as e:
        err_msg = f"Couldn't load function from {filepath} : {str(e)}."
        raise RuntimeError(err_msg)
    if classname and hasattr(module, classname):
        return getattr(module, classname)
    classes = [obj for _, obj in inspect.getmembers(module, inspect.isclass) if obj.__module__ == module.__name__]
    if len(classes) != 1:
        raise ImportError(f'{filepath} contains {len(classes)} classes, please specify the correct class to load by naming the function with the same name in the CREATE query.')
    return classes[0]

class GenericUtilsTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_string_matching_case_insensitive(self):
        """
        A simple test for string_matching_case_insensitive in generic_utils
        used by statement_binder
        """
        test_string_exact_match = string_comparison_case_insensitive('HuggingFace', 'HuggingFace')
        test_string_case_insensitive_match = string_comparison_case_insensitive('HuggingFace', 'hugGingFaCe')
        test_string_no_match = string_comparison_case_insensitive('HuggingFace', 'HuggingFae')
        test_one_string_null = string_comparison_case_insensitive(None, 'HuggingFace')
        test_both_strings_null = string_comparison_case_insensitive(None, None)
        self.assertTrue(test_string_exact_match)
        self.assertTrue(test_string_case_insensitive_match)
        self.assertFalse(test_string_no_match)
        self.assertFalse(test_one_string_null)
        self.assertFalse(test_both_strings_null)

def string_comparison_case_insensitive(string_1, string_2) -> bool:
    """
    Case insensitive string comparison for two strings which gives
    a bool response whether the strings are the same or not

    Arguments:
        string_1 (str)
        string_2 (str)

    Returns:
        True/False (bool): Returns True if the strings are same, false otherwise
    """
    if string_1 is None or string_2 is None:
        return False
    return string_1.lower() == string_2.lower()

class Memo:
    """
    For now, we assume every group has only one logic expression.
    """

    def __init__(self):
        self._group_exprs: Dict[int, GroupExpression] = dict()
        self._groups = dict()

    @property
    def groups(self):
        return self._groups

    @property
    def group_exprs(self):
        return self._group_exprs

    def find_duplicate_expr(self, expr: GroupExpression) -> GroupExpression:
        if hash(expr) in self.group_exprs:
            return self.group_exprs[hash(expr)]
        else:
            return None

    def get_group_by_id(self, group_id: int) -> GroupExpression:
        if group_id in self._groups.keys():
            return self._groups[group_id]
        else:
            logger.error('Missing group id')
    '\n    For the consistency of the memo, all modification should use the\n    following functions.\n    '

    def _get_table_aliases(self, expr: GroupExpression) -> List[str]:
        """
        Collects table aliases of all the children
        """
        aliases = []
        for child_grp_id in expr.children:
            child_grp = self._groups[child_grp_id]
            aliases.extend(child_grp.aliases)
        if expr.opr.opr_type == OperatorType.LOGICALGET or expr.opr.opr_type == OperatorType.LOGICALQUERYDERIVEDGET:
            aliases.append(expr.opr.alias)
        elif expr.opr.opr_type == OperatorType.LOGICALFUNCTIONSCAN:
            aliases.append(expr.opr.alias)
        return aliases

    def _create_new_group(self, expr: GroupExpression):
        """
        Create new group for the expr
        """
        new_group_id = len(self._groups)
        aliases = self._get_table_aliases(expr)
        self._groups[new_group_id] = Group(new_group_id, aliases)
        self._insert_expr(expr, new_group_id)

    def _insert_expr(self, expr: GroupExpression, group_id: int):
        """
        Insert a group expression into a particular group
        """
        assert group_id < len(self.groups), 'Group Id out of the bound'
        group = self.groups[group_id]
        group.add_expr(expr)
        self._group_exprs[hash(expr)] = expr

    def erase_group(self, group_id: int):
        """
        Remove all the expr from the group_id
        """
        group = self.groups[group_id]
        for expr in group.logical_exprs:
            del self._group_exprs[hash(expr)]
        for expr in group.physical_exprs:
            del self._group_exprs[hash(expr)]
        group.clear_grp_exprs()

    def add_group_expr(self, expr: GroupExpression, group_id: int=UNDEFINED_GROUP_ID) -> GroupExpression:
        """
        Add an expression into the memo.
        If expr exists, we return it.
        If group_id is not specified, creates a new group
        Otherwise, inserts the expr into specified group.
        """
        duplicate_expr = self.find_duplicate_expr(expr)
        if duplicate_expr is not None:
            return duplicate_expr
        expr.group_id = group_id
        if expr.group_id == UNDEFINED_GROUP_ID:
            self._create_new_group(expr)
        else:
            self._insert_expr(expr, group_id)
        assert expr.group_id is not UNDEFINED_GROUP_ID, 'Expr should have a valid group id'
        return expr

class Group:

    def __init__(self, group_id: int, aliases: List[str]=None):
        self._group_id = group_id
        self._aliases = aliases
        self._logical_exprs = []
        self._physical_exprs = []
        self._winner_exprs: Dict[Property, Winner] = {}
        self._is_explored = False

    @property
    def group_id(self):
        return self._group_id

    @property
    def aliases(self):
        return self._aliases

    @property
    def logical_exprs(self):
        return self._logical_exprs

    @property
    def physical_exprs(self):
        return self._physical_exprs

    def is_explored(self):
        return self._is_explored

    def mark_explored(self):
        self._is_explored = True

    def __str__(self) -> str:
        return '%s(%s)' % (type(self).__name__, ', '.join(('%s=%s' % item for item in vars(self).items())))

    def add_expr(self, expr: GroupExpression):
        if expr.group_id == UNDEFINED_GROUP_ID:
            expr.group_id = self.group_id
        if expr.group_id != self.group_id:
            logger.error('Expected group id {}, found {}'.format(self.group_id, expr.group_id))
            return
        if expr.opr.is_logical():
            self._add_logical_expr(expr)
        else:
            self._add_physical_expr(expr)

    def get_best_expr(self, property: Property) -> GroupExpression:
        winner = self._winner_exprs.get(property, None)
        if winner:
            return winner.grp_expr
        else:
            return None

    def get_best_expr_cost(self, property: Property):
        winner = self._winner_exprs.get(property, None)
        if winner:
            return winner.cost
        else:
            return None

    def add_expr_cost(self, expr: GroupExpression, property, cost):
        existing_winner = self._winner_exprs.get(property, None)
        if not existing_winner or existing_winner.cost > cost:
            self._winner_exprs[property] = Winner(expr, cost)

    def clear_grp_exprs(self):
        self._logical_exprs.clear()
        self._physical_exprs.clear()

    def _add_logical_expr(self, expr: GroupExpression):
        self._logical_exprs.append(expr)

    def _add_physical_expr(self, expr: GroupExpression):
        self._physical_exprs.append(expr)

def get_bound_func_expr_outputs_as_tuple_value_expr(func_expr: FunctionExpression):
    output_cols = []
    for obj, alias in zip(func_expr.output_objs, func_expr.alias.col_names):
        col_alias = '{}.{}'.format(func_expr.alias.alias_name, alias)
        alias_obj = TupleValueExpression(name=alias, table_alias=func_expr.alias.alias_name, col_object=obj, col_alias=col_alias)
        output_cols.append(alias_obj)
    return output_cols

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

def is_video_table(table: TableCatalogEntry):
    return table.table_type == TableType.VIDEO_DATA

class CreateFunctionExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: CreateFunctionPlan):
        super().__init__(db, node)
        self.function_dir = Path(EvaDB_INSTALLATION_DIR) / 'functions'

    def handle_huggingface_function(self):
        """Handle HuggingFace functions

        HuggingFace functions are special functions that are not loaded from a file.
        So we do not need to call the setup method on them like we do for other functions.
        """
        try_to_import_torch()
        impl_path = f'{self.function_dir}/abstract/hf_abstract_function.py'
        io_list = gen_hf_io_catalog_entries(self.node.name, self.node.metadata)
        return (self.node.name, impl_path, self.node.function_type, io_list, self.node.metadata)

    def handle_ludwig_function(self):
        """Handle ludwig functions

        Use Ludwig's auto_train engine to train/tune models.
        """
        try_to_import_ludwig()
        from ludwig.automl import auto_train
        assert len(self.children) == 1, 'Create ludwig function expects 1 child, finds {}.'.format(len(self.children))
        aggregated_batch_list = []
        child = self.children[0]
        for batch in child.exec():
            aggregated_batch_list.append(batch)
        aggregated_batch = Batch.concat(aggregated_batch_list, copy=False)
        aggregated_batch.drop_column_alias()
        arg_map = {arg.key: arg.value for arg in self.node.metadata}
        start_time = int(time.time())
        auto_train_results = auto_train(dataset=aggregated_batch.frames, target=arg_map['predict'], tune_for_memory=arg_map.get('tune_for_memory', False), time_limit_s=arg_map.get('time_limit', DEFAULT_TRAIN_TIME_LIMIT), output_directory=self.db.catalog().get_configuration_catalog_value('tmp_dir'))
        train_time = int(time.time()) - start_time
        model_path = os.path.join(self.db.catalog().get_configuration_catalog_value('model_dir'), self.node.name)
        auto_train_results.best_model.save(model_path)
        best_score = auto_train_results.experiment_analysis.best_result['metric_score']
        self.node.metadata.append(FunctionMetadataCatalogEntry('model_path', model_path))
        impl_path = Path(f'{self.function_dir}/ludwig.py').absolute().as_posix()
        io_list = self._resolve_function_io(None)
        return (self.node.name, impl_path, self.node.function_type, io_list, self.node.metadata, best_score, train_time)

    def handle_sklearn_function(self):
        """Handle sklearn functions

        Use Sklearn's regression to train models.
        """
        try_to_import_flaml_automl()
        assert len(self.children) == 1, 'Create sklearn function expects 1 child, finds {}.'.format(len(self.children))
        aggregated_batch_list = []
        child = self.children[0]
        for batch in child.exec():
            aggregated_batch_list.append(batch)
        aggregated_batch = Batch.concat(aggregated_batch_list, copy=False)
        aggregated_batch.drop_column_alias()
        arg_map = {arg.key: arg.value for arg in self.node.metadata}
        from flaml import AutoML
        model = AutoML()
        sklearn_model = arg_map.get('model', DEFAULT_SKLEARN_TRAIN_MODEL)
        if sklearn_model not in SKLEARN_SUPPORTED_MODELS:
            raise ValueError(f'Sklearn Model {sklearn_model} provided as input is not supported.')
        settings = {'time_budget': arg_map.get('time_limit', DEFAULT_TRAIN_TIME_LIMIT), 'metric': arg_map.get('metric', DEFAULT_TRAIN_REGRESSION_METRIC), 'estimator_list': [sklearn_model], 'task': arg_map.get('task', DEFAULT_XGBOOST_TASK)}
        start_time = int(time.time())
        model.fit(dataframe=aggregated_batch.frames, label=arg_map['predict'], **settings)
        train_time = int(time.time()) - start_time
        score = model.best_loss
        model_path = os.path.join(self.db.catalog().get_configuration_catalog_value('model_dir'), self.node.name)
        pickle.dump(model, open(model_path, 'wb'))
        self.node.metadata.append(FunctionMetadataCatalogEntry('model_path', model_path))
        self.node.metadata.append(FunctionMetadataCatalogEntry('predict_col', arg_map['predict']))
        impl_path = Path(f'{self.function_dir}/sklearn.py').absolute().as_posix()
        io_list = self._resolve_function_io(None)
        return (self.node.name, impl_path, self.node.function_type, io_list, self.node.metadata, score, train_time)

    def convert_to_numeric(self, x):
        x = re.sub('[^0-9.,]', '', str(x))
        locale.setlocale(locale.LC_ALL, '')
        x = float(locale.atof(x))
        if x.is_integer():
            return int(x)
        else:
            return x

    def handle_xgboost_function(self):
        """Handle xgboost functions

        We use the Flaml AutoML model for training xgboost models.
        """
        try_to_import_flaml_automl()
        assert len(self.children) == 1, 'Create sklearn function expects 1 child, finds {}.'.format(len(self.children))
        aggregated_batch_list = []
        child = self.children[0]
        for batch in child.exec():
            aggregated_batch_list.append(batch)
        aggregated_batch = Batch.concat(aggregated_batch_list, copy=False)
        aggregated_batch.drop_column_alias()
        arg_map = {arg.key: arg.value for arg in self.node.metadata}
        from flaml import AutoML
        model = AutoML()
        settings = {'time_budget': arg_map.get('time_limit', DEFAULT_TRAIN_TIME_LIMIT), 'metric': arg_map.get('metric', DEFAULT_TRAIN_REGRESSION_METRIC), 'estimator_list': ['xgboost'], 'task': arg_map.get('task', DEFAULT_XGBOOST_TASK)}
        start_time = int(time.time())
        model.fit(dataframe=aggregated_batch.frames, label=arg_map['predict'], **settings)
        train_time = int(time.time()) - start_time
        model_path = os.path.join(self.db.catalog().get_configuration_catalog_value('model_dir'), self.node.name)
        pickle.dump(model, open(model_path, 'wb'))
        self.node.metadata.append(FunctionMetadataCatalogEntry('model_path', model_path))
        self.node.metadata.append(FunctionMetadataCatalogEntry('predict_col', arg_map['predict']))
        impl_path = Path(f'{self.function_dir}/xgboost.py').absolute().as_posix()
        io_list = self._resolve_function_io(None)
        best_score = model.best_loss
        return (self.node.name, impl_path, self.node.function_type, io_list, self.node.metadata, best_score, train_time)

    def handle_ultralytics_function(self):
        """Handle Ultralytics functions"""
        try_to_import_ultralytics()
        impl_path = Path(f'{self.function_dir}/yolo_object_detector.py').absolute().as_posix()
        function = self._try_initializing_function(impl_path, function_args=get_metadata_properties(self.node))
        io_list = self._resolve_function_io(function)
        return (self.node.name, impl_path, self.node.function_type, io_list, self.node.metadata)

    def handle_forecasting_function(self):
        """Handle forecasting functions"""
        aggregated_batch_list = []
        child = self.children[0]
        for batch in child.exec():
            aggregated_batch_list.append(batch)
        aggregated_batch = Batch.concat(aggregated_batch_list, copy=False)
        aggregated_batch.drop_column_alias()
        arg_map = {arg.key: arg.value for arg in self.node.metadata}
        if not self.node.impl_path:
            impl_path = Path(f'{self.function_dir}/forecast.py').absolute().as_posix()
        else:
            impl_path = self.node.impl_path.absolute().as_posix()
        library = 'statsforecast'
        supported_libraries = ['statsforecast', 'neuralforecast']
        if 'horizon' not in arg_map.keys():
            raise ValueError('Horizon must be provided while creating function of type FORECASTING')
        try:
            horizon = int(arg_map['horizon'])
        except Exception as e:
            err_msg = f'{str(e)}. HORIZON must be integral.'
            logger.error(err_msg)
            raise FunctionIODefinitionError(err_msg)
        if 'library' in arg_map.keys():
            try:
                assert arg_map['library'].lower() in supported_libraries
            except Exception:
                err_msg = 'EvaDB currently supports ' + str(supported_libraries) + ' only.'
                logger.error(err_msg)
                raise FunctionIODefinitionError(err_msg)
            library = arg_map['library'].lower()
        '\n        The following rename is needed for statsforecast/neuralforecast, which requires the column name to be the following:\n        - The unique_id (string, int or category) represents an identifier for the series.\n        - The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp.\n        - The y (numeric) represents the measurement we wish to forecast.\n        For reference: https://nixtla.github.io/statsforecast/docs/getting-started/getting_started_short.html\n        '
        aggregated_batch.rename(columns={arg_map['predict']: 'y'})
        if 'time' in arg_map.keys():
            aggregated_batch.rename(columns={arg_map['time']: 'ds'})
        if 'id' in arg_map.keys():
            aggregated_batch.rename(columns={arg_map['id']: 'unique_id'})
        if 'conf' in arg_map.keys():
            try:
                conf = round(arg_map['conf'])
            except Exception:
                err_msg = 'Confidence must be a number.'
                logger.error(err_msg)
                raise FunctionIODefinitionError(err_msg)
        else:
            conf = 90
        if conf > 100:
            err_msg = 'Confidence must <= 100.'
            logger.error(err_msg)
            raise FunctionIODefinitionError(err_msg)
        data = aggregated_batch.frames
        if 'unique_id' not in list(data.columns):
            data['unique_id'] = [1 for x in range(len(data))]
        if 'ds' not in list(data.columns):
            data['ds'] = [x + 1 for x in range(len(data))]
        '\n            Set or infer data frequency\n        '
        if 'frequency' not in arg_map.keys() or arg_map['frequency'] == 'auto':
            arg_map['frequency'] = pd.infer_freq(data['ds'])
        frequency = arg_map['frequency']
        if frequency is None:
            raise RuntimeError(f'Can not infer the frequency for {self.node.name}. Please explicitly set it.')
        season_dict = {'H': 24, 'M': 12, 'Q': 4, 'SM': 24, 'BM': 12, 'BMS': 12, 'BQ': 4, 'BH': 24}
        new_freq = frequency.split('-')[0] if '-' in frequency else frequency
        season_length = season_dict[new_freq] if new_freq in season_dict else 1
        '\n            Neuralforecast implementation\n        '
        if library == 'neuralforecast':
            try_to_import_neuralforecast()
            from neuralforecast import NeuralForecast
            from neuralforecast.auto import AutoDeepAR, AutoFEDformer, AutoInformer, AutoNBEATS, AutoNHITS, AutoPatchTST, AutoTFT
            from neuralforecast.losses.pytorch import MQLoss
            from neuralforecast.models import NBEATS, NHITS, TFT, DeepAR, FEDformer, Informer, PatchTST
            model_dict = {'AutoNBEATS': AutoNBEATS, 'AutoNHITS': AutoNHITS, 'NBEATS': NBEATS, 'NHITS': NHITS, 'PatchTST': PatchTST, 'AutoPatchTST': AutoPatchTST, 'DeepAR': DeepAR, 'AutoDeepAR': AutoDeepAR, 'FEDformer': FEDformer, 'AutoFEDformer': AutoFEDformer, 'Informer': Informer, 'AutoInformer': AutoInformer, 'TFT': TFT, 'AutoTFT': AutoTFT}
            if 'model' not in arg_map.keys():
                arg_map['model'] = 'TFT'
            if 'auto' not in arg_map.keys() or (arg_map['auto'].lower()[0] == 't' and 'auto' not in arg_map['model'].lower()):
                arg_map['model'] = 'Auto' + arg_map['model']
            try:
                model_here = model_dict[arg_map['model']]
            except Exception:
                err_msg = 'Supported models: ' + str(model_dict.keys())
                logger.error(err_msg)
                raise FunctionIODefinitionError(err_msg)
            model_args = {}
            if 'auto' not in arg_map['model'].lower():
                model_args['input_size'] = 2 * horizon
                model_args['early_stop_patience_steps'] = 20
            else:
                model_args_config = {'input_size': 2 * horizon, 'early_stop_patience_steps': 20}
            if len(data.columns) >= 4:
                exogenous_columns = [x for x in list(data.columns) if x not in ['ds', 'y', 'unique_id']]
                if 'auto' not in arg_map['model'].lower():
                    model_args['hist_exog_list'] = exogenous_columns
                else:
                    model_args_config['hist_exog_list'] = exogenous_columns
            if 'auto' in arg_map['model'].lower():

                def get_optuna_config(trial):
                    return model_args_config
                model_args['config'] = get_optuna_config
                model_args['backend'] = 'optuna'
            model_args['h'] = horizon
            model_args['loss'] = MQLoss(level=[conf])
            model = NeuralForecast([model_here(**model_args)], freq=new_freq)
        else:
            if 'auto' in arg_map.keys() and arg_map['auto'].lower()[0] != 't':
                raise RuntimeError('Statsforecast implementation only supports automatic hyperparameter optimization. Please set AUTO to true.')
            try_to_import_statsforecast()
            from statsforecast import StatsForecast
            from statsforecast.models import AutoARIMA, AutoCES, AutoETS, AutoTheta
            model_dict = {'AutoARIMA': AutoARIMA, 'AutoCES': AutoCES, 'AutoETS': AutoETS, 'AutoTheta': AutoTheta}
            if 'model' not in arg_map.keys():
                arg_map['model'] = 'ARIMA'
            if 'auto' not in arg_map['model'].lower():
                arg_map['model'] = 'Auto' + arg_map['model']
            try:
                model_here = model_dict[arg_map['model']]
            except Exception:
                err_msg = 'Supported models: ' + str(model_dict.keys())
                logger.error(err_msg)
                raise FunctionIODefinitionError(err_msg)
            model = StatsForecast([model_here(season_length=season_length)], freq=new_freq)
        data['ds'] = pd.to_datetime(data['ds'])
        model_save_dir_name = library + '_' + arg_map['model'] + '_' + new_freq if 'statsforecast' in library else library + '_' + str(conf) + '_' + arg_map['model'] + '_' + new_freq
        if len(data.columns) >= 4 and library == 'neuralforecast':
            model_save_dir_name += '_exogenous_' + str(sorted(exogenous_columns))
        model_dir = os.path.join(self.db.catalog().get_configuration_catalog_value('model_dir'), 'tsforecasting', model_save_dir_name, str(hashlib.sha256(data.to_string().encode()).hexdigest()))
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        model_save_name = 'horizon' + str(horizon) + '.pkl'
        model_path = os.path.join(model_dir, model_save_name)
        existing_model_files = sorted(os.listdir(model_dir), key=lambda x: int(x.split('horizon')[1].split('.pkl')[0]))
        existing_model_files = [x for x in existing_model_files if int(x.split('horizon')[1].split('.pkl')[0]) >= horizon]
        if len(existing_model_files) == 0:
            logger.info('Training, please wait...')
            for column in data.columns:
                if column != 'ds' and column != 'unique_id':
                    data[column] = data.apply(lambda x: self.convert_to_numeric(x[column]), axis=1)
            rmses = []
            if library == 'neuralforecast':
                cuda_devices_here = '0'
                if 'CUDA_VISIBLE_DEVICES' in os.environ:
                    cuda_devices_here = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
                with set_env(CUDA_VISIBLE_DEVICES=cuda_devices_here):
                    model.fit(df=data, val_size=horizon)
                    model.save(model_path, overwrite=True)
                    if 'metrics' in arg_map and arg_map['metrics'].lower()[0] == 't':
                        crossvalidation_df = model.cross_validation(df=data, val_size=horizon)
                        for uid in crossvalidation_df.unique_id.unique():
                            crossvalidation_df_here = crossvalidation_df[crossvalidation_df.unique_id == uid]
                            rmses.append(root_mean_squared_error(crossvalidation_df_here.y, crossvalidation_df_here[arg_map['model'] + '-median']) / np.mean(crossvalidation_df_here.y))
                            mean_rmse = np.mean(rmses)
                            with open(model_path + '_rmse', 'w') as f:
                                f.write(str(mean_rmse) + '\n')
            else:
                for col in data['unique_id'].unique():
                    if len(data[data['unique_id'] == col]) == 1:
                        data = data._append([data[data['unique_id'] == col]], ignore_index=True)
                model.fit(df=data[['ds', 'y', 'unique_id']])
                hypers = ''
                if 'arima' in arg_map['model'].lower():
                    from statsforecast.arima import arima_string
                    hypers += arima_string(model.fitted_[0, 0].model_)
                f = open(model_path, 'wb')
                pickle.dump(model, f)
                f.close()
                if 'metrics' not in arg_map or arg_map['metrics'].lower()[0] == 't':
                    crossvalidation_df = model.cross_validation(df=data[['ds', 'y', 'unique_id']], h=horizon, step_size=24, n_windows=1).reset_index()
                    for uid in crossvalidation_df.unique_id.unique():
                        crossvalidation_df_here = crossvalidation_df[crossvalidation_df.unique_id == uid]
                        rmses.append(root_mean_squared_error(crossvalidation_df_here.y, crossvalidation_df_here[arg_map['model']]) / np.mean(crossvalidation_df_here.y))
                    mean_rmse = np.mean(rmses)
                    with open(model_path + '_rmse', 'w') as f:
                        f.write(str(mean_rmse) + '\n')
                        f.write(hypers + '\n')
        elif not Path(model_path).exists():
            model_path = os.path.join(model_dir, existing_model_files[-1])
        io_list = self._resolve_function_io(None)
        data['ds'] = data.ds.astype(str)
        metadata_here = [FunctionMetadataCatalogEntry('model_name', arg_map['model']), FunctionMetadataCatalogEntry('model_path', model_path), FunctionMetadataCatalogEntry('predict_column_rename', arg_map.get('predict', 'y')), FunctionMetadataCatalogEntry('time_column_rename', arg_map.get('time', 'ds')), FunctionMetadataCatalogEntry('id_column_rename', arg_map.get('id', 'unique_id')), FunctionMetadataCatalogEntry('horizon', horizon), FunctionMetadataCatalogEntry('library', library), FunctionMetadataCatalogEntry('conf', conf)]
        return (self.node.name, impl_path, self.node.function_type, io_list, metadata_here)

    def handle_generic_function(self):
        """Handle generic functions

        Generic functions are loaded from a file. We check for inputs passed by the user during CREATE or try to load io from decorators.
        """
        impl_path = self.node.impl_path.absolute().as_posix()
        function = self._try_initializing_function(impl_path)
        io_list = self._resolve_function_io(function)
        return (self.node.name, impl_path, self.node.function_type, io_list, self.node.metadata)

    def exec(self, *args, **kwargs):
        """Create function executor

        Calls the catalog to insert a function catalog entry.
        """
        assert (self.node.if_not_exists and self.node.or_replace) is False, 'OR REPLACE and IF NOT EXISTS can not be both set for CREATE FUNCTION.'
        overwrite = False
        best_score = False
        train_time = False
        if self.catalog().get_function_catalog_entry_by_name(self.node.name):
            if self.node.if_not_exists:
                msg = f'Function {self.node.name} already exists, nothing added.'
                yield Batch(pd.DataFrame([msg]))
                return
            elif self.node.or_replace:
                from evadb.executor.drop_object_executor import DropObjectExecutor
                drop_executor = DropObjectExecutor(self.db, None)
                try:
                    drop_executor._handle_drop_function(self.node.name, if_exists=False)
                except RuntimeError:
                    pass
                else:
                    overwrite = True
            else:
                msg = f'Function {self.node.name} already exists.'
                logger.error(msg)
                raise RuntimeError(msg)
        if string_comparison_case_insensitive(self.node.function_type, 'HuggingFace'):
            name, impl_path, function_type, io_list, metadata = self.handle_huggingface_function()
        elif string_comparison_case_insensitive(self.node.function_type, 'ultralytics'):
            name, impl_path, function_type, io_list, metadata = self.handle_ultralytics_function()
        elif string_comparison_case_insensitive(self.node.function_type, 'Ludwig'):
            name, impl_path, function_type, io_list, metadata, best_score, train_time = self.handle_ludwig_function()
        elif string_comparison_case_insensitive(self.node.function_type, 'Sklearn'):
            name, impl_path, function_type, io_list, metadata, best_score, train_time = self.handle_sklearn_function()
        elif string_comparison_case_insensitive(self.node.function_type, 'XGBoost'):
            name, impl_path, function_type, io_list, metadata, best_score, train_time = self.handle_xgboost_function()
        elif string_comparison_case_insensitive(self.node.function_type, 'Forecasting'):
            name, impl_path, function_type, io_list, metadata = self.handle_forecasting_function()
        else:
            name, impl_path, function_type, io_list, metadata = self.handle_generic_function()
        self.catalog().insert_function_catalog_entry(name, impl_path, function_type, io_list, metadata)
        if overwrite:
            msg = f'Function {self.node.name} overwritten.'
        else:
            msg = f'Function {self.node.name} added to the database.'
        if best_score and train_time:
            yield Batch(pd.DataFrame([msg, 'Validation Score: ' + str(best_score), 'Training time: ' + str(train_time) + ' secs.']))
        else:
            yield Batch(pd.DataFrame([msg]))

    def _try_initializing_function(self, impl_path: str, function_args: Dict={}) -> FunctionCatalogEntry:
        """Attempts to initialize function given the implementation file path and arguments.

        Args:
            impl_path (str): The file path of the function implementation file.
            function_args (Dict, optional): Dictionary of arguments to pass to the function. Defaults to {}.

        Returns:
            FunctionCatalogEntry: A FunctionCatalogEntry object that represents the initialized function.

        Raises:
            RuntimeError: If an error occurs while initializing the function.
        """
        try:
            function = load_function_class_from_file(impl_path, self.node.name)
            function(**function_args)
        except Exception as e:
            err_msg = f'Error creating function {self.node.name}: {str(e)}'
            raise RuntimeError(err_msg)
        return function

    def _resolve_function_io(self, function: FunctionCatalogEntry) -> List[FunctionIOCatalogEntry]:
        """Private method that resolves the input/output definitions for a given function.
        It first searches for the input/outputs in the CREATE statement. If not found, it resolves them using decorators. If not found there as well, it raises an error.

        Args:
            function (FunctionCatalogEntry): The function for which to resolve input and output definitions.

        Returns:
            A List of FunctionIOCatalogEntry objects that represent the resolved input and
            output definitions for the function.

        Raises:
            RuntimeError: If an error occurs while resolving the function input/output
            definitions.
        """
        io_list = []
        try:
            if self.node.inputs:
                io_list.extend(self.node.inputs)
            else:
                io_list.extend(load_io_from_function_decorators(function, is_input=True))
            if self.node.outputs:
                io_list.extend(self.node.outputs)
            else:
                io_list.extend(load_io_from_function_decorators(function, is_input=False))
        except FunctionIODefinitionError as e:
            err_msg = f'Error creating function, input/output definition incorrect: {str(e)}'
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        return io_list

def try_to_import_ultralytics():
    try:
        import ultralytics
    except ImportError:
        raise ValueError('Could not import ultralytics python package.\n                Please install it with `pip install ultralytics`.')

def get_metadata_properties(function_obj: FunctionCatalogEntry) -> Dict:
    """
    Return all the metadata properties as key value pair

    Args:
        function_obj (FunctionCatalogEntry): An object of type `FunctionCatalogEntry` which is
        used to extract metadata information.
    Returns:
        Dict: key-value for each metadata entry
    """
    properties = {}
    for metadata in function_obj.metadata:
        properties[metadata.key] = metadata.value
    return properties

def load_io_from_function_decorators(function: Type[AbstractFunction], is_input=False) -> List[Type[FunctionIOCatalogEntry]]:
    """Load the inputs/outputs from the function decorators and return a list of FunctionIOCatalogEntry objects

    Args:
        function (Object): Function object
        is_input (bool, optional): True if inputs are to be loaded. Defaults to False.

    Returns:
        Type[FunctionIOCatalogEntry]: FunctionIOCatalogEntry object created from the input decorator in setup
    """
    tag_key = 'input' if is_input else 'output'
    io_signature = None
    if hasattr(function.forward, 'tags') and tag_key in function.forward.tags:
        io_signature = function.forward.tags[tag_key]
    else:
        for base_class in function.__bases__:
            if hasattr(base_class, 'forward') and hasattr(base_class.forward, 'tags'):
                if tag_key in base_class.forward.tags:
                    io_signature = base_class.forward.tags[tag_key]
                    break
    assert io_signature is not None, f'Cannot infer {tag_key} signature from the decorator for {function}.\n {missing_io_signature_helper()}'
    result_list = []
    for io in io_signature:
        result_list.extend(io.generate_catalog_entries(is_input))
    return result_list

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

def drop_row_id_from_target_list(target_list: List[AbstractExpression]) -> List[AbstractExpression]:
    """
    This function is intended to be used by CREATE FUNCTION FROM (SELECT * FROM ...) and CREATE TABLE AS SELECT * FROM ... to exclude the row_id column.
    """
    filtered_list = []
    for expr in target_list:
        if isinstance(expr, TupleValueExpression):
            if expr.name == IDENTIFIER_COLUMN:
                continue
        filtered_list.append(expr)
    return filtered_list

def get_column_definition_from_select_target_list(target_list: List[AbstractExpression]) -> List[ColumnDefinition]:
    """
    This function is used by CREATE TABLE AS (SELECT...) and
    CREATE FUNCTION FROM (SELECT ...) to get the output objs from the
    child SELECT statement.
    """
    binded_col_list = []
    for expr in target_list:
        output_objs = [(expr.name, expr.col_object)] if expr.etype == ExpressionType.TUPLE_VALUE else zip(expr.projection_columns, expr.output_objs)
        for col_name, output_obj in output_objs:
            binded_col_list.append(ColumnDefinition(col_name.lower(), output_obj.type, output_obj.array_type, output_obj.array_dimensions))
    return binded_col_list

def check_column_name_is_string(col_ref) -> None:
    if not is_string_col(col_ref.col_object):
        err_msg = 'LIKE only supported for string columns'
        raise BinderError(err_msg)

def extend_star(binder_context: StatementBinderContext) -> List[TupleValueExpression]:
    col_objs = binder_context._get_all_alias_and_col_name()
    target_list = list([TupleValueExpression(name=col_name, table_alias=alias) for alias, col_name in col_objs])
    return target_list

def check_table_object_is_groupable(table_ref: TableRef) -> None:
    table_obj = table_ref.table.table_obj
    if not (is_video_table(table_obj) or is_document_table(table_obj) or is_pdf_table(table_obj)):
        raise BinderError('GROUP BY only supported for video and document tables')

def check_groupby_pattern(table_ref: TableRef, groupby_string: str) -> None:
    pattern = re.search('^\\d+\\s*(?:frames|samples|paragraphs)$', groupby_string)
    if not pattern:
        err_msg = 'Incorrect GROUP BY pattern: {}'.format(groupby_string)
        raise BinderError(err_msg)
    match_string = pattern.group(0)
    suffix_string = re.sub('^\\d+\\s*', '', match_string)
    if suffix_string not in ['frames', 'samples', 'paragraphs']:
        err_msg = 'Grouping only supported by frames for videos, by samples for audio, and by paragraphs for documents'
        raise BinderError(err_msg)
    if suffix_string == 'frames' and (not is_video_table(table_ref.table.table_obj)):
        err_msg = 'Grouping by frames only supported for videos'
        raise BinderError(err_msg)
    if suffix_string == 'samples' and (not is_video_table(table_ref.table.table_obj)):
        err_msg = 'Grouping by samples only supported for videos'
        raise BinderError(err_msg)
    if suffix_string == 'paragraphs' and (not is_pdf_table(table_ref.table.table_obj)):
        err_msg = 'Grouping by paragraphs only supported for pdf tables'
        raise BinderError(err_msg)

def is_document_table(table: TableCatalogEntry):
    return table.table_type == TableType.DOCUMENT_DATA

def bind_table_info(catalog: CatalogManager, table_info: TableInfo):
    """
    Uses catalog to bind the table information .

    Arguments:
         catalog (CatalogManager): catalog manager to use
         table_info (TableInfo): table information obtained from SQL query

    Returns:
        TableCatalogEntry  -  corresponding table catalog entry for the input table info
    """
    if table_info.database_name is not None:
        bind_native_table_info(catalog, table_info)
    else:
        bind_evadb_table_info(catalog, table_info)

def bind_func_expr(binder: StatementBinder, node: FunctionExpression):
    gpus_ids = binder._catalog().get_configuration_catalog_value('gpu_ids')
    node._context = Context(gpus_ids)
    if node.name.upper() == str(FunctionType.EXTRACT_OBJECT):
        handle_bind_extract_object_function(node, binder)
        return
    if len(node.children) == 1 and isinstance(node.children[0], TupleValueExpression) and (node.children[0].name == '*'):
        node.children = extend_star(binder._binder_context)
    for child in node.children:
        binder.bind(child)
    function_obj = binder._catalog().get_function_catalog_entry_by_name(node.name)
    if function_obj is None:
        err_msg = f"Function '{node.name}' does not exist in the catalog. Please create the function using CREATE FUNCTION command."
        logger.error(err_msg)
        raise BinderError(err_msg)
    if string_comparison_case_insensitive(function_obj.type, 'HuggingFace'):
        node.function = assign_hf_function(function_obj)
    elif string_comparison_case_insensitive(function_obj.type, 'Ludwig'):
        function_class = load_function_class_from_file(function_obj.impl_file_path, 'GenericLudwigModel')
        function_metadata = get_metadata_properties(function_obj)
        assert 'model_path' in function_metadata, "Ludwig models expect 'model_path'."
        node.function = lambda: function_class(model_path=function_metadata['model_path'])
    else:
        if function_obj.type == 'ultralytics':
            function_dir = Path(EvaDB_INSTALLATION_DIR) / 'functions'
            function_obj.impl_file_path = Path(f'{function_dir}/yolo_object_detector.py').absolute().as_posix()
        try:
            function_class = load_function_class_from_file(function_obj.impl_file_path, function_obj.name)
            properties = get_metadata_properties(function_obj)
            if string_comparison_case_insensitive(node.name, 'CHATGPT'):
                if 'OPENAI_API_KEY' not in properties.keys():
                    openai_key = binder._catalog().get_configuration_catalog_value('OPENAI_API_KEY')
                    properties['openai_api_key'] = openai_key
            node.function = lambda: function_class(**properties)
        except Exception as e:
            err_msg = f'{str(e)}. Please verify that the function class name in the implementation file matches the function name.'
            logger.error(err_msg)
            raise BinderError(err_msg)
    node.function_obj = function_obj
    output_objs = binder._catalog().get_function_io_catalog_output_entries(function_obj)
    if node.output:
        for obj in output_objs:
            if obj.name.lower() == node.output:
                node.output_objs = [obj]
        if not node.output_objs:
            err_msg = f'Output {node.output} does not exist for {function_obj.name}.'
            logger.error(err_msg)
            raise BinderError(err_msg)
        node.projection_columns = [node.output]
    else:
        node.output_objs = output_objs
        node.projection_columns = [obj.name.lower() for obj in output_objs]
    resolve_alias_table_value_expression(node)

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

def bind_evadb_table_info(catalog: CatalogManager, table_info: TableInfo):
    obj = catalog.get_table_catalog_entry(table_info.table_name, table_info.database_name)
    if obj and obj.table_type == TableType.SYSTEM_STRUCTURED_DATA:
        err_msg = f'The query attempted to access or modify the internal table{table_info.table_name} of the system, but permission was denied.'
        logger.error(err_msg)
        raise BinderError(err_msg)
    if obj:
        table_info.table_obj = obj
    else:
        error = '{} does not exist. Create the table using CREATE TABLE.'.format(table_info.table_name)
        logger.error(error)
        raise BinderError(error)

def is_pdf_table(table: TableCatalogEntry):
    return table.table_type == TableType.PDF_DATA

def is_string_col(col: ColumnCatalogEntry):
    return col.type == ColumnType.TEXT or col.array_type == NdArrayType.STR

def assign_hf_function(function_obj: FunctionCatalogEntry):
    """
    Assigns the correct HF Model to the Function. The model assigned depends on
    the task type for the Function. This is done so that we can
    process the input correctly before passing it to the HF model.
    """
    inputs = function_obj.args
    assert len(inputs) == 1, 'Only single input models are supported.'
    task = get_metadata_entry_or_val(function_obj, 'task', None)
    assert task is not None, 'task not specified in Hugging Face Function'
    model_class = MODEL_FOR_TASK[task]
    return lambda: model_class(function_obj)

class SchemaUtils(object):

    @staticmethod
    def xform_to_sqlalchemy_column(df_column: ColumnCatalogEntry) -> Column:
        column_type = df_column.type
        sqlalchemy_column = None
        if column_type == ColumnType.INTEGER:
            sqlalchemy_column = Column(Integer)
        elif column_type == ColumnType.FLOAT:
            sqlalchemy_column = Column(Float)
        elif column_type == ColumnType.TEXT:
            sqlalchemy_column = Column(TEXT)
        elif column_type == ColumnType.NDARRAY:
            sqlalchemy_column = Column(LargeBinary)
        else:
            msg = 'Invalid column type: ' + str(column_type)
            logger.error(msg)
            raise NotImplementedError
        return sqlalchemy_column

    @staticmethod
    def xform_to_sqlalchemy_schema(column_list: List[ColumnCatalogEntry]) -> Dict[str, Column]:
        """Converts the list of DataFrameColumns to SQLAlchemyColumns

        Args:
            column_list (List[ColumnCatalog]): columns to be converted

        Returns:
            Dict[str, Column]: mapping from column_name to sqlalchemy column object
        """
        return {column.name: SchemaUtils.xform_to_sqlalchemy_column(column) for column in column_list}

class TableCatalogService(BaseService):

    def __init__(self, db_session: Session):
        super().__init__(TableCatalog, db_session)
        self._column_service: ColumnCatalogService = ColumnCatalogService(db_session)

    def insert_entry(self, name: str, file_url: str, identifier_column: str, table_type: TableType, column_list) -> TableCatalogEntry:
        """Insert a new table entry into table catalog.
        Arguments:
            name (str): name of the table
            file_url (str): file path of the table.
            table_type (TableType): type of data in the table
        Returns:
            TableCatalogEntry
        """
        try:
            table_catalog_obj = self.model(name=name, file_url=file_url, identifier_column=identifier_column, table_type=table_type)
            table_catalog_obj = table_catalog_obj.save(self.session)
            for column in column_list:
                column.table_id = table_catalog_obj._row_id
            column_list = self._column_service.create_entries(column_list)
            try:
                self.session.add_all(column_list)
                self.session.commit()
            except Exception as e:
                self.session.rollback()
                self.session.delete(table_catalog_obj)
                self.session.commit()
                logger.exception(f'Failed to insert entry into table catalog with exception {str(e)}')
                raise CatalogError(e)
        except Exception as e:
            logger.exception(f'Failed to insert entry into table catalog with exception {str(e)}')
            raise CatalogError(e)
        else:
            return table_catalog_obj.as_dataclass()

    def get_entry_by_id(self, table_id: int, return_alchemy=False) -> TableCatalogEntry:
        """
        Returns the table by ID
        Arguments:
            table_id (int)
            return_alchemy (bool): if True, return a sqlalchemy object
        Returns:
           TableCatalogEntry
        """
        entry = self.session.execute(select(self.model).filter(self.model._row_id == table_id)).scalar_one()
        return entry if return_alchemy else entry.as_dataclass()

    def get_entry_by_name(self, database_name, table_name, return_alchemy=False) -> TableCatalogEntry:
        """
        Get the table catalog entry with given table name.
        Arguments:
            database_name  (str): Database to which table belongs # TODO:
            use this field
            table_name (str): name of the table
        Returns:
            TableCatalogEntry - catalog entry for given table_name
        """
        entry = self.session.execute(select(self.model).filter(self.model._name == table_name)).scalar_one_or_none()
        if entry:
            return entry if return_alchemy else entry.as_dataclass()
        return entry

    def delete_entry(self, table: TableCatalogEntry):
        """Delete table from the db
        Arguments:
            table  (TableCatalogEntry): table to delete
        Returns:
            True if successfully removed else false
        """
        try:
            table_obj = self.session.execute(select(self.model).filter(self.model._row_id == table.row_id)).scalar_one_or_none()
            table_obj.delete(self.session)
            return True
        except Exception as e:
            err_msg = f'Delete table failed for {table} with error {str(e)}.'
            logger.exception(err_msg)
            raise CatalogError(err_msg)

    def rename_entry(self, table: TableCatalogEntry, new_name: str):
        try:
            table_obj = self.session.execute(select(self.model).filter(self.model._row_id == table.row_id)).scalar_one_or_none()
            if table_obj:
                table_obj.update(self.session, _name=new_name)
        except Exception as e:
            err_msg = 'Update table name failed for {} with error {}'.format(table.name, str(e))
            logger.error(err_msg)
            raise RuntimeError(err_msg)

class FunctionIOCatalogService(BaseService):

    def __init__(self, db_session: Session):
        super().__init__(FunctionIOCatalog, db_session)

    def get_input_entries_by_function_id(self, function_id: int) -> List[FunctionIOCatalogEntry]:
        try:
            result = self.session.execute(select(self.model).filter(self.model._function_id == function_id, self.model._is_input == True)).scalars().all()
            return [obj.as_dataclass() for obj in result]
        except Exception as e:
            error = f'Getting inputs for function id {function_id} raised {e}'
            logger.error(error)
            raise RuntimeError(error)

    def get_output_entries_by_function_id(self, function_id: int) -> List[FunctionIOCatalogEntry]:
        try:
            result = self.session.execute(select(self.model).filter(self.model._function_id == function_id, self.model._is_input == False)).scalars().all()
            return [obj.as_dataclass() for obj in result]
        except Exception as e:
            error = f'Getting outputs for function id {function_id} raised {e}'
            logger.error(error)
            raise RuntimeError(error)

    def create_entries(self, io_list: List[FunctionIOCatalogEntry]):
        io_objs = []
        for io in io_list:
            io_obj = FunctionIOCatalog(name=io.name, type=io.type, is_nullable=io.is_nullable, array_type=io.array_type, array_dimensions=io.array_dimensions, is_input=io.is_input, function_id=io.function_id)
            io_objs.append(io_obj)
        return io_objs

class FunctionMetadataCatalogService(BaseService):

    def __init__(self, db_session: Session):
        super().__init__(FunctionMetadataCatalog, db_session)

    def create_entries(self, entries: List[FunctionMetadataCatalogEntry]):
        metadata_objs = []
        try:
            for entry in entries:
                metadata_obj = FunctionMetadataCatalog(key=entry.key, value=entry.value, function_id=entry.function_id)
                metadata_objs.append(metadata_obj)
            return metadata_objs
        except Exception as e:
            raise CatalogError(e)

    def get_entries_by_function_id(self, function_id: int) -> List[FunctionMetadataCatalogEntry]:
        try:
            result = self.session.execute(select(self.model).filter(self.model._function_id == function_id)).scalars().all()
            return [obj.as_dataclass() for obj in result]
        except Exception as e:
            error = f'Getting metadata entries for function id {function_id} raised {e}'
            logger.error(error)
            raise CatalogError(error)

class CustomModel:
    """This overrides the default `_declarative_constructor` constructor.

    It skips the attributes that are not present for the model, thus if a
    dict is passed with some unknown attributes for the model on creation,
    it won't complain for `unknown field`s.
    Declares and int `_row_id` field for all tables
    """
    _row_id = Column('_row_id', Integer, primary_key=True)

    def __init__(self, **kwargs):
        cls_ = type(self)
        for k in kwargs:
            if hasattr(cls_, k):
                setattr(self, k, kwargs[k])
            else:
                continue

    def save(self, db_session):
        """Add and commit

        Returns: saved object

        """
        try:
            db_session.add(self)
            self._commit(db_session)
        except Exception as e:
            db_session.rollback()
            logger.error(f'Database save failed : {str(e)}')
            raise e
        return self

    def update(self, db_session, **kwargs):
        """Update and commit

        Args:
            **kwargs: attributes to update

        Returns: updated object

        """
        try:
            for attr, value in kwargs.items():
                if hasattr(self, attr):
                    setattr(self, attr, value)
            return self.save(db_session)
        except Exception as e:
            db_session.rollback()
            logger.error(f'Database update failed : {str(e)}')
            raise e

    def delete(self, db_session):
        """Delete and commit"""
        try:
            db_session.delete(self)
            self._commit(db_session)
        except Exception as e:
            db_session.rollback()
            logger.error(f'Database delete failed : {str(e)}')
            raise e

    def _commit(self, db_session):
        """Try to commit. If an error is raised, the session is rollbacked."""
        try:
            db_session.commit()
        except SQLAlchemyError as e:
            db_session.rollback()
            logger.error(f'Database commit failed : {str(e)}')
            raise e

class Expressions:

    def string_literal(self, tree):
        text = tree.children[0]
        assert text is not None
        return ConstantValueExpression(text[1:-1], ColumnType.TEXT)

    def array_literal(self, tree):
        array_elements = []
        for child in tree.children:
            if isinstance(child, Tree):
                array_element = self.visit(child).value
                array_elements.append(array_element)
        res = ConstantValueExpression(np.array(array_elements), ColumnType.NDARRAY)
        return res

    def boolean_literal(self, tree):
        text = tree.children[0]
        if text == 'TRUE':
            return ConstantValueExpression(True, ColumnType.BOOLEAN)
        return ConstantValueExpression(False, ColumnType.BOOLEAN)

    def constant(self, tree):
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'real_literal':
                    real_literal = self.visit(child)
                    return ConstantValueExpression(real_literal, ColumnType.FLOAT)
                elif child.data == 'decimal_literal':
                    decimal_literal = self.visit(child)
                    return ConstantValueExpression(decimal_literal, ColumnType.INTEGER)
        return self.visit_children(tree)

    def logical_expression(self, tree):
        left = self.visit(tree.children[0])
        op = self.visit(tree.children[1])
        right = self.visit(tree.children[2])
        return LogicalExpression(op, left, right)

    def binary_comparison_predicate(self, tree):
        left = self.visit(tree.children[0])
        op = self.visit(tree.children[1])
        right = self.visit(tree.children[2])
        return ComparisonExpression(op, left, right)

    def nested_expression_atom(self, tree):
        expr = tree.children[0]
        return self.visit(expr)

    def comparison_operator(self, tree):
        op = str(tree.children[0])
        if op == '=':
            return ExpressionType.COMPARE_EQUAL
        elif op == '<':
            return ExpressionType.COMPARE_LESSER
        elif op == '>':
            return ExpressionType.COMPARE_GREATER
        elif op == '>=':
            return ExpressionType.COMPARE_GEQ
        elif op == '<=':
            return ExpressionType.COMPARE_LEQ
        elif op == '!=':
            return ExpressionType.COMPARE_NEQ
        elif op == '@>':
            return ExpressionType.COMPARE_CONTAINS
        elif op == '<@':
            return ExpressionType.COMPARE_IS_CONTAINED
        elif op == 'LIKE':
            return ExpressionType.COMPARE_LIKE

    def logical_operator(self, tree):
        op = str(tree.children[0])
        if string_comparison_case_insensitive(op, 'OR'):
            return ExpressionType.LOGICAL_OR
        elif string_comparison_case_insensitive(op, 'AND'):
            return ExpressionType.LOGICAL_AND
        else:
            raise NotImplementedError('Unsupported logical operator: {}'.format(op))

    def expressions_with_defaults(self, tree):
        expr_list = []
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'expression_or_default':
                    expression = self.visit(child)
                    expr_list.append(expression)
        return expr_list

    def sample_params(self, tree):
        sample_type = None
        sample_freq = None
        for child in tree.children:
            if child.data == 'sample_clause':
                sample_freq = self.visit(child)
            elif child.data == 'sample_clause_with_type':
                sample_type, sample_freq = self.visit(child)
        return (sample_type, sample_freq)

    def sample_clause(self, tree):
        sample_list = self.visit_children(tree)
        assert len(sample_list) == 2
        return ConstantValueExpression(sample_list[1])

    def sample_clause_with_type(self, tree):
        sample_list = self.visit_children(tree)
        assert len(sample_list) == 3 or len(sample_list) == 2
        if len(sample_list) == 3:
            return (ConstantValueExpression(sample_list[1]), ConstantValueExpression(sample_list[2]))
        else:
            return (ConstantValueExpression(sample_list[1]), ConstantValueExpression(1))

    def chunk_params(self, tree):
        chunk_params = self.visit_children(tree)
        assert len(chunk_params) == 2 or len(chunk_params) == 4
        if len(chunk_params) == 4:
            return {'chunk_size': chunk_params[1], 'chunk_overlap': chunk_params[3]}
        elif len(chunk_params) == 2:
            if chunk_params[0] == 'CHUNK_SIZE':
                return {'chunk_size': chunk_params[1]}
            elif chunk_params[0] == 'CHUNK_OVERLAP':
                return {'chunk_overlap': chunk_params[1]}
            else:
                assert f'incorrect keyword found {chunk_params[0]}'

    def colon_param_dict(self, tree):
        param_dict = {}
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'colon_param':
                    param = self.visit(child)
                    key = param[0].value
                    value = param[1].value
                    param_dict[key] = value
        return param_dict

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

class CommonClauses:

    def table_name(self, tree):
        child = self.visit(tree.children[0])
        if isinstance(child, tuple):
            database_name, table_name = (child[0], child[1])
        else:
            database_name, table_name = (None, child)
        if table_name is not None:
            return TableInfo(table_name=table_name, database_name=database_name)
        else:
            error = 'Invalid Table Name'
            logger.error(error)

    def full_id(self, tree):
        if len(tree.children) == 1:
            return self.visit(tree.children[0])
        elif len(tree.children) == 2:
            return (self.visit(tree.children[0]), self.visit(tree.children[1]))

    def uid(self, tree):
        if hasattr(tree.children[0], 'type') and tree.children[0].type == 'REVERSE_QUOTE_ID':
            tree.children[0].type = 'simple_id'
            non_tick_string = str(tree.children[0]).replace('`', '')
            return non_tick_string
        return self.visit(tree.children[0])

    def full_column_name(self, tree):
        uid = self.visit(tree.children[0])
        if len(tree.children) > 1:
            dotted_id = self.visit(tree.children[1])
            return TupleValueExpression(table_alias=uid, name=dotted_id)
        else:
            return TupleValueExpression(name=uid)

    def dotted_id(self, tree):
        dotted_id = str(tree.children[0])
        dotted_id = dotted_id.lstrip('.')
        return dotted_id

    def simple_id(self, tree):
        simple_id = str(tree.children[0])
        return simple_id

    def decimal_literal(self, tree):
        decimal = None
        token = tree.children[0]
        if str.upper(token) == 'ANYDIM':
            decimal = Dimension.ANYDIM
        else:
            decimal = int(str(token))
        return decimal

    def real_literal(self, tree):
        real_literal = float(tree.children[0])
        return real_literal

def get_metadata_entry_or_val(function_obj: FunctionCatalogEntry, key: str, default_val: Any=None) -> str:
    """
    Return the metadata value for the given key, or the default value if the
    key is not found.

    Args:
        function_obj (FunctionCatalogEntry): An object of type `FunctionCatalogEntry` which is
        used to extract metadata information.
        key (str): The metadata key for which the corresponding value needs to be retrieved.
        default_val (Any): The default value to be returned if the metadata key is not found.

    Returns:
        str: metadata value
    """
    for metadata in function_obj.metadata:
        if metadata.key == key:
            return metadata.value
    return default_val

class Yolo(AbstractFunction, GPUCompatible):
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score
    """

    @property
    def name(self) -> str:
        return 'yolo'

    @setup(cacheable=True, function_type='object_detection', batchable=True)
    def setup(self, model: str, threshold=0.3):
        try_to_import_ultralytics()
        from ultralytics import YOLO
        self.threshold = threshold
        self.model = YOLO(model)
        self.device = 'cpu'

    @forward(input_signatures=[PandasDataframe(columns=['data'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])], output_signatures=[PandasDataframe(columns=['labels', 'bboxes', 'scores'], column_types=[NdArrayType.STR, NdArrayType.FLOAT32, NdArrayType.FLOAT32], column_shapes=[(None,), (None,), (None,)])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (np.ndarray): Frames on which predictions need
            to be performed
        Returns:
            tuple containing predicted_classes (List[List[str]]),
            predicted_boxes (List[List[BoundingBox]]),
            predicted_scores (List[List[float]])
        """
        outcome = []
        frames = np.ravel(frames.to_numpy())
        list_of_numpy_images = [its for its in frames]
        predictions = self.model.predict(list_of_numpy_images, device=self.device, conf=self.threshold, verbose=False)
        for pred in predictions:
            single_result = pred.boxes
            pred_class = [self.model.names[i] for i in single_result.cls.tolist()]
            pred_score = single_result.conf.tolist()
            pred_score = [round(conf, 2) for conf in single_result.conf.tolist()]
            pred_boxes = single_result.xyxy.tolist()
            sorted_list = list(map(lambda i: i < self.threshold, pred_score))
            t = sorted_list.index(True) if True in sorted_list else len(sorted_list)
            outcome.append({'labels': pred_class[:t], 'bboxes': pred_boxes[:t], 'scores': pred_score[:t]})
        return pd.DataFrame(outcome, columns=['labels', 'bboxes', 'scores'])

    def to_device(self, device: str):
        self.device = device
        return self

def missing_io_signature_helper() -> str:
    """Helper function to print the error message when the input/output signature is missing

    Args:
        io_type (str, optional): "input" or "output". Defaults to "input".

    Returns:
        str: Error message
    """
    signature_template = '\n    You can define the io signature using the decorator as follows:\n    @forward(\n        input_signatures=[\n            PandasDataframe(\n                columns=[<List of cols>],\n                column_types=[<List of col types>],\n                column_shapes=[<List of col shapes>],\n            )\n        ],\n        output_signatures=[\n            PandasDataframe(\n                columns=[<List of cols>],\n                column_types=[<List of col types>],\n                column_shapes=[<List of col shapes>],\n            )\n        ],\n    )\n    More information on the how to create the forward decorator can be found here:\n    https://evadb.readthedocs.io/en/stable/source/reference/ai/custom-ai-function.html#part-1-writing-a-custom-function\n    '
    return signature_template

class AbstractMediaStorageEngine(AbstractStorageEngine):

    def __init__(self, db: EvaDBDatabase):
        super().__init__(db)
        self._rdb_handler: SQLStorageEngine = SQLStorageEngine(db)

    def _get_metadata_table(self, table: TableCatalogEntry):
        return self.db.catalog().get_multimedia_metadata_table_catalog_entry(table)

    def _create_metadata_table(self, table: TableCatalogEntry):
        return self.db.catalog().create_and_insert_multimedia_metadata_table_catalog_entry(table)

    def _xform_file_url_to_file_name(self, file_url: Path) -> str:
        file_path_str = str(file_url)
        file_path = re.sub('[^a-zA-Z0-9 \\.\\n]', '_', file_path_str)
        return file_path

    def create(self, table: TableCatalogEntry, if_not_exists=True):
        """
        Create the directory to store the images.
        Create a sqlite table to persist the file urls
        """
        dir_path = Path(table.file_url)
        try:
            dir_path.mkdir(parents=True)
        except FileExistsError:
            if if_not_exists:
                return True
            error = 'Failed to load the image as directory                         already exists: {}'.format(dir_path)
            logger.error(error)
            raise FileExistsError(error)
        self._rdb_handler.create(self._create_metadata_table(table))
        return True

    def drop(self, table: TableCatalogEntry):
        try:
            dir_path = Path(table.file_url)
            shutil.rmtree(str(dir_path))
            metadata_table = self._get_metadata_table(table)
            self._rdb_handler.drop(metadata_table)
            self.db.catalog().delete_table_catalog_entry(metadata_table)
        except Exception as e:
            err_msg = f'Failed to drop the image table {e}'
            logger.exception(err_msg)
            raise Exception(err_msg)

    def delete(self, table: TableCatalogEntry, rows: Batch):
        try:
            media_metadata_table = self._get_metadata_table(table)
            for media_file_path in rows.file_paths():
                dst_file_name = self._xform_file_url_to_file_name(Path(media_file_path))
                image_file = Path(table.file_url) / dst_file_name
                self._rdb_handler.delete(media_metadata_table, where_clause={media_metadata_table.identifier_column: str(media_file_path)})
                image_file.unlink()
        except Exception as e:
            error = f'Deleting file path {media_file_path} failed with exception {e}'
            logger.exception(error)
            raise RuntimeError(error)
        return True

    def write(self, table: TableCatalogEntry, rows: Batch):
        try:
            dir_path = Path(table.file_url)
            copied_files = []
            for media_file_path in rows.file_paths():
                media_file = Path(media_file_path)
                dst_file_name = self._xform_file_url_to_file_name(media_file)
                dst_path = dir_path / dst_file_name
                if dst_path.exists():
                    raise FileExistsError(f'Duplicate File: {media_file} already exists in the table {table.name}')
                src_path = Path.cwd() / media_file
                os.symlink(src_path, dst_path)
                copied_files.append(dst_path)
            self._rdb_handler.write(self._get_metadata_table(table), Batch(pd.DataFrame({'file_url': list(rows.file_paths())})))
        except Exception as e:
            for file in copied_files:
                logger.info(f'Rollback file {file}')
                file.unlink()
            logger.exception(str(e))
            raise RuntimeError(str(e))
        else:
            return True

    def rename(self, old_table: TableCatalogEntry, new_name: TableInfo):
        try:
            self.db.catalog().rename_table_catalog_entry(old_table, new_name)
        except Exception as e:
            raise Exception(f'Failed to rename table {new_name} with exception {e}')

