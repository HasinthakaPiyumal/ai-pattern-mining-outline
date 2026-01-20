# Cluster 7

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

def generate_file_path(dataset_location: str, name: str='') -> Path:
    """Generates a arbitrary file_path(md5 hash) based on the a random salt
    and name

    Arguments:
        dataset_location(str): parent directory where a file needs to be created
        name (str): Input file_name.

    Returns:
        Path: pathlib.Path object

    """
    dataset_location = Path(dataset_location)
    dataset_location.mkdir(parents=True, exist_ok=True)
    salt = uuid.uuid4().hex
    file_name = hashlib.md5(salt.encode() + name.encode()).hexdigest()
    path = dataset_location / file_name
    return path.resolve()

class CMDClientTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_mock_stdin_reader(self) -> asyncio.StreamReader:
        stdin_reader = asyncio.StreamReader()
        stdin_reader.feed_data(b'EXIT;\n')
        stdin_reader.feed_eof()
        return stdin_reader

    @patch('evadb.evadb_cmd_client.start_cmd_client')
    @patch('evadb.server.interpreter.create_stdin_reader')
    def test_evadb_client(self, mock_stdin_reader, mock_client):
        mock_stdin_reader.return_value = self.get_mock_stdin_reader()
        mock_client.side_effect = Exception('Test')

        async def test():
            with self.assertRaises(Exception):
                await evadb_client('0.0.0.0', 8803)
        asyncio.run(test())
        mock_client.reset_mock()
        mock_client.side_effect = KeyboardInterrupt

        async def test2():
            await evadb_client('0.0.0.0', 8803)
        asyncio.run(test2())

    @patch('argparse.ArgumentParser.parse_known_args')
    @patch('evadb.evadb_cmd_client.start_cmd_client')
    def test_evadb_client_with_cmd_arguments(self, mock_start_cmd_client, mock_parse_known_args):
        mock_parse_known_args.return_value = (argparse.Namespace(host='127.0.0.1', port='8800'), [])
        main()
        mock_start_cmd_client.assert_called_once_with('127.0.0.1', '8800')

    @patch('argparse.ArgumentParser.parse_known_args')
    @patch('evadb.evadb_cmd_client.start_cmd_client')
    def test_main_without_cmd_arguments(self, mock_start_cmd_client, mock_parse_known_args):
        mock_parse_known_args.return_value = (argparse.Namespace(host=None, port=None), [])
        main()
        mock_start_cmd_client.assert_called_once_with(BASE_EVADB_CONFIG['host'], BASE_EVADB_CONFIG['port'])

def main():
    parser = argparse.ArgumentParser(description='EvaDB Client')
    parser.add_argument('--host', help='Specify the host address of the server you want to connect to.')
    parser.add_argument('--port', help='Specify the port number of the server you want to connect to.')
    args, unknown = parser.parse_known_args()
    host = args.host if args.host else BASE_EVADB_CONFIG['host']
    port = args.port if args.port else BASE_EVADB_CONFIG['port']
    asyncio.run(evadb_client(host, port))

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

class CreateExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: CreatePlan):
        super().__init__(db, node)

    def exec(self, *args, **kwargs):
        is_native_table = self.node.table_info.database_name is not None
        check_if_exists = handle_if_not_exists(self.catalog(), self.node.table_info, self.node.if_not_exists)
        name = self.node.table_info.table_name
        if check_if_exists:
            yield Batch(pd.DataFrame([f'Table {name} already exists']))
            return
        create_table_done = False
        logger.debug(f'Creating table {self.node.table_info}')
        if not is_native_table:
            catalog_entry = self.catalog().create_and_insert_table_catalog_entry(self.node.table_info, self.node.column_list)
        else:
            catalog_entry = create_table_catalog_entry_for_native_table(self.node.table_info, self.node.column_list)
        storage_engine = StorageEngine.factory(self.db, catalog_entry)
        try:
            storage_engine.create(table=catalog_entry)
            create_table_done = True
            msg = f'The table {name} has been successfully created'
            if self.children != []:
                assert len(self.children) == 1, 'Create table from query expects 1 child, finds {}'.format(len(self.children))
                child = self.children[0]
                rows = 0
                for batch in child.exec():
                    batch.drop_column_alias()
                    storage_engine.write(catalog_entry, batch)
                    rows += len(batch)
                msg = f'The table {name} has been successfully created with {rows} rows.'
            yield Batch(pd.DataFrame([msg]))
        except Exception as e:
            with contextlib.suppress(CatalogError):
                if create_table_done:
                    storage_engine.drop(catalog_entry)
            with contextlib.suppress(CatalogError):
                self.catalog().delete_table_catalog_entry(catalog_entry)
            raise e

def handle_if_not_exists(catalog: 'CatalogManager', table_info: TableInfo, if_not_exist=False):
    if catalog.check_table_exists(table_info.table_name, table_info.database_name):
        err_msg = 'Table: {} already exists'.format(table_info)
        if if_not_exist:
            logger.warn(err_msg)
            return True
        else:
            logger.error(err_msg)
            raise ExecutorError(err_msg)
    else:
        return False

def create_table_catalog_entry_for_native_table(table_info: TableInfo, column_list: List[ColumnDefinition]):
    column_catalog_entries = xform_column_definitions_to_catalog_entries(column_list)
    table_catalog_entry = TableCatalogEntry(name=table_info.table_name, file_url=None, table_type=TableType.NATIVE_DATA, columns=column_catalog_entries, database_name=table_info.database_name)
    return table_catalog_entry

def xform_column_definitions_to_catalog_entries(col_list: List[ColumnDefinition]) -> List[ColumnCatalogEntry]:
    """Create column catalog entries for the input parsed column list.

    Arguments:
        col_list {List[ColumnDefinition]} -- parsed col list to be created
    """
    result_list = []
    for col in col_list:
        column_entry = ColumnCatalogEntry(name=col.name, type=col.type, array_type=col.array_type, array_dimensions=col.dimension, is_nullable=col.cci.nullable)
        result_list.append(column_entry)
    return result_list

class FunctionExpression(AbstractExpression):
    """
    Consider FunctionExpression: ObjDetector -> (labels, boxes)

    `output`: If the user wants only subset of outputs. Eg,
    ObjDetector.labels the parser with set output to 'labels'

    `output_objs`: It is populated by the binder. In case the
    output is None, the binder sets output_objs to list of all
    output columns of the FunctionExpression. Eg, ['labels',
    'boxes']. Otherwise, only the output columns.

    FunctionExpression also needs to prepend its alias to all the
    projected columns. This is important as other parts of the query
    might be assessing the results using alias. Eg,

    `Select Detector.labels
     FROM Video JOIN LATERAL ObjDetector AS Detector;`
    """

    def __init__(self, func: Callable, name: str, output: str=None, alias: Alias=None, **kwargs):
        super().__init__(ExpressionType.FUNCTION_EXPRESSION, **kwargs)
        self._context = Context()
        self._name = name
        self._function = func
        self._function_instance = None
        self._output = output
        self.alias = alias
        self.function_obj: FunctionCatalogEntry = None
        self.output_objs: List[FunctionIOCatalogEntry] = []
        self.projection_columns: List[str] = []
        self._cache: FunctionExpressionCache = None
        self._stats = FunctionStats()

    @property
    def name(self):
        return self._name

    @property
    def output(self):
        return self._output

    @property
    def col_alias(self):
        col_alias_list = []
        if self.alias is not None:
            for col in self.alias.col_names:
                col_alias_list.append('{}.{}'.format(self.alias.alias_name, col))
        return col_alias_list

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, func: Callable):
        self._function = func

    def enable_cache(self, cache: 'FunctionExpressionCache'):
        self._cache = cache
        return self

    def has_cache(self):
        return self._cache is not None

    def consolidate_stats(self):
        if self.function_obj is None:
            return
        if self.has_cache() and self._stats.cache_misses > 0:
            cost_per_func_call = self._stats.timer.total_elapsed_time / self._stats.cache_misses
        else:
            cost_per_func_call = self._stats.timer.total_elapsed_time / self._stats.num_calls
        if abs(self._stats.prev_cost - cost_per_func_call) > cost_per_func_call / 10:
            self._stats.prev_cost = cost_per_func_call

    def evaluate(self, batch: Batch, **kwargs) -> Batch:
        func = self._gpu_enabled_function()
        with self._stats.timer:
            outcomes = self._apply_function_expression(func, batch, **kwargs)
            if outcomes.frames.empty is False:
                outcomes = outcomes.project(self.projection_columns)
                outcomes.modify_column_alias(self.alias)
        self._stats.num_calls += len(batch)
        try:
            self.consolidate_stats()
        except Exception as e:
            logger.warn(f'Persisting FunctionExpression {str(self)} stats failed with {str(e)}')
        return outcomes

    def signature(self) -> str:
        """It constructs the signature of the function expression.
        It traverses the children (function arguments) and compute signature for each
        child. The output is in the form `function_name[row_id](arg1, arg2, ...)`.

        Returns:
            str: signature string
        """
        child_sigs = []
        for child in self.children:
            child_sigs.append(child.signature())
        func_sig = f'{self.name}[{self.function_obj.row_id}]({','.join(child_sigs)})'
        return func_sig

    def _gpu_enabled_function(self):
        if self._function_instance is None:
            self._function_instance = self.function()
            if isinstance(self._function_instance, GPUCompatible):
                device = self._context.gpu_device()
                if device != NO_GPU:
                    self._function_instance = self._function_instance.to_device(device)
        return self._function_instance

    def _apply_function_expression(self, func: Callable, batch: Batch, **kwargs):
        """
        If cache is not enabled, call the func on the batch and return.
        If cache is enabled:
        (1) iterate over the input batch rows and check if we have the value in the
        cache;
        (2) for all cache miss rows, call the func;
        (3) iterate over each cache miss row and store the results in the cache;
        (4) stitch back the partial cache results with the new func calls.
        """
        func_args = Batch.merge_column_wise([child.evaluate(batch, **kwargs) for child in self.children])
        if not self._cache:
            return func_args.apply_function_expression(func)
        output_cols = [obj.name for obj in self.function_obj.outputs]
        results = np.full([len(batch), len(output_cols)], None)
        cache_keys = func_args
        if self._cache.key:
            cache_keys = Batch.merge_column_wise([child.evaluate(batch, **kwargs) for child in self._cache.key])
            assert len(cache_keys) == len(batch), 'Not all rows have the cache key'
        cache_miss = np.full(len(batch), True)
        for idx, (_, key) in enumerate(cache_keys.iterrows()):
            val = self._cache.store.get(key.to_numpy())
            results[idx] = val
            cache_miss[idx] = val is None
        self._stats.cache_misses += sum(cache_miss)
        if cache_miss.any():
            func_args = func_args[list(cache_miss)]
            cache_miss_results = func_args.apply_function_expression(func)
            missing_keys = cache_keys[list(cache_miss)]
            for key, value in zip(missing_keys.iterrows(), cache_miss_results.iterrows()):
                self._cache.store.set(key[1].to_numpy(), value[1].to_numpy())
            results[cache_miss] = cache_miss_results.to_numpy()
        return Batch(pd.DataFrame(results, columns=output_cols))

    def __str__(self) -> str:
        args = [str(child) for child in self.children]
        expr_str = f'{self.name}({','.join(args)})'
        return expr_str

    def __eq__(self, other):
        is_subtree_equal = super().__eq__(other)
        if not isinstance(other, FunctionExpression):
            return False
        return is_subtree_equal and self.name == other.name and (self.output == other.output) and (self.alias == other.alias) and (self.function == other.function) and (self.output_objs == other.output_objs) and (self._cache == other._cache)

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.name, self.output, self.alias, self.function, tuple(self.output_objs), self._cache))

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

class TableCatalog(BaseModel):
    """The `TableCatalog` catalog stores information about all tables (structured, media, etc.) and materialized views. It has the following columns, not all of which are relevant for all table types.
    `_row_id:` an autogenerated unique identifier.
    `_name:` the name of the table, view, etc.
    `_file_url:` the path to the data file on disk
    `_table_type:` the type of the table (refer to TableType).
    """
    __tablename__ = 'table_catalog'
    _name = Column('name', String(100), unique=True)
    _file_url = Column('file_url', String(100))
    _identifier_column = Column('identifier_column', String(100))
    _table_type = Column('table_type', Enum(TableType))
    _columns = relationship('ColumnCatalog', back_populates='_table_catalog', cascade='all, delete, delete-orphan')

    def __init__(self, name: str, file_url: str, table_type: int, identifier_column='id'):
        self._name = name
        self._file_url = file_url
        self._identifier_column = identifier_column
        self._table_type = table_type

    def as_dataclass(self) -> 'TableCatalogEntry':
        column_entries = [col_obj.as_dataclass() for col_obj in self._columns]
        return TableCatalogEntry(row_id=self._row_id, name=self._name, file_url=self._file_url, identifier_column=self._identifier_column, table_type=self._table_type, columns=column_entries)

class FaceDetector(AbstractClassifierFunction, GPUCompatible):
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score
    """

    def setup(self, threshold=0.85):
        self.threshold = threshold
        try_to_import_torch()
        try_to_import_torchvision()
        try_to_import_facenet_pytorch()
        from facenet_pytorch import MTCNN
        self.model = MTCNN()

    @property
    def name(self) -> str:
        return 'FaceDetector'

    def to_device(self, device: str):
        try_to_import_facenet_pytorch()
        import torch
        from facenet_pytorch import MTCNN
        gpu = 'cuda:{}'.format(device)
        self.model = MTCNN(device=torch.device(gpu))
        return self

    @property
    def labels(self) -> List[str]:
        return []

    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (np.ndarray): Frames on which predictions need
            to be performed
        Returns:
            face boxes (List[List[BoundingBox]])
        """
        frames_list = frames.transpose().values.tolist()[0]
        frames = np.asarray(frames_list)
        detections = self.model.detect(frames)
        boxes, scores = detections
        outcome = []
        for frame_boxes, frame_scores in zip(boxes, scores):
            pred_boxes = []
            pred_scores = []
            if frame_boxes is not None and frame_scores is not None:
                if not np.isnan(pred_boxes):
                    pred_boxes = np.asarray(frame_boxes, dtype='int')
                    pred_scores = frame_scores
                else:
                    logger.warn(f'Nan entry in box {frame_boxes}')
            outcome.append({'bboxes': pred_boxes, 'scores': pred_scores})
        return pd.DataFrame(outcome, columns=['bboxes', 'scores'])

