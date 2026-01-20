# Cluster 6

def remove_directory_contents(dir_path):
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.warning(f'Failed to delete {file_path}. Reason: {str(e)}')

@pytest.mark.notparallel
class InsertExecutorTest(unittest.TestCase):

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        self.video_file_path = create_sample_video()
        query = 'CREATE TABLE IF NOT EXISTS CSVTable\n            (\n                name TEXT(100)\n            );\n        '
        execute_query_fetch_all(self.evadb, query)
        query = 'CREATE TABLE IF NOT EXISTS books\n            (\n                name    TEXT(100),\n                author  TEXT(100),\n                year    INTEGER\n            );\n        '
        execute_query_fetch_all(self.evadb, query)

    def tearDown(self):
        shutdown_ray()
        file_remove('dummy.avi')
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS books;')

    @unittest.skip('Not supported in current version')
    def test_should_load_video_in_table(self):
        query = f"LOAD VIDEO '{self.video_file_path}' INTO MyVideo;"
        execute_query_fetch_all(self.evadb, query)
        insert_query = ' INSERT INTO MyVideo (id, data) VALUES\n            (40, [[40, 40, 40], [40, 40, 40]],\n                 [[40, 40, 40], [40, 40, 40]]);'
        execute_query_fetch_all(self.evadb, insert_query)
        insert_query_2 = ' INSERT INTO MyVideo (id, data) VALUES\n        ( 41, [[41, 41, 41] , [41, 41, 41]],\n                [[41, 41, 41], [41, 41, 41]]);'
        execute_query_fetch_all(self.evadb, insert_query_2)
        query = 'SELECT id, data FROM MyVideo WHERE id = 40'
        batch = execute_query_fetch_all(self.evadb, query)
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['data'][0], np.array([[[40, 40, 40], [40, 40, 40]], [[40, 40, 40], [40, 40, 40]]])))
        query = 'SELECT id, data FROM MyVideo WHERE id = 41;'
        batch = execute_query_fetch_all(self.evadb, query)
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['data'][0], np.array([[[41, 41, 41], [41, 41, 41]], [[41, 41, 41], [41, 41, 41]]])))

    def test_should_insert_tuples_in_table(self):
        data = pd.read_csv('./test/data/features.csv')
        for i in data.iterrows():
            logger.info(i[1][1])
            query = f"INSERT INTO CSVTable (name) VALUES (\n                            '{i[1][1]}'\n                        );"
            logger.info(query)
            batch = execute_query_fetch_all(self.evadb, query)
        query = 'SELECT name FROM CSVTable;'
        batch = execute_query_fetch_all(self.evadb, query)
        logger.info(batch)
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['csvtable.name'].array, np.array(['test_evadb/similarity/data/sad.jpg', 'test_evadb/similarity/data/happy.jpg', 'test_evadb/similarity/data/angry.jpg'])))
        query = "SELECT name FROM CSVTable WHERE name LIKE '.*(sad|happy)';"
        batch = execute_query_fetch_all(self.evadb, query)
        self.assertEqual(len(batch._frames), 2)

    def test_insert_one_tuple_in_table(self):
        query = "\n            INSERT INTO books (name, author, year) VALUES (\n                'Harry Potter', 'JK Rowling', 1997\n            );\n        "
        execute_query_fetch_all(self.evadb, query)
        query = 'SELECT * FROM books;'
        batch = execute_query_fetch_all(self.evadb, query)
        logger.info(batch)
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['books.name'].array, np.array(['Harry Potter'])))
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['books.author'].array, np.array(['JK Rowling'])))
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['books.year'].array, np.array([1997])))

    def test_insert_multiple_tuples_in_table(self):
        query = "\n            INSERT INTO books (name, author, year) VALUES\n            ('Fantastic Beasts Collection', 'JK Rowling', 2001),\n            ('Magic Tree House Collection', 'Mary Pope Osborne', 1992),\n            ('Sherlock Holmes', 'Arthur Conan Doyle', 1887);\n        "
        execute_query_fetch_all(self.evadb, query)
        query = 'SELECT * FROM books;'
        batch = execute_query_fetch_all(self.evadb, query)
        logger.info(batch)
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['books.name'].array, np.array(['Fantastic Beasts Collection', 'Magic Tree House Collection', 'Sherlock Holmes'])))
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['books.author'].array, np.array(['JK Rowling', 'Mary Pope Osborne', 'Arthur Conan Doyle'])))
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['books.year'].array, np.array([2001, 1992, 1887])))

class EvaServer:
    """
    Receives messages and offloads them to another task for processing them.
    """

    def __init__(self):
        self._server = None
        self._clients = {}
        self._evadb = None

    async def start_evadb_server(self, db_dir: str, host: string, port: int, custom_db_uri: str=None):
        """
        Start the server
        Server objects are asynchronous context managers.

        hostname: hostname of the server
        port: port of the server
        """
        from pprint import pprint
        pprint(f'EvaDB server started at host {host} and port {port}')
        self._evadb = init_evadb_instance(db_dir, host, port, custom_db_uri)
        self._server = await asyncio.start_server(self.accept_client, host, port)
        mode = self._evadb.catalog().get_configuration_catalog_value('mode')
        init_builtin_functions(self._evadb, mode=mode)
        async with self._server:
            await self._server.serve_forever()
        logger.warn('EvaDB server stopped')

    async def stop_evadb_server(self):
        logger.warn('EvaDB server stopped')
        if self._server is not None:
            await self._server.close()

    async def accept_client(self, client_reader: StreamReader, client_writer: StreamWriter):
        task = asyncio.Task(self.handle_client(client_reader, client_writer))
        self._clients[task] = (client_reader, client_writer)

        def close_client(task):
            del self._clients[task]
            client_writer.close()
            logger.info('End connection')
        logger.info('New client connection')
        task.add_done_callback(close_client)

    async def handle_client(self, client_reader: StreamReader, client_writer: StreamWriter):
        try:
            while True:
                data = await asyncio.wait_for(client_reader.readline(), timeout=None)
                if data == b'':
                    break
                message = data.decode().rstrip()
                logger.debug('Received --|%s|--', message)
                if message.upper() in ['EXIT;', 'QUIT;']:
                    logger.info('Close client')
                    return
                logger.debug('Handle request')
                from evadb.server.command_handler import handle_request
                asyncio.create_task(handle_request(self._evadb, client_writer, message))
        except Exception as e:
            logger.critical('Error reading from client.', exc_info=e)

class LoadMultimediaExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: LoadDataPlan):
        super().__init__(db, node)
        self.media_type = self.node.file_options['file_format']
        if self.media_type == FileFormatType.IMAGE:
            try_to_import_cv2()
        elif self.media_type == FileFormatType.VIDEO:
            try_to_import_decord()
            try_to_import_cv2()

    def exec(self, *args, **kwargs):
        storage_engine = None
        table_obj = None
        try:
            video_files = []
            valid_files = []
            if self.node.file_path.as_posix().startswith('s3:/'):
                s3_dir = Path(self.catalog().get_configuration_catalog_value('s3_download_dir'))
                dst_path = s3_dir / self.node.table_info.table_name
                dst_path.mkdir(parents=True, exist_ok=True)
                video_files = download_from_s3(self.node.file_path, dst_path)
            else:
                video_files = list(iter_path_regex(self.node.file_path))
            valid_files, invalid_files = ([], [])
            if len(video_files) < mp.cpu_count() * 2:
                valid_bitmap = [self._is_media_valid(path) for path in video_files]
            else:
                pool = Pool(mp.cpu_count())
                valid_bitmap = pool.map(self._is_media_valid, video_files)
            if False in valid_bitmap:
                invalid_files = [str(path) for path, is_valid in zip(video_files, valid_bitmap) if not is_valid]
                invalid_files_str = '\n'.join(invalid_files)
                err_msg = f"no valid file found at -- '{invalid_files_str}'."
                logger.error(err_msg)
            valid_files = [str(path) for path, is_valid in zip(video_files, valid_bitmap) if is_valid]
            if not valid_files:
                raise DatasetFileNotFoundError(f"no file found at -- '{str(self.node.file_path)}'.")
            table_info = self.node.table_info
            database_name = table_info.database_name
            table_name = table_info.table_name
            do_create = False
            table_obj = self.catalog().get_table_catalog_entry(table_name, database_name)
            if table_obj:
                msg = f'Adding to an existing table {table_name}.'
                logger.info(msg)
            else:
                table_obj = self.catalog().create_and_insert_multimedia_table_catalog_entry(table_name, self.media_type)
                do_create = True
            storage_engine = StorageEngine.factory(self.db, table_obj)
            if do_create:
                storage_engine.create(table_obj)
            storage_engine.write(table_obj, Batch(pd.DataFrame({'file_path': valid_files})))
        except Exception as e:
            if storage_engine and table_obj:
                self._rollback_load(storage_engine, table_obj, do_create)
            err_msg = f'Load {self.media_type.name} failed: {str(e)}'
            raise ExecutorError(err_msg)
        else:
            yield Batch(pd.DataFrame([f'Number of loaded {self.media_type.name}: {str(len(valid_files))}']))

    def _rollback_load(self, storage_engine: AbstractStorageEngine, table_obj: TableCatalogEntry, do_create: bool):
        if do_create:
            storage_engine.drop(table_obj)

    def _is_media_valid(self, file_path: Path):
        file_path = Path(file_path)
        if validate_media(file_path, self.media_type):
            return True
        return False

def download_from_s3(s3_uri, save_dir):
    """
    Downloads a file from s3 to the local file system
    """
    try_to_import_moto()
    import boto3
    s3_client = boto3.client('s3')
    s3_uri = s3_uri.as_posix()
    bucket_name, regex_key = parse_s3_uri(s3_uri)
    s3_bucket = boto3.resource('s3').Bucket(bucket_name)
    file_save_paths = []
    for obj in s3_bucket.objects.all():
        if re.search(re.sub('\\*', '.*', regex_key), obj.key):
            key = obj.key.replace('/', '_')
            save_path = Path(save_dir) / key
            s3_client.download_file(bucket_name, key, save_path)
            file_save_paths.append(save_path)
    return file_save_paths

def iter_path_regex(path_regex: Path) -> Generator[str, None, None]:
    return glob.iglob(os.path.expanduser(path_regex), recursive=True)

def parse_s3_uri(s3_uri):
    """
    Parses the S3 URI and returns the bucket name and key
    """
    s3_uri = s3_uri.replace('s3:/', '')
    bucket_name, key = s3_uri.split('/', 1)
    return (bucket_name, key)

class Timer:
    """Class used for logging time metrics.

    This is not thread safe"""

    def __init__(self):
        self._start_time = None
        self._total_time = 0.0

    def __enter__(self):
        assert self._start_time is None, 'Concurrent calls are not supported'
        self._start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._start_time is not None, 'exit called with starting the context'
        time_elapsed = time.perf_counter() - self._start_time
        self._total_time += time_elapsed
        self._start_time = None

    @property
    def total_elapsed_time(self):
        return self._total_time

    def log_elapsed_time(self, context: str):
        logger.info('{:s}: {:0.4f} sec'.format(context, self.total_elapsed_time))

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

def init_db(engine: Engine):
    """Create database if doesn't exist and create all tables."""
    if not database_exists(engine.url):
        logger.info('Database does not exist, creating database.')
        create_database(engine.url)
    logger.info('Creating tables')
    BaseModel.metadata.create_all(bind=engine)

def drop_all_tables_except_catalog(engine: Engine):
    """drop all the tables except the catalog"""
    BaseModel.metadata.reflect(bind=engine)
    insp = sqlalchemy.inspect(engine)
    if database_exists(engine.url):
        with contextlib.closing(engine.connect()) as con:
            trans = con.begin()
            for table in reversed(BaseModel.metadata.sorted_tables):
                if table.name not in CATALOG_TABLES:
                    if insp.has_table(table.name):
                        table.drop(con)
            trans.commit()

def truncate_catalog_tables(engine: Engine, tables_not_to_truncate: List[str]=[]):
    """Truncate all the catalog tables"""
    BaseModel.metadata.reflect(bind=engine)
    insp = sqlalchemy.inspect(engine)
    if database_exists(engine.url):
        with contextlib.closing(engine.connect()) as con:
            trans = con.begin()
            for table in reversed(BaseModel.metadata.sorted_tables):
                if table.name not in tables_not_to_truncate:
                    if insp.has_table(table.name):
                        con.execute(table.delete())
            trans.commit()

class CSVReader(AbstractReader):

    def __init__(self, *args, column_list, **kwargs):
        """
        Reads a CSV file and yields frame data.
        Args:
            column_list: list of columns (TupleValueExpression)
            to read from the CSV file
        """
        self._column_list = column_list
        super().__init__(*args, **kwargs)

    def _read(self) -> Iterator[Dict]:

        def convert_csv_string_to_ndarray(row_string):
            """
            Convert a string of comma separated values to a numpy
            float array
            """
            return np.array([np.float32(val) for val in row_string.split(',')])
        logger.info('Reading CSV frames')
        col_list_names = [col.name for col in self._column_list if col.name != IDENTIFIER_COLUMN]
        col_map = {col.name: col for col in self._column_list}
        for chunk in pd.read_csv(self.file_url, chunksize=512, usecols=col_list_names):
            for col in chunk.columns:
                if isinstance(chunk[col].iloc[0], str) and col_map[col].col_object.type.name == 'NDARRAY':
                    chunk[col] = chunk[col].apply(convert_csv_string_to_ndarray)
            for chunk_index, chunk_row in chunk.iterrows():
                yield chunk_row

class AbstractReader(metaclass=ABCMeta):
    """
    Abstract class for defining data reader. All other media readers use this
    abstract class. Media readers are expected to return Batch
    in an iterative manner.

    Attributes:
        file_url (str): path to read data from
    """

    def __init__(self, file_url: str, batch_mem_size: int=30000000):
        if not Path(file_url).exists():
            raise DatasetFileNotFoundError()
        if isinstance(file_url, Path):
            file_url = str(file_url)
        self.file_url = file_url
        self.batch_mem_size = batch_mem_size

    def read(self) -> Iterator[Batch]:
        """
        This calls the sub class read implementation and
        yields the batch to the caller
        """
        data_batch = []
        row_size = None
        for data in self._read():
            if row_size is None:
                row_size = 0
                row_size = get_size(data)
            data_batch.append(data)
            if len(data_batch) * row_size >= self.batch_mem_size:
                yield Batch(pd.DataFrame(data_batch))
                data_batch = []
        if data_batch:
            yield Batch(pd.DataFrame(data_batch))

    @abstractmethod
    def _read(self) -> Iterator[Dict]:
        """
        Every sub class implements it's own logic
        to read the file and yields an object iterator.
        """

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

