# Cluster 91

class JobSchedulerTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_dummy_job_catalog_entry(self, active, job_name, next_run):
        return JobCatalogEntry(name=job_name, queries=None, start_time=None, end_time=None, repeat_interval=None, active=active, next_scheduled_run=next_run, created_at=None, updated_at=None)

    def test_sleep_time_calculation(self):
        past_job = self.get_dummy_job_catalog_entry(True, 'past_job', datetime.now() - timedelta(seconds=10))
        future_job = self.get_dummy_job_catalog_entry(True, 'future_job', datetime.now() + timedelta(seconds=20))
        job_scheduler = JobScheduler(MagicMock())
        self.assertEqual(job_scheduler._get_sleep_time(past_job), 0)
        self.assertGreaterEqual(job_scheduler._get_sleep_time(future_job), 10)
        self.assertEqual(job_scheduler._get_sleep_time(None), 30)

    def test_update_next_schedule_run(self):
        future_time = datetime.now() + timedelta(seconds=1000)
        job_scheduler = JobScheduler(MagicMock())
        job_entry = self.get_dummy_job_catalog_entry(True, 'job', datetime.now())
        job_entry.end_time = future_time
        status, next_run = job_scheduler._update_next_schedule_run(job_entry)
        self.assertEqual(status, False, 'status for one time job should be false')
        job_entry.end_time = future_time
        job_entry.repeat_interval = 120
        expected_next_run = datetime.now() + timedelta(seconds=120)
        status, next_run = job_scheduler._update_next_schedule_run(job_entry)
        self.assertEqual(status, True, 'status for recurring time job should be true')
        self.assertGreaterEqual(next_run, expected_next_run)
        job_entry.end_time = datetime.now() + timedelta(seconds=60)
        job_entry.repeat_interval = 120
        expected_next_run = datetime.now() + timedelta(seconds=120)
        status, next_run = job_scheduler._update_next_schedule_run(job_entry)
        self.assertEqual(status, False, 'status for rexpired ecurring time job should be false')
        self.assertLessEqual(next_run, datetime.now())

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

def validate_media(file_path: Path, media_type: FileFormatType) -> bool:
    if media_type == FileFormatType.VIDEO:
        return validate_video(file_path)
    elif media_type == FileFormatType.IMAGE:
        return validate_image(file_path)
    elif media_type == FileFormatType.DOCUMENT:
        return validate_document(file_path)
    elif media_type == FileFormatType.PDF:
        return validate_pdf(file_path)
    else:
        raise ValueError(f'Unsupported Media type {str(media_type)}')

class DropObjectExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: DropObjectPlan):
        super().__init__(db, node)

    def exec(self, *args, **kwargs):
        """Drop Object executor"""
        if self.node.object_type == ObjectType.TABLE:
            yield self._handle_drop_table(self.node.name, self.node.if_exists)
        elif self.node.object_type == ObjectType.INDEX:
            yield self._handle_drop_index(self.node.name, self.node.if_exists)
        elif self.node.object_type == ObjectType.FUNCTION:
            yield self._handle_drop_function(self.node.name, self.node.if_exists)
        elif self.node.object_type == ObjectType.DATABASE:
            yield self._handle_drop_database(self.node.name, self.node.if_exists)
        elif self.node.object_type == ObjectType.JOB:
            yield self._handle_drop_job(self.node.name, self.node.if_exists)

    def _handle_drop_table(self, table_name: str, if_exists: bool):
        if not self.catalog().check_table_exists(table_name):
            err_msg = 'Table: {} does not exist'.format(table_name)
            if if_exists:
                return Batch(pd.DataFrame([err_msg]))
            else:
                raise ExecutorError(err_msg)
        table_obj = self.catalog().get_table_catalog_entry(table_name)
        storage_engine = StorageEngine.factory(self.db, table_obj)
        logger.debug(f'Dropping table {table_name}')
        storage_engine.drop(table=table_obj)
        for col_obj in table_obj.columns:
            for cache in col_obj.dep_caches:
                self.catalog().drop_function_cache_catalog_entry(cache)
        assert self.catalog().delete_table_catalog_entry(table_obj), 'Failed to drop {}'.format(table_name)
        return Batch(pd.DataFrame({'Table Successfully dropped: {}'.format(table_name)}, index=[0]))

    def _handle_drop_function(self, function_name: str, if_exists: bool):
        if not self.catalog().get_function_catalog_entry_by_name(function_name):
            err_msg = f'Function {function_name} does not exist, therefore cannot be dropped.'
            if if_exists:
                logger.warning(err_msg)
                return Batch(pd.DataFrame([err_msg]))
            else:
                raise RuntimeError(err_msg)
        else:
            function_entry = self.catalog().get_function_catalog_entry_by_name(function_name)
            for cache in function_entry.dep_caches:
                self.catalog().drop_function_cache_catalog_entry(cache)
            self.catalog().delete_function_catalog_entry_by_name(function_name)
            return Batch(pd.DataFrame({f'Function {function_name} successfully dropped'}, index=[0]))

    def _handle_drop_index(self, index_name: str, if_exists: bool):
        index_obj = self.catalog().get_index_catalog_entry_by_name(index_name)
        if not index_obj:
            err_msg = f'Index {index_name} does not exist, therefore cannot be dropped.'
            if if_exists:
                logger.warning(err_msg)
                return Batch(pd.DataFrame([err_msg]))
            else:
                raise RuntimeError(err_msg)
        else:
            index = VectorStoreFactory.init_vector_store(index_obj.type, index_obj.name, **handle_vector_store_params(index_obj.type, index_obj.save_file_path, self.catalog))
            assert index is not None, f'Failed to initialize the vector store handler for {index_obj.type}'
            if index:
                index.delete()
            self.catalog().drop_index_catalog_entry(index_name)
            return Batch(pd.DataFrame({f'Index {index_name} successfully dropped'}, index=[0]))

    def _handle_drop_database(self, database_name: str, if_exists: bool):
        db_catalog_entry = self.catalog().get_database_catalog_entry(database_name)
        if not db_catalog_entry:
            err_msg = f'Database {database_name} does not exist, therefore cannot be dropped.'
            if if_exists:
                logger.warning(err_msg)
                return Batch(pd.DataFrame([err_msg]))
            else:
                raise RuntimeError(err_msg)
        logger.debug(f'Dropping database {database_name}')
        self.catalog().drop_database_catalog_entry(db_catalog_entry)
        return Batch(pd.DataFrame({f'Database {database_name} successfully dropped'}, index=[0]))

    def _handle_drop_job(self, job_name: str, if_exists: bool):
        job_catalog_entry = self.catalog().get_job_catalog_entry(job_name)
        if not job_catalog_entry:
            err_msg = f'Job {job_name} does not exist, therefore cannot be dropped.'
            if if_exists:
                logger.warning(err_msg)
                return Batch(pd.DataFrame([err_msg]))
            else:
                raise RuntimeError(err_msg)
        logger.debug(f'Dropping Job {job_name}')
        self.catalog().drop_job_catalog_entry(job_catalog_entry)
        return Batch(pd.DataFrame({f'Job {job_name} successfully dropped'}, index=[0]))

def handle_vector_store_params(vector_store_type: VectorStoreType, index_path: str, catalog) -> dict:
    """Handle vector store parameters based on the vector store type and index path.

    Args:
        vector_store_type (VectorStoreType): The type of vector store.
        index_path (str): The path to store the index.

    Returns:
        dict: Dictionary containing the appropriate vector store parameters.


    Raises:
        ValueError: If the vector store type in the node is not supported.
    """
    if vector_store_type == VectorStoreType.FAISS:
        return {'index_path': index_path}
    elif vector_store_type == VectorStoreType.QDRANT:
        return {'index_db': str(Path(index_path).parent)}
    elif vector_store_type == VectorStoreType.CHROMADB:
        return {'index_path': str(Path(index_path).parent)}
    elif vector_store_type == VectorStoreType.PINECONE:
        return {'PINECONE_API_KEY': catalog().get_configuration_catalog_value('PINECONE_API_KEY'), 'PINECONE_ENV': catalog().get_configuration_catalog_value('PINECONE_ENV')}
    elif vector_store_type == VectorStoreType.WEAVIATE:
        return {'WEAVIATE_API_KEY': catalog().get_configuration_catalog_value('WEAVIATE_API_KEY'), 'WEAVIATE_API_URL': catalog().get_configuration_catalog_value('WEAVIATE_API_URL')}
    elif vector_store_type == VectorStoreType.MILVUS:
        return {'MILVUS_URI': catalog().get_configuration_catalog_value('MILVUS_URI'), 'MILVUS_USER': catalog().get_configuration_catalog_value('MILVUS_USER'), 'MILVUS_PASSWORD': catalog().get_configuration_catalog_value('MILVUS_PASSWORD'), 'MILVUS_DB_NAME': catalog().get_configuration_catalog_value('MILVUS_DB_NAME'), 'MILVUS_TOKEN': catalog().get_configuration_catalog_value('MILVUS_TOKEN')}
    else:
        raise ValueError('Unsupported vector store type: {}'.format(vector_store_type))

class CreateDatabaseExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: CreateDatabaseStatement):
        super().__init__(db, node)

    def exec(self, *args, **kwargs):
        db_catalog_entry = self.catalog().get_database_catalog_entry(self.node.database_name)
        if db_catalog_entry is not None:
            if self.node.if_not_exists:
                msg = f'{self.node.database_name} already exists, nothing added.'
                yield Batch(pd.DataFrame([msg]))
                return
            else:
                raise ExecutorError(f'{self.node.database_name} already exists.')
        logger.debug(f'Trying to connect to the provided engine {self.node.engine} with params {self.node.param_dict}')
        with get_database_handler(self.node.engine, **self.node.param_dict):
            pass
        logger.debug(f'Creating database {self.node}')
        self.catalog().insert_database_catalog_entry(self.node.database_name, self.node.engine, self.node.param_dict)
        yield Batch(pd.DataFrame([f'The database {self.node.database_name} has been successfully created.']))

@contextmanager
def get_database_handler(engine: str, **kwargs):
    handler = _get_database_handler(engine, **kwargs)
    try:
        resp = handler.connect()
        if not resp.status:
            raise ExecutorError(f'Cannot establish connection due to {resp.error}')
        yield handler
    finally:
        handler.disconnect()

class UseExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: UseStatement):
        super().__init__(db, node)
        self._database_name = node.database_name
        self._query_string = node.query_string

    def exec(self, *args, **kwargs) -> Iterator[Batch]:
        db_catalog_entry = self.db.catalog().get_database_catalog_entry(self._database_name)
        if db_catalog_entry is None:
            raise ExecutorError(f'{self._database_name} data source does not exist. Use CREATE DATABASE to add a new data source.')
        with get_database_handler(db_catalog_entry.engine, **db_catalog_entry.params) as handler:
            resp = handler.execute_native_query(self._query_string)
        if resp and resp.error is None:
            yield Batch(resp.data)
        else:
            raise ExecutorError(resp.error)

class VectorIndexScanExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: VectorIndexScanPlan):
        super().__init__(db, node)
        self.index_name = node.index.name
        self.vector_store_type = node.index.type
        self.feat_column = node.index.feat_column
        self.limit_count = node.limit_count
        self.search_query_expr = node.search_query_expr

    def exec(self, *args, **kwargs) -> Iterator[Batch]:
        if self.vector_store_type == VectorStoreType.PGVECTOR:
            return self._native_vector_index_scan()
        else:
            return self._evadb_vector_index_scan(*args, **kwargs)

    def _get_search_query_results(self):
        dummy_batch = Batch(frames=pd.DataFrame({'0': [0]}))
        search_batch = self.search_query_expr.evaluate(dummy_batch)
        feature_col_name = self.search_query_expr.output_objs[0].name
        search_batch.drop_column_alias()
        search_feat = search_batch.column_as_numpy_array(feature_col_name)[0]
        search_feat = search_feat.reshape(1, -1)
        return search_feat

    def _native_vector_index_scan(self):
        search_feat = self._get_search_query_results()
        search_feat = search_feat.reshape(-1).tolist()
        tb_catalog_entry = list(self.node.find_all(StoragePlan))[0].table
        db_catalog_entry = self.db.catalog().get_database_catalog_entry(tb_catalog_entry.database_name)
        with get_database_handler(db_catalog_entry.engine, **db_catalog_entry.params) as handler:
            resp = handler.execute_native_query(f"SELECT * FROM {tb_catalog_entry.name}\n                                                ORDER BY {self.feat_column.name} <-> '{search_feat}'\n                                                LIMIT {self.limit_count}")
            if resp.error is not None:
                raise ExecutorError(f'Native index can encounters {resp.error}')
            res = Batch(frames=resp.data)
            res.modify_column_alias(tb_catalog_entry.name)
            yield res

    def _evadb_vector_index_scan(self, *args, **kwargs):
        index_catalog_entry = self.catalog().get_index_catalog_entry_by_name(self.index_name)
        self.index_path = index_catalog_entry.save_file_path
        self.index = VectorStoreFactory.init_vector_store(self.vector_store_type, self.index_name, **handle_vector_store_params(self.vector_store_type, self.index_path, self.db.catalog))
        search_feat = self._get_search_query_results()
        index_result = self.index.query(VectorIndexQuery(search_feat, self.limit_count.value))
        row_num_np = index_result.ids
        row_num_col_name = None
        num_required_results = self.limit_count.value
        if len(index_result.ids) < self.limit_count.value:
            num_required_results = len(index_result.ids)
            logger.warning(f'The index {self.index_name} returned only {num_required_results} results, which is fewer than the required {self.limit_count.value}.')
        final_df = pd.DataFrame()
        res_data_list = []
        row_num_df = pd.DataFrame({'row_num_np': row_num_np})
        for batch in self.children[0].exec(**kwargs):
            if not row_num_col_name:
                column_list = batch.columns
                row_num_alias = get_row_num_column_alias(column_list)
                row_num_col_name = '{}.{}'.format(row_num_alias, ROW_NUM_COLUMN)
            if not batch.frames[row_num_col_name].isin(row_num_df['row_num_np']).any():
                continue
            for index, row in batch.frames.iterrows():
                row_dict = row.to_dict()
                res_data_list.append(row_dict)
        result_df = pd.DataFrame(res_data_list)
        final_df = pd.merge(row_num_df, result_df, left_on='row_num_np', right_on=row_num_col_name, how='inner')
        if 'row_num_np' in final_df:
            del final_df['row_num_np']
        yield Batch(final_df)

def get_row_num_column_alias(column_list):
    for column in column_list:
        alias, col_name = column.split('.')
        if col_name == ROW_NUM_COLUMN:
            return alias

class OrderByExecutor(AbstractExecutor):
    """
    Sort the frames which satisfy the condition

    Arguments:
        node (AbstractPlan): The OrderBy Plan

    """

    def __init__(self, db: EvaDBDatabase, node: OrderByPlan):
        super().__init__(db, node)
        self._orderby_list = node.orderby_list
        self._columns = node.columns
        self._sort_types = node.sort_types
        self.batch_sizes = []

    def _extract_column_name(self, col):
        col_name = []
        if isinstance(col, TupleValueExpression):
            col_name += [col.col_alias]
        elif isinstance(col, FunctionExpression):
            col_name += col.col_alias
        else:
            raise ExecutorError('Expression type {} is not supported.'.format(type(col)))
        return col_name

    def extract_column_names(self):
        """extracts the string name of the column"""
        col_name_list = []
        for col in self._columns:
            col_name_list += self._extract_column_name(col)
        return col_name_list

    def extract_sort_types(self):
        """extracts the sort type for the column"""
        sort_type_bools = []
        for st in self._sort_types:
            if st is ParserOrderBySortType.ASC:
                sort_type_bools.append(True)
            else:
                sort_type_bools.append(False)
        return sort_type_bools

    def exec(self, *args, **kwargs) -> Iterator[Batch]:
        child_executor = self.children[0]
        aggregated_batch_list = []
        for batch in child_executor.exec(**kwargs):
            self.batch_sizes.append(len(batch))
            aggregated_batch_list.append(batch)
        aggregated_batch = Batch.concat(aggregated_batch_list, copy=False)
        if not len(aggregated_batch):
            return
        merge_batch_list = [aggregated_batch]
        for col in self._columns:
            col_name_list = self._extract_column_name(col)
            for col_name in col_name_list:
                if col_name not in aggregated_batch.columns:
                    batch = col.evaluate(aggregated_batch)
                    merge_batch_list.append(batch)
        if len(merge_batch_list) > 1:
            aggregated_batch = Batch.merge_column_wise(merge_batch_list)
        try:
            aggregated_batch.sort_orderby(by=self.extract_column_names(), sort_type=self.extract_sort_types())
        except KeyError:
            pass
        index = 0
        for i in self.batch_sizes:
            batch = aggregated_batch[index:index + i]
            batch.reset_index()
            index += i
            yield batch

@ray.remote(num_cpus=0)
def _ray_wait_and_alert(tasks: List[ray.ObjectRef], queue: Queue):
    try:
        ray.get(tasks)
        queue.put(StageCompleteSignal)
    except RayTaskError as e:
        queue.put(ExecutorError(e.cause))

class CreateJobExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: CreateJobStatement):
        super().__init__(db, node)

    def _parse_datetime_str(self, datetime_str: str) -> datetime:
        datetime_format = '%Y-%m-%d %H:%M:%S'
        date_format = '%Y-%m-%d'
        if re.match('\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}', datetime_str):
            try:
                return datetime.strptime(datetime_str, datetime_format)
            except ValueError:
                raise ExecutorError(f'{datetime_str} is not in the correct datetime format. expected format: {datetime_format}.')
        elif re.match('\\d{4}-\\d{2}-\\d{2}', datetime_str):
            try:
                return datetime.strptime(datetime_str, date_format)
            except ValueError:
                raise ExecutorError(f'{datetime_str} is not in the correct date format. expected format: {date_format}.')
        else:
            raise ValueError(f'{datetime_str} does not match the expected date or datetime format')

    def _get_repeat_time_interval_seconds(self, repeat_interval: int, repeat_period: str) -> int:
        unit_to_seconds = {'seconds': 1, 'minute': 60, 'minutes': 60, 'min': 60, 'hour': 3600, 'hours': 3600, 'day': 86400, 'days': 86400, 'week': 604800, 'weeks': 604800, 'month': 2592000, 'months': 2592000}
        assert repeat_period is None or repeat_period in unit_to_seconds, 'repeat period should be one of these values: seconds | minute | minutes | min | hour | hours | day | days | week | weeks | month | months'
        repeat_interval = 1 if repeat_interval is None else repeat_interval
        return repeat_interval * unit_to_seconds.get(repeat_period, 0)

    def exec(self, *args, **kwargs):
        job_catalog_entry = self.catalog().get_job_catalog_entry(self.node.job_name)
        if job_catalog_entry is not None:
            if self.node.if_not_exists:
                msg = f'A job with name {self.node.job_name} already exists, nothing added.'
                yield Batch(pd.DataFrame([msg]))
                return
            else:
                raise ExecutorError(f'A job with name {self.node.job_name} already exists.')
        logger.debug(f'Creating job {self.node}')
        job_name = self.node.job_name
        queries = []
        parser = Parser()
        for q in self.node.queries:
            try:
                curr_query = str(q)
                parser.parse(curr_query)
                queries.append(curr_query)
            except Exception:
                error_msg = f'Failed to parse the job query: {curr_query}'
                logger.exception(error_msg)
                raise ExecutorError(error_msg)
        start_time = self._parse_datetime_str(self.node.start_time) if self.node.start_time is not None else datetime.datetime.now()
        end_time = self._parse_datetime_str(self.node.end_time) if self.node.end_time is not None else None
        repeat_interval = self._get_repeat_time_interval_seconds(self.node.repeat_interval, self.node.repeat_period)
        active = True
        next_schedule_run = start_time
        self.catalog().insert_job_catalog_entry(job_name, queries, start_time, end_time, repeat_interval, active, next_schedule_run)
        yield Batch(pd.DataFrame([f'The job {self.node.job_name} has been successfully created.']))

class StorageExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: StoragePlan):
        super().__init__(db, node)

    def exec(self, *args, **kwargs) -> Iterator[Batch]:
        try:
            storage_engine = StorageEngine.factory(self.db, self.node.table)
            if self.node.table.table_type == TableType.VIDEO_DATA:
                return storage_engine.read(self.node.table, self.node.batch_mem_size, predicate=self.node.predicate, sampling_rate=self.node.sampling_rate, sampling_type=self.node.sampling_type, read_audio=self.node.table_ref.get_audio, read_video=self.node.table_ref.get_video)
            elif self.node.table.table_type == TableType.IMAGE_DATA:
                return storage_engine.read(self.node.table)
            elif self.node.table.table_type == TableType.DOCUMENT_DATA:
                return storage_engine.read(self.node.table, self.node.chunk_params)
            elif self.node.table.table_type == TableType.STRUCTURED_DATA:
                return storage_engine.read(self.node.table, self.node.batch_mem_size)
            elif self.node.table.table_type == TableType.NATIVE_DATA:
                return storage_engine.read(self.node.table)
            elif self.node.table.table_type == TableType.PDF_DATA:
                return storage_engine.read(self.node.table)
            else:
                raise ExecutorError(f'Unsupported TableType {self.node.table.table_type} encountered')
        except Exception as e:
            logger.error(e)
            raise ExecutorError(e)

class LoadDataExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: LoadDataPlan):
        super().__init__(db, node)

    def exec(self, *args, **kwargs):
        """
        Use TYPE to determine the type of data to load.
        """
        if self.node.file_options['file_format'] is None:
            err_msg = 'Invalid file format, please use supported file formats: CSV | VIDEO | IMAGE | DOCUMENT | PDF'
            raise ExecutorError(err_msg)
        if self.node.file_options['file_format'] in [FileFormatType.VIDEO, FileFormatType.IMAGE, FileFormatType.DOCUMENT, FileFormatType.PDF]:
            executor = LoadMultimediaExecutor(self.db, self.node)
        elif self.node.file_options['file_format'] == FileFormatType.CSV:
            executor = LoadCSVExecutor(self.db, self.node)
        for batch in executor.exec():
            yield batch

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

def validate_image(image_path: Path) -> bool:
    try:
        try_to_import_cv2()
        import cv2
        data = cv2.imread(str(image_path))
        return data is not None
    except Exception as e:
        logger.warning(f'Unexpected Exception {e} occurred while reading image file {image_path}')
        return False

def validate_video(video_path: Path) -> bool:
    try:
        try_to_import_cv2()
        import cv2
        vid = cv2.VideoCapture(str(video_path))
        if not vid.isOpened():
            return False
        return True
    except Exception as e:
        logger.warning(f'Unexpected Exception {e} occurred while reading video file {video_path}')

def validate_document(doc_path: Path) -> bool:
    return doc_path.suffix in SUPPORTED_TYPES

def validate_pdf(doc_path: Path) -> bool:
    return doc_path.suffix == '.pdf'

class CreateIndexExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: CreateIndexPlan):
        super().__init__(db, node)
        self.name = self.node.name
        self.if_not_exists = self.node.if_not_exists
        self.table_ref = self.node.table_ref
        self.col_list = self.node.col_list
        self.vector_store_type = self.node.vector_store_type
        self.project_expr_list = self.node.project_expr_list
        self.index_def = self.node.index_def

    def exec(self, *args, **kwargs):
        if self.vector_store_type == VectorStoreType.PGVECTOR:
            self._create_native_index()
        else:
            self._create_evadb_index()
        yield Batch(pd.DataFrame([f'Index {self.name} successfully added to the database.']))

    def _create_native_index(self):
        table = self.table_ref.table
        db_catalog_entry = self.catalog().get_database_catalog_entry(table.database_name)
        with get_database_handler(db_catalog_entry.engine, **db_catalog_entry.params) as handler:
            resp = handler.execute_native_query(f'CREATE INDEX {self.name} ON {table.table_name}\n                    USING hnsw ({self.col_list[0].name} vector_l2_ops)')
            if resp.error is not None:
                raise ExecutorError(f'Native engine create index encounters error: {resp.error}')

    def _get_evadb_index_save_path(self) -> Path:
        index_dir = Path(self.db.catalog().get_configuration_catalog_value('index_dir'))
        if not index_dir.exists():
            index_dir.mkdir(parents=True, exist_ok=True)
        return str(index_dir / Path('{}_{}.index'.format(self.vector_store_type, self.name)))

    def _create_evadb_index(self):
        function_expression, function_expression_signature = (None, None)
        for project_expr in self.project_expr_list:
            if isinstance(project_expr, FunctionExpression):
                function_expression = project_expr
                function_expression_signature = project_expr.signature()
        feat_tb_catalog_entry = self.table_ref.table.table_obj
        feat_col_name = self.col_list[0].name
        feat_col_catalog_entry = [col for col in feat_tb_catalog_entry.columns if col.name == feat_col_name][0]
        if function_expression is not None:
            feat_col_name = function_expression.output_objs[0].name
        index_catalog_entry = self.catalog().get_index_catalog_entry_by_name(self.name)
        index_path = self._get_evadb_index_save_path()
        if index_catalog_entry is not None:
            msg = f'Index {self.name} already exists.'
            if self.if_not_exists:
                if index_catalog_entry.feat_column == feat_col_catalog_entry and index_catalog_entry.function_signature == function_expression_signature and (index_catalog_entry.type == self.node.vector_store_type):
                    logger.warn(msg + ' It will be updated on existing table.')
                    index = VectorStoreFactory.init_vector_store(self.vector_store_type, self.name, **handle_vector_store_params(self.vector_store_type, index_path, self.catalog))
                else:
                    logger.warn(msg)
                    return
            else:
                logger.error(msg)
                raise ExecutorError(msg)
        else:
            index = None
        try:
            for input_batch in self.children[0].exec():
                input_batch.drop_column_alias()
                feat = input_batch.column_as_numpy_array(feat_col_name)
                row_num = input_batch.column_as_numpy_array(ROW_NUM_COLUMN)
                for i in range(len(input_batch)):
                    row_feat = feat[i].reshape(1, -1)
                    if index is None:
                        input_dim = row_feat.shape[1]
                        index = VectorStoreFactory.init_vector_store(self.vector_store_type, self.name, **handle_vector_store_params(self.vector_store_type, index_path, self.catalog))
                        index.create(input_dim)
                    index.add([FeaturePayload(row_num[i], row_feat)])
            index.persist()
            if index_catalog_entry is None:
                self.catalog().insert_index_catalog_entry(self.name, index_path, self.vector_store_type, feat_col_catalog_entry, function_expression_signature, self.index_def)
        except Exception as e:
            if index:
                index.delete()
            raise ExecutorError(str(e))

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

def _get_database_handler(engine: str, **kwargs):
    """
    Return the database handler. User should modify this function for
    their new integrated handlers.
    """
    try:
        mod = dynamic_import(engine)
    except ImportError:
        req_file = os.path.join(os.path.dirname(__file__), engine, 'requirements.txt')
        if os.path.isfile(req_file):
            with open(req_file) as f:
                raise ImportError(f'Please install the following packages {f.read()}')
    if engine == 'postgres':
        return mod.PostgresHandler(engine, **kwargs)
    elif engine == 'sqlite':
        return mod.SQLiteHandler(engine, **kwargs)
    elif engine == 'mysql':
        return mod.MysqlHandler(engine, **kwargs)
    elif engine == 'mariadb':
        return mod.MariaDbHandler(engine, **kwargs)
    elif engine == 'clickhouse':
        return mod.ClickHouseHandler(engine, **kwargs)
    elif engine == 'snowflake':
        return mod.SnowFlakeDbHandler(engine, **kwargs)
    elif engine == 'github':
        return mod.GithubHandler(engine, **kwargs)
    elif engine == 'hackernews':
        return mod.HackernewsSearchHandler(engine, **kwargs)
    elif engine == 'slack':
        return mod.SlackHandler(engine, **kwargs)
    else:
        raise NotImplementedError(f'Engine {engine} is not supported')

class PineconeVectorStore(VectorStore):

    def __init__(self, index_name: str, **kwargs) -> None:
        try_to_import_pinecone_client()
        global _pinecone_init_done
        self._index_name = index_name.strip().lower()
        self._api_key = kwargs.get('PINECONE_API_KEY')
        if not self._api_key:
            self._api_key = os.environ.get('PINECONE_API_KEY')
        assert self._api_key, 'Please set your `PINECONE_API_KEY` using set command or environment variable (PINECONE_KEY). It can be found at Pinecone Dashboard > API Keys > Value'
        self._environment = kwargs.get('PINECONE_ENV')
        if not self._environment:
            self._environment = os.environ.get('PINECONE_ENV')
        assert self._environment, 'Please set your `PINECONE_ENV` or environment variable (PINECONE_ENV). It can be found Pinecone Dashboard > API Keys > Environment.'
        if not _pinecone_init_done:
            import pinecone
            pinecone.init(api_key=self._api_key, environment=self._environment)
            _pinecone_init_done = True
        self._client = None

    def create(self, vector_dim: int):
        import pinecone
        pinecone.create_index(self._index_name, dimension=vector_dim, metric='cosine')
        logger.warning(f'Created index {self._index_name}. Please note that Pinecone is eventually consistent, hence any additions to the Vector Index may not get immediately reflected in queries.')
        self._client = pinecone.Index(self._index_name)

    def add(self, payload: List[FeaturePayload]):
        self._client.upsert(vectors=[{'id': str(row.id), 'values': row.embedding.reshape(-1).tolist()} for row in payload])

    def delete(self) -> None:
        import pinecone
        pinecone.delete_index(self._index_name)

    def query(self, query: VectorIndexQuery) -> VectorIndexQueryResult:
        import pinecone
        if not self._client:
            self._client = pinecone.Index(self._index_name)
        response = self._client.query(top_k=query.top_k, vector=query.embedding.reshape(-1).tolist())
        distances, ids = ([], [])
        for row in response['matches']:
            distances.append(row['score'])
            ids.append(int(row['id']))
        return VectorIndexQueryResult(distances, ids)

class EvaDBConnection:

    def __init__(self, evadb: EvaDBDatabase, reader, writer):
        self._reader = reader
        self._writer = writer
        self._cursor = None
        self._result: Batch = None
        self._evadb = evadb
        self._jobs_process = None

    def cursor(self):
        """Retrieves a cursor associated with the connection.

        Returns:
            EvaDBCursor: The cursor object used to execute queries.


        Examples:
            >>> import evadb
            >>> connection = evadb.connection()
            >>> cursor = connection.cursor()

        The cursor can be used to execute queries.

            >>> cursor.query('SELECT * FROM sample_table;').df()
               col1  col2
            0     1     2
            1     3     4
            2     5     6

        """
        if self._cursor is None:
            self._cursor = EvaDBCursor(self)
        return self._cursor

    def start_jobs(self):
        if self._jobs_process and self._jobs_process.is_alive():
            logger.debug('The job scheduler is already running')
            return
        job_scheduler = JobScheduler(self._evadb)
        self._jobs_process = multiprocessing.Process(target=job_scheduler.execute)
        self._jobs_process.daemon = True
        self._jobs_process.start()
        logger.debug('Job scheduler process started')

    def stop_jobs(self):
        if self._jobs_process is not None and self._jobs_process.is_alive():
            self._jobs_process.terminate()
            self._jobs_process.join()
            logger.debug('Job scheduler process stopped')

class EvaDBCursor(object):

    def __init__(self, connection):
        self._connection = connection
        self._evadb = connection._evadb
        self._pending_query = False
        self._result = None

    async def execute_async(self, query: str):
        """
        Send query to the EvaDB server.
        """
        if self._pending_query:
            raise SystemError('EvaDB does not support concurrent queries.                     Call fetch_all() to complete the pending query')
        query = self._multiline_query_transformation(query)
        self._connection._writer.write((query + '\n').encode())
        await self._connection._writer.drain()
        self._pending_query = True
        return self

    async def fetch_one_async(self) -> Response:
        """
        fetch_one returns one batch instead of one row for now.
        """
        response = Response()
        prefix = await self._connection._reader.readline()
        if prefix != b'':
            message_length = int(prefix)
            message = await self._connection._reader.readexactly(message_length)
            response = Response.deserialize(message)
        self._pending_query = False
        return response

    async def fetch_all_async(self) -> Response:
        """
        fetch_all is the same as fetch_one for now.
        """
        return await self.fetch_one_async()

    def _multiline_query_transformation(self, query: str) -> str:
        query = query.replace('\n', ' ')
        query = query.lstrip()
        query = query.rstrip(' ;')
        query += ';'
        logger.debug('Query: ' + query)
        return query

    def stop_query(self):
        self._pending_query = False

    def __getattr__(self, name):
        """
        Auto generate sync function calls from async
        Sync function calls should not be used in an async environment.
        """
        function_name_list = ['table', 'load', 'execute', 'query', 'create_function', 'create_table', 'create_vector_index', 'drop_table', 'drop_function', 'drop_index', 'df', 'show', 'insertexplain', 'rename', 'fetch_one']
        if name not in function_name_list:
            nearest_function = find_nearest_word(name, function_name_list)
            raise AttributeError(f"EvaDBCursor does not contain a function named: '{name}'. Did you mean to run: '{nearest_function}()'?")
        try:
            func = object.__getattribute__(self, '%s_async' % name)
        except Exception as e:
            raise e

        def func_sync(*args, **kwargs):
            loop = asyncio.get_event_loop()
            res = loop.run_until_complete(func(*args, **kwargs))
            return res
        return func_sync

    def table(self, table_name: str, chunk_size: int=None, chunk_overlap: int=None) -> EvaDBQuery:
        """
        Retrieves data from a table in the database.

        Args:
            table_name (str): The name of the table to retrieve data from.
            chunk_size (int, optional): The size of the chunk to break the document into. Only valid for DOCUMENT tables.
                If not provided, the default value is 4000.
            chunk_overlap (int, optional): The overlap between consecutive chunks. Only valid for DOCUMENT tables.
                If not provided, the default value is 200.

        Returns:
            EvaDBQuery: An EvaDBQuery object representing the table query.

        Examples:
            >>> relation = cursor.table("sample_table")
            >>> relation = cursor.select('*')
            >>> relation.df()
               col1  col2
            0     1     2
            1     3     4
            2     5     6

            Read a document table using chunk_size 100 and chunk_overlap 10.

            >>> relation = cursor.table("doc_table", chunk_size=100, chunk_overlap=10)
        """
        table = parse_table_clause(table_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        select_stmt = SelectStatement(target_list=[TupleValueExpression(name='*')], from_table=table)
        try_binding(self._evadb.catalog, select_stmt)
        return EvaDBQuery(self._evadb, select_stmt, alias=Alias(table_name.lower()))

    def df(self) -> pandas.DataFrame:
        """
        Returns the result as a pandas DataFrame.

        Returns:
            pandas.DataFrame: The result as a DataFrame.

        Raises:
            Exception: If no valid result is available with the current connection.

        Examples:
            >>> result = cursor.query("CREATE TABLE IF NOT EXISTS youtube_video_text AS SELECT SpeechRecognizer(audio) FROM youtube_video;").df()
            >>> result
            Empty DataFrame
            >>> relation = cursor.table("youtube_video_text").select('*').df()
                speechrecognizer.response
            0	"Sample Text from speech recognizer"
        """
        if not self._result:
            raise Exception('No valid result with the current cursor')
        return self._result.frames

    def create_vector_index(self, index_name: str, table_name: str, expr: str, using: str) -> 'EvaDBCursor':
        """
        Creates a vector index using the provided expr on the table.
        This feature directly works on IMAGE tables.
        For VIDEO tables, the feature should be extracted first and stored in an intermediate table, before creating the index.

        Args:
            index_name (str): Name of the index.
            table_name (str): Name of the table.
            expr (str): Expression used to build the vector index.

            using (str): Method used for indexing, can be `FAISS` or `QDRANT` or `PINECONE` or `CHROMADB` or `WEAVIATE` or `MILVUS`.

        Returns:
            EvaDBCursor: The EvaDBCursor object.

        Examples:
            Create a Vector Index using QDRANT

            >>> cursor.create_vector_index(
                    "faiss_index",
                    table_name="meme_images",
                    expr="SiftFeatureExtractor(data)",
                    using="QDRANT"
                ).df()
                        0
                0	Index faiss_index successfully added to the database
            >>> relation = cursor.table("PDFs")
            >>> relation.order("Similarity(ImageFeatureExtractor(Open('/images/my_meme')), ImageFeatureExtractor(data) ) DESC")
            >>> relation.df()


        """
        stmt = parse_create_vector_index(index_name, table_name, expr, using)
        self._result = execute_statement(self._evadb, stmt)
        return self

    def load(self, file_regex: str, table_name: str, format: str, **kwargs) -> EvaDBQuery:
        """
        Loads data from files into a table.

        Args:
            file_regex (str): Regular expression specifying the files to load.
            table_name (str): Name of the table.
            format (str): File format of the data.
            **kwargs: Additional keyword arguments for configuring the load operation.

        Returns:
            EvaDBQuery: The EvaDBQuery object representing the load query.

        Examples:
            Load the online_video.mp4 file into table named 'youtube_video'.

            >>> cursor.load(file_regex="online_video.mp4", table_name="youtube_video", format="video").df()
                    0
            0	Number of loaded VIDEO: 1

        """
        stmt = parse_load(table_name, file_regex, format, **kwargs)
        return EvaDBQuery(self._evadb, stmt)

    def drop_table(self, table_name: str, if_exists: bool=True) -> 'EvaDBQuery':
        """
        Drop a table in the database.

        Args:
            table_name (str): Name of the table to be dropped.
            if_exists (bool): If True, do not raise an error if the Table does not already exist. If False, raise an error.

        Returns:
            EvaDBQuery: The EvaDBQuery object representing the DROP TABLE.

        Examples:
            Drop table 'sample_table'

            >>> cursor.drop_table("sample_table", if_exists = True).df()
                0
            0	Table Successfully dropped: sample_table
        """
        stmt = parse_drop_table(table_name, if_exists)
        return EvaDBQuery(self._evadb, stmt)

    def drop_function(self, function_name: str, if_exists: bool=True) -> 'EvaDBQuery':
        """
        Drop a function in the database.

        Args:
            function_name (str): Name of the function to be dropped.
            if_exists (bool): If True, do not raise an error if the function does not already exist. If False, raise an error.

        Returns:
            EvaDBQuery: The EvaDBQuery object representing the DROP FUNCTION.

        Examples:
            Drop FUNCTION 'ObjectDetector'

            >>> cursor.drop_function("ObjectDetector", if_exists = True)
                0
            0	Function Successfully dropped: ObjectDetector
        """
        stmt = parse_drop_function(function_name, if_exists)
        return EvaDBQuery(self._evadb, stmt)

    def drop_index(self, index_name: str, if_exists: bool=True) -> 'EvaDBQuery':
        """
        Drop an index in the database.

        Args:
            index_name (str): Name of the index to be dropped.
            if_exists (bool): If True, do not raise an error if the index does not already exist. If False, raise an error.

        Returns:
            EvaDBQuery: The EvaDBQuery object representing the DROP INDEX.

        Examples:
            Drop the index with name 'faiss_index'

            >>> cursor.drop_index("faiss_index", if_exists = True)
        """
        stmt = parse_drop_index(index_name, if_exists)
        return EvaDBQuery(self._evadb, stmt)

    def create_function(self, function_name: str, if_not_exists: bool=True, impl_path: str=None, type: str=None, **kwargs) -> 'EvaDBQuery':
        """
        Create a function in the database.

        Args:
            function_name (str): Name of the function to be created.
            if_not_exists (bool): If True, do not raise an error if the function already exist. If False, raise an error.
            impl_path (str): Path string to function's implementation.
            type (str): Type of the function (e.g. HuggingFace).
            **kwargs: Additional keyword arguments for configuring the create function operation.

        Returns:
            EvaDBQuery: The EvaDBQuery object representing the function created.

        Examples:
            >>> cursor.create_function("MnistImageClassifier", if_exists = True, 'mnist_image_classifier.py')
                0
            0	Function Successfully created: MnistImageClassifier
        """
        stmt = parse_create_function(function_name, if_not_exists, impl_path, type, **kwargs)
        return EvaDBQuery(self._evadb, stmt)

    def create_table(self, table_name: str, if_not_exists: bool=True, columns: str=None, **kwargs) -> 'EvaDBQuery':
        '''
        Create a function in the database.

        Args:
            function_name (str): Name of the function to be created.
            if_not_exists (bool): If True, do not raise an error if the function already exist. If False, raise an error.
            impl_path (str): Path string to function's implementation.
            type (str): Type of the function (e.g. HuggingFace).
            **kwargs: Additional keyword arguments for configuring the create function operation.

        Returns:
            EvaDBQuery: The EvaDBQuery object representing the function created.

        Examples:
            >>> cursor.create_table("MyCSV", if_exists = True, columns="""
                    id INTEGER UNIQUE,
                    frame_id INTEGER,
                    video_id INTEGER,
                    dataset_name TEXT(30),
                    label TEXT(30),
                    bbox NDARRAY FLOAT32(4),
                    object_id INTEGER"""
                    )
                0
            0	Table Successfully created: MyCSV
        '''
        stmt = parse_create_table(table_name, if_not_exists, columns, **kwargs)
        return EvaDBQuery(self._evadb, stmt)

    def query(self, sql_query: str) -> EvaDBQuery:
        """
        Executes a SQL query.

        Args:
            sql_query (str): The SQL query to be executed

        Returns:
            EvaDBQuery: The EvaDBQuery object.

        Examples:
            >>> cursor.query("DROP FUNCTION IF EXISTS SentenceFeatureExtractor;")
            >>> cursor.query('SELECT * FROM sample_table;').df()
               col1  col2
            0     1     2
            1     3     4
            2     5     6
        """
        stmt = parse_query(sql_query)
        return EvaDBQuery(self._evadb, stmt)

    def show(self, object_type: str, **kwargs) -> EvaDBQuery:
        """
        Shows all entries of the current object_type.

        Args:
            show_type (str): The type of SHOW query to be executed
            **kwargs: Additional keyword arguments for configuring the SHOW operation.

        Returns:
            EvaDBQuery: The EvaDBQuery object.

        Examples:
            >>> cursor.show("tables").df()
                name
            0	SampleTable1
            1	SampleTable2
            2	SampleTable3
        """
        stmt = parse_show(object_type, **kwargs)
        return EvaDBQuery(self._evadb, stmt)

    def explain(self, sql_query: str) -> EvaDBQuery:
        """
        Executes an EXPLAIN query.

        Args:
            sql_query (str): The SQL query to be explained

        Returns:
            EvaDBQuery: The EvaDBQuery object.

        Examples:
            >>> proposed_plan = cursor.explain("SELECT * FROM sample_table;").df()
            >>> for step in proposed_plan[0]:
            >>>   pprint(step)
             |__ ProjectPlan
                |__ SeqScanPlan
                    |__ StoragePlan
        """
        stmt = parse_explain(sql_query)
        return EvaDBQuery(self._evadb, stmt)

    def insert(self, table_name, columns, values, **kwargs) -> EvaDBQuery:
        """
        Executes an INSERT query.

        Args:
            table_name (str): The name of the table to insert into
            columns (list): The list of columns to insert into
            values (list): The list of values to insert
            **kwargs: Additional keyword arguments for configuring the INSERT operation.

        Returns:
            EvaDBQuery: The EvaDBQuery object.

        Examples:
            >>> cursor.insert("sample_table", ["id", "name"], [1, "Alice"])
        """
        stmt = parse_insert(table_name, columns, values, **kwargs)
        return EvaDBQuery(self._evadb, stmt)

    def rename(self, table_name, new_table_name, **kwargs) -> EvaDBQuery:
        """
        Executes a RENAME query.

        Args:
            table_name (str): The name of the table to rename
            new_table_name (str): The new name of the table
            **kwargs: Additional keyword arguments for configuring the RENAME operation.

        Returns:
            EvaDBQuery: The EvaDBQuery object.

        Examples:
            >>> cursor.show("tables").df()
                name
            0	SampleVideoTable
            1	SampleTable
            2	MyCSV
            3	videotable
            >>> cursor.rename("videotable", "sample_table").df()
            _
            >>> cursor.show("tables").df()
                        name
            0	SampleVideoTable
            1	SampleTable
            2	MyCSV
            3	sample_table

        """
        stmt = parse_rename(table_name, new_table_name, **kwargs)
        return EvaDBQuery(self._evadb, stmt)

    def close(self):
        """
        Closes the connection.

        Args: None

        Returns:  None

        Examples:
            >>> cursor.close()
        """
        self._evadb.catalog().close()
        ray_enabled = self._evadb.config.get_value('experimental', 'ray')
        if is_ray_enabled_and_installed(ray_enabled):
            import ray
            ray.shutdown()

@dataclass(frozen=True)
class Response:
    """
    Data model for EvaDB server response
    """
    status: ResponseStatus = ResponseStatus.FAIL
    batch: Batch = None
    error: Optional[str] = None
    query_time: Optional[float] = None

    def serialize(self):
        return PickleSerializer.serialize(self)

    @classmethod
    def deserialize(cls, data):
        obj = PickleSerializer.deserialize(data)
        return obj

    def as_df(self):
        if self.error is not None:
            raise ExecutorError(self.error)
        if self.batch is None:
            raise ExecutorError('Empty batch')
        return self.batch.frames

    def __str__(self):
        if self.query_time is not None:
            return '@status: %s\n@batch: \n %s\n@query_time: %s' % (self.status, self.batch, self.query_time)
        else:
            return '@status: %s\n@batch: \n %s\n@error: %s' % (self.status, self.batch, self.error)

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

class SQLStorageEngine(AbstractStorageEngine):

    def __init__(self, db: EvaDBDatabase):
        """
        Grab the existing sql session
        """
        super().__init__(db)
        self._sql_session = db.catalog().sql_config.session
        self._sql_engine = db.catalog().sql_config.engine
        self._serializer = PickleSerializer

    def _dict_to_sql_row(self, dict_row: dict, columns: List[ColumnCatalogEntry]):
        for col in columns:
            if col.type == ColumnType.NDARRAY:
                dict_row[col.name] = self._serializer.serialize(dict_row[col.name])
            elif isinstance(dict_row[col.name], (np.generic,)):
                dict_row[col.name] = dict_row[col.name].tolist()
        return dict_row

    def _deserialize_sql_row(self, sql_row: dict, columns: List[ColumnCatalogEntry]):
        dict_row = {}
        for idx, col in enumerate(columns):
            if col.type == ColumnType.NDARRAY:
                dict_row[col.name] = self._serializer.deserialize(sql_row[col.name])
            else:
                dict_row[col.name] = sql_row[col.name]
        dict_row[ROW_NUM_COLUMN] = dict_row[IDENTIFIER_COLUMN]
        return dict_row

    def _try_loading_table_via_reflection(self, table_name: str):
        metadata_obj = BaseModel.metadata
        if table_name in metadata_obj.tables:
            return metadata_obj.tables[table_name]
        insp = inspect(self._sql_engine)
        if insp.has_table(table_name):
            table = Table(table_name, metadata_obj)
            insp.reflect_table(table, None)
            return table
        else:
            err_msg = f'No table found with name {table_name}'
            logger.exception(err_msg)
            raise Exception(err_msg)

    def create(self, table: TableCatalogEntry, **kwargs):
        """
        Create an empty table in sql.
        It dynamically constructs schema in sqlaclchemy
        to create the table
        """
        attr_dict = {'__tablename__': table.name}
        table_columns = [col for col in table.columns if col.name != IDENTIFIER_COLUMN and col.name != ROW_NUM_COLUMN]
        sqlalchemy_schema = SchemaUtils.xform_to_sqlalchemy_schema(table_columns)
        attr_dict.update(sqlalchemy_schema)
        insp = inspect(self._sql_engine)
        if insp.has_table(table.name):
            logger.warning('Table {table.name} already exists')
            return BaseModel.metadata.tables[table.name]
        new_table = type(f'__placeholder_class_name__{table.name}', (BaseModel,), attr_dict)()
        table = BaseModel.metadata.tables[table.name]
        if not insp.has_table(table.name):
            BaseModel.metadata.tables[table.name].create(self._sql_engine)
        self._sql_session.commit()
        return new_table

    def drop(self, table: TableCatalogEntry):
        try:
            table_to_remove = self._try_loading_table_via_reflection(table.name)
            insp = inspect(self._sql_engine)
            if insp.has_table(table_to_remove.name):
                table_to_remove.drop(self._sql_engine)
                BaseModel.metadata.remove(table_to_remove)
            self._sql_session.commit()
        except Exception as e:
            err_msg = f'Failed to drop the table {table.name} with Exception {str(e)}'
            logger.exception(err_msg)
            raise Exception(err_msg)

    def write(self, table: TableCatalogEntry, rows: Batch):
        """
        Write rows into the sql table.

        Arguments:
            table: table metadata object to write into
            rows : batch to be persisted in the storage.
        """
        try:
            table_to_update = self._try_loading_table_via_reflection(table.name)
            columns = rows.frames.keys()
            data = []
            table_columns = [col for col in table.columns if col.name != IDENTIFIER_COLUMN and col.name != ROW_NUM_COLUMN]
            for record in rows.frames.values:
                row_data = {col: record[idx] for idx, col in enumerate(columns) if col != ROW_NUM_COLUMN}
                data.append(self._dict_to_sql_row(row_data, table_columns))
            self._sql_session.execute(table_to_update.insert(), data)
            self._sql_session.commit()
        except Exception as e:
            err_msg = f'Failed to update the table {table.name} with exception {str(e)}'
            logger.exception(err_msg)
            raise Exception(err_msg)

    def read(self, table: TableCatalogEntry, batch_mem_size: int=30000000) -> Iterator[Batch]:
        """
        Reads the table and return a batch iterator for the
        tuples.

        Argument:
            table: table metadata object of the table to read
            batch_mem_size (int): memory size of the batch read from storage
        Return:
            Iterator of Batch read.
        """
        try:
            table_to_read = self._try_loading_table_via_reflection(table.name)
            result = self._sql_session.execute(table_to_read.select()).fetchall()
            result_iter = (self._deserialize_sql_row(row._asdict(), table.columns) for row in result)
            for df in rebatch(result_iter, batch_mem_size):
                yield Batch(pd.DataFrame(df))
        except Exception as e:
            err_msg = f'Failed to read the table {table.name} with exception {str(e)}'
            logger.exception(err_msg)
            raise Exception(err_msg)

    def delete(self, table: TableCatalogEntry, sqlalchemy_filter_clause: 'ColumnElement[bool]'):
        """Delete tuples from the table where rows satisfy the where_clause.
        The current implementation only handles equality predicates.

        Argument:
            table: table metadata object of the table
            where_clause: clause used to find the tuples to remove.
        """
        try:
            table_to_delete_from = self._try_loading_table_via_reflection(table.name)
            d = table_to_delete_from.delete().where(sqlalchemy_filter_clause)
            self._sql_session.execute(d)
            self._sql_session.commit()
        except Exception as e:
            err_msg = f'Failed to delete from the table {table.name} with exception {str(e)}'
            logger.exception(err_msg)
            raise Exception(err_msg)

    def rename(self, old_table: TableCatalogEntry, new_name: TableInfo):
        raise Exception('Rename not supported for structured data table')

class NativeStorageEngine(AbstractStorageEngine):

    def __init__(self, db: EvaDBDatabase):
        super().__init__(db)

    def _get_database_catalog_entry(self, database_name):
        db_catalog_entry = self.db.catalog().get_database_catalog_entry(database_name)
        if db_catalog_entry is None:
            raise Exception(f'Could not find database with name {database_name}. Please register the database using the `CREATE DATABASE` command.')
        return db_catalog_entry

    def create(self, table: TableCatalogEntry):
        try:
            db_catalog_entry = self._get_database_catalog_entry(table.database_name)
            uri = None
            with get_database_handler(db_catalog_entry.engine, **db_catalog_entry.params) as handler:
                uri = handler.get_sqlalchmey_uri()
            sqlalchemy_schema = SchemaUtils.xform_to_sqlalchemy_schema(table.columns)
            create_table(uri, table.name, sqlalchemy_schema)
        except Exception as e:
            err_msg = f'Failed to create the table {table.name} in data source {table.database_name} with exception {str(e)}'
            logger.exception(err_msg)
            raise Exception(err_msg)

    def write(self, table: TableCatalogEntry, rows: Batch):
        try:
            db_catalog_entry = self._get_database_catalog_entry(table.database_name)
            with get_database_handler(db_catalog_entry.engine, **db_catalog_entry.params) as handler:
                uri = handler.get_sqlalchmey_uri()
            engine = create_engine(uri)
            metadata = MetaData()
            table_to_update = Table(table.name, metadata, autoload_with=engine)
            columns = rows.frames.keys()
            data = []
            for record in rows.frames.values:
                row_data = {col: record[idx] for idx, col in enumerate(columns)}
                data.append(_dict_to_sql_row(row_data, table.columns))
            Session = sessionmaker(bind=engine)
            session = Session()
            session.execute(table_to_update.insert(), data)
            session.commit()
            session.close()
        except Exception as e:
            err_msg = f'Failed to write to the table {table.name} in data source {table.database_name} with exception {str(e)}'
            logger.exception(err_msg)
            raise Exception(err_msg)

    def read(self, table: TableCatalogEntry, batch_mem_size: int=30000000) -> Iterator[Batch]:
        try:
            db_catalog_entry = self._get_database_catalog_entry(table.database_name)
            with get_database_handler(db_catalog_entry.engine, **db_catalog_entry.params) as handler:
                handler_response = handler.select(table.name)
                result = []
                if handler_response.data_generator:
                    result = handler_response.data_generator
                elif handler_response.data:
                    result = handler_response.data
                if handler.is_sqlalchmey_compatible():
                    cols = result[0]._fields
                    index_dict = {element.lower(): index for index, element in enumerate(cols)}
                    try:
                        ordered_columns = sorted(table.columns, key=lambda x: index_dict[x.name.lower()])
                    except KeyError as e:
                        raise Exception(f'Column mismatch with error {e}')
                    result = (_deserialize_sql_row(row, ordered_columns) for row in result)
                for df in rebatch(result, batch_mem_size):
                    yield Batch(pd.DataFrame(df))
        except Exception as e:
            err_msg = f'Failed to read the table {table.name} in data source {table.database_name} with exception {str(e)}'
            logger.exception(err_msg)
            raise Exception(err_msg)

    def drop(self, table: TableCatalogEntry):
        try:
            db_catalog_entry = self._get_database_catalog_entry(table.database_name)
            with get_database_handler(db_catalog_entry.engine, **db_catalog_entry.params) as handler:
                uri = handler.get_sqlalchmey_uri()
            engine = create_engine(uri)
            metadata = MetaData()
            Session = sessionmaker(bind=engine)
            session = Session()
            table_to_remove = Table(table.name, metadata, autoload_with=engine)
            table_to_remove.drop(engine)
            session.commit()
            session.close()
        except Exception as e:
            err_msg = f'Failed to drop the table {table.name} in data source {table.database_name} with exception {str(e)}'
            logger.error(err_msg)
            raise Exception(err_msg)

