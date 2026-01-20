# Cluster 3

def get_size(obj, seen=None):
    """Recursively finds size of objects
    https://goshippo.com/blog/measure-real-size-any-python-object/
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and (not isinstance(obj, (str, bytes, bytearray))):
        size += sum([get_size(i, seen) for i in obj])
    return size

def rebatch(it: Iterator, batch_mem_size: int=30000000) -> Iterator:
    """
    Utility function to rebatch the rows
    Args:
        it (Iterator): an iterator for rows, every row is a dictionary
        batch_mem_size (int): the maximum batch memory size
    Yields:
        data_batch (List): a list of rows, every row is a dictionary
    """
    data_batch = []
    row_size = None
    for row in it:
        data_batch.append(row)
        if row_size is None:
            row_size = get_size(data_batch)
        if len(data_batch) * row_size >= batch_mem_size:
            yield data_batch
            data_batch = []
    if data_batch:
        yield data_batch

class FunctionCacheCatalogService(BaseService):

    def __init__(self, db_session: Session):
        super().__init__(FunctionCacheCatalog, db_session)
        self._column_service: ColumnCatalogService = ColumnCatalogService(db_session)
        self._function_service: FunctionCatalogService = FunctionCatalogService(db_session)

    def insert_entry(self, entry: FunctionCacheCatalogEntry) -> FunctionCacheCatalogEntry:
        """Insert a new function cache entry into function cache catalog.
        Arguments:
            `name` (str): name of the cache table
            `function_id` (int): `row_id` of the function on which the cache is built
            `cache_path` (str): path of the cache table
            `args` (List[Any]): arguments of the function whose output is being cached
            `function_depends` (List[FunctionCatalogEntry]): dependent function  entries
            `col_depends` (List[ColumnCatalogEntry]): dependent column entries
        Returns:
            `FunctionCacheCatalogEntry`
        """
        try:
            cache_obj = self.model(name=entry.name, function_id=entry.function_id, cache_path=entry.cache_path, args=entry.args)
            cache_obj._function_depends = [self._function_service.get_entry_by_id(function_id, return_alchemy=True) for function_id in entry.function_depends]
            cache_obj._col_depends = [self._column_service.get_entry_by_id(col_id, return_alchemy=True) for col_id in entry.col_depends]
            cache_obj = cache_obj.save(self.session)
        except Exception as e:
            err_msg = f'Failed to insert entry into function cache catalog with exception {str(e)}'
            logger.exception(err_msg)
            raise CatalogError(err_msg)
        else:
            return cache_obj.as_dataclass()

    def get_entry_by_name(self, name: str) -> FunctionCacheCatalogEntry:
        try:
            entry = self.session.execute(select(self.model).filter(self.model._name == name)).scalar_one()
            return entry.as_dataclass()
        except NoResultFound:
            return None

    def delete_entry(self, cache: FunctionCacheCatalogEntry):
        """Delete cache table from the db
        Arguments:
            cache  (FunctionCacheCatalogEntry): cache to delete
        Returns:
            True if successfully removed else false
        """
        try:
            obj = self.session.execute(select(self.model).filter(self.model._row_id == cache.row_id)).scalar_one()
            obj.delete(self.session)
            return True
        except Exception as e:
            err_msg = f'Delete cache failed for {cache} with error {str(e)}.'
            logger.exception(err_msg)
            raise CatalogError(err_msg)

class IndexCatalogService(BaseService):

    def __init__(self, db_session: Session):
        super().__init__(IndexCatalog, db_session)

    def insert_entry(self, name: str, save_file_path: str, type: VectorStoreType, feat_column: ColumnCatalogEntry, function_signature: str, index_def: str) -> IndexCatalogEntry:
        index_entry = IndexCatalog(name, save_file_path, type, feat_column.row_id, function_signature, index_def)
        index_entry = index_entry.save(self.session)
        return index_entry.as_dataclass()

    def get_entry_by_name(self, name: str) -> IndexCatalogEntry:
        try:
            entry = self.query.filter(self.model._name == name).one()
            return entry.as_dataclass()
        except NoResultFound:
            return None

    def get_entry_by_id(self, id: int) -> IndexCatalogEntry:
        try:
            entry = self.query.filter(self.model._row_id == id).one()
            return entry.as_dataclass()
        except NoResultFound:
            return None

    def get_entry_by_column_and_function_signature(self, column: ColumnCatalogEntry, function_signature: str):
        try:
            entry = self.query.filter(self.model._feat_column_id == column.row_id, self.model._function_signature == function_signature).one()
            return entry.as_dataclass()
        except NoResultFound:
            return None

    def delete_entry_by_name(self, name: str):
        try:
            index_obj = self.query.filter(self.model._name == name).one()
            index_metadata = index_obj.as_dataclass()
            if os.path.exists(index_metadata.save_file_path):
                if os.path.isfile(index_metadata.save_file_path):
                    os.remove(index_metadata.save_file_path)
            index_obj.delete(self.session)
        except Exception:
            logger.exception('Delete index failed for name {}'.format(name))
            return False
        return True

class ConfigurationCatalogService(BaseService):

    def __init__(self, db_session: Session):
        super().__init__(ConfigurationCatalog, db_session)

    def insert_entry(self, key: str, value: any):
        try:
            config_catalog_obj = self.model(key=key, value=value)
            config_catalog_obj = config_catalog_obj.save(self.session)
        except Exception as e:
            logger.exception(f'Failed to insert entry into database catalog with exception {str(e)}')
            raise CatalogError(e)

    def get_entry_by_name(self, key: str) -> ConfigurationCatalogEntry:
        """
        Get the table catalog entry with given table name.
        Arguments:
            key  (str): key name
        Returns:
            Configuration Catalog Entry - catalog entry for given key name
        """
        entry = self.session.execute(select(self.model).filter(self.model._key == key)).scalar_one_or_none()
        if entry:
            return entry.as_dataclass()
        return entry

    def upsert_entry(self, key: str, value: any):
        try:
            entry = self.session.execute(select(self.model).filter(self.model._key == key)).scalar_one_or_none()
            if entry:
                entry.update(self.session, _value=value)
            else:
                self.insert_entry(key, value)
        except Exception as e:
            raise CatalogError(f'Error while upserting entry to ConfigurationCatalog: {str(e)}')

class JobCatalogService(BaseService):

    def __init__(self, db_session: Session):
        super().__init__(JobCatalog, db_session)

    def insert_entry(self, name: str, queries: list, start_time: datetime, end_time: datetime, repeat_interval: int, active: bool, next_schedule_run: datetime) -> JobCatalogEntry:
        try:
            job_catalog_obj = self.model(name=name, queries=json.dumps(queries), start_time=start_time, end_time=end_time, repeat_interval=repeat_interval, active=active, next_schedule_run=next_schedule_run)
            job_catalog_obj = job_catalog_obj.save(self.session)
        except Exception as e:
            logger.exception(f'Failed to insert entry into job catalog with exception {str(e)}')
            raise CatalogError(e)
        return job_catalog_obj.as_dataclass()

    def get_entry_by_name(self, job_name: str) -> JobCatalogEntry:
        """
        Get the job catalog entry with given job name.
        Arguments:
            job_name  (str): Job name
        Returns:
            JobCatalogEntry - catalog entry for given job name
        """
        entry = self.session.execute(select(self.model).filter(self.model._name == job_name)).scalar_one_or_none()
        if entry:
            return entry.as_dataclass()
        return entry

    def delete_entry(self, job_entry: JobCatalogEntry):
        """Delete Job from the catalog
        Arguments:
            job  (JobCatalogEntry): job to delete
        Returns:
            True if successfully removed else false
        """
        try:
            job_catalog_obj = self.session.execute(select(self.model).filter(self.model._row_id == job_entry.row_id)).scalar_one_or_none()
            job_catalog_obj.delete(self.session)
            return True
        except Exception as e:
            err_msg = f'Delete Job failed for {job_entry} with error {str(e)}.'
            logger.exception(err_msg)
            raise CatalogError(err_msg)

    def get_all_overdue_jobs(self) -> list:
        """Get the list of jobs that are overdue to be triggered
        Arguments:
            None
        Returns:
            Returns the list of all active overdue jobs
        """
        entries = self.session.execute(select(self.model).filter(and_(self.model._next_scheduled_run <= datetime.datetime.now(), self.model._active == true()))).scalars().all()
        entries = [row.as_dataclass() for row in entries]
        return entries

    def get_next_executable_job(self, only_past_jobs: bool) -> JobCatalogEntry:
        """Get the oldest job that is ready to be triggered by trigger time
        Arguments:
            only_past_jobs (bool): boolean flag to denote if only jobs with trigger time in
                past should be considered
        Returns:
            Returns the first job to be triggered
        """
        entry = self.session.execute(select(self.model).filter(and_(self.model._next_scheduled_run <= datetime.datetime.now(), self.model._active == true()) if only_past_jobs else self.model._active == true()).order_by(self.model._next_scheduled_run.asc()).limit(1)).scalar_one_or_none()
        if entry:
            return entry.as_dataclass()
        return entry

    def update_next_scheduled_run(self, job_name: str, next_scheduled_run: datetime, active: bool):
        """Update the next_scheduled_run and active column as per the provided values
        Arguments:
            job_name (str): job which should be updated

            next_run_time (datetime): the next trigger time for the job

            active (bool): the active status for the job
        Returns:
            void
        """
        job = self.session.query(self.model).filter(self.model._name == job_name).first()
        if job:
            job._next_scheduled_run = next_scheduled_run
            job._active = active
            self.session.commit()

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

class FunctionCatalogService(BaseService):

    def __init__(self, db_session: Session):
        super().__init__(FunctionCatalog, db_session)
        self._function_io_service = FunctionIOCatalogService(db_session)
        self._function_metadata_service = FunctionMetadataCatalogService(db_session)

    def insert_entry(self, name: str, impl_path: str, type: str, checksum: str, function_io_list: List[FunctionIOCatalogEntry], function_metadata_list: List[FunctionMetadataCatalogEntry]) -> FunctionCatalogEntry:
        """Insert a new function entry

        Arguments:
            name (str): name of the function
            impl_path (str): path to the function implementation relative to evadb/function
            type (str): function operator kind, classification or detection or etc
            checksum(str): checksum of the function file content, used for consistency

        Returns:
            FunctionCatalogEntry: Returns the new entry created
        """
        function_obj = self.model(name, impl_path, type, checksum)
        function_obj = function_obj.save(self.session)
        for function_io in function_io_list:
            function_io.function_id = function_obj._row_id
        io_objs = self._function_io_service.create_entries(function_io_list)
        for function_metadata in function_metadata_list:
            function_metadata.function_id = function_obj._row_id
        metadata_objs = self._function_metadata_service.create_entries(function_metadata_list)
        try:
            self.session.add_all(io_objs)
            self.session.add_all(metadata_objs)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            self.session.delete(function_obj)
            self.session.commit()
            logger.exception(f'Failed to insert entry into function catalog with exception {str(e)}')
            raise CatalogError(e)
        else:
            return function_obj.as_dataclass()

    def get_entry_by_name(self, name: str) -> FunctionCatalogEntry:
        """return the function entry that matches the name provided.
           None if no such entry found.

        Arguments:
            name (str): name to be searched
        """
        function_obj = self.session.execute(select(self.model).filter(self.model._name == name)).scalar_one_or_none()
        if function_obj:
            return function_obj.as_dataclass()
        return None

    def get_entry_by_id(self, id: int, return_alchemy=False) -> FunctionCatalogEntry:
        """return the function entry that matches the id provided.
           None if no such entry found.

        Arguments:
            id (int): id to be searched
        """
        function_obj = self.session.execute(select(self.model).filter(self.model._row_id == id)).scalar_one_or_none()
        if function_obj:
            return function_obj if return_alchemy else function_obj.as_dataclass()
        return function_obj

    def delete_entry_by_name(self, name: str):
        """Delete a function entry from the catalog FunctionCatalog

        Arguments:
            name (str): function name to be deleted

        Returns:
            True if successfully deleted else True
        """
        try:
            function_obj = self.session.execute(select(self.model).filter(self.model._name == name)).scalar_one()
            function_obj.delete(self.session)
        except Exception as e:
            logger.exception(f'Delete function failed for name {name} with error {str(e)}')
            return False
        return True

class DatabaseCatalogService(BaseService):

    def __init__(self, db_session: Session):
        super().__init__(DatabaseCatalog, db_session)

    def insert_entry(self, name: str, engine: str, params: dict):
        try:
            db_catalog_obj = self.model(name=name, engine=engine, params=params)
            db_catalog_obj = db_catalog_obj.save(self.session)
        except Exception as e:
            logger.exception(f'Failed to insert entry into database catalog with exception {str(e)}')
            raise CatalogError(e)

    def get_entry_by_name(self, database_name: str) -> DatabaseCatalogEntry:
        """
        Get the table catalog entry with given table name.
        Arguments:
            database_name  (str): Database name
        Returns:
            DatabaseCatalogEntry - catalog entry for given database name
        """
        entry = self.session.execute(select(self.model).filter(self.model._name == database_name)).scalar_one_or_none()
        if entry:
            return entry.as_dataclass()
        return entry

    def delete_entry(self, database_entry: DatabaseCatalogEntry):
        """Delete database from the catalog
        Arguments:
            database  (DatabaseCatalogEntry): database to delete
        Returns:
            True if successfully removed else false
        """
        try:
            db_catalog_obj = self.session.execute(select(self.model).filter(self.model._row_id == database_entry.row_id)).scalar_one_or_none()
            db_catalog_obj.delete(self.session)
            return True
        except Exception as e:
            err_msg = f'Delete database failed for {database_entry} with error {str(e)}.'
            logger.exception(err_msg)
            raise CatalogError(err_msg)

class JobHistoryCatalogService(BaseService):

    def __init__(self, db_session: Session):
        super().__init__(JobHistoryCatalog, db_session)

    def insert_entry(self, job_id: str, job_name: str, execution_start_time: datetime, execution_end_time: datetime) -> JobHistoryCatalogEntry:
        try:
            job_history_catalog_obj = self.model(job_id=job_id, job_name=job_name, execution_start_time=execution_start_time, execution_end_time=execution_end_time)
            job_history_catalog_obj = job_history_catalog_obj.save(self.session)
        except Exception as e:
            logger.exception(f'Failed to insert entry into job history catalog with exception {str(e)}')
            raise CatalogError(e)
        return job_history_catalog_obj.as_dataclass()

    def get_entry_by_job_id(self, job_id: int) -> List[JobHistoryCatalogEntry]:
        """
        Get all the job history catalog entry with given job id.
        Arguments:
            job_id (int): Job id
        Returns:
            list[JobHistoryCatalogEntry]: all history catalog entries for given job id
        """
        entries = self.session.execute(select(self.model).filter(self.model._job_id == job_id)).scalars().all()
        entries = [row.as_dataclass() for row in entries]
        return entries

    def update_entry_end_time(self, job_id: int, execution_start_time: datetime, execution_end_time: datetime):
        """Update the execution_end_time of the entry as per the provided values
        Arguments:
            job_id (int): id of the job whose history entry which should be updated

            execution_start_time (datetime): the start time for the job history entry

            execution_end_time (datetime): the end time for the job history entry
        Returns:
            void
        """
        job_history_entry = self.session.query(self.model).filter(and_(self.model._job_id == job_id, self.model._execution_start_time == execution_start_time)).first()
        if job_history_entry:
            job_history_entry._execution_end_time = execution_end_time
            self.session.commit()

class FunctionCostCatalogService(BaseService):

    def __init__(self, db_session: Session):
        super().__init__(FunctionCostCatalog, db_session)

    def insert_entry(self, function_id: int, name: str, cost: int) -> FunctionCostCatalogEntry:
        """Insert a new function cost entry

        Arguments:
            function_id(int): id of the function
            name (str) : name of the function
            cost(int)  : cost of the function

        Returns:
            FunctionCostCatalogEntry: Returns the new entry created
        """
        try:
            function_obj = self.model(function_id, name, cost)
            function_obj.save(self.session)
        except Exception as e:
            raise CatalogError(f'Error while inserting entry to FunctionCostCatalog: {str(e)}')

    def upsert_entry(self, function_id: int, name: str, new_cost: int):
        """Upserts a new function cost entry

        Arguments:
            function_id(int): id of the function
            name (str) : name of the function
            cost(int)  : cost of the function
        """
        try:
            function_obj = self.session.execute(select(self.model).filter(self.model._function_id == function_id)).scalar_one_or_none()
            if function_obj:
                function_obj.update(self.session, _cost=new_cost)
            else:
                self.insert_entry(function_id, name, new_cost)
        except Exception as e:
            raise CatalogError(f'Error while upserting entry to FunctionCostCatalog: {str(e)}')

    def get_entry_by_name(self, name: str) -> FunctionCostCatalogEntry:
        """return the function cost entry that matches the name provided.
           None if no such entry found.

        Arguments:
            name (str): name to be searched
        """
        try:
            function_obj = self.session.execute(select(self.model).filter(self.model._function_name == name)).scalar_one_or_none()
            if function_obj:
                return function_obj.as_dataclass()
            return None
        except Exception as e:
            raise CatalogError(f'Error while getting entry for function {name} from FunctionCostCatalog: {str(e)}')

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

def _dict_to_sql_row(dict_row: dict, columns: List[ColumnCatalogEntry]):
    for col in columns:
        if col.type == ColumnType.NDARRAY:
            dict_row[col.name] = PickleSerializer.serialize(dict_row[col.name])
        elif isinstance(dict_row[col.name], (np.generic,)):
            dict_row[col.name] = dict_row[col.name].tolist()
    return dict_row

def _deserialize_sql_row(sql_row: tuple, columns: List[ColumnCatalogEntry]):
    dict_row = {}
    for idx, col in enumerate(columns):
        if col.type == ColumnType.NDARRAY and isinstance(sql_row[col.name], bytes):
            dict_row[col.name] = PickleSerializer.deserialize(sql_row[idx])
        else:
            dict_row[col.name] = sql_row[idx]
    return dict_row

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

