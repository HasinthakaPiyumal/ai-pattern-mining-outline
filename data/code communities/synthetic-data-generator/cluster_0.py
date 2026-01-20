# Cluster 0

@pytest.fixture
def manager():
    yield DataExporterManager()

@click.command()
@torch_run_warpper
@click.option('--save_dir', type=str, required=True, default='', help='The directory to save the synthesizer')
@click.option('--model', type=str, required=True, help='The name of the model.')
@click.option('--model_path', type=str, default=None, help='The path of the model to load')
@click.option('--model_kwargs', type=str, default=None, help='[Json String] The kwargs of the model for initialization')
@click.option('--load_dir', type=str, default=None, help='The directory to load the synthesizer, if it is specified, ``model_path`` will be ignored.')
@click.option('--metadata_path', type=str, default=None, help='The path of the metadata to load')
@click.option('--data_connector', type=str, default=None, help='The name of the data connector to use')
@click.option('--data_connector_kwargs', type=str, default=None, help='[Json String] The kwargs of the data connector to use')
@click.option('--raw_data_loaders_kwargs', type=str, default=None, help='[Json String] The kwargs of the raw data loader to use')
@click.option('--processed_data_loaders_kwargs', type=str, default=None, help='[Json String] The kwargs of the processed data loader to use')
@click.option('--data_processors', type=str, default=None, help="[Comma separated list] The name of the data processors to use, e.g. 'processor_x,processor_y'")
@click.option('--data_processors_kwargs', type=str, default=None, help='[Json String] The kwargs of the data processors to use')
@click.option('--inspector_max_chunk', type=int, default=None, help='The max chunk of the inspector to load')
@click.option('--metadata_include_inspectors', type=str, default=None, help="[Comma separated list] The name of the inspectors to include, e.g. 'inspector_x,inspector_y'")
@click.option('--metadata_exclude_inspectors', type=str, default=None, help="[Comma separated list] The name of the inspectors to exclude, e.g. 'inspector_x,inspector_y'")
@click.option('--inspector_init_kwargs', type=str, default=None, help='[Json String] The kwargs of the inspector to use')
@click.option('--model_fit_kwargs', type=str, default=None, help='[Json String] The kwargs of the model fit method')
@click.option('--dry_run', type=bool, default=False, help='Only init the synthesizer without fitting and save.')
@cli_wrapper
def fit(save_dir: str, model: str, model_path: str | None, model_kwargs: str | None, load_dir: str | None, metadata_path: str | None, data_connector: str | None, data_connector_kwargs: str | None, raw_data_loaders_kwargs: str | None, processed_data_loaders_kwargs: str | None, data_processors: str | None, data_processors_kwargs: str | None, inspector_max_chunk: int | None, metadata_include_inspectors: str | None, metadata_exclude_inspectors: str | None=None, inspector_init_kwargs: str | None=None, model_fit_kwargs: str | None=None, dry_run: bool=False):
    """
    Fit the synthesizer or load a synthesizer for fitnuning/retraining/continue training...
    """
    if data_processors is not None:
        data_processors = data_processors.strip().split(',')
    if model_kwargs is not None:
        model_kwargs = json.loads(model_kwargs)
    if data_connector_kwargs is not None:
        data_connector_kwargs = json.loads(data_connector_kwargs)
    if raw_data_loaders_kwargs is not None:
        raw_data_loaders_kwargs = json.loads(raw_data_loaders_kwargs)
    if processed_data_loaders_kwargs is not None:
        processed_data_loaders_kwargs = json.loads(processed_data_loaders_kwargs)
    if data_processors_kwargs is not None:
        data_processors_kwargs = json.loads(data_processors_kwargs)
    fit_kwargs = {}
    if inspector_max_chunk is not None:
        fit_kwargs['inspector_max_chunk'] = inspector_max_chunk
    if metadata_include_inspectors is not None:
        fit_kwargs['metadata_include_inspectors'] = metadata_include_inspectors.strip().split(',')
    if metadata_exclude_inspectors is not None:
        fit_kwargs['metadata_exclude_inspectors'] = metadata_exclude_inspectors.strip().split(',')
    if inspector_init_kwargs is not None:
        fit_kwargs['inspector_init_kwargs'] = json.loads(inspector_init_kwargs)
    if model_fit_kwargs is not None:
        fit_kwargs['model_fit_kwargs'] = json.loads(model_fit_kwargs)
    if not save_dir:
        save_dir = Path(f'./sdgx-fit-model-{model}-{time.time()}')
    else:
        save_dir = Path(save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
    if load_dir:
        if model_path:
            logger.warning('Both ``model_path`` and ``load_dir`` are specified, ``model_path`` will be ignored.')
        synthesizer = Synthesizer.load(load_dir=load_dir, metadata_path=metadata_path, data_connector=data_connector, data_connector_kwargs=data_connector_kwargs, raw_data_loaders_kwargs=raw_data_loaders_kwargs, processed_data_loaders_kwargs=processed_data_loaders_kwargs, data_processors=data_processors, data_processors_kwargs=data_processors_kwargs)
    else:
        if model_kwargs and model_path:
            logger.warning('Both ``model_kwargs`` and ``model_path`` are specified, ``model_kwargs`` will be ignored.')
        synthesizer = Synthesizer(model=model, model_kwargs=model_kwargs, model_path=model_path, metadata_path=metadata_path, data_connector=data_connector, data_connector_kwargs=data_connector_kwargs, raw_data_loaders_kwargs=raw_data_loaders_kwargs, processed_data_loaders_kwargs=processed_data_loaders_kwargs, data_processors=data_processors, data_processors_kwargs=data_processors_kwargs)
    if dry_run:
        return
    synthesizer.fit(**fit_kwargs)
    save_dir = synthesizer.save(save_dir)
    return save_dir.absolute().as_posix()

@click.command()
@torch_run_warpper
@click.option('--load_dir', type=str, required=True, help='The directory to load the synthesizer.')
@click.option('--model', type=str, required=True, help='The name of the model.')
@click.option('--raw_data_loaders_kwargs', type=str, default=None, help='[Json String] The kwargs of the raw data loaders.')
@click.option('--processed_data_loaders_kwargs', type=str, default=None, help='[Json String] The kwargs of the processed data loaders.')
@click.option('--data_processors', type=str, default=None, help="[Comma separated list] The name of the data processors, e.g. 'data_processor_1,data_processor_2'.")
@click.option('--data_processors_kwargs', type=str, default=None, help='[Json String] The kwargs of the data processors.')
@click.option('--count', type=int, default=100, help='The number of samples to generate.')
@click.option('--chunksize', type=int, default=None, help='The size of each chunk. If count is very large, chunksize is recommended.')
@click.option('--model_sample_args', type=str, default=None, help='[Json String] The kwargs of the model.sample.')
@click.option('--data_exporter', type=str, default='CsvExporter', required=True, help='The name of the data exporter.')
@click.option('--data_exporter_kwargs', type=str, default=None, help='[Json String] The kwargs of the data exporter.')
@click.option('--export_dst', type=str, default=None, help='The destination of the exported data.')
@click.option('--dry_run', type=bool, default=False, help='Dry run. Only initialize the synthesizer without sampling.')
@cli_wrapper
def sample(load_dir: str, model: str, raw_data_loaders_kwargs: str | None, processed_data_loaders_kwargs: str | None, data_processors: str | None, data_processors_kwargs: str | None, count: int, chunksize: int | None, model_sample_args: str | None, data_exporter: str, data_exporter_kwargs: str | None, export_dst: str | None, dry_run: bool):
    """
    Load a synthesizer and sample.

    ``load_dir`` should contain model and metadata. Please check :ref:`Synthesizer <Synthesizer>`'s `load` method for more details.
    """
    if data_processors is not None:
        data_processors = data_processors.strip().split(',')
    if raw_data_loaders_kwargs is not None:
        raw_data_loaders_kwargs = json.loads(raw_data_loaders_kwargs)
    if processed_data_loaders_kwargs is not None:
        processed_data_loaders_kwargs = json.loads(processed_data_loaders_kwargs)
    if data_processors_kwargs is not None:
        data_processors_kwargs = json.loads(data_processors_kwargs)
    if model_sample_args is not None:
        model_sample_args = json.loads(model_sample_args)
    if data_exporter_kwargs is not None:
        data_exporter_kwargs = json.loads(data_exporter_kwargs)
    else:
        data_exporter_kwargs = {}
    if not export_dst:
        export_dst = Path(f'./sdgx-{model}-{time.time()}/sample-data.csv').expanduser().resolve()
    synthesizer = Synthesizer.load(load_dir=load_dir, model=model, raw_data_loaders_kwargs=raw_data_loaders_kwargs, processed_data_loaders_kwargs=processed_data_loaders_kwargs, data_processors=data_processors, data_processors_kwargs=data_processors_kwargs)
    exporter = DataExporterManager().init_exporter(data_exporter, **data_exporter_kwargs)
    if dry_run:
        return
    exporter.write(export_dst, synthesizer.sample(count=count, chunksize=chunksize, model_sample_args=model_sample_args))
    return export_dst

@click.command()
@cli_wrapper
def list_data_exporters():
    for model_name, model_cls in DataExporterManager().registed_exporters.items():
        print(f'{model_name} is registed as class: {model_cls}.')

class DiskCache(Cacher):
    """
    Cacher that cache data in disk with parquet format

    Args:
        blocksize (int): The blocksize of the cache.
        cache_dir (str | Path | None, optional): The directory where the cache will be stored. Defaults to None.
        identity (str | None, optional): The identity of the data source. Defaults to None.

    Todo:
        * Add partial cache when blocksize > chunksize
        * Improve cache invalidation
        * Improve performance if blocksize > chunksize
    """

    def __init__(self, cache_dir: str | Path | None=None, identity: str | None=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not cache_dir:
            cache_dir = Path.cwd() / '.sdgx_cache'
            if identity:
                cache_dir = cache_dir / identity
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def clear_cache(self) -> None:
        """
        Clear all cache in cache_dir.
        """
        for f in self.cache_dir.glob('*.parquet'):
            f.unlink()
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def clear_invalid_cache(self):
        """
        Clear all cache in cache_dir.

        TODO: Improve cache invalidation
        """
        return self.clear_cache()

    def _get_cache_filename(self, offset: int) -> Path:
        """
        Get cache filename
        """
        return self.cache_dir / f'{offset}.parquet'

    def is_cached(self, offset: int) -> bool:
        """
        Check if the data is cached by checking if the cache file exists
        """
        return self._get_cache_filename(offset).exists()

    def _refresh(self, offset: int, data: pd.DataFrame) -> None:
        """
        Refresh cache, will write data to cache file in parquet format.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if len(data) < self.blocksize:
            data.to_parquet(self._get_cache_filename(offset))
        elif len(data) > self.blocksize:
            for i in range(0, len(data), self.blocksize):
                data[i:i + self.blocksize].to_parquet(self._get_cache_filename(offset + i))
        else:
            data.to_parquet(self._get_cache_filename(offset))

    def load(self, offset: int, chunksize: int, data_connector: DataConnector) -> pd.DataFrame:
        """
        Load data from data_connector or cache
        """
        if chunksize % self.blocksize != 0:
            raise CacheError('chunksize must be multiple of blocksize, current chunksize is {} and blocksize is {}'.format(chunksize, self.blocksize))
        if chunksize != self.blocksize:
            logger.warning('chunksize must be equal to blocksize, may cause performance issue.')
        if self.is_cached(offset):
            cached_data = pd.read_parquet(self._get_cache_filename(offset))
            if len(cached_data) >= chunksize:
                return cached_data[:chunksize]
            return cached_data
        limit = max(self.blocksize, chunksize)
        data = data_connector.read(offset=offset, limit=limit)
        if data is None:
            return data
        data_list: List[pd.DataFrame] = [data]
        while len(data) < limit:
            next_data = data_connector.read(offset=offset + len(data), limit=limit - len(data))
            if next_data is None or len(next_data) == 0:
                break
            data_list.append(next_data)
        data = pd.concat(data_list, ignore_index=True) if len(data_list) > 1 else data
        self._refresh(offset, data)
        if len(data) < chunksize:
            return data
        return data[:chunksize]

    def iter(self, chunksize: int, data_connector: DataConnector) -> Generator[pd.DataFrame, None, None]:
        """
        Load data from data_connector or cache in chunk
        """
        offset = 0
        while True:
            data = self.load(offset, chunksize, data_connector)
            if data is None or len(data) == 0:
                break
            yield data
            offset += len(data)

class DataProcessor:
    """
    Base class for data processors.
    """
    fitted = False

    def check_fitted(self):
        """Check if the processor is fitted.

        Raises:
            SynthesizerProcessorError: If the processor is not fitted.
        """
        if not self.fitted:
            raise SynthesizerProcessorError('Processor NOT fitted.')

    def fit(self, metadata: Metadata | None=None, **kwargs: Dict[str, Any]):
        self._fit(metadata, **kwargs)
        self.fitted = True

    def _fit(self, metadata: Metadata | None=None, **kwargs: Dict[str, Any]):
        """Fit the data processor.

        Called before ``convert`` and ``reverse_convert``.

        Args:
            metadata (Metadata, optional): Metadata. Defaults to None.
        """
        return

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Convert raw data into processed data.

        Args:
            raw_data (pd.DataFrame): Raw data

        Returns:
            pd.DataFrame: Processed data
        """
        return raw_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """Convert processed data into raw data.

        Args:
            processed_data (pd.DataFrame): Processed data

        Returns:
            pd.DataFrame: Raw data
        """
        return processed_data

    @staticmethod
    def remove_columns(tabular_data: pd.DataFrame, column_name_to_remove: list) -> pd.DataFrame:
        """
        Remove specified columns from the input tabular data.

        Args:
            - tabular_data (pd.DataFrame): Processed tabular data
            - column_name_to_remove (list): List of column names to be removed

        Returns:
            - result_data (pd.DataFrame): Tabular data with specified columns removed
        """
        result_data = tabular_data.copy()
        try:
            result_data = result_data.drop(columns=column_name_to_remove)
        except KeyError:
            logger.warning('Duplicate column removal occurred, which might lead to unintended consequences.')
        return result_data

    @staticmethod
    def attach_columns(tabular_data: pd.DataFrame, new_columns: pd.DataFrame) -> pd.DataFrame:
        """
        Attach additional columns to an existing DataFrame.

        Args:
            - tabular_data (pd.DataFrame): The original DataFrame.
            - new_columns (pd.DataFrame): The DataFrame containing additional columns to be attached.

        Returns:
            - result_data (pd.DataFrame): The DataFrame with new_columns attached.

        Raises:
            - ValueError: If the number of rows in tabular_data and new_columns are not the same.
        """
        if tabular_data.shape[0] != new_columns.shape[0]:
            raise ValueError('Number of rows in tabular_data and new_columns must be the same.')
        result_data = pd.concat([tabular_data, new_columns], axis=1)
        return result_data

class DiscreteTransformer(Transformer):
    """
    A transformer class for handling discrete values in the input data.

    This class uses one-hot encoding to convert discrete values into a format that can be used by machine learning models.

    Attributes:
        discrete_columns (list): A list of column names that are of discrete type.
        one_hot_warning_cnt (int): The warning count for one-hot encoding. If the number of new columns after one-hot encoding exceeds this count, a warning message will be issued.
        one_hot_encoders (dict): A dictionary that stores the OneHotEncoder objects for each discrete column. The keys are the column names, and the values are the corresponding OneHotEncoder objects.
        one_hot_column_names (dict): A dictionary that stores the new column names after one-hot encoding for each discrete column. The keys are the column names, and the values are lists of new column names.
        onehot_encoder_handle_unknown (str): The parameter to handle unknown categories in the OneHotEncoder. If set to 'ignore', new categories will be ignored. If set to 'error', an error will be raised when new categories are encountered.

    Methods:
        fit(metadata: Metadata, tabular_data: DataLoader | pd.DataFrame): Fit the transformer to the input data.
        _fit_column(column_name: str, column_data: pd.DataFrame): Fit a single discrete column.
        convert(raw_data: pd.DataFrame) -> pd.DataFrame: Convert the input data using one-hot encoding.
        reverse_convert(processed_data: pd.DataFrame) -> pd.DataFrame: Reverse the one-hot encoding process to get the original data.
    """
    discrete_columns: list
    '\n    Record which columns are of discrete type.\n    '
    one_hot_warning_cnt: int
    '\n    The warning count for one-hot encoding.\n    If the number of new columns after one-hot encoding exceeds this count, a warning message will be issued.\n    '
    one_hot_encoders: dict
    '\n    A dictionary that stores the OneHotEncoder objects for each discrete column.\n    The keys are the column names, and the values are the corresponding OneHotEncoder objects.\n    '
    one_hot_column_names: dict
    '\n    A dictionary that stores the new column names after one-hot encoding for each discrete column.\n    The keys are the column names, and the values are lists of new column names.\n    '
    onehot_encoder_handle_unknown: str
    "\n    The parameter to handle unknown categories in the OneHotEncoder.\n    If set to 'ignore', new categories will be ignored.\n    If set to 'error', an error will be raised when new categories are encountered.\n    "

    def __init__(self):
        self.discrete_columns = []
        self.one_hot_warning_cnt = 512
        self.one_hot_encoders = {}
        self.one_hot_column_names = {}
        self.onehot_encoder_handle_unknown = 'ignore'

    def fit(self, metadata: Metadata, tabular_data: DataLoader | pd.DataFrame):
        """
        Fit method for the DiscreteTransformer.
        """
        logger.info('Fitting using DiscreteTransformer...')
        self.discrete_columns = metadata.get('discrete_columns')
        datetime_columns = metadata.get('datetime_columns')
        if len(self.discrete_columns) == 0:
            logger.info('Fitting using DiscreteTransformer... Finished (No Columns).')
            return
        for each_datgetime_col in datetime_columns:
            if each_datgetime_col in self.discrete_columns:
                self.discrete_columns.remove(each_datgetime_col)
                logger.info(f'Datetime column {each_datgetime_col} removed from discrete column.')
        for each_col in self.discrete_columns:
            self._fit_column(each_col, tabular_data[[each_col]])
        logger.info('Fitting using DiscreteTransformer... Finished.')
        self.fitted = True
        return

    def _fit_column(self, column_name: str, column_data: pd.DataFrame):
        """
        Fit every discrete column in `_fit_column`.

        Args:
            - column_data (pd.DataFrame): A dataframe containing a column.
            - column_name: str: column name.
        """
        self.one_hot_encoders[column_name] = OneHotEncoder(handle_unknown=self.onehot_encoder_handle_unknown, sparse_output=False)
        self.one_hot_encoders[column_name].fit(column_data)
        logger.debug(f'Discrete column {column_name} fitted.')

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert method to handle discrete values in the input data.
        """
        logger.info('Converting data using DiscreteTransformer...')
        if len(self.discrete_columns) == 0:
            logger.info('Converting data using DiscreteTransformer... Finished (No column).')
            return
        processed_data = raw_data.copy()
        for each_col in self.discrete_columns:
            new_onehot_columns = self.one_hot_encoders[each_col].transform(raw_data[[each_col]])
            new_onehot_column_names = self.one_hot_encoders[each_col].get_feature_names_out()
            self.one_hot_column_names[each_col] = new_onehot_column_names
            if len(new_onehot_column_names) > self.one_hot_warning_cnt:
                logger.warning(f'Column {each_col} has too many discrete values ({len(new_onehot_column_names)} values), may consider as a continous column?')
            processed_data = self.attach_columns(processed_data, pd.DataFrame(new_onehot_columns, columns=new_onehot_column_names))
            logger.debug(f'Column {each_col} converted.')
        logger.info(f'Processed data shape: {processed_data.shape}.')
        logger.info('Converting data using DiscreteTransformer... Finished.')
        processed_data = self.remove_columns(processed_data, self.discrete_columns)
        return processed_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse_convert method for the transformer.

        Args:
            - processed_data (pd.DataFrame): A dataframe containing onehot encoded columns.

        Returns:
            - pd.DataFrame: inverse transformed processed data.
        """
        reversed_data = processed_data.copy()
        for each_col in self.discrete_columns:
            one_hot_column_set = processed_data[self.one_hot_column_names[each_col]]
            res_column_data = self.one_hot_encoders[each_col].inverse_transform(pd.DataFrame(one_hot_column_set, columns=self.one_hot_column_names[each_col]))
            reversed_data = self.attach_columns(reversed_data, pd.DataFrame(res_column_data, columns=[each_col]))
            reversed_data = self.remove_columns(reversed_data, self.one_hot_column_names[each_col])
        logger.info('Data inverse-converted by DiscreteTransformer.')
        return reversed_data
    pass

class DatetimeFormatter(Formatter):
    """
    A class for formatting datetime columns in a pandas DataFrame.

    DatetimeFormatter is designed to handle the conversion of datetime columns to timestamp format and vice versa.
    It uses metadata to identify datetime columns and their corresponding datetime formats.

    Attributes:
        datetime_columns (list): List of column names that are of datetime type.
        datetime_formats (dict): Dictionary with column names as keys and datetime formats as values.
        dead_columns (list): List of column names that are no longer needed or to be removed.
        fitted (bool): Indicates whether the formatter has been fitted.

    Methods:
        fit(metadata: Metadata | None = None, **kwargs: dict[str, Any]): Fits the formatter by recording the datetime columns and their formats.
        convert(raw_data: pd.DataFrame) -> pd.DataFrame: Converts datetime columns in raw_data to timestamp format.
        reverse_convert(processed_data: pd.DataFrame) -> pd.DataFrame: Converts timestamp columns in processed_data back to datetime format.
    """
    datetime_columns: list
    '\n    List to store the columns that are of datetime type.\n    '
    datetime_formats: Dict
    '\n    Dictionary to store the datetime formats for each column, with default value as an empty string.\n    '
    dead_columns: list
    '\n    List to store columns that are no longer needed or to be removed.\n    '

    def __init__(self):
        self.fitted = False
        self.datetime_columns = []
        self.datetime_formats = defaultdict(str)
        self.dead_columns = []

    def fit(self, metadata: Metadata | None=None, **kwargs: dict[str, Any]):
        """
        Fit method for datetime formatter, the datetime column and datetime format need to be recorded.

        If there is a column without format, the default format will be used for output (this may cause some problems).

        Formatter need to use metadata to record which columns belong to datetime type, and convert timestamp back to datetime type during post-processing.
        """
        self.datetime_formats = metadata.get('datetime_format')
        datetime_columns = []
        dead_columns = []
        meta_datetime_columns = metadata.get('datetime_columns')
        for each_col in meta_datetime_columns:
            if each_col in self.datetime_formats.keys():
                datetime_columns.append(each_col)
            else:
                dead_columns.append(each_col)
                logger.warning(f'Column {each_col} has no datetime_format, DatetimeFormatter will REMOVE this column！')
        if not set(datetime_columns) - set(metadata.discrete_columns):
            metadata.change_column_type(datetime_columns, 'discrete', 'datetime')
        metadata.remove_column(dead_columns)
        self.datetime_columns = datetime_columns
        self.dead_columns = dead_columns
        logger.info('DatetimeFormatter Fitted.')
        self.fitted = True
        return

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert method to convert datetime samples into timestamp.

        Args:
            - raw_data (pd.DataFrame): Unprocessed table data
        """
        if len(self.datetime_columns) == 0:
            logger.info('Converting data using DatetimeFormatter... Finished (No datetime columns).')
            return raw_data
        for each_col in self.dead_columns:
            raw_data = self.remove_columns(raw_data, [each_col])
            logger.warning(f'Column {each_col} was removed because lack of format info.')
        logger.info('Converting data using DatetimeFormatter...')
        res_data = self.convert_datetime_columns(self.datetime_columns, self.datetime_formats, raw_data)
        logger.info('Converting data using DatetimeFormatter... Finished.')
        return res_data

    @staticmethod
    def convert_datetime_columns(datetime_column_list, datetime_formats, processed_data):
        """
        Convert datetime columns in processed_data from string to timestamp (int)

        Args:
            - datetime_column_list (list): List of columns that are date time type
            - processed_data (pd.DataFrame): Processed table data

        Returns:
            - result_data (pd.DataFrame): Processed table data with datetime columns converted to timestamp
        """

        def datetime_formatter(each_value, datetime_format):
            """
            convert each single column datetime string to timestamp int value.
            """
            try:
                datetime_obj = datetime.strptime(str(each_value), datetime_format)
                each_stamp = datetime.timestamp(datetime_obj)
            except Exception as e:
                logger.warning(f'An error occured when convert str to timestamp {e}, we set as mean.')
                logger.warning(f'Input parameters: ({str(each_value)}, {datetime_format})')
                logger.warning(f'Input type: ({type(each_value)}, {type(datetime_format)})')
                each_stamp = np.nan
            return each_stamp
        result_data: pd.DataFrame = processed_data.copy()
        for column in datetime_column_list:
            result_data[column] = result_data[column].apply(datetime_formatter, datetime_format=datetime_formats[column])
            result_data[column].fillna(result_data[column].mean(), inplace=True)
        return result_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        reverse_convert method for datetime formatter.

        Does not require any action.
        """
        if len(self.datetime_columns) == 0:
            logger.info('Data reverse-converted by DatetimeFormatter (No datetime columns).')
            return processed_data
        logger.info('Data reverse-converting by DatetimeFormatter...')
        logger.info(f'parameters : {self.datetime_columns}, {self.datetime_formats}')
        result_data = self.convert_timestamp_to_datetime(self.datetime_columns, self.datetime_formats, processed_data)
        logger.info('Data reverse-converted by DatetimeFormatter... Finished.')
        return result_data

    @staticmethod
    def convert_timestamp_to_datetime(timestamp_column_list, format_dict, processed_data):
        """
        Convert timestamp columns to datetime format in a DataFrame.

        Parameters:
            - timestamp_column_list (list): List of column names in the DataFrame which are of timestamp type.
            - datetime_column_dict (dict): Dictionary with column names as keys and datetime format as values.
            - processed_data (pd.DataFrame): DataFrame containing the processed data.

        Returns:
            - result_data (pd.DataFrame): DataFrame with timestamp columns converted to datetime format.

        TODO:
            if the value <0, the result will be `No Datetime`, try to fix it.
        """

        def column_timestamp_formatter(each_stamp: int, timestamp_format: str) -> str:
            try:
                each_str = datetime.fromtimestamp(each_stamp).strftime(timestamp_format)
            except Exception as e:
                logger.debug(f'An error occured when convert timestamp to str {e}.')
                each_str = 'No Datetime'
            return each_str
        result_data = processed_data.copy()
        for column in timestamp_column_list:
            if column in result_data.columns:
                result_data[column] = result_data[column].apply(column_timestamp_formatter, timestamp_format=format_dict[column])
            else:
                logger.error(f"Column {column} not in processed data's column list!")
        return result_data

def test_save_and_load(synthesizer, save_dir):
    assert synthesizer.save(save_dir)
    assert (save_dir / synthesizer.METADATA_SAVE_NAME).exists()
    assert (save_dir / synthesizer.MODEL_SAVE_DIR).exists()
    synthesizer = Synthesizer.load(save_dir, model=MockModel)
    assert synthesizer

@pytest.fixture
def manager():
    yield DataExporterManager()

