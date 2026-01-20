# Cluster 7

class Synthesizer:
    """
    Synthesizer is the high level interface for synthesizing data.

    We provided several example usage in our `Github repository <https://github.com/hitsz-ids/synthetic-data-generator/tree/main/example>`_.

    Args:

        model (str | SynthesizerModel | type[SynthesizerModel]): The name of the model or the model itself. Type of model must be :class:`~sdgx.models.base.SynthesizerModel`.
            When model is a string, it must be registered in :class:`~sdgx.models.manager.ModelManager`.
        model_path (str | Path, optional): The path to the model file. Defaults to None. Used to load the model if ``model`` is a string or type of :class:`~sdgx.models.base.SynthesizerModel`.
        model_kwargs (dict[str, Any], optional): The keyword arguments for model. Defaults to None.
        metadata (Metadata, optional): The metadata to use. Defaults to None.
        metadata_path (str | Path, optional): The path to the metadata file. Defaults to None. Used to load the metadata if ``metadata`` is None.
        data_connector (DataConnector | type[DataConnector] | str, optional): The data connector to use. Defaults to None.
            When data_connector is a string, it must be registered in :class:`~sdgx.data_connectors.manager.DataConnectorManager`.
        data_connector_kwargs (dict[str, Any], optional): The keyword arguments for data connectors. Defaults to None.
        raw_data_loaders_kwargs (dict[str, Any], optional): The keyword arguments for raw data loaders. Defaults to None.
        processed_data_loaders_kwargs (dict[str, Any], optional): The keyword arguments for processed data loaders. Defaults to None.
        data_processors (list[str | DataProcessor | type[DataProcessor]], optional): The data processors to use. Defaults to None.
            When data_processor is a string, it must be registered in :class:`~sdgx.data_processors.manager.DataProcessorManager`.
        data_processors_kwargs (dict[str, dict[str, Any]], optional): The keyword arguments for data processors. Defaults to None.

    Example:

        .. code-block:: python

            from sdgx.data_connectors.csv_connector import CsvConnector
            from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
            from sdgx.synthesizer import Synthesizer
            from sdgx.utils import download_demo_data

            dataset_csv = download_demo_data()
            data_connector = CsvConnector(path=dataset_csv)
            synthesizer = Synthesizer(
                model=CTGANSynthesizerModel(epochs=1),  # For quick demo
                data_connector=data_connector,
            )
            synthesizer.fit()
            sampled_data = synthesizer.sample(1000)
    """
    METADATA_SAVE_NAME = 'metadata.json'
    '\n    Default name for metadata file\n    '
    MODEL_SAVE_DIR = 'model'
    '\n    Default name for model directory\n    '

    def __init__(self, model: str | SynthesizerModel | type[SynthesizerModel], model_path: None | str | Path=None, model_kwargs: None | dict[str, Any]=None, metadata: None | Metadata=None, metadata_path: None | str | Path=None, data_connector: None | str | DataConnector | type[DataConnector]=None, data_connector_kwargs: None | dict[str, Any]=None, raw_data_loaders_kwargs: None | dict[str, Any]=None, processed_data_loaders_kwargs: None | dict[str, Any]=None, data_processors: None | list[str | DataProcessor | type[DataProcessor]]=None, data_processors_kwargs: None | dict[str, Any]=None):
        if isinstance(data_connector, str) or isinstance(data_connector, type):
            data_connector = DataConnectorManager().init_data_connector(data_connector, **data_connector_kwargs or {})
        if data_connector:
            self.dataloader = DataLoader(data_connector, **raw_data_loaders_kwargs or {})
        else:
            logger.warning('No data_connector provided, will not support `fit`')
            self.dataloader = None
        self.data_processors_manager = DataProcessorManager()
        if not data_processors:
            data_processors = self.data_processors_manager.registed_default_processor_list
        logger.info(f'Using data processors: {data_processors}')
        self.data_processors = [d if isinstance(d, DataProcessor) else self.data_processors_manager.init_data_processor(d, **data_processors_kwargs or {}) for d in data_processors]
        if metadata and metadata_path:
            raise SynthesizerInitError('metadata and metadata_path cannot be specified at the same time')
        if metadata:
            self.metadata = metadata
        elif metadata_path:
            self.metadata = Metadata.load(metadata_path)
        else:
            self.metadata = None
        self.model_manager = ModelManager()
        if isinstance(model, SynthesizerModel) and model_path:
            raise SynthesizerInitError('model as instance and model_path cannot be specified at the same time')
        if (isinstance(model, str) or isinstance(model, type)) and model_path:
            self.model = self.model_manager.load(model, model_path, **model_kwargs or {})
            if model_kwargs:
                logger.warning('model_kwargs will be ignored when loading model from model_path')
        elif isinstance(model, str) or isinstance(model, type):
            self.model = self.model_manager.init_model(model, **model_kwargs or {})
        elif isinstance(model, SynthesizerModel) or isinstance(model, StatisticSynthesizerModel):
            self.model = model
            if model_kwargs:
                logger.warning('model_kwargs will be ignored when using already initialized model')
        else:
            raise SynthesizerInitError('model or model_path must be specified')
        self.processed_data_loaders_kwargs = processed_data_loaders_kwargs or {}

    def save(self, save_dir: str | Path) -> Path:
        """
        Dump metadata and model to file

        Args:
            save_dir (str | Path): The directory to save the model.

        Returns:
            Path: The directory to save the synthesizer.
        """
        save_dir = Path(save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Saving synthesizer to {save_dir}')
        if self.metadata:
            self.metadata.save(save_dir / self.METADATA_SAVE_NAME)
        model_save_dir = save_dir / self.MODEL_SAVE_DIR
        model_save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(model_save_dir)
        return save_dir

    @classmethod
    def load(cls, load_dir: str | Path, model: str | type[SynthesizerModel], metadata: None | Metadata=None, data_connector: None | str | DataConnector | type[DataConnector]=None, data_connector_kwargs: None | dict[str, Any]=None, raw_data_loaders_kwargs: None | dict[str, Any]=None, processed_data_loaders_kwargs: None | dict[str, Any]=None, data_processors: None | list[str | DataProcessor | type[DataProcessor]]=None, data_processors_kwargs: None | dict[str, dict[str, Any]]=None, model_kwargs=None) -> 'Synthesizer':
        """
        Load metadata and model, allow rebuilding Synthesizer for finetuning or other use cases.

        We need ``model`` as not every model support *pickle* way to save and load.

        Args:
            load_dir (str | Path): The directory to load the model.
            model (str | type[SynthesizerModel]): The name of the model or the model itself. Type of model must be :class:`~sdgx.models.base.SynthesizerModel`.
                When model is a string, it must be registered in :class:`~sdgx.models.manager.ModelManager`.
            metadata (Metadata, optional): The metadata to use. Defaults to None.
            data_connector (DataConnector | type[DataConnector] | str, optional): The data connector to use. Defaults to None.
                When data_connector is a string, it must be registered in :class:`~sdgx.data_connectors.manager.DataConnectorManager`.
            data_connector_kwargs (dict[str, Any], optional): The keyword arguments for data connectors. Defaults to None.
            raw_data_loaders_kwargs (dict[str, Any], optional): The keyword arguments for raw data loaders. Defaults to None.
            processed_data_loaders_kwargs (dict[str, Any], optional): The keyword arguments for processed data loaders. Defaults to None.
            data_processors (list[str | DataProcessor | type[DataProcessor]], optional): The data processors to use. Defaults to None.
                When data_processor is a string, it must be registered in :class:`~sdgx.data_processors.manager.DataProcessorManager`.
            data_processors_kwargs (dict[str, dict[str, Any]], optional): The keyword arguments for data processors. Defaults to None.

        Returns:
            Synthesizer: The synthesizer instance.
        """
        load_dir = Path(load_dir).expanduser().resolve()
        logger.info(f'Loading synthesizer from {load_dir}')
        if not load_dir.exists():
            raise SynthesizerInitError(f'{load_dir.as_posix()} does not exist')
        model_path = load_dir / cls.MODEL_SAVE_DIR
        if not model_path.exists():
            raise SynthesizerInitError(f'{model_path.as_posix()} does not exist, cannot load model.')
        metadata_path = load_dir / cls.METADATA_SAVE_NAME
        if not metadata_path.exists():
            metadata_path = None
        return Synthesizer(model=model, model_path=model_path, metadata=metadata, metadata_path=metadata_path, model_kwargs=model_kwargs, data_connector=data_connector, data_connector_kwargs=data_connector_kwargs, raw_data_loaders_kwargs=raw_data_loaders_kwargs, processed_data_loaders_kwargs=processed_data_loaders_kwargs, data_processors=data_processors, data_processors_kwargs=data_processors_kwargs)

    def fit(self, metadata: None | Metadata=None, inspector_max_chunk: int=10, metadata_include_inspectors: None | list[str]=None, metadata_exclude_inspectors: None | list[str]=None, inspector_init_kwargs: None | dict[str, Any]=None, model_fit_kwargs: None | dict[str, Any]=None):
        """
        Fit the synthesizer with metadata and data processors.

        Raw data will be loaded from the dataloader and processed by the data processors in a Generator.
        The Generator, which prevents the processed data, will be wrapped into a DataLoader, aka ProcessedDataLoader.
        The ProcessedDataLoader will be used to fit the model.

        For more information about DataLoaders, please refer to the :class:`~sdgx.data_loaders.base.DataLoader`.

        For more information about DataProcessors, please refer to the :class:`~sdgx.data_processors.base.DataProcessor`.

        For more information about DataConnectors, please refer to the :class:`~sdgx.data_connectors.base.DataConnector`. Especially, the :class:`~sdgx.data_connectors.generator_connector.GeneratorConnector`.

        Args:
            metadata (Metadata, optional): The metadata to use. Defaults to None. If None, it will be inferred from the dataloader with the :func:`~sdgx.data_models.metadata.Metadata.from_dataloader` method.
            inspector_max_chunk (int, optional): The maximum number of chunks to inspect. Defaults to 10.
            metadata_include_inspectors (list[str], optional): The list of metadata inspectors to include. Defaults to None.
            metadata_exclude_inspectors (list[str], optional): The list of metadata inspectors to exclude. Defaults to None.
            inspector_init_kwargs (dict[str, Any], optional): The keyword arguments for metadata inspectors. Defaults to None.
            model_fit_kwargs (dict[str, Any], optional): The keyword arguments for model.fit. Defaults to None.
        """
        if self.dataloader is None:
            raise SynthesizerInitError('Cannot fit without dataloader, check `data_connector` parameter when initializing Synthesizer')
        metadata = metadata or self.metadata or Metadata.from_dataloader(self.dataloader, max_chunk=inspector_max_chunk, include_inspectors=metadata_include_inspectors, exclude_inspectors=metadata_exclude_inspectors, inspector_init_kwargs=inspector_init_kwargs)
        self.metadata = metadata.model_copy()
        logger.info('Fitting data processors...')
        if not self.dataloader:
            logger.info('Fitting without dataloader.')
        start_time = time.time()
        for d in self.data_processors:
            if self.dataloader:
                d.fit(metadata=metadata, tabular_data=self.dataloader)
            else:
                d.fit(metadata=metadata)
        logger.info(f'Fitted {len(self.data_processors)} data processors in  {time.time() - start_time}s.')

        def chunk_generator() -> Generator[pd.DataFrame, None, None]:
            for chunk in self.dataloader.iter():
                for d in self.data_processors:
                    chunk = d.convert(chunk)
                yield chunk
        logger.info('Initializing processed data loader...')
        start_time = time.time()
        processed_dataloader = DataLoader(GeneratorConnector(chunk_generator), identity=self.dataloader.identity, **self.processed_data_loaders_kwargs)
        logger.info(f'Initialized processed data loader in {time.time() - start_time}s')
        try:
            logger.info('Model fit Started...')
            self.model.fit(metadata, processed_dataloader, **model_fit_kwargs or {})
            logger.info('Model fit... Finished')
        finally:
            processed_dataloader.finalize(clear_cache=True)

    def sample(self, count: int, chunksize: None | int=None, metadata: None | Metadata=None, model_sample_args: None | dict[str, Any]=None) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
        """
        Sample data from the synthesizer.

        Args:
            count (int): The number of samples to generate.
            chunksize (int, optional): The chunksize to use. Defaults to None. If is not None, the data will be sampled in chunks.
                And will return a generator that yields chunks of samples.
            metadata (Metadata, optional): The metadata to use. Defaults to None. If None, will use the metadata in fit first.
            model_sample_args (dict[str, Any], optional): The keyword arguments for model.sample. Defaults to None.

        Returns:
            pd.DataFrame | typing.Generator[pd.DataFrame, None, None]: The sampled data. When chunksize is not None, it will be a generator.
        """
        logger.info('Sampling...')
        metadata = metadata or self.metadata
        self.metadata = metadata
        if not model_sample_args:
            model_sample_args = {}
        if chunksize is None:
            return self._sample_once(count, model_sample_args)
        if chunksize > count:
            raise SynthesizerSampleError('chunksize must be less than or equal to count')

        def generator_sample_caller():
            sample_times = count // chunksize
            for _ in range(sample_times):
                sample_data = self._sample_once(chunksize, model_sample_args)
                for d in self.data_processors:
                    sample_data = d.reverse_convert(sample_data)
                yield sample_data
            if count % chunksize > 0:
                sample_data = self._sample_once(count % chunksize, model_sample_args)
                for d in self.data_processors:
                    sample_data = d.reverse_convert(sample_data)
                yield sample_data
        return generator_sample_caller()

    def _sample_once(self, count: int, model_sample_args: None | dict[str, Any]=None) -> pd.DataFrame:
        """
        Sample data once.

        DataProcessors may drop some broken data after reverse_convert.
        So we oversample first and then take the first `count` samples.

        TODO:

            - Use an adaptive scale for oversampling will be better for performance.

        """
        missing_count = count
        max_trails = 50
        sample_data_list = []
        psb = tqdm.tqdm(total=count, desc='Sampling')
        batch_size: int = 0
        multiply_factor: float = 4.0
        if isinstance(self.model, BatchedSynthesizer):
            batch_size = self.model.get_batch_size()
            multiply_factor = 1.2
            if isinstance(self.model, CTGANSynthesizerModel):
                model_sample_args = {'drop_more': False}
        while missing_count > 0 and max_trails > 0:
            sample_data = self.model.sample(max(int(missing_count * multiply_factor), batch_size), **model_sample_args)
            for d in self.data_processors:
                sample_data = d.reverse_convert(sample_data)
            sample_data = sample_data.dropna(how='all')
            sample_data_list.append(sample_data)
            missing_count = missing_count - len(sample_data)
            psb.update(len(sample_data))
            max_trails -= 1
        return pd.concat(sample_data_list)[:count]

    def cleanup(self):
        """
        Cleanup resources. This will cause model unavailable and clear the cache.

        It useful when Synthesizer object is no longer needed and may hold large resources like GPUs.
        """
        if self.dataloader:
            self.dataloader.finalize(clear_cache=True)
        if hasattr(self, 'model'):
            del self.model

    def __del__(self):
        self.cleanup()

class SynthesizerModel:
    use_dataloader: bool = False
    use_raw_data: bool = False

    def __init__(self, *args, **kwargs) -> None:
        if 'use_dataloader' in kwargs.keys():
            self.use_dataloader = kwargs['use_dataloader']
        if 'use_raw_data' in kwargs.keys():
            self.use_raw_data = kwargs['use_raw_data']

    def _check_access_type(self):
        if self.use_dataloader == self.use_raw_data == False:
            raise SynthesizerInitError('Data access type not specified, please use `use_raw_data: bool` or `use_dataloader: bool` to specify data access type.')
        elif self.use_dataloader == self.use_raw_data == True:
            raise SynthesizerInitError('Duplicate data access type found.')

    def fit(self, metadata: Metadata, dataloader: DataLoader, *args, **kwargs):
        """
        Fit the model using the given metadata and dataloader.

        Args:
            metadata (Metadata): The metadata to use.
            dataloader (DataLoader): The dataloader to use.
        """
        raise NotImplementedError

    def sample(self, count: int, *args, **kwargs) -> pd.DataFrame:
        """
        Sample data from the model.

        Args:
            count (int): The number of samples to generate.

        Returns:
            pd.DataFrame: The generated data.
        """
        raise NotImplementedError

    def save(self, save_dir: str | Path):
        """
        Dump model to file.

        Args:
            save_dir (str | Path): The directory to save the model.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, save_dir: str | Path, **kwargs) -> 'SynthesizerModel':
        """
        Load model from file.

        Args:
            save_dir (str | Path): The directory to load the model from.
        """
        raise NotImplementedError

class LLMBaseModel(SynthesizerModel):
    """
    This is a base class for generating synthetic data using LLM (Large Language Model).

    Note:
    - When using the data loader, the original data is transformed to pd.DataFrame format for subsequent processing.
    - It is not recommended to use this model with large data tables due to excessive token consumption in some expensive LLM service.
    - Generating data based on metadata is a potential way to generate data that cannot be made public and contains sensitive information.
    """
    use_raw_data = False
    '\n    By default, we use raw_data for data access.\n\n    When using the data loader, due to the need of randomization operation, we currently use the `.load_all()` to transform the original data to pd.DataFrame format for subsequent processing.\n\n    Due to the characteristics of the OpenAI GPT service, we do not recommend running this model with large data tables, which will consume your tokens excessively.\n    '
    use_metadata = False
    '\n    In this model, we accept a data generation paradigm that only provides metadata.\n\n    When only metadata is provided, sdgx will format the metadata of the data set into a message and transmit it to GPT, and GPT will generate similar data based on what it knows.\n\n    This is a potential way to generate data that cannot be made public and contains sensitive information.\n    '
    _metadata = None
    '\n    the metadata.\n    '
    off_table_features = []
    '\n    * Experimental Feature\n\n    Whether infer data columns that do not exist in the real data table, the effect may not be very good.\n    '
    prompts = {'message_prefix': 'Suppose you are the best data generating model in this world, we have some data samples with the following information:\n\n', 'message_suffix': '\nGenerate synthetic data samples based on the above information and your knowledge, each sample should be output on one line (do not output in multiple lines), the output format of the sample is the same as the example in this message, such as "column_name_1 is value_1", the count of the generated data samples is ', 'system_role_content': 'You are a powerful synthetic data generation model.'}
    '\n    Prompt words for generating data (preliminary version, improvements welcome).\n    '
    columns = []
    '\n    The columns of the data set.\n    '
    dataset_description = ''
    '\n    The description of the data set.\n    '
    _responses = []
    '\n    A list to store the responses received from the LLM.\n    '
    _message_list = []
    '\n    A list to store the messages used to ask LLM.\n    '

    def _check_access_type(self):
        """
        Checks the data access type.

        Raises:
            SynthesizerInitError: If data access type is not specified or if duplicate data access type is found.
        """
        if self.use_dataloader == self.use_raw_data == self.use_metadata == False:
            raise SynthesizerInitError('Data access type not specified, please use `use_raw_data: bool` or `use_dataloader: bool` to specify data access type.')
        if self.use_dataloader == self.use_raw_data == True:
            raise SynthesizerInitError('Duplicate data access type found.')

    def _form_columns_description(self):
        """
        We believe that giving information about a column helps improve data quality.

        Currently, we leave this function to Good First Issue until March 2024, if unclaimed we will implement it quickly.
        """
        raise NotImplementedError

    def _form_message_with_offtable_features(self):
        """
        This function forms a message with off-table features.

        If there are more off-table columns, additional processing is excuted here.
        """
        if self.off_table_features:
            logger.info(f'Use off_table_feature = {self.off_table_features}.')
            return f'Also, you should try to infer another {len(self.off_table_features)} columns based on your knowledge, the name of these columns are : {self.off_table_features}, attach these columns after the original table. \n'
        else:
            logger.info('No off_table_feature needed in current model.')
            return ''

    def _form_dataset_description(self):
        """
        This function is used to form the dataset description.

        Returns:
            str: The description of the generated table.
        """
        if self.dataset_description:
            logger.info(f'Use dataset_description = {self.dataset_description}.')
            return '\nThe description of the generated table is ' + self.dataset_description + '\n'
        else:
            logger.info('No dataset_description given in current model.')
            return ''

class MultiTableSynthesizerModel(SynthesizerModel):
    """MultiTableSynthesizerModel

    The base model of multi-table statistic models.
    """
    metadata_combiner: MetadataCombiner = None
    "\n    metadata_combiner is a sdgx builtin class, it stores all tables' metadata and relationships.\n\n    This parameter must be specified when initializing the multi-table class.\n    "
    tables_data_frame: Dict[str, Any] = defaultdict()
    "\n    tables_data_frame is a dict contains every table's csv data frame.\n    For a small amount of data, this scheme can be used.\n    "
    tables_data_loader: Dict[str, Any] = defaultdict()
    "\n    tables_data_loader is a dict contains every table's data loader.\n    "
    _parent_id: List = []
    "\n    _parent_id is used to store all parent table's parimary keys in list.\n    "
    _table_synthesizers: Dict[str, Any] = {}
    '\n    _table_synthesizers is a dict to store model for each table.\n    '
    parent_map: Dict = defaultdict()
    '\n    The mapping from all child tables to their parent table.\n    '
    child_map: Dict = defaultdict()
    '\n    The mapping from all parent tabels to their child table.\n    '

    def __init__(self, metadata_combiner: MetadataCombiner, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata_combiner = metadata_combiner
        self._calculate_parent_and_child_map()
        self.check()

    def _calculate_parent_and_child_map(self):
        """Get the mapping from all parent tables to self._parent_map
        - key(str) is a child map;
        - value(str) is the parent map.
        """
        relationships = self.metadata_combiner.relationships
        for each_relationship in relationships:
            parent_table = each_relationship.parent_table
            child_table = each_relationship.child_table
            self.parent_map[child_table] = parent_table
            self.child_map[parent_table] = child_table

    def _get_foreign_keys(self, parent_table, child_table):
        """Get the foreign key list from a relationship"""
        relationships = self.metadata_combiner.relationships
        for each_relationship in relationships:
            if each_relationship.parent_table == parent_table and each_relationship.child_table == child_table:
                return each_relationship.foreign_keys
        return []

    def _get_all_foreign_keys(self, child_table):
        """Given a child table, return ALL foreign keys from metadata."""
        all_foreign_keys = []
        relationships = self.metadata_combiner.relationships
        for each_relationship in relationships:
            if each_relationship.child_table == child_table:
                all_foreign_keys.append(each_relationship.foreign_keys)
        return all_foreign_keys

    def _finalize(self):
        """Finalize the"""
        raise NotImplementedError

    def check(self, check_circular=True):
        """Excute necessary checks

        - check access type
        - check metadata_combiner
        - check relationship
        - check each metadata
        - validate circular relationships
        - validate child map_circular relationship
        - validate all tables connect relationship
        - validate column relationships foreign keys
        """
        self._check_access_type()
        if not isinstance(self.metadata_combiner, MetadataCombiner):
            raise SynthesizerInitError('Wrong Metadata Combiner found.')
        pass

    def fit(self, dataloader: Dict[str, DataLoader], raw_data: Dict[str, pd.DataFrame], *args, **kwargs):
        """
        Fit the model using the given metadata and dataloader.

        Args:
            dataloader (Dict[str, DataLoader]): The dataloader to use to fit the model.
            raw_data (Dict[str, pd.DataFrame]): The raw pd.DataFrame to use to fit the model.
        """
        raise NotImplementedError

    def sample(self, count: int, *args, **kwargs) -> pd.DataFrame:
        """
        Sample data from the model.

        Args:
            count (int): The number of samples to generate.

        Returns:
            pd.DataFrame: The generated data.
        """
        raise NotImplementedError

    def save(self, save_dir: str | Path):
        pass

    @classmethod
    def load(target_path: str | Path):
        pass
    pass

@pytest.fixture
def dummy_single_table_metadata(dummy_single_table_data_loader):
    yield Metadata.from_dataloader(dummy_single_table_data_loader)

@pytest.fixture
def demo_single_table_metadata(demo_single_table_data_loader):
    yield Metadata.from_dataloader(demo_single_table_data_loader)

@pytest.fixture
def demo_multi_data_parent_matadata(demo_multi_table_data_loader):
    yield Metadata.from_dataloader(demo_multi_table_data_loader['store'])

@pytest.fixture
def demo_multi_data_child_matadata(demo_multi_table_data_loader):
    yield Metadata.from_dataloader(demo_multi_table_data_loader['train'])

def test_datetime_formatter_test_df(datetime_test_df: pd.DataFrame):

    def df_generator():
        yield datetime_test_df
    data_processors = [DatetimeFormatter()]
    dataconnector = GeneratorConnector(df_generator)
    dataloader = DataLoader(dataconnector, chunksize=CHUNK_SIZE)
    metadata = Metadata.from_dataloader(dataloader)
    metadata.datetime_columns = ['date']
    metadata.discrete_columns = []
    metadata.datetime_format = {'date': '%Y-%m-%d'}
    for d in data_processors:
        d.fit(metadata=metadata, tabular_data=dataloader)

    def chunk_generator() -> Generator[pd.DataFrame, None, None]:
        for chunk in dataloader.iter():
            for d in data_processors:
                chunk = d.convert(chunk)
            assert not chunk.isna().any().any()
            assert not chunk.isnull().any().any()
            yield chunk
    processed_dataloader = DataLoader(GeneratorConnector(chunk_generator), identity=dataloader.identity)
    df = processed_dataloader.load_all()
    assert not df.isna().any().any()
    assert not df.isnull().any().any()
    reverse_converted_df = df
    for d in data_processors:
        reverse_converted_df = d.reverse_convert(df)
    assert reverse_converted_df.eq(datetime_test_df).all().all()

def test_datetime_formatter_test_df_dead_column(datetime_test_df: pd.DataFrame):
    """
    Test the DatetimeFormatter class with a DataFrame that has datetime columns.

    Parameters:
    datetime_test_df (pd.DataFrame): The DataFrame to test.

    Returns:
    None

    Raises:
    AssertionError: If any of the assertions fail.
    """
    assert datetime_test_df.shape == (1000, 7)
    metadata_df = Metadata.from_dataframe(datetime_test_df)
    assert metadata_df.datetime_columns == {'simple_datetime_2', 'date_with_time', 'simple_datetime'}
    metadata_df.datetime_format = {}
    transformer = DatetimeFormatter()
    transformer.fit(metadata=metadata_df)
    assert transformer.datetime_columns == []
    assert set(transformer.dead_columns) == {'simple_datetime_2', 'date_with_time', 'simple_datetime'}

@pytest.fixture
def metadata(dataloader):
    yield Metadata.from_dataloader(dataloader)

@pytest.mark.parametrize('cacher', ['NoCache', 'DiskCache'])
def test_demo_dataloader(dataloader_builder: DataLoader, cacher, demo_single_table_data_connector):
    d: DataLoader = dataloader_builder(data_connector=demo_single_table_data_connector, cacher=cacher)
    assert len(d) == 48842
    assert sorted(d.columns()) == sorted(d.keys()) == sorted(['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
    assert d.shape == (48842, 15)
    assert d.load_all().shape == (48842, 15)
    assert d[:].shape == d.shape
    assert d[:100].shape == (100, 15)
    assert d[100:].shape == (48842 - 100, 15)
    assert d[100:10000].shape == (10000 - 100, 15)
    assert d[100:10000:2].shape == ((10000 - 100) // 2, 15)
    assert d[['age', 'workclass']].shape == (48842, 2)
    for df in d.iter():
        assert len(df) == d.chunksize
        break

@pytest.fixture
def dataloader_builder(cacher_kwargs):
    yield partial(DataLoader, cacher_kwargs=cacher_kwargs)

@pytest.fixture
def generator_connector():
    yield GeneratorConnector(generator_caller)

@pytest.mark.parametrize('cacher', ['NoCache', 'DiskCache'])
def test_loader_with_generator_connector(dataloader_builder, cacher, generator_connector):
    if cacher == 'NoCache':
        with pytest.raises(DataLoaderInitError):
            d: DataLoader = dataloader_builder(data_connector=generator_connector, cacher=cacher)
        return
    d: DataLoader = dataloader_builder(data_connector=generator_connector, cacher=cacher)
    df_all = pd.concat(generator_caller(), ignore_index=True)
    pd.testing.assert_frame_equal(d.load_all(), df_all)
    pd.testing.assert_frame_equal(d[:], df_all[:])
    pd.testing.assert_frame_equal(d[1:], df_all[1:])
    pd.testing.assert_frame_equal(d[:3], df_all[:3])
    pd.testing.assert_frame_equal(d[['a']], df_all[['a']])

def generator_caller():
    yield pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    yield pd.DataFrame({'a': [7, 8, 9], 'b': [10, 11, 12]})
    yield pd.DataFrame({'a': [13, 14, 15], 'b': [16, 17, 18]})

def test_generator_connector():
    c = GeneratorConnector(generator_caller)
    assert c._columns() == ['a', 'b']
    assert_frame_equal(c._read(), pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}))
    assert_frame_equal(c._read(offset=1), pd.DataFrame({'a': [7, 8, 9], 'b': [10, 11, 12]}))
    assert_frame_equal(c._read(offset=2), pd.DataFrame({'a': [13, 14, 15], 'b': [16, 17, 18]}))
    assert c._read(offset=3) is None
    assert_frame_equal(c._read(offset=0), pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}))
    assert_frame_equal(c._read(offset=1000), pd.DataFrame({'a': [7, 8, 9], 'b': [10, 11, 12]}))
    assert_frame_equal(c._read(offset=3), pd.DataFrame({'a': [13, 14, 15], 'b': [16, 17, 18]}))
    assert c._read(offset=5555) is None
    for d, g in zip(c.iter(), generator_caller()):
        assert_frame_equal(d, g)

