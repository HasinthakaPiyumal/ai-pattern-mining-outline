# Cluster 6

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

class PositiveNegativeFilter(Filter):
    """
    A data processor for filtering positive and negative values.

    This filter is used to ensure that values in specific columns remain positive or negative.
    During the reverse conversion process, rows that do not meet the expected positivity or
    negativity will be removed.

    Attributes:
        int_columns (set): A set of column names containing integer values.
        float_columns (set): A set of column names containing float values.
        positive_columns (set): A set of column names that should contain positive values.
        negative_columns (set): A set of column names that should contain negative values.
    """
    int_columns: set
    '\n    A set of column names that contain integer values.\n    '
    float_columns: set
    '\n    A set of column names that contain float values.\n    '
    positive_columns: set
    '\n    A set of column names that are identified as containing positive numeric values.\n    '
    negative_columns: set
    '\n    A set of column names that are identified as containing negative numeric values.\n    '

    def __init__(self):
        self.int_columns = set()
        self.float_columns = set()
        self.positive_columns = set()
        self.negative_columns = set()

    def fit(self, metadata: Metadata | None=None, **kwargs: dict[str, Any]):
        """
        Fit method for the data filter.
        """
        logger.info('PositiveNegativeFilter Fitted.')
        self.int_columns = metadata.int_columns
        self.float_columns = metadata.float_columns
        self.positive_columns = set(metadata.numeric_format['positive'])
        self.negative_columns = set(metadata.numeric_format['negative'])
        self.fitted = True

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert method for data filter (No Action).
        """
        logger.info('Converting data using PositiveNegativeFilter... Finished (No Action)')
        return raw_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse_convert method for the pos_neg data filter.

        Iterate through each row of data, check if there are negative values in positive_columns,
        or positive values in negative_columns. If the conditions are not met, discard the row.
        """
        logger.info(f'Data reverse-converted by PositiveNegativeFilter Start with Shape: {processed_data.shape}.')
        mask = pd.Series(True, index=processed_data.index)
        for col in self.positive_columns:
            if col in processed_data.columns and pd.api.types.is_numeric_dtype(processed_data[col]):
                mask &= processed_data[col] >= 0
        for col in self.negative_columns:
            if col in processed_data.columns and pd.api.types.is_numeric_dtype(processed_data[col]):
                mask &= processed_data[col] <= 0
        filtered_data = processed_data[mask]
        logger.info(f'Data reverse-converted by PositiveNegativeFilter with Output Shape: {filtered_data.shape}.')
        return filtered_data

class NumericValueTransformer(Transformer):
    """
    A transformer class for numeric data.

    This class is used to transform numeric data by scaling it using the StandardScaler from sklearn.

    Attributes:
        standard_scale (bool): A flag indicating whether to scale the data using StandardScaler.
        int_columns (Set): A set of column names that are of integer type.
        float_columns (Set): A set of column names that are of float type.
        scalers (Dict): A dictionary of scalers for each numeric column.
    """
    standard_scale: bool = True
    '\n    A flag indicating whether to scale the data using StandardScaler.\n    If True, the data will be scaled using StandardScaler.\n    If False, the data will not be scaled.\n    '
    int_columns: Set
    '\n    A set of column names that are of integer type.\n    These columns will be considered for scaling if `standard_scale` is True.\n    '
    float_columns: Set
    '\n    A set of column names that are of float type.\n    These columns will be considered for scaling if `standard_scale` is True.\n    '
    scalers: Dict
    '\n    A dictionary of scalers for each numeric column.\n    The keys are the column names and the values are the corresponding scalers.\n    '

    def __init__(self):
        self.int_columns = set()
        self.float_columns = set()
        self.scalers = {}

    def fit(self, metadata: Metadata | None=None, tabular_data: DataLoader | pd.DataFrame=None, **kwargs: dict[str, Any]):
        """
        The fit method.

        Data columns of int and float types need to be recorded here (Get data from metadata).
        """
        for each_col in metadata.int_columns:
            if each_col not in metadata.column_list:
                continue
            if metadata.get_column_data_type(each_col) == 'int':
                self.int_columns.add(each_col)
                continue
            if metadata.get_column_data_type(each_col) == 'id':
                self.int_columns.add(each_col)
        for each_col in metadata.float_columns:
            if each_col not in metadata.column_list:
                continue
            if metadata.get_column_data_type(each_col) == 'float':
                self.float_columns.add(each_col)
        if len(self.int_columns) == 0 and len(self.float_columns) == 0:
            logger.info('NumericValueTransformer Fitted (No numeric columns).')
            return
        for each_col in list(self.int_columns) + list(self.float_columns):
            self._fit_column(each_col, tabular_data[[each_col]])
        self.fitted = True
        logger.info('NumericValueTransformer Fitted.')

    def _fit_column(self, column_name: str, column_data: pd.DataFrame) -> np.ndarray:
        """
        Fit every numeric (include int and float) column in `_fit_column`.
        """
        if self.standard_scale:
            self._fit_column_scale(column_name, column_data)
            return
        return

    def _fit_column_scale(self, column_name: str, column_data: pd.DataFrame) -> np.ndarray:
        """
        Fit every numeric (include int and float) column using sklearn StandardScaler.
        """
        self.scalers[column_name] = StandardScaler()
        self.scalers[column_name].fit(column_data)

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert method to handle missing values in the input data.
        """
        logger.info('Converting data using NumericValueTransformer...')
        if len(self.int_columns) == 0 and len(self.float_columns) == 0:
            logger.info('Converting data using NumericValueTransformer... Finished (No column).')
            return
        processed_data = raw_data.copy()
        for each_col in list(self.int_columns) + list(self.float_columns):
            processed_col = self._covert_column(each_col, processed_data[[each_col]])
            processed_data[each_col] = processed_col
        logger.info('Converting data using NumericValueTransformer... Finished.')
        return processed_data

    def _covert_column(self, column_name: str, column_data: pd.DataFrame):
        """
        Convert every numeric (include int and float) column.
        """
        if self.standard_scale:
            return self._covert_column_scale(column_name=column_name, column_data=column_data)
        pass

    def _covert_column_scale(self, column_name: str, column_data: pd.DataFrame):
        """
        Convert every numeric (include int and float) column using sklearn StandardScaler.
        """
        scaled_data = self.scalers[column_name].transform(column_data)
        return scaled_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse convert method, convert generated data into processed data.
        """
        for each_col in list(self.int_columns) + list(self.float_columns):
            processed_col = self._reverse_convert_column(each_col, processed_data[[each_col]])
            processed_data[each_col] = processed_col
        logger.info('Data reverse-converted by NumericValueTransformer (No Action).')
        return processed_data

    def _reverse_convert_column(self, column_name: str, column_data: pd.DataFrame):
        """
        Reverse convert method for each column.
        """
        if self.standard_scale:
            return self._reverse_convert_column_scale(column_name=column_name, column_data=column_data)
        return

    def _reverse_convert_column_scale(self, column_name: str, column_data: pd.DataFrame):
        """
        Reverse convert method for input column using scale method.
        """
        reverse_converted_data = self.scalers[column_name].inverse_transform(column_data)
        return reverse_converted_data
    pass

class ConstValueTransformer(Transformer):
    """
    A transformer that replaces the input with a constant value.

    This class is used to transform any input data into a predefined constant value.
    It is particularly useful in scenarios where a consistent output is required regardless of the input.

    Attributes:
        const_value (dict[Any]): The constant value that will be returned.
    """
    const_columns: list
    const_values: dict[Any, Any]

    def __init__(self):
        self.const_columns = []
        self.const_values = {}

    def fit(self, metadata: Metadata | None=None, **kwargs: dict[str, Any]):
        """
        Fit method for the transformer.

        This method processes the metadata to identify columns that should be replaced with a constant value.
        It updates the internal state of the transformer with the columns and their corresponding constant values.

        Args:
            metadata (Metadata | None): The metadata object containing information about the columns and their data types.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            None
        """
        for each_col in metadata.column_list:
            if metadata.get_column_data_type(each_col) == 'const':
                self.const_columns.append(each_col)
        logger.info('ConstValueTransformer Fitted.')
        self.fitted = True

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert method to handle missing values in the input data by replacing specified columns with constant values.

        This method iterates over the columns identified for replacement with constant values and removes them from the input DataFrame.
        The removal is based on the columns specified during the fitting process.

        Args:
            raw_data (pd.DataFrame): The input DataFrame containing the data to be processed.

        Returns:
            pd.DataFrame: A DataFrame with the specified columns removed.
        """
        processed_data = copy.deepcopy(raw_data)
        logger.info('Converting data using ConstValueTransformer...')
        for each_col in self.const_columns:
            if each_col not in self.const_values.keys():
                self.const_values[each_col] = processed_data[each_col].unique()[0]
            processed_data = self.remove_columns(processed_data, [each_col])
        logger.info('Converting data using ConstValueTransformer... Finished.')
        return processed_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse_convert method for the transformer.

        This method restores the original columns that were replaced with constant values during the conversion process.
        It iterates over the columns identified for replacement with constant values and adds them back to the DataFrame
        with the predefined constant values.

        Args:
            processed_data (pd.DataFrame): The input DataFrame containing the processed data.

        Returns:
            pd.DataFrame: A DataFrame with the original columns restored, filled with their corresponding constant values.
        """
        df_length = processed_data.shape[0]
        for each_col_name in self.const_columns:
            each_value = self.const_values[each_col_name]
            each_const_col = [each_value for _ in range(df_length)]
            each_const_df = pd.DataFrame({each_col_name: each_const_col})
            processed_data = self.attach_columns(processed_data, each_const_df)
        logger.info('Data reverse-converted by ConstValueTransformer.')
        return processed_data

class FixedCombinationTransformer(Transformer):
    """
    A transformer that handles columns with fixed combinations in a DataFrame.

    This transformer goal to auto identifies and processes columns that have fixed relationships (high covariance) in
    a given DataFrame.

    The relationships between columns include:
      - Numerical function relationships: assess them based on covariance between the columns.
      - Categorical mapping relationships: check for duplicate values for each column.

    Note that we support one-to-one mappings between columns now, and each corresponding relationship will not
    include duplicate columns.

    For example:
    we detect that,
    1 numerical relationship: (key1, Value1, Value2)
    3 one-to-one relationships: (key1, Key2) , (Category1, Category2)

    | Key1 | Key2 | Category1 | Category2 | Value1 | Value2 |
    | :--: | :--: | :-------: | :-------: | :----: | :----: |
    |  1   |  A   |   1001   |   Apple   |   10   |   20   |
    |  2   |  B   |   1002   | Broccoli  |   15   |   30   |
    |  2   |  B   |   1001   |  Apple   |   20   |   20   |
    """
    fixed_combinations: dict[str, set[str]]
    '\n    A dictionary mapping column names to sets of column names that have fixed relationships with them.\n    '
    simplified_fixed_combinations: dict[str, set[str]]
    '\n    A dictionary mapping column names to sets of column names that have fixed relationships with them.\n    '
    column_mappings: dict[(str, str), dict[str, str]]
    '\n    A dictionary mapping tuples of column names to dictionaries of value mappings.\n    '
    is_been_specified: bool
    "\n    A boolean that flag if exist specific combinations by user.\n    If true, needn't running this auto detect transform.\n    "

    def __init__(self):
        super().__init__()
        self.fixed_combinations: dict[str, set[str]] = {}
        self.simplified_fixed_combinations: dict[str, set[str]] = {}
        self.column_mappings: dict[(str, str), dict[str, str]] = {}
        self.is_been_specified = False

    @property
    def is_exist_fixed_combinations(self) -> bool:
        """
        A boolean that flag if inspector have inspected some fixed combinations.
        If False, needn't running this auto detect transform.
        """
        return bool(self.fixed_combinations)

    def fit(self, metadata: Metadata | None=None, **kwargs: dict[str, Any]):
        """Fit the transformer and save the relationships between columns.

        Args:
            metadata (Metadata): Metadata object
        """
        if metadata.get('specific_combinations'):
            logger.info('Fit data using FixedCombinationTransformer(been specified)... Finished (No action).')
            self.is_been_specified = True
            self.fitted = True
            return
        self.fixed_combinations = metadata.get('fixed_combinations') or dict()
        if not self.is_exist_fixed_combinations:
            logger.info('Fit data using FixedCombinationTransformer(not existed)... Finished (No action).')
            self.fitted = True
            return
        simplified_fixed_combinations = {}
        seen = set()
        if not isinstance(self.fixed_combinations, dict):
            raise TypeError('fixed_combinations should be a dict, rather than {}'.format(type(self.fixed_combinations).__name__))
        for base_col, related_cols in self.fixed_combinations.items():
            combination = frozenset([base_col]) | frozenset(related_cols)
            if combination not in seen:
                simplified_fixed_combinations[base_col] = related_cols
                seen.add(combination)
        self.simplified_fixed_combinations = simplified_fixed_combinations
        self.has_column_mappings = False
        self.fitted = True

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Convert the input DataFrame by identifying and storing fixed column relationships.

        This method analyzes the relationships between columns specified in simplified_fixed_combinations
        and stores their value mappings. The mappings are only computed once for the first batch of data
        to optimize performance.

        NOTE:
            TODO-Enhance-Refactor Inspector by chain-of-responsibility, base one-to-one on Identified discrete_columns.
            The current implementation has space for optimization:
            - The column_mappings definition depends on the first batch of data from the DataLoader
            - This might miss some edge cases where column relationships are very comprehensive
              (e.g., some column correspondences might only appear in later batches)
            - While processing each batch separately could avoid this issue, it would incur
              significant performance overhead
            - The current function is sufficient for most scenarios
            - In the future, we may introduce parameters to control these strategies

        Args:
            raw_data (pd.DataFrame): The input DataFrame to be processed

        Returns:
            pd.DataFrame: The processed DataFrame (unchanged in this implementation)
        """
        if self.is_been_specified:
            logger.info('Converting data using FixedCombinationTransformer(been specified)... Finished (No action).')
            return raw_data
        if not self.is_exist_fixed_combinations:
            logger.info('Converting data using FixedCombinationTransformer(not existed)... Finished (No action).')
            return raw_data
        if self.has_column_mappings:
            logger.info('Converting data using FixedCombinationTransformer... Finished (No action).')
            return raw_data
        logger.info('Converting data using FixedCombinationTransformer... ')
        for base_col, related_cols in self.simplified_fixed_combinations.items():
            if base_col not in raw_data.columns:
                continue
            base_values = raw_data[base_col].unique()
            for related_col in related_cols:
                if related_col not in raw_data.columns:
                    continue
                value_mapping = {}
                for base_val in base_values:
                    related_vals = raw_data[raw_data[base_col] == base_val][related_col].unique()
                    if len(related_vals) == 1:
                        value_mapping[base_val] = related_vals[0]
                if value_mapping and (not any((pd.isna(v) for v in value_mapping.values()))):
                    self.column_mappings[base_col, related_col] = value_mapping
                    logger.debug(f'Saved mapping relationship between {base_col} and {related_col}')
        logger.info('Converting data using FixedCombinationTransformer... Finished.')
        self.has_column_mappings = True
        return raw_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverses the conversion process applied by the FixedCombinationTransformer.

        This method takes the processed DataFrame and uses the saved column mappings
        to restore the original values based on the relationships defined during the
        conversion process. If a base value does not have a corresponding related value,
        a random base value is selected to ensure that the DataFrame remains consistent.

        Args:
            processed_data (pd.DataFrame): The input DataFrame containing the processed data.

        Returns:
            pd.DataFrame: The DataFrame with original values restored based on the defined mappings.
        """
        if self.is_been_specified:
            logger.info('Reverse converting data using FixedCombinationTransformer(been specified)... Finished (No action).')
            return processed_data
        if not self.is_exist_fixed_combinations:
            logger.info('Reverse converting data using FixedCombinationTransformer(not existed)... Finished (No action).')
            return processed_data
        result_df = processed_data.copy()
        logger.info('Reverse converting data using FixedCombinationTransformer...')
        for (base_col, related_col), mapping in self.column_mappings.items():
            if base_col not in result_df.columns or related_col not in result_df.columns:
                continue

            def replace_row(row):
                base_val = row[base_col]
                if base_val in mapping:
                    new_related_val = mapping[base_val]
                    return pd.Series({base_col: base_val, related_col: new_related_val})
                else:
                    new_base_val = random.choice(list(mapping.keys()))
                    new_related_val = mapping[new_base_val]
                    return pd.Series({base_col: new_base_val, related_col: new_related_val})
            replaced = result_df.apply(replace_row, axis=1)
            result_df[base_col] = replaced[base_col]
            result_df[related_col] = replaced[related_col]
        logger.info('Reverse converting data using FixedCombinationTransformer... Finished.')
        return result_df

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

class SpecificCombinationTransformer(Transformer):
    """
    A transformer used to handle specific combinations of columns in tabular data.

    The relationships between columns can be quite complex. Currently, we introduced `FixedCombinationTransformer`
    is not capable of comprehensive automatic detection. This transformer allows users to manually specify the
    mapping relationships between columns, specifically for multiple corresponding relationships. Users can define
    multiple groups, with each group supporting multiple columns. The transformer will record the combination values
    of each column, and in the `reverse_convert()`, it will restore any mismatched combinations from the recorded
    relationships.

    For example:

    | Category A | Category B | Category C | Category D | Category E |
    | :--------: | :--------: | :--------: | :--------: | :--------: |
    |     A1     |     B1     |     C1     |     D1     |     E1     |
    |     A1     |     B1     |     C2     |     D2     |     E2     |
    |     A2     |     B2     |     C1     |     D1     |     E3     |

    Here user can specific combination like (Category A, Category B), (Category C, Category D, Category E).

    For now, the `specific_combinations` passing by `Metadata`

    """
    column_groups: List[Set[str]]
    '\n    Define a list where each element is a set containing string type column names\n    '
    mappings: Dict[frozenset, pd.DataFrame]
    '\n    Define a dictionary variable `mappings` where the keys are frozensets and the values are pandas DataFrame objects\n    '
    specified: bool
    '\n    Define a boolean that flag if user specified the combination, if true, that handle the `specific_combinations`\n    '

    def __init__(self):
        self.column_groups: List[Set[str]] = []
        self.mappings: Dict[frozenset, pd.DataFrame] = {}
        self.specified = False

    def fit(self, metadata: Metadata | None=None, tabular_data: DataLoader | pd.DataFrame=None):
        """
        Study the combination relationships and value mapping of columns.

        Args:
            metadata: Metadata containing information about specific column combinations.
            tabular_data: The tabular data to be fitted, can be a DataLoader object or a pandas DataFrame.
        """
        specific_combinations = metadata.get('specific_combinations')
        if specific_combinations is None or len(specific_combinations) == 0:
            logger.info('Fit data using SpecificCombinationTransformer(No specified)... Finished (No action).')
            self.fitted = True
            return
        df = tabular_data
        self.column_groups = [set(cols) for cols in specific_combinations]
        for group in self.column_groups:
            group_df = df[list(group)].drop_duplicates()
            self.mappings[frozenset(group)] = group_df
        self.fitted = True
        self.specified = True
        logger.info('SpecificCombinationTransformer Fitted.')

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the raw data based on the learned mapping relationships.

        Args:
           raw_data: The raw data to be converted.

        Returns:
           The converted data.
        """
        if not self.specified:
            logger.info('Converting data using SpecificCombinationTransformer(No specified)... Finished (No action).')
            return super().convert(raw_data)
        logger.info('SpecificCombinationTransformer convert doing nothing...')
        return super().convert(raw_data)

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse convert the processed data to ensure it conforms to the original format.

        Args:
            processed_data: The processed data to be reverse converted.

        Returns:
            The reverse converted data.
        """
        if not self.specified:
            logger.info('Reverse converting data using SpecificCombinationTransformer(No specified)... Finished (No action).')
            return processed_data
        result_df = processed_data.copy()
        n_rows = len(result_df)
        for group in self.column_groups:
            group_mapping = self.mappings[frozenset(group)]
            group_cols = list(group)
            random_indices = np.random.choice(len(group_mapping), size=n_rows)
            random_mappings = group_mapping.iloc[random_indices]
            result_df[group_cols] = random_mappings[group_cols].values
        return result_df

class OutlierTransformer(Transformer):
    """
    A transformer class to handle outliers in the data by converting them to specified fill values.

    Attributes:
        int_columns (set): A set of column names that contain integer values.
        int_outlier_fill_value (int): The value to fill in for outliers in integer columns. Default is 0.
        float_columns (set): A set of column names that contain float values.
        float_outlier_fill_value (float): The value to fill in for outliers in float columns. Default is 0.
    """
    int_columns: set
    '\n    set: A set of column names that contain integer values. These columns will have their outliers replaced by `int_outlier_fill_value`.\n    '
    int_outlier_fill_value: int
    '\n    int: The value to fill in for outliers in integer columns. Default is 0.\n    '
    float_columns: set
    '\n    set: A set of column names that contain float values. These columns will have their outliers replaced by `float_outlier_fill_value`.\n    '
    float_outlier_fill_value: float
    '\n    float: The value to fill in for outliers in float columns. Default is 0.\n    '

    def __init__(self):
        self.int_columns = set()
        self.int_outlier_fill_value = 0
        self.float_columns = set()
        self.float_outlier_fill_value = float(0)

    def fit(self, metadata: Metadata | None=None, **kwargs: dict[str, Any]):
        """
        Fit method for the transformer.

        Records the names of integer and float columns from the metadata.

        Args:
            metadata (Metadata | None): The metadata object containing column type information.
            **kwargs: Additional keyword arguments.
        """
        for each_col in metadata.int_columns:
            if each_col not in metadata.column_list:
                continue
            if metadata.get_column_data_type(each_col) == 'int':
                self.int_columns.add(each_col)
        for each_col in metadata.float_columns:
            if each_col not in metadata.column_list:
                continue
            if metadata.get_column_data_type(each_col) == 'float':
                self.float_columns.add(each_col)
        self.fitted = True
        logger.info('OutlierTransformer Fitted.')

    def convert(self, raw_data: DataFrame) -> DataFrame:
        """
        Convert method to handle outliers in the input data by replacing them with specified fill values.

        Args:
            raw_data (DataFrame): The input DataFrame containing the data to be processed.

        Returns:
            DataFrame: The processed DataFrame with outliers replaced by fill values.
        """
        res = raw_data
        logger.info('Converting data using OutlierTransformer...')

        def convert_to_int(value):
            try:
                return int(value)
            except ValueError:
                return self.int_outlier_fill_value
        for each_col in self.int_columns:
            res[each_col] = res[each_col].apply(convert_to_int)

        def convert_to_float(value):
            try:
                return float(value)
            except ValueError:
                return self.float_outlier_fill_value
        for each_col in self.float_columns:
            res[each_col] = res[each_col].apply(convert_to_float)
        logger.info('Converting data using OutlierTransformer... Finished.')
        return res

    def reverse_convert(self, processed_data: DataFrame) -> DataFrame:
        """
        Reverse_convert method for the transformer (No action for OutlierTransformer).

        Args:
            processed_data (DataFrame): The processed DataFrame.

        Returns:
            DataFrame: The same processed DataFrame.
        """
        logger.info('Data reverse-converted by OutlierTransformer (No Action).')
        return processed_data

class NonValueTransformer(Transformer):
    """
    A transformer class designed to handle missing values in a DataFrame. It can either drop rows with missing values or fill them with specified values.

    Attributes:
        int_columns (set): A set of column names that contain integer values.
        float_columns (set): A set of column names that contain float values.
        column_list (list): A list of all column names in the DataFrame.
        fill_na_value_int (int): The value to fill missing integer values with. Default is 0.
        fill_na_value_float (float): The value to fill missing float values with. Default is 0.0.
        fill_na_value_default (str): The value to fill missing values for non-numeric columns with. Default is 'NAN_VALUE'.
        drop_na (bool): A flag indicating whether to drop rows with missing values. If True, rows with missing values are dropped. If False, missing values are filled with specified values. Default is False.
    """
    int_columns: set
    '\n    A set of column names that contain integer values.\n    '
    float_columns: set
    '\n    A set of column names that contain float values.\n    '
    column_list: list
    '\n    A list of all column names in the DataFrame.\n    '
    fill_na_value_int: int
    '\n    The value to fill missing integer values with. Default is 0.\n    '
    fill_na_value_float: float
    '\n    The value to fill missing float values with. Default is 0.0.\n    '
    fill_na_value_default: str
    "\n    The value to fill missing values for non-numeric columns with. Default is 'NAN_VALUE'.\n    "
    drop_na: bool
    '\n    A boolean flag indicating whether to drop rows with missing values or fill them with `fill_na_value`.\n\n    If `True`, rows with missing values will be dropped.\n    If `False`, missing values will be filled with `fill_na_value`.\n\n    Currently, the default setting is False, which means rows with missing values are not dropped.\n    '

    def __init__(self):
        self.int_columns = set()
        self.float_columns = set()
        self.column_list = []
        self.fill_na_value_int = 0
        self.fill_na_value_float = 0.0
        self.fill_na_value_default = 'NAN_VALUE'
        self.drop_na = False

    def fit(self, metadata: Metadata | None=None, **kwargs: dict[str, Any]):
        """
        Fit method for the transformer.
        """
        logger.info('NonValueTransformer Fitted.')
        for key, value in kwargs.items():
            if key == 'drop_na':
                if not isinstance(value, str):
                    raise ValueError('fill_na_value must be of type <str>')
                self.drop_na = value
        for each_col in metadata.int_columns:
            if each_col not in metadata.column_list:
                continue
            if metadata.get_column_data_type(each_col) == 'int':
                self.int_columns.add(each_col)
        logger.info(f'NonValueTransformer get int columns: {self.int_columns}.')
        for each_col in metadata.float_columns:
            if each_col not in metadata.column_list:
                continue
            if metadata.get_column_data_type(each_col) == 'float':
                self.float_columns.add(each_col)
        logger.info(f'NonValueTransformer get float columns: {self.float_columns}.')
        self.column_list = metadata.column_list
        logger.info(f'NonValueTransformer get column list from metadata: {self.column_list}.')
        self.fitted = True

    def convert(self, raw_data: DataFrame) -> DataFrame:
        """
        Convert method to handle missing values in the input data.
        """
        logger.info('Converting data using NonValueTransformer...')
        if self.drop_na:
            logger.info('Converting data using NonValueTransformer... Finished (Drop NA).')
            return raw_data.dropna()
        res = raw_data
        for each_col in self.int_columns:
            res[each_col] = res[each_col].fillna(self.fill_na_value_int)
        for each_col in self.float_columns:
            res[each_col] = res[each_col].fillna(self.fill_na_value_float)
        for each_col in self.column_list:
            if each_col in self.int_columns or each_col in self.float_columns:
                continue
            res[each_col] = res[each_col].fillna(self.fill_na_value_default)
        logger.info('Converting data using NonValueTransformer... Finished.')
        return res

    def reverse_convert(self, processed_data: DataFrame) -> DataFrame:
        """
        Reverse_convert method for the transformer.

        Does not require any action.
        """

        def replace_nan_value(df):
            """
            Scans all rows and columns in the DataFrame and replaces all cells with the value "NAN_VALUE", which is self.fill_na_value_default, with an empty string.

            Parameters:
            df (pd.DataFrame): The input DataFrame.

            Returns:
            pd.DataFrame: The DataFrame after replacement.
            """
            df_replaced = df.replace(self.fill_na_value_default, '')
            return df_replaced
        logger.info('Data reverse-converted by NonValueTransformer.')
        return replace_nan_value(processed_data)
    pass

class EmptyTransformer(Transformer):
    """
    A transformer that handles empty columns in a DataFrame.

    This transformer identifies and processes columns that contain no data (empty columns) in a given DataFrame.
    It can remove these columns during the conversion process and restore them during the reverse conversion process.

    Attributes:
        empty_columns (list): A list of column names that are identified as empty.

    Methods:
        fit(metadata: Metadata | None = None, **kwargs: dict[str, Any]):
            Fits the transformer to the data by identifying empty columns based on provided metadata.
        convert(raw_data: pd.DataFrame) -> pd.DataFrame:
            Converts the raw data by removing the identified empty columns.
        reverse_convert(processed_data: pd.DataFrame) -> pd.DataFrame:
            Reverses the conversion by restoring the previously removed empty columns.
    """
    empty_columns: set
    '\n    Set of column names that are identified as empty. This attribute is populated during the fitting process\n    and is used to remove these columns during the conversion process and restore them during the reverse conversion process.\n    '

    def __init__(self):
        self.empty_columns = set()

    def fit(self, metadata: Metadata | None=None, **kwargs: dict[str, Any]):
        """
        Fit method for the transformer.
        Remember the empty_columns from all columns.

        Args:
            metadata (Metadata | None): The metadata containing information about the data, including empty columns.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            None
        """
        for each_col in metadata.get('empty_columns'):
            if metadata.get_column_data_type(each_col) == 'empty':
                self.empty_columns.add(each_col)
        logger.info('EmptyTransformer Fitted.')
        self.fitted = True
        return

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the raw data by removing the identified empty columns.

        Args:
            raw_data (pd.DataFrame): The input DataFrame containing the raw data.

        Returns:
            pd.DataFrame: The processed DataFrame with empty columns removed.
        """
        processed_data = raw_data
        logger.info('Converting data using EmptyTransformer...')
        for each_col in self.empty_columns:
            processed_data = self.remove_columns(processed_data, [each_col])
        logger.info('Converting data using EmptyTransformer... Finished (No action).')
        return processed_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverses the conversion by restoring the previously removed empty columns.

        Args:
            processed_data (pd.DataFrame): The input DataFrame containing the processed data.

        Returns:
            pd.DataFrame: The DataFrame with previously removed empty columns restored.
        """
        if not self.fitted or not self.empty_columns:
            return processed_data
        for col_name in self.empty_columns:
            empty_df = pd.DataFrame({col_name: [None] * len(processed_data)})
            processed_data = self.attach_columns(processed_data, empty_df)
        return processed_data

class ColumnOrderTransformer(Transformer):
    """
    A transformer that rearranges the columns of a DataFrame to a specified order.

    Attributes:
        column_list (list): The list of column names in the desired order.

    Methods:
        fit(metadata: Metadata | None = None, **kwargs: dict[str, Any]): Fits the transformer by remembering the order of the columns.
        convert(raw_data: pd.DataFrame) -> pd.DataFrame: Converts the input DataFrame by rearranging its columns.
        reverse_convert(processed_data: pd.DataFrame) -> pd.DataFrame: Reverse-converts the processed DataFrame by rearranging its columns back to their original order.
        rearrange_columns(column_list, processed_data): Rearranges the columns of a DataFrame according to the provided column list.
    """
    column_list: list
    "\n    The list of tabular data's columns.\n    "

    def __init__(self):
        self.column_list = None

    def fit(self, metadata: Metadata | None=None, **kwargs: dict[str, Any]):
        """
        Fit method for the transformer.

        Remember the order of the columns.
        """
        self.column_list = list(metadata.column_list)
        logger.info('ColumnOrderTransformer Fitted.')
        self.fitted = True
        return

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert method to handle missing values in the input data.
        """
        logger.info('Converting data using ColumnOrderTransformer...')
        logger.info('Converting data using ColumnOrderTransformer... Finished (No action).')
        return raw_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse_convert method for the transformer.
        """
        res = self.rearrange_columns(self.column_list, processed_data)
        logger.info('Data reverse-converted by ColumnOrderTransformer.')
        return res

    @staticmethod
    def rearrange_columns(column_list, processed_data):
        """
        This method rearranges the columns of a given DataFrame according to the provided column list.

        Any columns in the DataFrame that are not in the column list are dropped.

        Args:
            - column_list (list): A list of column names in the order they should appear in the output DataFrame.
            - processed_data (pd.DataFrame): The DataFrame to be rearranged.

        Returns:
            - result_data (pd.DataFrame): The rearranged DataFrame.
        """
        result_data = processed_data.reindex(columns=column_list)
        return result_data

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

class IntValueFormatter(Formatter):
    """
    Formatter class for handling Int values in pd.DataFrame.
    """
    int_columns: set
    '\n    List of column names that are of type int, populated by the fit method using metadata.\n    '

    def __init__(self):
        self.int_columns = set()

    def fit(self, metadata: Metadata | None=None, **kwargs: dict[str, Any]):
        """
        Fit method for the formatter.

        Formatter need to use metadata to record which columns belong to the int type, and convert them back to the int type during post-processing.
        """
        for each_col in metadata.int_columns:
            if each_col not in metadata.column_list:
                continue
            if metadata.get_column_data_type(each_col) == 'int':
                self.int_columns.add(each_col)
                continue
            if metadata.get_column_data_type(each_col) == 'id':
                self.int_columns.add(each_col)
        logger.info('IntValueFormatter Fitted.')
        self.fitted = True
        return

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        No action for convert.
        """
        logger.info('Converting data using IntValueFormatter... Finished  (No Action).')
        return raw_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        reverse_convert method for the formatter.

        Do format conversion for int columns.
        """
        for col in self.int_columns:
            if col in processed_data.columns:
                processed_data[col] = processed_data[col].astype(int)
            else:
                logger.error('Column {} not found in processed_data.'.format(col))
        logger.info('Data reverse-converted by IntValueFormatter.')
        return processed_data

class MetadataCombiner(BaseModel):
    """
    Combine different tables with relationship, used for describing the relationship between tables.

    Args:
        version (str): version
        named_metadata (Dict[str, Any]): pairs of table name and metadata
        relationships (List[Any]): list of relationships
    """
    version: str = '1.0'
    named_metadata: Dict[str, Metadata] = {}
    relationships: List[Relationship] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def check(self):
        """Do necessary checks:

        - Whether number of tables corresponds to relationships.
        - Whether table names corresponds to the relationship between tables;
        """
        for m in self.named_metadata.values():
            m.check()
        table_names = set(self.named_metadata.keys())
        relationship_parents = set((r.parent_table for r in self.relationships))
        relationship_children = set((r.child_table for r in self.relationships))
        if not table_names.issuperset(relationship_parents):
            raise MetadataCombinerInvalidError(f"Relationships' parent table {relationship_parents - table_names} is missing.")
        if not table_names.issuperset(relationship_children):
            raise MetadataCombinerInvalidError(f"Relationships' child table {relationship_children - table_names} is missing.")
        if not (relationship_parents | relationship_children).issuperset(table_names):
            raise MetadataCombinerInvalidError(f'Table {table_names - (relationship_parents + relationship_children)} is missing in relationships.')
        logger.info('MultiTableCombiner check finished.')

    @classmethod
    def from_dataloader(cls, dataloaders: list[DataLoader], metadata_from_dataloader_kwargs: None | dict=None, relationshipe_inspector: None | str | type[Inspector]='SubsetRelationshipInspector', relationships_inspector_kwargs: None | dict=None, relationships: None | list[Relationship]=None):
        """
        Combine multiple dataloaders with relationship.

        Args:
            dataloaders (list[DataLoader]): list of dataloaders
            max_chunk (int): max chunk count for relationship inspector.
            metadata_from_dataloader_kwargs (dict): kwargs for :func:`Metadata.from_dataloader`
            relationshipe_inspector (str | type[Inspector]): relationship inspector
            relationships_inspector_kwargs (dict): kwargs for :func:`InspectorManager.init`
            relationships (list[Relationship]): list of relationships
        """
        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]
        metadata_from_dataloader_kwargs = metadata_from_dataloader_kwargs or {}
        named_metadata = {d.identity: Metadata.from_dataloader(d, **metadata_from_dataloader_kwargs) for d in dataloaders}
        if relationships is None and relationshipe_inspector is not None:
            if relationships_inspector_kwargs is None:
                relationships_inspector_kwargs = {}
            inspector = InspectorManager().init(relationshipe_inspector, **relationships_inspector_kwargs)
            for d in dataloaders:
                for chunk in d.iter():
                    inspector.fit(chunk, name=d.identity, metadata=named_metadata[d.identity])
            relationships = inspector.inspect()['relationships']
        return cls(named_metadata=named_metadata, relationships=relationships)

    @classmethod
    def from_dataframe(cls, dataframes: list[pd.DataFrame], names: list[str], metadata_from_dataloader_kwargs: None | dict=None, relationshipe_inspector: None | str | type[Inspector]='SubsetRelationshipInspector', relationships_inspector_kwargs: None | dict=None, relationships: None | list[Relationship]=None) -> 'MetadataCombiner':
        """
        Combine multiple dataframes with relationship.

        Args:
            dataframes (list[pd.DataFrame]): list of dataframes
            names (list[str]): list of names
            metadata_from_dataloader_kwargs (dict): kwargs for :func:`Metadata.from_dataloader`
            relationshipe_inspector (str | type[Inspector]): relationship inspector
            relationships_inspector_kwargs (dict): kwargs for :func:`InspectorManager.init`
            relationships (list[Relationship]): list of relationships
        """
        if not isinstance(dataframes, list):
            dataframes = [dataframes]
        if not isinstance(names, list):
            names = [names]
        metadata_from_dataloader_kwargs = metadata_from_dataloader_kwargs or {}
        if len(dataframes) != len(names):
            raise MetadataCombinerInitError('dataframes and names should have same length.')
        named_metadata = {n: Metadata.from_dataframe(d, **metadata_from_dataloader_kwargs) for n, d in zip(names, dataframes)}
        if relationships is None and relationshipe_inspector is not None:
            if relationships_inspector_kwargs is None:
                relationships_inspector_kwargs = {}
            inspector = InspectorManager().init(relationshipe_inspector, **relationships_inspector_kwargs)
            for n, d in zip(names, dataframes):
                inspector.fit(d, name=n, metadata=named_metadata[n])
            relationships = inspector.inspect()['relationships']
        return cls(named_metadata=named_metadata, relationships=relationships)

    def _dump_json(self):
        return self.model_dump_json()

    def save(self, save_dir: str | Path, metadata_subdir: str='metadata', relationship_subdir: str='relationship'):
        """
        Save metadata to json file.

        This will create several subdirectories for metadata and relationship.

        Args:
            save_dir (str | Path): directory to save
            metadata_subdir (str): subdirectory for metadata, default is "metadata"
            relationship_subdir (str): subdirectory for relationship, default is "relationship"
        """
        save_dir = Path(save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        version_file = save_dir / 'version'
        version_file.write_text(self.version)
        metadata_subdir = save_dir / metadata_subdir
        relationship_subdir = save_dir / relationship_subdir
        metadata_subdir.mkdir(parents=True, exist_ok=True)
        for name, metadata in self.named_metadata.items():
            metadata.save(metadata_subdir / f'{name}.json')
        relationship_subdir.mkdir(parents=True, exist_ok=True)
        for relationship in self.relationships:
            relationship.save(relationship_subdir / f'{relationship.parent_table}_{relationship.child_table}.json')

    @classmethod
    def load(cls, save_dir: str | Path, metadata_subdir: str='metadata', relationship_subdir: str='relationship', version: None | str=None) -> 'MetadataCombiner':
        """
        Load metadata from json file.

        Args:
            save_dir (str | Path): directory to save
            metadata_subdir (str): subdirectory for metadata, default is "metadata"
            relationship_subdir (str): subdirectory for relationship, default is "relationship"
            version (str): Manual version, if not specified, try to load from version file
        """
        save_dir = Path(save_dir).expanduser().resolve()
        if not version:
            logger.debug('No version specified, try to load from version file.')
            version_file = save_dir / 'version'
            if version_file.exists():
                version = version_file.read_text().strip()
            else:
                logger.info('No version file found, assume version is 1.0')
                version = '1.0'
        named_metadata = {p.stem: Metadata.load(p) for p in (save_dir / metadata_subdir).glob('*')}
        relationships = [Relationship.load(p) for p in (save_dir / relationship_subdir).glob('*')]
        cls.upgrade(version, named_metadata, relationships)
        return cls(version=version, named_metadata=named_metadata, relationships=relationships)

    @classmethod
    def upgrade(cls, old_version: str, named_metadata: dict[str, Metadata], relationships: list[Relationship]) -> None:
        """
        Upgrade metadata from old version to new version

        :ref:`Metadata.upgrade` and :ref:`Relationship.upgrade` will try upgrade when loading.
        So here we just do Combiner's upgrade.
        """
        pass

    @property
    def fields(self) -> Iterable[str]:
        """
        Return all fields in MetadataCombiner.
        """
        return chain((k for k in self.model_fields if k.endswith('_columns')))

    def __eq__(self, other):
        if not isinstance(other, MetadataCombiner):
            return super().__eq__(other)
        return self.version == other.version and all((self.get(key) == other.get(key) for key in set(chain(self.fields, other.fields)))) and (set(self.fields) == set(other.fields))

class CTGANSynthesizerModel(MLSynthesizerModel, BatchedSynthesizer):
    """
    Modified from ``sdgx.models.components.sdv_ctgan.synthesizers.ctgan.CTGANSynthesizer``.
    A CTGANSynthesizer but provided :ref:`SynthesizerModel` interface with chunked fit.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.


    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        device (str):
            Device to run the training on. Preferred to be 'cuda' for GPU if available.
    """
    MODEL_SAVE_NAME = 'ctgan.pkl'

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256), generator_lr=0.0002, generator_decay=1e-06, discriminator_lr=0.0002, discriminator_decay=1e-06, batch_size=500, discriminator_steps=1, log_frequency=True, epochs=300, pac=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
        assert batch_size % 2 == 0
        BatchedSynthesizer.__init__(self, batch_size=batch_size)
        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim
        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._epochs = epochs
        self.pac = pac
        self._device = torch.device(device)
        self._transformer: Optional[DataTransformer] = None
        self._data_sampler: Optional[DataSampler] = None
        self._generator = None
        self._ndarry_loader: Optional[NDArrayLoader] = None
        self.data_dim: Optional[int] = None

    def fit(self, metadata: Metadata, dataloader: DataLoader, epochs=None, *args, **kwargs):
        discrete_columns = list(metadata.get('discrete_columns'))
        if epochs is not None:
            self._epochs = epochs
        self._pre_fit(dataloader, discrete_columns, metadata)
        if self.fit_data_empty:
            logger.info('CTGAN fit finished because of empty df detected.')
            return
        logger.info('CTGAN prefit finished, start CTGAN training.')
        self._fit(len(self._ndarry_loader))
        logger.info('CTGAN training finished.')

    def _pre_fit(self, dataloader: DataLoader, discrete_columns: list[str]=None, metadata: Metadata=None):
        if not discrete_columns:
            discrete_columns = []
        discrete_columns = self._filter_discrete_columns(dataloader.columns(), discrete_columns)
        if self.fit_data_empty:
            return
        self._transformer = DataTransformer(metadata=metadata)
        logger.info("Fitting model's transformer...")
        self._transformer.fit(dataloader, discrete_columns)
        logger.info('Transforming data...')
        self._ndarry_loader = self._transformer.transform(dataloader)
        logger.info('Sampling data.')
        self._data_sampler = DataSampler(self._ndarry_loader, self._transformer.output_info_list, self._log_frequency)
        logger.info('Initialize Generator.')
        self.data_dim = self._transformer.output_dimensions
        self._generator = Generator(self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, self.data_dim).to(self._device)

    @random_state
    def _fit(self, data_size: int):
        """Fit the CTGAN Synthesizer models to the training data."""
        logger.info(f'Fit using data_size:{data_size}, data_dim: {self.data_dim}.')
        epochs = self._epochs
        discriminator = Discriminator(self.data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=self.pac).to(self._device)
        optimizerG = optim.Adam(self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9), weight_decay=self._generator_decay)
        optimizerD = optim.Adam(discriminator.parameters(), lr=self._discriminator_lr, betas=(0.5, 0.9), weight_decay=self._discriminator_decay)
        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1
        logger.info('Starting model training, epochs: {}'.format(epochs))
        steps_per_epoch = max(data_size // self._batch_size, 1)
        for i in range(epochs):
            start_time = time.time()
            for id_ in tqdm.tqdm(range(steps_per_epoch), desc='Fitting batches', delay=3):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)
                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = (None, None, None, None)
                        real = self._data_sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)
                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]
                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)
                    real = torch.from_numpy(real.astype('float32')).to(self._device)
                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact
                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)
                    pen = discriminator.calc_gradient_penalty(real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()
                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                if condvec is None:
                    c1, m1, col, opt = (None, None, None, None)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)
                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)
                loss_g = -torch.mean(y_fake) + cross_entropy
                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()
            logger.info(f'Epoch {i + 1}, Loss G: {loss_g.detach().cpu(): .4f}, Loss D: {loss_d.detach().cpu(): .4f}, Time: {time.time() - start_time: .4f}')

    def sample(self, count: int, *args, **kwargs) -> pd.DataFrame:
        if self.fit_data_empty:
            return pd.DataFrame(index=range(count))
        return self._sample(count, *args, **kwargs)

    @random_state
    def _sample(self, n, condition_column=None, condition_value=None, drop_more=True):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(condition_info, self._batch_size)
        else:
            global_condition_vec = None
        steps = math.ceil(n / self._batch_size)
        data = []
        for _ in tqdm.tqdm(range(steps), desc='Sampling batches', delay=3):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)
            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)
            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)
            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())
        data = np.concatenate(data, axis=0)
        logger.info('CTGAN Generated {} raw samples.'.format(data.shape[0]))
        if drop_more:
            data = data[:n]
        return self._transformer.inverse_transform(data)

    def save(self, save_dir: str | Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        return SDVBaseSynthesizer.save(self, save_dir / self.MODEL_SAVE_NAME)

    @classmethod
    def load(cls, save_dir: str | Path, device: str=None) -> 'CTGANSynthesizerModel':
        return SDVBaseSynthesizer.load(save_dir / cls.MODEL_SAVE_NAME, device)

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse('1.2.0'):
            for i in range(10):
                transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError('gumbel_softmax returning NaN.')
        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                elif span_info.activation_fn == 'linear':
                    ed = st + span_info.dim
                    transformed = data[:, st:ed].clone()
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')
        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction='none')
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c
        loss = torch.stack(loss, dim=1)
        return (loss * m).sum() / data.size()[0]

    def _filter_discrete_columns(self, train_data: List[str], discrete_columns: List[str]):
        """
        We filter PII Column here, which PII would only be discrete for now.
        As PII would be generating from PII Generator which not synthetic from model.

        Besides we need to figure it out when to stop model fitting:
        The original data consists entirely of discrete column data, and all of this discrete column data is PII.

        For `train_data`, there are three possibilities for the columns type.
         - train_data = valid_discrete + valid_continue
         - train_data = valid_continue
         - train_data = valid_discrete

        For `discrete_columns`, discrete_columns = invalid_discrete(PII) + valid_discrete

        Thus, valid_discrete = discrete_columns - invalid_discrete
                             = discrete_columns - Set.intersection(train_data, discrete_columns)

        Thus, original_data_is_all_PII: discrete_columns is not empty & train_data is empty
        """
        if len(discrete_columns) == 0:
            return discrete_columns
        if len(train_data) == 0:
            self.fit_data_empty = True
            return discrete_columns
        invalid_columns = set(discrete_columns) - set(train_data)
        return set(discrete_columns) - set(invalid_columns)

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame or list):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        elif isinstance(train_data, list):
            invalid_columns = set(discrete_columns) - set(train_data)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')
        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)

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

class SingleTableGPTModel(LLMBaseModel):
    """
    This is a synthetic data generation model powered by OpenAI GPT, a state-of-the-art language model. This model is based on groundbreaking research presented in the ICLR paper titled "Language Models are Realistic Tabular Data Generators".

    Our model harnesses the power of GPT to generate synthetic tabular data that closely resembles real-world datasets. By utilizing the advanced capabilities of GPT, we aim to provide a reliable and efficient solution for generating simulated data that can be used for various purposes, such as testing, training, and analysis.

    With this synthetic data generation model, users can easily generate diverse and realistic tabular datasets, mimicking the characteristics and patterns found in real data.
    """
    openai_API_key = ''
    '\n    The API key required to access the OpenAI GPT model. Please provide your own API key for authentication.\n    '
    openai_API_url = 'https://api.openai.com/v1/'
    '\n    The URL endpoint for the OpenAI GPT API. Please specify the appropriate URL for accessing the API.\n    '
    max_tokens = 4000
    '\n    The maximum number of tokens allowed in the generated response. This parameter helps in limiting the length of the output text.\n    '
    temperature = 0.1
    '\n    A parameter that controls the randomness of the generated text. Lower values like 0.1 make the output more focused and deterministic, while higher values like 1.0 introduce more randomness.\n    '
    timeout = 90
    '\n    The maximum time (in seconds) to wait for a response from the OpenAI GPT API. If the response is not received within this time, the request will be timed out.\n    '
    gpt_model = 'gpt-3.5-turbo'
    '\n    The specific GPT model to be used for generating text. The default model is "gpt-3.5-turbo", which is known for its high performance and versatility.\n    '
    query_batch = 30
    '\n    This parameter is the number of samples submitted to GPT each time and the number of returned samples.\n\n    This size has a certain relationship with the max_token parameter.\n\n    We do not recommend setting too large a value, as this may cause potential problems or errors.\n    '
    _sample_lines = []
    '\n    A list to store the sample lines of generated data.\n    '
    _result_list = []
    '\n    A list to store the generated data samples.\n    '

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the class instance.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._get_openai_setting_from_env()

    def check(self):
        """
        Performs various checks.

        Raises:
            SynthesizerInitError: If data access type is not specified or if duplicate data access type is found.
        """
        self._check_openAI_setting()
        self._set_openAI()
        self._check_access_type()

    def set_openAI_settings(self, API_url='https://api.openai.com/v1/', API_key=''):
        """
        Sets the OpenAI settings.

        Args:
            API_url (str): The OpenAI API URL. Defaults to "https://api.openai.com/v1/".
            API_key (str): The OpenAI API key. Defaults to an empty string.
        """
        self.openai_API_url = API_url
        self.openai_API_key = API_key
        self._set_openAI()

    def _set_openAI(self):
        """
        Sets the OpenAI API key and base URL.
        """
        openai.api_key = self.openai_API_key
        openai.base_url = self.openai_API_url

    def _check_openAI_setting(self):
        """
        Checks if the OpenAI settings are properly initialized.

        Raises:
            InitializationError: If openai_API_url or openai_API_key is not found.
        """
        if not self.openai_API_url:
            raise InitializationError('openai_API_url NOT found.')
        if not self.openai_API_key:
            raise InitializationError('openai_API_key NOT found.')
        logger.debug('OpenAI setting check passed.')

    def _get_openai_setting_from_env(self):
        """
        Retrieves OpenAI settings from environment variables.
        """
        if os.getenv('OPENAI_KEY'):
            self.openai_API_key = os.getenv('OPENAI_KEY')
            logger.debug('Get OPENAI_KEY from ENV.')
        if os.getenv('OPENAI_URL'):
            self.openai_API_url = os.getenv('OPENAI_URL')
            logger.debug('Get OPENAI_URL from ENV.')

    def openai_client(self):
        """
        Generate a openai request client.
        """
        return openai.OpenAI(api_key=self.openai_API_key, base_url=self.openai_API_url)

    def ask_gpt(self, question, model=None):
        """
        Sends a question to the GPT model.

        Args:
            question (str): The question to ask.
            model (str): The GPT model to use. Defaults to None.

        Returns:
            str: The response from the GPT model.

        Raises:
            SynthesizerInitError: If the check method fails.
        """
        self.check()
        if model:
            model = model
        else:
            model = self.gpt_model
        client = self.openai_client()
        logger.info(f'Ask GPT with temperature = {self.temperature}.')
        response = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': question}], temperature=self.temperature, max_tokens=self.max_tokens, timeout=self.timeout)
        logger.info('Ask GPT Finished.')
        self._responses.append(response)
        return response.choices[0].message.content

    def fit(self, raw_data: pd.DataFrame | DataLoader=None, metadata: Metadata=None, *args, **kwargs):
        """
        Fits this model to the provided data.
        Please note that no actual algorithmic training is excuted here.

        Args:
            raw_data (pd.DataFrame | DataLoader): The raw data to fit the model to. It can be either a pandas DataFrame or a DataLoader object.
            metadata (Metadata): The metadata associated with the raw data.

        Returns:
            None

        Raises:
            InitializationError: If neither raw_data nor metadata is provided.
        """
        if raw_data is not None and type(raw_data) in [pd.DataFrame, DataLoader]:
            if metadata:
                self._metadata = metadata
            self._fit_with_data(raw_data)
            return
        if type(raw_data) is Metadata:
            self._fit_with_metadata(raw_data)
            return
        if metadata is not None and type(metadata) is Metadata:
            self._fit_with_metadata(metadata)
            return
        raise InitializationError('Ple1ase pass at least one valid parameter, train_data or metadata')

    def _fit_with_metadata(self, metadata):
        """
        Fit the model using metadata.

        Args:
            metadata: Metadata object.

        Returns:
            None
        """
        logger.info('Fitting model with metadata...')
        self.use_metadata = True
        self._metadata = metadata
        self.columns = list(metadata.column_list)
        logger.info('Fitting model with metadata... Finished.')

    def _fit_with_data(self, train_data):
        """
        Fit the model using data.

        Args:
            train_data: Training data.

        Returns:
            None
        """
        logger.info('Fitting model with raw data...')
        self.use_raw_data = True
        self.use_dataloader = False
        if type(train_data) is DataLoader:
            self.columns = list(train_data.columns())
            train_data = train_data.load_all()
        if not self.columns:
            self.columns = list(train_data.columns)
        if not self._metadata:
            self._metadata = Metadata.from_dataframe(train_data)
        sample_lines = []
        for _, row in train_data.iterrows():
            each_line = ''
            shuffled_columns = copy(self.columns)
            random.shuffle(shuffled_columns)
            for column in shuffled_columns:
                value = str(row[column])
                each_line += f'{column} is {value}, '
            each_line = each_line[:-2]
            each_line += '\n'
            sample_lines.append(each_line)
        self._sample_lines = sample_lines
        logger.info('Fitting model with raw data... Finished.')

    @staticmethod
    def _select_random_elements(input_list, cnt):
        """
        This function selects a random sample of elements from the input list.

        Args:
            input_list (list): The list from which elements will be selected.
            cnt (int): The number of elements to be selected.

        Returns:
            list: A list of randomly selected elements from the input list.

        Raises:
            ValueError: If cnt is greater than the length of the input list.
        """
        if cnt > len(input_list):
            raise ValueError('cnt should not be greater than the length of the list')
        return random.sample(input_list, cnt)

    def _form_message_with_data(self, sample_list, current_cnt):
        """
        This function forms a message with data.

        Args:
            sample_list (list): A list of samples.
            current_cnt (int): The current count of samples.

        Returns:
            str: The formed message with data.
        """
        sample_str = ''
        for i in range(current_cnt):
            each_sample = sample_list[i]
            each_str = f'sample {i}: ' + each_sample + '\n'
            sample_str += each_str
        message = self.prompts['message_prefix'] + sample_str
        message = message + self._form_dataset_description()
        message = message + self._form_message_with_offtable_features()
        message = message + f'Please note that the generated table has total {len(self.columns) + len(self.off_table_features)} columns of the generated data, the column names are {self.columns + self.off_table_features}, every column should not be missed when generating the data. \n'
        message = message + self.prompts['message_suffix'] + str(current_cnt) + '.'
        self._message_list.append(message)
        logger.debug('Message Generated.')
        return message

    def extract_samples_from_response(self, response_content):
        """
        Extracts samples from the response content.

        Args:
            response_content (dict): The response content as a dictionary.

        Returns:
            list: A list of extracted samples.
        """

        def dict_to_list(input_dict, header):
            """
            Converts a dictionary to a list based on the given header.

            Args:
                input_dict (dict): The input dictionary.
                header (list): The list of keys to extract from the dictionary.

            Returns:
                list: A list of values extracted from the dictionary based on the header.
            """
            res = []
            for each_col in header:
                each_value = input_dict.get(each_col, None)
                res.append(each_value)
            return res
        logger.info('Extracting samples from response ...')
        header = self.columns + self.off_table_features
        features = []
        for line in response_content.split('\n'):
            feature = {}
            for field in header:
                pattern = '\\b' + field + '\\s*(?:is|=)\\s*([^,\\n]+)'
                match = re.search(pattern, line)
                if match:
                    feature[field] = match.group(1).strip()
            if feature:
                features.append(dict_to_list(feature, header))
        logger.info(f'Extracting samples from response ... Finished, {len(features)} extracted.')
        return features

    def sample(self, count=50, dataset_desp='', *args, **kwargs):
        """
        This function samples data from either raw data or metadata based on the given parameters.

        Args:
            count (int): The number of samples to be generated. Default is 50.
            dataset_desp (str): The description of the dataset. Default is an empty string.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            res: The sampled data.
        """
        logger.info('Sampling use GPT model ...')
        self.dataset_description = dataset_desp
        if self.use_raw_data:
            res = self._sample_with_data(count, *args, **kwargs)
        elif self.use_metadata:
            res = self._sample_with_metadata(count, *args, **kwargs)
        logger.info('Sampling use GPT model ... Finished.')
        return res

    def _form_message_with_metadata(self, current_cnt):
        """
        This function forms a message with metadata for table data generation task.

        Args:
            current_cnt (int): The current count of the message.

        Returns:
            str: The formed message with metadata.
        """
        message = ''
        message = message + self.prompts['message_prefix']
        message = message + self._form_dataset_description()
        message = message + 'This table data generation task will only have metadata and no data samples. The header (columns infomation) of the tabular data is: '
        message = message + str(self.columns) + '. \n'
        message = message + self._form_message_with_offtable_features()
        message = message + f'Note that the generated table has total {len(self.columns) + len(self.off_table_features)} columns, the column names are {self.columns + self.off_table_features}, every column should NOT be missed in generated data.\n'
        message = message + self.prompts['message_suffix'] + str(current_cnt) + '.'
        self._message_list.append(message)
        return message

    def _sample_with_metadata(self, count, *args, **kwargs):
        """
        This method samples data with metadata.

        Args:
            count (int): The number of samples to be generated.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            int: The input count.

        """
        logger.info('Sampling with metadata.')
        result = []
        remaining_cnt = count
        while remaining_cnt > 0:
            if remaining_cnt - self.query_batch >= 0:
                current_cnt = self.query_batch
            else:
                current_cnt = remaining_cnt
            message = self._form_message_with_metadata(current_cnt)
            response = self.ask_gpt(message)
            generated_batch = self.extract_samples_from_response(response)
            result += generated_batch
            remaining_cnt = remaining_cnt - current_cnt
        self._result_list.append(result)
        final_columns = self.columns + self.off_table_features
        return pd.DataFrame(result, columns=final_columns)

    def _sample_with_data(self, count, *args, **kwargs):
        """
        This function samples data with a given count and returns a DataFrame with the sampled data.

        Args:
            count (int): The number of data samples to be generated.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pd.DataFrame: A DataFrame containing the sampled data.

        """
        logger.info('Sampling with raw_data.')
        result = []
        remaining_cnt = count
        while remaining_cnt > 0:
            if remaining_cnt - self.query_batch >= 0:
                current_cnt = self.query_batch
            else:
                current_cnt = remaining_cnt
            sample_list = self._select_random_elements(self._sample_lines, current_cnt)
            message = self._form_message_with_data(sample_list, current_cnt)
            response = self.ask_gpt(message)
            generated_batch = self.extract_samples_from_response(response)
            result += generated_batch
            remaining_cnt = remaining_cnt - current_cnt
        self._result_list.append(result)
        final_columns = self.columns + self.off_table_features
        return pd.DataFrame(result, columns=final_columns)

