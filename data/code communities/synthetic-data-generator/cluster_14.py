# Cluster 14

class Manager(metaclass=Singleton):
    """
    Base class for all manager.

    Manager is a singleton class for preventing multiple initialization.

    Define following attributes in subclass:
        * register_type: Base class for registered class
        * project_name: Name of entry-point for extensio
        * hookspecs_model: Hook specification model(where @hookspec is defined)

    For available managers, please refer to :ref:`Plugin-supported modules`

    """
    register_type: type = object
    '\n    Base class for registered class\n    '
    project_name: str = ''
    '\n    Name of entry-point for extension\n    '
    hookspecs_model = None
    '\n    Hook specification model(where @hookspec is defined)\n    '

    def __init__(self):
        self.pm = pluggy.PluginManager(self.project_name)
        self.pm.add_hookspecs(self.hookspecs_model)
        self._registed_cls: dict[str, type[self.register_type]] = {}
        self.pm.load_setuptools_entrypoints(self.project_name)
        self.load_all_local_model()

    def load_all_local_model(self):
        """
        Implement this function to load all local model
        """
        return

    @property
    def registed_cls(self) -> dict[str, type]:
        """
        Access all registed class.

        Lazy load, only load once.
        """
        if self._registed_cls:
            return self._registed_cls
        for f in self.pm.hook.register(manager=self):
            try:
                f()
            except Exception as e:
                logger.exception(RegisterError(e))
                continue
        return self._registed_cls

    def _load_dir(self, module):
        """
        Import all python files in a submodule.
        """
        modules = glob.glob(join(dirname(module.__file__), '*.py'))
        sub_packages = (basename(f)[:-3] for f in modules if isfile(f) and (not f.endswith('__init__.py')))
        packages = (str(module.__package__) + '.' + i for i in sub_packages)
        for p in packages:
            self.pm.register(importlib.import_module(p))

    def _normalize_name(self, name: str) -> str:
        return name.strip().lower()

    def register(self, cls_name, cls: type):
        """
        Register a new model, if the model is already registed, skip it.
        """
        cls_name = self._normalize_name(cls_name)
        logger.debug(f'Register for new model: {cls_name}')
        if cls in self._registed_cls.values():
            logger.error(f'SKIP: {cls_name} is already registed')
            return
        if not issubclass(cls, self.register_type):
            logger.error(f'SKIP: {cls_name} is not a subclass of {self.register_type}')
            return
        self._registed_cls[cls_name] = cls

    def init(self, c, **kwargs: dict[str, Any]):
        """
        Init a new subclass of self.register_type.

        Raises:
            NotFoundError: if cls_name is not registered
            InitializationError: if failed to initialize
        """
        if isinstance(c, self.register_type):
            return c
        if isinstance(c, type):
            cls_type = c
        else:
            c = self._normalize_name(c)
            if not c in self.registed_cls:
                raise NotFoundError
            cls_type = self.registed_cls[c]
        try:
            instance = cls_type(**kwargs)
            if not isinstance(instance, self.register_type):
                raise InitializationError(f'{c} is not a subclass of {self.register_type}.')
            return instance
        except Exception as e:
            raise InitializationError(e)

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

class Metadata(BaseModel):
    """
    Metadata is mainly used to describe the data types of all columns in a single data table.

    For each column, there should be an instance of the Data Type object.

    .. Note::

        Use ``get``, ``set``, ``add``, ``delete`` to update tags in the metadata. And use `query` for querying a column for its tags.

    Args:
        primary_keys(List[str]): The primary key, a field used to uniquely identify each row in the table.
        The primary key of each row must be unique and not empty.

        column_list(list[str]): list of the comlumn name in the table, other columns lists are used to store column information.
    """
    primary_keys: Set[str] = set()
    '\n    primary_keys is used to store single primary key or composite primary key\n    '
    column_list: List[str] = Field(default_factory=list, title='The List of Column Names')
    '"\n    column_list is the actual value of self.column_list\n    '

    @field_validator('column_list')
    @classmethod
    def check_column_list(cls, value) -> Any:
        if len(value) == len(set(value)):
            return value
        raise MetadataInitError('column_list has duplicate element!')
    column_inspect_level: Dict[str, int] = defaultdict(lambda: 10)
    "\n    column_inspect_level is used to store every inspector's level, to specify the true type of each column.\n    "
    pii_columns: Set[str] = set()
    "\n    pii_columns is used to store all PII columns' name\n    "
    id_columns: Set[str] = set()
    int_columns: Set[str] = set()
    float_columns: Set[str] = set()
    bool_columns: Set[str] = set()
    discrete_columns: Set[str] = set()
    datetime_columns: Set[str] = set()
    const_columns: Set[str] = set()
    datetime_format: Dict = defaultdict(str)
    numeric_format: Dict = defaultdict(list)
    categorical_encoder: Union[Dict[str, CategoricalEncoderType], None] = defaultdict(str)
    categorical_threshold: Union[Dict[int, CategoricalEncoderType], None] = None
    version: str = '1.0'
    _extend: Dict[str, Set[str]] = defaultdict(set)
    '\n    For extend information, use ``get`` and ``set``\n    '

    def get_column_encoder_by_categorical_threshold(self, num_categories: int) -> Union[CategoricalEncoderType, None]:
        encoder_type = None
        if self.categorical_threshold is None:
            return encoder_type
        for threshold in sorted(self.categorical_threshold.keys()):
            if num_categories > threshold:
                encoder_type = self.categorical_threshold[threshold]
            else:
                break
        return encoder_type

    def get_column_encoder_by_name(self, column_name) -> Union[CategoricalEncoderType, None]:
        encoder_type = None
        if self.categorical_encoder and column_name in self.categorical_encoder:
            encoder_type = self.categorical_encoder[column_name]
        return encoder_type

    @property
    def tag_fields(self) -> Iterable[str]:
        """
        Return all tag fields in this metadata.
        """
        return chain((k for k in self.model_fields if k.endswith('_columns')), (k for k in self._extend.keys() if k.endswith('_columns')))

    @property
    def format_fields(self) -> Iterable[str]:
        """
        Return all tag fields in this metadata.
        """
        return chain((k for k in self.model_fields if k.endswith('_format')), (k for k in self._extend.keys() if k.endswith('_format')))

    def __eq__(self, other):
        if not isinstance(other, Metadata):
            return super().__eq__(other)
        return set(self.tag_fields) == set(other.tag_fields) and all((self.get(key) == other.get(key) for key in set(chain(self.tag_fields, other.tag_fields)))) and all((self.get(key) == other.get(key) for key in set(chain(self.format_fields, other.format_fields)))) and (self.version == other.version)

    def query(self, field: str) -> Iterable[str]:
        """
        Query all tags of a field.

        Args:
            field(str): The field to query.

        Example:

            .. code-block:: python

                # Assume that user_id looks like 1,2,3,4
                m.query("user_id") == ["id_columns", "numeric_columns"]
        """
        return (k for k in self.tag_fields if field in self.get(k))

    def get(self, key: str) -> Set[str]:
        """
        Get all tags by key.

        Args:
            key(str): The key to get.

        Example:

            .. code-block:: python

                # Get all id columns
                m.get("id_columns") == {"user_id", "ticket_id"}
        """
        if key == '_extend':
            raise MetadataInitError('Cannot get _extend directly')
        return getattr(self, key) if key in self.model_fields else self._extend[key]

    def set(self, key: str, value: Any):
        """
        Set tags, will convert value to set if value is not a set.

        Args:
            key(str): The key to set.
            value(Any): The value to set.

        Example:

            .. code-block:: python

                # Set all id columns
                m.set("id_columns", {"user_id", "ticket_id"})
        """
        if key == '_extend':
            raise MetadataInitError('Cannot set _extend directly')
        old_value = self.get(key)
        if key in self.model_fields and key not in self.tag_fields and (key not in self.format_fields):
            raise MetadataInitError(f'Set {key} not in tag_fields, try set it directly as m.{key} = value')
        if isinstance(old_value, Iterable) and (not isinstance(old_value, str)):
            value = value if isinstance(value, Iterable) and (not isinstance(value, str)) else [value]
            try:
                value = type(old_value)(value)
            except TypeError as e:
                if type(old_value) == defaultdict:
                    value = dict(value)
                else:
                    raise e
        if key in self.model_fields:
            setattr(self, key, value)
        else:
            self._extend[key] = value

    def add(self, key: str, values: str | Iterable[str]):
        """
        Add tags.

        Args:
            key(str): The key to add.
            values(str | Iterable[str]): The value to add.

        Example:

            .. code-block:: python

                # Add all id columns
                m.add("id_columns", "user_id")
                m.add("id_columns", "ticket_id")
                # OR
                m.add("id_columns", ["user_id", "ticket_id"])
                # OR
                # add datetime format
                m.add('datetime_format',{"col_1": "%Y-%m-%d %H:%M:%S", "col_2": "%d %b %Y"})
        """
        values = values if isinstance(values, Iterable) and (not isinstance(values, str)) else [values]
        if isinstance(values, dict):
            if key in list(self.format_fields):
                self.get(key).update(values)
            if self._extend.get(key, None) is None:
                self._extend[key] = values
            else:
                self._extend[key].update(values)
            return
        for value in values:
            self.get(key).add(value)

    def delete(self, key: str, value: str):
        """
        Delete tags.

        Args:
            key(str): The key to delete.
            value(str): The value to delete.

        Example:

            .. code-block:: python

                # Delete misidentification id columns
                m.delete("id_columns", "not_an_id_columns")

        """
        try:
            self.get(key).remove(value)
        except KeyError:
            pass

    def update(self, attributes: dict[str, Any]):
        """
        Update tags.
        """
        for k, v in attributes.items():
            self.add(k, v)
        return self

    @classmethod
    def from_dataloader(cls, dataloader: DataLoader, max_chunk: int=10, primary_keys: Set[str]=None, include_inspectors: Iterable[str] | None=None, exclude_inspectors: Iterable[str] | None=None, inspector_init_kwargs: dict[str, Any] | None=None, check: bool=False) -> 'Metadata':
        """Initialize a metadata from DataLoader and Inspectors

        Args:
            dataloader(DataLoader): the input DataLoader.
            max_chunk(int): max chunk count.
            primary_keys(list[str]): primary keys, see :class:`~sdgx.data_models.metadata.Metadata` for more details.
            include_inspectors(list[str]): data type inspectors used in this metadata (table).
            exclude_inspectors(list[str]): data type inspectors NOT used in this metadata (table).
            inspector_init_kwargs(dict): inspector args.
        """
        logger.info('Inspecting metadata...')
        im = InspectorManager()
        exclude_inspectors = exclude_inspectors or []
        exclude_inspectors.extend((name for name, inspector_type in im.registed_inspectors.items() if issubclass(inspector_type, RelationshipInspector)))
        inspectors = im.init_inspcetors(include_inspectors, exclude_inspectors, **inspector_init_kwargs or {})
        for inspector in inspectors:
            inspector.ready = False
        for i, chunk in enumerate(dataloader.iter()):
            for inspector in inspectors:
                if not inspector.ready:
                    inspector.fit(chunk)
            if all((i.ready for i in inspectors)) or i > max_chunk:
                break
        if primary_keys is None:
            primary_keys = set()
        metadata = Metadata(primary_keys=primary_keys, column_list=dataloader.columns())
        for inspector in inspectors:
            inspect_res = inspector.inspect()
            metadata.update(inspect_res)
            if inspector.pii:
                for each_key in inspect_res:
                    metadata.update({'pii_columns': inspect_res[each_key]})
            for each_key in inspect_res:
                if 'columns' in each_key:
                    metadata.column_inspect_level[each_key] = inspector.inspect_level
        if not primary_keys:
            metadata.update_primary_key(metadata.id_columns)
        if check:
            metadata.check()
        return metadata

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, include_inspectors: list[str] | None=None, exclude_inspectors: list[str] | None=None, inspector_init_kwargs: dict[str, Any] | None=None, check: bool=False) -> 'Metadata':
        """Initialize a metadata from DataFrame and Inspectors

        Args:
            df(pd.DataFrame): the input DataFrame.
            include_inspectors(list[str]): data type inspectors used in this metadata (table).
            exclude_inspectors(list[str]): data type inspectors NOT used in this metadata (table).
            inspector_init_kwargs(dict): inspector args.
        """
        im = InspectorManager()
        exclude_inspectors = exclude_inspectors or []
        exclude_inspectors.extend((name for name, inspector_type in im.registed_inspectors.items() if issubclass(inspector_type, RelationshipInspector)))
        inspectors = im.init_inspcetors(include_inspectors, exclude_inspectors, **inspector_init_kwargs or {})
        for inspector in inspectors:
            inspector.fit(df)
        metadata = Metadata(primary_keys=[df.columns[0]], column_list=df.columns)
        for inspector in inspectors:
            inspect_res = inspector.inspect()
            metadata.update(inspect_res)
            if inspector.pii:
                for each_key in inspect_res:
                    metadata.update({'pii_columns': inspect_res[each_key]})
            for each_key in inspect_res:
                if 'columns' in each_key:
                    metadata.column_inspect_level[each_key] = inspector.inspect_level
        if check:
            metadata.check()
        return metadata

    def _dump_json(self) -> str:
        return self.model_dump_json(indent=4)

    def save(self, path: str | Path):
        """
        Save metadata to json file.
        """
        with path.open('w') as f:
            f.write(self._dump_json())

    @classmethod
    def loads(cls, attributes):
        return Metadata(**attributes)

    @classmethod
    def load(cls, path: str | Path) -> 'Metadata':
        """
        Load metadata from json file.
        """
        path = Path(path).expanduser().resolve()
        attributes = json.load(path.open('r'))
        version = attributes.get('version', None)
        if version:
            cls.upgrade(version, attributes)
        m = Metadata(**attributes)
        return m

    @classmethod
    def upgrade(cls, old_version: str, fields: dict[str, Any]) -> None:
        pass

    def check_single_primary_key(self, input_key: str):
        """Check whether a primary key in column_list and has ID data type.

        Args:
            input_key(str): the input primary_key str
        """
        if input_key not in self.column_list:
            raise MetadataInvalidError(f'Primary Key {input_key} not Exist in columns.')

    def get_all_data_type_columns(self):
        """Get all column names from `self.xxx_columns`.

        All Lists with the suffix _columns in model fields and extend fields need to be collected.
        All defined column names will be counted.

        Returns:
            all_dtype_cols(set): set of all column names.
        """
        all_dtype_cols = set()
        for each_key in list(self.model_fields.keys()) + list(self._extend.keys()):
            if each_key.endswith('_columns'):
                column_names = self.get(each_key)
                all_dtype_cols = all_dtype_cols.union(set(column_names))
        return all_dtype_cols

    def check(self):
        """Checks column info.

        When passing as input to the next module, perform necessary checks, including:
            -Is the primary key correctly defined(in column list) and has ID data type.
            -Is there any missing definition of each column in table.
            -Are there any unknown columns that have been incorrectly updated.
        """
        for each_key in self.primary_keys:
            self.check_single_primary_key(each_key)
        if len(self.primary_keys) == 1 and list(self.primary_keys)[0] not in self.id_columns:
            raise MetadataInvalidError(f'Primary Key {self.primary_keys} should has ID DataType.')
        all_dtype_columns = self.get_all_data_type_columns()
        if set(self.column_list) - set(all_dtype_columns):
            raise MetadataInvalidError(f'Undefined data type for column {set(self.column_list) - set(all_dtype_columns)}.')
        if set(all_dtype_columns) - set(self.column_list):
            raise MetadataInvalidError(f'Found undefined column: {set(all_dtype_columns) - set(self.column_list)}.')
        if self.categorical_encoder is not None:
            for i in self.categorical_encoder.keys():
                if not isinstance(i, str) or i not in self.discrete_columns:
                    raise MetadataInvalidError(f'categorical_encoder key {i} is invalid, it should be an str and is a discrete column name.')
            if self.categorical_encoder.values() not in CategoricalEncoderType:
                raise MetadataInvalidError(f'In categorical_encoder values, categorical encoder type invalid, now supports {list(CategoricalEncoderType)}.')
        if self.categorical_threshold is not None:
            for i in self.categorical_threshold.keys():
                if not isinstance(i, int) or i < 0:
                    raise MetadataInvalidError(f'categorical threshold {i} is invalid, it should be an positive int.')
            if self.categorical_threshold.values() not in CategoricalEncoderType:
                raise MetadataInvalidError(f'In categorical_threshold values, categorical encoder type invalid, now supports {list(CategoricalEncoderType)}.')
        logger.debug('Metadata check succeed.')

    def update_primary_key(self, primary_keys: Iterable[str] | str):
        """Update the primary key of the table

        When update the primary key, the original primary key will be erased.

        Args:
            primary_keys(Iterable[str]): the primary keys of this table.
        """
        if not isinstance(primary_keys, Iterable) and (not isinstance(primary_keys, str)):
            raise MetadataInvalidError('Primary key should be Iterable or str.')
        primary_keys = set(primary_keys if isinstance(primary_keys, Iterable) else [primary_keys])
        if not primary_keys.issubset(set(self.column_list)):
            raise MetadataInvalidError('Primary key not exist in table columns.')
        self.primary_keys = primary_keys
        logger.info(f'Primary Key updated: {primary_keys}.')

    def dump(self):
        """Dump model dict, can be used in downstream process, like processor.

        Returns:
            dict: dumped dict.
        """
        model_dict = self.model_dump()
        model_dict['column_data_type'] = {}
        for each_col in self.column_list:
            model_dict['column_data_type'][each_col] = self.get_column_data_type(each_col)
        return model_dict

    def get_column_data_type(self, column_name: str):
        """Get the exact type of specific column.
        Args:
            column_name(str): The query colmun name.
        Returns:
            str: The data type query result.
        """
        if column_name not in self.column_list:
            raise MetadataInvalidError(f'Column {column_name}not exists in metadata.')
        current_type = None
        current_level = 0
        for each_key in list(self.model_fields.keys()) + list(self._extend.keys()):
            if each_key != 'pii_columns' and each_key.endswith('_columns') and (column_name in self.get(each_key)) and (current_level < self.column_inspect_level[each_key]):
                current_level = self.column_inspect_level[each_key]
                current_type = each_key
        if not current_type:
            raise MetadataInvalidError(f'Column {column_name} has no data type.')
        return current_type.split('_columns')[0]

    def get_column_pii(self, column_name: str):
        """Return if a column is a PII column.
        Args:
            column_name(str): The query colmun name.
        Returns:
            bool: The PII query result.
        """
        if column_name not in self.column_list:
            raise MetadataInvalidError(f'Column {column_name}not exists in metadata.')
        if column_name in self.pii_columns:
            return True
        return False

    def change_column_type(self, column_names: str | List[str], column_original_type: str, column_new_type: str):
        """Change the type of column."""
        if not column_names:
            return
        if isinstance(column_names, str):
            column_names = [column_names]
        all_fields = list(self.tag_fields)
        original_type = f'{column_original_type}_columns'
        new_type = f'{column_new_type}_columns'
        if original_type not in all_fields:
            raise MetadataInvalidError(f'Column type {column_original_type} not exist in metadata.')
        if new_type not in all_fields:
            raise MetadataInvalidError(f'Column type {column_new_type} not exist in metadata.')
        type_columns = self.get(original_type)
        diff = set(column_names).difference(type_columns)
        if diff:
            raise MetadataInvalidError(f'Columns {column_names} not exist in {original_type}.')
        self.add(new_type, column_names)
        type_columns = type_columns.difference(column_names)
        self.set(original_type, type_columns)

    def remove_column(self, column_names: List[str] | str):
        """
        Remove a column from all columns type.
        Args:
            column_names: List[str]: To removed columns name list.
        """
        if not column_names:
            return
        if isinstance(column_names, str):
            column_names = [column_names]
        column_names = frozenset(column_names)
        inter = column_names.intersection(self.column_list)
        if not inter:
            raise MetadataInvalidError(f'Columns {inter} not exist in metadata.')

        def do_remove_columns(key, get=True, to_removes=column_names):
            obj = self
            if get:
                target = obj.get(key)
            else:
                target = getattr(obj, key)
            res = None
            if isinstance(target, list):
                res = [item for item in target if item not in to_removes]
            elif isinstance(target, dict):
                if key == 'numeric_format':
                    obj.set(key, {k: {v2 for v2 in v if v2 not in to_removes} for k, v in target.items()})
                else:
                    res = {k: v for k, v in target.items() if k not in to_removes}
            elif isinstance(target, set):
                res = target.difference(to_removes)
            if res is not None:
                if get:
                    obj.set(key, res)
                else:
                    setattr(obj, key, res)
        to_remove_attribute = list(self.tag_fields)
        to_remove_attribute.extend(list(self.format_fields))
        for attr in to_remove_attribute:
            do_remove_columns(attr)
        for attr in ['column_list', 'primary_keys']:
            do_remove_columns(attr, False)
        self.check()

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

class DataTransformer(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def __init__(self, max_clusters=10, weight_threshold=0.005, metadata=None):
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold (float):
                Weight threshold for a Gaussian distribution to be kept.
        """
        self.metadata: Metadata = metadata
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

    def _fit_categorical_encoder(self, column_name: str, data: pd.DataFrame, encoder_type: CategoricalEncoderType) -> Tuple[CategoricalEncoderInstanceType, int, ActivationFuncType]:
        if encoder_type not in CategoricalEncoderMapper.keys():
            raise ValueError('Unsupported encoder type {0}.'.format(encoder_type))
        p: CategoricalEncoderParams = CategoricalEncoderMapper[encoder_type]
        encoder = p.encoder()
        encoder.fit(data, column_name)
        num_categories = p.categories_caculator(encoder)
        activate_fn = p.activate_fn
        return (encoder, num_categories, activate_fn)

    def _fit_continuous(self, data):
        """Train Bayesian GMM for continuous columns.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        gm = ClusterBasedNormalizer(model_missing_values=True, max_clusters=min(len(data), 10))
        gm.fit(data, column_name)
        num_components = sum(gm.valid_component_indicator)
        return ColumnTransformInfo(column_name=column_name, column_type='continuous', transform=gm, output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')], output_dimensions=1 + num_components)

    def _fit_discrete(self, data, encoder_type: CategoricalEncoderType=None):
        """Fit one hot encoder for discrete column.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        encoder, activate_fn, selected_encoder_type = (None, None, None)
        column_name = data.columns[0]
        if encoder_type is None and self.metadata:
            selected_encoder_type = encoder_type = self.metadata.get_column_encoder_by_name(column_name)
        if encoder_type is None:
            encoder_type = 'onehot'
        num_categories = -1
        if encoder_type == 'onehot':
            encoder, num_categories, activate_fn = self._fit_categorical_encoder(column_name, data, encoder_type)
        if not selected_encoder_type and self.metadata and (num_categories != -1):
            encoder_type = self.metadata.get_column_encoder_by_categorical_threshold(num_categories) or encoder_type
        if encoder_type == 'onehot':
            pass
        else:
            encoder, num_categories, activate_fn = self._fit_categorical_encoder(column_name, data, encoder_type)
        assert encoder and activate_fn
        return ColumnTransformInfo(column_name=column_name, column_type='discrete', transform=encoder, output_info=[SpanInfo(num_categories, activate_fn)], output_dimensions=num_categories)

    def fit(self, data_loader: DataLoader, discrete_columns=()):
        """Fit the ``DataTransformer``.

        Fits a ``ClusterBasedNormalizer`` for continuous columns and a
        ``OneHotEncoder`` for discrete columns.

        This step also counts the #columns in matrix data and span information.
        """
        self.output_info_list: List[List[SpanInfo]] = []
        self.output_dimensions: int = 0
        self.dataframe: bool = True
        self._column_raw_dtypes = data_loader[:data_loader.chunksize].infer_objects().dtypes
        self._column_transform_info_list: List[ColumnTransformInfo] = []
        for column_name in tqdm.tqdm(data_loader.columns(), desc='Preparing data', delay=3):
            if column_name in discrete_columns:
                logger.debug(f'Fitting discrete column {column_name}...')
                column_transform_info = self._fit_discrete(data_loader[[column_name]])
            else:
                logger.debug(f'Fitting continuous column {column_name}...')
                column_transform_info = self._fit_continuous(data_loader[[column_name]])
            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(self, column_transform_info, data):
        logger.debug(f'Transforming continuous column {column_transform_info.column_name}...')
        column_name = data.columns[0]
        data[column_name] = data[column_name].to_numpy().flatten()
        gm = column_transform_info.transform
        transformed = gm.transform(data)
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
        index = transformed[f'{column_name}.component'].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0
        return output

    def _transform_discrete(self, column_transform_info, data):
        logger.debug(f'Transforming discrete column {column_transform_info.column_name}...')
        encoder = column_transform_info.transform
        return encoder.transform(data).to_numpy()

    def _synchronous_transform(self, raw_data, column_transform_info_list) -> NDArrayLoader:
        """Take a Pandas DataFrame and transform columns synchronous.

        Outputs a list with Numpy arrays.
        """
        loader = NDArrayLoader.get_auto_save(raw_data)
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            if column_transform_info.column_type == 'continuous':
                loader.store(self._transform_continuous(column_transform_info, data).astype(float))
            else:
                loader.store(self._transform_discrete(column_transform_info, data).astype(float))
        return loader

    def _parallel_transform(self, raw_data, column_transform_info_list) -> NDArrayLoader:
        """Take a Pandas DataFrame and transform columns in parallel.

        Outputs a list with Numpy arrays.
        """
        processes = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            process = None
            if column_transform_info.column_type == 'continuous':
                process = delayed(self._transform_continuous)(column_transform_info, data)
            else:
                process = delayed(self._transform_discrete)(column_transform_info, data)
            processes.append(process)
        p = Parallel(n_jobs=-1, return_as='generator')
        loader = NDArrayLoader.get_auto_save(raw_data)
        for ndarray in tqdm.tqdm(p(processes), desc='Transforming data', total=len(processes), delay=3):
            loader.store(ndarray.astype(float))
        return loader

    def transform(self, dataloader: DataLoader) -> NDArrayLoader:
        """Take raw data and output a matrix data."""
        if dataloader.shape[0] < 500:
            loader = self._synchronous_transform(dataloader, self._column_transform_info_list)
        else:
            loader = self._parallel_transform(dataloader, self._column_transform_info_list)
        return loader

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes()))
        data = data.astype(float)
        data.iloc[:, 1] = np.argmax(column_data[:, 1:], axis=1)
        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value
        return gm.reverse_transform(data)

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
        return ohe.reverse_transform(data)[column_transform_info.column_name]

    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in tqdm.tqdm(self._column_transform_info_list, desc='Inverse transforming', delay=3):
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]
            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(column_transform_info, column_data, sigmas, st)
            else:
                recovered_column_data = self._inverse_transform_discrete(column_transform_info, column_data)
            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim
        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = pd.DataFrame(recovered_data, columns=column_names).astype(self._column_raw_dtypes)
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()
        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        """Get the ids of the given `column_name`."""
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == 'discrete':
                discrete_counter += 1
            column_id += 1
        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")
        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")
        return {'discrete_column_id': discrete_counter, 'column_id': column_id, 'value_id': np.argmax(one_hot)}

