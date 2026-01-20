# Cluster 13

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

class SingleTableMetric:
    """SingleTableMetric

    Metrics used to evaluate the quality of single table synthetic data.
    """
    upper_bound = None
    lower_bound = None
    metric_name = None
    metadata = None

    def __init__(self, metadata: dict) -> None:
        """Initialization

        Args:
            metadata(dict): This parameter accepts a metadata description dict, which is used to describe the column description information of the table.
        """
        self.metadata = metadata
        pass

    @classmethod
    def check_input(cls, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """Input check for single table input.

        Args:
            real_data(pd.DataFrame): the real (original) data table.

            synthetic_data(pd.DataFrame): the synthetic (generated) data table.
        """
        if real_data is None or synthetic_data is None:
            raise TypeError('Input contains None.')
        if type(real_data) is not type(synthetic_data):
            raise TypeError('Data type of real_data and synthetic data should be the same.')
        if isinstance(real_data, pd.DataFrame):
            return (real_data, synthetic_data)
        try:
            real_data = pd.DataFrame(real_data)
            synthetic_data = pd.DataFrame(synthetic_data)
            return (real_data, synthetic_data)
        except Exception as e:
            logger.error(f'An error occurred while converting to pd.DataFrame: {e}')
        return (None, None)

    def calculate(cls, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """Calculate the metric value between a real table and a synthetic table.

        Args:
            real_data(pd.DataFrame): the real (original) data table.

            synthetic_data(pd.DataFrame): the synthetic (generated) data table.
        """
        raise NotImplementedError()

    @classmethod
    def check_output(raw_metric_value: float):
        """Check the output value.

        Args:
            raw_metric_value (float):  the calculated raw value of the Mutual Information Similarity.
        """
        raise NotImplementedError()
    pass

class ColumnMetric(object):
    """ColumnMetric

    Metrics used to evaluate the quality of synthetic data columns.
    """
    upper_bound = None
    lower_bound = None
    metric_name = 'Accuracy'

    def __init__(self) -> None:
        pass

    @classmethod
    def check_input(cls, real_data: pd.Series | pd.DataFrame, synthetic_data: pd.Series | pd.DataFrame):
        """Input check for column or table input.

        Args:
            real_data(pd.DataFrame or pd.Series): the real (original) data table / column.

            synthetic_data(pd.DataFrame or pd.Series): the synthetic (generated) data table / column.
        """
        if real_data is None or synthetic_data is None:
            raise TypeError('Input contains None.')
        if type(real_data) is not type(synthetic_data):
            raise TypeError('Data type of real_data and synthetic data should be the same.')
        if type(real_data) in [int, float, str]:
            raise TypeError("real_data's type must not be None, int, float or str")
        if isinstance(real_data, pd.Series) or isinstance(real_data, pd.DataFrame):
            return (real_data, synthetic_data)
        try:
            real_data = pd.Series(real_data)
            synthetic_data = pd.Series(synthetic_data)
            return (real_data, synthetic_data)
        except Exception as e:
            logger.error(f'An error occurred while converting to pd.Series: {e}')
        return (None, None)

    @classmethod
    def calculate(cls, real_data: pd.Series | pd.DataFrame, synthetic_data: pd.Series | pd.DataFrame):
        """Calculate the metric value between columns between real table and synthetic table.
        Args:
            real_data(pd.DataFrame or pd.Series): the real (original) data table / column.
            synthetic_data(pd.DataFrame or pd.Series): the synthetic (generated) data table / column.
        """
        real_data, synthetic_data = ColumnMetric.check_input(real_data, synthetic_data)
        raise NotImplementedError()

    @classmethod
    def check_output(cls, raw_metric_value: float):
        """Check the output value.

        Args:
            raw_metric_value (float):  the calculated raw value of the JSD metric.
        """
        raise NotImplementedError()
    pass

class PairMetric(object):
    """PairMetric
    Metrics used to evaluate the quality of synthetic data columns.
    """
    upper_bound = None
    lower_bound = None
    metric_name = 'Correlation'

    def __init__(self) -> None:
        pass

    @classmethod
    def check_input(cls, src_col: pd.Series, tar_col: pd.Series, metadata: dict):
        """Input check for table input.
        Args:
            src_data(pd.Series ): the source data column.
            tar_data(pd.Series): the target data column .
            metadata(dict): The metadata that describes the data type of each column
        """
        if real_data is None or synthetic_data is None:
            raise TypeError('Input contains None.')
        tar_name = tar_col.name
        src_name = src_col.name
        if metadata[tar_name] != metadata[src_name]:
            raise TypeError('Type of Pair is Conflicting.')
        if isinstance(real_data, pd.Series):
            return (src_col, tar_col)
        try:
            src_col = pd.Series(src_col)
            tar_col = pd.Series(tar_col)
            return (src_col, tar_col)
        except Exception as e:
            logger.error(f'An error occurred while converting to pd.Series: {e}')
        return (None, None)

    @classmethod
    def calculate(cls, src_col: pd.Series, tar_col: pd.Series, metadata):
        """Calculate the metric value between pair-columns between real table and synthetic table.

        Args:
            src_data(pd.Series ): the source data column.
            tar_data(pd.Series): the target data column .
            metadata(dict): The metadata that describes the data type of each column
        """
        real_data, synthetic_data = PairMetric.check_input(src_col, tar_col)
        raise NotImplementedError()

    @classmethod
    def check_output(cls, raw_metric_value: float):
        """Check the output value.

        Args:
            raw_metric_value (float):  the calculated raw value of the Mutual Information.
        """
        raise NotImplementedError()
    pass

class MultiTableMetric:
    """MultiTableMetric

    Metrics used to evaluate the quality of synthetic multi-table data.
    """
    upper_bound = None
    lower_bound = None
    metric_name = None
    metadata = None
    table_list = []

    def __init__(self, metadata: dict) -> None:
        """Initialization

        Args:
            metadata(dict): This parameter accepts a metadata description dict, which is used to describe the table relations and column description information for each table.
        """
        self.metadata = metadata

    @classmethod
    def check_input(cls, real_data: dict, synthetic_data: dict):
        """Format check for single table input.

        The `real_data` and `synthetic_data` should be dict, which contains tables (in pd.DataFrame).

        Args:
            real_data(dict): the real (original) data table.

            synthetic_data(dict): the synthetic (generated) data table.
        """
        if real_data is None or synthetic_data is None:
            raise TypeError('Input contains None.')
        if type(real_data) is not type(synthetic_data):
            raise TypeError('Data type of real_data and synthetic data should be the same.')
        if isinstance(real_data, dict) and len(real_data.keys()) > 0 and (len(synthetic_data.keys()) > 0):
            return (real_data, synthetic_data)
        logger.error('An error occurred while checking the input.')
        return (None, None)

    def calculate(self, real_data: dict, synthetic_data: dict):
        """Calculate the metric value between real tables and synthetic tables.

        Args:

            real_data(dict): the real (original) data table.

            synthetic_data(dict): the synthetic (generated) data table.
        """
        raise NotImplementedError()

    @classmethod
    def check_output(raw_metric_value: float):
        """Check the output value.
        Args:

            raw_metric_value (float):  the calculated raw value of the Mutual Information Similarity.
        """
        raise NotImplementedError()
    pass

