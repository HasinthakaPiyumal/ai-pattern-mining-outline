# Cluster 21

class Inspector:
    """
    Base Inspector class

    Inspector is used to inspect data and generate metadata automatically.

    Parameters:
        ready (bool): Ready to inspect, maybe all fields are fitted, or indicate if there is more data, inspector will be more precise.
    """
    pii = False
    '\n    PII refers if a column contains private or sensitive information.\n    '
    _inspect_level: int = 10
    "\n    Private variable used to store property inspect_level's value.\n    "
    ready: bool = False
    '\n    Indicates whether the inspector has completed its inference.\n\n    When completed, ready == True.\n    '

    @property
    def inspect_level(self):
        """
        Inspected level is a concept newly introduced in version 0.1.6. Since a single column in the table may be marked by different inspectors at the same time (for example: the email column may be recognized as email, but it may also be recognized as the id column, and it may also be recognized by different inspectors at the same time identified as a discrete column, which will cause confusion in subsequent processing), the inspect_leve is used when determining the specific type of a column.

        We will preset different inspector levels for different inspectors, usually more specific inspectors will get higher levels, and general inspectors (like discrete) will have inspect_level.

        The value of the variable inspect_level is limited to 1-100. In baseclass and bool, discrete and numeric types, the inspect_level is set to 10. For datetime and id types, the inspect_level is set to 20.

        Current inspect_level value will make it easier for developers to insert a custom inspector from the middle.
        """
        return self._inspect_level

    @inspect_level.setter
    def inspect_level(self, value: int):
        if value > 0 and value <= 100:
            self._inspect_level = value
        else:
            raise InspectorInitError('The inspect_level should be set in [1, 100].')

    def __init__(self, inspect_level=None, *args, **kwargs):
        self.ready: bool = False
        if inspect_level:
            self.inspect_level = inspect_level

    def fit(self, raw_data: pd.DataFrame, *args, **kwargs):
        """Fit the inspector.

        Args:
            raw_data (pd.DataFrame): Raw data
        """
        return

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""

class RegexInspector(Inspector):
    """RegexInspector
    RegexInspector is a sdgx inspector that uses regular expression rules to detect column data types. It can be initialized with a custom expression, or it can be inherited and applied to specific data types,such as email, US address, HKID etc.

    By default, we will not directly register the RegexInspector to the Inspector Manager. Instead, use it as a baseclass or user-defined regex, then put it into the Inspector Manager or use it alone
    """
    pattern: str = None
    '\n    pattern is the regular expression string of current inspector.\n    '
    data_type_name: str = None
    '\n    data_type_name is the name of the data type, such as email, US address, HKID etc.\n    '
    _match_percentage: float = 0.8
    "\n    Private variable used to store property match_percentage's value.\n    "

    @property
    def match_percentage(self):
        """
        The match_percentage shoud > 0.5 and < 1.

        Due to the existence of empty data, wrong data, etc., the match_percentage is the proportion of the current regular expression compound. When the number of compound regular expressions is higher than this ratio, the column can be considered fit the current data type.
        """
        return self._match_percentage

    @match_percentage.setter
    def match_percentage(self, value):
        if value > 0.5 and value <= 1:
            self._match_percentage = value
        else:
            raise InspectorInitError('The match_percentage should be set in (0.5, 1].')

    def __init__(self, pattern: str=None, data_type_name: str=None, match_percentage: float=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regex_columns: set[str] = set()
        if pattern:
            self.pattern = pattern
        if self.pattern is None:
            raise InspectorInitError('Regular expression NOT found.')
        self.p = re.compile(self.pattern)
        if data_type_name:
            if data_type_name.endswith('_columns'):
                self.data_type_name = data_type_name[:-8]
            else:
                self.data_type_name = data_type_name
        elif not self.data_type_name:
            self.data_type_name = f'regex_{self.pattern}_columns'
        if self.data_type_name is None:
            raise InspectorInitError("Inspector's data type undefined.")
        if match_percentage:
            self.match_percentage = match_percentage

    def fit(self, input_raw_data: pd.DataFrame, *args, **kwargs):
        """Fit the inspector.

        Finds the list of regex columns from the tabular data (in pd.DataFrame).

        Args:
            raw_data (pd.DataFrame): Raw data
        """
        for each_col in input_raw_data.columns:
            each_match_rate = self._fit_column(input_raw_data[each_col])
            if each_match_rate > self.match_percentage:
                self.regex_columns.add(each_col)
        self.ready = True

    def domain_verification(self, each_sample: str):
        """
        The function domain_verification is used to add custom domain verification logic. When a sample matches a regular expression, the domain_verification function is executed for further verification.

        Additional logic checks can be performed beyond regular expressions, making it more flexible. For example, in a company name, there may be address information. When determining the type of address, if the sample ends with "Company", domain_verification can return False to avoid misclassification, thus improving the accuracy of the inspector.

        This function has the power to veto. When the function outputs False, the sample will be classified as not matching the corresponding data type of the inspector.

        If this function is not overwritten, domain_verification will default to return True.

        Args:
            each_sample (str): string of each sample.
        """
        return True

    def _fit_column(self, column_data: pd.Series):
        """
        Regular expression matching for a single column, returning the matching ratio.

        Args:
             column_data (pd.Series): the column data.
        """
        length = len(column_data)
        unmatch_cnt = 0
        match_cnt = 0
        for i in column_data:
            m = re.match(self.p, str(i))
            d = self.domain_verification(str(i))
            if m and d:
                match_cnt += 1
            else:
                unmatch_cnt += 1
                if unmatch_cnt > length * (1 - self.match_percentage) + 1:
                    break
        return match_cnt / length

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""
        return {self.data_type_name + '_columns': list(self.regex_columns)}

