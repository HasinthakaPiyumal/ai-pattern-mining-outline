# Cluster 39

class FloatFormatter(BaseTransformer):
    """Transformer for numerical data.

    This transformer replaces integer values with their float equivalent.
    Non null float values are not modified.

    Null values are replaced using a ``NullTransformer``.

    Args:
        missing_value_replacement (object or None):
            Indicate what to do with the null values. If an integer or float is given,
            replace them with the given value. If the strings ``'mean'`` or ``'mode'`` are
            given, replace them with the corresponding aggregation. If ``None`` is given,
            do not replace them. Defaults to ``None``.
        model_missing_values (bool):
            Whether to create a new column to indicate which values were null or not. The column
            will be created only if there are null values. If ``True``, create the new column if
            there are null values. If ``False``, do not create the new column even if there
            are null values. Defaults to ``False``.
        learn_rounding_scheme (bool):
            Whether or not to learn what place to round to based on the data seen during ``fit``.
            If ``True``, the data returned by ``reverse_transform`` will be rounded to that place.
            Defaults to ``False``.
        enforce_min_max_values (bool):
            Whether or not to clip the data returned by ``reverse_transform`` to the min and
            max values seen during ``fit``. Defaults to ``False``.
        computer_representation (dtype):
            Accepts ``'Int8'``, ``'Int16'``, ``'Int32'``, ``'Int64'``, ``'UInt8'``, ``'UInt16'``,
            ``'UInt32'``, ``'UInt64'``, ``'Float'``.
            Defaults to ``'Float'``.
    """
    INPUT_SDTYPE = 'numerical'
    DETERMINISTIC_TRANSFORM = True
    DETERMINISTIC_REVERSE = True
    COMPOSITION_IS_IDENTITY = True
    null_transformer = None
    missing_value_replacement = None
    _dtype = None
    _rounding_digits = None
    _min_value = None
    _max_value = None

    def __init__(self, missing_value_replacement=None, model_missing_values=False, learn_rounding_scheme=False, enforce_min_max_values=False, computer_representation='Float'):
        self.missing_value_replacement = missing_value_replacement
        self.model_missing_values = model_missing_values
        self.learn_rounding_scheme = learn_rounding_scheme
        self.enforce_min_max_values = enforce_min_max_values
        self.computer_representation = computer_representation

    def get_output_sdtypes(self):
        """Return the output sdtypes supported by the transformer.

        Returns:
            dict:
                Mapping from the transformed column names to supported sdtypes.
        """
        output_sdtypes = {'value': 'float'}
        if self.null_transformer and self.null_transformer.models_missing_values():
            output_sdtypes['is_null'] = 'float'
        return self._add_prefix(output_sdtypes)

    def is_composition_identity(self):
        """Return whether composition of transform and reverse transform produces the input data.

        Returns:
            bool:
                Whether or not transforming and then reverse transforming returns the input data.
        """
        if self.null_transformer and (not self.null_transformer.models_missing_values()):
            return False
        return self.COMPOSITION_IS_IDENTITY

    @staticmethod
    def _learn_rounding_digits(data):
        data = np.array(data)
        roundable_data = data[~(np.isinf(data) | pd.isna(data))]
        if (roundable_data % 1 != 0).any():
            if (roundable_data == roundable_data.round(MAX_DECIMALS)).all():
                for decimal in range(MAX_DECIMALS + 1):
                    if (roundable_data == roundable_data.round(decimal)).all():
                        return decimal
        return None

    def _raise_out_of_bounds_error(self, value, name, bound_type, min_bound, max_bound):
        raise ValueError(f"The {bound_type} value in column '{name}' is {value}. All values represented by '{self.computer_representation}' must be in the range [{min_bound}, {max_bound}].")

    def _validate_values_within_bounds(self, data):
        if self.computer_representation != 'Float':
            fractions = data[~data.isna() & data % 1 != 0]
            if not fractions.empty:
                raise ValueError(f"The column '{data.name}' contains float values {fractions.tolist()}. All values represented by '{self.computer_representation}' must be integers.")
            min_value = data.min()
            max_value = data.max()
            min_bound, max_bound = INTEGER_BOUNDS[self.computer_representation]
            if min_value < min_bound:
                self._raise_out_of_bounds_error(min_value, data.name, 'minimum', min_bound, max_bound)
            if max_value > max_bound:
                self._raise_out_of_bounds_error(max_value, data.name, 'maximum', min_bound, max_bound)

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit.
        """
        self._validate_values_within_bounds(data)
        self._dtype = data.dtype
        if self.enforce_min_max_values:
            self._min_value = data.min()
            self._max_value = data.max()
        if self.learn_rounding_scheme:
            self._rounding_digits = self._learn_rounding_digits(data)
        self.null_transformer = NullTransformer(self.missing_value_replacement, self.model_missing_values)
        self.null_transformer.fit(data)

    def _transform(self, data):
        """Transform numerical data.

        Integer values are replaced by their float equivalent. Non null float values
        are left unmodified.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        self._validate_values_within_bounds(data)
        data = data.astype(np.float64)
        return self.null_transformer.transform(data)

    def _reverse_transform(self, data):
        """Convert data back into the original format.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()
        if self.missing_value_replacement is not None:
            data = self.null_transformer.reverse_transform(data)
        if self.enforce_min_max_values:
            data = data.clip(self._min_value, self._max_value)
        elif self.computer_representation != 'Float':
            min_bound, max_bound = INTEGER_BOUNDS[self.computer_representation]
            data = data.clip(min_bound, max_bound)
        is_integer = np.dtype(self._dtype).kind == 'i'
        if self.learn_rounding_scheme or is_integer:
            data = data.round(self._rounding_digits or 0)
        if pd.isna(data).any() and is_integer:
            return data
        return data.astype(self._dtype)

class RegexGenerator(BaseTransformer):
    """RegexGenerator transformer.

    This transformer will drop a column and regenerate it with the previously specified
    ``regex`` format. The transformer will also be able to handle nulls and regenerate null values
    if specified.

    Args:
        regex (str):
            String representing the regex function.
        missing_value_replacement (object or None):
            Indicate what to do with the null values. If an integer or float is given,
            replace them with the given value. If the strings ``'mean'`` or ``'mode'`` are
            given, replace them with the corresponding aggregation. If ``None`` is given,
            do not replace them. Defaults to ``None``.
        model_missing_values (bool):
            Whether to create a new column to indicate which values were null or not. The column
            will be created only if there are null values. If ``True``, create the new column if
            there are null values. If ``False``, do not create the new column even if there
            are null values. Defaults to ``False``.
    """
    DETERMINISTIC_TRANSFORM = False
    DETERMINISTIC_REVERSE = False
    INPUT_SDTYPE = 'text'
    null_transformer = None

    def __init__(self, regex_format='[A-Za-z]{5}', missing_value_replacement=None, model_missing_values=False):
        self.missing_value_replacement = missing_value_replacement
        self.model_missing_values = model_missing_values
        self.regex_format = regex_format
        self.data_length = None

    def get_output_sdtypes(self):
        """Return the output sdtypes supported by the transformer.

        Returns:
            dict:
                Mapping from the transformed column names to supported sdtypes.
        """
        output_sdtypes = {}
        if self.null_transformer and self.null_transformer.models_missing_values():
            output_sdtypes['is_null'] = 'float'
        return self._add_prefix(output_sdtypes)

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self.null_transformer = NullTransformer(self.missing_value_replacement, self.model_missing_values)
        self.null_transformer.fit(data)
        self.data_length = len(data)

    def _transform(self, data):
        """Return ``null`` column if ``models_missing_values``.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            (numpy.ndarray or None):
                If ``self.model_missing_values`` is ``True`` then will return a ``numpy.ndarray``
                indicating which values should be ``nan``, else will return ``None``. In both
                scenarios the original column is being dropped.
        """
        if self.null_transformer and self.null_transformer.models_missing_values():
            return self.null_transformer.transform(data)[:, 1].astype(float)
        return None

    def _reverse_transform(self, data):
        """Generate new data using the provided ``regex_format``.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        if data is not None and len(data):
            sample_size = len(data)
        else:
            sample_size = self.data_length
        generator, size = strings_from_regex(self.regex_format)
        if sample_size > size:
            warnings.warn(f"The data has {sample_size} rows but the regex for '{self.get_input_column()}' can only create {size} unique values. Some values in '{self.get_input_column()}' may be repeated.")
        if size > sample_size:
            reverse_transformed = np.array([next(generator) for _ in range(sample_size)], dtype=object)
        else:
            generated_values = list(generator)
            reverse_transformed = []
            while len(reverse_transformed) < sample_size:
                remaining = sample_size - len(reverse_transformed)
                reverse_transformed.extend(generated_values[:remaining])
            reverse_transformed = np.array(reverse_transformed, dtype=object)
        if self.null_transformer.models_missing_values():
            reverse_transformed = np.column_stack((reverse_transformed, data))
        return self.null_transformer.reverse_transform(reverse_transformed)

class UnixTimestampEncoder(BaseTransformer):
    """Transformer for datetime data.

    This transformer replaces datetime values with an integer timestamp
    transformed to float.

    Null values are replaced using a ``NullTransformer``.

    Args:
        missing_value_replacement (object or None):
            Indicate what to do with the null values. If an object is given, replace them
            with the given value. If the strings ``'mean'`` or ``'mode'`` are given, replace
            them with the corresponding aggregation. If ``None`` is given, do not replace them.
            Defaults to ``None``.
        model_missing_values (bool):
            Whether to create a new column to indicate which values were null or not. The column
            will be created only if there are null values. If ``True``, create the new column if
            there are null values. If ``False``, do not create the new column even if there
            are null values. Defaults to ``False``.
        datetime_format (str):
            The strftime to use for parsing time. For more information, see
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
    """
    INPUT_SDTYPE = 'datetime'
    DETERMINISTIC_TRANSFORM = True
    DETERMINISTIC_REVERSE = True
    COMPOSITION_IS_IDENTITY = True
    null_transformer = None

    def __init__(self, missing_value_replacement=None, model_missing_values=False, datetime_format=None):
        self.missing_value_replacement = missing_value_replacement
        self.model_missing_values = model_missing_values
        self.datetime_format = datetime_format
        self._dtype = None

    def is_composition_identity(self):
        """Return whether composition of transform and reverse transform produces the input data.

        Returns:
            bool:
                Whether or not transforming and then reverse transforming returns the input data.
        """
        if self.null_transformer and (not self.null_transformer.models_missing_values()):
            return False
        return self.COMPOSITION_IS_IDENTITY

    def get_output_sdtypes(self):
        """Return the output sdtypes supported by the transformer.

        Returns:
            dict:
                Mapping from the transformed column names to supported sdtypes.
        """
        output_sdtypes = {'value': 'float'}
        if self.null_transformer and self.null_transformer.models_missing_values():
            output_sdtypes['is_null'] = 'float'
        return self._add_prefix(output_sdtypes)

    def _convert_to_datetime(self, data):
        if data.dtype == 'object':
            try:
                pandas_datetime_format = None
                if self.datetime_format:
                    pandas_datetime_format = self.datetime_format.replace('%-', '%')
                data = pd.to_datetime(data, format=pandas_datetime_format)
            except ValueError as error:
                if 'Unknown string format:' in str(error):
                    message = 'Data must be of dtype datetime, or castable to datetime.'
                    raise TypeError(message) from None
                raise ValueError('Data does not match specified datetime format.') from None
        return data

    def _transform_helper(self, datetimes):
        """Transform datetime values to integer."""
        datetimes = self._convert_to_datetime(datetimes)
        nulls = datetimes.isna()
        integers = pd.to_numeric(datetimes, errors='coerce').to_numpy().astype(np.float64)
        integers[nulls] = np.nan
        transformed = pd.Series(integers)
        return transformed

    def _reverse_transform_helper(self, data):
        """Transform integer values back into datetimes."""
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()
        if self.model_missing_values or self.missing_value_replacement is not None:
            data = self.null_transformer.reverse_transform(data)
        data = np.round(data.astype(np.float64))
        return data

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit the transformer to.
        """
        self._dtype = data.dtype
        if self.datetime_format is None:
            datetime_array = data.astype(str).to_numpy()
            self.datetime_format = _guess_datetime_format_for_array(datetime_array)
        transformed = self._transform_helper(data)
        self.null_transformer = NullTransformer(self.missing_value_replacement, self.model_missing_values)
        self.null_transformer.fit(transformed)

    def _transform(self, data):
        """Transform datetime values to float values.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        data = self._transform_helper(data)
        return self.null_transformer.transform(data)

    def _reverse_transform(self, data):
        """Convert float values back to datetimes.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        data = self._reverse_transform_helper(data)
        datetime_data = pd.to_datetime(data)
        if not isinstance(datetime_data, pd.Series):
            datetime_data = pd.Series(datetime_data)
        if self.datetime_format:
            if self._dtype == 'object':
                datetime_data = datetime_data.dt.strftime(self.datetime_format)
            elif is_datetime64_dtype(self._dtype) and '.%f' not in self.datetime_format:
                datetime_data = pd.to_datetime(datetime_data.dt.strftime(self.datetime_format))
        return datetime_data

class BinaryEncoder(BaseTransformer):
    """Transformer for boolean data.

    This transformer replaces boolean values with their integer representation
    transformed to float.

    Null values are replaced using a ``NullTransformer``.

    Args:
        missing_value_replacement (object or None):
            Indicate what to do with the null values. If an object is given, replace them
            with the given value. If the string ``'mode'`` is given, replace them with the
            most common value. If ``None`` is given, do not replace them.
            Defaults to ``None``.
        model_missing_values (bool):
            Whether to create a new column to indicate which values were null or not. The column
            will be created only if there are null values. If ``True``, create the new column if
            there are null values. If ``False``, do not create the new column even if there
            are null values. Defaults to ``False``.
    """
    INPUT_SDTYPE = 'boolean'
    DETERMINISTIC_TRANSFORM = True
    DETERMINISTIC_REVERSE = True
    null_transformer = None

    def __init__(self, missing_value_replacement=None, model_missing_values=False):
        self.missing_value_replacement = missing_value_replacement
        self.model_missing_values = model_missing_values

    def get_output_sdtypes(self):
        """Return the output sdtypes returned by this transformer.

        Returns:
            dict:
                Mapping from the transformed column names to the produced sdtypes.
        """
        output_sdtypes = {'value': 'float'}
        if self.null_transformer and self.null_transformer.models_missing_values():
            output_sdtypes['is_null'] = 'float'
        return self._add_prefix(output_sdtypes)

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self.null_transformer = NullTransformer(self.missing_value_replacement, self.model_missing_values)
        self.null_transformer.fit(data)

    def _transform(self, data):
        """Transform boolean to float.

        The boolean values will be replaced by the corresponding integer
        representations as float values.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns
            pandas.DataFrame or pandas.Series
        """
        data = pd.to_numeric(data, errors='coerce')
        return self.null_transformer.transform(data).astype(float)

    def _reverse_transform(self, data):
        """Transform float values back to the original boolean values.

        Args:
            data (pandas.DataFrame or pandas.Series):
                Data to revert.

        Returns:
            pandas.Series:
                Reverted data.
        """
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()
        if self.missing_value_replacement is not None:
            data = self.null_transformer.reverse_transform(data)
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                data = data[:, 0]
            data = pd.Series(data)
        isna = data.isna()
        data = np.round(data).clip(0, 1).astype('boolean').astype('object')
        data[isna] = np.nan
        return data

class AnonymizedFaker(BaseTransformer):
    """Personal Identifiable Information Anonymizer using Faker.

    This transformer will drop a column and regenerate it with the previously specified
    ``Faker`` provider and ``function``. The transformer will also be able to handle nulls
    and regenerate null values if specified.

    Args:
        provider_name (str):
            The name of the provider in ``Faker``. If ``None`` the ``BaseProvider`` is used.
            Defaults to ``None``.
        function_name (str):
            The name of the function to use within the ``faker.provider``. Defaults to
            ``lexify``.
        function_kwargs (dict):
            Keyword args to pass into the ``function_name`` when being called.
        locales (list):
            List of localized providers to use instead of the global provider.
        missing_value_replacement (object or None):
            Indicate what to do with the null values. If an integer or float is given,
            replace them with the given value. If the strings ``'mean'`` or ``'mode'`` are
            given, replace them with the corresponding aggregation. If ``None`` is given,
            do not replace them. Defaults to ``None``.
        model_missing_values (bool):
            Whether to create a new column to indicate which values were null or not. The column
            will be created only if there are null values. If ``True``, create the new column if
            there are null values. If ``False``, do not create the new column even if there
            are null values. Defaults to ``False``.
    """
    DETERMINISTIC_TRANSFORM = False
    DETERMINISTIC_REVERSE = False
    INPUT_SDTYPE = 'pii'
    OUTPUT_SDTYPES = {}
    null_transformer = None

    @staticmethod
    def check_provider_function(provider_name, function_name):
        """Check that the provider and the function exist.

        Attempt to get the provider from ``faker.providers`` and then get the ``function``
        from the provider object. If one of them fails, it will raise an ``AttributeError``.

        Raises:
            ``AttributeError`` if the provider or the function is not found.
        """
        try:
            module = getattr(faker.providers, provider_name)
            if provider_name.lower() == 'baseprovider':
                getattr(module, function_name)
            else:
                provider = getattr(module, 'Provider')
                getattr(provider, function_name)
        except AttributeError as exception:
            raise Error(f"The '{provider_name}' module does not contain a function named '{function_name}'.\nRefer to the Faker docs to find the correct function: https://faker.readthedocs.io/en/master/providers.html") from exception

    def _check_locales(self):
        """Check if the locales exist for the provided provider."""
        locales = self.locales if isinstance(self.locales, list) else [self.locales]
        missed_locales = []
        for locale in locales:
            spec = importlib.util.find_spec(f'faker.providers.{self.provider_name}.{locale}')
            if spec is None:
                missed_locales.append(locale)
        if missed_locales:
            warnings.warn(f"Locales {missed_locales} do not support provider '{self.provider_name}' and function '{self.function_name}'.\nIn place of these locales, 'en_US' will be used instead. Please refer to the localized provider docs for more information: https://faker.readthedocs.io/en/master/locales.html")

    def __init__(self, provider_name=None, function_name=None, function_kwargs=None, locales=None, missing_value_replacement=None, model_missing_values=False):
        self.data_length = None
        self.provider_name = provider_name if provider_name else 'BaseProvider'
        if self.provider_name != 'BaseProvider' and function_name is None:
            raise Error(f"Please specify the function name to use from the '{self.provider_name}' provider.")
        self.function_name = function_name if function_name else 'lexify'
        self.function_kwargs = deepcopy(function_kwargs) if function_kwargs else {}
        self.check_provider_function(self.provider_name, self.function_name)
        self.missing_value_replacement = missing_value_replacement
        self.model_missing_values = model_missing_values
        self.locales = locales
        self.faker = faker.Faker(locales)
        if self.locales:
            self._check_locales()

    def _function(self):
        """Return a callable ``faker`` function."""
        return getattr(self.faker, self.function_name)(**self.function_kwargs)

    def get_output_sdtypes(self):
        """Return the output sdtypes supported by the transformer.

        Returns:
            dict:
                Mapping from the transformed column names to supported sdtypes.
        """
        output_sdtypes = self.OUTPUT_SDTYPES
        if self.null_transformer and self.null_transformer.models_missing_values():
            output_sdtypes['is_null'] = 'float'
        return self._add_prefix(output_sdtypes)

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self.null_transformer = NullTransformer(self.missing_value_replacement, self.model_missing_values)
        self.null_transformer.fit(data)
        self.data_length = len(data)

    def _transform(self, data):
        """Return ``null`` column if ``models_missing_values``.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            (numpy.ndarray or None):
                If ``self.model_missing_values`` is ``True`` then will return a ``numpy.ndarray``
                indicating which values should be ``nan``, else will return ``None``. In both
                scenarios the original column is being dropped.
        """
        if self.null_transformer and self.null_transformer.models_missing_values():
            return self.null_transformer.transform(data)[:, 1].astype(float)
        return None

    def _reverse_transform(self, data):
        """Generate new anonymized data using a ``faker.provider.function``.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        if data is not None and len(data):
            sample_size = len(data)
        else:
            sample_size = self.data_length
        reverse_transformed = np.array([self._function() for _ in range(sample_size)], dtype=object)
        if self.null_transformer.models_missing_values():
            reverse_transformed = np.column_stack((reverse_transformed, data))
        return self.null_transformer.reverse_transform(reverse_transformed)

    def __repr__(self):
        """Represent initialization of transformer as text.

        Returns:
            str:
                The name of the transformer followed by any non-default parameters.
        """
        class_name = self.__class__.__name__
        custom_args = []
        args = inspect.getfullargspec(self.__init__)
        keys = args.args[1:]
        defaults = dict(zip(keys, args.defaults))
        instanced = {key: getattr(self, key) for key in keys}
        defaults['function_name'] = None
        for arg, value in instanced.items():
            if value and defaults[arg] != value and (value != 'BaseProvider'):
                value = f"'{value}'" if isinstance(value, str) else value
                custom_args.append(f'{arg}={value}')
        args_string = ', '.join(custom_args)
        return f'{class_name}({args_string})'

