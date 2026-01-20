# Cluster 36

def strings_from_regex(regex, max_repeat=16):
    """Generate strings that match the given regular expression.

    The output is a generator that produces regular expressions that match
    the indicated regular expressions alongside an integer indicating the
    total length of the generator.

    WARNING: Subpatterns are currently not supported.

    Args:
        regex (str):
            String representing a valid python regular expression.
        max_repeat (int):
            Maximum number of repetitions to produce when the regular
            expression allows an infinte amount. Defaults to 16.

    Returns:
        tuple:
            * Generator that produces strings that match the given regex.
            * Total length of the generator.
    """
    parsed = sre_parse.parse(regex, flags=sre_parse.SRE_FLAG_UNICODE)
    generators = []
    sizes = []
    for option, args in reversed(parsed):
        if option != sre_parse.AT:
            generator, size = _GENERATORS[option](args, max_repeat)
            generators.append((generator, option, args))
            sizes.append(size)
    return (_from_generators(generators, max_repeat), np.prod(sizes))

def _from_generators(generators, max_repeat):
    previous = [None] + [next(generator) for generator, _, _ in generators[1:]]
    remaining = True
    while remaining:
        generated = []
        for index, (generator, option, args) in enumerate(generators):
            remaining = True
            try:
                value = next(generator)
                generated.append(value)
                previous[index] = value
                generated.extend(previous[index + 1:])
                break
            except StopIteration:
                generator = _GENERATORS[option](args, max_repeat)[0]
                generators[index] = (generator, option, args)
                value = next(generator)
                previous[index] = value
                generated.append(value)
                remaining = False
        if remaining:
            yield ''.join(reversed(generated))

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

