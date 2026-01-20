# Cluster 37

def get_transformer_name(transformer):
    """Return the fully qualified path of the transformer.

    Args:
        transformer:
            A transformer class.

    Raises:
        ValueError:
            Crashes when the transformer is not passed as a class.

    Returns:
        string:
            The path of the transformer.
    """
    if inspect.isclass(transformer):
        return transformer.__module__ + '.' + transformer.__name__
    raise ValueError(f'The transformer {transformer} must be passed as a class.')

@lru_cache()
def get_transformers_by_type():
    """Build a ``dict`` mapping sdtypes to valid existing transformers for that sdtype.

    Returns:
        dict:
            Mapping of sdtypes to a list of existing transformers that take that
            sdtype as an input.
    """
    sdtype_transformers = defaultdict(list)
    transformer_classes = BaseTransformer.get_subclasses()
    for transformer in transformer_classes:
        input_sdtype = transformer.get_input_sdtype()
        sdtype_transformers[input_sdtype].append(transformer)
    return sdtype_transformers

@lru_cache()
def get_default_transformers():
    """Build a ``dict`` mapping sdtypes to a default transformer for that sdtype.

    Returns:
        dict:
            Mapping of sdtypes to a transformer.
    """
    transformers_by_type = get_transformers_by_type()
    defaults = deepcopy(DEFAULT_TRANSFORMERS)
    for sdtype, transformers in transformers_by_type.items():
        if sdtype not in defaults:
            defaults[sdtype] = transformers[0]()
    return defaults

@lru_cache()
def get_default_transformer(sdtype):
    """Get default transformer for a sdtype.

    Returns:
        Transformer:
            Default transformer for sdtype.
    """
    default_transformers = get_default_transformers()
    return default_transformers[sdtype]

