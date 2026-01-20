# Cluster 38

def get_transformer_instance(transformer):
    """Load a new instance of a ``Transformer``.

    The ``transformer`` is expected to be the transformers path as a ``string``,
    a transformer instance or a transformer type.

    Args:
        transformer (str or BaseTransformer):
            String with the transformer path or instance of a BaseTransformer subclass.

    Returns:
        BaseTransformer:
            BaseTransformer subclass instance.
    """
    if isinstance(transformer, BaseTransformer):
        return deepcopy(transformer)
    if inspect.isclass(transformer) and issubclass(transformer, BaseTransformer):
        return transformer()
    return get_transformer_class(transformer)()

def get_transformer_class(transformer):
    """Return a ``transformer`` class from a ``str``.

    Args:
        transformer (str):
            Python path.

    Returns:
        BaseTransformer:
            BaseTransformer subclass class object.
    """
    if transformer in TRANSFORMERS:
        return TRANSFORMERS[transformer]
    package, name = transformer.rsplit('.', 1)
    return getattr(importlib.import_module(package), name)

