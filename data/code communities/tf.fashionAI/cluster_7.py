# Cluster 7

def layer_variable_getter(getter, *args, **kwargs):
    kwargs['rename'] = rename
    return _model_variable_getter(getter, *args, **kwargs)

def _model_variable_getter(getter, name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=True, collections=None, caching_device=None, partitioner=None, rename=None, use_resource=None, **_):
    """Getter that uses model_variable for compatibility with core layers."""
    short_name = name.split('/')[-1]
    if rename and short_name in rename:
        name_components = name.split('/')
        name_components[-1] = rename[short_name]
        name = '/'.join(name_components)
    return variables.model_variable(name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, collections=collections, trainable=trainable, caching_device=caching_device, partitioner=partitioner, custom_getter=getter, use_resource=use_resource)

