# Cluster 20

def recurse(obj):
    if isinstance(obj, (list, tuple, set)):
        return [recurse(x) for x in obj]
    if isinstance(obj, dict):
        return [[recurse(x), recurse(y)] for x, y in obj.items()]
    if isinstance(obj, (str, int, float, bool, bytes, bytearray)):
        return None
    if f'{type(obj).__module__}.{type(obj).__name__}' in ['numpy.ndarray', 'torch.Tensor', 'torch.nn.parameter.Parameter']:
        return None
    if is_persistent(obj):
        return None
    return obj

def is_persistent(obj):
    """Test whether the given object or class is persistent, i.e.,
    whether it will save its source code when pickled.
    """
    try:
        if obj in _decorators:
            return True
    except TypeError:
        pass
    return type(obj) in _decorators

def _check_pickleable(obj):
    """Check that the given object is pickleable, raising an exception if
    it is not. This function is expected to be considerably more efficient
    than actually pickling the object.
    """

    def recurse(obj):
        if isinstance(obj, (list, tuple, set)):
            return [recurse(x) for x in obj]
        if isinstance(obj, dict):
            return [[recurse(x), recurse(y)] for x, y in obj.items()]
        if isinstance(obj, (str, int, float, bool, bytes, bytearray)):
            return None
        if f'{type(obj).__module__}.{type(obj).__name__}' in ['numpy.ndarray', 'torch.Tensor', 'torch.nn.parameter.Parameter']:
            return None
        if is_persistent(obj):
            return None
        return obj
    with io.BytesIO() as f:
        pickle.dump(recurse(obj), f)

