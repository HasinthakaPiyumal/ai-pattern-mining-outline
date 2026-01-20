# Cluster 13

def serialize(obj, max_depth=5, compress=False):
    """
    dump into json, including only basic types, list types and dict types.
    If other types are included, they will be converted into string.
    """
    if max_depth <= 0:
        return '...'
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, list) or isinstance(obj, tuple):
        if not compress or len(obj) <= 5:
            return [serialize(item, max_depth - 1, compress) for item in obj]
        else:
            return [serialize(item, max_depth - 1, True) for item in obj[:5]] + ['...(total: %d)' % len(obj)]
    elif isinstance(obj, dict):
        if not compress or len(obj) <= 5:
            return {str(key): serialize(obj[key], max_depth - 1, compress) for key in obj}
        else:
            ret = {str(key): serialize(obj[key], max_depth - 1, True) for key in list(obj.keys())[:5]}
            ret['...total...'] = len(obj)
            return ret
    elif hasattr(obj, '__dict__'):
        return serialize(obj.__dict__, max_depth, True)
    else:
        ret = str(obj)
        if len(ret) > 100:
            ret = ret[:45] + '   ...   ' + ret[-45:]
        return ret

