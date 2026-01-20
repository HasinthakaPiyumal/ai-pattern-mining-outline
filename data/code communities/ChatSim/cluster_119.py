# Cluster 119

@functools.wraps(op)
def oplist(*args, **kwargs):
    if len(args) == 0:
        raise ValueError('Must be at least one argument without keyword (i.e. operand).')
    if len(args) == 1:
        if islist(args[0]):
            return TensorList([op(a, **kwargs) for a in args[0]])
    else:
        if islist(args[0]) and islist(args[1]):
            return TensorList([op(a, b, *args[2:], **kwargs) for a, b in zip(*args[:2])])
        if islist(args[0]):
            return TensorList([op(a, *args[1:], **kwargs) for a in args[0]])
        if islist(args[1]):
            return TensorList([op(args[0], b, *args[2:], **kwargs) for b in args[1]])
    return op(*args, **kwargs)

def islist(a):
    return isinstance(a, TensorList)

