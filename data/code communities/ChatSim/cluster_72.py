# Cluster 72

def get_shape(t):
    if torch.is_tensor(t):
        return tuple(t.shape)
    elif isinstance(t, dict):
        return {n: get_shape(q) for n, q in t.items()}
    elif isinstance(t, (list, tuple)):
        return [get_shape(q) for q in t]
    elif isinstance(t, numbers.Number):
        return type(t)
    else:
        raise ValueError('unexpected type {}'.format(type(t)))

def get_shape(t):
    if torch.is_tensor(t):
        return tuple(t.shape)
    elif isinstance(t, dict):
        return {n: get_shape(q) for n, q in t.items()}
    elif isinstance(t, (list, tuple)):
        return [get_shape(q) for q in t]
    elif isinstance(t, numbers.Number):
        return type(t)
    else:
        raise ValueError('unexpected type {}'.format(type(t)))

