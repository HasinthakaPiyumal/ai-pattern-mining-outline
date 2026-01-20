# Cluster 107

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    if isinstance(obj, collections.Sequence):
        return [as_variable(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_variable(v) for k, v in obj.items()}
    else:
        return Variable(obj)

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    if isinstance(obj, collections.Sequence):
        return [as_variable(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_variable(v) for k, v in obj.items()}
    else:
        return Variable(obj)

