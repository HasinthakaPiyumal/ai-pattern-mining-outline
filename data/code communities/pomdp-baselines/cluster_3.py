# Cluster 3

def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum((get_dim(subspace) for subspace in space.spaces))
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise NotImplementedError

