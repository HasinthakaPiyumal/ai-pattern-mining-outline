# Cluster 112

def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).
    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).
    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.
    """
    distances = distances = _pdist(x, y)
    return np.maximum(0.0, torch.min(distances, axis=0)[0].numpy())

def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """
    a, b = (np.asarray(a), np.asarray(b))
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = (np.square(a).sum(axis=1), np.square(b).sum(axis=1))
    r2 = -2.0 * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0.0, float(np.inf))
    return r2

