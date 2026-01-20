# Cluster 18

def report0(name, value):
    """Broadcasts the given set of scalars by the first process (`rank = 0`),
    but ignores any scalars provided by the other processes.
    See `report()` for further details.
    """
    report(name, value if _rank == 0 else [])
    return value

@misc.profiled_function
def report(name, value):
    """Broadcasts the given set of scalars to all interested instances of
    `Collector`, across device and process boundaries.

    This function is expected to be extremely cheap and can be safely
    called from anywhere in the training loop, loss function, or inside a
    `torch.nn.Module`.

    Warning: The current implementation expects the set of unique names to
    be consistent across processes. Please make sure that `report()` is
    called at least once for each unique name by each process, and in the
    same order. If a given process has no scalars to broadcast, it can do
    `report(name, [])` (empty list).

    Args:
        name:   Arbitrary string specifying the name of the statistic.
                Averages are accumulated separately for each unique name.
        value:  Arbitrary set of scalars. Can be a list, tuple,
                NumPy array, PyTorch tensor, or Python scalar.

    Returns:
        The same `value` that was passed in.
    """
    if name not in _counters:
        _counters[name] = dict()
    elems = torch.as_tensor(value)
    if elems.numel() == 0:
        return value
    elems = elems.detach().flatten().to(_reduce_dtype)
    moments = torch.stack([torch.ones_like(elems).sum(), elems.sum(), elems.square().sum()])
    assert moments.ndim == 1 and moments.shape[0] == _num_moments
    moments = moments.to(_counter_dtype)
    device = moments.device
    if device not in _counters[name]:
        _counters[name][device] = torch.zeros_like(moments)
    _counters[name][device].add_(moments)
    return value

