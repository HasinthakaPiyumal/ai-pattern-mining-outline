# Cluster 22

def assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape):
        raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}')
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings():
                symbolic_assert(torch.equal(torch.as_tensor(size), ref_size), f'Wrong size for dimension {idx}')
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings():
                symbolic_assert(torch.equal(size, torch.as_tensor(ref_size)), f'Wrong size for dimension {idx}: expected {ref_size}')
        elif size != ref_size:
            raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')

@contextlib.contextmanager
def suppress_tracer_warnings():
    flt = ('ignore', None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)

