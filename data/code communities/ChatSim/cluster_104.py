# Cluster 104

class TorchTestCase(unittest.TestCase):

    def assertTensorClose(self, a, b, atol=0.001, rtol=0.001):
        npa, npb = (as_numpy(a), as_numpy(b))
        self.assertTrue(np.allclose(npa, npb, atol=atol), 'Tensor close check failed\n{}\n{}\nadiff={}, rdiff={}'.format(a, b, np.abs(npa - npb).max(), np.abs((npa - npb) / np.fmax(npa, 1e-05)).max()))

def as_numpy(v):
    if isinstance(v, Variable):
        v = v.data
    return v.cpu().numpy()

def as_numpy(obj):
    if isinstance(obj, collections.Sequence):
        return [as_numpy(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, Variable):
        return obj.data.cpu().numpy()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy()
    else:
        return np.array(obj)

class TorchTestCase(unittest.TestCase):

    def assertTensorClose(self, a, b, atol=0.001, rtol=0.001):
        npa, npb = (as_numpy(a), as_numpy(b))
        self.assertTrue(np.allclose(npa, npb, atol=atol), 'Tensor close check failed\n{}\n{}\nadiff={}, rdiff={}'.format(a, b, np.abs(npa - npb).max(), np.abs((npa - npb) / np.fmax(npa, 1e-05)).max()))

def as_numpy(obj):
    if isinstance(obj, collections.Sequence):
        return [as_numpy(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, Variable):
        return obj.data.cpu().numpy()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy()
    else:
        return np.array(obj)

