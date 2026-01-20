# Cluster 70

def test_ndarray_loader(ndarray_list, ndarray_loaders):
    for ndarray_loader in ndarray_loaders:
        subtest_ndarray_loader_function(ndarray_loader, ndarray_list)
        subtest_ndarray_loader_slice(ndarray_loader, ndarray_list)

def subtest_ndarray_loader_function(ndarray_loader: NDArrayLoader, ndarray_list):
    ndarray_all = np.concatenate(ndarray_list, axis=1)
    for i, ndarray in enumerate(ndarray_loader.iter()):
        np.testing.assert_equal(ndarray, ndarray_list[i])
    np.testing.assert_equal(ndarray_loader.get_all(), ndarray_all)
    assert ndarray_loader.shape == ndarray_all.shape

def subtest_ndarray_loader_slice(ndarray_loader: NDArrayLoader, ndarray_list):
    ndarray_all = np.concatenate(ndarray_list, axis=1)
    np.testing.assert_equal(ndarray_loader[:], ndarray_all[:])
    np.testing.assert_equal(ndarray_loader[:], ndarray_all[:])
    np.testing.assert_equal(ndarray_loader[:, :], ndarray_all[:, :])
    np.testing.assert_equal(ndarray_loader[:, :], ndarray_all[:, :])
    np.testing.assert_equal(ndarray_loader[:, 1], ndarray_all[:, 1])
    np.testing.assert_equal(ndarray_loader[1, :], ndarray_all[1, :])
    np.testing.assert_equal(ndarray_loader[1:3], ndarray_all[1:3])
    '\n    2, 3\n    5, 6\n    8, 9\n    '
    np.testing.assert_equal(ndarray_loader[1:3, 1], ndarray_all[1:3, 1])
    '\n    5\n    6\n    '
    np.testing.assert_equal(ndarray_loader[1:3, 1:3], ndarray_all[1:3, 1:3])
    '\n    5, 6\n    8, 9\n    '

