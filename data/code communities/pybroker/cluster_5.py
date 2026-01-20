# Cluster 5

@pytest.mark.parametrize('values, expected_iqr', [([1, 3, 5, 7, 8, 10, 11, 13], 6.5), ([1], 0), ([1, 2], 0), ([1, 1, 1, 1, 1], 0), ([], 0)])
def test_iqr(values, expected_iqr):
    assert iqr(np.array(values)) == expected_iqr

def iqr(values: NDArray[np.float64]) -> float:
    """Computes the `interquartile range (IQR)
    <https://en.wikipedia.org/wiki/Interquartile_range>`_ of ``values``."""
    x = values[~np.isnan(values)]
    if not len(x):
        return 0
    percentiles: NDArray[np.float64] = np.percentile(x, [75, 25], method='midpoint')
    q75: float = float(percentiles[0])
    q25: float = float(percentiles[1])
    return q75 - q25

