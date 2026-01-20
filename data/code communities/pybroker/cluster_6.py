# Cluster 6

@njit
def lowv(array: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Calculates the lowest values for every ``n`` period in ``array``.

    Args:
        array: :class:`numpy.ndarray` of data.
        n: Length of period.

    Returns:
        :class:`numpy.ndarray` of the lowest values for every ``n`` period in
        ``array``.
    """
    if not len(array):
        return np.array(tuple())
    _verify_input(array, n)
    out_len = len(array)
    out = np.array([np.nan for _ in range(out_len)])
    for i in range(n, out_len + 1):
        out[i - 1] = np.min(array[i - n:i])
    return out

@njit
def _verify_input(array: NDArray[np.float64], n: int):
    assert n > 0, 'n needs to be >= 1.'
    assert n <= len(array), 'n is greater than array length.'

@njit
def highv(array: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Calculates the highest values for every ``n`` period in ``array``.

    Args:
        array: :class:`numpy.ndarray` of data.
        n: Length of period.

    Returns:
        :class:`numpy.ndarray` of the highest values for every ``n`` period in
        ``array``.
    """
    if not len(array):
        return np.array(tuple())
    _verify_input(array, n)
    out_len = len(array)
    out = np.array([np.nan for _ in range(out_len)])
    for i in range(n, out_len + 1):
        out[i - 1] = np.max(array[i - n:i])
    return out

@njit
def sumv(array: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Calculates the sums for every ``n`` period in ``array``.

    Args:
        array: :class:`numpy.ndarray` of data.
        n: Length of period.

    Returns:
        :class:`numpy.ndarray` of the sums for every ``n`` period in ``array``.
    """
    if not len(array):
        return np.array(tuple())
    _verify_input(array, n)
    out_len = len(array)
    out = np.array([np.nan for _ in range(out_len)])
    for i in range(n, out_len + 1):
        out[i - 1] = np.sum(array[i - n:i])
    return out

@njit
def returnv(array: NDArray[np.float64], n: int=1) -> NDArray[np.float64]:
    """Calculates returns.

    Args:
        n: Return period. Defaults to 1.

    Returns:
        :class:`numpy.ndarray` of returns.
    """
    if not len(array):
        return np.array(tuple())
    _verify_input(array, n)
    out_len = len(array)
    out = np.array([np.nan for _ in range(out_len)])
    for i in range(n, out_len):
        out[i] = (array[i] - array[i - n]) / array[i - n]
    return out

@njit
def cross(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Checks for crossover of ``a`` above ``b``.

    Args:
        a: :class:`numpy.ndarray` of data.
        b: :class:`numpy.ndarray` of data.

    Returns:
        :class:`numpy.ndarray` containing values of ``1`` when ``a`` crosses
        above ``b``, otherwise values of ``0``.
    """
    assert len(a), 'a cannot be empty.'
    assert len(b), 'b cannot be empty.'
    assert len(a) == len(b), 'a and b must be same length.'
    assert len(a) >= 2, 'a and b must have length >= 2.'
    crossed = np.where(a > b, 1, 0)
    return (sumv(crossed > 0, 2) == 1) * crossed

def _highest(data: BarData):
    values = getattr(data, field)
    return highv(values, period)

def _lowest(data: BarData):
    values = getattr(data, field)
    return lowv(values, period)

def _returns(data: BarData):
    values = getattr(data, field)
    return returnv(values, period)

@pytest.mark.parametrize('array, n, expected', [([3, 3, 4, 2, 5, 6, 1, 3], 3, [np.nan, np.nan, 3, 2, 2, 2, 1, 1]), ([3, 3, 4, 2, 5, 6, 1, 3], 1, [3, 3, 4, 2, 5, 6, 1, 3]), ([4, 3, 2, 1], 4, [np.nan, np.nan, np.nan, 1]), ([1], 1, [1]), ([], 5, [])])
def test_lowv(array, n, expected):
    assert np.array_equal(lowv(np.array(array), n), expected, equal_nan=True)

@pytest.mark.parametrize('array, n, expected', [([3, 3, 4, 2, 5, 6, 1, 3], 3, [np.nan, np.nan, 4, 4, 5, 6, 6, 6]), ([3, 3, 4, 2, 5, 6, 1, 3], 1, [3, 3, 4, 2, 5, 6, 1, 3]), ([4, 3, 2, 1], 4, [np.nan, np.nan, np.nan, 4]), ([1], 1, [1]), ([], 5, [])])
def test_highv(array, n, expected):
    assert np.array_equal(highv(np.array(array), n), expected, equal_nan=True)

@pytest.mark.parametrize('array, n, expected', [([3, 3, 4, 2, 5, 6, 1, 3], 3, [np.nan, np.nan, 10, 9, 11, 13, 12, 10]), ([3, 3, 4, 2, 5, 6, 1, 3], 1, [3, 3, 4, 2, 5, 6, 1, 3]), ([4, 3, 2, 1], 4, [np.nan, np.nan, np.nan, 10]), ([1], 1, [1]), ([], 5, [])])
def test_sumv(array, n, expected):
    assert np.array_equal(sumv(np.array(array), n), expected, equal_nan=True)

@pytest.mark.parametrize('array, n, expected', [([1, 1.5, 1.7, 1.3, 1.2, 1.4], 1, [np.nan, 0.5, 0.13333333, -0.23529412, -0.07692308, 0.16666667]), ([1, 1.5, 1.7, 1.3, 1.2, 1.4], 2, [np.nan, np.nan, 0.7, -0.133333, -0.294118, 0.076923]), ([1], 1, [np.nan]), ([], 5, [])])
def test_returnv(array, n, expected):
    assert np.array_equal(np.round(returnv(np.array(array), n), 6), np.round(expected, 6), equal_nan=True)

@pytest.mark.parametrize('a, b, expected', [([3, 3, 4, 2, 5, 6, 1, 3], [3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 1, 0, 1, 0, 0, 0]), ([3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 4, 2, 5, 6, 1, 3], [0, 0, 0, 1, 0, 0, 1, 0]), ([1, 1], [1, 1], [0, 0])])
def test_cross(a, b, expected):
    assert np.array_equal(cross(np.array(a), np.array(b)), expected, equal_nan=True)

@pytest.mark.parametrize('a, b, expected_msg', [([1, 2, 3], [3, 3, 3, 3], 'a and b must be same length.'), ([3, 3, 3, 3], [1, 2, 3], 'a and b must be same length.'), ([1, 2, 3], [], 'b cannot be empty.'), ([], [1, 2, 3], 'a cannot be empty.'), ([1], [1], 'a and b must have length >= 2.')])
def test_cross_when_invalid_input_then_error(a, b, expected_msg):
    with pytest.raises(AssertionError, match=re.escape(expected_msg)):
        cross(np.array(a), np.array(b))

@pytest.fixture()
def hhv_ind(scope):
    return indicator('hhv', lambda bar_data, n: highv(bar_data.close, n), n=5)

@pytest.fixture()
def llv_ind(scope):
    return indicator('llv', lambda bar_data, n: lowv(bar_data.close, n), n=3)

@pytest.fixture()
def sumv_ind(scope):
    return indicator('sumv', lambda bar_data, n: sumv(bar_data.close, n), n=2)

@pytest.fixture()
def ind_df(data_source_df, hhv_ind, llv_ind, sumv_ind):
    return pd.DataFrame({hhv_ind.name: hhv_ind(data_source_df), llv_ind.name: llv_ind(data_source_df), sumv_ind.name: sumv_ind(data_source_df)})

def hhv(bar_data, period):
    return highv(bar_data.high, period)

