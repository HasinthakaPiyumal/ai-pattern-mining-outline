# Cluster 10

@njit
def aroon_up(high: NDArray[np.float64], low: NDArray[np.float64], lookback: int) -> NDArray[np.float64]:
    """Computes Aroon Upward Trend.

    Args:
        high: High prices.
        low: Low prices.
        lookback: Number of lookback bars.

    Returns:
        :class:`numpy.ndarray` of computed values.
    """
    return _aroon(high, low, lookback, 'up')

@njit
def _aroon(high: NDArray[np.float64], low: NDArray[np.float64], lookback: int, aroon_type: Literal['up', 'down', 'diff']) -> NDArray[np.float64]:
    assert len(high) == len(low)
    assert lookback > 0
    n = len(high)
    output = np.zeros(n)
    if aroon_type == 'up' or aroon_type == 'down':
        output[0] = 50
    elif aroon_type == 'diff':
        output[0] = 0
    for i in range(1, n):
        if aroon_type == 'up' or aroon_type == 'diff':
            i_max = i
            x_max = high[i]
            for i in range(i - 1, i - lookback - 1, -1):
                if i < 0:
                    break
                if high[i] > x_max:
                    x_max = high[i]
                    i_max = i
        if aroon_type == 'down' or aroon_type == 'diff':
            i_min = i
            x_min = low[i]
            for i in range(i - 1, i - lookback - 1, -1):
                if i < 0:
                    break
                if low[i] < x_min:
                    x_min = low[i]
                    i_min = i
        if aroon_type == 'up':
            output[i] = 100 * (lookback - (i - i_max)) / lookback
        elif aroon_type == 'down':
            output[i] = 100 * (lookback - (i - i_min)) / lookback
        else:
            max_val = 100 * (lookback - (i - i_max)) / lookback
            min_val = 100 * (lookback - (i - i_min)) / lookback
            output[i] = max_val - min_val
    return output

@njit
def aroon_down(high: NDArray[np.float64], low: NDArray[np.float64], lookback: int) -> NDArray[np.float64]:
    """Computes Aroon Downward Trend.

    Args:
        high: High prices.
        low: Low prices.
        lookback: Number of lookback bars.

    Returns:
        :class:`numpy.ndarray` of computed values.
    """
    return _aroon(high, low, lookback, 'down')

@njit
def aroon_diff(high: NDArray[np.float64], low: NDArray[np.float64], lookback: int) -> NDArray[np.float64]:
    """Computes Aroon Upward Trend minus Aroon Downward Trend.

    Args:
        high: High prices.
        low: Low prices.
        lookback: Number of lookback bars.

    Returns:
        :class:`numpy.ndarray` of computed values.
    """
    return _aroon(high, low, lookback, 'diff')

