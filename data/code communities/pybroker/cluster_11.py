# Cluster 11

@njit
def intraday_intensity(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], volume: NDArray[np.float64], lookback: int, smoothing: float=0.0) -> NDArray[np.float64]:
    """Computes Intraday Intensity.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        volume: Trading volume.
        lookback: Number of lookback bars.
        smoothing: Amount of smoothing; <= 1 for none. Defaults to ``0``.

    Returns:
        :class:`numpy.ndarray` of computed values.
    """
    return _flow(high, low, close, volume, lookback, smoothing, 'intraday')

@njit
def _flow(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], volume: NDArray[np.float64], lookback: int, smoothing: float, flow_type: Literal['intraday', 'money_flow']) -> NDArray[np.float64]:
    assert len(high) == len(low) and len(high) == len(close) and (len(high) == len(volume))
    assert lookback > 0
    assert smoothing >= 0
    n = len(close)
    front_bad = lookback - 1
    for first_volume in range(n):
        if volume[first_volume] > 0:
            break
    front_bad += first_volume
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(first_volume, n):
        if high[i] > low[i]:
            output[i] = 100.0 * (2.0 * close[i] - high[i] - low[i]) / (high[i] - low[i]) * volume[i]
        else:
            output[i] = 0.0
    if lookback > 1:
        for i in range(n - 1, front_bad - 1, -1):
            total = 0.0
            for j in range(lookback):
                total += output[i - j]
            output[i] = total / lookback
    if flow_type == 'money_flow':
        for i in range(front_bad, n):
            total = 0.0
            for j in range(lookback):
                total += volume[i - j]
            total /= lookback
            if total > 0.0:
                output[i] /= total
            else:
                output[i] = 0.0
    elif smoothing > 1:
        alpha = 2.0 / (smoothing + 1.0)
        smoothed = volume[first_volume]
        for i in range(first_volume, n):
            smoothed = alpha * volume[i] + (1.0 - alpha) * smoothed
            if smoothed > 0.0:
                output[i] /= smoothed
            else:
                output[i] = 0.0
    for i in range(front_bad):
        output[i] = 0.0
    return output

@njit
def money_flow(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], volume: NDArray[np.float64], lookback: int, smoothing: float=0.0) -> NDArray[np.float64]:
    """Computes Chaikin's Money Flow.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        volume: Trading volume.
        lookback: Number of lookback bars.
        smoothing: Amount of smoothing; <= 1 for none. Defaults to ``0``.

    Returns:
        :class:`numpy.ndarray` of computed values.
    """
    return _flow(high, low, close, volume, lookback, smoothing, 'money_flow')

