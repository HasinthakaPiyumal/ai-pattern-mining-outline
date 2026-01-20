# Cluster 18

def highest(name: str, field: str, period: int) -> Indicator:
    """Creates a rolling high :class:`.Indicator`.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field for computing the rolling
            high.
        period: Lookback period.

    Returns:
        Rolling high :class:`.Indicator`.
    """

    def _highest(data: BarData):
        values = getattr(data, field)
        return highv(values, period)
    return indicator(name, _highest)

def indicator(name: str, fn: Callable[..., NDArray[np.float64]], **kwargs) -> Indicator:
    """Creates an :class:`.Indicator` instance and registers it globally with
    ``name``.

    Args:
        name: Name for referencing the indicator globally.
        fn: ``Callable[[BarData, ...], NDArray[float]]`` used to compute the
            series of indicator values.
        \\**kwargs: Additional arguments to pass to ``fn``.

    Returns:
        :class:`.Indicator` instance.
    """
    scope = StaticScope.instance()
    indicator = Indicator(name, fn, kwargs)
    scope.set_indicator(indicator)
    return indicator

def lowest(name: str, field: str, period: int) -> Indicator:
    """Creates a rolling low :class:`.Indicator`.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field for computing the rolling
            low.
        period: Lookback period.

    Returns:
        Rolling low :class:`.Indicator`.
    """

    def _lowest(data: BarData):
        values = getattr(data, field)
        return lowv(values, period)
    return indicator(name, _lowest)

def returns(name: str, field: str, period: int=1) -> Indicator:
    """Creates a rolling returns :class:`.Indicator`.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field for computing the rolling
            returns.
        period: Returns period. Defaults to 1.

    Returns:
        Rolling returns :class:`.Indicator`.
    """

    def _returns(data: BarData):
        values = getattr(data, field)
        return returnv(values, period)
    return indicator(name, _returns)

def detrended_rsi(name: str, field: str, short_length: int, long_length: int, reg_length: int) -> Indicator:
    """Detrended Relative Strength Index (RSI).

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        short_length: Lookback for the short-term RSI.
        long_length: Lookback for the long-term RSI.
        reg_length: Number of bars used for linear regressions.

    Returns:
        Detrended RSI :class:`.Indicator`.
    """

    def _detrended_rsi(data: BarData):
        values = getattr(data, field)
        return vect.detrended_rsi(values, short_length=short_length, long_length=long_length, reg_length=reg_length)
    return indicator(name, _detrended_rsi)

def macd(name: str, short_length: int, long_length: int, smoothing: float=0.0, scale: float=1.0) -> Indicator:
    """Moving Average Convergence Divergence.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        short_length: Short-term lookback.
        long_length: Long-term lookback.
        smoothing: Compute MACD minus smoothed if >= 2.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        Moving Average Convergence Divergence :class:`.Indicator`.
    """

    def _macd(data: BarData):
        return vect.macd(high=data.high, low=data.low, close=data.close, short_length=short_length, long_length=long_length, smoothing=smoothing, scale=scale)
    return indicator(name, _macd)

def stochastic(name: str, lookback: int, smoothing: int=0) -> Indicator:
    """Stochastic.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        smoothing: Number of times the raw stochastic is smoothed, either 0,
            1, or 2 times. Defaults to ``0``.

    Returns:
        Stochastic :class:`.Indicator`.
    """

    def _stochastic(data: BarData):
        return vect.stochastic(high=data.high, low=data.low, close=data.close, lookback=lookback, smoothing=smoothing)
    return indicator(name, _stochastic)

def stochastic_rsi(name: str, field: str, rsi_lookback: int, sto_lookback: int, smoothing: float=0.0) -> Indicator:
    """Stochastic Relative Strength Index (RSI).

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        rsi_lookback: Lookback length for RSI calculation.
        sto_lookback: Lookback length for Stochastic calculation.
        smoothing: Amount of smoothing; <= 1 for none. Defaults to ``0``.

    Returns:
        Stochastic RSI :class:`.Indicator`.
    """

    def _stochastic_rsi(data: BarData):
        values = getattr(data, field)
        return vect.stochastic_rsi(values, rsi_lookback=rsi_lookback, sto_lookback=sto_lookback, smoothing=smoothing)
    return indicator(name, _stochastic_rsi)

def linear_trend(name: str, field: str, lookback: int, atr_length: int, scale: float=1.0) -> Indicator:
    """Linear Trend Strength.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        lookback: Number of lookback bars.
        atr_length: Lookback length used for Average True Range (ATR)
            normalization.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        Linear Trend Strength :class:`.Indicator`.
    """

    def _linear_trend(data: BarData):
        values = getattr(data, field)
        return vect.linear_trend(values, high=data.high, low=data.low, close=data.close, lookback=lookback, atr_length=atr_length, scale=scale)
    return indicator(name, _linear_trend)

def quadratic_trend(name: str, field: str, lookback: int, atr_length: int, scale: float=1.0) -> Indicator:
    """Quadratic Trend Strength.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        lookback: Number of lookback bars.
        atr_length: Lookback length used for Average True Range (ATR)
            normalization.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        Quadratic Trend Strength :class:`.Indicator`.
    """

    def _quadratic_trend(data: BarData):
        values = getattr(data, field)
        return vect.quadratic_trend(values, high=data.high, low=data.low, close=data.close, lookback=lookback, atr_length=atr_length, scale=scale)
    return indicator(name, _quadratic_trend)

def cubic_trend(name: str, field: str, lookback: int, atr_length: int, scale: float=1.0) -> Indicator:
    """Cubic Trend Strength.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        lookback: Number of lookback bars.
        atr_length: Lookback length used for Average True Range (ATR)
            normalization.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        Cubic Trend Strength :class:`.Indicator`.
    """

    def _cubic_trend(data: BarData):
        values = getattr(data, field)
        return vect.cubic_trend(values, high=data.high, low=data.low, close=data.close, lookback=lookback, atr_length=atr_length, scale=scale)
    return indicator(name, _cubic_trend)

def adx(name: str, lookback: int) -> Indicator:
    """Average Directional Movement Index.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.

    Returns:
        Average Directional Movement Index :class:`.Indicator`.
    """

    def _adx(data: BarData):
        return vect.adx(high=data.high, low=data.low, close=data.close, lookback=lookback)
    return indicator(name, _adx)

def aroon_up(name: str, lookback: int) -> Indicator:
    """Aroon Upward Trend.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.

    Returns:
        Aroon Upward Trend :class:`.Indicator`.
    """

    def _aroon_up(data: BarData):
        return vect.aroon_up(high=data.high, low=data.low, lookback=lookback)
    return indicator(name, _aroon_up)

def aroon_down(name: str, lookback: int) -> Indicator:
    """Aroon Downward Trend.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.

    Returns:
        Aroon Downward Trend :class:`.Indicator`.
    """

    def _aroon_down(data: BarData):
        return vect.aroon_down(high=data.high, low=data.low, lookback=lookback)
    return indicator(name, _aroon_down)

def aroon_diff(name: str, lookback: int) -> Indicator:
    """Aroon Upward Trend minus Aroon Downward Trend.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.

    Returns:
        Aroon Upward Trend minus Aroon Downward Trend :class:`.Indicator`.
    """

    def _aroon_diff(data: BarData):
        return vect.aroon_diff(high=data.high, low=data.low, lookback=lookback)
    return indicator(name, _aroon_diff)

def close_minus_ma(name: str, lookback: int, atr_length: int, scale: float=1.0) -> Indicator:
    """Close Minus Moving Average.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        atr_length: Lookback length used for Average True Range (ATR)
            normalization.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        Close Minus Moving Average :class:`.Indicator`.
    """

    def _close_minus_ma(data: BarData):
        return vect.close_minus_ma(high=data.high, low=data.low, close=data.close, lookback=lookback, atr_length=atr_length, scale=scale)
    return indicator(name, _close_minus_ma)

def linear_deviation(name: str, field: str, lookback: int, scale: float=0.6) -> Indicator:
    """Deviation from Linear Trend.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        Deviation from Linear Trend :class:`.Indicator`.
    """

    def _linear_deviation(data: BarData):
        values = getattr(data, field)
        return vect.linear_deviation(values, lookback=lookback, scale=scale)
    return indicator(name, _linear_deviation)

def quadratic_deviation(name: str, field: str, lookback: int, scale: float=0.6) -> Indicator:
    """Deviation from Quadratic Trend.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        Deviation from Quadratic Trend :class:`.Indicator`.
    """

    def _quadratic_deviation(data: BarData):
        values = getattr(data, field)
        return vect.quadratic_deviation(values, lookback=lookback, scale=scale)
    return indicator(name, _quadratic_deviation)

def cubic_deviation(name: str, field: str, lookback: int, scale: float=0.6) -> Indicator:
    """Deviation from Cubic Trend.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        Deviation from Cubic Trend :class:`.Indicator`.
    """

    def _cubic_deviation(data: BarData):
        values = getattr(data, field)
        return vect.cubic_deviation(values, lookback=lookback, scale=scale)
    return indicator(name, _cubic_deviation)

def price_intensity(name: str, smoothing: float=0.0, scale: float=0.8) -> Indicator:
    """Price Intensity.

    Args:
        name: Indicator name.
        smoothing: Amount of smoothing. Defaults to ``0``.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.8``.

    Returns:
        Price Intensity :class:`.Indicator`.
    """

    def _price_intensity(data: BarData):
        return vect.price_intensity(open=data.open, high=data.high, low=data.low, close=data.close, smoothing=smoothing, scale=scale)
    return indicator(name, _price_intensity)

def price_change_oscillator(name: str, short_length: int, multiplier: int, scale: float=4.0) -> Indicator:
    """Price Change Oscillator.

    Args:
        name: Indicator name.
        short_length: Number of short lookback bars.
        multiplier: Multiplier used to compute number of long lookback bars =
            ``multiplier * short_length``.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``4.0``.

    Returns:
        Price Change Oscillator :class:`.Indicator`.
    """

    def _price_change_oscillator(data: BarData):
        return vect.price_change_oscillator(high=data.high, low=data.low, close=data.close, short_length=short_length, multiplier=multiplier, scale=scale)
    return indicator(name, _price_change_oscillator)

def intraday_intensity(name: str, lookback: int, smoothing: float=0.0) -> Indicator:
    """Intraday Intensity.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        smoothing: Amount of smoothing; <= 1 for none. Defaults to ``0``.

    Returns:
        Intraday Intensity :class:`.Indicator`.
    """

    def _intraday_intensity(data: BarData):
        return vect.intraday_intensity(high=data.high, low=data.low, close=data.close, volume=data.volume, lookback=lookback, smoothing=smoothing)
    return indicator(name, _intraday_intensity)

def money_flow(name: str, lookback: int, smoothing: float=0.0) -> Indicator:
    """Chaikin's Money Flow.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        smoothing: Amount of smoothing; <= 1 for none. Defaults to ``0``.

    Returns:
        Chaikin's Money Flow :class:`.Indicator`.
    """

    def _money_flow(data: BarData):
        return vect.money_flow(high=data.high, low=data.low, close=data.close, volume=data.volume, lookback=lookback, smoothing=smoothing)
    return indicator(name, _money_flow)

def reactivity(name: str, lookback: int, smoothing: float=0.0, scale: float=0.6) -> Indicator:
    """Reactivity.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        smoothing: Smoothing multiplier.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        Reactivity :class:`.Indicator`.
    """

    def _reactivity(data: BarData):
        return vect.reactivity(high=data.high, low=data.low, close=data.close, volume=data.volume, lookback=lookback, smoothing=smoothing, scale=scale)
    return indicator(name, _reactivity)

def price_volume_fit(name: str, lookback: int, scale: float=9.0) -> Indicator:
    """Price Volume Fit.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``9.0``.

    Returns:
        Price Volume Fit :class:`.Indicator`.
    """

    def _price_volume_fit(data: BarData):
        return vect.price_volume_fit(close=data.close, volume=data.volume, lookback=lookback, scale=scale)
    return indicator(name, _price_volume_fit)

def volume_weighted_ma_ratio(name: str, lookback: int, scale: float=1.0) -> Indicator:
    """Volume-Weighted Moving Average Ratio.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        Volume-Weighted Moving Average Ratio :class:`.Indicator`.
    """

    def _volume_weighted_ma_ratio(data: BarData):
        return vect.volume_weighted_ma_ratio(close=data.close, volume=data.volume, lookback=lookback, scale=scale)
    return indicator(name, _volume_weighted_ma_ratio)

def normalized_on_balance_volume(name: str, lookback: int, scale: float=0.6) -> Indicator:
    """Normalized On-Balance Volume.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        Normalized On-Balance Volume :class:`.Indicator`.
    """

    def _normalized_on_balance_volume(data: BarData):
        return vect.normalized_on_balance_volume(close=data.close, volume=data.volume, lookback=lookback, scale=scale)
    return indicator(name, _normalized_on_balance_volume)

def delta_on_balance_volume(name: str, lookback: int, delta_length: int=0, scale: float=0.6) -> Indicator:
    """Delta On-Balance Volume.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        delta_length: Lag for differencing.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        Delta On-Balance Volume :class:`.Indicator`.
    """

    def _delta_on_balance_volume(data: BarData):
        return vect.delta_on_balance_volume(close=data.close, volume=data.volume, lookback=lookback, delta_length=delta_length, scale=scale)
    return indicator(name, _delta_on_balance_volume)

def normalized_positive_volume_index(name: str, lookback: int, scale: float=0.5) -> Indicator:
    """Normalized Positive Volume Index.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.5``.

    Returns:
        Normalized Positive Volume Index :class:`.Indicator`.
    """

    def _normalized_positive_volume_index(data: BarData):
        return vect.normalized_positive_volume_index(close=data.close, volume=data.volume, lookback=lookback, scale=scale)
    return indicator(name, _normalized_positive_volume_index)

def normalized_negative_volume_index(name: str, lookback: int, scale: float=0.5) -> Indicator:
    """Normalized Negative Volume Index.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.5``.

    Returns:
        Normalized Negative Volume Index :class:`.Indicator`.
    """

    def _normalized_negative_volume_index(data: BarData):
        return vect.normalized_negative_volume_index(close=data.close, volume=data.volume, lookback=lookback, scale=scale)
    return indicator(name, _normalized_negative_volume_index)

def volume_momentum(name: str, short_length: int, multiplier: int=2, scale: float=3.0) -> Indicator:
    """Volume Momentum.

    Args:
        name: Indicator name.
        short_length: Number of short lookback bars.
        multiplier: Lookback multiplier. Defaults to ``2``.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``3.0``.

    Returns:
        Volume Momentum :class:`.Indicator`.
    """

    def _volume_momentum(data: BarData):
        return vect.volume_momentum(volume=data.volume, short_length=short_length, multiplier=multiplier, scale=scale)
    return indicator(name, _volume_momentum)

def laguerre_rsi(name: str, fe_length: int=13) -> Indicator:
    """Laguerre Relative Strength Index (RSI).

    Args:
        name: Indicator name.
        fe_length: Fractal Energy length. Defaults to ``13``.

    Returns:
        Laguerre RSI :class:`.Indicator`.
    """

    def _laguerre_rsi(data: BarData):
        return vect.laguerre_rsi(open=data.open, high=data.high, low=data.low, close=data.close, fe_length=fe_length)
    return indicator(name, _laguerre_rsi)

@pytest.mark.usefixtures('setup_teardown')
def test_indicator():
    ind = indicator('llv', lowv)
    assert isinstance(ind, Indicator)
    assert ind.name == 'llv'

@pytest.mark.usefixtures('setup_teardown')
class TestIndicator:

    def test_call_with_kwargs(self, hhv_ind, data_source_df):
        data = hhv_ind(data_source_df)
        assert len(data) == len(data_source_df['date'])
        assert isinstance(data.index[0], pd.Timestamp)

    def test_call_when_invalid_shape_then_error(self, data_source_df):

        def invalid_shape(_data):
            return np.array([[1, 2, 3], [4, 5, 6]])
        ind_invalid_shape = indicator('invalid_shape', invalid_shape)
        with pytest.raises(ValueError, match=re.escape('Indicator invalid_shape must return a one-dimensional array.')):
            ind_invalid_shape(data_source_df)

    def test_iqr(self, hhv_ind, data_source_df):
        assert isinstance(hhv_ind.iqr(data_source_df), float)

    def test_relative_entropy(self, hhv_ind, data_source_df):
        assert isinstance(hhv_ind.relative_entropy(data_source_df), float)

    def test_repr(self, hhv_ind):
        assert repr(hhv_ind) == "Indicator('hhv', {'n': 5})"

@pytest.mark.parametrize('fn, values, period, expected', [(highest, [3, 3, 4, 2, 5, 6, 1, 3], 3, [np.nan, np.nan, 4, 4, 5, 6, 6, 6]), (highest, [3, 3, 4, 2, 5, 6, 1, 3], 1, [3, 3, 4, 2, 5, 6, 1, 3]), (highest, [4, 3, 2, 1], 4, [np.nan, np.nan, np.nan, 4]), (highest, [1], 1, [1]), (lowest, [3, 3, 4, 2, 5, 6, 1, 3], 3, [np.nan, np.nan, 3, 2, 2, 2, 1, 1]), (lowest, [3, 3, 4, 2, 5, 6, 1, 3], 1, [3, 3, 4, 2, 5, 6, 1, 3]), (lowest, [4, 3, 2, 1], 4, [np.nan, np.nan, np.nan, 1]), (lowest, [1], 1, [1]), (returns, [1, 1.5, 1.7, 1.3, 1.2, 1.4], 1, [np.nan, 0.5, 0.13333333, -0.23529412, -0.07692308, 0.16666667]), (returns, [1, 1.5, 1.7, 1.3, 1.2, 1.4], 2, [np.nan, np.nan, 0.7, -0.133333, -0.294118, 0.076923]), (returns, [1], 1, [np.nan]), (returns, [], 5, [])])
def test_wrappers(fn, values, period, expected):
    n = len(values)
    dates = pd.date_range(start='1/1/2018', end='1/1/2019').to_numpy()[:n]
    bar_data = BarData(date=dates, open=np.zeros(n), high=np.zeros(n), low=np.zeros(n), close=np.array(values), volume=None, vwap=None)
    indicator = fn('my_indicator', 'close', period)
    assert isinstance(indicator, Indicator)
    assert indicator.name == 'my_indicator'
    series = indicator(bar_data)
    assert np.array_equal(series.index.to_numpy(), dates)
    assert np.array_equal(np.round(series.values, 6), np.round(expected, 6), equal_nan=True)

@pytest.mark.parametrize('fn, args', [(detrended_rsi, {'field': 'close', 'short_length': 5, 'long_length': 10, 'reg_length': 20}), (macd, {'short_length': 5, 'long_length': 10, 'smoothing': 2.0}), (stochastic, {'lookback': 10, 'smoothing': 2}), (stochastic_rsi, {'field': 'close', 'rsi_lookback': 10, 'sto_lookback': 10, 'smoothing': 2.0}), (linear_trend, {'field': 'close', 'lookback': 10, 'atr_length': 20, 'scale': 0.5}), (quadratic_trend, {'field': 'close', 'lookback': 10, 'atr_length': 20, 'scale': 0.5}), (cubic_trend, {'field': 'close', 'lookback': 10, 'atr_length': 20, 'scale': 0.5}), (adx, {'lookback': 10}), (aroon_up, {'lookback': 10}), (aroon_down, {'lookback': 10}), (aroon_diff, {'lookback': 10}), (close_minus_ma, {'lookback': 10, 'atr_length': 20, 'scale': 0.5}), (linear_deviation, {'field': 'close', 'lookback': 10, 'scale': 0.5}), (quadratic_deviation, {'field': 'close', 'lookback': 10, 'scale': 0.5}), (cubic_deviation, {'field': 'close', 'lookback': 10, 'scale': 0.5}), (price_intensity, {'smoothing': 1.0, 'scale': 0.5}), (price_change_oscillator, {'short_length': 5, 'multiplier': 3, 'scale': 0.5}), (intraday_intensity, {'lookback': 10, 'smoothing': 1.0}), (money_flow, {'lookback': 10, 'smoothing': 1.0}), (reactivity, {'lookback': 10, 'smoothing': 1.0, 'scale': 0.5}), (price_volume_fit, {'lookback': 10, 'scale': 0.5}), (volume_weighted_ma_ratio, {'lookback': 10, 'scale': 0.5}), (normalized_on_balance_volume, {'lookback': 10, 'scale': 0.5}), (delta_on_balance_volume, {'lookback': 10, 'delta_length': 5, 'scale': 0.5}), (normalized_positive_volume_index, {'lookback': 10, 'scale': 0.5}), (normalized_negative_volume_index, {'lookback': 10, 'scale': 0.5}), (volume_momentum, {'short_length': 5, 'multiplier': 2, 'scale': 2.0}), (laguerre_rsi, {'fe_length': 20})])
def test_indicators(fn, args):
    dates = pd.date_range(start='1/1/2018', end='1/1/2019').to_numpy()
    n = len(dates)
    bar_data = BarData(date=dates, open=np.random.rand(n), high=np.random.rand(n), low=np.random.rand(n), close=np.random.rand(n), volume=np.random.rand(n), vwap=None)
    indicator = fn(fn.__name__, **args)
    assert isinstance(indicator, Indicator)
    assert indicator.name == fn.__name__
    series = indicator(bar_data)
    assert len(series) == n
    assert np.array_equal(series.index.to_numpy(), dates)

