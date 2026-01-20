# Cluster 12

def to_seconds(timeframe: Optional[str]) -> int:
    """Converts a timeframe string to seconds, where ``timeframe`` supports the
    following units:

    - ``"s"``/``"sec"``: seconds
    - ``"m"``/``"min"``: minutes
    - ``"h"``/``"hour"``: hours
    - ``"d"``/``"day"``: days
    - ``"w"``/``"week"``: weeks

    An example timeframe string is ``1h 30m``.

    Returns:
        The converted number of seconds.
    """
    if not timeframe:
        return 0
    seconds = {'sec': 1, 'min': 60, 'hour': 60 * 60, 'day': 24 * 60 * 60, 'week': 7 * 24 * 60 * 60}
    return sum((part[0] * seconds[part[1]] for part in parse_timeframe(timeframe)))

def parse_timeframe(timeframe: str) -> list[tuple[int, str]]:
    """Parses timeframe string with the following units:

    - ``"s"``/``"sec"``: seconds
    - ``"m"``/``"min"``: minutes
    - ``"h"``/``"hour"``: hours
    - ``"d"``/``"day"``: days
    - ``"w"``/``"week"``: weeks

    An example timeframe string is ``1h 30m``.

    Returns:
        ``list`` of ``tuple[int, str]``, where each tuple contains an ``int``
        value and ``str`` unit of one of the following: ``sec``, ``min``,
        ``hour``, ``day``, ``week``.
    """
    parts = _tf_pattern.findall(timeframe)
    if not parts or len(parts) != len(timeframe.split()):
        raise ValueError('Invalid timeframe format.')
    result = []
    units = frozenset(_tf_abbr.values())
    seen_units = set()
    for part in parts:
        unit = part[1].lower()
        if unit in _tf_abbr:
            unit = _tf_abbr[unit]
        if unit not in units:
            raise ValueError('Invalid timeframe format.')
        if unit in seen_units:
            raise ValueError('Invalid timeframe format.')
        result.append((int(part[0]), unit))
        seen_units.add(unit)
    return result

@pytest.mark.parametrize('tf, expected', [('1day 2h 3min', [(1, 'day'), (2, 'hour'), (3, 'min')]), ('10week', [(10, 'week')]), ('3d 20m', [(3, 'day'), (20, 'min')]), ('30s', [(30, 'sec')])])
def test_parse_timeframe_success(tf, expected):
    assert parse_timeframe(tf) == expected

@pytest.mark.parametrize('tf', ['10foo', '20days', '10d 5 m', '1w 2w 3w 5min', 'dd ff cc', 'w d m', '1d5m', '1d 5mm', ''])
def test_parse_timeframe_invalid(tf):
    with pytest.raises(ValueError, match=re.escape('Invalid timeframe format.')):
        parse_timeframe(tf)

@pytest.mark.parametrize('tf, expected', [('1day 2h 3min', 24 * 60 * 60 + 2 * 60 * 60 + 3 * 60), ('10week', 10 * 7 * 24 * 60 * 60), ('3d 20m', 3 * 24 * 60 * 60 + 20 * 60), ('30s', 30), (None, 0)])
def test_to_seconds(tf, expected):
    assert to_seconds(tf) == expected

class TestDataSourceCacheMixin:

    @pytest.mark.usefixtures('scope')
    def test_set_cached(self, alpaca_df, symbols, mock_cache):
        cache_mixin = DataSourceCacheMixin()
        cache_mixin.set_cached(TIMEFRAME, START_DATE, END_DATE, ADJUST, alpaca_df)
        assert len(mock_cache.set.call_args_list) == len(symbols)
        for i, sym in enumerate(symbols):
            expected_cache_key = DataSourceCacheKey(symbol=sym, tf_seconds=to_seconds(TIMEFRAME), start_date=START_DATE, end_date=END_DATE, adjust=ADJUST)
            cache_key, sym_df = mock_cache.set.call_args_list[i].args
            assert cache_key == repr(expected_cache_key)
            assert sym_df.equals(alpaca_df[alpaca_df['symbol'] == sym])

    @pytest.mark.usefixtures('scope')
    @pytest.mark.parametrize('query_symbols', [[], LazyFixture('symbols')])
    def test_get_cached_when_empty(self, mock_cache, query_symbols, request):
        query_symbols = get_fixture(request, query_symbols)
        cache_mixin = DataSourceCacheMixin()
        df, uncached_syms = cache_mixin.get_cached(query_symbols, TIMEFRAME, START_DATE, END_DATE, ADJUST)
        assert df.empty
        assert uncached_syms == query_symbols
        assert len(mock_cache.get.call_args_list) == len(query_symbols)
        for i, sym in enumerate(query_symbols):
            expected_cache_key = DataSourceCacheKey(symbol=sym, tf_seconds=to_seconds(TIMEFRAME), start_date=START_DATE, end_date=END_DATE, adjust=ADJUST)
            cache_key = mock_cache.get.call_args_list[i].args[0]
            assert cache_key == repr(expected_cache_key)

    @pytest.mark.usefixtures('setup_enabled_ds_cache')
    def test_set_and_get_cached(self, alpaca_df, symbols):
        cache_mixin = DataSourceCacheMixin()
        cache_mixin.set_cached(TIMEFRAME, START_DATE, END_DATE, ADJUST, alpaca_df)
        df, uncached_syms = cache_mixin.get_cached(symbols, TIMEFRAME, START_DATE, END_DATE, ADJUST)
        assert df.equals(alpaca_df)
        assert not len(uncached_syms)

    @pytest.mark.usefixtures('setup_enabled_ds_cache')
    def test_set_and_get_cached_when_partial(self, alpaca_df, symbols):
        cache_mixin = DataSourceCacheMixin()
        cached_df = alpaca_df[alpaca_df['symbol'].isin(symbols[:2])]
        cache_mixin.set_cached(TIMEFRAME, START_DATE, END_DATE, ADJUST, cached_df)
        df, uncached_syms = cache_mixin.get_cached(symbols, TIMEFRAME, START_DATE, END_DATE, ADJUST)
        assert df.equals(cached_df)
        assert uncached_syms == symbols[2:]

    @pytest.mark.usefixtures('mock_cache')
    @pytest.mark.parametrize('timeframe, start_date, end_date, error', [('dffdfdf', datetime.strptime('2022-02-02', '%Y-%m-%d'), datetime.strptime('2021-02-02', '%Y-%m-%d'), ValueError), ('1m', 'sdfdfdfg', datetime.strptime('2022-02-02', '%Y-%m-%d'), Exception), ('1m', datetime.strptime('2021-02-02', '%Y-%m-%d'), 'sdfsdf', Exception)])
    def test_set_and_get_cached_when_invalid_times_then_error(self, alpaca_df, symbols, timeframe, start_date, end_date, error):
        cache_mixin = DataSourceCacheMixin()
        with pytest.raises(error):
            cache_mixin.set_cached(timeframe, start_date, end_date, ADJUST, alpaca_df)
        with pytest.raises(error):
            cache_mixin.get_cached(symbols, timeframe, start_date, end_date, ADJUST)

    def test_set_and_get_cached_when_cache_disabled(self, alpaca_df, symbols):
        cache_mixin = DataSourceCacheMixin()
        cache_mixin.set_cached(TIMEFRAME, START_DATE, END_DATE, ADJUST, alpaca_df)
        df, uncached_syms = cache_mixin.get_cached(symbols, TIMEFRAME, START_DATE, END_DATE, ADJUST)
        assert df.empty
        assert uncached_syms == symbols

