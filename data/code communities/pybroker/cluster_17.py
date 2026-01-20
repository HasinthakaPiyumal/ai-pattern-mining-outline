# Cluster 17

class IndicatorsMixin:
    """Mixin implementing indicator related functionality."""

    def compute_indicators(self, df: pd.DataFrame, indicator_syms: Iterable[IndicatorSymbol], cache_date_fields: Optional[CacheDateFields], disable_parallel: bool) -> dict[IndicatorSymbol, pd.Series]:
        """Computes indicator data for the provided
        :class:`pybroker.common.IndicatorSymbol` pairs.

        Args:
            df: :class:`pandas.DataFrame` used to compute the indicator values.
            indicator_syms: ``Iterable`` of
                :class:`pybroker.common.IndicatorSymbol` pairs of indicators
                to compute.
            cache_date_fields: Date fields used to key cache data. Pass
                ``None`` to disable caching.
            disable_parallel: If ``True``, indicator data is computed
                serially for all :class:`pybroker.common.IndicatorSymbol`
                pairs. If ``False``, indicator data is computed in parallel
                using multiple processes.

        Returns:
            ``dict`` mapping each :class:`pybroker.common.IndicatorSymbol` pair
            to a computed :class:`pandas.Series` of indicator values.
        """
        if not indicator_syms or df.empty:
            return {}
        scope = StaticScope.instance()
        indicator_data, uncached_ind_syms = self._get_cached_indicators(indicator_syms, cache_date_fields)
        if not uncached_ind_syms:
            scope.logger.loaded_indicator_data()
            scope.logger.info_loaded_indicator_data(indicator_syms)
            return indicator_data
        if indicator_data:
            scope.logger.info_loaded_indicator_data(indicator_data.keys())
        scope.logger.indicator_data_start(uncached_ind_syms)
        scope.logger.info_indicator_data_start(uncached_ind_syms)
        sym_data: dict[str, dict[str, Optional[NDArray]]] = defaultdict(dict)
        for _, sym in uncached_ind_syms:
            if sym in sym_data:
                continue
            data = df[df[DataCol.SYMBOL.value] == sym]
            for col in scope.all_data_cols:
                if col not in data.columns:
                    sym_data[sym][col] = None
                    continue
                sym_data[sym][col] = data[col].to_numpy()
        for i, (ind_sym, series) in enumerate(self._run_indicators(sym_data, uncached_ind_syms, disable_parallel)):
            indicator_data[ind_sym] = series
            self._set_cached_indicator(series, ind_sym, cache_date_fields)
            scope.logger.indicator_data_loading(i + 1)
        return indicator_data

    def _get_cached_indicators(self, indicator_syms: Iterable[IndicatorSymbol], cache_date_fields: Optional[CacheDateFields]) -> tuple[dict[IndicatorSymbol, pd.Series], list[IndicatorSymbol]]:
        indicator_syms = sorted(indicator_syms)
        indicator_data: dict[IndicatorSymbol, pd.Series] = {}
        if cache_date_fields is None:
            return (indicator_data, indicator_syms)
        scope = StaticScope.instance()
        if scope.indicator_cache is None:
            return (indicator_data, indicator_syms)
        uncached_ind_syms = []
        for ind_sym in indicator_syms:
            cache_key = IndicatorCacheKey(symbol=ind_sym.symbol, ind_name=ind_sym.ind_name, **asdict(cache_date_fields))
            scope.logger.debug_get_indicator_cache(cache_key)
            data = scope.indicator_cache.get(repr(cache_key))
            if data is not None:
                indicator_data[ind_sym] = data
            else:
                uncached_ind_syms.append(ind_sym)
        return (indicator_data, uncached_ind_syms)

    def _set_cached_indicator(self, series: pd.Series, ind_sym: IndicatorSymbol, cache_date_fields: Optional[CacheDateFields]):
        if cache_date_fields is None:
            return
        scope = StaticScope.instance()
        if scope.indicator_cache is None:
            return
        cache_key = IndicatorCacheKey(symbol=ind_sym.symbol, ind_name=ind_sym.ind_name, **asdict(cache_date_fields))
        scope.logger.debug_set_indicator_cache(cache_key)
        scope.indicator_cache.set(repr(cache_key), series)

    def _run_indicators(self, sym_data: Mapping[str, Mapping[str, Optional[NDArray]]], ind_syms: Collection[IndicatorSymbol], disable_parallel: bool) -> Iterable[tuple[IndicatorSymbol, pd.Series]]:
        fns = {}
        for ind_name, _ in ind_syms:
            if ind_name in fns:
                continue
            fns[ind_name] = _decorate_indicator_fn(ind_name)
        scope = StaticScope.instance()

        def args_fn(ind_name, sym):
            return {'symbol': sym, 'ind_name': ind_name, 'custom_col_data': {col: sym_data[sym][col] for col in scope.custom_data_cols}, **{col: sym_data[sym][col] for col in scope.default_data_cols}}
        if disable_parallel or len(ind_syms) == 1:
            scope.logger.debug_compute_indicators(is_parallel=False)
            return tuple((fns[ind_name](**args_fn(ind_name, sym)) for ind_name, sym in ind_syms))
        else:
            scope.logger.debug_compute_indicators(is_parallel=True)
            with default_parallel() as parallel:
                return parallel((delayed(fns[ind_name])(**args_fn(ind_name, sym)) for ind_name, sym in ind_syms))

def _decorate_indicator_fn(ind_name: str):
    fn = StaticScope.instance().get_indicator(ind_name).__call__

    def decorated_indicator_fn(symbol: str, ind_name: str, date: NDArray[np.datetime64], open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], volume: Optional[NDArray[np.float64]], vwap: Optional[NDArray[np.float64]], custom_col_data: Mapping[str, Optional[NDArray]]) -> tuple[IndicatorSymbol, pd.Series]:
        bar_data = BarData(date=date, open=open, high=high, low=low, close=close, volume=volume, vwap=vwap, **custom_col_data)
        series = fn(bar_data)
        return (IndicatorSymbol(ind_name, symbol), series)
    return decorated_indicator_fn

def default_parallel() -> Parallel:
    """Returns a :class:`joblib.Parallel` instance with ``n_jobs`` equal to
    the number of CPUs on the host machine.
    """
    return Parallel(n_jobs=os.cpu_count(), prefer='processes', backend='loky')

def test_default_parallel():
    assert type(default_parallel()) is Parallel

