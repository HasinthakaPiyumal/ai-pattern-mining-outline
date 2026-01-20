# Cluster 0

class BacktestMixin:
    """Mixin implementing backtesting functionality."""

    def backtest_executions(self, config: StrategyConfig, executions: set[Execution], before_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]], after_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]], sessions: Mapping[str, MutableMapping], models: Mapping[ModelSymbol, TrainedModel], indicator_data: Mapping[IndicatorSymbol, pd.Series], test_data: pd.DataFrame, portfolio: Portfolio, pos_size_handler: Optional[Callable[[PosSizeContext], None]], exit_dates: Mapping[str, np.datetime64], train_only: bool=False, slippage_model: Optional[SlippageModel]=None, enable_fractional_shares: bool=False, round_fill_price: bool=True, warmup: Optional[int]=None) -> dict[str, pd.DataFrame]:
        """Backtests a ``set`` of :class:`.Execution`\\ s that implement
        trading logic.

        Args:
            config: :class:`pybroker.config.StrategyConfig`.
            executions: :class:`.Execution`\\ s to run.
            sessions: :class:`Mapping` of symbols to :class:`Mapping` of custom
                data that persists for every bar during the
                :class:`.Execution`.
            models: :class:`Mapping` of :class:`pybroker.common.ModelSymbol`
                pairs to :class:`pybroker.common.TrainedModel`\\ s.
            indicator_data: :class:`Mapping` of
                :class:`pybroker.common.IndicatorSymbol` pairs to
                :class:`pandas.Series` of :class:`pybroker.indicator.Indicator`
                values.
            test_data: :class:`pandas.DataFrame` of test data.
            portfolio: :class:`pybroker.portfolio.Portfolio`.
            pos_size_handler: :class:`Callable` that sets position sizes when
                placing orders for buy and sell signals.
            exit_dates: :class:`Mapping` of symbols to exit dates.
            train_only: Whether the backtest is run with trading rules or
                only trains models.
            enable_fractional_shares: Whether to enable trading fractional
                shares.
            round_fill_price: Whether to round fill prices to the nearest cent.
            warmup: Number of bars that need to pass before running the
                executions.

        Returns:
            Dictionary of :class:`pandas.DataFrame`\\ s containing bar data,
            indicator data, and model predictions for each symbol when
            :attr:`pybroker.config.StrategyConfig.return_signals` is ``True``.
        """
        test_dates = get_unique_sorted_dates(test_data[DataCol.DATE.value])
        test_syms = sorted(test_data[DataCol.SYMBOL.value].unique())
        test_data = test_data.reset_index(drop=True).set_index([DataCol.SYMBOL.value, DataCol.DATE.value]).sort_index()
        col_scope = ColumnScope(test_data)
        ind_scope = IndicatorScope(indicator_data, test_dates)
        input_scope = ModelInputScope(col_scope, ind_scope, models)
        pred_scope = PredictionScope(models, input_scope)
        if train_only:
            if config.return_signals:
                return get_signals(test_syms, col_scope, ind_scope, pred_scope)
            return {}
        sym_end_index: dict[str, int] = defaultdict(int)
        price_scope = PriceScope(col_scope, sym_end_index, round_fill_price)
        pending_order_scope = PendingOrderScope()
        exec_ctxs: dict[str, ExecContext] = {}
        exec_fns: dict[str, Callable[[ExecContext], None]] = {}
        for sym in test_syms:
            for exec in executions:
                if sym not in exec.symbols:
                    continue
                exec_ctxs[sym] = ExecContext(symbol=sym, config=config, portfolio=portfolio, col_scope=col_scope, ind_scope=ind_scope, input_scope=input_scope, pred_scope=pred_scope, pending_order_scope=pending_order_scope, models=models, sym_end_index=sym_end_index, session=sessions[sym])
                if exec.fn is not None:
                    exec_fns[sym] = exec.fn
        sym_exec_dates = {sym: frozenset(test_data.loc[pd.IndexSlice[sym, :]].index.values) for sym in exec_ctxs.keys()}
        cover_sched: dict[np.datetime64, list[ExecResult]] = defaultdict(list)
        buy_sched: dict[np.datetime64, list[ExecResult]] = defaultdict(list)
        sell_sched: dict[np.datetime64, list[ExecResult]] = defaultdict(list)
        if pos_size_handler is not None:
            pos_ctx = PosSizeContext(config=config, portfolio=portfolio, col_scope=col_scope, ind_scope=ind_scope, input_scope=input_scope, pred_scope=pred_scope, pending_order_scope=pending_order_scope, models=models, sessions=sessions, sym_end_index=sym_end_index)
        logger = StaticScope.instance().logger
        logger.backtest_executions_start(test_dates)
        cover_results: deque[ExecResult] = deque()
        buy_results: deque[ExecResult] = deque()
        sell_results: deque[ExecResult] = deque()
        exit_ctxs: deque[ExecContext] = deque()
        active_ctxs: dict[str, ExecContext] = {}
        for i, date in enumerate(test_dates):
            active_ctxs.clear()
            for sym, ctx in exec_ctxs.items():
                if date not in sym_exec_dates[sym]:
                    continue
                sym_end_index[sym] += 1
                if warmup and sym_end_index[sym] <= warmup:
                    continue
                active_ctxs[sym] = ctx
                set_exec_ctx_data(ctx, date)
                if exit_dates and sym in exit_dates and (date == exit_dates[sym]):
                    exit_ctxs.append(ctx)
            is_cover_sched = date in cover_sched
            is_buy_sched = date in buy_sched
            is_sell_sched = date in sell_sched
            if config.max_long_positions is not None or pos_size_handler is not None:
                if is_cover_sched:
                    cover_sched[date].sort(key=_sort_by_score, reverse=True)
                elif is_buy_sched:
                    buy_sched[date].sort(key=_sort_by_score, reverse=True)
            if is_sell_sched and (config.max_short_positions is not None or pos_size_handler is not None):
                sell_sched[date].sort(key=_sort_by_score, reverse=True)
            if pos_size_handler is not None and (is_cover_sched or is_buy_sched or is_sell_sched):
                pos_size_buy_results = None
                if is_cover_sched:
                    pos_size_buy_results = cover_sched[date]
                elif is_buy_sched:
                    pos_size_buy_results = buy_sched[date]
                self._set_pos_sizes(pos_size_handler=pos_size_handler, pos_ctx=pos_ctx, buy_results=pos_size_buy_results, sell_results=sell_sched[date] if is_sell_sched else None)
            portfolio.check_stops(date, price_scope)
            if is_cover_sched:
                self._place_buy_orders(date=date, price_scope=price_scope, pending_order_scope=pending_order_scope, buy_sched=cover_sched, portfolio=portfolio, enable_fractional_shares=enable_fractional_shares)
            if is_sell_sched:
                self._place_sell_orders(date=date, price_scope=price_scope, pending_order_scope=pending_order_scope, sell_sched=sell_sched, portfolio=portfolio, enable_fractional_shares=enable_fractional_shares)
            if is_buy_sched:
                self._place_buy_orders(date=date, price_scope=price_scope, pending_order_scope=pending_order_scope, buy_sched=buy_sched, portfolio=portfolio, enable_fractional_shares=enable_fractional_shares)
            portfolio.capture_bar(date, test_data)
            if before_exec_fn is not None and active_ctxs:
                before_exec_fn(active_ctxs)
            for sym, ctx in active_ctxs.items():
                if sym in exec_fns:
                    exec_fns[sym](ctx)
            if after_exec_fn is not None and active_ctxs:
                after_exec_fn(active_ctxs)
            for ctx in active_ctxs.values():
                if slippage_model and (not ctx._exiting_pos) and (ctx.buy_shares or ctx.sell_shares):
                    self._apply_slippage(slippage_model, ctx)
                result = ctx.to_result()
                if result is None:
                    continue
                if result.buy_shares is not None:
                    if result.cover:
                        cover_results.append(result)
                    else:
                        buy_results.append(result)
                if result.sell_shares is not None:
                    sell_results.append(result)
            while cover_results:
                self._schedule_order(result=cover_results.popleft(), created=date, sym_end_index=sym_end_index, delay=config.buy_delay, sched=cover_sched, col_scope=col_scope, pending_order_scope=pending_order_scope)
            while buy_results:
                self._schedule_order(result=buy_results.popleft(), created=date, sym_end_index=sym_end_index, delay=config.buy_delay, sched=buy_sched, col_scope=col_scope, pending_order_scope=pending_order_scope)
            while sell_results:
                self._schedule_order(result=sell_results.popleft(), created=date, sym_end_index=sym_end_index, delay=config.sell_delay, sched=sell_sched, col_scope=col_scope, pending_order_scope=pending_order_scope)
            while exit_ctxs:
                self._exit_position(portfolio=portfolio, date=date, ctx=exit_ctxs.popleft(), exit_cover_fill_price=config.exit_cover_fill_price, exit_sell_fill_price=config.exit_sell_fill_price, price_scope=price_scope)
            portfolio.incr_bars()
            if i % 10 == 0 or i == len(test_dates) - 1:
                logger.backtest_executions_loading(i + 1)
        return get_signals(test_syms, col_scope, ind_scope, pred_scope) if config.return_signals else {}

    def _apply_slippage(self, slippage_model: SlippageModel, ctx: ExecContext):
        buy_shares = to_decimal(ctx.buy_shares) if ctx.buy_shares else None
        sell_shares = to_decimal(ctx.sell_shares) if ctx.sell_shares else None
        slippage_model.apply_slippage(ctx, buy_shares=buy_shares, sell_shares=sell_shares)

    def _exit_position(self, portfolio: Portfolio, date: np.datetime64, ctx: ExecContext, exit_cover_fill_price: Union[PriceType, Callable[[str, BarData], Union[int, float, Decimal]]], exit_sell_fill_price: Union[PriceType, Callable[[str, BarData], Union[int, float, Decimal]]], price_scope: PriceScope):
        buy_fill_price = price_scope.fetch(ctx.symbol, exit_cover_fill_price)
        sell_fill_price = price_scope.fetch(ctx.symbol, exit_sell_fill_price)
        portfolio.exit_position(date, ctx.symbol, buy_fill_price=buy_fill_price, sell_fill_price=sell_fill_price)

    def _set_pos_sizes(self, pos_size_handler: Callable[[PosSizeContext], None], pos_ctx: PosSizeContext, buy_results: Optional[list[ExecResult]], sell_results: Optional[list[ExecResult]]):
        set_pos_size_ctx_data(ctx=pos_ctx, buy_results=buy_results, sell_results=sell_results)
        pos_size_handler(pos_ctx)
        for id, shares in pos_ctx._signal_shares.items():
            if id < 0:
                raise ValueError(f'Invalid ExecSignal id: {id}')
            if buy_results is not None and sell_results is not None:
                if id >= len(buy_results) + len(sell_results):
                    raise ValueError(f'Invalid ExecSignal id: {id}')
                if id < len(buy_results):
                    buy_results[id].buy_shares = to_decimal(shares)
                else:
                    sell_results[id - len(buy_results)].sell_shares = to_decimal(shares)
            elif buy_results is not None:
                if id >= len(buy_results):
                    raise ValueError(f'Invalid ExecSignal id: {id}')
                buy_results[id].buy_shares = to_decimal(shares)
            elif sell_results is not None:
                if id >= len(sell_results):
                    raise ValueError(f'Invalid ExecSignal id: {id}')
                sell_results[id].sell_shares = to_decimal(shares)
            else:
                raise ValueError('buy_results and sell_results cannot both be None.')

    def _schedule_order(self, result: ExecResult, created: np.datetime64, sym_end_index: Mapping[str, int], delay: int, sched: Mapping[np.datetime64, list[ExecResult]], col_scope: ColumnScope, pending_order_scope: PendingOrderScope):
        date_loc = sym_end_index[result.symbol] - 1
        dates = col_scope.fetch(result.symbol, DataCol.DATE.value)
        if dates is None:
            raise ValueError('Dates not found.')
        logger = StaticScope.instance().logger
        if date_loc + delay < len(dates):
            date = dates[date_loc + delay]
            order_type: Literal['buy', 'sell']
            if result.buy_shares is not None:
                order_type = 'buy'
                shares = result.buy_shares
                limit_price = result.buy_limit_price
                fill_price = result.buy_fill_price
            elif result.sell_shares is not None:
                order_type = 'sell'
                shares = result.sell_shares
                limit_price = result.sell_limit_price
                fill_price = result.sell_fill_price
            else:
                raise ValueError('buy_shares or sell_shares needs to be set.')
            result.pending_order_id = pending_order_scope.add(type=order_type, symbol=result.symbol, created=created, exec_date=date, shares=shares, limit_price=limit_price, fill_price=fill_price)
            sched[date].append(result)
            logger.debug_schedule_order(date, result)
        else:
            logger.debug_unscheduled_order(result)

    def _place_buy_orders(self, date: np.datetime64, price_scope: PriceScope, pending_order_scope: PendingOrderScope, buy_sched: dict[np.datetime64, list[ExecResult]], portfolio: Portfolio, enable_fractional_shares: bool):
        buy_results = buy_sched[date]
        for result in buy_results:
            if result.buy_shares is None:
                continue
            if result.pending_order_id is None or not pending_order_scope.contains(result.pending_order_id):
                continue
            pending_order_scope.remove(result.pending_order_id)
            buy_shares = self._get_shares(result.buy_shares, enable_fractional_shares)
            fill_price = price_scope.fetch(result.symbol, result.buy_fill_price)
            order = portfolio.buy(date=date, symbol=result.symbol, shares=buy_shares, fill_price=fill_price, limit_price=result.buy_limit_price, stops=result.long_stops)
            logger = StaticScope.instance().logger
            if order is None:
                logger.debug_unfilled_buy_order(date=date, symbol=result.symbol, shares=buy_shares, fill_price=fill_price, limit_price=result.buy_limit_price)
            else:
                logger.debug_filled_buy_order(date=date, symbol=result.symbol, shares=buy_shares, fill_price=fill_price, limit_price=result.buy_limit_price)
        del buy_sched[date]

    def _place_sell_orders(self, date: np.datetime64, price_scope: PriceScope, pending_order_scope: PendingOrderScope, sell_sched: dict[np.datetime64, list[ExecResult]], portfolio: Portfolio, enable_fractional_shares: bool):
        sell_results = sell_sched[date]
        for result in sell_results:
            if result.sell_shares is None:
                continue
            if result.pending_order_id is None or not pending_order_scope.contains(result.pending_order_id):
                continue
            pending_order_scope.remove(result.pending_order_id)
            sell_shares = self._get_shares(result.sell_shares, enable_fractional_shares)
            fill_price = price_scope.fetch(result.symbol, result.sell_fill_price)
            order = portfolio.sell(date=date, symbol=result.symbol, shares=sell_shares, fill_price=fill_price, limit_price=result.sell_limit_price, stops=result.short_stops)
            logger = StaticScope.instance().logger
            if order is None:
                logger.debug_unfilled_sell_order(date=date, symbol=result.symbol, shares=sell_shares, fill_price=fill_price, limit_price=result.sell_limit_price)
            else:
                logger.debug_filled_sell_order(date=date, symbol=result.symbol, shares=sell_shares, fill_price=fill_price, limit_price=result.sell_limit_price)
        del sell_sched[date]

    def _get_shares(self, shares: Union[int, float, Decimal], enable_fractional_shares: bool) -> Decimal:
        if enable_fractional_shares:
            return to_decimal(shares)
        else:
            return to_decimal(int(shares))

def set_exec_ctx_data(ctx: ExecContext, date: np.datetime64):
    """Sets data on an :class:`.ExecContext` instance.

    Args:
        ctx: :class:`.ExecContext`.
        date: Current bar's date.
    """
    ctx._curr_date = date
    ctx._dt = None
    ctx._foreign.clear()
    ctx._cover = False
    ctx._exiting_pos = False
    ctx.buy_fill_price = None
    ctx.buy_shares = None
    ctx.buy_limit_price = None
    ctx.sell_fill_price = None
    ctx.sell_shares = None
    ctx.sell_limit_price = None
    ctx.hold_bars = None
    ctx.score = None
    ctx.stop_loss = None
    ctx.stop_loss_pct = None
    ctx.stop_loss_limit = None
    ctx.stop_profit = None
    ctx.stop_profit_pct = None
    ctx.stop_profit_limit = None
    ctx.stop_trailing = None
    ctx.stop_trailing_pct = None
    ctx.stop_trailing_limit = None

class ModelsMixin:
    """Mixin implementing model related functionality."""

    def train_models(self, model_syms: Iterable[ModelSymbol], train_data: pd.DataFrame, test_data: pd.DataFrame, indicator_data: Mapping[IndicatorSymbol, pd.Series], cache_date_fields: CacheDateFields) -> dict[ModelSymbol, TrainedModel]:
        """Trains models for the provided :class:`pybroker.common.ModelSymbol`
        pairs.

        Args:
            model_syms: ``Iterable`` of
                :class:`pybroker.common.ModelSymbol` pairs of models to train.
            train_data: :class:`pandas.DataFrame` of training data.
            test_data: :class:`pandas.DataFrame` of test data.
            indicator_data: ``Mapping`` of
                :class:`pybroker.common.IndicatorSymbol` pairs to
                ``pandas.Series`` of :class:`pybroker.indicator.Indicator`
                values.
            cache_date_fields: Date fields used to key cache data.

        Returns:
            ``dict`` mapping each :class:`pybroker.common.ModelSymbol` pair
            to a :class:`pybroker.common.TrainedModel`.
        """
        if train_data.empty or not model_syms:
            return {}
        scope = StaticScope.instance()
        train_dates = get_unique_sorted_dates(train_data[DataCol.DATE.value])
        test_dates = get_unique_sorted_dates(test_data[DataCol.DATE.value])
        scope.logger.train_split_start(train_dates)
        scope.logger.info_train_split_start(model_syms)
        models, uncached_model_syms = self._get_cached_models(model_syms, cache_date_fields)
        if not uncached_model_syms:
            scope.logger.loaded_models()
            scope.logger.info_loaded_models(model_syms)
            return models
        if models:
            scope.logger.info_loaded_models(models.keys())
        start_date = to_datetime(train_dates[0])
        end_date = to_datetime(train_dates[-1])
        for model_sym in uncached_model_syms:
            if model_sym in models:
                continue
            model_name, sym = model_sym
            source = scope.get_model_source(model_name)
            if isinstance(source, ModelTrainer):
                sym_train_data = self._slice_by_symbol(sym, train_data)
                sym_test_data = self._slice_by_symbol(sym, test_data)
                for ind_name in source.indicators:
                    ind_series = indicator_data[IndicatorSymbol(ind_name, sym)]
                    if not sym_train_data.empty:
                        sym_train_data[ind_name] = ind_series[ind_series.index.isin(train_dates)].values
                    if not sym_test_data.empty:
                        sym_test_data[ind_name] = ind_series[ind_series.index.isin(test_dates)].values
                scope.logger.info_train_model_start(model_sym)
                model_result = source(sym, sym_train_data, sym_test_data)
                scope.logger.info_train_model_completed(model_sym)
            elif isinstance(source, ModelLoader):
                model_result = source(sym, start_date, end_date)
                scope.logger.info_loaded_model(model_sym)
            else:
                raise TypeError(f'Invalid ModelSource type: {type(source)}')
            input_cols: Optional[tuple[str]] = None
            if isinstance(model_result, tuple):
                model = model_result[0]
                input_cols = tuple(model_result[1])
            else:
                model = model_result
            models[model_sym] = TrainedModel(name=model_name, instance=model, predict_fn=source._predict_fn, input_cols=input_cols)
            self._set_cached_model(model, input_cols, model_sym, cache_date_fields)
        scope.logger.train_split_completed()
        return models

    def _slice_by_symbol(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[df[DataCol.SYMBOL.value] == symbol].drop(columns=DataCol.SYMBOL.value).sort_values(DataCol.DATE.value)

    def _get_cached_models(self, model_syms: Iterable[ModelSymbol], cache_date_fields: CacheDateFields) -> tuple[dict[ModelSymbol, TrainedModel], list[ModelSymbol]]:
        model_syms = sorted(model_syms)
        models: dict[ModelSymbol, TrainedModel] = {}
        scope = StaticScope.instance()
        if scope.model_cache is None:
            return (models, model_syms)
        uncached_model_syms = []
        for model_sym in model_syms:
            cache_key = ModelCacheKey(symbol=model_sym.symbol, model_name=model_sym.model_name, **asdict(cache_date_fields))
            scope.logger.debug_get_model_cache(cache_key)
            cached_data = scope.model_cache.get(repr(cache_key))
            if cached_data is not None:
                input_cols = None
                if isinstance(cached_data, CachedModel):
                    model = cached_data.model
                    input_cols = cached_data.input_cols
                else:
                    model = cached_data
                source = scope.get_model_source(model_sym.model_name)
                models[model_sym] = TrainedModel(name=model_sym.model_name, instance=model, predict_fn=source._predict_fn, input_cols=input_cols)
            else:
                uncached_model_syms.append(model_sym)
        return (models, uncached_model_syms)

    def _set_cached_model(self, model: Any, input_cols: Optional[tuple[str]], model_sym: ModelSymbol, cache_date_fields: CacheDateFields):
        scope = StaticScope.instance()
        if scope.model_cache is None:
            return
        cache_key = ModelCacheKey(symbol=model_sym.symbol, model_name=model_sym.model_name, **asdict(cache_date_fields))
        cached_model = CachedModel(model, input_cols)
        scope.logger.debug_set_model_cache(cache_key)
        scope.model_cache.set(repr(cache_key), cached_model)

def get_unique_sorted_dates(col: pd.Series) -> Sequence[np.datetime64]:
    """Returns sorted unique values from a DataFrame column of dates.
    Guarantees compatability between Pandas 1 and 2.
    """
    result = col.unique()
    if hasattr(result, 'to_numpy'):
        result = result.to_numpy()
    result.sort()
    return result

def to_datetime(date: Union[str, datetime, np.datetime64, pd.Timestamp]) -> datetime:
    """Converts ``date`` to :class:`datetime`."""
    if isinstance(date, pd.Timestamp):
        return date.to_pydatetime()
    elif isinstance(date, datetime):
        return date
    elif isinstance(date, str):
        return pd.to_datetime(date).to_pydatetime()
    elif isinstance(date, np.datetime64):
        return pd.Timestamp(date).to_pydatetime()
    else:
        raise TypeError(f'Unsupported date type: {type(date)}')

class AKShare(DataSource):
    """Retrieves data from `AKShare <https://akshare.akfamily.xyz/>`_."""
    _tf_to_period = {'': 'daily', '1day': 'daily', '1week': 'weekly'}

    def _fetch_data(self, symbols: frozenset[str], start_date: datetime, end_date: datetime, timeframe: Optional[str], adjust: Optional[str]) -> pd.DataFrame:
        """:meta private:"""
        start_date_str = to_datetime(start_date).strftime('%Y%m%d')
        end_date_str = to_datetime(end_date).strftime('%Y%m%d')
        symbols_list = list(symbols)
        symbols_simple = [item.split('.')[0] for item in symbols_list]
        result = pd.DataFrame()
        formatted_tf = self._format_timeframe(timeframe)
        if formatted_tf in AKShare._tf_to_period:
            period = AKShare._tf_to_period[formatted_tf]
            for i in range(len(symbols_list)):
                temp_df = akshare.stock_zh_a_hist(symbol=symbols_simple[i], start_date=start_date_str, end_date=end_date_str, period=period, adjust=adjust if adjust is not None else '')
                if not temp_df.columns.empty:
                    temp_df['symbol'] = symbols_list[i]
                result = pd.concat([result, temp_df], ignore_index=True)
        if result.columns.empty:
            return pd.DataFrame(columns=[DataCol.SYMBOL.value, DataCol.DATE.value, DataCol.OPEN.value, DataCol.HIGH.value, DataCol.LOW.value, DataCol.CLOSE.value, DataCol.VOLUME.value])
        if result.empty:
            return result
        result.rename(columns={'日期': DataCol.DATE.value, '开盘': DataCol.OPEN.value, '收盘': DataCol.CLOSE.value, '最高': DataCol.HIGH.value, '最低': DataCol.LOW.value, '成交量': DataCol.VOLUME.value}, inplace=True)
        result['date'] = pd.to_datetime(result['date'])
        result = result[[DataCol.DATE.value, DataCol.SYMBOL.value, DataCol.OPEN.value, DataCol.HIGH.value, DataCol.LOW.value, DataCol.CLOSE.value, DataCol.VOLUME.value]]
        return result

@pytest.mark.parametrize('date, expected', [('2022-02-02', datetime.strptime('2022-02-02', '%Y-%m-%d')), (datetime.strptime('2021-05-05', '%Y-%m-%d'), datetime.strptime('2021-05-05', '%Y-%m-%d')), (np.datetime64('2019-03-03'), datetime.strptime('2019-03-03', '%Y-%m-%d')), (pd.Timestamp('2020-03-03'), datetime.strptime('2020-03-03', '%Y-%m-%d'))])
def test_to_datetime(date, expected):
    dt = to_datetime(date)
    assert isinstance(dt, datetime)
    assert dt == expected

def test_to_datetime_type_error():
    with pytest.raises(TypeError, match='Unsupported date type: .*'):
        to_datetime(1000)

@pytest.fixture()
def cache_date_fields(data_source_df):
    return CacheDateFields(start_date=to_datetime(sorted(data_source_df['date'].unique())[0]), end_date=to_datetime(sorted(data_source_df['date'].unique())[-1]), tf_seconds=TF_SECONDS, between_time=BETWEEN_TIME, days=None)

class TestStrategy:

    @pytest.mark.parametrize('data_source', [FakeDataSource(), LazyFixture('data_source_df')])
    @pytest.mark.parametrize('executions', [LazyFixture('executions_train_only'), LazyFixture('executions_only'), LazyFixture('executions_with_indicators'), LazyFixture('executions_with_models'), LazyFixture('executions_with_models_and_indicators')])
    def test_walkforward(self, data_source, executions, date_range, days, between_time, calc_bootstrap, disable_parallel, request):
        data_source = get_fixture(request, data_source)
        executions = get_fixture(request, executions)
        config = StrategyConfig(bootstrap_samples=100, bootstrap_sample_size=10)
        strategy = Strategy(data_source, START_DATE, END_DATE, config)
        for exec in executions:
            strategy.add_execution(**exec)
        result = strategy.walkforward(start_date=date_range[0], end_date=date_range[1], windows=3, lookahead=1, timeframe='1d', days=days, between_time=between_time, calc_bootstrap=calc_bootstrap, disable_parallel=disable_parallel, adjust='adjustment')
        if date_range[0] is None:
            expected_start_date = datetime.strptime(START_DATE, '%Y-%m-%d')
        else:
            expected_start_date = pd.to_datetime(date_range[0])
        if date_range[1] is None:
            expected_end_date = datetime.strptime(END_DATE, '%Y-%m-%d')
        else:
            expected_end_date = pd.to_datetime(date_range[1])
        if all(map(lambda e: not e['fn'], executions)):
            assert result.start_date == expected_start_date
            assert result.end_date == expected_end_date
            assert result.portfolio.empty
            assert result.positions.empty
            assert result.orders.empty
            assert result.trades.empty
            assert result.metrics == EvalMetrics()
            assert result.bootstrap is None
            assert result.signals is None
            return
        assert isinstance(result, TestResult)
        assert result.metrics is not None
        assert isinstance(result.metrics_df, pd.DataFrame)
        assert not result.metrics_df.empty
        assert result.start_date == expected_start_date
        assert result.end_date == expected_end_date
        assert isinstance(result.portfolio, pd.DataFrame)
        assert not result.portfolio.empty
        assert isinstance(result.positions, pd.DataFrame)
        assert isinstance(result.orders, pd.DataFrame)
        if calc_bootstrap:
            assert not result.bootstrap.conf_intervals.empty
            assert not result.bootstrap.drawdown_conf.empty
        else:
            assert result.bootstrap is None

    @pytest.mark.parametrize('return_signals', [True, False])
    @pytest.mark.parametrize('return_stops', [True, False])
    def test_walkforward_results(self, data_source_df, return_signals, return_stops):

        def exec_fn(ctx):
            if not ctx.long_pos():
                ctx.buy_shares = 100
                ctx.stop_trailing = 100
                ctx.stop_profit_pct = 100
        data_source_df = data_source_df[data_source_df['date'] <= to_datetime(END_DATE)]
        config = StrategyConfig(return_signals=return_signals, return_stops=return_stops)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(exec_fn, ['AAPL', 'SPY'])
        result = strategy.walkforward(windows=3, calc_bootstrap=False)
        dates = set()
        for _, test_idx in strategy.walkforward_split(data_source_df, windows=3, lookahead=1, train_size=0.5):
            df = data_source_df.loc[test_idx]
            df = df[df['symbol'].isin(['AAPL', 'SPY'])]
            dates.update(df['date'].values)
        assert result.start_date == to_datetime(START_DATE)
        assert result.end_date == to_datetime(END_DATE)
        dates_list = list(dates)
        dates_list.sort()
        assert np.array_equal(result.portfolio.index, dates_list)
        assert len(result.positions) == 2 * len(dates) - 2
        assert np.array_equal(result.positions.index.get_level_values(1).unique(), dates_list[1:])
        assert len(result.orders) == 2
        assert not len(result.trades)
        if return_signals:
            assert len(result.signals) == 2
            assert not result.signals['AAPL'].empty
            assert not result.signals['SPY'].empty
        else:
            assert result.signals is None
        if return_stops:
            assert not result.stops.empty
            assert set(result.stops.columns) == {'date', 'symbol', 'stop_id', 'stop_type', 'pos_type', 'curr_value', 'curr_bars', 'percent', 'points', 'bars', 'fill_price', 'limit_price', 'exit_price'}
        else:
            assert result.stops is None

    def test_walkforward_when_no_executions_then_error(self, data_source_df):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        with pytest.raises(ValueError, match=re.escape('No executions were added.')):
            strategy.walkforward(windows=3, lookahead=1)

    def test_walkforward_when_empty_data_source_then_error(self):
        df = pd.DataFrame(columns=[col.value for col in DataCol])
        strategy = Strategy(df, START_DATE, END_DATE)
        strategy.add_execution(None, 'SPY')
        with pytest.raises(ValueError, match=re.escape('DataSource is empty.')):
            strategy.walkforward(windows=3, lookahead=1)

    @pytest.mark.parametrize('start_date_1, end_date_1, start_date_2, end_date_2, expected_msg', [('2020-03-01', '2020-02-20', None, None, 'start_date (.*) must be on or before end_date (.*)\\.'), ('2020-03-01', '2020-09-30', '2020-01-01', None, 'start_date must be between .* and .*\\.'), ('2020-03-01', '2020-09-30', '2020-10-01', None, 'start_date must be between .* and .*\\.'), ('2020-03-01', '2020-09-30', None, '2020-02-01', 'end_date must be between .* and .*\\.'), ('2020-03-01', '2020-09-30', None, '2020-10-31', 'end_date must be between .* and .*\\.'), ('2020-03-01', '2020-09-30', '2020-05-01', '2020-04-01', 'start_date (.*) must be on or before end_date (.*)\\.')])
    def test_walkforward_when_invalid_dates_then_error(self, executions_only, data_source_df, start_date_1, end_date_1, start_date_2, end_date_2, expected_msg):
        with pytest.raises(ValueError, match=expected_msg):
            strategy = Strategy(data_source_df, start_date_1, end_date_1)
            for exec in executions_only:
                strategy.add_execution(**exec)
            strategy.walkforward(windows=3, lookahead=1, start_date=start_date_2, end_date=end_date_2)

    def test_backtest(self, executions_only, data_source_df):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        for exec in executions_only:
            strategy.add_execution(**exec)
        result = strategy.backtest(calc_bootstrap=True)
        assert isinstance(result, TestResult)
        assert result.start_date == datetime.strptime(START_DATE, '%Y-%m-%d')
        assert result.end_date == datetime.strptime(END_DATE, '%Y-%m-%d')
        assert not result.portfolio.empty
        assert not result.bootstrap.conf_intervals.empty
        assert not result.bootstrap.drawdown_conf.empty

    @pytest.mark.parametrize('tz', ['UTC', None])
    @pytest.mark.parametrize('between_time, expected_hour', [(None, None), (('10:00', '1:00'), (10, 13))])
    @pytest.mark.parametrize('days, expected_days', [(None, None), ('tues', {1}), (['weds', 'fri'], {2, 4})])
    def test_filter_dates(self, tz, between_time, expected_hour, days, expected_days, data_source_df):
        data_source_df['date'] = data_source_df['date'].dt.tz_localize(tz)
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        start_date = pd.to_datetime('1/1/2021').to_pydatetime()
        end_date = pd.to_datetime('12/1/2021').to_pydatetime()
        df = strategy._filter_dates(data_source_df, start_date, end_date, between_time=between_time, days=strategy._to_day_ids(days))
        assert df.iloc[0]['date'] >= start_date
        assert df.iloc[-1]['date'] <= end_date
        row_days = set()
        for _, row in df.iterrows():
            if between_time is not None:
                assert row['date'].hour >= expected_hour[0]
                assert row['date'].hour <= expected_hour[1]
            row_days.add(row['date'].weekday())
        if expected_days is not None:
            assert row_days == expected_days

    def test_filter_dates_when_empty(self, data_source_df):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        start_date = pd.to_datetime('1/1/2021').to_pydatetime()
        end_date = pd.to_datetime('12/1/2021').to_pydatetime()
        df = strategy._filter_dates(data_source_df, start_date, end_date, between_time=('9:00', '10:00'), days=strategy._to_day_ids('tues'))
        assert df.empty

    def test_filter_dates_when_invalid_between_time_then_error(self, data_source_df):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        start_date = pd.to_datetime('1/1/2021').to_pydatetime()
        end_date = pd.to_datetime('12/1/2021').to_pydatetime()
        with pytest.raises(ValueError, match=re.escape("between_time must be a tuple[str, str] of start time and end time, received '9:00'.")):
            strategy._filter_dates(data_source_df, start_date, end_date, days=None, between_time='9:00')

    def test_add_execution_when_empty_symbols_then_error(self, data_source_df):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        with pytest.raises(ValueError, match=re.escape('symbols cannot be empty.')):
            strategy.add_execution(None, [])

    def test_add_execution_when_duplicate_symbol_then_error(self, data_source_df):

        def exec_fn_1(ctx):
            ctx.buy_shares = 100

        def exec_fn_2(ctx):
            ctx.sell_shares = 100
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn_1, ['AAPL', 'SPY'])
        with pytest.raises(ValueError, match=re.escape('AAPL was already added to an execution.')):
            strategy.add_execution(exec_fn_2, 'AAPL')

    @pytest.mark.parametrize('initial_cash, max_long_positions, max_short_positions, buy_delay,sell_delay, bootstrap_samples, bootstrap_sample_size, expected_msg', [(-1, None, None, 1, 1, 100, 10, 'initial_cash must be greater than 0.'), (10000, 0, None, 1, 1, 100, 10, 'max_long_positions must be greater than 0.'), (10000, None, 0, 1, 1, 100, 10, 'max_short_positions must be greater than 0.'), (10000, None, None, 0, 1, 100, 10, 'buy_delay must be greater than 0.'), (10000, None, None, 1, 0, 100, 10, 'sell_delay must be greater than 0.'), (10000, None, None, 1, 1, 0, 10, 'bootstrap_samples must be greater than 0.'), (10000, None, None, 1, 1, 100, 0, 'bootstrap_sample_size must be greater than 0.')])
    def test_when_invalid_config_then_error(self, data_source_df, initial_cash, max_long_positions, max_short_positions, buy_delay, sell_delay, bootstrap_samples, bootstrap_sample_size, expected_msg):
        config = StrategyConfig(initial_cash=initial_cash, max_long_positions=max_long_positions, max_short_positions=max_short_positions, buy_delay=buy_delay, sell_delay=sell_delay, bootstrap_samples=bootstrap_samples, bootstrap_sample_size=bootstrap_sample_size)
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            Strategy(data_source_df, START_DATE, END_DATE, config)

    def test_when_data_source_missing_columns_then_error(self):
        values = np.repeat(1, 100)
        df = pd.DataFrame({'symbol': ['SPY'] * 100, 'open': values, 'high': values, 'low': values, 'close': values})
        with pytest.raises(ValueError, match=re.escape("DataFrame is missing required columns: ['date']")):
            Strategy(df, START_DATE, END_DATE)

    def test_when_invalid_data_source_type_then_error(self):
        with pytest.raises(TypeError, match='Invalid data_source type: .*'):
            Strategy({}, START_DATE, END_DATE)

    def test_clear_executions(self):
        df = pd.DataFrame(columns=[col.value for col in DataCol])
        strategy = Strategy(df, START_DATE, END_DATE)
        strategy.add_execution(None, 'SPY')
        strategy.clear_executions()
        assert not strategy._executions

    @pytest.mark.parametrize('enable_fractional_shares, expected_shares_type,expected_short_shares, expected_long_shares', [(True, np.float64, 0.1, 3.14), (False, np.int_, 0, 3)])
    def test_to_test_result_when_fractional_shares(self, data_source_df, enable_fractional_shares, expected_shares_type, expected_long_shares, expected_short_shares):
        portfolio = Portfolio(100000)
        portfolio.bars = deque((PortfolioBar(date=np.datetime64(START_DATE), cash=Decimal(100000), equity=Decimal(100000), margin=Decimal(), market_value=Decimal(100000), pnl=Decimal(1000), unrealized_pnl=Decimal(), fees=Decimal()),))
        portfolio.position_bars = deque((PositionBar(symbol='SPY', date=np.datetime64(START_DATE), long_shares=Decimal('3.14'), short_shares=Decimal('0.1'), close=Decimal(100), equity=Decimal(100000), market_value=Decimal(100000), margin=Decimal(), unrealized_pnl=Decimal(100)),))
        portfolio.orders = deque((Order(id=1, type='buy', symbol='SPY', date=np.datetime64(START_DATE), shares=Decimal('3.14'), limit_price=Decimal(100), fill_price=Decimal(99), fees=Decimal()),))
        portfolio.trades = deque((Trade(id=1, type='long', symbol='SPY', entry_date=np.datetime64(START_DATE), exit_date=np.datetime64(END_DATE), entry=Decimal(100), exit=Decimal(101), shares=Decimal('3.14'), pnl=Decimal(1000), return_pct=Decimal('10.3'), agg_pnl=Decimal(1000), bars=2, pnl_per_bar=Decimal(500), stop=None, mae=Decimal(-10), mfe=Decimal(10)),))
        config = StrategyConfig(enable_fractional_shares=enable_fractional_shares)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        result = strategy._to_test_result(START_DATE, END_DATE, portfolio, calc_bootstrap=False, train_only=False, signals=None)
        assert np.issubdtype(result.positions['long_shares'].dtype, expected_shares_type)
        assert np.issubdtype(result.positions['short_shares'].dtype, expected_shares_type)
        assert np.issubdtype(result.orders['shares'].dtype, expected_shares_type)
        assert np.issubdtype(result.trades['shares'].dtype, expected_shares_type)
        assert result.positions['long_shares'].values[0] == expected_long_shares
        assert result.positions['short_shares'].values[0] == expected_short_shares
        assert result.orders['shares'].values[0] == expected_long_shares
        assert result.trades['shares'].values[0] == expected_long_shares

    def test_to_result_when_round_test_result_is_false(self, data_source_df):
        portfolio = Portfolio(100000)
        portfolio.bars = deque((PortfolioBar(date=np.datetime64(START_DATE), cash=Decimal(100000), equity=Decimal(100000), margin=Decimal(), market_value=Decimal(100000), pnl=Decimal('1000.111'), unrealized_pnl=Decimal(), fees=Decimal()),))
        portfolio.position_bars = deque((PositionBar(symbol='SPY', date=np.datetime64(START_DATE), long_shares=Decimal('3.144'), short_shares=Decimal('0.111'), close=Decimal(100), equity=Decimal(100000), market_value=Decimal(100000), margin=Decimal(), unrealized_pnl=Decimal(100)),))
        portfolio.orders = deque((Order(id=1, type='buy', symbol='SPY', date=np.datetime64(START_DATE), shares=Decimal('3.144'), limit_price=Decimal(100), fill_price=Decimal(99), fees=Decimal()),))
        portfolio.trades = deque((Trade(id=1, type='long', symbol='SPY', entry_date=np.datetime64(START_DATE), exit_date=np.datetime64(END_DATE), entry=Decimal(100), exit=Decimal(101), shares=Decimal('3.144'), pnl=Decimal(1000), return_pct=Decimal('10.33'), agg_pnl=Decimal(1000), bars=2, pnl_per_bar=Decimal(500), stop=None, mae=Decimal(-10), mfe=Decimal(10)),))
        config = StrategyConfig(enable_fractional_shares=True, round_test_result=False)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        result = strategy._to_test_result(START_DATE, END_DATE, portfolio, calc_bootstrap=False, train_only=False, signals=None)
        assert result.positions['long_shares'].values[0] == 3.144
        assert result.positions['short_shares'].values[0] == 0.111
        assert result.portfolio['pnl'].values[0] == 1000.111
        assert result.orders['shares'].values[0] == 3.144
        assert result.trades['shares'].values[0] == 3.144

    def test_to_test_result_when_empty(self, data_source_df):
        portfolio = Portfolio(100000)
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        result = strategy._to_test_result(START_DATE, END_DATE, portfolio, calc_bootstrap=False, train_only=False, signals=None)
        assert result.portfolio.empty
        assert result.positions.empty
        assert result.orders.empty
        assert result.trades.empty
        assert result.signals is None

    def test_backtest_when_exit_long_on_last_bar(self, data_source_df):

        def buy_exec_fn(ctx):
            if not ctx.long_pos():
                ctx.buy_shares = 100
                ctx.buy_fill_price = 150

        def sell_fill_price(_symbol, _bar_data):
            return 199.99
        config = StrategyConfig(exit_on_last_bar=True, exit_sell_fill_price=sell_fill_price)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(buy_exec_fn, 'SPY')
        result = strategy.backtest(calc_bootstrap=False)
        dates = data_source_df[data_source_df['symbol'] == 'SPY']['date'].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade['type'] == 'long'
        assert trade['symbol'] == 'SPY'
        assert trade['entry_date'] == dates[1]
        assert trade['exit_date'] == dates[-1]
        assert trade['entry'] == 150
        assert trade['exit'] == 199.99
        assert trade['shares'] == 100

    def test_backtest_when_exit_short_on_last_bar(self, data_source_df):

        def sell_exec_fn(ctx):
            if not ctx.short_pos():
                ctx.sell_shares = 100
                ctx.sell_fill_price = 200

        def buy_fill_price(_symbol, _bar_data):
            return 99.99
        config = StrategyConfig(exit_on_last_bar=True, exit_cover_fill_price=buy_fill_price)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(sell_exec_fn, 'SPY')
        result = strategy.backtest(calc_bootstrap=False)
        dates = data_source_df[data_source_df['symbol'] == 'SPY']['date'].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade['type'] == 'short'
        assert trade['symbol'] == 'SPY'
        assert trade['entry_date'] == dates[1]
        assert trade['exit_date'] == dates[-1]
        assert trade['entry'] == 200
        assert trade['exit'] == 99.99
        assert trade['shares'] == 100

    def test_backtest_when_buy_shares_and_sell_shares_then_error(self, data_source_df):

        def exec_fn(ctx):
            ctx.buy_shares = 100
            ctx.sell_shares = 100
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, ['AAPL', 'SPY'])
        with pytest.raises(ValueError, match=re.escape('For each symbol, only one of buy_shares or sell_shares can be set per bar.')):
            strategy.backtest()

    def test_backtest_pending_orders(self, data_source_df):
        buy_delay = 2
        dates = data_source_df[data_source_df['symbol'] == 'SPY']['date'].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]

        def buy_exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
            elif ctx.bars == 2:
                orders = tuple(ctx.pending_orders())
                assert len(orders) == 1
                assert orders[0] == PendingOrder(id=1, type='buy', symbol='SPY', created=ctx.date[0], exec_date=dates[buy_delay], shares=100, limit_price=None, fill_price=PriceType.MIDDLE)
            else:
                assert not tuple(ctx.pending_orders())
        config = StrategyConfig(buy_delay=buy_delay)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(buy_exec_fn, 'SPY')
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.orders) == 1
        order = result.orders.iloc[0]
        assert order['type'] == 'buy'
        assert order['symbol'] == 'SPY'
        assert order['date'] == dates[2]
        assert np.isnan(order['limit_price'])
        assert order['shares'] == 100

    def test_backtest_when_pending_orders_canceled(self, data_source_df):
        dates = data_source_df[data_source_df['symbol'] == 'SPY']['date'].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        buy_delay = 10
        sell_delay = 5

        def exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
                ctx.buy_limit_price = 99
            elif ctx.bars == 2:
                ctx.sell_shares = 200
                ctx.sell_limit_price = 100
            elif ctx.bars == 3:
                orders = tuple(ctx.pending_orders())
                assert len(orders) == 2
                assert orders[0] == PendingOrder(id=1, type='buy', symbol='SPY', created=ctx.date[0], exec_date=dates[buy_delay], shares=100, limit_price=99, fill_price=PriceType.MIDDLE)
                assert orders[1] == PendingOrder(id=2, type='sell', symbol='SPY', created=ctx.date[1], exec_date=dates[1 + sell_delay], shares=200, limit_price=100, fill_price=PriceType.MIDDLE)
                ctx.cancel_all_pending_orders()
            else:
                assert not tuple(ctx.pending_orders())
        config = StrategyConfig(buy_delay=buy_delay, sell_delay=sell_delay)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(exec_fn, 'SPY')
        result = strategy.backtest(calc_bootstrap=False)
        assert not len(result.orders)

    def test_backtest_when_buy_hold_bars(self, data_source_df):

        def buy_exec_fn(ctx):
            ctx.buy_fill_price = PriceType.CLOSE
            ctx.sell_fill_price = PriceType.OPEN
            ctx.buy_shares = 100
            ctx.hold_bars = 2
        df = data_source_df[data_source_df['symbol'] == 'SPY']
        dates = df['date'].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        buy_dates = dates[1:]
        sell_dates = dates[3:]
        config = StrategyConfig(initial_cash=500000)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(buy_exec_fn, 'SPY')
        result = strategy.backtest(calc_bootstrap=False)
        orders = result.orders
        buy_orders = orders[orders['type'] == 'buy']
        assert len(buy_orders) == len(buy_dates)
        for buy_date in buy_dates:
            row = buy_orders[buy_orders['date'] == buy_date]
            assert row['symbol'].item() == 'SPY'
            assert row['shares'].item() == 100
            assert np.isnan(row['limit_price'].item())
            assert row['fill_price'].item() == round(df[df['date'] == buy_date]['close'].item(), 2)
            assert row['fees'].item() == 0
        sell_orders = orders[orders['type'] == 'sell']
        assert len(sell_orders) == len(sell_dates)
        for sell_date in sell_dates:
            row = sell_orders[sell_orders['date'] == sell_date]
            assert row['symbol'].item() == 'SPY'
            assert row['shares'].item() == 100
            assert np.isnan(row['limit_price'].item())
            assert row['fill_price'].item() == round(df[df['date'] == sell_date]['open'].item(), 2)
            assert row['fees'].item() == 0
        assert (result.trades['stop'] == 'bar').all()

    def test_backtest_when_sell_hold_bars(self, data_source_df):

        def sell_exec_fn(ctx):
            ctx.sell_fill_price = PriceType.OPEN
            ctx.buy_fill_price = PriceType.CLOSE
            ctx.sell_shares = 100
            ctx.hold_bars = 1
        df = data_source_df[data_source_df['symbol'] == 'SPY']
        dates = df['date'].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        buy_dates = dates[2:]
        sell_dates = dates[1:]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(sell_exec_fn, 'SPY')
        result = strategy.backtest(calc_bootstrap=False)
        orders = result.orders
        sell_orders = orders[orders['type'] == 'sell']
        assert len(sell_orders) == len(sell_dates)
        for sell_date in sell_dates:
            row = sell_orders[sell_orders['date'] == sell_date]
            assert row['symbol'].item() == 'SPY'
            assert row['shares'].item() == 100
            assert np.isnan(row['limit_price'].item())
            assert row['fill_price'].item() == round(df[df['date'] == sell_date]['open'].item(), 2)
            assert row['fees'].item() == 0
        buy_orders = orders[orders['type'] == 'buy']
        assert len(buy_orders) == len(buy_dates)
        for buy_date in buy_dates:
            row = buy_orders[buy_orders['date'] == buy_date]
            assert row['symbol'].item() == 'SPY'
            assert row['shares'].item() == 100
            assert np.isnan(row['limit_price'].item())
            assert row['fill_price'].item() == round(df[df['date'] == buy_date]['close'].item(), 2)
            assert row['fees'].item() == 0
        assert len(result.trades) == len(buy_orders)
        assert (result.trades['stop'] == 'bar').all()

    def test_backtest_when_slippage(self, data_source_df):

        class FakeSlippageModel(SlippageModel):

            def apply_slippage(self, ctx: ExecContext, buy_shares, sell_shares):
                ctx.buy_shares = 99

        def buy_exec_fn(ctx):
            ctx.buy_fill_price = PriceType.CLOSE
            ctx.sell_fill_price = PriceType.OPEN
            ctx.buy_shares = 100
            ctx.hold_bars = 2
        df = data_source_df[data_source_df['symbol'] == 'SPY']
        dates = df['date'].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        buy_dates = dates[1:]
        sell_dates = dates[3:]
        config = StrategyConfig(initial_cash=500000)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.set_slippage_model(FakeSlippageModel())
        strategy.add_execution(buy_exec_fn, 'SPY')
        result = strategy.backtest(calc_bootstrap=False)
        orders = result.orders
        buy_orders = orders[orders['type'] == 'buy']
        assert len(buy_orders) == len(buy_dates)
        for buy_date in buy_dates:
            row = buy_orders[buy_orders['date'] == buy_date]
            assert row['symbol'].item() == 'SPY'
            assert row['shares'].item() == 99
            assert np.isnan(row['limit_price'].item())
            assert row['fill_price'].item() == round(df[df['date'] == buy_date]['close'].item(), 2)
            assert row['fees'].item() == 0
        sell_orders = orders[orders['type'] == 'sell']
        assert len(sell_orders) == len(sell_dates)
        for sell_date in sell_dates:
            row = sell_orders[sell_orders['date'] == sell_date]
            assert row['symbol'].item() == 'SPY'
            assert row['shares'].item() == 99
            assert np.isnan(row['limit_price'].item())
            assert row['fill_price'].item() == round(df[df['date'] == sell_date]['open'].item(), 2)
            assert row['fees'].item() == 0
        assert (result.trades['stop'] == 'bar').all()

    def test_backtest_when_slippage_and_sell_all_shares(self, data_source_df):

        class FakeSlippageModel(SlippageModel):

            def apply_slippage(self, ctx: ExecContext, buy_shares, sell_shares):
                if sell_shares:
                    ctx.sell_shares = 90

        def buy_exec_fn(ctx):
            if not ctx.long_pos():
                ctx.buy_shares = 100
            elif ctx.bars == 2:
                ctx.sell_all_shares()
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.set_slippage_model(FakeSlippageModel())
        strategy.add_execution(buy_exec_fn, 'SPY')
        result = strategy.backtest(calc_bootstrap=False)
        orders = result.orders
        sell_orders = orders[orders['type'] == 'sell']
        assert len(sell_orders) == 1
        assert sell_orders.iloc[0]['shares'] == 100

    def test_backtest_when_slippage_and_cover_all_shares(self, data_source_df):

        class FakeSlippageModel(SlippageModel):

            def apply_slippage(self, ctx: ExecContext, buy_shares, sell_shares):
                if buy_shares:
                    ctx.buy_shares = 90

        def buy_exec_fn(ctx):
            if not ctx.short_pos():
                ctx.sell_shares = 100
            elif ctx.bars == 2:
                ctx.cover_all_shares()
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.set_slippage_model(FakeSlippageModel())
        strategy.add_execution(buy_exec_fn, 'SPY')
        result = strategy.backtest(calc_bootstrap=False)
        orders = result.orders
        buy_orders = orders[orders['type'] == 'buy']
        assert len(buy_orders) == 1
        assert buy_orders.iloc[0]['shares'] == 100

    def test_backtest_when_stop_loss(self, data_source_df):

        def exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
                ctx.stop_loss = 10
        df = data_source_df[data_source_df['symbol'].isin(['SPY', 'AAPL'])]
        dates = df['date'].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, ['SPY', 'AAPL'])
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.trades) == 2
        trade = result.trades.iloc[0]
        assert trade['type'] == 'long'
        assert trade['symbol'] == 'SPY'
        assert trade['entry_date'] == dates[1]
        assert trade['exit'] == trade['entry'] - 10
        assert trade['shares'] == 100
        assert trade['pnl'] == -1000
        assert trade['agg_pnl'] == -1000
        assert trade['pnl_per_bar'] == round(-1000 / trade['bars'], 2)
        assert trade['stop'] == 'loss'
        trade = result.trades.iloc[1]
        assert trade['type'] == 'long'
        assert trade['symbol'] == 'AAPL'
        assert trade['entry_date'] == dates[1]
        assert trade['exit'] == trade['entry'] - 10
        assert trade['shares'] == 100
        assert trade['pnl'] == -1000
        assert trade['agg_pnl'] == -2000
        assert trade['pnl_per_bar'] == round(-1000 / trade['bars'], 2)
        assert trade['stop'] == 'loss'
        assert len(result.orders) == 4
        buy_order = result.orders.iloc[0]
        assert buy_order['type'] == 'buy'
        assert buy_order['symbol'] == 'AAPL'
        assert buy_order['date'] == dates[1]
        assert buy_order['shares'] == 100
        assert np.isnan(buy_order['limit_price'])
        assert buy_order['fees'] == 0
        buy_order = result.orders.iloc[1]
        assert buy_order['type'] == 'buy'
        assert buy_order['symbol'] == 'SPY'
        assert buy_order['date'] == dates[1]
        assert buy_order['shares'] == 100
        assert np.isnan(buy_order['limit_price'])
        assert buy_order['fees'] == 0
        sell_order = result.orders.iloc[2]
        assert sell_order['type'] == 'sell'
        assert sell_order['symbol'] == 'SPY'
        assert sell_order['shares'] == 100
        assert np.isnan(sell_order['limit_price'])
        assert sell_order['fees'] == 0
        sell_order = result.orders.iloc[3]
        assert sell_order['type'] == 'sell'
        assert sell_order['symbol'] == 'AAPL'
        assert sell_order['shares'] == 100
        assert np.isnan(sell_order['limit_price'])
        assert sell_order['fees'] == 0

    def test_backtest_when_sell_before_stop_loss(self, data_source_df):

        def exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
                ctx.stop_loss = 10
            elif ctx.bars == 10:
                ctx.sell_all_shares()
        df = data_source_df[data_source_df['symbol'] == 'SPY']
        dates = df['date'].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, 'SPY')
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade['type'] == 'long'
        assert trade['symbol'] == 'SPY'
        assert trade['entry_date'] == dates[1]
        assert trade['exit_date'] == dates[10]
        assert trade['shares'] == 100
        assert trade['stop'] is None
        assert len(result.orders) == 2
        buy_order = result.orders.iloc[0]
        assert buy_order['type'] == 'buy'
        assert buy_order['symbol'] == 'SPY'
        assert buy_order['date'] == dates[1]
        assert buy_order['shares'] == 100
        assert np.isnan(buy_order['limit_price'])
        assert buy_order['fees'] == 0
        sell_order = result.orders.iloc[1]
        assert sell_order['type'] == 'sell'
        assert sell_order['symbol'] == 'SPY'
        assert sell_order['date'] == dates[10]
        assert sell_order['shares'] == 100
        assert np.isnan(sell_order['limit_price'])
        assert sell_order['fees'] == 0

    def test_backtest_when_cancel_stop(self, data_source_df):

        def exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
                ctx.stop_loss = 10
            elif ctx.bars == 10:
                entry = tuple(ctx.long_pos().entries)[0]
                stop = next(iter(entry.stops))
                assert ctx.cancel_stop(stop_id=stop.id)
        df = data_source_df[data_source_df['symbol'] == 'SPY']
        dates = df['date'].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, 'SPY')
        result = strategy.backtest(calc_bootstrap=False)
        assert not len(result.trades)
        assert len(result.orders) == 1
        buy_order = result.orders.iloc[0]
        assert buy_order['type'] == 'buy'
        assert buy_order['symbol'] == 'SPY'
        assert buy_order['date'] == dates[1]
        assert buy_order['shares'] == 100
        assert np.isnan(buy_order['limit_price'])
        assert buy_order['fees'] == 0

    def test_backtest_when_cancel_stops(self, data_source_df):

        def exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
                ctx.stop_loss = 10
                ctx.stop_trailing = 10
            elif ctx.bars == 10:
                ctx.cancel_stops('SPY')
        df = data_source_df[data_source_df['symbol'] == 'SPY']
        dates = df['date'].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, 'SPY')
        result = strategy.backtest(calc_bootstrap=False)
        assert not len(result.trades)
        assert len(result.orders) == 1
        buy_order = result.orders.iloc[0]
        assert buy_order['type'] == 'buy'
        assert buy_order['symbol'] == 'SPY'
        assert buy_order['date'] == dates[1]
        assert buy_order['shares'] == 100
        assert np.isnan(buy_order['limit_price'])
        assert buy_order['fees'] == 0

    def test_backtest_when_pos_size_handler_zero_shares(self, data_source_df):

        def buy_exec_fn(ctx):
            ctx.buy_shares = 100

        def sell_exec_fn(ctx):
            ctx.sell_shares = 100

        def pos_size_handler(ctx):
            signals = tuple(ctx.signals())
            ctx.set_shares(signals[0], shares=0)
            ctx.set_shares(signals[1], shares=0)
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(buy_exec_fn, 'SPY')
        strategy.add_execution(sell_exec_fn, 'AAPL')
        strategy.set_pos_size_handler(pos_size_handler)
        result = strategy.backtest(calc_bootstrap=False)
        assert not len(result.orders)

    def test_backtest_when_no_stops(self, data_source_df):

        def exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
            elif ctx.long_pos() and ctx.bars > 30:
                ctx.sell_all_shares()
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, 'SPY')
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.trades) == 1
        assert result.trades.iloc[0]['stop'] is None

    def test_backtest_when_before_exec(self, data_source_df):

        def before_exec_fn(ctxs):
            assert len(ctxs) == 2
            assert isinstance(ctxs['SPY'], ExecContext)
            assert isinstance(ctxs['AAPL'], ExecContext)
            ctxs['SPY'].session['foo'] = 'bar'

        def exec_fn(ctx):
            if ctx.symbol == 'AAPL' and (not ctx.long_pos()):
                ctx.buy_shares = 200
            if ctx.symbol == 'SPY':
                assert ctx.session['foo'] == 'bar'
        df = data_source_df[data_source_df['symbol'] == 'AAPL']
        dates = df['date'].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, ['SPY', 'AAPL'])
        strategy.set_before_exec(before_exec_fn)
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.orders) == 1
        buy_order = result.orders.iloc[0]
        assert buy_order['type'] == 'buy'
        assert buy_order['symbol'] == 'AAPL'
        assert buy_order['date'] == dates[1]
        assert buy_order['shares'] == 200
        assert np.isnan(buy_order['limit_price'])
        assert buy_order['fees'] == 0

    def test_backtest_when_before_exec_and_no_executions(self, data_source_df):

        def before_exec_fn(ctxs):
            assert len(ctxs) == 2
            assert isinstance(ctxs['SPY'], ExecContext)
            assert isinstance(ctxs['AAPL'], ExecContext)
            if not ctxs['AAPL'].long_pos():
                ctxs['AAPL'].buy_shares = 200
        df = data_source_df[data_source_df['symbol'] == 'AAPL']
        dates = df['date'].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(None, ['SPY', 'AAPL'])
        strategy.set_before_exec(before_exec_fn)
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.orders) == 1
        buy_order = result.orders.iloc[0]
        assert buy_order['type'] == 'buy'
        assert buy_order['symbol'] == 'AAPL'
        assert buy_order['date'] == dates[1]
        assert buy_order['shares'] == 200
        assert np.isnan(buy_order['limit_price'])
        assert buy_order['fees'] == 0

    def test_backtest_when_after_exec(self, data_source_df):

        def after_exec_fn(ctxs):
            assert len(ctxs) == 2
            assert isinstance(ctxs['SPY'], ExecContext)
            assert isinstance(ctxs['AAPL'], ExecContext)
            if not ctxs['AAPL'].long_pos():
                ctxs['AAPL'].buy_shares = 300

        def exec_fn(ctx):
            if ctx.symbol == 'AAPL' and (not ctx.long_pos()):
                ctx.buy_shares = 200
        df = data_source_df[data_source_df['symbol'] == 'AAPL']
        dates = df['date'].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, ['SPY', 'AAPL'])
        strategy.set_after_exec(after_exec_fn)
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.orders) == 1
        buy_order = result.orders.iloc[0]
        assert buy_order['type'] == 'buy'
        assert buy_order['symbol'] == 'AAPL'
        assert buy_order['date'] == dates[1]
        assert buy_order['shares'] == 300
        assert np.isnan(buy_order['limit_price'])
        assert buy_order['fees'] == 0

    def test_backtest_when_after_exec_and_no_executions(self, data_source_df):

        def after_exec_fn(ctxs):
            assert len(ctxs) == 2
            assert isinstance(ctxs['SPY'], ExecContext)
            assert isinstance(ctxs['AAPL'], ExecContext)
            if not ctxs['AAPL'].long_pos():
                ctxs['AAPL'].buy_shares = 200
        df = data_source_df[data_source_df['symbol'] == 'AAPL']
        dates = df['date'].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(None, ['SPY', 'AAPL'])
        strategy.set_after_exec(after_exec_fn)
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.orders) == 1
        buy_order = result.orders.iloc[0]
        assert buy_order['type'] == 'buy'
        assert buy_order['symbol'] == 'AAPL'
        assert buy_order['date'] == dates[1]
        assert buy_order['shares'] == 200
        assert np.isnan(buy_order['limit_price'])
        assert buy_order['fees'] == 0

    def test_backtest_when_warmup(self, data_source_df):

        def exec_fn(ctx):
            if ctx.bars <= 10:
                raise AssertionError('Warmup failed.')
            elif not ctx.long_pos():
                ctx.buy_shares = 100
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, 'SPY')
        result = strategy.backtest(warmup=10)
        assert len(result.orders) == 1

    def test_backtest_when_warmup_invalid_then_error(self, data_source_df):

        def exec_fn(ctx):
            pass
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, 'SPY')
        with pytest.raises(ValueError, match=re.escape('warmup must be > 0.')):
            strategy.backtest(warmup=-1)

@pytest.fixture()
def ctx(col_scope, ind_scope, input_scope, pred_scope, pending_order_scope, portfolio, trained_models, sym_end_index, session, symbol, date):
    ctx = ExecContext(symbol=symbol, config=StrategyConfig(max_long_positions=5), portfolio=portfolio, col_scope=col_scope, ind_scope=ind_scope, input_scope=input_scope, pred_scope=pred_scope, pending_order_scope=pending_order_scope, models=trained_models, sym_end_index=sym_end_index, session=session)
    set_exec_ctx_data(ctx, date)
    return ctx

@pytest.fixture()
def ctx_with_pos(col_scope, ind_scope, input_scope, pred_scope, pending_order_scope, portfolio, trained_models, sym_end_index, session, symbol, symbols, date):
    portfolio.long_positions = {sym: Position(sym, 200, 'long') for sym in symbols}
    portfolio.short_positions = {sym: Position(sym, 100, 'short') for sym in symbols}
    ctx = ExecContext(symbol=symbol, config=StrategyConfig(max_long_positions=5), portfolio=portfolio, col_scope=col_scope, ind_scope=ind_scope, input_scope=input_scope, pred_scope=pred_scope, pending_order_scope=pending_order_scope, models=trained_models, sym_end_index=sym_end_index, session=session)
    set_exec_ctx_data(ctx, date)
    return ctx

@pytest.fixture()
def ctx_with_orders(col_scope, ind_scope, input_scope, pred_scope, pending_order_scope, portfolio, trained_models, sym_end_index, session, symbol, date, orders, trades):
    portfolio.orders = deque(orders)
    portfolio.trades = deque(trades)
    portfolio.win_rate = 1
    portfolio.lose_rate = 0
    ctx = ExecContext(symbol=symbol, config=StrategyConfig(max_long_positions=5), portfolio=portfolio, col_scope=col_scope, ind_scope=ind_scope, input_scope=input_scope, pred_scope=pred_scope, pending_order_scope=pending_order_scope, models=trained_models, sym_end_index=sym_end_index, session=session)
    set_exec_ctx_data(ctx, date)
    return ctx

def test_dt(ctx, date):
    assert ctx.dt == to_datetime(date)

def test_set_exec_ctx_data(ctx, sym_end_index):
    date = np.datetime64('2020-01-01')
    ctx._foreign = {'SPY': np.random.rand(100)}
    ctx._cover = True
    ctx._exiting_pos = True
    ctx.buy_fill_price = PriceType.AVERAGE
    ctx.buy_shares = 100
    ctx.buy_limit_price = 99
    ctx.sell_fill_price = PriceType.CLOSE
    ctx.sell_shares = 200
    ctx.sell_limit_price = 80
    ctx.hold_bars = 5
    ctx.score = 45.5
    ctx.stop_loss = 10
    ctx.stop_loss_pct = 20
    ctx.stop_loss_limit = 99
    ctx.stop_profit = 20
    ctx.stop_profit_pct = 30
    ctx.stop_profit_limit = 99.99
    ctx.stop_trailing = 100
    ctx.stop_trailing_pct = 15
    ctx.stop_trailing_limit = 80.8
    set_exec_ctx_data(ctx, date)
    assert ctx.dt == to_datetime(date)
    assert ctx.bars == sym_end_index[ctx.symbol]
    assert not ctx._foreign
    assert ctx._cover is False
    assert ctx._exiting_pos is False
    assert ctx.buy_fill_price is None
    assert ctx.buy_shares is None
    assert ctx.buy_limit_price is None
    assert ctx.sell_fill_price is None
    assert ctx.sell_shares is None
    assert ctx.sell_limit_price is None
    assert ctx.hold_bars is None
    assert ctx.score is None
    assert ctx.stop_loss is None
    assert ctx.stop_loss_pct is None
    assert ctx.stop_loss_limit is None
    assert ctx.stop_profit is None
    assert ctx.stop_profit_pct is None
    assert ctx.stop_profit_limit is None
    assert ctx.stop_trailing is None
    assert ctx.stop_trailing_pct is None
    assert ctx.stop_trailing_limit is None

@pytest.fixture()
def cache_date_fields(train_data):
    return CacheDateFields(start_date=to_datetime(sorted(train_data['date'].unique())[0]), end_date=to_datetime(sorted(train_data['date'].unique())[-1]), tf_seconds=TF_SECONDS, between_time=BETWEEN_TIME, days=None)

@pytest.fixture()
def start_date(train_data):
    return to_datetime(sorted(train_data['date'].unique())[0])

@pytest.fixture()
def end_date(train_data):
    return to_datetime(sorted(train_data['date'].unique())[-1])

