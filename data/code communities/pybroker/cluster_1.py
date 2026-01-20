# Cluster 1

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

def set_pos_size_ctx_data(ctx: PosSizeContext, buy_results: Optional[list[ExecResult]], sell_results: Optional[list[ExecResult]]):
    """Sets data on a :class:`.PosSizeContext` instance.

    Args:
        ctx: :class:`.PosSizeContext`.
        buy_results: :class:`.ExecResult`\\ s of buy signals.
        sell_results: :class:`.ExecResult`\\ s of sell signals.
    """
    ctx._signal_shares.clear()
    ctx._buy_results = buy_results
    ctx._sell_results = sell_results

def test_set_pos_ctx_data(date, portfolio, col_scope, ind_scope, input_scope, pred_scope, pending_order_scope, trained_models, sym_end_index):
    buy_results = [ExecResult(symbol='SPY', date=date, buy_shares=100, buy_fill_price=99, buy_limit_price=99, sell_shares=None, sell_fill_price=None, sell_limit_price=None, hold_bars=None, score=1, long_stops=None, short_stops=None), ExecResult(symbol='AAPL', date=date, buy_shares=200, buy_fill_price=90, buy_limit_price=90, sell_shares=None, sell_fill_price=None, sell_limit_price=None, hold_bars=None, score=2, long_stops=None, short_stops=None)]
    sell_results = [ExecResult(symbol='TSLA', date=date, buy_shares=None, buy_fill_price=None, buy_limit_price=None, sell_shares=100, sell_fill_price=80, sell_limit_price=80, hold_bars=None, score=1, long_stops=None, short_stops=None)]
    sessions = {'SPY': {}, 'AAPL': {}, 'TSLA': {'foo': 1}}
    ctx = PosSizeContext(StrategyConfig(max_long_positions=1), portfolio, col_scope, ind_scope, input_scope, pred_scope, pending_order_scope, trained_models, sessions, sym_end_index)
    set_pos_size_ctx_data(ctx, buy_results, sell_results)
    assert ctx.sessions == sessions
    buy_signals = list(ctx.signals('buy'))
    assert len(buy_signals) == 1
    assert buy_signals[0].id == 0
    assert buy_signals[0].symbol == 'SPY'
    assert buy_signals[0].shares == 100
    assert buy_signals[0].score == 1
    assert buy_signals[0].type == 'buy'
    assert buy_signals[0].bar_data is not None
    sell_signals = list(ctx.signals('sell'))
    assert len(sell_signals) == 1
    assert sell_signals[0].id == 2
    assert sell_signals[0].symbol == 'TSLA'
    assert sell_signals[0].shares == 100
    assert sell_signals[0].score == 1
    assert sell_signals[0].type == 'sell'
    assert sell_signals[0].bar_data is not None
    all_signals = list(ctx.signals())
    assert len(all_signals) == 2
    assert all_signals[0].id == buy_signals[0].id
    assert all_signals[0].symbol == buy_signals[0].symbol
    assert all_signals[0].shares == buy_signals[0].shares
    assert all_signals[0].score == buy_signals[0].score
    assert all_signals[0].type == buy_signals[0].type
    assert all_signals[0].bar_data is not None
    assert all_signals[1].id == sell_signals[0].id
    assert all_signals[1].symbol == sell_signals[0].symbol
    assert all_signals[1].shares == sell_signals[0].shares
    assert all_signals[1].score == sell_signals[0].score
    assert all_signals[1].type == sell_signals[0].type
    assert all_signals[1].bar_data is not None

