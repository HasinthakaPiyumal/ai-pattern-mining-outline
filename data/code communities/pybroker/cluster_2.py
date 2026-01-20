# Cluster 2

class Strategy(BacktestMixin, EvaluateMixin, IndicatorsMixin, ModelsMixin, WalkforwardMixin):
    """Class representing a trading strategy to backtest.

    Args:
        data_source: :class:`pybroker.data.DataSource` or
            :class:`pandas.DataFrame` of backtesting data.
        start_date: Starting date of the data to fetch from ``data_source``
            (inclusive).
        end_date: Ending date of the data to fetch from ``data_source``
            (inclusive).
        config: ``Optional`` :class:`pybroker.config.StrategyConfig`.
    """
    _execution_id: int = 0

    def __init__(self, data_source: Union[DataSource, pd.DataFrame], start_date: Union[str, datetime], end_date: Union[str, datetime], config: Optional[StrategyConfig]=None):
        self._verify_data_source(data_source)
        self._data_source = data_source
        self._start_date = to_datetime(start_date)
        self._end_date = to_datetime(end_date)
        verify_date_range(self._start_date, self._end_date)
        if config is not None:
            self._verify_config(config)
            self._config = config
        else:
            self._config = StrategyConfig()
        self._executions: set[Execution] = set()
        self._before_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]] = None
        self._after_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]] = None
        self._pos_size_handler: Optional[Callable[[PosSizeContext], None]] = None
        self._slippage_model: Optional[SlippageModel] = None
        self._scope = StaticScope.instance()
        self._logger = self._scope.logger

    def _verify_config(self, config: StrategyConfig):
        if config.initial_cash <= 0:
            raise ValueError('initial_cash must be greater than 0.')
        if config.max_long_positions is not None and config.max_long_positions <= 0:
            raise ValueError('max_long_positions must be greater than 0.')
        if config.max_short_positions is not None and config.max_short_positions <= 0:
            raise ValueError('max_short_positions must be greater than 0.')
        if config.buy_delay <= 0:
            raise ValueError('buy_delay must be greater than 0.')
        if config.sell_delay <= 0:
            raise ValueError('sell_delay must be greater than 0.')
        if config.bootstrap_samples <= 0:
            raise ValueError('bootstrap_samples must be greater than 0.')
        if config.bootstrap_sample_size <= 0:
            raise ValueError('bootstrap_sample_size must be greater than 0.')

    def _verify_data_source(self, data_source: Union[DataSource, pd.DataFrame]):
        if isinstance(data_source, pd.DataFrame):
            verify_data_source_columns(data_source)
        elif not isinstance(data_source, DataSource):
            raise TypeError(f'Invalid data_source type: {type(data_source)}')

    def set_slippage_model(self, slippage_model: Optional[SlippageModel]):
        """Sets :class:`pybroker.slippage.SlippageModel`."""
        self._slippage_model = slippage_model

    def add_execution(self, fn: Optional[Callable[[ExecContext], None]], symbols: Union[str, Iterable[str]], models: Optional[Union[ModelSource, Iterable[ModelSource]]]=None, indicators: Optional[Union[Indicator, Iterable[Indicator]]]=None):
        """Adds an execution to backtest.

        Args:
            fn: :class:`Callable` invoked on every bar of data during the
                backtest and passed an :class:`pybroker.context.ExecContext`
                for each ticker symbol in ``symbols``.
            symbols: Ticker symbols used to run ``fn``, where ``fn`` is called
                separately for each symbol.
            models: :class:`Iterable` of :class:`pybroker.model.ModelSource`\\ s
                to train/load for backtesting.
            indicators: :class:`Iterable` of
                :class:`pybroker.indicator.Indicator`\\ s to compute for
                backtesting.
        """
        symbols = frozenset((symbols,)) if isinstance(symbols, str) else frozenset(symbols)
        if not symbols:
            raise ValueError('symbols cannot be empty.')
        for sym in symbols:
            for exec in self._executions:
                if sym in exec.symbols:
                    raise ValueError(f'{sym} was already added to an execution.')
        if models is not None:
            for model in (models,) if isinstance(models, ModelSource) else models:
                if not self._scope.has_model_source(model.name):
                    raise ValueError(f'ModelSource {model.name!r} was not registered.')
                if model is not self._scope.get_model_source(model.name):
                    raise ValueError(f'ModelSource {model.name!r} does not match registered ModelSource.')
        model_names = (frozenset((models.name,)) if isinstance(models, ModelSource) else frozenset((model.name for model in models))) if models is not None else frozenset()
        if indicators is not None:
            for ind in (indicators,) if isinstance(indicators, Indicator) else indicators:
                if not self._scope.has_indicator(ind.name):
                    raise ValueError(f'Indicator {ind.name!r} was not registered.')
                if ind is not self._scope.get_indicator(ind.name):
                    raise ValueError(f'Indicator {ind.name!r} does not match registered Indicator.')
        ind_names = (frozenset((indicators.name,)) if isinstance(indicators, Indicator) else frozenset((ind.name for ind in indicators))) if indicators is not None else frozenset()
        self._execution_id += 1
        self._executions.add(Execution(id=self._execution_id, symbols=symbols, fn=fn, model_names=model_names, indicator_names=ind_names))

    def set_before_exec(self, fn: Optional[Callable[[Mapping[str, ExecContext]], None]]):
        """:class:`Callable[[Mapping[str, ExecContext]]` that runs before all
        execution functions.

        Args:
            fn: :class:`Callable` that takes a :class:`Mapping` of all ticker
                symbols to :class:`ExecContext`\\ s.
        """
        self._before_exec_fn = fn

    def set_after_exec(self, fn: Optional[Callable[[Mapping[str, ExecContext]], None]]):
        """:class:`Callable[[Mapping[str, ExecContext]]` that runs after all
        execution functions.

        Args:
            fn: :class:`Callable` that takes a :class:`Mapping` of all ticker
                symbols to :class:`ExecContext`\\ s.
        """
        self._after_exec_fn = fn

    def clear_executions(self):
        """Clears executions that were added with :meth:`.add_execution`."""
        self._executions.clear()

    def set_pos_size_handler(self, fn: Optional[Callable[[PosSizeContext], None]]):
        """Sets a :class:`Callable` that determines position sizes to use for
        buy and sell signals.

        Args:
            fn: :class:`Callable` invoked before placing orders for buy and
                sell signals, and is passed a
                :class:`pybroker.context.PosSizeContext`.
        """
        self._pos_size_handler = fn

    def backtest(self, start_date: Optional[Union[str, datetime]]=None, end_date: Optional[Union[str, datetime]]=None, timeframe: str='', between_time: Optional[tuple[str, str]]=None, days: Optional[Union[str, Day, Iterable[Union[str, Day]]]]=None, lookahead: int=1, train_size: float=0, shuffle: bool=False, calc_bootstrap: bool=False, disable_parallel: bool=False, warmup: Optional[int]=None, portfolio: Optional[Portfolio]=None, adjust: Optional[Any]=None) -> TestResult:
        """Backtests the trading strategy by running executions that were added
        with :meth:`.add_execution`.

        Args:
            start_date: Starting date of the backtest (inclusive). Must be
                within ``start_date`` and ``end_date`` range that was passed to
                :meth:`.__init__`.
            end_date: Ending date of the backtest (inclusive). Must be
                within ``start_date`` and ``end_date`` range that was passed to
                :meth:`.__init__`.
            timeframe: Formatted string that specifies the timeframe
                resolution of the backtesting data. The timeframe string
                supports the following units:

                - ``"s"``/``"sec"``: seconds
                - ``"m"``/``"min"``: minutes
                - ``"h"``/``"hour"``: hours
                - ``"d"``/``"day"``: days
                - ``"w"``/``"week"``: weeks

                An example timeframe string is ``1h 30m``.
            between_time: ``tuple[str, str]`` of times of day e.g.
                ('9:30', '16:00') used to filter the backtesting data
                (inclusive).
            days: Days (e.g. ``"mon"``, ``"tues"`` etc.) used to filter the
                backtesting data.
            lookahead: Number of bars in the future of the target prediction.
                For example, predicting returns for the next bar would have a
                ``lookahead`` of ``1``. This quantity is needed to prevent
                training data from leaking into the test boundary.
            train_size: Amount of :class:`pybroker.data.DataSource` data to use
                for training, where the max ``train_size`` is ``1``. For
                example, a ``train_size`` of ``0.9`` would result in 90% of
                data being used for training and the remaining 10% of data
                being used for testing.
            shuffle: Whether to randomly shuffle the data used for training.
                Defaults to ``False``. Disabled when model caching is enabled
                via :meth:`pybroker.cache.enable_model_cache`.
            calc_bootstrap: Whether to compute randomized bootstrap evaluation
                metrics. Defaults to ``False``.
            disable_parallel: If ``True``,
                :class:`pybroker.indicator.Indicator` data is computed
                serially. If ``False``, :class:`pybroker.indicator.Indicator`
                data is computed in parallel using multiple processes.
                Defaults to ``False``.
            warmup: Number of bars that need to pass before running the
                executions.
            portfolio: Custom :class:`pybroker.portfolio.Portfolio` to use for
                backtests.
            adjust: The type of adjustment to make to the
                :class:`pybroker.data.DataSource`.

        Returns:
            :class:`.TestResult` containing portfolio balances, order
            history, and evaluation metrics.
        """
        return self.walkforward(windows=1, lookahead=lookahead, start_date=start_date, end_date=end_date, timeframe=timeframe, between_time=between_time, days=days, train_size=train_size, shuffle=shuffle, calc_bootstrap=calc_bootstrap, disable_parallel=disable_parallel, warmup=warmup, portfolio=portfolio, adjust=adjust)

    def walkforward(self, windows: int, lookahead: int=1, start_date: Optional[Union[str, datetime]]=None, end_date: Optional[Union[str, datetime]]=None, timeframe: str='', between_time: Optional[tuple[str, str]]=None, days: Optional[Union[str, Day, Iterable[Union[str, Day]]]]=None, train_size: float=0.5, shuffle: bool=False, calc_bootstrap: bool=False, disable_parallel: bool=False, warmup: Optional[int]=None, portfolio: Optional[Portfolio]=None, adjust: Optional[Any]=None) -> TestResult:
        """Backtests the trading strategy using `Walkforward Analysis
        <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.
        Backtesting data supplied by the :class:`pybroker.data.DataSource` is
        divided into ``windows`` number of equal sized time windows, with each
        window split into train and test data as specified by ``train_size``.
        The backtest "walks forward" in time through each window, running
        executions that were added with :meth:`.add_execution`.

        Args:
            windows: Number of walkforward time windows.
            start_date: Starting date of the Walkforward Analysis (inclusive).
                Must be within ``start_date`` and ``end_date`` range that was
                passed to :meth:`.__init__`.
            end_date: Ending date of the Walkforward Analysis (inclusive). Must
                be within ``start_date`` and ``end_date`` range that was passed
                to :meth:`.__init__`.
            timeframe: Formatted string that specifies the timeframe
                resolution of the backtesting data. The timeframe string
                supports the following units:

                - ``"s"``/``"sec"``: seconds
                - ``"m"``/``"min"``: minutes
                - ``"h"``/``"hour"``: hours
                - ``"d"``/``"day"``: days
                - ``"w"``/``"week"``: weeks

                An example timeframe string is ``1h 30m``.
            between_time: ``tuple[str, str]`` of times of day e.g.
                ('9:30', '16:00') used to filter the backtesting data
                (inclusive).
            days: Days (e.g. ``"mon"``, ``"tues"`` etc.) used to filter the
                backtesting data.
            lookahead: Number of bars in the future of the target prediction.
                For example, predicting returns for the next bar would have a
                ``lookahead`` of ``1``. This quantity is needed to prevent
                training data from leaking into the test boundary.
            train_size: Amount of :class:`pybroker.data.DataSource` data to use
                for training, where the max ``train_size`` is ``1``. For
                example, a ``train_size`` of ``0.9`` would result in 90% of
                data being used for training and the remaining 10% of data
                being used for testing.
            shuffle: Whether to randomly shuffle the data used for training.
                Defaults to ``False``. Disabled when model caching is enabled
                via :meth:`pybroker.cache.enable_model_cache`.
            calc_bootstrap: Whether to compute randomized bootstrap evaluation
                metrics. Defaults to ``False``.
            disable_parallel: If ``True``,
                :class:`pybroker.indicator.Indicator` data is computed
                serially. If ``False``, :class:`pybroker.indicator.Indicator`
                data is computed in parallel using multiple processes.
                Defaults to ``False``.
            warmup: Number of bars that need to pass before running the
                executions.
            portfolio: Custom :class:`pybroker.portfolio.Portfolio` to use for
                backtests.
            adjust: The type of adjustment to make to the
                :class:`pybroker.data.DataSource`.

        Returns:
            :class:`.TestResult` containing portfolio balances, order
            history, and evaluation metrics.
        """
        if warmup is not None and warmup < 1:
            raise ValueError('warmup must be > 0.')
        scope = StaticScope.instance()
        try:
            scope.freeze_data_cols()
            if not self._executions:
                raise ValueError('No executions were added.')
            start_dt = self._start_date if start_date is None else to_datetime(start_date)
            if start_dt < self._start_date or start_dt > self._end_date:
                raise ValueError(f'start_date must be between {self._start_date} and {self._end_date}.')
            end_dt = self._end_date if end_date is None else to_datetime(end_date)
            if end_dt < self._start_date or end_dt > self._end_date:
                raise ValueError(f'end_date must be between {self._start_date} and {self._end_date}.')
            if start_dt is not None and end_dt is not None:
                verify_date_range(start_dt, end_dt)
            self._logger.walkforward_start(start_dt, end_dt)
            df = self._fetch_data(timeframe, adjust)
            day_ids = self._to_day_ids(days)
            df = self._filter_dates(df=df, start_date=start_dt, end_date=end_dt, between_time=between_time, days=day_ids)
            tf_seconds = to_seconds(timeframe)
            indicator_data = self._fetch_indicators(df=df, cache_date_fields=CacheDateFields(start_date=start_dt, end_date=end_dt, tf_seconds=tf_seconds, between_time=between_time, days=day_ids), disable_parallel=disable_parallel)
            train_only = self._before_exec_fn is None and self._after_exec_fn is None and all(map(lambda e: e.fn is None, self._executions))
            if portfolio is None:
                portfolio = Portfolio(self._config.initial_cash, self._config.fee_mode, self._config.fee_amount, self._config.subtract_fees, self._fractional_shares_enabled(), self._config.position_mode, self._config.max_long_positions, self._config.max_short_positions, self._config.return_stops)
            signals = self._run_walkforward(portfolio=portfolio, df=df, indicator_data=indicator_data, tf_seconds=tf_seconds, between_time=between_time, days=day_ids, windows=windows, lookahead=lookahead, train_size=train_size, shuffle=shuffle, train_only=train_only, warmup=warmup)
            if train_only:
                self._logger.walkforward_completed()
            return self._to_test_result(start_dt, end_dt, portfolio, calc_bootstrap, train_only, signals if self._config.return_signals else None)
        finally:
            scope.unfreeze_data_cols()

    def _to_day_ids(self, days: Optional[Union[str, Day, Iterable[Union[str, Day]]]]) -> Optional[tuple[int]]:
        if days is None:
            return None
        days = (days,) if isinstance(days, str) or isinstance(days, Day) else days
        return tuple(sorted((day.value if isinstance(day, Day) else Day[day.upper()].value for day in set(days))))

    def _fractional_shares_enabled(self):
        return self._config.enable_fractional_shares or isinstance(self._data_source, AlpacaCrypto)

    def _run_walkforward(self, portfolio: Portfolio, df: pd.DataFrame, indicator_data: dict[IndicatorSymbol, pd.Series], tf_seconds: int, between_time: Optional[tuple[str, str]], days: Optional[tuple[int]], windows: int, lookahead: int, train_size: float, shuffle: bool, train_only: bool, warmup: Optional[int]) -> dict[str, pd.DataFrame]:
        sessions: dict[str, dict] = defaultdict(dict)
        exit_dates: dict[str, np.datetime64] = {}
        if self._config.exit_on_last_bar:
            for exec in self._executions:
                for sym in exec.symbols:
                    sym_dates = df[df[DataCol.SYMBOL.value] == sym][DataCol.DATE.value].values
                    if len(sym_dates):
                        sym_dates.sort()
                        exit_dates[sym] = sym_dates[-1]
        signals: dict[str, pd.DataFrame] = {}
        for train_idx, test_idx in self.walkforward_split(df=df, windows=windows, lookahead=lookahead, train_size=train_size, shuffle=shuffle):
            models: dict[ModelSymbol, TrainedModel] = {}
            train_data = df.loc[train_idx]
            test_data = df.loc[test_idx]
            if not train_data.empty:
                model_syms = {ModelSymbol(model_name, sym) for sym in train_data[DataCol.SYMBOL.value].unique() for execution in self._executions for model_name in execution.model_names if sym in execution.symbols}
                train_dates = get_unique_sorted_dates(train_data[DataCol.DATE.value])
                models = self.train_models(model_syms=model_syms, train_data=train_data, test_data=test_data, indicator_data=indicator_data, cache_date_fields=CacheDateFields(start_date=to_datetime(train_dates[0]), end_date=to_datetime(train_dates[-1]), tf_seconds=tf_seconds, between_time=between_time, days=days))
            if test_data.empty:
                return signals
            split_signals = self.backtest_executions(config=self._config, executions=self._executions, before_exec_fn=self._before_exec_fn, after_exec_fn=self._after_exec_fn, sessions=sessions, models=models, indicator_data=indicator_data, test_data=test_data, portfolio=portfolio, pos_size_handler=self._pos_size_handler, exit_dates=exit_dates, train_only=train_only, slippage_model=self._slippage_model, enable_fractional_shares=self._fractional_shares_enabled(), round_fill_price=self._config.round_fill_price, warmup=warmup)
            for sym, signals_df in split_signals.items():
                if sym in signals:
                    signals[sym] = pd.concat([signals[sym], signals_df])
                else:
                    signals[sym] = signals_df
        return signals

    def _filter_dates(self, df: pd.DataFrame, start_date: datetime, end_date: datetime, between_time: Optional[tuple[str, str]], days: Optional[tuple[int]]) -> pd.DataFrame:
        if start_date != self._start_date or end_date != self._end_date:
            df = _between(df, start_date, end_date).reset_index(drop=True)
        if df[DataCol.DATE.value].dt.tz is not None:
            df[DataCol.DATE.value] = df[DataCol.DATE.value].dt.tz_convert(None)
        is_time_range = between_time is not None or days is not None
        if is_time_range:
            df = df.reset_index(drop=True).set_index(DataCol.DATE.value)
        if days is not None:
            self._logger.info_walkforward_on_days(days)
            df = df[df.index.weekday.isin(frozenset(days))]
        if between_time is not None:
            if len(between_time) != 2:
                raise ValueError(f'between_time must be a tuple[str, str] of start time and end time, received {between_time!r}.')
            self._logger.info_walkforward_between_time(between_time)
            df = df.between_time(*between_time)
        if is_time_range:
            df = df.reset_index()
        return df

    def _fetch_indicators(self, df: pd.DataFrame, cache_date_fields: CacheDateFields, disable_parallel: bool) -> dict[IndicatorSymbol, pd.Series]:
        indicator_syms = set()
        for execution in self._executions:
            for sym in execution.symbols:
                for model_name in execution.model_names:
                    ind_names = self._scope.get_indicator_names(model_name)
                    for ind_name in ind_names:
                        indicator_syms.add(IndicatorSymbol(ind_name, sym))
                for ind_name in execution.indicator_names:
                    indicator_syms.add(IndicatorSymbol(ind_name, sym))
        return self.compute_indicators(df=df, indicator_syms=indicator_syms, cache_date_fields=cache_date_fields, disable_parallel=disable_parallel)

    def _fetch_data(self, timeframe: str, adjust: Optional[Any]) -> pd.DataFrame:
        unique_syms = {sym for execution in self._executions for sym in execution.symbols}
        if isinstance(self._data_source, DataSource):
            df = self._data_source.query(unique_syms, self._start_date, self._end_date, timeframe, adjust)
        else:
            df = _between(self._data_source, self._start_date, self._end_date)
            df = df[df[DataCol.SYMBOL.value].isin(unique_syms)]
        if df.empty:
            raise ValueError('DataSource is empty.')
        return df.reset_index(drop=True)

    def _to_test_result(self, start_date: datetime, end_date: datetime, portfolio: Portfolio, calc_bootstrap: bool, train_only: bool, signals: Optional[dict[str, pd.DataFrame]]) -> TestResult:
        if train_only:
            return TestResult(start_date=start_date, end_date=end_date, portfolio=pd.DataFrame(), positions=pd.DataFrame(), orders=pd.DataFrame(), trades=pd.DataFrame(), metrics=EvalMetrics(), metrics_df=pd.DataFrame(), bootstrap=None, signals=signals, stops=None)
        pos_df = pd.DataFrame.from_records(portfolio.position_bars, columns=PositionBar._fields)
        for col in ('close', 'equity', 'market_value', 'margin', 'unrealized_pnl'):
            pos_df[col] = quantize(pos_df, col, self._config.round_test_result)
        pos_df.set_index(['symbol', 'date'], inplace=True)
        portfolio_df = pd.DataFrame.from_records(portfolio.bars, columns=PortfolioBar._fields, index='date')
        for col in ('cash', 'equity', 'margin', 'market_value', 'pnl', 'unrealized_pnl', 'fees'):
            portfolio_df[col] = quantize(portfolio_df, col, self._config.round_test_result)
        orders_df = pd.DataFrame.from_records(portfolio.orders, columns=Order._fields, index='id')
        for col in ('limit_price', 'fill_price', 'fees'):
            orders_df[col] = quantize(orders_df, col, self._config.round_test_result)
        trades_df = pd.DataFrame.from_records(portfolio.trades, columns=Trade._fields, index='id')
        trades_df['bars'] = trades_df['bars'].astype(int)
        for col in ('entry', 'exit', 'pnl', 'return_pct', 'agg_pnl', 'pnl_per_bar', 'mae', 'mfe'):
            trades_df[col] = quantize(trades_df, col, self._config.round_test_result)
        shares_type = float if self._fractional_shares_enabled() else int
        pos_df['long_shares'] = pos_df['long_shares'].astype(shares_type)
        pos_df['short_shares'] = pos_df['short_shares'].astype(shares_type)
        orders_df['shares'] = orders_df['shares'].astype(shares_type)
        trades_df['shares'] = trades_df['shares'].astype(shares_type)
        eval_result = self.evaluate(portfolio_df=portfolio_df, trades_df=trades_df, calc_bootstrap=calc_bootstrap, bootstrap_sample_size=self._config.bootstrap_sample_size, bootstrap_samples=self._config.bootstrap_samples, bars_per_year=self._config.bars_per_year)
        metrics = [(k, v) for k, v in dataclasses.asdict(eval_result.metrics).items() if v is not None]
        metrics_df = pd.DataFrame(metrics, columns=['name', 'value'])
        stops_df = None
        if self._config.return_stops:
            stops_df = pd.DataFrame.from_records(portfolio._stop_records, columns=StopRecord._fields)
        self._logger.walkforward_completed()
        return TestResult(start_date=start_date, end_date=end_date, portfolio=portfolio_df, positions=pos_df, orders=orders_df, trades=trades_df, metrics=eval_result.metrics, metrics_df=metrics_df, bootstrap=eval_result.bootstrap, signals=signals, stops=stops_df)

def _between(df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df[DataCol.DATE.value].dt.tz_localize(None) >= start_date) & (df[DataCol.DATE.value].dt.tz_localize(None) <= end_date)]

