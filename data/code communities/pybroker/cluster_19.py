# Cluster 19

@njit
def bca_boot_conf(x: NDArray[np.float64], n: int, n_boot: int, fn: Callable[[NDArray[np.float64]], float]) -> BootConfIntervals:
    """Computes confidence intervals for a user-defined parameter using the
    `bias corrected and accelerated (BCa) bootstrap method.
    <https://blogs.sas.com/content/iml/2017/07/12/bootstrap-bca-interval.html>`_

    Args:
        x: :class:`numpy.ndarray` containing the data for the randomized
            bootstrap sampling.
        n: Number of elements in each random bootstrap sample.
        n_boot: Number of random bootstrap samples to use.
        fn: :class:`Callable` for computing the parameter used for the
            confidence intervals.

    Returns:
        :class:`.BootConfIntervals` containing the computed confidence
        intervals.
    """
    if n <= 0:
        raise ValueError('Bootstrap sample size must be greater than 0.')
    if n_boot <= 0:
        raise ValueError('Number of boostrap samples must be greater than 0.')
    n_x = len(x)
    if not n_x:
        return BootConfIntervals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if n_x <= n:
        n = n_x
        n_boot = 1

    def clamp(k: int):
        return min(max(k, 0), n_boot - 1)
    x_buff = np.zeros(n)
    boot = np.zeros(n_boot)
    theta_hat = fn(x[:n])
    z0_count = 0
    for i in range(n_boot):
        for j in range(n):
            k = np.random.choice(n_x)
            x_buff[j] = x[k]
        param = fn(x_buff)
        boot[i] = param
        if param < theta_hat:
            z0_count += 1
    z0_count = min(z0_count, n_boot - 1)
    z0_count = max(z0_count, 1)
    z0 = inverse_normal_cdf(z0_count / n_boot)
    theta_dot = 0.0
    for i in range(n):
        x_temp, x[i] = (x[i], x[n - 1])
        param = fn(x[:n - 1])
        theta_dot += param
        x_buff[i] = param
        x[i] = x_temp
    theta_dot /= n
    numer = denom = 0
    for i in range(n):
        diff = theta_dot - x_buff[i]
        diff_sq = diff ** 2
        denom += diff_sq
        numer += diff_sq * diff
    denom = np.power(np.sqrt(denom), 3)
    accel = numer / (6 * denom + 1e-60)
    boot.sort()
    zlo = inverse_normal_cdf(0.025)
    zhi = inverse_normal_cdf(0.975)
    alo = normal_cdf(z0 + (z0 + zlo) / (1 - accel * (z0 + zlo)))
    ahi = normal_cdf(z0 + (z0 + zhi) / (1 - accel * (z0 + zhi)))
    k = int(alo * (n_boot + 1)) - 1
    k = clamp(k)
    low_2p5 = boot[k]
    k = int((1 - ahi) * (n_boot + 1)) - 1
    k = clamp(k)
    high_2p5 = boot[n_boot - 1 - k]
    zlo = inverse_normal_cdf(0.05)
    zhi = inverse_normal_cdf(0.95)
    alo = normal_cdf(z0 + (z0 + zlo) / (1 - accel * (z0 + zlo)))
    ahi = normal_cdf(z0 + (z0 + zhi) / (1 - accel * (z0 + zhi)))
    k = int(alo * (n_boot + 1)) - 1
    k = clamp(k)
    low_5 = boot[k]
    k = int((1 - ahi) * (n_boot + 1)) - 1
    k = clamp(k)
    high_5 = boot[n_boot - 1 - k]
    zlo = inverse_normal_cdf(0.1)
    zhi = inverse_normal_cdf(0.9)
    alo = normal_cdf(z0 + (z0 + zlo) / (1 - accel * (z0 + zlo)))
    ahi = normal_cdf(z0 + (z0 + zhi) / (1 - accel * (z0 + zhi)))
    k = int(alo * (n_boot + 1)) - 1
    k = clamp(k)
    low_10 = boot[k]
    k = int((1 - ahi) * (n_boot + 1)) - 1
    k = clamp(k)
    high_10 = boot[n_boot - 1 - k]
    return BootConfIntervals(low_2p5, high_2p5, low_5, high_5, low_10, high_10)

@njit
def inverse_normal_cdf(p: float) -> float:
    """Computes the inverse CDF of the standard normal distribution."""
    pp = p if p <= 0.5 else 1 - p
    if pp == 0:
        pp = 1e-10
    t = np.sqrt(np.log(1 / (pp * pp)))
    numer = (0.010328 * t + 0.802853) * t + 2.515517
    denom = ((0.001308 * t + 0.189269) * t + 1.432788) * t + 1
    x = t - numer / denom
    return -x if p <= 0.5 else x

def clamp(k: int):
    return min(max(k, 0), n_boot - 1)

def conf_profit_factor(x: NDArray[np.float64], n: int, n_boot: int) -> BootConfIntervals:
    """Computes confidence intervals for :func:`.profit_factor`."""
    intervals = bca_boot_conf(x, n, n_boot, log_profit_factor)
    return BootConfIntervals(low_2p5=np.exp(intervals.low_2p5), high_2p5=np.exp(intervals.high_2p5), low_5=np.exp(intervals.low_5), high_5=np.exp(intervals.high_5), low_10=np.exp(intervals.low_10), high_10=np.exp(intervals.high_10))

def conf_sharpe_ratio(x: NDArray[np.float64], n: int, n_boot: int, obs: Optional[int]=None) -> BootConfIntervals:
    """Computes confidence intervals for :func:`.sharpe_ratio`."""
    intervals = bca_boot_conf(x, n, n_boot, sharpe_ratio)
    if obs is not None:
        factor = np.sqrt(obs)
        intervals = BootConfIntervals(low_2p5=intervals.low_2p5 * factor, high_2p5=intervals.high_2p5 * factor, low_5=intervals.low_5 * factor, high_5=intervals.high_5 * factor, low_10=intervals.low_10 * factor, high_10=intervals.high_10 * factor)
    return intervals

class EvaluateMixin:
    """Mixin for computing evaluation metrics."""

    def evaluate(self, portfolio_df: pd.DataFrame, trades_df: pd.DataFrame, calc_bootstrap: bool, bootstrap_sample_size: int, bootstrap_samples: int, bars_per_year: Optional[int]) -> EvalResult:
        """Computes evaluation metrics.

        Args:
            portfolio_df: :class:`pandas.DataFrame` of portfolio market values
                per bar.
            trades_df: :class:`pandas.DataFrame` of trades.
            calc_bootstrap: ``True`` to calculate randomized bootstrap metrics.
            bootstrap_sample_size: Size of each random bootstrap sample.
            bootstrap_samples: Number of random bootstrap samples to use.
            bars_per_year: Number of observations per years that will be used
                to annualize evaluation metrics. For example, a value of
                ``252`` would be used to annualize the Sharpe Ratio for daily
                returns.

        Returns:
            :class:`.EvalResult` containing evaluation metrics.
        """
        market_values = portfolio_df['market_value'].to_numpy()
        fees = portfolio_df['fees'].to_numpy()
        bar_returns = self._calc_bar_returns(portfolio_df)
        bar_return_dates = bar_returns.index.to_series().reset_index(drop=True)
        bar_returns = bar_returns.to_numpy()
        bar_changes = self._calc_bar_changes(portfolio_df)
        if not len(market_values) or not len(bar_returns) or (not len(bar_changes)):
            return EvalResult(EvalMetrics(), None)
        pnls = trades_df['pnl'].to_numpy()
        return_pcts = trades_df['return_pct'].to_numpy()
        bars = trades_df['bars'].to_numpy()
        winning_trades = trades_df[trades_df['pnl'] > 0]
        winning_bars = winning_trades['bars'].to_numpy()
        losing_trades = trades_df[trades_df['pnl'] < 0]
        losing_bars = losing_trades['bars'].to_numpy()
        largest_win = winning_trades[winning_trades['pnl'] == winning_trades['pnl'].max()]
        largest_win_pct = 0 if largest_win.empty else largest_win['return_pct'].values[0]
        largest_win_bars = 0 if largest_win.empty else largest_win['bars'].values[0]
        largest_loss = losing_trades[losing_trades['pnl'] == losing_trades['pnl'].min()]
        largest_loss_pct = 0 if largest_loss.empty else largest_loss['return_pct'].values[0]
        largest_loss_bars = 0 if largest_loss.empty else largest_loss['bars'].values[0]
        metrics = self._calc_eval_metrics(market_values, bar_changes, bar_returns, bar_return_dates, pnls, return_pcts, bars=bars, winning_bars=winning_bars, losing_bars=losing_bars, largest_win_num_bars=largest_win_bars, largest_win_pct=largest_win_pct, largest_loss_num_bars=largest_loss_bars, largest_loss_pct=largest_loss_pct, fees=fees, bars_per_year=bars_per_year)
        logger = StaticScope.instance().logger
        if not calc_bootstrap:
            return EvalResult(metrics, None)
        if len(bar_returns) <= bootstrap_sample_size:
            logger.warn_bootstrap_sample_size(len(bar_returns), bootstrap_sample_size)
        logger.calc_bootstrap_metrics_start(samples=bootstrap_samples, sample_size=bootstrap_sample_size)
        confs_result = self._calc_conf_intervals(changes=bar_changes, returns=bar_returns, sample_size=bootstrap_sample_size, samples=bootstrap_samples, bars_per_year=bars_per_year)
        dd_result = self._calc_drawdown_conf(changes=bar_changes, returns=bar_returns, sample_size=bootstrap_sample_size, samples=bootstrap_samples)
        bootstrap = BootstrapResult(conf_intervals=confs_result.df, drawdown_conf=dd_result.df, profit_factor=confs_result.profit_factor, sharpe=confs_result.sharpe, drawdown=dd_result.metrics)
        logger.calc_bootstrap_metrics_completed()
        return EvalResult(metrics, bootstrap)

    def _calc_bar_returns(self, df: pd.DataFrame) -> pd.Series:
        prev_market_value = df['market_value'].shift(1)
        returns = (df['market_value'] - prev_market_value) / prev_market_value
        return returns.dropna()

    def _calc_bar_changes(self, df: pd.DataFrame) -> NDArray[np.float64]:
        changes = df['market_value'] - df['market_value'].shift(1)
        return changes.dropna().to_numpy()

    def _calc_eval_metrics(self, market_values: NDArray[np.float64], bar_changes: NDArray[np.float64], bar_returns: NDArray[np.float64], bar_return_dates: pd.Series, pnls: NDArray[np.float64], return_pcts: NDArray[np.float64], bars: NDArray[np.int_], winning_bars: NDArray[np.int_], losing_bars: NDArray[np.int_], largest_win_num_bars: int, largest_win_pct: float, largest_loss_num_bars: int, largest_loss_pct: float, fees: NDArray[np.float64], bars_per_year: Optional[int]) -> EvalMetrics:
        total_fees = fees[-1] if len(fees) else 0
        max_dd = max_drawdown(bar_changes)
        max_dd_pct, max_dd_index = max_drawdown_percent(bar_returns)
        max_dd_date = bar_return_dates.iloc[max_dd_index].to_pydatetime() if max_dd_index else None
        sharpe = sharpe_ratio(bar_returns, bars_per_year)
        sortino = sortino_ratio(bar_returns, bars_per_year)
        pf = profit_factor(bar_changes)
        r2 = r_squared(market_values)
        ui = ulcer_index(market_values)
        upi_ = upi(market_values, ui=ui)
        std_error = float(np.std(market_values))
        largest_win = 0.0
        largest_loss = 0.0
        win_rate = 0.0
        loss_rate = 0.0
        winning_trades = 0
        losing_trades = 0
        avg_pnl = 0.0
        avg_return_pct = 0.0
        avg_trade_bars = 0.0
        avg_profit = 0.0
        avg_loss = 0.0
        avg_profit_pct = 0.0
        avg_loss_pct = 0.0
        avg_winning_trade_bars = 0.0
        avg_losing_trade_bars = 0.0
        total_profit = 0.0
        total_loss = 0.0
        total_pnl = 0.0
        unrealized_pnl = 0.0
        max_wins = 0
        max_losses = 0
        if len(pnls):
            largest_win, largest_loss = largest_win_loss(pnls)
            win_rate, loss_rate = win_loss_rate(pnls)
            winning_trades, losing_trades = winning_losing_trades(pnls)
            avg_profit, avg_loss = avg_profit_loss(pnls)
            avg_profit_pct, avg_loss_pct = avg_profit_loss(return_pcts)
            total_profit, total_loss = total_profit_loss(pnls)
            max_wins, max_losses = max_wins_losses(pnls)
            total_pnl = float(np.sum(pnls))
            if len(pnls):
                avg_pnl = float(np.mean(pnls))
            if len(return_pcts):
                avg_return_pct = float(np.mean(return_pcts))
            if len(bars):
                avg_trade_bars = float(np.mean(bars))
            if len(winning_bars):
                avg_winning_trade_bars = float(np.mean(winning_bars))
            if len(losing_bars):
                avg_losing_trade_bars = float(np.mean(losing_bars))
        total_return_pct = total_return_percent(initial_value=market_values[0], pnl=total_pnl)
        unrealized_pnl = market_values[-1] - market_values[0] - total_pnl
        annual_return_pct = None
        annual_std_error = None
        annual_volatility_pct = None
        calmar = None
        if bars_per_year is not None:
            annual_return_pct = annual_total_return_percent(initial_value=market_values[0], pnl=total_pnl, bars_per_year=bars_per_year, total_bars=len(market_values))
            annual_std_error = std_error * np.sqrt(bars_per_year)
            annual_volatility_pct = float(np.std(bar_returns * 100) * np.sqrt(bars_per_year))
            calmar = calmar_ratio(bar_returns, bars_per_year)
        return EvalMetrics(trade_count=len(pnls), initial_market_value=market_values[0], end_market_value=market_values[-1], max_drawdown=max_dd, max_drawdown_pct=max_dd_pct, max_drawdown_date=max_dd_date, largest_win=largest_win, largest_win_pct=largest_win_pct, largest_win_bars=largest_win_num_bars, largest_loss=largest_loss, largest_loss_pct=largest_loss_pct, largest_loss_bars=largest_loss_num_bars, max_wins=max_wins, max_losses=max_losses, win_rate=win_rate, loss_rate=loss_rate, winning_trades=winning_trades, losing_trades=losing_trades, avg_pnl=avg_pnl, avg_return_pct=avg_return_pct, avg_trade_bars=avg_trade_bars, avg_profit=avg_profit, avg_profit_pct=avg_profit_pct, avg_winning_trade_bars=avg_winning_trade_bars, avg_loss=avg_loss, avg_loss_pct=avg_loss_pct, avg_losing_trade_bars=avg_losing_trade_bars, total_profit=total_profit, total_loss=total_loss, total_pnl=total_pnl, unrealized_pnl=unrealized_pnl, total_return_pct=total_return_pct, annual_return_pct=annual_return_pct, total_fees=total_fees, sharpe=sharpe, sortino=sortino, calmar=calmar, profit_factor=pf, equity_r2=r2, ulcer_index=ui, upi=upi_, std_error=std_error, annual_std_error=annual_std_error, annual_volatility_pct=annual_volatility_pct)

    def _calc_conf_intervals(self, changes: NDArray[np.float64], returns: NDArray[np.float64], sample_size: int, samples: int, bars_per_year: Optional[int]) -> _ConfsResult:
        pf_intervals = conf_profit_factor(changes, sample_size, samples)
        pf_conf = self._to_conf_intervals('Profit Factor', pf_intervals)
        sr_intervals = conf_sharpe_ratio(returns, sample_size, samples, bars_per_year)
        sharpe_conf = self._to_conf_intervals('Sharpe Ratio', sr_intervals)
        df = pd.DataFrame.from_records(pf_conf + sharpe_conf, columns=ConfInterval._fields)
        df.set_index(['name', 'conf'], inplace=True)
        return _ConfsResult(df=df, profit_factor=pf_intervals, sharpe=sr_intervals)

    def _to_conf_intervals(self, name: str, conf: BootConfIntervals) -> deque[ConfInterval]:
        results: deque[ConfInterval] = deque()
        results.append(ConfInterval(name, '97.5%', conf.low_2p5, conf.high_2p5))
        results.append(ConfInterval(name, '95%', conf.low_5, conf.high_5))
        results.append(ConfInterval(name, '90%', conf.low_10, conf.high_10))
        return results

    def _calc_drawdown_conf(self, changes: NDArray[np.float64], returns: NDArray[np.float64], sample_size: int, samples: int) -> _DrawdownResult:
        metrics = drawdown_conf(changes, returns, sample_size, samples)
        df = pd.DataFrame(zip(('99.9%', '99%', '95%', '90%'), *metrics), columns=('conf', 'amount', 'percent'))
        df.set_index('conf', inplace=True)
        return _DrawdownResult(df=df, metrics=metrics)

@pytest.mark.parametrize('n, n_boot, expected_msg', [(0, 100, 'Bootstrap sample size must be greater than 0.'), (-1, 100, 'Bootstrap sample size must be greater than 0.'), (10, 0, 'Number of boostrap samples must be greater than 0.'), (10, -1, 'Number of boostrap samples must be greater than 0.')])
def test_bca_boot_conf_when_invalid_params_then_error(n, n_boot, expected_msg):
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        bca_boot_conf(np.random.rand(100), n, n_boot, profit_factor)

@pytest.mark.parametrize('n, n_boot', [(1, 100), (1, 1), (10, 100), (10, 1)])
def test_conf_profit_factor(n, n_boot, rand_values):
    intervals = conf_profit_factor(rand_values, n, n_boot)
    assert len(intervals) == 6

@pytest.mark.parametrize('n, n_boot', [(1, 100), (1, 1), (10, 100), (10, 1)])
def test_conf_sharpe_ratio(n, n_boot, rand_values):
    intervals = conf_sharpe_ratio(rand_values, n, n_boot)
    assert len(intervals) == 6

