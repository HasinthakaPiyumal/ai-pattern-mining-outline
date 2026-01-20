# Cluster 20

@njit
def log_profit_factor(changes: NDArray[np.float64]) -> np.floating:
    """Computes the log transformed profit factor, which is the ratio of gross
    profit to gross loss.

    Args:
        changes: Array of differences between each bar and the previous bar.
    """
    return profit_factor(changes, use_log=True)

@njit
def profit_factor(changes: NDArray[np.float64], use_log: bool=False) -> np.floating:
    """Computes the profit factor, which is the ratio of gross profit to gross
    loss.

    Args:
        changes: Array of differences between each bar and the previous bar.
        use_log: Whether to log transform the profit factor. Defaults to False.
    """
    wins = changes[changes > 0]
    losses = changes[changes < 0]
    if not len(wins) and (not len(losses)):
        return np.float64(0)
    numer = denom = 1e-10
    numer += np.sum(wins)
    denom -= np.sum(losses)
    if use_log:
        return np.log(numer / denom)
    else:
        return np.divide(numer, denom)

def sortino_ratio(returns: NDArray[np.float64], obs: Optional[int]=None) -> float:
    """Computes the
    `Sortino Ratio <https://en.wikipedia.org/wiki/Sortino_ratio>`_.

    Args:
        returns: Array of returns centered at 0.
        obs: Number of observations used to annualize the Sortino Ratio. For
            example, a value of ``252`` would be used to annualize daily
            returns.
    """
    return float(sharpe_ratio(returns, obs, downside_only=True))

@njit
def sharpe_ratio(returns: NDArray[np.float64], obs: Optional[int]=None, downside_only: bool=False) -> np.floating:
    """Computes the
    `Sharpe Ratio <https://en.wikipedia.org/wiki/Sharpe_ratio>`_.

    Args:
        returns: Array of returns centered at 0.
        obs: Number of observations used to annualize the Sharpe Ratio. For
            example, a value of ``252`` would be used to annualize daily
            returns.
    """
    std_changes = returns[returns < 0] if downside_only else returns
    if not len(std_changes):
        return np.float64(0)
    std = np.std(std_changes)
    if std == 0:
        return np.float64(0)
    sr = np.mean(returns) / std
    if obs is not None:
        sr *= np.sqrt(obs)
    return sr

def calmar_ratio(returns: NDArray[np.float64], bars_per_year: int) -> float:
    """Computes the Calmar Ratio.

    Args:
        returns: Array of returns centered at 0.
        bars_per_year: Number of bars per annum.
    """
    if not len(returns):
        return 0
    max_dd = np.abs(max_drawdown(returns))
    if max_dd == 0:
        return 0
    return np.mean(returns) * bars_per_year / max_dd

@njit
def ulcer_index(values: NDArray[np.float64], period: int=14) -> float:
    """Computes the
    `Ulcer Index <https://en.wikipedia.org/wiki/Ulcer_index>`_ of ``values``.
    """
    n = len(values)
    if n <= period:
        return 0
    start = period - 1
    dd = np.zeros(n - start)
    max_values = highv(values, period)
    for i in range(start, n):
        if max_values[i] == 0:
            dd[i - start] = 0
            continue
        dd[i - start] = (values[i] - max_values[i]) / max_values[i] * 100
    return np.sqrt(np.mean(np.square(dd)))

@njit
def upi(values: NDArray[np.float64], period: int=14, ui: Optional[float]=None) -> float:
    """Computes the `Ulcer Performance Index
    <https://en.wikipedia.org/wiki/Ulcer_index>`_ of ``values``.
    """
    if len(values) <= 1:
        return 0
    if ui is None:
        ui = ulcer_index(values, period)
    if ui == 0:
        return 0
    r = np.zeros(len(values) - 1)
    for i in range(len(r)):
        r[i] = (values[i + 1] - values[i]) / values[i] * 100
    return float(np.mean(r) / ui)

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

def r_squared(values: NDArray[np.float64]) -> float:
    """Computes R-squared of ``values``."""
    n = len(values)
    if not n:
        return 0
    x = np.arange(n)
    try:
        coeffs = np.polyfit(x, values, 1)
        pred = np.poly1d(coeffs)(x)
        y_hat = np.mean(values)
        ssres = float(np.sum((values - pred) ** 2))
        sstot = float(np.sum((values - y_hat) ** 2))
        if sstot == 0:
            return 0
        return 1 - ssres / sstot
    except Exception:
        return 0

def largest_win_loss(pnls: NDArray[np.float64]) -> tuple[float, float]:
    """Computes the largest profit and largest loss of all trades.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[float, float]`` of largest profit and largest loss.
    """
    profits = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    return (np.max(profits) if len(profits) else 0, np.min(losses) if len(losses) else 0)

def win_loss_rate(pnls: NDArray[np.float64]) -> tuple[float, float]:
    """Computes the win rate and loss rate as percentages.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[float, float]`` of win rate and loss rate.
    """
    pnls = pnls[pnls != 0]
    n = len(pnls)
    if not n:
        return (0, 0)
    win_rate = len(pnls[pnls > 0]) / n * 100
    loss_rate = len(pnls[pnls < 0]) / n * 100
    return (win_rate, loss_rate)

def winning_losing_trades(pnls: NDArray[np.float64]) -> tuple[int, int]:
    """Returns the number of winning and losing trades.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[int, int]`` containing numbers of winning and losing trades.
    """
    pnls = pnls[pnls != 0]
    if not len(pnls):
        return (0, 0)
    return (len(pnls[pnls > 0]), len(pnls[pnls < 0]))

def avg_profit_loss(pnls: NDArray[np.float64]) -> tuple[float, float]:
    """Computes the average profit and average loss per trade.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[float, float]`` of average profit and average loss.
    """
    profits = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    return (float(np.mean(profits)) if len(profits) else 0, float(np.mean(losses)) if len(losses) else 0)

def total_profit_loss(pnls: NDArray[np.float64]) -> tuple[float, float]:
    """Computes total profit and loss.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[float, float]`` of total profit and total loss.
    """
    profits = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    return (np.sum(profits) if len(profits) else 0, np.sum(losses) if len(losses) else 0)

@njit
def max_wins_losses(pnls: NDArray[np.float64]) -> tuple[int, int]:
    """Computes the max consecutive wins and max consecutive losses.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[int, int]`` of max consecutive wins and max consecutive losses.
    """
    max_wins = max_losses = wins = losses = 0
    for pnl in pnls:
        if pnl > 0:
            wins += 1
            max_wins = max(max_wins, wins)
        else:
            wins = 0
        if pnl < 0:
            losses += 1
            max_losses = max(max_losses, losses)
        else:
            losses = 0
    return (max_wins, max_losses)

def total_return_percent(initial_value: float, pnl: float) -> float:
    """Computes total return as percentage.

    Args:
        initial_value: Initial value.
        pnl: Total profit and loss (PnL).
    """
    if initial_value == 0:
        return 0
    return ((pnl + initial_value) / initial_value - 1) * 100

def annual_total_return_percent(initial_value: float, pnl: float, bars_per_year: int, total_bars: int) -> float:
    """Computes annualized total return as percentage.

    Args:
        initial_value: Initial value.
        pnl: Total profit and loss (PnL).
        bars_per_year: Number of bars per annum.
        total_bars: Total number of bars of the return.
    """
    if initial_value == 0 or total_bars == 0:
        return 0
    return (np.power((pnl + initial_value) / initial_value, bars_per_year / total_bars) - 1) * 100

@pytest.mark.parametrize('values, expected_pf', [([0.1, -0.2, 0.3, 0, -0.4, 0.5], 1.499999), ([1, 1, 1, 1], 40000000001), ([1], 10000000001), ([-1], 0), ([0, 0, 0, 0], 0), ([], 0)])
def test_profit_factor(values, expected_pf):
    pf = profit_factor(np.array(values))
    assert truncate(pf, 6) == truncate(expected_pf, 6)

def truncate(value, n):
    return math.floor(value * 10 ** n) / 10 ** n

@pytest.mark.parametrize('values, obs, expected_sharpe', [([0.1, -0.2, 0.3, 0, -0.4, 0.5], None, 0.167443), ([0.1, -0.2, 0.3, 0, -0.4, 0.5], 252, 0.16744367165578425 * np.sqrt(252)), ([1, 1, 1, 1], None, 0), ([1], None, 0), ([], None, 0)])
def test_sharpe_ratio(values, obs, expected_sharpe):
    sharpe = sharpe_ratio(np.array(values), obs)
    assert truncate(sharpe, 6) == truncate(expected_sharpe, 6)

@pytest.mark.parametrize('values, obs, expected_sortino', [([0.1, -0.2, 0.3, 0, -0.4, 0.5], None, 0.499999), ([0.1, -0.2, 0.3, 0, -0.4, 0.5], 252, 0.4999999999999999 * np.sqrt(252)), ([1, 1, 1, 1], None, 0), ([1], None, 0), ([], None, 0)])
def test_sortino_ratio(values, obs, expected_sortino):
    sortino = sortino_ratio(np.array(values), obs)
    assert truncate(sortino, 6) == truncate(expected_sortino, 6)

@pytest.mark.parametrize('values, bars_per_year, expected_calmar', [([0.1, 0.15, -0.05, 0.1, -0.25, -0.15, 0], 252, -9), ([0.1, -0.4], 252, -94.5), ([1, 1, 1, 1], 252, 0), ([1], 252, 0), ([], 252, 0)])
def test_calmar_ratio(values, bars_per_year, expected_calmar):
    calmar = calmar_ratio(np.array(values), bars_per_year)
    assert truncate(calmar, 6) == expected_calmar

@pytest.mark.parametrize('values, expected_entropy', [([0.1, 0.2, 0.3, -0.2, 0.11, -0.3, -0.4, 0, 0.1, 0.2, 0.2], 0.782775), ([1, 1, 1, 1], 0), ([1], 0), ([], 0)])
def test_relative_entropy(values, expected_entropy):
    entropy = relative_entropy(np.array(values))
    assert truncate(entropy, 6) == expected_entropy

@njit
def relative_entropy(values: NDArray[np.float64]) -> float:
    """Computes the relative `entropy
    <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.
    """
    x = values[~np.isnan(values)]
    n = len(x)
    if not n:
        return 0
    n_bins = 3
    if n >= 10000:
        n_bins = 20
    elif n >= 1000:
        n_bins = 10
    elif n >= 100:
        n_bins = 5
    min_val = float(np.min(x))
    max_val = float(np.max(x))
    factor = (n_bins - 1e-10) / (max_val - min_val + 1e-60)
    count = np.zeros(n_bins)
    for v in x:
        k = int(factor * (v - min_val))
        count[k] += 1
    sum_ = 0
    for c in count:
        if c == 0:
            continue
        p = c / n
        sum_ += p * np.log(p)
    return -sum_ / np.log(n_bins)

@pytest.mark.parametrize('values, period, expected_ui', [([100, 101, 102, 100, 99, 103, 103, 102], 2, 0.909259), ([100, 101, 102, 100, 99, 103, 103, 102], 1, 0), ([0, 0, 0, 0, 0], 2, 0), ([1, 1, 1, 1, 1], 2, 0), ([100], 14, 0), ([100], 1, 0), ([], 2, 0)])
def test_ulcer_index(values, period, expected_ui):
    assert truncate(ulcer_index(np.array(values), period), 6) == expected_ui

@pytest.mark.parametrize('values, period', [([100, 101, 102], 0), ([100, 101, 102], -1)])
def test_ulcer_index_when_invalid_period_then_error(values, period):
    with pytest.raises(AssertionError, match=re.escape('n needs to be >= 1.')):
        ulcer_index(np.array(values), period)

@pytest.mark.parametrize('values, period, ui, expected_upi', [([100, 101, 102, 100, 99, 103, 103, 102], 2, None, 0.329757), ([100, 101, 102, 100, 99, 103, 103, 102], 2, 0, 0), ([100, 101, 102, 100, 99, 103, 103, 102], 2, 1, 0.299834), ([100, 101, 102, 100, 99, 103, 103, 102], 1, None, 0), ([0, 0, 0, 0, 0], 2, None, 0), ([1, 1, 1, 1, 1], 2, None, 0), ([100], 14, None, 0), ([100], 1, None, 0), ([], 2, None, 0), ([], 14, None, 0), ([], 14, 0, 0), ([], 14, 1.5, 0), ([100], 14, None, 0), ([100], 14, 0, 0), ([100], 14, 1.5, 0), ([100], 1, None, 0), ([100, 101], 14, None, 0), ([100, 101], 14, 0, 0), ([100, 101, 102], 2, 0, 0)])
def test_upi(values, period, ui, expected_upi):
    upi_ = upi(np.array(values), period=period, ui=ui)
    assert truncate(upi_, 6) == expected_upi

@pytest.mark.parametrize('values, period', [([100, 101, 102], 0), ([100, 101, 102], -1)])
def test_upi_when_invalid_period_then_error(values, period):
    with pytest.raises(AssertionError, match=re.escape('n needs to be >= 1.')):
        upi(np.array(values), period)

@pytest.mark.parametrize('values, expected_win_rate, expected_loss_rate', [([0.1, 0.2, 0.3, -0.2, 0.11, -0.3, -0.4, 0, 0.1, 0.2, 0.2], 70, 30), ([0.1], 100, 0), ([-0.1], 0, 100), ([0, 0, 0, 0, 0], 0, 0), ([], 0, 0)])
def test_win_loss_rate(values, expected_win_rate, expected_loss_rate):
    pnls = np.array(values)
    win_rate, loss_rate = win_loss_rate(pnls)
    assert win_rate == expected_win_rate
    assert loss_rate == expected_loss_rate

@pytest.mark.parametrize('values, expected_winning_trades, expected_losing_trades', [([0.1, 0.2, 0.3, -0.2, 0.11, -0.3, -0.4, 0, 0.1, 0.2, 0.2], 7, 3), ([0.1], 1, 0), ([-0.1], 0, 1), ([0, 0, 0, 0, 0], 0, 0), ([], 0, 0)])
def test_winning_losing_trades(values, expected_winning_trades, expected_losing_trades):
    pnls = np.array(values)
    winning_trades, losing_trades = winning_losing_trades(pnls)
    assert winning_trades == expected_winning_trades
    assert losing_trades == expected_losing_trades

@pytest.mark.parametrize('values, expected_profit, expected_loss', [([0.1, -0.2, 0.3, 0, -0.4, 0.5], 0.9, -0.6), ([0, 0, 0, 0, 0], 0, 0), ([0.1], 0.1, 0), ([-0.1], 0, -0.1), ([], 0, 0)])
def test_total_profit_loss(values, expected_profit, expected_loss):
    pnls = np.array(values)
    profit, loss = total_profit_loss(pnls)
    assert profit == expected_profit
    assert round(loss, 2) == expected_loss

@pytest.mark.parametrize('values, expected_profit, expected_loss', [([0.1, -0.2, 0.3, 0, -0.4, 0.5], 0.3, -0.3), ([1, 1, 1, 1, 1], 1, 0), ([-1, -1, -1, -1, -1], 0, -1), ([0, 0, 0, 0, 0], 0, 0), ([], 0, 0)])
def test_avg_profit_loss(values, expected_profit, expected_loss):
    pnls = np.array(values)
    profit, loss = avg_profit_loss(pnls)
    assert profit == expected_profit
    assert round(loss, 2) == expected_loss

@pytest.mark.parametrize('values, expected_win, expected_loss', [([0.1, 0.2, 0.3, -0.2, 0.11, -0.3, -0.4, 0, 0.1, 0.2, 0.2], 0.3, -0.4), ([1, 1, 1, 1, 1], 1, 0), ([-1, -1, -1, -1, -1], 0, -1), ([0, 0, 0, 0, 0], 0, 0), ([], 0, 0)])
def test_largest_win_loss(values, expected_win, expected_loss):
    pnls = np.array(values)
    win, loss = largest_win_loss(pnls)
    assert win == expected_win
    assert loss == expected_loss

@pytest.mark.parametrize('values, expected_wins, expected_losses', [([0.1, 0.2, 0.3, -0.2, 0.11, -0.3, -0.4, 0, 0.1, 0.2, 0.2], 3, 2), ([1, 1, 1, 1, 1], 5, 0), ([-1, -1, -1, -1, -1], 0, 5), ([0, 0, 0, 0, 0], 0, 0), ([], 0, 0)])
def test_max_wins_losses(values, expected_wins, expected_losses):
    pnls = np.array(values)
    wins, losses = max_wins_losses(pnls)
    assert wins == expected_wins
    assert losses == expected_losses

@pytest.mark.parametrize('values, expected_r2', [([1, 3, 5, 7, 8, 10, 11, 13], 0.992907), ([1], 0), ([-1], 0), ([1, 1, 1, 1, 1], 0), ([0, 0, 0, 0, 0], 0), ([], 0)])
def test_r_squared(values, expected_r2):
    r2 = r_squared(np.array(values))
    assert truncate(r2, 6) == expected_r2

@pytest.mark.parametrize('initial_value, pnl, expected_return', [(100, 10, 10), (0, 10, 0)])
def test_total_return_percent(initial_value, pnl, expected_return):
    return_pct = total_return_percent(initial_value, pnl)
    assert truncate(return_pct, 2) == expected_return

@pytest.mark.parametrize('initial_value, pnl, bars_per_year, total_bars, expected_return', [(100, 10, 252, 756, 3.22), (0, 10, 252, 756, 0), (100, 10, 252, 0, 0)])
def test_annual_total_return_percent(initial_value, pnl, bars_per_year, total_bars, expected_return):
    return_pct = annual_total_return_percent(initial_value, pnl, bars_per_year, total_bars)
    assert truncate(return_pct, 2) == expected_return

class TestEvaluateMixin:

    @pytest.mark.parametrize('bars_per_year, expected_sharpe, expected_sortino', [(None, 0.026013464180574847, 0.02727734785007549), (252, 0.026013464180574847 * np.sqrt(252), 0.02727734785007549 * np.sqrt(252))])
    @pytest.mark.parametrize('bootstrap_sample_size, bootstrap_samples', [(10, 100), (100000, 100)])
    def test_evaluate(self, bootstrap_sample_size, bootstrap_samples, portfolio_df, trades_df, calc_bootstrap, bars_per_year, expected_sharpe, expected_sortino):
        mixin = EvaluateMixin()
        result = mixin.evaluate(portfolio_df, trades_df, calc_bootstrap, bootstrap_sample_size=bootstrap_sample_size, bootstrap_samples=bootstrap_samples, bars_per_year=bars_per_year)
        assert result.metrics is not None
        if not calc_bootstrap:
            assert result.bootstrap is None
        else:
            assert result.bootstrap is not None
            assert result.bootstrap.conf_intervals is not None
            assert result.bootstrap.drawdown_conf is not None
            assert result.bootstrap.profit_factor is not None
            assert result.bootstrap.sharpe is not None
            assert result.bootstrap.drawdown is not None
            ci = result.bootstrap.conf_intervals
            assert ci.columns.tolist() == ['lower', 'upper']
            names = ci.index.get_level_values(0).unique().tolist()
            assert names == ['Profit Factor', 'Sharpe Ratio']
            for name in names:
                df = ci[ci.index.get_level_values(0) == name]
                confs = df.index.get_level_values(1).tolist()
                assert confs == ['97.5%', '95%', '90%']
            dd = result.bootstrap.drawdown_conf
            assert dd.columns.tolist() == ['amount', 'percent']
            conf = dd.index.get_level_values(0).tolist()
            assert conf == ['99.9%', '99%', '95%', '90%']
        metrics = result.metrics
        assert metrics.initial_market_value == 500000
        assert metrics.end_market_value == 693111.87
        assert metrics.total_pnl == 165740.2
        assert metrics.unrealized_pnl == metrics.end_market_value - metrics.initial_market_value - metrics.total_pnl
        assert metrics.total_return_pct == 33.14804
        assert metrics.total_profit == 403511.07999999996
        assert metrics.total_loss == -237770.88
        assert metrics.max_drawdown == -56721.59999999998
        assert metrics.max_drawdown_pct == -7.908428778116649
        assert metrics.max_drawdown_date == datetime(2022, 1, 25, 5, 0)
        assert metrics.win_rate == 52.57731958762887
        assert metrics.loss_rate == 47.42268041237113
        assert metrics.winning_trades == 204
        assert metrics.losing_trades == 184
        assert metrics.avg_pnl == 427.1654639175258
        assert metrics.avg_return_pct == 0.279639175257732
        assert metrics.avg_trade_bars == 2.4149484536082473
        assert metrics.avg_profit == 1977.9954901960782
        assert metrics.avg_profit_pct == 3.1687745098039217
        assert metrics.avg_winning_trade_bars == 2.465686274509804
        assert metrics.avg_loss == -1292.233043478261
        assert metrics.avg_loss_pct == -2.9235326086956523
        assert metrics.avg_losing_trade_bars == 2.358695652173913
        assert metrics.largest_win == 21069.63
        assert metrics.largest_win_pct == 14.49
        assert metrics.largest_win_bars == 3
        assert metrics.largest_loss == -11487.43
        assert metrics.largest_loss_pct == -6.49
        assert metrics.largest_loss_bars == 3
        assert metrics.max_wins == 7
        assert metrics.max_losses == 7
        assert metrics.sharpe == expected_sharpe
        assert metrics.sortino == expected_sortino
        assert metrics.profit_factor == 1.0759385033768167
        assert metrics.ulcer_index == 1.898347959437099
        assert metrics.upi == 0.01844528848501509
        assert metrics.equity_r2 == 0.8979045919638434
        assert metrics.std_error == 69646.36129687089
        assert metrics.total_fees == 0
        if bars_per_year is not None:
            assert metrics.calmar == 1.1557170701224246
            assert truncate(metrics.annual_return_pct, 6) == truncate(5.897743691129764, 6)
            assert metrics.annual_std_error == 1105601.710272446
            assert metrics.annual_volatility_pct == 21.36797425126505
        else:
            assert metrics.calmar is None
            assert metrics.annual_return_pct is None
            assert metrics.annual_std_error is None
            assert metrics.annual_volatility_pct is None

    def test_evaluate_when_portfolio_empty(self, trades_df, calc_bootstrap):
        mixin = EvaluateMixin()
        result = mixin.evaluate(pd.DataFrame(columns=['market_value', 'fees']), trades_df, calc_bootstrap, bootstrap_sample_size=10, bootstrap_samples=100, bars_per_year=None)
        assert result.metrics is not None
        for field in get_type_hints(EvalMetrics):
            if field in ('calmar', 'annual_return_pct', 'annual_std_error', 'annual_volatility_pct', 'max_drawdown_date'):
                assert getattr(result.metrics, field) is None
            else:
                assert getattr(result.metrics, field) == 0
        assert result.bootstrap is None

    def test_evaluate_when_single_market_value(self, trades_df, calc_bootstrap):
        mixin = EvaluateMixin()
        result = mixin.evaluate(pd.DataFrame([[1000, 0]], columns=['market_value', 'fees'], index=[pd.Timestamp('2023-04-12 00:00:00')]), trades_df, calc_bootstrap, bootstrap_sample_size=10, bootstrap_samples=100, bars_per_year=None)
        assert result.metrics is not None
        for field in get_type_hints(EvalMetrics):
            if field in ('calmar', 'annual_return_pct', 'annual_std_error', 'annual_volatility_pct', 'max_drawdown_date'):
                assert getattr(result.metrics, field) is None
            else:
                assert getattr(result.metrics, field) == 0
        assert result.bootstrap is None

    def test_evaluate_when_trades_empty(self, portfolio_df, calc_bootstrap):
        mixin = EvaluateMixin()
        result = mixin.evaluate(portfolio_df, pd.DataFrame(columns=['pnl', 'return_pct', 'bars']), calc_bootstrap, bootstrap_sample_size=10, bootstrap_samples=100, bars_per_year=None)
        metrics = result.metrics
        assert metrics is not None
        assert metrics.total_pnl == 0
        assert metrics.total_return_pct == 0
        assert metrics.total_profit == 0
        assert metrics.total_loss == 0
        assert metrics.win_rate == 0
        assert metrics.loss_rate == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.avg_pnl == 0
        assert metrics.avg_return_pct == 0
        assert metrics.avg_trade_bars == 0
        assert metrics.avg_profit == 0
        assert metrics.avg_profit_pct == 0
        assert metrics.avg_winning_trade_bars == 0
        assert metrics.avg_loss == 0
        assert metrics.avg_loss_pct == 0
        assert metrics.avg_losing_trade_bars == 0
        assert metrics.largest_win == 0
        assert metrics.largest_win_pct == 0
        assert metrics.largest_win_bars == 0
        assert metrics.largest_loss == 0
        assert metrics.largest_loss_pct == 0
        assert metrics.largest_loss_bars == 0
        assert metrics.max_wins == 0
        assert metrics.max_losses == 0
        assert metrics.total_fees == 0
        if calc_bootstrap:
            assert result.bootstrap is not None
            assert result.bootstrap.conf_intervals is not None
            assert result.bootstrap.drawdown_conf is not None
            assert result.bootstrap.profit_factor is not None
            assert result.bootstrap.sharpe is not None
            assert result.bootstrap.drawdown is not None
        else:
            assert result.bootstrap is None

