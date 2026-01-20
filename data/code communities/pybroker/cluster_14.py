# Cluster 14

def rebalance(ctxs: dict[str, ExecContext]):
    if start_of_month(ctxs):
        target = 1 / len(ctxs)
        set_target_shares(ctxs, {symbol: target for symbol in ctxs.keys()})

def start_of_month(ctxs: dict[str, ExecContext]) -> bool:
    dt = tuple(ctxs.values())[0].dt
    if dt.month != pyb.param('current_month'):
        pyb.param('current_month', dt.month)
        return True
    return False

def set_target_shares(ctxs: dict[str, ExecContext], targets: dict[str, float]):
    for symbol, target in targets.items():
        ctx = ctxs[symbol]
        target_shares = ctx.calc_target_shares(target)
        pos = ctx.long_pos()
        if pos is None:
            ctx.buy_shares = target_shares
        elif pos.shares < target_shares:
            ctx.buy_shares = target_shares - pos.shares
        elif pos.shares > target_shares:
            ctx.sell_shares = pos.shares - target_shares

def optimization(ctxs: dict[str, ExecContext]):
    lookback = pyb.param('lookback')
    if start_of_month(ctxs):
        Y = calculate_returns(ctxs, lookback)
        port = rp.Portfolio(returns=Y)
        port.assets_stats(method_mu='hist', method_cov='hist')
        w = port.optimization(model='Classic', rm='CVaR', obj='MinRisk', rf=0, l=0, hist=True)
        targets = {symbol: w.T[symbol].values[0] for symbol in ctxs.keys()}
        set_target_shares(ctxs, targets)

def calculate_returns(ctxs: dict[str, ExecContext], lookback: int):
    prices = {}
    for ctx in ctxs.values():
        prices[ctx.symbol] = ctx.adj_close[-lookback:]
    df = pd.DataFrame(prices)
    return df.pct_change().dropna()

