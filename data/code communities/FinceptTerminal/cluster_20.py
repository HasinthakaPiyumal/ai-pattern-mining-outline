# Cluster 20

def portfolio_performance(self, weights: pd.Series=None) -> Tuple[float, float, float]:
    """
    Calculate portfolio performance metrics

    Parameters:
    -----------
    weights : pd.Series
        Portfolio weights (uses optimized weights if None)

    Returns:
    --------
    Tuple of (expected_return, volatility, sharpe_ratio)
    """
    weights = weights if weights is not None else pd.Series(self.weights)
    if self.expected_returns is None:
        self.calculate_expected_returns()
    if self.risk_model is None:
        self.calculate_risk_model()
    ret, vol, sharpe = portfolio_performance(weights, self.expected_returns, self.risk_model, risk_free_rate=self.config.risk_free_rate)
    self.performance_metrics = {'expected_return': ret, 'volatility': vol, 'sharpe_ratio': sharpe}
    return (ret, vol, sharpe)

