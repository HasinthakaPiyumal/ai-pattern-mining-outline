# Cluster 27

class EfficientFrontierAnalysis:
    """Efficient frontier construction and analysis"""

    def __init__(self, expected_returns: np.ndarray, cov_matrix: np.ndarray, parameters: PortfolioParameters=DEFAULT_PORTFOLIO_PARAMS):
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.parameters = parameters
        self.n_assets = len(expected_returns)
        if not validate_covariance_matrix(cov_matrix):
            raise ValueError(ERROR_MESSAGES['singular_matrix'])

    def generate_frontier(self) -> Dict:
        """Generate complete efficient frontier"""
        min_var_result = PortfolioMath.find_minimum_variance_portfolio(self.cov_matrix, {'min_weight': self.parameters.min_weight, 'max_weight': self.parameters.max_weight})
        frontier_result = OptimizationEngine.efficient_frontier(self.expected_returns, self.cov_matrix, self.parameters.num_frontier_points, {'min_weight': self.parameters.min_weight, 'max_weight': self.parameters.max_weight})
        max_sharpe_result = OptimizationEngine.maximum_sharpe_portfolio(self.expected_returns, self.cov_matrix, self.parameters.risk_free_rate, {'min_weight': self.parameters.min_weight, 'max_weight': self.parameters.max_weight})
        return {'frontier_returns': frontier_result['returns'], 'frontier_stds': frontier_result['stds'], 'frontier_weights': frontier_result['weights'], 'frontier_sharpe_ratios': frontier_result['sharpe_ratios'], 'min_variance_portfolio': min_var_result, 'max_sharpe_portfolio': max_sharpe_result, 'capital_market_line': self._calculate_cml(max_sharpe_result)}

    def _calculate_cml(self, max_sharpe_portfolio: Dict) -> Dict:
        """Calculate Capital Market Line parameters"""
        return CapitalAllocationLine.calculate_cal(max_sharpe_portfolio['expected_return'], max_sharpe_portfolio['std'], self.parameters.risk_free_rate)

    def portfolio_on_frontier(self, target_return: float) -> Dict:
        """Find specific portfolio on efficient frontier"""
        from scipy import optimize

        def objective(weights):
            return PortfolioMath.calculate_portfolio_variance(weights, self.cov_matrix)
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, {'type': 'eq', 'fun': lambda w: PortfolioMath.calculate_portfolio_return(w, self.expected_returns) - target_return}]
        bounds = tuple(((self.parameters.min_weight, self.parameters.max_weight) for _ in range(self.n_assets)))
        x0 = np.ones(self.n_assets) / self.n_assets
        result = optimize.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        if result.success:
            weights = result.x
            portfolio_std = PortfolioMath.calculate_portfolio_std(weights, self.cov_matrix)
            sharpe_ratio = (target_return - self.parameters.risk_free_rate) / portfolio_std
            return {'weights': weights, 'expected_return': target_return, 'standard_deviation': portfolio_std, 'sharpe_ratio': sharpe_ratio, 'on_efficient_frontier': True}
        else:
            return {'on_efficient_frontier': False, 'error': 'Optimization failed'}

def validate_covariance_matrix(cov_matrix: np.ndarray) -> bool:
    """Validate covariance matrix properties"""
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        return False
    if not np.allclose(cov_matrix, cov_matrix.T):
        return False
    eigenvals = np.linalg.eigvals(cov_matrix)
    return np.all(eigenvals >= -1e-08)

