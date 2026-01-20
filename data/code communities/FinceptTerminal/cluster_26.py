# Cluster 26

class SystematicRiskAnalysis:
    """Systematic vs. Nonsystematic risk analysis"""

    @staticmethod
    def decompose_risk(asset_returns: np.ndarray, market_returns: np.ndarray) -> Dict:
        """Decompose total risk into systematic and nonsystematic components"""
        beta = StatisticalCalculations.calculate_beta(asset_returns, market_returns)
        total_variance = StatisticalCalculations.calculate_variance(asset_returns)
        market_variance = StatisticalCalculations.calculate_variance(market_returns)
        systematic_variance = beta ** 2 * market_variance
        nonsystematic_variance = total_variance - systematic_variance
        correlation = StatisticalCalculations.calculate_correlation(asset_returns, market_returns)
        r_squared = correlation ** 2
        return {'beta': beta, 'total_variance': total_variance, 'total_std': np.sqrt(total_variance), 'systematic_variance': systematic_variance, 'systematic_std': np.sqrt(systematic_variance), 'nonsystematic_variance': max(0, nonsystematic_variance), 'nonsystematic_std': np.sqrt(max(0, nonsystematic_variance)), 'r_squared': r_squared, 'systematic_risk_percentage': systematic_variance / total_variance * 100, 'nonsystematic_risk_percentage': max(0, nonsystematic_variance) / total_variance * 100}

    @staticmethod
    def portfolio_beta(individual_betas: np.ndarray, weights: np.ndarray) -> float:
        """Calculate portfolio beta as weighted average of individual betas"""
        if not validate_weights(weights):
            raise ValueError(ERROR_MESSAGES['invalid_weights'])
        return np.dot(weights, individual_betas)

def validate_weights(weights: Union[List, np.ndarray], tolerance: float=1e-06) -> bool:
    """Validate portfolio weights sum to 1.0"""
    return abs(np.sum(weights) - 1.0) <= tolerance

