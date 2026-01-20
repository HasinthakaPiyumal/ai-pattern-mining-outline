# Cluster 25

class ValueAddedMeasurement:
    """Measurement of value added by active management"""

    @staticmethod
    def calculate_value_added(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray, portfolio_weights: Optional[np.ndarray]=None) -> Dict:
        """Calculate value added by active management"""
        if not validate_returns(portfolio_returns) or not validate_returns(benchmark_returns):
            raise ValueError(ERROR_MESSAGES['insufficient_data'])
        active_returns = np.array(portfolio_returns) - np.array(benchmark_returns)
        mean_active_return = np.mean(active_returns)
        tracking_error = np.std(active_returns, ddof=1)
        annualized_active_return = mean_active_return * MathConstants.TRADING_DAYS_YEAR
        annualized_tracking_error = tracking_error * MathConstants.SQRT_TRADING_DAYS
        information_ratio = mean_active_return / tracking_error if tracking_error > 0 else 0
        annualized_ir = information_ratio * MathConstants.SQRT_TRADING_DAYS
        hit_rate = np.sum(active_returns > 0) / len(active_returns)
        value_decomposition = ValueAddedMeasurement._decompose_value_added(portfolio_returns, benchmark_returns, portfolio_weights)
        return {'active_return_daily': mean_active_return, 'active_return_annualized': annualized_active_return, 'tracking_error_daily': tracking_error, 'tracking_error_annualized': annualized_tracking_error, 'information_ratio_daily': information_ratio, 'information_ratio_annualized': annualized_ir, 'hit_rate': hit_rate, 'value_added_decomposition': value_decomposition, 'statistical_significance': ValueAddedMeasurement._test_statistical_significance(active_returns)}

    @staticmethod
    def _decompose_value_added(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray, portfolio_weights: Optional[np.ndarray]) -> Dict:
        """Decompose value added into sources"""
        total_active_return = np.mean(portfolio_returns) - np.mean(benchmark_returns)
        estimated_allocation_effect = total_active_return * 0.3
        estimated_selection_effect = total_active_return * 0.7
        return {'total_active_return': total_active_return * MathConstants.TRADING_DAYS_YEAR, 'estimated_allocation_effect': estimated_allocation_effect * MathConstants.TRADING_DAYS_YEAR, 'estimated_selection_effect': estimated_selection_effect * MathConstants.TRADING_DAYS_YEAR, 'interaction_effect': 0.0, 'note': 'Decomposition requires detailed holdings data for precision'}

    @staticmethod
    def _test_statistical_significance(active_returns: np.ndarray) -> Dict:
        """Test statistical significance of active returns"""
        mean_active = np.mean(active_returns)
        std_active = np.std(active_returns, ddof=1)
        n_periods = len(active_returns)
        t_statistic = mean_active / (std_active / np.sqrt(n_periods)) if std_active > 0 else 0
        critical_90 = 1.645
        critical_95 = 1.96
        critical_99 = 2.576
        significance_level = 'Not significant'
        if abs(t_statistic) > critical_99:
            significance_level = '99% significant'
        elif abs(t_statistic) > critical_95:
            significance_level = '95% significant'
        elif abs(t_statistic) > critical_90:
            significance_level = '90% significant'
        return {'t_statistic': t_statistic, 'significance_level': significance_level, 'is_significant_95': abs(t_statistic) > critical_95, 'sample_size': n_periods, 'standard_error': std_active / np.sqrt(n_periods)}

def validate_returns(returns: Union[List, np.ndarray]) -> bool:
    """Validate return data"""
    returns_array = np.array(returns)
    return not np.any(np.isnan(returns_array)) and len(returns_array) >= 2

