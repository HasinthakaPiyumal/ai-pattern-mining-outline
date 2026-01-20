# Cluster 28

def run_ml4t_demo():
    """Demonstration of the ML4T framework"""
    print('Initializing ML4T Framework...')
    config = ML4TConfig()
    ml4t = ML4TFramework(config)
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    ml4t.load_data(symbols, start_date, end_date)
    feature_data = ml4t.engineer_features(symbols)
    models_config = [{'name': 'ridge_model', 'type': 'ridge', 'params': {'alpha': 1.0}}, {'name': 'random_forest', 'type': 'random_forest', 'params': {'n_estimators': 100}}, {'name': 'gradient_boosting', 'type': 'gradient_boosting', 'params': {'n_estimators': 100}}]
    ml4t.train_models(feature_data, models_config)
    feature_columns = [col for col in feature_data.columns if col not in ['date', 'symbol', 'future_returns', 'open', 'high', 'low', 'close', 'volume']]
    strategies_config = [{'name': 'mean_reversion', 'type': 'mean_reversion', 'params': {'lookback_window': 20}}, {'name': 'momentum', 'type': 'momentum', 'params': {'lookback_window': 12}}, {'name': 'ml_ridge', 'type': 'ml_strategy', 'params': {'model_name': 'ridge_model', 'features': feature_columns[:10]}}, {'name': 'ml_rf', 'type': 'ml_strategy', 'params': {'model_name': 'random_forest', 'features': feature_columns[:10]}}]
    ml4t.create_strategies(strategies_config)
    ml4t.run_backtests()
    summary = ml4t.generate_report()
    return (ml4t, summary)

