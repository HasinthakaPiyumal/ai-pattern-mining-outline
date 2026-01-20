# Cluster 22

def demo_portfolio_analytics():
    """Demonstration of the portfolio analytics engine"""
    print('Loading sample data...')
    prices = load_sp500_dataset()
    config = create_sample_config()
    engine = PortfolioAnalyticsEngine(config)
    engine.load_data(prices)
    print('\nOptimizing portfolio...')
    results = engine.optimize_portfolio()
    weights = pd.Series(engine.model.weights_, index=engine.returns.columns)
    print('\nTop 10 Positions:')
    print(weights.sort_values(key=abs, ascending=False)[:10].to_string())
    print('\nPerforming hyperparameter tuning...')
    tuning_results = engine.hyperparameter_tuning()
    print(f'Best parameters: {tuning_results['best_params']}')
    print(f'Best score: {tuning_results['best_score']:.4f}')
    print('\nRunning backtest...')
    backtest_df = engine.backtest_strategy()
    print(f'Backtest performance metrics:')
    for metric, value in engine.backtest_results['metrics'].items():
        if isinstance(value, (int, float)):
            print(f'{metric}: {value:.4f}')
    print('\nRisk attribution analysis...')
    risk_attrib = engine.risk_attribution()
    print('Top 5 risk contributors:')
    print(risk_attrib['asset_attribution'][['Asset', 'Weight_Pct', 'Risk_Contribution_Pct']].head().to_string(index=False))
    print('\nGenerating comprehensive report...')
    report_file = engine.save_report()
    weights_file = engine.export_weights_to_csv()
    print(f'\nDemo completed successfully!')
    print(f'Files generated: {report_file}, {weights_file}')
    return engine

def create_sample_config() -> PortfolioConfig:
    """Create a sample configuration for testing"""
    return PortfolioConfig(optimization_method='mean_risk', objective_function='maximize_ratio', risk_measure='cvar', covariance_estimator='ledoit_wolf', mu_estimator='shrunk', max_weight_single_asset=0.15, l2_coef=0.01, cv_method='walk_forward', cv_folds=5, use_uncertainty_set=True)

