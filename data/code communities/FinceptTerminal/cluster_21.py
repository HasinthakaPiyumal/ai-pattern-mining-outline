# Cluster 21

def demo_pypfopt_analytics():
    """Demonstration of PyPortfolioOpt analytics engine"""
    print('=== PyPortfolioOpt Portfolio Analytics Demo ===\n')
    import yfinance as yf
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'PG', 'XOM', 'GLD']
    try:
        print('Downloading sample data...')
        prices = yf.download(assets, start='2018-01-01', end='2023-12-31')['Adj Close']
    except:
        print('Creating synthetic data...')
        np.random.seed(42)
        dates = pd.date_range('2018-01-01', '2023-12-31', freq='D')
        n_assets = len(assets)
        correlation_matrix = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
        np.fill_diagonal(correlation_matrix, 1.0)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        returns = pd.DataFrame(np.random.multivariate_normal(np.random.uniform(0.0005, 0.0015, n_assets), correlation_matrix * 0.0004, len(dates)), index=dates, columns=assets)
        prices = (1 + returns).cumprod() * 100
    config = create_sample_pypfopt_config()
    engine = PyPortfolioOptAnalyticsEngine(config)
    engine.load_data(prices)
    print(f'Data loaded: {len(prices)} periods, {len(prices.columns)} assets\n')
    print('Performing Max Sharpe optimization...')
    weights = engine.optimize_portfolio()
    ret, vol, sharpe = engine.portfolio_performance()

def create_sample_pypfopt_config() -> PyPortfolioOptConfig:
    """Create sample configuration for PyPortfolioOpt"""
    return PyPortfolioOptConfig(optimization_method='efficient_frontier', objective='max_sharpe', expected_returns_method='mean_historical_return', risk_model_method='sample_cov', risk_free_rate=0.02, weight_bounds=(0, 0.4), gamma=0.1, total_portfolio_value=100000)

