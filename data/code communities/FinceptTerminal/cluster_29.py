# Cluster 29

def setup_default_providers(alpha_vantage_key: Optional[str]=None):
    """Setup default data providers"""
    yahoo_provider = YahooFinanceProvider()
    data_factory.register_provider('yahoo', yahoo_provider, is_primary=True)
    if alpha_vantage_key:
        av_provider = AlphaVantageProvider(alpha_vantage_key)
        data_factory.register_provider('alphavantage', av_provider)
    manual_provider = ManualDataProvider()
    data_factory.register_provider('manual', manual_provider)

