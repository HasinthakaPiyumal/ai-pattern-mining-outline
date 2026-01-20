# Cluster 24

def validate_configuration():
    """Validate configuration settings on module load"""
    if getcontext().prec < 20:
        raise ConfigurationError('Decimal precision too low for financial calculations')
    required_configs = ['MARKET_CONVENTIONS', 'PRECISION_CONFIG', 'DEFAULT_PARAMS']
    for config in required_configs:
        if config not in globals():
            raise ConfigurationError(f'Missing required configuration: {config}')

