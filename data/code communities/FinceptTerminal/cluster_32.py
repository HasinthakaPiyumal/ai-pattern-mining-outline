# Cluster 32

def get_portfolio_config_path():
    """Get portfolio configuration file path in .fincept directory"""
    config_dir = Path.home() / '.fincept'
    config_dir.mkdir(exist_ok=True)
    return config_dir / 'portfolio_settings.json'

