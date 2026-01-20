# Cluster 61

def get_stock_data(symbol: str, period: str='1d', interval: str='1m') -> Dict[str, Any]:
    """Convenience function for getting stock data"""
    dsm = get_data_source_manager()
    if dsm:
        return dsm.get_stock_data(symbol, period, interval)
    else:
        logger.error('DataSourceManager not available')
        return {'success': False, 'error': 'DataSourceManager not available'}

def get_data_source_manager(app=None):
    """Get global data source manager instance"""
    global data_source_manager
    try:
        if data_source_manager is None and app:
            logger.info('Creating global DataSourceManager instance')
            data_source_manager = DataSourceManager(app)
            data_source_manager._start_time = time.time()
        return data_source_manager
    except Exception as e:
        logger.error('Failed to create DataSourceManager instance', context={'error': str(e)}, exc_info=True)
        return None

def get_forex_data(pair: str, period: str='1d') -> Dict[str, Any]:
    """Convenience function for getting forex data"""
    dsm = get_data_source_manager()
    if dsm:
        return dsm.get_forex_data(pair, period)
    else:
        logger.error('DataSourceManager not available')
        return {'success': False, 'error': 'DataSourceManager not available'}

def get_news_data(category: str='financial', limit: int=20) -> Dict[str, Any]:
    """Convenience function for getting news data"""
    dsm = get_data_source_manager()
    if dsm:
        return dsm.get_news_data(category, limit)
    else:
        logger.error('DataSourceManager not available')
        return {'success': False, 'error': 'DataSourceManager not available'}

