# Cluster 28

def test_disable_logging(mock_logger):
    disable_logging()
    mock_logger.disable.assert_called_once()

def disable_logging():
    """Disables event logging."""
    StaticScope.instance().logger.disable()

