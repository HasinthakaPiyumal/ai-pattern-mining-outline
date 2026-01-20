# Cluster 27

def test_enable_logging(mock_logger):
    enable_logging()
    mock_logger.enable.assert_called_once()

def enable_logging():
    """Enables event logging."""
    StaticScope.instance().logger.enable()

