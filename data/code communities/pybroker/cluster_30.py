# Cluster 30

def test_disable_progress_bar(mock_logger):
    disable_progress_bar()
    mock_logger.disable_progress_bar.assert_called_once()

def disable_progress_bar():
    """Disables logging a progress bar."""
    StaticScope.instance().logger.disable_progress_bar()

