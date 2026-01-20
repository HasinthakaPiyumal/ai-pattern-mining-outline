# Cluster 29

def test_enable_progress_bar(mock_logger):
    enable_progress_bar()
    mock_logger.enable_progress_bar.assert_called_once()

def enable_progress_bar():
    """Enables logging a progress bar."""
    StaticScope.instance().logger.enable_progress_bar()

