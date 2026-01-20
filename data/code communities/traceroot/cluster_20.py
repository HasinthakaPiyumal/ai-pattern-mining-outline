# Cluster 20

def get_token_tracker() -> TokenTracker:
    """Get the global TokenTracker instance."""
    global _token_tracker
    if _token_tracker is None:
        _token_tracker = TokenTracker()
    return _token_tracker

