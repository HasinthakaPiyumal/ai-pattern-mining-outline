# Cluster 27

def get_traces_and_logs_tracker() -> TracesAndLogsTracker:
    """Get the global TracesAndLogsTracker instance."""
    global _traces_and_logs_tracker
    if _traces_and_logs_tracker is None:
        _traces_and_logs_tracker = TracesAndLogsTracker()
    return _traces_and_logs_tracker

