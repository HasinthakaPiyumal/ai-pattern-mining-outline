# Cluster 7

def set_screencast_running(running: bool=True) -> None:
    """Set the active_screencast_running flag.

    Args:
        running: Whether the screencast is running

    Returns:
        None
    """
    global active_screencast_running
    active_screencast_running = running

def set_url_and_task(url: str, task: str):
    """Sets the current URL and task and broadcasts it to all connected clients."""
    global current_url, current_task
    current_url = url
    current_task = task

