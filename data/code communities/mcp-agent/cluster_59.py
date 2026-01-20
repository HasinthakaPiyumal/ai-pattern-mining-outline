# Cluster 59

class ProgressListener(LifecycleAwareListener):
    """
    Listens for all events pre-filtering and converts them to progress events
    for display. By inheriting directly from LifecycleAwareListener instead of
    FilteredListener, we get events before any filtering occurs.
    """

    def __init__(self, display=None, token_counter=None):
        """Initialize the progress listener.
        Args:
            display: Optional display handler. If None, the shared progress_display will be used if available.
        """
        self.display = display
        if self.display is None:
            from mcp_agent.logging.progress_display import create_progress_display
            self.display = create_progress_display(token_counter=token_counter)

    async def start(self):
        """Start the progress display."""
        if self.display:
            self.display.start()

    async def stop(self):
        """Stop the progress display."""
        if self.display:
            self.display.stop()

    async def handle_event(self, event: Event):
        """Process an incoming event and display progress if relevant."""
        if self.display and event.data:
            progress_event = convert_log_event(event)
            if progress_event:
                self.display.update(progress_event)

def create_progress_display(token_counter=None) -> RichProgressDisplay:
    """Create a new progress display instance.

    Args:
        token_counter: Optional TokenCounter instance for token tracking
    """
    return RichProgressDisplay(console, token_counter)

