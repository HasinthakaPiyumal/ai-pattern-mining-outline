# Cluster 23

class TimingTracker:
    """Tracks timing metrics for operations"""

    def __init__(self):
        self.metrics: List[TimingMetrics] = []
        self._current_operation: Optional[TimingMetrics] = None
        self._operation_stack: List[TimingMetrics] = []

    @contextmanager
    def track(self, operation_name: str, **metadata):
        """Context manager for tracking operation timing"""
        metrics = TimingMetrics(operation_name=operation_name, start_time=time.time(), metadata=metadata)
        if self._current_operation:
            self._current_operation.children.append(metrics)
        else:
            self.metrics.append(metrics)
        self._operation_stack.append(metrics)
        self._current_operation = metrics
        try:
            yield metrics
        finally:
            metrics.end_time = time.time()
            self._operation_stack.pop()
            self._current_operation = self._operation_stack[-1] if self._operation_stack else None

    @asynccontextmanager
    async def track_async(self, operation_name: str, **metadata):
        """Async context manager for tracking operation timing"""
        metrics = TimingMetrics(operation_name=operation_name, start_time=time.time(), metadata=metadata)
        if self._current_operation:
            self._current_operation.children.append(metrics)
        else:
            self.metrics.append(metrics)
        self._operation_stack.append(metrics)
        self._current_operation = metrics
        try:
            yield metrics
        finally:
            metrics.end_time = time.time()
            self._operation_stack.pop()
            self._current_operation = self._operation_stack[-1] if self._operation_stack else None

    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get all metrics in dictionary format"""
        return [metric.to_dict() for metric in self.metrics]

    def clear(self):
        """Clear all metrics"""
        self.metrics.clear()
        self._current_operation = None
        self._operation_stack.clear()

