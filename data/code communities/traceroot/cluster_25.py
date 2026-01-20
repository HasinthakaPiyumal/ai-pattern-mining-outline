# Cluster 25

def sort_spans_recursively(spans: list[Span]) -> None:
    """
    Sort spans by start_time recursively, including all child spans.

    Args:
        spans: List of spans to sort (modified in-place)
    """
    spans.sort(key=lambda span: span.start_time, reverse=False)
    for span in spans:
        if span.spans:
            sort_spans_recursively(span.spans)

def accumulate_span_logs_recursively(span: Span) -> tuple[int, int, int, int, int]:
    """Recursively accumulate logs from a span and all its children.
        Also updates the span's own log counts to include children's logs.

        Args:
            span: The span to accumulate logs (modified in-place).

        Returns:
            tuple: (debug_logs, info_logs, warning_logs,
                error_logs, critical_logs)
        """
    debug_logs = span.num_debug_logs or 0
    info_logs = span.num_info_logs or 0
    warning_logs = span.num_warning_logs or 0
    error_logs = span.num_error_logs or 0
    critical_logs = span.num_critical_logs or 0
    for child_span in span.spans:
        child_debug, child_info, child_warning, child_error, child_critical = accumulate_span_logs_recursively(child_span)
        debug_logs += child_debug
        info_logs += child_info
        warning_logs += child_warning
        error_logs += child_error
        critical_logs += child_critical
    span.num_debug_logs = debug_logs
    span.num_info_logs = info_logs
    span.num_warning_logs = warning_logs
    span.num_error_logs = error_logs
    span.num_critical_logs = critical_logs
    return (debug_logs, info_logs, warning_logs, error_logs, critical_logs)

def accumulate_num_logs_to_traces(traces: list[Trace]) -> None:
    """Accumulate log counts from all child spans recursively
    to traces.

    Args:
        traces: List of traces to accumulate logs
            (modified in-place).
    """

    def accumulate_span_logs_recursively(span: Span) -> tuple[int, int, int, int, int]:
        """Recursively accumulate logs from a span and all its children.
        Also updates the span's own log counts to include children's logs.

        Args:
            span: The span to accumulate logs (modified in-place).

        Returns:
            tuple: (debug_logs, info_logs, warning_logs,
                error_logs, critical_logs)
        """
        debug_logs = span.num_debug_logs or 0
        info_logs = span.num_info_logs or 0
        warning_logs = span.num_warning_logs or 0
        error_logs = span.num_error_logs or 0
        critical_logs = span.num_critical_logs or 0
        for child_span in span.spans:
            child_debug, child_info, child_warning, child_error, child_critical = accumulate_span_logs_recursively(child_span)
            debug_logs += child_debug
            info_logs += child_info
            warning_logs += child_warning
            error_logs += child_error
            critical_logs += child_critical
        span.num_debug_logs = debug_logs
        span.num_info_logs = info_logs
        span.num_warning_logs = warning_logs
        span.num_error_logs = error_logs
        span.num_critical_logs = critical_logs
        return (debug_logs, info_logs, warning_logs, error_logs, critical_logs)
    for trace in traces:
        if len(trace.spans) > 0:
            trace.start_time = trace.spans[0].start_time
            trace.end_time = trace.spans[0].end_time
            for span in trace.spans:
                debug_logs, info_logs, warning_logs, error_logs, critical_logs = accumulate_span_logs_recursively(span)
                trace.num_debug_logs = (trace.num_debug_logs or 0) + debug_logs
                trace.num_info_logs = (trace.num_info_logs or 0) + info_logs
                trace.num_warning_logs = (trace.num_warning_logs or 0) + warning_logs
                trace.num_error_logs = (trace.num_error_logs or 0) + error_logs
                trace.num_critical_logs = (trace.num_critical_logs or 0) + critical_logs

def construct_traces(service_names: list[str | None], service_environments: list[str | None], trace_ids: list[str], start_times: list[datetime], durations: list[float]) -> list[Trace]:
    """Construct traces from trace IDs, start times, durations, and end times.

    Args:
        service_names (list[str]): List of service names
        service_environments (list[str]): List of service environments
        trace_ids (list[str]): List of trace IDs
        start_times (list[datetime]): List of start times
        durations (list[float]): List of durations

    Returns:
        list[Trace]: List of traces
    """
    traces: list[Trace] = []
    end_times: list[float] = [start_time + duration for start_time, duration in zip(start_times, durations)]
    if durations and len(durations) > 100:
        durations_array = np.array(durations)
        p50 = np.percentile(durations_array, 50)
        p90 = np.percentile(durations_array, 90)
        p95 = np.percentile(durations_array, 95)
    else:
        p50 = p90 = p95 = 0
    for i, trace_id in enumerate(trace_ids):
        start_time = start_times[i]
        duration = durations[i]
        end_time = end_times[i]
        if len(durations) > 100:
            if duration <= p50:
                percentile = Percentile.P50
            elif duration <= p90:
                percentile = Percentile.P90
            elif duration <= p95:
                percentile = Percentile.P95
            else:
                percentile = Percentile.P99
        else:
            percentile = Percentile.P50
        service_name = service_names[i]
        service_environment = service_environments[i]
        traces.append(Trace(id=trace_id, start_time=start_time, end_time=end_time, duration=duration, percentile=percentile, service_name=service_name, service_environment=service_environment, num_debug_logs=0, num_info_logs=0, num_warning_logs=0, num_error_logs=0, num_critical_logs=0, telemetry_sdk_language=set()))
    return traces

def ensure_utc_datetime(dt: datetime) -> datetime:
    """
    Ensure datetime object is timezone-aware and in UTC.

    Args:
        dt: datetime object (timezone-aware or naive)

    Returns:
        datetime object in UTC timezone
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    else:
        return dt.astimezone(timezone.utc)

