# Cluster 22

def accumulate_span_logs(span: Span) -> None:
    """Recursively accumulate logs from child spans to parent span.

        Args:
            span (Span): The span to accumulate logs from
        """
    for child_span in span.spans:
        accumulate_span_logs(child_span)
    for child_span in span.spans:
        if child_span.num_debug_logs is not None:
            span.num_debug_logs = (span.num_debug_logs or 0) + child_span.num_debug_logs
        if child_span.num_info_logs is not None:
            span.num_info_logs = (span.num_info_logs or 0) + child_span.num_info_logs
        if child_span.num_warning_logs is not None:
            span.num_warning_logs = (span.num_warning_logs or 0) + child_span.num_warning_logs
        if child_span.num_error_logs is not None:
            span.num_error_logs = (span.num_error_logs or 0) + child_span.num_error_logs
        if child_span.num_critical_logs is not None:
            span.num_critical_logs = (span.num_critical_logs or 0) + child_span.num_critical_logs

def accumulate_logs(span_data: dict) -> dict:
    """Accumulate the number of logs from child spans to the parent span.

    Args:
        span_data (dict): The span data

    Returns:
        dict: The span data with the accumulated logs
    """

    def accumulate_span_logs(span: Span) -> None:
        """Recursively accumulate logs from child spans to parent span.

        Args:
            span (Span): The span to accumulate logs from
        """
        for child_span in span.spans:
            accumulate_span_logs(child_span)
        for child_span in span.spans:
            if child_span.num_debug_logs is not None:
                span.num_debug_logs = (span.num_debug_logs or 0) + child_span.num_debug_logs
            if child_span.num_info_logs is not None:
                span.num_info_logs = (span.num_info_logs or 0) + child_span.num_info_logs
            if child_span.num_warning_logs is not None:
                span.num_warning_logs = (span.num_warning_logs or 0) + child_span.num_warning_logs
            if child_span.num_error_logs is not None:
                span.num_error_logs = (span.num_error_logs or 0) + child_span.num_error_logs
            if child_span.num_critical_logs is not None:
                span.num_critical_logs = (span.num_critical_logs or 0) + child_span.num_critical_logs
    for _, spans in span_data.items():
        for span in spans:
            accumulate_span_logs(span)
    return span_data

