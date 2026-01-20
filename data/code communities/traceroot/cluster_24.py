# Cluster 24

def accumulate_span_languages_recursively(span: Span) -> set[str]:
    """Recursively accumulate telemetry SDK languages from a span and
        all its children. Also updates the span's own telemetry_sdk_language
        set to include children's languages.

        Args:
            span: The span to accumulate languages from (modified in-place).

        Returns:
            set[str]: Set of telemetry SDK languages from this span and all
            children.
        """
    languages = set()
    if span.telemetry_sdk_language is not None:
        languages.add(span.telemetry_sdk_language)
    for child_span in span.spans:
        child_languages = accumulate_span_languages_recursively(child_span)
        languages.update(child_languages)
    return languages

def accumulate_telemetry_languages_to_traces(traces: list[Trace]) -> None:
    """Accumulate telemetry SDK languages from all child spans recursively
    to traces.

    Args:
        traces: List of traces to accumulate telemetry SDK languages
            (modified in-place).
    """

    def accumulate_span_languages_recursively(span: Span) -> set[str]:
        """Recursively accumulate telemetry SDK languages from a span and
        all its children. Also updates the span's own telemetry_sdk_language
        set to include children's languages.

        Args:
            span: The span to accumulate languages from (modified in-place).

        Returns:
            set[str]: Set of telemetry SDK languages from this span and all
            children.
        """
        languages = set()
        if span.telemetry_sdk_language is not None:
            languages.add(span.telemetry_sdk_language)
        for child_span in span.spans:
            child_languages = accumulate_span_languages_recursively(child_span)
            languages.update(child_languages)
        return languages
    for trace in traces:
        if len(trace.spans) > 0:
            trace.start_time = trace.spans[0].start_time
            trace.end_time = trace.spans[0].end_time
            for span in trace.spans:
                span_languages = accumulate_span_languages_recursively(span)
                trace.telemetry_sdk_language.update(span_languages)

