# Cluster 7

def test_convert_span_no_logs_no_children():
    """Test converting span with no logs or children."""
    span = Span(id='span123', parent_id=None, name='test.function', start_time=1672574400.0, end_time=1672574401.0, duration=1.0)
    logs_map = {}
    result = convert_span_to_span_node(span, logs_map)
    assert result.span_id == 'span123'
    assert result.func_full_name == 'test.function'
    assert result.span_latency == 1.0
    assert result.span_utc_start_time == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert result.span_utc_end_time == datetime(2023, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
    assert result.logs == []
    assert result.children_spans == []

def convert_span_to_span_node(span: Span, logs_map: dict[str, list[LogEntry]]) -> SpanNode:
    """Convert Span to SpanNode recursively."""
    span_logs: list[LogNode] = []
    if span.id in logs_map:
        for log_entry in logs_map[span.id]:
            log_node = convert_log_entry_to_log_node(log_entry)
            span_logs.append(log_node)
    span_logs.sort(key=lambda log: log.log_utc_timestamp)
    children_spans: list[SpanNode] = []
    for child_span in span.spans:
        child_span_node = convert_span_to_span_node(child_span, logs_map)
        children_spans.append(child_span_node)
    children_spans.sort(key=lambda child: child.span_utc_start_time)
    return SpanNode(span_id=span.id, func_full_name=span.name, span_latency=span.duration, span_utc_start_time=datetime.fromtimestamp(span.start_time, tz=timezone.utc), span_utc_end_time=datetime.fromtimestamp(span.end_time, tz=timezone.utc), logs=span_logs, children_spans=children_spans)

def test_convert_span_with_children():
    """Test converting span with child spans."""
    child_span = Span(id='child123', parent_id='parent123', name='child.function', start_time=1672574400.5, end_time=1672574400.8, duration=0.3)
    parent_span = Span(id='parent123', parent_id=None, name='parent.function', start_time=1672574400.0, end_time=1672574401.0, duration=1.0, spans=[child_span])
    logs_map = {}
    result = convert_span_to_span_node(parent_span, logs_map)
    assert len(result.children_spans) == 1
    assert result.children_spans[0].span_id == 'child123'
    assert result.children_spans[0].func_full_name == 'child.function'

def test_convert_span_sorts_logs_by_timestamp():
    """Test that logs are sorted by timestamp."""
    span = Span(id='span123', parent_id=None, name='test.function', start_time=1672574400.0, end_time=1672574401.0, duration=1.0)
    log_entry1 = LogEntry(time=1672574400.8, level='INFO', message='Second log', function_name='test_func', file_name='test.py', line_number=20)
    log_entry2 = LogEntry(time=1672574400.2, level='DEBUG', message='First log', function_name='test_func', file_name='test.py', line_number=10)
    logs_map = {'span123': [log_entry1, log_entry2]}
    result = convert_span_to_span_node(span, logs_map)
    assert len(result.logs) == 2
    assert result.logs[0].log_message == 'First log'
    assert result.logs[1].log_message == 'Second log'

def test_convert_span_sorts_children_by_start_time():
    """Test that child spans are sorted by start time."""
    child_span1 = Span(id='child1', parent_id='parent123', name='child1.function', start_time=1672574400.8, end_time=1672574400.9, duration=0.1)
    child_span2 = Span(id='child2', parent_id='parent123', name='child2.function', start_time=1672574400.2, end_time=1672574400.4, duration=0.2)
    parent_span = Span(id='parent123', parent_id=None, name='parent.function', start_time=1672574400.0, end_time=1672574401.0, duration=1.0, spans=[child_span1, child_span2])
    logs_map = {}
    result = convert_span_to_span_node(parent_span, logs_map)
    assert len(result.children_spans) == 2
    assert result.children_spans[0].span_id == 'child2'
    assert result.children_spans[1].span_id == 'child1'

def test_build_heterogeneous_tree_simple():
    """Test building a simple heterogeneous tree."""
    span = Span(id='root', parent_id=None, name='root.function', start_time=1672574400.0, end_time=1672574401.0, duration=1.0)
    log_entry = LogEntry(time=1672574400.5, level='INFO', message='Root log', function_name='root_func', file_name='root.py', line_number=5)
    trace_logs = [{'root': [log_entry]}]
    result = build_heterogeneous_tree(span, trace_logs)
    assert result.span_id == 'root'
    assert result.func_full_name == 'root.function'
    assert len(result.logs) == 1
    assert result.logs[0].log_message == 'Root log'
    assert result.children_spans == []

def build_heterogeneous_tree(span: Span, trace_logs: list[dict[str, list[LogEntry]]]) -> SpanNode:
    """Build a heterogeneous tree from a trace and trace logs.
    """
    logs_map = create_logs_map(trace_logs)
    span_node = convert_span_to_span_node(span, logs_map)
    return span_node

def test_build_heterogeneous_tree_complex():
    """Test building a complex heterogeneous tree
    with nested spans and multiple logs."""
    child_span = Span(id='child', parent_id='parent', name='child.function', start_time=1672574400.3, end_time=1672574400.7, duration=0.4)
    parent_span = Span(id='parent', parent_id=None, name='parent.function', start_time=1672574400.0, end_time=1672574401.0, duration=1.0, spans=[child_span])
    parent_log = LogEntry(time=1672574400.1, level='INFO', message='Parent log', function_name='parent_func', file_name='parent.py', line_number=10)
    child_log = LogEntry(time=1672574400.5, level='DEBUG', message='Child log', function_name='child_func', file_name='child.py', line_number=15)
    trace_logs = [{'parent': [parent_log]}, {'child': [child_log]}]
    result = build_heterogeneous_tree(parent_span, trace_logs)
    assert result.span_id == 'parent'
    assert result.func_full_name == 'parent.function'
    assert len(result.logs) == 1
    assert result.logs[0].log_message == 'Parent log'
    assert len(result.children_spans) == 1
    child_result = result.children_spans[0]
    assert child_result.span_id == 'child'
    assert child_result.func_full_name == 'child.function'
    assert len(child_result.logs) == 1
    assert child_result.logs[0].log_message == 'Child log'

def test_build_heterogeneous_tree_empty_logs():
    """Test building tree with empty trace logs."""
    span = Span(id='span123', parent_id=None, name='test.function', start_time=1672574400.0, end_time=1672574401.0, duration=1.0)
    trace_logs = []
    result = build_heterogeneous_tree(span, trace_logs)
    assert result.span_id == 'span123'
    assert result.func_full_name == 'test.function'
    assert result.logs == []
    assert result.children_spans == []

class JaegerTraceClient(TraceClient):
    """Client for querying traces from Jaeger."""

    def __init__(self, jaeger_url: str | None=None):
        """Initialize the Jaeger trace client.

        Args:
            jaeger_url (str | None): Jaeger base URL. If None,
                uses JAEGER_URL env var or defaults to localhost.
        """
        if jaeger_url is None:
            jaeger_url = os.getenv('JAEGER_URL', 'http://localhost:16686')
        api_url = f'{jaeger_url}/api'
        self.traces_url = f'{api_url}/traces'
        self.services_url = f'{api_url}/services'

    async def get_trace_by_id(self, trace_id: str, categories: list[str] | None=None, values: list[str] | None=None, operations: list[str] | None=None) -> Trace | None:
        """Get a single trace by ID from Jaeger.

        Args:
            trace_id: The trace ID to fetch
            categories: Not used for Jaeger (kept for interface consistency)
            values: Not used for Jaeger (kept for interface consistency)
            operations: Not used for Jaeger (kept for interface consistency)

        Returns:
            Trace object if found, None otherwise
        """
        try:
            url = f'{self.traces_url}/{trace_id}'
            response = await self._make_request(url)
            if response and 'data' in response:
                traces_data = response['data']
                if traces_data and len(traces_data) > 0:
                    trace = await self._convert_jaeger_trace_to_trace(traces_data[0])
                    return trace
            return None
        except Exception as e:
            print(f'Error getting trace by ID {trace_id}: {e}')
            return None

    async def get_recent_traces(self, start_time: datetime, end_time: datetime, log_group_name: str, service_name_values: list[str] | None=None, service_name_operations: list[str] | None=None, service_environment_values: list[str] | None=None, service_environment_operations: list[str] | None=None, categories: list[str] | None=None, values: list[str] | None=None, operations: list[str] | None=None, pagination_state: dict | None=None) -> tuple[list[Trace], dict | None]:
        """Get recent traces from Jaeger.

        Args:
            start_time (datetime): Start time of the trace
            end_time (datetime): End time of the trace
            log_group_name (str): The log group name
            service_name_values (list[str], optional): Filter values for
                service names if provided
            service_name_operations (list[str], optional): Filter operations
                for service names if provided
            service_environment_values (list[str], optional): Filter values for
                service environments if provided
            service_environment_operations (list[str], optional): Filter
                operations for service environments if provided
            categories (list[str], optional): Filter by categories
                if provided (service names are now included in categories)
            values (list[str], optional): Filter by values if provided
            operations (list[str], optional): Filter operations
                for values if provided

        Returns:
            tuple[list[Trace], dict | None]: Tuple of (traces, next_pagination_state)
        """
        end_time = ensure_utc_datetime(end_time)
        start_time = ensure_utc_datetime(start_time)
        start_time_us = int(start_time.timestamp() * 1000000)
        if categories:
            services = categories
        else:
            current_time = datetime.now(timezone.utc)
            time_window_seconds = int((current_time - start_time).total_seconds())
            services = await self._get_services(lookback_seconds=time_window_seconds)
            if not services:
                return ([], None)
        if 'jaeger' in services:
            services.remove('jaeger')
        if pagination_state and 'last_trace_start_time' in pagination_state:
            end_time_us = pagination_state.get('last_trace_start_time') - 1
        else:
            end_time_us = int(end_time.timestamp() * 1000000)
        traces_with_service: list[tuple[Trace, str]] = []
        for service in services:
            curr_traces = await self._get_traces(service_name=service, start_time=start_time_us, end_time=end_time_us, limit=PAGE_SIZE, offset=0)
            if curr_traces:
                for trace_data in curr_traces:
                    trace = await self._convert_jaeger_trace_to_trace(trace_data)
                    if trace:
                        traces_with_service.append((trace, service))
        if len(traces_with_service) == 0:
            return ([], None)
        traces_with_service.sort(key=lambda x: x[0].start_time, reverse=True)
        page_with_service = traces_with_service[:PAGE_SIZE + 1]
        has_more = len(page_with_service) > PAGE_SIZE
        if has_more:
            page_with_service = page_with_service[:PAGE_SIZE]
        page_traces = [trace for trace, _ in page_with_service]
        if has_more:
            last_trace_start_time_us = int(page_traces[-1].start_time * 1000000)
            next_pagination_state = {'last_trace_start_time': last_trace_start_time_us}
        else:
            next_pagination_state = None
        return (page_traces, next_pagination_state)

    async def _get_services(self, lookback_seconds: int=10 * 60) -> list[str]:
        """Get list of available services from Jaeger."""
        try:
            params = {'lookback': f'{lookback_seconds}s'}
            response = await self._make_request(f'{self.services_url}', params=params)
            if response and 'data' in response:
                return response['data']
            return []
        except Exception as e:
            print(f'Error getting services: {e}')
            return []

    async def _get_traces(self, service_name: str, start_time: int, end_time: int, limit: int, offset: int=0) -> list[dict[str, Any]]:
        """Get traces from Jaeger API.

        Args:
            service_name: Name of the service to query
            start_time: Start time in microseconds
            end_time: End time in microseconds
            limit: Maximum number of traces to fetch
            offset: Number of traces to skip (for pagination)

        Returns:
            List of trace data dictionaries from Jaeger
        """
        try:
            params = {'service': service_name, 'start': start_time, 'end': end_time, 'limit': limit}
            if offset > 0:
                params['offset'] = offset
            response = await self._make_request(f'{self.traces_url}', params=params)
            if response and 'data' in response:
                return response['data']
            return []
        except Exception as e:
            print(f'Error getting traces: {e}')
            return []

    async def _convert_jaeger_trace_to_trace(self, trace_data: dict[str, Any]) -> Optional[Trace]:
        """Convert Jaeger trace data to our Trace model."""
        try:
            trace_id = trace_data.get('traceID')
            if not trace_id:
                return None
            spans_data = trace_data.get('spans', [])
            if not spans_data:
                return None
            spans_dict: dict[str, Span] = {}
            for span_data in spans_data:
                span: Span | None = self._convert_jaeger_span_to_span(span_data)
                if span:
                    spans_dict[span.id] = span
            if not spans_dict:
                return None
            root_spans: list[Span] = self._build_span_hierarchy(spans_data, spans_dict)
            start_times = [span.start_time for span in spans_dict.values()]
            end_times = [span.end_time for span in spans_dict.values()]
            trace_start_time = min(start_times)
            trace_end_time = max(end_times)
            trace_duration = trace_end_time - trace_start_time
            service_name: str | None = None
            service_environment: str | None = None
            if spans_data:
                first_span_data = spans_data[0]
                for tag in first_span_data.get('tags', []):
                    if tag.get('key') == 'service_name':
                        service_name = tag.get('value')
                    if tag.get('key') == 'service_environment':
                        service_environment = tag.get('value')
            traces = construct_traces(service_names=[service_name], service_environments=[service_environment], trace_ids=[trace_id], start_times=[trace_start_time], durations=[trace_duration])
            if traces:
                trace = traces[0]
                trace.spans = root_spans
                sort_spans_recursively(trace.spans)
                accumulate_num_logs_to_traces([trace])
                return trace
            return None
        except Exception as e:
            print(f'Error converting Jaeger trace: {e}')
            return None

    def _build_span_hierarchy(self, spans_data: list[dict[str, Any]], spans_dict: dict[str, Span]) -> list[Span]:
        """Build hierarchical span structure from flat span data.

        Jaeger provides spans in a flat structure with parentSpanID references.
        This method constructs the proper parent-child relationships by:
        1. Creating a mapping of span_id to parent_span_id
        2. Organizing spans into their parent's spans list
        3. Returning only root spans (spans with no parent)

        Args:
            spans_data: Raw span data from Jaeger API
            spans_dict: Dictionary mapping span IDs to Span objects

        Returns:
            List of root spans with properly nested child spans
        """
        parent_map: dict[str, str] = {}
        for span_data in spans_data:
            span_id = span_data.get('spanID')
            references = span_data.get('references', [])
            for ref in references:
                if ref.get('refType') == 'CHILD_OF':
                    parent_span_id = ref.get('spanID')
                    if span_id and parent_span_id:
                        parent_map[span_id] = parent_span_id
                        break
        root_spans: list[Span] = []
        for span_id, span in spans_dict.items():
            if span_id in parent_map:
                parent_id = parent_map[span_id]
                if parent_id in spans_dict:
                    spans_dict[parent_id].spans.append(span)
                else:
                    root_spans.append(span)
            else:
                root_spans.append(span)
        return root_spans

    def _convert_jaeger_span_to_span(self, span_data: dict[str, Any]) -> Span | None:
        """Convert Jaeger span data to our Span model."""
        try:
            span_id = span_data.get('spanID')
            operation_name = span_data.get('operationName', '')
            num_debug_logs: int = 0
            num_info_logs: int = 0
            num_warning_logs: int = 0
            num_error_logs: int = 0
            num_critical_logs: int = 0
            telemetry_sdk_language: str | None = None
            for tag in span_data.get('tags', []):
                if tag.get('key') == 'num_debug_logs':
                    num_debug_logs = int(tag.get('value'))
                if tag.get('key') == 'num_info_logs':
                    num_info_logs = int(tag.get('value'))
                if tag.get('key') == 'num_warning_logs':
                    num_warning_logs = int(tag.get('value'))
                if tag.get('key') == 'num_error_logs':
                    num_error_logs = int(tag.get('value'))
                if tag.get('key') == 'num_critical_logs':
                    num_critical_logs = int(tag.get('value'))
                if tag.get('key') == 'telemetry.sdk.language':
                    telemetry_sdk_language = tag.get('value')
            start_time = span_data.get('startTime', 0) / 1000000.0
            duration_us = span_data.get('duration', 0)
            duration = duration_us / 1000000.0
            end_time = start_time + duration
            span = Span(id=span_id, parent_id=None, name=operation_name, start_time=start_time, end_time=end_time, duration=duration, num_debug_logs=num_debug_logs, num_info_logs=num_info_logs, num_warning_logs=num_warning_logs, num_error_logs=num_error_logs, num_critical_logs=num_critical_logs, telemetry_sdk_language=telemetry_sdk_language, spans=[])
            return span
        except Exception as e:
            print(f'Error converting Jaeger span: {e}')
            return None

    async def _make_request(self, url: str, params: Optional[dict]=None) -> Optional[dict[str, Any]]:
        """Make HTTP request to Jaeger API."""
        loop = asyncio.get_event_loop()

        def _request():
            try:
                response = requests.get(url, params=params)
                if response.ok:
                    if response.content.strip():
                        return response.json()
                    else:
                        print(f'Empty response from {url}')
                        return None
                else:
                    print(f'Error: {response.status_code} - {response.text}')
                    return None
            except requests.exceptions.JSONDecodeError as e:
                print(f'JSON decode error for {url}: {e}')
                return None
            except Exception as e:
                print(f'Request error for {url}: {e}')
                return None
        return await loop.run_in_executor(None, _request)

