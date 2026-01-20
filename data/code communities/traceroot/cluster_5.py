# Cluster 5

def test_create_logs_map_empty_input():
    """Test create_logs_map with empty input."""
    result = create_logs_map([])
    assert result == {}

def create_logs_map(trace_logs: list[dict[str, list[LogEntry]]]) -> dict[str, list[LogEntry]]:
    """Create a mapping from span_id to list of LogEntry objects."""
    logs_map: dict[str, list[LogEntry]] = {}
    for logs_dict in trace_logs:
        for span_id, log_entries in logs_dict.items():
            if span_id not in logs_map:
                logs_map[span_id] = []
            logs_map[span_id].extend(log_entries)
    return logs_map

def test_create_logs_map_single_dict():
    """Test create_logs_map with single dictionary."""
    log_entry = LogEntry(time=1672574400.0, level='INFO', message='Test message', function_name='test_func', file_name='test.py', line_number=10)
    trace_logs = [{'span1': [log_entry]}]
    result = create_logs_map(trace_logs)
    expected = {'span1': [log_entry]}
    assert result == expected

def test_create_logs_map_multiple_dicts():
    """Test create_logs_map with multiple dictionaries."""
    log_entry1 = LogEntry(time=1672574400.0, level='INFO', message='First message', function_name='func1', file_name='file1.py', line_number=10)
    log_entry2 = LogEntry(time=1672574401.0, level='ERROR', message='Second message', function_name='func2', file_name='file2.py', line_number=20)
    trace_logs = [{'span1': [log_entry1]}, {'span1': [log_entry2], 'span2': [log_entry1]}]
    result = create_logs_map(trace_logs)
    expected = {'span1': [log_entry1, log_entry2], 'span2': [log_entry1]}
    assert result == expected

def test_convert_log_entry_basic():
    """Test converting basic LogEntry to LogNode."""
    log_entry = LogEntry(time=1672574400.0, level='DEBUG', message='Debug message', function_name='debug_func', file_name='debug.py', line_number=25)
    result = convert_log_entry_to_log_node(log_entry)
    assert result.log_utc_timestamp == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert result.log_level == 'DEBUG'
    assert result.log_file_name == 'debug.py'
    assert result.log_func_name == 'debug_func'
    assert result.log_message == 'Debug message'
    assert result.log_line_number == 25
    assert result.log_source_code_line == ''
    assert result.log_source_code_lines_above == []
    assert result.log_source_code_lines_below == []

def convert_log_entry_to_log_node(log_entry: LogEntry) -> LogNode:
    """Convert LogEntry to LogNode."""
    return LogNode(log_utc_timestamp=datetime.fromtimestamp(log_entry.time, tz=timezone.utc), log_level=log_entry.level, log_file_name=log_entry.file_name, log_func_name=log_entry.function_name, log_message=log_entry.message, log_line_number=log_entry.line_number, log_source_code_line=log_entry.line or '', log_source_code_lines_above=log_entry.lines_above or [], log_source_code_lines_below=log_entry.lines_below or [])

def test_convert_log_entry_with_source_code():
    """Test converting LogEntry with source code to LogNode."""
    log_entry = LogEntry(time=1672574400.0, level='INFO', message='Info message', function_name='info_func', file_name='info.py', line_number=30, line="logger.info('Info message')", lines_above=['def info_func():', '    # some setup'], lines_below=['    return True'])
    result = convert_log_entry_to_log_node(log_entry)
    assert result.log_source_code_line == "logger.info('Info message')"
    assert result.log_source_code_lines_above == ['def info_func():', '    # some setup']
    assert result.log_source_code_lines_below == ['    return True']

def test_convert_span_with_logs():
    """Test converting span with logs."""
    span = Span(id='span123', parent_id=None, name='test.function', start_time=1672574400.0, end_time=1672574401.0, duration=1.0)
    log_entry = LogEntry(time=1672574400.5, level='INFO', message='Test log', function_name='test_func', file_name='test.py', line_number=10)
    logs_map = {'span123': [log_entry]}
    result = convert_span_to_span_node(span, logs_map)
    assert len(result.logs) == 1
    assert result.logs[0].log_message == 'Test log'
    assert result.logs[0].log_level == 'INFO'

def process_log_events(all_events: list[dict[str, Any]]) -> TraceLogs:
    """Process log events into structured TraceLogs with proper chronological ordering.

    Args:
        all_events: List of raw log events (from CloudWatch, Jaeger, etc.)

    Returns:
        TraceLogs: Structured trace logs with LogEntry objects
    """
    logs: list[dict[str, list[LogEntry]]] = []
    span_logs: dict[str, list[LogEntry]] | None = None
    for event in all_events:
        message: str = event['message']
        if message.startswith('{') and message.endswith('}'):
            log_entry, span_id = _load_json(message)
        else:
            log_entry, span_id = _string_manipulation(message)
        if log_entry is None or span_id is None:
            continue
        if span_logs is None:
            span_logs = {span_id: [log_entry]}
        elif span_id in span_logs:
            span_logs[span_id].append(log_entry)
        else:
            logs.append(span_logs)
            span_logs = {span_id: [log_entry]}
    if span_logs is not None:
        logs.append(span_logs)
    for span_log_dict in logs:
        for span_id, log_entries in span_log_dict.items():
            log_entries.sort(key=lambda entry: (entry.time, getattr(entry, 'line_number', 0)))
    trace_logs = TraceLogs(logs=logs)
    return trace_logs

def _string_manipulation(message: str) -> tuple[LogEntry, str] | tuple[None, None]:
    if 'no-trace' in message:
        return (None, None)
    items = message.split(';')
    time_str = items[0]
    if ',' in time_str:
        date_part, ms_part = time_str.split(',')
        ms_part = ms_part.ljust(6, '0')
        time_str = f'{date_part},{ms_part}'
    time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S,%f')
    time_obj = time_obj.replace(tzinfo=timezone.utc)
    time = time_obj.timestamp()
    level = items[1].upper()
    service_name = items[2]
    message = items[10]
    github_owner = items[4]
    github_repo = items[5]
    commit_id = items[3]
    github_url = f'https://github.com/{github_owner}/{github_repo}/tree/{commit_id}/'
    stack = items[9]
    stack_items = stack.split(' -> ')
    code_info = stack_items[-1]
    code_info = code_info.replace('///(rsc)/./', '')
    code_info_items = code_info.split(':')
    file_path = code_info_items[-3]
    function_name = code_info_items[-2]
    line_number = int(code_info_items[-1])
    github_url = f'{github_url}{file_path}?plain=1#L{line_number}'
    trace_id = items[7] if len(items) > 7 else 'unknown'
    span_id = items[8] if len(items) > 8 else 'no-span'
    log_entry = LogEntry(time=time, level=level, message=message, service_name=service_name, file_name=file_path, function_name=function_name, line_number=line_number, trace_id=trace_id, span_id=span_id, git_url=github_url, commit_id=commit_id)
    return (log_entry, span_id)

