# Cluster 2

def test_log_node_creation():
    """Test creating a LogNode instance."""
    timestamp = datetime.now(timezone.utc)
    log_node = LogNode(log_utc_timestamp=timestamp, log_level='INFO', log_file_name='test.py', log_func_name='test_function', log_message='Test message', log_line_number=42, log_source_code_line="print('hello')", log_source_code_lines_above=['# comment above'], log_source_code_lines_below=['# comment below'])
    assert log_node.log_utc_timestamp == timestamp
    assert log_node.log_level == 'INFO'
    assert log_node.log_file_name == 'test.py'
    assert log_node.log_func_name == 'test_function'
    assert log_node.log_message == 'Test message'
    assert log_node.log_line_number == 42
    assert log_node.log_source_code_line == "print('hello')"
    assert log_node.log_source_code_lines_above == ['# comment above']
    assert log_node.log_source_code_lines_below == ['# comment below']

def test_log_node_to_dict_all_features():
    """Test LogNode to_dict with all features."""
    timestamp = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    log_node = LogNode(log_utc_timestamp=timestamp, log_level='ERROR', log_file_name='error.py', log_func_name='error_function', log_message='Error occurred', log_line_number=100, log_source_code_line='raise Exception()', log_source_code_lines_above=['try:', '    do_something()'], log_source_code_lines_below=['except:', '    handle_error()'])
    all_features = list(LogFeature)
    result = log_node.to_dict(all_features)
    expected = {'log utc timestamp': '2023-01-01 12:00:00+00:00', 'log level': 'ERROR', 'file name': 'error.py', 'function name': 'error_function', 'log message value': 'Error occurred', 'line number': '100', 'log line source code': 'raise Exception()', 'lines above log source code': ['try:', '    do_something()'], 'lines below log source code': ['except:', '    handle_error()']}
    assert result == expected

def test_log_node_to_dict_subset_features():
    """Test LogNode to_dict with only a subset of features."""
    timestamp = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    log_node = LogNode(log_utc_timestamp=timestamp, log_level='WARNING', log_file_name='warn.py', log_func_name='warn_function', log_message='Warning message', log_line_number=50, log_source_code_line="warn('test')", log_source_code_lines_above=[], log_source_code_lines_below=[])
    features = [LogFeature.LOG_LEVEL, LogFeature.LOG_MESSAGE_VALUE, LogFeature.LOG_LINE_NUMBER]
    result = log_node.to_dict(features)
    expected = {'log level': 'WARNING', 'log message value': 'Warning message', 'line number': '50'}
    assert result == expected

def test_span_node_creation():
    """Test creating a SpanNode instance."""
    start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2023, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
    span_node = SpanNode(span_id='span123', func_full_name='module.function', span_latency=1.5, span_utc_start_time=start_time, span_utc_end_time=end_time)
    assert span_node.span_id == 'span123'
    assert span_node.func_full_name == 'module.function'
    assert span_node.span_latency == 1.5
    assert span_node.span_utc_start_time == start_time
    assert span_node.span_utc_end_time == end_time
    assert span_node.logs == []
    assert span_node.children_spans == []

def test_span_node_to_dict_no_children_no_logs():
    """Test SpanNode to_dict with no children or logs."""
    start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2023, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
    span_node = SpanNode(span_id='span123', func_full_name='module.function', span_latency=1.5, span_utc_start_time=start_time, span_utc_end_time=end_time)
    result = span_node.to_dict()
    expected = {'span_id': 'span123', 'func_full_name': 'module.function', 'span latency': '1.5', 'span utc start time': '2023-01-01 12:00:00+00:00', 'span utc end time': '2023-01-01 12:00:01+00:00'}
    assert result == expected

def test_span_node_to_dict_with_logs():
    """Test SpanNode to_dict with logs."""
    start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2023, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
    log_time = datetime(2023, 1, 1, 12, 0, 0, 500000, tzinfo=timezone.utc)
    log_node = LogNode(log_utc_timestamp=log_time, log_level='INFO', log_file_name='test.py', log_func_name='test_func', log_message='Test log', log_line_number=10, log_source_code_line="print('test')", log_source_code_lines_above=[], log_source_code_lines_below=[])
    span_node = SpanNode(span_id='span123', func_full_name='module.function', span_latency=1.5, span_utc_start_time=start_time, span_utc_end_time=end_time, logs=[log_node])
    span_features = [SpanFeature.SPAN_LATENCY]
    log_features = [LogFeature.LOG_LEVEL, LogFeature.LOG_MESSAGE_VALUE]
    result = span_node.to_dict(span_features, log_features)
    expected = {'span_id': 'span123', 'func_full_name': 'module.function', 'span latency': '1.5', 'log_0': {'log level': 'INFO', 'log message value': 'Test log'}}
    assert result == expected

def test_span_node_to_dict_with_children():
    """Test SpanNode to_dict with child spans."""
    parent_start = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    parent_end = datetime(2023, 1, 1, 12, 0, 2, tzinfo=timezone.utc)
    child_start = datetime(2023, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
    child_end = datetime(2023, 1, 1, 12, 0, 1, 500000, tzinfo=timezone.utc)
    child_span = SpanNode(span_id='child123', func_full_name='child.function', span_latency=0.5, span_utc_start_time=child_start, span_utc_end_time=child_end)
    parent_span = SpanNode(span_id='parent123', func_full_name='parent.function', span_latency=2.0, span_utc_start_time=parent_start, span_utc_end_time=parent_end, children_spans=[child_span])
    span_features = [SpanFeature.SPAN_LATENCY]
    result = parent_span.to_dict(span_features, [])
    expected = {'span_id': 'parent123', 'func_full_name': 'parent.function', 'span latency': '2.0', 'child123': {'span_id': 'child123', 'func_full_name': 'child.function', 'span latency': '0.5'}}
    assert result == expected

