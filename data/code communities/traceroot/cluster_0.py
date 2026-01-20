# Cluster 0

def test_load_json_with_skip_fields():
    """Test that _load_json creates a filtered JSON message with non-skipped fields"""
    test_data = {'message': 'Test log message', 'level': 'info', 'timestamp': '2025-08-21 02:14:39,175', 'service_name': 'test-service', 'github_commit_hash': 'abc123', 'github_owner': 'test-owner', 'github_repo_name': 'test-repo', 'environment': 'development', 'userId': 'user123', 'requestId': 'req456', 'trace_id': '1-998fb9ed-9c366715fad58dfe34c822a4', 'span_id': 'e046c62b8f0667e7', 'ingestionTime': 1755767680543, 'eventId': '39154927641119437560701839122329487456907060165707235328', 'stack_trace': 'examples/simple-example.ts:makeRequest:8'}
    message_json = json.dumps(test_data)
    log_entry, span_id = _load_json(message_json)
    assert log_entry is not None
    assert span_id is not None
    assert span_id == 'e046c62b8f0667e7'
    parsed_message = json.loads(log_entry.message)
    assert parsed_message['message'] == 'Test log message'
    assert parsed_message['userId'] == 'user123'
    assert parsed_message['requestId'] == 'req456'
    for field in SKIP_LOG_FIELDS:
        not_in_message = field not in parsed_message
        if field in test_data:
            assert not_in_message, f"Skipped field '{field}' should not be in"
    assert isinstance(parsed_message, dict)

def _load_json(message: str) -> tuple[LogEntry, str] | tuple[None, None]:
    try:
        json_data = json.loads(message)
        time_str = json_data.get('timestamp')
        if not time_str:
            return (None, None)
        try:
            if 'T' in time_str:
                if time_str.endswith('Z'):
                    time_obj = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                else:
                    time_obj = datetime.fromisoformat(time_str)
                    if time_obj.tzinfo is None:
                        time_obj = time_obj.replace(tzinfo=timezone.utc)
            else:
                time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S,%f')
                time_obj = time_obj.replace(tzinfo=timezone.utc)
        except ValueError as ve:
            print(f"[DEBUG] Failed to parse timestamp '{time_str}': {ve}")
            return (None, None)
        time = time_obj.timestamp()
    except Exception as e:
        print(f'[DEBUG] Error parsing JSON log: {e}, message: {message[:200]}')
        return (None, None)
    level = json_data['level'].upper()
    if level == 'WARN':
        level = 'WARNING'
    filtered_data = {}
    for key, value in json_data.items():
        if key not in SKIP_LOG_FIELDS:
            filtered_data[key] = value
    if len(filtered_data) > 1:
        message = json.dumps(filtered_data)
    else:
        message = json_data['message']
    if 'stack_trace' in json_data:
        stack = json_data['stack_trace']
        stack_items = stack.split(' -> ')
        code_info = stack_items[-1]
        code_info = code_info.replace('///(rsc)/./', '')
        code_info_items = code_info.split(':')
        file_path = code_info_items[-3]
        function_name = code_info_items[-2]
        line_number = int(code_info_items[-1])
    else:
        return (None, None)
    github_owner = json_data['github_owner']
    github_repo = json_data['github_repo_name']
    commit_id = json_data['github_commit_hash']
    span_id = json_data['span_id']
    trace_id = json_data.get('trace_id', 'unknown')
    service_name = json_data.get('service_name')
    github_url = f'https://github.com/{github_owner}/{github_repo}/tree/{commit_id}/'
    github_url = f'{github_url}{file_path}?plain=1#L{line_number}'
    log_entry = LogEntry(time=time, level=level, message=message, service_name=service_name, file_name=file_path, function_name=function_name, line_number=line_number, trace_id=trace_id, span_id=span_id, git_url=github_url, commit_id=commit_id)
    return (log_entry, span_id)

def test_load_json_without_extra_fields():
    """Test that _load_json returns plain message when all
    fields are skipped except message
    """
    test_data = {'message': 'Simple message', 'level': 'info', 'timestamp': '2025-08-21 02:14:39,175', 'github_owner': 'test-owner', 'github_repo_name': 'test-repo', 'github_commit_hash': 'abc123', 'span_id': 'span123', 'stack_trace': 'examples/simple-example.ts:makeRequest:8'}
    message_json = json.dumps(test_data)
    log_entry, span_id = _load_json(message_json)
    assert log_entry is not None
    assert span_id == 'span123'
    assert log_entry.message == 'Simple message'

def test_load_json_with_mixed_fields():
    """Test that _load_json correctly filters mixed skipped and non-skipped fields"""
    test_data = {'message': 'Mixed fields message', 'level': 'error', 'timestamp': '2025-08-21 02:14:39,175', 'service_name': 'should-be-skipped', 'customField': 'should-be-included', 'github_owner': 'test-owner', 'github_repo_name': 'test-repo', 'github_commit_hash': 'abc123', 'span_id': 'span456', 'requestId': 'req789', 'stack_trace': 'examples/simple-example.ts:makeRequest:8'}
    message_json = json.dumps(test_data)
    log_entry, span_id = _load_json(message_json)
    assert log_entry is not None
    assert span_id == 'span456'
    parsed_message = json.loads(log_entry.message)
    assert parsed_message['message'] == 'Mixed fields message'
    assert parsed_message['customField'] == 'should-be-included'
    assert parsed_message['requestId'] == 'req789'
    assert 'service_name' not in parsed_message
    assert 'level' not in parsed_message
    assert 'timestamp' not in parsed_message
    assert 'github_owner' not in parsed_message
    assert 'span_id' not in parsed_message

def test_load_json_returns_plain_message_when_only_message_field():
    """Test that _load_json returns plain message string when
    filtered_data has only message field.
    """
    test_data = {'message': 'Only message should remain', 'level': 'info', 'timestamp': '2025-08-21 02:14:39,175', 'service_name': 'test-service', 'github_owner': 'test-owner', 'github_repo_name': 'test-repo', 'github_commit_hash': 'abc123', 'span_id': 'span123', 'stack_trace': 'examples/simple-example.ts:makeRequest:8'}
    message_json = json.dumps(test_data)
    log_entry, span_id = _load_json(message_json)
    assert log_entry is not None
    assert span_id == 'span123'
    assert log_entry.message == 'Only message should remain'
    try:
        parsed = json.loads(log_entry.message)
        assert isinstance(parsed, str)
    except json.JSONDecodeError:
        pass

