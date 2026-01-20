# Cluster 22

def parse_response(response) -> Optional[Policy]:
    has_usage = hasattr(response, 'usage')
    prompt_tokens = has_usage and hasattr(response.usage, 'prompt_tokens')
    completion_tokens = has_usage and hasattr(response.usage, 'completion_tokens')
    if hasattr(response, 'choices'):
        choice = response.choices[0]
        input_tokens = response.usage.prompt_tokens if prompt_tokens else 0
        output_tokens = response.usage.completion_tokens if completion_tokens else 0
    else:
        choice = response.content[0]
        input_tokens = response.usage.input_tokens if prompt_tokens else 0
        output_tokens = response.usage.output_tokens if completion_tokens else 0
    total_tokens = input_tokens + output_tokens
    try:
        result = PythonParser.extract_code(choice)
        if result is None:
            return None
        code, text_response = result
    except Exception as e:
        print(f'Failed to extract code from choice: {str(e)}')
        return None
    if not code:
        return None
    policy = Policy(code=code, meta=PolicyMeta(output_tokens=output_tokens, input_tokens=input_tokens, total_tokens=total_tokens, text_response=text_response))
    return policy

def track_timing(operation_name: Optional[str]=None):
    """Decorator for tracking function performance"""

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            with timing_tracker.track(name, function=func.__name__, args=args, kwargs=kwargs):
                return func(*args, **kwargs)
        return wrapper
    if callable(operation_name):
        func = operation_name
        operation_name = None
        return decorator(func)
    return decorator

def decorator(func):

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        name = operation_name or func.__name__
        async with timing_tracker.track_async(name, function=func.__name__):
            return await func(*args, **kwargs)
    return wrapper

def track_timing_async(operation_name: Optional[str]=None):
    """Decorator for tracking async function performance"""

    def decorator(func):

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            async with timing_tracker.track_async(name, function=func.__name__):
                return await func(*args, **kwargs)
        return wrapper
    if callable(operation_name):
        func = operation_name
        operation_name = None
        return decorator(func)
    return decorator

def remove_whitespace_blocks(messages):
    return [message for message in messages if isinstance(message['content'], str) and message['content'].strip() or (isinstance(message['content'], list) and len(message['content']) > 0)]

def merge_contiguous_messages(messages):
    if not messages:
        return messages
    merged_messages = [messages[0]]
    for message in messages[1:]:
        if message['role'] == merged_messages[-1]['role']:
            if isinstance(merged_messages[-1]['content'], str) and isinstance(message['content'], str):
                merged_messages[-1]['content'] += '\n\n' + message['content']
            else:
                merged_messages.append(message)
        else:
            merged_messages.append(message)
    return merged_messages

