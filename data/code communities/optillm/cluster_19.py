# Cluster 19

def none_approach(client: Any, model: str, original_messages: List[Dict[str, str]], request_id: str=None, **kwargs) -> Dict[str, Any]:
    """
    Direct proxy approach that passes through all parameters to the underlying endpoint.
    
    Args:
        client: OpenAI client instance
        model: Model identifier
        original_messages: Original messages from the request
        request_id: Optional request ID for conversation logging
        **kwargs: Additional parameters to pass through
    
    Returns:
        Dict[str, Any]: Full OpenAI API response
    """
    if model.startswith('none-'):
        model = model[5:]
    try:
        normalized_messages = normalize_message_content(original_messages)
        provider_request = {'model': model, 'messages': normalized_messages, **kwargs}
        response = client.chat.completions.create(model=model, messages=normalized_messages, **kwargs)
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
        if conversation_logger and request_id:
            conversation_logger.log_provider_call(request_id, provider_request, response_dict)
        return response_dict
    except Exception as e:
        if conversation_logger and request_id:
            conversation_logger.log_error(request_id, f'Error in none approach: {str(e)}')
        logger.error(f'Error in none approach: {str(e)}')
        raise

def normalize_message_content(messages):
    """
    Ensure all message content fields are strings, not lists.
    Some models don't handle list-format content correctly.
    """
    normalized_messages = []
    for message in messages:
        normalized_message = message.copy()
        content = message.get('content', '')
        if isinstance(content, list):
            text_content = ' '.join((item.get('text', '') for item in content if isinstance(item, dict) and item.get('type') == 'text'))
            normalized_message['content'] = text_content
        normalized_messages.append(normalized_message)
    return normalized_messages

def log_provider_call(request_id: str, provider_request: Dict[str, Any], provider_response: Dict[str, Any]) -> None:
    """Log a provider call using the global logger instance"""
    if _global_logger and _global_logger.enabled:
        _global_logger.log_provider_call(request_id, provider_request, provider_response)

def log_error(request_id: str, error_message: str) -> None:
    """Log an error using the global logger instance"""
    if _global_logger and _global_logger.enabled:
        _global_logger.log_error(request_id, error_message)

def process_single_response(text):
    if not has_conversation_tags(text):
        return text
    messages = []
    parts = re.split('(?=(User:|Assistant:))', text.strip())
    parts = [p for p in parts if p.strip()]
    for part in parts:
        part = part.strip()
        if part.startswith('User:'):
            messages.append({'role': 'user', 'content': part[5:].strip()})
        elif part.startswith('Assistant:'):
            messages.append({'role': 'assistant', 'content': part[10:].strip()})
    return messages

def has_conversation_tags(text):
    return 'User:' in text or 'Assistant:' in text

def tagged_conversation_to_messages(response_text):
    """Convert a tagged conversation string or list of strings into a list of messages.
    If the input doesn't contain User:/Assistant: tags, return it as is.
    
    Args:
        response_text: Either a string containing "User:" and "Assistant:" tags,
                      or a list of such strings.
    
    Returns:
        If input has tags: A list of message dictionaries.
        If input has no tags: The original input.
    """

    def has_conversation_tags(text):
        return 'User:' in text or 'Assistant:' in text

    def process_single_response(text):
        if not has_conversation_tags(text):
            return text
        messages = []
        parts = re.split('(?=(User:|Assistant:))', text.strip())
        parts = [p for p in parts if p.strip()]
        for part in parts:
            part = part.strip()
            if part.startswith('User:'):
                messages.append({'role': 'user', 'content': part[5:].strip()})
            elif part.startswith('Assistant:'):
                messages.append({'role': 'assistant', 'content': part[10:].strip()})
        return messages
    if isinstance(response_text, list):
        processed = [process_single_response(text) for text in response_text]
        if all((isinstance(p, str) for p in processed)):
            return response_text
        return processed
    else:
        return process_single_response(response_text)

@app.route('/v1/chat/completions', methods=['POST'])
def proxy():
    logger.info('Received request to /v1/chat/completions')
    data = request.get_json()
    auth_header = request.headers.get('Authorization')
    bearer_token = ''
    if auth_header and auth_header.startswith('Bearer '):
        bearer_token = auth_header.split('Bearer ')[1].strip()
        logger.debug(f'Intercepted Bearer Token: {bearer_token}')
    logger.debug(f'Request data: {data}')
    stream = data.get('stream', False)
    messages = data.get('messages', [])
    model = data.get('model', server_config['model'])
    n = data.get('n', server_config['n'])
    response_format = data.get('response_format', None)
    explicit_keys = {'stream', 'messages', 'model', 'n', 'response_format'}
    request_config = {k: v for k, v in data.items() if k not in explicit_keys}
    request_config.update({'stream': stream, 'n': n, 'response_format': response_format})
    optillm_approach = data.get('optillm_approach', server_config['approach'])
    logger.debug(data)
    server_config['mcts_depth'] = data.get('mcts_depth', server_config['mcts_depth'])
    server_config['mcts_exploration'] = data.get('mcts_exploration', server_config['mcts_exploration'])
    server_config['mcts_simulations'] = data.get('mcts_simulations', server_config['mcts_simulations'])
    system_prompt, initial_query, message_optillm_approach = parse_conversation(messages)
    if message_optillm_approach:
        optillm_approach = message_optillm_approach
    if optillm_approach != 'auto':
        model = f'{optillm_approach}-{model}'
    base_url = server_config['base_url']
    default_client, api_key = get_config()
    operation, approaches, model = parse_combined_approach(model, known_approaches, plugin_approaches)
    request_id = None
    if conversation_logger and conversation_logger.enabled:
        request_id = conversation_logger.start_conversation(client_request={'messages': messages, 'model': data.get('model', server_config['model']), 'stream': stream, 'n': n, **{k: v for k, v in data.items() if k not in {'messages', 'model', 'stream', 'n'}}}, approach=approaches[0] if len(approaches) == 1 else f'{operation}({','.join(approaches)})', model=model)
    request_id_str = f' [Request: {request_id}]' if request_id else ''
    logger.info(f'Using approach(es) {approaches}, operation {operation}, with model {model}{request_id_str}')
    if request_id:
        logger.info(f'Request {request_id}: Starting processing')
    if bearer_token != '' and bearer_token.startswith('sk-'):
        api_key = bearer_token
        if base_url != '':
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)
    else:
        client = default_client
    try:
        if request_batcher is not None:
            try:
                batch_request_data = {'system_prompt': system_prompt, 'initial_query': initial_query, 'client': client, 'model': model, 'request_config': request_config, 'approaches': approaches, 'operation': operation, 'n': n, 'stream': stream, 'optillm_approach': optillm_approach}
                logger.debug('Routing request to batch processor')
                result = request_batcher.add_request(batch_request_data)
                return (jsonify(result), 200)
            except BatchingError as e:
                logger.error(f'Batch processing failed: {e}')
                return (jsonify({'error': str(e)}), 500)
        contains_none = any((approach == 'none' for approach in approaches))
        if operation == 'SINGLE' and approaches[0] == 'none':
            result, completion_tokens = execute_single_approach(approaches[0], system_prompt, initial_query, client, model, request_config, request_id)
            logger.debug(f'Direct proxy response: {result}')
            if conversation_logger and request_id:
                conversation_logger.log_final_response(request_id, result)
                conversation_logger.finalize_conversation(request_id)
            if stream:
                if request_id:
                    logger.info(f'Request {request_id}: Completed (streaming response)')
                return Response(generate_streaming_response(extract_contents(result), model), content_type='text/event-stream')
            else:
                if request_id:
                    logger.info(f'Request {request_id}: Completed')
                return (jsonify(result), 200)
        elif operation == 'AND' or operation == 'OR':
            if contains_none:
                raise ValueError("'none' approach cannot be combined with other approaches")
        response, completion_tokens = execute_n_times(n, approaches, operation, system_prompt, initial_query, client, model, request_config, request_id)
        if operation == 'SINGLE' and isinstance(response, dict) and ('choices' in response) and ('usage' in response):
            if conversation_logger and request_id:
                conversation_logger.log_final_response(request_id, response)
                conversation_logger.finalize_conversation(request_id)
            if stream:
                if request_id:
                    logger.info(f'Request {request_id}: Completed (streaming response)')
                return Response(generate_streaming_response(extract_contents(response), model), content_type='text/event-stream')
            else:
                if request_id:
                    logger.info(f'Request {request_id}: Completed')
                return (jsonify(response), 200)
    except Exception as e:
        if conversation_logger and request_id:
            conversation_logger.log_error(request_id, str(e))
            conversation_logger.finalize_conversation(request_id)
        request_id_str = f' {request_id}' if request_id else ''
        logger.error(f'Error processing request{request_id_str}: {str(e)}')
        return (jsonify({'error': str(e)}), 500)
    if isinstance(response, list):
        processed_response = tagged_conversation_to_messages(response)
        if processed_response != response:
            response = [msg[-1]['content'] if isinstance(msg, list) and msg else msg for msg in processed_response]
    else:
        messages = tagged_conversation_to_messages(response)
        if isinstance(messages, list) and messages:
            response = messages[-1]['content']
    if stream:
        return Response(generate_streaming_response(response, model), content_type='text/event-stream')
    else:
        reasoning_tokens = 0
        if isinstance(response, str):
            reasoning_tokens = count_reasoning_tokens(response)
        elif isinstance(response, list) and response:
            reasoning_tokens = sum((count_reasoning_tokens(resp) for resp in response if isinstance(resp, str)))
        response_data = {'model': model, 'choices': [], 'usage': {'completion_tokens': completion_tokens, 'completion_tokens_details': {'reasoning_tokens': reasoning_tokens}}}
        if isinstance(response, list):
            for index, resp in enumerate(response):
                response_data['choices'].append({'index': index, 'message': {'role': 'assistant', 'content': resp}, 'finish_reason': 'stop'})
        else:
            response_data['choices'].append({'index': 0, 'message': {'role': 'assistant', 'content': response}, 'finish_reason': 'stop'})
        if conversation_logger and request_id:
            conversation_logger.log_final_response(request_id, response_data)
            conversation_logger.finalize_conversation(request_id)
        logger.debug(f'API response: {response_data}')
        if request_id:
            logger.info(f'Request {request_id}: Completed')
        return (jsonify(response_data), 200)

def parse_conversation(messages):
    system_prompt = ''
    conversation = []
    optillm_approach = None
    for message in messages:
        role = message['role']
        content = message['content']
        if isinstance(content, list):
            text_content = ' '.join((item['text'] for item in content if isinstance(item, dict) and item.get('type') == 'text'))
        else:
            text_content = content
        if role == 'system':
            system_prompt, optillm_approach = extract_optillm_approach(text_content)
        elif role == 'user':
            if not optillm_approach:
                text_content, optillm_approach = extract_optillm_approach(text_content)
            conversation.append(f'User: {text_content}')
        elif role == 'assistant':
            conversation.append(f'Assistant: {text_content}')
    initial_query = '\n'.join(conversation)
    return (system_prompt, initial_query, optillm_approach)

def parse_combined_approach(model: str, known_approaches: list, plugin_approaches: dict):
    if model == 'auto':
        return ('SINGLE', ['none'], model)
    parts = model.split('-')
    approaches = []
    operation = 'SINGLE'
    model_parts = []
    parsing_approaches = True
    for part in parts:
        if parsing_approaches:
            if part in known_approaches or part in plugin_approaches:
                approaches.append(part)
            elif '&' in part:
                operation = 'AND'
                approaches.extend(part.split('&'))
            elif '|' in part:
                operation = 'OR'
                approaches.extend(part.split('|'))
            else:
                parsing_approaches = False
                model_parts.append(part)
        else:
            model_parts.append(part)
    if not approaches:
        approaches = ['none']
        operation = 'SINGLE'
    actual_model = '-'.join(model_parts)
    return (operation, approaches, actual_model)

def generate_streaming_response(final_response, model):
    if isinstance(final_response, list):
        for index, response in enumerate(final_response):
            yield ('data: ' + json.dumps({'choices': [{'delta': {'content': response}, 'index': index, 'finish_reason': 'stop'}], 'model': model}) + '\n\n')
    else:
        yield ('data: ' + json.dumps({'choices': [{'delta': {'content': final_response}, 'index': 0, 'finish_reason': 'stop'}], 'model': model}) + '\n\n')
    yield 'data: [DONE]\n\n'

def extract_contents(response_obj):
    contents = []
    responses = response_obj if isinstance(response_obj, list) else [response_obj]
    for response in responses:
        if response.get('choices') and len(response['choices']) > 0 and response['choices'][0].get('message') and response['choices'][0]['message'].get('content'):
            contents.append(response['choices'][0]['message']['content'])
    return contents

