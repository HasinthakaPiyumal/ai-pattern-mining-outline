# Cluster 7

def format_model_status(status: Dict[str, Any], config: QueryPipelineConfig) -> Optional[Dict[str, Any]]:
    """Format model status updates for streaming response."""
    model = status.get('model', 'unknown')
    status_type = status.get('status')
    if status_type == 'pulling':
        content = f'Starting to download model {model}...'
    elif status_type == 'progress':
        percentage = status.get('percentage', 0)
        content = f'Downloading model {model}: `{percentage}%` complete'
    elif status_type == 'complete':
        content = f'Successfully downloaded model {model}'
    elif status_type == 'error' and 'pull' in status.get('error', '').lower():
        error_msg = status.get('error', 'Unknown error')
        content = f'Error downloading model {model}: {error_msg}'
    else:
        return None
    content += '\n'
    return format_stream_response(config, content=content)

def format_stream_response(config: QueryPipelineConfig, content: str='', done: bool=False, done_reason: Optional[str]=None, images: Optional[List[str]]=None, tool_calls: Optional[List[Dict[str, Any]]]=None, **metrics) -> Dict[str, Any]:
    """Format streaming response according to Ollama-API specification."""
    response = {'model': config.model_name, 'created_at': datetime.now(timezone.utc).isoformat(), 'done': done}
    if not done:
        message = {'role': 'assistant', 'content': content}
        if images:
            message['images'] = images
        if tool_calls:
            message['tool_calls'] = tool_calls
        response['message'] = message
    else:
        if done_reason:
            response['done_reason'] = done_reason
        if done_reason == 'error':
            response['message'] = {'role': 'assistant', 'content': content}
        response.update({'total_duration': metrics.get('total_duration', 0), 'load_duration': metrics.get('load_duration', 0), 'prompt_eval_count': metrics.get('prompt_eval_count', 0), 'prompt_eval_duration': metrics.get('prompt_eval_duration', 0), 'eval_count': metrics.get('eval_count', 0), 'eval_duration': metrics.get('eval_duration', 0)})
    return response

def streaming_callback(chunk):
    nonlocal prompt_start
    if prompt_start is None:
        prompt_start = time.time_ns()
    if chunk.content:
        if format_schema and chunk.is_final:
            try:
                content = json.loads(chunk.content)
                response_data = format_stream_response(config, json.dumps(content), done=True, done_reason='stop')
            except json.JSONDecodeError:
                response_data = format_stream_response(config, 'Error: Failed to generate valid JSON response.', done=True, done_reason='error')
        else:
            response_data = format_stream_response(config, chunk.content, images=getattr(chunk, 'images', None), tool_calls=getattr(chunk, 'tool_calls', None))
        q.put(json.dumps(response_data) + '\n')

def run_rag():
    try:
        load_start = time.time_ns()
        for status in rag.initialize_and_check_models():
            if (status_data := format_model_status(status, config)):
                q.put(json.dumps(status_data) + '\n')
            if status.get('status') == 'error':
                error_data = format_stream_response(config, f'Error: Model initialization failed - {status.get('error')}', done=True, done_reason='error')
                q.put(json.dumps(error_data) + '\n')
                return
        load_duration = time.time_ns() - load_start
        response_text = rag.run_query(query=query, conversation=conversation, print_response=DEBUG)
        end_time = time.time_ns()
        final_data = format_stream_response(config, done=True, done_reason='stop', total_duration=end_time - start_time, load_duration=load_duration, prompt_eval_count=len(conversation) + 1, prompt_eval_duration=end_time - (prompt_start or start_time), eval_count=len(response_text.split()) if response_text is not None else 0, eval_duration=end_time - (prompt_start or start_time))
        q.put(json.dumps(final_data) + '\n')
    except elasticsearch.BadRequestError as e:
        error_data = format_stream_response(config, content=f'Error: Embedding retriever error - {str(e)}', done=True, done_reason='error')
        q.put(json.dumps(error_data) + '\n')
    except Exception as e:
        error_data = format_stream_response(config, content=f'Error: {str(e)}', done=True, done_reason='error')
        logger.error(f'Error in RAG pipeline: {e}', exc_info=True)
        q.put(json.dumps(error_data) + '\n')

