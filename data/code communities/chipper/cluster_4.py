# Cluster 4

def get_elasticsearch_config(index: Optional[str]=None) -> dict[str, Any]:
    """Get Elasticsearch configuration if enabled."""
    if not (es_url := os.getenv(EnvKeys.ES_URL)):
        return {}
    config = {'es_url': es_url, 'es_index': index or os.getenv(EnvKeys.ES_INDEX), 'es_basic_auth_user': os.getenv(EnvKeys.ES_BASIC_AUTH_USER), 'es_basic_auth_password': os.getenv(EnvKeys.ES_BASIC_AUTH_PASSWORD)}
    for env_key, default in [(EnvKeys.ES_TOP_K, '5'), (EnvKeys.ES_NUM_CANDIDATES, '-1')]:
        if (value := get_env_value(env_key, int, default)):
            config[env_key.lower()] = value
    return config

def get_env_value(key: str, converter: Optional[Callable]=None, default: Optional[str]=None) -> Any:
    """Get and convert environment variable value with optional default."""
    value = os.getenv(key)
    if value is None:
        return None
    if converter:
        try:
            return converter(default if value == '' else value)
        except (ValueError, TypeError):
            return None
    return value

def create_pipeline_config(model: Optional[str]=None, index: Optional[str]=None, temperature: Optional[float]=None, top_k: Optional[int]=None, top_p: Optional[float]=None, min_p: Optional[float]=None, repeat_last_n: Optional[int]=None, repeat_penalty: Optional[float]=None, num_predict: Optional[int]=None, tfs_z: Optional[float]=None, context_window: Optional[int]=None, seed: Optional[int]=None, **additional_params: Dict[str, Any]) -> QueryPipelineConfig:
    """Create pipeline configuration from environment variables with optional parameter overrides."""
    config = get_provider_specific_config()
    if model:
        config['model_name'] = model
    params = GenerationParams()
    for param in params.__annotations__:
        env_key, converter, default = getattr(params, param)
        if (value := get_env_value(env_key, converter, default)):
            config[param] = value
    generation_params = {'temperature': temperature, 'top_k': top_k, 'top_p': top_p, 'min_p': min_p, 'repeat_last_n': repeat_last_n, 'repeat_penalty': repeat_penalty, 'num_predict': num_predict, 'tfs_z': tfs_z, 'context_window': context_window, 'seed': seed}
    config.update({k: v for k, v in generation_params.items() if v is not None})
    config.update(additional_params)
    if (mirostat := get_env_value('MIROSTAT', int)):
        config['mirostat'] = mirostat
        for param in ['MIROSTAT_ETA', 'MIROSTAT_TAU']:
            if (value := get_env_value(param, float)):
                config[param.lower()] = value
    if (allow_pull := os.getenv(EnvKeys.ALLOW_MODEL_PULL)):
        config['allow_model_pull'] = allow_pull.lower() == 'true'
    if (value := os.getenv(EnvKeys.ENABLE_CONVERSATION_LOGS)):
        config['enable_conversation_logs'] = value.lower() == 'true'
    if (stop_sequence := os.getenv('STOP_SEQUENCE')):
        config['stop_sequence'] = stop_sequence
    config.update(get_elasticsearch_config(index))
    logger.info('\nPipeline Configuration:')
    for key, value in sorted(config.items()):
        if any((sensitive in key.lower() for sensitive in ['password', 'key', 'auth'])):
            logger.info(f'  {key}: ****')
        else:
            logger.info(f'  {key}: {value}')
    return QueryPipelineConfig(**config)

def get_provider_specific_config() -> dict[str, Any]:
    """Get provider-specific configuration."""
    provider = ModelProvider.HUGGINGFACE if os.getenv(EnvKeys.PROVIDER, 'ollama').lower() == 'hf' else ModelProvider.OLLAMA
    config = {'provider': provider, 'model_name': os.getenv(EnvKeys.HF_MODEL_NAME if provider == ModelProvider.HUGGINGFACE else EnvKeys.MODEL_NAME), 'embedding_model': os.getenv(EnvKeys.HF_EMBEDDING_MODEL if provider == ModelProvider.HUGGINGFACE else EnvKeys.EMBEDDING_MODEL), 'system_prompt': SYSTEM_PROMPT_VALUE}
    if provider == ModelProvider.HUGGINGFACE:
        config['hf_api_key'] = os.getenv(EnvKeys.HF_API_KEY)
    elif (ollama_url := os.getenv(EnvKeys.OLLAMA_URL)):
        config['ollama_url'] = ollama_url
    return config

@app.route('/api/chat', methods=['POST'])
@require_api_key
def chat():
    try:
        if DEBUG:
            log_request_info(request)
        data = request.get_json()
        if not data:
            logger.error('No JSON payload received.')
            abort(400, description='Invalid JSON payload.')
        messages = data.get('messages', [])
        if not messages:
            abort(400, description='No messages provided')
        model = None
        if not IGNORE_MODEL_REQUEST:
            model = data.get('model')
            if model and (not ALLOW_MODEL_CHANGE):
                abort(403, description='Model changes are not allowed')
        for message in messages:
            if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                abort(400, description='Invalid message format')
            if message['role'] != '' and message['role'] not in ['system', 'user', 'assistant', 'tool']:
                abort(400, description='Invalid message role')
        options = data.get('options', {})
        stream = data.get('stream', True)
        temperature = None
        top_k = None
        top_p = None
        seed = None
        if ALLOW_MODEL_PARAMETER_CHANGE:
            temperature = data.get('temperature', None)
            top_k = data.get('top_k', None)
            top_p = data.get('top_p', None)
            seed = data.get('seed', None)
        index = options.get('index')
        if index and (not ALLOW_INDEX_CHANGE):
            abort(403, description='Index changes are not allowed')
        for message in messages:
            if 'images' in message and (not isinstance(message['images'], list)):
                abort(400, description='Images must be provided as a list')
        config = create_pipeline_config(model=model, index=index, temperature=temperature, top_k=top_k, top_p=top_p, seed=seed)
        query = None
        for message in reversed(messages):
            content = message.get('content')
            if content:
                query = content
                break
        if not query:
            abort(400, description='No message with content found')
        conversation = messages[:-1] if len(messages) > 1 else []
        if stream:
            return handle_streaming_response(config, query, conversation)
        else:
            return handle_standard_response(config, query, conversation)
    except Exception as e:
        logger.error(f'Error processing chat request: {str(e)}', exc_info=True)
        abort(500, description='Internal Server Error.')

@app.before_request
def log_request_info():
    if request.path == '/' or request.path == '/health':
        return
    log_data = {'method': request.method, 'path': request.path, 'remote_addr': request.remote_addr, 'user_agent': request.headers.get('User-Agent'), 'request_id': request.headers.get('X-Request-ID')}
    logger.debug('Incoming request', extra=log_data)

def handle_standard_response(config: QueryPipelineConfig, query: str, conversation: List[Dict[str, str]], format_schema: Optional[Dict[str, Any]]=None, options: Optional[Dict[str, Any]]=None) -> Response:
    start_time = time.time_ns()
    rag = RAGQueryPipeline(config=config)
    try:
        load_start = time.time_ns()
        for status in rag.initialize_and_check_models():
            if status.get('status') == 'error':
                raise Exception(f'Model initialization failed: {status.get('error')}')
        load_duration = time.time_ns() - load_start
        prompt_start = time.time_ns()
        result = rag.run_query(query=query, conversation=conversation, print_response=False)
        end_time = time.time_ns()
        response_content = result
        eval_count = len(response_content.split()) if response_content else 0
        response = {'model': config.model_name, 'created_at': datetime.now(timezone.utc).isoformat(), 'message': {'role': 'assistant', 'content': response_content}, 'done': True, 'done_reason': 'stop', 'total_duration': end_time - start_time, 'load_duration': load_duration, 'prompt_eval_count': len(conversation) + 1, 'prompt_eval_duration': end_time - prompt_start, 'eval_count': eval_count, 'eval_duration': end_time - prompt_start}
        logger.info(f'returning: {response}')
        return jsonify(response)
    except Exception as e:
        logger.error(f'Error in RAG pipeline: {e}', exc_info=True)
        error_response = {'model': config.model_name, 'created_at': datetime.now(timezone.utc).isoformat(), 'done': True, 'done_reason': 'error', 'error': 'An internal error has occurred. Please try again later.'}
        return jsonify(error_response)

