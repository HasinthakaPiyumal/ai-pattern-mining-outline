# Cluster 11

def generate():
    while True:
        try:
            data = q.get(timeout=120)
            if data:
                yield data
            if '"done": true' in data:
                logger.info('Streaming completed.')
                break
        except queue.Empty:
            yield (json.dumps({}) + '\n')
            logger.warning('Queue timeout. Sending heartbeat.')
        except Exception as e:
            logger.error(f'Streaming error: {e}')
            error_data = format_stream_response(config, 'Streaming error occurred.', done=True, done_reason='error')
            yield (json.dumps(error_data) + '\n')
            break

def handle_streaming_response(config: QueryPipelineConfig, query: str, conversation: List[Dict[str, str]], format_schema: Optional[Dict[str, Any]]=None, options: Optional[Dict[str, Any]]=None) -> Response:
    q = queue.Queue()
    start_time = time.time_ns()
    prompt_start = None

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
    rag = RAGQueryPipeline(config=config, streaming_callback=streaming_callback)

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
    thread = threading.Thread(target=run_rag, daemon=True)
    thread.start()

    def generate():
        while True:
            try:
                data = q.get(timeout=120)
                if data:
                    yield data
                if '"done": true' in data:
                    logger.info('Streaming completed.')
                    break
            except queue.Empty:
                yield (json.dumps({}) + '\n')
                logger.warning('Queue timeout. Sending heartbeat.')
            except Exception as e:
                logger.error(f'Streaming error: {e}')
                error_data = format_stream_response(config, 'Streaming error occurred.', done=True, done_reason='error')
                yield (json.dumps(error_data) + '\n')
                break
    return Response(stream_with_context(generate()), mimetype='application/x-ndjson', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no', 'Connection': 'keep-alive'})

class OllamaProxy:
    """
    A proxy class for interacting with the Ollama API.

    This class provides methods for all Ollama API endpoints, handling both streaming
    and non-streaming responses, and managing various model operations.
    Ref: https://github.com/ollama/ollama/blob/main/docs/api.md
    """

    def __init__(self, base_url: Optional[str]=None):
        """
        Initialize the OllamaProxy with a base URL.

        Args:
            base_url: The base URL for the Ollama API. Defaults to environment variable
                     OLLAMA_URL or 'http://localhost:11434'
        """
        self.base_url = base_url or os.getenv('OLLAMA_URL', 'http://localhost:11434')

    def _proxy_request(self, path: str, method: str='GET', stream: bool=False) -> Response:
        """
        Make a proxied request to the Ollama API.

        Args:
            path: The API endpoint path
            method: The HTTP method to use
            stream: Whether to stream the response

        Returns:
            A Flask Response object
        """
        url = f'{self.base_url}{path}'
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ['host', 'transfer-encoding']}
        data = request.get_data() if method != 'GET' else None
        try:
            response = requests.request(method=method, url=url, headers=headers, data=data, stream=stream)
            if stream:
                return self._handle_streaming_response(response)
            return self._handle_standard_response(response)
        except Exception as e:
            logger.error(f'Error proxying request to Ollama: {str(e)}', exc_info=True)
            return Response(json.dumps({'error': 'An internal error has occurred.'}), status=500, mimetype='application/json')

    def _handle_streaming_response(self, response: requests.Response) -> Response:
        """Handle streaming responses from the Ollama API."""

        def generate():
            try:
                for chunk in response.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
            except Exception as e:
                logger.error(f'Error streaming response: {str(e)}', exc_info=True)
                yield json.dumps({'error': 'An internal error has occurred.'}).encode()
        response_headers = {'Content-Type': response.headers.get('Content-Type', 'application/json')}
        return Response(stream_with_context(generate()), status=response.status_code, headers=response_headers)

    def _handle_standard_response(self, response: requests.Response) -> Response:
        """Handle non-streaming responses from the Ollama API."""
        return Response(response.content, status=response.status_code, headers={'Content-Type': response.headers.get('Content-Type', 'application/json')})

    def generate(self) -> Response:
        """Generate a completion for a given prompt."""
        return self._proxy_request('/api/generate', 'POST', stream=True)

    def chat(self) -> Response:
        """Generate the next message in a chat conversation."""
        return self._proxy_request('/api/chat', 'POST', stream=True)

    def embeddings(self) -> Response:
        """Generate embeddings (legacy endpoint)."""
        return self._proxy_request('/api/embeddings', 'POST')

    def embed(self) -> Response:
        """Generate embeddings from a model."""
        return self._proxy_request('/api/embed', 'POST')

    def create(self) -> Response:
        """Create a model."""
        return self._proxy_request('/api/create', 'POST', stream=True)

    def show(self) -> Response:
        """Show model information."""
        return self._proxy_request('/api/show', 'POST')

    def copy(self) -> Response:
        """Copy a model."""
        return self._proxy_request('/api/copy', 'POST')

    def delete(self) -> Response:
        """Delete a model."""
        return self._proxy_request('/api/delete', 'DELETE')

    def pull(self) -> Response:
        """Pull a model from the Ollama library."""
        return self._proxy_request('/api/pull', 'POST', stream=True)

    def push(self) -> Response:
        """Push a model to the Ollama library."""
        return self._proxy_request('/api/push', 'POST', stream=True)

    def check_blob(self, digest: str) -> Response:
        """Check if a blob exists."""
        return self._proxy_request(f'/api/blobs/{digest}', 'HEAD')

    def push_blob(self, digest: str) -> Response:
        """Push a blob to the server."""
        return self._proxy_request(f'/api/blobs/{digest}', 'POST')

    def list_local_models(self) -> Response:
        """List models available locally."""
        return self._proxy_request('/api/tags', 'GET')

    def list_running_models(self) -> Response:
        """List models currently loaded in memory."""
        return self._proxy_request('/api/ps', 'GET')

    def version(self) -> Response:
        """Get the Ollama version."""
        return self._proxy_request('/api/version', 'GET')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return (jsonify({'error': 'Invalid JSON payload', 'done': True, 'done_reason': 'error'}), 400)
        session_id = session.get('session_id')
        abort_flag = session_manager.get_abort_flag(session_id)
        session_manager.reset_abort_flag(session_id)
        if data.get('stream', True):
            api_response = make_api_request('/api/chat', data, stream=True)

            def generate():
                try:
                    for chunk in api_response.iter_lines():
                        if abort_flag.is_set():
                            logger.info(f'Aborting stream for session {session_id[:8]}...')
                            api_response.close()
                            yield 'data: {"type": "abort", "content": "Request aborted"}\n\n'
                            break
                        if chunk:
                            yield f'data: {chunk.decode()}\n\n'
                except Exception as e:
                    logger.error(f'Stream error: {str(e)}')
                    yield f'data: {{"error": "{str(e)}", "done": true}}\n\n'
            return Response(stream_with_context(generate()), mimetype='application/x-ndjson', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no', 'Connection': 'keep-alive'})
        else:
            logger.info('Processing non-streaming request')
            response = make_api_request('/api/chat', data)
            return response.json()
    except (ConnectionError, Timeout):
        return (jsonify({'error': 'Connection error', 'done': True, 'done_reason': 'error'}), 503)
    except RequestException as e:
        status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 500
        logger.error(f'RequestException: {str(e)}')
        return (jsonify({'error': 'An internal error has occurred', 'done': True, 'done_reason': 'error'}), status_code)

def make_api_request(endpoint: str, data: Dict, stream: bool=False) -> Any:
    api_url = os.getenv('API_URL', 'http://localhost:8000')
    headers = {'Content-Type': 'application/json', 'X-API-Key': os.getenv('API_KEY', 'EXAMPLE_API_KEY')}
    try:
        response = requests.post(f'{api_url}{endpoint}', headers=headers, json=data, stream=stream, timeout=120)
        response.raise_for_status()
        return response
    except (ConnectionError, Timeout) as e:
        logger.error(f'Connection error: {str(e)}')
        raise
    except RequestException as e:
        logger.error(f'Request error: {str(e)}')
        raise

