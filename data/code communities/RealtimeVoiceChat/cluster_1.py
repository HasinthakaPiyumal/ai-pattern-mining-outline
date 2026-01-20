# Cluster 1

class LLM:
    """
    Provides a unified interface for interacting with various LLM backends.

    Supports Ollama (via direct HTTP), OpenAI API, and LMStudio (via OpenAI-compatible API).
    Handles client initialization, streaming generation, request cancellation,
    system prompts, and basic connection management including an optional `ollama ps` check.
    """
    SUPPORTED_BACKENDS = ['ollama', 'openai', 'lmstudio']

    def __init__(self, backend: str, model: str, system_prompt: Optional[str]=None, api_key: Optional[str]=None, base_url: Optional[str]=None, no_think: bool=False):
        """
        Initializes the LLM interface for a specific backend and model.

        Args:
            backend: The name of the LLM backend to use (e.g., "ollama", "openai", "lmstudio").
            model: The identifier for the specific model to use within the backend.
            system_prompt: An optional system prompt to prepend to conversations.
            api_key: API key, primarily for OpenAI backend (can be omitted for others if not needed).
            base_url: Optional base URL for the backend API (overrides defaults/env vars).
            no_think: Experimental flag (currently unused in core logic, intended for future prompt modification).

        Raises:
            ValueError: If an unsupported backend is specified.
            ImportError: If required libraries for the selected backend are not installed.
        """
        logger.info(f'🤖⚙️ Initializing LLM with backend: {backend}, model: {model}, system_prompt: {system_prompt}')
        self.backend = backend.lower()
        if self.backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend '{backend}'. Supported: {self.SUPPORTED_BACKENDS}")
        if self.backend == 'ollama' and (not REQUESTS_AVAILABLE):
            raise ImportError("requests library is required for the 'ollama' backend but not installed.")
        if self.backend in ['openai', 'lmstudio'] and (not OPENAI_AVAILABLE):
            raise ImportError("openai library is required for the 'openai'/'lmstudio' backends but not installed.")
        self.model = model
        self.system_prompt = system_prompt
        self._api_key = api_key
        self._base_url = base_url
        self.no_think = no_think
        self.client: Optional[OpenAI] = None
        self.ollama_session: Optional[Session] = None
        self._client_initialized: bool = False
        self._client_init_lock = Lock()
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._requests_lock = Lock()
        self._ollama_connection_ok: bool = False
        logger.info(f"🤖⚙️ Configuring LLM instance: backend='{self.backend}', model='{self.model}'")
        self.effective_openai_key = self._api_key or OPENAI_API_KEY
        self.effective_ollama_url = self._base_url or OLLAMA_BASE_URL if self.backend == 'ollama' else None
        self.effective_lmstudio_url = self._base_url or LMSTUDIO_BASE_URL if self.backend == 'lmstudio' else None
        self.effective_openai_base_url = self._base_url if self.backend == 'openai' and self._base_url else None
        if self.backend == 'ollama' and self.effective_ollama_url:
            url = self.effective_ollama_url
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            url = url.replace('/api/chat', '').replace('/api/generate', '').rstrip('/')
            self.effective_ollama_url = url
            logger.debug(f'🤖⚙️ Normalized Ollama URL: {self.effective_ollama_url}')
        if self.backend == 'ollama' and REQUESTS_AVAILABLE:
            self.ollama_session = requests.Session()
            logger.info('🤖🔌 Initialized requests.Session for Ollama backend.')
        self.system_prompt_message = None
        if self.system_prompt:
            self.system_prompt_message = {'role': 'system', 'content': self.system_prompt}
            logger.info(f'🤖💬 System prompt set.')

    def _lazy_initialize_clients(self) -> bool:
        """
        Initializes backend clients or checks connections on first use (thread-safe).

        Creates the appropriate HTTP client (OpenAI SDK or requests.Session) and performs
        an initial connection check for Ollama. If the Ollama check fails, optionally
        attempts to run `ollama ps` as a fallback before retrying the connection check.

        Returns:
            True if the client is initialized and ready (or connection check passed for Ollama),
            False otherwise.
        """
        if self._client_initialized:
            if self.backend in ['openai', 'lmstudio']:
                return self.client is not None
            if self.backend == 'ollama':
                return self.ollama_session is not None and self._ollama_connection_ok
            return False
        with self._client_init_lock:
            if self._client_initialized:
                if self.backend in ['openai', 'lmstudio']:
                    return self.client is not None
                if self.backend == 'ollama':
                    return self.ollama_session is not None and self._ollama_connection_ok
                return False
            logger.debug(f'🤖🔄 Lazy initializing/checking connection for backend: {self.backend}')
            init_ok = False
            self._ollama_connection_ok = False
            try:
                if self.backend == 'openai':
                    self.client = _create_openai_client(self.effective_openai_key, base_url=self.effective_openai_base_url)
                    init_ok = self.client is not None
                elif self.backend == 'lmstudio':
                    self.client = _create_openai_client(api_key='lmstudio-key', base_url=self.effective_lmstudio_url)
                    init_ok = self.client is not None
                elif self.backend == 'ollama':
                    if self.ollama_session and self.effective_ollama_url:
                        initial_check_ok = _check_ollama_connection(self.effective_ollama_url, self.ollama_session)
                        if initial_check_ok:
                            init_ok = True
                            self._ollama_connection_ok = True
                        else:
                            logger.warning(f"🤖🔌 Initial Ollama connection check failed for {self.effective_ollama_url}. Attempting 'ollama ps' fallback.")
                            if _run_ollama_ps():
                                logger.info("🤖⏳ 'ollama ps' succeeded, waiting 3 seconds before re-checking connection...")
                                time.sleep(3)
                                second_check_ok = _check_ollama_connection(self.effective_ollama_url, self.ollama_session)
                                if second_check_ok:
                                    logger.info("🤖🔌✅ Ollama connection successful after running 'ollama ps'.")
                                    init_ok = True
                                    self._ollama_connection_ok = True
                                else:
                                    logger.error(f"🤖💥 Ollama connection check still failed after running 'ollama ps'.")
                                    init_ok = False
                            else:
                                logger.error(f"🤖💥 'ollama ps' command failed or not found. Cannot verify/start server. Initialization failed for {self.effective_ollama_url}.")
                                init_ok = False
                    else:
                        logger.error('🤖💥 Ollama session object is None or URL not set during lazy init.')
                        init_ok = False
                if init_ok:
                    logger.info(f'🤖✅ Client/Connection initialized successfully for backend: {self.backend}.')
                else:
                    logger.error(f'🤖💥 Initialization failed for backend: {self.backend}.')
            except Exception as e:
                logger.exception(f'🤖💥 Critical failure during lazy initialization for {self.backend}: {e}')
                init_ok = False
            finally:
                self._client_initialized = True
                if self.backend == 'ollama' and (not init_ok):
                    self._ollama_connection_ok = False
            return init_ok

    def cancel_generation(self, request_id: Optional[str]=None) -> bool:
        """
        Requests cancellation of active generation streams.

        If `request_id` is provided, cancels that specific stream.
        If `request_id` is None, attempts to cancel all currently active streams.
        Cancellation involves removing the request from tracking and attempting to
        close the underlying network stream/response object.

        Args:
            request_id: The unique ID of the generation request to cancel, or None to cancel all.

        Returns:
            True if at least one request cancellation was attempted, False otherwise.
        """
        cancelled_any = False
        with self._requests_lock:
            ids_to_cancel = []
            if request_id is None:
                if not self._active_requests:
                    logger.debug('🤖🗑️ Cancel all requested, but no active requests found.')
                    return False
                logger.info(f'🤖🗑️ Attempting to cancel ALL active generation requests ({len(self._active_requests)}).')
                ids_to_cancel = list(self._active_requests.keys())
            else:
                if request_id not in self._active_requests:
                    logger.warning(f"🤖🗑️ Cancel requested for ID '{request_id}', but it's not an active request.")
                    return False
                logger.info(f'🤖🗑️ Attempting to cancel generation request: {request_id}')
                ids_to_cancel.append(request_id)
            for req_id in ids_to_cancel:
                if self._cancel_single_request_unsafe(req_id):
                    cancelled_any = True
        return cancelled_any

    def _cancel_single_request_unsafe(self, request_id: str) -> bool:
        """
        Internal helper to handle cancellation for a single request (thread-unsafe).

        Removes the request data from the `_active_requests` dictionary and attempts
        to call the `close()` method on the associated stream/response object, if available.
        Must be called while holding `_requests_lock`.

        Args:
            request_id: The unique ID of the request to cancel.

        Returns:
            True if the request was found and removal/close attempt was made, False otherwise.
        """
        request_data = self._active_requests.pop(request_id, None)
        if not request_data:
            logger.debug(f'🤖🗑️ Request {request_id} already removed before cancellation attempt.')
            return False
        request_type = request_data.get('type', 'unknown')
        stream_obj = request_data.get('stream')
        logger.debug(f'🤖🗑️ Cancelling request {request_id} (type: {request_type}). Stream object: {type(stream_obj)}')
        if stream_obj:
            try:
                if hasattr(stream_obj, 'close') and callable(stream_obj.close):
                    logger.debug(f'🤖🗑️ [{request_id}] Attempting to close stream/response object...')
                    stream_obj.close()
                    logger.info(f'🤖🗑️ Closed stream/response for cancelled request {request_id}.')
                else:
                    logger.warning(f"🤖⚠️ [{request_id}] Stream object of type {type(stream_obj)} does not have a callable 'close' method. Cannot explicitly close.")
            except Exception as e:
                logger.error(f'🤖💥 Error closing stream/response for request {request_id}: {e}', exc_info=False)
        else:
            logger.warning(f'🤖⚠️ [{request_id}] No stream object found in request data to close.')
        logger.info(f'🤖🗑️ Removed generation request {request_id} from tracking (close attempted).')
        return True

    def _register_request(self, request_id: str, request_type: str, stream_obj: Optional[Any]):
        """
        Registers an active generation stream for cancellation tracking (thread-safe).

        Stores the request ID, type, stream object, and start time internally.

        Args:
            request_id: The unique ID for the generation request.
            request_type: The backend type (e.g., "openai", "ollama").
            stream_obj: The underlying stream/response object associated with the request.
        """
        with self._requests_lock:
            if request_id in self._active_requests:
                logger.warning(f'🤖⚠️ Request ID {request_id} already registered. Overwriting.')
            self._active_requests[request_id] = {'type': request_type, 'stream': stream_obj, 'start_time': time.time()}
            logger.debug(f'🤖ℹ️ Registered active request: {request_id} (Type: {request_type}, Stream: {type(stream_obj)}, Count: {len(self._active_requests)})')

    def cleanup_stale_requests(self, timeout_seconds: int=300):
        """
        Finds and attempts to cancel requests older than the specified timeout.

        Iterates through active requests and calls `cancel_generation` for any
        request whose start time exceeds the timeout duration.

        Args:
            timeout_seconds: The maximum age in seconds before a request is considered stale.

        Returns:
            The number of stale requests for which cancellation was attempted.
        """
        stale_ids = []
        now = time.time()
        with self._requests_lock:
            stale_ids = [req_id for req_id, req_data in self._active_requests.items() if now - req_data.get('start_time', 0) > timeout_seconds]
        if stale_ids:
            logger.info(f'🤖🧹 Found {len(stale_ids)} potentially stale requests (>{timeout_seconds}s). Cleaning up...')
            cleaned_count = 0
            for req_id in stale_ids:
                if self.cancel_generation(req_id):
                    cleaned_count += 1
            logger.info(f'🤖🧹 Cleaned up {cleaned_count}/{len(stale_ids)} stale requests (attempted stream close).')
            return cleaned_count
        return 0

    def prewarm(self, max_retries: int=1) -> bool:
        """
        Attempts to "prewarm" the LLM connection and potentially load the model.

        Runs a simple, short generation task ("Respond with only the word 'OK'.")
        to trigger lazy initialization (including potential `ollama ps` check)
        and ensure the backend is responsive before actual use. Includes basic retry logic.

        Args:
            max_retries: The number of times to retry the generation task if a
                         connection/timeout error occurs (0 means one attempt total).

        Returns:
            True if the prewarm generation completed successfully (even with no content),
            False if initialization or generation failed after retries.
        """
        prompt = "Respond with only the word 'OK'."
        logger.info(f"🤖🔥 Attempting prewarm for '{self.model}' on backend '{self.backend}'...")
        if not self._lazy_initialize_clients():
            logger.error('🤖🔥💥 Prewarm failed: Could not initialize backend client/connection.')
            return False
        attempts = 0
        last_error = None
        while attempts <= max_retries:
            prewarm_start_time = time.time()
            prewarm_request_id = f'prewarm-{self.backend}-{uuid.uuid4()}'
            generator = None
            full_response = ''
            token_count = 0
            first_token_time = None
            try:
                logger.info(f'🤖🔥 Prewarm Attempt {attempts + 1}/{max_retries + 1} calling generate (ID: {prewarm_request_id})...')
                generator = self.generate(text=prompt, history=None, use_system_prompt=True, request_id=prewarm_request_id, temperature=0.1)
                gen_start_time = time.time()
                for token in generator:
                    if first_token_time is None:
                        first_token_time = time.time()
                        logger.info(f'🤖🔥⏱️ Prewarm TTFT: {first_token_time - gen_start_time:.4f}s')
                    full_response += token
                    token_count += 1
                gen_end_time = time.time()
                logger.info(f"🤖🔥ℹ️ Prewarm consumed {token_count} tokens in {gen_end_time - gen_start_time:.4f}s. Full response: '{full_response}'")
                if token_count == 0 and (not full_response):
                    logger.warning(f'🤖🔥⚠️ Prewarm yielded no response content, but generation finished.')
                prewarm_end_time = time.time()
                logger.info(f'🤖🔥✅ Prewarm successful (generation finished naturally). Total time: {prewarm_end_time - prewarm_start_time:.4f}s.')
                return True
            except (APIConnectionError, requests.exceptions.ConnectionError, ConnectionError, TimeoutError, APITimeoutError, requests.exceptions.Timeout) as e:
                last_error = e
                logger.warning(f'🤖🔥⚠️ Prewarm attempt {attempts + 1}/{max_retries + 1} connection/timeout error during generation: {e}')
                if attempts < max_retries:
                    attempts += 1
                    wait_time = 2 * attempts
                    logger.info(f'🤖🔥🔄 Retrying prewarm generation in {wait_time}s...')
                    time.sleep(wait_time)
                    self._client_initialized = False
                    logger.debug('🤖🔥🔄 Resetting client initialized flag to force re-check on retry.')
                    continue
                else:
                    logger.error(f'🤖🔥💥 Prewarm failed permanently after {attempts + 1} generation attempts due to connection issues.')
                    return False
            except (APIError, RateLimitError, requests.exceptions.RequestException, RuntimeError) as e:
                last_error = e
                logger.error(f'🤖🔥💥 Prewarm attempt {attempts + 1}/{max_retries + 1} API/Request/Runtime error: {e}')
                if isinstance(e, ConnectionError) and 'connection failed' in str(e):
                    logger.error('   (This likely indicates the initial lazy initialization failed its connection check or `ollama ps` fallback)')
                elif isinstance(e, RuntimeError) and 'client failed' in str(e):
                    logger.error('   (This might indicate the initial lazy initialization failed)')
                return False
            except Exception as e:
                last_error = e
                logger.exception(f'🤖🔥💥 Prewarm attempt {attempts + 1}/{max_retries + 1} unexpected error.')
                return False
            finally:
                logger.debug(f"🤖🔥ℹ️ [{prewarm_request_id}] Prewarm attempt finished. generate()'s finally handles tracking cleanup.")
                if generator is not None and hasattr(generator, 'close'):
                    try:
                        generator.close()
                    except Exception as close_err:
                        logger.warning(f'🤖🔥⚠️ [{prewarm_request_id}] Error closing generator in prewarm finally: {close_err}', exc_info=False)
                generator = None
            if attempts >= max_retries:
                break
        logger.error(f'🤖🔥💥 Prewarm failed after exhausting retries. Last error: {last_error}')
        return False

    def generate(self, text: str, history: Optional[List[Dict[str, str]]]=None, use_system_prompt: bool=True, request_id: Optional[str]=None, **kwargs: Any) -> Generator[str, None, None]:
        """
        Generates text using the configured backend, yielding tokens as a stream.

        Handles lazy initialization (including potential `ollama ps` check), message formatting,
        backend-specific API calls, stream registration, token yielding, and resource cleanup.

        Args:
            text: The user's input prompt/text.
            history: An optional list of previous messages (dicts with "role" and "content").
            use_system_prompt: If True, prepends the configured system prompt (if any).
            request_id: An optional unique ID for this generation request. If None, one is generated.
            **kwargs: Additional backend-specific keyword arguments (e.g., temperature, top_p, stop sequences).

        Yields:
            str: Individual tokens (or small chunks of text) as they are generated by the LLM.

        Raises:
            RuntimeError: If the backend client fails to initialize.
            ConnectionError: If communication with the backend fails (initial connection or during streaming).
            ValueError: If configuration is invalid (e.g., missing Ollama URL).
            APIError: For backend-specific API errors (OpenAI/LMStudio).
            RateLimitError: For backend-specific rate limit errors (OpenAI/LMStudio).
            requests.exceptions.RequestException: For Ollama HTTP request errors.
            Exception: For other unexpected errors during the generation process.
        """
        if not self._lazy_initialize_clients():
            if self.backend == 'ollama' and (not self._ollama_connection_ok):
                raise ConnectionError(f"LLM backend '{self.backend}' connection failed. Could not connect to {self.effective_ollama_url} even after attempting 'ollama ps'. Check server status and configuration.")
            raise RuntimeError(f"LLM backend '{self.backend}' client failed to initialize.")
        req_id = request_id if request_id else f'{self.backend}-{uuid.uuid4()}'
        logger.info(f'🤖💬 Starting generation (Request ID: {req_id})')
        messages = []
        if use_system_prompt and self.system_prompt_message:
            messages.append(self.system_prompt_message)
        if history:
            messages.extend(history)
        if len(messages) == 0 or messages[-1]['role'] != 'user':
            added_text = text
            if self.no_think:
                added_text = f'{text}/nothink'
            logger.info(f'🧠💬 llm_module.py generate adding role user to messages, content: {added_text}')
            messages.append({'role': 'user', 'content': added_text})
        logger.debug(f'🤖💬 [{req_id}] Prepared messages count: {len(messages)}')
        stream_iterator = None
        stream_object_to_register = None
        try:
            if self.backend == 'openai':
                if self.client is None:
                    raise RuntimeError('OpenAI client not initialized (should have been caught by lazy_init).')
                payload = {'model': self.model, 'messages': messages, 'stream': True, **kwargs}
                logger.info(f'🤖💬 [{req_id}] Sending OpenAI request with payload:')
                logger.info(f'{json.dumps(payload, indent=2)}')
                stream_iterator = self.client.chat.completions.create(model=self.model, messages=messages, stream=True, **kwargs)
                stream_object_to_register = stream_iterator
                self._register_request(req_id, 'openai', stream_object_to_register)
                yield from self._yield_openai_chunks(stream_iterator, req_id)
            elif self.backend == 'lmstudio':
                if self.client is None:
                    raise RuntimeError('LM Studio client not initialized (should have been caught by lazy_init).')
                if 'temperature' not in kwargs:
                    kwargs['temperature'] = 0.7
                payload = {'model': self.model, 'messages': messages, 'stream': True, **kwargs}
                logger.info(f'🤖💬 [{req_id}] Sending LM Studio request with payload:')
                logger.info(f'{json.dumps(payload, indent=2)}')
                stream_iterator = self.client.chat.completions.create(model=self.model, messages=messages, stream=True, **kwargs)
                stream_object_to_register = stream_iterator
                self._register_request(req_id, 'lmstudio', stream_object_to_register)
                yield from self._yield_openai_chunks(stream_iterator, req_id)
            elif self.backend == 'ollama':
                if self.ollama_session is None:
                    raise RuntimeError('Ollama session not initialized (should have been caught by lazy_init).')
                if not self.effective_ollama_url:
                    raise ValueError('Ollama base URL not configured.')
                ollama_api_url = f'{self.effective_ollama_url}/api/chat'
                valid_options = {'temperature', 'top_k', 'top_p', 'num_predict', 'stop'}
                options = {k: v for k, v in kwargs.items() if k in valid_options}
                if 'temperature' not in options:
                    options['temperature'] = 0.7
                payload = {'model': self.model, 'messages': messages, 'stream': True, 'options': options}
                logger.info(f'🤖💬 [{req_id}] Sending Ollama request to {ollama_api_url} with payload:')
                logger.info(f'{json.dumps(payload, indent=2)}')
                response = self.ollama_session.post(ollama_api_url, json=payload, stream=True, timeout=(10.0, 600.0))
                response.raise_for_status()
                stream_object_to_register = response
                self._register_request(req_id, 'ollama', stream_object_to_register)
                yield from self._yield_ollama_chunks(response, req_id)
            else:
                raise ValueError(f"Backend '{self.backend}' generation logic not implemented.")
            logger.info(f'🤖✅ Finished generating stream successfully (request_id: {req_id})')
        except (requests.exceptions.ConnectionError, ConnectionError, APITimeoutError, requests.exceptions.Timeout) as e:
            logger.error(f'🤖💥 Connection/Timeout Error during generation for {req_id}: {e}', exc_info=False)
            raise ConnectionError(f'Communication error during generation: {e}') from e
        except (APIError, RateLimitError, requests.exceptions.RequestException) as e:
            logger.error(f'🤖💥 API/Request Error during generation for {req_id}: {e}', exc_info=False)
            raise
        except Exception as e:
            logger.error(f'🤖💥 Unexpected error in generation pipeline for {req_id}: {e}', exc_info=True)
            raise
        finally:
            logger.debug(f'🤖ℹ️ [{req_id}] Entering finally block for generate.')
            with self._requests_lock:
                if req_id in self._active_requests:
                    logger.debug(f"🤖🗑️ [{req_id}] Removing request from tracking and attempting stream close in generate's finally block.")
                    self._cancel_single_request_unsafe(req_id)
                else:
                    logger.debug(f'🤖🗑️ [{req_id}] Request already removed from tracking before finally block completion.')
            logger.debug(f'🤖ℹ️ [{req_id}] Exiting finally block. Active requests: {len(self._active_requests)}')

    def _yield_openai_chunks(self, stream, request_id: str) -> Generator[str, None, None]:
        """
        Iterates over an OpenAI/LMStudio stream, yielding content chunks.

        Handles extracting content from stream chunks and checks for cancellation
        before processing each chunk. Ensures the stream is closed upon completion,
        error, or cancellation.

        Args:
            stream: The stream object returned by the OpenAI client's `create` method.
            request_id: The unique ID associated with this generation stream.

        Yields:
            str: Content chunks from the stream's delta messages.

        Raises:
            ConnectionError: If a connection error occurs during streaming, unless likely due to cancellation.
            APIError: If an API error occurs during streaming.
            Exception: For other unexpected errors during streaming.
        """
        token_count = 0
        try:
            for chunk in stream:
                with self._requests_lock:
                    if request_id not in self._active_requests:
                        logger.info(f'🤖🗑️ OpenAI/LMStudio stream {request_id} cancelled or finished externally during iteration.')
                        break
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    content = delta.content
                    if content:
                        token_count += 1
                        yield content
            logger.debug(f'🤖✅ [{request_id}] Finished yielding {token_count} OpenAI/LMStudio tokens.')
        except APIConnectionError as e:
            is_cancelled = False
            with self._requests_lock:
                is_cancelled = request_id not in self._active_requests
            if is_cancelled:
                logger.warning(f'🤖⚠️ OpenAI/LMStudio stream connection error likely due to cancellation for {request_id}: {e}')
            else:
                logger.error(f'🤖💥 OpenAI API connection error during streaming ({request_id}): {e}')
                raise ConnectionError(f'OpenAI communication error during streaming: {e}') from e
        except APIError as e:
            logger.error(f'🤖💥 OpenAI API error during streaming ({request_id}): {e}')
            raise
        except Exception as e:
            is_cancelled = False
            with self._requests_lock:
                is_cancelled = request_id not in self._active_requests
            if is_cancelled:
                logger.warning(f'🤖⚠️ OpenAI/LMStudio stream error likely due to cancellation for {request_id}: {e}')
            else:
                logger.error(f'🤖💥 Unexpected error during OpenAI streaming ({request_id}): {e}', exc_info=True)
                raise
        finally:
            if stream and hasattr(stream, 'close') and callable(stream.close):
                try:
                    logger.debug(f'🤖🗑️ [{request_id}] Closing OpenAI stream in _yield_openai_chunks finally.')
                    stream.close()
                except Exception as close_err:
                    logger.warning(f'🤖⚠️ [{request_id}] Error closing OpenAI stream in finally: {close_err}', exc_info=False)

    def _yield_ollama_chunks(self, response: requests.Response, request_id: str) -> Generator[str, None, None]:
        """
        Iterates over an Ollama HTTP response stream, decoding JSON lines and yielding content.

        Handles reading bytes, decoding UTF-8, parsing JSON chunks, extracting message content,
        and checking for the 'done' signal. Checks for cancellation before processing each chunk.
        Ensures the response is closed upon completion, error, or cancellation.

        Args:
            response: The streaming requests.Response object from the Ollama API call.
            request_id: The unique ID associated with this generation stream.

        Yields:
            str: Content chunks from the stream's message objects.

        Raises:
            RuntimeError: If the Ollama stream returns an error message.
            ConnectionError: If a connection error occurs during streaming, unless likely due to cancellation.
            requests.exceptions.RequestException: For other request-related errors during streaming.
            Exception: For JSON decoding errors or other unexpected issues.
        """
        token_count = 0
        buffer = ''
        processed_done = False
        try:
            try:
                for chunk_bytes in response.iter_content(chunk_size=None):
                    with self._requests_lock:
                        if request_id not in self._active_requests:
                            logger.info(f'🤖🗑️ Ollama stream {request_id} cancelled or finished externally during iteration (pre-chunk check).')
                            break
                    if not chunk_bytes:
                        continue
                    buffer += chunk_bytes.decode('utf-8')
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                            if chunk.get('error'):
                                logger.error(f'🤖💥 Ollama stream returned error for {request_id}: {chunk['error']}')
                                raise RuntimeError(f'Ollama stream error: {chunk['error']}')
                            content = chunk.get('message', {}).get('content')
                            if content:
                                token_count += 1
                                yield content
                            if chunk.get('done'):
                                logger.debug(f"🤖✅ [{request_id}] Ollama signalled 'done'.")
                                buffer = ''
                                processed_done = True
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"🤖⚠️ [{request_id}] Failed to decode JSON line: '{line[:100]}...'")
                        except Exception as e:
                            logger.error(f'🤖💥 [{request_id}] Error processing Ollama stream chunk: {e}', exc_info=True)
                            raise
                    if processed_done:
                        break
            except AttributeError as e:
                is_cancelled = False
                with self._requests_lock:
                    is_cancelled = request_id not in self._active_requests
                if "'NoneType' object has no attribute 'read'" in str(e):
                    if is_cancelled:
                        logger.warning(f"🤖⚠️ [{request_id}] Caught AttributeError ('NoneType' has no attribute 'read') during Ollama stream iteration, likely due to concurrent cancellation. Stopping iteration.")
                    else:
                        logger.warning(f"🤖⚠️ [{request_id}] Caught AttributeError ('NoneType' has no attribute 'read') during Ollama stream iteration. Request *might* not be marked cancelled yet, but stopping iteration as stream is likely closed.")
                else:
                    logger.error(f'🤖💥 [{request_id}] Caught unexpected AttributeError during Ollama stream iteration: {e}', exc_info=True)
                    raise e
            if not processed_done:
                with self._requests_lock:
                    if request_id not in self._active_requests:
                        logger.info(f'🤖🗑️ Ollama stream {request_id} processing stopped due to cancellation flag after loop.')
            logger.debug(f'🤖✅ [{request_id}] Finished yielding {token_count} Ollama tokens (processed_done={processed_done}).')
        except requests.exceptions.ChunkedEncodingError as e:
            is_cancelled = False
            with self._requests_lock:
                is_cancelled = request_id not in self._active_requests
            if is_cancelled:
                logger.warning(f'🤖⚠️ Ollama chunked encoding error likely due to cancellation for {request_id}: {e}')
            else:
                logger.error(f'🤖💥 Ollama chunked encoding error during streaming ({request_id}): {e}')
                raise ConnectionError(f'Ollama communication error during streaming: {e}') from e
        except requests.exceptions.RequestException as e:
            is_cancelled = False
            with self._requests_lock:
                is_cancelled = request_id not in self._active_requests
            if is_cancelled:
                logger.warning(f'🤖⚠️ Ollama requests error likely due to cancellation for {request_id}: {e}')
            else:
                logger.error(f'🤖💥 Ollama requests error during streaming ({request_id}): {e}')
                raise ConnectionError(f'Ollama communication error during streaming: {e}') from e
        except Exception as e:
            if not isinstance(e, AttributeError):
                logger.error(f'🤖💥 Unexpected error during Ollama streaming ({request_id}): {e}', exc_info=True)
            raise
        finally:
            if response:
                try:
                    logger.debug(f'🤖🗑️ [{request_id}] Closing Ollama response in _yield_ollama_chunks finally.')
                    response.close()
                except Exception as close_err:
                    logger.warning(f'🤖⚠️ [{request_id}] Error closing Ollama response in finally: {close_err}', exc_info=False)

    def measure_inference_time(self, num_tokens: int=10, **kwargs: Any) -> Optional[float]:
        """
        Measures the time taken to generate a target number of initial tokens.

        Uses a fixed, predefined prompt designed to elicit a somewhat predictable
        response length. Times the generation process from the moment the generator
        is obtained until the target number of tokens is yielded or generation ends.
        Ensures the backend client is initialized first.

        Args:
            num_tokens: The target number of tokens to generate before stopping measurement.
            **kwargs: Additional keyword arguments passed to the `generate` method
                      (e.g., temperature=0.1).

        Returns:
            The time taken in milliseconds to generate the actual number of tokens
            produced (up to `num_tokens`), or None if generation failed, produced 0 tokens,
            or encountered an error during initialization or generation.
        """
        if num_tokens <= 0:
            logger.warning('🤖⏱️ Cannot measure inference time for 0 or negative tokens.')
            return None
        if not self._lazy_initialize_clients():
            logger.error(f'🤖⏱️💥 Measurement failed: Could not initialize backend client/connection for {self.backend}.')
            return None
        measurement_system_prompt = 'You are a precise assistant. Follow instructions exactly.'
        measurement_user_prompt = 'Repeat the following sequence exactly, word for word: one two three four five six seven eight nine ten eleven twelve'
        measurement_history = [{'role': 'system', 'content': measurement_system_prompt}, {'role': 'user', 'content': measurement_user_prompt}]
        req_id = f'measure-{self.backend}-{uuid.uuid4()}'
        logger.info(f'🤖⏱️ Measuring inference time for {num_tokens} tokens (Request ID: {req_id}). Using fixed measurement prompt.')
        logger.debug(f'🤖⏱️ [{req_id}] Measurement history: {measurement_history}')
        token_count = 0
        start_time = None
        end_time = None
        generator = None
        actual_tokens_generated = 0
        try:
            generator = self.generate(text='', history=measurement_history, use_system_prompt=False, request_id=req_id, **kwargs)
            start_time = time.time()
            for token in generator:
                if token_count == 0:
                    pass
                token_count += 1
                if token_count >= num_tokens:
                    end_time = time.time()
                    logger.debug(f'🤖⏱️ [{req_id}] Reached target {num_tokens} tokens.')
                    break
            if end_time is None:
                end_time = time.time()
                logger.debug(f'🤖⏱️ [{req_id}] Generation finished naturally after {token_count} tokens (may be less than requested {num_tokens}).')
            actual_tokens_generated = token_count
        except (ConnectionError, APIError, RuntimeError, Exception) as e:
            logger.error(f'🤖⏱️💥 Error during inference time measurement ({req_id}): {e}', exc_info=False)
            return None
        finally:
            if generator and hasattr(generator, 'close'):
                try:
                    logger.debug(f'🤖⏱️🗑️ [{req_id}] Closing generator in measure_inference_time finally.')
                    generator.close()
                except Exception as close_err:
                    logger.warning(f'🤖⏱️⚠️ [{req_id}] Error closing generator in finally: {close_err}', exc_info=False)
            generator = None
        if start_time is None or end_time is None:
            logger.error(f'🤖⏱️💥 [{req_id}] Measurement failed: Start or end time not recorded.')
            return None
        if actual_tokens_generated == 0:
            logger.warning(f'🤖⏱️⚠️ [{req_id}] Measurement invalid: 0 tokens were generated.')
            return None
        duration_sec = end_time - start_time
        duration_ms = duration_sec * 1000
        logger.info(f"🤖⏱️✅ Measured ~{duration_ms:.2f} ms for {actual_tokens_generated} tokens (target: {num_tokens}) for model '{self.model}' on backend '{self.backend}' using fixed prompt. (Request ID: {req_id})")
        return duration_ms

def _create_openai_client(api_key: Optional[str], base_url: Optional[str]=None) -> OpenAI:
    """
    Creates and configures an OpenAI API client instance.

    Handles API key logic (using a placeholder if none provided for local models)
    and optional base URL configuration. Sets default timeout and retries.

    Args:
        api_key: The OpenAI API key, or None if not required (e.g., for LMStudio).
        base_url: The base URL for the API endpoint (e.g., for LMStudio or custom deployments).

    Returns:
        An initialized OpenAI client instance.

    Raises:
        ImportError: If the 'openai' library is not installed.
        Exception: If client initialization fails for other reasons.
    """
    if not OPENAI_AVAILABLE:
        raise ImportError('openai library is required for this backend but not installed.')
    try:
        effective_key = api_key if api_key else 'no-key-needed'
        client_args = {'api_key': effective_key, 'timeout': 30.0, 'max_retries': 2}
        if base_url:
            client_args['base_url'] = base_url
        client = OpenAI(**client_args)
        logger.info(f'🤖🔌 Prepared OpenAI-compatible client (Base URL: {base_url or 'Default'}).')
        return client
    except Exception as e:
        logger.error(f'🤖💥 Failed to initialize OpenAI client: {e}')
        raise

def _check_ollama_connection(base_url: str, session: Optional[Session]) -> bool:
    """
    Performs a quick HTTP GET request to check connectivity with an Ollama server.

    Uses the provided requests Session and base URL to attempt a connection.
    Logs success or specific connection errors.

    Args:
        base_url: The base URL of the Ollama server (e.g., "http://127.0.0.1:11434").
        session: An active requests.Session object to use for the check.

    Returns:
        True if the connection check is successful (HTTP 2xx status), False otherwise.
    """
    if not REQUESTS_AVAILABLE:
        logger.warning('🤖⚠️ Cannot check Ollama connection: requests library not installed.')
        return False
    if not session:
        logger.warning('🤖⚠️ Cannot check Ollama connection: requests session not provided.')
        return False
    try:
        base_check_url = base_url.rstrip('/')
        if not base_check_url.startswith(('http://', 'https://')):
            base_check_url = 'http://' + base_check_url
        check_endpoint = f'{base_check_url}/'
        logger.debug(f'🤖🔌 Checking Ollama connection via GET to {check_endpoint}...')
        response = session.get(check_endpoint, timeout=5.0)
        response.raise_for_status()
        logger.info(f'🤖🔌 Successfully connected to Ollama server via HTTP at: {base_url}')
        return True
    except requests.exceptions.ConnectionError as e:
        logger.warning(f'🤖🔌❌ Connection Error checking Ollama at {base_url}: {e}')
        return False
    except requests.exceptions.Timeout:
        logger.warning(f'🤖🔌❌ Timeout checking Ollama connection at {base_url}.')
        return False
    except requests.exceptions.RequestException as e:
        logger.warning(f'🤖🔌❌ Error checking Ollama connection at {base_url}: {e}')
        return False
    except Exception as e:
        logger.error(f'🤖💥 Unexpected error during Ollama connection check: {e}')
        return False

def _run_ollama_ps():
    """
    Attempts to run the 'ollama ps' command via subprocess.

    This is used as a potential fallback diagnostic/recovery step if the initial
    HTTP connection check to the Ollama server fails. It assumes the `ollama` CLI
    is installed and in the system PATH.

    Returns:
        True if the command executes successfully (exit code 0), False otherwise
        (command not found, execution error, timeout).
    """
    try:
        logger.info("🤖🩺 Attempting to run 'ollama ps' to check server status...")
        result = subprocess.run(['ollama', 'ps'], check=True, capture_output=True, text=True, timeout=10.0)
        logger.info(f"🤖🩺 'ollama ps' executed successfully. Output:\n{result.stdout.strip()}")
        return True
    except FileNotFoundError:
        logger.error("🤖💥 'ollama ps' command not found. Make sure Ollama is installed and in your PATH.")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"🤖💥 'ollama ps' command failed with exit code {e.returncode}:")
        if e.stderr:
            logger.error(f'   stderr: {e.stderr.strip()}')
        if e.stdout:
            logger.error(f'   stdout: {e.stdout.strip()}')
        return False
    except subprocess.TimeoutExpired:
        logger.error("🤖💥 'ollama ps' command timed out after 10 seconds.")
        return False
    except Exception as e:
        logger.error(f"🤖💥 An unexpected error occurred while running 'ollama ps': {e}")
        return False

