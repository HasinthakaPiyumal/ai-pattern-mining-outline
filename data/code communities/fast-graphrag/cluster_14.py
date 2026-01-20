# Cluster 14

@dataclass
class VoyageAIEmbeddingService(BaseEmbeddingService):
    """Base class for VoyageAI embeddings implementations."""
    embedding_dim: int = field(default=1024)
    max_elements_per_request: int = field(default=128)
    model: Optional[str] = field(default='voyage-3')
    api_version: Optional[str] = field(default=None)
    max_requests_concurrent: int = field(default=int(os.getenv('CONCURRENT_TASK_LIMIT', 1024)))
    max_requests_per_minute: int = field(default=1800)
    max_requests_per_second: int = field(default=100)
    rate_limit_per_second: bool = field(default=False)

    def __post_init__(self):
        self.embedding_max_requests_concurrent = asyncio.Semaphore(self.max_requests_concurrent) if self.rate_limit_concurrency else NoopAsyncContextManager()
        self.embedding_per_minute_limiter = AsyncLimiter(self.max_requests_per_minute, 60) if self.rate_limit_per_minute else NoopAsyncContextManager()
        self.embedding_per_second_limiter = AsyncLimiter(self.max_requests_per_second, 1) if self.rate_limit_per_second else NoopAsyncContextManager()
        self.embedding_async_client: client_async.AsyncClient = client_async.AsyncClient(api_key=self.api_key, max_retries=4)
        logger.debug('Initialized VoyageAIEmbeddingService.')

    async def encode(self, texts: list[str], model: Optional[str]=None) -> np.ndarray[Any, np.dtype[np.float32]]:
        try:
            'Get the embedding representation of the input text.\n\n            Args:\n                texts (str): The input text to embed.\n                model (str, optional): The name of the model to use. Defaults to the model provided in the config.\n\n            Returns:\n                list[float]: The embedding vector as a list of floats.\n            '
            logger.debug(f'Getting embedding for texts: {texts}')
            model = model or self.model
            if model is None:
                raise ValueError('Model name must be provided.')
            batched_texts = [texts[i * self.max_elements_per_request:(i + 1) * self.max_elements_per_request] for i in range((len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request)]
            response = await asyncio.gather(*[self._embedding_request(b, model) for b in batched_texts])
            data = chain(*[r.embeddings for r in response])
            embeddings = np.array(list(data))
            logger.debug(f'Received embedding response: {len(embeddings)} embeddings')
            return embeddings
        except Exception:
            logger.exception('An error occurred:', exc_info=True)
            raise

    async def _embedding_request(self, input: List[str], model: str) -> EmbeddingsObject:
        async with self.embedding_max_requests_concurrent:
            async with self.embedding_per_minute_limiter:
                async with self.embedding_per_second_limiter:
                    return await self.embedding_async_client.embed(model=model, texts=input, output_dimension=self.embedding_dim)

@dataclass
class OpenAILLMService(BaseLLMService):
    """LLM Service for OpenAI LLMs."""
    model: str = field(default='gpt-4o-mini')
    mode: instructor.Mode = field(default=instructor.Mode.JSON)
    client: Literal['openai', 'azure'] = field(default='openai')
    api_version: Optional[str] = field(default=None)

    def __post_init__(self):
        self.encoding = None
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except Exception as e:
            logger.info(f"LLM: failed to load tokenizer for model '{self.model}' ({e}). Falling back to naive tokenization.")
            self.encoding = None
        self.llm_max_requests_concurrent = asyncio.Semaphore(self.max_requests_concurrent) if self.rate_limit_concurrency else NoopAsyncContextManager()
        self.llm_per_minute_limiter = AsyncLimiter(self.max_requests_per_minute, 60) if self.rate_limit_per_minute else NoopAsyncContextManager()
        self.llm_per_second_limiter = AsyncLimiter(self.max_requests_per_second, 1) if self.rate_limit_per_second else NoopAsyncContextManager()
        if self.client == 'azure':
            assert self.base_url is not None and self.api_version is not None, 'Azure OpenAI requires a base url and an api version.'
            self.llm_async_client = instructor.from_openai(AsyncAzureOpenAI(azure_endpoint=self.base_url, api_key=self.api_key, api_version=self.api_version, timeout=TIMEOUT_SECONDS), mode=self.mode)
        elif self.client == 'openai':
            self.llm_async_client = instructor.from_openai(AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=TIMEOUT_SECONDS), mode=self.mode)
        else:
            raise ValueError("Invalid client type. Must be 'openai' or 'azure'")
        logger.debug('Initialized OpenAILLMService with patched OpenAI client.')

    def count_tokens(self, text: str) -> int:
        """Returns the number of tokens for a given text using the encoding appropriate for the model."""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            return len(TOKEN_PATTERN.findall(text))

    async def send_message(self, prompt: str, system_prompt: str | None=None, history_messages: list[dict[str, str]] | None=None, response_model: Type[T_model] | None=None, **kwargs: Any) -> Tuple[T_model, list[dict[str, str]]]:
        """Send a message to the language model and receive a response.

    Args:
        prompt (str): The input message to send to the language model.
        model (str): The name of the model to use. Defaults to the model provided in the config.
        system_prompt (str, optional): The system prompt to set the context for the conversation. Defaults to None.
        history_messages (list, optional): A list of previous messages in the conversation. Defaults to empty.
        response_model (Type[T], optional): The Pydantic model to parse the response. Defaults to None.
        **kwargs: Additional keyword arguments that may be required by specific LLM implementations.

    Returns:
        str: The response from the language model.
    """
        async with self.llm_max_requests_concurrent:
            async with self.llm_per_minute_limiter:
                async with self.llm_per_second_limiter:
                    try:
                        logger.debug(f'Sending message with prompt: {prompt}')
                        model = self.model
                        messages: list[dict[str, str]] = []
                        if system_prompt:
                            messages.append({'role': 'system', 'content': system_prompt})
                            logger.debug(f'Added system prompt: {system_prompt}')
                        if history_messages:
                            messages.extend(history_messages)
                            logger.debug(f'Added history messages: {history_messages}')
                        messages.append({'role': 'user', 'content': prompt})
                        llm_response: T_model = await self.llm_async_client.chat.completions.create(model=model, messages=messages, response_model=response_model.Model if response_model and issubclass(response_model, BaseModelAlias) else response_model, **kwargs, max_retries=AsyncRetrying(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)))
                        if not llm_response:
                            logger.error('No response received from the language model.')
                            raise LLMServiceNoResponseError('No response received from the language model.')
                        messages.append({'role': 'assistant', 'content': llm_response.model_dump_json() if isinstance(llm_response, BaseModel) else str(llm_response)})
                        logger.debug(f'Received response: {llm_response}')
                        if response_model and issubclass(response_model, BaseModelAlias):
                            llm_response = cast(T_model, cast(BaseModelAlias.Model, llm_response).to_dataclass(llm_response))
                        return (llm_response, messages)
                    except Exception:
                        logger.exception('An error occurred:', exc_info=True)
                        raise

@dataclass
class OpenAIEmbeddingService(BaseEmbeddingService):
    """Base class for Language Model implementations."""
    embedding_dim: int = field(default=1536)
    max_elements_per_request: int = field(default=32)
    model: Optional[str] = field(default='text-embedding-3-small')
    client: Literal['openai', 'azure'] = field(default='openai')
    api_version: Optional[str] = field(default=None)

    def __post_init__(self):
        self.embedding_max_requests_concurrent = asyncio.Semaphore(self.max_requests_concurrent) if self.rate_limit_concurrency else NoopAsyncContextManager()
        self.embedding_per_minute_limiter = AsyncLimiter(self.max_requests_per_minute, 60) if self.rate_limit_per_minute else NoopAsyncContextManager()
        self.embedding_per_second_limiter = AsyncLimiter(self.max_requests_per_second, 1) if self.rate_limit_per_second else NoopAsyncContextManager()
        if self.client == 'azure':
            assert self.base_url is not None and self.api_version is not None, 'Azure OpenAI requires a base url and an api version.'
            self.embedding_async_client = AsyncAzureOpenAI(azure_endpoint=self.base_url, api_key=self.api_key, api_version=self.api_version)
        elif self.client == 'openai':
            self.embedding_async_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            raise ValueError("Invalid client type. Must be 'openai' or 'azure'")
        logger.debug('Initialized OpenAIEmbeddingService with OpenAI client.')

    async def encode(self, texts: list[str], model: Optional[str]=None) -> np.ndarray[Any, np.dtype[np.float32]]:
        try:
            'Get the embedding representation of the input text.\n\n            Args:\n                texts (str): The input text to embed.\n                model (str, optional): The name of the model to use. Defaults to the model provided in the config.\n\n            Returns:\n                list[float]: The embedding vector as a list of floats.\n            '
            logger.debug(f'Getting embedding for texts: {texts}')
            model = model or self.model
            if model is None:
                raise ValueError('Model name must be provided.')
            batched_texts = [texts[i * self.max_elements_per_request:(i + 1) * self.max_elements_per_request] for i in range((len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request)]
            response = await asyncio.gather(*[self._embedding_request(b, model) for b in batched_texts])
            data = chain(*[r.data for r in response])
            embeddings = np.array([dp.embedding for dp in data])
            logger.debug(f'Received embedding response: {len(embeddings)} embeddings')
            return embeddings
        except Exception:
            logger.exception('An error occurred:', exc_info=True)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type((RateLimitError, APIConnectionError, TimeoutError)))
    async def _embedding_request(self, input: List[str], model: str) -> Any:
        async with self.embedding_max_requests_concurrent:
            async with self.embedding_per_minute_limiter:
                async with self.embedding_per_second_limiter:
                    return await self.embedding_async_client.embeddings.create(model=model, input=input, dimensions=self.embedding_dim, encoding_format='float')

@dataclass
class GeminiLLMService(BaseLLMService):
    model: str = field(default='gemini-2.0-flash')
    mode: instructor.Mode = field(default=instructor.Mode.JSON)
    client: Literal['gemini', 'vertex'] = field(default='gemini')
    api_key: Optional[str] = field(default=None)
    temperature: float = field(default=0.7)
    candidate_count: int = field(default=1)
    max_requests_concurrent: int = field(default=int(os.getenv('CONCURRENT_TASK_LIMIT', 1024)))
    max_requests_per_minute: int = field(default=2000)
    max_requests_per_second: int = field(default=500)
    project_id: Optional[str] = field(default=None)
    location: Optional[str] = field(default=None)
    safety_settings: list[types.SafetySetting] = field(default_factory=default_safety_settings)

    def __post_init__(self):
        """Post-initialization.

    • Sets up concurrency semaphores and rate limiters based on the provided configuration.
    • Instantiates the appropriate asynchronous LLM client for either Vertex or Gemini.
    • Initializes a local tokenizer for the Gemini model.
    """
        self.llm_max_requests_concurrent = asyncio.Semaphore(self.max_requests_concurrent) if self.rate_limit_concurrency else NoopAsyncContextManager()
        self.llm_per_minute_limiter = AsyncLimiter(self.max_requests_per_minute, 60) if self.rate_limit_per_minute else NoopAsyncContextManager()
        self.llm_per_second_limiter = AsyncLimiter(self.max_requests_per_second, 1) if self.rate_limit_per_second else NoopAsyncContextManager()
        if self.client == 'vertex':
            assert self.project_id is not None and self.location is not None and (self.api_key is None) or (self.project_id is None and self.location is None and (self.api_key is not None)), 'Azure OpenAI requires a project id and location, or an express API key.'
            if self.api_key is not None:
                self.llm_async_client: genai.Client = genai.Client(vertexai=True, api_key=self.api_key)
            else:
                self.llm_async_client = genai.Client(vertexai=True, project=self.project_id, location=self.location)
        elif self.client == 'gemini':
            self.llm_async_client: genai.Client = genai.Client(api_key=self.api_key)
        else:
            raise ValueError("Invalid client type. Must be 'openai' or 'azure'")
        self.tokenizer = get_tokenizer_for_model('gemini-1.5-flash-002')
        logger.debug('Initialized GeminiLLMService.')

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the provided text utilizing the local Gemini tokenizer.

    Args:
        text (str): The input text whose tokens are to be counted.
        model (Optional[str]): An optional model override (not used in current implementation).
        **kwargs: Additional keyword arguments (currently unused).

    Returns:
        int: Total token count.
    """
        return self.tokenizer.count_tokens(contents=text).total_tokens

    @retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=1, min=4, max=60), retry=retry_if_exception_type((TimeoutError, Exception)))
    async def send_message(self, prompt: str, system_prompt: str | None=None, history_messages: list[dict[str, str]] | None=None, response_model: Type[T_model] | None=None, **kwargs: Any) -> Tuple[T_model, list[dict[str, str]]]:
        """Sends a message to the Gemini AI language model and handles.

      • Concurrency and rate limiting.
      • Request retries with inner attempt loops.
      • Response validation/parsing including JSON repair.

    Args:
        prompt (str): The main user input.
        model (Optional[str]): Optional override for the model name.
        system_prompt (Optional[str]): Optional system-level instructions.
        history_messages (Optional[list[dict[str, str]]]): Prior conversation messages.
        response_model (Optional[Type[T_model]]): Pydantic model (or alias) dictating the response structure.
        temperature (float): Generation temperature setting.
        **kwargs: Additional generation parameters (unused here).

    Returns:
        Tuple[T_model, list[dict[str, str]]]: A tuple containing the parsed response and the updated message history.

    Raises:
        ValueError: If the model name is missing.
        LLMServiceNoResponseError: If no valid response is obtained after all retries.
        errors.APIError: For unrecoverable API errors.
    """
        async with self.llm_max_requests_concurrent:
            async with self.llm_per_minute_limiter:
                async with self.llm_per_second_limiter:
                    model = self.model
                    messages: List[Dict[str, str]] = []
                    if history_messages:
                        messages.extend(history_messages)
                    messages.append({'role': 'user', 'content': prompt})
                    combined_prompt = '\n'.join([f'{msg['role']}: {msg['content']}' for msg in messages])
                    try:

                        def validate_generate_content(response: Any, attempt: int, max_attempts: int) -> bool:
                            if not response or not getattr(response, 'text', ''):
                                return False
                            if response_model is not None and attempt != max_attempts - 1:
                                if not getattr(response, 'parsed', ''):
                                    return False
                            return True
                        generate_config = types.GenerateContentConfig(system_instruction=system_prompt, response_mime_type='application/json', response_schema=response_model.Model if issubclass(response_model, BaseModelAlias) else response_model, candidate_count=self.candidate_count, temperature=self.temperature, safety_settings=self.safety_settings) if response_model else types.GenerateContentConfig(system_instruction=system_prompt, candidate_count=self.candidate_count, temperature=self.temperature, safety_settings=self.safety_settings)
                        response = await _execute_with_inner_retries(operation=lambda: self.llm_async_client.aio.models.generate_content(model=model, contents=combined_prompt, config=generate_config), validate=validate_generate_content, max_attempts=4, short_sleep=0.01, error_sleep=0.2)
                        if not response or not getattr(response, 'text', ''):
                            raise LLMServiceNoResponseError('Failed to obtain a valid response for content.')
                        try:
                            if response_model:
                                if response.parsed:
                                    if issubclass(response_model, BaseModelAlias):
                                        llm_response = TypeAdapter(response_model.Model).validate_python(response.parsed)
                                    else:
                                        llm_response = TypeAdapter(response_model).validate_python(response.parsed)
                                else:
                                    fixed_json = cast(str, repair_json(response.parsed))
                                    if issubclass(response_model, BaseModelAlias):
                                        llm_response = TypeAdapter(response_model.Model).validate_json(fixed_json)
                                    else:
                                        llm_response = TypeAdapter(response_model).validate_json(fixed_json)
                            else:
                                llm_response = response.text
                        except ValidationError as e:
                            raise LLMServiceNoResponseError(f'Invalid JSON response: {str(e)}') from e
                        messages.append({'role': 'model', 'content': llm_response.model_dump_json() if isinstance(llm_response, BaseModel) else str(llm_response)})
                        if response_model and issubclass(response_model, BaseModelAlias):
                            llm_response = cast(T_model, cast(BaseModelAlias.Model, llm_response).to_dataclass(llm_response))
                        return (cast(T_model, llm_response), messages)
                    except errors.APIError as e:
                        if e.code == 429 or (e.details and e.details.get('code') == 429):
                            logger.warning(f'Rate limit error encountered: {e.code} - {e.message}. Attempting retry.')
                            raise
                        elif e.code in (400, 403, 404):
                            logger.error(f'Client error encountered: {e.code} - {e.message}. Check your request parameters or API key.')
                            raise
                        elif e.code in (500, 503, 504):
                            logger.error(f'Server error encountered: {e.code} - {e.message}. Consider retrying after a short delay.')
                            raise
                        else:
                            logger.exception(f'Unexpected API error encountered: {e.code} - {e.message}')
                            raise
                    except Exception as e:
                        logger.exception(f'Unexpected error: {e}')
                        raise

@dataclass
class GeminiEmbeddingService(BaseEmbeddingService):
    """Service implementation to retrieve embeddings for texts using the Gemini model."""
    embedding_dim: int = field(default=768)
    max_elements_per_request: int = field(default=99)
    model: Optional[str] = field(default='text-embedding-004')
    api_version: Optional[str] = field(default=None)
    max_requests_concurrent: int = field(default=int(os.getenv('CONCURRENT_TASK_LIMIT', 150)))
    max_requests_per_minute: int = field(default=80)
    max_requests_per_second: int = field(default=20)

    def __post_init__(self):
        """Post-initialization.

    • Sets up concurrency semaphores and rate limiters for embedding requests.
    • Instantiates the asynchronous client for embedding requests.
    """
        self.embedding_max_requests_concurrent = asyncio.Semaphore(self.max_requests_concurrent) if self.rate_limit_concurrency else NoopAsyncContextManager()
        self.embedding_per_minute_limiter = AsyncLimiter(self.max_requests_per_minute, 60) if self.rate_limit_per_minute else NoopAsyncContextManager()
        self.embedding_per_second_limiter = AsyncLimiter(self.max_requests_per_second, 1) if self.rate_limit_per_second else NoopAsyncContextManager()
        self.embedding_async_client: genai.Client = genai.Client(api_key=self.api_key)
        logger.debug('Initialized GeminiEmbeddingService.')

    async def encode(self, texts: list[str], model: Optional[str]=None) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Obtain embedding vectors for provided input texts.

    This method internally splits the texts into batches (based on max_elements_per_request)
    and sends concurrent requests for embedding. The responses are then concatenated into a single numpy array.

    Args:
        texts (list[str]): List of input texts to be embedded.
        model (Optional[str]): Optional model override; defaults to the service's model if not provided.

    Returns:
        np.ndarray: Array of embedding vectors.

    Raises:
        Exception: Propagates any exception encountered during embedding requests.
    """
        try:
            logger.debug(f'Getting embedding for texts: {texts}')
            model = model or self.model
            if model is None:
                raise ValueError('Model name must be provided.')
            batched_texts = [texts[i * self.max_elements_per_request:(i + 1) * self.max_elements_per_request] for i in range((len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request)]
            response = await asyncio.gather(*[self._embedding_request(batch, model) for batch in batched_texts])
            data = chain(*list(response))
            embeddings = np.array([dp.values for dp in data])
            logger.debug(f'Received embedding response: {len(embeddings)} embeddings')
            return embeddings
        except Exception:
            logger.exception('An error occurred during embedding encoding:', exc_info=True)
            raise

    @retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=1, min=5, max=60), retry=retry_if_exception_type((TimeoutError, Exception)))
    async def _embedding_request(self, input: list[Any], model: str) -> list[types.ContentEmbedding]:
        """Makes an embedding request for a batch of input texts.

    Applies internal retry logic and rate limiting to ensure a valid response is obtained.

    Args:
        input (list[Any]): A batch of texts to be embedded.
        model (str): The model name to be used for generating embeddings.

    Returns:
        list[types.ContentEmbedding]: A list of embedding objects.

    Raises:
        LLMServiceNoResponseError: If a valid response is not obtained after retries.
        errors.APIError: For unrecoverable API errors.
    """
        async with self.embedding_max_requests_concurrent:
            async with self.embedding_per_minute_limiter:
                async with self.embedding_per_second_limiter:
                    try:

                        def validate_embedding_response(response: Any, attempt: int, max_attempts: int) -> bool:
                            if not response or not getattr(response, 'embeddings', None) or response.embeddings == []:
                                return False
                            return True
                        response = await _execute_with_inner_retries(operation=lambda: self.embedding_async_client.aio.models.embed_content(model=model, contents=input), validate=validate_embedding_response, max_attempts=4, short_sleep=0.01, error_sleep=0.2)
                        if not response or not getattr(response, 'embeddings', None) or response.embeddings == []:
                            raise LLMServiceNoResponseError('Failed to obtain a valid response for embeddings.')
                        return response.embeddings
                    except errors.APIError as e:
                        if e.code == 429 or (e.details and e.details.get('code') == 429):
                            logger.warning(f'Rate limit error encountered: {e.code} - {e.message}. Delegating to outer retry.')
                            raise
                        elif e.code in (400, 403, 404):
                            logger.error(f'Client error encountered: {e.code} - {e.message}. Check your request parameters or API key.')
                            raise
                        elif e.code in (500, 503, 504):
                            logger.error(f'Server error encountered: {e.code} - {e.message}. Consider retrying after a short delay.')
                            raise
                        else:
                            logger.exception(f'Unexpected API error encountered: {e.code} - {e.message}')
                            raise
                    except Exception as e:
                        logger.exception(f'Unexpected error during embedding request: {e}')
                        raise

