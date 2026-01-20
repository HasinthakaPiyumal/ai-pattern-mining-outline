# Cluster 13

def workflow_task(_fn: Callable[..., R] | None=None, *, name: str=None, schedule_to_close_timeout: timedelta=None, retry_policy: Dict[str, Any]=None, **meta_kwargs) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Static decorator to mark a function as a workflow task without requiring direct app access.
    These tasks will be registered with the MCPApp during app initialization.

    Args:
        name: Optional custom name for the activity
        schedule_to_close_timeout: Maximum time the task can take to complete
        retry_policy: Retry policy configuration
        **meta_kwargs: Additional metadata passed to the activity registration

    Returns:
        Decorated function that preserves async and typing information
    """

    def decorator(target: Callable[..., R]) -> Callable[..., R]:
        func = unwrap(target)
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(f'{func.__qualname__} must be async')
        activity_name = name or f'{func.__module__}.{func.__qualname__}'
        metadata = {'activity_name': activity_name, 'schedule_to_close_timeout': schedule_to_close_timeout or timedelta(minutes=10), 'retry_policy': retry_policy or {}, **meta_kwargs}
        registry = GlobalWorkflowTaskRegistry()
        registry.register_task(target, metadata)
        func.is_workflow_task = True
        func.execution_metadata = metadata
        return target
    if _fn is None:
        return decorator
    return decorator(_fn)

def decorator(target: Callable[..., R]) -> Callable[..., R]:
    func = unwrap(target)
    if not asyncio.iscoroutinefunction(func):
        raise TypeError(f'{func.__qualname__} must be async')
    activity_name = name or f'{func.__module__}.{func.__qualname__}'
    metadata = {'activity_name': activity_name, 'schedule_to_close_timeout': schedule_to_close_timeout or timedelta(minutes=10), 'retry_policy': retry_policy or {}, **meta_kwargs}
    registry = GlobalWorkflowTaskRegistry()
    registry.register_task(target, metadata)
    func.is_workflow_task = True
    func.execution_metadata = metadata
    return target

def serialize_attribute(key: str, value: Any) -> Dict[str, Any]:
    """Serialize a single attribute value into a flat dict of OpenTelemetry-compatible values."""
    serialized = {}
    if is_otel_serializable(value):
        serialized[key] = value
    elif isinstance(value, dict):
        for sub_key, sub_value in value.items():
            serialized.update(serialize_attribute(f'{key}.{sub_key}', sub_value))
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            serialized.update(serialize_attribute(f'{key}.{idx}', item))
    elif isinstance(value, Callable):
        serialized[f'{key}_callable_name'] = getattr(value, '__qualname__', str(value))
        serialized[f'{key}_callable_module'] = getattr(value, '__module__', 'unknown')
        serialized[f'{key}_is_coroutine'] = asyncio.iscoroutinefunction(value)
    elif inspect.iscoroutine(value):
        serialized[f'{key}_coroutine'] = str(value)
        serialized[f'{key}_is_coroutine'] = True
    else:
        s = str(value)
        serialized[key] = s if len(s) < 256 else s[:255] + '…'
    return serialized

def serialize_attributes(attributes: Dict[str, Any], prefix: str='') -> Dict[str, Any]:
    """Serialize a dict of attributes into a flat OpenTelemetry-compatible dict."""
    serialized = {}
    prefix = f'{prefix}.' if prefix else ''
    for key, value in attributes.items():
        full_key = f'{prefix}{key}'
        serialized.update(serialize_attribute(full_key, value))
    return serialized

def record_attribute(span: trace.Span, key, value):
    """Record a single serializable value on the span."""
    if is_otel_serializable(value):
        span.set_attribute(key, value)
    else:
        serialized = serialize_attribute(key, value)
        for attr_key, attr_value in serialized.items():
            span.set_attribute(attr_key, attr_value)

def record_attributes(span: trace.Span, attributes: Dict[str, Any], prefix: str=''):
    """Record a dict of attributes on the span after serialization."""
    serialized = serialize_attributes(attributes, prefix)
    for attr_key, attr_value in serialized.items():
        span.set_attribute(attr_key, attr_value)

def get_tracer(context: 'Context') -> trace.Tracer:
    """
    Get the OpenTelemetry tracer for the context.
    """
    return getattr(context, 'tracer', None) or trace.get_tracer('mcp-agent')

def track_tokens(node_type: str='llm') -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to track token usage for AugmentedLLM methods.
    Automatically pushes/pops token context around method execution.

    Args:
        node_type: The type of node for token tracking. Default is "llm" for base AugmentedLLM classes.
                  Higher-order AugmentedLLM classes should use "agent".
    """

    def decorator(method: Callable[..., T]) -> Callable[..., T]:

        @functools.wraps(method)
        async def wrapper(self, *args, **kwargs) -> T:
            is_temporal_replay = False
            is_temporal_engine = False
            try:
                cfg = getattr(getattr(self, 'context', None), 'config', None)
                is_temporal_engine = getattr(cfg, 'execution_engine', None) == 'temporal'
            except Exception:
                is_temporal_engine = False
            if is_temporal_engine:
                try:
                    from temporalio import workflow as _twf
                    if _twf.in_workflow():
                        is_temporal_replay = _twf.unsafe.is_replaying()
                except Exception:
                    is_temporal_replay = False
            if hasattr(self, 'context') and self.context and self.context.token_counter and (not is_temporal_replay):
                metadata = {'method': method.__name__, 'class': self.__class__.__name__}
                if hasattr(self, 'provider'):
                    metadata['provider'] = getattr(self, 'provider')
                async with self.context.token_counter.scope(name=getattr(self, 'name', self.__class__.__name__), node_type=node_type, metadata=metadata):
                    return await method(self, *args, **kwargs)
            else:
                return await method(self, *args, **kwargs)
        return wrapper
    return decorator

def to_application_error(error: BaseException, *, message: str | None=None, type: str | None=None, non_retryable: bool | None=None, details: object | None=None) -> WorkflowApplicationError:
    """Wrap an existing exception as a WorkflowApplicationError."""
    msg = message or str(error)
    err_type = type or getattr(error, 'type', None) or error.__class__.__name__
    nr = non_retryable
    if nr is None:
        nr = bool(getattr(error, 'non_retryable', False))
    det = details
    if det is None:
        det = getattr(error, 'details', None)
    if isinstance(det, tuple):
        det = list(det)
    return WorkflowApplicationError(msg, type=err_type, non_retryable=nr, details=det)

def ensure_serializable(data: BaseModel) -> BaseModel:
    """
    Workaround for https://github.com/pydantic/pydantic/issues/7713, see https://github.com/pydantic/pydantic/issues/7713#issuecomment-2604574418
    """
    try:
        json.dumps(data)
    except TypeError:
        data_json_from_dicts = json.dumps(data, default=lambda x: vars(x))
        data_obj = json.loads(data_json_from_dicts)
        data = type(data)(**data_obj)
    return data

def _raise_non_retryable_azure(error: Exception, status_code: int | None=None) -> None:
    message = str(error)
    if status_code is not None:
        message = f'{status_code}: {message}'
    raise to_application_error(error, message=message, non_retryable=True) from error

class ModelSelector(ContextDependent):
    """
    A heuristic-based selector to choose the best model from a list of models.

    Because LLMs can vary along multiple dimensions, choosing the "best" model is
    rarely straightforward.  Different models excel in different areas—some are
    faster but less capable, others are more capable but more expensive, and so
    on.

    MCP's ModelPreferences interface allows servers to express their priorities across multiple
    dimensions to help clients make an appropriate selection for their use case.
    """

    def __init__(self, models: List[ModelInfo]=None, benchmark_weights: Dict[str, float] | None=None, context: Optional['Context']=None):
        super().__init__(context=context)
        if not models:
            self.models = load_default_models()
        else:
            self.models = models
        if benchmark_weights:
            self.benchmark_weights = benchmark_weights
        else:
            self.benchmark_weights = {'mmlu': 0.4, 'gsm8k': 0.3, 'bbh': 0.3}
        if abs(sum(self.benchmark_weights.values()) - 1.0) > 1e-06:
            raise ValueError('Benchmark weights must sum to 1.0')
        self.max_values = self._calculate_max_scores(self.models)
        self.models_by_provider = self._models_by_provider(self.models)

    def select_best_model(self, model_preferences: ModelPreferences, provider: str | None=None, min_tokens: int | None=None, max_tokens: int | None=None, tool_calling: bool | None=None, structured_outputs: bool | None=None) -> ModelInfo:
        """
        Select the best model from a given list of models based on the given model preferences.

        Args:
            model_preferences: MCP ModelPreferences with cost, speed, and intelligence priorities
            provider: Optional provider to filter models by
            min_tokens: Minimum context window size (in tokens) required
            max_tokens: Maximum context window size (in tokens) allowed
            tool_calling: If True, only include models with tool calling support; if None, no filter
            structured_outputs: If True, only include models with structured outputs support; if None, no filter

        Returns:
            ModelInfo: The best model based on the preferences and filters

        Raises:
            ValueError: If no models match the specified criteria
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.select_best_model') as span:
            if self.context.tracing_enabled and self.benchmark_weights:
                for k, v in self.benchmark_weights.items():
                    span.set_attribute(f'benchmark_weights.{k}', v)
            if min_tokens is not None:
                span.set_attribute('min_tokens', min_tokens)
            if max_tokens is not None:
                span.set_attribute('max_tokens', max_tokens)
            if tool_calling is not None:
                span.set_attribute('tool_calling', tool_calling)
            if structured_outputs is not None:
                span.set_attribute('structured_outputs', structured_outputs)
            models: List[ModelInfo] = []
            if provider:
                provider_key = provider.lower()
                models = self.models_by_provider.get(provider_key, [])
                if not models:
                    models = self.models
                span.set_attribute('provider', provider)
            else:
                models = self.models
            if not models:
                raise ValueError(f'No models available for selection. Provider={provider}')
            span.set_attribute('models', [model.name for model in models])
            candidate_models = models
            if model_preferences.hints:
                candidate_models = []
                for model in models:
                    for hint in model_preferences.hints:
                        passes_hint = self._check_model_hint(model, hint)
                        span.set_attribute(f'model_hint.{hint.name}', passes_hint)
                        if passes_hint:
                            candidate_models.append(model)
                if not candidate_models:
                    candidate_models = models
            filtered_models = []
            for model in candidate_models:
                if min_tokens is not None and model.context_window is not None:
                    if model.context_window < min_tokens:
                        continue
                if max_tokens is not None and model.context_window is not None:
                    if model.context_window > max_tokens:
                        continue
                if tool_calling is not None and model.tool_calling is not None:
                    if tool_calling and (not model.tool_calling):
                        continue
                if structured_outputs is not None and model.structured_outputs is not None:
                    if structured_outputs and (not model.structured_outputs):
                        continue
                filtered_models.append(model)
            candidate_models = filtered_models
            if not candidate_models:
                raise ValueError(f'No models match the specified criteria. min_tokens={min_tokens}, max_tokens={max_tokens}, tool_calling={tool_calling}, structured_outputs={structured_outputs}')
            scores = []
            for model in candidate_models:
                cost_score = self._calculate_cost_score(model, model_preferences, max_cost=self.max_values['max_cost'])
                speed_score = self._calculate_speed_score(model, max_tokens_per_second=self.max_values['max_tokens_per_second'], max_time_to_first_token_ms=self.max_values['max_time_to_first_token_ms'])
                intelligence_score = self._calculate_intelligence_score(model, self.max_values)
                model_score = (model_preferences.costPriority or 0) * cost_score + (model_preferences.speedPriority or 0) * speed_score + (model_preferences.intelligencePriority or 0) * intelligence_score
                scores.append((model_score, model))
                if self.context.tracing_enabled:
                    span.set_attribute(f'model.{model.name}.cost_score', cost_score)
                    span.set_attribute(f'model.{model.name}.speed_score', speed_score)
                    span.set_attribute(f'model.{model.name}.intelligence_score', intelligence_score)
                    span.set_attribute(f'model.{model.name}.total_score', model_score)
            best_model = max(scores, key=lambda x: x[0])[1]
            span.set_attribute('best_model', best_model.name)
            return best_model

    def _models_by_provider(self, models: List[ModelInfo]) -> Dict[str, List[ModelInfo]]:
        """
        Group models by provider.
        """
        provider_models: Dict[str, List[ModelInfo]] = {}
        for model in models:
            key = (model.provider or '').lower()
            if key not in provider_models:
                provider_models[key] = []
            provider_models[key].append(model)
        return provider_models

    def _check_model_hint(self, model: ModelInfo, hint: ModelHint) -> bool:
        """
        Check if a model matches a specific hint.
        """
        desired_name: str | None = hint.name
        desired_provider: str | None = getattr(hint, 'provider', None)
        if desired_name and ':' in desired_name and (not desired_provider):
            lhs, rhs = desired_name.split(':', 1)
            if lhs.strip() and rhs.strip():
                desired_provider = lhs.strip()
                desired_name = rhs.strip()
        name_match = True
        if desired_name:
            dn = desired_name.lower()
            mn = (model.name or '').lower()
            name_match = dn == mn or dn in mn or mn in dn
        provider_match = True
        if desired_provider:
            dp = desired_provider.lower()
            mp = (model.provider or '').lower()
            provider_match = dp == mp
        return name_match and provider_match

    def _calculate_total_cost(self, model: ModelInfo, io_ratio: float=3.0) -> float:
        """
        Calculate a single cost metric of a model based on input/output token costs,
        and a ratio of input to output tokens.

        Args:
            model: The model to calculate the cost for.
            io_ratio: The estimated ratio of input to output tokens. Defaults to 3.0.
        """
        if model.metrics.cost.blended_cost_per_1m is not None:
            return model.metrics.cost.blended_cost_per_1m
        input_cost = model.metrics.cost.input_cost_per_1m
        output_cost = model.metrics.cost.output_cost_per_1m
        if input_cost is not None and output_cost is not None:
            return (input_cost * io_ratio + output_cost) / (1 + io_ratio)
        if input_cost is not None:
            return input_cost
        if output_cost is not None:
            return output_cost
        return 0.0

    def _calculate_cost_score(self, model: ModelInfo, model_preferences: ModelPreferences, max_cost: float) -> float:
        """Normalized 0->1 cost score for a model."""
        try:
            io_ratio = getattr(model_preferences, 'ioRatio', 3.0) or 3.0
        except Exception:
            io_ratio = 3.0
        total_cost = self._calculate_total_cost(model, io_ratio)
        if max_cost <= 0:
            return 1.0
        return max(0.0, 1 - total_cost / max_cost)

    def _calculate_intelligence_score(self, model: ModelInfo, max_values: Dict[str, float]) -> float:
        """
        Return a normalized 0->1 intelligence score for a model based on its benchmark metrics.
        """
        scores = []
        weights = []
        benchmark_dict: Dict[str, float] = model.metrics.intelligence.model_dump()
        use_weights = True
        for bench, score in benchmark_dict.items():
            key = f'max_{bench}'
            if score is not None and key in max_values:
                scores.append(score / max_values[key])
                if bench in self.benchmark_weights:
                    weights.append(self.benchmark_weights[bench])
                else:
                    use_weights = False
        if not scores:
            return 0
        elif use_weights:
            return average(scores, weights=weights)
        else:
            return average(scores)

    def _calculate_speed_score(self, model: ModelInfo, max_tokens_per_second: float, max_time_to_first_token_ms: float) -> float:
        """Normalized 0->1 cost score for a model."""
        time_to_first_token_score = 1 - model.metrics.speed.time_to_first_token_ms / max_time_to_first_token_ms
        tokens_per_second_score = model.metrics.speed.tokens_per_second / max_tokens_per_second
        latency_score = average([time_to_first_token_score, tokens_per_second_score], weights=[0.4, 0.6])
        return latency_score

    def _calculate_max_scores(self, models: List[ModelInfo]) -> Dict[str, float]:
        """
        Of all the models, calculate the maximum value for each benchmark metric.
        """
        max_dict: Dict[str, float] = {}
        max_dict['max_cost'] = max((self._calculate_total_cost(m) for m in models))
        max_dict['max_tokens_per_second'] = max(max((m.metrics.speed.tokens_per_second for m in models)), 1e-06)
        max_dict['max_time_to_first_token_ms'] = max(max((m.metrics.speed.time_to_first_token_ms for m in models)), 1e-06)
        for model in models:
            benchmark_dict: Dict[str, float] = model.metrics.intelligence.model_dump()
            for bench, score in benchmark_dict.items():
                if score is None:
                    continue
                key = f'max_{bench}'
                if key in max_dict:
                    max_dict[key] = max(max_dict[key], score)
                else:
                    max_dict[key] = score
        return max_dict

class AugmentedLLM(ContextDependent, AugmentedLLMProtocol[MessageParamT, MessageT]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    Our current models can actively use these capabilities—generating their own search queries,
    selecting appropriate tools, and determining what information to retain.
    """
    provider: str | None = None
    logger: Union['Logger', None] = None
    token_node_type: str = 'llm'

    def __init__(self, agent: Optional['Agent']=None, server_names: List[str] | None=None, instruction: str | None=None, name: str | None=None, default_request_params: RequestParams | None=None, type_converter: Type[ProviderToMCPConverter[MessageParamT, MessageT]]=None, context: Optional['Context']=None, **kwargs):
        """
        Initialize the LLM with a list of server names and an instruction.
        If a name is provided, it will be used to identify the LLM.
        If an agent is provided, all other properties are optional
        """
        super().__init__(context=context, **kwargs)
        self.executor = self.context.executor
        self.name = self._gen_name(name or (agent.name if agent else None), prefix=None)
        self.instruction = instruction or (agent.instruction if agent else None)
        if not self.name:
            raise ValueError('An AugmentedLLM must have a name or be provided with an agent that has a name')
        if agent:
            self.agent = agent
        else:
            from mcp_agent.agents.agent import Agent
            self.agent = Agent(name=self.name, **{'instruction': self.instruction} if self.instruction is not None else {}, server_names=server_names or [], llm=self)
        self.history: Memory[MessageParamT] = SimpleMemory[MessageParamT]()
        self.default_request_params = default_request_params
        self.model_preferences = self.default_request_params.modelPreferences if self.default_request_params else None
        self.model_selector = self.context.model_selector
        self.type_converter = type_converter

    async def __aenter__(self):
        if self.agent:
            await self.agent.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.agent:
            await self.agent.__aexit__(exc_type, exc_val, exc_tb)

    @abstractmethod
    async def generate(self, message: MessageTypes, request_params: RequestParams | None=None) -> List[MessageT]:
        """Request an LLM generation, which may run multiple iterations, and return the result"""

    @abstractmethod
    async def generate_str(self, message: MessageTypes, request_params: RequestParams | None=None) -> str:
        """Request an LLM generation and return the string representation of the result"""

    @abstractmethod
    async def generate_structured(self, message: MessageTypes, response_model: Type[ModelT], request_params: RequestParams | None=None) -> ModelT:
        """Request a structured LLM generation and return the result as a Pydantic model."""

    @classmethod
    def get_provider_config(cls, context: Optional['Context']):
        """Return the provider-specific settings object from the app context, or None."""
        return None

    async def select_model(self, request_params: RequestParams | None=None) -> str | None:
        """
        Select an LLM based on the request parameters.
        If a model is specified in the request, it will override the model selection criteria.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.{self.name}.select_model') as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)
            model_preferences = self.model_preferences
            if request_params is not None:
                model_preferences = request_params.modelPreferences or model_preferences
                model = request_params.model
                if model:
                    span.set_attribute('request_params.model', model)
                    span.set_attribute('model', model)
                    return model
            if not self.model_selector:
                self.model_selector = ModelSelector(context=self.context)
            try:
                model_info = self.model_selector.select_best_model(model_preferences=model_preferences, provider=self.provider)
                selected = model_info.name
                span.set_attribute('model', selected)
                return selected
            except ValueError as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                model = self.default_request_params.model if self.default_request_params else None
                if model:
                    span.set_attribute('model', model)
                return model

    def get_request_params(self, request_params: RequestParams | None=None, default: RequestParams | None=None) -> RequestParams:
        """
        Get request parameters with merged-in defaults and overrides.
        Args:
            request_params: The request parameters to use as overrides.
            default: The default request parameters to use as the base.
                If unspecified, self.default_request_params will be used.
        """
        default_request_params = default or self.default_request_params
        params = default_request_params.model_dump() if default_request_params else {}
        if request_params:
            params.update(request_params.model_dump(exclude_unset=True))
        return RequestParams(**params)

    def to_mcp_message_result(self, result: MessageT) -> MCPMessageResult:
        """Convert an LLM response to an MCP message result type."""
        return self.type_converter.to_mcp_message_result(result)

    def from_mcp_message_result(self, result: MCPMessageResult) -> MessageT:
        """Convert an MCP message result to an LLM response type."""
        return self.type_converter.from_mcp_message_result(result)

    def to_mcp_message_param(self, param: MessageParamT) -> MCPMessageParam:
        """Convert an LLM input to an MCP message (SamplingMessage) type."""
        return self.type_converter.to_mcp_message_param(param)

    def from_mcp_message_param(self, param: MCPMessageParam) -> MessageParamT:
        """Convert an MCP message (SamplingMessage) to an LLM input type."""
        return self.type_converter.from_mcp_message_param(param)

    def from_mcp_tool_result(self, result: CallToolResult, tool_use_id: str) -> MessageParamT:
        """Convert an MCP tool result to an LLM input type"""
        return self.type_converter.from_mcp_tool_result(result, tool_use_id)

    @classmethod
    def convert_message_to_message_param(cls, message: MessageT, **kwargs) -> MessageParamT:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        return message

    async def get_last_message(self) -> MessageParamT | None:
        """
        Return the last message generated by the LLM or None if history is empty.
        This is useful for prompt chaining workflows where the last message from one LLM is used as input to another.
        """
        history = self.history.get()
        return history[-1] if history else None

    async def get_last_message_str(self) -> str | None:
        """Return the string representation of the last message generated by the LLM or None if history is empty."""
        last_message = await self.get_last_message()
        return self.message_param_str(last_message) if last_message else None

    async def pre_tool_call(self, tool_call_id: str | None, request: CallToolRequest) -> CallToolRequest | bool:
        """Called before a tool is executed. Return False to prevent execution."""
        return request

    async def post_tool_call(self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult) -> CallToolResult:
        """Called after a tool execution. Can modify the result before it's returned."""
        return result

    async def call_tool(self, request: CallToolRequest, tool_call_id: str | None=None) -> CallToolResult:
        """Call a tool with the given parameters and optional ID"""
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.{self.name}.call_tool') as span:
            if self.context.tracing_enabled:
                span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)
                if tool_call_id:
                    span.set_attribute(GEN_AI_TOOL_CALL_ID, tool_call_id)
                    span.set_attribute('request.method', request.method)
                span.set_attribute('request.params.name', request.params.name)
                if request.params.arguments:
                    record_attributes(span, request.params.arguments, 'request.params.arguments')
            try:
                preprocess = await self.pre_tool_call(tool_call_id=tool_call_id, request=request)
                if isinstance(preprocess, bool):
                    if not preprocess:
                        span.set_attribute('preprocess', False)
                        span.set_status(trace.Status(trace.StatusCode.ERROR))
                        res = CallToolResult(isError=True, content=[TextContent(text=f"Error: Tool '{request.params.name}' was not allowed to run.")])
                        span.record_exception(Exception(res.content[0].text))
                        return res
                else:
                    request = preprocess
                tool_name = request.params.name
                tool_args = request.params.arguments
                span.set_attribute(f'processed.request.{GEN_AI_TOOL_NAME}', tool_name)
                if self.context.tracing_enabled and tool_args:
                    record_attributes(span, tool_args, 'processed.request.tool_args')
                result = await self.agent.call_tool(tool_name, tool_args)
                self._annotate_span_for_call_tool_result(span, result)
                postprocess = await self.post_tool_call(tool_call_id=tool_call_id, request=request, result=result)
                if isinstance(postprocess, CallToolResult):
                    result = postprocess
                    self._annotate_span_for_call_tool_result(span, result, processed=True)
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                return CallToolResult(isError=True, content=[TextContent(type='text', text=f"Error executing tool '{request.params.name}': {str(e)}")])

    async def list_tools(self, server_name: str | None=None, tool_filter: Dict[str, Set[str]] | None=None) -> ListToolsResult:
        """Call the underlying agent's list_tools method for a given server."""
        return await self.agent.list_tools(server_name=server_name, tool_filter=tool_filter)

    async def list_resources(self, server_name: str | None=None) -> ListResourcesResult:
        """Call the underlying agent's list_resources method for a given server."""
        return await self.agent.list_resources(server_name=server_name)

    async def read_resource(self, uri: str, server_name: str | None=None) -> ReadResourceResult:
        """Call the underlying agent's read_resource method for a given server."""
        return await self.agent.read_resource(uri=uri, server_name=server_name)

    async def list_prompts(self, server_name: str | None=None) -> ListPromptsResult:
        """Call the underlying agent's list_prompts method for a given server."""
        return await self.agent.list_prompts(server_name=server_name)

    async def get_prompt(self, name: str, server_name: str | None=None) -> GetPromptResult:
        """Call the underlying agent's get_prompt method for a given server."""
        return await self.agent.get_prompt(name=name, server_name=server_name)

    async def close(self):
        """Close underlying agent connections."""
        await self.agent.close()

    def message_param_str(self, message: MessageParamT) -> str:
        """Convert an input message to a string representation."""
        return str(message)

    def message_str(self, message: MessageT, content_only: bool=False) -> str:
        """Convert an output message to a string representation."""
        return str(message)

    def _log_chat_progress(self, chat_turn: Optional[int]=None, model: str | None=None):
        """Log a chat progress event"""
        data = {'progress_action': 'Chatting', 'model': model, 'agent_name': self.name, 'chat_turn': chat_turn if chat_turn is not None else None}
        self.logger.debug('Chat in progress', data=data)

    def _log_chat_finished(self, model: str | None=None):
        """Log a chat finished event"""
        data = {'progress_action': 'Finished', 'model': model, 'agent_name': self.name}
        self.logger.debug('Chat finished', data=data)

    @staticmethod
    def annotate_span_with_request_params(span: trace.Span, request_params: RequestParams):
        """Annotate the span with request parameters"""
        if hasattr(request_params, 'maxTokens'):
            span.set_attribute(GEN_AI_REQUEST_MAX_TOKENS, request_params.maxTokens)
        if hasattr(request_params, 'max_iterations'):
            span.set_attribute('request_params.max_iterations', request_params.max_iterations)
        if hasattr(request_params, 'temperature'):
            span.set_attribute(GEN_AI_REQUEST_TEMPERATURE, request_params.temperature)
        if hasattr(request_params, 'use_history'):
            span.set_attribute('request_params.use_history', request_params.use_history)
        if hasattr(request_params, 'parallel_tool_calls'):
            span.set_attribute('request_params.parallel_tool_calls', request_params.parallel_tool_calls)
        if hasattr(request_params, 'model') and request_params.model:
            span.set_attribute(GEN_AI_REQUEST_MODEL, request_params.model)
        if hasattr(request_params, 'modelPreferences') and request_params.modelPreferences:
            for attr, value in request_params.modelPreferences.model_dump(exclude_unset=True).items():
                if attr == 'hints' and value is not None:
                    span.set_attribute('request_params.modelPreferences.hints', [hint.name for hint in value])
                else:
                    record_attribute(span, f'request_params.modelPreferences.{attr}', value)
        if hasattr(request_params, 'systemPrompt') and request_params.systemPrompt:
            span.set_attribute('request_params.systemPrompt', request_params.systemPrompt)
        if hasattr(request_params, 'includeContext') and request_params.includeContext:
            span.set_attribute('request_params.includeContext', request_params.includeContext)
        if hasattr(request_params, 'stopSequences') and request_params.stopSequences:
            span.set_attribute(GEN_AI_REQUEST_STOP_SEQUENCES, request_params.stopSequences)
        if hasattr(request_params, 'metadata') and request_params.metadata:
            record_attributes(span, request_params.metadata, 'request_params.metadata')

    def _annotate_span_for_generation_message(self, span: trace.Span, message: str | MessageParamT | List[MessageParamT]) -> None:
        """Annotate the span with the message content."""
        if not self.context.tracing_enabled:
            return
        if isinstance(message, str):
            span.set_attribute('message.content', message)
        elif isinstance(message, list):
            for i, msg in enumerate(message):
                if isinstance(msg, str):
                    span.set_attribute(f'message.{i}', msg)
                else:
                    span.set_attribute(f'message.{i}.content', str(msg))
        else:
            span.set_attribute('message', str(message))

    def _extract_message_param_attributes_for_tracing(self, message_param: MessageParamT, prefix: str='message') -> dict[str, Any]:
        """
        Return a flat dict of span attributes for a given MessageParamT.
        Override this for the AugmentedLLM subclass MessageParamT type.
        """
        return {}

    def _annotate_span_for_call_tool_result(self, span: trace.Span, result: CallToolResult, processed: bool=False):
        if not self.context.tracing_enabled:
            return
        prefix = 'processed.result' if processed else 'result'
        span.set_attribute(f'{prefix}.isError', result.isError)
        if result.isError:
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            error_message = result.content[0].text if len(result.content) > 0 and result.content[0].type == 'text' else 'Error calling tool'
            span.record_exception(Exception(error_message))
        else:
            for idx, content in enumerate(result.content):
                span.set_attribute(f'{prefix}.content.{idx}.type', content.type)
                if content.type == 'text':
                    span.set_attribute(f'{prefix}.content.{idx}.text', result.content[idx].text)

    def extract_response_message_attributes_for_tracing(self, message: MessageT, prefix: str | None=None) -> dict[str, Any]:
        """
        Return a flat dict of span attributes for a given MessageT.
        Override this for the AugmentedLLM subclass MessageT type.
        """
        return {}

    def _gen_name(self, name: str | None, prefix: str | None) -> str:
        """
        Generate a name for the LLM based on the provided name or the default prefix.
        """
        if name:
            return name
        if not prefix:
            prefix = self.__class__.__name__
        identifier: str | None = None
        if not self.context or not self.context.executor:
            import uuid
            identifier = str(uuid.uuid4())
        else:
            identifier = str(self.context.executor.uuid())
        return f'{prefix}-{identifier}'

    async def get_token_node(self, return_all_matches: bool=False, node_type: str | None=None):
        """Return this LLM's token node(s) from the global counter."""
        if not self.context or not getattr(self.context, 'token_counter', None):
            return [] if return_all_matches else None
        counter = self.context.token_counter
        t = node_type or getattr(self, 'token_node_type', None)
        if return_all_matches:
            if t == 'llm':
                return await counter.get_llm_node(self.name, return_all_matches=True)
            if t == 'agent':
                return await counter.get_agent_node(self.name, return_all_matches=True)
            nodes = await counter.get_llm_node(self.name, return_all_matches=True)
            nodes += await counter.get_agent_node(self.name, return_all_matches=True)
            return nodes
        else:
            if t == 'agent':
                node = await counter.get_agent_node(self.name)
                if node:
                    return node
            if t == 'llm' or not t:
                node = await counter.get_llm_node(self.name)
                if node:
                    return node
            return await counter.get_agent_node(self.name)

    async def get_token_usage(self, node_type: str | None=None):
        """Return aggregated token usage for this LLM node (including children)."""
        if not self.context or not getattr(self.context, 'token_counter', None):
            return None
        counter = self.context.token_counter
        t = node_type or getattr(self, 'token_node_type', None)
        if t == 'agent':
            return await counter.get_agent_usage(self.name)
        if t == 'llm':
            return await counter.get_node_usage(self.name, 'llm')
        return await counter.get_node_usage(self.name)

    async def get_token_cost(self, node_type: str | None=None) -> float:
        """Return total cost for this LLM node (including children)."""
        if not self.context or not getattr(self.context, 'token_counter', None):
            return 0.0
        counter = self.context.token_counter
        t = node_type or getattr(self, 'token_node_type', None)
        if t:
            return await counter.get_node_cost(self.name, t)
        return await counter.get_node_cost(self.name)

    async def watch_tokens(self, callback, *, threshold: int | None=None, throttle_ms: int | None=None, include_subtree: bool=True, node_type: str | None=None) -> str | None:
        """Watch this LLM's token usage. Returns a watch_id or None if not available."""
        if not self.context or not getattr(self.context, 'token_counter', None):
            return None
        counter = self.context.token_counter
        t = node_type or getattr(self, 'token_node_type', None) or 'llm'
        return await counter.watch(callback=callback, node_name=self.name, node_type=t, threshold=threshold, throttle_ms=throttle_ms, include_subtree=include_subtree)

def create_anthropic_instance(settings: AnthropicSettings):
    """Select and initialise the appropriate anthropic client instance based on settings"""
    if settings.provider == 'bedrock':
        anthropic = AnthropicBedrock(aws_access_key=settings.aws_access_key_id, aws_secret_key=settings.aws_secret_access_key, aws_session_token=settings.aws_session_token, aws_region=settings.aws_region)
    elif settings.provider == 'vertexai':
        anthropic = AnthropicVertex(region=settings.location, project_id=settings.project)
    else:
        anthropic = Anthropic(api_key=settings.api_key)
    return anthropic

def get_planning_context(objective: str, progress_summary: str='', completed_steps: list=None, knowledge_items: list=None, available_servers: list=None, available_agents: dict=None) -> str:
    """Build planning context with XML structure."""
    context_parts = ['<planning_context>']
    context_parts.append(f'  <objective>{objective}</objective>')
    if progress_summary:
        context_parts.append('  <progress>')
        context_parts.append(f'    <summary>{progress_summary}</summary>')
        if completed_steps:
            context_parts.append('    <completed_steps>')
            for step in completed_steps[:5]:
                context_parts.append(f'      <step>{step}</step>')
            context_parts.append('    </completed_steps>')
        context_parts.append('  </progress>')
    if knowledge_items:
        context_parts.append('  <accumulated_knowledge>')
        for item in knowledge_items[:10]:
            context_parts.append(f'    <knowledge confidence="{item.get('confidence', 0.8):.2f}" category="{item.get('category', 'general')}">')
            context_parts.append(f'      <key>{item.get('key', 'Unknown')}</key>')
            value_str = str(item.get('value', ''))[:200]
            context_parts.append(f'      <value>{value_str}</value>')
            context_parts.append('    </knowledge>')
        context_parts.append('  </accumulated_knowledge>')
    context_parts.append('  <resources>')
    if available_servers:
        context_parts.append(f'    <mcp_servers>{', '.join(available_servers)}</mcp_servers>')
        context_parts.append('    <important>You MUST only use these exact server names. Do NOT invent or guess server names.</important>')
    else:
        context_parts.append('    <mcp_servers>None available</mcp_servers>')
        context_parts.append('    <important>No MCP servers are available. All tasks must have empty server lists.</important>')
    if available_agents:
        context_parts.append(f'    <agents>{', '.join(available_agents.keys())}</agents>')
        context_parts.append('    <important>You MUST only use these exact agent names if specifying an agent. Do NOT invent or guess agent names. Leave agent field unset for dynamic creation.</important>')
    else:
        context_parts.append('    <agents>None available - all tasks must have agent field unset</agents>')
        context_parts.append('    <important>No predefined agents are available. All tasks must leave the agent field unset for dynamic agent creation.</important>')
    context_parts.append('  </resources>')
    context_parts.append('</planning_context>')
    return '\n'.join(context_parts)

def get_full_plan_prompt(context: str) -> str:
    """Get prompt for creating a full execution plan."""
    return f'<plan_request>\n{context}\n\nCreate a comprehensive plan to achieve the objective.\n</plan_request>'

def get_verification_context(objective: str, progress_summary: str, knowledge_summary: str='', artifacts: dict=None) -> str:
    """Build verification context."""
    context_parts = ['<verification_context>', f'  <original_objective>{objective}</original_objective>', f'  <execution_summary>{progress_summary}</execution_summary>']
    if knowledge_summary:
        context_parts.append('  <accumulated_knowledge>')
        context_parts.append(knowledge_summary)
        context_parts.append('  </accumulated_knowledge>')
    if artifacts:
        context_parts.append('  <created_artifacts>')
        for name, content in list(artifacts.items())[-5:]:
            context_parts.append(f'    <artifact name="{name}">')
            preview = content[:200] + '...' if len(content) > 200 else content
            context_parts.append(f'      {preview}')
            context_parts.append('    </artifact>')
        context_parts.append('  </created_artifacts>')
    context_parts.append('</verification_context>')
    return '\n'.join(context_parts)

def get_verification_prompt(context: str) -> str:
    """Get prompt for verification."""
    return f'{context}\n\n<request>Verify if the objective has been completed.</request>'

def get_synthesis_context(objective: str, execution_summary: dict, completed_steps: list, knowledge_by_category: dict, artifacts: dict) -> str:
    """Build comprehensive synthesis context."""
    context_parts = ['<synthesis_context>', f'  <original_objective>{objective}</original_objective>', '', '  <execution_summary>', f'    <iterations>{execution_summary.get('iterations', 0)}</iterations>', f'    <steps_completed>{execution_summary.get('steps_completed', 0)}</steps_completed>', f'    <tasks_completed>{execution_summary.get('tasks_completed', 0)}</tasks_completed>', f'    <tokens_used>{execution_summary.get('tokens_used', 0)}</tokens_used>', f'    <cost>${execution_summary.get('cost', 0):.2f}</cost>', '  </execution_summary>', '', '  <completed_work>']
    for step in completed_steps:
        context_parts.append(f'    <step name="{step.get('description', 'Unknown')}">')
        for task_result in step.get('task_results', []):
            if task_result.get('success'):
                task_desc = task_result.get('description', 'Unknown task')
                output_summary = task_result.get('output', '')[:300]
                if len(task_result.get('output', '')) > 300:
                    output_summary += '...'
                context_parts.append('      <task_result>')
                context_parts.append(f'        <task>{task_desc}</task>')
                context_parts.append(f'        <output>{output_summary}</output>')
                context_parts.append('      </task_result>')
        context_parts.append('    </step>')
    context_parts.append('  </completed_work>')
    if knowledge_by_category:
        context_parts.append('')
        context_parts.append('  <accumulated_knowledge>')
        for category, items in knowledge_by_category.items():
            context_parts.append(f'    <category name="{category}">')
            for item in items[:5]:
                context_parts.append(f'      <knowledge confidence="{item.confidence:.2f}">')
                context_parts.append(f'        <key>{item.key}</key>')
                value_str = str(item.value)[:200] + '...' if len(str(item.value)) > 200 else str(item.value)
                context_parts.append(f'        <value>{value_str}</value>')
                context_parts.append('      </knowledge>')
            context_parts.append('    </category>')
        context_parts.append('  </accumulated_knowledge>')
    if artifacts:
        context_parts.append('')
        context_parts.append('  <artifacts_created>')
        for name, content in list(artifacts.items())[-10:]:
            content_preview = content[:500] + '...' if len(content) > 500 else content
            context_parts.append(f'    <artifact name="{name}">')
            context_parts.append(f'      {content_preview}')
            context_parts.append('    </artifact>')
        context_parts.append('  </artifacts_created>')
    context_parts.append('</synthesis_context>')
    return '\n'.join(context_parts)

def get_synthesis_prompt(context: str) -> str:
    """Get prompt for final synthesis."""
    return f'{context}\n\n<synthesis_request>\nCreate the final deliverable that fully addresses the original objective.\nSynthesize all work completed, knowledge gained, and artifacts created into a comprehensive response.\n</synthesis_request>'

def get_emergency_context(objective: str, error: str, progress_summary: str, partial_knowledge: list=None, artifacts_created: list=None) -> str:
    """Build emergency completion context."""
    context_parts = ['<emergency_context>', f'  <objective>{objective}</objective>', f'  <error>{error}</error>', f'  <progress>{progress_summary}</progress>']
    if partial_knowledge:
        context_parts.append('  <partial_knowledge>')
        for item in partial_knowledge[:10]:
            key = item.get('key', 'Unknown')
            value = str(item.get('value', ''))[:100]
            context_parts.append(f'    - {key}: {value}')
        context_parts.append('  </partial_knowledge>')
    if artifacts_created:
        artifacts_str = ', '.join(artifacts_created[:5])
        context_parts.append(f'  <artifacts_created>{artifacts_str}</artifacts_created>')
    context_parts.append('</emergency_context>')
    return '\n'.join(context_parts)

def get_emergency_prompt(context: str) -> str:
    """Get prompt for emergency completion."""
    return f'{context}\n\nProvide the most helpful response possible given the circumstances.'

def _score(query: str, entry: Dict[str, str]) -> int:
    score = 0
    for token in query.split():
        if len(token) < 3:
            continue
        token_lower = token.lower()
        if token_lower in entry['topic'].lower():
            score += 3
        if token_lower in entry['summary'].lower():
            score += 2
        if token_lower in entry['faq'].lower():
            score += 1
    return score

def test_to_application_error_from_exception():

    class CustomError(Exception):

        def __init__(self, message):
            super().__init__(message)
            self.type = 'Custom'
            self.non_retryable = True
            self.details = ['detail']
    original = CustomError('boom')
    converted = to_application_error(original)
    assert isinstance(converted, WorkflowApplicationError)
    assert converted.type == 'Custom'
    assert converted.non_retryable is True
    assert converted.workflow_details == ['detail']

