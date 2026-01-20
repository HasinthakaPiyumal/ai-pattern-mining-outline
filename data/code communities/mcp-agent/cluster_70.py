# Cluster 70

class OpenAIAugmentedLLM(AugmentedLLM[ChatCompletionMessageParam, ChatCompletionMessage]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    This implementation uses OpenAI's ChatCompletion as the LLM.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, type_converter=MCPOpenAITypeConverter, **kwargs)
        self.provider = 'OpenAI'
        self.logger = get_logger(f'{__name__}.{self.name}' if self.name else __name__)
        self.model_preferences = self.model_preferences or ModelPreferences(costPriority=0.3, speedPriority=0.4, intelligencePriority=0.3)
        if 'default_model' in kwargs:
            default_model = kwargs['default_model']
        else:
            default_model = 'gpt-4o'
        self._reasoning_effort = 'medium'
        if self.context and self.context.config and self.context.config.openai:
            if hasattr(self.context.config.openai, 'default_model'):
                default_model = self.context.config.openai.default_model
            if hasattr(self.context.config.openai, 'reasoning_effort'):
                self._reasoning_effort = self.context.config.openai.reasoning_effort
        self._reasoning = lambda model: model and model.startswith(('o1', 'o3', 'o4', 'gpt-5'))
        if self._reasoning(default_model):
            self.logger.info(f"Using reasoning model '{default_model}' with '{self._reasoning_effort}' reasoning effort")
        self.default_request_params = self.default_request_params or RequestParams(model=default_model, modelPreferences=self.model_preferences, maxTokens=4096, systemPrompt=self.instruction, parallel_tool_calls=False, max_iterations=10, use_history=True)

    @classmethod
    def get_provider_config(cls, context):
        return getattr(getattr(context, 'config', None), 'openai', None)

    @classmethod
    def convert_message_to_message_param(cls, message: ChatCompletionMessage, **kwargs) -> ChatCompletionMessageParam:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        assistant_message_params = {'role': 'assistant', 'audio': message.audio, 'refusal': message.refusal, **kwargs}
        if message.content is not None:
            assistant_message_params['content'] = message.content
        if message.tool_calls is not None:
            assistant_message_params['tool_calls'] = message.tool_calls
        return ChatCompletionAssistantMessageParam(**assistant_message_params)

    @track_tokens()
    async def generate(self, message, request_params: RequestParams | None=None):
        """
        Process a query using an LLM and available tools.
        The default implementation uses OpenAI's ChatCompletion as the LLM.
        Override this method to use a different LLM.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.{self.name}.generate') as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)
            self._annotate_span_for_generation_message(span, message)
            messages: List[ChatCompletionMessageParam] = []
            params = self.get_request_params(request_params)
            if self.context.tracing_enabled:
                AugmentedLLM.annotate_span_with_request_params(span, params)
            if params.use_history:
                messages.extend(self.history.get())
            system_prompt = self.instruction or params.systemPrompt
            if system_prompt and len(messages) == 0:
                span.set_attribute('system_prompt', system_prompt)
                messages.append(ChatCompletionSystemMessageParam(role='system', content=system_prompt))
            messages.extend(OpenAIConverter.convert_mixed_messages_to_openai(message))
            response: ListToolsResult = await self.agent.list_tools(tool_filter=params.tool_filter)
            available_tools: List[ChatCompletionToolParam] = [ChatCompletionToolParam(type='function', function={'name': tool.name, 'description': tool.description, 'parameters': tool.inputSchema}) for tool in response.tools]
            if self.context.tracing_enabled:
                span.set_attribute('available_tools', [t.get('function', {}).get('name') for t in available_tools])
            if not available_tools:
                available_tools = None
            responses: List[ChatCompletionMessage] = []
            model = await self.select_model(params)
            if model:
                span.set_attribute(GEN_AI_REQUEST_MODEL, model)
            user = params.user or getattr(self.context.config.openai, 'user', None)
            if self.context.tracing_enabled and user:
                span.set_attribute('user', user)
            total_input_tokens = 0
            total_output_tokens = 0
            finish_reasons = []
            for i in range(params.max_iterations):
                arguments = {'model': model, 'messages': messages, 'tools': available_tools}
                if user:
                    arguments['user'] = user
                if params.stopSequences is not None:
                    arguments['stop'] = params.stopSequences
                if self._reasoning(model):
                    arguments = {**arguments, 'max_completion_tokens': params.maxTokens, 'reasoning_effort': self._reasoning_effort}
                else:
                    arguments = {**arguments, 'max_tokens': params.maxTokens}
                if params.metadata:
                    arguments = {**arguments, **params.metadata}
                self.logger.debug('Completion request arguments:', data=arguments)
                self._log_chat_progress(chat_turn=len(messages) // 2, model=model)
                request = RequestCompletionRequest(config=self.context.config.openai, payload=arguments)
                self._annotate_span_for_completion_request(span, request, i)
                response: ChatCompletion = await self.executor.execute(OpenAICompletionTasks.request_completion_task, ensure_serializable(request))
                self.logger.debug('OpenAI ChatCompletion response:', data=response)
                if isinstance(response, BaseException):
                    self.logger.error(f'Error: {response}')
                    span.record_exception(response)
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    break
                self._annotate_span_for_completion_response(span, response, i)
                iteration_input = response.usage.prompt_tokens
                iteration_output = response.usage.completion_tokens
                total_input_tokens += iteration_input
                total_output_tokens += iteration_output
                if self.context.token_counter:
                    await self.context.token_counter.record_usage(input_tokens=iteration_input, output_tokens=iteration_output, model_name=model, provider=self.provider)
                if not response.choices or len(response.choices) == 0:
                    break
                choice = response.choices[0]
                message = choice.message
                responses.append(message)
                finish_reasons.append(choice.finish_reason)
                sanitized_name = re.sub('[^a-zA-Z0-9_-]', '_', self.name) if isinstance(self.name, str) else None
                converted_message = self.convert_message_to_message_param(message, name=sanitized_name)
                messages.append(converted_message)
                if choice.finish_reason in ['tool_calls', 'function_call'] and message.tool_calls:
                    tool_tasks = [functools.partial(self.execute_tool_call, tool_call=tool_call) for tool_call in message.tool_calls]
                    tool_results = await self.executor.execute_many(tool_tasks)
                    self.logger.debug(f'Iteration {i}: Tool call results: {(str(tool_results) if tool_results else 'None')}')
                    for result in tool_results:
                        if isinstance(result, BaseException):
                            self.logger.error(f'Warning: Unexpected error during tool execution: {result}. Continuing...')
                            span.record_exception(result)
                            continue
                        if result is not None:
                            messages.append(result)
                elif choice.finish_reason == 'length':
                    self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'length'")
                    span.set_attribute('finish_reason', 'length')
                    break
                elif choice.finish_reason == 'content_filter':
                    self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'content_filter'")
                    span.set_attribute('finish_reason', 'content_filter')
                    break
                elif choice.finish_reason == 'stop':
                    self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'stop'")
                    span.set_attribute('finish_reason', 'stop')
                    break
            if params.use_history:
                self.history.set(messages)
            self._log_chat_finished(model=model)
            if self.context.tracing_enabled:
                span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, total_input_tokens)
                span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, total_output_tokens)
                span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)
                for i, res in enumerate(responses):
                    response_data = self.extract_response_message_attributes_for_tracing(res, prefix=f'response.{i}')
                    span.set_attributes(response_data)
            return responses

    async def generate_str(self, message, request_params: RequestParams | None=None):
        """
        Process a query using an LLM and available tools.
        The default implementation uses OpenAI's ChatCompletion as the LLM.
        Override this method to use a different LLM.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.{self.name}.generate_str') as span:
            if self.context.tracing_enabled:
                span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)
                self._annotate_span_for_generation_message(span, message)
                if request_params:
                    AugmentedLLM.annotate_span_with_request_params(span, request_params)
            responses = await self.generate(message=message, request_params=request_params)
            final_text: List[str] = []
            for response in responses:
                content = response.content
                if not content:
                    continue
                if isinstance(content, str):
                    final_text.append(content)
                    continue
            res = '\n'.join(final_text)
            span.set_attribute('response', res)
            return res

    async def generate_structured(self, message, response_model: Type[ModelT], request_params: RequestParams | None=None) -> ModelT:
        """
        Use OpenAI native structured outputs via response_format (JSON schema).
        """
        import json
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.{self.name}.generate_structured') as span:
            if self.context.tracing_enabled:
                span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)
                self._annotate_span_for_generation_message(span, message)
            params = self.get_request_params(request_params)
            model = await self.select_model(params) or (self.default_request_params.model or 'gpt-4o')
            if self.context.tracing_enabled:
                AugmentedLLM.annotate_span_with_request_params(span, params)
                span.set_attribute(GEN_AI_REQUEST_MODEL, model)
                span.set_attribute('response_model', response_model.__name__)
            messages: List[ChatCompletionMessageParam] = []
            system_prompt = self.instruction or params.systemPrompt
            if system_prompt:
                messages.append(ChatCompletionSystemMessageParam(role='system', content=system_prompt))
            if params.use_history:
                messages.extend(self.history.get())
            messages.extend(OpenAIConverter.convert_mixed_messages_to_openai(message))
            schema = response_model.model_json_schema()

            def _ensure_no_additional_props_and_require_all(node: dict):
                if not isinstance(node, dict):
                    return
                node_type = node.get('type')
                if node_type == 'object':
                    if 'additionalProperties' not in node:
                        node['additionalProperties'] = False
                    props = node.get('properties')
                    if isinstance(props, dict):
                        node['required'] = list(props.keys())
                for key in ('properties', '$defs', 'definitions'):
                    sub = node.get(key)
                    if isinstance(sub, dict):
                        for v in sub.values():
                            _ensure_no_additional_props_and_require_all(v)
                if 'items' in node:
                    _ensure_no_additional_props_and_require_all(node['items'])
                for key in ('oneOf', 'anyOf', 'allOf'):
                    subs = node.get(key)
                    if isinstance(subs, list):
                        for v in subs:
                            _ensure_no_additional_props_and_require_all(v)
            if params.strict:
                _ensure_no_additional_props_and_require_all(schema)
            response_format = {'type': 'json_schema', 'json_schema': {'name': getattr(response_model, '__name__', 'StructuredOutput'), 'schema': schema, 'strict': params.strict}}
            payload = {'model': model, 'messages': messages, 'response_format': response_format}
            if self._reasoning(model):
                payload['max_completion_tokens'] = params.maxTokens
                payload['reasoning_effort'] = self._reasoning_effort
            else:
                payload['max_tokens'] = params.maxTokens
            user = params.user or getattr(self.context.config.openai, 'user', None)
            if user:
                payload['user'] = user
            if params.stopSequences is not None:
                payload['stop'] = params.stopSequences
            if params.metadata:
                payload.update(params.metadata)
            completion: ChatCompletion = await self.executor.execute(OpenAICompletionTasks.request_completion_task, RequestCompletionRequest(config=self.context.config.openai, payload=payload))
            if isinstance(completion, BaseException):
                raise completion
            if not completion.choices or completion.choices[0].message.content is None:
                raise ValueError('No structured content returned by model')
            content = completion.choices[0].message.content
            try:
                data = json.loads(content)
                return response_model.model_validate(data)
            except Exception:
                return response_model.model_validate_json(content)

    async def pre_tool_call(self, tool_call_id: str | None, request: CallToolRequest):
        return request

    async def post_tool_call(self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult):
        return result

    async def execute_tool_call(self, tool_call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessageParam:
        """
        Execute a single tool call and return the result message.
        Returns a single ChatCompletionToolMessageParam object.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.{self.name}.execute_tool_call') as span:
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments
            tool_call_id = tool_call.id
            tool_args = {}
            if self.context.tracing_enabled:
                span.set_attribute(GEN_AI_TOOL_CALL_ID, tool_call_id)
                span.set_attribute(GEN_AI_TOOL_NAME, tool_name)
                span.set_attribute('tool_args', tool_args_str)
            try:
                if tool_args_str:
                    tool_args = json.loads(tool_args_str)
            except json.JSONDecodeError as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                return ChatCompletionToolMessageParam(role='tool', tool_call_id=tool_call_id, content=f"Invalid JSON provided in tool call arguments for '{tool_name}'. Failed to load JSON: {str(e)}")
            tool_call_request = CallToolRequest(method='tools/call', params=CallToolRequestParams(name=tool_name, arguments=tool_args))
            result = await self.call_tool(request=tool_call_request, tool_call_id=tool_call_id)
            self._annotate_span_for_call_tool_result(span, result)
            return ChatCompletionToolMessageParam(role='tool', tool_call_id=tool_call_id, content=[mcp_content_to_openai_content_part(c) for c in result.content])

    def message_param_str(self, message: ChatCompletionMessageParam) -> str:
        """Convert an input message to a string representation."""
        if message.get('content'):
            content = message['content']
            if isinstance(content, str):
                return content
            else:
                final_text: List[str] = []
                for part in content:
                    text_part = part.get('text')
                    if text_part:
                        final_text.append(str(text_part))
                    else:
                        final_text.append(str(part))
                return '\n'.join(final_text)
        return str(message)

    def message_str(self, message: ChatCompletionMessage, content_only: bool=False) -> str:
        """Convert an output message to a string representation."""
        content = message.content
        if content:
            return content
        elif content_only:
            return ''
        return str(message)

    def _annotate_span_for_generation_message(self, span: trace.Span, message: MessageTypes) -> None:
        """Annotate the span with the message content."""
        if not self.context.tracing_enabled:
            return
        if isinstance(message, str):
            span.set_attribute('message.content', message)
        elif isinstance(message, list):
            for i, msg in enumerate(message):
                if isinstance(msg, str):
                    span.set_attribute(f'message.{i}.content', msg)
                else:
                    span.set_attribute(f'message.{i}', str(msg))
        else:
            span.set_attribute('message', str(message))

    def _extract_message_param_attributes_for_tracing(self, message_param: ChatCompletionMessageParam, prefix: str='message') -> dict[str, Any]:
        """Return a flat dict of span attributes for a given ChatCompletionMessageParam."""
        attrs = {}
        return attrs

    def _annotate_span_for_completion_request(self, span: trace.Span, request: RequestCompletionRequest, turn: int) -> None:
        """Annotate the span with the completion request as an event."""
        if not self.context.tracing_enabled:
            return
        event_data = {'completion.request.turn': turn, 'config.reasoning_effort': request.config.reasoning_effort}
        if request.config.base_url:
            event_data['config.base_url'] = request.config.base_url
        for key, value in request.payload.items():
            if key == 'messages':
                for i, message in enumerate(cast(List[ChatCompletionMessageParam], value)):
                    role = message.get('role')
                    event_data[f'messages.{i}.role'] = role
                    message_content = message.get('content')
                    match role:
                        case 'developer' | 'system' | 'user':
                            if isinstance(message_content, str):
                                event_data[f'messages.{i}.content'] = message_content
                            elif message_content is not None:
                                for j, part in enumerate(message_content):
                                    event_data[f'messages.{i}.content.{j}.type'] = part['type']
                                    if part['type'] == 'text':
                                        event_data[f'messages.{i}.content.{j}.text'] = part['text']
                                    elif part['type'] == 'image_url':
                                        event_data[f'messages.{i}.content.{j}.image_url.url'] = part['image_url']['url']
                                        event_data[f'messages.{i}.content.{j}.image_url.detail'] = part['image_url']['detail']
                                    elif part['type'] == 'input_audio':
                                        event_data[f'messages.{i}.content.{j}.input_audio.format'] = part['input_audio']['format']
                        case 'assistant':
                            if isinstance(message_content, str):
                                event_data[f'messages.{i}.content'] = message_content
                            elif message_content is not None:
                                for j, part in enumerate(message_content):
                                    event_data[f'messages.{i}.content.{j}.type'] = part['type']
                                    if part['type'] == 'text':
                                        event_data[f'messages.{i}.content.{j}.text'] = part['text']
                                    elif part['type'] == 'refusal':
                                        event_data[f'messages.{i}.content.{j}.refusal'] = part['refusal']
                            if message.get('audio') is not None:
                                event_data[f'messages.{i}.audio.id'] = message.get('audio').get('id')
                            if message.get('function_call') is not None:
                                event_data[f'messages.{i}.function_call.name'] = message.get('function_call').get('name')
                                event_data[f'messages.{i}.function_call.arguments'] = message.get('function_call').get('arguments')
                            if message.get('name') is not None:
                                event_data[f'messages.{i}.name'] = message.get('name')
                            if message.get('refusal') is not None:
                                event_data[f'messages.{i}.refusal'] = message.get('refusal')
                            if message.get('tool_calls') is not None:
                                for j, tool_call in enumerate(message.get('tool_calls')):
                                    event_data[f'messages.{i}.tool_calls.{j}.{GEN_AI_TOOL_CALL_ID}'] = tool_call.id
                                    event_data[f'messages.{i}.tool_calls.{j}.function.name'] = tool_call.function.name
                                    event_data[f'messages.{i}.tool_calls.{j}.function.arguments'] = tool_call.function.arguments
                        case 'tool':
                            event_data[f'messages.{i}.{GEN_AI_TOOL_CALL_ID}'] = message.get('tool_call_id')
                            if isinstance(message_content, str):
                                event_data[f'messages.{i}.content'] = message_content
                            elif message_content is not None:
                                for j, part in enumerate(message_content):
                                    event_data[f'messages.{i}.content.{j}.type'] = part['type']
                                    if part['type'] == 'text':
                                        event_data[f'messages.{i}.content.{j}.text'] = part['text']
                        case 'function':
                            event_data[f'messages.{i}.name'] = message.get('name')
                            event_data[f'messages.{i}.content'] = message_content
            elif key == 'tools':
                if value is not None:
                    event_data['tools'] = [tool.get('function', {}).get('name') for tool in value]
            elif is_otel_serializable(value):
                event_data[key] = value
        event_name = f'completion.request.{turn}'
        latest_message_role = request.payload.get('messages', [{}])[-1].get('role')
        if latest_message_role:
            event_name = f'gen_ai.{latest_message_role}.message'
        span.add_event(event_name, event_data)

    def _annotate_span_for_completion_response(self, span: trace.Span, response: ChatCompletion, turn: int) -> None:
        """Annotate the span with the completion response as an event."""
        if not self.context.tracing_enabled:
            return
        event_data = {'completion.response.turn': turn}
        event_data.update(self._extract_chat_completion_attributes_for_tracing(response))
        event_name = f'completion.response.{turn}'
        if response.choices and len(response.choices) > 0:
            latest_message_role = response.choices[0].message.role
            event_name = f'gen_ai.{latest_message_role}.message'
        span.add_event(event_name, event_data)

    def extract_response_message_attributes_for_tracing(self, message: ChatCompletionMessage, prefix: str | None=None) -> Dict[str, Any]:
        """
        Extract relevant attributes from the ChatCompletionMessage for tracing.
        """
        if not self.context.tracing_enabled:
            return {}
        attr_prefix = f'{prefix}.' if prefix else ''
        attrs = {f'{attr_prefix}role': message.role}
        if message.content is not None:
            attrs[f'{attr_prefix}content'] = message.content
        if message.refusal:
            attrs[f'{attr_prefix}refusal'] = message.refusal
        if message.audio is not None:
            attrs[f'{attr_prefix}audio.id'] = message.audio.id
            attrs[f'{attr_prefix}audio.expires_at'] = message.audio.expires_at
            attrs[f'{attr_prefix}audio.transcript'] = message.audio.transcript
        if message.function_call is not None:
            attrs[f'{attr_prefix}function_call.name'] = message.function_call.name
            attrs[f'{attr_prefix}function_call.arguments'] = message.function_call.arguments
        if message.tool_calls:
            for j, tool_call in enumerate(message.tool_calls):
                attrs[f'{attr_prefix}tool_calls.{j}.{GEN_AI_TOOL_CALL_ID}'] = tool_call.id
                attrs[f'{attr_prefix}tool_calls.{j}.function.name'] = tool_call.function.name
                attrs[f'{attr_prefix}tool_calls.{j}.function.arguments'] = tool_call.function.arguments
        return attrs

    def _extract_chat_completion_attributes_for_tracing(self, response: ChatCompletion, prefix: str | None=None) -> Dict[str, Any]:
        """
        Extract relevant attributes from the ChatCompletion response for tracing.
        """
        if not self.context.tracing_enabled:
            return {}
        attr_prefix = f'{prefix}.' if prefix else ''
        attrs = {f'{attr_prefix}id': response.id, f'{attr_prefix}model': response.model, f'{attr_prefix}object': response.object, f'{attr_prefix}created': response.created}
        if response.service_tier:
            attrs[f'{attr_prefix}service_tier'] = response.service_tier
        if response.system_fingerprint:
            attrs[f'{attr_prefix}system_fingerprint'] = response.system_fingerprint
        if response.usage:
            attrs[f'{attr_prefix}{GEN_AI_USAGE_INPUT_TOKENS}'] = response.usage.prompt_tokens
            attrs[f'{attr_prefix}{GEN_AI_USAGE_OUTPUT_TOKENS}'] = response.usage.completion_tokens
        finish_reasons = []
        for i, choice in enumerate(response.choices):
            attrs[f'{attr_prefix}choices.{i}.index'] = choice.index
            attrs[f'{attr_prefix}choices.{i}.finish_reason'] = choice.finish_reason
            finish_reasons.append(choice.finish_reason)
            message_attrs = self.extract_response_message_attributes_for_tracing(choice.message, f'{attr_prefix}choices.{i}.message')
            attrs.update(message_attrs)
        attrs[GEN_AI_RESPONSE_FINISH_REASONS] = finish_reasons
        return attrs

def is_otel_serializable(value: Any) -> bool:
    """
    Check if a value is serializable by OpenTelemetry
    """
    allowed_types = (bool, str, bytes, int, float)
    if isinstance(value, allowed_types):
        return True
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
        return all((isinstance(item, allowed_types) for item in value))
    return False

class AnthropicAugmentedLLM(AugmentedLLM[MessageParam, Message]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    Our current models can actively use these capabilities—generating their own search queries,
    selecting appropriate tools, and determining what information to retain.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, type_converter=AnthropicMCPTypeConverter, **kwargs)
        self.provider = 'Anthropic'
        self.logger = get_logger(f'{__name__}.{self.name}' if self.name else __name__)
        self.model_preferences = self.model_preferences or ModelPreferences(costPriority=0.3, speedPriority=0.4, intelligencePriority=0.3)
        default_model = 'claude-sonnet-4-20250514'
        if self.context.config.anthropic:
            self.provider = self.context.config.anthropic.provider
            if self.context.config.anthropic.provider == 'bedrock':
                default_model = 'anthropic.claude-sonnet-4-20250514-v1:0'
            elif self.context.config.anthropic.provider == 'vertexai':
                default_model = 'claude-sonnet-4@20250514'
            if hasattr(self.context.config.anthropic, 'default_model'):
                default_model = self.context.config.anthropic.default_model
        self.default_request_params = self.default_request_params or RequestParams(model=default_model, modelPreferences=self.model_preferences, maxTokens=2048, systemPrompt=self.instruction, parallel_tool_calls=False, max_iterations=10, use_history=True)

    @classmethod
    def get_provider_config(cls, context):
        return getattr(getattr(context, 'config', None), 'anthropic', None)

    @track_tokens()
    async def generate(self, message, request_params: RequestParams | None=None):
        """
        Process a query using an LLM and available tools.
        The default implementation uses Claude as the LLM.
        Override this method to use a different LLM.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.{self.name}.generate') as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)
            self._annotate_span_for_generation_message(span, message)
            config = self.context.config
            messages: List[MessageParam] = []
            params = self.get_request_params(request_params)
            if self.context.tracing_enabled:
                AugmentedLLM.annotate_span_with_request_params(span, params)
            if params.use_history:
                messages.extend(self.history.get())
            messages.extend(AnthropicConverter.convert_mixed_messages_to_anthropic(message))
            list_tools_result = await self.agent.list_tools(tool_filter=params.tool_filter)
            available_tools: List[ToolParam] = [{'name': tool.name, 'description': tool.description, 'input_schema': tool.inputSchema} for tool in list_tools_result.tools]
            responses: List[Message] = []
            model = await self.select_model(params)
            if model:
                span.set_attribute(GEN_AI_REQUEST_MODEL, model)
            total_input_tokens = 0
            total_output_tokens = 0
            finish_reasons = []
            for i in range(params.max_iterations):
                if i == params.max_iterations - 1 and responses and (responses[-1].stop_reason == 'tool_use'):
                    final_prompt_message = MessageParam(role='user', content="We've reached the maximum number of iterations. \n                        Please stop using tools now and provide your final comprehensive answer based on all tool results so far. \n                        At the beginning of your response, clearly indicate that your answer may be incomplete due to reaching the maximum number of tool usage iterations, \n                        and explain what additional information you would have needed to provide a more complete answer.")
                    messages.append(final_prompt_message)
                arguments = {'model': model, 'max_tokens': params.maxTokens, 'messages': messages, 'stop_sequences': params.stopSequences or [], 'tools': available_tools}
                if (system := (self.instruction or params.systemPrompt)):
                    arguments['system'] = system
                if params.metadata:
                    arguments = {**arguments, **params.metadata}
                self.logger.debug('Completion request arguments:', data=arguments)
                self._log_chat_progress(chat_turn=(len(messages) + 1) // 2, model=model)
                request = RequestCompletionRequest(config=config.anthropic, payload=arguments)
                self._annotate_span_for_completion_request(span, request, i)
                response: Message = await self.executor.execute(AnthropicCompletionTasks.request_completion_task, ensure_serializable(request))
                if isinstance(response, BaseException):
                    self.logger.error(f'Error: {response}')
                    span.record_exception(response)
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    break
                self.logger.debug(f'{model} response:', data=response)
                self._annotate_span_for_completion_response(span, response, i)
                iteration_input = response.usage.input_tokens
                iteration_output = response.usage.output_tokens
                total_input_tokens += iteration_input
                total_output_tokens += iteration_output
                response_as_message = self.convert_message_to_message_param(response)
                messages.append(response_as_message)
                responses.append(response)
                finish_reasons.append(response.stop_reason)
                if self.context.token_counter:
                    await self.context.token_counter.record_usage(input_tokens=iteration_input, output_tokens=iteration_output, model_name=model, provider=self.provider)
                if response.stop_reason == 'end_turn':
                    self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'end_turn'")
                    span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, ['end_turn'])
                    break
                elif response.stop_reason == 'stop_sequence':
                    self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'stop_sequence'")
                    span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, ['stop_sequence'])
                    break
                elif response.stop_reason == 'max_tokens':
                    self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'max_tokens'")
                    span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, ['max_tokens'])
                    break
                else:
                    for content in response.content:
                        if content.type == 'tool_use':
                            tool_name = content.name
                            tool_args = content.input
                            tool_use_id = content.id
                            tool_call_request = CallToolRequest(method='tools/call', params=CallToolRequestParams(name=tool_name, arguments=tool_args))
                            result = await self.call_tool(request=tool_call_request, tool_call_id=tool_use_id)
                            message = self.from_mcp_tool_result(result, tool_use_id)
                            messages.append(message)
            if params.use_history:
                self.history.set(messages)
            self._log_chat_finished(model=model)
            if self.context.tracing_enabled:
                span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, total_input_tokens)
                span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, total_output_tokens)
                span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)
                for i, response in enumerate(responses):
                    response_data = self.extract_response_message_attributes_for_tracing(response, prefix=f'response.{i}')
                    span.set_attributes(response_data)
            return responses

    async def generate_str(self, message, request_params: RequestParams | None=None) -> str:
        """
        Process a query using an LLM and available tools.
        The default implementation uses Claude as the LLM.
        Override this method to use a different LLM.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.{self.name}.generate_str') as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)
            self._annotate_span_for_generation_message(span, message)
            if self.context.tracing_enabled and request_params:
                AugmentedLLM.annotate_span_with_request_params(span, request_params)
            responses: List[Message] = await self.generate(message=message, request_params=request_params)
            final_text: List[str] = []
            for response in responses:
                for content in response.content:
                    if content.type == 'text':
                        final_text.append(content.text)
                    elif content.type == 'tool_use':
                        final_text.append(f'[Calling tool {content.name} with args {content.input}]')
            res = '\n'.join(final_text)
            span.set_attribute('response', res)
            return res

    async def generate_structured(self, message, response_model: Type[ModelT], request_params: RequestParams | None=None) -> ModelT:
        import json
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.{self.name}.generate_structured') as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)
            self._annotate_span_for_generation_message(span, message)
            params = self.get_request_params(request_params)
            if self.context.tracing_enabled:
                AugmentedLLM.annotate_span_with_request_params(span, params)
            model_name = await self.select_model(params) or self.default_request_params.model
            span.set_attribute(GEN_AI_REQUEST_MODEL, model_name)
            messages: List[MessageParam] = []
            if params.use_history:
                messages.extend(self.history.get())
            messages.extend(AnthropicConverter.convert_mixed_messages_to_anthropic(message))
            schema = response_model.model_json_schema()
            tools: List[ToolParam] = [{'name': 'return_structured_output', 'description': 'Return the response in the required JSON format', 'input_schema': schema}]
            args = {'model': model_name, 'messages': messages, 'system': self.instruction or params.systemPrompt, 'tools': tools, 'tool_choice': {'type': 'tool', 'name': 'return_structured_output'}}
            if params.maxTokens is not None:
                args['max_tokens'] = params.maxTokens
            if params.stopSequences:
                args['stop_sequences'] = params.stopSequences
            base_url = None
            if self.context and self.context.config and self.context.config.anthropic:
                base_url = self.context.config.anthropic.base_url
                api_key = self.context.config.anthropic.api_key
                client = AsyncAnthropic(api_key=api_key, base_url=base_url)
            else:
                client = AsyncAnthropic()
            async with client:
                stream_method = client.messages.stream
                if all((hasattr(stream_method, attr) for attr in ('__aenter__', '__aexit__'))):
                    async with stream_method(**args) as stream:
                        final = await stream.get_final_message()
                else:
                    final = await client.messages.create(**args)
            for block in final.content:
                if getattr(block, 'type', None) == 'tool_use' and getattr(block, 'name', '') == 'return_structured_output':
                    data = getattr(block, 'input', None)
                    try:
                        if isinstance(data, str):
                            return response_model.model_validate(json.loads(data))
                        return response_model.model_validate(data)
                    except Exception:
                        break
            raise ValueError('Failed to obtain structured output from Anthropic response')

    @classmethod
    def convert_message_to_message_param(cls, message: Message, **kwargs) -> MessageParam:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        content = []
        for content_block in message.content:
            if content_block.type == 'text':
                content.append(TextBlockParam(type='text', text=content_block.text))
            elif content_block.type == 'tool_use':
                content.append(ToolUseBlockParam(type='tool_use', name=content_block.name, input=content_block.input, id=content_block.id))
        return MessageParam(role='assistant', content=content, **kwargs)

    def message_param_str(self, message: MessageParam) -> str:
        """Convert an input message to a string representation."""
        if message.get('content'):
            content = message['content']
            if isinstance(content, str):
                return content
            else:
                final_text: List[str] = []
                for block in content:
                    if block.text:
                        final_text.append(str(block.text))
                    else:
                        final_text.append(str(block))
                return '\n'.join(final_text)
        return str(message)

    def message_str(self, message: Message, content_only: bool=False) -> str:
        """Convert an output message to a string representation."""
        content = message.content
        if content:
            if isinstance(content, list):
                final_text: List[str] = []
                for block in content:
                    if block.text:
                        final_text.append(str(block.text))
                    else:
                        final_text.append(str(block))
                return '\n'.join(final_text)
            else:
                return str(content)
        elif content_only:
            return ''
        return str(message)

    def _extract_message_param_attributes_for_tracing(self, message_param: MessageParam, prefix: str='message') -> dict[str, Any]:
        """Return a flat dict of span attributes for a given MessageParam."""
        if not self.context.tracing_enabled:
            return {}
        attrs = {}
        attrs[f'{prefix}.role'] = message_param.get('role')
        message_content = message_param.get('content')
        if isinstance(message_content, str):
            attrs[f'{prefix}.content'] = message_content
        elif isinstance(message_content, list):
            for j, part in enumerate(message_content):
                message_content_prefix = f'{prefix}.content.{j}'
                attrs[f'{message_content_prefix}.type'] = part.get('type')
                match part.get('type'):
                    case 'text':
                        attrs[f'{message_content_prefix}.text'] = part.get('text')
                    case 'image':
                        source_type = part.get('source', {}).get('type')
                        attrs[f'{message_content_prefix}.source.type'] = source_type
                        if source_type == 'base64':
                            attrs[f'{message_content_prefix}.source.media_type'] = part.get('source', {}).get('media_type')
                        elif source_type == 'url':
                            attrs[f'{message_content_prefix}.source.url'] = part.get('source', {}).get('url')
                    case 'tool_use':
                        attrs[f'{message_content_prefix}.id'] = part.get('id')
                        attrs[f'{message_content_prefix}.name'] = part.get('name')
                    case 'tool_result':
                        attrs[f'{message_content_prefix}.tool_use_id'] = part.get('tool_use_id')
                        attrs[f'{message_content_prefix}.is_error'] = part.get('is_error')
                        part_content = part.get('content')
                        if isinstance(part_content, str):
                            attrs[f'{message_content_prefix}.content'] = part_content
                        elif isinstance(part_content, list):
                            for k, sub_part in enumerate(part_content):
                                sub_part_type = sub_part.get('type')
                                if sub_part_type == 'text':
                                    attrs[f'{message_content_prefix}.content.{k}.text'] = sub_part.get('text')
                                elif sub_part_type == 'image':
                                    sub_part_source = sub_part.get('source')
                                    sub_part_source_type = sub_part_source.get('type')
                                    attrs[f'{message_content_prefix}.content.{k}.source.type'] = sub_part_source_type
                                    if sub_part_source_type == 'base64':
                                        attrs[f'{message_content_prefix}.content.{k}.source.media_type'] = sub_part_source.get('media_type')
                                    elif sub_part_source_type == 'url':
                                        attrs[f'{message_content_prefix}.content.{k}.source.url'] = sub_part_source.get('url')
                    case 'document':
                        if part.get('context') is not None:
                            attrs[f'{message_content_prefix}.context'] = part.get('context')
                        if part.get('title') is not None:
                            attrs[f'{message_content_prefix}.title'] = part.get('title')
                        if part.get('citations') is not None:
                            attrs[f'{message_content_prefix}.citations.enabled'] = part.get('citations').get('enabled')
                        part_source_type = part.get('source', {}).get('type')
                        attrs[f'{message_content_prefix}.source.type'] = part_source_type
                        if part_source_type == 'text':
                            attrs[f'{message_content_prefix}.source.data'] = part.get('source', {}).get('data')
                        elif part_source_type == 'url':
                            attrs[f'{message_content_prefix}.source.url'] = part.get('source', {}).get('url')
                    case 'thinking':
                        attrs[f'{message_content_prefix}.thinking'] = part.get('thinking')
                        attrs[f'{message_content_prefix}.signature'] = part.get('signature')
                    case 'redacted_thinking':
                        attrs[f'{message_content_prefix}.redacted_thinking'] = part.get('data')
        return attrs

    def extract_response_message_attributes_for_tracing(self, message: Message, prefix: str | None=None) -> dict[str, Any]:
        """Return a flat dict of span attributes for a given Message."""
        if not self.context.tracing_enabled:
            return {}
        attr_prefix = f'{prefix}.' if prefix else ''
        attrs = {f'{attr_prefix}id': message.id, f'{attr_prefix}model': message.model, f'{attr_prefix}role': message.role}
        if message.stop_reason:
            attrs[f'{attr_prefix}{GEN_AI_RESPONSE_FINISH_REASONS}'] = [message.stop_reason]
        if message.stop_sequence:
            attrs[f'{attr_prefix}stop_sequence'] = message.stop_sequence
        if message.usage:
            attrs[f'{attr_prefix}{GEN_AI_USAGE_INPUT_TOKENS}'] = message.usage.input_tokens
            attrs[f'{attr_prefix}{GEN_AI_USAGE_OUTPUT_TOKENS}'] = message.usage.output_tokens
        for i, block in enumerate(message.content):
            attrs[f'{attr_prefix}content.{i}.type'] = block.type
            match block.type:
                case 'text':
                    attrs[f'{attr_prefix}content.{i}.text'] = block.text
                case 'tool_use':
                    attrs[f'{attr_prefix}content.{i}.tool_use_id'] = block.id
                    attrs[f'{attr_prefix}content.{i}.name'] = block.name
                case 'thinking':
                    attrs[f'{attr_prefix}content.{i}.thinking'] = block.thinking
                    attrs[f'{attr_prefix}content.{i}.signature'] = block.signature
                case 'redacted_thinking':
                    attrs[f'{attr_prefix}content.{i}.redacted_thinking'] = block.data
        return attrs

    def _annotate_span_for_completion_request(self, span: trace.Span, request: RequestCompletionRequest, turn: int):
        """Annotate the span with the completion request as an event."""
        if not self.context.tracing_enabled:
            return
        event_data = {'completion.request.turn': turn}
        for key, value in request.payload.items():
            if key == 'messages':
                for i, message in enumerate(cast(List[MessageParam], value)):
                    event_data.update(self._extract_message_param_attributes_for_tracing(message, prefix=f'messages.{i}'))
            elif key == 'tools':
                if value is not None:
                    event_data['tools'] = [tool.get('name') for tool in value]
            elif is_otel_serializable(value):
                event_data[key] = value
        event_name = f'completion.request.{turn}'
        latest_message_role = request.payload.get('messages', [{}])[-1].get('role')
        if latest_message_role:
            event_name = f'gen_ai.{latest_message_role}.message'
        span.add_event(event_name, event_data)

    def _annotate_span_for_completion_response(self, span: trace.Span, response: Message, turn: int):
        """Annotate the span with the completion response as an event."""
        if not self.context.tracing_enabled:
            return
        event_data = {'completion.response.turn': turn}
        event_data.update(self.extract_response_message_attributes_for_tracing(response))
        span.add_event(f'gen_ai.{response.role}.message', event_data)

