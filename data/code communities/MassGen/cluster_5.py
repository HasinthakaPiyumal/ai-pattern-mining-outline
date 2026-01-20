# Cluster 5

def log_backend_agent_message(agent_id: str, direction: str, message: dict, backend_name: str=None):
    """
    Log backend-to-LLM messages for debugging.

    Args:
        agent_id: ID of the agent
        direction: "SEND" or "RECV"
        message: Message content as dictionary
        backend_name: Optional name of the backend provider
    """
    func_name, line_num = _get_caller_info()
    if backend_name:
        log_name = f'backend.{backend_name}→{agent_id}:{func_name}:{line_num}'
        log = logger.bind(name=log_name)
    else:
        log_name = f'backend→{agent_id}:{func_name}:{line_num}'
        log = logger.bind(name=log_name)
    if _DEBUG_MODE:
        if direction == 'SEND':
            log.opt(colors=True).debug('<yellow>⚙️📤 [{}] Backend sending to LLM: {}</yellow>', log_name, _format_message(message))
        elif direction == 'RECV':
            log.opt(colors=True).debug('<yellow>⚙️📥 [{}] Backend received from LLM: {}</yellow>', log_name, _format_message(message))
        else:
            log.opt(colors=True).debug('<yellow>⚙️📨 [{}] {}: {}</yellow>', log_name, direction, _format_message(message))

def log_backend_activity(backend_name: str, activity: str, details: dict=None, agent_id: str=None):
    """
    Log backend activities for debugging.

    Args:
        backend_name: Name of the backend (e.g., "openai", "claude")
        activity: Description of the activity
        details: Additional details as dictionary
        agent_id: Optional ID of the agent using this backend
    """
    func_name, line_num = _get_caller_info()
    if agent_id:
        log_name = f'{agent_id}.{backend_name}'
        log = logger.bind(name=f'{log_name}:{func_name}:{line_num}')
    else:
        log_name = backend_name
        log = logger.bind(name=f'backend.{backend_name}:{func_name}:{line_num}')
    if _DEBUG_MODE:
        log.opt(colors=True).debug('<yellow>⚙️ [{}] {}: {}</yellow>', log_name, activity, details or {})

def log_stream_chunk(source: str, chunk_type: str, content: Any=None, agent_id: str=None):
    """
    Log stream chunks at INFO level (always logged to file).

    Args:
        source: Source of the stream chunk (e.g., "orchestrator", "backend.claude_code")
        chunk_type: Type of the chunk (e.g., "content", "tool_call", "error")
        content: Content of the chunk
        agent_id: Optional agent ID for context
    """
    frame = inspect.currentframe()
    if frame and frame.f_back:
        caller_frame = frame.f_back
        function_name = caller_frame.f_code.co_name
        line_number = caller_frame.f_lineno
    else:
        function_name = 'unknown'
        line_number = 0
    if agent_id:
        log_name = f'{source}.{agent_id}'
    else:
        log_name = source
    log = logger.bind(name=f'{log_name}:{function_name}:{line_number}')
    if content:
        if isinstance(content, dict):
            log.info('Stream chunk [{}]: {}', chunk_type, content)
        else:
            log.info('Stream chunk [{}]: {}', chunk_type, content)
    else:
        log.info('Stream chunk [{}]', chunk_type)

def log_tool_call(agent_id: str, tool_name: str, arguments: dict, result: Any=None, backend_name: str=None):
    """
    Log tool calls made by agents.

    Args:
        agent_id: ID of the agent making the tool call
        tool_name: Name of the tool being called
        arguments: Arguments passed to the tool
        result: Result returned by the tool (optional)
        backend_name: Optional name of the backend provider
    """
    if backend_name:
        log_name = f'{agent_id}.{backend_name}'
        log = logger.bind(name=f'{log_name}.tools')
    else:
        log_name = agent_id
        log = logger.bind(name=f'{agent_id}.tools')
    if _DEBUG_MODE:
        if result is not None:
            log.opt(colors=True).debug("<light-black>🔧 [{}] Tool '{}' called with args: {} -> Result: {}</light-black>", log_name, tool_name, arguments, result)
        else:
            log.opt(colors=True).debug("<light-black>🔧 [{}] Calling tool '{}' with args: {}</light-black>", log_name, tool_name, arguments)

class ChatCompletionsBackend(MCPBackend):
    """Complete OpenAI-compatible Chat Completions API backend.

    Can be used directly with any OpenAI-compatible provider by setting provider name.
    Supports Cerebras AI, Together AI, Fireworks AI, DeepInfra, and other compatible providers.

    Environment Variables:
        Provider-specific API keys are automatically detected based on provider name.
        See ProviderRegistry.PROVIDERS for the complete list.

    """

    def __init__(self, api_key: Optional[str]=None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.backend_name = self.get_provider_name()
        self.formatter = ChatCompletionsFormatter()
        self.api_params_handler = ChatCompletionsAPIParamsHandler(self)

    def supports_upload_files(self) -> bool:
        """Chat Completions backend supports upload_files preprocessing."""
        return True

    async def stream_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Stream response using OpenAI Response API with unified MCP/non-MCP processing."""
        async for chunk in super().stream_with_tools(messages, tools, **kwargs):
            yield chunk

    async def _stream_with_mcp_tools(self, current_messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], client, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Recursively stream MCP responses, executing function calls as needed."""
        all_params = {**self.config, **kwargs}
        api_params = await self.api_params_handler.build_api_params(current_messages, tools, all_params)
        provider_tools = self.api_params_handler.get_provider_tools(all_params)
        if provider_tools:
            if 'tools' not in api_params:
                api_params['tools'] = []
            api_params['tools'].extend(provider_tools)
        stream = await client.chat.completions.create(**api_params)
        captured_function_calls = []
        current_tool_calls = {}
        response_completed = False
        content = ''
        async for chunk in stream:
            try:
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, 'delta') and choice.delta:
                        delta = choice.delta
                        if getattr(delta, 'content', None):
                            content_chunk = delta.content
                            content += content_chunk
                            yield StreamChunk(type='content', content=content_chunk)
                        if getattr(delta, 'tool_calls', None):
                            for tool_call_delta in delta.tool_calls:
                                index = getattr(tool_call_delta, 'index', 0)
                                if index not in current_tool_calls:
                                    current_tool_calls[index] = {'id': '', 'function': {'name': '', 'arguments': ''}}
                                if getattr(tool_call_delta, 'id', None):
                                    current_tool_calls[index]['id'] = tool_call_delta.id
                                if hasattr(tool_call_delta, 'function') and tool_call_delta.function:
                                    if getattr(tool_call_delta.function, 'name', None):
                                        current_tool_calls[index]['function']['name'] = tool_call_delta.function.name
                                    if getattr(tool_call_delta.function, 'arguments', None):
                                        current_tool_calls[index]['function']['arguments'] += tool_call_delta.function.arguments
                    if getattr(choice, 'finish_reason', None):
                        if choice.finish_reason == 'tool_calls' and current_tool_calls:
                            final_tool_calls = []
                            for index in sorted(current_tool_calls.keys()):
                                call = current_tool_calls[index]
                                function_name = call['function']['name']
                                arguments_str = call['function']['arguments']
                                arguments_str_sanitized = arguments_str if arguments_str.strip() else '{}'
                                final_tool_calls.append({'id': call['id'], 'type': 'function', 'function': {'name': function_name, 'arguments': arguments_str_sanitized}})
                            for tool_call in final_tool_calls:
                                args_value = tool_call['function']['arguments']
                                if not isinstance(args_value, str):
                                    args_value = self.formatter._serialize_tool_arguments(args_value)
                                captured_function_calls.append({'call_id': tool_call['id'], 'name': tool_call['function']['name'], 'arguments': args_value})
                            yield StreamChunk(type='tool_calls', tool_calls=final_tool_calls)
                            response_completed = True
                            break
                        elif choice.finish_reason in ['stop', 'length']:
                            response_completed = True
                            yield StreamChunk(type='done')
                            return
            except Exception as chunk_error:
                yield StreamChunk(type='error', error=f'Chunk processing error: {chunk_error}')
                continue
        if captured_function_calls and response_completed:
            non_mcp_functions = [call for call in captured_function_calls if call['name'] not in self._mcp_functions]
            if non_mcp_functions:
                logger.info(f'Non-MCP function calls detected (will be ignored in MCP execution): {[call['name'] for call in non_mcp_functions]}')
            if not await self._check_circuit_breaker_before_execution():
                yield StreamChunk(type='mcp_status', status='mcp_blocked', content='⚠️ [MCP] All servers blocked by circuit breaker', source='circuit_breaker')
                yield StreamChunk(type='done')
                return
            mcp_functions_executed = False
            updated_messages = current_messages.copy()
            if self.is_planning_mode_enabled():
                logger.info('[MCP] Planning mode enabled - blocking all MCP tool execution')
                yield StreamChunk(type='mcp_status', status='planning_mode_blocked', content='🚫 [MCP] Planning mode active - MCP tools blocked during coordination', source='planning_mode')
                yield StreamChunk(type='done')
                return
            if captured_function_calls:
                all_tool_calls = []
                for call in captured_function_calls:
                    all_tool_calls.append({'id': call['call_id'], 'type': 'function', 'function': {'name': call['name'], 'arguments': self.formatter._serialize_tool_arguments(call['arguments'])}})
                if all_tool_calls:
                    assistant_message = {'role': 'assistant', 'content': content.strip() if content.strip() else None, 'tool_calls': all_tool_calls}
                    updated_messages.append(assistant_message)
            tool_results = []
            for call in captured_function_calls:
                function_name = call['name']
                if self.is_mcp_tool_call(function_name):
                    yield StreamChunk(type='mcp_status', status='mcp_tool_called', content=f'🔧 [MCP Tool] Calling {function_name}...', source=f'mcp_{function_name}')
                    tools_info = f' ({len(self._mcp_functions)} tools available)' if self._mcp_functions else ''
                    yield StreamChunk(type='mcp_status', status='mcp_tools_initiated', content=f'MCP tool call initiated (call #{self._mcp_tool_calls_count}){tools_info}: {function_name}', source=f'mcp_{function_name}')
                    try:
                        result_str, result_obj = await self._execute_mcp_function_with_retry(function_name, call['arguments'])
                        if isinstance(result_str, str) and result_str.startswith('Error:'):
                            logger.warning(f'MCP function {function_name} failed after retries: {result_str}')
                            tool_results.append({'tool_call_id': call['call_id'], 'content': result_str, 'success': False})
                        else:
                            yield StreamChunk(type='mcp_status', status='mcp_tools_success', content=f'MCP tool call succeeded (call #{self._mcp_tool_calls_count})', source=f'mcp_{function_name}')
                            tool_results.append({'tool_call_id': call['call_id'], 'content': result_str, 'success': True, 'result_obj': result_obj})
                    except Exception as e:
                        logger.error(f'Unexpected error in MCP function execution: {e}')
                        error_msg = f'Error executing {function_name}: {str(e)}'
                        tool_results.append({'tool_call_id': call['call_id'], 'content': error_msg, 'success': False})
                        continue
                    yield StreamChunk(type='mcp_status', status='function_call', content=f'Arguments for Calling {function_name}: {call['arguments']}', source=f'mcp_{function_name}')
                    logger.info(f'Executed MCP function {function_name} (stdio/streamable-http)')
                    mcp_functions_executed = True
                else:
                    logger.info(f'Non-MCP function {function_name} detected, creating placeholder response')
                    tool_results.append({'tool_call_id': call['call_id'], 'content': f'Function {function_name} is not available in this MCP session.', 'success': False})
            for result in tool_results:
                result_text = str(result['content'])
                if result.get('success') and hasattr(result.get('result_obj'), 'content') and result['result_obj'].content:
                    obj = result['result_obj']
                    if isinstance(obj.content, list) and len(obj.content) > 0:
                        first_item = obj.content[0]
                        if hasattr(first_item, 'text'):
                            result_text = first_item.text
                yield StreamChunk(type='mcp_status', status='function_call_output', content=f'Results for Calling {function_name}: {result_text}', source=f'mcp_{function_name}')
                function_output_msg = {'role': 'tool', 'tool_call_id': result['tool_call_id'], 'content': result['content']}
                updated_messages.append(function_output_msg)
                yield StreamChunk(type='mcp_status', status='mcp_tool_response', content=f'✅ [MCP Tool] {function_name} completed', source=f'mcp_{function_name}')
            if mcp_functions_executed:
                updated_messages = self._trim_message_history(updated_messages)
                async for chunk in self._stream_with_mcp_tools(updated_messages, tools, client, **kwargs):
                    yield chunk
            else:
                yield StreamChunk(type='done')
                return
        elif response_completed:
            yield StreamChunk(type='mcp_status', status='mcp_session_complete', content='✅ [MCP] Session completed', source='mcp_session')
            return

    async def _process_stream(self, stream, all_params, agent_id) -> AsyncGenerator[StreamChunk, None]:
        """Handle standard Chat Completions API streaming format with logging."""
        content = ''
        current_tool_calls = {}
        search_sources_used = 0
        provider_name = self.get_provider_name()
        enable_web_search = all_params.get('enable_web_search', False)
        log_prefix = f'backend.{provider_name.lower().replace(' ', '_')}'
        async for chunk in stream:
            try:
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, 'delta') and choice.delta:
                        delta = choice.delta
                        if getattr(delta, 'content', None):
                            reasoning_chunk = self._handle_reasoning_transition(log_prefix, agent_id)
                            if reasoning_chunk:
                                yield reasoning_chunk
                            content_chunk = delta.content
                            content += content_chunk
                            log_backend_agent_message(agent_id or 'default', 'RECV', {'content': content_chunk}, backend_name=provider_name)
                            log_stream_chunk(log_prefix, 'content', content_chunk, agent_id)
                            yield StreamChunk(type='content', content=content_chunk)
                        if getattr(delta, 'reasoning_content', None):
                            reasoning_active_key = '_reasoning_active'
                            setattr(self, reasoning_active_key, True)
                            thinking_delta = getattr(delta, 'reasoning_content')
                            if thinking_delta:
                                log_stream_chunk(log_prefix, 'reasoning', thinking_delta, agent_id)
                                yield StreamChunk(type='reasoning', content=thinking_delta, reasoning_delta=thinking_delta)
                        if getattr(delta, 'tool_calls', None):
                            reasoning_chunk = self._handle_reasoning_transition(log_prefix, agent_id)
                            if reasoning_chunk:
                                yield reasoning_chunk
                            for tool_call_delta in delta.tool_calls:
                                index = getattr(tool_call_delta, 'index', 0)
                                if index not in current_tool_calls:
                                    current_tool_calls[index] = {'id': '', 'function': {'name': '', 'arguments': ''}}
                                if getattr(tool_call_delta, 'id', None):
                                    current_tool_calls[index]['id'] = tool_call_delta.id
                                if hasattr(tool_call_delta, 'function') and tool_call_delta.function:
                                    if getattr(tool_call_delta.function, 'name', None):
                                        current_tool_calls[index]['function']['name'] = tool_call_delta.function.name
                                    if getattr(tool_call_delta.function, 'arguments', None):
                                        current_tool_calls[index]['function']['arguments'] += tool_call_delta.function.arguments
                    if getattr(choice, 'finish_reason', None):
                        reasoning_chunk = self._handle_reasoning_transition(log_prefix, agent_id)
                        if reasoning_chunk:
                            yield reasoning_chunk
                        if choice.finish_reason == 'tool_calls' and current_tool_calls:
                            final_tool_calls = []
                            for index in sorted(current_tool_calls.keys()):
                                call = current_tool_calls[index]
                                function_name = call['function']['name']
                                arguments_str = call['function']['arguments']
                                arguments_str_sanitized = arguments_str if arguments_str.strip() else '{}'
                                final_tool_calls.append({'id': call['id'], 'type': 'function', 'function': {'name': function_name, 'arguments': arguments_str_sanitized}})
                            log_stream_chunk(log_prefix, 'tool_calls', final_tool_calls, agent_id)
                            yield StreamChunk(type='tool_calls', tool_calls=final_tool_calls)
                            complete_message = {'role': 'assistant', 'content': content.strip(), 'tool_calls': final_tool_calls}
                            yield StreamChunk(type='complete_message', complete_message=complete_message)
                            log_stream_chunk(log_prefix, 'done', None, agent_id)
                            yield StreamChunk(type='done')
                            return
                        elif choice.finish_reason in ['stop', 'length']:
                            if search_sources_used > 0:
                                search_complete_msg = f'\n✅ [Live Search Complete] Used {search_sources_used} sources\n'
                                log_stream_chunk(log_prefix, 'content', search_complete_msg, agent_id)
                                yield StreamChunk(type='content', content=search_complete_msg)
                            if hasattr(chunk, 'citations') and chunk.citations:
                                if enable_web_search:
                                    citation_text = '\n📚 **Citations:**\n'
                                    for i, citation in enumerate(chunk.citations, 1):
                                        citation_text += f'{i}. {citation}\n'
                                    log_stream_chunk(log_prefix, 'content', citation_text, agent_id)
                                    yield StreamChunk(type='content', content=citation_text)
                            complete_message = {'role': 'assistant', 'content': content.strip()}
                            yield StreamChunk(type='complete_message', complete_message=complete_message)
                            log_stream_chunk(log_prefix, 'done', None, agent_id)
                            yield StreamChunk(type='done')
                            return
                if hasattr(chunk, 'usage') and chunk.usage:
                    if getattr(chunk.usage, 'num_sources_used', 0) > 0:
                        search_sources_used = chunk.usage.num_sources_used
                        if enable_web_search:
                            search_msg = f'\n📊 [Live Search] Using {search_sources_used} sources for real-time data\n'
                            log_stream_chunk(log_prefix, 'content', search_msg, agent_id)
                            yield StreamChunk(type='content', content=search_msg)
            except Exception as chunk_error:
                error_msg = f'Chunk processing error: {chunk_error}'
                log_stream_chunk(log_prefix, 'error', error_msg, agent_id)
                yield StreamChunk(type='error', error=error_msg)
                continue
        log_stream_chunk(log_prefix, 'done', None, agent_id)
        yield StreamChunk(type='done')

    def create_tool_result_message(self, tool_call: Dict[str, Any], result_content: str) -> Dict[str, Any]:
        """Create tool result message for Chat Completions format."""
        tool_call_id = self.extract_tool_call_id(tool_call)
        return {'role': 'tool', 'tool_call_id': tool_call_id, 'content': result_content}

    def extract_tool_result_content(self, tool_result_message: Dict[str, Any]) -> str:
        """Extract content from Chat Completions tool result message."""
        return tool_result_message.get('content', '')

    def _convert_messages_for_mcp_chat_completions(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert messages for MCP Chat Completions format if needed."""
        converted_messages = []
        for message in messages:
            if message.get('type') == 'function_call_output':
                converted_message = {'role': 'tool', 'tool_call_id': message.get('call_id'), 'content': message.get('output', '')}
                converted_messages.append(converted_message)
            else:
                converted_messages.append(message.copy())
        return converted_messages

    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        if 'provider' in self.config:
            return self.config['provider']
        elif 'provider_name' in self.config:
            return self.config['provider_name']
        base_url = self.config.get('base_url', '')
        if 'openai.com' in base_url:
            return 'OpenAI'
        elif 'cerebras.ai' in base_url:
            return 'Cerebras AI'
        elif 'together.xyz' in base_url:
            return 'Together AI'
        elif 'fireworks.ai' in base_url:
            return 'Fireworks AI'
        elif 'groq.com' in base_url:
            return 'Groq'
        elif 'openrouter.ai' in base_url:
            return 'OpenRouter'
        elif 'z.ai' in base_url or 'bigmodel.cn' in base_url:
            return 'ZAI'
        elif 'nebius.com' in base_url:
            return 'Nebius AI Studio'
        elif 'moonshot.ai' in base_url or 'moonshot.cn' in base_url:
            return 'Kimi'
        elif 'poe.com' in base_url:
            return 'POE'
        elif 'aliyuncs.com' in base_url:
            return 'Qwen'
        else:
            return 'ChatCompletion'

    def get_filesystem_support(self) -> FilesystemSupport:
        """Chat Completions supports filesystem through MCP servers."""
        return FilesystemSupport.MCP

    def get_supported_builtin_tools(self) -> List[str]:
        """Get list of builtin tools supported by this provider."""
        return []

    def _create_client(self, **kwargs) -> AsyncOpenAI:
        """Create OpenAI client with consistent configuration."""
        import openai
        all_params = {**self.config, **kwargs}
        base_url = all_params.get('base_url', 'https://api.openai.com/v1')
        return openai.AsyncOpenAI(api_key=self.api_key, base_url=base_url)

    def _handle_reasoning_transition(self, log_prefix: str, agent_id: Optional[str]) -> Optional[StreamChunk]:
        """Handle reasoning state transition and return StreamChunk if transition occurred."""
        reasoning_active_key = '_reasoning_active'
        if hasattr(self, reasoning_active_key):
            if getattr(self, reasoning_active_key) is True:
                setattr(self, reasoning_active_key, False)
                log_stream_chunk(log_prefix, 'reasoning_done', '', agent_id)
                return StreamChunk(type='reasoning_done', content='')
        return None

class ResponseBackend(MCPBackend):
    """Backend using the standard Response API format with multimodal support."""

    def __init__(self, api_key: Optional[str]=None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.formatter = ResponseFormatter()
        self.api_params_handler = ResponseAPIParamsHandler(self)
        self._pending_image_saves = []
        self._vector_store_ids: List[str] = []
        self._uploaded_file_ids: List[str] = []

    def supports_upload_files(self) -> bool:
        return True

    async def stream_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Stream response using OpenAI Response API with unified MCP/non-MCP processing.

        Wraps parent implementation to ensure File Search cleanup happens after streaming completes.
        """
        try:
            async for chunk in super().stream_with_tools(messages, tools, **kwargs):
                yield chunk
        finally:
            await self._cleanup_file_search_if_needed(**kwargs)

    async def _cleanup_file_search_if_needed(self, **kwargs) -> None:
        """Cleanup File Search resources if needed."""
        if not (self._vector_store_ids or self._uploaded_file_ids):
            return
        agent_id = kwargs.get('agent_id')
        logger.info('Cleaning up File Search resources...')
        client = None
        try:
            client = self._create_client(**kwargs)
            await self._cleanup_file_search_resources(client, agent_id)
        except Exception as cleanup_error:
            logger.error(f'Error during File Search cleanup: {cleanup_error}', extra={'agent_id': agent_id})
        finally:
            if client and hasattr(client, 'aclose'):
                try:
                    await client.aclose()
                except Exception:
                    pass

    async def _stream_without_mcp_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], client, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        agent_id = kwargs.get('agent_id')
        all_params = {**self.config, **kwargs}
        processed_messages = await self._process_upload_files(messages, all_params)
        if all_params.get('_has_file_search_files'):
            logger.info('Processing File Search uploads...')
            processed_messages, vector_store_id = await self._upload_files_and_create_vector_store(processed_messages, client, agent_id)
            if vector_store_id:
                existing_ids = list(all_params.get('_file_search_vector_store_ids', []))
                existing_ids.append(vector_store_id)
                all_params['_file_search_vector_store_ids'] = existing_ids
                logger.info(f'File Search enabled with vector store: {vector_store_id}')
            all_params.pop('_has_file_search_files', None)
        api_params = await self.api_params_handler.build_api_params(processed_messages, tools, all_params)
        if 'tools' in api_params:
            non_mcp_tools = []
            for tool in api_params.get('tools', []):
                if tool.get('type') == 'function':
                    name = tool.get('function', {}).get('name') if 'function' in tool else tool.get('name')
                    if name and name in self._mcp_function_names:
                        continue
                elif tool.get('type') == 'mcp':
                    continue
                non_mcp_tools.append(tool)
            api_params['tools'] = non_mcp_tools
        stream = await client.responses.create(**api_params)
        async for chunk in self._process_stream(stream, all_params, agent_id):
            yield chunk

    async def _stream_with_mcp_tools(self, current_messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], client, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Recursively stream MCP responses, executing function calls as needed."""
        agent_id = kwargs.get('agent_id')
        all_params = {**self.config, **kwargs}
        if all_params.get('_has_file_search_files'):
            logger.info('Processing File Search uploads...')
            current_messages, vector_store_id = await self._upload_files_and_create_vector_store(current_messages, client, agent_id)
            if vector_store_id:
                existing_ids = list(all_params.get('_file_search_vector_store_ids', []))
                existing_ids.append(vector_store_id)
                all_params['_file_search_vector_store_ids'] = existing_ids
                logger.info(f'File Search enabled with vector store: {vector_store_id}')
            all_params.pop('_has_file_search_files', None)
        api_params = await self.api_params_handler.build_api_params(current_messages, tools, all_params)
        stream = await client.responses.create(**api_params)
        captured_function_calls = []
        current_function_call = None
        response_completed = False
        async for chunk in stream:
            if hasattr(chunk, 'type'):
                if chunk.type == 'response.output_item.added' and hasattr(chunk, 'item') and chunk.item and (getattr(chunk.item, 'type', None) == 'function_call'):
                    current_function_call = {'call_id': getattr(chunk.item, 'call_id', ''), 'name': getattr(chunk.item, 'name', ''), 'arguments': ''}
                    logger.info(f'Function call detected: {current_function_call['name']}')
                elif chunk.type == 'response.function_call_arguments.delta' and current_function_call is not None:
                    delta = getattr(chunk, 'delta', '')
                    current_function_call['arguments'] += delta
                elif chunk.type == 'response.output_item.done' and current_function_call is not None:
                    captured_function_calls.append(current_function_call)
                    current_function_call = None
                elif chunk.type == 'response.output_text.delta':
                    delta = getattr(chunk, 'delta', '')
                    yield TextStreamChunk(type=ChunkType.CONTENT, content=delta, source='response_api')
                else:
                    result = self._process_stream_chunk(chunk, agent_id)
                    yield result
                if chunk.type == 'response.completed':
                    response_completed = True
                    if captured_function_calls:
                        break
                    else:
                        yield TextStreamChunk(type=ChunkType.DONE, source='response_api')
                        return
        if captured_function_calls and response_completed:
            non_mcp_functions = [call for call in captured_function_calls if call['name'] not in self._mcp_functions]
            if non_mcp_functions:
                logger.info(f'Non-MCP function calls detected: {[call['name'] for call in non_mcp_functions]}. Ending MCP processing.')
                yield TextStreamChunk(type=ChunkType.DONE, source='response_api')
                return
            if not await super()._check_circuit_breaker_before_execution():
                logger.warning('All MCP servers blocked by circuit breaker')
                yield TextStreamChunk(type=ChunkType.MCP_STATUS, status='mcp_blocked', content='⚠️ [MCP] All servers blocked by circuit breaker', source='circuit_breaker')
                yield TextStreamChunk(type=ChunkType.DONE, source='response_api')
                return
            mcp_functions_executed = False
            updated_messages = current_messages.copy()
            if self.is_planning_mode_enabled():
                logger.info('[MCP] Planning mode enabled - blocking all MCP tool execution')
                yield StreamChunk(type='mcp_status', status='planning_mode_blocked', content='🚫 [MCP] Planning mode active - MCP tools blocked during coordination', source='planning_mode')
                yield StreamChunk(type='done')
                return
            processed_call_ids = set()
            for call in captured_function_calls:
                function_name = call['name']
                if function_name in self._mcp_functions:
                    yield TextStreamChunk(type=ChunkType.MCP_STATUS, status='mcp_tool_called', content=f'🔧 [MCP Tool] Calling {function_name}...', source=f'mcp_{function_name}')
                    try:
                        result, result_obj = await super()._execute_mcp_function_with_retry(function_name, call['arguments'])
                        if isinstance(result, str) and result.startswith('Error:'):
                            logger.warning(f'MCP function {function_name} failed after retries: {result}')
                            function_call_msg = {'type': 'function_call', 'call_id': call['call_id'], 'name': function_name, 'arguments': call['arguments']}
                            updated_messages.append(function_call_msg)
                            error_output_msg = {'type': 'function_call_output', 'call_id': call['call_id'], 'output': result}
                            updated_messages.append(error_output_msg)
                            processed_call_ids.add(call['call_id'])
                            mcp_functions_executed = True
                            continue
                    except Exception as e:
                        logger.error(f'Unexpected error in MCP function execution: {e}')
                        error_msg = f'Error executing {function_name}: {str(e)}'
                        function_call_msg = {'type': 'function_call', 'call_id': call['call_id'], 'name': function_name, 'arguments': call['arguments']}
                        updated_messages.append(function_call_msg)
                        error_output_msg = {'type': 'function_call_output', 'call_id': call['call_id'], 'output': error_msg}
                        updated_messages.append(error_output_msg)
                        processed_call_ids.add(call['call_id'])
                        mcp_functions_executed = True
                        continue
                    function_call_msg = {'type': 'function_call', 'call_id': call['call_id'], 'name': function_name, 'arguments': call['arguments']}
                    updated_messages.append(function_call_msg)
                    yield TextStreamChunk(type=ChunkType.MCP_STATUS, status='function_call', content=f'Arguments for Calling {function_name}: {call['arguments']}', source=f'mcp_{function_name}')
                    function_output_msg = {'type': 'function_call_output', 'call_id': call['call_id'], 'output': str(result)}
                    updated_messages.append(function_output_msg)
                    yield TextStreamChunk(type=ChunkType.MCP_STATUS, status='function_call_output', content=f'Results for Calling {function_name}: {str(result_obj.content[0].text)}', source=f'mcp_{function_name}')
                    logger.info(f'Executed MCP function {function_name} (stdio/streamable-http)')
                    processed_call_ids.add(call['call_id'])
                    yield TextStreamChunk(type=ChunkType.MCP_STATUS, status='mcp_tool_response', content=f'✅ [MCP Tool] {function_name} completed', source=f'mcp_{function_name}')
                    mcp_functions_executed = True
            for call in captured_function_calls:
                if call['call_id'] not in processed_call_ids:
                    logger.warning(f'Tool call {call['call_id']} for function {call['name']} was not processed - adding error result')
                    function_call_msg = {'type': 'function_call', 'call_id': call['call_id'], 'name': call['name'], 'arguments': call['arguments']}
                    updated_messages.append(function_call_msg)
                    error_output_msg = {'type': 'function_call_output', 'call_id': call['call_id'], 'output': f'Error: Tool call {call['call_id']} for function {call['name']} was not processed. This may indicate a validation or execution error.'}
                    updated_messages.append(error_output_msg)
                    mcp_functions_executed = True
            if mcp_functions_executed:
                updated_messages = super()._trim_message_history(updated_messages)
                async for chunk in self._stream_with_mcp_tools(updated_messages, tools, client, **kwargs):
                    yield chunk
            else:
                yield TextStreamChunk(type=ChunkType.DONE, source='response_api')
                return
        elif response_completed:
            yield TextStreamChunk(type=ChunkType.MCP_STATUS, status='mcp_session_complete', content='✅ [MCP] Session completed', source='mcp_session')
            yield TextStreamChunk(type=ChunkType.DONE, source='response_api')
            return

    async def _upload_files_and_create_vector_store(self, messages: List[Dict[str, Any]], client: AsyncOpenAI, agent_id: Optional[str]=None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Upload file_pending_upload items and create a vector store."""
        try:
            pending_files: List[Dict[str, Any]] = []
            file_locations: List[Tuple[int, int]] = []
            for message_index, message in enumerate(messages):
                content = message.get('content')
                if not isinstance(content, list):
                    continue
                for item_index, item in enumerate(content):
                    if isinstance(item, dict) and item.get('type') == 'file_pending_upload':
                        pending_files.append(item)
                        file_locations.append((message_index, item_index))
            if not pending_files:
                return (messages, None)
            uploaded_file_ids: List[str] = []
            http_client: Optional[httpx.AsyncClient] = None
            try:
                for pending in pending_files:
                    source = pending.get('source')
                    if source == 'local':
                        path_str = pending.get('path')
                        if not path_str:
                            logger.warning('Missing local path for file_pending_upload entry')
                            continue
                        file_path = Path(path_str)
                        if not file_path.exists():
                            raise UploadFileError(f'File not found for upload: {file_path}')
                        try:
                            with file_path.open('rb') as file_handle:
                                uploaded_file = await client.files.create(purpose='assistants', file=file_handle)
                        except Exception as exc:
                            raise UploadFileError(f'Failed to upload file {file_path}: {exc}') from exc
                    elif source == 'url':
                        file_url = pending.get('url')
                        if not file_url:
                            logger.warning('Missing URL for file_pending_upload entry')
                            continue
                        parsed = urlparse(file_url)
                        if parsed.scheme not in {'http', 'https'}:
                            raise UploadFileError(f'Unsupported URL scheme for file upload: {file_url}')
                        if http_client is None:
                            http_client = httpx.AsyncClient()
                        try:
                            response = await http_client.get(file_url, timeout=30.0)
                            response.raise_for_status()
                        except httpx.HTTPError as exc:
                            raise UploadFileError(f'Failed to download file from URL {file_url}: {exc}') from exc
                        filename = Path(parsed.path).name or 'remote_file'
                        file_bytes = BytesIO(response.content)
                        try:
                            uploaded_file = await client.files.create(purpose='assistants', file=(filename, file_bytes))
                        except Exception as exc:
                            raise UploadFileError(f'Failed to upload file from URL {file_url}: {exc}') from exc
                    else:
                        raise UploadFileError(f'Unknown file_pending_upload source: {source}')
                    file_id = getattr(uploaded_file, 'id', None)
                    if not file_id:
                        raise UploadFileError('Uploaded file response missing ID')
                    uploaded_file_ids.append(file_id)
                    self._uploaded_file_ids.append(file_id)
                    logger.info(f'Uploaded file for File Search (file_id={file_id})')
            finally:
                if http_client is not None:
                    await http_client.aclose()
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            vector_store_name = f'massgen_file_search_{agent_id or 'default'}_{timestamp}'
            try:
                vector_store = await client.vector_stores.create(name=vector_store_name)
            except Exception as exc:
                raise UploadFileError(f'Failed to create vector store: {exc}') from exc
            vector_store_id = getattr(vector_store, 'id', None)
            if not vector_store_id:
                raise UploadFileError('Vector store response missing ID')
            self._vector_store_ids.append(vector_store_id)
            logger.info('Created vector store for File Search', extra={'vector_store_id': vector_store_id, 'file_count': len(uploaded_file_ids)})
            for file_id in uploaded_file_ids:
                try:
                    vs_file = await client.vector_stores.files.create_and_poll(vector_store_id=vector_store_id, file_id=file_id)
                    logger.info('File indexed and attached to vector store', extra={'vector_store_id': vector_store_id, 'file_id': file_id, 'status': getattr(vs_file, 'status', None)})
                except Exception as exc:
                    raise UploadFileError(f'Failed to attach and index file {file_id} to vector store {vector_store_id}: {exc}') from exc
            if uploaded_file_ids:
                logger.info('All files indexed for File Search; waiting 2s for vector store to stabilize', extra={'vector_store_id': vector_store_id, 'file_count': len(uploaded_file_ids)})
                await asyncio.sleep(2)
            updated_messages = []
            for message in messages:
                cloned = dict(message)
                if isinstance(message.get('content'), list):
                    cloned['content'] = [dict(item) if isinstance(item, dict) else item for item in message['content']]
                updated_messages.append(cloned)
            for message_index, item_index in reversed(file_locations):
                content_list = updated_messages[message_index].get('content')
                if isinstance(content_list, list):
                    content_list.pop(item_index)
                    if not content_list:
                        content_list.append({'type': 'text', 'text': '[Files uploaded for search integration]'})
            return (updated_messages, vector_store_id)
        except Exception as error:
            logger.warning(f'File Search upload failed: {error}. Continuing without file search.')
            return (messages, None)

    async def _cleanup_file_search_resources(self, client: AsyncOpenAI, agent_id: Optional[str]=None) -> None:
        """Clean up File Search vector stores and uploaded files."""
        for vector_store_id in list(self._vector_store_ids):
            try:
                await client.vector_stores.delete(vector_store_id)
                logger.info('Deleted File Search vector store', extra={'vector_store_id': vector_store_id, 'agent_id': agent_id})
            except Exception as exc:
                logger.warning(f'Failed to delete vector store {vector_store_id}: {exc}', extra={'agent_id': agent_id})
        for file_id in list(self._uploaded_file_ids):
            try:
                await client.files.delete(file_id)
                logger.debug('Deleted File Search uploaded file', extra={'file_id': file_id, 'agent_id': agent_id})
            except Exception as exc:
                logger.warning(f'Failed to delete file {file_id}: {exc}', extra={'agent_id': agent_id})
        self._vector_store_ids.clear()
        self._uploaded_file_ids.clear()

    def _convert_mcp_tools_to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert MCP tools (stdio + streamable-http) to OpenAI function declarations."""
        if not self._mcp_functions:
            return []
        converted_tools = []
        for function in self._mcp_functions.values():
            converted_tools.append(function.to_openai_format())
        logger.debug(f'Converted {len(converted_tools)} MCP tools (stdio + streamable-http) to OpenAI format')
        return converted_tools

    async def _process_stream(self, stream, all_params, agent_id=None):
        async for chunk in stream:
            processed = self._process_stream_chunk(chunk, agent_id)
            if processed.type == 'complete_response':
                yield processed
                log_stream_chunk('backend.response', 'done', None, agent_id)
                yield TextStreamChunk(type=ChunkType.DONE, source='response_api')
            else:
                yield processed

    def _process_stream_chunk(self, chunk, agent_id) -> Union[TextStreamChunk, StreamChunk]:
        """
        Process individual stream chunks and convert to appropriate chunk format.

        Returns TextStreamChunk for text/reasoning/tool content,
        or legacy StreamChunk for backward compatibility.
        """
        if not hasattr(chunk, 'type'):
            return StreamChunk(type='content', content='')
        chunk_type = chunk.type
        if chunk_type == 'response.output_text.delta' and hasattr(chunk, 'delta'):
            log_backend_agent_message(agent_id or 'default', 'RECV', {'content': chunk.delta}, backend_name=self.get_provider_name())
            log_stream_chunk('backend.response', 'content', chunk.delta, agent_id)
            return TextStreamChunk(type=ChunkType.CONTENT, content=chunk.delta, source='response_api')
        elif chunk_type == 'response.reasoning_text.delta' and hasattr(chunk, 'delta'):
            log_stream_chunk('backend.response', 'reasoning', chunk.delta, agent_id)
            return TextStreamChunk(type=ChunkType.REASONING, content=f'🧠 [Reasoning] {chunk.delta}', reasoning_delta=chunk.delta, item_id=getattr(chunk, 'item_id', None), content_index=getattr(chunk, 'content_index', None), source='response_api')
        elif chunk_type == 'response.reasoning_text.done':
            reasoning_text = getattr(chunk, 'text', '')
            log_stream_chunk('backend.response', 'reasoning_done', reasoning_text, agent_id)
            return TextStreamChunk(type=ChunkType.REASONING_DONE, content='\n🧠 [Reasoning Complete]\n', reasoning_text=reasoning_text, item_id=getattr(chunk, 'item_id', None), content_index=getattr(chunk, 'content_index', None), source='response_api')
        elif chunk_type == 'response.reasoning_summary_text.delta' and hasattr(chunk, 'delta'):
            log_stream_chunk('backend.response', 'reasoning_summary', chunk.delta, agent_id)
            return TextStreamChunk(type=ChunkType.REASONING_SUMMARY, content=chunk.delta, reasoning_summary_delta=chunk.delta, item_id=getattr(chunk, 'item_id', None), summary_index=getattr(chunk, 'summary_index', None), source='response_api')
        elif chunk_type == 'response.reasoning_summary_text.done':
            summary_text = getattr(chunk, 'text', '')
            log_stream_chunk('backend.response', 'reasoning_summary_done', summary_text, agent_id)
            return TextStreamChunk(type=ChunkType.REASONING_SUMMARY_DONE, content='\n📋 [Reasoning Summary Complete]\n', reasoning_summary_text=summary_text, item_id=getattr(chunk, 'item_id', None), summary_index=getattr(chunk, 'summary_index', None), source='response_api')
        elif chunk_type == 'response.file_search_call.in_progress':
            item_id = getattr(chunk, 'item_id', None)
            output_index = getattr(chunk, 'output_index', None)
            log_stream_chunk('backend.response', 'file_search', 'Starting file search', agent_id)
            return TextStreamChunk(type=ChunkType.CONTENT, content='\n📁 [File Search] Starting search...', item_id=item_id, content_index=output_index, source='response_api')
        elif chunk_type == 'response.file_search_call.searching':
            item_id = getattr(chunk, 'item_id', None)
            output_index = getattr(chunk, 'output_index', None)
            queries = getattr(chunk, 'queries', None)
            query_text = ''
            if queries:
                try:
                    if isinstance(queries, (list, tuple)):
                        query_text = ', '.join((str(q) for q in queries if q))
                    else:
                        query_text = str(queries)
                except Exception:
                    query_text = ''
            message = '\n📁 [File Search] Searching...'
            if query_text:
                message += f' Query: {query_text}'
            log_stream_chunk('backend.response', 'file_search', f'Searching files{(f' for {query_text}' if query_text else '')}', agent_id)
            return TextStreamChunk(type=ChunkType.CONTENT, content=message, item_id=item_id, content_index=output_index, source='response_api')
        elif chunk_type == 'response.file_search_call.completed':
            item_id = getattr(chunk, 'item_id', None)
            output_index = getattr(chunk, 'output_index', None)
            results = getattr(chunk, 'results', None)
            if results is None:
                results = getattr(chunk, 'search_results', None)
            queries = getattr(chunk, 'queries', None)
            query_text = ''
            if queries:
                try:
                    if isinstance(queries, (list, tuple)):
                        query_text = ', '.join((str(q) for q in queries if q))
                    else:
                        query_text = str(queries)
                except Exception:
                    query_text = ''
            if results is not None:
                try:
                    result_count = len(results)
                except Exception:
                    result_count = None
            else:
                result_count = None
            message_parts = ['\n✅ [File Search] Completed']
            if query_text:
                message_parts.append(f'Query: {query_text}')
            if result_count is not None:
                message_parts.append(f'Results: {result_count}')
            message = ' '.join(message_parts)
            log_stream_chunk('backend.response', 'file_search', f'Completed file search{(f' for {query_text}' if query_text else '')}{(f' with {result_count} results' if result_count is not None else '')}', agent_id)
            return TextStreamChunk(type=ChunkType.CONTENT, content=message, item_id=item_id, content_index=output_index, source='response_api')
        elif chunk_type == 'response.web_search_call.in_progress':
            log_stream_chunk('backend.response', 'web_search', 'Starting search', agent_id)
            return TextStreamChunk(type=ChunkType.CONTENT, content='\n🔍 [Provider Tool: Web Search] Starting search...', source='response_api')
        elif chunk_type == 'response.web_search_call.searching':
            log_stream_chunk('backend.response', 'web_search', 'Searching', agent_id)
            return TextStreamChunk(type=ChunkType.CONTENT, content='\n🔍 [Provider Tool: Web Search] Searching...', source='response_api')
        elif chunk_type == 'response.web_search_call.completed':
            log_stream_chunk('backend.response', 'web_search', 'Search completed', agent_id)
            return TextStreamChunk(type=ChunkType.CONTENT, content='\n✅ [Provider Tool: Web Search] Search completed', source='response_api')
        elif chunk_type == 'response.code_interpreter_call.in_progress':
            log_stream_chunk('backend.response', 'code_interpreter', 'Starting execution', agent_id)
            return TextStreamChunk(type=ChunkType.CONTENT, content='\n💻 [Provider Tool: Code Interpreter] Starting execution...', source='response_api')
        elif chunk_type == 'response.code_interpreter_call.executing':
            log_stream_chunk('backend.response', 'code_interpreter', 'Executing', agent_id)
            return TextStreamChunk(type=ChunkType.CONTENT, content='\n💻 [Provider Tool: Code Interpreter] Executing...', source='response_api')
        elif chunk_type == 'response.code_interpreter_call.completed':
            log_stream_chunk('backend.response', 'code_interpreter', 'Execution completed', agent_id)
            return TextStreamChunk(type=ChunkType.CONTENT, content='\n✅ [Provider Tool: Code Interpreter] Execution completed', source='response_api')
        elif chunk_type == 'response.image_generation_call.in_progress':
            log_stream_chunk('backend.response', 'image_generation', 'Starting image generation', agent_id)
            return TextStreamChunk(type=ChunkType.CONTENT, content='\n🎨 [Provider Tool: Image Generation] Starting generation...', source='response_api')
        elif chunk_type == 'response.image_generation_call.generating':
            log_stream_chunk('backend.response', 'image_generation', 'Generating image', agent_id)
            return TextStreamChunk(type=ChunkType.CONTENT, content='\n🎨 [Provider Tool: Image Generation] Generating image...', source='response_api')
        elif chunk_type == 'response.image_generation_call.completed':
            log_stream_chunk('backend.response', 'image_generation', 'Image generation completed', agent_id)
            return TextStreamChunk(type=ChunkType.CONTENT, content='\n✅ [Provider Tool: Image Generation] Image generated successfully', source='response_api')
        elif chunk_type == 'image_generation.completed':
            if hasattr(chunk, 'b64_json'):
                log_stream_chunk('backend.response', 'image_generation', 'Image data received', agent_id)
                return TextStreamChunk(type=ChunkType.CONTENT, content='\n✅ [Image Generation] Image successfully created', source='response_api')
        elif chunk.type == 'response.output_item.done':
            if hasattr(chunk, 'item') and chunk.item:
                if hasattr(chunk.item, 'type') and chunk.item.type == 'web_search_call':
                    if hasattr(chunk.item, 'action') and 'query' in chunk.item.action:
                        search_query = chunk.item.action['query']
                        if search_query:
                            log_stream_chunk('backend.response', 'search_query', search_query, agent_id)
                            return TextStreamChunk(type=ChunkType.CONTENT, content=f"\n🔍 [Search Query] '{search_query}'\n", source='response_api')
                elif hasattr(chunk.item, 'type') and chunk.item.type == 'code_interpreter_call':
                    if hasattr(chunk.item, 'code') and chunk.item.code:
                        log_stream_chunk('backend.response', 'code_executed', chunk.item.code, agent_id)
                        return TextStreamChunk(type=ChunkType.CONTENT, content=f'💻 [Code Executed]\n```\n{chunk.item.code}\n```\n', source='response_api')
                    if hasattr(chunk.item, 'outputs') and chunk.item.outputs:
                        for output in chunk.item.outputs:
                            output_text = None
                            if hasattr(output, 'text') and output.text:
                                output_text = output.text
                            elif hasattr(output, 'content') and output.content:
                                output_text = output.content
                            elif hasattr(output, 'data') and output.data:
                                output_text = str(output.data)
                            elif isinstance(output, str):
                                output_text = output
                            elif isinstance(output, dict):
                                if 'text' in output:
                                    output_text = output['text']
                                elif 'content' in output:
                                    output_text = output['content']
                                elif 'data' in output:
                                    output_text = str(output['data'])
                            if output_text and output_text.strip():
                                log_stream_chunk('backend.response', 'code_result', output_text.strip(), agent_id)
                                return TextStreamChunk(type=ChunkType.CONTENT, content=f'📊 [Result] {output_text.strip()}\n', source='response_api')
                elif hasattr(chunk.item, 'type') and chunk.item.type == 'image_generation_call':
                    if hasattr(chunk.item, 'action') and chunk.item.action:
                        prompt = chunk.item.action.get('prompt', '')
                        size = chunk.item.action.get('size', '1024x1024')
                        if prompt:
                            log_stream_chunk('backend.response', 'image_prompt', prompt, agent_id)
                            return TextStreamChunk(type=ChunkType.CONTENT, content=f"\n🎨 [Image Generated] Prompt: '{prompt}' (Size: {size})\n", source='response_api')
        elif chunk_type == 'response.mcp_list_tools.started':
            return TextStreamChunk(type=ChunkType.MCP_STATUS, content='\n🔧 [MCP] Listing available tools...', source='response_api')
        elif chunk_type == 'response.mcp_list_tools.completed':
            return TextStreamChunk(type=ChunkType.MCP_STATUS, content='\n✅ [MCP] Tool listing completed', source='response_api')
        elif chunk_type == 'response.mcp_list_tools.failed':
            return TextStreamChunk(type=ChunkType.MCP_STATUS, content='\n❌ [MCP] Tool listing failed', source='response_api')
        elif chunk_type == 'response.mcp_call.started':
            tool_name = getattr(chunk, 'tool_name', 'unknown')
            return TextStreamChunk(type=ChunkType.MCP_STATUS, content=f"\n🔧 [MCP] Calling tool '{tool_name}'...", source='response_api')
        elif chunk_type == 'response.mcp_call.in_progress':
            return TextStreamChunk(type=ChunkType.MCP_STATUS, content='\n⏳ [MCP] Tool execution in progress...', source='response_api')
        elif chunk_type == 'response.mcp_call.completed':
            tool_name = getattr(chunk, 'tool_name', 'unknown')
            return TextStreamChunk(type=ChunkType.MCP_STATUS, content=f"\n✅ [MCP] Tool '{tool_name}' completed", source='response_api')
        elif chunk_type == 'response.mcp_call.failed':
            tool_name = getattr(chunk, 'tool_name', 'unknown')
            error_msg = getattr(chunk, 'error', 'unknown error')
            return TextStreamChunk(type=ChunkType.MCP_STATUS, content=f"\n❌ [MCP] Tool '{tool_name}' failed: {error_msg}", source='response_api')
        elif chunk.type == 'response.completed':
            if hasattr(chunk, 'response'):
                response_dict = self._convert_to_dict(chunk.response)
                if isinstance(response_dict, dict) and 'output' in response_dict:
                    for item in response_dict['output']:
                        if item.get('type') == 'code_interpreter_call':
                            status = item.get('status', 'unknown')
                            code = item.get('code', '')
                            outputs = item.get('outputs')
                            content = f'\n🔧 Code Interpreter [{status.title()}]'
                            if code:
                                content += f': {code}'
                            if outputs:
                                content += f' → {outputs}'
                            log_stream_chunk('backend.response', 'code_interpreter_result', content, agent_id)
                            return TextStreamChunk(type=ChunkType.CONTENT, content=content, source='response_api')
                        elif item.get('type') == 'web_search_call':
                            status = item.get('status', 'unknown')
                            query = item.get('action', {}).get('query', '')
                            results = item.get('results')
                            if query:
                                content = f'\n🔧 Web Search [{status.title()}]: {query}'
                                if results:
                                    content += f' → Found {len(results)} results'
                                log_stream_chunk('backend.response', 'web_search_result', content, agent_id)
                                return TextStreamChunk(type=ChunkType.CONTENT, content=content, source='response_api')
                        elif item.get('type') == 'image_generation_call':
                            status = item.get('status', 'unknown')
                            action = item.get('action', {})
                            prompt = action.get('prompt', '')
                            size = action.get('size', '1024x1024')
                            if prompt:
                                content = f'\n🔧 Image Generation [{status.title()}]: {prompt} (Size: {size})'
                                log_stream_chunk('backend.response', 'image_generation_result', content, agent_id)
                                return TextStreamChunk(type=ChunkType.CONTENT, content=content, source='response_api')
                log_stream_chunk('backend.response', 'complete_response', 'Response completed', agent_id)
                return TextStreamChunk(type=ChunkType.COMPLETE_RESPONSE, response=response_dict, source='response_api')
        return StreamChunk(type='content', content='')

    def create_tool_result_message(self, tool_call: Dict[str, Any], result_content: str) -> Dict[str, Any]:
        """Create tool result message for OpenAI Responses API format."""
        tool_call_id = self.extract_tool_call_id(tool_call)
        return {'type': 'function_call_output', 'call_id': tool_call_id, 'output': result_content}

    def extract_tool_result_content(self, tool_result_message: Dict[str, Any]) -> str:
        """Extract content from OpenAI Responses API tool result message."""
        return tool_result_message.get('output', '')

    def _create_client(self, **kwargs) -> AsyncOpenAI:
        return openai.AsyncOpenAI(api_key=self.api_key)

    def _convert_to_dict(self, obj) -> Dict[str, Any]:
        """Convert any object to dictionary with multiple fallback methods."""
        try:
            if hasattr(obj, 'model_dump'):
                return obj.model_dump()
            elif hasattr(obj, 'dict'):
                return obj.dict()
            else:
                return dict(obj)
        except Exception:
            return {key: getattr(obj, key, None) for key in dir(obj) if not key.startswith('_') and (not callable(getattr(obj, key, None)))}

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return 'OpenAI'

    def get_filesystem_support(self) -> FilesystemSupport:
        """OpenAI supports filesystem through MCP servers."""
        return FilesystemSupport.MCP

    def get_supported_builtin_tools(self) -> List[str]:
        """Get list of builtin tools supported by OpenAI."""
        return ['web_search', 'code_interpreter']

class GeminiBackend(LLMBackend):
    """Google Gemini backend using structured output for coordination and MCP tool integration."""

    def __init__(self, api_key: Optional[str]=None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        self.search_count = 0
        self.code_execution_count = 0
        self.mcp_servers = self.config.get('mcp_servers', [])
        self.allowed_tools = kwargs.pop('allowed_tools', None)
        self.exclude_tools = kwargs.pop('exclude_tools', None)
        self._mcp_client: Optional[MCPClient] = None
        self._mcp_initialized = False
        self._mcp_tool_calls_count = 0
        self._mcp_tool_failures = 0
        self._mcp_tool_successes = 0
        self.mcp_extractor = MCPResponseExtractor()
        self._max_mcp_message_history = kwargs.pop('max_mcp_message_history', 200)
        self._mcp_connection_retries = 0
        self._circuit_breakers_enabled = kwargs.pop('circuit_breaker_enabled', True)
        self._mcp_tools_circuit_breaker = None
        self.agent_id = kwargs.get('agent_id', None)
        if self._circuit_breakers_enabled:
            if MCPCircuitBreakerManager is None:
                raise RuntimeError('Circuit breakers enabled but MCPCircuitBreakerManager is not available')
            try:
                from ..mcp_tools.circuit_breaker import MCPCircuitBreaker
                if MCPConfigHelper is not None:
                    mcp_tools_config = MCPConfigHelper.build_circuit_breaker_config('mcp_tools', backend_name='gemini')
                else:
                    mcp_tools_config = None
                if mcp_tools_config:
                    self._mcp_tools_circuit_breaker = MCPCircuitBreaker(mcp_tools_config, backend_name='gemini', agent_id=self.agent_id)
                    log_backend_activity('gemini', 'Circuit breaker initialized for MCP tools', {'enabled': True}, agent_id=self.agent_id)
                else:
                    log_backend_activity('gemini', 'Circuit breaker config unavailable', {'fallback': 'disabled'}, agent_id=self.agent_id)
                    self._circuit_breakers_enabled = False
            except ImportError:
                log_backend_activity('gemini', 'Circuit breaker import failed', {'fallback': 'disabled'}, agent_id=self.agent_id)
                self._circuit_breakers_enabled = False

    def _setup_permission_hooks(self):
        """Override base class - Gemini uses session-based permissions, not function hooks."""
        logger.debug('[Gemini] Using session-based permissions, skipping function hook setup')

    async def _setup_mcp_with_status_stream(self, agent_id: Optional[str]=None) -> AsyncGenerator[StreamChunk, None]:
        """Initialize MCP client with status streaming."""
        status_queue: asyncio.Queue[StreamChunk] = asyncio.Queue()

        async def status_callback(status: str, details: Dict[str, Any]) -> None:
            """Callback to queue status updates as StreamChunks."""
            chunk = StreamChunk(type='mcp_status', status=status, content=details.get('message', ''), source='mcp_tools')
            await status_queue.put(chunk)
        setup_task = asyncio.create_task(self._setup_mcp_tools_internal(agent_id, status_callback))
        while not setup_task.done():
            try:
                chunk = await asyncio.wait_for(status_queue.get(), timeout=0.1)
                yield chunk
            except asyncio.TimeoutError:
                continue
        try:
            await setup_task
        except Exception as e:
            yield StreamChunk(type='mcp_status', status='error', content=f'MCP setup failed: {e}', source='mcp_tools')

    async def _setup_mcp_tools(self, agent_id: Optional[str]=None) -> None:
        """Initialize MCP client (sessions only) - backward compatibility."""
        if not self.mcp_servers or self._mcp_initialized:
            return
        async for _ in self._setup_mcp_with_status_stream(agent_id):
            pass

    async def _setup_mcp_tools_internal(self, agent_id: Optional[str]=None, status_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]]=None) -> None:
        """Internal MCP setup logic."""
        if not self.mcp_servers or self._mcp_initialized:
            return
        if MCPClient is None:
            reason = 'MCP import failed - MCPClient not available'
            log_backend_activity('gemini', 'MCP import failed', {'reason': reason, 'fallback': 'workflow_tools'}, agent_id=agent_id)
            if status_callback:
                await status_callback('error', {'message': 'MCP import failed - falling back to workflow tools'})
            self.mcp_servers = []
            return
        try:
            validated_config = {'mcp_servers': self.mcp_servers, 'allowed_tools': self.allowed_tools, 'exclude_tools': self.exclude_tools}
            if MCPConfigValidator is not None:
                try:
                    backend_config = {'mcp_servers': self.mcp_servers, 'allowed_tools': self.allowed_tools, 'exclude_tools': self.exclude_tools}
                    validator = MCPConfigValidator()
                    validated_config = validator.validate_backend_mcp_config(backend_config)
                    self.mcp_servers = validated_config.get('mcp_servers', self.mcp_servers)
                    log_backend_activity('gemini', 'MCP configuration validated', {'server_count': len(self.mcp_servers)}, agent_id=agent_id)
                    if status_callback:
                        await status_callback('info', {'message': f'MCP configuration validated: {len(self.mcp_servers)} servers'})
                    if True:
                        server_names = [server.get('name', 'unnamed') for server in self.mcp_servers]
                        log_backend_activity('gemini', 'MCP servers validated', {'servers': server_names}, agent_id=agent_id)
                except MCPConfigurationError as e:
                    log_backend_activity('gemini', 'MCP configuration validation failed', {'error': e.original_message}, agent_id=agent_id)
                    if status_callback:
                        await status_callback('error', {'message': f'Invalid MCP configuration: {e.original_message}'})
                    self._mcp_client = None
                    raise RuntimeError(f'Invalid MCP configuration: {e.original_message}') from e
                except MCPValidationError as e:
                    log_backend_activity('gemini', 'MCP validation failed', {'error': e.original_message}, agent_id=agent_id)
                    if status_callback:
                        await status_callback('error', {'message': f'MCP validation error: {e.original_message}'})
                    self._mcp_client = None
                    raise RuntimeError(f'MCP validation error: {e.original_message}') from e
                except Exception as e:
                    if isinstance(e, (ImportError, AttributeError)):
                        log_backend_activity('gemini', 'MCP validation unavailable', {'reason': str(e)}, agent_id=agent_id)
                    else:
                        log_backend_activity('gemini', 'MCP validation error', {'error': str(e)}, agent_id=agent_id)
                        self._mcp_client = None
                        raise RuntimeError(f'MCP configuration validation failed: {e}') from e
            else:
                log_backend_activity('gemini', 'MCP validation skipped', {'reason': 'validator_unavailable'}, agent_id=agent_id)
            normalized_servers = MCPSetupManager.normalize_mcp_servers(self.mcp_servers)
            log_backend_activity('gemini', 'Setting up MCP sessions', {'server_count': len(normalized_servers)}, agent_id=agent_id)
            if status_callback:
                await status_callback('info', {'message': f'Setting up MCP sessions for {len(normalized_servers)} servers'})
            if self._circuit_breakers_enabled and self._mcp_tools_circuit_breaker:
                filtered_servers = MCPCircuitBreakerManager.apply_circuit_breaker_filtering(normalized_servers, self._mcp_tools_circuit_breaker, backend_name='gemini', agent_id=agent_id)
            else:
                filtered_servers = normalized_servers
            if not filtered_servers:
                log_backend_activity('gemini', 'All MCP servers blocked by circuit breaker', {}, agent_id=agent_id)
                if status_callback:
                    await status_callback('warning', {'message': 'All MCP servers blocked by circuit breaker'})
                return
            if len(filtered_servers) < len(normalized_servers):
                log_backend_activity('gemini', 'Circuit breaker filtered servers', {'filtered_count': len(normalized_servers) - len(filtered_servers)}, agent_id=agent_id)
                if status_callback:
                    await status_callback('warning', {'message': f'Circuit breaker filtered {len(normalized_servers) - len(filtered_servers)} servers'})
            allowed_tools = validated_config.get('allowed_tools')
            exclude_tools = validated_config.get('exclude_tools')
            if allowed_tools:
                log_backend_activity('gemini', 'MCP tool filtering configured', {'allowed_tools': allowed_tools}, agent_id=agent_id)
            if exclude_tools:
                log_backend_activity('gemini', 'MCP tool filtering configured', {'exclude_tools': exclude_tools}, agent_id=agent_id)
            self._mcp_client = MCPClient(filtered_servers, timeout_seconds=30, allowed_tools=allowed_tools, exclude_tools=exclude_tools, status_callback=status_callback, hooks=self.filesystem_manager.get_pre_tool_hooks() if self.filesystem_manager else {})
            await self._mcp_client.connect()
            try:
                connected_server_names = self._mcp_client.get_server_names()
            except Exception:
                connected_server_names = []
            if not connected_server_names:
                if self._circuit_breakers_enabled and self._mcp_tools_circuit_breaker:
                    await MCPCircuitBreakerManager.record_event(filtered_servers, self._mcp_tools_circuit_breaker, 'failure', error_message='No servers connected', backend_name='gemini', agent_id=agent_id)
                log_backend_activity('gemini', 'MCP connection failed: no servers connected', {}, agent_id=agent_id)
                if status_callback:
                    await status_callback('error', {'message': 'MCP connection failed: no servers connected'})
                self._mcp_client = None
                return
            connected_server_configs = [server for server in filtered_servers if server.get('name') in connected_server_names]
            if connected_server_configs:
                if self._circuit_breakers_enabled and self._mcp_tools_circuit_breaker:
                    await MCPCircuitBreakerManager.record_event(connected_server_configs, self._mcp_tools_circuit_breaker, 'success', backend_name='gemini', agent_id=agent_id)
            self._mcp_initialized = True
            log_backend_activity('gemini', 'MCP sessions initialized successfully', {}, agent_id=agent_id)
            if status_callback:
                await status_callback('success', {'message': f'MCP sessions initialized successfully with {len(connected_server_names)} servers'})
        except Exception as e:
            if self._circuit_breakers_enabled and self._mcp_tools_circuit_breaker:
                servers = MCPSetupManager.normalize_mcp_servers(self.mcp_servers)
                await MCPCircuitBreakerManager.record_event(servers, self._mcp_tools_circuit_breaker, 'failure', error_message=str(e), backend_name='gemini', agent_id=agent_id)
            if isinstance(e, RuntimeError) and 'MCP configuration' in str(e):
                raise
            elif isinstance(e, MCPConnectionError):
                log_backend_activity('gemini', 'MCP connection failed during setup', {'error': str(e)}, agent_id=agent_id)
                if status_callback:
                    await status_callback('error', {'message': f'Failed to establish MCP connections: {e}'})
                self._mcp_client = None
                raise RuntimeError(f'Failed to establish MCP connections: {e}') from e
            elif isinstance(e, MCPTimeoutError):
                log_backend_activity('gemini', 'MCP connection timeout during setup', {'error': str(e)}, agent_id=agent_id)
                if status_callback:
                    await status_callback('error', {'message': f'MCP connection timeout: {e}'})
                self._mcp_client = None
                raise RuntimeError(f'MCP connection timeout: {e}') from e
            elif isinstance(e, MCPServerError):
                log_backend_activity('gemini', 'MCP server error during setup', {'error': str(e)}, agent_id=agent_id)
                if status_callback:
                    await status_callback('error', {'message': f'MCP server error: {e}'})
                self._mcp_client = None
                raise RuntimeError(f'MCP server error: {e}') from e
            elif isinstance(e, MCPError):
                log_backend_activity('gemini', 'MCP error during setup', {'error': str(e)}, agent_id=agent_id)
                if status_callback:
                    await status_callback('error', {'message': f'MCP error during setup: {e}'})
                self._mcp_client = None
                return
            else:
                log_backend_activity('gemini', 'MCP session setup failed', {'error': str(e)}, agent_id=agent_id)
                if status_callback:
                    await status_callback('error', {'message': f'MCP session setup failed: {e}'})
                self._mcp_client = None

    def detect_coordination_tools(self, tools: List[Dict[str, Any]]) -> bool:
        """Detect if tools contain vote/new_answer coordination tools."""
        if not tools:
            return False
        tool_names = set()
        for tool in tools:
            if tool.get('type') == 'function':
                if 'function' in tool:
                    tool_names.add(tool['function'].get('name', ''))
                elif 'name' in tool:
                    tool_names.add(tool.get('name', ''))
        return 'vote' in tool_names and 'new_answer' in tool_names

    def build_structured_output_prompt(self, base_content: str, valid_agent_ids: Optional[List[str]]=None) -> str:
        """Build prompt that encourages structured output for coordination."""
        agent_list = ''
        if valid_agent_ids:
            agent_list = f'Valid agents: {', '.join(valid_agent_ids)}'
        return f"""{base_content}\n\nIMPORTANT: You must respond with a structured JSON decision at the end of your response.\n\nIf you want to VOTE for an existing agent's answer:\n{{\n  "action_type": "vote",\n  "vote_data": {{\n    "action": "vote",\n    "agent_id": "agent1",  // Choose from: {agent_list or 'agent1, agent2, agent3, etc.'}\n    "reason": "Brief reason for your vote"\n  }}\n}}\n\nIf you want to provide a NEW ANSWER:\n{{\n  "action_type": "new_answer",\n  "answer_data": {{\n    "action": "new_answer",\n    "content": "Your complete improved answer here"\n  }}\n}}\n\nMake your decision and include the JSON at the very end of your response."""

    def extract_structured_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Extract structured JSON response from model output."""
        try:
            markdown_json_pattern = '```json\\s*(\\{.*?\\})\\s*```'
            markdown_matches = re.findall(markdown_json_pattern, response_text, re.DOTALL)
            for match in reversed(markdown_matches):
                try:
                    parsed = json.loads(match.strip())
                    if isinstance(parsed, dict) and 'action_type' in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue
            json_pattern = '\\{[^{}]*(?:\\{[^{}]*\\}[^{}]*)*\\}'
            json_matches = re.findall(json_pattern, response_text, re.DOTALL)
            for match in reversed(json_matches):
                try:
                    cleaned_match = match.strip()
                    parsed = json.loads(cleaned_match)
                    if isinstance(parsed, dict) and 'action_type' in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue
            brace_count = 0
            json_start = -1
            for i, char in enumerate(response_text):
                if char == '{':
                    if brace_count == 0:
                        json_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and json_start >= 0:
                        json_block = response_text[json_start:i + 1]
                        try:
                            parsed = json.loads(json_block)
                            if isinstance(parsed, dict) and 'action_type' in parsed:
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        json_start = -1
            lines = response_text.strip().split('\n')
            json_candidates = []
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('{') and stripped.endswith('}'):
                    json_candidates.append(stripped)
                elif stripped.startswith('{'):
                    json_text = stripped
                    for j in range(i + 1, len(lines)):
                        json_text += '\n' + lines[j].strip()
                        if lines[j].strip().endswith('}'):
                            json_candidates.append(json_text)
                            break
            for candidate in reversed(json_candidates):
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and 'action_type' in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue
            return None
        except Exception:
            return None

    def convert_structured_to_tool_calls(self, structured_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert structured response to tool call format."""
        action_type = structured_response.get('action_type')
        if action_type == 'vote':
            vote_data = structured_response.get('vote_data', {})
            return [{'id': f'vote_{abs(hash(str(vote_data))) % 10000 + 1}', 'type': 'function', 'function': {'name': 'vote', 'arguments': {'agent_id': vote_data.get('agent_id', ''), 'reason': vote_data.get('reason', '')}}}]
        elif action_type == 'new_answer':
            answer_data = structured_response.get('answer_data', {})
            return [{'id': f'new_answer_{abs(hash(str(answer_data))) % 10000 + 1}', 'type': 'function', 'function': {'name': 'new_answer', 'arguments': {'content': answer_data.get('content', '')}}}]
        return []

    async def _handle_mcp_retry_error(self, error: Exception, retry_count: int, max_retries: int) -> tuple[bool, AsyncGenerator[StreamChunk, None]]:
        """Handle MCP retry errors with specific messaging and fallback logic.

        Returns:
            tuple: (should_continue_retrying, error_chunks_generator)
        """
        log_type, user_message, _ = MCPErrorHandler.get_error_details(error, None, log=False)
        log_backend_activity('gemini', f'MCP {log_type} on retry', {'attempt': retry_count, 'error': str(error)}, agent_id=self.agent_id)
        if retry_count >= max_retries:

            async def error_chunks():
                yield StreamChunk(type='content', content=f'\n⚠️  {user_message} after {max_retries} attempts; falling back to workflow tools\n')
            return (False, error_chunks())

        async def empty_chunks():
            if False:
                yield
        return (True, empty_chunks())

    async def _handle_mcp_error_and_fallback(self, error: Exception) -> AsyncGenerator[StreamChunk, None]:
        """Handle MCP errors with specific messaging"""
        self._mcp_tool_failures += 1
        log_type, user_message, _ = MCPErrorHandler.get_error_details(error, None, log=False)
        log_backend_activity('gemini', 'MCP tool call failed', {'call_number': self._mcp_tool_calls_count, 'error_type': log_type, 'error': str(error)}, agent_id=self.agent_id)
        yield StreamChunk(type='content', content=f'\n⚠️  {user_message} ({error}); continuing without MCP tools\n')

    async def _execute_mcp_function_with_retry(self, function_name: str, args: Dict[str, Any], agent_id: Optional[str]=None) -> Any:
        """Execute MCP function with exponential backoff retry logic."""
        if MCPExecutionManager is None:
            raise RuntimeError('MCPExecutionManager is not available - MCP backend utilities are missing')

        async def stats_callback(action: str) -> int:
            if action == 'increment_calls':
                self._mcp_tool_calls_count += 1
                return self._mcp_tool_calls_count
            elif action == 'increment_failures':
                self._mcp_tool_failures += 1
                return self._mcp_tool_failures
            return 0

        async def circuit_breaker_callback(event: str, error_msg: str) -> None:
            if event == 'failure':
                if self._circuit_breakers_enabled and self._mcp_tools_circuit_breaker:
                    servers = MCPSetupManager.normalize_mcp_servers(self.mcp_servers)
                    await MCPCircuitBreakerManager.record_event(servers, self._mcp_tools_circuit_breaker, 'failure', error_message=error_msg, backend_name='gemini', agent_id=agent_id)
            else:
                connected_names: List[str] = []
                try:
                    if self._mcp_client:
                        connected_names = self._mcp_client.get_server_names()
                except Exception:
                    connected_names = []
                if connected_names:
                    servers_to_record = [{'name': name} for name in connected_names]
                    if self._circuit_breakers_enabled and self._mcp_tools_circuit_breaker:
                        await MCPCircuitBreakerManager.record_event(servers_to_record, self._mcp_tools_circuit_breaker, 'success', backend_name='gemini', agent_id=agent_id)
        return await MCPExecutionManager.execute_function_with_retry(function_name=function_name, args=args, functions=self.functions, max_retries=3, stats_callback=stats_callback, circuit_breaker_callback=circuit_breaker_callback, logger_instance=logger)

    async def stream_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Stream response using Gemini API with structured output for coordination and MCP tool support."""
        agent_id = self.agent_id or kwargs.get('agent_id', None)
        client = None
        stream = None
        log_backend_activity('gemini', 'Starting stream_with_tools', {'num_messages': len(messages), 'num_tools': len(tools) if tools else 0}, agent_id=agent_id)
        if self.mcp_servers and MCPMessageManager is not None and hasattr(self, '_max_mcp_message_history') and (self._max_mcp_message_history > 0):
            original_count = len(messages)
            messages = MCPMessageManager.trim_message_history(messages, self._max_mcp_message_history)
            if len(messages) < original_count:
                log_backend_activity('gemini', 'Trimmed MCP message history', {'original': original_count, 'trimmed': len(messages), 'limit': self._max_mcp_message_history}, agent_id=agent_id)
        try:
            from google import genai
            if not self._mcp_initialized and self.mcp_servers:
                async for chunk in self._setup_mcp_with_status_stream(agent_id):
                    yield chunk
            elif not self._mcp_initialized:
                await self._setup_mcp_tools(agent_id)
            all_params = {**self.config, **kwargs}
            enable_web_search = all_params.get('enable_web_search', False)
            enable_code_execution = all_params.get('enable_code_execution', False)
            using_sdk_mcp = bool(self.mcp_servers)
            is_coordination = self.detect_coordination_tools(tools)
            valid_agent_ids = None
            if is_coordination:
                for tool in tools:
                    if tool.get('type') == 'function':
                        func_def = tool.get('function', {})
                        if func_def.get('name') == 'vote':
                            agent_id_param = func_def.get('parameters', {}).get('properties', {}).get('agent_id', {})
                            if 'enum' in agent_id_param:
                                valid_agent_ids = agent_id_param['enum']
                            break
            conversation_content = ''
            system_message = ''
            for msg in messages:
                role = msg.get('role')
                if role == 'system':
                    system_message = msg.get('content', '')
                elif role == 'user':
                    conversation_content += f'User: {msg.get('content', '')}\n'
                elif role == 'assistant':
                    conversation_content += f'Assistant: {msg.get('content', '')}\n'
                elif role == 'tool':
                    tool_output = msg.get('content', '')
                    conversation_content += f'Tool Result: {tool_output}\n'
            if is_coordination:
                conversation_content = self.build_structured_output_prompt(conversation_content, valid_agent_ids)
            full_content = ''
            if system_message:
                full_content += f'{system_message}\n\n'
            full_content += conversation_content
            client = genai.Client(api_key=self.api_key)
            builtin_tools = []
            if enable_web_search:
                try:
                    from google.genai import types
                    grounding_tool = types.Tool(google_search=types.GoogleSearch())
                    builtin_tools.append(grounding_tool)
                except ImportError:
                    yield StreamChunk(type='content', content='\n⚠️  Web search requires google.genai.types\n')
            if enable_code_execution:
                try:
                    from google.genai import types
                    code_tool = types.Tool(code_execution=types.ToolCodeExecution())
                    builtin_tools.append(code_tool)
                except ImportError:
                    yield StreamChunk(type='content', content='\n⚠️  Code execution requires google.genai.types\n')
            config = {}
            excluded_params = self.get_base_excluded_config_params() | {'enable_web_search', 'enable_code_execution', 'use_multi_mcp', 'mcp_sdk_auto', 'allowed_tools', 'exclude_tools'}
            for key, value in all_params.items():
                if key not in excluded_params and value is not None:
                    if key == 'max_tokens':
                        config['max_output_tokens'] = value
                    elif key == 'model':
                        model_name = value
                    else:
                        config[key] = value
            all_tools = []
            if using_sdk_mcp and self.mcp_servers:
                if not self._mcp_client or not getattr(self._mcp_client, 'is_connected', lambda: False)():
                    max_mcp_retries = 5
                    mcp_connected = False
                    for retry_count in range(1, max_mcp_retries + 1):
                        try:
                            self._mcp_connection_retries = retry_count
                            if retry_count > 1:
                                log_backend_activity('gemini', 'MCP connection retry', {'attempt': retry_count, 'max_retries': max_mcp_retries}, agent_id=agent_id)
                                yield StreamChunk(type='mcp_status', status='mcp_retry', content=f'Retrying MCP connection (attempt {retry_count}/{max_mcp_retries})', source='mcp_tools')
                                await asyncio.sleep(0.5 * retry_count)
                            if self._circuit_breakers_enabled and self._mcp_tools_circuit_breaker:
                                filtered_retry_servers = MCPCircuitBreakerManager.apply_circuit_breaker_filtering(self.mcp_servers, self._mcp_tools_circuit_breaker, backend_name='gemini', agent_id=agent_id)
                            else:
                                filtered_retry_servers = self.mcp_servers
                            if not filtered_retry_servers:
                                log_backend_activity('gemini', 'All MCP servers blocked during retry', {}, agent_id=agent_id)
                                yield StreamChunk(type='mcp_status', status='mcp_blocked', content='All MCP servers blocked by circuit breaker', source='mcp_tools')
                                using_sdk_mcp = False
                                break
                            backend_config = {'mcp_servers': self.mcp_servers}
                            if MCPConfigValidator is not None:
                                try:
                                    validator = MCPConfigValidator()
                                    validated_config_retry = validator.validate_backend_mcp_config(backend_config)
                                    allowed_tools_retry = validated_config_retry.get('allowed_tools')
                                    exclude_tools_retry = validated_config_retry.get('exclude_tools')
                                except Exception:
                                    allowed_tools_retry = None
                                    exclude_tools_retry = None
                            else:
                                allowed_tools_retry = None
                                exclude_tools_retry = None
                            self._mcp_client = await MCPClient.create_and_connect(filtered_retry_servers, timeout_seconds=30, allowed_tools=allowed_tools_retry, exclude_tools=exclude_tools_retry)
                            if self._circuit_breakers_enabled and self._mcp_tools_circuit_breaker:
                                await MCPCircuitBreakerManager.record_event(filtered_retry_servers, self._mcp_tools_circuit_breaker, 'success', backend_name='gemini', agent_id=agent_id)
                            mcp_connected = True
                            log_backend_activity('gemini', 'MCP connection successful on retry', {'attempt': retry_count}, agent_id=agent_id)
                            yield StreamChunk(type='mcp_status', status='mcp_connected', content=f'MCP connection successful on attempt {retry_count}', source='mcp_tools')
                            break
                        except (MCPConnectionError, MCPTimeoutError, MCPServerError, MCPError, Exception) as e:
                            if self._circuit_breakers_enabled and self._mcp_tools_circuit_breaker:
                                servers = MCPSetupManager.normalize_mcp_servers(self.mcp_servers)
                                await MCPCircuitBreakerManager.record_event(servers, self._mcp_tools_circuit_breaker, 'failure', error_message=str(e), backend_name='gemini', agent_id=agent_id)
                            should_continue, error_chunks = await self._handle_mcp_retry_error(e, retry_count, max_mcp_retries)
                            if not should_continue:
                                async for chunk in error_chunks:
                                    yield chunk
                                using_sdk_mcp = False
                    if not mcp_connected:
                        using_sdk_mcp = False
                        self._mcp_client = None
            if not using_sdk_mcp:
                all_tools.extend(builtin_tools)
                if all_tools:
                    config['tools'] = all_tools
            if is_coordination:
                if not using_sdk_mcp and (not all_tools):
                    config['response_mime_type'] = 'application/json'
                    config['response_schema'] = CoordinationResponse.model_json_schema()
                else:
                    pass
            log_backend_agent_message(agent_id or 'default', 'SEND', {'content': full_content, 'builtin_tools': len(builtin_tools) if builtin_tools else 0}, backend_name='gemini')
            full_content_text = ''
            final_response = None
            if using_sdk_mcp and self.mcp_servers:
                try:
                    if not self._mcp_client:
                        raise RuntimeError('MCP client not initialized')
                    mcp_sessions = self._mcp_client.get_active_sessions()
                    if not mcp_sessions:
                        raise RuntimeError('No active MCP sessions available')
                    if self.filesystem_manager:
                        logger.info(f'[Gemini] Converting {len(mcp_sessions)} MCP sessions to permission sessions')
                        try:
                            from ..mcp_tools.hooks import convert_sessions_to_permission_sessions
                            mcp_sessions = convert_sessions_to_permission_sessions(mcp_sessions, self.filesystem_manager.path_permission_manager)
                        except Exception as e:
                            logger.error(f'[Gemini] Failed to convert sessions to permission sessions: {e}')
                    else:
                        logger.debug('[Gemini] No filesystem manager found, using standard sessions')
                    session_config = dict(config)
                    available_tools = []
                    if self._mcp_client:
                        available_tools = list(self._mcp_client.tools.keys())
                    if self.is_planning_mode_enabled():
                        logger.info('[Gemini] Planning mode enabled - blocking MCP tools during coordination')
                        log_backend_activity('gemini', 'MCP tools blocked in planning mode', {'blocked_tools': len(available_tools), 'session_count': len(mcp_sessions)}, agent_id=agent_id)
                    else:
                        logger.debug(f'[Gemini] Passing {len(mcp_sessions)} sessions to SDK: {[type(s).__name__ for s in mcp_sessions]}')
                        session_config['tools'] = mcp_sessions
                    self._mcp_tool_calls_count += 1
                    log_backend_activity('gemini', 'MCP tool call initiated', {'call_number': self._mcp_tool_calls_count, 'session_count': len(mcp_sessions), 'available_tools': available_tools[:], 'total_tools': len(available_tools)}, agent_id=agent_id)
                    log_tool_call(agent_id, 'mcp_session_tools', {'session_count': len(mcp_sessions), 'call_number': self._mcp_tool_calls_count, 'available_tools': available_tools}, backend_name='gemini')
                    tools_info = f' ({len(available_tools)} tools available)' if available_tools else ''
                    yield StreamChunk(type='mcp_status', status='mcp_tools_initiated', content=f'MCP tool call initiated (call #{self._mcp_tool_calls_count}){tools_info}: {', '.join(available_tools[:5])}{('...' if len(available_tools) > 5 else '')}', source='mcp_tools')
                    stream = await client.aio.models.generate_content_stream(model=model_name, contents=full_content, config=session_config)
                    mcp_tracker = MCPCallTracker()
                    mcp_response_tracker = MCPResponseTracker()
                    mcp_tools_used = []
                    async for chunk in stream:
                        if hasattr(chunk, 'automatic_function_calling_history') and chunk.automatic_function_calling_history:
                            for history_item in chunk.automatic_function_calling_history:
                                if hasattr(history_item, 'parts') and history_item.parts is not None:
                                    for part in history_item.parts:
                                        if hasattr(part, 'function_call') and part.function_call:
                                            call_data = self.mcp_extractor.extract_function_call(part.function_call)
                                            if call_data:
                                                tool_name = call_data['name']
                                                tool_args = call_data['arguments']
                                                if mcp_tracker.is_new_call(tool_name, tool_args):
                                                    call_record = mcp_tracker.add_call(tool_name, tool_args)
                                                    mcp_tools_used.append({'name': tool_name, 'arguments': tool_args, 'timestamp': call_record['timestamp']})
                                                    timestamp_str = time.strftime('%H:%M:%S', time.localtime(call_record['timestamp']))
                                                    yield StreamChunk(type='mcp_status', status='mcp_tool_called', content=f'🔧 MCP Tool Called: {tool_name} at {timestamp_str} with args: {json.dumps(tool_args, indent=2)}', source='mcp_tools')
                                                    log_tool_call(agent_id, tool_name, tool_args, backend_name='gemini')
                                        elif hasattr(part, 'function_response') and part.function_response:
                                            response_data = self.mcp_extractor.extract_function_response(part.function_response)
                                            if response_data:
                                                tool_name = response_data['name']
                                                tool_response = response_data['response']
                                                if mcp_response_tracker.is_new_response(tool_name, tool_response):
                                                    response_record = mcp_response_tracker.add_response(tool_name, tool_response)
                                                    response_text = None
                                                    if isinstance(tool_response, dict) and 'result' in tool_response:
                                                        result = tool_response['result']
                                                        if hasattr(result, 'content') and result.content:
                                                            first_content = result.content[0]
                                                            if hasattr(first_content, 'text'):
                                                                response_text = first_content.text
                                                    if response_text is None:
                                                        response_text = str(tool_response)
                                                    timestamp_str = time.strftime('%H:%M:%S', time.localtime(response_record['timestamp']))
                                                    yield StreamChunk(type='mcp_status', status='mcp_tool_response', content=f'✅ MCP Tool Response from {tool_name} at {timestamp_str}: {response_text}', source='mcp_tools')
                                                    log_backend_activity('gemini', 'MCP tool response received', {'tool_name': tool_name, 'response_preview': str(tool_response)[:]}, agent_id=agent_id)
                            if not hasattr(self, '_mcp_stream_started'):
                                self._mcp_tool_successes += 1
                                self._mcp_stream_started = True
                                log_backend_activity('gemini', 'MCP tool call succeeded', {'call_number': self._mcp_tool_calls_count}, agent_id=agent_id)
                                log_tool_call(agent_id, 'mcp_session_tools', {'session_count': len(mcp_sessions), 'call_number': self._mcp_tool_calls_count}, result='success', backend_name='gemini')
                                yield StreamChunk(type='mcp_status', status='mcp_tools_success', content=f'MCP tool call succeeded (call #{self._mcp_tool_calls_count})', source='mcp_tools')
                        if hasattr(chunk, 'text') and chunk.text:
                            chunk_text = chunk.text
                            full_content_text += chunk_text
                            log_backend_agent_message(agent_id, 'RECV', {'content': chunk_text}, backend_name='gemini')
                            log_stream_chunk('backend.gemini', 'content', chunk_text, agent_id)
                            yield StreamChunk(type='content', content=chunk_text)
                    if hasattr(self, '_mcp_stream_started'):
                        delattr(self, '_mcp_stream_started')
                    tools_summary = mcp_tracker.get_summary()
                    if not tools_summary or tools_summary == 'No MCP tools called':
                        tools_summary = 'MCP session completed (no tools explicitly called)'
                    else:
                        tools_summary = f'MCP session complete - {tools_summary}'
                    log_stream_chunk('backend.gemini', 'mcp_indicator', tools_summary, agent_id)
                    yield StreamChunk(type='mcp_status', status='mcp_session_complete', content=f'MCP session complete - {tools_summary}', source='mcp_tools')
                except (MCPConnectionError, MCPTimeoutError, MCPServerError, MCPError, Exception) as e:
                    log_stream_chunk('backend.gemini', 'mcp_error', str(e), agent_id)
                    async for chunk in self._handle_mcp_error_and_fallback(e):
                        yield chunk
                    manual_config = dict(config)
                    if all_tools:
                        manual_config['tools'] = all_tools
                    stream = await client.aio.models.generate_content_stream(model=model_name, contents=full_content, config=manual_config)
                    async for chunk in stream:
                        if hasattr(chunk, 'text') and chunk.text:
                            chunk_text = chunk.text
                            full_content_text += chunk_text
                            log_stream_chunk('backend.gemini', 'fallback_content', chunk_text, agent_id)
                            yield StreamChunk(type='content', content=chunk_text)
            else:
                stream = await client.aio.models.generate_content_stream(model=model_name, contents=full_content, config=config)
                async for chunk in stream:
                    if hasattr(chunk, 'text') and chunk.text:
                        chunk_text = chunk.text
                        full_content_text += chunk_text
                        log_stream_chunk('backend.gemini', 'content', chunk_text, agent_id)
                        log_backend_agent_message(agent_id, 'RECV', {'content': chunk_text}, backend_name='gemini')
                        yield StreamChunk(type='content', content=chunk_text)
            content = full_content_text
            tool_calls_detected: List[Dict[str, Any]] = []
            if is_coordination and content.strip() and (not tool_calls_detected):
                structured_response = None
                try:
                    structured_response = json.loads(content.strip())
                except json.JSONDecodeError:
                    structured_response = self.extract_structured_response(content)
                if structured_response and isinstance(structured_response, dict) and ('action_type' in structured_response):
                    tool_calls = self.convert_structured_to_tool_calls(structured_response)
                    if tool_calls:
                        tool_calls_detected = tool_calls
                        log_stream_chunk('backend.gemini', 'tool_calls', tool_calls, agent_id)
                        try:
                            for tool_call in tool_calls:
                                log_tool_call(agent_id, tool_call.get('function', {}).get('name', 'unknown_coordination_tool'), tool_call.get('function', {}).get('arguments', {}), result='coordination_tool_called', backend_name='gemini')
                        except Exception:
                            pass
            if builtin_tools and final_response and hasattr(final_response, 'candidates') and final_response.candidates:
                candidate = final_response.candidates[0]
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    search_actually_used = False
                    search_queries = []
                    if hasattr(candidate.grounding_metadata, 'web_search_queries') and candidate.grounding_metadata.web_search_queries:
                        try:
                            for query in candidate.grounding_metadata.web_search_queries:
                                if query and query.strip():
                                    search_queries.append(query.strip())
                                    search_actually_used = True
                        except (TypeError, AttributeError):
                            pass
                    if hasattr(candidate.grounding_metadata, 'grounding_chunks') and candidate.grounding_metadata.grounding_chunks:
                        try:
                            if len(candidate.grounding_metadata.grounding_chunks) > 0:
                                search_actually_used = True
                        except (TypeError, AttributeError):
                            pass
                    if search_actually_used:
                        log_stream_chunk('backend.gemini', 'web_search_result', {'queries': search_queries, 'results_integrated': True}, agent_id)
                        log_tool_call(agent_id, 'google_search_retrieval', {'queries': search_queries, 'chunks_found': len(candidate.grounding_metadata.grounding_chunks) if hasattr(candidate.grounding_metadata, 'grounding_chunks') else 0}, result='search_completed', backend_name='gemini')
                        yield StreamChunk(type='content', content='🔍 [Builtin Tool: Web Search] Results integrated\n')
                        for query in search_queries:
                            log_stream_chunk('backend.gemini', 'web_search_result', {'queries': search_queries, 'results_integrated': True}, agent_id)
                            yield StreamChunk(type='content', content=f"🔍 [Search Query] '{query}'\n")
                        self.search_count += 1
                if enable_code_execution and hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    code_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'executable_code') and part.executable_code:
                            code_content = getattr(part.executable_code, 'code', str(part.executable_code))
                            code_parts.append(f'Code: {code_content}')
                        elif hasattr(part, 'code_execution_result') and part.code_execution_result:
                            result_content = getattr(part.code_execution_result, 'output', str(part.code_execution_result))
                            code_parts.append(f'Result: {result_content}')
                    if code_parts:
                        log_stream_chunk('backend.gemini', 'code_execution', 'Code executed', agent_id)
                        try:
                            log_tool_call(agent_id, 'code_execution', {'code_parts_count': len(code_parts)}, result='code_executed', backend_name='gemini')
                        except Exception:
                            pass
                        yield StreamChunk(type='content', content='💻 [Builtin Tool: Code Execution] Code executed\n')
                        for part in code_parts:
                            if part.startswith('Code: '):
                                code_content = part[6:]
                                log_stream_chunk('backend.gemini', 'code_execution_result', {'code_parts': len(code_parts), 'execution_successful': True, 'snippet': code_content}, agent_id)
                                yield StreamChunk(type='content', content=f'💻 [Code Executed]\n```python\n{code_content}\n```\n')
                            elif part.startswith('Result: '):
                                result_content = part[8:]
                                log_stream_chunk('backend.gemini', 'code_execution_result', {'code_parts': len(code_parts), 'execution_successful': True, 'result': result_content}, agent_id)
                                yield StreamChunk(type='content', content=f'📊 [Result] {result_content}\n')
                        self.code_execution_count += 1
            if tool_calls_detected:
                log_stream_chunk('backend.gemini', 'tool_calls_yielded', {'tool_count': len(tool_calls_detected), 'tool_names': [tc.get('function', {}).get('name') for tc in tool_calls_detected]}, agent_id)
                yield StreamChunk(type='tool_calls', tool_calls=tool_calls_detected)
            complete_message = {'role': 'assistant', 'content': content.strip()}
            if tool_calls_detected:
                complete_message['tool_calls'] = tool_calls_detected
            log_stream_chunk('backend.gemini', 'complete_message', {'content_length': len(content.strip()), 'has_tool_calls': bool(tool_calls_detected)}, agent_id)
            yield StreamChunk(type='complete_message', complete_message=complete_message)
            log_stream_chunk('backend.gemini', 'done', None, agent_id)
            yield StreamChunk(type='done')
        except Exception as e:
            error_msg = f'Gemini API error: {e}'
            log_stream_chunk('backend.gemini', 'stream_error', {'error_type': type(e).__name__, 'error_message': str(e)}, agent_id)
            yield StreamChunk(type='error', error=error_msg)
        finally:
            await self._cleanup_resources(stream, client)
            try:
                await self.__aexit__(None, None, None)
            except Exception as e:
                log_backend_activity('gemini', 'MCP cleanup failed', {'error': str(e)}, agent_id=self.agent_id)

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return 'Gemini'

    def get_filesystem_support(self) -> FilesystemSupport:
        """Gemini supports filesystem through MCP servers."""
        return FilesystemSupport.MCP

    def get_supported_builtin_tools(self) -> List[str]:
        """Get list of builtin tools supported by Gemini."""
        return ['google_search_retrieval', 'code_execution']

    def get_mcp_results(self) -> Dict[str, Any]:
        """
        Get all captured MCP tool calls and responses.

        Returns:
            Dict containing:
            - calls: List of all MCP tool calls
            - responses: List of all MCP tool responses
            - pairs: List of matched call-response pairs
            - summary: Statistical summary of interactions
        """
        return {'calls': self.mcp_extractor.mcp_calls, 'responses': self.mcp_extractor.mcp_responses, 'pairs': self.mcp_extractor.call_response_pairs, 'summary': self.mcp_extractor.get_summary()}

    def get_mcp_paired_results(self) -> List[Dict[str, Any]]:
        """
        Get only the paired MCP tool calls and responses.

        Returns:
            List of dictionaries containing matched call-response pairs
        """
        return self.mcp_extractor.call_response_pairs

    def get_mcp_summary(self) -> Dict[str, Any]:
        """
        Get a summary of MCP tool interactions.

        Returns:
            Dictionary with statistics about MCP tool usage
        """
        return self.mcp_extractor.get_summary()

    def clear_mcp_results(self):
        """Clear all stored MCP interaction data."""
        self.mcp_extractor.clear()

    def reset_tool_usage(self):
        """Reset tool usage tracking."""
        self.search_count = 0
        self.code_execution_count = 0
        self._mcp_tool_calls_count = 0
        self._mcp_tool_failures = 0
        self._mcp_tool_successes = 0
        self._mcp_connection_retries = 0
        self.mcp_extractor.clear()
        super().reset_token_usage()

    async def cleanup_mcp(self):
        """Cleanup MCP connections."""
        if self._mcp_client:
            try:
                await self._mcp_client.disconnect()
                log_backend_activity('gemini', 'MCP client disconnected', {}, agent_id=self.agent_id)
            except (MCPConnectionError, MCPTimeoutError, MCPServerError, MCPError, Exception) as e:
                MCPErrorHandler.get_error_details(e, 'disconnect', log=True)
            finally:
                self._mcp_client = None
                self._mcp_initialized = False

    async def _cleanup_resources(self, stream, client):
        """Cleanup google-genai resources to avoid unclosed aiohttp sessions."""
        try:
            if stream is not None:
                close_fn = getattr(stream, 'aclose', None) or getattr(stream, 'close', None)
                if close_fn is not None:
                    maybe = close_fn()
                    if hasattr(maybe, '__await__'):
                        await maybe
        except Exception as e:
            log_backend_activity('gemini', 'Stream cleanup failed', {'error': str(e)}, agent_id=self.agent_id)
        try:
            if client is not None:
                base_client = getattr(client, '_api_client', None)
                if base_client is not None:
                    session = getattr(base_client, '_aiohttp_session', None)
                    if session is not None and hasattr(session, 'close'):
                        if not session.closed:
                            await session.close()
                            log_backend_activity('gemini', 'Closed google-genai aiohttp session', {}, agent_id=self.agent_id)
                        base_client._aiohttp_session = None
                        await asyncio.sleep(0)
        except Exception as e:
            log_backend_activity('gemini', 'Failed to close google-genai aiohttp session', {'error': str(e)}, agent_id=self.agent_id)
        try:
            if client is not None and hasattr(client, 'aio') and (client.aio is not None):
                aio_obj = client.aio
                for method_name in ('close', 'stop'):
                    method = getattr(aio_obj, method_name, None)
                    if method:
                        maybe = method()
                        if hasattr(maybe, '__await__'):
                            await maybe
                        break
        except Exception as e:
            log_backend_activity('gemini', 'Client AIO cleanup failed', {'error': str(e)}, agent_id=self.agent_id)
        try:
            if client is not None:
                for method_name in ('aclose', 'close'):
                    method = getattr(client, method_name, None)
                    if method:
                        maybe = method()
                        if hasattr(maybe, '__await__'):
                            await maybe
                        break
        except Exception as e:
            log_backend_activity('gemini', 'Client cleanup failed', {'error': str(e)}, agent_id=self.agent_id)

    async def __aenter__(self) -> 'GeminiBackend':
        """Async context manager entry."""
        try:
            await self._setup_mcp_tools(agent_id=self.agent_id)
        except Exception as e:
            log_backend_activity('gemini', 'MCP setup failed during context entry', {'error': str(e)}, agent_id=self.agent_id)
        return self

    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[object]) -> None:
        """Async context manager exit with automatic resource cleanup."""
        _ = (exc_type, exc_val, exc_tb)
        try:
            await self.cleanup_mcp()
        except Exception as e:
            log_backend_activity('gemini', 'Backend cleanup error', {'error': str(e)}, agent_id=self.agent_id)

def convert_sessions_to_permission_sessions(sessions: List[ClientSession], permission_manager) -> List[PermissionClientSession]:
    """
    Convert a list of ClientSession objects to PermissionClientSession subclasses.

    Args:
        sessions: List of ClientSession objects to convert
        permission_manager: Object with pre_tool_use_hook method

    Returns:
        List of PermissionClientSession objects that apply permission hooks
    """
    logger.debug(f'[PermissionClientSession] Converting {len(sessions)} sessions to permission sessions')
    converted = []
    for session in sessions:
        perm_session = PermissionClientSession(session, permission_manager)
        converted.append(perm_session)
    logger.debug(f'[PermissionClientSession] Successfully converted {len(converted)} sessions')
    return converted

class GrokBackend(ChatCompletionsBackend):
    """Grok backend using xAI's OpenAI-compatible API."""

    def __init__(self, api_key: Optional[str]=None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.getenv('XAI_API_KEY')
        self.base_url = 'https://api.x.ai/v1'

    def _create_client(self, **kwargs) -> AsyncOpenAI:
        """Create OpenAI client configured for xAI's Grok API."""
        import openai
        return openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def _build_base_api_params(self, messages: List[Dict[str, Any]], all_params: Dict[str, Any]) -> Dict[str, Any]:
        """Build base API params for xAI's Grok API."""
        api_params = super()._build_base_api_params(messages, all_params)
        enable_web_search = all_params.get('enable_web_search', False)
        if enable_web_search:
            existing_extra = api_params.get('extra_body', {})
            if isinstance(existing_extra, dict) and 'search_parameters' in existing_extra:
                error_message = "Conflict: Cannot use both 'enable_web_search: true' and manual 'extra_body.search_parameters'. Use one or the other."
                log_stream_chunk('backend.grok', 'error', error_message, self.agent_id)
                raise ValueError(error_message)
            search_params = {'mode': 'auto', 'return_citations': True}
            merged_extra = existing_extra.copy()
            merged_extra['search_parameters'] = search_params
            api_params['extra_body'] = merged_extra
        return api_params

    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return 'Grok'

    def get_supported_builtin_tools(self) -> List[str]:
        """Get list of builtin tools supported by Grok."""
        return ['web_search']

