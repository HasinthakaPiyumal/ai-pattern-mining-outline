# Cluster 69

def google_parts_to_mcp_content(google_parts: list[types.Part]) -> list[TextContent | ImageContent | EmbeddedResource]:
    mcp_content: list[TextContent | ImageContent | EmbeddedResource] = []
    for part in google_parts:
        if part.text:
            mcp_content.append(TextContent(type='text', text=part.text))
        elif part.file_data:
            if part.file_data.file_uri.startswith('data:') and part.file_data.mime_type.startswith('image/'):
                _, base64_data = image_url_to_mime_and_base64(part.file_data.file_uri)
                mcp_content.append(ImageContent(type='image', mimeType=part.file_data.mime_type, data=base64_data))
            else:
                mcp_content.append(EmbeddedResource(type='resource', resource=BlobResourceContents(mimeType=part.file_data.mime_type, uri=part.file_data.file_uri)))
        elif part.function_call:
            mcp_content.append(TextContent(type='text', text=str(part.function_call)))
        else:
            mcp_content.append(TextContent(type='text', text=str(part)))
    return mcp_content

def image_url_to_mime_and_base64(image_url: str) -> tuple[str, str]:
    """
    Extract mime type and base64 data from ImageUrl
    """
    import re
    match = re.match('data:(image/[\\w.+-]+);base64,(.*)', image_url)
    if not match:
        raise ValueError(f'Invalid image data URI: {image_url[:30]}...')
    mime_type, base64_data = match.groups()
    return (mime_type, base64_data)

class MCPOpenAITypeConverter(ProviderToMCPConverter[ChatCompletionMessageParam, ChatCompletionMessage]):
    """
    Convert between OpenAI and MCP types.
    """

    @classmethod
    def from_mcp_message_result(cls, result: MCPMessageResult) -> ChatCompletionMessage:
        if result.role != 'assistant':
            raise ValueError(f"Expected role to be 'assistant' but got '{result.role}' instead.")
        return ChatCompletionMessage(role='assistant', content=result.content.text or str(result.context))

    @classmethod
    def to_mcp_message_result(cls, result: ChatCompletionMessage) -> MCPMessageResult:
        return MCPMessageResult(role=result.role, content=TextContent(type='text', text=result.content), model='', stopReason=None, **result.model_dump(exclude={'role', 'content'}))

    @classmethod
    def from_mcp_message_param(cls, param: MCPMessageParam) -> ChatCompletionMessageParam:
        if param.role == 'assistant':
            extras = param.model_dump(exclude={'role', 'content'})
            return ChatCompletionAssistantMessageParam(role='assistant', content=[mcp_content_to_openai_content_part(param.content)], **extras)
        elif param.role == 'user':
            extras = param.model_dump(exclude={'role', 'content'})
            return ChatCompletionUserMessageParam(role='user', content=[mcp_content_to_openai_content_part(param.content)], **extras)
        else:
            raise ValueError(f"Unexpected role: {param.role}, MCP only supports 'assistant' and 'user'")

    @classmethod
    def to_mcp_message_param(cls, param: ChatCompletionMessageParam) -> MCPMessageParam:
        contents = openai_content_to_mcp_content(param.content)
        if len(contents) > 1:
            raise NotImplementedError('Multiple content elements in a single message are not supported')
        mcp_content: TextContent | ImageContent | EmbeddedResource = contents[0]
        if param.role == 'assistant':
            return MCPMessageParam(role='assistant', content=mcp_content, **typed_dict_extras(param, ['role', 'content']))
        elif param.role == 'user':
            return MCPMessageParam(role='user', content=mcp_content, **typed_dict_extras(param, ['role', 'content']))
        elif param.role == 'tool':
            raise NotImplementedError('Tool messages are not supported in SamplingMessage yet')
        elif param.role == 'system':
            raise NotImplementedError('System messages are not supported in SamplingMessage yet')
        elif param.role == 'developer':
            raise NotImplementedError('Developer messages are not supported in SamplingMessage yet')
        elif param.role == 'function':
            raise NotImplementedError('Function messages are not supported in SamplingMessage yet')
        else:
            raise ValueError(f"Unexpected role: {param.role}, MCP only supports 'assistant', 'user', 'tool', 'system', 'developer', and 'function'")

def typed_dict_extras(d: dict, exclude: List[str]):
    extras = {k: v for k, v in d.items() if k not in exclude}
    return extras

def openai_content_to_mcp_content(content: str | Iterable[ChatCompletionContentPartParam | ChatCompletionContentPartRefusalParam]) -> Iterable[TextContent | ImageContent | EmbeddedResource]:
    mcp_content = []
    if isinstance(content, str):
        mcp_content = [TextContent(type='text', text=content)]
    else:
        for c in content:
            if c['type'] == 'text':
                mcp_content.append(TextContent(type='text', text=c['text'], **typed_dict_extras(c, ['text'])))
            elif c['type'] == 'image_url':
                if c['image_url'].startswith('data:'):
                    mime_type, base64_data = image_url_to_mime_and_base64(c['image_url'])
                    mcp_content.append(ImageContent(type='image', data=base64_data, mimeType=mime_type))
                else:
                    raise NotImplementedError('Image content conversion not implemented')
            elif c['type'] == 'input_audio':
                raise NotImplementedError('Audio content conversion not implemented')
            elif c['type'] == 'refusal':
                mcp_content.append(TextContent(type='text', text=c['refusal'], **typed_dict_extras(c, ['refusal'])))
            else:
                raise ValueError(f'Unexpected content type: {c['type']}')
    return mcp_content

class MCPAzureTypeConverter(ProviderToMCPConverter[MessageParam, ResponseMessage]):
    """
    Convert between Azure and MCP types.
    """

    @classmethod
    def from_mcp_message_result(cls, result: MCPMessageResult) -> ResponseMessage:
        if result.role != 'assistant':
            raise ValueError(f"Expected role to be 'assistant' but got '{result.role}' instead.")
        if isinstance(result.content, TextContent):
            return AssistantMessage(content=result.content.text)
        else:
            return AssistantMessage(content=f'{result.content.mimeType}:{result.content.data}')

    @classmethod
    def to_mcp_message_result(cls, result: ResponseMessage) -> MCPMessageResult:
        return MCPMessageResult(role=result.role, content=TextContent(type='text', text=result.content), model='', stopReason=None)

    @classmethod
    def from_mcp_message_param(cls, param: MCPMessageParam) -> MessageParam:
        if param.role == 'assistant':
            extras = param.model_dump(exclude={'role', 'content'})
            return AssistantMessage(content=mcp_content_to_azure_content([param.content]), **extras)
        elif param.role == 'user':
            extras = param.model_dump(exclude={'role', 'content'})
            return UserMessage(content=mcp_content_to_azure_content([param.content], str_only=False), **extras)
        else:
            raise ValueError(f"Unexpected role: {param.role}, MCP only supports 'assistant' and 'user'")

    @classmethod
    def to_mcp_message_param(cls, param: MessageParam) -> MCPMessageParam:
        contents = azure_content_to_mcp_content(param.content)
        if len(contents) > 1:
            raise NotImplementedError('Multiple content elements in a single message are not supported')
        elif len(contents) == 0:
            raise ValueError('No content elements in a message')
        mcp_content: TextContent | ImageContent | EmbeddedResource = contents[0]
        if param.role == ChatRole.ASSISTANT:
            return MCPMessageParam(role='assistant', content=mcp_content, **typed_dict_extras(param, ['role', 'content']))
        elif param.role == ChatRole.USER:
            return MCPMessageParam(role='user', content=mcp_content, **typed_dict_extras(param, ['role', 'content']))
        elif param.role == ChatRole.TOOL:
            raise NotImplementedError('Tool messages are not supported in SamplingMessage yet')
        elif param.role == ChatRole.SYSTEM:
            raise NotImplementedError('System messages are not supported in SamplingMessage yet')
        elif param.role == ChatRole.DEVELOPER:
            raise NotImplementedError('Developer messages are not supported in SamplingMessage yet')
        else:
            raise ValueError(f"Unexpected role: {param.role}, Azure only supports 'assistant', 'user', 'tool', 'system', 'developer'")

def azure_content_to_mcp_content(content: str | list[ContentItem] | None) -> Iterable[TextContent | ImageContent | EmbeddedResource]:
    mcp_content: Iterable[TextContent | ImageContent | EmbeddedResource] = []
    if content is None:
        return mcp_content
    elif isinstance(content, str):
        return [TextContent(type='text', text=content)]
    for item in content:
        if isinstance(item, TextContentItem):
            mcp_content.append(TextContent(type='text', text=item.text))
        elif isinstance(item, ImageContentItem):
            mime_type, base64_data = image_url_to_mime_and_base64(item.image_url)
            mcp_content.append(ImageContent(type='image', mimeType=mime_type, data=base64_data))
        elif isinstance(item, AudioContentItem):
            raise NotImplementedError('Audio content conversion not implemented')
    return mcp_content

class BedrockMCPTypeConverter(ProviderToMCPConverter[MessageUnionTypeDef, MessageUnionTypeDef]):
    """
    Convert between Bedrock and MCP types.
    """

    @classmethod
    def from_mcp_message_result(cls, result: MCPMessageResult) -> MessageUnionTypeDef:
        if result.role != 'assistant':
            raise ValueError(f"Expected role to be 'assistant' but got '{result.role}' instead.")
        return {'role': 'assistant', 'content': mcp_content_to_bedrock_content(result.content)}

    @classmethod
    def to_mcp_message_result(cls, result: MessageUnionTypeDef) -> MCPMessageResult:
        contents = bedrock_content_to_mcp_content(result['content'])
        if len(contents) > 1:
            raise NotImplementedError('Multiple content elements in a single message are not supported in MCP yet')
        mcp_content = contents[0]
        return MCPMessageResult(role=result.role, content=mcp_content, model=None, stopReason=None)

    @classmethod
    def from_mcp_message_param(cls, param: MCPMessageParam) -> MessageUnionTypeDef:
        return {'role': param.role, 'content': mcp_content_to_bedrock_content([param.content])}

    @classmethod
    def to_mcp_message_param(cls, param: MessageUnionTypeDef) -> MCPMessageParam:
        contents = bedrock_content_to_mcp_content(param['content'])
        if len(contents) > 1:
            raise NotImplementedError('Multiple content elements in a single message are not supported')
        mcp_content = contents[0]
        return MCPMessageParam(role=param['role'], content=mcp_content, **typed_dict_extras(param, ['role', 'content']))

class AnthropicMCPTypeConverter(ProviderToMCPConverter[MessageParam, Message]):
    """
    Convert between Anthropic and MCP types.
    """

    @classmethod
    def from_mcp_message_result(cls, result: MCPMessageResult) -> Message:
        if result.role != 'assistant':
            raise ValueError(f"Expected role to be 'assistant' but got '{result.role}' instead.")
        return Message(role='assistant', type='message', content=[mcp_content_to_anthropic_content(result.content)], model=result.model, stop_reason=mcp_stop_reason_to_anthropic_stop_reason(result.stopReason), id=result.id or None, usage=result.usage or None)

    @classmethod
    def to_mcp_message_result(cls, result: Message) -> MCPMessageResult:
        contents = anthropic_content_to_mcp_content(result.content)
        if len(contents) > 1:
            raise NotImplementedError('Multiple content elements in a single message are not supported in MCP yet')
        mcp_content = contents[0]
        return MCPMessageResult(role=result.role, content=mcp_content, model=result.model, stopReason=anthropic_stop_reason_to_mcp_stop_reason(result.stop_reason), **result.model_dump(exclude={'role', 'content', 'model', 'stop_reason'}))

    @classmethod
    def from_mcp_message_param(cls, param: MCPMessageParam) -> MessageParam:
        extras = param.model_dump(exclude={'role', 'content'})
        return MessageParam(role=param.role, content=[mcp_content_to_anthropic_content(param.content, for_message_param=True)], **extras)

    @classmethod
    def to_mcp_message_param(cls, param: MessageParam) -> MCPMessageParam:
        contents = anthropic_content_to_mcp_content(param.content)
        if len(contents) > 1:
            raise NotImplementedError('Multiple content elements in a single message are not supported')
        mcp_content = contents[0]
        return MCPMessageParam(role=param.role, content=mcp_content, **typed_dict_extras(param, ['role', 'content']))

    @classmethod
    def from_mcp_tool_result(cls, result: CallToolResult, tool_use_id: str) -> MessageParam:
        """Convert mcp tool result to user MessageParam"""
        tool_result_block_content: list[TextBlockParam | ImageBlockParam] = []
        for content in result.content:
            converted_content = mcp_content_to_anthropic_content(content, for_message_param=True)
            if converted_content['type'] in ['text', 'image']:
                tool_result_block_content.append(converted_content)
        if not tool_result_block_content:
            tool_result_block_content = [TextBlockParam(type='text', text='No result returned')]
            result.isError = True
        return MessageParam(role='user', content=[ToolResultBlockParam(type='tool_result', tool_use_id=tool_use_id, content=tool_result_block_content, is_error=result.isError)])

class TestBedrockAugmentedLLM:
    """
    Tests for the BedrockAugmentedLLM class.
    """

    @pytest.fixture
    def mock_llm(self, mock_context):
        """
        Creates a mock Bedrock LLM instance with common mocks set up.
        """
        mock_context.config.bedrock = MagicMock()
        mock_context.config.bedrock = BedrockSettings(api_key='test_key')
        mock_context.config.bedrock.default_model = 'us.amazon.nova-lite-v1:0'
        llm = BedrockAugmentedLLM(name='test', context=mock_context)
        llm.history = MagicMock()
        llm.history.get = MagicMock(return_value=[])
        llm.history.set = MagicMock()
        llm.select_model = AsyncMock(return_value='us.amazon.nova-lite-v1:0')
        llm._log_chat_progress = MagicMock()
        llm._log_chat_finished = MagicMock()
        llm.bedrock_client = MagicMock()
        llm.bedrock_client.converse = AsyncMock()
        return llm

    @staticmethod
    def create_text_response(text, stop_reason='end_turn', usage=None):
        """
        Creates a text response for testing.
        """
        return {'output': {'message': {'role': 'assistant', 'content': [{'text': text}]}}, 'stopReason': stop_reason, 'usage': usage or {'inputTokens': 150, 'outputTokens': 100, 'totalTokens': 250}}

    @staticmethod
    def create_tool_use_response(tool_name, tool_args, tool_id, stop_reason='tool_use', usage=None):
        """
        Creates a tool use response for testing.
        """
        return {'output': {'message': {'role': 'assistant', 'content': [{'toolUse': {'name': tool_name, 'input': tool_args, 'toolUseId': tool_id}}]}}, 'stopReason': stop_reason, 'usage': usage or {'inputTokens': 150, 'outputTokens': 100, 'totalTokens': 250}}

    @staticmethod
    def create_tool_result_message(tool_result, tool_id, status='success'):
        """
        Creates a tool result message for testing.
        """
        return {'role': 'user', 'content': [{'toolResult': {'content': tool_result, 'toolUseId': tool_id, 'status': status}}]}

    @staticmethod
    def create_multiple_tool_use_response(tool_uses, text_prefix=None, stop_reason='tool_use', usage=None):
        """
        Creates a response with multiple tool uses for testing.
        """
        content = []
        if text_prefix:
            content.append({'text': text_prefix})
        for tool_use in tool_uses:
            content.append({'toolUse': {'name': tool_use['name'], 'input': tool_use.get('input', {}), 'toolUseId': tool_use['toolUseId']}})
        return {'output': {'message': {'role': 'assistant', 'content': content}}, 'stopReason': stop_reason, 'usage': usage or {'inputTokens': 150, 'outputTokens': 100, 'totalTokens': 250}}

    @pytest.mark.asyncio
    async def test_basic_text_generation(self, mock_llm):
        """
        Tests basic text generation without tools.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('This is a test response'))
        responses = await mock_llm.generate('Test query')
        assert len(responses) == 1
        assert responses[0]['content'][0]['text'] == 'This is a test response'
        assert mock_llm.executor.execute.call_count == 1
        first_call_args = mock_llm.executor.execute.call_args[0][1]
        assert first_call_args.payload['modelId'] == 'us.amazon.nova-lite-v1:0'
        assert first_call_args.payload['messages'][0]['role'] == 'user'
        assert first_call_args.payload['messages'][0]['content'][0]['text'] == 'Test query'

    @pytest.mark.asyncio
    async def test_generate_str(self, mock_llm):
        """
        Tests the generate_str method which returns string output.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('This is a test response'))
        response_text = await mock_llm.generate_str('Test query')
        assert response_text == 'This is a test response'
        assert mock_llm.executor.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_structured(self, mock_llm):
        """
        Tests structured output generation using Instructor.
        """

        class TestResponseModel(BaseModel):
            name: str
            value: int
        mock_llm.generate_str = AsyncMock(return_value='name: Test, value: 42')
        mock_llm.executor.execute = AsyncMock(return_value=TestResponseModel(name='Test', value=42))
        result = await BedrockAugmentedLLM.generate_structured(mock_llm, 'Test query', TestResponseModel)
        assert isinstance(result, TestResponseModel)
        assert result.name == 'Test'
        assert result.value == 42

    @pytest.mark.asyncio
    async def test_with_history(self, mock_llm):
        """
        Tests generation with message history.
        """
        history_message = {'role': 'user', 'content': [{'text': 'Previous message'}]}
        mock_llm.history.get = MagicMock(return_value=[history_message])
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Response with history'))
        responses = await mock_llm.generate('Follow-up query', RequestParams(use_history=True))
        assert len(responses) == 1
        first_call_args = mock_llm.executor.execute.call_args[0][1]
        assert len(first_call_args.payload['messages']) >= 2
        assert first_call_args.payload['messages'][0] == history_message
        assert first_call_args.payload['messages'][1]['content'][0]['text'] == 'Follow-up query'

    @pytest.mark.asyncio
    async def test_without_history(self, mock_llm):
        """
        Tests generation without message history.
        """
        mock_history = MagicMock(return_value=[{'role': 'user', 'content': [{'text': 'Ignored history'}]}])
        mock_llm.history.get = mock_history
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Response without history'))
        await mock_llm.generate('New query', RequestParams(use_history=False))
        mock_history.assert_not_called()
        call_args = mock_llm.executor.execute.call_args[0][1]
        assert len([m for m in call_args.payload['messages'] if m.get('content') == 'Ignored history']) == 0

    @pytest.mark.asyncio
    async def test_tool_usage(self, mock_llm: BedrockAugmentedLLM):
        """
        Tests tool usage in the LLM.
        """
        call_count = 0

        async def custom_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return self.create_tool_use_response('test_tool', {'query': 'test query'}, 'tool_123')
            else:
                return self.create_text_response('Final response after tool use', stop_reason='end_turn')
        mock_llm.executor.execute = AsyncMock(side_effect=custom_side_effect)
        mock_llm.call_tool = AsyncMock(return_value=MagicMock(content=[TextContent(type='text', text='Tool result')], isError=False))
        responses = await mock_llm.generate('Test query with tool')
        assert len(responses) == 3
        assert 'toolUse' in responses[0]['content'][0]
        assert responses[0]['content'][0]['toolUse']['name'] == 'test_tool'
        assert responses[1]['content'][0]['toolResult']['toolUseId'] == 'tool_123'
        assert responses[2]['content'][0]['text'] == 'Final response after tool use'
        assert mock_llm.call_tool.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, mock_llm):
        """
        Tests handling of errors from tool calls.
        """
        call_count = 0

        async def custom_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return self.create_tool_use_response('test_tool', {'query': 'test query'}, 'tool_123')
            else:
                return self.create_text_response('Response after tool error', stop_reason='end_turn')
        mock_llm.executor.execute = AsyncMock(side_effect=custom_side_effect)
        mock_llm.call_tool = AsyncMock(return_value=MagicMock(content=[TextContent(type='text', text='Tool execution failed with error')], isError=True))
        responses = await mock_llm.generate('Test query with tool error')
        assert len(responses) == 3
        assert 'toolUse' in responses[0]['content'][0]
        assert responses[-1]['content'][0]['text'] == 'Response after tool error'
        assert mock_llm.call_tool.call_count == 1

    @pytest.mark.asyncio
    async def test_api_error_handling(self, mock_llm):
        """
        Tests handling of API errors.
        """
        mock_llm.executor.execute = AsyncMock(return_value=Exception('API Error'))
        responses = await mock_llm.generate('Test query with API error')
        assert len(responses) == 0
        assert mock_llm.executor.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_model_selection(self, mock_llm):
        """
        Tests model selection logic.
        """
        mock_llm.select_model = AsyncMock(return_value='us.amazon.nova-v3:0')
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Model selection test'))
        request_params = RequestParams(model='us.amazon.claude-v2:1')
        await mock_llm.generate('Test query', request_params)
        assert mock_llm.select_model.call_count == 1
        assert mock_llm.select_model.call_args[0][0].model == 'us.amazon.claude-v2:1'

    @pytest.mark.asyncio
    async def test_request_params_merging(self, mock_llm):
        """
        Tests merging of request parameters with defaults.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Params test'))
        request_params = RequestParams(maxTokens=2000, temperature=0.8, max_iterations=5)
        await mock_llm.generate('Test query', request_params)
        merged_params = mock_llm.get_request_params(request_params)
        assert merged_params.maxTokens == 2000
        assert merged_params.temperature == 0.8
        assert merged_params.max_iterations == 5
        assert merged_params.model == mock_llm.default_request_params.model

    def test_type_conversion(self):
        """
        Tests the BedrockMCPTypeConverter for converting between Bedrock and MCP types.
        """
        bedrock_message = {'role': 'assistant', 'content': [{'text': 'Test content'}]}
        mcp_result = BedrockMCPTypeConverter.to_mcp_message_param(bedrock_message)
        assert mcp_result.role == 'assistant'
        assert mcp_result.content.text == 'Test content'
        mcp_message = SamplingMessage(role='user', content=TextContent(type='text', text='Test MCP content'))
        bedrock_param = BedrockMCPTypeConverter.from_mcp_message_param(mcp_message)
        assert bedrock_param['role'] == 'user'
        assert isinstance(bedrock_param['content'], list)
        assert bedrock_param['content'][0]['text'] == 'Test MCP content'

    def test_content_block_conversions(self):
        """
        Tests conversion between MCP content formats and Bedrock content blocks.
        """
        text_content = [TextContent(type='text', text='Hello world')]
        bedrock_blocks = mcp_content_to_bedrock_content(text_content)
        assert len(bedrock_blocks) == 1
        assert bedrock_blocks[0]['text'] == 'Hello world'
        mcp_blocks = bedrock_content_to_mcp_content(bedrock_blocks)
        assert len(mcp_blocks) == 1
        assert isinstance(mcp_blocks[0], TextContent)
        assert mcp_blocks[0].text == 'Hello world'
        image_content = [ImageContent(type='image', data='base64data', mimeType='image/png')]
        bedrock_blocks = mcp_content_to_bedrock_content(image_content)
        assert len(bedrock_blocks) == 1
        assert bedrock_blocks[0]['image']['source'] == 'base64data'
        assert bedrock_blocks[0]['image']['format'] == 'image/png'

    @pytest.mark.asyncio
    async def test_stop_reasons(self, mock_llm):
        """
        Tests handling of different Bedrock stop reasons.
        """
        stop_reasons = ['end_turn', 'stop_sequence', 'max_tokens', 'guardrail_intervened', 'content_filtered']
        for stop_reason in stop_reasons:
            mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response(f'Response with {stop_reason}', stop_reason=stop_reason))
            responses = await mock_llm.generate(f'Test query with {stop_reason}')
            assert len(responses) == 1
            assert responses[0]['content'][0]['text'] == f'Response with {stop_reason}'
            assert mock_llm.executor.execute.call_count == 1
            mock_llm.executor.execute.reset_mock()

    def test_typed_dict_extras(self):
        """
        Tests the typed_dict_extras helper function.
        """
        test_dict = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
        extras = typed_dict_extras(test_dict, ['key1', 'key3'])
        assert 'key1' not in extras
        assert 'key3' not in extras
        assert extras['key2'] == 'value2'
        extras = typed_dict_extras(test_dict, [])
        assert len(extras) == 3
        extras = typed_dict_extras(test_dict, ['key1', 'key2', 'key3'])
        assert len(extras) == 0

    @pytest.mark.asyncio
    async def test_tool_configuration(self, mock_llm: BedrockAugmentedLLM):
        """
        Tests that tool configuration is properly set up.
        """
        mock_llm.agent.list_tools = AsyncMock(return_value=ListToolsResult(tools=[Tool(name='test_tool', description='A test tool', inputSchema={'type': 'object', 'properties': {'query': {'type': 'string'}}})]))
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Tool config test'))
        await mock_llm.generate('Test query with tools')
        call_kwargs = mock_llm.executor.execute.call_args[0][1]
        assert 'toolConfig' in call_kwargs.payload
        assert len(call_kwargs.payload['toolConfig']['tools']) == 1
        assert call_kwargs.payload['toolConfig']['tools'][0]['toolSpec']['name'] == 'test_tool'
        assert call_kwargs.payload['toolConfig']['toolChoice']['auto'] == {}

    @pytest.mark.asyncio
    async def test_generate_with_string_input(self, mock_llm):
        """
        Tests generate() method with string input.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('String input response'))
        responses = await mock_llm.generate('This is a simple string message')
        assert len(responses) == 1
        assert responses[0]['content'][0]['text'] == 'String input response'
        req = mock_llm.executor.execute.call_args[0][1]
        assert req.payload['messages'][0]['role'] == 'user'
        assert req.payload['messages'][0]['content'][0]['text'] == 'This is a simple string message'

    @pytest.mark.asyncio
    async def test_generate_with_message_param_input(self, mock_llm):
        """
        Tests generate() method with MessageParamT input (Bedrock message dict).
        """
        message_param = {'role': 'user', 'content': [{'text': 'This is a MessageParamT message'}]}
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('MessageParamT input response'))
        responses = await mock_llm.generate(message_param)
        assert len(responses) == 1
        assert responses[0]['content'][0]['text'] == 'MessageParamT input response'
        req = mock_llm.executor.execute.call_args[0][1]
        assert req.payload['messages'][0]['role'] == 'user'
        assert req.payload['messages'][0]['content'][0]['text'] == 'This is a MessageParamT message'

    @pytest.mark.asyncio
    async def test_generate_with_prompt_message_input(self, mock_llm):
        """
        Tests generate() method with PromptMessage input (MCP PromptMessage).
        """
        from mcp.types import PromptMessage, TextContent
        prompt_message = PromptMessage(role='user', content=TextContent(type='text', text='This is a PromptMessage'))
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('PromptMessage input response'))
        responses = await mock_llm.generate(prompt_message)
        assert len(responses) == 1
        assert responses[0]['content'][0]['text'] == 'PromptMessage input response'
        req = mock_llm.executor.execute.call_args[0][1]
        assert req.payload['messages'][0]['role'] == 'user'
        assert req.payload['messages'][0]['content'][0]['text'] == 'This is a PromptMessage'

    @pytest.mark.asyncio
    async def test_generate_with_mixed_message_types(self, mock_llm):
        """
        Tests generate() method with a list containing mixed message types.
        """
        from mcp.types import PromptMessage, TextContent
        messages = ['String message', {'role': 'user', 'content': [{'text': 'MessageParamT response'}]}, PromptMessage(role='user', content=TextContent(type='text', text='PromptMessage content'))]
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Mixed message types response'))
        responses = await mock_llm.generate(messages)
        assert len(responses) == 1
        assert responses[0]['content'][0]['text'] == 'Mixed message types response'

    @pytest.mark.asyncio
    async def test_generate_str_with_mixed_message_types(self, mock_llm):
        """
        Tests generate_str() method with mixed message types.
        """
        from mcp.types import PromptMessage, TextContent
        messages = ['String message', {'role': 'user', 'content': [{'text': 'MessageParamT response'}]}, PromptMessage(role='user', content=TextContent(type='text', text='PromptMessage content'))]
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Mixed types string response'))
        response_text = await mock_llm.generate_str(messages)
        assert response_text == 'Mixed types string response'

    @pytest.mark.asyncio
    async def test_generate_structured_with_mixed_message_types(self, mock_llm):
        """
        Tests generate_structured() method with mixed message types.
        """
        from pydantic import BaseModel
        from mcp.types import PromptMessage, TextContent

        class TestResponseModel(BaseModel):
            name: str
            value: int
        messages = ['String message', {'role': 'user', 'content': [{'text': 'MessageParamT response'}]}, PromptMessage(role='user', content=TextContent(type='text', text='PromptMessage content'))]
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('{"name": "MixedTypes", "value": 123}'))
        mock_llm.generate_str = AsyncMock(return_value='{"name": "MixedTypes", "value": 123}')
        mock_llm.executor.execute = AsyncMock(return_value=TestResponseModel(name='MixedTypes', value=123))
        result = await BedrockAugmentedLLM.generate_structured(mock_llm, messages, TestResponseModel)
        assert isinstance(result, TestResponseModel)
        assert result.name == 'MixedTypes'
        assert result.value == 123

    @pytest.mark.asyncio
    async def test_multiple_tool_usage(self, mock_llm: BedrockAugmentedLLM):
        """
        Tests multiple tool uses in a single response.
        Verifies that all tool results are combined into a single message.
        """
        mock_llm.executor.execute = AsyncMock(side_effect=[self.create_multiple_tool_use_response(tool_uses=[{'name': 'test_tool', 'input': {}, 'toolUseId': 'tool_1'}, {'name': 'test_tool', 'input': {}, 'toolUseId': 'tool_2'}], text_prefix='Processing with multiple tools'), self.create_text_response('Final response after both tools')])
        mock_llm.call_tool = AsyncMock(side_effect=[MagicMock(content=[TextContent(type='text', text='Tool 1 result')], isError=False), MagicMock(content=[TextContent(type='text', text='Tool 2 result')], isError=False)])
        responses = await mock_llm.generate('Test multiple tools')
        assert len(responses) == 3
        assert responses[0]['role'] == 'assistant'
        assert len(responses[0]['content']) == 3
        assert responses[1]['role'] == 'user'
        assert len(responses[1]['content']) == 2
        assert responses[1]['content'][0]['toolResult']['toolUseId'] == 'tool_1'
        assert responses[1]['content'][1]['toolResult']['toolUseId'] == 'tool_2'
        assert responses[2]['content'][0]['text'] == 'Final response after both tools'
        assert mock_llm.call_tool.call_count == 2

class TestAnthropicAugmentedLLM:
    """
    Tests for the AnthropicAugmentedLLM class.
    """

    @pytest.fixture
    def mock_llm(self, mock_context):
        """
        Creates a mock LLM instance with common mocks set up.
        """
        mock_context.config.anthropic = AnthropicSettings(api_key='test_key')
        mock_context.config.default_model = 'claude-3-7-sonnet-latest'
        llm = AnthropicAugmentedLLM(name='test', context=mock_context)
        llm.agent = MagicMock()
        llm.agent.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        llm.history = MagicMock()
        llm.history.get = MagicMock(return_value=[])
        llm.history.set = MagicMock()
        llm.select_model = AsyncMock(return_value='claude-3-7-sonnet-latest')
        llm._log_chat_progress = MagicMock()
        llm._log_chat_finished = MagicMock()
        llm.executor = MagicMock()
        llm.executor.execute = AsyncMock()
        return llm

    @pytest.fixture
    def default_usage(self):
        """
        Returns a default usage object for testing.
        """
        return Usage(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=2789, output_tokens=89)

    @staticmethod
    def create_tool_use_message(call_count, usage):
        """
        Creates a tool use message for testing.
        """
        return Message(role='assistant', content=[ToolUseBlock(type='tool_use', name='search_tool', input={'query': 'test query'}, id=f'tool_{call_count}')], model='claude-3-7-sonnet-latest', stop_reason='tool_use', id=f'resp_{call_count}', type='message', usage=usage)

    @staticmethod
    def create_text_message(text, usage, role='assistant', stop_reason='end_turn'):
        """
        Creates a text message for testing.
        """
        return Message(role=role, content=[TextBlock(type='text', text=text)], model='claude-3-7-sonnet-latest', stop_reason=stop_reason, id='final_response', type='message', usage=usage)

    @staticmethod
    def create_tool_result_message(result_text, tool_id, usage, is_error=False):
        """
        Creates a tool result message for testing.
        """
        return {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': tool_id, 'content': [{'type': 'text', 'text': result_text}], 'is_error': is_error}]}

    @staticmethod
    def check_final_iteration_prompt_in_messages(messages):
        """
        Checks if there's a final iteration prompt in the given messages.
        """
        for msg in messages:
            if msg.get('role') == 'user' and isinstance(msg.get('content'), str) and ('please stop using tools' in msg.get('content', '').lower()):
                return True
        return False

    def create_tool_use_side_effect(self, max_iterations, default_usage):
        """
        Creates a side effect function for tool use testing.
        """
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            messages = kwargs.get('messages', [])
            has_final_iteration_prompt = self.check_final_iteration_prompt_in_messages(messages)
            if call_count == max_iterations or has_final_iteration_prompt:
                return self.create_text_message('Here is my final answer based on all the tool results gathered so far...', default_usage, stop_reason='end_turn')
            else:
                return self.create_tool_use_message(call_count, default_usage)
        return side_effect

    @pytest.mark.asyncio
    async def test_basic_text_generation(self, mock_llm, default_usage):
        """
        Tests basic text generation without tools.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_message('This is a test response', default_usage))
        responses = await mock_llm.generate('Test query')
        assert len(responses) == 1
        assert responses[0].content[0].text == 'This is a test response'
        assert mock_llm.executor.execute.call_count == 1
        first_call_args = mock_llm.executor.execute.call_args[0][1]
        assert first_call_args.payload['model'] == 'claude-3-7-sonnet-latest'
        assert first_call_args.payload['messages'][0]['role'] == 'user'
        assert first_call_args.payload['messages'][0]['content'] == 'Test query'

    @pytest.mark.asyncio
    async def test_generate_str(self, mock_llm, default_usage):
        """
        Tests the generate_str method which returns string output.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_message('This is a test response', default_usage))
        response_text = await mock_llm.generate_str('Test query')
        assert response_text == 'This is a test response'
        assert mock_llm.executor.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_structured(self, mock_llm, default_usage):
        """
        Tests structured output generation using native Anthropic API.
        """
        from unittest.mock import patch

        class TestResponseModel(BaseModel):
            name: str
            value: int
        tool_use_block = ToolUseBlock(type='tool_use', id='tool_123', name='return_structured_output', input={'name': 'Test', 'value': 42})
        mock_message = Message(type='message', id='msg_123', role='assistant', content=[tool_use_block], model='claude-3-7-sonnet-latest', stop_reason='tool_use', usage=default_usage)
        with patch('mcp_agent.workflows.llm.augmented_llm_anthropic.AsyncAnthropic') as MockAsyncAnthropic:
            mock_client = MockAsyncAnthropic.return_value
            mock_stream = AsyncMock()
            mock_stream.get_final_message = AsyncMock(return_value=mock_message)
            mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_stream.__aexit__ = AsyncMock(return_value=None)
            mock_client.messages.stream = MagicMock(return_value=mock_stream)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            result = await AnthropicAugmentedLLM.generate_structured(mock_llm, 'Test query', TestResponseModel)
            assert isinstance(result, TestResponseModel)
            assert result.name == 'Test'
            assert result.value == 42

    @pytest.mark.asyncio
    async def test_with_history(self, mock_llm, default_usage):
        """
        Tests generation with message history.
        """
        history_message = {'role': 'user', 'content': 'Previous message'}
        mock_llm.history.get = MagicMock(return_value=[history_message])
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_message('Response with history', default_usage))
        responses = await mock_llm.generate('Follow-up query', RequestParams(use_history=True))
        assert len(responses) == 1
        first_call_args = mock_llm.executor.execute.call_args[0][1]
        assert len(first_call_args.payload['messages']) >= 2
        assert first_call_args.payload['messages'][0] == history_message
        assert first_call_args.payload['messages'][1]['content'] == 'Follow-up query'

    @pytest.mark.asyncio
    async def test_without_history(self, mock_llm, default_usage):
        """
        Tests generation without message history.
        """
        mock_history = MagicMock(return_value=[{'role': 'user', 'content': 'Ignored history'}])
        mock_llm.history.get = mock_history
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_message('Response without history', default_usage))
        await mock_llm.generate('New query', RequestParams(use_history=False))
        mock_history.assert_not_called()
        call_args = mock_llm.executor.execute.call_args[0][1]
        assert len([content for content in call_args.payload['messages'] if content == 'Ignored history']) == 0

    @pytest.mark.asyncio
    async def test_tool_usage(self, mock_llm, default_usage):
        """
        Tests tool usage in the LLM.
        """
        call_count = 0

        async def custom_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return self.create_tool_use_message(1, default_usage)
            else:
                return self.create_text_message('Final response after tool use', default_usage)
        mock_llm.executor.execute = AsyncMock(side_effect=custom_side_effect)
        mock_llm.call_tool = AsyncMock(return_value=MagicMock(content=[TextContent(type='text', text='Tool result')], isError=False, tool_call_id='tool_1'))
        responses = await mock_llm.generate('Test query with tool')
        assert len(responses) == 2
        assert responses[0].content[0].type == 'tool_use'
        assert responses[0].content[0].name == 'search_tool'
        assert responses[1].content[0].text == 'Final response after tool use'
        assert mock_llm.call_tool.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, mock_llm, default_usage):
        """
        Tests handling of errors from tool calls.
        """
        call_count = 0

        async def custom_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return self.create_tool_use_message(1, default_usage)
            else:
                return self.create_text_message('Response after tool error', default_usage)
        mock_llm.executor.execute = AsyncMock(side_effect=custom_side_effect)
        mock_llm.call_tool = AsyncMock(return_value=MagicMock(content=[TextContent(type='text', text='Tool execution failed with error')], isError=True, tool_call_id='tool_1'))
        responses = await mock_llm.generate('Test query with tool error')
        assert len(responses) == 2
        assert responses[0].content[0].type == 'tool_use'
        assert responses[1].content[0].text == 'Response after tool error'
        assert mock_llm.call_tool.call_count == 1

    @pytest.mark.asyncio
    async def test_api_error_handling(self, mock_llm):
        """
        Tests handling of API errors.
        """
        mock_llm.executor.execute = AsyncMock(return_value=Exception('API Error'))
        responses = await mock_llm.generate('Test query with API error')
        assert len(responses) == 0
        assert mock_llm.executor.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_model_selection(self, mock_llm, default_usage):
        """
        Tests model selection logic.
        """
        mock_llm.select_model = AsyncMock(return_value='claude-3-8-haiku-latest')
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_message('Model selection test', default_usage))
        request_params = RequestParams(model='claude-3-opus-latest')
        await mock_llm.generate('Test query', request_params)
        assert mock_llm.select_model.call_count == 1
        assert mock_llm.select_model.call_args[0][0].model == 'claude-3-opus-latest'

    @pytest.mark.asyncio
    async def test_request_params_merging(self, mock_llm, default_usage):
        """
        Tests merging of request parameters with defaults.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_message('Params test', default_usage))
        request_params = RequestParams(maxTokens=2000, temperature=0.8, max_iterations=5)
        await mock_llm.generate('Test query', request_params)
        merged_params = mock_llm.get_request_params(request_params)
        assert merged_params.maxTokens == 2000
        assert merged_params.temperature == 0.8
        assert merged_params.max_iterations == 5
        assert merged_params.model == mock_llm.default_request_params.model

    def test_type_conversion(self, default_usage):
        """
        Tests the AnthropicMCPTypeConverter for converting between Anthropic and MCP types.
        """
        anthropic_message = Message(role='assistant', content=[TextBlock(type='text', text='Test content')], model='claude-3-7-sonnet-latest', stop_reason='end_turn', id='test_id', type='message', usage=default_usage)
        mcp_result = AnthropicMCPTypeConverter.to_mcp_message_result(anthropic_message)
        assert mcp_result.role == 'assistant'
        assert mcp_result.content.text == 'Test content'
        assert mcp_result.stopReason == 'endTurn'
        assert mcp_result.id == 'test_id'
        mcp_message = SamplingMessage(role='user', content=TextContent(type='text', text='Test MCP content'))
        anthropic_param = AnthropicMCPTypeConverter.from_mcp_message_param(mcp_message)
        assert anthropic_param['role'] == 'user'
        assert len(anthropic_param['content']) == 1
        assert anthropic_param['content'][0]['type'] == 'text'
        assert anthropic_param['content'][0]['text'] == 'Test MCP content'

    def test_content_block_conversions(self):
        """
        Tests conversion between MCP content formats and Anthropic content blocks.
        """
        text_content = TextContent(type='text', text='Hello world')
        anthropic_content = mcp_content_to_anthropic_content(text_content, for_message_param=True)
        assert anthropic_content['type'] == 'text'
        assert anthropic_content['text'] == 'Hello world'
        anthropic_content_list = [anthropic_content]
        mcp_blocks = anthropic_content_to_mcp_content(anthropic_content_list)
        assert len(mcp_blocks) == 1
        assert isinstance(mcp_blocks[0], TextContent)
        assert mcp_blocks[0].text == 'Hello world'

    def test_stop_reason_conversion(self):
        """
        Tests conversion between MCP and Anthropic stop reasons.
        """
        assert mcp_stop_reason_to_anthropic_stop_reason('endTurn') == 'end_turn'
        assert mcp_stop_reason_to_anthropic_stop_reason('maxTokens') == 'max_tokens'
        assert mcp_stop_reason_to_anthropic_stop_reason('stopSequence') == 'stop_sequence'
        assert mcp_stop_reason_to_anthropic_stop_reason('toolUse') == 'tool_use'
        assert anthropic_stop_reason_to_mcp_stop_reason('end_turn') == 'endTurn'
        assert anthropic_stop_reason_to_mcp_stop_reason('max_tokens') == 'maxTokens'
        assert anthropic_stop_reason_to_mcp_stop_reason('stop_sequence') == 'stopSequence'
        assert anthropic_stop_reason_to_mcp_stop_reason('tool_use') == 'toolUse'

    @pytest.mark.asyncio
    async def test_system_prompt_handling(self, mock_llm, default_usage):
        """
        Tests system prompt is correctly passed to the API.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_message('System prompt test', default_usage))
        system_prompt = 'You are a helpful assistant that speaks like a pirate.'
        request_params = RequestParams(systemPrompt=system_prompt)
        await mock_llm.generate('Ahoy matey', request_params)
        call_args = mock_llm.executor.execute.call_args[0][1]
        assert call_args.payload['system'] == system_prompt

    def test_typed_dict_extras(self):
        """
        Tests the typed_dict_extras helper function.
        """
        test_dict = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
        extras = typed_dict_extras(test_dict, ['key1', 'key3'])
        assert 'key1' not in extras
        assert 'key3' not in extras
        assert extras['key2'] == 'value2'
        extras = typed_dict_extras(test_dict, [])
        assert len(extras) == 3
        extras = typed_dict_extras(test_dict, ['key1', 'key2', 'key3'])
        assert len(extras) == 0

    @pytest.mark.asyncio
    async def test_final_response_after_max_iterations_with_tool_use(self, mock_llm, default_usage):
        """
        Tests whether we get a final text response when reaching max_iterations with tool_use.
        """
        mock_llm.executor.execute = AsyncMock(side_effect=self.create_tool_use_side_effect(3, default_usage))
        mock_llm.call_tool = AsyncMock(return_value=MagicMock(content=[TextContent(type='text', text='Tool result')], isError=False, tool_call_id='tool_1'))
        request_params = RequestParams(model='claude-3-7-sonnet-latest', maxTokens=1000, max_iterations=3, use_history=True)
        responses = await mock_llm.generate('Test query', request_params)
        assert responses[-1].stop_reason == 'end_turn'
        assert responses[-1].content[0].type == 'text'
        assert 'final answer' in responses[-1].content[0].text.lower()
        assert mock_llm.executor.execute.call_count == request_params.max_iterations
        calls = mock_llm.executor.execute.call_args_list
        final_call_args = calls[-1][0][1]
        messages = final_call_args.payload['messages']
        assert self.check_final_iteration_prompt_in_messages(messages), 'No message requesting to stop using tools was found'

    @pytest.mark.asyncio
    async def test_generate_with_string_input(self, mock_llm, default_usage):
        """
        Tests generate() method with string input (Message type from Union).
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_message('String input response', default_usage))
        responses = await mock_llm.generate('This is a simple string message')
        assert len(responses) == 1
        assert responses[0].content[0].text == 'String input response'
        first_call_args = mock_llm.executor.execute.call_args[0][1]
        assert first_call_args.payload['messages'][0]['role'] == 'user'
        assert first_call_args.payload['messages'][0]['content'] == 'This is a simple string message'

    @pytest.mark.asyncio
    async def test_generate_with_message_param_input(self, mock_llm, default_usage):
        """
        Tests generate() method with MessageParamT input (Anthropic message dict).
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_message('MessageParamT input response', default_usage))
        message_param = {'role': 'user', 'content': 'This is a MessageParamT message'}
        responses = await mock_llm.generate(message_param)
        assert len(responses) == 1
        assert responses[0].content[0].text == 'MessageParamT input response'
        first_call_args = mock_llm.executor.execute.call_args[0][1]
        assert first_call_args.payload['messages'][0]['role'] == 'user'
        assert first_call_args.payload['messages'][0]['content'] == 'This is a MessageParamT message'

    @pytest.mark.asyncio
    async def test_generate_with_prompt_message_input(self, mock_llm, default_usage):
        """
        Tests generate() method with PromptMessage input (MCP PromptMessage).
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_message('PromptMessage input response', default_usage))
        prompt_message = PromptMessage(role='user', content=TextContent(type='text', text='This is a PromptMessage'))
        responses = await mock_llm.generate(prompt_message)
        assert len(responses) == 1
        assert responses[0].content[0].text == 'PromptMessage input response'

    @pytest.mark.asyncio
    async def test_generate_with_mixed_message_types(self, mock_llm, default_usage):
        """
        Tests generate() method with a list containing mixed message types.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_message('Mixed message types response', default_usage))
        messages = ['String message', {'role': 'assistant', 'content': 'MessageParamT response'}, PromptMessage(role='user', content=TextContent(type='text', text='PromptMessage content'))]
        responses = await mock_llm.generate(messages)
        assert len(responses) == 1
        assert responses[0].content[0].text == 'Mixed message types response'

    @pytest.mark.asyncio
    async def test_generate_str_with_mixed_message_types(self, mock_llm, default_usage):
        """
        Tests generate_str() method with mixed message types.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_message('Mixed types string response', default_usage))
        messages = ['String message', {'role': 'assistant', 'content': 'MessageParamT response'}, PromptMessage(role='user', content=TextContent(type='text', text='PromptMessage content'))]
        response_text = await mock_llm.generate_str(messages)
        assert response_text == 'Mixed types string response'

    @pytest.mark.asyncio
    async def test_generate_structured_with_mixed_message_types(self, mock_llm):
        """
        Tests generate_structured() method with mixed message types.
        """
        from unittest.mock import patch

        class TestResponseModel(BaseModel):
            name: str
            value: int
        messages = ['String message', {'role': 'assistant', 'content': 'MessageParamT response'}, PromptMessage(role='user', content=TextContent(type='text', text='PromptMessage content'))]
        tool_use_block = ToolUseBlock(type='tool_use', id='tool_456', name='return_structured_output', input={'name': 'MixedTypes', 'value': 123})
        mock_message = Message(type='message', id='msg_456', role='assistant', content=[tool_use_block], model='claude-3-7-sonnet-latest', stop_reason='tool_use', usage=Usage(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=100, output_tokens=50))
        with patch('mcp_agent.workflows.llm.augmented_llm_anthropic.AsyncAnthropic') as MockAsyncAnthropic:
            mock_client = MockAsyncAnthropic.return_value
            mock_stream = AsyncMock()
            mock_stream.get_final_message = AsyncMock(return_value=mock_message)
            mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_stream.__aexit__ = AsyncMock(return_value=None)
            mock_client.messages.stream = MagicMock(return_value=mock_stream)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            result = await mock_llm.generate_structured(messages, TestResponseModel)
            assert isinstance(result, TestResponseModel)
            assert result.name == 'MixedTypes'
            assert result.value == 123

    @pytest.mark.asyncio
    async def test_system_prompt_not_none_in_api_call(self, mock_llm, default_usage):
        """
        Tests that system prompt is not None when passed to anthropic.messages.create.
        This verifies the fix for the system prompt handling bug.
        """
        captured_payload = None

        async def capture_execute(*args, **kwargs):
            nonlocal captured_payload
            captured_payload = args[1].payload
            return self.create_text_message('Test response', default_usage)
        mock_llm.executor.execute = AsyncMock(side_effect=capture_execute)
        system_prompt = 'You are a helpful assistant.'
        request_params = RequestParams(systemPrompt=system_prompt)
        await mock_llm.generate('Test query', request_params)
        assert 'system' in captured_payload
        assert captured_payload['system'] == system_prompt
        assert captured_payload['system'] is not None
        mock_llm.instruction = 'You are a pirate assistant.'
        await mock_llm.generate('Test query')
        assert 'system' in captured_payload
        assert captured_payload['system'] == 'You are a pirate assistant.'
        assert captured_payload['system'] is not None
        mock_llm.instruction = 'Default instruction'
        request_params = RequestParams(systemPrompt='Override system prompt')
        await mock_llm.generate('Test query', request_params)
        assert 'system' in captured_payload
        assert captured_payload['system'] == 'Default instruction'
        assert captured_payload['system'] is not None
        mock_llm.instruction = None
        request_params = RequestParams()
        await mock_llm.generate('Test query', request_params)
        assert 'system' not in captured_payload

class TestAzureAugmentedLLM:
    """
    Tests for the AzureAugmentedLLM class.
    """

    @pytest.fixture
    def mock_llm(self, mock_context):
        """
        Creates a mock Azure LLM instance with common mocks set up.
        """
        from mcp_agent.config import AzureSettings
        azure_settings = AzureSettings(api_key='test_key', endpoint='https://test-endpoint.openai.azure.com', default_model='gpt-4o-mini', api_version='2025-04-01-preview', credential_scopes=['https://cognitiveservices.azure.com/.default'])
        mock_context.config.azure = azure_settings
        llm = AzureAugmentedLLM(name='test', context=mock_context)
        llm.history = MagicMock()
        llm.history.get = MagicMock(return_value=[])
        llm.history.set = MagicMock()
        llm.select_model = AsyncMock(return_value='gpt-4o-mini')
        llm._log_chat_progress = MagicMock()
        llm._log_chat_finished = MagicMock()
        llm.azure_client = MagicMock()
        llm.azure_client.complete = AsyncMock()
        llm.executor.execute_many = AsyncMock(side_effect=lambda tool_tasks: [ToolMessage(tool_call_id='tool_123', content='Tool result') if hasattr(task, 'cr_code') or hasattr(task, '__await__') else task for task in tool_tasks])
        return llm

    @pytest.fixture
    def default_usage(self):
        """
        Returns a default usage object for testing.
        """
        return {'completion_tokens': 100, 'prompt_tokens': 150, 'total_tokens': 250}

    @staticmethod
    def create_text_response(text, finish_reason='stop', usage=None):
        """
        Creates a text response for testing.
        """
        message = ChatResponseMessage(role='assistant', content=text)
        response = MagicMock()
        response.choices = [MagicMock(message=message, finish_reason=finish_reason, index=0)]
        response.id = 'chatcmpl-123'
        response.created = 1677858242
        response.model = 'gpt-4o-mini'
        response.usage = usage
        return response

    @staticmethod
    def create_tool_use_response(tool_name, tool_args, tool_id, finish_reason='tool_calls', usage=None):
        """
        Creates a tool use response for testing.
        """
        function_call = FunctionCall(name=tool_name, arguments=json.dumps(tool_args))
        tool_call = ChatCompletionsToolCall(id=tool_id, type='function', function=function_call)
        message = ChatResponseMessage(role='assistant', content=None, tool_calls=[tool_call])
        response = MagicMock()
        response.choices = [MagicMock(message=message, finish_reason=finish_reason, index=0)]
        response.id = 'chatcmpl-123'
        response.created = 1677858242
        response.model = 'gpt-4o-mini'
        response.usage = usage
        return response

    @pytest.mark.asyncio
    async def test_basic_text_generation(self, mock_llm: AzureAugmentedLLM, default_usage):
        """
        Tests basic text generation without tools.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('This is a test response', usage=default_usage))
        responses = await mock_llm.generate('Test query')
        assert len(responses) == 1
        assert responses[0].content == 'This is a test response'
        assert mock_llm.executor.execute.call_count == 1
        req = mock_llm.executor.execute.call_args_list[0][0][1]
        assert req.payload['model'] == 'gpt-4o-mini'
        assert isinstance(req.payload['messages'][0], UserMessage)
        assert req.payload['messages'][0].content == 'Test query'

    @pytest.mark.asyncio
    async def test_generate_str(self, mock_llm: AzureAugmentedLLM, default_usage):
        """
        Tests the generate_str method which returns string output.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('This is a test response', usage=default_usage))
        response_text = await mock_llm.generate_str('Test query')
        assert response_text == 'This is a test response'
        assert mock_llm.executor.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_structured(self, mock_llm: AzureAugmentedLLM, default_usage):
        """
        Tests structured output generation using Azure's JsonSchemaFormat.
        """

        class TestResponseModel(BaseModel):
            name: str
            value: int
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('{"name": "Test", "value": 42}', usage=default_usage))
        result = await mock_llm.generate_structured('Test query', TestResponseModel)
        assert isinstance(result, TestResponseModel)
        assert result.name == 'Test'
        assert result.value == 42
        req = mock_llm.executor.execute.call_args_list[0][0][1]
        assert 'response_format' in req.payload
        assert req.payload['response_format'].name == 'TestResponseModel'

    @pytest.mark.asyncio
    async def test_with_history(self, mock_llm: AzureAugmentedLLM, default_usage):
        """
        Tests generation with message history.
        """
        history_message = UserMessage(content='Previous message')
        mock_llm.history.get = MagicMock(return_value=[history_message])
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Response with history', usage=default_usage))
        responses = await mock_llm.generate('Follow-up query', RequestParams(use_history=True))
        assert len(responses) == 1
        req = mock_llm.executor.execute.call_args_list[0][0][1]
        assert len(req.payload['messages']) >= 2
        assert req.payload['messages'][0] == history_message
        assert isinstance(req.payload['messages'][1], UserMessage)
        assert req.payload['messages'][1].content == 'Follow-up query'

    @pytest.mark.asyncio
    async def test_without_history(self, mock_llm: AzureAugmentedLLM, default_usage):
        """
        Tests generation without message history.
        """
        mock_history = MagicMock(return_value=[UserMessage(content='Ignored history')])
        mock_llm.history.get = mock_history
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Response without history', usage=default_usage))
        await mock_llm.generate('New query', RequestParams(use_history=False))
        mock_history.assert_not_called()
        req = mock_llm.executor.execute.call_args[0][1]
        assert len(req.payload['messages']) == 2
        assert req.payload['messages'][0].content == 'New query'
        assert req.payload['messages'][1].content == 'Response without history'

    @pytest.mark.asyncio
    async def test_tool_usage(self, mock_llm, default_usage):
        """
        Tests tool usage in the LLM.
        """
        call_count = 0

        async def custom_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock(message=self.create_tool_use_response('test_tool', {'query': 'test query'}, 'tool_123', usage=default_usage).choices[0].message, finish_reason='tool_calls', index=0)]
                return mock_response
            else:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock(message=self.create_text_response('Final response after tool use', usage=default_usage).choices[0].message, finish_reason='stop', index=0)]
                return mock_response
        mock_llm.executor.execute = AsyncMock(side_effect=custom_side_effect)
        responses = await mock_llm.generate('Test query with tool')
        assert len(responses) == 3
        assert hasattr(responses[0], 'tool_calls')
        assert responses[0].tool_calls is not None
        assert responses[0].tool_calls[0].function.name == 'test_tool'
        assert responses[1].tool_call_id == 'tool_123'
        assert responses[2].content == 'Final response after tool use'

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, mock_llm, default_usage):
        """
        Tests handling of errors from tool calls.
        """
        mock_llm.executor.execute = AsyncMock(side_effect=[self.create_tool_use_response('test_tool', {'query': 'test query'}, 'tool_123', usage=default_usage), self.create_text_response('Response after tool error', usage=default_usage)])
        mock_llm.executor.execute_many = AsyncMock(return_value=[ToolMessage(tool_call_id='tool_123', content='Tool execution failed with error')])
        responses = await mock_llm.generate('Test query with tool error')
        assert len(responses) == 3
        assert responses[-1].content == 'Response after tool error'

    @pytest.mark.asyncio
    async def test_api_error_handling(self, mock_llm):
        """
        Tests handling of API errors.
        """
        mock_llm.executor.execute = AsyncMock(return_value=Exception('API Error'))
        responses = await mock_llm.generate('Test query with API error')
        assert len(responses) == 0
        assert mock_llm.executor.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_model_selection(self, mock_llm, default_usage):
        """
        Tests model selection logic.
        """
        mock_llm.select_model = AsyncMock(return_value='gpt-4-turbo')
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Model selection test', usage=default_usage))
        request_params = RequestParams(model='gpt-4-custom')
        await mock_llm.generate('Test query', request_params)
        assert mock_llm.select_model.call_count == 1
        assert mock_llm.select_model.call_args[0][0].model == 'gpt-4-custom'

    @pytest.mark.asyncio
    async def test_request_params_merging(self, mock_llm, default_usage):
        """
        Tests merging of request parameters with defaults.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Params test', usage=default_usage))
        request_params = RequestParams(maxTokens=2000, temperature=0.8, max_iterations=5)
        await mock_llm.generate('Test query', request_params)
        merged_params = mock_llm.get_request_params(request_params)
        assert merged_params.maxTokens == 2000
        assert merged_params.temperature == 0.8
        assert merged_params.max_iterations == 5
        assert merged_params.model == mock_llm.default_request_params.model

    def test_type_conversion(self):
        """
        Tests the MCPAzureTypeConverter for converting between Azure and MCP types.
        """
        azure_message = ChatResponseMessage(role='assistant', content='Test content')
        mcp_result = MCPAzureTypeConverter.to_mcp_message_result(azure_message)
        assert mcp_result.role == 'assistant'
        assert mcp_result.content.text == 'Test content'
        mcp_message = SamplingMessage(role='user', content=TextContent(type='text', text='Test MCP content'))
        azure_param = MCPAzureTypeConverter.from_mcp_message_param(mcp_message)
        assert azure_param.role == 'user'
        if isinstance(azure_param.content, str):
            assert azure_param.content == 'Test MCP content'
        else:
            assert isinstance(azure_param.content, list)
            assert len(azure_param.content) == 1
            assert isinstance(azure_param.content[0], TextContentItem)
            assert azure_param.content[0].text == 'Test MCP content'

    def test_content_type_handling(self):
        """
        Tests handling of different content types in messages.
        """
        text_content = 'Hello world'
        azure_message = ChatResponseMessage(role='assistant', content=text_content)
        converted = MCPAzureTypeConverter.to_mcp_message_result(azure_message)
        assert converted.content.text == text_content
        content_items = [TextContentItem(text='Hello'), TextContentItem(text='World')]
        message_with_items = UserMessage(content=content_items)
        message_str = AzureAugmentedLLM.message_param_str(None, message_with_items)
        assert 'Hello' in message_str
        assert 'World' in message_str

    def test_missing_azure_config(self, mock_context):
        """
        Tests that an error is raised when Azure configuration is missing.
        """
        mock_context.config.azure = None
        with pytest.raises(ValueError) as excinfo:
            AzureAugmentedLLM(name='test', context=mock_context)
        assert 'Azure configuration not found' in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_execute_tool_call_direct(self, mock_llm):
        """
        Tests the execute_tool_call method directly.
        """
        function_call = FunctionCall(name='test_tool', arguments=json.dumps({'param1': 'value1'}))
        tool_call = ChatCompletionsToolCall(id='tool_123', type='function', function=function_call)
        tool_result = CallToolResult(isError=False, content=[TextContent(type='text', text='Tool executed successfully')])
        mock_llm.call_tool = AsyncMock(return_value=tool_result)
        result = await mock_llm.execute_tool_call(tool_call)
        assert result is not None
        assert result.tool_call_id == 'tool_123'
        assert result.content == 'Tool executed successfully'
        mock_llm.call_tool.assert_called_once()
        call_args = mock_llm.call_tool.call_args[1]
        assert call_args['tool_call_id'] == 'tool_123'
        assert call_args['request'].params.name == 'test_tool'
        assert call_args['request'].params.arguments == {'param1': 'value1'}

    @pytest.mark.asyncio
    async def test_execute_tool_call_invalid_json(self, mock_llm):
        """
        Tests execute_tool_call with invalid JSON arguments.
        """
        function_call = FunctionCall(name='test_tool', arguments="{'invalid': json}")
        tool_call = ChatCompletionsToolCall(id='tool_123', type='function', function=function_call)
        from unittest.mock import AsyncMock
        mock_llm.call_tool = AsyncMock()
        result = await mock_llm.execute_tool_call(tool_call)
        assert result is not None
        assert result.tool_call_id == 'tool_123'
        assert 'Invalid JSON' in result.content
        assert not mock_llm.call_tool.called

    def test_message_str(self):
        """
        Tests the message_str method for different response types.
        """
        message_with_content = ChatResponseMessage(role='assistant', content='This is a test message')
        result = AzureAugmentedLLM.message_str(None, message_with_content)
        assert result == 'This is a test message'
        tool_call = ChatCompletionsToolCall(id='tool_123', type='function', function=FunctionCall(name='test_tool', arguments='{}'))
        message_without_content = ChatResponseMessage(role='assistant', content=None, tool_calls=[tool_call])
        result = AzureAugmentedLLM.message_str(None, message_without_content)
        assert str(tool_call) in result
        assert 'tool_calls' in result

    def test_message_param_str_with_various_content(self):
        """
        Tests the message_param_str method with various content types.
        """
        message_with_string = UserMessage(content='String content')
        result = AzureAugmentedLLM.message_param_str(None, message_with_string)
        assert result == 'String content'
        message_with_text_items = UserMessage(content=[TextContentItem(text='Text item 1'), TextContentItem(text='Text item 2')])
        result = AzureAugmentedLLM.message_param_str(None, message_with_text_items)
        assert 'Text item 1' in result
        assert 'Text item 2' in result
        image_url = ImageUrl(url='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=')
        message_with_image = UserMessage(content=[ImageContentItem(image_url=image_url)])
        result = AzureAugmentedLLM.message_param_str(None, message_with_image)
        assert 'Image url:' in result
        assert 'data:image/png;base64' in result
        message_without_content = UserMessage(content=None)
        result = AzureAugmentedLLM.message_param_str(None, message_without_content)
        assert result == "{'role': 'user'}"

    @pytest.mark.parametrize('str_only', [True, False])
    def test_mcp_content_to_azure_content(self, str_only):
        """
        Tests the mcp_content_to_azure_content helper function.
        """
        from mcp_agent.workflows.llm.augmented_llm_azure import mcp_content_to_azure_content
        text_content = TextContent(type='text', text='Test text')
        image_content = ImageContent(type='image', mimeType='image/png', data='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=')
        text_resource = TextResourceContents(uri='resource://dummy', text='Resource text')
        embedded_resource = EmbeddedResource(resource=text_resource, type='resource')
        result = mcp_content_to_azure_content([text_content], str_only=str_only)
        if str_only:
            assert isinstance(result, str)
            assert 'Test text' in result
        else:
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], TextContentItem)
            assert result[0].text == 'Test text'
        result = mcp_content_to_azure_content([text_content, image_content, embedded_resource], str_only=str_only)
        if str_only:
            assert isinstance(result, str)
            assert 'Test text' in result
            assert 'image/png' in result
            assert 'Resource text' in result
        else:
            assert isinstance(result, list)
            assert len(result) == 3
            assert isinstance(result[0], TextContentItem)
            assert isinstance(result[1], ImageContentItem)
            assert isinstance(result[2], TextContentItem)

    def test_azure_content_to_mcp_content(self):
        """
        Tests the azure_content_to_mcp_content helper function.
        """
        from mcp_agent.workflows.llm.augmented_llm_azure import azure_content_to_mcp_content
        string_content = 'Simple string content'
        result = azure_content_to_mcp_content(string_content)
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].text == 'Simple string content'
        content_items = [TextContentItem(text='Text item'), ImageContentItem(image_url=ImageUrl(url='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='))]
        result = azure_content_to_mcp_content(content_items)
        assert len(result) == 2
        assert isinstance(result[0], TextContent)
        assert result[0].text == 'Text item'
        assert isinstance(result[1], ImageContent)
        assert result[1].type == 'image'
        assert result[1].mimeType == 'image/png'
        result = azure_content_to_mcp_content(None)
        assert len(result) == 0

    def test_image_url_to_mime_and_base64(self):
        """
        Tests the image_url_to_mime_and_base64 helper function.
        """
        from mcp_agent.workflows.llm.augmented_llm_azure import image_url_to_mime_and_base64
        valid_url = ImageUrl(url='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=')
        mime_type, base64_data = image_url_to_mime_and_base64(valid_url)
        assert mime_type == 'image/png'
        assert base64_data == 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='
        invalid_url = ImageUrl(url='invalid-data-url')
        with pytest.raises(ValueError) as excinfo:
            image_url_to_mime_and_base64(invalid_url)
        assert 'Invalid image data URI' in str(excinfo.value)

    def test_typed_dict_extras(self):
        """
        Tests the typed_dict_extras helper function.
        """
        from mcp_agent.workflows.llm.augmented_llm_azure import typed_dict_extras
        test_dict = {'field1': 'value1', 'field2': 'value2', 'exclude_me': 'value3', 'also_exclude': 'value4'}
        result = typed_dict_extras(test_dict, ['exclude_me', 'also_exclude'])
        assert 'field1' in result
        assert 'field2' in result
        assert 'exclude_me' not in result
        assert 'also_exclude' not in result
        assert result['field1'] == 'value1'
        assert result['field2'] == 'value2'
        result = typed_dict_extras({}, ['any_field'])
        assert result == {}
        result = typed_dict_extras(test_dict, [])
        assert len(result) == 4
        assert 'exclude_me' in result

    def test_type_converter_comprehensive(self):
        """
        Comprehensive tests for the MCPAzureTypeConverter.
        """
        user_message = SamplingMessage(role='user', content=TextContent(type='text', text='User content'))
        azure_user = MCPAzureTypeConverter.from_mcp_message_param(user_message)
        assert azure_user.role == 'user'
        assistant_message = SamplingMessage(role='assistant', content=TextContent(type='text', text='Assistant content'))
        azure_assistant = MCPAzureTypeConverter.from_mcp_message_param(assistant_message)
        assert azure_assistant.role == 'assistant'
        with pytest.raises(ValueError) as excinfo:
            MCPAzureTypeConverter.from_mcp_message_param(SamplingMessage(role='unsupported_role', content=TextContent(type='text', text='content')))
        assert "Input should be 'user' or 'assistant'" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_parallel_tool_calls(self, mock_llm, default_usage):
        """
        Tests parallel tool calls where multiple tools are called in a single response.
        """
        function_call1 = FunctionCall(name='tool1', arguments=json.dumps({'param': 'value1'}))
        function_call2 = FunctionCall(name='tool2', arguments=json.dumps({'param': 'value2'}))
        tool_call1 = ChatCompletionsToolCall(id='call_1', type='function', function=function_call1)
        tool_call2 = ChatCompletionsToolCall(id='call_2', type='function', function=function_call2)
        message = ChatResponseMessage(role='assistant', content=None, tool_calls=[tool_call1, tool_call2])
        response = MagicMock()
        response.choices = [MagicMock(message=message, finish_reason='tool_calls', index=0)]
        response.id = 'chatcmpl-123'
        response.created = 1677858242
        response.model = 'gpt-4o-mini'
        response.usage = default_usage
        mock_llm.executor.execute = AsyncMock(side_effect=[response, self.create_text_response('Final response after parallel tools', usage=default_usage)])
        mock_llm.executor.execute_many = AsyncMock(return_value=[ToolMessage(tool_call_id='call_1', content='Tool 1 result'), ToolMessage(tool_call_id='call_2', content='Tool 2 result')])
        request_params = RequestParams(parallel_tool_calls=True)
        responses = await mock_llm.generate('Test parallel tools', request_params)
        assert len(responses) >= 3
        assert hasattr(responses[0], 'tool_calls')
        assert len(responses[0].tool_calls) == 2
        assert 'tool1' in [tc.function.name for tc in responses[0].tool_calls]
        assert 'tool2' in [tc.function.name for tc in responses[0].tool_calls]

    @pytest.mark.asyncio
    async def test_multiple_iterations(self, mock_llm, default_usage):
        """
        Tests multiple iterations of generate with multiple tool calls.
        """
        mock_llm.executor.execute = AsyncMock(side_effect=[self.create_tool_use_response('tool_iter1', {'query': 'data1'}, 'tool_id1', usage=default_usage), self.create_tool_use_response('tool_iter2', {'query': 'data2'}, 'tool_id2', usage=default_usage), self.create_text_response('Final response after multiple iterations', usage=default_usage)])
        mock_llm.executor.execute_many = AsyncMock(side_effect=[[ToolMessage(tool_call_id='tool_id1', content='Result from first tool')], [ToolMessage(tool_call_id='tool_id2', content='Result from second tool')]])
        request_params = RequestParams(max_iterations=5)
        responses = await mock_llm.generate('Test multiple iterations', request_params)
        assert len(responses) > 4
        assert mock_llm.executor.execute.call_count == 3
        tool_call_responses = [r for r in responses if hasattr(r, 'tool_calls') and r.tool_calls]
        tool_result_responses = [r for r in responses if hasattr(r, 'tool_call_id')]
        text_responses = [r for r in responses if hasattr(r, 'content') and r.content]
        assert len(tool_call_responses) == 2
        assert len(tool_result_responses) == 2
        assert len(text_responses) >= 2
        assert 'Final response' in responses[-1].content

    @pytest.mark.asyncio
    async def test_system_prompt_handling(self, mock_llm, default_usage):
        """
        Tests handling of system prompts in generate requests.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Response with system prompt', usage=default_usage))
        test_prompt = 'This is a test system prompt'
        mock_llm.instruction = test_prompt
        mock_llm.history.get = MagicMock(return_value=[])
        await mock_llm.generate('Test query')
        req = mock_llm.executor.execute.call_args_list[0][0][1]
        messages = req.payload['messages']
        assert len(messages) >= 2
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == test_prompt
        request_prompt = 'Override system prompt'
        request_params = RequestParams(systemPrompt=request_prompt)
        mock_llm.executor.execute.reset_mock()
        await mock_llm.generate('Test query', request_params)
        req = mock_llm.executor.execute.call_args_list[0][0][1]
        messages = req.payload['messages']
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == test_prompt

    @pytest.mark.asyncio
    async def test_execute_tool_call_exception(self, mock_llm):
        """
        Tests execute_tool_call with an exception during tool call.
        """
        function_call = FunctionCall(name='failing_tool', arguments=json.dumps({'param': 'value'}))
        tool_call = ChatCompletionsToolCall(id='tool_123', type='function', function=function_call)
        mock_llm.call_tool = AsyncMock(side_effect=Exception('Tool execution failed'))
        result = await mock_llm.execute_tool_call(tool_call)
        assert result is not None
        assert result.tool_call_id == 'tool_123'
        assert 'Error executing tool' in result.content
        assert 'Tool execution failed' in result.content

    def test_convert_message_to_message_param(self):
        """
        Tests the convert_message_to_message_param method.
        """
        response_message = ChatResponseMessage(role='assistant', content='Test response content', tool_calls=[ChatCompletionsToolCall(id='tool_123', type='function', function=FunctionCall(name='test_tool', arguments='{}'))])
        param_message = AzureAugmentedLLM.convert_message_to_message_param(response_message)
        assert isinstance(param_message, AssistantMessage)
        assert param_message.content == 'Test response content'
        assert param_message.tool_calls is not None
        assert len(param_message.tool_calls) == 1
        assert param_message.tool_calls[0].function.name == 'test_tool'

    @pytest.mark.asyncio
    async def test_generate_with_string_input(self, mock_llm, default_usage):
        """
        Tests generate() method with string input.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('String input response', usage=default_usage))
        responses = await mock_llm.generate('This is a simple string message')
        assert len(responses) == 1
        assert responses[0].content == 'String input response'
        req = mock_llm.executor.execute.call_args[0][1]
        assert isinstance(req.payload['messages'][0], UserMessage)
        assert req.payload['messages'][0].content == 'This is a simple string message'

    @pytest.mark.asyncio
    async def test_generate_with_message_param_input(self, mock_llm, default_usage):
        """
        Tests generate() method with MessageParamT input (Azure message dict).
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('MessageParamT input response', usage=default_usage))
        message_param = UserMessage(content='This is a MessageParamT message')
        responses = await mock_llm.generate(message_param)
        assert len(responses) == 1
        assert responses[0].content == 'MessageParamT input response'
        req = mock_llm.executor.execute.call_args[0][1]
        assert isinstance(req.payload['messages'][0], UserMessage)
        assert req.payload['messages'][0].content == 'This is a MessageParamT message'

    @pytest.mark.asyncio
    async def test_generate_with_prompt_message_input(self, mock_llm, default_usage):
        """
        Tests generate() method with PromptMessage input (MCP PromptMessage).
        """
        from mcp.types import PromptMessage, TextContent
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('PromptMessage input response', usage=default_usage))
        prompt_message = PromptMessage(role='user', content=TextContent(type='text', text='This is a PromptMessage'))
        responses = await mock_llm.generate(prompt_message)
        assert len(responses) == 1
        assert responses[0].content == 'PromptMessage input response'
        req = mock_llm.executor.execute.call_args[0][1]
        assert isinstance(req.payload['messages'][0], UserMessage)
        assert req.payload['messages'][0].content[0].text == 'This is a PromptMessage'

    @pytest.mark.asyncio
    async def test_generate_with_mixed_message_types(self, mock_llm, default_usage):
        """
        Tests generate() method with a list containing mixed message types.
        """
        from mcp.types import PromptMessage, TextContent
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Mixed message types response', usage=default_usage))
        messages = ['String message', UserMessage(content='MessageParamT response'), PromptMessage(role='user', content=TextContent(type='text', text='PromptMessage content'))]
        responses = await mock_llm.generate(messages)
        assert len(responses) == 1
        assert responses[0].content == 'Mixed message types response'

    @pytest.mark.asyncio
    async def test_generate_str_with_mixed_message_types(self, mock_llm, default_usage):
        """
        Tests generate_str() method with mixed message types.
        """
        from mcp.types import PromptMessage, TextContent
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Mixed types string response', usage=default_usage))
        messages = ['String message', UserMessage(content='MessageParamT response'), PromptMessage(role='user', content=TextContent(type='text', text='PromptMessage content'))]
        response_text = await mock_llm.generate_str(messages)
        assert response_text == 'Mixed types string response'

    @pytest.mark.asyncio
    async def test_generate_structured_with_mixed_message_types(self, mock_llm, default_usage):
        """
        Tests generate_structured() method with mixed message types.
        """
        from pydantic import BaseModel
        from mcp.types import PromptMessage, TextContent

        class TestResponseModel(BaseModel):
            name: str
            value: int
        messages = ['String message', UserMessage(content='MessageParamT response'), PromptMessage(role='user', content=TextContent(type='text', text='PromptMessage content'))]
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('{"name": "MixedTypes", "value": 123}', usage=default_usage))
        result = await mock_llm.generate_structured(messages, TestResponseModel)
        assert isinstance(result, TestResponseModel)
        assert result.name == 'MixedTypes'
        assert result.value == 123

class TestGoogleAugmentedLLM:
    """
    Tests for the GoogleAugmentedLLM class.
    """

    @pytest.fixture
    def mock_llm(self, mock_context):
        """
        Creates a mock Google LLM instance with common mocks set up.
        """
        mock_context.config.google = GoogleSettings(api_key='test_api_key', default_model='gemini-2.0-flash')
        llm = GoogleAugmentedLLM(name='test', context=mock_context)
        llm.history = MagicMock()
        llm.history.get = MagicMock(return_value=[])
        llm.history.set = MagicMock()
        llm.select_model = AsyncMock(return_value='gemini-2.0-flash')
        llm._log_chat_progress = MagicMock()
        llm._log_chat_finished = MagicMock()
        llm.google_client = MagicMock()
        llm.google_client.models = MagicMock()
        llm.google_client.models.generate_content = AsyncMock()
        return llm

    @staticmethod
    def create_text_response(text, finish_reason='STOP', usage=None):
        """
        Creates a text response for testing in Google's format.
        """
        from google.genai import types
        return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role='model', parts=[types.Part.from_text(text=text)]), finish_reason=finish_reason, safety_ratings=[], citation_metadata=None)], prompt_feedback=None, usage_metadata=usage or {'prompt_token_count': 150, 'candidates_token_count': 100, 'total_token_count': 250})

    @staticmethod
    def create_tool_use_response(tool_name, tool_args, tool_id, finish_reason='STOP', usage=None):
        """
        Creates a tool use response for testing in Google's format.
        """
        from google.genai import types
        function_call = types.FunctionCall(name=tool_name, args=tool_args, id=tool_id)
        return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role='model', parts=[types.Part(function_call=function_call)]), finish_reason=finish_reason, safety_ratings=[], citation_metadata=None)], prompt_feedback=None, usage_metadata=usage or {'prompt_token_count': 150, 'candidates_token_count': 100, 'total_token_count': 250})

    @staticmethod
    def create_tool_result_message(tool_result, tool_name, status='success'):
        """
        Creates a tool result message for testing in Google's format.
        """
        from google.genai import types
        if status == 'success':
            function_response = {'result': tool_result}
        else:
            function_response = {'error': tool_result}
        return types.Content(role='tool', parts=[types.Part.from_function_response(name=tool_name, response=function_response)])

    @pytest.mark.asyncio
    async def test_basic_text_generation(self, mock_llm):
        """
        Tests basic text generation without tools.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('This is a test response'))
        responses = await mock_llm.generate('Test query')
        assert len(responses) == 1
        assert responses[0].parts[0].text == 'This is a test response'
        assert mock_llm.executor.execute.call_count == 1
        first_call_args = mock_llm.executor.execute.call_args[0][1]
        assert first_call_args.payload['model'] == 'gemini-2.0-flash'
        assert first_call_args.payload['contents'][0].role == 'user'
        assert first_call_args.payload['contents'][0].parts[0].text == 'Test query'

    @pytest.mark.asyncio
    async def test_generate_str(self, mock_llm):
        """
        Tests the generate_str method which returns string output.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('This is a test response'))
        response_text = await mock_llm.generate_str('Test query')
        assert response_text == 'This is a test response'
        assert mock_llm.executor.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_structured(self, mock_llm: GoogleAugmentedLLM):
        """
        Tests structured output generation using Instructor.
        """

        class TestResponseModel(BaseModel):
            name: str
            value: int
        import json
        json_content = json.dumps({'name': 'Test', 'value': 42})
        response = self.create_text_response(json_content)
        mock_llm.executor.execute = AsyncMock(return_value=response)
        result = await mock_llm.generate_structured('Test query', TestResponseModel)
        assert isinstance(result, TestResponseModel)
        assert result.name == 'Test'
        assert result.value == 42

    @pytest.mark.asyncio
    async def test_with_history(self, mock_llm: GoogleAugmentedLLM):
        """
        Tests generation with message history.
        """
        from google.genai import types
        history_message = types.Content(role='user', parts=[types.Part.from_text(text='Previous message')])
        mock_llm.history.get = MagicMock(return_value=[history_message])
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Response with history'))
        mock_llm.executor.execute_many = AsyncMock(return_value=[None])
        responses = await mock_llm.generate('Follow-up query', RequestParams(use_history=True))
        assert len(responses) == 1
        first_call_args = mock_llm.executor.execute.call_args_list[0][0]
        request_obj = first_call_args[1]
        assert len(request_obj.payload['contents']) >= 2
        assert request_obj.payload['contents'][0] == history_message
        assert request_obj.payload['contents'][1].parts[0].text == 'Follow-up query'

    @pytest.mark.asyncio
    async def test_without_history(self, mock_llm: GoogleAugmentedLLM):
        """
        Tests generation without message history.
        """
        from google.genai import types
        mock_history = MagicMock(return_value=[types.Content(role='user', parts=[types.Part.from_text(text='Ignored history')])])
        mock_llm.history.get = mock_history
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Response without history'))
        await mock_llm.generate('New query', RequestParams(use_history=False))
        mock_history.assert_not_called()
        mock_llm.executor.execute_many = AsyncMock(return_value=[None])
        call_args = mock_llm.executor.execute.call_args[0]
        request_obj = call_args[1]
        assert len([content for content in request_obj.payload['contents'] if content.parts[0].text == 'Ignored history']) == 0

    @pytest.mark.asyncio
    async def test_tool_usage(self, mock_llm: GoogleAugmentedLLM):
        """
        Tests tool usage in the LLM.
        """
        mock_tool_schema = {'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'The query for the tool'}}, 'required': ['query']}
        mock_tool_declaration = MagicMock()
        mock_tool_declaration.name = 'test_tool'
        mock_tool_declaration.description = 'A tool that executes a test query.'
        mock_tool_declaration.inputSchema = mock_tool_schema
        call_count = 0

        async def custom_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return self.create_tool_use_response(tool_name='test_tool', tool_args={'query': 'test query'}, tool_id='tool_123')
            elif call_count == 2:
                return self.create_text_response('Final response after tool use', finish_reason='STOP')
            raise AssertionError(f'custom_side_effect called too many times: {call_count}')
        mock_llm.executor.execute = AsyncMock(side_effect=custom_side_effect)
        mock_llm.executor.execute_many = AsyncMock(return_value=[None])
        mock_llm.call_tool = AsyncMock(return_value=MagicMock(content=[TextContent(type='text', text='Tool executed successfully: Tool result')], isError=False, tool_call_id='tool_123'))
        responses = await mock_llm.generate('Test query with tool')
        assert len(responses) == 2
        assert responses[0].parts[0].function_call is not None
        assert responses[0].parts[0].function_call.name == 'test_tool'
        assert responses[0].parts[0].function_call.args == {'query': 'test query'}
        assert responses[1].parts[0].text == 'Final response after tool use'

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, mock_llm: GoogleAugmentedLLM):
        """
        Tests handling of errors from tool calls.
        """
        mock_tool_schema = {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}
        mock_tool_declaration = MagicMock()
        mock_tool_declaration.name = 'test_tool'
        mock_tool_declaration.description = 'A test tool.'
        mock_tool_declaration.inputSchema = mock_tool_schema
        executor_call_count = 0

        async def custom_executor_side_effect(*args, **kwargs):
            nonlocal executor_call_count
            executor_call_count += 1
            if executor_call_count == 1:
                return self.create_tool_use_response(tool_name='test_tool', tool_args={'query': 'test query'}, tool_id='tool_error_123')
            elif executor_call_count == 2:
                return self.create_text_response('Response after tool error', finish_reason='STOP')
            raise AssertionError(f'custom_executor_side_effect called too many times: {executor_call_count}')
        mock_llm.executor.execute = AsyncMock(side_effect=custom_executor_side_effect)
        mock_llm.executor.execute_many = AsyncMock(return_value=[None])
        mock_llm.call_tool = AsyncMock(return_value=MagicMock(content=[TextContent(type='text', text='Tool execution failed with error')], isError=True, tool_call_id='tool_error_123'))
        responses = await mock_llm.generate('Test query with tool error')
        assert len(responses) == 2
        assert responses[0].parts[0].function_call is not None
        assert responses[0].parts[0].function_call.name == 'test_tool'
        assert responses[0].parts[0].function_call.args == {'query': 'test query'}
        assert responses[1].parts[0].text == 'Response after tool error'

    @pytest.mark.asyncio
    async def test_api_error_handling(self, mock_llm):
        """
        Tests handling of API errors.
        """
        mock_llm.executor.execute = AsyncMock(return_value=Exception('API Error'))
        responses = await mock_llm.generate('Test query with API error')
        assert len(responses) == 0
        assert mock_llm.executor.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_model_selection(self, mock_llm):
        """
        Tests model selection logic.
        """
        mock_llm.select_model = AsyncMock(return_value='gemini-2.0-pro')
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Model selection test'))
        request_params = RequestParams(model='gemini-1.5-flash')
        await mock_llm.generate('Test query', request_params)
        assert mock_llm.select_model.call_count == 1
        assert mock_llm.select_model.call_args[0][0].model == 'gemini-1.5-flash'

    @pytest.mark.asyncio
    async def test_request_params_merging(self, mock_llm):
        """
        Tests merging of request parameters with defaults.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Params test'))
        request_params = RequestParams(maxTokens=2000, temperature=0.8, max_iterations=5)
        await mock_llm.generate('Test query', request_params)
        merged_params = mock_llm.get_request_params(request_params)
        assert merged_params.maxTokens == 2000
        assert merged_params.temperature == 0.8
        assert merged_params.max_iterations == 5
        assert merged_params.model == mock_llm.default_request_params.model

    def test_type_conversion(self):
        """
        Tests the GoogleMCPTypeConverter for converting between Google and MCP types.
        """
        from google.genai import types
        google_message = types.Content(role='model', parts=[types.Part.from_text(text='Test content')])
        mcp_result = GoogleMCPTypeConverter.to_mcp_message_result(google_message)
        assert mcp_result.role == 'assistant'
        assert mcp_result.content.text == 'Test content'
        mcp_message = SamplingMessage(role='user', content=TextContent(type='text', text='Test MCP content'))
        google_param = GoogleMCPTypeConverter.from_mcp_message_param(mcp_message)
        assert google_param.role == 'user'
        assert len(google_param.parts) == 1
        assert google_param.parts[0].text == 'Test MCP content'

    def test_content_block_conversions(self):
        """
        Tests conversion between MCP content formats and Google content blocks.
        """
        text_content = [TextContent(type='text', text='Hello world')]
        google_parts = mcp_content_to_google_parts(text_content)
        assert len(google_parts) == 1
        assert google_parts[0].text == 'Hello world'
        mcp_blocks = google_parts_to_mcp_content(google_parts)
        assert len(mcp_blocks) == 1
        assert isinstance(mcp_blocks[0], TextContent)
        assert mcp_blocks[0].text == 'Hello world'
        import base64
        test_image_data = base64.b64encode(b'fake image data').decode('utf-8')
        image_content = [ImageContent(type='image', data=test_image_data, mimeType='image/png')]
        google_parts = mcp_content_to_google_parts(image_content)
        assert len(google_parts) == 1
        assert google_parts[0].file_data is None

    def test_transform_mcp_tool_schema(self):
        """
        Tests the transformation of MCP tool schema to Google compatible schema.
        """
        basic_schema = {'type': 'object', 'properties': {'name': {'type': 'string', 'description': 'The name'}, 'age': {'type': 'integer', 'minimum': 0}}, 'required': ['name']}
        transformed = transform_mcp_tool_schema(basic_schema)
        assert transformed['type'] == 'object'
        assert 'name' in transformed['properties']
        assert transformed['properties']['name']['type'] == 'string'
        assert 'age' in transformed['properties']
        assert transformed['properties']['age']['type'] == 'integer'
        assert transformed['properties']['age']['minimum'] == 0
        assert 'required' in transformed
        camel_case_schema = {'type': 'object', 'properties': {'longText': {'type': 'string', 'maxLength': 100}}}
        transformed = transform_mcp_tool_schema(camel_case_schema)
        assert 'max_length' in transformed['properties']['longText']
        assert transformed['properties']['longText']['max_length'] == 100
        nested_schema = {'type': 'object', 'properties': {'user': {'type': 'object', 'properties': {'firstName': {'type': 'string'}, 'lastName': {'type': 'string'}}}}}
        transformed = transform_mcp_tool_schema(nested_schema)
        assert 'user' in transformed['properties']
        assert transformed['properties']['user']['type'] == 'object'
        assert 'firstName' in transformed['properties']['user']['properties']
        assert 'lastName' in transformed['properties']['user']['properties']
        nullable_schema = {'type': 'object', 'properties': {'optionalField': {'anyOf': [{'type': 'string'}, {'type': 'null'}]}}}
        transformed = transform_mcp_tool_schema(nullable_schema)
        assert 'optionalField' in transformed['properties']
        assert transformed['properties']['optionalField']['type'] == 'string'
        assert transformed['properties']['optionalField']['nullable'] is True

    @pytest.mark.asyncio
    async def test_generate_with_string_input(self, mock_llm):
        """
        Tests generate() method with string input.
        """
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('String input response'))
        responses = await mock_llm.generate('This is a simple string message')
        assert len(responses) == 1
        assert responses[0].parts[0].text == 'String input response'
        req = mock_llm.executor.execute.call_args[0][1]
        assert req.payload['contents'][0].role == 'user'
        assert req.payload['contents'][0].parts[0].text == 'This is a simple string message'

    @pytest.mark.asyncio
    async def test_generate_with_message_param_input(self, mock_llm):
        """
        Tests generate() method with MessageParamT input (Google Content).
        """
        from google.genai import types
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('MessageParamT input response'))
        message_param = types.Content(role='user', parts=[types.Part.from_text(text='This is a MessageParamT message')])
        responses = await mock_llm.generate(message_param)
        assert len(responses) == 1
        assert responses[0].parts[0].text == 'MessageParamT input response'
        req = mock_llm.executor.execute.call_args[0][1]
        assert req.payload['contents'][0].role == 'user'
        assert req.payload['contents'][0].parts[0].text == 'This is a MessageParamT message'

    @pytest.mark.asyncio
    async def test_generate_with_prompt_message_input(self, mock_llm):
        """
        Tests generate() method with PromptMessage input (MCP PromptMessage).
        """
        from mcp.types import PromptMessage, TextContent
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('PromptMessage input response'))
        prompt_message = PromptMessage(role='user', content=TextContent(type='text', text='This is a PromptMessage'))
        responses = await mock_llm.generate(prompt_message)
        assert len(responses) == 1
        assert responses[0].parts[0].text == 'PromptMessage input response'
        req = mock_llm.executor.execute.call_args[0][1]
        assert req.payload['contents'][0].role == 'user'
        assert req.payload['contents'][0].parts[0].text == 'This is a PromptMessage'

    @pytest.mark.asyncio
    async def test_generate_with_mixed_message_types(self, mock_llm):
        """
        Tests generate() method with a list containing mixed message types.
        """
        from mcp.types import PromptMessage, TextContent
        from google.genai import types
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Mixed message types response'))
        messages = ['String message', types.Content(role='user', parts=[types.Part.from_text(text='MessageParamT response')]), PromptMessage(role='user', content=TextContent(type='text', text='PromptMessage content'))]
        responses = await mock_llm.generate(messages)
        assert len(responses) == 1
        assert responses[0].parts[0].text == 'Mixed message types response'

    @pytest.mark.asyncio
    async def test_generate_str_with_mixed_message_types(self, mock_llm):
        """
        Tests generate_str() method with mixed message types.
        """
        from mcp.types import PromptMessage, TextContent
        from google.genai import types
        mock_llm.executor.execute = AsyncMock(return_value=self.create_text_response('Mixed types string response'))
        messages = ['String message', types.Content(role='user', parts=[types.Part.from_text(text='MessageParamT response')]), PromptMessage(role='user', content=TextContent(type='text', text='PromptMessage content'))]
        response_text = await mock_llm.generate_str(messages)
        assert response_text == 'Mixed types string response'

    @pytest.mark.asyncio
    async def test_generate_structured_with_mixed_message_types(self, mock_llm):
        """
        Tests generate_structured() method with mixed message types.
        """
        from pydantic import BaseModel
        from mcp.types import PromptMessage, TextContent
        from google.genai import types

        class TestResponseModel(BaseModel):
            name: str
            value: int
        messages = ['String message', types.Content(role='user', parts=[types.Part.from_text(text='MessageParamT response')]), PromptMessage(role='user', content=TextContent(type='text', text='PromptMessage content'))]
        import json
        json_content = json.dumps({'name': 'MixedTypes', 'value': 123})
        response = self.create_text_response(json_content)
        mock_llm.executor.execute = AsyncMock(return_value=response)
        result = await mock_llm.generate_structured(messages, TestResponseModel)
        assert isinstance(result, TestResponseModel)
        assert result.name == 'MixedTypes'
        assert result.value == 123

    @pytest.mark.asyncio
    async def test_parallel_tool_calls(self, mock_llm: GoogleAugmentedLLM):
        """
        Tests that parallel tool calls return a single Content with multiple function response parts.
        """
        from google.genai import types
        parallel_tool_response = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role='model', parts=[types.Part(function_call=types.FunctionCall(name='tool1', args={'param': 'value1'}, id='call_1')), types.Part(function_call=types.FunctionCall(name='tool2', args={'param': 'value2'}, id='call_2'))]), finish_reason='STOP')])
        final_response = self.create_text_response('Final response after parallel tools')
        mock_llm.executor.execute = AsyncMock(side_effect=[parallel_tool_response, final_response])

        async def mock_execute_tool_call(function_call):
            if function_call.name == 'tool1':
                return types.Content(role='tool', parts=[types.Part.from_function_response(name='tool1', response={'result': 'Result from tool 1'})])
            elif function_call.name == 'tool2':
                return types.Content(role='tool', parts=[types.Part.from_function_response(name='tool2', response={'result': 'Result from tool 2'})])
        mock_llm.execute_tool_call = AsyncMock(side_effect=mock_execute_tool_call)
        mock_llm.executor.execute_many = AsyncMock(return_value=[types.Content(role='tool', parts=[types.Part.from_function_response(name='tool1', response={'result': 'Result from tool 1'})]), types.Content(role='tool', parts=[types.Part.from_function_response(name='tool2', response={'result': 'Result from tool 2'})])])
        original_messages = []

        def track_messages(messages):
            original_messages.extend(messages)
            return messages
        mock_llm.history.set = MagicMock(side_effect=track_messages)
        responses = await mock_llm.generate('Test parallel tool calls')
        assert len(responses) == 2
        assert len(responses[0].parts) == 2
        assert responses[0].parts[0].function_call.name == 'tool1'
        assert responses[0].parts[1].function_call.name == 'tool2'
        assert responses[1].parts[0].text == 'Final response after parallel tools'
        tool_messages = [msg for msg in original_messages if hasattr(msg, 'role') and msg.role == 'tool']
        assert len(tool_messages) == 1, f'Expected 1 tool message, got {len(tool_messages)}'
        tool_message = tool_messages[0]
        assert len(tool_message.parts) == 2, f'Expected 2 parts in tool message, got {len(tool_message.parts)}'
        part_names = [part.function_response.name for part in tool_message.parts if part.function_response]
        assert 'tool1' in part_names, 'tool1 response not found in combined message'
        assert 'tool2' in part_names, 'tool2 response not found in combined message'

def mcp_content_to_google_parts(content: list[TextContent | ImageContent | EmbeddedResource]) -> list[types.Part]:
    google_parts: list[types.Part] = []
    for block in content:
        if isinstance(block, TextContent):
            google_parts.append(types.Part.from_text(text=block.text))
        elif isinstance(block, ImageContent):
            google_parts.append(types.Part.from_bytes(data=base64.b64decode(block.data), mime_type=block.mimeType))
        elif isinstance(block, EmbeddedResource):
            if isinstance(block.resource, TextResourceContents):
                google_parts.append(types.Part.from_text(text=block.text))
            else:
                google_parts.append(types.Part.from_bytes(data=base64.b64decode(block.resource.blob), mime_type=block.resource.mimeType))
        else:
            google_parts.append(types.Part.from_text(text=str(block)))
    return google_parts

