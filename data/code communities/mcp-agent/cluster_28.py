# Cluster 28

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

def mcp_content_to_bedrock_content(content: list[TextContent | ImageContent | EmbeddedResource]) -> list[ContentBlockUnionTypeDef]:
    bedrock_content: list[ContentBlockUnionTypeDef] = []
    for block in content:
        if isinstance(block, TextContent):
            bedrock_content.append({'text': block.text})
        elif isinstance(block, ImageContent):
            bedrock_content.append({'image': {'format': block.mimeType, 'source': block.data}})
        elif isinstance(block, EmbeddedResource):
            if isinstance(block.resource, TextResourceContents):
                bedrock_content.append({'text': block.resource.text})
            else:
                bedrock_content.append({'document': {'format': block.resource.mimeType, 'source': block.resource.blob}})
        else:
            bedrock_content.append({'text': str(block)})
    return bedrock_content

def bedrock_content_to_mcp_content(content: list[ContentBlockUnionTypeDef]) -> list[TextContent | ImageContent | EmbeddedResource]:
    mcp_content = []
    for block in content:
        if block.get('text'):
            mcp_content.append(TextContent(type='text', text=block['text']))
        elif block.get('image'):
            mcp_content.append(ImageContent(type='image', data=block['image']['source'], mimeType=block['image']['format']))
        elif block.get('toolUse'):
            mcp_content.append(TextContent(type='text', text=str(block['toolUse'])))
        elif block.get('document'):
            mcp_content.append(EmbeddedResource(type='document', resource=BlobResourceContents(mimeType=block['document']['format'], blob=block['document']['source'])))
    return mcp_content

