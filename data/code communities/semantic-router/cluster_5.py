# Cluster 5

class OpenAILLM(BaseLLM):
    """LLM for OpenAI. Requires an OpenAI API key from https://platform.openai.com/api-keys."""
    _client: Optional[openai.OpenAI] = PrivateAttr(default=None)
    _async_client: Optional[openai.AsyncOpenAI] = PrivateAttr(default=None)

    def __init__(self, name: Optional[str]=None, openai_api_key: Optional[str]=None, temperature: float=0.01, max_tokens: int=200):
        """Initialize the OpenAILLM.

        :param name: The name of the OpenAI model to use.
        :type name: Optional[str]
        :param openai_api_key: The OpenAI API key.
        :type openai_api_key: Optional[str]
        :param temperature: The temperature of the LLM.
        :type temperature: float
        :param max_tokens: The maximum number of tokens to generate.
        :type max_tokens: int
        """
        if name is None:
            name = EncoderDefault.OPENAI.value['language_model']
        super().__init__(name=name)
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError("OpenAI API key cannot be 'None'.")
        try:
            self._async_client = openai.AsyncOpenAI(api_key=api_key)
            self._client = openai.OpenAI(api_key=api_key)
        except Exception as e:
            raise ValueError(f'OpenAI API client failed to initialize. Error: {e}') from e
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _extract_tool_calls_info(self, tool_calls: List[ChatCompletionMessageToolCall]) -> List[Dict[str, Any]]:
        """Extract the tool calls information from the tool calls.

        :param tool_calls: The tool calls to extract the information from.
        :type tool_calls: List[ChatCompletionMessageToolCall]
        :return: The tool calls information.
        :rtype: List[Dict[str, Any]]
        """
        tool_calls_info = []
        for tool_call in tool_calls:
            if tool_call.function.arguments is None:
                raise ValueError('Invalid output, expected arguments to be specified for each tool call.')
            tool_calls_info.append({'function_name': tool_call.function.name, 'arguments': json.loads(tool_call.function.arguments)})
        return tool_calls_info

    async def async_extract_tool_calls_info(self, tool_calls: List[ChatCompletionMessageToolCall]) -> List[Dict[str, Any]]:
        """Extract the tool calls information from the tool calls.

        :param tool_calls: The tool calls to extract the information from.
        :type tool_calls: List[ChatCompletionMessageToolCall]
        :return: The tool calls information.
        :rtype: List[Dict[str, Any]]
        """
        tool_calls_info = []
        for tool_call in tool_calls:
            if tool_call.function.arguments is None:
                raise ValueError('Invalid output, expected arguments to be specified for each tool call.')
            tool_calls_info.append({'function_name': tool_call.function.name, 'arguments': json.loads(tool_call.function.arguments)})
        return tool_calls_info

    def __call__(self, messages: List[Message], function_schemas: Optional[List[Dict[str, Any]]]=None) -> str:
        """Call the OpenAILLM.

        :param messages: The messages to pass to the OpenAILLM.
        :type messages: List[Message]
        :param function_schemas: The function schemas to pass to the OpenAILLM.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :return: The response from the OpenAILLM.
        :rtype: str
        """
        if self._client is None:
            raise ValueError('OpenAI client is not initialized.')
        try:
            tools: Union[List[Dict[str, Any]], NotGiven] = function_schemas if function_schemas else NOT_GIVEN
            completion = self._client.chat.completions.create(model=self.name, messages=[m.to_openai() for m in messages], temperature=self.temperature, max_tokens=self.max_tokens, tools=tools)
            if function_schemas:
                tool_calls = completion.choices[0].message.tool_calls
                if tool_calls is None:
                    raise ValueError('Invalid output, expected a tool call.')
                if len(tool_calls) < 1:
                    raise ValueError('Invalid output, expected at least one tool to be specified.')
                output = str(self._extract_tool_calls_info(tool_calls))
            else:
                content = completion.choices[0].message.content
                if content is None:
                    raise ValueError('Invalid output, expected content.')
                output = content
            return output
        except Exception as e:
            logger.error(f'LLM error: {e}')
            raise Exception(f'LLM error: {e}') from e

    async def acall(self, messages: List[Message], function_schemas: Optional[List[Dict[str, Any]]]=None) -> str:
        """Call the OpenAILLM asynchronously.

        :param messages: The messages to pass to the OpenAILLM.
        :type messages: List[Message]
        :param function_schemas: The function schemas to pass to the OpenAILLM.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :return: The response from the OpenAILLM.
        :rtype: str
        """
        if self._async_client is None:
            raise ValueError('OpenAI async_client is not initialized.')
        try:
            tools: Union[List[Dict[str, Any]], NotGiven] = function_schemas if function_schemas is not None else NOT_GIVEN
            completion = await self._async_client.chat.completions.create(model=self.name, messages=[m.to_openai() for m in messages], temperature=self.temperature, max_tokens=self.max_tokens, tools=tools)
            if function_schemas:
                tool_calls = completion.choices[0].message.tool_calls
                if tool_calls is None:
                    raise ValueError('Invalid output, expected a tool call.')
                if len(tool_calls) < 1:
                    raise ValueError('Invalid output, expected at least one tool to be specified.')
                output = str(await self.async_extract_tool_calls_info(tool_calls))
            else:
                content = completion.choices[0].message.content
                if content is None:
                    raise ValueError('Invalid output, expected content.')
                output = content
            return output
        except Exception as e:
            logger.error(f'LLM error: {e}')
            raise Exception(f'LLM error: {e}') from e

    def extract_function_inputs(self, query: str, function_schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract the function inputs from the query.

        :param query: The query to extract the function inputs from.
        :type query: str
        :param function_schemas: The function schemas to extract the function inputs from.
        :type function_schemas: List[Dict[str, Any]]
        :return: The function inputs.
        :rtype: List[Dict[str, Any]]
        """
        system_prompt = 'You are an intelligent AI. Given a command or request from the user, call the function to complete the request.'
        messages = [Message(role='system', content=system_prompt), Message(role='user', content=query)]
        output = self(messages=messages, function_schemas=function_schemas)
        if not output:
            raise Exception('No output generated for extract function input')
        output = output.replace("'", '"')
        function_inputs = json.loads(output)
        if not self._is_valid_inputs(function_inputs, function_schemas):
            raise ValueError('Invalid inputs')
        return function_inputs

    async def async_extract_function_inputs(self, query: str, function_schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract the function inputs from the query asynchronously.

        :param query: The query to extract the function inputs from.
        :type query: str
        :param function_schemas: The function schemas to extract the function inputs from.
        :type function_schemas: List[Dict[str, Any]]
        :return: The function inputs.
        :rtype: List[Dict[str, Any]]
        """
        system_prompt = 'You are an intelligent AI. Given a command or request from the user, call the function to complete the request.'
        messages = [Message(role='system', content=system_prompt), Message(role='user', content=query)]
        output = await self.acall(messages=messages, function_schemas=function_schemas)
        if not output:
            raise Exception('No output generated for extract function input')
        output = output.replace("'", '"')
        function_inputs = json.loads(output)
        if not self._is_valid_inputs(function_inputs, function_schemas):
            raise ValueError('Invalid inputs')
        return function_inputs

    def _is_valid_inputs(self, inputs: List[Dict[str, Any]], function_schemas: List[Dict[str, Any]]) -> bool:
        """Determine if the functions chosen by the LLM exist within the function_schemas,
        and if the input arguments are valid for those functions.

        :param inputs: The inputs to check for validity.
        :type inputs: List[Dict[str, Any]]
        :param function_schemas: The function schemas to check against.
        :type function_schemas: List[Dict[str, Any]]
        :return: True if the inputs are valid, False otherwise.
        :rtype: bool
        """
        try:
            for input_dict in inputs:
                if 'function_name' not in input_dict or 'arguments' not in input_dict:
                    logger.error("Missing 'function_name' or 'arguments' in inputs")
                    return False
                function_name = input_dict['function_name']
                arguments = input_dict['arguments']
                matching_schema = next((schema['function'] for schema in function_schemas if schema['function']['name'] == function_name), None)
                if not matching_schema:
                    logger.error(f'No matching function schema found for function name: {function_name}')
                    return False
                if not self._validate_single_function_inputs(arguments, matching_schema):
                    logger.error(f'Validation failed for function name: {function_name}')
                    return False
            return True
        except Exception as e:
            logger.error(f'Input validation error: {str(e)}')
            return False

    def _validate_single_function_inputs(self, inputs: Dict[str, Any], function_schema: Dict[str, Any]) -> bool:
        """Validate the extracted inputs against the function schema.

        :param inputs: The inputs to validate.
        :type inputs: Dict[str, Any]
        :param function_schema: The function schema to validate against.
        :type function_schema: Dict[str, Any]
        :return: True if the inputs are valid, False otherwise.
        """
        try:
            parameters = function_schema['parameters']['properties']
            required_params = function_schema['parameters'].get('required', [])
            for param_name in required_params:
                if param_name not in inputs:
                    logger.error(f"Required input '{param_name}' missing from query")
                    return False
            for param_name, param_info in parameters.items():
                if param_name in inputs:
                    expected_type = param_info['type']
                    if expected_type == 'string' and (not isinstance(inputs[param_name], str)):
                        logger.error(f"Input type for '{param_name}' is not {expected_type}")
                        return False
            return True
        except Exception as e:
            logger.error(f'Single input validation error: {str(e)}')
            return False

class TestMessageDataclass:

    def test_message_creation(self):
        message = Message(role='user', content='Hello!')
        assert message.role == 'user'
        assert message.content == 'Hello!'
        with pytest.raises(ValidationError):
            Message(user_role='invalid_role', message='Hello!')

    def test_message_to_openai(self):
        message = Message(role='user', content='Hello!')
        openai_format = message.to_openai()
        assert openai_format == {'role': 'user', 'content': 'Hello!'}
        message = Message(role='invalid_role', content='Hello!')
        with pytest.raises(ValueError):
            message.to_openai()

    def test_message_to_cohere(self):
        message = Message(role='user', content='Hello!')
        cohere_format = message.to_cohere()
        assert cohere_format == {'role': 'user', 'message': 'Hello!'}

class TestOpenRouterLLM:

    def test_openrouter_llm_init_with_api_key(self, openrouter_llm):
        assert openrouter_llm._client is not None, 'Client should be initialized'
        assert openrouter_llm.name == 'mistralai/mistral-7b-instruct', 'Default name not set correctly'

    def test_openrouter_llm_init_success(self, mocker):
        mocker.patch('os.getenv', return_value='fake-api-key')
        llm = OpenRouterLLM()
        assert llm._client is not None

    def test_openrouter_llm_init_without_api_key(self, mocker):
        mocker.patch('os.getenv', return_value=None)
        with pytest.raises(ValueError) as _:
            OpenRouterLLM()

    def test_openrouter_llm_call_uninitialized_client(self, openrouter_llm):
        openrouter_llm._client = None
        with pytest.raises(ValueError) as e:
            llm_input = [Message(role='user', content='test')]
            openrouter_llm(llm_input)
        assert 'OpenRouter client is not initialized.' in str(e.value)

    def test_openrouter_llm_init_exception(self, mocker):
        mocker.patch('os.getenv', return_value='fake-api-key')
        mocker.patch('openai.OpenAI', side_effect=Exception('Initialization error'))
        with pytest.raises(ValueError) as e:
            OpenRouterLLM()
        assert 'OpenRouter API client failed to initialize. Error: Initialization error' in str(e.value)

    def test_openrouter_llm_call_success(self, openrouter_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = 'test'
        mocker.patch('os.getenv', return_value='fake-api-key')
        mocker.patch.object(openrouter_llm._client.chat.completions, 'create', return_value=mock_completion)
        llm_input = [Message(role='user', content='test')]
        output = openrouter_llm(llm_input)
        assert output == 'test'

@pytest.fixture
def openrouter_llm(mocker):
    mocker.patch('openai.Client')
    return OpenRouterLLM(openrouter_api_key='test_api_key')

class TestOpenAILLM:

    def test_azure_openai_llm_init_with_api_key(self, azure_openai_llm):
        assert azure_openai_llm._client is not None, 'Client should be initialized'
        assert azure_openai_llm.name == 'gpt-4o', 'Default name not set correctly'

    def test_azure_openai_llm_init_success(self, mocker):
        mocker.patch('os.getenv', return_value='fake-api-key')
        llm = AzureOpenAILLM()
        assert llm._client is not None

    def test_azure_openai_llm_init_without_api_key(self, mocker):
        mocker.patch('os.getenv', return_value=None)
        with pytest.raises(ValueError) as _:
            AzureOpenAILLM()

    def test_azure_openai_llm_init_without_azure_endpoint(self, mocker):
        mocker.patch('os.getenv', side_effect=lambda key, default=None: {'OPENAI_CHAT_MODEL_NAME': 'test-model-name'}.get(key, default))
        with pytest.raises(ValueError) as e:
            AzureOpenAILLM(openai_api_key='test_api_key')
        assert "Azure endpoint API key cannot be 'None'" in str(e.value)

    def test_azure_openai_llm_call_uninitialized_client(self, azure_openai_llm):
        azure_openai_llm._client = None
        with pytest.raises(ValueError) as e:
            llm_input = [Message(role='user', content='test')]
            azure_openai_llm(llm_input)
        assert 'AzureOpenAI client is not initialized.' in str(e.value)

    def test_azure_openai_llm_init_exception(self, mocker):
        mocker.patch('os.getenv', return_value='fake-api-key')
        mocker.patch('openai.AzureOpenAI', side_effect=Exception('Initialization error'))
        with pytest.raises(ValueError) as e:
            AzureOpenAILLM()
        assert 'AzureOpenAI API client failed to initialize. Error: Initialization error' in str(e.value)

    def test_azure_openai_llm_temperature_max_tokens_initialization(self):
        test_temperature = 0.5
        test_max_tokens = 100
        azure_llm = AzureOpenAILLM(openai_api_key='test_api_key', azure_endpoint='test_endpoint', temperature=test_temperature, max_tokens=test_max_tokens)
        assert azure_llm.temperature == test_temperature, 'Temperature not set correctly'
        assert azure_llm.max_tokens == test_max_tokens, 'Max tokens not set correctly'

    def test_azure_openai_llm_call_success(self, azure_openai_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = 'test'
        mocker.patch('os.getenv', return_value='fake-api-key')
        mocker.patch.object(azure_openai_llm._client.chat.completions, 'create', return_value=mock_completion)
        llm_input = [Message(role='user', content='test')]
        output = azure_openai_llm(llm_input)
        assert output == 'test'

@pytest.fixture
def azure_openai_llm(mocker):
    mocker.patch('openai.Client')
    return AzureOpenAILLM(openai_api_key='test_api_key', azure_endpoint='test_endpoint')

@pytest.fixture
def ollama_llm():
    return OllamaLLM()

class TestOllamaLLM:

    def test_ollama_llm_init_success(self, ollama_llm):
        assert ollama_llm.temperature == 0.2
        assert ollama_llm.name == 'openhermes'
        assert ollama_llm.max_tokens == 200
        assert ollama_llm.stream is False

    def test_ollama_llm_call_success(self, ollama_llm, mocker):
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {'message': {'content': 'test response'}}
        mocker.patch('requests.post', return_value=mock_response)
        output = ollama_llm([Message(role='user', content='test')])
        assert output == 'test response'

    def test_ollama_llm_error_handling(self, ollama_llm, mocker):
        mocker.patch('requests.post', side_effect=Exception('LLM error'))
        with pytest.raises(Exception) as exc_info:
            ollama_llm([Message(role='user', content='test')])
        assert 'LLM error' in str(exc_info.value)

class TestMistralAILLM:

    def test_mistral_llm_import_errors(self):
        with patch.dict('sys.modules', {'mistralai': None}):
            with pytest.raises(ImportError) as error:
                MistralAILLM()
        assert "Please install MistralAI to use MistralAI LLM. You can install it with: `pip install 'semantic-router[mistralai]'`" in str(error.value)

    def test_mistralai_llm_init_with_api_key(self, mistralai_llm):
        assert mistralai_llm._client is not None, 'Client should be initialized'
        assert mistralai_llm.name == 'mistral-tiny', 'Default name not set correctly'

    def test_mistralai_llm_init_success(self, mocker):
        mocker.patch('os.getenv', return_value='fake-api-key')
        llm = MistralAILLM()
        assert llm._client is not None

    def test_mistralai_llm_init_without_api_key(self, mocker):
        mocker.patch('os.getenv', return_value=None)
        with pytest.raises(ValueError) as _:
            MistralAILLM()

    def test_mistralai_llm_call_uninitialized_client(self, mistralai_llm):
        mistralai_llm._client = None
        with pytest.raises(ValueError) as e:
            llm_input = [Message(role='user', content='test')]
            mistralai_llm(llm_input)
        assert 'MistralAI client is not initialized.' in str(e.value)

    def test_mistralai_llm_init_exception(self, mocker):
        mocker.patch('mistralai.client.MistralClient', side_effect=Exception('Initialization error'))
        with pytest.raises(ValueError) as e:
            MistralAILLM()
        assert "MistralAI API key cannot be 'None'." in str(e.value)

    def test_mistralai_llm_call_success(self, mistralai_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = 'test'
        mocker.patch('os.getenv', return_value='fake-api-key')
        mocker.patch.object(mistralai_llm._client, 'chat', return_value=mock_completion)
        llm_input = [Message(role='user', content='test')]
        output = mistralai_llm(llm_input)
        assert output == 'test'

@pytest.fixture
def mistralai_llm(mocker):
    mocker.patch('mistralai.client.MistralClient')
    return MistralAILLM(mistralai_api_key='test_api_key')

class TestCohereLLM:

    def test_initialization_with_api_key(self, cohere_llm):
        assert cohere_llm._client is not None, 'Client should be initialized'
        assert cohere_llm.name == 'command', 'Default name not set correctly'

    def test_initialization_without_api_key(self, mocker, monkeypatch):
        monkeypatch.delenv('COHERE_API_KEY', raising=False)
        mocker.patch('cohere.Client')
        with pytest.raises(ValueError):
            CohereLLM()

    def test_call_method(self, cohere_llm, mocker):
        mock_llm = mocker.MagicMock()
        mock_llm.text = 'test'
        cohere_llm._client.chat.return_value = mock_llm
        llm_input = [Message(role='user', content='test')]
        result = cohere_llm(llm_input)
        assert isinstance(result, str), 'Result should be a str'
        cohere_llm._client.chat.assert_called_once()

    def test_raises_value_error_if_cohere_client_fails_to_initialize(self, mocker):
        mocker.patch('cohere.Client', side_effect=Exception('Failed to initialize client'))
        with pytest.raises(ValueError):
            CohereLLM(cohere_api_key='test_api_key')

    def test_raises_value_error_if_cohere_client_is_not_initialized(self, mocker):
        mocker.patch('cohere.Client', return_value=None)
        llm = CohereLLM(cohere_api_key='test_api_key')
        with pytest.raises(ValueError):
            llm('test')

    def test_call_method_raises_error_on_api_failure(self, cohere_llm, mocker):
        mocker.patch.object(cohere_llm._client, '__call__', side_effect=Exception('API call failed'))
        with pytest.raises(ValueError):
            cohere_llm('test')

@pytest.fixture
def cohere_llm(mocker):
    mocker.patch('cohere.Client')
    return CohereLLM(cohere_api_key='test_api_key')

@pytest.fixture
def openai_llm(mocker):
    mocker.patch('openai.Client')
    return OpenAILLM(openai_api_key='test_api_key')

class TestOpenAILLM:

    def test_openai_llm_init_with_api_key(self, openai_llm):
        assert openai_llm._client is not None, 'Client should be initialized'
        assert openai_llm.name == 'gpt-4o', 'Default name not set correctly'

    def test_openai_llm_init_success(self, mocker):
        mocker.patch('os.getenv', return_value='fake-api-key')
        llm = OpenAILLM()
        assert llm._client is not None

    def test_openai_llm_init_without_api_key(self, mocker):
        mocker.patch('os.getenv', return_value=None)
        with pytest.raises(ValueError) as _:
            OpenAILLM()

    def test_openai_llm_call_uninitialized_client(self, openai_llm):
        openai_llm._client = None
        with pytest.raises(ValueError) as e:
            llm_input = [Message(role='user', content='test')]
            openai_llm(llm_input)
        assert 'OpenAI client is not initialized.' in str(e.value)

    def test_openai_llm_init_exception(self, mocker):
        mocker.patch('os.getenv', return_value='fake-api-key')
        mocker.patch('openai.OpenAI', side_effect=Exception('Initialization error'))
        with pytest.raises(ValueError) as e:
            OpenAILLM()
        assert 'OpenAI API client failed to initialize. Error: Initialization error' in str(e.value)

    def test_openai_llm_call_success(self, openai_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.content = 'test'
        mocker.patch('os.getenv', return_value='fake-api-key')
        mocker.patch.object(openai_llm._client.chat.completions, 'create', return_value=mock_completion)
        llm_input = [Message(role='user', content='test')]
        output = openai_llm(llm_input)
        assert output == 'test'

    def test_get_schemas_openai_with_valid_callable(self):

        def sample_function(param1: int, param2: str='default') -> str:
            """Sample function for testing."""
            return f'param1: {param1}, param2: {param2}'
        expected_schema = [{'type': 'function', 'function': {'name': 'sample_function', 'description': 'Sample function for testing.', 'parameters': {'type': 'object', 'properties': {'param1': {'type': 'number', 'description': 'No description available.'}, 'param2': {'type': 'string', 'description': 'No description available.'}}, 'required': ['param1']}}}]
        schema = get_schemas_openai([sample_function])
        assert schema == expected_schema, 'Schema did not match expected output.'

    def test_get_schemas_openai_with_non_callable(self):
        non_callable = 'I am not a function'
        with pytest.raises(ValueError):
            get_schemas_openai([non_callable])

    def test_openai_llm_call_with_function_schema(self, openai_llm, mocker):
        mock_function = mocker.MagicMock(arguments='{"timezone":"America/New_York"}')
        mock_function.name = 'sample_function'
        mock_tool_call = mocker.MagicMock(function=mock_function)
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.tool_calls = [mock_tool_call]
        mocker.patch.object(openai_llm._client.chat.completions, 'create', return_value=mock_completion)
        llm_input = [Message(role='user', content='test')]
        function_schemas = [{'type': 'function', 'name': 'sample_function'}]
        output = openai_llm(llm_input, function_schemas)
        assert output == "[{'function_name': 'sample_function', 'arguments': {'timezone': 'America/New_York'}}]", 'Output did not match expected result with function schema'

    def test_openai_llm_call_with_invalid_tool_calls(self, openai_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.tool_calls = None
        mocker.patch.object(openai_llm._client.chat.completions, 'create', return_value=mock_completion)
        llm_input = [Message(role='user', content='test')]
        function_schemas = [{'type': 'function', 'name': 'sample_function'}]
        with pytest.raises(Exception) as exc_info:
            openai_llm(llm_input, function_schemas)
        expected_error_message = 'LLM error: Invalid output, expected a tool call.'
        actual_error_message = str(exc_info.value)
        assert expected_error_message in actual_error_message, f"Expected error message: '{expected_error_message}', but got: '{actual_error_message}'"

    def test_openai_llm_call_with_no_arguments_in_tool_calls(self, openai_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.tool_calls = [mocker.MagicMock(function=mocker.MagicMock(arguments=None))]
        mocker.patch.object(openai_llm._client.chat.completions, 'create', return_value=mock_completion)
        llm_input = [Message(role='user', content='test')]
        function_schemas = [{'type': 'function', 'name': 'sample_function'}]
        with pytest.raises(Exception) as exc_info:
            openai_llm(llm_input, function_schemas)
        expected_error_message = 'LLM error: Invalid output, expected arguments to be specified for each tool call.'
        actual_error_message = str(exc_info.value)
        assert expected_error_message in actual_error_message, f"Expected error message: '{expected_error_message}', but got: '{actual_error_message}'"

    def test_extract_function_inputs(self, openai_llm, mocker):
        query = 'fetch user data'
        function_schemas = get_user_data_schema
        mocker.patch.object(OpenAILLM, '__call__', return_value='[{"function_name": "get_user_data", "arguments": {"user_id": "123"}}]')
        result = openai_llm.extract_function_inputs(query, function_schemas)
        expected_messages = [Message(role='system', content='You are an intelligent AI. Given a command or request from the user, call the function to complete the request.'), Message(role='user', content=query)]
        openai_llm.__call__.assert_called_once_with(messages=expected_messages, function_schemas=function_schemas)
        assert result == [{'function_name': 'get_user_data', 'arguments': {'user_id': '123'}}], 'The function inputs should match the expected dictionary.'

    def test_openai_llm_call_with_no_tool_calls_specified(self, openai_llm, mocker):
        mock_completion = mocker.MagicMock()
        mock_completion.choices[0].message.tool_calls = []
        mocker.patch.object(openai_llm._client.chat.completions, 'create', return_value=mock_completion)
        llm_input = [Message(role='user', content='test')]
        function_schemas = [{'type': 'function', 'name': 'sample_function'}]
        with pytest.raises(Exception) as exc_info:
            openai_llm(llm_input, function_schemas)
        expected_error_message = 'LLM error: Invalid output, expected at least one tool to be specified.'
        assert str(exc_info.value) == expected_error_message, f"Expected error message: '{expected_error_message}', but got: '{str(exc_info.value)}'"

    def test_extract_function_inputs_no_output(self, openai_llm, mocker):
        query = 'fetch user data'
        function_schemas = [{'type': 'function', 'name': 'get_user_data'}]
        mocker.patch.object(OpenAILLM, '__call__', return_value='')
        with pytest.raises(Exception) as exc_info:
            openai_llm.extract_function_inputs(query, function_schemas)
        assert str(exc_info.value) == 'No output generated for extract function input', 'Expected exception message not found'

    def test_extract_function_inputs_invalid_output(self, openai_llm, mocker):
        query = 'fetch user data'
        function_schemas = [{'type': 'function', 'name': 'get_user_data'}]
        mocker.patch.object(OpenAILLM, '__call__', return_value='[{"function_name": "get_user_data", "arguments": {"user_id": "123"}}]')
        mocker.patch.object(OpenAILLM, '_is_valid_inputs', return_value=False)
        with pytest.raises(ValueError) as exc_info:
            openai_llm.extract_function_inputs(query, function_schemas)
        assert str(exc_info.value) == 'Invalid inputs', 'Expected exception message not found'

    def test_is_valid_inputs_missing_function_name(self, openai_llm, mocker):
        mocked_logger = mocker.patch('semantic_router.utils.logger.logger.error')
        inputs = [{'arguments': {'user_id': '123'}}]
        function_schemas = get_user_data_schema
        result = openai_llm._is_valid_inputs(inputs, function_schemas)
        assert not result, "The method should return False when 'function_name' is missing"
        mocked_logger.assert_called_once_with("Missing 'function_name' or 'arguments' in inputs")

    def test_is_valid_inputs_missing_arguments(self, openai_llm, mocker):
        mocked_logger = mocker.patch('semantic_router.utils.logger.logger.error')
        inputs = [{'function_name': 'get_user_data'}]
        function_schemas = get_user_data_schema
        result = openai_llm._is_valid_inputs(inputs, function_schemas)
        assert not result, "The method should return False when 'arguments' is missing"
        mocked_logger.assert_called_once_with("Missing 'function_name' or 'arguments' in inputs")

    def test_is_valid_inputs_no_matching_schema(self, openai_llm, mocker):
        mocked_logger = mocker.patch('semantic_router.utils.logger.logger.error')
        inputs = [{'function_name': 'name_that_does_not_exist_in_schema', 'arguments': {'user_id': '123'}}]
        function_schemas = get_user_data_schema
        result = openai_llm._is_valid_inputs(inputs, function_schemas)
        assert not result, 'The method should return False when no matching function schema is found'
        expected_error_message = 'No matching function schema found for function name: name_that_does_not_exist_in_schema'
        mocked_logger.assert_called_once_with(expected_error_message)

    def test_is_valid_inputs_validation_failed(self, openai_llm, mocker):
        mocked_logger = mocker.patch('semantic_router.utils.logger.logger.error')
        inputs = [{'function_name': 'get_user_data', 'arguments': {'user_id': 123}}]
        function_schemas = get_user_data_schema
        mocker.patch.object(OpenAILLM, '_validate_single_function_inputs', return_value=False)
        result = openai_llm._is_valid_inputs(inputs, function_schemas)
        assert not result, 'The method should return False when validation fails'
        expected_error_message = 'Validation failed for function name: get_user_data'
        mocked_logger.assert_called_once_with(expected_error_message)

    def test_is_valid_inputs_exception_handling(self, openai_llm, mocker):
        mocked_logger = mocker.patch('semantic_router.utils.logger.logger.error')
        inputs = [{'function_name': 'get_user_data', 'arguments': {'user_id': '123'}}]
        function_schemas = get_user_data_schema
        mocker.patch.object(OpenAILLM, '_validate_single_function_inputs', side_effect=Exception('Test exception'))
        result = openai_llm._is_valid_inputs(inputs, function_schemas)
        assert not result, 'The method should return False when an exception occurs'
        mocked_logger.assert_called_once_with('Input validation error: Test exception')

    def test_validate_single_function_inputs_missing_required_param(self, openai_llm, mocker):
        mocked_logger = mocker.patch('semantic_router.utils.logger.logger.error')
        function_schema = example_function_schema
        inputs = {}
        result = openai_llm._validate_single_function_inputs(inputs, function_schema)
        assert not result, 'The method should return False when a required parameter is missing'
        expected_error_message = "Required input 'user_id' missing from query"
        mocked_logger.assert_called_once_with(expected_error_message)

    def test_validate_single_function_inputs_incorrect_type(self, openai_llm, mocker):
        mocked_logger = mocker.patch('semantic_router.utils.logger.logger.error')
        function_schema = example_function_schema
        inputs = {'user_id': 123}
        result = openai_llm._validate_single_function_inputs(inputs, function_schema)
        assert not result, 'The method should return False when input type is incorrect'
        expected_error_message = "Input type for 'user_id' is not string"
        mocked_logger.assert_called_once_with(expected_error_message)

    def test_validate_single_function_inputs_exception_handling(self, openai_llm, mocker):
        mocked_logger = mocker.patch('semantic_router.utils.logger.logger.error')

        class SchemaSimulator:

            def __getitem__(self, item):
                raise Exception('Test exception')
        function_schema = SchemaSimulator()
        result = openai_llm._validate_single_function_inputs({'user_id': '123'}, function_schema)
        assert not result, 'The method should return False when an exception occurs'
        mocked_logger.assert_called_once_with('Single input validation error: Test exception')

