# Cluster 7

def get_schemas_openai(items: List[Callable]) -> List[Dict[str, Any]]:
    """Get function schemas for the OpenAI LLM from a list of functions.

    :param items: The functions to get function schemas for.
    :type items: List[Callable]
    :return: The schemas for the OpenAI LLM.
    :rtype: List[Dict[str, Any]]
    """
    schemas = []
    for item in items:
        if not callable(item):
            raise ValueError('Provided item must be a callable function.')
        basic_schema = get_schema(item)
        function_schema = {'name': basic_schema['name'], 'description': basic_schema['description'], 'parameters': {'type': 'object', 'properties': {}, 'required': []}}
        signature = inspect.signature(item)
        docstring = inspect.getdoc(item)
        param_doc_regex = re.compile(':param (\\w+):(.*?)\\n(?=:\\w|$)', re.S)
        doc_params = param_doc_regex.findall(docstring) if docstring else []
        for param_name, param in signature.parameters.items():
            param_type = param.annotation.__name__ if param.annotation != inspect.Parameter.empty else 'Any'
            param_description = 'No description available.'
            param_required = param.default is inspect.Parameter.empty
            for doc_param_name, doc_param_desc in doc_params:
                if doc_param_name == param_name:
                    param_description = doc_param_desc.strip()
                    break
            function_schema['parameters']['properties'][param_name] = {'type': convert_python_type_to_json_type(param_type), 'description': param_description}
            if param_required:
                function_schema['parameters']['required'].append(param_name)
        schemas.append({'type': 'function', 'function': function_schema})
    return schemas

def get_schema(item: Union[BaseModel, Callable]) -> Dict[str, Any]:
    """Get a function schema from a function or Pydantic BaseModel.

    :param item: The function or BaseModel to get the schema for.
    :type item: Union[BaseModel, Callable]
    :return: The function schema.
    :rtype: Dict[str, Any]
    """
    if isinstance(item, BaseModel):
        signature_parts = []
        for field_name, field_model in item.__annotations__.items():
            field_info = item.__fields__[field_name]
            default_value = field_info.default
            if default_value:
                default_repr = repr(default_value)
                signature_part = f'{field_name}: {field_model.__name__} = {default_repr}'
            else:
                signature_part = f'{field_name}: {field_model.__name__}'
            signature_parts.append(signature_part)
        signature = f'({', '.join(signature_parts)}) -> str'
        schema = {'name': item.__class__.__name__, 'description': item.__doc__, 'signature': signature}
    else:
        schema = {'name': item.__name__, 'description': str(inspect.getdoc(item)), 'signature': str(inspect.signature(item)), 'output': str(inspect.signature(item).return_annotation)}
    return schema

def convert_python_type_to_json_type(param_type: str) -> str:
    """Convert a Python type to a JSON type.

    :param param_type: The type of the parameter.
    :type param_type: str
    :return: The JSON type.
    :rtype: str
    """
    if param_type == 'int':
        return 'number'
    if param_type == 'float':
        return 'number'
    if param_type == 'str':
        return 'string'
    if param_type == 'bool':
        return 'boolean'
    if param_type == 'NoneType':
        return 'null'
    if param_type == 'list':
        return 'array'
    else:
        return 'object'

@pytest.fixture
def llamacpp_llm(mocker):
    mock_llama = mocker.patch('llama_cpp.Llama', spec=Llama)
    llm = mock_llama.return_value
    return LlamaCppLLM(llm=llm)

class TestLlamaCppLLM:

    def test_llama_cpp_import_errors(self, llamacpp_llm):
        with patch.dict('sys.modules', {'llama_cpp': None}):
            with pytest.raises(ImportError) as error:
                LlamaCppLLM(llamacpp_llm.llm)
        assert "Please install LlamaCPP to use Llama CPP llm. You can install it with: `pip install 'semantic-router[local]'`" in str(error.value)

    def test_llamacpp_llm_init_success(self, llamacpp_llm):
        assert llamacpp_llm.name == 'llama.cpp'
        assert llamacpp_llm.temperature == 0.2
        assert llamacpp_llm.max_tokens == 200
        assert llamacpp_llm.llm is not None

    def test_llamacpp_llm_call_success(self, llamacpp_llm, mocker):
        llamacpp_llm.llm.create_chat_completion = mocker.Mock(return_value={'choices': [{'message': {'content': 'test'}}]})
        llm_input = [Message(role='user', content='test')]
        output = llamacpp_llm(llm_input)
        assert output == 'test'

    def test_llamacpp_llm_grammar(self, llamacpp_llm):
        llamacpp_llm._grammar()

    def test_llamacpp_extract_function_inputs(self, llamacpp_llm, mocker):
        llamacpp_llm.llm.create_chat_completion = mocker.Mock(return_value={'choices': [{'message': {'content': "{'timezone': 'America/New_York'}"}}]})
        test_schema = {'name': 'get_time', 'description': 'Finds the current time in a specific timezone.\n\n:param timezone: The timezone to find the current time in, should\n    be a valid timezone from the IANA Time Zone Database like\n    "America/New_York" or "Europe/London". Do NOT put the place\n    name itself like "rome", or "new york", you must provide\n    the IANA format.\n:type timezone: str\n:return: The current time in the specified timezone.', 'signature': '(timezone: str) -> str', 'output': "<class 'str'>"}
        test_query = 'What time is it in America/New_York?'
        llamacpp_llm.extract_function_inputs(query=test_query, function_schemas=[test_schema])

    def test_llamacpp_extract_function_inputs_invalid(self, llamacpp_llm, mocker):
        with pytest.raises(ValueError):
            llamacpp_llm.llm.create_chat_completion = mocker.Mock(return_value={'choices': [{'message': {'content': "{'time': 'America/New_York'}"}}]})
            test_schema = {'name': 'get_time', 'description': 'Finds the current time in a specific timezone.\n\n:param timezone: The timezone to find the current time in, should\n    be a valid timezone from the IANA Time Zone Database like\n    "America/New_York" or "Europe/London". Do NOT put the place\n    name itself like "rome", or "new york", you must provide\n    the IANA format.\n:type timezone: str\n:return: The current time in the specified timezone.', 'signature': '(timezone: str) -> str', 'output': "<class 'str'>"}
            test_query = 'What time is it in America/New_York?'
            llamacpp_llm.extract_function_inputs(query=test_query, function_schemas=[test_schema])

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

def get_time(timezone: str) -> str:
    """Finds the current time in a specific timezone.

    :param timezone: The timezone to find the current time in, should
        be a valid timezone from the IANA Time Zone Database like
        "America/New_York" or "Europe/London". Do NOT put the place
        name itself like "rome", or "new york", you must provide
        the IANA format.
    :type timezone: str
    :return: The current time in the specified timezone."""
    now = datetime.now(ZoneInfo(timezone))
    return now.strftime('%H:%M')

def parse_response(response: str):
    for call in response.function_call:
        args = call['arguments']
        if call['function_name'] == 'get_time':
            result = get_time(**args)
            print(result)
        if call['function_name'] == 'get_time_difference':
            result = get_time_difference(**args)
            print(result)
        if call['function_name'] == 'convert_time':
            result = convert_time(**args)
            print(result)

def get_time_difference(timezone1: str, timezone2: str) -> str:
    """Calculates the time difference between two timezones.
    :param timezone1: The first timezone, should be a valid timezone from the IANA Time Zone Database like "America/New_York" or "Europe/London".
    :param timezone2: The second timezone, should be a valid timezone from the IANA Time Zone Database like "America/New_York" or "Europe/London".
    :type timezone1: str
    :type timezone2: str
    :return: The time difference in hours between the two timezones."""
    now_utc = datetime.utcnow().replace(tzinfo=ZoneInfo('UTC'))
    tz1_time = now_utc.astimezone(ZoneInfo(timezone1))
    tz2_time = now_utc.astimezone(ZoneInfo(timezone2))
    tz1_offset = tz1_time.utcoffset().total_seconds()
    tz2_offset = tz2_time.utcoffset().total_seconds()
    hours_difference = (tz2_offset - tz1_offset) / 3600
    return f'The time difference between {timezone1} and {timezone2} is {hours_difference} hours.'

def convert_time(time: str, from_timezone: str, to_timezone: str) -> str:
    """Converts a specific time from one timezone to another.
    :param time: The time to convert in HH:MM format.
    :param from_timezone: The original timezone of the time, should be a valid IANA timezone.
    :param to_timezone: The target timezone for the time, should be a valid IANA timezone.
    :type time: str
    :type from_timezone: str
    :type to_timezone: str
    :return: The converted time in the target timezone.
    :raises ValueError: If the time format or timezone strings are invalid.

    Example:
        convert_time("12:30", "America/New_York", "Asia/Tokyo") -> "03:30"
    """
    try:
        today = datetime.now().date()
        datetime_string = f'{today} {time}'
        time_obj = datetime.strptime(datetime_string, '%Y-%m-%d %H:%M').replace(tzinfo=ZoneInfo(from_timezone))
        converted_time = time_obj.astimezone(ZoneInfo(to_timezone))
        formatted_time = converted_time.strftime('%H:%M')
        return formatted_time
    except Exception as e:
        raise ValueError(f'Error converting time: {e}')

def semantic_layer(query: str):
    route = rl(query)
    if route.name == 'get_time':
        query += f' (SYSTEM NOTE: {get_time()})'
    elif route.name == 'supplement_brand':
        query += f' (SYSTEM NOTE: {supplement_brand()})'
    elif route.name == 'business_inquiry':
        query += f' (SYSTEM NOTE: {business_inquiry()})'
    elif route.name == 'product':
        query += f' (SYSTEM NOTE: {product()})'
    else:
        pass
    return query

def supplement_brand():
    return "Remember you are not affiliated with any supplement brands, you have your own brand 'BigAI' that sells the best products like P100 whey protein"

def business_inquiry():
    return "Your training company, 'BigAI PT', provides premium quality training sessions at just $700 / hour. Users can find out more at www.aurelio.ai/train"

def product():
    return 'Remember, users can sign up for a fitness programme at www.aurelio.ai/sign-up'

def parse_response(response: str):
    for call in response.function_call:
        args = call['arguments']
        if call['function_name'] == 'get_time':
            result = get_time(**args)
            print(result)
        if call['function_name'] == 'get_time_difference':
            result = get_time_difference(**args)
            print(result)
        if call['function_name'] == 'convert_time':
            result = convert_time(**args)
            print(result)

def route_and_execute(query, functions, layer):
    route_choice: RouteChoice = layer(query)
    for function in functions:
        if function.__name__ == route_choice.name:
            if route_choice.function_call:
                return function(**route_choice.function_call)
    msgs = [Message(role='user', content=query)]
    return llm(msgs)

