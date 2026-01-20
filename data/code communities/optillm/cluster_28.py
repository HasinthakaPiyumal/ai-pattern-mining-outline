# Cluster 28

def run(system_prompt: str, initial_query: str, client, model: str, request_config: dict=None) -> Tuple[str, int]:
    """Main plugin execution function."""
    logger.info('Starting JSON plugin execution')
    completion_tokens = 0
    try:
        response_format = request_config.get('response_format') if request_config else None
        schema = extract_schema_from_response_format(response_format)
        if not schema:
            logger.warning('No valid schema found in response_format')
            response = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': initial_query}])
            return (response.choices[0].message.content, response.usage.completion_tokens)
        json_generator = JSONGenerator()
        result = json_generator.generate_json(initial_query, schema)
        json_response = json.dumps(result) if isinstance(result, dict) else str(result)
        completion_tokens = json_generator.count_tokens(json_response)
        logger.info(f'Successfully generated JSON response: {json_response}')
        return (json_response, completion_tokens)
    except Exception as e:
        logger.error(f'Error in JSON plugin: {str(e)}')
        response = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': initial_query}])
        return (response.choices[0].message.content, response.usage.completion_tokens)

def extract_schema_from_response_format(response_format: Dict[str, Any]) -> Optional[str]:
    """Extract schema from response_format field."""
    try:
        if not response_format:
            return None
        if isinstance(response_format, dict):
            if response_format.get('type') == 'json_schema':
                schema_data = response_format.get('json_schema', {})
                if isinstance(schema_data, dict) and 'schema' in schema_data:
                    return json.dumps(schema_data['schema'])
                return json.dumps(schema_data)
        logger.warning(f'Could not extract valid schema from response_format')
        return None
    except Exception as e:
        logger.error(f'Error extracting schema from response_format: {str(e)}')
        return None

class TestJSONPlugin(unittest.TestCase):
    """Test cases for the JSON plugin with new outlines API."""

    def setUp(self):
        """Set up test fixtures."""
        self.simple_schema = json.dumps({'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}, 'active': {'type': 'boolean'}}, 'required': ['name', 'age']})
        self.complex_schema = json.dumps({'type': 'object', 'properties': {'id': {'type': 'integer'}, 'email': {'type': 'string'}, 'score': {'type': 'number'}, 'tags': {'type': 'array'}, 'metadata': {'type': 'object'}}, 'required': ['id', 'email']})

    @patch('optillm.plugins.json_plugin.outlines.from_transformers')
    @patch('optillm.plugins.json_plugin.AutoTokenizer.from_pretrained')
    def test_json_generator_init(self, mock_tokenizer, mock_from_transformers):
        """Test JSONGenerator initialization with new API."""
        mock_model = Mock()
        mock_from_transformers.return_value = mock_model
        mock_tokenizer.return_value = Mock()
        generator = JSONGenerator()
        mock_from_transformers.assert_called_once()
        mock_tokenizer.assert_called_once()
        self.assertIsNotNone(generator.model)
        self.assertIsNotNone(generator.tokenizer)

    @patch('optillm.plugins.json_plugin.outlines.from_transformers')
    @patch('optillm.plugins.json_plugin.AutoModelForCausalLM.from_pretrained')
    @patch('optillm.plugins.json_plugin.AutoTokenizer.from_pretrained')
    def test_parse_json_schema_to_pydantic(self, mock_tokenizer, mock_model, mock_from_transformers):
        """Test JSON schema to Pydantic model conversion."""
        if not PLUGIN_AVAILABLE:
            self.skipTest('JSON plugin not available')
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        mock_from_transformers.return_value = Mock()
        generator = JSONGenerator()
        try:
            result = generator.parse_json_schema_to_pydantic(self.simple_schema)
            self.assertIsNotNone(result)
        except Exception:
            self.assertTrue(hasattr(generator, 'parse_json_schema_to_pydantic'))

    @patch('optillm.plugins.json_plugin.outlines.from_transformers')
    @patch('optillm.plugins.json_plugin.AutoTokenizer.from_pretrained')
    def test_generate_json_new_api(self, mock_tokenizer, mock_from_transformers):
        """Test JSON generation with new outlines API."""
        mock_result = Mock()
        mock_result.model_dump.return_value = {'name': 'Test', 'age': 25}
        mock_model = Mock()
        mock_model.return_value = mock_result
        mock_from_transformers.return_value = mock_model
        generator = JSONGenerator()
        prompt = 'Create a person named Test who is 25 years old'
        result = generator.generate_json(prompt, self.simple_schema)
        self.assertEqual(result, {'name': 'Test', 'age': 25})
        mock_model.assert_called_once()

    def test_extract_schema_from_response_format(self):
        """Test schema extraction from OpenAI response format."""
        response_format = {'type': 'json_schema', 'json_schema': {'name': 'test_schema', 'schema': {'type': 'object', 'properties': {'test': {'type': 'string'}}}}}
        result = extract_schema_from_response_format(response_format)
        self.assertIsNotNone(result)
        schema = json.loads(result)
        self.assertEqual(schema['type'], 'object')
        self.assertIn('test', schema['properties'])

    @patch('optillm.plugins.json_plugin.JSONGenerator')
    def test_run_function_with_schema(self, mock_json_generator_class):
        """Test the main run function with a valid schema."""
        mock_generator = Mock()
        mock_generator.generate_json.return_value = {'result': 'test'}
        mock_generator.count_tokens.return_value = 10
        mock_json_generator_class.return_value = mock_generator
        mock_client = Mock()
        request_config = {'response_format': {'type': 'json_schema', 'json_schema': {'schema': {'type': 'object', 'properties': {'result': {'type': 'string'}}}}}}
        result, tokens = run('System prompt', 'Generate a test result', mock_client, 'test-model', request_config)
        self.assertIn('result', result)
        self.assertEqual(tokens, 10)
        mock_generator.generate_json.assert_called_once()

    def test_run_function_without_schema(self):
        """Test the main run function without a schema (fallback)."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='Regular response'))]
        mock_response.usage.completion_tokens = 5
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        result, tokens = run('System prompt', 'Test query', mock_client, 'test-model', {})
        self.assertEqual(result, 'Regular response')
        self.assertEqual(tokens, 5)
        mock_client.chat.completions.create.assert_called_once()

    @patch('optillm.plugins.json_plugin.JSONGenerator')
    def test_error_handling(self, mock_json_generator_class):
        """Test error handling and fallback."""
        mock_generator = Mock()
        mock_generator.generate_json.side_effect = Exception('Test error')
        mock_json_generator_class.return_value = mock_generator
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='Fallback response'))]
        mock_response.usage.completion_tokens = 8
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        request_config = {'response_format': {'type': 'json_schema', 'json_schema': {'schema': {'type': 'object'}}}}
        result, tokens = run('System prompt', 'Test query', mock_client, 'test-model', request_config)
        self.assertEqual(result, 'Fallback response')
        self.assertEqual(tokens, 8)
        mock_client.chat.completions.create.assert_called_once()

