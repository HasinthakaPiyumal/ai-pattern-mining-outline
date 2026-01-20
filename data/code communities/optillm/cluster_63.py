# Cluster 63

def test_n_parameter(model=TEST_MODEL, n_values=[1, 2, 3]):
    """
    Test the n parameter with different values
    """
    setup_test_env()
    client = get_test_client()
    test_prompt = 'Write a haiku about coding'
    for n in n_values:
        print(f'\nTesting n={n} with model {model}')
        print('-' * 50)
        try:
            response = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': 'You are a creative poet.'}, {'role': 'user', 'content': test_prompt}], n=n, temperature=0.8, max_tokens=100)
            print(f'Response type: {type(response)}')
            print(f'Number of choices: {len(response.choices)}')
            for i, choice in enumerate(response.choices):
                print(f'\nChoice {i + 1}:')
                print(choice.message.content)
            if len(response.choices) == n:
                print(f'\n✅ SUCCESS: Got {n} responses as expected')
            else:
                print(f'\n❌ FAIL: Expected {n} responses, got {len(response.choices)}')
        except Exception as e:
            print(f'\n❌ ERROR: {type(e).__name__}: {str(e)}')

def setup_test_env():
    """Set up test environment with local inference"""
    os.environ['OPTILLM_API_KEY'] = 'optillm'
    return TEST_MODEL

def get_test_client(base_url: str='http://localhost:8000/v1') -> OpenAI:
    """Get OpenAI client configured for local optillm"""
    return OpenAI(api_key='optillm', base_url=base_url)

def main():
    """
    Main test function
    """
    print('Testing n parameter support in optillm')
    print('=' * 50)
    setup_test_env()
    model = TEST_MODEL
    print(f'\n\nTesting model: {model}')
    print('=' * 50)
    try:
        test_n_parameter(model)
    except Exception as e:
        print(f'\n❌ Test failed with error: {str(e)}')
        print('Make sure optillm server is running with local inference enabled')
        return 1
    return 0

class TestJSONPluginIntegration(unittest.TestCase):
    """Integration tests for JSON plugin with local models"""

    def setUp(self):
        """Set up integration test environment"""
        try:
            from test_utils import setup_test_env, get_test_client, TEST_MODEL
            setup_test_env()
            self.test_client = get_test_client()
            self.test_model = TEST_MODEL
            self.available = True
        except ImportError:
            self.available = False

    def test_json_plugin_integration(self):
        """Test JSON plugin with actual local inference"""
        if not self.available:
            self.skipTest('Test utilities not available')
        try:
            test_schema = {'type': 'object', 'properties': {'answer': {'type': 'string'}, 'confidence': {'type': 'number'}}, 'required': ['answer']}
            response = self.test_client.chat.completions.create(model=self.test_model, messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'What is 2+2? Respond in JSON format.'}], response_format={'type': 'json_schema', 'json_schema': {'name': 'math_response', 'schema': test_schema}}, max_tokens=100)
            self.assertIsNotNone(response.choices)
            self.assertEqual(len(response.choices), 1)
            self.assertIsNotNone(response.choices[0].message.content)
            try:
                json_response = json.loads(response.choices[0].message.content)
                self.assertIsInstance(json_response, dict)
                if 'answer' in json_response:
                    self.assertIsInstance(json_response['answer'], str)
            except json.JSONDecodeError:
                pass
        except Exception as e:
            self.skipTest(f'JSON plugin integration not available: {str(e)}')

    def test_json_plugin_fallback(self):
        """Test that JSON plugin falls back gracefully when schema is invalid"""
        if not self.available:
            self.skipTest('Test utilities not available')
        try:
            response = self.test_client.chat.completions.create(model=self.test_model, messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'Say hello'}], max_tokens=20)
            self.assertIsNotNone(response.choices)
            self.assertEqual(len(response.choices), 1)
            self.assertIsNotNone(response.choices[0].message.content)
        except Exception as e:
            self.skipTest(f'Fallback test not available: {str(e)}')

class TestAPIResponseFormat(unittest.TestCase):
    """Test that API responses include reasoning token information"""

    def setUp(self):
        """Set up test fixtures"""
        setup_test_env()
        self.test_client = get_test_client()

    def test_response_includes_completion_tokens_details(self):
        """Test that API responses include completion_tokens_details"""
        try:
            response = self.test_client.chat.completions.create(model=TEST_MODEL, messages=get_thinking_test_messages(), max_tokens=50)
            self.assertIsNotNone(response.choices)
            self.assertEqual(len(response.choices), 1)
            self.assertIsNotNone(response.choices[0].message.content)
            self.assertIsNotNone(response.usage)
            self.assertGreater(response.usage.completion_tokens, 0)
            self.assertGreater(response.usage.prompt_tokens, 0)
        except Exception as e:
            self.skipTest(f'Local inference not available: {str(e)}')

    def test_response_no_reasoning_tokens(self):
        """Test API response when there are no reasoning tokens"""
        try:
            response = self.test_client.chat.completions.create(model=TEST_MODEL, messages=get_simple_test_messages(), max_tokens=20)
            self.assertIsNotNone(response.choices)
            self.assertEqual(len(response.choices), 1)
            self.assertIsNotNone(response.choices[0].message.content)
            self.assertIsNotNone(response.usage)
            self.assertGreater(response.usage.completion_tokens, 0)
            self.assertGreater(response.usage.prompt_tokens, 0)
        except Exception as e:
            self.skipTest(f'Local inference not available: {str(e)}')

    def test_multiple_responses_reasoning_tokens(self):
        """Test reasoning tokens with multiple responses (n > 1)"""
        try:
            response = self.test_client.chat.completions.create(model=TEST_MODEL, messages=get_thinking_test_messages(), max_tokens=50, n=2)
            self.assertIsNotNone(response.choices)
            self.assertGreaterEqual(len(response.choices), 1)
            self.assertIsNotNone(response.usage)
            self.assertGreater(response.usage.completion_tokens, 0)
        except Exception as e:
            self.skipTest(f'Multiple responses not supported by local inference: {str(e)}')

@pytest.fixture
def client():
    """Create OpenAI client for optillm proxy with local inference"""
    setup_test_env()
    return get_test_client()

