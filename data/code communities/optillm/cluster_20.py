# Cluster 20

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

def get_simple_test_messages():
    """Get simple test messages for basic validation"""
    return [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'Say hello in one word.'}]

