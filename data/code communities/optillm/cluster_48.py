# Cluster 48

class TestInferenceStructures(unittest.TestCase):
    """Test that inference structures support reasoning tokens"""

    def test_chat_completion_usage_with_reasoning_tokens(self):
        """Test ChatCompletionUsage supports reasoning_tokens"""
        from optillm.inference import ChatCompletionUsage
        usage = ChatCompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, reasoning_tokens=5)
        self.assertEqual(usage.reasoning_tokens, 5)
        usage_default = ChatCompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        self.assertEqual(usage_default.reasoning_tokens, 0)

    def test_chat_completion_model_dump_structure(self):
        """Test ChatCompletion model_dump includes reasoning_tokens"""
        from optillm.inference import ChatCompletion
        response_dict = {'id': 'test-123', 'object': 'chat.completion', 'created': 1234567890, 'model': 'test-model', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'test response'}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 15, 'total_tokens': 25, 'reasoning_tokens': 3}}
        completion = ChatCompletion(response_dict)
        result = completion.model_dump()
        self.assertIn('usage', result)
        self.assertIn('completion_tokens_details', result['usage'])
        self.assertIn('reasoning_tokens', result['usage']['completion_tokens_details'])
        self.assertEqual(result['usage']['completion_tokens_details']['reasoning_tokens'], 3)

class TestInferenceIntegration(unittest.TestCase):
    """Test integration with inference.py module"""

    def test_inference_usage_includes_reasoning_tokens(self):
        """Test that ChatCompletionUsage includes reasoning_tokens"""
        from optillm.inference import ChatCompletionUsage
        usage = ChatCompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, reasoning_tokens=5)
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 20)
        self.assertEqual(usage.total_tokens, 30)
        self.assertEqual(usage.reasoning_tokens, 5)

    def test_inference_usage_default_reasoning_tokens(self):
        """Test that ChatCompletionUsage defaults reasoning_tokens to 0"""
        from optillm.inference import ChatCompletionUsage
        usage = ChatCompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        self.assertEqual(usage.reasoning_tokens, 0)

    def test_chat_completion_model_dump_includes_reasoning_tokens(self):
        """Test that ChatCompletion.model_dump includes reasoning_tokens in usage"""
        from optillm.inference import ChatCompletion
        response_dict = {'id': 'test-id', 'object': 'chat.completion', 'created': 1234567890, 'model': 'test-model', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '<think>reasoning</think>answer'}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30, 'reasoning_tokens': 5}}
        completion = ChatCompletion(response_dict)
        result = completion.model_dump()
        self.assertIn('usage', result)
        self.assertIn('completion_tokens_details', result['usage'])
        self.assertIn('reasoning_tokens', result['usage']['completion_tokens_details'])
        self.assertEqual(result['usage']['completion_tokens_details']['reasoning_tokens'], 5)

class TestAPIResponseStructure(unittest.TestCase):
    """Test API response structure with reasoning tokens using mocks"""

    def test_chat_completion_response_structure(self):
        """Test that chat completion responses have proper structure"""
        from unittest.mock import Mock
        from optillm.inference import ChatCompletion, ChatCompletionUsage
        mock_usage = ChatCompletionUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40, reasoning_tokens=8)
        self.assertEqual(mock_usage.prompt_tokens, 15)
        self.assertEqual(mock_usage.completion_tokens, 25)
        self.assertEqual(mock_usage.total_tokens, 40)
        self.assertEqual(mock_usage.reasoning_tokens, 8)
        response_data = {'id': 'test-completion', 'object': 'chat.completion', 'created': 1234567890, 'model': TEST_MODEL, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '<think>Let me calculate: 2+2=4</think>The answer is 4.'}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 15, 'completion_tokens': 25, 'total_tokens': 40, 'reasoning_tokens': 8}}
        completion = ChatCompletion(response_data)
        result = completion.model_dump()
        self.assertIn('usage', result)
        self.assertIn('completion_tokens_details', result['usage'])
        self.assertIn('reasoning_tokens', result['usage']['completion_tokens_details'])
        self.assertEqual(result['usage']['completion_tokens_details']['reasoning_tokens'], 8)

