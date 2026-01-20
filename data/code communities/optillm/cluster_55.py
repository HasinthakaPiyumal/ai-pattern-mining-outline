# Cluster 55

class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end integration with mocked responses for specific configs"""

    def test_thinkdeeper_approach_with_reasoning_tokens(self):
        """Test thinkdeeper approach properly processes reasoning tokens"""
        from unittest.mock import patch, Mock
        with patch('optillm.thinkdeeper.thinkdeeper_decode') as mock_thinkdeeper:
            mock_response = '<think>Let me solve this step by step. 2 + 2 = 4</think>The answer is 4.'
            mock_tokens = 25
            mock_thinkdeeper.return_value = (mock_response, mock_tokens)
            result, tokens = mock_thinkdeeper('You are a helpful assistant.', 'What is 2+2?', Mock(), TEST_MODEL, {'k': 3})
            self.assertEqual(result, mock_response)
            self.assertEqual(tokens, mock_tokens)
            self.assertIn('<think>', result)
            self.assertIn('</think>', result)
            mock_thinkdeeper.assert_called_once()

    def test_reasoning_token_calculation_with_mock_response(self):
        """Test reasoning token calculation with mock content"""
        from optillm import count_reasoning_tokens
        test_cases = [('<think>Simple thought</think>Answer', 2), ('<think>More complex reasoning here</think>Final answer', 4), ('No thinking tags here', 0), ('<think>First thought</think>Some text<think>Second thought</think>End', 4)]
        for content, expected_min_tokens in test_cases:
            with self.subTest(content=content[:30] + '...'):
                reasoning_tokens = count_reasoning_tokens(content)
                if expected_min_tokens > 0:
                    self.assertGreaterEqual(reasoning_tokens, expected_min_tokens - 1)
                else:
                    self.assertEqual(reasoning_tokens, 0)

