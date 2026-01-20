# Cluster 47

class TestReasoningTokensCore(unittest.TestCase):
    """Test core reasoning token functionality"""

    def test_count_reasoning_tokens_with_think_tags(self):
        """Test counting tokens in think tags"""
        text = '<think>Let me think about this problem step by step</think>The answer is 42'
        result1 = optillm_count(text)
        result2 = inference_count(text)
        self.assertGreater(result1, 0, 'Should count tokens in think tags')
        self.assertEqual(result1, result2, 'Both functions should return same result')

    def test_count_reasoning_tokens_without_think_tags(self):
        """Test with text that has no think tags"""
        text = 'This is just a regular response without any thinking'
        result1 = optillm_count(text)
        result2 = inference_count(text)
        self.assertEqual(result1, 0, 'Should return 0 for text without think tags')
        self.assertEqual(result2, 0, 'Should return 0 for text without think tags')

    def test_count_reasoning_tokens_multiple_blocks(self):
        """Test with multiple think tag blocks"""
        text = '\n        <think>First block of reasoning</think>\n        Some output here\n        <think>Second block with more reasoning</think>\n        Final answer\n        '
        result = optillm_count(text)
        self.assertGreater(result, 0, 'Should count tokens from multiple blocks')

    def test_count_reasoning_tokens_empty_cases(self):
        """Test edge cases with empty or invalid input"""
        test_cases = ['', None, 123, '<think></think>']
        for case in test_cases:
            result1 = optillm_count(case)
            result2 = inference_count(case)
            self.assertGreaterEqual(result1, 0, f'Should handle {case} gracefully')
            self.assertGreaterEqual(result2, 0, f'Should handle {case} gracefully')

    def test_count_reasoning_tokens_with_mock_tokenizer(self):
        """Test with a simple mock tokenizer"""

        class MockTokenizer:

            def encode(self, text):
                return text.split()
        tokenizer = MockTokenizer()
        text = '<think>hello world test</think>answer'
        result = optillm_count(text, tokenizer)
        self.assertEqual(result, 3, 'Should use tokenizer when provided')

    def test_reasoning_tokens_fallback_estimation(self):
        """Test fallback estimation when tokenizer fails"""

        class FailingTokenizer:

            def encode(self, text):
                raise Exception('Tokenizer failed')
        tokenizer = FailingTokenizer()
        text = '<think>some reasoning content here</think>answer'
        result = optillm_count(text, tokenizer)
        self.assertGreater(result, 0, 'Should fallback to character estimation')

    def test_count_reasoning_tokens_truncated_response(self):
        """Test counting tokens when response is truncated (no closing </think> tag)"""
        truncated_text = '<think>This reasoning was cut off due to max tokens'
        result1 = optillm_count(truncated_text)
        result2 = inference_count(truncated_text)
        self.assertGreater(result1, 0, 'Should count tokens from truncated think block')
        self.assertEqual(result1, result2, 'Both functions should return same result')

    def test_count_reasoning_tokens_mixed_complete_and_truncated(self):
        """Test with both complete and truncated think blocks"""
        mixed_text = '\n        <think>First complete reasoning block</think>\n        Some output here\n        <think>This second block was truncated and never closed\n        '
        result = optillm_count(mixed_text)
        self.assertGreater(result, 0, 'Should count tokens from both complete and truncated blocks')
        first_block_only = '<think>First complete reasoning block</think>'
        first_result = optillm_count(first_block_only)
        self.assertGreater(result, first_result, 'Should include truncated content')

    def test_count_reasoning_tokens_no_false_positives(self):
        """Test that we don't count think-like content that isn't actually truncated"""
        text_with_complete_blocks = '<think>First block</think>Output<think>Second complete block</think>'
        result = optillm_count(text_with_complete_blocks)
        manual_count = optillm_count('<think>First blockSecond complete block</think>')
        self.assertEqual(result, manual_count, 'Should only count complete blocks, not detect false truncation')

    def test_count_reasoning_tokens_edge_cases_truncated(self):
        """Test edge cases with truncated responses"""
        test_cases = [('<think>', 0), ('<think>a', 1), ('Some output <think>reasoning here', None), ('<think>multi\nline\ntruncated', None)]
        for text, expected_min in test_cases:
            result = optillm_count(text)
            if expected_min is not None:
                if expected_min == 0:
                    self.assertEqual(result, expected_min, f'Should return {expected_min} for: {text}')
                else:
                    self.assertGreaterEqual(result, expected_min, f'Should be at least {expected_min} for: {text}')
            else:
                self.assertGreater(result, 0, f'Should count truncated content for: {text}')

