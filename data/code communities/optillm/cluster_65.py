# Cluster 65

class TestCountReasoningTokens(unittest.TestCase):
    """Test the count_reasoning_tokens function"""

    def test_count_reasoning_tokens_basic(self):
        """Test basic functionality of count_reasoning_tokens"""
        text_with_think = '<think>This is reasoning content</think>This is output'
        result1 = optillm_count_reasoning_tokens(text_with_think)
        result2 = inference_count_reasoning_tokens(text_with_think)
        self.assertGreater(result1, 0)
        self.assertEqual(result1, result2)

    def test_count_reasoning_tokens_no_think_tags(self):
        """Test with text that has no think tags"""
        text_without_think = 'This is just regular output text'
        result1 = optillm_count_reasoning_tokens(text_without_think)
        result2 = inference_count_reasoning_tokens(text_without_think)
        self.assertEqual(result1, 0)
        self.assertEqual(result2, 0)

    def test_count_reasoning_tokens_multiple_think_blocks(self):
        """Test with multiple think tag blocks"""
        text_multiple = '\n        <think>First reasoning block</think>\n        Some output here\n        <think>Second reasoning block with more content</think>\n        Final output\n        '
        result = optillm_count_reasoning_tokens(text_multiple)
        self.assertGreater(result, 0)
        single_block = '<think>First reasoning blockSecond reasoning block with more content</think>'
        single_result = optillm_count_reasoning_tokens(single_block)
        self.assertAlmostEqual(result, single_result, delta=2)

    def test_count_reasoning_tokens_empty_input(self):
        """Test with empty or None input"""
        self.assertEqual(optillm_count_reasoning_tokens(''), 0)
        self.assertEqual(optillm_count_reasoning_tokens(None), 0)
        self.assertEqual(optillm_count_reasoning_tokens(123), 0)

    def test_count_reasoning_tokens_malformed_tags(self):
        """Test with malformed think tags"""
        malformed_cases = ['<think>Unclosed think tag', 'Unopened think tag</think>', '<think><think>Nested tags</think></think>', '<THINK>Wrong case</THINK>', '<think></think>']
        for case in malformed_cases:
            result = optillm_count_reasoning_tokens(case)
            self.assertGreaterEqual(result, 0)

    def test_count_reasoning_tokens_with_tokenizer(self):
        """Test with a mock tokenizer for precise counting"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        text = '<think>Some reasoning text</think>Output'
        result = optillm_count_reasoning_tokens(text, mock_tokenizer)
        self.assertEqual(result, 5)
        mock_tokenizer.encode.assert_called_once_with('Some reasoning text')

    def test_count_reasoning_tokens_tokenizer_error(self):
        """Test fallback when tokenizer fails"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = Exception('Tokenizer error')
        text = '<think>Some reasoning text</think>Output'
        result = optillm_count_reasoning_tokens(text, mock_tokenizer)
        self.assertGreater(result, 0)
        mock_tokenizer.encode.assert_called_once()

    def test_count_reasoning_tokens_multiline(self):
        """Test with multiline think blocks"""
        multiline_text = '<think>\n        This is a multi-line reasoning block\n        with several lines of content\n        that spans multiple lines\n        </think>\n        This is the final output'
        result = optillm_count_reasoning_tokens(multiline_text)
        self.assertGreater(result, 10)

    def test_count_reasoning_tokens_special_characters(self):
        """Test with special characters in think blocks"""
        special_text = '<think>Content with émojis 🤔 and symbols @#$%^&*()</think>Output'
        result = optillm_count_reasoning_tokens(special_text)
        self.assertGreater(result, 0)

