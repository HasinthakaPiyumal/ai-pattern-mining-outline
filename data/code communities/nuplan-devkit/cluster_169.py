# Cluster 169

class TestLogHandlerConfig(unittest.TestCase):
    """Test LogHandlerConfig class."""

    @patch('os.makedirs')
    def test_init(self, mock_makedirs) -> None:
        """Test class initialization."""
        unique_path = str(uuid4())
        unique_dir = os.path.join(unique_path, '')
        handler = LogHandlerConfig('LEVEL', unique_dir, 'REGEX')
        self.assertEqual(handler.level, 'LEVEL')
        self.assertEqual(handler.path, unique_dir)
        self.assertEqual(handler.filter_regexp, 'REGEX')
        mock_makedirs.assert_called_once_with(unique_path)

