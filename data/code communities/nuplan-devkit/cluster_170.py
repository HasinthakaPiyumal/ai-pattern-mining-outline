# Cluster 170

class TestPathKeywordMatch(unittest.TestCase):
    """Test PathKeywordMatch class."""
    log_record = logging.LogRecord('', logging.NOTSET, '/my/filtered/path', 0, msg='', args=None, exc_info=None)

    def test_default_filter(self) -> None:
        """Test filtering by default pattern, which means no filter."""
        pkm = PathKeywordMatch()
        self.assertTrue(pkm.filter(self.log_record))

    def test_filter(self) -> None:
        """Test filtering by a custom pattern."""
        pkm = PathKeywordMatch(regexp='filtered')
        self.assertFalse(pkm.filter(self.log_record))

