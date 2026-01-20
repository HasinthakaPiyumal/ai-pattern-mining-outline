# Cluster 168

class TestTqdmLoggingHandler(unittest.TestCase):
    """Test TqdmLoggingHandler class."""

    def test_emit_normal(self) -> None:
        """Test emit function when no errors."""
        tlh = TqdmLoggingHandler()
        log_record = logging.LogRecord('', logging.NOTSET, '', 0, 'A normal logging message.', args=None, exc_info=None)
        self.assertIsNone(tlh.emit(log_record))

    def test_emit_keyboard_interrupt(self) -> None:
        """Test emit when KeyboardInterrupt exception is raised."""
        tlh = TqdmLoggingHandler()
        log_record = logging.LogRecord('', logging.NOTSET, '', 0, 'An interrupted logging message.', args=None, exc_info=None)
        tlh.flush = Mock()
        tlh.flush.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            tlh.emit(log_record)

    def test_emit_other_exception(self) -> None:
        """Test emit when an unexpected error is raised."""
        tlh = TqdmLoggingHandler()
        log_record = logging.LogRecord('', logging.NOTSET, '', 0, 'An error-handled logging message.', args=None, exc_info=None)
        tlh.flush = Mock()
        tlh.flush.side_effect = MemoryError
        tlh.handleError = Mock()
        tlh.emit(log_record)
        tlh.handleError.assert_called_once_with(log_record)

