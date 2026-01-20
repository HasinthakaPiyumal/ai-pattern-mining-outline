# Cluster 10

def main() -> None:
    """
    Main entry point for the CLI
    """
    cli()

def get_submission_logger(logger_name: str, logfile: str='/tmp/submission.log') -> logging.Logger:
    """
    Returns a logger with level WARNING that logs to the given file.
    :param logger_name: Name for the logger.
    :param logfile: Output file for the logger.
    :return: The logger.
    """
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

def set_default_path() -> None:
    """
    This function sets the default paths as environment variables if none are set.
    These can then be used by Hydra, unless the user overwrites them from the command line.
    """
    if 'NUPLAN_DATA_ROOT' not in os.environ:
        logger.info(f'Setting default NUPLAN_DATA_ROOT: {DEFAULT_DATA_ROOT}')
        os.environ['NUPLAN_DATA_ROOT'] = DEFAULT_DATA_ROOT
    if 'NUPLAN_EXP_ROOT' not in os.environ:
        logger.info(f'Setting default NUPLAN_EXP_ROOT: {DEFAULT_EXP_ROOT}')
        os.environ['NUPLAN_EXP_ROOT'] = DEFAULT_EXP_ROOT

class TestUtils(unittest.TestCase):
    """Tests for utils function."""

    def test_submission_logger(self) -> None:
        """Tests the two handlers of the submission logger."""
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        logfile = '/'.join([tmp_dir.name, 'bar.log'])
        logger = get_submission_logger('foo', logfile)
        logger.info('DONT MIND ME')
        logger.warning('HELLO')
        logger.error('WORLD!')
        with open(logfile, 'r') as f:
            self.assertEqual(len(f.readlines()), 2)

