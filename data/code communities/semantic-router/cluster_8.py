# Cluster 8

def setup_custom_logger(name):
    """Setup a custom logger."""
    logger = logging.getLogger(name)
    log_level = os.getenv('SEMANTIC_ROUTER_LOG_LEVEL') or os.getenv('LOG_LEVEL') or 'INFO'
    log_level = log_level.upper()
    add_coloured_handler(logger)
    logger.setLevel(log_level)
    logger.propagate = False
    return logger

def add_coloured_handler(logger):
    """Add a coloured handler to the logger."""
    formatter = CustomFormatter()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

