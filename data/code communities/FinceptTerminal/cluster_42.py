# Cluster 42

def setup_for_development():
    """Configure for development"""
    set_debug_mode(True)
    logger.enable_console(True)
    logger._direct_log(logging.INFO, 'Development logging configured', 'Setup')

def set_debug_mode(enable: bool):
    """Enable/disable debug mode"""
    logger.config.debug_mode = enable
    logger.set_level(logging.DEBUG if enable else logging.INFO)
    if hasattr(logger, 'console_handler'):
        logger.console_handler.setLevel(logging.DEBUG if enable else logging.INFO)

def setup_for_production():
    """Configure for production"""
    set_debug_mode(False)
    logger.enable_console(False)
    logger._direct_log(logging.INFO, 'Production logging configured', 'Setup')

def setup_for_gui():
    """Configure for GUI application"""
    set_debug_mode(False)
    logger.enable_console(False)
    logger._direct_log(logging.INFO, 'GUI logging configured', 'Setup')

