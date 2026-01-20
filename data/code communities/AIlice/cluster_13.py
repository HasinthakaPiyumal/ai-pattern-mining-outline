# Cluster 13

def get_process_logger(process_name=None):
    if process_name is None:
        process_name = f'{multiprocessing.current_process().name}_{os.getpid()}'
    log_format = '%(asctime)s.%(msecs)03d|%(levelname)s|%(thread)d|%(threadName)s|%(name)s|%(filename)s:%(lineno)d|%(funcName)s|%(message)s'
    logger_name = f'ailice.{process_name}'
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    LOG_DIR = os.path.join(appdirs.user_log_dir('ailice', 'Steven Lu'), 'logs')
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, f'ailice_{process_name}.log')
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger

