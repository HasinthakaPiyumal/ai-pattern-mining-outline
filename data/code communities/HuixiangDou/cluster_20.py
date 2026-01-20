# Cluster 20

def log(name):
    """
    @param name: python file name
    @return: Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:     %(asctime)s - %(module)s-%(funcName)s-line:%(lineno)d - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def safe_join(directory: str, path: str) -> str:
    """Safely path to a base directory to avoid escaping the base directory.
    Borrowed from: werkzeug.security.safe_join.

    @param directory:
    @param path:
    """
    _os_alt_seps: List[str] = [sep for sep in [os.path.sep, os.path.altsep] if sep is not None and sep != '/']
    if path == '':
        raise HTTPException(status_code=400, detail='path is empty')
    filename = os.path.normpath(path)
    full_path = os.path.join(directory, filename)
    if any((sep in filename for sep in _os_alt_seps)) or os.path.isabs(filename) or filename == '..' or filename.startswith('../') or os.path.isdir(full_path):
        raise HTTPException(status_code=400, detail='path is illegal')
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail='path is not existed')
    return full_path

def clear_other_log():
    for name, item in logging.Logger.manager.loggerDict.items():
        if not isinstance(item, logging.Logger):
            continue
        if 'aoe' not in name:
            item.setLevel(logging.CRITICAL)

