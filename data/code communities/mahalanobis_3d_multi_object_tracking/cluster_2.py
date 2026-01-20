# Cluster 2

def is_path_creatable(pathname):
    """
	if any previous level of parent folder exists, returns true
	"""
    if not is_path_valid(pathname):
        return False
    pathname = os.path.normpath(pathname)
    pathname = os.path.dirname(os.path.abspath(pathname))
    while not is_path_exists(pathname):
        pathname_new = os.path.dirname(os.path.abspath(pathname))
        if pathname_new == pathname:
            return False
        pathname = pathname_new
    return os.access(pathname, os.W_OK)

def is_path_exists(pathname):
    try:
        return is_path_valid(pathname) and os.path.exists(pathname)
    except OSError:
        return False

def is_path_exists_or_creatable(pathname):
    try:
        return is_path_exists(pathname) or is_path_creatable(pathname)
    except OSError:
        return False

def load_txt_file(file_path, debug=True):
    """
    load data or string from text file
    """
    file_path = safe_path(file_path)
    if debug:
        assert is_path_exists(file_path), 'text file is not existing at path: %s!' % file_path
    with open(file_path, 'r') as file:
        data = file.read().splitlines()
    num_lines = len(data)
    file.close()
    return (data, num_lines)

