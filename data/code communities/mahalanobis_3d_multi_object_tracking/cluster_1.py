# Cluster 1

def is_path_valid(pathname):
    try:
        if not isstring(pathname) or not pathname:
            return False
    except TypeError:
        return False
    else:
        return True

def isstring(string_test):
    try:
        return isinstance(string_test, basestring)
    except NameError:
        return isinstance(string_test, str)

def isfolder(pathname):
    """
	if '.' exists in the subfolder, the function still justifies it as a folder. e.g., /mnt/dome/adhoc_0.5x/abc is a folder
	if '.' exists after all slashes, the function will not justify is as a folder. e.g., /mnt/dome/adhoc_0.5x is NOT a folder
	"""
    if is_path_valid(pathname):
        pathname = os.path.normpath(pathname)
        if pathname == './':
            return True
        name = os.path.splitext(os.path.basename(pathname))[0]
        ext = os.path.splitext(pathname)[1]
        return len(name) > 0 and len(ext) == 0
    else:
        return False

