# Cluster 4

def safe_path(input_path, warning=True, debug=True):
    """
    convert path to a valid OS format, e.g., empty string '' to '.', remove redundant '/' at the end from 'aa/' to 'aa'

    parameters:
    	input_path:		a string

    outputs:
    	safe_data:		a valid path in OS format
    """
    if debug:
        assert isstring(input_path), 'path is not a string: %s' % input_path
    safe_data = copy.copy(input_path)
    safe_data = os.path.normpath(safe_data)
    return safe_data

def fileparts(input_path, warning=True, debug=True):
    """
	this function return a tuple, which contains (directory, filename, extension)
	if the file has multiple extension, only last one will be displayed

	parameters:
		input_path:     a string path

	outputs:
		directory:      the parent directory
		filename:       the file name without extension
		ext:            the extension
	"""
    good_path = safe_path(input_path, debug=debug)
    if len(good_path) == 0:
        return ('', '', '')
    if good_path[-1] == '/':
        if len(good_path) > 1:
            return (good_path[:-1], '', '')
        else:
            return (good_path, '', '')
    directory = os.path.dirname(os.path.abspath(good_path))
    filename = os.path.splitext(os.path.basename(good_path))[0]
    ext = os.path.splitext(good_path)[1]
    return (directory, filename, ext)

def mkdir_if_missing(input_path, warning=True, debug=True):
    """
	create a directory if not existing:
		1. if the input is a path of file, then create the parent directory of this file
		2. if the root directory does not exists for the input, then create all the root directories recursively until the parent directory of input exists

	parameters:
		input_path:     a string path
	"""
    good_path = safe_path(input_path, warning=warning, debug=debug)
    if debug:
        assert is_path_exists_or_creatable(good_path), 'input path is not valid or creatable: %s' % good_path
    dirname, _, _ = fileparts(good_path)
    if not is_path_exists(dirname):
        mkdir_if_missing(dirname)
    if isfolder(good_path) and (not is_path_exists(good_path)):
        os.mkdir(good_path)

