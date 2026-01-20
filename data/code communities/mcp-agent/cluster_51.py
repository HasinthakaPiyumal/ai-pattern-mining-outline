# Cluster 51

class TestNormalizeUri:

    def test_normalize_uri_empty_string(self):
        assert normalize_uri('') == ''

    def test_normalize_uri_already_valid_uri(self):
        uri = 'https://example.com/file.txt'
        assert normalize_uri(uri) == uri

    def test_normalize_uri_file_uri(self):
        uri = 'file:///path/to/file.txt'
        assert normalize_uri(uri) == uri

    def test_normalize_uri_absolute_path(self):
        path = '/path/to/file.txt'
        assert normalize_uri(path) == 'file:///path/to/file.txt'

    def test_normalize_uri_relative_path(self):
        path = 'path/to/file.txt'
        assert normalize_uri(path) == 'file:///path/to/file.txt'

    def test_normalize_uri_windows_path(self):
        path = 'C:\\path\\to\\file.txt'
        assert normalize_uri(path) == 'file:///C:/path/to/file.txt'

    def test_normalize_uri_simple_filename(self):
        filename = 'file.txt'
        assert normalize_uri(filename) == 'file:///file.txt'

def normalize_uri(uri_or_filename: str) -> str:
    """
    Normalize a URI or filename to ensure it's a valid URI.
    Converts simple filenames to file:// URIs if needed.

    Args:
        uri_or_filename: A URI string or simple filename

    Returns:
        A properly formatted URI string
    """
    if not uri_or_filename:
        return ''
    if '://' in uri_or_filename:
        return uri_or_filename
    normalized_path = uri_or_filename.replace('\\', '/')
    if normalized_path.startswith('/'):
        return f'file://{normalized_path}'
    else:
        return f'file:///{normalized_path}'

