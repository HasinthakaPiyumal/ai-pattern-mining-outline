# Cluster 53

def create_embedded_resource(resource_path: str, content: str, mime_type: str, is_binary: bool=False) -> EmbeddedResource:
    """Create an embedded resource content object"""
    resource_uri_str = create_resource_uri(resource_path)
    resource_args = {'uri': AnyUrl(url=resource_uri_str), 'mimeType': mime_type}
    if is_binary:
        return EmbeddedResource(type='resource', resource=BlobResourceContents(**resource_args, blob=content))
    else:
        return EmbeddedResource(type='resource', resource=TextResourceContents(**resource_args, text=content))

def create_resource_uri(path: str) -> str:
    """Create a resource URI from a path"""
    return f'resource://mcp-agent/{Path(path).name}'

class TestCreateResourceUri:

    def test_create_resource_uri(self):
        result = create_resource_uri('test/path/file.txt')
        assert result == 'resource://mcp-agent/file.txt'

    def test_create_resource_uri_simple_filename(self):
        result = create_resource_uri('file.txt')
        assert result == 'resource://mcp-agent/file.txt'

class TestCreateEmbeddedResource:

    def test_create_embedded_resource_text(self):
        result = create_embedded_resource('test.txt', 'Hello, world!', 'text/plain', False)
        assert isinstance(result, EmbeddedResource)
        assert result.type == 'resource'
        assert isinstance(result.resource, TextResourceContents)
        assert result.resource.uri == AnyUrl(url='resource://mcp-agent/test.txt')
        assert result.resource.mimeType == 'text/plain'
        assert result.resource.text == 'Hello, world!'

    def test_create_embedded_resource_binary(self):
        binary_content = base64.b64encode(b'binary data').decode('utf-8')
        result = create_embedded_resource('image.png', binary_content, 'image/png', True)
        assert isinstance(result, EmbeddedResource)
        assert result.type == 'resource'
        assert isinstance(result.resource, BlobResourceContents)
        assert result.resource.uri == AnyUrl(url='resource://mcp-agent/image.png')
        assert result.resource.mimeType == 'image/png'
        assert result.resource.blob == binary_content

