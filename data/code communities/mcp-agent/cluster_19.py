# Cluster 19

class TestCreateTextResource:

    def test_create_text_resource(self):
        content = 'Hello, world!'
        result = create_text_resource('file://test.txt', content, 'text/plain')
        assert isinstance(result, EmbeddedResource)
        assert result.type == 'resource'
        assert isinstance(result.resource, TextResourceContents)
        assert result.resource.uri == AnyUrl(url='file://test.txt')
        assert result.resource.mimeType == 'text/plain'
        assert result.resource.text == content

def create_text_resource(resource_path: str, content: str, mime_type: str) -> EmbeddedResource:
    """Create an embedded resource for text data"""
    return EmbeddedResource(type='resource', resource=TextResourceContents(uri=AnyUrl(url=resource_path), mimeType=mime_type, text=content))

