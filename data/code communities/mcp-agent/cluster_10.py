# Cluster 10

class TestCreateResourceReference:

    def test_create_resource_reference(self):
        uri = 'resource://test/file.txt'
        mime_type = 'text/plain'
        result = create_resource_reference(uri, mime_type)
        assert isinstance(result, EmbeddedResource)
        assert result.type == 'resource'
        assert isinstance(result.resource, TextResourceContents)
        assert str(result.resource.uri) == uri
        assert result.resource.mimeType == mime_type
        assert result.resource.text == ''

def create_resource_reference(uri: str, mime_type: str) -> 'EmbeddedResource':
    """
    Create a reference to a resource without embedding its content directly.

    This creates an EmbeddedResource that references another resource URI.
    When the client receives this, it will make a separate request to fetch
    the resource content using the provided URI.

    Args:
        uri: URI for the resource
        mime_type: MIME type of the resource

    Returns:
        An EmbeddedResource object
    """
    resource_contents = TextResourceContents(uri=uri, mimeType=mime_type, text='')
    return EmbeddedResource(type='resource', resource=resource_contents)

