# Cluster 92

class TestCreateBlobResource:

    def test_create_blob_resource(self):
        content = base64.b64encode(b'binary data').decode('utf-8')
        result = create_blob_resource('file://test.bin', content, 'application/octet-stream')
        assert isinstance(result, EmbeddedResource)
        assert result.type == 'resource'
        assert isinstance(result.resource, BlobResourceContents)
        assert result.resource.uri == AnyUrl(url='file://test.bin')
        assert result.resource.mimeType == 'application/octet-stream'
        assert result.resource.blob == content

def create_blob_resource(resource_path: str, content: str, mime_type: str) -> EmbeddedResource:
    """Create an embedded resource for binary data"""
    return EmbeddedResource(type='resource', resource=BlobResourceContents(uri=AnyUrl(url=resource_path), mimeType=mime_type, blob=content))

