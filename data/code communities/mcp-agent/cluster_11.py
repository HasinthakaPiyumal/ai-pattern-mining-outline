# Cluster 11

class TestCreateImageContent:

    def test_create_image_content(self):
        data = 'base64imagedata'
        mime_type = 'image/png'
        result = create_image_content(data, mime_type)
        assert result.type == 'image'
        assert result.data == data
        assert result.mimeType == mime_type

def create_image_content(data: str, mime_type: str) -> ImageContent:
    """Create an image content object from base64-encoded data"""
    return ImageContent(type='image', data=data, mimeType=mime_type)

