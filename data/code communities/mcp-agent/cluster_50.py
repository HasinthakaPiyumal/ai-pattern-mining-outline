# Cluster 50

def is_binary_content(mime_type: str) -> bool:
    """Check if content should be treated as binary."""
    return not is_text_mime_type(mime_type)

class TestIsBinaryContent:

    def test_is_binary_content_image(self):
        assert is_binary_content('image/png') is True

    def test_is_binary_content_pdf(self):
        assert is_binary_content('application/pdf') is True

    def test_is_binary_content_text(self):
        assert is_binary_content('text/plain') is False

    def test_is_binary_content_json(self):
        assert is_binary_content('application/json') is False

    def test_is_binary_content_xml(self):
        assert is_binary_content('application/xml') is False

