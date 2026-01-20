# Cluster 26

class TestCapabilityQueries:
    """Test capability query functions."""

    def test_get_capabilities_existing_backend(self):
        """Test getting capabilities for existing backends."""
        caps = get_capabilities('openai')
        assert caps is not None
        assert caps.backend_type == 'openai'
        assert caps.provider_name == 'OpenAI'

    def test_get_capabilities_nonexistent_backend(self):
        """Test getting capabilities for non-existent backend."""
        caps = get_capabilities('nonexistent_backend')
        assert caps is None

    def test_has_capability_true(self):
        """Test checking for existing capability."""
        assert has_capability('openai', 'web_search') is True

    def test_has_capability_false(self):
        """Test checking for non-existent capability."""
        assert has_capability('lmstudio', 'web_search') is False

    def test_has_capability_nonexistent_backend(self):
        """Test checking capability on non-existent backend."""
        assert has_capability('nonexistent', 'web_search') is False

    def test_get_all_backend_types(self):
        """Test getting all backend types."""
        backend_types = get_all_backend_types()
        assert len(backend_types) > 0
        assert 'openai' in backend_types
        assert 'claude' in backend_types
        assert 'gemini' in backend_types

    def test_get_backends_with_capability(self):
        """Test getting backends by capability."""
        web_search_backends = get_backends_with_capability('web_search')
        assert 'openai' in web_search_backends
        assert 'gemini' in web_search_backends
        assert 'grok' in web_search_backends
        assert 'claude_code' not in web_search_backends

def get_all_backend_types() -> List[str]:
    """Get list of all registered backend types.

    Returns:
        List of backend type strings
    """
    return list(BACKEND_CAPABILITIES.keys())

