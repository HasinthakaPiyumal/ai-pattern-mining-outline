# Cluster 27

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

def get_backends_with_capability(capability: str) -> List[str]:
    """Get all backends that support a given capability.

    Args:
        capability: The capability to search for (e.g., "web_search")

    Returns:
        List of backend types that support the capability
    """
    return [backend_type for backend_type, caps in BACKEND_CAPABILITIES.items() if capability in caps.supported_capabilities]

class TestConsistency:
    """Test consistency between related fields."""

    def test_filesystem_native_implies_capability(self):
        """Backends with native filesystem should have filesystem capability."""
        for backend_type, caps in BACKEND_CAPABILITIES.items():
            if caps.filesystem_support == 'native':
                assert 'filesystem_native' in caps.supported_capabilities or len(caps.builtin_tools) > 0, f'{backend_type}: native filesystem but no capability/tools'

    def test_mcp_capability_consistency(self):
        """All backends should support MCP except where explicitly excluded."""
        mcp_backends = get_backends_with_capability('mcp')
        assert len(mcp_backends) > 0
        assert 'openai' in mcp_backends
        assert 'claude' in mcp_backends
        assert 'gemini' in mcp_backends

