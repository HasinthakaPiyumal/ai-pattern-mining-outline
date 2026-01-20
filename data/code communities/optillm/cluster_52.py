# Cluster 52

class TestServerConfig:
    """Test ServerConfig dataclass functionality"""

    def test_default_stdio_config(self):
        """Test default configuration for stdio transport"""
        config = ServerConfig()
        assert config.transport == 'stdio'
        assert config.command is None
        assert config.args == []
        assert config.url is None
        assert config.headers == {}
        assert config.env == {}
        assert config.timeout == 5.0
        assert config.sse_read_timeout == 300.0

    def test_stdio_config_from_dict(self):
        """Test creating stdio config from dictionary"""
        config_dict = {'transport': 'stdio', 'command': 'npx', 'args': ['@modelcontextprotocol/server-filesystem', '/tmp'], 'env': {'PATH': '/usr/local/bin'}, 'description': 'Filesystem server'}
        config = ServerConfig.from_dict(config_dict)
        assert config.transport == 'stdio'
        assert config.command == 'npx'
        assert config.args == ['@modelcontextprotocol/server-filesystem', '/tmp']
        assert config.env == {'PATH': '/usr/local/bin'}
        assert config.description == 'Filesystem server'

    def test_sse_config_from_dict(self):
        """Test creating SSE config from dictionary"""
        config_dict = {'transport': 'sse', 'url': 'https://api.example.com/mcp', 'headers': {'Authorization': 'Bearer token123'}, 'timeout': 10.0, 'sse_read_timeout': 600.0, 'description': 'Remote SSE server'}
        config = ServerConfig.from_dict(config_dict)
        assert config.transport == 'sse'
        assert config.url == 'https://api.example.com/mcp'
        assert config.headers == {'Authorization': 'Bearer token123'}
        assert config.timeout == 10.0
        assert config.sse_read_timeout == 600.0
        assert config.description == 'Remote SSE server'

    def test_websocket_config_from_dict(self):
        """Test creating WebSocket config from dictionary"""
        config_dict = {'transport': 'websocket', 'url': 'wss://api.example.com/mcp', 'description': 'WebSocket server'}
        config = ServerConfig.from_dict(config_dict)
        assert config.transport == 'websocket'
        assert config.url == 'wss://api.example.com/mcp'
        assert config.description == 'WebSocket server'

