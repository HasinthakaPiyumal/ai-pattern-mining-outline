# Cluster 51

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

@pytest.mark.asyncio
class TestMCPServer:
    """Test MCP server connection and capability discovery"""

    def test_init(self):
        """Test MCPServer initialization"""
        config = ServerConfig()
        server = MCPServer('test_server', config)
        assert server.server_name == 'test_server'
        assert server.config == config
        assert server.tools == []
        assert server.resources == []
        assert server.prompts == []
        assert not server.connected
        assert not server.has_tools_capability
        assert not server.has_resources_capability
        assert not server.has_prompts_capability

    async def test_connect_stdio_validation(self):
        """Test stdio connection validation"""
        config = ServerConfig(transport='stdio')
        server = MCPServer('test_server', config)
        result = await server.connect_stdio_native()
        assert not result

    async def test_connect_sse_validation(self):
        """Test SSE connection validation"""
        config = ServerConfig(transport='sse')
        server = MCPServer('test_server', config)
        result = await server.connect_sse()
        assert not result

    async def test_connect_websocket_validation(self):
        """Test WebSocket connection validation"""
        config = ServerConfig(transport='websocket')
        server = MCPServer('test_server', config)
        result = await server.connect_websocket()
        assert not result

    async def test_connect_and_discover_unsupported_transport(self):
        """Test unsupported transport type"""
        config = ServerConfig(transport='invalid')
        server = MCPServer('test_server', config)
        result = await server.connect_and_discover()
        assert not result

    @patch('optillm.plugins.mcp_plugin.sse_client')
    async def test_connect_sse_success(self, mock_sse_client):
        """Test successful SSE connection"""
        mock_streams = (AsyncMock(), AsyncMock())
        mock_sse_client.return_value.__aenter__ = AsyncMock(return_value=mock_streams)
        mock_sse_client.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.capabilities = Mock()
        mock_session.initialize.return_value = mock_result
        config = ServerConfig(transport='sse', url='https://api.example.com/mcp', headers={'Authorization': 'Bearer token'})
        server = MCPServer('test_server', config)
        with patch.object(server, 'connect_stdio', return_value=True):
            with patch('optillm.plugins.mcp_plugin.LoggingClientSession') as mock_session_class:
                mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
                result = await server.connect_sse()
                assert result

class TestMockScenarios:
    """Test various scenarios with mocked dependencies"""

    @patch('optillm.plugins.mcp_plugin.find_executable')
    def test_stdio_command_not_found(self, mock_find_executable):
        """Test stdio transport when command is not found"""
        mock_find_executable.return_value = None
        config = ServerConfig(transport='stdio', command='nonexistent-command')

        async def test_async():
            result = await execute_tool_stdio(config, 'test_tool', {})
            assert 'error' in result
            assert 'Failed to find executable' in result['error']
        asyncio.run(test_async())

    def test_environment_variable_expansion(self):
        """Test environment variable expansion in SSE headers"""
        os.environ['TEST_TOKEN'] = 'test-token-value'
        try:
            config = ServerConfig(transport='sse', url='https://api.example.com/mcp', headers={'Authorization': 'Bearer ${TEST_TOKEN}'})
            server = MCPServer('test', config)
            expanded_headers = {}
            for key, value in config.headers.items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    expanded_value = os.environ.get(env_var)
                    if expanded_value:
                        expanded_headers[key] = expanded_value
                else:
                    expanded_headers[key] = value
            assert expanded_headers['Authorization'] == 'Bearer test-token-value'
        finally:
            del os.environ['TEST_TOKEN']

