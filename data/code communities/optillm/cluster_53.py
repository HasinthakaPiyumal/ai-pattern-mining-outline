# Cluster 53

class TestMCPConfigManager:
    """Test MCP configuration management"""

    def test_init_default_path(self):
        """Test default configuration path"""
        manager = MCPConfigManager()
        expected_path = Path.home() / '.optillm' / 'mcp_config.json'
        assert manager.config_path == expected_path

    def test_init_custom_path(self):
        """Test custom configuration path"""
        custom_path = '/tmp/custom_mcp_config.json'
        manager = MCPConfigManager(custom_path)
        assert manager.config_path == Path(custom_path)

    def test_create_default_config(self):
        """Test creating default configuration file"""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.json'
            manager = MCPConfigManager(str(config_path))
            success = manager.create_default_config()
            assert success
            assert config_path.exists()
            with open(config_path) as f:
                config = json.load(f)
            assert 'mcpServers' in config
            assert 'log_level' in config
            assert config['mcpServers'] == {}
            assert config['log_level'] == 'INFO'

    def test_load_valid_config(self):
        """Test loading valid configuration"""
        import tempfile
        config_data = {'mcpServers': {'test_server': {'transport': 'stdio', 'command': 'test-command', 'args': ['arg1', 'arg2']}}, 'log_level': 'DEBUG'}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        try:
            manager = MCPConfigManager(config_path)
            success = manager.load_config()
            assert success
            assert len(manager.servers) == 1
            assert 'test_server' in manager.servers
            assert manager.servers['test_server'].command == 'test-command'
            assert manager.log_level == 'DEBUG'
        finally:
            os.unlink(config_path)

    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration"""
        manager = MCPConfigManager('/nonexistent/path.json')
        success = manager.load_config()
        assert not success
        assert len(manager.servers) == 0

class TestMCPServerManager:
    """Test MCP server manager functionality"""

    def test_init(self):
        """Test MCPServerManager initialization"""
        config_manager = MCPConfigManager()
        manager = MCPServerManager(config_manager)
        assert manager.config_manager == config_manager
        assert manager.servers == {}
        assert not manager.initialized
        assert manager.all_tools == []
        assert manager.all_resources == []
        assert manager.all_prompts == []

    def test_get_tools_for_model_empty(self):
        """Test getting tools when no tools are available"""
        config_manager = MCPConfigManager()
        manager = MCPServerManager(config_manager)
        tools = manager.get_tools_for_model()
        assert tools == []

    def test_get_capabilities_description_no_servers(self):
        """Test getting capabilities description with no servers"""
        config_manager = MCPConfigManager()
        manager = MCPServerManager(config_manager)
        description = manager.get_capabilities_description()
        assert 'No MCP servers available' in description

