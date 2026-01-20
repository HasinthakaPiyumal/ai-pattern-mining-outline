# Cluster 99

def test_delete_app(patched_delete_app, mock_mcp_client):
    app = MCPApp(appId=MOCK_APP_ID, name='name', creatorId='creatorId', createdAt=datetime.datetime.now(), updatedAt=datetime.datetime.now())
    mock_mcp_client.get_app_or_config = AsyncMock(return_value=app)
    patched_delete_app(app_id_or_url=MOCK_APP_ID)
    patched_delete_app(app_id_or_url=MOCK_APP_ID, dry_run=False)
    mock_mcp_client.delete_app.assert_called_once_with(MOCK_APP_ID)

@pytest.fixture
def patched_delete_app(mock_mcp_client):
    """Patch the configure_app function for testing."""
    original_func = delete_app

    def wrapped_delete_app(**kwargs):
        with patch('mcp_agent.cli.cloud.commands.app.delete.main.MCPAppClient', return_value=mock_mcp_client), patch('mcp_agent.cli.cloud.commands.app.delete.main.typer.Exit', side_effect=ValueError):
            try:
                return original_func(**kwargs)
            except ValueError as e:
                raise RuntimeError(f'Typer exit with code: {e}')
    return wrapped_delete_app

def test_delete_app_config(patched_delete_app, mock_mcp_client):
    app_config = MCPAppConfiguration(appConfigurationId=MOCK_APP_CONFIG_ID, creatorId='creator')
    mock_mcp_client.get_app_or_config = AsyncMock(return_value=app_config)
    patched_delete_app(app_id_or_url=MOCK_APP_ID)
    patched_delete_app(app_id_or_url=MOCK_APP_ID, dry_run=False)
    mock_mcp_client.delete_app_configuration.assert_called_once_with(MOCK_APP_CONFIG_ID)

def test_missing_app_id(patched_delete_app):
    """Test with missing app_id."""
    with pytest.raises(CLIError):
        patched_delete_app(app_id_or_url='')
    with pytest.raises(CLIError):
        patched_delete_app(app_id_or_url=None)

def test_missing_api_key(patched_delete_app):
    """Test with missing API key."""
    with patch('mcp_agent.cli.cloud.commands.configure.main.settings') as mock_settings:
        mock_settings.API_KEY = None
        with patch('mcp_agent.cli.cloud.commands.configure.main.load_api_key_credentials', return_value=None):
            with pytest.raises(CLIError):
                patched_delete_app(app_id_or_url=MOCK_APP_ID)

def test_invalid_app_id(patched_delete_app):
    with pytest.raises(CLIError):
        patched_delete_app(app_id_or_url='foo')

