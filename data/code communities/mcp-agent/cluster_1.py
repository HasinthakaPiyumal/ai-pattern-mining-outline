# Cluster 1

def test_status_app(patched_workflows_app, mock_mcp_client):
    server_url = 'https://test-server.example.com'
    app_server_info = AppServerInfo(serverUrl=server_url, status='APP_SERVER_STATUS_ONLINE')
    app = MCPApp(appId=MOCK_APP_ID, name='name', creatorId='creatorId', createdAt=datetime.datetime.now(), updatedAt=datetime.datetime.now(), appServerInfo=app_server_info)
    mock_mcp_client.get_app_or_config = AsyncMock(return_value=app)
    mock_mcp_print_mcp_server_workflow_details = Mock()
    with patch('mcp_agent.cli.cloud.commands.app.workflows.main.print_mcp_server_workflow_details', side_effect=mock_mcp_print_mcp_server_workflow_details) as mocked_function:
        mock_mcp_print_mcp_server_workflow_details.return_value = None
        patched_workflows_app(app_id_or_url=MOCK_APP_ID, api_url=DEFAULT_API_BASE_URL, api_key=settings.API_KEY)
        mocked_function.assert_called_once_with(server_url=server_url, api_key=settings.API_KEY)

@pytest.fixture
def patched_workflows_app(mock_mcp_client):
    """Patch the configure_app function for testing."""
    original_func = list_app_workflows

    def wrapped_workflows_app(**kwargs):
        with patch('mcp_agent.cli.cloud.commands.app.workflows.main.MCPAppClient', return_value=mock_mcp_client), patch('mcp_agent.cli.cloud.commands.app.workflows.main.typer.Exit', side_effect=ValueError):
            try:
                return original_func(**kwargs)
            except ValueError as e:
                raise RuntimeError(f'Typer exit with code: {e}')
    return wrapped_workflows_app

def test_status_app_config(patched_workflows_app, mock_mcp_client):
    server_url = 'https://test-server.example.com'
    app_server_info = AppServerInfo(serverUrl=server_url, status='APP_SERVER_STATUS_ONLINE')
    app_config = MCPAppConfiguration(appConfigurationId=MOCK_APP_CONFIG_ID, creatorId='creator', appServerInfo=app_server_info)
    mock_mcp_client.get_app_or_config = AsyncMock(return_value=app_config)
    mock_mcp_print_mcp_server_workflow_details = Mock()
    with patch('mcp_agent.cli.cloud.commands.app.workflows.main.print_mcp_server_workflow_details', side_effect=mock_mcp_print_mcp_server_workflow_details) as mocked_function:
        mock_mcp_print_mcp_server_workflow_details.return_value = None
        patched_workflows_app(app_id_or_url=MOCK_APP_ID, api_url=DEFAULT_API_BASE_URL, api_key=settings.API_KEY)
        mocked_function.assert_called_once_with(server_url=server_url, api_key=settings.API_KEY)

def test_missing_app_id(patched_workflows_app):
    """Test with missing app_id."""
    with pytest.raises(CLIError):
        patched_workflows_app(app_id_or_url='')
    with pytest.raises(CLIError):
        patched_workflows_app(app_id_or_url=None)

def test_missing_api_key(patched_workflows_app):
    """Test with missing API key."""
    with patch('mcp_agent.cli.cloud.commands.configure.main.settings') as mock_settings:
        mock_settings.API_KEY = None
        with patch('mcp_agent.cli.cloud.commands.configure.main.load_api_key_credentials', return_value=None):
            with pytest.raises(CLIError):
                patched_workflows_app(app_id_or_url=MOCK_APP_ID, api_url=DEFAULT_API_BASE_URL)

def test_invalid_app_id(patched_workflows_app):
    with pytest.raises(CLIError):
        patched_workflows_app(app_id_or_url='foo', api_url=DEFAULT_API_BASE_URL)

