# Cluster 97

def test_no_required_secrets(patched_configure_app, mock_mcp_client):
    """Test when app has no required secrets."""
    result = patched_configure_app(app_server_url=MOCK_APP_SERVER_URL, secrets_file=None, secrets_output_file=None, dry_run=False, params=False, api_url='http://test-api', api_key='test-token', verbose=False)
    assert result == MOCK_APP_CONFIG_ID
    mock_mcp_client.list_config_params.assert_called_once_with(app_server_url=MOCK_APP_SERVER_URL)
    mock_mcp_client.configure_app.assert_called_once_with(app_server_url=MOCK_APP_SERVER_URL, config_params={})

@pytest.fixture
def patched_configure_app(mock_mcp_client):
    """Patch the configure_app function for testing."""
    original_func = configure_app

    def wrapped_configure_app(**kwargs):
        defaults = {'api_url': kwargs.get('api_url', 'http://test-api'), 'api_key': kwargs.get('api_key', 'test-token'), 'verbose': kwargs.get('verbose', False)}
        kwargs.update(defaults)
        mock_ctx = MagicMock()
        with patch('mcp_agent.cli.cloud.commands.configure.main.MCPAppClient', return_value=mock_mcp_client), patch('mcp_agent.cli.cloud.commands.configure.main.MockMCPAppClient', return_value=mock_mcp_client), patch('mcp_agent.cli.cloud.commands.configure.main.typer.Exit', side_effect=ValueError), patch('mcp_agent.cli.cloud.commands.configure.main.typer.confirm', return_value=True):
            try:
                return original_func(mock_ctx, **kwargs)
            except ValueError as e:
                raise RuntimeError(f'Typer exit with code: {e}')
    return wrapped_configure_app

def test_with_required_secrets_from_file(patched_configure_app, mock_mcp_client, tmp_path):
    """Test with required secrets from a file."""
    required_secrets = ['server.bedrock.api_key', 'server.openai.api_key']
    secret_values = {'server.bedrock.api_key': 'mcpac_sc_12345678-1234-1234-1234-123456789012', 'server.openai.api_key': 'mcpac_sc_87654321-4321-4321-4321-210987654321'}
    mock_mcp_client.list_config_params = AsyncMock(return_value=required_secrets)
    secrets_file = tmp_path / 'test_secrets.yaml'
    secrets_file.touch()
    with patch('mcp_agent.cli.secrets.processor.retrieve_secrets_from_config', return_value=secret_values) as mock_retrieve:
        result = patched_configure_app(app_server_url=MOCK_APP_SERVER_URL, secrets_file=secrets_file, secrets_output_file=None, dry_run=False, params=False, api_url='http://test-api', api_key='test-token')
        assert result == MOCK_APP_CONFIG_ID
        mock_mcp_client.list_config_params.assert_called_once_with(app_server_url=MOCK_APP_SERVER_URL)
        mock_retrieve.assert_called_once_with(str(secrets_file), required_secrets)
        mock_mcp_client.configure_app.assert_called_once_with(app_server_url=MOCK_APP_SERVER_URL, config_params=secret_values)

def test_missing_app_id(patched_configure_app):
    """Test with missing app_id."""
    with pytest.raises(CLIError):
        patched_configure_app(app_server_url='', secrets_file=None, secrets_output_file=None, dry_run=False, params=False)
    with pytest.raises(CLIError):
        patched_configure_app(app_server_url=None, secrets_file=None, secrets_output_file=None, dry_run=False, params=False)

def test_invalid_file_types(patched_configure_app, tmp_path):
    """Test with invalid file types."""
    invalid_secrets_file = tmp_path / 'invalid_secrets.txt'
    invalid_secrets_file.touch()
    with pytest.raises(CLIError):
        patched_configure_app(app_server_url=MOCK_APP_SERVER_URL, secrets_file=invalid_secrets_file, secrets_output_file=None, dry_run=False, params=False)
    invalid_output_file = tmp_path / 'invalid_output.txt'
    with pytest.raises(CLIError):
        patched_configure_app(app_server_url=MOCK_APP_SERVER_URL, secrets_file=None, secrets_output_file=invalid_output_file, dry_run=False, params=False)

def test_both_input_output_files(patched_configure_app, tmp_path):
    """Test with both secrets_file and secrets_output_file provided."""
    secrets_file = tmp_path / 'secrets.yaml'
    secrets_file.touch()
    secrets_output_file = tmp_path / 'output.yaml'
    with pytest.raises(CLIError):
        patched_configure_app(app_server_url=MOCK_APP_SERVER_URL, secrets_file=secrets_file, secrets_output_file=secrets_output_file, dry_run=False, params=False)

def test_missing_api_key(patched_configure_app):
    """Test with missing API key."""
    with patch('mcp_agent.cli.cloud.commands.configure.main.settings') as mock_settings:
        mock_settings.API_KEY = None
        with patch('mcp_agent.cli.cloud.commands.configure.main.load_api_key_credentials', return_value=None):
            with pytest.raises(CLIError):
                patched_configure_app(app_server_url=MOCK_APP_SERVER_URL, secrets_file=None, secrets_output_file=None, dry_run=False, params=False, api_key=None)

def test_list_config_params_error(patched_configure_app, mock_mcp_client):
    """Test when list_config_params raises an error."""
    mock_mcp_client.list_config_params = AsyncMock(side_effect=Exception('API error'))
    with pytest.raises(CLIError):
        patched_configure_app(app_server_url=MOCK_APP_SERVER_URL, secrets_file=None, secrets_output_file=None, dry_run=False, params=False, api_url='http://test-api', api_key='test-token')

def test_no_secrets_with_secrets_file(patched_configure_app, mock_mcp_client, tmp_path):
    """Test when app doesn't require secrets but a secrets file is provided."""
    mock_mcp_client.list_config_params = AsyncMock(return_value=[])
    secrets_file = tmp_path / 'test_secrets.yaml'
    secrets_file.touch()
    with pytest.raises(CLIError):
        patched_configure_app(app_server_url=MOCK_APP_SERVER_URL, secrets_file=secrets_file, secrets_output_file=None, dry_run=False, params=False, api_url='http://test-api', api_key='test-token')

