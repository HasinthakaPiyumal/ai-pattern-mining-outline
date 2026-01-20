# Cluster 18

@app.callback(invoke_without_command=True)
def configure(server_url: str=typer.Argument(...), client: str=typer.Option(..., '--client', help='cursor|claude|vscode|smithery|mcp.run'), write: bool=typer.Option(False, '--write'), open: bool=typer.Option(False, '--open'), format: str=typer.Option('text', '--format', help='text|json'), name: str | None=typer.Option(None, '--name', help='Optional server name override')) -> None:
    client_lc = client.lower()
    entry = _build_server_entry(server_url, name=name)
    snippet = {'mcp': {'servers': entry}}
    target: Path | None = None
    if client_lc == 'cursor':
        target = Path.home() / '.cursor' / 'mcp.json'
    elif client_lc == 'claude':
        target = Path.home() / '.claude' / 'mcp.json'
    elif client_lc == 'vscode':
        target = Path.cwd() / '.vscode' / 'mcp.json'
    elif client_lc == 'smithery':
        target = Path.cwd() / '.smithery' / 'mcp.json'
    elif client_lc == 'mcp.run':
        console.print('[yellow]mcp.run uses web interface for configuration.[/yellow]')
        console.print('Copy this configuration to your mcp.run dashboard:')
        _print_output(snippet, format)
        return
    else:
        console.print(f"[yellow]Client '{client}' not directly supported.[/yellow]")
        console.print('Use this configuration snippet in your client:')
        _print_output(snippet, format)
        return
    if write:
        try:
            if target.exists():
                existing = json.loads(target.read_text(encoding='utf-8'))
            else:
                existing = {}
        except Exception:
            existing = {}
        merged = _merge_mcp_json(existing, entry)
        try:
            _write_json(target, merged)
            console.print(f'Wrote config to {target}')
        except Exception as e:
            typer.secho(f'Failed to write: {e}', err=True, fg=typer.colors.RED)
            raise typer.Exit(5)
        if open:
            console.print(str(target))
        else:
            _print_output(merged, format)
    else:
        _print_output(snippet, format)

def _build_server_entry(url: str, name: str | None=None) -> dict:
    try:
        _name, transport, fixed_url = parse_server_url(url)
        server_name = name or _name
    except Exception:
        server_name = name or generate_server_name(url)
        fixed_url = url
        transport = 'sse' if url.rstrip('/').endswith('/sse') else 'http'
    entry = {server_name: {'url': fixed_url, 'transport': transport}}
    return entry

def _print_output(data: dict, fmt: str) -> None:
    if fmt.lower() == 'json':
        console.print_json(data=data)
    else:
        try:
            name = next(iter(data['mcp']['servers'].keys()))
        except Exception:
            name = 'server'
        console.print(f"Add this to your client's mcp.json under servers: '{name}'")
        console.print_json(data=data)

def _merge_mcp_json(existing: dict, addition: dict) -> dict:
    servers: dict = {}
    if isinstance(existing, dict):
        if 'mcp' in existing and isinstance(existing.get('mcp'), dict):
            servers = dict(existing['mcp'].get('servers') or {})
        elif 'servers' in existing and isinstance(existing.get('servers'), dict):
            servers = dict(existing.get('servers') or {})
        else:
            for k, v in existing.items():
                if isinstance(v, dict) and ('url' in v or 'transport' in v):
                    servers[k] = v
    servers.update(addition)
    return {'mcp': {'servers': servers}}

def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')

def walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() == 'authorization' and isinstance(v, str):
                obj[k] = 'Bearer ***'
            else:
                walk(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str) and v.lower().startswith('authorization: bearer '):
                obj[i] = 'Authorization: Bearer ***'
            else:
                walk(v)

def _redact_secrets(data: dict) -> dict:
    """Mask Authorization values and mcp-remote header args for safe display."""
    red = deepcopy(data)

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() == 'authorization' and isinstance(v, str):
                    obj[k] = 'Bearer ***'
                else:
                    walk(v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, str) and v.lower().startswith('authorization: bearer '):
                    obj[i] = 'Authorization: Bearer ***'
                else:
                    walk(v)
    walk(red)
    return red

def install(server_identifier: str=typer.Argument(..., help='Server URL to install'), client: str=typer.Option(..., '--client', '-c', help='Client to install to: vscode|claude_code|cursor|claude_desktop|chatgpt'), name: Optional[str]=typer.Option(None, '--name', '-n', help='Server name in client config (auto-generated if not provided)'), dry_run: bool=typer.Option(False, '--dry-run', help='Show what would be installed without writing files'), force: bool=typer.Option(False, '--force', '-f', help='Overwrite existing server configuration'), api_url: Optional[str]=typer.Option(settings.API_BASE_URL, '--api-url', help='API base URL', envvar=ENV_API_BASE_URL), api_key: Optional[str]=typer.Option(settings.API_KEY, '--api-key', help='API key for authentication', envvar=ENV_API_KEY)) -> None:
    """
    Install an MCP server to a client application.

    This command writes the server configuration to the client's config file.
    For authenticated clients (everything except ChatGPT), the server URL is
    added with an Authorization header using your MCP_API_KEY environment variable.

    URLs without /sse or /mcp suffix will automatically have /sse appended and
    use SSE transport for optimal performance.

    For ChatGPT, the server must have unauthenticated access enabled.

    Examples:
        # Install to VSCode (automatically appends /sse)
        mcp-agent install --client=vscode https://xxx.deployments.mcp-agent.com

        # Install to Claude Code with custom name
        mcp-agent install --client=claude_code --name=my-server https://xxx.deployments.mcp-agent.com

        # Install to ChatGPT (requires unauthenticated access)
        mcp-agent install --client=chatgpt https://xxx.deployments.mcp-agent.com
    """
    client_lc = client.lower()
    if client_lc not in CLIENT_CONFIGS and client_lc != 'chatgpt':
        raise CLIError(f'Unsupported client: {client}. Supported clients: vscode, claude_code, cursor, claude_desktop, chatgpt')
    effective_api_key = api_key or settings.API_KEY or load_api_key_credentials()
    if not effective_api_key:
        raise CLIError("Must be logged in to install. Run 'mcp-agent login', set MCP_API_KEY environment variable, or specify --api-key option.")
    server_url = server_identifier
    if not server_identifier.startswith('http://') and (not server_identifier.startswith('https://')):
        raise CLIError(f'Server identifier must be a URL starting with http:// or https://. Got: {server_identifier}')
    if not server_url.endswith('/sse') and (not server_url.endswith('/mcp')):
        server_url = server_url.rstrip('/') + '/sse'
        print_info(f'Using SSE transport: {server_url}')
    console.print('\n[bold cyan]Installing MCP Server[/bold cyan]\n')
    print_info(f'Server URL: {server_url}')
    print_info(f'Client: {CLIENT_CONFIGS.get(client_lc, {}).get('description', client_lc)}')
    mcp_client = MCPAppClient(api_url=api_url or DEFAULT_API_BASE_URL, api_key=effective_api_key)
    try:
        app_info = run_async(mcp_client.get_app(server_url=server_url))
        app_name = app_info.name if app_info else None
        print_info(f'App name: {app_name}')
    except Exception as e:
        print_info(f'Warning: Could not fetch app info: {e}')
        app_name = None
    if client_lc == 'chatgpt':
        try:
            has_unauth_access = app_info.unauthenticatedAccess is True or (app_info.appServerInfo and app_info.appServerInfo.unauthenticatedAccess is True)
            if not has_unauth_access:
                console.print(Panel(f'[bold red]❌ ChatGPT Requires Unauthenticated Access[/bold red]\n\nThis server requires authentication, but ChatGPT only supports:\n  • Unauthenticated (public) servers\n  • OAuth (not yet supported by mcp-agent install)\n\n[bold]Options:[/bold]\n\n1. Enable unauthenticated access for this server:\n   [cyan]mcp-agent cloud apps update --id {app_info.appId} --unauthenticated-access true[/cyan]\n\n2. Use a client that supports authentication:\n   [green]• Claude Code:[/green]    mcp-agent install {server_url} --client claude_code\n   [green]• Claude Desktop:[/green] mcp-agent install {server_url} --client claude_desktop\n   [green]• Cursor:[/green]         mcp-agent install {server_url} --client cursor\n   [green]• VSCode:[/green]         mcp-agent install {server_url} --client vscode', title='Installation Failed', border_style='red'))
                raise typer.Exit(1)
        except typer.Exit:
            raise
        except Exception as e:
            print_info(f'Warning: Could not verify unauthenticated access: {e}')
            print_info('Proceeding with installation, but ChatGPT may not be able to connect.')
        console.print(Panel(f'[bold]ChatGPT Setup Instructions[/bold]\n\n1. Open ChatGPT settings\n2. Navigate to the Apps & Connectors section\n3. Enable developer mode under advanced settings\n4. Select create on the top right corner of the panel\n5. Add a new server:\n   • URL: [cyan]{server_url}[/cyan]\n   • Transport: [cyan]sse[/cyan]\n\n[dim]Note: This server has unauthenticated access enabled.[/dim]', title='ChatGPT Configuration', border_style='green'))
        return
    server_name = name or app_name or 'mcp_agent'
    transport = 'sse' if server_url.rstrip('/').endswith('/sse') else 'http'
    if client_lc == 'claude_code':
        if dry_run:
            console.print('\n[bold yellow]DRY RUN - Would run:[/bold yellow]')
            console.print(f"claude mcp add {server_name} {server_url} -t {transport} -H 'Authorization: Bearer <api-key>' -s user")
            return
        try:
            cmd = ['claude', 'mcp', 'add', server_name, server_url, '-t', transport, '-H', f'Authorization: Bearer {effective_api_key}', '-s', 'user']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            print_success(f"Server '{server_name}' installed to Claude Code")
            console.print(result.stdout)
            return
        except subprocess.CalledProcessError as e:
            raise CLIError(f'Failed to add server to Claude Code: {e.stderr}') from e
        except FileNotFoundError:
            raise CLIError("Claude Code CLI not found. Make sure 'claude' command is available in your PATH.\nInstall from: https://docs.claude.com/en/docs/claude-code")
    if dry_run:
        print_info('[bold yellow]DRY RUN - No files will be written[/bold yellow]')
    client_config = CLIENT_CONFIGS[client_lc]
    config_path = client_config['path']()
    is_vscode = client_lc == 'vscode'
    is_claude_desktop = client_lc == 'claude_desktop'
    is_cursor = client_lc == 'cursor'
    existing_config = {}
    if config_path.exists():
        try:
            existing_config = json.loads(config_path.read_text(encoding='utf-8'))
            if is_claude_desktop or is_cursor:
                servers = existing_config.get('mcpServers', {})
            elif is_vscode:
                servers = existing_config.get('servers', {})
            else:
                servers = existing_config.get('mcp', {}).get('servers', {})
            if server_name in servers and (not force):
                raise CLIError(f"Server '{server_name}' already exists in {config_path}. Use --force to overwrite.")
        except json.JSONDecodeError as e:
            raise CLIError(f'Failed to parse existing config at {config_path}: {e}') from e
    server_config = _build_server_config(server_url, transport, for_claude_desktop=is_claude_desktop, for_vscode=is_vscode, api_key=effective_api_key)
    if is_claude_desktop or is_cursor:
        format_type = 'mcpServers'
    elif is_vscode:
        format_type = 'vscode'
    else:
        format_type = 'mcp'
    merged_config = _merge_mcp_json(existing_config, server_name, server_config, format_type)
    if dry_run:
        console.print('\n[bold]Would write to:[/bold]', config_path)
        console.print('\n[bold]Config:[/bold]')
        console.print_json(data=_redact_secrets(merged_config))
    else:
        try:
            _write_json(config_path, merged_config)
            print_success(f"Server '{server_name}' installed to {config_path}")
        except Exception as e:
            raise CLIError(f'Failed to write config file: {e}') from e
        if is_claude_desktop:
            auth_note = '[bold]Note:[/bold] Claude Desktop uses [cyan]mcp-remote[/cyan] to connect to HTTP/SSE servers\n[dim]API key embedded in config. Restart Claude Desktop to load the server.[/dim]'
        elif is_vscode:
            auth_note = f'[bold]Note:[/bold] VSCode format uses [cyan]type: {transport}[/cyan]\n[dim]API key embedded. Restart VSCode to load the server.[/dim]'
        elif is_cursor:
            auth_note = f'[bold]Note:[/bold] Cursor format uses [cyan]transport: {transport}[/cyan]\n[dim]API key embedded. Restart Cursor to load the server.[/dim]'
        else:
            auth_note = '[bold]Authentication:[/bold] API key embedded in config\n[dim]To update the key, re-run install with --force[/dim]'
        console.print(Panel(f'[bold green]✅ Installation Complete![/bold green]\n\nServer: [cyan]{server_name}[/cyan]\nURL: [cyan]{server_url}[/cyan]\nClient: [cyan]{client_config['description']}[/cyan]\nConfig: [cyan]{config_path}[/cyan]\n\n{auth_note}', title='MCP Server Installed', border_style='green'))
        console.print('\n💡 You may need to restart your MCP client for the changes to take effect.', style='dim')

def _build_server_config(server_url: str, transport: str='http', for_claude_desktop: bool=False, for_vscode: bool=False, api_key: str=None) -> dict:
    """Build server configuration dictionary with auth header.

    For Claude Desktop, wraps HTTP/SSE servers with mcp-remote stdio wrapper with actual API key.
    For VSCode, uses "type" field and top-level "servers" structure.
    For other clients (Cursor), uses "transport" field with "mcpServers" top-level structure.

    Args:
        server_url: The server URL
        transport: Transport type (http or sse)
        for_claude_desktop: Whether to use Claude Desktop format with mcp-remote
        for_vscode: Whether to use VSCode format with "type" field
        api_key: The actual API key (required for all clients)
    """
    if not api_key:
        raise ValueError('API key is required for server configuration')
    if for_claude_desktop:
        return {'command': 'npx', 'args': ['mcp-remote', server_url, '--header', f'Authorization: Bearer {api_key}']}
    elif for_vscode:
        return {'type': transport, 'url': server_url, 'headers': {'Authorization': f'Bearer {api_key}'}}
    else:
        return {'url': server_url, 'transport': transport, 'headers': {'Authorization': f'Bearer {api_key}'}}

def test_build_server_config():
    """Test server configuration building with auth header."""
    config = _build_server_config('https://example.com/mcp', 'http', api_key='test-key')
    assert config == {'url': 'https://example.com/mcp', 'transport': 'http', 'headers': {'Authorization': 'Bearer test-key'}}
    config_sse = _build_server_config('https://example.com/sse', 'sse', api_key='test-key')
    assert config_sse == {'url': 'https://example.com/sse', 'transport': 'sse', 'headers': {'Authorization': 'Bearer test-key'}}
    config_claude = _build_server_config('https://example.com/sse', 'sse', for_claude_desktop=True, api_key='test-api-key-123')
    assert config_claude == {'command': 'npx', 'args': ['mcp-remote', 'https://example.com/sse', '--header', 'Authorization: Bearer test-api-key-123']}

def test_merge_mcp_json_empty():
    """Test merging into empty config."""
    result = _merge_mcp_json({}, 'test-server', {'url': 'https://example.com', 'transport': 'http', 'headers': {'Authorization': 'Bearer test-key'}})
    assert result == {'mcp': {'servers': {'test-server': {'url': 'https://example.com', 'transport': 'http', 'headers': {'Authorization': 'Bearer test-key'}}}}}

def test_merge_mcp_json_claude_format():
    """Test merging with Claude Desktop format."""
    result = _merge_mcp_json({}, 'test-server', {'command': 'npx', 'args': ['mcp-remote', 'https://example.com/sse']}, format_type='mcpServers')
    assert result == {'mcpServers': {'test-server': {'command': 'npx', 'args': ['mcp-remote', 'https://example.com/sse']}}}

def test_merge_mcp_json_vscode_format():
    """Test merging with VSCode format."""
    result = _merge_mcp_json({}, 'test-server', {'type': 'sse', 'url': 'https://example.com', 'headers': {'Authorization': 'Bearer test-key'}}, format_type='vscode')
    assert result == {'servers': {'test-server': {'type': 'sse', 'url': 'https://example.com', 'headers': {'Authorization': 'Bearer test-key'}}}, 'inputs': []}

def test_merge_mcp_json_existing():
    """Test merging into existing config."""
    existing = {'mcp': {'servers': {'existing-server': {'url': 'https://existing.com', 'transport': 'http'}}}}
    result = _merge_mcp_json(existing, 'new-server', {'url': 'https://new.com', 'transport': 'http', 'headers': {'Authorization': 'Bearer test-key'}})
    assert result == {'mcp': {'servers': {'existing-server': {'url': 'https://existing.com', 'transport': 'http'}, 'new-server': {'url': 'https://new.com', 'transport': 'http', 'headers': {'Authorization': 'Bearer test-key'}}}}}

def test_merge_mcp_json_overwrite():
    """Test overwriting existing server."""
    existing = {'mcp': {'servers': {'test-server': {'url': 'https://old.com', 'transport': 'http'}}}}
    result = _merge_mcp_json(existing, 'test-server', {'url': 'https://new.com', 'transport': 'sse', 'headers': {'Authorization': 'Bearer test-key'}})
    assert result == {'mcp': {'servers': {'test-server': {'url': 'https://new.com', 'transport': 'sse', 'headers': {'Authorization': 'Bearer test-key'}}}}}

def test_install_missing_api_key(tmp_path):
    """Test install fails without API key."""
    with patch('mcp_agent.cli.commands.install.load_api_key_credentials', return_value=None):
        with patch('mcp_agent.cli.commands.install.settings') as mock_settings:
            mock_settings.API_KEY = None
            mock_settings.API_BASE_URL = 'http://test-api'
            with pytest.raises(CLIError, match='Must be logged in'):
                install(server_identifier=MOCK_APP_SERVER_URL, client='vscode', name=None, dry_run=False, force=False, api_url=None, api_key=None)

def test_install_invalid_client():
    """Test install fails with invalid client."""
    with patch('mcp_agent.cli.commands.install.load_api_key_credentials', return_value='test-key'):
        with patch('mcp_agent.cli.commands.install.settings') as mock_settings:
            mock_settings.API_KEY = 'test-key'
            mock_settings.API_BASE_URL = 'http://test-api'
            with pytest.raises(CLIError, match='Unsupported client'):
                install(server_identifier=MOCK_APP_SERVER_URL, client='invalid-client', name=None, dry_run=False, force=False, api_url=None, api_key=None)

def test_install_invalid_url():
    """Test install fails with non-URL identifier."""
    with patch('mcp_agent.cli.commands.install.load_api_key_credentials', return_value='test-key'):
        with patch('mcp_agent.cli.commands.install.settings') as mock_settings:
            mock_settings.API_KEY = 'test-key'
            mock_settings.API_BASE_URL = 'http://test-api'
            with pytest.raises(CLIError, match='must be a URL'):
                install(server_identifier='not-a-url', client='vscode', name=None, dry_run=False, force=False, api_url=None, api_key=None)

def test_install_vscode(tmp_path):
    """Test install to VSCode."""
    vscode_config = tmp_path / '.vscode' / 'mcp.json'
    with patch('mcp_agent.cli.commands.install.load_api_key_credentials', return_value='test-key'):
        with patch('mcp_agent.cli.commands.install.settings') as mock_settings:
            mock_settings.API_KEY = 'test-key'
            mock_settings.API_BASE_URL = 'http://test-api'
            with patch('mcp_agent.cli.commands.install.Path.cwd', return_value=tmp_path):
                install(server_identifier=MOCK_APP_SERVER_URL, client='vscode', name='test-server', dry_run=False, force=False, api_url='http://test-api', api_key='test-key')
                assert vscode_config.exists()
                config = json.loads(vscode_config.read_text())
                assert 'servers' in config
                assert 'inputs' in config
                assert 'test-server' in config['servers']
                server = config['servers']['test-server']
                assert server['url'] == MOCK_APP_SERVER_URL
                assert server['type'] == 'sse'
                assert server['headers']['Authorization'] == 'Bearer test-key'

def test_install_cursor_with_existing_config(tmp_path):
    """Test install to Cursor with existing configuration."""
    cursor_config = tmp_path / '.cursor' / 'mcp.json'
    cursor_config.parent.mkdir(parents=True, exist_ok=True)
    existing = {'mcpServers': {'existing-server': {'url': 'https://existing.com/mcp', 'transport': 'http'}}}
    cursor_config.write_text(json.dumps(existing, indent=2))
    with patch('mcp_agent.cli.commands.install.load_api_key_credentials', return_value='test-key'):
        with patch('mcp_agent.cli.commands.install.settings') as mock_settings:
            mock_settings.API_KEY = 'test-key'
            mock_settings.API_BASE_URL = 'http://test-api'
            with patch('mcp_agent.cli.commands.install.Path.home', return_value=tmp_path):
                install(server_identifier=MOCK_APP_SERVER_URL, client='cursor', name='new-server', dry_run=False, force=False, api_url='http://test-api', api_key='test-key')
                config = json.loads(cursor_config.read_text())
                assert len(config['mcpServers']) == 2
                assert 'existing-server' in config['mcpServers']
                assert 'new-server' in config['mcpServers']

def test_install_duplicate_without_force(tmp_path):
    """Test install fails when server already exists without --force."""
    vscode_config = tmp_path / '.vscode' / 'mcp.json'
    vscode_config.parent.mkdir(parents=True, exist_ok=True)
    existing = {'servers': {'test-server': {'url': 'https://old.com/mcp', 'type': 'http'}}, 'inputs': []}
    vscode_config.write_text(json.dumps(existing, indent=2))
    with patch('mcp_agent.cli.commands.install.load_api_key_credentials', return_value='test-key'):
        with patch('mcp_agent.cli.commands.install.settings') as mock_settings:
            mock_settings.API_KEY = 'test-key'
            mock_settings.API_BASE_URL = 'http://test-api'
            with patch('mcp_agent.cli.commands.install.Path.cwd', return_value=tmp_path):
                with pytest.raises(CLIError, match='already exists'):
                    install(server_identifier=MOCK_APP_SERVER_URL, client='vscode', name='test-server', dry_run=False, force=False, api_url='http://test-api', api_key='test-key')

def test_install_duplicate_with_force(tmp_path):
    """Test install overwrites when server exists with --force."""
    vscode_config = tmp_path / '.vscode' / 'mcp.json'
    vscode_config.parent.mkdir(parents=True, exist_ok=True)
    existing = {'servers': {'test-server': {'url': 'https://old.com/mcp', 'type': 'http'}}, 'inputs': []}
    vscode_config.write_text(json.dumps(existing, indent=2))
    with patch('mcp_agent.cli.commands.install.load_api_key_credentials', return_value='test-key'):
        with patch('mcp_agent.cli.commands.install.settings') as mock_settings:
            mock_settings.API_KEY = 'test-key'
            mock_settings.API_BASE_URL = 'http://test-api'
            with patch('mcp_agent.cli.commands.install.Path.cwd', return_value=tmp_path):
                install(server_identifier=MOCK_APP_SERVER_URL, client='vscode', name='test-server', dry_run=False, force=True, api_url='http://test-api', api_key='test-key')
                config = json.loads(vscode_config.read_text())
                assert config['servers']['test-server']['url'] == MOCK_APP_SERVER_URL

def test_install_chatgpt_requires_unauth_access(mock_app_with_auth):
    """Test ChatGPT install fails when server requires authentication."""
    import typer
    with patch('mcp_agent.cli.commands.install.load_api_key_credentials', return_value='test-key'):
        with patch('mcp_agent.cli.commands.install.settings') as mock_settings:
            mock_settings.API_KEY = 'test-key'
            mock_settings.API_BASE_URL = 'http://test-api'
            with patch('mcp_agent.cli.commands.install.MCPAppClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client.get_app = AsyncMock(return_value=mock_app_with_auth)
                mock_client_class.return_value = mock_client
                with pytest.raises(typer.Exit) as exc_info:
                    install(server_identifier=MOCK_APP_SERVER_URL, client='chatgpt', name=None, dry_run=False, force=False, api_url='http://test-api', api_key='test-key')
                assert exc_info.value.exit_code == 1

def test_install_chatgpt_with_unauth_server(mock_app_without_auth):
    """Test ChatGPT install succeeds with unauthenticated server."""
    with patch('mcp_agent.cli.commands.install.load_api_key_credentials', return_value='test-key'):
        with patch('mcp_agent.cli.commands.install.settings') as mock_settings:
            mock_settings.API_KEY = 'test-key'
            mock_settings.API_BASE_URL = 'http://test-api'
            with patch('mcp_agent.cli.commands.install.MCPAppClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client.get_app = AsyncMock(return_value=mock_app_without_auth)
                mock_client_class.return_value = mock_client
                install(server_identifier=MOCK_APP_SERVER_URL, client='chatgpt', name=None, dry_run=False, force=False, api_url='http://test-api', api_key='test-key')

def test_install_dry_run(tmp_path, capsys):
    """Test install in dry run mode."""
    with patch('mcp_agent.cli.commands.install.load_api_key_credentials', return_value='test-key'):
        with patch('mcp_agent.cli.commands.install.settings') as mock_settings:
            mock_settings.API_KEY = 'test-key'
            mock_settings.API_BASE_URL = 'http://test-api'
            with patch('mcp_agent.cli.commands.install.Path.cwd', return_value=tmp_path):
                install(server_identifier=MOCK_APP_SERVER_URL, client='vscode', name='test-server', dry_run=True, force=False, api_url='http://test-api', api_key='test-key')
                vscode_config = tmp_path / '.vscode' / 'mcp.json'
                assert not vscode_config.exists()

def test_install_sse_transport_detection(tmp_path):
    """Test that SSE transport is detected from URL."""
    vscode_config = tmp_path / '.vscode' / 'mcp.json'
    with patch('mcp_agent.cli.commands.install.load_api_key_credentials', return_value='test-key'):
        with patch('mcp_agent.cli.commands.install.settings') as mock_settings:
            mock_settings.API_KEY = 'test-key'
            mock_settings.API_BASE_URL = 'http://test-api'
            with patch('mcp_agent.cli.commands.install.Path.cwd', return_value=tmp_path):
                install(server_identifier='https://example.com/sse', client='vscode', name='test-server', dry_run=False, force=False, api_url='http://test-api', api_key='test-key')
                config = json.loads(vscode_config.read_text())
                assert config['servers']['test-server']['type'] == 'sse'

def test_install_http_transport_detection(tmp_path):
    """Test that HTTP transport is detected from URL."""
    vscode_config = tmp_path / '.vscode' / 'mcp.json'
    with patch('mcp_agent.cli.commands.install.load_api_key_credentials', return_value='test-key'):
        with patch('mcp_agent.cli.commands.install.settings') as mock_settings:
            mock_settings.API_KEY = 'test-key'
            mock_settings.API_BASE_URL = 'http://test-api'
            with patch('mcp_agent.cli.commands.install.Path.cwd', return_value=tmp_path):
                install(server_identifier='https://example.com/mcp', client='vscode', name='test-server', dry_run=False, force=False, api_url='http://test-api', api_key='test-key')
                config = json.loads(vscode_config.read_text())
                assert config['servers']['test-server']['type'] == 'http'

