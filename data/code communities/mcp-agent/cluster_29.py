# Cluster 29

def print_info(message: str, *args: Any, log: bool=True, console_output: bool=True, **kwargs: Any) -> None:
    """Print an informational message.

    Args:
        message: The message to print
        log: Whether to log to file
        console_output: Whether to print to console
    """
    if console_output:
        label = _create_label('', 'info')
        console.print(f'{label}{message}', *args, **kwargs)
    if log:
        logger.info(message)

def _create_label(text: str, style: str) -> str:
    """Create a fixed-width label with style markup."""
    dot = '⏺'
    return f' [{style}]{dot}[/{style}] '

def print_success(message: str, *args: Any, log: bool=True, console_output: bool=True, **kwargs: Any) -> None:
    """Print a success message."""
    if console_output:
        label = _create_label('', 'success')
        console.print(f'{label}{message}', *args, **kwargs)
    if log:
        logger.info(f'SUCCESS: {message}')

def print_warning(message: str, *args: Any, log: bool=True, console_output: bool=True, **kwargs: Any) -> None:
    """Print a warning message."""
    if console_output:
        label = _create_label('', 'warning')
        console.print(f'{label}{message}', *args, **kwargs)
    if log:
        logger.warning(message)

def retry_with_exponential_backoff(func: Callable, max_attempts: int=3, initial_delay: float=1.0, backoff_multiplier: float=2.0, max_delay: float=60.0, retryable_check: Optional[Callable[[Exception], bool]]=None, *args, **kwargs) -> Any:
    """Retry a function with exponential backoff.

    Args:
        func: The function to retry
        max_attempts: Maximum number of attempts (including the first one)
        initial_delay: Initial delay in seconds before first retry
        backoff_multiplier: Multiplier for delay between attempts
        max_delay: Maximum delay between attempts
        retryable_check: Function to determine if an error is retryable
        *args: Arguments to pass to func
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result of the successful function call

    Raises:
        RetryError: If all attempts fail with a retryable error
        Exception: The original exception if it's not retryable
    """
    if retryable_check is None:
        retryable_check = is_retryable_error
    last_exception = None
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt == max_attempts or not retryable_check(e):
                break
            print_warning(f'Attempt {attempt}/{max_attempts} failed: {e}. Retrying in {delay:.1f}s...')
            time.sleep(delay)
            delay = min(delay * backoff_multiplier, max_delay)
    if last_exception:
        if max_attempts > 1 and retryable_check(last_exception):
            raise RetryError(last_exception, max_attempts) from last_exception
        else:
            raise last_exception
    raise RuntimeError('Unexpected error in retry logic')

def _flush_version_check_message(timeout: float=0.5) -> None:
    """Wait briefly for the background check and print any queued message."""
    if not _version_check_started:
        return
    _version_check_event.wait(timeout)
    message = _version_check_message
    if message:
        print_info(message, console_output=True)

def get_app_defaults_from_config(config_file: Path | None) -> Tuple[str | None, str | None]:
    """Extract default app name/description from a config file."""
    if not config_file or not config_file.exists():
        return (None, None)
    try:
        loaded = get_settings(config_path=str(config_file), set_global=False)
    except Exception:
        return (None, None)
    app_name = loaded.name if isinstance(loaded.name, str) and loaded.name.strip() else None
    app_description = loaded.description if isinstance(loaded.description, str) and loaded.description.strip() else None
    return (app_name, app_description)

def delete_app(app_id_or_url: str=typer.Option(None, '--id', '-i', help='ID or server URL of the app or app configuration to delete.'), force: bool=typer.Option(False, '--force', '-f', help='Force delete the app or app configuration without confirmation.'), dry_run: bool=typer.Option(False, '--dry-run', help="Validate the deletion but don't actually delete."), api_url: Optional[str]=typer.Option(settings.API_BASE_URL, '--api-url', help='API base URL. Defaults to MCP_API_BASE_URL environment variable.', envvar=ENV_API_BASE_URL), api_key: Optional[str]=typer.Option(settings.API_KEY, '--api-key', help='API key for authentication. Defaults to MCP_API_KEY environment variable.', envvar=ENV_API_KEY)) -> None:
    """Delete an MCP App or App Configuration by ID."""
    effective_api_key = api_key or settings.API_KEY or load_api_key_credentials()
    if not effective_api_key:
        raise CLIError("Must be logged in to delete. Run 'mcp-agent login', set MCP_API_KEY environment variable or specify --api-key option.")
    client = MCPAppClient(api_url=api_url or DEFAULT_API_BASE_URL, api_key=effective_api_key)
    if not app_id_or_url:
        raise CLIError('You must provide an app ID, app config ID, or server URL to delete.')
    id_type = 'app'
    id_to_delete = None
    try:
        app_or_config = resolve_server(client, app_id_or_url)
        if isinstance(app_or_config, MCPAppConfiguration):
            id_to_delete = app_or_config.appConfigurationId
            id_type = 'app configuration'
        else:
            id_to_delete = app_or_config.appId
            id_type = 'app'
    except Exception as e:
        raise CLIError(f'Error retrieving app or config with ID or URL {app_id_or_url}: {str(e)}') from e
    if not force:
        confirmation = typer.confirm(f"Are you sure you want to delete the {id_type} with ID '{id_to_delete}'? This action cannot be undone.", default=False)
        if not confirmation:
            print_info('Deletion cancelled.')
            raise typer.Exit(0)
    if dry_run:
        try:
            can_delete = run_async(client.can_delete_app(id_to_delete) if id_type == 'app' else client.can_delete_app_configuration(id_to_delete))
            if can_delete:
                print_success(f"[Dry Run] Would delete {id_type} with ID '{id_to_delete}' if run without --dry-run flag.")
            else:
                print_error(f"[Dry Run] Cannot delete {id_type} with ID '{id_to_delete}'. Check permissions or if it exists.")
            return
        except Exception as e:
            raise CLIError(f'Error during dry run: {str(e)}') from e
    try:
        run_async(client.delete_app(id_to_delete) if id_type == 'app' else client.delete_app_configuration(id_to_delete))
        print_success(f"Successfully deleted the {id_type} with ID '{id_to_delete}'.")
    except UnauthenticatedError as e:
        raise CLIError("Invalid API key. Run 'mcp-agent login' or set MCP_API_KEY environment variable with new API key.") from e
    except Exception as e:
        raise CLIError(f'Error deleting {id_type}: {str(e)}') from e

def resolve_server(client: MCPAppClient, id_or_url_or_name: str) -> Union[MCPApp, MCPAppConfiguration]:
    """Resolve server from ID, server URL, app config ID, or app name (sync wrapper)."""
    return run_async(resolve_server_async(client, id_or_url_or_name))

def list_app_workflows(app_id_or_url: str=typer.Option(None, '--id', '-i', help='ID or server URL of the app or app configuration to list workflows from.'), api_url: Optional[str]=typer.Option(settings.API_BASE_URL, '--api-url', help='API base URL. Defaults to MCP_API_BASE_URL environment variable.', envvar=ENV_API_BASE_URL), api_key: Optional[str]=typer.Option(settings.API_KEY, '--api-key', help='API key for authentication. Defaults to MCP_API_KEY environment variable.', envvar=ENV_API_KEY)) -> None:
    """List workflow details (available workflows and recent workflow runs) for an MCP App."""
    effective_api_key = api_key or settings.API_KEY or load_api_key_credentials()
    if not effective_api_key:
        raise CLIError("Must be logged in list workflow details. Run 'mcp-agent login', set MCP_API_KEY environment variable or specify --api-key option.")
    client = MCPAppClient(api_url=api_url or DEFAULT_API_BASE_URL, api_key=effective_api_key)
    if not app_id_or_url:
        raise CLIError('You must provide an app ID or server URL to view its workflows.')
    try:
        app_or_config = resolve_server(client, app_id_or_url)
        if not app_or_config:
            raise CLIError(f"App or config with ID or URL '{app_id_or_url}' not found.")
        if not app_or_config.appServerInfo:
            raise CLIError(f"App or config with ID or URL '{app_id_or_url}' has no server info available.")
        server_url = app_or_config.appServerInfo.serverUrl
        if not server_url:
            raise CLIError('No server URL available for this app.')
        run_async(print_mcp_server_workflow_details(server_url=server_url, api_key=effective_api_key))
    except UnauthenticatedError as e:
        raise CLIError("Invalid API key. Run 'mcp-agent login' or set MCP_API_KEY environment variable with new API key.") from e
    except Exception as e:
        raise CLIError(f'Error listing workflow details for app or config with ID or URL {app_id_or_url}: {str(e)}') from e

def get_app_status(app_id_or_url: str=typer.Option(None, '--id', '-i', help='ID, server URL, or name of the app to get details for.'), api_url: Optional[str]=typer.Option(settings.API_BASE_URL, '--api-url', help='API base URL. Defaults to MCP_API_BASE_URL environment variable.', envvar=ENV_API_BASE_URL), api_key: Optional[str]=typer.Option(settings.API_KEY, '--api-key', help='API key for authentication. Defaults to MCP_API_KEY environment variable.', envvar=ENV_API_KEY)) -> None:
    """Get server details -- such as available tools, prompts, resources, and workflows -- for an MCP App."""
    effective_api_key = api_key or settings.API_KEY or load_api_key_credentials()
    if not effective_api_key:
        raise CLIError("Must be logged in to get app status. Run 'mcp-agent login', set MCP_API_KEY environment variable or specify --api-key option.", retriable=False)
    client = MCPAppClient(api_url=api_url or DEFAULT_API_BASE_URL, api_key=effective_api_key)
    if not app_id_or_url:
        raise CLIError('You must provide an app ID or server URL to get its status.')
    try:
        app_or_config = resolve_server(client, app_id_or_url)
        if not app_or_config:
            raise CLIError(f"App or config with ID or URL '{app_id_or_url}' not found.")
        if not app_or_config.appServerInfo:
            raise CLIError(f"App or config with ID or URL '{app_id_or_url}' has no server info available.")
        print_server_info(app_or_config.appServerInfo)
        server_url = app_or_config.appServerInfo.serverUrl
        if server_url:
            run_async(print_mcp_server_details(server_url=server_url, api_key=effective_api_key))
        else:
            raise CLIError('No server URL available for this app.')
    except UnauthenticatedError as e:
        raise CLIError("Invalid API key. Run 'mcp-agent login' or set MCP_API_KEY environment variable with new API key.", retriable=False) from e
    except Exception as e:
        raise CLIError(f'Error getting status for app or config with ID or URL {app_id_or_url}: {str(e)}') from e

def _print_servers_text(deployed_servers: List[MCPApp], configured_servers: List[MCPAppConfiguration], filter_param: Optional[str], sort_by: Optional[str]) -> None:
    """Print servers in text format."""
    print_info_header()
    if deployed_servers:
        num_servers = len(deployed_servers)
        print_info(f'Found {num_servers} deployed server(s):')
        print_servers(deployed_servers)
    else:
        console.print('\n[bold blue]🖥️  Deployed MCP Servers (0)[/bold blue]')
        print_info('No deployed servers found.')
    console.print('\n' + '─' * 80 + '\n')
    if configured_servers:
        num_configs = len(configured_servers)
        print_info(f'Found {num_configs} configured server(s):')
        print_server_configs(configured_servers)
    else:
        console.print('\n[bold blue]⚙️  Configured MCP Servers (0)[/bold blue]')
        print_info('No configured servers found.')
    if filter_param or sort_by:
        console.print(f'\n[dim]Applied filters: filter={filter_param or 'None'}, sort-by={sort_by or 'None'}[/dim]')
        filter_desc = f"filter='{filter_param}'" if filter_param else 'filter=None'
        sort_desc = f"sort-by='{sort_by}'" if sort_by else 'sort-by=None'
        print_info(f'Client-side {filter_desc}, {sort_desc}. Sort fields: name, created, status (-prefix for reverse).')

@handle_server_api_errors
def delete_server(id_or_url: str=typer.Argument(..., help='App ID, server URL, or app name to delete'), force: bool=typer.Option(False, '--force', '-f', help='Force deletion without confirmation prompt')) -> None:
    """Delete a specific MCP Server."""
    client = setup_authenticated_client()
    server = resolve_server(client, id_or_url)
    if isinstance(server, MCPApp):
        server_type = 'Deployed Server'
        delete_function = client.delete_app
    else:
        server_type = 'Configured Server'
        delete_function = client.delete_app_configuration
    server_name = get_server_name(server)
    server_id = get_server_id(server)
    if not force:
        console.print(Panel(f'Name: [cyan]{server_name}[/cyan]\nType: [cyan]{server_type}[/cyan]\nID: [cyan]{server_id}[/cyan]\n\n[bold red]⚠️  This action cannot be undone![/bold red]', title='Server to Delete', border_style='red', expand=False))
        confirm = typer.confirm(f'\nAre you sure you want to delete this {server_type.lower()}?')
        if not confirm:
            print_info('Deletion cancelled.')
            return
    if isinstance(server, MCPApp):
        can_delete = run_async(client.can_delete_app(server_id))
    else:
        can_delete = run_async(client.can_delete_app_configuration(server_id))
    if not can_delete:
        raise CLIError(f'You do not have permission to delete this {server_type.lower()}. You can only delete servers that you created.')
    deleted_id = run_async(delete_function(server_id))
    console.print(Panel(f'[green]✅ Successfully deleted {server_type.lower()}[/green]\n\nName: [cyan]{server_name}[/cyan]\nID: [cyan]{deleted_id}[/cyan]', title='Deletion Complete', border_style='green', expand=False))

def get_server_name(server: Union[MCPApp, MCPAppConfiguration]) -> str:
    """Get display name for a server.

    Args:
        server: Server object

    Returns:
        Server display name
    """
    if isinstance(server, MCPApp):
        return server.name or 'Unnamed'
    else:
        return server.app.name if server.app else 'Unnamed'

def get_server_id(server: Union[MCPApp, MCPAppConfiguration]) -> str:
    """Get ID for a server.

    Args:
        server: Server object

    Returns:
        Server ID
    """
    if isinstance(server, MCPApp):
        return server.appId
    else:
        return server.appConfigurationId

def _write_env_file(path: Path, values: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for key in sorted(values):
            handle.write(f'{key}={_format_env_value(values[key])}\n')

def _format_env_value(value: str) -> str:
    if value is None:
        return ''
    needs_quotes = bool(re.search('[^\\w@./-]', value))
    escaped = value.replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r').replace('"', '\\"')
    return f'"{escaped}"' if needs_quotes else escaped

def _confirm_overwrite(target: Path, force: bool, label: str) -> None:
    if target.exists() and (not force):
        overwrite = typer.confirm(f'{target} already exists. Overwrite {label}?', default=False)
        if not overwrite:
            print_info('Aborted.')
            raise typer.Exit(0)

def _resolve_app(app_identifier: Optional[str], config_dir: Path, api_url: Optional[str], api_key: str) -> MCPApp:
    """Resolve an MCP app from argument or config defaults."""
    client = MCPAppClient(api_url=api_url or settings.API_BASE_URL, api_key=api_key)
    config_file = config_dir / MCP_CONFIG_FILENAME if config_dir else None
    if app_identifier:
        server = resolve_server(client, app_identifier)
        if isinstance(server, MCPApp):
            return server
        if server.app:
            return server.app
        raise CLIError(f"Could not resolve MCP app for identifier '{app_identifier}'. Provide an app name or ID.")
    default_name, _ = get_app_defaults_from_config(config_file)
    if default_name:
        app_obj = run_async(client.get_app_by_name(default_name))
        if app_obj:
            return app_obj
    raise CLIError('Unable to determine which app to target. Provide an app name/id or run the command within a project directory.')

def _load_existing_handles(client: SecretsClient, app_id: str) -> Dict[str, str]:
    prefix = _env_secret_prefix(app_id)
    secrets = run_async(client.list_secrets(name_filter=prefix))
    handles: Dict[str, str] = {}
    for entry in secrets:
        handle = entry.get('secretId') or entry.get('secret_id')
        name = entry.get('name')
        if not handle or not name or (not name.startswith(prefix)):
            continue
        key = name[len(prefix):]
        handles[key] = handle
    return handles

def _env_secret_prefix(app_id: str) -> str:
    return f'apps/{app_id}/env/'

@app.command('list')
def list_secrets(app_name: Optional[str]=typer.Argument(None, help='App name, ID, or server URL. Defaults to project config.'), config_dir: Path=typer.Option(Path('.'), '--config-dir', '-c', help='Path to directory containing mcp_agent.config.yaml.', exists=True, file_okay=False, dir_okay=True, resolve_path=True), api_url: Optional[str]=typer.Option(settings.API_BASE_URL, '--api-url', help='API base URL. Defaults to MCP_API_BASE_URL environment variable.'), api_key: Optional[str]=typer.Option(settings.API_KEY, '--api-key', help='API key for authentication. Defaults to MCP_API_KEY environment variable.'), app_option: Optional[str]=typer.Option(None, '--app', '-a', help='App name, ID, or server URL (overrides positional argument).')) -> None:
    """List environment secrets associated with an app."""
    effective_key = _ensure_api_key(api_key)
    target_app = app_option or app_name
    app_obj = _resolve_app(target_app, config_dir, api_url, effective_key)
    client = _make_secrets_client(api_url, effective_key)
    handles = _load_existing_handles(client, app_obj.appId)
    if not handles:
        print_info(f"No secrets found for app '{app_obj.name or app_obj.appId}'.")
        return
    table = Table(show_header=True, header_style='bold magenta')
    table.add_column('Key', style='cyan')
    table.add_column('Secret Handle', style='green')
    for key, handle in sorted(handles.items()):
        masked = handle[:8] + '…' + handle[-6:] if len(handle) > 14 else handle
        table.add_row(key, masked)
    console.print(table)

def _ensure_api_key(api_key_option: Optional[str]) -> str:
    effective_key = api_key_option or settings.API_KEY or load_api_key_credentials()
    if not effective_key:
        raise CLIError("Must be logged in. Run 'mcp-agent login', set MCP_API_KEY, or pass --api-key.")
    return effective_key

def _make_secrets_client(api_url: Optional[str], api_key: str) -> SecretsClient:
    return SecretsClient(api_url=api_url or settings.API_BASE_URL, api_key=api_key)

@app.command('add')
def add_secret(key: Optional[str]=typer.Argument(None, help='Environment variable to store as a secret'), value: Optional[str]=typer.Argument(None, help='Secret value to store'), app_name_arg: Optional[str]=typer.Argument(None, help='App name, ID, or server URL. Defaults to project config.'), config_dir: Path=typer.Option(Path('.'), '--config-dir', '-c', help='Path to directory containing mcp_agent.config.yaml.', exists=True, file_okay=False, dir_okay=True, resolve_path=True), api_url: Optional[str]=typer.Option(settings.API_BASE_URL, '--api-url', help='API base URL. Defaults to MCP_API_BASE_URL environment variable.'), api_key: Optional[str]=typer.Option(settings.API_KEY, '--api-key', help='API key for authentication. Defaults to MCP_API_KEY environment variable.'), app_name_option: Optional[str]=typer.Option(None, '--app', '-a', help='App name, ID, or server URL (recommended when using --from-env-file).'), env_file: Optional[Path]=typer.Option(None, '--from-env-file', help='Path to a dotenv file to bulk add secrets.', exists=True, file_okay=True, dir_okay=False, resolve_path=True)) -> None:
    """Create or update environment secret(s)."""
    if env_file and (key or value):
        raise CLIError('Specify either --from-env-file or KEY/VALUE arguments (use --app to set the target app).')
    if not env_file and (not key or value is None):
        raise CLIError('KEY and VALUE are required unless --from-env-file is provided.')
    effective_key = _ensure_api_key(api_key)
    target_app = app_name_option or app_name_arg
    if env_file and (not target_app):
        raise CLIError('Provide an app via --app when using --from-env-file.')
    app_obj = _resolve_app(target_app, config_dir, api_url, effective_key)
    client = _make_secrets_client(api_url, effective_key)
    handles = _load_existing_handles(client, app_obj.appId)
    items: Dict[str, str] = {}
    if env_file:
        items = _load_env_file_values(env_file)
    else:
        items[key] = value
    for item_key, item_value in items.items():
        if not item_value:
            raise CLIError(f'Secret value must be non-empty for {item_key}.')
        handle = handles.get(item_key)
        if handle:
            run_async(client.set_secret_value(handle, item_value))
            print_success(f'Updated secret for {item_key}.')
        else:
            secret_name = f'{_env_secret_prefix(app_obj.appId)}{item_key}'
            handle = run_async(client.create_secret(name=secret_name, secret_type=SecretType.DEVELOPER, value=item_value))
            print_success(f'Created secret for {item_key}: {handle}')

def _load_env_file_values(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise CLIError(f'Env file not found: {path}')
    parsed = dotenv_values(path)
    values: Dict[str, str] = {}
    for key, value in parsed.items():
        if key and value is not None:
            values[key] = str(value)
    if not values:
        raise CLIError(f'No valid entries found in {path}')
    return values

@app.command('remove')
def remove_secret(key: str=typer.Argument(..., help='Environment variable to delete'), app_name: Optional[str]=typer.Argument(None, help='App name, ID, or server URL. Defaults to project config.'), config_dir: Path=typer.Option(Path('.'), '--config-dir', '-c', help='Path to directory containing mcp_agent.config.yaml.', exists=True, file_okay=False, dir_okay=True, resolve_path=True), api_url: Optional[str]=typer.Option(settings.API_BASE_URL, '--api-url', help='API base URL. Defaults to MCP_API_BASE_URL environment variable.'), api_key: Optional[str]=typer.Option(settings.API_KEY, '--api-key', help='API key for authentication. Defaults to MCP_API_KEY environment variable.'), app_name_option: Optional[str]=typer.Option(None, '--app', '-a', help='App name, ID, or server URL (overrides positional argument).')) -> None:
    """Delete a stored environment secret."""
    effective_key = _ensure_api_key(api_key)
    target_app = app_name_option or app_name
    app_obj = _resolve_app(target_app, config_dir, api_url, effective_key)
    client = _make_secrets_client(api_url, effective_key)
    handles = _load_existing_handles(client, app_obj.appId)
    handle = handles.get(key)
    if not handle:
        print_error(f'No secret stored for {key}.')
        raise typer.Exit(1)
    run_async(client.delete_secret(handle))
    print_success(f'Removed secret for {key}.')

@app.command('pull')
def pull_secrets(app_name: Optional[str]=typer.Argument(None, help='App name, ID, or server URL. Defaults to project config.'), config_dir: Path=typer.Option(Path('.'), '--config-dir', '-c', help='Path to directory containing mcp_agent.config.yaml.', exists=True, file_okay=False, dir_okay=True, resolve_path=True), format: str=typer.Option('env', '--format', '-f', help="Output format: 'env' writes a dotenv file, 'yaml' writes a secrets YAML.", case_sensitive=False), output: Optional[Path]=typer.Option(None, '--output', '-o', help='Destination file (defaults to .env.mcp-cloud for env format, mcp_agent.cloud.secrets.yaml for yaml format).', file_okay=True, dir_okay=False, resolve_path=True), force: bool=typer.Option(False, '--force', help='Overwrite output file without confirmation.'), api_url: Optional[str]=typer.Option(settings.API_BASE_URL, '--api-url', help='API base URL. Defaults to MCP_API_BASE_URL environment variable.'), api_key: Optional[str]=typer.Option(settings.API_KEY, '--api-key', help='API key for authentication. Defaults to MCP_API_KEY environment variable.'), app_name_option: Optional[str]=typer.Option(None, '--app', '-a', help='App name, ID, or server URL (overrides positional argument).')) -> None:
    """Fetch secret values and write them to a local YAML file."""
    effective_key = _ensure_api_key(api_key)
    target_app = app_name_option or app_name
    app_obj = _resolve_app(target_app, config_dir, api_url, effective_key)
    client = _make_secrets_client(api_url, effective_key)
    handles = _load_existing_handles(client, app_obj.appId)
    if not handles:
        print_info(f"No secrets found for app '{app_obj.name or app_obj.appId}'.")
        return
    resolved: Dict[str, str] = {}
    for key, handle in handles.items():
        value = run_async(client.get_secret_value(handle))
        resolved[key] = value
    format = format.lower()
    if format not in {'env', 'yaml'}:
        raise CLIError("Format must be either 'env' or 'yaml'.")
    default_path = Path('.env.mcp-cloud') if format == 'env' else Path('mcp_agent.cloud.secrets.yaml')
    dest = output or default_path
    label = 'dotenv file' if format == 'env' else 'YAML secrets file'
    _confirm_overwrite(dest, force, label)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if format == 'env':
        _write_env_file(dest, resolved)
    else:
        with open(dest, 'w', encoding='utf-8') as handle:
            yaml.safe_dump({'env': resolved}, handle, default_flow_style=False, sort_keys=True)
    print_success(f'Pulled {len(resolved)} secret(s) into {dest}.')

def tail_logs(app_identifier: str=typer.Argument(help='App ID, app configuration ID, or server URL to retrieve logs for'), since: Optional[str]=typer.Option(None, '--since', help="Show logs from duration ago (e.g., '1h', '30m', '2d')"), grep: Optional[str]=typer.Option(None, '--grep', help='Filter log messages matching this pattern (regex supported)'), follow: bool=typer.Option(False, '--follow', '-f', help='Stream logs continuously'), limit: Optional[int]=typer.Option(DEFAULT_LOG_LIMIT, '--limit', '-n', help=f'Maximum number of log entries to show (default: {DEFAULT_LOG_LIMIT})'), order_by: Optional[str]=typer.Option(None, '--order-by', help='Field to order by. Options: timestamp, severity (default: timestamp)'), asc: bool=typer.Option(False, '--asc', help='Sort in ascending order (oldest first)'), desc: bool=typer.Option(False, '--desc', help='Sort in descending order (newest first, default)'), format: Optional[str]=typer.Option('text', '--format', help='Output format. Options: text, json, yaml (default: text)')) -> None:
    """Tail logs for an MCP app deployment.

    Retrieve and optionally stream logs from deployed MCP apps. Supports filtering
    by time duration, text patterns, and continuous streaming.

    Examples:
        # Get last 50 logs from an app
        mcp-agent cloud logger tail app_abc123 --limit 50

        # Stream logs continuously
        mcp-agent cloud logger tail app_abc123 --follow

        # Show logs from the last hour with error filtering
        mcp-agent cloud logger tail app_abc123 --since 1h --grep "ERROR|WARN"

        # Follow logs and filter for specific patterns
        mcp-agent cloud logger tail app_abc123 --follow --grep "authentication.*failed"

        # Use server URL instead of app ID
        mcp-agent cloud logger tail https://abc123.mcpcloud.ai --follow
    """
    credentials = load_credentials()
    if not credentials and _settings.API_KEY:
        credentials = UserCredentials(api_key=_settings.API_KEY)
    if not credentials:
        print_error("Not authenticated. Set MCP_API_KEY environment variable or run 'mcp-agent login'.")
        raise typer.Exit(4)
    if follow and since:
        print_error('--since cannot be used with --follow (streaming mode)')
        raise typer.Exit(6)
    if follow and limit != DEFAULT_LOG_LIMIT:
        print_error('--limit cannot be used with --follow (streaming mode)')
        raise typer.Exit(6)
    if follow and order_by:
        print_error('--order-by cannot be used with --follow (streaming mode)')
        raise typer.Exit(6)
    if follow and (asc or desc):
        print_error('--asc/--desc cannot be used with --follow (streaming mode)')
        raise typer.Exit(6)
    if order_by and order_by not in ['timestamp', 'severity']:
        print_error("--order-by must be 'timestamp' or 'severity'")
        raise typer.Exit(6)
    if asc and desc:
        print_error('Cannot use both --asc and --desc together')
        raise typer.Exit(6)
    if format and format not in ['text', 'json', 'yaml']:
        print_error("--format must be 'text', 'json', or 'yaml'")
        raise typer.Exit(6)
    client = setup_authenticated_client()
    server = resolve_server(client, app_identifier)
    try:
        if follow:
            asyncio.run(_stream_logs(server=server, credentials=credentials, grep_pattern=grep, app_identifier=app_identifier, format=format))
        else:
            asyncio.run(_fetch_logs(server=server, since=since, grep_pattern=grep, limit=limit, order_by=order_by, asc=asc, desc=desc, format=format, app_identifier=app_identifier))
    except KeyboardInterrupt:
        console.print('\n[yellow]Interrupted by user[/yellow]')
        sys.exit(0)
    except Exception as e:
        raise CLIError(str(e))

def _handle_wrangler_error(e: subprocess.CalledProcessError) -> None:
    """Parse and present Wrangler errors in a clean format."""
    error_output = e.stderr or e.stdout or 'No error output available'
    clean_output = re.sub('\\x1B(?:[@-Z\\\\-_]|\\[[0-?]*[ -/]*[@-~])', '', error_output)
    console.print('\n')
    if 'Unauthorized 401' in clean_output or '401' in clean_output:
        print_error("Authentication failed: Invalid or expired API key for bundling. Run 'mcp-agent login' or set MCP_API_KEY environment variable with new API key.")
        return
    lines = clean_output.strip().split('\n')
    main_errors = []
    warnings = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.search('^\\[ERROR\\]|^✘.*\\[ERROR\\]', line):
            error_match = re.search('(?:\\[ERROR\\]|\\[97mERROR\\[.*?\\])\\s*(.*)', line)
            if error_match:
                main_errors.append(error_match.group(1).strip())
            else:
                main_errors.append(line)
        elif re.search('^\\[WARNING\\]|^▲.*\\[WARNING\\]', line):
            warning_match = re.search('(?:\\[WARNING\\]|\\[30mWARNING\\[.*?\\])\\s*(.*)', line)
            if warning_match:
                warnings.append(warning_match.group(1).strip())
        elif line.startswith('ERROR:') or line.startswith('Error:'):
            main_errors.append(line)
    if warnings:
        for warning in warnings:
            print_warning(warning)
    if main_errors:
        for error in main_errors:
            print_error(error)
    else:
        print_error('Bundling failed with error:')
        print_error(clean_output)

def logout() -> None:
    """Clear credentials.

    Removes stored authentication information.
    """
    credentials = load_credentials()
    if not credentials:
        print_info('Not currently logged in.')
        return
    user_info = 'current user'
    if credentials.username:
        user_info = f"user '{credentials.username}'"
    elif credentials.email:
        user_info = f"user '{credentials.email}'"
    if not Confirm.ask(f'Are you sure you want to logout {user_info}?', default=False):
        print_info('Logout cancelled.')
        return
    if clear_credentials():
        print_success('Successfully logged out.')
    else:
        print_info('No credentials were found to clear.')

def _load_user_credentials(api_key: str) -> UserCredentials:
    """Load credentials with user profile data fetched from API.

    Args:
        api_key: The API key

    Returns:
        UserCredentials object with profile data if available
    """

    async def fetch_profile() -> UserCredentials:
        """Fetch user profile from the API."""
        client = APIClient(settings.API_BASE_URL, api_key)
        response = await client.post('user/get_profile', {})
        user_data = response.json()
        user_profile = user_data.get('user', {})
        return UserCredentials(api_key=api_key, username=user_profile.get('name'), email=user_profile.get('email'))
    try:
        return asyncio.run(fetch_profile())
    except Exception as e:
        print_warning(f'Could not fetch user profile: {str(e)}')
        return UserCredentials(api_key=api_key)

def login(api_key: Optional[str]=typer.Option(None, '--api-key', help='Optionally set an existing API key to use for authentication, bypassing manual login.', envvar='MCP_API_KEY'), no_open: bool=typer.Option(False, '--no-open', help="Don't automatically open browser for authentication.")) -> str:
    """Authenticate to MCP Agent Cloud API.

    Direct to the api keys page for obtaining credentials, routing through login.

    Args:
        api_key: Optionally set an existing API key to use for authentication, bypassing manual login.
        no_open: Don't automatically open browser for authentication.

    Returns:
        API key string. Prints success message if login is successful.
    """
    existing_credentials = load_credentials()
    if existing_credentials and (not existing_credentials.is_token_expired):
        if not Confirm.ask('You are already logged in. Do you want to login again?'):
            print_info('Using existing credentials.')
            return existing_credentials.api_key
    if api_key:
        print_info('Using provided API key for authentication (MCP_API_KEY).')
        if not _is_valid_api_key(api_key):
            raise CLIError('Invalid API key provided.', retriable=False)
        credentials = _load_user_credentials(api_key)
        save_credentials(credentials)
        print_success('API key set.')
        if credentials.username:
            print_info(f'Logged in as: {credentials.username}')
        return api_key
    base_url = settings.API_BASE_URL
    return _handle_browser_auth(base_url, no_open)

def _handle_browser_auth(base_url: str, no_open: bool) -> str:
    """Handle browser-based authentication flow.

    Args:
        base_url: API base URL
        no_open: Whether to skip automatic browser opening

    Returns:
        API key string
    """
    auth_url = f'{base_url}/{DEFAULT_API_AUTH_PATH}'
    if not no_open:
        print_info('Opening MCP Agent Cloud API login in browser...')
        print_info(f"If the browser doesn't automatically open, you can manually visit: {auth_url}")
        typer.launch(auth_url)
    else:
        print_info(f'Please visit: {auth_url}')
    return _handle_manual_key_input()

def _handle_manual_key_input() -> str:
    """Handle manual API key input.

    Returns:
        API key string
    """
    input_api_key = Prompt.ask('Please enter your API key :key:')
    if not input_api_key:
        print_warning('No API key provided.')
        raise CLIError('Failed to set valid API key', retriable=False)
    if not _is_valid_api_key(input_api_key):
        print_warning('Invalid API key provided.')
        raise CLIError('Failed to set valid API key', retriable=False)
    credentials = _load_user_credentials(input_api_key)
    save_credentials(credentials)
    print_success('API key set.')
    if credentials.username:
        print_info(f'Logged in as: {credentials.username}')
    return input_api_key

def update_app(app_id_or_name: str=typer.Argument(..., help='ID, server URL, configuration ID, or name of the app to update.', show_default=False), name: Optional[str]=typer.Option(None, '--name', '-n', help='Set a new name for the app.'), description: Optional[str]=typer.Option(None, '--description', '-d', help='Set a new description for the app. Use an empty string to clear it.'), unauthenticated_access: Optional[bool]=typer.Option(None, '--no-auth/--auth', help='Allow unauthenticated access to the app server (--no-auth) or require authentication (--auth). If omitted, the current setting is preserved.'), api_url: Optional[str]=typer.Option(settings.API_BASE_URL, '--api-url', help='API base URL. Defaults to MCP_API_BASE_URL environment variable.', envvar=ENV_API_BASE_URL), api_key: Optional[str]=typer.Option(settings.API_KEY, '--api-key', help='API key for authentication. Defaults to MCP_API_KEY environment variable.', envvar=ENV_API_KEY)) -> None:
    """Update metadata or authentication settings for a deployed MCP App."""
    if name is None and description is None and (unauthenticated_access is None):
        raise CLIError('Specify at least one of --name, --description, or --no-auth/--auth to update.', retriable=False)
    effective_api_key = api_key or settings.API_KEY or load_api_key_credentials()
    if not effective_api_key:
        raise CLIError("Must be logged in to update an app. Run 'mcp-agent login', set MCP_API_KEY environment variable or specify --api-key option.", retriable=False)
    client = MCPAppClient(api_url=api_url or DEFAULT_API_BASE_URL, api_key=effective_api_key)
    try:
        resolved = resolve_server(client, app_id_or_name)
        if isinstance(resolved, MCPAppConfiguration):
            if not resolved.app:
                raise CLIError('Could not resolve the underlying app for the configuration provided.')
            target_app: MCPApp = resolved.app
        else:
            target_app = resolved
        updated_app = run_async(client.update_app(app_id=target_app.appId, name=name, description=description, unauthenticated_access=unauthenticated_access))
        short_id = f'{updated_app.appId[:8]}…'
        print_success(f"Updated app '{updated_app.name or target_app.name}' (ID: `{short_id}`)")
        if updated_app.description is not None:
            desc_text = updated_app.description or '(cleared)'
            print_info(f'Description: {desc_text}')
        app_server_info = updated_app.appServerInfo
        if app_server_info and app_server_info.serverUrl:
            print_info(f'Server URL: {app_server_info.serverUrl}')
            if app_server_info.unauthenticatedAccess is not None:
                auth_msg = 'Unauthenticated access allowed' if app_server_info.unauthenticatedAccess else 'Authentication required'
                print_info(f'Authentication: {auth_msg}')
    except UnauthenticatedError as e:
        raise CLIError("Invalid API key. Run 'mcp-agent login' or set MCP_API_KEY environment variable with new API key.") from e
    except CLIError:
        raise
    except Exception as e:
        raise CLIError(f'Error updating app: {str(e)}') from e

def list_apps(name_filter: str=typer.Option(None, '--name', '-n', help='Filter apps by name'), max_results: int=typer.Option(100, '--max-results', '-m', help='Maximum number of results to return'), api_url: Optional[str]=typer.Option(settings.API_BASE_URL, '--api-url', help='API base URL. Defaults to MCP_API_BASE_URL environment variable.', envvar=ENV_API_BASE_URL), api_key: Optional[str]=typer.Option(settings.API_KEY, '--api-key', help='API key for authentication. Defaults to MCP_API_KEY environment variable.', envvar=ENV_API_KEY)) -> None:
    """List MCP Apps with optional filtering by name."""
    effective_api_key = api_key or settings.API_KEY or load_api_key_credentials()
    if not effective_api_key:
        raise CLIError("Must be logged in to list apps. Run 'mcp-agent login', set MCP_API_KEY environment variable or specify --api-key option.")
    client = MCPAppClient(api_url=api_url or DEFAULT_API_BASE_URL, api_key=effective_api_key)
    try:

        async def parallel_requests():
            return await asyncio.gather(client.list_apps(name_filter=name_filter, max_results=max_results), client.list_app_configurations(name_filter=name_filter, max_results=max_results))
        list_apps_res, list_app_configs_res = run_async(parallel_requests())
        print_info_header()
        if list_apps_res.apps:
            num_apps = list_apps_res.totalCount or len(list_apps_res.apps)
            print_info(f'Found {num_apps} deployed app(s):')
            print_apps(list_apps_res.apps)
        else:
            console.print('\n[bold blue]📦 Deployed MCP Apps (0)[/bold blue]')
            print_info('No deployed apps found.')
        console.print('\n' + '─' * 80 + '\n')
        if list_app_configs_res.appConfigurations:
            num_configs = list_app_configs_res.totalCount or len(list_app_configs_res.appConfigurations)
            print_info(f'Found {num_configs} configured app(s):')
            print_app_configs(list_app_configs_res.appConfigurations)
        else:
            console.print('\n[bold blue]⚙️  Configured MCP Apps (0)[/bold blue]')
            print_info('No configured apps found.')
    except UnauthenticatedError as e:
        raise CLIError("Invalid API key. Run 'mcp-agent login' or set MCP_API_KEY environment variable with new API key.") from e
    except Exception as e:
        raise CLIError(f'Error listing apps: {str(e)}') from e

def print_info_header() -> None:
    """Print a styled header explaining the following tables"""
    console.print(Panel('Deployed Servers: [cyan]MCP Servers which you have bundled and deployed, as a developer[/cyan]\nConfigured Servers: [cyan]MCP Servers which you have configured to use with your MCP clients[/cyan]', title='MCP Servers', border_style='blue', expand=False))

def load_credentials() -> Optional[UserCredentials]:
    """Load user credentials from the credentials file.

    Returns:
        UserCredentials object if it exists, None otherwise
    """
    primary_path = os.path.expanduser(DEFAULT_CREDENTIALS_PATH)
    paths_to_try = [primary_path] + [os.path.expanduser(p) for p in ALTERNATE_CREDENTIALS_PATHS]
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return UserCredentials.from_json(f.read())
            except (json.JSONDecodeError, KeyError, ValueError):
                try:
                    print_warning(f"Detected corrupted credentials file at {path}. Please run 'mcp-agent login' again to re-authenticate.")
                except Exception:
                    pass
                continue
    return None

def load_api_key_credentials() -> Optional[str]:
    """Load an API key from the credentials file (backward compatibility).

    Returns:
        String. API key if it exists, None otherwise
    """
    credentials = load_credentials()
    return credentials.api_key if credentials else None

def test_format_env_value_quotes_special_characters():
    assert _format_env_value('plain') == 'plain'
    assert _format_env_value('token with spaces') == '"token with spaces"'
    assert _format_env_value('value"with"quotes') == '"value\\"with\\"quotes"'
    assert _format_env_value('multi\nline') == '"multi\\nline"'

def test_write_env_file(tmp_path: Path):
    values = {'B_KEY': 'b value', 'A_KEY': 'alpha'}
    env_path = tmp_path / '.env.mcp-cloud'
    _write_env_file(env_path, values)
    contents = env_path.read_text(encoding='utf-8').splitlines()
    assert contents == ['A_KEY=alpha', 'B_KEY="b value"']

def test_load_env_file_values(tmp_path: Path):
    env_path = tmp_path / '.env'
    env_path.write_text('A_KEY="alpha value"\nB_KEY=beta\n', encoding='utf-8')
    values = _load_env_file_values(env_path)
    assert values == {'A_KEY': 'alpha value', 'B_KEY': 'beta'}

def test_load_env_file_values_errors_for_missing_entries(tmp_path: Path):
    env_path = tmp_path / '.env'
    env_path.write_text('', encoding='utf-8')
    with pytest.raises(Exception):
        _load_env_file_values(env_path)

