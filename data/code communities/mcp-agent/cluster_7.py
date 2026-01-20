# Cluster 7

def get_settings(config_path: str | None=None, set_global: bool=True) -> Settings:
    """Get settings instance, automatically loading from config file if available.

    Args:
        config_path: Optional path to config file. If None, searches for config automatically.
        set_global: Whether to set the loaded settings as the global singleton. Default is True for backward
                    compatibility. Set to False for multi-threaded environments to avoid global state modification.

    Returns:
        Settings instance with loaded configuration.
    """

    def deep_merge(base: dict, update: dict, path: tuple=()) -> dict:
        """Recursively merge two dictionaries, preserving nested structures.

        Special handling for 'exporters' lists under 'otel' key:
        - Concatenates lists instead of replacing them
        - Allows combining exporters from config and secrets files
        """
        merged = base.copy()
        for key, value in update.items():
            current_path = path + (key,)
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = deep_merge(merged[key], value, current_path)
            elif key in merged and isinstance(merged[key], list) and isinstance(value, list) and (current_path in {('otel', 'exporters'), ('workflow_task_modules',)}):
                combined = merged[key] + value
                deduped = []
                for item in combined:
                    if not any((existing == item for existing in deduped)):
                        deduped.append(item)
                merged[key] = deduped
            else:
                merged[key] = value
        return merged
    if set_global:
        global _settings
        if _settings:
            return _settings
    merged_settings = {}
    preload_settings = PreloadSettings()
    preload_config = preload_settings.preload
    if preload_config:
        try:
            buf = StringIO()
            buf.write(preload_config)
            buf.seek(0)
            yaml_settings = yaml.safe_load(buf) or {}
            return Settings(**yaml_settings)
        except Exception as e:
            if preload_settings.preload_strict:
                raise ValueError('MCP App Preloaded Settings value failed validation') from e
            print(f'MCP App Preloaded Settings value failed validation: {e}', file=sys.stderr)
    if config_path:
        config_file = Path(config_path)
        if not _check_file_exists(config_file):
            raise FileNotFoundError(f'Config file not found: {config_path}')
    else:
        config_file = Settings.find_config()
    if config_file and _check_file_exists(config_file):
        file_content = _read_file_content(config_file)
        yaml_settings = _load_yaml_from_string(file_content)
        merged_settings = yaml_settings
        config_dir = config_file.parent
        secrets_found = False
        for secrets_filename in ['mcp-agent.secrets.yaml', 'mcp_agent.secrets.yaml']:
            secrets_file = config_dir / secrets_filename
            if _check_file_exists(secrets_file):
                secrets_content = _read_file_content(secrets_file)
                yaml_secrets = _load_yaml_from_string(secrets_content)
                merged_settings = deep_merge(merged_settings, yaml_secrets)
                secrets_found = True
                break
        if not secrets_found:
            secrets_file = Settings.find_secrets()
            if secrets_file and _check_file_exists(secrets_file):
                secrets_content = _read_file_content(secrets_file)
                yaml_secrets = _load_yaml_from_string(secrets_content)
                merged_settings = deep_merge(merged_settings, yaml_secrets)
        settings = Settings(**merged_settings)
        if set_global:
            _set_and_warn_global_settings(settings)
        return settings
    settings = Settings()
    if set_global:
        _set_and_warn_global_settings(settings)
    return settings

def _check_file_exists(file_path: str | Path) -> bool:
    """Check if a file exists at the given path."""
    return Path(file_path).exists()

def _read_file_content(file_path: str | Path) -> str:
    """Read and return the contents of a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def _load_yaml_from_string(yaml_content: str) -> dict:
    """Load YAML content from a string."""
    return yaml.safe_load(yaml_content) or {}

def _set_and_warn_global_settings(settings: Settings) -> None:
    """Set global settings and warn if called from non-main thread."""
    global _settings
    _settings = settings
    if threading.current_thread() is not threading.main_thread():
        warnings.warn('get_settings() is setting the global Settings singleton from a non-main thread. In multithreaded environments, use get_settings(set_global=False) to avoid global state modification, or pass the Settings instance explicitly to MCPApp(settings=...).', stacklevel=3)

@app.callback(invoke_without_command=True)
def serve(ctx: typer.Context, script: Optional[str]=typer.Option(None, '--script', '-s', help='Python script with MCPApp'), transport: str=typer.Option('stdio', '--transport', '-t', help='Transport: stdio|http|sse'), port: Optional[int]=typer.Option(None, '--port', '-p', help='Port for HTTP/SSE server'), host: str=typer.Option('0.0.0.0', '--host', '-H', help='Host for HTTP/SSE server'), reload: bool=typer.Option(False, '--reload', '-r', help='Auto-reload on code changes'), debug: bool=typer.Option(False, '--debug', '-d', help='Enable debug mode'), workers: int=typer.Option(1, '--workers', '-w', help='Number of worker processes (HTTP only)'), env: Optional[List[str]]=typer.Option(None, '--env', '-e', help='Environment variables (KEY=value)'), config: Optional[Path]=typer.Option(None, '--config', '-c', help='Config file path'), show_tools: bool=typer.Option(False, '--show-tools', help='Display available tools on startup'), monitor: bool=typer.Option(False, '--monitor', '-m', help='Enable monitoring dashboard'), ssl_certfile: Optional[Path]=typer.Option(None, '--ssl-certfile', help='Path to SSL certificate file (HTTP/SSE)'), ssl_keyfile: Optional[Path]=typer.Option(None, '--ssl-keyfile', help='Path to SSL private key file (HTTP/SSE)')) -> None:
    """
    Start an MCP server for your app.

    Examples:
        mcp-agent dev serve --script agent.py
        mcp-agent dev serve --transport http --port 8000
        mcp-agent dev serve --reload --debug
    """
    if ctx.invoked_subcommand:
        return
    if env:
        for env_pair in env:
            if '=' in env_pair:
                key, value = env_pair.split('=', 1)
                os.environ[key] = value
                if debug:
                    console.print(f'[dim]Set {key}={value}[/dim]')

    async def _run():
        script_path = detect_default_script(Path(script) if script else None)
        if not script_path.exists():
            console.print(f'[red]Script not found: {script_path}[/red]')
            console.print('\n[dim]Create a main.py (preferred) or agent.py file, or specify --script[/dim]')
            raise typer.Exit(1)
        console.print('\n[bold cyan]🚀 MCP-Agent Server[/bold cyan]')
        console.print(f'Script: [green]{script_path}[/green]')
        settings_override = None
        if config:
            try:
                from mcp_agent.config import get_settings as _get_settings
                settings_override = _get_settings(config_path=str(config))
                console.print(f'Config: [green]{config}[/green]')
            except Exception as _e:
                console.print(f'[red]Failed to load config: {_e}[/red]')
                if debug:
                    import traceback
                    console.print(f'[dim]{traceback.format_exc()}[/dim]')
                raise typer.Exit(1)
        try:
            app_obj = load_user_app(script_path, settings_override=settings_override)
        except Exception as e:
            console.print(f'[red]Failed to load app: {e}[/red]')
            if debug:
                import traceback
                console.print(f'[dim]{traceback.format_exc()}[/dim]')
            raise typer.Exit(1)
        await app_obj.initialize()
        mcp = create_mcp_server_for_app(app_obj)
        info_table = Table(show_header=False, box=None)
        info_table.add_column('Property', style='cyan')
        info_table.add_column('Value')
        info_table.add_row('App Name', app_obj.name)
        info_table.add_row('Transport', transport.upper())
        if transport == 'stdio':
            info_table.add_row('Mode', 'Standard I/O')
        else:
            address = f'{host}:{port or 8000}'
            info_table.add_row('Address', f'http://{address}')
            if transport == 'sse':
                info_table.add_row('SSE Endpoint', f'http://{address}/sse')
            elif transport == 'http':
                info_table.add_row('HTTP Endpoint', f'http://{address}/mcp')
        if hasattr(app_obj, 'workflows') and app_obj.workflows:
            info_table.add_row('Workflows', str(len(app_obj.workflows)))
        if hasattr(app_obj, 'agents') and app_obj.agents:
            info_table.add_row('Agents', str(len(app_obj.agents)))
        settings = get_settings()
        if settings.mcp and settings.mcp.servers:
            info_table.add_row('MCP Servers', str(len(settings.mcp.servers)))
        console.print(Panel(info_table, title='[bold]Server Information[/bold]', border_style='green'))
        if show_tools:
            try:
                tools_list = []
                if hasattr(mcp, 'list_tools'):
                    tools_response = await mcp.list_tools()
                    if tools_response and hasattr(tools_response, 'tools'):
                        tools_list = tools_response.tools
                if tools_list:
                    console.print('\n[bold]Available Tools:[/bold]')
                    tools_table = Table(show_header=True, header_style='cyan')
                    tools_table.add_column('Tool', style='green')
                    tools_table.add_column('Description')
                    for tool in tools_list[:10]:
                        desc = tool.description[:60] + '...' if len(tool.description) > 60 else tool.description
                        tools_table.add_row(tool.name, desc)
                    if len(tools_list) > 10:
                        tools_table.add_row('...', f'and {len(tools_list) - 10} more')
                    console.print(tools_table)
            except Exception:
                pass
        server_monitor = ServerMonitor() if monitor else None
        shutdown_event = asyncio.Event()

        def signal_handler(sig, frame):
            console.print('\n[yellow]Shutting down server...[/yellow]')
            shutdown_event.set()
            os._exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if transport == 'stdio':
            console.print('\n[green]Server running on STDIO[/green]')
            console.print('[dim]Ready for MCP client connections via standard I/O[/dim]\n')
            if debug:
                console.print('[yellow]Debug mode: Messages will be logged to stderr[/yellow]\n')
            try:
                await mcp.run_stdio_async()
            except Exception as e:
                if 'Broken pipe' not in str(e):
                    console.print(f'[red]Server error: {e}[/red]')
                    if debug:
                        import traceback
                        console.print(f'[dim]{traceback.format_exc()}[/dim]')
        elif transport in ['http', 'sse']:
            try:
                import uvicorn
                uvicorn_config = uvicorn.Config(mcp.streamable_http_app if transport == 'http' else mcp.sse_app, host=host, port=port or 8000, log_level='debug' if debug else 'info', reload=reload, workers=workers if not reload else 1, access_log=debug)
                if ssl_certfile and ssl_keyfile:
                    uvicorn_config.ssl_certfile = str(ssl_certfile)
                    uvicorn_config.ssl_keyfile = str(ssl_keyfile)
                server = uvicorn.Server(uvicorn_config)
                console.print(f'\n[green]Server running on {transport.upper()}[/green]')
                console.print(f'[bold]URL:[/bold] http://{host}:{port or 8000}')
                if transport == 'sse':
                    console.print(f'[bold]SSE:[/bold] http://{host}:{port or 8000}/sse')
                elif transport == 'http':
                    console.print(f'[bold]HTTP:[/bold] http://{host}:{port or 8000}/mcp')
                console.print('\n[dim]Press Ctrl+C to stop the server[/dim]\n')
                if monitor and server_monitor:
                    import time as _time
                    server_monitor.start_time = _time.time()

                    async def update_monitor():
                        with Live(auto_refresh=True, refresh_per_second=1) as live:
                            while not shutdown_event.is_set():
                                table = _create_status_table(server_monitor, transport, f'http://{host}:{port or 8000}')
                                live.update(Panel(table, title='[bold]Server Monitor[/bold]', border_style='cyan'))
                                await asyncio.sleep(1)
                    asyncio.create_task(update_monitor())
                await server.serve()
            except ImportError:
                console.print('[red]uvicorn not installed[/red]')
                console.print('\n[dim]Install with: pip install uvicorn[/dim]')
                raise typer.Exit(1)
            except Exception as e:
                console.print(f'[red]Failed to start {transport.upper()} server: {e}[/red]')
                if debug:
                    import traceback
                    console.print(f'[dim]{traceback.format_exc()}[/dim]')
                raise typer.Exit(1)
        else:
            console.print(f'[red]Unknown transport: {transport}[/red]')
            console.print('[dim]Supported: stdio, http, sse[/dim]')
            raise typer.Exit(1)
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print('\n[yellow]Server stopped[/yellow]')
    except Exception as e:
        if debug:
            console.print(f'[red]Unexpected error: {e}[/red]')
        sys.exit(1)

def _create_status_table(monitor: ServerMonitor, transport: str, address: str) -> Table:
    """Create a status table for the server."""
    stats = monitor.get_stats()
    table = Table(show_header=False, box=None)
    table.add_column('Key', style='cyan')
    table.add_column('Value')
    table.add_row('Transport', transport.upper())
    table.add_row('Address', address)
    table.add_row('Status', '[green]● Running[/green]')
    table.add_row('Uptime', f'{stats['uptime']}s')
    table.add_row('Requests', str(stats['requests']))
    table.add_row('Errors', str(stats['errors']))
    table.add_row('Connections', str(stats['active_connections']))
    return table

@app.command('show')
def show(verbose: bool=typer.Option(False, '--verbose', '-v', help='Show detailed information'), test: bool=typer.Option(False, '--test', '-t', help='Test API keys')) -> None:
    """Show configured API keys and their status."""
    from mcp_agent.config import get_settings
    if verbose:
        LOG_VERBOSE.set(True)
    verbose = LOG_VERBOSE.get()
    console.print('\n[bold cyan]🔑 API Key Status[/bold cyan]\n')
    settings = get_settings()
    table = Table(show_header=True, header_style='cyan')
    table.add_column('Provider', style='green')
    table.add_column('Status', justify='center')
    table.add_column('Source')
    table.add_column('Key (masked)')
    if verbose:
        table.add_column('Format')
    if test:
        table.add_column('Test', justify='center')
    for provider_key, config in PROVIDERS.items():
        env_var = config['env']
        provider_name = config['name']
        env_val = os.environ.get(env_var)
        provider_settings = getattr(settings, provider_key, None)
        cfg_val = getattr(provider_settings, 'api_key', None) if provider_settings else None
        active_key = cfg_val or env_val
        source = 'secrets' if cfg_val else 'env' if env_val else 'none'
        if active_key:
            valid, message = _validate_key(provider_key, active_key)
            if valid:
                status = '[green]✅[/green]'
            else:
                status = '[yellow]⚠️[/yellow]'
        else:
            status = '[red]❌[/red]'
        masked = _mask_key(active_key) if active_key else '-'
        row = [provider_name, status, source, masked]
        if verbose:
            row.append(config.get('format', 'N/A'))
        if test and active_key:
            import asyncio
            success, test_msg = asyncio.run(_test_key(provider_key, active_key))
            if success:
                row.append('[green]✅[/green]')
            else:
                row.append('[red]❌[/red]')
        elif test:
            row.append('-')
        table.add_row(*row)
    console.print(table)
    if verbose:
        additional_vars = []
        for provider_key, config in PROVIDERS.items():
            if 'additional_env' in config:
                for var, desc in config['additional_env'].items():
                    val = os.environ.get(var)
                    if val:
                        additional_vars.append(f'  • {var}: {_mask_key(val, 8)} ({desc})')
        if additional_vars:
            console.print('\n[bold]Additional Environment Variables:[/bold]')
            for var in additional_vars:
                console.print(var)
    console.print('\n[dim]Use [cyan]mcp-agent keys set <provider>[/cyan] to configure keys[/dim]')
    console.print('[dim]Use [cyan]mcp-agent keys test[/cyan] to validate all keys[/dim]')

def _validate_key(provider: str, key: str) -> Tuple[bool, str]:
    """Validate API key format for a provider."""
    if provider not in PROVIDERS:
        return (False, 'Unknown provider')
    config = PROVIDERS[provider]
    pattern = config.get('pattern')
    if not pattern:
        return (True, 'No validation available')
    if re.match(pattern, key):
        return (True, 'Valid format')
    else:
        return (False, f'Invalid format. Expected: {config.get('format', 'Unknown format')}')

def _mask_key(key: str, show_chars: int=4) -> str:
    """Mask an API key, showing only last few characters."""
    if not key:
        return ''
    if len(key) <= show_chars:
        return '***'
    return f'***{key[-show_chars:]}'

@app.command('set')
def set_key(provider: str=typer.Argument(..., help='Provider name'), key: Optional[str]=typer.Option(None, '--key', '-k', help='API key (will prompt if not provided)'), force: bool=typer.Option(False, '--force', '-f', help='Skip validation'), env_only: bool=typer.Option(False, '--env-only', help='Set in environment only, not secrets file')) -> None:
    """Set API key for a provider."""
    import yaml
    from mcp_agent.config import Settings
    if provider not in PROVIDERS:
        console.print(f'[red]Unknown provider: {provider}[/red]')
        console.print(f'Available providers: {', '.join(PROVIDERS.keys())}')
        raise typer.Exit(1)
    config = PROVIDERS[provider]
    provider_name = config['name']
    env_var = config['env']
    console.print(f'\n[bold]Setting {provider_name} API Key[/bold]\n')
    if not key:
        console.print(f'Format: {config.get('format', 'Any format')}')
        if config.get('docs'):
            console.print(f'Get your key at: [cyan]{config['docs']}[/cyan]')
        key = Prompt.ask(f'\n{provider_name} API key', password=True)
    if not key:
        console.print('[yellow]No key provided[/yellow]')
        raise typer.Exit(0)
    if not force:
        valid, message = _validate_key(provider, key)
        if not valid:
            console.print(f'[red]Validation failed: {message}[/red]')
            if not Confirm.ask('Continue anyway?', default=False):
                raise typer.Exit(1)
    os.environ[env_var] = key
    console.print(f'[green]✅[/green] Set {env_var} in environment')
    if 'additional_env' in config:
        console.print(f'\n[bold]{provider_name} requires additional configuration:[/bold]')
        for var, desc in config['additional_env'].items():
            current = os.environ.get(var, '')
            value = Prompt.ask(f'{desc} ({var})', default=current)
            if value:
                os.environ[var] = value
    if not env_only:
        sec_path = Settings.find_secrets()
        if not sec_path:
            sec_path = Path.cwd() / 'mcp_agent.secrets.yaml'
            data = {}
        else:
            try:
                data = yaml.safe_load(sec_path.read_text()) or {}
            except Exception:
                data = {}
        if provider not in data:
            data[provider] = {}
        data[provider]['api_key'] = key
        if 'additional_env' in config:
            for var, _ in config['additional_env'].items():
                val = os.environ.get(var)
                if val:
                    config_key = var.lower().replace(f'{provider.upper()}_', '').replace('_', '_')
                    data[provider][config_key] = val
        try:
            sec_path.write_text(yaml.safe_dump(data, sort_keys=False))
            console.print(f'[green]✅[/green] Saved to {sec_path}')
            try:
                import stat
                os.chmod(sec_path, stat.S_IRUSR | stat.S_IWUSR)
                console.print('[dim]Set secure permissions (600)[/dim]')
            except Exception:
                pass
        except Exception as e:
            console.print(f'[red]Failed to write secrets: {e}[/red]')
    if not force:
        console.print('\n[dim]Testing key...[/dim]')
        import asyncio
        success, message = asyncio.run(_test_key(provider, key))
        if success:
            console.print(f'[green]✅ {message}[/green]')
        else:
            console.print(f'[yellow]⚠️  {message}[/yellow]')
    console.print(f'\n[green bold]✅ {provider_name} key configured![/green bold]')

@app.command('test')
def test(provider: Optional[str]=typer.Argument(None, help='Provider to test (or all)'), verbose: bool=typer.Option(False, '--verbose', '-v', help='Show detailed results')) -> None:
    """Test API keys by making validation requests."""
    from mcp_agent.config import get_settings
    import asyncio
    console.print('\n[bold cyan]🧪 Testing API Keys[/bold cyan]\n')
    if verbose:
        LOG_VERBOSE.set(True)
    verbose = LOG_VERBOSE.get()
    settings = get_settings()
    if provider:
        if provider not in PROVIDERS:
            console.print(f'[red]Unknown provider: {provider}[/red]')
            raise typer.Exit(1)
        providers_to_test = [provider]
    else:
        providers_to_test = list(PROVIDERS.keys())
    results = []
    with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'), console=console) as progress:
        for provider_key in providers_to_test:
            config = PROVIDERS[provider_key]
            provider_name = config['name']
            task = progress.add_task(f'Testing {provider_name}...', total=None)
            env_var = config['env']
            env_val = os.environ.get(env_var)
            provider_settings = getattr(settings, provider_key, None)
            cfg_val = getattr(provider_settings, 'api_key', None) if provider_settings else None
            active_key = cfg_val or env_val
            if not active_key:
                progress.update(task, description=f'[yellow]⏭️  {provider_name}: Not configured[/yellow]')
                results.append((provider_name, 'Not configured', None))
                continue
            valid, format_msg = _validate_key(provider_key, active_key)
            success, test_msg = asyncio.run(_test_key(provider_key, active_key))
            if success:
                progress.update(task, description=f'[green]✅ {provider_name}: Valid[/green]')
                results.append((provider_name, 'Valid', test_msg))
            else:
                progress.update(task, description=f'[red]❌ {provider_name}: {test_msg}[/red]')
                results.append((provider_name, 'Invalid', test_msg))
    console.print('\n[bold]Test Results:[/bold]\n')
    summary_table = Table(show_header=True, header_style='cyan')
    summary_table.add_column('Provider', style='green')
    summary_table.add_column('Status', justify='center')
    if verbose:
        summary_table.add_column('Details')
    for provider_name, status, details in results:
        if status == 'Valid':
            status_icon = '[green]✅ Valid[/green]'
        elif status == 'Invalid':
            status_icon = '[red]❌ Invalid[/red]'
        else:
            status_icon = '[yellow]⏭️  Skipped[/yellow]'
        row = [provider_name, status_icon]
        if verbose and details:
            row.append(details)
        summary_table.add_row(*row)
    console.print(summary_table)
    valid_count = sum((1 for _, status, _ in results if status == 'Valid'))
    invalid_count = sum((1 for _, status, _ in results if status == 'Invalid'))
    skipped_count = sum((1 for _, status, _ in results if status == 'Not configured'))
    console.print(f'\n[bold]Summary:[/bold] {valid_count} valid, {invalid_count} invalid, {skipped_count} not configured')
    if invalid_count > 0:
        console.print('\n[dim]Use [cyan]mcp-agent keys set <provider>[/cyan] to fix invalid keys[/dim]')

@app.command('rotate')
def rotate(provider: str=typer.Argument(..., help='Provider name'), backup: bool=typer.Option(True, '--backup/--no-backup', help='Backup old key')) -> None:
    """Rotate API key for a provider (backup old, set new)."""
    from mcp_agent.config import get_settings
    if provider not in PROVIDERS:
        console.print(f'[red]Unknown provider: {provider}[/red]')
        raise typer.Exit(1)
    config = PROVIDERS[provider]
    provider_name = config['name']
    console.print(f'\n[bold cyan]🔄 Rotating {provider_name} API Key[/bold cyan]\n')
    settings = get_settings()
    provider_settings = getattr(settings, provider, None)
    old_key = getattr(provider_settings, 'api_key', None) if provider_settings else None
    if not old_key:
        old_key = os.environ.get(config['env'])
    if old_key and backup:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = Path.cwd() / f'.mcp-agent/backup_{provider}_{timestamp}.txt'
        backup_file.parent.mkdir(exist_ok=True, parents=True)
        backup_data = {'provider': provider, 'timestamp': timestamp, 'key': old_key, 'masked': _mask_key(old_key, 8)}
        backup_file.write_text(json.dumps(backup_data, indent=2))
        console.print(f'[green]✅[/green] Backed up old key to {backup_file}')
        try:
            import stat
            os.chmod(backup_file, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass
    console.print(f'\nEnter new {provider_name} API key')
    console.print(f'Format: {config.get('format', 'Any format')}')
    new_key = Prompt.ask('New API key', password=True)
    if not new_key:
        console.print('[yellow]No key provided[/yellow]')
        raise typer.Exit(0)
    set_key(provider=provider, key=new_key, force=False, env_only=False)
    console.print(f'\n[green bold]✅ {provider_name} key rotated successfully![/green bold]')
    if backup and old_key:
        console.print(f'[dim]Old key backed up to .mcp-agent/backup_{provider}_{timestamp}.txt[/dim]')

@app.command('export')
def export(output: Path=typer.Option(Path('keys.env'), '--output', '-o', help='Output file'), format: str=typer.Option('env', '--format', '-f', help='Format: env|json|yaml')) -> None:
    """Export all configured keys to a file."""
    from mcp_agent.config import get_settings
    console.print('\n[bold]Exporting API Keys[/bold]\n')
    settings = get_settings()
    keys = {}
    for provider_key, config in PROVIDERS.items():
        env_var = config['env']
        provider_settings = getattr(settings, provider_key, None)
        cfg_val = getattr(provider_settings, 'api_key', None) if provider_settings else None
        env_val = os.environ.get(env_var)
        active_key = cfg_val or env_val
        if active_key:
            keys[env_var] = active_key
            if 'additional_env' in config:
                for var in config['additional_env']:
                    val = os.environ.get(var)
                    if val:
                        keys[var] = val
    if not keys:
        console.print('[yellow]No keys to export[/yellow]')
        raise typer.Exit(0)
    if format == 'env':
        content = '\n'.join((f'{k}="{v}"' for k, v in keys.items()))
    elif format == 'json':
        content = json.dumps(keys, indent=2)
    elif format == 'yaml':
        content = yaml.safe_dump(keys, sort_keys=False)
    else:
        console.print(f'[red]Unknown format: {format}[/red]')
        raise typer.Exit(1)
    output.write_text(content)
    console.print(f'[green]✅[/green] Exported {len(keys)} keys to {output}')
    try:
        import stat
        os.chmod(output, stat.S_IRUSR | stat.S_IWUSR)
        console.print('[dim]Set secure permissions (600)[/dim]')
    except Exception:
        pass
    console.print('\n[yellow]⚠️  Warning: This file contains sensitive API keys![/yellow]')
    console.print("[dim]Keep it secure and don't commit to version control[/dim]")

@app.callback(invoke_without_command=True, no_args_is_help=False)
def go(ctx: typer.Context, name: str=typer.Option('mcp-agent', '--name'), instruction: Optional[str]=typer.Option(None, '--instruction', '-i'), config_path: Optional[str]=typer.Option(None, '--config-path', '-c'), servers: Optional[str]=typer.Option(None, '--servers'), urls: Optional[str]=typer.Option(None, '--url'), auth: Optional[str]=typer.Option(None, '--auth'), model: Optional[str]=typer.Option(None, '--model', '--models'), message: Optional[str]=typer.Option(None, '--message', '-m'), prompt_file: Optional[Path]=typer.Option(None, '--prompt-file', '-p'), npx: Optional[str]=typer.Option(None, '--npx'), uvx: Optional[str]=typer.Option(None, '--uvx'), stdio: Optional[str]=typer.Option(None, '--stdio'), script: Optional[Path]=typer.Option(None, '--script')) -> None:
    script = detect_default_script(script)
    server_list = servers.split(',') if servers else None
    url_servers = None
    if urls:
        try:
            parsed = parse_server_urls(urls, auth)
            url_servers = generate_server_configs(parsed)
            if url_servers and (not server_list):
                server_list = list(url_servers.keys())
            elif url_servers and server_list:
                server_list.extend(list(url_servers.keys()))
        except ValueError as e:
            typer.secho(f'Error parsing URLs: {e}', err=True, fg=typer.colors.RED)
            raise typer.Exit(6)
    stdio_cmds: List[str] = []
    if npx:
        stdio_cmds.append(f'npx {npx}')
    if uvx:
        stdio_cmds.append(f'uvx {uvx}')
    if stdio:
        stdio_cmds.append(stdio)
    stdio_servers = _parse_stdio_commands(stdio_cmds)
    if stdio_servers:
        if not server_list:
            server_list = list(stdio_servers.keys())
        else:
            server_list.extend(list(stdio_servers.keys()))
    resolved_server_list = select_servers_from_config(','.join(server_list) if server_list else None, url_servers, stdio_servers)
    if model and ',' in model:
        models = [m.strip() for m in model.split(',') if m.strip()]
        results: list[tuple[str, str | Exception]] = []
        for m in models:
            try:
                asyncio.run(_run_agent(app_script=script, server_list=resolved_server_list, model=m, message=message, prompt_file=prompt_file, url_servers=url_servers, stdio_servers=stdio_servers, agent_name=name, instruction=instruction))
            except Exception as e:
                results.append((m, e))
        return
    try:
        asyncio.run(_run_agent(app_script=script, server_list=resolved_server_list, model=model, message=message, prompt_file=prompt_file, url_servers=url_servers, stdio_servers=stdio_servers, agent_name=name, instruction=instruction))
    except KeyboardInterrupt:
        pass

def _parse_stdio_commands(cmds: List[str] | None) -> Dict[str, Dict[str, str]] | None:
    if not cmds:
        return None
    servers: Dict[str, Dict[str, str]] = {}
    for i, cmd in enumerate(cmds):
        parts = shlex.split(cmd)
        if not parts:
            continue
        command, args = (parts[0], parts[1:])
        name = command.replace('/', '_').replace('@', '').replace('.', '_')
        if len(cmds) > 1:
            name = f'{name}_{i + 1}'
        servers[name] = {'transport': 'stdio', 'command': command, 'args': args}
    return servers

@app.callback(invoke_without_command=True, no_args_is_help=False)
def chat(name: Optional[str]=typer.Option(None, '--name'), model: Optional[str]=typer.Option(None, '--model'), models: Optional[str]=typer.Option(None, '--models'), message: Optional[str]=typer.Option(None, '--message', '-m'), prompt_file: Optional[Path]=typer.Option(None, '--prompt-file', '-p'), servers_csv: Optional[str]=typer.Option(None, '--servers'), urls: Optional[str]=typer.Option(None, '--url'), auth: Optional[str]=typer.Option(None, '--auth'), npx: Optional[str]=typer.Option(None, '--npx'), uvx: Optional[str]=typer.Option(None, '--uvx'), stdio: Optional[str]=typer.Option(None, '--stdio'), script: Optional[Path]=typer.Option(None, '--script'), list_servers: bool=typer.Option(False, '--list-servers'), list_tools: bool=typer.Option(False, '--list-tools'), list_resources: bool=typer.Option(False, '--list-resources'), server: Optional[str]=typer.Option(None, '--server', help='Filter to a single server')) -> None:
    script = detect_default_script(script)
    server_list = servers_csv.split(',') if servers_csv else None
    url_servers = None
    if urls:
        try:
            parsed = parse_server_urls(urls, auth)
            url_servers = generate_server_configs(parsed)
            if url_servers and (not server_list):
                server_list = list(url_servers.keys())
            elif url_servers and server_list:
                server_list.extend(list(url_servers.keys()))
        except ValueError as e:
            typer.secho(f'Error parsing URLs: {e}', err=True, fg=typer.colors.RED)
            raise typer.Exit(6)
    stdio_servers = None
    stdio_cmds: List[str] = []
    if npx:
        stdio_cmds.append(f'npx {npx}')
    if uvx:
        stdio_cmds.append(f'uvx {uvx}')
    if stdio:
        stdio_cmds.append(stdio)
    if stdio_cmds:
        from .go import _parse_stdio_commands
        stdio_servers = _parse_stdio_commands(stdio_cmds)
        if stdio_servers:
            if not server_list:
                server_list = list(stdio_servers.keys())
            else:
                server_list.extend(list(stdio_servers.keys()))
    resolved_server_list = select_servers_from_config(servers_csv, url_servers, stdio_servers)
    if list_servers or list_tools or list_resources:
        try:

            async def _list():
                settings = get_settings()
                if settings.logger:
                    settings.logger.progress_display = False
                app_obj = load_user_app(script, settings_override=settings)
                await app_obj.initialize()
                attach_url_servers(app_obj, url_servers)
                attach_stdio_servers(app_obj, stdio_servers)
                async with app_obj.run():
                    cfg = app_obj.context.config
                    all_servers = list((cfg.mcp.servers or {}).keys()) if cfg.mcp else []
                    target_servers = [server] if server else all_servers
                    if list_servers:
                        for s in target_servers:
                            console.print(s)
                        if not (list_tools or list_resources):
                            return
                    agent = Agent(name='chat-lister', instruction='You list tools and resources', server_names=resolved_server_list or target_servers, context=app_obj.context)
                    async with agent:
                        if list_tools:
                            res = await agent.list_tools(server_name=server) if server else await agent.list_tools()
                            for t in res.tools:
                                console.print(t.name)
                        if list_resources:
                            res = await agent.list_resources(server_name=server) if server else await agent.list_resources()
                            for r in getattr(res, 'resources', []):
                                try:
                                    console.print(r.uri)
                                except Exception:
                                    console.print(str(getattr(r, 'uri', '')))
            asyncio.run(_list())
        except KeyboardInterrupt:
            pass
        return
    if models:
        model_list = [x.strip() for x in models.split(',') if x.strip()]
        if not message and (not prompt_file) and (not (list_servers or list_tools or list_resources)):

            async def _parallel_repl():
                settings = get_settings()
                if settings.logger:
                    settings.logger.progress_display = False
                app_obj = load_user_app(script, settings_override=settings)
                await app_obj.initialize()
                attach_url_servers(app_obj, url_servers)
                attach_stdio_servers(app_obj, stdio_servers)
                async with app_obj.run():
                    llms = []
                    for m in model_list:
                        provider = None
                        if ':' in m:
                            provider = m.split(':', 1)[0]
                        elif '.' in m:
                            prov_guess = m.split('.', 1)[0].lower()
                            if prov_guess in {'openai', 'anthropic', 'azure', 'google', 'bedrock', 'ollama'}:
                                provider = prov_guess
                        llm = create_llm(agent_name=m, server_names=resolved_server_list or [], provider=provider or 'openai', model=m, context=app_obj.context)
                        llms.append(llm)
                    console.print('Interactive parallel chat. Commands: /help, /servers, /tools [server], /resources [server], /models, /clear, /usage, /quit, /exit')
                    from mcp_agent.agents.agent import Agent as _Agent
                    while True:
                        try:
                            inp = input('> ')
                        except (EOFError, KeyboardInterrupt):
                            break
                        if not inp:
                            continue
                        if inp.startswith('/quit') or inp.startswith('/exit'):
                            break
                        if inp.startswith('/help'):
                            console.print('/servers, /tools [server], /resources [server], /models, /clear, /usage, /quit, /exit')
                            continue
                        if inp.startswith('/clear'):
                            console.clear()
                            continue
                        if inp.startswith('/models'):
                            console.print(f'\nActive models ({len(llms)}):')
                            for llm in llms:
                                console.print(f'  - {llm.name}')
                            continue
                        if inp.startswith('/servers'):
                            cfg = app_obj.context.config
                            svrs = list((cfg.mcp.servers or {}).keys()) if cfg.mcp else []
                            for s in svrs:
                                console.print(s)
                            continue
                        if inp.startswith('/tools'):
                            parts = inp.split()
                            srv = parts[1] if len(parts) > 1 else None
                            ag = _Agent(name='chat-lister', instruction='list tools', server_names=[srv] if srv else resolved_server_list or [], context=app_obj.context)
                            async with ag:
                                res = await ag.list_tools(server_name=srv) if srv else await ag.list_tools()
                                for t in res.tools:
                                    console.print(t.name)
                            continue
                        if inp.startswith('/resources'):
                            parts = inp.split()
                            srv = parts[1] if len(parts) > 1 else None
                            ag = _Agent(name='chat-lister', instruction='list resources', server_names=[srv] if srv else resolved_server_list or [], context=app_obj.context)
                            async with ag:
                                res = await ag.list_resources(server_name=srv) if srv else await ag.list_resources()
                                for r in getattr(res, 'resources', []):
                                    try:
                                        console.print(r.uri)
                                    except Exception:
                                        console.print(str(getattr(r, 'uri', '')))
                            continue
                        if inp.startswith('/usage'):
                            try:
                                from mcp_agent.cli.utils.display import TokenUsageDisplay
                                tc = getattr(app_obj.context, 'token_counter', None)
                                if tc:
                                    summary = await tc.get_summary()
                                    if summary:
                                        display = TokenUsageDisplay()
                                        summary_dict = summary.model_dump() if hasattr(summary, 'model_dump') else summary
                                        display.show_summary(summary_dict)
                                    else:
                                        console.print('(no usage data)')
                                else:
                                    console.print('(no token counter)')
                            except Exception as e:
                                console.print(f'(usage error: {e})')
                            continue
                        try:
                            from mcp_agent.cli.utils.display import ParallelResultsDisplay

                            async def _gen(llm_instance):
                                try:
                                    return (llm_instance.name, await llm_instance.generate_str(inp))
                                except Exception as e:
                                    return (llm_instance.name, f'ERROR: {e}')
                            results = await asyncio.gather(*[_gen(item) for item in llms])
                            display = ParallelResultsDisplay()
                            display.show_results(results)
                        except Exception as e:
                            console.print(f'ERROR: {e}')
            asyncio.run(_parallel_repl())
            return
        results = []
        for m in model_list:
            try:
                out = asyncio.run(_run_single_model(script=script, servers=resolved_server_list, url_servers=url_servers, stdio_servers=stdio_servers, model=m, message=message, prompt_file=prompt_file, agent_name=name or m))
                results.append((m, out))
            except Exception as e:
                results.append((m, f'ERROR: {e}'))
        for m, out in results:
            console.print(f'\n[bold]{m}[/bold]:\n{out}')
        return
    try:
        if not message and (not prompt_file) and (not models) and (not (list_servers or list_tools or list_resources)):

            async def _repl():
                settings = get_settings()
                if settings.logger:
                    settings.logger.progress_display = False
                app_obj = load_user_app(script, settings_override=settings)
                await app_obj.initialize()
                attach_url_servers(app_obj, url_servers)
                attach_stdio_servers(app_obj, stdio_servers)
                async with app_obj.run():
                    provider = None
                    model_id = model
                    if model_id and ':' not in model_id and ('.' in model_id):
                        maybe_provider = model_id.split('.', 1)[0].lower()
                        if maybe_provider in {'openai', 'anthropic', 'azure', 'google', 'bedrock', 'ollama'}:
                            provider = maybe_provider
                    if model_id and ':' in model_id:
                        provider = model_id.split(':', 1)[0]
                    llm = create_llm(agent_name=name or 'chat', server_names=resolved_server_list or [], provider=provider or 'openai', model=model_id, context=app_obj.context)
                    console.print('Interactive chat. Commands: /help, /servers, /tools [server], /resources [server], /models, /prompt <name> [args-json], /apply <file>, /attach <server> <resource-uri>, /history [clear], /save <file>, /clear, /usage, /quit, /exit, /model <name>')
                    last_output: str | None = None
                    attachments: list[str] = []
                    while True:
                        try:
                            inp = input('> ')
                        except (EOFError, KeyboardInterrupt):
                            break
                        if not inp:
                            continue
                        if inp.startswith('/quit') or inp.startswith('/exit'):
                            break
                        if inp.startswith('/help'):
                            console.print('/servers, /tools [server], /resources [server], /models, /prompt <name> [args-json], /apply <file>, /attach <server> <resource-uri>, /history [clear], /save <file>, /clear, /usage, /quit, /exit')
                            continue
                        if inp.startswith('/clear'):
                            console.clear()
                            continue
                        if inp.startswith('/models'):
                            from mcp_agent.workflows.llm.llm_selector import load_default_models
                            models = load_default_models()
                            console.print('\n[bold]Available models:[/bold]')
                            current_model_str = str(model_id) if model_id else 'default'
                            console.print(f'Current: {current_model_str}\n')
                            for m in models[:15]:
                                console.print(f'  {m.provider}.{m.name}')
                            if len(models) > 15:
                                console.print(f'  ... and {len(models) - 15} more')
                            continue
                        if inp.startswith('/model '):
                            try:
                                new_model = inp.split(' ', 1)[1].strip()
                                if not new_model:
                                    console.print('Usage: /model <provider.model or provider:model>')
                                    continue
                                model_id = new_model
                                prov = None
                                if ':' in new_model:
                                    prov = new_model.split(':', 1)[0]
                                elif '.' in new_model:
                                    prov = new_model.split('.', 1)[0]
                                llm_local = create_llm(agent_name=name or 'chat', server_names=resolved_server_list or [], provider=prov or 'openai', model=model_id, context=app_obj.context)
                                llm = llm_local
                                console.print(f'Switched model to: {model_id}')
                            except Exception as e:
                                console.print(f'/model error: {e}')
                            continue
                        if inp.startswith('/servers'):
                            cfg = app_obj.context.config
                            servers = list((cfg.mcp.servers or {}).keys()) if cfg.mcp else []
                            for s in servers:
                                console.print(s)
                            continue
                        if inp.startswith('/tools'):
                            from mcp_agent.cli.utils.display import format_tool_list
                            parts = inp.split()
                            srv = parts[1] if len(parts) > 1 else None
                            ag = Agent(name='chat-lister', instruction='list tools', server_names=[srv] if srv else resolved_server_list or [], context=app_obj.context)
                            async with ag:
                                res = await ag.list_tools(server_name=srv) if srv else await ag.list_tools()
                                format_tool_list(res.tools, server_name=srv)
                            continue
                        if inp.startswith('/resources'):
                            from mcp_agent.cli.utils.display import format_resource_list
                            parts = inp.split()
                            srv = parts[1] if len(parts) > 1 else None
                            ag = Agent(name='chat-lister', instruction='list resources', server_names=[srv] if srv else resolved_server_list or [], context=app_obj.context)
                            async with ag:
                                res = await ag.list_resources(server_name=srv) if srv else await ag.list_resources()
                                format_resource_list(getattr(res, 'resources', []), server_name=srv)
                            continue
                        if inp.startswith('/prompt'):
                            try:
                                parts = inp.split(maxsplit=2)
                                if len(parts) < 2:
                                    console.print('Usage: /prompt <name> [args-json]')
                                    continue
                                prompt_name = parts[1]
                                args_json = parts[2] if len(parts) > 2 else None
                                arguments = None
                                if args_json:
                                    import json as _json
                                    try:
                                        arguments = _json.loads(args_json)
                                    except Exception as e:
                                        console.print(f'Invalid JSON: {e}')
                                        continue
                                ag = llm.agent
                                prompt_msgs = await ag.create_prompt(prompt_name=prompt_name, arguments=arguments, server_names=resolved_server_list or [])
                                out = await llm.generate_str(prompt_msgs)
                                last_output = out
                                console.print(out)
                            except Exception as e:
                                console.print(f'/prompt error: {e}')
                            continue
                        if inp.startswith('/apply'):
                            parts = inp.split(maxsplit=1)
                            if len(parts) < 2:
                                console.print('Usage: /apply <file>')
                                continue
                            from pathlib import Path as _Path
                            p = _Path(parts[1]).expanduser()
                            if not p.exists():
                                console.print('File not found')
                                continue
                            text = p.read_text(encoding='utf-8')
                            try:
                                import json as _json
                                js = _json.loads(text)
                                out = await llm.generate_str(js)
                            except Exception:
                                out = await llm.generate_str(text)
                            last_output = out
                            console.print(out)
                            continue
                        if inp.startswith('/attach'):
                            parts = inp.split(maxsplit=2)
                            if len(parts) < 3:
                                console.print('Usage: /attach <server> <resource-uri>')
                                continue
                            srv, uri = (parts[1], parts[2])
                            try:
                                res = await llm.read_resource(uri=uri, server_name=srv)
                                content_text = None
                                try:
                                    from mcp_agent.utils.content_utils import get_text
                                    if getattr(res, 'contents', None):
                                        for c in res.contents:
                                            try:
                                                content_text = get_text(c)
                                                if content_text:
                                                    break
                                            except Exception:
                                                continue
                                except Exception:
                                    pass
                                if not content_text:
                                    content_text = str(res)
                                attachments.append(content_text)
                                console.print(f'Attached resource; size={len(content_text)} chars')
                            except Exception as e:
                                console.print(f'/attach error: {e}')
                            continue
                        if inp.startswith('/history'):
                            parts = inp.split()
                            if len(parts) > 1 and parts[1] == 'clear':
                                try:
                                    llm.history.clear()
                                    console.print('History cleared')
                                except Exception:
                                    console.print('Could not clear history')
                            else:
                                try:
                                    hist = llm.history.get()
                                    console.print(f'{len(hist)} messages in memory')
                                except Exception:
                                    console.print('(no history)')
                            continue
                        if inp.startswith('/save'):
                            parts = inp.split(maxsplit=1)
                            if len(parts) < 2:
                                console.print('Usage: /save <file>')
                                continue
                            if last_output is None:
                                console.print('No output to save')
                                continue
                            from pathlib import Path as _Path
                            _Path(parts[1]).expanduser().write_text(last_output, encoding='utf-8')
                            console.print('Saved')
                            continue
                        if inp.startswith('/usage'):
                            try:
                                from mcp_agent.cli.utils.display import TokenUsageDisplay
                                tc = getattr(app_obj.context, 'token_counter', None)
                                if tc:
                                    summary = await tc.get_summary()
                                    if summary:
                                        display = TokenUsageDisplay()
                                        summary_dict = summary.model_dump() if hasattr(summary, 'model_dump') else summary
                                        display.show_summary(summary_dict)
                                    else:
                                        console.print('(no usage data)')
                                else:
                                    console.print('(no token counter)')
                            except Exception as e:
                                console.print(f'(usage error: {e})')
                            continue
                        try:
                            payload = inp
                            if attachments:
                                prefix = '\n\n'.join(attachments) + '\n\n'
                                payload = prefix + inp
                                attachments.clear()
                            out = await llm.generate_str(payload)
                            last_output = out
                            console.print(out)
                        except Exception as e:
                            console.print(f'ERROR: {e}')
            asyncio.run(_repl())
        else:
            out = asyncio.run(_run_single_model(script=script, servers=resolved_server_list, url_servers=url_servers, stdio_servers=stdio_servers, model=model, message=message, prompt_file=prompt_file, agent_name=name or 'chat'))
            console.print(out)
    except KeyboardInterrupt:
        pass

def _preflight_ok() -> bool:
    settings = get_settings()
    ok = True
    servers = (settings.mcp.servers if settings.mcp else {}) or {}
    for name, s in servers.items():
        if s.transport == 'stdio' and s.command and (not shutil.which(s.command)):
            console.print(f"[yellow]Missing command for server '{name}': {s.command}[/yellow]")
            ok = False
    return ok

@app.callback(invoke_without_command=True)
def dev(script: Path=typer.Option(None, '--script')) -> None:
    """Run the user's app script with optional live reload and preflight checks."""

    def _preflight_ok() -> bool:
        settings = get_settings()
        ok = True
        servers = (settings.mcp.servers if settings.mcp else {}) or {}
        for name, s in servers.items():
            if s.transport == 'stdio' and s.command and (not shutil.which(s.command)):
                console.print(f"[yellow]Missing command for server '{name}': {s.command}[/yellow]")
                ok = False
        return ok

    def _run_script() -> subprocess.Popen:
        """Run the script as a subprocess."""
        console.print(f'Running {script}')
        return subprocess.Popen([sys.executable, str(script)], stdout=None, stderr=None, stdin=None)
    script = detect_default_script(script)
    _ = _preflight_ok()
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        import time

        class _Handler(FileSystemEventHandler):

            def __init__(self):
                self.touched = False

            def on_modified(self, event):
                if not event.is_directory:
                    self.touched = True

            def on_created(self, event):
                if not event.is_directory:
                    self.touched = True
        handler = _Handler()
        observer = Observer()
        observer.schedule(handler, path=str(script.parent), recursive=True)
        observer.start()
        console.print('Live reload enabled (watchdog)')
        process = _run_script()
        try:
            while True:
                time.sleep(0.5)
                if process.poll() is not None:
                    console.print(f'[red]Process exited with code {process.returncode}[/red]')
                    break
                if handler.touched:
                    handler.touched = False
                    console.print('Change detected. Restarting...')
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    process = _run_script()
        except KeyboardInterrupt:
            console.print('\n[yellow]Stopping...[/yellow]')
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            observer.stop()
            observer.join()
    except ImportError:
        console.print('[yellow]Watchdog not installed. Running without live reload.[/yellow]')
        process = _run_script()
        try:
            process.wait()
        except KeyboardInterrupt:
            console.print('\n[yellow]Stopping...[/yellow]')
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

def _run_script() -> subprocess.Popen:
    """Run the script as a subprocess."""
    console.print(f'Running {script}')
    return subprocess.Popen([sys.executable, str(script)], stdout=None, stderr=None, stdin=None)

@app.command('list')
def list_servers(available: bool=typer.Option(False, '--available', '-a', help='Show only available servers'), category: Optional[str]=typer.Option(None, '--category', '-c', help='Filter by category')) -> None:
    """List configured servers."""
    settings = get_settings()
    servers = (settings.mcp.servers if settings.mcp else {}) or {}
    if not servers:
        console.print('[yellow]No servers configured[/yellow]')
        console.print('\n[dim]Hint: Use [cyan]mcp-agent server add recipe <name>[/cyan] to add servers[/dim]')
        console.print('[dim]Or: [cyan]mcp-agent server recipes[/cyan] to see available recipes[/dim]')
        return
    table = Table(title='Configured Servers', show_header=True, header_style='cyan')
    table.add_column('Name', style='green')
    table.add_column('Transport')
    table.add_column('Target')
    table.add_column('Status', justify='center')
    for name, s in servers.items():
        target = s.url or s.command or ''
        if s.args and s.command:
            target = f'{s.command} {' '.join(s.args[:2])}...'
        status = '❓'
        if s.transport == 'stdio' and s.command:
            if _check_command_available(s.command.split()[0]):
                status = '✅'
            else:
                status = '❌'
        elif s.transport in ['http', 'sse'] and s.url:
            status = '🌐'
        if not available or status in ['✅', '🌐']:
            table.add_row(name, s.transport, target[:50], status)
    console.print(table)

def _check_command_available(cmd: str) -> bool:
    """Check if a command is available in PATH."""
    import shutil
    return shutil.which(cmd) is not None

@app.callback(invoke_without_command=True)
def build(check_only: bool=typer.Option(False, '--check-only', help='Run checks without creating manifest'), fix: bool=typer.Option(False, '--fix', help='Attempt to fix minor issues'), verbose: bool=typer.Option(False, '--verbose', '-v', help='Show detailed output'), output: Optional[Path]=typer.Option(None, '--output', '-o', help='Output directory for manifest')) -> None:
    """Run comprehensive preflight checks and generate build manifest."""
    if verbose:
        LOG_VERBOSE.set(True)
    verbose = LOG_VERBOSE.get()
    console.print('\n[bold cyan]🔍 MCP-Agent Build Preflight Checks[/bold cyan]\n')
    with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'), console=console) as progress:
        task = progress.add_task('Running preflight checks...', total=None)
        settings = get_settings()
        ok = True
        from datetime import datetime, timezone
        report = {'timestamp': datetime.now(timezone.utc).isoformat(), 'python_version': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}', 'providers': {}, 'servers': {}, 'environment': {}, 'files': {}, 'dependencies': {}, 'network': {}, 'warnings': []}
        progress.update(task, description='Checking provider configurations...')
        provs = [('openai', getattr(settings, 'openai', None), 'api_key'), ('anthropic', getattr(settings, 'anthropic', None), 'api_key'), ('google', getattr(settings, 'google', None), 'api_key'), ('azure', getattr(settings, 'azure', None), 'api_key'), ('bedrock', getattr(settings, 'bedrock', None), 'aws_access_key_id')]
        for name, obj, keyfield in provs:
            has_config = bool(getattr(obj, keyfield, None)) if obj else False
            has_env = bool(os.getenv(f'{name.upper()}_API_KEY')) or (name == 'bedrock' and bool(os.getenv('AWS_ACCESS_KEY_ID')))
            report['providers'][name] = {'configured': has_config, 'env_var': has_env, 'available': has_config or has_env}
        progress.update(task, description='Checking environment variables...')
        report['environment'] = _check_environment_vars(settings)
        progress.update(task, description='Checking file permissions...')
        config_file = Path('mcp_agent.config.yaml')
        secrets_file = Path('mcp_agent.secrets.yaml')
        report['files']['config'] = _check_file_permissions(config_file)
        report['files']['secrets'] = _check_file_permissions(secrets_file)
        if secrets_file.exists() and (not report['files']['secrets']['secure']):
            report['warnings'].append(f'Secrets file has unsafe permissions: {report['files']['secrets']['permissions']}')
        progress.update(task, description='Checking MCP servers...')
        servers = (settings.mcp.servers if settings.mcp else {}) or {}
        for name, s in servers.items():
            status = {'transport': s.transport}
            if s.transport == 'stdio':
                status['command'] = s.command
                found, version = _check_command(s.command)
                status['command_found'] = found
                status['version'] = version
                if not found:
                    ok = False
                    report['warnings'].append(f"Server '{name}' command not found: {s.command}")
            else:
                status['url'] = s.url
                reachable, response = _check_url(s.url)
                status['reachable'] = reachable
                status['response_time'] = response
                if not reachable and verbose:
                    report['warnings'].append(f"Server '{name}' not reachable: {response}")
            if s.env:
                status['env_vars'] = {}
                for key in s.env.keys():
                    status['env_vars'][key] = bool(os.getenv(key))
            report['servers'][name] = status
        if verbose:
            progress.update(task, description='Checking dependencies...')
            report['dependencies'] = _check_dependencies()
            for pkg, info in report['dependencies'].items():
                if pkg != 'python' and (not info.get('installed')):
                    report['warnings'].append(f'Missing dependency: {pkg}')
        if verbose:
            progress.update(task, description='Checking network connectivity...')
            report['network'] = _check_network_connectivity()
        progress.update(task, description='Validating configuration...')
        schema_warnings = _validate_config_schema(settings)
        report['warnings'].extend(schema_warnings)
    console.print('\n[bold]Preflight Check Results[/bold]\n')
    provider_table = Table(title='Provider Status', show_header=True, header_style='cyan')
    provider_table.add_column('Provider', style='green')
    provider_table.add_column('Config', justify='center')
    provider_table.add_column('Env Var', justify='center')
    provider_table.add_column('Status', justify='center')
    for name, info in report['providers'].items():
        config = '✅' if info['configured'] else '❌'
        env = '✅' if info['env_var'] else '❌'
        status = '[green]Ready[/green]' if info['available'] else '[yellow]Not configured[/yellow]'
        provider_table.add_row(name.capitalize(), config, env, status)
    console.print(provider_table)
    console.print()
    if report['servers']:
        server_table = Table(title='MCP Server Status', show_header=True, header_style='cyan')
        server_table.add_column('Server', style='green')
        server_table.add_column('Transport')
        server_table.add_column('Target')
        server_table.add_column('Status', justify='center')
        for name, info in report['servers'].items():
            if info['transport'] == 'stdio':
                target = info.get('command', 'N/A')
                if info['command_found']:
                    status = f'[green]✅ {info['version']}[/green]'
                else:
                    status = '[red]❌ Not found[/red]'
            else:
                target = info.get('url', 'N/A')[:40]
                if info.get('reachable'):
                    status = f'[green]✅ {info['response_time']}[/green]'
                else:
                    status = f'[yellow]⚠️  {info.get('response_time', 'Unknown')}[/yellow]'
            server_table.add_row(name, info['transport'], target, status)
        console.print(server_table)
        console.print()
    else:
        console.print('[yellow]No MCP servers found in configuration[/yellow]')
        console.print()
    if report['warnings']:
        console.print(Panel('\n'.join((f'• {w}' for w in report['warnings'])), title='[yellow]Warnings[/yellow]', border_style='yellow'))
        console.print()
    if not check_only:
        out_dir = output or Path('.mcp-agent')
        out_dir.mkdir(exist_ok=True, parents=True)
        manifest = out_dir / 'manifest.json'
        manifest.write_text(json.dumps(report, indent=2))
        console.print(f'[green]✅[/green] Wrote manifest: [cyan]{manifest}[/cyan]')
    if fix and (not ok):
        console.print('\n[bold yellow]🔧 Fix Suggestions:[/bold yellow]\n')
        for name, st in report['servers'].items():
            if st.get('transport') == 'stdio' and (not st.get('command_found')):
                cmd = st.get('command', '')
                if 'npx' in cmd:
                    console.print('• Install npm: [cyan]brew install node[/cyan] (macOS) or [cyan]apt install nodejs[/cyan]')
                elif 'uvx' in cmd:
                    console.print('• Install uv: [cyan]pip install uv[/cyan] or [cyan]brew install uv[/cyan]')
                else:
                    console.print(f"• Ensure '{cmd}' is installed and on PATH")
        if not any((p['available'] for p in report['providers'].values())):
            console.print('• Add API keys to mcp_agent.secrets.yaml or set environment variables')
    if ok:
        console.print('\n[green bold]✅ Preflight checks passed![/green bold]')
    else:
        console.print('\n[red bold]❌ Preflight checks failed[/red bold]')
        if not check_only:
            raise typer.Exit(1)

def _check_environment_vars(settings: Settings) -> Dict[str, Any]:
    """Check for environment variables that might override settings."""
    env_vars = {'OPENAI_API_KEY': bool(os.getenv('OPENAI_API_KEY')), 'ANTHROPIC_API_KEY': bool(os.getenv('ANTHROPIC_API_KEY')), 'GOOGLE_API_KEY': bool(os.getenv('GOOGLE_API_KEY')), 'AZURE_API_KEY': bool(os.getenv('AZURE_API_KEY')), 'AWS_ACCESS_KEY_ID': bool(os.getenv('AWS_ACCESS_KEY_ID')), 'AWS_SECRET_ACCESS_KEY': bool(os.getenv('AWS_SECRET_ACCESS_KEY'))}
    return env_vars

def _check_file_permissions(path: Path) -> Dict[str, Any]:
    """Check file permissions for sensitive files."""
    result = {'exists': path.exists(), 'readable': False, 'writable': False, 'permissions': None, 'secure': False}
    if path.exists():
        result['readable'] = os.access(path, os.R_OK)
        result['writable'] = os.access(path, os.W_OK)
        if 'secrets' in path.name:
            stat_info = path.stat()
            mode = stat_info.st_mode
            result['secure'] = not bool(mode & 4)
            result['permissions'] = oct(mode)[-3:]
    return result

def _check_command(cmd: str) -> tuple[bool, str]:
    """Check if a command is available and return version if possible."""
    parts = cmd.split()
    exe = parts[0]
    if not shutil.which(exe):
        return (False, 'Not found')
    version = 'Found'
    try:
        if exe in ['node', 'npm', 'npx', 'python', 'python3', 'pip', 'uv', 'uvx']:
            result = subprocess.run([exe, '--version'], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                version = result.stdout.strip()
    except Exception:
        pass
    return (True, version)

def _check_url(url: str, timeout: float=2.0) -> tuple[bool, str]:
    """Check if a URL is reachable and return response time."""
    try:
        from urllib.parse import urlparse
        import time
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        if not host:
            return (False, 'Invalid URL')
        start = time.time()
        with socket.create_connection((host, port), timeout=timeout):
            elapsed = time.time() - start
            return (True, f'{elapsed * 1000:.0f}ms')
    except socket.timeout:
        return (False, 'Timeout')
    except socket.gaierror:
        return (False, 'DNS error')
    except Exception as e:
        return (False, str(e)[:20])

def _check_dependencies() -> Dict[str, Any]:
    """Check Python dependencies and versions."""
    deps = {}
    required_packages = ['mcp', 'typer', 'rich', 'pydantic', 'httpx', 'yaml']
    for package in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            deps[package] = {'installed': True, 'version': version}
        except ImportError:
            deps[package] = {'installed': False, 'version': None}
    deps['python'] = {'version': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}', 'supported': sys.version_info >= (3, 10)}
    return deps

def _check_network_connectivity() -> Dict[str, bool]:
    """Check connectivity to common services."""
    endpoints = {'internet': ('8.8.8.8', 53), 'openai': ('api.openai.com', 443), 'anthropic': ('api.anthropic.com', 443), 'google': ('generativelanguage.googleapis.com', 443), 'github': ('api.github.com', 443)}
    results = {}
    for name, (host, port) in endpoints.items():
        try:
            with socket.create_connection((host, port), timeout=2):
                results[name] = True
        except Exception:
            results[name] = False
    return results

def _validate_config_schema(settings: Settings) -> List[str]:
    """Validate configuration against expected schema."""
    warnings = []
    if not settings.execution_engine:
        warnings.append('No execution_engine specified (defaulting to asyncio)')
    if settings.logger and settings.logger.type == 'file':
        if not settings.logger.path_settings:
            warnings.append("Logger type is 'file' but no path_settings configured")
    if settings.mcp and settings.mcp.servers:
        for name, server in settings.mcp.servers.items():
            if server.transport == 'stdio' and (not server.command):
                warnings.append(f"Server '{name}' missing command")
            elif server.transport in ['http', 'sse'] and (not server.url):
                warnings.append(f"Server '{name}' missing URL")
    return warnings

@app.command()
def validate(config_file: Path=typer.Option(Path('mcp_agent.config.yaml'), '--config', '-c'), secrets_file: Path=typer.Option(Path('mcp_agent.secrets.yaml'), '--secrets', '-s')) -> None:
    """Validate configuration files against schema."""
    console.print('\n[bold]Validating configuration files...[/bold]\n')
    errors = []
    if not config_file.exists():
        errors.append(f'Config file not found: {config_file}')
    if not secrets_file.exists():
        console.print(f'[yellow]Warning:[/yellow] Secrets file not found: {secrets_file}')
    if errors:
        for error in errors:
            console.print(f'[red]Error:[/red] {error}')
        raise typer.Exit(1)
    try:
        settings = get_settings()
        warnings = _validate_config_schema(settings)
        if warnings:
            console.print('[yellow]Validation warnings:[/yellow]')
            for warning in warnings:
                console.print(f'  • {warning}')
        else:
            console.print('[green]✅ Configuration is valid[/green]')
    except Exception as e:
        console.print(f'[red]Validation error:[/red] {e}')
        raise typer.Exit(1)

def select_servers_from_config(explicit_servers_csv: Optional[str], url_servers: Optional[Dict[str, Dict[str, Any]]], stdio_servers: Optional[Dict[str, Dict[str, Any]]]) -> List[str]:
    """Resolve which servers should be active based on inputs and config.

    - If explicit --servers provided, use those
    - Else, if dynamic URL/stdio servers provided, use their names
    - Else, use all servers from mcp_agent.config.yaml (if present)
    """
    if explicit_servers_csv:
        items = [s.strip() for s in explicit_servers_csv.split(',') if s.strip()]
        return items
    names: List[str] = []
    if url_servers:
        names.extend(list(url_servers.keys()))
    if stdio_servers:
        names.extend(list(stdio_servers.keys()))
    if names:
        return names
    settings = get_settings()
    if settings.mcp and settings.mcp.servers:
        return list(settings.mcp.servers.keys())
    return []

def send_usage_data():
    config = get_settings()
    if not config.usage_telemetry.enabled:
        logger.info('Usage tracking is disabled')
        return

class ServerRegistry:
    """
    A registry for managing server configurations and initialization logic.

    The `ServerRegistry` class is responsible for loading server configurations
    from a YAML file, registering initialization hooks, initializing servers,
    and executing post-initialization hooks dynamically.

    Attributes:
        config_path (str): Path to the YAML configuration file.
        registry (Dict[str, MCPServerSettings]): Loaded server configurations.
        init_hooks (Dict[str, InitHookCallable]): Registered initialization hooks.
    """

    def __init__(self, config: Settings | None=None, config_path: str | None=None):
        """
        Initialize the ServerRegistry with a configuration file.

        Args:
            config (Settings): The Settings object containing the server configurations.
            config_path (str): Path to the YAML configuration file.
        """
        mcp_servers = self.load_registry_from_file(config_path) if config is None else config.mcp.servers
        for server_name in mcp_servers:
            if mcp_servers[server_name].name is None:
                mcp_servers[server_name].name = server_name
        self.registry = mcp_servers
        self.init_hooks: Dict[str, InitHookCallable] = {}
        self.connection_manager = MCPConnectionManager(self)

    def load_registry_from_file(self, config_path: str | None=None) -> Dict[str, MCPServerSettings]:
        """
        Load the YAML configuration file and validate it.

        Returns:
            Dict[str, MCPServerSettings]: A dictionary of server configurations.

        Raises:
            ValueError: If the configuration is invalid.
        """
        servers = get_settings(config_path).mcp.servers or {}
        return servers

    @asynccontextmanager
    async def start_server(self, server_name: str, client_session_factory: Callable[[MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None, Optional['Context']], ClientSession]=ClientSession, session_id: str | None=None, context: Optional['Context']=None) -> AsyncGenerator[ClientSession, None]:
        """
        Starts the server process based on its configuration. To initialize, call initialize_server

        Args:
            server_name (str): The name of the server to initialize.

        Returns:
            StdioServerParameters: The server parameters for stdio transport.

        Raises:
            ValueError: If the server is not found or has an unsupported transport.
        """
        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry.")
        config = self.registry[server_name]
        read_timeout_seconds = timedelta(config.read_timeout_seconds) if config.read_timeout_seconds else None
        if config.transport == 'stdio':
            if not config.command and (not config.args):
                raise ValueError(f'Command and args are required for stdio transport: {server_name}')
            server_params = StdioServerParameters(command=config.command, args=config.args or [], env={**get_default_environment(), **(config.env or {})}, cwd=config.cwd or None)
            async with filtered_stdio_client(server_name=server_name, server=server_params) as (read_stream, write_stream):
                try:
                    session = client_session_factory(read_stream, write_stream, read_timeout_seconds, context=context)
                except TypeError:
                    session = client_session_factory(read_stream, write_stream, read_timeout_seconds)
                async with session:
                    logger.info(f'{server_name}: Connected to server using stdio transport.')
                    try:
                        yield session
                    finally:
                        logger.debug(f'{server_name}: Closed session to server')
        elif config.transport in ['streamable_http', 'streamable-http', 'http']:
            if not config.url:
                raise ValueError(f'URL is required for Streamable HTTP transport: {server_name}')
            if session_id:
                headers = config.headers.copy() if config.headers else {}
                headers[MCP_SESSION_ID] = session_id
            else:
                headers = config.headers
            kwargs = {'url': config.url, 'headers': headers, 'terminate_on_close': config.terminate_on_close}
            timeout = timedelta(seconds=config.http_timeout_seconds) if config.http_timeout_seconds else None
            if timeout is not None:
                kwargs['timeout'] = timeout
            sse_read_timeout = timedelta(seconds=config.read_timeout_seconds) if config.read_timeout_seconds else None
            if sse_read_timeout is not None:
                kwargs['sse_read_timeout'] = sse_read_timeout
            auth_handler = None
            oauth_cfg = config.auth.oauth if config.auth else None
            if oauth_cfg and oauth_cfg.enabled:
                if context is None or getattr(context, 'token_manager', None) is None:
                    logger.warning(f'{server_name}: OAuth configured but token manager not available; skipping auth')
                else:
                    auth_handler = OAuthHttpxAuth(token_manager=context.token_manager, context=context, server_name=server_name, server_config=config, scopes=oauth_cfg.scopes, identity_resolver=_resolve_identity_from_context)
            if auth_handler:
                kwargs['auth'] = auth_handler
            async with streamablehttp_client(**kwargs) as (read_stream, write_stream, session_id_callback):
                try:
                    session = client_session_factory(read_stream, write_stream, read_timeout_seconds, context=context)
                except TypeError:
                    session = client_session_factory(read_stream, write_stream, read_timeout_seconds)
                if session_id_callback and isinstance(session, MCPAgentClientSession):
                    session.set_session_id_callback(session_id_callback)
                    logger.debug(f'{server_name}: Session ID tracking enabled')
                async with session:
                    logger.info(f'{server_name}: Connected to server using Streamable HTTP transport.')
                    try:
                        yield session
                    finally:
                        logger.debug(f'{server_name}: Closed session to server')
        elif config.transport == 'sse':
            if not config.url:
                raise ValueError(f'URL is required for SSE transport: {server_name}')
            kwargs = {'url': config.url, 'headers': config.headers}
            if config.http_timeout_seconds:
                kwargs['timeout'] = config.http_timeout_seconds
            if config.read_timeout_seconds:
                kwargs['sse_read_timeout'] = config.read_timeout_seconds
            async with sse_client(**kwargs) as (read_stream, write_stream):
                try:
                    session = client_session_factory(read_stream, write_stream, read_timeout_seconds, context=context)
                except TypeError:
                    session = client_session_factory(read_stream, write_stream, read_timeout_seconds)
                async with session:
                    logger.info(f'{server_name}: Connected to server using SSE transport.')
                    try:
                        yield session
                    finally:
                        logger.debug(f'{server_name}: Closed session to server')
        elif config.transport == 'websocket':
            if not config.url:
                raise ValueError(f'URL is required for websocket transport: {server_name}')
            async with websocket_client(url=config.url) as (read_stream, write_stream):
                try:
                    session = client_session_factory(read_stream, write_stream, read_timeout_seconds, context=context)
                except TypeError:
                    session = client_session_factory(read_stream, write_stream, read_timeout_seconds)
                async with session:
                    logger.info(f'{server_name}: Connected to server using websocket transport.')
                    try:
                        yield session
                    finally:
                        logger.debug(f'{server_name}: Closed session to server')
        else:
            raise ValueError(f'Unsupported transport: {config.transport}')

    @asynccontextmanager
    async def initialize_server(self, server_name: str, client_session_factory: Callable[[MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None, Optional['Context']], ClientSession]=ClientSession, init_hook: InitHookCallable=None, session_id: str | None=None, context: Optional['Context']=None) -> AsyncGenerator[ClientSession, None]:
        """
        Initialize a server based on its configuration.
        After initialization, also calls any registered or provided initialization hook for the server.

        Args:
            server_name (str): The name of the server to initialize.
            init_hook (InitHookCallable): Optional initialization hook function to call after initialization.

        Returns:
            StdioServerParameters: The server parameters for stdio transport.

        Raises:
            ValueError: If the server is not found or has an unsupported transport.
        """
        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry.")
        config = self.registry[server_name]
        async with self.start_server(server_name, client_session_factory=client_session_factory, session_id=session_id, context=context) as session:
            try:
                logger.info(f'{server_name}: Initializing server...')
                await session.initialize()
                logger.info(f'{server_name}: Initialized.')
                intialization_callback = init_hook if init_hook is not None else self.init_hooks.get(server_name)
                if intialization_callback:
                    logger.info(f'{server_name}: Executing init hook')
                    intialization_callback(session, config.auth)
                logger.info(f'{server_name}: Up and running!')
                yield session
            finally:
                logger.info(f'{server_name}: Ending server session.')

    def register_init_hook(self, server_name: str, hook: InitHookCallable) -> None:
        """
        Register an initialization hook for a specific server. This will get called
        after the server is initialized.

        Args:
            server_name (str): The name of the server.
            hook (callable): The initialization function to register.
        """
        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry.")
        self.init_hooks[server_name] = hook

    def execute_init_hook(self, server_name: str, session=None) -> bool:
        """
        Execute the initialization hook for a specific server.

        Args:
            server_name (str): The name of the server.
            session: The session object to pass to the initialization hook.
        """
        if server_name in self.init_hooks:
            hook = self.init_hooks[server_name]
            config = self.registry[server_name]
            logger.info(f"Executing init hook for '{server_name}'")
            return hook(session, config.auth)
        else:
            logger.info(f"No init hook registered for '{server_name}'")

    def get_server_config(self, server_name: str) -> MCPServerSettings | None:
        """
        Get the configuration for a specific server.

        Args:
            server_name (str): The name of the server.

        Returns:
            MCPServerSettings: The server configuration.
        """
        server_config = self.registry.get(server_name)
        if server_config is None:
            logger.warning(f"Server '{server_name}' not found in registry.")
            return None
        elif server_config.name is None:
            server_config.name = server_name
        return server_config

def get_current_config():
    """
    Get the current application config.
    """
    return get_current_context().config or get_settings()

def get_current_context() -> Context:
    """
    Synchronous initializer/getter for global application context.
    For async usage, use aget_current_context instead.
    """
    request_ctx = get_current_request_context()
    if request_ctx is not None:
        return request_ctx
    global _global_context
    if _global_context is None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():

                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    return new_loop.run_until_complete(initialize_context())
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    _global_context = pool.submit(run_async).result()
            else:
                _global_context = loop.run_until_complete(initialize_context())
        except RuntimeError:
            _global_context = asyncio.run(initialize_context())
        warnings.warn('get_current_context() created a global Context. In multithreaded runs, instantiate an MCPApp per thread and use app.context instead.', stacklevel=2)
    return _global_context

class ContextDependent:
    """
    Mixin class for components that need context access.
    Provides both global fallback and instance-specific context support.
    """

    def __init__(self, context: Optional['Context']=None, **kwargs):
        self._context = context
        super().__init__(**kwargs)

    @property
    def context(self) -> 'Context':
        """
        Get context, with graceful fallback to global context if needed.
        Raises clear error if no context is available.
        """
        if self._context is not None:
            return self._context
        try:
            from mcp_agent.core.context import get_current_context
            return get_current_context()
        except Exception as e:
            raise RuntimeError(f'No context available for {self.__class__.__name__}. Either initialize MCPApp first or pass context explicitly.') from e

    @contextmanager
    def use_context(self, context: 'Context'):
        """Temporarily use a different context."""
        old_context = self._context
        self._context = context
        try:
            yield
        finally:
            self._context = old_context

def _load_settings():
    signature = inspect.signature(get_settings)
    if 'set_global' in signature.parameters:
        return get_settings(set_global=False)
    return get_settings()

class TestConfigEnvAliases:

    @pytest.fixture(autouse=True)
    def clear_settings(self):
        _clear_global_settings()

    @pytest.fixture(autouse=True)
    def isolate_env(self, monkeypatch):
        for key in ['OPENAI_API_KEY', 'OPENAI__API_KEY', 'openai__api_key', 'ANTHROPIC_API_KEY', 'ANTHROPIC__API_KEY', 'anthropic__api_key', 'ANTHROPIC__PROVIDER', 'AZURE_OPENAI_API_KEY', 'AZURE_AI_API_KEY', 'AZURE__API_KEY', 'azure__api_key', 'AZURE_OPENAI_ENDPOINT', 'AZURE_AI_ENDPOINT', 'AZURE__ENDPOINT', 'azure__endpoint', 'GOOGLE_API_KEY', 'GEMINI_API_KEY', 'GOOGLE__API_KEY', 'google__api_key', 'AWS_ACCESS_KEY_ID', 'bedrock__aws_access_key_id', 'AWS_SECRET_ACCESS_KEY', 'bedrock__aws_secret_access_key', 'AWS_SESSION_TOKEN', 'bedrock__aws_session_token', 'AWS_REGION', 'bedrock__aws_region', 'AWS_PROFILE', 'bedrock__profile', 'BEDROCK__AWS_ACCESS_KEY_ID', 'BEDROCK__AWS_SECRET_ACCESS_KEY', 'BEDROCK__AWS_SESSION_TOKEN', 'BEDROCK__AWS_REGION', 'BEDROCK__PROFILE']:
            monkeypatch.delenv(key, raising=False)

    @pytest.mark.parametrize('env_name', ['OPENAI_API_KEY', 'OPENAI__API_KEY'])
    def test_openai_api_key_env_variants(self, monkeypatch, env_name):
        value = 'sk-openai-env'
        monkeypatch.setenv(env_name, value)
        settings = get_settings()
        assert settings.openai is not None
        assert getattr(settings.openai, 'api_key') == value

    @pytest.mark.parametrize('env_name', ['ANTHROPIC_API_KEY', 'ANTHROPIC__API_KEY'])
    def test_anthropic_api_key_env_variants(self, monkeypatch, env_name):
        value = 'sk-anthropic-env'
        monkeypatch.setenv(env_name, value)
        settings = get_settings()
        assert settings.anthropic is not None
        assert getattr(settings.anthropic, 'api_key') == value

    @pytest.mark.parametrize('env_name', ['AZURE_OPENAI_API_KEY', 'AZURE_AI_API_KEY', 'AZURE__API_KEY'])
    def test_azure_api_key_env_variants(self, monkeypatch, env_name):
        value = 'az-key-env'
        monkeypatch.setenv(env_name, value)
        settings = get_settings()
        assert settings.azure is not None
        assert getattr(settings.azure, 'api_key') == value

    @pytest.mark.parametrize('env_name', ['AZURE_OPENAI_ENDPOINT', 'AZURE_AI_ENDPOINT', 'AZURE__ENDPOINT'])
    def test_azure_endpoint_env_variants(self, monkeypatch, env_name):
        value = 'https://azure.example'
        monkeypatch.setenv(env_name, value)
        settings = get_settings()
        assert settings.azure is not None
        assert getattr(settings.azure, 'endpoint') == value

    @pytest.mark.parametrize('env_name', ['GOOGLE_API_KEY', 'GEMINI_API_KEY', 'GOOGLE__API_KEY'])
    def test_google_api_key_env_variants(self, monkeypatch, env_name):
        value = 'g-api-env'
        monkeypatch.setenv(env_name, value)
        settings = get_settings()
        assert settings.google is not None
        assert getattr(settings.google, 'api_key') == value

    @pytest.mark.parametrize('env_name, attr, value', [('AWS_ACCESS_KEY_ID', 'aws_access_key_id', 'AKIA_ENV'), ('AWS_SECRET_ACCESS_KEY', 'aws_secret_access_key', 'SECRET_ENV'), ('AWS_SESSION_TOKEN', 'aws_session_token', 'TOKEN_ENV'), ('AWS_REGION', 'aws_region', 'us-east-1'), ('AWS_PROFILE', 'profile', 'dev')])
    def test_bedrock_flat_env(self, monkeypatch, env_name, attr, value):
        monkeypatch.setenv(env_name, value)
        settings = get_settings()
        assert settings.bedrock is not None
        assert getattr(settings.bedrock, attr) == value

    def test_aliases_from_yaml_preload(self, monkeypatch):
        yaml_payload = '\nopenai:\n  OPENAI_API_KEY: sk-openai-yaml\nanthropic:\n  ANTHROPIC_API_KEY: sk-anthropic-yaml\nazure:\n  AZURE_OPENAI_API_KEY: az-key-yaml\n  AZURE_OPENAI_ENDPOINT: https://azure.openai.example\ngoogle:\n  GEMINI_API_KEY: g-api-gemini-yaml\nbedrock:\n  AWS_ACCESS_KEY_ID: AKIA_YAML\n  AWS_SECRET_ACCESS_KEY: SECRET_YAML\n  AWS_SESSION_TOKEN: TOKEN_YAML\n  AWS_REGION: us-east-2\n  AWS_PROFILE: default\n'
        monkeypatch.setenv('MCP_APP_SETTINGS_PRELOAD', yaml_payload)
        settings = get_settings()
        assert settings.openai and getattr(settings.openai, 'api_key') == 'sk-openai-yaml'
        assert settings.anthropic and getattr(settings.anthropic, 'api_key') == 'sk-anthropic-yaml'
        assert settings.azure and getattr(settings.azure, 'api_key') == 'az-key-yaml'
        assert getattr(settings.azure, 'endpoint') == 'https://azure.openai.example'
        assert settings.google and getattr(settings.google, 'api_key') == 'g-api-gemini-yaml'
        assert settings.bedrock and getattr(settings.bedrock, 'aws_access_key_id') == 'AKIA_YAML'
        assert getattr(settings.bedrock, 'aws_secret_access_key') == 'SECRET_YAML'
        assert getattr(settings.bedrock, 'aws_session_token') == 'TOKEN_YAML'
        assert getattr(settings.bedrock, 'aws_region') == 'us-east-2'
        assert getattr(settings.bedrock, 'profile') == 'default'

    def test_preload_yaml_overrides_env(self, monkeypatch):
        monkeypatch.setenv('OPENAI_API_KEY', 'env-openai')
        yaml_payload = '\nopenai:\n  api_key: yaml-openai\n'
        monkeypatch.setenv('MCP_APP_SETTINGS_PRELOAD', yaml_payload)
        settings = get_settings()
        assert getattr(settings.openai, 'api_key') == 'yaml-openai'

    def test_yaml_used_when_env_missing_value(self, monkeypatch):
        yaml_payload = '\n    openai:\n      api_key: yaml-openai\n    '
        monkeypatch.setenv('MCP_APP_SETTINGS_PRELOAD', yaml_payload)
        settings = get_settings()
        assert getattr(settings.openai, 'api_key') == 'yaml-openai'
        monkeypatch.setenv('OPENAI_API_KEY', 'env-openai')
        settings = get_settings()
        assert getattr(settings.openai, 'api_key') == 'yaml-openai'

    def test_env_vs_secrets_yaml_precedence(self, monkeypatch):
        yaml_payload = '\nopenai:\n  api_key: yaml-openai\nanthropic:\n  api_key: yaml-claude\n'
        monkeypatch.setenv('MCP_APP_SETTINGS_PRELOAD', yaml_payload)
        settings = get_settings()
        assert getattr(settings.openai, 'api_key') == 'yaml-openai'
        assert getattr(settings.anthropic, 'api_key') == 'yaml-claude'
        monkeypatch.delenv('MCP_APP_SETTINGS_PRELOAD', raising=False)
        monkeypatch.setenv('OPENAI_API_KEY', 'env-openai')
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'env-claude')
        _clear_global_settings()
        settings = get_settings()
        assert getattr(settings.openai, 'api_key') == 'env-openai'
        assert getattr(settings.anthropic, 'api_key') == 'env-claude'

    def test_dotenv_loading_from_cwd(self, monkeypatch, tmp_path):
        proj = tmp_path / 'proj'
        proj.mkdir()
        env_file = proj / '.env'
        env_file.write_text('OPENAI_API_KEY=dotenv-openai\nANTHROPIC_API_KEY=dotenv-claude\n')
        monkeypatch.chdir(proj)
        _clear_global_settings()
        settings = get_settings()
        assert getattr(settings.openai, 'api_key') == 'dotenv-openai'
        assert getattr(settings.anthropic, 'api_key') == 'dotenv-claude'

    def test_nested_and_flat_env_compat(self, monkeypatch):
        monkeypatch.setenv('OPENAI_API_KEY', 'flat-openai')
        monkeypatch.setenv('ANTHROPIC__API_KEY', 'nested-claude')
        _clear_global_settings()
        settings = get_settings()
        assert getattr(settings.openai, 'api_key') == 'flat-openai'
        assert getattr(settings.anthropic, 'api_key') == 'nested-claude'

    def test_anthropic_provider_bedrock_via_nested_env(self, monkeypatch):
        monkeypatch.setenv('ANTHROPIC__PROVIDER', 'bedrock')
        monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'AKIA_TEST')
        monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'SECRET_TEST')
        monkeypatch.setenv('AWS_REGION', 'us-east-1')
        settings = get_settings()
        assert getattr(settings.anthropic, 'provider') == 'bedrock'
        assert getattr(settings.anthropic, 'aws_access_key_id') == 'AKIA_TEST'
        assert getattr(settings.anthropic, 'aws_secret_access_key') == 'SECRET_TEST'
        assert getattr(settings.anthropic, 'aws_region') == 'us-east-1'

def _clear_global_settings():
    """
    Convenience for testing - clear the global memoized settings.
    """
    global _settings
    _settings = None

class TestConfigPreload:

    @pytest.fixture(autouse=True)
    def clear_global_settings(self):
        _clear_global_settings()

    @pytest.fixture(autouse=True)
    def clear_test_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv('MCP_APP_SETTINGS_PRELOAD', raising=False)
        monkeypatch.delenv('MCP_APP_SETTINGS_PRELOAD_STRICT', raising=False)

    @pytest.fixture(scope='session')
    def example_settings(self):
        return _EXAMPLE_SETTINGS

    @pytest.fixture(scope='function')
    def settings_env(self, example_settings: Settings, monkeypatch: pytest.MonkeyPatch):
        settings_str = to_yaml_str(example_settings)
        monkeypatch.setenv('MCP_APP_SETTINGS_PRELOAD', settings_str)

    def test_config_preload(self, example_settings: Settings, settings_env):
        assert os.environ.get('MCP_APP_SETTINGS_PRELOAD')
        loaded_settings = get_settings()
        assert loaded_settings == example_settings

    def test_config_preload_override(self, example_settings: Settings, settings_env):
        assert os.environ.get('MCP_APP_SETTINGS_PRELOAD')
        loaded_settings = get_settings('./fake_path/mcp-agent.config.yaml')
        assert loaded_settings == example_settings

    @pytest.fixture(scope='function')
    def invalid_settings_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv('MCP_APP_SETTINGS_PRELOAD', '\n            badsadwewqeqr231232321\n        ')

    def test_config_preload_invalid_lenient(self, invalid_settings_env):
        assert os.environ.get('MCP_APP_SETTINGS_PRELOAD')
        assert os.environ.get('MCP_APP_SETTINGS_PRELOAD_STRICT') is None
        loaded_settings = get_settings()
        assert loaded_settings

    @pytest.fixture(scope='function')
    def strict_parsing_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv('MCP_APP_SETTINGS_PRELOAD_STRICT', 'true')

    def test_config_preload_invalid_throws(self, invalid_settings_env, strict_parsing_env):
        assert os.environ.get('MCP_APP_SETTINGS_PRELOAD')
        assert os.environ.get('MCP_APP_SETTINGS_PRELOAD_STRICT') == 'true'
        with pytest.raises(ValueError):
            get_settings()

class TestSetGlobalParameter:
    """Test suite for the set_global parameter in get_settings()."""

    @pytest.fixture(autouse=True)
    def clear_global_settings(self):
        """Clear global settings before and after each test."""
        _clear_global_settings()
        yield
        _clear_global_settings()

    @pytest.fixture(autouse=True)
    def clear_test_env(self, monkeypatch: pytest.MonkeyPatch):
        """Ensure a clean environment before each test."""
        monkeypatch.delenv('MCP_APP_SETTINGS_PRELOAD', raising=False)
        monkeypatch.delenv('MCP_APP_SETTINGS_PRELOAD_STRICT', raising=False)

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration dictionary."""
        return {'execution_engine': 'asyncio', 'logger': {'type': 'console', 'level': 'info'}, 'mcp': {'servers': {'test_server': {'command': 'python', 'args': ['-m', 'test_server']}}}}

    def test_default_sets_global_state(self, sample_config):
        """Test that get_settings() with default parameters sets global state."""
        assert mcp_agent.config._settings is None
        yaml_content = yaml.dump(sample_config)
        config_path = '/fake/path/config.yaml'
        with patch('mcp_agent.config._check_file_exists', return_value=True):
            with patch('mcp_agent.config._read_file_content', return_value=yaml_content):
                settings = get_settings(config_path=config_path)
                assert mcp_agent.config._settings is not None
                assert mcp_agent.config._settings == settings
                assert settings.execution_engine == 'asyncio'

    def test_set_global_false_no_global_state(self, sample_config):
        """Test that set_global=False doesn't modify global state."""
        assert mcp_agent.config._settings is None
        yaml_content = yaml.dump(sample_config)
        config_path = '/fake/path/config.yaml'
        with patch('mcp_agent.config._check_file_exists', return_value=True):
            with patch('mcp_agent.config._read_file_content', return_value=yaml_content):
                settings = get_settings(config_path=config_path, set_global=False)
                assert mcp_agent.config._settings is None
                assert settings is not None
                assert settings.execution_engine == 'asyncio'

    def test_explicit_set_global_true(self, sample_config):
        """Test explicitly passing set_global=True."""
        assert mcp_agent.config._settings is None
        yaml_content = yaml.dump(sample_config)
        config_path = '/fake/path/config.yaml'
        with patch('mcp_agent.config._check_file_exists', return_value=True):
            with patch('mcp_agent.config._read_file_content', return_value=yaml_content):
                settings = get_settings(config_path=config_path, set_global=True)
                assert mcp_agent.config._settings is not None
                assert mcp_agent.config._settings == settings

    def test_returns_cached_global_when_set(self, sample_config):
        """Test that subsequent calls return cached global settings."""
        yaml_content = yaml.dump(sample_config)
        config_path = '/fake/path/config.yaml'
        with patch('mcp_agent.config._check_file_exists', return_value=True):
            with patch('mcp_agent.config._read_file_content', return_value=yaml_content):
                settings1 = get_settings(config_path=config_path)
                settings2 = get_settings()
                assert settings1 is settings2
                assert mcp_agent.config._settings is settings1

    def test_no_cached_return_when_set_global_false(self, sample_config):
        """Test that set_global=False always loads fresh settings."""
        yaml_content = yaml.dump(sample_config)
        config_path = '/fake/path/config.yaml'
        with patch('mcp_agent.config._check_file_exists', return_value=True):
            with patch('mcp_agent.config._read_file_content', return_value=yaml_content):
                settings1 = get_settings(config_path=config_path, set_global=False)
                settings2 = get_settings(config_path=config_path, set_global=False)
                assert settings1 is not settings2
                assert settings1 == settings2
                assert mcp_agent.config._settings is None

    def test_preload_with_set_global_false(self, sample_config, monkeypatch):
        """Test preload configuration with set_global=False."""
        settings_str = to_yaml_str(Settings(**sample_config))
        monkeypatch.setenv('MCP_APP_SETTINGS_PRELOAD', settings_str)
        settings = get_settings(set_global=False)
        assert mcp_agent.config._settings is None
        assert settings is not None
        assert settings.execution_engine == 'asyncio'

    def test_explicit_config_path_with_cache_returns_cached(self, sample_config):
        """Test that explicit config_path still returns cached settings when global cache exists."""
        initial_config = {'execution_engine': 'asyncio', 'logger': {'type': 'console', 'level': 'info'}}
        updated_config = {'execution_engine': 'temporal', 'logger': {'type': 'file', 'level': 'debug'}}
        initial_yaml = yaml.dump(initial_config)
        updated_yaml = yaml.dump(updated_config)
        with patch('mcp_agent.config._check_file_exists', return_value=True):
            with patch('mcp_agent.config._read_file_content', return_value=initial_yaml):
                settings1 = get_settings(config_path='/fake/path/initial.yaml')
                assert settings1.execution_engine == 'asyncio'
                assert settings1.logger.type == 'console'
                assert settings1.logger.level == 'info'
                assert mcp_agent.config._settings == settings1
        settings2 = get_settings()
        assert settings2 is settings1
        assert settings2.execution_engine == 'asyncio'
        with patch('mcp_agent.config._check_file_exists', return_value=True):
            with patch('mcp_agent.config._read_file_content', return_value=updated_yaml):
                settings3 = get_settings(config_path='/fake/path/updated.yaml')
                assert settings3 is settings1
                assert settings3.execution_engine == 'asyncio'
                assert settings3.logger.type == 'console'
                assert settings3.logger.level == 'info'
                assert mcp_agent.config._settings == settings1
        with patch('mcp_agent.config._check_file_exists', return_value=True):
            with patch('mcp_agent.config._read_file_content', return_value=updated_yaml):
                settings4 = get_settings(config_path='/fake/path/updated.yaml', set_global=False)
                assert settings4.execution_engine == 'temporal'
                assert settings4.logger.type == 'file'
                assert settings4.logger.level == 'debug'
                assert mcp_agent.config._settings == settings1

class TestThreadSafety:
    """Test thread safety with the set_global parameter."""

    @pytest.fixture(autouse=True)
    def clear_global_settings(self):
        """Clear global settings before and after each test."""
        _clear_global_settings()
        yield
        _clear_global_settings()

    @pytest.fixture
    def simple_config(self):
        """Simple config for thread safety tests."""
        return {'execution_engine': 'asyncio'}

    def test_warning_from_non_main_thread_with_set_global(self):
        """Test that warning is issued when setting global from non-main thread."""
        warning_caught = []

        def load_in_thread():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                get_settings(set_global=True)
                if w:
                    warning_caught.extend(w)
        thread = threading.Thread(target=load_in_thread)
        thread.start()
        thread.join()
        assert len(warning_caught) > 0
        assert 'non-main thread' in str(warning_caught[0].message)
        assert 'set_global=False' in str(warning_caught[0].message)

    def test_no_warning_from_non_main_thread_without_set_global(self):
        """Test that no warning is issued with set_global=False from non-main thread."""
        warning_caught = []

        def load_in_thread():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                get_settings(set_global=False)
                if w:
                    warning_caught.extend(w)
        thread = threading.Thread(target=load_in_thread)
        thread.start()
        thread.join()
        assert len(warning_caught) == 0

    def test_no_warning_from_main_thread(self):
        """Test that no warning is issued from main thread."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            get_settings(set_global=True)
            thread_warnings = [warn for warn in w if 'non-main thread' in str(warn.message)]
            assert len(thread_warnings) == 0

    def test_multiple_threads_independent_settings(self, simple_config):
        """Test that multiple threads can load independent settings."""
        thread_settings = {}
        yaml_content = yaml.dump(simple_config)

        def load_settings(thread_id, config_path):
            settings = get_settings(config_path=config_path, set_global=False)
            thread_settings[thread_id] = settings
        with patch('mcp_agent.config._check_file_exists', return_value=True):
            with patch('mcp_agent.config._read_file_content', return_value=yaml_content):
                threads = []
                for i in range(3):
                    thread = threading.Thread(target=load_settings, args=(i, '/fake/path/config.yaml'))
                    threads.append(thread)
                    thread.start()
                for thread in threads:
                    thread.join()
        assert mcp_agent.config._settings is None
        assert len(thread_settings) == 3
        for i in range(3):
            assert thread_settings[i] is not None
            assert thread_settings[i].execution_engine == 'asyncio'

class TestConfigMergingWithSetGlobal:
    """Test configuration merging with set_global parameter."""

    @pytest.fixture(autouse=True)
    def clear_global_settings(self):
        """Clear global settings before and after each test."""
        _clear_global_settings()
        yield
        _clear_global_settings()

    @pytest.fixture
    def config_data_with_secrets(self):
        """Config and secrets data for testing merging."""
        config_data = {'execution_engine': 'asyncio', 'openai': {'api_key': 'config-key'}}
        secrets_data = {'openai': {'api_key': 'secret-key'}}
        return (config_data, secrets_data)

    def test_config_and_secrets_merge_with_set_global_false(self, config_data_with_secrets):
        """Test that config and secrets merge correctly without setting global state."""
        config_data, secrets_data = config_data_with_secrets
        merged_data = config_data.copy()
        merged_data['openai'] = secrets_data['openai']
        merged_yaml = yaml.dump(merged_data)
        config_path = '/fake/path/config.yaml'
        with patch('mcp_agent.config._check_file_exists', return_value=True):
            with patch('mcp_agent.config._read_file_content', return_value=merged_yaml):
                settings = get_settings(config_path=config_path, set_global=False)
                assert mcp_agent.config._settings is None
                assert settings.openai.api_key == 'secret-key'
                assert settings.execution_engine == 'asyncio'

    def test_default_settings_with_set_global_false(self):
        """Test loading default settings without setting global state."""
        settings = get_settings(set_global=False)
        assert mcp_agent.config._settings is None
        assert settings is not None
        assert isinstance(settings, Settings)

