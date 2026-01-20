# Cluster 23

def _persist_server_entry(name: str, settings: MCPServerSettings) -> None:
    import yaml
    cfg_path, data = _load_config_yaml()
    if 'mcp' not in data:
        data['mcp'] = {}
    if 'servers' not in data['mcp'] or data['mcp']['servers'] is None:
        data['mcp']['servers'] = {}
    entry = {'transport': settings.transport}
    if settings.transport == 'stdio':
        if settings.command:
            entry['command'] = settings.command
        if settings.args:
            entry['args'] = settings.args
        if settings.env:
            entry['env'] = settings.env
        if settings.cwd:
            entry['cwd'] = settings.cwd
    else:
        if settings.url:
            entry['url'] = settings.url
        if settings.headers:
            entry['headers'] = settings.headers
    data['mcp']['servers'][name] = entry
    if not cfg_path:
        from pathlib import Path as _Path
        cfg_path = _Path('mcp_agent.config.yaml')
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False))
    console.print(f"[green]✅[/green] Added server '[cyan]{name}[/cyan]' to {cfg_path}")

def _load_config_yaml(path: Settings | None=None):
    import yaml
    cfg_path = Settings.find_config()
    data = {}
    if cfg_path and cfg_path.exists():
        try:
            data = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            data = {}
    return (cfg_path, data)

@app.command('add')
def add(kind: str=typer.Argument(..., help='http|sse|stdio|npx|uvx|recipe|dxt|auto'), value: str=typer.Argument(..., help='URL, command, or recipe name'), name: Optional[str]=typer.Option(None, '--name', '-n', help='Server name'), auth: Optional[str]=typer.Option(None, '--auth', help='Authorization token'), env: Optional[str]=typer.Option(None, '--env', '-e', help='Environment variables (KEY=value,...)'), cwd: Optional[str]=typer.Option(None, '--cwd', help='Working directory for stdio server process'), write: bool=typer.Option(True, '--write/--no-write', help='Persist to config file'), force: bool=typer.Option(False, '--force', '-f', help='Overwrite existing server'), extract_to: Optional[str]=typer.Option(None, '--extract-to', help='Extraction dir for .dxt (defaults to .mcp-agent/extensions/<name>)')) -> None:
    """Add a server to configuration."""
    settings = get_settings()
    if settings.mcp is None:
        settings.mcp = MCPSettings()
    servers = settings.mcp.servers or {}
    env_dict = {}
    if env:
        for pair in env.split(','):
            if '=' in pair:
                k, v = pair.split('=', 1)
                env_dict[k.strip()] = v.strip()
    entry = MCPServerSettings()
    if kind == 'auto':
        if value.startswith('http://') or value.startswith('https://'):
            kind = 'http'
        elif value in SERVER_RECIPES:
            kind = 'recipe'
        elif '/' in value or '.' in value:
            kind = 'stdio'
        else:
            console.print('[yellow]Could not auto-detect server type[/yellow]')
            raise typer.Exit(1)
    if kind == 'recipe':
        recipe = SERVER_RECIPES.get(value)
        if not recipe:
            console.print(f'[red]Unknown recipe: {value}[/red]')
            console.print('[dim]Use [cyan]mcp-agent server recipes[/cyan] to see available recipes[/dim]')
            raise typer.Exit(1)
        if recipe.get('env_required'):
            missing = []
            import os
            for var in recipe['env_required']:
                if not os.getenv(var) and var not in env_dict:
                    missing.append(var)
            if missing:
                console.print('[yellow]Warning: Required environment variables not set:[/yellow]')
                for var in missing:
                    console.print(f'  • {var}')
                console.print('\n[dim]Add them to mcp_agent.secrets.yaml or set as environment variables[/dim]')
                if not Confirm.ask('Continue anyway?', default=False):
                    raise typer.Exit(0)
        entry.transport = recipe['transport']
        entry.command = recipe.get('command')
        entry.args = recipe.get('args', [])
        entry.env = {**recipe.get('env', {}), **env_dict}
        entry.cwd = recipe.get('cwd')
        srv_name = name or value
        console.print('\n[bold]Adding server from recipe:[/bold]')
        console.print(f'  Name: [cyan]{srv_name}[/cyan]')
        console.print(f'  Description: {recipe.get('description', 'N/A')}')
        console.print(f'  Command: {entry.command} {' '.join(entry.args)}')
    elif kind == 'dxt':
        from pathlib import Path as _Path
        import json as _json
        import zipfile
        dxt_path = _Path(value).expanduser()
        if not dxt_path.exists():
            console.print(f'[red]DXT not found: {dxt_path}[/red]')
            raise typer.Exit(1)
        default_name = name or dxt_path.stem
        base_extract_dir = _Path(extract_to) if extract_to else _Path.cwd() / '.mcp-agent' / 'extensions' / default_name
        manifest_data = None
        manifest_dir = None
        try:
            if dxt_path.is_file() and dxt_path.suffix.lower() == '.dxt':
                base_extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(str(dxt_path), 'r') as zf:
                    zf.extractall(base_extract_dir)
                manifest_dir = base_extract_dir
            else:
                manifest_dir = dxt_path
            manifest_file = manifest_dir / 'manifest.json'
            if not manifest_file.exists():
                console.print('[red]manifest.json not found in extension[/red]')
                raise typer.Exit(1)
            manifest_data = _json.loads(manifest_file.read_text(encoding='utf-8'))
        except Exception as e:
            console.print(f'[red]Failed to process DXT: {e}[/red]')
            raise typer.Exit(1)
        stdio_cfg = manifest_data.get('stdio') if isinstance(manifest_data, dict) else None
        cmd = None
        args = []
        env_vars = {}
        if isinstance(stdio_cfg, dict):
            cmd = stdio_cfg.get('command') or stdio_cfg.get('cmd')
            args = stdio_cfg.get('args') or []
            env_vars = stdio_cfg.get('env') or {}
        else:
            cmd = manifest_data.get('command') if isinstance(manifest_data, dict) else None
            args = (manifest_data.get('args') if isinstance(manifest_data, dict) else []) or []
            env_vars = (manifest_data.get('env') if isinstance(manifest_data, dict) else {}) or {}
        if not cmd:
            console.print('[red]DXT manifest missing stdio command[/red]')
            raise typer.Exit(1)
        entry.transport = 'stdio'
        entry.command = cmd
        entry.args = args
        entry.env = {**env_vars, **env_dict}
        srv_name = name or default_name
        console.print('\n[bold]Adding DXT server:[/bold]')
        console.print(f'  Name: [cyan]{srv_name}[/cyan]')
        console.print(f'  Extracted: {manifest_dir}')
        console.print(f'  Command: {cmd} {' '.join(args)}')
    elif kind in ('http', 'sse'):
        entry.transport = kind
        entry.url = value
        if auth:
            entry.headers = {'Authorization': f'Bearer {auth}'}
        if env_dict:
            entry.env = env_dict
        srv_name = name or value.split('/')[-1].split('?')[0]
    elif kind in ('npx', 'uvx'):
        entry.transport = 'stdio'
        entry.command = kind
        entry.args = [value] if ' ' not in value else value.split()
        entry.env = env_dict
        srv_name = name or value.split('/')[-1]
    else:
        entry.transport = 'stdio'
        parts = value.split()
        entry.command = parts[0]
        entry.args = parts[1:] if len(parts) > 1 else []
        entry.env = env_dict
        entry.cwd = cwd
        srv_name = name or parts[0].split('/')[-1]
    if srv_name in servers and (not force):
        console.print(f"[yellow]Server '{srv_name}' already exists[/yellow]")
        if not Confirm.ask('Overwrite?', default=False):
            raise typer.Exit(0)
    servers[srv_name] = entry
    if write:
        _persist_server_entry(srv_name, entry)
    else:
        console.print(f"[green]✅[/green] Added server '[cyan]{srv_name}[/cyan]' (not persisted)")

@app.command('remove')
def remove_server(name: str=typer.Argument(..., help='Server name to remove'), force: bool=typer.Option(False, '--force', '-f', help='Skip confirmation')) -> None:
    """Remove a server from configuration."""
    import yaml
    cfg_path, data = _load_config_yaml()
    if 'mcp' not in data or 'servers' not in data['mcp']:
        console.print('[yellow]No servers configured[/yellow]')
        raise typer.Exit(1)
    servers = data['mcp']['servers']
    if name not in servers:
        console.print(f"[red]Server '{name}' not found[/red]")
        raise typer.Exit(1)
    if not force:
        server_info = servers[name]
        console.print('[bold]Server to remove:[/bold]')
        console.print(f'  Name: [cyan]{name}[/cyan]')
        console.print(f'  Transport: {server_info.get('transport', 'N/A')}')
        if not Confirm.ask('Remove this server?', default=False):
            raise typer.Exit(0)
    del servers[name]
    if not cfg_path:
        from pathlib import Path as _Path
        cfg_path = _Path('mcp_agent.config.yaml')
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False))
    console.print(f"[green]✅[/green] Removed server '[cyan]{name}[/cyan]'")

@import_app.command('claude')
def import_claude(show_only: bool=typer.Option(False, '--show-only', help='Show servers without importing')) -> None:
    """Import servers from Claude Desktop configuration."""
    from pathlib import Path as _Path
    import platform
    if platform.system() == 'Darwin':
        config_paths = [_Path.home() / 'Library/Application Support/Claude/claude_desktop_config.json']
    elif platform.system() == 'Windows':
        config_paths = [_Path.home() / 'AppData/Roaming/Claude/claude_desktop_config.json']
    else:
        config_paths = [_Path.home() / '.config/Claude/claude_desktop_config.json']
    found = False
    for config_path in config_paths:
        if config_path.exists():
            found = True
            try:
                config = json.loads(config_path.read_text())
                servers = config.get('mcpServers', {})
                if not servers:
                    console.print('[yellow]No servers found in Claude Desktop config[/yellow]')
                    return
                console.print(f'[bold]Found {len(servers)} servers in Claude Desktop:[/bold]\n')
                for name, server_config in servers.items():
                    console.print(f'  • [cyan]{name}[/cyan]')
                    if show_only:
                        console.print(f'    Command: {server_config.get('command', 'N/A')}')
                        if server_config.get('args'):
                            console.print(f'    Args: {' '.join(server_config['args'])}')
                if not show_only:
                    if Confirm.ask('\nImport these servers?', default=True):
                        for name, server_config in servers.items():
                            entry = MCPServerSettings()
                            entry.transport = 'stdio'
                            entry.command = server_config.get('command', '')
                            entry.args = server_config.get('args', [])
                            entry.env = server_config.get('env', {})
                            entry.cwd = server_config.get('cwd')
                            _persist_server_entry(name, entry)
                        console.print(f'\n[green]✅ Imported {len(servers)} servers[/green]')
            except Exception as e:
                console.print(f'[red]Error reading Claude config: {e}[/red]')
    if not found:
        console.print('[yellow]Claude Desktop configuration not found[/yellow]')
        console.print('[dim]Expected locations:[/dim]')
        for path in config_paths:
            console.print(f'  • {path}')

@import_app.command('cursor')
def import_cursor() -> None:
    """Import servers from Cursor configuration."""
    from pathlib import Path as _Path
    candidates = [_Path('.cursor/mcp.json').resolve(), _Path.home() / '.cursor/mcp.json']
    imported_any = False
    for p in candidates:
        if p.exists():
            try:
                console.print(f'[bold]Found Cursor config: {p}[/bold]')
                imported = import_servers_from_mcp_json(p)
                if imported:
                    console.print(f'Importing {len(imported)} servers...')
                    for name, cfg in imported.items():
                        _persist_server_entry(name, cfg)
                        imported_any = True
            except Exception as e:
                console.print(f'[red]Error importing from {p}: {e}[/red]')
                continue
    if imported_any:
        console.print('[green]✅ Successfully imported servers from Cursor[/green]')
    else:
        console.print('[yellow]No Cursor mcp.json found[/yellow]')
        console.print('[dim]Expected locations:[/dim]')
        for path in candidates:
            console.print(f'  • {path}')

@import_app.command('vscode')
def import_vscode() -> None:
    """Import servers from VSCode/Continue configuration."""
    from pathlib import Path as _Path
    candidates = [_Path('.vscode/mcp.json').resolve(), _Path.home() / '.vscode/mcp.json', _Path.cwd() / 'mcp.json']
    imported_any = False
    for p in candidates:
        if p.exists():
            try:
                console.print(f'[bold]Found VSCode config: {p}[/bold]')
                imported = import_servers_from_mcp_json(p)
                if imported:
                    console.print(f'Importing {len(imported)} servers...')
                    for name, cfg in imported.items():
                        _persist_server_entry(name, cfg)
                        imported_any = True
            except Exception as e:
                console.print(f'[red]Error importing from {p}: {e}[/red]')
                continue
    if imported_any:
        console.print('[green]✅ Successfully imported servers from VSCode[/green]')
    else:
        console.print('[yellow]No VSCode mcp.json found[/yellow]')
        console.print('[dim]Expected locations:[/dim]')
        for path in candidates:
            console.print(f'  • {path}')

@import_app.command('mcp-json')
def import_mcp_json(path: str=typer.Argument(..., help='Path to mcp.json')) -> None:
    """Import servers from a generic mcp.json file."""
    from pathlib import Path as _Path
    p = _Path(path).expanduser()
    if not p.exists():
        console.print(f'[red]File not found: {p}[/red]')
        raise typer.Exit(1)
    try:
        servers = import_servers_from_mcp_json(p)
        if not servers:
            console.print('[yellow]No servers found in file[/yellow]')
            raise typer.Exit(1)
        for name, cfg in servers.items():
            _persist_server_entry(name, cfg)
        console.print(f'[green]✅ Imported {len(servers)} servers from {p}[/green]')
    except Exception as e:
        console.print(f'[red]Error importing from {p}: {e}[/red]')
        raise typer.Exit(1)

@import_app.command('dxt')
def import_dxt(path: str=typer.Argument(..., help='Path to .dxt or extracted manifest directory'), name: Optional[str]=typer.Option(None, '--name', '-n', help='Server name'), extract_to: Optional[str]=typer.Option(None, '--extract-to', help='Extraction dir for .dxt (defaults to .mcp-agent/extensions/<name>)')) -> None:
    """Import a Desktop Extension (.dxt) by delegating to 'server add dxt'."""
    try:
        add(kind='dxt', value=path, name=name, write=True, force=False, extract_to=extract_to)
    except typer.Exit as e:
        raise e
    except Exception as e:
        console.print(f'[red]Failed to import DXT: {e}[/red]')
        raise typer.Exit(1)

@import_app.command('smithery')
def import_smithery(url: str=typer.Argument(..., help='Smithery server URL'), name: Optional[str]=typer.Option(None, '--name', '-n', help='Server name')) -> None:
    """Import a server from smithery.ai."""
    import re
    match = re.search('smithery\\.ai/server/([^/]+)', url)
    if not match:
        console.print('[red]Invalid smithery URL[/red]')
        console.print('[dim]Expected format: https://smithery.ai/server/<server-name>[/dim]')
        raise typer.Exit(1)
    server_id = match.group(1)
    srv_name = name or server_id
    if server_id in SERVER_RECIPES:
        console.print(f'[green]Found recipe for {server_id}[/green]')
        add(kind='recipe', value=server_id, name=srv_name, write=True)
    else:
        console.print(f'[yellow]Unknown smithery server: {server_id}[/yellow]')
        console.print('[dim]You may need to manually configure this server[/dim]')
        if 'npx' in url or 'npm' in url:
            console.print(f'\n[dim]Try: mcp-agent server add npx @modelcontextprotocol/{server_id} --name {srv_name}[/dim]')
        else:
            console.print(f'\n[dim]Try: mcp-agent server add uvx {server_id} --name {srv_name}[/dim]')

