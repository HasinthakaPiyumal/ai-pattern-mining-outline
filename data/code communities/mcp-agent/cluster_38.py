# Cluster 38

def print_servers(servers: List[MCPApp]) -> None:
    """Print a list of deployed servers in a clean, copyable format."""
    console.print(f'\n[bold blue]🖥️  Deployed MCP Servers ({len(servers)})[/bold blue]')
    for i, server in enumerate(servers):
        if i > 0:
            console.print()
        status = _server_status_text(server.appServerInfo.status if server.appServerInfo else 'APP_SERVER_STATUS_OFFLINE')
        console.print(f'[bold cyan]{server.name or 'Unnamed'}[/bold cyan] {status}')
        console.print(f'  App ID: {server.appId}')
        if server.appServerInfo and server.appServerInfo.serverUrl:
            console.print(f'  Server URL: {server.appServerInfo.serverUrl}')
        if server.description:
            console.print(f'  Description: {server.description}')
        console.print(f'  Created: {server.createdAt.strftime('%Y-%m-%d %H:%M:%S')}')
        meta = getattr(server, 'deploymentMetadata', None)
        summary = _format_deploy_meta(meta)
        if summary:
            console.print(f'  Metadata: {summary}')

def _server_status_text(status: str) -> str:
    if status == 'APP_SERVER_STATUS_ONLINE':
        return '🟢 Online'
    elif status == 'APP_SERVER_STATUS_OFFLINE':
        return '🔴 Offline'
    else:
        return '❓ Unknown'

def print_server_configs(server_configs: List[MCPAppConfiguration]) -> None:
    """Print a list of configured servers in a clean, copyable format."""
    console.print(f'\n[bold blue]⚙️  Configured MCP Servers ({len(server_configs)})[/bold blue]')
    for i, config in enumerate(server_configs):
        if i > 0:
            console.print()
        status = _server_status_text(config.appServerInfo.status if config.appServerInfo else 'APP_SERVER_STATUS_OFFLINE')
        console.print(f'[bold cyan]{(config.app.name if config.app else 'Unnamed')}[/bold cyan] {status}')
        console.print(f'  Config ID: {config.appConfigurationId}')
        if config.app:
            console.print(f'  App ID: {config.app.appId}')
            if config.app.description:
                console.print(f'  Description: {config.app.description}')
        if config.appServerInfo and config.appServerInfo.serverUrl:
            console.print(f'  Server URL: {config.appServerInfo.serverUrl}')
        if config.createdAt:
            console.print(f'  Created: {config.createdAt.strftime('%Y-%m-%d %H:%M:%S')}')
        meta = getattr(config.app, 'deploymentMetadata', None) if getattr(config, 'app', None) else None
        summary = _format_deploy_meta(meta)
        if summary:
            console.print(f'  Metadata: {summary}')

def _print_server_text(server: Union[MCPApp, MCPAppConfiguration]) -> None:
    """Print server in text format."""
    if isinstance(server, MCPApp):
        server_type = 'Deployed Server'
        server_id = server.appId
        server_name = server.name
        server_description = server.description
        created_at = server.createdAt
        server_info = server.appServerInfo
    else:
        server_type = 'Configured Server'
        server_id = server.appConfigurationId
        server_name = server.app.name if server.app else 'Unnamed'
        server_description = server.app.description if server.app else None
        created_at = server.createdAt
        server_info = server.appServerInfo
    status_text = '❓ Unknown'
    server_url = 'N/A'
    if server_info:
        status_text = _server_status_text(server_info.status)
        server_url = server_info.serverUrl
    content_lines = [f'Name: [cyan]{server_name}[/cyan]', f'Type: [cyan]{server_type}[/cyan]', f'ID: [cyan]{server_id}[/cyan]', f'Status: {status_text}', f'Server URL: [cyan]{server_url}[/cyan]']
    if server_description:
        content_lines.append(f'Description: [cyan]{server_description}[/cyan]')
    if created_at:
        content_lines.append(f'Created: [cyan]{created_at.strftime('%Y-%m-%d %H:%M:%S')}[/cyan]')
    if isinstance(server, MCPAppConfiguration) and server.app:
        content_lines.extend(['', '[bold]Underlying App:[/bold]', f'  App ID: [cyan]{server.app.appId}[/cyan]', f'  App Name: [cyan]{server.app.name}[/cyan]'])
    console.print(Panel('\n'.join(content_lines), title='Server Description', border_style='blue', expand=False))

def print_apps(apps: List[MCPApp]) -> None:
    """Print a list of deployed apps in a clean, copyable format."""
    console.print(f'\n[bold blue]📦 Deployed MCP Apps ({len(apps)})[/bold blue]')
    for i, app in enumerate(apps):
        if i > 0:
            console.print()
        status = _server_status_text(app.appServerInfo.status if app.appServerInfo else 'APP_SERVER_STATUS_OFFLINE')
        console.print(f'[bold cyan]{app.name or 'Unnamed'}[/bold cyan] {status}')
        console.print(f'  App ID: {app.appId}')
        if app.appServerInfo and app.appServerInfo.serverUrl:
            console.print(f'  Server: {app.appServerInfo.serverUrl}')
        if app.description:
            console.print(f'  Description: {app.description}')
        console.print(f'  Created: {app.createdAt.strftime('%Y-%m-%d %H:%M:%S')}')
        meta = getattr(app, 'deploymentMetadata', None)
        summary = _format_deploy_meta(meta)
        if summary:
            console.print(f'  Metadata: {summary}')

def _format_deploy_meta(meta) -> Optional[str]:
    """Return a one-line deployment summary if metadata is present.

    Accepts either a dict or a JSON string.
    """
    try:
        if meta is None:
            return None
        if isinstance(meta, str):
            import json as _json
            try:
                meta = _json.loads(meta)
            except Exception:
                return None
        if not isinstance(meta, dict):
            return None
        source = meta.get('source')
        if source == 'git' or ('commit' in meta or 'short' in meta):
            short = meta.get('short') or (meta.get('commit') or '')[:7]
            branch = meta.get('branch')
            dirty = meta.get('dirty')
            details = []
            if branch:
                details.append(branch)
            if dirty is True:
                details.append('dirty')
            elif dirty is False:
                details.append('clean')
            base = short or 'unknown'
            return f'{base} ({', '.join(details)})' if details else base
        fp = meta.get('fingerprint') or meta.get('workspace_fingerprint')
        if fp:
            return f'workspace {str(fp)[:12]}'
        return None
    except Exception:
        return None

def print_app_configs(app_configs: List[MCPAppConfiguration]) -> None:
    """Print a list of configured apps in a clean, copyable format."""
    console.print(f'\n[bold blue]⚙️  Configured MCP Apps ({len(app_configs)})[/bold blue]')
    for i, config in enumerate(app_configs):
        if i > 0:
            console.print()
        status = _server_status_text(config.appServerInfo.status if config.appServerInfo else 'APP_SERVER_STATUS_OFFLINE')
        console.print(f'[bold cyan]{(config.app.name if config.app else 'Unnamed')}[/bold cyan] {status}')
        console.print(f'  Config ID: {config.appConfigurationId}')
        if config.app:
            console.print(f'  App ID: {config.app.appId}')
            if config.app.description:
                console.print(f'  Description: {config.app.description}')
        if config.appServerInfo and config.appServerInfo.serverUrl:
            console.print(f'  Server: {config.appServerInfo.serverUrl}')
        if config.createdAt:
            console.print(f'  Created: {config.createdAt.strftime('%Y-%m-%d %H:%M:%S')}')
        meta = getattr(config.app, 'deploymentMetadata', None) if getattr(config, 'app', None) else None
        summary = _format_deploy_meta(meta)
        if summary:
            console.print(f'  Metadata: {summary}')

