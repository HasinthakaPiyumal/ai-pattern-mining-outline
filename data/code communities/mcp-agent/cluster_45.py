# Cluster 45

def attach_url_servers(app: MCPApp, servers: Dict[str, Dict[str, Any]] | None) -> None:
    """Attach URL-based servers (http/sse/streamable_http) to app config."""
    if not servers:
        return
    ensure_mcp_servers(app)
    for name, desc in servers.items():
        settings = MCPServerSettings(transport=desc.get('transport', 'http'), url=desc.get('url'), headers=desc.get('headers'))
        app.context.config.mcp.servers[name] = settings

def ensure_mcp_servers(app: MCPApp) -> None:
    """Ensure app.context.config has mcp servers dict initialized."""
    cfg = app.context.config
    if cfg.mcp is None:
        cfg.mcp = MCPSettings()
    if cfg.mcp.servers is None:
        cfg.mcp.servers = {}

def attach_stdio_servers(app: MCPApp, servers: Dict[str, Dict[str, Any]] | None) -> None:
    """Attach stdio/npx/uvx servers to app config."""
    if not servers:
        return
    ensure_mcp_servers(app)
    for name, desc in servers.items():
        settings = MCPServerSettings(transport='stdio', command=desc.get('command'), args=desc.get('args', []), cwd=desc.get('cwd'))
        app.context.config.mcp.servers[name] = settings

