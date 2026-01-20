# Cluster 37

def _apply_filter(servers: List[Union[MCPApp, MCPAppConfiguration]], filter_expr: str) -> List[Union[MCPApp, MCPAppConfiguration]]:
    """Apply client-side filtering to servers."""
    if not filter_expr:
        return servers
    filtered_servers = []
    filter_lower = filter_expr.lower()
    for server in servers:
        try:
            if isinstance(server, MCPApp):
                name = server.name or ''
                description = server.description or ''
                status = server.appServerInfo.status if server.appServerInfo else 'APP_SERVER_STATUS_OFFLINE'
            elif hasattr(server, 'app'):
                name = server.app.name if server.app else ''
                description = server.app.description if server.app else ''
                status = server.appServerInfo.status if server.appServerInfo else 'APP_SERVER_STATUS_OFFLINE'
            else:
                name = getattr(server, 'name', '') or ''
                description = getattr(server, 'description', '') or ''
                server_info = getattr(server, 'appServerInfo', None)
                status = server_info.status if server_info else 'APP_SERVER_STATUS_OFFLINE'
        except Exception:
            continue
        clean_status = clean_server_status(status).lower()
        if filter_lower in name.lower() or filter_lower in description.lower() or filter_lower in clean_status:
            filtered_servers.append(server)
    return filtered_servers

def clean_server_status(status: str) -> str:
    """Convert server status from API format to clean format.

    Args:
        status: API status string

    Returns:
        Clean status string
    """
    if status == 'APP_SERVER_STATUS_ONLINE':
        return 'active'
    elif status == 'APP_SERVER_STATUS_OFFLINE':
        return 'offline'
    else:
        return 'unknown'

def get_sort_key(server):
    try:
        if isinstance(server, MCPApp):
            name = server.name or ''
            created_at = server.createdAt
            status = server.appServerInfo.status if server.appServerInfo else 'APP_SERVER_STATUS_OFFLINE'
        elif hasattr(server, 'app'):
            name = server.app.name if server.app else ''
            created_at = server.createdAt
            status = server.appServerInfo.status if server.appServerInfo else 'APP_SERVER_STATUS_OFFLINE'
        else:
            name = getattr(server, 'name', '') or ''
            created_at = getattr(server, 'createdAt', None)
            server_info = getattr(server, 'appServerInfo', None)
            status = server_info.status if server_info else 'APP_SERVER_STATUS_OFFLINE'
    except Exception:
        name = ''
        created_at = None
        status = 'APP_SERVER_STATUS_OFFLINE'
    if sort_field_lower == 'name':
        return name.lower()
    elif sort_field_lower in ['created', 'created_at', 'date']:
        return created_at or datetime.min.replace(tzinfo=None if created_at is None else created_at.tzinfo)
    elif sort_field_lower == 'status':
        return clean_server_status(status).lower()
    else:
        return name.lower()

def _server_to_dict(server: MCPApp) -> dict:
    """Convert MCPApp to dictionary."""
    status_raw = server.appServerInfo.status if server.appServerInfo else 'APP_SERVER_STATUS_OFFLINE'
    return {'id': server.appId, 'name': server.name or 'Unnamed', 'description': server.description, 'status': clean_server_status(status_raw), 'server_url': server.appServerInfo.serverUrl if server.appServerInfo else None, 'creator_id': server.creatorId, 'created_at': server.createdAt.isoformat() if server.createdAt else None, 'type': 'deployed', 'deployment_metadata': getattr(server, 'deploymentMetadata', None)}

def _server_config_to_dict(config: MCPAppConfiguration) -> dict:
    """Convert MCPAppConfiguration to dictionary."""
    status_raw = config.appServerInfo.status if config.appServerInfo else 'APP_SERVER_STATUS_OFFLINE'
    return {'config_id': config.appConfigurationId, 'app_id': config.app.appId if config.app else None, 'name': config.app.name if config.app else 'Unnamed', 'description': config.app.description if config.app else None, 'status': clean_server_status(status_raw), 'server_url': config.appServerInfo.serverUrl if config.appServerInfo else None, 'creator_id': config.creatorId, 'created_at': config.createdAt.isoformat() if config.createdAt else None, 'type': 'configured', 'deployment_metadata': getattr(config.app, 'deploymentMetadata', None) if getattr(config, 'app', None) else None}

def _print_server_json(server: Union[MCPApp, MCPAppConfiguration]) -> None:
    """Print server in JSON format."""
    server_data = _server_to_dict(server)
    print(json.dumps(server_data, indent=2, default=str))

def _print_server_yaml(server: Union[MCPApp, MCPAppConfiguration]) -> None:
    """Print server in YAML format."""
    server_data = _server_to_dict(server)
    print(yaml.dump(server_data, default_flow_style=False))

def _server_to_dict(server: Union[MCPApp, MCPAppConfiguration]) -> dict:
    """Convert server to dictionary."""
    if isinstance(server, MCPApp):
        server_type = 'deployed'
        server_id = server.appId
        server_name = server.name
        server_description = server.description
        created_at = server.createdAt
        server_info = server.appServerInfo
        underlying_app = None
    else:
        server_type = 'configured'
        server_id = server.appConfigurationId
        server_name = server.app.name if server.app else 'Unnamed'
        server_description = server.app.description if server.app else None
        created_at = server.createdAt
        server_info = server.appServerInfo
        underlying_app = {'app_id': server.app.appId, 'name': server.app.name} if server.app else None
    status_raw = server_info.status if server_info else 'APP_SERVER_STATUS_OFFLINE'
    server_url = server_info.serverUrl if server_info else None
    data = {'id': server_id, 'name': server_name, 'type': server_type, 'status': clean_server_status(status_raw), 'server_url': server_url, 'description': server_description, 'created_at': created_at.isoformat() if created_at else None}
    if underlying_app:
        data['underlying_app'] = underlying_app
    return data

