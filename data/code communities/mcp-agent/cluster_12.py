# Cluster 12

def _get_server_descriptions_as_string(server_registry: ServerRegistry | None, server_names: List[str]) -> str:
    servers = _get_server_descriptions(server_registry, server_names)
    server_strings = []
    for server in servers:
        if 'description' in server:
            server_strings.append(f'{server['name']}: {server['description']}')
        else:
            server_strings.append(f'{server['name']}')
    return '\n'.join(server_strings)

def _get_server_descriptions(server_registry: ServerRegistry | None, server_names: List[str]) -> List:
    servers: List[dict[str, str]] = []
    if server_registry:
        for server_name in server_names:
            config = server_registry.get_server_context(server_name)
            if config:
                servers.append({'name': config.name, 'description': config.description})
            else:
                servers.append({'name': server_name})
    else:
        servers = [{'name': server_name} for server_name in server_names]
    return servers

