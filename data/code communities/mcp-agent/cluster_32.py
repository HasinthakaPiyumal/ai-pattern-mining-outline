# Cluster 32

def parse_server_urls(urls_param: str, auth_token: str | None=None) -> List[Tuple[str, Literal['http', 'sse'], str, Dict[str, str] | None]]:
    if not urls_param:
        return []
    url_list = [u.strip() for u in urls_param.split(',') if u.strip()]
    headers = {'Authorization': f'Bearer {auth_token}'} if auth_token else None
    result = []
    for raw in url_list:
        name, transport, normalized = parse_server_url(raw)
        result.append((name, transport, normalized, headers))
    return result

def parse_server_url(url: str) -> Tuple[str, Literal['http', 'sse'], str]:
    """
    Parse a server URL and determine the transport type and normalized URL.

    Returns (server_name, transport_type, normalized_url)
    """
    if not url:
        raise ValueError('URL cannot be empty')
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError(f'URL must be http/https: {url}')
    if not parsed.netloc:
        raise ValueError(f'URL must include a hostname: {url}')
    transport: Literal['http', 'sse'] = 'http'
    if parsed.path.endswith('/sse'):
        transport = 'sse'
        normalized = url
    elif parsed.path.endswith('/mcp'):
        normalized = url
    else:
        base = url if url.endswith('/') else f'{url}/'
        normalized = f'{base}mcp'
    name = generate_server_name(normalized)
    return (name, transport, normalized)

