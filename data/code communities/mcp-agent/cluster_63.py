# Cluster 63

def _resolve_gateway_url(*, gateway_url: Optional[str]=None, context_gateway_url: Optional[str]=None) -> str:
    """Resolve the base URL for the MCP gateway.

    Precedence:
    1) Explicit override (gateway_url parameter)
    2) Context-provided URL (context_gateway_url)
    3) Environment variable MCP_GATEWAY_URL
    4) Fallback to http://127.0.0.1:8000 (dev default)
    """
    if gateway_url:
        return gateway_url.rstrip('/')
    if context_gateway_url:
        return context_gateway_url.rstrip('/')
    env_url = os.environ.get('MCP_GATEWAY_URL')
    if env_url:
        return env_url.rstrip('/')
    return 'http://127.0.0.1:8000'

