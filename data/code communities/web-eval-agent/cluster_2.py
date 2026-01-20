# Cluster 2

def should_log_network_request(request) -> bool:
    """Determine if a network request should be logged based on its type and URL.

    Args:
        request: The Playwright request object

    Returns:
        bool: True if the request should be logged, False if it should be filtered out
    """
    url = request.url
    if '/node_modules/' in url:
        return False
    if request.resource_type != 'xhr' and request.resource_type != 'fetch':
        return False
    extensions_to_filter = ['.js', '.css', '.woff', '.woff2', '.ttf', '.eot', '.svg', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.map']
    for ext in extensions_to_filter:
        if url.endswith(ext) or f'{ext}?' in url:
            return False
    return True

def _get_persisted_state() -> Optional[str]:
    """
    Check for and return the path to persisted browser state if it exists.

    Returns:
        Optional[str]: Path to the state file if it exists, None otherwise
    """
    state_file = os.path.expanduser('~/.operative/browser_state/state.json')
    return state_file if os.path.exists(state_file) else None

def get_backend_url(path: str='') -> str:
    """
    Get the backend URL based on environment configuration.
    
    Args:
        path: Optional path to append to the base URL
        
    Returns:
        str: The complete backend URL
    """
    use_local_env = os.getenv('USE_LOCAL_BACKEND')
    use_local = use_local_env is not None and use_local_env.lower() == 'true'
    if use_local:
        base_url = 'http://0.0.0.0:8000'
    else:
        base_url = 'https://operative-backend.onrender.com'
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    if path and path.startswith('/'):
        path = path[1:]
    if path:
        return f'{base_url}/{path}'
    else:
        return base_url

