# Cluster 9

def create_ip_based_limiter(redis_url: Optional[str]=None) -> Limiter:
    """Create limiter that rate limits by IP address."""
    return create_advanced_limiter(redis_url=redis_url, key_func=get_remote_address)

def create_advanced_limiter(redis_url: Optional[str]=None, key_func: Optional[Callable[[Request], str]]=None) -> Limiter:
    """Create a Limiter with advanced configuration.

    Args:
        redis_url: Redis connection URL for distributed rate limiting
        key_func: Custom function to extract rate limiting key from request

    Returns:
        Configured Limiter instance
    """
    if key_func is None:
        key_func = get_remote_address
    storage_uri = redis_url or os.getenv('REDIS_URL')
    if storage_uri and REDIS_AVAILABLE:
        return Limiter(key_func=key_func, storage_uri=storage_uri, headers_enabled=True)
    else:
        return Limiter(key_func=key_func, headers_enabled=True)

def create_user_based_limiter(redis_url: Optional[str]=None) -> Limiter:
    """Create limiter that rate limits by user ID."""
    return create_advanced_limiter(redis_url=redis_url, key_func=get_user_id_key)

def create_api_key_limiter(redis_url: Optional[str]=None) -> Limiter:
    """Create limiter that rate limits by API key."""
    return create_advanced_limiter(redis_url=redis_url, key_func=get_api_key_key)

def get_limiter_from_env() -> Limiter:
    """Create limiter based on environment variables."""
    redis_url = os.getenv('REDIS_URL')
    rate_limit_strategy = os.getenv('RATE_LIMIT_STRATEGY', 'ip').lower()
    if rate_limit_strategy == 'user':
        return create_user_based_limiter(redis_url)
    elif rate_limit_strategy == 'api_key':
        return create_api_key_limiter(redis_url)
    else:
        return create_ip_based_limiter(redis_url)

