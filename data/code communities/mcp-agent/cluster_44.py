# Cluster 44

def is_valid_app_id_format(app_id: str) -> bool:
    """Check if the given app ID has a valid format.

    Args:
        app_id: The app ID to validate

    Returns:
        bool: True if the app ID is a valid format, False otherwise
    """
    return app_id.startswith(APP_ID_PREFIX)

def is_valid_server_url_format(server_url: str) -> bool:
    """Check if the given server URL has a valid format.

    Args:
        server_url: The server URL to validate

    Returns:
        bool: True if the server URL is a valid format, False otherwise
    """
    parsed = urlparse(server_url)
    return parsed.scheme in {'http', 'https'} and bool(parsed.netloc)

def is_valid_app_config_id_format(app_config_id: str) -> bool:
    """Check if the given app configuration ID has a valid format.

    Args:
        app_config_id: The app configuration ID to validate

    Returns:
        bool: True if the app configuration ID is a valid format, False otherwise
    """
    return app_config_id.startswith(APP_CONFIG_ID_PREFIX)

