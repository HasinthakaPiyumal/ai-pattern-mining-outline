# Cluster 93

def setup_api_for_testing(mode: APIMode=APIMode.AUTO, force_check: bool=False) -> Tuple[str, str]:
    """Set up the API for testing.

    Args:
        mode: The API mode to use.
        force_check: Force checking the API connection even if it was already set up.

    Returns:
        Tuple of (api_url, api_key)
    """
    manager = get_api_manager(mode=mode, force_check=force_check)
    return manager.setup()

def get_api_manager(mode: APIMode=APIMode.AUTO, force_check: bool=False) -> APITestManager:
    """Get an APITestManager instance.

    Args:
        mode: The API mode to use.
        force_check: Force checking the API connection even if it was already set up.

    Returns:
        APITestManager instance.
    """
    return APITestManager(mode=mode, force_check=force_check)

