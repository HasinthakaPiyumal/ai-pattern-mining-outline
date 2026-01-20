# Cluster 36

def create_mock_api_client(user_type: str='guest') -> FinceptAPIClient:
    """Create a mock API client for testing purposes"""
    mock_session = {'user_type': user_type, 'authenticated': True, 'api_key': f'mock_key_{user_type}', 'device_id': 'mock_device', 'user_info': {'username': 'mock_user', 'email': 'mock@example.com'} if user_type == 'registered' else {}}
    return create_api_client(mock_session)

def create_api_client(session_data: Dict[str, Any]) -> FinceptAPIClient:
    """Create API client instance from session data"""
    client = FinceptAPIClient(session_data)
    import time
    client._session_start = time.time()
    return client

