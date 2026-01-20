# Cluster 4

def sync_detailed(*, client: Client, q: str='get', limit: Union[Unset, None, int]=20, offset: Union[Unset, None, int]=0) -> Response[Union[Any, SearchResponse]]:
    """
    Args:
        q (str):  Default: 'get'.
        limit (Union[Unset, None, int]):  Default: 20.
        offset (Union[Unset, None, int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, SearchResponse]]
    """
    kwargs = _get_kwargs(client=client, q=q, limit=limit, offset=offset)
    response = httpx.request(verify=client.verify_ssl, **kwargs)
    return _build_response(client=client, response=response)

def sync(*, client: Client, q: str='get', limit: Union[Unset, None, int]=20, offset: Union[Unset, None, int]=0) -> Optional[Union[Any, SearchResponse]]:
    """
    Args:
        q (str):  Default: 'get'.
        limit (Union[Unset, None, int]):  Default: 20.
        offset (Union[Unset, None, int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, SearchResponse]
    """
    return sync_detailed(client=client, q=q, limit=limit, offset=offset).parsed

def sync(*, client: Client, json_body: CompletionRequest) -> Optional[Union[Any, CompletionResponse]]:
    """
    Args:
        json_body (CompletionRequest):  Example: {'language': 'python', 'segments': {'prefix':
            'def fib(n):\\n    ', 'suffix': '\\n        return fib(n - 1) + fib(n - 2)'}}.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, CompletionResponse]
    """
    return sync_detailed(client=client, json_body=json_body).parsed

def sync(*, client: Client) -> Optional[HealthState]:
    """
    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HealthState
    """
    return sync_detailed(client=client).parsed

def sync(*, client: Client, q: str='get', limit: Union[Unset, None, int]=20, offset: Union[Unset, None, int]=0) -> Optional[Union[Any, SearchResponse]]:
    """
    Args:
        q (str):  Default: 'get'.
        limit (Union[Unset, None, int]):  Default: 20.
        offset (Union[Unset, None, int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, SearchResponse]
    """
    return sync_detailed(client=client, q=q, limit=limit, offset=offset).parsed

def sync(*, client: Client, json_body: CompletionRequest) -> Optional[Union[Any, CompletionResponse]]:
    """
    Args:
        json_body (CompletionRequest):  Example: {'language': 'python', 'segments': {'prefix':
            'def fib(n):\\n    ', 'suffix': '\\n        return fib(n - 1) + fib(n - 2)'}}.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, CompletionResponse]
    """
    return sync_detailed(client=client, json_body=json_body).parsed

def sync(*, client: Client) -> Optional[HealthState]:
    """
    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HealthState
    """
    return sync_detailed(client=client).parsed

