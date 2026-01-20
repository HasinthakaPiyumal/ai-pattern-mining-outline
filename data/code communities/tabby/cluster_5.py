# Cluster 5

def _get_kwargs(*, client: Client, q: str='get', limit: Union[Unset, None, int]=20, offset: Union[Unset, None, int]=0) -> Dict[str, Any]:
    url = '{}/v1beta/search'.format(client.base_url)
    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()
    params: Dict[str, Any] = {}
    params['q'] = q
    params['limit'] = limit
    params['offset'] = offset
    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}
    return {'method': 'get', 'url': url, 'headers': headers, 'cookies': cookies, 'timeout': client.get_timeout(), 'follow_redirects': client.follow_redirects, 'params': params}

def sync_detailed(*, client: Client, json_body: CompletionRequest) -> Response[Union[Any, CompletionResponse]]:
    """
    Args:
        json_body (CompletionRequest):  Example: {'language': 'python', 'segments': {'prefix':
            'def fib(n):\\n    ', 'suffix': '\\n        return fib(n - 1) + fib(n - 2)'}}.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, CompletionResponse]]
    """
    kwargs = _get_kwargs(client=client, json_body=json_body)
    response = httpx.request(verify=client.verify_ssl, **kwargs)
    return _build_response(client=client, response=response)

def sync_detailed(*, client: Client, json_body: CompletionRequest) -> Response[Union[Any, CompletionResponse]]:
    """
    Args:
        json_body (CompletionRequest):  Example: {'language': 'python', 'segments': {'prefix':
            'def fib(n):\\n    ', 'suffix': '\\n        return fib(n - 1) + fib(n - 2)'}}.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, CompletionResponse]]
    """
    kwargs = _get_kwargs(client=client, json_body=json_body)
    response = httpx.request(verify=client.verify_ssl, **kwargs)
    return _build_response(client=client, response=response)

def sync_detailed(*, client: Client) -> Response[HealthState]:
    """
    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HealthState]
    """
    kwargs = _get_kwargs(client=client)
    response = httpx.request(verify=client.verify_ssl, **kwargs)
    return _build_response(client=client, response=response)

