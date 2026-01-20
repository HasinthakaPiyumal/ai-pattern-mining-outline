# Cluster 2

def _build_response(*, client: Client, response: httpx.Response) -> Response[Union[Any, SearchResponse]]:
    return Response(status_code=HTTPStatus(response.status_code), content=response.content, headers=response.headers, parsed=_parse_response(client=client, response=response))

def sync_detailed(*, client: Client, json_body: LogEventRequest) -> Response[Any]:
    """
    Args:
        json_body (LogEventRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
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

def sync_detailed(*, client: Client, json_body: LogEventRequest) -> Response[Any]:
    """
    Args:
        json_body (LogEventRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
    """
    kwargs = _get_kwargs(client=client, json_body=json_body)
    response = httpx.request(verify=client.verify_ssl, **kwargs)
    return _build_response(client=client, response=response)

