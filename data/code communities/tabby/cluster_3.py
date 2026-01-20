# Cluster 3

def _parse_response(*, client: Client, response: httpx.Response) -> Optional[Union[Any, SearchResponse]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = SearchResponse.from_dict(response.json())
        return response_200
    if response.status_code == HTTPStatus.NOT_IMPLEMENTED:
        response_501 = cast(Any, None)
        return response_501
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None

def _build_response(*, client: Client, response: httpx.Response) -> Response[Union[Any, CompletionResponse]]:
    return Response(status_code=HTTPStatus(response.status_code), content=response.content, headers=response.headers, parsed=_parse_response(client=client, response=response))

def _build_response(*, client: Client, response: httpx.Response) -> Response[Any]:
    return Response(status_code=HTTPStatus(response.status_code), content=response.content, headers=response.headers, parsed=_parse_response(client=client, response=response))

def _build_response(*, client: Client, response: httpx.Response) -> Response[HealthState]:
    return Response(status_code=HTTPStatus(response.status_code), content=response.content, headers=response.headers, parsed=_parse_response(client=client, response=response))

def _build_response(*, client: Client, response: httpx.Response) -> Response[Union[Any, SearchResponse]]:
    return Response(status_code=HTTPStatus(response.status_code), content=response.content, headers=response.headers, parsed=_parse_response(client=client, response=response))

def _build_response(*, client: Client, response: httpx.Response) -> Response[Union[Any, CompletionResponse]]:
    return Response(status_code=HTTPStatus(response.status_code), content=response.content, headers=response.headers, parsed=_parse_response(client=client, response=response))

def _build_response(*, client: Client, response: httpx.Response) -> Response[Any]:
    return Response(status_code=HTTPStatus(response.status_code), content=response.content, headers=response.headers, parsed=_parse_response(client=client, response=response))

def _build_response(*, client: Client, response: httpx.Response) -> Response[HealthState]:
    return Response(status_code=HTTPStatus(response.status_code), content=response.content, headers=response.headers, parsed=_parse_response(client=client, response=response))

