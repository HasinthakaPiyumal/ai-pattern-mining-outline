# Cluster 46

def _raise_for_unauthenticated(response: httpx.Response):
    """Check if the response indicates an unauthenticated request.
    Raises:
        UnauthenticatedError: If the response status code is 401 or 403.
    """
    if response.status_code == 401 or (response.status_code == 307 and '/api/auth/signin' in response.headers.get('location', '')):
        raise UnauthenticatedError('Unauthenticated request. Please check your API key or login status.')

def _raise_for_status_with_details(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        content_type = response.headers.get('content-type', '')
        if 'application/json' in content_type:
            try:
                error_info = response.json()
                message = error_info.get('error') or error_info.get('message') or str(error_info)
            except Exception:
                message = response.text
        else:
            message = response.text
        raise httpx.HTTPStatusError(f'{exc.response.status_code} Error for {exc.request.url}: {message}', request=exc.request, response=exc.response) from exc

