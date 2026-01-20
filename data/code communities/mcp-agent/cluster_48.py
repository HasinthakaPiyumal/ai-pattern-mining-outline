# Cluster 48

def _canonicalize_url(url: str) -> str:
    parsed = URL(url)
    if parsed.scheme not in ('http', 'https'):
        raise OAuthFlowError(f'Unsupported URL scheme for canonicalization: {url}')
    host = parsed.host.lower() if parsed.host else parsed.host
    path = parsed.path.rstrip('/')
    if path == '/':
        path = ''
    canonical = parsed.copy_with(scheme=parsed.scheme, host=host, path=path, query=None, fragment=None)
    return str(canonical)

def _dedupe(sequence: Iterable[OAuthUserIdentity]) -> list[OAuthUserIdentity]:
    seen = set()
    result: list[OAuthUserIdentity] = []
    for identity in sequence:
        if identity is None:
            continue
        key = identity.cache_key
        if key in seen:
            continue
        seen.add(key)
        result.append(identity)
    return result

def normalize_resource(resource: str | None, fallback: str | None) -> str:
    candidate = resource or fallback
    if not candidate:
        raise ValueError('Unable to determine resource identifier for OAuth flow')
    parsed = URL(candidate)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError(f'Unsupported resource scheme: {parsed.scheme}')
    host = parsed.host.lower() if parsed.host else parsed.host
    path = parsed.path.rstrip('/')
    if path == '/':
        path = ''
    canonical = parsed.copy_with(scheme=parsed.scheme, host=host, path=path, query=None, fragment=None)
    return str(canonical)

def select_authorization_server(metadata: ProtectedResourceMetadata, preferred: str | None=None) -> str:
    candidates: List[str] = [str(url) for url in metadata.authorization_servers or []]
    if not candidates:
        raise ValueError('Protected resource metadata did not include authorization servers')
    if preferred:
        preferred_normalized = preferred.rstrip('/')
        candidates_normalized = [c.rstrip('/') for c in candidates]
        for i, candidate_normalized in enumerate(candidates_normalized):
            if candidate_normalized == preferred_normalized:
                return candidates[i]
        logger.warning('Preferred authorization server not listed; falling back to first entry', data={'preferred': preferred, 'candidates': candidates})
    return candidates[0]

def _candidate_resource_metadata_urls(parsed_resource: URL) -> list[str]:
    base = parsed_resource.copy_with(path='', query=None, fragment=None)
    path = parsed_resource.path.lstrip('/')
    candidates = []
    if path:
        candidates.append(str(base.copy_with(path=f'/.well-known/oauth-protected-resource/{path}')))
    candidates.append(str(base.copy_with(path='/.well-known/oauth-protected-resource')))
    seen = set()
    ordered: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered

def _candidate_authorization_metadata_urls(parsed_authorization_server: URL) -> list[str]:
    base = parsed_authorization_server.copy_with(path='', query=None, fragment=None)
    path = parsed_authorization_server.path.lstrip('/')
    candidates = []
    if path:
        candidates.append(str(base.copy_with(path=f'/.well-known/oauth-authorization-server/{path}')))
    candidates.append(str(base.copy_with(path='/.well-known/oauth-authorization-server')))
    seen = set()
    ordered: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered

def test_select_authorization_server_prefers_explicit():
    metadata = ProtectedResourceMetadata(resource='https://example.com', authorization_servers=['https://auth1.example.com', 'https://auth2.example.com'])
    assert select_authorization_server(metadata, 'https://auth2.example.com/') == 'https://auth2.example.com/'
    assert select_authorization_server(metadata, 'https://unknown.example.com') == 'https://auth1.example.com/'

def test_select_authorization_server_with_serialized_config():
    """Test that authorization server selection works after config json serialization.

    When MCPOAuthClientSettings is dumped with mode='json', the authorization_server
    AnyHttpUrl field gets a trailing slash. This test ensures select_authorization_server
    handles this correctly.
    """
    from mcp_agent.config import MCPOAuthClientSettings
    oauth_config = MCPOAuthClientSettings(enabled=True, authorization_server='https://auth.example.com', resource='https://api.example.com', client_id='test_client')
    dumped_config = oauth_config.model_dump(mode='json')
    reloaded_config = MCPOAuthClientSettings(**dumped_config)
    metadata = ProtectedResourceMetadata(resource='https://api.example.com', authorization_servers=['https://auth.example.com', 'https://other-auth.example.com'])
    dumped_metadata = metadata.model_dump(mode='json')
    reloaded_metadata = ProtectedResourceMetadata(**dumped_metadata)
    preferred = str(reloaded_config.authorization_server)
    selected = select_authorization_server(reloaded_metadata, preferred)
    assert selected.rstrip('/') == 'https://auth.example.com'

def test_select_authorization_server_trailing_slash_mismatch():
    """Test trailing slash handling in select_authorization_server with various combinations."""
    metadata1 = ProtectedResourceMetadata(resource='https://api.example.com', authorization_servers=['https://auth.example.com', 'https://other.example.com'])
    selected1 = select_authorization_server(metadata1, 'https://auth.example.com/')
    assert selected1.rstrip('/') == 'https://auth.example.com'
    metadata2 = ProtectedResourceMetadata(resource='https://api.example.com', authorization_servers=['https://auth.example.com/', 'https://other.example.com/'])
    selected2 = select_authorization_server(metadata2, 'https://auth.example.com')
    assert selected2.rstrip('/') == 'https://auth.example.com'

def test_normalize_resource_with_fallback():
    assert normalize_resource('https://example.com/api', None) == 'https://example.com/api'
    assert normalize_resource(None, 'https://fallback.example.com') == 'https://fallback.example.com'
    with pytest.raises(ValueError):
        normalize_resource(None, None)

def test_normalize_resource_canonicalizes_case():
    assert normalize_resource('https://Example.COM/', None) == 'https://example.com'

def test_candidate_resource_metadata_urls():
    parsed = URL('https://api.example.com/mcp')
    urls = _candidate_resource_metadata_urls(parsed)
    assert urls[0].endswith('/.well-known/oauth-protected-resource/mcp')
    assert urls[1].endswith('/.well-known/oauth-protected-resource')

def test_candidate_authorization_metadata_urls():
    parsed = URL('https://auth.example.com/tenant')
    urls = _candidate_authorization_metadata_urls(parsed)
    assert urls[0].endswith('/.well-known/oauth-authorization-server/tenant')
    assert urls[1].endswith('/.well-known/oauth-authorization-server')

