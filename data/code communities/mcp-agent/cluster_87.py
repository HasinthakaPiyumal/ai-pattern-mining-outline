# Cluster 87

def test_scope_fingerprint_ordering():
    scopes = ['email', 'profile', 'email']
    fingerprint = scope_fingerprint(scopes)
    assert fingerprint == 'email profile'

def scope_fingerprint(scopes: Iterable[str]) -> str:
    """Return a deterministic fingerprint for a scope list."""
    return ' '.join(sorted({scope.strip() for scope in scopes if scope}))

