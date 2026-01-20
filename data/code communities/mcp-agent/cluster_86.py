# Cluster 86

def _extract_all_audiences(payload: Dict[str, Any]) -> List[str]:
    """Extract all audience values from token payload per RFC 9068."""
    audiences = []
    aud_claim = payload.get('aud')
    if aud_claim:
        if isinstance(aud_claim, str):
            audiences.append(aud_claim)
        elif isinstance(aud_claim, (list, tuple)):
            audiences.extend([str(aud) for aud in aud_claim if aud])
    resource_claim = payload.get('resource')
    if resource_claim:
        if isinstance(resource_claim, str):
            audiences.append(resource_claim)
        elif isinstance(resource_claim, (list, tuple)):
            audiences.extend([str(res) for res in resource_claim if res])
    return list(set(audiences))

def test_audience_extraction_edge_cases():
    """Test audience extraction handles edge cases properly."""
    assert _extract_all_audiences({}) == []
    assert _extract_all_audiences({'aud': None, 'resource': None}) == []
    payload = {'aud': ['', 'https://valid.com', None], 'resource': ['https://another.com', '']}
    audiences = _extract_all_audiences(payload)
    expected = {'https://valid.com', 'https://another.com'}
    assert set(audiences) == expected
    payload = {'aud': ['https://api.com', 'https://api.com'], 'resource': 'https://api.com'}
    audiences = _extract_all_audiences(payload)
    assert audiences == ['https://api.com']

