# Cluster 31

def test_param_when_empty():
    assert param('bar') is None

def param(name: str, value: Optional[Any]=_EMPTY_PARAM) -> Optional[Any]:
    """Get or set a global parameter."""
    return StaticScope.instance().param(name, value)

@pytest.mark.parametrize('value', [42, None])
def test_param_when_set_and_get(value):
    param('foo', value)
    assert param('foo') == value

def test_param_when_set_to_none():
    param('baz', 11)
    assert param('baz') == 11
    param('baz', None)
    assert param('baz') is None

