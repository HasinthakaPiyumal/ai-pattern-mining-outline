# Cluster 92

@pytest.fixture
def int_inspector():
    yield RegexInspector(pattern='^[0-9]*$', data_type_name='int')

@pytest.fixture
def empty_inspector():
    yield RegexInspector(pattern='^$', data_type_name='empty_columns', match_percentage=0.88)

def test_parameter_missing_case():
    only_pattern_inspector = RegexInspector(pattern='^[0-9]*$')
    assert only_pattern_inspector.data_type_name == 'regex_^[0-9]*$_columns'
    has_error = False
    try:
        miss_pattern_inspector = RegexInspector(data_type_name='xx')
    except Exception as e:
        has_error = True
        assert type(e) == InspectorInitError
    assert has_error is True
    has_error = False
    try:
        dtype_pattern_inspector = RegexInspector()
    except Exception as e:
        has_error = True
        assert type(e) == InspectorInitError
    assert has_error is True

