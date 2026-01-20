# Cluster 88

@pytest.fixture
def inspector():
    yield DatetimeInspector()

def test_custom_format_detection(datetime_test_df: pd.DataFrame):
    inspector = DatetimeInspector(user_formats=['%Y-%m-%d %H:%M:%S'])
    inspector.fit(datetime_test_df)
    result = inspector.inspect()
    assert result['datetime_formats']['simple_datetime'] == '%Y-%m-%d %H:%M:%S'
    assert result['datetime_formats']['simple_datetime_2'] == '%d %b %Y'
    assert result['datetime_formats']['date_with_time'] == '%Y-%m-%d %H:%M:%S'
    assert inspector.inspect_level == 20

