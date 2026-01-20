# Cluster 24

def test_verify_data_source_columns():
    df = pd.DataFrame(columns=['symbol', 'date', 'open', 'high', 'low', 'close'])
    verify_data_source_columns(df)
    assert True

def verify_data_source_columns(df: pd.DataFrame):
    """Verifies that a :class:`pandas.DataFrame` contains all of the
    columns required by a :class:`pybroker.data.DataSource`.
    """
    required_cols = (DataCol.SYMBOL, DataCol.DATE, DataCol.OPEN, DataCol.HIGH, DataCol.LOW, DataCol.CLOSE)
    missing = []
    for col in required_cols:
        if col.value not in df.columns:
            missing.append(col.value)
    if missing:
        raise ValueError(f'DataFrame is missing required columns: {missing!r}')

def test_verify_data_source_columns_when_missing_then_error():
    df = pd.DataFrame(columns=['symbol', 'date', 'open', 'high', 'low'])
    with pytest.raises(ValueError, match=re.escape("DataFrame is missing required columns: ['close']")):
        verify_data_source_columns(df)

