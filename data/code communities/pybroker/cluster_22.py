# Cluster 22

def test_quantize():
    df = pd.DataFrame([[Decimal('0.9999'), Decimal('1.22222')], [Decimal('0.1'), Decimal('0.22')], [Decimal('0.33'), Decimal('0.2222')], [Decimal(1), Decimal('0.1')]], columns=['a', 'b'])
    df['a'] = quantize(df, 'a', True)
    assert (df['a'].values == [1.0, 0.1, 0.33, 1]).all()

def quantize(df: pd.DataFrame, col: str, round: bool) -> pd.Series:
    """Quantizes a :class:`pandas.DataFrame` column by rounding values to the
    nearest cent.

    Returns:
        The quantized column converted to ``float`` values.
    """
    if col not in df.columns:
        raise ValueError(f'Column {col!r} not found in DataFrame.')
    df = df[~df[col].isna()]
    values = df[col]
    if round:
        values = values.apply(lambda d: d.quantize(_CENTS, ROUND_HALF_UP))
    return values.astype(float)

def test_quantize_when_round_is_false():
    df = pd.DataFrame([[Decimal('0.9999'), Decimal('1.22222')], [Decimal('0.1'), Decimal('0.22')], [Decimal('0.33'), Decimal('0.2222')], [Decimal(1), Decimal('0.1')]], columns=['a', 'b'])
    df['a'] = quantize(df, 'a', False)
    assert (df['a'].values == [0.9999, 0.1, 0.33, 1]).all()

def test_quantize_when_column_not_found_then_error():
    df = pd.DataFrame([[Decimal('0.9999'), Decimal('1.22222')], [Decimal('0.1'), Decimal('0.22')], [Decimal('0.33'), Decimal('0.2222')], [Decimal(1), Decimal('0.1')]], columns=['a', 'b'])
    with pytest.raises(ValueError, match=re.escape("Column 'c' not found in DataFrame.")):
        quantize(df, 'c', True)

