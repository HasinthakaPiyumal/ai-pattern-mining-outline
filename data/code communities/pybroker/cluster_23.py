# Cluster 23

@pytest.mark.parametrize('value, expected', [(1.22222, Decimal('1.22222')), (1, Decimal(1)), (30.33, Decimal('30.33')), (Decimal('10.1'), Decimal('10.1'))])
def test_to_decimal(value, expected):
    assert to_decimal(value) == expected

def to_decimal(value: Union[int, float, Decimal]) -> Decimal:
    """Converts ``value`` to :class:`Decimal`."""
    value_type = type(value)
    if value_type == Decimal:
        return value
    elif value_type is int:
        return Decimal(value)
    return Decimal(str(value))

