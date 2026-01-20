# Cluster 25

def test_verify_date_range_when_invalid_then_error():
    with pytest.raises(ValueError, match='start_date (.*) must be on or before end_date (.*)\\.'):
        verify_date_range('2020-05-01', '2020-04-01')

def verify_date_range(start_date: datetime, end_date: datetime):
    """Verifies date range bounds."""
    if start_date > end_date:
        raise ValueError(f'start_date ({start_date}) must be on or before end_date ({end_date}).')

