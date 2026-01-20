# Cluster 66

@pytest.fixture
def demo_single_table_data_pos_neg_connector(demo_single_table_data_pos_neg):
    yield DataFrameConnector(df=demo_single_table_data_pos_neg)

def test_dataframe_connector(data_for_test):
    df = data_for_test.copy()
    c = DataFrameConnector(data_for_test)
    assert c._columns() == ['a', 'b']
    assert_frame_equal(c._read(), df)
    assert_frame_equal(c._read(offset=1), df[1:])
    assert_frame_equal(c._read(offset=2), df[2:])
    assert c._read(offset=3) is None
    assert_frame_equal(c._read(offset=0), df)
    assert c._read(offset=5555) is None
    for d, g in zip(c.iter(chunksize=3), [data_for_test]):
        assert_frame_equal(d, g)
    for d, g in zip(c.iter(chunksize=2), [data_for_test[:2], data_for_test[2:]]):
        assert_frame_equal(d, g)

