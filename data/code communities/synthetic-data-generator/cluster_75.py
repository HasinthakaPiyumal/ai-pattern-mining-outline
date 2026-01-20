# Cluster 75

def test_const_inspector(test_const_data: pd.DataFrame):
    inspector = ConstInspector()
    inspector.fit(test_const_data)
    assert inspector.ready
    assert inspector.const_columns
    assert sorted(inspector.inspect()['const_columns']) == sorted(['age', 'fnlwgt', 'workclass'])
    assert inspector.inspect_level == 80

