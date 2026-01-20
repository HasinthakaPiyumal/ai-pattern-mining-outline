# Cluster 9

def test_encoder(data_test: pd.DataFrame):
    for col in ['x', 'y', 'z']:
        nlabel_encoder = NormalizedFrequencyEncoder()
        nlabel_encoder.fit(data_test, col)
        td = nlabel_encoder.transform(data_test.copy())
        rd = nlabel_encoder.reverse_transform(td.copy())
        td.rename(columns={f'{col}.value': f'{col}'}, inplace=True)
        assert (rd[col].sort_values().values == data_test[col].sort_values().values).all()
        assert (td[col] >= -1).all()
        assert (td[col] <= 1).all()
        assert td[col].shape == data_test[col].shape
        assert len(td[col].unique()) == len(data_test[col].unique())

