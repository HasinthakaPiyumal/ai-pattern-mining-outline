# Cluster 5

def test_gaussian_copula(dummy_single_table_metadata, dummy_single_table_data_loader):
    model = GaussianCopulaSynthesizerModel()
    model.fit(dummy_single_table_metadata, dummy_single_table_data_loader)
    sampled_data = model.sample(10)
    original_data = dummy_single_table_data_loader.load_all()
    assert len(sampled_data) == 10
    assert sampled_data.columns.tolist() == original_data.columns.tolist()

