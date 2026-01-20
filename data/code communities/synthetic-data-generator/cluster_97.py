# Cluster 97

@pytest.fixture
def demo_base_multi_table_synthesizer(demo_multi_table_data_metadata_combiner, demo_multi_table_data_loader):
    yield MultiTableSynthesizerModel(use_dataloader=True, metadata_combiner=demo_multi_table_data_metadata_combiner, tables_data_loader=demo_multi_table_data_loader)

