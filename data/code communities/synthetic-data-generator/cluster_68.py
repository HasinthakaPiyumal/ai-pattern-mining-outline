# Cluster 68

@pytest.fixture
def demo_multi_table_data_metadata_combiner(demo_multi_data_parent_matadata: Metadata, demo_multi_data_child_matadata: Metadata, demo_multi_data_relationship: Relationship):
    metadata_dict = {}
    metadata_dict['store'] = demo_multi_data_parent_matadata
    metadata_dict['train'] = demo_multi_data_child_matadata
    m = MetadataCombiner(named_metadata=metadata_dict, relationships=[demo_multi_data_relationship])
    yield m

