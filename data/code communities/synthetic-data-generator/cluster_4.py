# Cluster 4

@pytest.fixture
def ctgan_synthesizer(demo_single_table_data_pos_neg_connector, demo_single_table_data_pos_neg_metadata):
    yield Synthesizer(metadata=demo_single_table_data_pos_neg_metadata, model=CTGANSynthesizerModel(epochs=1), data_connector=demo_single_table_data_pos_neg_connector)

@pytest.fixture
def synthesizer(cacher_kwargs):
    yield Synthesizer(MockModel(), data_connector=MockDataConnector(), raw_data_loaders_kwargs={'cacher_kwargs': cacher_kwargs}, data_processors=[MockDataProcessor()], processed_data_loaders_kwargs={'cacher_kwargs': cacher_kwargs}, metadata=Metadata())

@pytest.fixture
def dummy_single_table_data_connector(dummy_single_table_path):
    yield CsvConnector(path=dummy_single_table_path)

@pytest.fixture
def dummy_single_table_data_loader(dummy_single_table_data_connector, cacher_kwargs):
    d = DataLoader(dummy_single_table_data_connector, cacher_kwargs=cacher_kwargs)
    yield d
    d.finalize()

@pytest.fixture
def demo_single_table_data_connector(demo_single_table_path):
    yield CsvConnector(path=demo_single_table_path)

@pytest.fixture
def demo_single_table_data_loader(demo_single_table_data_connector, cacher_kwargs):
    d = DataLoader(demo_single_table_data_connector, cacher_kwargs=cacher_kwargs)
    yield d
    d.finalize()

@pytest.fixture
def demo_multi_table_data_connector(demo_multi_table_path):
    connector_dict = {}
    for each_table in demo_multi_table_path.keys():
        each_path = demo_multi_table_path[each_table]
        connector_dict[each_table] = CsvConnector(path=each_path)
    yield connector_dict

@pytest.fixture
def demo_multi_table_data_loader(demo_multi_table_data_connector, cacher_kwargs):
    loader_dict = {}
    for each_table in demo_multi_table_data_connector.keys():
        each_connector = demo_multi_table_data_connector[each_table]
        each_d = DataLoader(each_connector, cacher_kwargs=cacher_kwargs)
        loader_dict[each_table] = each_d
    yield loader_dict
    for each_table in demo_multi_table_data_connector.keys():
        demo_multi_table_data_connector[each_table].finalize()

@pytest.fixture
def dataloader(demo_single_table_path, cacher_kwargs):
    d = DataLoader(CsvConnector(path=demo_single_table_path), cacher_kwargs=cacher_kwargs)
    yield d
    d.finalize(clear_cache=True)

@pytest.fixture
def csv_connector(csv_file):
    return CsvConnector(path=csv_file)

@pytest.fixture
def ctgan():
    yield CTGANSynthesizerModel(epochs=1)

def download_file(url, path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as file:
            file.write(response.content)
        print(f'File downloaded successfully to {path}')
    else:
        print(f'Failed to download file from {url}')

