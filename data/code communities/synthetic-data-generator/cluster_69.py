# Cluster 69

def generator():
    cache_dir = tmp_path / 'ndarrycache'
    for save in [True, False]:
        loader = NDArrayLoader(cache_root=cache_dir, save_to_file=save)
        for ndarray in ndarray_list:
            loader.store(ndarray)
        yield loader
        loader.cleanup()

@pytest.fixture
def ndarray_loaders(tmp_path, ndarray_list):

    def generator():
        cache_dir = tmp_path / 'ndarrycache'
        for save in [True, False]:
            loader = NDArrayLoader(cache_root=cache_dir, save_to_file=save)
            for ndarray in ndarray_list:
                loader.store(ndarray)
            yield loader
            loader.cleanup()
    yield generator()

def test_csv_exporter_generator(csv_exporter: CsvExporter, export_dst):

    def generator():
        yield pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        yield pd.DataFrame({'a': [7, 8, 9], 'b': [10, 11, 12]})
    df_all = pd.concat(generator(), ignore_index=True)
    csv_exporter.write(export_dst, generator())
    pd.testing.assert_frame_equal(df_all, pd.read_csv(export_dst))

