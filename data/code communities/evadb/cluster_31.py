# Cluster 31

class ModulePathTest(unittest.TestCase):

    def test_helper_validates_kwargs(self):
        with self.assertRaises(TypeError):
            validate_kwargs({'a': 1, 'b': 2}, ['a'], 'Invalid keyword argument:')

    def test_should_return_correct_class_for_string(self):
        vl = str_to_class('evadb.readers.decord_reader.DecordReader')
        self.assertEqual(vl, DecordReader)

    def test_should_return_correct_class_for_path(self):
        vl = load_function_class_from_file('evadb/readers/decord_reader.py', 'DecordReader')
        assert vl.__qualname__ == DecordReader.__qualname__

    def test_should_return_correct_class_for_path_without_classname(self):
        vl = load_function_class_from_file('evadb/readers/decord_reader.py')
        assert vl.__qualname__ == DecordReader.__qualname__

    def test_should_raise_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_function_class_from_file('evadb/readers/opencv_reader_abdfdsfds.py')

    def test_should_raise_on_empty_file(self):
        Path('/tmp/empty_file.py').touch()
        with self.assertRaises(ImportError):
            load_function_class_from_file('/tmp/empty_file.py')
        Path('/tmp/empty_file.py').unlink()

    def test_should_raise_if_class_does_not_exists(self):
        with self.assertRaises(ImportError):
            load_function_class_from_file('evadb/utils/s3_utils.py')

    def test_should_raise_if_multiple_classes_exist_and_no_class_mentioned(self):
        with self.assertRaises(ImportError):
            load_function_class_from_file('evadb/utils/generic_utils.py')

    def test_should_use_torch_to_check_if_gpu_is_available(self):
        try:
            import builtins
        except ImportError:
            import __builtin__ as builtins
        realimport = builtins.__import__

        def missing_import(name, globals, locals, fromlist, level):
            if name == 'torch':
                raise ImportError
            return realimport(name, globals, locals, fromlist, level)
        builtins.__import__ = missing_import
        self.assertFalse(is_gpu_available())
        builtins.__import__ = realimport
        is_gpu_available()

    @windows_skip_marker
    def test_should_return_a_random_full_path(self):
        actual = generate_file_path(EvaDB_DATASET_DIR, 'test')
        self.assertTrue(actual.is_absolute())
        self.assertTrue(EvaDB_DATASET_DIR in str(actual.parent))

def validate_kwargs(kwargs, allowed_keys: List[str], required_keys: List[str], error_message='Keyword argument not understood:'):
    """Checks that all keyword arguments are in the set of allowed keys."""
    if required_keys is None:
        required_keys = allowed_keys
    for kwarg in kwargs:
        if kwarg not in allowed_keys:
            raise TypeError(error_message, kwarg)
    missing_keys = [key for key in required_keys if key not in kwargs]
    assert len(missing_keys) == 0, f'Missing required keys, {missing_keys}'

class VectorStoreFactory:

    @staticmethod
    def init_vector_store(vector_store_type: VectorStoreType, index_name: str, **kwargs):
        if vector_store_type == VectorStoreType.FAISS:
            from evadb.third_party.vector_stores.faiss import required_params
            validate_kwargs(kwargs, required_params, required_params)
            return FaissVectorStore(index_name, **kwargs)
        elif vector_store_type == VectorStoreType.QDRANT:
            from evadb.third_party.vector_stores.qdrant import required_params
            validate_kwargs(kwargs, required_params, required_params)
            return QdrantVectorStore(index_name, **kwargs)
        elif vector_store_type == VectorStoreType.PINECONE:
            from evadb.third_party.vector_stores.pinecone import required_params
            validate_kwargs(kwargs, required_params, required_params)
            return PineconeVectorStore(index_name, **kwargs)
        elif vector_store_type == VectorStoreType.CHROMADB:
            from evadb.third_party.vector_stores.chromadb import required_params
            validate_kwargs(kwargs, required_params, required_params)
            return ChromaDBVectorStore(index_name, **kwargs)
        elif vector_store_type == VectorStoreType.WEAVIATE:
            from evadb.third_party.vector_stores.weaviate import required_params
            validate_kwargs(kwargs, required_params, required_params)
            return WeaviateVectorStore(index_name, **kwargs)
        elif vector_store_type == VectorStoreType.MILVUS:
            from evadb.third_party.vector_stores.milvus import allowed_params, required_params
            validate_kwargs(kwargs, allowed_params, required_params)
            return MilvusVectorStore(index_name, **kwargs)
        else:
            raise Exception(f'Vector store {vector_store_type} not supported')

