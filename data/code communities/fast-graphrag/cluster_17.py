# Cluster 17

class VectorStorage:
    """Vector storage with HNSW."""

    def __init__(self, workspace: Workspace):
        """Create vector storage with HNSW."""
        self.workspace = workspace
        self.vdb = HNSWVectorStorage[int, Any](config=HNSWVectorStorageConfig(ef_construction=96, ef_search=48), namespace=workspace.make_for('vdb'), embedding_dim=1536)
        self.embedder = OpenAIEmbeddingService()
        self.ikv = PickleIndexedKeyValueStorage[int, Any](config=None, namespace=workspace.make_for('ikv'))

    async def upsert(self, ids: Iterable[int], data: Iterable[Tuple[str, str]]) -> None:
        """Add or update embeddings in the storage."""
        data = list(data)
        ids = list(ids)
        embeddings = await self.embedder.encode([f'{t}\n\n{c}' for t, c in data])
        await self.ikv.upsert([int(i) for i in ids], data)
        await self.vdb.upsert(ids, embeddings)

    async def get_context(self, query: str, top_k: int) -> List[Tuple[str, str]]:
        """Get the most similar embeddings to the query."""
        embedding = await self.embedder.encode([query])
        ids, _ = await self.vdb.get_knn(embedding, top_k)
        return [c for c in await self.ikv.get([int(i) for i in np.array(ids).flatten()]) if c is not None]

    async def query_start(self):
        """Load the storage for querying."""
        storages: List[BaseStorage] = [self.ikv, self.vdb]

        def _fn():
            tasks: List[Awaitable[Any]] = []
            for storage_inst in storages:
                tasks.append(storage_inst.query_start())
            return asyncio.gather(*tasks)
        await self.workspace.with_checkpoints(_fn)
        for storage_inst in storages:
            storage_inst.set_in_progress(True)

    async def query_done(self):
        """Finish querying the storage."""
        tasks: List[Awaitable[Any]] = []
        storages: List[BaseStorage] = [self.ikv, self.vdb]
        for storage_inst in storages:
            tasks.append(storage_inst.query_done())
        await asyncio.gather(*tasks)
        for storage_inst in storages:
            storage_inst.set_in_progress(False)

    async def insert_start(self):
        """Prepare the storage for inserting."""
        storages: List[BaseStorage] = [self.ikv, self.vdb]

        def _fn():
            tasks: List[Awaitable[Any]] = []
            for storage_inst in storages:
                tasks.append(storage_inst.insert_start())
            return asyncio.gather(*tasks)
        await self.workspace.with_checkpoints(_fn)
        for storage_inst in storages:
            storage_inst.set_in_progress(True)

    async def insert_done(self):
        """Finish inserting into the storage."""
        tasks: List[Awaitable[Any]] = []
        storages: List[BaseStorage] = [self.ikv, self.vdb]
        for storage_inst in storages:
            tasks.append(storage_inst.insert_done())
        await asyncio.gather(*tasks)
        for storage_inst in storages:
            storage_inst.set_in_progress(False)

class LLMService:
    """Service to interact with the LLM."""
    PROMPT = 'You are a helpful assistant analyzing the given input data to provide an helpful response to the user query.\n\n    # INPUT DATA\n    {context}\n\n    # USER QUERY\n    {query}\n\n    # INSTRUCTIONS\n    Your goal is to provide a response to the user query using the relevant information in the input data.\n    The "Sources" list contains raw text sources to help answer the query. It may contain noisy data, so pay attention when analyzing it.\n\n    Follow these steps:\n    1. Read and understand the user query.\n    2. Carefully analyze all the "Sources" to get detailed information. Information could be scattered across several sources.\n    4. Write the response to the user query based on the information you have gathered. Be very concise and answer the user query directly. If the response cannot be inferred from the input data, just say no relevant information was found. Do not make anything up or add unrelevant information.\n\n    Answer:\n    '

    def __init__(self):
        """Create the LLM service."""
        self.llm = OpenAILLMService()

    async def ask_query(self, context: str, query: str) -> str:
        """Ask a query to the LLM."""
        return (await format_and_send_prompt(prompt=self.PROMPT, llm=self.llm, format_kwargs={'context': context, 'query': query}))[0]

class TestHNSWVectorStorage(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        Index.max_size = 5
        Index.get_current_count.side_effect = lambda: len(Vdb)
        Index.get_max_elements.side_effect = lambda: Index.max_size
        Index.resize_index.side_effect = lambda x: setattr(Index, 'max_size', x)
        Index.add_items.side_effect = lambda data, ids, num_threads: Vdb.update(dict(zip(ids, data)))
        Vdb.clear()
        self.config = HNSWVectorStorageConfig()
        self.storage = HNSWVectorStorage(config=self.config, embedding_dim=128)
        self.storage._index = Index

    @patch('fast_graphrag._storage._vdb_hnswlib.logger')
    async def test_upsert(self, mock_logger):
        ids = [1, 2, 3]
        embeddings = np.random.rand(3, 128).astype(np.float32)
        metadata = [{'meta1': 'data1'}, {'meta2': 'data2'}, {'meta3': 'data3'}]
        await self.storage.upsert(ids, embeddings, metadata)
        self.assertEqual(self.storage.size, 3)
        self.assertEqual(self.storage._metadata[1], {'meta1': 'data1'})
        self.assertEqual(self.storage._metadata[2], {'meta2': 'data2'})
        self.assertEqual(self.storage._metadata[3], {'meta3': 'data3'})

    @patch('fast_graphrag._storage._vdb_hnswlib.logger')
    async def test_upsert_full_index(self, mock_logger):
        ids = [1, 2, 3, 4, 5, 6]
        embeddings = np.random.rand(6, 128).astype(np.float32)
        await self.storage.upsert(ids, embeddings)
        self.assertEqual(self.storage.size, 6)
        self.assertEqual(self.storage.max_size, 10)

    @patch('fast_graphrag._storage._vdb_hnswlib.logger')
    async def test_get_knn_empty_index(self, mock_logger):
        embeddings = np.random.rand(1, 128).astype(np.float32)
        ids, distances = await self.storage.get_knn(embeddings, top_k=5)
        self.assertEqual(ids, [])
        self.assertTrue(np.array_equal(distances, np.array([], dtype=np.float32)))
        mock_logger.info.assert_called_with('Querying knns in empty index.')

    @patch('fast_graphrag._storage._vdb_hnswlib.logger')
    async def test_get_knn(self, mock_logger):
        embeddings = np.random.rand(2, 128).astype(np.float32)
        Vdb.update({i: np.random.rand(128).astype(np.float32) for i in range(10)})
        self.storage._index.knn_query = MagicMock()
        self.storage._index.knn_query.return_value = ([[1, 2, 3], [4, 5, 6]], [[0, 0.2, 0.4], [1.6, 1.8, 2.0]])
        ids, similarity = await self.storage.get_knn(embeddings, top_k=3)
        self.storage._index.knn_query.assert_called_once_with(data=embeddings, k=3, num_threads=self.config.num_threads)
        self.assertEqual(ids, [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_almost_equal(similarity, np.array([[1.0, 0.9, 0.8], [0.2, 0.1, 0.0]], dtype=np.float32))

    @patch('fast_graphrag._storage._vdb_hnswlib.logger')
    async def test_score_all_empty_index(self, mock_logger):
        embeddings = np.random.rand(1, 128).astype(np.float32)
        Vdb.clear()
        scores = await self.storage.score_all(embeddings, top_k=1)
        self.assertEqual(scores.shape, (0, 0))
        mock_logger.warning.assert_called_with('No provided embeddings (128) or empty index (0).')

    @patch('fast_graphrag._storage._vdb_hnswlib.logger')
    async def test_score_all(self, mock_logger):
        embeddings = np.random.rand(2, 128).astype(np.float32)
        Vdb.update({i: np.random.rand(128).astype(np.float32) for i in range(10)})
        self.storage._index.knn_query = MagicMock()
        self.storage._index.knn_query.return_value = ([[1, 2, 3]], [[0.1, 0.2, 0.3]])
        scores = await self.storage.score_all(embeddings, top_k=3)
        self.storage._index.knn_query.assert_called_once_with(data=embeddings, k=3, num_threads=self.config.num_threads)
        self.assertEqual(scores.shape, (1, 10))
        np.testing.assert_almost_equal(scores.data, np.array([0.95, 0.9, 0.85], dtype=np.float32))

    @patch('fast_graphrag._storage._vdb_hnswlib.hnswlib.Index', lambda *args, **kwargs: Index)
    @patch('builtins.open', new_callable=mock_open, read_data=pickle.dumps({'key': 'value'}))
    @patch('fast_graphrag._storage._vdb_hnswlib.logger')
    async def test_insert_start_with_existing_files(self, mock_logger, mock_open):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_load_path.side_effect = lambda x: f'dummy_path_{x}'
        Vdb[1] = True
        await self.storage._insert_start()
        self.storage._index.load_index.assert_called_with('dummy_path_hnsw_index_128.bin', allow_replace_deleted=True)
        self.assertEqual(self.storage._metadata, {'key': 'value'})
        self.assertEqual(self.storage.size, 1)
        mock_logger.debug.assert_called_with("Loaded 1 elements from vectordb storage 'dummy_path_hnsw_index_128.bin'.")

    @patch('fast_graphrag._storage._vdb_hnswlib.hnswlib.Index', lambda *args, **kwargs: Index)
    @patch('fast_graphrag._storage._vdb_hnswlib.logger')
    async def test_insert_start_with_no_files(self, mock_logger):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_load_path.return_value = None
        await self.storage._insert_start()
        self.storage._index.init_index.assert_called_once()
        self.storage._index.set_ef.assert_called_with(self.config.ef_search)
        self.assertEqual(self.storage._metadata, {})
        self.assertEqual(self.storage.size, 0)

    @patch('fast_graphrag._storage._vdb_hnswlib.hnswlib.Index', lambda *args, **kwargs: Index)
    @patch('fast_graphrag._storage._vdb_hnswlib.logger')
    async def test_insert_start_with_no_namespace(self, mock_logger):
        self.storage.namespace = None
        await self.storage._insert_start()
        self.storage._index.init_index.assert_called_with(max_elements=self.storage.INITIAL_MAX_ELEMENTS, ef_construction=self.config.ef_construction, M=self.config.M, allow_replace_deleted=True)
        self.storage._index.set_ef.assert_called_with(self.config.ef_search)
        self.assertEqual(self.storage._metadata, {})
        self.assertEqual(self.storage.size, 0)
        mock_logger.debug.assert_called_with('Creating new volatile vectordb storage.')

    @patch('fast_graphrag._storage._vdb_hnswlib.hnswlib.Index')
    @patch('builtins.open', new_callable=mock_open)
    @patch('fast_graphrag._storage._vdb_hnswlib.logger')
    async def test_insert_done(self, mock_logger, mock_open, mock_index):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_save_path.side_effect = lambda x: f'dummy_path_{x}'
        self.storage._metadata = {'key': 'value'}
        ids = [1, 2, 3]
        embeddings = np.random.rand(3, 128).astype(np.float32)
        metadata = [{'meta1': 'data1'}, {'meta2': 'data2'}, {'meta3': 'data3'}]
        await self.storage.upsert(ids, embeddings, metadata)
        await self.storage._insert_done()
        self.storage._index.save_index.assert_called_with('dummy_path_hnsw_index_128.bin')
        mock_open.assert_called_with('dummy_path_hnsw_metadata.pkl', 'wb')
        mock_logger.debug.assert_called_with("Saving 3 elements from vectordb storage 'dummy_path_hnsw_index_128.bin'.")

    @patch('fast_graphrag._storage._vdb_hnswlib.hnswlib.Index', lambda *args, **kwargs: Index)
    @patch('builtins.open', new_callable=mock_open, read_data=pickle.dumps({'key': 'value'}))
    @patch('fast_graphrag._storage._vdb_hnswlib.logger')
    async def test_query_start_with_existing_files(self, mock_logger, mock_open):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_load_path.side_effect = lambda x: f'dummy_path_{x}'
        Vdb[1] = True
        Vdb[2] = True
        Vdb[3] = True
        await self.storage._query_start()
        self.storage._index.load_index.assert_called_with('dummy_path_hnsw_index_128.bin', allow_replace_deleted=True)
        self.assertEqual(self.storage._metadata, {'key': 'value'})
        mock_logger.debug.assert_called_with("Loaded 3 elements from vectordb storage 'dummy_path_hnsw_index_128.bin'.")

    @patch('fast_graphrag._storage._vdb_hnswlib.hnswlib.Index', lambda *args, **kwargs: Index)
    @patch('fast_graphrag._storage._vdb_hnswlib.logger')
    async def test_query_start_with_no_files(self, mock_logger):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_load_path.return_value = None
        await self.storage._query_start()
        self.assertEqual(self.storage._metadata, {})
        self.assertEqual(self.storage.size, 0)
        mock_logger.warning.assert_called_with("No data file found for vectordb storage 'None'. Loading empty vectordb.")

    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open)
    @patch('fast_graphrag._storage._vdb_hnswlib.logger')
    async def test_insert_start_with_invalid_file(self, mock_logger, mock_open, mock_exists):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_resource_path.side_effect = lambda x: f'dummy_path_{x}'
        with self.assertRaises(InvalidStorageError):
            await self.storage._insert_start()
        mock_logger.error.assert_called_once()

    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open)
    @patch('fast_graphrag._storage._vdb_hnswlib.logger')
    async def test_query_start_with_invalid_file(self, mock_logger, mock_open, mock_exists):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_resource_path.side_effect = lambda x: f'dummy_path_{x}'
        with self.assertRaises(InvalidStorageError):
            await self.storage._query_start()
        mock_logger.error.assert_called_once()

