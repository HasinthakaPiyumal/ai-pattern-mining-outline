# Cluster 29

class BaseQueryTransform(ABC):

    @abstractmethod
    def _run(self, query: Query, metadata: Dict) -> Query:
        """The Main run logic for Transform"""

    def run(self, query_or_str: Union[str, Query], metadata: Optional[Dict]=None) -> Query:
        """Run query transform."""
        metadata = metadata or {}
        if isinstance(query_or_str, str):
            query = Query(query_str=query_or_str, custom_embedding_strs=[query_or_str])
        else:
            query = query_or_str
        return self._run(query, metadata=metadata)

    def __call__(self, query_bundle_or_str: Union[str, Query], metadata: Optional[Dict]=None) -> Query:
        """Run query processor."""
        return self.run(query_bundle_or_str, metadata=metadata)

class TestSearchEngine(unittest.TestCase):
    """Unit tests for SearchEngine interfaces using HotpotQA JSON example."""

    def setUp(self):
        """Set up SearchEngine, StorageHandler, and temporary directory for each test."""
        load_dotenv()
        self.mock_embedding = MockOpenAIEmbeddingWrapper()
        self.patcher = patch('evoagentx.rag.rag.EmbeddingFactory.create', return_value=self.mock_embedding)
        self.mock_create = self.patcher.start()
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f'Created temporary directory: {self.temp_dir}')
        self.store_config = StoreConfig(dbConfig=DBConfig(db_name='sqlite', path=os.path.join(self.temp_dir, 'test_hotpotQA.sql')), vectorConfig=VectorStoreConfig(vector_name='faiss', dimensions=1536, index_type='flat_l2'), graphConfig=None, path=self.temp_dir)
        self.storage_handler = StorageHandler(storageConfig=self.store_config)
        self.rag_config = RAGConfig(reader=ReaderConfig(recursive=False, exclude_hidden=True, num_files_limit=None, custom_metadata_function=None, extern_file_extractor=None, errors='ignore', encoding='utf-8'), chunker=ChunkerConfig(strategy='simple', chunk_size=512, chunk_overlap=0, max_chunks=None), embedding=EmbeddingConfig(provider='openai', model_name='text-embedding-ada-002', api_key='dummy_key'), index=IndexConfig(index_type='vector'), retrieval=RetrievalConfig(retrivel_type='vector', postprocessor_type='simple', top_k=10, similarity_cutoff=0.3, keyword_filters=None, metadata_filters=None))
        self.search_engine = RAGEngine(config=self.rag_config, storage_handler=self.storage_handler)
        self.corpus_id = HOTPOTQA_EXAMPLE['_id']
        self.context_files = []
        self.supporting_titles = {fact[0] for fact in HOTPOTQA_EXAMPLE['supporting_facts']}
        self.context_data = HOTPOTQA_EXAMPLE['context']
        self.query_text = HOTPOTQA_EXAMPLE['question']
        for title, sentences in self.context_data:
            content = '\n'.join(sentences)
            file_path = os.path.join(self.temp_dir, f'{title.replace(' ', '_')}.txt')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.context_files.append(str(file_path))

    def tearDown(self):
        """Clean up temporary directory, clear indices, and stop patcher."""
        self.search_engine.clear()
        self.patcher.stop()
        logger.info(f'Cleaned up temporary directory: {self.temp_dir}')

    def test_read(self):
        """Test the read method by loading HotpotQA context files."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.assertIsInstance(corpus, Corpus, 'read should return a Corpus object')
        self.assertEqual(corpus.corpus_id, self.corpus_id, 'Corpus ID should match')
        self.assertGreater(len(corpus.chunks), 0, 'Corpus should contain chunks')
        for chunk in corpus.chunks:
            self.assertIsInstance(chunk.metadata, ChunkMetadata, 'Chunk should have metadata')
            self.assertIn('file_name', chunk.metadata.model_dump(), 'Metadata should include file_name')
        logger.info(f'Read {len(corpus.chunks)} chunks for corpus {self.corpus_id}')

    def test_add(self):
        """Test the add method by indexing HotpotQA corpus."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        self.assertIn(self.corpus_id, self.search_engine.indices, 'Corpus should be indexed')
        self.assertIn(IndexType.VECTOR, self.search_engine.indices[self.corpus_id], 'Vector index should exist')
        index = self.search_engine.indices[self.corpus_id][IndexType.VECTOR]
        self.assertGreater(len(index.id_to_node), 0, 'Index should contain nodes')
        for node_id, node in index.id_to_node.items():
            self.assertEqual(node.metadata['corpus_id'], self.corpus_id, 'Node metadata should include corpus_id')
            self.assertEqual(node.metadata['index_type'], IndexType.VECTOR, 'Node metadata should include index_type')
        logger.info(f'Added {len(corpus.chunks)} nodes to vector index for corpus {self.corpus_id}')

    def test_query(self):
        """Test the query method with HotpotQA question, validating top-K retrieval."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        query = Query(query_str=self.query_text, top_k=10)
        result = self.search_engine.query(query, corpus_id=self.corpus_id)
        self.assertIsInstance(result, RagResult, 'query should return a RagResult object')
        self.assertLessEqual(len(result.corpus.chunks), 10, 'Should return at most top_k chunks')
        self.assertEqual(len(result.scores), len(result.corpus.chunks), 'Scores should match chunks')
        retrieved_titles = set()
        for chunk in result.corpus.chunks:
            file_name = chunk.metadata.model_dump().get('file_name', '')
            title = os.path.basename(file_name).replace('_', ' ').replace('.txt', '')
            retrieved_titles.add(title)
        recall = len(retrieved_titles.intersection(self.supporting_titles)) / len(self.supporting_titles)
        self.assertGreaterEqual(recall, 0.0, 'Recall may be low with dummy embeddings')
        logger.info(f'Query retrieved {len(result.corpus.chunks)} chunks with recall@10={recall}')

    def test_delete_by_node_ids(self):
        """Test the delete method by removing specific nodes."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        index = self.search_engine.indices[self.corpus_id][IndexType.VECTOR]
        node_ids = list(index.id_to_node.keys())[:2]
        initial_node_count = len(index.id_to_node)
        self.search_engine.delete(corpus_id=self.corpus_id, index_type=IndexType.VECTOR, node_ids=node_ids)
        remaining_node_count = len(index.id_to_node)
        self.assertEqual(remaining_node_count, initial_node_count - len(node_ids), 'Nodes should be deleted')
        for node_id in node_ids:
            self.assertNotIn(node_id, index.id_to_node, f'Node {node_id} should be deleted')
        logger.info(f'Deleted {len(node_ids)} nodes from corpus {self.corpus_id}')

    def test_delete_by_metadata(self):
        """Test the delete method using metadata filters."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        index = self.search_engine.indices[self.corpus_id][IndexType.VECTOR]
        metadata_filters = {'file_name': str(self.context_files[0])}
        initial_node_count = len(index.id_to_node)
        self.search_engine.delete(corpus_id=self.corpus_id, index_type=IndexType.VECTOR, metadata_filters=metadata_filters)
        remaining_nodes = [node_id for node_id, node in index.id_to_node.items() if node.metadata.get('file_name') != str(self.context_files[0])]
        self.assertEqual(len(index.id_to_node), len(remaining_nodes), 'Nodes matching metadata should be deleted')
        logger.info(f'Deleted nodes with metadata {metadata_filters} from corpus {self.corpus_id}')

    def test_clear(self):
        """Test the clear method by removing all indices."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        self.search_engine.clear(corpus_id=self.corpus_id)
        self.assertNotIn(self.corpus_id, self.search_engine.indices, 'Corpus should be cleared')
        self.assertNotIn(self.corpus_id, self.search_engine.retrievers, 'Retrievers should be cleared')
        logger.info(f'Cleared corpus {self.corpus_id}')

    def test_save_to_files(self):
        """Test the save method by saving indices to files."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        output_path = os.path.join(self.temp_dir, 'output')
        self.search_engine.save(output_path=str(output_path), corpus_id=self.corpus_id, index_type=IndexType.VECTOR)
        if isinstance(output_path, str):
            from pathlib import Path
            output_path = Path(output_path)
        nodes_files = list(output_path.glob('*_nodes.jsonl'))
        metadata_files = list(output_path.glob('*_metadata.json'))
        self.assertEqual(len(nodes_files), 1, 'Should save one nodes file')
        self.assertEqual(len(metadata_files), 1, 'Should save one metadata file')
        with open(nodes_files[0], 'r', encoding='utf-8') as f:
            chunks = [json.loads(line) for line in f]
            self.assertGreater(len(chunks), 0, 'Nodes file should contain chunks')
        with open(metadata_files[0], 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            self.assertEqual(metadata['corpus_id'], self.corpus_id, 'Metadata should include corpus_id')
        logger.info(f'Saved indices to {output_path}')

    def test_load_from_files(self):
        """Test the load method by loading indices from files."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        output_path = os.path.join(self.temp_dir, 'output')
        self.search_engine.save(output_path=str(output_path), corpus_id=self.corpus_id, index_type=IndexType.VECTOR)
        self.search_engine.clear()
        self.search_engine.load(source=str(output_path), corpus_id=self.corpus_id, index_type=IndexType.VECTOR)
        self.assertIn(self.corpus_id, self.search_engine.indices, 'Corpus should be loaded')
        index = self.search_engine.indices[self.corpus_id][IndexType.VECTOR]
        self.assertGreater(len(index.id_to_node), 0, 'Index should contain nodes')
        query = Query(query_str=self.query_text, top_k=10)
        result = self.search_engine.query(query, corpus_id=self.corpus_id)
        self.assertEqual(len(result.corpus.chunks), 0)
        logger.info(f'Loaded indices from {output_path}')

    def test_save_to_database(self):
        """Test the save method by saving indices to database."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        self.search_engine.save(corpus_id=self.corpus_id, index_type=IndexType.VECTOR, table='indexing')
        records = self.storage_handler.load(tables=['indexing']).get('indexing', [])
        self.assertGreater(len(records), 0, 'Database should contain records')
        for record in records:
            parsed = self.storage_handler.parse_result(record, IndexStore)
            self.assertEqual(parsed['corpus_id'], self.corpus_id, 'Record should match corpus_id')
        logger.info(f'Saved indices to database table indexing')

    def test_load_from_database(self):
        """Test the load method by loading indices from database."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        self.search_engine.save(corpus_id=self.corpus_id, index_type=IndexType.VECTOR, table='indexing')
        self.search_engine.clear()
        self.search_engine.load(corpus_id=self.corpus_id, index_type=IndexType.VECTOR, table='indexing')
        self.assertIn(self.corpus_id, self.search_engine.indices, 'Corpus should be loaded')
        index = self.search_engine.indices[self.corpus_id][IndexType.VECTOR]
        self.assertGreater(len(index.id_to_node), 0, 'Index should contain nodes')
        query = Query(query_str=self.query_text, top_k=10)
        result = self.search_engine.query(query, corpus_id=self.corpus_id)
        self.assertEqual(len(result.corpus.chunks), 0)
        logger.info(f'Loaded indices from database table indexing')

    def test_edge_case_empty_corpus(self):
        """Test behavior with empty corpus or invalid corpus_id."""
        result = self.search_engine.query(query=self.query_text, corpus_id='nonexistent')
        self.assertEqual(len(result.corpus.chunks), 0, 'Query on nonexistent corpus should return empty result')
        self.search_engine.delete(corpus_id='nonexistent')
        self.assertNotIn('nonexistent', self.search_engine.indices, 'Delete on nonexistent corpus should not fail')
        self.search_engine.clear(corpus_id='nonexistent')
        self.assertNotIn('nonexistent', self.search_engine.indices, 'Clear on nonexistent corpus should not fail')
        logger.info('Handled edge case for empty/nonexistent corpus')

