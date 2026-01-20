# Cluster 2

@dataclass
class GraphRAG(BaseGraphRAG[TEmbedding, THash, TChunk, TEntity, TRelation, TId]):
    """A class representing a Graph-based Retrieval-Augmented Generation system."""

    @dataclass
    class Config:
        """Configuration for the GraphRAG class."""
        chunking_service_cls: Type[BaseChunkingService[TChunk]] = field(default=DefaultChunkingService)
        information_extraction_service_cls: Type[BaseInformationExtractionService[TChunk, TEntity, TRelation, TId]] = field(default=DefaultInformationExtractionService)
        information_extraction_upsert_policy: BaseGraphUpsertPolicy[TEntity, TRelation, TId] = field(default_factory=lambda: DefaultGraphUpsertPolicy(config=NodeUpsertPolicy_SummarizeDescription.Config(), nodes_upsert_cls=NodeUpsertPolicy_SummarizeDescription, edges_upsert_cls=EdgeUpsertPolicy_UpsertIfValidNodes))
        state_manager_cls: Type[BaseStateManagerService[TEntity, TRelation, THash, TChunk, TId, TEmbedding]] = field(default=DefaultStateManagerService)
        llm_service: BaseLLMService = field(default_factory=lambda: DefaultLLMService())
        embedding_service: BaseEmbeddingService = field(default_factory=lambda: DefaultEmbeddingService())
        graph_storage: BaseGraphStorage[TEntity, TRelation, TId] = field(default_factory=lambda: DefaultGraphStorage(DefaultGraphStorageConfig(node_cls=TEntity, edge_cls=TRelation)))
        entity_storage: DefaultVectorStorage[TIndex, TEmbedding] = field(default_factory=lambda: DefaultVectorStorage(DefaultVectorStorageConfig()))
        chunk_storage: DefaultIndexedKeyValueStorage[THash, TChunk] = field(default_factory=lambda: DefaultIndexedKeyValueStorage(None))
        entity_ranking_policy: RankingPolicy_WithThreshold = field(default_factory=lambda: RankingPolicy_WithThreshold(RankingPolicy_WithThreshold.Config(threshold=0.005)))
        relation_ranking_policy: RankingPolicy_TopK = field(default_factory=lambda: RankingPolicy_TopK(RankingPolicy_TopK.Config(top_k=64)))
        chunk_ranking_policy: RankingPolicy_TopK = field(default_factory=lambda: RankingPolicy_TopK(RankingPolicy_TopK.Config(top_k=8)))
        node_upsert_policy: NodeUpsertPolicy_SummarizeDescription = field(default_factory=lambda: NodeUpsertPolicy_SummarizeDescription())
        edge_upsert_policy: EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM = field(default_factory=lambda: EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM())

        def __post_init__(self):
            """Initialize the GraphRAG Config class."""
            self.entity_storage.embedding_dim = self.embedding_service.embedding_dim
    config: Config = field(default_factory=Config)

    def __post_init__(self):
        """Initialize the GraphRAG class."""
        self.llm_service = self.config.llm_service
        self.embedding_service = self.config.embedding_service
        self.chunking_service = self.config.chunking_service_cls()
        self.information_extraction_service = self.config.information_extraction_service_cls(graph_upsert=self.config.information_extraction_upsert_policy)
        self.state_manager = self.config.state_manager_cls(workspace=Workspace.new(self.working_dir, keep_n=self.n_checkpoints), embedding_service=self.embedding_service, graph_storage=self.config.graph_storage, entity_storage=self.config.entity_storage, chunk_storage=self.config.chunk_storage, entity_ranking_policy=self.config.entity_ranking_policy, relation_ranking_policy=self.config.relation_ranking_policy, chunk_ranking_policy=self.config.chunk_ranking_policy, node_upsert_policy=self.config.node_upsert_policy, edge_upsert_policy=self.config.edge_upsert_policy)

class TestWorkspace(unittest.IsolatedAsyncioTestCase):

    def setUp(self):

        def _(self: Workspace) -> None:
            pass
        Workspace.__del__ = _
        self.test_dir = 'test_workspace'
        self.workspace = Workspace(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_new_workspace(self):
        ws = Workspace.new(self.test_dir)
        self.assertIsInstance(ws, Workspace)
        self.assertEqual(ws.working_dir, self.test_dir)

    def test_get_load_path_no_checkpoint(self):
        self.assertEqual(self.workspace.get_load_path(), None)

    def test_get_save_path_creates_directory(self):
        save_path = self.workspace.get_save_path()
        self.assertTrue(os.path.exists(save_path))

    async def test_with_checkpoint_failures(self):
        for checkpoint in [1, 2, 3]:
            os.makedirs(os.path.join(self.test_dir, str(checkpoint)))
        self.workspace = Workspace(self.test_dir)

        async def sample_fn():
            if '1' not in cast(str, self.workspace.get_load_path()):
                raise Exception('Checkpoint not loaded')
            return 'success'
        result = await self.workspace.with_checkpoints(sample_fn)
        self.assertEqual(result, 'success')
        self.assertEqual(self.workspace.current_load_checkpoint, 1)
        self.assertEqual(self.workspace.failed_checkpoints, ['3', '2'])

    async def test_with_checkpoint_no_failure(self):
        for checkpoint in [1, 2, 3]:
            os.makedirs(os.path.join(self.test_dir, str(checkpoint)))
        self.workspace = Workspace(self.test_dir)

        async def sample_fn():
            return 'success'
        result = await self.workspace.with_checkpoints(sample_fn)
        self.assertEqual(result, 'success')
        self.assertEqual(self.workspace.current_load_checkpoint, 3)
        self.assertEqual(self.workspace.failed_checkpoints, [])

    async def test_with_checkpoint_all_failures(self):
        for checkpoint in [1, 2, 3]:
            os.makedirs(os.path.join(self.test_dir, str(checkpoint)))
        self.workspace = Workspace(self.test_dir)

        async def sample_fn():
            raise Exception('Checkpoint not loaded')
        with self.assertRaises(InvalidStorageError):
            await self.workspace.with_checkpoints(sample_fn)
        self.assertEqual(self.workspace.current_load_checkpoint, None)
        self.assertEqual(self.workspace.failed_checkpoints, ['3', '2', '1'])

    async def test_with_checkpoint_all_failures_accept_none(self):
        for checkpoint in [1, 2, 3]:
            os.makedirs(os.path.join(self.test_dir, str(checkpoint)))
        self.workspace = Workspace(self.test_dir)

        async def sample_fn():
            if self.workspace.get_load_path() is not None:
                raise Exception('Checkpoint not loaded')
        result = await self.workspace.with_checkpoints(sample_fn)
        self.assertEqual(result, None)
        self.assertEqual(self.workspace.current_load_checkpoint, None)
        self.assertEqual(self.workspace.failed_checkpoints, ['3', '2', '1'])

