# Cluster 15

class SparseEncoder(BaseModel):
    """An encoder that encodes documents into a sparse format."""
    name: str
    type: str = Field(default='base')
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self, docs: List[str]) -> List[SparseEmbedding]:
        """Sparsely encode a list of documents. Documents can be any type, but the encoder must
        be built to handle that data type. Typically, these types are strings or
        arrays representing images.

        :param docs: The documents to encode.
        :type docs: List[Any]
        :return: The encoded documents.
        :rtype: List[SparseEmbedding]
        """
        raise NotImplementedError('Subclasses must implement this method')

    async def acall(self, docs: List[Any]) -> List[SparseEmbedding]:
        """Encode a list of documents asynchronously. Documents can be any type, but the
        encoder must be built to handle that data type. Typically, these types are
        strings or arrays representing images.

        :param docs: The documents to encode.
        :type docs: List[Any]
        :return: The encoded documents.
        :rtype: List[SparseEmbedding]
        """
        raise NotImplementedError('Subclasses must implement this method')

    def _array_to_sparse_embeddings(self, sparse_arrays: np.ndarray) -> List[SparseEmbedding]:
        """Consumes several sparse vectors containing zero-values and returns a compact
        array.

        :param sparse_arrays: The sparse arrays to compact.
        :type sparse_arrays: np.ndarray
        :return: The compact array.
        :rtype: List[SparseEmbedding]
        """
        if hasattr(sparse_arrays, 'to_dense'):
            sparse_arrays = sparse_arrays.to_dense().cpu().numpy()
        if sparse_arrays.ndim != 2:
            raise ValueError(f'Expected a 2D array, got a {sparse_arrays.ndim}D array.')
        coords = np.nonzero(sparse_arrays)
        if coords[0].size == 0:
            return [SparseEmbedding(embedding=np.empty((1, 2)))]
        compact_array = np.array([coords[0], coords[1], sparse_arrays[coords]]).T
        arr_range = range(compact_array[:, 0].max().astype(int) + 1)
        arrs = [compact_array[compact_array[:, 0] == i, :][:, 1:3] for i in arr_range]
        return [SparseEmbedding.from_compact_array(arr) for arr in arrs]

class MockSymmetricSparseEncoder(SparseEncoder):

    def __call__(self, docs: List[str]) -> List[SparseEmbedding]:
        return [SparseEmbedding(embedding=np.array([[0, 0.1], [1, 0.2]])) for _ in docs]

    async def acall(self, docs: List[str]) -> List[SparseEmbedding]:
        return [SparseEmbedding(embedding=np.array([[0, 0.1], [1, 0.2]])) for _ in docs]

class MockAsymmetricSparseEncoder(SparseEncoder, AsymmetricSparseMixin):

    def __call__(self, docs: List[str]) -> List[SparseEmbedding]:
        return [SparseEmbedding(embedding=np.array([[0, 0.1], [1, 0.2]])) for _ in docs]

    async def acall(self, docs: List[str]) -> List[SparseEmbedding]:
        return [SparseEmbedding(embedding=np.array([[0, 0.1], [1, 0.2]])) for _ in docs]

    def encode_queries(self, docs: List[str]) -> List[SparseEmbedding]:
        return [SparseEmbedding(embedding=np.array([[0, 0.1], [1, 0.2]])) for _ in docs]

    def encode_documents(self, docs: List[str]) -> List[SparseEmbedding]:
        return [SparseEmbedding(embedding=np.array([[0, 0.1], [1, 0.2]])) for _ in docs]

    async def aencode_queries(self, docs: List[str]) -> List[SparseEmbedding]:
        return [SparseEmbedding(embedding=np.array([[0, 0.1], [1, 0.2]])) for _ in docs]

    async def aencode_documents(self, docs: List[str]) -> List[SparseEmbedding]:
        return [SparseEmbedding(embedding=np.array([[0, 0.1], [1, 0.2]])) for _ in docs]

def get_test_async_indexes():
    indexes = [LocalIndex]
    if importlib.util.find_spec('qdrant_client') is not None:
        indexes.append(QdrantIndex)
    if importlib.util.find_spec('pinecone') is not None:
        indexes.append(PineconeIndex)
    if importlib.util.find_spec('psycopg') is not None:
        indexes.append(PostgresIndex)
    return indexes

def get_test_indexes():
    indexes = [LocalIndex]
    if importlib.util.find_spec('qdrant_client') is not None:
        indexes.append(QdrantIndex)
    if importlib.util.find_spec('pinecone') is not None:
        indexes.append(PineconeIndex)
    if importlib.util.find_spec('psycopg') is not None:
        indexes.append(PostgresIndex)
    return indexes

def get_test_routers():
    routers = [SemanticRouter, HybridRouter]
    return routers

def get_test_encoders():
    encoders = [OpenAIEncoder]
    if importlib.util.find_spec('cohere') is not None:
        encoders.append(CohereEncoder)
    return encoders

