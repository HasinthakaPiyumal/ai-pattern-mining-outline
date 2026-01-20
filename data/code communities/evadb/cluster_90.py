# Cluster 90

class MilvusVectorStore(VectorStore):

    def __init__(self, index_name: str, **kwargs) -> None:
        self._milvus_uri = kwargs.get('MILVUS_URI')
        if not self._milvus_uri:
            self._milvus_uri = os.environ.get('MILVUS_URI')
        assert self._milvus_uri, 'Please set your Milvus URI in evadb.yml file (third_party, MILVUS_URI) or environment variable (MILVUS_URI).'
        self._milvus_user = kwargs.get('MILVUS_USER')
        if not self._milvus_user:
            self._milvus_user = os.environ.get('MILVUS_USER', '')
        self._milvus_password = kwargs.get('MILVUS_PASSWORD')
        if not self._milvus_password:
            self._milvus_password = os.environ.get('MILVUS_PASSWORD', '')
        self._milvus_db_name = kwargs.get('MILVUS_DB_NAME')
        if not self._milvus_db_name:
            self._milvus_db_name = os.environ.get('MILVUS_DB_NAME', '')
        self._milvus_token = kwargs.get('MILVUS_TOKEN')
        if not self._milvus_token:
            self._milvus_token = os.environ.get('MILVUS_TOKEN', '')
        self._client = get_milvus_client(milvus_uri=self._milvus_uri, milvus_user=self._milvus_user, milvus_password=self._milvus_password, milvus_db_name=self._milvus_db_name, milvus_token=self._milvus_token)
        self._collection_name = index_name

    def create(self, vector_dim: int):
        if self._collection_name in self._client.list_collections():
            self._client.drop_collection(self._collection_name)
        self._client.create_collection(collection_name=self._collection_name, dimension=vector_dim, metric_type='COSINE')

    def add(self, payload: List[FeaturePayload]):
        milvus_data = [{'id': feature_payload.id, 'vector': feature_payload.embedding.reshape(-1).tolist()} for feature_payload in payload]
        ids = [feature_payload.id for feature_payload in payload]
        self._client.delete(collection_name=self._collection_name, pks=ids)
        self._client.insert(collection_name=self._collection_name, data=milvus_data)

    def persist(self):
        self._client.flush(self._collection_name)

    def delete(self) -> None:
        self._client.drop_collection(collection_name=self._collection_name)

    def query(self, query: VectorIndexQuery) -> VectorIndexQueryResult:
        response = self._client.search(collection_name=self._collection_name, data=[query.embedding.reshape(-1).tolist()], limit=query.top_k)[0]
        distances, ids = ([], [])
        for result in response:
            distances.append(result['distance'])
            ids.append(result['id'])
        return VectorIndexQueryResult(distances, ids)

class PineconeVectorStore(VectorStore):

    def __init__(self, index_name: str, **kwargs) -> None:
        try_to_import_pinecone_client()
        global _pinecone_init_done
        self._index_name = index_name.strip().lower()
        self._api_key = kwargs.get('PINECONE_API_KEY')
        if not self._api_key:
            self._api_key = os.environ.get('PINECONE_API_KEY')
        assert self._api_key, 'Please set your `PINECONE_API_KEY` using set command or environment variable (PINECONE_KEY). It can be found at Pinecone Dashboard > API Keys > Value'
        self._environment = kwargs.get('PINECONE_ENV')
        if not self._environment:
            self._environment = os.environ.get('PINECONE_ENV')
        assert self._environment, 'Please set your `PINECONE_ENV` or environment variable (PINECONE_ENV). It can be found Pinecone Dashboard > API Keys > Environment.'
        if not _pinecone_init_done:
            import pinecone
            pinecone.init(api_key=self._api_key, environment=self._environment)
            _pinecone_init_done = True
        self._client = None

    def create(self, vector_dim: int):
        import pinecone
        pinecone.create_index(self._index_name, dimension=vector_dim, metric='cosine')
        logger.warning(f'Created index {self._index_name}. Please note that Pinecone is eventually consistent, hence any additions to the Vector Index may not get immediately reflected in queries.')
        self._client = pinecone.Index(self._index_name)

    def add(self, payload: List[FeaturePayload]):
        self._client.upsert(vectors=[{'id': str(row.id), 'values': row.embedding.reshape(-1).tolist()} for row in payload])

    def delete(self) -> None:
        import pinecone
        pinecone.delete_index(self._index_name)

    def query(self, query: VectorIndexQuery) -> VectorIndexQueryResult:
        import pinecone
        if not self._client:
            self._client = pinecone.Index(self._index_name)
        response = self._client.query(top_k=query.top_k, vector=query.embedding.reshape(-1).tolist())
        distances, ids = ([], [])
        for row in response['matches']:
            distances.append(row['score'])
            ids.append(int(row['id']))
        return VectorIndexQueryResult(distances, ids)

class ChromaDBVectorStore(VectorStore):

    def __init__(self, index_name: str, index_path: str) -> None:
        self._client = get_chromadb_client(index_path)
        self._collection_name = index_name

    def create(self, vector_dim: int):
        self._client.create_collection(name=self._collection_name, metadata={'hnsw:construction_ef': vector_dim, 'hnsw:space': 'cosine'})

    def add(self, payload: List[FeaturePayload]):
        ids = [str(row.id) for row in payload]
        embeddings = [row.embedding.reshape(-1).tolist() for row in payload]
        self._client.get_collection(self._collection_name).add(ids=ids, embeddings=embeddings)

    def delete(self) -> None:
        self._client.delete_collection(name=self._collection_name)

    def query(self, query: VectorIndexQuery) -> VectorIndexQueryResult:
        response = self._client.get_collection(self._collection_name).query(query_embeddings=query.embedding.reshape(-1).tolist(), n_results=query.top_k)
        distances, ids = ([], [])
        if 'ids' in response:
            for id in response['ids'][0]:
                ids.append(int(id))
            for distance in response['distances'][0]:
                distances.append(distance)
        return VectorIndexQueryResult(distances, ids)

class QdrantVectorStore(VectorStore):

    def __init__(self, index_name: str, index_db: str) -> None:
        self._client = get_qdrant_client(index_db)
        self._collection_name = index_name

    def create(self, vector_dim: int):
        from qdrant_client.models import Distance, VectorParams
        self._client.recreate_collection(collection_name=self._collection_name, vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE))

    def add(self, payload: List[FeaturePayload]):
        from qdrant_client.models import Batch
        ids = [int(row.id) for row in payload]
        embeddings = [row.embedding.reshape(-1).tolist() for row in payload]
        self._client.upsert(collection_name=self._collection_name, points=Batch.construct(ids=ids, vectors=embeddings))

    def delete(self) -> None:
        self._client.delete_collection(collection_name=self._collection_name)

    def query(self, query: VectorIndexQuery) -> VectorIndexQueryResult:
        response = self._client.search(collection_name=self._collection_name, query_vector=query.embedding.reshape(-1).tolist(), limit=query.top_k)
        distances, ids = ([], [])
        for point in response:
            distances.append(point.score)
            ids.append(int(point.id))
        return VectorIndexQueryResult(distances, ids)

class WeaviateVectorStore(VectorStore):

    def __init__(self, collection_name: str, **kwargs) -> None:
        try_to_import_weaviate_client()
        global _weaviate_init_done
        self._collection_name = collection_name
        self._api_key = kwargs.get('WEAVIATE_API_KEY')
        if not self._api_key:
            self._api_key = os.environ.get('WEAVIATE_API_KEY')
        assert self._api_key, 'Please set your `WEAVIATE_API_KEY` using set command or environment variable (WEAVIATE_API_KEY). It can be found at the Details tab in WCS Dashboard.'
        self._api_url = kwargs.get('WEAVIATE_API_URL')
        if not self._api_url:
            self._api_url = os.environ.get('WEAVIATE_API_URL')
        assert self._api_url, 'Please set your `WEAVIATE_API_URL` using set command or environment variable (WEAVIATE_API_URL). It can be found at the Details tab in WCS Dashboard.'
        if not _weaviate_init_done:
            import weaviate
            client = weaviate.Client(url=self._api_url, auth_client_secret=weaviate.AuthApiKey(api_key=self._api_key))
            client.schema.get()
            _weaviate_init_done = True
        self._client = client

    def create(self, vectorizer: str='text2vec-openai', properties: list=None, module_config: dict=None):
        properties = properties or []
        module_config = module_config or {}
        collection_obj = {'class': self._collection_name, 'properties': properties, 'vectorizer': vectorizer, 'moduleConfig': module_config}
        if self._client.schema.exists(self._collection_name):
            self._client.schema.delete_class(self._collection_name)
        self._client.schema.create_class(collection_obj)

    def add(self, payload: List[FeaturePayload]) -> None:
        with self._client.batch as batch:
            for item in payload:
                data_object = {'id': item.id, 'vector': item.embedding}
                batch.add_data_object(data_object, self._collection_name)

    def delete(self) -> None:
        self._client.schema.delete_class(self._collection_name)

    def query(self, query: VectorIndexQuery) -> VectorIndexQueryResult:
        response = self._client.query.get(self._collection_name, ['*']).with_near_vector({'vector': query.embedding}).with_limit(query.top_k).do()
        data = response.get('data', {})
        results = data.get('Get', {}).get(self._collection_name, [])
        similarities = [item['_additional']['distance'] for item in results]
        ids = [item['id'] for item in results]
        return VectorIndexQueryResult(similarities, ids)

class FaissVectorStore(VectorStore):

    def __init__(self, index_name: str, index_path: str) -> None:
        try_to_import_faiss()
        self._index_name = index_name
        self._index_path = index_path
        self._index = None
        import faiss
        self._existing_id_set = set([])
        if self._index is None and os.path.exists(self._index_path):
            self._index = faiss.read_index(self._index_path)
            for i in range(self._index.ntotal):
                self._existing_id_set.add(self._index.id_map.at(i))

    def create(self, vector_dim: int):
        import faiss
        self._index = faiss.IndexIDMap2(faiss.IndexHNSWFlat(vector_dim, 32))

    def add(self, payload: List[FeaturePayload]):
        assert self._index is not None, 'Please create an index before adding features.'
        for row in payload:
            embedding = np.array(row.embedding, dtype='float32')
            if len(embedding.shape) != 2:
                embedding = embedding.reshape(1, -1)
            if row.id not in self._existing_id_set:
                self._index.add_with_ids(embedding, np.array([row.id]))

    def persist(self):
        assert self._index is not None, 'Please create an index before calling persist.'
        import faiss
        faiss.write_index(self._index, self._index_path)

    def query(self, query: VectorIndexQuery) -> VectorIndexQueryResult:
        assert self._index is not None, 'Cannot query as index does not exists.'
        embedding = np.array(query.embedding, dtype='float32')
        if len(embedding.shape) != 2:
            embedding = embedding.reshape(1, -1)
        dists, indices = self._index.search(embedding, query.top_k)
        distances, ids = ([], [])
        for dis, idx in zip(dists[0], indices[0]):
            distances.append(dis)
            ids.append(idx)
        return VectorIndexQueryResult(distances, ids)

    def delete(self):
        index_path = Path(self._index_path)
        if index_path.exists():
            index_path.unlink()

