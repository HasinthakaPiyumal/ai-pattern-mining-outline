# Cluster 11

class PineconeRecord(BaseModel):
    id: str = ''
    values: List[float]
    sparse_values: Optional[dict[str, list]] = None
    route: str
    utterance: str
    function_schema: str = '{}'
    metadata: Dict[str, Any] = {}

    def __init__(self, **data):
        """Initialize PineconeRecord.

        :param **data: Keyword arguments to pass to the BaseModel constructor.
        :type **data: dict
        """
        super().__init__(**data)
        clean_route = clean_route_name(self.route)
        utterance_id = hashlib.sha256(self.utterance.encode()).hexdigest()
        self.id = f'{clean_route}#{utterance_id}'
        self.metadata.update({'sr_route': self.route, 'sr_utterance': self.utterance, 'sr_function_schema': self.function_schema})

    def to_dict(self):
        """Convert PineconeRecord to a dictionary.

        :return: Dictionary representation of the PineconeRecord.
        :rtype: dict
        """
        d = {'id': self.id, 'values': self.values, 'metadata': self.metadata}
        if self.sparse_values:
            d['sparse_values'] = self.sparse_values
        return d

def clean_route_name(route_name: str) -> str:
    return route_name.strip().replace(' ', '-')

class PineconeIndex(BaseIndex):
    index_prefix: str = 'semantic-router--'
    api_key: Optional[str] = None
    index_name: str = 'index'
    dimensions: Union[int, None] = None
    metric: str = 'dotproduct'
    cloud: str = 'aws'
    region: str = 'us-east-1'
    host: str = ''
    client: Any = Field(default=None, exclude=True)
    index: Optional[Any] = Field(default=None, exclude=True)
    ServerlessSpec: Any = Field(default=None, exclude=True)
    namespace: Optional[str] = ''
    base_url: Optional[str] = None
    headers: dict[str, str] = {}
    index_host: Optional[str] = 'http://localhost:5080'
    init_async_index: bool = False

    def __init__(self, api_key: Optional[str]=None, index_name: str='index', dimensions: Optional[int]=None, metric: str='dotproduct', cloud: str='aws', region: str='us-east-1', host: str='', namespace: Optional[str]='', base_url: Optional[str]='https://api.pinecone.io', init_async_index: bool=False):
        """Initialize PineconeIndex.

        :param api_key: Pinecone API key.
        :type api_key: Optional[str]
        :param index_name: Name of the index.
        :type index_name: str
        :param dimensions: Dimensions of the index.
        :type dimensions: Optional[int]
        :param metric: Metric of the index.
        :type metric: str
        :param cloud: Cloud provider of the index.
        :type cloud: str
        :param region: Region of the index.
        :type region: str
        :param host: Host of the index.
        :type host: str
        :param namespace: Namespace of the index.
        :type namespace: Optional[str]
        :param base_url: Base URL of the Pinecone API.
        :type base_url: Optional[str]
        :param init_async_index: Whether to initialize the index asynchronously.
        :type init_async_index: bool
        """
        super().__init__()
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        if not self.api_key:
            raise ValueError('Pinecone API key is required.')
        self.headers = {'Api-Key': self.api_key, 'Content-Type': 'application/json', 'User-Agent': 'source_tag=semanticrouter'}
        if base_url is not None or os.getenv('PINECONE_API_BASE_URL'):
            logger.info('Using pinecone remote API.')
            if os.getenv('PINECONE_API_BASE_URL'):
                self.base_url = os.getenv('PINECONE_API_BASE_URL')
            else:
                self.base_url = base_url
        if self.base_url and 'api.pinecone.io' in self.base_url:
            self.headers['X-Pinecone-API-Version'] = '2024-07'
        self.index_name = index_name
        self.dimensions = dimensions
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.host = host
        if namespace == 'sr_config':
            raise ValueError("Namespace 'sr_config' is reserved for internal use.")
        self.namespace = namespace
        self.type = 'pinecone'
        logger.warning('Default region changed from us-west-2 to us-east-1 in v0.1.0.dev6')
        self.client = self._initialize_client(api_key=self.api_key)
        if not init_async_index:
            self.index = self._init_index()

    def _initialize_client(self, api_key: Optional[str]=None):
        """Initialize the Pinecone client.

        :param api_key: Pinecone API key.
        :type api_key: Optional[str]
        :return: Pinecone client.
        :rtype: Pinecone
        """
        try:
            from pinecone import Pinecone, ServerlessSpec
            self.ServerlessSpec = ServerlessSpec
        except ImportError:
            raise ImportError("Please install pinecone-client to use PineconeIndex. You can install it with: `pip install 'semantic-router[pinecone]'`")
        pinecone_args = {'api_key': api_key, 'source_tag': 'semanticrouter', 'host': self.base_url}
        if self.namespace:
            pinecone_args['namespace'] = self.namespace
        return Pinecone(**pinecone_args)

    def _calculate_index_host(self):
        """Calculate the index host. Used to differentiate between normal
        Pinecone and Pinecone Local instance.

        :return: None
        :rtype: None
        """
        if self.index_host and self.base_url:
            if 'api.pinecone.io' in self.base_url:
                if not self.index_host.startswith('http'):
                    self.index_host = f'https://{self.index_host}'
            elif 'http' not in self.index_host:
                self.index_host = f'http://{self.base_url.split(':')[-2].strip('/')}:{self.index_host.split(':')[-1]}'
            elif not self.index_host.startswith('http://'):
                if 'localhost' in self.index_host:
                    self.index_host = f'http://{self.base_url.split(':')[-2].strip('/')}:{self.index_host.split(':')[-1]}'
                else:
                    self.index_host = f'http://{self.index_host}'

    def _init_index(self, force_create: bool=False) -> Union[Any, None]:
        """Initializing the index can be done after the object has been created
        to allow for the user to set the dimensions and other parameters.

        If the index doesn't exist and the dimensions are given, the index will
        be created. If the index exists, it will be returned. If the index doesn't
        exist and the dimensions are not given, the index will not be created and
        None will be returned.

        :param force_create: If True, the index will be created even if the
            dimensions are not given (which will raise an error).
        :type force_create: bool, optional
        """
        dimensions_given = self.dimensions is not None
        if self.index is None:
            index_exists = self.client.has_index(name=self.index_name)
            if dimensions_given and (not index_exists):
                self.client.create_index(name=self.index_name, dimension=self.dimensions, metric=self.metric, spec=self.ServerlessSpec(cloud=self.cloud, region=self.region))
                while not self.client.describe_index(self.index_name).status['ready']:
                    time.sleep(0.2)
                index = self.client.Index(self.index_name)
                self.index = index
                time.sleep(0.2)
            elif index_exists:
                self.index_host = self.client.describe_index(self.index_name).host
                self._calculate_index_host()
                index = self.client.Index(self.index_name, host=self.index_host)
                self.index = index
                self.dimensions = index.describe_index_stats()['dimension']
            elif force_create and (not dimensions_given):
                raise ValueError('Cannot create an index without specifying the dimensions.')
            else:
                logger.warning(f'Index could not be initialized. Init parameters: self.index_name={self.index_name!r}, self.dimensions={self.dimensions!r}, self.metric={self.metric!r}, self.cloud={self.cloud!r}, self.region={self.region!r}, self.host={self.host!r}, self.namespace={self.namespace!r}, force_create={force_create!r}')
                index = None
        else:
            index = self.index
        if self.index is not None and self.host == '':
            self.index_host = self.client.describe_index(self.index_name).host
            if self.index_host and self.base_url:
                self._calculate_index_host()
                index = self.client.Index(self.index_name, host=self.index_host)
                self.host = self.index_host
        return index

    async def _init_async_index(self, force_create: bool=False):
        """Initializing the index can be done after the object has been created
        to allow for the user to set the dimensions and other parameters.

        If the index doesn't exist and the dimensions are given, the index will
        be created. If the index exists, it will be returned. If the index doesn't
        exist and the dimensions are not given, the index will not be created and
        None will be returned.

        This method is used to initialize the index asynchronously.

        :param force_create: If True, the index will be created even if the
            dimensions are not given (which will raise an error).
        :type force_create: bool, optional
        """
        index_stats = None
        if self.dimensions is None:
            indexes = await self._async_list_indexes()
            index_names = [i['name'] for i in indexes['indexes']]
            index_exists = self.index_name in index_names
            if index_exists:
                index_stats = await self._async_describe_index(self.index_name)
                self.dimensions = index_stats['dimension']
            elif index_exists and (not force_create):
                logger.warning(f'Index could not be initialized. Init parameters: self.index_name={self.index_name!r}, self.dimensions={self.dimensions!r}, self.metric={self.metric!r}, self.cloud={self.cloud!r}, self.region={self.region!r}, self.host={self.host!r}, self.namespace={self.namespace!r}, force_create={force_create!r}')
            elif force_create:
                raise ValueError(f'Index could not be initialized. Init parameters: self.index_name={self.index_name!r}, self.dimensions={self.dimensions!r}, self.metric={self.metric!r}, ')
            else:
                raise NotImplementedError('Unexpected init conditions. Please report this issue in GitHub.')
        if self.dimensions:
            indexes = await self._async_list_indexes()
            index_names = [i['name'] for i in indexes['indexes']]
            index_exists = self.index_name in index_names
            if not index_exists:
                index_stats = await self._async_describe_index(self.index_name)
                index_status = index_stats.get('status', {})
                index_ready = index_status.get('ready', False) if isinstance(index_status, dict) else False
                if index_ready == 'true' or (isinstance(index_ready, bool) and index_ready):
                    self.index_host = index_stats['host']
                    self._calculate_index_host()
                    self.host = self.index_host
                    return index_stats
                else:
                    await self._async_create_index(name=self.index_name, dimension=self.dimensions, metric=self.metric, cloud=self.cloud, region=self.region)
                    index_ready = 'false'
                    while not (index_ready == 'true' or (isinstance(index_ready, bool) and index_ready)):
                        index_stats = await self._async_describe_index(self.index_name)
                        index_status = index_stats.get('status', {})
                        index_ready = index_status.get('ready', False) if isinstance(index_status, dict) else False
                        await asyncio.sleep(0.1)
                    self.index_host = index_stats['host']
                    self._calculate_index_host()
                    self.host = self.index_host
                    return index_stats
            else:
                index_stats = await self._async_describe_index(self.index_name)
                self.index_host = index_stats['host']
                self._calculate_index_host()
                self.host = self.index_host
                return index_stats
        if index_stats:
            self.index_host = index_stats['host']
            self._calculate_index_host()
            self.host = self.index_host
        else:
            self.host = ''

    def _batch_upsert(self, batch: List[Dict]):
        """Helper method for upserting a single batch of records.

        :param batch: The batch of records to upsert.
        :type batch: List[Dict]
        """
        if self.index is not None:
            self.index.upsert(vectors=batch, namespace=self.namespace)
        else:
            raise ValueError('Index is None, could not upsert.')

    def add(self, embeddings: List[List[float]], routes: List[str], utterances: List[str], function_schemas: Optional[Optional[List[Dict[str, Any]]]]=None, metadata_list: List[Dict[str, Any]]=[], batch_size: int=100, sparse_embeddings: Optional[Optional[List[SparseEmbedding]]]=None, **kwargs):
        """Add vectors to Pinecone in batches.

        :param embeddings: List of embeddings to upsert.
        :type embeddings: List[List[float]]
        :param routes: List of routes to upsert.
        :type routes: List[str]
        :param utterances: List of utterances to upsert.
        :type utterances: List[str]
        :param function_schemas: List of function schemas to upsert.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: List of metadata to upsert.
        :type metadata_list: List[Dict[str, Any]]
        :param batch_size: Number of vectors to upsert in a single batch.
        :type batch_size: int, optional
        :param sparse_embeddings: List of sparse embeddings to upsert.
        :type sparse_embeddings: Optional[List[SparseEmbedding]]
        """
        if self.index is None:
            self.dimensions = self.dimensions or len(embeddings[0])
            self.index = self._init_index(force_create=True)
        vectors_to_upsert = build_records(embeddings=embeddings, routes=routes, utterances=utterances, function_schemas=function_schemas, metadata_list=metadata_list, sparse_embeddings=sparse_embeddings)
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self._batch_upsert(batch)

    async def aadd(self, embeddings: List[List[float]], routes: List[str], utterances: List[str], function_schemas: Optional[Optional[List[Dict[str, Any]]]]=None, metadata_list: List[Dict[str, Any]]=[], batch_size: int=100, sparse_embeddings: Optional[Optional[List[SparseEmbedding]]]=None, **kwargs):
        """Add vectors to Pinecone in batches.

        :param embeddings: List of embeddings to upsert.
        :type embeddings: List[List[float]]
        :param routes: List of routes to upsert.
        :type routes: List[str]
        :param utterances: List of utterances to upsert.
        :type utterances: List[str]
        :param function_schemas: List of function schemas to upsert.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: List of metadata to upsert.
        :type metadata_list: List[Dict[str, Any]]
        :param batch_size: Number of vectors to upsert in a single batch.
        :type batch_size: int, optional
        :param sparse_embeddings: List of sparse embeddings to upsert.
        :type sparse_embeddings: Optional[List[SparseEmbedding]]
        """
        vectors_to_upsert = build_records(embeddings=embeddings, routes=routes, utterances=utterances, function_schemas=function_schemas, metadata_list=metadata_list, sparse_embeddings=sparse_embeddings)
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            await self._async_upsert(vectors=batch, namespace=self.namespace or '')

    def _remove_and_sync(self, routes_to_delete: dict):
        """Remove specified routes from index if they exist.

        :param routes_to_delete: Routes to delete.
        :type routes_to_delete: dict
        """
        for route, utterances in routes_to_delete.items():
            remote_routes = self._get_routes_with_ids(route_name=route)
            ids_to_delete = [r['id'] for r in remote_routes if (r['route'], r['utterance']) in zip([route] * len(utterances), utterances)]
            if ids_to_delete and self.index:
                self.index.delete(ids=ids_to_delete, namespace=self.namespace)

    async def _async_remove_and_sync(self, routes_to_delete: dict):
        """Remove specified routes from index if they exist.

        This method is asyncronous.

        :param routes_to_delete: Routes to delete.
        :type routes_to_delete: dict
        """
        for route, utterances in routes_to_delete.items():
            remote_routes = await self._async_get_routes_with_ids(route_name=route)
            ids_to_delete = [r['id'] for r in remote_routes if (r['route'], r['utterance']) in zip([route] * len(utterances), utterances)]
            if ids_to_delete and self.index:
                await self._async_delete(ids=ids_to_delete, namespace=self.namespace or '')

    def _get_route_ids(self, route_name: str):
        """Get the IDs of the routes in the index.

        :param route_name: Name of the route to get the IDs for.
        :type route_name: str
        :return: List of IDs of the routes.
        :rtype: list[str]
        """
        clean_route = clean_route_name(route_name)
        ids, _ = self._get_all(prefix=f'{clean_route}#')
        return ids

    async def _async_get_route_ids(self, route_name: str):
        """Get the IDs of the routes in the index.

        :param route_name: Name of the route to get the IDs for.
        :type route_name: str
        :return: List of IDs of the routes.
        :rtype: list[str]
        """
        clean_route = clean_route_name(route_name)
        ids, _ = await self._async_get_all(prefix=f'{clean_route}#')
        return ids

    def _get_routes_with_ids(self, route_name: str):
        """Get the routes with their IDs from the index.

        :param route_name: Name of the route to get the routes with their IDs for.
        :type route_name: str
        :return: List of routes with their IDs.
        :rtype: list[dict]
        """
        clean_route = clean_route_name(route_name)
        ids, metadata = self._get_all(prefix=f'{clean_route}#', include_metadata=True)
        route_tuples = []
        for id, data in zip(ids, metadata):
            route_tuples.append({'id': id, 'route': data['sr_route'], 'utterance': data['sr_utterance']})
        return route_tuples

    async def _async_get_routes_with_ids(self, route_name: str):
        """Get the routes with their IDs from the index.

        :param route_name: Name of the route to get the routes with their IDs for.
        :type route_name: str
        :return: List of routes with their IDs.
        :rtype: list[dict]
        """
        clean_route = clean_route_name(route_name)
        ids, metadata = await self._async_get_all(prefix=f'{clean_route}#', include_metadata=True)
        route_tuples = []
        for id, data in zip(ids, metadata):
            route_tuples.append({'id': id, 'route': data['sr_route'], 'utterance': data['sr_utterance']})
        return route_tuples

    def _get_all(self, prefix: Optional[str]=None, include_metadata: bool=False):
        """Retrieves all vector IDs from the Pinecone index using pagination.

        :param prefix: The prefix to filter the vectors by.
        :type prefix: Optional[str]
        :param include_metadata: Whether to include metadata in the response.
        :type include_metadata: bool
        :return: A tuple containing a list of vector IDs and a list of metadata dictionaries.
        :rtype: tuple[list[str], list[dict]]
        """
        if self.index is None:
            raise ValueError('Index is None, could not retrieve vector IDs.')
        all_vector_ids = []
        metadata = []
        for ids in self.index.list(prefix=prefix, namespace=self.namespace):
            all_vector_ids.extend(ids)
            if include_metadata:
                for id in ids:
                    res_meta = self.index.fetch(ids=[id], namespace=self.namespace) if self.index else {}
                    metadata.extend([x['metadata'] for x in res_meta['vectors'].values()])
        return (all_vector_ids, metadata)

    def delete(self, route_name: str) -> list[str]:
        """Delete specified route from index if it exists. Returns the IDs of the vectors
        deleted.

        :param route_name: Name of the route to delete.
        :type route_name: str
        :return: List of IDs of the vectors deleted.
        :rtype: list[str]
        """
        route_vec_ids = self._get_route_ids(route_name=route_name)
        if self.index is not None:
            logger.info('index is not None, deleting...')
            if self.base_url and 'api.pinecone.io' in self.base_url:
                self.index.delete(ids=route_vec_ids, namespace=self.namespace)
            else:
                response = requests.post(f'{self.index_host}/vectors/delete', json=DeleteRequest(ids=route_vec_ids, delete_all=True, namespace=self.namespace).model_dump(exclude_none=True), timeout=10)
                if response.status_code == 200:
                    logger.info(f'Deleted {len(route_vec_ids)} vectors from index {self.index_name}.')
                else:
                    error_message = response.text
                    raise Exception(f'Failed to delete vectors: {response.status_code} : {error_message}')
            return route_vec_ids
        else:
            raise ValueError('Index is None, could not delete.')

    async def adelete(self, route_name: str) -> list[str]:
        """Asynchronously delete specified route from index if it exists. Returns the IDs
        of the vectors deleted.

        :param route_name: Name of the route to delete.
        :type route_name: str
        :return: List of IDs of the vectors deleted.
        :rtype: list[str]
        """
        if not await self.ais_ready():
            raise ValueError('Async index is not initialized.')
        route_vec_ids = await self._async_get_route_ids(route_name=route_name)
        await self._async_delete(ids=route_vec_ids, namespace=self.namespace or '')
        return route_vec_ids

    def delete_all(self):
        """Delete all routes from index if it exists.

        :return: None
        :rtype: None
        """
        if self.index is not None:
            self.index.delete(delete_all=True, namespace=self.namespace)
        else:
            raise ValueError('Index is None, could not delete.')

    def describe(self) -> IndexConfig:
        """Describe the index.

        :return: IndexConfig
        :rtype: IndexConfig
        """
        if self.index is not None:
            stats = self.index.describe_index_stats()
            return IndexConfig(type=self.type, dimensions=stats['dimension'], vectors=stats['namespaces'][self.namespace]['vector_count'])
        else:
            return IndexConfig(type=self.type, dimensions=self.dimensions or 0, vectors=0)

    def is_ready(self) -> bool:
        """Checks if the index is ready to be used.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        return self.index is not None

    def query(self, vector: np.ndarray, top_k: int=5, route_filter: Optional[List[str]]=None, sparse_vector: dict[int, float] | SparseEmbedding | None=None) -> Tuple[np.ndarray, List[str]]:
        """Search the index for the query vector and return the top_k results.

        :param vector: The query vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of top results to return, defaults to 5.
        :type top_k: int, optional
        :param route_filter: A list of route names to filter the search results, defaults to None.
        :type route_filter: Optional[List[str]], optional
        :param sparse_vector: An optional sparse vector to include in the query.
        :type sparse_vector: Optional[SparseEmbedding]
        :param kwargs: Additional keyword arguments for the query, including sparse_vector.
        :type kwargs: Any
        :return: A tuple containing an array of scores and a list of route names.
        :rtype: Tuple[np.ndarray, List[str]]
        :raises ValueError: If the index is not populated.
        """
        if self.index is None:
            raise ValueError('Index is not populated.')
        query_vector_list = vector.tolist()
        if route_filter is not None:
            filter_query = {'sr_route': {'$in': route_filter}}
        else:
            filter_query = None
        if sparse_vector is not None:
            logger.error(f'sparse_vector exists:{sparse_vector}')
            if isinstance(sparse_vector, dict):
                sparse_vector = SparseEmbedding.from_dict(sparse_vector)
            if isinstance(sparse_vector, SparseEmbedding):
                sparse_vector = sparse_vector.to_pinecone()
        try:
            results = self.index.query(vector=[query_vector_list], sparse_vector=sparse_vector, top_k=top_k, filter=filter_query, include_metadata=True, namespace=self.namespace)
        except Exception:
            logger.error('retrying query with vector as str')
            results = self.index.query(vector=query_vector_list, sparse_vector=sparse_vector, top_k=top_k, filter=filter_query, include_metadata=True, namespace=self.namespace)
        scores = [result['score'] for result in results['matches']]
        route_names = [result['metadata']['sr_route'] for result in results['matches']]
        return (np.array(scores), route_names)

    def _read_config(self, field: str, scope: str | None=None) -> ConfigParameter:
        """Read a config parameter from the index.

        :param field: The field to read.
        :type field: str
        :param scope: The scope to read.
        :type scope: str | None
        :return: The config parameter that was read.
        :rtype: ConfigParameter
        """
        scope = scope or self.namespace
        if self.index is None:
            return ConfigParameter(field=field, value='', scope=scope)
        config_id = f'{field}#{scope}'
        config_record = self.index.fetch(ids=[config_id], namespace='sr_config')
        if config_record.get('vectors'):
            return ConfigParameter(field=field, value=config_record['vectors'][config_id]['metadata']['value'], created_at=config_record['vectors'][config_id]['metadata']['created_at'], scope=scope)
        else:
            logger.warning(f'Configuration for {field} parameter not found in index.')
            return ConfigParameter(field=field, value='', scope=scope)

    async def _async_read_config(self, field: str, scope: str | None=None) -> ConfigParameter:
        """Read a config parameter from the index asynchronously.

        :param field: The field to read.
        :type field: str
        :param scope: The scope to read.
        :type scope: str | None
        :return: The config parameter that was read.
        :rtype: ConfigParameter
        """
        scope = scope or self.namespace
        if self.index is None:
            return ConfigParameter(field=field, value='', scope=scope)
        config_id = f'{field}#{scope}'
        config_record = await self._async_fetch_metadata(vector_id=config_id, namespace='sr_config')
        if config_record:
            try:
                return ConfigParameter(field=field, value=config_record['value'], created_at=config_record['created_at'], scope=scope)
            except KeyError:
                raise ValueError(f'Found invalid config record during sync: {config_record}')
        else:
            logger.warning(f'Configuration for {field} parameter not found in index.')
            return ConfigParameter(field=field, value='', scope=scope)

    def _write_config(self, config: ConfigParameter) -> ConfigParameter:
        """Method to write a config parameter to the remote Pinecone index.

        :param config: The config parameter to write to the index.
        :type config: ConfigParameter
        """
        config.scope = config.scope or self.namespace
        if self.index is None:
            raise ValueError('Index has not been initialized.')
        if self.dimensions is None:
            raise ValueError('Must set PineconeIndex.dimensions before writing config.')
        self.index.upsert(vectors=[config.to_pinecone(dimensions=self.dimensions)], namespace='sr_config')
        return config

    async def _async_write_config(self, config: ConfigParameter) -> ConfigParameter:
        """Method to write a config parameter to the remote Pinecone index.

        :param config: The config parameter to write to the index.
        :type config: ConfigParameter
        """
        config.scope = config.scope or self.namespace
        if self.dimensions is None:
            raise ValueError('Must set PineconeIndex.dimensions before writing config.')
        pinecone_config = config.to_pinecone(dimensions=self.dimensions)
        await self._async_upsert(vectors=[pinecone_config], namespace='sr_config')
        return config

    async def aquery(self, vector: np.ndarray, top_k: int=5, route_filter: Optional[List[str]]=None, sparse_vector: dict[int, float] | SparseEmbedding | None=None) -> Tuple[np.ndarray, List[str]]:
        """Asynchronously search the index for the query vector and return the top_k results.

        :param vector: The query vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of top results to return, defaults to 5.
        :type top_k: int, optional
        :param route_filter: A list of route names to filter the search results, defaults to None.
        :type route_filter: Optional[List[str]], optional
        :param kwargs: Additional keyword arguments for the query, including sparse_vector.
        :type kwargs: Any
        :keyword sparse_vector: An optional sparse vector to include in the query.
        :type sparse_vector: Optional[dict]
        :return: A tuple containing an array of scores and a list of route names.
        :rtype: Tuple[np.ndarray, List[str]]
        :raises ValueError: If the index is not populated.
        """
        if not await self.ais_ready():
            raise ValueError('Async index is not initialized.')
        query_vector_list = vector.tolist()
        if route_filter is not None:
            filter_query = {'sr_route': {'$in': route_filter}}
        else:
            filter_query = None
        sparse_vector_obj: dict[str, Any] | None = None
        if sparse_vector is not None:
            if isinstance(sparse_vector, dict):
                sparse_vector_obj = SparseEmbedding.from_dict(sparse_vector)
            if isinstance(sparse_vector, SparseEmbedding):
                sparse_vector_obj = sparse_vector.to_pinecone()
        results = await self._async_query(vector=query_vector_list, sparse_vector=sparse_vector_obj, namespace=self.namespace or '', filter=filter_query, top_k=top_k, include_metadata=True)
        scores = [result['score'] for result in results['matches']]
        route_names = [result['metadata']['sr_route'] for result in results['matches']]
        return (np.array(scores), route_names)

    async def aget_routes(self) -> list[tuple]:
        """Asynchronously get a list of route and utterance objects currently
        stored in the index.

        :return: A list of (route_name, utterance) objects.
        :rtype: List[Tuple]
        """
        if not await self.ais_ready():
            raise ValueError('Async index is not initialized.')
        return await self._async_get_routes()

    def delete_index(self):
        """Delete the index.

        :return: None
        :rtype: None
        """
        self.client.delete_index(self.index_name)
        self.index = None

    async def adelete_index(self):
        """Asynchronously delete the index."""
        if not await self.ais_ready():
            raise ValueError('Async index is not initialized.')
        if not self.base_url:
            raise ValueError('base_url is not set for PineconeIndex.')
        async with aiohttp.ClientSession() as session:
            async with session.delete(f'{self.base_url}/indexes/{self.index_name}', headers=self.headers) as response:
                res = await response.json(content_type=None)
                if response.status != 202:
                    raise Exception(f'Failed to delete index: {response.status}', res)
        self.host = ''
        return res

    async def _async_query(self, vector: list[float], sparse_vector: dict[str, Any] | None=None, namespace: str='', filter: Optional[dict]=None, top_k: int=5, include_metadata: bool=False):
        """Asynchronously query the index for the query vector and return the top_k results.

        :param vector: The query vector to search for.
        :type vector: list[float]
        :param sparse_vector: The sparse vector to search for.
        :type sparse_vector: dict[str, Any] | None
        :param namespace: The namespace to search for.
        :type namespace: str
        :param filter: The filter to search for.
        :type filter: Optional[dict]
        :param top_k: The number of top results to return, defaults to 5.
        :type top_k: int, optional
        :param include_metadata: Whether to include metadata in the results, defaults to False.
        :type include_metadata: bool, optional
        """
        params = {'vector': vector, 'sparse_vector': sparse_vector, 'namespace': namespace, 'filter': filter, 'top_k': top_k, 'include_metadata': include_metadata, 'topK': top_k, 'includeMetadata': include_metadata}
        if not await self.ais_ready():
            raise ValueError('Async index is not initialized.')
        elif self.base_url and 'api.pinecone.io' in self.base_url:
            if not self.host.startswith('http'):
                logger.error(f'host exists:{self.host}')
                self.host = f'https://{self.host}'
        elif self.host.startswith('localhost') and self.base_url:
            self.host = f'http://{self.base_url.split(':')[-2].strip('/')}:{self.host.split(':')[-1]}'
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self.host}/query', json=params, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f'Error in query response: {error_text}')
                    return {}
                try:
                    return await response.json(content_type=None)
                except JSONDecodeError as e:
                    logger.error(f'JSON decode error: {e}')
                    return {}

    async def ais_ready(self, client_only: bool=False) -> bool:
        """Checks if class attributes exist to be used for async operations.

        :param client_only: Whether to check only the client attributes. If False
            attributes will be checked for both client and index operations. If True
            only attributes for client operations will be checked. Defaults to False.
        :type client_only: bool, optional
        :return: True if the class attributes exist, False otherwise.
        :rtype: bool
        """
        if not (self.cloud or self.region or self.base_url):
            return False
        if not client_only:
            if not (self.index_name and self.dimensions and self.metric and self.host and (self.host != '')):
                await self._init_async_index()
                if not (self.index_name and self.dimensions and self.metric and self.host and (self.host != '')):
                    return False
        return True

    async def _async_list_indexes(self):
        """Asynchronously lists all indexes within the current Pinecone project.

        :return: List of indexes.
        :rtype: list[dict]
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{self.base_url}/indexes', headers=self.headers) as response:
                return await response.json(content_type=None)

    async def _async_upsert(self, vectors: list[dict], namespace: str=''):
        """Asynchronously upserts vectors into the index.

        :param vectors: The vectors to upsert.
        :type vectors: list[dict]
        :param namespace: The namespace to upsert the vectors into.
        :type namespace: str
        """
        if not await self.ais_ready():
            raise ValueError('Async index is not initialized.')
        params = {'vectors': vectors, 'namespace': namespace}
        if self.base_url and 'api.pinecone.io' in self.base_url:
            if not self.host.startswith('http'):
                logger.error(f'host exists:{self.host}')
                self.host = f'https://{self.host}'
        elif self.host.startswith('localhost') and self.base_url:
            self.host = f'http://{self.base_url.split(':')[-2].strip('/')}:{self.host.split(':')[-1]}'
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self.host}/vectors/upsert', json=params, headers=self.headers) as response:
                res = await response.json(content_type=None)
                return res

    async def _async_create_index(self, name: str, dimension: int, cloud: str, region: str, metric: str='dotproduct'):
        """Asynchronously creates a new index in Pinecone.

        :param name: The name of the index to create.
        :type name: str
        :param dimension: The dimension of the index.
        :type dimension: int
        :param cloud: The cloud provider to create the index on.
        :type cloud: str
        :param region: The region to create the index in.
        :type region: str
        :param metric: The metric to use for the index, defaults to "dotproduct".
        :type metric: str, optional
        """
        params = {'name': name, 'dimension': dimension, 'metric': metric, 'spec': {'serverless': {'cloud': cloud, 'region': region}}, 'deletion_protection': 'disabled'}
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self.base_url}/indexes', json=params, headers=self.headers) as response:
                return await response.json(content_type=None)

    async def _async_delete(self, ids: list[str], namespace: str=''):
        """Asynchronously deletes vectors from the index.

        :param ids: The IDs of the vectors to delete.
        :type ids: list[str]
        :param namespace: The namespace to delete the vectors from.
        :type namespace: str
        """
        params = {'ids': ids, 'namespace': namespace}
        if self.base_url and 'api.pinecone.io' in self.base_url:
            if not self.host.startswith('http'):
                logger.error(f'host exists:{self.host}')
                self.host = f'https://{self.host}'
        elif self.host.startswith('localhost') and self.base_url:
            self.host = f'http://{self.base_url.split(':')[-2].strip('/')}:{self.host.split(':')[-1]}'
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self.host}/vectors/delete', json=params, headers=self.headers) as response:
                return await response.json(content_type=None)

    async def _async_describe_index(self, name: str):
        """Asynchronously describes the index.

        :param name: The name of the index to describe.
        :type name: str
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{self.base_url}/indexes/{name}', headers=self.headers) as response:
                return await response.json(content_type=None)

    async def _async_get_all(self, prefix: Optional[str]=None, include_metadata: bool=False) -> tuple[list[str], list[dict]]:
        """Retrieves all vector IDs from the Pinecone index using pagination
        asynchronously.

        :param prefix: The prefix to filter the vectors by.
        :type prefix: Optional[str]
        :param include_metadata: Whether to include metadata in the response.
        :type include_metadata: bool
        :return: A tuple containing a list of vector IDs and a list of metadata dictionaries.
        :rtype: tuple[list[str], list[dict]]
        """
        if not await self.ais_ready():
            raise ValueError('Async index is not initialized.')
        all_vector_ids = []
        next_page_token = None
        if prefix:
            prefix_str = f'?prefix={prefix}'
        else:
            prefix_str = ''
        if self.base_url and 'api.pinecone.io' in self.base_url:
            if not self.host.startswith('http'):
                logger.error(f'host exists:{self.host}')
                self.host = f'https://{self.host}'
        elif self.host.startswith('localhost') and self.base_url:
            self.host = f'http://{self.base_url.split(':')[-2].strip('/')}:{self.host.split(':')[-1]}'
        list_url = f'{self.host}/vectors/list{prefix_str}'
        params: dict = {}
        if self.namespace:
            params['namespace'] = self.namespace
        metadata = []
        async with aiohttp.ClientSession() as session:
            while True:
                if next_page_token:
                    params['paginationToken'] = next_page_token
                async with session.get(list_url, params=params, headers=self.headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f'Error fetching vectors: {error_text}')
                        break
                    response_data = await response.json(content_type=None)
                vector_ids = [vec['id'] for vec in response_data.get('vectors', [])]
                if not vector_ids:
                    break
                all_vector_ids.extend(vector_ids)
                if include_metadata:
                    metadata_tasks = [self._async_fetch_metadata(id) for id in vector_ids]
                    metadata_results = await asyncio.gather(*metadata_tasks)
                    metadata.extend(metadata_results)
                next_page_token = response_data.get('pagination', {}).get('next')
                if not next_page_token:
                    break
        return (all_vector_ids, metadata)

    async def _async_fetch_metadata(self, vector_id: str, namespace: str | None=None) -> dict:
        """Fetch metadata for a single vector ID asynchronously using the
        ClientSession.

        :param vector_id: The ID of the vector to fetch metadata for.
        :type vector_id: str
        :param namespace: The namespace to fetch metadata for.
        :type namespace: str | None
        :return: A dictionary containing the metadata for the vector.
        :rtype: dict
        """
        if not await self.ais_ready():
            raise ValueError('Async index is not initialized.')
        if self.base_url and 'api.pinecone.io' in self.base_url:
            if not self.host.startswith('http'):
                logger.error(f'host exists:{self.host}')
                self.host = f'https://{self.host}'
        elif self.host.startswith('localhost') and self.base_url:
            self.host = f'http://{self.base_url.split(':')[-2].strip('/')}:{self.host.split(':')[-1]}'
        url = f'{self.host}/vectors/fetch'
        params = {'ids': [vector_id]}
        if namespace:
            params['namespace'] = [namespace]
        elif self.namespace:
            params['namespace'] = [self.namespace]
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f'Error fetching metadata: {error_text}')
                    return {}
                try:
                    response_data = await response.json(content_type=None)
                except Exception as e:
                    logger.warning(f'No metadata found for vector {vector_id}: {e}')
                    return {}
                return response_data.get('vectors', {}).get(vector_id, {}).get('metadata', {})

    def __len__(self):
        """Returns the total number of vectors in the index. If the index is not initialized
        returns 0.

        :return: The total number of vectors.
        :rtype: int
        """
        if self.index is None:
            logger.warning('Index is not initialized, returning 0')
            return 0
        namespace_stats = self.index.describe_index_stats()['namespaces'].get(self.namespace)
        if namespace_stats:
            return namespace_stats['vector_count']
        else:
            return 0

    async def alen(self):
        """Async version of __len__. Returns the total number of vectors in the index.
        If the index is not initialized, initializes it first or returns 0.

        :return: The total number of vectors.
        :rtype: int
        """
        if not await self.ais_ready():
            logger.warning('Index is not ready, returning 0')
            return 0
        namespace_stats = await self._async_describe_index_stats()
        if namespace_stats and 'namespaces' in namespace_stats:
            ns_stats = namespace_stats['namespaces'].get(self.namespace)
            if ns_stats:
                return ns_stats['vectorCount']
        return 0

    async def _async_describe_index_stats(self):
        """Async version of describe_index_stats.

        :return: Index statistics.
        :rtype: dict
        """
        url = f'{self.index_host}/describe_index_stats'
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json={'namespace': self.namespace}, timeout=aiohttp.ClientTimeout(total=300)) as response:
                response.raise_for_status()
                return await response.json()

class PostgresIndex(BaseIndex):
    """Postgres implementation of Index."""
    connection_string: Optional[str] = None
    index_prefix: str = 'semantic_router_'
    index_name: str = 'index'
    metric: Metric = Metric.COSINE
    namespace: Optional[str] = ''
    conn: Optional['psycopg.Connection'] = None
    async_conn: Optional['psycopg.AsyncConnection'] = None
    type: str = 'postgres'
    index_type: IndexType = IndexType.FLAT
    init_async_index: bool = False

    def __init__(self, connection_string: Optional[str]=None, index_prefix: str='semantic_router_', index_name: str='index', metric: Metric=Metric.COSINE, namespace: Optional[str]='', dimensions: int | None=None, init_async_index: bool=False):
        """Initializes the Postgres index with the specified parameters.

        :param connection_string: The connection string for the PostgreSQL database.
        :type connection_string: Optional[str]
        :param index_prefix: The prefix for the index table name.
        :type index_prefix: str
        :param index_name: The name of the index table.
        :type index_name: str
        :param dimensions: The number of dimensions for the vectors.
        :type dimensions: int
        :param metric: The metric used for vector comparisons.
        :type metric: Metric
        :param namespace: An optional namespace for the index.
        :type namespace: Optional[str]
        :param init_async_index: Whether to initialize the index asynchronously.
        :type init_async_index: bool
        """
        if not _psycopg_installed:
            raise ImportError("Please install psycopg to use PostgresIndex. You can install it with: `pip install 'semantic-router[postgres]'`")
        super().__init__()
        if index_prefix:
            logger.warning('`index_prefix` is deprecated and will be removed in 0.2.0')
        if connection_string or (connection_string := os.getenv('POSTGRES_CONNECTION_STRING')):
            pass
        else:
            required_env_vars = ['POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_DB']
            missing = [var for var in required_env_vars if not os.getenv(var)]
            if missing:
                raise ValueError(f'Missing required environment variables for Postgres connection: {', '.join(missing)}')
            connection_string = f'postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}'
        self.connection_string = connection_string
        self.index = self
        self.index_prefix = index_prefix
        self.index_name = index_name
        self.dimensions = dimensions
        self.metric = metric
        self.namespace = namespace
        self.init_async_index = init_async_index
        self.conn = None
        self.async_conn = None

    def _init_index(self, force_create: bool=False) -> Union[Any, None]:
        """Initializing the index can be done after the object has been created
        to allow for the user to set the dimensions and other parameters.

        If the index doesn't exist and the dimensions are given, the index will
        be created. If the index exists, it will be returned. If the index doesn't
        exist and the dimensions are not given, the index will not be created and
        None will be returned.

        :param force_create: If True, the index will be created even if the
            dimensions are not given (which will raise an error).
        :type force_create: bool, optional
        """
        if not self.connection_string:
            raise ValueError('No `self.connection_string` attribute set')
        self.conn = psycopg.connect(conninfo=self.connection_string)
        if not self.has_connection():
            raise ValueError('Index has not established a connection to Postgres')
        dimensions_given = self.dimensions is not None
        if not dimensions_given:
            raise ValueError('Dimensions are required for PostgresIndex')
        table_name = self._get_table_name()
        if not self._check_embeddings_dimensions():
            raise ValueError(f'The length of the vector embeddings in the existing table {table_name} does not match the expected dimensions of {self.dimensions}.')
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError('Index has not established a connection to Postgres')
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"\n                    CREATE EXTENSION IF NOT EXISTS vector;\n                    CREATE TABLE IF NOT EXISTS {table_name} (\n                        id VARCHAR(255) PRIMARY KEY,\n                        route VARCHAR(255),\n                        utterance TEXT,\n                        vector VECTOR({self.dimensions})\n                    );\n                    COMMENT ON COLUMN {table_name}.vector IS '{self.dimensions}';\n                    ")
                self.conn.commit()
            self._create_route_index()
            self._create_index()
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise
        return self

    async def _init_async_index(self, force_create: bool=False) -> Union[Any, None]:
        logging.warning('[DEBUG] Entering _init_async_index for PostgresIndex')
        if self.async_conn is None:
            if not self.connection_string:
                raise ValueError('No `self.connection_string` attribute set')
            logging.warning(f'[DEBUG] Connecting async to Postgres with: {self.connection_string}')
            self.async_conn = await psycopg.AsyncConnection.connect(self.connection_string)
            logging.warning(f'[DEBUG] Async connection established: {self.async_conn}')
        if self.dimensions is None and (not force_create):
            logging.warning('[DEBUG] No dimensions and not force_create, returning None from _init_async_index')
            return None
        if self.dimensions is None:
            raise ValueError('Dimensions are required for PostgresIndex')
        table_name = self._get_table_name()
        logging.warning(f'[DEBUG] Table name for async index: {table_name}')
        if not await self._async_check_embeddings_dimensions():
            raise ValueError(f'The length of the vector embeddings in the existing table {table_name} does not match the expected dimensions of {self.dimensions}.')
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError('Index has not established a connection to async Postgres')
        try:
            async with self.async_conn.cursor() as cur:
                logging.warning(f'[DEBUG] Creating extension/table for {table_name}')
                await cur.execute(f"\n                    CREATE EXTENSION IF NOT EXISTS vector;\n                    CREATE TABLE IF NOT EXISTS {table_name} (\n                        id VARCHAR(255) PRIMARY KEY,\n                        route VARCHAR(255),\n                        utterance TEXT,\n                        vector VECTOR({self.dimensions})\n                    );\n                    COMMENT ON COLUMN {table_name}.vector IS '{self.dimensions}';\n                    ")
                await self.async_conn.commit()
                await self._async_create_route_index()
                await self._async_create_index()
                logging.warning(f'[DEBUG] Finished async index/table creation for {table_name}')
        except Exception as e:
            logging.warning(f'[DEBUG] Exception in _init_async_index: {e}')
            await self.async_conn.rollback()
            raise e
        logging.warning('[DEBUG] Exiting _init_async_index for PostgresIndex')
        return self

    def _get_table_name(self) -> str:
        """
        Returns the name of the table for the index.

        :return: The table name.
        :rtype: str
        """
        return f'{self.index_prefix}{self.index_name}'

    def _get_metric_operator(self) -> str:
        """Returns the PostgreSQL operator for the specified metric.

        :return: The PostgreSQL operator.
        :rtype: str
        """
        return MetricPgVecOperatorMap[self.metric.value].value

    def _get_score_query(self, embeddings_str: str) -> str:
        """Creates the select statement required to return the embeddings distance.

        :param embeddings_str: The string representation of the embeddings.
        :type embeddings_str: str
        :return: The SQL query part for scoring.
        :rtype: str
        """
        operator = self._get_metric_operator()
        if self.metric == Metric.COSINE:
            return f'1 - (vector {operator} {embeddings_str}) AS score'
        elif self.metric == Metric.DOTPRODUCT:
            return f'(vector {operator} {embeddings_str}) * -1 AS score'
        elif self.metric == Metric.EUCLIDEAN:
            return f'vector {operator} {embeddings_str} AS score'
        elif self.metric == Metric.MANHATTAN:
            return f'vector {operator} {embeddings_str} AS score'
        else:
            raise ValueError(f'Unsupported metric: {self.metric}')

    def _get_vector_operator(self) -> str:
        if self.metric == Metric.COSINE:
            return 'vector_cosine_ops'
        elif self.metric == Metric.DOTPRODUCT:
            return 'vector_ip_ops'
        elif self.metric == Metric.EUCLIDEAN:
            return 'vector_l2_ops'
        elif self.metric == Metric.MANHATTAN:
            return 'vector_l1_ops'
        else:
            raise ValueError(f'Unsupported metric: {self.metric}')

    def _create_route_index(self) -> None:
        """Creates a index on the route column."""
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError('Index has not established a connection to Postgres')
        try:
            with self.conn.cursor() as cur:
                cur.execute(f'CREATE INDEX IF NOT EXISTS {table_name}_route_idx ON {table_name} USING btree (route);')
                self.conn.commit()
        except psycopg.errors.DuplicateTable:
            if self.conn is not None:
                self.conn.rollback()
            pass
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def _async_create_route_index(self) -> None:
        """Asynchronously creates an index on the route column."""
        table_name = self._get_table_name()
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError('Index has not established a connection to async Postgres')
        try:
            async with self.async_conn.cursor() as cur:
                await cur.execute(f'CREATE INDEX IF NOT EXISTS {table_name}_route_idx ON {table_name} USING btree (route);')
            await self.async_conn.commit()
        except psycopg.errors.DuplicateTable:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            pass
        except Exception:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            raise

    def _create_index(self) -> None:
        """Creates an index on the vector column based on index_type."""
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError('Index has not established a connection to Postgres')
        opclass = self._get_vector_operator()
        try:
            with self.conn.cursor() as cur:
                if self.index_type == IndexType.HNSW:
                    cur.execute(f'\n                        CREATE INDEX IF NOT EXISTS {table_name}_vector_idx ON {table_name} USING hnsw (vector {opclass});\n                        ')
                elif self.index_type == IndexType.IVFFLAT:
                    cur.execute(f'\n                        CREATE INDEX IF NOT EXISTS {table_name}_vector_idx ON {table_name} USING ivfflat (vector {opclass}) WITH (lists = 100);\n                        ')
                elif self.index_type == IndexType.FLAT:
                    cur.execute(f'\n                        CREATE INDEX IF NOT EXISTS {table_name}_vector_idx ON {table_name} USING ivfflat (vector {opclass}) WITH (lists = 1);\n                        ')
                self.conn.commit()
        except psycopg.errors.DuplicateTable:
            if self.conn is not None:
                self.conn.rollback()
            pass
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def _async_create_index(self) -> None:
        """Asynchronously creates an index on the vector column based on index_type."""
        table_name = self._get_table_name()
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError('Index has not established a connection to async Postgres')
        opclass = self._get_vector_operator()
        try:
            async with self.async_conn.cursor() as cur:
                if self.index_type == IndexType.HNSW:
                    await cur.execute(f'\n                        CREATE INDEX IF NOT EXISTS {table_name}_vector_idx ON {table_name} USING hnsw (vector {opclass});\n                        ')
                elif self.index_type == IndexType.IVFFLAT:
                    await cur.execute(f'\n                        CREATE INDEX IF NOT EXISTS {table_name}_vector_idx ON {table_name} USING ivfflat (vector {opclass}) WITH (lists = 100);\n                        ')
                elif self.index_type == IndexType.FLAT:
                    await cur.execute(f'\n                        CREATE INDEX IF NOT EXISTS {table_name}_vector_idx ON {table_name} USING ivfflat (vector {opclass}) WITH (lists = 1);\n                        ')
            await self.async_conn.commit()
        except psycopg.errors.DuplicateTable:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            pass
        except Exception:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            raise

    @deprecated('Use _init_index or sync methods such as `auto_sync` (read more https://docs.aurelio.ai/semantic-router/user-guide/features/sync). This method will be removed in 0.2.0')
    def setup_index(self) -> None:
        """Sets up the index by creating the table and vector extension if they do not exist.

        :raises ValueError: If the existing table's vector dimensions do not match the expected dimensions.
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not self._check_embeddings_dimensions():
            raise ValueError(f'The length of the vector embeddings in the existing table {table_name} does not match the expected dimensions of {self.dimensions}.')
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError('Index has not established a connection to Postgres')
        with self.conn.cursor() as cur:
            cur.execute(f"\n                CREATE EXTENSION IF NOT EXISTS vector;\n                CREATE TABLE IF NOT EXISTS {table_name} (\n                    id VARCHAR(255) PRIMARY KEY,\n                    route VARCHAR(255),\n                    utterance TEXT,\n                    vector VECTOR({self.dimensions})\n                );\n                COMMENT ON COLUMN {table_name}.vector IS '{self.dimensions}';\n                ")
            self.conn.commit()
        self._create_route_index()
        self._create_index()

    def _check_embeddings_dimensions(self) -> bool:
        """Checks if the length of the vector embeddings in the table matches the expected
        dimensions, or if no table exists.

        :return: True if the dimensions match or the table does not exist, False otherwise.
        :rtype: bool
        :raises ValueError: If the vector column comment does not contain a valid integer.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError('Index has not established a connection to Postgres')
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='{table_name}');")
                fetch_result = cur.fetchone()
                exists = fetch_result[0] if fetch_result else None
                if not exists:
                    return True
                cur.execute(f"SELECT col_description('{table_name}'::regclass, attnum) AS column_comment\n                        FROM pg_attribute\n                        WHERE attrelid = '{table_name}'::regclass\n                        AND attname='vector'")
                result = cur.fetchone()
                dimension_comment = result[0] if result else None
                if dimension_comment:
                    try:
                        vector_length = int(dimension_comment.split()[-1])
                        return vector_length == self.dimensions
                    except ValueError:
                        raise ValueError("The 'vector' column comment does not contain a valid integer.")
                else:
                    raise ValueError("No comment found for the 'vector' column.")
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def _async_check_embeddings_dimensions(self) -> bool:
        """Asynchronously checks if the vector embedding dimensions match the expected ones.

        Returns True if dimensions match or table does not exist, False otherwise.

        :return: True if the dimensions match or the table does not exist, False otherwise.
        :rtype: bool
        :raises ValueError: If the vector column comment does not contain a valid integer.
        """
        table_name = self._get_table_name()
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError('Index has not established a connection to async Postgres')
        try:
            async with self.async_conn.cursor() as cur:
                await cur.execute(f"SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='{table_name}');")
                fetch_result = await cur.fetchone()
                exists = fetch_result[0] if fetch_result else None
                if not exists:
                    return True
                await cur.execute(f"SELECT col_description('{table_name}'::regclass, attnum) AS column_comment\n                        FROM pg_attribute\n                        WHERE attrelid = '{table_name}'::regclass\n                        AND attname = 'vector';")
                result = await cur.fetchone()
                dimension_comment = result[0] if result else None
                if dimension_comment:
                    try:
                        vector_length = int(dimension_comment.split()[-1])
                        return vector_length == self.dimensions
                    except ValueError:
                        raise ValueError("The 'vector' column comment does not contain a valid integer.")
                else:
                    raise ValueError("No comment found for the 'vector' column.")
        except Exception:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            raise

    def add(self, embeddings: List[List[float]], routes: List[str], utterances: List[str], function_schemas: Optional[List[Dict[str, Any]]]=None, metadata_list: List[Dict[str, Any]]=[], **kwargs) -> None:
        """Adds records to the index.

        :param embeddings: A list of vector embeddings to add.
        :type embeddings: List[List[float]]
        :param routes: A list of route names corresponding to the embeddings.
        :type routes: List[str]
        :param utterances: A list of utterances corresponding to the embeddings.
        :type utterances: List[Any]
        :param function_schemas: A list of function schemas corresponding to the embeddings.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: A list of metadata corresponding to the embeddings.
        :type metadata_list: List[Dict[str, Any]]
        :raises ValueError: If the vector embeddings being added do not match the expected dimensions.
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        new_embeddings_length = len(embeddings[0])
        if new_embeddings_length != self.dimensions:
            raise ValueError(f'The vector embeddings being added are of length {new_embeddings_length}, which does not match the expected dimensions of {self.dimensions}.')
        records = [PostgresIndexRecord(vector=vector, route=route, utterance=utterance) for vector, route, utterance in zip(embeddings, routes, utterances)]
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError('Index has not established a connection to Postgres')
        try:
            with self.conn.cursor() as cur:
                cur.executemany(f'INSERT INTO {table_name} (id, route, utterance, vector) VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING', [(record.id, record.route, record.utterance, record.vector) for record in records])
                self.conn.commit()
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def aadd(self, embeddings: List[List[float]], routes: List[str], utterances: List[str], function_schemas: Optional[List[Dict[str, Any]]]=None, metadata_list: List[Dict[str, Any]]=[], batch_size: int=100, **kwargs) -> None:
        """
        Asynchronously adds records to the index in batches.

        :param embeddings: A list of vector embeddings to add.
        :param routes: A list of route names corresponding to the embeddings.
        :param utterances: A list of utterances corresponding to the embeddings.
        :param function_schemas: (Optional) List of function schemas.
        :param metadata_list: (Optional) List of metadata dictionaries.
        :param batch_size: Number of records per batch insert.
        :raises ValueError: If the vector embeddings don't match expected dimensions.
        :raises TypeError: If connection is not an async Postgres connection.
        """
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError('Index has not established an async connection to Postgres')
        table_name = self._get_table_name()
        new_embeddings_length = len(embeddings[0])
        if new_embeddings_length != self.dimensions:
            raise ValueError(f'The vector embeddings being added are of length {new_embeddings_length}, which does not match the expected dimensions of {self.dimensions}.')
        try:
            async with self.async_conn.cursor() as cur:
                for i in range(0, len(embeddings), batch_size):
                    batch_embeddings = embeddings[i:i + batch_size]
                    batch_routes = routes[i:i + batch_size]
                    batch_utterances = utterances[i:i + batch_size]
                    values = [(str(uuid.uuid4()), route, utterance, vector) for route, utterance, vector in zip(batch_routes, batch_utterances, batch_embeddings)]
                    await cur.executemany(f'INSERT INTO {table_name} (id, route, utterance, vector) VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING', values)
                await self.async_conn.commit()
        except Exception:
            await self.async_conn.rollback()
            raise

    def delete(self, route_name: str) -> None:
        """Deletes records with the specified route name.

        :param route_name: The name of the route to delete records for.
        :type route_name: str
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError('Index has not established a connection to Postgres')
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"DELETE FROM {table_name} WHERE route = '{route_name}'")
                self.conn.commit()
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def adelete(self, route_name: str) -> list[str]:
        """Asynchronously delete specified route from index if it exists. Returns the IDs
        of the vectors deleted.

        :param route_name: Name of the route to delete.
        :type route_name: str
        :return: List of IDs of the vectors deleted.
        :rtype: list[str]
        """
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError('Index has not established an async connection to Postgres')
        table_name = self._get_table_name()
        try:
            async with self.async_conn.cursor() as cur:
                await cur.execute(f'SELECT id FROM {table_name} WHERE route = %s', (route_name,))
                result = await cur.fetchall()
                deleted_ids = [row[0] for row in result]
                await cur.execute(f'DELETE FROM {table_name} WHERE route = %s', (route_name,))
                await self.async_conn.commit()
                return deleted_ids
        except Exception:
            await self.async_conn.rollback()
            raise

    def describe(self) -> IndexConfig:
        """Describes the index by returning its type, dimensions, and total vector count.

        :return: An IndexConfig object containing the index's type, dimensions, and total vector count.
        :rtype: IndexConfig
        """
        table_name = self._get_table_name()
        if not isinstance(self.async_conn, psycopg.Connection):
            logger.warning('Index has not established a connection to Postgres')
            return IndexConfig(type=self.type, dimensions=self.dimensions or 0, vectors=0)
        try:
            with self.async_conn.cursor() as cur:
                cur.execute(f'SELECT COUNT(*) FROM {table_name}')
                result = cur.fetchone()
                count = result[0] if result is not None else 0
                return IndexConfig(type=self.type, dimensions=self.dimensions or 0, vectors=count)
        except Exception:
            if self.async_conn is not None:
                self.async_conn.rollback()
            raise

    def is_ready(self) -> bool:
        """Checks if the index is ready to be used.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        return isinstance(self.conn, psycopg.Connection)

    async def ais_ready(self) -> bool:
        """Checks if the index is ready to be used.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        return isinstance(self.async_conn, psycopg.AsyncConnection)

    def query(self, vector: np.ndarray, top_k: int=5, route_filter: Optional[List[str]]=None, sparse_vector: dict[int, float] | SparseEmbedding | None=None) -> Tuple[np.ndarray, List[str]]:
        """Searches the index for the query vector and returns the top_k results.

        :param vector: The query vector.
        :type vector: np.ndarray
        :param top_k: The number of top results to return.
        :type top_k: int
        :param route_filter: Optional list of routes to filter the results by.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: Optional sparse vector to filter the results by.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple containing the scores and routes of the top_k results.
        :rtype: Tuple[np.ndarray, List[str]]
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError('Index has not established a connection to Postgres')
        try:
            with self.conn.cursor() as cur:
                filter_query = f' AND route = ANY(ARRAY{route_filter})' if route_filter else ''
                vector_str = f"'[{','.join(map(str, vector.tolist()))}]'"
                score_query = self._get_score_query(vector_str)
                operator = self._get_metric_operator()
                query = f'SELECT route, {score_query} FROM {table_name} WHERE true{filter_query} ORDER BY vector {operator} {vector_str} LIMIT {top_k}'
                cur.execute(query)
                results = cur.fetchall()
                return (np.array([result[1] for result in results]), [result[0] for result in results])
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def aquery(self, vector: np.ndarray, top_k: int=5, route_filter: Optional[List[str]]=None, sparse_vector: dict[int, float] | SparseEmbedding | None=None) -> Tuple[np.ndarray, List[str]]:
        """Asynchronously search the index for the query vector and return the top_k results.

        :param vector: The query vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of top results to return, defaults to 5.
        :type top_k: int, optional
        :param route_filter: A list of route names to filter the search results, defaults to None.
        :type route_filter: Optional[List[str]], optional
        :param sparse_vector: An optional sparse vector to include in the query.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple containing an array of scores and a list of route names.
        :rtype: Tuple[np.ndarray, List[str]]
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError('Index has not established an async connection to Postgres')
        try:
            async with self.async_conn.cursor() as cur:
                filter_query = f' AND route = ANY(ARRAY{route_filter})' if route_filter else ''
                vector_str = f"'[{','.join(map(str, vector.tolist()))}]'"
                score_query = self._get_score_query(vector_str)
                operator = self._get_metric_operator()
                query = f'SELECT route, {score_query} FROM {table_name} WHERE true{filter_query} ORDER BY vector {operator} {vector_str} LIMIT {top_k}'
                await cur.execute(query)
                results = await cur.fetchall()
                return (np.array([result[1] for result in results]), [result[0] for result in results])
        except Exception:
            await self.async_conn.rollback()
            raise

    def _get_route_ids(self, route_name: str):
        """Retrieves all vector IDs for a specific route.

        :param route_name: The name of the route to retrieve IDs for.
        :type route_name: str
        :return: A list of vector IDs.
        :rtype: List[str]
        """
        clean_route = clean_route_name(route_name)
        try:
            ids, _ = self._get_all(route_name=f'{clean_route}')
            return ids
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def _async_get_route_ids(self, route_name: str) -> list[str]:
        """Get the IDs of the routes in the index asynchronously.

        :param route_name: Name of the route to get the IDs for.
        :type route_name: str
        :return: List of IDs of the routes.
        :rtype: list[str]
        """
        clean_route = clean_route_name(route_name)
        try:
            ids, _ = await self._async_get_all(route_name=f'{clean_route}')
            return ids
        except Exception:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            raise

    def _get_all(self, route_name: Optional[str]=None, include_metadata: bool=False):
        """Retrieves all vector IDs and optionally metadata from the Postgres index.

        :param route_name: Optional route name to filter the results by.
        :type route_name: Optional[str]
        :param include_metadata: Whether to include metadata in the results.
        :type include_metadata: bool
        :return: A tuple containing the list of vector IDs and optionally metadata.
        :rtype: Tuple[List[str], List[Dict]]
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError('Index has not established a connection to Postgres')
        try:
            query = 'SELECT id'
            if include_metadata:
                query += ', route, utterance'
            query += f' FROM {table_name}'
            if route_name:
                query += f" WHERE route LIKE '{route_name}%'"
            all_vector_ids = []
            metadata = []
            with self.conn.cursor() as cur:
                cur.execute(query)
                results = cur.fetchall()
                for row in results:
                    all_vector_ids.append(row[0])
                    if include_metadata:
                        metadata.append({'sr_route': row[1], 'sr_utterance': row[2]})
            return (all_vector_ids, metadata)
        except psycopg.errors.UndefinedTable:
            if self.conn is not None:
                self.conn.rollback()
            return ([], [])
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def _async_get_all(self, route_name: Optional[str]=None, include_metadata: bool=False) -> Tuple[List[str], List[Dict]]:
        """Retrieves all vector IDs and optionally metadata from the Postgres index asynchronously.

        :param route_name: Optional route name to filter the results by.
        :type route_name: Optional[str]
        :param include_metadata: Whether to include metadata in the results.
        :type include_metadata: bool
        :return: A tuple containing the list of vector IDs and optionally metadata.
        :rtype: Tuple[List[str], List[Dict]]
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError('Index has not established a connection to async Postgres')
        try:
            query = 'SELECT id'
            if include_metadata:
                query += ', route, utterance'
            query += f' FROM {table_name}'
            if route_name:
                query += f" WHERE route LIKE '{route_name}%'"
            all_vector_ids = []
            metadata = []
            async with self.async_conn.cursor() as cur:
                await cur.execute(query)
                results = await cur.fetchall()
                for row in results:
                    all_vector_ids.append(row[0])
                    if include_metadata:
                        metadata.append({'sr_route': row[1], 'sr_utterance': row[2]})
            return (all_vector_ids, metadata)
        except psycopg.errors.UndefinedTable:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            return ([], [])
        except Exception:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            raise

    def _remove_and_sync(self, routes_to_delete: dict):
        """
        Remove embeddings in a routes syncing process from the Postgres index.

        :param routes_to_delete: Dictionary of routes to delete.
        :type routes_to_delete: dict
        :return: List of (route, utterance) tuples that were removed.
        """
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError('Index has not established a connection to Postgres')
        table_name = self._get_table_name()
        removed = []
        try:
            with self.conn.cursor() as cur:
                for route, utterances in routes_to_delete.items():
                    for utterance in utterances:
                        cur.execute(f'SELECT route, utterance FROM {table_name} WHERE route = %s AND utterance = %s', (route, utterance))
                        result = cur.fetchone()
                        if result:
                            removed.append(result)
                        cur.execute(f'DELETE FROM {table_name} WHERE route = %s AND utterance = %s', (route, utterance))
            self.conn.commit()
            return removed
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def _async_remove_and_sync(self, routes_to_delete: dict) -> list[tuple[str, str]]:
        """Remove specified routes from index if they exist.

        This method is asynchronous.

        :param routes_to_delete: Routes to delete.
        :type routes_to_delete: dict
        :return: List of (route, utterance) tuples that were removed.
        :rtype: list[tuple[str, str]]
        """
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError('Index has not established an async connection to Postgres')
        table_name = self._get_table_name()
        removed = []
        try:
            async with self.async_conn.cursor() as cur:
                for route, utterances in routes_to_delete.items():
                    for utterance in utterances:
                        await cur.execute(f'SELECT route, utterance FROM {table_name} WHERE route = %s AND utterance = %s', (route, utterance))
                        result = await cur.fetchone()
                        if result:
                            removed.append(result)
                        await cur.execute(f'DELETE FROM {table_name} WHERE route = %s AND utterance = %s', (route, utterance))
            await self.async_conn.commit()
            return removed
        except Exception:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            raise

    def delete_all(self):
        """Deletes all records from the Postgres index.

        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError('Index has not established a connection to Postgres')
        try:
            with self.conn.cursor() as cur:
                cur.execute(f'DELETE FROM {table_name}')
                self.conn.commit()
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    def delete_index(self) -> None:
        """Deletes the entire table for the index.

        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError('Index has not established a connection to Postgres')
        try:
            with self.conn.cursor() as cur:
                cur.execute('\n                    SELECT pg_terminate_backend(pid)\n                    FROM pg_stat_activity\n                    WHERE datname = current_database()\n                      AND pid <> pg_backend_pid();\n                    ')
                self.conn.commit()
                cur.execute(f'DROP TABLE IF EXISTS {table_name}')
                self.conn.commit()
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def adelete_index(self) -> None:
        """Asynchronously delete the entire table for the index.

        :raises TypeError: If the async database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError('Index has not established an async connection to Postgres')
        try:
            async with self.async_conn.cursor() as cur:
                await cur.execute(f'DROP TABLE IF EXISTS {table_name}')
                await self.async_conn.commit()
        except Exception:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            raise

    async def aget_routes(self) -> list[tuple]:
        """Asynchronously get a list of route and utterance objects currently
        stored in the index.

        :return: A list of (route_name, utterance) objects.
        :rtype: List[Tuple]
        :raises TypeError: If the database connection is not established.
        """
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError('Index has not established an async connection to Postgres')
        return await self._async_get_routes()

    def _write_config(self, config: ConfigParameter):
        """Write the config to the index.

        :param config: The config to write to the index.
        :type config: ConfigParameter
        """
        logger.warning('No config is written for PostgresIndex.')

    def __len__(self):
        """Returns the total number of vectors in the index. If the index is not initialized
        returns 0.

        :return: The total number of vectors.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            logger.warning('Index has not established a connection to Postgres, returning 0')
            return 0
        with self.conn.cursor() as cur:
            try:
                cur.execute(f'SELECT COUNT(*) FROM {table_name}')
                count = cur.fetchone()
                if count is None:
                    return 0
                return count[0]
            except psycopg.errors.UndefinedTable:
                logger.warning('Table does not exist, returning 0')
                return 0

    async def alen(self):
        """Async version of __len__. Returns the total number of vectors in the index.

        :return: The total number of vectors.
        :rtype: int
        """
        table_name = self._get_table_name()
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            logger.warning('Index has not established an async connection to Postgres, returning 0')
            return 0
        async with self.async_conn.cursor() as cur:
            try:
                await cur.execute(f'SELECT COUNT(*) FROM {table_name}')
                count = await cur.fetchone()
                if count is None:
                    return 0
                return count[0]
            except psycopg.errors.UndefinedTable:
                logger.warning('Table does not exist, returning 0')
                return 0

    def close(self):
        """Closes the psycopg connection if it exists."""
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception as e:
                logger.warning(f'Error closing Postgres connection: {e}')
            self.conn = None

    def __del__(self):
        self.close()

    def has_connection(self) -> bool:
        """Returns True if there is an active and valid psycopg connection, otherwise False."""
        if self.conn is None or self.conn.closed:
            return False
        try:
            with self.conn.cursor() as cur:
                cur.execute('SELECT 1;')
                cur.fetchone()
            return True
        except Exception:
            return False
    'Configuration for the Pydantic BaseModel.'
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

