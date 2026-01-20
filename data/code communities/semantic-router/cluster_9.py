# Cluster 9

class BaseIndex(BaseModel):
    """
    Base class for indices using Pydantic's BaseModel.
    This class outlines the expected interface for index classes.
    Actual method implementations should be provided in subclasses.
    """
    routes: Optional[np.ndarray] = None
    utterances: Optional[np.ndarray] = None
    dimensions: Union[int, None] = None
    type: str = 'base'
    init_async_index: bool = False
    index: Optional[Any] = None

    def add(self, embeddings: List[List[float]], routes: List[str], utterances: List[Any], function_schemas: Optional[List[Dict[str, Any]]]=None, metadata_list: List[Dict[str, Any]]=[], **kwargs):
        """Add embeddings to the index.
        This method should be implemented by subclasses.

        :param embeddings: List of embeddings to add to the index.
        :type embeddings: List[List[float]]
        :param routes: List of routes to add to the index.
        :type routes: List[str]
        :param utterances: List of utterances to add to the index.
        :type utterances: List[str]
        :param function_schemas: List of function schemas to add to the index.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: List of metadata to add to the index.
        :type metadata_list: List[Dict[str, Any]]
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    async def aadd(self, embeddings: List[List[float]], routes: List[str], utterances: List[str], function_schemas: Optional[Optional[List[Dict[str, Any]]]]=None, metadata_list: List[Dict[str, Any]]=[], **kwargs):
        """Add vectors to the index asynchronously.
        This method should be implemented by subclasses.

        :param embeddings: List of embeddings to add to the index.
        :type embeddings: List[List[float]]
        :param routes: List of routes to add to the index.
        :type routes: List[str]
        :param utterances: List of utterances to add to the index.
        :type utterances: List[str]
        :param function_schemas: List of function schemas to add to the index.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: List of metadata to add to the index.
        :type metadata_list: List[Dict[str, Any]]
        """
        logger.warning('Async method not implemented.')
        return self.add(embeddings=embeddings, routes=routes, utterances=utterances, function_schemas=function_schemas, metadata_list=metadata_list, **kwargs)

    def get_utterances(self, include_metadata: bool=False) -> List[Utterance]:
        """Gets a list of route and utterance objects currently stored in the
        index, including additional metadata.

        :param include_metadata: Whether to include function schemas and metadata in
        the returned Utterance objects.
        :type include_metadata: bool
        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        if self.index is None:
            logger.warning('Index is None, could not retrieve utterances.')
            return []
        _, metadata = self._get_all(include_metadata=True)
        route_tuples = parse_route_info(metadata=metadata)
        if not include_metadata:
            route_tuples = [x[:2] for x in route_tuples]
        return [Utterance.from_tuple(x) for x in route_tuples]

    async def aget_utterances(self, include_metadata: bool=False) -> List[Utterance]:
        """Gets a list of route and utterance objects currently stored in the
        index, including additional metadata.

        :param include_metadata: Whether to include function schemas and metadata in
        the returned Utterance objects.
        :type include_metadata: bool
        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        if self.index is None:
            logger.warning('Index is None, could not retrieve utterances.')
            return []
        _, metadata = await self._async_get_all(include_metadata=True)
        route_tuples = parse_route_info(metadata=metadata)
        if not include_metadata:
            route_tuples = [x[:2] for x in route_tuples]
        return [Utterance.from_tuple(x) for x in route_tuples]

    def get_routes(self) -> List[Route]:
        """Gets a list of route objects currently stored in the index.

        :return: A list of Route objects.
        :rtype: List[Route]
        """
        utterances = self.get_utterances(include_metadata=True)
        routes_dict: Dict[str, Route] = {}
        for utt in utterances:
            if utt.route not in routes_dict:
                routes_dict[utt.route] = Route(name=utt.route, utterances=[utt.utterance], function_schemas=utt.function_schemas, metadata=utt.metadata)
            else:
                routes_dict[utt.route].utterances.append(utt.utterance)
        routes: List[Route] = []
        for route_name, route in routes_dict.items():
            routes.append(route)
        return routes

    def _remove_and_sync(self, routes_to_delete: dict):
        """
        Remove embeddings in a routes syncing process from the index.
        This method should be implemented by subclasses.

        :param routes_to_delete: Dictionary of routes to delete.
        :type routes_to_delete: dict
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    async def _async_remove_and_sync(self, routes_to_delete: dict):
        """
        Remove embeddings in a routes syncing process from the index asynchronously.
        This method should be implemented by subclasses.

        :param routes_to_delete: Dictionary of routes to delete.
        :type routes_to_delete: dict
        """
        logger.warning('Async method not implemented.')
        return self._remove_and_sync(routes_to_delete=routes_to_delete)

    def delete(self, route_name: str):
        """Deletes route by route name.
        This method should be implemented by subclasses.

        :param route_name: Name of the route to delete.
        :type route_name: str
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    async def adelete(self, route_name: str) -> list[str]:
        """Asynchronously delete specified route from index if it exists. Returns the IDs
        of the vectors deleted.
        This method should be implemented by subclasses.

        :param route_name: Name of the route to delete.
        :type route_name: str
        :return: List of IDs of the vectors deleted.
        :rtype: list[str]
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    def describe(self) -> IndexConfig:
        """Returns an IndexConfig object with index details such as type, dimensions,
        and total vector count.
        This method should be implemented by subclasses.

        :return: An IndexConfig object.
        :rtype: IndexConfig
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    def is_ready(self) -> bool:
        """Checks if the index is ready to be used.
        This method should be implemented by subclasses.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    async def ais_ready(self) -> bool:
        """Checks if the index is ready to be used asynchronously.
        This method should be implemented by subclasses.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    def query(self, vector: np.ndarray, top_k: int=5, route_filter: Optional[List[str]]=None, sparse_vector: dict[int, float] | SparseEmbedding | None=None) -> Tuple[np.ndarray, List[str]]:
        """Search the index for the query_vector and return top_k results.
        This method should be implemented by subclasses.

        :param vector: The vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of results to return.
        :type top_k: int
        :param route_filter: The routes to filter the search by.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: The sparse vector to search for.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple containing the query vector and a list of route names.
        :rtype: Tuple[np.ndarray, List[str]]
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    async def aquery(self, vector: np.ndarray, top_k: int=5, route_filter: Optional[List[str]]=None, sparse_vector: dict[int, float] | SparseEmbedding | None=None) -> Tuple[np.ndarray, List[str]]:
        """Search the index for the query_vector and return top_k results.
        This method should be implemented by subclasses.

        :param vector: The vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of results to return.
        :type top_k: int
        :param route_filter: The routes to filter the search by.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: The sparse vector to search for.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple containing the query vector and a list of route names.
        :rtype: Tuple[np.ndarray, List[str]]
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    def aget_routes(self):
        """
        Asynchronously get a list of route and utterance objects currently stored in the index.
        This method should be implemented by subclasses.

        :returns: A list of tuples, each containing a route name and an associated utterance.
        :rtype: list[tuple]
        :raises NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    def delete_all(self):
        """Deletes all records from the index.
        This method should be implemented by subclasses.

        :raises NotImplementedError: If the method is not implemented by the subclass.
        """
        logger.warning('This method should be implemented by subclasses.')
        self.index = None
        self.routes = None
        self.utterances = None

    def delete_index(self):
        """Deletes or resets the index.
        This method should be implemented by subclasses.

        :raises NotImplementedError: If the method is not implemented by the subclass.
        """
        logger.warning('This method should be implemented by subclasses.')
        self.index = None

    async def adelete_index(self):
        """Deletes or resets the index asynchronously.
        This method should be implemented by subclasses.
        """
        logger.warning('This method should be implemented by subclasses.')
        self.index = None

    def _read_config(self, field: str, scope: str | None=None) -> ConfigParameter:
        """Read a config parameter from the index.

        :param field: The field to read.
        :type field: str
        :param scope: The scope to read.
        :type scope: str | None
        :return: The config parameter that was read.
        :rtype: ConfigParameter
        """
        logger.warning('This method should be implemented by subclasses.')
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
        logger.warning('_async_read_config method not implemented.')
        return self._read_config(field=field, scope=scope)

    def _write_config(self, config: ConfigParameter) -> ConfigParameter:
        """Write a config parameter to the index.

        :param config: The config parameter to write.
        :type config: ConfigParameter
        :return: The config parameter that was written.
        :rtype: ConfigParameter
        """
        logger.warning('This method should be implemented by subclasses.')
        return config

    async def _async_write_config(self, config: ConfigParameter) -> ConfigParameter:
        """Write a config parameter to the index asynchronously.

        :param config: The config parameter to write.
        :type config: ConfigParameter
        :return: The config parameter that was written.
        :rtype: ConfigParameter
        """
        logger.warning('Async method not implemented.')
        return self._write_config(config=config)

    def _read_hash(self) -> ConfigParameter:
        """Read the hash of the previously written index.

        :return: The config parameter that was read.
        :rtype: ConfigParameter
        """
        return self._read_config(field='sr_hash')

    async def _async_read_hash(self) -> ConfigParameter:
        """Read the hash of the previously written index asynchronously.

        :return: The config parameter that was read.
        :rtype: ConfigParameter
        """
        return await self._async_read_config(field='sr_hash')

    def _is_locked(self, scope: str | None=None) -> bool:
        """Check if the index is locked for a given scope (if applicable).

        :param scope: The scope to check.
        :type scope: str | None
        :return: True if the index is locked, False otherwise.
        :rtype: bool
        """
        lock_config = self._read_config(field='sr_lock', scope=scope)
        if lock_config.value == 'True':
            return True
        elif lock_config.value == 'False' or not lock_config.value:
            return False
        else:
            raise ValueError(f'Invalid lock value: {lock_config.value}')

    async def _ais_locked(self, scope: str | None=None) -> bool:
        """Check if the index is locked for a given scope (if applicable).

        :param scope: The scope to check.
        :type scope: str | None
        :return: True if the index is locked, False otherwise.
        :rtype: bool
        """
        lock_config = await self._async_read_config(field='sr_lock', scope=scope)
        if lock_config.value == 'True':
            return True
        elif lock_config.value == 'False' or not lock_config.value:
            return False
        else:
            raise ValueError(f'Invalid lock value: {lock_config.value}')

    def lock(self, value: bool, wait: int=0, scope: str | None=None) -> ConfigParameter:
        """Lock/unlock the index for a given scope (if applicable). If index
        already locked/unlocked, raises ValueError.

        :param scope: The scope to lock.
        :type scope: str | None
        :param wait: The number of seconds to wait for the index to be unlocked, if
        set to 0, will raise an error if index is already locked/unlocked.
        :type wait: int
        :return: The config parameter that was locked.
        :rtype: ConfigParameter
        """
        start_time = datetime.now()
        while True:
            if self._is_locked(scope=scope) != value:
                break
            elif not value:
                break
            if (datetime.now() - start_time).total_seconds() < wait:
                time.sleep(RETRY_WAIT_TIME)
            else:
                raise ValueError(f'Index is already {('locked' if value else 'unlocked')}.')
        lock_param = ConfigParameter(field='sr_lock', value=str(value), scope=scope)
        self._write_config(lock_param)
        return lock_param

    async def alock(self, value: bool, wait: int=0, scope: str | None=None) -> ConfigParameter:
        """Lock/unlock the index for a given scope (if applicable). If index
        already locked/unlocked, raises ValueError.
        """
        start_time = datetime.now()
        while True:
            if await self._ais_locked(scope=scope) != value:
                break
            if (datetime.now() - start_time).total_seconds() < wait:
                await asyncio.sleep(RETRY_WAIT_TIME)
            else:
                raise ValueError(f'Index is already {('locked' if value else 'unlocked')}.')
        lock_param = ConfigParameter(field='sr_lock', value=str(value), scope=scope)
        await self._async_write_config(lock_param)
        return lock_param

    def _get_all(self, prefix: Optional[str]=None, include_metadata: bool=False):
        """Retrieves all vector IDs from the index.
        This method should be implemented by subclasses.

        :param prefix: The prefix to filter the vectors by.
        :type prefix: Optional[str]
        :param include_metadata: Whether to include metadata in the response.
        :type include_metadata: bool
        :return: A tuple containing a list of vector IDs and a list of metadata dictionaries.
        :rtype: tuple[list[str], list[dict]]
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    async def _async_get_all(self, prefix: Optional[str]=None, include_metadata: bool=False) -> tuple[list[str], list[dict]]:
        """Retrieves all vector IDs from the index asynchronously.
        This method should be implemented by subclasses.

        :param prefix: The prefix to filter the vectors by.
        :type prefix: Optional[str]
        :param include_metadata: Whether to include metadata in the response.
        :type include_metadata: bool
        :return: A tuple containing a list of vector IDs and a list of metadata dictionaries.
        :rtype: tuple[list[str], list[dict]]
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    async def _async_get_routes(self) -> List[Tuple]:
        """Asynchronously gets a list of route and utterance objects currently
        stored in the index, including additional metadata.

        :return: A list of tuples, each containing route, utterance, function
        schema and additional metadata.
        :rtype: List[Tuple]
        """
        if self.index is None:
            logger.warning('Index is None, could not retrieve route info.')
            return []
        _, metadata = await self._async_get_all(include_metadata=True)
        route_info = parse_route_info(metadata=metadata)
        return route_info
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def _init_index(self, force_create: bool=False) -> Union[Any, None]:
        """Initializing the index can be done after the object has been created
        to allow for the user to set the dimensions and other parameters.

        If the index doesn't exist and the dimensions are given, the index will
        be created. If the index exists, it will be returned. If the index doesn't
        exist and the dimensions are not given, the index will not be created and
        None will be returned.

        This method must be implemented by subclasses.

        :param force_create: If True, the index will be created even if the
            dimensions are not given (which will raise an error).
        :type force_create: bool, optional
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    async def _init_async_index(self, force_create: bool=False):
        """Initializing the index can be done after the object has been created
        to allow for the user to set the dimensions and other parameters.

        If the index doesn't exist and the dimensions are given, the index will
        be created. If the index exists, it will be returned. If the index doesn't
        exist and the dimensions are not given, the index will not be created and
        None will be returned.

        This method is used to initialize the index asynchronously.

        This method must be implemented by subclasses.

        :param force_create: If True, the index will be created even if the
            dimensions are not given (which will raise an error).
        :type force_create: bool, optional
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    def __len__(self):
        """Returns the total number of vectors in the index. If the index is not initialized
        returns 0.

        :return: The total number of vectors.
        :rtype: int
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    async def alen(self):
        """Async version of __len__. Returns the total number of vectors in the index.
        Default implementation just calls the sync version.

        :return: The total number of vectors.
        :rtype: int
        """
        return len(self)

class QdrantIndex(BaseIndex):
    """The name of the collection to use"""
    index_name: str = Field(default=DEFAULT_COLLECTION_NAME, description=f"Name of the Qdrant collection.Default: '{DEFAULT_COLLECTION_NAME}'")
    location: Optional[str] = Field(default=':memory:', description="If ':memory:' - use an in-memory Qdrant instance.Used as 'url' value otherwise")
    url: Optional[str] = Field(default=None, description='Qualified URL of the Qdrant instance.Optional[scheme], host, Optional[port], Optional[prefix]')
    port: Optional[int] = Field(default=6333, description='Port of the REST API interface.')
    grpc_port: int = Field(default=6334, description='Port of the gRPC interface.')
    prefer_grpc: Optional[bool] = Field(default=None, description='Whether to use gPRC interface whenever possible in methods')
    https: Optional[bool] = Field(default=None, description='Whether to use HTTPS(SSL) protocol.')
    api_key: Optional[str] = Field(default=None, description='API key for authentication in Qdrant Cloud.')
    prefix: Optional[str] = Field(default=None, description='Prefix to the REST URL path. Example: `http://localhost:6333/some/prefix/{qdrant-endpoint}`.')
    timeout: Optional[int] = Field(default=None, description='Timeout for REST and gRPC API requests.')
    host: Optional[str] = Field(default=None, description="Host name of Qdrant service.If url and host are None, set to 'localhost'.")
    path: Optional[str] = Field(default=None, description='Persistence path for Qdrant local')
    grpc_options: Optional[Dict[str, Any]] = Field(default=None, description='Options to be passed to the low-level GRPC client, if used.')
    dimensions: Union[int, None] = Field(default=None, description='Embedding dimensions.Defaults to the embedding length of the configured encoder.')
    metric: Metric = Field(default=Metric.COSINE, description='Distance metric to use for similarity search.')
    config: Optional[Dict[str, Any]] = Field(default={}, description='Collection options passed to `QdrantClient#create_collection`.')
    client: Any = Field(default=None, exclude=True)
    aclient: Any = Field(default=None, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = 'qdrant'
        self.client, self.aclient = self._initialize_clients()

    def _initialize_clients(self):
        """Initialize the clients for the Qdrant index.

        :return: A tuple of the sync and async clients.
        :rtype: Tuple[QdrantClient, Optional[AsyncQdrantClient]]
        """
        try:
            from qdrant_client import AsyncQdrantClient, QdrantClient
            sync_client = QdrantClient(location=self.location, url=self.url, port=self.port, grpc_port=self.grpc_port, prefer_grpc=self.prefer_grpc, https=self.https, api_key=self.api_key, prefix=self.prefix, timeout=self.timeout, host=self.host, path=self.path, grpc_options=self.grpc_options)
            async_client: Optional[AsyncQdrantClient] = None
            if all([self.location != ':memory:', self.path is None]):
                async_client = AsyncQdrantClient(location=self.location, url=self.url, port=self.port, grpc_port=self.grpc_port, prefer_grpc=self.prefer_grpc, https=self.https, api_key=self.api_key, prefix=self.prefix, timeout=self.timeout, host=self.host, path=self.path, grpc_options=self.grpc_options)
            return (sync_client, async_client)
        except ImportError as e:
            raise ImportError("Please install 'qdrant-client' to use QdrantIndex.You can install it with: `pip install 'semantic-router[qdrant]'`") from e

    def _init_collection(self) -> None:
        """Initialize the collection for the Qdrant index.

        :return: None
        :rtype: None
        """
        from qdrant_client import QdrantClient, models
        self.client: QdrantClient
        if not self.client.collection_exists(self.index_name):
            if not self.dimensions:
                raise ValueError('Cannot create a collection without specifying the dimensions.')
            self.client.create_collection(collection_name=self.index_name, vectors_config=models.VectorParams(size=self.dimensions, distance=self.convert_metric(self.metric)), **self.config)

    def _remove_and_sync(self, routes_to_delete: dict):
        """Remove and sync the index.

        :param routes_to_delete: The routes to delete.
        :type routes_to_delete: dict
        """
        logger.error('Sync remove is not implemented for QdrantIndex.')

    def add(self, embeddings: List[List[float]], routes: List[str], utterances: List[str], function_schemas: Optional[List[Dict[str, Any]]]=None, metadata_list: List[Dict[str, Any]]=[], batch_size: int=DEFAULT_UPLOAD_BATCH_SIZE, **kwargs):
        """Add records to the index.

        :param embeddings: The embeddings to add.
        :type embeddings: List[List[float]]
        :param routes: The routes to add.
        :type routes: List[str]
        :param utterances: The utterances to add.
        :type utterances: List[str]
        :param function_schemas: The function schemas to add.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: The metadata to add.
        :type metadata_list: List[Dict[str, Any]]
        :param batch_size: The batch size to use for the upload.
        :type batch_size: int
        """
        self.dimensions = self.dimensions or len(embeddings[0])
        self._init_collection()
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f'{route}:{utterance}')) for route, utterance in zip(routes, utterances)]
        if not metadata_list or len(metadata_list) != len(utterances):
            metadata_list = [{} for _ in utterances]
        payloads = [{SR_ROUTE_PAYLOAD_KEY: route, SR_UTTERANCE_PAYLOAD_KEY: utterance, 'metadata': metadata if metadata is not None else {}} for route, utterance, metadata in zip(routes, utterances, metadata_list)]
        self.client.upload_collection(self.index_name, vectors=embeddings, payload=payloads, ids=ids, batch_size=batch_size)

    def get_utterances(self, include_metadata: bool=False) -> List[Utterance]:
        """Gets a list of route and utterance objects currently stored in the index.

        :param include_metadata: Whether to include function schemas and metadata in
        the returned Utterance objects - QdrantIndex does not currently support this
        parameter so it is ignored. If required for your use-case please reach out to
        semantic-router maintainers on GitHub via an issue or PR.
        :type include_metadata: bool
        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        if not self.client.collection_exists(self.index_name):
            return []
        from qdrant_client import grpc
        results = []
        next_offset = None
        stop_scrolling = False
        try:
            while not stop_scrolling:
                records, next_offset = self.client.scroll(self.index_name, limit=SCROLL_SIZE, offset=next_offset, with_payload=True)
                stop_scrolling = next_offset is None or (isinstance(next_offset, grpc.PointId) and next_offset.num == 0 and (next_offset.uuid == ''))
                results.extend(records)
            utterances: List[Utterance] = [Utterance(route=x.payload[SR_ROUTE_PAYLOAD_KEY], utterance=x.payload[SR_UTTERANCE_PAYLOAD_KEY], function_schemas=None, metadata=x.payload.get('metadata', {})) for x in results]
        except ValueError as e:
            logger.warning(f'Index likely empty, error: {e}')
            return []
        return utterances

    def delete(self, route_name: str):
        """Delete records from the index.

        :param route_name: The name of the route to delete.
        :type route_name: str
        """
        from qdrant_client import models
        self.client.delete(self.index_name, points_selector=models.Filter(must=[models.FieldCondition(key=SR_ROUTE_PAYLOAD_KEY, match=models.MatchText(text=route_name))]))

    def describe(self) -> IndexConfig:
        """Describe the index.

        :return: The index configuration.
        :rtype: IndexConfig
        """
        collection_info = self.client.get_collection(self.index_name)
        return IndexConfig(type=self.type, dimensions=collection_info.config.params.vectors.size, vectors=collection_info.points_count)

    def is_ready(self) -> bool:
        """Checks if the index is ready to be used.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        return self.client.collection_exists(self.index_name)

    def query(self, vector: np.ndarray, top_k: int=5, route_filter: Optional[List[str]]=None, sparse_vector: dict[int, float] | SparseEmbedding | None=None) -> Tuple[np.ndarray, List[str]]:
        """Query the index.

        :param vector: The vector to query.
        :type vector: np.ndarray
        :param top_k: The number of results to return.
        :type top_k: int
        :param route_filter: The route filter to apply.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: The sparse vector to query.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple of the scores and route names.
        :rtype: Tuple[np.ndarray, List[str]]
        """
        from qdrant_client import QdrantClient, models
        self.client: QdrantClient
        filter = None
        if route_filter is not None:
            filter = models.Filter(must=[models.FieldCondition(key=SR_ROUTE_PAYLOAD_KEY, match=models.MatchAny(any=route_filter))])
        results = self.client.query_points(self.index_name, query=vector, limit=top_k, with_payload=True, query_filter=filter)
        scores = [result.score for result in results.points]
        route_names = [result.payload[SR_ROUTE_PAYLOAD_KEY] for result in results.points]
        return (np.array(scores), route_names)

    async def aquery(self, vector: np.ndarray, top_k: int=5, route_filter: Optional[List[str]]=None, sparse_vector: dict[int, float] | SparseEmbedding | None=None) -> Tuple[np.ndarray, List[str]]:
        """Asynchronously query the index.

        :param vector: The vector to query.
        :type vector: np.ndarray
        :param top_k: The number of results to return.
        :type top_k: int
        :param route_filter: The route filter to apply.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: The sparse vector to query.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple of the scores and route names.
        :rtype: Tuple[np.ndarray, List[str]]
        """
        from qdrant_client import AsyncQdrantClient, models
        self.aclient: Optional[AsyncQdrantClient]
        if self.aclient is None:
            logger.warning('Cannot use async query with an in-memory Qdrant instance')
            return self.query(vector, top_k, route_filter)
        filter = None
        if route_filter is not None:
            filter = models.Filter(must=[models.FieldCondition(key=SR_ROUTE_PAYLOAD_KEY, match=models.MatchAny(any=route_filter))])
        results = await self.aclient.query_points(self.index_name, query=vector, limit=top_k, with_payload=True, query_filter=filter)
        scores = [result.score for result in results.points]
        route_names = [result.payload[SR_ROUTE_PAYLOAD_KEY] for result in results.points]
        return (np.array(scores), route_names)

    def aget_routes(self):
        """Asynchronously get all routes from the index.

        :return: A list of routes.
        :rtype: List[str]
        """
        logger.error('Sync remove is not implemented for QdrantIndex.')

    def delete_index(self):
        """Delete the index.

        :return: None
        :rtype: None
        """
        self.client.delete_collection(self.index_name)

    def convert_metric(self, metric: Metric):
        """Convert the metric to a Qdrant distance metric.

        :param metric: The metric to convert.
        :type metric: Metric
        :return: The converted metric.
        :rtype: Distance
        """
        from qdrant_client.models import Distance
        mapping = {Metric.COSINE: Distance.COSINE, Metric.EUCLIDEAN: Distance.EUCLID, Metric.DOTPRODUCT: Distance.DOT, Metric.MANHATTAN: Distance.MANHATTAN}
        if metric not in mapping:
            raise ValueError(f'Unsupported Qdrant similarity metric: {metric}')
        return mapping[metric]

    def _init_config_collection(self):
        """Ensure the config collection exists."""
        from qdrant_client import models
        if not self.client.collection_exists('sr_config'):
            self.client.create_collection(collection_name='sr_config', vectors_config=models.VectorParams(size=1, distance=self.convert_metric(self.metric)))

    def _config_point_id(self, field: str, scope: str | None=None) -> str:
        """Generate a deterministic UUID string for config/hash/lock points."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f'{field}#{scope or self.index_name}'))

    def _write_config(self, config: ConfigParameter):
        """Write a config parameter to the Qdrant config collection."""
        self._init_config_collection()
        from qdrant_client import models
        point_id = self._config_point_id(config.field, config.scope)
        payload = {'field': config.field, 'scope': config.scope or self.index_name, 'value': config.value, 'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        self.client.upsert(collection_name='sr_config', points=[models.PointStruct(id=point_id, vector=[0.0], payload=payload)])
        return config

    def _read_config(self, field: str, scope: str | None=None) -> ConfigParameter:
        """Read a config parameter from the Qdrant config collection."""
        self._init_config_collection()
        point_id = self._config_point_id(field, scope)
        res = self.client.retrieve(collection_name='sr_config', ids=[point_id], with_payload=True)
        if res:
            payload = res[0].payload
            return ConfigParameter(field=payload.get('field', field), value=payload.get('value', ''), created_at=payload.get('created_at'), scope=payload.get('scope', scope or self.index_name))
        else:
            logger.warning(f'Configuration for {field} parameter not found in Qdrant.')
            return ConfigParameter(field=field, value='', scope=scope or self.index_name)

    async def _async_write_config(self, config: ConfigParameter):
        self._init_config_collection()
        from qdrant_client import models
        point_id = self._config_point_id(config.field, config.scope)
        payload = {'field': config.field, 'scope': config.scope or self.index_name, 'value': config.value, 'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        if self.aclient is None:
            return self._write_config(config)
        await self.aclient.upsert(collection_name='sr_config', points=[models.PointStruct(id=point_id, vector=[0.0], payload=payload)])
        return config

    async def _async_read_config(self, field: str, scope: str | None=None):
        self._init_config_collection()
        point_id = self._config_point_id(field, scope)
        if self.aclient is None:
            return self._read_config(field, scope)
        res = await self.aclient.retrieve(collection_name='sr_config', ids=[point_id], with_payload=True)
        if res:
            payload = res[0].payload
            return ConfigParameter(field=payload.get('field', field), value=payload.get('value', ''), created_at=payload.get('created_at'), scope=payload.get('scope', scope or self.index_name))
        else:
            logger.warning(f'Configuration for {field} parameter not found in Qdrant.')
            return ConfigParameter(field=field, value='', scope=scope or self.index_name)

    def __len__(self):
        """Returns the total number of vectors in the index. If the index is not initialized
        returns 0.

        :return: The total number of vectors.
        :rtype: int
        """
        try:
            return self.client.get_collection(self.index_name).points_count
        except ValueError as e:
            logger.warning(f'No collection found, {e}')
            return 0

    async def adelete(self, route_name: str) -> list[str]:
        """Asynchronously delete records from the index by route name.

        :param route_name: The name of the route to delete.
        :type route_name: str
        :return: List of IDs of the vectors deleted (empty list, as Qdrant does not return IDs).
        :rtype: list[str]
        """
        from qdrant_client import models
        if self.aclient is None:
            logger.warning('Cannot use async delete with an in-memory Qdrant instance; falling back to sync delete.')
            self.delete(route_name)
            return []
        await self.aclient.delete(self.index_name, points_selector=models.Filter(must=[models.FieldCondition(key=SR_ROUTE_PAYLOAD_KEY, match=models.MatchText(text=route_name))]))
        return []

    async def adelete_index(self):
        """Asynchronously delete the index (collection) from Qdrant.

        :return: None
        :rtype: None
        """
        if self.aclient is None:
            logger.warning('Cannot use async delete_index with an in-memory Qdrant instance; falling back to sync delete_index.')
            self.delete_index()
            return
        await self.aclient.delete_collection(self.index_name)

    async def ais_ready(self) -> bool:
        """Checks if the index is ready to be used asynchronously."""
        if self.aclient is None:
            return False
        try:
            return await self.aclient.collection_exists(self.index_name)
        except Exception as e:
            logger.warning(f'Async QdrantIndex readiness check failed: {e}')
            return False

    async def aadd(self, embeddings: List[List[float]], routes: List[str], utterances: List[str], function_schemas: Optional[List[Dict[str, Any]]]=None, metadata_list: List[Dict[str, Any]]=[], batch_size: int=DEFAULT_UPLOAD_BATCH_SIZE, **kwargs):
        """Asynchronously add records to the index, including metadata in the payload."""
        self.dimensions = self.dimensions or len(embeddings[0])
        if self.aclient is None:
            logger.warning('Cannot use async add with an in-memory Qdrant instance; falling back to sync add.')
            return self.add(embeddings, routes, utterances, function_schemas, metadata_list, batch_size, **kwargs)
        if not metadata_list or len(metadata_list) != len(utterances):
            metadata_list = [{} for _ in utterances]
        payloads = [{SR_ROUTE_PAYLOAD_KEY: route, SR_UTTERANCE_PAYLOAD_KEY: utterance, 'metadata': metadata if metadata is not None else {}} for route, utterance, metadata in zip(routes, utterances, metadata_list)]
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f'{route}:{utterance}')) for route, utterance in zip(routes, utterances)]
        await self.aclient.upload_collection(self.index_name, vectors=embeddings, payload=payloads, ids=ids, batch_size=batch_size)

    async def aget_utterances(self, include_metadata: bool=False) -> List[Utterance]:
        """Asynchronously gets a list of route and utterance objects currently stored in the index, including metadata."""
        if self.aclient is None:
            logger.warning('Cannot use async get_utterances with an in-memory Qdrant instance; falling back to sync get_utterances.')
            return self.get_utterances(include_metadata=include_metadata)
        from qdrant_client import grpc
        results = []
        next_offset = None
        stop_scrolling = False
        try:
            while not stop_scrolling:
                records, next_offset = await self.aclient.scroll(self.index_name, limit=SCROLL_SIZE, offset=next_offset, with_payload=True)
                stop_scrolling = next_offset is None or (isinstance(next_offset, grpc.PointId) and next_offset.num == 0 and (next_offset.uuid == ''))
                results.extend(records)
            utterances: List[Utterance] = [Utterance(route=x.payload[SR_ROUTE_PAYLOAD_KEY], utterance=x.payload[SR_UTTERANCE_PAYLOAD_KEY], function_schemas=None, metadata=x.payload.get('metadata', {})) for x in results]
        except ValueError as e:
            logger.warning(f'Index likely empty, error: {e}')
            return []
        return utterances

class LocalIndex(BaseIndex):
    type: str = 'local'
    metadata: Optional[np.ndarray] = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        if self.metadata is None:
            self.metadata = None
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def add(self, embeddings: List[List[float]], routes: List[str], utterances: List[str], function_schemas: Optional[List[Dict[str, Any]]]=None, metadata_list: List[Dict[str, Any]]=[], **kwargs):
        """Add embeddings to the index.

        :param embeddings: List of embeddings to add to the index.
        :type embeddings: List[List[float]]
        :param routes: List of routes to add to the index.
        :type routes: List[str]
        :param utterances: List of utterances to add to the index.
        :type utterances: List[str]
        :param function_schemas: List of function schemas to add to the index.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: List of metadata to add to the index.
        :type metadata_list: List[Dict[str, Any]]
        """
        embeds = np.array(embeddings)
        routes_arr = np.array(routes)
        if isinstance(utterances[0], str):
            utterances_arr = np.array(utterances)
        else:
            utterances_arr = np.array(utterances, dtype=object)
        if self.index is None:
            self.index = embeds
            self.routes = routes_arr
            self.utterances = utterances_arr
            self.metadata = np.array(metadata_list, dtype=object) if metadata_list else np.array([{} for _ in utterances], dtype=object)
        else:
            self.index = np.concatenate([self.index, embeds])
            self.routes = np.concatenate([self.routes, routes_arr])
            self.utterances = np.concatenate([self.utterances, utterances_arr])
            if self.metadata is not None:
                self.metadata = np.concatenate([self.metadata, np.array(metadata_list, dtype=object) if metadata_list else np.array([{} for _ in utterances], dtype=object)])
            else:
                self.metadata = np.array(metadata_list, dtype=object) if metadata_list else np.array([{} for _ in utterances], dtype=object)

    def _remove_and_sync(self, routes_to_delete: dict) -> np.ndarray:
        """Remove and sync the index.

        :param routes_to_delete: Dictionary of routes to delete.
        :type routes_to_delete: dict
        :return: A numpy array of the removed route utterances.
        :rtype: np.ndarray
        """
        if self.index is None or self.routes is None or self.utterances is None:
            raise ValueError('Index, routes, or utterances are not populated.')
        route_utterances = np.array([self.routes, self.utterances]).T
        mask = np.ones(len(route_utterances), dtype=bool)
        for route, utterances in routes_to_delete.items():
            for utterance in utterances:
                mask &= ~((route_utterances[:, 0] == route) & (route_utterances[:, 1] == utterance))
        self.index = self.index[mask]
        self.routes = self.routes[mask]
        self.utterances = self.utterances[mask]
        if self.metadata is not None:
            self.metadata = self.metadata[mask]
        return route_utterances[~mask]

    def get_utterances(self, include_metadata: bool=False) -> List[Utterance]:
        """Gets a list of route and utterance objects currently stored in the index.

        :param include_metadata: Whether to include function schemas and metadata in
        the returned Utterance objects - LocalIndex now includes metadata if present.
        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        if self.routes is None or self.utterances is None:
            return []
        if include_metadata and self.metadata is not None:
            return [Utterance(route=route, utterance=utterance, function_schemas=None, metadata=metadata) for route, utterance, metadata in zip(self.routes, self.utterances, self.metadata)]
        else:
            return [Utterance.from_tuple(x) for x in zip(self.routes, self.utterances)]

    def describe(self) -> IndexConfig:
        """Describe the index.

        :return: An IndexConfig object.
        :rtype: IndexConfig
        """
        return IndexConfig(type=self.type, dimensions=self.index.shape[1] if self.index is not None else 0, vectors=self.index.shape[0] if self.index is not None else 0)

    def is_ready(self) -> bool:
        """Checks if the index is ready to be used.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        return self.index is not None and self.routes is not None

    async def ais_ready(self) -> bool:
        """Checks if the index is ready to be used asynchronously.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        return self.index is not None and self.routes is not None

    def query(self, vector: np.ndarray, top_k: int=5, route_filter: Optional[List[str]]=None, sparse_vector: dict[int, float] | SparseEmbedding | None=None) -> Tuple[np.ndarray, List[str]]:
        """Search the index for the query and return top_k results.

        :param vector: The vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of results to return.
        :type top_k: int
        :param route_filter: The routes to filter the search by.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: The sparse vector to search for.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple containing the query vector and a list of route names.
        :rtype: Tuple[np.ndarray, List[str]]
        """
        if self.index is None or self.routes is None:
            raise ValueError('Index or routes are not populated.')
        if route_filter is not None:
            filtered_index = []
            filtered_routes = []
            for route, vec in zip(self.routes, self.index):
                if route in route_filter:
                    filtered_index.append(vec)
                    filtered_routes.append(route)
            if not filtered_routes:
                raise ValueError('No routes found matching the filter criteria.')
            sim = similarity_matrix(vector, np.array(filtered_index))
            scores, idx = top_scores(sim, top_k)
            route_names = [filtered_routes[i] for i in idx]
        else:
            sim = similarity_matrix(vector, self.index)
            scores, idx = top_scores(sim, top_k)
            route_names = [self.routes[i] for i in idx]
        return (scores, route_names)

    async def aquery(self, vector: np.ndarray, top_k: int=5, route_filter: Optional[List[str]]=None, sparse_vector: dict[int, float] | SparseEmbedding | None=None) -> Tuple[np.ndarray, List[str]]:
        """Search the index for the query and return top_k results.

        :param vector: The vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of results to return.
        :type top_k: int
        :param route_filter: The routes to filter the search by.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: The sparse vector to search for.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple containing the query vector and a list of route names.
        :rtype: Tuple[np.ndarray, List[str]]
        """
        if self.index is None or self.routes is None:
            raise ValueError('Index or routes are not populated.')
        if route_filter is not None:
            filtered_index = []
            filtered_routes = []
            for route, vec in zip(self.routes, self.index):
                if route in route_filter:
                    filtered_index.append(vec)
                    filtered_routes.append(route)
            if not filtered_routes:
                raise ValueError('No routes found matching the filter criteria.')
            sim = similarity_matrix(vector, np.array(filtered_index))
            scores, idx = top_scores(sim, top_k)
            route_names = [filtered_routes[i] for i in idx]
        else:
            sim = similarity_matrix(vector, self.index)
            scores, idx = top_scores(sim, top_k)
            route_names = [self.routes[i] for i in idx]
        return (scores, route_names)

    def aget_routes(self):
        """Get all routes from the index.

        :return: A list of routes.
        :rtype: List[str]
        """
        logger.error('Sync remove is not implemented for LocalIndex.')

    def _write_config(self, config: ConfigParameter):
        """Write the config to the index.

        :param config: The config to write to the index.
        :type config: ConfigParameter
        """
        logger.warning('No config is written for LocalIndex.')

    def delete(self, route_name: str):
        """Delete all records of a specific route from the index.

        :param route_name: The name of the route to delete.
        :type route_name: str
        """
        if self.index is not None and self.routes is not None and (self.utterances is not None):
            delete_idx = self._get_indices_for_route(route_name=route_name)
            self.index = np.delete(self.index, delete_idx, axis=0)
            self.routes = np.delete(self.routes, delete_idx, axis=0)
            self.utterances = np.delete(self.utterances, delete_idx, axis=0)
            if self.metadata is not None:
                self.metadata = np.delete(self.metadata, delete_idx, axis=0)
        else:
            raise ValueError('Attempted to delete route records but either index, routes or utterances is None.')

    async def adelete(self, route_name: str):
        """Delete all records of a specific route from the index. Note that this just points
        to the sync delete method as async makes no difference for the local computations
        of the LocalIndex.

        :param route_name: The name of the route to delete.
        :type route_name: str
        """
        self.delete(route_name)

    def delete_index(self):
        """Deletes the index, effectively clearing it and setting it to None.

        :return: None
        :rtype: None
        """
        self.index = None
        self.routes = None
        self.utterances = None
        self.metadata = None

    async def adelete_index(self):
        """Deletes the index, effectively clearing it and setting it to None. Note that this just points
        to the sync delete_index method as async makes no difference for the local computations
        of the LocalIndex.

        :return: None
        :rtype: None
        """
        self.index = None
        self.routes = None
        self.utterances = None
        self.metadata = None

    def _get_indices_for_route(self, route_name: str):
        """Gets an array of indices for a specific route.

        :param route_name: The name of the route to get indices for.
        :type route_name: str
        :return: An array of indices for the route.
        :rtype: np.ndarray
        """
        if self.routes is None:
            raise ValueError('Routes are not populated.')
        idx = [i for i, route in enumerate(self.routes) if route == route_name]
        return idx

    def __len__(self):
        if self.index is not None:
            return self.index.shape[0]
        else:
            return 0

class HybridLocalIndex(LocalIndex):
    type: str = 'hybrid_local'
    sparse_index: Optional[list[dict]] = None
    route_names: Optional[np.ndarray] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.metadata = None

    def add(self, embeddings: List[List[float]], routes: List[str], utterances: List[str], function_schemas: Optional[List[Dict[str, Any]]]=None, metadata_list: List[Dict[str, Any]]=[], sparse_embeddings: Optional[List[SparseEmbedding]]=None, **kwargs):
        """Add embeddings to the index.

        :param embeddings: List of embeddings to add to the index.
        :type embeddings: List[List[float]]
        :param routes: List of routes to add to the index.
        :type routes: List[str]
        :param utterances: List of utterances to add to the index.
        :type utterances: List[str]
        :param function_schemas: List of function schemas to add to the index.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: List of metadata to add to the index.
        :type metadata_list: List[Dict[str, Any]]
        :param sparse_embeddings: List of sparse embeddings to add to the index.
        :type sparse_embeddings: Optional[List[SparseEmbedding]]
        """
        if sparse_embeddings is None:
            raise ValueError('Sparse embeddings are required for HybridLocalIndex.')
        if function_schemas is not None:
            logger.warning('Function schemas are not supported for HybridLocalIndex.')
        if metadata_list:
            logger.warning('Metadata is not supported for HybridLocalIndex.')
        embeds = np.array(embeddings)
        routes_arr = np.array(routes)
        if isinstance(utterances[0], str):
            utterances_arr = np.array(utterances)
        else:
            utterances_arr = np.array(utterances, dtype=object)
        if self.index is None or self.sparse_index is None:
            self.index = embeds
            self.sparse_index = [x.to_dict() for x in sparse_embeddings]
            self.routes = routes_arr
            self.utterances = utterances_arr
            self.metadata = np.array(metadata_list, dtype=object) if metadata_list else np.array([{} for _ in utterances], dtype=object)
        else:
            self.index = np.concatenate([self.index, embeds])
            self.sparse_index.extend([x.to_dict() for x in sparse_embeddings])
            self.routes = np.concatenate([self.routes, routes_arr])
            self.utterances = np.concatenate([self.utterances, utterances_arr])
            if self.metadata is not None:
                self.metadata = np.concatenate([self.metadata, np.array(metadata_list, dtype=object) if metadata_list else np.array([{} for _ in utterances], dtype=object)])
            else:
                self.metadata = np.array(metadata_list, dtype=object) if metadata_list else np.array([{} for _ in utterances], dtype=object)

    async def aadd(self, embeddings: List[List[float]], routes: List[str], utterances: List[str], function_schemas: Optional[List[Dict[str, Any]]]=None, metadata_list: List[Dict[str, Any]]=[], sparse_embeddings: Optional[List[SparseEmbedding]]=None, **kwargs):
        """Add embeddings to the index - note that this is not truly async as it is a
        local index and there is no sense to make this method async. Instead, it will
        call the sync `add` method.

        :param embeddings: List of embeddings to add to the index.
        :type embeddings: List[List[float]]
        :param routes: List of routes to add to the index.
        :type routes: List[str]
        :param utterances: List of utterances to add to the index.
        :type utterances: List[str]
        :param function_schemas: List of function schemas to add to the index.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: List of metadata to add to the index.
        :type metadata_list: List[Dict[str, Any]]
        :param sparse_embeddings: List of sparse embeddings to add to the index.
        :type sparse_embeddings: Optional[List[SparseEmbedding]]
        """
        self.add(embeddings=embeddings, routes=routes, utterances=utterances, function_schemas=function_schemas, metadata_list=metadata_list, sparse_embeddings=sparse_embeddings)

    def get_utterances(self, include_metadata: bool=False) -> List[Utterance]:
        """Gets a list of route and utterance objects currently stored in the index.

        :param include_metadata: Whether to include function schemas and metadata in
        the returned Utterance objects - HybridLocalIndex doesn't include metadata so
        this parameter is ignored.
        :type include_metadata: bool
        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        if self.routes is None or self.utterances is None:
            return []
        if include_metadata and self.metadata is not None:
            return [Utterance(route=route, utterance=utterance, function_schemas=None, metadata=metadata) for route, utterance, metadata in zip(self.routes, self.utterances, self.metadata)]
        else:
            return [Utterance.from_tuple(x) for x in zip(self.routes, self.utterances)]

    def _sparse_dot_product(self, vec_a: dict[int, float], vec_b: dict[int, float]) -> float:
        """Calculate the dot product of two sparse vectors.

        :param vec_a: The first sparse vector.
        :type vec_a: dict[int, float]
        :param vec_b: The second sparse vector.
        :type vec_b: dict[int, float]
        :return: The dot product of the two sparse vectors.
        :rtype: float
        """
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = (vec_b, vec_a)
        return sum((vec_a[i] * vec_b.get(i, 0) for i in vec_a))

    def _sparse_index_dot_product(self, vec_a: dict[int, float]) -> list[float]:
        """Calculate the dot product of a sparse vector and a list of sparse vectors.

        :param vec_a: The sparse vector.
        :type vec_a: dict[int, float]
        :return: A list of dot products.
        :rtype: list[float]
        """
        if self.sparse_index is None:
            raise ValueError('self.sparse_index is not populated.')
        dot_products = [self._sparse_dot_product(vec_a, vec_b) for vec_b in self.sparse_index]
        return dot_products

    def query(self, vector: np.ndarray, top_k: int=5, route_filter: Optional[List[str]]=None, sparse_vector: dict[int, float] | SparseEmbedding | None=None) -> Tuple[np.ndarray, List[str]]:
        """Search the index for the query and return top_k results.

        :param vector: The query vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of top results to return, defaults to 5.
        :type top_k: int, optional
        :param route_filter: A list of route names to filter the search results, defaults to None.
        :type route_filter: Optional[List[str]], optional
        :param sparse_vector: The sparse vector to search for, must be provided.
        :type sparse_vector: dict[int, float]
        """
        if route_filter:
            raise ValueError('Route filter is not supported for HybridLocalIndex.')
        xq_d = vector.copy()
        if isinstance(sparse_vector, SparseEmbedding):
            xq_s = sparse_vector.to_dict()
        elif isinstance(sparse_vector, dict):
            xq_s = sparse_vector
        else:
            raise ValueError('Sparse vector must be a SparseEmbedding or dict.')
        if self.index is not None and self.sparse_index is not None:
            index_norm = norm(self.index, axis=1)
            xq_d_norm = norm(xq_d)
            sim_d = np.squeeze(np.dot(self.index, xq_d.T)) / (index_norm * xq_d_norm)
            sim_s = np.array(self._sparse_index_dot_product(xq_s))
            total_sim = sim_d + sim_s
            top_k = min(top_k, total_sim.shape[0])
            idx = np.argpartition(total_sim, -top_k)[-top_k:]
            scores = total_sim[idx]
            route_names = self.routes[idx] if self.routes is not None else []
            return (scores, route_names)
        else:
            logger.warning('Index or sparse index is not populated.')
            return (np.array([]), [])

    async def aquery(self, vector: np.ndarray, top_k: int=5, route_filter: Optional[List[str]]=None, sparse_vector: dict[int, float] | SparseEmbedding | None=None) -> Tuple[np.ndarray, List[str]]:
        """Search the index for the query and return top_k results. This method calls the
        sync `query` method as everything uses numpy computations which is CPU-bound
        and so no benefit can be gained from making this async.

        :param vector: The query vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of top results to return, defaults to 5.
        :type top_k: int, optional
        :param route_filter: A list of route names to filter the search results, defaults to None.
        :type route_filter: Optional[List[str]], optional
        :param sparse_vector: The sparse vector to search for, must be provided.
        :type sparse_vector: dict[int, float]
        """
        return self.query(vector=vector, top_k=top_k, route_filter=route_filter, sparse_vector=sparse_vector)

    def aget_routes(self):
        """Get all routes from the index.

        :return: A list of routes.
        :rtype: List[str]
        """
        logger.error(f'Sync remove is not implemented for {self.__class__.__name__}.')

    def _write_config(self, config: ConfigParameter):
        """Write the config to the index.

        :param config: The config to write to the index.
        :type config: ConfigParameter
        """
        logger.warning(f'No config is written for {self.__class__.__name__}.')

    def delete(self, route_name: str):
        """Delete all records of a specific route from the index.

        :param route_name: The name of the route to delete.
        :type route_name: str
        """
        if self.index is not None and self.routes is not None and (self.utterances is not None):
            delete_idx = self._get_indices_for_route(route_name=route_name)
            self.index = np.delete(self.index, delete_idx, axis=0)
            self.routes = np.delete(self.routes, delete_idx, axis=0)
            self.utterances = np.delete(self.utterances, delete_idx, axis=0)
            if self.metadata is not None:
                self.metadata = np.delete(self.metadata, delete_idx, axis=0)
        else:
            raise ValueError('Attempted to delete route records but either index, routes or utterances is None.')

    def delete_index(self):
        """Deletes the index, effectively clearing it and setting it to None.

        :return: None
        :rtype: None
        """
        self.index = None
        self.routes = None
        self.utterances = None
        self.metadata = None

    def _get_indices_for_route(self, route_name: str):
        """Gets an array of indices for a specific route.

        :param route_name: The name of the route to get indices for.
        :type route_name: str
        :return: An array of indices for the route.
        :rtype: np.ndarray
        """
        if self.routes is None:
            raise ValueError('Routes are not populated.')
        idx = [i for i, route in enumerate(self.routes) if route == route_name]
        return idx

    def __len__(self):
        if self.index is not None:
            return self.index.shape[0]
        else:
            return 0

class RouterConfig:
    """Generates a RouterConfig object that can be used for initializing routers."""
    routes: List[Route] = Field(default_factory=list)
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, routes: List[Route]=[], encoder_type: str='openai', encoder_name: Optional[str]=None):
        """Initialize a RouterConfig object.

        :param routes: A list of routes.
        :type routes: List[Route]
        :param encoder_type: The type of encoder to use.
        :type encoder_type: str
        :param encoder_name: The name of the encoder to use.
        :type encoder_name: Optional[str]
        """
        self.encoder_type = encoder_type
        if encoder_name is None:
            for encode_type in EncoderType:
                if encode_type.value == self.encoder_type:
                    if self.encoder_type == EncoderType.HUGGINGFACE.value:
                        raise NotImplementedError('HuggingFace encoder not supported by RouterConfig yet.')
                    encoder_name = EncoderDefault[encode_type.name].value['embedding_model']
                    break
            logger.info(f'Using default {encoder_type} encoder: {encoder_name}')
        self.encoder_name = encoder_name
        self.routes = routes

    @classmethod
    def from_file(cls, path: str) -> 'RouterConfig':
        """Initialize a RouterConfig from a file. Expects a JSON or YAML file with file
        extension .json, .yaml, or .yml.

        :param path: The path to the file to load the RouterConfig from.
        :type path: str
        """
        logger.info(f'Loading route config from {path}')
        _, ext = os.path.splitext(path)
        with open(path, 'r') as f:
            if ext == '.json':
                layer = json.load(f)
            elif ext in ['.yaml', '.yml']:
                layer = yaml.safe_load(f)
            else:
                raise ValueError('Unsupported file type. Only .json and .yaml are supported')
            if not is_valid(json.dumps(layer)):
                raise Exception('Invalid config JSON or YAML')
            encoder_type = layer['encoder_type']
            encoder_name = layer['encoder_name']
            routes = []
            for route_data in layer['routes']:
                if 'llm' in route_data and route_data['llm'] is not None:
                    llm_data = route_data.pop('llm')
                    llm_module_path = llm_data['module']
                    llm_module = importlib.import_module(llm_module_path)
                    llm_class = getattr(llm_module, llm_data['class'])
                    llm = llm_class(name=llm_data['model'])
                    route_data['llm'] = llm
                route = Route(**route_data)
                routes.append(route)
            return cls(encoder_type=encoder_type, encoder_name=encoder_name, routes=routes)

    @classmethod
    def from_tuples(cls, route_tuples: List[Tuple[str, str, Optional[List[Dict[str, Any]]], Dict[str, Any]]], encoder_type: str='openai', encoder_name: Optional[str]=None):
        """Initialize a RouterConfig from a list of tuples of routes and
        utterances.

        :param route_tuples: A list of tuples, each containing a route name and an
            associated utterance.
        :type route_tuples: List[Tuple[str, str]]
        :param encoder_type: The type of encoder to use, defaults to "openai".
        :type encoder_type: str, optional
        :param encoder_name: The name of the encoder to use, defaults to None.
        :type encoder_name: Optional[str], optional
        """
        routes_dict: Dict[str, Route] = {}
        for route_name, utterance, function_schema, metadata in route_tuples:
            if route_name not in routes_dict:
                routes_dict[route_name] = Route(name=route_name, utterances=[utterance], function_schemas=function_schema, metadata=metadata)
            else:
                routes_dict[route_name].utterances.append(utterance)
        routes: List[Route] = []
        for route_name, route in routes_dict.items():
            routes.append(route)
        return cls(routes=routes, encoder_type=encoder_type, encoder_name=encoder_name)

    @classmethod
    def from_index(cls, index: BaseIndex, encoder_type: str='openai', encoder_name: Optional[str]=None):
        """Initialize a RouterConfig from a BaseIndex object.

        :param index: The index to initialize the RouterConfig from.
        :type index: BaseIndex
        :param encoder_type: The type of encoder to use, defaults to "openai".
        :type encoder_type: str, optional
        :param encoder_name: The name of the encoder to use, defaults to None.
        :type encoder_name: Optional[str], optional
        """
        remote_routes = index.get_utterances(include_metadata=True)
        return cls.from_tuples(route_tuples=[utt.to_tuple() for utt in remote_routes], encoder_type=encoder_type, encoder_name=encoder_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the RouterConfig to a dictionary.

        :return: A dictionary representation of the RouterConfig.
        :rtype: Dict[str, Any]
        """
        return {'encoder_type': self.encoder_type, 'encoder_name': self.encoder_name, 'routes': [route.to_dict() for route in self.routes]}

    def to_file(self, path: str):
        """Save the routes to a file in JSON or YAML format.

        :param path: The path to save the RouterConfig to.
        :type path: str
        """
        logger.info(f'Saving route config to {path}')
        _, ext = os.path.splitext(path)
        if ext not in ['.json', '.yaml', '.yml']:
            raise ValueError('Unsupported file type. Only .json and .yaml are supported')
        dir_name = os.path.dirname(path)
        if dir_name and (not os.path.exists(dir_name)):
            os.makedirs(dir_name)
        with open(path, 'w') as f:
            if ext == '.json':
                json.dump(self.to_dict(), f, indent=4)
            elif ext in ['.yaml', '.yml']:
                yaml.safe_dump(self.to_dict(), f)

    def to_utterances(self) -> List[Utterance]:
        """Convert the routes to a list of Utterance objects.

        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        utterances = []
        for route in self.routes:
            utterances.extend([Utterance(route=route.name, utterance=x, function_schemas=route.function_schemas, metadata=route.metadata or {}) for x in route.utterances])
        return utterances

    def add(self, route: Route):
        """Add a route to the RouterConfig.

        :param route: The route to add.
        :type route: Route
        """
        self.routes.append(route)
        logger.info(f'Added route `{route.name}`')

    def get(self, name: str) -> Optional[Route]:
        """Get a route from the RouterConfig by name.

        :param name: The name of the route to get.
        :type name: str
        :return: The route if found, otherwise None.
        :rtype: Optional[Route]
        """
        for route in self.routes:
            if route.name == name:
                return route
        logger.error(f'Route `{name}` not found')
        return None

    def remove(self, name: str):
        """Remove a route from the RouterConfig by name.

        :param name: The name of the route to remove.
        :type name: str
        """
        if name not in [route.name for route in self.routes]:
            logger.error(f'Route `{name}` not found')
        else:
            self.routes = [route for route in self.routes if route.name != name]
            logger.info(f'Removed route `{name}`')

    def get_hash(self) -> ConfigParameter:
        """Get the hash of the RouterConfig. Used for syncing.

        :return: The hash of the RouterConfig.
        :rtype: ConfigParameter
        """
        layer = self.to_dict()
        return ConfigParameter(field='sr_hash', value=hashlib.sha256(json.dumps(layer).encode()).hexdigest())

@pytest.mark.parametrize('index_cls,router_cls', [(index, router) for index in get_test_indexes() for router in get_test_routers()])
class TestSemanticRouter:

    @pytest.mark.skipif(os.environ.get('PINECONE_API_KEY') is None, reason='Pinecone API key required')
    def test_initialization(self, openai_encoder, routes, index_cls, router_cls):
        index = init_index(index_cls, index_name=router_cls.__name__)
        _ = router_cls(encoder=openai_encoder, routes=routes, top_k=10, index=index, auto_sync='local')

    @pytest.mark.skipif(os.environ.get('PINECONE_API_KEY') is None, reason='Pinecone API key required')
    def test_second_initialization_sync(self, openai_encoder, routes, index_cls, router_cls):
        index = init_index(index_cls, index_name=router_cls.__name__)
        route_layer = router_cls(encoder=openai_encoder, routes=routes, index=index, auto_sync='local')
        assert route_layer.is_synced()

    @pytest.mark.skipif(os.environ.get('PINECONE_API_KEY') is None, reason='Pinecone API key required')
    def test_second_initialization_not_synced(self, openai_encoder, routes, routes_2, index_cls, router_cls):
        index = init_index(index_cls, index_name=router_cls.__name__)
        _ = router_cls(encoder=openai_encoder, routes=routes, index=index, auto_sync='local')
        route_layer = router_cls(encoder=openai_encoder, routes=routes_2, index=index)
        assert route_layer.is_synced() is False

    @pytest.mark.skipif(os.environ.get('PINECONE_API_KEY') is None, reason='Pinecone API key required')
    def test_utterance_diff(self, openai_encoder, routes, routes_2, index_cls, router_cls):
        index = init_index(index_cls, index_name=router_cls.__name__)
        _ = router_cls(encoder=openai_encoder, routes=routes, index=index, auto_sync='local')
        route_layer_2 = router_cls(encoder=openai_encoder, routes=routes_2, index=index)
        diff = route_layer_2.get_utterance_diff(include_metadata=True)
        assert '+ Route 1: Hello | None | {"type": "default"}' in diff
        assert '+ Route 1: Hi | None | {"type": "default"}' in diff
        assert '- Route 1: Hello | None | {}' in diff
        assert '+ Route 2: Au revoir | None | {}' in diff
        assert '- Route 2: Hi | None | {}' in diff
        assert '+ Route 2: Bye | None | {}' in diff
        assert '+ Route 2: Goodbye | None | {}' in diff
        assert '+ Route 3: Boo | None | {}' in diff

    @pytest.mark.skipif(os.environ.get('PINECONE_API_KEY') is None, reason='Pinecone API key required')
    def test_auto_sync_local(self, openai_encoder, routes, routes_2, index_cls, router_cls):
        if index_cls is PineconeIndex:
            pinecone_index = init_index(index_cls, index_name=router_cls.__name__)
            _ = router_cls(encoder=openai_encoder, routes=routes, index=pinecone_index)
            route_layer = router_cls(encoder=openai_encoder, routes=routes_2, index=pinecone_index, auto_sync='local')
            assert route_layer.index.get_utterances(include_metadata=True) == [Utterance(route='Route 1', utterance='Hello'), Utterance(route='Route 2', utterance='Hi')], 'The routes in the index should match the local routes'

    @pytest.mark.skipif(os.environ.get('PINECONE_API_KEY') is None, reason='Pinecone API key required')
    def test_auto_sync_remote(self, openai_encoder, routes, routes_2, index_cls, router_cls):
        if index_cls is PineconeIndex:
            pinecone_index = init_index(index_cls, index_name=router_cls.__name__)
            _ = router_cls(encoder=openai_encoder, routes=routes_2, index=pinecone_index, auto_sync='local')
            route_layer = router_cls(encoder=openai_encoder, routes=routes, index=pinecone_index, auto_sync='remote')
            assert route_layer.index.get_utterances(include_metadata=True) == [Utterance(route='Route 1', utterance='Hello'), Utterance(route='Route 2', utterance='Hi')], 'The routes in the index should match the local routes'

    @pytest.mark.skipif(os.environ.get('PINECONE_API_KEY') is None, reason='Pinecone API key required')
    def test_auto_sync_merge_force_local(self, openai_encoder, routes, routes_2, index_cls, router_cls):
        if index_cls is PineconeIndex:
            pinecone_index = init_index(index_cls, index_name=router_cls.__name__)
            route_layer = router_cls(encoder=openai_encoder, routes=routes, index=pinecone_index, auto_sync='local')
            route_layer = router_cls(encoder=openai_encoder, routes=routes_2, index=pinecone_index, auto_sync='merge-force-local')
            assert route_layer.is_synced()
            local_utterances = route_layer.index.get_utterances(include_metadata=False)
            local_utterances.sort(key=lambda x: x.to_str(include_metadata=False))
            assert local_utterances == [Utterance(route='Route 1', utterance='Hello'), Utterance(route='Route 1', utterance='Hi'), Utterance(route='Route 2', utterance='Au revoir'), Utterance(route='Route 2', utterance='Bye'), Utterance(route='Route 2', utterance='Goodbye'), Utterance(route='Route 2', utterance='Hi')], 'The routes in the index should match the local routes'

    @pytest.mark.skipif(os.environ.get('PINECONE_API_KEY') is None, reason='Pinecone API key required')
    def test_auto_sync_merge_force_remote(self, openai_encoder, routes, routes_2, index_cls, router_cls):
        if index_cls is PineconeIndex:
            pinecone_index = init_index(index_cls, index_name=router_cls.__name__)
            route_layer = router_cls(encoder=openai_encoder, routes=routes, index=pinecone_index, auto_sync='local')
            assert route_layer.is_synced()
            r1_utterances = [Utterance(route='Route 1', utterance='Hello', metadata={'type': 'default'}), Utterance(route='Route 1', utterance='Hi', metadata={'type': 'default'}), Utterance(route='Route 2', utterance='Au revoir'), Utterance(route='Route 2', utterance='Bye'), Utterance(route='Route 2', utterance='Goodbye'), Utterance(route='Route 3', utterance='Boo')]
            local_utterances = route_layer.index.get_utterances(include_metadata=True)
            local_utterances.sort(key=lambda x: x.to_str(include_metadata=True))
            assert local_utterances == r1_utterances
            route_layer = router_cls(encoder=openai_encoder, routes=routes_2, index=pinecone_index, auto_sync='merge-force-remote')
            assert route_layer.is_synced()
            local_utterances = route_layer.index.get_utterances(include_metadata=True)
            local_utterances.sort(key=lambda x: x.to_str(include_metadata=True))
            assert local_utterances == [Utterance(route='Route 1', utterance='Hello', metadata={'type': 'default'}), Utterance(route='Route 1', utterance='Hi', metadata={'type': 'default'}), Utterance(route='Route 2', utterance='Au revoir'), Utterance(route='Route 2', utterance='Bye'), Utterance(route='Route 2', utterance='Goodbye'), Utterance(route='Route 2', utterance='Hi'), Utterance(route='Route 3', utterance='Boo')], 'The routes in the index should match the local routes'

    @pytest.mark.skipif(os.environ.get('PINECONE_API_KEY') is None, reason='Pinecone API key required')
    def test_sync(self, openai_encoder, index_cls, router_cls):
        route_layer = router_cls(encoder=openai_encoder, routes=[], index=init_index(index_cls, index_name=router_cls.__name__), auto_sync=None)
        route_layer.sync('remote')
        assert route_layer.is_synced()

    @pytest.mark.skipif(os.environ.get('PINECONE_API_KEY') is None, reason='Pinecone API key required')
    def test_auto_sync_merge(self, openai_encoder, routes, routes_2, index_cls, router_cls):
        if index_cls is PineconeIndex:
            pinecone_index = init_index(index_cls, index_name=router_cls.__name__)
            route_layer = router_cls(encoder=openai_encoder, routes=routes_2, index=pinecone_index, auto_sync='local')
            route_layer = router_cls(encoder=openai_encoder, routes=routes, index=pinecone_index, auto_sync='merge')
            assert route_layer.is_synced()
            local_utterances = route_layer.index.get_utterances(include_metadata=True)
            local_utterances.sort(key=lambda x: x.to_str(include_metadata=True))
            assert local_utterances == [Utterance(route='Route 1', utterance='Hello', metadata={'type': 'default'}), Utterance(route='Route 1', utterance='Hi', metadata={'type': 'default'}), Utterance(route='Route 2', utterance='Au revoir'), Utterance(route='Route 2', utterance='Bye'), Utterance(route='Route 2', utterance='Goodbye'), Utterance(route='Route 2', utterance='Hi'), Utterance(route='Route 3', utterance='Boo')], 'The routes in the index should match the local routes'

    @pytest.mark.skipif(os.environ.get('PINECONE_API_KEY') is None, reason='Pinecone API key required')
    def test_sync_lock_prevents_concurrent_sync(self, openai_encoder, routes, routes_2, index_cls, router_cls):
        """Test that sync lock prevents concurrent synchronization operations"""
        index = init_index(index_cls, index_name=router_cls.__name__)
        route_layer = router_cls(encoder=openai_encoder, routes=routes_2, index=index, auto_sync='local')
        route_layer = router_cls(encoder=openai_encoder, routes=routes, index=index, auto_sync=None)
        route_layer.index.lock(value=True)
        with pytest.raises(Exception):
            route_layer.sync('local')
        route_layer.index.lock(value=False)
        route_layer.sync('local')
        assert route_layer.is_synced()

    @pytest.mark.skipif(os.environ.get('PINECONE_API_KEY') is None, reason='Pinecone API key required')
    def test_sync_lock_auto_releases(self, openai_encoder, routes, index_cls, router_cls):
        """Test that sync lock is automatically released after sync operations"""
        index = init_index(index_cls, index_name=router_cls.__name__)
        route_layer = router_cls(encoder=openai_encoder, routes=routes, index=index, auto_sync='local')
        route_layer.sync('local')
        assert route_layer.is_synced()
        if index_cls is PineconeIndex:
            route_layer.index.client.delete_index(route_layer.index.index_name)

