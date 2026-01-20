# Cluster 23

class HybridRouter(BaseRouter):
    """A hybrid layer that uses both dense and sparse embeddings to classify routes."""
    sparse_encoder: Optional[SparseEncoder] = Field(default=None)
    alpha: float = 0.3

    def __init__(self, encoder: DenseEncoder, sparse_encoder: Optional[SparseEncoder]=None, llm: Optional[BaseLLM]=None, routes: Optional[List[Route]]=None, index: Optional[HybridLocalIndex]=None, top_k: int=5, aggregation: str='mean', auto_sync: Optional[str]=None, alpha: float=0.3, init_async_index: bool=False):
        """Initialize the HybridRouter.

        :param encoder: The dense encoder to use.
        :type encoder: DenseEncoder
        :param sparse_encoder: The sparse encoder to use.
        :type sparse_encoder: Optional[SparseEncoder]
        """
        if index is None:
            logger.warning('No index provided. Using default HybridLocalIndex.')
            index = HybridLocalIndex()
        encoder = self._get_encoder(encoder=encoder)
        sparse_encoder = self._get_sparse_encoder(sparse_encoder=sparse_encoder)
        if isinstance(sparse_encoder, FittableMixin) and routes:
            sparse_encoder.fit(routes)
        super().__init__(encoder=encoder, sparse_encoder=sparse_encoder, llm=llm, routes=routes, index=index, top_k=top_k, aggregation=aggregation, auto_sync=auto_sync, init_async_index=init_async_index)
        self.alpha = alpha

    def _set_score_threshold(self):
        """Set the score threshold for the HybridRouter. Unlike the base router the
        encoder score threshold is not used directly. Instead, the dense encoder
        score threshold is multiplied by the alpha value, resulting in a lower
        score threshold. This is done to account for the difference in returned
        scores from the hybrid router.
        """
        if self.encoder.score_threshold is not None:
            self.score_threshold = self.encoder.score_threshold * self.alpha
            if self.score_threshold is None:
                logger.warning("No score threshold value found in encoder. Using the default 'None' value can lead to unexpected results.")

    def add(self, routes: List[Route] | Route):
        """Add a route to the local HybridRouter and index.

        :param route: The route to add.
        :type route: Route
        """
        if self.sparse_encoder is None:
            raise ValueError('Sparse Encoder not initialised.')
        current_local_hash = self._get_hash()
        current_remote_hash = self.index._read_hash()
        if current_remote_hash.value == '':
            current_remote_hash = current_local_hash
        if isinstance(routes, Route):
            routes = [routes]
        self.routes.extend(routes)
        if isinstance(self.sparse_encoder, FittableMixin) and self.routes:
            self.sparse_encoder.fit(self.routes)
        route_names, all_utterances, all_function_schemas, all_metadata = self._extract_routes_details(routes, include_metadata=True)
        dense_emb, sparse_emb = self._encode(all_utterances, input_type='documents')
        self.index.add(embeddings=dense_emb.tolist(), routes=route_names, utterances=all_utterances, function_schemas=all_function_schemas, metadata_list=all_metadata, sparse_embeddings=sparse_emb)
        if current_local_hash.value == current_remote_hash.value:
            self._write_hash()
        else:
            logger.warning(f'Local and remote route layers were not aligned. Remote hash not updated. Use `{self.__class__.__name__}.get_utterance_diff()` to see details.')

    async def aadd(self, routes: List[Route] | Route):
        """Add a route to the local HybridRouter and index asynchronously.

        :param routes: The route(s) to add.
        :type routes: List[Route] | Route
        """
        if self.sparse_encoder is None:
            raise ValueError('Sparse Encoder not initialised.')
        current_local_hash = self._get_hash()
        current_remote_hash = await self.index._async_read_hash()
        if current_remote_hash.value == '':
            current_remote_hash = current_local_hash
        if isinstance(routes, Route):
            routes = [routes]
        self.routes.extend(routes)
        if isinstance(self.sparse_encoder, FittableMixin) and self.routes:
            self.sparse_encoder.fit(self.routes)
        route_names, all_utterances, all_function_schemas, all_metadata = self._extract_routes_details(routes, include_metadata=True)
        dense_emb, sparse_emb = await self._async_encode(all_utterances, input_type='documents')
        await self.index.aadd(embeddings=dense_emb.tolist(), routes=route_names, utterances=all_utterances, function_schemas=all_function_schemas, metadata_list=all_metadata, sparse_embeddings=sparse_emb)
        if current_local_hash.value == current_remote_hash.value:
            await self._async_write_hash()
        else:
            logger.warning(f'Local and remote route layers were not aligned. Remote hash not updated. Use `{self.__class__.__name__}.get_utterance_diff()` to see details.')

    def _execute_sync_strategy(self, strategy: Dict[str, Dict[str, List[Utterance]]]):
        """Executes the provided sync strategy, either deleting or upserting
        routes from the local and remote instances as defined in the strategy.

        :param strategy: The sync strategy to execute.
        :type strategy: Dict[str, Dict[str, List[Utterance]]]
        """
        if self.sparse_encoder is None:
            raise ValueError('Sparse Encoder not initialised.')
        if strategy['remote']['delete']:
            data_to_delete = {}
            for utt_obj in strategy['remote']['delete']:
                data_to_delete.setdefault(utt_obj.route, []).append(utt_obj.utterance)
            self.index._remove_and_sync(data_to_delete)
        if strategy['remote']['upsert']:
            utterances_text = [utt.utterance for utt in strategy['remote']['upsert']]
            dense_emb, sparse_emb = self._encode(utterances_text, input_type='documents')
            self.index.add(embeddings=dense_emb.tolist(), routes=[utt.route for utt in strategy['remote']['upsert']], utterances=utterances_text, function_schemas=[utt.function_schemas for utt in strategy['remote']['upsert']], metadata_list=[utt.metadata for utt in strategy['remote']['upsert']], sparse_embeddings=sparse_emb)
        if strategy['local']['delete']:
            self._local_delete(utterances=strategy['local']['delete'])
        if strategy['local']['upsert']:
            self._local_upsert(utterances=strategy['local']['upsert'])
        self._write_hash()
        if isinstance(self.sparse_encoder, FittableMixin) and self.routes:
            self.sparse_encoder.fit(self.routes)

    def _get_index(self, index: Optional[BaseIndex]) -> BaseIndex:
        """Get the index.

        :param index: The index to get.
        :type index: Optional[BaseIndex]
        :return: The index.
        :rtype: BaseIndex
        """
        if index is None:
            logger.warning('No index provided. Using default HybridLocalIndex.')
            index = HybridLocalIndex()
        else:
            index = index
        return index

    def _get_sparse_encoder(self, sparse_encoder: Optional[SparseEncoder]) -> SparseEncoder:
        """Get the sparse encoder.

        :param sparse_encoder: The sparse encoder to get.
        :type sparse_encoder: Optional[SparseEncoder]
        :return: The sparse encoder.
        :rtype: Optional[SparseEncoder]
        """
        if sparse_encoder is None:
            logger.warning('No sparse_encoder provided. Using default BM25Encoder.')
            sparse_encoder = BM25Encoder()
        else:
            sparse_encoder = sparse_encoder
        return sparse_encoder

    def _encode(self, text: list[str], input_type: EncodeInputType) -> tuple[np.ndarray, list[SparseEmbedding]]:
        """Given some text, generates dense and sparse embeddings, then scales them
        using the chosen alpha value.

        :param text: List of texts to encode
        :type text: List[str]
        :param input_type: Specify whether encoding 'queries' or 'documents', used in asymmetric retrieval
        :type input_type: semantic_router.encoders.encode_input_type.EncodeInputType
        :return: Tuple of dense and sparse embeddings
        """
        if self.sparse_encoder is None:
            raise ValueError('self.sparse_encoder is not set.')
        if isinstance(self.encoder, AsymmetricDenseMixin):
            match input_type:
                case 'queries':
                    dense_v = self.encoder.encode_queries(text)
                case 'documents':
                    dense_v = self.encoder.encode_documents(text)
        else:
            dense_v = self.encoder(text)
        xq_d = np.array(dense_v)
        if isinstance(self.sparse_encoder, AsymmetricSparseMixin):
            match input_type:
                case 'queries':
                    xq_s = self.sparse_encoder.encode_queries(text)
                case 'documents':
                    xq_s = self.sparse_encoder.encode_documents(text)
        else:
            xq_s = self.sparse_encoder(text)
        xq_d, xq_s = self._convex_scaling(dense=xq_d, sparse=xq_s)
        return (xq_d, xq_s)

    async def _async_encode(self, text: List[str], input_type: EncodeInputType) -> tuple[np.ndarray, list[SparseEmbedding]]:
        """Given some text, generates dense and sparse embeddings, then scales them
        using the chosen alpha value.

        :param text: The text to encode.
        :type text: List[str]
        :param input_type: Specify whether encoding 'queries' or 'documents', used in asymmetric retrieval
        :type input_type: semantic_router.encoders.encode_input_type.EncodeInputType
        :return: A tuple of the dense and sparse embeddings.
        :rtype: tuple[np.ndarray, list[SparseEmbedding]]
        """
        if self.sparse_encoder is None:
            raise ValueError('self.sparse_encoder is not set.')
        if isinstance(self.encoder, AsymmetricDenseMixin):
            match input_type:
                case 'queries':
                    dense_coro = self.encoder.aencode_queries(text)
                case 'documents':
                    dense_coro = self.encoder.aencode_documents(text)
        else:
            dense_coro = self.encoder.acall(text)
        if isinstance(self.sparse_encoder, AsymmetricSparseMixin):
            match input_type:
                case 'queries':
                    sparse_coro = self.sparse_encoder.aencode_queries(text)
                case 'documents':
                    sparse_coro = self.sparse_encoder.aencode_documents(text)
        else:
            sparse_coro = self.sparse_encoder.acall(text)
        dense_vec, xq_s = await asyncio.gather(dense_coro, sparse_coro)
        xq_d = np.array(dense_vec)
        xq_d, xq_s = self._convex_scaling(dense=xq_d, sparse=xq_s)
        return (xq_d, xq_s)

    def __call__(self, text: Optional[str]=None, vector: Optional[List[float] | np.ndarray]=None, simulate_static: bool=False, route_filter: Optional[List[str]]=None, limit: int | None=1, sparse_vector: dict[int, float] | SparseEmbedding | None=None) -> RouteChoice | list[RouteChoice]:
        """Call the HybridRouter.

        :param text: The text to encode.
        :type text: Optional[str]
        :param vector: The vector to encode.
        :type vector: Optional[List[float] | np.ndarray]
        :param simulate_static: Whether to simulate a static route.
        :type simulate_static: bool
        :param route_filter: The route filter to use.
        :type route_filter: Optional[List[str]]
        :param limit: The number of routes to return, defaults to 1. If set to None, no
            limit is applied and all routes are returned.
        :type limit: int | None
        :param sparse_vector: The sparse vector to use.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A RouteChoice or a list of RouteChoices.
        :rtype: RouteChoice | list[RouteChoice]
        """
        if not self.index.is_ready():
            raise ValueError('Index is not ready.')
        if self.sparse_encoder is None:
            raise ValueError('Sparse encoder is not set.')
        potential_sparse_vector: List[SparseEmbedding] | None = None
        if vector is None:
            if text is None:
                raise ValueError('Either text or vector must be provided')
            xq_d = np.array(self.encoder([text]))
            xq_s = self.sparse_encoder([text])
            vector, potential_sparse_vector = self._convex_scaling(dense=xq_d, sparse=xq_s)
        vector = xq_reshape(vector)
        if sparse_vector is None:
            if text is None:
                raise ValueError('Either text or sparse_vector must be provided')
            sparse_vector = potential_sparse_vector[0] if potential_sparse_vector else None
        if sparse_vector is None:
            raise ValueError('Sparse vector is required for HybridLocalIndex.')
        scores, route_names = self.index.query(vector=vector[0], top_k=self.top_k, route_filter=route_filter, sparse_vector=sparse_vector)
        query_results = [{'route': d, 'score': s.item()} for d, s in zip(route_names, scores)]
        scored_routes = self._score_routes(query_results=query_results)
        route_choices = self._pass_routes(scored_routes=scored_routes, simulate_static=simulate_static, text=text, limit=limit)
        return route_choices

    async def acall(self, text: Optional[str]=None, vector: Optional[List[float] | np.ndarray]=None, limit: int | None=1, simulate_static: bool=False, route_filter: Optional[List[str]]=None, sparse_vector: dict[int, float] | SparseEmbedding | None=None) -> RouteChoice | list[RouteChoice]:
        """Asynchronously call the router to get a route choice.

        :param text: The text to route.
        :type text: Optional[str]
        :param vector: The vector to route.
        :type vector: Optional[List[float] | np.ndarray]
        :param simulate_static: Whether to simulate a static route (ie avoid dynamic route
            LLM calls during fit or evaluate).
        :type simulate_static: bool
        :param route_filter: The route filter to use.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: The sparse vector to use.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: The route choice.
        :rtype: RouteChoice
        """
        if not await self.index.ais_ready():
            await self._async_init_index_state()
        if self.sparse_encoder is None:
            raise ValueError('Sparse encoder is not set.')
        potential_sparse_vector: List[SparseEmbedding] | None = None
        if vector is None:
            if text is None:
                raise ValueError('Either text or vector must be provided')
            vector, potential_sparse_vector = await self._async_encode(text=[text], input_type='queries')
        vector = xq_reshape(xq=vector)
        if sparse_vector is None:
            if text is None:
                raise ValueError('Either text or sparse_vector must be provided')
            sparse_vector = potential_sparse_vector[0] if potential_sparse_vector else None
        scores, routes = await self.index.aquery(vector=vector[0], top_k=self.top_k, route_filter=route_filter, sparse_vector=sparse_vector)
        query_results = [{'route': d, 'score': s.item()} for d, s in zip(routes, scores)]
        scored_routes = self._score_routes(query_results=query_results)
        return await self._async_pass_routes(scored_routes=scored_routes, simulate_static=simulate_static, text=text, limit=limit)

    async def _async_execute_sync_strategy(self, strategy: Dict[str, Dict[str, List[Utterance]]]):
        """Executes the provided sync strategy, either deleting or upserting
        routes from the local and remote instances as defined in the strategy.

        :param strategy: The sync strategy to execute.
        :type strategy: Dict[str, Dict[str, List[Utterance]]]
        """
        if self.sparse_encoder is None:
            raise ValueError('Sparse encoder is not set.')
        if strategy['remote']['delete']:
            data_to_delete = {}
            for utt_obj in strategy['remote']['delete']:
                data_to_delete.setdefault(utt_obj.route, []).append(utt_obj.utterance)
            await self.index._async_remove_and_sync(data_to_delete)
        if strategy['remote']['upsert']:
            utterances_text = [utt.utterance for utt in strategy['remote']['upsert']]
            await self.index.aadd(embeddings=await self.encoder.acall(docs=utterances_text), sparse_embeddings=await self.sparse_encoder.acall(docs=utterances_text), routes=[utt.route for utt in strategy['remote']['upsert']], utterances=utterances_text, function_schemas=[utt.function_schemas for utt in strategy['remote']['upsert']], metadata_list=[utt.metadata for utt in strategy['remote']['upsert']])
        if strategy['local']['delete']:
            self._local_delete(utterances=strategy['local']['delete'])
        if strategy['local']['upsert']:
            self._local_upsert(utterances=strategy['local']['upsert'])
        await self._async_write_hash()

    def _convex_scaling(self, dense: np.ndarray, sparse: list[SparseEmbedding]) -> tuple[np.ndarray, list[SparseEmbedding]]:
        """Convex scaling of the dense and sparse vectors.

        :param dense: The dense vector to scale.
        :type dense: np.ndarray
        :param sparse: The sparse vector to scale.
        :type sparse: list[SparseEmbedding]
        """
        sparse_dicts = [sparse_vec.to_dict() for sparse_vec in sparse]
        scaled_dense = np.array(dense) * self.alpha
        scaled_sparse = []
        for sparse_dict in sparse_dicts:
            scaled_sparse.append(SparseEmbedding.from_dict({k: v * (1 - self.alpha) for k, v in sparse_dict.items()}))
        return (scaled_dense, scaled_sparse)

    def fit(self, X: List[str], y: List[str], batch_size: int=500, max_iter: int=500, local_execution: bool=False):
        """Fit the HybridRouter.

        :param X: The input data.
        :type X: List[str]
        :param y: The output data.
        :type y: List[str]
        :param batch_size: The batch size to use for fitting.
        :type batch_size: int
        :param max_iter: The maximum number of iterations to use for fitting.
        :type max_iter: int
        :param local_execution: Whether to execute the fitting locally.
        :type local_execution: bool
        """
        original_index = self.index
        if self.sparse_encoder is None:
            raise ValueError('Sparse encoder is not set.')
        if local_execution:
            from semantic_router.index.hybrid_local import HybridLocalIndex
            remote_utterances = self.index.get_utterances(include_metadata=True)
            routes = []
            utterances = []
            metadata = []
            for utterance in remote_utterances:
                routes.append(utterance.route)
                utterances.append(utterance.utterance)
                metadata.append(utterance.metadata)
            embeddings = self.encoder(utterances) if not isinstance(self.encoder, AsymmetricDenseMixin) else self.encoder.encode_documents(utterances)
            sparse_embeddings = self.sparse_encoder(utterances) if not isinstance(self.sparse_encoder, AsymmetricSparseMixin) else self.sparse_encoder.encode_documents(utterances)
            self.index = HybridLocalIndex()
            self.index.add(embeddings=embeddings, sparse_embeddings=sparse_embeddings, routes=routes, utterances=utterances, metadata_list=metadata)
        Xq_d: List[List[float]] = []
        Xq_s: List[SparseEmbedding] = []
        for i in tqdm(range(0, len(X), batch_size), desc='Generating embeddings'):
            emb_d = np.array(self.encoder(X[i:i + batch_size]) if not isinstance(self.encoder, AsymmetricDenseMixin) else self.encoder.encode_queries(X[i:i + batch_size]))
            emb_s = self.sparse_encoder(X[i:i + batch_size]) if not isinstance(self.sparse_encoder, AsymmetricSparseMixin) else self.sparse_encoder.encode_queries(X[i:i + batch_size])
            Xq_d.extend(emb_d)
            Xq_s.extend(emb_s)
        best_acc = self._vec_evaluate(Xq_d=np.array(Xq_d), Xq_s=Xq_s, y=y)
        best_thresholds = self.get_thresholds()
        for _ in (pbar := tqdm(range(max_iter), desc='Training')):
            pbar.set_postfix({'acc': round(best_acc, 2)})
            thresholds = threshold_random_search(route_layer=self, search_range=0.8)
            self._update_thresholds(route_thresholds=thresholds)
            acc = self._vec_evaluate(Xq_d=np.array(Xq_d), Xq_s=Xq_s, y=y)
            if acc > best_acc:
                best_acc = acc
                best_thresholds = thresholds
        self._update_thresholds(route_thresholds=best_thresholds)
        if local_execution:
            self.index = original_index

    def evaluate(self, X: List[str], y: List[str], batch_size: int=500) -> float:
        """Evaluate the accuracy of the route selection.

        :param X: The input data.
        :type X: List[str]
        :param y: The output data.
        :type y: List[str]
        :param batch_size: The batch size to use for evaluation.
        :type batch_size: int
        :return: The accuracy of the route selection.
        :rtype: float
        """
        if self.sparse_encoder is None:
            raise ValueError('Sparse encoder is not set.')
        Xq_d: List[List[float]] = []
        Xq_s: List[SparseEmbedding] = []
        for i in tqdm(range(0, len(X), batch_size), desc='Generating embeddings'):
            emb_d = np.array(self.encoder(X[i:i + batch_size]) if not isinstance(self.encoder, AsymmetricDenseMixin) else self.encoder.encode_queries(X[i:i + batch_size]))
            emb_s = self.sparse_encoder(X[i:i + batch_size]) if not isinstance(self.sparse_encoder, AsymmetricSparseMixin) else self.sparse_encoder.encode_queries(X[i:i + batch_size])
            Xq_d.extend(emb_d)
            Xq_s.extend(emb_s)
        accuracy = self._vec_evaluate(Xq_d=np.array(Xq_d), Xq_s=Xq_s, y=y)
        return accuracy

    def _vec_evaluate(self, Xq_d: Union[List[float], Any], Xq_s: list[SparseEmbedding], y: List[str]) -> float:
        """Evaluate the accuracy of the route selection.

        :param Xq_d: The dense vectors to evaluate.
        :type Xq_d: Union[List[float], Any]
        :param Xq_s: The sparse vectors to evaluate.
        :type Xq_s: list[SparseEmbedding]
        :param y: The output data.
        :type y: List[str]
        :return: The accuracy of the route selection.
        :rtype: float
        """
        correct = 0
        for xq_d, xq_s, target_route in zip(Xq_d, Xq_s, y):
            route_choice = self(vector=xq_d, sparse_vector=xq_s, simulate_static=True)
            if isinstance(route_choice, list):
                route_name = route_choice[0].name
            else:
                route_name = route_choice.name
            if route_name == target_route:
                correct += 1
        accuracy = correct / len(Xq_d)
        return accuracy

def threshold_random_search(route_layer: BaseRouter, search_range: Union[int, float]) -> Dict[str, float]:
    """Performs a random search iteration given a route layer and a search range.

    :param route_layer: The route layer to search.
    :type route_layer: BaseRouter
    :param search_range: The search range to use.
    :type search_range: Union[int, float]
    :return: A dictionary of route names and their associated thresholds.
    :rtype: Dict[str, float]
    """
    routes = route_layer.get_thresholds()
    route_names = list(routes.keys())
    route_thresholds = list(routes.values())
    score_threshold_values = []
    for threshold in route_thresholds:
        score_threshold_values.append(np.linspace(start=max(threshold - search_range, 0.0), stop=min(threshold + search_range, 1.0), num=100))
    score_thresholds = {route: random.choice(score_threshold_values[i]) for i, route in enumerate(route_names)}
    return score_thresholds

@pytest.fixture
def encoder(requests_mock):
    requests_mock.post('https://api-inference.huggingface.co/models/bert-base-uncased', json=[0.1, 0.2, 0.3], status_code=200)
    return HFEndpointEncoder(huggingface_url='https://api-inference.huggingface.co/models/bert-base-uncased', huggingface_api_key='test-api-key', score_threshold=0.8)

class TestHFEndpointEncoder:

    def test_initialization(self, encoder):
        assert encoder.huggingface_url == 'https://api-inference.huggingface.co/models/bert-base-uncased'
        assert encoder.huggingface_api_key == 'test-api-key'
        assert encoder.score_threshold == 0.8

    def test_initialization_failure_no_api_key(self):
        with pytest.raises(ValueError) as exc_info:
            HFEndpointEncoder(huggingface_url='https://api-inference.huggingface.co/models/bert-base-uncased')
        assert "HuggingFace API key cannot be 'None'" in str(exc_info.value)

    def test_initialization_failure_no_url(self):
        with pytest.raises(ValueError) as exc_info:
            HFEndpointEncoder(huggingface_api_key='test-api-key')
        assert "HuggingFace endpoint url cannot be 'None'" in str(exc_info.value)

    def test_query_success(self, encoder, requests_mock):
        requests_mock.post('https://api-inference.huggingface.co/models/bert-base-uncased', json=[0.1, 0.2, 0.3], status_code=200)
        response = encoder.query({'inputs': 'Hello World!', 'parameters': {}})
        assert response == [0.1, 0.2, 0.3]

    def test_query_failure(self, encoder, requests_mock):
        requests_mock.post('https://api-inference.huggingface.co/models/bert-base-uncased', text='Error', status_code=400)
        with pytest.raises(ValueError) as exc_info:
            encoder.query({'inputs': 'Hello World!', 'parameters': {}})
        assert 'Query failed with status 400: Error' in str(exc_info.value)

    def test_encode_documents_success(self, encoder, requests_mock):
        requests_mock.post('https://api-inference.huggingface.co/models/bert-base-uncased', json=[0.1, 0.2, 0.3], status_code=200)
        embeddings = encoder(['Hello World!'])
        assert embeddings == [[0.1, 0.2, 0.3]]

@pytest.mark.parametrize('index_cls,encoder_cls,router_cls', [(index, encoder, router) for index in get_test_indexes() for encoder in [OpenAIEncoder] for router in get_test_routers()])
class TestSemanticRouter:

    def test_initialization_dynamic_route(self, dynamic_routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=dynamic_routes, index=index, auto_sync='local')
        score_threshold = route_layer.score_threshold
        if isinstance(route_layer, HybridRouter):
            assert score_threshold == encoder.score_threshold * route_layer.alpha
        else:
            assert score_threshold == encoder.score_threshold

    def test_add_single_utterance(self, routes, route_single_utterance, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')
        route_layer.add(routes=route_single_utterance)
        score_threshold = route_layer.score_threshold
        if isinstance(route_layer, HybridRouter):
            assert score_threshold == encoder.score_threshold * route_layer.alpha
        else:
            assert score_threshold == encoder.score_threshold

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_index_populated():
            _ = route_layer('Hello')
            assert len(route_layer.index.get_utterances()) == 6
        check_index_populated()

    def test_init_and_add_single_utterance(self, route_single_utterance, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, index=index, auto_sync='local')
        if index_cls is PineconeIndex:
            time.sleep(PINECONE_SLEEP)
        route_layer.add(routes=route_single_utterance)
        score_threshold = route_layer.score_threshold
        if isinstance(route_layer, HybridRouter):
            assert score_threshold == encoder.score_threshold * route_layer.alpha
        else:
            assert score_threshold == encoder.score_threshold

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_index_populated():
            _ = route_layer('Hello')
            assert len(route_layer.index.get_utterances()) == 1
        check_index_populated()

    def test_delete_index(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def delete_index():
            route_layer.index.delete_index()
            assert route_layer.index.get_utterances() == []
        delete_index()

    def test_add_route(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=[], index=index, auto_sync='local')
        assert route_layer.routes == []

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_index_empty():
            assert route_layer.index.get_utterances() == []
        check_index_empty()
        route_layer.add(routes=routes[0])

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_index_populated1():
            assert route_layer.routes == [routes[0]]
            assert route_layer.index is not None
            assert len(route_layer.index.get_utterances()) == 2
        check_index_populated1()
        route_layer.add(routes=routes[1])

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_index_populated2():
            assert route_layer.routes == [routes[0], routes[1]]
            assert len(route_layer.index.get_utterances()) == 5
        check_index_populated2()

    def test_list_route_names(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_route_names():
            route_names = route_layer.list_route_names()
            assert set(route_names) == {route.name for route in routes}, 'The list of route names should match the names of the routes added.'
        check_route_names()

    def test_delete_route(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def delete_route_by_name():
            route_to_delete = routes[0].name
            route_layer.delete(route_to_delete)
            assert route_to_delete not in route_layer.list_route_names(), 'The route should be deleted from the route layer.'
            for utterance in routes[0].utterances:
                assert utterance not in route_layer.index, "The route's utterances should be deleted from the index."
        delete_route_by_name()

    def test_remove_route_not_found(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def delete_non_existent_route():
            non_existent_route = 'non-existent-route'
            route_layer.delete(non_existent_route)
        delete_non_existent_route()

    def test_add_multiple_routes(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, index=index, auto_sync='local')

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_index_populated():
            route_layer.add(routes=routes)
            assert route_layer.index is not None
            assert len(route_layer.index.get_utterances()) == 5
        check_index_populated()

    def test_query_and_classification(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        encoder.score_threshold = 0.1
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local', aggregation='max')

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_query_result():
            query_result = route_layer(text='Hello').name
            assert query_result in ['Route 1', 'Route 2']
        check_query_result()

    def test_query_filter(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        encoder.score_threshold = 0.1
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local', aggregation='max')

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_raises_value_error():
            try:
                route_layer(text='Hello', route_filter=['Route 8']).name
            except ValueError:
                assert True
        check_raises_value_error()

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_query_result():
            query_result = route_layer(text='Hello', route_filter=['Route 1']).name
            assert query_result in ['Route 1']
        check_query_result()

    @pytest.mark.skipif(os.environ.get('PINECONE_API_KEY') is None, reason='Pinecone API key required')
    def test_namespace_pinecone_index(self, routes, index_cls, encoder_cls, router_cls):
        if index_cls is PineconeIndex:
            encoder = encoder_cls()
            encoder.score_threshold = 0.2
            index = init_index(index_cls, namespace='test', index_name=encoder.__class__.__name__)
            route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')
            time.sleep(PINECONE_SLEEP)

            @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
            def check_query_result():
                query_result = route_layer(text='Hello', route_filter=['Route 1']).name
                assert query_result in ['Route 1']
            check_query_result()

            @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
            def delete_namespace():
                route_layer.index.index.delete(namespace='test', delete_all=True)
            delete_namespace()

    def test_query_with_no_index(self, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        route_layer = router_cls(encoder=encoder)
        with pytest.raises(ValueError):
            assert route_layer(text='Anything').name is None

    def test_query_with_vector(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        encoder.score_threshold = 0.1
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local', aggregation='max')
        vector = encoder(['hello'])
        if router_cls is HybridRouter:
            sparse_vector = route_layer.sparse_encoder(['hello'])[0]

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_query_result():
            if router_cls is HybridRouter:
                query_result = route_layer(vector=vector, sparse_vector=sparse_vector).name
            else:
                query_result = route_layer(vector=vector).name
            assert query_result in ['Route 1', 'Route 2']
        check_query_result()

    def test_query_with_no_text_or_vector(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        with pytest.raises(ValueError):
            route_layer()

    def test_is_ready(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_is_ready():
            assert route_layer.index.is_ready()
        check_is_ready()

@pytest.mark.parametrize('index_cls,encoder_cls,router_cls', [(index, encoder, router) for index in [LocalIndex] for encoder in [OpenAIEncoder] for router in get_test_routers()])
class TestRouterOnly:

    def test_semantic_classify(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')
        classification, score = route_layer._semantic_classify([{'route': 'Route 1', 'score': 0.9}, {'route': 'Route 2', 'score': 0.1}])
        assert classification == 'Route 1'
        assert score == [0.9]

    def test_semantic_classify_multiple_routes(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')
        classification, score = route_layer._semantic_classify([{'route': 'Route 1', 'score': 0.9}, {'route': 'Route 2', 'score': 0.1}, {'route': 'Route 1', 'score': 0.8}])
        assert classification == 'Route 1'
        assert score == [0.9, 0.8]

    def test_query_no_text_dynamic_route(self, dynamic_routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=dynamic_routes, index=index)
        vector = encoder(['hello'])
        if router_cls is HybridRouter:
            sparse_vector = route_layer.sparse_encoder(['hello'])[0]
        with pytest.raises(ValueError):
            if router_cls is HybridRouter:
                route_layer(vector=vector, sparse_vector=sparse_vector)
            else:
                route_layer(vector=vector)

    def test_failover_score_threshold(self, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, index=index, auto_sync='local')
        if router_cls is HybridRouter:
            assert route_layer.score_threshold == 0.3 * route_layer.alpha
        else:
            assert route_layer.score_threshold == 0.3

    def test_json(self, routes, index_cls, encoder_cls, router_cls):
        temp = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
        try:
            temp_path = temp.name
            temp.close()
            encoder = encoder_cls()
            index = init_index(index_cls, index_name=encoder.__class__.__name__)
            route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')
            route_layer.to_json(temp_path)
            assert os.path.exists(temp_path)
            route_layer_from_file = SemanticRouter.from_json(temp_path)
            assert route_layer_from_file.index is not None and route_layer_from_file._get_route_names() is not None
        finally:
            os.remove(temp_path)

    def test_yaml(self, routes, index_cls, encoder_cls, router_cls):
        temp = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
        try:
            temp_path = temp.name
            temp.close()
            encoder = encoder_cls()
            index = init_index(index_cls, index_name=encoder.__class__.__name__)
            route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')
            route_layer.to_yaml(temp_path)
            assert os.path.exists(temp_path)
            route_layer_from_file = SemanticRouter.from_yaml(temp_path)
            assert route_layer_from_file.index is not None and route_layer_from_file._get_route_names() is not None
        finally:
            os.remove(temp_path)

    def test_config(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        layer_config = route_layer.to_config()
        assert layer_config.routes == route_layer.routes
        route_layer_from_config = SemanticRouter.from_config(layer_config, index)
        assert route_layer_from_config._get_route_names() == route_layer._get_route_names()
        if router_cls is HybridRouter:
            pass
        else:
            assert route_layer_from_config.score_threshold == route_layer.score_threshold

    def test_get_thresholds(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        if router_cls is HybridRouter:
            target = encoder.score_threshold * route_layer.alpha
            assert route_layer.get_thresholds() == {'Route 1': target, 'Route 2': target}
        else:
            assert route_layer.get_thresholds() == {'Route 1': 0.3, 'Route 2': 0.3}

    def test_with_multiple_routes_passing_threshold(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')
        route_layer.set_threshold(threshold=0.0)
        results = route_layer(text='Hello', limit=2)
        assert len(results) == 2
        assert results[0].name == 'Route 1', f'Expected Route 1 in position 0, got {results}'
        assert results[1].name == 'Route 2', f'Expected Route 2 in position 1, got {results}'

    def test_with_no_routes_passing_threshold(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')
        route_layer.set_threshold(threshold=1.0)
        results = route_layer(text='Hello', limit=None)
        assert results == RouteChoice()

    def test_with_no_query_results(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')
        route_layer.set_threshold(threshold=0.5)
        results = route_layer(text='this should not be similar to anything', limit=None)
        assert results == RouteChoice()

    def test_with_unrecognized_route(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')
        route_layer.set_threshold(threshold=0.5)
        query_results = [{'route': 'UnrecognizedRoute', 'score': 0.9}]
        results = route_layer._semantic_classify(query_results)
        assert results == ('UnrecognizedRoute', [0.9]), 'Semantic classify can return unrecognized routes'

    def test_set_aggregation_method_with_unsupported_value(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        unsupported_aggregation = 'unsupported_aggregation_method'
        with pytest.raises(ValueError, match=f"Unsupported aggregation method chosen: {unsupported_aggregation}. Choose either 'SUM', 'MEAN', or 'MAX'."):
            route_layer._set_aggregation_method(unsupported_aggregation)

    def test_refresh_routes_not_implemented(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        with pytest.raises(NotImplementedError, match='This method has not yet been implemented.'):
            route_layer._refresh_routes()

    def test_update_threshold(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        route_name = 'Route 1'
        new_threshold = 0.8
        route_layer.update(name=route_name, threshold=new_threshold)
        updated_route = route_layer.get(route_name)
        assert updated_route.score_threshold == new_threshold, f'Expected threshold to be updated to {new_threshold}, but got {updated_route.score_threshold}'

    def test_update_non_existent_route(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        non_existent_route = 'Non-existent Route'
        with pytest.raises(ValueError, match=f"Route '{non_existent_route}' not found. Nothing updated."):
            route_layer.update(name=non_existent_route, threshold=0.7)

    def test_update_without_parameters(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        with pytest.raises(ValueError, match="At least one of 'threshold' or 'utterances' must be provided."):
            route_layer.update(name='Route 1')

    def test_update_utterances_not_implemented(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        with pytest.raises(NotImplementedError, match='The update method cannot be used for updating utterances yet.'):
            route_layer.update(name='Route 1', utterances=['New utterance'])

