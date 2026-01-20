# Cluster 29

def init_index(index_cls, dimensions: int=3, namespace: str='', index_name: str | None=None, init_async_index: bool=False):
    """Initialize indexes for unit testing."""
    if index_cls is QdrantIndex:
        index_name = index_name or f'test_{uuid.uuid4().hex}'
        return QdrantIndex(index_name=index_name, init_async_index=init_async_index)
    if index_cls is PineconeIndex:
        index_name = f'test-{datetime.now().strftime('%Y%m%d%H%M%S')}' if not index_name else index_name
        index = index_cls(index_name=index_name, dimensions=dimensions, namespace=namespace, init_async_index=init_async_index, base_url=PINECONE_BASE_URL)
    elif index_cls is PostgresIndex:
        index = index_cls(index_name=index_name or 'test_index', index_prefix='', namespace=namespace, dimensions=dimensions, init_async_index=init_async_index)
    elif index_cls is None:
        return None
    else:
        index = index_cls(init_async_index=init_async_index)
    return index

@pytest.mark.parametrize('router_cls,index_cls', [(router, index) for router in [HybridRouter, SemanticRouter] for index in [None] + get_test_indexes()])
class TestRouter:

    def test_query_parameter(self, router_cls, index_cls, routes_5, mocker):
        """Test that we return expected values in RouteChoice objects."""
        dense_encoder = MockSymmetricDenseEncoder(name='Dense Encoder')
        index = init_index(index_cls) if index_cls else None
        if router_cls == HybridRouter:
            sparse_encoder = MockSymmetricSparseEncoder(name='Sparse Encoder')
            router = router_cls(encoder=dense_encoder, sparse_encoder=sparse_encoder, routes=routes_5, index=index, auto_sync='local')
        else:
            router = router_cls(encoder=dense_encoder, routes=routes_5, index=index, auto_sync='local')
        _ = mocker.patch.object(router, '_score_routes', return_value=[('Route 1', 0.9, [0.1, 0.2, 0.3]), ('Route 2', 0.8, [0.4, 0.5, 0.6]), ('Route 3', 0.7, [0.7, 0.8, 0.9]), ('Route 4', 0.6, [1.0, 1.1, 1.2])])
        result = router('test query')
        assert result is not None
        assert isinstance(result, RouteChoice)
        assert result.name == 'Route 1'
        assert result.similarity_score == 0.9
        assert result.function_call is None

    def test_limit_parameter(self, router_cls, index_cls, routes_5, mocker):
        """Test that the limit parameter works correctly for sync router calls."""
        dense_encoder = MockSymmetricDenseEncoder(name='Dense Encoder')
        index = init_index(index_cls) if index_cls else None
        if router_cls == HybridRouter:
            sparse_encoder = MockSymmetricSparseEncoder(name='Sparse Encoder')
            router = router_cls(encoder=dense_encoder, sparse_encoder=sparse_encoder, routes=routes_5, index=index, auto_sync='local')
        else:
            router = router_cls(encoder=dense_encoder, routes=routes_5, index=index, auto_sync='local')
        _ = mocker.patch.object(router, '_score_routes', return_value=[('Route 1', 0.9, [0.1, 0.2, 0.3]), ('Route 2', 0.8, [0.4, 0.5, 0.6]), ('Route 3', 0.7, [0.7, 0.8, 0.9]), ('Route 4', 0.6, [1.0, 1.1, 1.2])])
        result = router('test query')
        assert result is not None
        assert isinstance(result, RouteChoice)
        result = router('test query', limit=2)
        assert result is not None
        assert len(result) == 2
        result = router('test query', limit=None)
        assert result is not None
        assert len(result) == 4

    def test_index_operations(self, router_cls, index_cls, routes, openai_encoder):
        """Test index-specific operations like add, delete, and sync."""
        if index_cls is None:
            pytest.skip('Test only for specific index implementations')
        index = init_index(index_cls)
        if router_cls == HybridRouter:
            sparse_encoder = MockSymmetricSparseEncoder(name='Sparse Encoder')
            router = router_cls(encoder=openai_encoder, sparse_encoder=sparse_encoder, routes=[], index=index, auto_sync='local')
        else:
            router = router_cls(encoder=openai_encoder, routes=[], index=index, auto_sync='local')
        assert len(router.index) == 0
        router.add(routes[0])
        assert len(router.index) == 2
        router.add(routes[1])
        assert len(router.index) == 5
        router.delete('Route 1')
        assert len(router.index) == 3
        router.index.delete_index()
        assert len(router.index) == 0

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

@pytest.mark.parametrize('index_cls,encoder_cls,router_cls', [(index, encoder, router) for index in get_test_indexes() for encoder in get_test_encoders() for router in get_test_routers()])
class TestIndexEncoders:

    def test_initialization(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local', top_k=10)
        score_threshold = route_layer.score_threshold
        if isinstance(route_layer, HybridRouter):
            assert score_threshold == encoder.score_threshold * route_layer.alpha
        else:
            assert score_threshold == encoder.score_threshold
        assert route_layer.top_k == 10

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_index_populated():
            assert len(route_layer.index) == 5
        check_index_populated()
        assert len(set(route_layer._get_route_names())) if route_layer._get_route_names() is not None else 0 == 2

    def test_initialization_different_encoders(self, encoder_cls, index_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, index=index)
        score_threshold = route_layer.score_threshold
        if isinstance(route_layer, HybridRouter):
            assert score_threshold == encoder.score_threshold * route_layer.alpha
        else:
            assert score_threshold == encoder.score_threshold

    def test_initialization_no_encoder(self, index_cls, encoder_cls, router_cls):
        route_layer_none = router_cls(encoder=None)
        score_threshold = route_layer_none.score_threshold
        if isinstance(route_layer_none, HybridRouter):
            assert score_threshold == 0.3 * route_layer_none.alpha
        else:
            assert score_threshold == 0.3

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

@pytest.mark.parametrize('index_cls,encoder_cls,router_cls', [(index, encoder, router) for index in get_test_indexes() for encoder in [OpenAIEncoder] for router in get_test_routers()])
class TestLayerFit:

    def test_eval(self, routes, test_data, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_is_ready():
            assert route_layer.index.is_ready()
        check_is_ready()
        X, y = zip(*test_data)
        route_layer.evaluate(X=list(X), y=list(y), batch_size=int(len(X) / 5))

    def test_fit(self, routes, test_data, index_cls, encoder_cls, router_cls):
        if index_cls is PineconeIndex:
            return
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_is_ready():
            assert route_layer.index.is_ready()
        check_is_ready()
        X, y = zip(*test_data)
        route_layer.fit(X=list(X), y=list(y), batch_size=int(len(X) / 5))

    def test_fit_local(self, routes, test_data, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index, auto_sync='local')

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_is_ready():
            assert route_layer.index.is_ready()
        check_is_ready()
        X, y = zip(*test_data)
        route_layer.fit(X=list(X), y=list(y), batch_size=int(len(X) / 5), local_execution=True)

