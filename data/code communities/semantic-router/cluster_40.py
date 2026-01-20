# Cluster 40

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

def retry(max_retries: int=5, delay: int=8):
    """Retry decorator, currently used for PineconeIndex which often needs some time
    to be populated and have all correct data. Once full Pinecone mock is built we
    should remove this decorator.

    :param max_retries: Maximum number of retries.
    :param delay: Delay between retries in seconds.
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            count = 0
            last_exception = None
            while count < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f'Attempt {count} | Error in {func.__name__}: {e}')
                    last_exception = e
                    count += 1
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

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

