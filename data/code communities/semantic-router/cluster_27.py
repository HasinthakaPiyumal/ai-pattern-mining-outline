# Cluster 27

class TestRouterConfig:

    def test_from_file_json(self, tmp_path):
        config_path = tmp_path / 'config.json'
        config_path.write_text(layer_json())
        layer_config = RouterConfig.from_file(str(config_path))
        assert layer_config.encoder_type == 'cohere'
        assert layer_config.encoder_name == 'embed-english-v3.0'
        assert len(layer_config.routes) == 2
        assert layer_config.routes[0].name == 'politics'

    def test_from_file_yaml(self, tmp_path):
        config_path = tmp_path / 'config.yaml'
        config_path.write_text(layer_yaml())
        layer_config = RouterConfig.from_file(str(config_path))
        assert layer_config.encoder_type == 'cohere'
        assert layer_config.encoder_name == 'embed-english-v3.0'
        assert len(layer_config.routes) == 2
        assert layer_config.routes[0].name == 'politics'

    def test_from_file_invalid_path(self):
        with pytest.raises(FileNotFoundError) as excinfo:
            RouterConfig.from_file('nonexistent_path.json')
        assert "[Errno 2] No such file or directory: 'nonexistent_path.json'" in str(excinfo.value)

    def test_from_file_unsupported_type(self, tmp_path):
        config_path = tmp_path / 'config.unsupported'
        config_path.write_text(layer_json())
        with pytest.raises(ValueError) as excinfo:
            RouterConfig.from_file(str(config_path))
        assert 'Unsupported file type' in str(excinfo.value)

    def test_from_file_invalid_config(self, tmp_path):
        invalid_config_json = '\n        {\n            "encoder_type": "cohere",\n            "encoder_name": "embed-english-v3.0",\n            "routes": "This should be a list, not a string"\n        }'
        config_path = tmp_path / 'invalid_config.json'
        with open(config_path, 'w') as file:
            file.write(invalid_config_json)
        with patch('semantic_router.routers.base.is_valid', return_value=False):
            with pytest.raises(Exception) as excinfo:
                RouterConfig.from_file(str(config_path))
            assert 'Invalid config JSON or YAML' in str(excinfo.value), 'Loading an invalid configuration should raise an exception.'

    def test_from_file_with_llm(self, tmp_path):
        llm_config_json = '\n        {\n            "encoder_type": "cohere",\n            "encoder_name": "embed-english-v3.0",\n            "routes": [\n                {\n                    "name": "llm_route",\n                    "utterances": ["tell me a joke", "say something funny"],\n                    "llm": {\n                        "module": "semantic_router.llms.base",\n                        "class": "BaseLLM",\n                        "model": "fake-model-v1"\n                    }\n                }\n            ]\n        }'
        config_path = tmp_path / 'config_with_llm.json'
        with open(config_path, 'w') as file:
            file.write(llm_config_json)
        layer_config = RouterConfig.from_file(str(config_path))
        assert isinstance(layer_config.routes[0].llm, BaseLLM), 'LLM should be instantiated and associated with the route based on the '
        'config'
        assert layer_config.routes[0].llm.name == 'fake-model-v1', "LLM instance should have the 'name' attribute set correctly"

    def test_init(self):
        layer_config = RouterConfig()
        assert layer_config.routes == []

    def test_to_file_json(self):
        route = Route(name='test', utterances=['utterance'])
        layer_config = RouterConfig(routes=[route])
        with patch('builtins.open', mock_open()) as mocked_open:
            layer_config.to_file('data/test_output.json')
            mocked_open.assert_called_once_with('data/test_output.json', 'w')

    def test_to_file_yaml(self):
        route = Route(name='test', utterances=['utterance'])
        layer_config = RouterConfig(routes=[route])
        with patch('builtins.open', mock_open()) as mocked_open:
            layer_config.to_file('data/test_output.yaml')
            mocked_open.assert_called_once_with('data/test_output.yaml', 'w')

    def test_to_file_invalid(self):
        route = Route(name='test', utterances=['utterance'])
        layer_config = RouterConfig(routes=[route])
        with pytest.raises(ValueError):
            layer_config.to_file('test_output.txt')

    def test_from_file_invalid(self):
        with open('test.txt', 'w') as f:
            f.write('dummy content')
        with pytest.raises(ValueError):
            RouterConfig.from_file('test.txt')
        os.remove('test.txt')

    def test_to_dict(self):
        route = Route(name='test', utterances=['utterance'])
        layer_config = RouterConfig(routes=[route])
        assert layer_config.to_dict()['routes'] == [route.to_dict()]

    def test_add(self):
        route = Route(name='test', utterances=['utterance'])
        route2 = Route(name='test2', utterances=['utterance2'])
        layer_config = RouterConfig()
        layer_config.add(route)
        assert layer_config.routes == [route]
        layer_config.add(route2)
        assert layer_config.routes == [route, route2]

    def test_get(self):
        route = Route(name='test', utterances=['utterance'])
        layer_config = RouterConfig(routes=[route])
        assert layer_config.get('test') == route

    def test_get_not_found(self):
        route = Route(name='test', utterances=['utterance'])
        layer_config = RouterConfig(routes=[route])
        assert layer_config.get('not_found') is None

    def test_remove(self):
        route = Route(name='test', utterances=['utterance'])
        layer_config = RouterConfig(routes=[route])
        layer_config.remove('test')
        assert layer_config.routes == []

    def test_setting_aggregation_methods(self, openai_encoder, routes):
        for agg in ['sum', 'mean', 'max']:
            route_layer = SemanticRouter(encoder=openai_encoder, routes=routes, aggregation=agg)
            assert route_layer.aggregation == agg

    def test_semantic_classify_multiple_routes_with_different_aggregation(self, openai_encoder, routes):
        route_scores = [{'route': 'Route 1', 'score': 0.5}, {'route': 'Route 1', 'score': 0.5}, {'route': 'Route 1', 'score': 0.5}, {'route': 'Route 1', 'score': 0.5}, {'route': 'Route 2', 'score': 0.4}, {'route': 'Route 2', 'score': 0.6}, {'route': 'Route 2', 'score': 0.8}, {'route': 'Route 3', 'score': 0.1}, {'route': 'Route 3', 'score': 1.0}]
        for agg in ['sum', 'mean', 'max']:
            route_layer = SemanticRouter(encoder=openai_encoder, routes=routes, aggregation=agg)
            classification, score = route_layer._semantic_classify(route_scores)
            if agg == 'sum':
                assert classification == 'Route 1'
                assert score == [0.5, 0.5, 0.5, 0.5]
            elif agg == 'mean':
                assert classification == 'Route 2'
                assert score == [0.4, 0.6, 0.8]
            elif agg == 'max':
                assert classification == 'Route 3'
                assert score == [0.1, 1.0]

def layer_json():
    return '{\n    "encoder_type": "cohere",\n    "encoder_name": "embed-english-v3.0",\n    "routes": [\n        {\n            "name": "politics",\n            "utterances": [\n                "isn\'t politics the best thing ever",\n                "why don\'t you tell me about your political opinions"\n            ],\n            "description": null,\n            "function_schemas": null\n        },\n        {\n            "name": "chitchat",\n            "utterances": [\n                "how\'s the weather today?",\n                "how are things going?"\n            ],\n            "description": null,\n            "function_schemas": null\n        }\n    ]\n}'

