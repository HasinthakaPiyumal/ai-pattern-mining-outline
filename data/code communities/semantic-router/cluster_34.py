# Cluster 34

class TestRoute:

    def test_value_error_in_route_call(self):
        function_schemas = [{'name': 'test_function', 'type': 'function'}]
        route = Route(name='test_function', utterances=['utterance1', 'utterance2'], function_schemas=function_schemas)
        with pytest.raises(ValueError):
            route('test_query')

    def test_generate_dynamic_route(self):
        mock_llm = MockLLM(name='test')
        function_schemas = {'name': 'test_function', 'type': 'function'}
        route = Route._generate_dynamic_route(llm=mock_llm, function_schemas=function_schemas, route_name='test_route')
        assert route.name == 'test_function'
        assert route.utterances == ['example_utterance_1', 'example_utterance_2', 'example_utterance_3', 'example_utterance_4', 'example_utterance_5']

    def test_to_dict(self):
        route = Route(name='test', utterances=['utterance'])
        expected_dict = {'name': 'test', 'utterances': ['utterance'], 'description': None, 'function_schemas': None, 'llm': None, 'score_threshold': None, 'metadata': {}}
        assert route.to_dict() == expected_dict

    def test_from_dict(self):
        route_dict = {'name': 'test', 'utterances': ['utterance']}
        route = Route.from_dict(route_dict)
        assert route.name == 'test'
        assert route.utterances == ['utterance']

    def test_from_dynamic_route(self):
        mock_llm = MockLLM(name='test')

        def test_function(input: str):
            """Test function docstring"""
            pass
        dynamic_route = Route.from_dynamic_route(llm=mock_llm, entities=[test_function], route_name='test_route')
        assert dynamic_route.name == 'test_function'
        assert dynamic_route.utterances == ['example_utterance_1', 'example_utterance_2', 'example_utterance_3', 'example_utterance_4', 'example_utterance_5']

    def test_parse_route_config(self):
        config = '\n        <config>\n        {\n            "name": "test_function",\n            "utterances": [\n                "example_utterance_1",\n                "example_utterance_2",\n                "example_utterance_3",\n                "example_utterance_4",\n                "example_utterance_5"]\n        }\n        </config>\n        '
        expected_config = '\n        {\n            "name": "test_function",\n            "utterances": [\n                "example_utterance_1",\n                "example_utterance_2",\n                "example_utterance_3",\n                "example_utterance_4",\n                "example_utterance_5"]\n        }\n        '
        assert Route._parse_route_config(config).strip() == expected_config.strip()

