# Cluster 31

def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    model_name = 'en_core_web_lg'
    download_model(model_name)
    analyzer = get_analyzer_engine()
    analyzer_results = analyzer.analyze(text=initial_query, language='en')
    anonymizer_engine = get_anonymizer_engine()
    entity_mapping = dict()
    anonymized_result = anonymizer_engine.anonymize(initial_query, analyzer_results, {'DEFAULT': OperatorConfig('entity_counter', {'entity_mapping': entity_mapping})})
    response = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': anonymized_result.text}])
    final_response = response.choices[0].message.content.strip()
    final_response = replace_entities(entity_mapping, final_response)
    return (final_response, response.usage.completion_tokens)

def download_model(model_name):
    global _model_downloaded
    if not _model_downloaded:
        if not spacy.util.is_package(model_name):
            print(f'Downloading {model_name} model...')
            spacy.cli.download(model_name)
        else:
            print(f'{model_name} model already downloaded.')
        _model_downloaded = True

def get_analyzer_engine() -> AnalyzerEngine:
    """Get or create singleton AnalyzerEngine instance."""
    global _analyzer_engine
    if _analyzer_engine is None:
        _analyzer_engine = AnalyzerEngine()
        _analyzer_engine.analyze(text='warm up', language='en')
    return _analyzer_engine

def get_anonymizer_engine() -> AnonymizerEngine:
    """Get or create singleton AnonymizerEngine instance."""
    global _anonymizer_engine
    if _anonymizer_engine is None:
        _anonymizer_engine = AnonymizerEngine()
        _anonymizer_engine.add_anonymizer(InstanceCounterAnonymizer)
    return _anonymizer_engine

def replace_entities(entity_map, text):
    reverse_map = {}
    for entity_type, entities in entity_map.items():
        for entity_name, placeholder in entities.items():
            reverse_map[placeholder] = entity_name

    def replace_placeholder(match):
        placeholder = match.group(0)
        return reverse_map.get(placeholder, placeholder)
    import re
    pattern = '<[A-Z_]+_\\d+>'
    replaced_text = re.sub(pattern, replace_placeholder, text)
    return replaced_text

def test_privacy_plugin_resource_caching():
    """
    Test that expensive resources (AnalyzerEngine, AnonymizerEngine) are created only once
    and reused across multiple plugin invocations.
    """
    print('Testing privacy plugin resource caching...')
    if 'optillm.plugins.privacy_plugin' in sys.modules:
        del sys.modules['optillm.plugins.privacy_plugin']
    with patch('presidio_analyzer.AnalyzerEngine') as MockAnalyzerEngine, patch('presidio_anonymizer.AnonymizerEngine') as MockAnonymizerEngine, patch('spacy.util.is_package', return_value=True):
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.analyze.return_value = []
        MockAnalyzerEngine.return_value = mock_analyzer_instance
        mock_anonymizer_instance = MagicMock()
        mock_anonymizer_instance.anonymize.return_value = MagicMock(text='anonymized text')
        mock_anonymizer_instance.add_anonymizer = MagicMock()
        MockAnonymizerEngine.return_value = mock_anonymizer_instance
        import optillm.plugins.privacy_plugin as privacy_plugin
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='response'))]
        mock_response.usage.completion_tokens = 10
        mock_client.chat.completions.create.return_value = mock_response
        print('First invocation...')
        result1, tokens1 = privacy_plugin.run('system', 'query 1', mock_client, 'model')
        assert MockAnalyzerEngine.call_count == 1, f'AnalyzerEngine created {MockAnalyzerEngine.call_count} times, expected 1'
        assert MockAnonymizerEngine.call_count == 1, f'AnonymizerEngine created {MockAnonymizerEngine.call_count} times, expected 1'
        print('Second invocation...')
        result2, tokens2 = privacy_plugin.run('system', 'query 2', mock_client, 'model')
        assert MockAnalyzerEngine.call_count == 1, f'AnalyzerEngine created {MockAnalyzerEngine.call_count} times after 2nd call, expected 1'
        assert MockAnonymizerEngine.call_count == 1, f'AnonymizerEngine created {MockAnonymizerEngine.call_count} times after 2nd call, expected 1'
        print('Third invocation...')
        result3, tokens3 = privacy_plugin.run('system', 'query 3', mock_client, 'model')
        assert MockAnalyzerEngine.call_count == 1, f'AnalyzerEngine created {MockAnalyzerEngine.call_count} times after 3rd call, expected 1'
        assert MockAnonymizerEngine.call_count == 1, f'AnonymizerEngine created {MockAnonymizerEngine.call_count} times after 3rd call, expected 1'
        print('✅ Privacy plugin resource caching test PASSED - Resources are properly cached!')
        return True

def test_privacy_plugin_performance():
    """
    Test that multiple invocations of the privacy plugin don't have degraded performance.
    This catches the actual performance issue even without mocking.
    """
    print('\nTesting privacy plugin performance (real execution)...')
    try:
        import optillm.plugins.privacy_plugin as privacy_plugin
        try:
            import spacy
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
        except ImportError as e:
            print(f'⚠️  Skipping performance test - dependencies not installed: {e}')
            return True
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='response'))]
        mock_response.usage.completion_tokens = 10
        mock_client.chat.completions.create.return_value = mock_response
        print('Warm-up call...')
        start = time.time()
        privacy_plugin.run('system', 'warm up query', mock_client, 'model')
        warmup_time = time.time() - start
        print(f'Warm-up time: {warmup_time:.2f}s')
        print('First measurement call...')
        start = time.time()
        privacy_plugin.run('system', 'test query 1', mock_client, 'model')
        first_time = time.time() - start
        print(f'First call time: {first_time:.2f}s')
        print('Second measurement call...')
        start = time.time()
        privacy_plugin.run('system', 'test query 2', mock_client, 'model')
        second_time = time.time() - start
        print(f'Second call time: {second_time:.2f}s')
        print('Third measurement call...')
        start = time.time()
        privacy_plugin.run('system', 'test query 3', mock_client, 'model')
        third_time = time.time() - start
        print(f'Third call time: {third_time:.2f}s')
        max_acceptable_time = 2.0
        if second_time > max_acceptable_time:
            raise AssertionError(f'Second call took {second_time:.2f}s, expected < {max_acceptable_time}s. Resources might not be cached!')
        if third_time > max_acceptable_time:
            raise AssertionError(f'Third call took {third_time:.2f}s, expected < {max_acceptable_time}s. Resources might not be cached!')
        print(f'✅ Privacy plugin performance test PASSED - Subsequent calls are fast ({second_time:.2f}s, {third_time:.2f}s)!')
        return True
    except Exception as e:
        print(f'❌ Performance test failed: {e}')
        raise

def test_singleton_instances_are_reused():
    """
    Direct test that singleton instances are the same object across calls.
    """
    print('\nTesting singleton instance reuse...')
    try:
        import optillm.plugins.privacy_plugin as privacy_plugin
        importlib.reload(privacy_plugin)
        analyzer1 = privacy_plugin.get_analyzer_engine()
        anonymizer1 = privacy_plugin.get_anonymizer_engine()
        analyzer2 = privacy_plugin.get_analyzer_engine()
        anonymizer2 = privacy_plugin.get_anonymizer_engine()
        assert analyzer1 is analyzer2, 'AnalyzerEngine instances are not the same object!'
        assert anonymizer1 is anonymizer2, 'AnonymizerEngine instances are not the same object!'
        print('✅ Singleton instance test PASSED - Same objects are reused!')
        return True
    except ImportError as e:
        print(f'⚠️  Skipping singleton test - dependencies not installed: {e}')
        return True
    except Exception as e:
        print(f'❌ Singleton test failed: {e}')
        raise

def test_recognizers_not_reloaded():
    """
    Test that recognizers are not fetched/reloaded on each analyze() call.
    This prevents the performance regression where "Fetching all recognizers for language en"
    appears in logs on every request.
    """
    print('\nTesting that recognizers are not reloaded on each call...')
    if 'optillm.plugins.privacy_plugin' in sys.modules:
        del sys.modules['optillm.plugins.privacy_plugin']
    try:
        with patch('presidio_analyzer.AnalyzerEngine') as MockAnalyzerEngine, patch('spacy.util.is_package', return_value=True):
            mock_analyzer_instance = MagicMock()
            mock_registry = MagicMock()
            mock_registry.get_recognizers = MagicMock(return_value=[])
            mock_analyzer_instance.registry = mock_registry
            mock_analyzer_instance.analyze = MagicMock(return_value=[])
            MockAnalyzerEngine.return_value = mock_analyzer_instance
            import optillm.plugins.privacy_plugin as privacy_plugin
            analyzer1 = privacy_plugin.get_analyzer_engine()
            initial_analyze_calls = mock_analyzer_instance.analyze.call_count
            print(f'Warm-up analyze calls: {initial_analyze_calls}')
            assert initial_analyze_calls == 1, f'Expected 1 warm-up analyze call, got {initial_analyze_calls}'
            analyzer2 = privacy_plugin.get_analyzer_engine()
            second_analyze_calls = mock_analyzer_instance.analyze.call_count
            print(f'Total analyze calls after second get_analyzer_engine: {second_analyze_calls}')
            assert second_analyze_calls == 1, f'Analyzer should not call analyze() again on cached retrieval, got {second_analyze_calls} calls'
            assert analyzer1 is analyzer2, 'Should return the same cached analyzer instance'
            print('✅ Recognizer reload test PASSED - Recognizers are pre-warmed and not reloaded!')
            return True
    except ImportError as e:
        print(f'⚠️  Skipping recognizer reload test - dependencies not installed: {e}')
        return True
    except Exception as e:
        print(f'❌ Recognizer reload test failed: {e}')
        raise

