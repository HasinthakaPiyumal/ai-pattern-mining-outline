# Cluster 26

def _get_system_message_support(proxy_client, model: str) -> bool:
    """
    Get cached system message support status, testing if not cached.
    Thread-safe with locking.
    """
    cache_key = f'{getattr(proxy_client, '_base_identifier', 'default')}:{model}'
    with _cache_lock:
        if cache_key not in _system_message_support_cache:
            logger.debug(f'Testing system message support for {model}')
            _system_message_support_cache[cache_key] = _test_system_message_support(proxy_client, model)
        return _system_message_support_cache[cache_key]

def _test_system_message_support(proxy_client, model: str) -> bool:
    """
    Test if a model supports system messages by making a minimal test request.
    Returns True if supported, False otherwise.
    """
    try:
        test_response = proxy_client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': 'test'}, {'role': 'user', 'content': 'hi'}], max_tokens=1, temperature=0)
        return True
    except Exception as e:
        error_msg = str(e).lower()
        if any((pattern in error_msg for pattern in ['developer instruction', 'system message', 'not enabled', 'not supported'])):
            logger.info(f'Model {model} does not support system messages: {str(e)[:100]}')
            return False
        else:
            logger.debug(f'System message test failed for {model}, assuming supported: {str(e)[:100]}')
            return True

def run(system_prompt: str, initial_query: str, client, model: str, request_config: dict=None) -> Tuple[str, int]:
    """
    Main proxy plugin entry point.
    
    Supports three usage modes:
    1. Standalone proxy: model="proxy-gpt-4"
    2. Wrapping approach: extra_body={"optillm_approach": "proxy", "proxy_wrap": "moa"}
    3. Combined approach: model="bon&proxy-gpt-4"
    
    Args:
        system_prompt: System message for the LLM
        initial_query: User's query
        client: Original OpenAI client (used as fallback)
        model: Model identifier
        request_config: Additional request configuration
    
    Returns:
        Tuple of (response_text, token_count)
    """
    try:
        config = ProxyConfig.load()
        if not config.get('providers'):
            logger.warning('No providers configured, falling back to original client')
            response = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': initial_query}])
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            return (response_dict, 0)
        config_key = str(config)
        if config_key not in _proxy_client_cache:
            logger.debug('Creating new proxy client instance')
            _proxy_client_cache[config_key] = ProxyClient(config=config, fallback_client=client)
        else:
            logger.debug('Reusing existing proxy client instance')
        proxy_client = _proxy_client_cache[config_key]
        wrapped_approach = None
        if request_config:
            wrapped_approach = request_config.get('proxy_wrap') or request_config.get('wrapped_approach') or request_config.get('wrap')
        if wrapped_approach:
            logger.info(f'Proxy wrapping approach/plugin: {wrapped_approach}')
            handler = ApproachHandler()
            result = handler.handle(wrapped_approach, system_prompt, initial_query, proxy_client, model, request_config)
            if result is not None:
                return result
            else:
                logger.warning(f"Approach/plugin '{wrapped_approach}' not found, using direct proxy")
        if '-' in model and (not wrapped_approach):
            parts = model.split('-', 1)
            potential_approach = parts[0]
            actual_model = parts[1] if len(parts) > 1 else model
            handler = ApproachHandler()
            result = handler.handle(potential_approach, system_prompt, initial_query, proxy_client, actual_model, request_config)
            if result is not None:
                logger.info(f'Proxy routing approach/plugin: {potential_approach}')
                return result
        logger.info(f'Direct proxy routing for model: {model}')
        supports_system_messages = _get_system_message_support(proxy_client, model)
        messages = _format_messages_for_model(system_prompt, initial_query, supports_system_messages)
        if not supports_system_messages:
            logger.info(f'Using fallback message formatting for {model} (no system message support)')
        response = proxy_client.chat.completions.create(model=model, messages=messages, **request_config or {})
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
        return (response_dict, 0)
    except Exception as e:
        logger.error(f'Proxy plugin error: {e}', exc_info=True)
        logger.info('Falling back to original client')
        response = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': initial_query}])
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
        return (response_dict, 0)

def _format_messages_for_model(system_prompt: str, initial_query: str, supports_system_messages: bool) -> list:
    """
    Format messages based on whether the model supports system messages.
    """
    if supports_system_messages:
        return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': initial_query}]
    else:
        if system_prompt.strip():
            combined_message = f'{system_prompt}\n\nUser: {initial_query}'
        else:
            combined_message = initial_query
        return [{'role': 'user', 'content': combined_message}]

def test_plugin_approach_detection():
    """Test plugin approach detection after loading"""
    load_plugins()
    expected_plugins = ['memory', 'readurls', 'privacy', 'web_search', 'deep_research', 'deepthink', 'longcepo', 'spl', 'proxy', 'mcp']
    for plugin_name in expected_plugins:
        assert plugin_name in plugin_approaches, f'Plugin {plugin_name} not loaded'

def test_proxy_plugin_token_counts():
    """Test that proxy plugin returns complete token usage information"""
    import optillm.plugins.proxy_plugin as plugin
    from unittest.mock import Mock, MagicMock
    mock_client = Mock()
    mock_response = MagicMock()
    mock_response.choices = [Mock(message=Mock(content='Test response'))]
    mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    mock_response.model_dump.return_value = {'choices': [{'message': {'content': 'Test response'}}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
    mock_client.chat.completions.create.return_value = mock_response
    result, _ = plugin.run(system_prompt='Test system', initial_query='Test query', client=mock_client, model='test-model')
    assert isinstance(result, dict), 'Result should be a dictionary'
    assert 'usage' in result, 'Result should contain usage information'
    assert 'prompt_tokens' in result['usage'], 'Usage should contain prompt_tokens'
    assert 'completion_tokens' in result['usage'], 'Usage should contain completion_tokens'
    assert 'total_tokens' in result['usage'], 'Usage should contain total_tokens'
    assert result['usage']['prompt_tokens'] == 10
    assert result['usage']['completion_tokens'] == 5
    assert result['usage']['total_tokens'] == 15

def test_proxy_plugin_timeout_config():
    """Test that proxy plugin properly configures timeout settings"""
    from optillm.plugins.proxy.config import ProxyConfig
    import tempfile
    import yaml
    config = {'providers': [{'name': 'test_provider', 'base_url': 'http://localhost:8000/v1', 'api_key': 'test-key'}], 'timeouts': {'request': 10, 'connect': 3}, 'queue': {'max_concurrent': 50, 'timeout': 30}}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    try:
        loaded_config = ProxyConfig.load(config_path)
        assert 'timeouts' in loaded_config, 'Config should contain timeouts section'
        assert loaded_config['timeouts'].get('request') == 10, 'Request timeout should be 10'
        assert loaded_config['timeouts'].get('connect') == 3, 'Connect timeout should be 3'
        assert 'queue' in loaded_config, 'Config should contain queue section'
        assert loaded_config['queue']['max_concurrent'] == 50, 'Max concurrent should be 50'
        assert loaded_config['queue']['timeout'] == 30, 'Queue timeout should be 30'
    finally:
        import os
        os.unlink(config_path)

def test_proxy_plugin_timeout_handling():
    """Test that proxy plugin handles timeouts correctly"""
    from optillm.plugins.proxy.client import ProxyClient
    from unittest.mock import Mock, patch
    import concurrent.futures
    config = {'providers': [{'name': 'slow_provider', 'base_url': 'http://localhost:8001/v1', 'api_key': 'test-key-1'}, {'name': 'fast_provider', 'base_url': 'http://localhost:8002/v1', 'api_key': 'test-key-2'}], 'routing': {'strategy': 'round_robin', 'health_check': {'enabled': False}}, 'timeouts': {'request': 2, 'connect': 1}, 'queue': {'max_concurrent': 10, 'timeout': 5}}
    proxy_client = ProxyClient(config)
    assert proxy_client.request_timeout == 2, 'Request timeout should be 2'
    assert proxy_client.connect_timeout == 1, 'Connect timeout should be 1'
    assert proxy_client.max_concurrent_requests == 10, 'Max concurrent should be 10'
    assert proxy_client.queue_timeout == 5, 'Queue timeout should be 5'

def test_plugin_module_imports():
    """Test that plugin modules can be imported"""
    plugin_modules = ['optillm.plugins.memory_plugin', 'optillm.plugins.readurls_plugin', 'optillm.plugins.privacy_plugin', 'optillm.plugins.genselect_plugin', 'optillm.plugins.majority_voting_plugin', 'optillm.plugins.web_search_plugin', 'optillm.plugins.deep_research_plugin', 'optillm.plugins.deepthink_plugin', 'optillm.plugins.longcepo_plugin', 'optillm.plugins.spl_plugin', 'optillm.plugins.proxy_plugin', 'optillm.plugins.mcp_plugin']
    for module_name in plugin_modules:
        try:
            module = importlib.import_module(module_name)
            assert hasattr(module, 'run'), f"{module_name} missing 'run' function"
            assert hasattr(module, 'SLUG'), f"{module_name} missing 'SLUG' attribute"
        except ImportError as e:
            if pytest:
                pytest.fail(f'Failed to import {module_name}: {e}')
            else:
                raise AssertionError(f'Failed to import {module_name}: {e}')

def test_memory_plugin_structure():
    """Test memory plugin has required structure"""
    import optillm.plugins.memory_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert plugin.SLUG == 'memory'
    assert hasattr(plugin, 'Memory')

def test_genselect_plugin():
    """Test genselect plugin module"""
    import optillm.plugins.genselect_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert hasattr(plugin, 'DEFAULT_NUM_CANDIDATES')
    assert plugin.SLUG == 'genselect'

def test_majority_voting_plugin():
    """Test majority voting plugin module"""
    import optillm.plugins.majority_voting_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert hasattr(plugin, 'extract_final_answer')
    assert hasattr(plugin, 'normalize_response')
    assert plugin.SLUG == 'majority_voting'

def test_web_search_plugin():
    """Test web search plugin module"""
    import optillm.plugins.web_search_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert hasattr(plugin, 'GoogleSearcher')
    assert hasattr(plugin, 'extract_search_queries')
    assert plugin.SLUG == 'web_search'

def test_deep_research_plugin():
    """Test deep research plugin module"""
    import optillm.plugins.deep_research_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert hasattr(plugin, 'DeepResearcher')
    assert plugin.SLUG == 'deep_research'

def test_deepthink_plugin_imports():
    """Test deepthink plugin and its submodules can be imported"""
    import optillm.plugins.deepthink_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert plugin.SLUG == 'deepthink'
    from optillm.plugins.deepthink import SelfDiscover, UncertaintyRoutedCoT
    assert SelfDiscover is not None
    assert UncertaintyRoutedCoT is not None

def test_longcepo_plugin():
    """Test longcepo plugin module"""
    import optillm.plugins.longcepo_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert plugin.SLUG == 'longcepo'
    from optillm.plugins.longcepo import run_longcepo
    assert run_longcepo is not None

def test_spl_plugin():
    """Test spl plugin module"""
    import optillm.plugins.spl_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert plugin.SLUG == 'spl'
    from optillm.plugins.spl import run_spl
    assert run_spl is not None

def test_proxy_plugin():
    """Test proxy plugin module"""
    import optillm.plugins.proxy_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert plugin.SLUG == 'proxy'
    from optillm.plugins.proxy import client, config, approach_handler
    assert client is not None
    assert config is not None
    assert approach_handler is not None

def test_mcp_plugin():
    """Test MCP plugin module"""
    import optillm.plugins.mcp_plugin as plugin
    assert hasattr(plugin, 'run')
    assert hasattr(plugin, 'SLUG')
    assert hasattr(plugin, 'ServerConfig')
    assert hasattr(plugin, 'MCPServer')
    assert hasattr(plugin, 'execute_tool')
    assert plugin.SLUG == 'mcp'

def test_plugin_subdirectory_imports():
    """Test all plugins with subdirectories can import their submodules"""
    from optillm.plugins.deep_research import DeepResearcher
    assert DeepResearcher is not None
    from optillm.plugins.deepthink import SelfDiscover, UncertaintyRoutedCoT
    assert SelfDiscover is not None
    assert UncertaintyRoutedCoT is not None
    from optillm.plugins.longcepo import run_longcepo
    assert run_longcepo is not None
    from optillm.plugins.spl import run_spl
    assert run_spl is not None
    from optillm.plugins.proxy import client, config, approach_handler
    assert client is not None
    assert config is not None
    assert approach_handler is not None

def test_no_relative_import_errors():
    """Test that plugins load without relative import errors"""
    import importlib
    import sys
    plugins_with_subdirs = ['optillm.plugins.deepthink_plugin', 'optillm.plugins.deep_research_plugin', 'optillm.plugins.longcepo_plugin', 'optillm.plugins.spl_plugin', 'optillm.plugins.proxy_plugin']
    for plugin_name in plugins_with_subdirs:
        modules_to_clear = [k for k in sys.modules.keys() if k.startswith(plugin_name)]
        for mod in modules_to_clear:
            del sys.modules[mod]
        try:
            module = importlib.import_module(plugin_name)
            assert hasattr(module, 'run'), f'{plugin_name} missing run function'
        except ImportError as e:
            if 'attempted relative import' in str(e):
                if pytest:
                    pytest.fail(f'Relative import error in {plugin_name}: {e}')
                else:
                    raise AssertionError(f'Relative import error in {plugin_name}: {e}')
            else:
                raise

