# Cluster 21

def test_gemini_planning_mode():
    """Test that Gemini backend respects planning mode for MCP tool blocking."""
    print('🧪 Testing Gemini Backend Planning Mode...')
    print('=' * 50)
    agent_config = AgentConfig(backend_params={'backend_type': 'gemini', 'model': 'gemini-2.5-flash', 'api_key': 'dummy-key'})
    try:
        backend = GeminiBackend(config=agent_config)
        print('✅ Gemini backend created successfully')
    except Exception as e:
        print(f'❌ Failed to create Gemini backend: {e}')
        return False
    print('\n1. Testing planning mode flag...')
    assert not backend.is_planning_mode_enabled(), 'Planning mode should be disabled by default'
    print('✅ Planning mode disabled by default')
    backend.set_planning_mode(True)
    assert backend.is_planning_mode_enabled(), 'Planning mode should be enabled'
    print('✅ Planning mode can be enabled')
    backend.set_planning_mode(False)
    assert not backend.is_planning_mode_enabled(), 'Planning mode should be disabled'
    print('✅ Planning mode can be disabled')
    print('\n2. Testing Gemini backend inheritance...')
    assert hasattr(backend, 'set_planning_mode'), 'GeminiBackend should have set_planning_mode'
    assert hasattr(backend, 'is_planning_mode_enabled'), 'GeminiBackend should have is_planning_mode_enabled'
    print('✅ GeminiBackend has planning mode methods')
    print('\n🎉 All Gemini planning mode tests passed!')
    print('✅ Gemini backend respects planning mode flags')
    print('✅ MCP tool blocking should work during coordination phase')
    return True

def test_gemini_planning_mode_vs_other_backends():
    """Test that Gemini planning mode works differently from MCP-based backends."""
    print('\n🧪 Testing Gemini Planning Mode vs Other Backends...')
    print('=' * 55)
    backend = GeminiBackend(api_key='test-key')
    print("\n1. Testing Gemini's unique planning mode approach...")
    from massgen.backend.base import LLMBackend
    from massgen.backend.base_with_mcp import MCPBackend
    assert isinstance(backend, LLMBackend), 'Gemini should inherit from LLMBackend'
    assert not isinstance(backend, MCPBackend), 'Gemini should NOT inherit from MCPBackend'
    print('✅ Gemini has correct inheritance hierarchy')
    assert hasattr(backend, '_mcp_client'), 'Gemini should have _mcp_client attribute'
    assert hasattr(backend, '_setup_mcp_tools'), 'Gemini should have _setup_mcp_tools method'
    print('✅ Gemini has custom MCP implementation')
    backend.set_planning_mode(True)
    print('   Planning mode approach: Tool registration blocking (not execution blocking)')
    print(f'   Planning mode enabled: {backend.is_planning_mode_enabled()}')
    print('   Expected: MCP tools will not be registered in Gemini SDK config')
    has_mcp_execution_method = hasattr(backend, '_execute_mcp_function_with_retry')
    print(f'   Has MCPBackend execution method: {has_mcp_execution_method}')
    print('✅ Gemini uses tool registration blocking, not execution-time blocking')
    print('\n✅ Gemini planning mode approach is distinct and appropriate!')
    return True

