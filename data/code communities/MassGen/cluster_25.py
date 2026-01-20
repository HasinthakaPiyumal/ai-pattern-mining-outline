# Cluster 25

def print_config_example():
    """Print example configuration for users."""
    print('\n📋 Example YAML Configuration with Timeout Settings:')
    print('=' * 50)
    example_config = '\n# Conservative timeout settings to prevent runaway costs\ntimeout_settings:\n  orchestrator_timeout_seconds: 600   # 10 minutes max coordination\n\nagents:\n  - id: "agent1"\n    backend:\n      type: "openai"\n      model: "gpt-4o-mini"\n    system_message: "You are a helpful assistant."\n'
    print(example_config)
    print('\n🖥️  CLI Examples:')
    print('python -m massgen.cli --config config.yaml --orchestrator-timeout 300 "Complex task"')

def is_rich_available() -> bool:
    """Check if Rich library is available."""
    return RICH_AVAILABLE

def test_orchestrator_initialization_with_context_sharing(test_workspace, mock_agents):
    """Test orchestrator initializes with context sharing parameters."""
    orchestrator = Orchestrator(agents=mock_agents, snapshot_storage=test_workspace['snapshot_storage'], agent_temporary_workspace=test_workspace['temp_workspace'])
    assert orchestrator._snapshot_storage == test_workspace['snapshot_storage']
    assert orchestrator._agent_temporary_workspace == test_workspace['temp_workspace']
    assert len(orchestrator._agent_id_mapping) == 3
    assert 'claude_code_1' in orchestrator._agent_id_mapping
    assert 'claude_code_2' in orchestrator._agent_id_mapping
    assert 'claude_code_3' in orchestrator._agent_id_mapping
    assert orchestrator._agent_id_mapping['claude_code_1'] == 'agent_1'
    assert orchestrator._agent_id_mapping['claude_code_2'] == 'agent_2'
    assert orchestrator._agent_id_mapping['claude_code_3'] == 'agent_3'
    assert Path(test_workspace['snapshot_storage']).exists()
    assert Path(test_workspace['temp_workspace']).exists()
    for agent_id in mock_agents.keys():
        snapshot_dir = Path(test_workspace['snapshot_storage']) / agent_id
        temp_dir = Path(test_workspace['temp_workspace']) / agent_id
        assert snapshot_dir.exists()
        assert temp_dir.exists()

def test_non_claude_code_agents_ignored(test_workspace):
    """Test that non-Claude Code agents are ignored for context sharing."""
    agents = {'claude_code_1': MockClaudeCodeAgent('claude_code_1'), 'regular_agent': MagicMock(backend=MagicMock(get_provider_name=lambda: 'openai'))}
    orchestrator = Orchestrator(agents=agents, snapshot_storage=test_workspace['snapshot_storage'], agent_temporary_workspace=test_workspace['temp_workspace'])
    assert 'claude_code_1' in orchestrator._agent_id_mapping
    assert 'regular_agent' not in orchestrator._agent_id_mapping
    assert len(orchestrator._agent_id_mapping) == 1

