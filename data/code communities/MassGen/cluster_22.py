# Cluster 22

@patch('massgen.adapters.ag2_adapter.setup_agent_from_config')
def test_adapter_init_single_agent(mock_setup):
    """Test adapter initialization with single agent config."""
    mock_agent = MagicMock()
    mock_setup.return_value = mock_agent
    agent_config = {'type': 'assistant', 'name': 'test', 'llm_config': {'api_type': 'openai', 'model': 'gpt-4o'}}
    adapter = AG2Adapter(agent_config=agent_config)
    assert adapter.is_group_chat is False
    assert adapter.agent == mock_agent
    mock_setup.assert_called_once_with(agent_config)

def test_adapter_init_requires_config():
    """Test that adapter requires either agent_config or group_config."""
    with pytest.raises(ValueError) as exc_info:
        AG2Adapter()
    assert 'agent_config' in str(exc_info.value) or 'group_config' in str(exc_info.value)

def test_adapter_init_rejects_both_configs():
    """Test that adapter rejects both agent_config and group_config."""
    with pytest.raises(ValueError) as exc_info:
        AG2Adapter(agent_config={'name': 'test', 'llm_config': []}, group_config={'agents': []})
    assert 'not both' in str(exc_info.value).lower()

@patch('massgen.adapters.ag2_adapter.setup_agent_from_config')
def test_register_tools_single_agent(mock_setup):
    """Test tool registration with single agent."""
    mock_agent = MagicMock()
    mock_setup.return_value = mock_agent
    agent_config = {'type': 'assistant', 'name': 'test', 'llm_config': {'api_type': 'openai', 'model': 'gpt-4o'}}
    adapter = AG2Adapter(agent_config=agent_config)
    tools = [{'type': 'function', 'function': {'name': 'search', 'description': 'Search tool'}}]
    adapter._register_tools(tools)
    assert mock_agent.update_tool_signature.call_count == len(tools)

@patch('massgen.adapters.ag2_adapter.setup_agent_from_config')
def test_register_tools_empty_list(mock_setup):
    """Test that empty tool list doesn't call update_tool_signature."""
    mock_agent = MagicMock()
    mock_setup.return_value = mock_agent
    agent_config = {'type': 'assistant', 'name': 'test', 'llm_config': {'api_type': 'openai', 'model': 'gpt-4o'}}
    adapter = AG2Adapter(agent_config=agent_config)
    adapter._register_tools([])
    mock_agent.update_tool_signature.assert_not_called()

@patch('massgen.adapters.ag2_adapter.ConversableAgent')
@patch('massgen.adapters.ag2_adapter.setup_agent_from_config')
@patch('massgen.adapters.ag2_adapter.AutoPattern')
def test_adapter_init_group_chat(mock_pattern, mock_setup, mock_conversable):
    """Test adapter initialization with group chat config."""
    mock_agent1 = MagicMock()
    mock_agent1.name = 'Agent1'
    mock_agent2 = MagicMock()
    mock_agent2.name = 'Agent2'
    mock_user_agent = MagicMock()
    mock_user_agent.name = 'User'
    mock_setup.side_effect = [mock_agent1, mock_agent2]
    mock_conversable.return_value = mock_user_agent
    group_config = {'llm_config': {'api_type': 'openai', 'model': 'gpt-4o'}, 'agents': [{'type': 'assistant', 'name': 'Agent1', 'llm_config': {'api_type': 'openai', 'model': 'gpt-4o'}}, {'type': 'assistant', 'name': 'Agent2', 'llm_config': {'api_type': 'openai', 'model': 'gpt-4o'}}], 'pattern': {'type': 'auto', 'initial_agent': 'Agent1', 'group_manager_args': {'llm_config': {'api_type': 'openai', 'model': 'gpt-4o'}}}}
    adapter = AG2Adapter(group_config=group_config)
    assert adapter.is_group_chat is True
    assert len(adapter.agents) == 2
    assert adapter.user_agent is not None
    mock_pattern.assert_called_once()

@patch('massgen.adapters.ag2_adapter.setup_agent_from_config')
def test_adapter_separate_workflow_and_other_tools(mock_setup):
    """Test separation of workflow and other tools."""
    mock_agent = MagicMock()
    mock_setup.return_value = mock_agent
    agent_config = {'type': 'assistant', 'name': 'test', 'llm_config': {'api_type': 'openai', 'model': 'gpt-4o'}}
    adapter = AG2Adapter(agent_config=agent_config)
    tools = [{'type': 'function', 'function': {'name': 'new_answer', 'description': 'Submit answer'}}, {'type': 'function', 'function': {'name': 'vote', 'description': 'Vote for answer'}}, {'type': 'function', 'function': {'name': 'search', 'description': 'Search tool'}}]
    workflow_tools, other_tools = adapter._separate_workflow_and_other_tools(tools)
    assert len(workflow_tools) == 2
    assert len(other_tools) == 1
    assert any((t['function']['name'] == 'new_answer' for t in workflow_tools))
    assert any((t['function']['name'] == 'vote' for t in workflow_tools))
    assert other_tools[0]['function']['name'] == 'search'

@patch('massgen.adapters.ag2_adapter.ConversableAgent')
@patch('massgen.adapters.ag2_adapter.setup_agent_from_config')
@patch('massgen.adapters.ag2_adapter.AutoPattern')
def test_adapter_setup_user_agent_custom(mock_pattern, mock_setup, mock_conversable):
    """Test setting up custom user agent."""
    mock_user_agent = MagicMock()
    mock_user_agent.name = 'User'
    mock_agent = MagicMock()
    mock_agent.name = 'TestAgent'
    mock_setup.side_effect = [mock_agent, mock_user_agent]
    group_config = {'llm_config': {'api_type': 'openai', 'model': 'gpt-4o'}, 'agents': [{'type': 'assistant', 'name': 'TestAgent', 'llm_config': {'api_type': 'openai', 'model': 'gpt-4o'}}], 'pattern': {'type': 'auto', 'initial_agent': 'TestAgent', 'group_manager_args': {'llm_config': {'api_type': 'openai', 'model': 'gpt-4o'}}}, 'user_agent': {'name': 'User', 'system_message': 'Custom user agent', 'llm_config': {'api_type': 'openai', 'model': 'gpt-4o'}}}
    adapter = AG2Adapter(group_config=group_config)
    assert adapter.user_agent.name == 'User'
    assert mock_setup.call_count == 2

@patch('massgen.adapters.ag2_adapter.setup_agent_from_config')
@patch('massgen.adapters.ag2_adapter.AutoPattern')
def test_adapter_invalid_pattern_type(mock_pattern, mock_setup):
    """Test that invalid pattern type raises error."""
    mock_agent = MagicMock()
    mock_agent.name = 'Agent1'
    mock_setup.return_value = mock_agent
    group_config = {'llm_config': {'api_type': 'openai', 'model': 'gpt-4o'}, 'agents': [{'type': 'assistant', 'name': 'Agent1', 'llm_config': {'api_type': 'openai', 'model': 'gpt-4o'}}], 'pattern': {'type': 'invalid_pattern', 'initial_agent': 'Agent1'}}
    with pytest.raises(NotImplementedError) as exc_info:
        AG2Adapter(group_config=group_config)
    assert 'invalid_pattern' in str(exc_info.value)

