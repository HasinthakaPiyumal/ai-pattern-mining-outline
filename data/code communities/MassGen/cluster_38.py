# Cluster 38

def test_register_tools_for_agent():
    """Test registering tools with agent."""
    mock_agent = MagicMock()
    tools = [{'type': 'function', 'function': {'name': 'search', 'description': 'Search tool'}}, {'type': 'function', 'function': {'name': 'calc', 'description': 'Calculator tool'}}]
    register_tools_for_agent(tools, mock_agent)
    assert mock_agent.update_tool_signature.call_count == len(tools)
    for call in mock_agent.update_tool_signature.call_args_list:
        assert call[1]['is_remove'] is False

def register_tools_for_agent(tools: List[Dict[str, Any]], agent: ConversableAgent) -> None:
    """Register all tools to single agent."""
    for tool in tools:
        agent.update_tool_signature(tool_sig=tool, is_remove=False, silent_override=True)

