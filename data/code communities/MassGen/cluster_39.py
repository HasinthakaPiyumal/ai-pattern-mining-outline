# Cluster 39

def test_unregister_tools_for_agent():
    """Test unregistering tools from agent."""
    mock_agent = MagicMock()
    tools = [{'type': 'function', 'function': {'name': 'search', 'description': 'Search tool'}}]
    unregister_tools_for_agent(tools, mock_agent)
    mock_agent.update_tool_signature.assert_called_once()
    call_kwargs = mock_agent.update_tool_signature.call_args[1]
    assert call_kwargs['is_remove'] is True

def unregister_tools_for_agent(tools: List[Dict[str, Any]], agent: ConversableAgent) -> None:
    """Unregister all tools from single agent."""
    for tool in tools:
        agent.update_tool_signature(tool_sig=tool, is_remove=True, silent_override=True)

