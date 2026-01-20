# Cluster 8

def create_workflow_tools(mcp: FastMCP, server_context: ServerContext):
    """
    Create workflow-specific tools for registered workflows.
    This is called at server start to register specific endpoints for each workflow.
    """
    if not server_context:
        logger.warning('Server config not available for creating workflow tools')
        return
    registered_workflow_tools = _get_registered_workflow_tools(mcp)
    for workflow_name, workflow_cls in server_context.workflows.items():
        if getattr(workflow_cls, '__mcp_agent_sync_tool__', False):
            continue
        if getattr(workflow_cls, '__mcp_agent_async_tool__', False):
            continue
        if workflow_name not in registered_workflow_tools:
            create_workflow_specific_tools(mcp, workflow_name, workflow_cls)
            registered_workflow_tools.add(workflow_name)
    setattr(mcp, '_registered_workflow_tools', registered_workflow_tools)

def _get_registered_workflow_tools(mcp: FastMCP) -> Set[str]:
    """Return the set of registered workflow tools for the FastMCP server, if any."""
    return getattr(mcp, '_registered_workflow_tools', set())

def test_workflow_tools_idempotent_registration():
    """Test that workflow tools are only registered once per workflow"""
    mock_mcp = MagicMock()
    mock_app = MagicMock()
    mock_context = MagicMock(app=mock_app)
    if hasattr(mock_mcp, '_registered_workflow_tools'):
        delattr(mock_mcp, '_registered_workflow_tools')
    mock_app.workflows = {}
    mock_context.workflow_registry = None
    mock_context.config = MagicMock()
    mock_context.config.execution_engine = 'asyncio'
    server_context = ServerContext(mcp=mock_mcp, context=mock_context)
    mock_workflow_class = MagicMock()
    mock_workflow_class.__doc__ = 'Test workflow'
    mock_run = MagicMock()
    mock_run.__name__ = 'run'
    mock_workflow_class.run = mock_run
    mock_app.workflows = {'workflow1': mock_workflow_class, 'workflow2': mock_workflow_class}
    tools_created = []

    def track_tool_calls(*args, **kwargs):

        def decorator(func):
            tools_created.append(kwargs.get('name', args[0] if args else 'unknown'))
            return func
        return decorator
    mock_mcp.tool = track_tool_calls
    create_workflow_tools(mock_mcp, server_context)
    expected_tools = ['workflows-workflow1-run', 'workflows-workflow2-run']
    assert len(tools_created) == 2
    for expected_tool in expected_tools:
        assert expected_tool in tools_created
    assert hasattr(mock_mcp, '_registered_workflow_tools')
    assert mock_mcp._registered_workflow_tools == {'workflow1', 'workflow2'}
    tools_created.clear()
    create_workflow_tools(mock_mcp, server_context)
    assert len(tools_created) == 0
    assert mock_mcp._registered_workflow_tools == {'workflow1', 'workflow2'}
    new_workflow_class = MagicMock()
    new_workflow_class.__doc__ = 'New workflow'
    new_mock_run = MagicMock()
    new_mock_run.__name__ = 'run'
    new_workflow_class.run = new_mock_run
    server_context.register_workflow('workflow3', new_workflow_class)
    assert 'workflow3' in server_context.workflows
    assert 'workflow3' in mock_mcp._registered_workflow_tools
    assert len(tools_created) == 1
    assert 'workflows-workflow3-run' in tools_created
    tools_created.clear()
    server_context.register_workflow('workflow3', new_workflow_class)
    assert len(tools_created) == 0
    assert mock_mcp._registered_workflow_tools == {'workflow1', 'workflow2', 'workflow3'}

def test_workflow_tools_persistent_across_sse_requests():
    """Test that workflow tools registration persists across SSE request context recreation"""
    mock_mcp = MagicMock()
    if hasattr(mock_mcp, '_registered_workflow_tools'):
        delattr(mock_mcp, '_registered_workflow_tools')
    mock_workflow_class = MagicMock()
    mock_workflow_class.__doc__ = 'Test workflow'
    mock_run = MagicMock()
    mock_run.__name__ = 'run'
    mock_workflow_class.run = mock_run
    tools_created = []

    def track_tool_calls(*args, **kwargs):

        def decorator(func):
            tools_created.append(kwargs.get('name', args[0] if args else 'unknown'))
            return func
        return decorator
    mock_mcp.tool = track_tool_calls
    mock_app1 = MagicMock()
    mock_context1 = MagicMock(app=mock_app1)
    mock_context1.workflow_registry = None
    mock_context1.config = MagicMock()
    mock_context1.config.execution_engine = 'asyncio'
    mock_app1.workflows = {'workflow1': mock_workflow_class}
    server_context1 = ServerContext(mcp=mock_mcp, context=mock_context1)
    create_workflow_tools(mock_mcp, server_context1)
    assert len(tools_created) == 1
    assert 'workflows-workflow1-run' in tools_created
    assert hasattr(mock_mcp, '_registered_workflow_tools')
    assert 'workflow1' in mock_mcp._registered_workflow_tools
    tools_created.clear()
    mock_app2 = MagicMock()
    mock_context2 = MagicMock(app=mock_app2)
    mock_context2.workflow_registry = None
    mock_context2.config = MagicMock()
    mock_context2.config.execution_engine = 'asyncio'
    mock_app2.workflows = {'workflow1': mock_workflow_class}
    server_context2 = ServerContext(mcp=mock_mcp, context=mock_context2)
    assert hasattr(mock_mcp, '_registered_workflow_tools')
    assert isinstance(mock_mcp._registered_workflow_tools, set)
    assert mock_mcp._registered_workflow_tools == {'workflow1'}
    create_workflow_tools(mock_mcp, server_context2)
    assert len(tools_created) == 0
    assert mock_mcp._registered_workflow_tools == {'workflow1'}

def _make_ctx(server_context):
    from types import SimpleNamespace
    ctx = SimpleNamespace()
    if not hasattr(server_context, 'workflow_registry'):
        from mcp_agent.executor.workflow_registry import InMemoryWorkflowRegistry
        server_context.workflow_registry = InMemoryWorkflowRegistry()
    req = SimpleNamespace(lifespan_context=server_context)
    ctx.request_context = req
    ctx.fastmcp = SimpleNamespace(_mcp_agent_app=None)
    return ctx

