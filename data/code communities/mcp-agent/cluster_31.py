# Cluster 31

def test_session_prefers_explicit_upstream():
    upstream = object()
    ctx = _make_context()
    ctx.upstream_session = upstream
    assert ctx.session is upstream

def _make_context(*, app: SimpleNamespace | None=None) -> Context:
    ctx = Context()
    if app is not None:
        ctx.app = app
    return ctx

def test_fastmcp_fallback_to_app():
    dummy_mcp = object()
    app = SimpleNamespace(mcp=dummy_mcp, logger=None)
    ctx = _make_context(app=app)
    assert ctx.fastmcp is dummy_mcp
    bound = ctx.bind_request(SimpleNamespace(), fastmcp='request_mcp')
    assert bound.fastmcp == 'request_mcp'
    assert ctx.fastmcp is dummy_mcp

def test_logger_property_uses_app_logger():
    dummy_logger = _DummyLogger()
    app = SimpleNamespace(mcp=None, logger=dummy_logger, name='demo-app')
    ctx = _make_context(app=app)
    assert ctx.logger is dummy_logger

def test_logger_property_without_app_creates_logger():
    ctx = _make_context()
    logger = ctx.logger
    assert isinstance(logger, AgentLogger)
    assert getattr(logger, '_bound_context', None) is ctx

def test_name_and_description_properties():
    app = SimpleNamespace(mcp=None, logger=_DummyLogger(), name='app-name', description='app-desc')
    ctx = _make_context(app=app)
    ctx.config = SimpleNamespace(name='config-name', description='config-desc')
    assert ctx.name == 'app-name'
    assert ctx.description == 'app-desc'
    ctx_no_app = _make_context()
    assert ctx_no_app.name is None
    assert ctx_no_app.description is None

