# Cluster 88

def test_set_upstream_updates_session_each_request():
    fastmcp, app, app_context = make_attached_app()
    try:
        ctx1 = DummyMCPContext('session-one', fastmcp)
        bound_ctx1, token1 = app_server._enter_request_context(ctx1)
        assert bound_ctx1.upstream_session is ctx1.session
        assert app_context.upstream_session is ctx1.session
        assert 'session-one' in app_context.identity_registry
        assert app_context.identity_registry['session-one'].subject == 'session-one'
        assert app_context.session_id == 'app-session'
        app_server._exit_request_context(bound_ctx1, token1)
        assert app_context.upstream_session is None
        ctx2 = DummyMCPContext('session-two', fastmcp)
        bound_ctx2, token2 = app_server._enter_request_context(ctx2)
        assert bound_ctx2.upstream_session is ctx2.session
        assert app_context.upstream_session is ctx2.session
        assert 'session-two' in app_context.identity_registry
        assert app_context.identity_registry['session-two'].subject == 'session-two'
        assert app_context.identity_registry['session-one'].subject == 'session-one'
        assert app_context.session_id == 'app-session'
        app_server._exit_request_context(bound_ctx2, token2)
        assert app_context.upstream_session is None
    finally:
        reset_identity()

def make_attached_app():
    fastmcp = FastMCP(name='test', instructions='test')
    app_context = Context()
    app_context.session_id = 'app-session'
    app = SimpleNamespace(context=app_context, _session_id_override='app-default')
    setattr(fastmcp, '_mcp_agent_app', app)
    return (fastmcp, app, app_context)

def reset_identity():
    app_server._set_current_identity(None)

def test_resolve_identity_prefers_request_session(monkeypatch):
    fastmcp, app, app_context = make_attached_app()
    ctx = DummyMCPContext('client-session', fastmcp)
    bound_ctx, token = app_server._enter_request_context(ctx)
    identity = app_server._resolve_identity_for_request(ctx=ctx, app_context=app_context, execution_id=None)
    assert isinstance(identity, OAuthUserIdentity)
    assert identity.subject == 'client-session'
    app_server._exit_request_context(bound_ctx, token)

