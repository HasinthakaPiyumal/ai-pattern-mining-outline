# Cluster 91

def get_current_request_context() -> Optional['Context']:
    """Return the currently bound request-scoped context, if any."""
    try:
        return _CURRENT_REQUEST_CONTEXT.get()
    except LookupError:
        return None

def test_logger_uses_request_context_and_restores_default():
    base_ctx = Context()
    base_ctx.session_id = 'base-session'
    set_default_bound_context(base_ctx)
    logger = get_logger('tests.request_scope', context=base_ctx)
    original_emit = logger._emit_event
    events: list = []
    try:
        logger._emit_event = lambda event: events.append(event)
        ctx_a = base_ctx.bind_request(None)
        ctx_a.upstream_session = object()
        ctx_a.request_session_id = 'client-a'
        token_a = set_current_request_context(ctx_a)
        try:
            logger.info('from client A')
        finally:
            reset_current_request_context(token_a)
        assert get_current_request_context() is None
        event_a = events[0]
        assert event_a.upstream_session is ctx_a.upstream_session
        assert event_a.context is not None and event_a.context.session_id == 'client-a'
        assert getattr(base_ctx, 'upstream_session', None) is None
        ctx_b = base_ctx.bind_request(None)
        ctx_b.upstream_session = object()
        ctx_b.request_session_id = 'client-b'
        token_b = set_current_request_context(ctx_b)
        try:
            logger.info('from client B')
        finally:
            reset_current_request_context(token_b)
        event_b = events[1]
        assert event_b.upstream_session is ctx_b.upstream_session
        assert event_b.context is not None and event_b.context.session_id == 'client-b'
        assert event_a.upstream_session is not event_b.upstream_session
    finally:
        logger._emit_event = original_emit
        set_default_bound_context(None)

def set_default_bound_context(ctx: Any | None) -> None:
    global _default_bound_context
    _default_bound_context = ctx

def set_current_request_context(ctx: Optional['Context']) -> Token:
    """Bind the given context to the current execution context."""
    return _CURRENT_REQUEST_CONTEXT.set(ctx)

def reset_current_request_context(token: Token | None) -> None:
    """Reset the request context to a previous state."""
    if token is None:
        return
    try:
        _CURRENT_REQUEST_CONTEXT.reset(token)
    except Exception:
        pass

def test_exit_request_context_clears_session_level():
    ctx = Context()
    ctx.request_session_id = 'client-exit'
    token = set_current_request_context(ctx)
    try:
        LoggingConfig.set_session_min_level('client-exit', 'warning')
        assert LoggingConfig.get_session_min_level('client-exit') == 'warning'
    finally:
        app_server._exit_request_context(ctx, token)
    assert LoggingConfig.get_session_min_level('client-exit') == 'warning'
    LoggingConfig.clear_session_min_level('client-exit')

def test_base_context_delegates_to_request_clone():
    base = Context()
    request_ctx = base.bind_request(request_context=None)
    request_ctx.upstream_session = object()
    token = set_current_request_context(request_ctx)
    try:
        assert base.upstream_session is request_ctx.upstream_session
    finally:
        reset_current_request_context(token)
    assert base.upstream_session is None

