# Cluster 9

def _clear_cached_session_refs(target: Any, session: Any | None) -> None:
    if target is None or session is None:
        return
    try:
        if getattr(target, '_last_known_upstream_session', None) is session:
            setattr(target, '_last_known_upstream_session', None)
    except Exception:
        pass

def _resolve_identity_for_request(ctx: MCPContext | None=None, app_context: 'Context' | None=None, execution_id: str | None=None) -> OAuthUserIdentity:
    identity = _CURRENT_IDENTITY.get()
    if identity is None and execution_id:
        identity = _get_identity_for_execution(execution_id)
    request_session_id: str | None = None
    if ctx is not None:
        request_session_id = _extract_session_id_from_context(ctx)
    if app_context is None and ctx is not None:
        app = _get_attached_app(ctx.fastmcp)
        if app is not None and getattr(app, 'context', None) is not None:
            app_context = app.context
    if identity is None and request_session_id:
        resolved = get_identity_for_session(request_session_id, app_context)
        if resolved:
            logger.debug('Resolved identity from session registry', data={'session_id': request_session_id, 'identity': resolved.cache_key})
            identity = resolved
    if identity is None and app_context is not None:
        session_id = getattr(app_context, 'session_id', None)
        if session_id and session_id != request_session_id:
            identity = get_identity_for_session(session_id, app_context)
    if identity is None:
        identity = DEFAULT_PRECONFIGURED_IDENTITY
    return identity

def _get_identity_for_execution(execution_id: str) -> OAuthUserIdentity | None:
    return _RUN_IDENTITY_REGISTRY.get(execution_id)

def _enter_request_context(ctx: MCPContext | None) -> Tuple[Optional['Context'], Token | None]:
    """Prepare and bind a per-request context, returning it alongside the contextvar token."""
    if ctx is None:
        return (None, None)
    try:
        session = ctx.session
    except (AttributeError, ValueError):
        session = None
    session_id = _extract_session_id_from_context(ctx)
    identity: OAuthUserIdentity | None = None
    try:
        auth_user = auth_context_var.get()
    except LookupError:
        auth_user = None
    if isinstance(auth_user, AuthenticatedUser):
        access_token = getattr(auth_user, 'access_token', None)
        if access_token is not None:
            try:
                from mcp_agent.oauth.access_token import MCPAccessToken
                if isinstance(access_token, MCPAccessToken):
                    identity = OAuthUserIdentity.from_access_token(access_token)
                else:
                    token_dict = getattr(access_token, 'model_dump', None)
                    if callable(token_dict):
                        maybe_token = MCPAccessToken.model_validate(token_dict())
                        if maybe_token is not None:
                            identity = OAuthUserIdentity.from_access_token(maybe_token)
            except Exception:
                identity = None
    base_context: Context | None = None
    lifespan_ctx = getattr(ctx.request_context, 'lifespan_context', None)
    if lifespan_ctx is not None and hasattr(lifespan_ctx, 'context') and (getattr(lifespan_ctx, 'context', None) is not None):
        base_context = lifespan_ctx.context
    if base_context is None:
        app: MCPApp | None = _get_attached_app(ctx.fastmcp)
        if app is not None and getattr(app, 'context', None) is not None:
            base_context = app.context
    if identity is None and session_id:
        identity = _session_identity_from_value(session_id)
    if identity is None:
        identity = DEFAULT_PRECONFIGURED_IDENTITY
    bound_context: Context | None = None
    token: Token | None = None
    if base_context is not None:
        previous_session = None
        try:
            previous_session = getattr(base_context, 'upstream_session', None)
        except Exception:
            previous_session = None
        bound_context = base_context.bind_request(getattr(ctx, 'request_context', None), getattr(ctx, 'fastmcp', None))
        if session is not None:
            bound_context.upstream_session = session
        try:
            setattr(bound_context, '_scoped_upstream_session', session)
        except Exception:
            pass
        try:
            setattr(bound_context, '_previous_upstream_session', previous_session)
        except Exception:
            pass
        bound_context.request_session_id = session_id
        bound_context.request_identity = identity
        token = set_current_request_context(bound_context)
        try:
            setattr(bound_context, '_base_context_ref', base_context)
        except Exception:
            pass
        if session is not None:
            try:
                setattr(base_context, '_last_known_upstream_session', session)
            except Exception:
                pass
            app_ref = getattr(base_context, 'app', None)
            if app_ref is not None:
                try:
                    setattr(app_ref, '_last_known_upstream_session', session)
                except Exception:
                    pass
        if session_id and identity is not None:
            try:
                base_context.identity_registry[session_id] = identity
                logger.debug('Registered identity for session', data={'session_id': session_id, 'identity': identity.cache_key})
            except Exception:
                pass
    else:
        token = None
    _set_current_identity(identity)
    return (bound_context, token)

def _get_attached_app(mcp: FastMCP) -> MCPApp | None:
    """Return the MCPApp instance attached to the FastMCP server, if any."""
    return getattr(mcp, '_mcp_agent_app', None)

def _set_current_identity(identity: OAuthUserIdentity | None) -> None:
    _CURRENT_IDENTITY.set(identity)

def _exit_request_context(bound_context: Optional['Context'], token: Token | None=None) -> None:
    reset_current_request_context(token)
    try:
        _set_current_identity(None)
    except Exception:
        pass
    if not isinstance(bound_context, Context):
        return
    base_context = getattr(bound_context, '_base_context_ref', None) or getattr(bound_context, '_parent_context', None)
    session = getattr(bound_context, '_scoped_upstream_session', None)
    targets: list[Any] = []
    app_ref = None
    if base_context is not None:
        targets.append(base_context)
        app_ref = getattr(base_context, 'app', None)
        if app_ref is not None:
            targets.append(app_ref)
    for target in targets:
        _clear_cached_session_refs(target, session)
    if base_context is not None and session is not None:
        previous_session = getattr(bound_context, '_previous_upstream_session', None)
        try:
            if getattr(base_context, 'upstream_session', None) is session:
                base_context.upstream_session = previous_session
        except Exception:
            pass
        if app_ref is not None:
            try:
                if getattr(app_ref, 'upstream_session', None) is session:
                    app_ref.upstream_session = previous_session
            except Exception:
                pass
    for attr in ('_base_context_ref', '_scoped_upstream_session', '_previous_upstream_session'):
        try:
            delattr(bound_context, attr)
        except Exception:
            pass

def _resolve_workflows_and_context(ctx: MCPContext, bound_context: Optional['Context']=None) -> Tuple[Dict[str, Type['Workflow']] | None, Optional['Context']]:
    """Resolve the workflows mapping and underlying app context regardless of startup mode.

    Tries lifespan ServerContext first (including compatible mocks), then attached app.
    """
    lifespan_ctx = getattr(ctx.request_context, 'lifespan_context', None)
    if lifespan_ctx is not None and hasattr(lifespan_ctx, 'workflows') and hasattr(lifespan_ctx, 'context'):
        workflows = lifespan_ctx.workflows
        context = bound_context or getattr(lifespan_ctx, 'context', None)
        return (workflows, context)
    app: MCPApp | None = _get_attached_app(ctx.fastmcp)
    if app is not None:
        return (app.workflows, bound_context or app.context)
    return (None, bound_context)

def _resolve_workflow_registry(ctx: MCPContext) -> WorkflowRegistry | None:
    """Resolve the workflow registry regardless of startup mode."""
    lifespan_ctx = getattr(ctx.request_context, 'lifespan_context', None)
    if lifespan_ctx is not None and hasattr(lifespan_ctx, 'context'):
        ctx_inner = getattr(lifespan_ctx, 'context', None)
        if ctx_inner is not None and hasattr(ctx_inner, 'workflow_registry'):
            return ctx_inner.workflow_registry
    if lifespan_ctx is not None and hasattr(lifespan_ctx, 'workflow_registry'):
        return lifespan_ctx.workflow_registry
    app: MCPApp | None = _get_attached_app(ctx.fastmcp)
    if app is not None and app.context is not None:
        return app.context.workflow_registry
    return None

def _build_run_param_tool(workflow_cls: Type['Workflow']) -> FastTool:
    """Return a FastTool for schema purposes, filtering internals like 'self', 'app_ctx', and FastMCP Context."""
    param_source = _get_param_source_function_from_workflow(workflow_cls)
    import inspect as _inspect

    def _make_filtered_schema_proxy(fn):

        def _schema_fn_proxy(*args, **kwargs):
            return None
        sig = _inspect.signature(fn)
        params = list(sig.parameters.values())
        if params and params[0].name == 'self':
            params = params[1:]
        try:
            from mcp.server.fastmcp import Context as _Ctx
        except Exception:
            _Ctx = None
        filtered_params = []
        for p in params:
            if p.name == 'app_ctx':
                continue
            if p.name in ('ctx', 'context'):
                continue
            ann = p.annotation
            if ann is not _inspect._empty and _Ctx is not None and (ann is _Ctx):
                continue
            filtered_params.append(p)
        ann_map = dict(getattr(fn, '__annotations__', {}))
        for k in ['self', 'app_ctx', 'ctx', 'context']:
            if k in ann_map:
                ann_map.pop(k, None)
        _schema_fn_proxy.__annotations__ = ann_map
        _schema_fn_proxy.__signature__ = _inspect.Signature(parameters=filtered_params, return_annotation=sig.return_annotation)
        return _schema_fn_proxy
    if param_source is getattr(workflow_cls, 'run'):
        return FastTool.from_function(_make_filtered_schema_proxy(param_source))
    return FastTool.from_function(_make_filtered_schema_proxy(param_source))

def _get_param_source_function_from_workflow(workflow_cls: Type['Workflow']):
    """Return the function to use for parameter schema for a workflow's run.

    For auto-generated workflows from @app.tool/@app.async_tool, prefer the original
    function that defined the parameters if available; fall back to the class run.
    """
    return getattr(workflow_cls, '__mcp_agent_param_source_fn__', None) or getattr(workflow_cls, 'run')

def _make_filtered_schema_proxy(fn):

    def _schema_fn_proxy(*args, **kwargs):
        return None
    sig = _inspect.signature(fn)
    params = list(sig.parameters.values())
    if params and params[0].name == 'self':
        params = params[1:]
    try:
        from mcp.server.fastmcp import Context as _Ctx
    except Exception:
        _Ctx = None
    filtered_params = []
    for p in params:
        if p.name == 'app_ctx':
            continue
        if p.name in ('ctx', 'context'):
            continue
        ann = p.annotation
        if ann is not _inspect._empty and _Ctx is not None and (ann is _Ctx):
            continue
        filtered_params.append(p)
    ann_map = dict(getattr(fn, '__annotations__', {}))
    for k in ['self', 'app_ctx', 'ctx', 'context']:
        if k in ann_map:
            ann_map.pop(k, None)
    _schema_fn_proxy.__annotations__ = ann_map
    _schema_fn_proxy.__signature__ = _inspect.Signature(parameters=filtered_params, return_annotation=sig.return_annotation)
    return _schema_fn_proxy

def _get_fallback_upstream_session() -> Any | None:
    """Best-effort fallback to the most recent upstream session captured on the app context.

            This helps when a workflow run's mapping has not been refreshed after a client reconnect.
            """
    active_ctx = None
    try:
        active_ctx = get_current_request_context()
    except Exception:
        active_ctx = None
    if active_ctx is not None:
        try:
            upstream = getattr(active_ctx, 'upstream_session', None)
            if upstream is not None:
                return upstream
        except Exception:
            pass
    try:
        app_obj: MCPApp | None = _get_attached_app(mcp_server)
    except Exception:
        app_obj = None
    if not app_obj:
        return None
    for candidate in (getattr(app_obj, '_last_known_upstream_session', None), getattr(app_obj, '_upstream_session', None)):
        if candidate is not None:
            return candidate
    base_ctx = getattr(app_obj, 'context', None)
    if base_ctx is None:
        return None
    for candidate in (getattr(base_ctx, '_last_known_upstream_session', None), getattr(base_ctx, '_upstream_session', None)):
        if candidate is not None:
            return candidate
    return None

def _install_internal_routes(mcp_server: FastMCP) -> None:

    def _get_fallback_upstream_session() -> Any | None:
        """Best-effort fallback to the most recent upstream session captured on the app context.

            This helps when a workflow run's mapping has not been refreshed after a client reconnect.
            """
        active_ctx = None
        try:
            active_ctx = get_current_request_context()
        except Exception:
            active_ctx = None
        if active_ctx is not None:
            try:
                upstream = getattr(active_ctx, 'upstream_session', None)
                if upstream is not None:
                    return upstream
            except Exception:
                pass
        try:
            app_obj: MCPApp | None = _get_attached_app(mcp_server)
        except Exception:
            app_obj = None
        if not app_obj:
            return None
        for candidate in (getattr(app_obj, '_last_known_upstream_session', None), getattr(app_obj, '_upstream_session', None)):
            if candidate is not None:
                return candidate
        base_ctx = getattr(app_obj, 'context', None)
        if base_ctx is None:
            return None
        for candidate in (getattr(base_ctx, '_last_known_upstream_session', None), getattr(base_ctx, '_upstream_session', None)):
            if candidate is not None:
                return candidate
        return None

    @mcp_server.custom_route('/internal/oauth/callback/{flow_id}', methods=['GET', 'POST'], include_in_schema=False)
    async def _oauth_callback(request: Request):
        flow_id = request.path_params.get('flow_id')
        if not flow_id:
            return JSONResponse({'error': 'missing_flow_id'}, status_code=400)
        payload: Dict[str, Any] = {}
        try:
            payload.update({k: v for k, v in request.query_params.multi_items()})
        except Exception:
            payload.update(dict(request.query_params))
        if request.method.upper() == 'POST':
            content_type = request.headers.get('content-type', '')
            try:
                if 'application/json' in content_type:
                    body_data = await request.json()
                else:
                    form = await request.form()
                    body_data = {k: v for k, v in form.multi_items()}
            except Exception:
                body_data = {}
            payload.update(body_data)
        delivered = await callback_registry.deliver(flow_id, payload)
        if not delivered:
            return JSONResponse({'error': 'unknown_flow'}, status_code=404)
        html = '<!DOCTYPE html><html><body><h3>Authorization complete.</h3><p>You may close this window and return to MCP Agent.</p></body></html>'
        return HTMLResponse(html)

    @mcp_server.custom_route('/internal/session/by-run/{execution_id}/notify', methods=['POST'], include_in_schema=False)
    async def _relay_notify(request: Request):
        body = await request.json()
        execution_id = request.path_params.get('execution_id')
        method = body.get('method')
        params = body.get('params') or {}
        mapped_context = _get_context_for_execution(execution_id) if execution_id else None
        auth_error = _check_gateway_auth(request)
        if auth_error:
            return auth_error
        idempotency_key = params.get('idempotency_key')
        if idempotency_key:
            async with _IDEMPOTENCY_KEYS_LOCK:
                seen = _IDEMPOTENCY_KEYS_SEEN.setdefault(execution_id or '', set())
                if idempotency_key in seen:
                    return JSONResponse({'ok': True, 'idempotent': True})
                seen.add(idempotency_key)
        mapped_context = _get_context_for_execution(execution_id) if execution_id else None
        latest_session = _get_fallback_upstream_session()
        tried_latest = False
        if latest_session is not None:
            tried_latest = True
            try:
                if method == 'notifications/message':
                    level = str(params.get('level', 'info'))
                    data = params.get('data')
                    logger_name = params.get('logger')
                    related_request_id = params.get('related_request_id')
                    await latest_session.send_log_message(level=level, data=data, logger=logger_name, related_request_id=related_request_id)
                elif method == 'notifications/progress':
                    progress_token = params.get('progressToken')
                    progress = params.get('progress')
                    total = params.get('total')
                    message = params.get('message')
                    await latest_session.send_progress_notification(progress_token=progress_token, progress=progress, total=total, message=message)
                else:
                    rpc = getattr(latest_session, 'rpc', None)
                    if rpc and hasattr(rpc, 'notify'):
                        await rpc.notify(method, params)
                    else:
                        return JSONResponse({'ok': False, 'error': f'unsupported method: {method}'}, status_code=400)
                try:
                    identity = _get_identity_for_execution(execution_id)
                    existing_context = _get_context_for_execution(execution_id)
                    await _register_session(run_id=execution_id, execution_id=execution_id, session=latest_session, identity=identity, context=existing_context, session_id=getattr(existing_context, 'request_session_id', None))
                except Exception:
                    pass
                return JSONResponse({'ok': True})
            except Exception as e_latest:
                logger.warning(f'[notify] latest session delivery failed for execution_id={execution_id}: {e_latest}')
        mapped_session = await _get_session(execution_id)
        mapped_context = _get_context_for_execution(execution_id) if execution_id else None
        if not mapped_session:
            logger.warning(f'[notify] session_not_available for execution_id={execution_id} (tried_latest={tried_latest})')
            return JSONResponse({'ok': False, 'error': 'session_not_available'}, status_code=503)
        ctx_token: Token | None = None
        if mapped_context is not None:
            ctx_token = set_current_request_context(mapped_context)
        try:
            if method == 'notifications/message':
                level = str(params.get('level', 'info'))
                data = params.get('data')
                logger_name = params.get('logger')
                related_request_id = params.get('related_request_id')
                await mapped_session.send_log_message(level=level, data=data, logger=logger_name, related_request_id=related_request_id)
            elif method == 'notifications/progress':
                progress_token = params.get('progressToken')
                progress = params.get('progress')
                total = params.get('total')
                message = params.get('message')
                await mapped_session.send_progress_notification(progress_token=progress_token, progress=progress, total=total, message=message)
            else:
                rpc = getattr(mapped_session, 'rpc', None)
                if rpc and hasattr(rpc, 'notify'):
                    await rpc.notify(method, params)
                else:
                    return JSONResponse({'ok': False, 'error': f'unsupported method: {method}'}, status_code=400)
            return JSONResponse({'ok': True})
        except Exception as e_mapped:
            if isinstance(method, str) and method.startswith('notifications/'):
                return JSONResponse({'ok': True, 'dropped': True})
            return JSONResponse({'ok': False, 'error': str(e_mapped)}, status_code=500)
        finally:
            reset_current_request_context(ctx_token)

    def _check_gateway_auth(request: Request) -> JSONResponse | None:
        """
            Check optional shared-secret authentication for internal endpoints.
            Returns JSONResponse with error if auth fails, None if auth passes.
            """
        gw_token = os.environ.get('MCP_GATEWAY_TOKEN')
        if not gw_token:
            return None
        bearer = request.headers.get('Authorization', '')
        bearer_token = bearer.split(' ', 1)[1] if bearer.lower().startswith('bearer ') else ''
        header_tok = request.headers.get('X-MCP-Gateway-Token', '')
        if not (secrets.compare_digest(header_tok, gw_token) or secrets.compare_digest(bearer_token, gw_token)):
            return JSONResponse({'ok': False, 'error': 'unauthorized'}, status_code=401)
        return None

    async def _handle_request_via_rpc(session, method: str, params: dict, execution_id: str, log_prefix: str='request'):
        """Handle request via generic RPC if available."""
        rpc = getattr(session, 'rpc', None)
        if rpc and hasattr(rpc, 'request'):
            result = await rpc.request(method, params)
            logger.debug(f"[{log_prefix}] delivered via session_id={id(session)} (generic '{method}')")
            return result
        return None

    async def _handle_specific_request(session: Any, method: str, params: dict, identity: OAuthUserIdentity, context: 'Context', log_prefix: str='request'):
        """Handle specific request types with structured request/response."""
        from mcp.types import CreateMessageRequest, CreateMessageRequestParams, CreateMessageResult, ElicitRequest, ElicitRequestParams, ElicitResult, ListRootsRequest, ListRootsResult, PingRequest, EmptyResult, ServerRequest
        if method == 'sampling/createMessage':
            req = ServerRequest(CreateMessageRequest(method='sampling/createMessage', params=CreateMessageRequestParams(**params)))
            callback_data = await session.send_request(request=req, result_type=CreateMessageResult)
            return callback_data.model_dump(by_alias=True, mode='json', exclude_none=True)
        elif method == 'elicitation/create':
            req = ServerRequest(ElicitRequest(method='elicitation/create', params=ElicitRequestParams(**params)))
            callback_data = await session.send_request(request=req, result_type=ElicitResult)
            return callback_data.model_dump(by_alias=True, mode='json', exclude_none=True)
        elif method == 'roots/list':
            req = ServerRequest(ListRootsRequest(method='roots/list'))
            callback_data = await session.send_request(request=req, result_type=ListRootsResult)
            return callback_data.model_dump(by_alias=True, mode='json', exclude_none=True)
        elif method == 'ping':
            req = ServerRequest(PingRequest(method='ping'))
            callback_data = await session.send_request(request=req, result_type=EmptyResult)
            return callback_data.model_dump(by_alias=True, mode='json', exclude_none=True)
        elif method == 'auth/request':
            server_name = params['server_name']
            scopes = params.get('scopes', [])
            try:
                if context and hasattr(context, 'token_manager'):
                    manager = context.token_manager
                    if manager:
                        server_config = context.server_registry.get_server_config(server_name)
                        token = await manager.get_access_token_if_present(context=context, server_name=server_name, server_config=server_config, scopes=scopes, identity=identity)
                        if token:
                            return token
            except Exception:
                pass
            record = await _perform_auth_flow(context, params, scopes, session)
            try:
                if context and hasattr(context, 'token_manager'):
                    manager = context.token_manager
                    if manager:
                        server_config = context.server_registry.get_server_config(server_name)
                        token_data = {'access_token': record.access_token, 'refresh_token': record.refresh_token, 'scopes': record.scopes, 'authorization_server': record.authorization_server, 'expires_at': record.expires_at, 'token_type': 'Bearer'}
                        await manager.store_user_token(context=context, user=identity, server_name=server_name, server_config=server_config, token_data=token_data)
            except Exception:
                pass
            return {'token_record': record.model_dump_json()}
        else:
            raise ValueError(f'unsupported method: {method}')

    async def _perform_auth_flow(context, params, scopes, session):
        from mcp.types import ElicitRequest, ElicitRequestParams, ElicitResult

        class AuthToken(BaseModel):
            confirmation: str = Field(description='Please press enter to confirm this message has been received')
        flow_id = params['flow_id']
        flow_timeout_seconds = params.get('flow_timeout_seconds')
        state = params['state']
        token_endpoint = params['token_endpoint']
        redirect_uri = params['redirect_uri']
        client_id = params['client_id']
        code_verifier = params['code_verifier']
        resource = params.get('resource')
        scope_param = params.get('scope_param')
        extra_token_params = params.get('extra_token_params', {})
        client_secret = params.get('client_secret')
        issuer_str = params.get('issuer_str')
        authorization_server_url = params.get('authorization_server_url')
        callback_future = await callback_registry.create_handle(flow_id)
        req = ElicitRequest(method='elicitation/create', params=ElicitRequestParams(message=params['message'] + '\n\n' + params['url'], requestedSchema=AuthToken.model_json_schema()))
        await session.send_request(request=req, result_type=ElicitResult)
        timeout = 300
        try:
            callback_data = await asyncio.wait_for(callback_future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise CallbackTimeoutError(f'Timed out waiting for OAuth callback after {timeout} seconds') from exc
        try:
            if callback_data and callback_data.get('url'):
                callback_data = _parse_callback_params(callback_data['url'])
                if callback_future is not None:
                    await callback_registry.discard(flow_id)
            elif callback_data and callback_data.get('code'):
                callback_data = callback_data
                if callback_future is not None:
                    await callback_registry.discard(flow_id)
            elif callback_future is not None:
                timeout = flow_timeout_seconds or 300
                try:
                    callback_data = await asyncio.wait_for(callback_future, timeout=timeout)
                except asyncio.TimeoutError as exc:
                    raise CallbackTimeoutError(f'Timed out waiting for OAuth callback after {timeout} seconds') from exc
            else:
                raise AuthorizationDeclined('Authorization request was declined by the user')
        finally:
            if callback_future is not None:
                await callback_registry.discard(flow_id)
        error = callback_data.get('error')
        if error:
            description = callback_data.get('error_description') or error
            raise OAuthFlowError(f'Authorization server returned error: {description}')
        returned_state = callback_data.get('state')
        if returned_state != state:
            raise OAuthFlowError('State mismatch detected in OAuth callback')
        authorization_code = callback_data.get('code')
        if not authorization_code:
            raise OAuthFlowError('Authorization callback did not include code')
        token_endpoint = str(token_endpoint)
        data: Dict[str, Any] = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': redirect_uri, 'client_id': client_id, 'code_verifier': code_verifier, 'resource': resource}
        if scope_param:
            data['scope'] = scope_param
        if extra_token_params:
            data.update(extra_token_params)
        auth = None
        if client_secret:
            data['client_secret'] = client_secret
        try:
            if context and hasattr(context, 'token_manager'):
                manager = context.token_manager
                if manager:
                    http_client = manager._http_client
        except Exception:
            http_client = None
        if not http_client:
            http_client = httpx.AsyncClient(timeout=30.0)
        token_response = await http_client.post(token_endpoint, data=data, auth=auth, headers={'Accept': 'application/json'})
        token_response.raise_for_status()
        try:
            callback_data = token_response.json()
        except JSONDecodeError:
            callback_data = _parse_callback_params('?' + token_response.text)
        access_token = callback_data.get('access_token')
        if not access_token:
            raise OAuthFlowError('Token endpoint response missing access_token')
        refresh_token = callback_data.get('refresh_token')
        expires_in = callback_data.get('expires_in')
        expires_at = None
        if isinstance(expires_in, (int, float)):
            expires_at = time.time() + float(expires_in)
        scope_from_payload = callback_data.get('scope')
        if isinstance(scope_from_payload, str) and scope_from_payload.strip():
            effective_scopes = tuple(scope_from_payload.split())
        else:
            effective_scopes = tuple(scopes)
        record = TokenRecord(access_token=access_token, refresh_token=refresh_token, expires_at=expires_at, scopes=effective_scopes, token_type=str(callback_data.get('token_type', 'Bearer')), resource=resource, authorization_server=issuer_str, metadata={'raw': token_response.text, 'authorization_server_url': authorization_server_url})
        return record

    async def _try_session_request(session, method: str, params: dict, execution_id: str, context: Optional['Context'], log_prefix: str='request', register_session: bool=False):
        """Try to handle a request via session, with optional registration."""
        try:
            identity = _get_identity_for_execution(execution_id)
        except Exception:
            identity = None
        try:
            result = await _handle_request_via_rpc(session, method, params, execution_id, log_prefix)
            if result is not None:
                if register_session:
                    try:
                        await _register_session(run_id=execution_id, execution_id=execution_id, session=session, identity=identity, context=context, session_id=getattr(context, 'request_session_id', None))
                    except Exception:
                        pass
                return result
            result = await _handle_specific_request(session, method, params, identity, context, log_prefix)
            if register_session:
                try:
                    await _register_session(run_id=execution_id, execution_id=execution_id, session=session, identity=identity, context=context, session_id=getattr(context, 'request_session_id', None))
                except Exception:
                    pass
            return result
        except Exception as e:
            if 'unsupported method' in str(e):
                raise
            logger.warning(f'[{log_prefix}] session delivery failed for execution_id={execution_id} method={method}: {e}')
            raise

    @mcp_server.custom_route('/internal/session/by-run/{execution_id}/request', methods=['POST'], include_in_schema=False)
    async def _relay_request(request: Request):
        app = _get_attached_app(mcp_server)
        if app and app.context:
            app_context = app.context
        else:
            app_context = None
        body = await request.json()
        execution_id = request.path_params.get('execution_id')
        method = body.get('method')
        params = body.get('params') or {}
        mapped_context = _get_context_for_execution(execution_id) if execution_id else None
        effective_context = mapped_context or app_context
        auth_error = _check_gateway_auth(request)
        if auth_error:
            return auth_error
        latest_session = _get_fallback_upstream_session()
        if latest_session is not None:
            try:
                ctx_token_latest: Token | None = None
                if effective_context is not None:
                    ctx_token_latest = set_current_request_context(effective_context)
                try:
                    result = await _try_session_request(latest_session, method, params, execution_id, effective_context, log_prefix='request', register_session=True)
                finally:
                    reset_current_request_context(ctx_token_latest)
                return JSONResponse(result)
            except Exception as e_latest:
                if 'unsupported method' not in str(e_latest):
                    logger.warning(f'[request] latest session delivery failed for execution_id={execution_id} method={method}: {e_latest}')
        mapped_context = _get_context_for_execution(execution_id) if execution_id else None
        effective_context = mapped_context or app_context
        session = await _get_session(execution_id)
        if not session:
            logger.warning(f'[request] session_not_available for execution_id={execution_id}')
            return JSONResponse({'error': 'session_not_available'}, status_code=503)
        ctx_token_mapped: Token | None = None
        if effective_context is not None:
            ctx_token_mapped = set_current_request_context(effective_context)
        try:
            result = await _try_session_request(session, method, params, execution_id, effective_context, log_prefix='request', register_session=False)
            return JSONResponse(result)
        except Exception as e:
            if 'unsupported method' in str(e):
                return JSONResponse({'error': f'unsupported method: {method}'}, status_code=400)
            try:
                logger.error(f'[request] error forwarding for execution_id={execution_id} method={method}: {e}')
            except Exception:
                pass
            return JSONResponse({'error': str(e)}, status_code=500)
        finally:
            reset_current_request_context(ctx_token_mapped)

    @mcp_server.custom_route('/internal/session/by-run/{workflow_id}/{execution_id}/async-request', methods=['POST'], include_in_schema=False)
    async def _async_relay_request(request: Request):
        body = await request.json()
        execution_id = request.path_params.get('execution_id')
        workflow_id = request.path_params.get('workflow_id')
        method = body.get('method')
        params = body.get('params') or {}
        signal_name = body.get('signal_name')
        auth_error = _check_gateway_auth(request)
        if auth_error:
            return auth_error
        try:
            logger.info(f'[async-request] incoming execution_id={execution_id} method={method}')
        except Exception:
            pass
        if method != 'sampling/createMessage' and method != 'elicitation/create':
            logger.error(f'async not supported for method {method}')
            return JSONResponse({'error': f'async not supported for method {method}'}, status_code=405)
        if not signal_name:
            return JSONResponse({'error': 'missing_signal_name'}, status_code=400)

        async def _handle_async_request_task():
            app = _get_attached_app(mcp_server)
            if app and app.context:
                app_context = app.context
            else:
                app_context = None
            mapped_context = _get_context_for_execution(execution_id) if execution_id else None
            effective_context = mapped_context or app_context
            task_token: Token | None = None
            if effective_context is not None:
                task_token = set_current_request_context(effective_context)
            try:
                result = None
                latest_session = _get_fallback_upstream_session()
                if latest_session is not None:
                    try:
                        ctx_token_latest: Token | None = None
                        if effective_context is not None:
                            ctx_token_latest = set_current_request_context(effective_context)
                        try:
                            result = await _try_session_request(latest_session, method, params, execution_id, effective_context, log_prefix='async-request', register_session=True)
                        finally:
                            reset_current_request_context(ctx_token_latest)
                    except Exception as e_latest:
                        logger.warning(f'[async-request] latest session delivery failed for execution_id={execution_id} method={method}: {e_latest}')
                if result is None:
                    session = await _get_session(execution_id)
                    if session:
                        try:
                            ctx_token_mapped: Token | None = None
                            if mapped_context is not None:
                                ctx_token_mapped = set_current_request_context(mapped_context)
                            try:
                                result = await _try_session_request(session, method, params, execution_id, mapped_context or app_context, log_prefix='async-request', register_session=False)
                            finally:
                                reset_current_request_context(ctx_token_mapped)
                        except Exception as e:
                            logger.error(f'[async-request] error forwarding for execution_id={execution_id} method={method}: {e}')
                            result = {'error': str(e)}
                    else:
                        logger.warning(f'[async-request] session_not_available for execution_id={execution_id}')
                        result = {'error': 'session_not_available'}
                try:
                    if app_context and hasattr(app_context, 'executor'):
                        executor = app_context.executor
                        if hasattr(executor, 'client'):
                            client = executor.client
                            try:
                                workflow_handle = client.get_workflow_handle(workflow_id=workflow_id, run_id=execution_id)
                                await workflow_handle.signal(signal_name, result)
                                logger.info(f'[async-request] signaled workflow {execution_id} with {method} result using signal')
                            except Exception as signal_error:
                                logger.warning(f'[async-request] failed to signal workflow {execution_id}: {signal_error}')
                except Exception as e:
                    logger.error(f'[async-request] failed to signal workflow: {e}')
            except Exception as e:
                logger.error(f'[async-request] background task error: {e}')
            finally:
                reset_current_request_context(task_token)
        asyncio.create_task(_handle_async_request_task())
        return JSONResponse({'status': 'received', 'execution_id': execution_id, 'method': method, 'signal_name': signal_name})

    @mcp_server.custom_route('/internal/workflows/log', methods=['POST'], include_in_schema=False)
    async def _internal_workflows_log(request: Request):
        body = await request.json()
        execution_id = body.get('execution_id')
        level = str(body.get('level', 'info')).lower()
        namespace = body.get('namespace') or 'mcp_agent'
        message = body.get('message') or ''
        data = body.get('data') or {}
        try:
            logger.info(f'[log] incoming execution_id={execution_id} level={level} ns={namespace}')
        except Exception:
            pass
        auth_error = _check_gateway_auth(request)
        if auth_error:
            return auth_error
        mapped_context = _get_context_for_execution(execution_id) if execution_id else None
        latest_session = _get_fallback_upstream_session()
        if latest_session is not None:
            try:
                latest_token: Token | None = None
                if mapped_context is not None:
                    latest_token = set_current_request_context(mapped_context)
                try:
                    await latest_session.send_log_message(level=level, data={'message': message, 'namespace': namespace, 'data': data}, logger=namespace)
                finally:
                    reset_current_request_context(latest_token)
                logger.debug(f'[log] delivered via latest session_id={id(latest_session)} level={level} ns={namespace}')
                try:
                    identity = _get_identity_for_execution(execution_id)
                    existing_context = _get_context_for_execution(execution_id)
                    await _register_session(run_id=execution_id, execution_id=execution_id, session=latest_session, identity=identity, context=existing_context, session_id=getattr(existing_context, 'request_session_id', None))
                    logger.info(f'[log] rebound mapping to latest session_id={id(latest_session)} for execution_id={execution_id}')
                except Exception:
                    pass
                return JSONResponse({'ok': True})
            except Exception as e_latest:
                logger.warning(f'[log] latest session delivery failed for execution_id={execution_id}: {e_latest}')
        session = await _get_session(execution_id)
        if not session:
            logger.warning(f'[log] session_not_available for execution_id={execution_id}')
            return JSONResponse({'ok': False, 'error': 'session_not_available'}, status_code=503)
        if level not in ('debug', 'info', 'warning', 'error'):
            level = 'info'
        try:
            mapped_token: Token | None = None
            if mapped_context is not None:
                mapped_token = set_current_request_context(mapped_context)
            try:
                await session.send_log_message(level=level, data={'message': message, 'namespace': namespace, 'data': data}, logger=namespace)
            finally:
                reset_current_request_context(mapped_token)
            return JSONResponse({'ok': True})
        except Exception as e:
            return JSONResponse({'ok': False, 'error': str(e)}, status_code=500)

    @mcp_server.custom_route('/internal/human/prompts', methods=['POST'], include_in_schema=False)
    async def _internal_human_prompts(request: Request):
        body = await request.json()
        execution_id = body.get('execution_id')
        prompt = body.get('prompt') or {}
        metadata = body.get('metadata') or {}
        try:
            logger.info(f'[human] incoming execution_id={execution_id} signal_name={metadata.get('signal_name', 'human_input')}')
        except Exception:
            pass
        auth_error = _check_gateway_auth(request)
        if auth_error:
            return auth_error
        app_obj = _get_attached_app(mcp_server)
        app_context = getattr(app_obj, 'context', None) if app_obj else None
        mapped_context = _get_context_for_execution(execution_id) if execution_id else None
        effective_context = mapped_context or app_context
        latest_session = _get_fallback_upstream_session()
        import uuid
        request_id = str(uuid.uuid4())
        payload = {'kind': 'human_input_request', 'request_id': request_id, 'prompt': prompt if isinstance(prompt, dict) else {'text': str(prompt)}, 'metadata': metadata}
        try:
            async with _PENDING_PROMPTS_LOCK:
                _PENDING_PROMPTS[request_id] = {'workflow_id': metadata.get('workflow_id'), 'execution_id': execution_id, 'signal_name': metadata.get('signal_name', 'human_input'), 'session_id': metadata.get('session_id')}
            if latest_session is not None:
                try:
                    latest_token: Token | None = None
                    if effective_context is not None:
                        latest_token = set_current_request_context(effective_context)
                    try:
                        await latest_session.send_log_message(level='info', data=payload, logger='mcp_agent.human')
                    finally:
                        reset_current_request_context(latest_token)
                    try:
                        identity = _get_identity_for_execution(execution_id)
                        if identity is None:
                            identity = _session_identity_from_value(metadata.get('session_id') or metadata.get('sessionId'))
                        existing_context = _get_context_for_execution(execution_id)
                        session_key = metadata.get('session_id') or metadata.get('sessionId')
                        await _register_session(run_id=execution_id, execution_id=execution_id, session=latest_session, identity=identity, context=existing_context, session_id=session_key or getattr(existing_context, 'request_session_id', None))
                        logger.info(f'[human] rebound mapping to latest session_id={id(latest_session)} for execution_id={execution_id}')
                    except Exception:
                        pass
                    return JSONResponse({'request_id': request_id})
                except Exception as e_latest:
                    logger.warning(f'[human] latest session delivery failed for execution_id={execution_id}: {e_latest}')
            mapped_context = _get_context_for_execution(execution_id) if execution_id else None
            effective_context = mapped_context or app_context
            session = await _get_session(execution_id)
            if not session:
                return JSONResponse({'error': 'session_not_available'}, status_code=503)
            mapped_token: Token | None = None
            if effective_context is not None:
                mapped_token = set_current_request_context(effective_context)
            try:
                await session.send_log_message(level='info', data=payload, logger='mcp_agent.human')
            finally:
                reset_current_request_context(mapped_token)
            return JSONResponse({'request_id': request_id})
        except Exception as e:
            return JSONResponse({'error': str(e)}, status_code=500)

def _get_context_for_execution(execution_id: str) -> 'Context' | None:
    return _RUN_CONTEXT_REGISTRY.get(execution_id)

def _check_gateway_auth(request: Request) -> JSONResponse | None:
    """
            Check optional shared-secret authentication for internal endpoints.
            Returns JSONResponse with error if auth fails, None if auth passes.
            """
    gw_token = os.environ.get('MCP_GATEWAY_TOKEN')
    if not gw_token:
        return None
    bearer = request.headers.get('Authorization', '')
    bearer_token = bearer.split(' ', 1)[1] if bearer.lower().startswith('bearer ') else ''
    header_tok = request.headers.get('X-MCP-Gateway-Token', '')
    if not (secrets.compare_digest(header_tok, gw_token) or secrets.compare_digest(bearer_token, gw_token)):
        return JSONResponse({'ok': False, 'error': 'unauthorized'}, status_code=401)
    return None

@mcp.tool(name='workflows-list', icons=[phetch])
def list_workflows(ctx: MCPContext) -> Dict[str, Dict[str, Any]]:
    """
        List all available workflow types with their detailed information.
        Returns information about each workflow type including name, description, and parameters.
        This helps in making an informed decision about which workflow to run.
        """
    bound_ctx, token = _enter_request_context(ctx)
    try:
        result: Dict[str, Dict[str, Any]] = {}
        workflows, _ = _resolve_workflows_and_context_safe(ctx, bound_ctx)
        workflows = workflows or {}
    finally:
        _exit_request_context(bound_ctx, token)
    for workflow_name, workflow_cls in workflows.items():
        run_fn_tool = _build_run_param_tool(workflow_cls)
        if getattr(workflow_cls, '__mcp_agent_sync_tool__', False):
            endpoints = [f'{workflow_name}']
        elif getattr(workflow_cls, '__mcp_agent_async_tool__', False):
            endpoints = [f'{workflow_name}']
        else:
            endpoints = [f'workflows-{workflow_name}-run']
        result[workflow_name] = {'name': workflow_name, 'description': workflow_cls.__doc__ or run_fn_tool.description, 'capabilities': ['run'], 'tool_endpoints': endpoints, 'run_parameters': run_fn_tool.parameters}
    return result

def _resolve_workflows_and_context_safe(ctx: MCPContext, bound_context: Optional['Context']=None) -> Tuple[Dict[str, Type['Workflow']] | None, Optional['Context']]:
    resolver = _resolve_workflows_and_context
    try:
        return resolver(ctx, bound_context)
    except TypeError:
        return resolver(ctx)

def create_declared_function_tools(mcp: FastMCP, server_context: ServerContext):
    """
    Register tools declared via @app.tool/@app.async_tool on the attached app.
    - @app.tool registers a synchronous tool with the same signature as the function
    - @app.async_tool registers alias tools <name>-run and <name>-get_status
      that proxy to the workflow run/status utilities.
    """
    app = _get_attached_app(mcp)
    if app is None:
        app = getattr(server_context, 'app', None)
        if app is None:
            ctx = getattr(server_context, 'context', None)
            if ctx is not None:
                app = getattr(ctx, 'app', None)
    if app is None:
        return
    declared = getattr(app, '_declared_tools', []) or []
    if not declared:
        return
    registered = _get_registered_function_tools(mcp)
    import inspect
    import asyncio
    import time
    import typing as _typing
    try:
        from mcp.server.fastmcp import Context as _Ctx
    except Exception:
        _Ctx = None

    def _annotation_is_fast_ctx(annotation) -> bool:
        if _Ctx is None or annotation is inspect._empty:
            return False
        if annotation is _Ctx:
            return True
        if inspect.isclass(annotation):
            try:
                if issubclass(annotation, _Ctx):
                    return True
            except TypeError:
                pass
        try:
            origin = _typing.get_origin(annotation)
            if origin is not None:
                return any((_annotation_is_fast_ctx(arg) for arg in _typing.get_args(annotation)))
        except Exception:
            pass
        try:
            return 'fastmcp' in str(annotation)
        except Exception:
            return False

    def _detect_context_param(signature: inspect.Signature) -> str | None:
        for param in signature.parameters.values():
            if param.name == 'app_ctx':
                continue
            if _annotation_is_fast_ctx(param.annotation):
                return param.name
            if param.annotation is inspect._empty and param.name in {'ctx', 'context'}:
                return param.name
        return None

    async def _wait_for_completion(ctx: MCPContext, run_id: str, *, workflow_id: str | None=None, timeout: float | None=None, registration_grace: float=1.0, poll_initial: float=0.05, poll_max: float=1.0):
        registry = _resolve_workflow_registry(ctx)
        if not registry:
            raise ToolError('Workflow registry not found for MCPApp Server.')
        DEFAULT_SYNC_TOOL_TIMEOUT = 120.0
        overall_timeout = timeout or DEFAULT_SYNC_TOOL_TIMEOUT
        deadline = time.monotonic() + overall_timeout

        def remaining() -> float:
            return max(0.0, deadline - time.monotonic())

        async def _await_task(task: asyncio.Task):
            return await asyncio.wait_for(task, timeout=remaining())
        try:
            wf = await registry.get_workflow(run_id, workflow_id)
            if wf is not None:
                task = getattr(wf, '_run_task', None)
                if isinstance(task, asyncio.Task):
                    return await _await_task(task)
        except Exception:
            pass
        sleep = poll_initial
        grace_deadline = time.monotonic() + registration_grace
        while time.monotonic() < grace_deadline and remaining() > 0:
            try:
                wf = await registry.get_workflow(run_id)
                if wf is not None:
                    task = getattr(wf, '_run_task', None)
                    if isinstance(task, asyncio.Task):
                        return await _await_task(task)
            except Exception:
                pass
            await asyncio.sleep(sleep)
            sleep = min(poll_max, sleep * 1.5)
        sleep = poll_initial
        while True:
            if remaining() <= 0:
                raise ToolError('Timed out waiting for workflow completion')
            status = await _workflow_status(ctx, run_id, workflow_id)
            s = str(status.get('status') or (status.get('state') or {}).get('status') or '').lower()
            if s in {'completed', 'error', 'cancelled'}:
                if s == 'completed':
                    return status.get('result')
                err = status.get('error') or status
                raise ToolError(f'Workflow ended with status={s}: {err}')
            await asyncio.sleep(sleep)
            sleep = min(poll_max, sleep * 2.0)
    for decl in declared:
        name = decl['name']
        if name in registered:
            continue
        mode = decl['mode']
        workflow_name = decl['workflow_name']
        fn = decl.get('source_fn')
        description = decl.get('description')
        structured_output = decl.get('structured_output')
        title = decl.get('title')
        annotations = decl.get('annotations')
        icons = decl.get('icons')
        meta = decl.get('meta')
        name_local = name
        wname_local = workflow_name
        if mode == 'sync' and fn is not None:
            sig = inspect.signature(fn)
            return_ann = sig.return_annotation

            def _make_wrapper(bound_wname: str):

                async def _wrapper(**kwargs):
                    ctx: MCPContext = kwargs.pop('__context__')
                    bound_ctx, token = _enter_request_context(ctx)
                    try:
                        result_ids = await _workflow_run(ctx, bound_wname, kwargs, bound_context=bound_ctx)
                        run_id = result_ids['run_id']
                        result = await _wait_for_completion(ctx, run_id)
                    finally:
                        _exit_request_context(bound_ctx, token)
                    try:
                        from mcp_agent.executor.workflow import WorkflowResult as _WFRes
                    except Exception:
                        _WFRes = None
                    if _WFRes is not None and isinstance(result, _WFRes):
                        return getattr(result, 'value', None)
                    if isinstance(result, dict) and result.get('kind') == 'workflow_result':
                        return result.get('value')
                    return result
                return _wrapper
            _wrapper = _make_wrapper(wname_local)
            ann = dict(getattr(fn, '__annotations__', {}))
            ann.pop('app_ctx', None)
            existing_ctx_param = _detect_context_param(sig)
            ctx_param_name = existing_ctx_param or 'ctx'
            if _Ctx is not None:
                ann[ctx_param_name] = _Ctx
            ann['return'] = getattr(fn, '__annotations__', {}).get('return', return_ann)
            _wrapper.__annotations__ = ann
            _wrapper.__name__ = name_local
            _wrapper.__doc__ = description or (fn.__doc__ or '')
            params = [p for p in sig.parameters.values() if p.name != 'app_ctx']
            if existing_ctx_param is None:
                ctx_param = inspect.Parameter(ctx_param_name, kind=inspect.Parameter.KEYWORD_ONLY, annotation=_Ctx)
                signature_params = params + [ctx_param]
            else:
                signature_params = params
            _wrapper.__signature__ = inspect.Signature(parameters=signature_params, return_annotation=return_ann)

            def _make_adapter(context_param_name: str, inner_wrapper):

                async def _adapter(**kw):
                    if context_param_name not in kw:
                        raise ToolError('Context not provided')
                    kw['__context__'] = kw.pop(context_param_name)
                    return await inner_wrapper(**kw)
                _adapter.__annotations__ = _wrapper.__annotations__
                _adapter.__name__ = _wrapper.__name__
                _adapter.__doc__ = _wrapper.__doc__
                _adapter.__signature__ = _wrapper.__signature__
                return _adapter
            _adapter = _make_adapter(ctx_param_name, _wrapper)
            mcp.add_tool(_adapter, name=name_local, title=title, description=description or (fn.__doc__ or ''), annotations=annotations, icons=icons, meta=meta, structured_output=structured_output)
            registered.add(name_local)
        elif mode == 'async':
            run_tool_name = f'{name_local}'
            if run_tool_name not in registered:

                def _make_async_wrapper(bound_wname: str):

                    async def _async_wrapper(**kwargs):
                        ctx: MCPContext = kwargs.pop('__context__')
                        bound_ctx, token = _enter_request_context(ctx)
                        try:
                            return await _workflow_run(ctx, bound_wname, kwargs, bound_context=bound_ctx)
                        finally:
                            _exit_request_context(bound_ctx, token)
                    return _async_wrapper
                _async_wrapper = _make_async_wrapper(wname_local)
                ann = dict(getattr(fn, '__annotations__', {}))
                ann.pop('app_ctx', None)
                try:
                    sig_async = inspect.signature(fn)
                except Exception:
                    sig_async = None
                existing_ctx_param = _detect_context_param(sig_async) if sig_async else None
                ctx_param_name = existing_ctx_param or 'ctx'
                if _Ctx is not None:
                    ann[ctx_param_name] = _Ctx
                from typing import Dict as _Dict
                ann['return'] = _Dict[str, str]
                _async_wrapper.__annotations__ = ann
                _async_wrapper.__name__ = run_tool_name
                base_desc = description or (fn.__doc__ or '')
                async_note = f"\n\nThis tool starts the '{wname_local}' workflow asynchronously and returns 'workflow_id' and 'run_id'. Use the 'workflows-get_status' tool with the returned 'workflow_id' and the returned 'run_id' to retrieve status/results."
                full_desc = (base_desc or '').strip() + async_note
                _async_wrapper.__doc__ = full_desc
                params = []
                if sig_async is not None:
                    for p in sig_async.parameters.values():
                        if p.name == 'app_ctx':
                            continue
                        if existing_ctx_param is None and (_annotation_is_fast_ctx(p.annotation) or p.name in ('ctx', 'context')):
                            continue
                        params.append(p)
                if existing_ctx_param is None:
                    if _Ctx is not None:
                        ctx_param = inspect.Parameter(ctx_param_name, kind=inspect.Parameter.KEYWORD_ONLY, annotation=_Ctx)
                    else:
                        ctx_param = inspect.Parameter(ctx_param_name, kind=inspect.Parameter.KEYWORD_ONLY)
                    signature_params = params + [ctx_param]
                else:
                    signature_params = params
                _async_wrapper.__signature__ = inspect.Signature(parameters=signature_params, return_annotation=ann.get('return'))

                def _make_async_adapter(context_param_name: str, inner_wrapper):

                    async def _adapter(**kw):
                        if context_param_name not in kw:
                            raise ToolError('Context not provided')
                        kw['__context__'] = kw.pop(context_param_name)
                        return await inner_wrapper(**kw)
                    _adapter.__annotations__ = _async_wrapper.__annotations__
                    _adapter.__name__ = _async_wrapper.__name__
                    _adapter.__doc__ = _async_wrapper.__doc__
                    _adapter.__signature__ = _async_wrapper.__signature__
                    return _adapter
                _async_adapter = _make_async_adapter(ctx_param_name, _async_wrapper)
                mcp.add_tool(_async_adapter, name=run_tool_name, title=title, description=full_desc, annotations=annotations, icons=icons, meta=meta, structured_output=False)
                registered.add(run_tool_name)
    _set_registered_function_tools(mcp, registered)

def _get_registered_function_tools(mcp: FastMCP) -> Set[str]:
    return getattr(mcp, '_registered_function_tools', set())

def _annotation_is_fast_ctx(annotation) -> bool:
    if _Ctx is None or annotation is inspect._empty:
        return False
    if annotation is _Ctx:
        return True
    if inspect.isclass(annotation):
        try:
            if issubclass(annotation, _Ctx):
                return True
        except TypeError:
            pass
    try:
        origin = _typing.get_origin(annotation)
        if origin is not None:
            return any((_annotation_is_fast_ctx(arg) for arg in _typing.get_args(annotation)))
    except Exception:
        pass
    try:
        return 'fastmcp' in str(annotation)
    except Exception:
        return False

def _detect_context_param(signature: inspect.Signature) -> str | None:
    for param in signature.parameters.values():
        if param.name == 'app_ctx':
            continue
        if _annotation_is_fast_ctx(param.annotation):
            return param.name
        if param.annotation is inspect._empty and param.name in {'ctx', 'context'}:
            return param.name
    return None

def remaining() -> float:
    return max(0.0, deadline - time.monotonic())

def _make_wrapper(bound_wname: str):

    async def _wrapper(**kwargs):
        ctx: MCPContext = kwargs.pop('__context__')
        bound_ctx, token = _enter_request_context(ctx)
        try:
            result_ids = await _workflow_run(ctx, bound_wname, kwargs, bound_context=bound_ctx)
            run_id = result_ids['run_id']
            result = await _wait_for_completion(ctx, run_id)
        finally:
            _exit_request_context(bound_ctx, token)
        try:
            from mcp_agent.executor.workflow import WorkflowResult as _WFRes
        except Exception:
            _WFRes = None
        if _WFRes is not None and isinstance(result, _WFRes):
            return getattr(result, 'value', None)
        if isinstance(result, dict) and result.get('kind') == 'workflow_result':
            return result.get('value')
        return result
    return _wrapper

def _make_adapter(context_param_name: str, inner_wrapper):

    async def _adapter(**kw):
        if context_param_name not in kw:
            raise ToolError('Context not provided')
        kw['__context__'] = kw.pop(context_param_name)
        return await inner_wrapper(**kw)
    _adapter.__annotations__ = _wrapper.__annotations__
    _adapter.__name__ = _wrapper.__name__
    _adapter.__doc__ = _wrapper.__doc__
    _adapter.__signature__ = _wrapper.__signature__
    return _adapter

def _make_async_wrapper(bound_wname: str):

    async def _async_wrapper(**kwargs):
        ctx: MCPContext = kwargs.pop('__context__')
        bound_ctx, token = _enter_request_context(ctx)
        try:
            return await _workflow_run(ctx, bound_wname, kwargs, bound_context=bound_ctx)
        finally:
            _exit_request_context(bound_ctx, token)
    return _async_wrapper

def _make_async_adapter(context_param_name: str, inner_wrapper):

    async def _adapter(**kw):
        if context_param_name not in kw:
            raise ToolError('Context not provided')
        kw['__context__'] = kw.pop(context_param_name)
        return await inner_wrapper(**kw)
    _adapter.__annotations__ = _async_wrapper.__annotations__
    _adapter.__name__ = _async_wrapper.__name__
    _adapter.__doc__ = _async_wrapper.__doc__
    _adapter.__signature__ = _async_wrapper.__signature__
    return _adapter

def _set_registered_function_tools(mcp: FastMCP, tools: Set[str]):
    setattr(mcp, '_registered_function_tools', tools)

def create_workflow_specific_tools(mcp: FastMCP, workflow_name: str, workflow_cls: Type['Workflow']):
    """Create specific tools for a given workflow."""
    param_source = _get_param_source_function_from_workflow(workflow_cls)
    import inspect as _inspect
    if param_source is getattr(workflow_cls, 'run'):

        def _schema_fn_proxy(*args, **kwargs):
            return None
        sig = _inspect.signature(param_source)
        params = list(sig.parameters.values())
        if params and params[0].name == 'self':
            params = params[1:]
        _schema_fn_proxy.__annotations__ = dict(getattr(param_source, '__annotations__', {}))
        if 'self' in _schema_fn_proxy.__annotations__:
            _schema_fn_proxy.__annotations__.pop('self', None)
        _schema_fn_proxy.__signature__ = _inspect.Signature(parameters=params, return_annotation=sig.return_annotation)
        run_fn_tool = FastTool.from_function(_schema_fn_proxy)
    else:
        run_fn_tool = FastTool.from_function(param_source)
    run_fn_tool_params = json.dumps(run_fn_tool.parameters, indent=2)

    @mcp.tool(name=f'workflows-{workflow_name}-run', icons=[phetch], description=f"\n        Run the '{workflow_name}' workflow and get a dict with workflow_id and run_id back.\n        Workflow Description: {workflow_cls.__doc__}\n\n        {run_fn_tool.description}\n\n        Args:\n            run_parameters: Dictionary of parameters for the workflow run.\n            The schema for these parameters is as follows:\n            {run_fn_tool_params}\n\n        Returns:\n            A dict with workflow_id and run_id for the started workflow run, can be passed to\n            workflows/get_status, workflows/resume, and workflows/cancel.\n        ")
    async def run(ctx: MCPContext, run_parameters: Dict[str, Any] | None=None) -> Dict[str, str]:
        bound_ctx, token = _enter_request_context(ctx)
        try:
            return await _workflow_run(ctx, workflow_name, run_parameters, bound_context=bound_ctx)
        finally:
            _exit_request_context(bound_ctx, token)

def _normalize_gateway_url(url: str | None) -> str | None:
    if not url:
        return url
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url)
        host = parsed.hostname or ''
        if host in ('0.0.0.0', '::', '[::]'):
            new_host = '127.0.0.1' if host == '0.0.0.0' else 'localhost'
            netloc = parsed.netloc.replace(host, new_host)
            parsed = parsed._replace(netloc=netloc)
            return urlunparse(parsed)
    except Exception:
        pass
    return url

def _annotation_is_fast_ctx(annotation) -> bool:
    if _Ctx is None or annotation is inspect._empty:
        return False
    if annotation is _Ctx:
        return True
    try:
        origin = _typing.get_origin(annotation)
        if origin is not None:
            return any((_annotation_is_fast_ctx(arg) for arg in _typing.get_args(annotation)))
    except Exception:
        pass
    try:
        return 'fastmcp' in str(annotation)
    except Exception:
        return False

def create_tool_adapter_signature(fn: Callable[..., Any], tool_name: str, description: Optional[str]=None) -> Callable[..., Any]:
    """
    Create a function with the transformed signature that app_server.py creates.

    This transforms the function signature by:
    1. Removing app_ctx parameter
    2. Adding ctx parameter with FastMCP Context type
    3. Preserving all other parameters and annotations

    Args:
        fn: The original function to adapt
        tool_name: Name of the tool
        description: Optional description for the tool

    Returns:
        A function with the transformed signature suitable for MCP tools

    This is used for validation in app.py to ensure the transformed
    signature can be converted to JSON schema.
    """
    sig = inspect.signature(fn)

    def _annotation_is_fast_ctx(annotation) -> bool:
        if _Ctx is None or annotation is inspect._empty:
            return False
        if annotation is _Ctx:
            return True
        try:
            origin = _typing.get_origin(annotation)
            if origin is not None:
                return any((_annotation_is_fast_ctx(arg) for arg in _typing.get_args(annotation)))
        except Exception:
            pass
        try:
            return 'fastmcp' in str(annotation)
        except Exception:
            return False
    existing_ctx_param = None
    for param in sig.parameters.values():
        if param.name == 'app_ctx':
            continue
        annotation = param.annotation
        if annotation is inspect._empty and param.name in ('ctx', 'context'):
            existing_ctx_param = param.name
            break
        if _annotation_is_fast_ctx(annotation):
            existing_ctx_param = param.name
            break
    return_ann = sig.return_annotation
    ann = dict(getattr(fn, '__annotations__', {}))
    ann.pop('app_ctx', None)
    ctx_param_name = existing_ctx_param or 'ctx'
    if _Ctx is not None:
        ann[ctx_param_name] = _Ctx
    ann['return'] = getattr(fn, '__annotations__', {}).get('return', return_ann)
    params = []
    for p in sig.parameters.values():
        if p.name == 'app_ctx':
            continue
        if existing_ctx_param is None and (p.annotation is inspect._empty and p.name in ('ctx', 'context') or _annotation_is_fast_ctx(p.annotation)):
            continue
        params.append(p)
    if existing_ctx_param is None:
        ctx_param = inspect.Parameter(ctx_param_name, kind=inspect.Parameter.KEYWORD_ONLY, annotation=_Ctx)
        signature_params = params + [ctx_param]
    else:
        signature_params = params

    async def _transformed(**kwargs):
        pass
    _transformed.__annotations__ = ann
    _transformed.__name__ = tool_name
    _transformed.__doc__ = description or (fn.__doc__ or '')
    _transformed.__signature__ = inspect.Signature(parameters=signature_params, return_annotation=return_ann)
    return _transformed

def validate_tool_schema(fn: Callable[..., Any], tool_name: str) -> None:
    """
    Validate that a function can be converted to an MCP tool.

    This creates the adapter function with transformed signature and attempts
    to generate a JSON schema from it, raising a descriptive error if it fails.

    Args:
        fn: The function to validate
        tool_name: Name of the tool for error messages

    Raises:
        ValueError: If the function cannot be converted to a valid MCP tool
    """
    from mcp.server.fastmcp.tools import Tool as FastTool
    transformed_fn = create_tool_adapter_signature(fn, tool_name)
    try:
        FastTool.from_function(transformed_fn)
    except Exception as e:
        error_msg = str(e)
        if 'PydanticInvalidForJsonSchema' in error_msg or 'Cannot generate a JsonSchema' in error_msg:
            sig = inspect.signature(fn)
            param_info = []
            for param_name, param in sig.parameters.items():
                if param_name in ('app_ctx', 'self', 'cls'):
                    continue
                if param.annotation != inspect.Parameter.empty:
                    param_info.append(f'  - {param_name}: {param.annotation}')
            params_str = '\n'.join(param_info) if param_info else '  (no typed parameters)'
            raise ValueError(f"Tool '{tool_name}' cannot be registered because its parameters or return type cannot be serialized to JSON schema.\n\nFunction parameters (after filtering):\n{params_str}\n\nError: {error_msg}\n\nCommon causes:\n  - Parameters with types containing Callable fields (e.g., Agent, MCPApp)\n  - Custom classes without proper Pydantic model definitions\n  - Complex nested types that Pydantic cannot serialize\n\nSuggestions:\n  - Replace complex objects with simple identifiers (e.g., agent_name: str instead of agent: Agent)\n  - Use primitive types (str, int, dict, list) for tool parameters\n  - Create simplified Pydantic models for complex data structures\n\nNote: The 'app_ctx' parameter is automatically filtered out and does not cause this error.") from e
        raise

