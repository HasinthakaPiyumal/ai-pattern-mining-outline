# Cluster 56

def main(log_file: str):
    """Replay MCP Agent events from a log file with progress display."""
    events = load_events(Path(log_file))
    progress = RichProgressDisplay()
    progress.start()
    try:
        for event in events:
            progress_event = convert_log_event(event)
            if progress_event:
                progress.update(progress_event)
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        progress.stop()

def create_mcp_server_for_app(app: MCPApp, **kwargs: Any) -> FastMCP:
    """
    Create an MCP server for a given MCPApp instance.

    Args:
        app: The MCPApp instance to create a server for
        kwargs: Optional FastMCP settings to configure the server.

    Returns:
        A configured FastMCP server instance
    """
    auth_settings_config = None
    try:
        if app.context and app.context.config:
            auth_settings_config = app.context.config.authorization
    except Exception:
        auth_settings_config = None
    effective_auth_settings: AuthSettings | None = None
    token_verifier: MCPAgentTokenVerifier | None = None
    owns_token_verifier = False
    if auth_settings_config and auth_settings_config.enabled:
        try:
            effective_auth_settings = AuthSettings(issuer_url=auth_settings_config.issuer_url, resource_server_url=auth_settings_config.resource_server_url, service_documentation_url=auth_settings_config.service_documentation_url, required_scopes=auth_settings_config.required_scopes or None)
            token_verifier = MCPAgentTokenVerifier(auth_settings_config)
        except Exception as exc:
            logger.error('Failed to configure authorization server integration', exc_info=True, data={'error': str(exc)})
            effective_auth_settings = None
            token_verifier = None

    @asynccontextmanager
    async def app_specific_lifespan(mcp: FastMCP) -> AsyncIterator[ServerContext]:
        """Initialize and manage MCPApp lifecycle."""
        await app.initialize()
        server_context = ServerContext(mcp=mcp, context=app.context)
        create_workflow_tools(mcp, server_context)
        create_declared_function_tools(mcp, server_context)
        try:
            yield server_context
        finally:
            if owns_token_verifier and token_verifier is not None:
                try:
                    await token_verifier.aclose()
                except Exception:
                    pass

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
    if app.mcp:
        mcp = app.mcp
        setattr(mcp, '_mcp_agent_app', app)
        if not hasattr(mcp, '_mcp_agent_server_context'):
            server_context = ServerContext(mcp=mcp, context=app.context)
            setattr(mcp, '_mcp_agent_server_context', server_context)
        else:
            server_context = getattr(mcp, '_mcp_agent_server_context')
        create_workflow_tools(mcp, server_context)
        create_declared_function_tools(mcp, server_context)
        try:
            _install_internal_routes(mcp)
        except Exception:
            pass
    else:
        if 'icons' not in kwargs and app._icons:
            kwargs['icons'] = app._icons
        if 'auth' not in kwargs and effective_auth_settings is not None:
            kwargs['auth'] = effective_auth_settings
        if 'token_verifier' not in kwargs and token_verifier is not None:
            kwargs['token_verifier'] = token_verifier
            owns_token_verifier = True
        mcp = FastMCP(name=app.name or 'mcp_agent_server', instructions=f'MCP server exposing {app.name} workflows and agents as tools. Description: {app.description}', lifespan=app_specific_lifespan, **kwargs)
        app.mcp = mcp
        setattr(mcp, '_mcp_agent_app', app)
        try:
            _install_internal_routes(mcp)
        except Exception:
            pass
    lowlevel_server = getattr(mcp, '_mcp_server', None)
    try:
        if lowlevel_server is not None:

            @lowlevel_server.set_logging_level()
            async def _set_level(level: str) -> None:
                ctx_obj: MCPContext | None = None
                try:
                    ctx_obj = mcp.get_context() if hasattr(mcp, 'get_context') else None
                except Exception:
                    ctx_obj = None
                bound_ctx: Context | None = None
                token: Token | None = None
                if ctx_obj is not None:
                    try:
                        bound_ctx, token = _enter_request_context(ctx_obj)
                    except Exception:
                        bound_ctx, token = (None, None)
                try:
                    session_id = getattr(bound_ctx, 'request_session_id', None) if bound_ctx is not None else None
                    if session_id:
                        LoggingConfig.set_session_min_level(session_id, level)
                    else:
                        LoggingConfig.set_min_level(level)
                except Exception:
                    pass
                finally:
                    _exit_request_context(bound_ctx, token)
    except Exception:
        pass

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

    @mcp.tool(name='workflows-runs-list', icons=[phetch])
    async def list_workflow_runs(ctx: MCPContext, limit: int=100, page_size: int | None=100, next_page_token: str | None=None) -> List[Dict[str, Any]] | WorkflowRunsPage:
        """
        List all workflow instances (runs) with their detailed status information.

        This returns information about actual workflow instances (runs), not workflow types.
        For each running workflow, returns its ID, name, current state, and available operations.
        This helps in identifying and managing active workflow instances.


        Args:
            limit: Maximum number of runs to return. Default: 100.
            page_size: Page size for paginated backends. Default: 100.
            next_page_token: Optional Base64-encoded token for pagination resume. Only provide if you received a next_page_token from a previous call.

        Returns:
            A list of workflow run status dictionaries with detailed workflow information.
        """
        bound_ctx, token = _enter_request_context(ctx)
        try:
            server_context = getattr(ctx.request_context, 'lifespan_context', None) or _get_attached_server_context(ctx.fastmcp)
            if server_context is None or not hasattr(server_context, 'workflow_registry'):
                raise ToolError('Server context not available for MCPApp Server.')
            token_bytes = None
            if next_page_token:
                try:
                    import base64 as _b64
                    token_bytes = _b64.b64decode(next_page_token)
                except Exception:
                    token_bytes = None
            workflow_statuses = await server_context.workflow_registry.list_workflow_statuses(query=None, limit=limit, page_size=page_size, next_page_token=token_bytes)
            return workflow_statuses
        finally:
            _exit_request_context(bound_ctx, token)

    @mcp.tool(name='workflows-run', icons=[phetch])
    async def run_workflow(ctx: MCPContext, workflow_name: str, run_parameters: Dict[str, Any] | None=None, **kwargs: Any) -> Dict[str, str]:
        """
        Run a workflow with the given name.

        Args:
            workflow_name: The name of the workflow to run.
            run_parameters: Arguments to pass to the workflow run.
                workflows/list method will return the run_parameters schema for each workflow.
            kwargs: Ignore, for internal use only.

        Returns:
            A dict with workflow_id and run_id for the started workflow run, can be passed to
            workflows/get_status, workflows/resume, and workflows/cancel.
        """
        bound_ctx, token = _enter_request_context(ctx)
        try:
            return await _workflow_run(ctx, workflow_name, run_parameters, bound_context=bound_ctx, **kwargs)
        finally:
            _exit_request_context(bound_ctx, token)

    @mcp.tool(name='workflows-get_status', icons=[phetch])
    async def get_workflow_status(ctx: MCPContext, run_id: str | None=None, workflow_id: str | None=None) -> Dict[str, Any]:
        """
        Get the status of a running workflow.

        Provides detailed information about a workflow instance including its current state,
        whether it's running or completed, and any results or errors encountered.

        Args:
            run_id: Optional run ID of the workflow to check.
                If omitted, the server will use the latest run for the workflow_id provided.
                Received from workflows/run or workflows/runs/list.
            workflow_id: Optional workflow identifier (usually the tool/workflow name).
                If omitted, the server will infer it from the run metadata when possible.
                Received from workflows/run or workflows/runs/list.

        Returns:
            A dictionary with comprehensive information about the workflow status.
        """
        bound_ctx, token = _enter_request_context(ctx)
        try:
            try:
                sess = getattr(ctx, 'session', None)
                if sess and run_id:
                    exec_id = _RUN_EXECUTION_ID_REGISTRY.get(run_id, run_id)
                    app_obj = _get_attached_app(ctx.fastmcp)
                    app_ctx = getattr(app_obj, 'context', None) if app_obj else None
                    identity = _resolve_identity_for_request(ctx, app_ctx, exec_id)
                    await _register_session(run_id=run_id, execution_id=exec_id, session=sess, identity=identity, context=bound_ctx, session_id=getattr(bound_ctx, 'request_session_id', None))
            except Exception:
                pass
            return await _workflow_status(ctx, run_id=run_id, workflow_id=workflow_id, bound_context=bound_ctx)
        finally:
            _exit_request_context(bound_ctx, token)

    @mcp.tool(name='workflows-resume', icons=[phetch])
    async def resume_workflow(ctx: MCPContext, run_id: str | None=None, workflow_id: str | None=None, signal_name: str | None='resume', payload: Dict[str, Any] | None=None) -> bool:
        """
        Resume a paused workflow.

        Args:
            run_id: The ID of the workflow to resume,
                received from workflows/run or workflows/runs/list.
                If not specified, the latest run for the workflow_id will be used.
            workflow_id: The ID of the workflow to resume,
                received from workflows/run or workflows/runs/list.
            signal_name: Optional name of the signal to send to resume the workflow.
                This will default to "resume", but can be a custom signal name
                if the workflow was paused on a specific signal.
            payload: Optional payload to provide the workflow upon resumption.
                For example, if a workflow is waiting for human input,
                this can be the human input.

        Returns:
            True if the workflow was resumed, False otherwise.
        """
        bound_ctx, token = _enter_request_context(ctx)
        try:
            try:
                sess = getattr(ctx, 'session', None)
                if sess and run_id:
                    exec_id = _RUN_EXECUTION_ID_REGISTRY.get(run_id, run_id)
                    app_obj = _get_attached_app(ctx.fastmcp)
                    app_ctx = getattr(app_obj, 'context', None) if app_obj else None
                    identity = _resolve_identity_for_request(ctx, app_ctx, exec_id)
                    await _register_session(run_id=run_id, execution_id=exec_id, session=sess, identity=identity, context=bound_ctx, session_id=getattr(bound_ctx, 'request_session_id', None))
            except Exception:
                pass
            if run_id is None and workflow_id is None:
                raise ToolError('Either run_id or workflow_id must be provided.')
            workflow_registry: WorkflowRegistry | None = _resolve_workflow_registry(ctx)
            if not workflow_registry:
                raise ToolError('Workflow registry not found for MCPApp Server.')
            logger.info(f"Resuming workflow ID {workflow_id or 'unknown'}, run ID {run_id or 'unknown'} with signal '{signal_name}' and payload '{payload}'")
            result = await workflow_registry.resume_workflow(run_id=run_id, workflow_id=workflow_id, signal_name=signal_name, payload=payload)
            if result:
                logger.debug(f"Signaled workflow ID {workflow_id or 'unknown'}, run ID {run_id or 'unknown'} with signal '{signal_name}' and payload '{payload}'")
            else:
                logger.error(f"Failed to signal workflow ID {workflow_id or 'unknown'}, run ID {run_id or 'unknown'} with signal '{signal_name}' and payload '{payload}'")
            return result
        finally:
            _exit_request_context(bound_ctx, token)

    @mcp.tool(name='workflows-cancel', icons=[phetch])
    async def cancel_workflow(ctx: MCPContext, run_id: str | None=None, workflow_id: str | None=None) -> bool:
        """
        Cancel a running workflow.

        Args:
            run_id: The ID of the workflow instance to cancel,
                received from workflows/run or workflows/runs/list.
                If not provided, will attempt to cancel the latest run for the
                provided workflow ID.
            workflow_id: The ID of the workflow to cancel,
                received from workflows/run or workflows/runs/list.

        Returns:
            True if the workflow was cancelled, False otherwise.
        """
        bound_ctx, token = _enter_request_context(ctx)
        try:
            try:
                sess = getattr(ctx, 'session', None)
                if sess and run_id:
                    exec_id = _RUN_EXECUTION_ID_REGISTRY.get(run_id, run_id)
                    app_obj = _get_attached_app(ctx.fastmcp)
                    app_ctx = getattr(app_obj, 'context', None) if app_obj else None
                    identity = _resolve_identity_for_request(ctx, app_ctx, exec_id)
                    await _register_session(run_id=run_id, execution_id=exec_id, session=sess, identity=identity, context=bound_ctx, session_id=getattr(bound_ctx, 'request_session_id', None))
            except Exception:
                pass
            if run_id is None and workflow_id is None:
                raise ToolError('Either run_id or workflow_id must be provided.')
            workflow_registry: WorkflowRegistry | None = _resolve_workflow_registry(ctx)
            if not workflow_registry:
                raise ToolError('Workflow registry not found for MCPApp Server.')
            logger.info(f'Cancelling workflow ID {workflow_id or 'unknown'}, run ID {run_id or 'unknown'}')
            result = await workflow_registry.cancel_workflow(run_id=run_id, workflow_id=workflow_id)
            if result:
                logger.debug(f'Cancelled workflow ID {workflow_id or 'unknown'}, run ID {run_id or 'unknown'}')
            else:
                logger.error(f'Failed to cancel workflow {workflow_id or 'unknown'} with ID {run_id or 'unknown'}')
            return result
        finally:
            _exit_request_context(bound_ctx, token)

    @mcp.tool(name='workflows-store-credentials')
    async def workflow_store_credentials(ctx: MCPContext, workflow_name: str, tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Store OAuth tokens for a workflow to use with MCP servers.

        Persisting tokens ahead of time lets workflows authenticate with external services
        without needing an interactive OAuth flow at execution time.

        Args:
            workflow_name: The name of the workflow that will use these tokens.
            tokens: List of OAuth token objects, each containing:
                - access_token (str): The OAuth access token
                - refresh_token (str, optional): The OAuth refresh token
                - server_name (str): Name/identifier of the MCP server
                - scopes (List[str], optional): List of OAuth scopes
                - expires_at (float, optional): Token expiration timestamp
                - authorization_server (str, optional): Authorization server URL

        Returns:
            Dictionary with success status and count of stored tokens.
        """
        bound_ctx, token = _enter_request_context(ctx)
        try:
            workflows_dict, app_context = _resolve_workflows_and_context_safe(ctx, bound_ctx)
            if not workflows_dict or not app_context:
                raise ToolError('Server context not available for MCPApp Server.')
            if workflow_name not in workflows_dict:
                raise ToolError(f"Workflow '{workflow_name}' not found.")
            if not app_context.token_manager:
                raise ToolError('OAuth token manager not available.')
            identity = _resolve_identity_for_request(ctx, app_context)
            if not tokens:
                raise ToolError('At least one token must be provided.')
            stored_count = 0
            errors = []
            for i, token_data in enumerate(tokens):
                try:
                    if not isinstance(token_data, dict):
                        errors.append(f'Token {i}: must be a dictionary')
                        continue
                    access_token = token_data.get('access_token')
                    server_name = token_data.get('server_name')
                    if not access_token:
                        errors.append(f"Token {i}: missing required 'access_token' field")
                        continue
                    if not server_name:
                        errors.append(f"Token {i}: missing required 'server_name' field")
                        continue
                    server_config = app_context.server_registry.registry.get(server_name)
                    if not server_config:
                        errors.append(f"Token {i}: server '{server_name}' not recognized")
                        continue
                    await app_context.token_manager.store_user_token(context=app_context, user=identity, server_name=server_name, server_config=server_config, token_data=token_data, workflow_name=workflow_name)
                    stored_count += 1
                except Exception as e:
                    errors.append(f'Token {i}: {str(e)}')
                    logger.error(f"Error storing token {i} for workflow '{workflow_name}': {e}")
            if errors and stored_count == 0:
                raise ToolError(f'Failed to store any tokens. Errors: {'; '.join(errors)}')
            result = {'success': True, 'workflow_name': workflow_name, 'stored_tokens': stored_count, 'total_tokens': len(tokens)}
            if errors:
                result['errors'] = errors
                result['partial_success'] = True
            logger.info(f"Pre-authorization completed for workflow '{workflow_name}': {stored_count}/{len(tokens)} tokens stored")
            return result
        except Exception as e:
            logger.error(f"Error in workflow pre-authorization for '{workflow_name}': {e}")
            raise ToolError(f'Failed to store tokens: {str(e)}')
        finally:
            _exit_request_context(bound_ctx, token)
    return mcp

def _get_attached_server_context(mcp: FastMCP) -> ServerContext | None:
    """Return the ServerContext attached to the FastMCP server, if any."""
    return getattr(mcp, '_mcp_agent_server_context', None)

@app.command()
def test(script: Optional[str]=typer.Option(None, '--script', '-s', help='Script to test'), timeout: float=typer.Option(5.0, '--timeout', '-t', help='Test timeout')) -> None:
    """Test if the server can be loaded and initialized."""
    script_path = detect_default_script(Path(script) if script else None)
    if not script_path.exists():
        console.print(f'[red]Script not found: {script_path}[/red]')
        console.print('\n[dim]Create a main.py (preferred) or agent.py file, or specify --script[/dim]')
        raise typer.Exit(1)
    console.print(f'\n[bold]Testing server: {script_path}[/bold]\n')
    with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'), console=console) as progress:

        async def _test():
            task = progress.add_task('Loading app...', total=None)
            try:
                app_obj = load_user_app(script_path)
                progress.update(task, description='[green]✅ App loaded[/green]')
            except Exception as e:
                progress.update(task, description=f'[red]❌ Failed to load: {e}[/red]')
                raise typer.Exit(1)
            task = progress.add_task('Initializing app...', total=None)
            try:
                await asyncio.wait_for(app_obj.initialize(), timeout=timeout)
                progress.update(task, description='[green]✅ App initialized[/green]')
            except asyncio.TimeoutError:
                progress.update(task, description=f'[red]❌ Initialization timeout ({timeout}s)[/red]')
                raise typer.Exit(1)
            except Exception as e:
                progress.update(task, description=f'[red]❌ Failed to initialize: {e}[/red]')
                raise typer.Exit(1)
            task = progress.add_task('Creating MCP server...', total=None)
            try:
                create_mcp_server_for_app(app_obj)
                progress.update(task, description='[green]✅ Server created[/green]')
            except Exception as e:
                progress.update(task, description=f'[red]❌ Failed to create server: {e}[/red]')
                raise typer.Exit(1)
            components = []
            if hasattr(app_obj, 'workflows') and app_obj.workflows:
                components.append(f'{len(app_obj.workflows)} workflows')
            if hasattr(app_obj, 'agents') and app_obj.agents:
                components.append(f'{len(app_obj.agents)} agents')
            return (app_obj, components)
        try:
            app_obj, components = asyncio.run(_test())
            console.print('\n[green bold]✅ Server test passed![/green bold]\n')
            summary = Table(show_header=False, box=None)
            summary.add_column('Property', style='cyan')
            summary.add_column('Value')
            summary.add_row('App Name', app_obj.name)
            if hasattr(app_obj, 'description') and app_obj.description:
                summary.add_row('Description', app_obj.description)
            if components:
                summary.add_row('Components', ', '.join(components))
            console.print(Panel(summary, title='[bold]Server Summary[/bold]', border_style='green'))
            console.print('\n[dim]Server is ready to run with:[/dim]')
            console.print(f'  [cyan]mcp-agent dev serve --script {script_path}[/cyan]')
        except Exception:
            console.print('\n[red bold]❌ Server test failed[/red bold]')
            raise typer.Exit(1)

def collect_model_usage(node: TokenNode):
    """Recursively collect model usage from a node tree"""
    if node.usage.model_name:
        model_name = node.usage.model_name
        provider = node.usage.model_info.provider if node.usage.model_info else None
        model_key = (model_name, provider)
        if model_key not in model_usage:
            model_usage[model_key] = {'model_name': model_name, 'provider': provider, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        model_usage[model_key]['input_tokens'] += node.usage.input_tokens
        model_usage[model_key]['output_tokens'] += node.usage.output_tokens
        model_usage[model_key]['total_tokens'] += node.usage.total_tokens
    for child in node.children:
        collect_model_usage(child)

def _tool_meta() -> Dict[str, Any]:
    return {'openai.com/widget': _embedded_widget_resource().model_dump(mode='json'), 'openai/outputTemplate': WIDGET.template_uri, 'openai/toolInvocation/invoking': WIDGET.invoking, 'openai/toolInvocation/invoked': WIDGET.invoked, 'openai/widgetAccessible': True, 'openai/resultCanProduceWidget': True}

def _embedded_widget_resource() -> types.EmbeddedResource:
    return types.EmbeddedResource(type='resource', resource=types.TextResourceContents(uri=WIDGET.template_uri, mimeType=MIME_TYPE, text=WIDGET.html, title=WIDGET.title))

@mcp.resource(uri=WIDGET.template_uri, title=WIDGET.title, description=_resource_description(), mime_type=MIME_TYPE)
def get_widget_html() -> str:
    """Provide the HTML template for the coin flip widget."""
    return WIDGET.html

def _resource_description() -> str:
    return 'Coin flip widget markup'

def load_agent_specs_from_file(path: str, context=None) -> List[AgentSpec]:
    ext = os.path.splitext(path)[1].lower()
    fmt = None
    if ext in ('.yaml', '.yml'):
        fmt = 'yaml'
    elif ext == '.json':
        fmt = 'json'
    elif ext in ('.md', '.markdown'):
        fmt = 'md'
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return load_agent_specs_from_text(text, fmt=fmt, context=context)

def load_agent_specs_from_dir(path: str, pattern: str='**/*.*', context=None) -> List[AgentSpec]:
    """Load AgentSpec list by scanning a directory for yaml/json/md files."""
    results: List[AgentSpec] = []
    for fp in glob(os.path.join(path, pattern), recursive=True):
        if os.path.isdir(fp):
            continue
        ext = os.path.splitext(fp)[1].lower()
        if ext not in ('.yaml', '.yml', '.json', '.md', '.markdown'):
            continue
        try:
            results.extend(load_agent_specs_from_file(fp, context=context))
        except Exception:
            continue
    return results

def collect_model_usage(node: TokenNode):
    """Recursively collect model usage from a node tree"""
    if node.usage.model_name:
        model_name = node.usage.model_name
        provider = node.usage.model_info.provider if node.usage.model_info else None
        model_key = (model_name, provider)
        if model_key not in model_usage:
            model_usage[model_key] = {'model_name': model_name, 'provider': provider, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        model_usage[model_key]['input_tokens'] += node.usage.input_tokens
        model_usage[model_key]['output_tokens'] += node.usage.output_tokens
        model_usage[model_key]['total_tokens'] += node.usage.total_tokens
    for child in node.children:
        collect_model_usage(child)

def _load_settings():
    signature = inspect.signature(get_settings)
    kwargs = {}
    config_path = Path(__file__).with_name('mcp_agent.config.yaml')
    if 'config_path' in signature.parameters:
        kwargs['config_path'] = str(config_path)
    if 'set_global' in signature.parameters:
        kwargs['set_global'] = False
    return get_settings(**kwargs)

def apply_tool_filter(llm_instance, tool_filter: Optional[ToolFilter]):
    """
    Apply a tool filter to an LLM instance without modifying its source code.

    This function wraps the LLM's generate methods to filter tools during execution.

    Args:
        llm_instance: An instance of AugmentedLLM (e.g., OpenAIAugmentedLLM)
        tool_filter: The ToolFilter to apply, or None to remove filtering

    Returns:
        The same LLM instance with filtering applied

    Example:
        llm = await agent.attach_llm(OpenAIAugmentedLLM)
        filter = ToolFilter(allowed=["read_file", "list_directory"])
        apply_tool_filter(llm, filter)
    """
    if not hasattr(llm_instance, '_original_generate'):
        llm_instance._original_generate = llm_instance.generate
    if not hasattr(llm_instance, '_filter_lock'):
        llm_instance._filter_lock = asyncio.Lock()
    if tool_filter is None:
        if hasattr(llm_instance, '_original_generate'):
            logger.info('Tool filter removed from LLM instance')
            llm_instance.generate = llm_instance._original_generate
        return llm_instance
    filter_info = []
    if tool_filter.allowed_global:
        filter_info.append(f'allowed: {list(tool_filter.allowed_global)}')
    if tool_filter.excluded_global:
        filter_info.append(f'excluded: {list(tool_filter.excluded_global)}')
    if tool_filter.server_filters:
        filter_info.append(f'server-specific: {tool_filter.server_filters}')
    if tool_filter.custom_filter:
        filter_info.append('custom filter function')
    logger.info(f'Tool filter applied to LLM instance with: {(', '.join(filter_info) if filter_info else 'no constraints')}')

    async def filtered_generate(message, request_params=None):
        async with llm_instance._filter_lock:
            original_list_tools = llm_instance.agent.list_tools

            async def filtered_list_tools(server_name=None):
                result = await original_list_tools(server_name)
                if tool_filter:
                    result.tools = tool_filter.filter_tools(result.tools)
                return result
            llm_instance.agent.list_tools = filtered_list_tools
            try:
                return await llm_instance._original_generate(message, request_params)
            except Exception as e:
                logger.error(f'Error during filtered generate: {e}')
                raise
            finally:
                llm_instance.agent.list_tools = original_list_tools
    llm_instance.generate = filtered_generate
    return llm_instance

def main():
    sse_server_transport: SseServerTransport = SseServerTransport('/messages/')
    server: Server = Server('test-service')

    @server.list_tools()
    @telemetry.traced(kind=trace.SpanKind.SERVER)
    async def handle_list_tools() -> list[Tool]:
        return [Tool(name='get-magic-number', description='Returns a magic number', inputSchema={'type': 'object', 'properties': {'original_number': {'type': 'number'}}})]

    @server.call_tool()
    @telemetry.traced(kind=trace.SpanKind.SERVER)
    async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent | ImageContent | EmbeddedResource]:
        span = trace.get_current_span()
        res = str(get_magic_number(arguments.get('original_number', 0)))
        span.set_attribute(GEN_AI_TOOL_NAME, name)
        span.set_attribute('result', res)
        if arguments:
            record_attributes(span, arguments, 'arguments')
        return [TextContent(type='text', text=res)]
    initialization_options: InitializationOptions = InitializationOptions(server_name=server.name, server_version='1.0.0', capabilities=server.get_capabilities(notification_options=NotificationOptions(), experimental_capabilities={}))

    async def handle_sse(request):
        async with sse_server_transport.connect_sse(scope=request.scope, receive=request.receive, send=request._send) as streams:
            await server.run(read_stream=streams[0], write_stream=streams[1], initialization_options=initialization_options)
    starlette_app: Starlette = Starlette(routes=[Route('/sse', endpoint=handle_sse), Mount('/messages/', app=sse_server_transport.handle_post_message)])
    uvicorn.run(starlette_app, host='0.0.0.0', port=8000, log_level=-10000)

def get_magic_number(original_number: int=0) -> int:
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span('some_tool_function') as span:
        span.set_attribute('example.attribute', 'value')
        result = 42 + original_number
        span.set_attribute('result', result)
        return result

def _configure_server_otel():
    """
    Configure OpenTelemetry for the MCP server.
    This function sets up the global textmap propagator and initializes the tracer provider.
    """
    MCPInstrumentor().instrument()

def _tool_meta() -> Dict[str, Any]:
    return {'openai.com/widget': _embedded_widget_resource().model_dump(mode='json'), 'openai/outputTemplate': WIDGET.template_uri, 'openai/toolInvocation/invoking': WIDGET.invoking, 'openai/toolInvocation/invoked': WIDGET.invoked, 'openai/widgetAccessible': True, 'openai/resultCanProduceWidget': True}

@mcp.resource(uri=WIDGET.template_uri, title=WIDGET.title, description=_resource_description(), mime_type=MIME_TYPE)
def get_widget_html() -> str:
    """Provide the HTML template for the coin flip widget."""
    return WIDGET.html

@app.cell
def _():
    import marimo as mo
    return (mo,)

def format_list_tools_result(list_tools_result: ListToolsResult):
    res = ''
    for tool in list_tools_result.tools:
        res += f'- **{tool.name}**: {tool.description}\n\n'
    return res

def detect_platform(request: str) -> str:
    """
    Detect the intended platform from the user's request.
    Defaults to 'linkedin' if no platform is found.
    """
    request_lower = request.lower()
    platforms = ['twitter', 'linkedin', 'instagram', 'facebook', 'email', 'reddit']
    for platform in platforms:
        if platform in request_lower:
            return platform
    return 'linkedin'

def load_company_config() -> dict:
    """
    Load the company configuration from CONFIG_FILE.
    Returns a default config if the file is not found.
    """
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f'⚠️ {CONFIG_FILE} not found. Using default config...')
        return {'company': {'name': 'Your Company'}, 'platforms': {'linkedin': {'max_word_count': 150}}}

def initialize_collection():
    """Create and add data to collection."""
    client = QdrantClient('http://localhost:6333')
    client.set_model('BAAI/bge-small-en-v1.5')
    if client.collection_exists('my_collection'):
        return
    client.add(collection_name='my_collection', documents=SAMPLE_TEXTS)

def test_load_agents_from_dir(tmp_path):
    (tmp_path / 'agents.yaml').write_text(dedent('\n            agents:\n              - name: one\n                servers: [filesystem]\n              - name: two\n                servers: [fetch]\n            '), encoding='utf-8')
    (tmp_path / 'agent.json').write_text('{"agent": {"name": "json-agent", "servers": ["fetch"]}}', encoding='utf-8')
    specs = load_agent_specs_from_dir(str(tmp_path))
    names = {s.name for s in specs}
    assert {'one', 'two', 'json-agent'}.issubset(names)

def test_load_agent_specs_from_file_markdown(tmp_path):
    md_path = tmp_path / 'agent.md'
    md_path.write_text(dedent('\n            ---\n            name: data-scientist\n            description: Data analysis expert\n            tools: Bash, Read, Write\n            ---\n\n            You are a data scientist specializing in SQL and BigQuery analysis.\n            '), encoding='utf-8')
    specs = load_agent_specs_from_file(str(md_path))
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == 'data-scientist'
    assert spec.server_names == ['Bash', 'Read', 'Write']
    assert 'data scientist' in (spec.instruction or '')

