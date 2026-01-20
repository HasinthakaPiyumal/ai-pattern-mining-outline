# Cluster 4

def log_orchestrator_activity(orchestrator_id: str, activity: str, details: dict=None):
    """
    Log orchestrator activities for debugging.

    Args:
        orchestrator_id: ID of the orchestrator
        activity: Description of the activity
        details: Additional details as dictionary
    """
    func_name, line_num = _get_caller_info()
    log = logger.bind(name=f'orchestrator.{orchestrator_id}:{func_name}:{line_num}')
    if _DEBUG_MODE:
        log.opt(colors=True).debug('<magenta>🎯 {}: {}</magenta>', activity, details or {})

def _get_caller_info():
    """
    Get the caller's line number and function name from the stack frame.

    Returns:
        Tuple of (function_name, line_number where the logging function was called)
    """
    frame = inspect.currentframe()
    if frame and frame.f_back and frame.f_back.f_back:
        caller_frame = frame.f_back.f_back
        function_name = caller_frame.f_code.co_name
        line_number = caller_frame.f_lineno
        return (function_name, line_number)
    return ('unknown', 0)

def log_orchestrator_agent_message(agent_id: str, direction: str, message: dict, backend_name: str=None):
    """
    Log orchestrator-to-agent messages for debugging.

    Args:
        agent_id: ID of the agent
        direction: "SEND" or "RECV"
        message: Message content as dictionary
        backend_name: Optional name of the backend provider
    """
    func_name, line_num = _get_caller_info()
    if backend_name:
        log_name = f'orchestrator→{agent_id}.{backend_name}:{func_name}:{line_num}'
        log = logger.bind(name=log_name)
    else:
        log_name = f'orchestrator→{agent_id}:{func_name}:{line_num}'
        log = logger.bind(name=log_name)
    if _DEBUG_MODE:
        if direction == 'SEND':
            log.opt(colors=True).debug('<magenta>🎯📤 [{}] Orchestrator sending to agent: {}</magenta>', log_name, _format_message(message))
        elif direction == 'RECV':
            log.opt(colors=True).debug('<magenta>🎯📥 [{}] Orchestrator received from agent: {}</magenta>', log_name, _format_message(message))
        else:
            log.opt(colors=True).debug('<magenta>🎯📨 [{}] {}: {}</magenta>', log_name, direction, _format_message(message))

def log_mcp_activity(backend_name: str, message: str, details: dict=None, agent_id: str=None):
    """
    Log MCP (Model Context Protocol) activities at INFO level.

    Args:
        backend_name: Name of the backend (e.g., "claude", "openai")
        message: Description of the MCP activity
        details: Additional details as dictionary
        agent_id: Optional ID of the agent using this backend
    """
    func_name, line_num = _get_caller_info()
    if agent_id:
        log_name = f'{agent_id}.{backend_name}'
        log = logger.bind(name=f'{log_name}:{func_name}:{line_num}')
    else:
        log_name = backend_name
        log = logger.bind(name=f'backend.{backend_name}:{func_name}:{line_num}')
    log.info('MCP: {} - {}', message, details or {})

def log_coordination_step(step: str, details: dict=None):
    """
    Log coordination workflow steps.

    Args:
        step: Description of the coordination step
        details: Additional details as dictionary
    """
    log = logger.bind(name='coordination')
    if _DEBUG_MODE:
        log.opt(colors=True).debug('<red>🔄 {}: {}</red>', step, details or {})

class MCPErrorHandler:
    """Standardized MCP error handling utilities."""

    @staticmethod
    def get_error_details(error: Exception, context: str | None=None, *, log: bool=False) -> tuple[str, str, str]:
        """Return standardized MCP error info and optionally log.

        Returns:
            Tuple of (log_type, user_message, error_category)
        """
        if isinstance(error, MCPConnectionError):
            details = ('connection error', 'MCP connection failed', 'connection')
        elif isinstance(error, MCPTimeoutError):
            details = ('timeout error', 'MCP session timeout', 'timeout')
        elif isinstance(error, MCPServerError):
            details = ('server error', 'MCP server error', 'server')
        elif isinstance(error, MCPValidationError):
            details = ('validation error', 'MCP validation failed', 'validation')
        elif isinstance(error, MCPAuthenticationError):
            details = ('authentication error', 'MCP authentication failed', 'auth')
        elif isinstance(error, MCPResourceError):
            details = ('resource error', 'MCP resource unavailable', 'resource')
        elif isinstance(error, MCPError):
            details = ('MCP error', 'MCP error', 'general')
        else:
            details = ('unexpected error', 'MCP connection failed', 'unknown')
        if log:
            log_type, user_message, error_category = details
            logger.warning(f'MCP {log_type}: {error}', extra={'context': context or 'none'})
        return details

    @staticmethod
    def is_transient_error(error: Exception) -> bool:
        """Determine if an error is transient and should be retried."""
        if isinstance(error, (MCPConnectionError, MCPTimeoutError)):
            return True
        elif isinstance(error, MCPServerError):
            error_str = str(error).lower()
            return any((keyword in error_str for keyword in ['timeout', 'connection', 'network', 'temporary', 'unavailable', '503', '502', '504', '500', 'retry']))
        elif isinstance(error, (ConnectionError, TimeoutError, OSError)):
            return True
        elif isinstance(error, MCPResourceError):
            return True
        return False

    @staticmethod
    def log_error(error: Exception, context: str, level: str='auto', backend_name: str | None=None, agent_id: str | None=None) -> None:
        """Log MCP error with appropriate level and context."""
        log_type, user_message, error_category = MCPErrorHandler.get_error_details(error)
        if level == 'auto':
            level = 'warning' if error_category in ['connection', 'timeout', 'resource'] else 'error'
        log_message = f'MCP {log_type} during {context}: {error}'
        log_mcp_activity(backend_name, f'error ({level})', {'message': log_message}, agent_id=agent_id)

    @staticmethod
    def get_retry_delay(attempt: int, base_delay: float=DEFAULT_RETRY_BASE_DELAY) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        backoff_delay = base_delay * 2 ** attempt
        jitter = random.uniform(DEFAULT_RETRY_JITTER_MIN, DEFAULT_RETRY_JITTER_MAX) * backoff_delay
        return backoff_delay + jitter

    @staticmethod
    def is_auth_or_resource_error(error: Exception) -> bool:
        """Check if error is authentication or resource related (non-retryable)."""
        return isinstance(error, (MCPAuthenticationError, MCPResourceError))

class MCPConfigHelper:
    """MCP configuration management utilities."""

    @staticmethod
    def extract_tool_filtering_params(config: dict[str, Any]) -> tuple[list | None, list | None]:
        """Extract allowed_tools and exclude_tools from configuration."""
        allowed_tools = config.get('allowed_tools')
        exclude_tools = config.get('exclude_tools')
        if allowed_tools is not None and (not isinstance(allowed_tools, list)):
            if isinstance(allowed_tools, str):
                allowed_tools = [allowed_tools]
            else:
                logger.warning('MCP invalid allowed_tools type', extra={'type': type(allowed_tools).__name__, 'action': 'ignoring'})
                allowed_tools = None
        if exclude_tools is not None and (not isinstance(exclude_tools, list)):
            if isinstance(exclude_tools, str):
                exclude_tools = [exclude_tools]
            else:
                logger.warning('MCP invalid exclude_tools type', extra={'type': type(exclude_tools).__name__, 'action': 'ignoring'})
                exclude_tools = None
        return (allowed_tools, exclude_tools)

    @staticmethod
    def build_circuit_breaker_config(transport_type: str='mcp_tools', backend_name: str | None=None, agent_id: str | None=None) -> Any | None:
        """Build circuit breaker configuration for transport type."""
        if CircuitBreakerConfig is None:
            log_mcp_activity(backend_name, 'CircuitBreakerConfig unavailable', {}, agent_id=agent_id)
            return None
        try:
            config = CircuitBreakerConfig(max_failures=DEFAULT_CIRCUIT_BREAKER_MAX_FAILURES, reset_time_seconds=DEFAULT_CIRCUIT_BREAKER_RESET_TIME, backoff_multiplier=DEFAULT_CIRCUIT_BREAKER_BACKOFF_MULTIPLIER, max_backoff_multiplier=DEFAULT_CIRCUIT_BREAKER_MAX_BACKOFF_MULTIPLIER)
            log_mcp_activity(backend_name, 'created circuit breaker config', {'transport_type': transport_type}, agent_id=agent_id)
            return config
        except Exception as e:
            log_mcp_activity(backend_name, 'failed to create circuit breaker config', {'error': str(e)}, agent_id=agent_id)
            return None

class MCPCircuitBreakerManager:
    """Circuit breaker management utilities for MCP integration."""

    @staticmethod
    def apply_circuit_breaker_filtering(servers: list[dict[str, Any]], circuit_breaker, backend_name: str | None=None, agent_id: str | None=None) -> list[dict[str, Any]]:
        """Apply circuit breaker filtering to servers.

        Args:
            servers: List of server configurations
            circuit_breaker: Circuit breaker instance
            backend_name: Optional backend name for logging context
            agent_id: Optional agent ID for logging context

        Returns:
            List of servers that pass circuit breaker filtering
        """
        if not circuit_breaker:
            return servers
        filtered_servers = []
        for server in servers:
            server_name = server.get('name', 'unnamed')
            if not circuit_breaker.should_skip_server(server_name, agent_id=agent_id):
                filtered_servers.append(server)
            else:
                log_mcp_activity(backend_name, 'circuit breaker skipping server', {'server_name': server_name, 'reason': 'circuit_open'}, agent_id=agent_id)
        return filtered_servers

    @staticmethod
    async def record_event(servers: list[dict[str, Any]], circuit_breaker, event: Literal['success', 'failure'], error_message: str | None=None, backend_name: str | None=None, agent_id: str | None=None) -> None:
        """Record circuit breaker events for servers.

        Args:
            servers: List of server configurations
            event: Event type ("success" or "failure")
            circuit_breaker: Circuit breaker instance
            error_message: Optional error message for failure events
            backend_name: Optional backend name for logging context
            agent_id: Optional agent ID for logging context
        """
        if not circuit_breaker:
            return
        count = 0
        for server in servers:
            server_name = server.get('name', 'unnamed')
            try:
                if event == 'success':
                    circuit_breaker.record_success(server_name, agent_id=agent_id)
                else:
                    circuit_breaker.record_failure(server_name, agent_id=agent_id)
                count += 1
            except Exception as cb_error:
                log_mcp_activity(backend_name, 'circuit breaker record failed', {'event': event, 'server_name': server_name, 'error': str(cb_error)}, agent_id=agent_id)
        if count > 0:
            if event == 'success':
                log_mcp_activity(backend_name, 'circuit breaker recorded success', {'server_count': count}, agent_id=agent_id)
            else:
                log_mcp_activity(backend_name, 'circuit breaker recorded failure', {'server_count': count, 'error': error_message}, agent_id=agent_id)

class MCPResourceManager:
    """Resource management utilities for MCP integration."""

    @staticmethod
    async def setup_mcp_client(servers: list[dict[str, Any]], allowed_tools: list[str] | None, exclude_tools: list[str] | None, circuit_breaker=None, timeout_seconds: int=DEFAULT_TIMEOUT_SECONDS, backend_name: str | None=None, agent_id: str | None=None) -> Any | None:
        """Setup MCP client for stdio/streamable-http servers with circuit breaker protection.

        Args:
            servers: List of server configurations
            allowed_tools: Optional list of allowed tool names
            exclude_tools: Optional list of excluded tool names
            circuit_breaker: Optional circuit breaker for failure tracking
            timeout_seconds: Connection timeout in seconds
            backend_name: Optional backend name for logging context
            agent_id: Optional agent ID for logging context

        Returns:
            Connected MCPClient or None if setup failed
        """
        if MCPClient is None:
            log_mcp_activity(backend_name, 'MCPClient unavailable', {'functionality': 'disabled'}, agent_id=agent_id)
            return None
        normalized_servers = MCPSetupManager.normalize_mcp_servers(servers, backend_name, agent_id)
        stdio_streamable_servers = MCPSetupManager.separate_stdio_streamable_servers(normalized_servers, backend_name, agent_id)
        if not stdio_streamable_servers:
            log_mcp_activity(backend_name, 'no stdio/streamable-http servers configured', {}, agent_id=agent_id)
            return None
        if circuit_breaker:
            filtered_servers = MCPCircuitBreakerManager.apply_circuit_breaker_filtering(stdio_streamable_servers, circuit_breaker, backend_name, agent_id)
        else:
            filtered_servers = stdio_streamable_servers
        if not filtered_servers:
            log_mcp_activity(backend_name, 'all servers filtered by circuit breaker', {'transport_types': ['stdio', 'streamable-http']}, agent_id=agent_id)
            return None
        max_retries = DEFAULT_MAX_RETRIES
        for retry in range(max_retries):
            try:
                if retry > 0:
                    delay = MCPErrorHandler.get_retry_delay(retry - 1)
                    log_mcp_activity(backend_name, 'connection retry', {'attempt': retry, 'max_retries': max_retries - 1, 'delay_seconds': delay}, agent_id=agent_id)
                    await asyncio.sleep(delay)
                client = await MCPClient.create_and_connect(filtered_servers, timeout_seconds=timeout_seconds, allowed_tools=allowed_tools, exclude_tools=exclude_tools)
                if circuit_breaker:
                    await MCPCircuitBreakerManager.record_event(filtered_servers, circuit_breaker, 'success', backend_name=backend_name, agent_id=agent_id)
                log_mcp_activity(backend_name, 'connection successful', {'attempt': retry + 1}, agent_id=agent_id)
                return client
            except (MCPConnectionError, MCPTimeoutError, MCPServerError) as e:
                if retry < max_retries - 1:
                    MCPErrorHandler.log_error(e, f'MCP connection attempt {retry + 1}')
                    continue
                if circuit_breaker:
                    await MCPCircuitBreakerManager.record_event(filtered_servers, circuit_breaker, 'failure', str(e), backend_name, agent_id)
                log_mcp_activity(backend_name, 'connection failed after retries', {'max_retries': max_retries, 'error': str(e)}, agent_id=agent_id)
                return None
            except Exception as e:
                MCPErrorHandler.log_error(e, f'Unexpected error during MCP connection attempt {retry + 1}', 'error')
                if retry < max_retries - 1:
                    continue
                return None
        return None

    @staticmethod
    def convert_tools_to_functions(mcp_client, backend_name: str | None=None, agent_id: str | None=None, hook_manager=None) -> dict[str, Function]:
        """Convert MCP tools to Function objects with hook support.

        Args:
            mcp_client: Connected MCPClient instance
            backend_name: Optional backend name for logging context
            agent_id: Optional agent ID for logging context
            hook_manager: Optional hook manager for function hooks

        Returns:
            Dictionary mapping tool names to Function objects
        """
        if not mcp_client or not hasattr(mcp_client, 'tools'):
            return {}
        functions = {}
        hook_mgr = hook_manager
        for tool_name, tool in mcp_client.tools.items():
            try:

                def create_tool_entrypoint(captured_tool_name: str=tool_name):

                    async def tool_entrypoint(input_str: str) -> Any:
                        try:
                            arguments = json.loads(input_str)
                        except (json.JSONDecodeError, ValueError) as e:
                            log_mcp_activity(backend_name, 'invalid JSON arguments for tool', {'tool_name': captured_tool_name, 'error': str(e)}, agent_id=agent_id)
                            raise MCPValidationError(f'Invalid JSON arguments for tool {captured_tool_name}: {e}', field='arguments', value=input_str)
                        return await mcp_client.call_tool(captured_tool_name, arguments)
                    return tool_entrypoint
                entrypoint = create_tool_entrypoint()
                description = tool.description
                if description is None or not isinstance(description, str):
                    description = f'MCP tool: {tool_name}'
                    log_mcp_activity(backend_name, 'tool description sanitized', {'tool_name': tool_name, 'original': tool.description}, agent_id=agent_id)
                parameters = tool.inputSchema
                if parameters is None or not isinstance(parameters, dict):
                    parameters = {'type': 'object', 'properties': {}}
                    log_mcp_activity(backend_name, 'tool parameters sanitized', {'tool_name': tool_name, 'original': tool.inputSchema}, agent_id=agent_id)
                function_hooks = hook_mgr.get_hooks_for_function(tool_name) if hook_mgr else {}
                function = Function(name=tool_name, description=description, parameters=parameters, entrypoint=entrypoint, hooks=function_hooks)
                function._backend_name = backend_name
                function._agent_id = agent_id
                functions[function.name] = function
            except Exception as e:
                log_mcp_activity(backend_name, 'failed to register tool', {'tool_name': tool_name, 'error': str(e)}, agent_id=agent_id)
        log_mcp_activity(backend_name, 'registered tools as Function objects', {'tool_count': len(functions)}, agent_id=agent_id)
        return functions

    @staticmethod
    async def cleanup_mcp_client(client, backend_name: str | None=None, agent_id: str | None=None) -> None:
        """Clean up MCP client connections.

        Args:
            client: MCPClient instance to clean up
            backend_name: Optional backend name for logging context
            agent_id: Optional agent ID for logging context
        """
        if client:
            try:
                await client.disconnect()
                log_mcp_activity(backend_name, 'client cleanup completed', {}, agent_id=agent_id)
            except Exception as e:
                log_mcp_activity(backend_name, 'error during client cleanup', {'error': str(e)}, agent_id=agent_id)

    @staticmethod
    async def setup_mcp_context_manager(backend_instance, backend_name: str | None=None, agent_id: str | None=None):
        """Setup MCP tools if configured during context manager entry."""
        if hasattr(backend_instance, 'mcp_servers') and backend_instance.mcp_servers and (not backend_instance._mcp_initialized):
            try:
                await backend_instance._setup_mcp_tools()
            except Exception as e:
                log_mcp_activity(backend_name, 'setup failed during context entry', {'error': str(e)}, agent_id=agent_id)
        return backend_instance

    @staticmethod
    async def cleanup_mcp_context_manager(backend_instance, logger_instance=None, backend_name: str | None=None, agent_id: str | None=None) -> None:
        """Clean up MCP resources during context manager exit."""
        log = logger_instance or logger
        try:
            if hasattr(backend_instance, 'cleanup_mcp'):
                await backend_instance.cleanup_mcp()
        except Exception as e:
            log.error(f"Error during MCP cleanup for backend '{backend_name}': {e}")
            log_mcp_activity(backend_name, 'error during cleanup', {'error': str(e)}, agent_id=agent_id)

class MCPSetupManager:
    """MCP setup and initialization utilities."""

    @staticmethod
    def normalize_mcp_servers(servers: Any, backend_name: str | None=None, agent_id: str | None=None) -> list[dict[str, Any]]:
        """Validate and normalize mcp_servers into a list of dicts.

        Args:
            servers: MCP servers configuration (list, dict, or None)
            backend_name: Optional backend name for logging context
            agent_id: Optional agent ID for logging context

        Returns:
            Normalized list of server dictionaries
        """
        if not servers:
            return []
        if isinstance(servers, dict):
            if 'type' in servers:
                servers = [servers]
            else:
                converted = []
                for name, server_config in servers.items():
                    if isinstance(server_config, dict):
                        server = server_config.copy()
                        server['name'] = name
                        converted.append(server)
                servers = converted
        if not isinstance(servers, list):
            log_mcp_activity(backend_name, 'invalid mcp_servers type', {'type': type(servers).__name__, 'expected': 'list or dict'}, agent_id=agent_id)
            return []
        normalized = []
        for i, server in enumerate(servers):
            if not isinstance(server, dict):
                log_mcp_activity(backend_name, 'skipping invalid server', {'index': i, 'server': str(server)}, agent_id=agent_id)
                continue
            if 'type' not in server:
                log_mcp_activity(backend_name, 'server missing type field', {'index': i}, agent_id=agent_id)
                continue
            if 'name' not in server:
                server = server.copy()
                server['name'] = f'server_{i}'
            normalized.append(server)
        return normalized

    @staticmethod
    def separate_stdio_streamable_servers(servers: list[dict[str, Any]], backend_name: str | None=None, agent_id: str | None=None) -> list[dict[str, Any]]:
        """Extract only stdio and streamable-http servers.

        Args:
            servers: List of server configurations
            backend_name: Optional backend name for logging context
            agent_id: Optional agent ID for logging context

        Returns:
            List containing only stdio and streamable-http servers
        """
        stdio_streamable = []
        for server in servers:
            transport_type = server.get('type', '').lower()
            if transport_type in ['stdio', 'streamable-http']:
                stdio_streamable.append(server)
        return stdio_streamable

class MCPCircuitBreaker:
    """
    Circuit breaker for MCP server failure handling.

    Provides consistent failure tracking and exponential backoff across all MCP integrations.
    Prevents repeated connection attempts to failing servers while allowing recovery.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig]=None, backend_name: Optional[str]=None, agent_id: Optional[str]=None):
        """
        Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration. Uses default if None.
            backend_name: Name of the backend using this circuit breaker for logging context.
            agent_id: Optional agent ID for logging context.
        """
        self.config = config or CircuitBreakerConfig()
        self.backend_name = backend_name
        self.agent_id = agent_id
        self._server_status: Dict[str, ServerStatus] = {}

    def should_skip_server(self, server_name: str, agent_id: Optional[str]=None) -> bool:
        """
        Check if server should be skipped due to circuit breaker.

        Args:
            server_name: Name of the server to check

        Returns:
            True if server should be skipped, False otherwise
        """
        if server_name not in self._server_status:
            return False
        status = self._server_status[server_name]
        if status.failure_count < self.config.max_failures:
            return False
        current_time = time.monotonic()
        time_since_failure = current_time - status.last_failure_time
        backoff_time = self._calculate_backoff_time(status.failure_count)
        if time_since_failure > backoff_time:
            log_mcp_activity(self.backend_name, 'Circuit breaker reset for server', {'server_name': server_name, 'backoff_time_seconds': backoff_time}, agent_id=self.agent_id or agent_id)
            self._reset_server(server_name)
            return False
        return True

    def record_failure(self, server_name: str, agent_id: Optional[str]=None) -> None:
        """
        Record a server failure for circuit breaker.

        Args:
            server_name: Name of the server that failed
        """
        current_time = time.monotonic()
        if server_name not in self._server_status:
            self._server_status[server_name] = ServerStatus()
        status = self._server_status[server_name]
        status.failure_count += 1
        status.last_failure_time = current_time
        if status.failure_count >= self.config.max_failures:
            backoff_time = self._calculate_backoff_time(status.failure_count)
            log_mcp_activity(self.backend_name, 'Server circuit breaker opened', {'server_name': server_name, 'failure_count': status.failure_count, 'backoff_time_seconds': backoff_time}, agent_id=self.agent_id or agent_id)
        else:
            log_mcp_activity(self.backend_name, 'Server failure recorded', {'server_name': server_name, 'failure_count': status.failure_count, 'max_failures': self.config.max_failures}, agent_id=self.agent_id or agent_id)

    def record_success(self, server_name: str, agent_id: Optional[str]=None) -> None:
        """
        Record a successful connection, resetting failure count.

        Args:
            server_name: Name of the server that succeeded
        """
        if server_name in self._server_status:
            old_status = self._server_status[server_name]
            if old_status.failure_count > 0:
                log_mcp_activity(self.backend_name, 'Server recovered', {'server_name': server_name, 'previous_failure_count': old_status.failure_count}, agent_id=self.agent_id or agent_id)
            self._reset_server(server_name)

    def _reset_server(self, server_name: str) -> None:
        """Reset circuit breaker state for a specific server."""
        if server_name in self._server_status:
            del self._server_status[server_name]

    def _calculate_backoff_time(self, failure_count: int) -> float:
        """
        Calculate backoff time based on failure count.

        Args:
            failure_count: Number of failures

        Returns:
            Backoff time in seconds
        """
        if failure_count < self.config.max_failures:
            return 0.0
        exponent = failure_count - self.config.max_failures
        multiplier = min(self.config.backoff_multiplier ** exponent, self.config.max_backoff_multiplier)
        return self.config.reset_time_seconds * multiplier

    def __repr__(self) -> str:
        """String representation for debugging."""
        failing_count = len([s for s in self._server_status.values() if s.is_failing])
        total_servers = len(self._server_status)
        return f'MCPCircuitBreaker(failing={failing_count}/{total_servers}, config={self.config})'

