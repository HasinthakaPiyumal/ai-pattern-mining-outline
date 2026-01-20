# Cluster 16

class MCPClient:

    def __init__(self, server_configs: Union[Dict[str, Any], List[Dict[str, Any]]], connect_timeout: float=120.0):
        if isinstance(server_configs, dict):
            self.server_configs = [server_configs]
        else:
            self.server_configs = server_configs
        self.event_loop = asyncio.new_event_loop()
        self.sessions: list[Client] = []
        self.mcp_tools: list[list[Any]] = []
        self.task = None
        self.thread_running = threading.Event()
        self.working_thread = threading.Thread(target=self._run_event, daemon=True)
        self.connect_timeout = connect_timeout
        self.tools = None
        self.tool_schemas = None
        self.tool_descriptions = None

    def _disconnect(self):
        if hasattr(self, 'shutdown_event') and self.shutdown_event:
            self.event_loop.call_soon_threadsafe(self.shutdown_event.set)
        if self.task and (not self.task.done()):
            self.event_loop.call_soon_threadsafe(self.task.cancel)
        if hasattr(self, 'working_thread') and self.working_thread.is_alive():
            self.working_thread.join(timeout=5)
        if hasattr(self, 'event_loop') and (not self.event_loop.is_closed()):
            self.event_loop.close()

    def _connect(self):
        self.working_thread.start()
        if not self.thread_running.wait(timeout=self.connect_timeout):
            self._disconnect()
            raise TimeoutError(f"Couldn't connect to the MCP server after {self.connect_timeout} seconds")

    def __enter__(self):
        self._connect()
        return self.get_toolkits()

    def __del__(self):
        try:
            self._disconnect()
        except Exception:
            pass

    def __exit__(self, exc_type, exc_value, traceback):
        self._disconnect()

    def _run_event(self):
        """Runs the event loop in a separate thread (for synchronous usage)."""
        print('Running event loop')
        asyncio.set_event_loop(self.event_loop)

        async def setup():
            try:
                async with AsyncExitStack() as stack:
                    connections = [await stack.enter_async_context(self._start_server(config)) for config in self.server_configs]
                    self.sessions, self.mcp_tools = [list(c) for c in zip(*connections)]
                    self.thread_running.set()
                    self.shutdown_event = asyncio.Event()
                    await self.shutdown_event.wait()
            except Exception as e:
                logger.error(f'Error in MCP event loop: {str(e)}')
                self.thread_running.set()
                raise
        self.task = self.event_loop.create_task(setup())
        try:
            self.event_loop.run_until_complete(self.task)
        except asyncio.CancelledError:
            logger.info('MCP client event loop was cancelled')
        except Exception as e:
            logger.error(f'Error in MCP event loop: {str(e)}')
        finally:
            if not self.event_loop.is_closed():
                self.event_loop.close()

    @asynccontextmanager
    async def _start_server(self, config: Dict[str, Any]):
        client = Client(config)
        async with client:
            tools = await client.list_tools()
            yield (client, tools)

    def create_tool(self, session: Client, mcp_tools: List[Any], config: Dict[str, Any]) -> Toolkit:

        def _sync_call_tool(name: str, **kwargs) -> Any:
            try:
                if 'arguments' in kwargs and len(kwargs) == 1:
                    arguments = kwargs['arguments']
                else:
                    arguments = kwargs
                logger.info(f'Calling MCP tool: {name} with arguments: {arguments}')
                future = asyncio.run_coroutine_threadsafe(session.call_tool(name, arguments), self.event_loop)
                result = future.result(timeout=30)
                logger.info(f'MCP tool {name} call completed successfully')
                return result
            except (TimeoutError, ClientError, McpError) as e:
                logger.error(f'Error calling MCP tool {name}: {str(e)}')
                raise
            except Exception as e:
                logger.error(f'Unexpected error calling MCP tool {name}: {str(e)}')
                raise
        all_tools = []
        for mcp_tool in mcp_tools:
            input_schema = getattr(mcp_tool, 'inputSchema', {})
            if not input_schema and hasattr(mcp_tool, 'input_schema'):
                input_schema = mcp_tool.input_schema
            properties = input_schema.get('properties', {})
            required = input_schema.get('required', [])
            inputs = properties
            partial_func = partial(_sync_call_tool, mcp_tool.name)
            partial_func.__name__ = mcp_tool.name
            tool = MCPTool(name=mcp_tool.name, description=getattr(mcp_tool, 'description', None) or '', inputs=inputs, required=required, function=partial_func)
            all_tools.append(tool)
        tool_collection = Toolkit(name=next(iter(config.get('mcpServers').keys())), tools=all_tools)
        return tool_collection

    def get_toolkits(self) -> List[Toolkit]:
        """Return a list ofToolkits, one per server."""
        if not self.sessions:
            raise RuntimeError('Session not initialized')
        return [self.create_tool(session, tools, config) for session, tools, config in zip(self.sessions, self.mcp_tools, self.server_configs)]

