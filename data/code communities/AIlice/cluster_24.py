# Cluster 24

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--addr', type=str, help='The address where the service runs on.')
    parser.add_argument('--clusterURL', type=str, help='The weaviate cluster url.')
    parser.add_argument('--apiKey', type=str, help='The api key for this cluster.')
    parser.add_argument('--oaiKey', type=str, help='The OpenAI api key.')
    args = parser.parse_args()
    makeServer(AStorageWeaviate, {'clusterURL': args.clusterURL, 'apiKey': args.apiKey, 'oaiKey': args.oaiKey}, args.addr, ['ModuleInfo', 'Open', 'Reset', 'Store', 'Query', 'Recall']).Run()

def makeServer(objCls, objArgs, url, APIList, serverPrivateKeyPath=None, clientPublicKeysDir=None, validateReturn=True, atomicCall=True):
    return GenesisRPCServer(objCls, objArgs, url, APIList, serverPrivateKeyPath, clientPublicKeysDir, validateReturn, atomicCall)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--addr', type=str, help='The address where the service runs on.')
    args = parser.parse_args()
    makeServer(ADuckDuckGo, dict(), args.addr, ['ModuleInfo', 'DuckDuckGo', 'ScrollDown']).Run()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--addr', type=str, help='The address where the service runs on.')
    args = parser.parse_args()
    makeServer(AStorageVecDB, dict(), args.addr, ['ModuleInfo', 'Open', 'Reset', 'Store', 'Query', 'Recall']).Run()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--addr', type=str, help='The address where the service runs on.')
    args = parser.parse_args()
    makeServer(AComputer, dict(), args.addr, ['ModuleInfo', 'ScreenShot', 'LocateAndClick', 'LocateAndScroll', 'TypeWrite', 'ReadImage', 'WriteImage', 'WriteFile', 'Proxy'], atomicCall=False).Run()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--addr', type=str, help='The address where the service runs on.')
    parser.add_argument('--incontainer', action='store_true', help='Run in container. Please DO NOT turn on this switch on non-virtual machines, otherwise it will cause serious security risks.')
    args = parser.parse_args()
    makeServer(AScripter, {'incontainer': args.incontainer}, args.addr, ['ModuleInfo', 'PlatformInfo', 'CheckOutput', 'RunBash', 'RunPython', 'ScrollUp', 'Save2File']).Run()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--addr', type=str, help='The address where the service runs on.')
    args = parser.parse_args()
    makeServer(AStorageVecDB, dict(), args.addr, ['ModuleInfo', 'Open', 'Reset', 'Store', 'Query', 'Recall', 'Release']).Run()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--addr', type=str, help='The address where the service runs on.')
    args = parser.parse_args()
    makeServer(AArxiv, dict(), args.addr, ['ModuleInfo', 'ArxivSearch', 'ScrollDown'], atomicCall=False).Run()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--addr', type=str, help='The address where the service runs on.')
    args = parser.parse_args()
    makeServer(ASpeech, dict(), args.addr, ['ModuleInfo', 'PrepareModel', 'Speech2Text', 'Text2Speech', 'GetAudio', 'Speak', 'SwitchTone', 'SetDevices']).Run()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--addr', type=str, help='The address where the service runs on.')
    args = parser.parse_args()
    makeServer(AGoogle, dict(), args.addr, ['ModuleInfo', 'Google', 'ScrollDown']).Run()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--addr', type=str, help='The address where the service runs on.')
    parser.add_argument('--pdfOutputDir', type=str, default='', help='You can set it as a directory to store the OCR results of PDF files to avoid repeated OCR computation.')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmpdir:
        makeServer(ABrowser, {'pdfOutputDir': args.pdfOutputDir if '' != args.pdfOutputDir.strip() else tmpdir}, args.addr, ['ModuleInfo', 'Browse', 'Edit', 'ScrollDown', 'ScrollUp', 'SearchDown', 'SearchUp', 'GetFullText', 'GetLink', 'ExecuteJS', 'Replace', 'ReplaceAll', 'SaveTo', 'Destroy']).Run()

def AddActionMethod(kls, name: str, tool):

    async def async_action_method(self, jsonParam: str) -> str:
        try:
            return await self.session.call_tool(name, arguments=json.loads(jsonParam))
        except Exception as e:
            return json.dumps({'error': str(e)})

    def action_method(self, jsonParam: str) -> list:
        ret = asyncio.run_coroutine_threadsafe(async_action_method(self, jsonParam), loop).result()
        result = []
        for item in ret.content:
            if type(item) == mcp.types.TextContent:
                result.append(str(item))
            elif type(item) == mcp.types.ImageContent:
                result.append(AImage.FromJson({'data': item.data}))
            elif type(item) == mcp.types.EmbeddedResource:
                result.append('[Unsupported EmbeddedResource content]')
        return result
    action_method.__name__ = name
    action_method.__qualname__ = f'{kls.__name__}.{name}'
    setattr(kls, name, action_method)
    kls.MODULE_INFO['ACTIONS'][name] = {'func': name, 'prompt': ConstructPrompt(name, tool), 'type': 'primary'}
    return

def ConstructPrompt(name: str, tool) -> str:
    return f'{name} is a tool called through stringified json. Its description is: {tool.description}. Its schema is: {str(tool.inputSchema)}. You need to put json in triple quotes as a string type parameter.'

def MakeWrapper(serverParams=None, serverUrl=None):

    class AMCPWrapper:
        MODULE_INFO = {'NAME': serverParams.args[0] if serverParams is not None else serverUrl, 'ACTIONS': {}}

        def __init__(self, serverParams=None, serverUrl=None):
            self.serverParams = serverParams
            self.serverUrl = serverUrl
            self.exit_stack = None
            self.stdio = None
            self.write = None
            self.session = None
            asyncio.run_coroutine_threadsafe(self.initialize(), loop).result()
            return

        def __del__(self):
            if hasattr(self, 'exit_stack') and self.exit_stack:
                try:
                    asyncio.run_coroutine_threadsafe(self.close(), loop).result()
                except Exception as e:
                    print(f'Error closing resources: {e}')

        async def initialize(self):
            self.exit_stack = AsyncExitStack()
            if self.serverParams is not None:
                self.MODULE_INFO['NAME'] = self.serverParams.args[0]
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(self.serverParams))
                self.session = await self.exit_stack.enter_async_context(ClientSession(*stdio_transport))
            elif self.serverUrl is not None:
                self.MODULE_INFO['NAME'] = self.serverUrl
                streams = await self.exit_stack.enter_async_context(sse_client(url=self.serverUrl))
                self.session = await self.exit_stack.enter_async_context(ClientSession(*streams))
            await self.session.initialize()

        async def close(self):
            if self.exit_stack:
                await self.exit_stack.aclose()

        def ModuleInfo(self):
            return self.MODULE_INFO
    maxRetries = 3
    for attempt in range(maxRetries):
        try:
            toolsInfo = asyncio.run_coroutine_threadsafe(LoadMeta(serverParams=serverParams, serverUrl=serverUrl), loop).result()
            break
        except Exception as e:
            if attempt == maxRetries - 1:
                raise
            print(f'Attempt {attempt + 1} failed: {e}, retrying...')
            time.sleep(1)
    actions = []
    for tool in toolsInfo.tools:
        print(tool)
        actions.append(tool.name)
        AddActionMethod(AMCPWrapper, tool.name, tool)
    return (AMCPWrapper, actions)

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Ailice MCP Wrapper', epilog='\nExamples:\n  # SSE mode example\n  ailice_mcp_wrapper --addr localhost:59200 sse --server_url http://example:8000/sse\n  \n  # Stdio mode example\n  ailice_mcp_wrapper --addr localhost:59200 stdio python3 mcp_demo_echo.py\n  \n  # Stdio mode with environment variables\n  ailice_mcp_wrapper --addr localhost:59200 stdio python3 mcp_demo_echo.py --env PYTHONPATH=/path/to/modules --env DEBUG=1\n', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--addr', type=str, required=True, help='The address where the service runs on. This address will be given to Ailice, to connect to the service.')
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    sse_parser = subparsers.add_parser('sse', help='SSE mode', description='Run in SSE mode, connecting to an existing MCP server', epilog='Example: ailice_mcp_wrapper --addr localhost:59200 sse --server_url http://example:8000/sse')
    sse_parser.add_argument('--server_url', type=str, required=True, help='The url of MCP server.')
    stdio_parser = subparsers.add_parser('stdio', help='Stdio mode', description='Run in stdio mode, executing a local command', epilog='Example: ailice_mcp_wrapper --addr localhost:59200 stdio python3 mcp_demo_echo.py')
    stdio_parser.add_argument('cmd', help='The command to be executed.')
    stdio_parser.add_argument('cmd_args', nargs='*', help='Command arguments.')
    stdio_parser.add_argument('--env', action='append', dest='env_vars', metavar='KEY=VALUE', help='Set up env variables, the format should be: KEY=VALUE')
    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        sys.exit(1)
    env = os.environ.copy()
    if args.mode == 'stdio' and args.env_vars:
        for env_var in args.env_vars:
            try:
                key, value = env_var.split('=', 1)
                env[key] = value
            except ValueError:
                print(f'Warning: Incorrect environment variable format: {env_var}, should be KEY=VALUE format.')
    try:
        if args.mode == 'stdio':
            serverParams = StdioServerParameters(command=args.cmd, args=args.cmd_args, env=env)
            kls, actions = MakeWrapper(serverParams=serverParams)
            server = makeServer(kls, {'serverParams': serverParams}, args.addr, ['ModuleInfo'] + actions)
        else:
            kls, actions = MakeWrapper(serverUrl=args.server_url)
            server = makeServer(kls, {'serverUrl': args.server_url}, args.addr, ['ModuleInfo'] + actions)
        print(f'We will run the MCP server next, you can use this address {args.addr} to load the wrapped ext-module and use the corresponding tools.')
        server.Run()
    except Exception as e:
        print(f'Error in main: {e}', traceback.print_tb(e.__traceback__))
    finally:
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=5)

