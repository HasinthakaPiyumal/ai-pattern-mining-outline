# Cluster 25

class ADuckDuckGo:

    def __init__(self):
        self.baseURL = 'https://api.duckduckgo.com/'
        self.sessions = {}
        self.functions = {'SCROLLDOWN': '#scroll down the page: \nSCROLL-DOWN-DUCKDUCKGO<!|session: str|!>'}
        return

    def ModuleInfo(self):
        return {'NAME': 'duckduckgo', 'ACTIONS': {'DUCKDUCKGO': {'func': 'DuckDuckGo', 'prompt': 'Use duckduckgo to search internet content.', 'type': 'primary'}, 'SCROLL-DOWN-DUCKDUCKGO': {'func': 'ScrollDown', 'prompt': 'Scrolldown the results.', 'type': 'supportive'}}}

    def GetSessionID(self) -> str:
        id = f'session-{str(random.randint(0, 99999999))}'
        while id in self.sessions:
            id = f'session-{str(random.randint(0, 99999999))}'
        return id

    def DuckDuckGo(self, keywords: str) -> str:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(keywords, max_results=10)]
            ret = str(results) if len(results) > 0 else 'No search results were found. Please check if you used overly complex keywords or unsupported search syntax. Note that relaxing your search terms is an effective strategy when no valid search results are returned.'
        except Exception as e:
            print(f'Error during the request: {e}')
            ret = str(e)
        finally:
            loop.close()
        session = self.GetSessionID()
        self.sessions[session] = AScrollablePage(functions=self.functions)
        self.sessions[session].LoadPage(str(ret), 'TOP')
        return self.sessions[session]() + '\n\n' + f'Session name: "{session}"\n'

    def ScrollDown(self, session: str) -> str:
        return self.sessions[session].ScrollDown() + '\n\n' + f'Session name: "{session}"\n'

class AGoogle:

    def __init__(self, api_key: str, cse_id: str):
        self.api_key = api_key
        self.cse_id = cse_id
        self.service = build('customsearch', 'v1', developerKey=self.api_key)
        self.sessions = {}
        self.functions = {'SCROLLDOWN': '#scroll down the page: \nSCROLL-DOWN-GOOGLE<!|session: str|!>'}
        return

    def ModuleInfo(self):
        return {'NAME': 'google', 'ACTIONS': {'GOOGLE': {'func': 'Google', 'prompt': 'Use Google to search the web.', 'type': 'primary'}, 'SCROLL-DOWN-GOOGLE': {'func': 'ScrollDown', 'prompt': 'Scroll down the search results.', 'type': 'supportive'}}}

    def GetSessionID(self) -> str:
        id = f'session-{str(random.randint(0, 99999999))}'
        while id in self.sessions:
            id = f'session-{str(random.randint(0, 99999999))}'
        return id

    def Google(self, keywords: str) -> str:
        try:
            res = self.service.cse().list(q=keywords, cx=self.cse_id).execute()
            results = res.get('items', [])
            ret = str(results) if len(results) > 0 else 'No search results were found. Please check if you used overly complex keywords or unsupported search syntax. Note that relaxing your search terms is an effective strategy when no valid search results are returned.'
        except Exception as e:
            print('Google Search exception: ', e)
            ret = f'Google Search exception: {str(e)}'
        session = self.GetSessionID()
        self.sessions[session] = AScrollablePage(functions=self.functions)
        self.sessions[session].LoadPage(str(ret), 'TOP')
        return self.sessions[session]() + '\n\n' + f'Session name: "{session}"\n'

    def ScrollDown(self, session: str) -> str:
        return self.sessions[session].ScrollDown() + '\n\n' + f'Session name: "{session}"\n'

class AScripter:

    def __init__(self, incontainer=False):
        self.incontainer = incontainer
        self.sessions = {}
        self.sessionsLock = threading.Lock()
        self.reader = threading.Thread(target=self.OutputReader, args=())
        self.reader.start()
        self.functions = {'SCROLLUP': '#scroll up the page: \nSCROLL-UP-TERM<!|session: str|!>'}
        return

    def ModuleInfo(self):
        return {'NAME': 'scripter', 'ACTIONS': {'PLATFORM-INFO': {'func': 'PlatformInfo', 'prompt': 'Get the platform information of the current code execution environment.', 'type': 'primary'}, 'BASH': {'func': 'RunBash', 'prompt': 'Create a bash execution environment and execute a bash script. A timeout error will occur for programs that have not been completed for a long time. Different calls to a BASH function are independent of each other. The state from previous calls, such as custom environment variables and the current directory, will not affect subsequent calls. Note that this means you might need to redefine some environment variables or re-enter certain directories in each BASH call.', 'type': 'primary'}, 'PYTHON': {'func': 'RunPython', 'prompt': 'Execute python code. Please note that you need to copy the complete code here, and you must not use references.', 'type': 'primary'}, 'CHECK-OUTPUT': {'func': 'CheckOutput', 'prompt': 'Obtain script execution output result.', 'type': 'supportive'}, 'SCROLL-UP-TERM': {'func': 'ScrollUp', 'prompt': 'Scroll up the results.', 'type': 'supportive'}, 'SAVE-TO-FILE': {'func': 'Save2File', 'prompt': 'Save text or code to file.', 'type': 'primary'}}}

    def GetSessionID(self) -> str:
        id = f'session-{str(random.randint(0, 99999999))}'
        while id in self.sessions:
            id = f'session-{str(random.randint(0, 99999999))}'
        return id

    def RunCMD(self, session: str, cmd: list[str], timeout: int=30):
        env = os.environ.copy()
        env['A_IN_CONTAINER'] = '1' if self.incontainer else '0'
        self.sessions[session]['proc'] = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        if os.name != 'nt':
            os.set_blocking(self.sessions[session]['proc'].stdout.fileno(), False)
        self.Wait(process=self.sessions[session]['proc'], timeout=timeout)
        return

    def Wait(self, process, timeout):
        t0 = time.time()
        while time.time() < t0 + timeout:
            if process.poll() is not None:
                return
            time.sleep(0.5)

    def CheckProcOutput(self, session: str) -> tuple[str, bool]:
        process = self.sessions[session]['proc']
        output = ''
        completed = False
        if process.poll() is not None:
            for i in range(2):
                remainingOutput = ''
                try:
                    remainingOutput = process.stdout.read()
                    break
                except TypeError as e:
                    time.sleep(1)
                    remainingOutput += str(e) if 1 == i else ''
                    continue
            if remainingOutput:
                output += remainingOutput
            completed = True
        else:
            while True:
                line = process.stdout.readline()
                if line:
                    output += line
                else:
                    break
        return (output, completed)

    def UpdateSession(self, session: str):
        try:
            output, completed = self.CheckProcOutput(session=session)
            self.sessions[session]['completed'] = completed
            self.sessions[session]['output'] += output
            p = '\nThe program takes longer to complete. You can use WAIT to wait for a while and then use CHECK-OUTPUT function to get new output.' if not completed else '\nExecution completed.'
        except Exception as e:
            p = f'Exception when check the output of program execution: {str(e)}\n {traceback.format_exc()}'
            print(p)
        finally:
            self.sessions[session]['pages'].LoadPage(self.sessions[session]['output'] + '\n\n---\n\n' + p, 'BOTTOM')

    def OutputReader(self):
        while True:
            with self.sessionsLock:
                for session in self.sessions:
                    if self.sessions[session]['completed']:
                        continue
                    self.UpdateSession(session)
            time.sleep(1.0)
        return

    def CheckOutput(self, session: str) -> str:
        with self.sessionsLock:
            return self.sessions[session]['pages']() + '\n\n' + f'Session name: "{session}"\n'

    def PlatformInfo(self) -> str:
        info = platform.uname()
        currentPath = os.getcwd()
        contents = os.listdir(currentPath)
        newline = '\n'
        return f'system: {info.system}, release: {info.release}, version: {info.version}, machine: {info.machine} current path: {currentPath} contents of current path: {(newline.join(contents) if len(contents) <= 32 else newline.join(contents[:32]) + '....[The tail content has been ignored. You can use BASH function to execute system commands to view the remaining content]')}'

    def RunBash(self, code: str) -> str:
        with self.sessionsLock:
            try:
                session = self.GetSessionID()
                self.sessions[session] = {'proc': None, 'pages': AScrollablePage(functions=self.functions), 'output': '', 'lock': threading.Lock()}
                self.RunCMD(session, ['bash', '-c', code])
            except Exception as e:
                self.sessions[session]['output'] += f'Exception: {str(e)}\n {traceback.format_exc()}'
            self.UpdateSession(session)
            return f'{self.sessions[session]['pages']()}\nNote that each BASH function execution is independent, so if you want to use the state from the current execution (current directory / custom environment variables, etc.) in subsequent BASH functions, you need to redefine them.\n\n\nSession name: "{session}"\n'

    def RunPython(self, code: str) -> str:
        with self.sessionsLock:
            with tempfile.NamedTemporaryFile(mode='w', delete=True) as temp:
                temp.write(code)
                temp.flush()
                try:
                    session = self.GetSessionID()
                    self.sessions[session] = {'proc': None, 'pages': AScrollablePage(functions=self.functions), 'output': '', 'lock': threading.Lock()}
                    self.RunCMD(session, ['python3', '-u', temp.name])
                except Exception as e:
                    self.sessions[session]['output'] += f'Exception: {str(e)}\n {traceback.format_exc()}'
            self.UpdateSession(session)
            return self.sessions[session]['pages']() + '\n\n' + f'Session name: "{session}"\n'

    def ScrollUp(self, session: str) -> str:
        with self.sessionsLock:
            return self.sessions[session]['pages'].ScrollUp() + '\n\n' + f'Session name: "{session}"\n'

    def Save2File(self, filePath: str, code: str) -> str:
        try:
            dirPath = os.path.dirname(filePath)
            if '' != dirPath:
                os.makedirs(dirPath, exist_ok=True)
            with open(filePath, 'w') as f:
                f.write(code)
            return f'The file contents has been written.'
        except Exception as e:
            return f'Exception encountered while writing to file. EXCEPTION: {str(e)}'

class AArxiv:

    def __init__(self):
        self.sessions = {}
        self.functions = {'SCROLLDOWN': '#scroll down the page: \nSCROLL-DOWN-ARXIV<!|session: str|!>'}
        self.lock = threading.Lock()
        return

    def ModuleInfo(self):
        return {'NAME': 'arxiv', 'ACTIONS': {'ARXIV': {'func': 'ArxivSearch', 'prompt': 'Search arXiv for academic papers.\nParameters:\n- query (str): The search query. Construct queries with:\n    - Logical combinations: AND/OR/ANDNOT operators\n    - Field restrictions: Limit searches to specific fields using these options:\n        - \'ti\': Title\n        - \'au\': Author\n        - \'abs\': Abstract\n        - \'co\': Comment\n        - \'jr\': Journal Reference\n        - \'cat\': Subject Category\n        - \'rn\': Report Number\n        - \'id\': Id\n        - \'all\': All of the above\n\n- options (str): A JSON string with search parameters. Pass \'{}\' to use all default values.\n  - sort_by (optional, str): Sort criterion. Default: \'relevance\'. Options: \'relevance\', \'lastUpdatedDate\', \'submittedDate\'.\n  - sort_order (optional, str): Sort order. Default: \'descending\'. Options: \'ascending\', \'descending\'.\n  - max_results (optional, int): Number of results to return. Default: 10.\n  \nExamples:\nARXIV<!|query="transformer architecture", options=\'{"max_results": 5, "sort_by": "submittedDate"}\'|!>\nARXIV<!|query=\'cat:hep-ph ANDNOT ti:"quantum gravity"\', options=\'{"sort_by": "submittedDate", "sort_order": "descending", "max_results": 5}\'|!>', 'type': 'primary'}, 'SCROLL-DOWN-ARXIV': {'func': 'ScrollDown', 'prompt': 'Scroll down the results.', 'type': 'supportive'}}}

    def GetSessionID(self) -> str:
        with self.lock:
            id = f'session-{str(random.randint(0, 99999999))}'
            while id in self.sessions:
                id = f'session-{str(random.randint(0, 99999999))}'
            return id

    def ParseEntry(self, entry: arxiv.Result) -> dict:
        return {'arxiv_id': entry.entry_id.split('/')[-1], 'title': entry.title, 'authors': [author.name for author in entry.authors], 'summary': entry.summary.replace('\n', ' '), 'published_date': entry.published.isoformat(), 'pdf_url': entry.pdf_url}

    def FormatResults(self, results: list) -> str:
        if not results:
            return 'No search results were found. Please check if you used overly complex keywords or unsupported search syntax. Note that relaxing your search terms is an effective strategy when no valid search results are returned.'
        return '\n\n---\n\n'.join((f'Result {i + 1}:\n  ID: {r['arxiv_id']}\n  Title: {r['title']}\n  Authors: {', '.join(r['authors'])}\n  Summary: {r['summary']}\n  Published: {r['published_date']}\n  PDF URL: {r['pdf_url']}' for i, r in enumerate(results)))

    def ArxivSearch(self, query: str, options: str) -> str:
        try:
            try:
                opts = json.loads(options) if options else {}
            except json.JSONDecodeError:
                return 'Error: Invalid JSON format in options parameter.'
            sort_by = opts.get('sort_by', 'relevance')
            sort_order = opts.get('sort_order', 'descending')
            max_results = opts.get('max_results', 10)
            sort_criterion = {'relevance': arxiv.SortCriterion.Relevance, 'lastUpdatedDate': arxiv.SortCriterion.LastUpdatedDate, 'submittedDate': arxiv.SortCriterion.SubmittedDate}[sort_by]
            sort_order_enum = {'ascending': arxiv.SortOrder.Ascending, 'descending': arxiv.SortOrder.Descending}[sort_order]
            search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_criterion, sort_order=sort_order_enum)
            ret = self.FormatResults([self.ParseEntry(r) for r in list(search.results())[:max_results]])
        except Exception as e:
            ret = f'arxiv exception: {str(e)}'
        session = self.GetSessionID()
        content = AScrollablePage(functions=self.functions)
        content.LoadPage(str(ret), 'TOP')
        with self.lock:
            self.sessions[session] = content
        return content() + '\n\n' + f'Session name: "{session}"\n'

    def ScrollDown(self, session: str) -> str:
        with self.lock:
            if session not in self.sessions:
                return 'Invalid session ID.'
            return self.sessions[session].ScrollDown() + '\n\n' + f'Session name: "{session}"\n'

class AGoogle:

    def __init__(self):
        self.sessions = {}
        self.functions = {'SCROLLDOWN': '#scroll down the page: \nSCROLL-DOWN-GOOGLE<!|session: str|!>'}
        return

    def ModuleInfo(self):
        return {'NAME': 'google', 'ACTIONS': {'GOOGLE': {'func': 'Google', 'prompt': 'Use google to search internet content.', 'type': 'primary'}, 'SCROLL-DOWN-GOOGLE': {'func': 'ScrollDown', 'prompt': 'Scroll down the results.', 'type': 'supportive'}}}

    def GetSessionID(self) -> str:
        id = f'session-{str(random.randint(0, 99999999))}'
        while id in self.sessions:
            id = f'session-{str(random.randint(0, 99999999))}'
        return id

    def Google(self, keywords: str) -> str:
        try:
            res = search(keywords, num_results=20, advanced=True, sleep_interval=5)
            results = list(res)
            ret = str(results) if len(results) > 0 else 'No search results were found. Please check if you used overly complex keywords or unsupported search syntax. Note that relaxing your search terms is an effective strategy when no valid search results are returned.'
        except Exception as e:
            print('google excetption: ', e)
            ret = f'google excetption: {str(e)}'
        session = self.GetSessionID()
        self.sessions[session] = AScrollablePage(functions=self.functions)
        self.sessions[session].LoadPage(str(ret), 'TOP')
        return self.sessions[session]() + '\n\n' + f'Session name: "{session}"\n'

    def ScrollDown(self, session: str) -> str:
        return self.sessions[session].ScrollDown() + '\n\n' + f'Session name: "{session}"\n'

