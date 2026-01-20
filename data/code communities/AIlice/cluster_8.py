# Cluster 8

@app.route('/get_announcement', methods=['GET'])
def get_announcement():
    logger.debug('Announcement requested')
    return jsonify({'announcement': ''})

@app.route('/list_agent_type', methods=['GET'])
def list_agent_type():
    logger.debug('Agent type list requested')
    return jsonify(['main', 'researcher', 'search-engine', 'coder-proxy', 'coder', 'doc-reader', 'module-coder', 'chat'])

@app.route('/current_session', methods=['GET'])
def current_session():
    logger.debug('Current session requested')
    try:
        return jsonify(context.CurrentSession().Description())
    except AWExceptionSessionNotExist as e:
        return jsonify({})

@app.route('/message', methods=['POST'])
def message():
    message = request.form.get('message', '')
    if 'files_for_perception[]' in request.files:
        message = f'{message}\n\n<!-- The following audio/video/image files are marked with extended markdown syntax for your analysis -->'
        files_for_perception = request.files.getlist('files_for_perception[]')
        for file in files_for_perception:
            if file and file.filename:
                file_path = context.CurrentSession().UploadFile(file.filename, file.read(), file.content_type)
                message = f'{message}\n\n![{file_path}]({file_path})'
    if 'files_for_processing[]' in request.files:
        message = f'{message}\n\n<!-- Here are other attached files. To make the agent actually read the file content (not just see the path), mark it with extended markdown syntax. Check file size first to avoid consuming too many tokens on large files. -->\n\nAttached files:'
        files_for_processing = request.files.getlist('files_for_processing[]')
        for file in files_for_processing:
            if file and file.filename:
                file_path = context.CurrentSession().UploadFile(file.filename, file.read(), file.content_type)
                message = f'{message}\n\ntype: {file.content_type}   length: {file.content_length}   path: {file_path}'
    logger.info(f'Chat message received')
    logger.debug(f'Message content: {message[:50]}{('...' if len(message) > 50 else '')}\nfiles_for_perception: {(len(files_for_perception) if 'files_for_perception' in locals() else 0)}\nfiles_for_processing: {(len(files_for_processing) if 'files_for_processing' in locals() else 0)}')
    context.CurrentSession().Message(message)
    return ('', 204)

@app.route('/delete_history/<string:sessionName>', methods=['DELETE'])
def delete_history(sessionName):
    if context.DeleteSession(sessionName):
        logger.info(f"Chat history '{sessionName}' deleted")
        return ('', 204)
    else:
        logger.warning(f"Failed to delete chat history '{sessionName}': History not found")
        return (jsonify({'error': 'History item not found'}), 404)

@app.route('/list_histories')
def list_histories():
    logger.debug(f'Listing chat histories')
    return jsonify(context.ListSessions())

@app.route('/sendmsg', methods=['POST'])
def sendmsg():
    message = request.form.get('message', '')
    if 'files_for_perception[]' in request.files:
        message = f'{message}\n\n<!-- The following audio/video/image files are marked with extended markdown syntax for your analysis -->'
        files_for_perception = request.files.getlist('files_for_perception[]')
        for file in files_for_perception:
            if file and file.filename:
                file_path = context.CurrentSession().UploadFile(file.filename, file.read(), file.content_type)
                message = f'{message}\n\n![{file_path}]({file_path})'
    if 'files_for_processing[]' in request.files:
        message = f'{message}\n\n<!-- Here are other attached files. To make the agent actually read the file content (not just see the path), mark it with extended markdown syntax. Check file size first to avoid consuming too many tokens on large files. -->\n\nAttached files:'
        files_for_processing = request.files.getlist('files_for_processing[]')
        for file in files_for_processing:
            if file and file.filename:
                file_path = context.CurrentSession().UploadFile(file.filename, file.read(), file.content_type)
                message = f'{message}\n\ntype: {file.content_type}   length: {file.content_length}   path: {file_path}'
    context.CurrentSession().SendMsg(message)
    logger.info(f'Message sent')
    logger.debug(f'Message content: {message[:50]}{('...' if len(message) > 50 else '')}\nfiles_for_perception: {(len(files_for_perception) if 'files_for_perception' in locals() else 0)}\nfiles_for_processing: {(len(files_for_processing) if 'files_for_processing' in locals() else 0)}')
    return jsonify({'status': 'message sent', 'message': message})

@app.route('/get_settings', methods=['POST'])
def get_settings():
    patches = request.get_json().get('patches')
    logger.debug(f'Settings requested')
    return jsonify({'schema': context.Setup(patches, apply=False)})

class UserContext:
    states = ['init', 'ready', 'released']
    settings = ['agentModelConfig', 'models', 'temperature', 'contextWindowRatio']
    allowedPathes = [['agentModelConfig'], ['models', ['oai', 'groq', 'openrouter', 'apipie', 'deepseek', 'mistral', 'anthropic'], 'apikey'], ['temperature'], ['contextWindowRatio']]

    def __init__(self, userID: str):
        self.userID = userID
        self.config = AConfig()
        self.currentSession = None
        self.context = dict()
        self.speech = None
        self.methodLock = threading.RLock()
        self.machine = LockedMachine(model=self, states=UserContext.states, initial='init')
        self.machine.add_transition(trigger='create', source='init', dest='ready')
        self.machine.add_transition(trigger='release', source='*', dest='released')
        self.machine.add_transition(trigger='session_call', source='ready', dest='ready')
        self.serverPublicFile = None
        self.serverSecretFile = None
        self.serverPublicFile = None
        self.serverSecretFile = None
        logger.debug(f'UserContext initialized for user ID: {userID}')
        return

    def GetPath(self, pathType: str='', sessionName: str=''):
        pathes = {'': '', 'user_config': 'user_config.json', 'certificates': 'certificates', 'sessions': 'sessions', 'session': f'sessions/{sessionName}', 'history': f'sessions/{sessionName}/ailice_history.json'}
        return os.path.join(self.config.chatHistoryPath, str(self.userID), pathes[pathType])

    def StoreConfig(self):
        userCfg = {'agentModelConfig': self.config.agentModelConfig, 'models': {providerName: {k: v for k, v in providerCfg.items() if k != 'modelList'} for providerName, providerCfg in self.config.models.items() if providerName not in ['default']}, 'temperature': self.config.temperature, 'contextWindowRatio': self.config.contextWindowRatio}
        config_path = self.GetPath(pathType='user_config')
        with open(config_path, 'w') as f:
            json.dump(userCfg, f, indent=2)
        logger.info(f'User configuration stored to {config_path}')
        return

    def UpdateConfig(self, updatedConfig):
        logger.info(f'Updating configuration for user {self.userID}')
        if 'agentModelConfig' in updatedConfig:
            self.config.agentModelConfig = updatedConfig['agentModelConfig']
            logger.debug('Updated agentModelConfig')
        if 'models' in updatedConfig:
            updateModels = {providerName: providerCfg for providerName, providerCfg in updatedConfig['models'].items() if providerName not in ['default']}
            for providerName in updateModels:
                updateModels[providerName]['modelList'] = self.config.models[providerName]['modelList']
            self.config.models.update(updateModels)
            logger.debug(f'Updated models configuration for providers: {list(updateModels.keys())}')
        if 'temperature' in updatedConfig:
            self.config.temperature = float(updatedConfig['temperature'])
            logger.debug(f'Updated temperature to {self.config.temperature}')
        if 'contextWindowRatio' in updatedConfig:
            self.config.contextWindowRatio = float(updatedConfig['contextWindowRatio'])
            logger.debug(f'Updated contextWindowRatio to {self.config.contextWindowRatio}')
        return

    def InitConfig(self):
        logger.info(f'Initializing configuration for user {self.userID}')
        self.config.__dict__.update(copy.deepcopy(global_config.ToJson()))
        configFile = self.GetPath(pathType='user_config')
        userConfig = {}
        if os.path.exists(configFile):
            with open(configFile, 'r') as f:
                userConfig = json.load(f)
                logger.info(f'Loaded user configuration from {configFile}')
        else:
            logger.info(f'No user configuration found at {configFile}, using defaults')
        self.UpdateConfig(userConfig)
        return

    @atomic_transition('create')
    def Create(self):
        logger.info(f'Creating user context for user {self.userID}')
        os.makedirs(self.GetPath(), exist_ok=True)
        os.makedirs(self.GetPath('sessions'), exist_ok=True)
        self.InitConfig()
        self.serverPublicFile, self.serverSecretFile = GenerateCertificates(self.GetPath(pathType='certificates'), 'server')
        self.clientPublicFile, self.clientSecretFile = GenerateCertificates(self.GetPath(pathType='certificates'), 'client')
        logger.info('Certificates generated')
        return

    @atomic_transition('release')
    def Release(self):
        logger.info(f'Releasing user context for user {self.userID}')
        for sessionName, session in self.context.items():
            logger.info(f'Releasing session {sessionName}')
            session.Stop()
            cleaner.AddSessionToGC(sessionName, session)
        self.context.clear()
        return

    @atomic_transition('session_call')
    def Setup(self, patches: list, apply=False) -> dict:
        updatedConfig = self.config.__dict__
        if patches is not None and type(patches) is list and (len(patches) > 0):
            logger.info(f'Setting up user context with patches: {patches}')
            try:
                validatedPatches = validate_patches(patches=patches)
                logger.debug('Patches validated')
            except Exception as e:
                logger.error(f'Setup() Exception: Invalid patches input. {patches}', exc_info=True)
                raise AWExceptionIllegalInput()
            if not all([any([check_path(p['path'], pattern) for pattern in UserContext.allowedPathes]) for p in patches]):
                logger.error(f'Setup() Exception. Invalid path input: {patches}')
                raise AWExceptionIllegalInput()
            updatedConfig = apply_patches(self.config.__dict__, validatedPatches)
            logger.debug('Patches applied to configuration')
            try:
                AiliceWebConfig.model_validate({k: v for k, v in updatedConfig.items() if k in UserContext.settings})
                logger.debug('Configuration validated')
            except Exception as e:
                logger.error(f'Configuration validation failed: {str(e)}', exc_info=True)
                raise
            for k, v in updatedConfig.items():
                if 'agentModelConfig' == k:
                    modelIDs = [f'{modelType}:{model}' for modelType in self.config.models for model in self.config.models[modelType]['modelList']]
                    if any([mid not in modelIDs for agentType, mid in v.items()]):
                        logger.error(f'Setup() Exception. Invalid modelID input: {patches}')
                        raise AWExceptionIllegalInput()
            if apply:
                self.UpdateConfig(updatedConfig)
                self.StoreConfig()
                for sessionName, session in self.context.items():
                    logger.info(f'Releasing session {sessionName} due to configuration update')
                    session.Stop()
                    cleaner.AddSessionToGC(sessionName, session)
                self.context.clear()
                sessionName = self.currentSession
                self.currentSession = None
                if sessionName is not None:
                    logger.info(f'Reloading current session {sessionName}')
                    self.Load(sessionName)
        ret = {}
        for k in UserContext.settings:
            if k == 'models':
                ret[k] = {provider: {'modelWrapper': providerCfg['modelWrapper'], 'apikey': None, 'baseURL': None, 'modelList': providerCfg['modelList']} if provider in ['default'] else providerCfg for provider, providerCfg in (self.config.__dict__[k].items() if apply else updatedConfig[k].items())}
            else:
                ret[k] = self.config.__dict__[k] if apply else updatedConfig[k]
        logger.debug('Settings schema built')
        return build_settings_schema(ret)

    @atomic_transition('session_call')
    def CurrentSession(self):
        if not self.currentSession:
            logger.warning(f'No current session for user {self.userID}')
            raise AWExceptionSessionNotExist()
        logger.debug(f'Returning current session {self.currentSession}')
        return self.context[self.currentSession]

    def Load(self, sessionName: str):
        logger.info(f'Loading session {sessionName} for user {self.userID}')
        try:
            if sessionName == self.currentSession:
                logger.info(f'Session {sessionName} already loaded, return now.')
                return
            logger.info(f'Release session {self.currentSession} at {self.GetPath(pathType='session', sessionName=self.currentSession)}')
            if self.currentSession in self.context:
                self.context[self.currentSession].Stop()
                cleaner.AddSessionToGC(self.currentSession, self.context[self.currentSession])
                self.context.pop(self.currentSession)
            self.currentSession = None
            sessionPath = self.GetPath(pathType='session', sessionName=sessionName)
            logger.info(f'Creating new TaskSession for {sessionName} at {sessionPath}')
            if cleaner.IsSessionInGC(sessionName) and (not time.sleep(5)) and cleaner.IsSessionInGC(sessionName):
                raise AWExceptionSessionBusy()
            self.context[sessionName] = TaskSession(sessionName, sessionPath, self.clientSecretFile, self.serverPublicFile)
            self.context[sessionName].Create(config=self.config)
            self.currentSession = sessionName
            logger.info(f'Session {sessionName} loaded successfully')
        except Exception as e:
            logger.error(f'Exception loading session {sessionName}: {str(e)}, currentSession: {self.currentSession}', exc_info=True)
            if hasattr(e, 'tb'):
                logger.error(e.tb)
            if sessionName in self.context:
                logger.info(f'Cleaning up failed session {sessionName}')
                self.context[sessionName].Stop()
                cleaner.AddSessionToGC(sessionName, self.context[sessionName])
                self.context.pop(sessionName)
                raise e
        return

    @atomic_transition('session_call')
    def NewSession(self) -> str:
        sessionName = 'ailice_' + str(int(time.time()))
        logger.info(f'Creating new session {sessionName} for user {self.userID}')
        self.Load(sessionName=sessionName)
        return sessionName

    @atomic_transition('session_call')
    def LoadSession(self, sessionName: str):
        logger.info(f'Loading session history for {sessionName}')
        sessions_dir = self.GetPath(pathType='sessions')
        if sessionName not in os.listdir(sessions_dir):
            logger.error(f'Session {sessionName} not found in {sessions_dir}')
            raise AWExceptionSessionNotExist()
        needLoading = sessionName != self.currentSession
        if needLoading:
            logger.info(f'Session {sessionName} is not current, loading it')
            self.Load(sessionName=sessionName)
        return

    @atomic_transition('session_call')
    def GetSession(self, sessionName: str):
        logger.info(f'Getting session history for {sessionName}')
        sessions_dir = self.GetPath(pathType='sessions')
        if sessionName not in os.listdir(sessions_dir):
            logger.error(f'Session {sessionName} not found in {sessions_dir}')
            raise AWExceptionSessionNotExist()

        def historyFilter(data):
            conversations = [(f'{conv['role']}_{data['name']}', conv['msg']) for conv in data['conversation']]
            ret = {'conversation': conversations}
            if 'subProcessors' in data:
                ret['subProcessors'] = {agentName: historyFilter(subProcessor) for agentName, subProcessor in data['subProcessors'].items()}
            else:
                ret['subProcessors'] = {}
            return ret
        historyPath = self.GetPath(pathType='history', sessionName=sessionName)
        if os.path.exists(historyPath):
            with open(historyPath, 'r') as f:
                data = json.load(f)
                conversations = historyFilter(data)
                logger.info(f'Got {len(conversations)} conversation entries from {historyPath}')
        else:
            logger.info(f'No history file found at {historyPath}, returning empty conversation')
            conversations = {}
        return conversations

    @atomic_transition('session_call')
    def DeleteSession(self, sessionName: str) -> bool:
        logger.info(f'Deleting session {sessionName} for user {self.userID}')
        if sessionName in self.context:
            logger.info(f'Releasing active session {sessionName}')
            self.context[sessionName].Stop()
            cleaner.AddSessionToGC(sessionName, self.context[sessionName])
            self.context.pop(sessionName)
            self.currentSession = None if sessionName == self.currentSession else self.currentSession
        sessions_dir = self.GetPath(pathType='sessions')
        if sessionName not in os.listdir(sessions_dir):
            logger.warning(f'Session {sessionName} not found in {sessions_dir}')
            return False
        historyDir = self.GetPath(pathType='session', sessionName=sessionName)
        shutil.rmtree(historyDir)
        logger.info(f'Deleted session directory {historyDir}')
        return True

    @atomic_transition('session_call')
    def ListSessions(self):
        logger.info(f'Listing sessions for user {self.userID}')
        histories = []
        sessions_dir = self.GetPath(pathType='sessions')
        for d in os.listdir(sessions_dir):
            p = self.GetPath(pathType='history', sessionName=d)
            if os.path.exists(p) and os.path.getsize(p) > 0:
                with open(p, 'r') as f:
                    try:
                        content = json.load(f)
                        if len(content.get('conversation', [])) > 0:
                            histories.append((d, content.get('conversation')[0]['msg']))
                    except Exception as e:
                        logger.error(f'Error loading history file {p}: {str(e)}', exc_info=True)
                        continue
        sorted_histories = sorted(histories, key=lambda x: os.path.getmtime(self.GetPath(pathType='history', sessionName=x[0])), reverse=True)
        logger.info(f'Found {len(sorted_histories)} sessions')
        return sorted_histories

class TaskSession:
    states = ['init', 'ready', 'generating', 'interrupted', 'stopping']

    def __init__(self, sessionName, sessionPath, clientSecretFile, serverPublicFile):
        self.sessionName = sessionName
        self.title = None
        self.sessionPath = sessionPath
        self.clientSecretFile = str(clientSecretFile)
        self.serverPublicFile = str(serverPublicFile)
        self.config = None
        self.threadLLM = None
        self.clientPool = None
        self.processor = None
        self.llmPool = None
        self.messenger = None
        self.alogger = None
        self.stopFlag = False
        self.streamNum = 0
        self.machine = LockedMachine(model=self, states=TaskSession.states, initial='init')
        self.machine.add_transition(trigger='create', source='init', dest='ready')
        self.machine.add_transition(trigger='stop', source='*', dest='stopping')
        self.machine.add_transition(trigger='release', source='stopping', dest='init')
        self.machine.add_transition(trigger='upload', source=['ready', 'interrupted'], dest='=')
        self.machine.add_transition(trigger='input', source='ready', dest='generating')
        self.machine.add_transition(trigger='interrupt', source=['generating', 'interrupted'], dest='interrupted')
        self.machine.add_transition(trigger='sendmsg', source='interrupted', dest='generating')
        self.machine.add_transition(trigger='answered', source='generating', dest='ready')
        self.machine.add_transition(trigger='dump', source=['ready', 'stopping'], dest='=')
        self.machine.add_transition(trigger='desc', source='*', dest='=')
        self.machine.add_transition(trigger='stream', source='*', dest='=')
        self.methodLock = threading.RLock()
        logger.debug(f'TaskSession initialized: {sessionName}')
        return

    @atomic_transition('dump')
    def Save(self):
        if self.sessionPath is not None and self.processor is not None:
            try:
                history_path = os.path.join(self.sessionPath, 'ailice_history.json')
                with open(history_path, 'w') as f:
                    json.dump(self.processor.ToJson(), f, indent=2)
                logger.info(f'Saved history to {history_path}')
            except Exception as e:
                logger.error(f'TaskSession::Save() Exception: Dump history FAILED. {str(e)}', exc_info=True)
        return

    @atomic_transition('create')
    def Create(self, config):
        sessionName, sessionPath = (self.sessionName, self.sessionPath)
        self.config = config
        logger.info(f'Creating session {sessionName} at {sessionPath}')
        os.makedirs(sessionPath, exist_ok=True)
        os.makedirs(os.path.join(sessionPath, 'storage'), exist_ok=True)
        os.makedirs(os.path.join(sessionPath, 'uploads'), exist_ok=True)
        history = None
        p = os.path.join(sessionPath, 'ailice_history.json')
        if os.path.exists(p):
            with open(p, 'r') as f:
                if os.path.getsize(p) > 0:
                    history = json.load(f)
                    logger.info(f'Loaded history from {p}')
                else:
                    logger.info(f'Empty history file: {p}')
        self.CreateAilice(history=history)
        logger.info(f'Session {sessionName} created successfully')
        return

    def CreateAilice(self, history: typing.Optional[dict]=None):
        sessionName, sessionPath = (self.sessionName, self.sessionPath)
        self.clientPool = AClientPool()
        for i in range(5):
            try:
                self.clientPool.Init()
                logger.info('Client pool initialized successfully')
                break
            except Exception as e:
                if i == 4:
                    error_msg = f'It seems that some peripheral module services failed to start. EXCEPTION: {str(e)}'
                    logger.error(error_msg, exc_info=True)
                    if hasattr(e, 'tb'):
                        logger.error(e.tb)
                    else:
                        logger.error(traceback.format_exc())
                time.sleep(5)
                continue
        self.InitSpeech(self.clientPool)
        logger.info('Starting vector database. This may include downloading model weights.')
        storage = self.clientPool.GetClient(self.config.services['storage']['addr'])
        with storage.Timeout(-1):
            msg = storage.Open(os.path.join(sessionPath, 'storage'))
        logger.info(f'Storage response: {msg}')
        llmPool = ALLMPool(config=self.config)
        llmPool.Init([self.config.modelID])
        logger.info(f'LLM Pool initialized with model: {self.config.modelID}')
        promptsManager = APromptsManager()
        promptsManager.Init(storage=storage, collection=sessionName)
        promptsManager.RegisterPrompts([APromptChat, APromptMain, APromptSearchEngine, APromptResearcher, APromptCoder, APromptModuleCoder, APromptCoderProxy, APromptDocReader])
        logger.info('Prompts manager initialized and prompts registered')
        messenger = AMessenger()
        alogger = KMsgQue()
        alogger.Load(ExtractMessages(history, 0))
        processor = AProcessor(name='AIlice', modelID=self.config.modelID, promptName=self.config.prompt, llmPool=llmPool, promptsManager=promptsManager, services=self.clientPool, messenger=messenger, outputCB=alogger.Receiver, gasTank=AGasTank(100000000.0), config=self.config, collection=sessionName)
        moduleList = [self.config.services['arxiv']['addr'], self.config.services['google']['addr'], self.config.services['duckduckgo']['addr'], self.config.services['browser']['addr'], self.config.services['scripter']['addr'], self.config.services['computer']['addr']]
        if self.config.speechOn:
            moduleList.append(self.config.services['speech']['addr'])
        processor.RegisterModules(moduleList)
        logger.info(f'Processor initialized with {len(moduleList)} modules')
        if history is not None:
            processor.FromJson(history)
            self.title = history['conversation'][0]['msg'] if len(history['conversation']) > 0 else 'New Chat'
            logger.info(f'History Loaded.')
        self.processor = processor
        self.llmPool = llmPool
        self.messenger = messenger
        self.alogger = alogger
        logger.info(f'Session {sessionName} created successfully')
        return

    @atomic_transition('stop')
    def Stop(self):
        try:
            logger.info(f'Stopping session {self.sessionName}')
            self.stopFlag = True
            if self.threadLLM is not None and self.threadLLM.is_alive():
                logger.info('Stopping active LLM thread')
                self.messenger.Lock()
                self.messenger.Put('/stop')
                self.messenger.Unlock()
            logger.info(f'Session {str(self.sessionName)} stop triggered')
        except Exception as e:
            logger.error(f'Session {str(self.sessionName)} stop FAILED. {str(e)}\n\ntraceback: {traceback.format_exc()}')
            pass
        return

    @atomic_transition('release')
    def Release(self):
        try:
            logger.info(f'Releasing session {self.sessionName}')
            if self.threadLLM is not None and self.threadLLM.is_alive():
                logger.info(f'Thread is still alive while releasing session {self.sessionName}, TERMINATE NOW.')
                StopThread(self.threadLLM)
            if self.clientPool is not None:
                logger.info(f'Destroy client pool.')
                self.clientPool.Destroy()
            self.threadLLM = None
            self.processor = None
            self.llmPool = None
            self.messenger = None
            self.alogger = None
            self.config = None
            self.title = None
            logger.info(f'Session {str(self.sessionName)} released')
        except Exception as e:
            logger.error(f'Session {str(self.sessionName)} release FAILED. {str(e)}\n\ntraceback: {traceback.format_exc()}')
            pass
        return

    def IsStopped(self) -> bool:
        return (self.threadLLM is None or not self.threadLLM.is_alive()) and 0 == self.streamNum

    def InitSpeech(self, clientPool):
        if self.config.speechOn:
            import sounddevice as sd
            logger.info('Initializing speech module')
            self.speech = clientPool.GetClient(self.config.services['speech']['addr'])
            logger.info('Preparing speech recognition and TTS models. This may include downloading weight data.')
            with self.speech.Timeout(-1):
                self.speech.PrepareModel()
            logger.info('Speech module model preparation completed')
            if any([re.fullmatch('(cuda|cpu)(:(\\d+))?', s) == None for s in [self.config.ttsDevice, self.config.sttDevice]]):
                error_msg = 'The value of ttsDevice and sttDevice should be a valid cuda device, such as cuda, cuda:0, or cpu, the default is cpu.'
                logger.error(error_msg)
                exit(-1)
            else:
                self.speech.SetDevices({'tts': self.config.ttsDevice, 'stt': self.config.sttDevice})
                logger.info(f'Speech devices set: TTS={self.config.ttsDevice}, STT={self.config.sttDevice}')
        else:
            self.speech = None
            logger.debug('Speech module not enabled')
        return

    @atomic_transition('desc')
    def Description(self) -> dict[str, str]:
        logger.info(f'Get decription of session {self.sessionName}')
        return {'sessionName': self.sessionName, 'title': self.title, 'state': self.state}

    @atomic_transition('stream')
    def MsgStream(self) -> typing.Generator:
        if self.stopFlag:
            return
        self.streamNum += 1
        try:
            buffer = self.alogger.Get(getBuffer=True)
            if len(buffer) > 0:
                yield f'data: {json.dumps(buffer)}\n\n'
            while not self.stopFlag:
                try:
                    msg = self.alogger.Get(timeout=1)
                except Empty as e:
                    continue
                yield f'data: {json.dumps(msg)}\n\n'
        except Exception as e:
            error_msg = {'error': f'Stream error: {e}', 'type': 'error'}
            logger.error(f'Stream error: {e}', exc_info=True)
            yield f'data: {json.dumps(error_msg)}\n\n'
        finally:
            self.streamNum -= 1

    @atomic_transition('input')
    def Message(self, msg: str):
        logger.info(f'Chat request received in session {self.sessionName}')
        logger.debug(f'Message content: {msg[:50]}...' if len(msg) > 50 else f'Message content: {msg}')
        if self.title in [None, 'New Chat']:
            self.title = msg
        logger.info(f'Generating response for message in session {self.sessionName}')

        def response(msg: str):
            try:
                self.processor(msg)
            except Exception as e:
                pass
            finally:
                if 'generating' == self.state:
                    self.answered()
                self.Save()
        self.processor.SetGas(100000000.0)
        self.threadLLM = threading.Thread(target=response, args=(msg,))
        self.threadLLM.start()
        return

    @atomic_transition('upload')
    def UploadFile(self, fileName: str, fileData: bytes, contentType: str) -> str:
        filename = secure_filename(fileName)
        filepath = os.path.join(self.sessionPath, 'uploads', filename)
        with open(filepath, 'wb') as f:
            f.write(fileData)
        logger.info(f"{contentType} file '{filename}' uploaded to {filepath}")
        return filepath

    @atomic_transition('interrupt')
    def Interrupt(self):
        logger.info(f'Interrupting session {self.sessionName}')
        self.messenger.Lock()
        return

    @atomic_transition('sendmsg')
    def SendMsg(self, msg: str):
        logger.info(f'Sending message to session {self.sessionName}')
        logger.debug(f'Message content: {msg[:50]}...' if len(msg) > 50 else f'Message content: {msg}')
        self.messenger.Put(msg)
        self.messenger.Unlock()
        return

    def Proxy(self, href: str, method: str) -> typing.Generator:
        logger.debug(f'Proxy request for {href} with method {method}')
        var = self.processor.interpreter.env.get(href, None)
        if var and type(var).__name__ in ['AImage', 'AVideo']:
            logger.debug(f'Proxy request resolved as variable of type {type(var).__name__}')
            yield {'type': 'variable', 'data': var}
        else:
            logger.debug(f'Proxy request forwarded to computer module')
            computer = self.processor.services['computer']
            res = computer.Proxy(href, method)
            responseInfo = next(res)
            yield {'type': 'href', 'responseInfo': responseInfo}
            yield from res

