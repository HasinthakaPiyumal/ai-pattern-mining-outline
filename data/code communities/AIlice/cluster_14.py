# Cluster 14

class AClientPool:

    def __init__(self):
        self.pool = dict()
        return

    def Init(self):
        for serviceName, cfg in config.services.items():
            if not config.speechOn and 'speech' == serviceName:
                continue
            try:
                self.pool[cfg['addr']] = {'name': serviceName, 'client': makeClient(url=cfg['addr'], clientPrivateKeyPath=cfg.get('clientPrivateKeyPath', None), serverPublicKeyPath=cfg.get('serverPublicKeyPath', None))}
            except Exception as e:
                print(f'Connecting module {serviceName} FAILED. You can try running the module manually and observe its error messages. EXCEPTION: {str(e)}')
                raise e
        return

    def GetClient(self, moduleAddr: str, clientPrivateKeyPath=None, serverPublicKeyPath=None):
        if moduleAddr not in self.pool:
            self.pool[moduleAddr] = {'client': makeClient(url=moduleAddr, clientPrivateKeyPath=clientPrivateKeyPath, serverPublicKeyPath=serverPublicKeyPath)}
            self.pool[moduleAddr]['name'] = self.pool[moduleAddr]['client'].ModuleInfo()['NAME']
        return self.pool[moduleAddr]['client']

    def __getitem__(self, key: str):
        for addr, client in self.pool.items():
            if key == client['name']:
                return client['client']
        return None

    def Destroy(self):
        for _, client in self.pool.items():
            destroy = getattr(client['client'], 'Destroy', None)
            if callable(destroy):
                try:
                    destroy()
                    destroyClient(client['client'])
                except Exception as e:
                    print(f'AClientPool.Destroy Exception: {str(e)}')
                    continue
        self.pool.clear()
        return

def makeClient(url, returnClass=False, clientPrivateKeyPath=None, serverPublicKeyPath=None, validateReturn=True, timeout=300 * 1000, retries=3):
    clientPrivateKeyPath = clientPrivateKeyPath or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'certificates/client/client.key_secret')
    serverPublicKeyPath = serverPublicKeyPath or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'certificates/server/server.key')
    enableSecurity = os.path.exists(serverPublicKeyPath) and os.path.exists(clientPrivateKeyPath)
    if enableSecurity:
        print('lightRPC client encryption ENABLED.')
        clientPublic, clientSecret = zmq.auth.load_certificate(clientPrivateKeyPath)
        serverPublic, _ = zmq.auth.load_certificate(serverPublicKeyPath)

    class RemoteGenerator:

        def __init__(self, client, generatorID):
            self.client = client
            self.generatorID = generatorID

        def __iter__(self):
            return self

        def __next__(self):
            ret = self.client.Send({'NEXT': '', 'clientID': self.client.clientID, 'generatorID': self.generatorID})
            if 'exception' in ret:
                raise ALightRPCException(ret['exception'])
            if ret['finished']:
                raise StopIteration
            return ret['ret']

    class GenesisRPCClientTemplate(object):

        def __init__(self):
            self.url = url
            self.context = context
            self.enableSecurity = enableSecurity
            self.timeout = timeout
            self.retries = retries
            if self.enableSecurity:
                self.clientPublic, self.clientSecret, self.serverPublic = (clientPublic, clientSecret, serverPublic)
            ret = self.Send({'CREATE': ''})
            if 'exception' in ret:
                raise ALightRPCException(ret['exception'])
            self.clientID = ret['clientID']
            return

        def Send(self, msg):
            for attempt in range(self.retries):
                try:
                    with self.context.socket(zmq.REQ) as socket:
                        if self.enableSecurity:
                            socket.setsockopt(zmq.CURVE_PUBLICKEY, self.clientPublic)
                            socket.setsockopt(zmq.CURVE_SECRETKEY, self.clientSecret)
                            socket.setsockopt(zmq.CURVE_SERVERKEY, self.serverPublic)
                        socket.setsockopt(zmq.CONNECT_TIMEOUT, 10000)
                        socket.setsockopt(zmq.HEARTBEAT_IVL, 2000)
                        socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 10000)
                        socket.setsockopt(zmq.RCVTIMEO, self.timeout)
                        socket.setsockopt(zmq.SNDTIMEO, self.timeout)
                        socket.connect(self.url)
                        SendMsg(socket, msg)
                        return ReceiveMsg(socket)
                except zmq.Again:
                    if attempt == self.retries - 1:
                        raise TimeoutError(f'RPC call timeout after {self.retries} retries')
                    continue
                except Exception as e:
                    if attempt == self.retries - 1:
                        raise
                    continue

        def RemoteCall(self, funcName, args, kwargs):
            ret = self.Send({'clientID': self.clientID, 'function': funcName, 'args': args, 'kwargs': kwargs})
            if 'exception' in ret:
                raise ALightRPCException(ret['exception'])
            if isinstance(ret['ret'], dict) and 'generatorID' in ret['ret']:
                return RemoteGenerator(self, ret['ret']['generatorID'])
            return ret['ret']

        @contextmanager
        def Timeout(self, timeoutTemp=-1):
            try:
                t = self.timeout
                self.timeout = timeoutTemp
                yield self
            finally:
                self.timeout = t
    with context.socket(zmq.REQ) as socket:
        if enableSecurity:
            socket.setsockopt(zmq.CURVE_PUBLICKEY, clientPublic)
            socket.setsockopt(zmq.CURVE_SECRETKEY, clientSecret)
            socket.setsockopt(zmq.CURVE_SERVERKEY, serverPublic)
        socket.setsockopt(zmq.CONNECT_TIMEOUT, 10000)
        socket.setsockopt(zmq.SNDTIMEO, 10000)
        socket.setsockopt(zmq.RCVTIMEO, 10000)
        socket.connect(url)
        SendMsg(socket, {'GET_META': ''})
        ret = ReceiveMsg(socket)
    for funcName, methodMeta in ret['META']['methods'].items():
        AddMethod(GenesisRPCClientTemplate, funcName, methodMeta)
    return validate_methods(GenesisRPCClientTemplate, None, validateReturn) if returnClass else validate_methods(GenesisRPCClientTemplate, None, validateReturn)()

class GenesisRPCServer(object):

    def __init__(self, objCls, objArgs, url, APIList, serverPrivateKeyPath=None, clientPublicKeysDir=None, validateReturn=True, atomicCall=True):
        global auth, authLock
        self.objCls = validate_methods(objCls, APIList, validateReturn)
        self.objArgs = objArgs
        self.url = url
        self.objPool = dict()
        self.objPoolLock = threading.Lock()
        self.APIList = APIList
        self.atomicCall = atomicCall
        self.context = context
        self.domain = str(uuid.uuid4())
        self.WORKERS_ADDR = f'inproc://workers-{self.domain}'
        self.receiver = self.context.socket(zmq.ROUTER)
        if serverPrivateKeyPath is None:
            serverPrivateKeyPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'certificates/server/server.key_secret')
            clientPublicKeysDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'certificates/client/')
            if not os.path.exists(clientPublicKeysDir):
                clientPublicKeysDir = None
        self.enableSecurity = os.path.exists(serverPrivateKeyPath)
        if self.enableSecurity:
            print('lightRPC server encryption ENABLED.')
            with authLock:
                if auth is None:
                    auth = ThreadAuthenticator(context)
                    auth.start()
                auth.configure_curve(domain=self.domain, location=zmq.auth.CURVE_ALLOW_ANY if clientPublicKeysDir is None else clientPublicKeysDir)
            serverPublic, serverSecret = zmq.auth.load_certificate(serverPrivateKeyPath)
            self.receiver.setsockopt_string(zmq.ZAP_DOMAIN, self.domain)
            self.receiver.setsockopt(zmq.CURVE_PUBLICKEY, serverPublic)
            self.receiver.setsockopt(zmq.CURVE_SECRETKEY, serverSecret)
            self.receiver.setsockopt(zmq.CURVE_SERVER, True)
        self.receiver.bind(url)
        self.dealer = self.context.socket(zmq.DEALER)
        self.dealer.bind(self.WORKERS_ADDR)
        return

    def Run(self):
        try:
            for i in range(16):
                thread = threading.Thread(target=self.Worker, name='RPC-Worker-%d' % (i + 1))
                thread.daemon = True
                thread.start()
            zmq.device(zmq.QUEUE, self.receiver, self.dealer)
        except Exception as e:
            print('GenesisRPCServer:Run() FATAL EXCEPTION. ', self.url, ', ', str(e))
            sys.exit(1)
        finally:
            self.receiver.close()
            self.dealer.close()

    def CreateSocket(self):
        socket = self.context.socket(zmq.DEALER)
        socket.setsockopt(zmq.HEARTBEAT_IVL, 2000)
        socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 10000)
        socket.connect(self.WORKERS_ADDR)
        return socket

    def Worker(self):
        socket = self.CreateSocket()
        while True:
            try:
                frames = socket.recv_multipart()
                if len(frames) >= 3:
                    clientIdentity = frames[0]
                    delimiter = frames[1]
                    msgData = frames[2]
                    msg = json.loads(msgData.decode('utf-8'), cls=AJSONDecoder)
                    ret = None
                    try:
                        if 'clientID' in msg and msg['clientID'] not in self.objPool:
                            ret = {'exception': str(KeyError(f'clientID {msg['clientID']} not exist.'))}
                        elif 'GET_META' in msg:
                            methods = inspect.getmembers(self.objCls, predicate=lambda x: inspect.isfunction(x) and x.__name__ in self.APIList)
                            ret = {'META': {'methods': {methodName: {'signature': str(inspect.signature(method)), 'is_generator': inspect.isgeneratorfunction(method)} for methodName, method in methods}}}
                        elif 'CREATE' in msg:
                            with self.objPoolLock:
                                newID = str(secrets.token_hex(64))
                                self.objPool[newID] = GeneratorStorage(self.objCls(**self.objArgs), self.atomicCall)
                            ret = {'clientID': newID}
                        elif 'DEL' in msg:
                            with self.objPoolLock:
                                if msg['clientID'] in self.objPool:
                                    del self.objPool[msg['clientID']]
                                ret = {}
                        elif 'NEXT' in msg:
                            with self.objPoolLock:
                                storageObj = self.objPool[msg['clientID']]
                            try:
                                with storageObj.lock:
                                    ret = {'ret': next(storageObj.GetGenerator(msg['generatorID'])), 'finished': False}
                            except StopIteration:
                                ret = {'ret': None, 'finished': True}
                        else:
                            with self.objPoolLock:
                                storageObj = self.objPool[msg['clientID']]
                            with storageObj.lock:
                                result = getattr(storageObj, msg['function'])(*msg['args'], **msg['kwargs'])
                            if inspect.isgenerator(result):
                                generatorID = str(id(result))
                                with self.objPoolLock:
                                    self.objPool[msg['clientID']].SaveGenerator(generatorID, result)
                                ret = {'ret': {'generatorID': generatorID}}
                            else:
                                ret = {'ret': result}
                    except Exception as e:
                        e.tb = ''.join(traceback.format_tb(e.__traceback__))
                        ret = {'exception': f'{str(e)}\n\n{e.tb}'}
                        traceback.print_tb(e.__traceback__)
                        print('Exception. msg:', str(msg), '. Except:', str(e))
                    if ret is not None:
                        responseData = json.dumps(ret, cls=AJSONEncoder).encode('utf-8')
                        socket.send_multipart([clientIdentity, delimiter, responseData])
                else:
                    print(f'Warning: Received unexpected frame count: {len(frames)}')
            except Exception as e:
                print('Worker exception:', str(e))
                traceback.print_tb(e.__traceback__)
                socket.close()
                socket = self.CreateSocket()
                continue

def validate_methods(cls, methodList=None, validateReturn=True):
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not name.startswith('_') and (methodList is None or name in methodList):
            setattr(cls, name, validate_call(method, validate_return=validateReturn and (not inspect.isgeneratorfunction(method))))
    return cls

def AddMethod(kls, methodName, methodMeta):
    signature = methodMeta['signature']
    is_generator = methodMeta['is_generator']
    newSignature = SignatureFromString(signature)

    def methodTemplate(self, *args, **kwargs):
        return self.RemoteCall(methodName, args, kwargs)
    methodTemplate.__is_generator__ = is_generator
    methodTemplate.__annotations__ = AnnotationsFromSignature(newSignature)
    methodTemplate.__signature__ = newSignature
    setattr(kls, methodName, methodTemplate)

def SignatureFromString(sig_str: str) -> inspect.Signature:
    funcDefNode = ast.parse(f'def f{sig_str}:\n    pass', mode='exec').body[0]

    def BuildArg(arg, kind):
        annotation = BuildTypeFromAST(arg.annotation, TYPE_NAMESPACE) if arg.annotation else inspect.Parameter.empty
        return inspect.Parameter(name=arg.arg, kind=kind, annotation=annotation)
    parameters = []
    for arg in funcDefNode.args.args:
        parameters.append(BuildArg(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD))
    if funcDefNode.args.vararg:
        parameters.append(BuildArg(funcDefNode.args.vararg, inspect.Parameter.VAR_POSITIONAL))
    if funcDefNode.args.kwarg:
        parameters.append(BuildArg(funcDefNode.args.kwarg, inspect.Parameter.VAR_KEYWORD))
    defaults = funcDefNode.args.defaults
    if defaults:
        offset = len(parameters) - len(defaults)
        for i, default in enumerate(defaults):
            try:
                defaultValue = ast.literal_eval(default)
            except (ValueError, SyntaxError):
                defaultValue = inspect.Parameter.empty
            parameters[offset + i] = parameters[offset + i].replace(default=defaultValue)
    returnAnnotation = BuildTypeFromAST(funcDefNode.returns, TYPE_NAMESPACE) if funcDefNode.returns else inspect.Parameter.empty
    return inspect.Signature(parameters=parameters, return_annotation=returnAnnotation)

def AnnotationsFromSignature(signature: inspect.Signature) -> dict:
    annotations = {}
    for param_name, param in signature.parameters.items():
        if param.annotation is not inspect.Parameter.empty:
            annotations[param_name] = param.annotation
    if signature.return_annotation is not inspect.Parameter.empty:
        annotations['return'] = signature.return_annotation
    return annotations

class GenesisRPCClientTemplate(object):

    def __init__(self):
        self.url = url
        self.context = context
        self.enableSecurity = enableSecurity
        self.timeout = timeout
        self.retries = retries
        if self.enableSecurity:
            self.clientPublic, self.clientSecret, self.serverPublic = (clientPublic, clientSecret, serverPublic)
        ret = self.Send({'CREATE': ''})
        if 'exception' in ret:
            raise ALightRPCException(ret['exception'])
        self.clientID = ret['clientID']
        return

    def Send(self, msg):
        for attempt in range(self.retries):
            try:
                with self.context.socket(zmq.REQ) as socket:
                    if self.enableSecurity:
                        socket.setsockopt(zmq.CURVE_PUBLICKEY, self.clientPublic)
                        socket.setsockopt(zmq.CURVE_SECRETKEY, self.clientSecret)
                        socket.setsockopt(zmq.CURVE_SERVERKEY, self.serverPublic)
                    socket.setsockopt(zmq.CONNECT_TIMEOUT, 10000)
                    socket.setsockopt(zmq.HEARTBEAT_IVL, 2000)
                    socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 10000)
                    socket.setsockopt(zmq.RCVTIMEO, self.timeout)
                    socket.setsockopt(zmq.SNDTIMEO, self.timeout)
                    socket.connect(self.url)
                    SendMsg(socket, msg)
                    return ReceiveMsg(socket)
            except zmq.Again:
                if attempt == self.retries - 1:
                    raise TimeoutError(f'RPC call timeout after {self.retries} retries')
                continue
            except Exception as e:
                if attempt == self.retries - 1:
                    raise
                continue

    def RemoteCall(self, funcName, args, kwargs):
        ret = self.Send({'clientID': self.clientID, 'function': funcName, 'args': args, 'kwargs': kwargs})
        if 'exception' in ret:
            raise ALightRPCException(ret['exception'])
        if isinstance(ret['ret'], dict) and 'generatorID' in ret['ret']:
            return RemoteGenerator(self, ret['ret']['generatorID'])
        return ret['ret']

    @contextmanager
    def Timeout(self, timeoutTemp=-1):
        try:
            t = self.timeout
            self.timeout = timeoutTemp
            yield self
        finally:
            self.timeout = t

def SendMsg(conn, msg):
    try:
        conn.send(json.dumps(msg, cls=AJSONEncoder).encode('utf-8'))
    except Exception as e:
        print('Exception: ', str(e))
        traceback.print_tb(e.__traceback__)
    return

def ReceiveMsg(conn):
    return json.loads(conn.recv().decode('utf-8'), cls=AJSONDecoder)

def BuildArg(arg, kind):
    annotation = BuildTypeFromAST(arg.annotation, TYPE_NAMESPACE) if arg.annotation else inspect.Parameter.empty
    return inspect.Parameter(name=arg.arg, kind=kind, annotation=annotation)

