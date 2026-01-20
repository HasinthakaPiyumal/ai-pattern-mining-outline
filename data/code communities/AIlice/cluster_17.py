# Cluster 17

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

