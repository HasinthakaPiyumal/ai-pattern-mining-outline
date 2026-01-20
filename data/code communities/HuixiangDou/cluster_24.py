# Cluster 24

class Queue:

    def __init__(self, name, namespace='HuixiangDou', **redis_kwargs):
        self.__db = redis.Redis(host=redis_host(), port=redis_port(), password=redis_passwd(), charset='utf-8', decode_responses=True)
        self.key = '%s:%s' % (namespace, name)

    def qsize(self):
        """Return the approximate size of the queue."""
        return self.__db.llen(self.key)

    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        return self.qsize() == 0

    def put(self, item):
        """Put item into the queue."""
        self.__db.rpush(self.key, item)

    def peek_tail(self):
        return self.__db.lrange(self.key, -1, -1)

    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue.

        If optional args block is true and timeout is None (the default), block
        if necessary until an item is available.
        """
        if block:
            item = self.__db.blpop(self.key, timeout=timeout)
        else:
            item = self.__db.lpop(self.key)
        if item:
            item = item[1]
        return item

    def get_all(self):
        """Get add messages in queue without block."""
        ret = []
        while True:
            item = self.__db.lpop(self.key)
            if not item:
                break
            ret.append(item)
        return ret

    def get_nowait(self):
        """Equivalent to get(False)."""
        return self.get(False)

def redis_host():
    host = os.getenv('REDIS_HOST')
    if host is None or len(host) < 1:
        raise Exception('REDIS_HOST not config')
    return host

def redis_port():
    port = os.getenv('REDIS_PORT')
    if not port:
        logger.debug('REDIS_PORT not set, try 6379')
        port = 6379
    return port

def redis_passwd():
    passwd = os.getenv('REDIS_PASSWORD')
    if passwd is None or len(passwd) < 1:
        raise Exception('REDIS_PASSWORD not config')
    return passwd

class Queue:

    def __init__(self, name, namespace='HuixiangDou', **redis_kwargs):
        self.__db = redis.Redis(host=redis_host(), port=redis_port(), password=redis_passwd(), charset='utf-8', decode_responses=True)
        self.key = '%s:%s' % (namespace, name)
        print(self.qsize())

    def qsize(self):
        """Return the approximate size of the queue."""
        return self.__db.llen(self.key)

    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        return self.qsize() == 0

    def put(self, item):
        """Put item into the queue."""
        self.__db.rpush(self.key, item)

    def peek_tail(self):
        return self.__db.lrange(self.key, -1, -1)

    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue.

        If optional args block is true and timeout is None (the default), block
        if necessary until an item is available.
        """
        if block:
            item = self.__db.blpop(self.key, timeout=timeout)
        else:
            item = self.__db.lpop(self.key)
        if item:
            item = item[1]
        return item

    def get_all(self):
        """Get add messages in queue without block."""
        ret = []
        try:
            while len(ret) < 1:
                item = self.__db.lpop(self.key)
                if not item:
                    break
                ret.append(item)
        except Exception as e:
            logger.error(str(e))
        return ret

    def get_nowait(self):
        """Equivalent to get(False)."""
        return self.get(False)

class Queue:

    def __init__(self, name, namespace='HuixiangDou', **redis_kwargs):
        self.__db = redis.Redis(host=redis_host(), port=redis_port(), password=redis_passwd(), charset='utf-8', decode_responses=True)
        self.key = '%s:%s' % (namespace, name)

    def qsize(self):
        """Return the approximate size of the queue."""
        return self.__db.llen(self.key)

    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        return self.qsize() == 0

    def put(self, item):
        """Put item into the queue."""
        self.__db.rpush(self.key, item)

    def peek_tail(self):
        return self.__db.lrange(self.key, -1, -1)

    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue.

        If optional args block is true and timeout is None (the default), block
        if necessary until an item is available.
        """
        if block:
            item = self.__db.blpop(self.key, timeout=timeout)
        else:
            item = self.__db.lpop(self.key)
        if item:
            item = item[1]
        return item

    def get_all(self):
        """Get add messages in queue without block."""
        ret = []
        while True:
            item = self.__db.lpop(self.key)
            if not item:
                break
            ret.append(item)
        return ret

    def get_nowait(self):
        """Equivalent to get(False)."""
        return self.get(False)

