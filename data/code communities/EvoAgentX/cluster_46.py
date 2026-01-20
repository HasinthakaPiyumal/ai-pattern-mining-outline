# Cluster 46

def atomic(lock=None):
    """
    threading safe decorator, it can be used to decorate a function or receive a lock:
    1. directly decorate a function: @atomic 
    2. receive a lock: @atomic(lock=shared_lock)
    """
    lock = lock or threading.Lock()

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return decorator if not callable(lock) else decorator(lock)

def decorator(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)
    return wrapper

