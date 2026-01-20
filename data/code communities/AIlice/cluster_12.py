# Cluster 12

@wraps(func)
def wrapper(self, *args, **kwargs):
    if not hasattr(self, 'methodLock'):
        raise AttributeError('Object has no methodLock attribute')
    with self.methodLock:
        if getattr(self, f'may_{action}')():
            try:
                ret = func(self, *args, **kwargs)
                getattr(self, action)()
            except Exception as e:
                raise e
            return ret
        else:
            logger.error(f"Method '{func.__name__}' cannot be called in state '{self.state}'")
            raise AWExceptionNotReadyForOperation(f"Method '{func.__name__}' cannot be called in state '{self.state}'. ")

