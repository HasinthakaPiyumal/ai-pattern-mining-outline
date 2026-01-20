# Cluster 15

def throttle_async_func_call(max_concurrent: int=2048, stagger_time: Optional[float]=None, waiting_time: float=0.001):
    _wrappedFn = TypeVar('_wrappedFn', bound=Callable[..., Any])

    def decorator(func: _wrappedFn) -> _wrappedFn:
        semaphore = asyncio.Semaphore(max_concurrent)

        @wraps(func)
        async def wait_func(*args: Any, **kwargs: Any) -> Any:
            async with semaphore:
                try:
                    if stagger_time:
                        await asyncio.sleep(stagger_time)
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f'Error in throttled function {func.__name__}: {e}')
                    raise e
        return wait_func
    return decorator

