# Cluster 40

@wraps(func)
def wrapper(*args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        func_module = module or func.__module__.split('.')[-1]
        title = title_template.format(func_name=func.__name__)
        message = message_template.format(error=str(e))
        notify_error(title, message, func_module)
        raise

def notify_success(title: str, message: str, module: Optional[str]=None, **kwargs) -> bool:
    """Send success notification"""
    return notifier.success(title, message, module, **kwargs)

def notify_error(title: str, message: str, module: Optional[str]=None, **kwargs) -> bool:
    """Send error notification"""
    return notifier.error(title, message, module, **kwargs)

