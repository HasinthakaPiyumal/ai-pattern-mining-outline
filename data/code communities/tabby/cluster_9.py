# Cluster 9

def supervisor(serve_args):
    launcher = TabbyLauncher(serve_args)
    proxy = asgi_proxy('http://localhost:8081')
    timer = None

    async def callback(scope, receive, send):
        nonlocal timer
        if not launcher.is_running:
            launcher.start()
        elif timer is not None:
            timer = timer.cancel()
        timer = Timer(600, launcher.stop)
        return await proxy(scope, receive, send)
    return callback

