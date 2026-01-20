# Cluster 0

@stub.function(gpu=GPU_CONFIG, allow_concurrent_inputs=int(PARALLELISM), container_idle_timeout=120, timeout=360)
@asgi_app()
def app():
    import os
    import socket
    import subprocess
    import time
    from asgi_proxy import asgi_proxy
    model_id = os.environ.get('MODEL_ID')
    parallelism = os.environ.get('PARALLELISM')
    env = os.environ.copy()
    env['TABBY_DISABLE_USAGE_COLLECTION'] = '1'
    launcher = subprocess.Popen(['/opt/tabby/bin/tabby', 'serve', '--model', model_id, '--port', '8000', '--device', 'cuda', '--parallelism', parallelism], env=env)

    def tabby_ready():
        try:
            socket.create_connection(('127.0.0.1', 8000), timeout=1).close()
            return True
        except (socket.timeout, ConnectionRefusedError):
            retcode = launcher.poll()
            if retcode is not None:
                raise RuntimeError(f'launcher exited unexpectedly with code {retcode}')
            return False
    while not tabby_ready():
        time.sleep(1.0)
    print('Tabby server ready!')
    return asgi_proxy('http://localhost:8000')

def tabby_ready():
    try:
        socket.create_connection(('127.0.0.1', 8000), timeout=1).close()
        return True
    except (socket.timeout, ConnectionRefusedError):
        retcode = launcher.poll()
        if retcode is not None:
            raise RuntimeError(f'launcher exited unexpectedly with code {retcode}')
        return False

@app.function(gpu=GPU_CONFIG, allow_concurrent_inputs=10, container_idle_timeout=120, timeout=360)
@asgi_app()
def app_serve():
    import socket
    import subprocess
    import time
    from asgi_proxy import asgi_proxy
    launcher = subprocess.Popen([TABBY_BIN, 'serve', '--model', MODEL_ID, '--chat-model', CHAT_MODEL_ID, '--port', '8000', '--device', 'cuda', '--parallelism', '1'])

    def tabby_ready():
        try:
            socket.create_connection(('127.0.0.1', 8000), timeout=1).close()
            return True
        except (socket.timeout, ConnectionRefusedError):
            retcode = launcher.poll()
            if retcode is not None:
                raise RuntimeError(f'launcher exited unexpectedly with code {retcode}')
            return False
    while not tabby_ready():
        time.sleep(1.0)
    print('Tabby server ready!')
    return asgi_proxy('http://localhost:8000')

