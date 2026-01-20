# Cluster 2

def _init_pyglet_device():
    available_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    if available_devices is not None and len(available_devices) > 0:
        os.environ['PYGLET_HEADLESS_DEVICE'] = available_devices.split(',')[0] if len(available_devices) > 1 else available_devices

