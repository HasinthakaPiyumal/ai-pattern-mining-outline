# Cluster 2

def check_external_ollama_requirement(gpu_profile: str) -> bool:
    system = platform.system()
    release = platform.release()
    log_info(f'Platform: {system}/{release}')
    log_info(f'GPU Profile: {gpu_profile}')
    is_darwin = system in ['Darwin']
    is_cpu_profile = gpu_profile == 'cpu'
    is_amd_linux = gpu_profile == 'amd-linux'
    requires_external = is_darwin or is_cpu_profile or is_amd_linux
    if requires_external:
        log_info(f'Using external Ollama server at {DEFAULT_EXTERNAL_OLLAMA_URL}')
        if not check_ollama_availability(DEFAULT_EXTERNAL_LOCAL_OLLAMA_URL):
            message = "Cannot connect to local Ollama server\n\n--------------------------------------------------------------------------------\n\nThe internal Ollama container is not supported on your platform.\nYou must install and run Ollama manually before using Chipper.\n\n1. Download and install Ollama from: https://ollama.com\n2. Start the Ollama service\n3. Ensure it's running at: " + DEFAULT_EXTERNAL_LOCAL_OLLAMA_URL + '\n\nNote: GPU support in Docker Desktop is currently only available\non Windows with the WSL2 backend\nor via the Linux NVIDIA Container Toolkit.\n\nYou can ignore this message if you are using an external\nOllama endpoint or HuggingFace inference service.\n\n--------------------------------------------------------------------------------\n\n'
            log_warning(message)
            return True
        return True
    is_wsl = 'microsoft' in release.lower()
    if is_wsl:
        log_info('WSL Linux detected')
    return False

def check_ollama_availability(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or 11434
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except (socket.error, ValueError):
        return False

def log_warning(message):
    print(f'{Colors.YELLOW}[WARN]{Colors.NC} {message}')

def detect_gpu_profile():
    system = platform.system()
    if system == 'Darwin':
        log_info('Detected macOS system')
        return 'metal'
    try:
        subprocess.run(['nvidia-smi'], capture_output=True, check=True)
        log_info('Detected NVIDIA GPU')
        return 'nvidia'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    if system == 'Linux':
        if Path('/dev/dri').exists() and Path('/dev/kfd').exists():
            log_info('Detected AMD GPU with ROCm support')
            return 'amd-linux'
    elif system == 'Windows':
        try:
            wmic_output = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], capture_output=True, text=True).stdout
            if 'AMD' in wmic_output or 'Radeon' in wmic_output:
                log_info('Detected AMD GPU')
                return 'amd'
        except Exception:
            pass
    log_warning('No GPU detected or unsupported GPU configuration')
    return 'cpu'

