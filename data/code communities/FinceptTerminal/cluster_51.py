# Cluster 51

def download_model(model_name: str, ollama_url: str) -> bool:
    """Download a model in Docker environment."""
    print(f'{Fore.YELLOW}Downloading model {model_name} to the Docker Ollama container...{Style.RESET_ALL}')
    print(f'{Fore.CYAN}This may take some time. Please be patient.{Style.RESET_ALL}')
    try:
        response = requests.post(f'{ollama_url}/api/pull', json={'name': model_name}, timeout=10)
        if response.status_code != 200:
            print(f'{Fore.RED}Failed to initiate model download. Status code: {response.status_code}{Style.RESET_ALL}')
            if response.text:
                print(f'{Fore.RED}Error: {response.text}{Style.RESET_ALL}')
            return False
    except requests.RequestException as e:
        print(f'{Fore.RED}Error initiating download request: {e}{Style.RESET_ALL}')
        return False
    print(f'{Fore.CYAN}Download initiated. Checking periodically for completion...{Style.RESET_ALL}')
    total_wait_time = 0
    max_wait_time = 1800
    check_interval = 10
    while total_wait_time < max_wait_time:
        available_models = get_available_models(ollama_url)
        if model_name in available_models:
            print(f'{Fore.GREEN}Model {model_name} downloaded successfully.{Style.RESET_ALL}')
            return True
        time.sleep(check_interval)
        total_wait_time += check_interval
        if total_wait_time % 60 == 0:
            minutes = total_wait_time // 60
            print(f'{Fore.CYAN}Download in progress... ({minutes} minute{('s' if minutes != 1 else '')} elapsed){Style.RESET_ALL}')
    print(f'{Fore.RED}Timed out waiting for model download to complete after {max_wait_time // 60} minutes.{Style.RESET_ALL}')
    return False

def get_available_models(ollama_url: str) -> list:
    """Get list of available models in Docker environment."""
    try:
        response = requests.get(f'{ollama_url}/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [m['name'] for m in models]
        print(f'{Fore.RED}Failed to get available models from Ollama service. Status code: {response.status_code}{Style.RESET_ALL}')
        return []
    except requests.RequestException as e:
        print(f'{Fore.RED}Error getting available models: {e}{Style.RESET_ALL}')
        return []

