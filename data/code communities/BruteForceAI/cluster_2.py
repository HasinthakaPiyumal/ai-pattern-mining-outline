# Cluster 2

def _validate_llm_setup(llm_provider, llm_model, llm_api_key=None, ollama_url=None):
    """
    Validate LLM setup and provide helpful error messages
    
    Args:
        llm_provider: LLM provider ('ollama' or 'groq')
        llm_model: Model name
        llm_api_key: API key for Groq (optional)
        ollama_url: Ollama server URL (optional)
        
    Returns:
        bool: True if setup is valid, False otherwise (exits script)
    """
    if not llm_provider or not llm_model:
        return True
    if llm_provider.lower() == 'ollama':
        ollama_url = ollama_url or 'http://localhost:11434'
        print(f'🔍 Checking Ollama setup at {ollama_url}...')
        if not _check_ollama_availability(ollama_url):
            print(f'❌ Ollama Error: Ollama is not running or not accessible at {ollama_url}')
            print('')
            print('🔧 To fix this:')
            print('1. Install Ollama: https://ollama.ai/download')
            print('2. Start Ollama service')
            print('3. Check if the URL is correct')
            print('4. Or use Groq instead: --llm-provider groq --llm-api-key YOUR_KEY')
            print('')
            print("💡 Quick test: Try running 'ollama --version' in your terminal")
            print('')
            exit(1)
        if not _check_ollama_model(llm_model, ollama_url):
            print(f"❌ Model Error: Model '{llm_model}' is not installed in Ollama at {ollama_url}")
            print('')
            print('🔧 To fix this:')
            print(f'1. Install the model: ollama pull {llm_model}')
            print('2. Or use a different model with: --llm-model MODEL_NAME')
            print('3. Or use Groq instead: --llm-provider groq --llm-api-key YOUR_KEY')
            print('')
            print('📋 Popular models you can install:')
            print('   ollama pull llama3.2:3b     # Fast, good for most tasks')
            print('   ollama pull llama3.2:1b     # Very fast, smaller model')
            print('   ollama pull qwen2.5:3b      # Alternative option')
            print('')
            exit(1)
        print(f"✅ Ollama setup verified - model '{llm_model}' is ready at {ollama_url}")
    elif llm_provider.lower() == 'groq':
        print('🔍 Checking Groq setup...')
        if not llm_api_key:
            print('❌ Groq Error: API key is required for Groq')
            print('')
            print('🔧 To fix this:')
            print('1. Get API key from: https://console.groq.com/')
            print('2. Use: --llm-api-key YOUR_GROQ_API_KEY')
            print('3. Or use Ollama instead: --llm-provider ollama')
            print('')
            exit(1)
        if not llm_api_key.startswith('gsk_'):
            print("⚠️  Warning: Groq API keys usually start with 'gsk_'")
            print("   Make sure you're using the correct API key format")
            print('')
        print(f"✅ Groq setup configured - model '{llm_model}' will be validated on first use")
        if llm_model == 'llama-3.1-8b-instant':
            print(f'💡 Tip: For better analysis quality, try: --llm-model llama-3.3-70b-versatile')
        elif llm_model not in ['llama-3.3-70b-versatile', 'llama3-70b-8192', 'gemma2-9b-it']:
            print(f'💡 Recommended models: llama-3.3-70b-versatile (best), llama3-70b-8192 (fast), gemma2-9b-it (lightweight)')
    return True

def _check_ollama_availability(ollama_url='http://localhost:11434'):
    """
    Check if Ollama is installed and running
    
    Args:
        ollama_url: Ollama server URL (default: http://localhost:11434)
        
    Returns:
        bool: True if Ollama is available, False otherwise
    """
    try:
        response = requests.get(f'{ollama_url}/api/tags', timeout=3)
        return response.status_code == 200
    except:
        return False

def _check_ollama_model(model_name, ollama_url='http://localhost:11434'):
    """
    Check if a specific model is installed in Ollama
    
    Args:
        model_name: Name of the model to check
        ollama_url: Ollama server URL (default: http://localhost:11434)
        
    Returns:
        bool: True if model is available, False otherwise
    """
    try:
        response = requests.get(f'{ollama_url}/api/tags', timeout=3)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            installed_models = [model.get('name', '').split(':')[0] for model in models]
            model_base = model_name.split(':')[0]
            return model_name in [m.get('name', '') for m in models] or model_base in installed_models
        return False
    except:
        return False

