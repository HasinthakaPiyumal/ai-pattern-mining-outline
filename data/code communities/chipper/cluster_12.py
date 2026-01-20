# Cluster 12

def load_systemprompt(base_path: str) -> str:
    default_prompt = ''
    env_var_name = 'SYSTEM_PROMPT'
    env_prompt = os.getenv(env_var_name)
    if env_prompt is not None and env_prompt.strip() != '':
        content = env_prompt.strip()
        logger.info(f"Using system prompt from '{env_var_name}' environment variable; content: '{content}'")
        return content
    file = Path(base_path) / '.systemprompt'
    if not file.exists():
        logger.info('No .systemprompt file found. Using default prompt.')
        return default_prompt
    try:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            logger.warning('System prompt file is empty. Using default prompt.')
            return default_prompt
        logger.info(f"Successfully loaded system prompt from {file}; content: '{content}'")
        return content
    except Exception as e:
        logger.error(f'Error reading system prompt file: {e}')
        return default_prompt

