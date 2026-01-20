# Cluster 5

def has_example_api_key_set(env_file):
    try:
        with open(env_file, 'r') as file:
            content = file.read()
        return f'API_KEY={EXAMPLE_API_KEY}' in content
    except Exception as e:
        log_error(f'Failed to read {env_file}: {str(e)}')
        return False

def copy_example_files():
    example_mappings = {'.env.example': '.env', '.ragignore.example': '.ragignore', '.systemprompt.example': '.systemprompt'}
    found_files = []
    files_needing_update = []
    for example_pattern in example_mappings.keys():
        for example_file in Path('.').rglob(example_pattern):
            actual_file = example_file.with_name(example_mappings[example_pattern])
            found_files.append(actual_file)
            if not actual_file.exists():
                shutil.copy(example_file, actual_file)
                log_info(f'Created {actual_file} from {example_file}')
            if example_pattern == '.env.example' or has_example_api_key_set(str(actual_file)):
                files_needing_update.append(actual_file)
    return (found_files, files_needing_update)

def main():
    parser = argparse.ArgumentParser(description='Environment file setup utility')
    parser.add_argument('--clean', action='store_true', help='Remove all generated files')
    parser.add_argument('--docker-only', action='store_true', help='Only create Docker environment file')
    parser.add_argument('--ollama-url', help=f'URL for external Ollama server (default: {DEFAULT_INTERNAL_OLLAMA_URL})')
    args = parser.parse_args()
    if args.clean:
        clean_env_files()
        return 0
    global EXTERNAL_OLLAMA_URL
    EXTERNAL_OLLAMA_URL = args.ollama_url or os.environ.get('OLLAMA_URL') or DEFAULT_EXTERNAL_OLLAMA_URL
    generate_api_key()
    gpu_profile = detect_gpu_profile()
    create_docker_env(gpu_profile)
    use_external_ollama = check_external_ollama_requirement(gpu_profile)
    if args.docker_only:
        return 0
    log_info('Starting to search for example files...')
    found_env_files, files_needing_update = copy_example_files()
    if not found_env_files:
        log_error('No example .env files found!')
        return 1
    for env_file in files_needing_update:
        if str(env_file).endswith('.env'):
            updates = {}
            if has_example_api_key_set(str(env_file)):
                updates['API_KEY'] = SHARED_API_KEY
            if has_ollama_key(str(env_file)) and use_external_ollama:
                updates['OLLAMA_URL'] = EXTERNAL_OLLAMA_URL
            if updates:
                update_env_file(str(env_file), updates)
    log_info('Setup completed successfully!')
    return 0

def generate_api_key():
    global SHARED_API_KEY
    if SHARED_API_KEY is None:
        SHARED_API_KEY = secrets.token_hex(32)
    return SHARED_API_KEY

