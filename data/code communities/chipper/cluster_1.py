# Cluster 1

def log_info(message):
    print(f'{Colors.GREEN}[INFO]{Colors.NC} {message}')

def log_error(message):
    print(f'{Colors.RED}[ERROR]{Colors.NC} {message}')

def has_ollama_key(env_file):
    try:
        with open(env_file, 'r') as file:
            content = file.read()
        return f'OLLAMA_URL={DEFAULT_INTERNAL_OLLAMA_URL}' in content
    except Exception as e:
        log_error(f'Failed to read {env_file}: {str(e)}')
        return False

def update_env_file(env_file, updates):
    try:
        with open(env_file, 'r') as file:
            content = file.read()
        for key, value in updates.items():
            if f'{key}=' in content:
                if key == 'API_KEY':
                    content = content.replace(f'{key}={EXAMPLE_API_KEY}', f'{key}={value}')
                elif key == 'OLLAMA_URL':
                    lines = content.split('\n')
                    ollama_found = False
                    for i, line in enumerate(lines):
                        if line.startswith('OLLAMA_URL='):
                            lines[i] = f'OLLAMA_URL={value}'
                            ollama_found = True
                            break
                    if not ollama_found:
                        lines.append(f'OLLAMA_URL={value}')
                    content = '\n'.join(lines)
                else:
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if line.startswith(f'{key}='):
                            lines[i] = f'{key}={value}'
                    content = '\n'.join(lines)
            else:
                content += f'\n{key}={value}'
        with open(env_file, 'w') as file:
            file.write(content)
        log_info(f'Updated {env_file}')
    except Exception as e:
        log_error(f'Failed to update {env_file}: {str(e)}')

def create_docker_env(profile):
    docker_dir = Path('docker')
    if not docker_dir.exists():
        docker_dir.mkdir(exist_ok=True)
        log_info('Created docker directory')
    docker_env = docker_dir / '.env'
    env_content = f'# Automatically generated Docker environment file\nCOMPOSE_PROFILES={profile}'
    try:
        with open(docker_env, 'w') as f:
            f.write(env_content)
        log_info(f'Created {docker_env} with profile configuration')
    except Exception as e:
        log_error(f'Failed to create {docker_env}: {str(e)}')

def clean_env_files():
    global SHARED_API_KEY, EXTERNAL_OLLAMA_URL
    SHARED_API_KEY = None
    EXTERNAL_OLLAMA_URL = None
    files_to_remove = ['.env', '.ragignore', '.systemprompt']
    docker_env = Path('docker/.env')
    if docker_env.exists():
        try:
            docker_env.unlink()
            log_info(f'Removed {docker_env}')
        except Exception as e:
            log_error(f'Failed to remove {docker_env}: {str(e)}')
    count = 0
    for pattern in files_to_remove:
        for file in Path('.').rglob(pattern):
            try:
                file.unlink()
                count += 1
                log_info(f'Removed {file}')
            except Exception as e:
                log_error(f'Failed to remove {file}: {str(e)}')
    if count > 0:
        log_info(f'Removed {count} file{('s' if count > 1 else '')}')
    else:
        log_info('No files found to remove')

