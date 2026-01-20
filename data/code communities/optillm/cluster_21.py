# Cluster 21

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLM inference with various approaches.')
    try:
        from optillm import __version__ as package_version
    except ImportError:
        package_version = 'unknown'
    parser.add_argument('--version', action='version', version=f'%(prog)s {package_version}', help="Show program's version number and exit")
    args_env = [('--optillm-api-key', 'OPTILLM_API_KEY', str, '', 'Optional API key for client authentication to optillm'), ('--approach', 'OPTILLM_APPROACH', str, 'auto', 'Inference approach to use', known_approaches + list(plugin_approaches.keys())), ('--mcts-simulations', 'OPTILLM_SIMULATIONS', int, 2, 'Number of MCTS simulations'), ('--mcts-exploration', 'OPTILLM_EXPLORATION', float, 0.2, 'Exploration weight for MCTS'), ('--mcts-depth', 'OPTILLM_DEPTH', int, 1, 'Simulation depth for MCTS'), ('--model', 'OPTILLM_MODEL', str, 'gpt-4o-mini', 'OpenAI model to use'), ('--rstar-max-depth', 'OPTILLM_RSTAR_MAX_DEPTH', int, 3, 'Maximum depth for rStar algorithm'), ('--rstar-num-rollouts', 'OPTILLM_RSTAR_NUM_ROLLOUTS', int, 5, 'Number of rollouts for rStar algorithm'), ('--rstar-c', 'OPTILLM_RSTAR_C', float, 1.4, 'Exploration constant for rStar algorithm'), ('--n', 'OPTILLM_N', int, 1, 'Number of final responses to be returned'), ('--return-full-response', 'OPTILLM_RETURN_FULL_RESPONSE', bool, False, 'Return the full response including the CoT with <thinking> tags'), ('--port', 'OPTILLM_PORT', int, 8000, 'Specify the port to run the proxy'), ('--log', 'OPTILLM_LOG', str, 'info', 'Specify the logging level', list(logging_levels.keys())), ('--launch-gui', 'OPTILLM_LAUNCH_GUI', bool, False, 'Launch a Gradio chat interface'), ('--plugins-dir', 'OPTILLM_PLUGINS_DIR', str, '', 'Path to the plugins directory'), ('--log-conversations', 'OPTILLM_LOG_CONVERSATIONS', bool, False, 'Enable conversation logging with full metadata'), ('--conversation-log-dir', 'OPTILLM_CONVERSATION_LOG_DIR', str, str(Path.home() / '.optillm' / 'conversations'), 'Directory to save conversation logs')]
    for arg, env, type_, default, help_text, *extra in args_env:
        env_value = os.environ.get(env)
        if env_value is not None:
            if type_ == bool:
                default = env_value.lower() in ('true', '1', 'yes')
            else:
                default = type_(env_value)
        if extra and extra[0]:
            parser.add_argument(arg, type=type_, default=default, help=help_text, choices=extra[0])
        elif type_ == bool:
            parser.add_argument(arg, action='store_true', default=default, help=help_text)
        else:
            parser.add_argument(arg, type=type_, default=default, help=help_text)
    best_of_n_default = int(os.environ.get('OPTILLM_BEST_OF_N', 3))
    parser.add_argument('--best-of-n', '--best_of_n', dest='best_of_n', type=int, default=best_of_n_default, help='Number of samples for best_of_n approach')
    base_url_default = os.environ.get('OPTILLM_BASE_URL', '')
    parser.add_argument('--base-url', '--base_url', dest='base_url', type=str, default=base_url_default, help='Base url for OpenAI compatible endpoint')
    ssl_verify_default = os.environ.get('OPTILLM_SSL_VERIFY', 'true').lower() in ('true', '1', 'yes')
    parser.add_argument('--ssl-verify', dest='ssl_verify', action='store_true' if ssl_verify_default else 'store_false', default=ssl_verify_default, help='Enable SSL certificate verification (default: True)')
    parser.add_argument('--no-ssl-verify', dest='ssl_verify', action='store_false', help='Disable SSL certificate verification')
    ssl_cert_path_default = os.environ.get('OPTILLM_SSL_CERT_PATH', '')
    parser.add_argument('--ssl-cert-path', dest='ssl_cert_path', type=str, default=ssl_cert_path_default, help='Path to custom CA certificate bundle for SSL verification')
    default_config_path = get_config_path()
    batch_mode_default = os.environ.get('OPTILLM_BATCH_MODE', 'false').lower() == 'true'
    batch_size_default = int(os.environ.get('OPTILLM_BATCH_SIZE', 4))
    batch_wait_ms_default = int(os.environ.get('OPTILLM_BATCH_WAIT_MS', 50))
    parser.add_argument('--batch-mode', action='store_true', default=batch_mode_default, help='Enable automatic request batching (fail-fast, no fallback)')
    parser.add_argument('--batch-size', type=int, default=batch_size_default, help='Maximum batch size for request batching')
    parser.add_argument('--batch-wait-ms', dest='batch_wait_ms', type=int, default=batch_wait_ms_default, help='Maximum wait time in milliseconds for batch formation')
    for field in fields(CepoConfig):
        parser.add_argument(f'--cepo_{field.name}', dest=f'cepo_{field.name}', type=field.type, default=None, help=f'CePO configuration for {field.name}')
    parser.add_argument('--cepo_config_file', dest='cepo_config_file', type=str, default=default_config_path, help='Path to CePO configuration file')
    args = parser.parse_args()
    args_dict = vars(args)
    for key in list(args_dict.keys()):
        new_key = key.replace('-', '_')
        if new_key != key:
            args_dict[new_key] = args_dict.pop(key)
    return args

def get_config_path():
    import optillm
    package_config_dir = os.path.join(os.path.dirname(optillm.__file__), 'cepo', 'configs')
    package_config_path = os.path.join(package_config_dir, 'cepo_config.yaml')
    current_dir = os.getcwd() if server_config.get('config_dir', '') == '' else server_config['config_dir']
    local_config_dir = os.path.join(current_dir, 'optillm', 'cepo', 'configs')
    local_config_path = os.path.join(local_config_dir, 'cepo_config.yaml')
    if os.path.exists(local_config_path) and local_config_path != package_config_path:
        logger.debug(f'Using local config from: {local_config_path}')
        return local_config_path
    logger.debug(f'Using package config from: {package_config_path}')
    return package_config_path

