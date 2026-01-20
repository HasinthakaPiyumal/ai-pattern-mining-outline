# Cluster 3

def setup_logging(debug: bool=False, log_file: Optional[str]=None, turn: Optional[int]=None):
    """
    Configure MassGen logging system using loguru.

    Args:
        debug: Enable debug mode with verbose logging
        log_file: Optional path to log file for persistent logging
        turn: Optional turn number for multi-turn conversations
    """
    global _DEBUG_MODE, _CONSOLE_HANDLER_ID, _CONSOLE_SUPPRESSED
    _DEBUG_MODE = debug
    _CONSOLE_SUPPRESSED = False
    logger.remove()
    if debug:

        def custom_format(record):
            name = record['extra'].get('name', '')
            if 'orchestrator' in name:
                name_color = 'magenta'
            elif 'backend' in name:
                name_color = 'yellow'
            elif 'agent' in name:
                name_color = 'cyan'
            elif 'coordination' in name:
                name_color = 'red'
            else:
                name_color = 'white'
            formatted_name = name if name else '{name}'
            return f'<green>{{time:HH:mm:ss.SSS}}</green> | <level>{{level: <8}}</level> | <{name_color}>{formatted_name}</{name_color}>:<{name_color}>{{function}}</{name_color}>:<{name_color}>{{line}}</{name_color}> - {{message}}\n{{exception}}'
        _CONSOLE_HANDLER_ID = logger.add(sys.stderr, format=custom_format, level='DEBUG', colorize=True, backtrace=True, diagnose=True)
        if not log_file:
            log_session_dir = get_log_session_dir(turn=turn)
            log_file = log_session_dir / 'massgen_debug.log'
        logger.add(str(log_file), format=custom_format, level='DEBUG', rotation='100 MB', retention='1 week', compression='zip', backtrace=True, diagnose=True, enqueue=True, colorize=False)
        logger.info('Debug logging enabled - logging to console and file: {}', log_file)
    else:
        console_format = '<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>'
        _CONSOLE_HANDLER_ID = logger.add(sys.stderr, format=console_format, level='WARNING', colorize=True)
        if not log_file:
            log_session_dir = get_log_session_dir(turn=turn)
            log_file = log_session_dir / 'massgen.log'
        logger.add(str(log_file), format=console_format, level='INFO', rotation='10 MB', retention='3 days', compression='zip', enqueue=True, colorize=False)
        logger.info('Logging enabled - logging INFO+ to file: {}', log_file)

def load_env_file():
    """Load environment variables from .env files.

    Search order (later files override earlier ones):
    1. MassGen package .env (development fallback)
    2. User home ~/.massgen/.env (global user config)
    3. Current directory .env (project-specific, highest priority)
    """
    load_dotenv(Path(__file__).parent / '.env')
    load_dotenv(Path.home() / '.massgen' / '.env')
    load_dotenv()

def load_previous_turns(session_info: Dict[str, Any], session_storage: str) -> List[Dict[str, Any]]:
    """
    Load previous turns from session storage.

    Returns:
        List of previous turn metadata dicts
    """
    session_id = session_info.get('session_id')
    if not session_id:
        return []
    session_dir = Path(session_storage) / session_id
    if not session_dir.exists():
        return []
    previous_turns = []
    turn_num = 1
    while True:
        turn_dir = session_dir / f'turn_{turn_num}'
        if not turn_dir.exists():
            break
        metadata_file = turn_dir / 'metadata.json'
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
            workspace_path = (turn_dir / 'workspace').resolve()
            previous_turns.append({'turn': turn_num, 'path': str(workspace_path), 'task': metadata.get('task', ''), 'winning_agent': metadata.get('winning_agent', '')})
        turn_num += 1
    return previous_turns

def print_example_config(name: str):
    """Print an example config to stdout.

    Args:
        name: Name of the example (can include or exclude @examples/ prefix)
    """
    try:
        if name.startswith('@examples/'):
            name = name[10:]
        resolved = resolve_config_path(f'@examples/{name}')
        if resolved:
            with open(resolved, 'r') as f:
                print(f.read())
        else:
            print(f"Error: Could not find example '{name}'", file=sys.stderr)
            print('Use --list-examples to see available configs', file=sys.stderr)
            sys.exit(1)
    except ConfigurationError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'Error printing example config: {e}', file=sys.stderr)
        sys.exit(1)

def resolve_config_path(config_arg: Optional[str]) -> Optional[Path]:
    """Resolve config file with flexible syntax.

    Priority order:

    **If --config flag provided (highest priority):**
    1. @examples/NAME → Package examples (search configs directory)
    2. Absolute/relative paths (exact path as specified)
    3. Named configs in ~/.config/massgen/agents/

    **If NO --config flag (auto-discovery):**
    1. .massgen/config.yaml (project-level config in current directory)
    2. ~/.config/massgen/config.yaml (global default config)
    3. None → trigger config builder

    Args:
        config_arg: Config argument from --config flag (can be @examples/NAME, path, or None)

    Returns:
        Path to config file, or None if config builder should run

    Raises:
        ConfigurationError: If config file not found
    """
    if not config_arg:
        project_config = Path.cwd() / '.massgen' / 'config.yaml'
        if project_config.exists():
            return project_config
        global_config = Path.home() / '.config/massgen/config.yaml'
        if global_config.exists():
            return global_config
        return None
    if config_arg.startswith('@examples/'):
        name = config_arg[10:]
        try:
            from importlib.resources import files
            configs_root = files('massgen') / 'configs'
            for config_file in configs_root.rglob('*.yaml'):
                if name in config_file.name or name in str(config_file):
                    return Path(str(config_file))
            raise ConfigurationError(f"Config '{config_arg}' not found in package.\nUse --list-examples to see available configs.")
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f'Error loading package config: {e}')
    path = Path(config_arg).expanduser()
    if path.exists():
        return path
    user_agents_dir = Path.home() / '.config/massgen/agents'
    user_config = user_agents_dir / f'{config_arg}.yaml'
    if user_config.exists():
        return user_config
    if not config_arg.endswith(('.yaml', '.yml')):
        user_config_with_ext = user_agents_dir / f'{config_arg}.yaml'
        if user_config_with_ext.exists():
            return user_config_with_ext
    raise ConfigurationError(f'Configuration file not found: {config_arg}\nSearched in:\n  - Current directory: {Path.cwd() / config_arg}\n  - User configs: {user_agents_dir / config_arg}.yaml\nUse --list-examples to see available package configs.')

def prompt_for_context_paths(original_config: Dict[str, Any], orchestrator_cfg: Dict[str, Any]) -> bool:
    """Prompt user to add context paths in interactive mode.

    Returns True if config was modified, False otherwise.
    """
    agent_entries = [original_config['agent']] if 'agent' in original_config else original_config.get('agents', [])
    has_filesystem = any(('cwd' in agent.get('backend', {}) for agent in agent_entries))
    if not has_filesystem:
        return False
    existing_paths = orchestrator_cfg.get('context_paths', [])
    cwd = Path.cwd()
    from rich.console import Console as RichConsole
    from rich.panel import Panel as RichPanel
    rich_console = RichConsole()
    context_content = []
    if existing_paths:
        for path_config in existing_paths:
            path = path_config.get('path') if isinstance(path_config, dict) else path_config
            permission = path_config.get('permission', 'read') if isinstance(path_config, dict) else 'read'
            context_content.append(f'  [green]✓[/green] {path} [dim]({permission})[/dim]')
    else:
        context_content.append('  [yellow]No context paths configured[/yellow]')
    context_panel = RichPanel('\n'.join(context_content), title='[bold bright_cyan]📂 Context Paths[/bold bright_cyan]', border_style='cyan', padding=(0, 2), width=80)
    rich_console.print(context_panel)
    print()
    cwd_str = str(cwd)
    cwd_already_added = any(((path_config.get('path') if isinstance(path_config, dict) else path_config) == cwd_str for path_config in existing_paths))
    if not cwd_already_added:
        prompt_content = ['[bold cyan]Add current directory as context path?[/bold cyan]', f'  [yellow]{cwd}[/yellow]', '', '  [dim]Context paths give agents access to your project files.[/dim]', '  [dim]• Read-only during coordination (prevents conflicts)[/dim]', '  [dim]• Write permission for final agent to save results[/dim]', '', '  [dim]Options:[/dim]', '  [green]Y[/green] → Add with write permission (default)', '  [cyan]P[/cyan] → Add with protected paths (e.g., .env, secrets)', '  [yellow]N[/yellow] → Skip', '  [blue]C[/blue] → Add custom path']
        prompt_panel = RichPanel('\n'.join(prompt_content), border_style='cyan', padding=(1, 2), width=80)
        rich_console.print(prompt_panel)
        print()
        try:
            response = input(f'   {BRIGHT_CYAN}Your choice [Y/P/N/C]:{RESET} ').strip().lower()
            if response in ['y', 'yes', '']:
                if 'context_paths' not in orchestrator_cfg:
                    orchestrator_cfg['context_paths'] = []
                orchestrator_cfg['context_paths'].append({'path': cwd_str, 'permission': 'write'})
                print(f'   {BRIGHT_GREEN}✅ Added: {cwd} (write){RESET}', flush=True)
                return True
            elif response in ['p', 'protected']:
                protected_paths = []
                print(f'\n   {BRIGHT_CYAN}Enter protected paths (one per line, empty to finish):{RESET}', flush=True)
                print(f'   {BRIGHT_YELLOW}Tip: Protected paths are relative to {cwd}{RESET}', flush=True)
                while True:
                    protected_input = input(f'   {BRIGHT_CYAN}→{RESET} ').strip()
                    if not protected_input:
                        break
                    protected_paths.append(protected_input)
                    print(f'     {BRIGHT_GREEN}✓ Added: {protected_input}{RESET}', flush=True)
                if 'context_paths' not in orchestrator_cfg:
                    orchestrator_cfg['context_paths'] = []
                context_config = {'path': cwd_str, 'permission': 'write'}
                if protected_paths:
                    context_config['protected_paths'] = protected_paths
                orchestrator_cfg['context_paths'].append(context_config)
                print(f'\n   {BRIGHT_GREEN}✅ Added: {cwd} (write) with {len(protected_paths)} protected path(s){RESET}', flush=True)
                return True
            elif response in ['n', 'no']:
                return False
            elif response in ['c', 'custom']:
                print()
                while True:
                    custom_path = input(f'   {BRIGHT_CYAN}Enter path (absolute or relative):{RESET} ').strip()
                    if not custom_path:
                        print(f'   {BRIGHT_YELLOW}⚠️  Cancelled{RESET}', flush=True)
                        return False
                    abs_path = str(Path(custom_path).resolve())
                    if not Path(abs_path).exists():
                        print(f'   {BRIGHT_RED}✗ Path does not exist: {abs_path}{RESET}', flush=True)
                        retry = input(f'   {BRIGHT_CYAN}Try again? [Y/n]:{RESET} ').strip().lower()
                        if retry in ['n', 'no']:
                            return False
                        continue
                    permission = input(f'   {BRIGHT_CYAN}Permission [read/write] (default: write):{RESET} ').strip().lower() or 'write'
                    if permission not in ['read', 'write']:
                        permission = 'write'
                    protected_paths = []
                    if permission == 'write':
                        add_protected = input(f'   {BRIGHT_CYAN}Add protected paths? [y/N]:{RESET} ').strip().lower()
                        if add_protected in ['y', 'yes']:
                            print(f'   {BRIGHT_CYAN}Enter protected paths (one per line, empty to finish):{RESET}', flush=True)
                            while True:
                                protected_input = input(f'   {BRIGHT_CYAN}→{RESET} ').strip()
                                if not protected_input:
                                    break
                                protected_paths.append(protected_input)
                                print(f'     {BRIGHT_GREEN}✓ Added: {protected_input}{RESET}', flush=True)
                    if 'context_paths' not in orchestrator_cfg:
                        orchestrator_cfg['context_paths'] = []
                    context_config = {'path': abs_path, 'permission': permission}
                    if protected_paths:
                        context_config['protected_paths'] = protected_paths
                    orchestrator_cfg['context_paths'].append(context_config)
                    if protected_paths:
                        print(f'   {BRIGHT_GREEN}✅ Added: {abs_path} ({permission}) with {len(protected_paths)} protected path(s){RESET}', flush=True)
                    else:
                        print(f'   {BRIGHT_GREEN}✅ Added: {abs_path} ({permission}){RESET}', flush=True)
                    return True
            else:
                print(f"\n   {BRIGHT_RED}✗ Invalid option: '{response}'{RESET}", flush=True)
                print(f'   {BRIGHT_YELLOW}Please choose: Y (yes), P (protected), N (no), or C (custom){RESET}', flush=True)
                return False
        except (KeyboardInterrupt, EOFError):
            print()
            return False
    return False

def print_help_messages():
    """Display help messages using Rich for better formatting."""
    rich_console = Console()
    help_content = '[dim]💬  Type your questions below\n💡  Use slash commands: [cyan]/help[/cyan], [cyan]/quit[/cyan], [cyan]/reset[/cyan], [cyan]/status[/cyan], [cyan]/config[/cyan]\n⌨️   Press [cyan]Ctrl+C[/cyan] to exit[/dim]'
    help_panel = Panel(help_content, border_style='dim', padding=(0, 2), width=80)
    rich_console.print(help_panel)

def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.

    Search order:
    1. Exact path as provided (absolute or relative to CWD)
    2. If just a filename, search in package's configs/ directory
    3. If a relative path, also try within package's configs/ directory

    Supports variable substitution: ${cwd} in any string will be replaced with the agent's cwd value.
    """
    path = Path(config_path)
    if path.exists():
        pass
    elif path.is_absolute():
        raise ConfigurationError(f'Configuration file not found: {config_path}')
    else:
        package_configs_dir = Path(__file__).parent / 'configs'
        candidate1 = package_configs_dir / path.name
        candidate2 = package_configs_dir / path
        if candidate1.exists():
            path = candidate1
        elif candidate2.exists():
            path = candidate2
        else:
            raise ConfigurationError(f'Configuration file not found: {config_path}\nSearched in:\n  - {Path.cwd() / config_path}\n  - {candidate1}\n  - {candidate2}')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ConfigurationError(f'Unsupported config file format: {path.suffix}')
    except Exception as e:
        raise ConfigurationError(f'Error reading config file: {e}')

def create_simple_config(backend_type: str, model: str, system_message: Optional[str]=None, base_url: Optional[str]=None, ui_config: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    """Create a simple single-agent configuration."""
    backend_config = {'type': backend_type, 'model': model}
    if base_url:
        backend_config['base_url'] = base_url
    if backend_type == 'claude_code':
        backend_config['cwd'] = 'workspace1'
    if ui_config is None:
        ui_config = {'display_type': 'rich_terminal', 'logging_enabled': True}
    config = {'agent': {'id': 'agent1', 'backend': backend_config, 'system_message': system_message or 'You are a helpful AI assistant.'}, 'ui': ui_config}
    if backend_type == 'claude_code':
        config['orchestrator'] = {'snapshot_storage': '.massgen/snapshots', 'agent_temporary_workspace': '.massgen/temp_workspaces', 'session_storage': '.massgen/sessions'}
    return config

def validate_context_paths(config: Dict[str, Any]) -> None:
    """Validate that all context paths in the config exist.

    Context paths can be either files or directories.
    File-level context paths allow access to specific files without exposing sibling files.
    Raises ConfigurationError with clear message if any paths don't exist.
    """
    orchestrator_cfg = config.get('orchestrator', {})
    context_paths = orchestrator_cfg.get('context_paths', [])
    missing_paths = []
    for context_path_config in context_paths:
        if isinstance(context_path_config, dict):
            path = context_path_config.get('path')
        else:
            path = context_path_config
        if path:
            path_obj = Path(path)
            if not path_obj.exists():
                missing_paths.append(path)
    if missing_paths:
        errors = ['Context paths not found:']
        for path in missing_paths:
            errors.append(f'  - {path}')
        errors.append('\nPlease update your configuration with valid paths.')
        raise ConfigurationError('\n'.join(errors))

def relocate_filesystem_paths(config: Dict[str, Any]) -> None:
    """Relocate filesystem paths (orchestrator paths and agent workspaces) to be under .massgen/ directory.

    Modifies the config in-place to ensure all MassGen state is organized
    under .massgen/ for clean project structure.
    """
    massgen_dir = Path('.massgen')
    orchestrator_cfg = config.get('orchestrator', {})
    if orchestrator_cfg:
        path_fields = ['snapshot_storage', 'agent_temporary_workspace', 'session_storage']
        for field in path_fields:
            if field in orchestrator_cfg:
                user_path = orchestrator_cfg[field]
                if Path(user_path).is_absolute() or user_path.startswith('.massgen/'):
                    continue
                orchestrator_cfg[field] = str(massgen_dir / user_path)
    agent_entries = [config['agent']] if 'agent' in config else config.get('agents', [])
    for agent_data in agent_entries:
        backend_config = agent_data.get('backend', {})
        if 'cwd' in backend_config:
            user_cwd = backend_config['cwd']
            if Path(user_cwd).is_absolute() or user_cwd.startswith('.massgen/'):
                continue
            backend_config['cwd'] = str(massgen_dir / 'workspaces' / user_cwd)

def cli_main():
    """Synchronous wrapper for CLI entry point."""
    parser = argparse.ArgumentParser(description='MassGen - Multi-Agent Coordination CLI', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  # Use configuration file\n  python -m massgen.cli --config config.yaml "What is machine learning?"\n\n  # Quick single agent setup\n  python -m massgen.cli --backend openai --model gpt-4o-mini "Explain quantum computing"\n  python -m massgen.cli --backend claude --model claude-sonnet-4-20250514 "Analyze this data"\n\n  # Use ChatCompletion backend with custom base URL\n  python -m massgen.cli --backend chatcompletion --model gpt-oss-120b --base-url https://api.cerebras.ai/v1/chat/completions "What is 2+2?"\n\n  # Interactive mode\n  python -m massgen.cli --config config.yaml\n\n  # Timeout control examples\n  python -m massgen.cli --config config.yaml --orchestrator-timeout 600 "Complex task"\n\n  # Create sample configurations\n  python -m massgen.cli --create-samples\n\nEnvironment Variables:\n    OPENAI_API_KEY      - Required for OpenAI backend\n    XAI_API_KEY         - Required for Grok backend\n    ANTHROPIC_API_KEY   - Required for Claude backend\n    GOOGLE_API_KEY      - Required for Gemini backend (or GEMINI_API_KEY)\n    ZAI_API_KEY         - Required for ZAI backend\n\n    CEREBRAS_API_KEY    - For Cerebras AI (cerebras.ai)\n    TOGETHER_API_KEY    - For Together AI (together.ai, together.xyz)\n    FIREWORKS_API_KEY   - For Fireworks AI (fireworks.ai)\n    GROQ_API_KEY        - For Groq (groq.com)\n    NEBIUS_API_KEY      - For Nebius AI Studio (studio.nebius.ai)\n    OPENROUTER_API_KEY  - For OpenRouter (openrouter.ai)\n    POE_API_KEY         - For POE (poe.com)\n\n  Note: The chatcompletion backend auto-detects the provider from the base_url\n        and uses the appropriate environment variable for API key.\n        ')
    parser.add_argument('question', nargs='?', help='Question to ask (optional - if not provided, enters interactive mode)')
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument('--config', type=str, help='Path to YAML/JSON configuration file or @examples/NAME')
    config_group.add_argument('--backend', type=str, choices=['chatcompletion', 'claude', 'gemini', 'grok', 'openai', 'azure_openai', 'claude_code', 'zai', 'lmstudio', 'vllm', 'sglang'], help='Backend type for quick setup')
    parser.add_argument('--model', type=str, default=None, help='Model name for quick setup')
    parser.add_argument('--system-message', type=str, help='System message for quick setup')
    parser.add_argument('--base-url', type=str, help='Base URL for API endpoint (e.g., https://api.cerebras.ai/v1/chat/completions)')
    parser.add_argument('--no-display', action='store_true', help='Disable visual coordination display')
    parser.add_argument('--no-logs', action='store_true', help='Disable logging')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose logging')
    parser.add_argument('--init', action='store_true', help='Launch interactive configuration builder to create config file')
    parser.add_argument('--setup-keys', action='store_true', help='Launch interactive API key setup wizard to configure credentials')
    parser.add_argument('--list-examples', action='store_true', help='List available example configurations from package')
    parser.add_argument('--example', type=str, help='Print example config to stdout (e.g., --example basic_multi)')
    parser.add_argument('--show-schema', action='store_true', help='Display configuration schema and available parameters')
    parser.add_argument('--schema-backend', type=str, help='Show schema for specific backend (use with --show-schema)')
    parser.add_argument('--with-examples', action='store_true', help='Include example configurations in schema display')
    timeout_group = parser.add_argument_group('timeout settings', 'Override timeout settings from config')
    timeout_group.add_argument('--orchestrator-timeout', type=int, help='Maximum time for orchestrator coordination in seconds (default: 1800)')
    args = parser.parse_args()
    setup_logging(debug=args.debug)
    if args.debug:
        logger.info('Debug mode enabled')
        logger.debug(f'Command line arguments: {vars(args)}')
    if args.list_examples:
        show_available_examples()
        return
    if args.example:
        print_example_config(args.example)
        return
    if args.show_schema:
        from .schema_display import show_schema
        show_schema(backend=args.schema_backend, show_examples=args.with_examples)
        return
    if args.setup_keys:
        from .config_builder import ConfigBuilder
        builder = ConfigBuilder()
        api_keys = builder.interactive_api_key_setup()
        if any(api_keys.values()):
            print(f'\n{BRIGHT_GREEN}✅ API key setup complete!{RESET}')
            print(f'{BRIGHT_CYAN}💡 You can now use MassGen with these providers{RESET}\n')
        else:
            print(f'\n{BRIGHT_YELLOW}⚠️  No API keys configured{RESET}')
            print(f"{BRIGHT_CYAN}💡 You can run 'massgen --setup-keys' anytime to set them up{RESET}\n")
        return
    if args.init:
        from .config_builder import ConfigBuilder
        builder = ConfigBuilder()
        result = builder.run()
        if result and len(result) == 2:
            filepath, question = result
            if filepath and question:
                args.config = filepath
                args.question = question
            elif filepath:
                print(f'\n✅ Configuration saved to: {filepath}')
                print(f'Run with: python -m massgen.cli --config {filepath} "Your question"')
                return
            else:
                return
        else:
            return
    if not args.question and (not args.config) and (not args.model) and (not args.backend):
        if should_run_builder():
            print()
            print()
            print(f'{BRIGHT_CYAN}{'=' * 60}{RESET}')
            print(f'{BRIGHT_CYAN}  👋  Welcome to MassGen!{RESET}')
            print(f'{BRIGHT_CYAN}{'=' * 60}{RESET}')
            print()
            print("  Let's set up your default configuration...")
            print()
            from .config_builder import ConfigBuilder
            builder = ConfigBuilder(default_mode=True)
            result = builder.run()
            if result and len(result) == 2:
                filepath, question = result
                if filepath:
                    args.config = filepath
                    if question:
                        args.question = question
                    else:
                        print('\n✅ Configuration saved! You can now run queries.')
                        print('Example: massgen "Your question here"')
                        return
                else:
                    return
            else:
                return
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass

def show_available_examples():
    """Display available example configurations from package."""
    try:
        from importlib.resources import files
        configs_root = files('massgen') / 'configs'
        print(f'\n{BRIGHT_CYAN}Available Example Configurations{RESET}')
        print('=' * 60)
        categories = {}
        for config_file in sorted(configs_root.rglob('*.yaml')):
            rel_path = str(config_file).replace(str(configs_root) + '/', '')
            parts = rel_path.split('/')
            category = parts[0] if len(parts) > 1 else 'root'
            if category not in categories:
                categories[category] = []
            short_name = rel_path.replace('.yaml', '').replace('/', '_')
            categories[category].append((short_name, rel_path))
        for category, configs in sorted(categories.items()):
            print(f'\n{BRIGHT_YELLOW}{category.title()}:{RESET}')
            for short_name, rel_path in configs[:10]:
                print(f'  {BRIGHT_GREEN}@examples/{short_name:<40}{RESET} {rel_path}')
            if len(configs) > 10:
                print(f'  ... and {len(configs) - 10} more')
        print(f'\n{BRIGHT_BLUE}Usage:{RESET}')
        print('  massgen --config @examples/SHORTNAME "Your question"')
        print('  massgen --example SHORTNAME > my-config.yaml')
        print()
    except Exception as e:
        print(f'Error listing examples: {e}')
        print('Examples may not be available (development mode?)')

def should_run_builder() -> bool:
    """Check if config builder should run automatically.

    Returns True if:
    - No default config exists at ~/.config/massgen/config.yaml
    """
    default_config = Path.home() / '.config/massgen/config.yaml'
    return not default_config.exists()

def test_config_creation():
    """Test that we can create simple configurations."""
    try:
        from massgen.cli import create_simple_config
        print('  Testing OpenAI config creation...')
        config = create_simple_config(backend_type='openai', model='gpt-4o-mini')
        print(f'  Config result: {config}')
        if config and 'agent' in config and ('backend' in config['agent']) and (config['agent']['backend']['type'] == 'openai'):
            print('✅ OpenAI config creation works')
        else:
            print('❌ OpenAI config creation failed')
            print(f"  Expected: agent.backend.type 'openai', Got: {config}")
            return False
        print('  Testing Azure OpenAI config creation...')
        config = create_simple_config(backend_type='azure_openai', model='gpt-4.1')
        print(f'  Config result: {config}')
        if config and 'agent' in config and ('backend' in config['agent']) and (config['agent']['backend']['type'] == 'azure_openai'):
            print('✅ Azure OpenAI config creation works')
        else:
            print('❌ Azure OpenAI config creation failed')
            print(f"  Expected: agent.backend.type 'azure_openai', Got: {config}")
            return False
        return True
    except Exception as e:
        print(f'❌ Error during config creation test: {e}')
        import traceback
        traceback.print_exc()
        return False

