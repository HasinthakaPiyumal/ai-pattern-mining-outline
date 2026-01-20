# Cluster 1

def main():
    parser = argparse.ArgumentParser(description='BruteForceAI - AI-Powered Login Form Analysis and Brute Force Attack Tool using LLM', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  # Analyze - Analyze login forms (simplest - uses default ollama + llama3.2:3b)\n  python BruteForceAI.py analyze --urls urls.txt\n\n  # Analyze - Analyze login forms (with default models)\n  python BruteForceAI.py analyze --urls urls.txt --llm-provider ollama\n  python BruteForceAI.py analyze --urls urls.txt --llm-provider groq --llm-api-key "your_api_key"\n\n  # Analyze - Analyze login forms (with specific models)\n  python BruteForceAI.py analyze --urls urls.txt --llm-provider ollama --llm-model llama3.2:3b\n  python BruteForceAI.py analyze --urls urls.txt --llm-provider groq --llm-model llama-3.1-70b-versatile --llm-api-key "your_api_key"\n  \n  # Analyze - Custom Ollama server\n  python BruteForceAI.py analyze --urls urls.txt --llm-provider ollama --ollama-url http://192.168.1.100:11434\n\n  # Attack - Brute force attack\n  python BruteForceAI.py attack --urls urls.txt --usernames usernames.txt --passwords passwords.txt\n\n  # Attack - Password spray with threads\n  python BruteForceAI.py attack --urls urls.txt --usernames usernames.txt --passwords passwords.txt --mode passwordspray --threads 3\n\n  # Attack with Discord webhook notifications\n  python BruteForceAI.py attack --urls urls.txt --usernames usernames.txt --passwords passwords.txt --discord-webhook "https://discord.com/api/webhooks/..."\n\n  # Attack with Telegram notifications\n  python BruteForceAI.py attack --urls urls.txt --usernames usernames.txt --passwords passwords.txt --telegram-webhook "BOT_TOKEN" --telegram-chat-id "CHAT_ID"\n\n  # Attack with multiple webhooks\n  python BruteForceAI.py attack --urls urls.txt --usernames usernames.txt --passwords passwords.txt --discord-webhook "..." --slack-webhook "..."\n\n  # Save output to file\n  python BruteForceAI.py attack --urls urls.txt --usernames usernames.txt --passwords passwords.txt --output results.txt\n\n  # Clean database\n  python BruteForceAI.py clean-db\n\n  # Check for updates\n  python BruteForceAI.py check-updates\n\n  # Skip version check for faster startup\n  python BruteForceAI.py attack --urls urls.txt --usernames usernames.txt --passwords passwords.txt --skip-version-check\n        ')
    parser.add_argument('--no-color', '-nc', action='store_true', help='Disable colored output')
    parser.add_argument('--output', '-o', help='Save all output to file (from start to finish)')
    parser.add_argument('--skip-version-check', action='store_true', help='Skip automatic version checking')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    analyze_parser = subparsers.add_parser('analyze', help='Analyze login forms and identify selectors')
    analyze_parser.add_argument('--urls', required=True, help='File containing URLs (one per line)')
    analyze_parser.add_argument('--llm-provider', choices=['ollama', 'groq'], help='LLM provider for analysis (default: ollama)')
    analyze_parser.add_argument('--llm-model', help='LLM model name (default: llama3.2:3b for Ollama, llama-3.3-70b-versatile for Groq)')
    analyze_parser.add_argument('--llm-api-key', help='API key for Groq (not needed for Ollama)')
    analyze_parser.add_argument('--ollama-url', help='Ollama server URL (default: http://localhost:11434)')
    analyze_parser.add_argument('--selector-retry', type=int, default=10, help='Number of retry attempts for selectors')
    analyze_parser.add_argument('--show-browser', action='store_true', help='Show browser window during analysis')
    analyze_parser.add_argument('--browser-wait', type=int, default=0, help='Wait time in seconds when browser is visible')
    analyze_parser.add_argument('--proxy', help='Proxy server (e.g., http://127.0.0.1:8080)')
    analyze_parser.add_argument('--database', default='bruteforce.db', help='SQLite database file path')
    analyze_parser.add_argument('--force-reanalyze', action='store_true', help='Force re-analysis even if selectors exist')
    analyze_parser.add_argument('--debug', action='store_true', help='Enable debug output')
    analyze_parser.add_argument('--user-agents', help='File containing User-Agent strings (one per line) for random selection')
    analyze_parser.add_argument('--output', '-o', help='Save all output to file (from start to finish)')
    analyze_parser.add_argument('--no-color', '-nc', action='store_true', help='Disable colored output')
    analyze_parser.add_argument('--skip-version-check', action='store_true', help='Skip automatic version checking')
    attack_parser = subparsers.add_parser('attack', help='Execute login attacks using analyzed selectors')
    attack_parser.add_argument('--urls', required=True, help='File containing URLs (one per line)')
    attack_parser.add_argument('--usernames', required=True, help='File containing usernames (one per line)')
    attack_parser.add_argument('--passwords', required=True, help='File containing passwords (one per line)')
    attack_parser.add_argument('--mode', choices=['bruteforce', 'passwordspray'], default='bruteforce', help='Attack mode: bruteforce (all combinations) or passwordspray (each password against all users)')
    attack_parser.add_argument('--attack', choices=['playwright'], default='playwright', help='Attack method (only playwright supported)')
    attack_parser.add_argument('--threads', type=int, default=1, help='Number of threads to use for parallel attacks')
    attack_parser.add_argument('--retry-attempts', type=int, default=3, help='Number of retry attempts for network errors (default: 3)')
    attack_parser.add_argument('--dom-threshold', type=int, default=100, help='DOM length difference threshold for success detection (default: 100)')
    attack_parser.add_argument('--delay', type=float, default=0, help='Delay in seconds between attempts (bruteforce: between passwords for same user, passwordspray: between passwords)')
    attack_parser.add_argument('--jitter', type=float, default=0, help='Random jitter in seconds to add to delays (0-jitter range) for more human-like timing')
    attack_parser.add_argument('--success-exit', action='store_true', help='Stop attack for each URL after first successful login is found')
    attack_parser.add_argument('--user-agents', help='File containing User-Agent strings (one per line) for random selection')
    attack_parser.add_argument('--show-browser', action='store_true', help='Show browser window during attacks')
    attack_parser.add_argument('--browser-wait', type=int, default=0, help='Wait time in seconds when browser is visible')
    attack_parser.add_argument('--proxy', help='Proxy server (e.g., http://127.0.0.1:8080)')
    attack_parser.add_argument('--database', default='bruteforce.db', help='SQLite database file path')
    attack_parser.add_argument('--debug', action='store_true', help='Enable debug output')
    attack_parser.add_argument('--verbose', action='store_true', help='Show detailed timestamps for each attempt')
    attack_parser.add_argument('--force-retry', action='store_true', help='Force retry attempts that already exist in the database (default: skip existing)')
    attack_parser.add_argument('--output', '-o', help='Save all output to file (from start to finish)')
    attack_parser.add_argument('--no-color', '-nc', action='store_true', help='Disable colored output')
    attack_parser.add_argument('--skip-version-check', action='store_true', help='Skip automatic version checking')
    attack_parser.add_argument('--discord-webhook', help='Discord webhook URL for success notifications')
    attack_parser.add_argument('--slack-webhook', help='Slack webhook URL for success notifications')
    attack_parser.add_argument('--teams-webhook', help='Microsoft Teams webhook URL for success notifications')
    attack_parser.add_argument('--telegram-webhook', help='Telegram bot token for success notifications')
    attack_parser.add_argument('--telegram-chat-id', help='Telegram chat ID for notifications (required with --telegram-webhook)')
    clean_parser = subparsers.add_parser('clean-db', help='Clean (truncate) all database tables')
    clean_parser.add_argument('--database', default='bruteforce.db', help='SQLite database file path')
    clean_parser.add_argument('--output', '-o', help='Save all output to file (from start to finish)')
    clean_parser.add_argument('--no-color', '-nc', action='store_true', help='Disable colored output')
    clean_parser.add_argument('--skip-version-check', action='store_true', help='Skip automatic version checking')
    updates_parser = subparsers.add_parser('check-updates', help='Check for software updates')
    updates_parser.add_argument('--output', '-o', help='Save all output to file (from start to finish)')
    updates_parser.add_argument('--no-color', '-nc', action='store_true', help='Disable colored output')
    updates_parser.add_argument('--skip-version-check', action='store_true', help='Skip automatic version checking')
    args = parser.parse_args()
    global_skip_version_check = '--skip-version-check' in sys.argv
    output_capture = None
    output_file_arg = getattr(args, 'output', None) or args.output if hasattr(args, 'output') else None
    if output_file_arg:
        output_file = output_file_arg
        if not output_file.endswith('.txt') and (not output_file.endswith('.log')):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'{output_file}_{timestamp}.txt'
        output_capture = OutputCapture(output_file)
        if not output_capture.start():
            sys.exit(1)
        print(f'📄 Output capture started - saving to: {output_file}')
        print(f'🕐 Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')
        print('=' * 80)
    try:
        no_color = getattr(args, 'no_color', False)
        skip_version_check = getattr(args, 'skip_version_check', False) or global_skip_version_check
        print_banner(no_color=no_color, check_updates=not skip_version_check)
        if not args.command:
            parser.print_help()
            sys.exit(1)
        if args.command == 'analyze':
            execute_analyze(args)
        elif args.command == 'attack':
            execute_attack(args)
        elif args.command == 'clean-db':
            execute_clean_db(args)
        elif args.command == 'check-updates':
            execute_check_updates(args)
    except KeyboardInterrupt:
        print('\n\n🛑 Operation interrupted by user (Ctrl+C)')
        if output_capture:
            print(f'🕐 Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')
    except Exception as e:
        print(f'\n\n❌ Unexpected error: {e}')
        if output_capture:
            print(f'🕐 Session ended with error: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')
        raise
    finally:
        if output_capture:
            print('\n' + '=' * 80)
            print(f'🕐 Session completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')
            output_capture.stop()

def print_banner(no_color=False, check_updates=True):
    """Print colorful banner with tool information"""
    if no_color:
        Colors.disable()
    banner = f'{Colors.RED}{Colors.BOLD}\n  █▀▄ █▀▄ █ █ ▀█▀ █▀▀ █▀▀ █▀█ █▀▄ █▀▀ █▀▀   █▀█ ▀█▀ \n  █▀▄ █▀▄ █ █  █  █▀▀ █▀▀ █ █ █▀▄ █   █▀▀   █▀█  █   \n  ▀▀  ▀ ▀ ▀▀▀  ▀  ▀▀▀ ▀   ▀▀▀ ▀ ▀ ▀▀▀ ▀▀▀   ▀ ▀ ▀▀▀ {Colors.RESET}\n{Colors.YELLOW}{Colors.BOLD}🤖 BruteForceAI Attack - Smart brute-force tool using LLM 🧠{Colors.RESET}\n{Colors.CYAN}{Colors.BOLD}Version {CURRENT_VERSION} | Author: Mor David (www.mordavid.com) | License: Non-Commercial{Colors.RESET}\n'
    print(banner)
    if check_updates:
        check_for_updates(silent=False)

