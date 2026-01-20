# Cluster 3

def execute_attack(args):
    """Execute Attack - Login attacks"""
    print('🚀 BruteForceAI Attack - Login Attacks')
    print('=' * 80)
    print(f'Mode: {args.mode}')
    print(f'Attack method: {args.attack}')
    print(f'Threads: {args.threads}')
    print(f'Retry attempts: {args.retry_attempts}')
    print(f'DOM threshold: {args.dom_threshold}')
    print(f'Delay: {args.delay}s')
    print(f'Jitter: {args.jitter}s')
    print(f'Success exit: {args.success_exit}')
    print(f'User agents: {args.user_agents or 'Default browser'}')
    print(f'URLs file: {args.urls}')
    print(f'Usernames file: {args.usernames}')
    print(f'Passwords file: {args.passwords}')
    print(f'Database: {args.database}')
    print(f'Show browser: {args.show_browser}')
    print(f'Browser wait: {args.browser_wait}s')
    print(f'Proxy: {args.proxy or 'None'}')
    print(f'Debug: {args.debug}')
    print(f'Verbose: {args.verbose}')
    print(f'Force retry: {args.force_retry}')
    webhooks_configured = []
    if getattr(args, 'discord_webhook', None):
        webhooks_configured.append('Discord')
    if getattr(args, 'slack_webhook', None):
        webhooks_configured.append('Slack')
    if getattr(args, 'teams_webhook', None):
        webhooks_configured.append('Teams')
    if getattr(args, 'telegram_webhook', None) and getattr(args, 'telegram_chat_id', None):
        webhooks_configured.append('Telegram')
    if webhooks_configured:
        print(f'Webhooks: {', '.join(webhooks_configured)}')
    else:
        print(f'Webhooks: None')
    print('=' * 80)
    bf = BruteForceAI(urls_file=args.urls, usernames_file=args.usernames, passwords_file=args.passwords, show_browser=args.show_browser, browser_wait=args.browser_wait, proxy=args.proxy, database=args.database, debug=args.debug, retry_attempts=args.retry_attempts, dom_threshold=args.dom_threshold, verbose=args.verbose, delay=args.delay, jitter=args.jitter, success_exit=args.success_exit, user_agents_file=args.user_agents, force_retry=args.force_retry, discord_webhook=getattr(args, 'discord_webhook', None), slack_webhook=getattr(args, 'slack_webhook', None), teams_webhook=getattr(args, 'teams_webhook', None), telegram_webhook=getattr(args, 'telegram_webhook', None), telegram_chat_id=getattr(args, 'telegram_chat_id', None), ollama_url=getattr(args, 'ollama_url', None))
    bf.stage2(mode=args.mode, attack=args.attack, threads=args.threads)
    print('\n✅ Attack completed!')

def execute_clean_db(args):
    """Clean database tables"""
    print('🧹 BruteForceAI Database Cleanup')
    print('=' * 50)
    print(f'Database: {args.database}')
    print('=' * 50)
    bf = BruteForceAI(urls_file=[], usernames_file=[], passwords_file=[], database=args.database)
    bf.clean_database()

