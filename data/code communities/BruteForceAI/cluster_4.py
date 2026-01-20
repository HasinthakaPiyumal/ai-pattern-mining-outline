# Cluster 4

def execute_analyze(args):
    """Execute Analyze - Login form analysis"""
    print('🚀 BruteForceAI Analyze - Login Form Analysis')
    print('=' * 50)
    print(f'URLs file: {args.urls}')
    print(f'LLM provider: {args.llm_provider or 'None'}')
    llm_provider = args.llm_provider
    llm_model = args.llm_model
    if not llm_provider:
        llm_provider = 'ollama'
        print(f'LLM provider: {llm_provider} (default)')
    else:
        print(f'LLM provider: {llm_provider}')
    if not llm_model:
        if llm_provider == 'ollama':
            llm_model = 'llama3.2:3b'
            print(f'LLM model: {llm_model} (default for Ollama)')
        elif llm_provider == 'groq':
            llm_model = 'llama-3.3-70b-versatile'
            print(f'LLM model: {llm_model} (default for Groq)')
    else:
        print(f'LLM model: {llm_model}')
    print(f'Selector retry: {args.selector_retry}')
    print(f'Show browser: {args.show_browser}')
    print(f'Browser wait: {args.browser_wait}s')
    print(f'Proxy: {args.proxy or 'None'}')
    print(f'Database: {args.database}')
    print(f'Force reanalyze: {args.force_reanalyze}')
    print(f'Debug: {args.debug}')
    print(f'User agents: {args.user_agents or 'Default browser'}')
    print('=' * 50)
    from BruteForceCore import _validate_llm_setup
    _validate_llm_setup(llm_provider, llm_model, args.llm_api_key, args.ollama_url)
    bf = BruteForceAI(urls_file=args.urls, usernames_file=[], passwords_file=[], selector_retry=args.selector_retry, show_browser=args.show_browser, browser_wait=args.browser_wait, proxy=args.proxy, database=args.database, llm_provider=llm_provider, llm_model=llm_model, llm_api_key=args.llm_api_key, ollama_url=args.ollama_url, force_reanalyze=args.force_reanalyze, debug=args.debug, user_agents_file=args.user_agents)
    print(f'\n🚀 Starting analysis of {len(bf.urls)} URL(s)...')
    for i, url in enumerate(bf.urls, 1):
        print(f'\n[{i}/{len(bf.urls)}] Analyzing: {url}')
        result = bf.stage1(url)
        if result and result.get('success'):
            print(f'✅ Analysis completed successfully for {url}')
        else:
            print(f'❌ Analysis failed for {url}')
    print('\n✅ Analyze completed!')

