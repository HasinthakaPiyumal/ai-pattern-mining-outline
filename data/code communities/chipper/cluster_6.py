# Cluster 6

def main():
    parser = argparse.ArgumentParser(description=f'Chat CLI {APP_VERSION}.{BUILD_NUMBER}')
    parser.add_argument('--host', default=os.getenv('API_HOST', '0.0.0.0'), help='API Host')
    parser.add_argument('--port', default=os.getenv('API_PORT', '8000'), help='API Port')
    parser.add_argument('--api_key', default=os.getenv('API_KEY'), help='API Key')
    parser.add_argument('--timeout', type=int, default=int(os.getenv('API_TIMEOUT', '120')), help='API Timeout')
    parser.add_argument('--verify_ssl', action='store_true', default=os.getenv('REQUIRE_SECURE', 'False').lower() == 'true', help='Verify SSL')
    parser.add_argument('--log_level', default=os.getenv('LOG_LEVEL', 'INFO'), help='Log Level')
    parser.add_argument('--max_context_size', type=int, default=int(os.getenv('MAX_CONTEXT_SIZE', '10')), help='Maximum Context Size')
    parser.add_argument('--model', default=os.getenv('MODEL_NAME'), help='Model name to use')
    parser.add_argument('--index', default=os.getenv('ES_INDEX'), help='Index to use')
    args = parser.parse_args()
    base_url = f'http://{args.host}:{args.port}'
    config = Config(base_url=base_url, api_key=args.api_key, timeout=args.timeout, verify_ssl=args.verify_ssl, log_level=args.log_level, max_context_size=args.max_context_size, max_retries=3, retry_delay=1.0, model=args.model, index=args.index, streaming=False)
    setup_logging(config.log_level)
    chat = ChatInterface(config)
    asyncio.run(chat.run())

def setup_logging(log_level):
    logging.basicConfig(level=log_level, format='%(message)s', handlers=[RichHandler(rich_tracebacks=True)])

