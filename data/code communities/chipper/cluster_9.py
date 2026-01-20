# Cluster 9

def show_welcome():
    RED = '\x1b[31m'
    YELLOW = '\x1b[33m'
    RESET = '\x1b[0m'
    print('\n', flush=True)
    print(f'{RED}', flush=True)
    print('        __    _                      ', flush=True)
    print('  _____/ /_  (_)___  ____  ___  _____', flush=True)
    print(' / ___/ __ \\/ / __ \\/ __ \\/ _ \\/ ___/', flush=True)
    print('/ /__/ / / / / /_/ / /_/ /  __/ /    ', flush=True)
    print('\\___/_/ /_/_/ .___/ .___/\\___/_/     ', flush=True)
    print('           /_/   /_/                 ', flush=True)
    print(f'{RESET}', flush=True)
    print(f'{YELLOW}       Chipper Embed {APP_VERSION}.{BUILD_NUMBER}', flush=True)
    print(f'{RESET}\n', flush=True)

def main():
    args = parse_args()
    asyncio.run(run_scrapers(args))

def parse_args():
    parser = argparse.ArgumentParser(description=f'Chipper Embed CLI {APP_VERSION}.{BUILD_NUMBER}')
    parser.add_argument('--path', type=str, default='/app/data', help='Base path to process documents from')
    parser.add_argument('--extensions', type=str, nargs='+', default=['.txt', '.md', '.rst', '.log', '.csv', '.json', '.yaml', '.yml', '.html', '.htm', '.css', '.js', '.jsx', '.ts', '.tsx', '.php', '.py', '.pyx', '.pyi', '.ipynb', '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx', '.java', '.kt', '.gradle', '.cs', '.csproj', '.cshtml', '.rb', '.erb', '.rake', '.sh', '.bash', '.zsh', '.bat', '.cmd', '.ps1', '.vbs', '.vbe', '.js', '.jse', '.wsf', '.wsh', '.scpt', '.scptd', '.applescript', '.xml', '.ini', '.conf', '.cfg', '.toml', '.qml', '.ui', '.rs', '.go', '.swift'], help='List of file extensions to process')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug logging')
    parser.add_argument('--provider', type=str, default=None, choices=['ollama', 'hf'], help='Embedding provider')
    parser.add_argument('--es-url', type=str, default=os.getenv('ES_URL', 'http://localhost:9200'), help='URL for the Elasticsearch service')
    parser.add_argument('--es-index', type=str, default=os.getenv('ES_INDEX', 'default'), help='Index for the Elasticsearch service')
    parser.add_argument('--es-basic-auth-user', type=str, default=os.getenv('ES_BASIC_AUTH_USERNAME', ''), help='Username for the Elasticsearch service authentication')
    parser.add_argument('--es-basic-auth-password', type=str, default=os.getenv('ES_BASIC_AUTH_PASSWORD', ''), help='Password for the Elasticsearch service authentication')
    parser.add_argument('--ollama-url', type=str, default=os.getenv('OLLAMA_URL', 'http://localhost:11434'), help='URL for the Ollama service')
    parser.add_argument('--embedding-model', type=str, default=None, help='Model to use for embeddings')
    parser.add_argument('--split-by', type=str, default='word', choices=['word', 'sentence', 'passage', 'page', 'line'], help='Method to split text documents')
    parser.add_argument('--split-length', type=int, default=200, help='Number of units per split')
    parser.add_argument('--split-overlap', type=int, default=20, help='Number of units to overlap between splits')
    parser.add_argument('--split-threshold', type=int, default=5, help='Minimum length of split to keep')
    parser.add_argument('--stats', action='store_true', default=False, help='Enable statistics logging')
    args = parser.parse_args()
    return args

def create_app():
    try:
        setup_all_routes(app)
        logger.info(f'Initialized Chipper API {APP_VERSION}.{BUILD_NUMBER}')
        return app
    except Exception as e:
        logger.error(f'Failed to initialize application: {e}', exc_info=True)
        raise

def get_server_config():
    return {'host': os.getenv('HOST', '0.0.0.0'), 'port': int(os.getenv('PORT', '8000')), 'debug': os.getenv('DEBUG', 'False').lower() == 'true'}

