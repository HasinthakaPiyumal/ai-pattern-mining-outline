# Cluster 8

def process_documents(args) -> List[Document]:
    logger.info('Starting document processing')
    blocklist = load_blocklist('./')
    processor = DocumentProcessor(base_path=args.path, file_extensions=args.extensions, blocklist=blocklist, split_by=args.split_by, split_length=args.split_length, split_overlap=args.split_overlap, split_threshold=args.split_threshold)
    documents = processor.process_files()
    logger.info(f'Processed {len(documents)} documents')
    return documents

def load_blocklist(base_path: str) -> Set[str]:
    blocklist_file = Path(base_path) / '.ragignore'
    default_blocklist = set()
    if blocklist_file.exists():
        try:
            with open(blocklist_file, 'r') as f:
                custom_blocklist = {line.strip() for line in f if line.strip() and (not line.startswith('#'))}
            logger.info(f'Loaded custom blocklist from .ragignore: {custom_blocklist}')
            return default_blocklist.union(custom_blocklist)
        except Exception as e:
            logger.warning(f'Error reading .ragignore file: {e}. Using default blocklist.')
            return default_blocklist
    else:
        logger.info('No .ragignore file found. Using default blocklist.')
        return default_blocklist

def main():
    args = parse_args()
    log_args(args)
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('Debug mode enabled')
    try:
        logger.info('Initializing RAG Embedder')
        embedder = RAGEmbedder(provider_name=args.provider, ollama_url=args.ollama_url, es_url=args.es_url, es_index=args.es_index, es_basic_auth_user=args.es_basic_auth_user, es_basic_auth_password=args.es_basic_auth_password, embedding_model=args.embedding_model)
        logger.debug('RAG Embedder initialized successfully')
        if not args.path:
            logger.fatal('No path provided')
            exit(1)
        documents = process_documents(args)
        if documents:
            logger.info('Starting document embedding')
            embedder.embed_documents(documents)
            logger.info(f'Successfully embedded {len(documents)} documents')
            embedder.finalize()
        else:
            logger.warning('No documents to embed')
        if args.stats:
            logger.info('Retrieving pipeline statistics')
            embedder_stats = embedder.metrics_tracker.metrics
            logger.info(f'Embedder Metrics: {embedder_stats}')
    except Exception:
        logger.error('Error in pipeline execution', exc_info=True)
        raise

def log_args(args):
    logger.info('Configuration:')
    config_dict = {'Elasticsearch URL': args.es_url, 'Ollama URL': args.ollama_url, 'Embedding Model': args.embedding_model, 'Document Path': args.path or 'Not specified', 'File Extensions': ', '.join(args.extensions), 'Debug Mode': args.debug}
    for key, value in config_dict.items():
        logger.info(f'{key}: {value}')

