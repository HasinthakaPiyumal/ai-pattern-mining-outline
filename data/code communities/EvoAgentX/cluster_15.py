# Cluster 15

def _create_default_storage_config(db_path: Optional[str]=None) -> StoreConfig:
    """
    Create a default storage configuration with proper path handling.
    
    Args:
        db_path (str, optional): Custom database path
        
    Returns:
        StoreConfig: Configured storage configuration
    """
    from ..storages.storages_config import StoreConfig, DBConfig, VectorStoreConfig
    if db_path is None:
        db_path = './faiss_db.sqlite'
    validated_db_path = _ensure_database_path(db_path)
    logger.info(f'Using validated database path: {validated_db_path}')
    index_cache_path = str(Path(validated_db_path).parent.resolve() / 'index_cache')
    storage_config = StoreConfig(dbConfig=DBConfig(db_name='sqlite', path=validated_db_path), vectorConfig=VectorStoreConfig(vector_name='faiss', dimensions=1536, index_type='flat_l2'), path=index_cache_path)
    Path(index_cache_path).mkdir(parents=True, exist_ok=True)
    return storage_config

def _ensure_database_path(db_path: str) -> str:
    """
    Ensure the database path exists and is properly configured.
    
    Args:
        db_path (str): The database file path
        
    Returns:
        str: The validated and prepared database path
        
    Raises:
        ValueError: If the path is invalid or cannot be created
    """
    if not db_path:
        raise ValueError('Database path cannot be empty')
    path = Path(db_path).resolve()
    if path.exists() and path.is_dir():
        raise ValueError(f'Database path points to a directory: {db_path}')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ValueError(f'Cannot create directory for database path {db_path}: {e}')
    if path.exists():
        logger.info(f'Found existing database at: {db_path}')
        try:
            import sqlite3
            conn = sqlite3.connect(str(path))
            conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            conn.close()
            logger.info('Database validation successful')
        except Exception as e:
            logger.warning(f'Database validation failed: {e}. Will create new database.')
            try:
                path.unlink()
            except Exception as unlink_error:
                logger.error(f'Failed to remove corrupted database file: {unlink_error}')
                raise ValueError(f'Cannot remove corrupted database file: {unlink_error}')
    else:
        logger.info(f'Database not found at: {db_path}. Will create new database.')
    return str(path)

class FaissToolkit(Toolkit):
    """
    Toolkit for FAISS vector database operations.
    
    This toolkit provides a comprehensive set of tools for interacting with FAISS vector databases,
    including semantic search, document insertion, deletion, and database management operations.
    
    The toolkit integrates with the existing RAG engine and storage infrastructure to provide
    a unified interface for vector database operations that can be easily used by agents.
    """

    def __init__(self, name: str='FaissToolkit', storage_config: Optional[StoreConfig]=None, rag_config: Optional[RAGConfig]=None, default_corpus_id: str='default', default_index_type: str='vector', db_path: Optional[str]=None, storage_handler: StorageHandler=None, file_handler: FileStorageHandler=None, **kwargs):
        """
        Initialize the FAISS toolkit.
        
        Args:
            name (str): Name of the toolkit
            storage_config (StoreConfig, optional): Configuration for storage backends
            rag_config (RAGConfig, optional): Configuration for RAG pipeline
            default_corpus_id (str): Default corpus ID for operations
            default_index_type (str): Default index type for vector operations
            db_path (str, optional): Custom database path. If provided, will check for existing database or create new one
            storage_handler (StorageHandler, optional): Storage handler for file operations
            file_handler (FileStorageHandler, optional): File handler for file operations
            **kwargs: Additional arguments
        """
        if storage_config is None:
            storage_config = _create_default_storage_config(db_path)
        if rag_config is None:
            rag_config = _create_default_rag_config()
        faiss_database = FaissDatabase(storage_config=storage_config, rag_config=rag_config, default_corpus_id=default_corpus_id, default_index_type=default_index_type, storage_handler=storage_handler, file_handler=file_handler)
        tools = [FaissQueryTool(faiss_database), FaissInsertTool(faiss_database), FaissDeleteTool(faiss_database), FaissListTool(faiss_database), FaissStatsTool(faiss_database)]
        super().__init__(name=name, tools=tools, **kwargs)
        self.faiss_database = faiss_database
        logger.info(f'Initialized {name} with {len(tools)} tools')

    def get_database(self) -> FaissDatabase:
        """
        Get the underlying FAISS database instance.
        
        Returns:
            FaissDatabase: The FAISS database instance
        """
        return self.faiss_database

def _create_default_rag_config() -> RAGConfig:
    """
    Create a default RAG configuration.
    
    Returns:
        RAGConfig: Configured RAG configuration
    """
    from ..rag.rag_config import RAGConfig, EmbeddingConfig, ChunkerConfig
    return RAGConfig(embedding=EmbeddingConfig(provider='openai', model_name='text-embedding-ada-002'), chunker=ChunkerConfig(chunk_size=500, chunk_overlap=50))

