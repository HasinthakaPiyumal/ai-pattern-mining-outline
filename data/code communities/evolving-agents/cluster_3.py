# Cluster 3

class LLMCache:
    """
    Cache for LLM completions and embeddings using MongoDB with TTL.
    """

    def __init__(self, mongodb_client: MongoDBClient, ttl: int=86400 * 7, collection_name: str='eat_llm_cache'):
        """
        Initialize the LLM cache with MongoDB.

        Args:
            mongodb_client: Instance of MongoDBClient for database operations.
            ttl: Time-to-live for cache entries in seconds.
            collection_name: Name of the MongoDB collection for caching.
        """
        self.mongodb_client = mongodb_client
        self.ttl = ttl
        self.cache_collection_name = collection_name
        self.cache_collection = self.mongodb_client.get_collection(self.cache_collection_name)
        asyncio.create_task(self._ensure_indexes())
        logger.info(f"Initialized LLM cache with MongoDB collection '{self.cache_collection_name}' and TTL {self.ttl}s.")

    async def _ensure_indexes(self):
        """Ensure necessary indexes on the cache collection, especially TTL."""
        try:
            await self.cache_collection.create_index([('created_at', pymongo.ASCENDING)], expireAfterSeconds=self.ttl, name='ttl_created_at_index')
            logger.info(f"Ensured TTL index on '{self.cache_collection_name}' for 'created_at' field.")
        except Exception as e:
            logger.error(f'Error creating TTL index for {self.cache_collection_name}: {e}', exc_info=True)

    def _generate_key(self, data: Any, model_id: str, cache_type: str) -> str:
        """Generate a unique key for the given input data, model, and type."""
        try:
            if isinstance(data, list) and all((isinstance(item, UserMessage) for item in data)):
                serializable_data = []
                for msg in data:
                    content = msg.content.text if hasattr(msg.content, 'text') else str(msg.content)
                    serializable_data.append({'content': content, 'role': msg.role})
            else:
                serializable_data = data if isinstance(data, str) else json_serializable(data)
            key_structure = {'data': serializable_data, 'model_id': model_id, 'cache_type': cache_type}
            try:
                data_str = json.dumps(key_structure, sort_keys=True, default=json_serializable)
            except (TypeError, OverflowError) as e:
                logger.warning(f'Error serializing cache key data: {e}. Using string representation.')
                if isinstance(serializable_data, str):
                    data_str = f'{serializable_data}::{model_id}::{cache_type}'
                else:
                    data_str = f'{str(serializable_data)}::{model_id}::{cache_type}'
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f'Error generating cache key: {e}', exc_info=True)
            fallback = f'{str(data)[:100]}::{model_id}::{cache_type}::{time.time()}'
            return hashlib.md5(fallback.encode()).hexdigest()

    async def get_completion(self, messages: List[UserMessage], model_id: str) -> Optional[str]:
        """Get a cached completion result from MongoDB."""
        try:
            key = self._generate_key(messages, model_id, 'completion')
            cached_doc = await self.cache_collection.find_one({'_id': key})
            if cached_doc:
                logger.info(f'Cache hit for completion (key: {key[:8]}...).')
                return cached_doc.get('cached_data')
        except Exception as e:
            logger.warning(f'Error reading completion cache from MongoDB: {e}', exc_info=True)
        return None

    async def save_completion(self, messages: List[UserMessage], model_id: str, response: str) -> None:
        """Save a completion result to MongoDB cache."""
        try:
            key = self._generate_key(messages, model_id, 'completion')
            cache_entry = {'_id': key, 'model_id': model_id, 'cache_type': 'completion', 'cached_data': response, 'created_at': datetime.now(timezone.utc)}
            await self.cache_collection.replace_one({'_id': key}, cache_entry, upsert=True)
            logger.info(f'Cached completion (key: {key[:8]}...).')
        except Exception as e:
            logger.warning(f'Error writing completion cache to MongoDB: {e}', exc_info=True)

    async def get_embedding(self, text: str, model_id: str) -> Optional[List[float]]:
        """Get a cached embedding from MongoDB."""
        try:
            key = self._generate_key(text, model_id, 'embedding')
            cached_doc = await self.cache_collection.find_one({'_id': key})
            if cached_doc:
                logger.info(f'Cache hit for embedding (key: {key[:8]}...).')
                cached_data = cached_doc.get('cached_data')
                if isinstance(cached_data, list) and all((isinstance(item, (int, float)) for item in cached_data)):
                    return cached_data
                else:
                    logger.warning(f'Invalid embedding data format in cache. Expected list of floats, got: {type(cached_data)}')
                    return None
        except Exception as e:
            logger.warning(f'Error reading embedding cache from MongoDB: {e}', exc_info=True)
        return None

    async def save_embedding(self, text: str, model_id: str, embedding: List[float]) -> None:
        """Save an embedding to MongoDB cache."""
        try:
            if not isinstance(embedding, list) or not all((isinstance(item, (int, float)) for item in embedding)):
                logger.warning(f'Invalid embedding format. Expected list of floats, got: {type(embedding)}')
                return
            key = self._generate_key(text, model_id, 'embedding')
            cache_entry = {'_id': key, 'model_id': model_id, 'cache_type': 'embedding', 'cached_data': [float(x) for x in embedding], 'created_at': datetime.now(timezone.utc)}
            await self.cache_collection.replace_one({'_id': key}, cache_entry, upsert=True)
            logger.info(f'Cached embedding (key: {key[:8]}...).')
        except Exception as e:
            logger.warning(f'Error writing embedding cache to MongoDB: {e}', exc_info=True)

    async def get_batch_embeddings(self, texts: List[str], model_id: str) -> Optional[List[List[float]]]:
        """Get cached batch embeddings if all are available from MongoDB."""
        results = []
        all_found = True
        for text in texts:
            embedding = await self.get_embedding(text, model_id)
            if embedding is None:
                all_found = False
                break
            results.append(embedding)
        return results if all_found else None

    async def clear_cache(self, older_than_seconds: Optional[int]=None) -> int:
        """
        Clear the MongoDB cache.
        If older_than_seconds is specified, removes entries older than that.
        Otherwise, removes all entries.

        Args:
            older_than_seconds: Clear entries older than this many seconds. If None, clear all.

        Returns:
            Number of entries removed.
        """
        query_filter = {}
        if older_than_seconds is not None:
            cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=older_than_seconds)
            query_filter = {'created_at': {'$lt': cutoff_time}}
        try:
            result = await self.cache_collection.delete_many(query_filter)
            deleted_count = result.deleted_count
            logger.info(f'Cleared {deleted_count} cache entries from MongoDB based on filter: {query_filter}.')
            return deleted_count
        except Exception as e:
            logger.error(f'Error clearing cache from MongoDB: {e}', exc_info=True)
            return 0

def json_serializable(obj):
    """Convert an object to a JSON-serializable format."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, set):
        return list(obj)
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()
        return str(obj)

