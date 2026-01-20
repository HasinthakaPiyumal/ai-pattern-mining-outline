# Cluster 28

class GraphRetriever(BaseRetrieverWrapper):
    """Wrapper for graph-based retrieval."""

    def __init__(self, llm: BaseLLM, graph_store: PropertyGraphStore, embed_model: Optional[BaseEmbedding], include_text: bool=True, _use_async: bool=True, vector_store: Optional[BasePydanticVectorStore]=None, top_k: int=5):
        super().__init__()
        self.graph_store = graph_store
        self._embed_model = embed_model
        self.vector_store = vector_store
        self._llm = llm
        sub_retrievers = [BasicLLMSynonymRetriever(graph_store=graph_store, include_text=include_text, llm=llm)]
        if self._embed_model and (self.graph_store.supports_vector_queries or self.vector_store):
            sub_retrievers.append(VectorContextRetriever(graph_store=self.graph_store, vector_store=self.vector_store, include_text=include_text, embed_model=self._embed_model, similarity_top_k=top_k))
        self.retriever = PGRetriever(sub_retrievers, use_async=_use_async)

    async def aretrieve(self, query: Query) -> RagResult:
        try:
            subretriever_bool = [isinstance(sub, VectorContextRetriever) for sub in self.retriever.sub_retrievers]
            if any(subretriever_bool):
                ind = subretriever_bool.index(True)
                self.retriever.sub_retrievers[ind]._similarity_top_k = query.top_k
            nodes = await self.retriever.aretrieve(query.query_str)
            corpus = Corpus()
            scores = []
            if nodes is None:
                return RagResult(corpus=corpus, scores=scores, metadata={'query': query.query_str, 'retriever': 'graph'})
            for score_node in nodes:
                node = score_node.node
                node.metadata = json.loads(node.metadata.get('metadata', '{}'))
                chunk = Chunk.from_llama_node(node)
                chunk.metadata.similarity_score = score_node.score or 0.0
                corpus.add_chunk(chunk)
                scores.extend([score_node.score or 0.0])
            result = RagResult(corpus=corpus, scores=scores, metadata={'query': query.query_str, 'retriever': 'graph'})
            logger.info(f'Graph retrieved {len(corpus.chunks)} chunks')
            return result
        except Exception as e:
            logger.error(f'Graph retrieval failed: {str(e)}')
            raise

    def retrieve(self, query: Query) -> RagResult:
        try:
            subretriever_bool = [isinstance(sub, VectorContextRetriever) for sub in self.retrieve.sub_retrievers]
            if any(subretriever_bool):
                ind = subretriever_bool.index(True)
                self.retriever[ind].similarity_top_k = query.top_k
            nodes = self.retriever.retrieve(query.query_str)
            corpus = Corpus()
            scores = []
            if nodes is None:
                return RagResult(corpus=corpus, scores=scores, metadata={'query': query.query_str, 'retriever': 'graph'})
            for score_node in nodes:
                node = score_node.node
                flattened_metadata = {}
                for key, value in node.metadata.items():
                    flattened_metadata[key] = json.loads(value)
                node.metadata = flattened_metadata
                chunk = Chunk.from_llama_node(score_node.node)
                chunk.metadata.similarity_score = score_node.score or 0.0
                corpus.add_chunk(chunk)
                scores.extend([score_node.score or 0.0])
            result = RagResult(corpus=corpus, scores=scores, metadata={'query': query.query_str, 'retriever': 'graph'})
            logger.info(f'Vector retrieved {len(corpus.chunks)} chunks')
            return result
        except Exception as e:
            logger.error(f'Vector retrieval failed: {str(e)}')
            raise

    def get_retriever(self) -> PGRetriever:
        logger.debug('Returning graph retriever')
        return self.retriever

class VectorRetriever(BaseRetrieverWrapper):
    """Wrapper for vector-based retrieval."""

    def __init__(self, index: BaseIndex, top_k: int=5, chunk_class=None):
        super().__init__()
        self.index = index
        self.top_k = top_k
        self.chunk_class = chunk_class
        self.retriever = VectorIndexRetriever(index=self.index, similarity_top_k=self.top_k)

    async def aretrieve(self, query: Query) -> RagResult:
        try:
            self.retriever.similarity_top_k = query.top_k
            nodes = await self.retriever.aretrieve(query.query_str)
            corpus = Corpus()
            scores = []
            if nodes is None:
                return RagResult(corpus=corpus, scores=scores, metadata={'query': query.query_str, 'retriever': 'vector'})
            for score_node in nodes:
                if self.chunk_class is None:
                    raise ValueError('chunk_class not set - RAGEngine must pass chunk class based on config')
                chunk = self.chunk_class.from_llama_node(score_node.node)
                chunk.metadata.similarity_score = score_node.score or 0.0
                corpus.add_chunk(chunk)
                scores.extend([score_node.score or 0.0])
            result = RagResult(corpus=corpus, scores=scores, metadata={'query': query.query_str, 'retriever': 'vector'})
            logger.info(f'Vector retrieved {len(corpus.chunks)} chunks')
            return result
        except Exception as e:
            logger.error(f'Vector retrieval failed: {str(e)}')
            raise

    def retrieve(self, query: Query) -> RagResult:
        try:
            self.retriever.similarity_top_k = query.top_k
            nodes = self.retriever.retrieve(query.query_str)
            corpus = Corpus()
            scores = []
            if nodes is None:
                return RagResult(corpus=corpus, scores=scores, metadata={'query': query.query_str, 'retriever': 'vector'})
            for score_node in nodes:
                if self.chunk_class is None:
                    raise ValueError('chunk_class not set - RAGEngine must pass chunk class based on config')
                chunk = self.chunk_class.from_llama_node(score_node.node)
                chunk.metadata.similarity_score = score_node.score or 0.0
                corpus.add_chunk(chunk)
                scores.extend([score_node.score or 0.0])
            result = RagResult(corpus=corpus, scores=scores, metadata={'query': query.query_str, 'retriever': 'vector'})
            logger.info(f'Vector retrieved {len(corpus.chunks)} chunks')
            return result
        except Exception as e:
            logger.error(f'Vector retrieval failed: {str(e)}')
            raise

    def get_retriever(self) -> VectorIndexRetriever:
        logger.debug('Returning vector retriever')
        return self.retriever

class SemanticChunker(BaseChunker):
    """Chunker that splits documents based on semantic similarity.

    Uses LlamaIndex's SemanticChunker with an embedding model to create chunks that preserve
    semantic coherence, ideal for improving retrieval accuracy in RAG pipelines.

    Attributes:
        embed_model (BaseEmbedding): The embedding model for semantic similarity.
        parser (SemanticChunker): The LlamaIndex parser for semantic chunking.
    """

    def __init__(self, embed_model: BaseEmbedding, similarity_threshold: float=0.7, max_workers=4, **kwargs):
        """Initialize the SemanticChunker.

        Args:
            embed_model_name (BaseEmbedding): the embedding model.
            similarity_threshold (float, optional): Threshold for semantic similarity to split chunks (default: 0.7).
        """
        self.embed_model = embed_model
        self.parser = SemanticSplitterNodeParser(embed_model=self.embed_model, similarity_threshold=similarity_threshold)
        self.max_workers = max_workers

    def _process_document(self, doc: Document) -> List[Chunk]:
        """Process a single document into chunks.

        Args:
            doc (Document): The document to chunk.

        Returns:
            List[Chunk]: List of Chunk objects with metadata.
        """
        try:
            llama_doc = doc.to_llama_document()
            llama_doc.metadata['doc_id'] = doc.doc_id
            nodes = asyncio.run(self.parser.aget_nodes_from_documents([llama_doc]))
            chunks = []
            for idx, node in enumerate(nodes):
                chunk = Chunk.from_llama_node(node)
                chunk.metadata.chunking_strategy = ChunkingStrategy.SIMPLE
                chunks.extend([chunk])
            logger.debug(f'Processed document {doc.doc_id} into {len(chunks)} chunks')
            return chunks
        except Exception as e:
            logger.error(f'Failed to process document {doc.doc_id}: {str(e)}')
            return []

    def chunk(self, documents: List[Document], **kwargs) -> Corpus:
        """Chunk documents based on semantic similarity.

        Args:
            documents (List[Document]): List of Document objects to chunk.
            **kwargs: Additional parameters (e.g., max_chunk_size).

        Returns:
            Corpus: A collection of Chunk objects with semantic metadata.
        """
        if not documents:
            logger.info('No documents provided, returning empty Corpus')
            return Corpus([])
        chunks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_doc = {executor.submit(self._process_document, doc): doc for doc in documents}
            for future in future_to_doc:
                doc = future_to_doc[future]
                try:
                    chunks.extend(future.result())
                except Exception as e:
                    logger.error(f'Error processing document {doc.doc_id}: {str(e)}')
        logger.info(f'Chunked {len(documents)} documents into {len(chunks)} chunks')
        return Corpus(chunks=chunks)

class SimpleChunker(BaseChunker):
    """Chunker that splits documents into fixed-size chunks using multi-threading and async parsing.

    Uses LlamaIndex's SimpleNodeParser with async support to create chunks with a specified size
    and overlap, suitable for general-purpose text splitting in RAG pipelines.

    Attributes:
        chunk_size (int): The target size of each chunk in characters.
        chunk_overlap (int): The number of overlapping characters between adjacent chunks.
        parser (SimpleNodeParser): The LlamaIndex parser for chunking.
        max_workers (int): Maximum number of threads for parallel processing.
    """

    def __init__(self, chunk_size: int=1024, chunk_overlap: int=20, tokenizer=None, chunking_tokenizer_fn=None, include_metadata: bool=True, include_prev_next_rel: bool=True, max_workers: int=4):
        """Initialize the SimpleChunker.

        Args:
            chunk_size (int, optional): Target size of each chunk in characters (default: 1024).
            chunk_overlap (int, optional): Overlap between adjacent chunks in characters (default: 20).
            tokenizer: Optional tokenizer for chunking.
            chunking_tokenizer_fn: Optional tokenizer function for chunking.
            include_metadata (bool): Whether to include metadata in nodes (default: True).
            include_prev_next_rel (bool): Whether to include previous/next relationships (default: True).
            max_workers (int): Maximum number of threads for parallel processing (default: 4).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer
        self.chunking_tokenizer_fn = chunking_tokenizer_fn
        self.max_workers = max_workers
        self.parser = SimpleNodeParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=tokenizer, chunking_tokenizer_fn=chunking_tokenizer_fn, include_metadata=include_metadata, include_prev_next_rel=include_prev_next_rel)

    def _process_document(self, doc: Document) -> List[Chunk]:
        """Process a single document into chunks in a thread.

        Args:
            doc (Document): The document to chunk.

        Returns:
            List[Chunk]: List of Chunk objects with metadata.
        """
        try:
            llama_doc = doc.to_llama_document()
            llama_doc.metadata['doc_id'] = doc.doc_id
            nodes = asyncio.run(self.parser.aget_nodes_from_documents([llama_doc]))
            chunks = []
            for idx, node in enumerate(nodes):
                chunk = Chunk.from_llama_node(node)
                chunk.metadata.chunking_strategy = ChunkingStrategy.SIMPLE
                chunks.extend([chunk])
            logger.debug(f'Processed document {doc.doc_id} into {len(chunks)} chunks')
            return chunks
        except Exception as e:
            logger.error(f'Failed to process document {doc.doc_id}: {str(e)}')
            return []

    def chunk(self, documents: List[Document], **kwargs) -> Corpus:
        """Chunk documents into fixed-size chunks using multi-threading.

        Args:
            documents (List[Document]): List of Document objects to chunk.

        Returns:
            Corpus: A collection of Chunk objects with metadata.
        """
        if not documents:
            logger.info('No documents provided, returning empty Corpus')
            return Corpus([])
        chunks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_doc = {executor.submit(self._process_document, doc): doc for doc in documents}
            for future in future_to_doc:
                doc = future_to_doc[future]
                try:
                    chunks.extend(future.result())
                except Exception as e:
                    logger.error(f'Error processing document {doc.doc_id}: {str(e)}')
        logger.info(f'Chunked {len(documents)} documents into {len(chunks)} chunks')
        return Corpus(chunks=chunks)

class HierarchicalChunker(BaseChunker):
    """Enhanced hierarchical chunker with dynamic hierarchy level assignment.

    Creates a multi-level hierarchy of chunks with full node relationships:
    - SOURCE: The source document.
    - PREVIOUS/NEXT: Sequential nodes in the document.
    - PARENT/CHILD: Hierarchical relationships.

    Supports custom level parsers or default chunk sizes, with dynamic hierarchy level
    assignment based on node parser IDs. Uses multi-threading and async parsing.

    Attributes:
        level_parsers (Dict[str, BaseChunker]): Custom parsers for each hierarchy level.
        chunk_sizes (List[int]): Chunk sizes for default parsers (e.g., [2048, 512, 128]).
        chunk_overlap (int): Overlap between adjacent chunks.
        parser (HierarchicalNodeParser): LlamaIndex parser for hierarchical chunking.
        include_metadata (bool): Whether to include metadata in nodes.
        include_prev_next_rel (bool): Whether to include previous/next node relationships.
        max_workers (int): Maximum number of threads for parallel processing.
        parser_to_level (Dict[str, int]): Mapping of node_parser_id to hierarchy level.
    """

    def __init__(self, level_parsers: Dict[str, BaseChunker]=None, chunk_sizes: Optional[List[int]]=None, chunk_overlap: int=20, include_metadata: bool=True, include_prev_next_rel: bool=True, max_workers: int=4):
        """Initialize the HierarchicalChunker.

        Args:
            level_parsers (Dict[str, BaseChunker], optional): Custom parsers for hierarchy levels.
            chunk_sizes (List[int], optional): Chunk sizes for default parsers (default: [2048, 512, 128]).
            chunk_overlap (int): Overlap between adjacent chunks (default: 20).
            include_metadata (bool): Include metadata in nodes (default: True).
            include_prev_next_rel (bool): Include prev/next relationships (default: True).
            max_workers (int): Maximum number of threads for parallel processing (default: 4).
        """
        self.level_parsers = level_parsers or {}
        self.chunk_sizes = chunk_sizes or [2048, 512, 128]
        self.chunk_overlap = chunk_overlap
        self.include_metadata = include_metadata
        self.include_prev_next_rel = include_prev_next_rel
        self.max_workers = max_workers
        node_parser_ids = None
        node_parser_map = None
        if not self.level_parsers:
            node_parser_ids = [f'chunk_size_{size}' for size in self.chunk_sizes]
            node_parser_map = {node_id: SimpleChunker(chunk_size=size, chunk_overlap=chunk_overlap, include_metadata=include_metadata, include_prev_next_rel=include_prev_next_rel).parser for size, node_id in zip(self.chunk_sizes, node_parser_ids)}
        else:
            if chunk_sizes is not None:
                raise ValueError('If level_parsers is provided, chunk_sizes should be None.')
            node_parser_ids = list(self.level_parsers.keys())
            node_parser_map = {k: v.parser for k, v in self.level_parsers.items()}
        self.parser_to_level = {pid: idx + 1 for idx, pid in enumerate(node_parser_ids)}
        self.parser = HierarchicalNodeParser.from_defaults(chunk_sizes=None, chunk_overlap=self.chunk_overlap, node_parser_ids=node_parser_ids, node_parser_map=node_parser_map, include_metadata=include_metadata, include_prev_next_rel=include_prev_next_rel)

    def _process_document(self, doc: Document, custom_metadata: Dict=None) -> List[Chunk]:
        """Process a single document into chunks in a thread.

        Args:
            doc (Document): The document to chunk.
            custom_metadata (Dict, optional): User-defined metadata for sections.

        Returns:
            List[Chunk]: List of Chunk objects with metadata.
        """
        try:
            llama_doc = doc.to_llama_document()
            llama_doc.metadata['doc_id'] = doc.doc_id
            nodes = self.parser.get_nodes_from_documents([llama_doc])
            chunks = []
            for i, node in enumerate(nodes):
                chunk = Chunk.from_llama_node(node)
                chunk.metadata.chunking_strategy = ChunkingStrategy.HIERARCHICAL
                chunks.extend([chunk])
            logger.debug(f'Processed document {doc.doc_id} into {len(chunks)} chunks')
            return chunks
        except Exception as e:
            logger.error(f'Failed to process document {doc.doc_id}: {str(e)}')
            return []

    def chunk(self, documents: List[Document], **kwargs) -> Corpus:
        """Chunk documents using hierarchical strategy with dynamic chunk size adjustment.

        Args:
            documents (List[Document]): List of Document objects to chunk.
            **kwargs: Additional parameters, e.g., custom_metadata for section titles.

        Returns:
            Corpus: A collection of hierarchically organized chunks.
        """
        if not documents:
            logger.info('No documents provided, returning empty Corpus')
            return Corpus(chunks=[])
        chunks = []
        custom_metadata = kwargs.get('custom_metadata', {})
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_doc = {executor.submit(self._process_document, doc, custom_metadata): doc for doc in documents}
            for future in future_to_doc:
                doc = future_to_doc[future]
                try:
                    chunks.extend(future.result())
                except Exception as e:
                    logger.error(f'Error processing document {doc.doc_id}: {str(e)}')
        logger.info(f'Chunked {len(documents)} documents into {len(chunks)} chunks')
        return Corpus(chunks=chunks)

class SimpleReranker(BasePostprocessor):
    """Post-processor for reranking retrieval results."""

    def __init__(self, similarity_cutoff: Optional[float]=None, keyword_filters: Optional[List[str]]=None):
        super().__init__()
        self.postprocessors = []
        if similarity_cutoff:
            self.postprocessors.append(SimilarityPostprocessor(similarity_cutoff=similarity_cutoff))
        if keyword_filters:
            self.postprocessors.append(KeywordNodePostprocessor(required_keywords=keyword_filters))

    def postprocess(self, query: Query, results: List[RagResult]) -> RagResult:
        try:
            if not self.postprocessors:
                corpus = Corpus()
                scores = []
                for result in results:
                    for chunk in result.corpus.chunks:
                        corpus.add_chunk(chunk)
                    scores.extend(result.scores)
                final_result = RagResult(corpus=corpus, scores=scores, metadata={'query': query.query_str, 'postprocessor': 'simple_passthrough'})
                self.logger.info(f'Simple passthrough: {len(corpus.chunks)} chunks')
                return final_result
            chunk_to_original = {}
            nodes = []
            for result in results:
                for chunk, score in zip(result.corpus.chunks, result.scores):
                    node = chunk.to_llama_node()
                    nodes.append(NodeWithScore(node=node, score=score))
                    chunk_to_original[node.id_] = chunk
            for postprocessor in self.postprocessors:
                nodes = postprocessor.postprocess_nodes(nodes)
            corpus = Corpus()
            scores = []
            for score_node in nodes:
                original_chunk = chunk_to_original.get(score_node.node.id_)
                if original_chunk:
                    original_chunk.metadata.similarity_score = score_node.score or 0.0
                    corpus.add_chunk(original_chunk)
                    scores.append(score_node.score or 0.0)
                else:
                    chunk_class = type(results[0].corpus.chunks[0]) if results and results[0].corpus.chunks else Chunk
                    try:
                        chunk = chunk_class.from_llama_node(score_node.node)
                        chunk.metadata.similarity_score = score_node.score or 0.0
                        corpus.add_chunk(chunk)
                        scores.append(score_node.score or 0.0)
                    except Exception as e:
                        self.logger.warning(f'Failed to reconstruct chunk from node: {e}')
                        continue
            result = RagResult(corpus=corpus, scores=scores, metadata={'query': query.query_str, 'postprocessor': 'reranker'})
            self.logger.info(f'Reranked to {len(corpus.chunks)} chunks')
            return result
        except Exception as e:
            self.logger.error(f'Reranking failed: {str(e)}')
            raise

class LongTermMemory(BaseMemory):
    """
    Manages long-term storage and retrieval of memories, integrating with RAGEngine for indexing
    and StorageHandler for persistence.
    """
    storage_handler: StorageHandler = Field(..., description='Handler for persistent storage')
    rag_config: RAGConfig = Field(..., description='Configuration for RAG engine')
    rag_engine: RAGEngine = Field(default=None, description='RAG engine for indexing and retrieval')
    memory_table: str = Field(default='memory', description='Database table for storing memories')
    default_corpus_id: Optional[str] = Field(default=None, description='Default corpus ID for memory indexing')

    def init_module(self):
        """Initialize the RAG engine and memory indices."""
        super().init_module()
        if self.rag_engine is None:
            self.rag_engine = RAGEngine(config=self.rag_config, storage_handler=self.storage_handler)
        if self.default_corpus_id is None:
            self.default_corpus_id = str(uuid4())
        logger.info(f'Initialized LongTermMemory with corpus_id {self.default_corpus_id}')

    def _create_memory_chunk(self, message: Message, memory_id: str) -> Chunk:
        """Convert a Message to a Chunk for RAG indexing."""
        metadata = ChunkMetadata(corpus_id=self.default_corpus_id, memory_id=memory_id, timestamp=message.timestamp, action=message.action, wf_goal=message.wf_goal, agent=message.agent, msg_type=message.msg_type.value if message.msg_type else None, prompt=message.prompt, next_actions=message.next_actions, wf_task=message.wf_task, wf_task_desc=message.wf_task_desc, message_id=message.message_id, content=json.dumps(message.content))
        return Chunk(chunk_id=memory_id, text=str(message.content), metadata=metadata, start_char_idx=0, end_char_idx=len(str(message.content)))

    def _chunk_to_message(self, chunk: Chunk) -> Message:
        """Convert a Chunk to a Message object."""
        return Message(content=chunk.metadata.content, action=chunk.metadata.action, wf_goal=chunk.metadata.wf_goal, timestamp=chunk.metadata.timestamp, agent=chunk.metadata.agent, msg_type=chunk.metadata.msg_type, prompt=chunk.metadata.prompt, next_actions=chunk.metadata.next_actions, wf_task=chunk.metadata.wf_task, wf_task_desc=chunk.metadata.wf_task_desc, message_id=chunk.metadata.message_id)

    def add(self, messages: Union[Message, str, List[Union[Message, str]]]) -> List[str]:
        """Store messages in memory and index them in RAGEngine, returning memory_ids."""
        if not isinstance(messages, list):
            messages = [messages]
        messages = [Message(content=msg) if isinstance(msg, str) else msg for msg in messages]
        messages = [msg for msg in messages if msg.content]
        if not messages:
            logger.warning('No valid messages to add')
            return []
        existing_hashes = {record['content_hash'] for record in self.storage_handler.load(tables=[self.memory_table]).get(self.memory_table, []) if 'content_hash' in record}
        memory_ids = [str(uuid4()) for _ in messages]
        final_messages = []
        final_memory_ids = []
        final_chunks = []
        for msg, memory_id in zip(messages, memory_ids):
            content_hash = hashlib.sha256(str(msg.content).encode()).hexdigest()
            if content_hash in existing_hashes:
                logger.info(f'Duplicate message found (hash): {msg.content[:50]}...')
                existing_id = next((r['memory_id'] for r in self.storage_handler.load(tables=[self.memory_table]).get(self.memory_table, []) if r.get('content_hash') == content_hash), None)
                if existing_id:
                    final_memory_ids.append(existing_id)
                    continue
            final_messages.append(msg)
            final_memory_ids.append(memory_id)
            chunk = self._create_memory_chunk(msg, memory_id)
            chunk.metadata.content_hash = content_hash
            final_chunks.append(chunk)
        if not final_chunks:
            logger.info('No messages added after deduplication')
            return final_memory_ids
        for msg in final_messages:
            super().add_message(msg)
        corpus = Corpus(chunks=final_chunks, corpus_id=self.default_corpus_id)
        chunk_ids = self.rag_engine.add(index_type=self.rag_config.index.index_type, nodes=corpus, corpus_id=self.default_corpus_id)
        if not chunk_ids:
            logger.error('Failed to index memories')
            return final_memory_ids
        return final_memory_ids

    async def get(self, memory_ids: Union[str, List[str]], return_chunk: bool=True) -> List[Tuple[Union[Chunk, Message], str]]:
        """Retrieve memories by memory_ids, returning (Message/Chunk, memory_id) tuples."""
        if not isinstance(memory_ids, list):
            memory_ids = [memory_ids]
        if not memory_ids:
            logger.warning('No memory_ids provided for get')
            return []
        try:
            chunks = await self.rag_engine.aget(corpus_id=self.default_corpus_id, index_type=self.rag_config.index.index_type, node_ids=memory_ids)
            results = [(self._chunk_to_message(chunk), chunk.metadata.memory_id) if not return_chunk else (chunk, chunk.metadata.memory_id) for chunk in chunks if chunk]
            logger.info(f'Retrieved {len(results)} memories for memory_ids: {memory_ids}')
            return results
        except Exception as e:
            logger.error(f'Failed to get memories: {str(e)}')
            return []

    def delete(self, memory_ids: Union[str, List[str]]) -> List[bool]:
        """Delete memories by memory_ids, returning success status for each."""
        if not isinstance(memory_ids, list):
            memory_ids = [memory_ids]
        if not memory_ids:
            logger.warning('No memory_ids provided for deletion')
            return []
        successes = [False] * len(memory_ids)
        valid_memory_ids = []
        existing_chunks = asyncio.run(self.get(memory_ids, return_chunk=True))
        for idx, (chunk, mid) in enumerate(existing_chunks):
            if chunk:
                valid_memory_ids.append(mid)
                super().remove_message(self._chunk_to_message(chunk))
                successes[idx] = True
        if not valid_memory_ids:
            logger.info('No memories found for deletion')
            return successes
        self.rag_engine.delete(corpus_id=self.default_corpus_id, index_type=self.rag_config.index.index_type, node_ids=valid_memory_ids)
        return successes

    def update(self, updates: Union[Tuple[str, Union[Message, str]], List[Tuple[str, Union[Message, str]]]]) -> List[bool]:
        """Update memories with new content, returning success status for each."""
        if not isinstance(updates, list):
            updates = [updates]
        updates = [(mid, Message(content=msg) if isinstance(msg, str) else msg) for mid, msg in updates]
        updates_dict = {mid: msg for mid, msg in updates if msg.content}
        if not updates_dict:
            logger.warning('No valid updates provided')
            return []
        memory_ids = list(updates_dict.keys())
        existing_memories = asyncio.run(self.get(memory_ids, return_chunk=False))
        existing_dict = {mid: msg for msg, mid in existing_memories}
        successes = [False] * len(updates)
        final_updates = []
        final_memory_ids = []
        for mid, msg in updates_dict.items():
            if mid not in existing_dict:
                logger.warning(f'No memory found with memory_id {mid}')
                continue
            final_updates.append((mid, msg))
            final_memory_ids.append(mid)
            successes[memory_ids.index(mid)] = True
            super().remove_message(existing_dict[mid])
        if not final_updates:
            logger.info('No memories updated')
            return successes
        chunks = [self._create_memory_chunk(msg, mid) for mid, msg in final_updates]
        for msg in [msg for _, msg in final_updates]:
            super().add_message(msg)
        corpus = Corpus(chunks=chunks, corpus_id=self.default_corpus_id)
        chunk_ids = self.rag_engine.add(index_type=self.rag_config.index.index_type, nodes=corpus, corpus_id=self.default_corpus_id)
        if not chunk_ids:
            logger.error(f'Failed to update memories in RAG index: {final_memory_ids}')
            return [False] * len(updates)
        return successes

    async def search_async(self, query: Union[str, Query], n: Optional[int]=None, metadata_filters: Optional[Dict]=None, return_chunk=False) -> List[Tuple[Message, str]]:
        """Retrieve messages from RAG index asynchronously based on a query, returning messages and memory_ids."""
        if isinstance(query, str):
            query_obj = Query(query_str=query, top_k=n or self.rag_config.retrieval.top_k, metadata_filters=metadata_filters or {})
        else:
            query_obj = query
            query_obj.top_k = n or self.rag_config.retrieval.top_k
            if metadata_filters:
                query_obj.metadata_filters = {**query_obj.metadata_filters, **metadata_filters} if query_obj.metadata_filters else metadata_filters
        try:
            result: RagResult = await self.rag_engine.query_async(query_obj, corpus_id=self.default_corpus_id)
            if return_chunk:
                return [(chunk, chunk.metadata.memory_id) for chunk in result.corpus.chunks]
            else:
                messages = [(self._chunk_to_message(chunk), chunk.metadata.memory_id) for chunk in result.corpus.chunks]
            logger.info(f'Retrieved {len(messages)} memories for query: {query_obj.query_str}')
            return messages[:n] if n else messages
        except Exception as e:
            logger.error(f'Failed to search memories: {str(e)}')
            return []

    def search(self, query: Union[str, Query], n: Optional[int]=None, metadata_filters: Optional[Dict]=None) -> List[Tuple[Message, str]]:
        """Synchronous wrapper for searching memories."""
        return asyncio.run(self.search_async(query, n, metadata_filters))

    def clear(self) -> None:
        """Clear all messages and indices."""
        super().clear()
        self.rag_engine.clear(corpus_id=self.default_corpus_id)
        logger.info(f'Cleared LongTermMemory with corpus_id {self.default_corpus_id}')

    def save(self, save_path: Optional[str]=None) -> None:
        """Save all indices and memory data to database."""
        self.rag_engine.save(output_path=save_path, corpus_id=self.default_corpus_id, table=self.memory_table)

    def load(self, save_path: Optional[str]=None) -> List[str]:
        """Load memory data from database and reconstruct indices, returning memory_ids."""
        return self.rag_engine.load(source=save_path, corpus_id=self.default_corpus_id, table=self.memory_table)

def create_corpus_from_context(context: List[List], corpus_id: str) -> Corpus:
    """Convert HotPotQA context into a Corpus for indexing."""
    chunks = []
    for title, sentences in context:
        for idx, sentence in enumerate(sentences):
            chunk = Chunk(chunk_id=f'{title}_{idx}', text=sentence, metadata=ChunkMetadata(doc_id=str(idx), corpus_id=corpus_id), start_char_idx=0, end_char_idx=len(sentence), excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={})
            chunk.metadata.title = title
            chunks.append(chunk)
    return Corpus(chunks=chunks[:4], corpus_id=corpus_id)

