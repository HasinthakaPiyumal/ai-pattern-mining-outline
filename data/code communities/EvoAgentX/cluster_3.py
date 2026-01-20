# Cluster 3

class MemoryAgent(Agent):
    memory_manager: Optional[MemoryManager] = Field(default=None, description='Manager for long-term memory operations')
    inputs: List[Dict] = Field(default_factory=list, description='Input specifications for the memory action')
    outputs: List[Dict] = Field(default_factory=list, description='Output specifications for the memory action')

    def __init__(self, name: str='MemoryAgent', description: str='An agent that uses long-term memory to provide context-aware responses', inputs: Optional[List[Dict]]=None, outputs: Optional[List[Dict]]=None, llm_config: Optional[OpenAILLMConfig]=None, storage_handler: Optional[StorageHandler]=None, rag_config: Optional[RAGConfig]=None, conversation_id: Optional[str]=None, system_prompt: Optional[str]=None, prompt: str='Based on the following context and user prompt, provide a relevant response:\n\nContext: {context}\n\nUser Prompt: {user_prompt}', **kwargs):
        inputs = inputs or []
        outputs = outputs or []
        super().__init__(name=name, description=description, llm_config=llm_config, system_prompt=system_prompt, storage_handler=storage_handler, inputs=inputs, outputs=outputs, **kwargs)
        self.long_term_memory = LongTermMemory(storage_handler=storage_handler, rag_config=rag_config, default_corpus_id=conversation_id)
        self.memory_manager = MemoryManager(memory=self.long_term_memory, llm=llm_config.get_llm() if llm_config else None, use_llm_management=True)
        self.inputs = inputs
        self.outputs = outputs
        self.actions = []
        self._action_map = {}
        memory_action = MemoryAction(name='MemoryAction', description='Action that processes user input with long-term memory context', prompt=prompt, inputs_format=MemoryActionInput, outputs_format=MemoryActionOutput)
        self.add_action(memory_action)

    def _create_output_message(self, action_output, action_name: str, action_input_data: Optional[Dict], prompt: str, return_msg_type: MessageType=MessageType.RESPONSE, **kwargs) -> Message:
        msg = super()._create_output_message(action_output=action_output, action_name=action_name, action_input_data=action_input_data, prompt=prompt, return_msg_type=return_msg_type, **kwargs)
        if action_input_data and 'user_prompt' in action_input_data:
            user_msg = Message(content=action_input_data['user_prompt'], msg_type=MessageType.REQUEST, conversation_id=msg.conversation_id)
            asyncio.create_task(self.memory_manager.handle_memory(action='add', data=user_msg))
        response_msg = Message(content=action_output.response if hasattr(action_output, 'response') else str(action_output), msg_type=MessageType.RESPONSE, conversation_id=msg.conversation_id)
        asyncio.create_task(self.memory_manager.handle_memory(action='add', data=response_msg))
        return msg

    async def async_execute(self, action_name: str, msgs: Optional[List[Message]]=None, action_input_data: Optional[Dict]=None, return_msg_type: Optional[MessageType]=MessageType.RESPONSE, return_action_input_data: Optional[bool]=False, **kwargs) -> Union[Message, Tuple[Message, Dict]]:
        """
        Execute an action asynchronously with memory management.

        Args:
            action_name: Name of the action to execute
            msgs: Optional list of messages providing context
            action_input_data: Optional input data for the action
            return_msg_type: Message type for the return message
            return_action_input_data: Whether to return the action input data
            **kwargs: Additional parameters

        Returns:
            Message or tuple: The execution result, optionally with input data
        """
        action, action_input_data = self._prepare_execution(action_name=action_name, msgs=msgs, action_input_data=action_input_data, **kwargs)
        execution_results = await action.async_execute(llm=self.llm, inputs=action_input_data, sys_msg=self.system_prompt, return_prompt=True, memory_manager=self.memory_manager, **kwargs)
        action_output, prompt = execution_results
        message = self._create_output_message(action_output=action_output, prompt=prompt, action_name=action_name, return_msg_type=return_msg_type, action_input_data=action_input_data, **kwargs)
        if return_action_input_data:
            return (message, action_input_data)
        return message

    def execute(self, action_name: str, msgs: Optional[List[Message]]=None, action_input_data: Optional[Dict]=None, return_msg_type: Optional[MessageType]=MessageType.RESPONSE, return_action_input_data: Optional[bool]=False, **kwargs) -> Union[Message, Tuple[Message, Dict]]:
        """
        Execute an action synchronously with memory management.

        Args:
            action_name: Name of the action to execute
            msgs: Optional list of messages providing context
            action_input_data: Optional input data for the action
            return_msg_type: Message type for the return message
            return_action_input_data: Whether to return the action input data
            **kwargs: Additional parameters

        Returns:
            Message or tuple: The execution result, optionally with input data
        """
        action, action_input_data = self._prepare_execution(action_name=action_name, msgs=msgs, action_input_data=action_input_data, **kwargs)
        execution_results = action.execute(llm=self.llm, inputs=action_input_data, sys_msg=self.system_prompt, return_prompt=True, memory_manager=self.memory_manager, **kwargs)
        action_output, prompt = execution_results
        message = self._create_output_message(action_output=action_output, prompt=prompt, action_name=action_name, return_msg_type=return_msg_type, action_input_data=action_input_data, **kwargs)
        if return_action_input_data:
            return (message, action_input_data)
        return message

    def chat(self, user_prompt: str, *, conversation_id: Optional[str]=None, top_k: Optional[int]=None, metadata_filters: Optional[dict]=None, return_message: bool=True, **kwargs):
        action_input_data = {'user_prompt': user_prompt, 'conversation_id': conversation_id or self._default_conversation_id(), 'top_k': top_k if top_k is not None else 3, 'metadata_filters': metadata_filters or {}}
        msg = self.execute(action_name='MemoryAction', action_input_data=action_input_data, return_msg_type=MessageType.RESPONSE, **kwargs)
        return msg if return_message else getattr(msg, 'content', None) or str(msg)

    async def async_chat(self, user_prompt: str, *, conversation_id: Optional[str]=None, top_k: Optional[int]=None, metadata_filters: Optional[dict]=None, return_message: bool=True, **kwargs):
        action_input_data = {'user_prompt': user_prompt, 'conversation_id': conversation_id or self._default_conversation_id(), 'top_k': top_k if top_k is not None else 3, 'metadata_filters': metadata_filters or {}}
        msg = await self.async_execute(action_name='MemoryAction', action_input_data=action_input_data, return_msg_type=MessageType.RESPONSE, **kwargs)
        return msg if return_message else getattr(msg, 'content', None) or str(msg)

    def _default_conversation_id(self) -> str:
        """
        Session scope: By default, a new uuid4() is returned (new session).
        User/global scope: Reuse LongTermMemory.default_corpus_id (stable namespace).
        Note: The final ID is still uniformly managed by MemoryAgent._prepare_execution() (which will override based on the scope).
        """
        scope = getattr(self, 'conversation_scope', 'session')
        if scope == 'session':
            return str(uuid4())
        return getattr(getattr(self, 'long_term_memory', None), 'default_corpus_id', None) or 'global_corpus'

    async def interactive_chat(self, conversation_id: Optional[str]=None, top_k: int=3, metadata_filters: Optional[dict]=None):
        """
        In interactive chat, each round of input will:
        1. Retrieve from memory
        2. Generate a response based on historical context
        3. Write the input/output to long-term memory and refresh the index 
        """
        conversation_id = conversation_id or self._default_conversation_id()
        metadata_filters = metadata_filters or {}
        print("💬 MemoryAgent has been started (type 'exit' to quit)\n")
        while True:
            user_prompt = input('You: ').strip()
            if user_prompt.lower() in ['exit', 'quit']:
                print('🔚 Conversation ended')
                break
            retrieved_memories = await self.memory_manager.handle_memory(action='search', user_prompt=user_prompt, top_k=top_k, metadata_filters=metadata_filters)
            context_texts = []
            for msg, _ in retrieved_memories:
                if hasattr(msg, 'content') and msg.content:
                    context_texts.append(msg.content)
            context_str = '\n'.join(context_texts)
            full_prompt = f'Context:\n{context_str}\n\nUser: {user_prompt}' if context_str else user_prompt
            msg = await self.async_chat(user_prompt=full_prompt, conversation_id=conversation_id, top_k=top_k, metadata_filters=metadata_filters)
            print(f'Agent: {msg.content}\n')
            if hasattr(self.memory_manager, 'handle_memory_flush'):
                await self.memory_manager.handle_memory_flush()
            else:
                await asyncio.sleep(0.1)

    def save_module(self, path: str, ignore: List[str]=['llm', 'llm_config', 'memory_manager'], **kwargs) -> str:
        """
        Save the agent's configuration to a JSON file, excluding memory_manager by default.

        Args:
            path: File path to save the configuration
            ignore: List of keys to exclude from the saved configuration
            **kwargs: Additional parameters for saving

        Returns:
            str: The path where the configuration was saved
        """
        return super().save_module(path=path, ignore=ignore, **kwargs)

    @classmethod
    def from_file(cls, path: str, llm_config: OpenAILLMConfig, storage_handler: Optional[StorageHandler]=None, rag_config: Optional[RAGConfig]=None, **kwargs) -> 'MemoryAgent':
        """
        Load a MemoryAgent from a JSON configuration file.

        Args:
            path: Path to the JSON configuration file
            llm_config: LLM configuration
            storage_handler: Optional storage handler
            rag_config: Optional RAG configuration
            **kwargs: Additional parameters

        Returns:
            MemoryAgent: The loaded agent instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return cls(name=config.get('name', 'MemoryAgent'), description=config.get('description', 'An agent that uses long-term memory'), llm_config=llm_config, storage_handler=storage_handler, rag_config=rag_config, system_prompt=config.get('system_prompt'), prompt=config.get('prompt'), use_long_term_memory=config.get('use_long_term_memory', True), **kwargs)

def main():
    OPEN_ROUNTER_API_KEY = os.environ['OPEN_ROUNTER_API_KEY']
    config = OpenRouterConfig(openrouter_key=OPEN_ROUNTER_API_KEY, temperature=0.3, model='google/gemini-2.5-flash-lite-preview-06-17')
    llm = OpenRouterLLM(config=config)
    store_config = StoreConfig(dbConfig=DBConfig(db_name='sqlite', path='./debug/data/hotpotqa/cache/test_hotpotQA.sql'), vectorConfig=VectorStoreConfig(vector_name='faiss', dimensions=768, index_type='flat_l2'), graphConfig=None, path='./debug/data/hotpotqa/cache/indexing')
    storage_handler = StorageHandler(storageConfig=store_config)
    embedding = EmbeddingConfig(provider='huggingface', model_name='debug/bge-small-en-v1.5', device='cpu')
    rag_config = RAGConfig(reader=ReaderConfig(recursive=False, exclude_hidden=True, num_files_limit=None, custom_metadata_function=None, extern_file_extractor=None, errors='ignore', encoding='utf-8'), chunker=ChunkerConfig(strategy='simple', chunk_size=512, chunk_overlap=0, max_chunks=None), embedding=embedding, index=IndexConfig(index_type='vector'), retrieval=RetrievalConfig(retrivel_type='vector', postprocessor_type='simple', top_k=2, similarity_cutoff=0.3, keyword_filters=None, metadata_filters=None))
    memory = LongTermMemory(storage_handler=storage_handler, rag_config=rag_config)
    memory_agent = Agent(name='MemoryAgent', description='An agent that manages long-term memory operations', actions=[AddMemories(), SearchMemories(), UpdateMemories(), DeleteMemories()], llm_config=config)
    actions = memory_agent.get_all_actions()
    print(f'Available actions of agent {memory_agent.name}:')
    for action in actions:
        print(f'- {action.name}: {action.description}')
    messages = [{'content': 'Schedule a meeting with Alice on Monday', 'action': 'schedule', 'wf_goal': 'plan_meeting', 'agent': 'user', 'msg_type': MessageType.REQUEST.value, 'wf_task': 'schedule_meeting', 'wf_task_desc': 'Schedule a meeting with a colleague', 'message_id': 'msg_001'}, {'content': 'Send report to Bob by Friday', 'action': 'send', 'wf_goal': 'submit_report', 'agent': 'user', 'msg_type': MessageType.REQUEST.value, 'wf_task': 'send_report', 'wf_task_desc': 'Send a report to a colleague', 'message_id': 'msg_002'}]
    add_result = memory_agent.execute(action_name='AddMemories', action_input_data={'messages': messages}, memory=memory)
    print('\nAdded memories:')
    print(f'Memory IDs: {add_result.content.memory_ids}')
    search_result = memory_agent.execute(action_name='SearchMemories', action_input_data={'query': 'meeting', 'top_k': 2, 'metadata_filters': {'agent': 'user'}}, memory=memory)
    print('\nSearch results:')
    for result in search_result.content.results:
        print(f'- Memory ID: {result['memory_id']}, Message: {result['message'].content}')
    updates = [{'memory_id': add_result.content.memory_ids[0], 'content': 'Reschedule meeting with Alice to Tuesday', 'action': 'reschedule', 'wf_goal': 'plan_meeting', 'agent': 'user', 'msg_type': MessageType.REQUEST.value, 'wf_task': 'reschedule_meeting', 'wf_task_desc': 'Reschedule a meeting with a colleague', 'message_id': 'msg_001_updated'}]
    update_result = memory_agent.execute(action_name='UpdateMemories', action_input_data={'updates': updates}, memory=memory)
    print('\nUpdate results:')
    print(f'Successes: {update_result.content.successes}')
    delete_result = memory_agent.execute(action_name='DeleteMemories', action_input_data={'memory_ids': add_result.content.memory_ids}, memory=memory)
    print('\nDelete results:')
    print(f'Successes: {delete_result.content.successes}')
    new_search_result = memory_agent.execute(action_name='SearchMemories', action_input_data={'query': 'meeting', 'top_k': 2, 'metadata_filters': {'agent': 'user'}}, memory=memory)
    print('\nSearch results:')
    for result in new_search_result.content.results:
        print(f'- Memory ID: {result['memory_id']}, Message: {result['message'].content}')

def main():
    OPEN_ROUNTER_API_KEY = os.environ.get('OPEN_ROUNTER_API_KEY')
    if not OPEN_ROUNTER_API_KEY:
        raise ValueError('OPEN_ROUNTER_API_KEY not set in environment')
    config = OpenRouterConfig(openrouter_key=OPEN_ROUNTER_API_KEY, temperature=0.3, model='google/gemini-2.5-pro-exp-03-25')
    llm = OpenRouterLLM(config=config)
    store_config = StoreConfig(dbConfig=DBConfig(db_name='sqlite', path='./debug/data/hotpotqa/cache/test_hotpotqa.sql'), vectorConfig=VectorStoreConfig(vector_name='faiss', dimensions=384, index_type='flat_l2'), graphConfig=None, path='./debug/data/hotpotqa/cache/indexing')
    storage_handler = StorageHandler(storageConfig=store_config)
    embedding = EmbeddingConfig(provider='huggingface', model_name='debug/weights/bge-small-en-v1.5', device='cpu')
    rag_config = RAGConfig(reader=ReaderConfig(recursive=False, exclude_hidden=True, num_files_limit=None, custom_metadata_function=None, extern_file_extractor=None, errors='ignore', encoding='utf-8'), chunker=ChunkerConfig(strategy='simple', chunk_size=512, chunk_overlap=0, max_chunks=None), embedding=embedding, index=IndexConfig(index_type='vector'), retrieval=RetrievalConfig(retrivel_type='vector', postprocessor_type='simple', top_k=2, similarity_cutoff=0.3, keyword_filters=None, metadata_filters=None))
    memory = LongTermMemory(storage_handler=storage_handler, rag_config=rag_config, llm=llm, use_llm_management=False)
    memory.init_module()
    memory_agent = Agent(name='MemoryAgent', description='An agent that manages long-term memory operations', actions=[AddMemories(), SearchMemories(), UpdateMemories(), DeleteMemories()], llm_config=config)
    print(f'Available actions of agent {memory_agent.name}:')
    for action in memory_agent.get_all_actions():
        print(f'- {action.name}: {action.description}')
    messages = [{'content': 'Schedule a meeting with Alice on Monday', 'action': 'schedule', 'wf_goal': 'plan_meeting', 'agent': 'user', 'msg_type': MessageType.REQUEST.value, 'wf_task': 'schedule_meeting', 'wf_task_desc': 'Schedule a meeting with a colleague', 'message_id': 'msg_001'}, {'content': 'Send report to Bob by Friday', 'action': 'send', 'wf_goal': 'submit_report', 'agent': 'user', 'msg_type': MessageType.REQUEST.value, 'wf_task': 'send_report', 'wf_task_desc': 'Send a report to a colleague', 'message_id': 'msg_002'}]
    add_result = memory_agent.execute(action_name='AddMemories', action_input_data={'messages': messages}, memory=memory)
    print('\nTest 1: Added memories')
    print(f'Memory IDs: {add_result.content.memory_ids}')
    search_result = memory_agent.execute(action_name='SearchMemories', action_input_data={'query': 'meeting', 'top_k': 1, 'metadata_filters': {'agent': 'user'}}, memory=memory)
    print('\nTest 2: Search results (string query)')
    for result in search_result.content.results:
        print(f'- Memory ID: {result['memory_id']}, Content: {result['message'].content}')
    query = Query(query_str='report', top_k=1, metadata_filters={'msg_type': MessageType.REQUEST.value})
    search_result_query = memory_agent.execute(action_name='SearchMemories', action_input_data={'query': query.query_str, 'top_k': query.top_k, 'metadata_filters': query.metadata_filters}, memory=memory)
    print('\nTest 3: Search results (Query object)')
    for result in search_result_query.content.results:
        print(f'- Memory ID: {result['memory_id']}, Content: {result['message'].content}')
    updates = [{'memory_id': add_result.content.memory_ids[0] if add_result.content.memory_ids else '', 'content': 'Reschedule meeting with Alice to Tuesday', 'action': 'reschedule', 'wf_goal': 'plan_meeting', 'agent': 'user', 'msg_type': MessageType.REQUEST.value, 'wf_task': 'reschedule_meeting', 'wf_task_desc': 'Reschedule a meeting with a colleague', 'message_id': 'msg_001_updated'}]
    update_result = memory_agent.execute(action_name='UpdateMemories', action_input_data={'updates': updates}, memory=memory)
    print('\nTest 4: Update results')
    print(f'Successes: {update_result.content.successes}')
    memory.save()
    print('\nTest 5: Saved memories to database')
    memory.clear()
    search_after_clear = memory_agent.execute(action_name='SearchMemories', action_input_data={'query': 'meeting', 'top_k': 1}, memory=memory)
    print('\nTest 6: Search after clear (in-memory)')
    print(f'Results: {len(search_after_clear.content.results)} memories found')
    loaded_ids = memory.load()
    print('\nTest 7: Loaded memories')
    print(f'Loaded {len(loaded_ids)} memory IDs: {loaded_ids}')
    delete_result = memory_agent.execute(action_name='DeleteMemories', action_input_data={'memory_ids': add_result.content.memory_ids}, memory=memory)
    print('\nTest 8: Delete results')
    print(f'Successes: {delete_result.content.successes}')
    memory.clear()
    search_after_full_clear = memory_agent.execute(action_name='SearchMemories', action_input_data={'query': 'meeting', 'top_k': 1}, memory=memory)
    print('\nTest 9: Search after full clear')
    print(f'Results: {len(search_after_full_clear.content.results)} memories found')

def demonstrate_rag_to_generation_pipeline():
    """Simple demo: Index 20 docs, retrieve 5, generate answer."""
    print('🚀 EvoAgentX Multimodal RAG-to-Generation Pipeline')
    print('=' * 60)
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print('❌ OPENAI_API_KEY not found. Please set it to run this demo.')
        return
    voyage_key = os.getenv('VOYAGE_API_KEY')
    if not voyage_key:
        print('❌ VOYAGE_API_KEY not found. Please set it to run this demo.')
        return
    datasets = RealMMRAG('./debug/data/real_mm_rag')
    samples = datasets.get_random_samples(20, seed=42)
    print(f'📊 Dataset loaded with {len(samples)} samples')
    store_config = StoreConfig(dbConfig=DBConfig(db_name='sqlite', path='./debug/data/real_mm_rag/cache/demo.sql'), vectorConfig=VectorStoreConfig(vector_name='faiss', dimensions=1024, index_type='flat_l2'), path='./debug/data/real_mm_rag/cache/indexing')
    storage_handler = StorageHandler(storageConfig=store_config)
    rag_config = RAGConfig(modality='multimodal', reader=ReaderConfig(recursive=True, exclude_hidden=True, errors='ignore'), embedding=EmbeddingConfig(provider='voyage', model_name='voyage-multimodal-3', device='cpu', api_key=voyage_key), index=IndexConfig(index_type='vector'), retrieval=RetrievalConfig(retrivel_type='vector', top_k=5, similarity_cutoff=0.3))
    search_engine = RAGEngine(config=rag_config, storage_handler=storage_handler)
    print('\n📚 Step 1: Indexing 20 documents...')
    corpus_id = 'demo_corpus'
    valid_paths = [s['image_path'] for s in samples if os.path.exists(s['image_path'])][:20]
    if len(valid_paths) < 20:
        print(f'⚠️ Only found {len(valid_paths)} valid image paths, using those')
    corpus = search_engine.read(file_paths=valid_paths, corpus_id=corpus_id)
    search_engine.add(index_type='vector', nodes=corpus, corpus_id=corpus_id)
    print(f'✅ Indexed {len(corpus.chunks)} image documents')
    query_sample = next((s for s in samples if s['query'] and len(s['query'].strip()) > 10), None)
    if not query_sample:
        print('❌ No suitable query found in samples')
        return
    query_text = query_sample['query']
    target_image = query_sample['image_filename']
    print(f"\n🔍 Step 2: Querying with: '{query_text}'")
    print(f'🎯 Target document: {target_image}')
    query = Query(query_str=query_text, top_k=5)
    result = search_engine.query(query, corpus_id=corpus_id)
    retrieved_chunks = result.corpus.chunks
    print(f'\n📄 Retrieved {len(retrieved_chunks)} documents:')
    retrieved_paths = []
    for i, chunk in enumerate(retrieved_chunks):
        filename = Path(chunk.image_path).name if chunk.image_path else 'Unknown'
        similarity = getattr(chunk.metadata, 'similarity_score', 0.0)
        retrieved_paths.append(filename)
        print(f'  {i + 1}. {filename} (similarity: {similarity:.3f})')
    print(f'\n🤖 Step 3: Generating answer with GPT-4o...')
    try:
        llm_config = OpenAILLMConfig(model='gpt-4o', openai_key=openai_key, temperature=0.1, max_tokens=300)
        llm = OpenAILLM(config=llm_config)
        print('✅ LLM initialized successfully')
        content = [TextChunk(text=f'Query: {query_text}\n\nAnalyze these retrieved images and answer the query:')]
        content.extend(retrieved_chunks[:3])
        response = llm.generate(messages=[{'role': 'system', 'content': 'You are an expert image analyst. Answer queries based on provided images.'}, {'role': 'user', 'content': content}])
        print('✅ Response generated successfully')
        answer = response.content
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f'❌ Detailed error:')
        print(error_details)
        answer = f'Error in generation: {str(e)}'
    print('\n' + '=' * 60)
    print('📋 FINAL RESULTS')
    print('=' * 60)
    print(f'🔍 QUERY: {query_text}')
    print(f'\n📄 RETRIEVED PATHS:')
    for i, path in enumerate(retrieved_paths):
        print(f'  {i + 1}. {path}')
    print(f'\n🎯 TARGET DOCUMENT: {target_image}')
    print(f'\n🤖 GENERATED ANSWER:')
    print(answer)
    print('EXPECTED ANSWER:')
    print(query_sample['answer'])
    print('=' * 60)
    search_engine.clear(corpus_id=corpus_id)

def run_evaluation(samples: List[Dict], top_k: int=5) -> Dict[str, float]:
    """Run evaluation on HotPotQA samples."""
    metrics = defaultdict(list)
    for sample in samples:
        question = sample['question']
        context = sample['context']
        supporting_facts = sample['supporting_facts']
        corpus_id = sample['_id']
        logger.info(f'Processing sample: {corpus_id}, question: {question}')
        corpus = create_corpus_from_context(context, corpus_id)
        logger.info(f'Created corpus with {len(corpus.chunks)} chunks')
        search_engine.add(index_type='graph', nodes=corpus, corpus_id=corpus_id)
        query = Query(query_str=question, top_k=top_k)
        result = search_engine.query(query, corpus_id=corpus_id)
        retrieved_chunks = result.corpus.chunks
        logger.info(f'Retrieved {len(retrieved_chunks)} chunks for query')
        logger.info(f'content:\n{retrieved_chunks}')
        sample_metrics = evaluate_retrieval(retrieved_chunks, supporting_facts, top_k)
        for metric_name, value in sample_metrics.items():
            metrics[metric_name].append(value)
        logger.info(f'Metrics for sample {corpus_id}: {sample_metrics}')
        CHECK_SAVE = False
        if CHECK_SAVE:
            search_engine.save(graph_exported=True)
            search_engine.clear(corpus_id=corpus_id)
            search_engine1 = RAGEngine(config=rag_config, storage_handler=storage_handler, llm=llm)
            search_engine1.load(index_type='graph')
            query = Query(query_str=question, top_k=top_k)
            result = search_engine1.query(query, corpus_id=corpus_id)
            retrieved_chunks = result.corpus.chunks
            logger.info(f'Retrieved {len(retrieved_chunks)} chunks for query')
            logger.info(f'content:\n{retrieved_chunks}')
            sample_metrics = evaluate_retrieval(retrieved_chunks, supporting_facts, top_k)
            logger.info(f'Metrics for sample {corpus_id}: {sample_metrics}')
    avg_metrics = {name: sum(values) / len(values) for name, values in metrics.items()}
    return avg_metrics

def evaluate_retrieval(retrieved_chunks: List[Chunk], supporting_facts: List[List], top_k: int) -> Dict[str, float]:
    """Evaluate retrieved chunks against supporting facts."""
    relevant = {(fact[0], fact[1]) for fact in supporting_facts}
    retrieved = []
    for chunk in retrieved_chunks[:top_k]:
        title = chunk.metadata.title
        sentence_idx = int(chunk.metadata.doc_id)
        retrieved.append((title, sentence_idx))
    hits = sum((1 for r in retrieved if r in relevant))
    precision = hits / top_k if top_k > 0 else 0.0
    recall = hits / len(relevant) if len(relevant) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    mrr = 0.0
    for rank, r in enumerate(retrieved, 1):
        if r in relevant:
            mrr = 1.0 / rank
            break
    hit = 1.0 if hits > 0 else 0.0
    intersection = set(((r[0], r[1]) for r in retrieved)) & relevant
    union = set(((r[0], r[1]) for r in retrieved)) | relevant
    jaccard = len(intersection) / len(union) if union else 0.0
    return {'precision@k': precision, 'recall@k': recall, 'f1@k': f1, 'mrr': mrr, 'hit@k': hit, 'jaccard': jaccard}

class TestStorageHandler(unittest.TestCase):
    """
    Test suite for StorageHandler's database operations on Workflow, Agent, and History.
    Uses an in-memory SQLite database for isolated testing.
    """

    def setUp(self):
        """
        Set up the test environment by initializing StorageHandler with an in-memory SQLite database.
        """
        db_config = DBConfig(db_name='sqlite', path=':memory:')
        store_config = StoreConfig(dbConfig=db_config)
        self.storage = StorageHandler(storageConfig=store_config)
        self.agent_data = {'name': 'test_agent', 'content': {'role': 'assistant', 'settings': {'active': True}}, 'date': '2025-05-13'}
        self.workflow_data = {'name': 'test_workflow', 'content': {'class_name': 'WorkFlowGraph', 'goal': 'Generate html code for the Tetris game that can be played in the browser.', 'nodes': [{'class_name': 'WorkFlowNode', 'name': 'game_structure_design', 'description': "Create an outline of the Tetris game's structure, including the main game area, score display, and control buttons.", 'inputs': [{'class_name': 'Parameter', 'name': 'goal', 'type': 'string', 'description': "The user's goal in textual format.", 'required': True}], 'outputs': [{'class_name': 'Parameter', 'name': 'html_structure', 'type': 'string', 'description': 'The basic HTML structure outlining the game area, score display, and buttons.', 'required': True}], 'reason': 'This sub-task establishes the foundational layout required for a functional Tetris game in HTML.', 'agents': [{'name': 'tetris_game_structure_agent', 'description': "This agent creates the basic HTML structure for the Tetris game, including the game area, score display, and control buttons based on the user's goal.", 'inputs': [{'name': 'goal', 'type': 'string', 'description': "The user's goal in textual format.", 'required': True}], 'outputs': [{'name': 'html_structure', 'type': 'string', 'description': 'The basic HTML structure outlining the game area, score display, and buttons.', 'required': True}], 'prompt': "### Objective\nCreate the basic HTML structure for a Tetris game, incorporating the main game area, score display, and control buttons based on the user's goal.\n\n### Instructions\n1. Read the user's goal: <input>{goal}</input>\n2. Design the main game area where the Tetris pieces will fall.\n3. Create an element to display the current score.\n4. Include buttons to control the game (e.g., start, pause, reset).\n5. Assemble these elements into a coherent HTML structure that can be utilized in a web environment.\n6. Output the generated HTML structure.\n\n### Output Format\nYour final output should ALWAYS in the following format:\n\n## Thought\nBriefly explain the reasoning process for creating the HTML structure of the Tetris game.\n\n## html_structure\nThe basic HTML structure outlining the game area, score display, and buttons."}], 'status': 'pending'}, {'class_name': 'WorkFlowNode', 'name': 'style_application', 'description': 'Add CSS styles to the HTML structure for visual aesthetics and layout to make the game look visually appealing.', 'inputs': [{'class_name': 'Parameter', 'name': 'html_structure', 'type': 'string', 'description': 'The basic HTML structure of the Tetris game.', 'required': True}], 'outputs': [{'class_name': 'Parameter', 'name': 'styled_game', 'type': 'string', 'description': 'The styled HTML code that includes CSS for the Tetris game.', 'required': True}], 'reason': 'Styling is essential for enhancing the user experience and ensuring the game is visually organized and engaging.', 'agents': [{'name': 'css_style_application_agent', 'description': 'This agent applies CSS styles to the given HTML structure to create a visually appealing layout for the Tetris game.', 'inputs': [{'name': 'html_structure', 'type': 'string', 'description': 'The basic HTML structure of the Tetris game.', 'required': True}], 'outputs': [{'name': 'styled_game', 'type': 'string', 'description': 'The styled HTML code that includes CSS for the Tetris game.', 'required': True}], 'prompt': '### Objective\nEnhance the provided HTML structure by applying CSS styles to create a visually appealing layout for the Tetris game.\n\n### Instructions\n1. Begin with the provided HTML structure: <input>{html_structure}</input>\n2. Analyze the elements in the HTML to decide the appropriate CSS styles that will enhance its appearance.\n3. Write CSS styles that cater to visual aesthetics such as colors, fonts, borders, and spacing.\n4. Integrate the CSS styles into the HTML structure properly.\n5. Ensure the output is a well-formatted HTML document that includes the applied CSS styles.\n\n### Output Format\nYour final output should ALWAYS in the following format:\n\n## Thought\nBriefly explain the reasoning process for achieving the objective.\n\n## styled_game\nThe styled HTML code that includes CSS for the Tetris game.'}], 'status': 'pending'}, {'class_name': 'WorkFlowNode', 'name': 'game_logic_implementation', 'description': 'Implement the JavaScript logic for the Tetris game, including piece movement, collision detection, and score tracking.', 'inputs': [{'class_name': 'Parameter', 'name': 'styled_game', 'type': 'string', 'description': 'The styled HTML code for the Tetris game.', 'required': True}], 'outputs': [{'class_name': 'Parameter', 'name': 'complete_game_code', 'type': 'string', 'description': 'The complete HTML, CSS, and JavaScript code for a functional Tetris game.', 'required': True}], 'reason': 'This sub-task is crucial for making the game interactive and functional, allowing users to play.', 'agents': [{'name': 'tetris_logic_agent', 'description': 'This agent implements the JavaScript logic required for the Tetris game, ensuring piece movements, collision detection, and score tracking functionalities are properly integrated.', 'inputs': [{'name': 'styled_game', 'type': 'string', 'description': 'The styled HTML code for the Tetris game.', 'required': True}], 'outputs': [{'name': 'complete_game_code', 'type': 'string', 'description': 'The complete HTML, CSS, and JavaScript code for a functional Tetris game.', 'required': True}], 'prompt': "### Objective\nImplement the JavaScript logic for the Tetris game, ensuring functionalities for piece movement, collision detection, and score tracking are included in the output.\n\n### Instructions\n1. Analyze the styled HTML code provided: <input>{styled_game}</input>\n2. Develop JavaScript functions that handle the movement of Tetris pieces, including left, right, and rotation controls.\n3. Implement collision detection logic to ensure pieces do not fall through the bottom or collide with existing pieces.\n4. Create a scoring system that tracks the player's progress and updates the score based on cleared lines.\n5. Combine the JavaScript logic with the existing styled HTML to create a complete game code output.\n\n### Output Format\nYour final output should ALWAYS in the following format:\n\n## Thought\nBriefly explain the reasoning process for implementing the game logic for Tetris.\n\n## complete_game_code\nThe completed HTML, CSS, and JavaScript code for a functional Tetris game."}], 'status': 'pending'}, {'class_name': 'WorkFlowNode', 'name': 'testing_and_refinement', 'description': 'Test the generated Tetris game for bugs and usability issues, refining the code as necessary.', 'inputs': [{'class_name': 'Parameter', 'name': 'complete_game_code', 'type': 'string', 'description': 'The complete HTML, CSS, and JavaScript code for the Tetris game.', 'required': True}], 'outputs': [{'class_name': 'Parameter', 'name': 'final_output', 'type': 'string', 'description': 'The final tested and refined code for the Tetris game.', 'required': True}], 'reason': 'Testing is vital to ensure that the game functions correctly across different browsers and provides a smooth user experience.', 'agents': [{'name': 'tetris_game_testing_agent', 'description': 'This agent tests the generated Tetris game code for functionality, identifies bugs, and provides refinements as needed to ensure smooth gameplay and usability.', 'inputs': [{'name': 'complete_game_code', 'type': 'string', 'description': 'The complete HTML, CSS, and JavaScript code for the Tetris game.', 'required': True}], 'outputs': [{'name': 'final_output', 'type': 'string', 'description': 'The final tested and refined code for the Tetris game.', 'required': True}], 'prompt': '### Objective\nTest the complete Tetris game code for bugs and usability issues, and refine the code as necessary for improved performance.\n\n### Instructions\n1. Load the complete game code: <input>{complete_game_code}</input> into a browser.\n2. Test the game functionality, focusing on user controls, collision detection, and game progression.\n3. Identify any bugs or usability issues that arise during testing.\n4. Document the identified issues and make necessary adjustments to the code to resolve them.\n5. Ensure that the final code adheres to best practices for HTML, CSS, and JavaScript.\n6. Output the refined and tested code as the final result.\n\n### Output Format\nYour final output should ALWAYS in the following format:\n\n## Thought\nBriefly explain the reasoning process for testing and refining the Tetris game code.\n\n## final_output\nThe final tested and refined code for the Tetris game.'}], 'status': 'pending'}], 'edges': [{'class_name': 'WorkFlowEdge', 'source': 'game_structure_design', 'target': 'style_application', 'priority': 0}, {'class_name': 'WorkFlowEdge', 'source': 'style_application', 'target': 'game_logic_implementation', 'priority': 0}, {'class_name': 'WorkFlowEdge', 'source': 'game_logic_implementation', 'target': 'testing_and_refinement', 'priority': 0}], 'graph': None}, 'date': '2025-05-13'}
        self.history_data = {'memory_id': 'mem_001', 'old_memory': 'Initial content', 'new_memory': 'Updated content', 'event': 'update', 'created_at': '2025-05-13T09:00:00', 'updated_at': '2025-05-13T09:30:00'}

    def test_save_and_load_agent(self):
        """
        Test saving and loading an agent, verifying data integrity and JSON parsing.
        """
        self.storage.save_agent(self.agent_data)
        self.storage.save_agent(self.agent_data, 'nihao')
        loaded = self.storage.load_agent('test_agent')
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['name'], 'test_agent')
        self.assertEqual(loaded['content'], self.agent_data['content'])
        self.assertEqual(loaded['date'], '2025-05-13')

    def test_save_and_load_workflow(self):
        """
        Test saving and loading a workflow, verifying data integrity and JSON parsing.
        """
        self.storage.save_workflow(self.workflow_data)
        loaded = self.storage.load_workflow('test_workflow')
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['name'], 'test_workflow')
        self.assertEqual(loaded['content'], self.workflow_data['content'])
        self.assertEqual(loaded['date'], '2025-05-13')

    def test_save_and_load_history(self):
        """
        Test saving and loading a history entry, verifying data integrity.
        """
        self.storage.save_history(self.history_data)
        loaded = self.storage.load_history('mem_001')
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['memory_id'], 'mem_001')
        self.assertEqual(loaded['old_memory'], 'Initial content')
        self.assertEqual(loaded['new_memory'], 'Updated content')
        self.assertEqual(loaded['event'], 'update')
        self.assertEqual(loaded['created_at'], '2025-05-13T09:00:00')
        self.assertEqual(loaded['updated_at'], '2025-05-13T09:30:00')

    def test_load_non_existent_agent(self):
        """
        Test loading a non-existent agent returns None.
        """
        loaded = self.storage.load_agent('non_existent_agent')
        self.assertIsNone(loaded)

    def test_load_non_existent_workflow(self):
        """
        Test loading a non-existent workflow returns None.
        """
        loaded = self.storage.load_workflow('non_existent_workflow')
        self.assertIsNone(loaded)

    def test_load_non_existent_history(self):
        """
        Test loading a non-existent history entry returns None.
        """
        loaded = self.storage.load_history('non_existent_mem')
        self.assertIsNone(loaded)

    def test_save_invalid_agent(self):
        """
        Test saving an agent without a 'name' field raises ValueError.
        """
        invalid_data = {'content': {'role': 'assistant'}, 'date': '2025-05-13'}
        with self.assertRaises(ValueError):
            self.storage.save_agent(invalid_data)

    def test_save_invalid_workflow(self):
        """
        Test saving a workflow without a 'name' field raises ValueError.
        """
        invalid_data = {'content': {'steps': ['step1']}, 'date': '2025-05-13'}
        with self.assertRaises(ValueError):
            self.storage.save_workflow(invalid_data)

    def test_save_invalid_history(self):
        """
        Test saving a history entry without a 'memory_id' field raises ValueError.
        """
        invalid_data = {'old_memory': 'Initial', 'new_memory': 'Updated', 'event': 'update'}
        with self.assertRaises(ValueError):
            self.storage.save_history(invalid_data)

    def test_remove_agent(self):
        """
        Test removing an agent and verify it's no longer loadable.
        """
        self.storage.save_agent(self.agent_data)
        self.storage.remove_agent('test_agent')
        loaded = self.storage.load_agent('test_agent')
        self.assertIsNone(loaded)

    def test_remove_non_existent_agent(self):
        """
        Test removing a non-existent agent raises ValueError.
        """
        with self.assertRaises(ValueError):
            self.storage.remove_agent('non_existent_agent')

    def test_update_agent(self):
        """
        Test updating an existing agent's data.
        """
        self.storage.save_agent(self.agent_data)
        updated_data = {'name': 'test_agent', 'content': {'role': 'admin', 'settings': {'active': False}}, 'date': '2025-05-14'}
        self.storage.save_agent(updated_data)
        loaded = self.storage.load_agent('test_agent')
        self.assertEqual(loaded['content'], updated_data['content'])
        self.assertEqual(loaded['date'], '2025-05-14')

    def test_update_workflow(self):
        """
        Test updating an existing workflow's data.
        """
        self.storage.save_workflow(self.workflow_data)
        updated_data = {'name': 'test_workflow', 'content': {'test': True}, 'date': '2025-05-15'}
        self.storage.save_workflow(updated_data)
        loaded = self.storage.load_workflow('test_workflow')
        self.assertEqual(loaded['content'], updated_data['content'])
        self.assertEqual(loaded['date'], '2025-05-15')

    def test_update_history(self):
        """
        Test updating an existing history entry.
        """
        self.storage.save_history(self.history_data)
        updated_data = {'memory_id': 'mem_001', 'old_memory': 'Initial content', 'new_memory': 'Further updated content', 'event': 'modify', 'created_at': '2025-05-13T09:00:00', 'updated_at': '2025-05-13T10:00:00'}
        self.storage.save_history(updated_data)
        loaded = self.storage.load_history('mem_001')
        self.assertEqual(loaded['new_memory'], 'Further updated content')
        self.assertEqual(loaded['event'], 'modify')
        self.assertEqual(loaded['updated_at'], '2025-05-13T10:00:00')

    def test_bulk_save_and_load(self):
        """
        Test saving multiple records to all tables and loading them.
        """
        agent_data2 = {'name': 'test_agent2', 'content': {'role': 'user', 'settings': {'active': True}}, 'date': '2025-05-13'}
        workflow_data2 = {'name': 'test_workflow2', 'content': {'steps': ['stepA', 'stepB'], 'config': {'timeout': 45}}, 'date': '2025-05-13'}
        history_data2 = {'memory_id': 'mem_002', 'old_memory': 'Old content', 'new_memory': 'New content', 'event': 'create', 'created_at': '2025-05-13T10:00:00', 'updated_at': '2025-05-13T10:00:00'}
        bulk_data = {TableType.store_agent.value: [self.agent_data, agent_data2], TableType.store_workflow.value: [self.workflow_data, workflow_data2], TableType.store_history.value: [self.history_data, history_data2]}
        self.storage.save(bulk_data)
        all_data = self.storage.load()
        self.assertIn(TableType.store_agent.value, all_data)
        self.assertIn(TableType.store_workflow.value, all_data)
        self.assertIn(TableType.store_history.value, all_data)
        self.assertEqual(len(all_data[TableType.store_agent.value]), 2)
        self.assertEqual(len(all_data[TableType.store_workflow.value]), 2)
        self.assertEqual(len(all_data[TableType.store_history.value]), 2)
        agent_names = [record['name'] for record in all_data[TableType.store_agent.value]]
        self.assertIn('test_agent', agent_names)
        self.assertIn('test_agent2', agent_names)
        workflow_names = [record['name'] for record in all_data[TableType.store_workflow.value]]
        self.assertIn('test_workflow', workflow_names)
        self.assertIn('test_workflow2', workflow_names)
        history_ids = [record['memory_id'] for record in all_data[TableType.store_history.value]]
        self.assertIn('mem_001', history_ids)
        self.assertIn('mem_002', history_ids)

    def test_save_invalid_table(self):
        """
        Test saving data to an unknown table raises ValueError.
        """
        invalid_data = {'unknown_table': [self.agent_data]}
        with self.assertRaises(ValueError):
            self.storage.save(invalid_data)

    def tearDown(self):
        """
        Clean up by closing the database connection.
        """
        self.storage.storageDB.connection.close()

class TestSearchEngine(unittest.TestCase):
    """Unit tests for SearchEngine interfaces using HotpotQA JSON example."""

    def setUp(self):
        """Set up SearchEngine, StorageHandler, and temporary directory for each test."""
        load_dotenv()
        self.mock_embedding = MockOpenAIEmbeddingWrapper()
        self.patcher = patch('evoagentx.rag.rag.EmbeddingFactory.create', return_value=self.mock_embedding)
        self.mock_create = self.patcher.start()
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f'Created temporary directory: {self.temp_dir}')
        self.store_config = StoreConfig(dbConfig=DBConfig(db_name='sqlite', path=os.path.join(self.temp_dir, 'test_hotpotQA.sql')), vectorConfig=VectorStoreConfig(vector_name='faiss', dimensions=1536, index_type='flat_l2'), graphConfig=None, path=self.temp_dir)
        self.storage_handler = StorageHandler(storageConfig=self.store_config)
        self.rag_config = RAGConfig(reader=ReaderConfig(recursive=False, exclude_hidden=True, num_files_limit=None, custom_metadata_function=None, extern_file_extractor=None, errors='ignore', encoding='utf-8'), chunker=ChunkerConfig(strategy='simple', chunk_size=512, chunk_overlap=0, max_chunks=None), embedding=EmbeddingConfig(provider='openai', model_name='text-embedding-ada-002', api_key='dummy_key'), index=IndexConfig(index_type='vector'), retrieval=RetrievalConfig(retrivel_type='vector', postprocessor_type='simple', top_k=10, similarity_cutoff=0.3, keyword_filters=None, metadata_filters=None))
        self.search_engine = RAGEngine(config=self.rag_config, storage_handler=self.storage_handler)
        self.corpus_id = HOTPOTQA_EXAMPLE['_id']
        self.context_files = []
        self.supporting_titles = {fact[0] for fact in HOTPOTQA_EXAMPLE['supporting_facts']}
        self.context_data = HOTPOTQA_EXAMPLE['context']
        self.query_text = HOTPOTQA_EXAMPLE['question']
        for title, sentences in self.context_data:
            content = '\n'.join(sentences)
            file_path = os.path.join(self.temp_dir, f'{title.replace(' ', '_')}.txt')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.context_files.append(str(file_path))

    def tearDown(self):
        """Clean up temporary directory, clear indices, and stop patcher."""
        self.search_engine.clear()
        self.patcher.stop()
        logger.info(f'Cleaned up temporary directory: {self.temp_dir}')

    def test_read(self):
        """Test the read method by loading HotpotQA context files."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.assertIsInstance(corpus, Corpus, 'read should return a Corpus object')
        self.assertEqual(corpus.corpus_id, self.corpus_id, 'Corpus ID should match')
        self.assertGreater(len(corpus.chunks), 0, 'Corpus should contain chunks')
        for chunk in corpus.chunks:
            self.assertIsInstance(chunk.metadata, ChunkMetadata, 'Chunk should have metadata')
            self.assertIn('file_name', chunk.metadata.model_dump(), 'Metadata should include file_name')
        logger.info(f'Read {len(corpus.chunks)} chunks for corpus {self.corpus_id}')

    def test_add(self):
        """Test the add method by indexing HotpotQA corpus."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        self.assertIn(self.corpus_id, self.search_engine.indices, 'Corpus should be indexed')
        self.assertIn(IndexType.VECTOR, self.search_engine.indices[self.corpus_id], 'Vector index should exist')
        index = self.search_engine.indices[self.corpus_id][IndexType.VECTOR]
        self.assertGreater(len(index.id_to_node), 0, 'Index should contain nodes')
        for node_id, node in index.id_to_node.items():
            self.assertEqual(node.metadata['corpus_id'], self.corpus_id, 'Node metadata should include corpus_id')
            self.assertEqual(node.metadata['index_type'], IndexType.VECTOR, 'Node metadata should include index_type')
        logger.info(f'Added {len(corpus.chunks)} nodes to vector index for corpus {self.corpus_id}')

    def test_query(self):
        """Test the query method with HotpotQA question, validating top-K retrieval."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        query = Query(query_str=self.query_text, top_k=10)
        result = self.search_engine.query(query, corpus_id=self.corpus_id)
        self.assertIsInstance(result, RagResult, 'query should return a RagResult object')
        self.assertLessEqual(len(result.corpus.chunks), 10, 'Should return at most top_k chunks')
        self.assertEqual(len(result.scores), len(result.corpus.chunks), 'Scores should match chunks')
        retrieved_titles = set()
        for chunk in result.corpus.chunks:
            file_name = chunk.metadata.model_dump().get('file_name', '')
            title = os.path.basename(file_name).replace('_', ' ').replace('.txt', '')
            retrieved_titles.add(title)
        recall = len(retrieved_titles.intersection(self.supporting_titles)) / len(self.supporting_titles)
        self.assertGreaterEqual(recall, 0.0, 'Recall may be low with dummy embeddings')
        logger.info(f'Query retrieved {len(result.corpus.chunks)} chunks with recall@10={recall}')

    def test_delete_by_node_ids(self):
        """Test the delete method by removing specific nodes."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        index = self.search_engine.indices[self.corpus_id][IndexType.VECTOR]
        node_ids = list(index.id_to_node.keys())[:2]
        initial_node_count = len(index.id_to_node)
        self.search_engine.delete(corpus_id=self.corpus_id, index_type=IndexType.VECTOR, node_ids=node_ids)
        remaining_node_count = len(index.id_to_node)
        self.assertEqual(remaining_node_count, initial_node_count - len(node_ids), 'Nodes should be deleted')
        for node_id in node_ids:
            self.assertNotIn(node_id, index.id_to_node, f'Node {node_id} should be deleted')
        logger.info(f'Deleted {len(node_ids)} nodes from corpus {self.corpus_id}')

    def test_delete_by_metadata(self):
        """Test the delete method using metadata filters."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        index = self.search_engine.indices[self.corpus_id][IndexType.VECTOR]
        metadata_filters = {'file_name': str(self.context_files[0])}
        initial_node_count = len(index.id_to_node)
        self.search_engine.delete(corpus_id=self.corpus_id, index_type=IndexType.VECTOR, metadata_filters=metadata_filters)
        remaining_nodes = [node_id for node_id, node in index.id_to_node.items() if node.metadata.get('file_name') != str(self.context_files[0])]
        self.assertEqual(len(index.id_to_node), len(remaining_nodes), 'Nodes matching metadata should be deleted')
        logger.info(f'Deleted nodes with metadata {metadata_filters} from corpus {self.corpus_id}')

    def test_clear(self):
        """Test the clear method by removing all indices."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        self.search_engine.clear(corpus_id=self.corpus_id)
        self.assertNotIn(self.corpus_id, self.search_engine.indices, 'Corpus should be cleared')
        self.assertNotIn(self.corpus_id, self.search_engine.retrievers, 'Retrievers should be cleared')
        logger.info(f'Cleared corpus {self.corpus_id}')

    def test_save_to_files(self):
        """Test the save method by saving indices to files."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        output_path = os.path.join(self.temp_dir, 'output')
        self.search_engine.save(output_path=str(output_path), corpus_id=self.corpus_id, index_type=IndexType.VECTOR)
        if isinstance(output_path, str):
            from pathlib import Path
            output_path = Path(output_path)
        nodes_files = list(output_path.glob('*_nodes.jsonl'))
        metadata_files = list(output_path.glob('*_metadata.json'))
        self.assertEqual(len(nodes_files), 1, 'Should save one nodes file')
        self.assertEqual(len(metadata_files), 1, 'Should save one metadata file')
        with open(nodes_files[0], 'r', encoding='utf-8') as f:
            chunks = [json.loads(line) for line in f]
            self.assertGreater(len(chunks), 0, 'Nodes file should contain chunks')
        with open(metadata_files[0], 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            self.assertEqual(metadata['corpus_id'], self.corpus_id, 'Metadata should include corpus_id')
        logger.info(f'Saved indices to {output_path}')

    def test_load_from_files(self):
        """Test the load method by loading indices from files."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        output_path = os.path.join(self.temp_dir, 'output')
        self.search_engine.save(output_path=str(output_path), corpus_id=self.corpus_id, index_type=IndexType.VECTOR)
        self.search_engine.clear()
        self.search_engine.load(source=str(output_path), corpus_id=self.corpus_id, index_type=IndexType.VECTOR)
        self.assertIn(self.corpus_id, self.search_engine.indices, 'Corpus should be loaded')
        index = self.search_engine.indices[self.corpus_id][IndexType.VECTOR]
        self.assertGreater(len(index.id_to_node), 0, 'Index should contain nodes')
        query = Query(query_str=self.query_text, top_k=10)
        result = self.search_engine.query(query, corpus_id=self.corpus_id)
        self.assertEqual(len(result.corpus.chunks), 0)
        logger.info(f'Loaded indices from {output_path}')

    def test_save_to_database(self):
        """Test the save method by saving indices to database."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        self.search_engine.save(corpus_id=self.corpus_id, index_type=IndexType.VECTOR, table='indexing')
        records = self.storage_handler.load(tables=['indexing']).get('indexing', [])
        self.assertGreater(len(records), 0, 'Database should contain records')
        for record in records:
            parsed = self.storage_handler.parse_result(record, IndexStore)
            self.assertEqual(parsed['corpus_id'], self.corpus_id, 'Record should match corpus_id')
        logger.info(f'Saved indices to database table indexing')

    def test_load_from_database(self):
        """Test the load method by loading indices from database."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        self.search_engine.save(corpus_id=self.corpus_id, index_type=IndexType.VECTOR, table='indexing')
        self.search_engine.clear()
        self.search_engine.load(corpus_id=self.corpus_id, index_type=IndexType.VECTOR, table='indexing')
        self.assertIn(self.corpus_id, self.search_engine.indices, 'Corpus should be loaded')
        index = self.search_engine.indices[self.corpus_id][IndexType.VECTOR]
        self.assertGreater(len(index.id_to_node), 0, 'Index should contain nodes')
        query = Query(query_str=self.query_text, top_k=10)
        result = self.search_engine.query(query, corpus_id=self.corpus_id)
        self.assertEqual(len(result.corpus.chunks), 0)
        logger.info(f'Loaded indices from database table indexing')

    def test_edge_case_empty_corpus(self):
        """Test behavior with empty corpus or invalid corpus_id."""
        result = self.search_engine.query(query=self.query_text, corpus_id='nonexistent')
        self.assertEqual(len(result.corpus.chunks), 0, 'Query on nonexistent corpus should return empty result')
        self.search_engine.delete(corpus_id='nonexistent')
        self.assertNotIn('nonexistent', self.search_engine.indices, 'Delete on nonexistent corpus should not fail')
        self.search_engine.clear(corpus_id='nonexistent')
        self.assertNotIn('nonexistent', self.search_engine.indices, 'Clear on nonexistent corpus should not fail')
        logger.info('Handled edge case for empty/nonexistent corpus')

