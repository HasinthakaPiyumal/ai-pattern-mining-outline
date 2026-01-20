# Cluster 67

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

class AddMemories(Action):

    def __init__(self, name: str='AddMemories', description: str='Add multiple messages to long-term memory', prompt: str='Add the following messages to memory: {messages}', inputs_format: ActionInput=None, outputs_format: ActionOutput=None, **kwargs):
        inputs_format = inputs_format or AddMemoriesInput
        outputs_format = outputs_format or AddMemoriesOutput
        super().__init__(name=name, description=description, prompt=prompt, inputs_format=inputs_format, outputs_format=outputs_format, **kwargs)

    def execute(self, llm: Optional[BaseLLM]=None, inputs: Optional[Dict]=None, sys_msg: Optional[str]=None, return_prompt: bool=False, memory: Optional[LongTermMemory]=None, **kwargs) -> AddMemoriesOutput:
        if memory is None:
            raise ValueError('LongTermMemory instance required')
        action_input_attrs = self.inputs_format.get_attrs()
        action_input_data = {attr: inputs.get(attr, []) for attr in action_input_attrs}
        messages = [Message(content=msg.get('content', ''), action=msg.get('action'), wf_goal=msg.get('wf_goal'), timestamp=msg.get('timestamp', datetime.now().isoformat()), agent=msg.get('agent', 'user'), msg_type=msg.get('msg_type', MessageType.REQUEST), prompt=msg.get('prompt'), next_actions=msg.get('next_actions'), wf_task=msg.get('wf_task'), wf_task_desc=msg.get('wf_task_desc'), message_id=msg.get('message_id')) for msg in action_input_data['messages']]
        memory_ids = memory.add(messages)
        output = AddMemoriesOutput(memory_ids=memory_ids)
        if return_prompt:
            prompt = self.prompt.format(messages=[msg.model_dump() for msg in messages])
            return (output, prompt)
        return output

class UpdateMemories(Action):

    def __init__(self, name: str='UpdateMemories', description: str='Update multiple memories by IDs', prompt: str='Update the memories with the following data: {updates}', inputs_format: ActionInput=None, outputs_format: ActionOutput=None, **kwargs):
        inputs_format = inputs_format or UpdateMemoriesInput
        outputs_format = outputs_format or UpdateMemoriesOutput
        super().__init__(name=name, description=description, prompt=prompt, inputs_format=inputs_format, outputs_format=outputs_format, **kwargs)

    def execute(self, llm: Optional[BaseLLM]=None, inputs: Optional[Dict]=None, sys_msg: Optional[str]=None, return_prompt: bool=False, memory: Optional[LongTermMemory]=None, **kwargs) -> UpdateMemoriesOutput:
        if memory is None:
            raise ValueError('LongTermMemory instance required')
        action_input_attrs = self.inputs_format.get_attrs()
        action_input_data = {attr: inputs.get(attr, []) for attr in action_input_attrs}
        updates = [(update['memory_id'], Message(content=update.get('content', ''), action=update.get('action'), wf_goal=update.get('wf_goal'), timestamp=update.get('timestamp', datetime.now().isoformat()), agent=update.get('agent', 'user'), msg_type=update.get('msg_type', MessageType.REQUEST), prompt=update.get('prompt'), next_actions=update.get('next_actions'), wf_task=update.get('wf_task'), wf_task_desc=update.get('wf_task_desc'), message_id=update.get('message_id'))) for update in action_input_data['updates']]
        successes = memory.update(updates)
        output = UpdateMemoriesOutput(successes=successes)
        if return_prompt:
            prompt = self.prompt.format(updates=[{'memory_id': mid, 'message': msg.model_dump()} for mid, msg in updates])
            return (output, prompt)
        return output

class AddMemories(Action):

    def __init__(self, name: str='AddMemories', description: str='Add multiple messages to long-term memory', prompt: str='Add the following messages to memory: {messages}', inputs_format: ActionInput=None, outputs_format: ActionOutput=None, **kwargs):
        inputs_format = inputs_format or AddMemoriesInput
        outputs_format = outputs_format or AddMemoriesOutput
        super().__init__(name=name, description=description, prompt=prompt, inputs_format=inputs_format, outputs_format=outputs_format, **kwargs)

    def execute(self, llm: Optional[BaseLLM]=None, inputs: Optional[Dict]=None, sys_msg: Optional[str]=None, return_prompt: bool=False, memory: Optional[LongTermMemory]=None, **kwargs) -> AddMemoriesOutput:
        if memory is None:
            raise ValueError('LongTermMemory instance required')
        action_input_attrs = self.inputs_format.get_attrs()
        action_input_data = {attr: inputs.get(attr, []) for attr in action_input_attrs}
        messages = [Message(content=msg.get('content', ''), action=msg.get('action'), wf_goal=msg.get('wf_goal'), timestamp=msg.get('timestamp', datetime.now().isoformat()), agent=msg.get('agent', 'user'), msg_type=msg.get('msg_type', MessageType.REQUEST), prompt=msg.get('prompt'), next_actions=msg.get('next_actions'), wf_task=msg.get('wf_task'), wf_task_desc=msg.get('wf_task_desc'), message_id=msg.get('message_id')) for msg in action_input_data['messages']]
        memory_ids = memory.add(messages)
        output = AddMemoriesOutput(memory_ids=memory_ids)
        if return_prompt:
            prompt = self.prompt.format(messages=[msg.model_dump() for msg in messages])
            return (output, prompt)
        return output

class UpdateMemories(Action):

    def __init__(self, name: str='UpdateMemories', description: str='Update multiple memories by IDs', prompt: str='Update the memories with the following data: {updates}', inputs_format: ActionInput=None, outputs_format: ActionOutput=None, **kwargs):
        inputs_format = inputs_format or UpdateMemoriesInput
        outputs_format = outputs_format or UpdateMemoriesOutput
        super().__init__(name=name, description=description, prompt=prompt, inputs_format=inputs_format, outputs_format=outputs_format, **kwargs)

    def execute(self, llm: Optional[BaseLLM]=None, inputs: Optional[Dict]=None, sys_msg: Optional[str]=None, return_prompt: bool=False, memory: Optional[LongTermMemory]=None, **kwargs) -> UpdateMemoriesOutput:
        if memory is None:
            raise ValueError('LongTermMemory instance required')
        action_input_attrs = self.inputs_format.get_attrs()
        action_input_data = {attr: inputs.get(attr, []) for attr in action_input_attrs}
        updates = [(update['memory_id'], Message(content=update.get('content', ''), action=update.get('action'), wf_goal=update.get('wf_goal'), timestamp=update.get('timestamp', datetime.now().isoformat()), agent=update.get('agent', 'user'), msg_type=update.get('msg_type', MessageType.REQUEST), prompt=update.get('prompt'), next_actions=update.get('next_actions'), wf_task=update.get('wf_task'), wf_task_desc=update.get('wf_task_desc'), message_id=update.get('message_id'))) for update in action_input_data['updates']]
        successes = memory.update(updates)
        output = UpdateMemoriesOutput(successes=successes)
        if return_prompt:
            prompt = self.prompt.format(updates=[{'memory_id': mid, 'message': msg.model_dump()} for mid, msg in updates])
            return (output, prompt)
        return output

class TestWorkFlowManager(unittest.TestCase):

    def setUp(self):
        self.mock_llm = Mock(spec=BaseLLM)
        self.mock_llm.generate = Mock()
        self.mock_llm.async_generate = AsyncMock()
        self.task_output = TaskSchedulerOutput(decision='forward', task_name='Task2', reason='This is the next logical step')
        self.action_output = NextAction(agent='TestAgent', action='TestAction', reason='This is the appropriate action')
        self.mock_llm.generate.return_value = self.task_output
        self.mock_llm.async_generate.return_value = self.task_output
        self.workflow_manager = WorkFlowManager(llm=self.mock_llm)
        self.create_test_workflow()
        self.env = Environment()

    def create_test_workflow(self):
        """Create a test workflow with 3 tasks in sequence"""
        task1 = WorkFlowNode(name='Task1', description='First task', inputs=[Parameter(name='input1', type='string', description='Input 1')], outputs=[Parameter(name='output1', type='string', description='Output 1')], agents=['TestAgent'], status=WorkFlowNodeState.PENDING)
        task2 = WorkFlowNode(name='Task2', description='Second task', inputs=[Parameter(name='output1', type='string', description='Output from Task1')], outputs=[Parameter(name='output2', type='string', description='Output 2')], agents=['TestAgent'], status=WorkFlowNodeState.PENDING)
        task3 = WorkFlowNode(name='Task3', description='Third task', inputs=[Parameter(name='output2', type='string', description='Output from Task2')], outputs=[Parameter(name='final_output', type='string', description='Final output')], agents=['TestAgent'], status=WorkFlowNodeState.PENDING)
        edge1 = WorkFlowEdge(source='Task1', target='Task2')
        edge2 = WorkFlowEdge(source='Task2', target='Task3')
        self.workflow = WorkFlowGraph(goal='Test Workflow', nodes=[task1, task2, task3], edges=[edge1, edge2])

    def test_workflow_initialization(self):
        """Test that the workflow manager is correctly initialized"""
        self.assertIsNotNone(self.workflow_manager)
        self.assertEqual(self.mock_llm, self.workflow_manager.llm)
        self.assertIsNotNone(self.workflow_manager.task_scheduler)
        self.assertIsNotNone(self.workflow_manager.action_scheduler)

    @pytest.mark.asyncio
    @patch('evoagentx.workflow.workflow_manager.TaskScheduler.async_execute')
    async def test_sync_task_scheduling_with_single_task(self, mock_task_scheduler_execute):
        """Test that the task scheduler correctly handles the case of a single candidate task"""
        single_task_output = TaskSchedulerOutput(decision='forward', task_name='Task2', reason='Only one candidate task is available')
        mock_task_scheduler_execute.return_value = single_task_output
        self.workflow.set_node_status('Task1', WorkFlowNodeState.COMPLETED)
        task = await self.workflow_manager.schedule_next_task(graph=self.workflow, env=self.env)
        self.assertEqual('Task2', task.name)
        mock_task_scheduler_execute.assert_called_once()
        self.assertEqual(1, len(self.env.trajectory))
        message = self.env.trajectory[0].message
        self.assertIsInstance(message.content, TaskSchedulerOutput)
        self.assertEqual('Task2', message.content.task_name)
        self.assertEqual(MessageType.COMMAND, message.msg_type)

    @pytest.mark.asyncio
    @patch('evoagentx.workflow.workflow_manager.ActionScheduler.async_execute')
    async def test_action_scheduling(self, mock_action_scheduler_execute):
        """Test scheduling the next action for a task"""
        mock_action_scheduler_execute.return_value = (self.action_output, 'mock prompt')
        task = self.workflow.get_node('Task1')
        mock_agent_manager = Mock()
        action = await self.workflow_manager.schedule_next_action(goal='Test Goal', task=task, agent_manager=mock_agent_manager, env=self.env)
        self.assertEqual(self.action_output, action)
        mock_action_scheduler_execute.assert_called_once()
        self.assertEqual(1, len(self.env.trajectory))
        message = self.env.trajectory[0].message
        self.assertIsInstance(message.content, NextAction)
        self.assertEqual('TestAgent', message.content.agent)
        self.assertEqual('TestAction', message.content.action)
        self.assertEqual(MessageType.COMMAND, message.msg_type)

    @pytest.mark.asyncio
    async def test_async_task_scheduling(self):
        """Test async task scheduling with multiple candidate tasks"""
        self.mock_llm.async_generate.return_value = self.task_output
        task = await self.workflow_manager.schedule_next_task(graph=self.workflow, env=self.env)
        self.assertIsNotNone(task)
        self.assertEqual('Task2', task.name)
        self.assertEqual(1, len(self.env.trajectory))
        message = self.env.trajectory[0].message
        self.assertEqual(self.task_output, message.content)
        self.assertEqual(TrajectoryState.COMPLETED, self.env.trajectory[0].status)

    @pytest.mark.asyncio
    async def test_async_action_scheduling(self):
        """Test async action scheduling"""
        self.mock_llm.async_generate.return_value = self.action_output
        task = self.workflow.get_node('Task1')
        mock_agent_manager = Mock()
        action = await self.workflow_manager.schedule_next_action(goal='Test Goal', task=task, agent_manager=mock_agent_manager, env=self.env)
        self.assertIsNotNone(action)
        self.assertEqual('TestAgent', action.agent)
        self.assertEqual('TestAction', action.action)
        self.assertEqual(1, len(self.env.trajectory))
        message = self.env.trajectory[0].message
        self.assertEqual(self.action_output, message.content)
        self.assertEqual(TrajectoryState.COMPLETED, self.env.trajectory[0].status)

    @pytest.mark.asyncio
    async def test_output_extraction(self):
        """Test extracting the output from the workflow execution"""
        output_parser = MockLLMOutputParser()
        self.mock_llm.async_generate.return_value = output_parser
        self.workflow.set_node_status('Task1', WorkFlowNodeState.COMPLETED)
        self.workflow.set_node_status('Task2', WorkFlowNodeState.COMPLETED)
        self.workflow.set_node_status('Task3', WorkFlowNodeState.COMPLETED)
        for task_name in ['Task1', 'Task2', 'Task3']:
            message = Message(content='Task output', agent='TestAgent', action='TestAction', prompt='Test prompt', msg_type=MessageType.RESPONSE, wf_goal='Test Workflow', wf_task=task_name)
            self.env.update(message=message, state=TrajectoryState.COMPLETED)
        output = await self.workflow_manager.extract_output(graph=self.workflow, env=self.env)
        self.assertEqual('Test output', output)
        self.mock_llm.async_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_edge_case_handling(self):
        """Test edge case handling in workflow management"""
        self.workflow.set_node_status('Task1', WorkFlowNodeState.COMPLETED)
        self.workflow.set_node_status('Task2', WorkFlowNodeState.COMPLETED)
        self.workflow.set_node_status('Task3', WorkFlowNodeState.COMPLETED)
        task = await self.workflow_manager.schedule_next_task(graph=self.workflow, env=self.env)
        self.assertIsNone(task)
        self.workflow.reset_graph()
        task_no_agents = WorkFlowNode(name='TaskNoAgents', description='Task with no agents', inputs=[], outputs=[], agents=[], status=WorkFlowNodeState.PENDING)
        mock_agent_manager = Mock()
        with self.assertRaises(ValueError):
            await self.workflow_manager.schedule_next_action(goal='Test Goal', task=task_no_agents, agent_manager=mock_agent_manager, env=self.env)

class TestWorkFlowGraph(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.task1 = WorkFlowNode(name='Task1', description='First task', inputs=[Parameter(name='input1', type='string', description='Input 1')], outputs=[Parameter(name='output1', type='string', description='Output 1')], agents=['TestAgent'], status=WorkFlowNodeState.PENDING)
        self.task2 = WorkFlowNode(name='Task2', description='Second task', inputs=[Parameter(name='output1', type='string', description='Output from Task1')], outputs=[Parameter(name='output2', type='string', description='Output 2')], agents=['TestAgent'], status=WorkFlowNodeState.PENDING)
        self.task3 = WorkFlowNode(name='Task3', description='Third task', inputs=[Parameter(name='output2', type='string', description='Output from Task2')], outputs=[Parameter(name='final_output', type='string', description='Final output')], agents=['TestAgent'], status=WorkFlowNodeState.PENDING)
        self.task4 = WorkFlowNode(name='Task4', description='Fourth task (join)', inputs=[Parameter(name='output2', type='string', description='Output from Task2'), Parameter(name='final_output', type='string', description='Output from Task3')], outputs=[Parameter(name='result', type='string', description='Final result')], agents=['TestAgent'], status=WorkFlowNodeState.PENDING)
        self.linear_graph = WorkFlowGraph(goal='Simple Linear Workflow', nodes=[self.task1, self.task2, self.task3], edges=[WorkFlowEdge(source='Task1', target='Task2'), WorkFlowEdge(source='Task2', target='Task3')])
        self.fork_join_graph = WorkFlowGraph(goal='Fork-Join Workflow', nodes=[self.task1, self.task2, self.task3, self.task4], edges=[WorkFlowEdge(source='Task1', target='Task2'), WorkFlowEdge(source='Task2', target='Task3'), WorkFlowEdge(source='Task2', target='Task4'), WorkFlowEdge(source='Task3', target='Task4')])
        self.cycle_graph = WorkFlowGraph(goal='Workflow with Cycle', nodes=[self.task1, self.task2, self.task3], edges=[WorkFlowEdge(source='Task1', target='Task2'), WorkFlowEdge(source='Task2', target='Task3'), WorkFlowEdge(source='Task3', target='Task2')])

    def test_graph_initialization(self):
        """Test that graph is correctly initialized with nodes and edges."""
        self.assertEqual(3, len(self.linear_graph.nodes))
        self.assertEqual(2, len(self.linear_graph.edges))
        self.assertEqual('Task1', self.linear_graph.nodes[0].name)
        self.assertEqual('Task2', self.linear_graph.nodes[1].name)
        self.assertEqual('Task3', self.linear_graph.nodes[2].name)
        edge_pairs = [(edge.source, edge.target) for edge in self.linear_graph.edges]
        self.assertIn(('Task1', 'Task2'), edge_pairs)
        self.assertIn(('Task2', 'Task3'), edge_pairs)

    def test_find_initial_nodes(self):
        """Test finding initial nodes in a workflow."""
        initial_nodes = self.linear_graph.find_initial_nodes()
        self.assertEqual(1, len(initial_nodes))
        self.assertEqual('Task1', initial_nodes[0])
        initial_nodes = self.fork_join_graph.find_initial_nodes()
        self.assertEqual(1, len(initial_nodes))
        self.assertEqual('Task1', initial_nodes[0])
        initial_nodes = self.cycle_graph.find_initial_nodes()
        self.assertEqual(1, len(initial_nodes))

    def test_find_end_nodes(self):
        """Test finding end nodes in a workflow."""
        end_nodes = self.linear_graph.find_end_nodes()
        self.assertEqual(1, len(end_nodes))
        self.assertEqual('Task3', end_nodes[0])
        end_nodes = self.fork_join_graph.find_end_nodes()
        self.assertEqual(1, len(end_nodes))
        self.assertEqual('Task4', end_nodes[0])
        end_nodes = self.cycle_graph.find_end_nodes()
        self.assertEqual(0, len(end_nodes))

    def test_next_execution(self):
        """Test the 'next' method to determine the next executable tasks."""
        next_tasks = self.linear_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual('Task1', next_tasks[0].name)
        self.linear_graph.set_node_status('Task1', WorkFlowNodeState.COMPLETED)
        next_tasks = self.linear_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual('Task2', next_tasks[0].name)
        self.linear_graph.set_node_status('Task2', WorkFlowNodeState.COMPLETED)
        next_tasks = self.linear_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual('Task3', next_tasks[0].name)
        self.linear_graph.set_node_status('Task3', WorkFlowNodeState.COMPLETED)
        next_tasks = self.linear_graph.next()
        self.assertEqual(0, len(next_tasks))

    def test_fork_join_execution(self):
        """Test execution in a fork-join workflow."""
        next_tasks = self.fork_join_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual('Task1', next_tasks[0].name)
        self.fork_join_graph.set_node_status('Task1', WorkFlowNodeState.COMPLETED)
        next_tasks = self.fork_join_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual('Task2', next_tasks[0].name)
        self.fork_join_graph.set_node_status('Task2', WorkFlowNodeState.COMPLETED)
        next_tasks = self.fork_join_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual('Task3', next_tasks[0].name)
        self.fork_join_graph.set_node_status('Task3', WorkFlowNodeState.COMPLETED)
        next_tasks = self.fork_join_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual('Task4', next_tasks[0].name)
        self.fork_join_graph.set_node_status('Task4', WorkFlowNodeState.COMPLETED)
        next_tasks = self.fork_join_graph.next()
        self.assertEqual(0, len(next_tasks))

    def test_cycle_detection(self):
        """Test cycle detection in a workflow."""
        loops = self.cycle_graph._find_all_loops()
        self.assertTrue(loops)
        self.assertTrue(self.cycle_graph.is_loop_start('Task2'))
        self.assertTrue(self.cycle_graph.is_loop_end('Task3'))

    def test_node_status_management(self):
        """Test node status management."""
        self.assertEqual(WorkFlowNodeState.PENDING, self.linear_graph.get_node_status('Task1'))
        self.linear_graph.set_node_status('Task1', WorkFlowNodeState.RUNNING)
        self.assertEqual(WorkFlowNodeState.RUNNING, self.linear_graph.get_node_status('Task1'))
        self.assertTrue(self.linear_graph.running('Task1'))
        self.linear_graph.set_node_status('Task1', WorkFlowNodeState.COMPLETED)
        self.assertEqual(WorkFlowNodeState.COMPLETED, self.linear_graph.get_node_status('Task1'))
        self.assertTrue(self.linear_graph.completed('Task1'))
        self.linear_graph.set_node_status('Task1', WorkFlowNodeState.FAILED)
        self.assertEqual(WorkFlowNodeState.FAILED, self.linear_graph.get_node_status('Task1'))
        self.assertTrue(self.linear_graph.failed('Task1'))

    def test_graph_reset(self):
        """Test resetting the graph to initial state."""
        for node in self.linear_graph.nodes:
            self.linear_graph.set_node_status(node.name, WorkFlowNodeState.COMPLETED)
        for node in self.linear_graph.nodes:
            self.assertEqual(WorkFlowNodeState.COMPLETED, node.status)
        self.linear_graph.reset_graph()
        for node in self.linear_graph.nodes:
            self.assertEqual(WorkFlowNodeState.PENDING, node.status)

    def test_graph_dependency_checking(self):
        """Test checking dependencies between nodes."""
        self.assertFalse(self.linear_graph.are_dependencies_complete('Task2'))
        self.linear_graph.set_node_status('Task1', WorkFlowNodeState.COMPLETED)
        self.assertTrue(self.linear_graph.are_dependencies_complete('Task2'))
        self.assertFalse(self.fork_join_graph.are_dependencies_complete('Task4'))
        self.fork_join_graph.set_node_status('Task1', WorkFlowNodeState.COMPLETED)
        self.fork_join_graph.set_node_status('Task2', WorkFlowNodeState.COMPLETED)
        self.assertFalse(self.fork_join_graph.are_dependencies_complete('Task4'))
        self.fork_join_graph.set_node_status('Task3', WorkFlowNodeState.COMPLETED)
        self.assertTrue(self.fork_join_graph.are_dependencies_complete('Task4'))

class TestModule(unittest.TestCase):

    def test_message(self):
        m1 = Message(content='test_content', agent='agent1', action='action1', next_actions=['action2'], msg_type=MessageType.REQUEST)
        time.sleep(5)
        m2 = Message(content=ToyContent(content='test_content2'), agent='agent2', action='action3', msg_type=MessageType.RESPONSE)
        time.sleep(5)
        m3 = Message(content='test_content', agent='agent1', action='action1', next_actions=['action2'], msg_type=MessageType.REQUEST)
        self.assertTrue(m3 != m1)
        m3_message_id = m3.message_id
        m3.message_id = m1.message_id
        self.assertTrue(m1 == m3)
        m3.message_id = m3_message_id
        message_str = str(m2)
        self.assertTrue('Content: test_content2' in message_str)
        sorted_message_based_on_timestamp = Message.sort([m3, m2, m1])
        self.assertEqual(sorted_message_based_on_timestamp[0].message_id, m1.message_id)
        self.assertEqual(sorted_message_based_on_timestamp[1].message_id, m2.message_id)
        self.assertEqual(sorted_message_based_on_timestamp[2].message_id, m3.message_id)
        merged_message = Message.merge([[m3], [m1, m2]], sort=True)
        self.assertEqual(merged_message[0].message_id, m1.message_id)
        self.assertEqual(merged_message[1].message_id, m2.message_id)
        self.assertEqual(merged_message[2].message_id, m3.message_id)

