# Cluster 20

class GymAgent(AgentABC):

    def __init__(self, model: str, system_prompt: str, task: Any, agent_idx: Optional[int]=None, observation_formatter: Optional[BasicObservationFormatter]=None, system_prompt_formatter: Optional[SystemPromptFormatter]=None, api_key_config_file: Optional[str]=None, *args, **kwargs):
        instructions = self._get_instructions(system_prompt, task, agent_idx)
        super().__init__(model, instructions, *args, **kwargs)
        self.task = task
        self.api_factory = APIFactory(model, api_key_config_file=api_key_config_file)
        self.observation_formatter = observation_formatter or BasicObservationFormatter()
        self.conversation = Conversation()
        self.formatter = RecursiveReportFormatter(chunk_size=16, llm_call=self.api_factory.acall, cache_dir='.fle/summary_cache')
        self.generation_params = GenerationParameters(n=1, max_tokens=4096, model=model)
        self.system_prompt_formatter = system_prompt_formatter

    def _get_instructions(self, system_prompt: str, task: TaskABC, agent_idx: Optional[int]=None):
        agent_instructions = ''
        if agent_idx is not None and task.get_agent_instructions(agent_idx) is not None:
            player_idx = agent_idx + 1
            agent_instructions = f'### Specific Instructions for Agent {player_idx}\n{task.get_agent_instructions(agent_idx)}\n\n'
        instructions = GYM_AGENT_INSTRUCTIONS.format(system_prompt=system_prompt, goal_description=task.goal_description, agent_instructions=agent_instructions)
        return instructions.rstrip()

    def reset(self, conversation: Conversation):
        self.conversation = copy.deepcopy(conversation)
        if self.conversation.messages[0].role != 'system':
            self.conversation.set_system_message(self.system_prompt)

    async def update_conversation(self, observation: Observation, previous_program: Optional[Program]=None):
        if previous_program:
            formatted_program = f'```python\n{previous_program.code}\n```'
            self.conversation.add_agent_message(formatted_program)
        formatted_obs = self.observation_formatter.format(observation).raw_str
        self.conversation.add_user_message(formatted_obs)
        self.conversation = await self.formatter.format_conversation(self.conversation)

    async def generate_policy(self, observation: Optional[Observation]=None, previous_program: Optional[Program]=None) -> Policy:
        """Generate a policy from the current observation.

        Returns:
            Policy if generation was successful, None otherwise
        """
        if observation:
            await self.update_conversation(observation, previous_program)
        try:
            model_response = await self.api_factory.acall(messages=self.formatter.to_llm_messages(self.conversation), n_samples=1, temperature=self.generation_params.temperature, max_tokens=self.generation_params.max_tokens, model=self.generation_params.model)
            policy = parse_response(model_response)
            if not policy:
                raise Exception('Policy not valid Python. Skipping.')
            policy.input_conversation = self.conversation
            return policy
        except Exception as e:
            print(f'Policy generation failed: {str(e)}')
            return None

    async def step(self, conversation: Conversation) -> Policy:
        pass

    async def end(self, completion: CompletionResult):
        """Cleanup when the trajectory ends"""
        pass

class RecursiveFormatter(ConversationFormatter):
    """
    Formatter that maintains a fixed context window through hierarchical summarization.
    Recursively summarizes from left to right, incorporating newer messages into the summary.
    """

    def __init__(self, chunk_size: int=16, api_factory: Optional[APIFactory]=None, cache_dir: str='.conversation_cache', summary_instructions: str=DEFAULT_INSTRUCTIONS, truncate_entity_data: bool=True, summarize_history: bool=True):
        """

        @param chunk_size:
        @param api_factory:
        @param cache_dir:
        @param summary_instructions:
        @param truncate_entity_data: Whether we should truncate historical (stale) entity observations when summarizing.
        """
        self.chunk_size = chunk_size
        self.api_factory = api_factory
        self.cache_dir = cache_dir
        self.summary_instructions = summary_instructions
        self.truncate_entity_data = truncate_entity_data
        self.entity_data_pattern = re.compile('(?:,|:) \\[((.|[\\n])+)\\]\\)')
        self.summarize_history = summarize_history
        os.makedirs(cache_dir, exist_ok=True)

    def _get_chunk_hash(self, messages: List[Message]) -> str:
        """Generate a deterministic hash for a chunk of messages."""
        chunk_content = json.dumps([{'role': msg.role, 'content': msg.content, 'metadata': msg.metadata} for msg in messages], sort_keys=True)
        return hashlib.sha256(chunk_content.encode()).hexdigest()

    def _get_cache_path(self, chunk_hash: str) -> str:
        """Get the file path for a cached summary."""
        return os.path.join(self.cache_dir, f'{chunk_hash}.json')

    def _load_cached_summary(self, chunk_hash: str) -> Optional[Message]:
        """Load a cached summary if it exists."""
        cache_path = self._get_cache_path(chunk_hash)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                return Message(role='assistant', content=data['content'], metadata={'summarized': True, 'summary_range': data['summary_range']})
            except Exception as e:
                print(f'Error loading cached summary: {e}')
                return None
        return None

    def _save_summary_cache(self, chunk_hash: str, summary: Message):
        """Save a generated summary to the cache."""
        cache_path = self._get_cache_path(chunk_hash)
        try:
            with open(cache_path, 'w') as f:
                json.dump({'content': summary.content, 'summary_range': summary.metadata['summary_range']}, f)
        except Exception as e:
            print(f'Error saving summary cache: {e}')

    def _truncate_entity_data(self, message: Message, is_recent: bool=False) -> Message:
        """
        Truncate entity data in message content if enabled and message is not recent.
        Returns a new Message instance with modified content if truncation occurred.
        """
        if not self.truncate_entity_data or is_recent or message.role in ('assistant', 'system'):
            return message
        if isinstance(message.content, str):
            new_content = self.entity_data_pattern.sub(': <STALE_ENTITY_DATA_OMITTED/>', message.content)
            if new_content != message.content:
                return Message(role=message.role, content=new_content, metadata=message.metadata)
        return message

    async def _generate_summary(self, messages: List[Message], start_idx: int, end_idx: int, system_message: Message) -> Message:
        """Generate a summary of messages using the LLM."""
        if not self.api_factory:
            raise ValueError('LLM factory required for summary generation')
        summary_prompt = [{'role': 'system', 'content': self.summary_instructions}]
        for msg in messages:
            summary_prompt.append({'role': msg.role, 'content': msg.content})
        response = await self.api_factory.acall(messages=summary_prompt, max_tokens=1024, temperature=0.3)
        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
        else:
            content = response.content[0].text
        return Message(role='user', content=content, metadata={'summarized': True, 'summary_range': f'[{start_idx}-{end_idx}]'})

    async def _summarize_chunk(self, messages: List[Message], start_idx: int, end_idx: int, system_message: Message) -> Message:
        """Summarize a chunk of messages, using cache if available."""
        chunk_hash = self._get_chunk_hash(messages)
        cached_summary = self._load_cached_summary(chunk_hash)
        if cached_summary:
            return cached_summary
        summary = await self._generate_summary(messages, start_idx, end_idx, system_message)
        self._save_summary_cache(chunk_hash, summary)
        return summary

    async def _recursive_summarize_left(self, messages: List[Message], system_message: Message) -> Message:
        """
        Recursively summarize messages from left to right:
        1. Take first chunk_size messages and summarize
        2. Take that summary and next chunk of messages, summarize together
        3. Continue until all messages are incorporated
        """
        if len(messages) <= self.chunk_size:
            return await self._summarize_chunk(messages, 1, len(messages), system_message)
        left_chunk = messages[:self.chunk_size]
        current_summary = await self._summarize_chunk(left_chunk, 1, self.chunk_size, system_message)
        remaining = messages[self.chunk_size:]
        current_end = self.chunk_size
        while remaining:
            next_chunk_size = min(len(remaining), self.chunk_size)
            next_chunk = remaining[:next_chunk_size]
            messages_to_summarize = [current_summary] + next_chunk
            current_summary = await self._summarize_chunk(messages_to_summarize, 1, current_end + next_chunk_size, system_message)
            remaining = remaining[next_chunk_size:]
            current_end += next_chunk_size
        return current_summary

    async def format_conversation(self, conversation: Conversation) -> List[Message]:
        """
        Format conversation by recursively summarizing historical messages from left to right.
        Returns [system_message (if present), historical_summary, recent_messages].
        """
        messages = conversation.messages
        if len(messages) <= self.chunk_size:
            return [self._truncate_entity_data(msg, is_recent=i >= len(messages) - 1) for i, msg in enumerate(messages)]
        system_message = None
        if messages[0].role == 'system':
            system_message = messages[0]
            messages = messages[1:]
        recent_messages = messages[-self.chunk_size:]
        if self.summarize_history:
            historical_messages = messages[:-self.chunk_size]
        else:
            historical_messages = []
        if historical_messages:
            historical_summary = await self._recursive_summarize_left(historical_messages, system_message)
            formatted = [historical_summary] + [self._truncate_entity_data(msg, is_recent=i >= len(recent_messages) - 1) for i, msg in enumerate(recent_messages)]
        else:
            formatted = [self._truncate_entity_data(msg, is_recent=i >= len(recent_messages) - 1) for i, msg in enumerate(recent_messages)]
        if system_message:
            formatted = [system_message] + formatted
        return formatted

    def format_message(self, message: Message) -> Message:
        """Format a single message - apply entity data truncation if enabled."""
        return self._truncate_entity_data(message, is_recent=True)

class StructurePreservingFormatter(ConversationFormatter):
    """
    Formatter that preserves program structure through comments while reducing token usage.
    It replaces all code not in the most recent history with `<LINE X CUT/>` tags.
    """

    def __init__(self, planning=True):
        self.code_processor = CodeProcessor()
        self.planning = planning

    def format_message(self, message: Message, should_format: bool=True) -> Optional[Message]:
        if message.role == 'system':
            return Message(role='system', content=message.content)
        elif message.role == 'assistant':
            if should_format:
                content = self.code_processor.summarize_code_block(message.content)
                return Message(role='assistant', content=content, metadata={'summarized': True})
            else:
                return Message(role='assistant', content=message.content, metadata={'summarized': False})
        elif message.role == 'user':
            content = message.content
            try:
                if 'Execution result:' in content:
                    result = content.split('Execution result:')[1].split('Updated state:')[0]
                    return Message(role='user', content=result.strip())
                else:
                    return Message(role='user', content=content.strip())
            except Exception as e:
                print(f'Error formatting user message: {str(e)}')
                return None
        return None

    def format_conversation(self, conversation: Conversation, namespace: FactorioNamespace) -> List[Message]:
        formatted = []
        if conversation.messages and conversation.messages[0].role == 'system':
            formatted.append(self.format_message(conversation.messages[0]))
            messages = conversation.messages[1:]
        else:
            messages = conversation.messages
        last_message_role = messages[-1].role
        for i, msg in enumerate(messages):
            if last_message_role == 'assistant':
                formatted_msg = self.format_message(msg, should_format=i != len(messages) - 1)
            elif last_message_role == 'user':
                formatted_msg = self.format_message(msg, should_format=i != len(messages) - 2)
            if formatted_msg:
                formatted.append(formatted_msg)
        return formatted

class RecursiveReportFormatter(ConversationFormatter):
    """
    Formatter that maintains a fixed context window through hierarchical summarization.
    Recursively summarizes from left to right, incorporating newer messages into the summary.
    """

    def __init__(self, chunk_size: int=16, llm_call: Callable[[LLMParams], Awaitable[Any]]=None, cache_dir: str='.conversation_cache', truncate_entity_data: bool=True, summarize_history: bool=True, max_chars: int=200000):
        """
        @param chunk_size:
        @param api_factory:
        @param cache_dir:
        @param summary_instructions:
        @param truncate_entity_data: Whether we should truncate historical (stale) entity observations when summarizing.
        """
        self.llm_call = llm_call
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir
        self.summary_instructions = DEFAULT_INSTRUCTIONS
        self.truncate_entity_data = truncate_entity_data
        self.entity_data_pattern = re.compile('\\[([\\n\\t\\\\\\t\\n].+)+[\\"\\]]')
        self.summarize_history = summarize_history
        self.max_chars = max_chars
        os.makedirs(cache_dir, exist_ok=True)

    def _get_conversation_length(self, messages: List[Message]) -> int:
        """Calculate total character length of all messages in conversation."""
        return sum((len(msg.content) for msg in messages))

    def _trim_conversation_to_limit(self, messages: List[Message]) -> List[Message]:
        """
        Trim conversation to stay within character limit while preserving system message.
        Removes oldest messages first (after system message).
        """
        if not messages:
            return messages
        system_message = None
        working_messages = messages.copy()
        if working_messages[0].role == 'system':
            system_message = working_messages.pop(0)
        current_length = self._get_conversation_length(working_messages)
        if system_message:
            current_length += len(system_message.content)
        if current_length <= self.max_chars:
            return messages
        while working_messages and current_length > self.max_chars:
            removed_message = working_messages.pop(0)
            current_length -= len(removed_message.content)
        final_messages = working_messages
        if system_message:
            final_messages.insert(0, system_message)
        return final_messages

    def _get_chunk_hash(self, messages: List[Message]) -> str:
        """Generate a deterministic hash for a chunk of messages."""
        chunk_content = json.dumps([{'role': msg.role, 'content': msg.content, 'metadata': msg.metadata} for msg in messages], sort_keys=True)
        return hashlib.sha256(chunk_content.encode()).hexdigest()

    def _get_cache_path(self, chunk_hash: str) -> str:
        """Get the file path for a cached summary."""
        return os.path.join(self.cache_dir, f'{chunk_hash}.json')

    def _load_cached_summary(self, chunk_hash: str) -> Optional[str]:
        """Load a cached summary if it exists."""
        cache_path = self._get_cache_path(chunk_hash)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                return data['content']
            except Exception as e:
                print(f'Error loading cached summary: {e}')
                return None
        return None

    def _save_summary_cache(self, chunk_hash: str, summary: str):
        """Save a generated summary to the cache."""
        cache_path = self._get_cache_path(chunk_hash)
        try:
            with open(cache_path, 'w') as f:
                json.dump({'content': summary}, f)
        except Exception as e:
            print(f'Error saving summary cache: {e}')

    def _truncate_entity_data(self, message: Message, is_recent: bool=False, message_index=0) -> Message:
        """
        Truncate entity data in message content if enabled and message is not recent.
        Returns a new Message instance with modified content if truncation occurred.
        """
        if not self.truncate_entity_data or message.role in ('assistant', 'system'):
            return message
        if isinstance(message.content, str):
            if is_recent:
                new_content = message.content
            else:
                new_content = self.entity_data_pattern.sub(': <STALE_ENTITY_DATA_OMITTED/>', message.content)
            if new_content != message.content:
                pass
            new_content = f'Step {message_index} execution log\n{new_content}'
            return Message(role=message.role, content=new_content, metadata=message.metadata)
        return message

    async def _generate_summary(self, messages: List[Message], previous_report: str, last_summary_step: int) -> Message:
        """Generate a summary of messages using the LLM."""
        if not self.llm_call:
            raise ValueError('LLM factory required for summary generation')
        summary_prompt = [{'role': 'system', 'content': self.summary_instructions}]
        if last_summary_step == 0:
            steps = 'These are the first steps so there is no historical report to summarize.\n\n'
            steps += f'Here are the first {int(self.chunk_size / 2)} execution steps of the agent:\n\n'
        else:
            steps = f'Here is the report from step 0 until step {int(last_summary_step / 2) - 1}\n\n{previous_report}\n\n'
            steps += f'Here are the next {int(self.chunk_size / 2)} execution steps of the agent:\n\n'
        for msg in messages:
            if msg.role == 'user':
                steps += msg.content + '\n\n'
        summary_prompt.append({'role': 'user', 'content': steps})
        response = await self.llm_call(messages=summary_prompt, max_tokens=1024, temperature=0.3)
        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
        else:
            content = response.content[0].text
        return content

    def _get_chunk_hash(self, messages: List[Message]) -> str:
        """Generate a deterministic hash for a chunk of messages."""
        chunk_content = json.dumps([{'content': msg.content} for msg in messages if msg.role == 'user'], sort_keys=True)
        return hashlib.sha256(chunk_content.encode()).hexdigest()

    async def format_conversation(self, conversation: Conversation, namespace: Optional[FactorioNamespace]=None) -> Conversation:
        """
        Format conversation by recursively summarizing historical messages from left to right.
        Returns [system_message (if present), historical_summary, recent_messages].
        """
        messages = conversation.messages
        total_length = len(messages)
        if len(messages) <= self.chunk_size:
            messages = [self._truncate_entity_data(msg, is_recent=i >= len(messages) - 1, message_index=int((total_length - len(messages)) / 2) + int(i / 2)) for i, msg in enumerate(messages)]
            messages = self._trim_conversation_to_limit(messages)
            return Conversation(messages=messages)
        system_message = None
        if messages[0].role == 'system':
            system_message = messages[0]
            messages = messages[1:]
        function_definitions = namespace.get_functions() if namespace else []
        if function_definitions:
            system_message.content += '# Your utility functions:\n\n' + '\n\n'.join([str(f) for f in function_definitions])
        new_messages = copy.deepcopy(messages[-self.chunk_size:])
        new_formatted_messages = [self._truncate_entity_data(msg, is_recent=i >= len(new_messages) - 1, message_index=int((total_length - len(new_messages)) / 2) + int(i / 2)) for i, msg in enumerate(new_messages)]
        if self.summarize_history:
            nr_of_messages = total_length - 1
            if nr_of_messages % self.chunk_size == 0:
                nr_of_messages_in_report = nr_of_messages - self.chunk_size
                messages_in_report = messages[:nr_of_messages_in_report]
                if nr_of_messages_in_report > 0:
                    messages_hash = self._get_chunk_hash(messages_in_report)
                    report = self._load_cached_summary(messages_hash)
                else:
                    report = ''
                historical_report = await self._generate_summary(new_formatted_messages, previous_report=report, last_summary_step=nr_of_messages_in_report)
                new_hash = self._get_chunk_hash(messages)
                self._save_summary_cache(new_hash, historical_report)
                nr_of_messages_in_report = nr_of_messages
            else:
                nr_of_messages_in_report = len(messages) // self.chunk_size * self.chunk_size
                messages_in_report = messages[:nr_of_messages_in_report]
                messages_hash = self._get_chunk_hash(messages_in_report)
                historical_report = self._load_cached_summary(messages_hash)
            if system_message:
                sys = system_message.content.split('Historical report of actions, observations, variables and functions until step')[0].strip()
                system_message.content = sys + f'\n\nHistorical report of actions, observations, variables and functions until step {int(nr_of_messages_in_report / 2) - 1}\n\n{historical_report}\n\n'
                new_formatted_messages = [system_message] + new_formatted_messages
        elif system_message:
            new_formatted_messages = [system_message] + new_formatted_messages
        new_formatted_messages = self._trim_conversation_to_limit(new_formatted_messages)
        return Conversation(messages=new_formatted_messages)

    def format_message(self, message: Message) -> Message:
        """Format a single message - apply entity data truncation if enabled."""
        return self._truncate_entity_data(message, is_recent=True)

class Conversation(BaseModel):
    """Tracks dialogue between LLM and Factorio"""
    messages: List[Message] = Field(default_factory=list)

    @classmethod
    def parse_raw(cls, data: Dict[str, Any]) -> 'Conversation':
        messages = [Message(**msg) if isinstance(msg, dict) else msg for msg in data['messages']]
        return cls(messages=messages)

    def set_system_message(self, message: str):
        if self.messages and self.messages[0].role == 'system':
            self.messages[0] = Message(role='system', content=message)
        else:
            self.messages.insert(0, Message(role='system', content=message))

    def add_agent_message(self, message: str, **kwargs):
        self.messages.append(Message(role='assistant', content=message, metadata=kwargs))

    def add_user_message(self, message: str, **kwargs):
        self.messages.append(Message(role='user', content=message, metadata=kwargs))

    def add_result(self, program: str, response: str, **kwargs):
        """Add program execution result to conversation"""
        self.add_agent_message(program, **kwargs)
        self.add_user_message(response, **kwargs)

class ParallelPlanningMCTS:

    def __init__(self, instances: List[FactorioInstance], db_client: DBClient, api_factory: Any, config: ParallelMCTSConfig, version=26, version_description='', formatter: ConversationFormatter=StructurePreservingFormatter()):
        """
        Initialize parallel planning MCTS

        Args:
            instances: List of Factorio instances to distribute
            db_client: Database client
            api_factory: Factory for creating language models
            config: Configuration parameters including model paths and prompts
        """
        self.console = Console()
        self.config = config
        self.sampler = config.sampler
        self.db_client = db_client
        self.llm = api_factory
        self.version = version
        self.version_description = version_description
        self.formatter = formatter
        self._validate_instance_count(len(instances), config.n_parallel)
        instances_per_group = floor(len(instances) / config.n_parallel)
        self.logger = GroupedFactorioLogger(n_groups=config.n_parallel, instances_per_group=instances_per_group)
        self.logger.start()
        self.max_steps_per_objective = config.max_steps_per_objective
        self.number_of_steps_for_judge = config.number_of_steps_for_judge
        self.planning_model = config.mcts_kwargs['planning_model']
        self.executor_model = config.mcts_kwargs['executor_model']
        self.objective_model = config.mcts_kwargs['objective_model']
        self.step_executor_prompt_path = config.mcts_kwargs['step_executor_prompt_path']
        self.step_generator_prompt_path = config.mcts_kwargs['step_generator_prompt_path']
        self.step_judge_prompt_path = config.mcts_kwargs['step_judge_prompt_path']
        self.example_plan_prompt_path = config.mcts_kwargs['example_plan_prompt_path']
        self.step_executor_system_prompt, self.step_executor_user_prompt = self.read_in_prompts(config.mcts_kwargs['step_executor_prompt_path'])
        self.step_generator_system_prompt, self.step_generator_user_prompt = self.read_in_prompts(config.mcts_kwargs['step_generator_prompt_path'])
        self.step_judge_system_prompt, self.step_judge_user_prompt = self.read_in_prompts(config.mcts_kwargs['step_judge_prompt_path'])
        self.example_plan_system_prompt, self.example_plan_user_prompt = self.read_in_prompts(config.mcts_kwargs['example_plan_prompt_path'])
        self.instance_groups = self._create_instance_groups(instances)
        self.api_description = self.instance_groups[0].evaluator.instances[0].get_system_prompt()
        self.step_executor_system_prompt = self.step_executor_system_prompt.format(schema=self.api_description)
        self.example_plan_system_prompt = self.example_plan_system_prompt.format(schema=self.api_description)

    def read_in_prompts(self, path):
        system_prompt_path = os.path.join(path, 'system_prompt.md')
        user_prompt_path = os.path.join(path, 'user_prompt.md')
        with open(system_prompt_path, 'r') as f:
            system_prompt = f.read()
        with open(user_prompt_path, 'r') as f:
            user_prompt = f.read()
        return (system_prompt, user_prompt)

    def _create_instance_groups(self, instances: List['FactorioInstance']) -> List[PlanningGroup]:
        """Create instance groups for parallel execution"""
        instances_per_group = floor(len(instances) / self.config.n_parallel)
        groups = []
        for group_id in range(self.config.n_parallel):
            start_idx = group_id * instances_per_group
            end_idx = start_idx + instances_per_group
            group_instances = instances[start_idx:end_idx]
            active_instances = group_instances[:-1]
            holdout_instance = group_instances[-1]
            evaluator = Evaluator(db_client=self.db_client, instances=group_instances, value_accrual_time=3, logger=self.logger, error_penalty=self.config.mcts_kwargs['error_penalty'])
            mcts = self.config.mcts_class(api_factory=self.llm, db_client=self.db_client, sampler=self.sampler, evaluator=evaluator, **self.config.mcts_kwargs)
            groups.append(PlanningGroup(group_id=group_id, mcts=mcts, evaluator=evaluator, active_instances=active_instances, holdout_instance=holdout_instance))
        return groups

    def _validate_instance_count(self, total_instances: int, n_parallel: int):
        min_required = n_parallel * 2
        if total_instances < min_required:
            raise ValueError(f'Need at least {min_required} instances for {n_parallel} parallel searches (got {total_instances})')
        instances_per_group = floor(total_instances / n_parallel)
        if instances_per_group < 2:
            raise ValueError(f'Not enough instances per group (need at least 2, got {instances_per_group})')

    async def search(self, n_iterations: int, skip_failures: bool=False):
        """
        Run truly parallel MCTS search across all groups

        Args:
            n_iterations: Number of iterations to run
            skip_failures: Whether to skip failed program generations
        """
        try:
            search_tasks = []
            for group in self.instance_groups:
                task = asyncio.create_task(self._run_group_search(group=group, n_iterations=n_iterations, skip_failures=skip_failures))
                search_tasks.append(task)
            await asyncio.gather(*search_tasks)
        except Exception as e:
            print(f'Error during parallel search: {str(e)}')
            raise
        finally:
            self.cleanup()

    async def _run_group_search(self, group: PlanningGroup, n_iterations: int, skip_failures: bool=False):
        """Run parallel planning search across all groups"""
        try:
            for iteration in range(n_iterations):
                parent = await self.sampler.sample_parent(version=self.version)
                group.evaluator.set_status('Generating tasks')
                tasks, start_state = await self._get_tasks(group, parent)
                group.evaluator.set_status('Generating plans')
                group.plans = await self.generate_plans(tasks)
                saved_step_ids = []
                for step_idx in range(self.max_steps_per_objective):
                    if step_idx == 0:
                        for instance_id, instance in enumerate(group.evaluator.instances):
                            instance.reset(start_state)
                    plans = await self._process_group_step(group, step_idx, skip_failures, start_state, parent)
                    for plan in plans:
                        try:
                            step_to_save = plan.steps[-1]
                            if step_to_save.program.id not in saved_step_ids:
                                await self.save_step(plan, step_to_save)
                                saved_step_ids.append(step_to_save.program.id)
                        except Exception:
                            print('Could not save step - possibly missing (in case of skipping errors)')
                    group.evaluator.logger.update_progress()
        except Exception as e:
            print(f'Error during parallel search: {str(e)}')
            raise
        finally:
            self.cleanup()

    async def _process_group_step(self, group: PlanningGroup, step_idx: int, skip_failures: bool, start_state: GameState, parent: Program) -> List[PlanOutput]:
        """Process a single step for a group"""
        try:
            group.evaluator.set_status(f'Getting candidates for step {step_idx}')
            group.plans = await self.generate_next_step_candidates(group)
            group.evaluator.set_status(f'Judging candidates for step {step_idx}')
            group.plans = await self.get_next_step_with_judge(group)
            group.evaluator.set_status(f'Generating programs for step {step_idx}')
            group.plans = await self.get_next_step_programs(group)
            eval_futures = []
            completed_plans = []
            for instance_id, (instance, plan) in enumerate(zip(group.active_instances, group.plans.values())):
                if plan.success:
                    if plan.steps[-1].program is None:
                        plan.steps[-1].program = self._create_output_completed_program(plan, parent.id if parent else None)
                    completed_plans.append(plan)
                    continue
                latest_program = plan.steps[-1].program
                group.evaluator.logger.update_instance(instance_id, program_id=latest_program.id, status='evaluating')
                parent_id = parent.id if parent else None
                eval_futures.append(self._process_last_step(plan=plan, start_state=start_state, group=group, instance_id=instance_id, parent_id=parent_id, skip_failures=skip_failures))
            return await asyncio.gather(*eval_futures) + completed_plans
        except Exception as e:
            print(f'Error in group {group.group_id}, step {step_idx}: {str(e)}')
            raise

    def cleanup(self):
        """Clean up resources"""
        self.logger.stop()
        for group in self.instance_groups:
            if hasattr(group.evaluator, 'logger'):
                group.evaluator.logger.stop()

    def get_group_metrics(self, group_id: int) -> Dict[str, Any]:
        """Get metrics for a specific group"""
        if 0 <= group_id < len(self.instance_groups):
            group = self.instance_groups[group_id]
            return {'active_instances': len(group.active_instances), 'total_programs': sum((inst.total_programs for inst in group.evaluator.logger.groups[group_id].instances.values())), 'error_count': sum((inst.error_count for inst in group.evaluator.logger.groups[group_id].instances.values()))}
        return {}

    async def _evaluate_step(self, step: Step, start_state: GameState, group: PlanningGroup, instance_id: int, parent_id) -> Tuple[Step, float, List]:
        """Modified to work with instance groups"""
        group.holdout_instance.reset(start_state)
        entity_list = []
        try:
            instance = group.active_instances[instance_id]
            step.start_state = GameState.from_instance(instance)
            group.evaluator.logger.update_instance(instance_id, status='executing')
            reward, state, response, entities, achievements, ticks = await group.evaluator._evaluate_single(instance_id, step.program, instance)
            entity_list.append(entities)
            step.end_state = state
            step.reward = reward
        except Exception as e:
            print(f'Error during evaluation in group {group.group_id}, instance {instance_id}: {e}')
            raise e
        step.program.value = step.reward
        step.program.raw_reward = step.reward
        step.program.holdout_value = step.reward
        step.program.state = step.end_state
        step.program.response = response
        step.program.parent_id = parent_id
        step.program.achievements = achievements
        return (step, step.reward, entity_list)

    async def save_step(self, plan: PlanOutput, step: Step):
        candidate_step_meta = []
        if step.judge_step_str == '':
            for candidate_step in step.candidate_language_outputs:
                try:
                    messages = candidate_step.conversation.model_dump()['messages']
                except:
                    messages = candidate_step.conversation.dict()['messages']
                output = candidate_step.response
                candidate_step_meta.append({'output': output, 'messages': messages})
            step.program.meta['candidate_steps'] = candidate_step_meta
            await self.db_client.create_program(step.program)
            return
        objective = plan.task.task
        initial_plan = plan.initial_plan.initial_plan
        parent_id = None
        for current_step, next_step in zip(plan.steps[:-1], plan.steps[1:]):
            if next_step.final_step == step.final_step:
                parent_id = current_step.program.id
        for candidate_step in step.candidate_language_outputs:
            try:
                messages = candidate_step.conversation.model_dump()['messages']
            except:
                messages = candidate_step.conversation.dict()['messages']
            output = candidate_step.response
            candidate_step_meta.append({'output': output, 'messages': messages})
            mining_setup = candidate_step.meta['mining_setup']
            starting_inventory = candidate_step.meta['starting_inventory']
        try:
            judge_messages = step.judge_language_output_step.conversation.model_dump()['messages']
        except:
            judge_messages = step.judge_language_output_step.conversation.dict()['messages']
        judge_output = step.judge_step_str
        executor_step = step.final_step
        meta = {'objective': objective, 'initial_plan': initial_plan, 'candidate_steps': candidate_step_meta, 'judge_step': {'messages': judge_messages, 'output': judge_output}, 'executor_step': {'input_step': executor_step, 'natural_language_plan': step.program.meta['text_response'], 'model': step.program.meta['model']}, 'mining_setup': mining_setup, 'starting_inventory': starting_inventory, 'final_output': plan.final_output}
        program = step.program
        program.meta = meta
        program.parent_id = parent_id
        await self.db_client.create_program(program)
        parent_id = program.id

    async def save_plan(self, plan: PlanOutput):
        for step in plan.steps:
            await self.save_step(plan, step)

    async def _process_last_step(self, plan: PlanOutput, start_state: GameState, group: PlanningGroup, instance_id: int, parent_id: Optional[int], skip_failures: bool) -> PlanOutput:
        try:
            step_to_process = plan.steps[-1]
            step_to_process, _, entity_list = await self._evaluate_step(step_to_process, start_state, group, instance_id, parent_id)
            if skip_failures and 'error' in step_to_process.program.response.lower():
                raise Exception('Found error in response. Skipping step.')
            plan.steps[-1] = step_to_process
            log_str = f'Step {len(plan.steps)}: {step_to_process.final_step}\n{step_to_process.program.response}'
            plan.logs.append(log_str)
            return plan
        except Exception as e:
            print(f'Failed to evaluate program on instance {instance_id}: {str(e)}')
            plan.steps.pop()
            return plan

    def _create_output_completed_program(self, plan: PlanOutput, parent_id: Optional[int]) -> PlanOutput:
        objective = f"'{plan.task.task}'"
        python_code = f"print('Objective {objective} has been completed. Now lets prepare the next objective.')"
        program_parent_id = plan.steps[-2].program.id if len(plan.steps) > 1 else parent_id
        program = Program(id=hash((python_code, plan.task.task, program_parent_id)), code=python_code, conversation=Conversation(messages=[]), parent_id=program_parent_id, version=self.version, version_description=self.version_description, meta={'objective': plan.task.task})
        return program

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_natural_language_batch(self, conversation: Conversation, generation_params: GenerationParameters, meta: dict) -> List[LanguageOutput]:
        """Generate multiple programs in a single API call using 'n' parameter"""
        formatted_messages = self.formatter.to_llm_messages(self.formatter.format_conversation(conversation))
        if 'claude' in generation_params.model:
            assert generation_params.n == 1, 'Number of samples must be 1 for Claude'
        try:
            response = await self.llm.acall(messages=formatted_messages, n_samples=generation_params.n, temperature=generation_params.temperature, max_tokens=generation_params.max_tokens, logit_bias=generation_params.logit_bias, stop_sequences=generation_params.stop_sequences, model=generation_params.model)
            outputs = []
            try:
                messages = conversation.model_dump()['messages']
            except:
                messages = conversation.dict()['messages']
            if 'claude' in generation_params.model:
                str_output = response.content[0].text
                outputs.append(LanguageOutput(id=hash((str_output, json.dumps(messages))), response=str_output, conversation=conversation, token_usage=response.usage.output_tokens + response.usage.input_tokens if hasattr(response, 'usage') else None, completion_token_usage=response.usage.output_tokens if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.input_tokens if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta=meta))
            else:
                for choice in response.choices:
                    str_output = choice.message.content
                    outputs.append(LanguageOutput(id=hash((str_output, json.dumps(messages))), response=str_output, conversation=conversation, token_usage=response.usage.total_tokens // generation_params.n if hasattr(response, 'usage') else None, completion_token_usage=response.usage.completion_tokens // generation_params.n if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.prompt_tokens // generation_params.n if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta=meta))
            return outputs
        except Exception as e:
            print(f'Batch program generation failed: {str(e)}')
            return []

    def get_inventory_dict(self, inventory):
        inventory_dict = {}
        for item in inventory:
            if isinstance(item, tuple):
                inventory_dict[item[0]] = inventory[item]
            else:
                inventory_dict[item] = inventory[item]
        return inventory_dict

    async def _get_tasks(self, group: PlanningGroup, parent: Program=None) -> Tuple[List[TaskOutput], GameState]:
        """Modified to work with instance groups"""
        start_state = parent.state if parent else self.config.initial_state
        first_instance = group.active_instances[0]
        first_instance.reset(start_state)
        mining_setup = get_mining_setup(first_instance)
        starting_inventory = first_instance.inspect_inventory()
        conversation = Conversation(messages=[Message(role='system', content=self.config.system_prompt), Message(role='user', content=f"Your starting inventory is {starting_inventory}. {mining_setup}. Create an incrementally useful task that you can carry out in the current game, in order to grow your factory's _automatic_ throughput.")])
        generation_params = GenerationParameters(model=self.config.mcts_kwargs['objective_model'], n=len(group.active_instances), stop_sequences=['\n'])
        inventory_dict = self.get_inventory_dict(starting_inventory)
        game_state_str = GameState.from_instance(first_instance).entities
        tasks = await self._generate_natural_language_batch(conversation, generation_params, meta={'type': 'objective_generation', 'inventory': inventory_dict, 'mining_setup': mining_setup, 'game_state': game_state_str, 'group_id': group.group_id})
        task_outputs = []
        for task in tasks:
            task_string = task.response.split('\n')[0].strip()
            task_string = task_string.lower().replace('sure! the task i will carry out is', '').strip()
            if '.' in task_string:
                task_string = task_string.split('.')[0]
            task_outputs.append(TaskOutput(task=task_string, language_output=task))
        return (task_outputs, start_state)

    async def generate_plans(self, task_outputs: List[TaskOutput]) -> List[InitialPlanOutput]:
        generation_params = GenerationParameters(model=self.executor_model, stop_sequences=['```'], logits={'7032': -100})
        conversations_to_process = [Conversation(messages=[Message(role='system', content=self.example_plan_system_prompt), Message(role='user', content=self.example_plan_user_prompt.format(task=task_output.task))]) for task_output in task_outputs]
        initial_plans = [asyncio.ensure_future(self._generate_natural_language_batch(conversation, generation_params, meta={'type': 'initial_plan_generation'})) for conversation in conversations_to_process]
        responses = await asyncio.gather(*initial_plans)
        plan_outputs = {}
        for idx, response in enumerate(responses):
            initial_plan = response[0].response.strip()
            new_line_idx = initial_plan.rfind('\n')
            if new_line_idx != -1:
                initial_plan = initial_plan[:new_line_idx].strip()
            initial_plan_output = InitialPlanOutput(initial_plan=initial_plan, language_output=response[0])
            plan_output = PlanOutput(task=task_outputs[idx], initial_plan=initial_plan_output, meta={'plan_id': idx})
            plan_outputs[idx] = plan_output
        return plan_outputs

    def format_log_string(self, plan_output: PlanOutput):
        return '\n\n'.join(plan_output.logs) if plan_output.logs else 'The agent has not yet interacted with the world'

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_next_step_candidates(self, group) -> List[PlanOutput]:
        plan_outputs = group.plans
        generation_params = GenerationParameters(model=self.planning_model, max_tokens=4096)
        conversations_to_process = []
        for instance_id, plan_output in plan_outputs.items():
            if plan_output.success:
                continue
            instance = group.evaluator.instances[instance_id]
            mining_setup = get_mining_setup(instance)
            starting_inventory = instance.inspect_inventory()
            starting_inventory_dict = self.get_inventory_dict(starting_inventory)
            log_str = self.format_log_string(plan_output)
            objective = plan_output.task.task
            initial_plan = plan_output.initial_plan.initial_plan
            user_message = self.step_generator_user_prompt.format(mining_setup=mining_setup, starting_inventory=starting_inventory, logs=log_str, objective=objective, plan=initial_plan)
            conversations_to_process += [(Conversation(messages=[Message(role='system', content=self.step_generator_system_prompt), Message(role='user', content=user_message)]), plan_output.meta['plan_id'])] * self.number_of_steps_for_judge
        step_outputs = [asyncio.ensure_future(self._generate_natural_language_batch(conversation[0], generation_params, meta={'type': 'next_step_candidates', 'plan_id': conversation[1], 'mining_setup': mining_setup, 'starting_inventory': starting_inventory_dict})) for conversation in conversations_to_process]
        responses = await asyncio.gather(*step_outputs)
        step_output_objects = {}
        for idx, response in enumerate(responses):
            output = response[0]
            plan_id = output.meta['plan_id']
            step_output = output.response.strip()
            if plan_id not in step_output_objects:
                step_output_objects[plan_id] = Step(candidate_language_outputs=[])
            step_output_objects[plan_id].candidate_language_outputs.append(output)
            if '#output' in step_output.lower() and '#step' not in step_output.lower():
                step_output = step_output.lower().split('#output')[-2].strip()
                plan_outputs[plan_id].success = True
                plan_outputs[plan_id].final_output = step_output
                step_output_objects[plan_id].final_step = step_output
        for plan_id, step_output in step_output_objects.items():
            plan_outputs[plan_id].steps.append(step_output)
        return plan_outputs

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_next_step_with_judge(self, group) -> List[PlanOutput]:
        plan_outputs = group.plans
        generation_params = GenerationParameters(model=self.planning_model, max_tokens=4096)
        conversations_to_process = []
        for instance_id, plan_output in plan_outputs.items():
            if plan_output.success:
                continue
            instance = group.evaluator.instances[instance_id]
            mining_setup = get_mining_setup(instance)
            starting_inventory = instance.inspect_inventory()
            log_str = self.format_log_string(plan_output)
            objective = plan_output.task.task
            initial_plan = plan_output.initial_plan.initial_plan
            step_to_process = plan_output.steps[-1].candidate_language_outputs
            step_candidate_str = ''
            for step_idx, step_candidate in enumerate(step_to_process):
                step_candidate_str += f'Step {step_idx}\n{step_candidate.response}\n\n'
            user_message = self.step_judge_user_prompt.format(objective=objective, starting_inventory=starting_inventory, mining_setup=mining_setup, logs=log_str, plan=initial_plan, analysis_step_str=step_candidate_str)
            conversations_to_process.append((Conversation(messages=[Message(role='system', content=self.step_judge_system_prompt), Message(role='user', content=user_message)]), plan_output.meta['plan_id']))
        step_outputs = [asyncio.ensure_future(self._generate_natural_language_batch(conversation[0], generation_params, meta={'type': 'next_step_judge', 'plan_id': conversation[1]})) for conversation in conversations_to_process]
        responses = await asyncio.gather(*step_outputs)
        for idx, response in enumerate(responses):
            output = response[0]
            plan_id = output.meta['plan_id']
            step_output = output.response.strip()
            plan_outputs[plan_id].steps[-1].judge_language_output_step = output
            plan_outputs[plan_id].steps[-1].judge_step_str = step_output
            if '#STEP' in step_output:
                step = step_output.split('#STEP')[-2].strip()
            elif 'OUTPUT' in step_output:
                step = step_output.split('OUTPUT')[-1].strip()
            else:
                step = None
            if step:
                plan_outputs[plan_id].steps[-1].final_step = step
        return plan_outputs

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_next_step_programs(self, group: PlanningGroup) -> List[PlanOutput]:
        """Generate programs for the next step in a specific group"""
        plan_outputs = group.plans
        generation_params = GenerationParameters(model=self.config.mcts_kwargs['executor_model'], temperature=0.5, max_tokens=4096, logits={'7032': -100})
        conversations_to_process = []
        for instance_id, plan_output in plan_outputs.items():
            if plan_output.success:
                continue
            instance = group.active_instances[instance_id]
            mining_setup = get_mining_setup(instance)
            starting_inventory = instance.inspect_inventory()
            executor_objective = plan_output.steps[-1].final_step
            user_message = self.step_executor_user_prompt.format(task=executor_objective, starting_inventory=starting_inventory, mining_setup=mining_setup)
            conversations_to_process.append((Conversation(messages=[Message(role='system', content=self.step_executor_system_prompt), Message(role='user', content=user_message)]), {'plan_id': plan_output.meta['plan_id'], 'group_id': group.group_id}))
        step_outputs = [asyncio.ensure_future(self._generate_programs_batch(conversation[0], generation_params, meta={'type': 'next_step_program', 'plan_id': conversation[1]['plan_id'], 'group_id': conversation[1]['group_id']})) for conversation in conversations_to_process]
        responses = await asyncio.gather(*step_outputs)
        for idx, response in enumerate(responses):
            output = response[0]
            plan_id = output.meta['plan_id']
            plan_outputs[plan_id].steps[-1].program = output
        return plan_outputs

    def _verify_response_is_python(self, content):
        code = content
        try:
            compile(code, filename='<ast>', mode='exec')
        except SyntaxError:
            code = code.rsplit('\n', 1)[0] + '\n'
            compile(code, filename='<ast>', mode='exec')
        return code

    def _extract_code_from_choice(self, choice) -> Optional[str]:
        """Extract code from a single completion choice"""
        code = ''
        try:
            content = choice.message.content
            code = self._verify_response_is_python(content)
            code = code.replace('from factorio_instance import *', '')
            return (code, None)
        except Exception:
            try:
                content = choice.message.content
                content_split = content.split('```python')
                code = content_split[1].split('```')[0].strip()
                text_response = content_split[0].strip()
                code = self._verify_response_is_python(code)
                code = code.replace('from factorio_instance import *', '')
                return (code, text_response)
            except Exception as e1:
                content = '\n'.join(choice.message.content.split('\n')[1:])
                try:
                    code = self._verify_response_is_python(content)
                    code = code.replace('from factorio_instance import *', '')
                    return (code, None)
                except Exception:
                    try:
                        content_split = content.split('from factorio_instance import *')
                        code = content_split[1].strip()
                        text_response = content_split[0].strip()
                        code = self._verify_response_is_python(code)
                        return (code, text_response)
                    except Exception:
                        chain_of_thoughts = '"""\n' + content.strip().strip('"') + '\n"""'
                        return (chain_of_thoughts, content.strip())
                print(f'Failed to extract code from choice: {str(e1)}')

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_programs_batch(self, conversation: Conversation, generation_params: GenerationParameters, meta={}) -> List[Program]:
        """Generate multiple programs in a single API call using 'n' parameter"""
        formatted_messages = self.formatter.to_llm_messages(self.formatter.format_conversation(conversation))
        if 'claude' in generation_params.model:
            assert generation_params.n == 1, 'Number of samples must be 1 for Claude'
        try:
            response = await self.llm.acall(messages=formatted_messages, n_samples=generation_params.n, temperature=generation_params.temperature, max_tokens=generation_params.max_tokens, logit_bias=generation_params.logit_bias, stop_sequences=generation_params.stop_sequences, model=generation_params.model, presence_penalty=generation_params.presence_penalty)
            programs = []
            try:
                messages = conversation.model_dump()['messages']
            except Exception:
                messages = conversation.dict()['messages']
            if 'claude' in generation_params.model:
                code, text_response = self._extract_code_from_choice(response)
                if code:
                    programs.append(Program(id=hash((code, json.dumps(messages))), code=code, conversation=conversation, response=response.message.content, token_usage=response.usage.total_tokens if hasattr(response, 'usage') else None, completion_token_usage=response.usage.completion_tokens if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.prompt_tokens if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta={'text_response': text_response, 'model': generation_params.model}))
                    if meta:
                        programs[0].meta.update(meta)
            else:
                for choice in response.choices:
                    code, text_response = self._extract_code_from_choice(choice)
                    if code:
                        programs.append(Program(id=hash((code, json.dumps(messages))), code=code, conversation=conversation, response=choice.message.content, token_usage=response.usage.total_tokens // generation_params.n if hasattr(response, 'usage') else None, completion_token_usage=response.usage.completion_tokens // generation_params.n if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.prompt_tokens // generation_params.n if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta={'text_response': text_response, 'model': generation_params.model}))
                        if meta:
                            programs[-1].meta.update(meta)
            return programs
        except Exception as e:
            print(f'Batch program generation failed: {str(e)}')
            return []

def get_mining_setup(instance):
    mining_setup = instance.namespace.get_entities()
    if len(mining_setup) == 0:
        mining_setup = 'There are no entities on the map'
    else:
        mining_setup = f'The following entities comprise your factory: {mining_setup}'
    return mining_setup

class SupervisedTaskExecutorABC(ABC):

    def __init__(self, instances: List[FactorioInstance], db_client: DBClient, api_factory: Any, config: SupervisedExecutorConfig, version=None, version_description=''):
        """
        Initialize parallel planning MCTS

        Args:
            instances: List of Factorio instances to distribute
            db_client: Database client
            api_factory: Factory for creating language models
            config: Configuration parameters including model paths and prompts
        """
        self.console = Console()
        self.config = config
        self.db_client = db_client
        self.llm = api_factory
        self.version = version
        self.version_description = version_description
        self.model_to_evaluate = config.model_to_evaluate
        self.formatter = DefaultFormatter()
        self._validate_instance_count(len(instances), config.n_parallel)
        instances_per_group = floor(len(instances) / config.n_parallel)
        self.logger = GroupedFactorioLogger(n_groups=config.n_parallel, instances_per_group=instances_per_group)
        self.logger.start()
        self.instance_groups = self._create_instance_groups(instances)

    def read_in_prompts(self, path):
        system_prompt_path = os.path.join(path, 'system_prompt.md')
        user_prompt_path = os.path.join(path, 'user_prompt.md')
        with open(system_prompt_path, 'r') as f:
            system_prompt = f.read()
        with open(user_prompt_path, 'r') as f:
            user_prompt = f.read()
        return (system_prompt, user_prompt)

    def _create_instance_groups(self, instances: List['FactorioInstance']) -> List[PlanningGroupV2]:
        """Create instance groups for parallel execution"""
        instances_per_group = floor(len(instances) / self.config.n_parallel)
        groups = []
        for group_id in range(self.config.n_parallel):
            start_idx = group_id * instances_per_group
            end_idx = start_idx + instances_per_group
            group_instances = instances[start_idx:end_idx]
            active_instances = group_instances
            evaluator = Evaluator(db_client=self.db_client, instances=group_instances, value_accrual_time=3, logger=self.logger)
            groups.append(PlanningGroupV2(group_id=group_id, evaluator=evaluator, active_instances=active_instances))
        return groups

    def _validate_instance_count(self, total_instances: int, n_parallel: int):
        min_required = n_parallel * 2
        if total_instances < min_required:
            raise ValueError(f'Need at least {min_required} instances for {n_parallel} parallel searches (got {total_instances})')
        instances_per_group = floor(total_instances / n_parallel)
        if instances_per_group < 2:
            raise ValueError(f'Not enough instances per group (need at least 2, got {instances_per_group})')

    async def search_supervised(self, n_iterations: int, task: TaskABC, skip_failures: bool=False, run_id: str=''):
        """
        Run truly parallel MCTS search across all groups

        Args:
            n_iterations: Number of iterations to run
            skip_failures: Whether to skip failed program generations
        """
        try:
            search_tasks = []
            for group in self.instance_groups:
                search_task = asyncio.create_task(self._run_group_search(group=group, n_iterations=n_iterations, skip_failures=skip_failures, task=task, run_id=run_id))
                search_tasks.append(search_task)
            results = await asyncio.gather(*search_tasks)
        except Exception as e:
            print(f'Error during parallel search: {str(e)}')
            raise
        finally:
            self.cleanup()
            results = [item for sublist in results for item in sublist]
            return results

    async def generate_plans(self, task: TaskABC, nr_of_beams: int) -> List[InitialPlanOutput]:
        plan_outputs = {}
        for idx in range(nr_of_beams):
            plan_output = PlanOutput(task=TaskOutput(task=task.task), meta={'plan_id': idx})
            plan_outputs[idx] = plan_output
        return plan_outputs

    @abstractmethod
    async def _run_group_search(self, group: PlanningGroupV2, task: TaskABC, n_iterations: int, skip_failures: bool=False):
        """Run parallel planning search across all groups"""
        '\n        Need to check again over what to do mcts exactly\n        '
        pass

    def cleanup(self):
        """Clean up resources"""
        self.logger.stop()
        for group in self.instance_groups:
            if hasattr(group.evaluator, 'logger'):
                group.evaluator.logger.stop()

    def get_group_metrics(self, group_id: int) -> Dict[str, Any]:
        """Get metrics for a specific group"""
        if 0 <= group_id < len(self.instance_groups):
            group = self.instance_groups[group_id]
            return {'active_instances': len(group.active_instances), 'total_programs': sum((inst.total_programs for inst in group.evaluator.logger.groups[group_id].instances.values())), 'error_count': sum((inst.error_count for inst in group.evaluator.logger.groups[group_id].instances.values()))}
        return {}

    async def _evaluate_step(self, step: Step, start_state: GameState, group: PlanningGroupV2, instance_id: int, parent_id) -> Tuple[Step, float, List]:
        """Modified to work with instance groups"""
        entity_list = []
        try:
            instance = group.active_instances[instance_id]
            group.evaluator.logger.update_instance(instance_id, status='executing')
            for program in step.sampled_programs:
                instance.reset(step.start_state)
                if not isinstance(program, Program):
                    print(f'Weird program 1: {program}')
                instance.reset(step.start_state)
                reward, state, response, entities, achievements, profits, error = await group.evaluator._evaluate_single(instance_id, program, instance)
                if not isinstance(program, Program):
                    print(f'Weird program 2: {program}')
                if error:
                    print(f'Error in group {group.group_id}, instance {instance_id}: {error}')
                step.program = program
                if not error:
                    break
            entity_list.append(entities)
            step.end_state = state
            step.reward = reward
            post_production_flows = instance.get_production_stats()
            step.program.meta['post_production_flows'] = post_production_flows
            step.program.meta['profits'] = profits
        except Exception as e:
            print(f'Error during evaluation in group {group.group_id}, instance {instance_id}: {e}')
            raise e
        step.program.value = step.reward
        step.program.raw_reward = step.reward
        step.program.state = step.end_state
        step.program.response = response
        step.program.parent_id = parent_id
        step.program.achievements = achievements
        return (step, entity_list)

    async def _process_last_step(self, plan: PlanOutput, start_state: GameState, group: PlanningGroupV2, instance_id: int, parent_id: Optional[int], skip_failures: bool) -> PlanOutput:
        try:
            step_to_process = plan.steps[-1]
            step_to_process, entity_list = await self._evaluate_step(step_to_process, start_state, group, instance_id, parent_id)
            if skip_failures and 'error' in step_to_process.program.response.lower():
                raise Exception('Found error in response. Skipping step.')
            plan.steps[-1] = step_to_process
            log_str = f'Step {len(plan.steps)}: {step_to_process.final_step}\n{step_to_process.program.response}'
            plan.logs.append(log_str)
            return plan
        except Exception as e:
            print(f'Failed to evaluate program on instance {instance_id}: {str(e)}')
            plan.steps.pop()
            return plan

    def check_for_task_completion(self, task: TaskABC, plan: PlanOutput, group: PlanningGroupV2) -> bool:
        sleep_seconds = 60
        instance_id = plan.meta['plan_id']
        instance = group.evaluator.instances[instance_id]
        start_state = plan.steps[-1].start_state
        instance.reset(start_state)
        instance_inventory = instance.inspect_inventory()
        result, achievements, post_production_flows = group.evaluator._evaluate_for_achievements(code=f'sleep({sleep_seconds})', instance=instance)
        for check_dict in task.check_dicts:
            if check_dict['task_type'] == 'craft':
                item = check_dict['item']
                quantity = check_dict['quantity']
                if instance_inventory[item] < quantity:
                    return (False, post_production_flows)
            elif check_dict['task_type'] == 'dynamic':
                item = check_dict['item']
                quantity = check_dict['quantity']
                item_dynamic_value = achievements['dynamic'].get(item, 0)
                item_dynamic_value_per_second = item_dynamic_value
                if item_dynamic_value_per_second < quantity:
                    return (False, post_production_flows)
            elif check_dict['task_type'] == 'production_flows_output':
                item = check_dict['item']
                quantity = check_dict['quantity']
                production_flows_output_item_value = post_production_flows['output'].get(item, 0)
                if production_flows_output_item_value < quantity:
                    return (False, post_production_flows)
            elif check_dict['task_type'] == 'production_flows_input':
                item = check_dict['item']
                quantity = check_dict['quantity']
                production_flows_output_item_value = post_production_flows['input'].get(item, 0)
                if production_flows_output_item_value < quantity:
                    return (False, post_production_flows)
        return (True, post_production_flows)

    def _create_output_completed_program(self, plan: PlanOutput, parent_id: Optional[int], task: TaskABC, group: PlanningGroupV2) -> PlanOutput:
        if task.check_for_completion:
            check_result, post_production_flows = self.check_for_task_completion(task, plan, group)
            post_production_flows.pop('price_list', None)
        else:
            check_result = None
            post_production_flows = None
        objective = f"'{plan.task.task}'"
        python_code = f"print('Objective {objective} has been completed. Now lets prepare the next objective.')"
        program_parent_id = plan.steps[-2].program.id if len(plan.steps) > 1 else parent_id
        program = Program(id=hash((python_code, plan.task.task, program_parent_id)), code=python_code, conversation=Conversation(messages=[]), parent_id=program_parent_id, version=self.version, version_description=self.version_description, model=self.model_to_evaluate, meta={'objective': plan.task.task, 'type': 'completed_objective', 'checked_result_correct': check_result, 'nr_of_steps': len(plan.steps), 'post_production_flows': post_production_flows})
        return program

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_natural_language_batch(self, conversation: Conversation, generation_params: GenerationParameters, meta: dict) -> List[LanguageOutput]:
        """Generate multiple programs in a single API call using 'n' parameter"""
        formatted_messages = self.formatter.to_llm_messages(self.formatter.format_conversation(conversation))
        if 'claude' in generation_params.model:
            assert generation_params.n == 1, 'Number of samples must be 1 for Claude'
        try:
            response = await self.llm.acall(messages=formatted_messages, n_samples=generation_params.n, temperature=generation_params.temperature, max_tokens=generation_params.max_tokens, logit_bias=generation_params.logit_bias, stop_sequences=generation_params.stop_sequences, model=generation_params.model)
            outputs = []
            try:
                messages = conversation.model_dump()['messages']
            except:
                messages = conversation.dict()['messages']
            if 'claude' in generation_params.model:
                str_output = response.content[0].text
                outputs.append(LanguageOutput(id=hash((str_output, json.dumps(messages))), response=str_output, conversation=conversation, token_usage=response.usage.output_tokens + response.usage.input_tokens if hasattr(response, 'usage') else None, completion_token_usage=response.usage.output_tokens if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.input_tokens if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta=meta))
            else:
                for choice in response.choices:
                    str_output = choice.message.content
                    outputs.append(LanguageOutput(id=hash((str_output, json.dumps(messages))), response=str_output, conversation=conversation, token_usage=response.usage.total_tokens // generation_params.n if hasattr(response, 'usage') else None, completion_token_usage=response.usage.completion_tokens // generation_params.n if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.prompt_tokens // generation_params.n if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta=meta))
            return outputs
        except Exception as e:
            print(f'Batch program generation failed: {str(e)}')
            return []

    def get_inventory_dict(self, inventory):
        inventory_dict = {}
        for item in inventory:
            if isinstance(item, tuple):
                inventory_dict[item[0]] = inventory[item]
            else:
                inventory_dict[item] = inventory[item]
        return inventory_dict

    def format_log_string(self, logs: list):
        return '\n\n'.join(logs) if logs else 'The agent has not yet interacted with the world'

    def _verify_response_is_python(self, content):
        code = content
        try:
            compile(code, filename='<ast>', mode='exec')
        except SyntaxError:
            code = code.rsplit('\n', 1)[0] + '\n'
            compile(code, filename='<ast>', mode='exec')
        return code

    def _extract_code_from_choice(self, input_str) -> Optional[str]:
        """Extract code from a single completion choice"""
        code = ''
        try:
            content = input_str
            code = self._verify_response_is_python(content)
            code = code.replace('from factorio_instance import *', '')
            return (code, None)
        except Exception:
            try:
                content = input_str
                content_split = content.split('```python')
                code = content_split[1].split('```')[0].strip()
                text_response = content_split[0].strip()
                code = self._verify_response_is_python(code)
                code = code.replace('from factorio_instance import *', '')
                code = code.strip()
                return (code, text_response)
            except Exception as e1:
                content = '\n'.join(input_str.split('\n')[1:])
                try:
                    code = self._verify_response_is_python(content)
                    code = code.replace('from factorio_instance import *', '')
                    code = code.strip()
                    return (code, None)
                except Exception:
                    try:
                        content_split = content.split('from factorio_instance import *')
                        code = content_split[1].strip()
                        text_response = content_split[0].strip()
                        code = self._verify_response_is_python(code)
                        code = code.strip()
                        return (code, text_response)
                    except Exception:
                        return ('', content.strip())
                print(f'Failed to extract code from choice: {str(e1)}')

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_programs_batch(self, conversation: Conversation, generation_params: GenerationParameters, meta={}) -> List[Program]:
        """Generate multiple programs in a single API call using 'n' parameter"""
        conversation = copy.deepcopy(conversation)
        formatted = await self.formatter.format_conversation(conversation)
        formatted_messages = self.formatter.to_llm_messages(formatted)
        if 'claude' in generation_params.model:
            assert generation_params.n == 1, 'Number of samples must be 1 for Claude'
        try:
            response = await self.llm.acall(messages=formatted_messages, n_samples=generation_params.n, temperature=generation_params.temperature, max_tokens=generation_params.max_tokens, logit_bias=generation_params.logit_bias, stop_sequences=generation_params.stop_sequences, model=generation_params.model, presence_penalty=generation_params.presence_penalty)
            programs = []
            try:
                messages = conversation.model_dump()['messages']
            except Exception:
                messages = conversation.dict()['messages']
            if 'claude' in generation_params.model:
                code, text_response = self._extract_code_from_choice(response.content[0].text)
                programs.append(Program(id=hash((code, json.dumps(messages))), code=code, conversation=conversation, response=response.content[0].text, token_usage=response.usage.input_tokens + response.usage.output_tokens if hasattr(response, 'usage') else None, completion_token_usage=response.usage.output_tokens if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.input_tokens if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta={'text_response': text_response, 'model': generation_params.model}))
                if meta:
                    programs[0].meta.update(meta)
            else:
                for choice in response.choices:
                    code, text_response = self._extract_code_from_choice(choice.message.content)
                    programs.append(Program(id=hash((code, json.dumps(messages))), code=code, conversation=conversation, response=choice.message.content, token_usage=response.usage.total_tokens // generation_params.n if hasattr(response, 'usage') else None, completion_token_usage=response.usage.completion_tokens // generation_params.n if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.prompt_tokens // generation_params.n if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta={'text_response': text_response, 'model': generation_params.model}))
                    if meta:
                        programs[-1].meta.update(meta)
            return programs
        except Exception as e:
            print(f'Batch program generation failed: {str(e)}')
            return []

class PlanSampler:

    def __init__(self, model: str, system_prompt_path: str, starting_scenarios_folder: str):
        self.model = model
        self.system_prompt_path = system_prompt_path
        self.llm_factory = APIFactory(model)
        self.starting_scenarios_folder = starting_scenarios_folder
        self.planning_addition_for_prompt = '\nFirst bring out a thorough step-by-step plan how you can achieve this task and then create the python script to achieve the task.\nFor your plan, follow this structure:\n1) What entities are needed for the task\n2) What entities do we have on the map, in different entity inventories or in our inventory\n3) What entities are we missing for the task\n4) Execution -- Taking into account 1,2 and 3, what steps do we need to take to successfully carry out the task\n\nCreate the python script based on your plan.\n'

    def get_game_state(self, instance: FactorioInstance, scenario: str) -> Optional[GameState]:
        """Load a scenario and return the corresponding game state"""
        try:
            instance.reset()
            scenario_path = os.path.join(self.starting_scenarios_folder, scenario)
            if not os.path.exists(scenario_path):
                return None
            config_file = os.path.join(scenario_path, 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                if 'inventory' in config:
                    instance.set_inventory(config['inventory'])
                if 'setup_script' in config:
                    setup_script_path = os.path.join(scenario_path, config['setup_script'])
                    if os.path.exists(setup_script_path):
                        with open(setup_script_path, 'r') as f:
                            setup_code = f.read()
                        instance.eval(setup_code, timeout=30)
            return GameState.from_instance(instance)
        except Exception as e:
            print(f'Error loading scenario {scenario}: {str(e)}')
            return None

    async def __call__(self, instance: FactorioInstance, game_state: GameState) -> Tuple[str, Any]:
        """Generate an objective/plan for the given game state"""
        try:
            with open(self.system_prompt_path, 'r') as f:
                system_prompt = f.read()
            inventory_info = f'Current inventory: {game_state.inventories[0]}'
            messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': f'{inventory_info}\n\n{self.planning_addition_for_prompt}'}]
            response = await self.llm_factory.acall(messages=messages, temperature=0.7, max_tokens=1024)
            if hasattr(response, 'choices'):
                objective = response.choices[0].message.content
            elif hasattr(response, 'content'):
                objective = response.content[0].text if hasattr(response.content[0], 'text') else response.content
            else:
                objective = str(response)
            return (objective.strip(), response)
        except Exception as e:
            print(f'Error generating plan: {str(e)}')
            return ('', None)

class BlueprintsToPrograms:
    """Samples scenarios from existing Factorio blueprint implementations in the `skills` table."""

    def __init__(self, db_config: Dict[str, str], system_prompt: str):
        """
        Initialize the blueprint sampler

        Args:
            db_config: Database configuration for skills DB
            system_prompt: System prompt for the conversation
        """
        self.db_config = db_config
        self.system_prompt = system_prompt
        self.conn = psycopg2.connect(**db_config)

    def _get_blueprint_scenarios(self, limit: int=100, version: str='v1.4') -> List[Dict]:
        """Get blueprint scenarios from skills database"""
        query = '\n            SELECT description, implementation, score, complexity, dependencies\n            FROM skills\n            WHERE implementation IS NOT NULL \n            AND description IS NOT NULL\n            AND version = %s\n            ORDER BY RANDOM()\n            LIMIT %s\n        '
        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (version, limit))
            return [dict(row) for row in cur.fetchall()]

    def sample_scenarios(self, n_samples: int=10, version: int=17, skill_version: str='v1.4') -> List[Program]:
        """
        Sample scenarios and create seed programs

        Args:
            instance: Factorio instance for state setup
            n_samples: Number of scenarios to sample

        Returns:
            List of Program objects ready for seeding
        """
        blueprints = self._get_blueprint_scenarios(limit=n_samples, version=skill_version)
        programs = []
        for blueprint in blueprints:
            dependencies = blueprint['dependencies']
            inventory = {}
            for dependency in dependencies:
                item, count = dependency.strip().split(':')
                inventory[item.strip("'")] = int(count)
            implementation = blueprint['implementation']
            objective = blueprint['description']
            if not objective.startswith('"""'):
                objective = f'"""\n{objective}\n"""'
            conversation = Conversation(messages=[Message(role='system', content=self.system_prompt), Message(role='user', content=f'Starting Inventory: {inventory}'), Message(role='assistant', content=objective), Message(role='user', content='Execution result: \n'), Message(role='assistant', content=implementation.strip())])
            program = Program(id=hash((objective, str(conversation.messages))), code=implementation.strip(), conversation=conversation, value=float(blueprint['score']), state=None, version=version, version_description='Blueprint-based scenario seed')
            programs.append(program)
        return programs

    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()

class ChunkedMCTS(MCTS):

    def __init__(self, *args, logit_bias: Optional[float]=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.logit_bias = logit_bias
    from typing import List

    def _split_into_chunks(self, program_code: str) -> List[Program]:
        """Split the program code into chunks based on docstrings and comments."""
        program_code = program_code.replace('from factorio_instance import *', '').strip()
        if not program_code:
            return []
        try:
            module = ast.parse(program_code)
            docstring_nodes = [node for node in module.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and node.value.value.strip().startswith('"""')]
            if docstring_nodes:
                chunks = []
                lines = program_code.splitlines()
                last_end = 0
                for node in docstring_nodes:
                    start_line = node.lineno - 1
                    if start_line > last_end:
                        chunk_lines = lines[last_end:start_line]
                        chunk_content = '\n'.join(chunk_lines).strip()
                        if chunk_content:
                            chunks.append(Program(code=chunk_content, conversation=Conversation(messages=[])))
                    current_chunk = []
                    docstring_quotes = 1
                    for line in lines[start_line:]:
                        current_chunk.append(line)
                        docstring_quotes += line.strip().count('"""')
                        if docstring_quotes % 2 == 0:
                            break
                    last_end = start_line + len(current_chunk)
                    while last_end < len(lines):
                        line = lines[last_end]
                        if line.strip().startswith('"""'):
                            break
                        current_chunk.append(line)
                        last_end += 1
                    chunk_content = '\n'.join(current_chunk).strip()
                    if chunk_content:
                        chunks.append(Program(code=chunk_content, conversation=Conversation(messages=[])))
                if last_end < len(lines):
                    chunk_content = '\n'.join(lines[last_end:]).strip()
                    if chunk_content:
                        chunks.append(Program(code=chunk_content, conversation=Conversation(messages=[])))
                return chunks
        except SyntaxError:
            pass
        lines = program_code.splitlines()
        has_numbered_comments = any((re.match('^#\\s*\\d+\\.', line.strip()) for line in lines))
        if has_numbered_comments:
            chunks = []
            current_chunk = None
            current_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    if current_lines:
                        current_lines.append('')
                    continue
                comment_match = re.match('^#\\s*(\\d+)\\.\\s*(.+)$', line)
                if comment_match:
                    if current_chunk and current_lines:
                        current_chunk.code = '\n'.join(current_lines).strip()
                        chunks.append(current_chunk)
                        current_lines = []
                    current_chunk = Program(code='', conversation=Conversation(messages=[]))
                    current_lines = [line]
                elif current_chunk is not None:
                    current_lines.append(line)
                else:
                    current_chunk = Program(code='', conversation=Conversation(messages=[]))
                    current_lines = [line]
            if current_chunk and current_lines:
                current_chunk.code = '\n'.join(current_lines).strip()
                chunks.append(current_chunk)
            return chunks
        else:
            chunks = []
            current_chunk = []
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if not line_stripped:
                    if current_chunk:
                        current_chunk.append('')
                    continue
                if line_stripped.startswith('#') and (i == 0 or not lines[i - 1].strip()) and (i + 1 < len(lines)) and lines[i + 1].strip() and (not lines[i + 1].strip().startswith('#')):
                    if current_chunk:
                        chunk_content = '\n'.join(current_chunk).strip()
                        if chunk_content:
                            chunks.append(Program(code=chunk_content, conversation=Conversation(messages=[])))
                    current_chunk = []
                current_chunk.append(line)
            if current_chunk:
                chunk_content = '\n'.join(current_chunk).strip()
                if chunk_content:
                    chunks.append(Program(code=chunk_content, conversation=Conversation(messages=[])))
            return chunks if chunks else [Program(code=program_code.strip(), conversation=Conversation(messages=[]))]

    async def _evaluate_chunks(self, chunks: List[Program], start_state: GameState, instance_id: int) -> Tuple[List[Program], List[List[Union[Entity, EntityGroup]]]]:
        """
        Evaluate chunks sequentially while computing holdout values for each chunk.

        Args:
            chunks: List of program chunks to evaluate
            start_state: Initial game state
            instance_id: ID of the instance to use for evaluation

        Returns:
            Tuple containing:
            - List of evaluated program chunks
            - List of holdout values (one per chunk)
            - List of entity lists (one per chunk)
        """
        current_state = start_state
        try:
            vars = pickle.loads(current_state.namespaces[0])
        except Exception:
            pass
        entity_list = []
        achievement_list = []
        try:
            executed_chunks = []
            for chunk in chunks:
                instance = self.evaluator.instances[instance_id]
                instance.reset(current_state)
                if self.evaluator.logger:
                    self.evaluator.logger.update_instance(self.evaluator.instance_to_port[instance_id], program_id=chunk.id, status='executing')
                reward, state, response, entities, achievements, ticks = await self.evaluator._evaluate_single(instance_id, chunk, instance)
                executed_chunks.append(chunk)
                achievement_list.append(achievements)
                entity_list.append(entities)
                chunk.state = state
                chunk.value = reward
                chunk.advantage = reward
                chunk.response = response
                current_state = state
                if 'error' in response.lower():
                    break
            return (executed_chunks, entity_list, achievement_list)
        except Exception as e:
            print('Error during chunk evaluation:')
            print(f'Instance ID: {instance_id}')
            print(f'Number of chunks: {len(chunks)}')
            print(f'Error: {str(e)}')
            import traceback
            traceback.print_exc()
            raise e

    async def search(self, n_iterations: int, samples_per_iteration: int, skip_failures: bool=False):
        for iteration in range(n_iterations):
            await self.run_iteration(samples_per_iteration, skip_failures, iteration)
            self.evaluator.logger.update_progress()

    async def run_iteration(self, samples_per_iteration, skip_failures, iteration, n_iterations):
        parent = await self.sampler.sample_parent(version=self.version)
        start_state = parent.state if parent else self.initial_state
        if not parent:
            self.evaluator.instances[0].reset(start_state)
            entities = self.evaluator.instances[0].get_entities()
            conversation = Conversation(messages=[Message(role='system', content=self.system_prompt), Message(role='assistant', content="print(f'Inventory: {inspect_inventory()}')\nprint(f'Entities: {get_entities()}')\n"), Message(role='user', content=f"1: ('Inventory: {start_state.inventories[0].__dict__}')\n2: ('Entities: {entities}')")])
        else:
            conversation = parent.conversation
        if not any((msg.role == 'system' for msg in conversation.messages)):
            raise RuntimeError('System message has been lost somehow...')
        self.evaluator.set_sampling_status()
        self.evaluator.set_iteration(iteration, n_iterations)
        raw_programs = await self._generate_programs_batch(conversation, samples_per_iteration)
        eval_futures = []
        for i, (program, chunks) in enumerate(raw_programs):
            instance_id = i % len(self.evaluator.instances)
            self.evaluator.instances[instance_id].reset(start_state)
            self.evaluator.logger.update_instance(self.evaluator.instances[i].tcp_port, program_id=program.id, status='resetting', n_iterations=n_iterations)
            eval_futures.append(self._process_program_chunks(program=program, chunks=chunks, start_state=start_state, instance_id=instance_id, parent_id=parent.id if parent else None, skip_failures=skip_failures))
        await asyncio.gather(*eval_futures)
        if parent:
            await self.sampler.visit(parent.id, len(eval_futures))

    async def _process_program_chunks(self, program: Program, chunks: List[Program], start_state: GameState, instance_id: int, parent_id: Optional[int], skip_failures: bool):
        """Process and evaluate a program's chunks with updated holdout calculation"""
        try:
            evaluated_chunks, entity_list, achievement_list = await self._evaluate_chunks(chunks, start_state, instance_id)
            last_chunk_id = parent_id
            last_conversation_stage = program.conversation
            depth = program.depth
            for chunk, entities, achievements in zip(evaluated_chunks, entity_list, achievement_list):
                try:
                    chunk_program = Program(code=chunk.code, conversation=program.conversation, value=chunk.value - (abs(self.error_penalty) if 'error' in chunk.response.lower() else 0), raw_reward=chunk.value, advantage=chunk.advantage - (abs(self.error_penalty) if 'error' in chunk.response.lower() else 0), holdout_value=0, state=chunk.state, response=chunk.response, version=self.version, version_description=self.version_description, parent_id=last_chunk_id, token_usage=program.token_usage // len(evaluated_chunks), completion_token_usage=program.completion_token_usage // len(evaluated_chunks), prompt_token_usage=program.prompt_token_usage // len(evaluated_chunks), achievements=achievements, instance=instance_id, depth=depth + 2)
                    depth += 1
                    last_conversation_stage.add_result(chunk.code, chunk.response, score=chunk.value, advantage=chunk.value)
                    chunk_program.id = hash((chunk.code, json.dumps(chunk_program.conversation.model_dump()['messages'])))
                    chunk_program.conversation = last_conversation_stage
                    if skip_failures and 'error' in chunk.response.lower():
                        print('Skipping chunk due to error in response:')
                        print(f'Response: {chunk.response}')
                        break
                    created = await self.db.create_program(chunk_program)
                    last_chunk_id = created.id
                except Exception as e:
                    print('Error processing chunk:')
                    print(f'Chunk code: {chunk.code}')
                    print(f'Response: {chunk.response}')
                    print(f'Error: {str(e)}')
                    import traceback
                    traceback.print_exc()
                    raise
        except Exception as e:
            print(f'Failed to evaluate program on instance {instance_id}:')
            print(f'Program code: {program.code}')
            print(f'Number of chunks: {len(chunks)}')
            print(f'Error: {str(e)}')
            import traceback
            traceback.print_exc()
            raise

    async def _generate_programs_batch(self, conversation: Conversation, n_samples: int) -> List[Tuple[Program, List[Program]]]:
        generation_parameters = GenerationParameters(n=n_samples, model=self.llm.model, logit_bias=self.logit_bias)
        programs = (await super()._generate_programs_batch(conversation, generation_params=generation_parameters))[:n_samples]
        chunked_programs = []
        for i, program in enumerate(programs):
            try:
                chunks = self._split_into_chunks(program.code)
                if not chunks:
                    chunks = [program]
                chunked_programs.append((program, chunks))
            except Exception as e:
                print(f'Failed to process chunks for program: {str(e)}')
                continue
        return chunked_programs

class ObjectiveMCTS(MCTS):

    def __init__(self, api_factory: 'APIFactory', db_client: DBClient, evaluator: Evaluator, sampler: DBSampler, system_prompt: str, initial_state: GameState, formatter: ConversationFormatter=DefaultFormatter(), version=1, version_description='', logit_bias=[], presence_penalty=0, frequency_penalty=0, objective_model: str='ft:gpt-4o-mini-2024-07-18:paperplane-ai:plans-tree:AcZ8gHSo'):
        self.logit_bias = logit_bias
        self.objective_tree_sampler = ObjectiveTreeSampler(APIFactory(model=objective_model))
        super().__init__(api_factory, db_client, evaluator, sampler, system_prompt, initial_state, formatter, version, version_description, presence_penalty, frequency_penalty)

    async def _get_objectives(self, conversation: Conversation) -> List[str]:
        if len(conversation.messages) == 0:
            previous_objectives = []
        elif 'objectives' not in conversation.messages[-1].metadata:
            previous_objectives = []
        elif not conversation.messages[-1].metadata['objectives']:
            previous_objectives = []
        else:
            previous_objectives = conversation.messages[-1].metadata['objectives']
        objective = await self.objective_tree_sampler.sample_tree(previous_objectives, number=1)
        return previous_objectives + objective

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_programs_batch(self, conversation: Conversation, generation_params: GenerationParameters, meta={}) -> List[Program]:
        """Generate multiple programs in a single API call using 'n' parameter"""
        if 'objectives' not in meta:
            objectives = await self._get_objectives(conversation)
            meta['objectives'] = objectives
            objective = objectives[-1]
            self._append_inventory_check_messages(conversation, objective)
            conversation.messages[-1].metadata = {**conversation.messages[-1].metadata, 'objectives': objectives}
        formatted_messages = self.formatter.to_llm_messages(self.formatter.format_conversation(conversation))
        if 'claude' in generation_params.model:
            assert generation_params.n == 1, 'Number of samples must be 1 for Claude'
        try:
            response = await self._generate_llm_response(formatted_messages, generation_params)
            return await self._process_llm_response(response, conversation, generation_params, meta)
        except Exception as e:
            print(f'Batch program generation failed: {str(e)}')
            return []

    def _append_inventory_check_messages(self, conversation: Conversation, objective: str):
        """Append inventory check messages to the conversation"""
        conversation.messages.extend([Message(role='assistant', content=f'"""\nObjective: {objective}\n"""\nprint("Inventory: ", inspect_inventory())\nprint("Entities: ", get_entities())\n', metadata={'objectives': [objective]}), Message(role='user', content="Execution Result: \n0: ('Inventory: ', {})\n1: ('Entities: ': {})", metadata={'objectives': [objective]})])

    async def _generate_llm_response(self, formatted_messages: list, params: GenerationParameters):
        """Generate response from LLM with given parameters"""
        return await self.llm.acall(messages=formatted_messages, n_samples=params.n, temperature=params.temperature, max_tokens=params.max_tokens, logit_bias=params.logit_bias, stop_sequences=params.stop_sequences, model=params.model, presence_penalty=self.presence_penalty, frequency_penalty=self.frequency_penalty)

    async def _process_llm_response(self, response, conversation: Conversation, params: GenerationParameters, meta: dict) -> List[Program]:
        """Process LLM response and create Program objects"""
        programs = []
        try:
            messages = conversation.model_dump()['messages']
        except Exception:
            messages = conversation.dict()['messages']
        if 'claude' in params.model:
            programs = await self._handle_claude_response(response, messages, meta, params.model)
        else:
            programs = await self._handle_openai_response(response, messages, meta, params)
        return programs

    async def _handle_claude_response(self, response, messages, meta, model):
        """Handle Claude-specific response format"""
        programs = []
        code, text_response = self._extract_code_from_choice(response)
        if not code:
            objectives = await self._get_objectives(Conversation(messages=[Message(**msg.dict()) for msg in messages]))
            code = f'"""\nObjective: {objectives[-1]}\n"""'
        if code:
            new_conversation = Conversation(messages=[Message(**msg.dict()) for msg in messages])
            objectives = await self._get_objectives(new_conversation)
            self._append_inventory_check_messages(new_conversation, objectives[-1])
            program = self._create_program(code=code, messages=new_conversation.messages, conversation=new_conversation, response_content=response.message.content, token_usage=self._get_token_usage(response), model=model, text_response=text_response, meta={**meta, 'objectives': objectives})
            programs.append(program)
        return programs

    async def _handle_openai_response(self, response, messages, meta, params):
        """Handle OpenAI response format with multiple choices"""
        programs = []
        for choice in response.choices:
            code, text_response = self._extract_code_from_choice(choice)
            objectives = messages[-1]['metadata']['objectives']
            if not code:
                new_conversation = Conversation(messages=[Message(**msg) for msg in messages])
                objectives = await self._get_objectives(new_conversation)
                code = f'"""\nObjective: {objectives[-1]}\n"""'
            if code:
                new_conversation = Conversation(messages=[Message(**msg) for msg in messages])
                program = self._create_program(code=code, messages=new_conversation.messages, conversation=new_conversation, response_content=choice.message.content, token_usage=self._get_token_usage(response, divide_by_n=params.n), model=params.model, text_response=text_response, meta={**meta, 'objectives': objectives})
                programs.append(program)
        return programs

    def _create_program(self, code, messages, conversation, response_content, token_usage, model, text_response, meta):
        """Create a Program object with given parameters"""
        program = Program(id=hash((code, json.dumps([msg.__dict__ if isinstance(msg, Message) else msg for msg in messages]))), code=code, conversation=conversation, response=response_content, token_usage=token_usage.get('total'), completion_token_usage=token_usage.get('completion'), prompt_token_usage=token_usage.get('prompt'), version=self.version, version_description=self.version_description, meta={'text_response': text_response, 'model': model})
        if meta:
            program.meta.update(meta)
        return program

    def _get_token_usage(self, response, divide_by_n=1):
        """Extract token usage from response"""
        if not hasattr(response, 'usage'):
            return {'total': None, 'completion': None, 'prompt': None}
        return {'total': response.usage.total_tokens // divide_by_n, 'completion': response.usage.completion_tokens // divide_by_n, 'prompt': response.usage.prompt_tokens // divide_by_n}

    async def search(self, n_iterations: int, samples_per_iteration: int, skip_failures: bool=False):
        """
        Search for the best program using Monte Carlo Tree Search (MCTS).
        :param n_iterations: Number of iterations to perform.
        :param samples_per_iteration: Number of programs to sample per iteration.
        :param skip_failures: Whether to skip saving failed program generations.
        """
        for iteration in range(n_iterations):
            print(f'Starting iteration {iteration}')
            await self.run_iteration(samples_per_iteration, skip_failures)
            self.evaluator.logger.update_progress()

    @tenacity.retry(retry=retry_if_exception_type(psycopg2.Error), wait=wait_exponential(multiplier=1, min=1, max=4), stop=tenacity.stop_after_attempt(3))
    async def run_iteration(self, samples_per_iteration, skip_failures):
        """Run a single MCTS iteration with retries for concurrent operations"""
        try:
            parent = await self.sampler.sample_parent(version=self.version)
            if parent:
                start_state = parent.state
                conversation = parent.conversation
            else:
                start_state = self.initial_state
                conversation = Conversation(messages=[Message(role='system', content=self.system_prompt), Message(role='user', content=OBJECTIVE_PLANNING_PROMPT, metadata={'objectives': ['1. Automate resource production']})])
            self.evaluator.set_sampling_status()
            generation_parameters = GenerationParameters(n=samples_per_iteration, model=self.llm.model, stop_sequences=['Objective:'], logit_bias=self.logit_bias, presence_penalty=0.7)
            programs = await self._generate_programs_batch(conversation, generation_parameters)
            if not programs:
                return
            programs = [p for p in programs if p is not None]
            for program in programs:
                program.parent_id = parent.id if parent else None
            evaluated_programs = await self.evaluator.evaluate_batch(programs, start_state)
            save_tasks = []
            for program in evaluated_programs:
                if program.state is not None:
                    if not skip_failures or program.value is not None:
                        save_tasks.append(self.db.create_program(program))
            if save_tasks:
                await asyncio.gather(*save_tasks)
        except Exception as e:
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                print(f'Max retries ({self.max_retries}) reached. Error: {str(e)}')
                self.retry_count = 0
                raise e
            raise e

class MCTSFactory:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not MCTSFactory._initialized:
            self.db_client = None
            self.api_factory = None
            self.instances = None
            self.sampler = None
            MCTSFactory._initialized = True

    def initialize(self, instances, db_client, config: Union[BaseConfig, PlanningConfig, ChunkedConfig], sampler_config: SamplerConfig):
        self.instances = instances
        self.db_client = db_client
        self.config = config
        self.api_factory = APIFactory(model=config.model)
        self.sampler = _get_sampler(config.sampler_type, db_client, **sampler_config.__dict__)

    def create_mcts(self, config: Union[BaseConfig, PlanningConfig, ChunkedConfig, ObjectiveConfig]):
        if not all([self.instances, self.db_client, self.api_factory, self.sampler]):
            raise ValueError('Factory not initialized. Call initialize() first.')
        if config.mcts_type == MCTSType.CHUNKED:
            return self._create_chunked_mcts(config)
        elif config.mcts_type == MCTSType.PLANNING:
            return self._create_planning_mcts(config)
        elif config.mcts_type == MCTSType.OBJECTIVE:
            return self._create_objective_mcts(config)
        elif config.mcts_type == MCTSType.NORMAL:
            return self._create_mcts(config)
        raise ValueError(f'Unknown MCTS type: {config.mcts_type}')

    def _create_mcts(self, config: BaseConfig):
        from eval.algorithms.mcts import MCTS
        from eval.algorithms.mcts import ParallelMCTS
        from eval.algorithms.mcts import ParallelMCTSConfig
        mcts_config = ParallelMCTSConfig(n_parallel=config.n_parallel, system_prompt=config.system_prompt, initial_state=GameState.from_instance(self.instances[0]), mcts_class=MCTS, sampler=self.sampler, mcts_kwargs={'version': config.version, 'version_description': config.version_description, 'error_penalty': config.error_penalty})
        return ParallelMCTS(instances=self.instances, db_client=self.db_client, api_factory=self.api_factory, config=mcts_config, version=config.version, version_description=config.version_description)

    def _create_chunked_mcts(self, config: ChunkedConfig):
        from .mcts import ChunkedMCTS
        from .parallel_mcts import ParallelMCTS
        from .parallel_mcts_config import ParallelMCTSConfig
        from fle.agents.formatters.conversation_formatter_abc import StructurePreservingFormatter
        mcts_config = ParallelMCTSConfig(n_parallel=config.n_parallel, system_prompt=config.system_prompt, initial_state=GameState.from_instance(self.instances[0]), mcts_class=ChunkedMCTS, sampler=self.sampler, mcts_kwargs={'logit_bias': config.logit_bias, 'version': config.version, 'version_description': config.version_description, 'formatter': StructurePreservingFormatter(planning=True), 'presence_penalty': config.presence_penalty, 'frequency_penalty': config.frequency_penalty, 'error_penalty': config.error_penalty})
        return ParallelMCTS(instances=self.instances, db_client=self.db_client, api_factory=self.api_factory, config=mcts_config, version=config.version, version_description=config.version_description)

    def _create_objective_mcts(self, config: ObjectiveConfig):
        from eval.algorithms.mcts import ObjectiveMCTS
        from eval.algorithms.mcts import ParallelMCTS
        from eval.algorithms.mcts import ParallelMCTSConfig
        from fle.agents.formatters.conversation_formatter_abc import StructurePreservingFormatter
        mcts_config = ParallelMCTSConfig(n_parallel=config.n_parallel, system_prompt=config.system_prompt, initial_state=GameState.from_instance(self.instances[0]), mcts_class=ObjectiveMCTS, sampler=self.sampler, mcts_kwargs={'objective_model': config.objective_model, 'logit_bias': config.logit_bias, 'version': config.version, 'version_description': config.version_description, 'formatter': StructurePreservingFormatter(planning=True), 'presence_penalty': config.presence_penalty, 'frequency_penalty': config.frequency_penalty, 'error_penalty': config.error_penalty})
        return ParallelMCTS(instances=self.instances, db_client=self.db_client, api_factory=self.api_factory, config=mcts_config, version=config.version, version_description=config.version_description)

    def _create_planning_mcts(self, config: PlanningConfig):
        from eval.algorithms.mcts import PlanningMCTS
        from eval.algorithms.mcts import ParallelPlanningMCTS
        from eval.algorithms.mcts import ParallelMCTSConfig
        game_state = GameState.from_instance(self.instances[0])
        mcts_config = ParallelMCTSConfig(n_parallel=config.n_parallel, mcts_class=PlanningMCTS, sampler=self.sampler, system_prompt=config.system_prompt, initial_state=game_state, max_steps_per_objective=config.max_steps_per_objective, number_of_steps_for_judge=config.number_of_steps_for_judge, mcts_kwargs={'planning_model': config.planning_model, 'executor_model': config.executor_model, 'objective_model': config.objective_model, 'step_executor_prompt_path': config.step_executor_prompt_path, 'step_generator_prompt_path': config.step_generator_prompt_path, 'step_judge_prompt_path': config.step_judge_prompt_path, 'example_plan_prompt_path': config.example_plan_prompt_path, 'system_prompt': config.system_prompt, 'initial_state': game_state, 'error_penalty': config.error_penalty})
        return ParallelPlanningMCTS(instances=self.instances, db_client=self.db_client, api_factory=self.api_factory, config=mcts_config, version=config.version, version_description=config.version_description)

    @staticmethod
    def get_config_from_cli(default_version=42) -> Tuple[Union[BaseConfig, PlanningConfig, ChunkedConfig], SamplerConfig]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--type', choices=['chunked', 'planning', 'normal', 'objective'], help='MCTS type')
        parser.add_argument('--no-interactive', action='store_true', help='Skip interactive prompts')
        args, _ = parser.parse_known_args()
        if args.no_interactive:
            config, sampler_config = MCTSFactory._get_config_from_args(parser)
        else:
            config, sampler_config = MCTSFactory._get_config_interactive(args.type, default_version)
        MCTSFactory._save_config(config, sampler_config)
        return (config, sampler_config)

    @staticmethod
    def _get_config_from_args(parser) -> Tuple[Union[BaseConfig, PlanningConfig, ChunkedConfig], SamplerConfig]:
        parser.add_argument('--model', required=True)
        parser.add_argument('--version', type=int, required=True)
        parser.add_argument('--version-description', required=True)
        parser.add_argument('--n-parallel', type=int, default=4)
        parser.add_argument('--error-penalty', type=float, default=-10)
        parser.add_argument('--temperature', type=float, default=0.7)
        parser.add_argument('--compression-strength', type=float, default=None)
        parser.add_argument('--max-conversation-length', type=int, default=30)
        parser.add_argument('--adaptive-period', type=int, default=200)
        parser.add_argument('--window-size', type=int, default=200)
        parser.add_argument('--planning-model', default='claude-3-5-sonnet-20241022')
        parser.add_argument('--executor-model', default='ft:gpt-4o-2024-08-06:paperplane-ai:fact-instruct-1:ATSVGf4d:ckpt-step-214')
        parser.add_argument('--objective-model', default='ft:gpt-4o-2024-08-06:paperplane-ai:fact-self-gen-planning:AQzcPI91')
        parser.add_argument('--step-executor-prompt-path', default='../../prompts/bottoms_up_prompts/finetuning_prompts/step_supervised')
        parser.add_argument('--step-generator-prompt-path', default='../../prompts/bottoms_up_prompts/finetuning_prompts/step_generator')
        parser.add_argument('--step-judge-prompt-path', default='../../prompts/bottoms_up_prompts/finetuning_prompts/step_judge')
        parser.add_argument('--example-plan-prompt-path', default='../../prompts/bottoms_up_prompts/finetuning_prompts/executor_plan')
        args = parser.parse_args()
        mcts_type = MCTSType(args.type)
        if mcts_type == MCTSType.PLANNING:
            mcts_config = PlanningConfig(mcts_type=mcts_type, model=args.model, version=args.version, version_description=args.version_description, n_parallel=args.n_parallel, system_prompt='', planning_model=args.planning_model, executor_model=args.executor_model, objective_model=args.objective_model, step_executor_prompt_path=Path(args.step_executor_prompt_path), step_generator_prompt_path=Path(args.step_generator_prompt_path), step_judge_prompt_path=Path(args.step_judge_prompt_path), example_plan_prompt_path=Path(args.example_plan_prompt_path), error_penalty=args.error_penalty)
        elif mcts_type == MCTSType.CHUNKED:
            mcts_config = ChunkedConfig(mcts_type=mcts_type, model=args.model, version=args.version, version_description=args.version_description, n_parallel=args.n_parallel, system_prompt='', error_penalty=args.error_penalty)
        elif mcts_type == MCTSType.OBJECTIVE:
            mcts_config = ObjectiveConfig(objective_model=args.objective_model, mcts_type=mcts_type, model=args.model, version=args.version, version_description=args.version_description, n_parallel=args.n_parallel, system_prompt='', error_penalty=args.error_penalty)
        else:
            mcts_config = BaseConfig(mcts_type=mcts_type, model=args.model, version=args.version, version_description=args.version_description, n_parallel=args.n_parallel, system_prompt='', error_penalty=args.error_penalty)
        sampler_config = SamplerConfig(temperature=args.temperature, compression_strength=args.compression_strength, max_conversation_length=args.max_conversation_length, adaptive_period=args.adaptive_period, window_size=args.window_size)
        return (mcts_config, sampler_config)

    @staticmethod
    def _get_config_interactive(default_type=None, default_version=42) -> Tuple[Union[BaseConfig, PlanningConfig, ChunkedConfig], SamplerConfig]:
        mcts_type = default_type or questionary.select('Select MCTS type:', choices=['chunked', 'normal', 'planning', 'objective'], instruction='Choose MCTS algorithm variant. Planning is recommended for complex tasks.').ask()
        model = 'gpt-4o'
        if mcts_type != 'planning':
            model = questionary.select('Model name:', choices=['gemini-2.0-flash-exp', 'gemini-2.0-flash-thinking-exp-1219', 'gemini-exp-1206', 'deepseek-chat', 'gpt-4o', 'claude-3-5-sonnet-20241022', 'meta-llama/Llama-3.3-70B-Instruct-Turbo', 'meta-llama/Meta-Llama-3.3-8B-Instruct-Turbo', 'Qwen/Qwen2.5-7B-Instruct-Turbo', 'Qwen/Qwen2.5-72B-Instruct-Turbo', 'ft:gpt-4o-mini-2024-07-18:paperplane-ai:mcts-pruned-masked:AYIViDdb'], instruction='Model to use for program synthesis.').ask()
        base_config = {'mcts_type': MCTSType(mcts_type), 'model': model, 'version': int(questionary.text('Version:', default=str(default_version), instruction='The run version number. Higher versions may include bug fixes or improvements.').ask()), 'n_parallel': int(questionary.text('Number of parallel instances:', default='4').ask()), 'presence_penalty': float(questionary.text('Fixed presence penalty applied across previously sampled logits. -2 to 2.', default='0').ask()), 'frequency_penalty': float(questionary.text('Dynamic frequency penalty applied across previously sampled logits. -2 to 2.', default='0').ask()), 'error_penalty': float(questionary.text('Penalty applied when there is an execution error(e.g. syntax error).', default='-10').ask()), 'system_prompt': ''}
        if mcts_type == 'planning':
            mcts_config = PlanningConfig(**base_config, planning_model=questionary.text('Planning model:', default='claude-3-5-sonnet-20241022', instruction='The model that samples plans by reasoning over objectives and game states.').ask(), executor_model=questionary.text('Executor model:', default='ft:gpt-4o-2024-08-06:paperplane-ai:fact-instruct-1:ATSVGf4d:ckpt-step-214', instruction='The model that samples programs.').ask(), objective_model=questionary.text('Objective model:', default='ft:gpt-4o-2024-08-06:paperplane-ai:fact-self-gen-planning:AQzcPI91', instruction='The model that generates new objectives.').ask(), max_steps_per_objective=int(questionary.text('Maximum steps per objective:', default='12').ask()), number_of_steps_for_judge=int(questionary.text('Number of steps for judge:', default='3', instruction='The branching factor for the planning tree. Higher values increase quality but use more tokens.').ask()))
        elif mcts_type == 'objective':
            mcts_config = ObjectiveConfig(**base_config, objective_model=questionary.text('Objective model:', default='ft:gpt-4o-mini-2024-07-18:paperplane-ai:plans-tree:AcZ8gHSo', instruction='The model that samples objectives.').ask())
        elif mcts_type == 'chunked':
            mcts_config = ChunkedConfig(**base_config)
        else:
            mcts_config = BaseConfig(**base_config)
        mcts_config.sampler_type = SamplerType(questionary.select('Select MCTS node sampler type:', choices=['weighted reward', 'kld', 'beam'], instruction='Choose the sampling method for selecting actions. KLD priorities varied game states. Weighted reward prioritizes high-reward states.').ask())
        skip_failures = questionary.select('Skip failures?', choices=['no', 'yes'], instruction='Shall we skip nodes that trigger an exception/error?').ask()
        mcts_config.skip_failures = skip_failures == 'yes'
        if mcts_config.sampler_type == SamplerType.KLD:
            sampler_config = SamplerConfig(temperature=float(questionary.text('Temperature:', default='1', instruction='Higher values are closer to uniform sampling. Zero means greedy sampling from reward.').ask()), window_size=int(questionary.text('Window size:', default='100', instruction='The number of recent programs to consider when sampling the next node').ask()), maximum_lookback=int(questionary.text('Maximum lookback steps', default='20').ask()))
        elif mcts_config.sampler_type == SamplerType.BEAM:
            sampler_config = SamplerConfig(beam_width=int(questionary.text('Beam width:', default='8', instruction='The number of nodes to keep in the beam for sampling subsequent nodes').ask()), exploration_prob=float(questionary.text('Exploration probability:', default='0.1', instruction='The probability to sample outside of the beam (for exploration)').ask()), maximum_lookback=int(questionary.text('Maximum lookback steps', default='20').ask()))
        else:
            compression_strength = float(questionary.text('Compression strength:', instruction='Between 0-1. Higher values mean more exploration. Lower means more exploitation. -1 means adaptively cycle', default='-1').ask())
            if compression_strength < 0:
                compression_strength = None
            sampler_config = SamplerConfig(compression_strength=compression_strength, max_conversation_length=int(questionary.text('Maximum conversation length:', instruction='The maximum number of assistant actions in the dialogue', default='100').ask()))
            if compression_strength is not None:
                sampler_config.adaptive_period = int(questionary.text('Adaptive period:', instruction='The period for cycling exploration and exploitation', default='50').ask())
            sampler_config.maximum_lookback = int(questionary.text('Maximum lookback steps', default='20').ask())
        version_description = ''
        for key, value in mcts_config.__dict__.items():
            if isinstance(value, Path):
                value = str(value)
            version_description += f'{key}:{value}\n'
        for key, value in sampler_config.__dict__.items():
            if isinstance(value, Path):
                value = str(value)
            version_description += f'{key}:{value}\n'
        mcts_config.version_description = version_description
        return (mcts_config, sampler_config)

    @staticmethod
    def _save_config(config: Union[BaseConfig, PlanningConfig, ChunkedConfig], sampler_config: SamplerConfig):
        """Save the run configuration to a JSON file"""
        runs_dir = Path(f'runs/{config.version}')
        runs_dir.mkdir(exist_ok=True)
        config_dict = {k: str(v) if isinstance(v, (Path, MCTSType, SamplerType)) else v for k, v in asdict(config).items() if not k.endswith('_model') and (not isinstance(v, (Path, type(None))))}
        sampler_dict = {k: v for k, v in dataclasses.asdict(sampler_config).items() if v is not None}
        save_data = {'mcts_config': config_dict, 'sampler_config': sampler_dict, 'timestamp': datetime.now().isoformat()}
        config_file = runs_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(save_data, f, indent=2)

def _get_sampler(sampler_type: SamplerType, db_client, compression_strength=None, max_conversation_length=20, adaptive_period=200, window_size: int=300, temperature: float=1.0, beam_width: int=8, exploration_prob=0.1, maximum_lookback=10):
    from fle.eval.algorithms.mcts.samplers import KLDiversityAchievementSampler
    from fle.eval.algorithms.mcts.samplers import DynamicRewardWeightedSampler
    if sampler_type == SamplerType.KLD:
        return KLDiversityAchievementSampler(db_client, window_size, temperature)
    elif sampler_type == SamplerType.BEAM:
        return BeamSampler(db_client, beam_width, max_conversation_length, exploration_prob)
    return DynamicRewardWeightedSampler(db_client, compression_strength, max_conversation_length, adaptive_period, maximum_lookback)

class PlanningMCTS(MCTS):

    def __init__(self, *args, planning_model, executor_model, objective_model, step_executor_prompt_path, step_generator_prompt_path, step_judge_prompt_path, example_plan_prompt_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.planning_model = planning_model
        self.executor_model = executor_model
        self.objective_model = objective_model
        self.api_description = self.evaluator.instances[0].get_system_prompt()
        self.step_executor_prompt_path = step_executor_prompt_path
        self.step_generator_prompt_path = step_generator_prompt_path
        self.step_judge_prompt_path = step_judge_prompt_path
        self.example_plan_prompt_path = example_plan_prompt_path
        self.step_executor_system_prompt, self.step_executor_user_prompt = self.read_in_prompts(step_executor_prompt_path)
        self.step_generator_system_prompt, self.step_generator_user_prompt = self.read_in_prompts(step_generator_prompt_path)
        self.step_judge_system_prompt, self.step_judge_user_prompt = self.read_in_prompts(step_judge_prompt_path)
        self.example_plan_system_prompt, self.example_plan_user_prompt = self.read_in_prompts(example_plan_prompt_path)
        self.step_executor_system_prompt = self.step_executor_system_prompt.format(schema=self.api_description)
        self.example_plan_system_prompt = self.example_plan_system_prompt.format(schema=self.api_description)
        self.max_steps_per_objective = 12
        self.number_of_steps_for_judge = 3

    def read_in_prompts(self, path):
        system_prompt_path = os.path.join(path, 'system_prompt.md')
        user_prompt_path = os.path.join(path, 'user_prompt.md')
        with open(system_prompt_path, 'r') as f:
            system_prompt = f.read()
        with open(user_prompt_path, 'r') as f:
            user_prompt = f.read()
        return (system_prompt, user_prompt)

    def _split_into_chunks(self, program_code: str) -> List[Program]:
        """Split the program code into chunks based on docstrings."""
        program_code = program_code.replace('from factorio_instance import *', '').strip()
        lines = program_code.splitlines()
        chunks = []
        module = ast.parse(program_code)
        docstring_positions = []
        for node in module.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                docstring_positions.append((node.lineno - 1, node.end_lineno - 1, node.value.s))
        if not docstring_positions:
            return []
        first_docstring_start = docstring_positions[0][0]
        if first_docstring_start > 0:
            preamble = lines[:first_docstring_start]
            if any((line.strip() for line in preamble)):
                chunks.append(Program(code='\n'.join(preamble), conversation=Conversation(messages=[])))
        for i, (start_pos, end_pos, docstring) in enumerate(docstring_positions):
            chunk_lines = []
            chunk_lines.extend(lines[start_pos:end_pos + 1])
            if i < len(docstring_positions) - 1:
                next_start = docstring_positions[i + 1][0]
                chunk_lines.extend(lines[end_pos + 1:next_start])
            else:
                chunk_lines.extend(lines[end_pos + 1:])
            if chunk_lines:
                chunks.append(Program(code='\n'.join(chunk_lines), conversation=Conversation(messages=[])))
        return chunks

    async def _evaluate_step(self, step: Step, start_state: GameState, instance_id: int, parent_id) -> Tuple[List[Program], float]:
        entity_list = []
        try:
            instance = self.evaluator.instances[instance_id]
            step.start_state = GameState.from_instance(instance)
            self.evaluator.logger.update_instance(instance_id, status='executing')
            reward, state, response, entities, achievements, ticks = await self.evaluator._evaluate_single(instance_id, step.program, instance)
            entity_list.append(entities)
            step.end_state = state
            step.reward = reward
        except Exception as e:
            print(f'Error during evaluation: {e}')
            raise e
        step.program.value = step.reward - (abs(self.error_penalty) if 'error' in response else 0)
        step.program.achievements = achievements
        step.program.raw_reward = step.reward
        step.program.holdout_value = step.program.value
        step.program.state = step.end_state
        step.program.response = response
        step.program.parent_id = parent_id
        step.program.conversation.add_result(step.program.code, response, score=step.reward, advantage=step.reward)
        return (step, step.program.value, entity_list)

    def get_inventory_dict(self, inventory):
        inventory_dict = {}
        for item in inventory:
            if isinstance(item, tuple):
                inventory_dict[item[0]] = inventory[item]
            else:
                inventory_dict[item] = inventory[item]
        return inventory_dict

    async def _get_tasks(self, parent: Program=None, samples_per_iteration: int=1) -> List[LanguageOutput]:
        start_state = parent.state if parent else self.initial_state
        self.evaluator.instances[0].reset(start_state)
        mining_setup = get_mining_setup(self.evaluator.instances[0])
        starting_inventory = self.evaluator.instances[0].inspect_inventory()
        conversation = Conversation(messages=[Message(role='system', content=self.system_prompt), Message(role='user', content=f'Your starting inventory is {starting_inventory}. Your initial mining setup is: {mining_setup}. Create a useful task that you can carry out in the current game and the python script to achieve the task')])
        generation_params = GenerationParameters(model=self.objective_model, n=samples_per_iteration, stop_sequences=['\n'])
        inventory_dict = self.get_inventory_dict(starting_inventory)
        game_state_str = GameState.from_instance(self.evaluator.instances[0]).entities
        tasks = await self._generate_natural_language_batch(conversation, generation_params, meta={'type': 'objective_generation', 'inventory': inventory_dict, 'mining_setup': mining_setup, 'game_state': game_state_str})
        task_outputs = []
        for task in tasks:
            task_string = task.response.split('\n')[0].strip()
            task_string = task_string.lower().replace('sure! the task i will carry out is', '').strip()
            if '.' in task_string:
                task_string = task_string.split('.')[0]
            task_outputs.append(TaskOutput(task=task_string, language_output=task))
        return (task_outputs, start_state)

    async def generate_plans(self, task_outputs: List[TaskOutput]) -> List[InitialPlanOutput]:
        generation_params = GenerationParameters(model=self.executor_model, stop_sequences=['```'])
        conversations_to_process = [Conversation(messages=[Message(role='system', content=self.example_plan_system_prompt), Message(role='user', content=self.example_plan_user_prompt.format(task=task_output.task))]) for task_output in task_outputs]
        initial_plans = [asyncio.ensure_future(self._generate_natural_language_batch(conversation, generation_params, meta={'type': 'initial_plan_generation'})) for conversation in conversations_to_process]
        responses = await asyncio.gather(*initial_plans)
        plan_outputs = {}
        for idx, response in enumerate(responses):
            initial_plan = response[0].response.strip()
            new_line_idx = initial_plan.rfind('\n')
            if new_line_idx != -1:
                initial_plan = initial_plan[:new_line_idx].strip()
            initial_plan_output = InitialPlanOutput(initial_plan=initial_plan, language_output=response[0])
            plan_output = PlanOutput(task=task_outputs[idx], initial_plan=initial_plan_output, meta={'plan_id': idx})
            plan_outputs[idx] = plan_output
        return plan_outputs

    def format_log_string(self, plan_output: PlanOutput):
        return '\n\n'.join(plan_output.logs) if plan_output.logs else 'The agent has not yet interacted with the world'

    async def generate_next_step_candidates(self, plan_outputs: dict[int, PlanOutput]) -> List[PlanOutput]:
        generation_params = GenerationParameters(model=self.planning_model, max_tokens=4096)
        conversations_to_process = []
        for instance_id, plan_output in plan_outputs.items():
            if plan_output.success:
                continue
            instance = self.evaluator.instances[instance_id]
            mining_setup = get_mining_setup(instance)
            starting_inventory = instance.inspect_inventory()
            starting_inventory_dict = self.get_inventory_dict(starting_inventory)
            log_str = self.format_log_string(plan_output)
            objective = plan_output.task.task
            initial_plan = plan_output.initial_plan.initial_plan
            user_message = self.step_generator_user_prompt.format(mining_setup=mining_setup, starting_inventory=starting_inventory, logs=log_str, objective=objective, plan=initial_plan)
            conversations_to_process += [(Conversation(messages=[Message(role='system', content=self.step_generator_system_prompt), Message(role='user', content=user_message)]), plan_output.meta['plan_id'])] * self.number_of_steps_for_judge
        step_outputs = [asyncio.ensure_future(self._generate_natural_language_batch(conversation[0], generation_params, meta={'type': 'next_step_candidates', 'plan_id': conversation[1], 'mining_setup': mining_setup, 'starting_inventory': starting_inventory_dict})) for conversation in conversations_to_process]
        responses = await asyncio.gather(*step_outputs)
        step_output_objects = {}
        for idx, response in enumerate(responses):
            output = response[0]
            plan_id = output.meta['plan_id']
            step_output = output.response.strip()
            if plan_id not in step_output_objects:
                step_output_objects[plan_id] = Step(candidate_language_outputs=[])
            step_output_objects[plan_id].candidate_language_outputs.append(output)
            if '#output' in step_output.lower() and '#step' not in step_output.lower():
                step_output = step_output.lower().split('#output')[-2].strip()
                plan_outputs[plan_id].success = True
                plan_outputs[plan_id].final_output = step_output
                step_output_objects[plan_id].final_step = step_output
        for plan_id, step_output in step_output_objects.items():
            plan_outputs[plan_id].steps.append(step_output)
        return plan_outputs

    async def get_next_step_with_judge(self, plan_outputs: dict[int, PlanOutput]) -> List[PlanOutput]:
        generation_params = GenerationParameters(model=self.planning_model, max_tokens=4096)
        conversations_to_process = []
        for instance_id, plan_output in plan_outputs.items():
            if plan_output.success:
                continue
            instance = self.evaluator.instances[instance_id]
            mining_setup = get_mining_setup(instance)
            starting_inventory = instance.inspect_inventory()
            log_str = self.format_log_string(plan_output)
            objective = plan_output.task.task
            initial_plan = plan_output.initial_plan.initial_plan
            step_to_process = plan_output.steps[-1].candidate_language_outputs
            step_candidate_str = ''
            for step_idx, step_candidate in enumerate(step_to_process):
                step_candidate_str += f'Step {step_idx}\n{step_candidate.response}\n\n'
            user_message = self.step_judge_user_prompt.format(objective=objective, starting_inventory=starting_inventory, mining_setup=mining_setup, logs=log_str, plan=initial_plan, analysis_step_str=step_candidate_str)
            conversations_to_process.append((Conversation(messages=[Message(role='system', content=self.step_judge_system_prompt), Message(role='user', content=user_message)]), plan_output.meta['plan_id']))
        step_outputs = [asyncio.ensure_future(self._generate_natural_language_batch(conversation[0], generation_params, meta={'type': 'next_step_judge', 'plan_id': conversation[1]})) for conversation in conversations_to_process]
        responses = await asyncio.gather(*step_outputs)
        for idx, response in enumerate(responses):
            output = response[0]
            plan_id = output.meta['plan_id']
            step_output = output.response.strip()
            plan_outputs[plan_id].steps[-1].judge_language_output_step = output
            plan_outputs[plan_id].steps[-1].judge_step_str = step_output
            if '#STEP' in step_output:
                step = step_output.split('#STEP')[-2].strip()
            elif 'OUTPUT' in step_output:
                step = step_output.split('OUTPUT')[-1].strip()
            else:
                step = None
            if step:
                plan_outputs[plan_id].steps[-1].final_step = step
        return plan_outputs

    async def get_next_step_programs(self, plan_outputs: dict[int, PlanOutput]) -> List[PlanOutput]:
        generation_params = GenerationParameters(model=self.executor_model, temperature=0.5, max_tokens=4096)
        conversations_to_process = []
        for instance_id, plan_output in plan_outputs.items():
            if plan_output.success:
                continue
            instance = self.evaluator.instances[instance_id]
            mining_setup = get_mining_setup(instance)
            starting_inventory = instance.inspect_inventory()
            executor_objective = plan_output.steps[-1].final_step
            user_message = self.step_executor_user_prompt.format(task=executor_objective, starting_inventory=starting_inventory, mining_setup=mining_setup)
            conversations_to_process.append((Conversation(messages=[Message(role='system', content=self.step_executor_system_prompt), Message(role='user', content=user_message)]), plan_output.meta['plan_id']))
        step_outputs = [asyncio.ensure_future(self._generate_programs_batch(conversation[0], generation_params, meta={'type': 'next_step_program', 'plan_id': conversation[1]})) for conversation in conversations_to_process]
        responses = await asyncio.gather(*step_outputs)
        for idx, response in enumerate(responses):
            output = response[0]
            plan_id = output.meta['plan_id']
            plan_outputs[plan_id].steps[-1].program = output
        return plan_outputs

    async def search(self, n_iterations: int, samples_per_iteration: int, skip_failures: bool=False):
        for iteration in range(n_iterations):
            parent = await self.sampler.sample_parent(version=self.version)
            tasks, start_state = await self._get_tasks(parent, samples_per_iteration)
            plans = await self.generate_plans(tasks)
            for step_idx in range(self.max_steps_per_objective):
                if step_idx == 0:
                    for instance_id, instance in enumerate(self.evaluator.instances):
                        instance.reset(start_state)
                self.evaluator.set_status(f'Getting candidates for step {step_idx}')
                plans = await self.generate_next_step_candidates(plans)
                self.evaluator.set_status(f'Judging candidates for step {step_idx}')
                plans = await self.get_next_step_with_judge(plans)
                self.evaluator.set_status(f'Generating programs for step {step_idx}')
                plans = await self.get_next_step_programs(plans)
                eval_futures = []
                for instance_id, plan_object in plans.items():
                    if plan_object.success:
                        continue
                    self.evaluator.instances[instance_id].reset(start_state)
                    latest_program = plan_object.steps[-1].program
                    self.evaluator.logger.update_instance(instance_id, program_id=latest_program.id, status='starting to evaluate')
                    eval_futures.append(self._process_last_step(plan=plan_object, start_state=start_state, instance_id=instance_id, parent_id=parent.id if parent else None, skip_failures=skip_failures))
                await asyncio.gather(*eval_futures)
                self.evaluator.logger.update_progress()
            for plan in plans.values():
                if plan.success:
                    self.save_plan(plan)

    def save_plan(self, plan: PlanOutput):
        objective = plan.task.task
        initial_plan = plan.initial_plan.initial_plan
        parent_id = None
        for step in plan.steps:
            candidate_step_meta = []
            for candidate_step in step.candidate_language_outputs:
                try:
                    messages = candidate_step.conversation.model_dump()['messages']
                except:
                    messages = candidate_step.conversation.dict()['messages']
                output = candidate_step.response
                start_state = step.start_state
                candidate_step_meta.append({'output': output, 'messages': messages})
                mining_setup = candidate_step.meta['mining_setup']
                starting_inventory = candidate_step.meta['starting_inventory']
            judge_messages = step.judge_language_output_step.conversation.model_dump()['messages']
            executor_step = step.final_step
            meta = {'objective': objective, 'initial_plan': initial_plan, 'candidate_steps': candidate_step_meta, 'judge_messages': judge_messages, 'executor_step': executor_step, 'start_state': start_state, 'mining_setup': mining_setup, 'starting_inventory': starting_inventory, 'final_output': plan.final_output}
            program = step.program
            program.meta = meta
            program.parent_id = parent_id
            self.db.create_program(program)
            parent_id = program.id

    async def _process_last_step(self, plan: PlanOutput, start_state: GameState, instance_id: int, parent_id: Optional[int], skip_failures: bool):
        try:
            step_to_process = plan.steps[-1]
            step_to_process, _, entity_list = await self._evaluate_step(step_to_process, start_state, instance_id, parent_id)
            plan.steps[-1] = step_to_process
            log_str = f'Step {len(plan.steps)}: {step_to_process.final_step}\n{step_to_process.program.response}'
            plan.logs.append(log_str)
            return plan
        except Exception as e:
            print(f'Failed to evaluate program on instance {instance_id}: {str(e)}')
            plan.steps.pop()
            return plan

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_natural_language_batch(self, conversation: Conversation, generation_params: GenerationParameters, meta: dict) -> List[LanguageOutput]:
        """Generate multiple programs in a single API call using 'n' parameter"""
        formatted_messages = self.formatter.to_llm_messages(self.formatter.format_conversation(conversation))
        if 'claude' in generation_params.model:
            assert generation_params.n == 1, 'Number of samples must be 1 for Claude'
        try:
            response = self.llm.call(messages=formatted_messages, n_samples=generation_params.n, temperature=generation_params.temperature, max_tokens=generation_params.max_tokens, logit_bias=generation_params.logit_bias, stop_sequences=generation_params.stop_sequences, model=generation_params.model)
            outputs = []
            try:
                messages = conversation.model_dump()['messages']
            except:
                messages = conversation.dict()['messages']
            if 'claude' in generation_params.model:
                str_output = response.content[0].text
                outputs.append(LanguageOutput(id=hash((str_output, json.dumps(messages))), response=str_output, conversation=conversation, token_usage=response.usage.output_tokens + response.usage.input_tokens if hasattr(response, 'usage') else None, completion_token_usage=response.usage.output_tokens if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.input_tokens if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta=meta))
            else:
                for choice in response.choices:
                    str_output = choice.message.content
                    outputs.append(LanguageOutput(id=hash((str_output, json.dumps(messages))), response=str_output, conversation=conversation, token_usage=response.usage.total_tokens // generation_params.n if hasattr(response, 'usage') else None, completion_token_usage=response.usage.completion_tokens // generation_params.n if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.prompt_tokens // generation_params.n if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta=meta))
            return outputs
        except Exception as e:
            print(f'Batch program generation failed: {str(e)}')
            return []

class BacktrackingAgent(AgentABC):

    def __init__(self, model, system_prompt, task, *args, **kwargs):
        backtrack_instructions = GENERAL_INSTRUCTIONS_BACKTRACKING + system_prompt + FINAL_INSTRUCTION
        self.task = task
        backtrack_instructions += f'\n\n### General Goal\n{task.goal_description}\n\n'
        self.instructions = backtrack_instructions
        super().__init__(model, backtrack_instructions, *args, **kwargs)
        self.api_factory = APIFactory(model)
        self.formatter = RecursiveReportFormatter(chunk_size=16, llm_call=self.api_factory.acall, cache_dir='.fle/summary_cache')
        self.generation_params = GenerationParameters(n=1, max_tokens=2048, model=model)
        self.current_step_memory = deque([])
        self.max_nr_of_steps = 8
        self.current_step = 0

    def create_backtracking_conversation(self, conversation: Conversation, namespace: FactorioNamespace, response) -> Conversation:
        system_message = Message(role='system', content=self.instructions)
        new_conversation = Conversation(messages=[system_message])
        new_conversation.messages.extend(conversation.messages[-2:])
        new_conversation.messages[-1].content = f'Last successful step:\n\n{new_conversation.messages[-1].content}\n\n This is the environment state before the error occurred. The environment has not been altered since the last successful step'
        latest_program = f'Original attempt at carrying out the next step:\n\n```python{response.code}```' if len(self.current_step_memory) == 0 else f'Error fixing attempt number {len(self.current_step_memory)}:\n\n```python{response.code}```'
        latest_program_message = Message(role='assistant', content=latest_program, metadata={})
        error_mesasage_str = f'Error message:\n\n{response.response}\n\n NB: This is the error message from the failed attempt. The environment has not been altered since the last successful step'
        error_message = Message(role='user', content=error_mesasage_str, metadata={})
        self.current_step_memory.append({'assistant_message': latest_program_message, 'environment_message': error_message})
        if len(self.current_step_memory) > self.max_nr_of_steps:
            original_attempt = self.current_step_memory.popleft()
            self.current_step_memory.popleft()
            self.current_step_memory.appendleft(original_attempt)
        for step in self.current_step_memory:
            new_conversation.messages.append(step['assistant_message'])
            new_conversation.messages.append(step['environment_message'])
        return new_conversation

    async def step(self, conversation: Conversation, response: Optional[Response], namespace: FactorioNamespace) -> Policy:
        conversation = self.create_backtracking_conversation(conversation, namespace, response)
        temp_conv = Conversation(messages=copy.deepcopy(conversation.messages[3:]))
        temp_conv = await self.formatter.format_conversation(temp_conv, namespace)
        formatted_conversation = Conversation(messages=conversation.messages[:3] + temp_conv.messages)
        return (await self._get_policy(formatted_conversation), None)

    @tenacity.retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _get_policy(self, conversation: Conversation):
        response = await self.api_factory.acall(messages=self.formatter.to_llm_messages(conversation), n_samples=1, temperature=self.generation_params.temperature, max_tokens=self.generation_params.max_tokens, model=self.generation_params.model)
        policy = parse_response(response)
        if not policy:
            raise Exception('Not a valid Python policy')
        policy.input_conversation = conversation
        return policy

    def clear_memory(self):
        self.current_step = 0
        self.current_step_memory = deque([])

    @property
    def memory_full(self):
        return len(self.current_step_memory) >= self.max_nr_of_steps

    async def end(self, conversation: Conversation, completion: CompletionResult):
        pass

class VisualAgent(AgentABC):
    """
    An agent that renders the Factorio map at each step to provide visual context.
    """

    def __init__(self, model, system_prompt, task, render_radius=20, *args, **kwargs):
        """
        Initialize the Visual Agent.

        Args:
            model: The LLM model to use
            system_prompt: System prompt for the agent
            task: The task to perform
            render_radius: Radius around player to render (default: 20)
        """
        instructions = GENERAL_INSTRUCTIONS + system_prompt + FINAL_INSTRUCTION + VISUAL_INSTRUCTIONS
        self.task = task
        instructions += f'\n\n### Goal\n{task.goal_description}\n\n'
        super().__init__(model, instructions, *args, **kwargs)
        self.render_radius = render_radius
        self.api_factory = APIFactory(model)
        self.formatter = RecursiveReportFormatter(chunk_size=16, llm_call=self.api_factory.acall, cache_dir='.fle/summary_cache')
        self.generation_params = GenerationParameters(n=1, max_tokens=2048, model=model)
        self.last_image_base64 = None

    async def step(self, conversation: Conversation, response: Response, namespace: FactorioNamespace) -> Policy:
        """
        Execute a step in the agent's process, rendering the map and incorporating it into the prompt.

        Args:
            conversation: Current conversation state
            response: Last response from the environment
            namespace: Current namespace with variables and functions

        Returns:
            Policy: Next actions to execute
        """
        try:
            render_image = await self._render_map(namespace)
            formatted_conversation = await self.formatter.format_conversation(conversation, namespace)
            if render_image and len(formatted_conversation.messages) > 0:
                for i in range(len(formatted_conversation.messages) - 1, -1, -1):
                    if formatted_conversation.messages[i].role == 'user':
                        original_content = formatted_conversation.messages[i].content
                        formatted_conversation.messages[i].content = [{'type': 'text', 'text': original_content}, {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': render_image}}, {'type': 'text', 'text': f'[Current map view (radius: {self.render_radius}) - Use this visual information to guide your decisions. Be sure to reference to legend to understand what each entity is.]'}]
                        break
            self.set_conversation(formatted_conversation)
            return await self._get_policy(formatted_conversation)
        except Exception as e:
            print(f'Error in visual agent step: {str(e)}')
            formatted_conversation = await self.formatter.format_conversation(conversation, namespace)
            self.set_conversation(formatted_conversation)
            return await self._get_policy(formatted_conversation)

    async def _render_map(self, namespace: FactorioNamespace) -> Optional[str]:
        """
        Render the current map state and convert to base64.

        Args:
            namespace: Current namespace with game state

        Returns:
            str: Base64-encoded image or None if rendering fails
        """
        try:
            player_pos = Position(0, 0)
            if hasattr(namespace, 'PLAYER') and hasattr(namespace.PLAYER, 'position'):
                player_pos = namespace.PLAYER.position
            elif hasattr(namespace, 'player_location'):
                player_pos = namespace.player_location
            render = namespace._render(position=player_pos, layers=Layer.ALL)
            self.last_image_base64 = render.to_base64()
            return self.last_image_base64
        except Exception as e:
            print(f'Error rendering map: {str(e)}')
            return None

    @tenacity.retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _get_policy(self, conversation: Conversation):
        """
        Get the next policy from the LLM.

        Args:
            conversation: Current conversation state

        Returns:
            Policy: Next actions to execute
        """
        response = await self.api_factory.acall(messages=self.formatter.to_llm_messages(conversation), n_samples=1, temperature=self.generation_params.temperature, max_tokens=self.generation_params.max_tokens, model=self.generation_params.model)
        policy = parse_response(response)
        if not policy:
            raise Exception('Not a valid Python policy')
        return policy

    async def end(self, conversation: Conversation, completion: CompletionResult):
        """
        Cleanup when a trajectory ends.

        Args:
            conversation: Final conversation state
            completion: Completion result
        """
        pass

class BasicAgent(AgentABC):

    def __init__(self, model, system_prompt, task, agent_idx: Optional[int]=None, *args, **kwargs):
        instructions = GENERAL_INSTRUCTIONS + system_prompt + FINAL_INSTRUCTION
        self.task = task
        instructions += f'\n\n### Goal\n{task.goal_description}\n\n'
        if agent_idx is not None and task.get_agent_instructions(agent_idx) is not None:
            player_idx = agent_idx + 1
            instructions += f'### Specific Instructions for Agent {player_idx}\n{task.get_agent_instructions(agent_idx)}\n\n'
        super().__init__(model, instructions, *args, **kwargs)
        self.api_factory = APIFactory(model)
        self.formatter = RecursiveReportFormatter(chunk_size=16, llm_call=self.api_factory.acall, cache_dir='.fle/summary_cache')
        self.generation_params = GenerationParameters(n=1, max_tokens=4096, model=model)

    @track_timing_async('agent_step')
    async def step(self, conversation: Conversation, response: Response, namespace: FactorioNamespace) -> Policy:
        async with timing_tracker.track_async('format_conversation'):
            formatted_conversation = await self.formatter.format_conversation(conversation, namespace)
        self.set_conversation(formatted_conversation)
        return (await self._get_policy(formatted_conversation), None)

    @tenacity.retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10))
    @track_timing_async('get_policy')
    async def _get_policy(self, conversation: Conversation):
        async with timing_tracker.track_async('llm_call'):
            messages = self.formatter.to_llm_messages(conversation)
            response = await self.api_factory.acall(messages=messages, n_samples=1, temperature=self.generation_params.temperature, max_tokens=self.generation_params.max_tokens, model=self.generation_params.model)
        async with timing_tracker.track_async('parse_response'):
            policy = parse_response(response)
            if not policy:
                raise Exception('Not a valid Python policy')
            policy.input_conversation = conversation
            return policy

    @track_timing_async('agent_end')
    async def end(self, conversation: Conversation, completion: CompletionResult):
        pass

@pytest.fixture
def basic_conversation():
    """Create a basic conversation object for testing"""
    return Conversation(messages=[Message(role='system', content='You are a helpful assistant'), Message(role='user', content='Hello'), Message(role='assistant', content='Hi there!')])

def test_message_collection_in_conversation(trajectory_runner, mock_config):
    """Test that messages are collected in the conversation"""
    instance = trajectory_runner.evaluator.instance
    instance.namespaces[0].send_message('Hello Agent 1', recipient=1)
    new_messages_text = trajectory_runner._collect_new_messages(1)
    assert 'Hello Agent 1' in new_messages_text
    base_conversation = Conversation(messages=[Message(role='assistant', content='wow what a game'), Message(role='user', content='indeed.')])
    mock_config.agents[1].set_conversation(base_conversation)
    last_user_message = mock_config.agents[1].conversation.messages[-1]
    last_user_message.content += new_messages_text
    assert 'Hello Agent 1' in last_user_message.content

def create_test_conversation(length: int) -> Conversation:
    """Helper to create a test conversation of specified length."""
    messages = [Message(role='system', content='You are a helpful assistant.')]
    for i in range(length):
        messages.extend([Message(role='user', content=f'Message {i}'), Message(role='assistant', content=f'Response {i}')])
    return Conversation(messages=messages)

class TestStructurePreservingFormatter(unittest.TestCase):

    def setUp(self):
        self.formatter = StructurePreservingFormatter(planning=False)
        self.conversation = Conversation(messages=[Message(role='system', content='You are a helpful assistant.'), Message(role='user', content='Inventory: {}'), Message(role='assistant', content='# Gather iron ore\nprint(0)\nprint(1)\n# Construct stone furnace'), Message(role='user', content='Execution result: 1: 0\n2: 1'), Message(role='assistant', content='# Gather more iron ore\nprint(2)\nprint(3)\n# Construct stone furnace'), Message(role='user', content='Execution result: 3: 0\n4: 1')])

    def test_code_extractor(self):
        code_snippets = CodeProcessor.extract_code_blocks(self.conversation.messages[2].content)
        self.assertEqual(len(code_snippets), 1)

    def test_code_extractor_2(self):
        code_snippets = CodeProcessor.extract_code_blocks(self.conversation.messages[2].content)
        self.assertIn('print(0)\nprint(1)', code_snippets[0])

    def test_code_summariser(self):
        code_block = '# Gather iron ore\nprint(0)\nprint(1)\n# Construct stone furnace'
        CodeProcessor.extract_code_blocks(code_block)
        summarized = CodeProcessor.summarize_code_block(code_block, preserve_comments=True)
        self.assertEqual('# Gather iron ore\n<LINES 2-3 CUT/>\n# Construct stone furnace', summarized)
        summarized2 = CodeProcessor.summarize_code_block('# Gather iron ore\n# Gather more iron ore\nprint(0)\nprint(1)\n# Construct stone furnace', preserve_comments=True)
        self.assertEqual(summarized2, '# Gather iron ore\n# Gather more iron ore\n<LINES 3-4 CUT/>\n# Construct stone furnace')

    def test_format_conversation(self):
        formatted = self.formatter.format_conversation(self.conversation)
        self.assertEqual(len(formatted), 6)
        self.assertEqual(formatted[0].role, 'system')
        self.assertEqual(formatted[0].content, 'You are a helpful assistant.')
        assistant1 = formatted[2]
        self.assertEqual(assistant1.role, 'assistant')
        expected_summary = '# Gather iron ore\n<LINES 2-3 CUT/>\n# Construct stone furnace'
        self.assertEqual(assistant1.content, expected_summary)
        assistant2 = formatted[4]
        self.assertEqual(assistant2.role, 'assistant')
        expected_full = '# Gather more iron ore\nprint(2)\nprint(3)\n# Construct stone furnace'
        self.assertEqual(assistant2.content, expected_full)
        self.assertEqual(assistant2.metadata, {'summarized': False})
        user1 = formatted[3]
        self.assertEqual(user1.role, 'user')
        self.assertEqual(user1.content, 'Execution result:\n1: 0\n2: 1')
        user2 = formatted[5]
        self.assertEqual(user2.role, 'user')
        self.assertEqual(user2.content, 'Execution result:\n3: 0\n4: 1')

    def test_format_single_message(self):
        message = Message(role='assistant', content='# First task\nprint(1)\n# Second task\nprint(2)\n')
        formatted = self.formatter.format_message(message, should_format=True)
        expected = '# First task\n<LINE 2 CUT/>\n# Second task\n<LINE 4 CUT/>'
        self.assertEqual(formatted.content, expected)
        self.assertEqual(formatted.metadata, {'summarized': True})
        formatted_last = self.formatter.format_message(message, should_format=False)
        self.assertEqual(formatted_last.content, '# First task\nprint(1)\n# Second task\nprint(2)\n')
        self.assertEqual(formatted_last.metadata, {'summarized': False})

    def test_empty_code_blocks(self):
        message = Message(role='assistant', content='print(1)\nprint(2)\n')
        formatted = self.formatter.format_message(message, should_format=True)
        self.assertEqual(formatted.content, '<LINES 1-2 CUT/>')

    def test_docstring_code_summariser(self):
        code_block = 'from factorio_instance import *\n\n    """\n    Objective: We need to get 20 copper plates\n\n    Planning:\n    1. Print the recipe for copper plates\n    2. Analyze the current game state\n    """\n\n    """\n    Step 1: Print recipe for copper plates\n    """\n    print("Copper Plate Recipe:")\n    print("Crafting requires smelting")\n\n    """\n    Step 2: Analyze current game state\n    """\n    inventory = inspect_inventory()\n    print(f"Current inventory: {inventory}")'
        self.formatter.code_processor.summarize_code_block(code_block, preserve_comments=True)
        pass

class TestRecursiveFormatter(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_llm = Mock(spec=APIFactory)
        self.formatter = RecursiveFormatter(chunk_size=4, api_factory=self.mock_llm, cache_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def create_test_conversation(self, length: int) -> Conversation:
        """Helper to create a test conversation of specified length."""
        messages = [Message(role='system', content='You are a helpful assistant.')]
        for i in range(length):
            messages.extend([Message(role='user', content=f'Message {i}'), Message(role='assistant', content=f'Response {i}')])
        return Conversation(messages=messages)

    def test_chunk_hash_deterministic(self):
        """Test that the same messages produce the same hash."""
        messages = [Message(role='user', content='Hello'), Message(role='assistant', content='Hi')]
        hash1 = self.formatter._get_chunk_hash(messages)
        hash2 = self.formatter._get_chunk_hash(messages)
        self.assertEqual(hash1, hash2)

    def test_cache_operations(self):
        """Test cache save and load operations."""
        messages = [Message(role='user', content='Test message'), Message(role='assistant', content='Test response')]
        chunk_hash = self.formatter._get_chunk_hash(messages)
        summary = Message(role='assistant', content='Summary content', metadata={'summarized': True})
        self.formatter._save_summary_cache(chunk_hash, summary, 1, 2)
        loaded_summary = self.formatter._load_cached_summary(chunk_hash)
        self.assertIsNotNone(loaded_summary)
        self.assertEqual(loaded_summary.content, 'Summary content')
        self.assertEqual(loaded_summary.metadata['summarized'], True)

    async def test_basic_summarization(self):
        """Test basic summarization of a conversation chunk."""
        mock_response = Mock()
        mock_response.content = 'Summarized content'
        self.mock_llm.acall.return_value = mock_response
        messages = [Message(role='user', content=f'Message {i}') for i in range(5)]
        summary = await self.formatter._summarize_chunk(messages, 1, 5)
        self.assertEqual(summary.content, 'Summarized content')
        self.assertTrue(summary.metadata['summarized'])
        self.assertEqual(summary.metadata['summary_range'], '[1-5]')

    async def test_recursive_summarization(self):
        """Test recursive summarization of a longer conversation."""
        mock_response = Mock()
        mock_response.content = 'Summarized chunk'
        self.mock_llm.acall.return_value = mock_response
        conversation = self.create_test_conversation(4)
        formatted = await self.formatter.format_conversation(conversation)
        self.assertGreater(len(formatted), 1)
        self.assertEqual(formatted[0].role, 'system')
        self.assertTrue(any((msg.metadata.get('summarized') for msg in formatted)))

    async def test_very_long_conversation(self):
        """Test handling of a conversation requiring multiple levels of recursion."""
        mock_response = Mock()
        mock_response.content = 'Summarized content'
        self.mock_llm.acall.return_value = mock_response
        conversation = self.create_test_conversation(16)
        formatted = await self.formatter.format_conversation(conversation)
        self.assertLess(len(formatted), len(conversation.messages))
        summaries = [msg for msg in formatted if msg.metadata.get('summarized')]
        self.assertGreater(len(summaries), 0)

    def test_error_handling(self):
        """Test error handling in cache operations."""
        with patch('builtins.open', side_effect=IOError):
            messages = [Message(role='user', content='Test')]
            chunk_hash = self.formatter._get_chunk_hash(messages)
            summary = Message(role='assistant', content='Summary')
            self.formatter._save_summary_cache(chunk_hash, summary, 1, 1)
            loaded = self.formatter._load_cached_summary(chunk_hash)
            self.assertIsNone(loaded)

