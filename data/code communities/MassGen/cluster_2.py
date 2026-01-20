# Cluster 2

def save_execution_metadata(query: str, config_path: Optional[str]=None, config_content: Optional[dict]=None):
    """Save the query and config metadata to the log directory.

    This allows reconstructing what was executed in this session.

    Args:
        query: The user's query/prompt
        config_path: Path to the config file that was used (optional)
        config_content: The actual config dictionary (optional)
    """
    import yaml
    log_dir = get_log_session_dir()
    metadata = {'query': query, 'timestamp': datetime.now().isoformat()}
    if config_path:
        metadata['config_path'] = str(config_path)
    if config_content:
        metadata['config'] = config_content
    metadata_file = log_dir / 'execution_metadata.yaml'
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info(f'Saved execution metadata to: {metadata_file}')
    except Exception as e:
        logger.warning(f'Failed to save execution metadata: {e}')

def get_log_session_dir(turn: Optional[int]=None) -> Path:
    """Get the current log session directory.

    Args:
        turn: Optional turn number for multi-turn conversations

    Returns:
        Path to the log directory
    """
    global _LOG_SESSION_DIR, _LOG_BASE_SESSION_DIR, _CURRENT_TURN
    if _LOG_BASE_SESSION_DIR is None:
        cwd = Path.cwd()
        pyproject_file = cwd / 'pyproject.toml'
        if pyproject_file.exists():
            try:
                content = pyproject_file.read_text()
                if 'name = "massgen"' in content:
                    pass
            except Exception:
                pass
        log_base_dir = Path('.massgen') / 'massgen_logs'
        log_base_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        _LOG_BASE_SESSION_DIR = log_base_dir / f'log_{timestamp}'
        _LOG_BASE_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    if turn is not None and turn != _CURRENT_TURN:
        _CURRENT_TURN = turn
        _LOG_SESSION_DIR = None
    if _LOG_SESSION_DIR is None:
        if _CURRENT_TURN and _CURRENT_TURN > 0:
            _LOG_SESSION_DIR = _LOG_BASE_SESSION_DIR / f'turn_{_CURRENT_TURN}'
        else:
            _LOG_SESSION_DIR = _LOG_BASE_SESSION_DIR
        _LOG_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_SESSION_DIR

class Orchestrator(ChatAgent):
    """
    Orchestrator Agent - Unified chat interface with sub-agent coordination.

    The orchestrator acts as a single agent from the user's perspective, but internally
    coordinates multiple sub-agents using the proven binary decision framework.

    Key Features:
    - Unified chat interface (same as any individual agent)
    - Automatic sub-agent coordination and conflict resolution
    - Transparent MassGen workflow execution
    - Real-time streaming with proper source attribution
    - Graceful restart mechanism for dynamic case transitions
    - Session management

    TODO - Missing Configuration Options:
    - Option to include/exclude voting details in user messages
    - Configurable timeout settings for agent responses
    - Configurable retry limits and backoff strategies
    - Custom voting strategies beyond simple majority
    - Configurable presentation formats for final answers
    - Advanced coordination workflows (hierarchical, weighted voting, etc.)

    TODO (v0.0.14 Context Sharing Enhancement - See docs/dev_notes/v0.0.14-context.md):
    - Add permission validation logic for agent workspace access
    - Implement validate_agent_access() method to check if agent has required permission for resource
    - Replace current prompt-based access control with explicit system-level enforcement
    - Add PermissionManager integration for managing agent access rules
    - Implement audit logging for all access attempts to workspace resources
    - Support dynamic permission negotiation during runtime
    - Add configurable policy framework for permission management
    - Integrate with workspace snapshot mechanism for controlled context sharing

    Restart Behavior:
    When an agent provides new_answer, all agents gracefully restart to ensure
    consistent coordination state. This allows all agents to transition to Case 2
    evaluation with the new answers available.
    """

    def __init__(self, agents: Dict[str, ChatAgent], orchestrator_id: str='orchestrator', session_id: Optional[str]=None, config: Optional[AgentConfig]=None, snapshot_storage: Optional[str]=None, agent_temporary_workspace: Optional[str]=None, previous_turns: Optional[List[Dict[str, Any]]]=None):
        """
        Initialize MassGen orchestrator.

        Args:
            agents: Dictionary of {agent_id: ChatAgent} - can be individual agents or other orchestrators
            orchestrator_id: Unique identifier for this orchestrator (default: "orchestrator")
            session_id: Optional session identifier
            config: Optional AgentConfig for customizing orchestrator behavior
            snapshot_storage: Optional path to store agent workspace snapshots
            agent_temporary_workspace: Optional path for agent temporary workspaces
            previous_turns: List of previous turn metadata for multi-turn conversations (loaded by CLI)
        """
        super().__init__(session_id)
        self.orchestrator_id = orchestrator_id
        self.agents = agents
        self.agent_states = {aid: AgentState() for aid in agents.keys()}
        self.config = config or AgentConfig.create_openai_config()
        self.message_templates = self.config.message_templates or MessageTemplates()
        self.workflow_tools = self.message_templates.get_standard_tools(list(agents.keys()))
        self.current_task: Optional[str] = None
        self.workflow_phase: str = 'idle'
        self._coordination_messages: List[Dict[str, str]] = []
        self._selected_agent: Optional[str] = None
        self._final_presentation_content: Optional[str] = None
        self.total_tokens: int = 0
        self.coordination_start_time: float = 0
        self.is_orchestrator_timeout: bool = False
        self.timeout_reason: Optional[str] = None
        self._active_streams: Dict = {}
        self._active_tasks: Dict = {}
        self._snapshot_storage: Optional[str] = snapshot_storage
        self._agent_temporary_workspace: Optional[str] = agent_temporary_workspace
        self._previous_turns: List[Dict[str, Any]] = previous_turns or []
        self.coordination_tracker = CoordinationTracker()
        self.coordination_tracker.initialize_session(list(agents.keys()))
        if snapshot_storage:
            self._snapshot_storage = snapshot_storage
            snapshot_path = Path(self._snapshot_storage)
            if snapshot_path.exists() and any(snapshot_path.iterdir()):
                shutil.rmtree(snapshot_path)
            snapshot_path.mkdir(parents=True, exist_ok=True)
        for agent_id, agent in self.agents.items():
            if agent.backend.filesystem_manager:
                agent.backend.filesystem_manager.setup_orchestration_paths(agent_id=agent_id, snapshot_storage=self._snapshot_storage, agent_temporary_workspace=self._agent_temporary_workspace)
                agent.backend.filesystem_manager.update_backend_mcp_config(agent.backend.config)

    @staticmethod
    def _get_chunk_type_value(chunk) -> str:
        """
        Extract chunk type as string, handling both legacy and typed chunks.

        Args:
            chunk: StreamChunk, TextStreamChunk, or MultimodalStreamChunk

        Returns:
            String representation of chunk type (e.g., "content", "tool_calls")
        """
        chunk_type = chunk.type
        if isinstance(chunk_type, ChunkType):
            return chunk_type.value
        return str(chunk_type)

    async def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]=None, reset_chat: bool=False, clear_history: bool=False) -> AsyncGenerator[StreamChunk, None]:
        """
        Main chat interface - handles user messages and coordinates sub-agents.

        Args:
            messages: List of conversation messages
            tools: Ignored by orchestrator (uses internal workflow tools)
            reset_chat: If True, reset conversation and start fresh
            clear_history: If True, clear history before processing

        Yields:
            StreamChunk: Streaming response chunks
        """
        _ = tools
        if clear_history:
            self.conversation_history.clear()
        if reset_chat:
            self.reset()
        conversation_context = self._build_conversation_context(messages)
        user_message = conversation_context.get('current_message')
        if not user_message:
            log_stream_chunk('orchestrator', 'error', 'No user message found in conversation')
            yield StreamChunk(type='error', error='No user message found in conversation')
            return
        self.add_to_history('user', user_message)
        if self.workflow_phase == 'idle':
            self.current_task = user_message
            self.coordination_tracker.initialize_session(list(self.agents.keys()), self.current_task)
            self.workflow_phase = 'coordinating'
            if conversation_context and conversation_context.get('conversation_history'):
                self._clear_agent_workspaces()
            async for chunk in self._coordinate_agents_with_timeout(conversation_context):
                yield chunk
        elif self.workflow_phase == 'presenting':
            async for chunk in self._handle_followup(user_message, conversation_context):
                yield chunk
        else:
            log_stream_chunk('orchestrator', 'content', '🔄 Coordinating agents, please wait...')
            yield StreamChunk(type='content', content='🔄 Coordinating agents, please wait...')

    async def chat_simple(self, user_message: str) -> AsyncGenerator[StreamChunk, None]:
        """
        Backwards compatible simple chat interface.

        Args:
            user_message: Simple string message from user

        Yields:
            StreamChunk: Streaming response chunks
        """
        messages = [{'role': 'user', 'content': user_message}]
        async for chunk in self.chat(messages):
            yield chunk

    def _build_conversation_context(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build conversation context from message list."""
        conversation_history = []
        current_message = None
        for message in messages:
            role = message.get('role')
            content = message.get('content', '')
            if role == 'user':
                current_message = content
                if len(conversation_history) > 0 or len(messages) > 1:
                    conversation_history.append(message.copy())
            elif role == 'assistant':
                conversation_history.append(message.copy())
            elif role == 'system':
                pass
        if conversation_history and conversation_history[-1].get('role') == 'user':
            conversation_history.pop()
        return {'current_message': current_message, 'conversation_history': conversation_history, 'full_messages': messages}

    def save_coordination_logs(self):
        """Public method to save coordination logs after final presentation is complete."""
        self.coordination_tracker._end_session()
        log_session_dir = get_log_session_dir()
        if log_session_dir:
            self.coordination_tracker.save_coordination_logs(log_session_dir)

    async def _coordinate_agents_with_timeout(self, conversation_context: Optional[Dict[str, Any]]=None) -> AsyncGenerator[StreamChunk, None]:
        """Execute coordination with orchestrator-level timeout protection."""
        self.coordination_start_time = time.time()
        self.total_tokens = 0
        self.is_orchestrator_timeout = False
        self.timeout_reason = None
        log_orchestrator_activity(self.orchestrator_id, 'Starting coordination with timeout', {'timeout_seconds': self.config.timeout_config.orchestrator_timeout_seconds, 'agents': list(self.agents.keys())})
        self._active_streams = {}
        self._active_tasks = {}
        timeout_seconds = self.config.timeout_config.orchestrator_timeout_seconds
        try:
            async with asyncio.timeout(timeout_seconds):
                async for chunk in self._coordinate_agents(conversation_context):
                    if hasattr(chunk, 'content') and chunk.content:
                        self.total_tokens += len(chunk.content.split())
                    yield chunk
        except asyncio.TimeoutError:
            self.is_orchestrator_timeout = True
            elapsed = time.time() - self.coordination_start_time
            self.timeout_reason = f'Time limit exceeded ({elapsed:.1f}s/{timeout_seconds}s)'
            for agent_id in self.agent_states.keys():
                if not self.agent_states[agent_id].has_voted:
                    self.coordination_tracker.track_agent_action(agent_id, ActionType.TIMEOUT, self.timeout_reason)
            await self._cleanup_active_coordination()
        if self.is_orchestrator_timeout:
            async for chunk in self._handle_orchestrator_timeout():
                yield chunk

    async def _coordinate_agents(self, conversation_context: Optional[Dict[str, Any]]=None) -> AsyncGenerator[StreamChunk, None]:
        """Execute unified MassGen coordination workflow with real-time streaming."""
        log_coordination_step('Starting multi-agent coordination', {'agents': list(self.agents.keys()), 'has_context': conversation_context is not None})
        if self.config.skip_coordination_rounds:
            log_stream_chunk('orchestrator', 'content', '⚡ [DEBUG MODE] Skipping coordination rounds, going straight to final presentation...\n\n', self.orchestrator_id)
            yield StreamChunk(type='content', content='⚡ [DEBUG MODE] Skipping coordination rounds, going straight to final presentation...\n\n', source=self.orchestrator_id)
            self._selected_agent = list(self.agents.keys())[0]
            log_coordination_step('Skipped coordination, selected first agent', {'selected_agent': self._selected_agent})
            async for chunk in self._present_final_answer():
                yield chunk
            return
        log_stream_chunk('orchestrator', 'content', '🚀 Starting multi-agent coordination...\n\n', self.orchestrator_id)
        yield StreamChunk(type='content', content='🚀 Starting multi-agent coordination...\n\n', source=self.orchestrator_id)
        votes = {}
        for agent_id in self.agents.keys():
            self.agent_states[agent_id].has_voted = False
            self.agent_states[agent_id].restart_pending = True
        log_stream_chunk('orchestrator', 'content', '## 📋 Agents Coordinating\n', self.orchestrator_id)
        yield StreamChunk(type='content', content='## 📋 Agents Coordinating\n', source=self.orchestrator_id)
        async for chunk in self._stream_coordination_with_agents(votes, conversation_context):
            yield chunk
        current_answers = {aid: state.answer for aid, state in self.agent_states.items() if state.answer}
        self._selected_agent = self._determine_final_agent_from_votes(votes, current_answers)
        log_coordination_step('Final agent selected', {'selected_agent': self._selected_agent, 'votes': votes})
        async for chunk in self._present_final_answer():
            yield chunk

    async def _stream_coordination_with_agents(self, votes: Dict[str, Dict], conversation_context: Optional[Dict[str, Any]]=None) -> AsyncGenerator[StreamChunk, None]:
        """
        Coordinate agents with real-time streaming of their outputs.

        Processes agent stream signals:
        - "content": Streams real-time agent output to user
        - "result": Records votes/answers, triggers restart_pending for other agents
        - "error": Displays error and closes agent stream (self-terminating)
        - "done": Closes agent stream gracefully

        Restart Mechanism:
        When any agent provides new_answer, all other agents get restart_pending=True
        and gracefully terminate their current work before restarting.
        """
        active_streams = {}
        active_tasks = {}
        self._active_streams = active_streams
        self._active_tasks = active_tasks
        while not all((state.has_voted for state in self.agent_states.values())):
            self.coordination_tracker.start_new_iteration()
            if self.is_orchestrator_timeout:
                break
            current_answers = {aid: state.answer for aid, state in self.agent_states.items() if state.answer}
            for agent_id in self.agents.keys():
                if agent_id not in active_streams and (not self.agent_states[agent_id].has_voted) and (not self.agent_states[agent_id].is_killed):
                    active_streams[agent_id] = self._stream_agent_execution(agent_id, self.current_task, current_answers, conversation_context)
            if not active_streams:
                break
            for agent_id, stream in active_streams.items():
                if agent_id not in active_tasks:
                    active_tasks[agent_id] = asyncio.create_task(self._get_next_chunk(stream))
            if not active_tasks:
                break
            done, _ = await asyncio.wait(active_tasks.values(), return_when=asyncio.FIRST_COMPLETED)
            reset_signal = False
            voted_agents = {}
            answered_agents = {}
            completed_agent_ids = set()
            for task in done:
                agent_id = next((aid for aid, t in active_tasks.items() if t is task))
                del active_tasks[agent_id]
                try:
                    chunk_type, chunk_data = await task
                    if chunk_type == 'content':
                        log_stream_chunk('orchestrator', 'content', chunk_data, agent_id)
                        yield StreamChunk(type='content', content=chunk_data, source=agent_id)
                    elif chunk_type == 'reasoning':
                        log_stream_chunk('orchestrator', 'reasoning', chunk_data, agent_id)
                        yield chunk_data
                    elif chunk_type == 'result':
                        result_type, result_data = chunk_data
                        completed_agent_ids.add(agent_id)
                        log_stream_chunk('orchestrator', f'result.{result_type}', result_data, agent_id)
                        yield StreamChunk(type='agent_status', source=agent_id, status='completed', content='')
                        await self._close_agent_stream(agent_id, active_streams)
                        if result_type == 'answer':
                            agent = self.agents.get(agent_id)
                            agent_context = self.get_last_context(agent_id)
                            answer_timestamp = await self._save_agent_snapshot(agent_id, answer_content=result_data, context_data=agent_context)
                            if agent and agent.backend.filesystem_manager:
                                agent.backend.filesystem_manager.log_current_state('after providing answer')
                            answered_agents[agent_id] = result_data
                            self.coordination_tracker.add_agent_answer(agent_id, result_data, snapshot_timestamp=answer_timestamp)
                            restart_triggered_id = agent_id
                            reset_signal = True
                            log_stream_chunk('orchestrator', 'content', '✅ Answer provided\n', agent_id)
                            log_stream_chunk('orchestrator', 'content', '✅ Answer provided\n', agent_id)
                            yield StreamChunk(type='content', content='✅ Answer provided\n', source=agent_id)
                        elif result_type == 'vote':
                            if self._check_restart_pending(agent_id):
                                voted_for = result_data.get('agent_id', '<unknown>')
                                reason = result_data.get('reason', 'No reason provided')
                                self.coordination_tracker.track_agent_action(agent_id, ActionType.VOTE_IGNORED, f'Voted for {voted_for} but ignored due to restart')
                                log_stream_chunk('orchestrator', 'content', f'🔄 Vote for [{voted_for}] ignored (reason: {reason}) - restarting due to new answers', agent_id)
                                yield StreamChunk(type='content', content=f'🔄 Vote for [{voted_for}] ignored (reason: {reason}) - restarting due to new answers', source=agent_id)
                            else:
                                vote_timestamp = await self._save_agent_snapshot(agent_id=agent_id, vote_data=result_data, context_data=self.get_last_context(agent_id))
                                agent = self.agents.get(agent_id)
                                if agent and agent.backend.filesystem_manager:
                                    self.agents.get(agent_id).backend.filesystem_manager.log_current_state('after voting')
                                voted_agents[agent_id] = result_data
                                self.coordination_tracker.add_agent_vote(agent_id, result_data, snapshot_timestamp=vote_timestamp)
                                voted_for = result_data.get('agent_id', '<unknown>')
                                reason = result_data.get('reason', 'No reason provided')
                                log_stream_chunk('orchestrator', 'content', f'✅ Vote recorded for [{result_data['agent_id']}]', agent_id)
                                yield StreamChunk(type='content', content=f'✅ Vote recorded for [{result_data['agent_id']}]', source=agent_id)
                    elif chunk_type == 'error':
                        self.coordination_tracker.track_agent_action(agent_id, ActionType.ERROR, chunk_data)
                        completed_agent_ids.add(agent_id)
                        log_stream_chunk('orchestrator', 'error', chunk_data, agent_id)
                        yield StreamChunk(type='content', content=f'❌ {chunk_data}', source=agent_id)
                        log_stream_chunk('orchestrator', 'agent_status', 'completed', agent_id)
                        yield StreamChunk(type='agent_status', source=agent_id, status='completed', content='')
                        await self._close_agent_stream(agent_id, active_streams)
                    elif chunk_type == 'debug':
                        log_stream_chunk('orchestrator', 'debug', chunk_data, agent_id)
                        yield StreamChunk(type='debug', content=chunk_data, source=agent_id)
                    elif chunk_type == 'mcp_status':
                        mcp_message = f'🔧 MCP: {chunk_data}'
                        log_stream_chunk('orchestrator', 'mcp_status', chunk_data, agent_id)
                        yield StreamChunk(type='content', content=mcp_message, source=agent_id)
                    elif chunk_type == 'done':
                        completed_agent_ids.add(agent_id)
                        log_stream_chunk('orchestrator', 'done', None, agent_id)
                        yield StreamChunk(type='agent_status', source=agent_id, status='completed', content='')
                        await self._close_agent_stream(agent_id, active_streams)
                except Exception as e:
                    self.coordination_tracker.track_agent_action(agent_id, ActionType.ERROR, f'Stream error - {e}')
                    completed_agent_ids.add(agent_id)
                    log_stream_chunk('orchestrator', 'error', f'❌ Stream error - {e}', agent_id)
                    yield StreamChunk(type='content', content=f'❌ Stream error - {e}', source=agent_id)
                    await self._close_agent_stream(agent_id, active_streams)
            if reset_signal:
                for state in self.agent_states.values():
                    state.has_voted = False
                votes.clear()
                for agent_id in self.agent_states.keys():
                    self.agent_states[agent_id].restart_pending = True
                self.coordination_tracker.track_restart_signal(restart_triggered_id, list(self.agent_states.keys()))
                self.coordination_tracker.complete_agent_restart(restart_triggered_id)
            else:
                for agent_id, vote_data in voted_agents.items():
                    self.agent_states[agent_id].has_voted = True
                    votes[agent_id] = vote_data
            for agent_id, answer in answered_agents.items():
                self.agent_states[agent_id].answer = answer
            for agent_id in completed_agent_ids:
                if agent_id in answered_agents:
                    self.coordination_tracker.change_status(agent_id, AgentStatus.ANSWERED)
                elif agent_id in voted_agents:
                    self.coordination_tracker.change_status(agent_id, AgentStatus.VOTED)
        for agent_id, task in active_tasks.items():
            if not task.done():
                self.coordination_tracker.track_agent_action(agent_id, ActionType.CANCELLED, 'All agents voted - coordination complete')
            task.cancel()
        for agent_id in list(active_streams.keys()):
            await self._close_agent_stream(agent_id, active_streams)

    async def _copy_all_snapshots_to_temp_workspace(self, agent_id: str) -> Optional[str]:
        """Copy all agents' latest workspace snapshots to a temporary workspace for context sharing.

        TODO (v0.0.14 Context Sharing Enhancement - See docs/dev_notes/v0.0.14-context.md):
        - Validate agent permissions before restoring snapshots
        - Check if agent has read access to other agents' workspaces
        - Implement fine-grained control over which snapshots can be accessed
        - Add audit logging for snapshot access attempts

        Args:
            agent_id: ID of the Claude Code agent receiving the context

        Returns:
            Path to the agent's workspace directory if successful, None otherwise
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return None
        if not agent.backend.filesystem_manager:
            return None
        agent_mapping = {}
        sorted_agent_ids = sorted(self.agents.keys())
        for i, real_agent_id in enumerate(sorted_agent_ids, 1):
            agent_mapping[real_agent_id] = f'agent{i}'
        all_snapshots = {}
        if self._snapshot_storage:
            snapshot_base = Path(self._snapshot_storage)
            for source_agent_id in self.agents.keys():
                source_snapshot = snapshot_base / source_agent_id
                if source_snapshot.exists() and source_snapshot.is_dir():
                    all_snapshots[source_agent_id] = source_snapshot
        workspace_path = await agent.backend.filesystem_manager.copy_snapshots_to_temp_workspace(all_snapshots, agent_mapping)
        return str(workspace_path) if workspace_path else None

    async def _save_agent_snapshot(self, agent_id: str, answer_content: str=None, vote_data: Dict[str, Any]=None, is_final: bool=False, context_data: Any=None) -> str:
        """
        Save a snapshot of an agent's working directory and answer/vote with the same timestamp.

        Creates a timestamped directory structure:
        - agent_id/timestamp/workspace/ - Contains the workspace files
        - agent_id/timestamp/answer.txt - Contains the answer text (if provided)
        - agent_id/timestamp/vote.json - Contains the vote data (if provided)
        - agent_id/timestamp/context.txt - Contains the context used (if provided)

        Args:
            agent_id: ID of the agent
            answer_content: The answer content to save (if provided)
            vote_data: The vote data to save (if provided)
            is_final: If True, save as final snapshot for presentation
            context_data: The context data to save (conversation, answers, etc.)

        Returns:
            The timestamp used for this snapshot
        """
        logger.info(f'[Orchestrator._save_agent_snapshot] Called for agent_id={agent_id}, has_answer={bool(answer_content)}, has_vote={bool(vote_data)}, is_final={is_final}')
        agent = self.agents.get(agent_id)
        if not agent:
            logger.warning(f'[Orchestrator._save_agent_snapshot] Agent {agent_id} not found in agents dict')
            return None
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        if answer_content:
            try:
                log_session_dir = get_log_session_dir()
                if log_session_dir:
                    if is_final:
                        timestamped_dir = log_session_dir / 'final' / agent_id
                    else:
                        timestamped_dir = log_session_dir / agent_id / timestamp
                    timestamped_dir.mkdir(parents=True, exist_ok=True)
                    answer_file = timestamped_dir / 'answer.txt'
                    answer_file.write_text(answer_content)
                    logger.info(f'[Orchestrator._save_agent_snapshot] Saved answer to {answer_file}')
            except Exception as e:
                logger.warning(f'[Orchestrator._save_agent_snapshot] Failed to save answer for {agent_id}: {e}')
        if vote_data:
            try:
                log_session_dir = get_log_session_dir()
                if log_session_dir:
                    timestamped_dir = log_session_dir / agent_id / timestamp
                    timestamped_dir.mkdir(parents=True, exist_ok=True)
                    vote_file = timestamped_dir / 'vote.json'
                    current_answers = {aid: state.answer for aid, state in self.agent_states.items() if state.answer}
                    agent_mapping = {}
                    for i, real_id in enumerate(sorted(self.agents.keys()), 1):
                        agent_mapping[f'agent{i}'] = real_id
                    comprehensive_vote_data = {'voter_id': agent_id, 'voter_anon_id': next((anon for anon, real in agent_mapping.items() if real == agent_id), agent_id), 'voted_for': vote_data.get('agent_id', 'unknown'), 'voted_for_anon': next((anon for anon, real in agent_mapping.items() if real == vote_data.get('agent_id')), 'unknown'), 'reason': vote_data.get('reason', ''), 'timestamp': timestamp, 'unix_timestamp': time.time(), 'iteration': self.coordination_tracker.current_iteration if self.coordination_tracker else None, 'coordination_round': self.coordination_tracker.max_round if self.coordination_tracker else None, 'available_options': list(current_answers.keys()), 'available_options_anon': [next((anon for anon, real in agent_mapping.items() if real == aid), aid) for aid in sorted(current_answers.keys())], 'agent_mapping': agent_mapping, 'vote_context': {'total_agents': len(self.agents), 'agents_with_answers': len(current_answers), 'current_task': self.current_task}}
                    with open(vote_file, 'w', encoding='utf-8') as f:
                        json.dump(comprehensive_vote_data, f, indent=2)
                    logger.info(f'[Orchestrator._save_agent_snapshot] Saved comprehensive vote to {vote_file}')
            except Exception as e:
                logger.error(f'[Orchestrator._save_agent_snapshot] Failed to save vote for {agent_id}: {e}')
                logger.error(f'[Orchestrator._save_agent_snapshot] Traceback: {traceback.format_exc()}')
        if agent.backend.filesystem_manager:
            logger.info(f'[Orchestrator._save_agent_snapshot] Agent {agent_id} has filesystem_manager, calling save_snapshot with timestamp={(timestamp if not is_final else None)}')
            await agent.backend.filesystem_manager.save_snapshot(timestamp=timestamp if not is_final else None, is_final=is_final)
            if not is_final:
                agent.backend.filesystem_manager.clear_workspace()
                logger.info(f'[Orchestrator._save_agent_snapshot] Cleared workspace for {agent_id} after saving snapshot')
        else:
            logger.info(f'[Orchestrator._save_agent_snapshot] Agent {agent_id} does not have filesystem_manager')
        if context_data and (answer_content or vote_data):
            try:
                log_session_dir = get_log_session_dir()
                if log_session_dir:
                    if is_final:
                        timestamped_dir = log_session_dir / 'final' / agent_id
                    else:
                        timestamped_dir = log_session_dir / agent_id / timestamp
                    context_file = timestamped_dir / 'context.txt'
                    if isinstance(context_data, dict):
                        context_file.write_text(json.dumps(context_data, indent=2, default=str))
                    else:
                        context_file.write_text(str(context_data))
                    logger.info(f'[Orchestrator._save_agent_snapshot] Saved context to {context_file}')
            except Exception as ce:
                logger.warning(f'[Orchestrator._save_agent_snapshot] Failed to save context for {agent_id}: {ce}')
        return timestamp if not is_final else 'final'

    def get_last_context(self, agent_id: str) -> Any:
        """Get the last context for an agent, or None if not available."""
        return self.agent_states[agent_id].last_context if agent_id in self.agent_states else None

    async def _close_agent_stream(self, agent_id: str, active_streams: Dict[str, AsyncGenerator]) -> None:
        """Close and remove an agent stream safely."""
        if agent_id in active_streams:
            try:
                await active_streams[agent_id].aclose()
            except Exception:
                pass
            del active_streams[agent_id]

    def _check_restart_pending(self, agent_id: str) -> bool:
        """Check if agent should restart and yield restart message if needed. This will always be called when exiting out of _stream_agent_execution()."""
        restart_pending = self.agent_states[agent_id].restart_pending
        return restart_pending

    async def _save_partial_work_on_restart(self, agent_id: str) -> None:
        """
        Save partial work snapshot when agent is restarting due to new answers from others.
        This ensures that any work done before the restart is preserved and shared with other agents.

        Args:
            agent_id: ID of the agent being restarted
        """
        agent = self.agents.get(agent_id)
        if not agent or not agent.backend.filesystem_manager:
            return
        logger.info(f'[Orchestrator._save_partial_work_on_restart] Saving partial work for {agent_id} before restart')
        await self._save_agent_snapshot(agent_id, answer_content=None, context_data=self.get_last_context(agent_id), is_final=False)
        agent.backend.filesystem_manager.log_current_state('after saving partial work on restart')

    def _normalize_workspace_paths_in_answers(self, answers: Dict[str, str], viewing_agent_id: Optional[str]=None) -> Dict[str, str]:
        """Normalize absolute workspace paths in agent answers to accessible temporary workspace paths.

        This addresses the issue where agents working in separate workspace directories
        reference the same logical files using different absolute paths, causing them
        to think they're working on different tasks when voting.

        Converts workspace paths to temporary workspace paths where the viewing agent can actually
        access other agents' files for verification during context sharing.

        TODO: Replace with Docker volume mounts to ensure consistent paths across agents.

        Args:
            answers: Dict mapping agent_id to their answer content
            viewing_agent_id: The agent who will be reading these answers.
                            If None, normalizes to generic "workspace/" prefix.

        Returns:
            Dict with same keys but normalized answer content with accessible paths
        """
        normalized_answers = {}
        temp_workspace_base = None
        if viewing_agent_id:
            viewing_agent = self.agents.get(viewing_agent_id)
            if viewing_agent and viewing_agent.backend.filesystem_manager:
                temp_workspace_base = str(viewing_agent.backend.filesystem_manager.agent_temporary_workspace)
        agent_mapping = {}
        sorted_agent_ids = sorted(self.agents.keys())
        for i, real_agent_id in enumerate(sorted_agent_ids, 1):
            agent_mapping[real_agent_id] = f'agent{i}'
        for agent_id, answer in answers.items():
            normalized_answer = answer
            for other_agent_id, other_agent in self.agents.items():
                if not other_agent.backend.filesystem_manager:
                    continue
                anon_agent_id = agent_mapping.get(other_agent_id, f'agent_{other_agent_id}')
                replace_path = os.path.join(temp_workspace_base, anon_agent_id) if temp_workspace_base else anon_agent_id
                other_workspace = str(other_agent.backend.filesystem_manager.get_current_workspace())
                logger.debug(f'[Orchestrator._normalize_workspace_paths_in_answers] Replacing {other_workspace} in answer from {agent_id} with path {replace_path}. original answer: {normalized_answer}')
                normalized_answer = normalized_answer.replace(other_workspace, replace_path)
                logger.debug(f'[Orchestrator._normalize_workspace_paths_in_answers] Intermediate normalized answer: {normalized_answer}')
            normalized_answers[agent_id] = normalized_answer
        return normalized_answers

    def _normalize_workspace_paths_for_comparison(self, content: str, replacement_path: str='/workspace') -> str:
        """
        Normalize all workspace paths in content to a canonical form for equality comparison.

        Unlike _normalize_workspace_paths_in_answers which normalizes paths for specific agents,
        this method normalizes ALL workspace paths to a neutral canonical form (like '/workspace')
        so that content can be compared for equality regardless of which agent workspace it came from.

        Args:
            content: Content that may contain workspace paths

        Returns:
            Content with all workspace paths normalized to canonical form
        """
        normalized_content = content
        for agent_id, agent in self.agents.items():
            if not agent.backend.filesystem_manager:
                continue
            workspace_path = str(agent.backend.filesystem_manager.get_current_workspace())
            normalized_content = normalized_content.replace(workspace_path, replacement_path)
        return normalized_content

    async def _cleanup_active_coordination(self) -> None:
        """Force cleanup of active coordination streams and tasks on timeout."""
        if hasattr(self, '_active_tasks') and self._active_tasks:
            for agent_id, task in self._active_tasks.items():
                if not task.done():
                    if not self.is_orchestrator_timeout:
                        self.coordination_tracker.track_agent_action(agent_id, ActionType.CANCELLED, 'Coordination cleanup')
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
            self._active_tasks.clear()
        if hasattr(self, '_active_streams') and self._active_streams:
            for agent_id in list(self._active_streams.keys()):
                await self._close_agent_stream(agent_id, self._active_streams)

    def _create_tool_error_messages(self, agent: 'ChatAgent', tool_calls: List[Dict[str, Any]], primary_error_msg: str, secondary_error_msg: str=None) -> List[Dict[str, Any]]:
        """
        Create tool error messages for all tool calls in a response.

        Args:
            agent: The ChatAgent instance for backend access
            tool_calls: List of tool calls that need error responses
            primary_error_msg: Error message for the first tool call
            secondary_error_msg: Error message for additional tool calls (defaults to primary_error_msg)

        Returns:
            List of tool result messages that can be sent back to the agent
        """
        if not tool_calls:
            return []
        if secondary_error_msg is None:
            secondary_error_msg = primary_error_msg
        enforcement_msgs = []
        first_tool_call = tool_calls[0]
        error_result_msg = agent.backend.create_tool_result_message(first_tool_call, primary_error_msg)
        enforcement_msgs.append(error_result_msg)
        for additional_tool_call in tool_calls[1:]:
            neutral_msg = agent.backend.create_tool_result_message(additional_tool_call, secondary_error_msg)
            enforcement_msgs.append(neutral_msg)
        return enforcement_msgs

    async def _stream_agent_execution(self, agent_id: str, task: str, answers: Dict[str, str], conversation_context: Optional[Dict[str, Any]]=None) -> AsyncGenerator[tuple, None]:
        """
        Stream agent execution with real-time content and final result.

        Yields:
            ("content", str): Real-time agent output (source attribution added by caller)
            ("result", (type, data)): Final result - ("vote", vote_data) or ("answer", content)
            ("error", str): Error message (self-terminating)
            ("done", None): Graceful completion signal

        Restart Behavior:
            If restart_pending is True, agent gracefully terminates with "done" signal.
            restart_pending is cleared at the beginning of execution.
        """
        agent = self.agents[agent_id]
        backend_name = None
        if hasattr(agent, 'backend') and hasattr(agent.backend, 'get_provider_name'):
            backend_name = agent.backend.get_provider_name()
        log_orchestrator_activity(self.orchestrator_id, f'Starting agent execution: {agent_id}', {'agent_id': agent_id, 'backend': backend_name, 'task': task if task else None, 'has_answers': bool(answers), 'num_answers': len(answers) if answers else 0})
        logger.info(f'[Orchestrator] Agent {agent_id} starting execution loop...')
        self.agent_states[agent_id].is_killed = False
        self.agent_states[agent_id].timeout_reason = None
        if self.agent_states[agent_id].restart_pending:
            self.coordination_tracker.complete_agent_restart(agent_id)
        self.agent_states[agent_id].restart_pending = False
        await self._copy_all_snapshots_to_temp_workspace(agent_id)
        if agent.backend.filesystem_manager:
            agent.backend.filesystem_manager.log_current_state('before execution')
        try:
            agent_system_message = agent.get_configurable_system_message()
            if agent.backend.filesystem_manager:
                main_workspace = str(agent.backend.filesystem_manager.get_current_workspace())
                temp_workspace = str(agent.backend.filesystem_manager.agent_temporary_workspace) if agent.backend.filesystem_manager.agent_temporary_workspace else None
                context_paths = agent.backend.filesystem_manager.path_permission_manager.get_context_paths() if agent.backend.filesystem_manager.path_permission_manager else []
                previous_turns_context = self._get_previous_turns_context_paths()
                current_turn_num = len(previous_turns_context) + 1 if previous_turns_context else 1
                turns_to_show = [t for t in previous_turns_context if t['turn'] < current_turn_num - 1]
                workspace_prepopulated = len(previous_turns_context) > 0
                enable_image_generation = False
                if hasattr(agent, 'config') and agent.config:
                    enable_image_generation = agent.config.backend_params.get('enable_image_generation', False)
                elif hasattr(agent, 'backend') and hasattr(agent.backend, 'backend_params'):
                    enable_image_generation = agent.backend.backend_params.get('enable_image_generation', False)
                enable_command_execution = False
                if hasattr(agent, 'config') and agent.config:
                    enable_command_execution = agent.config.backend_params.get('enable_mcp_command_line', False)
                elif hasattr(agent, 'backend') and hasattr(agent.backend, 'backend_params'):
                    enable_command_execution = agent.backend.backend_params.get('enable_mcp_command_line', False)
                filesystem_system_message = self.message_templates.filesystem_system_message(main_workspace=main_workspace, temp_workspace=temp_workspace, context_paths=context_paths, previous_turns=turns_to_show, workspace_prepopulated=workspace_prepopulated, enable_image_generation=enable_image_generation, agent_answers=answers, enable_command_execution=enable_command_execution)
                agent_system_message = f'{agent_system_message}\n\n{filesystem_system_message}' if agent_system_message else filesystem_system_message
            normalized_answers = self._normalize_workspace_paths_in_answers(answers, agent_id) if answers else answers
            if normalized_answers:
                logger.info(f'[Orchestrator] Agent {agent_id} sees normalized answers: {normalized_answers}')
            else:
                logger.info(f'[Orchestrator] Agent {agent_id} sees no existing answers')
            is_coordination_phase = self.workflow_phase == 'coordinating'
            planning_mode_enabled = self.config.coordination_config and self.config.coordination_config.enable_planning_mode and is_coordination_phase if self.config and hasattr(self.config, 'coordination_config') else False
            if planning_mode_enabled and self.config.coordination_config.planning_mode_instruction:
                planning_instructions = f'\n\n{self.config.coordination_config.planning_mode_instruction}'
                agent_system_message = f'{agent_system_message}{planning_instructions}' if agent_system_message else planning_instructions.strip()
            if conversation_context and conversation_context.get('conversation_history'):
                conversation = self.message_templates.build_conversation_with_context(current_task=task, conversation_history=conversation_context.get('conversation_history', []), agent_summaries=normalized_answers, valid_agent_ids=list(normalized_answers.keys()) if normalized_answers else None, base_system_message=agent_system_message)
            else:
                conversation = self.message_templates.build_initial_conversation(task=task, agent_summaries=normalized_answers, valid_agent_ids=list(normalized_answers.keys()) if normalized_answers else None, base_system_message=agent_system_message)
            self.coordination_tracker.track_agent_context(agent_id, answers, conversation.get('conversation_history', []), conversation)
            self.agent_states[agent_id].last_context = conversation
            backend_name = None
            if hasattr(agent, 'backend') and hasattr(agent.backend, 'get_provider_name'):
                backend_name = agent.backend.get_provider_name()
            log_orchestrator_agent_message(agent_id, 'SEND', {'system': conversation['system_message'], 'user': conversation['user_message']}, backend_name=backend_name)
            if hasattr(agent.backend, 'set_planning_mode'):
                agent.backend.set_planning_mode(planning_mode_enabled)
                if planning_mode_enabled:
                    logger.info(f'[Orchestrator] Backend planning mode ENABLED for {agent_id} - MCP tools blocked')
                else:
                    logger.info(f'[Orchestrator] Backend planning mode DISABLED for {agent_id} - MCP tools allowed')
            max_attempts = 3
            conversation_messages = [{'role': 'system', 'content': conversation['system_message']}, {'role': 'user', 'content': conversation['user_message']}]
            enforcement_msg = self.message_templates.enforcement_message()
            self.coordination_tracker.change_status(agent_id, AgentStatus.STREAMING)
            for attempt in range(max_attempts):
                logger.info(f'[Orchestrator] Agent {agent_id} attempt {attempt + 1}/{max_attempts}')
                if self._check_restart_pending(agent_id):
                    logger.info(f'[Orchestrator] Agent {agent_id} restarting due to restart_pending flag')
                    await self._save_partial_work_on_restart(agent_id)
                    yield ('content', f'🔁 [{agent_id}] gracefully restarting due to new answer detected\n')
                    yield ('done', None)
                    return
                if attempt == 0:
                    chat_stream = agent.chat(conversation_messages, self.workflow_tools, reset_chat=True, current_stage=CoordinationStage.INITIAL_ANSWER)
                elif isinstance(enforcement_msg, list):
                    chat_stream = agent.chat(enforcement_msg, self.workflow_tools, reset_chat=False, current_stage=CoordinationStage.ENFORCEMENT)
                else:
                    enforcement_message = {'role': 'user', 'content': enforcement_msg}
                    chat_stream = agent.chat([enforcement_message], self.workflow_tools, reset_chat=False, current_stage=CoordinationStage.ENFORCEMENT)
                response_text = ''
                tool_calls = []
                workflow_tool_found = False
                logger.info(f'[Orchestrator] Agent {agent_id} starting to stream chat response...')
                async for chunk in chat_stream:
                    chunk_type = self._get_chunk_type_value(chunk)
                    if chunk_type == 'content':
                        response_text += chunk.content
                        yield ('content', chunk.content)
                        backend_name = None
                        if hasattr(agent, 'backend') and hasattr(agent.backend, 'get_provider_name'):
                            backend_name = agent.backend.get_provider_name()
                        log_orchestrator_agent_message(agent_id, 'RECV', {'content': chunk.content}, backend_name=backend_name)
                    elif chunk_type in ['reasoning', 'reasoning_done', 'reasoning_summary', 'reasoning_summary_done']:
                        reasoning_chunk = StreamChunk(type=chunk.type, content=chunk.content, source=agent_id, reasoning_delta=getattr(chunk, 'reasoning_delta', None), reasoning_text=getattr(chunk, 'reasoning_text', None), reasoning_summary_delta=getattr(chunk, 'reasoning_summary_delta', None), reasoning_summary_text=getattr(chunk, 'reasoning_summary_text', None), item_id=getattr(chunk, 'item_id', None), content_index=getattr(chunk, 'content_index', None), summary_index=getattr(chunk, 'summary_index', None))
                        yield ('reasoning', reasoning_chunk)
                    elif chunk_type == 'backend_status':
                        pass
                    elif chunk_type == 'mcp_status':
                        mcp_content = f'🔧 MCP: {chunk.content}'
                        yield ('content', mcp_content)
                    elif chunk_type == 'debug':
                        yield ('debug', chunk.content)
                    elif chunk_type == 'tool_calls':
                        chunk_tool_calls = getattr(chunk, 'tool_calls', []) or []
                        tool_calls.extend(chunk_tool_calls)
                        backend_name = None
                        if hasattr(agent, 'backend') and hasattr(agent.backend, 'get_provider_name'):
                            backend_name = agent.backend.get_provider_name()
                        for tool_call in chunk_tool_calls:
                            tool_name = agent.backend.extract_tool_name(tool_call)
                            tool_args = agent.backend.extract_tool_arguments(tool_call)
                            if tool_name == 'new_answer':
                                content = tool_args.get('content', '')
                                yield ('content', f'💡 Providing answer: "{content}"')
                                log_tool_call(agent_id, 'new_answer', {'content': content}, None, backend_name)
                            elif tool_name == 'vote':
                                agent_voted_for = tool_args.get('agent_id', '')
                                reason = tool_args.get('reason', '')
                                log_tool_call(agent_id, 'vote', {'agent_id': agent_voted_for, 'reason': reason}, None, backend_name)
                                real_agent_id = agent_voted_for
                                if answers:
                                    agent_mapping = {}
                                    for i, real_id in enumerate(sorted(answers.keys()), 1):
                                        agent_mapping[f'agent{i}'] = real_id
                                    real_agent_id = agent_mapping.get(agent_voted_for, agent_voted_for)
                                yield ('content', f'🗳️ Voting for [{real_agent_id}] (options: {', '.join(sorted(answers.keys()))}) : {reason}')
                            else:
                                yield ('content', f'🔧 Using {tool_name}')
                                log_tool_call(agent_id, tool_name, tool_args, None, backend_name)
                    elif chunk_type == 'error':
                        error_msg = getattr(chunk, 'error', str(chunk.content)) if hasattr(chunk, 'error') else str(chunk.content)
                        yield ('content', f'❌ Error: {error_msg}\n')
                vote_calls = [tc for tc in tool_calls if agent.backend.extract_tool_name(tc) == 'vote']
                if len(vote_calls) > 1:
                    if attempt < max_attempts - 1:
                        if self._check_restart_pending(agent_id):
                            await self._save_partial_work_on_restart(agent_id)
                            yield ('content', f'🔁 [{agent_id}] gracefully restarting due to new answer detected\n')
                            yield ('done', None)
                            return
                        error_msg = f'Multiple vote calls not allowed. Made {len(vote_calls)} calls but must make exactly 1. Call vote tool once with chosen agent.'
                        yield ('content', f'❌ {error_msg}')
                        enforcement_msg = self._create_tool_error_messages(agent, tool_calls, error_msg, 'Vote rejected due to multiple votes.')
                        continue
                    else:
                        yield ('error', f'Agent made {len(vote_calls)} vote calls in single response after max attempts')
                        yield ('done', None)
                        return
                new_answer_calls = [tc for tc in tool_calls if agent.backend.extract_tool_name(tc) == 'new_answer']
                if len(vote_calls) > 0 and len(new_answer_calls) > 0:
                    if attempt < max_attempts - 1:
                        if self._check_restart_pending(agent_id):
                            await self._save_partial_work_on_restart(agent_id)
                            yield ('content', f'🔁 [{agent_id}] gracefully restarting due to new answer detected\n')
                            yield ('done', None)
                            return
                        error_msg = "Cannot use both 'vote' and 'new_answer' in same response. Choose one: vote for existing answer OR provide new answer."
                        yield ('content', f'❌ {error_msg}')
                        enforcement_msg = self._create_tool_error_messages(agent, tool_calls, error_msg)
                        continue
                    else:
                        yield ('error', 'Agent used both vote and new_answer tools in single response after max attempts')
                        yield ('done', None)
                        return
                if tool_calls:
                    for tool_call in tool_calls:
                        tool_name = agent.backend.extract_tool_name(tool_call)
                        tool_args = agent.backend.extract_tool_arguments(tool_call)
                        if tool_name == 'vote':
                            logger.info(f'[Orchestrator] Agent {agent_id} voting from options: {(list(answers.keys()) if answers else 'No answers available')}')
                            if self._check_restart_pending(agent_id):
                                await self._save_partial_work_on_restart(agent_id)
                                yield ('content', f'🔄 [{agent_id}] Vote invalid - restarting due to new answers')
                                yield ('done', None)
                                return
                            workflow_tool_found = True
                            if not answers:
                                if attempt < max_attempts - 1:
                                    if self._check_restart_pending(agent_id):
                                        await self._save_partial_work_on_restart(agent_id)
                                        yield ('content', f'🔁 [{agent_id}] gracefully restarting due to new answer detected\n')
                                        yield ('done', None)
                                        return
                                    error_msg = 'Cannot vote when no answers exist. Use new_answer tool.'
                                    yield ('content', f'❌ {error_msg}')
                                    enforcement_msg = self._create_tool_error_messages(agent, [tool_call], error_msg)
                                    continue
                                else:
                                    yield ('error', 'Cannot vote when no answers exist after max attempts')
                                    yield ('done', None)
                                    return
                            voted_agent_anon = tool_args.get('agent_id')
                            reason = tool_args.get('reason', '')
                            agent_mapping = {}
                            for i, real_agent_id in enumerate(sorted(answers.keys()), 1):
                                agent_mapping[f'agent{i}'] = real_agent_id
                            voted_agent = agent_mapping.get(voted_agent_anon, voted_agent_anon)
                            if voted_agent not in answers:
                                if attempt < max_attempts - 1:
                                    if self._check_restart_pending(agent_id):
                                        await self._save_partial_work_on_restart(agent_id)
                                        yield ('content', f'🔁 [{agent_id}] gracefully restarting due to new answer detected\n')
                                        yield ('done', None)
                                        return
                                    reverse_mapping = {real_id: f'agent{i}' for i, real_id in enumerate(sorted(answers.keys()), 1)}
                                    valid_anon_agents = [reverse_mapping[real_id] for real_id in answers.keys()]
                                    error_msg = f"Invalid agent_id '{voted_agent_anon}'. Valid agents: {', '.join(valid_anon_agents)}"
                                    yield ('content', f'❌ {error_msg}')
                                    enforcement_msg = self._create_tool_error_messages(agent, [tool_call], error_msg)
                                    continue
                                else:
                                    yield ('error', f'Invalid agent_id after {max_attempts} attempts')
                                    yield ('done', None)
                                    return
                            self.agent_states[agent_id].votes = {'agent_id': voted_agent, 'reason': reason}
                            yield ('result', ('vote', {'agent_id': voted_agent, 'reason': reason}))
                            yield ('done', None)
                            return
                        elif tool_name == 'new_answer':
                            workflow_tool_found = True
                            content = tool_args.get('content', response_text.strip())
                            normalized_new_content = self._normalize_workspace_paths_for_comparison(content)
                            for existing_agent_id, existing_content in answers.items():
                                normalized_existing_content = self._normalize_workspace_paths_for_comparison(existing_content)
                                if normalized_new_content.strip() == normalized_existing_content.strip():
                                    if attempt < max_attempts - 1:
                                        if self._check_restart_pending(agent_id):
                                            await self._save_partial_work_on_restart(agent_id)
                                            yield ('content', f'🔁 [{agent_id}] gracefully restarting due to new answer detected\n')
                                            yield ('done', None)
                                            return
                                        error_msg = f'Answer already provided by {existing_agent_id}. Provide different answer or vote for existing one.'
                                        yield ('content', f'❌ {error_msg}')
                                        enforcement_msg = self._create_tool_error_messages(agent, [tool_call], error_msg)
                                        continue
                                    else:
                                        yield ('error', f'Duplicate answer provided after {max_attempts} attempts')
                                        yield ('done', None)
                                        return
                            yield ('result', ('answer', content))
                            yield ('done', None)
                            return
                        elif tool_name.startswith('mcp'):
                            pass
                        else:
                            yield ('content', f'🔧 used {tool_name} tool (not implemented)')
                if not workflow_tool_found:
                    if self._check_restart_pending(agent_id):
                        await self._save_partial_work_on_restart(agent_id)
                        yield ('content', f'🔁 [{agent_id}] gracefully restarting due to new answer detected\n')
                        yield ('done', None)
                        return
                    if attempt < max_attempts - 1:
                        yield ('content', '🔄 needs to use workflow tools...\n')
                        enforcement_msg = self.message_templates.enforcement_message()
                        continue
                    else:
                        yield ('error', f'Agent failed to use workflow tools after {max_attempts} attempts')
                        yield ('done', None)
                        return
        except Exception as e:
            yield ('error', f'Agent execution failed: {str(e)}')
            yield ('done', None)

    async def _get_next_chunk(self, stream: AsyncGenerator[tuple, None]) -> tuple:
        """Get the next chunk from an agent stream."""
        try:
            return await stream.__anext__()
        except StopAsyncIteration:
            return ('done', None)
        except Exception as e:
            return ('error', str(e))

    async def _present_final_answer(self) -> AsyncGenerator[StreamChunk, None]:
        """Present the final coordinated answer."""
        log_stream_chunk('orchestrator', 'content', '## 🎯 Final Coordinated Answer\n')
        yield StreamChunk(type='content', content='## 🎯 Final Coordinated Answer\n')
        if not self._selected_agent:
            self._selected_agent = self._determine_final_agent_from_states()
            if self._selected_agent:
                log_stream_chunk('orchestrator', 'content', f'🏆 Selected Agent: {self._selected_agent}\n')
                yield StreamChunk(type='content', content=f'🏆 Selected Agent: {self._selected_agent}\n')
        if self._selected_agent and self._selected_agent in self.agent_states and self.agent_states[self._selected_agent].answer:
            final_answer = self.agent_states[self._selected_agent].answer
            self.add_to_history('assistant', final_answer)
            log_stream_chunk('orchestrator', 'content', f'🏆 Selected Agent: {self._selected_agent}\n')
            yield StreamChunk(type='content', content=f'🏆 Selected Agent: {self._selected_agent}\n')
            log_stream_chunk('orchestrator', 'content', final_answer)
            yield StreamChunk(type='content', content=final_answer)
            log_stream_chunk('orchestrator', 'content', f'\n\n---\n*Coordinated by {len(self.agents)} agents via MassGen framework*')
            yield StreamChunk(type='content', content=f'\n\n---\n*Coordinated by {len(self.agents)} agents via MassGen framework*')
        else:
            error_msg = '❌ Unable to provide coordinated answer - no successful agents'
            self.add_to_history('assistant', error_msg)
            log_stream_chunk('orchestrator', 'error', error_msg)
            yield StreamChunk(type='content', content=error_msg)
        self.workflow_phase = 'presenting'
        log_stream_chunk('orchestrator', 'done', None)
        yield StreamChunk(type='done')

    async def _handle_orchestrator_timeout(self) -> AsyncGenerator[StreamChunk, None]:
        """Handle orchestrator timeout by jumping directly to get_final_presentation."""
        log_stream_chunk('orchestrator', 'content', f'\n⚠️ **Orchestrator Timeout**: {self.timeout_reason}\n', self.orchestrator_id)
        yield StreamChunk(type='content', content=f'\n⚠️ **Orchestrator Timeout**: {self.timeout_reason}\n', source=self.orchestrator_id)
        available_answers = {aid: state.answer for aid, state in self.agent_states.items() if state.answer and (not state.is_killed)}
        log_stream_chunk('orchestrator', 'content', f'📊 Current state: {len(available_answers)} answers available\n', self.orchestrator_id)
        yield StreamChunk(type='content', content=f'📊 Current state: {len(available_answers)} answers available\n', source=self.orchestrator_id)
        if len(available_answers) == 0:
            log_stream_chunk('orchestrator', 'error', '❌ No answers available from any agents due to timeout. No agents had enough time to provide responses.\n', self.orchestrator_id)
            yield StreamChunk(type='content', content='❌ No answers available from any agents due to timeout. No agents had enough time to provide responses.\n', source=self.orchestrator_id)
            self.workflow_phase = 'presenting'
            log_stream_chunk('orchestrator', 'done', None)
            yield StreamChunk(type='done')
            return
        current_votes = {aid: state.votes for aid, state in self.agent_states.items() if state.votes and (not state.is_killed)}
        self._selected_agent = self._determine_final_agent_from_votes(current_votes, available_answers)
        vote_results = self._get_vote_results()
        log_stream_chunk('orchestrator', 'content', f'🎯 Jumping to final presentation with {self._selected_agent} (selected despite timeout)\n', self.orchestrator_id)
        yield StreamChunk(type='content', content=f'🎯 Jumping to final presentation with {self._selected_agent} (selected despite timeout)\n', source=self.orchestrator_id)
        async for chunk in self.get_final_presentation(self._selected_agent, vote_results):
            yield chunk

    def _determine_final_agent_from_votes(self, votes: Dict[str, Dict], agent_answers: Dict[str, str]) -> str:
        """Determine which agent should present the final answer based on votes."""
        if not votes:
            return next(iter(agent_answers)) if agent_answers else None
        vote_counts = {}
        for vote_data in votes.values():
            voted_for = vote_data.get('agent_id')
            if voted_for:
                vote_counts[voted_for] = vote_counts.get(voted_for, 0) + 1
        if not vote_counts:
            return next(iter(agent_answers)) if agent_answers else None
        max_votes = max(vote_counts.values())
        tied_agents = [agent_id for agent_id, count in vote_counts.items() if count == max_votes]
        for agent_id in agent_answers.keys():
            if agent_id in tied_agents:
                return agent_id
        return tied_agents[0] if tied_agents else next(iter(agent_answers)) if agent_answers else None

    async def get_final_presentation(self, selected_agent_id: str, vote_results: Dict[str, Any]) -> AsyncGenerator[StreamChunk, None]:
        """Ask the winning agent to present their final answer with voting context."""
        self.coordination_tracker.start_final_round(selected_agent_id)
        if selected_agent_id not in self.agents:
            log_stream_chunk('orchestrator', 'error', f'Selected agent {selected_agent_id} not found')
            yield StreamChunk(type='error', error=f'Selected agent {selected_agent_id} not found')
            return
        agent = self.agents[selected_agent_id]
        if agent.backend.filesystem_manager:
            agent.backend.filesystem_manager.path_permission_manager.set_context_write_access_enabled(True)
        if hasattr(agent.backend, 'set_planning_mode'):
            agent.backend.set_planning_mode(False)
            logger.info(f'[Orchestrator] Backend planning mode DISABLED for final presentation: {selected_agent_id} - MCP tools now allowed')
        temp_workspace_path = await self._copy_all_snapshots_to_temp_workspace(selected_agent_id)
        yield StreamChunk(type='debug', content=f'Restored workspace context for final presentation: {temp_workspace_path}', source=selected_agent_id)
        vote_counts = vote_results.get('vote_counts', {})
        voter_details = vote_results.get('voter_details', {})
        is_tie = vote_results.get('is_tie', False)
        voting_summary = f'You received {vote_counts.get(selected_agent_id, 0)} vote(s)'
        if voter_details.get(selected_agent_id):
            reasons = [v['reason'] for v in voter_details[selected_agent_id]]
            voting_summary += f' with feedback: {'; '.join(reasons)}'
        if is_tie:
            voting_summary += ' (tie-broken by registration order)'
        all_answers = {aid: s.answer for aid, s in self.agent_states.items() if s.answer}
        normalized_voting_summary = self._normalize_workspace_paths_in_answers({selected_agent_id: voting_summary}, selected_agent_id)[selected_agent_id]
        normalized_all_answers = self._normalize_workspace_paths_in_answers(all_answers, selected_agent_id)
        presentation_content = self.message_templates.build_final_presentation_message(original_task=self.current_task or 'Task coordination', vote_summary=normalized_voting_summary, all_answers=normalized_all_answers, selected_agent_id=selected_agent_id)
        agent_system_message = agent.get_configurable_system_message()
        enable_image_generation = False
        if hasattr(agent, 'config') and agent.config:
            enable_image_generation = agent.config.backend_params.get('enable_image_generation', False)
        elif hasattr(agent, 'backend') and hasattr(agent.backend, 'backend_params'):
            enable_image_generation = agent.backend.backend_params.get('enable_image_generation', False)
        enable_command_execution = False
        if hasattr(agent, 'config') and agent.config:
            enable_command_execution = agent.config.backend_params.get('enable_mcp_command_line', False)
        elif hasattr(agent, 'backend') and hasattr(agent.backend, 'backend_params'):
            enable_command_execution = agent.backend.backend_params.get('enable_mcp_command_line', False)
        enable_audio_generation = False
        if hasattr(agent, 'config') and agent.config:
            enable_audio_generation = agent.config.backend_params.get('enable_audio_generation', False)
        elif hasattr(agent, 'backend') and hasattr(agent.backend, 'backend_params'):
            enable_audio_generation = agent.backend.backend_params.get('enable_audio_generation', False)
        has_irreversible_actions = False
        if agent.backend.filesystem_manager:
            context_paths = agent.backend.filesystem_manager.path_permission_manager.get_context_paths()
            has_irreversible_actions = any((cp.get('permission') == 'write' for cp in context_paths))
        base_system_message = self.message_templates.final_presentation_system_message(agent_system_message, enable_image_generation, enable_audio_generation, has_irreversible_actions, enable_command_execution)
        for aid, state in self.agent_states.items():
            if aid != selected_agent_id:
                self.coordination_tracker.change_status(aid, AgentStatus.COMPLETED)
        self.coordination_tracker.set_final_agent(selected_agent_id, voting_summary, all_answers)
        if agent.backend.filesystem_manager and temp_workspace_path:
            main_workspace = str(agent.backend.filesystem_manager.get_current_workspace())
            temp_workspace = str(agent.backend.filesystem_manager.agent_temporary_workspace) if agent.backend.filesystem_manager.agent_temporary_workspace else None
            context_paths = agent.backend.filesystem_manager.path_permission_manager.get_context_paths() if agent.backend.filesystem_manager.path_permission_manager else []
            previous_turns_context = self._get_previous_turns_context_paths()
            current_turn_num = len(previous_turns_context) + 1 if previous_turns_context else 1
            turns_to_show = [t for t in previous_turns_context if t['turn'] < current_turn_num - 1]
            workspace_prepopulated = len(previous_turns_context) > 0
            base_system_message = self.message_templates.filesystem_system_message(main_workspace=main_workspace, temp_workspace=temp_workspace, context_paths=context_paths, previous_turns=turns_to_show, workspace_prepopulated=workspace_prepopulated, enable_image_generation=enable_image_generation, agent_answers=all_answers, enable_command_execution=enable_command_execution) + '\n\n## Instructions\n' + base_system_message
        presentation_messages = [{'role': 'system', 'content': base_system_message}, {'role': 'user', 'content': presentation_content}]
        self.agent_states[selected_agent_id].last_context = {'messages': presentation_messages, 'is_final': True, 'vote_summary': voting_summary, 'all_answers': all_answers, 'complete_vote_results': vote_results, 'vote_counts': vote_counts, 'voter_details': voter_details, 'all_votes': {aid: state.votes for aid, state in self.agent_states.items() if state.votes}}
        log_stream_chunk('orchestrator', 'status', f'🎤  [{selected_agent_id}] presenting final answer\n')
        yield StreamChunk(type='status', content=f'🎤  [{selected_agent_id}] presenting final answer\n')
        presentation_content = ''
        try:
            async for chunk in agent.chat(presentation_messages, reset_chat=True, current_stage=CoordinationStage.PRESENTATION):
                chunk_type = self._get_chunk_type_value(chunk)
                self.coordination_tracker.start_new_iteration()
                if chunk_type == 'content' and chunk.content:
                    presentation_content += chunk.content
                    log_stream_chunk('orchestrator', 'content', chunk.content, selected_agent_id)
                    yield StreamChunk(type='content', content=chunk.content, source=selected_agent_id)
                elif chunk_type in ['reasoning', 'reasoning_done', 'reasoning_summary', 'reasoning_summary_done']:
                    reasoning_chunk = StreamChunk(type=chunk_type, content=chunk.content, source=selected_agent_id, reasoning_delta=getattr(chunk, 'reasoning_delta', None), reasoning_text=getattr(chunk, 'reasoning_text', None), reasoning_summary_delta=getattr(chunk, 'reasoning_summary_delta', None), reasoning_summary_text=getattr(chunk, 'reasoning_summary_text', None), item_id=getattr(chunk, 'item_id', None), content_index=getattr(chunk, 'content_index', None), summary_index=getattr(chunk, 'summary_index', None))
                    log_stream_chunk('orchestrator', chunk.type, chunk.content, selected_agent_id)
                    yield reasoning_chunk
                elif chunk_type == 'backend_status':
                    import json
                    status_json = json.loads(chunk.content)
                    cwd = status_json['cwd']
                    session_id = status_json['session_id']
                    content = f'Final Temp Working directory: {cwd}.\n    Final Session ID: {session_id}.\n    '
                    log_stream_chunk('orchestrator', 'content', content, selected_agent_id)
                    yield StreamChunk(type='content', content=content, source=selected_agent_id)
                elif chunk_type == 'mcp_status':
                    mcp_content = f'🔧 MCP: {chunk.content}'
                    log_stream_chunk('orchestrator', 'content', mcp_content, selected_agent_id)
                    yield StreamChunk(type='content', content=mcp_content, source=selected_agent_id)
                elif chunk_type == 'done':
                    final_answer = presentation_content.strip() if presentation_content.strip() else self.agent_states[selected_agent_id].answer
                    final_context = self.get_last_context(selected_agent_id)
                    await self._save_agent_snapshot(self._selected_agent, answer_content=final_answer, is_final=True, context_data=final_context)
                    self.coordination_tracker.set_final_answer(selected_agent_id, final_answer, snapshot_timestamp='final')
                    log_stream_chunk('orchestrator', 'done', None, selected_agent_id)
                    yield StreamChunk(type='done', source=selected_agent_id)
                elif chunk_type == 'error':
                    log_stream_chunk('orchestrator', 'error', chunk.error, selected_agent_id)
                    yield StreamChunk(type='error', error=chunk.error, source=selected_agent_id)
                elif hasattr(chunk, 'source'):
                    log_stream_chunk('orchestrator', chunk_type, getattr(chunk, 'content', ''), selected_agent_id)
                    yield StreamChunk(type=chunk_type, content=getattr(chunk, 'content', ''), source=selected_agent_id, **{k: v for k, v in chunk.__dict__.items() if k not in ['type', 'content', 'source']})
                else:
                    log_stream_chunk('orchestrator', chunk_type, getattr(chunk, 'content', ''), selected_agent_id)
                    yield StreamChunk(type=chunk_type, content=getattr(chunk, 'content', ''), source=selected_agent_id, **{k: v for k, v in chunk.__dict__.items() if k not in ['type', 'content', 'source']})
        finally:
            if presentation_content.strip():
                self._final_presentation_content = presentation_content.strip()
            else:
                stored_answer = self.agent_states[selected_agent_id].answer
                if stored_answer:
                    fallback_content = f'\n📋 Using stored answer as final presentation:\n\n{stored_answer}'
                    log_stream_chunk('orchestrator', 'content', fallback_content, selected_agent_id)
                    yield StreamChunk(type='content', content=fallback_content, source=selected_agent_id)
                    self._final_presentation_content = stored_answer
                else:
                    log_stream_chunk('orchestrator', 'error', '\n❌ No content generated for final presentation and no stored answer available.', selected_agent_id)
                    yield StreamChunk(type='content', content='\n❌ No content generated for final presentation and no stored answer available.', source=selected_agent_id)
            self.coordination_tracker.change_status(selected_agent_id, AgentStatus.COMPLETED)
            self.save_coordination_logs()

    def _get_vote_results(self) -> Dict[str, Any]:
        """Get current vote results and statistics."""
        agent_answers = {aid: state.answer for aid, state in self.agent_states.items() if state.answer}
        votes = {aid: state.votes for aid, state in self.agent_states.items() if state.votes}
        vote_counts = {}
        voter_details = {}
        for voter_id, vote_data in votes.items():
            voted_for = vote_data.get('agent_id')
            if voted_for:
                vote_counts[voted_for] = vote_counts.get(voted_for, 0) + 1
                if voted_for not in voter_details:
                    voter_details[voted_for] = []
                voter_details[voted_for].append({'voter': voter_id, 'reason': vote_data.get('reason', 'No reason provided')})
        winner = None
        is_tie = False
        if vote_counts:
            max_votes = max(vote_counts.values())
            tied_agents = [agent_id for agent_id, count in vote_counts.items() if count == max_votes]
            is_tie = len(tied_agents) > 1
            for agent_id in agent_answers.keys():
                if agent_id in tied_agents:
                    winner = agent_id
                    break
            if not winner:
                winner = tied_agents[0] if tied_agents else None
        agent_mapping = {}
        for i, real_id in enumerate(sorted(agent_answers.keys()), 1):
            agent_mapping[f'agent{i}'] = real_id
        return {'vote_counts': vote_counts, 'voter_details': voter_details, 'winner': winner, 'is_tie': is_tie, 'total_votes': len(votes), 'agents_with_answers': len(agent_answers), 'agents_voted': len([v for v in votes.values() if v.get('agent_id')]), 'agent_mapping': agent_mapping}

    def _determine_final_agent_from_states(self) -> Optional[str]:
        """Determine final agent based on current agent states."""
        agents_with_answers = {aid: state.answer for aid, state in self.agent_states.items() if state.answer}
        if not agents_with_answers:
            return None
        return next(iter(agents_with_answers))

    async def _handle_followup(self, user_message: str, conversation_context: Optional[Dict[str, Any]]=None) -> AsyncGenerator[StreamChunk, None]:
        """Handle follow-up questions after presenting final answer with conversation context."""
        if conversation_context and len(conversation_context.get('conversation_history', [])) > 0:
            log_stream_chunk('orchestrator', 'content', f"🤔 Thank you for your follow-up question in our ongoing conversation. I understand you're asking: '{user_message}'. Currently, the coordination is complete, but I can help clarify the answer or coordinate a new task that takes our conversation history into account.")
            yield StreamChunk(type='content', content=f"🤔 Thank you for your follow-up question in our ongoing conversation. I understand you're asking: '{user_message}'. Currently, the coordination is complete, but I can help clarify the answer or coordinate a new task that takes our conversation history into account.")
        else:
            log_stream_chunk('orchestrator', 'content', f"🤔 Thank you for your follow-up: '{user_message}'. The coordination is complete, but I can help clarify the answer or coordinate a new task if needed.")
            yield StreamChunk(type='content', content=f"🤔 Thank you for your follow-up: '{user_message}'. The coordination is complete, but I can help clarify the answer or coordinate a new task if needed.")
        log_stream_chunk('orchestrator', 'done', None)
        yield StreamChunk(type='done')

    def add_agent(self, agent_id: str, agent: ChatAgent) -> None:
        """Add a new sub-agent to the orchestrator."""
        self.agents[agent_id] = agent
        self.agent_states[agent_id] = AgentState()

    def remove_agent(self, agent_id: str) -> None:
        """Remove a sub-agent from the orchestrator."""
        if agent_id in self.agents:
            del self.agents[agent_id]
        if agent_id in self.agent_states:
            del self.agent_states[agent_id]

    def get_final_result(self) -> Optional[Dict[str, Any]]:
        """
        Get final result for session persistence.

        Returns:
            Dict with final_answer, winning_agent_id, and workspace_path, or None if not available
        """
        if not self._selected_agent or not self._final_presentation_content:
            return None
        winning_agent = self.agents.get(self._selected_agent)
        workspace_path = None
        if winning_agent and winning_agent.backend.filesystem_manager:
            workspace_path = str(winning_agent.backend.filesystem_manager.get_current_workspace())
        return {'final_answer': self._final_presentation_content, 'winning_agent_id': self._selected_agent, 'workspace_path': workspace_path}

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        vote_results = self._get_vote_results()
        return {'session_id': self.session_id, 'workflow_phase': self.workflow_phase, 'current_task': self.current_task, 'selected_agent': self._selected_agent, 'final_presentation_content': self._final_presentation_content, 'vote_results': vote_results, 'agents': {aid: {'agent_status': agent.get_status(), 'coordination_state': {'answer': state.answer, 'has_voted': state.has_voted}} for aid, (agent, state) in zip(self.agents.keys(), zip(self.agents.values(), self.agent_states.values()))}, 'conversation_length': len(self.conversation_history)}

    def get_configurable_system_message(self) -> Optional[str]:
        """
        Get the configurable system message for the orchestrator.

        This can define how the orchestrator should coordinate agents, construct messages,
        handle conflicts, make decisions, etc. For example:
        - Custom voting strategies
        - Message construction templates
        - Conflict resolution approaches
        - Coordination workflow preferences

        Returns:
            Orchestrator's configurable system message if available, None otherwise
        """
        if self.config and hasattr(self.config, 'get_configurable_system_message'):
            return self.config.get_configurable_system_message()
        elif self.config and hasattr(self.config, 'custom_system_instruction'):
            return self.config.custom_system_instruction
        elif self.config and self.config.backend_params:
            backend_params = self.config.backend_params
            if 'system_prompt' in backend_params:
                return backend_params['system_prompt']
            elif 'append_system_prompt' in backend_params:
                return backend_params['append_system_prompt']
        return None

    def _clear_agent_workspaces(self) -> None:
        """
        Clear all agent workspaces and pre-populate with previous turn's results.

        This creates a WRITABLE copy of turn n-1 in each agent's workspace.
        Note: CLI separately provides turn n-1 as a READ-ONLY context path, allowing
        agents to both modify files (in workspace) and reference originals (via context path).
        """
        previous_turn_workspace = None
        if self._previous_turns:
            latest_turn = self._previous_turns[-1]
            previous_turn_workspace = Path(latest_turn['path'])
        for agent_id, agent in self.agents.items():
            if agent.backend.filesystem_manager:
                workspace_path = agent.backend.filesystem_manager.get_current_workspace()
                if workspace_path and Path(workspace_path).exists():
                    for item in Path(workspace_path).iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    logger.info(f'[Orchestrator] Cleared workspace for {agent_id}: {workspace_path}')
                    if previous_turn_workspace and previous_turn_workspace.exists():
                        logger.info(f'[Orchestrator] Pre-populating {agent_id} workspace with writable copy of turn n-1 from {previous_turn_workspace}')
                        for item in previous_turn_workspace.iterdir():
                            dest = Path(workspace_path) / item.name
                            if item.is_file():
                                shutil.copy2(item, dest)
                            elif item.is_dir():
                                shutil.copytree(item, dest, dirs_exist_ok=True)
                        logger.info(f'[Orchestrator] Pre-populated {agent_id} workspace with writable copy of turn n-1')

    def _get_previous_turns_context_paths(self) -> List[Dict[str, Any]]:
        """
        Get previous turns as context paths for current turn's agents.

        Returns:
            List of previous turn information with path, turn number, and task
        """
        return self._previous_turns

    async def reset(self) -> None:
        """Reset orchestrator state for new task."""
        self.conversation_history.clear()
        self.current_task = None
        self.workflow_phase = 'idle'
        self._coordination_messages.clear()
        self._selected_agent = None
        self._final_presentation_content = None
        for state in self.agent_states.values():
            state.answer = None
            state.has_voted = False
            state.restart_pending = False
            state.is_killed = False
            state.timeout_reason = None
        self.total_tokens = 0
        self.coordination_start_time = 0
        self.is_orchestrator_timeout = False
        self.timeout_reason = None
        self._active_streams = {}
        self._active_tasks = {}

def _validate_value(value: Any, depth: int=0) -> Any:
    if depth > max_depth:
        raise ValueError(f'Tool arguments nested too deeply: {depth} > {max_depth}')
    if isinstance(value, dict):
        if len(value) > MAX_DICT_KEYS:
            raise ValueError(f'Dictionary too large: {len(value)} > {MAX_DICT_KEYS} keys')
        _add_size(2)
        validated: Dict[str, Any] = {}
        first = True
        for k, v in value.items():
            if not isinstance(k, str):
                k = str(k)
            if not first:
                _add_size(1)
            first = False
            _add_size(_size_for_primitive(k) + 1)
            validated[k] = _validate_value(v, depth + 1)
        return validated
    elif isinstance(value, list):
        if len(value) > MAX_LIST_ITEMS:
            raise ValueError(f'List too large: {len(value)} > {MAX_LIST_ITEMS} items')
        _add_size(2)
        validated_list = []
        for idx, item in enumerate(value):
            if idx > 0:
                _add_size(1)
            validated_list.append(_validate_value(item, depth + 1))
        return validated_list
    elif isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            raise ValueError(f'String too long: {len(value)} > {MAX_STRING_LENGTH} characters')
        _add_size(_size_for_primitive(value))
        return value
    elif isinstance(value, (int, float, bool)) or value is None:
        _add_size(_size_for_primitive(value))
        return value
    else:
        str_value = str(value)
        if len(str_value) > MAX_STRING_LENGTH:
            raise ValueError(f'Value too large when converted to string: {len(str_value)} > {MAX_STRING_LENGTH}')
        _add_size(_size_for_primitive(str_value))
        return str_value

def _add_size(amount: int) -> None:
    nonlocal current_size
    current_size += amount
    if current_size > max_size:
        raise ValueError(f'Tool arguments too large: ~{current_size} > {max_size} bytes')

def _size_for_primitive(value: Any) -> int:
    if value is None:
        return 4
    if isinstance(value, bool):
        return 4 if value else 5
    if isinstance(value, (int, float)):
        return len(str(value))
    if isinstance(value, str):
        return len(value) + 2
    return len(str(value)) + 2

def validate_tool_arguments(arguments: Dict[str, Any], max_depth: int=MAX_TOOL_ARG_DEPTH, max_size: int=MAX_TOOL_ARG_SIZE) -> Dict[str, Any]:
    """
    Validate tool arguments for security and size limits.

    Args:
        arguments: Tool arguments dictionary
        max_depth: Maximum nesting depth allowed
        max_size: Maximum total size of arguments (rough estimate)

    Returns:
        Validated arguments dictionary

    Raises:
        ValueError: If arguments are invalid or too large
    """
    if not isinstance(arguments, dict):
        raise ValueError('Tool arguments must be a dictionary')
    current_size = 0

    def _add_size(amount: int) -> None:
        nonlocal current_size
        current_size += amount
        if current_size > max_size:
            raise ValueError(f'Tool arguments too large: ~{current_size} > {max_size} bytes')

    def _size_for_primitive(value: Any) -> int:
        if value is None:
            return 4
        if isinstance(value, bool):
            return 4 if value else 5
        if isinstance(value, (int, float)):
            return len(str(value))
        if isinstance(value, str):
            return len(value) + 2
        return len(str(value)) + 2

    def _validate_value(value: Any, depth: int=0) -> Any:
        if depth > max_depth:
            raise ValueError(f'Tool arguments nested too deeply: {depth} > {max_depth}')
        if isinstance(value, dict):
            if len(value) > MAX_DICT_KEYS:
                raise ValueError(f'Dictionary too large: {len(value)} > {MAX_DICT_KEYS} keys')
            _add_size(2)
            validated: Dict[str, Any] = {}
            first = True
            for k, v in value.items():
                if not isinstance(k, str):
                    k = str(k)
                if not first:
                    _add_size(1)
                first = False
                _add_size(_size_for_primitive(k) + 1)
                validated[k] = _validate_value(v, depth + 1)
            return validated
        elif isinstance(value, list):
            if len(value) > MAX_LIST_ITEMS:
                raise ValueError(f'List too large: {len(value)} > {MAX_LIST_ITEMS} items')
            _add_size(2)
            validated_list = []
            for idx, item in enumerate(value):
                if idx > 0:
                    _add_size(1)
                validated_list.append(_validate_value(item, depth + 1))
            return validated_list
        elif isinstance(value, str):
            if len(value) > MAX_STRING_LENGTH:
                raise ValueError(f'String too long: {len(value)} > {MAX_STRING_LENGTH} characters')
            _add_size(_size_for_primitive(value))
            return value
        elif isinstance(value, (int, float, bool)) or value is None:
            _add_size(_size_for_primitive(value))
            return value
        else:
            str_value = str(value)
            if len(str_value) > MAX_STRING_LENGTH:
                raise ValueError(f'Value too large when converted to string: {len(str_value)} > {MAX_STRING_LENGTH}')
            _add_size(_size_for_primitive(str_value))
            return str_value
    return _validate_value(arguments)

class MCPClient:
    """
    Unified MCP client for communicating with single or multiple MCP servers.
    Provides improved security, error handling, and async context management.

    Accepts a list of server configurations and automatically handles:
    - Consistent tool naming: Always uses prefixed names (mcp__server__tool)
    - Circuit breaker protection for all servers
    - Parallel connection for multi-server scenarios
    - Sequential connection for single-server scenarios
    """

    def __init__(self, server_configs: List[Dict[str, Any]], *, timeout_seconds: int=30, allowed_tools: Optional[List[str]]=None, exclude_tools: Optional[List[str]]=None, status_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]]=None, hooks: Optional[Dict[HookType, List[Callable[[str, Dict[str, Any]], Awaitable[bool]]]]]=None):
        """
        Initialize MCP client.

        Args:
            server_configs: List of server configuration dicts (always a list, even for single server)
            timeout_seconds: Timeout for operations in seconds
            allowed_tools: Optional list of tool names to include (if None, includes all)
            exclude_tools: Optional list of tool names to exclude (if None, excludes none)
            status_callback: Optional async callback for status updates
            hooks: Optional dict mapping hook types to lists of hook functions
        """
        self._server_configs = [MCPConfigValidator.validate_server_config(config) for config in server_configs]
        self.name = self._server_configs[0]['name']
        self.timeout_seconds = timeout_seconds
        self.allowed_tools = allowed_tools
        self.exclude_tools = exclude_tools
        self.status_callback = status_callback
        self.hooks = hooks or {}
        self._circuit_breaker = MCPCircuitBreaker()
        self._server_clients: Dict[str, _ServerClient] = {}
        for config in self._server_configs:
            self._server_clients[config['name']] = _ServerClient()
        self.tools: Dict[str, mcp_types.Tool] = {}
        self._tool_to_server: Dict[str, str] = {}
        self._initialized = False
        self._cleanup_done = False
        self._cleanup_lock = asyncio.Lock()
        self._context_managed = False

    @property
    def session(self) -> Optional[ClientSession]:
        """Return first server's session for backward compatibility."""
        if self._server_configs:
            first_server_name = self._server_configs[0]['name']
            server_client = self._server_clients.get(first_server_name)
            if server_client:
                return server_client.session
        return None

    def _get_server_session(self, server_name: str) -> ClientSession:
        """Get session for server, raising error if not connected."""
        server_client = self._server_clients.get(server_name)
        if not server_client or not server_client.session:
            raise MCPConnectionError(f"Server '{server_name}' not connected", server_name=server_name)
        return server_client.session

    async def connect(self) -> None:
        """Connect to MCP server(s) and discover capabilities with circuit breaker integration."""
        if self._initialized:
            return
        logger.info(f'Connecting to {len(self._server_configs)} MCP server(s)...')
        if self.status_callback:
            await self.status_callback('connecting', {'message': f'Connecting to {len(self._server_configs)} MCP server(s)', 'server_count': len(self._server_configs)})
        if len(self._server_configs) > 1:
            await self._connect_all_parallel()
        else:
            await self._connect_single()
        self._initialized = any((sc.initialized for sc in self._server_clients.values()))
        successful_count = len([sc for sc in self._server_clients.values() if sc.initialized])
        failed_count = len(self._server_configs) - successful_count
        if self.status_callback:
            await self.status_callback('connection_summary', {'message': f'Connected to {successful_count}/{len(self._server_configs)} server(s)' + (f' ({failed_count} failed)' if failed_count > 0 else ''), 'successful_count': successful_count, 'failed_count': failed_count, 'total_count': len(self._server_configs), 'tools_count': len(self.tools)})

    async def _connect_server(self, server_name: str, config: Dict[str, Any]) -> bool:
        """Connect to a single server with circuit breaker integration.

        Returns:
            True on success, False on failure
        """
        server_client = self._server_clients[server_name]
        async with server_client.connection_lock:
            if self._circuit_breaker.should_skip_server(server_name):
                logger.warning(f'Skipping server {server_name} due to circuit breaker')
                server_client.connection_state = ConnectionState.FAILED
                return False
            server_client.connection_state = ConnectionState.CONNECTING
            try:
                server_client.manager_task = asyncio.create_task(self._run_manager(server_name, config))
                await asyncio.wait_for(server_client.connected_event.wait(), timeout=30.0)
                if not server_client.initialized or server_client.connection_state != ConnectionState.CONNECTED:
                    raise MCPConnectionError(f'Failed to connect to {server_name}')
                self._circuit_breaker.record_success(server_name)
                logger.info(f"✅ MCP server '{server_name}' connected successfully!")
                return True
            except Exception as e:
                self._circuit_breaker.record_failure(server_name)
                server_client.connection_state = ConnectionState.FAILED
                logger.error(f'Failed to connect to {server_name}: {e}')
                if server_client.manager_task and (not server_client.manager_task.done()):
                    server_client.disconnect_event.set()
                    try:
                        await asyncio.wait_for(server_client.manager_task, timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"Manager task for {server_name} didn't shutdown gracefully, cancelling")
                        server_client.manager_task.cancel()
                        try:
                            await server_client.manager_task
                        except asyncio.CancelledError:
                            pass
                    except Exception as cleanup_error:
                        logger.error(f'Error cleaning up manager task for {server_name}: {cleanup_error}')
                    finally:
                        server_client.manager_task = None
                return False

    async def _connect_single(self) -> None:
        """Connect to single server."""
        config = self._server_configs[0]
        server_name = config['name']
        success = await self._connect_server(server_name, config)
        if not success:
            raise MCPConnectionError(f'Failed to connect to {server_name}')

    async def _connect_all_parallel(self) -> None:
        """Connect to all servers in parallel."""
        tasks = [self._connect_server(c['name'], c) for c in self._server_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = sum((1 for r in results if r is True))
        logger.info(f'Connected to {successful}/{len(self._server_configs)} servers')

    def _create_transport_context(self, config: Dict[str, Any]):
        """Create the appropriate transport context manager based on config."""
        transport_type = config.get('type', 'stdio')
        server_name = config['name']
        if transport_type == 'stdio':
            command = config.get('command', [])
            args = config.get('args', [])
            logger.debug(f'Setting up stdio transport for {server_name}: command={command}, args={args}')
            if isinstance(command, str):
                full_command = prepare_command(command)
                if args:
                    full_command.extend(args)
            elif isinstance(command, list):
                full_command = command + (args or [])
            else:
                full_command = args or []
            if not full_command:
                raise MCPConnectionError(f'No command specified for stdio transport in {server_name}')
            env = config.get('env', {})
            if env:
                env = {**get_default_environment(), **env}
            else:
                env = get_default_environment()
            substituted_args = []
            for arg in full_command[1:] if len(full_command) > 1 else []:
                if isinstance(arg, str):
                    try:
                        substituted_args.append(substitute_env_variables(arg))
                    except ValueError as e:
                        raise MCPConnectionError(f'Environment variable substitution failed in args: {e}', server_name=server_name) from e
                else:
                    substituted_args.append(arg)
            for key, value in list(env.items()):
                if isinstance(value, str):
                    try:
                        env[key] = substitute_env_variables(value)
                    except ValueError as e:
                        raise MCPConnectionError(f'Environment variable substitution failed for {key}: {e}', server_name=server_name) from e
            cwd = config.get('cwd')
            server_params = StdioServerParameters(command=full_command[0], args=substituted_args, env=env, cwd=cwd)
            from ..logger_config import get_log_session_dir
            log_dir = get_log_session_dir()
            errlog_path = log_dir / f'mcp_{server_name}_stderr.log'
            errlog_file = open(errlog_path, 'w', encoding='utf-8')
            if not hasattr(self, '_errlog_files'):
                self._errlog_files = {}
            self._errlog_files[server_name] = errlog_file
            return stdio_client(server_params, errlog=errlog_file)
        elif transport_type == 'streamable-http':
            url = config['url']
            headers = config.get('headers', {})
            substituted_headers = {}
            for key, value in headers.items():
                if isinstance(value, str):
                    try:
                        substituted_headers[key] = substitute_env_variables(value)
                    except ValueError as e:
                        raise MCPConnectionError(f'Environment variable substitution failed in header {key}: {e}', server_name=server_name) from e
                else:
                    substituted_headers[key] = value
            timeout_raw = config.get('timeout', self.timeout_seconds)
            http_read_timeout_raw = config.get('http_read_timeout', 60 * 5)
            timeout = _ensure_timedelta(timeout_raw, self.timeout_seconds)
            http_read_timeout = _ensure_timedelta(http_read_timeout_raw, 60 * 5)
            return streamablehttp_client(url=url, headers=substituted_headers, timeout=timeout, sse_read_timeout=http_read_timeout)
        else:
            raise MCPConnectionError(f'Unsupported transport type: {transport_type}')

    async def _run_manager(self, server_name: str, config: Dict[str, Any]) -> None:
        """Background task that owns the transport and session contexts for a server."""
        server_client = self._server_clients[server_name]
        connection_successful = False
        try:
            transport_ctx = self._create_transport_context(config)
            async with transport_ctx as session_params:
                read, write = session_params[0:2]
                session_timeout_timedelta = _ensure_timedelta(self.timeout_seconds, 30.0)
                async with ClientSession(read, write, read_timeout_seconds=session_timeout_timedelta) as session:
                    server_client.session = session
                    await session.initialize()
                    await self._discover_capabilities(server_name, config)
                    server_client.initialized = True
                    server_client.connection_state = ConnectionState.CONNECTED
                    connection_successful = True
                    server_client.connected_event.set()
                    logger.info(f"✅ MCP server '{server_name}' connected successfully!")
                    if self.status_callback:
                        await self.status_callback('connected', {'server': server_name, 'message': f"Server '{server_name}' ready"})
                    await server_client.disconnect_event.wait()
        except Exception as e:
            logger.error(f'MCP manager error for {server_name}: {e}', exc_info=True)
            if self.status_callback:
                await self.status_callback('error', {'server': server_name, 'message': f"Failed to connect to MCP server '{server_name}': {e}", 'error': str(e)})
            if not server_client.connected_event.is_set():
                server_client.connected_event.set()
        finally:
            server_client.initialized = False
            server_client.session = None
            if not connection_successful:
                server_client.connection_state = ConnectionState.FAILED
                if not server_client.connected_event.is_set():
                    server_client.connected_event.set()
            else:
                server_client.connection_state = ConnectionState.DISCONNECTED

    async def _discover_capabilities(self, server_name: str, config: Dict[str, Any]) -> None:
        """Discover server capabilities (tools, resources, prompts) with name prefixing for multi-server."""
        logger.debug(f'Discovering capabilities for {server_name}')
        session = self._get_server_session(server_name)
        try:
            server_exclude = config.get('exclude_tools', [])
            combined_exclude = list(set((self.exclude_tools or []) + server_exclude))
            server_allowed = config.get('allowed_tools')
            combined_allowed = server_allowed if server_allowed is not None else self.allowed_tools
            available_tools = await session.list_tools()
            tools_list = getattr(available_tools, 'tools', []) if available_tools else []
            for tool in tools_list:
                if combined_exclude and tool.name in combined_exclude:
                    continue
                if combined_allowed is None or tool.name in combined_allowed:
                    prefixed_name = sanitize_tool_name(tool.name, server_name)
                    self.tools[prefixed_name] = tool
                    self._tool_to_server[prefixed_name] = server_name
            logger.info(f'Discovered capabilities for {server_name}: {len([t for t, s in self._tool_to_server.items() if s == server_name])} tools')
        except Exception as e:
            logger.error(f'Failed to discover server capabilities for {server_name}: {e}', exc_info=True)
            raise MCPConnectionError(f'Failed to discover server capabilities: {e}') from e

    async def disconnect(self) -> None:
        """Disconnect from all MCP servers."""
        if not self._initialized:
            return
        tasks = [self._disconnect_one(name, client) for name, client in self._server_clients.items() if client.connection_state != ConnectionState.DISCONNECTED]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._initialized = False

    async def _disconnect_one(self, server_name: str, server_client: _ServerClient) -> None:
        """Disconnect a single server."""
        server_client.connection_state = ConnectionState.DISCONNECTING
        if server_client.manager_task and (not server_client.manager_task.done()):
            server_client.disconnect_event.set()
            try:
                await asyncio.wait_for(server_client.manager_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"Manager task for {server_name} didn't shutdown gracefully, cancelling")
                server_client.manager_task.cancel()
                try:
                    await server_client.manager_task
                except asyncio.CancelledError:
                    logger.debug(f'Manager task for {server_name} cancelled successfully')
            except Exception as e:
                logger.error(f'Error during manager task shutdown for {server_name}: {e}')
            finally:
                server_client.manager_task = None
        server_client.initialized = False
        server_client.connection_state = ConnectionState.DISCONNECTED

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call an MCP tool with validation and timeout handling.

        Args:
            tool_name: Name of the tool to call (always prefixed as mcp__server__toolname)
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            MCPError: If tool is not available
            MCPConnectionError: If no active session
            MCPValidationError: If arguments are invalid
            MCPTimeoutError: If tool call times out
            MCPServerError: If tool execution fails
        """
        if tool_name not in self.tools:
            available_tools = list(self.tools.keys())
            raise MCPError(f"Tool '{tool_name}' not available", context={'available_tools': available_tools, 'total': len(available_tools)})
        try:
            validated_arguments = validate_tool_arguments(arguments)
        except ValueError as e:
            raise MCPValidationError(f'Invalid tool arguments: {e}', field='arguments', value=arguments, context={'tool_name': tool_name}) from e
        pre_tool_hooks = self.hooks.get(HookType.PRE_TOOL_USE, [])
        for hook in pre_tool_hooks:
            try:
                allowed = await hook(tool_name, validated_arguments)
                if not allowed:
                    raise MCPValidationError('Tool call blocked by pre-tool hook', field='tool_name', value=tool_name, context={'arguments': validated_arguments})
            except Exception as e:
                if isinstance(e, MCPValidationError):
                    raise
                logger.warning(f'Pre-tool hook error for {tool_name}: {e}', exc_info=True)
        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            raise MCPError(f"Tool '{tool_name}' not mapped to any server")
        original_tool_name = tool_name[len(f'mcp__{server_name}__'):]
        session = self._get_server_session(server_name)
        logger.debug(f'Calling tool {original_tool_name} on {server_name} with arguments: {validated_arguments}')
        if self.status_callback:
            await self.status_callback('tool_call_start', {'server': server_name, 'tool': original_tool_name, 'message': f"Calling tool '{original_tool_name}' on server '{server_name}'", 'arguments': validated_arguments})
        try:
            result = await asyncio.wait_for(session.call_tool(original_tool_name, validated_arguments), timeout=self.timeout_seconds)
            logger.debug(f'Tool {original_tool_name} completed successfully on {server_name}')
            if self.status_callback:
                await self.status_callback('tool_call_success', {'server': server_name, 'tool': original_tool_name, 'message': f"Tool '{original_tool_name}' executed successfully"})
            return result
        except asyncio.TimeoutError:
            if self.status_callback:
                await self.status_callback('tool_call_timeout', {'server': server_name, 'tool': original_tool_name, 'message': f"Tool '{original_tool_name}' timed out after {self.timeout_seconds} seconds", 'timeout': self.timeout_seconds})
            self._circuit_breaker.record_failure(server_name)
            raise MCPTimeoutError(f'Tool call timed out after {self.timeout_seconds} seconds', timeout_seconds=self.timeout_seconds, operation=f'call_tool({original_tool_name})', context={'tool_name': original_tool_name, 'server_name': server_name})
        except Exception as e:
            logger.error(f'Tool call failed for {original_tool_name} on {server_name}: {e}', exc_info=True)
            self._circuit_breaker.record_failure(server_name)
            if self.status_callback:
                await self.status_callback('tool_call_error', {'server': server_name, 'tool': original_tool_name, 'message': f"Tool '{original_tool_name}' failed: {e}", 'error': str(e)})
            raise MCPServerError(f'Tool call failed: {e}', server_name=server_name, context={'tool_name': original_tool_name, 'arguments': validated_arguments}) from e

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())

    def is_connected(self) -> bool:
        """Check if any servers are connected."""
        return self._initialized and any((sc.initialized for sc in self._server_clients.values()))

    def get_server_names(self) -> List[str]:
        """Get list of connected server names."""
        return [name for name, sc in self._server_clients.items() if sc.initialized]

    def get_active_sessions(self) -> List[ClientSession]:
        """Return active MCP ClientSession objects for all connected servers."""
        sessions = []
        for server_client in self._server_clients.values():
            if server_client.session is not None and server_client.initialized:
                sessions.append(server_client.session)
        return sessions

    async def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health check on all connected MCP servers.

        Returns:
            Dictionary mapping server names to health status
        """
        health_status = {}
        for server_name, server_client in self._server_clients.items():
            if not server_client.initialized or not server_client.session:
                health_status[server_name] = False
                continue
            try:
                await server_client.session.list_tools()
                health_status[server_name] = True
            except Exception as e:
                logger.warning(f'Health check failed for {server_name}: {e}')
                health_status[server_name] = False
        return health_status

    async def health_check(self) -> bool:
        """
        Perform a health check on all servers.

        Returns:
            True if all connected servers are healthy, False otherwise
        """
        health_status = await self.health_check_all()
        return all(health_status.values()) if health_status else False

    async def _reconnect_failed_servers(self, max_retries: int=3) -> Dict[str, bool]:
        """
        Attempt to reconnect any failed servers with circuit breaker integration.

        Args:
            max_retries: Maximum number of reconnection attempts per server

        Returns:
            Dictionary mapping server names to reconnection success status
        """
        health_status = await self.health_check_all()
        reconnect_results = {}
        for server_name, is_healthy in health_status.items():
            if not is_healthy:
                if self._circuit_breaker.should_skip_server(server_name):
                    logger.warning(f'Skipping reconnection for {server_name} due to circuit breaker')
                    reconnect_results[server_name] = False
                    continue
                logger.info(f'Attempting to reconnect failed server: {server_name}')
                config = next((c for c in self._server_configs if c['name'] == server_name), None)
                if not config:
                    reconnect_results[server_name] = False
                    continue
                success = False
                for attempt in range(max_retries):
                    try:
                        if attempt > 0:
                            await asyncio.sleep(1.0 * 2 ** attempt)
                        server_client = self._server_clients[server_name]
                        await self._disconnect_one(server_name, server_client)
                        server_client.connected_event = asyncio.Event()
                        server_client.disconnect_event = asyncio.Event()
                        server_client.manager_task = asyncio.create_task(self._run_manager(server_name, config))
                        await asyncio.wait_for(server_client.connected_event.wait(), timeout=30.0)
                        if server_client.initialized:
                            self._circuit_breaker.record_success(server_name)
                            success = True
                            logger.info(f'Successfully reconnected server: {server_name}')
                            break
                    except Exception as e:
                        logger.warning(f'Reconnection attempt {attempt + 1} failed for {server_name}: {e}')
                        self._circuit_breaker.record_failure(server_name)
                reconnect_results[server_name] = success
            else:
                reconnect_results[server_name] = True
        return reconnect_results

    async def reconnect(self, max_retries: int=3) -> bool:
        """
        Attempt to reconnect all servers with circuit breaker integration.

        Args:
            max_retries: Maximum number of reconnection attempts
                Uses exponential backoff between retries: 2s, 4s, 8s, 16s...

        Returns:
            True if all reconnections successful, False otherwise
        """
        results = await self._reconnect_failed_servers(max_retries)
        return all(results.values()) if results else False

    async def _cleanup(self) -> None:
        """Comprehensive cleanup of all resources."""
        async with self._cleanup_lock:
            if self._cleanup_done:
                return
            logger.debug('Starting cleanup for MCPClient')
            try:
                await self.disconnect()
                if hasattr(self, '_errlog_files'):
                    for server_name, errlog_file in self._errlog_files.items():
                        try:
                            errlog_file.close()
                        except Exception as e:
                            logger.debug(f'Error closing errlog file for {server_name}: {e}')
                    self._errlog_files.clear()
                self.tools.clear()
                self._tool_to_server.clear()
                self._cleanup_done = True
                logger.debug('Cleanup completed for MCPClient')
            except Exception as e:
                logger.error(f'Error during cleanup: {e}')
                raise

    async def __aenter__(self) -> 'MCPClient':
        """Async context manager entry."""
        self._context_managed = True
        await self.connect()
        return self

    async def __aexit__(self, _exc_type: Optional[type], _exc_val: Optional[BaseException], _exc_tb: Optional[TracebackType]) -> None:
        """Async context manager exit."""
        try:
            await self._cleanup()
        except Exception as e:
            logger.error(f'Error during context manager cleanup: {e}')
        finally:
            self._context_managed = False

    @classmethod
    async def create_and_connect(cls, server_configs: List[Dict[str, Any]], *, timeout_seconds: int=30, allowed_tools: Optional[List[str]]=None, exclude_tools: Optional[List[str]]=None) -> 'MCPClient':
        """
        Create and connect MCP client in one step.

        Args:
            server_configs: List of server configuration dictionaries
            timeout_seconds: Timeout for operations in seconds
            allowed_tools: Optional list of tool names to include
            exclude_tools: Optional list of tool names to exclude

        Returns:
            Connected MCPClient instance
        """
        client = cls(server_configs, timeout_seconds=timeout_seconds, allowed_tools=allowed_tools, exclude_tools=exclude_tools)
        await client.connect()
        return client

def substitute_env_variables(text: str) -> str:
    """Substitute environment variables in text using ${VAR_NAME} pattern.

    Raises:
        ValueError: If referenced environment variable is not set or empty
    """
    if not isinstance(text, str) or '${' not in text:
        return text

    def replace_env_var(match):
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None or env_value.strip() == '':
            raise ValueError(f"Required environment variable '{var_name}' is not set")
        return env_value
    return re.sub('\\$\\{([A-Z_][A-Z0-9_]*)\\}', replace_env_var, text)

def _ensure_timedelta(value: Union[int, float, timedelta], default_seconds: float) -> timedelta:
    """
    Ensure a value is converted to timedelta for consistent timeout handling.

    Raises:
        MCPValidationError: If value is invalid
    """
    if isinstance(value, timedelta):
        if value.total_seconds() <= 0:
            raise MCPValidationError(f'Timeout must be positive, got {value.total_seconds()} seconds', field='timeout', value=value.total_seconds())
        return value
    elif isinstance(value, (int, float)):
        if value <= 0:
            raise MCPValidationError(f'Timeout must be positive, got {value} seconds', field='timeout', value=value)
        return timedelta(seconds=value)
    else:
        logger.warning(f'Invalid timeout value {value}, using default {default_seconds}s')
        return timedelta(seconds=default_seconds)

class FilesystemManager:
    """
    Manages filesystem operations for backends with MCP filesystem support.

    This class handles:
    - Workspace directory lifecycle (creation, cleanup)
    - Snapshot storage and restoration for context sharing
    - Path management for MCP filesystem server configuration
    """

    def __init__(self, cwd: str, agent_temporary_workspace_parent: str=None, context_paths: List[Dict[str, Any]]=None, context_write_access_enabled: bool=False, enforce_read_before_delete: bool=True, enable_image_generation: bool=False, enable_mcp_command_line: bool=False, command_line_allowed_commands: List[str]=None, command_line_blocked_commands: List[str]=None, command_line_execution_mode: str='local', command_line_docker_image: str='massgen/mcp-runtime:latest', command_line_docker_memory_limit: Optional[str]=None, command_line_docker_cpu_limit: Optional[float]=None, command_line_docker_network_mode: str='none', enable_audio_generation: bool=False):
        """
        Initialize FilesystemManager.

        Args:
            cwd: Working directory path for the agent
            agent_temporary_workspace_parent: Parent directory for temporary workspaces
            context_paths: List of context path configurations for access control
            context_write_access_enabled: Whether write access is enabled for context paths
            enforce_read_before_delete: Whether to enforce read-before-delete policy for workspace files
            enable_image_generation: Whether to enable image generation tools
            enable_mcp_command_line: Whether to enable MCP command line execution tool
            command_line_allowed_commands: Whitelist of allowed command patterns (regex)
            command_line_blocked_commands: Blacklist of blocked command patterns (regex)
            command_line_execution_mode: Execution mode - "local" or "docker"
            command_line_docker_image: Docker image to use for containers
            command_line_docker_memory_limit: Memory limit for Docker containers (e.g., "2g")
            command_line_docker_cpu_limit: CPU limit for Docker containers (e.g., 2.0 for 2 CPUs)
            command_line_docker_network_mode: Network mode for Docker containers (none/bridge/host)
        """
        self.agent_id = None
        self.enable_image_generation = enable_image_generation
        self.enable_mcp_command_line = enable_mcp_command_line
        self.command_line_allowed_commands = command_line_allowed_commands
        self.command_line_blocked_commands = command_line_blocked_commands
        self.command_line_execution_mode = command_line_execution_mode
        self.command_line_docker_image = command_line_docker_image
        self.command_line_docker_memory_limit = command_line_docker_memory_limit
        self.command_line_docker_cpu_limit = command_line_docker_cpu_limit
        self.command_line_docker_network_mode = command_line_docker_network_mode
        self.docker_manager = None
        if enable_mcp_command_line and command_line_execution_mode == 'docker':
            from ._docker_manager import DockerManager
            self.docker_manager = DockerManager(image=command_line_docker_image, network_mode=command_line_docker_network_mode, memory_limit=command_line_docker_memory_limit, cpu_limit=command_line_docker_cpu_limit)
        self.enable_audio_generation = enable_audio_generation
        self.path_permission_manager = PathPermissionManager(context_write_access_enabled=context_write_access_enabled, enforce_read_before_delete=enforce_read_before_delete)
        if context_paths:
            self.path_permission_manager.add_context_paths(context_paths)
        self.agent_temporary_workspace_parent = agent_temporary_workspace_parent
        if self.agent_temporary_workspace_parent:
            temp_parent = self.agent_temporary_workspace_parent
            temp_parent_path = Path(temp_parent)
            if not temp_parent_path.is_absolute():
                temp_parent_path = temp_parent_path.resolve()
            self.agent_temporary_workspace_parent = temp_parent_path
            self.clear_temp_workspace()
        self.cwd = self._setup_workspace(cwd)
        self.path_permission_manager.add_path(self.cwd, Permission.WRITE, 'workspace')
        self.path_permission_manager.add_path(self.agent_temporary_workspace_parent, Permission.READ, 'temp_workspace')
        self.snapshot_storage = None
        self.agent_temporary_workspace = None
        self._using_temporary = False
        self._original_cwd = self.cwd

    def setup_orchestration_paths(self, agent_id: str, snapshot_storage: Optional[str]=None, agent_temporary_workspace: Optional[str]=None) -> None:
        """
        Setup orchestration-specific paths for snapshots and temporary workspace.
        Called by orchestrator to configure paths for this specific orchestration.

        Args:
            agent_id: The agent identifier for this orchestration
            snapshot_storage: Base path for storing workspace snapshots
            agent_temporary_workspace: Base path for temporary workspace during context sharing
        """
        logger.info(f'[FilesystemManager.setup_orchestration_paths] Called for agent_id={agent_id}, snapshot_storage={snapshot_storage}, agent_temporary_workspace={agent_temporary_workspace}')
        self.agent_id = agent_id
        if snapshot_storage and self.agent_id:
            self.snapshot_storage = Path(snapshot_storage) / self.agent_id
            self.snapshot_storage.mkdir(parents=True, exist_ok=True)
        if agent_temporary_workspace and self.agent_id:
            self.agent_temporary_workspace = self._setup_workspace(self.agent_temporary_workspace_parent / self.agent_id)
        if self.agent_id:
            log_session_dir = get_log_session_dir()
            if log_session_dir:
                agent_log_dir = log_session_dir / self.agent_id
                agent_log_dir.mkdir(parents=True, exist_ok=True)
        if self.docker_manager and self.agent_id:
            context_paths = self.path_permission_manager.get_context_paths()
            self.docker_manager.create_container(agent_id=self.agent_id, workspace_path=self.cwd, temp_workspace_path=self.agent_temporary_workspace_parent if self.agent_temporary_workspace_parent else None, context_paths=context_paths)
            logger.info(f'[FilesystemManager] Docker container created for agent {self.agent_id}')

    def update_backend_mcp_config(self, backend_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update MCP server configuration with agent_id after it's available.

        This should be called by the backend after setup_orchestration_paths() sets agent_id.

        Args:
            backend_config: Backend configuration dict containing mcp_servers

        Returns:
            Updated backend configuration
        """
        if not self.enable_mcp_command_line or self.command_line_execution_mode != 'docker':
            return backend_config
        if not self.agent_id:
            logger.warning('[FilesystemManager] agent_id not set, cannot update MCP config for Docker mode')
            return backend_config
        mcp_servers = backend_config.get('mcp_servers', [])
        for server in mcp_servers:
            if isinstance(server, dict) and server.get('name') == 'command_line':
                args = server.get('args', [])
                if '--agent-id' not in args:
                    args.extend(['--agent-id', self.agent_id])
                    server['args'] = args
                    logger.info(f'[FilesystemManager] Updated command_line MCP server config with agent_id: {self.agent_id}')
                break
        return backend_config

    def _setup_workspace(self, cwd: str) -> Path:
        """Setup workspace directory, creating if needed and clearing existing files safely."""
        Path(cwd)
        workspace = Path(cwd).resolve()
        if not workspace.is_absolute():
            raise AssertionError('Workspace must be absolute')
        if workspace == Path('/') or len(workspace.parts) < 3:
            raise AssertionError(f'Refusing unsafe workspace path: {workspace}')
        workspace.mkdir(parents=True, exist_ok=True)
        if workspace.exists() and workspace.is_dir():
            for item in workspace.iterdir():
                if item.is_symlink():
                    logger.warning(f'[FilesystemManager.save_snapshot] Skipping symlink during clear: {item}')
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        return workspace

    def get_mcp_filesystem_config(self) -> Dict[str, Any]:
        """
        Generate MCP filesystem server configuration.

        Returns:
            Dictionary with MCP server configuration for filesystem access
        """
        paths = self.path_permission_manager.get_mcp_filesystem_paths()
        config = {'name': 'filesystem', 'type': 'stdio', 'command': 'npx', 'args': ['-y', '@modelcontextprotocol/server-filesystem'] + paths, 'cwd': str(self.cwd), 'exclude_tools': ['read_media_file']}
        return config

    def get_workspace_tools_mcp_config(self) -> Dict[str, Any]:
        """
        Generate workspace tools MCP server configuration.

        Returns:
            Dictionary with MCP server configuration for workspace tools (copy, delete, compare)
        """
        context_paths = self.path_permission_manager.get_context_paths()
        ','.join([cp['path'] for cp in context_paths])
        script_path = Path(wc_module.__file__).resolve()
        paths = self.path_permission_manager.get_mcp_filesystem_paths()
        env = {'FASTMCP_SHOW_CLI_BANNER': 'false'}
        config = {'name': 'workspace_tools', 'type': 'stdio', 'command': 'fastmcp', 'args': ['run', f'{script_path}:create_server'] + ['--', '--allowed-paths'] + paths, 'env': env, 'cwd': str(self.cwd)}
        if not self.enable_image_generation:
            config['exclude_tools'] = ['generate_and_store_image_with_input_images', 'generate_and_store_image_no_input_images']
        if not self.enable_audio_generation:
            if 'exclude_tools' not in config:
                config['exclude_tools'] = []
            config['exclude_tools'].extend(['generate_and_store_audio_with_input_audios', 'generate_and_store_audio_no_input_audios'])
        return config

    def get_command_line_mcp_config(self) -> Dict[str, Any]:
        """
        Generate command line execution MCP server configuration.

        Returns:
            Dictionary with MCP server configuration for command execution
            (supports bash on Unix/Mac, cmd/PowerShell on Windows, and Docker isolation)
        """
        script_path = Path(ce_module.__file__).resolve()
        paths = self.path_permission_manager.get_mcp_filesystem_paths()
        env = {'FASTMCP_SHOW_CLI_BANNER': 'false'}
        if 'DOCKER_HOST' in os.environ:
            env['DOCKER_HOST'] = os.environ['DOCKER_HOST']
        config = {'name': 'command_line', 'type': 'stdio', 'command': 'fastmcp', 'args': ['run', f'{script_path}:create_server', '--', '--allowed-paths'] + paths, 'env': env, 'cwd': str(self.cwd)}
        config['args'].extend(['--execution-mode', self.command_line_execution_mode])
        if self.command_line_execution_mode == 'docker' and self.agent_id:
            config['args'].extend(['--agent-id', self.agent_id])
        if self.command_line_allowed_commands:
            config['args'].extend(['--allowed-commands'] + self.command_line_allowed_commands)
        if self.command_line_blocked_commands:
            config['args'].extend(['--blocked-commands'] + self.command_line_blocked_commands)
        return config

    def inject_filesystem_mcp(self, backend_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject filesystem and workspace tools MCP servers into backend configuration.

        Args:
            backend_config: Original backend configuration

        Returns:
            Modified configuration with MCP servers added
        """
        mcp_servers = backend_config.get('mcp_servers', [])
        if isinstance(mcp_servers, dict):
            existing_names = list(mcp_servers.keys())
            converted_servers = []
            for name, server_config in mcp_servers.items():
                if isinstance(server_config, dict):
                    server = server_config.copy()
                    server['name'] = name
                    converted_servers.append(server)
            mcp_servers = converted_servers
        elif isinstance(mcp_servers, list):
            existing_names = [server.get('name') for server in mcp_servers if isinstance(server, dict)]
        else:
            existing_names = []
            mcp_servers = []
        try:
            if 'filesystem' not in existing_names:
                mcp_servers.append(self.get_mcp_filesystem_config())
            else:
                logger.warning('[FilesystemManager.inject_filesystem_mcp] Custom filesystem MCP server already present')
            if 'workspace_tools' not in existing_names:
                mcp_servers.append(self.get_workspace_tools_mcp_config())
            else:
                logger.warning('[FilesystemManager.inject_filesystem_mcp] Custom workspace_tools MCP server already present')
            if self.enable_mcp_command_line and 'command_line' not in existing_names:
                mcp_servers.append(self.get_command_line_mcp_config())
            elif self.enable_mcp_command_line:
                logger.warning('[FilesystemManager.inject_filesystem_mcp] Custom command_line MCP server already present')
        except Exception as e:
            logger.warning(f'[FilesystemManager.inject_filesystem_mcp] Error checking existing MCP servers: {e}')
        backend_config['mcp_servers'] = mcp_servers
        return backend_config

    def inject_command_line_mcp(self, backend_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject only the command_line MCP server into backend configuration.

        Used for NATIVE backends (like Claude Code) that have built-in filesystem tools
        but need the execute_command MCP tool when using docker mode for code execution.

        Args:
            backend_config: Original backend configuration

        Returns:
            Modified configuration with command_line MCP server added
        """
        mcp_servers = backend_config.get('mcp_servers', [])
        if isinstance(mcp_servers, dict):
            existing_names = list(mcp_servers.keys())
            converted_servers = []
            for name, server_config in mcp_servers.items():
                if isinstance(server_config, dict):
                    server = server_config.copy()
                    server['name'] = name
                    converted_servers.append(server)
            mcp_servers = converted_servers
        elif isinstance(mcp_servers, list):
            existing_names = [server.get('name') for server in mcp_servers if isinstance(server, dict)]
        else:
            existing_names = []
            mcp_servers = []
        try:
            if 'command_line' not in existing_names:
                mcp_servers.append(self.get_command_line_mcp_config())
                logger.info('[FilesystemManager.inject_command_line_mcp] Added command_line MCP server for docker mode')
            else:
                logger.warning('[FilesystemManager.inject_command_line_mcp] Custom command_line MCP server already present')
        except Exception as e:
            logger.warning(f'[FilesystemManager.inject_command_line_mcp] Error adding command_line MCP server: {e}')
        backend_config['mcp_servers'] = mcp_servers
        return backend_config

    def get_pre_tool_hooks(self) -> Dict[str, List]:
        """
        Get pre-tool hooks configuration for MCP clients.

        Returns:
            Dict mapping hook types to lists of hook functions
        """

        async def mcp_hook_wrapper(tool_name: str, tool_args: Dict[str, Any]) -> bool:
            """Wrapper to adapt our hook signature to MCP client expectations."""
            allowed, reason = await self.path_permission_manager.pre_tool_use_hook(tool_name, tool_args)
            if not allowed and reason:
                logger.warning(f'[FilesystemManager] Tool blocked: {tool_name} - {reason}')
            return allowed
        return {HookType.PRE_TOOL_USE: [mcp_hook_wrapper]}

    def get_claude_code_hooks_config(self) -> Dict[str, Any]:
        """
        Get Claude Agent SDK hooks configuration.

        Returns:
            Hooks configuration dict for ClaudeAgentOptions
        """
        return self.path_permission_manager.get_claude_code_hooks_config()

    def enable_write_access(self) -> None:
        """
        Enable write access for this filesystem manager.

        This should be called for final agents to allow them to modify
        files with write permissions in their context paths.
        """
        self.path_permission_manager.context_write_access_enabled = True
        logger.info('[FilesystemManager] Context write access enabled - agent can now modify files with write permissions')

    async def save_snapshot(self, timestamp: Optional[str]=None, is_final: bool=False) -> None:
        """
        Save a snapshot of the workspace. Always saves to snapshot_storage if available (keeping only most recent).
        Additionally saves to log directories if logging is enabled.
        Then, clear the workspace so it is ready for next execution.

        Args:
            timestamp: Optional timestamp to use for the snapshot directory (if not provided, generates one)
            is_final: If True, save as final snapshot for presentation

        TODO: reimplement without 'shutil' and 'os' operations for true async, though we may not need to worry about race conditions here since only one agent writes at a time
        """
        logger.info(f'[FilesystemManager.save_snapshot] Called for agent_id={self.agent_id}, is_final={is_final}, snapshot_storage={self.snapshot_storage}')
        source_dir = self.cwd
        source_path = Path(source_dir)
        if not source_path.exists() or not source_path.is_dir():
            logger.warning(f'[FilesystemManager] Source path invalid - exists: {source_path.exists()}, is_dir: {(source_path.is_dir() if source_path.exists() else False)}')
            return
        if not any(source_path.iterdir()):
            logger.warning(f'[FilesystemManager.save_snapshot] Source path {source_path} is empty, skipping snapshot')
            return
        try:
            if self.snapshot_storage:
                if self.snapshot_storage.exists():
                    shutil.rmtree(self.snapshot_storage)
                self.snapshot_storage.mkdir(parents=True, exist_ok=True)
                items_copied = 0
                for item in source_path.iterdir():
                    if item.is_symlink():
                        logger.warning(f'[FilesystemManager.save_snapshot] Skipping symlink: {item}')
                        continue
                    if item.is_file():
                        shutil.copy2(item, self.snapshot_storage / item.name)
                    elif item.is_dir():
                        shutil.copytree(item, self.snapshot_storage / item.name)
                    items_copied += 1
                logger.info(f'[FilesystemManager] Saved snapshot with {items_copied} items to {self.snapshot_storage}')
            log_session_dir = get_log_session_dir()
            if log_session_dir and self.agent_id:
                if is_final:
                    dest_dir = log_session_dir / 'final' / self.agent_id / 'workspace'
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f'[FilesystemManager.save_snapshot] Final log snapshot dest_dir: {dest_dir}')
                else:
                    if not timestamp:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    dest_dir = log_session_dir / self.agent_id / timestamp / 'workspace'
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f'[FilesystemManager.save_snapshot] Regular log snapshot dest_dir: {dest_dir}')
                items_copied = 0
                for item in source_path.iterdir():
                    if item.is_symlink():
                        logger.warning(f'[FilesystemManager.save_snapshot] Skipping symlink: {item}')
                        continue
                    if item.is_file():
                        shutil.copy2(item, dest_dir / item.name)
                    elif item.is_dir():
                        shutil.copytree(item, dest_dir / item.name, dirs_exist_ok=True)
                    items_copied += 1
                logger.info(f'[FilesystemManager] Saved {('final' if is_final else 'regular')} log snapshot with {items_copied} items to {dest_dir}')
        except Exception as e:
            logger.exception(f'[FilesystemManager.save_snapshot] Snapshot failed: {e}')
            return
        logger.info('[FilesystemManager] Snapshot saved successfully, workspace preserved for logs and debugging')

    def clear_workspace(self) -> None:
        """
        Clear the current workspace to prepare for a new agent execution.

        This should be called at the START of agent execution, not at the end,
        to preserve workspace contents for logging and debugging.
        """
        workspace_path = self.get_current_workspace()
        if not workspace_path.exists() or not workspace_path.is_dir():
            logger.debug(f'[FilesystemManager] Workspace does not exist or is not a directory: {workspace_path}')
            return
        if workspace_path == Path('/') or len(workspace_path.parts) < 3:
            logger.error(f'[FilesystemManager] Refusing to clear unsafe workspace path: {workspace_path}')
            return
        try:
            logger.info('[FilesystemManager] Clearing workspace at agent startup. Current contents:')
            items_to_clear = list(workspace_path.iterdir())
            for item in items_to_clear:
                logger.info(f' - {item}')
                if item.is_symlink():
                    logger.warning(f'[FilesystemManager] Skipping symlink during clear: {item}')
                    continue
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            logger.info('[FilesystemManager] Workspace cleared successfully, ready for new agent execution')
        except Exception as e:
            logger.error(f'[FilesystemManager] Failed to clear workspace: {e}')

    def clear_temp_workspace(self) -> None:
        """
        Clear the temporary workspace parent directory at orchestration startup.

        This clears the entire temp workspace parent (e.g., temp_workspaces/),
        removing all agent directories from previous runs to prevent cross-contamination.
        """
        if not self.agent_temporary_workspace_parent:
            logger.debug('[FilesystemManager] No temp workspace parent configured to clear')
            return
        if not self.agent_temporary_workspace_parent.exists():
            logger.debug(f'[FilesystemManager] Temp workspace parent does not exist: {self.agent_temporary_workspace_parent}')
            return
        if self.agent_temporary_workspace_parent == Path('/') or len(self.agent_temporary_workspace_parent.parts) < 3:
            logger.error(f'[FilesystemManager] Refusing to clear unsafe temp workspace parent path: {self.agent_temporary_workspace_parent}')
            return
        try:
            logger.info(f'[FilesystemManager] Clearing temp workspace parent at orchestration startup: {self.agent_temporary_workspace_parent}')
            items_to_clear = list(self.agent_temporary_workspace_parent.iterdir())
            for item in items_to_clear:
                logger.info(f' - Removing temp workspace item: {item}')
                if item.is_symlink():
                    logger.warning(f'[FilesystemManager] Skipping symlink during temp clear: {item}')
                    continue
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            logger.info('[FilesystemManager] Temp workspace parent cleared successfully')
        except Exception as e:
            logger.error(f'[FilesystemManager] Failed to clear temp workspace parent: {e}')

    async def copy_snapshots_to_temp_workspace(self, all_snapshots: Dict[str, Path], agent_mapping: Dict[str, str]) -> Optional[Path]:
        """
        Copy snapshots from multiple agents to temporary workspace for context sharing.

        This method is called by the orchestrator before starting an agent that needs context from others.
        It copies the latest snapshots from log directories to a temporary workspace.

        Args:
            all_snapshots: Dictionary mapping agent_id to snapshot path (from log directories)
            agent_mapping: Dictionary mapping real agent_id to anonymous agent_id

        Returns:
            Path to the temporary workspace with restored snapshots

        TODO: reimplement without 'shutil' and 'os' operations for true async
        """
        if not self.agent_temporary_workspace:
            return None
        if self.agent_temporary_workspace.exists():
            shutil.rmtree(self.agent_temporary_workspace)
        self.agent_temporary_workspace.mkdir(parents=True, exist_ok=True)
        for agent_id, snapshot_path in all_snapshots.items():
            if snapshot_path.exists() and snapshot_path.is_dir():
                anon_id = agent_mapping.get(agent_id, agent_id)
                dest_dir = self.agent_temporary_workspace / anon_id
                if any(snapshot_path.iterdir()):
                    shutil.copytree(snapshot_path, dest_dir, dirs_exist_ok=True)
        return self.agent_temporary_workspace

    def _log_workspace_contents(self, workspace_path: Path, workspace_name: str, context: str='') -> None:
        """
        Log the contents of a workspace directory for visibility.

        Args:
            workspace_path: Path to the workspace to log
            workspace_name: Human-readable name for the workspace
            context: Additional context (e.g., "before execution", "after execution")
        """
        if not workspace_path or not workspace_path.exists():
            logger.info(f'[FilesystemManager.{workspace_name}] {context} - Workspace does not exist: {workspace_path}')
            return
        try:
            files = list(workspace_path.rglob('*'))
            file_paths = [str(f.relative_to(workspace_path)) for f in files if f.is_file()]
            dir_paths = [str(f.relative_to(workspace_path)) for f in files if f.is_dir()]
            logger.info(f'[FilesystemManager.{workspace_name}] {context} - Workspace: {workspace_path}')
            if file_paths:
                logger.info(f'[FilesystemManager.{workspace_name}] {context} - Files ({len(file_paths)}): {file_paths}')
            if dir_paths:
                logger.info(f'[FilesystemManager.{workspace_name}] {context} - Directories ({len(dir_paths)}): {dir_paths}')
            if not file_paths and (not dir_paths):
                logger.info(f'[FilesystemManager.{workspace_name}] {context} - Empty workspace')
        except Exception as e:
            logger.warning(f'[FilesystemManager.{workspace_name}] {context} - Error reading workspace: {e}')

    def log_current_state(self, context: str='') -> None:
        """
        Log the current state of both main and temp workspaces.

        Args:
            context: Context for the logging (e.g., "before execution", "after answer")
        """
        agent_context = f'agent_id={self.agent_id}, {context}' if context else f'agent_id={self.agent_id}'
        self._log_workspace_contents(self.get_current_workspace(), 'main_workspace', agent_context)
        if self.agent_temporary_workspace:
            self._log_workspace_contents(self.agent_temporary_workspace, 'temp_workspace', agent_context)

    def set_temporary_workspace(self, use_temporary: bool=True) -> None:
        """
        Switch between main workspace and temporary workspace.

        Args:
            use_temporary: If True, use temporary workspace; if False, use main workspace
        """
        self._using_temporary = use_temporary
        if use_temporary and self.agent_temporary_workspace:
            self.cwd = self.agent_temporary_workspace
        else:
            self.cwd = self._original_cwd

    def get_current_workspace(self) -> Path:
        """
        Get the current active workspace path.

        Returns:
            Path to the current workspace
        """
        return self.cwd

    def cleanup(self) -> None:
        """Cleanup temporary resources (not the main workspace) and Docker containers."""
        if self.docker_manager and self.agent_id:
            self.docker_manager.cleanup(self.agent_id)
        p = self.agent_temporary_workspace
        if not p:
            return
        try:
            p = p.resolve()
            if not p.exists():
                return
            assert p.is_absolute(), 'Temporary workspace must be absolute'
            assert p.is_dir(), 'Temporary workspace must be a directory'
            if self.agent_temporary_workspace_parent:
                parent = Path(self.agent_temporary_workspace_parent).resolve()
                try:
                    p.relative_to(parent)
                except ValueError:
                    raise AssertionError(f'Refusing to delete workspace outside of parent: {p}')
            if p == Path('/') or len(p.parts) < 3:
                raise AssertionError(f'Unsafe path for deletion: {p}')
            shutil.rmtree(p)
        except Exception as e:
            logger.warning(f'[FilesystemManager] cleanup failed for {p}: {e}')

class RichTerminalDisplay(TerminalDisplay):
    """Enhanced terminal display using Rich library for beautiful formatting."""

    def __init__(self, agent_ids: List[str], **kwargs: Any) -> None:
        """Initialize rich terminal display.

        Args:
            agent_ids: List of agent IDs to display
            **kwargs: Additional configuration options
                - theme: Color theme ('dark', 'light', 'cyberpunk') (default: 'dark')
                - refresh_rate: Display refresh rate in Hz (default: 4)
                - enable_syntax_highlighting: Enable code syntax highlighting (default: True)
                - max_content_lines: Base lines per agent column before scrolling (default: 8)
                - show_timestamps: Show timestamps for events (default: True)
                - enable_status_jump: Enable jumping to latest status when agent status changes (default: True)
                - truncate_web_search_on_status_change: Truncate web search content when status changes (default: True)
                - max_web_search_lines_on_status_change: Max web search lines to keep on status changes (default: 3)
                - enable_flush_output: Enable flush output for final answer display (default: True)
                - flush_char_delay: Delay between characters in flush output (default: 0.03)
                - flush_word_delay: Extra delay after punctuation in flush output (default: 0.08)
        """
        if not RICH_AVAILABLE:
            raise ImportError('Rich library is required for RichTerminalDisplay. Install with: pip install rich')
        super().__init__(agent_ids, **kwargs)
        self._terminal_performance = self._detect_terminal_performance()
        self.refresh_rate = self._get_adaptive_refresh_rate(kwargs.get('refresh_rate'))
        self.theme = kwargs.get('theme', 'dark')
        self.enable_syntax_highlighting = kwargs.get('enable_syntax_highlighting', True)
        self.max_content_lines = kwargs.get('max_content_lines', 8)
        self.max_line_length = kwargs.get('max_line_length', 100)
        self.show_timestamps = kwargs.get('show_timestamps', True)
        self.console = Console(force_terminal=True, legacy_windows=False)
        self.terminal_size = self.console.size
        self.num_agents = len(agent_ids)
        self.fixed_column_width = max(20, self.terminal_size.width // self.num_agents - 1)
        self.agent_panel_height = max(10, self.terminal_size.height - 13)
        self.orchestrator = kwargs.get('orchestrator', None)
        self._resize_lock = threading.Lock()
        self._setup_resize_handler()
        self.live = None
        self._lock = threading.RLock()
        self._last_update = 0
        self._update_interval = self._get_adaptive_update_interval()
        self._last_full_refresh = 0
        self._full_refresh_interval = self._get_adaptive_full_refresh_interval()
        self._refresh_times: List[float] = []
        self._dropped_frames = 0
        self._performance_check_interval = 5.0
        self._refresh_executor = ThreadPoolExecutor(max_workers=min(len(agent_ids) * 2 + 8, 20))
        self._agent_panels_cache: Dict[str, Panel] = {}
        self._header_cache = None
        self._footer_cache = None
        self._layout_update_lock = threading.Lock()
        self._pending_updates: set[str] = set()
        self._shutdown_flag = False
        self._priority_updates: set[str] = set()
        self._status_update_executor = ThreadPoolExecutor(max_workers=4)
        self._setup_theme()
        self._keyboard_interactive_mode = kwargs.get('keyboard_interactive_mode', True)
        self._safe_keyboard_mode = kwargs.get('safe_keyboard_mode', False)
        self._key_handler = None
        self._input_thread = None
        self._stop_input_thread = False
        self._original_settings = None
        self._agent_selector_active = False
        self._stored_final_presentation = None
        self._stored_presentation_agent = None
        self._stored_vote_results = None
        self._final_presentation_active = False
        self._final_presentation_content = ''
        self._final_presentation_agent = None
        self._final_presentation_vote_results = None
        self.code_patterns = ['```(\\w+)?\\n(.*?)\\n```', '`([^`]+)`', 'def\\s+\\w+\\s*\\(', 'class\\s+\\w+\\s*[:(\\s]', 'import\\s+\\w+', 'from\\s+\\w+\\s+import']
        self.agent_progress = {agent_id: 0 for agent_id in agent_ids}
        self.agent_activity = {agent_id: 'waiting' for agent_id in agent_ids}
        self._last_agent_status = {agent_id: 'waiting' for agent_id in agent_ids}
        self._last_agent_activity = {agent_id: 'waiting' for agent_id in agent_ids}
        self._last_content_hash = {agent_id: '' for agent_id in agent_ids}
        self._debounce_timers: Dict[str, threading.Timer] = {}
        self._debounce_delay = self._get_adaptive_debounce_delay()
        self._critical_updates: set[str] = set()
        self._normal_updates: set[str] = set()
        self._decorative_updates: set[str] = set()
        self._important_content_types = {'presentation', 'status', 'tool', 'error'}
        self._status_change_keywords = {'completed', 'failed', 'waiting', 'error', 'voted', 'voting', 'tool', 'vote recorded'}
        self._important_event_keywords = {'completed', 'failed', 'voting', 'voted', 'final', 'error', 'started', 'coordination', 'tool', 'vote recorded'}
        self._status_jump_enabled = kwargs.get('enable_status_jump', True)
        self._web_search_truncate_on_status_change = kwargs.get('truncate_web_search_on_status_change', True)
        self._max_web_search_lines = kwargs.get('max_web_search_lines_on_status_change', 3)
        self._enable_flush_output = kwargs.get('enable_flush_output', True)
        self._flush_char_delay = kwargs.get('flush_char_delay', 0.03)
        self._flush_word_delay = kwargs.get('flush_word_delay', 0.08)
        from massgen.logger_config import get_log_session_dir
        log_session_dir = get_log_session_dir()
        self.output_dir = kwargs.get('output_dir', log_session_dir / 'agent_outputs')
        self.agent_files: Dict[str, Path] = {}
        self.system_status_file = None
        self._selected_agent = None
        self._setup_agent_files()
        self._text_buffers = {agent_id: '' for agent_id in agent_ids}
        self._max_buffer_length = self._get_adaptive_buffer_length()
        self._buffer_timeout = self._get_adaptive_buffer_timeout()
        self._buffer_timers = {agent_id: None for agent_id in agent_ids}
        self._update_batch = set()
        self._batch_timer = None
        self._batch_timeout = self._get_adaptive_batch_timeout()

    def _setup_resize_handler(self) -> None:
        """Setup SIGWINCH signal handler for terminal resize detection."""
        if not sys.stdin.isatty():
            return
        try:
            signal.signal(signal.SIGWINCH, self._handle_resize_signal)
        except (AttributeError, OSError):
            pass

    def _handle_resize_signal(self, signum: int, frame: Any) -> None:
        """Handle SIGWINCH signal when terminal is resized."""
        threading.Thread(target=self._handle_terminal_resize, daemon=True).start()

    def _handle_terminal_resize(self) -> None:
        """Handle terminal resize by recalculating layout and refreshing display."""
        with self._resize_lock:
            try:
                if self._terminal_performance['type'] == 'vscode':
                    time.sleep(0.05)
                new_size = self.console.size
                if new_size.width != self.terminal_size.width or new_size.height != self.terminal_size.height:
                    self.terminal_size = new_size
                    if self._terminal_performance['type'] == 'vscode':
                        time.sleep(0.02)
                    self._recalculate_layout()
                    self._invalidate_display_cache()
                    with self._lock:
                        self._pending_updates.add('header')
                        self._pending_updates.add('footer')
                        self._pending_updates.update(self.agent_ids)
                        self._schedule_async_update(force_update=True)
                    time.sleep(0.1)
            except Exception:
                pass

    def _recalculate_layout(self) -> None:
        """Recalculate layout dimensions based on current terminal size."""
        self.fixed_column_width = max(20, self.terminal_size.width // self.num_agents - 1)
        self.agent_panel_height = max(10, self.terminal_size.height - 13)

    def _invalidate_display_cache(self) -> None:
        """Invalidate all cached display components to force refresh."""
        self._agent_panels_cache.clear()
        self._header_cache = None
        self._footer_cache = None

    def _setup_agent_files(self) -> None:
        """Setup individual txt files for each agent and system status file."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        for agent_id in self.agent_ids:
            file_path = Path(self.output_dir) / f'{agent_id}.txt'
            self.agent_files[agent_id] = file_path
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f'=== {agent_id.upper()} OUTPUT LOG ===\n\n')
        self.system_status_file = Path(self.output_dir) / 'system_status.txt'
        with open(str(self.system_status_file), 'w', encoding='utf-8') as f:
            f.write('=== SYSTEM STATUS LOG ===\n\n')

    def _detect_terminal_performance(self) -> Dict[str, Any]:
        """Detect terminal performance characteristics for adaptive refresh rates."""
        terminal_info = {'type': 'unknown', 'performance_tier': 'medium', 'supports_unicode': True, 'supports_color': True, 'buffer_size': 'normal'}
        try:
            term = os.environ.get('TERM', '').lower()
            term_program = os.environ.get('TERM_PROGRAM', '').lower()
            if 'iterm.app' in term_program or 'iterm' in term_program.lower():
                terminal_info['performance_tier'] = 'high'
                terminal_info['type'] = 'iterm'
                terminal_info['supports_unicode'] = True
            elif 'vscode' in term_program or 'code' in term_program or self._detect_vscode_terminal():
                terminal_info['performance_tier'] = 'medium'
                terminal_info['type'] = 'vscode'
                terminal_info['supports_unicode'] = True
                terminal_info['buffer_size'] = 'large'
                terminal_info['needs_flush_delay'] = True
                terminal_info['refresh_stabilization'] = True
            elif 'apple_terminal' in term_program or term_program == 'terminal':
                terminal_info['performance_tier'] = 'high'
                terminal_info['type'] = 'macos_terminal'
                terminal_info['supports_unicode'] = True
            elif 'xterm-256color' in term or 'alacritty' in term_program:
                terminal_info['performance_tier'] = 'high'
                terminal_info['type'] = 'modern'
            elif 'screen' in term or 'tmux' in term:
                terminal_info['performance_tier'] = 'low'
                terminal_info['type'] = 'multiplexer'
            elif 'xterm' in term:
                terminal_info['performance_tier'] = 'medium'
                terminal_info['type'] = 'xterm'
            elif term in ['dumb', 'vt100', 'vt220']:
                terminal_info['performance_tier'] = 'low'
                terminal_info['type'] = 'legacy'
                terminal_info['supports_unicode'] = False
            if os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_CLIENT'):
                if terminal_info['performance_tier'] == 'high':
                    terminal_info['performance_tier'] = 'medium'
                elif terminal_info['performance_tier'] == 'medium':
                    terminal_info['performance_tier'] = 'low'
            colorterm = os.environ.get('COLORTERM', '').lower()
            if colorterm in ['truecolor', '24bit']:
                terminal_info['supports_color'] = True
            elif not self.console.is_terminal or term == 'dumb':
                terminal_info['supports_color'] = False
        except Exception:
            terminal_info['performance_tier'] = 'low'
        return terminal_info

    def _detect_vscode_terminal(self) -> bool:
        """Additional VSCode terminal detection using multiple indicators."""
        try:
            vscode_indicators = ['VSCODE_INJECTION', 'VSCODE_PID', 'VSCODE_IPC_HOOK', 'VSCODE_IPC_HOOK_CLI', 'TERM_PROGRAM_VERSION']
            for indicator in vscode_indicators:
                if os.environ.get(indicator):
                    return True
            try:
                import psutil
                current_process = psutil.Process()
                parent = current_process.parent()
                if parent and ('code' in parent.name().lower() or 'vscode' in parent.name().lower()):
                    return True
            except (ImportError, psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            term_program = os.environ.get('TERM_PROGRAM', '').lower()
            if term_program and any((pattern in term_program for pattern in ['code', 'vscode'])):
                return True
            return False
        except Exception:
            return False

    def _get_adaptive_refresh_rate(self, user_override: Optional[int]=None) -> int:
        """Get adaptive refresh rate based on terminal performance."""
        if user_override is not None:
            return user_override
        perf_tier = self._terminal_performance['performance_tier']
        term_type = self._terminal_performance['type']
        if term_type == 'vscode':
            return 2
        refresh_rates = {'high': 10, 'medium': 5, 'low': 2}
        return refresh_rates.get(perf_tier, 8)

    def _get_adaptive_update_interval(self) -> float:
        """Get adaptive update interval based on terminal performance."""
        perf_tier = self._terminal_performance['performance_tier']
        intervals = {'high': 0.02, 'medium': 0.05, 'low': 0.1}
        return intervals.get(perf_tier, 0.05)

    def _get_adaptive_full_refresh_interval(self) -> float:
        """Get adaptive full refresh interval based on terminal performance."""
        perf_tier = self._terminal_performance['performance_tier']
        intervals = {'high': 0.1, 'medium': 0.2, 'low': 0.5}
        return intervals.get(perf_tier, 0.2)

    def _get_adaptive_debounce_delay(self) -> float:
        """Get adaptive debounce delay based on terminal performance."""
        perf_tier = self._terminal_performance['performance_tier']
        term_type = self._terminal_performance['type']
        delays = {'high': 0.01, 'medium': 0.03, 'low': 0.05}
        base_delay = delays.get(perf_tier, 0.03)
        if term_type in ['iterm', 'macos_terminal']:
            base_delay *= 2.0
        return base_delay

    def _get_adaptive_buffer_length(self) -> int:
        """Get adaptive buffer length based on terminal performance."""
        perf_tier = self._terminal_performance['performance_tier']
        term_type = self._terminal_performance['type']
        lengths = {'high': 800, 'medium': 500, 'low': 200}
        base_length = lengths.get(perf_tier, 500)
        if term_type in ['iterm', 'macos_terminal']:
            base_length = min(base_length, 400)
        return base_length

    def _get_adaptive_buffer_timeout(self) -> float:
        """Get adaptive buffer timeout based on terminal performance."""
        perf_tier = self._terminal_performance['performance_tier']
        term_type = self._terminal_performance['type']
        timeouts = {'high': 0.5, 'medium': 1.0, 'low': 2.0}
        base_timeout = timeouts.get(perf_tier, 1.0)
        if term_type in ['iterm', 'macos_terminal']:
            base_timeout *= 1.5
        return base_timeout

    def _get_adaptive_batch_timeout(self) -> float:
        """Get adaptive batch timeout for update batching."""
        perf_tier = self._terminal_performance['performance_tier']
        timeouts = {'high': 0.05, 'medium': 0.1, 'low': 0.2}
        return timeouts.get(perf_tier, 0.1)

    def _monitor_performance(self) -> None:
        """Monitor refresh performance and adjust if needed."""
        time.time()
        if len(self._refresh_times) > 20:
            self._refresh_times = self._refresh_times[-20:]
        if len(self._refresh_times) >= 5:
            avg_refresh_time = sum(self._refresh_times) / len(self._refresh_times)
            expected_refresh_time = 1.0 / self.refresh_rate
            if avg_refresh_time > expected_refresh_time * 2:
                self._dropped_frames += 1
                if self._dropped_frames >= 3:
                    self.refresh_rate = max(2, int(self.refresh_rate * 0.7))
                    self._dropped_frames = 0
                    self._update_interval = 1.0 / self.refresh_rate
                    self._full_refresh_interval *= 1.5
                    if self.live and self.live.is_started:
                        try:
                            self.live.refresh_per_second = self.refresh_rate
                        except Exception:
                            self._fallback_to_simple_display()

    def _create_live_display_with_fallback(self) -> Optional[Live]:
        """Create Live display with terminal compatibility checks and fallback."""
        try:
            if not self._test_terminal_capabilities():
                self._fallback_to_simple_display()
                return None
            live_settings = self._get_adaptive_live_settings()
            live = Live(self._create_layout(), console=self.console, **live_settings)
            try:
                live.start()
                live.stop()
                return live
            except Exception:
                self._fallback_to_simple_display()
                return None
        except Exception:
            self._fallback_to_simple_display()
            return None

    def _test_terminal_capabilities(self) -> bool:
        """Test if terminal supports rich Live display features."""
        try:
            if not self.console.is_terminal:
                return False
            perf_tier = self._terminal_performance['performance_tier']
            term_type = self._terminal_performance['type']
            if term_type == 'legacy' or perf_tier == 'low':
                term = os.environ.get('TERM', '').lower()
                if term in ['dumb', 'vt100']:
                    return False
            if term_type in ['iterm', 'macos_terminal']:
                return True
            test_size = self.console.size
            if test_size.width < 20 or test_size.height < 10:
                return False
            return True
        except Exception:
            return False

    def _get_adaptive_live_settings(self) -> Dict[str, Any]:
        """Get Live display settings adapted to terminal performance."""
        perf_tier = self._terminal_performance['performance_tier']
        settings = {'refresh_per_second': self.refresh_rate, 'vertical_overflow': 'ellipsis', 'transient': False}
        if perf_tier == 'low':
            current_rate = settings['refresh_per_second']
            assert isinstance(current_rate, int)
            settings['refresh_per_second'] = min(current_rate, 3)
            settings['transient'] = True
        elif perf_tier == 'medium':
            current_rate = settings['refresh_per_second']
            assert isinstance(current_rate, int)
            settings['refresh_per_second'] = min(current_rate, 8)
        if self._terminal_performance['type'] == 'multiplexer':
            settings['auto_refresh'] = False
        if self._terminal_performance['type'] in ['iterm', 'macos_terminal']:
            current_rate = settings['refresh_per_second']
            assert isinstance(current_rate, int)
            settings['refresh_per_second'] = min(current_rate, 5)
            settings['transient'] = False
            settings['vertical_overflow'] = 'ellipsis'
        if self._terminal_performance['type'] == 'vscode':
            current_rate = settings['refresh_per_second']
            assert isinstance(current_rate, int)
            settings['refresh_per_second'] = min(current_rate, 6)
            settings['transient'] = False
            settings['vertical_overflow'] = 'ellipsis'
            settings['auto_refresh'] = True
        return settings

    def _fallback_to_simple_display(self) -> None:
        """Fallback to simple console output when Live display is not supported."""
        self._simple_display_mode = True
        try:
            self.console.print('\n[yellow]Terminal compatibility: Using simple display mode[/yellow]')
            self.console.print(f'[dim]Monitoring {len(self.agent_ids)} agents...[/dim]\n')
        except Exception:
            print('\nUsing simple display mode...')
            print(f'Monitoring {len(self.agent_ids)} agents...\n')
        return None

    def _update_display_safe(self) -> None:
        """Safely update display with fallback support and terminal-specific synchronization."""
        term_type = self._terminal_performance['type']
        use_safe_mode = term_type in ['iterm', 'macos_terminal', 'vscode']
        if term_type == 'vscode' and self._terminal_performance.get('refresh_stabilization'):
            time.sleep(0.01)
        try:
            if use_safe_mode:
                with self._layout_update_lock:
                    with self._lock:
                        if hasattr(self, '_simple_display_mode') and self._simple_display_mode:
                            self._update_simple_display()
                        else:
                            self._update_live_display_safe()
            else:
                with self._layout_update_lock:
                    if hasattr(self, '_simple_display_mode') and self._simple_display_mode:
                        self._update_simple_display()
                    else:
                        self._update_live_display()
        except Exception:
            self._fallback_to_simple_display()
        if term_type == 'vscode' and self._terminal_performance.get('needs_flush_delay'):
            time.sleep(0.005)

    def _update_simple_display(self) -> None:
        """Update display in simple mode without Live."""
        try:
            current_time = time.time()
            if not hasattr(self, '_last_simple_update'):
                self._last_simple_update = 0
            if current_time - self._last_simple_update > 2.0:
                status_line = f'[{time.strftime('%H:%M:%S')}] Agents: '
                for agent_id in self.agent_ids:
                    status = self.agent_status.get(agent_id, 'waiting')
                    status_line += f'{agent_id}:{status} '
                try:
                    self.console.print(f'\r{status_line[:80]}', end='')
                except Exception:
                    print(f'\r{status_line[:80]}', end='')
                self._last_simple_update = current_time
        except Exception:
            pass

    def _update_live_display(self) -> None:
        """Update Live display mode."""
        try:
            if self.live:
                self.live.update(self._create_layout())
        except Exception:
            self._fallback_to_simple_display()

    def _update_live_display_safe(self) -> None:
        """Update Live display mode with extra safety for macOS terminals."""
        try:
            if self.live and self.live.is_started:
                import time
                time.sleep(0.001)
                self.live.update(self._create_layout())
            elif self.live:
                try:
                    self.live.start()
                    self.live.update(self._create_layout())
                except Exception:
                    self._fallback_to_simple_display()
        except Exception:
            self._fallback_to_simple_display()

    def _setup_theme(self) -> None:
        """Setup color theme configuration."""
        unified_colors = {'primary': '#0066CC', 'secondary': '#4A90E2', 'success': '#00AA44', 'warning': '#CC6600', 'error': '#CC0000', 'info': '#6633CC', 'text': 'default', 'border': '#4A90E2', 'panel_style': '#4A90E2', 'header_style': 'bold #0066CC'}
        themes = {'dark': unified_colors.copy(), 'light': unified_colors.copy(), 'cyberpunk': {'primary': 'bright_magenta', 'secondary': 'bright_cyan', 'success': 'bright_green', 'warning': 'bright_yellow', 'error': 'bright_red', 'info': 'bright_blue', 'text': 'bright_white', 'border': 'bright_magenta', 'panel_style': 'bright_magenta', 'header_style': 'bold bright_magenta'}}
        self.colors = themes.get(self.theme, themes['dark'])
        if self._terminal_performance['type'] == 'vscode':
            vscode_adjustments = {'primary': '#0066CC', 'secondary': '#4A90E2', 'border': '#4A90E2', 'panel_style': '#4A90E2'}
            self.colors.update(vscode_adjustments)
            self._setup_vscode_emoji_fallbacks()

    def _setup_vscode_emoji_fallbacks(self) -> None:
        """Setup emoji fallbacks for VSCode terminal compatibility."""
        self._emoji_fallbacks = {'🚀': '>>', '🎯': '>', '💭': '...', '⚡': '!', '🎨': '*', '📝': '=', '✅': '[OK]', '❌': '[X]', '⭐': '*', '🔍': '?', '📊': '|'}
        if not self._terminal_performance.get('supports_unicode', True):
            self._use_emoji_fallbacks = True
        else:
            self._use_emoji_fallbacks = False

    def _safe_emoji(self, emoji: str) -> str:
        """Get safe emoji for current terminal, with VSCode fallbacks."""
        if self._terminal_performance['type'] == 'vscode' and self._use_emoji_fallbacks and (emoji in self._emoji_fallbacks):
            return self._emoji_fallbacks[emoji]
        return emoji

    def initialize(self, question: str, log_filename: Optional[str]=None) -> None:
        """Initialize the rich display with question and optional log file."""
        self.log_filename = log_filename
        self.question = question
        self.console.clear()
        from massgen.logger_config import suppress_console_logging
        suppress_console_logging()
        self._create_initial_display()
        if self._keyboard_interactive_mode:
            self._setup_keyboard_handler()
        self.live = self._create_live_display_with_fallback()
        if self.live:
            self.live.start()
        self._write_system_status()

    def _create_initial_display(self) -> None:
        """Create the initial welcome display."""
        welcome_text = Text()
        welcome_text.append('🚀 MassGen Coordination Dashboard 🚀\n', style=self.colors['header_style'])
        welcome_text.append(f'Multi-Agent System with {len(self.agent_ids)} agents\n', style=self.colors['primary'])
        if self.log_filename:
            welcome_text.append(f'📁 Log: {self.log_filename}\n', style=self.colors['info'])
        welcome_text.append(f'🎨 Theme: {self.theme.title()}', style=self.colors['secondary'])
        welcome_panel = Panel(welcome_text, box=DOUBLE, border_style=self.colors['border'], title='[bold]Welcome[/bold]', title_align='center')
        self.console.print(welcome_panel)
        self.console.print()

    def _create_layout(self) -> Layout:
        """Create the main layout structure with cached components."""
        layout = Layout()
        header = self._header_cache if self._header_cache else self._create_header()
        agent_columns = self._create_agent_columns_from_cache()
        footer = self._footer_cache if self._footer_cache else self._create_footer()
        if self._final_presentation_active:
            presentation_panel = self._create_final_presentation_panel()
            layout.split_column(Layout(presentation_panel, name='presentation'), Layout(footer, name='footer', size=8))
        else:
            layout.split_column(Layout(header, name='header', size=5), Layout(agent_columns, name='main'), Layout(footer, name='footer', size=8))
        return layout

    def _create_agent_columns_from_cache(self) -> Columns:
        """Create agent columns using cached panels with fixed widths."""
        agent_panels = []
        for agent_id in self.agent_ids:
            if agent_id in self._agent_panels_cache:
                agent_panels.append(self._agent_panels_cache[agent_id])
            else:
                panel = self._create_agent_panel(agent_id)
                self._agent_panels_cache[agent_id] = panel
                agent_panels.append(panel)
        return Columns(agent_panels, equal=False, expand=False, width=self.fixed_column_width)

    def _create_header(self) -> Panel:
        """Create the header panel."""
        header_text = Text()
        header_text.append('🚀 MassGen Multi-Agent Coordination System', style=self.colors['header_style'])
        if hasattr(self, 'question'):
            header_text.append(f'\n💡 Question: {self.question}', style=self.colors['info'])
        return Panel(Align.center(header_text), box=ROUNDED, border_style=self.colors['border'], height=5)

    def _create_agent_columns(self) -> Columns:
        """Create columns for each agent with fixed widths."""
        agent_panels = []
        for agent_id in self.agent_ids:
            panel = self._create_agent_panel(agent_id)
            agent_panels.append(panel)
        return Columns(agent_panels, equal=False, expand=False, width=self.fixed_column_width)

    def _setup_keyboard_handler(self) -> None:
        """Setup keyboard handler for interactive agent selection."""
        try:
            self._agent_keys = {}
            for i, agent_id in enumerate(self.agent_ids):
                key = str(i + 1)
                self._agent_keys[key] = agent_id
            if self._keyboard_interactive_mode:
                self._start_input_thread()
        except ImportError:
            self._keyboard_interactive_mode = False

    def _start_input_thread(self) -> None:
        """Start background thread for keyboard input during Live mode."""
        if not sys.stdin.isatty():
            return
        self._stop_input_thread = False
        term_type = self._terminal_performance['type']
        if self._safe_keyboard_mode or term_type in ['iterm', 'macos_terminal']:
            self._input_thread = threading.Thread(target=self._input_thread_worker_safe, daemon=True)
            self._input_thread.start()
        else:
            try:
                self._input_thread = threading.Thread(target=self._input_thread_worker_improved, daemon=True)
                self._input_thread.start()
            except Exception:
                self._input_thread = threading.Thread(target=self._input_thread_worker_fallback, daemon=True)
                self._input_thread.start()

    def _input_thread_worker_improved(self) -> None:
        """Improved background thread worker that doesn't interfere with Rich rendering."""
        if not UNIX_TERMINAL_SUPPORT:
            return self._input_thread_worker_fallback()
        try:
            if sys.stdin.isatty():
                self._original_settings = termios.tcgetattr(sys.stdin.fileno())
                new_settings = termios.tcgetattr(sys.stdin.fileno())
                new_settings[3] = new_settings[3] & ~(termios.ICANON | termios.ECHO)
                new_settings[6][termios.VMIN] = 0
                new_settings[6][termios.VTIME] = 1
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, new_settings)
            while not self._stop_input_thread:
                try:
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        char = sys.stdin.read(1)
                        if char:
                            self._handle_key_press(char)
                except (BlockingIOError, OSError):
                    continue
        except (KeyboardInterrupt, EOFError):
            pass
        except Exception:
            pass
        finally:
            self._restore_terminal_settings()

    def _input_thread_worker_fallback(self) -> None:
        """Fallback keyboard input method using simple polling without terminal mode changes."""
        import time
        self.console.print('\n[dim]Keyboard support active. Press keys during Live display:[/dim]')
        self.console.print("[dim]1-{} to open agent files, 's' for system status, 'q' to quit[/dim]\n".format(len(self.agent_ids)))
        try:
            while not self._stop_input_thread:
                time.sleep(0.1)
        except (KeyboardInterrupt, EOFError):
            pass
        except Exception:
            pass

    def _input_thread_worker_safe(self) -> None:
        """Completely safe keyboard input that never changes terminal settings."""
        try:
            while not self._stop_input_thread:
                time.sleep(0.5)
        except Exception:
            pass

    def _restore_terminal_settings(self) -> None:
        """Restore original terminal settings."""
        try:
            if UNIX_TERMINAL_SUPPORT and sys.stdin.isatty():
                if self._original_settings:
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._original_settings)
                    self._original_settings = None
                else:
                    try:
                        current = termios.tcgetattr(sys.stdin.fileno())
                        current[3] = current[3] | termios.ECHO | termios.ICANON
                        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, current)
                    except Exception:
                        pass
        except Exception:
            pass

    def _ensure_clean_keyboard_state(self) -> None:
        """Ensure clean keyboard state before starting agent selector."""
        self._stop_input_thread = True
        if self._input_thread and self._input_thread.is_alive():
            try:
                self._input_thread.join(timeout=0.5)
            except Exception:
                pass
        self._restore_terminal_settings()
        try:
            if UNIX_TERMINAL_SUPPORT and sys.stdin.isatty():
                termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except Exception:
            pass
        import time
        time.sleep(0.1)

    def _handle_key_press(self, key: str) -> None:
        """Handle key press events for agent selection."""
        if key in self._agent_keys:
            agent_id = self._agent_keys[key]
            self._open_agent_in_default_text_editor(agent_id)
        elif key == 's':
            self._open_system_status_in_default_text_editor()
        elif key == 'f':
            self._open_final_presentation_in_default_text_editor()
        elif key == 'q':
            self._stop_input_thread = True
            self._restore_terminal_settings()

    def _open_agent_in_default_text_editor(self, agent_id: str) -> None:
        """Open agent's txt file in default text editor."""
        if agent_id not in self.agent_files:
            return
        file_path = self.agent_files[agent_id]
        if not file_path.exists():
            return
        try:
            if sys.platform == 'darwin':
                subprocess.run(['open', str(file_path)], check=False)
            elif sys.platform.startswith('linux'):
                subprocess.run(['xdg-open', str(file_path)], check=False)
            elif sys.platform == 'win32':
                subprocess.run(['start', str(file_path)], check=False, shell=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self._open_agent_in_external_app(agent_id)

    def _open_agent_in_vscode_new_window(self, agent_id: str) -> None:
        """Open agent's txt file in a new VS Code window."""
        if agent_id not in self.agent_files:
            return
        file_path = self.agent_files[agent_id]
        if not file_path.exists():
            return
        try:
            subprocess.run(['code', '--new-window', str(file_path)], check=False)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self._open_agent_in_external_app(agent_id)

    def _open_system_status_in_default_text_editor(self) -> None:
        """Open system status file in default text editor."""
        if not self.system_status_file or not self.system_status_file.exists():
            return
        try:
            if sys.platform == 'darwin':
                subprocess.run(['open', str(self.system_status_file)], check=False)
            elif sys.platform.startswith('linux'):
                subprocess.run(['xdg-open', str(self.system_status_file)], check=False)
            elif sys.platform == 'win32':
                subprocess.run(['start', str(self.system_status_file)], check=False, shell=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self._open_system_status_in_external_app()

    def _open_final_presentation_in_default_text_editor(self) -> None:
        """Open final presentation file in default text editor."""
        final_presentation_file = None
        if hasattr(self, '_final_presentation_file_path') and self._final_presentation_file_path:
            final_presentation_file = self._final_presentation_file_path
        elif hasattr(self, '_stored_presentation_agent') and self._stored_presentation_agent:
            agent_name = self._stored_presentation_agent
            final_presentation_file = self.output_dir / f'{agent_name}_final_presentation.txt'
        else:
            return
        if not final_presentation_file.exists():
            return
        try:
            if sys.platform == 'darwin':
                subprocess.run(['open', str(final_presentation_file)], check=False)
            elif sys.platform.startswith('linux'):
                subprocess.run(['xdg-open', str(final_presentation_file)], check=False)
            elif sys.platform == 'win32':
                subprocess.run(['start', str(final_presentation_file)], check=False, shell=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    def _open_system_status_in_vscode_new_window(self) -> None:
        """Open system status file in a new VS Code window."""
        if not self.system_status_file or not self.system_status_file.exists():
            return
        try:
            subprocess.run(['code', '--new-window', str(self.system_status_file)], check=False)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self._open_system_status_in_external_app()

    def _open_agent_in_external_app(self, agent_id: str) -> None:
        """Open agent's txt file in external editor or terminal viewer."""
        if agent_id not in self.agent_files:
            return
        file_path = self.agent_files[agent_id]
        if not file_path.exists():
            return
        try:
            if sys.platform == 'darwin':
                editors = ['code', 'subl', 'atom', 'nano', 'vim', 'open']
                for editor in editors:
                    try:
                        if editor == 'open':
                            subprocess.run(['open', '-a', 'TextEdit', str(file_path)], check=False)
                        else:
                            subprocess.run([editor, str(file_path)], check=False)
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
            elif sys.platform.startswith('linux'):
                editors = ['code', 'gedit', 'kate', 'nano', 'vim', 'xdg-open']
                for editor in editors:
                    try:
                        subprocess.run([editor, str(file_path)], check=False)
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
            elif sys.platform == 'win32':
                editors = ['code', 'notepad++', 'notepad']
                for editor in editors:
                    try:
                        subprocess.run([editor, str(file_path)], check=False, shell=True)
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
        except Exception:
            pass

    def _open_system_status_in_external_app(self) -> None:
        """Open system status file in external editor or terminal viewer."""
        if not self.system_status_file or not self.system_status_file.exists():
            return
        try:
            if sys.platform == 'darwin':
                editors = ['code', 'subl', 'atom', 'nano', 'vim', 'open']
                for editor in editors:
                    try:
                        if editor == 'open':
                            subprocess.run(['open', '-a', 'TextEdit', str(self.system_status_file)], check=False)
                        else:
                            subprocess.run([editor, str(self.system_status_file)], check=False)
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
            elif sys.platform.startswith('linux'):
                editors = ['code', 'gedit', 'kate', 'nano', 'vim', 'xdg-open']
                for editor in editors:
                    try:
                        subprocess.run([editor, str(self.system_status_file)], check=False)
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
            elif sys.platform == 'win32':
                editors = ['code', 'notepad++', 'notepad']
                for editor in editors:
                    try:
                        subprocess.run([editor, str(self.system_status_file)], check=False, shell=True)
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
        except Exception:
            pass

    def _show_agent_full_content(self, agent_id: str) -> None:
        """Display full content of selected agent from txt file."""
        if agent_id not in self.agent_files:
            return
        try:
            file_path = self.agent_files[agent_id]
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '[' in content:
                        content = content.replace('[', '\\[')
                self.console.print('\n' + '=' * 80 + '\n')
                header_text = Text()
                header_text.append(f'📄 {agent_id.upper()} - Full Content', style=self.colors['header_style'])
                header_text.append('\nPress any key to return to main view', style=self.colors['info'])
                header_panel = Panel(header_text, box=DOUBLE, border_style=self.colors['border'])
                content_panel = Panel(content, title=f'[bold]{agent_id.upper()} Output[/bold]', border_style=self.colors['border'], box=ROUNDED)
                self.console.print(header_panel)
                self.console.print(content_panel)
                input('Press Enter to return to agent selector...')
                self.console.print('\n' + '=' * 80 + '\n')
        except Exception:
            pass

    def show_agent_selector(self) -> None:
        """Show agent selector and handle user input."""
        if not self._keyboard_interactive_mode or not hasattr(self, '_agent_keys'):
            return
        if self._agent_selector_active:
            return
        self._agent_selector_active = True
        self._ensure_clean_keyboard_state()
        try:
            loop_count = 0
            while True:
                loop_count += 1
                options_text = Text()
                options_text.append("This is a system inspection interface for diving into the multi-agent collaboration behind the scenes in MassGen. It lets you examine each agent's original output and compare it to the final MassGen answer in terms of quality. You can explore the detailed communication, collaboration, voting, and decision-making process.\n", style=self.colors['text'])
                options_text.append('\n🎮 Select an agent to view full output:\n', style=self.colors['primary'])
                for key, agent_id in self._agent_keys.items():
                    options_text.append(f'  {key}: ', style=self.colors['warning'])
                    options_text.append('Inspect the original answer and working log of agent ', style=self.colors['text'])
                    options_text.append(f'{agent_id}\n', style=self.colors['warning'])
                options_text.append('  s: Inspect the orchestrator working log including the voting process\n', style=self.colors['warning'])
                options_text.append('  r: Display coordination table to see the full history of agent interactions and decisions\n', style=self.colors['warning'])
                if self._stored_final_presentation and self._stored_presentation_agent:
                    options_text.append(f'  f: Show final presentation from Selected Agent ({self._stored_presentation_agent})\n', style=self.colors['success'])
                options_text.append('  q: Quit Inspection\n', style=self.colors['info'])
                self.console.print(Panel(options_text, title='[bold]Agent Selector[/bold]', border_style=self.colors['border']))
                try:
                    choice = input('Enter your choice: ').strip().lower()
                    if choice in self._agent_keys:
                        self._show_agent_full_content(self._agent_keys[choice])
                    elif choice == 's':
                        self._show_system_status()
                    elif choice == 'r':
                        self.display_coordination_table()
                    elif choice == 'f' and self._stored_final_presentation:
                        self._redisplay_final_presentation()
                    elif choice == 'q':
                        break
                    else:
                        self.console.print(f'[{self.colors['error']}]Invalid choice. Please try again.[/{self.colors['error']}]')
                except KeyboardInterrupt:
                    break
        finally:
            self._agent_selector_active = True

    def _redisplay_final_presentation(self) -> None:
        """Redisplay the stored final presentation."""
        if not self._stored_final_presentation or not self._stored_presentation_agent:
            self.console.print(f'[{self.colors['error']}]No final presentation stored.[/{self.colors['error']}]')
            return
        self.console.print('\n' + '=' * 80 + '\n')
        self._display_final_presentation_content(self._stored_presentation_agent, self._stored_final_presentation)
        input('\nPress Enter to return to agent selector...')
        self.console.print('\n' + '=' * 80 + '\n')

    def _show_coordination_rounds_table(self) -> None:
        """Display the coordination rounds table with rich formatting."""
        self.display_coordination_table()

    def _show_system_status(self) -> None:
        """Display system status from txt file."""
        if not self.system_status_file or not self.system_status_file.exists():
            self.console.print(f'[{self.colors['error']}]System status file not found.[/{self.colors['error']}]')
            return
        try:
            with open(self.system_status_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if '[' in content:
                    content = content.replace('[', '\\[')
            self.console.print('\n' + '=' * 80 + '\n')
            header_text = Text()
            header_text.append('📊 SYSTEM STATUS - Full Log', style=self.colors['header_style'])
            header_text.append('\nPress any key to return to agent selector', style=self.colors['info'])
            header_panel = Panel(header_text, box=DOUBLE, border_style=self.colors['border'])
            content_panel = Panel(content, title='[bold]System Status Log[/bold]', border_style=self.colors['border'], box=ROUNDED)
            self.console.print(header_panel)
            self.console.print(content_panel)
            input('Press Enter to return to agent selector...')
            self.console.print('\n' + '=' * 80 + '\n')
        except Exception as e:
            self.console.print(f'[{self.colors['error']}]Error reading system status file: {e}[/{self.colors['error']}]')

    def _create_agent_panel(self, agent_id: str) -> Panel:
        """Create a panel for a specific agent."""
        agent_content = self.agent_outputs.get(agent_id, [])
        status = self.agent_status.get(agent_id, 'waiting')
        activity = self.agent_activity.get(agent_id, 'waiting')
        content_text = Text()
        max_lines = max(0, self.agent_panel_height - 3)
        if not agent_content:
            content_text.append('No activity yet...', style=self.colors['text'])
        else:
            for line in agent_content[-max_lines:]:
                formatted_line = self._format_content_line(line)
                content_text.append(formatted_line)
                content_text.append('\n')
        status_emoji = self._get_status_emoji(status, activity)
        status_color = self._get_status_color(status)
        backend_name = self._get_backend_name(agent_id)
        title = f'{status_emoji} {agent_id.upper()}'
        if backend_name != 'Unknown':
            title += f' ({backend_name})'
        if self._keyboard_interactive_mode and hasattr(self, '_agent_keys'):
            agent_key = next((k for k, v in self._agent_keys.items() if v == agent_id), None)
            if agent_key:
                title += f' [Press {agent_key}]'
        return Panel(content_text, title=f'[{status_color}]{title}[/{status_color}]', border_style=status_color, box=ROUNDED, height=self.agent_panel_height, width=self.fixed_column_width)

    def _format_content_line(self, line: str) -> Text:
        """Format a content line with syntax highlighting and styling."""
        formatted = Text()
        if not line.strip():
            return formatted
        if self._is_web_search_content(line):
            return self._format_web_search_line(line)
        is_error_message = any((error_indicator in line for error_indicator in ['❌ Error:', 'Error:', 'Exception:', 'Traceback', '❌']))
        if len(line) > self.max_line_length and (not is_error_message):
            wrapped_lines = []
            remaining = line
            while len(remaining) > self.max_line_length:
                break_point = remaining[:self.max_line_length].rfind(' ')
                if break_point == -1:
                    break_point = self.max_line_length
                wrapped_lines.append(remaining[:break_point])
                remaining = remaining[break_point:].lstrip()
            if remaining:
                wrapped_lines.append(remaining)
            line = '\n'.join(wrapped_lines)
        if line.startswith('→'):
            formatted.append('→ ', style=self.colors['warning'])
            formatted.append(line[2:], style=self.colors['text'])
        elif line.startswith('🎤'):
            formatted.append('🎤 ', style=self.colors['success'])
            formatted.append(line[3:], style=f'bold {self.colors['success']}')
        elif line.startswith('⚡'):
            formatted.append('⚡ ', style=self.colors['warning'])
            if 'jumped to latest' in line:
                formatted.append(line[3:], style=f'bold {self.colors['info']}')
            else:
                formatted.append(line[3:], style=f'italic {self.colors['warning']}')
        elif self._is_code_content(line):
            if self.enable_syntax_highlighting:
                formatted = self._apply_syntax_highlighting(line)
            else:
                formatted.append(line, style=f'bold {self.colors['info']}')
        else:
            formatted.append(line, style=self.colors['text'])
        return formatted

    def _create_final_presentation_panel(self) -> Panel:
        """Create a panel for the final presentation display."""
        if not self._final_presentation_active:
            return None
        content_text = Text()
        if not self._final_presentation_content:
            content_text.append('No activity yet...', style=self.colors['text'])
        else:
            lines = self._final_presentation_content.split('\n')
            available_height = max(10, self.terminal_size.height - 16)
            display_lines = lines[-available_height:] if len(lines) > available_height else lines
            for line in display_lines:
                if line.strip():
                    formatted_line = self._format_content_line(line)
                    content_text.append(formatted_line)
                content_text.append('\n')
        title = f'🎤 Final Presentation from {self._final_presentation_agent}'
        if self._final_presentation_vote_results and self._final_presentation_vote_results.get('vote_counts'):
            vote_count = self._final_presentation_vote_results['vote_counts'].get(self._final_presentation_agent, 0)
            title += f' (Selected with {vote_count} votes)'
        title += ' [Press f]'
        return Panel(content_text, title=f'[{self.colors['success']}]{title}[/{self.colors['success']}]', border_style=self.colors['success'], box=DOUBLE, expand=True)

    def _format_presentation_content(self, content: str) -> Text:
        """Format presentation content with enhanced styling for orchestrator queries."""
        formatted = Text()
        lines = content.split('\n') if '\n' in content else [content]
        for line in lines:
            if not line.strip():
                formatted.append('\n')
                continue
            if line.startswith('**') and line.endswith('**'):
                clean_line = line.strip('*').strip()
                formatted.append(clean_line, style=f'bold {self.colors['success']}')
            elif line.startswith('- ') or line.startswith('• '):
                formatted.append(line[:2], style=self.colors['primary'])
                formatted.append(line[2:], style=self.colors['text'])
            elif line.startswith('#'):
                header_level = len(line) - len(line.lstrip('#'))
                clean_header = line.lstrip('# ').strip()
                if header_level <= 2:
                    formatted.append(clean_header, style=f'bold {self.colors['header_style']}')
                else:
                    formatted.append(clean_header, style=f'bold {self.colors['primary']}')
            elif self._is_code_content(line):
                if self.enable_syntax_highlighting:
                    formatted.append(self._apply_syntax_highlighting(line))
                else:
                    formatted.append(line, style=f'bold {self.colors['info']}')
            else:
                formatted.append(line, style=self.colors['text'])
            if line != lines[-1]:
                formatted.append('\n')
        return formatted

    def _is_web_search_content(self, line: str) -> bool:
        """Check if content is from web search and needs special formatting."""
        web_search_indicators = ['[Provider Tool: Web Search]', '🔍 [Search Query]', '✅ [Provider Tool: Web Search]', '🔍 [Provider Tool: Web Search]']
        return any((indicator in line for indicator in web_search_indicators))

    def _format_web_search_line(self, line: str) -> Text:
        """Format web search content with better truncation and styling."""
        formatted = Text()
        if '[Provider Tool: Web Search] Starting search' in line:
            formatted.append('🔍 ', style=self.colors['info'])
            formatted.append('Web search starting...', style=self.colors['text'])
        elif '[Provider Tool: Web Search] Searching' in line:
            formatted.append('🔍 ', style=self.colors['warning'])
            formatted.append('Searching...', style=self.colors['text'])
        elif '[Provider Tool: Web Search] Search completed' in line:
            formatted.append('✅ ', style=self.colors['success'])
            formatted.append('Search completed', style=self.colors['text'])
        elif any((pattern in line for pattern in ['🔍 [Search Query]', 'Search Query:', '[Search Query]'])):
            query = None
            patterns = [('🔍 [Search Query]', ''), ('[Search Query]', ''), ('Search Query:', ''), ('Query:', '')]
            for pattern, _ in patterns:
                if pattern in line:
                    parts = line.split(pattern, 1)
                    if len(parts) > 1:
                        query = parts[1].strip().strip('\'"').strip()
                        break
            if query:
                formatted.append('🔍 Search: ', style=self.colors['info'])
                formatted.append(f'"{query}"', style=f'italic {self.colors['text']}')
            else:
                formatted.append('🔍 Search query', style=self.colors['info'])
        else:
            max_web_length = min(self.max_line_length // 2, 60)
            if len(line) > max_web_length:
                truncated = line[:max_web_length]
                for break_char in ['. ', '! ', '? ', ', ', ': ']:
                    last_break = truncated.rfind(break_char)
                    if last_break > max_web_length // 2:
                        truncated = truncated[:last_break + 1]
                        break
                line = truncated + '...'
            formatted.append(line, style=self.colors['text'])
        return formatted

    def _should_filter_content(self, content: str, content_type: str) -> bool:
        """Determine if content should be filtered out to reduce noise."""
        if content_type in ['status', 'presentation', 'error']:
            return False
        if len(content) > 1000 and self._is_web_search_content(content):
            url_count = content.count('http')
            technical_indicators = content.count('[') + content.count(']') + content.count('(') + content.count(')')
            if url_count > 5 or technical_indicators > len(content) * 0.1:
                return True
        return False

    def _should_filter_line(self, line: str) -> bool:
        """Determine if a specific line should be filtered out."""
        filter_patterns = ['^\\s*\\([^)]+\\)\\s*$', '^\\s*\\[[^\\]]+\\]\\s*$', '^\\s*https?://\\S+\\s*$', '^\\s*\\.\\.\\.\\s*$']
        for pattern in filter_patterns:
            if re.match(pattern, line):
                return True
        return False

    def _truncate_web_search_content(self, agent_id: str) -> None:
        """Truncate web search content when important status updates occur."""
        if agent_id not in self.agent_outputs or not self.agent_outputs[agent_id]:
            return
        content_lines = self.agent_outputs[agent_id]
        web_search_lines = []
        non_web_search_lines = []
        for line in content_lines:
            if self._is_web_search_content(line):
                web_search_lines.append(line)
            else:
                non_web_search_lines.append(line)
        if len(web_search_lines) > self._max_web_search_lines:
            truncated_web_search = web_search_lines[:1] + ['🔍 ... (web search content truncated due to status update) ...'] + web_search_lines[-(self._max_web_search_lines - 2):]
            recent_non_web = non_web_search_lines[-max(5, self.max_content_lines - len(truncated_web_search)):]
            self.agent_outputs[agent_id] = recent_non_web + truncated_web_search
        if len(web_search_lines) > self._max_web_search_lines:
            self.agent_outputs[agent_id].append('⚡  Status updated - jumped to latest')

    def _is_code_content(self, content: str) -> bool:
        """Check if content appears to be code."""
        for pattern in self.code_patterns:
            if re.search(pattern, content, re.DOTALL | re.IGNORECASE):
                return True
        return False

    def _apply_syntax_highlighting(self, content: str) -> Text:
        """Apply syntax highlighting to content."""
        try:
            language = self._detect_language(content)
            if language:
                return Text(content, style=f'bold {self.colors['info']}')
            else:
                return Text(content, style=f'bold {self.colors['info']}')
        except Exception:
            return Text(content, style=f'bold {self.colors['info']}')

    def _detect_language(self, content: str) -> Optional[str]:
        """Detect programming language from content."""
        content_lower = content.lower()
        if any((keyword in content_lower for keyword in ['def ', 'import ', 'class ', 'python'])):
            return 'python'
        elif any((keyword in content_lower for keyword in ['function', 'var ', 'let ', 'const '])):
            return 'javascript'
        elif any((keyword in content_lower for keyword in ['<', '>', 'html', 'div'])):
            return 'html'
        elif any((keyword in content_lower for keyword in ['{', '}', 'json'])):
            return 'json'
        return None

    def _get_status_emoji(self, status: str, activity: str) -> str:
        """Get emoji for agent status."""
        if status == 'working':
            return '🔄'
        elif status == 'completed':
            if 'voted' in activity.lower():
                return '🗳️'
            elif 'failed' in activity.lower():
                return '❌'
            else:
                return '✅'
        elif status == 'waiting':
            return '⏳'
        else:
            return '❓'

    def _get_status_color(self, status: str) -> str:
        """Get color for agent status."""
        status_colors = {'working': self.colors['warning'], 'completed': self.colors['success'], 'waiting': self.colors['info'], 'failed': self.colors['error']}
        return status_colors.get(status, self.colors['text'])

    def _get_backend_name(self, agent_id: str) -> str:
        """Get backend name for agent."""
        try:
            if hasattr(self, 'orchestrator') and self.orchestrator and hasattr(self.orchestrator, 'agents'):
                agent = self.orchestrator.agents.get(agent_id)
                if agent and hasattr(agent, 'backend') and hasattr(agent.backend, 'get_provider_name'):
                    return agent.backend.get_provider_name()
        except Exception:
            pass
        return 'Unknown'

    def _create_footer(self) -> Panel:
        """Create the footer panel with status and events."""
        footer_content = Text()
        footer_content.append('📊 Agent Status: ', style=self.colors['primary'])
        status_counts = {}
        for status in self.agent_status.values():
            status_counts[status] = status_counts.get(status, 0) + 1
        status_parts = []
        for status, count in status_counts.items():
            emoji = self._get_status_emoji(status, status)
            status_parts.append(f'{emoji} {status.title()}: {count}')
        if self._final_presentation_active:
            status_parts.append('🎤 Final Presentation: Active')
        elif hasattr(self, '_stored_final_presentation') and self._stored_final_presentation:
            status_parts.append('🎤 Final Presentation: Complete')
        footer_content.append(' | '.join(status_parts), style=self.colors['text'])
        footer_content.append('\n')
        if self.orchestrator_events:
            footer_content.append('📋 Recent Events:\n', style=self.colors['primary'])
            recent_events = self.orchestrator_events[-3:]
            for event in recent_events:
                footer_content.append(f'  • {event}\n', style=self.colors['text'])
        if self.log_filename:
            footer_content.append(f'📁 Log: {self.log_filename}\n', style=self.colors['info'])
        if self._keyboard_interactive_mode and hasattr(self, '_agent_keys'):
            if self._safe_keyboard_mode:
                footer_content.append('📂 Safe Mode: Keyboard disabled to prevent rendering issues\n', style=self.colors['warning'])
                footer_content.append(f'Output files saved in: {self.output_dir}/', style=self.colors['info'])
            else:
                footer_content.append('🎮 Live Mode Hotkeys: Press 1-', style=self.colors['primary'])
                hotkeys = f"{len(self.agent_ids)} to open agent files in editor, 's' for system status"
                if hasattr(self, '_stored_final_presentation') and self._stored_final_presentation:
                    hotkeys += ", 'f' for final presentation"
                footer_content.append(hotkeys, style=self.colors['text'])
                footer_content.append(f'\n📂 Output files saved in: {self.output_dir}/', style=self.colors['info'])
        return Panel(footer_content, title='[bold]System Status [Press s][/bold]', border_style=self.colors['border'], box=ROUNDED)

    def update_agent_content(self, agent_id: str, content: str, content_type: str='thinking') -> None:
        """Update content for a specific agent with rich formatting and file output."""
        if agent_id not in self.agent_ids:
            return
        with self._lock:
            if agent_id not in self.agent_outputs:
                self.agent_outputs[agent_id] = []
            self._write_to_agent_file(agent_id, content, content_type)
            is_status_change = content_type in ['status', 'presentation', 'tool'] or any((keyword in content.lower() for keyword in self._status_change_keywords))
            if self._status_jump_enabled and is_status_change and self._web_search_truncate_on_status_change and self.agent_outputs[agent_id]:
                self._truncate_web_search_content(agent_id)
            if self._should_filter_content(content, content_type):
                return
            self._process_content_with_buffering(agent_id, content, content_type)
            self._categorize_update(agent_id, content_type, content)
            is_critical = content_type in ['tool', 'status', 'presentation', 'error'] or any((keyword in content.lower() for keyword in self._status_change_keywords))
            self._schedule_layered_update(agent_id, is_critical)

    def _process_content_with_buffering(self, agent_id: str, content: str, content_type: str) -> None:
        """Process content with buffering to accumulate text chunks."""
        if self._buffer_timers.get(agent_id):
            self._buffer_timers[agent_id].cancel()
            self._buffer_timers[agent_id] = None
        if content_type in ['tool', 'status', 'presentation', 'error'] or '\n' in content:
            self._flush_buffer(agent_id)
            if '\n' in content:
                for line in content.splitlines():
                    if line.strip() and (not self._should_filter_line(line)):
                        self.agent_outputs[agent_id].append(line)
            elif content.strip():
                self.agent_outputs[agent_id].append(content.strip())
            return
        self._text_buffers[agent_id] += content
        buffer = self._text_buffers[agent_id]
        if len(buffer) >= self._max_buffer_length:
            self._flush_buffer(agent_id)
            return
        self._set_buffer_timer(agent_id)

    def _flush_buffer(self, agent_id: str) -> None:
        """Flush the buffer for a specific agent."""
        if agent_id in self._text_buffers and self._text_buffers[agent_id]:
            buffer_content = self._text_buffers[agent_id].strip()
            if buffer_content:
                self.agent_outputs[agent_id].append(buffer_content)
            self._text_buffers[agent_id] = ''
        if self._buffer_timers.get(agent_id):
            self._buffer_timers[agent_id].cancel()
            self._buffer_timers[agent_id] = None

    def _set_buffer_timer(self, agent_id: str) -> None:
        """Set a timer to flush the buffer after a timeout."""
        if self._shutdown_flag:
            return
        if self._buffer_timers.get(agent_id):
            self._buffer_timers[agent_id].cancel()

        def timeout_flush() -> None:
            with self._lock:
                if agent_id in self._text_buffers and self._text_buffers[agent_id]:
                    self._flush_buffer(agent_id)
                    self._pending_updates.add(agent_id)
                    self._schedule_async_update(force_update=True)
        self._buffer_timers[agent_id] = threading.Timer(self._buffer_timeout, timeout_flush)
        self._buffer_timers[agent_id].start()

    def _write_to_agent_file(self, agent_id: str, content: str, content_type: str) -> None:
        """Write content to agent's individual txt file."""
        if agent_id not in self.agent_files:
            return
        if content_type == 'debug':
            return
        try:
            file_path = self.agent_files[agent_id]
            timestamp = time.strftime('%H:%M:%S')
            has_emoji = any((ord(char) > 127 and ord(char) in range(128512, 128591) or ord(char) in range(127744, 128511) or ord(char) in range(128640, 128767) or (ord(char) in range(9728, 9983)) or (ord(char) in range(9984, 10175)) for char in content))
            if has_emoji:
                formatted_content = f'\n[{timestamp}] {content}\n'
            else:
                formatted_content = f'{content}'
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(formatted_content)
        except Exception:
            pass

    def _write_system_status(self) -> None:
        """Write current system status to system status file - shows orchestrator events chronologically by time."""
        if not self.system_status_file:
            return
        try:
            with open(self.system_status_file, 'w', encoding='utf-8') as f:
                f.write('=== SYSTEM STATUS LOG ===\n\n')
                f.write('📊 Agent Status:\n')
                status_counts = {}
                for status in self.agent_status.values():
                    status_counts[status] = status_counts.get(status, 0) + 1
                for status, count in status_counts.items():
                    emoji = self._get_status_emoji(status, status)
                    f.write(f'  {emoji} {status.title()}: {count}\n')
                if self._final_presentation_active:
                    f.write('  🎤 Final Presentation: Active\n')
                elif hasattr(self, '_stored_final_presentation') and self._stored_final_presentation:
                    f.write('  🎤 Final Presentation: Complete\n')
                f.write('\n')
                f.write('📋 Orchestrator Events:\n')
                if self.orchestrator_events:
                    for event in self.orchestrator_events:
                        f.write(f'  • {event}\n')
                else:
                    f.write('  • No orchestrator events yet\n')
                f.write('\n')
        except Exception:
            pass

    def update_agent_status(self, agent_id: str, status: str) -> None:
        """Update status for a specific agent with rich indicators."""
        if agent_id not in self.agent_ids:
            return
        with self._lock:
            old_status = self.agent_status.get(agent_id, 'waiting')
            last_tracked_status = self._last_agent_status.get(agent_id, 'waiting')
            current_activity = self.agent_activity.get(agent_id, '')
            is_vote_status = 'voted' in status.lower() or 'voted' in current_activity.lower()
            should_update = old_status != status and last_tracked_status != status or is_vote_status
            if should_update:
                if self._status_jump_enabled and self._web_search_truncate_on_status_change and (old_status != status) and (agent_id in self.agent_outputs) and self.agent_outputs[agent_id]:
                    self._truncate_web_search_content(agent_id)
                super().update_agent_status(agent_id, status)
                self._last_agent_status[agent_id] = status
                self._priority_updates.add(agent_id)
                self._pending_updates.add(agent_id)
                self._pending_updates.add('footer')
                self._schedule_priority_update(agent_id)
                self._schedule_async_update(force_update=True)
                self._write_system_status()
            elif old_status != status:
                super().update_agent_status(agent_id, status)

    def add_orchestrator_event(self, event: str) -> None:
        """Add an orchestrator coordination event with timestamp."""
        with self._lock:
            if self.show_timestamps:
                timestamp = time.strftime('%H:%M:%S')
                formatted_event = f'[{timestamp}] {event}'
            else:
                formatted_event = event
            if hasattr(self, 'orchestrator_events') and self.orchestrator_events and (self.orchestrator_events[-1] == formatted_event):
                return
            super().add_orchestrator_event(formatted_event)
            if any((keyword in event.lower() for keyword in self._important_event_keywords)):
                self._pending_updates.add('footer')
                self._schedule_async_update(force_update=True)
                self._write_system_status()

    def display_vote_results(self, vote_results: Dict[str, Any]) -> None:
        """Display voting results in a formatted rich panel."""
        if not vote_results or not vote_results.get('vote_counts'):
            return
        self.live is not None
        if self.live:
            self.live.stop()
            self.live = None
        vote_counts = vote_results.get('vote_counts', {})
        voter_details = vote_results.get('voter_details', {})
        winner = vote_results.get('winner')
        is_tie = vote_results.get('is_tie', False)
        vote_content = Text()
        vote_content.append('📊 Vote Count:\n', style=self.colors['primary'])
        for agent_id, count in sorted(vote_counts.items(), key=lambda x: x[1], reverse=True):
            winner_mark = '🏆' if agent_id == winner else '  '
            tie_mark = ' (tie-broken)' if is_tie and agent_id == winner else ''
            vote_content.append(f'   {winner_mark} {agent_id}: {count} vote{('s' if count != 1 else '')}{tie_mark}\n', style=self.colors['success'] if agent_id == winner else self.colors['text'])
        if voter_details:
            vote_content.append('\n🔍 Vote Details:\n', style=self.colors['primary'])
            for voted_for, voters in voter_details.items():
                vote_content.append(f'   → {voted_for}:\n', style=self.colors['info'])
                for voter_info in voters:
                    voter = voter_info['voter']
                    reason = voter_info['reason']
                    vote_content.append(f'     • {voter}: "{reason}"\n', style=self.colors['text'])
        agent_mapping = vote_results.get('agent_mapping', {})
        if agent_mapping:
            vote_content.append('\n🔀 Agent Mapping:\n', style=self.colors['primary'])
            for anon_id, real_id in sorted(agent_mapping.items()):
                vote_content.append(f'   {anon_id} → {real_id}\n', style=self.colors['info'])
        if is_tie:
            vote_content.append('\n⚖️  Tie broken by agent registration order\n', style=self.colors['warning'])
        total_votes = vote_results.get('total_votes', 0)
        agents_voted = vote_results.get('agents_voted', 0)
        vote_content.append(f'\n📈 Summary: {agents_voted}/{total_votes} agents voted', style=self.colors['info'])
        voting_panel = Panel(vote_content, title='[bold bright_cyan]🗳️  VOTING RESULTS[/bold bright_cyan]', border_style=self.colors['primary'], box=DOUBLE, expand=False)
        self.console.print(voting_panel)

    def display_coordination_table(self) -> None:
        """Display the coordination table showing the full coordination flow."""
        try:
            self.live is not None
            if self.live:
                self.live.stop()
                self.live = None
            if not hasattr(self, 'orchestrator') or not self.orchestrator:
                print('No orchestrator available for table generation')
                return
            tracker = getattr(self.orchestrator, 'coordination_tracker', None)
            if not tracker:
                print('No coordination tracker available')
                return
            events_data = [event.to_dict() for event in tracker.events]
            session_data = {'session_metadata': {'user_prompt': tracker.user_prompt, 'agent_ids': tracker.agent_ids, 'start_time': tracker.start_time, 'end_time': tracker.end_time, 'final_winner': tracker.final_winner}, 'events': events_data}
            from massgen.frontend.displays.create_coordination_table import CoordinationTableBuilder
            builder = CoordinationTableBuilder(session_data)
            result = builder.generate_rich_event_table()
            if result:
                legend, rich_table = result
                from rich.console import Console
                from rich.panel import Panel
                from rich.text import Text
                from massgen.frontend.displays.create_coordination_table import display_scrollable_content_macos, display_with_native_pager, get_optimal_display_method
                temp_console = Console()
                content = []
                title_text = Text()
                title_text.append('📊 COORDINATION TABLE', style='bold bright_green')
                title_text.append('\n\nNavigation: ↑/↓ or j/k to scroll, q to quit', style='dim cyan')
                title_panel = Panel(title_text, border_style='bright_blue', padding=(1, 2))
                content.append(title_panel)
                content.append('')
                content.append(rich_table)
                if legend:
                    content.append('')
                    content.append('')
                    content.append(legend)
                display_method = get_optimal_display_method()
                try:
                    if display_method == 'macos_simple':
                        display_scrollable_content_macos(temp_console, content, '📊 COORDINATION TABLE')
                    elif display_method == 'native_pager':
                        display_with_native_pager(temp_console, content, '📊 COORDINATION TABLE')
                    else:
                        with temp_console.pager(styles=True):
                            for item in content:
                                temp_console.print(item)
                except (KeyboardInterrupt, EOFError):
                    pass
                self.console.print('\n' + '=' * 80 + '\n')
            else:
                table_content = builder.generate_event_table()
                table_panel = Panel(table_content, title='[bold bright_green]📊 COORDINATION TABLE[/bold bright_green]', border_style=self.colors['success'], box=DOUBLE, expand=False)
                self.console.print('\n')
                self.console.print(table_panel)
                self.console.print()
        except Exception as e:
            print(f'Error displaying coordination table: {e}')
            import traceback
            traceback.print_exc()

    async def display_final_presentation(self, selected_agent: str, presentation_stream: Any, vote_results: Optional[Dict[str, Any]]=None) -> None:
        """Display final presentation with streaming box followed by clean final answer box."""
        if not selected_agent:
            return ''
        self._final_presentation_active = True
        self._final_presentation_content = ''
        self._final_presentation_agent = selected_agent
        self._final_presentation_vote_results = vote_results
        self._final_presentation_file_path = None
        self.console.print('\n')
        was_live = self.live is not None and self.live.is_started
        if not was_live:
            self.console.clear()
            self.live = Live(self._create_layout(), console=self.console, refresh_per_second=self.refresh_rate, vertical_overflow='ellipsis', transient=False)
            self.live.start()
        self._update_footer_cache()
        self._update_final_presentation_panel()
        presentation_content = ''
        chunk_count = 0
        presentation_file_path = self._initialize_final_presentation_file(selected_agent)
        self._final_presentation_file_path = presentation_file_path
        try:
            async for chunk in presentation_stream:
                chunk_count += 1
                content = getattr(chunk, 'content', '') or ''
                chunk_type = getattr(chunk, 'type', '')
                source = getattr(chunk, 'source', selected_agent)
                if chunk_type == 'debug':
                    continue
                if content:
                    if isinstance(content, list):
                        content = ' '.join((str(item) for item in content))
                    elif not isinstance(content, str):
                        content = str(content)
                    processed_content = self.process_reasoning_content(chunk_type, content, source)
                    self._final_presentation_content += processed_content
                    presentation_content += processed_content
                    if processed_content.strip():
                        truncated_content = processed_content.strip()[:150]
                        if len(processed_content.strip()) > 150:
                            truncated_content += '...'
                        self.add_orchestrator_event(f'🎤 {selected_agent}: {truncated_content}')
                    self._append_to_final_presentation_file(presentation_file_path, processed_content)
                    self._update_final_presentation_panel()
                else:
                    processed_content = self.process_reasoning_content(chunk_type, '', source)
                    if processed_content:
                        self._final_presentation_content += processed_content
                        presentation_content += processed_content
                        self._append_to_final_presentation_file(presentation_file_path, processed_content)
                        self._update_final_presentation_panel()
                if chunk_type == 'done':
                    break
        except Exception as e:
            error_msg = f'\n❌ Error during final presentation: {e}\n'
            self._final_presentation_content += error_msg
            self._update_final_presentation_panel()
            if hasattr(self, 'orchestrator') and self.orchestrator:
                try:
                    status = self.orchestrator.get_status()
                    if selected_agent in status.get('agent_states', {}):
                        stored_answer = status['agent_states'][selected_agent].get('answer', '')
                        if stored_answer:
                            fallback_msg = f'\n📋 Fallback to stored answer:\n{stored_answer}\n'
                            self._final_presentation_content += fallback_msg
                            presentation_content = stored_answer
                            self._update_final_presentation_panel()
                except Exception:
                    pass
        if presentation_content:
            self._stored_final_presentation = presentation_content
            self._stored_presentation_agent = selected_agent
            self._stored_vote_results = vote_results
            self._update_footer_cache()
        self._finalize_final_presentation_file(presentation_file_path)
        if self.live and self.live.is_started:
            self.live.stop()
            self.live = None
        self._final_presentation_active = False
        self._update_footer_cache()
        stats_text = Text()
        stats_text.append('✅ Presentation completed by ', style='bold green')
        stats_text.append(selected_agent, style=f'bold {self.colors['success']}')
        if chunk_count > 0:
            stats_text.append(f' | 📊 {chunk_count} chunks processed', style='dim')
        summary_panel = Panel(stats_text, border_style='green', box=ROUNDED, expand=True)
        self.console.print(summary_panel)
        return presentation_content

    def _format_multiline_content(self, content: str) -> Text:
        """Format multiline content for display in a panel."""
        formatted = Text()
        lines = content.split('\n')
        for line in lines:
            if line.strip():
                formatted_line = self._format_content_line(line)
                formatted.append(formatted_line)
            formatted.append('\n')
        return formatted

    def show_final_answer(self, answer: str, vote_results: Dict[str, Any]=None, selected_agent: str=None):
        """Display the final coordinated answer prominently with voting results, final presentation, and agent selector."""
        with self._lock:
            self._flush_all_buffers()
        if self.live:
            self.live.stop()
            self.live = None
        if vote_results is None or selected_agent is None:
            try:
                if hasattr(self, 'orchestrator') and self.orchestrator:
                    status = self.orchestrator.get_status()
                    vote_results = vote_results or status.get('vote_results', {})
                    selected_agent = selected_agent or status.get('selected_agent')
            except Exception:
                pass
        with self._lock:
            for agent_id in self.agent_ids:
                self._pending_updates.add(agent_id)
            self._pending_updates.add('footer')
            self._schedule_async_update(force_update=True)
        time.sleep(0.5)
        self._force_display_final_vote_statuses()
        time.sleep(0.5)
        if vote_results and vote_results.get('vote_counts'):
            self.display_vote_results(vote_results)
            time.sleep(1.0)
        if selected_agent:
            selected_agent_text = Text(f'🏆 Selected agent: {selected_agent}', style=self.colors['success'])
        else:
            is_timeout = False
            if hasattr(self, 'orchestrator') and self.orchestrator:
                is_timeout = getattr(self.orchestrator, 'is_orchestrator_timeout', False)
            if is_timeout:
                selected_agent_text = Text()
                selected_agent_text.append('No agent selected\n', style=self.colors['warning'])
                selected_agent_text.append('The orchestrator timed out before any agent could complete voting or provide an answer.', style=self.colors['warning'])
            else:
                selected_agent_text = Text('No agent selected', style=self.colors['warning'])
        final_panel = Panel(Align.center(selected_agent_text), title='[bold bright_green]🎯 FINAL COORDINATED ANSWER[/bold bright_green]', border_style=self.colors['success'], box=DOUBLE, expand=False)
        self.console.print(final_panel)
        if selected_agent:
            selection_text = Text()
            selection_text.append(f'✅ Selected by: {selected_agent}', style=self.colors['success'])
            if vote_results and vote_results.get('vote_counts'):
                vote_summary = ', '.join([f'{agent}: {count}' for agent, count in vote_results['vote_counts'].items()])
                selection_text.append(f'\n🗳️ Vote results: {vote_summary}', style=self.colors['info'])
            selection_panel = Panel(selection_text, border_style=self.colors['info'], box=ROUNDED)
            self.console.print(selection_panel)
        if selected_agent and hasattr(self, 'orchestrator') and self.orchestrator:
            try:
                self._show_orchestrator_final_presentation(selected_agent, vote_results)
                time.sleep(1.0)
            except Exception as e:
                error_text = Text(f'❌ Error getting final presentation: {e}', style=self.colors['error'])
                self.console.print(error_text)
        if self._keyboard_interactive_mode and hasattr(self, '_agent_keys') and (not self._safe_keyboard_mode):
            self.show_agent_selector()

    def _display_answer_with_flush(self, answer: str) -> None:
        """Display answer with flush output effect - streaming character by character."""
        import sys
        import time
        char_delay = self._flush_char_delay
        word_delay = self._flush_word_delay
        line_delay = 0.2
        try:
            lines = answer.split('\n')
            for line_idx, line in enumerate(lines):
                if not line.strip():
                    self.console.print()
                    continue
                for i, char in enumerate(line):
                    styled_char = Text(char, style=self.colors['text'])
                    self.console.print(styled_char, end='', highlight=False)
                    sys.stdout.flush()
                    if char in [' ', ',', ';']:
                        time.sleep(word_delay)
                    elif char in ['.', '!', '?', ':']:
                        time.sleep(word_delay * 2)
                    else:
                        time.sleep(char_delay)
                if line_idx < len(lines) - 1:
                    self.console.print()
                    time.sleep(line_delay)
            self.console.print()
        except KeyboardInterrupt:
            self.console.print(f'\n{Text(answer, style=self.colors['text'])}')
        except Exception:
            self.console.print(Text(answer, style=self.colors['text']))

    def _get_selected_agent_final_answer(self, selected_agent: str) -> str:
        """Get the final provided answer from the selected agent."""
        if not selected_agent:
            return ''
        try:
            if hasattr(self, 'orchestrator') and self.orchestrator:
                status = self.orchestrator.get_status()
                if hasattr(self.orchestrator, 'agent_states') and selected_agent in self.orchestrator.agent_states:
                    stored_answer = self.orchestrator.agent_states[selected_agent].answer
                    if stored_answer:
                        return stored_answer.replace('\\', '\n').replace('**', '').strip()
                if 'agent_states' in status and selected_agent in status['agent_states']:
                    agent_state = status['agent_states'][selected_agent]
                    if hasattr(agent_state, 'answer') and agent_state.answer:
                        return agent_state.answer.replace('\\', '\n').replace('**', '').strip()
                    elif isinstance(agent_state, dict) and 'answer' in agent_state:
                        return agent_state['answer'].replace('\\', '\n').replace('**', '').strip()
        except Exception:
            pass
        if selected_agent not in self.agent_outputs:
            return ''
        agent_output = self.agent_outputs[selected_agent]
        if not agent_output:
            return ''
        answer_lines = []
        for line in reversed(agent_output):
            line = line.strip()
            if not line:
                continue
            if any((marker in line for marker in ['⚡', '🔄', '✅', '🗳️', '❌', 'voted', '🔧', 'status'])):
                continue
            if any((marker in line.lower() for marker in ['final coordinated', 'coordination', 'voting'])):
                break
            answer_lines.insert(0, line)
            if len(answer_lines) >= 10 or len('\n'.join(answer_lines)) > 500:
                break
        if answer_lines:
            answer = '\n'.join(answer_lines).strip()
            answer = answer.replace('**', '').replace('##', '').strip()
            return answer
        return ''

    def _extract_presentation_content(self, selected_agent: str) -> str:
        """Extract presentation content from the selected agent's output."""
        if selected_agent not in self.agent_outputs:
            return ''
        agent_output = self.agent_outputs[selected_agent]
        presentation_lines = []
        collecting_presentation = False
        for line in agent_output:
            if '🎤' in line or 'presentation' in line.lower():
                collecting_presentation = True
                continue
            if not line.strip() or line.startswith('⚡') or line.startswith('🔄'):
                continue
            if collecting_presentation and line.strip():
                if any((marker in line for marker in ['✅', '🗳️', '🔄', '❌', 'voted', 'Final', 'coordination'])):
                    break
                presentation_lines.append(line.strip())
        if not presentation_lines and agent_output:
            for line in reversed(agent_output[-10:]):
                if line.strip() and (not line.startswith('⚡')) and (not line.startswith('🔄')) and (not any((marker in line for marker in ['voted', '🗳️', '✅', 'status']))):
                    presentation_lines.insert(0, line.strip())
                    if len(presentation_lines) >= 5:
                        break
        return '\n'.join(presentation_lines) if presentation_lines else ''

    def _display_final_presentation_content(self, selected_agent: str, presentation_content: str) -> None:
        """Display the final presentation content in a formatted panel with orchestrator query enhancements."""
        if not presentation_content.strip():
            return
        self._stored_final_presentation = presentation_content
        self._stored_presentation_agent = selected_agent
        header_text = Text()
        header_text.append(f'🎤 Final Presentation from {selected_agent}', style=self.colors['header_style'])
        header_panel = Panel(Align.center(header_text), border_style=self.colors['success'], box=DOUBLE, title='[bold]Final Presentation[/bold]')
        self.console.print(header_panel)
        self.console.print('=' * 60)
        content_text = Text()
        formatted_content = self._format_presentation_content(presentation_content)
        content_text.append(formatted_content)
        content_panel = Panel(content_text, title=f'[bold]{selected_agent.upper()} Final Presentation[/bold]', border_style=self.colors['primary'], box=ROUNDED, subtitle='[italic]Final presentation content[/italic]')
        self.console.print(content_panel)
        self.console.print('=' * 60)
        completion_text = Text()
        completion_text.append('✅ Final presentation completed successfully', style=self.colors['success'])
        completion_panel = Panel(Align.center(completion_text), border_style=self.colors['success'], box=ROUNDED)
        self.console.print(completion_panel)
        self._save_final_presentation_to_file(selected_agent, presentation_content)

    def _save_final_presentation_to_file(self, selected_agent: str, presentation_content: str) -> None:
        """Save the final presentation content to a text file in agent_outputs directory."""
        try:
            filename = f'final_presentation_{selected_agent}.txt'
            file_path = Path(self.output_dir) / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f'=== FINAL PRESENTATION FROM {selected_agent.upper()} ===\n')
                f.write(f'Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n')
                f.write('=' * 60 + '\n\n')
                f.write(presentation_content)
                f.write('\n\n' + '=' * 60 + '\n')
                f.write('End of Final Presentation\n')
            latest_link = Path(self.output_dir) / f'final_presentation_{selected_agent}_latest.txt'
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(filename)
        except Exception:
            pass

    def _initialize_final_presentation_file(self, selected_agent: str) -> Path:
        """Initialize a new final presentation file and return the file path."""
        try:
            filename = f'final_presentation_{selected_agent}.txt'
            file_path = Path(self.output_dir) / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f'=== FINAL PRESENTATION FROM {selected_agent.upper()} ===\n')
                f.write(f'Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n')
                f.write('=' * 60 + '\n\n')
            latest_link = Path(self.output_dir) / f'final_presentation_{selected_agent}_latest.txt'
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(filename)
            return file_path
        except Exception:
            return None

    def _append_to_final_presentation_file(self, file_path: Path, content: str) -> None:
        """Append content to the final presentation file."""
        try:
            if file_path and file_path.exists():
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(content)
                    f.flush()
                    import os
                    os.fsync(f.fileno())
        except Exception:
            pass

    def _finalize_final_presentation_file(self, file_path: Path) -> None:
        """Add closing content to the final presentation file."""
        try:
            if file_path and file_path.exists():
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write('\n\n' + '=' * 60 + '\n')
                    f.write('End of Final Presentation\n')
        except Exception:
            pass

    def _show_orchestrator_final_presentation(self, selected_agent: str, vote_results: Dict[str, Any]=None) -> None:
        """Show the final presentation from the orchestrator for the selected agent."""
        import time
        try:
            if not hasattr(self, 'orchestrator') or not self.orchestrator:
                return
            if hasattr(self.orchestrator, 'get_final_presentation'):
                import asyncio

                async def _get_and_display_presentation() -> None:
                    """Helper to get and display presentation asynchronously."""
                    try:
                        presentation_stream = self.orchestrator.get_final_presentation(selected_agent, vote_results)
                        await self.display_final_presentation(selected_agent, presentation_stream, vote_results)
                    except Exception:
                        raise
                import nest_asyncio
                nest_asyncio.apply()
                try:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    loop.run_until_complete(_get_and_display_presentation())
                    time.sleep(0.5)
                except Exception:
                    try:
                        asyncio.run(_get_and_display_presentation())
                        time.sleep(0.5)
                    except Exception:
                        self._display_final_presentation_content(selected_agent, 'Unable to retrieve live presentation.')
            else:
                status = self.orchestrator.get_status()
                if selected_agent in status.get('agent_states', {}):
                    stored_answer = status['agent_states'][selected_agent].get('answer', '')
                    if stored_answer:
                        self._display_final_presentation_content(selected_agent, stored_answer)
                    else:
                        print('DEBUG: No stored answer found')
                else:
                    print(f'DEBUG: Agent {selected_agent} not found in agent_states')
        except Exception as e:
            error_text = Text(f'❌ Error in final presentation: {e}', style=self.colors['error'])
            self.console.print(error_text)

    def _force_display_final_vote_statuses(self) -> None:
        """Force display update to show all agents' final vote statuses."""
        with self._lock:
            for agent_id in self.agent_ids:
                self._pending_updates.add(agent_id)
            self._pending_updates.add('footer')
            self._schedule_async_update(force_update=True)
        import time
        time.sleep(0.3)

    def _flush_all_buffers(self) -> None:
        """Flush all text buffers to ensure no content is lost."""
        for agent_id in self.agent_ids:
            if agent_id in self._text_buffers and self._text_buffers[agent_id]:
                buffer_content = self._text_buffers[agent_id].strip()
                if buffer_content:
                    self.agent_outputs[agent_id].append(buffer_content)
                self._text_buffers[agent_id] = ''

    def cleanup(self) -> None:
        """Clean up display resources."""
        with self._lock:
            self._flush_all_buffers()
            if self.live:
                try:
                    self.live.stop()
                except Exception:
                    pass
                finally:
                    self.live = None
            self._stop_input_thread = True
            if self._input_thread and self._input_thread.is_alive():
                try:
                    self._input_thread.join(timeout=1.0)
                except Exception:
                    pass
            try:
                self._restore_terminal_settings()
            except Exception:
                pass
            self._agent_selector_active = False
            self._final_answer_shown = False
            try:
                signal.signal(signal.SIGWINCH, signal.SIG_DFL)
            except (AttributeError, OSError):
                pass
            if self._key_handler:
                try:
                    self._key_handler.stop()
                except Exception:
                    pass
            self._shutdown_flag = True
            for timer in self._debounce_timers.values():
                timer.cancel()
            self._debounce_timers.clear()
            for timer in self._buffer_timers.values():
                if timer:
                    timer.cancel()
            self._buffer_timers.clear()
            if self._batch_timer:
                self._batch_timer.cancel()
                self._batch_timer = None
            if hasattr(self, '_refresh_executor'):
                self._refresh_executor.shutdown(wait=True)
            if hasattr(self, '_status_update_executor'):
                self._status_update_executor.shutdown(wait=True)
            try:
                for agent_id, file_path in self.agent_files.items():
                    if file_path.exists():
                        with open(file_path, 'a', encoding='utf-8') as f:
                            f.write(f'\n=== SESSION ENDED at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n')
            except Exception:
                pass
        from massgen.logger_config import restore_console_logging
        restore_console_logging()

    def _schedule_priority_update(self, agent_id: str) -> None:
        """Schedule immediate priority update for critical agent status changes."""
        if self._shutdown_flag:
            return

        def priority_update() -> None:
            try:
                self._update_agent_panel_cache(agent_id)
                self._update_display_safe()
            except Exception:
                pass
        self._status_update_executor.submit(priority_update)

    def _categorize_update(self, agent_id: str, content_type: str, content: str) -> None:
        """Categorize update by priority for layered refresh strategy."""
        if content_type in ['status', 'error', 'tool'] or any((keyword in content.lower() for keyword in ['error', 'failed', 'completed', 'voted'])):
            self._critical_updates.add(agent_id)
            self._normal_updates.discard(agent_id)
            self._decorative_updates.discard(agent_id)
        elif content_type in ['thinking', 'presentation']:
            if agent_id not in self._critical_updates:
                self._normal_updates.add(agent_id)
                self._decorative_updates.discard(agent_id)
        elif agent_id not in self._critical_updates and agent_id not in self._normal_updates:
            self._decorative_updates.add(agent_id)

    def _schedule_layered_update(self, agent_id: str, is_critical: bool=False) -> None:
        """Schedule update using layered refresh strategy with intelligent batching."""
        if is_critical:
            self._flush_update_batch()
            self._pending_updates.add(agent_id)
            self._schedule_async_update(force_update=True)
        else:
            perf_tier = self._terminal_performance['performance_tier']
            if perf_tier == 'high':
                self._pending_updates.add(agent_id)
                self._schedule_async_update(force_update=False)
            else:
                self._add_to_update_batch(agent_id)

    def _schedule_delayed_update(self) -> None:
        """Schedule delayed update for non-critical content."""
        delay = self._debounce_delay * 2

        def delayed_update() -> None:
            if self._pending_updates:
                self._schedule_async_update(force_update=False)
        if 'delayed' in self._debounce_timers:
            self._debounce_timers['delayed'].cancel()
        self._debounce_timers['delayed'] = threading.Timer(delay, delayed_update)
        self._debounce_timers['delayed'].start()

    def _add_to_update_batch(self, agent_id: str) -> None:
        """Add update to batch for efficient processing."""
        self._update_batch.add(agent_id)
        if self._batch_timer:
            self._batch_timer.cancel()
        self._batch_timer = threading.Timer(self._batch_timeout, self._process_update_batch)
        self._batch_timer.start()

    def _process_update_batch(self) -> None:
        """Process accumulated batch of updates."""
        if self._update_batch:
            self._pending_updates.update(self._update_batch)
            self._update_batch.clear()
            self._schedule_async_update(force_update=False)

    def _flush_update_batch(self) -> None:
        """Immediately flush any pending batch updates."""
        if self._batch_timer:
            self._batch_timer.cancel()
            self._batch_timer = None
        if self._update_batch:
            self._pending_updates.update(self._update_batch)
            self._update_batch.clear()

    def _schedule_async_update(self, force_update: bool=False):
        """Schedule asynchronous update with debouncing to prevent jitter."""
        current_time = time.time()
        if not force_update and self._should_skip_frame():
            return
        if current_time - self._last_full_refresh > self._full_refresh_interval:
            with self._lock:
                self._pending_updates.add('header')
                self._pending_updates.add('footer')
                self._pending_updates.update(self.agent_ids)
            self._last_full_refresh = current_time
        if force_update:
            self._last_update = current_time
            self._refresh_executor.submit(self._async_update_components)
            return
        if 'main' in self._debounce_timers:
            self._debounce_timers['main'].cancel()

        def debounced_update() -> None:
            current_time = time.time()
            time_since_last_update = current_time - self._last_update
            if time_since_last_update >= self._update_interval:
                self._last_update = current_time
                self._refresh_executor.submit(self._async_update_components)
        self._debounce_timers['main'] = threading.Timer(self._debounce_delay, debounced_update)
        self._debounce_timers['main'].start()

    def _should_skip_frame(self) -> bool:
        """Determine if we should skip this frame update to maintain stability."""
        term_type = self._terminal_performance['type']
        if term_type in ['iterm', 'macos_terminal']:
            if self._dropped_frames > 1:
                return True
            if hasattr(self._refresh_executor, '_work_queue') and self._refresh_executor._work_queue.qsize() > 2:
                return True
        return False

    def _async_update_components(self) -> None:
        """Asynchronously update only the components that have changed."""
        start_time = time.time()
        try:
            updates_to_process = None
            with self._lock:
                if self._pending_updates:
                    updates_to_process = self._pending_updates.copy()
                    self._pending_updates.clear()
            if not updates_to_process:
                return
            futures = []
            for update_id in updates_to_process:
                if update_id == 'header':
                    future = self._refresh_executor.submit(self._update_header_cache)
                    futures.append(future)
                elif update_id == 'footer':
                    future = self._refresh_executor.submit(self._update_footer_cache)
                    futures.append(future)
                elif update_id in self.agent_ids:
                    future = self._refresh_executor.submit(self._update_agent_panel_cache, update_id)
                    futures.append(future)
            for future in futures:
                future.result()
            self._update_display_safe()
        except Exception:
            pass
        finally:
            refresh_time = time.time() - start_time
            self._refresh_times.append(refresh_time)
            self._monitor_performance()

    def _update_header_cache(self) -> None:
        """Update the cached header panel."""
        try:
            self._header_cache = self._create_header()
        except Exception:
            pass

    def _update_footer_cache(self) -> None:
        """Update the cached footer panel."""
        try:
            self._footer_cache = self._create_footer()
        except Exception:
            pass

    def _update_agent_panel_cache(self, agent_id: str):
        """Update the cached panel for a specific agent."""
        try:
            self._agent_panels_cache[agent_id] = self._create_agent_panel(agent_id)
        except Exception:
            pass

    def _update_final_presentation_panel(self) -> None:
        """Update the live display to show the latest final presentation content."""
        try:
            if self.live and self.live.is_started:
                with self._lock:
                    self.live.update(self._create_layout())
        except Exception:
            pass

    def _refresh_display(self) -> None:
        """Override parent's refresh method to use async updates."""
        if self._pending_updates:
            self._schedule_async_update()

    def _is_content_important(self, content: str, content_type: str) -> bool:
        """Determine if content is important enough to trigger a display update."""
        if content_type in self._important_content_types:
            return True
        if any((keyword in content.lower() for keyword in self._status_change_keywords)):
            return True
        if any((keyword in content.lower() for keyword in ['error', 'exception', 'failed', 'timeout'])):
            return True
        return False

    def set_status_jump_enabled(self, enabled: bool):
        """Enable or disable status jumping functionality.

        Args:
            enabled: Whether to enable status jumping
        """
        with self._lock:
            self._status_jump_enabled = enabled

    def set_web_search_truncation(self, enabled: bool, max_lines: int=3):
        """Configure web search content truncation on status changes.

        Args:
            enabled: Whether to enable web search truncation
            max_lines: Maximum web search lines to keep when truncating
        """
        with self._lock:
            self._web_search_truncate_on_status_change = enabled
            self._max_web_search_lines = max_lines

    def set_flush_output(self, enabled: bool, char_delay: float=0.03, word_delay: float=0.08):
        """Configure flush output settings for final answer display.

        Args:
            enabled: Whether to enable flush output effect
            char_delay: Delay between characters in seconds
            word_delay: Extra delay after punctuation in seconds
        """
        with self._lock:
            self._enable_flush_output = enabled
            self._flush_char_delay = char_delay
            self._flush_word_delay = word_delay

