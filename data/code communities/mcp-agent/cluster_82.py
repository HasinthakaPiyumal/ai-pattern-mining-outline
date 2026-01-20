# Cluster 82

def get_rcm_logger(name: str):
    """Get logger with RCM-specific formatting"""
    logger = get_logger(f'rcm.{name}')
    return logger

def report_step(message: str, details: str=''):
    """Report a step using the global reporter"""
    reporter = get_progress_reporter()
    if reporter:
        reporter.step(message, details)

def get_progress_reporter() -> Optional[ProgressReporter]:
    """Get the current progress reporter"""
    return _global_reporter

def report_thinking(message: str='Processing'):
    """Report thinking using the global reporter"""
    reporter = get_progress_reporter()
    if reporter:
        reporter.thinking(message)

def report_quality_check(score: float, issues: int=0):
    """Report quality check using the global reporter"""
    reporter = get_progress_reporter()
    if reporter:
        reporter.quality_check(score, issues)

def report_requirement_extraction(count: int):
    """Report requirement extraction using the global reporter"""
    reporter = get_progress_reporter()
    if reporter:
        reporter.requirement_extraction(count)

def report_context_consolidation(from_chars: int, to_chars: int):
    """Report context consolidation using the global reporter"""
    reporter = get_progress_reporter()
    if reporter:
        reporter.context_consolidation(from_chars, to_chars)

def show_llm_interaction(role: str, prompt: str, response: str, truncate_at: int=500):
    """Show LLM interaction using the global reporter"""
    reporter = get_progress_reporter()
    if reporter:
        reporter.show_llm_interaction(role, prompt, response, truncate_at)

class ConversationWorkflow(Workflow[Dict[str, Any]]):
    """
    Core conversation workflow implementing paper findings.
    Supports both AsyncIO and Temporal execution modes.
    """

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.state: Optional[ConversationState] = None
        self.config: Optional[ConversationConfig] = None
        self.logger = get_rcm_logger('conversation_workflow')

    async def run(self, args: Dict[str, Any]) -> WorkflowResult[Dict[str, Any]]:
        """Main conversation loop - handles both execution modes"""
        rcm_config = extract_rcm_config(self.app.context.config)
        self.config = ConversationConfig.from_dict(rcm_config)
        execution_engine = self.app.context.config.execution_engine
        if execution_engine == 'temporal':
            return await self._run_temporal_conversation(args)
        else:
            return await self._run_asyncio_conversation(args)

    async def _run_asyncio_conversation(self, args: Dict[str, Any]) -> WorkflowResult[Dict[str, Any]]:
        """AsyncIO mode - single turn processing for REPL"""
        if 'state' in args and args['state']:
            self.state = ConversationState.from_dict(args['state'])
            log_conversation_event(self.logger, 'state_restored', self.state.conversation_id, {'turn': self.state.current_turn})
        else:
            conversation_id = args.get('conversation_id', f'rcm_{int(time.time())}_{str(uuid.uuid4())[:8]}')
            self.state = ConversationState(conversation_id=conversation_id, is_temporal_mode=False)
            await self._add_system_message()
            log_conversation_event(self.logger, 'conversation_started', self.state.conversation_id)
        user_input = args['user_input']
        await self._process_turn(user_input)
        response_data = {'response': self.state.messages[-1].content if self.state.messages else '', 'state': self.state.to_dict(), 'metrics': self.state.quality_history[-1].to_dict() if self.state.quality_history else {}, 'turn_number': self.state.current_turn}
        log_conversation_event(self.logger, 'turn_completed', self.state.conversation_id, {'turn': self.state.current_turn, 'response_length': len(response_data['response'])})
        return WorkflowResult(value=response_data)

    async def _run_temporal_conversation(self, args: Dict[str, Any]) -> WorkflowResult[Dict[str, Any]]:
        """Temporal mode - full conversation lifecycle (to be implemented in Phase 6)"""
        raise NotImplementedError('Temporal mode will be implemented in Phase 6')

    async def _add_system_message(self):
        """Add initial system message to conversation"""
        system_message = ConversationMessage(role='system', content='You are a helpful AI assistant engaged in a multi-turn conversation. Maintain context across turns and provide thoughtful, accurate responses.', turn_number=0)
        self.state.messages.append(system_message)
        log_workflow_step(self.logger, self.state.conversation_id, 'system_message_added')

    async def _process_turn(self, user_input: str):
        """
        Process single conversation turn with quality control pipeline.
        Implements paper's quality refinement methodology from Phase 2.
        """
        log_workflow_step(self.logger, self.state.conversation_id, 'turn_processing_started', {'turn': self.state.current_turn + 1})
        self.state.current_turn += 1
        user_message = ConversationMessage(role='user', content=user_input, turn_number=self.state.current_turn)
        self.state.messages.append(user_message)
        try:
            from tasks.task_functions import process_turn_with_quality
            result = await process_turn_with_quality({'state': self.state.to_dict(), 'config': self.config.to_dict()})
            response = result['response']
            self.state.requirements = [Requirement.from_dict(req_dict) for req_dict in result['requirements']]
            self.state.consolidated_context = result['consolidated_context']
            metrics = QualityMetrics.from_dict(result['metrics'])
            self.state.quality_history.append(metrics)
            if result.get('context_consolidated'):
                self.state.consolidation_turns.append(self.state.current_turn)
            log_workflow_step(self.logger, self.state.conversation_id, 'quality_controlled_processing_completed', {'response_length': len(response), 'quality_score': metrics.overall_score, 'refinement_attempts': result.get('refinement_attempts', 1), 'requirements_tracked': len(self.state.requirements)})
        except Exception as e:
            log_workflow_step(self.logger, self.state.conversation_id, 'quality_control_fallback', {'error': str(e)})
            response = await self._generate_basic_response(user_input)
            basic_metrics = QualityMetrics(clarity=0.7, completeness=0.7, assumptions=0.3, verbosity=0.3, premature_attempt=False, middle_turn_reference=0.5, requirement_tracking=0.5)
            self.state.quality_history.append(basic_metrics)
        assistant_message = ConversationMessage(role='assistant', content=response, turn_number=self.state.current_turn)
        self.state.messages.append(assistant_message)
        self.state.answer_lengths.append(len(response))
        if self.state.first_answer_attempt_turn is None and len(response) > 100:
            self.state.first_answer_attempt_turn = self.state.current_turn
        log_workflow_step(self.logger, self.state.conversation_id, 'turn_processing_completed', {'response_length': len(response)})

    async def _generate_basic_response(self, user_input: str) -> str:
        """
        Generate basic response using LLM.
        This will be enhanced with quality control in Phase 2.
        """
        log_workflow_step(self.logger, self.state.conversation_id, 'response_generation_started')
        try:
            response_agent = Agent(name='basic_responder', instruction='You are a helpful assistant. Provide clear, accurate responses based on the conversation context.', server_names=self.config.mcp_servers)
            async with response_agent:
                llm_class = get_llm_class(self.config.evaluator_model_provider)
                llm = await response_agent.attach_llm(llm_class)
                conversation_context = self._build_conversation_context()
                full_prompt = f'{conversation_context}\n\nUser: {user_input}\n\nAssistant:'
                response = await llm.generate_str(full_prompt)
                log_workflow_step(self.logger, self.state.conversation_id, 'response_generation_completed', {'response_length': len(response)})
                return response
        except Exception as e:
            log_workflow_step(self.logger, self.state.conversation_id, 'response_generation_fallback', {'error': str(e)})
            mock_response = f"Thank you for your message: '{user_input}'. This is a mock response for testing purposes."
            log_workflow_step(self.logger, self.state.conversation_id, 'response_generation_completed', {'response_length': len(mock_response), 'mode': 'mock'})
            return mock_response

    def _build_conversation_context(self) -> str:
        """Build context string from conversation history"""
        context_parts = []
        recent_messages = self.state.messages[-5:] if len(self.state.messages) > 5 else self.state.messages
        for msg in recent_messages:
            if msg.role != 'system':
                role_label = 'User' if msg.role == 'user' else 'Assistant'
                context_parts.append(f'{role_label}: {msg.content}')
        return '\n'.join(context_parts) if context_parts else 'This is the start of our conversation.'

def extract_rcm_config(app_config: Any) -> dict:
    """Extract RCM-specific configuration from app config"""
    rcm_config = {}
    if hasattr(app_config, 'rcm'):
        rcm_config.update(app_config.rcm)
    rcm_config.setdefault('quality_threshold', 0.8)
    rcm_config.setdefault('max_refinement_attempts', 3)
    rcm_config.setdefault('consolidation_interval', 3)
    rcm_config.setdefault('use_claude_code', False)
    rcm_config.setdefault('evaluator_model_provider', 'openai')
    rcm_config.setdefault('verbose_metrics', False)
    rcm_config.setdefault('mcp_servers', [])
    return rcm_config

def log_conversation_event(logger, event_type: str, conversation_id: str, data: Optional[Dict[str, Any]]=None):
    """Log conversation-specific events with consistent formatting"""
    log_data = {'event_type': event_type, 'conversation_id': conversation_id, **(data or {})}
    logger.info(f'Conversation event: {event_type}', data=log_data)

def log_workflow_step(logger, conversation_id: str, step: str, details: Optional[Dict[str, Any]]=None):
    """Log workflow execution steps for debugging"""
    log_data = {'conversation_id': conversation_id, 'workflow_step': step, **(details or {})}
    logger.debug(f'Workflow step: {step}', data=log_data)

def get_llm_class(provider: str='openai') -> Type:
    """Get LLM class based on provider name"""
    if provider.lower() == 'anthropic':
        return AnthropicAugmentedLLM
    else:
        return OpenAIAugmentedLLM

def _calculate_overall_score(metrics: Dict[str, Any]) -> float:
    """Calculate overall quality score from paper's formula"""
    clarity = metrics.get('clarity', 0.5)
    completeness = metrics.get('completeness', 0.5)
    assumptions = metrics.get('assumptions', 0.5)
    verbosity = metrics.get('verbosity', 0.5)
    middle_turn_reference = metrics.get('middle_turn_reference', 0.5)
    requirement_tracking = metrics.get('requirement_tracking', 0.5)
    premature_attempt = metrics.get('premature_attempt', False)
    base = (clarity + completeness + middle_turn_reference + requirement_tracking + (1 - assumptions) + (1 - verbosity)) / 6
    if premature_attempt:
        base *= 0.5
    return base

def _detect_complete_solution_attempt(response: str) -> bool:
    """Detect if response contains markers of complete solution attempts"""
    solution_markers = ["here's the complete", 'here is the full', 'final solution', 'complete implementation', 'this should handle everything', 'final answer', 'complete response', "here's everything you need"]
    response_lower = response.lower()
    return any((marker in response_lower for marker in solution_markers))

def _should_consolidate_context(state: ConversationState, config: Dict[str, Any]) -> bool:
    """Determine if context consolidation is needed based on paper findings"""
    consolidation_interval = config.get('consolidation_interval', 3)
    return state.current_turn % consolidation_interval == 0 or len(state.consolidated_context) > 2000 or state.current_turn == 1

