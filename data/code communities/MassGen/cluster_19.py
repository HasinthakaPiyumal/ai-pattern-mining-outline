# Cluster 19

def create_config_from_models(models: List[str], orchestrator_config: Optional[Dict[str, Any]]=None, streaming_config: Optional[Dict[str, Any]]=None, logging_config: Optional[Dict[str, Any]]=None) -> MassConfig:
    """
    Create a MassGen configuration from a list of model names.

    Args:
        models: List of model names (e.g., ["gpt-4o", "gemini-2.5-flash"])
        orchestrator_config: Optional orchestrator configuration overrides
        streaming_config: Optional streaming display configuration overrides
        logging_config: Optional logging configuration overrides

    Returns:
        MassConfig object ready to use
    """
    from .utils import get_agent_type_from_model
    agents = []
    for i, model in enumerate(models):
        agent_type = get_agent_type_from_model(model)
        model_config = ModelConfig(model=model, tools=['live_search', 'code_execution'], max_retries=10, max_rounds=10, temperature=None, inference_timeout=180)
        agent_config = AgentConfig(agent_id=i + 1, agent_type=agent_type, model_config=model_config)
        agents.append(agent_config)
    orchestrator = OrchestratorConfig(**orchestrator_config or {})
    streaming_display = StreamingDisplayConfig(**streaming_config or {})
    logging = LoggingConfig(**logging_config or {})
    config = MassConfig(orchestrator=orchestrator, agents=agents, streaming_display=streaming_display, logging=logging)
    config.validate()
    return config

def get_agent_type_from_model(model: str) -> str:
    """
    Determine the agent type based on the model name.

    Args:
        model: The model name (e.g., "gpt-4", "gemini-pro", "grok-1")

    Returns:
        Agent type string ("openai", "gemini", "grok")
    """
    if not model:
        return 'openai'
    model_lower = model.lower()
    for key, models in MODEL_MAPPINGS.items():
        if model_lower in models:
            return key
    raise ValueError(f'Unknown model: {model}')

def _run_single_agent_simple(question: str, config: MassConfig) -> Dict[str, Any]:
    """
    Simple single-agent processing that bypasses the multi-agent orchestration system.

    Args:
        question: The question to solve
        config: MassConfig object with exactly one agent

    Returns:
        Dict containing the answer and detailed results
    """
    start_time = time.time()
    agent_config = config.agents[0]
    logger.info(f'🤖 Running single agent mode with {agent_config.model_config.model}')
    logger.info(f'   Question: {question}')
    log_manager = MassLogManager(log_dir=config.logging.log_dir, session_id=config.logging.session_id, non_blocking=config.logging.non_blocking)
    try:
        agent = create_agent(agent_type=agent_config.agent_type, agent_id=agent_config.agent_id, orchestrator=None, model_config=agent_config.model_config, stream_callback=None)
        messages = [{'role': 'system', 'content': "You are an expert agent equipped with tools to solve complex tasks. Please provide a comprehensive answer to the user's question."}, {'role': 'user', 'content': question}]
        tools = agent_config.model_config.tools if agent_config.model_config.tools else []
        result = agent.process_message(messages=messages, tools=tools)
        session_duration = time.time() - start_time
        response = {'answer': result.text if result.text else 'No response generated', 'consensus_reached': True, 'representative_agent_id': agent_config.agent_id, 'session_duration': session_duration, 'summary': {'total_agents': 1, 'failed_agents': 0, 'total_votes': 1, 'final_vote_distribution': {agent_config.agent_id: 1}}, 'model_used': agent_config.model_config.model, 'citations': result.citations if hasattr(result, 'citations') else [], 'code': result.code if hasattr(result, 'code') else [], 'single_agent_mode': True}
        if log_manager and (not log_manager.non_blocking):
            try:
                result_file = log_manager.session_dir / 'result.json'
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(response, f, indent=2, ensure_ascii=False, default=str)
                logger.info(f'💾 Single agent result saved to {result_file}')
            except Exception as e:
                logger.warning(f'⚠️ Failed to save result.json: {e}')
        logger.info(f'✅ Single agent completed in {session_duration:.1f}s')
        return response
    except Exception as e:
        session_duration = time.time() - start_time
        logger.error(f'❌ Single agent failed: {e}')
        error_response = {'answer': f'Error in single agent processing: {str(e)}', 'consensus_reached': False, 'representative_agent_id': None, 'session_duration': session_duration, 'summary': {'total_agents': 1, 'failed_agents': 1, 'total_votes': 0, 'final_vote_distribution': {}}, 'model_used': agent_config.model_config.model, 'citations': [], 'code': [], 'single_agent_mode': True, 'error': str(e)}
        if log_manager and (not log_manager.non_blocking):
            try:
                result_file = log_manager.session_dir / 'result.json'
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(error_response, f, indent=2, ensure_ascii=False, default=str)
                logger.info(f'💾 Single agent error result saved to {result_file}')
            except Exception as e:
                logger.warning(f'⚠️ Failed to save result.json: {e}')
        return error_response
    finally:
        if log_manager:
            try:
                log_manager.cleanup()
            except Exception as e:
                logger.warning(f'⚠️ Error cleaning up log manager: {e}')

def create_agent(agent_type: str, agent_id: int, orchestrator=None, model_config: Optional[ModelConfig]=None, **kwargs) -> MassAgent:
    """
    Factory function to create agents of different types.

    Args:
        agent_type: Type of agent ("openai", "gemini", "grok")
        agent_id: Unique identifier for the agent
        orchestrator: Reference to the MassOrchestrator
        model_config: Model configuration
        **kwargs: Additional arguments

    Returns:
        MassAgent instance of the specified type
    """
    agent_classes = {'openai': OpenAIMassAgent, 'gemini': GeminiMassAgent, 'grok': GrokMassAgent}
    if agent_type not in agent_classes:
        raise ValueError(f'Unknown agent type: {agent_type}. Available types: {list(agent_classes.keys())}')
    return agent_classes[agent_type](agent_id=agent_id, orchestrator=orchestrator, model_config=model_config, **kwargs)

def run_mass_with_config(question: str, config: MassConfig) -> Dict[str, Any]:
    """
    Run MassGen system with a complete configuration object.

    Args:
        question: The question to solve
        config: Complete MassConfig object

    Returns:
        Dict containing the answer and detailed results
    """
    config.validate()
    if len(config.agents) == 1:
        logger.info('🔄 Single agent detected - using simple processing mode')
        return _run_single_agent_simple(question, config)
    logger.info('🔄 Multiple agents detected - using multi-agent orchestration')
    task = TaskInput(question=question)
    log_manager = MassLogManager(log_dir=config.logging.log_dir, session_id=config.logging.session_id, non_blocking=config.logging.non_blocking)
    streaming_orchestrator = None
    if config.streaming_display.display_enabled:
        streaming_orchestrator = create_streaming_display(display_enabled=config.streaming_display.display_enabled, max_lines=config.streaming_display.max_lines, save_logs=config.streaming_display.save_logs, stream_callback=config.streaming_display.stream_callback, answers_dir=str(log_manager.answers_dir) if not log_manager.non_blocking else None)
    orchestrator = MassOrchestrator(max_duration=config.orchestrator.max_duration, consensus_threshold=config.orchestrator.consensus_threshold, max_debate_rounds=config.orchestrator.max_debate_rounds, status_check_interval=config.orchestrator.status_check_interval, thread_pool_timeout=config.orchestrator.thread_pool_timeout, streaming_orchestrator=streaming_orchestrator)
    orchestrator.log_manager = log_manager
    for agent_config in config.agents:
        stream_callback = None
        if streaming_orchestrator:

            def create_stream_callback(agent_id):

                def callback(content):
                    streaming_orchestrator.stream_output(agent_id, content)
                return callback
            stream_callback = create_stream_callback(agent_config.agent_id)
        agent = create_agent(agent_type=agent_config.agent_type, agent_id=agent_config.agent_id, orchestrator=orchestrator, model_config=agent_config.model_config, stream_callback=stream_callback)
        orchestrator.register_agent(agent)
    logger.info(f'🚀 Starting MassGen with {len(config.agents)} agents')
    logger.info(f'   Question: {question}')
    logger.info(f'   Models: {[agent.model_config.model for agent in config.agents]}')
    logger.info(f'   Max duration: {config.orchestrator.max_duration}s')
    logger.info(f'   Consensus threshold: {config.orchestrator.consensus_threshold}')
    try:
        result = orchestrator.start_task(task)
        logger.info('✅ MassGen completed successfully')
        return result
    except Exception as e:
        logger.error(f'❌ MassGen failed: {e}')
        raise
    finally:
        orchestrator.cleanup()

def create_streaming_display(display_enabled: bool=True, stream_callback: Optional[Callable]=None, max_lines: int=10, save_logs: bool=True, answers_dir: Optional[str]=None) -> StreamingOrchestrator:
    """Create a streaming orchestrator with display capabilities."""
    return StreamingOrchestrator(display_enabled, stream_callback, max_lines, save_logs, answers_dir)

def create_stream_callback(agent_id):

    def callback(content):
        streaming_orchestrator.stream_output(agent_id, content)
    return callback

class MassSystem:
    """
    Enhanced MassGen system interface with configuration support.
    """

    def __init__(self, config: MassConfig):
        """
        Initialize the MassGen system.

        Args:
            config: MassConfig object with complete configuration.
        """
        self.config = config

    def run(self, question: str) -> Dict[str, Any]:
        """
        Run MassGen system on a question using the configured setup.

        Args:
            question: The question to solve

        Returns:
            Dict containing the answer and detailed results
        """
        return run_mass_with_config(question, self.config)

    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update
        """
        if 'max_duration' in kwargs:
            self.config.orchestrator.max_duration = kwargs['max_duration']
        if 'consensus_threshold' in kwargs:
            self.config.orchestrator.consensus_threshold = kwargs['consensus_threshold']
        if 'max_debate_rounds' in kwargs:
            self.config.orchestrator.max_debate_rounds = kwargs['max_debate_rounds']
        self.config.validate()

def run_mass_agents(question: str, models: List[str], max_duration: int=600, consensus_threshold: float=0.0, streaming_display: bool=True, **kwargs) -> Dict[str, Any]:
    """
    Simple function to run MassGen agents on a question (backward compatibility).

    Args:
        question: The question to solve
        models: List of model names (e.g., ["gpt-4o", "gemini-2.5-flash"])
        max_duration: Maximum duration in seconds
        consensus_threshold: Consensus threshold
        streaming_display: Whether to show real-time progress
        **kwargs: Additional configuration parameters

    Returns:
        Dict containing the answer and detailed results
    """
    config = create_config_from_models(models=models, orchestrator_config={'max_duration': max_duration, 'consensus_threshold': consensus_threshold, **{k: v for k, v in kwargs.items() if k in ['max_debate_rounds', 'status_check_interval']}}, streaming_config={'display_enabled': streaming_display, **{k: v for k, v in kwargs.items() if k in ['max_lines', 'save_logs']}})
    return run_mass_with_config(question, config)

def run_interactive_mode(config):
    """Run MassGen in interactive mode, asking for questions repeatedly."""
    print('\n🤖 MassGen Interactive Mode')
    print('=' * 60)
    print('📋 Current Configuration:')
    print('-' * 30)
    if hasattr(config, 'agents') and config.agents:
        print(f'🤖 Agents ({len(config.agents)}):')
        for i, agent in enumerate(config.agents, 1):
            model_name = getattr(agent.model_config, 'model', 'Unknown') if hasattr(agent, 'model_config') else 'Unknown'
            agent_type = getattr(agent, 'agent_type', 'Unknown')
            tools = getattr(agent.model_config, 'tools', []) if hasattr(agent, 'model_config') else []
            tools_str = ', '.join(tools) if tools else 'None'
            print(f'   {i}. {model_name} ({agent_type})')
            print(f'      Tools: {tools_str}')
    else:
        print('🤖 Single Agent Mode')
    if hasattr(config, 'orchestrator'):
        orch = config.orchestrator
        print('⚙️  Orchestrator:')
        print(f'   • Duration: {getattr(orch, 'max_duration', 'Default')}s')
        print(f'   • Consensus: {getattr(orch, 'consensus_threshold', 'Default')}')
        print(f'   • Max Debate Rounds: {getattr(orch, 'max_debate_rounds', 'Default')}')
    if hasattr(config, 'agents') and config.agents and hasattr(config.agents[0], 'model_config'):
        model_config = config.agents[0].model_config
        print('🔧 Model Config:')
        temp = getattr(model_config, 'temperature', 'Default')
        timeout = getattr(model_config, 'inference_timeout', 'Default')
        max_rounds = getattr(model_config, 'max_rounds', 'Default')
        print(f'   • Temperature: {temp}')
        print(f'   • Timeout: {timeout}s')
        print(f'   • Max Debate Rounds: {max_rounds}')
    if hasattr(config, 'streaming_display'):
        display = config.streaming_display
        display_status = '✅ Enabled' if getattr(display, 'display_enabled', True) else '❌ Disabled'
        logs_status = '✅ Enabled' if getattr(display, 'save_logs', True) else '❌ Disabled'
        print(f'📺 Display: {display_status}')
        print(f'📁 Logs: {logs_status}')
    print('-' * 30)
    print("💬 Type your questions below. Type 'quit', 'exit', or press Ctrl+C to stop.")
    print('=' * 60)
    chat_history = ''
    try:
        while True:
            try:
                question = input('\n👤 User: ').strip()
                chat_history += f'User: {question}\n'
                if question.lower() in ['quit', 'exit', 'q']:
                    print('👋 Goodbye!')
                    break
                if not question:
                    print("Please enter a question or type 'quit' to exit.")
                    continue
                print('\n🔄 Processing your question...')
                result = run_mass_with_config(chat_history, config)
                response = result['answer']
                chat_history += f'Assistant: {response}\n'
                print(f'\n{BRIGHT_CYAN}{'=' * 80}{RESET}')
                print(f'{BOLD}{BRIGHT_WHITE}💬 CONVERSATION EXCHANGE{RESET}')
                print(f'{BRIGHT_CYAN}{'=' * 80}{RESET}')
                print(f'\n{BRIGHT_BLUE}👤 User:{RESET}')
                print(f'    {BRIGHT_WHITE}{question}{RESET}')
                print(f'\n{BRIGHT_GREEN}🤖 Assistant:{RESET}')
                agents = {f'Agent {agent.agent_id}': agent.model_config.model for agent in config.agents}
                if result.get('single_agent_mode', False):
                    print(f'    {BRIGHT_YELLOW}📋 Mode:{RESET} Single Agent')
                    print(f'    {BRIGHT_MAGENTA}🤖 Agents:{RESET} {agents}')
                    print(f'    {BRIGHT_CYAN}🎯 Representative:{RESET} {result['representative_agent_id']}')
                    print(f'    {BRIGHT_GREEN}🔧 Model:{RESET} {result.get('model_used', 'Unknown')}')
                    print(f'    {BRIGHT_BLUE}⏱️  Duration:{RESET} {result['session_duration']:.1f}s')
                    if result.get('citations'):
                        print(f'    {BRIGHT_WHITE}📚 Citations:{RESET} {len(result['citations'])}')
                    if result.get('code'):
                        print(f'    {BRIGHT_WHITE}💻 Code blocks:{RESET} {len(result['code'])}')
                else:
                    print(f'    {BRIGHT_YELLOW}📋 Mode:{RESET} Multi-Agent')
                    print(f'    {BRIGHT_MAGENTA}🤖 Agents:{RESET} {agents}')
                    print(f'    {BRIGHT_CYAN}🎯 Representative:{RESET} {result['representative_agent_id']}')
                    print(f'    {BRIGHT_GREEN}✅ Consensus:{RESET} {result['consensus_reached']}')
                    print(f'    {BRIGHT_BLUE}⏱️  Duration:{RESET} {result['session_duration']:.1f}s')
                    print(f'    {BRIGHT_YELLOW}📊 Vote Distribution:{RESET}')
                    display_vote_distribution(result['summary']['final_vote_distribution'])
                print(f'\n    {BRIGHT_RED}💡 Response:{RESET}')
                for line in response.split('\n'):
                    print(f'        {line}')
                print(f'\n{BRIGHT_CYAN}{'=' * 80}{RESET}')
            except KeyboardInterrupt:
                print('\n👋 Goodbye!')
                break
            except Exception as e:
                print(f'❌ Error processing question: {e}')
                print("Please try again or type 'quit' to exit.")
    except KeyboardInterrupt:
        print('\n👋 Goodbye!')

def display_vote_distribution(vote_distribution):
    """Display the vote distribution in a more readable format."""
    sorted_keys = sorted(vote_distribution.keys())
    for agent_id in sorted_keys:
        print(f'      {BRIGHT_CYAN}Agent {agent_id}{RESET}: {BRIGHT_GREEN}{vote_distribution[agent_id]}{RESET} votes')

def main():
    """Clean CLI interface for MassGen."""
    parser = argparse.ArgumentParser(description='MassGen (Multi-Agent Scaling System) - Clean CLI', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  # Use YAML configuration\n  uv run python -m massgen.v1.cli "What is the capital of France?" --config examples/production.yaml\n\n  # Use model names directly (single or multiple agents)\n  uv run python -m massgen.v1.cli "What is 2+2?" --models gpt-4o gemini-2.5-flash\n  uv run python -m massgen.v1.cli "What is 2+2?" --models gpt-4o  # Single agent mode\n\n  # Interactive mode (no question provided)\n  uv run python -m massgen.v1.cli --models gpt-4o grok-4\n\n  # Override parameters\n  uv run python -m massgen.v1.cli "Question" --models gpt-4o gemini-2.5-flash --max-duration 1200 --consensus 0.8\n        ')
    parser.add_argument('question', nargs='?', help='Question to solve (optional - if not provided, enters interactive mode)')
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument('--config', type=str, help='Path to YAML configuration file')
    config_group.add_argument('--models', nargs='+', help='Model names (e.g., gpt-4o gemini-2.5-flash)')
    parser.add_argument('--max-duration', type=int, default=None, help='Max duration in seconds')
    parser.add_argument('--consensus', type=float, default=None, help='Consensus threshold (0.0-1.0)')
    parser.add_argument('--max-debates', type=int, default=None, help='Maximum debate rounds')
    parser.add_argument('--no-display', action='store_true', help='Disable streaming display')
    parser.add_argument('--no-logs', action='store_true', help='Disable file logging')
    args = parser.parse_args()
    try:
        if args.config:
            config = load_config_from_yaml(args.config)
        else:
            config = create_config_from_models(args.models)
        if args.max_duration is not None:
            config.orchestrator.max_duration = args.max_duration
        if args.consensus is not None:
            config.orchestrator.consensus_threshold = args.consensus
        if args.max_debates is not None:
            config.orchestrator.max_debate_rounds = args.max_debates
        if args.no_display:
            config.streaming_display.display_enabled = False
        if args.no_logs:
            config.streaming_display.save_logs = False
        config.validate()
        agents = {f'Agent {agent.agent_id}': agent.model_config.model for agent in config.agents}
        if args.question:
            result = run_mass_with_config(args.question, config)
            print('\n' + '=' * 60)
            print(f'🎯 FINAL ANSWER (Agent {result['representative_agent_id']}):')
            print('=' * 60)
            print(result['answer'])
            print('\n' + '=' * 60)
            if result.get('single_agent_mode', False):
                print('🤖 Single Agent Mode')
                print(f'🤖 Agents: {agents}')
                print(f'⏱️  Duration: {result['session_duration']:.1f}s')
                if result.get('citations'):
                    print(f'📚 Citations: {len(result['citations'])}')
                if result.get('code'):
                    print(f'💻 Code blocks: {len(result['code'])}')
            else:
                print(f'🤖 Agents: {agents}')
                print(f'🎯 Representative Agent: {result['representative_agent_id']}')
                print(f'✅ Consensus: {result['consensus_reached']}')
                print(f'⏱️  Duration: {result['session_duration']:.1f}s')
                print('📊 Votes:')
                display_vote_distribution(result['summary']['final_vote_distribution'])
        else:
            run_interactive_mode(config)
    except ConfigurationError as e:
        print(f'❌ Configuration error: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'❌ Error: {e}')
        sys.exit(1)

def load_config_from_yaml(config_path: Union[str, Path]) -> MassConfig:
    """
    Load MassGen configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        MassConfig object with loaded configuration

    Raises:
        ConfigurationError: If configuration is invalid or file cannot be loaded
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigurationError(f'Configuration file not found: {config_path}')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f'Invalid YAML format: {e}')
    except Exception as e:
        raise ConfigurationError(f'Failed to read configuration file: {e}')
    if not yaml_data:
        raise ConfigurationError('Empty configuration file')
    return _dict_to_config(yaml_data)

class MassAgent(ABC):
    """
    Abstract base class for all agents in the MassGen system.

    All agent implementations must inherit from this class and implement
    the required methods while following the standardized workflow.
    """

    def __init__(self, agent_id: int, orchestrator=None, model_config: Optional[ModelConfig]=None, stream_callback: Optional[Callable]=None, **kwargs):
        """
        Initialize the agent with configuration parameters.

        Args:
            agent_id: Unique identifier for this agent
            orchestrator: Reference to the MassOrchestrator
            model_config: Configuration object containing model parameters (model, tools,
                         temperature, top_p, max_tokens, inference_timeout, max_retries, stream)
            stream_callback: Optional callback function for streaming chunks
            agent_type: Type of agent ("openai", "gemini", "grok") to determine backend
            **kwargs: Additional parameters specific to the agent implementation
        """
        self.agent_id = agent_id
        self.orchestrator = orchestrator
        self.state = AgentState(agent_id=agent_id)
        if model_config is None:
            model_config = ModelConfig()
        self.model = model_config.model
        self.agent_type = get_agent_type_from_model(self.model)
        process_message_impl_map = {'openai': oai.process_message, 'gemini': gemini.process_message, 'grok': grok.process_message}
        if self.agent_type not in process_message_impl_map:
            raise ValueError(f'Unknown agent type: {self.agent_type}. Available types: {list(process_message_impl_map.keys())}')
        self.process_message_impl = process_message_impl_map[self.agent_type]
        self.tools = model_config.tools
        self.max_retries = model_config.max_retries
        self.max_rounds = model_config.max_rounds
        self.max_tokens = model_config.max_tokens
        self.temperature = model_config.temperature
        self.top_p = model_config.top_p
        self.inference_timeout = model_config.inference_timeout
        self.stream = model_config.stream
        self.stream_callback = stream_callback
        self.kwargs = kwargs

    def process_message(self, messages: List[Dict[str, str]], tools: List[str]=None) -> AgentResponse:
        """
        Core LLM inference function for task processing.

        This method handles the actual LLM interaction using the agent's
        specific backend (OpenAI, Gemini, Grok, etc.) and returns a standardized response.
        All configuration parameters are stored as instance variables and accessed
        via self.model, self.tools, self.temperature, etc.

        Args:
            messages: List of messages in OpenAI format
            tools: List of tools to use

        Returns:
            AgentResponse containing the agent's response text, code, citations, etc.
        """
        config = {'model': self.model, 'max_retries': self.max_retries, 'max_tokens': self.max_tokens, 'temperature': self.temperature, 'top_p': self.top_p, 'api_key': None, 'stream': self.stream, 'stream_callback': self.stream_callback}
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.process_message_impl, messages=messages, tools=tools, **config)
                try:
                    result = future.result(timeout=self.inference_timeout)
                    return result
                except FutureTimeoutError:
                    timeout_msg = f'Agent {self.agent_id} timed out after {self.inference_timeout} seconds'
                    self.mark_failed(timeout_msg)
                    return AgentResponse(text=f'Agent processing timed out after {self.inference_timeout} seconds', code=[], citations=[], function_calls=[])
        except Exception as e:
            return AgentResponse(text=f'Error in {self.model} agent processing: {str(e)}', code=[], citations=[], function_calls=[])

    def add_answer(self, new_answer: str):
        """
        Record your work on the task: your analysis, approach, solution, and reasoning. Update when you solve the problem, find better solutions, or incorporate valuable insights from other agents.

        Args:
            answer: The new answer, which should be self-contained, complete, and ready to serve as the definitive final response.
        """
        self.orchestrator.notify_answer_update(self.agent_id, new_answer)
        return 'The new answer has been added.'

    def vote(self, agent_id: int, reason: str='', invalid_vote_options: List[int]=[]):
        """
        Vote for the representative agent, who you believe has found the correct solution.

        Args:
            agent_id: ID of the voted agent
            reason: Your full explanation of why you voted for this agent
            invalid_vote_options: The list of agent IDs that are invalid to vote for (have new updates)
        """
        if agent_id in invalid_vote_options:
            return f'Error: Voting for agent {agent_id} is not allowed as its answer has been updated!'
        self.orchestrator.cast_vote(self.agent_id, agent_id, reason)
        return f'Your vote for Agent {agent_id} has been cast.'

    def check_update(self) -> List[int]:
        """
        Check if there are any updates from other agents since this agent last saw them.
        """
        agents_with_update = set()
        for other_id, other_state in self.orchestrator.agent_states.items():
            if other_id != self.agent_id and other_state.updated_answers:
                for update in other_state.updated_answers:
                    last_seen = self.state.seen_updates_timestamps.get(other_id, 0)
                    if update.timestamp > last_seen:
                        self.state.seen_updates_timestamps[other_id] = update.timestamp
                        agents_with_update.add(other_id)
        return list(agents_with_update)

    def mark_failed(self, reason: str=''):
        """
        Mark this agent as failed.

        Args:
            reason: Optional reason for the failure
        """
        self.orchestrator.mark_agent_failed(self.agent_id, reason)

    def deduplicate_function_calls(self, function_calls: List[Dict]):
        """Deduplicate function calls by their name and arguments."""
        deduplicated_function_calls = []
        for func_call in function_calls:
            if func_call not in deduplicated_function_calls:
                deduplicated_function_calls.append(func_call)
        return deduplicated_function_calls

    def _execute_function_calls(self, function_calls: List[Dict], invalid_vote_options: List[int]=[]):
        """Execute function calls and return function outputs."""
        from .tools import register_tool
        function_outputs = []
        successful_called = []
        for func_call in function_calls:
            func_call_id = func_call.get('call_id')
            func_name = func_call.get('name')
            func_args = func_call.get('arguments', {})
            if isinstance(func_args, str):
                func_args = json.loads(func_args)
            try:
                if func_name == 'add_answer':
                    result = self.add_answer(func_args.get('new_answer', ''))
                elif func_name == 'vote':
                    result = self.vote(func_args.get('agent_id'), func_args.get('reason', ''), invalid_vote_options)
                elif func_name in register_tool:
                    result = register_tool[func_name](**func_args)
                else:
                    result = {'type': 'function_call_output', 'call_id': func_call_id, 'output': f"Error: Function '{func_name}' not found in tool mapping"}
                function_output = {'type': 'function_call_output', 'call_id': func_call_id, 'output': str(result)}
                function_outputs.append(function_output)
                successful_called.append(True)
            except Exception as e:
                error_output = {'type': 'function_call_output', 'call_id': func_call_id, 'output': f'Error executing function: {str(e)}'}
                function_outputs.append(error_output)
                successful_called.append(False)
                print(f'Error executing function {func_name}: {e}')
                with open('function_calls.txt', 'a') as f:
                    f.write(f'[{time.strftime('%Y-%m-%d %H:%M:%S')}] Agent {self.agent_id} ({self.model}):\n')
                    f.write(f'{json.dumps(error_output, indent=2)}\n')
                    f.write(f'Successful called: {False}\n')
        return (function_outputs, successful_called)

    def _get_system_tools(self) -> List[Dict[str, Any]]:
        """
        The system tools available to this agent for orchestration:
        - add_answer: Your added new answer, which should be self-contained, complete, and ready to serve as the definitive final response.
        - vote: Vote for the representative agent, who you believe has found the correct solution.
        """
        add_answer_schema = {'type': 'function', 'name': 'add_answer', 'description': 'Add your new answer if you believe it is better than the current answers.', 'parameters': {'type': 'object', 'properties': {'new_answer': {'type': 'string', 'description': 'Your new answer, which should be self-contained, complete, and ready to serve as the definitive final response.'}}, 'required': ['new_answer']}}
        vote_schema = {'type': 'function', 'name': 'vote', 'description': 'Vote for the best agent to present final answer. Submit its agent_id (integer) and reason for your vote.', 'parameters': {'type': 'object', 'properties': {'agent_id': {'type': 'integer', 'description': 'The ID of the agent you believe has found the best answer that addresses the original message.'}, 'reason': {'type': 'string', 'description': 'Your full explanation of why you voted for this agent.'}}, 'required': ['agent_id', 'reason']}}
        available_options = [agent_id for agent_id, agent_state in self.orchestrator.agent_states.items() if agent_state.curr_answer]
        return [add_answer_schema, vote_schema] if available_options else [add_answer_schema]

    def _get_registered_tools(self) -> List[Dict[str, Any]]:
        """Return the tool schema for the tools that are available to this agent."""
        custom_tools = []
        from .tools import register_tool
        for tool_name, tool_func in register_tool.items():
            if tool_name in self.tools:
                tool_schema = function_to_json(tool_func)
                custom_tools.append(tool_schema)
        return custom_tools

    def _get_builtin_tools(self) -> List[Dict[str, Any]]:
        """
        Override the parent method due to the Gemini's limitation.
        Return the built-in tools that are available to Gemini models.
        live_search and code_execution are supported right now.
        However, the built-in tools and function call are not supported at the same time.
        """
        builtin_tools = []
        for tool in self.tools:
            if tool in ['live_search', 'code_execution']:
                builtin_tools.append(tool)
        return builtin_tools

    def _get_all_answers(self) -> List[str]:
        """Get all answers from all agents.
        Format:
        **Agent 1**: Answer 1
        **Agent 2**: Answer 2
        ...
        """
        agent_answers = []
        for agent_id, agent_state in self.orchestrator.agent_states.items():
            if agent_state.curr_answer:
                agent_answers.append(f'**Agent {agent_id}**: {agent_state.curr_answer}')
        return agent_answers

    def _get_all_votes(self) -> List[str]:
        """Get all votes from all agents.
        Format:
        **Vote for Agent 1**: Reason 1
        **Vote for Agent 2**: Reason 2
        ...
        """
        agent_votes = []
        for agent_id, agent_state in self.orchestrator.agent_states.items():
            if agent_state.curr_vote:
                agent_votes.append(f'**Vote for Agent {agent_state.curr_vote.target_id}**: {agent_state.curr_vote.reason}')
        return agent_votes

    def _get_task_input(self, task: TaskInput) -> str:
        """Get the initial task input as the user message. Return Both the current status and the task input."""
        if not self.state.curr_answer:
            status = 'initial'
            task_input = AGENT_ANSWER_MESSAGE.format(task=task.question, agent_answers='None') + 'There are no current answers right now. Please use your expertise and tools (if available) to provide a new answer and submit it using the `add_answer` tool first.'
            return (status, task_input)
        all_agent_answers = self._get_all_answers()
        all_agent_answers_str = '\n\n'.join(all_agent_answers)
        voted_agents = [agent_id for agent_id, agent_state in self.orchestrator.agent_states.items() if agent_state.curr_vote is not None]
        if len(voted_agents) == len(self.orchestrator.agent_states):
            all_agent_votes = self._get_all_votes()
            all_agent_votes_str = '\n\n'.join(all_agent_votes)
            status = 'debate'
            task_input = AGENT_ANSWER_AND_VOTE_MESSAGE.format(task=task.question, agent_answers=all_agent_answers_str, agent_votes=all_agent_votes_str)
        else:
            status = 'working'
            task_input = AGENT_ANSWER_MESSAGE.format(task=task.question, agent_answers=all_agent_answers_str)
        return (status, task_input)

    def _get_task_input_messages(self, user_input: str) -> List[Dict[str, str]]:
        """Get the task input messages for the agent."""
        return [{'role': 'system', 'content': SYSTEM_INSTRUCTION}, {'role': 'user', 'content': user_input}]

    def _get_curr_messages_and_tools(self, task: TaskInput):
        """Get the current messages and tools for the agent."""
        working_status, user_input = self._get_task_input(task)
        working_messages = self._get_task_input_messages(user_input)
        all_tools = []
        all_tools.extend(self._get_builtin_tools())
        all_tools.extend(self._get_registered_tools())
        all_tools.extend(self._get_system_tools())
        return (working_status, working_messages, all_tools)

    def work_on_task(self, task: TaskInput) -> List[Dict[str, str]]:
        """
        Work on the task with conversation continuation.

        Args:
            task: The task to work on
            messages: Current conversation history
            restart_instruction: Optional instruction for restarting work (e.g., updates from other agents)

        Returns:
            Updated conversation history including agent's work

        This method should be implemented by concrete agent classes.
        The agent continues the conversation until it votes or reaches max rounds.
        """
        curr_round = 0
        working_status, working_messages, all_tools = self._get_curr_messages_and_tools(task)
        while curr_round < self.max_rounds and self.state.status == 'working':
            try:
                result = self.process_message(messages=working_messages, tools=all_tools)
                agents_with_update = self.check_update()
                has_update = len(agents_with_update) > 0
                if result.text:
                    working_messages.append({'role': 'assistant', 'content': result.text})
                if result.function_calls:
                    result.function_calls = self.deduplicate_function_calls(result.function_calls)
                    function_outputs, successful_called = self._execute_function_calls(result.function_calls, invalid_vote_options=agents_with_update)
                    renew_conversation = False
                    for function_call, function_output, successful_called in zip(result.function_calls, function_outputs, successful_called):
                        if function_call.get('name') == 'add_answer' and successful_called:
                            renew_conversation = True
                            break
                        if function_call.get('name') == 'vote' and successful_called:
                            renew_conversation = True
                            break
                    if not renew_conversation:
                        for function_call, function_output in zip(result.function_calls, function_outputs):
                            working_messages.extend([function_call, function_output])
                    else:
                        working_status, working_messages, all_tools = self._get_curr_messages_and_tools(task)
                elif self.state.status == 'voted':
                    break
                elif has_update and working_status != 'initial':
                    working_status, working_messages, all_tools = self._get_curr_messages_and_tools(task)
                else:
                    working_messages.append({'role': 'user', 'content': 'Finish your work above by making a tool call of `vote` or `add_answer`. Make sure you actually call the tool.'})
                curr_round += 1
                self.state.chat_round += 1
                if self.state.status in ['voted', 'failed']:
                    break
            except Exception as e:
                print(f'❌ Agent {self.agent_id} error in round {self.state.chat_round}: {e}')
                if self.orchestrator:
                    self.orchestrator.mark_agent_failed(self.agent_id, str(e))
                self.state.chat_round += 1
                curr_round += 1
                break
        return working_messages

