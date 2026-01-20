# Cluster 7

def get_backend_type_from_model(model: str) -> str:
    """
    Determine the agent type based on the model name.

    Args:
        model: The model name (e.g., "gpt-4", "gemini-pro", "grok-1")

    Returns:
        Agent type string ("openai", "gemini", "grok", etc.)
    """
    if not model:
        return 'openai'
    model_lower = model.lower()
    for key, models in MODEL_MAPPINGS.items():
        if model_lower in models:
            return key
    raise ValueError(f'Unknown model: {model}')

def _substitute_variables(obj: Any, variables: Dict[str, str]) -> Any:
    """Recursively substitute ${var} references in config with actual values.

    Args:
        obj: Config object (dict, list, str, or other)
        variables: Dict of variable names to values

    Returns:
        Config object with variables substituted
    """
    if isinstance(obj, dict):
        return {k: _substitute_variables(v, variables) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_variables(item, variables) for item in obj]
    elif isinstance(obj, str):
        result = obj
        for var_name, var_value in variables.items():
            result = result.replace(f'${{{var_name}}}', var_value)
        return result
    else:
        return obj

def create_backend(backend_type: str, **kwargs) -> Any:
    """Create backend instance from type and parameters.

    Supported backend types:
    - openai: OpenAI API (requires OPENAI_API_KEY)
    - grok: xAI Grok (requires XAI_API_KEY)
    - sglang: SGLang inference server (local)
    - claude: Anthropic Claude (requires ANTHROPIC_API_KEY)
    - gemini: Google Gemini (requires GOOGLE_API_KEY or GEMINI_API_KEY)
    - chatcompletion: OpenAI-compatible providers (auto-detects API key based on base_url)

    Supported backend with external dependencies:
    - ag2/autogen: AG2 (AutoGen) framework agents

    For chatcompletion backend, the following providers are auto-detected:
    - Cerebras AI (cerebras.ai) -> CEREBRAS_API_KEY
    - Together AI (together.ai/together.xyz) -> TOGETHER_API_KEY
    - Fireworks AI (fireworks.ai) -> FIREWORKS_API_KEY
    - Groq (groq.com) -> GROQ_API_KEY
    - Nebius AI Studio (studio.nebius.ai) -> NEBIUS_API_KEY
    - OpenRouter (openrouter.ai) -> OPENROUTER_API_KEY
    - POE (poe.com) -> POE_API_KEY
    - Qwen (dashscope.aliyuncs.com) -> QWEN_API_KEY

    External agent frameworks are supported via the adapter registry.
    """
    backend_type = backend_type.lower()
    from massgen.adapters import adapter_registry
    if backend_type in adapter_registry:
        from massgen.backend.external import ExternalAgentBackend
        return ExternalAgentBackend(adapter_type=backend_type, **kwargs)
    if backend_type == 'openai':
        api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ConfigurationError('OpenAI API key not found. Set OPENAI_API_KEY environment variable.\nYou can add it to a .env file in:\n  - Current directory: .env\n  - Global config: ~/.massgen/.env')
        return ResponseBackend(api_key=api_key, **kwargs)
    elif backend_type == 'grok':
        api_key = kwargs.get('api_key') or os.getenv('XAI_API_KEY')
        if not api_key:
            raise ConfigurationError('Grok API key not found. Set XAI_API_KEY environment variable.\nYou can add it to a .env file in:\n  - Current directory: .env\n  - Global config: ~/.massgen/.env')
        return GrokBackend(api_key=api_key, **kwargs)
    elif backend_type == 'claude':
        api_key = kwargs.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ConfigurationError('Claude API key not found. Set ANTHROPIC_API_KEY environment variable.\nYou can add it to a .env file in:\n  - Current directory: .env\n  - Global config: ~/.massgen/.env')
        return ClaudeBackend(api_key=api_key, **kwargs)
    elif backend_type == 'gemini':
        api_key = kwargs.get('api_key') or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ConfigurationError('Gemini API key not found. Set GOOGLE_API_KEY environment variable.\nYou can add it to a .env file in:\n  - Current directory: .env\n  - Global config: ~/.massgen/.env')
        return GeminiBackend(api_key=api_key, **kwargs)
    elif backend_type == 'chatcompletion':
        api_key = kwargs.get('api_key')
        base_url = kwargs.get('base_url')
        if not api_key:
            if base_url and 'cerebras.ai' in base_url:
                api_key = os.getenv('CEREBRAS_API_KEY')
                if not api_key:
                    raise ConfigurationError('Cerebras AI API key not found. Set CEREBRAS_API_KEY environment variable.\nYou can add it to a .env file in:\n  - Current directory: .env\n  - Global config: ~/.massgen/.env')
            elif base_url and 'together.xyz' in base_url:
                api_key = os.getenv('TOGETHER_API_KEY')
                if not api_key:
                    raise ConfigurationError('Together AI API key not found. Set TOGETHER_API_KEY environment variable.\nYou can add it to a .env file in:\n  - Current directory: .env\n  - Global config: ~/.massgen/.env')
            elif base_url and 'fireworks.ai' in base_url:
                api_key = os.getenv('FIREWORKS_API_KEY')
                if not api_key:
                    raise ConfigurationError('Fireworks AI API key not found. Set FIREWORKS_API_KEY environment variable.\nYou can add it to a .env file in:\n  - Current directory: .env\n  - Global config: ~/.massgen/.env')
            elif base_url and 'groq.com' in base_url:
                api_key = os.getenv('GROQ_API_KEY')
                if not api_key:
                    raise ConfigurationError('Groq API key not found. Set GROQ_API_KEY environment variable.\nYou can add it to a .env file in:\n  - Current directory: .env\n  - Global config: ~/.massgen/.env')
            elif base_url and 'nebius.com' in base_url:
                api_key = os.getenv('NEBIUS_API_KEY')
                if not api_key:
                    raise ConfigurationError('Nebius AI Studio API key not found. Set NEBIUS_API_KEY environment variable.\nYou can add it to a .env file in:\n  - Current directory: .env\n  - Global config: ~/.massgen/.env')
            elif base_url and 'openrouter.ai' in base_url:
                api_key = os.getenv('OPENROUTER_API_KEY')
                if not api_key:
                    raise ConfigurationError('OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.\nYou can add it to a .env file in:\n  - Current directory: .env\n  - Global config: ~/.massgen/.env')
            elif base_url and ('z.ai' in base_url or 'bigmodel.cn' in base_url):
                api_key = os.getenv('ZAI_API_KEY')
                if not api_key:
                    raise ConfigurationError('ZAI API key not found. Set ZAI_API_KEY environment variable.\nYou can add it to a .env file in:\n  - Current directory: .env\n  - Global config: ~/.massgen/.env')
            elif base_url and ('moonshot.ai' in base_url or 'moonshot.cn' in base_url):
                api_key = os.getenv('MOONSHOT_API_KEY') or os.getenv('KIMI_API_KEY')
                if not api_key:
                    raise ConfigurationError('Kimi/Moonshot API key not found. Set MOONSHOT_API_KEY or KIMI_API_KEY environment variable.\nYou can add it to a .env file in:\n  - Current directory: .env\n  - Global config: ~/.massgen/.env')
            elif base_url and 'poe.com' in base_url:
                api_key = os.getenv('POE_API_KEY')
                if not api_key:
                    raise ConfigurationError('POE API key not found. Set POE_API_KEY environment variable.\nYou can add it to a .env file in:\n  - Current directory: .env\n  - Global config: ~/.massgen/.env')
            elif base_url and 'aliyuncs.com' in base_url:
                api_key = os.getenv('QWEN_API_KEY')
                if not api_key:
                    raise ConfigurationError('Qwen API key not found. Set QWEN_API_KEY environment variable.\nYou can add it to a .env file in:\n  - Current directory: .env\n  - Global config: ~/.massgen/.env')
        return ChatCompletionsBackend(api_key=api_key, **kwargs)
    elif backend_type == 'zai':
        api_key = kwargs.get('api_key') or os.getenv('ZAI_API_KEY')
        if not api_key:
            raise ConfigurationError('ZAI API key not found. Set ZAI_API_KEY environment variable.\nYou can add it to a .env file in:\n  - Current directory: .env\n  - Global config: ~/.massgen/.env')
        return ChatCompletionsBackend(api_key=api_key, **kwargs)
    elif backend_type == 'lmstudio':
        return LMStudioBackend(**kwargs)
    elif backend_type == 'vllm':
        return InferenceBackend(backend_type='vllm', **kwargs)
    elif backend_type == 'sglang':
        return InferenceBackend(backend_type='sglang', **kwargs)
    elif backend_type == 'claude_code':
        try:
            pass
        except ImportError:
            raise ConfigurationError('claude-code-sdk not found. Install with: pip install claude-code-sdk')
        return ClaudeCodeBackend(**kwargs)
    elif backend_type == 'azure_openai':
        api_key = kwargs.get('api_key') or os.getenv('AZURE_OPENAI_API_KEY')
        endpoint = kwargs.get('base_url') or os.getenv('AZURE_OPENAI_ENDPOINT')
        if not api_key:
            raise ConfigurationError('Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY or provide in config.')
        if not endpoint:
            raise ConfigurationError('Azure OpenAI endpoint not found. Set AZURE_OPENAI_ENDPOINT or provide base_url in config.')
        return AzureOpenAIBackend(**kwargs)
    else:
        raise ConfigurationError(f'Unsupported backend type: {backend_type}')

def create_agents_from_config(config: Dict[str, Any], orchestrator_config: Optional[Dict[str, Any]]=None) -> Dict[str, ConfigurableAgent]:
    """Create agents from configuration."""
    agents = {}
    agent_entries = [config['agent']] if 'agent' in config else config.get('agents', None)
    if not agent_entries:
        raise ConfigurationError("Configuration must contain either 'agent' or 'agents' section")
    for i, agent_data in enumerate(agent_entries, start=1):
        backend_config = agent_data.get('backend', {})
        if 'cwd' in backend_config:
            variables = {'cwd': backend_config['cwd']}
            backend_config = _substitute_variables(backend_config, variables)
        backend_type = backend_config.get('type') or (get_backend_type_from_model(backend_config['model']) if 'model' in backend_config else None)
        if not backend_type:
            raise ConfigurationError('Backend type must be specified or inferrable from model')
        if orchestrator_config:
            if 'agent_temporary_workspace' in orchestrator_config:
                backend_config['agent_temporary_workspace'] = orchestrator_config['agent_temporary_workspace']
            if 'context_paths' in orchestrator_config:
                agent_context_paths = backend_config.get('context_paths', [])
                orchestrator_context_paths = orchestrator_config['context_paths']
                merged_paths = orchestrator_context_paths.copy()
                orchestrator_paths_set = {path.get('path') for path in orchestrator_context_paths}
                for agent_path in agent_context_paths:
                    if agent_path.get('path') not in orchestrator_paths_set:
                        merged_paths.append(agent_path)
                backend_config['context_paths'] = merged_paths
        backend = create_backend(backend_type, **backend_config)
        backend_params = {k: v for k, v in backend_config.items() if k != 'type'}
        backend_type_lower = backend_type.lower()
        if backend_type_lower == 'openai':
            agent_config = AgentConfig.create_openai_config(**backend_params)
        elif backend_type_lower == 'claude':
            agent_config = AgentConfig.create_claude_config(**backend_params)
        elif backend_type_lower == 'grok':
            agent_config = AgentConfig.create_grok_config(**backend_params)
        elif backend_type_lower == 'gemini':
            agent_config = AgentConfig.create_gemini_config(**backend_params)
        elif backend_type_lower == 'zai':
            agent_config = AgentConfig.create_zai_config(**backend_params)
        elif backend_type_lower == 'chatcompletion':
            agent_config = AgentConfig.create_chatcompletion_config(**backend_params)
        elif backend_type_lower == 'lmstudio':
            agent_config = AgentConfig.create_lmstudio_config(**backend_params)
        elif backend_type_lower == 'vllm':
            agent_config = AgentConfig.create_vllm_config(**backend_params)
        elif backend_type_lower == 'sglang':
            agent_config = AgentConfig.create_sglang_config(**backend_params)
        else:
            agent_config = AgentConfig(backend_params=backend_config)
        agent_config.agent_id = agent_data.get('id', f'agent{i}')
        system_msg = agent_data.get('system_message')
        if system_msg:
            if backend_type_lower == 'claude_code':
                agent_config.backend_params['append_system_prompt'] = system_msg
            else:
                agent_config.custom_system_instruction = system_msg
        agent = ConfigurableAgent(config=agent_config, backend=backend)
        agents[agent.config.agent_id] = agent
    return agents

def test_initialization_with_valid_adapter():
    """Test backend initialization with valid adapter type."""
    backend = ExternalAgentBackend(adapter_type='test')
    assert backend.adapter_type == 'test'
    assert isinstance(backend.adapter, SimpleTestAdapter)
    assert backend.get_provider_name() == 'test'

def test_initialization_with_invalid_adapter():
    """Test backend initialization with invalid adapter type."""
    with pytest.raises(ValueError) as exc_info:
        ExternalAgentBackend(adapter_type='nonexistent')
    assert 'Unsupported framework' in str(exc_info.value)
    assert 'nonexistent' in str(exc_info.value)

def test_adapter_type_case_insensitive():
    """Test that adapter type is case-insensitive."""
    backend1 = ExternalAgentBackend(adapter_type='TEST')
    backend2 = ExternalAgentBackend(adapter_type='Test')
    backend3 = ExternalAgentBackend(adapter_type='test')
    assert backend1.adapter_type == 'test'
    assert backend2.adapter_type == 'test'
    assert backend3.adapter_type == 'test'

def test_extract_adapter_config():
    """Test extraction of adapter-specific config."""
    backend = ExternalAgentBackend(adapter_type='test', type='test', agent_id='test_agent', session_id='session_1', custom_param='value', temperature=0.7)
    assert 'custom_param' in backend.adapter.config
    assert 'temperature' in backend.adapter.config
    assert 'type' not in backend.adapter.config
    assert 'agent_id' not in backend.adapter.config
    assert 'session_id' not in backend.adapter.config

def test_is_stateful_default():
    """Test stateful check with default adapter."""
    backend = ExternalAgentBackend(adapter_type='test')
    assert backend.is_stateful() is False

def test_clear_history():
    """Test clearing history."""
    backend = ExternalAgentBackend(adapter_type='test')
    backend.adapter._conversation_history = [{'role': 'user', 'content': 'test'}]
    backend.clear_history()
    assert len(backend.adapter._conversation_history) == 0

def test_reset_state():
    """Test resetting state."""
    backend = ExternalAgentBackend(adapter_type='test')
    backend.adapter._conversation_history = [{'role': 'user', 'content': 'test'}]
    backend.reset_state()
    assert len(backend.adapter._conversation_history) == 0

