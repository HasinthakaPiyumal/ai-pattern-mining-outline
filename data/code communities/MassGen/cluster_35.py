# Cluster 35

def setup_agent_from_config(config: Dict[str, Any], default_llm_config: Any=None) -> ConversableAgent:
    """
    Set up a ConversableAgent from configuration.

    Args:
        config: Agent configuration dict
        default_llm_config: Default llm_config to use if agent doesn't provide one

    Returns:
        ConversableAgent or AssistantAgent instance
    """
    cfg = config.copy()
    has_llm_config = 'llm_config' in cfg
    validate_agent_config(cfg, require_llm_config=not default_llm_config)
    agent_type = cfg.pop('type', 'conversable')
    if has_llm_config:
        llm_config = create_llm_config(cfg.pop('llm_config'))
    elif default_llm_config:
        llm_config = create_llm_config(default_llm_config)
    else:
        raise ValueError('No llm_config provided for agent and no default_llm_config available')
    code_executor = None
    if 'code_execution_config' in cfg:
        code_exec_config = cfg.pop('code_execution_config')
        if 'executor' in code_exec_config:
            code_executor = create_code_executor(code_exec_config['executor'])
    agent_kwargs = build_agent_kwargs(cfg, llm_config, code_executor)
    if agent_type == 'assistant':
        return AssistantAgent(**agent_kwargs)
    elif agent_type == 'conversable':
        return ConversableAgent(**agent_kwargs)
    else:
        raise ValueError(f"Unsupported AG2 agent type: {agent_type}. Use 'assistant' or 'conversable' for ag2 agents.")

def validate_agent_config(cfg: Dict[str, Any], require_llm_config: bool=True) -> None:
    """
    Validate required fields in agent configuration.

    Args:
        cfg: Agent configuration dict
        require_llm_config: If True, llm_config is required. If False, it's optional.
    """
    if require_llm_config and 'llm_config' not in cfg:
        raise ValueError("Each AG2 agent configuration must include 'llm_config'.")
    if 'name' not in cfg:
        raise ValueError("Each AG2 agent configuration must include 'name'.")

def create_llm_config(llm_config_data: Any) -> LLMConfig:
    """
    Create LLMConfig from dict or list format.

    Supports new AG2 syntax:
    - Single dict: LLMConfig({'model': 'gpt-4', 'api_key': '...'})
    - List of dicts: LLMConfig({'model': 'gpt-4', ...}, {'model': 'gpt-3.5', ...})
    """
    if isinstance(llm_config_data, list):
        return LLMConfig(*llm_config_data)
    elif isinstance(llm_config_data, dict):
        return LLMConfig(llm_config_data)
    else:
        raise ValueError(f'llm_config must be a dict or list, got {type(llm_config_data)}')

def create_code_executor(executor_config: Dict[str, Any]) -> Any:
    """Create code executor from configuration."""
    executor_type = executor_config.get('type')
    if not executor_type:
        raise ValueError("code_execution_config.executor must include 'type' field")
    executor_params = {k: v for k, v in executor_config.items() if k != 'type'}
    if executor_type == 'LocalCommandLineCodeExecutor':
        from autogen.coding import LocalCommandLineCodeExecutor
        return LocalCommandLineCodeExecutor(**executor_params)
    elif executor_type == 'DockerCommandLineCodeExecutor':
        from autogen.coding import DockerCommandLineCodeExecutor
        return DockerCommandLineCodeExecutor(**executor_params)
    elif executor_type == 'YepCodeCodeExecutor':
        from autogen.coding import YepCodeCodeExecutor
        return YepCodeCodeExecutor(**executor_params)
    elif executor_type == 'JupyterCodeExecutor':
        from autogen.coding.jupyter import JupyterCodeExecutor
        return JupyterCodeExecutor(**executor_params)
    else:
        raise ValueError(f'Unsupported code executor type: {executor_type}. Supported types: LocalCommandLineCodeExecutor, DockerCommandLineCodeExecutor, YepCodeCodeExecutor, JupyterCodeExecutor')

def build_agent_kwargs(cfg: Dict[str, Any], llm_config: LLMConfig, code_executor: Any=None) -> Dict[str, Any]:
    """Build kwargs for agent initialization."""
    agent_kwargs = {'name': cfg['name'], 'system_message': cfg.get('system_message', 'You are a helpful AI assistant.'), 'human_input_mode': 'NEVER', 'llm_config': llm_config}
    if code_executor is not None:
        agent_kwargs['code_execution_config'] = {'executor': code_executor}
    return agent_kwargs

def test_create_llm_config_from_dict():
    """Test creating LLMConfig from dictionary."""
    config_dict = {'api_type': 'openai', 'model': 'gpt-4o', 'temperature': 0.7}
    llm_config = create_llm_config(config_dict)
    assert llm_config is not None
    assert hasattr(llm_config, 'config_list')

def test_create_llm_config_from_list():
    """Test creating LLMConfig from list of configs."""
    config_list = [{'api_type': 'openai', 'model': 'gpt-4o'}, {'api_type': 'google', 'model': 'gemini-pro'}]
    llm_config = create_llm_config(config_list)
    assert llm_config is not None
    assert hasattr(llm_config, 'config_list')

@patch('massgen.adapters.utils.ag2_utils.AssistantAgent')
def test_setup_agent_from_config_assistant(mock_assistant):
    """Test setting up AssistantAgent from config."""
    config = {'type': 'assistant', 'name': 'test_agent', 'system_message': 'You are helpful', 'llm_config': {'api_type': 'openai', 'model': 'gpt-4o'}}
    setup_agent_from_config(config)
    mock_assistant.assert_called_once()
    call_kwargs = mock_assistant.call_args[1]
    assert call_kwargs['name'] == 'test_agent'
    assert call_kwargs['system_message'] == 'You are helpful'
    assert call_kwargs['human_input_mode'] == 'NEVER'

@patch('massgen.adapters.utils.ag2_utils.ConversableAgent')
def test_setup_agent_from_config_conversable(mock_conversable):
    """Test setting up ConversableAgent from config."""
    config = {'type': 'conversable', 'name': 'test_agent', 'llm_config': [{'api_type': 'openai', 'model': 'gpt-4o'}]}
    setup_agent_from_config(config)
    mock_conversable.assert_called_once()
    call_kwargs = mock_conversable.call_args[1]
    assert call_kwargs['name'] == 'test_agent'
    assert call_kwargs['human_input_mode'] == 'NEVER'

def test_setup_agent_missing_llm_config():
    """Test that missing llm_config raises error."""
    config = {'type': 'assistant', 'name': 'test_agent'}
    with pytest.raises(ValueError) as exc_info:
        setup_agent_from_config(config)
    assert 'llm_config' in str(exc_info.value)

def test_setup_agent_missing_name():
    """Test that missing name raises error."""
    config = {'type': 'assistant', 'llm_config': [{'api_type': 'openai', 'model': 'gpt-4o'}]}
    with pytest.raises(ValueError) as exc_info:
        setup_agent_from_config(config)
    assert 'name' in str(exc_info.value)

@patch('autogen.coding.LocalCommandLineCodeExecutor')
@patch('massgen.adapters.utils.ag2_utils.AssistantAgent')
def test_setup_agent_with_local_code_executor(mock_assistant, mock_executor):
    """Test setting up agent with LocalCommandLineCodeExecutor."""
    config = {'type': 'assistant', 'name': 'coder', 'llm_config': [{'api_type': 'openai', 'model': 'gpt-4o'}], 'code_execution_config': {'executor': {'type': 'LocalCommandLineCodeExecutor', 'timeout': 60, 'work_dir': './workspace'}}}
    setup_agent_from_config(config)
    mock_executor.assert_called_once_with(timeout=60, work_dir='./workspace')
    call_kwargs = mock_assistant.call_args[1]
    assert 'code_execution_config' in call_kwargs
    assert 'executor' in call_kwargs['code_execution_config']

@patch('autogen.coding.DockerCommandLineCodeExecutor')
@patch('massgen.adapters.utils.ag2_utils.ConversableAgent')
def test_setup_agent_with_docker_executor(mock_conversable, mock_executor):
    """Test setting up agent with DockerCommandLineCodeExecutor."""
    config = {'type': 'conversable', 'name': 'docker_coder', 'llm_config': [{'api_type': 'openai', 'model': 'gpt-4o'}], 'code_execution_config': {'executor': {'type': 'DockerCommandLineCodeExecutor', 'image': 'python:3.10', 'timeout': 120}}}
    setup_agent_from_config(config)
    mock_executor.assert_called_once_with(image='python:3.10', timeout=120)
    call_kwargs = mock_conversable.call_args[1]
    assert 'code_execution_config' in call_kwargs

def test_setup_agent_invalid_executor_type():
    """Test that invalid executor type raises error."""
    config = {'type': 'assistant', 'name': 'coder', 'llm_config': [{'api_type': 'openai', 'model': 'gpt-4o'}], 'code_execution_config': {'executor': {'type': 'InvalidExecutor', 'timeout': 60}}}
    with pytest.raises(ValueError) as exc_info:
        setup_agent_from_config(config)
    assert 'Unsupported code executor type' in str(exc_info.value)
    assert 'InvalidExecutor' in str(exc_info.value)

def test_setup_agent_missing_executor_type():
    """Test that missing executor type raises error."""
    config = {'type': 'assistant', 'name': 'coder', 'llm_config': [{'api_type': 'openai', 'model': 'gpt-4o'}], 'code_execution_config': {'executor': {'timeout': 60}}}
    with pytest.raises(ValueError) as exc_info:
        setup_agent_from_config(config)
    assert "must include 'type' field" in str(exc_info.value)

