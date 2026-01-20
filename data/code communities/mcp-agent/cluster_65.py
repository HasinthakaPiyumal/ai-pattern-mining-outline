# Cluster 65

def create_llm(agent: Agent | AgentSpec | None=None, agent_name: str | None=None, server_names: List[str] | None=None, instruction: str | None=None, provider: str='openai', model: str | ModelPreferences | None=None, request_params: RequestParams | None=None, context: Context | None=None) -> AugmentedLLM:
    """
    Create an Augmented LLM from an agent, agent spec, or agent name.
    """
    if isinstance(agent_name, str):
        agent_obj = agent_from_spec(AgentSpec(name=agent_name, instruction=instruction, server_names=server_names or []), context=context)
    elif isinstance(agent, AgentSpec):
        agent_obj = agent_from_spec(agent, context=context)
    else:
        agent_obj = agent
    factory = _llm_factory(provider=provider, model=model, request_params=request_params, context=context)
    return factory(agent=agent_obj)

def agent_from_spec(spec: AgentSpec, context: Context | None=None) -> Agent:
    return Agent(name=spec.name, instruction=spec.instruction, server_names=spec.server_names or [], functions=getattr(spec, 'functions', []), connection_persistence=spec.connection_persistence, human_input_callback=getattr(spec, 'human_input_callback', None) or (context.human_input_handler if context else None), context=context)

def create_orchestrator(*, available_agents: Sequence[AgentSpec | Agent | AugmentedLLM], planner: AgentSpec | Agent | AugmentedLLM | None=None, synthesizer: AgentSpec | Agent | AugmentedLLM | None=None, plan_type: Literal['full', 'iterative']='full', provider: SupportedLLMProviders='openai', model: str | ModelPreferences | None=None, overrides: OrchestratorOverrides | None=None, name: str | None=None, context: Context | None=None, **kwargs) -> Orchestrator:
    """
    In the orchestrator-workers workflow, a planner LLM dynamically breaks down tasks,
    delegates them to worker LLMs, and synthesizes their results. It does this
    in a loop until the task is complete.

    This is a simpler (and faster) form of the [deep orchestrator](https://github.com/lastmile-ai/mcp-agent/blob/main/src/mcp_agent/workflows/deep_orchestrator/README.md) workflow,
    which is more suitable for complex, long-running tasks with multiple agents and MCP servers where the number of agents is not known in advance.

    Args:
        available_agents: The agents/LLMs/workflows that can be used to execute the task.
        plan_type: The type of plan to use for the orchestrator ["full", "iterative"].
            "full" planning generates the full plan first, then executes. "iterative" plans the next step, and loops until success.
        provider: The provider to use for the LLM.
        model: The model to use as the LLM.
        overrides: Optional overrides for instructions and prompt templates.
        name: The name of this orchestrator workflow. Can be used as an identifier.
        context: The context to use for the orchestrator.
    """
    factory = _llm_factory(provider=provider, model=model, context=context)
    agents: List[Agent | AugmentedLLM] = []
    for item in available_agents:
        if isinstance(item, AgentSpec):
            agents.append(agent_from_spec(item, context=context))
        else:
            agents.append(item)
    planner_obj: Agent | AugmentedLLM | None = None
    synthesizer_obj: Agent | AugmentedLLM | None = None
    if planner:
        planner_obj = planner if isinstance(planner, Agent | AugmentedLLM) else agent_from_spec(planner, context=context)
    if synthesizer:
        synthesizer_obj = synthesizer if isinstance(synthesizer, Agent | AugmentedLLM) else agent_from_spec(synthesizer, context=context)
    return Orchestrator(llm_factory=factory, name=name, planner=planner_obj, synthesizer=synthesizer_obj, available_agents=agents, plan_type=plan_type, overrides=overrides, context=context, **kwargs)

def create_deep_orchestrator(*, available_agents: Sequence[AgentSpec | Agent | AugmentedLLM], config: DeepOrchestratorConfig | None=None, name: str | None=None, provider: SupportedLLMProviders='openai', model: str | ModelPreferences | None=None, context: Context | None=None, **kwargs) -> DeepOrchestrator:
    """
    Create a deep research-style orchestrator workflow that can be used to execute complex, long-running tasks with
    multiple agents and MCP servers.

    Args:
        available_agents: The agents/LLMs/workflows that can be used to execute the task.
        config: The configuration for the deep orchestrator.
        name: The name of this deep orchestrator workflow. Can be used as an identifier.
        provider: The provider to use for the LLM.
        model: The model to use as the LLM.
        context: The context to use for the LLM.
    """
    factory = _llm_factory(provider=provider, model=model, context=context)
    agents: List[Agent | AugmentedLLM] = config.available_agents if config and config.available_agents else []
    for item in available_agents:
        if isinstance(item, AgentSpec):
            agents.append(agent_from_spec(item, context=context))
        else:
            agents.append(item)
    if config is None:
        config = DeepOrchestratorConfig.from_simple()
    config.available_agents = agents
    config.name = name or config.name
    return DeepOrchestrator(llm_factory=factory, config=config, context=context, **kwargs)

def create_parallel_llm(*, fan_in: AgentSpec | Agent | AugmentedLLM | Callable[[FanInInput], Any], fan_out: List[AgentSpec | Agent | AugmentedLLM | Callable] | None=None, name: str | None=None, provider: SupportedLLMProviders | None='openai', model: str | ModelPreferences | None=None, request_params: RequestParams | None=None, context=None, **kwargs) -> ParallelLLM:
    """
    Create a parallel workflow that can fan out to multiple agents to execute in parallel, and fan in/aggregate the results.

    Args:
        fan_in: The agent/LLM/workflow that generates responses.
        fan_out: The agents/LLMs/workflows that generate responses.
        name: The name of the parallel workflow. Can be used to identify the workflow in logs.
        provider: The provider to use for the LLM.
        model: The model to use as the LLM.
        request_params: The default request parameters to use for the LLM.
        context: The context to use for the LLM.
    """
    factory = _llm_factory(provider=provider, model=model, request_params=request_params, context=context)
    fan_in_agent_or_llm: Agent | AugmentedLLM | Callable[[FanInInput], Any]
    if isinstance(fan_in, AgentSpec):
        fan_in_agent_or_llm = agent_from_spec(fan_in, context=context)
    else:
        fan_in_agent_or_llm = fan_in
    fan_out_agents: List[Agent | AugmentedLLM] = []
    fan_out_functions: List[Callable] = []
    for item in fan_out or []:
        if isinstance(item, AgentSpec):
            fan_out_agents.append(agent_from_spec(item, context=context))
        elif isinstance(item, Agent):
            fan_out_agents.append(item)
        elif isinstance(item, AugmentedLLM):
            fan_out_agents.append(item)
        elif callable(item):
            fan_out_functions.append(item)
    return ParallelLLM(fan_in_agent=fan_in_agent_or_llm, fan_out_agents=fan_out_agents or None, fan_out_functions=fan_out_functions or None, name=name, llm_factory=factory, context=context, **kwargs)

def create_evaluator_optimizer_llm(*, optimizer: AgentSpec | Agent | AugmentedLLM, evaluator: str | AgentSpec | Agent | AugmentedLLM, name: str | None=None, min_rating: int | None=None, max_refinements: int=3, provider: SupportedLLMProviders | None=None, model: str | ModelPreferences | None=None, request_params: RequestParams | None=None, context: Context | None=None, **kwargs) -> EvaluatorOptimizerLLM:
    """
    Create an evaluator-optimizer workflow that generates responses and evaluates them iteratively until they achieve a necessary quality criteria.

    Args:
        optimizer: The agent/LLM/workflow that generates responses.
        evaluator: The agent/LLM that evaluates responses
        name: The name of the evaluator-optimizer workflow.
        min_rating: Minimum acceptable quality rating
        max_refinements: Maximum refinement iterations (max number of times to refine the response)
        provider: The provider to use for the LLM.
        model: The model to use as the LLM.
        request_params: The default request parameters to use for the LLM.
        context: The context to use for the LLM.

    """
    factory = _llm_factory(provider=provider, model=model, request_params=request_params, context=context)
    optimizer_obj: AugmentedLLM | Agent
    evaluator_obj: str | AugmentedLLM | Agent
    optimizer_obj = agent_from_spec(optimizer, context=context) if isinstance(optimizer, AgentSpec) else optimizer
    if isinstance(evaluator, AgentSpec):
        evaluator_obj = agent_from_spec(evaluator, context=context)
    else:
        evaluator_obj = evaluator
    return EvaluatorOptimizerLLM(optimizer=optimizer_obj, evaluator=evaluator_obj, name=name, min_rating=min_rating, max_refinements=max_refinements, llm_factory=factory, context=context, **kwargs)

