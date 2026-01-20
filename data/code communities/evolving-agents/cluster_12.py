# Cluster 12

class SystemAgentFactory:

    @staticmethod
    async def create_agent(llm_service: Optional[LLMService]=None, smart_library: Optional[SmartLibrary]=None, agent_bus: Optional[SmartAgentBus]=None, mongodb_client: Optional[MongoDBClient]=None, memory_type: str='token', container: Optional[DependencyContainer]=None) -> ReActAgent:
        logger.debug(f'SystemAgentFactory: Received container: {container is not None}')

        def _resolve_dependency(name: str, provided_instance: Optional[Any], default_factory: Optional[callable]=None):
            instance = provided_instance
            if not instance and container and container.has(name):
                instance = container.get(name)
                logger.debug(f"SystemAgentFactory: Retrieved '{name}' from container.")
            elif instance:
                logger.debug(f"SystemAgentFactory: Using directly passed '{name}'.")
            elif default_factory:
                logger.warning(f"SystemAgentFactory: '{name}' not in container or provided, creating default.")
                instance = default_factory()
                if container and instance:
                    container.register(name, instance)
            else:
                logger.error(f"SystemAgentFactory: Critical dependency '{name}' not found or provided, and no default factory.")
                raise ValueError(f'{name} is required but was not found or provided.')
            if instance is None:
                raise ValueError(f'Critical Error: {name} resolved to None unexpectedly.')
            return instance
        resolved_llm_service = _resolve_dependency('llm_service', llm_service, lambda: LLMService())
        chat_model = resolved_llm_service.chat_model
        logger.debug(f'SystemAgentFactory: Using LLMService with chat_model: {chat_model is not None}')

        def smart_lib_factory():
            if container:
                return SmartLibrary(container=container)
            else:
                resolved_mongo_for_lib = mongodb_client or (container.get('mongodb_client') if container and container.has('mongodb_client') else MongoDBClient())
                return SmartLibrary(llm_service=resolved_llm_service, mongodb_client=resolved_mongo_for_lib)
        resolved_smart_library = _resolve_dependency('smart_library', smart_library, smart_lib_factory)
        resolved_mongodb_client = _resolve_dependency('mongodb_client', mongodb_client, lambda: MongoDBClient())

        def agent_bus_factory():
            if container:
                return SmartAgentBus(container=container)
            else:
                return SmartAgentBus(smart_library=resolved_smart_library, llm_service=resolved_llm_service, mongodb_client=resolved_mongodb_client)
        resolved_agent_bus = _resolve_dependency('agent_bus', agent_bus, agent_bus_factory)
        resolved_firmware = _resolve_dependency('firmware', None, lambda: Firmware())
        logger.debug('SystemAgentFactory: Instantiating tools...')
        search_tool = SearchComponentTool(resolved_smart_library)
        create_tool = CreateComponentTool(resolved_smart_library, resolved_llm_service, resolved_firmware)
        evolve_tool = EvolveComponentTool(resolved_smart_library, resolved_llm_service, resolved_firmware)
        register_tool = RegisterAgentTool(resolved_agent_bus)
        request_tool = RequestAgentTool(resolved_agent_bus)
        discover_tool = DiscoverAgentTool(resolved_agent_bus)
        generate_workflow_tool = GenerateWorkflowTool(resolved_llm_service, resolved_smart_library)
        process_workflow_tool = ProcessWorkflowTool(mongodb_client=resolved_mongodb_client, container=container)
        task_context_tool = TaskContextTool(resolved_llm_service)
        contextual_search_tool = ContextualSearchTool(task_context_tool, search_tool)
        workflow_design_review_tool = WorkflowDesignReviewTool()
        component_selection_review_tool = ComponentSelectionReviewTool()
        approve_plan_tool = ApprovePlanTool(llm_service=resolved_llm_service, mongodb_client=resolved_mongodb_client, container=container)
        experience_recorder_tool = ExperienceRecorderTool(agent_bus=resolved_agent_bus, memory_manager_agent_id=MEMORY_MANAGER_AGENT_ID)
        context_builder_tool = ContextBuilderTool(agent_bus=resolved_agent_bus, smart_library=resolved_smart_library, memory_manager_agent_id=MEMORY_MANAGER_AGENT_ID, llm_service=resolved_llm_service)
        tools = [contextual_search_tool, task_context_tool, search_tool, create_tool, evolve_tool, register_tool, request_tool, discover_tool, generate_workflow_tool, process_workflow_tool, workflow_design_review_tool, component_selection_review_tool, approve_plan_tool, experience_recorder_tool, context_builder_tool]
        logger.debug(f'SystemAgentFactory: {len(tools)} tools instantiated.')
        meta = AgentMeta(name='SystemAgent', description="I am the System Agent, the central orchestrator for the agent ecosystem. My primary purpose is to help you reuse, evolve, and create agents and tools to solve your problems efficiently. I leverage a Smart Memory system to learn from past experiences and build rich, task-relevant context. I find the most relevant components by deeply understanding the specific task context you're working in, often using the ContextBuilderTool to gather historical data and library components. After significant tasks, I use the ExperienceRecorderTool to log outcomes for future learning. I can also operate in a human-in-the-loop workflow where my plans are reviewed before execution to ensure safety and appropriateness.", extra_description="When faced with a complex task, I first consider if similar tasks have been done before by using the ContextBuilderTool to retrieve relevant experiences and summarize message histories. This tool also helps me find suitable components from the SmartLibrary. This enriched context informs my planning, whether it's designing a workflow, selecting components, or delegating to other agents. After completing a significant workflow or achieving a key objective, I should remember to use the ExperienceRecorderTool to save the process and outcome. This helps the entire system learn and improve. My goal is to deliver the final result effectively and learn from each interaction.", tools=tools)
        memory = UnconstrainedMemory() if memory_type == 'unconstrained' else TokenMemory(chat_model)
        system_agent = ReActAgent(llm=chat_model, tools=tools, memory=memory, meta=meta)
        system_agent.tools_map = {tool_instance.name: tool_instance for tool_instance in tools}
        logger.debug(f'SystemAgent tools_map contains: {list(system_agent.tools_map.keys())}')
        if container and (not container.has('system_agent')):
            container.register('system_agent', system_agent)
            logger.debug('SystemAgentFactory: Registered SystemAgent instance in container.')
        if resolved_agent_bus:
            if resolved_agent_bus._system_agent_instance is None:
                resolved_agent_bus._system_agent_instance = system_agent
                logger.debug('SystemAgentFactory: Set SystemAgent instance on the resolved AgentBus.')
            elif resolved_agent_bus._system_agent_instance is not system_agent:
                logger.warning('SystemAgentFactory: AgentBus already had a different SystemAgent instance. Overwriting.')
                resolved_agent_bus._system_agent_instance = system_agent
        else:
            logger.error('SystemAgentFactory: resolved_agent_bus is None, cannot set system_agent property on it.')
        logger.info('SystemAgent created successfully with updated tool initializations.')
        return system_agent

