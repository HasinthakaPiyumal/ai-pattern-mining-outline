# Cluster 4

def extract_id_from_json_text(response_text: str, key_to_find: str='id') -> Optional[str]:
    if not response_text:
        return None
    try:
        json_match = re.search('(\\{.*\\})', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            if isinstance(data, dict):
                if key_to_find in data and isinstance(data[key_to_find], str):
                    return data[key_to_find]
                for k_var in ['record_id', 'evolved_id', 'plan_id', 'id']:
                    if k_var == key_to_find:
                        continue
                    if k_var in data and isinstance(data[k_var], str):
                        return data[k_var]
                for nested_key in ['record', 'evolved_record', 'saved_record', 'component', 'agent', 'tool', 'evolved_agent', 'created_agent']:
                    if nested_key in data and isinstance(data[nested_key], dict):
                        target_dict = data[nested_key]
                        if key_to_find in target_dict and isinstance(target_dict[key_to_find], str):
                            return target_dict[key_to_find]
                        if 'id' in target_dict and isinstance(target_dict['id'], str):
                            return target_dict['id']
                if 'results' in data and isinstance(data['results'], list) and data['results']:
                    if isinstance(data['results'][0], dict):
                        first_result = data['results'][0]
                        if key_to_find in first_result:
                            return first_result.get(key_to_find)
                        if 'id' in first_result:
                            return first_result.get('id')
    except Exception:
        pass
    patterns = [f'"{key_to_find}":\\s*"([^"]+)"']
    if key_to_find not in ['id', 'record_id', 'evolved_id']:
        patterns.extend([f'"record_id":\\s*"([^"]+)"', f'"evolved_id":\\s*"([^"]+)"', f'"id":\\s*"([^"]+)"'])
    patterns.extend(['ID:\\s*([a-fA-F0-9]{8}-(?:[a-fA-F0-9]{4}-){3}[a-fA-F0-9]{12}|[a-fA-F0-9]{24}|[a-zA-Z0-9_.\\-]+)', '([a-fA-F0-9]{8}-(?:[a-fA-F0-9]{4}-){3}[a-fA-F0-9]{12})'])
    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match and match.group(1):
            extracted_id = match.group(1).strip()
            if len(extracted_id) > 5 and (not extracted_id.lower().startswith('http')):
                logger.info(f"Regex extracted ID '{extracted_id}' for key '{key_to_find}' using pattern: {pattern}")
                return extracted_id
    logger.warning(f"Could not extract ID for key '{key_to_find}' from SystemAgent response: {response_text[:200]}...")
    return None

class AgentFactory:
    """
    Factory for creating and executing agents across different frameworks.
    """

    def __init__(self, smart_library: SmartLibrary, llm_service: LLMService, provider_registry: Optional[ProviderRegistry]=None):
        """
        Initialize the agent factory.
        
        Args:
            smart_library: Smart library for retrieving agent records
            llm_service: LLM service for text generation
            provider_registry: Optional provider registry to use
        """
        self.library = smart_library
        self.llm_service = llm_service
        self.active_agents = {}
        self.provider_registry = provider_registry or ProviderRegistry()
        if not self.provider_registry.list_available_providers():
            self._register_default_providers()
        logger.info(f'Agent Factory initialized with providers: {self.provider_registry.list_available_providers()}')

    def _register_default_providers(self) -> None:
        """Register the default set of providers."""
        self.provider_registry.register_provider(BeeAIProvider(self.llm_service))
        logger.info(f'Registered default providers: {self.provider_registry.list_available_providers()}')

    def clean_code_snippet(code_snippet):
        """Clean code snippet by removing markdown formatting and fixing common syntax issues."""
        if '```' in code_snippet:
            lines = code_snippet.split('\n')
            clean_lines = []
            inside_code_block = False
            for line in lines:
                if line.strip().startswith('```'):
                    inside_code_block = not inside_code_block
                    continue
                if not inside_code_block:
                    clean_lines.append(line)
            code_snippet = '\n'.join(clean_lines)
        return code_snippet.strip()

    async def create_agent(self, record: Dict[str, Any], firmware_content: Optional[str]=None, tools: Optional[List[Any]]=None, config: Optional[Dict[str, Any]]=None) -> Any:
        """
        Create an agent instance from a library record.
        
        Args:
            record: Agent record from the Smart Library
            firmware_content: Optional firmware content to inject
            tools: Optional list of tools to provide to the agent
            config: Optional configuration parameters
            
        Returns:
            Instantiated agent
        """
        metadata = record.get('metadata', {})
        framework_name = metadata.get('framework', 'beeai')
        config = config or {}
        logger.info(f"Creating agent '{record['name']}' using framework '{framework_name}'")
        if framework_name.lower() == 'beeai':
            try:
                agent = await self._create_beeai_agent_directly(record, tools, config)
                if agent:
                    self.active_agents[record['name']] = {'record': record, 'instance': agent, 'type': 'AGENT', 'framework': framework_name, 'provider_id': 'BeeAIProvider'}
                    return agent
            except Exception as e:
                logger.warning(f'Direct BeeAI agent creation failed: {str(e)}, falling back to provider')
        provider = self.provider_registry.get_provider_for_framework(framework_name)
        if not provider:
            error_msg = f"No provider found for framework '{framework_name}'"
            logger.error(error_msg)
            raise ValueError(error_msg)
        try:
            agent_instance = await provider.create_agent(record=record, tools=tools, firmware_content=firmware_content, config=config)
            self.active_agents[record['name']] = {'record': record, 'instance': agent_instance, 'type': 'AGENT', 'framework': framework_name, 'provider_id': provider.__class__.__name__}
            logger.info(f"Successfully created agent '{record['name']}' with framework '{framework_name}'")
            return agent_instance
        except Exception as e:
            logger.error(f"Error creating agent '{record['name']}': {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    async def _create_beeai_agent_directly(self, record: Dict[str, Any], tools: Optional[List[Any]]=None, config: Optional[Dict[str, Any]]=None) -> Optional[ReActAgent]:
        """
        Attempt to create a BeeAI agent directly from the code without using a provider.
        
        Args:
            record: Agent record from the Smart Library
            tools: Optional tools to provide to the agent
            config: Optional configuration
            
        Returns:
            BeeAI agent instance if successful, None otherwise
        """
        code_snippet = record['code_snippet']
        class_match = re.search('class\\s+(\\w+)(?:\\(.*\\))?:', code_snippet)
        if not class_match:
            return None
        initializer_class_name = class_match.group(1)
        if 'def create_agent' not in code_snippet:
            return None
        try:
            module_name = f'dynamic_agent_{record['id'].replace('-', '_')}'
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
                f.write(code_snippet)
                temp_file = f.name
            try:
                spec = importlib.util.spec_from_file_location(module_name, temp_file)
                if not spec or not spec.loader:
                    return None
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                if hasattr(module, initializer_class_name):
                    initializer_class = getattr(module, initializer_class_name)
                    if hasattr(initializer_class, 'create_agent'):
                        chat_model = self.llm_service.chat_model
                        import inspect
                        sig = inspect.signature(initializer_class.create_agent)
                        if 'tools' in sig.parameters:
                            agent = initializer_class.create_agent(chat_model, tools)
                        else:
                            agent = initializer_class.create_agent(chat_model)
                        if isinstance(agent, ReActAgent):
                            return agent
            finally:
                os.unlink(temp_file)
                if module_name in sys.modules:
                    del sys.modules[module_name]
        except Exception as e:
            logger.error(f'Error creating BeeAI agent directly: {str(e)}')
            import traceback
            logger.error(traceback.format_exc())
        return None

    async def execute_agent(self, agent_name: str, input_text: str, execution_config: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        """
        Execute an agent by name with input text.
        
        Args:
            agent_name: Name of the agent to execute
            input_text: Input text to process
            execution_config: Optional execution configuration parameters
            
        Returns:
            Execution result dictionary
        """
        if agent_name not in self.active_agents:
            error_msg = f"Agent '{agent_name}' not found in active agents"
            logger.error(error_msg)
            return {'status': 'error', 'message': error_msg}
        agent_info = self.active_agents[agent_name]
        agent_instance = agent_info['instance']
        framework_name = agent_info.get('framework', 'default')
        logger.info(f"Executing agent '{agent_name}' using framework '{framework_name}'")
        provider = self.provider_registry.get_provider_for_framework(framework_name)
        if not provider:
            error_msg = f"No provider found for framework '{framework_name}'"
            logger.error(error_msg)
            return {'status': 'error', 'message': error_msg}
        try:
            result = await provider.execute_agent(agent_instance=agent_instance, input_text=input_text, execution_config=execution_config)
            if hasattr(self.library, 'update_usage_metrics') and 'record' in agent_info:
                await self.library.update_usage_metrics(agent_info['record']['id'], result['status'] == 'success')
            return result
        except Exception as e:
            logger.error(f"Error executing agent '{agent_name}': {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'status': 'error', 'message': f"Error executing agent '{agent_name}': {str(e)}"}

class ExecuteOpenAIAgentTool(Tool[ExecuteOpenAIAgentInput, None, StringToolOutput]):
    """
    Tool for executing OpenAI agents from the Smart Library.
    This tool loads and runs agents created with the OpenAI Agents SDK.
    """
    name = 'ExecuteOpenAIAgentTool'
    description = 'Execute OpenAI agents stored in the Smart Library with input text and optional tools'
    input_schema = ExecuteOpenAIAgentInput

    def __init__(self, smart_library: SmartLibrary, openai_provider: Optional[OpenAIAgentsProvider]=None, tool_factory=None, options: Optional[Dict[str, Any]]=None):
        super().__init__(options=options or {})
        self.library = smart_library
        self.provider = openai_provider or OpenAIAgentsProvider()
        self.tool_factory = tool_factory
        self.active_agents = {}

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=['tool', 'openai', 'execute_agent'], creator=self)

    async def _run(self, input: ExecuteOpenAIAgentInput, options: Optional[Dict[str, Any]]=None, context: Optional[RunContext]=None) -> StringToolOutput:
        """
        Execute an OpenAI agent from the Smart Library.
        
        Args:
            input: The execution parameters
        
        Returns:
            StringToolOutput containing the execution result
        """
        try:
            agent_record = await self._get_agent_record(input.agent_id_or_name)
            if not agent_record:
                return StringToolOutput(json.dumps({'status': 'error', 'message': f"Agent '{input.agent_id_or_name}' not found"}, indent=2))
            metadata = agent_record.get('metadata', {})
            framework = metadata.get('framework', '').lower()
            if framework != 'openai-agents':
                return StringToolOutput(json.dumps({'status': 'error', 'message': f"Agent '{input.agent_id_or_name}' is not an OpenAI agent"}, indent=2))
            agent, tools = await self._get_or_create_agent(agent_record, input.tools_to_use, input.apply_firmware)
            result = await self._execute_agent(agent, input.input_text, input.max_turns)
            return StringToolOutput(json.dumps({'status': 'success', 'agent_name': agent_record['name'], 'input': input.input_text, 'output': result['result'], 'tools_used': [t.name for t in tools] if tools else [], 'execution_details': {'max_turns': input.max_turns, 'firmware_applied': input.apply_firmware}}, indent=2))
        except Exception as e:
            import traceback
            logger.error(f'Error executing OpenAI agent: {str(e)}')
            logger.error(traceback.format_exc())
            return StringToolOutput(json.dumps({'status': 'error', 'message': f'Error executing OpenAI agent: {str(e)}', 'details': traceback.format_exc()}, indent=2))

    async def _get_agent_record(self, agent_id_or_name: str) -> Optional[Dict[str, Any]]:
        """Get an agent record by ID or name."""
        record = await self.library.find_record_by_id(agent_id_or_name)
        if record:
            return record
        return await self.library.find_record_by_name(agent_id_or_name, 'AGENT')

    async def _get_or_create_agent(self, record: Dict[str, Any], tool_names: Optional[List[str]]=None, apply_firmware: bool=True) -> Tuple[OpenAIAgent, List[Any]]:
        """Get an existing agent instance or create a new one."""
        if record['id'] in self.active_agents:
            return (self.active_agents[record['id']]['instance'], self.active_agents[record['id']]['tools'])
        tools = []
        if tool_names and self.tool_factory:
            for tool_name in tool_names:
                tool_record = await self.library.find_record_by_name(tool_name, 'TOOL')
                if tool_record:
                    tool = await self.tool_factory.create_tool(tool_record)
                    tools.append(tool)
        openai_tools = []
        if tools:
            openai_tools = [OpenAIToolAdapter.convert_evolving_tool_to_openai(t) for t in tools]
        agent = await self.provider.create_agent(record=record, tools=openai_tools, firmware_content='Apply governance rules' if apply_firmware else None, config={'apply_firmware': apply_firmware})
        self.active_agents[record['id']] = {'instance': agent, 'tools': tools, 'record': record}
        return (agent, tools)

    async def _execute_agent(self, agent: OpenAIAgent, input_text: str, max_turns: int=10) -> Dict[str, Any]:
        """Execute an OpenAI agent with input text."""
        try:
            context = {}
            result = await Runner.run(agent, input_text, context=context, max_turns=max_turns)
            return {'status': 'success', 'result': str(result.final_output), 'raw_result': result}
        except Exception as e:
            logger.error(f'Error in _execute_agent: {str(e)}')
            import traceback
            logger.error(traceback.format_exc())
            raise

