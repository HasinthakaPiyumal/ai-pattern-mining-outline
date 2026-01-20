# Cluster 1

class CreateComponentTool(Tool[CreateComponentInput, None, StringToolOutput]):
    """
    Tool for creating new components (agents or tools) in the Smart Library.
    This tool can generate code based on natural language requirements using templates
    appropriate for the component type and framework. It handles all aspects of
    component creation including code generation, metadata, and registration.
    """
    name = 'CreateComponentTool'
    description = 'Create new agents and tools from requirements or specifications, with automatic code generation'
    input_schema = CreateComponentInput

    def __init__(self, smart_library: SmartLibrary, llm_service: LLMService, firmware: Optional[Firmware]=None, options: Optional[Dict[str, Any]]=None):
        super().__init__(options=options or {})
        self.library = smart_library
        self.llm = llm_service
        self.firmware = firmware or Firmware()

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=['tool', 'library', 'create'], creator=self)

    async def _run(self, input: CreateComponentInput, options: Optional[Dict[str, Any]]=None, context: Optional[RunContext]=None) -> StringToolOutput:
        """
        Create a new component in the Smart Library.
        
        Args:
            input: The component creation parameters
        
        Returns:
            StringToolOutput containing the creation result in JSON format
        """
        try:
            if not input.code_snippet and input.requirements:
                firmware_content = self.firmware.get_firmware_prompt(input.domain)
                code_snippet = await self._generate_code_from_requirements(input.record_type, input.domain, input.name, input.description, input.requirements, firmware_content, input.framework)
            else:
                code_snippet = input.code_snippet or '# No code provided'
            metadata = input.metadata or {}
            if 'framework' not in metadata:
                metadata['framework'] = input.framework
            if input.record_type == 'AGENT':
                required_tools = self._extract_required_tools(code_snippet)
                if required_tools:
                    metadata['required_tools'] = required_tools
            metadata['creation_strategy'] = {'method': 'requirements' if input.requirements else 'direct_code', 'timestamp': datetime.now(timezone.utc).isoformat(), 'requirements_summary': self._summarize_text(input.requirements) if input.requirements else None}
            record = await self.library.create_record(name=input.name, record_type=input.record_type, domain=input.domain, description=input.description, code_snippet=code_snippet, tags=input.tags or [input.domain, input.record_type.lower()], metadata=metadata)
            created_at_iso = record['created_at']
            if isinstance(created_at_iso, datetime):
                created_at_iso = created_at_iso.isoformat()
            response_data = {'status': 'success', 'message': f"Created new {input.record_type} '{input.name}'", 'record_id': record['id'], 'record': {'name': record['name'], 'type': record['record_type'], 'domain': record['domain'], 'description': record['description'], 'version': record['version'], 'created_at': created_at_iso, 'code_size': len(code_snippet), 'metadata': metadata}, 'next_steps': ['Register this component with the Agent Bus to make it available to other agents', 'Test the component with sample inputs to verify functionality', f"Consider evolving this component if it doesn't fully meet requirements"]}
            return StringToolOutput(safe_json_dumps(response_data))
        except Exception as e:
            import traceback
            return StringToolOutput(safe_json_dumps({'status': 'error', 'message': f'Error creating component: {str(e)}', 'details': traceback.format_exc()}))

    def _extract_required_tools(self, code_snippet: str) -> List[str]:
        """Extract required tools from agent code."""
        import re
        tools_pattern = 'tools=\\[(.*?)\\]'
        tools_match = re.search(tools_pattern, code_snippet, re.DOTALL)
        if not tools_match:
            return []
        tools_str = tools_match.group(1)
        tool_classes = re.findall('(\\w+)\\(\\)', tools_str)
        return tool_classes

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def _summarize_text(self, text: str, max_length: int=100) -> str:
        """Create a short summary of text."""
        if not text:
            return ''
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + '...'

    async def _generate_code_from_requirements(self, record_type: str, domain: str, name: str, description: str, requirements: str, firmware_content: str, framework: str='beeai') -> str:
        """
        Generate code based on natural language requirements.
        
        This method contains the logic for creating component code from requirements,
        using the appropriate templates and frameworks.
        
        Args:
            record_type: Type of component (AGENT or TOOL)
            domain: Domain for the component
            name: Name of the component
            description: Description of the component
            requirements: Natural language requirements
            firmware_content: Firmware content to inject
            framework: Framework to use (default: beeai)
            
        Returns:
            Generated code snippet
        """
        import os
        templates_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'templates')
        os.makedirs(templates_dir, exist_ok=True)
        if framework.lower() == 'beeai':
            agent_template_path = os.path.join(templates_dir, 'beeai_agent_template.txt')
            tool_template_path = os.path.join(templates_dir, 'beeai_tool_template.txt')
        else:
            agent_template_path = os.path.join(templates_dir, 'beeai_agent_template.txt')
            tool_template_path = os.path.join(templates_dir, 'beeai_tool_template.txt')
        try:
            with open(agent_template_path, 'r') as f:
                agent_template = f.read()
        except FileNotFoundError:
            agent_template = '# Default agent template would be here'
        try:
            with open(tool_template_path, 'r') as f:
                tool_template = f.read()
        except FileNotFoundError:
            tool_template = '# Default tool template would be here'
        if record_type == 'AGENT':
            if framework.lower() == 'beeai':
                creation_prompt = f'\n                {firmware_content}\n\n                Create a Python agent using the BeeAI framework that fulfills these requirements:\n                "{requirements}"\n\n                AGENT NAME: {name}\n                DOMAIN: {domain}\n                DESCRIPTION: {description}\n\n                IMPORTANT REQUIREMENTS:\n                1. The agent must be a properly implemented BeeAI ReActAgent \n                2. Use the following framework imports:\n                - from beeai_framework.agents.react import ReActAgent\n                - from beeai_framework.agents.types import AgentMeta\n                - from beeai_framework.memory import TokenMemory or UnconstrainedMemory\n\n                3. The agent must follow this structure - implementing a class with a create_agent method:\n\n                REFERENCE TEMPLATE FOR A BEEAI AGENT:\n                ```python\n    {agent_template}\n                ```\n\n                YOUR TASK:\n                Create a similar agent class for: "{requirements}"\n                - Replace the WeatherAgentInitializer with {name}Initializer\n                - Adapt the description and functionality for the {domain} domain\n                - Include all required disclaimers from the firmware\n                - Specify any tools the agent should use\n                - The code must be complete and executable\n\n                CODE:\n                '
            else:
                creation_prompt = f'\n                {firmware_content}\n\n                Create a Python agent that fulfills these requirements:\n                "{requirements}"\n\n                AGENT NAME: {name}\n                DOMAIN: {domain}\n                DESCRIPTION: {description}\n\n                The agent should be properly implemented with:\n                - Clear class and method structure\n                - Appropriate error handling\n                - Domain-specific functionality for {domain}\n                - All required disclaimers from the firmware\n\n                CODE:\n                '
        elif framework.lower() == 'beeai':
            creation_prompt = f"""\n                {firmware_content}\n\n                Create a Python tool using the BeeAI framework that fulfills these requirements:\n                "{requirements}"\n\n                TOOL NAME: {name}\n                DOMAIN: {domain}\n                DESCRIPTION: {description}\n\n                IMPORTANT REQUIREMENTS:\n                1. The tool must be a properly implemented BeeAI Tool class\n                2. Use the following framework imports:\n                - from beeai_framework.tools.tool import Tool, StringToolOutput\n                - from beeai_framework.context import RunContext\n                - from pydantic import BaseModel, Field\n\n                3. The tool must follow this structure with an input schema and _run method:\n\n                REFERENCE TEMPLATE FOR A BEEAI TOOL:\n                ```python\n    {tool_template}\n                ```\n\n                YOUR TASK:\n                Create a similar tool class for: "{requirements}"\n                - Use {name} as the class name\n                - Create an appropriate input schema class named {name}Input\n                - Define proper input fields with descriptions\n                - Implement the _run method with appropriate logic\n                - Include error handling\n                - For domain '{domain}', include all required disclaimers\n                - The code must be complete and executable\n\n                CODE:\n                """
        else:
            creation_prompt = f'\n                {firmware_content}\n\n                Create a Python tool that fulfills these requirements:\n                "{requirements}"\n\n                TOOL NAME: {name}\n                DOMAIN: {domain}\n                DESCRIPTION: {description}\n\n                The tool should be properly implemented with:\n                - Clear input parameters\n                - Appropriate error handling\n                - Domain-specific functionality for {domain}\n                - All required disclaimers from the firmware\n\n                CODE:\n                '
        return await self.llm.generate(creation_prompt)

class EvolveComponentTool(Tool[EvolveComponentInput, None, StringToolOutput]):
    """
    Tool for evolving existing components in the Smart Library.
    This tool handles various evolution strategies to adapt components to new requirements
    or different domains. It can preserve or radically change functionality based on
    the selected strategy, while maintaining compatibility with the original component.
    """
    name = 'EvolveComponentTool'
    description = 'Evolve existing agents and tools with various strategies to adapt them to new requirements or domains'
    input_schema = EvolveComponentInput

    def __init__(self, smart_library: SmartLibrary, llm_service: LLMService, firmware: Optional[Firmware]=None, options: Optional[Dict[str, Any]]=None):
        super().__init__(options=options or {})
        self.library = smart_library
        self.llm = llm_service
        self.firmware = firmware or Firmware()
        self.evolution_strategies = {'standard': {'description': 'Balanced evolution that preserves core functionality while adding new features', 'preservation_level': 0.7, 'prompt_modifier': 'Evolve the code to implement the requested changes while preserving most of the original functionality.'}, 'conservative': {'description': 'Minimal changes to the original component, focusing on compatibility', 'preservation_level': 0.9, 'prompt_modifier': 'Make minimal changes to the code, preserving as much of the original functionality as possible while addressing the specific changes requested.'}, 'aggressive': {'description': 'Significant changes to optimize for the new requirements, less focus on preserving original functionality', 'preservation_level': 0.4, 'prompt_modifier': 'Reimagine the component with a focus on the new requirements, while maintaining only essential elements of the original code.'}, 'domain_adaptation': {'description': 'Specialized for adapting components to new domains', 'preservation_level': 0.6, 'prompt_modifier': 'Adapt the code to the new domain, adjusting domain-specific elements while preserving the core logic.'}}

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=['tool', 'library', 'evolve'], creator=self)

    async def _run(self, input: EvolveComponentInput, options: Optional[Dict[str, Any]]=None, context: Optional[RunContext]=None) -> StringToolOutput:
        """
        Evolve an existing component in the Smart Library.
        
        Args:
            input: Evolution parameters
        
        Returns:
            StringToolOutput containing the evolution result in JSON format
        """
        try:
            parent_record = await self.library.find_record_by_id(input.parent_id)
            if not parent_record:
                return StringToolOutput(safe_json_dumps({'status': 'error', 'message': f'Parent record with ID {input.parent_id} not found'}))
            strategy = input.evolution_strategy or 'standard'
            if input.target_domain and input.target_domain != parent_record.get('domain'):
                strategy = 'domain_adaptation'
            strategy_details = self.evolution_strategies.get(strategy, self.evolution_strategies['standard'])
            target_domain = input.target_domain or parent_record.get('domain', 'general')
            firmware_content = self.firmware.get_firmware_prompt(target_domain)
            new_code = await self._generate_evolved_code(parent_record, input.changes, input.new_requirements, firmware_content, target_domain, strategy_details)
            evolved_record = await self.library.evolve_record(parent_id=input.parent_id, new_code_snippet=new_code, description=input.new_description or parent_record['description'], new_version=input.new_version)
            if input.target_domain and input.target_domain != parent_record.get('domain'):
                evolved_record['domain'] = input.target_domain
                if 'metadata' not in evolved_record:
                    evolved_record['metadata'] = {}
                evolved_record['metadata']['domain_adaptation'] = {'original_domain': parent_record.get('domain', 'unknown'), 'target_domain': input.target_domain, 'adaptation_timestamp': datetime.now(timezone.utc).isoformat()}
                await self.library.save_record(evolved_record)
            if 'metadata' not in evolved_record:
                evolved_record['metadata'] = {}
            evolved_record['metadata']['evolution_strategy'] = {'strategy': strategy, 'description': strategy_details['description'], 'preservation_level': strategy_details['preservation_level'], 'timestamp': datetime.now(timezone.utc).isoformat()}
            await self.library.save_record(evolved_record)
            created_at_iso = evolved_record['created_at']
            if isinstance(created_at_iso, datetime):
                created_at_iso = created_at_iso.isoformat()
            response_data = {'status': 'success', 'message': f"Successfully evolved {parent_record['record_type']} '{parent_record['name']}' using '{strategy}' strategy", 'parent_id': input.parent_id, 'evolved_id': evolved_record['id'], 'strategy': {'name': strategy, 'description': strategy_details['description'], 'preservation_level': strategy_details['preservation_level']}, 'evolved_record': {'name': evolved_record['name'], 'type': evolved_record['record_type'], 'domain': evolved_record['domain'], 'description': evolved_record['description'], 'version': evolved_record['version'], 'created_at': created_at_iso}, 'next_steps': ['Test the evolved component to verify it meets the new requirements', 'Register the evolved component with the Agent Bus if needed', "Consider further evolution if it doesn't fully meet requirements"]}
            return StringToolOutput(safe_json_dumps(response_data))
        except Exception as e:
            import traceback
            return StringToolOutput(safe_json_dumps({'status': 'error', 'message': f'Error evolving component: {str(e)}', 'details': traceback.format_exc()}))

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    async def _generate_evolved_code(self, record: Dict[str, Any], changes: str, new_requirements: Optional[str]=None, firmware_content: str='', target_domain: Optional[str]=None, strategy: Dict[str, Any]=None) -> str:
        """
        Generate evolved code for a component using the specified strategy.
        
        This method contains the logic for evolving components, applying different
        strategies based on the evolution needs.
        
        Args:
            record: Original component record
            changes: Description of changes to make
            new_requirements: New requirements to incorporate
            firmware_content: Firmware content to inject
            target_domain: Target domain for adaptation
            strategy: Strategy details for evolution
            
        Returns:
            Evolved code snippet
        """
        if not strategy:
            strategy = self.evolution_strategies['standard']
        framework = record.get('metadata', {}).get('framework', 'beeai')
        prompt_modifier = strategy.get('prompt_modifier', '')
        domain_instruction = f'TARGET DOMAIN ADAPTATION: Adapt to the {target_domain} domain' if target_domain else ''
        evolution_prompt = f'\n        {firmware_content}\n\n        ORIGINAL {record['record_type']} CODE:\n        ```python\n        {record['code_snippet']}\n        ```\n\n        REQUESTED CHANGES:\n        {changes}\n\n        {(f'NEW REQUIREMENTS TO INCORPORATE: {new_requirements}' if new_requirements else '')}\n        {domain_instruction}\n\n        EVOLUTION STRATEGY:\n        {strategy.get('description', 'Standard evolution')}\n        \n        INSTRUCTIONS:\n        1. {prompt_modifier}\n        2. Ensure the code follows {framework} framework standards\n        3. Include appropriate error handling\n        4. Follow all firmware guidelines\n        5. Maintain compatibility with the original interface\n        6. Focus particularly on:\n           - Accurate implementation of the requested changes\n           - Proper integration with existing functionality\n           - Clear documentation of what has changed\n\n        EVOLVED CODE:\n        '
        return await self.llm.generate(evolution_prompt)

class CreateOpenAIAgentTool(Tool[CreateOpenAIAgentInput, None, StringToolOutput]):
    """
    Tool for creating OpenAI agents in the Smart Library.
    This tool creates agent records specifically designed to work with OpenAI's Agents SDK.
    """
    name = 'CreateOpenAIAgentTool'
    description = 'Create OpenAI agents with the OpenAI Agents SDK, configuring model, tools, and governance'
    input_schema = CreateOpenAIAgentInput

    def __init__(self, smart_library: SmartLibrary, llm_service: LLMService, firmware: Optional[Firmware]=None, options: Optional[Dict[str, Any]]=None):
        super().__init__(options=options or {})
        self.library = smart_library
        self.llm = llm_service
        self.firmware = firmware or Firmware()

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=['tool', 'openai', 'create_agent'], creator=self)

    async def _run(self, input: CreateOpenAIAgentInput, options: Optional[Dict[str, Any]]=None, context: Optional[RunContext]=None) -> StringToolOutput:
        """
        Create a new OpenAI agent in the Smart Library.
        
        Args:
            input: The agent creation parameters
        
        Returns:
            StringToolOutput containing the creation result in JSON format
        """
        try:
            code_snippet = f'\nfrom agents import Agent, Runner\nfrom agents.model_settings import ModelSettings\n\n# Create an OpenAI Agent\nagent = Agent(\n    name="{input.name}",\n    instructions="""\n{input.description}\n""",\n    model="{input.model}",\n    model_settings=ModelSettings(\n        temperature={input.temperature}\n    )\n)\n\n# Example usage with async\nasync def run_agent(input_text):\n    result = await Runner.run(agent, input_text)\n    return result.final_output\n\n# Example usage with sync\ndef run_agent_sync(input_text):\n    result = Runner.run_sync(agent, input_text)\n    return result.final_output\n'
            base_metadata = input.metadata or {}
            metadata = {**base_metadata, 'framework': 'openai-agents', 'model': input.model, 'model_settings': {'temperature': input.temperature}, 'guardrails_enabled': input.guardrails_enabled}
            if input.required_tools:
                metadata['required_tools'] = input.required_tools
            record = await self.library.create_record(name=input.name, record_type='AGENT', domain=input.domain, description=input.description, code_snippet=code_snippet, tags=input.tags or ['openai', input.domain, 'agent'], metadata=metadata)
            return StringToolOutput(json.dumps({'status': 'success', 'message': f"Created new OpenAI agent '{input.name}'", 'record_id': record['id'], 'record': {'name': record['name'], 'type': record['record_type'], 'domain': record['domain'], 'description': record['description'], 'version': record['version'], 'created_at': record['created_at'], 'framework': 'openai-agents', 'model': input.model}, 'next_steps': ['Use the OpenAI agent with appropriate tools', 'Execute the agent with the OpenAIAgentExecutionTool', 'Consider adding guardrails with the AddOpenAIGuardrailsTool']}, indent=2))
        except Exception as e:
            import traceback
            return StringToolOutput(json.dumps({'status': 'error', 'message': f'Error creating OpenAI agent: {str(e)}', 'details': traceback.format_exc()}, indent=2))

