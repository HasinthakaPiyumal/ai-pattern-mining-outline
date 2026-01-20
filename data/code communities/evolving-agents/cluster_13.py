# Cluster 13

def extract_component_id(response_text: str, id_field_key: str='record_id') -> Optional[str]:
    """
    Extracts a component ID from a response string.
    It first tries to parse the string as JSON, then falls back to regex.
    """
    if not response_text:
        return None
    try:
        json_match = re.search('\\{.*\\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            if isinstance(data, dict):
                if id_field_key in data and isinstance(data[id_field_key], str):
                    return data[id_field_key]
                if 'record_id' in data and isinstance(data['record_id'], str):
                    return data['record_id']
                if 'evolved_id' in data and isinstance(data['evolved_id'], str):
                    return data['evolved_id']
                if 'plan_id' in data and isinstance(data['plan_id'], str):
                    return data['plan_id']
                if 'id' in data and isinstance(data['id'], str):
                    return data['id']
                for nested_key in ['record', 'evolved_record', 'saved_record']:
                    if nested_key in data and isinstance(data[nested_key], dict):
                        if 'id' in data[nested_key] and isinstance(data[nested_key]['id'], str):
                            return data[nested_key]['id']
                if 'results' in data and isinstance(data['results'], list) and data['results']:
                    if isinstance(data['results'][0], dict) and 'id' in data['results'][0]:
                        return data['results'][0].get('id')
                logger.debug(f'JSON parsed from response, but no standard ID field found. Data: {str(data)[:200]}')
        else:
            logger.debug(f'No JSON object found in response_text. Falling back to regex. Response: {response_text[:200]}')
    except json.JSONDecodeError:
        logger.debug(f'Response_text is not valid JSON, or JSON part extraction failed. Falling back to regex. Response: {response_text[:200]}')
    except Exception as e:
        logger.error(f'Unexpected error during JSON parsing in extract_component_id: {e}')
    patterns = [f'"{id_field_key}":\\s*"([^"]+)"', f'"record_id":\\s*"([^"]+)"', f'"evolved_id":\\s*"([^"]+)"', f'"plan_id":\\s*"([^"]+)"', f'"id":\\s*"([^"]+)"', 'ID:\\s*([a-fA-F0-9]{8}-(?:[a-fA-F0-9]{4}-){3}[a-fA-F0-9]{12}|[a-fA-F0-9]{24}|[a-zA-Z0-9_.\\-]+)', 'record id:\\s*([a-fA-F0-9]{8}-(?:[a-fA-F0-9]{4}-){3}[a-fA-F0-9]{12}|[a-zA-Z0-9_.\\-]+)', 'evolved id:\\s*([a-fA-F0-9]{8}-(?:[a-fA-F0-9]{4}-){3}[a-fA-F0-9]{12}|[a-zA-Z0-9_.\\-]+)', 'plan id:\\s*([a-fA-F0-9]{8}-(?:[a-fA-F0-9]{4}-){3}[a-fA-F0-9]{12}|[a-zA-Z0-9_.\\-]+)', '([a-fA-F0-9]{8}-(?:[a-fA-F0-9]{4}-){3}[a-fA-F0-9]{12})']
    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match and match.group(1):
            extracted_id = match.group(1).strip()
            if len(extracted_id) > 5 and (not extracted_id.lower().startswith('http')):
                logger.info(f"Regex extracted ID: '{extracted_id}' using pattern: {pattern}")
                return extracted_id
            else:
                logger.debug(f"Regex match '{extracted_id}' for pattern {pattern} too short or looks like URL, skipping.")
    logger.warning(f'Could not extract component ID using JSON parsing or regex from: {response_text[:300]}...')
    return None

def ensure_templates_exist():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    templates_dir = os.path.join(base_dir, 'evolving_agents', 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    agent_template_path = os.path.join(templates_dir, 'beeai_agent_template.txt')
    if not os.path.exists(agent_template_path):
        agent_template = 'from typing import List, Dict, Any, Optional\nimport logging\n\nfrom beeai_framework.agents.react import ReActAgent\nfrom beeai_framework.agents.types import AgentMeta\n# TokenMemory is now imported at the top level\nfrom beeai_framework.backend.chat import ChatModel\nfrom beeai_framework.tools.tool import Tool\n\nclass WeatherAgentInitializer:\n    """\n    A specialized agent for providing weather information and forecasts.\n    This agent can retrieve current weather, forecasts, and provide recommendations.\n    """\n    \n    @staticmethod\n    def create_agent(\n        llm: ChatModel, \n        tools: Optional[List[Tool]] = None,\n        memory_type: str = "token"\n    ) -> ReActAgent:\n        """\n        Create and configure the weather agent.\n        \n        Args:\n            llm: The language model to use\n            tools: Optional list of tools to use\n            memory_type: Type of memory to use\n            \n        Returns:\n            Configured ReActAgent instance\n        """\n        if tools is None:\n            tools = []\n            \n        meta = AgentMeta(\n            name="WeatherAgent",\n            description=(\n                "I am a weather expert that can provide current conditions, "\n                "forecasts, and weather-related recommendations."\n            ),\n            tools=tools\n        )\n        \n        memory = TokenMemory(llm)\n        \n        agent = ReActAgent(\n            llm=llm,\n            tools=tools,\n            memory=memory,\n            meta=meta\n        )\n        \n        return agent'
        with open(agent_template_path, 'w') as f:
            f.write(agent_template)
    tool_template_path = os.path.join(templates_dir, 'beeai_tool_template.txt')
    if not os.path.exists(tool_template_path):
        tool_template = 'from typing import Dict, Any, Optional\nfrom pydantic import BaseModel, Field\n\nfrom beeai_framework.tools.tool import Tool, StringToolOutput\nfrom beeai_framework.context import RunContext\nfrom beeai_framework.emitter.emitter import Emitter\n\nclass WeatherToolInput(BaseModel):\n    """Input schema for WeatherTool."""\n    location: str = Field(description="City or location to get weather for")\n    units: str = Field(default="metric", description="Units to use (metric/imperial)")\n\nclass WeatherTool(Tool[WeatherToolInput, None, StringToolOutput]):\n    """Tool for retrieving current weather conditions and forecasts."""\n    \n    name = "WeatherTool"\n    description = "Get current weather conditions and forecasts for a location"\n    input_schema = WeatherToolInput\n    \n    def __init__(self, api_key: Optional[str] = None, options: Optional[Dict[str, Any]] = None):\n        super().__init__(options=options or {})\n        self.api_key = api_key\n        \n    def _create_emitter(self) -> Emitter:\n        return Emitter.root().child(\n            namespace=["tool", "weather", "conditions"],\n            creator=self,\n        )\n    \n    async def _run(\n        self, \n        tool_input: WeatherToolInput, \n        options: Optional[Dict[str, Any]] = None, \n        context: Optional[RunContext] = None\n    ) -> StringToolOutput:\n        """\n        Get weather for the specified location.\n        """\n        try:\n            weather_data = self._get_mock_weather(tool_input.location, tool_input.units)\n            \n            return StringToolOutput(\n                f"Weather for {tool_input.location}:\\n"\n                f"Temperature: {weather_data[\'temp\']}°{\'C\' if tool_input.units == \'metric\' else \'F\'}\\n"\n                f"Condition: {weather_data[\'condition\']}\\n"\n                f"Humidity: {weather_data[\'humidity\']}%"\n            )\n        except Exception as e:\n            return StringToolOutput(f"Error retrieving weather: {str(e)}")\n            \n    def _get_mock_weather(self, location: str, units: str) -> Dict[str, Any]:\n        """Mock weather data for demonstration."""\n        return {\n            "temp": 22 if units == "metric" else 72,\n            "condition": "Partly Cloudy",\n            "humidity": 65\n        }'
        with open(tool_template_path, 'w') as f:
            f.write(tool_template)
    print('✓ BeeAI templates verified')

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

