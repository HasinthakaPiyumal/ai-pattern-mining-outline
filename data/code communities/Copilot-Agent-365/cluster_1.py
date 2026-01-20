# Cluster 1

class Assistant:

    def __init__(self, declared_agents):
        self.config = {'assistant_name': str(os.environ.get('ASSISTANT_NAME', 'BusinessInsightBot')), 'characteristic_description': str(os.environ.get('CHARACTERISTIC_DESCRIPTION', 'helpful business assistant'))}
        try:
            self.client = AzureOpenAI(api_key=os.environ['AZURE_OPENAI_API_KEY'], api_version=os.environ['AZURE_OPENAI_API_VERSION'], azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'])
        except TypeError:
            self.client = AzureOpenAI(api_key=os.environ['AZURE_OPENAI_API_KEY'], azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'])
        self.known_agents = self.reload_agents(declared_agents)
        self.user_guid = DEFAULT_USER_GUID
        self.shared_memory = None
        self.user_memory = None
        self.storage_manager = AzureFileStorageManager()
        self._initialize_context_memory(DEFAULT_USER_GUID)

    def _check_first_message_for_guid(self, conversation_history):
        """Check if the first message contains only a GUID"""
        if not conversation_history or len(conversation_history) == 0:
            return None
        first_message = conversation_history[0]
        if first_message.get('role') == 'user':
            content = first_message.get('content')
            if content is None:
                return None
            content = str(content).strip()
            guid_pattern = re.compile('^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
            if guid_pattern.match(content):
                return content
        return None

    def _initialize_context_memory(self, user_guid=None):
        """Initialize context memory with separate shared and user-specific memories"""
        try:
            context_memory_agent = self.known_agents.get('ContextMemory')
            if not context_memory_agent:
                self.shared_memory = 'No shared context memory available.'
                self.user_memory = 'No specific context memory available.'
                return
            self.storage_manager.set_memory_context(None)
            self.shared_memory = str(context_memory_agent.perform(full_recall=True))
            if not user_guid:
                user_guid = DEFAULT_USER_GUID
            self.storage_manager.set_memory_context(user_guid)
            self.user_memory = str(context_memory_agent.perform(user_guid=user_guid, full_recall=True))
        except Exception as e:
            logging.warning(f'Error initializing context memory: {str(e)}')
            self.shared_memory = 'Context memory initialization failed.'
            self.user_memory = 'Context memory initialization failed.'

    def _extract_demo_state_from_history(self, conversation_history):
        """
        Extract active demo state from conversation history (stateless approach).
        Returns: (demo_name, current_step, demo_steps_list) or (None, 0, None)
        """
        if not conversation_history:
            return (None, 0, None)
        for message in reversed(conversation_history):
            if message.get('role') == 'system':
                content = str(message.get('content', ''))
                if 'DemoCompletion' in content or 'Demo finished' in content or 'DemoExit' in content:
                    return (None, 0, None)
                match = re.search('Performed (\\S+) and got result:.*Step (\\d+) of (\\d+)', content)
                if match:
                    demo_name = match.group(1)
                    current_step = int(match.group(2))
                    total_steps = int(match.group(3))
                    try:
                        demo_content = self.storage_manager.read_file('demos', f'{demo_name}.json')
                        if demo_content:
                            demo_data = json.loads(demo_content)
                            demo_steps = demo_data.get('conversation_flow', [])
                            logging.info(f'Extracted demo state from history: {demo_name}, step {current_step}/{len(demo_steps)}')
                            return (demo_name, current_step, demo_steps)
                    except Exception as e:
                        logging.error(f'Error loading demo {demo_name}: {str(e)}')
                        return (None, 0, None)
        return (None, 0, None)

    def extract_user_guid(self, text):
        """Try to extract a GUID from user input, but only if it's the entire message"""
        if text is None:
            return None
        text_str = str(text).strip()
        guid_pattern = re.compile('^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
        match = guid_pattern.match(text_str)
        if match:
            return match.group(0)
        labeled_guid_pattern = re.compile('^guid[:=\\s]+([0-9a-f-]{36})$', re.IGNORECASE)
        match = labeled_guid_pattern.match(text_str)
        if match:
            return match.group(1)
        return None

    def check_demo_trigger(self, user_message):
        """Check if user message matches any demo trigger phrases (stateless)"""
        try:
            demo_files = self.storage_manager.list_files('demos')
            user_message_lower = user_message.lower().strip()
            for file in demo_files:
                if not file.name.endswith('.json'):
                    continue
                try:
                    demo_content = self.storage_manager.read_file('demos', file.name)
                    if not demo_content:
                        continue
                    demo_data = json.loads(demo_content)
                    trigger_phrases = demo_data.get('trigger_phrases', [])
                    for phrase in trigger_phrases:
                        if phrase.lower().strip() == user_message_lower:
                            demo_name = file.name.replace('.json', '')
                            conversation_flow = demo_data.get('conversation_flow', [])
                            logging.info(f'Triggered demo: {demo_name} with {len(conversation_flow)} steps')
                            return {'triggered': True, 'demo_name': demo_name, 'demo_data': demo_data, 'conversation_flow': conversation_flow}
                except Exception as e:
                    logging.error(f'Error checking demo {file.name}: {str(e)}')
                    continue
            return {'triggered': False}
        except Exception as e:
            logging.error(f'Error in check_demo_trigger: {str(e)}')
            return {'triggered': False}

    def get_agent_metadata(self):
        agents_metadata = []
        for agent in self.known_agents.values():
            if hasattr(agent, 'metadata'):
                agents_metadata.append(agent.metadata)
        return agents_metadata

    def reload_agents(self, agent_objects):
        known_agents = {}
        if isinstance(agent_objects, dict):
            for agent_name, agent in agent_objects.items():
                if hasattr(agent, 'name'):
                    known_agents[agent.name] = agent
                else:
                    known_agents[str(agent_name)] = agent
        elif isinstance(agent_objects, list):
            for agent in agent_objects:
                if hasattr(agent, 'name'):
                    known_agents[agent.name] = agent
        else:
            logging.warning(f'Unexpected agent_objects type: {type(agent_objects)}')
        return known_agents

    def prepare_messages(self, conversation_history):
        if not isinstance(conversation_history, list):
            conversation_history = []
        messages = []
        current_datetime = datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')
        system_message = {'role': 'system', 'content': f"""\n<identity>\nYou are a Microsoft Copilot assistant named {str(self.config.get('assistant_name', 'Assistant'))}, operating within Microsoft Teams.\n</identity>\n\n<shared_memory_output>\nThese are memories accessible by all users of the system:\n{str(self.shared_memory)}\n</shared_memory_output>\n\n<specific_memory_output>\nThese are memories specific to the current conversation:\n{str(self.user_memory)}\n</specific_memory_output>\n\n<context_instructions>\n- <shared_memory_output> represents common knowledge shared across all conversations\n- <specific_memory_output> represents specific context for the current conversation\n- Apply specific context with higher precedence than shared context\n- Synthesize information from both contexts for comprehensive responses\n</context_instructions>\n\n<agent_usage>\nIMPORTANT: You must be honest and accurate about agent usage:\n- NEVER pretend or imply you've executed an agent when you haven't actually called it\n- NEVER say "using my agent" unless you are actually making a function call to that agent\n- NEVER fabricate success messages about data operations that haven't occurred\n- If you need to perform an action and don't have the necessary agent, say so directly\n- When a user requests an action, either:\n  1. Call the appropriate agent and report actual results, or\n  2. Say "I don't have the capability to do that" and suggest an alternative\n  3. If no details are provided besides the request to run an agent, infer the necessary input parameters by "reading between the lines" of the conversation context so far\n</agent_usage>\n\n<response_format>\nCRITICAL: You must structure your response in TWO distinct parts separated by the delimiter |||VOICE|||\n\n1. FIRST PART (before |||VOICE|||): Your full formatted response\n   - Use **bold** for emphasis\n   - Use `code blocks` for technical content\n   - Apply --- for horizontal rules to separate sections\n   - Utilize > for important quotes or callouts\n   - Format code with ```language syntax highlighting\n   - Create numbered lists with proper indentation\n   - Add personality when appropriate\n   - Apply # ## ### headings for clear structure\n\n2. SECOND PART (after |||VOICE|||): A concise voice response\n   - Maximum 1-2 sentences\n   - Pure conversational English with NO formatting\n   - Extract only the most critical information\n   - Sound like a colleague speaking casually over a cubicle wall\n   - Be natural and conversational, not robotic\n   - Focus on the key takeaway or action item\n   - Example: "I found those Q3 sales figures - revenue's up 12 percent from last quarter." or "Sure, I'll pull up that customer data for you right now."\n\nEXAMPLE FORMAT:\nHere's the detailed analysis you requested:\n\n**Key Findings:**\n- Revenue increased by 12%\n- Customer satisfaction scores improved\n\n|||VOICE|||\nRevenue's up 12 percent and customers are happier - looking good for Q3.\n</response_format>\n"""}
        messages.append(ensure_string_content(system_message))
        guid_only_first_message = self._check_first_message_for_guid(conversation_history)
        start_idx = 1 if guid_only_first_message else 0
        for i in range(start_idx, len(conversation_history)):
            messages.append(ensure_string_content(conversation_history[i]))
        return messages

    def get_openai_api_call(self, messages):
        try:
            deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-deployment')
            response = self.client.chat.completions.create(model=deployment_name, messages=messages, functions=self.get_agent_metadata(), function_call='auto')
            return response
        except Exception as e:
            logging.error(f'Error in OpenAI API call: {str(e)}')
            raise

    def parse_response_with_voice(self, content):
        """Parse the response to extract formatted and voice parts"""
        if not content:
            return ('', '')
        parts = content.split('|||VOICE|||')
        if len(parts) >= 2:
            formatted_response = parts[0].strip()
            voice_response = parts[1].strip()
        else:
            formatted_response = content.strip()
            sentences = formatted_response.split('.')
            if sentences:
                voice_response = sentences[0].strip() + '.'
                voice_response = re.sub('\\*\\*|`|#|>|---|[\\U00010000-\\U0010ffff]|[\\u2600-\\u26FF]|[\\u2700-\\u27BF]', '', voice_response)
                voice_response = re.sub('\\s+', ' ', voice_response).strip()
            else:
                voice_response = "I've completed your request."
        return (formatted_response, voice_response)

    def get_response(self, prompt, conversation_history, max_retries=3, retry_delay=2):
        guid_from_history = self._check_first_message_for_guid(conversation_history)
        guid_from_prompt = self.extract_user_guid(prompt)
        target_guid = guid_from_history or guid_from_prompt
        if target_guid and target_guid != self.user_guid:
            self.user_guid = target_guid
            self._initialize_context_memory(self.user_guid)
            logging.info(f'User GUID updated to: {self.user_guid}')
        elif not self.user_guid:
            self.user_guid = DEFAULT_USER_GUID
            self._initialize_context_memory(self.user_guid)
            logging.info(f'Using default User GUID: {self.user_guid}')
        prompt = str(prompt) if prompt is not None else ''
        if guid_from_prompt and prompt.strip() == guid_from_prompt and (self.user_guid == guid_from_prompt):
            formatted = "I've successfully loaded your conversation memory. How can I assist you today?"
            voice = "I've loaded your memory - what can I help you with?"
            return (formatted, voice, '')
        active_demo, current_step, demo_steps = self._extract_demo_state_from_history(conversation_history)
        if prompt.lower().strip() in ['exit demo', 'stop demo', 'end demo', 'cancel demo']:
            if active_demo:
                formatted = f'How can I help you?'
                voice = 'What can I help you with?'
                return (formatted, voice, f'Performed DemoExit and got result: {active_demo} terminated by user')
            else:
                formatted = 'How can I help you?'
                voice = 'What can I help you with?'
                return (formatted, voice, '')
        trigger_result = self.check_demo_trigger(prompt)
        if trigger_result.get('triggered'):
            demo_data = trigger_result.get('demo_data', {})
            demo_name = trigger_result.get('demo_name', '')
            conversation_flow = trigger_result.get('conversation_flow', [])
            total_steps = len(conversation_flow)
            scripted_demo_agent = self.known_agents.get('ScriptedDemo')
            if scripted_demo_agent:
                try:
                    canned_response = scripted_demo_agent.perform(action='respond', demo_name=demo_name, user_input=prompt, user_guid=self.user_guid)
                    formatted = canned_response
                    voice_sentences = canned_response.split('.')[:2]
                    voice = '.'.join(voice_sentences).strip()
                    voice = re.sub('\\*\\*|`|#|>|---|[\\U00010000-\\U0010ffff]|[\\u2600-\\u26FF]|[\\u2700-\\u27BF]', '', voice)
                    voice = re.sub('\\s+', ' ', voice).strip()
                    return (formatted, voice, f'Performed {demo_name} and got result: Demo activated - Step 1 of {total_steps}')
                except Exception as e:
                    logging.error(f'Error calling ScriptedDemoAgent on trigger: {str(e)}')
                    formatted = f'I apologize, but I encountered an error retrieving the demo response. Let me help you with that request.'
                    voice = f'I encountered an error, but let me help you with that.'
                    return (formatted, voice, f'Performed {demo_name} and got result: Demo activated - Step 1 (Error)')
            else:
                formatted = f'Let me help you with that!'
                voice = f'Let me help you with that.'
                return (formatted, voice, f'Performed {demo_name} and got result: Demo activated - Step 1 of {total_steps}')
        if active_demo and demo_steps:
            next_step_num = current_step + 1
            total_steps = len(demo_steps)
            if next_step_num > total_steps:
                formatted = f'How else can I help you today?'
                voice = 'What else can I help you with?'
                return (formatted, voice, f'Performed DemoCompletion and got result: {active_demo} finished successfully')
            logging.info(f'Continuing demo {active_demo}: step {next_step_num}/{total_steps}')
            scripted_demo_agent = self.known_agents.get('ScriptedDemo')
            if scripted_demo_agent:
                try:
                    canned_response = scripted_demo_agent.perform(action='respond', demo_name=active_demo, user_input=prompt, user_guid=self.user_guid)
                    formatted = canned_response
                    voice_sentences = canned_response.split('.')[:2]
                    voice = '.'.join(voice_sentences).strip()
                    voice = re.sub('\\*\\*|`|#|>|---|[\\U00010000-\\U0010ffff]|[\\u2600-\\u26FF]|[\\u2700-\\u27BF]', '', voice)
                    voice = re.sub('\\s+', ' ', voice).strip()
                    agent_log = f'Performed {active_demo} and got result: Step {next_step_num} of {total_steps} - Returned canned response'
                    return (formatted, voice, agent_log)
                except Exception as e:
                    logging.error(f'Error calling ScriptedDemoAgent: {str(e)}')
                    formatted = f'I apologize, but I encountered an error. Let me help you with that.'
                    voice = 'Sorry, I hit an error. Let me help you with that.'
                    return (formatted, voice, f'Performed {active_demo} and got result: Error - {str(e)}')
            else:
                formatted = "I'm sorry, I'm unable to access the demo script right now. How else can I help you?"
                voice = "The demo script isn't available right now. How else can I help?"
                return (formatted, voice, 'Performed DemoError and got result: ScriptedDemo agent not found')
        messages = self.prepare_messages(conversation_history)
        messages.append(ensure_string_content({'role': 'user', 'content': prompt}))
        agent_logs = []
        retry_count = 0
        needs_follow_up = False
        while retry_count < max_retries:
            try:
                response = self.get_openai_api_call(messages)
                assistant_msg = response.choices[0].message
                msg_contents = assistant_msg.content or ''
                if not assistant_msg.function_call:
                    formatted_response, voice_response = self.parse_response_with_voice(msg_contents)
                    return (formatted_response, voice_response, '\n'.join(map(str, agent_logs)))
                agent_name = str(assistant_msg.function_call.name)
                agent = self.known_agents.get(agent_name)
                if not agent:
                    return (f"Agent '{agent_name}' does not exist", "I couldn't find that agent.", '')
                json_data = ensure_string_function_args(assistant_msg.function_call)
                logging.info(f'JSON data before parsing: {json_data}')
                try:
                    agent_parameters = safe_json_loads(json_data)
                    sanitized_parameters = {}
                    for key, value in agent_parameters.items():
                        if value is None:
                            sanitized_parameters[key] = ''
                        else:
                            sanitized_parameters[key] = value
                    if agent_name in ['ManageMemory', 'ContextMemory']:
                        sanitized_parameters['user_guid'] = self.user_guid
                    result = agent.perform(**sanitized_parameters)
                    if result is None:
                        result = 'Agent completed successfully'
                    else:
                        result = str(result)
                    agent_logs.append(f'Performed {agent_name} and got result: {result}')
                except Exception as e:
                    return (f'Error parsing parameters: {str(e)}', 'I hit an error processing that.', '')
                messages.append({'role': 'function', 'name': agent_name, 'content': result})
                try:
                    result_json = json.loads(result)
                    needs_follow_up = False
                    if isinstance(result_json, dict):
                        if result_json.get('error') or result_json.get('status') == 'incomplete':
                            needs_follow_up = True
                        if result_json.get('requires_additional_action') == True:
                            needs_follow_up = True
                except:
                    needs_follow_up = False
                if not needs_follow_up:
                    final_response = self.get_openai_api_call(messages)
                    final_msg = final_response.choices[0].message
                    final_content = final_msg.content or ''
                    formatted_response, voice_response = self.parse_response_with_voice(final_content)
                    return (formatted_response, voice_response, '\n'.join(map(str, agent_logs)))
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    logging.warning(f'Error occurred: {str(e)}. Retrying in {retry_delay} seconds...')
                    time.sleep(retry_delay)
                else:
                    logging.error(f'Max retries reached. Error: {str(e)}')
                    return ('An error occurred. Please try again.', 'Something went wrong - try again.', '')
        return ('Service temporarily unavailable. Please try again later.', 'Service is down - try again later.', '')

class GitHubAgentLibraryManager(BasicAgent):
    """
    Comprehensive GitHub Agent Library Manager.
    Manages integration with the GitHub Agent Template Library at kody-w/AI-Agent-Templates.
    Handles both individual agent operations (discover, search, install) and GUID-based agent groups.
    """
    GITHUB_REPO = 'kody-w/AI-Agent-Templates'
    GITHUB_BRANCH = 'main'
    GITHUB_RAW_BASE = f'https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}'
    GITHUB_API_BASE = f'https://api.github.com/repos/{GITHUB_REPO}'

    def __init__(self):
        self.name = 'GitHubAgentLibrary'
        self.metadata = {'name': self.name, 'description': 'Comprehensive manager for the GitHub Agent Template Library at kody-w/AI-Agent-Templates. Discovers, searches, installs, and manages 65+ pre-built agents from the public repository. Also creates GUID-based agent groups for custom deployments. All agents are downloaded from GitHub raw URLs and automatically integrated into your system.', 'parameters': {'type': 'object', 'properties': {'action': {'type': 'string', 'description': "Action to perform: 'discover' (browse ALL 65+ available agents with no parameters needed), 'search' (find specific agents - REQUIRES search_query parameter with keyword like 'email' or 'sales'), 'install' (download and install an agent - REQUIRES agent_id from search/discover results, NEVER guess the agent_id), 'list_installed' (show installed GitHub agents - no parameters), 'update' (update an agent - REQUIRES agent_id), 'remove' (uninstall agent - REQUIRES agent_id), 'get_info' (detailed agent info - REQUIRES agent_id), 'sync_manifest' (refresh catalogue from GitHub - no parameters), 'create_group' (create a GUID-based agent group - REQUIRES agent_ids list), 'list_groups' (show all GUID-based agent groups - no parameters), 'get_group_info' (get details about a specific GUID group - REQUIRES guid parameter). CRITICAL: Before calling 'install', you MUST call 'search' or 'discover' first to get the exact agent_id.", 'enum': ['discover', 'search', 'install', 'list_installed', 'update', 'remove', 'get_info', 'sync_manifest', 'create_group', 'list_groups', 'get_group_info']}, 'agent_id': {'type': 'string', 'description': "REQUIRED for install/update/remove/get_info actions. The unique identifier of the agent (e.g., 'deal_progression_agent', 'email_agent'). CRITICAL: Get this EXACT value from discover or search results first. Do NOT guess or make up agent IDs - they must come from the GitHub library. If you don't have the exact agent_id from a prior search/discover, you MUST search first before attempting to install."}, 'agent_ids': {'type': 'array', 'items': {'type': 'string'}, 'description': "REQUIRED for create_group action: List of agent IDs to fetch from GitHub and group together. Example: ['deal_progression_agent', 'email_agent', 'sales_forecast_agent']. These must be valid agent IDs from the kody-w/AI-Agent-Templates repository."}, 'group_name': {'type': 'string', 'description': "OPTIONAL for create_group action: A friendly name for the agent group (e.g., 'Sales Team Agents'). This is stored with the GUID for reference."}, 'guid': {'type': 'string', 'description': 'REQUIRED for get_group_info action: The GUID of the agent group to retrieve information about.'}, 'stack_path': {'type': 'string', 'description': "OPTIONAL: Only needed when installing a stack agent. Path format: 'industry_stacks/stack_name' (e.g., 'b2b_sales_stacks/deal_progression_stack'). This is provided in search results for stack agents. Leave empty for singular agents."}, 'search_query': {'type': 'string', 'description': "REQUIRED for search action: Keyword to search for in agent names, descriptions, and features. Examples: 'email', 'sales', 'manufacturing', 'automation'. Use broad terms for better results."}, 'category': {'type': 'string', 'description': 'OPTIONAL: Additional filter to narrow results by industry vertical. Only use if user specifically mentions an industry. Available industries: b2b_sales, b2c_sales, energy, federal_government, financial_services, general, healthcare, manufacturing, professional_services, retail_cpg, slg_government, software_dp', 'enum': ['b2b_sales', 'b2c_sales', 'energy', 'federal_government', 'financial_services', 'general', 'healthcare', 'manufacturing', 'professional_services', 'retail_cpg', 'slg_government', 'software_dp']}, 'force': {'type': 'boolean', 'description': 'OPTIONAL: Set to true to reinstall an agent even if it already exists. Default is false. Use when updating/fixing an installed agent.'}}, 'required': ['action']}, 'examples': {'discover_all': {'description': 'Browse all available agents in the library', 'parameters': {'action': 'discover'}}, 'search_by_keyword': {'description': 'Find agents related to email', 'parameters': {'action': 'search', 'search_query': 'email'}}, 'search_by_industry': {'description': 'Find manufacturing agents', 'parameters': {'action': 'search', 'search_query': 'manufacturing', 'category': 'manufacturing'}}, 'search_before_install_workflow': {'description': "CORRECT WORKFLOW: First search for 'maintenance' agents, then use the agent_id from results to install", 'steps': [{'step': 1, 'action': 'search', 'parameters': {'action': 'search', 'search_query': 'maintenance'}}, {'step': 2, 'action': 'install', 'parameters': {'action': 'install', 'agent_id': 'asset_maintenance_forecast_agent'}, 'note': 'Use exact agent_id from step 1 results'}]}, 'install_agent': {'description': 'Install agent AFTER getting exact agent_id from search', 'parameters': {'action': 'install', 'agent_id': 'deal_progression_agent'}}, 'get_agent_details': {'description': 'Get detailed information about an agent', 'parameters': {'action': 'get_info', 'agent_id': 'email_agent'}}, 'list_installed': {'description': 'Show all installed GitHub agents', 'parameters': {'action': 'list_installed'}}, 'create_agent_group': {'description': 'Create a GUID-based group of agents for custom deployment', 'parameters': {'action': 'create_group', 'agent_ids': ['deal_progression_agent', 'email_agent', 'sales_forecast_agent'], 'group_name': 'Sales Team Agents'}}, 'list_groups': {'description': 'Show all created GUID-based agent groups', 'parameters': {'action': 'list_groups'}}, 'get_group_details': {'description': 'Get detailed information about a specific agent group', 'parameters': {'action': 'get_group_info', 'guid': '550e8400-e29b-41d4-a716-446655440000'}}}}
        self.storage_manager = AzureFileStorageManager()
        super().__init__(name=self.name, metadata=self.metadata)
        self._manifest_cache = None
        self._manifest_last_fetch = None

    def perform(self, **kwargs):
        action = kwargs.get('action')
        try:
            if action == 'discover':
                return self._discover_agents(kwargs)
            elif action == 'search':
                return self._search_agents(kwargs)
            elif action == 'install':
                return self._install_agent(kwargs)
            elif action == 'list_installed':
                return self._list_installed_agents()
            elif action == 'update':
                return self._update_agent(kwargs)
            elif action == 'remove':
                return self._remove_agent(kwargs)
            elif action == 'get_info':
                return self._get_agent_info(kwargs)
            elif action == 'sync_manifest':
                return self._sync_manifest()
            elif action == 'create_group':
                return self._create_agent_group(kwargs)
            elif action == 'list_groups':
                return self._list_agent_groups()
            elif action == 'get_group_info':
                return self._get_group_info(kwargs)
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            logging.error(f'Error in GitHubAgentLibrary: {str(e)}')
            return f'Error: {str(e)}'

    def _fetch_manifest(self, force_refresh=False):
        """Fetch the manifest.json from GitHub"""
        if not force_refresh and self._manifest_cache and self._manifest_last_fetch:
            if (datetime.now() - self._manifest_last_fetch).seconds < 300:
                return self._manifest_cache
        try:
            manifest_url = f'{self.GITHUB_RAW_BASE}/manifest.json'
            response = requests.get(manifest_url, timeout=10)
            response.raise_for_status()
            manifest = response.json()
            self._manifest_cache = manifest
            self._manifest_last_fetch = datetime.now()
            return manifest
        except Exception as e:
            logging.error(f'Error fetching manifest: {str(e)}')
            return None

    def _discover_agents(self, params):
        """Discover all available agents in the GitHub library"""
        manifest = self._fetch_manifest()
        if not manifest:
            return 'Error: Unable to fetch agent library manifest'
        category = params.get('category')
        singular_agents = manifest.get('agents', [])
        stacks = manifest.get('stacks', [])
        if category:
            category_key = f'{category}_stacks'
            stacks = [s for s in stacks if s.get('path', '').startswith(category_key)]
        total_singular = len(singular_agents)
        total_stack_agents = sum((len(stack.get('agents', [])) for stack in stacks))
        response = f'🔍 GitHub Agent Library Discovery\n\n'
        response += f'**Repository:** {self.GITHUB_REPO}\n'
        response += f'**Total Agents Available:** {total_singular + total_stack_agents}\n'
        response += f'  • Singular Agents: {total_singular}\n'
        response += f'  • Stack Agents: {total_stack_agents}\n\n'
        if singular_agents:
            response += f'## 📦 Singular Agents ({len(singular_agents)})\n\n'
            for i, agent in enumerate(singular_agents[:10], 1):
                response += f'{i}. **{agent['name']}** ({agent['id']})\n'
                response += f'   {agent.get('icon', '🤖')} {agent.get('description', 'No description')[:100]}\n'
                response += f"   Install: `agent_id='{agent['id']}'`\n\n"
            if len(singular_agents) > 10:
                response += f'   ... and {len(singular_agents) - 10} more singular agents\n\n'
        if stacks:
            response += f'## 🏢 Agent Stacks ({len(stacks)} stacks)\n\n'
            for stack in stacks[:5]:
                response += f'### {stack['name']}\n'
                response += f'**Industry:** {stack.get('industry', 'General')}\n'
                response += f'**Path:** {stack.get('path', 'N/A')}\n'
                response += f'**Agents in Stack:** {len(stack.get('agents', []))}\n\n'
                for agent in stack.get('agents', [])[:3]:
                    response += f'  • **{agent['name']}** ({agent['id']})\n'
                    response += f'    {agent.get('description', 'No description')[:80]}\n'
                    response += f"    Install: `agent_id='{agent['id']}', stack_path='{stack.get('path', '')}'`\n\n"
                if len(stack.get('agents', [])) > 3:
                    response += f'    ... and {len(stack.get('agents', [])) - 3} more agents in this stack\n\n'
            if len(stacks) > 5:
                response += f'... and {len(stacks) - 5} more stacks\n\n'
        response += f'\n💡 **Tips:**\n'
        response += f"• Use `action='search', search_query='keyword'` to find specific agents\n"
        response += f"• Use `action='install', agent_id='exact_id'` to install an agent\n"
        response += f"• Use `action='create_group', agent_ids=['id1', 'id2']` to create a GUID-based group\n"
        return response

    def _search_agents(self, params):
        """Search for agents by keyword"""
        search_query = params.get('search_query', '').lower()
        category = params.get('category')
        if not search_query:
            return 'Error: search_query is required for search action'
        manifest = self._fetch_manifest()
        if not manifest:
            return 'Error: Unable to fetch agent library manifest'
        results = []
        for agent in manifest.get('agents', []):
            if self._matches_search(agent, search_query):
                results.append({'agent': agent, 'type': 'singular', 'relevance': self._calculate_relevance(agent, search_query)})
        for stack in manifest.get('stacks', []):
            if category:
                category_key = f'{category}_stacks'
                if not stack.get('path', '').startswith(category_key):
                    continue
            for agent in stack.get('agents', []):
                if self._matches_search(agent, search_query):
                    results.append({'agent': agent, 'type': 'stack', 'stack_name': stack['name'], 'stack_path': stack.get('path', ''), 'stack_industry': stack.get('industry', 'General'), 'relevance': self._calculate_relevance(agent, search_query)})
        results.sort(key=lambda x: x['relevance'], reverse=True)
        if not results:
            response = f"❌ No agents found matching '{search_query}'\n\n"
            response += f'💡 Try:\n'
            response += f'• Using broader search terms\n'
            response += f"• Using `action='discover'` to browse all agents\n"
            response += f'• Checking the repository directly: {self.GITHUB_REPO}\n'
            return response
        response = f"🔍 Search Results for '{search_query}' ({len(results)} found)\n\n"
        for i, result in enumerate(results[:15], 1):
            agent = result['agent']
            response += f'{i}. **{agent['name']}**\n'
            response += f'   • ID: `{agent['id']}`\n'
            response += f'   • Type: {result['type']}\n'
            if result['type'] == 'stack':
                response += f'   • Stack: {result['stack_name']} ({result['stack_industry']})\n'
                response += f'   • Stack Path: `{result['stack_path']}`\n'
            response += f'   • Description: {agent.get('description', 'No description')[:120]}\n'
            response += f'   • Size: {agent.get('size_formatted', 'Unknown')}\n'
            if agent.get('features'):
                response += f'   • Features: {', '.join(agent['features'][:3])}\n'
            response += f'\n   **Install Command:**\n'
            response += f"   `action='install', agent_id='{agent['id']}'"
            if result['type'] == 'stack':
                response += f", stack_path='{result['stack_path']}'"
            response += f'`\n\n'
        if len(results) > 15:
            response += f'... and {len(results) - 15} more results. Refine your search for more specific results.\n'
        return response

    def _matches_search(self, agent, search_query):
        """Check if agent matches search query"""
        searchable_text = f'{agent.get('name', '')} {agent.get('id', '')} {agent.get('description', '')} {' '.join(agent.get('features', []))}'
        return search_query in searchable_text.lower()

    def _calculate_relevance(self, agent, search_query):
        """Calculate relevance score for search results"""
        score = 0
        if search_query in agent.get('name', '').lower():
            score += 10
        if search_query in agent.get('id', '').lower():
            score += 8
        if search_query in agent.get('description', '').lower():
            score += 5
        for feature in agent.get('features', []):
            if search_query in feature.lower():
                score += 3
        return score

    def _install_agent(self, params):
        """Install an agent from GitHub"""
        agent_id = params.get('agent_id')
        stack_path = params.get('stack_path')
        force = params.get('force', False)
        if not agent_id:
            return 'Error: agent_id is required'
        manifest = self._fetch_manifest()
        if not manifest:
            return 'Error: Unable to fetch agent library manifest'
        agent_info = None
        source_type = 'singular'
        for agent in manifest.get('agents', []):
            if agent['id'] == agent_id:
                agent_info = agent
                break
        if not agent_info:
            for stack in manifest.get('stacks', []):
                for agent in stack.get('agents', []):
                    if agent['id'] == agent_id:
                        agent_info = agent
                        source_type = 'stack'
                        agent_info['stack_info'] = {'name': stack['name'], 'path': stack.get('path', ''), 'industry': stack.get('industry', 'General')}
                        break
                if agent_info:
                    break
        if not agent_info:
            search_term = agent_id.replace('_agent', '').replace('_', ' ')
            return f"Error: Agent '{agent_id}' not found in GitHub library.\n\n❌ The agent_id you provided doesn't exist in the repository.\n\n💡 **What to do:**\n1. Use `action='search', search_query='{search_term}'` to find the correct agent_id\n2. Use `action='discover'` to browse all available agents\n3. Make sure you're using the exact agent_id from search results\n\n⚠️ **Important:** Never guess or make up agent IDs. Always get them from search/discover results first."
        if not force:
            log_data = self.storage_manager.read_file('agent_catalogue', 'installation_log.json')
            if log_data:
                installations = json.loads(log_data)
                if any((a['agent_id'] == agent_id for a in installations.get('installations', []))):
                    return f"⚠️ Agent '{agent_info['name']}' is already installed.\n\n**Options:**\n1. Use `action='update', agent_id='{agent_id}'` to reinstall/update\n2. Use `force=True` to force reinstall\n3. Use `action='list_installed'` to see all installed agents"
        try:
            response = requests.get(agent_info['url'], timeout=10)
            response.raise_for_status()
            agent_code = response.text
        except Exception as e:
            logging.error(f'Error fetching agent {agent_id}: {str(e)}')
            return f'Error: Failed to download agent from GitHub: {str(e)}'
        try:
            success = self.storage_manager.write_file('agents', agent_info['filename'], agent_code)
            if not success:
                return 'Error: Failed to write agent to Azure storage'
        except Exception as e:
            logging.error(f'Error storing agent {agent_id}: {str(e)}')
            return f'Error: Failed to save agent to storage: {str(e)}'
        try:
            log_data = self.storage_manager.read_file('agent_catalogue', 'installation_log.json')
            if log_data:
                installations = json.loads(log_data)
            else:
                installations = {'installations': []}
            installations['installations'] = [a for a in installations['installations'] if a['agent_id'] != agent_id]
            installation_record = {'agent_id': agent_id, 'agent_name': agent_info['name'], 'filename': agent_info['filename'], 'installed_at': datetime.now().isoformat(), 'source': 'github_library', 'type': source_type, 'size': agent_info.get('size_formatted', 'Unknown'), 'github_url': agent_info['url']}
            if source_type == 'stack' and agent_info.get('stack_info'):
                installation_record['stack'] = agent_info['stack_info']
            installations['installations'].append(installation_record)
            self.storage_manager.write_file('agent_catalogue', 'installation_log.json', json.dumps(installations, indent=2))
        except Exception as e:
            logging.error(f'Error updating installation log: {str(e)}')
        response = f'✅ Successfully installed: **{agent_info['name']}**\n\n'
        response += f'**Details:**\n'
        response += f'• ID: {agent_id}\n'
        response += f'• Filename: {agent_info['filename']}\n'
        response += f'• Type: {source_type}\n'
        response += f'• Size: {agent_info.get('size_formatted', 'Unknown')}\n'
        if source_type == 'stack' and agent_info.get('stack_info'):
            response += f'• Stack: {agent_info['stack_info']['name']}\n'
            response += f'• Industry: {agent_info['stack_info']['industry']}\n'
        response += f'\n**Features:**\n'
        for feature in agent_info.get('features', [])[:5]:
            response += f'• {feature}\n'
        response += f'\n**Status:**\n'
        response += f'• Downloaded from GitHub: ✅\n'
        response += f'• Saved to Azure storage: ✅\n'
        response += f'• Installation logged: ✅\n'
        response += f'• Ready to use: ✅\n'
        return response

    def _list_installed_agents(self):
        """List all installed GitHub agents"""
        try:
            log_data = self.storage_manager.read_file('agent_catalogue', 'installation_log.json')
            if not log_data:
                return 'No agents have been installed from the GitHub library yet.'
            installations = json.loads(log_data)
            installed_agents = installations.get('installations', [])
            if not installed_agents:
                return 'No agents have been installed from the GitHub library yet.'
            response = f'📦 Installed GitHub Library Agents ({len(installed_agents)}):\n\n'
            for i, agent in enumerate(installed_agents, 1):
                response += f'{i}. **{agent['agent_name']}**\n'
                response += f'   • ID: {agent['agent_id']}\n'
                response += f'   • Filename: {agent['filename']}\n'
                response += f'   • Type: {agent.get('type', 'singular')}\n'
                response += f'   • Installed: {agent['installed_at']}\n'
                response += f'   • Size: {agent.get('size', 'Unknown')}\n'
                if agent.get('stack'):
                    response += f'   • Stack: {agent['stack']['name']}\n'
                response += '\n'
            response += f'\n**Management Commands:**\n'
            response += f"• Update: `action='update', agent_id='agent_id'`\n"
            response += f"• Remove: `action='remove', agent_id='agent_id'`\n"
            response += f"• Details: `action='get_info', agent_id='agent_id'`\n"
            return response
        except Exception as e:
            logging.error(f'Error listing installed agents: {str(e)}')
            return f'Error: {str(e)}'

    def _update_agent(self, params):
        """Update an installed agent to the latest version"""
        agent_id = params.get('agent_id')
        if not agent_id:
            return 'Error: agent_id is required'
        params['force'] = True
        return self._install_agent(params)

    def _remove_agent(self, params):
        """Remove an installed agent"""
        agent_id = params.get('agent_id')
        if not agent_id:
            return 'Error: agent_id is required'
        try:
            log_data = self.storage_manager.read_file('agent_catalogue', 'installation_log.json')
            if not log_data:
                return f"Error: Agent '{agent_id}' not found in installation log"
            installations = json.loads(log_data)
            agent_entry = next((a for a in installations['installations'] if a['agent_id'] == agent_id), None)
            if not agent_entry:
                return f"Error: Agent '{agent_id}' not found in installation log"
            filename = agent_entry['filename']
            installations['installations'] = [a for a in installations['installations'] if a['agent_id'] != agent_id]
            self.storage_manager.write_file('agent_catalogue', 'installation_log.json', json.dumps(installations, indent=2))
            return f"✅ Agent '{agent_entry['agent_name']}' has been removed from the installation log.\n\nNote: The file may still exist in storage until manually deleted."
        except Exception as e:
            logging.error(f'Error removing agent: {str(e)}')
            return f'Error: {str(e)}'

    def _get_agent_info(self, params):
        """Get detailed information about an agent"""
        agent_id = params.get('agent_id')
        if not agent_id:
            return 'Error: agent_id is required'
        manifest = self._fetch_manifest()
        if not manifest:
            return 'Error: Unable to fetch agent library manifest'
        agent_info = None
        for agent in manifest.get('agents', []):
            if agent['id'] == agent_id:
                agent_info = agent
                break
        if not agent_info:
            for stack in manifest.get('stacks', []):
                for agent in stack.get('agents', []):
                    if agent['id'] == agent_id:
                        agent_info = agent
                        agent_info['stack_info'] = {'name': stack['name'], 'industry': stack.get('industry', 'General'), 'path': stack.get('path', '')}
                        break
                if agent_info:
                    break
        if not agent_info:
            search_term = agent_id.replace('_agent', '').replace('_', ' ')
            return f"Error: Agent '{agent_id}' not found in library.\n\n💡 Try searching to find the correct agent_id:\n   action='search', search_query='{search_term}'\n\nThe search will show available agents and their exact IDs."
        response = f'📋 Agent Information: {agent_info['name']}\n\n'
        response += f'**Basic Info:**\n'
        response += f'• ID: {agent_info['id']}\n'
        response += f'• Filename: {agent_info['filename']}\n'
        response += f'• Type: {agent_info.get('type', 'singular')}\n'
        response += f'• Size: {agent_info.get('size_formatted', 'Unknown')}\n'
        response += f'• Icon: {agent_info.get('icon', '🤖')}\n\n'
        response += f'**Description:**\n{agent_info.get('description', 'No description available')}\n\n'
        if agent_info.get('features'):
            response += f'**Features:**\n'
            for feature in agent_info['features']:
                response += f'• {feature}\n'
            response += '\n'
        if agent_info.get('stack_info'):
            response += f'**Stack Information:**\n'
            response += f'• Stack: {agent_info['stack_info']['name']}\n'
            response += f'• Industry: {agent_info['stack_info']['industry']}\n'
            response += f'• Path: {agent_info['stack_info']['path']}\n\n'
        response += f'**Installation:**\n'
        response += f"To install: `action='install', agent_id='{agent_id}'"
        if agent_info.get('stack_info'):
            response += f", stack_path='{agent_info['stack_info']['path']}'"
        response += '`\n'
        return response

    def _sync_manifest(self):
        """Force sync/refresh the manifest from GitHub"""
        manifest = self._fetch_manifest(force_refresh=True)
        if not manifest:
            return 'Error: Unable to sync manifest from GitHub'
        return f'✅ Manifest synced successfully\n\n**Library Stats:**\n• Singular Agents: {len(manifest.get('agents', []))}\n• Agent Stacks: {len(manifest.get('stacks', []))}\n• Last Generated: {manifest.get('generated', 'Unknown')}\n• Repository: {self.GITHUB_REPO}\n\nThe local cache has been refreshed with the latest agent library data.'

    def _create_agent_group(self, params):
        """
        Create a GUID-based agent group by downloading specific agents from GitHub.
        This allows creating custom agent deployments with a unique GUID.
        """
        agent_ids = params.get('agent_ids', [])
        group_name = params.get('group_name', 'Unnamed Agent Group')
        if not agent_ids or not isinstance(agent_ids, list):
            return 'Error: agent_ids is required and must be a list of agent IDs'
        if len(agent_ids) == 0:
            return 'Error: agent_ids list cannot be empty'
        try:
            manifest = self._fetch_manifest()
            if not manifest:
                return 'Error: Unable to fetch agent library manifest from GitHub'
            downloaded_agents = []
            errors = []
            for agent_id in agent_ids:
                result = self._download_agent_for_group(agent_id, manifest)
                if result['success']:
                    downloaded_agents.append(result['filename'])
                else:
                    errors.append(f'❌ {agent_id}: {result['error']}')
            if not downloaded_agents:
                error_msg = 'Error: No agents were successfully downloaded\n\n'
                error_msg += '\n'.join(errors)
                error_msg += "\n\n💡 Use `action='search', search_query='keyword'` to find valid agent IDs"
                return error_msg
            new_guid = str(uuid.uuid4())
            config_result = self._create_agent_config(new_guid, downloaded_agents, group_name, agent_ids)
            if not config_result:
                return 'Error: Failed to create agent configuration'
            response = f'✅ Successfully created agent group!\n\n'
            response += f'**Group Details:**\n'
            response += f'• Name: {group_name}\n'
            response += f'• GUID: `{new_guid}`\n'
            response += f'• Agents Downloaded: {len(downloaded_agents)}\n'
            response += f'• Total Requested: {len(agent_ids)}\n\n'
            response += f'**Downloaded Agents:**\n'
            for filename in downloaded_agents:
                response += f'• {filename}\n'
            if errors:
                response += f'\n**Warnings:**\n'
                response += '\n'.join(errors)
            response += f'\n\n**How to Use This Group:**\n'
            response += f"1. Include this GUID in your API requests: `user_guid: '{new_guid}'`\n"
            response += f'2. Only the agents in this group will be loaded from Azure storage\n'
            response += f'3. All local agents will still be available\n'
            response += f"4. Use `action='get_group_info', guid='{new_guid}'` to view group details later\n\n"
            response += f'💡 This GUID is now stored in Azure storage at: `agent_config/{new_guid}/`\n'
            return response
        except Exception as e:
            logging.error(f'Error in create_agent_group: {str(e)}')
            return f'Error: {str(e)}'

    def _download_agent_for_group(self, agent_id, manifest):
        """Download a single agent from GitHub for a group"""
        agent_info = None
        for agent in manifest.get('agents', []):
            if agent['id'] == agent_id:
                agent_info = agent
                break
        if not agent_info:
            for stack in manifest.get('stacks', []):
                for agent in stack.get('agents', []):
                    if agent['id'] == agent_id:
                        agent_info = agent
                        break
                if agent_info:
                    break
        if not agent_info:
            return {'success': False, 'error': f"Agent ID '{agent_id}' not found in GitHub library"}
        try:
            response = requests.get(agent_info['url'], timeout=10)
            response.raise_for_status()
            agent_code = response.text
        except Exception as e:
            logging.error(f'Error fetching agent {agent_id}: {str(e)}')
            return {'success': False, 'error': f'Failed to download from GitHub: {str(e)}'}
        try:
            success = self.storage_manager.write_file('agents', agent_info['filename'], agent_code)
            if not success:
                return {'success': False, 'error': 'Failed to write to Azure storage'}
            return {'success': True, 'filename': agent_info['filename'], 'agent_info': agent_info}
        except Exception as e:
            logging.error(f'Error storing agent {agent_id}: {str(e)}')
            return {'success': False, 'error': f'Failed to save to storage: {str(e)}'}

    def _create_agent_config(self, guid, agent_filenames, group_name, agent_ids):
        """Create the agent configuration file for the GUID"""
        try:
            config_path = f'agent_config/{guid}'
            enabled_agents_json = json.dumps(agent_filenames, indent=2)
            metadata = {'guid': guid, 'group_name': group_name, 'created_at': datetime.now().isoformat(), 'agent_ids': agent_ids, 'agent_filenames': agent_filenames, 'agent_count': len(agent_filenames), 'source': 'github_library'}
            metadata_json = json.dumps(metadata, indent=2)
            success1 = self.storage_manager.write_file(config_path, 'enabled_agents.json', enabled_agents_json)
            success2 = self.storage_manager.write_file(config_path, 'metadata.json', metadata_json)
            return success1 and success2
        except Exception as e:
            logging.error(f'Error creating agent config: {str(e)}')
            return False

    def _list_agent_groups(self):
        """List all GUID-based agent groups"""
        try:
            response = f'📦 GUID-Based Agent Groups\n\n'
            response += f"**Note:** To view a specific group's details, use:\n"
            response += f"`action='get_group_info', guid='your-guid-here'`\n\n"
            response += f'**How Groups Work:**\n'
            response += f'• Each group has a unique GUID that loads specific agents\n'
            response += f'• Groups are stored in Azure at: `agent_config/<guid>/`\n'
            response += f'• Include the GUID in API requests to use that group\n\n'
            response += f'**Available Actions:**\n'
            response += f"• Create: `action='create_group', agent_ids=['id1', 'id2'], group_name='Name'`\n"
            response += f"• View: `action='get_group_info', guid='guid-value'`\n"
            return response
        except Exception as e:
            logging.error(f'Error listing agent groups: {str(e)}')
            return f'Error: {str(e)}'

    def _get_group_info(self, params):
        """Get detailed information about a GUID-based agent group"""
        guid = params.get('guid')
        if not guid:
            return 'Error: guid parameter is required'
        try:
            config_path = f'agent_config/{guid}'
            metadata_json = self.storage_manager.read_file(config_path, 'metadata.json')
            if not metadata_json:
                return f"Error: Agent group with GUID '{guid}' not found"
            metadata = json.loads(metadata_json)
            enabled_agents_json = self.storage_manager.read_file(config_path, 'enabled_agents.json')
            enabled_agents = json.loads(enabled_agents_json) if enabled_agents_json else []
            response = f'📋 Agent Group Details\n\n'
            response += f'**Group Information:**\n'
            response += f'• Name: {metadata.get('group_name', 'Unnamed')}\n'
            response += f'• GUID: `{metadata.get('guid', guid)}`\n'
            response += f'• Created: {metadata.get('created_at', 'Unknown')}\n'
            response += f'• Agent Count: {metadata.get('agent_count', len(enabled_agents))}\n'
            response += f'• Source: {metadata.get('source', 'Unknown')}\n\n'
            response += f'**Agent IDs:**\n'
            for agent_id in metadata.get('agent_ids', []):
                response += f'• {agent_id}\n'
            response += '\n'
            response += f'**Agent Files:**\n'
            for filename in metadata.get('agent_filenames', enabled_agents):
                response += f'• {filename}\n'
            response += '\n'
            response += f'**Usage:**\n'
            response += f'Include this GUID in your API requests:\n'
            response += f"`user_guid: '{guid}'`\n\n"
            response += f'**Storage Location:**\n'
            response += f'`agent_config/{guid}/`\n'
            return response
        except Exception as e:
            logging.error(f'Error getting group info: {str(e)}')
            return f'Error: {str(e)}'

class ManageMemoryAgent(BasicAgent):

    def __init__(self):
        self.name = 'ManageMemory'
        self.metadata = {'name': self.name, 'description': 'Manages memories in the conversation system. This agent allows me to save important information to our memory system for future reference.', 'parameters': {'type': 'object', 'properties': {'memory_type': {'type': 'string', 'description': "Type of memory to store. Can be 'fact', 'preference', 'insight', or 'task'.", 'enum': ['fact', 'preference', 'insight', 'task']}, 'content': {'type': 'string', 'description': 'The content to store in memory. This should be a concise statement that captures the important information.'}, 'importance': {'type': 'integer', 'description': 'Importance rating from 1-5, where 5 is most important.', 'minimum': 1, 'maximum': 5}, 'tags': {'type': 'array', 'items': {'type': 'string'}, 'description': 'Optional list of tags to categorize this memory.'}, 'user_guid': {'type': 'string', 'description': 'Optional unique identifier of the user to store memory in a user-specific location.'}}, 'required': ['memory_type', 'content']}}
        self.storage_manager = AzureFileStorageManager()
        super().__init__(name=self.name, metadata=self.metadata)

    def perform(self, **kwargs):
        memory_type = kwargs.get('memory_type', 'fact')
        content = kwargs.get('content', '')
        importance = kwargs.get('importance', 3)
        tags = kwargs.get('tags', [])
        user_guid = kwargs.get('user_guid')
        if not content:
            return 'Error: No content provided for memory storage.'
        self.storage_manager.set_memory_context(user_guid)
        return self.store_memory(memory_type, content, importance, tags)

    def store_memory(self, memory_type, content, importance, tags):
        """Store a memory with consistent data structure"""
        memory_data = self.storage_manager.read_json()
        if not memory_data:
            memory_data = {}
        memory_id = str(uuid.uuid4())
        memory_data[memory_id] = {'conversation_id': self.storage_manager.current_guid or 'current', 'session_id': 'current', 'message': content, 'mood': 'neutral', 'theme': memory_type, 'date': datetime.now().strftime('%Y-%m-%d'), 'time': datetime.now().strftime('%H:%M:%S')}
        self.storage_manager.write_json(memory_data)
        memory_location = f'for user {self.storage_manager.current_guid}' if self.storage_manager.current_guid else 'in shared memory'
        return f'Successfully stored {memory_type} memory {memory_location}: "{content}"'

    def retrieve_memories_by_tags(self, tags, user_guid=None):
        """Retrieve memories that match specific tags"""
        if user_guid:
            self.storage_manager.set_memory_context(user_guid)
        memory_data = self.storage_manager.read_json()
        if not memory_data:
            return f'No memories found for this session.'
        legacy_matches = []
        for key, value in memory_data.items():
            if isinstance(value, dict) and 'theme' in value and ('message' in value):
                theme = str(value.get('theme', '')).lower()
                if any((tag.lower() in theme for tag in tags)):
                    legacy_matches.append(value)
        if legacy_matches:
            results = []
            for memory in legacy_matches:
                results.append(f'• {memory['message']} (Theme: {memory['theme']})')
            return f'Found {len(legacy_matches)} memories matching tags {', '.join(tags)}:\n' + '\n'.join(results)
        return f'No memories found matching tags: {', '.join(tags)}'

    def retrieve_memories_by_importance(self, min_importance=4, max_importance=5, user_guid=None):
        """Retrieve memories within a specified importance range"""
        if user_guid:
            self.storage_manager.set_memory_context(user_guid)
        memory_data = self.storage_manager.read_json()
        if not memory_data:
            return 'No important memories found for this session.'
        legacy_memories = []
        for key, value in memory_data.items():
            if isinstance(value, dict) and 'message' in value and ('theme' in value):
                legacy_memories.append(value)
        if legacy_memories:
            try:
                legacy_memories.sort(key=lambda x: (x.get('date', ''), x.get('time', '')), reverse=True)
            except:
                pass
            results = []
            for memory in legacy_memories[:5]:
                date_str = f', Date: {memory.get('date', 'Unknown')}' if memory.get('date') else ''
                results.append(f'• {memory['message']} (Theme: {memory['theme']}{date_str})')
            return f'Most recent memories:\n' + '\n'.join(results)
        return f'No memories found.'

    def retrieve_recent_memories(self, limit=5, user_guid=None):
        """Retrieve the most recently created memories"""
        if user_guid:
            self.storage_manager.set_memory_context(user_guid)
        memory_data = self.storage_manager.read_json()
        has_memories = any((isinstance(key, str) and isinstance(memory_data[key], dict) for key in memory_data.keys() if memory_data.get(key)))
        if not has_memories:
            return 'No recent memories found for this session.'
        legacy_memories = []
        for key, value in memory_data.items():
            if isinstance(value, dict) and 'date' in value and ('time' in value) and ('message' in value):
                legacy_memories.append(value)
        legacy_memories.sort(key=lambda x: (x.get('date', ''), x.get('time', '')), reverse=True)
        recent_legacy = legacy_memories[:limit]
        results = []
        for memory in recent_legacy:
            results.append(f'• {memory['message']} (Theme: {memory['theme']}, Date: {memory['date']})')
        if not results:
            return 'No recent memories found.'
        return f'Recent memories:\n' + '\n'.join(results)

    def retrieve_all_memories(self, user_guid=None):
        """Retrieve all memories"""
        if user_guid:
            self.storage_manager.set_memory_context(user_guid)
        memory_data = self.storage_manager.read_json()
        has_memories = len(memory_data) > 0
        if not has_memories:
            return 'No memories found for this session.'
        legacy_memories = []
        for key, value in memory_data.items():
            if isinstance(value, dict) and 'message' in value and ('theme' in value):
                legacy_memories.append(value)
        if legacy_memories:
            try:
                legacy_memories.sort(key=lambda x: (x.get('date', ''), x.get('time', '')), reverse=True)
            except:
                pass
            results = []
            for memory in legacy_memories:
                date_str = f', Date: {memory.get('date', 'Unknown')}' if memory.get('date') else ''
                results.append(f'• {memory['message']} (Theme: {memory['theme']}{date_str})')
        if not legacy_memories:
            return 'No memories found for this session.'
        total_count = len(legacy_memories)
        return f'All memories ({total_count}):\n' + '\n'.join(results)

class ContextMemoryAgent(BasicAgent):

    def __init__(self):
        self.name = 'ContextMemory'
        self.metadata = {'name': self.name, 'description': 'Recalls and provides context based on stored memories of the past interactions with the user.', 'parameters': {'type': 'object', 'properties': {'user_guid': {'type': 'string', 'description': 'Optional unique identifier of the user to recall memories from a user-specific location.'}, 'max_messages': {'type': 'integer', 'description': 'Optional maximum number of messages to include in the context. Default is 10.'}, 'keywords': {'type': 'array', 'items': {'type': 'string'}, 'description': 'Optional list of keywords to filter memories by. Only messages containing these keywords will be included.'}, 'full_recall': {'type': 'boolean', 'description': 'Optional flag to return all memories without filtering. Default is false.'}}, 'required': []}}
        self.storage_manager = AzureFileStorageManager()
        super().__init__(name=self.name, metadata=self.metadata)

    def perform(self, **kwargs):
        user_guid = kwargs.get('user_guid')
        max_messages = kwargs.get('max_messages', 10)
        keywords = kwargs.get('keywords', [])
        full_recall = kwargs.get('full_recall', False)
        if 'max_messages' not in kwargs and 'keywords' not in kwargs:
            full_recall = True
        self.storage_manager.set_memory_context(user_guid)
        return self._recall_context(max_messages, keywords, full_recall)

    def _recall_context(self, max_messages, keywords, full_recall=False):
        memory_data = self.storage_manager.read_json()
        if not memory_data:
            if self.storage_manager.current_guid:
                return f"I don't have any memories stored yet for user ID {self.storage_manager.current_guid}."
            else:
                return "I don't have any memories stored in the shared memory yet."
        legacy_memories = []
        for key, value in memory_data.items():
            if isinstance(value, dict) and 'message' in value:
                legacy_memories.append(value)
        if not legacy_memories:
            return 'No memories found for this session.'
        return self._format_legacy_memories(legacy_memories, max_messages, keywords, full_recall)

    def _format_legacy_memories(self, memories, max_messages, keywords, full_recall=False):
        """Format memories from legacy storage format (UUIDs as keys)"""
        if not memories:
            return 'No memories found in the format I understand.'
        if full_recall:
            sorted_memories = sorted(memories, key=lambda x: (x.get('date', ''), x.get('time', '')), reverse=True)
            memory_lines = []
            for memory in sorted_memories:
                message = memory.get('message', '')
                theme = memory.get('theme', 'Unknown')
                date = memory.get('date', '')
                time = memory.get('time', '')
                if date and time:
                    memory_lines.append(f'• {message} (Theme: {theme}, Recorded: {date} {time})')
                else:
                    memory_lines.append(f'• {message} (Theme: {theme})')
            if not memory_lines:
                return 'No memories found.'
            memory_source = f'for user ID {self.storage_manager.current_guid}' if self.storage_manager.current_guid else 'from shared memory'
            return f'All memories {memory_source}:\n' + '\n'.join(memory_lines)
        if keywords and len(keywords) > 0:
            filtered_memories = []
            for memory in memories:
                content = str(memory.get('message', '')).lower()
                theme = str(memory.get('theme', '')).lower()
                if any((keyword.lower() in content for keyword in keywords)) or any((keyword.lower() in theme for keyword in keywords)):
                    filtered_memories.append(memory)
            if filtered_memories:
                memories = filtered_memories
            else:
                memories = sorted(memories, key=lambda x: (x.get('date', ''), x.get('time', '')), reverse=True)[:max_messages]
        else:
            memories = sorted(memories, key=lambda x: (x.get('date', ''), x.get('time', '')), reverse=True)[:max_messages]
        memory_lines = []
        for memory in memories:
            message = memory.get('message', '')
            theme = memory.get('theme', 'Unknown')
            date = memory.get('date', '')
            time = memory.get('time', '')
            if date and time:
                memory_lines.append(f'• {message} (Theme: {theme}, Recorded: {date} {time})')
            else:
                memory_lines.append(f'• {message} (Theme: {theme})')
        if not memory_lines:
            return 'No matching memories found.'
        memory_source = f'for user ID {self.storage_manager.current_guid}' if self.storage_manager.current_guid else 'from shared memory'
        return f"Here's what I remember {memory_source}:\n" + '\n'.join(memory_lines)

    def _summarize_memory_item(self, item):
        """Helper to summarize various memory item formats"""
        if isinstance(item, dict):
            if all((key in item for key in ['date', 'time', 'theme', 'message'])):
                return f"On {item['date']} at {item['time']}, a memory was stored with the theme '{item['theme']}' and message '{item['message']}'."
        return None

