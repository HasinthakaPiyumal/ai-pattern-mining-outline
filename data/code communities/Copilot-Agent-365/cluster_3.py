# Cluster 3

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

def ensure_string_content(message):
    """
    Ensures message content is converted to a string regardless of input type.
    Handles all edge cases including None, undefined, or missing content.
    """
    if message is None:
        return {'role': 'user', 'content': ''}
    if not isinstance(message, dict):
        return {'role': 'user', 'content': str(message) if message is not None else ''}
    message = message.copy()
    if 'role' not in message:
        message['role'] = 'user'
    if 'content' in message:
        content = message['content']
        message['content'] = str(content) if content is not None else ''
    else:
        message['content'] = ''
    return message

def ensure_string_function_args(function_call):
    """
    Ensures function call arguments are properly stringified.
    Handles None and edge cases.
    """
    if not function_call:
        return None
    if not hasattr(function_call, 'arguments'):
        return None
    if function_call.arguments is None:
        return None
    if isinstance(function_call.arguments, (dict, list)):
        return json.dumps(function_call.arguments)
    return str(function_call.arguments)

