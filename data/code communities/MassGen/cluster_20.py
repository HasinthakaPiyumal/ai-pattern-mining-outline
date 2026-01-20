# Cluster 20

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

def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {str: 'string', int: 'integer', float: 'number', bool: 'boolean', list: 'array', dict: 'object', type(None): 'null'}
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(f'Failed to get signature for function {func.__name__}: {str(e)}')
    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, 'string')
        except KeyError as e:
            raise KeyError(f'Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}')
        parameters[param.name] = {'type': param_type}
    required = [param.name for param in signature.parameters.values() if param.default == inspect._empty]
    return {'type': 'function', 'name': func.__name__, 'description': func.__doc__ or '', 'parameters': {'type': 'object', 'properties': parameters, 'required': required}}

def parse_completion(completion, add_citations=True):
    """Parse the completion response from Gemini API using the official SDK."""
    text = ''
    code = []
    citations = []
    function_calls = []
    if hasattr(completion, 'candidates') and completion.candidates:
        candidate = completion.candidates[0]
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    text += part.text
                elif hasattr(part, 'executable_code') and part.executable_code:
                    if hasattr(part.executable_code, 'code') and part.executable_code.code:
                        code.append(part.executable_code.code)
                    elif hasattr(part.executable_code, 'language') and hasattr(part.executable_code, 'code'):
                        code.append(part.executable_code.code)
                elif hasattr(part, 'code_execution_result') and part.code_execution_result:
                    if hasattr(part.code_execution_result, 'output') and part.code_execution_result.output:
                        text += f'\n[Code Output]\n{part.code_execution_result.output}\n'
                elif hasattr(part, 'function_call'):
                    if part.function_call:
                        func_name = getattr(part.function_call, 'name', 'unknown')
                        func_args = {}
                        call_id = getattr(part.function_call, 'id', generate_random_id())
                        if hasattr(part.function_call, 'args') and part.function_call.args:
                            if hasattr(part.function_call.args, '_pb'):
                                try:
                                    func_args = dict(part.function_call.args)
                                except Exception:
                                    func_args = {}
                            else:
                                func_args = part.function_call.args
                        function_calls.append({'type': 'function_call', 'call_id': call_id, 'name': func_name, 'arguments': func_args})
                elif hasattr(part, 'function_response'):
                    pass
    if hasattr(completion, 'candidates') and completion.candidates:
        candidate = completion.candidates[0]
        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
            grounding = candidate.grounding_metadata
            if hasattr(grounding, 'grounding_chunks') and grounding.grounding_chunks:
                for chunk in grounding.grounding_chunks:
                    if hasattr(chunk, 'web') and chunk.web:
                        web_chunk = chunk.web
                        citation = {'url': getattr(web_chunk, 'uri', ''), 'title': getattr(web_chunk, 'title', ''), 'start_index': -1, 'end_index': -1}
                        citations.append(citation)
            if hasattr(grounding, 'search_entry_point') and grounding.search_entry_point:
                entry_point = grounding.search_entry_point
                if hasattr(entry_point, 'rendered_content') and entry_point.rendered_content:
                    pass
    if add_citations:
        try:
            text = add_citations_to_response(completion)
        except Exception as e:
            print(f'[GEMINI] Error adding citations to text: {e}')
    return AgentResponse(text=text, code=code, citations=citations, function_calls=function_calls)

def generate_random_id(length: int=24) -> str:
    """Generate a random ID string."""
    characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    return ''.join((random.choice(characters) for _ in range(length)))

def add_citations_to_response(response):
    text = response.text
    if not hasattr(response, 'candidates') or not response.candidates:
        return text
    candidate = response.candidates[0]
    if not hasattr(candidate, 'grounding_metadata') or not candidate.grounding_metadata:
        return text
    grounding_metadata = candidate.grounding_metadata
    supports = getattr(grounding_metadata, 'grounding_supports', None)
    chunks = getattr(grounding_metadata, 'grounding_chunks', None)
    if not supports or not chunks:
        return text
    sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)
    for support in sorted_supports:
        end_index = support.segment.end_index
        if support.grounding_chunk_indices:
            citation_links = []
            for i in support.grounding_chunk_indices:
                if i < len(chunks):
                    uri = chunks[i].web.uri
                    citation_links.append(f'[{i + 1}]({uri})')
            citation_string = ', '.join(citation_links)
            text = text[:end_index] + citation_string + text[end_index:]
    return text

def process_message(messages, model='gemini-2.5-flash', tools=None, max_retries=10, max_tokens=None, temperature=None, top_p=None, api_key=None, stream=False, stream_callback=None):
    """
    Generate content using Gemini API with the official google.genai SDK.

    Args:
        messages: List of messages in OpenAI format
        model: The Gemini model to use
        tools: List of tools to use
        max_retries: Maximum number of retry attempts
        max_tokens: Maximum number of tokens in response
        temperature: Temperature for generation
        top_p: Top-p value for generation
        api_key: Gemini API key (if None, will get from environment)
        stream: Whether to stream the response (default: False)
        stream_callback: Function to call with each chunk when streaming (default: None)

    Returns:
        dict: {"text": text, "code": code, "citations": citations, "function_calls": function_calls}
    """
    'Internal function that contains all the processing logic.'
    if api_key is None:
        api_key_val = os.getenv('GEMINI_API_KEY')
    else:
        api_key_val = api_key
    if not api_key_val:
        raise ValueError('GEMINI_API_KEY not found in environment variables')
    client = genai.Client(api_key=api_key_val)
    gemini_messages = []
    system_instruction = None
    function_calls = {}
    for message in messages:
        role = message.get('role', None)
        content = message.get('content', None)
        if role == 'system':
            system_instruction = content
        elif role == 'user':
            gemini_messages.append(types.Content(role='user', parts=[types.Part(text=content)]))
        elif role == 'assistant':
            gemini_messages.append(types.Content(role='model', parts=[types.Part(text=content)]))
        elif message.get('type', None) == 'function_call':
            function_calls[message['call_id']] = message
        elif message.get('type', None) == 'function_call_output':
            func_name = function_calls[message['call_id']]['name']
            func_resp = message['output']
            function_response_part = types.Part.from_function_response(name=func_name, response={'result': func_resp})
            gemini_messages.append(types.Content(role='user', parts=[function_response_part]))
    generation_config = {}
    if temperature is not None:
        generation_config['temperature'] = temperature
    if top_p is not None:
        generation_config['top_p'] = top_p
    if max_tokens is not None:
        generation_config['max_output_tokens'] = max_tokens
    gemini_tools = []
    has_native_tools = False
    custom_functions = []
    if tools:
        for tool in tools:
            if 'live_search' == tool:
                gemini_tools.append(types.Tool(google_search=types.GoogleSearch()))
                has_native_tools = True
            elif 'code_execution' == tool:
                gemini_tools.append(types.Tool(code_execution=types.ToolCodeExecution()))
                has_native_tools = True
            else:
                if hasattr(tool, 'function'):
                    function_declaration = tool['function']
                else:
                    function_declaration = copy.deepcopy(tool)
                    if 'type' in function_declaration:
                        del function_declaration['type']
                custom_functions.append(function_declaration)
    if custom_functions and has_native_tools:
        print("[WARNING] Gemini API doesn't support combining native tools with custom functions. Prioritizing built-in tools.")
    elif custom_functions and (not has_native_tools):
        gemini_tools.append(types.Tool(function_declarations=custom_functions))
    safety_settings = [types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE), types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE), types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE), types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE)]
    request_params = {'model': model, 'contents': gemini_messages, 'config': types.GenerateContentConfig(safety_settings=safety_settings, **generation_config)}
    if system_instruction:
        request_params['config'].system_instruction = types.Content(parts=[types.Part(text=system_instruction)])
    if gemini_tools:
        request_params['config'].tools = gemini_tools
    completion = None
    retry = 0
    while retry < max_retries:
        try:
            if stream and stream_callback:
                text = ''
                code = []
                citations = []
                function_calls = []
                code_lines_shown = 0
                truncation_message_sent = False
                stream_response = client.models.generate_content_stream(**request_params)
                for chunk in stream_response:
                    chunk_text_processed = False
                    if hasattr(chunk, 'text') and chunk.text:
                        chunk_text = chunk.text
                        text += chunk_text
                        try:
                            stream_callback(chunk_text)
                            chunk_text_processed = True
                        except Exception as e:
                            print(f'Stream callback error: {e}')
                    elif hasattr(chunk, 'candidates') and chunk.candidates:
                        candidate = chunk.candidates[0]
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text and (not chunk_text_processed):
                                    chunk_text = part.text
                                    text += chunk_text
                                    try:
                                        stream_callback(chunk_text)
                                        chunk_text_processed = True
                                    except Exception as e:
                                        print(f'Stream callback error: {e}')
                                elif hasattr(part, 'executable_code') and part.executable_code and hasattr(part.executable_code, 'code') and part.executable_code.code:
                                    code_text = part.executable_code.code
                                    code.append(code_text)
                                    code_lines = code_text.split('\n')
                                    if code_lines_shown == 0:
                                        try:
                                            stream_callback('\n💻 Starting code execution...\n')
                                        except Exception as e:
                                            print(f'Stream callback error: {e}')
                                    for line in code_lines:
                                        if code_lines_shown < 3:
                                            try:
                                                stream_callback(line + '\n')
                                                code_lines_shown += 1
                                            except Exception as e:
                                                print(f'Stream callback error: {e}')
                                        elif code_lines_shown == 3 and (not truncation_message_sent):
                                            try:
                                                stream_callback('\n[CODE_DISPLAY_ONLY]\n💻 ... (full code in log file)\n')
                                                truncation_message_sent = True
                                                code_lines_shown += 1
                                            except Exception as e:
                                                print(f'Stream callback error: {e}')
                                        else:
                                            try:
                                                stream_callback(f'[CODE_LOG_ONLY]{line}\n')
                                            except Exception as e:
                                                print(f'Stream callback error: {e}')
                                elif hasattr(part, 'function_call') and part.function_call:
                                    func_name = getattr(part.function_call, 'name', 'unknown')
                                    func_args = {}
                                    if hasattr(part.function_call, 'args') and part.function_call.args:
                                        if hasattr(part.function_call.args, '_pb'):
                                            try:
                                                func_args = dict(part.function_call.args)
                                            except Exception:
                                                func_args = {}
                                        else:
                                            func_args = part.function_call.args
                                    function_calls.append({'type': 'function_call', 'call_id': part.function_call.id, 'name': func_name, 'arguments': func_args})
                                    try:
                                        stream_callback(f'\n🔧 Calling {func_name}\n')
                                    except Exception as e:
                                        print(f'Stream callback error: {e}')
                                elif hasattr(part, 'function_response'):
                                    try:
                                        stream_callback('\n🔧 Function response received\n')
                                    except Exception as e:
                                        print(f'Stream callback error: {e}')
                                elif hasattr(part, 'code_execution_result') and part.code_execution_result:
                                    if hasattr(part.code_execution_result, 'output') and part.code_execution_result.output:
                                        result_text = f'\n[Code Output]\n{part.code_execution_result.output}\n'
                                        text += result_text
                                        try:
                                            stream_callback(result_text)
                                        except Exception as e:
                                            print(f'Stream callback error: {e}')
                        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                            grounding = candidate.grounding_metadata
                            if hasattr(grounding, 'grounding_chunks') and grounding.grounding_chunks:
                                for chunk_item in grounding.grounding_chunks:
                                    if hasattr(chunk_item, 'web') and chunk_item.web:
                                        web_chunk = chunk_item.web
                                        citation = {'url': getattr(web_chunk, 'uri', ''), 'title': getattr(web_chunk, 'title', ''), 'start_index': -1, 'end_index': -1}
                                        if citation not in citations:
                                            citations.append(citation)
                try:
                    stream_callback('\n✅ Generation finished\n')
                except Exception as e:
                    print(f'Stream callback error: {e}')
                return AgentResponse(text=text, code=code, citations=citations, function_calls=function_calls)
            else:
                completion = client.models.generate_content(**request_params)
            break
        except Exception as e:
            print(f'Error on attempt {retry + 1}: {e}')
            retry += 1
            time.sleep(1.5)
    if completion is None:
        print(f'Failed to get completion after {max_retries} retries, returning empty response')
        return AgentResponse(text='', code=[], citations=[], function_calls=[])
    result = parse_completion(completion, add_citations=True)
    return result

def parse_completion(response, add_citations=True):
    """Parse the completion response from Grok API."""
    text = response.content
    code = []
    citations = []
    function_calls = []
    if hasattr(response, 'citations') and response.citations:
        for citation in response.citations:
            citations.append({'url': citation, 'title': '', 'start_index': -1, 'end_index': -1})
    if citations and add_citations:
        citation_content = []
        for idx, citation in enumerate(citations):
            citation_content.append(f'[{idx}]({citation['url']})')
        text = text + '\n\nReferences:\n' + '\n'.join(citation_content)
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            if hasattr(tool_call, 'function'):
                function_calls.append({'type': 'function_call', 'call_id': tool_call.id, 'name': tool_call.function.name, 'arguments': tool_call.function.arguments})
            elif hasattr(tool_call, 'name') and hasattr(tool_call, 'arguments'):
                function_calls.append({'type': 'function_call', 'call_id': tool_call.id, 'name': tool_call.name, 'arguments': tool_call.arguments})
    return AgentResponse(text=text, code=code, citations=citations, function_calls=function_calls)

def process_message(messages, model='grok-3-mini', tools=None, max_retries=10, max_tokens=None, temperature=None, top_p=None, api_key=None, stream=False, stream_callback=None):
    """
    Generate content using Grok API with optional streaming support and custom tools.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Model name to use (default: "grok-4")
        tools: List of tool definitions for function calling, each tool should be a dict with OpenAI-compatible format:
               [
                   {
                       "type": "function",
                       "function": {
                           "name": "function_name",
                           "description": "Function description",
                           "parameters": {
                               "type": "object",
                               "properties": {
                                   "param1": {"type": "string", "description": "Parameter description"},
                                   "param2": {"type": "number", "description": "Another parameter"}
                               },
                               "required": ["param1"]
                           }
                       }
                   }
               ]
        enable_search: Boolean to enable live search functionality (default: False)
        max_retries: Maximum number of retry attempts (default: 10)
        max_tokens: Maximum tokens in response (default: 32000)
        temperature: Sampling temperature (default: None)
        top_p: Top-p sampling parameter (default: None)
        api_key: XAI API key (default: None, uses environment variable)
        stream: Enable streaming response (default: False)
        stream_callback: Callback function for streaming (default: None)

    Returns:
        Dict with keys: 'text', 'code', 'citations', 'function_calls'

    Note:
        - For backward compatibility, tools=["live_search"] is still supported and will enable search
        - Function calls will be returned in the 'function_calls' key as a list of dicts with 'name' and 'arguments'
        - The 'arguments' field will contain the function arguments as returned by the model
    """
    'Internal function that contains all the processing logic.'
    if api_key is None:
        api_key_val = os.getenv('XAI_API_KEY')
    else:
        api_key_val = api_key
    if not api_key_val:
        raise ValueError('XAI_API_KEY not found in environment variables')
    client = Client(api_key=api_key_val)
    enable_search = False
    custom_tools = []
    if tools and isinstance(tools, list) and (len(tools) > 0):
        for tool in tools:
            if tool == 'live_search':
                enable_search = True
            elif isinstance(tool, str):
                continue
            else:
                custom_tools.append(tool)
    search_parameters = None
    if enable_search:
        search_parameters = SearchParameters(mode='auto', return_citations=True)
    api_tools = None
    if custom_tools and isinstance(custom_tools, list) and (len(custom_tools) > 0):
        api_tools = []
        for custom_tool in custom_tools:
            if isinstance(custom_tool, dict) and custom_tool.get('type') == 'function':
                if 'function' in custom_tool:
                    func_def = custom_tool['function']
                else:
                    func_def = custom_tool
                xai_tool = xai_tool_func(name=func_def['name'], description=func_def['description'], parameters=func_def['parameters'])
                api_tools.append(xai_tool)
            else:
                api_tools.append(custom_tool)

    def make_grok_request(stream=False):
        chat_params = {'model': model, 'search_parameters': search_parameters}
        if temperature is not None:
            chat_params['temperature'] = temperature
        if top_p is not None:
            chat_params['top_p'] = top_p
        if max_tokens is not None:
            chat_params['max_tokens'] = max_tokens
        if api_tools is not None:
            chat_params['tools'] = api_tools
        chat = client.chat.create(**chat_params)
        for message in messages:
            role = message.get('role', None)
            content = message.get('content', None)
            if role == 'system':
                chat.append(system(content))
            elif role == 'user':
                chat.append(user(content))
            elif role == 'assistant':
                chat.append(assistant(content))
            elif message.get('type', None) == 'function_call':
                pass
            elif message.get('type', None) == 'function_call_output':
                content = message.get('output', None)
                chat.append(tool_result(content))
        if stream:
            return chat.stream()
        else:
            return chat.sample()
    completion = None
    retry = 0
    while retry < max_retries:
        try:
            is_streaming = stream and stream_callback is not None
            completion = make_grok_request(stream=is_streaming)
            break
        except Exception as e:
            print(f'Error on attempt {retry + 1}: {e}')
            retry += 1
            import time
            time.sleep(1.5)
    if completion is None:
        print(f'Failed to get completion after {max_retries} retries, returning empty response')
        return AgentResponse(text='', code=[], citations=[], function_calls=[])
    if stream and stream_callback is not None:
        text = ''
        code = []
        citations = []
        function_calls = []
        thinking_count = 0
        has_shown_search_indicator = False
        try:
            has_delta_content = False
            for response, chunk in completion:
                delta_content = None
                if hasattr(chunk, 'choices') and chunk.choices and (len(chunk.choices) > 0):
                    choice = chunk.choices[0]
                    if hasattr(choice, 'content') and choice.content:
                        delta_content = choice.content
                elif hasattr(chunk, 'content') and chunk.content:
                    delta_content = chunk.content
                elif hasattr(chunk, 'text') and chunk.text:
                    delta_content = chunk.text
                if delta_content:
                    has_delta_content = True
                    if delta_content.strip() == 'Thinking...':
                        thinking_count += 1
                        if thinking_count == 3 and (not has_shown_search_indicator) and search_parameters:
                            try:
                                stream_callback('\n🧠 Thinking...\n')
                            except Exception as e:
                                print(f'Stream callback error: {e}')
                            has_shown_search_indicator = True
                        try:
                            stream_callback(delta_content)
                        except Exception as e:
                            print(f'Stream callback error: {e}')
                    else:
                        text += delta_content
                        try:
                            stream_callback(delta_content)
                        except Exception as e:
                            print(f'Stream callback error: {e}')
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    for tool_call in response.tool_calls:
                        if hasattr(tool_call, 'function'):
                            _func_call = {'type': 'function_call', 'call_id': tool_call.id, 'name': tool_call.function.name, 'arguments': tool_call.function.arguments}
                            if _func_call not in function_calls:
                                function_calls.append(_func_call)
                        elif hasattr(tool_call, 'name') and hasattr(tool_call, 'arguments'):
                            _func_call = {'type': 'function_call', 'call_id': tool_call.id, 'name': tool_call.name, 'arguments': tool_call.arguments}
                            if _func_call not in function_calls:
                                function_calls.append(_func_call)
                elif hasattr(response, 'choices') and response.choices:
                    for choice in response.choices:
                        if hasattr(choice, 'message') and hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                            for tool_call in choice.message.tool_calls:
                                if hasattr(tool_call, 'function'):
                                    _func_call = {'type': 'function_call', 'call_id': tool_call.id, 'name': tool_call.function.name, 'arguments': tool_call.function.arguments}
                                    if _func_call not in function_calls:
                                        function_calls.append(_func_call)
                                elif hasattr(tool_call, 'name') and hasattr(tool_call, 'arguments'):
                                    _func_call = {'type': 'function_call', 'call_id': tool_call.id, 'name': tool_call.name, 'arguments': tool_call.arguments}
                                    if _func_call not in function_calls:
                                        function_calls.append(_func_call)
                if hasattr(response, 'citations') and response.citations:
                    citations = []
                    for citation in response.citations:
                        citations.append({'url': citation, 'title': '', 'start_index': -1, 'end_index': -1})
                    if citations and enable_search and (stream_callback is not None):
                        try:
                            stream_callback(f'\n\n🔍 Found {len(citations)} web sources\n')
                        except Exception as e:
                            print(f'Stream callback error: {e}')
            if not has_delta_content:
                if text:
                    stream_callback(text)
                if function_calls:
                    for function_call in function_calls:
                        stream_callback(f'🔧 Calling function: {function_call['name']}\n')
                        stream_callback(f'🔧 Arguments: {json.dumps(function_call['arguments'], indent=4)}\n\n')
        except Exception:
            completion = make_grok_request(stream=False)
            result = parse_completion(completion, add_citations=True)
            return result
        result = AgentResponse(text=text, code=code, citations=citations, function_calls=function_calls)
    else:
        result = parse_completion(completion, add_citations=True)
    return result

def make_grok_request(stream=False):
    chat_params = {'model': model, 'search_parameters': search_parameters}
    if temperature is not None:
        chat_params['temperature'] = temperature
    if top_p is not None:
        chat_params['top_p'] = top_p
    if max_tokens is not None:
        chat_params['max_tokens'] = max_tokens
    if api_tools is not None:
        chat_params['tools'] = api_tools
    chat = client.chat.create(**chat_params)
    for message in messages:
        role = message.get('role', None)
        content = message.get('content', None)
        if role == 'system':
            chat.append(system(content))
        elif role == 'user':
            chat.append(user(content))
        elif role == 'assistant':
            chat.append(assistant(content))
        elif message.get('type', None) == 'function_call':
            pass
        elif message.get('type', None) == 'function_call_output':
            content = message.get('output', None)
            chat.append(tool_result(content))
    if stream:
        return chat.stream()
    else:
        return chat.sample()

def parse_completion(response, add_citations=True):
    """Parse the completion response from OpenAI API.

    Mainly three types of output in the response:
    - reasoning (no summary provided): ResponseReasoningItem(id='rs_6876b0d566d08198ab9f992e1911bd0a02ec808107751c1f', summary=[], type='reasoning', status=None)
    - web_search_call (actions: search, open_page, find_in_page):
      ResponseFunctionWebSearch(id='ws_6876b0e3b83081988d6cddd9770357c402ec808107751c1f',
                                status='completed', type='web_search_call',
                                action={'type': 'search', 'query': 'Economy of China Wikipedia GDP table'})
    - message: response output (including text and citations, optional)
    - code_interpreter_call (code provided): code output
    - function_call: function call, arguments, and name provided
    """
    text = ''
    code = []
    citations = []
    function_calls = []
    reasoning_items = []
    for r in response.output:
        if r.type == 'message':
            for c in r.content:
                text += c.text
                if add_citations and hasattr(c, 'annotations') and c.annotations:
                    for annotation in c.annotations:
                        citations.append({'url': annotation.url, 'title': annotation.title, 'start_index': annotation.start_index, 'end_index': annotation.end_index})
        elif r.type == 'code_interpreter_call':
            code.append(r.code)
        elif r.type == 'web_search_call':
            pass
        elif r.type == 'reasoning':
            reasoning_items.append({'type': 'reasoning', 'id': r.id, 'summary': r.summary})
        elif r.type == 'function_call':
            function_calls.append({'type': 'function_call', 'name': r.name, 'arguments': r.arguments, 'call_id': getattr(r, 'call_id', None), 'id': getattr(r, 'id', None)})
    if add_citations and citations:
        try:
            new_text = text
            sorted_citations = sorted(citations, key=lambda c: c['end_index'], reverse=True)
            for idx, citation in enumerate(sorted_citations):
                end_index = citation['end_index']
                citation_link = f'[{len(citations) - idx}]({citation['url']})'
                new_text = new_text[:end_index] + citation_link + new_text[end_index:]
            text = new_text
        except Exception as e:
            print(f'[OAI] Error adding citations to text: {e}')
    return AgentResponse(text=text, code=code, citations=citations, function_calls=function_calls)

def process_message(messages, model='gpt-4.1-mini', tools=None, max_retries=10, max_tokens=None, temperature=None, top_p=None, api_key=None, stream=False, stream_callback=None):
    """
    Generate content using OpenAI API with optional streaming support.

    Args:
        messages: List of messages in OpenAI format
        model: The OpenAI model to use
        tools: List of tools to use
        max_retries: Maximum number of retry attempts
        max_tokens: Maximum number of tokens in response
        temperature: Temperature for generation
        top_p: Top-p value for generation
        api_key: OpenAI API key (if None, will get from environment)
        stream: Whether to stream the response (default: False)
        stream_callback: Optional callback function for streaming chunks

    Returns:
        dict: {"text": text, "code": code, "citations": citations, "function_calls": function_calls}
    """
    'Internal function that contains all the processing logic.'
    if api_key is None:
        api_key_val = os.getenv('OPENAI_API_KEY')
    else:
        api_key_val = api_key
    if not api_key_val:
        raise ValueError('OPENAI_API_KEY not found in environment variables')
    client = OpenAI(api_key=api_key_val)
    formatted_tools = []
    if tools:
        for tool in tools:
            if isinstance(tool, dict):
                formatted_tools.append(tool)
            elif callable(tool):
                formatted_tools.append(function_to_json(tool))
            elif tool == 'live_search':
                formatted_tools.append({'type': 'web_search_preview'})
            elif tool == 'code_execution':
                formatted_tools.append({'type': 'code_interpreter', 'container': {'type': 'auto'}})
            else:
                raise ValueError(f'Invalid tool type: {type(tool)}')
    input_text = []
    instructions = ''
    for message in messages:
        if message.get('role', '') == 'system':
            instructions = message['content']
        else:
            if message.get('type', '') == 'function_call' and message.get('id', None) is not None:
                del message['id']
            input_text.append(message)
    completion = None
    retry = 0
    while retry < max_retries:
        try:
            model_name = model
            params = {'model': model_name, 'tools': formatted_tools if formatted_tools else None, 'instructions': instructions if instructions else None, 'input': input_text, 'max_output_tokens': max_tokens if max_tokens else None, 'stream': True if stream and stream_callback else False}
            if formatted_tools and any((tool.get('type') == 'code_interpreter' for tool in formatted_tools)):
                params['include'] = ['code_interpreter_call.outputs']
            if temperature is not None and (not model_name.startswith('o')):
                params['temperature'] = temperature
            if top_p is not None and (not model_name.startswith('o')):
                params['top_p'] = top_p
            if model_name.startswith('o'):
                if model_name.endswith('-low'):
                    params['reasoning'] = {'effort': 'low'}
                    model_name = model_name.replace('-low', '')
                elif model_name.endswith('-medium'):
                    params['reasoning'] = {'effort': 'medium'}
                    model_name = model_name.replace('-medium', '')
                elif model_name.endswith('-high'):
                    params['reasoning'] = {'effort': 'high'}
                    model_name = model_name.replace('-high', '')
                else:
                    params['reasoning'] = {'effort': 'low'}
            params['model'] = model_name
            response = client.responses.create(**params)
            completion = response
            break
        except Exception as e:
            print(f'Error on attempt {retry + 1}: {e}')
            retry += 1
            import time
            time.sleep(1.5)
    if completion is None:
        print(f'Failed to get completion after {max_retries} retries, returning empty response')
        return AgentResponse(text='', code=[], citations=[], function_calls=[])
    if stream and stream_callback:
        text = ''
        code = []
        citations = []
        function_calls = []
        code_lines_shown = 0
        current_code_chunk = ''
        truncation_message_sent = False
        current_function_call = None
        current_function_arguments = ''
        for chunk in completion:
            if hasattr(chunk, 'type'):
                if chunk.type == 'response.output_text.delta':
                    if hasattr(chunk, 'delta') and chunk.delta:
                        chunk_text = chunk.delta
                        text += chunk_text
                        try:
                            stream_callback(chunk_text)
                        except Exception as e:
                            print(f'Stream callback error: {e}')
                elif chunk.type == 'response.function_call_output.delta':
                    try:
                        stream_callback(f'\n🔧 {(chunk.delta if hasattr(chunk, 'delta') else 'Function call')}\n')
                    except Exception as e:
                        print(f'Stream callback error: {e}')
                elif chunk.type == 'response.function_call_output.done':
                    try:
                        stream_callback('\n🔧 Function call completed\n')
                    except Exception as e:
                        print(f'Stream callback error: {e}')
                elif chunk.type == 'response.code_interpreter_call.in_progress':
                    code_lines_shown = 0
                    current_code_chunk = ''
                    truncation_message_sent = False
                    try:
                        stream_callback('\n💻 Starting code execution...\n')
                    except Exception as e:
                        print(f'Stream callback error: {e}')
                elif chunk.type == 'response.code_interpreter_call_code.delta':
                    if hasattr(chunk, 'delta') and chunk.delta:
                        try:
                            current_code_chunk += chunk.delta
                            new_lines = chunk.delta.count('\n')
                            if code_lines_shown < 3:
                                stream_callback(chunk.delta)
                                code_lines_shown += new_lines
                                if code_lines_shown >= 3 and (not truncation_message_sent):
                                    stream_callback('\n[CODE_DISPLAY_ONLY]\n💻 ... (full code in log file)\n')
                                    truncation_message_sent = True
                            else:
                                stream_callback(f'[CODE_LOG_ONLY]{chunk.delta}')
                        except Exception as e:
                            print(f'Stream callback error: {e}')
                elif chunk.type == 'response.code_interpreter_call_code.done':
                    if current_code_chunk:
                        code.append(current_code_chunk)
                    try:
                        stream_callback('\n💻 Code writing completed\n')
                    except Exception as e:
                        print(f'Stream callback error: {e}')
                elif chunk.type == 'response.code_interpreter_call_execution.in_progress':
                    try:
                        stream_callback('\n💻 Executing code...\n')
                    except Exception as e:
                        print(f'Stream callback error: {e}')
                elif chunk.type == 'response.code_interpreter_call_execution.done':
                    try:
                        stream_callback('\n💻 Code execution completed\n')
                    except Exception as e:
                        print(f'Stream callback error: {e}')
                elif chunk.type == 'response.output_item.added':
                    if hasattr(chunk, 'item') and chunk.item:
                        if hasattr(chunk.item, 'type') and chunk.item.type == 'web_search_call':
                            try:
                                stream_callback('\n🔍 Starting web search...\n')
                            except Exception as e:
                                print(f'Stream callback error: {e}')
                        elif hasattr(chunk.item, 'type') and chunk.item.type == 'reasoning':
                            try:
                                stream_callback('\n🧠 Reasoning in progress...\n')
                            except Exception as e:
                                print(f'Stream callback error: {e}')
                        elif hasattr(chunk.item, 'type') and chunk.item.type == 'code_interpreter_call':
                            try:
                                stream_callback('\n💻 Code interpreter starting...\n')
                            except Exception as e:
                                print(f'Stream callback error: {e}')
                        elif hasattr(chunk.item, 'type') and chunk.item.type == 'function_call':
                            function_call_data = {'type': 'function_call', 'name': getattr(chunk.item, 'name', None), 'arguments': getattr(chunk.item, 'arguments', None), 'call_id': getattr(chunk.item, 'call_id', None), 'id': getattr(chunk.item, 'id', None)}
                            function_calls.append(function_call_data)
                            current_function_call = function_call_data
                            current_function_arguments = ''
                            function_name = function_call_data.get('name', 'unknown')
                            try:
                                stream_callback(f"\n🔧 Calling function '{function_name}'...\n")
                            except Exception as e:
                                print(f'Stream callback error: {e}')
                elif chunk.type == 'response.output_item.done':
                    if hasattr(chunk, 'item') and chunk.item:
                        if hasattr(chunk.item, 'type') and chunk.item.type == 'web_search_call':
                            if hasattr(chunk.item, 'action') and hasattr(chunk.item.action, 'query'):
                                search_query = chunk.item.action.query
                                if search_query:
                                    try:
                                        stream_callback(f'\n🔍 Completed search for: {search_query}\n')
                                    except Exception as e:
                                        print(f'Stream callback error: {e}')
                        elif hasattr(chunk.item, 'type') and chunk.item.type == 'reasoning':
                            try:
                                stream_callback('\n🧠 Reasoning completed\n')
                            except Exception as e:
                                print(f'Stream callback error: {e}')
                        elif hasattr(chunk.item, 'type') and chunk.item.type == 'code_interpreter_call':
                            if hasattr(chunk.item, 'outputs') and chunk.item.outputs:
                                for output in chunk.item.outputs:
                                    if hasattr(output, 'get') and output.get('type') == 'logs':
                                        logs_content = output.get('logs')
                                        if logs_content:
                                            execution_result = f'\n[Code Execution Output]\n{logs_content}\n'
                                            text += execution_result
                                            try:
                                                stream_callback(execution_result)
                                            except Exception as e:
                                                print(f'Stream callback error: {e}')
                                    elif hasattr(output, 'type') and output.type == 'logs':
                                        if hasattr(output, 'logs') and output.logs:
                                            execution_result = f'\n[Code Execution Output]\n{output.logs}\n'
                                            text += execution_result
                                            try:
                                                stream_callback(execution_result)
                                            except Exception as e:
                                                print(f'Stream callback error: {e}')
                            try:
                                stream_callback('\n💻 Code interpreter completed\n')
                            except Exception as e:
                                print(f'Stream callback error: {e}')
                        elif hasattr(chunk.item, 'type') and chunk.item.type == 'function_call':
                            if hasattr(chunk.item, 'arguments'):
                                for fc in function_calls:
                                    if fc.get('id') == getattr(chunk.item, 'id', None):
                                        fc['arguments'] = chunk.item.arguments
                                        break
                            if current_function_call and current_function_arguments:
                                current_function_call['arguments'] = current_function_arguments
                            current_function_call = None
                            current_function_arguments = ''
                            function_name = getattr(chunk.item, 'name', 'unknown')
                            try:
                                stream_callback(f"\n🔧 Function '{function_name}' completed\n")
                            except Exception as e:
                                print(f'Stream callback error: {e}')
                elif chunk.type == 'response.web_search_call.in_progress':
                    try:
                        stream_callback('\n🔍 Search in progress...\n')
                    except Exception as e:
                        print(f'Stream callback error: {e}')
                elif chunk.type == 'response.web_search_call.searching':
                    try:
                        stream_callback('\n🔍 Searching...\n')
                    except Exception as e:
                        print(f'Stream callback error: {e}')
                elif chunk.type == 'response.web_search_call.completed':
                    try:
                        stream_callback('\n🔍 Search completed\n')
                    except Exception as e:
                        print(f'Stream callback error: {e}')
                elif chunk.type == 'response.output_text.annotation.added':
                    if hasattr(chunk, 'annotation'):
                        citation_data = {'url': getattr(chunk.annotation, 'url', None), 'title': getattr(chunk.annotation, 'title', None), 'start_index': getattr(chunk.annotation, 'start_index', None), 'end_index': getattr(chunk.annotation, 'end_index', None)}
                        citations.append(citation_data)
                    try:
                        stream_callback('\n📚 Citation added\n')
                    except Exception as e:
                        print(f'Stream callback error: {e}')
                elif chunk.type == 'response.function_call_arguments.delta':
                    if hasattr(chunk, 'delta') and chunk.delta:
                        current_function_arguments += chunk.delta
                        try:
                            stream_callback(chunk.delta)
                        except Exception as e:
                            print(f'Stream callback error: {e}')
                elif chunk.type == 'response.function_call_arguments.done':
                    if hasattr(chunk, 'arguments') and hasattr(chunk, 'item_id'):
                        for fc in function_calls:
                            if fc.get('id') == chunk.item_id:
                                fc['arguments'] = chunk.arguments
                                break
                    if current_function_call and current_function_arguments:
                        current_function_call['arguments'] = current_function_arguments
                    current_function_call = None
                    current_function_arguments = ''
                    try:
                        stream_callback('\n🔧 Function arguments complete\n')
                    except Exception as e:
                        print(f'Stream callback error: {e}')
                elif chunk.type == 'response.completed':
                    try:
                        stream_callback('\n✅ Response complete\n')
                    except Exception as e:
                        print(f'Stream callback error: {e}')
        result = AgentResponse(text=text, code=code, citations=citations, function_calls=function_calls)
    else:
        result = parse_completion(completion, add_citations=True)
    return result

