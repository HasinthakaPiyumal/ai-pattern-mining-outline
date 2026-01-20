# Cluster 1

class Chat:

    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            api_key = 'fake_openai_api_key'
            self.local_mode = True
        else:
            self.local_mode = False
        self.chat_client = AsyncOpenAI(api_key=api_key)
        self.system_prompt = CHAT_SYSTEM_PROMPT
        if self.local_mode:
            self.system_prompt += LOCAL_MODE_APPENDIX

    async def chat(self, trace_id: str, chat_id: str, user_message: str, model: ChatModel, db_client: TraceRootMongoDBClient | TraceRootSQLiteClient, timestamp: datetime, tree: SpanNode, user_sub: str, chat_history: list[dict] | None=None, openai_token: str | None=None) -> ChatbotResponse:
        """Main chat entrypoint for TraceRoot assistant."""
        if model == ChatModel.AUTO:
            model = ChatModel.GPT_4O
        client = AsyncOpenAI(api_key=openai_token) if openai_token else self.chat_client
        log_features, span_features, log_node_selector_output = await asyncio.gather(log_feature_selector(user_message=user_message, client=client, model=model), span_feature_selector(user_message=user_message, client=client, model=model), log_node_selector(user_message=user_message, client=client, model=model))
        try:
            if LogFeature.LOG_LEVEL in log_node_selector_output.log_features and len(log_node_selector_output.log_features) == 1:
                tree = filter_log_node(feature_types=log_node_selector_output.log_features, feature_values=log_node_selector_output.log_feature_values, feature_ops=log_node_selector_output.log_feature_ops, node=tree)
        except Exception as e:
            print(e)
        tree = tree.to_dict(log_features=log_features, span_features=span_features)
        context = f'{json.dumps(tree, indent=4)}'
        estimated_tokens = len(context) * 4
        stats_timestamp = datetime.now().astimezone(timezone.utc)
        await db_client.insert_chat_record(message={'chat_id': chat_id, 'timestamp': stats_timestamp, 'role': 'statistics', 'content': f'Number of estimated tokens for TraceRoot context: {estimated_tokens}', 'trace_id': trace_id, 'chunk_id': 0, 'action_type': ActionType.STATISTICS.value, 'status': ActionStatus.SUCCESS.value})
        context_chunks = self.get_context_messages(context)
        context_messages = [deepcopy(context_chunks[i]) for i in range(len(context_chunks))]
        for i, message in enumerate(context_chunks):
            context_messages[i] = f'{message}\n\nHere are my questions: {user_message}'
        messages = [{'role': 'system', 'content': self.system_prompt}]
        chat_history = [chat for chat in chat_history if chat['role'] != 'github' and chat['role'] != 'statistics']
        if chat_history is not None:
            for record in chat_history[-10:]:
                if 'user_message' in record and record['user_message'] is not None:
                    content = record['user_message']
                else:
                    content = record['content']
                messages.append({'role': record['role'], 'content': content})
        all_messages: list[list[dict[str, str]]] = [deepcopy(messages) for _ in range(len(context_messages))]
        for i in range(len(context_messages)):
            all_messages[i].append({'role': 'user', 'content': context_messages[i]})
            await db_client.insert_chat_record(message={'chat_id': chat_id, 'timestamp': timestamp, 'role': 'user', 'content': context_messages[i], 'trace_id': trace_id, 'user_message': user_message, 'context': context_chunks[i], 'chunk_id': i, 'action_type': ActionType.AGENT_CHAT.value, 'status': ActionStatus.PENDING.value})
        responses: list[ChatOutput] = await asyncio.gather(*[self.chat_with_context_chunks_streaming(messages, model, client, user_sub, db_client, chat_id, trace_id, i) for i, messages in enumerate(all_messages)])
        response_time = datetime.now().astimezone(timezone.utc)
        if len(responses) == 1:
            response = responses[0]
            response_content = response.answer
            response_references = response.reference
        else:
            response_answers = [response.answer for response in responses]
            response_references = [response.reference for response in responses]
            response = await chunk_summarize(response_answers=response_answers, response_references=response_references, client=client, model=model, user_sub=user_sub)
            response_content = response.answer
            response_references = response.reference
        print('References:', response_references)
        print('Message content:', response_content)
        await db_client.insert_chat_record(message={'chat_id': chat_id, 'timestamp': response_time, 'role': 'assistant', 'content': response_content, 'reference': [ref.model_dump() for ref in response_references], 'trace_id': trace_id, 'chunk_id': 0, 'action_type': ActionType.AGENT_CHAT.value, 'status': ActionStatus.SUCCESS.value})
        return ChatbotResponse(time=response_time, message=response_content, reference=response.reference, message_type=MessageType.ASSISTANT, chat_id=chat_id)

    async def chat_with_context_chunks_streaming(self, messages: list[dict[str, str]], model: ChatModel, chat_client: AsyncOpenAI, user_sub: str, db_client: TraceRootMongoDBClient | TraceRootSQLiteClient, chat_id: str, trace_id: str, chunk_id: int) -> ChatOutput:
        """Chat with context chunks in streaming mode with database updates.
        """
        start_time = datetime.now().astimezone(timezone.utc)
        await db_client.insert_chat_record(message={'chat_id': chat_id, 'timestamp': start_time, 'role': 'assistant', 'content': '', 'reference': [], 'trace_id': trace_id, 'chunk_id': chunk_id, 'action_type': ActionType.AGENT_CHAT.value, 'status': ActionStatus.PENDING.value, 'is_streaming': True})
        return await self._chat_with_context_chunks_streaming_with_db(messages, model, chat_client, user_sub, db_client, chat_id, trace_id, chunk_id, start_time)

    async def _chat_with_context_chunks_streaming_with_db(self, messages: list[dict[str, str]], model: ChatModel, chat_client: AsyncOpenAI, user_sub: str, db_client: TraceRootMongoDBClient | TraceRootSQLiteClient, chat_id: str, trace_id: str, chunk_id: int, start_time) -> ChatOutput:
        """Chat with context chunks in streaming mode with real-time database updates.
        """
        if model == ChatModel.GPT_5.value:
            model = ChatModel.GPT_4_1.value
        if model in {ChatModel.GPT_5.value, ChatModel.GPT_5_MINI.value, ChatModel.O4_MINI.value}:
            params = {}
        else:
            params = {'temperature': 0.8}
        response = await chat_client.chat.completions.create(model=model, messages=messages, stream=True, stream_options={'include_usage': True}, response_format={'type': 'json_object'}, **params)
        content_parts = []
        usage_data = None
        async for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    content_parts.append(delta.content)
                    await self._update_streaming_record(db_client, chat_id, trace_id, chunk_id, delta.content, start_time, ActionStatus.PENDING)
            if hasattr(chunk, 'usage') and chunk.usage:
                usage_data = chunk.usage
        await db_client.update_reasoning_status(chat_id, chunk_id, 'completed')
        full_content = ''.join(content_parts)
        if usage_data:
            mock_response = type('MockResponse', (), {'usage': usage_data, 'choices': [type('Choice', (), {'message': type('Message', (), {'content': full_content})()})()]})()
            await track_tokens_for_user(user_sub=user_sub, openai_response=mock_response, model=str(model))
        try:
            parsed_data = json.loads(full_content)
            references = []
            if 'reference' in parsed_data and isinstance(parsed_data['reference'], list):
                for ref_data in parsed_data['reference']:
                    if isinstance(ref_data, dict):
                        references.append(Reference(**ref_data))
            return ChatOutput(answer=parsed_data.get('answer', full_content), reference=references)
        except (json.JSONDecodeError, Exception) as e:
            print(f'JSON parsing failed in streaming mode: {e}')
            return ChatOutput(answer=full_content, reference=[])

    async def _update_streaming_record(self, db_client: TraceRootMongoDBClient | TraceRootSQLiteClient, chat_id: str, trace_id: str, chunk_id: int, content: str, start_time, status: ActionStatus):
        """Update the streaming record in the database using dedicated
        reasoning storage.
        """
        timestamp = datetime.now(timezone.utc)
        reasoning_data = {'chat_id': chat_id, 'chunk_id': chunk_id, 'content': content, 'status': 'pending' if status == ActionStatus.PENDING else 'completed', 'timestamp': timestamp, 'trace_id': trace_id}
        await db_client.insert_reasoning_record(reasoning_data)

    def get_context_messages(self, context: str) -> list[str]:
        """Get the context message.
        """
        context_chunks = list(sequential_chunk(context))
        if len(context_chunks) == 1:
            return [f'\n\nHere is the structure of the tree with related information:\n\n{context}']
        messages: list[str] = []
        for i, chunk in enumerate(context_chunks):
            messages.append(f'\n\nHere is the structure of the tree with related information of the {i + 1}th chunk of the tree:\n\n{chunk}')
        return messages

def sequential_chunk(text: str, chunk_size: int=CHUNK_SIZE, overlap_size: int=OVERLAP_SIZE) -> Iterator[str]:
    """Chunk the text sequentially with a defined overlap.

    Args:
        text (str): The text to chunk.
        chunk_size (int): The size of each chunk.
        overlap_size (int): The size of the character overlap
            between consecutive chunks.

    Returns:
        An iterator that yields string chunks.
    """
    if overlap_size >= chunk_size:
        raise ValueError('overlap_size must be smaller than chunk_size.')
    step_size = chunk_size - overlap_size
    for i in range(0, len(text), step_size):
        yield text[i:i + chunk_size]

class Agent:

    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            api_key = 'fake_openai_api_key'
        self.chat_client = AsyncOpenAI(api_key=api_key)
        self.system_prompt = AGENT_SYSTEM_PROMPT

    async def chat(self, trace_id: str, chat_id: str, user_message: str, model: ChatModel, db_client: TraceRootMongoDBClient, timestamp: datetime, tree: SpanNode, user_sub: str, chat_history: list[dict] | None=None, openai_token: str | None=None, github_token: str | None=None, github_file_tasks: set[tuple[str, str, str, str]] | None=None, is_github_issue: bool=False, is_github_pr: bool=False, provider: Provider | None=None) -> ChatbotResponse:
        """
        Args:
            chat_id (str): The ID of the chat.
            user_message (str): The message from the user.
            model (ChatModel): The model to use.
            db_client (TraceRootMongoDBClient):
                The database client.
            timestamp (datetime): The timestamp of the user message.
            tree (dict[str, Any] | None): The tree of the trace.
            chat_history (list[dict] | None): The history of the
                chat where there are keys including chat_id, timestamp, role
                and content.
            openai_token (str | None): The OpenAI token to use.
            github_token (str | None): The GitHub token to use.
            github_file_tasks (set[tuple[str, str, str, str]] | None):
                The tasks to be done on GitHub.
            is_github_issue (bool): Whether the user wants to create an issue.
            is_github_pr (bool): Whether the user wants to create a PR.
            provider (Provider): The provider to use.
        """
        if not (is_github_issue or is_github_pr):
            raise ValueError('Either is_github_issue or is_github_pr must be True.')
        if model == ChatModel.AUTO:
            model = ChatModel.GPT_4O
        if github_file_tasks is not None:
            github_str = '\n'.join([f'({task[0]}, {task[1]}, {task[2]}, {task[3]})' for task in github_file_tasks])
            github_message = f'Here are the github file tasks: {github_str} where the first element is the owner, the second element is the repo name, and the third element is the file path, and the fourth element is the base branch.'
        client = AsyncOpenAI(api_key=openai_token) if openai_token else self.chat_client
        log_features, span_features, log_node_selector_output = await self._selector_handler(user_message, client, model)
        try:
            if LogFeature.LOG_LEVEL in log_node_selector_output.log_features and len(log_node_selector_output.log_features) == 1:
                tree = filter_log_node(feature_types=log_node_selector_output.log_features, feature_values=log_node_selector_output.log_feature_values, feature_ops=log_node_selector_output.log_feature_ops, node=tree, is_github_pr=is_github_pr)
        except Exception as e:
            print(e)
        if is_github_pr:
            if LogFeature.LOG_SOURCE_CODE_LINE not in log_features:
                log_features.append(LogFeature.LOG_SOURCE_CODE_LINE)
            if LogFeature.LOG_SOURCE_CODE_LINES_ABOVE not in log_features:
                log_features.append(LogFeature.LOG_SOURCE_CODE_LINES_ABOVE)
            if LogFeature.LOG_SOURCE_CODE_LINES_BELOW not in log_features:
                log_features.append(LogFeature.LOG_SOURCE_CODE_LINES_BELOW)
        tree = tree.to_dict(log_features=log_features, span_features=span_features)
        context = f'{json.dumps(tree, indent=4)}'
        estimated_tokens = len(context) * 4
        await db_client.insert_chat_record(message={'chat_id': chat_id, 'timestamp': datetime.now().astimezone(timezone.utc), 'role': 'statistics', 'content': f'Number of estimated tokens for TraceRoot context: {estimated_tokens}', 'trace_id': trace_id, 'chunk_id': 0, 'action_type': ActionType.STATISTICS.value, 'status': ActionStatus.SUCCESS.value})
        context_chunks = self.get_context_messages(context)
        context_messages = [deepcopy(context_chunks[i]) for i in range(len(context_chunks))]
        for i, msg in enumerate(context_chunks):
            if is_github_issue:
                updated_message = self._context_chunk_msg_handler(msg, ISSUE_TYPE.GITHUB_ISSUE)
            elif is_github_pr:
                updated_message = self._context_chunk_msg_handler(msg, ISSUE_TYPE.GITHUB_PR)
            else:
                updated_message = msg
            context_messages[i] = f'{updated_message}\n\nHere are my questions: {user_message}\n\n{github_message}'
        messages = [{'role': 'system', 'content': self.system_prompt}]
        chat_history = [chat for chat in chat_history if chat['role'] != 'github' and chat['role'] != 'statistics']
        if chat_history is not None:
            for record in chat_history[-MAX_PREV_RECORD:]:
                if 'user_message' in record:
                    content = record['user_message']
                else:
                    content = record['content']
                messages.append({'role': record['role'], 'content': content})
        all_messages: list[list[dict[str, str]]] = [deepcopy(messages) for _ in range(len(context_messages))]
        for i in range(len(context_messages)):
            all_messages[i].append({'role': 'user', 'content': context_messages[i]})
            await db_client.insert_chat_record(message={'chat_id': chat_id, 'timestamp': timestamp, 'role': 'user', 'content': context_messages[i], 'trace_id': trace_id, 'user_message': user_message, 'context': context_chunks[i], 'chunk_id': i, 'action_type': ActionType.AGENT_CHAT.value, 'status': ActionStatus.PENDING.value})
        await self._add_fake_reasoning_message(db_client, chat_id, trace_id, 0, 'Analyzing trace data and determining appropriate GitHub actions...\n')
        await self._add_fake_reasoning_message(db_client, chat_id, trace_id, 0, 'Specifying corresponding GitHub tools...\n')
        responses = await asyncio.gather(*[self.chat_with_context_chunks_streaming(messages, model, client, provider, user_sub, db_client, chat_id, trace_id, i) for i, messages in enumerate(all_messages)])
        response = responses[0]
        if isinstance(response, dict) and response:
            if 'file_path_to_change' in response:
                await self._add_fake_reasoning_message(db_client, chat_id, trace_id, 0, f'Using GitHub PR tool to create pull request for {response.get('repo_name', 'repository')}...\n')
            elif 'title' in response and 'body' in response:
                await self._add_fake_reasoning_message(db_client, chat_id, trace_id, 0, f'Using GitHub Issue tool to create issue for {response.get('repo_name', 'repository')}...\n')
        github_client = GitHubClient()
        maybe_return_directly: bool = False
        if is_github_issue:
            content, action_type = await self._issue_handler(response, github_token, github_client)
        elif is_github_pr:
            if 'file_path_to_change' in response:
                _, content, action_type = await self._pr_handler(response, github_token, github_client)
            else:
                maybe_return_directly = True
        if not maybe_return_directly:
            await db_client.insert_chat_record(message={'chat_id': chat_id, 'timestamp': datetime.now().astimezone(timezone.utc), 'role': 'github', 'content': content, 'reference': [], 'trace_id': trace_id, 'chunk_id': 0, 'action_type': action_type, 'status': ActionStatus.SUCCESS.value})
        response_time = datetime.now().astimezone(timezone.utc)
        if not maybe_return_directly:
            summary_response = await client.chat.completions.create(model=ChatModel.GPT_4_1.value, messages=[{'role': 'system', 'content': 'You are a helpful assistant that can summarize the created issue or the created PR.'}, {'role': 'user', 'content': f'Here is the created issue or the created PR:\n{response}'}], stream=True, stream_options={'include_usage': True})
            content_parts = []
            usage_data = None
            async for chunk in summary_response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content_parts.append(delta.content)
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_data = chunk.usage
            await db_client.update_reasoning_status(chat_id, 0, 'completed')
            summary_content = ''.join(content_parts)
            if usage_data:
                mock_response = type('MockResponse', (), {'usage': usage_data, 'choices': [type('Choice', (), {'message': type('Message', (), {'content': summary_content})()})()]})()
                await track_tokens_for_user(user_sub=user_sub, openai_response=mock_response, model=str(model))
        else:
            summary_content = response['content']
        await db_client.insert_chat_record(message={'chat_id': chat_id, 'timestamp': response_time, 'role': 'assistant', 'content': summary_content, 'reference': [], 'trace_id': trace_id, 'chunk_id': 0, 'action_type': ActionType.AGENT_CHAT.value, 'status': ActionStatus.SUCCESS.value})
        return ChatbotResponse(time=response_time, message=summary_content, reference=[], message_type=MessageType.ASSISTANT, chat_id=chat_id)

    async def chat_with_context_chunks_streaming(self, messages: list[dict[str, str]], model: ChatModel, chat_client: AsyncOpenAI, provider: Provider, user_sub: str, db_client: TraceRootMongoDBClient, chat_id: str, trace_id: str, chunk_id: int) -> dict[str, Any]:
        """Chat with context chunks in streaming mode with database updates."""
        start_time = datetime.now().astimezone(timezone.utc)
        await db_client.insert_chat_record(message={'chat_id': chat_id, 'timestamp': start_time, 'role': 'assistant', 'content': '', 'reference': [], 'trace_id': trace_id, 'chunk_id': chunk_id, 'action_type': ActionType.AGENT_CHAT.value, 'status': ActionStatus.PENDING.value, 'is_streaming': True})
        return await self._chat_with_context_openai_streaming(messages, model, user_sub, chat_client, db_client, chat_id, trace_id, chunk_id, start_time)

    def get_context_messages(self, context: str) -> list[str]:
        """Get the context message."""
        context_chunks = list(sequential_chunk(context))
        if len(context_chunks) == 1:
            return [f'\n\nHere is the structure of the tree with related information:\n\n{context}']
        messages: list[str] = []
        for i, chunk in enumerate(context_chunks):
            messages.append(f'\n\nHere is the structure of the tree with related information of the {i + 1}th chunk of the tree:\n\n{chunk}')
        return messages

    async def _selector_handler(self, user_message, client, model) -> tuple[list[LogFeature], list[SpanFeature], LogNodeSelectorOutput]:
        return await asyncio.gather(log_feature_selector(user_message=user_message, client=client, model=model), span_feature_selector(user_message=user_message, client=client, model=model), log_node_selector(user_message=user_message, client=client, model=model))

    def _context_chunk_msg_handler(self, message: str, issue_type: ISSUE_TYPE):
        if issue_type == ISSUE_TYPE.GITHUB_ISSUE:
            return f'\n                {message}\nFor now please create an GitHub issue.\n\n            '
        if issue_type == ISSUE_TYPE.GITHUB_PR:
            return f'\n                {message}\nFor now please create a GitHub PR.\n\n            '

    async def _pr_handler(self, response: dict[str, Any], github_token: str | None, github_client: GitHubClient) -> Tuple[str, str, str]:
        pr_number = await github_client.create_pr_with_file_changes(title=response['title'], body=response['body'], owner=response['owner'], repo_name=response['repo_name'], base_branch=response['base_branch'], head_branch=response['head_branch'], file_path_to_change=response['file_path_to_change'], file_content_to_change=response['file_content_to_change'], commit_message=response['commit_message'], github_token=github_token)
        url = f'https://github.com/{response['owner']}/{response['repo_name']}/pull/{pr_number}'
        content = f'PR created: {url}'
        action_type = ActionType.GITHUB_CREATE_PR.value
        return (url, content, action_type)

    async def _issue_handler(self, response: dict[str, Any], github_token: str | None, github_client: GitHubClient) -> Tuple[str, str]:
        issue_number = await github_client.create_issue(title=response['title'], body=response['body'], owner=response['owner'], repo_name=response['repo_name'], github_token=github_token)
        url = f'https://github.com/{response['owner']}/{response['repo_name']}/issues/{issue_number}'
        content = f'Issue created: {url}'
        action_type = ActionType.GITHUB_CREATE_ISSUE.value
        return (content, action_type)

    async def _chat_with_context_openai_streaming(self, messages: list[dict[str, str]], model: ChatModel, user_sub: str, chat_client: AsyncOpenAI, db_client: TraceRootMongoDBClient=None, chat_id: str=None, trace_id: str=None, chunk_id: int=None, start_time=None):
        allowed_model = {ChatModel.GPT_5, ChatModel.O4_MINI}
        if model not in allowed_model:
            model = ChatModel.O4_MINI
        response = await chat_client.chat.completions.create(model=model, messages=messages, tools=[get_openai_tool_schema(create_issue), get_openai_tool_schema(create_pr_with_file_changes)], stream=False)
        tool_calls_data = None
        response.usage
        full_content = response.choices[0].message.content or ''
        tool_calls_data = response.choices[0].message.tool_calls
        await track_tokens_for_user(user_sub=user_sub, openai_response=response, model=str(model))
        if tool_calls_data is None or len(tool_calls_data) == 0:
            return {'content': full_content}
        else:
            arguments = tool_calls_data[0].function.arguments
            arguments = json.loads(arguments)
            return arguments

    async def _update_streaming_record(self, db_client: TraceRootMongoDBClient, chat_id: str, trace_id: str, chunk_id: int, content: str, start_time, status: ActionStatus):
        """Update the streaming record in the database."""
        timestamp = datetime.now().astimezone(timezone.utc)
        await db_client.insert_chat_record(message={'chat_id': chat_id, 'timestamp': timestamp, 'role': 'assistant', 'content': content, 'reference': [], 'trace_id': trace_id, 'chunk_id': chunk_id, 'action_type': ActionType.AGENT_CHAT.value, 'status': status.value, 'is_streaming': True, 'stream_update': True})
        from rest.routers.streaming import get_streaming_router_instance
        streaming_router = get_streaming_router_instance()
        if streaming_router:
            await streaming_router.broadcast_streaming_update(chat_id=chat_id, chunk_id=chunk_id, data={'content': content, 'status': status.value, 'timestamp': timestamp.isoformat(), 'trace_id': trace_id})

    async def _chat_with_context_openai(self, messages: list[dict[str, str]], model: ChatModel, user_sub: str, chat_client: AsyncOpenAI, stream: bool=False):
        allowed_model = {ChatModel.GPT_5, ChatModel.O4_MINI}
        if model not in allowed_model:
            model = ChatModel.O4_MINI
        response = await chat_client.chat.completions.create(model=model, messages=messages, tools=[get_openai_tool_schema(create_issue), get_openai_tool_schema(create_pr_with_file_changes)], stream=stream)
        if stream:
            content_parts = []
            tool_calls_data = None
            async for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content_parts.append(delta.content)
                    if delta.tool_calls:
                        if tool_calls_data is None:
                            tool_calls_data = delta.tool_calls
                        else:
                            for i, tool_call in enumerate(delta.tool_calls):
                                if i < len(tool_calls_data):
                                    tc_data = tool_calls_data[i]
                                    if tool_call.function and tool_call.function.arguments:
                                        tc_data.function.arguments += tool_call.function.arguments
                                else:
                                    tool_calls_data.append(tool_call)

            class MockResponse:

                def __init__(self, content, tool_calls):
                    self.usage = None
                    self.choices = [type('Choice', (), {'message': type('Message', (), {'content': content, 'tool_calls': tool_calls})()})()]
            full_content = ''.join(content_parts)
            mock_response = MockResponse(full_content, tool_calls_data)
            await track_tokens_for_user(user_sub=user_sub, openai_response=mock_response, model=str(model))
            if tool_calls_data is None or len(tool_calls_data) == 0:
                return {'content': full_content}
            else:
                arguments = tool_calls_data[0].function.arguments
                return json.loads(arguments)
        else:
            await track_tokens_for_user(user_sub=user_sub, openai_response=response, model=str(model))
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls is None or len(tool_calls) == 0:
                return {'content': response.choices[0].message.content}
            else:
                arguments = tool_calls[0].function.arguments
                return json.loads(arguments)

    async def _add_fake_reasoning_message(self, db_client: TraceRootMongoDBClient, chat_id: str, trace_id: str, chunk_id: int, content: str):
        """Add a fake reasoning message for better UX."""
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc)
        reasoning_data = {'chat_id': chat_id, 'chunk_id': chunk_id, 'content': content, 'status': 'pending', 'timestamp': timestamp, 'trace_id': trace_id}
        await db_client.insert_reasoning_record(reasoning_data)

