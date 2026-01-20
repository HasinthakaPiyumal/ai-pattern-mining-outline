# Cluster 8

class ChatLogic:
    """Business logic for chat and AI agent operations."""

    def __init__(self, local_mode: bool):
        """Initialize chat driver.

        Args:
            local_mode: Whether running in local mode (SQLite) or cloud mode (MongoDB)
        """
        self.local_mode = local_mode
        self.logger = logging.getLogger(__name__)
        self.single_rca_agent = SingleRCAAgent()
        self.code_agent = CodeAgent()
        self.general_agent = GeneralAgent()
        self.chat_router = ChatRouter()
        if self.local_mode:
            self.db_client = TraceRootSQLiteClient()
        else:
            self.db_client = TraceRootMongoDBClient()
        self.github = GitHubClient()
        self.cache = Cache()
        self.cache_helper = TraceCacheHelper(self.cache)
        if self.local_mode:
            self.default_observe_provider = ObservabilityProvider.create_jaeger_provider()
        else:
            self.default_observe_provider = ObservabilityProvider.create_aws_provider()

    async def get_line_context_content(self, url: str, user_email: str) -> dict[str, Any]:
        """Get file line context content from GitHub URL.

        This is called to show the code in the UI.

        Args:
            url: GitHub URL to fetch
            user_email: User email for GitHub token retrieval

        Returns:
            Dictionary of CodeResponse.model_dump()

        Raises:
            HTTPException: If URL is invalid or file cannot be retrieved
        """
        github_token = await self.get_github_token(user_email)
        owner, repo, ref, file_path, line_num = parse_github_url(url)
        return await self.github.get_file_with_context(owner=owner, repo=repo, file_path=file_path, ref=ref, line_num=line_num, github_token=github_token, cache=self.cache, line_context_len=4)

    async def get_chat_history(self, chat_id: str) -> dict[str, Any]:
        """Get chat history for a given chat ID.

        Args:
            chat_id: Chat ID to fetch history for

        Returns:
            Dictionary containing chat history
        """
        history: list[dict[str, Any]] = await self.db_client.get_chat_history(chat_id=chat_id)
        chat_history = ChatHistoryResponse(history=[])
        for item in history:
            if item['role'] == 'user':
                if 'user_message' in item:
                    message = item['user_message']
                else:
                    message = item['content']
            else:
                message = item['content']
            if item['role'] == 'assistant' and 'reference' in item:
                reference = [Reference(**ref) for ref in item['reference']]
            else:
                reference = []
            chunk_id = 0
            if 'chunk_id' in item and item['chunk_id'] > 0:
                continue
            if 'chunk_id' in item:
                chunk_id = int(item['chunk_id'])
            if 'action_type' in item:
                action_type = item['action_type']
            else:
                action_type = None
            if 'status' in item:
                status = item['status']
            else:
                status = None
            chat_history.history.append(ChatbotResponse(time=item['timestamp'], message=message, reference=reference, message_type=item['role'], chat_id=item['chat_id'], chunk_id=chunk_id, action_type=action_type, status=status))
        return chat_history.model_dump()

    async def get_chat_metadata_history(self, trace_id: str) -> dict[str, Any]:
        """Get chat metadata history for a given trace ID.

        Args:
            trace_id: Trace ID to fetch metadata for

        Returns:
            Dictionary containing chat metadata history
        """
        chat_metadata_history: ChatMetadataHistory = await self.db_client.get_chat_metadata_history(trace_id=trace_id)
        return chat_metadata_history.model_dump()

    async def get_chat_metadata(self, chat_id: str) -> dict[str, Any]:
        """Get chat metadata for a given chat ID.

        Args:
            chat_id: Chat ID to fetch metadata for

        Returns:
            Dictionary containing chat metadata, or empty dict if not found
        """
        chat_metadata: ChatMetadata | None = await self.db_client.get_chat_metadata(chat_id=chat_id)
        if chat_metadata is None:
            return {}
        return chat_metadata.model_dump()

    async def get_github_token(self, user_email: str) -> str | None:
        """Get GitHub token for a user.

        Args:
            user_email: User's email address

        Returns:
            GitHub token or None if not found
        """
        return await self.db_client.get_integration_token(user_email=user_email, token_type=ResourceType.GITHUB.value)

    async def post_chat(self, request: Request, req_data: ChatRequest, user_email: str, user_sub: str) -> dict[str, Any]:
        """Business logic for chat - orchestrates AI agents.

        This is the main chat handler that:
        1. Validates and extracts inputs
        2. Checks for confirmation flows
        3. Determines GitHub-related context
        4. Routes to appropriate AI agent
        5. Fetches traces, logs, and source code
        6. Coordinates multiple agents if needed

        Args:
            request: FastAPI request object
            req_data: Chat request data
            user_email: User's email address
            user_sub: User's subject ID

        Returns:
            Dictionary containing chatbot response
        """
        log_group_name = hash_user_sub(user_sub)
        trace_id = req_data.trace_id
        span_ids = req_data.span_ids
        start_time = req_data.start_time
        end_time = req_data.end_time
        model = req_data.model
        message = req_data.message
        chat_id = req_data.chat_id
        service_name = req_data.service_name
        mode = req_data.mode
        provider = req_data.provider
        if model == ChatModel.AUTO:
            model = ChatModel.GPT_4O
        elif provider == Provider.CUSTOM:
            model = ChatModel.GPT_4O
        if req_data.time.tzinfo:
            orig_time = req_data.time.astimezone(timezone.utc)
        else:
            orig_time = req_data.time.replace(tzinfo=timezone.utc)
        openai_token = await self.db_client.get_integration_token(user_email=user_email, token_type=ResourceType.OPENAI.value)
        if openai_token is None and self.single_rca_agent.local_mode:
            response = ChatbotResponse(time=orig_time, message='OpenAI token is not found, please add it in the settings page.', reference=[], message_type=MessageType.ASSISTANT, chat_id=chat_id)
            return response.model_dump()
        first_chat: bool = False
        if await self.db_client.get_chat_metadata(chat_id=chat_id) is None:
            first_chat = True
        github_token = await self.get_github_token(user_email)
        if not first_chat:
            early_chat_history = await self.db_client.get_chat_history(chat_id=chat_id)
            is_confirmation = self._check_user_confirmation_response(early_chat_history, message)
            if is_confirmation:
                return await self._handle_confirmation_response(early_chat_history=early_chat_history, message=message, chat_id=chat_id, trace_id=trace_id, orig_time=orig_time, github_token=github_token)
        title, github_related = await asyncio.gather(summarize_title(user_message=message, client=self.single_rca_agent.chat_client, openai_token=openai_token, model=ChatModel.GPT_4_1_MINI, first_chat=first_chat, user_sub=user_sub), is_github_related(user_message=message, client=self.single_rca_agent.chat_client, openai_token=openai_token, model=ChatModel.GPT_4O, user_sub=user_sub))
        if first_chat and title is not None:
            await self.db_client.insert_chat_metadata(metadata={'chat_id': chat_id, 'timestamp': orig_time, 'chat_title': title, 'trace_id': trace_id, 'user_id': user_sub})
        set_github_related(github_related)
        is_github_issue: bool = False
        is_github_pr: bool = False
        source_code_related: bool = False
        source_code_related = github_related.source_code_related
        if mode == ChatMode.AGENT:
            is_github_issue = github_related.is_github_issue
            is_github_pr = github_related.is_github_pr
        router_output = await self.chat_router.route_query(user_message=message, chat_mode=mode, model=ChatModel.GPT_4O, user_sub=user_sub, openai_token=openai_token, has_trace_context=bool(trace_id), is_github_issue=is_github_issue, is_github_pr=is_github_pr, source_code_related=source_code_related)
        await self.db_client.insert_chat_routing_record({'chat_id': chat_id, 'timestamp': orig_time, 'user_message': message, 'agent_type': router_output.agent_type, 'reasoning': router_output.reasoning, 'chat_mode': mode.value, 'trace_id': trace_id or '', 'user_sub': user_sub})
        if router_output.agent_type == 'general':
            chat_history = await self.db_client.get_chat_history(chat_id=chat_id)
            response = await self.general_agent.chat(chat_id=chat_id, user_message=message, model=model, db_client=self.db_client, timestamp=orig_time, user_sub=user_sub, chat_history=chat_history, openai_token=openai_token, trace_id=trace_id)
            return response.model_dump()
        observe_provider = await get_observe_provider(request=request, db_client=self.db_client, local_mode=self.local_mode, default_provider=self.default_observe_provider, trace_provider=req_data.trace_provider, log_provider=req_data.log_provider, trace_region=req_data.trace_region, log_region=req_data.log_region)
        selected_trace: Trace | None = None
        if trace_id:
            selected_trace = await observe_provider.trace_client.get_trace_by_id(trace_id=trace_id, categories=None, values=None, operations=None)
        else:
            simple_cache_key = self.cache_helper.build_simple_trace_cache_key(start_time, end_time, service_name, log_group_name)
            cached_traces: list[Trace] | None = await self.cache_helper.get_simple_traces(simple_cache_key)
            if cached_traces:
                traces = cached_traces
            else:
                traces: list[Trace] = await observe_provider.trace_client.get_recent_traces(start_time=start_time, end_time=end_time, log_group_name=log_group_name, service_name_values=None, service_name_operations=None, service_environment_values=None, service_environment_operations=None, categories=None, values=None, operations=None)
                await self.cache_helper.cache_simple_traces(simple_cache_key, traces)
            for trace in traces:
                if trace.id == trace_id:
                    selected_trace = trace
                    break
        spans_latency_dict: dict[str, float] = {}
        if selected_trace:
            collect_spans_latency_recursively(selected_trace.spans, spans_latency_dict)
            if len(span_ids) > 0:
                selected_spans_latency_dict: dict[str, float] = {}
                for span_id, latency in spans_latency_dict.items():
                    if span_id in span_ids:
                        selected_spans_latency_dict[span_id] = latency
                spans_latency_dict = selected_spans_latency_dict
        trace_start_time = None
        trace_end_time = None
        if selected_trace:
            if selected_trace.service_name == 'LimitExceeded' and selected_trace.start_time == 0.0 and (selected_trace.end_time == 0.0):
                try:
                    earliest, latest = await observe_provider.log_client.get_log_timestamps_by_trace_id(trace_id=trace_id, log_group_name=log_group_name, start_time=start_time, end_time=end_time)
                    if earliest and latest:
                        trace_start_time = earliest
                        trace_end_time = latest
                        selected_trace.start_time = earliest.timestamp()
                        selected_trace.end_time = latest.timestamp()
                        selected_trace.duration = latest.timestamp() - earliest.timestamp()
                        if selected_trace.spans and len(selected_trace.spans) > 0:
                            placeholder_span = selected_trace.spans[0]
                            placeholder_span.start_time = earliest.timestamp()
                            placeholder_span.end_time = latest.timestamp()
                            placeholder_span.duration = latest.timestamp() - earliest.timestamp()
                except Exception as e:
                    print(f'Failed to get log timestamps for LimitExceeded trace {trace_id}: {e}')
            else:
                trace_start_time = datetime.fromtimestamp(selected_trace.start_time, tz=timezone.utc)
                trace_end_time = datetime.fromtimestamp(selected_trace.end_time, tz=timezone.utc)
        log_start_time = trace_start_time if trace_start_time else start_time
        log_end_time = trace_end_time if trace_end_time else end_time
        log_cache_key = self.cache_helper.build_log_cache_key(trace_id, log_start_time, log_end_time, log_group_name)
        logs: TraceLogs | None = await self.cache_helper.get_logs(log_cache_key)
        if logs is None:
            logs = await observe_provider.log_client.get_logs_by_trace_id(trace_id=trace_id, start_time=log_start_time, end_time=log_end_time, log_group_name=log_group_name)
            await self.cache_helper.cache_logs(log_cache_key, logs)
        github_tasks: list[tuple[str, str, str, str]] = []
        log_entries_to_update: list = []
        github_task_keys: set[tuple[str, str, str, str]] = set()
        unique_file_tasks: dict = {}
        if source_code_related:
            for log in logs.logs:
                for span_id, span_logs in log.items():
                    for log_entry in span_logs:
                        if log_entry.git_url:
                            owner, repo_name, ref, file_path, line_number = parse_github_url(log_entry.git_url)
                            if is_github_pr:
                                line_context_len = 200
                            else:
                                line_context_len = 5
                            file_key = (owner, repo_name, file_path, ref)
                            if file_key not in unique_file_tasks:
                                task = self.github.get_file_with_context(owner=owner, repo=repo_name, file_path=file_path, ref=ref, line_num=line_number, github_token=github_token, cache=self.cache, line_context_len=line_context_len)
                                unique_file_tasks[file_key] = (task, [])
                                github_task_keys.add((owner, repo_name, file_path, ref))
                            unique_file_tasks[file_key][1].append((line_number, line_context_len, log_entry))
            file_keys_list = []
            for file_key, (task, entries) in unique_file_tasks.items():
                print(f'file_key: {file_key}')
                github_tasks.append(task)
                log_entries_to_update.append(entries)
                file_keys_list.append(file_key)
            batch_size = 20
            for i in range(0, len(github_tasks), batch_size):
                batch_tasks = github_tasks[i:i + batch_size]
                batch_log_entries_list = log_entries_to_update[i:i + batch_size]
                batch_file_keys = file_keys_list[i:i + batch_size]
                time = datetime.now().astimezone(timezone.utc)
                await self.db_client.insert_chat_record(message={'chat_id': chat_id, 'timestamp': time, 'role': MessageType.GITHUB.value, 'content': 'Fetching GitHub file content... ', 'trace_id': trace_id, 'chunk_id': i // batch_size, 'action_type': ActionType.GITHUB_GET_FILE.value, 'status': ActionStatus.PENDING.value})
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                num_failed = 0
                num_success = 0
                for file_key, entries_with_lines, code_response in zip(batch_file_keys, batch_log_entries_list, batch_results):
                    if isinstance(code_response, Exception):
                        num_failed += 1
                        continue
                    if code_response['error_message']:
                        num_failed += 1
                        continue
                    owner, repo_name, file_path, ref = file_key
                    for line_number, line_context_len, log_entry in entries_with_lines:
                        line_response = await self.github.get_file_with_context(owner=owner, repo=repo_name, file_path=file_path, ref=ref, line_num=line_number, github_token=github_token, cache=self.cache, line_context_len=line_context_len)
                        if not line_response['error_message']:
                            log_entry.line = line_response['line']
                            if not is_github_pr:
                                log_entry.lines_above = None
                                log_entry.lines_below = None
                            else:
                                log_entry.lines_above = line_response['lines_above']
                                log_entry.lines_below = line_response['lines_below']
                    num_success += 1
                time = datetime.now().astimezone(timezone.utc)
                await self.db_client.insert_chat_record(message={'chat_id': chat_id, 'timestamp': time, 'role': MessageType.GITHUB.value, 'content': f'Finished fetching GitHub file content for {num_success} times. Failed to fetch {num_failed} times.', 'trace_id': trace_id, 'chunk_id': i // batch_size, 'action_type': ActionType.GITHUB_GET_FILE.value, 'status': ActionStatus.SUCCESS.value})
        chat_history = await self.db_client.get_chat_history(chat_id=chat_id)
        if selected_trace.service_name == 'LimitExceeded':
            placeholder_span_id = selected_trace.spans[0].id
            reassigned_logs = []
            for log_dict in logs.logs:
                all_log_entries = []
                for span_id, log_entries in log_dict.items():
                    all_log_entries.extend(log_entries)
                if all_log_entries:
                    reassigned_logs.append({placeholder_span_id: all_log_entries})
            node: SpanNode = build_heterogeneous_tree(selected_trace.spans[0], reassigned_logs)
        else:
            node: SpanNode = build_heterogeneous_tree(selected_trace.spans[0], logs.logs)
        if len(span_ids) > 0:
            queue = deque([node])
            target_set = set(span_ids)
            while queue:
                current = queue.popleft()
                if current.span_id in target_set:
                    node = current
                    break
                for child in current.children_spans:
                    queue.append(child)
        if router_output.agent_type == 'code' and (is_github_issue or is_github_pr):
            issue_response: ChatbotResponse | None = None
            pr_response: ChatbotResponse | None = None
            issue_message: str = message
            pr_message: str = message
            if is_github_issue and is_github_pr:
                separate_issue_and_pr_output: SeparateIssueAndPrInput = await separate_issue_and_pr(user_message=message, client=self.single_rca_agent.chat_client, openai_token=openai_token, model=model, user_sub=user_sub)
                issue_message = separate_issue_and_pr_output.issue_message
                pr_message = separate_issue_and_pr_output.pr_message
            if is_github_issue:
                issue_response = await self.code_agent.chat(trace_id=trace_id, chat_id=chat_id, user_message=issue_message, model=model, db_client=self.db_client, chat_history=chat_history, timestamp=orig_time, tree=node, user_sub=user_sub, openai_token=openai_token, github_token=github_token, github_file_tasks=github_task_keys, is_github_issue=True, is_github_pr=False, provider=provider)
            if is_github_pr:
                pr_response = await self.code_agent.chat(trace_id=trace_id, chat_id=chat_id, user_message=pr_message, model=model, db_client=self.db_client, chat_history=chat_history, timestamp=orig_time, tree=node, user_sub=user_sub, openai_token=openai_token, github_token=github_token, github_file_tasks=github_task_keys, is_github_issue=False, is_github_pr=True, provider=provider)
            if issue_response and pr_response:
                summary_response = await summarize_chatbot_output(issue_response=issue_response, pr_response=pr_response, client=self.single_rca_agent.chat_client, openai_token=openai_token, model=model, user_sub=user_sub)
                return summary_response.model_dump()
            elif issue_response:
                return issue_response.model_dump()
            elif pr_response:
                return pr_response.model_dump()
            else:
                raise ValueError('Should not reach here')
        else:
            response: ChatbotResponse = await self.single_rca_agent.chat(trace_id=trace_id, chat_id=chat_id, user_message=message, model=model, db_client=self.db_client, chat_history=chat_history, timestamp=orig_time, tree=node, user_sub=user_sub, openai_token=openai_token)
            return response.model_dump()

    async def confirm_github_action(self, req_data: ConfirmActionRequest, user_email: str, user_sub: str) -> dict[str, Any]:
        """Confirm or reject a pending action (generic handler).

        Args:
            req_data: Confirmation request data
            user_email: User's email address
            user_sub: User's subject ID

        Returns:
            Confirmation response with result

        Raises:
            HTTPException: If action not found or execution fails
        """
        try:
            chat_history = await self.db_client.get_chat_history(chat_id=req_data.chat_id)
            pending_message = None
            for item in chat_history:
                if item.get('timestamp') and abs(item['timestamp'].timestamp() - req_data.message_timestamp) < 1.0 and (item.get('status') == ActionStatus.AWAITING_CONFIRMATION.value):
                    pending_message = item
                    break
            if not pending_message:
                raise HTTPException(status_code=404, detail='Pending action not found or already processed')
            action_metadata = pending_message.get('pending_action_data')
            if not action_metadata:
                raise HTTPException(status_code=400, detail='No pending action data found')
            action_kind = action_metadata.get('action_kind')
            action_data = action_metadata.get('action_data')
            if not action_kind or not action_data:
                raise HTTPException(status_code=400, detail='Invalid action metadata structure')
            if req_data.confirmed:
                result_data = await self._execute_confirmed_action(action_kind=action_kind, action_data=action_data, user_email=user_email)
                action_result_message = result_data['message']
                final_action_type = result_data['action_type']
            else:
                action_display_name = self._get_action_display_name(action_kind)
                action_result_message = f'{action_display_name} cancelled by user.'
                final_action_type = pending_message.get('action_type')
                result_data = {'message': action_result_message}
            await self.db_client.update_chat_record_status(chat_id=req_data.chat_id, timestamp=pending_message['timestamp'], status=ActionStatus.SUCCESS.value if req_data.confirmed else ActionStatus.CANCELLED.value, content=action_result_message, action_type=final_action_type, user_confirmation=req_data.confirmed)
            summary_response = await self._generate_confirmation_summary(chat_id=req_data.chat_id, confirmed=req_data.confirmed, action_kind=action_kind, action_data=action_data, result_message=action_result_message, user_email=user_email, user_sub=user_sub)
            response = ConfirmActionResponse(success=True, message=summary_response['summary'], data=result_data.get('data'))
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f'Error confirming action: {e}')
            raise HTTPException(status_code=500, detail=f'Error processing confirmation: {str(e)}')

    async def _execute_confirmed_action(self, action_kind: str, action_data: dict, user_email: str) -> dict[str, Any]:
        """Execute a confirmed action based on its kind.

        Args:
            action_kind: The type of action to execute
            action_data: The data needed to execute the action
            user_email: User's email for token retrieval

        Returns:
            Dict with message, action_type, and optional data

        Raises:
            HTTPException: If GitHub token not found or action execution fails
        """
        if action_kind == 'github_create_issue':
            github_token = await self.db_client.get_integration_token(user_email=user_email, token_type='github')
            if not github_token:
                raise HTTPException(status_code=400, detail='GitHub token not found. Please configure it in settings.')
            github_client = GitHubClient()
            issue_number = await github_client.create_issue(title=action_data['title'], body=action_data['body'], owner=action_data['owner'], repo_name=action_data['repo_name'], github_token=github_token)
            url = f'https://github.com/{action_data['owner']}/{action_data['repo_name']}/issues/{issue_number}'
            return {'message': f'Issue created: {url}', 'action_type': ActionType.GITHUB_CREATE_ISSUE.value, 'data': {'url': url, 'issue_number': issue_number}}
        elif action_kind == 'github_create_pr':
            github_token = await self.db_client.get_integration_token(user_email=user_email, token_type='github')
            if not github_token:
                raise HTTPException(status_code=400, detail='GitHub token not found. Please configure it in settings.')
            github_client = GitHubClient()
            pr_number = await github_client.create_pr_with_file_changes(title=action_data['title'], body=action_data['body'], owner=action_data['owner'], repo_name=action_data['repo_name'], base_branch=action_data['base_branch'], head_branch=action_data['head_branch'], file_path_to_change=action_data['file_path_to_change'], file_content_to_change=action_data['file_content_to_change'], commit_message=action_data['commit_message'], github_token=github_token)
            url = f'https://github.com/{action_data['owner']}/{action_data['repo_name']}/pull/{pr_number}'
            return {'message': f'PR created: {url}', 'action_type': ActionType.GITHUB_CREATE_PR.value, 'data': {'url': url, 'pr_number': pr_number}}
        else:
            raise HTTPException(status_code=400, detail=f'Unknown action kind: {action_kind}')

    def _get_action_display_name(self, action_kind: str) -> str:
        """Get a human-readable display name for an action kind.

        Args:
            action_kind: The action kind identifier

        Returns:
            Human-readable action name
        """
        action_names = {'github_create_issue': 'GitHub issue creation', 'github_create_pr': 'GitHub pull request creation'}
        return action_names.get(action_kind, 'Action')

    async def _generate_confirmation_summary(self, chat_id: str, confirmed: bool, action_kind: str, action_data: dict, result_message: str, user_email: str, user_sub: str) -> dict[str, str]:
        """Generate LLM summary after user confirms/rejects an action.

        Args:
            chat_id: The chat ID
            confirmed: Whether user confirmed or rejected
            action_kind: The type of action
            action_data: The action data
            result_message: The result message (success or cancellation)
            user_email: User's email
            user_sub: User's sub

        Returns:
            Dict with summary text
        """
        openai_token = await self.db_client.get_integration_token(user_email=user_email, token_type='openai')
        from openai import AsyncOpenAI
        if openai_token:
            client = AsyncOpenAI(api_key=openai_token)
        else:
            client = AsyncOpenAI()
        decision_text = 'confirmed' if confirmed else 'rejected'
        action_desc = self._get_action_display_name(action_kind)
        prompt = f'The user was asked to confirm a {action_desc} with the following details:\n\n{self._format_action_data(action_kind, action_data)}\n\nThe user {decision_text} this action.\nResult: {result_message}\n\nPlease provide a brief, friendly summary of what happened. Keep it conversational and to the point (1-2 sentences).'
        response = await client.chat.completions.create(model=ChatModel.GPT_4_1.value, messages=[{'role': 'system', 'content': 'You are a helpful assistant that provides brief, friendly summaries.'}, {'role': 'user', 'content': prompt}], stream=True, stream_options={'include_usage': True})
        content_parts = []
        usage_data = None
        timestamp = datetime.now().astimezone(timezone.utc)
        async for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    content_parts.append(delta.content)
            if hasattr(chunk, 'usage') and chunk.usage:
                usage_data = chunk.usage
        summary_content = ''.join(content_parts)
        await self.db_client.insert_chat_record(message={'chat_id': chat_id, 'timestamp': timestamp, 'role': 'assistant', 'content': summary_content, 'reference': [], 'trace_id': None, 'chunk_id': 0, 'action_type': ActionType.AGENT_CHAT.value, 'status': ActionStatus.SUCCESS.value})
        if usage_data:
            from rest.agent.token_tracker import track_tokens_for_user
            mock_response = type('MockResponse', (), {'usage': usage_data, 'choices': [type('Choice', (), {'message': type('Message', (), {'content': summary_content})()})()]})()
            await track_tokens_for_user(user_sub=user_sub, openai_response=mock_response, model=ChatModel.GPT_4_1.value)
        return {'summary': summary_content}

    def _format_action_data(self, action_kind: str, action_data: dict) -> str:
        """Format action data for LLM prompt.

        Args:
            action_kind: The type of action
            action_data: The action data

        Returns:
            Formatted string
        """
        if action_kind == 'github_create_issue':
            return f'Repository: {action_data['owner']}/{action_data['repo_name']}\nTitle: {action_data['title']}\nDescription: {action_data['body']}'
        elif action_kind == 'github_create_pr':
            return f'Repository: {action_data['owner']}/{action_data['repo_name']}\nTitle: {action_data['title']}\nBase Branch: {action_data['base_branch']} ← Head Branch: {action_data['head_branch']}\nDescription: {action_data['body']}'
        else:
            return str(action_data)

    def _check_user_confirmation_response(self, chat_history: list[dict] | None, user_message: str) -> bool:
        """Check if user message is a yes/no response to pending confirmation.

        Args:
            chat_history: The chat history
            user_message: The current user message

        Returns:
            True if this is a confirmation response, False otherwise
        """
        if not chat_history:
            return False
        for message in reversed(chat_history):
            if message.get('role') == 'user':
                continue
            if message.get('action_type') == ActionType.PENDING_CONFIRMATION.value and message.get('status') == ActionStatus.AWAITING_CONFIRMATION.value:
                user_msg_lower = user_message.lower().strip()
                yes_variations = ['yes', 'y', 'ok', 'okay', 'confirm', 'proceed', 'go ahead', 'do it']
                no_variations = ['no', 'n', 'cancel', 'stop', "don't", 'dont', 'nope', 'skip']
                return user_msg_lower in yes_variations or user_msg_lower in no_variations
        return False

    async def _handle_confirmation_response(self, early_chat_history: list[dict], message: str, chat_id: str, trace_id: str, orig_time: datetime, github_token: str | None) -> dict[str, Any]:
        """Handle user's yes/no response to a pending confirmation.

        Args:
            early_chat_history: Chat history
            message: User's message
            chat_id: Chat ID
            trace_id: Trace ID
            orig_time: Original time of request
            github_token: GitHub token

        Returns:
            Response dictionary
        """
        pending_action = None
        for msg in reversed(early_chat_history):
            if msg.get('role') == 'user':
                continue
            if msg.get('action_type') == ActionType.PENDING_CONFIRMATION.value and msg.get('status') == ActionStatus.AWAITING_CONFIRMATION.value:
                pending_action = msg
                break
        if not pending_action:
            return {'time': orig_time.timestamp() * 1000, 'message': 'No pending action found.', 'reference': [], 'message_type': MessageType.ASSISTANT.value, 'chat_id': chat_id}
        user_msg_lower = message.lower().strip()
        user_confirmed = user_msg_lower in ['yes', 'y', 'ok', 'okay', 'confirm', 'proceed', 'go ahead', 'do it']
        action_metadata = pending_action.get('pending_action_data', {})
        action_kind = action_metadata.get('action_kind')
        action_data = action_metadata.get('action_data', {})
        if user_confirmed:
            if action_kind == 'github_create_issue':
                issue_number = await self.github.create_issue(title=action_data['title'], body=action_data['body'], owner=action_data['owner'], repo_name=action_data['repo_name'], github_token=github_token)
                url = f'https://github.com/{action_data['owner']}/{action_data['repo_name']}/issues/{issue_number}'
                content = f'Issue created: {url}'
                action_type = ActionType.GITHUB_CREATE_ISSUE.value
                assistant_message = f'✓ GitHub issue created successfully!\n\nYou can view it here: {url}'
            elif action_kind == 'github_create_pr':
                pr_number = await self.github.create_pr_with_file_changes(title=action_data['title'], body=action_data['body'], owner=action_data['owner'], repo_name=action_data['repo_name'], base_branch=action_data['base_branch'], head_branch=action_data['head_branch'], file_path_to_change=action_data['file_path_to_change'], file_content_to_change=action_data['file_content_to_change'], commit_message=action_data['commit_message'], github_token=github_token)
                url = f'https://github.com/{action_data['owner']}/{action_data['repo_name']}/pull/{pr_number}'
                content = f'PR created: {url}'
                action_type = ActionType.GITHUB_CREATE_PR.value
                assistant_message = f'✓ Pull request created successfully!\n\nYou can view it here: {url}'
            else:
                content = 'Unknown action type'
                action_type = ActionType.AGENT_CHAT.value
                assistant_message = 'Error: Unknown action type'
        else:
            if action_kind == 'github_create_issue':
                action_display_name = 'GitHub issue'
            else:
                action_display_name = 'GitHub PR'
            content = f'{action_display_name} creation cancelled by user.'
            action_type = ActionType.AGENT_CHAT.value
            assistant_message = f'{action_display_name} creation cancelled.'
        await self.db_client.update_chat_record_status(chat_id=chat_id, timestamp=pending_action['timestamp'], status=ActionStatus.SUCCESS.value if user_confirmed else ActionStatus.CANCELLED.value, content=content, action_type=action_type, user_confirmation=user_confirmed)
        response_time = datetime.now().astimezone(timezone.utc)
        await self.db_client.insert_chat_record(message={'chat_id': chat_id, 'timestamp': response_time, 'role': 'assistant', 'content': assistant_message, 'reference': [], 'trace_id': trace_id, 'chunk_id': 0, 'action_type': ActionType.AGENT_CHAT.value, 'status': ActionStatus.SUCCESS.value})
        return {'time': response_time.timestamp() * 1000, 'message': assistant_message, 'reference': [], 'message_type': MessageType.ASSISTANT.value, 'chat_id': chat_id}

class TelemetryLogic:
    """Business logic for telemetry (trace and log) operations."""

    def __init__(self, local_mode: bool):
        """Initialize telemetry driver.

        Args:
            local_mode: Whether running in local mode (SQLite + Jaeger) or cloud mode
        """
        self.local_mode = local_mode
        self.logger = logging.getLogger(__name__)
        if self.local_mode:
            self.db_client = TraceRootSQLiteClient()
        else:
            self.db_client = TraceRootMongoDBClient()
        if self.local_mode:
            self.default_observe_provider = ObservabilityProvider.create_jaeger_provider()
        else:
            self.default_observe_provider = ObservabilityProvider.create_aws_provider()
        self.cache = Cache()
        self.cache_helper = TraceCacheHelper(self.cache)

    async def list_traces(self, request: Request, req_data: ListTraceRequest, user_sub: str) -> dict[str, Any]:
        """Business logic for listing traces.

        Handles three main use cases:
        1. Direct trace ID lookup - returns single trace (optimized path)
        2. Log search filtering - searches logs then fetches matching traces
        3. Normal trace filtering - fetches traces with optional filters

        Args:
            request: FastAPI request object
            req_data: Parsed request data
            user_sub: User's subject ID for log grouping

        Returns:
            Dictionary containing list of trace data

        Raises:
            ValueError: If request parameters are invalid
        """
        log_group_name = hash_user_sub(user_sub)
        filter_cats = separate_filter_categories(req_data.categories.copy(), req_data.values.copy(), req_data.operations.copy())
        if req_data.trace_id:
            return await self._get_single_trace_by_id(request, req_data.trace_id, filter_cats)
        pagination_state = PaginationHelper.decode(req_data.pagination_token)
        cache_key = self.cache_helper.build_cache_key(req_data.start_time, req_data.end_time, req_data.categories, req_data.values, req_data.operations, log_group_name, req_data.pagination_token)
        cached_result = await self.cache_helper.get_traces(cache_key)
        if cached_result:
            traces, next_state = cached_result
            next_token = PaginationHelper.encode(next_state)
            return TraceQueryHelper.format_response(traces, next_token)
        observe_provider = await get_observe_provider(request=request, db_client=self.db_client, local_mode=self.local_mode, default_provider=self.default_observe_provider)
        if filter_cats.has_log_search:
            log_search_values = filter_cats.log_search_values
            trace_provider = request.query_params.get('trace_provider', 'aws')
            traces, next_state = await self._get_traces_by_log_search_paginated(observe_provider=observe_provider, start_time=req_data.start_time, end_time=req_data.end_time, log_group_name=log_group_name, log_search_values=log_search_values, categories=filter_cats.remaining_categories, values=filter_cats.remaining_values, operations=filter_cats.remaining_operations, pagination_state=pagination_state, trace_provider=trace_provider)
        else:
            traces, next_state = await observe_provider.trace_client.get_recent_traces(start_time=req_data.start_time, end_time=req_data.end_time, log_group_name=log_group_name, service_name_values=filter_cats.service_name_values, service_name_operations=filter_cats.service_name_operations, service_environment_values=filter_cats.service_environment_values, service_environment_operations=filter_cats.service_environment_operations, categories=filter_cats.remaining_categories, values=filter_cats.remaining_values, operations=filter_cats.remaining_operations, pagination_state=pagination_state)
        await self.cache_helper.cache_traces(cache_key, traces, next_state)
        next_token = PaginationHelper.encode(next_state)
        return TraceQueryHelper.format_response(traces, next_token)

    async def get_logs_by_trace_id(self, request: Request, trace_id: str, start_time: datetime | None, end_time: datetime | None, user_sub: str) -> dict[str, Any]:
        """Business logic for getting logs by trace ID.

        Optimizes log queries by using trace timestamps when available.
        Handles special case of LimitExceeded traces.

        Args:
            request: FastAPI request object
            trace_id: Trace ID to fetch logs for
            start_time: Optional start time (if None, infer from trace)
            end_time: Optional end time (if None, infer from trace)
            user_sub: User's subject ID for log grouping

        Returns:
            Dictionary containing trace logs

        Raises:
            ValueError: If request parameters are invalid
        """
        log_group_name = hash_user_sub(user_sub)
        observe_provider = await get_observe_provider(request=request, db_client=self.db_client, local_mode=self.local_mode, default_provider=self.default_observe_provider)
        log_start_time = start_time
        log_end_time = end_time
        if log_start_time is None or log_end_time is None:
            trace = await observe_provider.trace_client.get_trace_by_id(trace_id=trace_id, categories=None, values=None, operations=None)
            if trace:
                if trace.service_name == 'LimitExceeded' and trace.start_time == 0.0 and (trace.end_time == 0.0):
                    try:
                        log_client = observe_provider.log_client
                        earliest, latest = await log_client.get_log_timestamps_by_trace_id(trace_id=trace_id, log_group_name=log_group_name, start_time=start_time, end_time=end_time)
                        if earliest and latest:
                            log_start_time = earliest
                            log_end_time = latest
                    except Exception as e:
                        self.logger.error(f'Failed to get log timestamps for LimitExceeded trace {trace_id}: {e}')
                else:
                    log_start_time = datetime.fromtimestamp(trace.start_time, tz=timezone.utc)
                    log_end_time = datetime.fromtimestamp(trace.end_time, tz=timezone.utc)
        log_cache_key = self.cache_helper.build_log_cache_key(trace_id, log_start_time, log_end_time, log_group_name)
        cached_logs: TraceLogs | None = await self.cache_helper.get_logs(log_cache_key)
        if cached_logs:
            resp = GetLogByTraceIdResponse(trace_id=trace_id, logs=cached_logs)
            return resp.model_dump()
        logs: TraceLogs = await observe_provider.log_client.get_logs_by_trace_id(trace_id=trace_id, start_time=log_start_time, end_time=log_end_time, log_group_name=log_group_name)
        await self.cache_helper.cache_logs(log_cache_key, logs)
        resp = GetLogByTraceIdResponse(trace_id=trace_id, logs=logs)
        return resp.model_dump()

    async def _get_single_trace_by_id(self, request: Request, trace_id: str, filter_cats: FilterCategories) -> dict[str, Any]:
        """Fetch a single trace by ID directly.

        This is an optimized path when user requests a specific trace.

        Args:
            request: FastAPI request object
            trace_id: Trace ID to fetch
            filter_cats: Filter categories to apply

        Returns:
            ListTraceResponse dict
        """
        observe_provider = await get_observe_provider(request=request, db_client=self.db_client, local_mode=self.local_mode, default_provider=self.default_observe_provider)
        trace = await observe_provider.trace_client.get_trace_by_id(trace_id=trace_id, categories=filter_cats.remaining_categories, values=filter_cats.remaining_values, operations=filter_cats.remaining_operations)
        if trace is None:
            resp = ListTraceResponse(traces=[])
            return resp.model_dump()
        resp = ListTraceResponse(traces=[trace])
        return resp.model_dump()

    async def _get_traces_by_log_search_paginated(self, observe_provider: ObservabilityProvider, start_time: datetime, end_time: datetime, log_group_name: str, log_search_values: list[str], categories: list[str] | None=None, values: list[str] | None=None, operations: list[Operation] | None=None, pagination_state: dict | None=None, page_size: int=50, trace_provider: str='aws') -> tuple[list[Trace], dict | None]:
        """Get traces matching log search criteria with pagination support.

        Private method - orchestrates cache, log search, and trace fetching.

        This method implements pagination by:
        1. First request: Query CloudWatch for ALL trace IDs and cache them
        2. Subsequent requests: Use cached trace IDs
        3. Fetch only a batch of traces per request

        ORDERING STRATEGY:
        Trace IDs are returned from CloudWatch sorted by LOG timestamp (newest first),
        NOT by span start_time. This is a deliberate trade-off:
        - Pro: Fast and cheap (single CloudWatch query, no X-Ray calls)
        - Con: Log timestamp ≠ span start time (usually close, but can differ)
        - Result: 99% of traces appear in chronological order, occasional outliers

        CRITICAL: The trace ID list from CloudWatch MUST preserve order for pagination
        to work correctly. See aws_log_client.py for implementation details.

        Args:
            observe_provider: Observability provider instance
            start_time: Start time for log query
            end_time: End time for log query
            log_group_name: Log group name
            log_search_values: List of search terms to look for in logs
            categories: Filter by categories if provided
            values: Filter by values if provided
            operations: Filter by operations if provided
            pagination_state: State from previous request (contains cache_key and offset)
            page_size: Number of traces to return per page
            trace_provider: Trace provider type (aws, tencent, jaeger)

        Returns:
            Tuple of (traces, next_pagination_state)
        """
        if not log_search_values:
            return ([], None)
        search_term = log_search_values[0]
        try:
            import hashlib
            cache_params = f'{start_time.isoformat()}_{end_time.isoformat()}_{log_group_name}_{search_term}'
            cache_key = f'log_search_trace_ids:{hashlib.md5(cache_params.encode()).hexdigest()}'
            if PaginationHelper.is_log_search(pagination_state):
                offset = pagination_state.get('offset', 0)
                cached_metadata = await self.cache_helper.get_trace_metadata(cache_key)
                if cached_metadata:
                    trace_id_to_metadata = cached_metadata
                else:
                    trace_id_to_metadata = await observe_provider.log_client.get_trace_metadata_from_logs(start_time=start_time, end_time=end_time, log_group_name=log_group_name, search_term=search_term)
                    await self.cache_helper.cache_trace_metadata(cache_key, trace_id_to_metadata)
            else:
                offset = 0
                trace_id_to_metadata = await observe_provider.log_client.get_trace_metadata_from_logs(start_time=start_time, end_time=end_time, log_group_name=log_group_name, search_term=search_term)
                await self.cache_helper.cache_trace_metadata(cache_key, trace_id_to_metadata)
            if not trace_id_to_metadata:
                return ([], None)
            traces, final_offset = await self._fetch_trace_batch_with_filters(observe_provider=observe_provider, trace_id_to_metadata=trace_id_to_metadata, offset=offset, page_size=page_size, categories=categories, values=values, operations=operations)
            self.logger.info(f'Successfully fetched {len(traces)} traces from log search')
            traces.sort(key=lambda t: t.start_time, reverse=True)
            next_offset = final_offset + page_size
            self.logger.debug(f'Calculating next state: next_offset={next_offset}, total_traces={len(trace_id_to_metadata)}')
            if next_offset < len(trace_id_to_metadata):
                next_state = PaginationHelper.create_log_search_state(offset=next_offset, search_term=search_term, cache_key=cache_key, provider=trace_provider)
            else:
                next_state = None
            return (traces, next_state)
        except Exception as e:
            self.logger.error(f'Failed to get traces by log search: {e}')
            return ([], None)

    async def _fetch_trace_batch_with_filters(self, observe_provider: ObservabilityProvider, trace_id_to_metadata: dict[str, dict], offset: int, page_size: int, categories: list[str] | None=None, values: list[str] | None=None, operations: list[Operation] | None=None) -> tuple[list[Trace], int]:
        """Fetch a batch of traces from log search results, applying categorical filters.

        This method handles pagination through trace IDs found in log search. When
        categorical filters are applied, it may need to fetch multiple batches to
        find matching traces (since filtering happens in get_trace_by_id).

        Args:
            observe_provider: Observability provider instance
            trace_id_to_metadata: Dict mapping trace_id to metadata
                {
                    'trace-id-1': {
                        'start_time': datetime,
                        'end_time': datetime,
                        'log_stream': str
                    }
                }
            offset: Starting offset in trace ID list
            page_size: Number of traces to fetch per batch
            categories: Filter by categories if provided
            values: Filter by values if provided
            operations: Filter by operations if provided

        Returns:
            Tuple of (matching traces, final offset used)

        Note:
            When categorical filters result in 0 matches for a batch, this method
            automatically fetches the next batch until matches are found or all
            trace IDs are exhausted.
        """
        has_categorical_filter = categories is not None and values is not None
        trace_ids_ordered = list(trace_id_to_metadata.keys())
        traces = []
        current_offset = offset
        batches_tried = 0
        while True:
            current_batch = trace_ids_ordered[current_offset:current_offset + page_size]
            if not current_batch:
                self.logger.debug(f'No more trace IDs at offset {current_offset}')
                break
            batches_tried += 1
            self.logger.debug(f'Batch {batches_tried}: Fetching {len(current_batch)} traces from offset {current_offset}...')
            batch_traces = []
            for i, trace_id in enumerate(current_batch):
                self.logger.debug(f'Fetching trace {i + 1}/{len(current_batch)}: {trace_id}')
                metadata = trace_id_to_metadata.get(trace_id, {})
                log_start_time = metadata.get('start_time')
                log_end_time = metadata.get('end_time')
                log_stream = metadata.get('log_stream')
                self.logger.debug(f'[_fetch_trace_batch] Trace {trace_id}: start_time={log_start_time}, end_time={log_end_time}, log_stream={log_stream}')
                trace = await observe_provider.trace_client.get_trace_by_id(trace_id=trace_id, categories=categories, values=values, operations=operations, log_start_time=log_start_time, log_end_time=log_end_time, log_stream=log_stream)
                if trace:
                    batch_traces.append(trace)
                    self.logger.debug('Trace fetched successfully')
                else:
                    self.logger.debug('Trace not found or filtered out')
            traces.extend(batch_traces)
            self.logger.debug(f'Batch {batches_tried} yielded {len(batch_traces)} matching traces')
            if len(traces) > 0 or not has_categorical_filter or current_offset + page_size > len(trace_ids_ordered):
                if batches_tried > 1 and len(traces) > 0:
                    self.logger.info(f'✓ Found {len(traces)} matching traces after checking {batches_tried} batches')
                break
            self.logger.debug(f'⚠️ Batch {batches_tried}: Categorical filter found 0 matches, trying next batch...')
            current_offset += page_size
        self.logger.info(f'Successfully fetched {len(traces)} traces total (tried {batches_tried} batches)')
        return (traces, current_offset)

class InternalRouter:
    """Internal router for OTLP usage tracking."""

    def __init__(self):
        self.router = APIRouter()
        self.db_client = TraceRootMongoDBClient()
        self._setup_routes()

    def _setup_routes(self):
        """Set up internal API routes."""
        self.router.post('/v1/traces')(self.receive_traces)

    async def receive_traces(self, request: Request) -> dict[str, str]:
        """Receive OTLP traces from collector and track usage.

        Args:
            request: FastAPI request with OTLP protobuf payload

        Returns:
            Status response
        """
        try:
            body = await request.body()
            content_type = request.headers.get('content-type', '').lower()
            content_encoding = request.headers.get('content-encoding', '').lower()
            if content_encoding == 'gzip':
                body = gzip.decompress(body)
            export_request = ExportTraceServiceRequest()
            if 'application/json' in content_type:
                json_data = json.loads(body)
                json_format.ParseDict(json_data, export_request)
            else:
                export_request.ParseFromString(body)
            for resource_span in export_request.resource_spans:
                for scope_span in resource_span.scope_spans:
                    for span in scope_span.spans:
                        user_hash = None
                        span_log_count = 0
                        for attr in span.attributes:
                            if attr.key == 'hash':
                                user_hash = attr.value.string_value
                            elif attr.key in ['num_debug_logs', 'num_info_logs', 'num_warning_logs', 'num_error_logs', 'num_critical_logs']:
                                span_log_count += attr.value.int_value
                        if user_hash:
                            usage_buffer[user_hash]['traces'] += 1
                            usage_buffer[user_hash]['logs'] += span_log_count
            await self._maybe_flush_usage()
            return {'status': 'ok'}
        except Exception as e:
            logger.error(f'Error receiving OTLP traces: {e}', exc_info=True)
            return {'status': 'error', 'message': str(e)}

    async def _maybe_flush_usage(self) -> None:
        """Flush usage buffer to Autumn if interval has passed."""
        global last_flush_time
        now = datetime.now()
        time_since_flush = (now - last_flush_time).total_seconds()
        if time_since_flush < FLUSH_INTERVAL_SECONDS:
            return
        last_flush_time = now
        if not usage_buffer:
            return
        current_usage = dict(usage_buffer)
        usage_buffer.clear()
        tracker = get_traces_and_logs_tracker()
        for user_hash, counts in current_usage.items():
            try:
                user_sub = await self.db_client.get_user_sub_by_hash(user_hash)
                if not user_sub:
                    logger.warning(f'Could not find user_sub for hash {user_hash[:16]}..., skipping')
                    continue
                await tracker.track_traces_and_logs(customer_id=user_sub, trace_count=counts['traces'], log_count=counts['logs'])
            except Exception as e:
                logger.error(f'Failed to track usage for {user_hash[:16]}...: {e}', exc_info=True)

