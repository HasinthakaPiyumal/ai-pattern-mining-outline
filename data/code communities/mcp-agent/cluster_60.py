# Cluster 60

def get_logger(namespace: str, session_id: str | None=None, context=None) -> Logger:
    """
    Get a logger instance for a given namespace.
    Creates a new logger if one doesn't exist for this namespace.

    Args:
        namespace: The namespace for the logger (e.g. "agent.helper", "workflow.demo")
        session_id: Optional session ID to associate with all events from this logger
        context: Deprecated/ignored. Present for backwards compatibility.

    Returns:
        A Logger instance for the given namespace
    """
    with _logger_lock:
        existing = _loggers.get(namespace)
        if existing is None:
            bound_ctx = context if context is not None else _default_bound_context
            logger = Logger(namespace, session_id, bound_ctx)
            _loggers[namespace] = logger
            return logger
        if session_id is not None:
            existing.session_id = session_id
        if context is not None:
            existing._bound_context = context
        return existing

class TokenCounter:
    """
    Hierarchical token counter with cost calculation.
    Tracks token usage across the call stack.
    """

    def __init__(self, execution_engine: Optional[str]=None):
        self._lock = asyncio.Lock()
        self._engine: Optional[str] = execution_engine
        self._is_temporal_engine: bool = execution_engine == 'temporal' if execution_engine is not None else False
        self._root: Optional[TokenNode] = None
        self._context_stack: contextvars.ContextVar[Optional[List[TokenNode]]] = contextvars.ContextVar('token_counter_stack', default=None)
        self._models: List[ModelInfo] = load_default_models()
        self._model_costs = self._build_cost_lookup()
        self._model_lookup = {(model.provider.lower(), model.name.lower()): model for model in self._models}
        self._models_by_provider = self._build_provider_lookup()
        self._model_cache: Dict[Tuple[str, Optional[str]], Optional[ModelInfo]] = {}
        self._usage_by_model: Dict[Tuple[str, Optional[str]], TokenUsage] = defaultdict(TokenUsage)
        self._watches: Dict[str, WatchConfig] = {}
        self._node_watches: Dict[int, Set[str]] = defaultdict(set)
        self._callback_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='token-watch')
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        atexit.register(self._cleanup_executor)

    def scope(self, name: str, node_type: str, metadata: Optional[Dict[str, Any]]=None) -> AsyncContextManager[None]:
        """Return an async context manager that pushes/pops a token scope safely.

        Example:
            async with counter.scope("MyAgent", "agent", {"method": "generate"}):
                ...
        """
        counter = self

        class _TokenScope:

            def __init__(self, name: str, node_type: str, metadata: Optional[Dict[str, Any]]=None) -> None:
                self._name = name
                self._node_type = node_type
                self._metadata = metadata or {}
                self._pushed = False

            async def __aenter__(self) -> None:
                try:
                    await counter.push(self._name, self._node_type, self._metadata)
                    self._pushed = True
                except Exception:
                    self._pushed = False

            async def __aexit__(self, exc_type, exc, tb) -> None:
                try:
                    if self._pushed:
                        await counter.pop()
                except Exception:
                    pass
        return _TokenScope(name, node_type, metadata)

    def _get_stack(self) -> List[TokenNode]:
        """Return the current task's stack (never None)."""
        stack = self._context_stack.get()
        return list(stack) if stack else []

    def _set_stack(self, new_stack: List[TokenNode]) -> None:
        """Set the current task's stack. Always pass a new list (no in-place mutation)."""
        self._context_stack.set(list(new_stack))

    def _get_current_node(self) -> Optional[TokenNode]:
        stack = self._get_stack()
        return stack[-1] if stack else None

    @property
    def _stack(self) -> List[TokenNode]:
        return self._get_stack()

    @property
    def _current(self) -> Optional[TokenNode]:
        return self._get_current_node()

    def _build_cost_lookup(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Build lookup table for model costs"""
        cost_lookup: Dict[Tuple[str, str], Dict[str, float]] = {}
        for model in self._models:
            if model.metrics.cost.blended_cost_per_1m is not None:
                blended_cost = model.metrics.cost.blended_cost_per_1m
            elif model.metrics.cost.input_cost_per_1m is not None and model.metrics.cost.output_cost_per_1m is not None:
                blended_cost = (model.metrics.cost.input_cost_per_1m * 3 + model.metrics.cost.output_cost_per_1m) / 4
            else:
                blended_cost = 1.0
            cost_lookup[model.provider.lower(), model.name.lower()] = {'blended_cost_per_1m': blended_cost, 'input_cost_per_1m': model.metrics.cost.input_cost_per_1m, 'output_cost_per_1m': model.metrics.cost.output_cost_per_1m}
        return cost_lookup

    def _build_provider_lookup(self) -> Dict[str, Dict[str, ModelInfo]]:
        """Build lookup table for models by provider"""
        provider_models: Dict[str, Dict[str, ModelInfo]] = {}
        for model in self._models:
            if model.provider not in provider_models:
                provider_models[model.provider] = {}
            provider_models[model.provider][model.name.lower()] = model
        return provider_models

    def find_model_info(self, model_name: str, provider: Optional[str]=None) -> Optional[ModelInfo]:
        """
        Find ModelInfo by name and optionally provider.

        Args:
            model_name: Name of the model
            provider: Optional provider to help disambiguate

        Returns:
            ModelInfo if found, None otherwise
        """
        cache_key = (model_name, provider)
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        def _candidates(name: str, prov: Optional[str]) -> List[str]:
            """Generate candidate normalized name keys for lookup."""
            vals = []
            nl = (name or '').lower()
            if nl:
                vals.append(nl)
                if '/' in nl:
                    vals.append(nl.rsplit('/', 1)[-1])
                if prov:
                    pref = prov.lower() + '_'
                    if nl.startswith(pref):
                        vals.append(nl[len(pref):])
            return list(dict.fromkeys(vals))
        if provider:
            prov_key = provider.lower()
            for cand in _candidates(model_name, provider):
                mi = self._model_lookup.get((prov_key, cand))
                if mi:
                    self._model_cache[cache_key] = mi
                    return mi
        provider_models: Dict[str, ModelInfo] = self._models_by_provider.get(provider, None) if provider else None
        if provider and (not provider_models):
            for key, models in self._models_by_provider.items():
                if key.lower() == provider.lower():
                    provider_models = models
                    break
        if provider_models:
            for cand in _candidates(model_name, provider):
                if cand in provider_models:
                    result = provider_models[cand]
                    self._model_cache[cache_key] = result
                    return result
            best_match = None
            best_match_score = 0
            for known_name, known_model in provider_models.items():
                score = 0
                if model_name.lower() == known_name:
                    score = 1000
                elif known_name.startswith(model_name.lower()):
                    score = 500 + len(model_name) / len(known_name) * 100
                elif model_name.lower() in known_name:
                    score = len(model_name) / len(known_name) * 100
                elif known_name in model_name.lower():
                    score = len(known_name) / len(model_name) * 50
                if score > best_match_score:
                    best_match = known_model
                    best_match_score = score
            if best_match:
                self._model_cache[cache_key] = best_match
                return best_match
        best_match = None
        best_match_score = 0
        for (prov_key, name_key), known_model in self._model_lookup.items():
            score = 0
            if model_name.lower() == name_key:
                score = 1000
            elif name_key.startswith(model_name.lower()):
                score = 500 + len(model_name) / len(name_key) * 100
            elif model_name.lower() in name_key:
                score = len(model_name) / len(name_key) * 100
            elif name_key in model_name.lower():
                score = len(name_key) / len(model_name) * 50
            if score > 0 and provider and (provider.lower() in known_model.provider.lower()):
                score += 50
            if score > best_match_score:
                best_match = known_model
                best_match_score = score
        if best_match:
            self._model_cache[cache_key] = best_match
            return best_match
        self._model_cache[cache_key] = None
        return None

    async def push(self, name: str, node_type: str, metadata: Optional[Dict[str, Any]]=None) -> None:
        """
        Push a new context onto the stack.
        This is called when entering a new scope (app, workflow, agent, etc).
        """
        try:
            async with self._lock:
                node = TokenNode(name=name, node_type=node_type, metadata=metadata or {})
                node._counter = self
                parent = self._get_current_node() or self._root
                if parent:
                    parent.add_child(node)
                else:
                    self._root = node
                stack = self._get_stack()
                stack.append(node)
                self._set_stack(stack)
        except Exception as e:
            logger.error(f'Error in TokenCounter.push: {e}', exc_info=True)

    async def pop(self) -> Optional[TokenNode]:
        """
        Pop the current context from the stack.
        Returns the popped node with aggregated usage.
        """
        try:
            async with self._lock:
                stack = self._get_stack()
                if not stack:
                    logger.warning('Attempted to pop from empty token stack')
                    return None
                node = stack[-1]
                self._set_stack(stack[:-1])
                return node
        except Exception as e:
            logger.error(f'Error in TokenCounter.pop: {e}', exc_info=True)
            return None

    async def record_usage(self, input_tokens: int, output_tokens: int, model_name: Optional[str]=None, provider: Optional[str]=None, model_info: Optional[ModelInfo]=None) -> None:
        """
        Record token usage at the current stack level.
        This is called by AugmentedLLM after each LLM call.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_name: Name of the model (e.g., "gpt-4", "claude-3-opus")
            provider: Optional provider name to help disambiguate models
            model_info: Optional full ModelInfo object with metadata
        """
        try:
            if self._is_temporal_engine:
                try:
                    from temporalio import workflow as _twf
                    if _twf.in_workflow():
                        if _twf.unsafe.is_replaying():
                            return
                except Exception:
                    pass
            input_tokens = int(input_tokens) if input_tokens is not None else 0
            output_tokens = int(output_tokens) if output_tokens is not None else 0
            if not self._get_current_node():
                logger.warning('No current token context; binding to root')
                try:
                    async with self._lock:
                        if not self._root:
                            self._root = TokenNode(name='root', node_type='app')
                            self._root._counter = self
                        self._set_stack([self._root])
                except Exception as e:
                    logger.error(f'Failed to bind to root context: {e}')
                    return
            async with self._lock:
                if model_name and (not model_info):
                    try:
                        model_info = self.find_model_info(model_name, provider)
                    except Exception as e:
                        logger.debug(f'Failed to find model info for {model_name}: {e}')
                current_node = self._get_current_node()
                if current_node and hasattr(current_node, 'usage'):
                    current_node.usage.input_tokens += input_tokens
                    current_node.usage.output_tokens += output_tokens
                    current_node.usage.total_tokens += input_tokens + output_tokens
                    if model_name and (not current_node.usage.model_name):
                        current_node.usage.model_name = model_name
                    if model_info and (not current_node.usage.model_info):
                        current_node.usage.model_info = model_info
                    current_node._cache_valid = False
                    current_node._cached_aggregate = None
                    self._trigger_watches(current_node)
                if model_name:
                    try:
                        provider_key = model_info.provider if model_info and hasattr(model_info, 'provider') else provider
                        usage_key = (model_name, provider_key)
                        model_usage = self._usage_by_model[usage_key]
                        model_usage.input_tokens += input_tokens
                        model_usage.output_tokens += output_tokens
                        model_usage.total_tokens += input_tokens + output_tokens
                        model_usage.model_name = model_name
                        if model_info and (not model_usage.model_info):
                            model_usage.model_info = model_info
                    except Exception as e:
                        logger.error(f'Failed to track global usage: {e}')
        except Exception as e:
            logger.error(f'Error in TokenCounter.record_usage: {e}', exc_info=True)

    def calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int, provider: Optional[str]=None) -> float:
        """Calculate cost for given token usage"""
        try:
            input_tokens = max(0, int(input_tokens) if input_tokens is not None else 0)
            output_tokens = max(0, int(output_tokens) if output_tokens is not None else 0)
            try:
                model_info = self.find_model_info(model_name, provider)
                if model_info:
                    model_name = model_info.name
            except Exception as e:
                logger.debug(f'Failed to find model info: {e}')
            cost_key: Optional[Tuple[str, str]] = None
            if model_name and provider:
                cost_key = (provider.lower(), model_name.lower())
            if model_info:
                cost_key = (model_info.provider.lower(), model_info.name.lower())
            if not cost_key or cost_key not in self._model_costs:
                logger.info(f'Model {model_name} (provider={provider}) not found in costs, using default estimate')
                return (input_tokens + output_tokens) * 0.5 / 1000000
            costs = self._model_costs.get(cost_key, {})
            input_cost_per_1m = costs.get('input_cost_per_1m')
            output_cost_per_1m = costs.get('output_cost_per_1m')
            if input_cost_per_1m is not None and output_cost_per_1m is not None:
                input_cost = input_tokens / 1000000 * input_cost_per_1m
                output_cost = output_tokens / 1000000 * output_cost_per_1m
                total_cost = input_cost + output_cost
                return total_cost
            else:
                total_tokens = input_tokens + output_tokens
                blended_cost_per_1m = costs.get('blended_cost_per_1m', 0.5)
                blended_cost = total_tokens / 1000000 * blended_cost_per_1m
                return blended_cost
        except Exception as e:
            logger.warning(f'Error in TokenCounter.calculate_cost: {e}', exc_info=True)
            return (input_tokens + output_tokens) * 0.5 / 1000000

    async def get_current_path(self) -> List[str]:
        """Get the current task's stack path (e.g., ['app', 'workflow', 'agent'])."""
        async with self._lock:
            stack = self._get_stack()
            return [node.name for node in stack]

    async def get_current_node(self) -> Optional[TokenNode]:
        """Return the current task's token node (top of the stack)."""
        async with self._lock:
            return self._get_current_node()

    async def format_node_tree(self, node: Optional[TokenNode]=None) -> str:
        """Return a human-friendly string of the node tree starting at node (defaults to app root)."""
        async with self._lock:
            start = node or self._root
        if not start:
            return '(no token usage)'
        lines: List[str] = []

        def _walk(n: TokenNode, prefix: str, is_last: bool):
            connector = '└── ' if is_last else '├── '
            usage = n.aggregate_usage()
            line = f'{prefix}{connector}{n.name} [{n.node_type}] — total {usage.total_tokens:,} (in {usage.input_tokens:,} / out {usage.output_tokens:,})'
            lines.append(line)
            child_prefix = prefix + ('    ' if is_last else '│   ')
            for idx, child in enumerate(n.children):
                _walk(child, child_prefix, idx == len(n.children) - 1)
        _walk(start, '', True)
        return '\n'.join(lines)

    async def get_tree(self) -> Optional[Dict[str, Any]]:
        """Get the full token usage tree"""
        async with self._lock:
            if self._root:
                return self._root.to_dict()
            return None

    async def get_summary(self) -> TokenSummary:
        """Get a complete summary of token usage across all models and nodes"""
        try:
            total_cost = 0.0
            model_costs: Dict[str, ModelUsageSummary] = {}
            total_usage = TokenUsage()
            async with self._lock:
                for (model_name, provider_key), usage in self._usage_by_model.items():
                    try:
                        provider = provider_key
                        if provider is None and usage.model_info:
                            provider = getattr(usage.model_info, 'provider', None)
                        cost = self.calculate_cost(model_name, usage.input_tokens, usage.output_tokens, provider)
                        total_cost += cost
                        model_info_dict = None
                        if usage.model_info:
                            try:
                                model_info_dict = {'provider': getattr(usage.model_info, 'provider', None), 'description': getattr(usage.model_info, 'description', None), 'context_window': getattr(usage.model_info, 'context_window', None), 'tool_calling': getattr(usage.model_info, 'tool_calling', None), 'structured_outputs': getattr(usage.model_info, 'structured_outputs', None)}
                            except Exception as e:
                                logger.debug(f'Failed to extract model info: {e}')
                        model_summary = ModelUsageSummary(model_name=model_name, provider=provider, usage=TokenUsageBase(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, total_tokens=usage.total_tokens), cost=cost, model_info=model_info_dict)
                        if provider:
                            summary_key = f'{model_name} ({provider})'
                        else:
                            summary_key = model_name
                        model_costs[summary_key] = model_summary
                    except Exception as e:
                        logger.error(f'Error processing model {model_name}: {e}')
                        continue
                if self._root:
                    try:
                        total_usage = self._root.aggregate_usage()
                    except Exception as e:
                        logger.error(f'Error aggregating total usage: {e}')
            if self._root:
                usage_tree = await self.get_tree()
            else:
                usage_tree = None
            return TokenSummary(usage=TokenUsageBase(input_tokens=total_usage.input_tokens, output_tokens=total_usage.output_tokens, total_tokens=total_usage.total_tokens), cost=total_cost, model_usage=model_costs, usage_tree=usage_tree)
        except Exception as e:
            logger.error(f'Error in get_summary: {e}', exc_info=True)
            return TokenSummary(usage=TokenUsageBase(), cost=0.0, model_usage={}, usage_tree=None)

    async def reset(self) -> None:
        """Reset all token tracking"""
        async with self._lock:
            self._root = None
            self._set_stack([])
            self._usage_by_model.clear()
            self._watches.clear()
            self._node_watches.clear()
            logger.debug('Token counter reset')

    async def find_node(self, name: str, node_type: Optional[str]=None) -> Optional[TokenNode]:
        """
        Find a node by name and optionally type.

        Args:
            name: The name of the node to find
            node_type: Optional node type to filter by

        Returns:
            The first matching node, or None if not found
        """
        async with self._lock:
            if not self._root:
                return None
            return self._find_node_recursive(self._root, name, node_type)

    def _find_node_recursive(self, node: TokenNode, name: str, node_type: Optional[str]=None) -> Optional[TokenNode]:
        """Recursively search for a node"""
        try:
            if node.name == name and (node_type is None or node.node_type == node_type):
                return node
            for child in node.children:
                try:
                    result = self._find_node_recursive(child, name, node_type)
                    if result:
                        return result
                except Exception as e:
                    logger.debug(f'Error searching child node: {e}')
                    continue
            return None
        except Exception as e:
            logger.error(f'Error in _find_node_recursive: {e}')
            return None

    async def find_nodes_by_type(self, node_type: str) -> List[TokenNode]:
        """
        Find all nodes of a specific type.

        Args:
            node_type: The type of nodes to find (e.g., 'agent', 'workflow', 'llm_call')

        Returns:
            List of matching nodes
        """
        async with self._lock:
            if not self._root:
                return []
            nodes = []
            self._find_nodes_by_type_recursive(self._root, node_type, nodes)
            return nodes

    def _find_nodes_by_type_recursive(self, node: TokenNode, node_type: str, nodes: List[TokenNode]) -> None:
        """Recursively collect nodes by type"""
        if node.node_type == node_type:
            nodes.append(node)
        for child in node.children:
            self._find_nodes_by_type_recursive(child, node_type, nodes)

    async def get_node_usage(self, name: str, node_type: Optional[str]=None) -> Optional[TokenUsage]:
        """
        Get aggregated token usage for a specific node (including its children).

        Args:
            name: The name of the node
            node_type: Optional node type to filter by

        Returns:
            Aggregated TokenUsage for the node and its children, or None if not found
        """
        async with self._lock:
            node = self._find_node_recursive(self._root, name, node_type) if self._root else None
            if node:
                return node.aggregate_usage()
            return None

    async def get_node_cost(self, name: str, node_type: Optional[str]=None) -> float:
        """
        Calculate the total cost for a specific node (including its children).

        Args:
            name: The name of the node
            node_type: Optional node type to filter by

        Returns:
            Total cost for the node and its children
        """
        async with self._lock:
            node = self._find_node_recursive(self._root, name, node_type) if self._root else None
            if not node:
                return 0.0
            return self._calculate_node_cost(node)

    def _calculate_node_cost(self, node: TokenNode) -> float:
        """Calculate cost for a node and its children"""
        try:
            total_cost = 0.0
            if node.usage.model_name:
                provider = None
                if node.usage.model_info:
                    provider = getattr(node.usage.model_info, 'provider', None)
                try:
                    cost = self.calculate_cost(node.usage.model_name, node.usage.input_tokens, node.usage.output_tokens, provider)
                    total_cost += cost
                except Exception as e:
                    logger.error(f'Error calculating cost for node {node.name}: {e}')
            for child in node.children:
                try:
                    total_cost += self._calculate_node_cost(child)
                except Exception as e:
                    logger.error(f'Error calculating cost for child {child.name}: {e}')
                    continue
            return total_cost
        except Exception as e:
            logger.error(f'Error in _calculate_node_cost: {e}')
            return 0.0

    async def get_app_usage(self) -> Optional[TokenUsage]:
        """Get total token usage for the entire application (root node)"""
        async with self._lock:
            if self._root:
                return self._root.aggregate_usage()
            return None

    async def get_agent_usage(self, name: str) -> Optional[TokenUsage]:
        """Get token usage for a specific agent"""
        return await self.get_node_usage(name, 'agent')

    async def get_workflow_usage(self, name: str) -> Optional[TokenUsage]:
        """Get token usage for a specific workflow"""
        return await self.get_node_usage(name, 'workflow')

    async def get_current_usage(self) -> Optional[TokenUsage]:
        """Get token usage for the current task's context"""
        async with self._lock:
            current = self._get_current_node()
            if current:
                return current.aggregate_usage()
            return None

    async def get_node_subtree(self, name: str, node_type: Optional[str]=None) -> Optional[TokenNode]:
        """
        Get a node and its entire subtree.

        Args:
            name: The name of the node
            node_type: Optional node type to filter by

        Returns:
            The node with all its children, or None if not found
        """
        return await self.find_node(name, node_type)

    async def find_node_by_metadata(self, metadata_key: str, metadata_value: Any, node_type: Optional[str]=None, return_all_matches: bool=False) -> Optional[TokenNode] | List[TokenNode]:
        """
        Find a node by a specific metadata key-value pair.

        Args:
            metadata_key: The metadata key to search for
            metadata_value: The value to match
            node_type: Optional node type to filter by
            return_all_matches: If True, return all matching nodes; if False, return first match

        Returns:
            If return_all_matches is False: The first matching node, or None if not found
            If return_all_matches is True: List of all matching nodes (empty if none found)
        """
        async with self._lock:
            if not self._root:
                return [] if return_all_matches else None
            matches = []
            self._find_node_by_metadata_recursive(self._root, metadata_key, metadata_value, node_type, matches)
            if return_all_matches:
                return matches
            else:
                return matches[0] if matches else None

    def _find_node_by_metadata_recursive(self, node: TokenNode, metadata_key: str, metadata_value: Any, node_type: Optional[str], matches: List[TokenNode]) -> None:
        """Recursively search for nodes by metadata"""
        try:
            if node_type is None or node.node_type == node_type:
                if hasattr(node, 'metadata') and node.metadata is not None and (metadata_key in node.metadata) and (node.metadata.get(metadata_key) == metadata_value):
                    matches.append(node)
            for child in node.children:
                try:
                    self._find_node_by_metadata_recursive(child, metadata_key, metadata_value, node_type, matches)
                except Exception as e:
                    logger.debug(f'Error searching child node: {e}')
                    continue
        except Exception as e:
            logger.error(f'Error in _find_node_by_metadata_recursive: {e}')

    async def get_app_node(self) -> Optional[TokenNode]:
        """Get the root application node"""
        async with self._lock:
            return self._root if self._root and self._root.node_type == 'app' else None

    async def get_workflow_node(self, name: Optional[str]=None, workflow_id: Optional[str]=None, run_id: Optional[str]=None, return_all_matches: bool=False) -> Optional[TokenNode] | List[TokenNode]:
        """
        Get a specific workflow node.

        Args:
            name: Name of the workflow
            workflow_id: Optional workflow_id to find specific workflow instances
            run_id: Optional run_id to find a specific workflow run (takes precedence)
            return_all_matches: If True, return all matching nodes

        Returns:
            The workflow node(s) if found
        """
        if run_id:
            return await self.find_node_by_metadata('run_id', run_id, 'workflow', return_all_matches)
        elif workflow_id:
            return await self.find_node_by_metadata('workflow_id', workflow_id, 'workflow', return_all_matches)
        elif name:
            if return_all_matches:
                nodes = await self.find_nodes_by_type('workflow')
                return nodes if name == '*' else [n for n in nodes if n.name == name]
            else:
                return await self.find_node(name, 'workflow')
        else:
            return [] if return_all_matches else None

    async def get_agent_node(self, name: str, return_all_matches: bool=False) -> Optional[TokenNode] | List[TokenNode]:
        """
        Get a specific agent (higher-order AugmentedLLM) node.

        Args:
            name: Name of the agent
            return_all_matches: If True, return all matching nodes

        Returns:
            The agent node(s) if found
        """
        if return_all_matches:
            nodes = await self.find_nodes_by_type('agent')
            return [n for n in nodes if n.name == name]
        else:
            return await self.find_node(name, 'agent')

    async def get_llm_node(self, name: str, return_all_matches: bool=False) -> Optional[TokenNode] | List[TokenNode]:
        """
        Get a specific LLM (base AugmentedLLM) node.

        Args:
            name: Name of the LLM
            return_all_matches: If True, return all matching nodes

        Returns:
            The LLM node(s) if found
        """
        if return_all_matches:
            nodes = await self.find_nodes_by_type('llm')
            return [n for n in nodes if n.name == name]
        else:
            return await self.find_node(name, 'llm')

    async def get_node_breakdown(self, name: str, node_type: Optional[str]=None) -> Optional[NodeUsageDetail]:
        """
        Get a detailed breakdown of token usage for a node and its children.

        Args:
            name: The name of the node
            node_type: Optional node type to filter by

        Returns:
            NodeUsageDetail with breakdown by child type and direct children, or None if not found
        """
        async with self._lock:
            node = self._find_node_recursive(self._root, name, node_type) if self._root else None
            if not node:
                return None
            children_by_type: Dict[str, List[TokenNode]] = defaultdict(list)
            for child in node.children:
                children_by_type[child.node_type].append(child)
            usage_by_node_type: Dict[str, NodeTypeUsage] = {}
            for child_type, children in children_by_type.items():
                type_usage = TokenUsage()
                for child in children:
                    child_usage = child.aggregate_usage()
                    type_usage.input_tokens += child_usage.input_tokens
                    type_usage.output_tokens += child_usage.output_tokens
                    type_usage.total_tokens += child_usage.total_tokens
                usage_by_node_type[child_type] = NodeTypeUsage(node_type=child_type, node_count=len(children), usage=TokenUsageBase(input_tokens=type_usage.input_tokens, output_tokens=type_usage.output_tokens, total_tokens=type_usage.total_tokens))
            child_usage: List[NodeSummary] = []
            for child in node.children:
                child_aggregated = child.aggregate_usage()
                child_usage.append(NodeSummary(name=child.name, node_type=child.node_type, usage=TokenUsageBase(input_tokens=child_aggregated.input_tokens, output_tokens=child_aggregated.output_tokens, total_tokens=child_aggregated.total_tokens)))
            aggregated = node.aggregate_usage()
            return NodeUsageDetail(name=node.name, node_type=node.node_type, direct_usage=TokenUsageBase(input_tokens=node.usage.input_tokens, output_tokens=node.usage.output_tokens, total_tokens=node.usage.total_tokens), usage=TokenUsageBase(input_tokens=aggregated.input_tokens, output_tokens=aggregated.output_tokens, total_tokens=aggregated.total_tokens), usage_by_node_type=usage_by_node_type, child_usage=child_usage)

    async def get_agents_breakdown(self) -> Dict[str, TokenUsage]:
        """Get token usage breakdown by agent"""
        agents = await self.find_nodes_by_type('agent')
        breakdown = {}
        for agent in agents:
            usage = agent.aggregate_usage()
            breakdown[agent.name] = usage
        return breakdown

    async def get_workflows_breakdown(self) -> Dict[str, TokenUsage]:
        """Get token usage breakdown by workflow"""
        workflows = await self.find_nodes_by_type('workflow')
        breakdown = {}
        for workflow in workflows:
            usage = workflow.aggregate_usage()
            breakdown[workflow.name] = usage
        return breakdown

    async def get_models_breakdown(self) -> List[ModelUsageDetail]:
        """
        Get detailed breakdown of usage by model.

        Returns:
            List of ModelUsageDetail containing usage details and nodes for each model
        """
        async with self._lock:
            if not self._root:
                return []
            model_nodes: Dict[Tuple[str, Optional[str]], List[TokenNode]] = defaultdict(list)
            self._collect_model_nodes(self._root, model_nodes)
            breakdown: List[ModelUsageDetail] = []
            for (model_name, provider), nodes in model_nodes.items():
                total_input = 0
                total_output = 0
                for node in nodes:
                    total_input += node.usage.input_tokens
                    total_output += node.usage.output_tokens
                total_tokens = total_input + total_output
                total_cost = self.calculate_cost(model_name, total_input, total_output, provider)
                breakdown.append(ModelUsageDetail(model_name=model_name, provider=provider, usage=TokenUsageBase(input_tokens=total_input, output_tokens=total_output, total_tokens=total_tokens), cost=total_cost, model_info=None, nodes=nodes))
            breakdown.sort(key=lambda x: x.total_tokens, reverse=True)
            return breakdown

    def _collect_model_nodes(self, node: TokenNode, model_nodes: Dict[Tuple[str, Optional[str]], List[TokenNode]]) -> None:
        """Recursively collect nodes that have model usage"""
        if node.usage.model_name:
            provider = None
            if node.usage.model_info:
                provider = node.usage.model_info.provider
            key = (node.usage.model_name, provider)
            model_nodes[key].append(node)
        for child in node.children:
            self._collect_model_nodes(child, model_nodes)

    async def watch(self, callback: Union[Callable[[TokenNode, TokenUsage], None], Callable[[TokenNode, TokenUsage], Awaitable[None]]], node: Optional[TokenNode]=None, node_name: Optional[str]=None, node_type: Optional[str]=None, threshold: Optional[int]=None, throttle_ms: Optional[int]=None, include_subtree: bool=True) -> str:
        """
        Watch a node or nodes for token usage changes.

        Args:
            callback: Function called when usage changes: (node, aggregated_usage) -> None
            node: Specific node instance to watch (highest priority)
            node_name: Node name pattern to watch (used if node not provided)
            node_type: Node type to watch (used if node not provided)
            threshold: Only trigger when total tokens exceed this value
            throttle_ms: Minimum milliseconds between callbacks for the same node
            include_subtree: Whether to trigger on subtree changes or just direct usage

        Returns:
            watch_id: Unique identifier for this watch (use to unwatch)

        Examples:
            # Watch a specific node
            watch_id = await counter.watch(callback, node=my_node)

            # Watch all workflow nodes
            watch_id = await counter.watch(callback, node_type="workflow")

            # Watch with threshold
            watch_id = await counter.watch(callback, node_name="my_agent", threshold=1000)
        """
        async with self._lock:
            watch_id = str(uuid.uuid4())
            is_async = asyncio.iscoroutinefunction(callback)
            config = WatchConfig(watch_id=watch_id, callback=callback, node=node, node_name=node_name, node_type=node_type, threshold=threshold, throttle_ms=throttle_ms, include_subtree=include_subtree, is_async=is_async)
            self._watches[watch_id] = config
            if node:
                self._node_watches[id(node)].add(watch_id)
            try:
                self._event_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            logger.debug(f'Added watch {watch_id} for node={node_name}, type={node_type}, async={is_async}')
            return watch_id

    async def unwatch(self, watch_id: str) -> bool:
        """
        Remove a watch.

        Args:
            watch_id: The watch identifier returned by watch()

        Returns:
            True if watch was removed, False if not found
        """
        async with self._lock:
            config = self._watches.pop(watch_id, None)
            if not config:
                return False
            if config.node:
                node_id = id(config.node)
                if node_id in self._node_watches:
                    self._node_watches[node_id].discard(watch_id)
                    if not self._node_watches[node_id]:
                        del self._node_watches[node_id]
            logger.debug(f'Removed watch {watch_id}')
            return True

    def _cleanup_executor(self) -> None:
        """Clean up thread pool executor on shutdown"""
        try:
            self._callback_executor.shutdown(wait=True, cancel_futures=False)
        except Exception as e:
            logger.error(f'Error shutting down callback executor: {e}')

    def _trigger_watches(self, node: TokenNode) -> None:
        """Trigger watches for a node and its ancestors

        Note: This is called from within record_usage which already holds the lock,
        so we don't acquire it again here.
        """
        try:
            callbacks_to_execute: List[Tuple[WatchConfig, TokenNode, TokenUsage]] = []
            current = node
            triggered_nodes = set()
            is_original_node = True
            while current:
                if id(current) in triggered_nodes:
                    break
                triggered_nodes.add(id(current))
                current._cache_valid = False
                current._cached_aggregate = None
                usage = current.aggregate_usage()
                for watch_id, config in self._watches.items():
                    try:
                        if not self._watch_matches_node(config, current):
                            continue
                        if not is_original_node and (not config.include_subtree):
                            continue
                        if config.threshold and usage.total_tokens < config.threshold:
                            continue
                        node_key = f'{id(current)}'
                        if config.throttle_ms:
                            last_triggered = config._last_triggered.get(node_key, 0)
                            now = time.time() * 1000
                            if now - last_triggered < config.throttle_ms:
                                continue
                            config._last_triggered[node_key] = now
                        usage_copy = TokenUsage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, total_tokens=usage.total_tokens, model_name=usage.model_name, model_info=usage.model_info)
                        callbacks_to_execute.append((config, current, usage_copy))
                        logger.debug(f'Queued watch {config.watch_id} for {current.name} ({current.node_type}) with {usage_copy.total_tokens} tokens')
                    except Exception as e:
                        logger.error(f'Error processing watch {watch_id}: {e}')
                current = current.parent
                is_original_node = False
            for config, callback_node, callback_usage in callbacks_to_execute:
                self._execute_callback(config, callback_node, callback_usage)
        except Exception as e:
            logger.error(f'Error in _trigger_watches: {e}', exc_info=True)

    def _execute_callback(self, config: WatchConfig, node: TokenNode, usage: TokenUsage) -> None:
        """Execute a callback, detecting async context at runtime"""
        try:
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            if loop and (not loop.is_closed()):
                if config.is_async:
                    task = loop.create_task(self._execute_async_callback_safely(config.callback, node, usage))
                    task.add_done_callback(self._handle_task_exception)
                else:
                    loop.run_in_executor(self._callback_executor, self._execute_callback_safely, config.callback, node, usage)
            elif config.is_async:
                logger.debug(f'Async callback {config.watch_id} called outside event loop context. Executing with asyncio.run in thread pool.')
                self._callback_executor.submit(lambda: asyncio.run(self._execute_async_callback_safely(config.callback, node, usage)))
            else:
                self._callback_executor.submit(self._execute_callback_safely, config.callback, node, usage)
        except Exception as e:
            logger.error(f'Error executing callback: {e}', exc_info=True)

    def _handle_task_exception(self, task: asyncio.Task) -> None:
        """Handle exceptions from async tasks"""
        try:
            task.result()
        except Exception as e:
            logger.error(f'Async task error: {e}', exc_info=True)

    def _execute_callback_safely(self, callback: Callable[[TokenNode, TokenUsage], None], node: TokenNode, usage: TokenUsage) -> None:
        """Execute a sync watch callback safely in thread pool"""
        try:
            callback(node, usage)
        except Exception as e:
            logger.error(f'Watch callback error: {e}', exc_info=True)

    async def _execute_async_callback_safely(self, callback: Callable[[TokenNode, TokenUsage], Awaitable[None]], node: TokenNode, usage: TokenUsage) -> None:
        """Execute an async watch callback safely"""
        try:
            await callback(node, usage)
        except Exception as e:
            logger.error(f'Async watch callback error: {e}', exc_info=True)

    def _watch_matches_node(self, config: WatchConfig, node: TokenNode) -> bool:
        """Check if a watch configuration matches a specific node"""
        if config.node:
            return config.node is node
        if config.node_type and node.node_type != config.node_type:
            return False
        if config.node_name and node.name != config.node_name:
            return False
        return True

@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context, version: Optional[bool]=typer.Option(None, '--version', '-v', help='Show version and exit', is_flag=True)) -> None:
    """MCP Agent Cloud CLI."""
    if version:
        v = metadata_version('mcp-agent')
        typer.echo(f'MCP Agent Cloud CLI version: {v}')
        raise typer.Exit()

def _process_slash_command(input_text: str) -> Optional[SlashCommandResult]:
    """Detect and map slash commands to actions."""
    if not input_text.startswith('/'):
        return None
    cmd = input_text.strip().lower()
    action = {'/decline': 'decline', '/cancel': 'cancel', '/help': 'help'}.get(cmd, 'unknown' if cmd != '/' else 'help')
    if action == 'unknown':
        console.print(f'\n[red]Unknown command: {cmd}[/red]')
        console.print('[dim]Type /help for available commands[/dim]\n')
    return SlashCommandResult(cmd, action)

def _print_slash_help() -> None:
    """Display available slash commands."""
    console.print('\n[cyan]Available commands:[/cyan]')
    for cmd, desc in SLASH_COMMANDS.items():
        console.print(f'  [green]{cmd}[/green] - {desc}')
    console.print()

def _process_field_value(field_type: str, value: str) -> Any:
    if field_type == 'boolean':
        v = value.lower()
        if v in ('true', 'yes', 'y', '1'):
            return True
        if v in ('false', 'no', 'n', '0'):
            return False
        console.print(f'[red]Invalid boolean value: {value}[/red]')
        return None
    if field_type == 'number':
        try:
            return float(value)
        except ValueError:
            console.print(f'[red]Invalid number: {value}[/red]')
            return None
    if field_type == 'integer':
        try:
            return int(value)
        except ValueError:
            console.print(f'[red]Invalid integer: {value}[/red]')
            return None
    return value

def _create_panel(request: ElicitRequestParams) -> Panel:
    """Generate styled panel for prompts."""
    title = f'ELICITATION RESPONSE NEEDED FROM: {request.server_name}' if request.server_name else 'ELICITATION RESPONSE NEEDED'
    content = f'[bold]Elicitation Request[/bold]\n\n{request.message}'
    content += '\n\n[dim]Type / to see available commands[/dim]'
    return Panel(content, title=title, style='blue', border_style='bold white', padding=(1, 2))

def _format_sampling_request_for_human(params: CreateMessageRequestParams) -> str:
    """Format sampling request for human review"""
    messages_text = ''
    for i, msg in enumerate(params.messages):
        content = msg.content.text if hasattr(msg.content, 'text') else str(msg.content)
        messages_text += f'  Message {i + 1} ({msg.role}): {content[:200]}{('...' if len(content) > 200 else '')}\n'
    system_prompt_display = 'None' if params.systemPrompt is None else f'{params.systemPrompt[:100]}{('...' if len(params.systemPrompt) > 100 else '')}'
    stop_sequences_display = 'None' if params.stopSequences is None else str(params.stopSequences)
    model_preferences_display = 'None'
    if params.modelPreferences is not None:
        prefs = []
        if params.modelPreferences.hints:
            hints = [hint.name for hint in params.modelPreferences.hints if hint.name is not None]
            prefs.append(f'hints: {hints}')
        if params.modelPreferences.costPriority is not None:
            prefs.append(f'cost: {params.modelPreferences.costPriority}')
        if params.modelPreferences.speedPriority is not None:
            prefs.append(f'speed: {params.modelPreferences.speedPriority}')
        if params.modelPreferences.intelligencePriority is not None:
            prefs.append(f'intelligence: {params.modelPreferences.intelligencePriority}')
        model_preferences_display = ', '.join(prefs) if prefs else 'None'
    return f'REQUEST DETAILS:\n- Max Tokens: {params.maxTokens}\n- System Prompt: {system_prompt_display}\n- Temperature: {(params.temperature if params.temperature is not None else 0.7)}\n- Stop Sequences: {stop_sequences_display}\n- Model Preferences: {model_preferences_display}\nMESSAGES:\n{messages_text}'

def _format_sampling_response_for_human(result: CreateMessageResult) -> str:
    """Format sampling response for human review"""
    content = result.content.text if hasattr(result.content, 'text') else str(result.content)
    return f'RESPONSE DETAILS:\n- Model: {result.model}\n- Role: {result.role}\nCONTENT:\n{content}'

def _should_ignore_exception(exc: Exception) -> bool:
    """
    Returns True when the exception represents a non-JSON stdout line that we can
    safely drop.
    """
    if not isinstance(exc, ValidationError):
        return False
    errors: Iterable[dict] = exc.errors()
    first = next(iter(errors), None)
    if not first or first.get('type') != 'json_invalid':
        return False
    input_value = first.get('input')
    if not isinstance(input_value, str):
        return False
    stripped = input_value.strip()
    if not stripped:
        return True
    first_char = stripped[0]
    lowered = stripped.lower()
    if first_char in _MESSAGE_START_CHARS or any((lowered.startswith(prefix) for prefix in _LITERAL_PREFIXES)):
        return False
    return True

def _truncate(value: str, length: int=120) -> str:
    """
    Truncate long log lines so debug output remains readable.
    """
    if len(value) <= length:
        return value
    return value[:length - 3] + '...'

class MCPAggregator(ContextDependent):
    """
    Aggregates multiple MCP servers. When a developer calls, e.g. call_tool(...),
    the aggregator searches all servers in its list for a server that provides that tool.
    """
    initialized: bool = False
    'Whether the aggregator has been initialized with tools and resources from all servers.'
    connection_persistence: bool = False
    'Whether to maintain a persistent connection to the server.'
    server_names: List[str]
    'A list of server names to connect to.'

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def __init__(self, server_names: List[str], connection_persistence: bool=True, context: Optional['Context']=None, name: str=None, **kwargs):
        """
        :param server_names: A list of server names to connect to.
        :param connection_persistence: Whether to maintain persistent connections to servers (default: True).
        Note: The server names must be resolvable by the gen_client function, and specified in the server registry.
        """
        super().__init__(context=context, **kwargs)
        self.server_names = server_names
        self.connection_persistence = connection_persistence
        self.agent_name = name
        self._persistent_connection_manager: MCPConnectionManager = None
        global logger
        logger_name = f'{__name__}.{name}' if name else __name__
        logger = get_logger(logger_name)
        self._namespaced_tool_map: Dict[str, NamespacedTool] = {}
        self._server_to_tool_map: Dict[str, List[NamespacedTool]] = {}
        self._tool_map_lock = asyncio.Lock()
        self._namespaced_prompt_map: Dict[str, NamespacedPrompt] = {}
        self._server_to_prompt_map: Dict[str, List[NamespacedPrompt]] = {}
        self._prompt_map_lock = asyncio.Lock()
        self._namespaced_resource_map: Dict[str, NamespacedResource] = {}
        self._server_to_resource_map: Dict[str, List[NamespacedResource]] = {}
        self._resource_map_lock = asyncio.Lock()

    async def initialize(self, force: bool=False):
        """Initialize the application."""
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.initialize') as span:
            span.set_attribute('server_names', self.server_names)
            span.set_attribute('force', force)
            span.set_attribute('connection_persistence', self.connection_persistence)
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent_name)
            span.set_attribute('initialized', self.initialized)
            if self.initialized and (not force):
                return
            if self.connection_persistence:
                connection_manager: MCPConnectionManager | None = None
                if not hasattr(self.context, '_mcp_connection_manager_lock'):
                    self.context._mcp_connection_manager_lock = asyncio.Lock()
                if not hasattr(self.context, '_mcp_connection_manager_ref_count'):
                    self.context._mcp_connection_manager_ref_count = int(0)
                async with self.context._mcp_connection_manager_lock:
                    self.context._mcp_connection_manager_ref_count += 1
                    if hasattr(self.context, '_mcp_connection_manager'):
                        connection_manager = self.context._mcp_connection_manager
                    else:
                        connection_manager = MCPConnectionManager(self.context.server_registry)
                        await connection_manager.__aenter__()
                        self.context._mcp_connection_manager = connection_manager
                    self._persistent_connection_manager = connection_manager
            await self.load_servers()
            span.add_event('initialized')
            self.initialized = True

    async def close(self):
        """
        Close all persistent connections when the aggregator is deleted.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.close') as span:
            span.set_attribute('server_names', self.server_names)
            span.set_attribute('connection_persistence', self.connection_persistence)
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent_name)
            if not self.connection_persistence or not self._persistent_connection_manager:
                self.initialized = False
                return
            try:
                if hasattr(self.context, '_mcp_connection_manager_lock') and hasattr(self.context, '_mcp_connection_manager_ref_count'):
                    async with self.context._mcp_connection_manager_lock:
                        self.context._mcp_connection_manager_ref_count -= 1
                        current_count = self.context._mcp_connection_manager_ref_count
                        logger.debug(f'Decremented connection ref count to {current_count}')
                        if current_count == 0:
                            logger.info('Last aggregator closing, shutting down all persistent connections...')
                            if hasattr(self.context, '_mcp_connection_manager') and self.context._mcp_connection_manager == self._persistent_connection_manager:
                                try:
                                    await asyncio.wait_for(self._persistent_connection_manager.close(), timeout=5.0)
                                except asyncio.TimeoutError:
                                    logger.warning('Timeout during connection manager close(), forcing shutdown')
                                except Exception as e:
                                    logger.warning(f'Error during connection manager close(): {e}')
                                delattr(self.context, '_mcp_connection_manager')
                                logger.info('Connection manager successfully closed and removed from context')
                        else:
                            logger.debug(f'Aggregator closing with ref count {current_count}, connection manager will remain active')
            except Exception as e:
                logger.error(f'Error during connection manager cleanup: {e}', exc_info=True)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                span.record_exception(e)
            finally:
                self.initialized = False

    @classmethod
    async def create(cls, server_names: List[str], connection_persistence: bool=False) -> 'MCPAggregator':
        """
        Factory method to create and initialize an MCPAggregator.
        Use this instead of constructor since we need async initialization.
        If connection_persistence is True, the aggregator will maintain a
        persistent connection to the servers for as long as this aggregator is around.
        By default we do not maintain a persistent connection.
        """
        logger.info(f'Creating MCPAggregator with servers: {server_names}')
        instance = cls(server_names=server_names, connection_persistence=connection_persistence)
        tracer = get_tracer(instance.context)
        with tracer.start_as_current_span(f'{cls.__name__}.create') as span:
            span.set_attribute('server_names', server_names)
            span.set_attribute('connection_persistence', connection_persistence)
            try:
                await instance.__aenter__()
                logger.debug('Loading servers...')
                await instance.load_servers()
                logger.debug('MCPAggregator created and initialized.')
                return instance
            except Exception as e:
                logger.error(f'Error creating MCPAggregator: {e}')
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                span.record_exception(e)
                try:
                    await instance.__aexit__(None, None, None)
                except Exception as cleanup_error:
                    logger.warning(f'Error during MCPAggregator cleanup: {cleanup_error}')

    async def load_server(self, server_name: str):
        """
        Load tools and prompts from a single server and update the index of namespaced tool/prompt names for that server.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.load_server') as span:
            span.set_attribute('server_name', server_name)
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent_name)
            if server_name not in self.server_names:
                raise ValueError(f"Server '{server_name}' not found in server list")
            _, tools, prompts, resources = await self._fetch_capabilities(server_name)
            async with self._tool_map_lock:
                self._server_to_tool_map[server_name] = []
                allowed_tools = None
                disabled_tool_count = 0
                if self.context is None or self.context.server_registry is None or (not hasattr(self.context.server_registry, 'get_server_config')):
                    logger.warning(f"No config found for server '{server_name}', no tool filter will be applied...")
                else:
                    allowed_tools = self.context.server_registry.get_server_config(server_name).allowed_tools
                    if allowed_tools is not None and len(allowed_tools) == 0:
                        logger.warning(f"Allowed tool list is explicitly empty for server '{server_name}'")
                for tool in tools:
                    if allowed_tools is not None and tool.name not in allowed_tools:
                        logger.debug(f"Filtering out tool '{tool.name}' from server '{server_name}' (not in allowed_tools)")
                        disabled_tool_count += 1
                        continue
                    namespaced_tool_name = f'{server_name}{SEP}{tool.name}'
                    namespaced_tool = NamespacedTool(tool=tool, server_name=server_name, namespaced_tool_name=namespaced_tool_name)
                    self._namespaced_tool_map[namespaced_tool_name] = namespaced_tool
                    self._server_to_tool_map[server_name].append(namespaced_tool)
            async with self._prompt_map_lock:
                self._server_to_prompt_map[server_name] = []
                for prompt in prompts:
                    namespaced_prompt_name = f'{server_name}{SEP}{prompt.name}'
                    namespaced_prompt = NamespacedPrompt(prompt=prompt, server_name=server_name, namespaced_prompt_name=namespaced_prompt_name)
                    self._namespaced_prompt_map[namespaced_prompt_name] = namespaced_prompt
                    self._server_to_prompt_map[server_name].append(namespaced_prompt)
            async with self._resource_map_lock:
                self._server_to_resource_map[server_name] = []
                for resource in resources:
                    namespaced_resource_name = f'{server_name}{SEP}{resource.name}'
                    namespaced_resource = NamespacedResource(resource=resource, server_name=server_name, namespaced_resource_name=namespaced_resource_name)
                    self._namespaced_resource_map[namespaced_resource_name] = namespaced_resource
                    self._server_to_resource_map[server_name].append(namespaced_resource)
            event_metadata = {'server_name': server_name, 'agent_name': self.agent_name, 'tool_count': len(tools), 'disabled_tool_count': disabled_tool_count, 'prompt_count': len(prompts), 'resource_count': len(resources)}
            logger.debug(f"MCP Aggregator initialized for server '{server_name}'", data={'progress_action': ProgressAction.INITIALIZED, **event_metadata})
            if self.context.tracing_enabled:
                span.add_event('load_server_complete', event_metadata)
                for tool in tools:
                    span.set_attribute(f'tool.{tool.name}', tool.description or 'No description')
                for prompt in prompts:
                    span.set_attribute(f'prompt.{prompt.name}', prompt.description or 'No description')
                for resource in resources:
                    span.set_attribute(f'resource.{resource.name}', resource.description or 'No description')
            return (tools, prompts, resources)

    async def load_servers(self, force: bool=False):
        """
        Discover tools and prompts from each server in parallel and build an index of namespaced tool/prompt names.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.load_servers') as span:
            span.set_attribute('server_names', self.server_names)
            span.set_attribute('force', force)
            span.set_attribute('connection_persistence', self.connection_persistence)
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent_name)
            span.set_attribute('initialized', self.initialized)
            if self.initialized and (not force):
                logger.debug('MCPAggregator already initialized. Skipping reload.')
                return
            async with self._tool_map_lock:
                self._namespaced_tool_map.clear()
                self._server_to_tool_map.clear()
            async with self._prompt_map_lock:
                self._namespaced_prompt_map.clear()
                self._server_to_prompt_map.clear()
            async with self._resource_map_lock:
                self._namespaced_resource_map.clear()
                self._server_to_resource_map.clear()
            results = await asyncio.gather(*(self.load_server(server_name) for server_name in self.server_names), return_exceptions=True)
            for server_name, result in zip(self.server_names, results):
                if isinstance(result, BaseException):
                    logger.error(f'Error loading server data: {result}. Attempting to continue')
                    span.record_exception(result, {'server_name': server_name})
                    continue
                else:
                    span.add_event('server_load_success', {'server_name': server_name})
            self.initialized = True

    async def get_server(self, server_name: str) -> Optional[ClientSession]:
        """Get a server connection if available."""
        if self.connection_persistence:
            try:
                server_conn = await self._persistent_connection_manager.get_server(server_name, client_session_factory=MCPAgentClientSession)
                return server_conn.session
            except Exception as e:
                logger.warning(f"Error getting server connection for '{server_name}': {e}")
                return None
        else:
            logger.debug(f'Creating temporary connection to server: {server_name}', data={'progress_action': ProgressAction.STARTING, 'server_name': server_name, 'agent_name': self.agent_name})
            async with gen_client(server_name, server_registry=self.context.server_registry) as client:
                return client

    async def get_capabilities(self, server_name: str):
        """Get server capabilities if available."""
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.get_capabilitites') as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent_name)
            span.set_attribute('server_names', self.server_names)
            span.set_attribute('connection_persistence', self.connection_persistence)
            span.set_attribute('server_name', server_name)

            def _annotate_span_for_capabilities(capabilities: ServerCapabilities):
                if not self.context.tracing_enabled:
                    return
                for attr in ['experimental', 'logging', 'prompts', 'resources', 'tools']:
                    value = getattr(capabilities, attr, None)
                    span.set_attribute(f'{server_name}.capabilities.{attr}', value is not None)
            if self.connection_persistence:
                try:
                    server_conn = await self._persistent_connection_manager.get_server(server_name, client_session_factory=MCPAgentClientSession)
                    res = server_conn.server_capabilities
                    _annotate_span_for_capabilities(res)
                    return res
                except Exception as e:
                    logger.warning(f"Error getting capabilities for server '{server_name}': {e}")
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    span.record_exception(e)
                    return None
            else:
                logger.debug(f'Creating temporary connection to server: {server_name}', data={'progress_action': ProgressAction.STARTING, 'server_name': server_name, 'agent_name': self.agent_name})
                async with self.context.server_registry.start_server(server_name, client_session_factory=MCPAgentClientSession) as session:
                    try:
                        initialize_result = await session.initialize()
                        res = initialize_result.capabilities
                        _annotate_span_for_capabilities(res)
                        return res
                    except Exception as e:
                        logger.warning(f"Error getting capabilities for server '{server_name}': {e}")
                        span.set_status(trace.Status(trace.StatusCode.ERROR))
                        span.record_exception(e)
                        return None

    async def refresh(self, server_name: str | None=None):
        """
        Refresh the tools and prompts from the specified server or all servers.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.refresh') as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent_name)
            if server_name:
                span.set_attribute('server_name', server_name)
                await self.load_server(server_name)
            else:
                await self.load_servers(force=True)

    async def list_servers(self) -> List[str]:
        """Return the list of server names aggregated by this agent."""
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.list_servers') as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent_name)
            span.set_attribute('initialized', self.initialized)
            if not self.initialized:
                await self.load_servers()
            span.set_attribute('server_names', self.server_names)
            return self.server_names

    async def list_tools(self, server_name: str | None=None) -> ListToolsResult:
        """
        :return: Tools from all servers aggregated, and renamed to be dot-namespaced by server name.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.list_tools') as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent_name)
            span.set_attribute('initialized', self.initialized)
            if not self.initialized:
                await self.load_servers()
            if server_name:
                span.set_attribute('server_name', server_name)
                result = ListToolsResult(tools=[namespaced_tool.tool.model_copy(update={'name': namespaced_tool.namespaced_tool_name}) for namespaced_tool in self._server_to_tool_map.get(server_name, [])])
            else:
                async with self._tool_map_lock:
                    result = ListToolsResult(tools=[namespaced_tool.tool.model_copy(update={'name': namespaced_tool_name}) for namespaced_tool_name, namespaced_tool in self._namespaced_tool_map.items()])
            if self.context.tracing_enabled:
                span.set_attribute('tool_count', len(result.tools))
                for tool in result.tools:
                    span.set_attribute(f'tool.{tool.name}', tool.description or 'No description')
            return result

    async def list_resources(self, server_name: str | None=None):
        """
        :return: Resources from all servers aggregated, and renamed to be dot-namespaced by server name.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.list_resources') as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent_name)
            span.set_attribute('initialized', self.initialized)
            if not self.initialized:
                await self.load_servers()
            if server_name:
                span.set_attribute('server_name', server_name)
                result = ListResourcesResult(resources=[namespaced_resource.resource.model_copy(update={'name': namespaced_resource.namespaced_resource_name}) for namespaced_resource in self._server_to_resource_map.get(server_name, [])])
            else:
                async with self._resource_map_lock:
                    result = ListResourcesResult(resources=[namespaced_resource.resource.model_copy(update={'name': namespaced_resource_name}) for namespaced_resource_name, namespaced_resource in self._namespaced_resource_map.items()])
            if self.context.tracing_enabled:
                span.set_attribute('resource_count', len(result.resources))
                for resource in result.resources:
                    span.set_attribute(f'resource.{resource.name}', resource.description or 'No description')
            return result

    async def read_resource(self, uri: str, server_name: str | None=None) -> ReadResourceResult:
        """
        Read a resource from a server by its URI.

        Args:
            uri: The URI of the resource to read.
            server_name: Optionally restrict search to a specific server.

        Returns:
            Resource object, or None if not found
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.read_resource') as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent_name)
            span.set_attribute('initialized', self.initialized)
            if not self.initialized:
                await self.load_servers()
            span.set_attribute('uri', uri)
            if server_name:
                span.set_attribute('server_name', server_name)
            else:
                server_name, _ = await self._parse_capability_name(uri, 'resource')
                span.set_attribute('parsed_server_name', server_name)
            if server_name is None:
                logger.error(f"Resource with uri '{uri}' not found in any server")
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                span.record_exception(ValueError(f"Resource with uri '{uri}' not found in any server"))
                return ReadResourceResult(contents=[])

            async def try_read_resource(client: ClientSession):
                try:
                    res = await client.read_resource(uri=uri)
                    return res
                except Exception as e:
                    logger.error(f"Error reading resource with uri '{uri}'" + (f" from server '{server_name}'" if server_name else '') + f': {e}')
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    span.record_exception(e)
                    return ReadResourceResult(contents=[])
            if self.connection_persistence:
                server_conn = await self._persistent_connection_manager.get_server(server_name, client_session_factory=MCPAgentClientSession)
                res = await try_read_resource(server_conn.session)
                return res
            else:
                logger.debug(f'Creating temporary connection to server: {server_name}', data={'progress_action': ProgressAction.STARTING, 'server_name': server_name, 'agent_name': self.agent_name})
                span.add_event('temporary_connection_created', {'server_name': server_name, GEN_AI_AGENT_NAME: self.agent_name})
                async with gen_client(server_name, server_registry=self.context.server_registry) as client:
                    result = await try_read_resource(client)
                    logger.debug(f'Closing temporary connection to server: {server_name}', data={'progress_action': ProgressAction.SHUTDOWN, 'server_name': server_name, 'agent_name': self.agent_name})
                    span.add_event('temporary_connection_closed', {'server_name': server_name, GEN_AI_AGENT_NAME: self.agent_name})
                    return result

    async def call_tool(self, name: str, arguments: dict | None=None, server_name: str | None=None) -> CallToolResult:
        """
        Call a namespaced tool, e.g., 'server_name.tool_name'.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.call_tool') as span:
            if self.context.tracing_enabled:
                span.set_attribute(GEN_AI_AGENT_NAME, self.agent_name)
                span.set_attribute(GEN_AI_TOOL_NAME, name)
                if arguments is not None:
                    record_attributes(span, arguments, 'arguments')
            if not self.initialized:
                await self.load_servers()
            server_name: str = None
            local_tool_name: str = None
            if server_name:
                span.set_attribute('server_name', server_name)
                local_tool_name = name
            else:
                server_name, local_tool_name = await self._parse_capability_name(name, 'tool')
                span.set_attribute('parsed_server_name', server_name)
                span.set_attribute('parsed_tool_name', local_tool_name)
            if server_name is None or local_tool_name is None:
                logger.error(f"Error: Tool '{name}' not found")
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                span.record_exception(ValueError(f"Tool '{name}' not found"))
                return CallToolResult(isError=True, content=[TextContent(type='text', text=f"Tool '{name}' not found")])
            logger.info('Requesting tool call', data={'progress_action': ProgressAction.CALLING_TOOL, 'tool_name': local_tool_name, 'server_name': server_name, 'agent_name': self.agent_name})
            span.add_event('request_tool_call', {GEN_AI_AGENT_NAME: self.agent_name, GEN_AI_TOOL_NAME: local_tool_name, 'server_name': server_name})

            def _annotate_span_for_result(result: CallToolResult):
                if not self.context.tracing_enabled:
                    return
                annotate_span_for_call_tool_result(span, result)

            async def try_call_tool(client: ClientSession):
                try:
                    res = await client.call_tool(name=local_tool_name, arguments=arguments)
                    _annotate_span_for_result(res)
                    return res
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    span.record_exception(e)
                    return CallToolResult(isError=True, content=[TextContent(type='text', text=f"Failed to call tool '{local_tool_name}' on server '{server_name}': {str(e)}")])
            if self.connection_persistence:
                server_connection = await self._persistent_connection_manager.get_server(server_name, client_session_factory=MCPAgentClientSession)
                res = await try_call_tool(server_connection.session)
                _annotate_span_for_result(res)
                return res
            else:
                logger.debug(f'Creating temporary connection to server: {server_name}', data={'progress_action': ProgressAction.STARTING, 'server_name': server_name, 'agent_name': self.agent_name})
                span.add_event('temporary_connection_created', {'server_name': server_name, GEN_AI_AGENT_NAME: self.agent_name})
                async with gen_client(server_name, server_registry=self.context.server_registry) as client:
                    result = await try_call_tool(client)
                    logger.debug(f'Closing temporary connection to server: {server_name}', data={'progress_action': ProgressAction.SHUTDOWN, 'server_name': server_name, 'agent_name': self.agent_name})
                    span.add_event('temporary_connection_closed', {'server_name': server_name, GEN_AI_AGENT_NAME: self.agent_name})
                    _annotate_span_for_result(result)
                    return result

    async def list_prompts(self, server_name: str | None=None) -> ListPromptsResult:
        """
        :return: Prompts from all servers aggregated, and renamed to be dot-namespaced by server name.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.list_prompts') as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent_name)
            span.set_attribute('initialized', self.initialized)
            if not self.initialized:
                await self.load_servers()
            if server_name:
                span.set_attribute('server_name', server_name)
                res = ListPromptsResult(prompts=[namespaced_prompt.prompt.model_copy(update={'name': namespaced_prompt.namespaced_prompt_name}) for namespaced_prompt in self._server_to_prompt_map.get(server_name, [])])
            else:
                async with self._prompt_map_lock:
                    res = ListPromptsResult(prompts=[namespaced_prompt.prompt.model_copy(update={'name': namespaced_prompt_name}) for namespaced_prompt_name, namespaced_prompt in self._namespaced_prompt_map.items()])
            if self.context.tracing_enabled:
                span.set_attribute('prompts', [prompt.name for prompt in res.prompts])
                for prompt in res.prompts:
                    if prompt.description:
                        span.set_attribute(f'prompt.{prompt.name}.description', prompt.description)
                    if prompt.arguments:
                        for arg in prompt.arguments:
                            for attr in ['description', 'required']:
                                value = getattr(arg, attr, None)
                                if value is not None:
                                    span.set_attribute(f'prompt.{prompt.name}.arguments.{arg.name}.{attr}', value)
            return res

    async def get_prompt(self, name: str, arguments: dict[str, str] | None=None, server_name: str | None=None) -> GetPromptResult:
        """
        Get a prompt from a server.

        Args:
            name: Name of the prompt, optionally namespaced with server name
                using the format 'server_name-prompt_name'
            arguments: Optional dictionary of string arguments to pass to the prompt template
                for prompt template resolution

        Returns:
            Fully resolved prompt returned by the server
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.get_prompt') as span:
            if self.context.tracing_enabled:
                span.set_attribute(GEN_AI_AGENT_NAME, self.agent_name)
                span.set_attribute('name', name)
                span.set_attribute('initialized', self.initialized)
                if arguments is not None:
                    record_attributes(span, arguments, 'arguments')
            if not self.initialized:
                await self.load_servers()
            if server_name:
                span.set_attribute('server_name', server_name)
                local_prompt_name = name
            else:
                server_name, local_prompt_name = await self._parse_capability_name(name, 'prompt')
                span.set_attribute('parsed_server_name', server_name)
                span.set_attribute('parsed_prompt_name', local_prompt_name)
            if server_name is None or local_prompt_name is None:
                logger.error(f"Error: Prompt '{name}' not found")
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                span.record_exception(ValueError(f"Prompt '{name}' not found"))
                return GetPromptResult(isError=True, description=f"Prompt '{name}' not found", messages=[])
            logger.info('Requesting prompt', data={'progress_action': ProgressAction.CALLING_TOOL, 'tool_name': local_prompt_name, 'server_name': server_name, 'agent_name': self.agent_name})
            span.add_event('request_prompt', {'prompt_name': local_prompt_name, 'server_name': server_name, 'agent_name': self.agent_name})

            async def try_get_prompt(client: ClientSession):
                try:
                    return await client.get_prompt(name=local_prompt_name, arguments=arguments)
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    span.record_exception(e)
                    return GetPromptResult(isError=True, description=f"Failed to get prompt '{local_prompt_name}' on server '{server_name}': {str(e)}", messages=[])
            result: GetPromptResult = GetPromptResult(messages=[])
            if self.connection_persistence:
                server_connection = await self._persistent_connection_manager.get_server(server_name, client_session_factory=MCPAgentClientSession)
                result = await try_get_prompt(server_connection.session)
            else:
                logger.debug(f'Creating temporary connection to server: {server_name}', data={'progress_action': ProgressAction.STARTING, 'server_name': server_name, 'agent_name': self.agent_name})
                span.add_event('temporary_connection_created', {'server_name': server_name, 'agent_name': self.agent_name})
                async with gen_client(server_name, server_registry=self.context.server_registry) as client:
                    result = await try_get_prompt(client)
                    logger.debug(f'Closing temporary connection to server: {server_name}', data={'progress_action': ProgressAction.SHUTDOWN, 'server_name': server_name, 'agent_name': self.agent_name})
                    span.add_event('temporary_connection_closed', {'server_name': server_name, 'agent_name': self.agent_name})
            if result and result.messages:
                result.server_name = server_name
                result.prompt_name = local_prompt_name
                result.namespaced_name = f'{server_name}{SEP}{local_prompt_name}'
                if arguments:
                    result.arguments = arguments
                if self.context.tracing_enabled:
                    for idx, message in enumerate(result.messages):
                        span.set_attribute(f'prompt.message.{idx}.role', message.role)
                        span.set_attribute(f'prompt.message.{idx}.content.type', message.content.type)
                        if message.content.type == 'text':
                            span.set_attribute(f'prompt.message.{idx}.content.text', message.content.text)
                    if result.description:
                        span.set_attribute('prompt.description', result.description)
            return result

    async def _parse_capability_name(self, name: str, capability: Literal['tool', 'prompt', 'resource']) -> tuple[str, str]:
        """
        Parse a capability name into server name and local capability name.

        Args:
            name: The tool, prompt, or resource URI, possibly namespaced
            capability: The type of capability, either 'tool', 'prompt', or 'resource'

        Returns:
            Tuple of (server_name, local_name)
        """
        if SEP in name:
            parts = name.split(SEP)
            for i in range(len(parts) - 1, 0, -1):
                prefix = SEP.join(parts[:i])
                if prefix in self.server_names:
                    return (prefix, SEP.join(parts[i:]))
        if capability == 'tool':
            lock = self._tool_map_lock
            capability_map = self._server_to_tool_map

            def getter(item: NamespacedTool):
                return item.tool.name
        elif capability == 'prompt':
            lock = self._prompt_map_lock
            capability_map = self._server_to_prompt_map

            def getter(item: NamespacedPrompt):
                return item.prompt.name
        elif capability == 'resource':
            lock = self._resource_map_lock
            capability_map = self._server_to_resource_map

            def getter(item: NamespacedResource):
                return str(item.resource.uri)
        else:
            raise ValueError(f'Unsupported capability: {capability}')
        async with lock:
            for srv_name in self.server_names:
                items = capability_map.get(srv_name, [])
                for item in items:
                    if getter(item) == name:
                        return (srv_name, name)
        return (None, None)

    async def _start_server(self, server_name: str):
        if self.connection_persistence:
            logger.info(f'Creating persistent connection to server: {server_name}', data={'progress_action': ProgressAction.STARTING, 'server_name': server_name, 'agent_name': self.agent_name})
            server_conn = await self._persistent_connection_manager.get_server(server_name, client_session_factory=MCPAgentClientSession)
            logger.info(f"MCP Server initialized for agent '{self.agent_name}'", data={'progress_action': ProgressAction.STARTING, 'server_name': server_name, 'agent_name': self.agent_name})
            return server_conn.session
        else:
            async with gen_client(server_name, server_registry=self.context.server_registry) as client:
                return client

    async def _fetch_tools(self, client: ClientSession, server_name: str) -> List[Tool]:
        capabilities = await self.get_capabilities(server_name)
        if not capabilities or not capabilities.tools:
            logger.debug(f"Server '{server_name}' does not support tools")
            return []
        tools: List[Tool] = []
        try:
            result = await client.list_tools()
            if not result:
                return []
            cursor = result.nextCursor
            tools.extend(result.tools or [])
            while cursor:
                result = await client.list_tools(cursor=cursor)
                if not result:
                    return tools
                cursor = result.nextCursor
                tools.extend(result.tools or [])
            return tools
        except Exception as e:
            logger.error(f"Error loading tools from server '{server_name}'", data=e)
            return tools

    async def _fetch_prompts(self, client: ClientSession, server_name: str) -> List[Prompt]:
        capabilities = await self.get_capabilities(server_name)
        if not capabilities or not capabilities.prompts:
            logger.debug(f"Server '{server_name}' does not support prompts")
            return []
        prompts: List[Prompt] = []
        try:
            result = await client.list_prompts()
            if not result:
                return prompts
            cursor = result.nextCursor
            prompts.extend(result.prompts or [])
            while cursor:
                result = await client.list_prompts(cursor=cursor)
                if not result:
                    return prompts
                cursor = result.nextCursor
                prompts.extend(result.prompts or [])
            return prompts
        except Exception as e:
            logger.error(f"Error loading prompts from server '{server_name}': {e}")
            return prompts

    async def _fetch_resources(self, client: ClientSession, server_name: str) -> list[Resource]:
        capabilities = await self.get_capabilities(server_name)
        if not capabilities or not getattr(capabilities, 'resources', None):
            logger.debug(f"Server '{server_name}' does not support resources")
            return []
        resources: List[Resource] = []
        try:
            result = await client.list_resources()
            if not result:
                return resources
            cursor = getattr(result, 'nextCursor', None)
            resources.extend(getattr(result, 'resources', []) or [])
            while cursor:
                result = await client.list_resources(cursor=cursor)
                if not result:
                    return resources
                cursor = getattr(result, 'nextCursor', None)
                resources.extend(getattr(result, 'resources', []) or [])
            return resources
        except Exception as e:
            logger.error(f"Error loading resources from server '{server_name}': {e}")
            return resources

    async def _fetch_capabilities(self, server_name: str):
        tools: List[Tool] = []
        prompts: List[Prompt] = []
        resources: List[Resource] = []
        if self.connection_persistence:
            server_connection = await self._persistent_connection_manager.get_server(server_name, client_session_factory=MCPAgentClientSession)
            tools = await self._fetch_tools(server_connection.session, server_name)
            prompts = await self._fetch_prompts(server_connection.session, server_name)
            resources = await self._fetch_resources(server_connection.session, server_name)
        else:
            async with gen_client(server_name, server_registry=self.context.server_registry) as client:
                tools = await self._fetch_tools(client, server_name)
                prompts = await self._fetch_prompts(client, server_name)
                resources = await self._fetch_resources(client, server_name)
        return (server_name, tools, prompts, resources)

class GoogleAugmentedLLM(AugmentedLLM[types.Content, types.Content]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, type_converter=GoogleMCPTypeConverter, **kwargs)
        self.provider = 'Google (AI_Studio)'
        self.logger = get_logger(f'{__name__}.{self.name}' if self.name else __name__)
        self.model_preferences = self.model_preferences or ModelPreferences(costPriority=0.3, speedPriority=0.4, intelligencePriority=0.3)
        default_model = 'gemini-2.5-flash'
        if self.context.config.google:
            if hasattr(self.context.config.google, 'default_model'):
                default_model = self.context.config.google.default_model
        self.default_request_params = self.default_request_params or RequestParams(model=default_model, modelPreferences=self.model_preferences, maxTokens=4096, systemPrompt=self.instruction, parallel_tool_calls=True, max_iterations=10, use_history=True)

    @track_tokens()
    async def generate(self, message, request_params: RequestParams | None=None):
        """
        Process a query using an LLM and available tools.
        The default implementation uses AWS Nova's ChatCompletion as the LLM.
        Override this method to use a different LLM.
        """
        messages: list[types.Content] = []
        params = self.get_request_params(request_params)
        if params.use_history:
            messages.extend(self.history.get())
        messages.extend(GoogleConverter.convert_mixed_messages_to_google(message))
        response = await self.agent.list_tools(tool_filter=params.tool_filter)
        tools = [types.Tool(function_declarations=[types.FunctionDeclaration(name=tool.name, description=tool.description, parameters=transform_mcp_tool_schema(tool.inputSchema))]) for tool in response.tools]
        responses: list[types.Content] = []
        model = await self.select_model(params)
        for i in range(params.max_iterations):
            inference_config = types.GenerateContentConfig(max_output_tokens=params.maxTokens, temperature=params.temperature, stop_sequences=params.stopSequences or [], system_instruction=self.instruction or params.systemPrompt, tools=tools, automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True), candidate_count=1, **params.metadata or {})
            arguments = {'model': model, 'contents': messages, 'config': inference_config}
            self.logger.debug('Completion request arguments:', data=arguments)
            self._log_chat_progress(chat_turn=(len(messages) + 1) // 2, model=model)
            response: types.GenerateContentResponse = await self.executor.execute(GoogleCompletionTasks.request_completion_task, RequestCompletionRequest(config=self.context.config.google, payload=arguments))
            if isinstance(response, BaseException):
                self.logger.error(f'Error: {response}')
                break
            self.logger.debug(f'{model} response:', data=response)
            if not response.candidates:
                break
            candidate = response.candidates[0]
            response_as_message = self.convert_message_to_message_param(candidate.content)
            messages.append(response_as_message)
            if not candidate.content or not candidate.content.parts:
                break
            responses.append(candidate.content)
            function_calls = [self.execute_tool_call(part.function_call) for part in candidate.content.parts if part.function_call]
            if function_calls:
                results: list[types.Content | BaseException | None] = await self.executor.execute_many(function_calls)
                self.logger.debug(f'Iteration {i}: Tool call results: {(str(results) if results else 'None')}')
                function_response_parts: list[types.Part] = []
                for result in results:
                    if result and (not isinstance(result, BaseException)) and result.parts:
                        function_response_parts.extend(result.parts)
                    else:
                        self.logger.error(f'Warning: Unexpected error during tool execution: {result}. Continuing...')
                        function_response_parts.append(types.Part.from_text(text=f'Error executing tool: {result}'))
                if function_response_parts:
                    function_response_content = types.Content(role='tool', parts=function_response_parts)
                    messages.append(function_response_content)
            else:
                self.logger.debug(f"Iteration {i}: Stopping because finish_reason is '{candidate.finish_reason}'")
                break
        if params.use_history:
            self.history.set(messages)
        self._log_chat_finished(model=model)
        return responses

    async def generate_str(self, message, request_params: RequestParams | None=None):
        """
        Process a query using an LLM and available tools.
        The default implementation uses gemini-2.0-flash as the LLM
        Override this method to use a different LLM.
        """
        contents = await self.generate(message=message, request_params=request_params)
        response = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role='model', parts=[part for content in contents for part in content.parts]))])
        return response.text or ''

    @classmethod
    def get_provider_config(cls, context):
        return getattr(getattr(context, 'config', None), 'google', None)

    async def generate_structured(self, message, response_model: Type[ModelT], request_params: RequestParams | None=None) -> ModelT:
        """
        Use Gemini native structured outputs via response_schema and response_mime_type.
        """
        import json
        params = self.get_request_params(request_params)
        model = await self.select_model(params) or (params.model or 'gemini-2.5-flash')
        messages = GoogleConverter.convert_mixed_messages_to_google(message)
        try:
            schema = response_model.model_json_schema()
        except Exception:
            schema = None
        config = types.GenerateContentConfig(max_output_tokens=params.maxTokens, temperature=params.temperature, stop_sequences=params.stopSequences or [], system_instruction=self.instruction or params.systemPrompt)
        config.response_mime_type = 'application/json'
        config.response_schema = schema if schema is not None else response_model
        conversation: list[types.Content] = []
        if params.use_history:
            conversation.extend(self.history.get())
        if isinstance(messages, list):
            conversation.extend(messages)
        else:
            conversation.append(messages)
        api_response: types.GenerateContentResponse = await self.executor.execute(GoogleCompletionTasks.request_completion_task, RequestCompletionRequest(config=self.context.config.google, payload={'model': model, 'contents': conversation, 'config': config}))
        text = None
        if api_response and api_response.candidates:
            cand = api_response.candidates[0]
            if cand.content and cand.content.parts:
                for part in cand.content.parts:
                    if part.text:
                        text = part.text
                        break
        if not text:
            raise ValueError('No structured response returned by Gemini')
        data = json.loads(text)
        return response_model.model_validate(data)

    @classmethod
    def convert_message_to_message_param(cls, message, **kwargs):
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        return message

    async def execute_tool_call(self, function_call: types.FunctionCall) -> types.Content | None:
        """
        Execute a single tool call and return the result message.
        Returns None if there's no content to add to messages.
        """
        tool_name = function_call.name
        tool_args = function_call.args
        tool_call_id = function_call.id
        tool_call_request = CallToolRequest(method='tools/call', params=CallToolRequestParams(name=tool_name, arguments=tool_args))
        result = await self.call_tool(request=tool_call_request, tool_call_id=tool_call_id)
        function_response_content = self.from_mcp_tool_result(result, tool_name)
        return function_response_content

    def message_param_str(self, message) -> str:
        """Convert an input message to a string representation."""
        return str(message.model_dump())

    def message_str(self, message, content_only: bool=False) -> str:
        """Convert an output message to a string representation."""
        return str(message.model_dump())

class OpenAIAugmentedLLM(AugmentedLLM[ChatCompletionMessageParam, ChatCompletionMessage]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    This implementation uses OpenAI's ChatCompletion as the LLM.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, type_converter=MCPOpenAITypeConverter, **kwargs)
        self.provider = 'OpenAI'
        self.logger = get_logger(f'{__name__}.{self.name}' if self.name else __name__)
        self.model_preferences = self.model_preferences or ModelPreferences(costPriority=0.3, speedPriority=0.4, intelligencePriority=0.3)
        if 'default_model' in kwargs:
            default_model = kwargs['default_model']
        else:
            default_model = 'gpt-4o'
        self._reasoning_effort = 'medium'
        if self.context and self.context.config and self.context.config.openai:
            if hasattr(self.context.config.openai, 'default_model'):
                default_model = self.context.config.openai.default_model
            if hasattr(self.context.config.openai, 'reasoning_effort'):
                self._reasoning_effort = self.context.config.openai.reasoning_effort
        self._reasoning = lambda model: model and model.startswith(('o1', 'o3', 'o4', 'gpt-5'))
        if self._reasoning(default_model):
            self.logger.info(f"Using reasoning model '{default_model}' with '{self._reasoning_effort}' reasoning effort")
        self.default_request_params = self.default_request_params or RequestParams(model=default_model, modelPreferences=self.model_preferences, maxTokens=4096, systemPrompt=self.instruction, parallel_tool_calls=False, max_iterations=10, use_history=True)

    @classmethod
    def get_provider_config(cls, context):
        return getattr(getattr(context, 'config', None), 'openai', None)

    @classmethod
    def convert_message_to_message_param(cls, message: ChatCompletionMessage, **kwargs) -> ChatCompletionMessageParam:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        assistant_message_params = {'role': 'assistant', 'audio': message.audio, 'refusal': message.refusal, **kwargs}
        if message.content is not None:
            assistant_message_params['content'] = message.content
        if message.tool_calls is not None:
            assistant_message_params['tool_calls'] = message.tool_calls
        return ChatCompletionAssistantMessageParam(**assistant_message_params)

    @track_tokens()
    async def generate(self, message, request_params: RequestParams | None=None):
        """
        Process a query using an LLM and available tools.
        The default implementation uses OpenAI's ChatCompletion as the LLM.
        Override this method to use a different LLM.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.{self.name}.generate') as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)
            self._annotate_span_for_generation_message(span, message)
            messages: List[ChatCompletionMessageParam] = []
            params = self.get_request_params(request_params)
            if self.context.tracing_enabled:
                AugmentedLLM.annotate_span_with_request_params(span, params)
            if params.use_history:
                messages.extend(self.history.get())
            system_prompt = self.instruction or params.systemPrompt
            if system_prompt and len(messages) == 0:
                span.set_attribute('system_prompt', system_prompt)
                messages.append(ChatCompletionSystemMessageParam(role='system', content=system_prompt))
            messages.extend(OpenAIConverter.convert_mixed_messages_to_openai(message))
            response: ListToolsResult = await self.agent.list_tools(tool_filter=params.tool_filter)
            available_tools: List[ChatCompletionToolParam] = [ChatCompletionToolParam(type='function', function={'name': tool.name, 'description': tool.description, 'parameters': tool.inputSchema}) for tool in response.tools]
            if self.context.tracing_enabled:
                span.set_attribute('available_tools', [t.get('function', {}).get('name') for t in available_tools])
            if not available_tools:
                available_tools = None
            responses: List[ChatCompletionMessage] = []
            model = await self.select_model(params)
            if model:
                span.set_attribute(GEN_AI_REQUEST_MODEL, model)
            user = params.user or getattr(self.context.config.openai, 'user', None)
            if self.context.tracing_enabled and user:
                span.set_attribute('user', user)
            total_input_tokens = 0
            total_output_tokens = 0
            finish_reasons = []
            for i in range(params.max_iterations):
                arguments = {'model': model, 'messages': messages, 'tools': available_tools}
                if user:
                    arguments['user'] = user
                if params.stopSequences is not None:
                    arguments['stop'] = params.stopSequences
                if self._reasoning(model):
                    arguments = {**arguments, 'max_completion_tokens': params.maxTokens, 'reasoning_effort': self._reasoning_effort}
                else:
                    arguments = {**arguments, 'max_tokens': params.maxTokens}
                if params.metadata:
                    arguments = {**arguments, **params.metadata}
                self.logger.debug('Completion request arguments:', data=arguments)
                self._log_chat_progress(chat_turn=len(messages) // 2, model=model)
                request = RequestCompletionRequest(config=self.context.config.openai, payload=arguments)
                self._annotate_span_for_completion_request(span, request, i)
                response: ChatCompletion = await self.executor.execute(OpenAICompletionTasks.request_completion_task, ensure_serializable(request))
                self.logger.debug('OpenAI ChatCompletion response:', data=response)
                if isinstance(response, BaseException):
                    self.logger.error(f'Error: {response}')
                    span.record_exception(response)
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    break
                self._annotate_span_for_completion_response(span, response, i)
                iteration_input = response.usage.prompt_tokens
                iteration_output = response.usage.completion_tokens
                total_input_tokens += iteration_input
                total_output_tokens += iteration_output
                if self.context.token_counter:
                    await self.context.token_counter.record_usage(input_tokens=iteration_input, output_tokens=iteration_output, model_name=model, provider=self.provider)
                if not response.choices or len(response.choices) == 0:
                    break
                choice = response.choices[0]
                message = choice.message
                responses.append(message)
                finish_reasons.append(choice.finish_reason)
                sanitized_name = re.sub('[^a-zA-Z0-9_-]', '_', self.name) if isinstance(self.name, str) else None
                converted_message = self.convert_message_to_message_param(message, name=sanitized_name)
                messages.append(converted_message)
                if choice.finish_reason in ['tool_calls', 'function_call'] and message.tool_calls:
                    tool_tasks = [functools.partial(self.execute_tool_call, tool_call=tool_call) for tool_call in message.tool_calls]
                    tool_results = await self.executor.execute_many(tool_tasks)
                    self.logger.debug(f'Iteration {i}: Tool call results: {(str(tool_results) if tool_results else 'None')}')
                    for result in tool_results:
                        if isinstance(result, BaseException):
                            self.logger.error(f'Warning: Unexpected error during tool execution: {result}. Continuing...')
                            span.record_exception(result)
                            continue
                        if result is not None:
                            messages.append(result)
                elif choice.finish_reason == 'length':
                    self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'length'")
                    span.set_attribute('finish_reason', 'length')
                    break
                elif choice.finish_reason == 'content_filter':
                    self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'content_filter'")
                    span.set_attribute('finish_reason', 'content_filter')
                    break
                elif choice.finish_reason == 'stop':
                    self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'stop'")
                    span.set_attribute('finish_reason', 'stop')
                    break
            if params.use_history:
                self.history.set(messages)
            self._log_chat_finished(model=model)
            if self.context.tracing_enabled:
                span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, total_input_tokens)
                span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, total_output_tokens)
                span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)
                for i, res in enumerate(responses):
                    response_data = self.extract_response_message_attributes_for_tracing(res, prefix=f'response.{i}')
                    span.set_attributes(response_data)
            return responses

    async def generate_str(self, message, request_params: RequestParams | None=None):
        """
        Process a query using an LLM and available tools.
        The default implementation uses OpenAI's ChatCompletion as the LLM.
        Override this method to use a different LLM.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.{self.name}.generate_str') as span:
            if self.context.tracing_enabled:
                span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)
                self._annotate_span_for_generation_message(span, message)
                if request_params:
                    AugmentedLLM.annotate_span_with_request_params(span, request_params)
            responses = await self.generate(message=message, request_params=request_params)
            final_text: List[str] = []
            for response in responses:
                content = response.content
                if not content:
                    continue
                if isinstance(content, str):
                    final_text.append(content)
                    continue
            res = '\n'.join(final_text)
            span.set_attribute('response', res)
            return res

    async def generate_structured(self, message, response_model: Type[ModelT], request_params: RequestParams | None=None) -> ModelT:
        """
        Use OpenAI native structured outputs via response_format (JSON schema).
        """
        import json
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.{self.name}.generate_structured') as span:
            if self.context.tracing_enabled:
                span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)
                self._annotate_span_for_generation_message(span, message)
            params = self.get_request_params(request_params)
            model = await self.select_model(params) or (self.default_request_params.model or 'gpt-4o')
            if self.context.tracing_enabled:
                AugmentedLLM.annotate_span_with_request_params(span, params)
                span.set_attribute(GEN_AI_REQUEST_MODEL, model)
                span.set_attribute('response_model', response_model.__name__)
            messages: List[ChatCompletionMessageParam] = []
            system_prompt = self.instruction or params.systemPrompt
            if system_prompt:
                messages.append(ChatCompletionSystemMessageParam(role='system', content=system_prompt))
            if params.use_history:
                messages.extend(self.history.get())
            messages.extend(OpenAIConverter.convert_mixed_messages_to_openai(message))
            schema = response_model.model_json_schema()

            def _ensure_no_additional_props_and_require_all(node: dict):
                if not isinstance(node, dict):
                    return
                node_type = node.get('type')
                if node_type == 'object':
                    if 'additionalProperties' not in node:
                        node['additionalProperties'] = False
                    props = node.get('properties')
                    if isinstance(props, dict):
                        node['required'] = list(props.keys())
                for key in ('properties', '$defs', 'definitions'):
                    sub = node.get(key)
                    if isinstance(sub, dict):
                        for v in sub.values():
                            _ensure_no_additional_props_and_require_all(v)
                if 'items' in node:
                    _ensure_no_additional_props_and_require_all(node['items'])
                for key in ('oneOf', 'anyOf', 'allOf'):
                    subs = node.get(key)
                    if isinstance(subs, list):
                        for v in subs:
                            _ensure_no_additional_props_and_require_all(v)
            if params.strict:
                _ensure_no_additional_props_and_require_all(schema)
            response_format = {'type': 'json_schema', 'json_schema': {'name': getattr(response_model, '__name__', 'StructuredOutput'), 'schema': schema, 'strict': params.strict}}
            payload = {'model': model, 'messages': messages, 'response_format': response_format}
            if self._reasoning(model):
                payload['max_completion_tokens'] = params.maxTokens
                payload['reasoning_effort'] = self._reasoning_effort
            else:
                payload['max_tokens'] = params.maxTokens
            user = params.user or getattr(self.context.config.openai, 'user', None)
            if user:
                payload['user'] = user
            if params.stopSequences is not None:
                payload['stop'] = params.stopSequences
            if params.metadata:
                payload.update(params.metadata)
            completion: ChatCompletion = await self.executor.execute(OpenAICompletionTasks.request_completion_task, RequestCompletionRequest(config=self.context.config.openai, payload=payload))
            if isinstance(completion, BaseException):
                raise completion
            if not completion.choices or completion.choices[0].message.content is None:
                raise ValueError('No structured content returned by model')
            content = completion.choices[0].message.content
            try:
                data = json.loads(content)
                return response_model.model_validate(data)
            except Exception:
                return response_model.model_validate_json(content)

    async def pre_tool_call(self, tool_call_id: str | None, request: CallToolRequest):
        return request

    async def post_tool_call(self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult):
        return result

    async def execute_tool_call(self, tool_call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessageParam:
        """
        Execute a single tool call and return the result message.
        Returns a single ChatCompletionToolMessageParam object.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.{self.name}.execute_tool_call') as span:
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments
            tool_call_id = tool_call.id
            tool_args = {}
            if self.context.tracing_enabled:
                span.set_attribute(GEN_AI_TOOL_CALL_ID, tool_call_id)
                span.set_attribute(GEN_AI_TOOL_NAME, tool_name)
                span.set_attribute('tool_args', tool_args_str)
            try:
                if tool_args_str:
                    tool_args = json.loads(tool_args_str)
            except json.JSONDecodeError as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                return ChatCompletionToolMessageParam(role='tool', tool_call_id=tool_call_id, content=f"Invalid JSON provided in tool call arguments for '{tool_name}'. Failed to load JSON: {str(e)}")
            tool_call_request = CallToolRequest(method='tools/call', params=CallToolRequestParams(name=tool_name, arguments=tool_args))
            result = await self.call_tool(request=tool_call_request, tool_call_id=tool_call_id)
            self._annotate_span_for_call_tool_result(span, result)
            return ChatCompletionToolMessageParam(role='tool', tool_call_id=tool_call_id, content=[mcp_content_to_openai_content_part(c) for c in result.content])

    def message_param_str(self, message: ChatCompletionMessageParam) -> str:
        """Convert an input message to a string representation."""
        if message.get('content'):
            content = message['content']
            if isinstance(content, str):
                return content
            else:
                final_text: List[str] = []
                for part in content:
                    text_part = part.get('text')
                    if text_part:
                        final_text.append(str(text_part))
                    else:
                        final_text.append(str(part))
                return '\n'.join(final_text)
        return str(message)

    def message_str(self, message: ChatCompletionMessage, content_only: bool=False) -> str:
        """Convert an output message to a string representation."""
        content = message.content
        if content:
            return content
        elif content_only:
            return ''
        return str(message)

    def _annotate_span_for_generation_message(self, span: trace.Span, message: MessageTypes) -> None:
        """Annotate the span with the message content."""
        if not self.context.tracing_enabled:
            return
        if isinstance(message, str):
            span.set_attribute('message.content', message)
        elif isinstance(message, list):
            for i, msg in enumerate(message):
                if isinstance(msg, str):
                    span.set_attribute(f'message.{i}.content', msg)
                else:
                    span.set_attribute(f'message.{i}', str(msg))
        else:
            span.set_attribute('message', str(message))

    def _extract_message_param_attributes_for_tracing(self, message_param: ChatCompletionMessageParam, prefix: str='message') -> dict[str, Any]:
        """Return a flat dict of span attributes for a given ChatCompletionMessageParam."""
        attrs = {}
        return attrs

    def _annotate_span_for_completion_request(self, span: trace.Span, request: RequestCompletionRequest, turn: int) -> None:
        """Annotate the span with the completion request as an event."""
        if not self.context.tracing_enabled:
            return
        event_data = {'completion.request.turn': turn, 'config.reasoning_effort': request.config.reasoning_effort}
        if request.config.base_url:
            event_data['config.base_url'] = request.config.base_url
        for key, value in request.payload.items():
            if key == 'messages':
                for i, message in enumerate(cast(List[ChatCompletionMessageParam], value)):
                    role = message.get('role')
                    event_data[f'messages.{i}.role'] = role
                    message_content = message.get('content')
                    match role:
                        case 'developer' | 'system' | 'user':
                            if isinstance(message_content, str):
                                event_data[f'messages.{i}.content'] = message_content
                            elif message_content is not None:
                                for j, part in enumerate(message_content):
                                    event_data[f'messages.{i}.content.{j}.type'] = part['type']
                                    if part['type'] == 'text':
                                        event_data[f'messages.{i}.content.{j}.text'] = part['text']
                                    elif part['type'] == 'image_url':
                                        event_data[f'messages.{i}.content.{j}.image_url.url'] = part['image_url']['url']
                                        event_data[f'messages.{i}.content.{j}.image_url.detail'] = part['image_url']['detail']
                                    elif part['type'] == 'input_audio':
                                        event_data[f'messages.{i}.content.{j}.input_audio.format'] = part['input_audio']['format']
                        case 'assistant':
                            if isinstance(message_content, str):
                                event_data[f'messages.{i}.content'] = message_content
                            elif message_content is not None:
                                for j, part in enumerate(message_content):
                                    event_data[f'messages.{i}.content.{j}.type'] = part['type']
                                    if part['type'] == 'text':
                                        event_data[f'messages.{i}.content.{j}.text'] = part['text']
                                    elif part['type'] == 'refusal':
                                        event_data[f'messages.{i}.content.{j}.refusal'] = part['refusal']
                            if message.get('audio') is not None:
                                event_data[f'messages.{i}.audio.id'] = message.get('audio').get('id')
                            if message.get('function_call') is not None:
                                event_data[f'messages.{i}.function_call.name'] = message.get('function_call').get('name')
                                event_data[f'messages.{i}.function_call.arguments'] = message.get('function_call').get('arguments')
                            if message.get('name') is not None:
                                event_data[f'messages.{i}.name'] = message.get('name')
                            if message.get('refusal') is not None:
                                event_data[f'messages.{i}.refusal'] = message.get('refusal')
                            if message.get('tool_calls') is not None:
                                for j, tool_call in enumerate(message.get('tool_calls')):
                                    event_data[f'messages.{i}.tool_calls.{j}.{GEN_AI_TOOL_CALL_ID}'] = tool_call.id
                                    event_data[f'messages.{i}.tool_calls.{j}.function.name'] = tool_call.function.name
                                    event_data[f'messages.{i}.tool_calls.{j}.function.arguments'] = tool_call.function.arguments
                        case 'tool':
                            event_data[f'messages.{i}.{GEN_AI_TOOL_CALL_ID}'] = message.get('tool_call_id')
                            if isinstance(message_content, str):
                                event_data[f'messages.{i}.content'] = message_content
                            elif message_content is not None:
                                for j, part in enumerate(message_content):
                                    event_data[f'messages.{i}.content.{j}.type'] = part['type']
                                    if part['type'] == 'text':
                                        event_data[f'messages.{i}.content.{j}.text'] = part['text']
                        case 'function':
                            event_data[f'messages.{i}.name'] = message.get('name')
                            event_data[f'messages.{i}.content'] = message_content
            elif key == 'tools':
                if value is not None:
                    event_data['tools'] = [tool.get('function', {}).get('name') for tool in value]
            elif is_otel_serializable(value):
                event_data[key] = value
        event_name = f'completion.request.{turn}'
        latest_message_role = request.payload.get('messages', [{}])[-1].get('role')
        if latest_message_role:
            event_name = f'gen_ai.{latest_message_role}.message'
        span.add_event(event_name, event_data)

    def _annotate_span_for_completion_response(self, span: trace.Span, response: ChatCompletion, turn: int) -> None:
        """Annotate the span with the completion response as an event."""
        if not self.context.tracing_enabled:
            return
        event_data = {'completion.response.turn': turn}
        event_data.update(self._extract_chat_completion_attributes_for_tracing(response))
        event_name = f'completion.response.{turn}'
        if response.choices and len(response.choices) > 0:
            latest_message_role = response.choices[0].message.role
            event_name = f'gen_ai.{latest_message_role}.message'
        span.add_event(event_name, event_data)

    def extract_response_message_attributes_for_tracing(self, message: ChatCompletionMessage, prefix: str | None=None) -> Dict[str, Any]:
        """
        Extract relevant attributes from the ChatCompletionMessage for tracing.
        """
        if not self.context.tracing_enabled:
            return {}
        attr_prefix = f'{prefix}.' if prefix else ''
        attrs = {f'{attr_prefix}role': message.role}
        if message.content is not None:
            attrs[f'{attr_prefix}content'] = message.content
        if message.refusal:
            attrs[f'{attr_prefix}refusal'] = message.refusal
        if message.audio is not None:
            attrs[f'{attr_prefix}audio.id'] = message.audio.id
            attrs[f'{attr_prefix}audio.expires_at'] = message.audio.expires_at
            attrs[f'{attr_prefix}audio.transcript'] = message.audio.transcript
        if message.function_call is not None:
            attrs[f'{attr_prefix}function_call.name'] = message.function_call.name
            attrs[f'{attr_prefix}function_call.arguments'] = message.function_call.arguments
        if message.tool_calls:
            for j, tool_call in enumerate(message.tool_calls):
                attrs[f'{attr_prefix}tool_calls.{j}.{GEN_AI_TOOL_CALL_ID}'] = tool_call.id
                attrs[f'{attr_prefix}tool_calls.{j}.function.name'] = tool_call.function.name
                attrs[f'{attr_prefix}tool_calls.{j}.function.arguments'] = tool_call.function.arguments
        return attrs

    def _extract_chat_completion_attributes_for_tracing(self, response: ChatCompletion, prefix: str | None=None) -> Dict[str, Any]:
        """
        Extract relevant attributes from the ChatCompletion response for tracing.
        """
        if not self.context.tracing_enabled:
            return {}
        attr_prefix = f'{prefix}.' if prefix else ''
        attrs = {f'{attr_prefix}id': response.id, f'{attr_prefix}model': response.model, f'{attr_prefix}object': response.object, f'{attr_prefix}created': response.created}
        if response.service_tier:
            attrs[f'{attr_prefix}service_tier'] = response.service_tier
        if response.system_fingerprint:
            attrs[f'{attr_prefix}system_fingerprint'] = response.system_fingerprint
        if response.usage:
            attrs[f'{attr_prefix}{GEN_AI_USAGE_INPUT_TOKENS}'] = response.usage.prompt_tokens
            attrs[f'{attr_prefix}{GEN_AI_USAGE_OUTPUT_TOKENS}'] = response.usage.completion_tokens
        finish_reasons = []
        for i, choice in enumerate(response.choices):
            attrs[f'{attr_prefix}choices.{i}.index'] = choice.index
            attrs[f'{attr_prefix}choices.{i}.finish_reason'] = choice.finish_reason
            finish_reasons.append(choice.finish_reason)
            message_attrs = self.extract_response_message_attributes_for_tracing(choice.message, f'{attr_prefix}choices.{i}.message')
            attrs.update(message_attrs)
        attrs[GEN_AI_RESPONSE_FINISH_REASONS] = finish_reasons
        return attrs

class AzureAugmentedLLM(AugmentedLLM[MessageParam, ResponseMessage]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, type_converter=MCPAzureTypeConverter, **kwargs)
        self.provider = 'Azure'
        self.logger = get_logger(f'{__name__}.{self.name}' if self.name else __name__)
        self.model_preferences = self.model_preferences or ModelPreferences(costPriority=0.3, speedPriority=0.4, intelligencePriority=0.3)
        default_model = 'gpt-4o-mini'
        self._is_openai_model = lambda model: model and model.lower().startswith('gpt-')
        if self.context.config.azure:
            if hasattr(self.context.config.azure, 'default_model'):
                default_model = self.context.config.azure.default_model
        if not self.context.config.azure:
            self.logger.error('Azure configuration not found. Please provide Azure configuration.')
            raise ValueError('Azure configuration not found. Please provide Azure configuration.')
        self.default_request_params = self.default_request_params or RequestParams(model=default_model, modelPreferences=self.model_preferences, maxTokens=4096, systemPrompt=self.instruction, parallel_tool_calls=True, max_iterations=10, use_history=True)

    @classmethod
    def get_provider_config(cls, context):
        return getattr(getattr(context, 'config', None), 'azure', None)

    @track_tokens()
    async def generate(self, message, request_params: RequestParams | None=None):
        """
        Process a query using an LLM and available tools.
        The default implementation uses Azure OpenAI 5 as the LLM.
        Override this method to use a different LLM.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'llm_azure.{self.name}.generate') as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)
            self._annotate_span_for_generation_message(span, message)
            messages: list[MessageParam] = []
            responses: list[ResponseMessage] = []
            params = self.get_request_params(request_params)
            if self.context.tracing_enabled:
                AugmentedLLM.annotate_span_with_request_params(span, params)
            if params.use_history:
                span.set_attribute('use_history', params.use_history)
                messages.extend(self.history.get())
            system_prompt = self.instruction or params.systemPrompt
            if system_prompt and len(messages) == 0:
                messages.append(SystemMessage(content=system_prompt))
                span.set_attribute('system_prompt', system_prompt)
            messages.extend(AzureConverter.convert_mixed_messages_to_azure(message))
            response = await self.agent.list_tools(tool_filter=params.tool_filter)
            tools: list[ChatCompletionsToolDefinition] = [ChatCompletionsToolDefinition(function=FunctionDefinition(name=tool.name, description=tool.description, parameters=tool.inputSchema)) for tool in response.tools]
            span.set_attribute('available_tools', [t.function.name for t in tools])
            model = await self.select_model(params)
            if model:
                span.set_attribute(GEN_AI_REQUEST_MODEL, model)
            total_input_tokens = 0
            total_output_tokens = 0
            finish_reasons = []
            for i in range(params.max_iterations):
                arguments = {'messages': messages, 'temperature': params.temperature, 'model': model, 'max_tokens': params.maxTokens, 'stop': params.stopSequences, 'tools': tools}
                user = params.user or getattr(self.context.config.azure, 'user', None)
                if user:
                    arguments['user'] = user
                if params.metadata:
                    arguments = {**arguments, **params.metadata}
                self.logger.debug('Completion request arguments:', data=arguments)
                self._log_chat_progress(chat_turn=(len(messages) + 1) // 2, model=model)
                request = RequestCompletionRequest(config=self.context.config.azure, payload=arguments)
                self._annotate_span_for_completion_request(span, request, i)
                if self._is_openai_model(model):
                    response = await self.executor.execute(AzureOpenAICompletionTasks.request_completion_task, request)
                else:
                    response = await self.executor.execute(AzureCompletionTasks.request_completion_task, request)
                if isinstance(response, BaseException):
                    self.logger.error(f'Error: {response}')
                    span.record_exception(response)
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    break
                self.logger.debug(f'{model} response:', data=response)
                self._annotate_span_for_completion_response(span, response, i)
                if isinstance(response.usage, dict):
                    iteration_input = response.usage['prompt_tokens']
                    iteration_output = response.usage['completion_tokens']
                else:
                    iteration_input = response.usage.prompt_tokens
                    iteration_output = response.usage.completion_tokens
                total_input_tokens += iteration_input
                total_output_tokens += iteration_output
                finish_reasons.append(response.choices[0].finish_reason)
                if self.context.token_counter:
                    await self.context.token_counter.record_usage(input_tokens=iteration_input, output_tokens=iteration_output, model_name=model, provider=self.provider)
                message = response.choices[0].message
                responses.append(message)
                assistant_message = self.convert_message_to_message_param(message)
                messages.append(assistant_message)
                if response.choices[0].finish_reason == CompletionsFinishReason.TOOL_CALLS:
                    if response.choices[0].message.tool_calls is not None and len(response.choices[0].message.tool_calls) > 0:
                        tool_tasks = [self.execute_tool_call(tool_call) for tool_call in response.choices[0].message.tool_calls]
                        tool_results = await self.executor.execute_many(tool_tasks)
                        self.logger.debug(f'Iteration {i}: Tool call results: {(str(tool_results) if tool_results else 'None')}')
                        for result in tool_results:
                            if isinstance(result, BaseException):
                                self.logger.error(f'Warning: Unexpected error during tool execution: {result}. Continuing...')
                                span.record_exception(result)
                                continue
                            elif isinstance(result, ToolMessage):
                                messages.append(result)
                                responses.append(result)
                else:
                    self.logger.debug(f"Iteration {i}: Stopping because finish_reason is '{response.choices[0].finish_reason}'")
                    break
            if params.use_history:
                self.history.set(messages)
            self._log_chat_finished(model=model)
            if self.context.tracing_enabled:
                span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, total_input_tokens)
                span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, total_output_tokens)
                span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)
                for i, res in enumerate(responses):
                    response_data = self.extract_response_message_attributes_for_tracing(res, prefix=f'response.{i}')
                    span.set_attributes(response_data)
            return responses

    async def generate_str(self, message, request_params: RequestParams | None=None):
        """
        Process a query using an LLM and available tools.
        The default implementation uses Azure OpenAI 4o-mini as the LLM.
        Override this method to use a different LLM.
        """
        responses = await self.generate(message=message, request_params=request_params)
        final_text: list[str] = []
        for response in responses:
            if response.content:
                if response.role == 'tool':
                    final_text.append(f'[Tool result: {response.content}]')
                else:
                    final_text.append(response.content)
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call.function.arguments:
                        final_text.append(f'[Calling tool {tool_call.function.name} with args {tool_call.function.arguments}]')
        return '\n'.join(final_text)

    async def generate_structured(self, message, response_model: Type[ModelT], request_params: RequestParams | None=None) -> ModelT:
        json_schema = response_model.model_json_schema()
        request_params = request_params or RequestParams()
        metadata = request_params.metadata or {}
        metadata['response_format'] = JsonSchemaFormat(name=response_model.__name__, description=response_model.__doc__, schema=json_schema, strict=request_params.strict)
        request_params.metadata = metadata
        response = await self.generate(message=message, request_params=request_params)
        json_data = json.loads(response[-1].content)
        structured_response = response_model.model_validate(json_data)
        return structured_response

    @classmethod
    def convert_message_to_message_param(cls, message: ResponseMessage) -> AssistantMessage:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        assistant_message = AssistantMessage(content=message.content, tool_calls=message.tool_calls)
        return assistant_message

    async def execute_tool_call(self, tool_call: ChatCompletionsToolCall) -> ToolMessage | None:
        """
        Execute a single tool call and return the result message.
        Returns None if there's no content to add to messages.
        """
        tool_name = tool_call.function.name
        tool_args_str = tool_call.function.arguments
        tool_call_id = tool_call.id
        tool_args = {}
        try:
            if tool_args_str:
                tool_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return ToolMessage(tool_call_id=tool_call_id, content=f"Invalid JSON provided in tool call arguments for '{tool_name}'. Failed to load JSON: {str(e)}")
        except Exception as e:
            return ToolMessage(tool_call_id=tool_call_id, content=f"Error executing tool '{tool_name}': {str(e)}")
        try:
            tool_call_request = CallToolRequest(method='tools/call', params=CallToolRequestParams(name=tool_name, arguments=tool_args))
            result = await self.call_tool(request=tool_call_request, tool_call_id=tool_call_id)
            if result.content:
                return ToolMessage(tool_call_id=tool_call_id, content=mcp_content_to_azure_content(result.content))
            return None
        except Exception as e:
            return ToolMessage(tool_call_id=tool_call_id, content=f"Error executing tool '{tool_name}': {str(e)}")

    def message_param_str(self, message: MessageParam) -> str:
        """Convert an input message to a string representation."""
        if message.content:
            if isinstance(message.content, str):
                return message.content
            content: list[str] = []
            for c in message.content:
                if isinstance(c, TextContentItem):
                    content.append(c.text)
                elif isinstance(c, ImageContentItem):
                    content.append(f'Image url: {c.image_url.url}')
                elif isinstance(c, AudioContentItem):
                    content.append(f'{c.input_audio.format}: {c.input_audio.data}')
                else:
                    content.append(str(c))
            return '\n'.join(content)
        else:
            return str(message)

    def message_str(self, message: ResponseMessage, content_only: bool=False) -> str:
        """Convert an output message to a string representation."""
        if message.content:
            return message.content
        elif content_only:
            return ''
        return str(message)

    def _annotate_span_for_completion_request(self, span: trace.Span, request: RequestCompletionRequest, turn: int) -> None:
        """Annotate the span with the completion request as an event."""
        if not self.context.tracing_enabled:
            return
        event_data = {'completion.request.turn': turn, 'config.endpoint': request.config.endpoint}
        event_name = f'completion.request.{turn}'
        latest_message_role = request.payload.get('messages', [{}])[-1].get('role')
        if latest_message_role:
            event_name = f'gen_ai.{latest_message_role}.message'
        span.add_event(event_name, event_data)

    def _annotate_span_for_completion_response(self, span: trace.Span, response: ResponseMessage, turn: int) -> None:
        """Annotate the span with the completion response as an event."""
        if not self.context.tracing_enabled:
            return
        event_data = {'completion.response.turn': turn}
        event_data.update(self.extract_response_message_attributes_for_tracing(response))
        event_name = f'completion.response.{turn}'
        if response.choices and len(response.choices) > 0:
            latest_message_role = response.choices[0].message.role
            event_name = f'gen_ai.{latest_message_role}.message'
        span.add_event(event_name, event_data)

    def _extract_message_param_attributes_for_tracing(self, message_param: MessageParam, prefix: str='message') -> dict[str, Any]:
        """Return a flat dict of span attributes for a given MessageParam."""
        attrs = {}
        return attrs

    def extract_response_message_attributes_for_tracing(self, message: ResponseMessage, prefix: str | None=None) -> dict[str, Any]:
        """Return a flat dict of span attributes for a given ResponseMessage."""
        attrs = {}
        return attrs

class BedrockAugmentedLLM(AugmentedLLM[MessageUnionTypeDef, MessageUnionTypeDef]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, type_converter=BedrockMCPTypeConverter, **kwargs)
        self.provider = 'Amazon Bedrock'
        self.logger = get_logger(f'{__name__}.{self.name}' if self.name else __name__)
        self.model_preferences = self.model_preferences or ModelPreferences(costPriority=0.3, speedPriority=0.4, intelligencePriority=0.3)
        default_model = 'us.amazon.nova-lite-v1:0'
        if self.context.config.bedrock:
            if hasattr(self.context.config.bedrock, 'default_model'):
                default_model = self.context.config.bedrock.default_model
        else:
            self.logger.error('Bedrock configuration not found. Please provide Bedrock configuration.')
            raise ValueError('Bedrock configuration not found. Please provide Bedrock configuration.')
        self.default_request_params = self.default_request_params or RequestParams(model=default_model, modelPreferences=self.model_preferences, maxTokens=4096, systemPrompt=self.instruction, parallel_tool_calls=True, max_iterations=10, use_history=True)

    @classmethod
    def get_provider_config(cls, context):
        return getattr(getattr(context, 'config', None), 'bedrock', None)

    @track_tokens()
    async def generate(self, message, request_params: RequestParams | None=None):
        """
        Process a query using an LLM and available tools.
        The default implementation uses AWS Nova's ChatCompletion as the LLM.
        Override this method to use a different LLM.
        """
        messages: list[MessageUnionTypeDef] = []
        params = self.get_request_params(request_params)
        if params.use_history:
            messages.extend(self.history.get())
        messages.extend(BedrockConverter.convert_mixed_messages_to_bedrock(message))
        response = await self.agent.list_tools(tool_filter=params.tool_filter)
        tool_config: ToolConfigurationTypeDef = {'tools': [{'toolSpec': {'name': tool.name, 'description': tool.description, 'inputSchema': {'json': tool.inputSchema}}} for tool in response.tools], 'toolChoice': {'auto': {}}}
        responses: list[MessageUnionTypeDef] = []
        model = await self.select_model(params)
        for i in range(params.max_iterations):
            inference_config = {'maxTokens': params.maxTokens, 'temperature': params.temperature, 'stopSequences': params.stopSequences or []}
            system_content = [{'text': self.instruction or params.systemPrompt}]
            arguments: ConverseRequestTypeDef = {'modelId': model, 'messages': messages, 'system': system_content, 'inferenceConfig': inference_config}
            if isinstance(tool_config['tools'], list) and len(tool_config['tools']) > 0:
                arguments['toolConfig'] = tool_config
            if params.metadata:
                arguments = {**arguments, 'additionalModelRequestFields': params.metadata}
            self.logger.debug('Completion request arguments:', data=arguments)
            self._log_chat_progress(chat_turn=(len(messages) + 1) // 2, model=model)
            response: ConverseResponseTypeDef = await self.executor.execute(BedrockCompletionTasks.request_completion_task, RequestCompletionRequest(config=self.context.config.bedrock, payload=arguments))
            if isinstance(response, BaseException):
                self.logger.error(f'Error: {response}')
                break
            self.logger.debug(f'{model} response:', data=response)
            response_as_message = self.convert_message_to_message_param(response['output']['message'])
            messages.append(response_as_message)
            responses.append(response['output']['message'])
            if response['stopReason'] == 'end_turn':
                self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'end_turn'")
                break
            elif response['stopReason'] == 'stop_sequence':
                self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'stop_sequence'")
                break
            elif response['stopReason'] == 'max_tokens':
                self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'max_tokens'")
                break
            elif response['stopReason'] == 'guardrail_intervened':
                self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'guardrail_intervened'")
                break
            elif response['stopReason'] == 'content_filtered':
                self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'content_filtered'")
                break
            elif response['stopReason'] == 'tool_use':
                tool_results = []
                for content in response['output']['message']['content']:
                    if content.get('toolUse'):
                        tool_use_block = content['toolUse']
                        tool_name = tool_use_block['name']
                        tool_args = tool_use_block['input']
                        tool_use_id = tool_use_block['toolUseId']
                        tool_call_request = CallToolRequest(method='tools/call', params=CallToolRequestParams(name=tool_name, arguments=tool_args))
                        result = await self.call_tool(request=tool_call_request, tool_call_id=tool_use_id)
                        tool_results.append({'toolResult': {'content': mcp_content_to_bedrock_content(result.content), 'toolUseId': tool_use_id, 'status': 'error' if result.isError else 'success'}})
                if tool_results:
                    tool_result_message = {'role': 'user', 'content': tool_results}
                    messages.append(tool_result_message)
                    responses.append(tool_result_message)
        if params.use_history:
            self.history.set(messages)
        self._log_chat_finished(model=model)
        return responses

    async def generate_str(self, message, request_params: RequestParams | None=None):
        """
        Process a query using an LLM and available tools.
        The default implementation uses AWS Nova's ChatCompletion as the LLM.
        Override this method to use a different LLM.
        """
        responses = await self.generate(message=message, request_params=request_params)
        final_text: list[str] = []
        for response in responses:
            for content in response['content']:
                if content.get('text'):
                    final_text.append(content['text'])
                elif content.get('toolUse'):
                    final_text.append(f'[Calling tool {content['toolUse']['name']} with args {content['toolUse']['input']}]')
                elif content.get('toolResult'):
                    final_text.append(f'[Tool result: {content['toolResult']['content']}]')
        return '\n'.join(final_text)

    async def generate_structured(self, message, response_model: Type[ModelT], request_params: RequestParams | None=None) -> ModelT:
        response = await self.generate_str(message=message, request_params=request_params)
        params = self.get_request_params(request_params)
        model = await self.select_model(params) or 'us.amazon.nova-lite-v1:0'
        serialized_response_model: str | None = None
        if self.executor and self.executor.execution_engine == 'temporal':
            serialized_response_model = serialize_model(response_model)
        structured_response = await self.executor.execute(BedrockCompletionTasks.request_structured_completion_task, RequestStructuredCompletionRequest(config=self.context.config.bedrock, response_model=response_model if not serialized_response_model else None, serialized_response_model=serialized_response_model, response_str=response, params=params, model=model))
        if isinstance(structured_response, dict):
            structured_response = response_model.model_validate(structured_response)
        return structured_response

    @classmethod
    def convert_message_to_message_param(cls, message: MessageOutputTypeDef, **kwargs) -> MessageUnionTypeDef:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        return message

    def message_str(self, message: MessageUnionTypeDef, content_only: bool=False) -> str:
        """Convert an output message to a string representation."""
        if message.get('content'):
            final_text: list[str] = []
            for content in message['content']:
                if content.get('text'):
                    final_text.append(content['text'])
                else:
                    final_text.append(str(content))
            return '\n'.join(final_text)
        elif content_only:
            return ''
        return str(message)

class AnthropicAugmentedLLM(AugmentedLLM[MessageParam, Message]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    Our current models can actively use these capabilities—generating their own search queries,
    selecting appropriate tools, and determining what information to retain.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, type_converter=AnthropicMCPTypeConverter, **kwargs)
        self.provider = 'Anthropic'
        self.logger = get_logger(f'{__name__}.{self.name}' if self.name else __name__)
        self.model_preferences = self.model_preferences or ModelPreferences(costPriority=0.3, speedPriority=0.4, intelligencePriority=0.3)
        default_model = 'claude-sonnet-4-20250514'
        if self.context.config.anthropic:
            self.provider = self.context.config.anthropic.provider
            if self.context.config.anthropic.provider == 'bedrock':
                default_model = 'anthropic.claude-sonnet-4-20250514-v1:0'
            elif self.context.config.anthropic.provider == 'vertexai':
                default_model = 'claude-sonnet-4@20250514'
            if hasattr(self.context.config.anthropic, 'default_model'):
                default_model = self.context.config.anthropic.default_model
        self.default_request_params = self.default_request_params or RequestParams(model=default_model, modelPreferences=self.model_preferences, maxTokens=2048, systemPrompt=self.instruction, parallel_tool_calls=False, max_iterations=10, use_history=True)

    @classmethod
    def get_provider_config(cls, context):
        return getattr(getattr(context, 'config', None), 'anthropic', None)

    @track_tokens()
    async def generate(self, message, request_params: RequestParams | None=None):
        """
        Process a query using an LLM and available tools.
        The default implementation uses Claude as the LLM.
        Override this method to use a different LLM.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.{self.name}.generate') as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)
            self._annotate_span_for_generation_message(span, message)
            config = self.context.config
            messages: List[MessageParam] = []
            params = self.get_request_params(request_params)
            if self.context.tracing_enabled:
                AugmentedLLM.annotate_span_with_request_params(span, params)
            if params.use_history:
                messages.extend(self.history.get())
            messages.extend(AnthropicConverter.convert_mixed_messages_to_anthropic(message))
            list_tools_result = await self.agent.list_tools(tool_filter=params.tool_filter)
            available_tools: List[ToolParam] = [{'name': tool.name, 'description': tool.description, 'input_schema': tool.inputSchema} for tool in list_tools_result.tools]
            responses: List[Message] = []
            model = await self.select_model(params)
            if model:
                span.set_attribute(GEN_AI_REQUEST_MODEL, model)
            total_input_tokens = 0
            total_output_tokens = 0
            finish_reasons = []
            for i in range(params.max_iterations):
                if i == params.max_iterations - 1 and responses and (responses[-1].stop_reason == 'tool_use'):
                    final_prompt_message = MessageParam(role='user', content="We've reached the maximum number of iterations. \n                        Please stop using tools now and provide your final comprehensive answer based on all tool results so far. \n                        At the beginning of your response, clearly indicate that your answer may be incomplete due to reaching the maximum number of tool usage iterations, \n                        and explain what additional information you would have needed to provide a more complete answer.")
                    messages.append(final_prompt_message)
                arguments = {'model': model, 'max_tokens': params.maxTokens, 'messages': messages, 'stop_sequences': params.stopSequences or [], 'tools': available_tools}
                if (system := (self.instruction or params.systemPrompt)):
                    arguments['system'] = system
                if params.metadata:
                    arguments = {**arguments, **params.metadata}
                self.logger.debug('Completion request arguments:', data=arguments)
                self._log_chat_progress(chat_turn=(len(messages) + 1) // 2, model=model)
                request = RequestCompletionRequest(config=config.anthropic, payload=arguments)
                self._annotate_span_for_completion_request(span, request, i)
                response: Message = await self.executor.execute(AnthropicCompletionTasks.request_completion_task, ensure_serializable(request))
                if isinstance(response, BaseException):
                    self.logger.error(f'Error: {response}')
                    span.record_exception(response)
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    break
                self.logger.debug(f'{model} response:', data=response)
                self._annotate_span_for_completion_response(span, response, i)
                iteration_input = response.usage.input_tokens
                iteration_output = response.usage.output_tokens
                total_input_tokens += iteration_input
                total_output_tokens += iteration_output
                response_as_message = self.convert_message_to_message_param(response)
                messages.append(response_as_message)
                responses.append(response)
                finish_reasons.append(response.stop_reason)
                if self.context.token_counter:
                    await self.context.token_counter.record_usage(input_tokens=iteration_input, output_tokens=iteration_output, model_name=model, provider=self.provider)
                if response.stop_reason == 'end_turn':
                    self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'end_turn'")
                    span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, ['end_turn'])
                    break
                elif response.stop_reason == 'stop_sequence':
                    self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'stop_sequence'")
                    span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, ['stop_sequence'])
                    break
                elif response.stop_reason == 'max_tokens':
                    self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'max_tokens'")
                    span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, ['max_tokens'])
                    break
                else:
                    for content in response.content:
                        if content.type == 'tool_use':
                            tool_name = content.name
                            tool_args = content.input
                            tool_use_id = content.id
                            tool_call_request = CallToolRequest(method='tools/call', params=CallToolRequestParams(name=tool_name, arguments=tool_args))
                            result = await self.call_tool(request=tool_call_request, tool_call_id=tool_use_id)
                            message = self.from_mcp_tool_result(result, tool_use_id)
                            messages.append(message)
            if params.use_history:
                self.history.set(messages)
            self._log_chat_finished(model=model)
            if self.context.tracing_enabled:
                span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, total_input_tokens)
                span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, total_output_tokens)
                span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)
                for i, response in enumerate(responses):
                    response_data = self.extract_response_message_attributes_for_tracing(response, prefix=f'response.{i}')
                    span.set_attributes(response_data)
            return responses

    async def generate_str(self, message, request_params: RequestParams | None=None) -> str:
        """
        Process a query using an LLM and available tools.
        The default implementation uses Claude as the LLM.
        Override this method to use a different LLM.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.{self.name}.generate_str') as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)
            self._annotate_span_for_generation_message(span, message)
            if self.context.tracing_enabled and request_params:
                AugmentedLLM.annotate_span_with_request_params(span, request_params)
            responses: List[Message] = await self.generate(message=message, request_params=request_params)
            final_text: List[str] = []
            for response in responses:
                for content in response.content:
                    if content.type == 'text':
                        final_text.append(content.text)
                    elif content.type == 'tool_use':
                        final_text.append(f'[Calling tool {content.name} with args {content.input}]')
            res = '\n'.join(final_text)
            span.set_attribute('response', res)
            return res

    async def generate_structured(self, message, response_model: Type[ModelT], request_params: RequestParams | None=None) -> ModelT:
        import json
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f'{self.__class__.__name__}.{self.name}.generate_structured') as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)
            self._annotate_span_for_generation_message(span, message)
            params = self.get_request_params(request_params)
            if self.context.tracing_enabled:
                AugmentedLLM.annotate_span_with_request_params(span, params)
            model_name = await self.select_model(params) or self.default_request_params.model
            span.set_attribute(GEN_AI_REQUEST_MODEL, model_name)
            messages: List[MessageParam] = []
            if params.use_history:
                messages.extend(self.history.get())
            messages.extend(AnthropicConverter.convert_mixed_messages_to_anthropic(message))
            schema = response_model.model_json_schema()
            tools: List[ToolParam] = [{'name': 'return_structured_output', 'description': 'Return the response in the required JSON format', 'input_schema': schema}]
            args = {'model': model_name, 'messages': messages, 'system': self.instruction or params.systemPrompt, 'tools': tools, 'tool_choice': {'type': 'tool', 'name': 'return_structured_output'}}
            if params.maxTokens is not None:
                args['max_tokens'] = params.maxTokens
            if params.stopSequences:
                args['stop_sequences'] = params.stopSequences
            base_url = None
            if self.context and self.context.config and self.context.config.anthropic:
                base_url = self.context.config.anthropic.base_url
                api_key = self.context.config.anthropic.api_key
                client = AsyncAnthropic(api_key=api_key, base_url=base_url)
            else:
                client = AsyncAnthropic()
            async with client:
                stream_method = client.messages.stream
                if all((hasattr(stream_method, attr) for attr in ('__aenter__', '__aexit__'))):
                    async with stream_method(**args) as stream:
                        final = await stream.get_final_message()
                else:
                    final = await client.messages.create(**args)
            for block in final.content:
                if getattr(block, 'type', None) == 'tool_use' and getattr(block, 'name', '') == 'return_structured_output':
                    data = getattr(block, 'input', None)
                    try:
                        if isinstance(data, str):
                            return response_model.model_validate(json.loads(data))
                        return response_model.model_validate(data)
                    except Exception:
                        break
            raise ValueError('Failed to obtain structured output from Anthropic response')

    @classmethod
    def convert_message_to_message_param(cls, message: Message, **kwargs) -> MessageParam:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        content = []
        for content_block in message.content:
            if content_block.type == 'text':
                content.append(TextBlockParam(type='text', text=content_block.text))
            elif content_block.type == 'tool_use':
                content.append(ToolUseBlockParam(type='tool_use', name=content_block.name, input=content_block.input, id=content_block.id))
        return MessageParam(role='assistant', content=content, **kwargs)

    def message_param_str(self, message: MessageParam) -> str:
        """Convert an input message to a string representation."""
        if message.get('content'):
            content = message['content']
            if isinstance(content, str):
                return content
            else:
                final_text: List[str] = []
                for block in content:
                    if block.text:
                        final_text.append(str(block.text))
                    else:
                        final_text.append(str(block))
                return '\n'.join(final_text)
        return str(message)

    def message_str(self, message: Message, content_only: bool=False) -> str:
        """Convert an output message to a string representation."""
        content = message.content
        if content:
            if isinstance(content, list):
                final_text: List[str] = []
                for block in content:
                    if block.text:
                        final_text.append(str(block.text))
                    else:
                        final_text.append(str(block))
                return '\n'.join(final_text)
            else:
                return str(content)
        elif content_only:
            return ''
        return str(message)

    def _extract_message_param_attributes_for_tracing(self, message_param: MessageParam, prefix: str='message') -> dict[str, Any]:
        """Return a flat dict of span attributes for a given MessageParam."""
        if not self.context.tracing_enabled:
            return {}
        attrs = {}
        attrs[f'{prefix}.role'] = message_param.get('role')
        message_content = message_param.get('content')
        if isinstance(message_content, str):
            attrs[f'{prefix}.content'] = message_content
        elif isinstance(message_content, list):
            for j, part in enumerate(message_content):
                message_content_prefix = f'{prefix}.content.{j}'
                attrs[f'{message_content_prefix}.type'] = part.get('type')
                match part.get('type'):
                    case 'text':
                        attrs[f'{message_content_prefix}.text'] = part.get('text')
                    case 'image':
                        source_type = part.get('source', {}).get('type')
                        attrs[f'{message_content_prefix}.source.type'] = source_type
                        if source_type == 'base64':
                            attrs[f'{message_content_prefix}.source.media_type'] = part.get('source', {}).get('media_type')
                        elif source_type == 'url':
                            attrs[f'{message_content_prefix}.source.url'] = part.get('source', {}).get('url')
                    case 'tool_use':
                        attrs[f'{message_content_prefix}.id'] = part.get('id')
                        attrs[f'{message_content_prefix}.name'] = part.get('name')
                    case 'tool_result':
                        attrs[f'{message_content_prefix}.tool_use_id'] = part.get('tool_use_id')
                        attrs[f'{message_content_prefix}.is_error'] = part.get('is_error')
                        part_content = part.get('content')
                        if isinstance(part_content, str):
                            attrs[f'{message_content_prefix}.content'] = part_content
                        elif isinstance(part_content, list):
                            for k, sub_part in enumerate(part_content):
                                sub_part_type = sub_part.get('type')
                                if sub_part_type == 'text':
                                    attrs[f'{message_content_prefix}.content.{k}.text'] = sub_part.get('text')
                                elif sub_part_type == 'image':
                                    sub_part_source = sub_part.get('source')
                                    sub_part_source_type = sub_part_source.get('type')
                                    attrs[f'{message_content_prefix}.content.{k}.source.type'] = sub_part_source_type
                                    if sub_part_source_type == 'base64':
                                        attrs[f'{message_content_prefix}.content.{k}.source.media_type'] = sub_part_source.get('media_type')
                                    elif sub_part_source_type == 'url':
                                        attrs[f'{message_content_prefix}.content.{k}.source.url'] = sub_part_source.get('url')
                    case 'document':
                        if part.get('context') is not None:
                            attrs[f'{message_content_prefix}.context'] = part.get('context')
                        if part.get('title') is not None:
                            attrs[f'{message_content_prefix}.title'] = part.get('title')
                        if part.get('citations') is not None:
                            attrs[f'{message_content_prefix}.citations.enabled'] = part.get('citations').get('enabled')
                        part_source_type = part.get('source', {}).get('type')
                        attrs[f'{message_content_prefix}.source.type'] = part_source_type
                        if part_source_type == 'text':
                            attrs[f'{message_content_prefix}.source.data'] = part.get('source', {}).get('data')
                        elif part_source_type == 'url':
                            attrs[f'{message_content_prefix}.source.url'] = part.get('source', {}).get('url')
                    case 'thinking':
                        attrs[f'{message_content_prefix}.thinking'] = part.get('thinking')
                        attrs[f'{message_content_prefix}.signature'] = part.get('signature')
                    case 'redacted_thinking':
                        attrs[f'{message_content_prefix}.redacted_thinking'] = part.get('data')
        return attrs

    def extract_response_message_attributes_for_tracing(self, message: Message, prefix: str | None=None) -> dict[str, Any]:
        """Return a flat dict of span attributes for a given Message."""
        if not self.context.tracing_enabled:
            return {}
        attr_prefix = f'{prefix}.' if prefix else ''
        attrs = {f'{attr_prefix}id': message.id, f'{attr_prefix}model': message.model, f'{attr_prefix}role': message.role}
        if message.stop_reason:
            attrs[f'{attr_prefix}{GEN_AI_RESPONSE_FINISH_REASONS}'] = [message.stop_reason]
        if message.stop_sequence:
            attrs[f'{attr_prefix}stop_sequence'] = message.stop_sequence
        if message.usage:
            attrs[f'{attr_prefix}{GEN_AI_USAGE_INPUT_TOKENS}'] = message.usage.input_tokens
            attrs[f'{attr_prefix}{GEN_AI_USAGE_OUTPUT_TOKENS}'] = message.usage.output_tokens
        for i, block in enumerate(message.content):
            attrs[f'{attr_prefix}content.{i}.type'] = block.type
            match block.type:
                case 'text':
                    attrs[f'{attr_prefix}content.{i}.text'] = block.text
                case 'tool_use':
                    attrs[f'{attr_prefix}content.{i}.tool_use_id'] = block.id
                    attrs[f'{attr_prefix}content.{i}.name'] = block.name
                case 'thinking':
                    attrs[f'{attr_prefix}content.{i}.thinking'] = block.thinking
                    attrs[f'{attr_prefix}content.{i}.signature'] = block.signature
                case 'redacted_thinking':
                    attrs[f'{attr_prefix}content.{i}.redacted_thinking'] = block.data
        return attrs

    def _annotate_span_for_completion_request(self, span: trace.Span, request: RequestCompletionRequest, turn: int):
        """Annotate the span with the completion request as an event."""
        if not self.context.tracing_enabled:
            return
        event_data = {'completion.request.turn': turn}
        for key, value in request.payload.items():
            if key == 'messages':
                for i, message in enumerate(cast(List[MessageParam], value)):
                    event_data.update(self._extract_message_param_attributes_for_tracing(message, prefix=f'messages.{i}'))
            elif key == 'tools':
                if value is not None:
                    event_data['tools'] = [tool.get('name') for tool in value]
            elif is_otel_serializable(value):
                event_data[key] = value
        event_name = f'completion.request.{turn}'
        latest_message_role = request.payload.get('messages', [{}])[-1].get('role')
        if latest_message_role:
            event_name = f'gen_ai.{latest_message_role}.message'
        span.add_event(event_name, event_data)

    def _annotate_span_for_completion_response(self, span: trace.Span, response: Message, turn: int):
        """Annotate the span with the completion response as an event."""
        if not self.context.tracing_enabled:
            return
        event_data = {'completion.response.turn': turn}
        event_data.update(self.extract_response_message_attributes_for_tracing(response))
        span.add_event(f'gen_ai.{response.role}.message', event_data)

class Context(MCPContext):
    """
    Context that is passed around through the application.
    This is a global context that is shared across the application.
    """
    config: Optional[Settings] = None
    executor: Optional[Executor] = None
    human_input_handler: Optional[HumanInputCallback] = None
    elicitation_handler: Optional[ElicitationCallback] = None
    signal_notification: Optional[SignalWaitCallback] = None
    model_selector: Optional[ModelSelector] = None
    session_id: str | None = None
    app: Optional['MCPApp'] = None
    loaded_subagents: List['AgentSpec'] = []
    server_registry: Optional[ServerRegistry] = None
    task_registry: Optional[ActivityRegistry] = None
    signal_registry: Optional[SignalRegistry] = None
    decorator_registry: Optional[DecoratorRegistry] = None
    workflow_registry: Optional['WorkflowRegistry'] = None
    tracer: Optional[trace.Tracer] = None
    tracing_enabled: bool = False
    tracing_config: Optional[TracingConfig] = None
    token_counter: Optional[TokenCounter] = None
    gateway_url: str | None = None
    gateway_token: str | None = None
    token_store: Optional[TokenStore] = None
    token_manager: Optional[TokenManager] = None
    identity_registry: Dict[str, OAuthUserIdentity] = Field(default_factory=dict)
    request_session_id: str | None = None
    request_identity: OAuthUserIdentity | None = None
    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)

    @property
    def upstream_session(self) -> ServerSession | None:
        """
        Resolve the active upstream session, preferring the request-scoped clone.

        The base application context keeps an optional session used by scripts or
        tests that set MCPApp.upstream_session directly. During an MCP request the
        request-bound context is stored in a ContextVar; whenever callers reach the
        base context while that request is active we return the request's session
        instead of whichever client touched the base context last.
        """
        request_ctx = get_current_request_context()
        if request_ctx is not None:
            if request_ctx is self:
                return getattr(self, '_upstream_session', None)
            current = request_ctx
            while current is not None:
                parent_ctx = getattr(current, '_parent_context', None)
                if parent_ctx is self:
                    return getattr(current, '_upstream_session', None)
                current = parent_ctx
        explicit = getattr(self, '_upstream_session', None)
        if explicit is not None:
            return explicit
        parent = getattr(self, '_parent_context', None)
        if parent is not None:
            return getattr(parent, '_upstream_session', None)
        return None

    @upstream_session.setter
    def upstream_session(self, value: ServerSession | None) -> None:
        object.__setattr__(self, '_upstream_session', value)

    @property
    def mcp(self) -> FastMCP | None:
        return self.app.mcp if self.app else None

    @property
    def fastmcp(self) -> FastMCP | None:
        """Return the FastMCP instance if available.

        Prefer the active request-bound FastMCP instance if present; otherwise
        fall back to the app's configured FastMCP server. Returns None if neither
        is available. This is more forgiving than the FastMCP Context default,
        which raises outside of a request.
        """
        try:
            if getattr(self, '_fastmcp', None) is not None:
                return getattr(self, '_fastmcp', None)
        except Exception:
            pass
        return self.mcp

    @property
    def session(self) -> ServerSession | None:
        """Best-effort ServerSession for upstream communication.

        Priority:
        - If explicitly provided, use `upstream_session`.
        - If running within an active FastMCP request, use parent session.
        - If an app FastMCP exists, use its current request context if any.

        Returns None when no session can be resolved (e.g., local scripts).
        """
        explicit = getattr(self, 'upstream_session', None)
        if explicit is not None:
            return explicit
        try:
            return super().session
        except Exception:
            pass
        try:
            mcp = self.mcp
            if mcp is not None:
                ctx = mcp.get_context()
                try:
                    return getattr(ctx, 'session', None)
                except Exception:
                    return None
        except Exception:
            pass
        return None

    @property
    def logger(self) -> 'Logger':
        if self.app:
            return self.app.logger
        namespace_components = ['mcp_agent', 'context']
        try:
            if getattr(self, 'session_id', None):
                namespace_components.append(str(self.session_id))
        except Exception:
            pass
        namespace = '.'.join(namespace_components)
        logger = get_logger(namespace, session_id=getattr(self, 'session_id', None), context=self)
        try:
            setattr(logger, '_bound_context', self)
        except Exception:
            pass
        return logger

    @property
    def name(self) -> str | None:
        if self.app and getattr(self.app, 'name', None):
            return self.app.name
        return None

    @property
    def description(self) -> str | None:
        if self.app and getattr(self.app, 'description', None):
            return self.app.description
        return None

    def bind_request(self, request_context: Any, fastmcp: FastMCP | None=None) -> 'Context':
        """Return a shallow-copied Context bound to a specific FastMCP request.

        - Shares app-wide state (config, registries, token counter, etc.) with the original Context
        - Attaches `_request_context` and `_fastmcp` so FastMCP Context APIs work during the request
        - Does not mutate the original Context (safe for concurrent requests)
        """
        bound: Context = self.model_copy(deep=False)
        object.__setattr__(bound, '_upstream_session', None)
        try:
            object.__setattr__(bound, '_parent_context', self)
        except Exception:
            pass
        bound.request_session_id = None
        bound.request_identity = None
        try:
            setattr(bound, '_request_context', request_context)
        except Exception:
            pass
        try:
            if fastmcp is None:
                fastmcp = getattr(self, '_fastmcp', None) or self.mcp
            setattr(bound, '_fastmcp', fastmcp)
        except Exception:
            pass
        return bound

    @property
    def client_id(self) -> str | None:
        try:
            return super().client_id
        except Exception:
            return None

    @property
    def request_id(self) -> str:
        try:
            return super().request_id
        except Exception:
            try:
                return str(self.session_id) if getattr(self, 'session_id', None) else ''
            except Exception:
                return ''

    async def log(self, level: "Literal['debug', 'info', 'warning', 'error']", message: str, *, logger_name: str | None=None) -> None:
        """Send a log to the client if possible; otherwise, log locally.

        Matches FastMCP Context API but avoids raising when no request context
        is active by falling back to the app's logger.
        """
        try:
            _ = self.request_context
        except Exception:
            pass
        else:
            try:
                return await super().log(level, message, logger_name=logger_name)
            except Exception:
                pass
        try:
            _logger = self.logger
            if _logger is not None:
                if level == 'debug':
                    _logger.debug(message)
                elif level == 'warning':
                    _logger.warning(message)
                elif level == 'error':
                    _logger.error(message)
                else:
                    _logger.info(message)
        except Exception:
            pass

    async def report_progress(self, progress: float, total: float | None=None, message: str | None=None) -> None:
        """Report progress to the client if a request is active.

        Outside of a request (e.g., local scripts), this is a no-op to avoid
        runtime errors as no progressToken exists.
        """
        try:
            _ = self.request_context
            return await super().report_progress(progress, total, message)
        except Exception:
            return None

    async def read_resource(self, uri: Any) -> Any:
        """Read a resource via FastMCP if possible; otherwise raise clearly.

        This provides a friendlier error outside of a request and supports
        fallback to the app's FastMCP instance if available.
        """
        try:
            return await super().read_resource(uri)
        except Exception:
            pass
        try:
            mcp = self.mcp
            if mcp is not None:
                return await mcp.read_resource(uri)
        except Exception:
            pass
        raise ValueError('read_resource is only available when an MCP server is active.')

def create_preferences_table(cost: float, speed: float, intelligence: float, provider: str, min_tokens: Optional[int]=None, max_tokens: Optional[int]=None, tool_calling: Optional[bool]=None, structured_outputs: Optional[bool]=None) -> Table:
    table = Table(title='Current Preferences', show_header=True, header_style='bold magenta')
    table.add_column('Priority', style='cyan')
    table.add_column('Value', style='green')
    table.add_row('Cost', f'{cost:.2f}')
    table.add_row('Speed', f'{speed:.2f}')
    table.add_row('Intelligence', f'{intelligence:.2f}')
    table.add_row('Provider', provider)
    if min_tokens is not None:
        table.add_row('Min Context Tokens', f'{min_tokens:,}')
    if max_tokens is not None:
        table.add_row('Max Context Tokens', f'{max_tokens:,}')
    if tool_calling is not None:
        table.add_row('Tool Calling', 'Required' if tool_calling else 'Not Required')
    if structured_outputs is not None:
        table.add_row('Structured Outputs', 'Required' if structured_outputs else 'Not Required')
    return table

def print_to_console(message: str):
    """
    A simple function that prints a message to the console.
    """
    logger = get_logger('workflow_router.print_to_console')
    logger.info(message)

