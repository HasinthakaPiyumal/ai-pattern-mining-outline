# Cluster 3

def filter_log_node(feature_types: list[LogFeature], feature_values: list[str], feature_ops: list[FeatureOps], node: SpanNode, is_github_pr: bool=False) -> SpanNode:
    """Filter the log node based on the log node features.

    Recursively filters logs in the span tree based on the provided criteria.
    All three lists (feature_types, feature_values, feature_ops) must have the
    same length and correspond to each other exactly.

    Args:
        feature_types (list[LogFeature]): List of LogFeature enums specifying
            which features to filter on.
        feature_values (list[str]): List of string values to compare against.
        feature_ops (list[FeatureOps]): List of FeatureOps enums specifying the
            perform.
        node: SpanNode to filter.
        is_github_pr (bool): Whether the current node is a GitHub PR.

    Returns:
        SpanNode: A new filtered SpanNode with only logs and children that
            match the criteria.
    """

    def matches_filters(log: LogNode) -> bool:
        """Check if a log matches all filter criteria."""
        for feature_type, feature_value, feature_op in zip(feature_types, feature_values, feature_ops):
            log_value = get_log_feature_value(log, feature_type)
            if not apply_operation(str(log_value), str(feature_value), feature_op):
                return False
        return True
    if len(feature_types) != len(feature_values) or len(feature_types) != len(feature_ops):
        raise ValueError('feature_types, feature_values, and feature_ops must have the same length')
    filtered_logs = [log for log in node.logs if matches_filters(log)]
    filtered_children = []
    for child in node.children_spans:
        filtered_child = filter_log_node(feature_types, feature_values, feature_ops, child)
        if filtered_child.logs or filtered_child.children_spans:
            filtered_children.append(filtered_child)
    return SpanNode(span_id=node.span_id, func_full_name=node.func_full_name, span_latency=node.span_latency, span_utc_start_time=node.span_utc_start_time, span_utc_end_time=node.span_utc_end_time, logs=filtered_logs, children_spans=filtered_children)

def get_trace_context_messages(context: str) -> list[str]:
    """
    Convert trace context into formatted message chunks.

    Chunks large trace contexts and formats them with appropriate headers.
    Used by agents that work with trace data (SingleRCAAgent, CodeAgent).

    Args:
        context (str): The trace context to be chunked (usually JSON string)

    Returns:
        list[str]: List of context message chunks with formatting
    """
    context_chunks = list(sequential_chunk(context))
    if len(context_chunks) == 1:
        messages = [f'\n\nHere is the structure of the tree with related information:\n\n{context_chunks[0]}']
        print(f'trace_context_messages: {messages}')
        return messages
    messages: list[str] = []
    for i, chunk in enumerate(context_chunks):
        messages.append(f'\n\nHere is the structure of the tree with related information of the {i + 1}th chunk of the tree:\n\n{chunk}')
    return messages

def build_chat_history_messages(chat_history: list[dict] | None, max_records: int=MAX_PREV_RECORD) -> list[dict[str, str]]:
    """
    Process and format chat history for agent context.

    Filters out system messages (github, statistics) and extracts the
    appropriate content field (user_message for user role, content otherwise).
    Returns the last N records to maintain context window limits.

    Args:
        chat_history (list[dict] | None): Raw chat history from database
        max_records (int): Maximum number of history records to include

    Returns:
        list[dict[str, str]]: Formatted messages with 'role' and 'content' keys
    """
    if chat_history is None:
        return []
    filtered_history = [chat for chat in chat_history if chat['role'] not in ['github', 'statistics']]
    recent_history = filtered_history[-max_records:]
    messages = []
    for record in recent_history:
        if 'user_message' in record and record['user_message'] is not None:
            content = record['user_message']
        else:
            content = record['content']
        messages.append({'role': record['role'], 'content': content})
    print(f'context messages: {messages}')
    return messages

