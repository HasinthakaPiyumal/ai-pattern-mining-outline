# Cluster 37

def test_get_group_initial_message():
    """Test getting initial message for group chat."""
    message = get_group_initial_message()
    assert isinstance(message, dict)
    assert 'role' in message
    assert 'content' in message
    assert message['role'] == 'system'
    assert 'CURRENT ANSWER' in message['content']
    assert 'ORIGINAL MESSAGE' in message['content']

def get_group_initial_message() -> Dict[str, Any] | None:
    """
    Create the initial system message for group chat.

    Returns:
        Dict with role and content for initial system message
    """
    initial_message = f'\n    CURRENT ANSWER from multiple agents for final response to a message is given.\n    Different agents may have different builtin tools and capabilities.\n    Does the best CURRENT ANSWER address the ORIGINAL MESSAGE well?\n\n    If CURRENT ANSWER is given, digest existing answers, combine their strengths, and do additional work to address their weaknesses.\n    if you think CURRENT ANSWER is good enough, you can also use it as your answer.\n\n    *Note*: The CURRENT TIME is **{time.strftime('%Y-%m-%d %H:%M:%S')}**.\n    '
    return {'role': 'system', 'content': initial_message}

