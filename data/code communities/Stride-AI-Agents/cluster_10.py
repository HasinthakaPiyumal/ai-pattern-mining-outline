# Cluster 10

def conversation_was_successful(messages) -> bool:
    conversation = f'CONVERSATION: {json.dumps(messages)}'
    result: BoolEvalResult = evaluate_with_llm_bool(CONVERSATIONAL_EVAL_SYSTEM_PROMPT, conversation)
    return result.value

def evaluate_with_llm_bool(instruction, data) -> BoolEvalResult:
    eval_result, _ = __client.chat.completions.create_with_completion(model='gpt-4o', messages=[{'role': 'system', 'content': instruction}, {'role': 'user', 'content': data}], response_model=BoolEvalResult)
    return eval_result

@pytest.mark.parametrize('messages', [[{'role': 'user', 'content': 'Who is the lead singer of U2'}, {'role': 'assistant', 'content': 'Bono is the lead singer of U2.'}], [{'role': 'user', 'content': 'Hello!'}, {'role': 'assistant', 'content': 'Hi there! How can I assist you today?'}, {'role': 'user', 'content': 'I want to make a refund.'}, {'role': 'tool', 'tool_name': 'transfer_to_refunds'}, {'role': 'user', 'content': 'Thank you!'}, {'role': 'assistant', 'content': "You're welcome! Have a great day!"}]])
def test_conversation_is_successful(messages):
    result = conversation_was_successful(messages)
    assert result == True

