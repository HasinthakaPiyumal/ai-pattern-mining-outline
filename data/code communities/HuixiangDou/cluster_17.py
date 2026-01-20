# Cluster 17

def test():
    tools = [{'name_for_human': '谷歌搜索', 'name_for_model': 'google_search', 'description_for_model': '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。', 'parameters': [{'name': 'search_query', 'description': '搜索关键词或短语', 'required': True, 'schema': {'type': 'string'}}]}]
    history = []
    for query in ['请问mmdet3.0依赖mmcv哪个版本', 'openmmlab和上海 AI Lab 是什么关系', '如何安装 mmdeploy', 'ncnn 全称是啥', '如果我要从高空检测安全帽，我应该用 mmdet 还是 mmrotate ']:
        print(f"User's Query:\n{query}\n")
        response, history = llm_with_plugin(prompt=query, history=history, list_of_plugin_info=tools)
        print(f"Qwen's Response:\n{response}\n")

def llm_with_plugin(prompt: str, history, list_of_plugin_info=()):
    chat_history = [(x['user'], x['bot']) for x in history] + [(prompt, '')]
    planning_prompt = build_input_text(chat_history, list_of_plugin_info)
    text = ''
    while True:
        output = text_completion(planning_prompt + text, stop_words=['Observation:', 'Observation:\n'])
        action, action_input, output = parse_latest_plugin_call(output)
        if action:
            observation = call_plugin(action, action_input)
            output += f'\nObservation: {observation}\nThought:'
            text += output
        else:
            text += output
            break
    new_history = []
    new_history.extend(history)
    new_history.append({'user': prompt, 'bot': text})
    return (text, new_history)

