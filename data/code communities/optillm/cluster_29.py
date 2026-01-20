# Cluster 29

def extract_key_information(system_message, text: str, query: str, client, model: str) -> List[str]:
    messages = [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': f"\n'''text\n{text}\n'''\nCopy over all context relevant to the query: {query}\nProvide the answer in the format: <YES/NO>#<Relevant context>.\nHere are rules:\n- If you don't know how to answer the query - start your answer with NO#\n- If the text is not related to the query - start your answer with NO#\n- If you can extract relevant information - start your answer with YES#\n- If the text does not mention the person by name - start your answer with NO#\nExample answers:\n- YES#Western philosophy originated in Ancient Greece in the 6th century BCE with the pre-Socratics.\n- NO#No relevant context.\n"}]
    try:
        response = client.chat.completions.create(model=model, messages=messages, max_tokens=1000)
        key_info = response.choices[0].message.content.strip()
    except Exception as e:
        print(f'Error parsing content: {str(e)}')
        return ([], 0)
    margins = []
    if classify_margin(key_info):
        margins.append(key_info.split('#', 1)[1])
    return (margins, response.usage.completion_tokens)

def classify_margin(margin):
    return margin.startswith('YES#')

def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    memory = Memory()
    query, context = extract_query(initial_query)
    completion_tokens = 0
    chunk_size = 100000
    for i in range(0, len(context), chunk_size):
        chunk = context[i:i + chunk_size]
        key_info, tokens = extract_key_information(system_prompt, chunk, query, client, model)
        completion_tokens += tokens
        for info in key_info:
            memory.add(info)
    relevant_info = memory.get_relevant(query)
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': f"\n\nI asked my assistant to read and analyse the above content page by page to help you complete this task. These are margin notes left on each page:\n'''text\n{relevant_info}\n'''\nRead again the note(s), take a deep breath and answer the query.\n{query}\n"}]
    response = client.chat.completions.create(model=model, messages=messages)
    final_response = response.choices[0].message.content.strip()
    completion_tokens += response.usage.completion_tokens
    return (final_response, completion_tokens)

def extract_query(text: str) -> Tuple[str, str]:
    query_index = text.rfind('Query:')
    if query_index != -1:
        context = text[:query_index].strip()
        query = text[query_index + 6:].strip()
    else:
        sentences = re.split('(?<=[.!?])\\s+', text.strip())
        if len(sentences) > 1:
            context = ' '.join(sentences[:-1])
            query = sentences[-1]
        else:
            context = text
            query = 'What is the main point of this text?'
    return (query, context)

