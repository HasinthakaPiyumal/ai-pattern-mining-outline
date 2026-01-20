# Cluster 5

def build_cli_history(history):
    prompt = ''
    for query, response in history:
        prompt += f'\n\nUser：{query.strip()}'
        prompt += f'\n\nRobot：{response.strip()}'
    return prompt

def chat():
    """
    非流式输出
    """
    history = []
    clear_command = 'cls' if os.name == 'nt' else 'clear'
    while True:
        query = input('\n输入:')
        if query.strip() == 'stop':
            break
        if query.strip() == 'clear':
            history = []
            os.system(clear_command)
            continue
        inputs = tokenizer.encode(query, return_tensors='pt', add_special_tokens=False).to(device)
        response = model.generate(inputs)
        response = tokenizer.decode(response[0].cpu(), skip_special_tokens=True)
        os.system(clear_command)
        print(build_cli_history(history + [(query, response)]), flush=True)

def stream_chat():
    """
    流式输出
    """
    streamer = TextIteratorStreamer(tokenizer)
    history = []
    clear_command = 'cls' if os.name == 'nt' else 'clear'
    while True:
        query = input('\n输入:')
        if query.strip() == 'stop':
            break
        if query.strip() == 'clear':
            history = []
            os.system(clear_command)
            continue
        inputs = tokenizer.encode(query, return_tensors='pt', add_special_tokens=False).to(device)
        generation_kwargs = dict({'input_ids': inputs}, streamer=streamer, max_new_tokens=512)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        response = ''
        for new_text in streamer:
            os.system(clear_command)
            response += new_text
            print(build_cli_history(history + [(query, response)]), flush=True)
        os.system(clear_command)
        print(build_cli_history(history + [(query, response)]), flush=True)

