# Cluster 6

def stream_chat():
    """流式"""
    streamer = TextIteratorStreamer(tokenizer)
    history = []
    clear_command = 'cls' if os.name == 'nt' else 'clear'
    while True:
        query = input('\nUser:')
        if query.strip() == 'stop':
            break
        if query.strip() == 'clear':
            history = []
            os.system(clear_command)
            continue
        query_new = build_prompt(query, history)
        inputs = tokenizer.encode(query_new, return_tensors='pt', add_special_tokens=False).to(device)
        generation_kwargs = dict({'input_ids': inputs}, streamer=streamer, max_new_tokens=512)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        response = ''
        for new_text in streamer:
            os.system(clear_command)
            response += new_text
            print(build_cli_history(history + [(query, response[len(query_new):])]), flush=True)
        os.system(clear_command)
        print(build_cli_history(history + [(query, response[len(query_new):])]), flush=True)

