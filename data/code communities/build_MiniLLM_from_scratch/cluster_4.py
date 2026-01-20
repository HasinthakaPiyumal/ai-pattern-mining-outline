# Cluster 4

def chat():
    """非流式"""
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
        inputs = tokenizer.encode(build_prompt(query, history), return_tensors='pt', add_special_tokens=False).to(device)
        response = model.generate(inputs)
        response = tokenizer.decode(response[0].cpu(), skip_special_tokens=True)
        os.system(clear_command)
        print(build_cli_history(history + [(query, response)]), flush=True)

def build_prompt(query, history) -> str:
    texts = ''
    for user_input, response in history:
        texts += f'{HUMAN}{user_input}{ROBOT}{response}'
    texts += f'{HUMAN}{query}{ROBOT}'
    return texts

