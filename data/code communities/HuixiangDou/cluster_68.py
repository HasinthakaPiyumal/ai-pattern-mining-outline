# Cluster 68

def list_model():
    model_list = client.models.list()
    model_data = model_list.data
    for i, model in enumerate(model_data):
        print(f'model[{i}]:', model.id)

def generate():
    """Test generate."""
    messages = [{'role': 'system', 'content': '你是一个语文专家，擅长对句子的结构进行分析'}, {'role': 'user', 'content': prompt}]
    whole_input = str(messages)
    print(whole_input)
    try:
        completion = client.chat.completions.create(model='kimi-k2-0711-preview', messages=messages, temperature=0.1, n=5)
    except Exception as e:
        return (prompt, str(e))
    results = []
    for choice in completion.choices:
        results.append(choice.message.content)
    return (prompt, results)

def main(args):
    n_garbage = args.max_tokens - 1000
    passed_tests = 0
    for j in range(args.num_tests):
        prompt, pass_key = generate_prompt_landmark(n_garbage=n_garbage, seed=5120 + j)
        try:
            response = generate(prompt='hello')
            print(response)
            response = generate(prompt=prompt)
        except Exception as e:
            print(e)
        print('result: ', response, pass_key)
        if pass_key in response.content:
            passed_tests += 1
    precision = passed_tests / args.num_tests
    print('precision {} @ {}'.format(precision, args.max_tokens))

def generate_prompt_landmark(n_garbage=60000, seed=666):
    """Generates a text file and inserts an passkey at a random position."""
    from numpy import random
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix
    task_description = 'There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.'
    garbage = 'The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.'
    garbage_inf = ' '.join([garbage] * 384000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 192000)
    information_line = f'The pass key is {pass_key}. Remember it. {pass_key} is the pass key.'
    final_question = 'What is the pass key? The pass key is'
    lines = [task_description, garbage_prefix, information_line, garbage_suffix, final_question]
    random.set_state(rnd_state)
    return ('\n'.join(lines), str(pass_key))

