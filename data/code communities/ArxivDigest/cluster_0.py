# Cluster 0

def generate_relevance_score(all_papers, query, model_name='gpt-3.5-turbo-16k', threshold_score=8, num_paper_in_prompt=4, temperature=0.4, top_p=1.0, sorting=True):
    ans_data = []
    request_idx = 1
    hallucination = False
    for id in tqdm.tqdm(range(0, len(all_papers), num_paper_in_prompt)):
        prompt_papers = all_papers[id:id + num_paper_in_prompt]
        prompt = encode_prompt(query, prompt_papers)
        decoding_args = utils.OpenAIDecodingArguments(temperature=temperature, n=1, max_tokens=128 * num_paper_in_prompt, top_p=top_p)
        request_start = time.time()
        response = utils.openai_completion(prompts=prompt, model_name=model_name, batch_size=1, decoding_args=decoding_args, logit_bias={'100257': -100})
        print('response', response['message']['content'])
        request_duration = time.time() - request_start
        process_start = time.time()
        batch_data, hallu = post_process_chat_gpt_response(prompt_papers, response, threshold_score=threshold_score)
        hallucination = hallucination or hallu
        ans_data.extend(batch_data)
        print(f'Request {request_idx + 1} took {request_duration:.2f}s')
        print(f'Post-processing took {time.time() - process_start:.2f}s')
    if sorting:
        ans_data = sorted(ans_data, key=lambda x: int(x['Relevancy score']), reverse=True)
    return (ans_data, hallucination)

def encode_prompt(query, prompt_papers):
    """Encode multiple prompt instructions into a single string."""
    prompt = open('src/relevancy_prompt.txt').read() + '\n'
    prompt += query['interest']
    for idx, task_dict in enumerate(prompt_papers):
        title, authors, abstract = (task_dict['title'], task_dict['authors'], task_dict['abstract'])
        if not title:
            raise
        prompt += f'###\n'
        prompt += f'{idx + 1}. Title: {title}\n'
        prompt += f'{idx + 1}. Authors: {authors}\n'
        prompt += f'{idx + 1}. Abstract: {abstract}\n'
    prompt += f'\n Generate response:\n1.'
    print(prompt)
    return prompt

def post_process_chat_gpt_response(paper_data, response, threshold_score=8):
    selected_data = []
    if response is None:
        return []
    json_items = response['message']['content'].replace('\n\n', '\n').split('\n')
    pattern = '^\\d+\\. |\\\\'
    import pprint
    try:
        score_items = [json.loads(re.sub(pattern, '', line)) for line in json_items if 'relevancy score' in line.lower()]
    except Exception:
        pprint.pprint([re.sub(pattern, '', line) for line in json_items if 'relevancy score' in line.lower()])
        raise RuntimeError('failed')
    pprint.pprint(score_items)
    scores = []
    for item in score_items:
        temp = item['Relevancy score']
        if isinstance(temp, str) and '/' in temp:
            scores.append(int(temp.split('/')[0]))
        else:
            scores.append(int(temp))
    if len(score_items) != len(paper_data):
        score_items = score_items[:len(paper_data)]
        hallucination = True
    else:
        hallucination = False
    for idx, inst in enumerate(score_items):
        if scores[idx] < threshold_score:
            continue
        output_str = 'Title: ' + paper_data[idx]['title'] + '\n'
        output_str += 'Authors: ' + paper_data[idx]['authors'] + '\n'
        output_str += 'Link: ' + paper_data[idx]['main_page'] + '\n'
        for key, value in inst.items():
            paper_data[idx][key] = value
            output_str += str(key) + ': ' + str(value) + '\n'
        paper_data[idx]['summarized_text'] = output_str
        selected_data.append(paper_data[idx])
    return (selected_data, hallucination)

