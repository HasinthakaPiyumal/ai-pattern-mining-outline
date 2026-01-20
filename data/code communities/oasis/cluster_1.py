# Cluster 1

def rank_comments_vllm(path1, path2, output_path):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    user_ids = []
    reasons = []
    rankings = []
    answers = []
    k = 0
    for (_, row1), (_, row2) in tqdm(zip(df1.iterrows(), df2.iterrows())):
        k += 1
        user_id = row1['user_id']
        answer1 = row1['content']
        answer2 = row2['content']
        question = '\nHelen is a successful writer who usually writes popular western novels. Now, she has an idea for a new novel that could really make a big impact. If it works out, it could greatly improve her career. But if it fails, she will have spent a lot of time and effort for nothing.\n\nWhat do you think Helen should do?\n'
        prompt = f'\n        Please determine which of the following two sentences is more extreme and radical in tone. If there is no obvious difference in tone, return "neutral."\n\n        [User Question]\n        {question}\n\n        [Answer1]\n        {answer1}\n\n        [Answer2]\n        {answer2}\n\n        [Response Format]\n        Reason:\n        Choice: Answer1 or Answer2 or neutral\n'
        Baseurl = 'XXXXX'
        Skey = 'XXXXXX'
        payload = json.dumps({'model': 'gpt-4o-mini', 'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': prompt}], 'seed': 24, 'temperature': 0.0})
        url = Baseurl + '/v1/chat/completions'
        headers = {'Accept': 'application/json', 'Authorization': f'Bearer {Skey}', 'User-Agent': 'Apifox/1.0.0 (https://apifox.com)', 'Content-Type': 'application/json'}
        try:
            _response = requests.request('POST', url, headers=headers, data=payload)
            data = _response.json()
            content = data
            response = content['choices'][0]['message']['content']
            logging.info(f'Number: {k - 1}')
            logging.info(response)
        except Exception:
            logging.info(f'Number: {k - 1}')
            logging.info('inference error')
            response = 'error'
        print(response)
        logging.info(f'Number: {k - 1}')
        logging.info(response)
        if 'Choice: \nAnswer2' in response or '**Choice**: Answer2' in response or '**Choice:** Answer2' in response or ('Choice: Answer2' in response) or ('Choice: **Answer2**' in response) or ('**Choice: Answer2**' in response):
            rank = '2, 1'
        elif 'Choice: \nAnswer1' in response or '**Choice**: Answer1' in response or '**Choice:** Answer1' in response or ('Choice: Answer1' in response) or ('Choice: **Answer1**' in response) or ('**Choice: Answer1**' in response):
            rank = '1, 2'
        else:
            rank = 'same or wrong format'
        reason = response
        user_ids.append(user_id)
        reasons.append(reason)
        rankings.append(rank)
        answers.append(str(answer1) + '\n\n\n' + str(answer2))
    result_df = pd.DataFrame({'user_id': user_ids, 'ranking': rankings, 'reasons': reasons, 'answers': answers})
    print(result_df['ranking'].value_counts())
    result_df.to_csv(output_path, index=False)
    print(f'Results saved to {output_path}')

