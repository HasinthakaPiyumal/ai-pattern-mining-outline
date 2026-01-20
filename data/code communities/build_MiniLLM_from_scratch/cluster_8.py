# Cluster 8

def collect_tokens(process_one, filename, data_format: Literal['jsonl', 'json']='jsonl'):
    """各个函数通用的处理token的方式"""

    def process_data_slice(data_slice, fi):
        if not USE_PARALLEL:
            log_info_once('Use single process to process data, maybe slow')
            train_samples = [process_one(line) for line in data_slice]
        else:
            log_info_once('Use multiprocess to accelerate data process')
            train_samples = parallel_apply(func=process_one, iterable=data_slice, workers=WORKERS, max_queue_size=MAX_QUEUE_SIZE, dummy=False, callback=None, unordered=False)
        train_samples = [{'input_ids': i[0], 'labels': i[1]} for i in train_samples if i[0] is not None and len(i[0]) > 1]
        save_path = os.path.join(args.dataset_save_dir, filename.replace('/', '--').replace('.jsonl', '').replace('.json', '') + f'_{fi}.jsonl')
        with open(save_path, 'w') as f:
            for item in train_samples:
                json.dump(item, f)
                f.write('\n')
        return len(train_samples)
    data_path = os.path.join(args.dataset_src_dir, filename)
    os.makedirs(args.dataset_save_dir, exist_ok=True)
    all_count = 0
    if data_format == 'json':
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if args.max_samples is not None:
            data = data[:args.max_samples]
        total_steps = int(np.ceil(len(data) / args.max_samples_per_file) * args.max_samples_per_file)
        for fi, start in enumerate(range(0, total_steps, args.max_samples_per_file), start=1):
            data_slice = data[start:start + args.max_samples_per_file]
            if len(data_slice) == 0:
                continue
            all_count += process_data_slice(data_slice, fi)
    else:
        data = []
        f = open(data_path, 'r', encoding='utf-8')
        fi = 1
        while True:
            line = f.readline()
            if not line:
                break
            data.append(line)
            data_len = len(data)
            if data_len >= args.max_samples_per_file:
                all_count += process_data_slice(data, fi)
                data = []
                fi += 1
            if args.max_samples is not None and data_len >= args.max_samples:
                break
        if len(data) > 0:
            all_count += process_data_slice(data, fi)
    return all_count

def process_alpaca(filename, tokenizer):
    """alpaca_gpt4_data_zh.json"""

    def process_one(per):
        q = tokenizer.encode(HUMAN + per['instruction'] + per['input'] + ROBOT, add_special_tokens=False)
        a = tokenizer.encode(per['output'], add_special_tokens=False)
        if len(q) + len(a) >= args.MAX_LENGTH:
            return (None, None)
        input_ids = q + a
        labels = [args.pad_token_id] * (len(q) - 1) + input_ids[len(q):] + [args.eos_token_id]
        assert len(input_ids) == len(labels)
        return (input_ids, labels)
    return collect_tokens(process_one, filename, data_format='json')

def process_self_cognition(filename, tokenizer):
    """Tongjilibo/self_cognition"""

    def replace_placeholder(query):
        mapping_ = {'<NAME>': args.name, '<AUTHOR>': args.author, '<DATE>': args.date}
        for key, value in mapping_.items():
            query = query.replace(key, value)
        return query

    def process_one(per):
        input = replace_placeholder(HUMAN + per['instruction'] + per['input'] + ROBOT)
        output = replace_placeholder(per['output'])
        q = tokenizer.encode(input, add_special_tokens=False)
        a = tokenizer.encode(output, add_special_tokens=False)
        if len(q) + len(a) >= args.MAX_LENGTH:
            return (None, None)
        input_ids = q + a
        labels = [args.pad_token_id] * (len(q) - 1) + input_ids[len(q):] + [args.eos_token_id]
        assert len(input_ids) == len(labels)
        return (input_ids, labels)
    return collect_tokens(process_one, filename, data_format='json')

def process_belle(filename, tokenizer):
    """Belle_open_source_1M.json"""

    def process_one(line):
        if not line:
            return (None, None)
        per = json.loads(line)
        q = tokenizer.encode(HUMAN + per['instruction'] + per['input'] + ROBOT, add_special_tokens=False)
        a = tokenizer.encode(per['output'], add_special_tokens=False)
        if len(q) + len(a) >= args.MAX_LENGTH:
            return (None, None)
        input_ids = q + a
        labels = [args.pad_token_id] * (len(q) - 1) + input_ids[len(q):] + [args.eos_token_id]
        assert len(input_ids) == len(labels)
        return (input_ids, labels)
    return collect_tokens(process_one, filename, data_format='jsonl')

def process_deepctrl(filename, tokenizer):
    """deepctrl-sft-data"""

    def process_one(line):
        if not line:
            return (None, None)
        try:
            per = json.loads(line)
        except:
            return (None, None)
        input_ids, labels = ([], [])
        for human, robot in per['history']:
            q = tokenizer.encode(HUMAN + human + ROBOT, add_special_tokens=False)
            a = tokenizer.encode(robot, add_special_tokens=False)
            if len(input_ids + q + a) >= args.MAX_LENGTH:
                return (None, None)
            input_ids.extend(q + a)
            labels.extend([args.pad_token_id] * (len(q) - 1) + a + [args.eos_token_id])
        q = tokenizer.encode(HUMAN + per['instruction'] + per['input'] + ROBOT, add_special_tokens=False)
        a = tokenizer.encode(per['output'], add_special_tokens=False)
        input_ids.extend(q + a)
        labels.extend([args.pad_token_id] * (len(q) - 1) + a + [args.eos_token_id])
        if len(input_ids) >= args.MAX_LENGTH:
            return (None, None)
        assert len(input_ids) == len(labels)
        return (input_ids, labels)
    return collect_tokens(process_one, filename, data_format='jsonl')

def process_moss002(filename, tokenizer):
    """fnlp@moss-002-sft-data"""

    def process_one(per):
        history = re.split('<eoh> \\[MOSS\\]: |<eoa> \\[Human\\]: |\\[Human\\]: |<eoa>', per['plain_text'])
        history = [i.strip() for i in history if i]
        input_ids, labels = ([], [])
        for human, robot in zip(history[0::2], history[1::2]):
            human = tokenizer.encode(HUMAN + human + ROBOT, add_special_tokens=False)
            robot = tokenizer.encode(robot, add_special_tokens=False)
            if len(input_ids + human + robot) >= args.MAX_LENGTH:
                break
            input_ids.extend(human + robot)
            labels.extend([args.pad_token_id] * (len(human) - 1) + robot + [args.eos_token_id])
        if len(input_ids) >= args.MAX_LENGTH:
            return (None, None)
        assert len(input_ids) == len(labels)
        return (input_ids, labels)
    return collect_tokens(process_one, filename, data_format='json')

def process_moss003(filename, tokenizer):
    """fnlp@moss-003-sft-data"""

    def process_one(line):
        if not line:
            return (None, None)
        per = json.loads(line)
        input_ids, labels = ([], [])
        for turn in per['chat'].values():
            if not re.search('[\\u4e00-\\u9fff]', turn['MOSS']):
                continue
            human = turn['Human'].replace('<|Human|>: ', '').replace('<eoh>\n', '')
            robot = turn['MOSS'].replace('<|MOSS|>: ', '').replace('<eom>\n', '')
            robot = re.sub('<sup><\\|[0-9]+\\|></sup>', '', robot).strip()
            human = tokenizer.encode(HUMAN + human + ROBOT, add_special_tokens=False)
            robot = tokenizer.encode(robot, add_special_tokens=False)
            if len(input_ids + human + robot) >= args.MAX_LENGTH:
                break
            input_ids.extend(human + robot)
            labels.extend([args.pad_token_id] * (len(human) - 1) + robot + [args.eos_token_id])
        if len(input_ids) >= args.MAX_LENGTH:
            return (None, None)
        assert len(input_ids) == len(labels)
        return (input_ids, labels)
    return collect_tokens(process_one, filename, data_format='jsonl')

def process_shareai(filename, tokenizer):
    """shareAI"""

    def process_one(line):
        if not line:
            return (None, None)
        per = json.loads(line)
        input_ids, labels = ([], [])
        for turn in per['conversation']:
            human = turn['human']
            robot = turn['assistant']
            human = tokenizer.encode(HUMAN + human + ROBOT, add_special_tokens=False)
            robot = tokenizer.encode(robot, add_special_tokens=False)
            if len(input_ids + human + robot) >= args.MAX_LENGTH:
                break
            input_ids.extend(human + robot)
            labels.extend([args.pad_token_id] * (len(human) - 1) + robot + [args.eos_token_id])
        if len(input_ids) >= args.MAX_LENGTH:
            return (None, None)
        assert len(input_ids) == len(labels)
        return (input_ids, labels)
    return collect_tokens(process_one, filename, data_format='jsonl')

def process_firefly(filename, tokenizer):
    """YeungNLP@firefly-train-1.1M"""

    def process_one(line):
        if not line:
            return (None, None)
        per = json.loads(line)
        q = tokenizer.encode(HUMAN + per['input'] + ROBOT, add_special_tokens=False)
        a = tokenizer.encode(per['target'], add_special_tokens=False)
        if len(q) + len(a) >= args.MAX_LENGTH:
            return (None, None)
        input_ids = q + a
        labels = [args.pad_token_id] * (len(q) - 1) + input_ids[len(q):] + [args.eos_token_id]
        assert len(input_ids) == len(labels)
        return (input_ids, labels)
    return collect_tokens(process_one, filename, data_format='jsonl')

def process_DPO_En_Zh_20k(filename, tokenizer):
    """hiyouga/DPO-En-Zh-20k"""

    def process_one(per):
        prompt_ids = tokenizer.encode(HUMAN + per['system'] + per['prompt'] + ROBOT, add_special_tokens=False)
        chosen_ids = tokenizer.encode(per['answer'][0], add_special_tokens=False)
        rejected_ids = tokenizer.encode(per['answer'][1], add_special_tokens=False)
        if len(prompt_ids) + len(chosen_ids) >= args.MAX_LENGTH or len(prompt_ids) + len(rejected_ids) >= args.MAX_LENGTH:
            return (None, None)
        return (prompt_ids, chosen_ids, rejected_ids)
    return collect_tokens(process_one, filename, data_format='json')

def hh_rlhf_cn(filename, tokenizer):
    """dikw/hh_rlhf_cn"""

    def process_one(line):
        per = json.loads(line)
        prompt_ids = []
        for context in per['context']:
            if context['role'] == 'human':
                q = tokenizer.encode(HUMAN + context['text'] + ROBOT, add_special_tokens=False)
                prompt_ids.extend(q)
            elif context['role'] == 'assistant':
                a = tokenizer.encode(context['text'], add_special_tokens=False)
                prompt_ids.extend(a)
            if len(prompt_ids) >= args.MAX_LENGTH:
                break
        chosen_ids = tokenizer.encode(per['chosen']['text'], add_special_tokens=False)
        rejected_ids = tokenizer.encode(per['rejected']['text'], add_special_tokens=False)
        if len(prompt_ids) + len(chosen_ids) >= args.MAX_LENGTH or len(prompt_ids) + len(rejected_ids) >= args.MAX_LENGTH:
            return (None, None)
        return (prompt_ids, chosen_ids, rejected_ids)
    return collect_tokens(process_one, filename)

def CValues_Comparison(filename, tokenizer):
    """diic/CValues-Comparison"""

    def process_one(per):
        prompt_ids = tokenizer.encode(HUMAN + per['prompt'] + ROBOT, add_special_tokens=False)
        chosen_ids = tokenizer.encode(per['pos_resp'], add_special_tokens=False)
        rejected_ids = tokenizer.encode(per['neg_resp'], add_special_tokens=False)
        if len(prompt_ids) + len(chosen_ids) >= args.MAX_LENGTH or len(prompt_ids) + len(rejected_ids) >= args.MAX_LENGTH:
            return (None, None)
        return (prompt_ids, chosen_ids, rejected_ids)
    return collect_tokens(process_one, filename, data_format='jsonl')

def zhihu_rlhf_3k(filename, tokenizer):
    """liyucheng/zhihu_rlhf_3k"""

    def process_one(per):
        prompt_ids = tokenizer.encode(HUMAN + per['prompt'] + ROBOT, add_special_tokens=False)
        chosen_ids = tokenizer.encode(per['chosen'], add_special_tokens=False)
        rejected_ids = tokenizer.encode(per['rejected'], add_special_tokens=False)
        if len(prompt_ids) + len(chosen_ids) >= args.MAX_LENGTH or len(prompt_ids) + len(rejected_ids) >= args.MAX_LENGTH:
            return (None, None)
        return (prompt_ids, chosen_ids, rejected_ids)
    return collect_tokens(process_one, filename, data_format='table')

def rlhf_reward_single_round_trans_chinese(filename, tokenizer):

    def process_one(per):
        prompt_ids = tokenizer.encode(HUMAN + per['prompt'] + ROBOT, add_special_tokens=False)
        chosen_ids = tokenizer.encode(per['chosen'], add_special_tokens=False)
        rejected_ids = tokenizer.encode(per['rejected'], add_special_tokens=False)
        if len(prompt_ids) + len(chosen_ids) >= args.MAX_LENGTH or len(prompt_ids) + len(rejected_ids) >= args.MAX_LENGTH:
            return (None, None)
        return (prompt_ids, chosen_ids, rejected_ids)
    return collect_tokens(process_one, filename, data_format='parquet')

