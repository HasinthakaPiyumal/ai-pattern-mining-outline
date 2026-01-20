# Cluster 2

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

def replace_placeholder(query):
    mapping_ = {'<NAME>': args.name, '<AUTHOR>': args.author, '<DATE>': args.date}
    for key, value in mapping_.items():
        query = query.replace(key, value)
    return query

def process_data_slice(data_slice, fi):
    if not USE_PARALLEL:
        log_info_once('Use single process to process data, maybe slow')
        train_samples = [process_one(line) for line in data_slice]
    else:
        log_info_once('Use multiprocess to accelerate data process')
        train_samples = parallel_apply(func=process_one, iterable=data_slice, workers=WORKERS, max_queue_size=MAX_QUEUE_SIZE, dummy=False, callback=None, unordered=False)
    train_samples = [{'prompt_ids': i[0], 'chosen_ids': i[1], 'rejected_ids': i[2]} for i in train_samples if i[0] is not None and len(i[0]) > 1]
    save_path = os.path.join(args.dataset_save_dir, filename.replace('/', '--').replace('.jsonl', '').replace('.json', '') + f'_{fi}.jsonl')
    with open(save_path, 'w') as f:
        for item in train_samples:
            json.dump(item, f)
            f.write('\n')
    return len(train_samples)

