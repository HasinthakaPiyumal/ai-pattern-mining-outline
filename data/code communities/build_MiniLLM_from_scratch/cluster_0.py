# Cluster 0

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

def collect_tokens(process_one, filename, data_format: Literal['jsonl', 'json', 'table']='jsonl'):
    """各个函数通用的处理token的方式"""

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
    data_path = os.path.join(args.dataset_src_dir, filename)
    os.makedirs(args.dataset_save_dir, exist_ok=True)
    all_count = 0
    data = None
    if data_format == 'parquet':
        df = pq.read_table(data_path).to_pandas()
        data = [df.loc[i].to_dict() for i in df.index]
        data_format = 'json'
    elif data_format == 'table':
        df = pd.read_table(data_path)
        data = [df.loc[i].to_dict() for i in df.index]
        data_format = 'json'
    if data_format == 'json':
        if data is None:
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
    elif data_format == 'jsonl':
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

