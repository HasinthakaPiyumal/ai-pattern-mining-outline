# Cluster 67

def load_dataset():
    text_labels = []
    with open(osp.join(osp.dirname(__file__), 'gt_good.txt')) as f:
        for line in f:
            text_labels.append((line, True))
    with open(osp.join(osp.dirname(__file__), 'gt_bad.txt')) as f:
        for line in f:
            text_labels.append((line, False))
    return text_labels

def calculate(chunk_size: int):
    config_path = 'config.ini'
    repo_dir = 'repodir'
    work_dir_base = 'workdir'
    work_dir = work_dir_base + str(chunk_size)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    text_labels = load_dataset()
    cache = CacheRetriever(config_path=config_path)
    fs_init = FeatureStore(embedder=cache.embedder, config_path=config_path, chunk_size=chunk_size)
    file_opr = FileOperation()
    files = file_opr.scan_dir(repo_dir=repo_dir)
    fs_init.preprocess(files=files, work_dir=work_dir)
    fs_init.build_dense(files=files, work_dir=work_dir, markdown_as_txt=True)
    del fs_init
    retriever = CacheRetriever(config_path=config_path).get(fs_id=str(chunk_size), work_dir=work_dir)
    start = 0.41
    stop = 0.5
    step = 0.01
    throttles = [round(start + step * i, 4) for i in range(int((stop - start) / step) + 1)]
    best_chunk_f1 = 0.0
    for throttle in tqdm(throttles):
        retriever.reject_throttle = throttle
        dts = []
        gts = []
        for text_label in text_labels:
            question = text_label[0]
            retriever.reject_throttle = throttle
            _, score = retriever.is_relative(query=question, enable_kg=False, enable_threshold=False)
            if score >= throttle:
                dts.append(True)
            else:
                dts.append(False)
            gts.append(text_label[1])
        f1 = f1_score(gts, dts)
        f1 = round(f1, 4)
        precision = precision_score(gts, dts)
        precision = round(precision, 4)
        recall = recall_score(gts, dts)
        recall = round(recall, 4)
        logger.info((throttle, precision, recall, f1))
        data = {'chunk_size': chunk_size, 'throttle': throttle, 'precision': precision, 'recall': recall, 'f1': f1}
        json_str = json.dumps(data)
        with open(osp.join(osp.dirname(__file__), 'chunk{}.jsonl'.format(chunk_size)), 'a') as f:
            f.write(json_str)
            f.write('\n')
        if f1 > best_chunk_f1:
            best_chunk_f1 = f1
    print(best_chunk_f1)
    return best_chunk_f1

def calculate(chunk_size: int):
    config_path = 'config.ini'
    repo_dir = 'repodir'
    work_dir_base = 'workdir'
    work_dir = work_dir_base + str(chunk_size)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    text_labels = load_dataset()
    fs_init = FeatureStore(embeddings=None, config_path=config_path, chunk_size=chunk_size, analyze_reject=True, rejecter_naive_splitter=True)
    file_opr = FileOperation()
    files = file_opr.scan_dir(repo_dir=repo_dir)
    fs_init.preprocess(files=files, work_dir=work_dir)
    docs = fs_init.build_dense(files=files, work_dir=work_dir)
    del fs_init
    col = init_milvus(col_name='test2', max_length_bytes=3 * chunk_size)
    subdocs = split_by_group(docs)
    for idx, docs in enumerate(subdocs):
        print('build step {}'.format(idx))
        texts = []
        sources = []
        reads = []
        for doc in docs:
            texts.append(doc.page_content[0:chunk_size])
            sources.append(doc.metadata['source'])
            reads.append(doc.metadata['read'])
        max_length = len(max(texts, key=lambda x: len(x)))
        docs_emb = ef(texts)
        entities = [texts, docs_emb['sparse'], docs_emb['dense']]
        try:
            col.insert(entities)
            col.flush()
        except Exception as e:
            print(e)
    print('insert finished')
    start = 0.05
    stop = 0.5
    step = 0.05
    sparse_ratios = []
    sparse_ratios = [round(start + step * i, 4) for i in range(int((stop - start) / step) + 1)]
    best_chunk_f1 = 0.0
    dts = []
    gts = []
    predictions = []
    labels = []
    for sparse_ratio in sparse_ratios:
        for text_label in tqdm(text_labels):
            query_embeddings = ef([text_label[0]])
            k = 1
            sparse_req = AnnSearchRequest(query_embeddings['sparse'], 'sparse_vector', {'metric_type': 'IP'}, limit=k)
            dense_req = AnnSearchRequest(query_embeddings['dense'], 'dense_vector', {'metric_type': 'IP'}, limit=k)
            res = col.hybrid_search([sparse_req, dense_req], rerank=WeightedRanker(sparse_ratio, 1.0 - sparse_ratio), limit=k, output_fields=['text'])
            res = res[0]
            if len(res) > 0:
                predictions.append(max(0.0, min(1.0, res[0].score)))
            else:
                predictions.append(0)
            if text_label[1]:
                labels.append(1)
            else:
                labels.append(0)
        precision, recall, thresholds = precision_recall_curve(labels, predictions)
        f1 = 2 * precision * recall / (precision + recall)
        logger.debug('sparse_ratio {} max f1 {} at {}, threshold {}'.format(sparse_ratio, np.max(f1), np.argmax(f1), thresholds[np.argmax(f1)]))

def init_milvus(col_name: str, max_length_bytes: int):
    conn = connections.connect('default', host='localhost', port='19530')
    fields = [FieldSchema(name='pk', dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100), FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=max_length_bytes), FieldSchema(name='sparse_vector', dtype=DataType.SPARSE_FLOAT_VECTOR), FieldSchema(name='dense_vector', dtype=DataType.FLOAT_VECTOR, dim=dense_dim)]
    schema = CollectionSchema(fields, '')
    col = Collection(col_name, schema, consistency_level='Strong')
    sparse_index = {'index_type': 'SPARSE_INVERTED_INDEX', 'metric_type': 'IP'}
    col.create_index('sparse_vector', sparse_index)
    dense_index = {'index_type': 'FLAT', 'metric_type': 'IP'}
    col.create_index('dense_vector', dense_index)
    col.load()
    return col

def split_by_group(inputs: list, groupsize: int=1024):
    num_sublists = len(inputs) // groupsize
    remaining_items = len(inputs) % groupsize
    sublists = []
    for i in range(num_sublists):
        start_index = i * groupsize
        end_index = start_index + groupsize
        sublists.append(inputs[start_index:end_index])
    if remaining_items > 0:
        sublists.append(inputs[num_sublists * groupsize:])
    gt = len(inputs)
    dt = 0
    for lis in sublists:
        dt += len(lis)
    assert gt == dt
    return sublists

