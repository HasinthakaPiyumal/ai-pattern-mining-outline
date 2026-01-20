# Cluster 65

def process(param: tuple):
    fsid, filedir = param
    queries = load_queries(fsid=fsid)
    if len(queries) < 1:
        return
    r = Record(fsid=fsid)
    if r.is_processed():
        logger.info('skip {}'.format(fsid))
        return
    config_path = 'config.ini'
    cache = CacheRetriever(config_path=config_path)
    fs_init = FeatureStore(embedder=cache.embedder, config_path=config_path)
    file_opr = FileOperation()
    files = file_opr.scan_dir(repo_dir=filedir)
    work_dir = os.path.join('workdir', fsid)
    fs_init.initialize(files=files, work_dir=work_dir)
    file_opr.summarize(files)
    del fs_init
    retriever = cache.get(config_path=config_path, work_dir=work_dir)
    if not os.path.exists('candidates'):
        os.makedirs('candidates')
    for query in queries:
        try:
            query = query[0:400]
            docs = retriever.compression_retriever.get_relevant_documents(query)
            candidates = []
            logger.info('{} docs count {}'.format(fsid, len(docs)))
            for doc in docs:
                data = {'content': doc.page_content, 'source': doc.metadata['read'], 'score': doc.metadata['relevance_score']}
                candidates.append(data)
            json_str = json.dumps({'query': query, 'candidates': candidates}, ensure_ascii=False)
            with open(os.path.join('candidates', fsid + '.jsonl'), 'a') as f:
                f.write(json_str)
                f.write('\n')
        except Exception as e:
            pdb.set_trace()
            print(e)
    r.mark_as_processed()

def load_queries(fsid: str):
    pwd = os.path.dirname(__file__)
    base = os.path.join(pwd, '..', 'queries')
    query_path = os.path.join(base, fsid + '.txt')
    if not os.path.exists(query_path):
        return []
    queries = []
    print(query_path)
    if fsid == '0000':
        with open(query_path) as f:
            for line in f:
                queries.append(line)
        return queries
    with open(query_path) as f:
        for line in f:
            queries = json.loads(line)
            break
    return queries

