# Cluster 2

def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description='Reconstruct group chat.')
    parser.add_argument('--output_dir', type=str, default='groups', help='Splitted group messages.')
    parser.add_argument('--input', type=str, default='/home/khj/github/huixiangdou/tests/history_recv_send.txt', help='Raw input messages.')
    parser.add_argument('--action', type=str, default='intention', help='"split"): split raw input into group messages; "intention"): decide which query being a question')
    args = parser.parse_args()
    return args

def build_feature_store(cache: CacheRetriever, payload: types.SimpleNamespace):
    abs_base = payload.file_abs_base
    fs_id = payload.feature_store_id
    path_list = []
    files = []
    file_opr = FileOperation()
    for filename in payload.file_list:
        abs_path = os.path.join(abs_base, filename)
        _type = file_opr.get_type(abs_path)
        files.append(FileName(root=abs_base, filename=filename, _type=_type))
    BASE = feature_store_base_dir()
    workdir = os.path.join(BASE, fs_id, 'workdir')
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    configpath = os.path.join(BASE, fs_id, 'config.ini')
    if not os.path.exists(configpath):
        template_file = 'config.ini'
        if not os.path.exists(template_file):
            raise Exception(f'{template_file} not exist')
        shutil.copy(template_file, configpath)
    with open(os.path.join(BASE, fs_id, 'desc'), 'w', encoding='utf8') as f:
        f.write(payload.name)
    fs = FeatureStore(config_path=configpath, embedder=cache.embedder)
    task_state = partial(callback_task_state, feature_store_id=fs_id, _type=TaskCode.FS_ADD_DOC.value)
    fs.initialize(files=files, work_dir=workdir, ner_file=None)
    files_state = []
    success_cnt = 0
    fail_cnt = 0
    skip_cnt = 0
    for file in files:
        files_state.append({'file': str(file.basename), 'status': bool(file.state), 'desc': str(file.reason)})
        if file.state:
            success_cnt += 1
        elif file.reason == 'skip':
            skip_cnt += 1
        else:
            fail_cnt += 1
    if success_cnt == len(files):
        task_state(code=ErrorCode.SUCCESS.value, state=ErrorCode.SUCCESS.describe(), files_state=files_state)
    elif success_cnt == 0:
        task_state(code=ErrorCode.FAILED.value, state='无文件被处理', files_state=files_state)
    else:
        state = f'完成{success_cnt}个文件，跳过{skip_cnt}个，{fail_cnt}个处理异常。请确认文件格式。'
        task_state(code=ErrorCode.SUCCESS.value, state=state, files_state=files_state)

def ymd():
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d')
    if not os.path.exists(date_string):
        os.makedirs(date_string)
    return date_string

def format_refs(refs: List[str]):
    refs_filter = list(set(refs))
    if len(refs) < 1:
        return ''
    text = ''
    if language == 'zh':
        text += '参考资料：\r\n'
    else:
        text += '**References:**\r\n'
    for file_or_url in refs_filter:
        text += '* {}\r\n'.format(file_or_url)
    text += '\r\n'
    return text

def event_stream():
    for sess in assistant.generate(query=query):
        status = {'state': str(sess.code), 'response': sess.response, 'refs': sess.references}
        pipeline['step'].append(status)
        yield json.dumps(pipeline)

def convert_history_to_tuple(history: List[Talk]):
    history = []
    for item in history:
        history.append({'role': 'user', 'content': item.query})
        history.append({'role': 'assistant', 'content': item.reply})
    return history

def main():
    """Function to start the server without running a separate process."""
    args = parse_args()
    server_ready = Value('i', 0)
    if not args.unittest:
        llm_serve(args.config_path, server_ready)
    else:
        queries = ['今天天气如何？']
        start_llm_server(config_path=args.config_path)
        from .llm_client import ChatClient
        client = ChatClient(config_path=args.config_path)
        for query in queries:
            print(client.generate_response(prompt=query, history=[], backend='local'))

def llm_serve(config_path: str, server_ready: Value):
    """Start the LLM server.

    Args:
        config_path (str): Path to the configuration file.
        server_ready (multiprocessing.Value): Shared variable to indicate when the server is ready.  # noqa E501
    """
    with open(config_path, encoding='utf8') as f:
        llm_config = pytoml.load(f)['llm']
        bind_port = int(llm_config['server']['local_llm_bind_port'])
    try:
        server = HybridLLMServer(llm_config=llm_config)
        server_ready.value = 1
    except Exception as e:
        server_ready.value = -1
        raise e

    async def inference(talk: Talk):
        """Call local llm inference."""
        logger.info(talk)
        prompt = talk.prompt
        history = talk.history
        backend = talk.backend
        parts = []
        try:
            async for text in server.chat_stream(prompt=prompt, history=history, backend=backend):
                parts.append(text)
            return {'text': ''.join(parts), 'error': ''}
        except Exception as e:
            return {'text': '', 'error': str(e)}

    async def stream(talk: Talk):
        """Call local llm inference."""
        logger.info(talk)
        prompt = talk.prompt
        history = talk.history
        backend = talk.backend

        async def generate():
            async for text in server.chat_stream(prompt=prompt, history=history, backend=backend):
                yield text
        return EventSourceResponse(generate())
    app = FastAPI(docs_url='/')
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
    router = APIRouter()
    router.add_api_route('/inference', inference, methods=['POST'])
    router.add_api_route('/stream', stream, methods=['POST'])
    app.include_router(router)
    uvicorn.run(app, host='0.0.0.0', port=bind_port, log_level='info')

def start_llm_server(config_path: str):
    set_start_method('spawn')
    server_ready = Value('i', 0)
    server_process = Process(target=llm_serve, args=(config_path, server_ready))
    server_process.daemon = True
    server_process.start()
    while True:
        if server_ready.value == 0:
            logger.info('waiting for server to be ready..')
            time.sleep(2)
        elif server_ready.value == 1:
            break
        else:
            logger.error('start local LLM server failed, quit.')
            raise Exception('local LLM path')
    logger.info('Hybrid LLM Server start.')

def fetch_web_content(target_link: str):
    """Fetches and parses the content of the target URL.

    Extracts the main content and title from the HTML of the page. Returns the
    title and content as a single string.
    """
    response = requests.get(target_link, timeout=60)
    doc = Document(response.text)
    content_html = doc.summary()
    title = doc.short_title()
    soup = BS(content_html, 'html.parser')
    ret = '{} {}'.format(title, soup.text)
    return ret

def calculate(config_path: str='config.ini'):
    kg = KnowledgeGraph(config_path=config_path, override=False)
    G = kg.load_networkx()
    if not G:
        logger.error('Knowledge graph not build, quit.')
        return
    text_labels = load_dataset()
    outpath = os.path.join(os.path.dirname(__file__), 'out.jsonl')
    for text, label in tqdm(text_labels):
        result = kg.retrieve(G=G, query=text)
        json_str = json.dumps({'query': text, 'result': result, 'gt': label}, ensure_ascii=False)
        with open(outpath, 'a') as f:
            f.write(json_str)
            f.write('\n')

def main():
    args = parse_args()
    if args.retrieve:
        calculate(args.config_path)
    else:
        summarize()

def summarize():
    outpath = os.path.join(os.path.dirname(__file__), 'out.jsonl')
    for throttle in range(0, 40, 5):
        dts = []
        gts = []
        max_ref_cnts = []
        with open(outpath) as f:
            for line in f:
                json_obj = json.loads(line)
                gts.append(json_obj['gt'])
                if not json_obj['result']:
                    dts.append(False)
                elif len(json_obj['result']) <= throttle:
                    dts.append(False)
                else:
                    dts.append(True)
                    max_ref_cnts.append(len(json_obj['result']))
        f1 = f1_score(gts, dts)
        f1 = round(f1, 2)
        precision = precision_score(gts, dts)
        precision = round(precision, 2)
        recall = recall_score(gts, dts)
        recall = round(recall, 2)
        logger.info(('throttle, precision, recall, F1', throttle, precision, recall, f1))

def main():
    args = parse_args()
    best_f1 = 0.0
    best_chunk_size = -1
    calculate(832)

def main():
    args = parse_args()
    best_f1 = 0.0
    best_chunk_size = -1
    calculate(2048)

