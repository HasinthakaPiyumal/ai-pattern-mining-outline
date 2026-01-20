# Cluster 27

def build_config_path():
    config_path = 'config-2G.ini'
    kimi_token, _ = load_secret()
    config = None
    with open(config_path) as f:
        config = pytoml.load(f)
        config['feature_store']['embedding_model_path'] = '/data2/khj/bce-embedding-base_v1/'
        config['feature_store']['reranker_model_path'] = '/data2/khj/bce-embedding-base_v1/'
        config['llm']['server']['remote_api_key'] = kimi_token
    config_path = None
    with tempfile.NamedTemporaryFile(delete=False, mode='w+b') as temp_file:
        tomlstr = pytoml.dumps(config)
        temp_file.write(tomlstr.encode('utf8'))
        config_path = temp_file.name
    return config_path

def run():
    config_path = build_config_path()
    actions = {'feature_store': 'python3 -m huixiangdou.services.store --config_path {}', 'main': 'python3 -m huixiangdou.main --config_path {}'}
    reports = ['HuixiangDou daily smoke:']
    for action, cmd in actions.items():
        cmd = cmd.format(config_path)
        log = command(cmd)
        if 'ConnectionResetError' in log:
            logger.info(f'*{action}, {cmd}')
            assert 0
        else:
            logger.info(f'*{action}, passed')

def command(txt: str):
    """Executes a shell command and returns its output.

    Args:
        txt (str): Command to be executed in the shell.

    Returns:
        str: Output of the shell command execution.
    """
    logger.debug('cmd: {}'.format(txt))
    cmd = os.popen(txt)
    return cmd.read().rstrip().lstrip()

def test_sg():
    config_path = build_config_path()
    llm = LLM(config_path=config_path)
    proxy = SourceGraphProxy(config_path=config_path)
    content = proxy.search(llm_client=llm, question='mmpose installation', groupname='mmpose dev group')
    assert len(content) > 0

def format_history(history):
    """format [{sender, content}] to [[user1, bot1],[user2,bot2]..] style."""
    ret = []
    last_id = -1
    user = ''
    concat_text = ''
    for item in history:
        if last_id == -1:
            last_id = item.sender
            concat_text = item.content
            continue
        if last_id == item.sender:
            concat_text += '\n'
            concat_text += item.content
            continue
        if last_id == 0:
            user = concat_text
        elif last_id == 1:
            ret.append([user, concat_text])
            user = ''
        last_id = item.sender
        concat_text = item.content
    if last_id == 0:
        ret.append([concat_text, ''])
        logger.warning('chat history should not ends with user')
    elif last_id == 1:
        ret.append([user, concat_text])
    return ret

def process():
    que = Queue(name='Task')
    fs_cache = CacheRetriever('config.ini')
    logger.info('start wait task queue..')
    while True:
        msg_pop = que.get(timeout=16)
        if msg_pop is None:
            continue
        msg, error = parse_json_str(msg_pop)
        logger.info(msg)
        if error is not None:
            raise error
        logger.debug(f'process {msg}')
        if msg.type == TaskCode.FS_ADD_DOC.value:
            fs_cache.pop(msg.payload.feature_store_id)
            build_feature_store(fs_cache, msg.payload)
        elif msg.type == TaskCode.FS_UPDATE_SAMPLE.value:
            fs_cache.pop(msg.payload.feature_store_id)
            update_sample(fs_cache, msg.payload)
        elif msg.type == TaskCode.FS_UPDATE_PIPELINE.value:
            update_pipeline(msg.payload)
        elif msg.type == TaskCode.CHAT.value:
            loop = always_get_an_event_loop()
            loop.run_until_complete(chat_with_featue_store(fs_cache, msg.payload))
        else:
            logger.warning(f'unknown type {msg}')

def update_sample(cache: CacheRetriever, payload: types.SimpleNamespace):
    positive = payload.positive
    negative = payload.negative
    fs_id = payload.feature_store_id
    task_state = partial(callback_task_state, feature_store_id=fs_id, _type=TaskCode.FS_UPDATE_SAMPLE.value)
    if len(positive) < 1 or len(negative) < 1:
        task_state(code=ErrorCode.BAD_PARAMETER.value, state='正例为空。请根据真实用户问题，填写正例；同时填写几句场景无关闲聊作负例')
        return
    for idx in range(len(positive)):
        if len(positive[idx]) < 1:
            positive[idx] += '.'
    for idx in range(len(negative)):
        if len(negative[idx]) < 1:
            negative[idx] += '.'
    BASE = feature_store_base_dir()
    fs_id = payload.feature_store_id
    workdir = os.path.join(BASE, fs_id, 'workdir')
    configpath = os.path.join(BASE, fs_id, 'config.ini')
    db_dense = os.path.join(workdir, 'db_dense')
    if not os.path.exists(workdir) or not os.path.exists(configpath) or (not os.path.exists(db_dense)):
        task_state(code=ErrorCode.INTERNAL_ERROR.value, state='知识库未建立或中途异常，已自动反馈研发。请重新建立知识库。')
        return
    retriever = cache.get(fs_id=fs_id, config_path=configpath, work_dir=workdir)
    retriever.update_throttle(config_path=configpath, good_questions=positive, bad_questions=negative)
    del retriever
    task_state(code=ErrorCode.SUCCESS.value, state=ErrorCode.SUCCESS.describe())

def update_pipeline(payload: types.SimpleNamespace):
    fs_id = payload.feature_store_id
    token = payload.web_search_token
    task_state = partial(callback_task_state, feature_store_id=fs_id, _type=TaskCode.FS_UPDATE_PIPELINE.value)
    BASE = feature_store_base_dir()
    fs_id = payload.feature_store_id
    workdir = os.path.join(BASE, fs_id, 'workdir')
    configpath = os.path.join(BASE, fs_id, 'config.ini')
    if not os.path.exists(workdir) or not os.path.exists(configpath):
        task_state(code=ErrorCode.INTERNAL_ERROR.value, state='知识库未建立或中途异常，已自动反馈研发。请重新建立知识库。')
        return
    with open(configpath, encoding='utf8') as f:
        config = pytoml.load(f)
    config['web_search']['x_api_key'] = token
    with open(configpath, 'w', encoding='utf8') as f:
        pytoml.dump(config, f)
    task_state(code=ErrorCode.SUCCESS.value, state=ErrorCode.SUCCESS.describe())

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.info('Creating a new event loop in a sub-thread.')
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def main():
    pwd = os.path.dirname(__file__)
    base = os.path.join(pwd, '..', 'feature_stores')
    dirs = os.listdir(base)
    params = []
    import pdb
    pdb.set_trace()
    for fsid in dirs:
        filedir = os.path.join(base, fsid, 'workdir/preprocess')
        process((fsid, filedir))
        params.append((fsid, filedir))

