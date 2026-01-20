# Cluster 18

def load_secret():
    kimi_token = ''
    serper_token = ''
    with open('unittest/token.json') as f:
        json_obj = json.load(f)
        kimi_token = json_obj['kimi']
        serper_token = json_obj['serper']
    return (kimi_token, serper_token)

def test_llm_backend_fail():
    remote_only_config = 'config-2G.ini'
    llm_config = None
    with open(remote_only_config, encoding='utf8') as f:
        llm_config = pytoml.load(f)['llm']
    server = HybridLLMServer(llm_config=llm_config)
    _, error = server.chat(prompt='hello', history=[], backend='kimi')
    logger.error(error)
    assert error is not None
    _, error = server.chat(prompt='hello', history=[], backend='deepseek')
    logger.error(error)
    assert error is not None
    _, error = server.chat(prompt='hello', history=[], backend='zhipuai')
    logger.error(error)
    assert error is not None
    _, error = server.chat(prompt='hello', history=[], backend='xi-api')
    logger.error(error)
    assert error is not None

def test_kimi_pass():
    remote_only_config = 'config-2G.ini'
    llm_config = None
    with open(remote_only_config, encoding='utf8') as f:
        llm_config = pytoml.load(f)['llm']
    secrets = load_secret()
    llm_config['server']['remote_type'] = 'kimi'
    llm_config['server']['remote_api_key'] = secrets['kimi']
    print('testing {}'.format(llm_config['server']))
    server = HybridLLMServer(llm_config=llm_config)
    response, error = server.chat(prompt=PROMPT, history=[], backend='kimi')
    score = get_score(relation=response, default=0)
    assert score >= 5
    assert error is None
    assert len(response) > 0

def get_score(relation: str, default=0):
    score = default
    filtered_relation = ''.join([c for c in relation if c.isdigit()])
    try:
        score_str = re.sub('[^\\d]', ' ', filtered_relation).strip()
        score = int(score_str.split(' ')[0])
    except Exception as e:
        logger.warning('primitive is_truth: {}, use default value {}'.format(str(e), default))
    return score

def test_zhipu_pass():
    remote_only_config = 'config-2G.ini'
    llm_config = None
    with open(remote_only_config, encoding='utf8') as f:
        llm_config = pytoml.load(f)['llm']
    secrets = load_secret()
    llm_config['server']['remote_type'] = 'zhipuai'
    llm_config['server']['remote_api_key'] = secrets['zhipuai']
    print('testing {}'.format(llm_config['server']))
    server = HybridLLMServer(llm_config=llm_config)
    response, error = server.chat(prompt=PROMPT, history=[], backend='zhipuai')
    score = get_score(relation=response, default=0)
    assert score >= 5
    assert error is None
    assert len(response) > 0

def test_deepseek_pass():
    remote_only_config = 'config-2G.ini'
    llm_config = None
    with open(remote_only_config, encoding='utf8') as f:
        llm_config = pytoml.load(f)['llm']
    secrets = load_secret()
    llm_config['server']['remote_type'] = 'deepseek'
    llm_config['server']['remote_api_key'] = secrets['deepseek']
    print('testing {}'.format(llm_config['server']))
    server = HybridLLMServer(llm_config=llm_config)
    response, error = server.chat(prompt=PROMPT, history=[], backend='deepseek')
    score = get_score(relation=response, default=0)
    assert score >= 5
    assert error is None
    assert len(response) > 0

def test_step_pass():
    remote_only_config = 'config-2G.ini'
    llm_config = None
    with open(remote_only_config, encoding='utf8') as f:
        llm_config = pytoml.load(f)['llm']
    secrets = load_secret()
    llm_config['server']['remote_type'] = 'step'
    llm_config['server']['remote_api_key'] = secrets['step']
    print('testing {}'.format(llm_config['server']))
    server = HybridLLMServer(llm_config=llm_config)
    response, error = server.chat(prompt=PROMPT, history=[], backend='step')
    score = get_score(relation=response, default=0)
    assert score >= 5
    assert error is None
    assert len(response) > 0

def test_puyu_pass():
    remote_only_config = 'config-2G.ini'
    llm_config = None
    with open(remote_only_config, encoding='utf8') as f:
        llm_config = pytoml.load(f)['llm']
    secrets = load_secret()
    llm_config['server']['remote_type'] = 'puyu'
    llm_config['server']['remote_api_key'] = secrets['puyu']
    print('testing {}'.format(llm_config['server']))
    server = HybridLLMServer(llm_config=llm_config)
    response, error = server.chat(prompt=PROMPT, history=[], backend='puyu')
    score = get_score(relation=response, default=0)
    assert score >= 5
    assert error is None

def test_siliconcloud_pass():
    remote_only_config = 'config-2G.ini'
    llm_config = None
    with open(remote_only_config, encoding='utf8') as f:
        llm_config = pytoml.load(f)['llm']
    secrets = load_secret()
    llm_config['server']['remote_type'] = 'siliconcloud'
    llm_config['server']['remote_api_key'] = secrets['siliconcloud']
    print('testing {}'.format(llm_config['server']))
    server = HybridLLMServer(llm_config=llm_config)
    response, error = server.chat(prompt=PROMPT, history=[], backend='siliconcloud')
    logger.info('siliconcloud response {}'.format(response))
    score = get_score(relation=response, default=0)
    assert score >= 5
    assert error is None
    assert len(response) > 0

def test_internlm_local():
    wrapper = InferenceWrapper(llm_local_path)
    repeat = 1
    for i in range(repeat):
        resp = wrapper.chat(prompt=PROMPT)
        logger.info(resp)
        logger.info(get_score(relation=resp))
    del wrapper

def test_internlm_local_():
    with open('config.ini', encoding='utf8') as f:
        llm_config = pytoml.load(f)['llm']
    llm_config['server']['local_llm_path'] = llm_local_path
    server = HybridLLMServer(llm_config)
    resp, error = server.chat(prompt=PROMPT)
    print(resp)
    del server

def build_config_path():
    config_path = 'config-2G.ini'
    kimi_token, serper_token, sg_token = load_secret()
    config = None
    with open(config_path) as f:
        config = pytoml.load(f)
        config['web_search']['engine'] = 'serper'
        config['web_search']['serper_x_api_key'] = serper_token
        config['feature_store']['embedding_model_path'] = '/data2/khj/bce-embedding-base_v1/'
        config['feature_store']['reranker_model_path'] = '/data2/khj/bce-embedding-base_v1/'
        config['llm']['server']['remote_api_key'] = kimi_token
        config['worker']['enable_sg_search'] = 1
        config['sg_search']['src_access_token'] = sg_token
    config_path = None
    with tempfile.NamedTemporaryFile(delete=False, mode='w+b') as temp_file:
        tomlstr = pytoml.dumps(config)
        temp_file.write(tomlstr.encode('utf8'))
        config_path = temp_file.name
    return config_path

