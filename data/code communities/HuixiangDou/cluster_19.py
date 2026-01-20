# Cluster 19

def test_ddgs():
    config_path = 'config-2G.ini'
    engine = WebSearch(config_path=config_path)
    articles, error = engine.get(query='mmpose installation')
    assert error is None
    assert len(articles[0]) > 100

def test_serper():
    config_path = 'config-2G.ini'
    _, serper_token = load_secret()
    config = None
    with open(config_path) as f:
        config = pytoml.load(f)
        config['web_search']['engine'] = 'serper'
        config['web_search']['serper_x_api_key'] = serper_token
    config_path = None
    with tempfile.NamedTemporaryFile(delete=False, mode='w+b') as temp_file:
        tomlstr = pytoml.dumps(config)
        temp_file.write(tomlstr.encode('utf8'))
        config_path = temp_file.name
    engine = WebSearch(config_path=config_path)
    articles, error = engine.get(query='mmpose installation')
    assert error is None
    assert len(articles[0]) > 100
    assert articles[0].brief == articles[0].content
    os.remove(temp_file.name)

def test_parse_zhihu():
    config_path = 'config-2G.ini'
    engine = WebSearch(config_path=config_path)
    article = engine.fetch_url(query='', target_link='https://zhuanlan.zhihu.com/p/699164101')
    if article is not None:
        assert check_str_useful(article.content)

def test_parse_hljnews():
    config_path = 'config-2G.ini'
    engine = WebSearch(config_path=config_path)
    article = engine.fetch_url(query='', target_link='http://www.hljnews.cn/ljxw/content/2023-10/17/content_729976.html?vp-fm')
    if article is not None:
        assert check_str_useful(article.content)

