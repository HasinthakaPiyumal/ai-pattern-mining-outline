# Cluster 6

def map_action_to_html(action, **kwargs):
    action_name, action_arg = parse_action(action)
    if action_name == 'start':
        path = os.path.join(TEMPLATE_DIR, 'search_page.html')
        html = render_template_string(read_html_template(path=path), session_id=kwargs['session_id'], instruction_text=kwargs['instruction_text'])
    elif action_name == 'search':
        path = os.path.join(TEMPLATE_DIR, 'results_page.html')
        html = render_template_string(read_html_template(path=path), session_id=kwargs['session_id'], products=kwargs['products'], keywords=kwargs['keywords'], page=kwargs['page'], total=kwargs['total'], instruction_text=kwargs['instruction_text'])
    elif action_name == 'click' and action_arg == END_BUTTON:
        path = os.path.join(TEMPLATE_DIR, 'done_page.html')
        html = render_template_string(read_html_template(path), session_id=kwargs['session_id'], reward=kwargs['reward'], asin=kwargs['asin'], options=kwargs['options'], reward_info=kwargs.get('reward_info'), goal_attrs=kwargs.get('goal_attrs'), purchased_attrs=kwargs.get('purchased_attrs'), goal=kwargs.get('goal'), mturk_code=kwargs.get('mturk_code'), query=kwargs.get('query'), category=kwargs.get('category'), product_category=kwargs.get('product_category'))
    elif action_name == 'click' and action_arg in ACTION_TO_TEMPLATE:
        path = os.path.join(TEMPLATE_DIR, ACTION_TO_TEMPLATE[action_arg])
        html = render_template_string(read_html_template(path), session_id=kwargs['session_id'], product_info=kwargs['product_info'], keywords=kwargs['keywords'], page=kwargs['page'], asin=kwargs['asin'], options=kwargs['options'], instruction_text=kwargs.get('instruction_text'))
    elif action_name == 'click':
        path = os.path.join(TEMPLATE_DIR, 'item_page.html')
        html = render_template_string(read_html_template(path), session_id=kwargs['session_id'], product_info=kwargs['product_info'], keywords=kwargs['keywords'], page=kwargs['page'], asin=kwargs['asin'], options=kwargs['options'], instruction_text=kwargs.get('instruction_text'), show_attrs=kwargs['show_attrs'])
    else:
        raise ValueError('Action name not recognized.')
    return html

@app.route('/search_results/<session_id>/<keywords>/<page>', methods=['GET', 'POST'])
def search_results(session_id, keywords, page):
    instruction_text = user_sessions[session_id]['goal']['instruction_text']
    page = convert_web_app_string_to_var('page', page)
    keywords = convert_web_app_string_to_var('keywords', keywords)
    top_n_products = get_top_n_product_from_keywords(keywords, search_engine, all_products, product_item_dict, attribute_to_asins)
    products = get_product_per_page(top_n_products, page)
    html = map_action_to_html('search', session_id=session_id, products=products, keywords=keywords, page=page, total=len(top_n_products), instruction_text=instruction_text)
    logger = logging.getLogger(session_id)
    logger.info(json.dumps(dict(page='search_results', url=request.url, goal=user_sessions[session_id]['goal'], content=dict(keywords=keywords, search_result_asins=[p['asin'] for p in products], page=page))))
    return html

def convert_web_app_string_to_var(name, string):
    if name == 'keywords':
        keywords = string
        if keywords.startswith('['):
            keywords = literal_eval(keywords)
        else:
            keywords = [keywords]
        var = keywords
    elif name == 'page':
        page = string
        page = int(page)
        var = page
    else:
        raise ValueError('Name of variable not recognized.')
    return var

def get_top_n_product_from_keywords(keywords, search_engine, all_products, product_item_dict, attribute_to_asins=None):
    if keywords[0] == '<r>':
        top_n_products = random.sample(all_products, k=SEARCH_RETURN_N)
    elif keywords[0] == '<a>':
        attribute = ' '.join(keywords[1:]).strip()
        asins = attribute_to_asins[attribute]
        top_n_products = [p for p in all_products if p['asin'] in asins]
    elif keywords[0] == '<c>':
        category = keywords[1].strip()
        top_n_products = [p for p in all_products if p['category'] == category]
    elif keywords[0] == '<q>':
        query = ' '.join(keywords[1:]).strip()
        top_n_products = [p for p in all_products if p['query'] == query]
    else:
        keywords = ' '.join(keywords)
        hits = search_engine.search(keywords, k=SEARCH_RETURN_N)
        docs = [search_engine.doc(hit.docid) for hit in hits]
        top_n_asins = [json.loads(doc.raw())['id'] for doc in docs]
        top_n_products = [product_item_dict[asin] for asin in top_n_asins if asin in product_item_dict]
    return top_n_products

def get_product_per_page(top_n_products, page):
    return top_n_products[(page - 1) * PRODUCT_WINDOW:page * PRODUCT_WINDOW]

@app.route('/item_page/<session_id>/<asin>/<keywords>/<page>/<options>', methods=['GET', 'POST'])
def item_page(session_id, asin, keywords, page, options):
    options = literal_eval(options)
    product_info = product_item_dict[asin]
    goal_instruction = user_sessions[session_id]['goal']['instruction_text']
    product_info['goal_instruction'] = goal_instruction
    html = map_action_to_html('click', session_id=session_id, product_info=product_info, keywords=keywords, page=page, asin=asin, options=options, instruction_text=goal_instruction, show_attrs=SHOW_ATTRS_TAB)
    logger = logging.getLogger(session_id)
    logger.info(json.dumps(dict(page='item_page', url=request.url, goal=user_sessions[session_id]['goal'], content=dict(keywords=keywords, page=page, asin=asin, options=options))))
    return html

@app.route('/item_sub_page/<session_id>/<asin>/<keywords>/<page>/<sub_page>/<options>', methods=['GET', 'POST'])
def item_sub_page(session_id, asin, keywords, page, sub_page, options):
    options = literal_eval(options)
    product_info = product_item_dict[asin]
    goal_instruction = user_sessions[session_id]['goal']['instruction_text']
    product_info['goal_instruction'] = goal_instruction
    html = map_action_to_html(f'click[{sub_page}]', session_id=session_id, product_info=product_info, keywords=keywords, page=page, asin=asin, options=options, instruction_text=goal_instruction)
    logger = logging.getLogger(session_id)
    logger.info(json.dumps(dict(page='item_sub_page', url=request.url, goal=user_sessions[session_id]['goal'], content=dict(keywords=keywords, page=page, asin=asin, options=options))))
    return html

class SimServer:
    """Lightweight simulator of WebShop Flask application for generating HTML observations"""

    def __init__(self, base_url, file_path, filter_goals=None, limit_goals=-1, num_products=None, human_goals=0, show_attrs=False):
        """
        Constructor for simulated server serving WebShop application
        
        Arguments:
        filter_goals (`func`) -- Select specific goal(s) for consideration based on criteria of custom function
        limit_goals (`int`) -- Limit to number of goals available
        num_products (`int`) -- Number of products to search across
        human_goals (`bool`) -- If true, load human goals; otherwise, load synthetic goals
        """
        self.base_url = base_url
        self.all_products, self.product_item_dict, self.product_prices, _ = load_products(filepath=file_path, num_products=num_products, human_goals=human_goals)
        self.search_engine = init_search_engine(num_products=num_products)
        self.goals = get_goals(self.all_products, self.product_prices, human_goals)
        self.show_attrs = show_attrs
        random.seed(233)
        random.shuffle(self.goals)
        if filter_goals is not None:
            self.goals = [goal for i, goal in enumerate(self.goals) if filter_goals(i, goal)]
        if limit_goals != -1 and limit_goals < len(self.goals):
            self.weights = [goal['weight'] for goal in self.goals]
            self.cum_weights = [0] + np.cumsum(self.weights).tolist()
            idxs = []
            while len(idxs) < limit_goals:
                idx = random_idx(self.cum_weights)
                if idx not in idxs:
                    idxs.append(idx)
            self.goals = [self.goals[i] for i in idxs]
        print(f'Loaded {len(self.goals)} goals.')
        self.weights = [goal['weight'] for goal in self.goals]
        self.cum_weights = [0] + np.cumsum(self.weights).tolist()
        self.user_sessions = dict()
        self.search_time = 0
        self.render_time = 0
        self.sample_time = 0
        self.assigned_instruction_text = None

    @app.route('/', methods=['GET', 'POST'])
    def index(self, session_id, **kwargs):
        """Redirect to the search page with the given session ID"""
        html = map_action_to_html('start', session_id=session_id, instruction_text=kwargs['instruction_text'])
        url = f'{self.base_url}/{session_id}'
        return (html, url)

    @app.route('/', methods=['GET', 'POST'])
    def search_results(self, session_id, **kwargs):
        """Initialize session and return the search results page"""
        session = self.user_sessions[session_id]
        keywords = kwargs['keywords']
        assert isinstance(keywords, list)
        page = 1 if 'page' not in kwargs else kwargs['page']
        session['page'] = page
        session['keywords'] = keywords
        session['actions']['search'] += 1
        session['asin'] = None
        session['options'] = {}
        old_time = time.time()
        top_n_products = get_top_n_product_from_keywords(keywords, self.search_engine, self.all_products, self.product_item_dict)
        self.search_time += time.time() - old_time
        products = get_product_per_page(top_n_products, page)
        keywords_url_string = '+'.join(keywords)
        url = f'{self.base_url}/search_results/{session_id}/{keywords_url_string}/{page}'
        old_time = time.time()
        html = map_action_to_html('search', session_id=session_id, products=products, keywords=session['keywords'], page=page, total=len(top_n_products), instruction_text=session['goal']['instruction_text'])
        self.render_time += time.time() - old_time
        return (html, url)

    @app.route('/', methods=['GET', 'POST'])
    def item_page(self, session_id, **kwargs):
        """Render and return the HTML for a product item page"""
        session = self.user_sessions[session_id]
        clickable_name = kwargs['clickable_name']
        text_to_clickable = kwargs['text_to_clickable']
        clickable = text_to_clickable[clickable_name]
        if clickable.get('class') is not None and clickable.get('class')[0] == 'product-link':
            session['asin'] = clickable_name.upper()
            session['actions']['asin'] += 1
            session['asins'].add(session['asin'])
        elif clickable.get('name') is not None:
            clickable_key = clickable['name'].lower()
            session['options'][clickable_key] = clickable_name
            session['actions']['options'] += 1
        product_info = self.product_item_dict[session['asin']]
        keywords_url_string = '+'.join(session['keywords'])
        option_string = json.dumps(session['options'])
        url = f'{self.base_url}/item_page/{session_id}/{session['asin']}/{keywords_url_string}/{session['page']}/{option_string}'
        html = map_action_to_html('click', session_id=session_id, product_info=product_info, keywords=session['keywords'], page=session['page'], asin=session['asin'], options=session['options'], instruction_text=session['goal']['instruction_text'], show_attrs=self.show_attrs)
        return (html, url)

    @app.route('/', methods=['GET', 'POST'])
    def item_sub_page(self, session_id, **kwargs):
        """Render and return the HTML for a product's sub page (i.e. description, features)"""
        session = self.user_sessions[session_id]
        clickable_name = kwargs['clickable_name']
        for k in ACTION_TO_TEMPLATE:
            if clickable_name.lower() == k.lower():
                clickable_name = k
                break
        product_info = self.product_item_dict[session['asin']]
        session['actions'][clickable_name] += 1
        keywords_url_string = '+'.join(session['keywords'])
        url = f'{self.base_url}/item_sub_page/{session_id}/{session['asin']}/{keywords_url_string}/{session['page']}/{clickable_name}/{session['options']}'
        html = map_action_to_html(f'click[{clickable_name}]', session_id=session_id, product_info=product_info, keywords=session['keywords'], page=session['page'], asin=session['asin'], options=session['options'], instruction_text=session['goal']['instruction_text'])
        return (html, url)

    @app.route('/', methods=['GET', 'POST'])
    def done(self, session_id, **kwargs):
        """Render and return HTML for done page"""
        session = self.user_sessions[session_id]
        goal = self.user_sessions[session_id]['goal']
        purchased_product = self.product_item_dict[session['asin']]
        session['actions']['purchase'] += 1
        price = self.product_prices.get(session['asin'])
        reward, info = get_reward(purchased_product, goal, price=price, options=session['options'], verbose=True)
        self.user_sessions[session_id]['verbose_info'] = info
        self.user_sessions[session_id]['done'] = True
        self.user_sessions[session_id]['reward'] = reward
        url = f'{self.base_url}/done/{session_id}/{session['asin']}/{session['options']}'
        html = map_action_to_html(f'click[{END_BUTTON}]', session_id=session_id, reward=reward, asin=session['asin'], options=session['options'], instruction_text=session['goal']['instruction_text'])
        return (html, url, reward)

    def receive(self, session_id, current_url, session_int=None, **kwargs):
        """Map action to the corresponding page"""
        status = dict(reward=0.0, done=False)
        with app.app_context(), app.test_request_context():
            if session_id not in self.user_sessions:
                idx = session_int if session_int is not None and isinstance(session_int, int) else random_idx(self.cum_weights)
                goal = self.goals[idx]
                instruction_text = goal['instruction_text']
                self.user_sessions[session_id] = {'goal': goal, 'done': False}
            else:
                instruction_text = self.user_sessions[session_id]['goal']['instruction_text']
            if self.assigned_instruction_text is not None:
                instruction_text = self.assigned_instruction_text
                self.user_sessions[session_id]['goal']['instruction_text'] = instruction_text
            session = self.user_sessions[session_id]
            if not kwargs:
                kwargs['instruction_text'] = instruction_text
                html, url = self.index(session_id, **kwargs)
                self.user_sessions[session_id].update({'keywords': None, 'page': None, 'asin': None, 'asins': set(), 'options': dict(), 'actions': defaultdict(int)})
            elif 'keywords' in kwargs:
                html, url = self.search_results(session_id, **kwargs)
            elif 'clickable_name' in kwargs:
                clickable_name = kwargs['clickable_name'].lower()
                if clickable_name == END_BUTTON.lower():
                    html, url, reward = self.done(session_id, **kwargs)
                    status['reward'] = reward
                    status['done'] = True
                elif clickable_name == BACK_TO_SEARCH.lower():
                    html, url, status = self.receive(session_id, current_url)
                elif clickable_name == NEXT_PAGE.lower() and self.get_page_name(current_url) == 'search_results':
                    html, url, status = self.receive(session_id, current_url, keywords=session['keywords'], page=session['page'] + 1)
                elif clickable_name == PREV_PAGE.lower() and self.get_page_name(current_url) == 'search_results':
                    html, url, status = self.receive(session_id, current_url, keywords=session['keywords'], page=session['page'] - 1)
                elif clickable_name == PREV_PAGE.lower() and self.get_page_name(current_url) == 'item_sub_page':
                    html, url = self.item_page(session_id, **kwargs)
                elif clickable_name == PREV_PAGE.lower() and self.get_page_name(current_url) == 'item_page':
                    html, url = self.search_results(session_id, keywords=session['keywords'], page=session['page'], **kwargs)
                elif clickable_name in [k.lower() for k in ACTION_TO_TEMPLATE]:
                    html, url = self.item_sub_page(session_id, **kwargs)
                else:
                    html, url = self.item_page(session_id, **kwargs)
            return (html, url, status)

    def get_page_name(self, url):
        """Determine which page (i.e. item_page, search_results) the given URL is pointing at"""
        if url is None:
            return None
        page_names = ['search_results', 'item_page', 'item_sub_page', 'done']
        for page_name in page_names:
            if page_name in url:
                return page_name
        return ''

def dict_to_fake_html(data, page_type, asin=None, sub_page_type=None, options=None, prod_map={}, query=''):
    global QUERY, product_map
    QUERY = query
    product_map = prod_map
    with app.app_context(), app.test_request_context():
        if page_type == Page.RESULTS:
            return search_results(data)
        if page_type == Page.ITEM_PAGE:
            return item_page(SESSION_ID, asin, KEYWORDS, 1, options)
        if page_type == Page.SUB_PAGE:
            if sub_page_type is not None:
                return item_sub_page(SESSION_ID, asin, KEYWORDS, 1, sub_page_type, options)
            else:
                raise Exception('Sub page of type', sub_page_type, 'unrecognized')

