# Cluster 2

@app.route('/<session_id>', methods=['GET', 'POST'])
def index(session_id):
    global user_log_dir
    global all_products, product_item_dict, product_prices, attribute_to_asins, search_engine, goals, weights, user_sessions
    if search_engine is None:
        all_products, product_item_dict, product_prices, attribute_to_asins = load_products(filepath=DEFAULT_FILE_PATH, num_products=DEBUG_PROD_SIZE)
        search_engine = init_search_engine(num_products=DEBUG_PROD_SIZE)
        goals = get_goals(all_products, product_prices)
        random.seed(233)
        random.shuffle(goals)
        weights = [goal['weight'] for goal in goals]
    if session_id not in user_sessions and 'fixed' in session_id:
        goal_dix = int(session_id.split('_')[-1])
        goal = goals[goal_dix]
        instruction_text = goal['instruction_text']
        user_sessions[session_id] = {'goal': goal, 'done': False}
        if user_log_dir is not None:
            setup_logger(session_id, user_log_dir)
    elif session_id not in user_sessions:
        goal = random.choices(goals, weights)[0]
        instruction_text = goal['instruction_text']
        user_sessions[session_id] = {'goal': goal, 'done': False}
        if user_log_dir is not None:
            setup_logger(session_id, user_log_dir)
    else:
        instruction_text = user_sessions[session_id]['goal']['instruction_text']
    if request.method == 'POST' and 'search_query' in request.form:
        keywords = request.form['search_query'].lower().split(' ')
        return redirect(url_for('search_results', session_id=session_id, keywords=keywords, page=1))
    if user_log_dir is not None:
        logger = logging.getLogger(session_id)
        logger.info(json.dumps(dict(page='index', url=request.url, goal=user_sessions[session_id]['goal'])))
    return map_action_to_html('start', session_id=session_id, instruction_text=instruction_text)

def load_products(filepath, num_products=None, human_goals=True):
    with open(filepath) as f:
        products = json.load(f)
    print('Products loaded.')
    products = clean_product_keys(products)
    all_reviews = dict()
    all_ratings = dict()
    if human_goals:
        with open(HUMAN_ATTR_PATH) as f:
            human_attributes = json.load(f)
    with open(DEFAULT_ATTR_PATH) as f:
        attributes = json.load(f)
    with open(HUMAN_ATTR_PATH) as f:
        human_attributes = json.load(f)
    print('Attributes loaded.')
    asins = set()
    all_products = []
    attribute_to_asins = defaultdict(set)
    if num_products is not None:
        products = products[:num_products]
    for i, p in tqdm(enumerate(products), total=len(products)):
        asin = p['asin']
        if asin == 'nan' or len(asin) > 10:
            continue
        if asin in asins:
            continue
        else:
            asins.add(asin)
        products[i]['category'] = p['category']
        products[i]['query'] = p['query']
        products[i]['product_category'] = p['product_category']
        products[i]['Title'] = p['name']
        products[i]['Description'] = p['full_description']
        products[i]['Reviews'] = all_reviews.get(asin, [])
        products[i]['Rating'] = all_ratings.get(asin, 'N.A.')
        for r in products[i]['Reviews']:
            if 'score' not in r:
                r['score'] = r.pop('stars')
            if 'review' not in r:
                r['body'] = ''
            else:
                r['body'] = r.pop('review')
        products[i]['BulletPoints'] = p['small_description'] if isinstance(p['small_description'], list) else [p['small_description']]
        pricing = p.get('pricing')
        if pricing is None or not pricing:
            pricing = [100.0]
            price_tag = '$100.0'
        else:
            pricing = [float(Decimal(re.sub('[^\\d.]', '', price))) for price in pricing.split('$')[1:]]
            if len(pricing) == 1:
                price_tag = f'${pricing[0]}'
            else:
                price_tag = f'${pricing[0]} to ${pricing[1]}'
                pricing = pricing[:2]
        products[i]['pricing'] = pricing
        products[i]['Price'] = price_tag
        options = dict()
        customization_options = p['customization_options']
        option_to_image = dict()
        if customization_options:
            for option_name, option_contents in customization_options.items():
                if option_contents is None:
                    continue
                option_name = option_name.lower()
                option_values = []
                for option_content in option_contents:
                    option_value = option_content['value'].strip().replace('/', ' | ').lower()
                    option_image = option_content.get('image', None)
                    option_values.append(option_value)
                    option_to_image[option_value] = option_image
                options[option_name] = option_values
        products[i]['options'] = options
        products[i]['option_to_image'] = option_to_image
        if asin in attributes and 'attributes' in attributes[asin]:
            products[i]['Attributes'] = attributes[asin]['attributes']
        else:
            products[i]['Attributes'] = ['DUMMY_ATTR']
        if human_goals:
            if asin in human_attributes:
                products[i]['instructions'] = human_attributes[asin]
        else:
            products[i]['instruction_text'] = attributes[asin].get('instruction', None)
            products[i]['instruction_attributes'] = attributes[asin].get('instruction_attributes', None)
        products[i]['MainImage'] = p['images'][0]
        products[i]['query'] = p['query'].lower().strip()
        all_products.append(products[i])
    for p in all_products:
        for a in p['Attributes']:
            attribute_to_asins[a].add(p['asin'])
    product_item_dict = {p['asin']: p for p in all_products}
    product_prices = generate_product_prices(all_products)
    return (all_products, product_item_dict, product_prices, attribute_to_asins)

def init_search_engine(num_products=None):
    if num_products == 100:
        indexes = 'indexes_100'
    elif num_products == 1000:
        indexes = 'indexes_1k'
    elif num_products == 100000:
        indexes = 'indexes_100k'
    elif num_products is None:
        indexes = 'indexes'
    else:
        raise NotImplementedError(f'num_products being {num_products} is not supported yet.')
    search_engine = LuceneSearcher(os.path.join(BASE_DIR, f'../search_engine/{indexes}'))
    return search_engine

def get_goals(all_products, product_prices, human_goals=True):
    if human_goals:
        return get_human_goals(all_products, product_prices)
    else:
        return get_synthetic_goals(all_products, product_prices)

def setup_logger(session_id, user_log_dir):
    """Creates a log file and logging object for the corresponding session ID"""
    logger = logging.getLogger(session_id)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(user_log_dir / f'{session_id}.jsonl', mode='w')
    file_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger

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

def clean_product_keys(products):
    for product in products:
        product.pop('product_information', None)
        product.pop('brand', None)
        product.pop('brand_url', None)
        product.pop('list_price', None)
        product.pop('availability_quantity', None)
        product.pop('availability_status', None)
        product.pop('total_reviews', None)
        product.pop('total_answered_questions', None)
        product.pop('seller_id', None)
        product.pop('seller_name', None)
        product.pop('fulfilled_by_amazon', None)
        product.pop('fast_track_message', None)
        product.pop('aplus_present', None)
        product.pop('small_description_old', None)
    print('Keys cleaned.')
    return products

def generate_product_prices(all_products):
    product_prices = dict()
    for product in all_products:
        asin = product['asin']
        pricing = product['pricing']
        if not pricing:
            price = 100.0
        elif len(pricing) == 1:
            price = pricing[0]
        else:
            price = random.uniform(*pricing[:2])
        product_prices[asin] = price
    return product_prices

def test_setup_logger():
    LOG_DIR = 'user_session_logs_test/'
    user_log_dir = Path(LOG_DIR)
    user_log_dir.mkdir(parents=True, exist_ok=True)
    session_id = 'ABC'
    logger = setup_logger(session_id, user_log_dir)
    log_file = Path(LOG_DIR + '/' + session_id + '.jsonl')
    assert Path(log_file).is_file()
    assert logger.level == logging.INFO
    content = 'Hello there'
    logger.info(content)
    assert log_file.read_text().strip('\n') == content
    shutil.rmtree(LOG_DIR)

