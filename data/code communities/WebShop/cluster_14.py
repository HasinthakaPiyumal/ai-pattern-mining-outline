# Cluster 14

def random_idx(cum_weights):
    """Generate random index by sampling uniformly from sum of all weights, then
    selecting the `min` between the position to keep the list sorted (via bisect)
    and the value of the second to last index
    """
    pos = random.uniform(0, cum_weights[-1])
    idx = bisect.bisect(cum_weights, pos)
    idx = min(idx, len(cum_weights) - 2)
    return idx

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

def test_random_idx():
    random.seed(24)
    weights = [random.randint(0, 10) for _ in range(0, 50)]
    cml_weights = [0] + np.cumsum(weights).tolist()
    idx_1, expected_1 = (random_idx(cml_weights), 44)
    idx_2, expected_2 = (random_idx(cml_weights), 15)
    idx_3, expected_3 = (random_idx(cml_weights), 36)
    assert idx_1 == expected_1
    assert idx_2 == expected_2
    assert idx_3 == expected_3

