# Cluster 8

@app.route('/done/<session_id>/<asin>/<options>', methods=['GET', 'POST'])
def done(session_id, asin, options):
    options = literal_eval(options)
    goal = user_sessions[session_id]['goal']
    purchased_product = product_item_dict[asin]
    price = product_prices[asin]
    reward, reward_info = get_reward(purchased_product, goal, price=price, options=options, verbose=True)
    user_sessions[session_id]['done'] = True
    user_sessions[session_id]['reward'] = reward
    print(user_sessions)
    logger = logging.getLogger(session_id)
    logger.info(json.dumps(dict(page='done', url=request.url, goal=goal, content=dict(asin=asin, options=options, price=price), reward=reward, reward_info=reward_info)))
    del logging.root.manager.loggerDict[session_id]
    return map_action_to_html(f'click[{END_BUTTON}]', session_id=session_id, reward=reward, asin=asin, options=options, reward_info=reward_info, query=purchased_product['query'], category=purchased_product['category'], product_category=purchased_product['product_category'], goal_attrs=user_sessions[session_id]['goal']['attributes'], purchased_attrs=purchased_product['Attributes'], goal=goal, mturk_code=generate_mturk_code(session_id))

def get_reward(purchased_product, goal, price, options, **kwargs):
    """Get cumulative reward score for purchased product and goal"""
    r_type_dict = get_type_reward(purchased_product, goal)
    r_price = price <= goal['price_upper'] if goal['price_upper'] > 0 else None
    r_att, num_attr_matches = get_attribute_reward(purchased_product, goal)
    r_option, num_option_matches = get_option_reward(list(options.values()), goal['goal_options'].items() if isinstance(goal['goal_options'], dict) else goal['goal_options'])
    total_reward = (num_attr_matches + num_option_matches + r_price) / (len(goal['attributes']) + len(goal['goal_options']) + 1)
    total_reward *= r_type_dict['r_type']
    if kwargs.get('verbose', False):
        info = {'r_type': r_type_dict['r_type'], 'r_att': r_att, 'w_att': len(goal['attributes']) / (len(goal['attributes']) + len(goal['goal_options']) + 1), 'query_match': r_type_dict['query_match'], 'category_match': r_type_dict['category_match'], 'title_score': r_type_dict['title_score']}
        if r_option is not None:
            info['r_option'] = r_option
            info['w_option'] = len(goal['goal_options']) / (len(goal['attributes']) + len(goal['goal_options']) + 1)
        if r_price is not None:
            info['r_price'] = r_price
            info['w_price'] = 1 / (len(goal['attributes']) + len(goal['goal_options']) + 1)
        return (total_reward, info)
    return total_reward

def generate_mturk_code(session_id: str) -> str:
    """Generates a redeem code corresponding to the session ID for an MTurk
    worker once the session is completed
    """
    sha = hashlib.sha1(session_id.encode())
    return sha.hexdigest()[:10].upper()

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

def get_type_reward(purchased_product, goal):
    """Determines the type reward - captures whether chosen product is in the same category"""
    query_match = purchased_product['query'] == goal['query']
    purchased_product_category = [x.strip() for x in purchased_product['product_category'].split('›')]
    goal_product_category = [x.strip() for x in goal['product_category'].split('›')]
    category_match = len(set(purchased_product_category) & set(goal_product_category)) >= 2
    purchased_type = purchased_product['name']
    desired_type = goal['name']
    purchased_type_parse = nlp(purchased_type)
    desired_type_parse = nlp(desired_type)
    purchased_type_parse = [t.text.lower() for t in purchased_type_parse if t.pos_ in ('PNOUN', 'NOUN', 'PROPN')]
    desired_type_parse = [t.text.lower() for t in desired_type_parse if t.pos_ in ('PNOUN', 'NOUN', 'PROPN')]
    n_intersect_type = len(set(purchased_type_parse) & set(desired_type_parse))
    if len(desired_type_parse) == 0:
        title_score = 0.2
    else:
        title_score = n_intersect_type / len(desired_type_parse)
    r_type = 1.0
    match = query_match or category_match or title_score > 0.2
    if not match:
        r_type = 0.5
    if title_score < 0.1:
        r_type = 0.1
    if title_score == 0.0:
        r_type = 0.0
    return dict(r_type=r_type, query_match=query_match, category_match=category_match, title_score=title_score)

def get_attribute_reward(purchased_product, goal):
    """Determines whether purchased products shares same attributes as goal"""
    purchased_attrs = purchased_product['Attributes']
    goal_attrs = goal['attributes']
    num_attr_matches = 0
    for g_attr in goal_attrs:
        matched = False
        for p_attr in purchased_attrs:
            score = fuzz.token_set_ratio(p_attr, g_attr)
            if score > 85:
                num_attr_matches += 1
                matched = True
                break
        if not matched and (g_attr in purchased_product['Title'].lower() or g_attr in ' '.join(purchased_product['BulletPoints']).lower() or g_attr in purchased_product['Description'].lower()):
            num_attr_matches += 1
            matched = True
    r_attr = num_attr_matches / len(goal_attrs)
    return (r_attr, num_attr_matches)

class WebEnv:
    """ A wrapper of textEnv for models. Returns valid actions at each step of the game. """

    def __init__(self, args, split, server=None, id=None):
        self.env = WebAgentTextEnv(observation_mode=args.state_format, server=server, filter_goals=None, limit_goals=-1, num_products=args.num, human_goals=args.human_goals, get_image=args.get_image, num_prev_obs=args.num_prev_obs, num_prev_actions=args.num_prev_actions, session_prefix=id)
        if args.num is None:
            if split == 'test':
                self.goal_idxs = range(500)
            elif split == 'eval':
                self.goal_idxs = range(500, 1500)
            elif split == 'train':
                self.goal_idxs = range(1500, len(self.env.server.goals))
        else:
            self.goal_idxs = range(len(self.env.server.goals))
        print(self.goal_idxs)
        self.steps = 0
        self.step_limit = args.step_limit
        self.stats = defaultdict(int)
        self.session = None
        self.click_item_name = args.click_item_name
        self.asin2name = {k.lower(): v['Title'].lower() for k, v in self.env.server.product_item_dict.items()}
        self.name2asin = {v: k for k, v in self.asin2name.items()}
        self.attributes_fail = defaultdict(int)
        self.attributes_success = defaultdict(int)
        self.items_clicked = defaultdict(int)
        self.harsh_reward = args.harsh_reward
        self.go_to_item = args.go_to_item
        self.go_to_search = args.go_to_search
        self.ban_buy = args.ban_buy
        self.prev_ob = self.cur_ob = None
        self.get_image = args.get_image
        self.item_rank = -1
        self.reduce_click = 1
        if args.extra_search_path != '':
            self.extra_search = json.load(open(args.extra_search_path))
            self.extra_search = {k.strip('.'): v for k, v in self.extra_search.items()}
        else:
            self.extra_search = None

    def get_search_texts(self, atts, query, inst):
        if self.extra_search is not None:
            if ', and price lower than' in inst:
                idx = inst.find(', and price lower than')
                inst_ = inst[:idx]
            else:
                inst_ = inst
            texts = self.extra_search.get(inst_, []) + [inst.lower()]
        else:
            texts = [query] + [f'{att} {query}' for att in atts] + [inst.lower()]
        return texts

    def get_valid_actions(self):
        valid_info = self.env.get_available_actions()
        if valid_info['has_search_bar']:
            atts = self.session['goal']['attributes']
            query = self.session['goal']['query']
            inst = self.session['goal']['instruction_text']
            texts = self.get_search_texts(atts, query, inst)
            valids = [f'search[{text}]' for text in texts]
        else:
            valids = []
            for text in valid_info['clickables']:
                if text == 'buy now' and self.ban_buy:
                    cur_options = len(self.session['options'])
                    all_options = len(self.env.server.product_item_dict[self.session['asin']]['customization_options'])
                    if cur_options != all_options:
                        continue
                if text != 'search':
                    if self.click_item_name and text in self.asin2name:
                        text = 'item - ' + self.asin2name[text]
                    valids.append(f'click[{text}]')
                if self.reduce_click and len(valids) > 20:
                    valids = valids[:6] + random.sample(valids[6:], 10)
        if len(valids) == 0:
            valids = ['finish']
        return valids

    def score(self):
        """
        Calculate the score of the current state.
        """
        valid_acts = self.get_valid_actions()
        if 'click[description]' not in valid_acts:
            return 0.0
        product = self.env.server.product_item_dict[self.session['asin']]
        goal = self.session['goal']
        price = self.env.server.product_prices.get(self.session['asin'])
        options = self.session['options']
        return get_reward(product, goal, price, options)

    def estimate_score(self, atts, opts, verify=False):
        """
        Calculate the score of the current state.
        """
        valid_acts = self.get_valid_actions()
        assert 'click[description]' in valid_acts
        desc = self.step('click[description]')[0].lower()
        self.step('click[< prev]')
        feat = self.step('click[features]')[0].lower()
        ob = self.step('click[< prev]')[0].lower()
        n_att = 0
        for att in atts:
            if att in desc or att in feat or att in ob:
                n_att += 1
        r_att = n_att / len(atts)
        n_opt = 0
        for opt in opts:
            for act in valid_acts:
                if opt in act:
                    n_opt += 1
                    break
        r_opt = n_opt / len(opts)
        r = (n_att + n_opt + 1) / (len(atts) + len(opts) + 1)
        return (r, r_att, r_opt)

    def step(self, action):
        if self.click_item_name and action.startswith('click[item - ') and (action[13:-1] in self.name2asin):
            valid_items = [_ for _ in self.get_valid_actions() if _.startswith('click[item - ')]
            if action in valid_items:
                self.item_rank = valid_items.index(action) + 1
            else:
                self.item_rank = -1
            action = f'click[{self.name2asin[action[13:-1]]}]'
        ob, reward, done, info = self.env.step(action)
        if action.startswith('click[') and action[6:-1] in self.asin2name:
            self.items_clicked[action[6:-1]] += 1
            desc = self.env.step('click[description]')[0].lower()
            self.env.step('click[< prev]')
            feat = self.env.step('click[features]')[0].lower()
            self.env.step('click[< prev]')
        else:
            desc = feat = ''
        r_visit = 0.0
        self.cur_ob, self.prev_ob = (ob, self.cur_ob)
        if info is None:
            info = {}
        self.steps += 1
        if self.step_limit and self.steps >= self.step_limit:
            done = True
        if done:
            info['verbose'] = self.session.get('verbose_info', {'r_att': 0.0, 'r_option': 0.0, 'r_price': 0.0, 'r_type': 0.0, 'w_att': 0.0, 'w_option': 0.0, 'w_price': 0.0})
            verbose = info['verbose']
            verbose['r_harsh'] = reward == 1
            verbose['r_exact'] = reward == 1 and self.session['goal']['asin'] == self.session['asin']
            verbose['r_norm'] = reward / self.steps
            verbose['r_visit'] = r_visit
            verbose['rank_item'] = self.item_rank
            if self.harsh_reward:
                reward = verbose['r_harsh']
            for k, v in self.session['actions'].items():
                self.stats[f'action_{k}'] += v
            cat = self.session['goal']['category']
            self.stats[f'cat_{cat}'] += 1
            for att in self.session['goal']['attributes']:
                if att in info['verbose'].get('purchased_attrs', []):
                    self.attributes_success[att] += 1
                else:
                    self.attributes_fail[att] += 1
        info.update({'valid': self.get_valid_actions(), 'goal': self.env.instruction_text, 'score': reward * 10, 'estimate_score': self.score(), 'prev_ob': self.prev_ob, 'desc': desc, 'feat': feat})
        if self.get_image:
            image_feat = self.env.get_image()
            info['image_feat'] = image_feat
        return (ob, (reward + r_visit) * 10, done, info)

    def reset(self, idx=None):
        if idx is None:
            idx = random.sample(self.goal_idxs, k=1)[0]
        ob, info = self.env.reset(idx)
        self.session = self.env.server.user_sessions[self.env.session]
        if info is None:
            info = {}
        self.cur_ob, self.prev_ob = (ob, None)
        info.update({'valid': self.get_valid_actions(), 'goal': self.env.instruction_text, 'score': 0, 'estimate_score': self.score(), 'prev_ob': self.prev_ob, 'desc': '', 'feat': ''})
        self.steps = 0
        if self.go_to_search or self.go_to_item:
            name = self.session['goal']['name'].lower()
            ob, _, _, info = self.step(f'search[{name}]')
            self.stats['action_go_to_search'] += 1
            if self.go_to_item:
                asin = self.session['goal']['asin'].lower()
                if asin in self.env.get_available_actions()['clickables']:
                    ob, _, _, info = self.step(f'click[{asin}]')
                    self.stats['action_go_to_item'] += 1
        self.item_rank = -1
        return (ob, info)

    def close(self):
        self.env.close()

def test_generate_mturk_code():
    suite = [('', 'DA39A3EE5E'), ('ABC', '3C01BDBB26'), ('123', '40BD001563'), ('1A1', '10E7DB0A44'), ('$%^ABC', '5D5607D24E')]
    for session_id, expected in suite:
        output = generate_mturk_code(session_id)
        assert type(expected) is str
        assert output == expected

def test_get_type_reward():
    goal = {'query': 'Query 1', 'product_category': 'a › b › c', 'name': 'Name 1'}
    purchased = {'query': 'Query 1', 'product_category': 'a › b › c', 'name': 'Name 1'}
    result = get_type_reward(purchased, goal)
    assert result['r_type'] == 1.0
    assert result['query_match'] == True
    assert result['category_match'] == True
    assert result['title_score'] == 1
    purchased['query'] = 'Query 2'
    result = get_type_reward(purchased, goal)
    assert result['query_match'] == False
    purchased['product_category'] = 'b › c › a'
    result = get_type_reward(purchased, goal)
    assert result['category_match'] == True
    purchased['product_category'] = 'd › e › f'
    result = get_type_reward(purchased, goal)
    assert result['category_match'] == False
    purchased['product_category'] = 'a › d › b'
    result = get_type_reward(purchased, goal)
    assert result['category_match'] == True
    purchased['product_category'] = 'a › a › b'
    result = get_type_reward(purchased, goal)
    assert result['category_match'] == True
    purchased['product_category'] = 'a › a › d'
    result = get_type_reward(purchased, goal)
    assert result['category_match'] == False
    goal['name'] = 'Mens D.O.N. Issue 2 Gca Basketball Sneakers Shoes Casual - Off White'
    purchased['name'] = 'PEAK High Top Mens Basketball Shoes Lou Williams Streetball Master Breathable Non Slip Outdoor Sneakers'
    result = get_type_reward(purchased, goal)
    assert isclose(result['title_score'], 0.333, abs_tol=0.01)
    goal['name'] = 'Saireed UL Listed 2 Prong Power Cord for JBL Bar 3.1 Bar 2.1 Channel 4K Ultra HD Soundbar Home Theater System Subwoofer'
    purchased['name'] = 'BRST AC Power Cord Outlet Socket Cable Plug Lead for Panasonic SC-HT830V DVD/VCR Combo Home Theater System'
    result = get_type_reward(purchased, goal)
    assert isclose(result['title_score'], 0.3, abs_tol=0.01)
    goal['name'] = 'Saireed UL Listed 2 Prong Power Cord for JBL Bar 3.1 Bar 2.1 Channel 4K Ultra HD Soundbar'
    purchased['name'] = 'BRST AC Power Cord Outlet Socket Cable Plug Lead for Panasonic SC-HT830V DVD/VCR Combo Home Theater System'
    result = get_type_reward(purchased, goal)
    assert isclose(result['title_score'], 0.15, abs_tol=0.01)
    goal['name'] = 'Rusticware 921ORB Kitchen and Bath Cabinet Knob'
    purchased['name'] = 'Minkissy 2pcs Stainless Steel Eyebrow Tweezers Blackhead Acne Remover Portable Makeup Tweezers (Silver)'
    result = get_type_reward(purchased, goal)
    assert result['title_score'] < 0.05

def test_get_attribute_reward():
    goal = {'attributes': ['tea tree', 'essential oils', 'natural ingredients']}
    purchased = {'Attributes': ['tea tree', 'essential oil', 'natural ingredients']}
    r_attr, num_attr_matches = get_attribute_reward(purchased, goal)
    assert r_attr == 1
    assert num_attr_matches == 3
    goal = {'attributes': ['tea tree', 'essential oils', 'natural ingredients']}
    purchased = {'Attributes': ['essential oil', 'natural ingredients'], 'Title': '', 'BulletPoints': [], 'Description': ''}
    r_attr, num_attr_matches = get_attribute_reward(purchased, goal)
    assert r_attr == 2.0 / 3.0
    assert num_attr_matches == 2
    goal = {'attributes': ['tea tree', 'essential oils', 'natural ingredients']}
    purchased = {'Attributes': [], 'Title': '', 'BulletPoints': ['This shampoo has essential oils and smells like lemons'], 'Description': 'Best shampoo on the market, made with natural ingredients'}
    r_attr, num_attr_matches = get_attribute_reward(purchased, goal)
    assert r_attr == 2.0 / 3.0
    assert num_attr_matches == 2
    goal = {'attributes': ['tea tree', 'essential oils', 'natural ingredients']}
    purchased = {'Attributes': ['tea bag', 'earl gray', 'lipton'], 'Title': 'English tea for breakfast', 'BulletPoints': ['Soothing aroma', 'Calming, great feeling'], 'Description': 'Best tea made by Lipton, great to pair with breakfast'}
    r_attr, num_attr_matches = get_attribute_reward(purchased, goal)
    assert r_attr == 0
    assert num_attr_matches == 0

def test_get_reward():
    goal = {'query': 'Query 1', 'product_category': 'a › b › c', 'name': 'Mens D.O.N. Issue 2 Gca Basketball Sneakers Shoes Casual - Off White', 'attributes': ['tea tree', 'essential oils', 'natural ingredients'], 'goal_options': {'color': 'grey', 'size': 'XL'}, 'price_upper': 40.0}
    purchased = {'query': 'Query 1', 'product_category': 'a › b › c', 'name': 'Mens D.O.N. Issue 2 Gca Basketball Sneakers Shoes Casual - Off White', 'Attributes': ['tea tree', 'essential oil', 'natural ingredients'], 'Title': '', 'BulletPoints': [], 'Description': '', 'goal_options': {'color': 'grey', 'size': 'XL'}}
    total_reward = get_reward(purchased, goal, 35, purchased['goal_options'])
    assert total_reward == 1
    purchased['Attributes'] = []
    purchased['Title'] = ''
    purchased['BulletPoints'] = 'This shampoo has essential oils and smells like lemons'
    purchased['Description'] = 'Best shampoo on the market, made with natural ingredients'
    total_reward = get_reward(purchased, goal, 35, purchased['goal_options'])
    assert isclose(total_reward, 2.0 / 3.0, abs_tol=0.01)
    goal['goal_options'] = {'color': 'grey', 'size': 'XL', 'amount': 'pack of 12'}
    total_reward = get_reward(purchased, goal, 35, purchased['goal_options'])
    assert isclose(total_reward, 0.5714, abs_tol=0.01)
    goal['name'] = 'Saireed UL Listed 2 Prong Power Cord for JBL Bar 3.1 Bar 2.1 Channel 4K Ultra HD Soundbar'
    purchased['name'] = 'BRST AC Power Cord Outlet Socket Cable Plug Lead for Panasonic SC-HT830V DVD/VCR Combo Home Theater System'
    purchased['query'] = 'Query 2'
    purchased['product_category'] = 'a › d › e'
    total_reward = get_reward(purchased, goal, 35, purchased['goal_options'])
    assert isclose(total_reward, 0.2857, abs_tol=0.01)

