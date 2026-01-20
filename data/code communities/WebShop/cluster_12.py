# Cluster 12

class WebAgentTextEnv(gym.Env):
    """Gym environment for Text mode of WebShop environment"""

    def __init__(self, observation_mode='html', file_path=DEFAULT_FILE_PATH, server=None, **kwargs):
        """
        Constructor for text environment

        Arguments:
        observation_mode (`str`) -- ['html' | 'text'] (default 'html')
        get_image
        filter_goals
        limit_goals
        num_products
        human_goals
        session
        session_prefix
        show_attrs
        """
        super(WebAgentTextEnv, self).__init__()
        self.observation_mode = observation_mode
        self.kwargs = kwargs
        self.file_path = file_path
        self.base_url = 'http://127.0.0.1:3000'
        self.server = SimServer(self.base_url, self.file_path, self.kwargs.get('filter_goals'), self.kwargs.get('limit_goals', -1), self.kwargs.get('num_products'), self.kwargs.get('human_goals'), self.kwargs.get('show_attrs', False)) if server is None else server
        self.browser = SimBrowser(self.server)
        self.session = self.kwargs.get('session')
        self.session_prefix = self.kwargs.get('session_prefix')
        if self.kwargs.get('get_image', 0):
            self.feats = torch.load(FEAT_CONV)
            self.ids = torch.load(FEAT_IDS)
            self.ids = {url: idx for idx, url in enumerate(self.ids)}
        self.prev_obs = []
        self.prev_actions = []
        self.num_prev_obs = self.kwargs.get('num_prev_obs', 0)
        self.num_prev_actions = self.kwargs.get('num_prev_actions', 0)
        self.reset()

    def step(self, action):
        """
        Takes an action, updates WebShop environment, and returns (observation, reward, done, info)

        Arguments:
        action (`str`): An action should be of the following structure:
          - search[keywords]
          - click[value]
        If action not valid, perform nothing.
        """
        info = None
        self.get_available_actions()
        action_name, action_arg = parse_action(action)
        if action_arg is not None:
            action_arg = action_arg.lower()
        if action_name == 'search' and action_arg is not None and (action_arg != ''):
            status = self.browser.search(action_arg)
        elif action_name == 'click' and action_arg in self.text_to_clickable.keys() and (action_arg != 'search'):
            status = self.browser.click(action_arg, self.text_to_clickable)
        else:
            status = dict(reward=0, done=False)
        ob = self.observation
        text_list = [ob]
        self.prev_actions.append(action)
        for i in range(1, 1 + max(self.num_prev_obs, self.num_prev_actions)):
            if len(self.prev_actions) >= i and self.num_prev_actions >= i:
                text_list.append(self.prev_actions[-i])
            if len(self.prev_obs) >= i and self.num_prev_obs >= i:
                text_list.append(self.prev_obs[-i])
        state = ' [SEP] '.join(text_list[::-1])
        self.prev_obs.append(ob)
        return (state, status['reward'], status['done'], info)

    def get_available_actions(self):
        """Returns list of available actions at the current step"""
        html_obj = self._parse_html()
        search_bar = html_obj.find(id='search_input')
        has_search_bar = True if search_bar is not None else False
        buttons = html_obj.find_all(class_='btn')
        product_links = html_obj.find_all(class_='product-link')
        buying_options = html_obj.select('input[type="radio"]')
        self.text_to_clickable = {f'{b.get_text()}'.lower(): b for b in buttons + product_links}
        for opt in buying_options:
            opt_value = opt.get('value')
            self.text_to_clickable[f'{opt_value}'] = opt
        return dict(has_search_bar=has_search_bar, clickables=list(self.text_to_clickable.keys()))

    def get_image(self):
        """Scrape image from page HTML and return as a list of pixel values"""
        html_obj = self._parse_html(self.browser.page_source)
        image_url = html_obj.find(id='product-image')
        if image_url is not None:
            image_url = image_url['src']
            if image_url in self.ids:
                image_idx = self.ids[image_url]
                image = self.feats[image_idx]
                return image
        return torch.zeros(512)

    def get_instruction_text(self):
        """Get corresponding instruction text for current environment session"""
        html_obj = self._parse_html(self.browser.page_source)
        instruction_text = html_obj.find(id='instruction-text').h4.text
        return instruction_text

    def _parse_html(self, html=None):
        """
        Returns web request result wrapped in BeautifulSoup object

        Arguments:
        url (`str`): If no url or html is provided, use the current
            observation (HTML) for parsing.
        """
        if html is None:
            html = self.state['html']
        html_obj = BeautifulSoup(html, 'html.parser')
        return html_obj

    @property
    def observation(self):
        """Compiles state into either the `html` or `text` observation mode"""
        html = self.state['html']
        if self.observation_mode == 'html':
            return html
        elif self.observation_mode == 'text':
            return self.convert_html_to_text(html, simple=True)
        elif self.observation_mode == 'text_rich':
            return self.convert_html_to_text(html, simple=False)
        elif self.observation_mode == 'url':
            return self.state['url']
        else:
            raise ValueError(f'Observation mode {self.observation_mode} not supported.')

    @property
    def state(self):
        """
        State that includes all information. The actual observation are
        likely to be a subset or reduced form of the state.
        """
        return dict(url=self.browser.current_url, html=self.browser.page_source, instruction_text=self.instruction_text)

    def convert_html_to_text(self, html, simple=False):
        """Strip HTML of tags and add separators to convert observation into simple mode"""
        texts = self._parse_html(html).findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        if simple:
            return ' [SEP] '.join((t.strip() for t in visible_texts if t != '\n'))
        else:
            observation = ''
            for t in visible_texts:
                if t == '\n':
                    continue
                if t.parent.name == 'button':
                    processed_t = f'[button] {t} [button_]'
                elif t.parent.name == 'label':
                    if f'"{t}"' in self.state['url']:
                        processed_t = f'  [clicked button] {t} [clicked button_]'
                        observation = f'You have clicked {t}.\n' + observation
                    else:
                        processed_t = f'  [button] {t} [button_]'
                elif t.parent.get('class') == ['product-link']:
                    if f'{t}' in self.server.user_sessions[self.session]['asins']:
                        processed_t = f'\n[clicked button] {t} [clicked button_]'
                    else:
                        processed_t = f'\n[button] {t} [button_]'
                else:
                    processed_t = str(t)
                observation += processed_t + '\n'
            return observation

    def reset(self, session=None, instruction_text=None):
        """Create a new session and reset environment variables"""
        session_int = None
        if session is not None:
            self.session = str(session)
            if isinstance(session, int):
                session_int = session
        else:
            self.session = ''.join(random.choices(string.ascii_lowercase, k=10))
        if self.session_prefix is not None:
            self.session = self.session_prefix + self.session
        init_url = f'{self.base_url}/{self.session}'
        self.browser.get(init_url, session_id=self.session, session_int=session_int)
        self.text_to_clickable = None
        self.instruction_text = self.get_instruction_text() if instruction_text is None else instruction_text
        obs = self.observation
        self.prev_obs = [obs]
        self.prev_actions = []
        return (obs, None)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

def parse_action(action):
    """
    Parse action string to action name and its arguments.
    """
    pattern = re.compile('(.+)\\[(.+)\\]')
    m = re.match(pattern, action)
    if m is None:
        action_name = action
        action_arg = None
    else:
        action_name, action_arg = m.groups()
    return (action_name, action_arg)

class WebAgentSiteEnv(gym.Env):
    """Gym environment for HTML mode of WebShop environment"""

    def __init__(self, observation_mode='html', **kwargs):
        """
        Constructor for HTML environment

        Arguments:
        observation_mode (`str`) -- ['html' | 'text'] (default 'html')
        pause (`float`) -- Pause (in seconds) after taking an action. 
            This is mainly for demo purposes.
            Recommended value: 2.0s
        render (`bool`) -- Show browser if set to `True`.
        session ('str') -- Session ID to initialize environment with
        """
        super(WebAgentSiteEnv, self).__init__()
        self.observation_mode = observation_mode
        self.kwargs = kwargs
        service = Service(join(dirname(abspath(__file__)), 'chromedriver'))
        options = Options()
        if 'render' not in kwargs or not kwargs['render']:
            options.add_argument('--headless')
        self.browser = webdriver.Chrome(service=service, options=options)
        self.text_to_clickable = None
        self.assigned_session = kwargs.get('session')
        self.session = None
        self.reset()

    def step(self, action):
        """
        Takes an action, updates WebShop environment, and returns (observation, reward, done, info)

        Arguments:
        action (`str`): An action should be of the following structure:
          - search[keywords]
          - click[value]
        If action not valid, perform nothing.
        """
        reward = 0.0
        done = False
        info = None
        action_name, action_arg = parse_action(action)
        if action_name == 'search':
            try:
                search_bar = self.browser.find_element_by_id('search_input')
            except Exception:
                pass
            else:
                search_bar.send_keys(action_arg)
                search_bar.submit()
        elif action_name == 'click':
            try:
                self.text_to_clickable[action_arg].click()
            except ElementNotInteractableException:
                button = self.text_to_clickable[action_arg]
                self.browser.execute_script('arguments[0].click();', button)
            reward = self.get_reward()
            if action_arg == END_BUTTON:
                done = True
        elif action_name == 'end':
            done = True
        else:
            print('Invalid action. No action performed.')
        if 'pause' in self.kwargs:
            time.sleep(self.kwargs['pause'])
        return (self.observation, reward, done, info)

    def get_available_actions(self):
        """Returns list of available actions at the current step"""
        try:
            search_bar = self.browser.find_element_by_id('search_input')
        except Exception:
            has_search_bar = False
        else:
            has_search_bar = True
        buttons = self.browser.find_elements_by_class_name('btn')
        product_links = self.browser.find_elements_by_class_name('product-link')
        buying_options = self.browser.find_elements_by_css_selector("input[type='radio']")
        self.text_to_clickable = {f'{b.text}': b for b in buttons + product_links}
        for opt in buying_options:
            opt_value = opt.get_attribute('value')
            self.text_to_clickable[f'{opt_value}'] = opt
        return dict(has_search_bar=has_search_bar, clickables=list(self.text_to_clickable.keys()))

    def _parse_html(self, html=None, url=None):
        """
        Returns web request result wrapped in BeautifulSoup object

        Arguments:
        url (`str`): If no url or html is provided, use the current
            observation (HTML) for parsing.
        """
        if html is None:
            if url is not None:
                html = requests.get(url)
            else:
                html = self.state['html']
        html_obj = BeautifulSoup(html, 'html.parser')
        return html_obj

    def get_reward(self):
        """Get reward value at current step of the environment"""
        html_obj = self._parse_html()
        r = html_obj.find(id='reward')
        r = float(r.findChildren('pre')[0].string) if r is not None else 0.0
        return r

    def get_instruction_text(self):
        """Get corresponding instruction text for environment current step"""
        html_obj = self._parse_html(self.browser.page_source)
        instruction_text = html_obj.find(id='instruction-text').h4.text
        return instruction_text

    def convert_html_to_text(self, html):
        """Strip HTML of tags and add separators to convert observation into simple mode"""
        texts = self._parse_html(html).findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        observation = ' [SEP] '.join((t.strip() for t in visible_texts if t != '\n'))
        return observation

    @property
    def state(self):
        """
        State that includes all information. The actual observation are
        likely to be a subset or reduced form of the state.
        """
        return dict(url=self.browser.current_url, html=self.browser.page_source, instruction_text=self.instruction_text)

    @property
    def observation(self):
        """Compiles state into either the `html` or `text` observation mode"""
        html = self.state['html']
        if self.observation_mode == 'html':
            return html
        elif self.observation_mode == 'text':
            return self.convert_html_to_text(html)
        else:
            raise ValueError(f'Observation mode {self.observation_mode} not supported.')

    @property
    def action_space(self):
        return NotImplementedError

    @property
    def observation_space(self):
        return NotImplementedError

    def reset(self):
        """Create a new session and reset environment variables"""
        if self.assigned_session is not None:
            self.session = self.assigned_session
        else:
            self.session = ''.join(random.choices(string.ascii_lowercase, k=5))
        init_url = f'http://127.0.0.1:3000/{self.session}'
        self.browser.get(init_url)
        self.instruction_text = self.get_instruction_text()
        return (self.observation, None)

    def render(self, mode='human'):
        return NotImplementedError

    def close(self):
        self.browser.close()
        print('Browser closed.')

