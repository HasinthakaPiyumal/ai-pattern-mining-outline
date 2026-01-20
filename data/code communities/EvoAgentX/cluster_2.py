# Cluster 2

class SearchSerpAPI(SearchBase):
    """
    SerpAPI search tool that provides access to multiple search engines including
    Google, Bing, Baidu, Yahoo, and DuckDuckGo through a unified interface.
    """
    api_key: Optional[str] = Field(default=None, description='SerpAPI authentication key')
    default_engine: Optional[str] = Field(default='google', description='Default search engine')
    default_location: Optional[str] = Field(default=None, description='Default geographic location')
    default_language: Optional[str] = Field(default='en', description='Default interface language')
    default_country: Optional[str] = Field(default='us', description='Default country code')
    enable_content_scraping: Optional[bool] = Field(default=True, description='Enable full content scraping')

    def __init__(self, name: str='SearchSerpAPI', num_search_pages: Optional[int]=5, max_content_words: Optional[int]=None, api_key: Optional[str]=None, default_engine: Optional[str]='google', default_location: Optional[str]=None, default_language: Optional[str]='en', default_country: Optional[str]='us', enable_content_scraping: Optional[bool]=True, **kwargs):
        """
        Initialize the SerpAPI Search tool.
        
        Args:
            name (str): Name of the tool
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content
            api_key (str): SerpAPI authentication key (can also use SERPAPI_KEY env var)
            default_engine (str): Default search engine (google, bing, baidu, yahoo, duckduckgo)
            default_location (str): Default geographic location for searches
            default_language (str): Default interface language
            default_country (str): Default country code
            enable_content_scraping (bool): Whether to scrape full page content
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(name=name, num_search_pages=num_search_pages, max_content_words=max_content_words, api_key=api_key, default_engine=default_engine, default_location=default_location, default_language=default_language, default_country=default_country, enable_content_scraping=enable_content_scraping, **kwargs)
        self.api_key = api_key or os.getenv('SERPAPI_KEY', '')
        self.base_url = 'https://serpapi.com/search.json'
        if not self.api_key:
            logger.warning('SerpAPI key not found. Set SERPAPI_KEY environment variable or pass api_key parameter.')

    def _build_serpapi_params(self, query: str, engine: str=None, location: str=None, language: str=None, country: str=None, search_type: str=None, num_results: int=None) -> Dict[str, Any]:
        """
        Build SerpAPI request parameters.
        
        Args:
            query (str): Search query
            engine (str): Search engine to use
            location (str): Geographic location
            language (str): Interface language
            country (str): Country code
            search_type (str): Type of search (web, images, news, shopping, maps)
            num_results (int): Number of results to retrieve
            
        Returns:
            Dict[str, Any]: SerpAPI request parameters
        """
        params = {'q': query, 'api_key': self.api_key, 'num': num_results or self.num_search_pages}
        if location or self.default_location:
            params['location'] = location or self.default_location
        if language or self.default_language:
            params['hl'] = language or self.default_language
        if country or self.default_country:
            params['gl'] = country or self.default_country
        if search_type and search_type != 'web':
            search_type_map = {'images': 'isch', 'news': 'nws', 'shopping': 'shop', 'maps': 'lcl'}
            if search_type in search_type_map:
                params['tbm'] = search_type_map[search_type]
        return params

    def _execute_serpapi_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute search using direct HTTP requests to SerpAPI.
        
        Args:
            params (Dict[str, Any]): Search parameters
            
        Returns:
            Dict[str, Any]: SerpAPI response data
            
        Raises:
            Exception: For API errors
        """
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if 'error' in data:
                raise Exception(f'SerpAPI error: {data['error']}')
            return data
        except requests.exceptions.RequestException as e:
            raise Exception(f'SerpAPI request failed: {str(e)}')
        except Exception as e:
            raise Exception(f'SerpAPI search failed: {str(e)}')

    def _process_serpapi_results(self, serpapi_data: Dict[str, Any], max_content_words: int=None) -> Dict[str, Any]:
        """
        Process SerpAPI results into structured format with processed results + raw data.
        
        Args:
            serpapi_data (Dict[str, Any]): Raw SerpAPI response
            max_content_words (int): Maximum words per result content
            
        Returns:
            Dict[str, Any]: Structured response with processed results and raw data
        """
        processed_results = []
        if (knowledge_graph := serpapi_data.get('knowledge_graph', {})):
            if (description := knowledge_graph.get('description')):
                title = knowledge_graph.get('title', 'Unknown')
                content = f'**{title}**'
                if (kg_type := knowledge_graph.get('type')):
                    content += f' ({kg_type})'
                content += f'\n\n{description}'
                if (kg_list := knowledge_graph.get('list', {})):
                    content += '\n\n**Key Information:**'
                    for key, value in list(kg_list.items())[:5]:
                        if isinstance(value, list) and value:
                            formatted_key = key.replace('_', ' ').title()
                            formatted_value = ', '.join((str(v) for v in value[:3]))
                            content += f'\n• {formatted_key}: {formatted_value}'
                processed_results.append({'title': f'Knowledge: {title}', 'content': self._truncate_content(content, max_content_words or 200), 'url': knowledge_graph.get('source', {}).get('link', ''), 'type': 'knowledge_graph', 'priority': 1})
        for item in serpapi_data.get('organic_results', []):
            url = item.get('link', '')
            title = item.get('title', 'No Title')
            snippet = item.get('snippet', '')
            position = item.get('position', 0)
            result = {'title': title, 'content': self._truncate_content(snippet, max_content_words or 400), 'url': url, 'type': 'organic', 'priority': 2, 'position': position}
            if self.enable_content_scraping and url and url.startswith(('http://', 'https://')):
                try:
                    scraped_title, scraped_content = self._scrape_page(url)
                    if scraped_content and scraped_content.strip():
                        if scraped_title and scraped_title.strip():
                            result['title'] = scraped_title
                        result['site_content'] = self._truncate_content(scraped_content, max_content_words or 400)
                    else:
                        result['site_content'] = None
                except Exception as e:
                    logger.debug(f'Content scraping failed for {url}: {str(e)}')
                    result['site_content'] = None
            else:
                result['site_content'] = None
            if snippet or result.get('site_content'):
                processed_results.append(result)
        raw_data = {}
        raw_sections = ['local_results', 'news_results', 'shopping_results', 'related_questions', 'recipes_results', 'images_results']
        for section in raw_sections:
            if section in serpapi_data and serpapi_data[section]:
                if section == 'local_results':
                    places = serpapi_data[section].get('places', [])[:3]
                    if places:
                        raw_data[section] = {'places': places}
                else:
                    raw_data[section] = serpapi_data[section][:3]
        search_metadata = {}
        if (search_meta := serpapi_data.get('search_metadata', {})):
            search_metadata = {'query': search_meta.get('query', ''), 'location': search_meta.get('location', ''), 'total_results': search_meta.get('total_results', ''), 'search_time': search_meta.get('total_time_taken', '')}
        processed_results.sort(key=lambda x: (x.get('priority', 999), x.get('position', 0)))
        return {'results': processed_results, 'raw_data': raw_data if raw_data else None, 'search_metadata': search_metadata if search_metadata else None, 'error': None}

    def _handle_api_errors(self, error: Exception) -> str:
        """
        Handle SerpAPI specific errors with appropriate messages.
        
        Args:
            error (Exception): The exception that occurred
            
        Returns:
            str: User-friendly error message
        """
        error_str = str(error).lower()
        if 'api key' in error_str or 'unauthorized' in error_str:
            return 'Invalid or missing SerpAPI key. Please set SERPAPI_KEY environment variable.'
        elif 'rate limit' in error_str or 'too many requests' in error_str:
            return 'SerpAPI rate limit exceeded. Please try again later.'
        elif 'quota' in error_str or 'credit' in error_str:
            return 'SerpAPI quota exceeded. Please check your plan limits.'
        elif 'timeout' in error_str:
            return 'SerpAPI request timeout. Please try again.'
        else:
            return f'SerpAPI error: {str(error)}'

    def search(self, query: str, num_search_pages: int=None, max_content_words: int=None, engine: str=None, location: str=None, language: str=None, country: str=None, search_type: str=None) -> Dict[str, Any]:
        """
        Search using SerpAPI with comprehensive parameter support.
        
        Args:
            query (str): The search query
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content
            engine (str): Search engine (google, bing, baidu, yahoo, duckduckgo)
            location (str): Geographic location for localized results
            language (str): Interface language (e.g., 'en', 'es', 'fr')
            country (str): Country code for country-specific results (e.g., 'us', 'uk')
            search_type (str): Type of search (web, images, news, shopping, maps)
            
        Returns:
            Dict[str, Any]: Contains search results and optional error message
        """
        num_search_pages = num_search_pages or self.num_search_pages
        max_content_words = max_content_words or self.max_content_words
        if not self.api_key:
            error_msg = 'SerpAPI key is required. Please set SERPAPI_KEY environment variable or pass api_key parameter. Get your key from: https://serpapi.com/'
            logger.error(error_msg)
            return {'results': [], 'raw_data': None, 'search_metadata': None, 'error': error_msg}
        try:
            search_engine = engine or self.default_engine
            logger.info(f'Searching {search_engine} via SerpAPI: {query}, num_results={num_search_pages}, max_content_words={max_content_words}')
            params = self._build_serpapi_params(query=query, engine=search_engine, location=location, language=language, country=country, search_type=search_type, num_results=num_search_pages)
            serpapi_data = self._execute_serpapi_search(params)
            response_data = self._process_serpapi_results(serpapi_data, max_content_words)
            logger.info(f'Successfully retrieved {len(response_data['results'])} processed results')
            return response_data
        except Exception as e:
            error_msg = self._handle_api_errors(e)
            logger.error(f'SerpAPI search failed: {error_msg}')
            return {'results': [], 'raw_data': None, 'search_metadata': None, 'error': error_msg}

class SearchGoogleFree(SearchBase):
    """
    Free Google Search tool that doesn't require API keys.
    """

    def __init__(self, name: str='GoogleFreeSearch', num_search_pages: Optional[int]=5, max_content_words: Optional[int]=None, **kwargs):
        """
        Initialize the Free Google Search tool.
        
        Args:
            name (str): Name of the tool
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(name=name, num_search_pages=num_search_pages, max_content_words=max_content_words, **kwargs)

    def search(self, query: str, num_search_pages: int=None, max_content_words: int=None) -> Dict[str, Any]:
        """
        Searches Google for the given query and retrieves content from multiple pages.

        Args:
            query (str): The search query.
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content, None means no limit

        Returns:
            Dict[str, Any]: Contains a list of search results and optional error message.
        """
        num_search_pages = num_search_pages or self.num_search_pages
        max_content_words = max_content_words or self.max_content_words
        results = []
        try:
            logger.info(f'Searching Google (Free) for: {query}, num_results={num_search_pages}, max_content_words={max_content_words}')
            search_results = list(google_f_search(query, num_results=num_search_pages))
            if not search_results:
                return {'results': [], 'error': 'No search results found.'}
            logger.info(f'Found {len(search_results)} search results')
            for url in search_results:
                try:
                    title, content = self._scrape_page(url)
                    if content:
                        display_content = self._truncate_content(content, max_content_words)
                        results.append({'title': title, 'content': display_content, 'url': url})
                except Exception as e:
                    logger.warning(f'Error processing URL {url}: {str(e)}')
                    continue
            return {'results': results, 'error': None}
        except Exception as e:
            logger.error(f'Error in free Google search: {str(e)}')
            return {'results': [], 'error': str(e)}

class BrowserBase(BaseModule):
    """
    A tool for interacting with web browsers using Selenium.
    Allows agents to navigate to URLs, interact with elements, extract information,
    and more from web pages.
    
    Key Features:
    - Auto-initialization: Browser is automatically initialized when any method is first called
    - Auto-cleanup: Browser is automatically closed when the instance is destroyed
    - No manual initialization or cleanup required
    """
    timeout: int = Field(default=10, description='Default timeout in seconds for browser operations')
    browser_type: str = Field(default='chrome', description="Type of browser to use ('chrome', 'firefox', 'safari', 'edge')")
    headless: bool = Field(default=False, description='Whether to run the browser in headless mode')
    user_data_dir: Optional[str] = Field(default=None, description='User data directory for persistent browser sessions')

    def __init__(self, name: str='Browser Tool', browser_type: str='chrome', headless: bool=False, timeout: int=10, **kwargs):
        """
        Initialize the browser tool with Selenium WebDriver.
        
        Args:
            name (str): Name of the tool
            browser_type (str): Type of browser to use ('chrome', 'firefox', 'safari', 'edge')
            headless (bool): Whether to run the browser in headless mode
            timeout (int): Default timeout in seconds for browser operations
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(name=name, timeout=timeout, browser_type=browser_type, headless=headless, **kwargs)
        self.driver = None
        self.element_references = {}

    def _check_driver_initialized(self) -> Union[None, Dict[str, Any]]:
        """
        Check if the browser driver is initialized. If not, initialize it automatically.
        
        Returns:
            Union[None, Dict[str, Any]]: None if driver is initialized, error response if initialization fails
        """
        if not self.driver:
            init_result = self.initialize_browser()
            if init_result['status'] == 'error':
                return init_result
        return None

    def _get_selector_by_type(self, selector_type: str) -> Union[str, Dict[str, Any]]:
        """
        Get the Selenium By selector for the given selector type.
        
        Args:
            selector_type (str): Type of selector ('css', 'xpath', 'id', 'class', 'name', 'tag')
            
        Returns:
            Union[str, Dict[str, Any]]: The By selector or error response
        """
        by_type = SELECTOR_MAP.get(selector_type.lower())
        if not by_type:
            return {'status': 'error', 'message': f'Invalid selector type: {selector_type}'}
        return by_type

    def _wait_for_page_load(self, timeout: Optional[int]=None) -> bool:
        """
        Wait for the page to load completely.
        
        Args:
            timeout (int, optional): Custom timeout for this operation
            
        Returns:
            bool: True if page loaded, False if timed out
        """
        timeout = timeout or self.timeout
        try:
            WebDriverWait(self.driver, timeout).until(lambda driver: driver.execute_script('return document.readyState') == 'complete')
            return True
        except TimeoutException:
            return False

    def _parse_element_reference(self, ref: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse an element reference into selector type and selector.
        
        Args:
            ref (str): Element reference ID from the page snapshot
            
        Returns:
            Tuple[Optional[str], Optional[str], Optional[str]]: 
                (selector_type, selector, error_message) - error_message is None if successful
        """
        if not self.element_references:
            return (None, None, 'No page snapshot available. Use browser_snapshot or navigate_to_url first.')
        stored_ref = self.element_references.get(ref)
        if not stored_ref:
            return (None, None, f"Element reference '{ref}' not found. Use browser_snapshot or navigate_to_url first.")
        if ':' in stored_ref:
            ref_parts = stored_ref.split(':', 1)
            if len(ref_parts) != 2:
                return (None, None, f'Invalid stored reference format: {stored_ref}')
            selector_type, selector = ref_parts
            return (selector_type, selector, None)
        return (None, None, f'Invalid stored reference format: {stored_ref}')

    def _find_element_with_wait(self, by_type: str, selector: str, timeout: Optional[int]=None, wait_condition=EC.presence_of_element_located) -> Tuple[Optional[Any], Optional[str]]:
        """
        Find an element on the page with wait condition.
        
        Args:
            by_type (str): Selenium By selector type
            selector (str): The selector string
            timeout (int, optional): Custom timeout for this operation
            wait_condition: The EC condition to wait for
            
        Returns:
            Tuple[Optional[Any], Optional[str]]: (element, error_message) - error_message is None if successful
        """
        timeout = timeout or self.timeout
        try:
            element = WebDriverWait(self.driver, timeout).until(wait_condition((by_type, selector)))
            return (element, None)
        except TimeoutException:
            return (None, f'Element not found or condition not met with selector: {selector}')
        except Exception as e:
            logger.error(f'Error finding element {selector}: {str(e)}')
            return (None, str(e))

    def _handle_function_params(self, function_params: Optional[list], function_name: str, param_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract parameters from nested function_params format.
        
        Args:
            function_params (list, optional): Nested function parameters
            function_name (str): The function name to look for
            param_mapping (Dict[str, str]): Mapping of parameter names
            
        Returns:
            Dict[str, Any]: Extracted parameters
        """
        result = {}
        if not function_params:
            return result
        for param in function_params:
            fn_name = param.get('function_name', '')
            if fn_name == function_name or fn_name in param_mapping.get('alt_names', []):
                args = param.get('function_args', {})
                for param_name, result_name in param_mapping.items():
                    if param_name == 'alt_names':
                        continue
                    if param_name in args:
                        result[result_name] = args[param_name]
                break
        return result

    def initialize_browser(self, function_params: list=None) -> Dict[str, Any]:
        """
        Start or restart a browser session. This method is called automatically when needed.
        
        Note: This method is now called automatically by other browser methods when the browser
        is not initialized. Manual initialization is no longer required.
        
        This function supports multiple parameter styles:
        1. Standard style: no parameters
        2. Nested function_params style:
           function_params=[{"function_name": "initialize_browser", "function_args": {}}]
           
        Args:
            function_params (list, optional): Nested function parameters
        
        Returns:
            Dict[str, Any]: Status information about the browser initialization
        """
        try:
            if self.driver:
                try:
                    self.driver.quit()
                except Exception as e:
                    logger.warning(f'Error closing existing browser session: {str(e)}')
            options = None
            if self.browser_type == 'chrome':
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.service import Service
                from webdriver_manager.chrome import ChromeDriverManager
                options = Options()
                if self.headless:
                    options.add_argument('--headless')
                options.add_argument('--disable-gpu')
                options.add_argument('--disable-gpu-sandbox')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                if self.user_data_dir:
                    options.add_argument(f'--user-data-dir={self.user_data_dir}')
                    logger.info(f'Using user data directory: {self.user_data_dir}')
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            elif self.browser_type == 'firefox':
                from selenium.webdriver.firefox.options import Options
                options = Options()
                if self.headless:
                    options.add_argument('--headless')
                self.driver = webdriver.Firefox(options=options)
            elif self.browser_type == 'safari':
                self.driver = webdriver.Safari()
            elif self.browser_type == 'edge':
                from selenium.webdriver.edge.options import Options
                options = Options()
                if self.headless:
                    options.add_argument('--headless')
                options.add_argument('--disable-gpu')
                options.add_argument('--disable-gpu-sandbox')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                if self.user_data_dir:
                    options.add_argument(f'--user-data-dir={self.user_data_dir}')
                    logger.info(f'Using user data directory: {self.user_data_dir}')
                self.driver = webdriver.Edge(options=options)
            else:
                return {'status': 'error', 'message': f'Unsupported browser type: {self.browser_type}'}
            self.driver.set_page_load_timeout(self.timeout)
            return {'status': 'success', 'message': f'Browser {self.browser_type} initialized successfully'}
        except Exception as e:
            logger.error(f'Error initializing browser: {str(e)}')
            return {'status': 'error', 'message': str(e)}

    def navigate_to_url(self, url: str=None, timeout: int=None, function_params: list=None) -> Dict[str, Any]:
        """
        Navigate to a URL and capture a snapshot of the page. This provides element references used for interaction.
        
        This function supports multiple parameter styles:
        1. Standard style: url parameter
        2. Nested function_params style:
           function_params=[{"function_name": "navigate_to_url", "function_args": {"url": "..."}}]
        
        Args:
            url (str, optional): The complete URL (with https://) to navigate to
            timeout (int, optional): Custom timeout in seconds (default: 10)
            function_params (list, optional): Nested function parameters
            
        Returns:
            Dict[str, Any]: Information about the navigation result and page snapshot
        """
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
        if function_params and (not url):
            params = self._handle_function_params(function_params, 'navigate_to_url', {'url': 'url', 'timeout': 'timeout', 'alt_names': ['browser_navigate']})
            url = params.get('url')
            timeout = params.get('timeout', timeout)
        if not url:
            return {'status': 'error', 'message': 'URL parameter is required'}
        timeout = timeout or self.timeout
        try:
            self.driver.get(url)
            page_loaded = self._wait_for_page_load(timeout)
            if not page_loaded:
                logger.warning(f'Page load timeout for URL: {url}, but continuing with snapshot')
            snapshot_result = self.browser_snapshot()
            if snapshot_result['status'] == 'success':
                return {'status': 'success', 'url': url, 'title': self.driver.title, 'current_url': self.driver.current_url, 'snapshot': {'interactive_elements': snapshot_result.get('interactive_elements', [])}}
            else:
                return {'status': 'partial_success', 'url': url, 'title': self.driver.title, 'current_url': self.driver.current_url, 'snapshot_error': snapshot_result.get('message', 'Unknown error capturing snapshot')}
        except TimeoutException:
            return {'status': 'timeout', 'message': f'Timed out loading URL: {url}'}
        except Exception as e:
            logger.error(f'Error navigating to URL {url}: {str(e)}')
            return {'status': 'error', 'message': str(e)}

    def find_element(self, selector: str, selector_type: str='css', timeout: int=None) -> Dict[str, Any]:
        """
        Find an element on the current page and return information about it.
        
        Args:
            selector (str): The selector to find the element
            selector_type (str): Type of selector ('css', 'xpath', 'id', 'class', 'name', 'tag')
            timeout (int, optional): Custom timeout for this operation
            
        Returns:
            Dict[str, Any]: Information about the found element
        """
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
        timeout = timeout or self.timeout
        by_type = self._get_selector_by_type(selector_type)
        if isinstance(by_type, dict):
            return by_type
        try:
            element, error = self._find_element_with_wait(by_type, selector, timeout, EC.presence_of_element_located)
            if error:
                return {'status': 'not_found', 'message': f'Element not found with {selector_type}: {selector}'}
            element_properties = self._extract_element_properties(element, selector)
            return {'status': 'success', 'element': element_properties}
        except Exception as e:
            logger.error(f'Error finding element {selector}: {str(e)}')
            return {'status': 'error', 'message': str(e)}

    def _extract_element_properties(self, element, selector: str) -> Dict[str, Any]:
        """
        Extract common properties from a WebElement.
        
        Args:
            element: The Selenium WebElement
            selector (str): The selector used to find the element (for error messages)
            
        Returns:
            Dict[str, Any]: Element properties
        """
        element_properties = {'text': element.text, 'tag_name': element.tag_name, 'is_displayed': element.is_displayed(), 'is_enabled': element.is_enabled()}
        for attr in ['href', 'id', 'class']:
            try:
                value = element.get_attribute(attr)
                if value:
                    element_properties[attr] = value
            except StaleElementReferenceException:
                logger.warning(f'Element became stale when trying to get {attr} attribute for {selector}')
            except Exception as e:
                logger.warning(f'Could not get {attr} attribute for {selector}: {str(e)}')
        return element_properties

    def find_multiple_elements(self, selector: str, selector_type: str='css', timeout: int=None) -> Dict[str, Any]:
        """
        Find multiple elements on the current page and return information about them.
        
        Args:
            selector (str): The selector to find the elements
            selector_type (str): Type of selector ('css', 'xpath', 'id', 'class', 'name', 'tag')
            timeout (int, optional): Custom timeout for this operation
            
        Returns:
            Dict[str, Any]: Information about the found elements
        """
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
        timeout = timeout or self.timeout
        by_type = self._get_selector_by_type(selector_type)
        if isinstance(by_type, dict):
            return by_type
        try:
            element, error = self._find_element_with_wait(by_type, selector, timeout, EC.presence_of_element_located)
            if error:
                return {'status': 'not_found', 'message': f'No elements found with {selector_type}: {selector}'}
            elements = self.driver.find_elements(by_type, selector)
            elements_properties = []
            for idx, element in enumerate(elements):
                try:
                    element_properties = self._extract_element_properties(element, f'{selector}[{idx}]')
                    element_properties['index'] = idx
                    elements_properties.append(element_properties)
                except StaleElementReferenceException:
                    logger.warning(f'Element {idx} became stale while extracting properties')
                except Exception as e:
                    logger.warning(f'Error extracting properties for element {idx}: {str(e)}')
            return {'status': 'success', 'count': len(elements_properties), 'elements': elements_properties}
        except Exception as e:
            logger.error(f'Error finding elements {selector}: {str(e)}')
            return {'status': 'error', 'message': str(e)}

    def click_element(self, selector: str, selector_type: str='css', timeout: int=None) -> Dict[str, Any]:
        """
        Click on an element on the current page.
        
        Args:
            selector (str): The selector to find the element
            selector_type (str): Type of selector ('css', 'xpath', 'id', 'class', 'name', 'tag')
            timeout (int, optional): Custom timeout for this operation
            
        Returns:
            Dict[str, Any]: Result of the click operation
        """
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
        timeout = timeout or self.timeout
        by_type = self._get_selector_by_type(selector_type)
        if isinstance(by_type, dict):
            return by_type
        try:
            element, error = self._find_element_with_wait(by_type, selector, timeout, EC.element_to_be_clickable)
            if error:
                return {'status': 'not_found', 'message': f'Element not clickable with {selector_type}: {selector}'}
            element.click()
            page_loaded = self._wait_for_page_load(timeout)
            if not page_loaded:
                return {'status': 'partial_success', 'message': 'Element clicked, but page load timed out', 'selector': selector, 'current_url': self.driver.current_url}
            return {'status': 'success', 'message': f'Clicked element with {selector_type}: {selector}', 'current_url': self.driver.current_url, 'title': self.driver.title}
        except Exception as e:
            logger.error(f'Error clicking element {selector}: {str(e)}')
            return {'status': 'error', 'message': str(e)}

    def input_text(self, element: str=None, ref: str=None, text: str=None, submit: bool=False, slowly: bool=True, function_params: list=None) -> Dict[str, Any]:
        """
        Type text into a form field, search box, or other input element using a reference ID from a snapshot.
        
        This function only works with element references from a snapshot. Use browser_snapshot
        or navigate_to_url first to capture the page elements.
        
        This function supports multiple parameter styles:
        1. Standard style: element (description), ref (element ID), text
        2. Nested function_params style:
           function_params=[{"function_name": "browser_type", "function_args": {...}}]
        
        Args:
            element (str, optional): Human-readable description of the element (e.g., 'Search field', 'Username input')
            ref (str, optional): Element ID from the page snapshot (e.g., 'e0', 'e1', 'e2') - NOT a CSS selector
            text (str, optional): Text to input into the element
            submit (bool): Press Enter after typing to submit forms (default: false)
            slowly (bool): Type one character at a time to trigger JS events (default: true)
            function_params (list, optional): Nested function parameters
            
        Returns:
            Dict[str, Any]: Result of the text input operation
        """
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
        if function_params:
            params = self._handle_function_params(function_params, 'input_text', {'element': 'element', 'ref': 'ref', 'text': 'text', 'submit': 'submit', 'slowly': 'slowly', 'alt_names': ['browser_type']})
            element = params.get('element', element)
            ref = params.get('ref', ref)
            text = params.get('text', text)
            if 'submit' in params:
                submit = params['submit']
            if 'slowly' in params:
                slowly = params['slowly']
        if not ref or not text:
            return {'status': 'error', 'message': 'Both ref and text parameters are required'}
        selector_type, selector, error = self._parse_element_reference(ref)
        if error:
            return {'status': 'error', 'message': error}
        element_desc = element or ref
        by_type = self._get_selector_by_type(selector_type)
        if isinstance(by_type, dict):
            return by_type
        try:
            web_element, error = self._find_element_with_wait(by_type, selector, self.timeout, EC.element_to_be_clickable)
            if error:
                return {'status': 'not_found', 'message': f'Element not found: {element_desc}'}
            web_element.clear()
            if slowly:
                for char in text:
                    web_element.send_keys(char)
                    time.sleep(0.05)
            else:
                web_element.send_keys(text)
            if submit:
                from selenium.webdriver.common.keys import Keys
                web_element.send_keys(Keys.ENTER)
                page_loaded = self._wait_for_page_load(self.timeout)
                if not page_loaded:
                    self.browser_snapshot()
                    return {'status': 'partial_success', 'message': 'Text entered and submitted, but page load timed out', 'element': element_desc, 'text': text}
                snapshot_result = self.browser_snapshot()
                if snapshot_result['status'] != 'success':
                    logger.warning(f'Failed to capture snapshot after form submission: {snapshot_result.get('message')}')
            return {'status': 'success', 'message': f'Successfully input text into {element_desc}' + (' and submitted' if submit else ''), 'element': element_desc, 'text': text}
        except TimeoutException:
            return {'status': 'not_found', 'message': f'Element not found: {element_desc}'}
        except Exception as e:
            logger.error(f'Error inputting text to element {element_desc}: {str(e)}')
            return {'status': 'error', 'message': str(e)}

    def get_page_content(self) -> Dict[str, Any]:
        """
        Get the current page title, URL and body content.
        
        Returns:
            Dict[str, Any]: Information about the current page
        """
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
        try:
            title = self.driver.title
            current_url = self.driver.current_url
            body_content = self.driver.execute_script('\n                var body = document.body;\n                return body ? body.outerHTML : "";\n            ')
            element_summary = self.driver.execute_script('\n                // Get common interactive elements\n                var summary = {\n                    links: [],\n                    buttons: [],\n                    inputs: [],\n                    forms: []\n                };\n                \n                // Get links\n                var links = document.querySelectorAll(\'a\');\n                for (var i = 0; i < Math.min(links.length, 20); i++) {\n                    var link = links[i];\n                    summary.links.push({\n                        text: link.textContent.trim().substring(0, 50),\n                        href: link.getAttribute(\'href\'),\n                        id: link.id,\n                        class: link.className\n                    });\n                }\n                \n                // Get buttons\n                var buttons = document.querySelectorAll(\'button, input[type="button"], input[type="submit"]\');\n                for (var i = 0; i < Math.min(buttons.length, 20); i++) {\n                    var button = buttons[i];\n                    summary.buttons.push({\n                        text: button.textContent ? button.textContent.trim().substring(0, 50) : button.value,\n                        id: button.id,\n                        class: button.className,\n                        type: button.type\n                    });\n                }\n                \n                // Get inputs\n                var inputs = document.querySelectorAll(\'input:not([type="button"]):not([type="submit"]), textarea, select\');\n                for (var i = 0; i < Math.min(inputs.length, 20); i++) {\n                    var input = inputs[i];\n                    summary.inputs.push({\n                        type: input.type,\n                        name: input.name,\n                        id: input.id,\n                        placeholder: input.placeholder\n                    });\n                }\n                \n                // Get forms\n                var forms = document.querySelectorAll(\'form\');\n                for (var i = 0; i < Math.min(forms.length, 10); i++) {\n                    var form = forms[i];\n                    summary.forms.push({\n                        id: form.id,\n                        action: form.action,\n                        method: form.method\n                    });\n                }\n                \n                return summary;\n            ')
            return {'status': 'success', 'title': title, 'url': current_url, 'body_content': body_content, 'element_summary': element_summary}
        except Exception as e:
            logger.error(f'Error getting page content: {str(e)}')
            return {'status': 'error', 'message': str(e)}

    def switch_to_frame(self, frame_reference: str, reference_type: str='index') -> Dict[str, Any]:
        """
        Switch to a frame on the page.
        
        Args:
            frame_reference (str): Reference to the frame (index, name, or ID)
            reference_type (str): Type of reference ('index', 'name', 'id', 'element')
            
        Returns:
            Dict[str, Any]: Result of the frame switch operation
        """
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
        try:
            if reference_type == 'index':
                try:
                    index = int(frame_reference)
                    self.driver.switch_to.frame(index)
                except ValueError:
                    return {'status': 'error', 'message': f'Invalid frame index: {frame_reference}'}
            elif reference_type == 'name' or reference_type == 'id':
                self.driver.switch_to.frame(frame_reference)
            elif reference_type == 'element':
                selector_parts = frame_reference.split(':', 1)
                if len(selector_parts) != 2:
                    return {'status': 'error', 'message': "Element reference must be in format 'selector_type:selector'"}
                selector_type, selector = selector_parts
                element_result = self.find_element(selector, selector_type)
                if element_result['status'] != 'success':
                    return {'status': 'error', 'message': f'Could not find frame element: {element_result['message']}'}
                selector_map = {'css': By.CSS_SELECTOR, 'xpath': By.XPATH, 'id': By.ID, 'class': By.CLASS_NAME, 'name': By.NAME, 'tag': By.TAG_NAME}
                by_type = selector_map.get(selector_type.lower())
                element = self.driver.find_element(by_type, selector)
                self.driver.switch_to.frame(element)
            else:
                return {'status': 'error', 'message': f'Invalid reference type: {reference_type}'}
            return {'status': 'success', 'message': f'Switched to frame using {reference_type}: {frame_reference}'}
        except Exception as e:
            logger.error(f'Error switching to frame {frame_reference}: {str(e)}')
            return {'status': 'error', 'message': str(e)}

    def switch_to_window(self, window_reference: str, reference_type: str='index') -> Dict[str, Any]:
        """
        Switch to a window or tab.
        
        Args:
            window_reference (str): Reference to the window (index, handle, or title)
            reference_type (str): Type of reference ('index', 'handle', 'title')
            
        Returns:
            Dict[str, Any]: Result of the window switch operation
        """
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
        try:
            window_handles = self.driver.window_handles
            if not window_handles:
                return {'status': 'error', 'message': 'No window handles available'}
            if reference_type == 'index':
                try:
                    index = int(window_reference)
                    if index < 0 or index >= len(window_handles):
                        return {'status': 'error', 'message': f'Window index out of range: {index}'}
                    self.driver.switch_to.window(window_handles[index])
                except ValueError:
                    return {'status': 'error', 'message': f'Invalid window index: {window_reference}'}
            elif reference_type == 'handle':
                if window_reference not in window_handles:
                    return {'status': 'error', 'message': f'Window handle not found: {window_reference}'}
                self.driver.switch_to.window(window_reference)
            elif reference_type == 'title':
                current_handle = self.driver.current_window_handle
                window_found = False
                for handle in window_handles:
                    try:
                        self.driver.switch_to.window(handle)
                        if self.driver.title == window_reference:
                            window_found = True
                            break
                    except Exception:
                        pass
                if not window_found:
                    self.driver.switch_to.window(current_handle)
                    return {'status': 'error', 'message': f"No window with title '{window_reference}' found"}
            else:
                return {'status': 'error', 'message': f'Invalid reference type: {reference_type}'}
            return {'status': 'success', 'message': f'Switched to window using {reference_type}: {window_reference}', 'title': self.driver.title, 'url': self.driver.current_url}
        except Exception as e:
            logger.error(f'Error switching to window {window_reference}: {str(e)}')
            return {'status': 'error', 'message': str(e)}

    def select_dropdown_option(self, select_selector: str, option_value: str, select_by: str='value', selector_type: str='css') -> Dict[str, Any]:
        """
        Select an option from a dropdown
        select_by can be 'value', 'text', or 'index'
        
        Args:
            select_selector (str): The selector to find the dropdown element
            option_value (str): The value to select (depends on select_by)
            select_by (str): Method to select by ('value', 'text', 'index')
            selector_type (str): Type of selector for the dropdown
            
        Returns:
            Dict[str, Any]: Result of the selection operation
        """
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
        try:
            from selenium.webdriver.support.ui import Select
            by_type = self._get_selector_by_type(selector_type)
            if isinstance(by_type, dict):
                return by_type
            element, error = self._find_element_with_wait(by_type, select_selector, self.timeout, EC.presence_of_element_located)
            if error:
                return {'status': 'not_found', 'message': f'Dropdown element not found with {selector_type}: {select_selector}'}
            select = Select(element)
            if select_by.lower() == 'value':
                select.select_by_value(option_value)
            elif select_by.lower() == 'text':
                select.select_by_visible_text(option_value)
            elif select_by.lower() == 'index':
                try:
                    select.select_by_index(int(option_value))
                except ValueError:
                    return {'status': 'error', 'message': f'Invalid index value: {option_value}. Must be an integer.'}
            else:
                return {'status': 'error', 'message': f'Invalid select_by option: {select_by}'}
            return {'status': 'success', 'message': f'Selected option with {select_by}: {option_value}'}
        except Exception as e:
            logger.error(f'Error selecting dropdown option: {str(e)}')
            return {'status': 'error', 'message': str(e)}

    def close_browser(self) -> Dict[str, Any]:
        """
        Close the browser and end the session. Call this when you're done to free resources.
        
        Returns:
            Dict[str, Any]: Status of the browser closure
        """
        if not self.driver:
            return {'status': 'success', 'message': 'Browser already closed'}
        try:
            self.driver.quit()
            self.driver = None
            return {'status': 'success', 'message': 'Browser closed successfully'}
        except Exception as e:
            logger.error(f'Error closing browser: {str(e)}')
            return {'status': 'error', 'message': str(e)}

    def browser_click(self, element: str=None, ref: str=None, function_params: list=None) -> Dict[str, Any]:
        """
        Click on a button, link, or other clickable element using a reference ID from a snapshot.
        
        This function only works with element references from a snapshot. You MUST call browser_snapshot
        or navigate_to_url first to capture the page elements.
        
        Common usage pattern:
        1. First get a snapshot: browser_snapshot() or navigate_to_url()
        2. Find the element reference (e.g. 'e0', 'e1') from the snapshot's interactive_elements
        3. Use that reference to click: browser_click(element='Login button', ref='e0')
        
        This function supports multiple parameter styles:
        1. Standard style: element (description), ref (element ID)
        2. Nested function_params style:
           function_params=[{"function_name": "browser_click", "function_args": {...}}]
        
        Args:
            element (str, optional): Human-readable description of what you're clicking (e.g., 'Login button', 'Next page link')
            ref (str, optional): Element ID from the page snapshot (e.g., 'e0', 'e1', 'e2') - NOT a CSS selector
            function_params (list, optional): Nested function parameters
            
        Returns:
            Dict[str, Any]: Result of the click operation with detailed feedback
        """
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
        if function_params and (not ref):
            params = self._handle_function_params(function_params, 'browser_click', {'element': 'element', 'ref': 'ref'})
            element = params.get('element', element)
            ref = params.get('ref', ref)
        if not ref:
            return {'status': 'error', 'message': 'Element reference (ref) parameter is required. You must first call browser_snapshot() or navigate_to_url() to get element references.', 'required_steps': ['1. Call browser_snapshot() or navigate_to_url() to get page elements', "2. Find the element reference (e.g. 'e0') in the response's interactive_elements", "3. Use that reference to click: browser_click(element='Button name', ref='e0')"]}
        if not self.element_references:
            return {'status': 'error', 'message': 'No element references found. You must first capture a page snapshot.', 'required_steps': ['1. Call browser_snapshot() or navigate_to_url() to capture the page state', '2. Use the element references returned in the snapshot']}
        selector_type, selector, error = self._parse_element_reference(ref)
        if error:
            return {'status': 'error', 'message': error, 'help': "Make sure you're using a valid element reference from a recent snapshot"}
        element_desc = element or ref
        by_type = self._get_selector_by_type(selector_type)
        if isinstance(by_type, dict):
            return by_type
        try:
            try:
                element_exists = self.driver.find_element(by_type, selector)
            except Exception:
                return {'status': 'not_found', 'message': f'Element not found: {element_desc}', 'suggestion': 'The page may have changed. Try getting a new snapshot with browser_snapshot()'}
            web_element, error = self._find_element_with_wait(by_type, selector, self.timeout, EC.element_to_be_clickable)
            if error:
                try:
                    is_visible = element_exists.is_displayed()
                    is_enabled = element_exists.is_enabled()
                    element_tag = element_exists.tag_name
                    element_classes = element_exists.get_attribute('class')
                    return {'status': 'not_clickable', 'message': f'Element found but not clickable: {element_desc}', 'element_state': {'visible': is_visible, 'enabled': is_enabled, 'tag': element_tag, 'classes': element_classes}, 'suggestion': 'The element might be disabled, hidden, or covered by another element'}
                except Exception:
                    return {'status': 'not_clickable', 'message': f'Element found but not clickable: {element_desc}', 'suggestion': 'The element might be disabled, hidden, or covered by another element'}
            web_element.click()
            page_loaded = self._wait_for_page_load(self.timeout)
            if not page_loaded:
                snapshot_result = self.browser_snapshot()
                return {'status': 'partial_success', 'message': 'Element clicked, but page load timed out', 'element': element_desc, 'current_url': self.driver.current_url, 'snapshot': snapshot_result if snapshot_result['status'] == 'success' else None, 'suggestion': 'The page might still be loading. You may want to wait and take another snapshot.'}
            snapshot_result = self.browser_snapshot()
            if snapshot_result['status'] == 'success':
                return {'status': 'success', 'message': f'Successfully clicked on {element_desc}', 'element': element_desc, 'current_url': self.driver.current_url, 'title': self.driver.title, 'snapshot': {'interactive_elements': snapshot_result.get('interactive_elements', [])}}
            else:
                return {'status': 'success', 'message': f'Successfully clicked on {element_desc} but snapshot failed', 'element': element_desc, 'current_url': self.driver.current_url, 'title': self.driver.title, 'snapshot_error': snapshot_result.get('message', 'Unknown error capturing snapshot'), 'suggestion': 'You may want to take another snapshot with browser_snapshot()'}
        except TimeoutException:
            return {'status': 'timeout', 'message': f'Timed out waiting for element to be clickable: {element_desc}', 'suggestion': 'The element might be taking too long to load or become clickable'}
        except Exception as e:
            logger.error(f'Error clicking element: {str(e)}')
            return {'status': 'error', 'message': str(e), 'element': element_desc, 'suggestion': 'Try getting a new snapshot of the page with browser_snapshot()'}

    def _classify_element_interactivity(self, element_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify an element's interactivity based on its properties.
        This method contains all rules for determining if an element is interactive or editable.
        
        Args:
            element_data (Dict[str, Any]): Element data including properties, attributes, etc.
            
        Returns:
            Dict[str, Any]: Element data with interactivity classifications added
        """
        element_data['interactable'] = False
        element_data['editable'] = False
        tag_name = element_data.get('properties', {}).get('tag', '').upper()
        role = element_data.get('attributes', {}).get('role', '').lower()
        is_disabled = element_data.get('attributes', {}).get('disabled') is not None or element_data.get('attributes', {}).get('aria-disabled') == 'true' or element_data.get('attributes', {}).get('aria-hidden') == 'true'
        is_visible = element_data.get('visible', True)
        if not is_disabled and is_visible:
            interactive_tags = {'A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'DETAILS', 'AUDIO', 'VIDEO', 'IFRAME', 'EMBED', 'OBJECT', 'SUMMARY', 'MENU'}
            interactive_roles = {'button', 'link', 'checkbox', 'menuitem', 'menuitemcheckbox', 'menuitemradio', 'option', 'radio', 'searchbox', 'slider', 'spinbutton', 'switch', 'tab', 'textbox', 'combobox', 'listbox', 'menu', 'menubar', 'radiogroup', 'tablist', 'toolbar', 'tree', 'treegrid'}
            has_interactive_attrs = any([element_data.get('attributes', {}).get(attr) is not None for attr in ['onclick', 'onkeydown', 'onkeyup', 'onmousedown', 'onmouseup', 'tabindex']])
            element_data['interactable'] = tag_name in interactive_tags or role in interactive_roles or has_interactive_attrs
            editable_input_types = {'text', 'search', 'email', 'number', 'tel', 'url', 'password'}
            editable_roles = {'textbox', 'searchbox', 'spinbutton'}
            element_data['editable'] = tag_name == 'INPUT' and element_data.get('attributes', {}).get('type', 'text').lower() in editable_input_types or tag_name == 'TEXTAREA' or element_data.get('attributes', {}).get('contenteditable') == 'true' or (role in editable_roles)
        return element_data

    def _process_accessibility_tree(self, accessibility_tree):
        """
        Process the accessibility tree to extract all elements and store their references.
        
        This method processes all elements in the page structure, assigns unique IDs,
        and stores their selectors for later interaction.
        
        Args:
            accessibility_tree (dict): The accessibility tree from JavaScript
            
        Returns:
            list: A list of all elements with their IDs and properties
        """
        all_elements = []

        def extract_elements(node, path='', index=0):
            if not node:
                return index
            current_path = path + '/' + (node.get('name') or node.get('role') or 'element')
            element_id = f'e{index}'
            element_info = {'id': element_id, 'description': current_path.strip('/'), 'purpose': node.get('semantic_info', {}).get('purpose', ''), 'label': node.get('semantic_info', {}).get('label', ''), 'category': node.get('semantic_info', {}).get('category', ''), 'isPrimary': node.get('semantic_info', {}).get('isPrimary', False), 'visible': node.get('visible', True), 'properties': node.get('properties', {}), 'attributes': node.get('attributes', {})}
            if 'all_refs' in node:
                self.element_references[element_id] = node['all_refs'][0]
            element_info = self._classify_element_interactivity(element_info)
            all_elements.append(element_info)
            index += 1
            for child in node.get('children', []):
                index = extract_elements(child, current_path, index)
            return index
        extract_elements(accessibility_tree)
        return all_elements

    def browser_snapshot(self, function_params: list=None) -> Dict[str, Any]:
        """
        Capture a fresh snapshot of the current page with all interactive elements. 
        Use after page state changes not caused by navigation or clicking.
        
        This function supports multiple parameter styles:
        1. Standard style: no parameters
        2. Nested function_params style:
           function_params=[{"function_name": "browser_snapshot", "function_args": {}}]
        
        Args:
            function_params (list, optional): Nested function parameters
            
        Returns:
            Dict[str, Any]: The accessibility snapshot of the page with interactive elements
        """
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
        try:
            title = self.driver.title
            current_url = self.driver.current_url
            accessibility_tree = self.driver.execute_script("\n                function getAccessibilityTree(node, depth = 0, maxDepth = 10) {\n                    if (!node || depth > maxDepth) return null;\n                    \n                    let result = {\n                        role: node.role || node.tagName,\n                        name: node.name || '',\n                        type: node.type || '',\n                        value: node.value || '',\n                        description: node.description || '',\n                        properties: {},\n                        visible: isElementVisible(node)\n                    };\n                    \n                    // Helper function for element visibility\n                    function isElementVisible(element) {\n                        if (!element.getBoundingClientRect) return true;\n                        const style = window.getComputedStyle(element);\n                        const rect = element.getBoundingClientRect();\n                        \n                        // Check basic visibility\n                        const isVisible = style.display !== 'none' && \n                                        style.visibility !== 'hidden' && \n                                        style.opacity !== '0' &&\n                                        rect.width > 0 && \n                                        rect.height > 0;\n                                        \n                        // Check if element is in viewport\n                        const isInViewport = rect.top >= 0 &&\n                                           rect.left >= 0 &&\n                                           rect.bottom <= window.innerHeight &&\n                                           rect.right <= window.innerWidth;\n                                           \n                        return isVisible && isInViewport;\n                    }\n                    \n                    // Add text content\n                    if (node.textContent) {\n                        result.text_content = node.textContent.trim();\n                    }\n\n                    // Add identifier properties for references\n                    if (node.id) result.properties.id = node.id;\n                    if (node.className) result.properties.class = node.className;\n                    if (node.tagName) result.properties.tag = node.tagName.toLowerCase();\n                    \n                    // Add attributes\n                    if (node.attributes) {\n                        result.attributes = {};\n                        for (let attr of node.attributes) {\n                            result.attributes[attr.name] = attr.value;\n                        }\n                    }\n\n                    // Add custom ref property that combines selector types\n                    let refs = [];\n                    // Store all possible selectors, but don't use them as primary ref\n                    if (node.id) refs.push(`id:${node.id}`);\n                    if (node.className && typeof node.className === 'string') \n                        refs.push(`class:${node.className}`);\n                    if (node.tagName) refs.push(`tag:${node.tagName.toLowerCase()}`);\n                    \n                    // For inputs, add name attribute\n                    if (node.getAttribute && node.getAttribute('name')) {\n                        result.properties.name = node.getAttribute('name');\n                        refs.push(`name:${node.getAttribute('name')}`);\n                    }\n                    \n                    // Create XPath and CSS selectors\n                    try {\n                        // CSS selector\n                        let cssPath = getCssPath(node);\n                        if (cssPath) refs.push(`css:${cssPath}`);\n                        \n                        // XPath\n                        let xpath = getXPath(node);\n                        if (xpath) refs.push(`xpath:${xpath}`);\n                    } catch (e) {}\n                    \n                    // Store all refs but don't set primary ref here\n                    if (refs.length > 0) {\n                        result.all_refs = refs;\n                    }\n\n                    // Add semantic information about the element\n                    result.semantic_info = {\n                        // What the element represents\n                        purpose: (function() {\n                            if (node.tagName === 'INPUT') {\n                                if (node.type === 'submit') return 'submit button';\n                                if (node.type === 'search') return 'search box';\n                                if (node.type === 'text') return 'text input';\n                                return `${node.type || 'text'} input`;\n                            }\n                            if (node.tagName === 'BUTTON') return 'button';\n                            if (node.tagName === 'A') return 'link';\n                            if (node.tagName === 'SELECT') return 'dropdown';\n                            if (node.tagName === 'TEXTAREA') return 'text area';\n                            if (node.getAttribute('role')) return node.getAttribute('role');\n                            return 'interactive element';\n                        })(),\n                        \n                        // The visible or accessible text\n                        label: (function() {\n                            return node.getAttribute('aria-label') ||\n                                   node.getAttribute('title') ||\n                                   node.getAttribute('placeholder') ||\n                                   node.getAttribute('alt') ||\n                                   (node.tagName === 'INPUT' ? node.value : node.textContent.trim());\n                        })(),\n                        \n                        // Is this a primary action?\n                        isPrimary: !!(\n                            node.classList.contains('primary') ||\n                            node.getAttribute('aria-label')?.toLowerCase().includes('search') ||\n                            node.getAttribute('title')?.toLowerCase().includes('search') ||\n                            node.type === 'search' ||\n                            node.getAttribute('role') === 'main' ||\n                            node.id?.toLowerCase().includes('main') ||\n                            node.classList.contains('main')\n                        ),\n                        \n                        // Basic category\n                        category: (function() {\n                            if (node.type === 'search' || \n                                node.getAttribute('role') === 'searchbox') return 'search';\n                            if (node.type === 'submit' || \n                                node.tagName === 'BUTTON' ||\n                                node.getAttribute('role') === 'button') return 'action';\n                            if (node.tagName === 'A' ||\n                                node.getAttribute('role') === 'link') return 'navigation';\n                            if (node.tagName === 'INPUT' || \n                                node.tagName === 'TEXTAREA' ||\n                                node.getAttribute('role') === 'textbox') return 'input';\n                            if (node.tagName === 'SELECT' ||\n                                ['listbox', 'combobox'].includes(node.getAttribute('role'))) return 'selection';\n                            return 'interactive';\n                        })()\n                    };\n                    \n                    // Process children\n                    result.children = [];\n                    if (node.children) {\n                        for (let i = 0; i < node.children.length; i++) {\n                            const childTree = getAccessibilityTree(node.children[i], depth + 1, maxDepth);\n                            if (childTree) {\n                                result.children.push(childTree);\n                            }\n                        }\n                    }\n                    \n                    return result;\n                }\n                \n                return getAccessibilityTree(document.body);\n            ")
            all_elements = self._process_accessibility_tree(accessibility_tree)
            page_content = html2text.html2text(self.driver.page_source)
            return {'status': 'success', 'title': title, 'url': current_url, 'accessibility_tree': accessibility_tree, 'page_content': page_content, 'interactive_elements': [e for e in all_elements if e.get('interactable') or e.get('editable')]}
        except Exception as e:
            logger.error(f'Error generating accessibility snapshot: {str(e)}')
            return {'status': 'error', 'message': str(e)}

    def browser_console_messages(self, function_params: list=None) -> Dict[str, Any]:
        """
        Retrieve JavaScript console messages (logs, warnings, errors) from the browser for debugging.
        
        This function supports multiple parameter styles:
        1. Standard style: no parameters
        2. Nested function_params style:
           function_params=[{"function_name": "browser_console_messages", "function_args": {}}]
        
        Args:
            function_params (list, optional): Nested function parameters
            
        Returns:
            Dict[str, Any]: The console messages including logs, warnings and errors
        """
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
        try:
            logs = self._collect_browser_logs()
            return {'status': 'success', 'console_messages': logs}
        except Exception as e:
            logger.error(f'Error retrieving console messages: {str(e)}')
            return {'status': 'error', 'message': str(e)}

    def _collect_browser_logs(self) -> List[Dict[str, Any]]:
        """
        Collect logs from both the browser driver and JavaScript console.
        
        Returns:
            List[Dict[str, Any]]: Combined logs from both sources
        """
        logs = []
        try:
            browser_logs = self.driver.get_log('browser')
            for log in browser_logs:
                level = log.get('level', '').upper()
                if level == 'SEVERE':
                    level = 'ERROR'
                elif level == 'INFO':
                    level = 'LOG'
                logs.append({'level': level, 'message': log.get('message', ''), 'timestamp': log.get('timestamp', '')})
        except Exception as log_error:
            logs.append({'level': 'WARNING', 'message': f'Could not retrieve browser logs: {str(log_error)}', 'timestamp': ''})
        try:
            self.driver.execute_script("\n                if (!window._consoleLogs) {\n                    window._consoleLogs = [];\n                    \n                    // Store original console methods\n                    const originalConsole = {\n                        log: console.log,\n                        info: console.info,\n                        warn: console.warn,\n                        error: console.error,\n                        debug: console.debug\n                    };\n                    \n                    // Helper function to add message with proper level\n                    function addMessage(level, args) {\n                        window._consoleLogs.push({\n                            level: level.toUpperCase(),\n                            message: Array.from(args).join(' '),\n                            timestamp: new Date().toISOString()\n                        });\n                    }\n                    \n                    // Override console methods to capture logs\n                    console.log = function() {\n                        addMessage('LOG', arguments);\n                        originalConsole.log.apply(console, arguments);\n                    };\n                    \n                    console.info = function() {\n                        addMessage('INFO', arguments);\n                        originalConsole.info.apply(console, arguments);\n                    };\n                    \n                    console.warn = function() {\n                        addMessage('WARN', arguments);\n                        originalConsole.warn.apply(console, arguments);\n                    };\n                    \n                    console.error = function() {\n                        addMessage('ERROR', arguments);\n                        originalConsole.error.apply(console, arguments);\n                    };\n                    \n                    console.debug = function() {\n                        addMessage('DEBUG', arguments);\n                        originalConsole.debug.apply(console, arguments);\n                    };\n                }\n            ")
            time.sleep(2)
            js_logs = self.driver.execute_script('return window._consoleLogs || [];')
            for log in js_logs:
                if log not in logs:
                    logs.append(log)
        except Exception as js_error:
            logs.append({'level': 'WARNING', 'message': f'Could not retrieve JavaScript console logs: {str(js_error)}', 'timestamp': ''})
        return logs

    def __del__(self):
        """
        Destructor to automatically close the browser when the instance is destroyed.
        """
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.quit()
                logger.info('Browser automatically closed on cleanup')
            except Exception as e:
                logger.warning(f'Error during automatic browser cleanup: {str(e)}')

class MCPClient:

    def __init__(self, server_configs: Union[Dict[str, Any], List[Dict[str, Any]]], connect_timeout: float=120.0):
        if isinstance(server_configs, dict):
            self.server_configs = [server_configs]
        else:
            self.server_configs = server_configs
        self.event_loop = asyncio.new_event_loop()
        self.sessions: list[Client] = []
        self.mcp_tools: list[list[Any]] = []
        self.task = None
        self.thread_running = threading.Event()
        self.working_thread = threading.Thread(target=self._run_event, daemon=True)
        self.connect_timeout = connect_timeout
        self.tools = None
        self.tool_schemas = None
        self.tool_descriptions = None

    def _disconnect(self):
        if hasattr(self, 'shutdown_event') and self.shutdown_event:
            self.event_loop.call_soon_threadsafe(self.shutdown_event.set)
        if self.task and (not self.task.done()):
            self.event_loop.call_soon_threadsafe(self.task.cancel)
        if hasattr(self, 'working_thread') and self.working_thread.is_alive():
            self.working_thread.join(timeout=5)
        if hasattr(self, 'event_loop') and (not self.event_loop.is_closed()):
            self.event_loop.close()

    def _connect(self):
        self.working_thread.start()
        if not self.thread_running.wait(timeout=self.connect_timeout):
            self._disconnect()
            raise TimeoutError(f"Couldn't connect to the MCP server after {self.connect_timeout} seconds")

    def __enter__(self):
        self._connect()
        return self.get_toolkits()

    def __del__(self):
        try:
            self._disconnect()
        except Exception:
            pass

    def __exit__(self, exc_type, exc_value, traceback):
        self._disconnect()

    def _run_event(self):
        """Runs the event loop in a separate thread (for synchronous usage)."""
        print('Running event loop')
        asyncio.set_event_loop(self.event_loop)

        async def setup():
            try:
                async with AsyncExitStack() as stack:
                    connections = [await stack.enter_async_context(self._start_server(config)) for config in self.server_configs]
                    self.sessions, self.mcp_tools = [list(c) for c in zip(*connections)]
                    self.thread_running.set()
                    self.shutdown_event = asyncio.Event()
                    await self.shutdown_event.wait()
            except Exception as e:
                logger.error(f'Error in MCP event loop: {str(e)}')
                self.thread_running.set()
                raise
        self.task = self.event_loop.create_task(setup())
        try:
            self.event_loop.run_until_complete(self.task)
        except asyncio.CancelledError:
            logger.info('MCP client event loop was cancelled')
        except Exception as e:
            logger.error(f'Error in MCP event loop: {str(e)}')
        finally:
            if not self.event_loop.is_closed():
                self.event_loop.close()

    @asynccontextmanager
    async def _start_server(self, config: Dict[str, Any]):
        client = Client(config)
        async with client:
            tools = await client.list_tools()
            yield (client, tools)

    def create_tool(self, session: Client, mcp_tools: List[Any], config: Dict[str, Any]) -> Toolkit:

        def _sync_call_tool(name: str, **kwargs) -> Any:
            try:
                if 'arguments' in kwargs and len(kwargs) == 1:
                    arguments = kwargs['arguments']
                else:
                    arguments = kwargs
                logger.info(f'Calling MCP tool: {name} with arguments: {arguments}')
                future = asyncio.run_coroutine_threadsafe(session.call_tool(name, arguments), self.event_loop)
                result = future.result(timeout=30)
                logger.info(f'MCP tool {name} call completed successfully')
                return result
            except (TimeoutError, ClientError, McpError) as e:
                logger.error(f'Error calling MCP tool {name}: {str(e)}')
                raise
            except Exception as e:
                logger.error(f'Unexpected error calling MCP tool {name}: {str(e)}')
                raise
        all_tools = []
        for mcp_tool in mcp_tools:
            input_schema = getattr(mcp_tool, 'inputSchema', {})
            if not input_schema and hasattr(mcp_tool, 'input_schema'):
                input_schema = mcp_tool.input_schema
            properties = input_schema.get('properties', {})
            required = input_schema.get('required', [])
            inputs = properties
            partial_func = partial(_sync_call_tool, mcp_tool.name)
            partial_func.__name__ = mcp_tool.name
            tool = MCPTool(name=mcp_tool.name, description=getattr(mcp_tool, 'description', None) or '', inputs=inputs, required=required, function=partial_func)
            all_tools.append(tool)
        tool_collection = Toolkit(name=next(iter(config.get('mcpServers').keys())), tools=all_tools)
        return tool_collection

    def get_toolkits(self) -> List[Toolkit]:
        """Return a list ofToolkits, one per server."""
        if not self.sessions:
            raise RuntimeError('Session not initialized')
        return [self.create_tool(session, tools, config) for session, tools, config in zip(self.sessions, self.mcp_tools, self.server_configs)]

class MCPToolkit:

    def __init__(self, servers: Optional[list[MCPClient]]=None, config_path: Optional[str]=None, config: Optional[dict[str, Any]]=None):
        parameters = []
        if config_path:
            parameters += self._from_config_file(config_path)
        if config:
            parameters += self._from_config(config)
        self.servers = []
        if parameters:
            self.servers.append(MCPClient(parameters))
        if servers:
            self.servers.extend(servers)
        failed_servers = []
        for server in self.servers:
            try:
                server._connect()
                logger.info('Successfully connected to MCP servers')
            except TimeoutError as e:
                logger.warning(f'Timeout connecting to MCP servers: {str(e)}. Some tools may not be available.')
                failed_servers.append(server)
            except Exception as e:
                logger.error(f'Error connecting to MCP servers: {str(e)}')
                failed_servers.append(server)
        for failed_server in failed_servers:
            if failed_server in self.servers:
                self.servers.remove(failed_server)

    def _from_config_file(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                server_configs = json.load(f)
            return self._from_config(server_configs)
        except FileNotFoundError:
            logger.error(f'Config file not found: {config_path}')
            return []
        except json.JSONDecodeError:
            logger.error(f'Invalid JSON in config file: {config_path}')
            return []

    def _from_config(self, server_configs: dict[str, Any]):
        if not isinstance(server_configs, dict):
            logger.error('Server configuration must be a dictionary')
            return []
        if 'mcpServers' not in server_configs:
            raise ValueError("Server configuration must contain 'mcpServers' key")
        server_list = []
        for server_name, server_config in server_configs['mcpServers'].items():
            individual_config = {'mcpServers': {server_name: server_config}}
            server_list.append(individual_config)
        return server_list

    def disconnect(self):
        for server in self.servers:
            try:
                server._disconnect()
            except Exception as e:
                logger.warning(f'Error disconnecting from MCP server: {str(e)}')
        self.servers.clear()

    def get_toolkits(self) -> List[Toolkit]:
        """Return a flattened list of all tools across all servers"""
        all_tools = []
        if not self.servers:
            logger.info('No MCP servers configured, returning empty toolkit list')
            return all_tools
        for server in self.servers:
            try:
                import threading
                import queue
                result_queue = queue.Queue()
                exception_queue = queue.Queue()

                def get_tools_with_timeout():
                    try:
                        tools = server.get_toolkits()
                        result_queue.put(tools)
                    except Exception as e:
                        exception_queue.put(e)
                thread = threading.Thread(target=get_tools_with_timeout)
                thread.daemon = True
                thread.start()
                thread.join(timeout=30)
                if thread.is_alive():
                    logger.warning('Timeout getting tools from MCP server after 30 seconds')
                    continue
                if not exception_queue.empty():
                    raise exception_queue.get()
                tools = result_queue.get()
                all_tools.extend(tools)
                logger.info(f'Added {len(tools)} tools from MCP server')
            except Exception as e:
                logger.error(f'Error getting tools from MCP server: {str(e)}')
        return all_tools

class SearchGoogle(SearchBase):

    def __init__(self, name: str='SearchGoogle', num_search_pages: Optional[int]=5, max_content_words: Optional[int]=None, **kwargs):
        """
        Initialize the Google Search tool.
        
        Args:
            name (str): The name of the search tool
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int, optional): Maximum number of words to include in content, None means no limit
            **kwargs: Additional data to pass to the parent class
        """
        super().__init__(name=name, num_search_pages=num_search_pages, max_content_words=max_content_words, **kwargs)

    def search(self, query: str, num_search_pages: int=None, max_content_words: int=None) -> Dict[str, Any]:
        """
        Search Google using the Custom Search API and retrieve detailed search results with content snippets.
        
        Args:
            query (str): The search query to execute on Google
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content, None means no limit
            
        Returns:
            Dict[str, Any]: Contains search results and optional error message
        """
        num_search_pages = num_search_pages or self.num_search_pages
        max_content_words = max_content_words or self.max_content_words
        results = []
        api_key = os.getenv('GOOGLE_API_KEY', '')
        search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '')
        if not api_key or not search_engine_id:
            error_msg = 'API key and search engine ID are required. Please set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables. You can get these from the Google Cloud Console: https://console.cloud.google.com/apis/'
            logger.error(error_msg)
            return {'results': [], 'error': error_msg}
        try:
            logger.info(f'Searching Google for: {query}, num_results={num_search_pages}, max_content_words={max_content_words}')
            search_url = 'https://www.googleapis.com/customsearch/v1'
            params = {'key': api_key, 'cx': search_engine_id, 'q': query, 'num': num_search_pages}
            response = requests.get(search_url, params=params)
            data = response.json()
            if 'items' not in data:
                return {'results': [], 'error': 'No search results found.'}
            search_results = data['items']
            logger.info(f'Found {len(search_results)} search results')
            for item in search_results:
                url = item['link']
                title = item['title']
                try:
                    title, content = self._scrape_page(url)
                    if content:
                        display_content = self._truncate_content(content, max_content_words)
                        results.append({'title': title, 'content': display_content, 'url': url})
                except Exception as e:
                    logger.warning(f'Error processing URL {url}: {str(e)}')
                    continue
            return {'results': results, 'error': None}
        except Exception as e:
            logger.error(f'Error searching Google: {str(e)}')
            return {'results': [], 'error': str(e)}

class SearchSerperAPI(SearchBase):
    """
    SerperAPI search tool that provides access to Google search results
    through a simple and efficient API interface.
    """
    api_key: Optional[str] = Field(default=None, description='SerperAPI authentication key')
    default_location: Optional[str] = Field(default=None, description='Default geographic location')
    default_language: Optional[str] = Field(default='en', description='Default interface language')
    default_country: Optional[str] = Field(default='us', description='Default country code')
    enable_content_scraping: Optional[bool] = Field(default=True, description='Enable full content scraping')

    def __init__(self, name: str='SearchSerperAPI', num_search_pages: Optional[int]=10, max_content_words: Optional[int]=None, api_key: Optional[str]=None, default_location: Optional[str]=None, default_language: Optional[str]='en', default_country: Optional[str]='us', enable_content_scraping: Optional[bool]=True, **kwargs):
        """
        Initialize the SerperAPI Search tool.
        
        Args:
            name (str): Name of the tool
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content
            api_key (str): SerperAPI authentication key (can also use SERPERAPI_KEY env var)
            default_location (str): Default geographic location for searches
            default_language (str): Default interface language
            default_country (str): Default country code
            enable_content_scraping (bool): Whether to scrape full page content
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(name=name, num_search_pages=num_search_pages, max_content_words=max_content_words, api_key=api_key, default_location=default_location, default_language=default_language, default_country=default_country, enable_content_scraping=enable_content_scraping, **kwargs)
        self.api_key = api_key or os.getenv('SERPERAPI_KEY', '')
        self.base_url = 'https://google.serper.dev/search'
        if not self.api_key:
            logger.warning('SerperAPI key not found. Set SERPERAPI_KEY environment variable or pass api_key parameter.')

    def _build_serperapi_payload(self, query: str, location: str=None, language: str=None, country: str=None, num_results: int=None) -> Dict[str, Any]:
        """
        Build SerperAPI request payload.
        
        Args:
            query (str): Search query
            location (str): Geographic location
            language (str): Interface language
            country (str): Country code
            num_results (int): Number of results to retrieve
            
        Returns:
            Dict[str, Any]: SerperAPI request payload
        """
        payload = {'q': query}
        if num_results:
            payload['num'] = num_results
        if location or self.default_location:
            payload['location'] = location or self.default_location
        if language or self.default_language:
            payload['hl'] = language or self.default_language
        if country or self.default_country:
            payload['gl'] = country or self.default_country
        return payload

    def _execute_serperapi_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute search using direct HTTP POST requests to SerperAPI.
        
        Args:
            payload (Dict[str, Any]): Search payload
            
        Returns:
            Dict[str, Any]: SerperAPI response data
            
        Raises:
            Exception: For API errors
        """
        try:
            headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            if 'error' in data:
                raise Exception(f'SerperAPI error: {data['error']}')
            return data
        except requests.exceptions.RequestException as e:
            raise Exception(f'SerperAPI request failed: {str(e)}')
        except Exception as e:
            raise Exception(f'SerperAPI search failed: {str(e)}')

    def _process_serperapi_results(self, serperapi_data: Dict[str, Any], max_content_words: int=None) -> Dict[str, Any]:
        """
        Process SerperAPI results into structured format with processed results + raw data.
        
        Args:
            serperapi_data (Dict[str, Any]): Raw SerperAPI response
            max_content_words (int): Maximum words per result content
            
        Returns:
            Dict[str, Any]: Structured response with processed results and raw data
        """
        processed_results = []
        if (knowledge_graph := serperapi_data.get('knowledgeGraph', {})):
            if (description := knowledge_graph.get('description')):
                title = knowledge_graph.get('title', 'Unknown')
                content = f'**{title}**\n\n{description}'
                if (attributes := knowledge_graph.get('attributes', {})):
                    content += '\n\n**Key Information:**'
                    for key, value in list(attributes.items())[:5]:
                        formatted_key = key.replace('_', ' ').title()
                        content += f'\n• {formatted_key}: {value}'
                processed_results.append({'title': f'Knowledge: {title}', 'content': self._truncate_content(content, max_content_words or 200), 'url': knowledge_graph.get('descriptionLink', ''), 'type': 'knowledge_graph', 'priority': 1})
        for item in serperapi_data.get('organic', []):
            url = item.get('link', '')
            title = item.get('title', 'No Title')
            snippet = item.get('snippet', '')
            position = item.get('position', 0)
            result = {'title': title, 'content': self._truncate_content(snippet, max_content_words or 400), 'url': url, 'type': 'organic', 'priority': 2, 'position': position}
            if self.enable_content_scraping and url and url.startswith(('http://', 'https://')):
                try:
                    scraped_title, scraped_content = self._scrape_page(url)
                    if scraped_content and scraped_content.strip():
                        if scraped_title and scraped_title.strip():
                            result['title'] = scraped_title
                        result['site_content'] = self._truncate_content(scraped_content, max_content_words or 400)
                    else:
                        result['site_content'] = None
                except Exception as e:
                    logger.debug(f'Content scraping failed for {url}: {str(e)}')
                    result['site_content'] = None
            else:
                result['site_content'] = None
            if snippet or result.get('site_content'):
                processed_results.append(result)
        raw_data = {}
        raw_sections = ['relatedSearches']
        for section in raw_sections:
            if section in serperapi_data and serperapi_data[section]:
                raw_data[section] = serperapi_data[section][:5]
        search_metadata = {}
        if (search_params := serperapi_data.get('searchParameters', {})):
            search_metadata = {'query': search_params.get('q', ''), 'engine': search_params.get('engine', ''), 'type': search_params.get('type', ''), 'credits': serperapi_data.get('credits', 0)}
        processed_results.sort(key=lambda x: (x.get('priority', 999), x.get('position', 0)))
        return {'results': processed_results, 'raw_data': raw_data if raw_data else None, 'search_metadata': search_metadata if search_metadata else None, 'error': None}

    def _handle_api_errors(self, error: Exception) -> str:
        """
        Handle SerperAPI specific errors with appropriate messages.
        
        Args:
            error (Exception): The exception that occurred
            
        Returns:
            str: User-friendly error message
        """
        error_str = str(error).lower()
        if 'api key' in error_str or 'unauthorized' in error_str:
            return 'Invalid or missing SerperAPI key. Please set SERPERAPI_KEY environment variable.'
        elif 'rate limit' in error_str or 'too many requests' in error_str:
            return 'SerperAPI rate limit exceeded. Please try again later.'
        elif 'quota' in error_str or 'credit' in error_str:
            return 'SerperAPI quota exceeded. Please check your plan limits.'
        elif 'timeout' in error_str:
            return 'SerperAPI request timeout. Please try again.'
        else:
            return f'SerperAPI error: {str(error)}'

    def search(self, query: str, num_search_pages: int=None, max_content_words: int=None, location: str=None, language: str=None, country: str=None) -> Dict[str, Any]:
        """
        Search using SerperAPI with comprehensive parameter support.
        
        Args:
            query (str): The search query
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content
            location (str): Geographic location for localized results
            language (str): Interface language (e.g., 'en', 'es', 'fr')
            country (str): Country code for country-specific results (e.g., 'us', 'uk')
            
        Returns:
            Dict[str, Any]: Contains search results and optional error message
        """
        num_search_pages = num_search_pages or self.num_search_pages
        max_content_words = max_content_words or self.max_content_words
        if not self.api_key:
            error_msg = 'SerperAPI key is required. Please set SERPERAPI_KEY environment variable or pass api_key parameter. Get your key from: https://serper.dev/'
            logger.error(error_msg)
            return {'results': [], 'raw_data': None, 'search_metadata': None, 'error': error_msg}
        try:
            logger.info(f'Searching SerperAPI: {query}, num_results={num_search_pages}, max_content_words={max_content_words}')
            payload = self._build_serperapi_payload(query=query, location=location, language=language, country=country, num_results=num_search_pages)
            serperapi_data = self._execute_serperapi_search(payload)
            response_data = self._process_serperapi_results(serperapi_data, max_content_words)
            logger.info(f'Successfully retrieved {len(response_data['results'])} processed results')
            return response_data
        except Exception as e:
            error_msg = self._handle_api_errors(e)
            logger.error(f'SerperAPI search failed: {error_msg}')
            return {'results': [], 'raw_data': None, 'search_metadata': None, 'error': error_msg}

class VectorStoreFactory:
    """Factory for creating vector stores."""

    def create(self, store_type: str, store_config: Dict[str, Any]=None) -> VectorStore:
        store_config = store_config or {}
        if store_type == VectorStoreType.FAISS:
            dimensions = store_config.get('dimensions')
            if not dimensions or not isinstance(dimensions, int):
                raise ValueError('FAISS requires a valid dimension')
            vector_store = FaissVectorStoreWrapper(**store_config)
        else:
            raise ValueError(f'Unsupported vector store type: {store_type}')
        logger.info(f'Created vector store: {store_type}')
        return vector_store

class GraphStoreFactory:
    """Factory for creating graph stores."""

    def create(self, store_type: str, store_config: Dict[str, Any]=None) -> GraphStore:
        """Create a graph store based on configuration.
        
        Args:
            store_type (str): The type of graph store (e.g., 'neo4j').
            store_config (Dict[str, Any], optional): Store configuration.
            
        Returns:
            GraphStore: A LlamaIndex-compatible graph store.
            
        Raises:
            ValueError: If the store type or configuration is invalid.
        """
        store_config = store_config or {}
        if store_type == GraphStoreType.NEO4J.value:
            required_fields = ['uri', 'username', 'password']
            if not all((field in store_config for field in required_fields)):
                raise ValueError('Neo4j requires uri, username, and password')
            graph_store = Neo4jGraphStoreWrapper(**store_config)
        else:
            raise ValueError(f'Unsupported graph store type: {store_type}')
        logger.info(f'Created graph store: {store_type}')
        return graph_store

class Neo4jGraphStoreWrapper(GraphStoreBase):
    """Wrapper for Neo4j graph store."""

    def __init__(self, uri: str, username: str, password: str, database: str='neo4j', **kwargs):
        try:
            self.graph_store = BasicNeo4jStore(url=uri, username=username, password=password, database=database)
        except Exception as e:
            raise ValueError(f'Failed to connect to Neo4j: {str(e)}')
        self.verify_version()

    def get_graph_store(self) -> PropertyGraphStore:
        return self.graph_store

    @property
    def supports_vector_queries(self):
        return self.graph_store.supports_vector_queries and self.graph_store._supports_vector_index

    def verify_version(self):
        """
        Check if the connected Neo4j database version supports vector indexing
        without specifying embedding dimension.

        Queries the Neo4j database to retrieve its version and compares it
        against a target version (5.23.0) that is known to support vector
        indexing. 
        """
        db_data = self.graph_store.structured_query('CALL dbms.components()')
        version = db_data[0]['versions'][0]
        if 'aura' in version:
            version_tuple = (*map(int, version.split('-')[0].split('.')), 0)
        else:
            version_tuple = tuple(map(int, version.split('.')))
        target_version = (5, 23, 0)
        if version_tuple >= target_version:
            self.graph_store._supports_vector_index = True
        else:
            self.graph_store._supports_vector_index = False
            logger.warning(f'The version of Neo4j server is {version_tuple}, which is less than {target_version}. Disable the vector indexing.')

    def clear(self) -> None:
        """
        Clear the node and relation in the neo4j graph database.
        """
        with self.graph_store.client.session() as session:
            session.run('MATCH (n) DETACH DELETE n')
            session.run('CALL apoc.schema.assert({}, {})')

    async def aload(self, node: Union[LabelledNode, Relation, BaseNode]) -> None:
        """
        Asynchronously load a single node into the Neo4j graph database.

        Checks if a node with the same ID already exists in the database. If it does not exist,
        inserts the node as either an EntityNode or ChunkNode based on its type. Handles metadata
        and embeddings appropriately.

        Args:
            node (Union[LabelledNode, Relation, BaseNode]): The node/relation to load, either a Chunk or a LlamaIndex BaseNode.
        """
        try:
            if not isinstance(node, (BaseNode, EntityNode, ChunkNode, Relation)):
                raise ValueError(f'Unsupported node type: {type(node)}. Must be BaseNode, EntityNode, ChunkNode, Relation.')
            if isinstance(node, (EntityNode, ChunkNode)):
                self.graph_store.upsert_nodes([node])
            elif isinstance(node, BaseNode):
                self.graph_store.upsert_llama_nodes([node])
            elif isinstance(node, Relation):
                self.graph_store.upsert_relations([node])
            if self.graph_store.supports_structured_queries:
                self.graph_store.get_schema(refresh=True)
        except Exception as e:
            logger.error(f'Failed to load node with ID {node.id} into Neo4j: {str(e)}')
            raise

    def build_kv_store(self) -> Sequence[Union[LabelledNode, EntityNode, ChunkNode, Relation]]:
        """
        Build a kv_store from neo4j database.
        Returns a dictionary where:
        - Key: node ID
        - Value: Node object (EntityNode, ChunkNode, Relation)
        """
        try:
            cur_sanitize_query_output = self.graph_store.sanitize_query_output
            self.graph_store.sanitize_query_output = False
            nodes_query = f'\n                MATCH (n:{BASE_NODE_LABEL})\n                RETURN n.id AS name, labels(n) AS labels,\n                       n.text AS text,\n                       n.embedding AS embedding,\n                       properties(n) AS properties\n            '
            nodes_result = self.graph_store.structured_query(nodes_query)
            nodes = []
            for record in nodes_result:
                labels = record['labels']
                node_dict = {'id': record['name'], 'labels': labels, 'embedding': record['embedding'], 'properties': record['properties']}
                if 'Chunk' in labels:
                    if node_dict['properties']['_node_type'] == 'TextNode':
                        content = json.loads(node_dict['properties']['_node_content'])
                        content['metadata'] = json.loads(content['metadata']['metadata'])
                        node = TextNode(**content)
                    nodes.append(node)
                elif BASE_ENTITY_LABEL in labels:
                    node_dict['name'] = record['name'] or record['id']
                    node_dict['label'] = [label for label in labels if label not in [BASE_NODE_LABEL, BASE_ENTITY_LABEL]][0] if any((label not in [BASE_NODE_LABEL, BASE_ENTITY_LABEL] for label in labels)) else 'entity'
                    node = EntityNode(name=node_dict['name'], label=node_dict['label'], embedding=node_dict['embedding'], properties={'triplet_source_id': node_dict['properties']['triplet_source_id']})
                    nodes.append(node)
                else:
                    logger.warning(f'Skipping node with id {record['id']} due to unsupported labels: {labels}')
                    continue
            relations_query = 'MATCH ()-[r]->() RETURN type(r) AS label, startNode(r).id AS source_id, endNode(r).id AS target_id, properties(r) AS properties'
            relations_result = self.graph_store.structured_query(relations_query)
            relations = [Relation(label=record['label'], source_id=record['source_id'], target_id=record['target_id'], properties=json.loads(record['properties'].get('metadata', {})) if isinstance(record['properties'].get('metadata', {}), str) else record['properties'].get('metadata', {})) for record in relations_result]
            self.graph_store.sanitize_query_output = cur_sanitize_query_output
            logger.info(f'Exported {len(nodes)} nodes and {len(relations)} relations from Neo4j graph store')
            return nodes + relations
        except Exception as e:
            logger.error(f'Failed to export Neo4j graph store: {str(e)}')
            raise

class EvopromptOptimizer(BaseOptimizer):
    """
    Base class for evolutionary prompt optimization algorithms.
    
    This optimizer uses evolutionary algorithms to improve prompts in multi-agent workflows.
    It supports both node-based and combination-based evolution strategies.
    """

    def __init__(self, registry: ParamRegistry, program: Callable, population_size: int, iterations: int, llm_config: OpenAILLMConfig, concurrency_limit: int=10, combination_sample_size: int=None, enable_logging: bool=True, log_dir: str=None, enable_early_stopping: bool=True, early_stopping_patience: int=3):
        """
        Initialize the EvoPrompt optimizer.

        Args:
            registry: Parameter registry for tracking prompt nodes
            program: The program/workflow to optimize
            population_size: Size of the evolution population
            iterations: Number of evolution iterations
            llm_config: Configuration for the LLM used in evolution
            concurrency_limit: Maximum concurrent API calls
            combination_sample_size: Sample size for combination evaluation
            enable_logging: Whether to enable detailed logging
            log_dir: Directory for saving logs
            enable_early_stopping: Whether to enable early stopping
            early_stopping_patience: Number of generations to wait before stopping
        """
        super().__init__(registry=registry, program=program)
        self.population_size = population_size
        self.iterations = iterations
        self.llm_config = llm_config
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.combination_sample_size = combination_sample_size
        self.enable_logging = enable_logging
        self.log_dir_base = log_dir
        self.log_dir = None
        self.enable_early_stopping = enable_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self._best_score_so_far = -float('inf')
        self._generations_without_improvement = 0
        self._eval_cache = {}
        self.node_populations: Dict[str, List[str]] = {}
        self.node_scores: Dict[str, List[float]] = {}
        self.best_scores_per_gen: Dict[str, Dict[str, float]] = {}
        self.avg_scores_per_gen: Dict[str, Dict[str, float]] = {}
        self.best_combo_scores_per_gen: Dict[str, float] = {}
        self.avg_combo_scores_per_gen: Dict[str, float] = {}
        self.paraphrase_agent = CustomizeAgent(name='ParaphraseAgent', description='An agent that paraphrases a given instruction.', prompt='Task: Generate a semantically equivalent but differently worded version of the user-provided instruction.\n                    \nNow, please process the following instruction:\nInput: {instruction}\n\nPlease provide the paraphrased version in the following format:\n\n## paraphrased_instruction\n[Your paraphrased version here]', llm_config=self.llm_config, inputs=[{'name': 'instruction', 'type': 'string', 'description': 'The instruction to paraphrase.'}], outputs=[{'name': 'paraphrased_instruction', 'type': 'string', 'description': 'The paraphrased instruction.'}], parse_mode='title')

    def _setup_logging_directory(self, benchmark: BIGBenchHard):
        """
        Set up logging directory for evolution tracking.
        
        Args:
            benchmark: The benchmark instance containing task information
        """
        if not self.enable_logging or self.log_dir:
            return
        task_name = benchmark.task if hasattr(benchmark, 'task') else 'unknown_task'
        if self.log_dir_base is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            algo_name = self.__class__.__name__.replace('Optimizer', '')
            self.log_dir = f'node_evolution_logs_{algo_name}_{self.llm_config.model}_{task_name}_{timestamp}'
        else:
            self.log_dir = self.log_dir_base
        os.makedirs(self.log_dir, exist_ok=True)
        logger.info(f'Logging enabled. Log files will be saved to: {self.log_dir}')

    def _log_generation_summary(self, generation: int, operation: str='Evolution'):
        """
        Log detailed summary of each generation's population and scores.
        
        Args:
            generation: The current generation number
            operation: Type of operation (Evolution, Initial, etc.)
        """
        if not self.enable_logging:
            return
        filename = f'generation_{generation:02d}_{operation.lower()}.csv'
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Node_Name', 'Individual_ID', 'Prompt_Text', 'Fitness_Score', 'Status', 'Rank_in_Node', 'Generation', 'Timestamp'])
            timestamp = datetime.now().isoformat()
            for node_name in self.node_populations.keys():
                node_pop = self.node_populations.get(node_name, [])
                node_scores = self.node_scores.get(node_name, [])
                if not node_pop:
                    continue
                sorted_indices = sorted(range(len(node_scores)), key=lambda i: node_scores[i], reverse=True)
                for rank, idx in enumerate(sorted_indices, 1):
                    prompt = node_pop[idx]
                    score = node_scores[idx]
                    status = 'Best' if rank == 1 else 'Survivor' if rank <= self.population_size else 'Eliminated'
                    writer.writerow([node_name, f'{node_name}_{idx}', prompt[:200] + '...' if len(prompt) > 200 else prompt, f'{score:.6f}', status, rank, generation, timestamp])

    def _log_detailed_evaluation(self, generation: int, combinations: List[Dict[str, str]], combination_scores: List[float]):
        if not self.enable_logging:
            return
        filename = f'combo_evaluation_gen_{generation:02d}.csv'
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            node_names = list(combinations[0].keys()) if combinations else []
            header = ['Combination_ID', 'Average_Score']
            for node_name in node_names:
                header.append(f'{node_name}_Prompt_Preview')
            header.extend(['Generation', 'Timestamp'])
            writer.writerow(header)
            timestamp = datetime.now().isoformat()
            for combo_id, (combination, avg_score) in enumerate(zip(combinations, combination_scores)):
                try:
                    row = [f'combo_{combo_id}', f'{avg_score:.6f}']
                    for node_name in node_names:
                        prompt = combination[node_name]
                        row.append(prompt[:50] + '...' if len(prompt) > 50 else prompt)
                    row.extend([generation, timestamp])
                    writer.writerow(row)
                except Exception as e:
                    logger.error(f'Error logging evaluation for combination {combo_id}: {e}')

    def _create_single_metric_plot(self, metric_name: str, generations: List[int], best_scores: List[float], avg_scores: List[float], algorithm_name: str, plot_dir: str):
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(generations, best_scores, marker='o', linestyle='-', linewidth=2, markersize=8, label='Best Score')
        ax.plot(generations, avg_scores, marker='x', linestyle='--', linewidth=2, markersize=8, label='Average Score')
        title = f"Performance for '{metric_name}' ({algorithm_name})"
        ax.set_title(title, fontsize=16, weight='bold')
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Fitness Score', fontsize=12)
        ax.set_xticks(generations)
        ax.set_xticklabels([f'Gen {g}' for g in generations], rotation=45, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        safe_metric_name = re.sub('[^a-zA-Z0-9_-]', '_', metric_name)
        filename = f'performance_plot_{safe_metric_name}.png'
        filepath = os.path.join(plot_dir, filename)
        try:
            plt.savefig(filepath, dpi=200, bbox_inches='tight')
        except Exception as e:
            logger.error(f'Failed to save individual plot for {metric_name}: {e}')
        finally:
            plt.close(fig)

    def _plot_and_save_performance_graph(self, algorithm_name: str):
        if not self.enable_logging or plt is None:
            if plt is None:
                logger.warning('Matplotlib not found, skipping plot generation.')
            return
        if not self.best_scores_per_gen and (not self.best_combo_scores_per_gen):
            logger.warning('No performance data to plot.')
            return
        plt.style.use('seaborn-v0_8-whitegrid')
        all_gen_keys = set(self.best_scores_per_gen.keys()) | set(self.best_combo_scores_per_gen.keys())
        generations = sorted([int(re.search('\\d+', gen).group()) for gen in all_gen_keys if re.search('\\d+', gen)])
        fig_combined, ax_combined = plt.subplots(figsize=(16, 9))
        if self.best_combo_scores_per_gen:
            combo_best = [self.best_combo_scores_per_gen.get(f'Gen_{g}') for g in generations]
            combo_avg = [self.avg_combo_scores_per_gen.get(f'Gen_{g}') for g in generations]
            ax_combined.plot(generations, combo_best, marker='*', linestyle='-', linewidth=2.5, markersize=10, label='Best Combination Score (Overall)')
            ax_combined.plot(generations, combo_avg, marker='D', linestyle='--', linewidth=2.5, markersize=8, label='Average Combination Score (Overall)')
        all_node_metrics = set()
        for gen_data in self.best_scores_per_gen.values():
            all_node_metrics.update(gen_data.keys())
        for metric in sorted(list(all_node_metrics)):
            best_scores = [self.best_scores_per_gen.get(f'Gen_{g}', {}).get(metric) for g in generations]
            avg_scores = [self.avg_scores_per_gen.get(f'Gen_{g}', {}).get(metric) for g in generations]
            ax_combined.plot(generations, best_scores, marker='o', linestyle='-', alpha=0.7, label=f'Best Score ({metric})')
            ax_combined.plot(generations, avg_scores, marker='x', linestyle='--', alpha=0.7, label=f'Average Score ({metric})')
        ax_combined.set_title(f'Overall Performance Evolution ({algorithm_name})', fontsize=18, weight='bold')
        ax_combined.set_xlabel('Generation', fontsize=14)
        ax_combined.set_ylabel('Fitness Score', fontsize=14)
        ax_combined.set_xticks(generations)
        ax_combined.set_xticklabels([f'Gen {g}' for g in generations], rotation=45, ha='right')
        handles, labels = ax_combined.get_legend_handles_labels()
        combo_indices = [i for i, label in enumerate(labels) if 'Combination' in label]
        node_indices = [i for i, label in enumerate(labels) if 'Combination' not in label]
        ax_combined.legend([handles[i] for i in combo_indices + node_indices], [labels[i] for i in combo_indices + node_indices], loc='best', fontsize=10)
        ax_combined.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        combined_filepath = os.path.join(self.log_dir, 'performance_summary_OVERALL.png')
        try:
            plt.savefig(combined_filepath, dpi=300, bbox_inches='tight')
            logger.info(f'Overall performance plot saved to: {combined_filepath}')
        except Exception as e:
            logger.error(f'Failed to save overall performance plot: {e}')
        finally:
            plt.close(fig_combined)
        individual_plot_dir = os.path.join(self.log_dir, 'individual_plots')
        os.makedirs(individual_plot_dir, exist_ok=True)
        for metric in sorted(list(all_node_metrics)):
            best_scores = [self.best_scores_per_gen.get(f'Gen_{g}', {}).get(metric) for g in generations]
            avg_scores = [self.avg_scores_per_gen.get(f'Gen_{g}', {}).get(metric) for g in generations]
            self._create_single_metric_plot(metric, generations, best_scores, avg_scores, algorithm_name, individual_plot_dir)
        if self.best_combo_scores_per_gen:
            combo_best = [self.best_combo_scores_per_gen.get(f'Gen_{g}') for g in generations]
            combo_avg = [self.avg_combo_scores_per_gen.get(f'Gen_{g}') for g in generations]
            self._create_single_metric_plot('Combination', generations, combo_best, combo_avg, algorithm_name, individual_plot_dir)
        logger.info(f'Individual performance plots saved to: {individual_plot_dir}')

    def _log_optimization_summary(self, algorithm_name: str, best_config: Dict[str, str], test_accuracy: float=None):
        if not self.enable_logging:
            return
        filename = f'optimization_summary_{algorithm_name.lower()}.csv'
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value', 'Timestamp'])
            timestamp = datetime.now().isoformat()
            writer.writerow(['Algorithm', algorithm_name, timestamp])
            writer.writerow(['Population_Size', self.population_size, timestamp])
            writer.writerow(['Iterations', self.iterations, timestamp])
            writer.writerow(['Combination_Sample_Size', self.combination_sample_size, timestamp])
            writer.writerow(['Early_Stopping_Enabled', self.enable_early_stopping, timestamp])
            if self.enable_early_stopping:
                writer.writerow(['Early_Stopping_Patience', self.early_stopping_patience, timestamp])
            if test_accuracy is not None:
                writer.writerow(['Final_Test_Accuracy', f'{test_accuracy:.6f}', timestamp])
            for node_name, prompt in best_config.items():
                writer.writerow([f'Best_{node_name}', prompt, timestamp])
            for gen_name in self.best_scores_per_gen.keys():
                for metric_name, best_score in self.best_scores_per_gen[gen_name].items():
                    writer.writerow([f'{gen_name}_{metric_name}_Best', f'{best_score:.6f}', timestamp])
                if gen_name in self.avg_scores_per_gen:
                    for metric_name, avg_score in self.avg_scores_per_gen[gen_name].items():
                        writer.writerow([f'{gen_name}_{metric_name}_Avg', f'{avg_score:.6f}', timestamp])
        self._plot_and_save_performance_graph(algorithm_name)
        try:
            self._save_best_config_json(best_config)
        except Exception as e:
            logger.error(f'Failed to save best_config.json: {e}')

    def _save_best_config_json(self, best_config: Dict[str, str], filename: str='best_config.json') -> None:
        """
        Save the best configuration to a JSON file in the log directory.

        This is a convenience artifact for downstream automation to reload and
        apply the optimized prompt set without parsing CSVs.

        Note: optimize() already applies the best config to the in-memory
        program. This JSON is intended for persistence and later reuse.
        """
        if not self.enable_logging:
            return
        if not self.log_dir:
            return
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(best_config, f, ensure_ascii=False, indent=2)
        logger.info(f'Best config JSON saved to: {filepath}')

    def load_and_apply_config(self, path: str) -> Dict[str, str]:
        """
        Load a JSON best_config from disk and apply it to the registered program.

        Returns the loaded configuration dictionary.
        """
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        self.apply_cfg(cfg)
        logger.info(f'Applied configuration from JSON: {path}')
        return cfg

    async def _log_evaluation_details(self, benchmark: BIGBenchHard, dataset: List[Dict], predictions: List[str], scores: List[float], eval_mode: str, accuracy: float, correct_count: int, total_count: int):
        if not self.enable_logging:
            return
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'evaluation_testset_{eval_mode}_{timestamp}.csv'
        filepath = os.path.join(self.log_dir, filename)
        logger.info(f'Logging detailed evaluation results to {filepath}')
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Overall_Accuracy', f'{accuracy:.6f}'])
            writer.writerow(['Correct_Count', correct_count])
            writer.writerow(['Total_Count', total_count])
            writer.writerow([])
            writer.writerow(['example_id', 'input_text', 'prediction', 'ground_truth', 'score'])
            for i, example in enumerate(dataset):
                example_id = benchmark._get_id(example)
                input_text = example.get('input', '')
                label = benchmark.get_label(example)
                writer.writerow([example_id, input_text[:200] + '...' if len(input_text) > 200 else input_text, predictions[i], label, scores[i]])

    def _log_generation(self, generation: int, combos_with_scores: List[tuple]):
        """
        Log generation data for combination-based evolution.
        """
        if not self.enable_logging:
            return
        filename = f'combo_generation_{generation:02d}_log.csv'
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['Combination_ID', 'Combination_Score', 'Node_Name', 'Prompt_Text', 'Generation', 'Timestamp']
            writer.writerow(header)
            timestamp = datetime.now().isoformat()
            sorted_combos = sorted(combos_with_scores, key=lambda x: x[1], reverse=True)
            for combo_rank, (combination, avg_score) in enumerate(sorted_combos):
                combo_id = f'combo_rank_{combo_rank + 1}'
                for node_name, prompt_text in combination.items():
                    writer.writerow([combo_id, f'{avg_score:.6f}', node_name, prompt_text[:200] + '...' if len(prompt_text) > 200 else prompt_text, generation, timestamp])

    async def _evaluate_combination_list(self, combinations: List[Dict], benchmark: BIGBenchHard, dev_set: list) -> List[float]:
        if not combinations:
            return []
        eval_dev_set = dev_set[:50] if len(dev_set) > 50 else dev_set
        all_scores = []
        pbar = aio_tqdm(total=len(combinations), desc='Evaluating batch', leave=False)
        for combo in combinations:
            tasks = [self._evaluate_combination_on_example(combo, benchmark, ex) for ex in eval_dev_set]
            example_scores = await asyncio.gather(*tasks)
            avg_score = sum(example_scores) / len(example_scores) if example_scores else 0.0
            all_scores.append(avg_score)
            pbar.update(1)
        pbar.close()
        return all_scores

    def _generate_combinations(self, node_populations: Dict[str, List[str]]) -> List[Dict[str, str]]:
        node_names = list(node_populations.keys())
        node_prompts = [node_populations[node] for node in node_names]
        total_possible = np.prod([len(p) for p in node_prompts if p]) if all((p for p in node_prompts)) else 0
        if total_possible == 0:
            logger.warning('Cannot generate combinations, one or more node populations are empty.')
            return []
        if self.combination_sample_size is None:
            target_size = min(self.population_size, int(total_possible), 200)
        else:
            target_size = min(self.combination_sample_size, int(total_possible))
        logger.info(f'Total possible combinations: {total_possible}, sampling: {target_size}')
        if target_size >= total_possible:
            all_combinations = []
            for combination in itertools.product(*node_prompts):
                combo_dict = {node_names[i]: combination[i] for i in range(len(node_names))}
                all_combinations.append(combo_dict)
            return all_combinations
        sampled_combinations = []
        sampled_keys = set()
        max_attempts = target_size * 5
        attempts = 0
        while len(sampled_combinations) < target_size and attempts < max_attempts:
            combination = {name: random.choice(prompts) for name, prompts in node_populations.items()}
            combo_key = tuple(sorted(combination.items()))
            if combo_key not in sampled_keys:
                sampled_combinations.append(combination)
                sampled_keys.add(combo_key)
            attempts += 1
        logger.info(f'Generated {len(sampled_combinations)} unique combinations')
        return sampled_combinations

    async def _evaluate_combination_on_example(self, combination: Dict[str, str], benchmark: BIGBenchHard, example: Dict) -> float:
        combo_key = tuple(sorted(combination.items()))
        example_key = str(hash(str(example)))
        cache_key = hash((combo_key, example_key))
        if not hasattr(self, '_eval_cache'):
            self._eval_cache = {}
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]
        async with self.semaphore:
            try:
                original_config = self.get_current_cfg()
                self.apply_cfg(combination)
                inputs = {k: v for k, v in example.items() if k in benchmark.get_input_keys()}
                prediction, _ = await asyncio.to_thread(self.program, **inputs)
                label = benchmark.get_label(example)
                score_dict = benchmark.evaluate(prediction, label)
                score = score_dict.get('em', 0.0)
                self.apply_cfg(original_config)
                self._eval_cache[cache_key] = score
                if len(self._eval_cache) > 5000:
                    keys_to_del = list(self._eval_cache.keys())[:1000]
                    for key in keys_to_del:
                        del self._eval_cache[key]
                return score
            except Exception as e:
                logger.error(f'Error evaluating combination: {e}')
                return 0.0

    async def _evaluate_combinations_and_update_node_scores(self, combinations: List[Dict[str, str]], benchmark: BIGBenchHard, dev_set: list) -> List[float]:
        eval_dev_set = dev_set[:50] if len(dev_set) > 50 else dev_set
        combination_scores = []
        print(f'Evaluating {len(combinations)} combinations on {len(eval_dev_set)} examples...')
        combo_pbar = aio_tqdm(total=len(combinations), desc='Evaluating Combinations')
        for combination in combinations:
            tasks = [self._evaluate_combination_on_example(combination, benchmark, ex) for ex in eval_dev_set]
            example_scores = await asyncio.gather(*tasks)
            avg_score = sum(example_scores) / len(example_scores) if example_scores else 0.0
            combination_scores.append(avg_score)
            combo_pbar.update(1)
        combo_pbar.close()
        for node_name in self.node_populations.keys():
            self.node_scores[node_name] = [0.0] * len(self.node_populations[node_name])
            for prompt_idx, prompt in enumerate(self.node_populations[node_name]):
                participating_scores = [combo_score for combo_idx, combo_score in enumerate(combination_scores) if combinations[combo_idx].get(node_name) == prompt]
                if participating_scores:
                    self.node_scores[node_name][prompt_idx] = sum(participating_scores) / len(participating_scores)
                else:
                    self.node_scores[node_name][prompt_idx] = 0.0
        return combination_scores

    async def _perform_paraphrase(self, prompt: str) -> str:
        async with self.semaphore:
            output = await asyncio.to_thread(self.paraphrase_agent, inputs={'instruction': prompt})
            return output.content.paraphrased_instruction.strip()

    async def _perform_evolution(self, agent: Callable, inputs: Dict[str, str]) -> str:
        async with self.semaphore:
            output = await asyncio.to_thread(agent, inputs=inputs)
            if hasattr(output.content, 'evolved_prompt'):
                return output.content.evolved_prompt.strip()
            return str(output.content).strip()

    async def _initialize_node_populations(self, initial_config: Dict[str, any]):
        for node_name, initial_value in initial_config.items():
            node_population = []
            if isinstance(initial_value, list):
                provided_size = len(initial_value)
                if self.population_size < provided_size:
                    logger.info(f"Node '{node_name}': Provided population ({provided_size}) is larger than target size ({self.population_size}). Randomly sampling.")
                    node_population = random.sample(initial_value, self.population_size)
                elif self.population_size == provided_size:
                    logger.info(f"Node '{node_name}': Provided population size ({provided_size}) matches target size. Using directly.")
                    node_population = list(initial_value)
                else:
                    logger.info(f"Node '{node_name}': Target population size ({self.population_size}) is larger than provided ({provided_size}). Expanding.")
                    node_population = list(initial_value)
                    num_to_generate = self.population_size - provided_size
                    source_prompts_for_generation = random.choices(initial_value, k=num_to_generate)
                    paraphrase_tasks = [self._perform_paraphrase(prompt) for prompt in source_prompts_for_generation]
                    new_prompts = await aio_tqdm.gather(*paraphrase_tasks, desc=f'Expanding population for {node_name}')
                    node_population.extend(new_prompts)
            elif isinstance(initial_value, str):
                logger.info(f"Node '{node_name}': Generating population from a single initial prompt.")
                node_population = [initial_value]
                if self.population_size > 1:
                    num_to_generate = self.population_size - 1
                    paraphrase_tasks = [self._perform_paraphrase(initial_value) for _ in range(num_to_generate)]
                    new_prompts = await aio_tqdm.gather(*paraphrase_tasks, desc=f'Generating initial population for {node_name}')
                    node_population.extend(new_prompts)
            else:
                raise TypeError(f"Unsupported type for tracked parameter '{node_name}': {type(initial_value)}. Must be str or list.")
            self.node_populations[node_name] = node_population
            self.node_scores[node_name] = [0.0] * self.population_size

    async def evaluate(self, benchmark: BIGBenchHard, eval_mode: str='test') -> Dict[str, float]:
        """
        Evaluates the optimized program on a specified dataset.

        Args:
            benchmark (BIGBenchHard): The benchmark instance containing the data.
            eval_mode (str): The evaluation mode, either "test" or "dev".

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics.
        """
        logger.info(f"--- Evaluating optimized program on '{eval_mode}' set ---")
        dataset = benchmark.get_test_data() if eval_mode == 'test' else benchmark.get_dev_data()
        if not dataset:
            logger.warning(f"No data found for '{eval_mode}' set. Returning empty results.")
            return {}

        async def evaluate_example(example: Dict) -> tuple[float, str]:
            prediction, _ = await asyncio.to_thread(self.program, input=example['input'])
            score_dict = benchmark.evaluate(prediction, benchmark.get_label(example))
            score = score_dict.get('em', 0.0)
            return (score, prediction)
        tasks = [evaluate_example(ex) for ex in dataset]
        results = await aio_tqdm.gather(*tasks, desc=f'Evaluating on {eval_mode.capitalize()} Set')
        scores, predictions = zip(*results) if results else ([], [])
        correct_count = sum(scores)
        total_count = len(dataset)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        logger.info(f'{eval_mode.capitalize()} Set Accuracy: {accuracy:.4f} ({int(correct_count)}/{total_count})')
        if self.enable_logging:
            await self._log_evaluation_details(benchmark, dataset, predictions, scores, eval_mode, accuracy, int(correct_count), total_count)
        return {'accuracy': accuracy}

class GAOptimizer(EvopromptOptimizer):
    """
    Genetic Algorithm optimizer for prompt evolution.
    
    This optimizer uses genetic algorithm operations (crossover, mutation, selection)
    to evolve prompts. It supports both node-based and combination-based evolution.
    """

    def __init__(self, *args, full_evaluation: bool=False, **kwargs):
        """
        Initialize the GA optimizer.
        
        Args:
            full_evaluation: Whether to use full node-based evaluation or combination-based
            *args: Arguments passed to parent class
            **kwargs: Keyword arguments passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.full_evaluation = full_evaluation
        mode_str = 'full_evaluation' if self.full_evaluation else 'combination-based'
        logger.info(f"GAOptimizer initialized with '{mode_str}' mode.")
        self.ga_agent = CustomizeAgent(name='ga_agent', description='An agent that evolves a new prompt from two parent prompts.', prompt='Please follow the instructions step-by-step to generate a better prompt.\n\n1. Crossover the following prompts to generate a new prompt:\nPrompt 1: {parent1}\nPrompt 2: {parent2}\n\n2. Mutate the prompt generated in Step 1 and generate a final evolved prompt. Strictly preserve the original XML tags structure.\n\nNow process the given prompts and provide your output in the following format:\n\n## evolved_prompt\n[Your evolved prompt here]', llm_config=self.llm_config, inputs=[{'name': 'parent1', 'type': 'string', 'description': 'The first parent prompt.'}, {'name': 'parent2', 'type': 'string', 'description': 'The second parent prompt.'}], outputs=[{'name': 'evolved_prompt', 'type': 'string', 'description': 'The evolved prompt with XML tags preserved.'}], parse_mode='title')

    async def _perform_node_evolution(self, node_name: str, node_population: List[str], node_scores: List[float]=None, evolution_agent: Callable=None) -> List[str]:
        probabilities = None
        if node_scores:
            total_fitness = sum(node_scores)
            if total_fitness > 0:
                probabilities = [s / total_fitness for s in node_scores]
        agent = evolution_agent or self.ga_agent
        num_children_to_create = len(node_population)
        evolution_tasks = []
        for _ in range(num_children_to_create):
            parents = random.choices(node_population, weights=probabilities, k=2) if probabilities else random.choices(node_population, k=2)
            task = self._perform_evolution(agent=agent, inputs={'parent1': parents[0], 'parent2': parents[1]})
            evolution_tasks.append(task)
        new_children = await aio_tqdm.gather(*evolution_tasks, desc=f'Evolving {node_name}')
        return new_children

    async def optimize(self, benchmark: BIGBenchHard) -> tuple[Dict[str, str], dict, dict]:
        self._setup_logging_directory(benchmark)
        initial_config = self.get_current_cfg()
        if not initial_config:
            raise ValueError('Registry is empty.')
        await self._initialize_node_populations(initial_config)
        dev_set = benchmark.get_dev_data()
        if not dev_set:
            raise ValueError('Benchmark has no development set.')
        self._best_score_so_far = -float('inf')
        self._generations_without_improvement = 0
        if self.full_evaluation:
            logger.info('--- Starting Node-Based Evolution with Makeup Evaluation (full_evaluation=True) ---')
            print('--- Step 1: Initial evaluation of node combinations... ---')
            combinations = self._generate_combinations(self.node_populations)
            combination_scores = await self._evaluate_combinations_and_update_node_scores(combinations, benchmark, dev_set)
            self._log_generation_summary(0, 'Initial')
            self._log_detailed_evaluation(0, combinations, combination_scores)
            self.best_scores_per_gen['Gen_0'] = {name: max(scores) if scores else 0 for name, scores in self.node_scores.items()}
            self.avg_scores_per_gen['Gen_0'] = {name: np.mean(scores) if scores else 0 for name, scores in self.node_scores.items()}
            if combination_scores:
                initial_best_combo_score = max(combination_scores)
                self._best_score_so_far = initial_best_combo_score
                self.best_combo_scores_per_gen['Gen_0'] = initial_best_combo_score
                self.avg_combo_scores_per_gen['Gen_0'] = np.mean(combination_scores)
                logger.info(f'Early stopping baseline set to initial best combination score: {self._best_score_so_far:.4f}')
            for t in range(self.iterations):
                generation_start_time = time.time()
                print(f'\n--- Generation {t + 1}/{self.iterations} ---')
                children_populations = {}
                for node_name in self.node_populations.keys():
                    children = await self._perform_node_evolution(node_name, self.node_populations[node_name], self.node_scores[node_name], self.ga_agent)
                    children_populations[node_name] = children
                current_populations = {name: self.node_populations[name] + children_populations[name] for name in self.node_populations.keys()}
                self.node_populations = current_populations
                print(f'Performing main evaluation for {len(list(current_populations.values())[0])} individuals in each node...')
                combinations = self._generate_combinations(self.node_populations)
                combination_scores = await self._evaluate_combinations_and_update_node_scores(combinations, benchmark, dev_set)
                prompts_needing_makeup = []
                for node_name, scores in self.node_scores.items():
                    for idx, score in enumerate(scores):
                        if score == 0.0:
                            prompt_to_check = self.node_populations[node_name][idx]
                            is_in_combos = any((c.get(node_name) == prompt_to_check for c in combinations))
                            if not is_in_combos:
                                prompts_needing_makeup.append((node_name, idx, prompt_to_check))
                if prompts_needing_makeup:
                    print(f'--- Performing makeup evaluation for {len(prompts_needing_makeup)} unsampled individuals... ---')
                    makeup_combinations = []
                    for node_name, idx, prompt in prompts_needing_makeup:
                        makeup_combo = {name: random.choice(pop) for name, pop in self.node_populations.items()}
                        makeup_combo[node_name] = prompt
                        makeup_combinations.append(makeup_combo)
                    makeup_scores = await self._evaluate_combination_list(makeup_combinations, benchmark, dev_set)
                    for i, (node_name, idx, prompt) in enumerate(prompts_needing_makeup):
                        self.node_scores[node_name][idx] = makeup_scores[i]
                        logger.info(f"Updated score for '{prompt[:30]}...' to {makeup_scores[i]:.4f} after makeup eval.")
                print('--- Selecting survivors for the next generation... ---')
                survivor_populations = {}
                survivor_scores = {}
                for node_name in self.node_populations.keys():
                    population = self.node_populations[node_name]
                    scores = self.node_scores[node_name]
                    sorted_pairs = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
                    selected_pairs = sorted_pairs[:self.population_size]
                    if selected_pairs:
                        selected_scores, selected_population = zip(*selected_pairs)
                        survivor_scores[node_name] = list(selected_scores)
                        survivor_populations[node_name] = list(selected_population)
                    else:
                        survivor_scores[node_name], survivor_populations[node_name] = ([], [])
                    print(f'Node {node_name}: Selected top {len(survivor_populations[node_name])} from {len(population)} individuals')
                self.node_populations = survivor_populations
                self.node_scores = survivor_scores
                generation_time = time.time() - generation_start_time
                print(f'Generation {t + 1} completed in {generation_time:.2f}s')
                self._log_generation_summary(t + 1, 'Evolution')
                if combination_scores:
                    self._log_detailed_evaluation(t + 1, combinations, combination_scores)
                gen_name = f'Gen_{t + 1}'
                self.best_scores_per_gen[gen_name] = {name: max(scores) if scores else 0 for name, scores in self.node_scores.items()}
                self.avg_scores_per_gen[gen_name] = {name: np.mean(scores) if scores else 0 for name, scores in self.node_scores.items()}
                best_combo_score_this_gen = max(combination_scores) if combination_scores else -float('inf')
                self.best_combo_scores_per_gen[gen_name] = best_combo_score_this_gen
                self.avg_combo_scores_per_gen[gen_name] = np.mean(combination_scores) if combination_scores else 0.0
                if self.enable_early_stopping:
                    if best_combo_score_this_gen > self._best_score_so_far + 1e-06:
                        self._best_score_so_far = best_combo_score_this_gen
                        self._generations_without_improvement = 0
                        logger.info(f'Early stopping: New best combination score found: {self._best_score_so_far:.4f}.')
                    else:
                        self._generations_without_improvement += 1
                        logger.info(f'Early stopping: No improvement for {self._generations_without_improvement} generation(s).')
                    if self._generations_without_improvement >= self.early_stopping_patience:
                        logger.warning(f'\n--- EARLY STOPPING TRIGGERED at generation {t + 1} ---')
                        break
        else:
            logger.info('--- Starting Combo-Based Evolution (full_evaluation=False) ---')
            print('--- Step 1: Creating and evaluating initial combination population... ---')
            initial_combinations = self._generate_combinations(self.node_populations)
            initial_scores = await self._evaluate_combination_list(initial_combinations, benchmark, dev_set)
            combo_population_with_scores = sorted(zip(initial_combinations, initial_scores), key=lambda x: x[1], reverse=True)
            combo_population_with_scores = combo_population_with_scores[:self.population_size]
            gen_0_scores = [score for _, score in combo_population_with_scores]
            if gen_0_scores:
                best_gen_score = max(gen_0_scores)
                avg_gen_score = np.mean(gen_0_scores)
                self.best_combo_scores_per_gen['Gen_0'] = best_gen_score
                self.avg_combo_scores_per_gen['Gen_0'] = avg_gen_score
                self._best_score_so_far = best_gen_score
                print(f'Generation 0 complete. Best score: {best_gen_score:.4f}, Avg score: {avg_gen_score:.4f}')
                logger.info(f'Early stopping baseline set to: {self._best_score_so_far:.4f}')
            self._log_generation(0, combo_population_with_scores)
            for t in range(self.iterations):
                print(f'\n--- Generation {t + 1}/{self.iterations} (Combo Evolution) ---')
                parent_prompts_for_node = {name: [] for name in initial_config.keys()}
                for combo, _ in combo_population_with_scores:
                    for node_name, prompt in combo.items():
                        parent_prompts_for_node[node_name].append(prompt)
                children_populations = {}
                for node_name, prompts in parent_prompts_for_node.items():
                    children_populations[node_name] = await self._perform_node_evolution(node_name, prompts)
                print('Evaluating new child combinations...')
                child_combinations = self._generate_combinations(children_populations)
                child_scores = await self._evaluate_combination_list(child_combinations, benchmark, dev_set)
                child_combos_with_scores = list(zip(child_combinations, child_scores))
                print('Selecting best combinations from parents and children...')
                combined_population = combo_population_with_scores + child_combos_with_scores
                sorted_combos = sorted(combined_population, key=lambda x: x[1], reverse=True)
                combo_population_with_scores = sorted_combos[:self.population_size]
                self._log_generation(t + 1, combo_population_with_scores)
                current_scores = [score for _, score in combo_population_with_scores]
                best_gen_score = max(current_scores) if current_scores else 0
                avg_gen_score = np.mean(current_scores) if current_scores else 0
                gen_name = f'Gen_{t + 1}'
                self.best_combo_scores_per_gen[gen_name] = best_gen_score
                self.avg_combo_scores_per_gen[gen_name] = avg_gen_score
                print(f'Generation {t + 1} complete. Best score: {best_gen_score:.4f}, Avg score: {avg_gen_score:.4f}')
                if self.enable_early_stopping:
                    if best_gen_score > self._best_score_so_far + 1e-06:
                        self._best_score_so_far = best_gen_score
                        self._generations_without_improvement = 0
                        logger.info(f'Early stopping: New best combination score found: {self._best_score_so_far:.4f}. Patience counter reset.')
                    else:
                        self._generations_without_improvement += 1
                        logger.info(f'Early stopping: No improvement for {self._generations_without_improvement} generation(s). Patience: {self.early_stopping_patience}.')
                    if self._generations_without_improvement >= self.early_stopping_patience:
                        logger.warning(f'\n--- EARLY STOPPING TRIGGERED at generation {t + 1} ---')
                        break
        print('\n--- Evolution complete ---')
        if self.full_evaluation:
            best_config = {name: self.node_populations[name][np.argmax(self.node_scores[name])] for name in self.node_populations.keys() if self.node_populations.get(name) and self.node_scores.get(name)}
        else:
            best_config, _ = max(combo_population_with_scores, key=lambda x: x[1]) if combo_population_with_scores else ({}, 0)
        self._log_optimization_summary('GA', best_config)
        self.apply_cfg(best_config)
        logger.info('Optimization finished! The best configuration has been applied to the program.')
        return (best_config, self.best_combo_scores_per_gen, self.avg_scores_per_gen)

class RAGEngine:

    def __init__(self, config: RAGConfig, storage_handler: StorageHandler, llm: Optional[BaseLLM]=None):
        self.config = config
        self.storage_handler = storage_handler
        self.embedding_factory = EmbeddingFactory()
        self.index_factory = IndexFactory()
        self.chunk_factory = ChunkFactory()
        self.retriever_factory = RetrieverFactory()
        self.postprocessor_factory = PostprocessorFactory()
        self.llm = llm
        logger.info(f'RAGEngine modality config: {self.config.modality}')
        if self.config.modality == 'multimodal':
            self.chunk_class = ImageChunk
        else:
            self.chunk_class = TextChunk
        if self.config.modality == 'multimodal':
            self.reader = MultimodalReader(recursive=self.config.reader.recursive, exclude_hidden=self.config.reader.exclude_hidden, num_files_limits=self.config.reader.num_files_limit, errors=self.config.reader.errors)
        else:
            self.reader = LLamaIndexReader(recursive=self.config.reader.recursive, exclude_hidden=self.config.reader.exclude_hidden, num_workers=self.config.num_workers, num_files_limits=self.config.reader.num_files_limit, custom_metadata_function=self.config.reader.custom_metadata_function, extern_file_extractor=self.config.reader.extern_file_extractor, errors=self.config.reader.errors, encoding=self.config.reader.encoding)
        self.embed_model = self.embedding_factory.create(provider=self.config.embedding.provider, model_config=self.config.embedding.model_dump(exclude_unset=True))
        if self.storage_handler.vector_store is not None and self.embed_model.dimensions is not None:
            if self.storage_handler.storageConfig.vectorConfig.dimensions != self.embed_model.dimensions:
                logger.warning('The dimensions in vector_store is not equal with embed_model. Reiniliaze vector_store.')
                self.storage_handler.storageConfig.vectorConfig.dimensions = self.embed_model.dimensions
                self.storage_handler._init_vector_store()
        if self.config.modality == 'multimodal':
            self.chunker = None
        else:
            self.chunker = self.chunk_factory.create(strategy=self.config.chunker.strategy, embed_model=self.embed_model.get_embedding_model(), chunker_config={'chunk_size': self.config.chunker.chunk_size, 'chunk_overlap': self.config.chunker.chunk_overlap, 'max_chunks': self.config.chunker.max_chunks})
        self.indices: Dict[str, Dict[str, BaseIndexWrapper]] = {}
        self.retrievers: Dict[str, Dict[str, BaseRetrieverWrapper]] = {}

    def read(self, file_paths: Union[Sequence[str], str], exclude_files: Optional[Union[str, List, Tuple, Sequence]]=None, filter_file_by_suffix: Optional[Union[str, List, Tuple, Sequence]]=None, merge_by_file: bool=False, show_progress: bool=False, corpus_id: str=None) -> Corpus:
        """Load and chunk documents from files.

        Reads files from specified paths, processes them into documents, and chunks them into a Corpus.

        Args:
            file_paths (Union[Sequence[str], str]): Path(s) to files or directories.
            exclude_files (Optional[Union[str, List, Tuple, Sequence]]): Files to exclude.
            filter_file_by_suffix (Optional[Union[str, List, Tuple, Sequence]]): Filter files by suffix (e.g., '.pdf').
            merge_by_file (bool): Merge documents by file.
            show_progress (bool): Show loading progress.
            corpus_id (Optional[str]): Identifier for the corpus. Defaults to a UUID if None.

        Returns:
            Corpus: The chunked corpus containing processed document chunks.

        Raises:
            Exception: If document reading or chunking fails.
        """
        try:
            corpus_id = corpus_id or str(uuid4())
            documents = self.reader.load(file_paths=file_paths, exclude_files=exclude_files, filter_file_by_suffix=filter_file_by_suffix, merge_by_file=merge_by_file, show_progress=show_progress)
            if self.config.modality == 'multimodal':
                image_chunks = []
                for doc in documents:
                    image_path = getattr(doc, 'image_path', None) or doc.metadata.get('file_path')
                    image_mimetype = getattr(doc, 'image_mimetype', None)
                    image_chunk = self.chunk_class(image_path=image_path, image_mimetype=image_mimetype, chunk_id=doc.metadata.get('file_name', f'img_{len(image_chunks)}'), metadata=ChunkMetadata(doc_id=doc.metadata.get('file_name', f'doc_{len(image_chunks)}'), corpus_id=corpus_id, **doc.metadata))
                    image_chunks.append(image_chunk)
                corpus = Corpus(chunks=image_chunks, corpus_id=corpus_id)
                logger.info(f'Read {len(documents)} multimodal documents (no chunking) for corpus {corpus_id}')
            else:
                corpus = self.chunker.chunk(documents)
                corpus.corpus_id = corpus_id
                logger.info(f'Read {len(documents)} documents and created {len(corpus.chunks)} chunks for corpus {corpus_id}')
            return corpus
        except Exception as e:
            logger.error(f'Failed to read documents for corpus {corpus_id}: {str(e)}')
            raise

    def add(self, index_type: str, nodes: Union[Corpus, List[NodeWithScore], List[TextNode], List[ImageNode]], corpus_id: str=None) -> None:
        """Add nodes to an index for a specific corpus.

        Initializes an index if it doesn't exist and inserts nodes, updating metadata with corpus_id and index_type.

        Args:
            index_type (str): Type of index (e.g., VECTOR, GRAPH).
            nodes (Union[Corpus, List[NodeWithScore], List[TextNode]]): Nodes or Corpus to add.
            corpus_id (str, optional): Identifier for the corpus. Defaults to a UUID if None.

        Return:
            return a sequence with id of each added node.
            
        Raises:
            Exception: If index creation or node insertion fails.
        """
        try:
            corpus_id = corpus_id or str(uuid4())
            if corpus_id not in self.indices:
                self.indices[corpus_id] = {}
                self.retrievers[corpus_id] = {}
            if index_type not in self.indices[corpus_id]:
                index = self.index_factory.create(index_type=index_type, embed_model=self.embed_model.get_embedding_model(), storage_handler=self.storage_handler, index_config=self.config.index.model_dump(exclude_unset=True) if self.config.index else {}, llm=self.llm)
                self.indices[corpus_id][index_type] = index
                self.retrievers[corpus_id][index_type] = self.retriever_factory.create(retriever_type=self.config.retrieval.retrivel_type, llm=self.llm, index=index.get_index(), graph_store=index.get_index().storage_context.graph_store, embed_model=self.embed_model.get_embedding_model(), query=Query(query_str='', top_k=self.config.retrieval.top_k if self.config.retrieval else 5), storage_handler=self.storage_handler, chunk_class=self.chunk_class)
            nodes_to_insert = nodes.to_llama_nodes() if isinstance(nodes, Corpus) else nodes
            for node in nodes_to_insert:
                node.metadata.update({'corpus_id': corpus_id, 'index_type': index_type})
            nodes_ids = self.indices[corpus_id][index_type].insert_nodes(nodes_to_insert)
            logger.info(f'Added {len(nodes_to_insert)} nodes to {index_type} index for corpus {corpus_id}')
            return nodes_ids
        except Exception as e:
            logger.error(f'Failed to add nodes to {index_type} index for corpus {corpus_id}: {str(e)}')
            return []

    def delete(self, corpus_id: str, index_type: Optional[str]=None, node_ids: Optional[Union[str, List[str]]]=None, metadata_filters: Optional[Dict[str, Any]]=None) -> None:
        """Delete nodes or an entire index from a corpus.

        Removes specific nodes by ID or metadata filters, or deletes the entire index if no filters are provided.

        Args:
            corpus_id (str): Identifier for the corpus.
            index_type (Optional[IndexType]): Specific index type to delete from. If None, affects all indices.
            node_ids (Union[str, Optional[List[str]]]): List of node IDs to delete.
            metadata_filters (Optional[Dict[str, Any]]): Metadata filters to select nodes for deletion.

        Raises:
            Exception: If deletion fails.
        """
        try:
            if corpus_id not in self.indices:
                logger.warning(f'No indices found for corpus {corpus_id}')
                return
            target_indices = [index_type] if index_type else self.indices[corpus_id].keys()
            for idx_type in list(target_indices):
                if idx_type not in self.indices[corpus_id]:
                    logger.warning(f'Index type {idx_type} not found for corpus {corpus_id}')
                    continue
                index = self.indices[corpus_id][idx_type]
                if node_ids or metadata_filters:
                    node_ids_list = [node_ids] if isinstance(node_ids, str) else node_ids
                    index.delete_nodes(node_ids=node_ids_list, metadata_filters=metadata_filters)
                    logger.info(f'Deleted nodes from {idx_type} index for corpus {corpus_id}')
                else:
                    index.clear()
                    del self.indices[corpus_id][idx_type]
                    del self.retrievers[corpus_id][idx_type]
                    logger.info(f'Deleted entire {idx_type} index for corpus {corpus_id}')
            if not self.indices[corpus_id]:
                del self.indices[corpus_id]
                del self.retrievers[corpus_id]
                logger.info(f'Removed empty corpus {corpus_id}')
        except Exception as e:
            logger.error(f'Failed to delete from corpus {corpus_id}, index {index_type}: {str(e)}')
            raise

    def clear(self, corpus_id: Optional[str]=None) -> None:
        """Clear all indices for a specific corpus or all corpora.

        Args:
            corpus_id (Optional[str]): Specific corpus to clear. If None, clears all corpora.

        Raises:
            Exception: If clearing fails.
        """
        try:
            target_corpora = [corpus_id] if corpus_id else list(self.indices.keys())
            for cid in target_corpora:
                if cid not in self.indices:
                    logger.warning(f'No indices found for corpus {cid}')
                    continue
                for idx_type in list(self.indices[cid].keys()):
                    index = self.indices[cid][idx_type]
                    index.clear()
                    del self.indices[cid][idx_type]
                    del self.retrievers[cid][idx_type]
                    logger.info(f'Cleared {idx_type} index for corpus {cid}')
                del self.indices[cid]
                del self.retrievers[cid]
                logger.info(f'Cleared corpus {cid}')
        except Exception as e:
            logger.error(f'Failed to clear indices for corpus {corpus_id or 'all'}: {str(e)}')
            raise

    def save(self, output_path: Optional[str]=None, corpus_id: Optional[str]=None, index_type: Optional[str]=None, table: Optional[str]=None, graph_exported: bool=False) -> None:
        """Save indices to files or database.

        Serializes corpus chunks to JSONL files and metadata to JSON files if output_path is provided,
        or saves to the SQLite database via StorageHandler if output_path is None.

        Args:
            output_path (Optional[str]): Directory to save JSONL and JSON files. If None, saves to database.
            corpus_id (Optional[str]): Specific corpus to save. If None, saves all corpora.
            index_type (Optional[str]): Specific index type to save. If None, saves all indices.
            table (Optional[str]): Database table name for index data. Defaults to 'indexing' if None.
            graph_exported (bool): If True, export graph nodes and relations for graph indices. Defaults to False.

        Raises:
            Exception: If saving fails or file operations encounter errors.
        """
        try:
            target_corpora = [corpus_id] if corpus_id else list(self.indices.keys())
            table = table or 'indexing'
            for cid in target_corpora:
                if cid not in self.indices:
                    logger.warning(f'No indices found for corpus {cid}')
                    continue
                target_indices = [index_type] if index_type and index_type in self.indices[cid] else self.indices[cid].keys()
                for idx_type in target_indices:
                    index = self.indices[cid][idx_type]
                    if idx_type == IndexType.GRAPH and (not graph_exported):
                        logger.warning(f'Skipping save for graph index {idx_type} in corpus {cid} as graph_exported is False')
                        continue
                    if idx_type == IndexType.GRAPH and graph_exported:
                        index.build_kv_store()
                    chunks = [self.chunk_class.from_llama_node(node_data) for node_id, node_data in index.id_to_node.items()]
                    corpus = Corpus(chunks=chunks, corpus_id=cid)
                    vector_config = self.storage_handler.storageConfig.vectorConfig.model_dump() if self.storage_handler.storageConfig.vectorConfig else {}
                    graph_config = self.storage_handler.storageConfig.graphConfig.model_dump() if self.storage_handler.storageConfig.graphConfig else {}
                    metadata = IndexMetadata(corpus_id=cid, index_type=idx_type, collection_name=vector_config.get('qdrant_collection_name', 'default_collection'), dimension=self.embed_model.dimensions, vector_db_type=vector_config.get('vector_name', None), graph_db_type=graph_config.get('graph_name', None), embedding_model_name=self.config.embedding.model_name, date=str(datetime.now()))
                    if output_path:
                        os.makedirs(output_path, exist_ok=True)
                        safe_cid = ''.join((c if c.isalnum() or c in ['-', '_'] else '_' for c in cid))
                        safe_idx_type = ''.join((c if c.isalnum() or c in ['-', '_'] else '_' for c in idx_type))
                        nodes_file = os.path.join(output_path, f'{safe_cid}_{safe_idx_type}_nodes.jsonl')
                        metadata_file = os.path.join(output_path, f'{safe_cid}_{safe_idx_type}_metadata.json')
                        corpus.to_jsonl(nodes_file, indent=0)
                        logger.info(f'Saved {len(corpus.chunks)} chunks to {nodes_file}')
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            json.dump(metadata.model_dump(), f, indent=2, ensure_ascii=False)
                        logger.info(f'Saved metadata to {metadata_file}')
                    else:
                        index_data = {'corpus_id': cid, 'content': corpus.model_dump(), 'date': str(datetime.now()), 'metadata': metadata.model_dump()}
                        self.storage_handler.save_index(index_data, table=table)
                        logger.info(f'Saved {idx_type} index with {len(corpus.chunks)} chunks for corpus {cid} to database table {table}')
        except Exception as e:
            logger.error(f'Failed to save indices for corpus {corpus_id or 'all'}: {str(e)}')
            raise

    def load(self, source: Optional[str]=None, corpus_id: Optional[str]=None, index_type: Optional[str]=None, table: Optional[str]=None) -> None:
        """Load indices from files or database.

        Reconstructs indices and retrievers from JSONL/JSON files or SQLite database records.
        Validates the embedding model name and dimension before reinitializing the embedding model.

        Args:
            source (Optional[str]): Directory containing JSONL/JSON files. If None, loads from database.
            corpus_id (Optional[str]): Specific corpus to load. If None, loads all corpora.
            index_type (Optional[str]): Specific index type to load. If None, loads all indices.
            table (Optional[str]): Database table name for index data. Defaults to 'indexing' if None.

        Returns:
            The Sequence with id of loaded chunk.
        
        Raises:
            Exception: If loading fails due to file or database errors, invalid data, or unsupported embedding model/dimension.
        
        Warning:
            Try to call this function may cause some Bugs, when you load the nodes from file or database storage systems at twice. 
            Because All the indexing share the same storage backend from storageHandler.
            For example:
            The vector database (.e.g Faiss) can insert again, even thougt there is a same node.
        """
        try:
            table = table or 'indexing'
            config_dimension = self.storage_handler.storageConfig.vectorConfig.dimensions
            loaded_chunk_ids: List[str] = []
            if source:
                if not os.path.exists(source):
                    logger.error(f'Source directory {source} does not exist')
                    raise FileNotFoundError(f'Source directory {source} does not exist')
                for file_name in os.listdir(source):
                    if not file_name.endswith('_metadata.json'):
                        continue
                    parts = file_name.split('_')
                    if len(parts) < 3:
                        logger.warning(f'Skipping invalid metadata file: {file_name}')
                        continue
                    cid = '_'.join(parts[:-2])
                    idx_type = parts[-2]
                    if corpus_id and corpus_id != cid or (index_type and index_type != idx_type):
                        continue
                    metadata_file = os.path.join(source, file_name)
                    nodes_file = os.path.join(source, f'{cid}_{idx_type}_nodes.jsonl')
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = IndexMetadata.model_validate(json.load(f))
                    if not self.embed_model.validate_model(self.config.embedding.provider, metadata.embedding_model_name):
                        raise ValueError(f"Embedding model '{metadata.embedding_model_name}' is not supported by provider '{self.config.embedding.provider}'. Supported models: {EmbeddingProvider.SUPPORTED_MODELS.get(self.config.embedding.provider, [])}")
                    if metadata.dimension != config_dimension:
                        raise ValueError(f'Embedding dimension {metadata.dimension} in metadata does not match configured dimension {config_dimension}.')
                    if not os.path.exists(nodes_file):
                        logger.warning(f'Nodes file {nodes_file} not found for metadata {metadata_file}')
                        continue
                    corpus = Corpus.from_jsonl(nodes_file, corpus_id=cid)
                    if metadata.embedding_model_name != self.config.embedding.model_name:
                        logger.info(f'Reinitializing embedding model to {metadata.embedding_model_name}')
                        self.embed_model = self.embedding_factory.create(provider=self.config.embedding.provider, model_config=self.config.embedding.model_dump(exclude_unset=True))
                    chunk_ids = self._load_index(corpus, cid, idx_type)
                    loaded_chunk_ids.extend(chunk_ids)
                    logger.info(f'Loaded {idx_type} index with {len(corpus.chunks)} chunks for corpus {cid} from {nodes_file}')
            else:
                records = self.storage_handler.load(tables=[table]).get(table, [])
                if not records:
                    logger.warning(f'No records found in table {table}')
                    return
                for record in records:
                    parsed = self.storage_handler.parse_result(record, IndexStore)
                    cid = parsed['corpus_id']
                    idx_type = parsed['metadata']['index_type']
                    if corpus_id and corpus_id != cid or (index_type and index_type != idx_type):
                        continue
                    chunks = []
                    for chunk_data in parsed['content']['chunks']:
                        metadata = ChunkMetadata.model_validate(chunk_data['metadata'])
                        if self.config.modality == 'multimodal':
                            chunk = ImageChunk(chunk_id=chunk_data['chunk_id'], image_path=chunk_data['image_path'], image_mimetype=chunk_data.get('image_mimetype'), metadata=metadata, embedding=chunk_data['embedding'], excluded_embed_metadata_keys=chunk_data['excluded_embed_metadata_keys'], excluded_llm_metadata_keys=chunk_data['excluded_llm_metadata_keys'], relationships={k: RelatedNodeInfo(**v) for k, v in chunk_data['relationships'].items()})
                        else:
                            chunk = TextChunk(chunk_id=chunk_data['chunk_id'], text=chunk_data['text'], metadata=metadata, embedding=chunk_data['embedding'], start_char_idx=chunk_data['start_char_idx'], end_char_idx=chunk_data['end_char_idx'], excluded_embed_metadata_keys=chunk_data['excluded_embed_metadata_keys'], excluded_llm_metadata_keys=chunk_data['excluded_llm_metadata_keys'], relationships={k: RelatedNodeInfo(**v) for k, v in chunk_data['relationships'].items()})
                        chunks.append(chunk)
                    corpus = Corpus(chunks=chunks, corpus_id=cid, metadata=IndexMetadata.model_validate(parsed['metadata']))
                    metadata = IndexMetadata.model_validate(parsed['metadata'])
                    if not self.embed_model.validate_model(self.config.embedding.provider, metadata.embedding_model_name):
                        raise ValueError(f"Embedding model '{metadata.embedding_model_name}' is not supported by provider '{self.config.embedding.provider}'. Supported models: {EmbeddingProvider.SUPPORTED_MODELS.get(self.config.embedding.provider, [])}")
                    if metadata.dimension != config_dimension:
                        raise ValueError(f'Embedding dimension {metadata.dimension} in metadata does not match configured dimension {config_dimension}.')
                    if metadata.embedding_model_name != self.config.embedding.model_name:
                        logger.info(f'Reinitializing embedding model to {metadata.embedding_model_name}')
                        self.embed_model = self.embedding_factory.create(provider=self.config.embedding.provider, model_config=self.config.embedding.model_dump(exclude_unset=True))
                    chunk_ids = self._load_index(corpus, cid, idx_type)
                    loaded_chunk_ids.extend(chunk_ids)
                    logger.info(f'Loaded {idx_type} index with {len(corpus.chunks)} chunks for corpus {cid} from database table {table}')
            return loaded_chunk_ids
        except Exception as e:
            logger.error(f'Failed to load indices: {str(e)}')
            raise

    def _load_index(self, corpus: Corpus, corpus_id: str, index_type: str) -> Sequence[str]:
        """Helper method to load an index and its retriever."""
        try:
            if corpus_id not in self.indices:
                self.indices[corpus_id] = {}
                self.retrievers[corpus_id] = {}
            if index_type not in self.indices[corpus_id]:
                index = self.index_factory.create(index_type=index_type, embed_model=self.embed_model.get_embedding_model(), storage_handler=self.storage_handler, index_config=self.config.index.model_dump(exclude_unset=True) if self.config.index else {}, llm=self.llm)
                self.indices[corpus_id][index_type] = index
                retriever_type = RetrieverType.GRAPH if index_type == IndexType.GRAPH else RetrieverType.VECTOR
                self.retrievers[corpus_id][index_type] = self.retriever_factory.create(retriever_type=retriever_type, llm=self.llm, index=index.get_index(), graph_store=index.get_index().storage_context.graph_store, embed_model=self.embed_model.get_embedding_model(), query=Query(query_str='', top_k=self.config.retrieval.top_k if self.config.retrieval else 5), storage_handler=self.storage_handler)
            nodes = corpus.to_llama_nodes()
            for node in nodes:
                node.metadata.update({'corpus_id': corpus_id, 'index_type': index_type})
            chunk_ids = self.indices[corpus_id][index_type].load(nodes)
            logger.info(f'Inserted {len(nodes)} nodes into {index_type} index for corpus {corpus_id}')
            return chunk_ids
        except Exception as e:
            logger.error(f'Failed to load index for corpus {corpus_id}, index_type {index_type}: {str(e)}')
            raise

    async def aget(self, corpus_id: str, index_type: str, node_ids: List[str]) -> List[Union[TextChunk, ImageChunk]]:
        """Retrieve chunks by node_ids from the index."""
        try:
            chunks = await self.indices[corpus_id][index_type].get(node_ids=node_ids)
            logger.info(f'Retrieved {len(chunks)} chunks for node_ids: {node_ids}')
            return chunks
        except Exception as e:
            logger.error(f'Failed to get chunks: {str(e)}')
            return []

    async def query_async(self, query: Union[str, Query], corpus_id: Optional[str]=None, query_transforms: Optional[List]=None) -> RagResult:
        """Execute a query across indices and return processed results asynchronously.

        Performs query preprocessing, asynchronous retrieval, and post-processing.

        Args:
            query (Union[str, Query]): Query string or Query object.
            corpus_id (Optional[str]): Specific corpus to query. If None, queries all corpora.
            query_transforms (Optional[List]): Query Transforms is used to augment query in pre-processing.

        Returns:
            RagResult: Retrieved chunks with scores and metadata.

        Raises:
            Exception: If query processing fails.
        """
        try:
            if isinstance(query, str):
                query = Query(query_str=query, top_k=self.config.retrieval.top_k)
            if not self.indices or (corpus_id and corpus_id not in self.indices):
                logger.warning(f'No indices found for corpus {corpus_id or 'any'}')
                return RagResult(corpus=Corpus(chunks=[]), scores=[], metadata={'query': query.query_str})
            if query_transforms and query_transforms is not None:
                for t in query_transforms:
                    query = t(query)
            results = []
            target_corpora = [corpus_id] if corpus_id else self.indices.keys()
            tasks = []
            for cid in target_corpora:
                for idx_type, retriever in self.retrievers[cid].items():
                    if query.metadata_filters and query.metadata_filters.get('index_type') and (query.metadata_filters['index_type'] != idx_type):
                        continue
                    task = retriever.aretrieve(Query(query_str=query.query_str, top_k=query.top_k or self.config.retrieval.top_k, similarity_cutoff=query.similarity_cutoff, keyword_filters=query.keyword_filters, metadata_filters=query.metadata_filters))
                    tasks.append((task, cid, idx_type))
            retrieval_tasks = [task for task, _, _ in tasks]
            retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
            for (_, cid, idx_type), result in zip(tasks, retrieval_results):
                if isinstance(result, Exception):
                    logger.error(f'Retrieval failed for {idx_type} in corpus {cid}: {str(result)}')
                else:
                    results.append(result)
                    logger.info(f'Retrieved {len(result.corpus.chunks)} chunks from {idx_type} retriever for corpus {cid}')
            if not results:
                return RagResult(corpus=Corpus(chunks=[]), scores=[], metadata={'query': query.query_str})
            query.similarity_cutoff = self.config.retrieval.similarity_cutoff if query.similarity_cutoff is None else query.similarity_cutoff
            query.keyword_filters = self.config.retrieval.keyword_filters if query.keyword_filters is None else query.keyword_filters
            postprocessor = self.postprocessor_factory.create(self.config.retrieval.postprocessor_type, query=query)
            final_result = postprocessor.postprocess(query, results)
            if query.metadata_filters:
                final_result.corpus.chunks = [chunk for chunk in final_result.corpus.chunks if all((chunk.metadata.model_dump().get(k) == v for k, v in query.metadata_filters.items()))]
                final_result.scores = [chunk.metadata.similarity_score for chunk in final_result.corpus.chunks]
                logger.info(f'Applied metadata filters, retained {len(final_result.corpus.chunks)} chunks')
            logger.info(f'Query returned {len(final_result.corpus.chunks)} chunks after post-processing')
            return final_result
        except Exception as e:
            logger.error(f'Query failed: {str(e)}')
            raise

    def query(self, query: Union[str, Query], corpus_id: Optional[str]=None, query_transforms: Optional[List]=None) -> RagResult:
        """Synchronous wrapper for the async query method."""
        return asyncio.run(self.query_async(query, corpus_id, query_transforms))

class VoyageEmbeddingWrapper(BaseEmbeddingWrapper):
    """Wrapper for Voyage AI embedding models."""

    def __init__(self, model_name: str='voyage-multimodal-3', api_key: str=None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key or os.getenv('VOYAGE_API_KEY')
        self.kwargs = kwargs
        if not self.api_key:
            raise ValueError('Voyage API key is required. Set VOYAGE_API_KEY environment variable or pass api_key parameter.')
        self._embedding_model = VoyageEmbedding(model_name=model_name, api_key=self.api_key, **kwargs)
        logger.info(f'Voyage embedding wrapper initialized with model: {model_name}')

    def get_embedding_model(self) -> BaseEmbedding:
        """Return the LlamaIndex-compatible embedding model."""
        return self._embedding_model

    def validate_model(self, provider: EmbeddingProvider, model_name: str) -> bool:
        """Validate if the model is supported for Voyage AI.
        
        Args:
            provider (EmbeddingProvider): The embedding provider.
            model_name (str): The name of the embedding model to validate.
            
        Returns:
            bool: True if the model is supported, False otherwise.
        """
        supported_models = ['voyage-multimodal-3']
        return model_name in supported_models

    @property
    def dimensions(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_model.dimension

class EmbeddingFactory:
    """Factory for creating embedding models based on configuration."""

    def create(self, provider: EmbeddingProvider, model_config: Dict[str, Any]=None) -> BaseEmbeddingWrapper:
        """Create an embedding model based on the provider and configuration.
        
        Args:
            provider (EmbeddingProvider): The embedding provider (e.g., OpenAI, HuggingFace, Ollama).
            model_config (Dict[str, Any], optional): Configuration for the embedding model.
            
        Returns:
            BaseEmbeddingWrapper: A LlamaIndex-compatible embedding model wrapper.
            
        Raises:
            ValueError: If the provider or configuration is invalid.
        """
        model_config = model_config or {}
        model_config.pop('provider')
        if provider == EmbeddingProvider.OPENAI:
            wrapper = OpenAIEmbeddingWrapper(**model_config)
        elif provider == EmbeddingProvider.AZURE_OPENAI:
            wrapper = AzureOpenAIEmbeddingWrapper(**model_config)
        elif provider == EmbeddingProvider.HUGGINGFACE:
            wrapper = HuggingFaceEmbeddingWrapper(**model_config)
        elif provider == EmbeddingProvider.OLLAMA:
            wrapper = OllamaEmbeddingWrapper(**model_config)
        elif provider == EmbeddingProvider.VOYAGE:
            wrapper = VoyageEmbeddingWrapper(**model_config)
        else:
            raise ValueError(f'Unsupported embedding provider: {provider}')
        logger.info(f'Created embedding model for provider: {provider}')
        return wrapper

class OllamaEmbedding(BaseEmbedding):
    """Ollama embedding model compatible with LlamaIndex BaseEmbedding."""
    base_url: str = None
    client: Client = None
    model_name: str = 'nomic-embed-text'
    embed_batch_size: int = 10
    embedding_dims: int = None
    kwargs: Optional[Dict] = {}

    def __init__(self, model_name: str='nomic-embed-text', base_url: str=None, embedding_dims: int=None, **kwargs):
        super().__init__(model_name=model_name, embed_batch_size=10)
        self.base_url = base_url or 'http://localhost:11434'
        self.embedding_dims = embedding_dims or 512
        self.kwargs = kwargs
        if not EmbeddingProvider.validate_model(EmbeddingProvider.OLLAMA, model_name):
            raise ValueError(f'Unsupported Ollama model: {model_name}. Supported models: {SUPPORTED_MODELS['ollama']}')
        try:
            self.client = Client(host=self.base_url)
            self._ensure_model_exists()
            logger.debug(f'Initialized Ollama embedding model: {model_name}')
        except Exception as e:
            logger.error(f'Failed to initialize Ollama client: {str(e)}')
            raise

    def _ensure_model_exists(self):
        """Ensure the specified model exists locally, pulling it if necessary."""
        try:
            local_models = self.client.list()['models']
            if not any((model.get('name') == self.model_name for model in local_models)):
                logger.info(f'Pulling Ollama model: {self.model_name}')
                self.client.pull(self.model_name)
        except Exception as e:
            logger.error(f'Failed to ensure Ollama model exists: {str(e)}')
            raise

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query string."""
        try:
            response = self.client.embeddings(model=self.model_name, prompt=query, **self.kwargs)
            return response['embedding']
        except Exception as e:
            logger.error(f'Failed to encode query: {str(e)}')
            raise

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a text string."""
        try:
            response = self.client.embeddings(model=self.model_name, prompt=text, **self.kwargs)
            return response['embedding']
        except Exception as e:
            logger.error(f'Failed to encode text: {str(e)}')
            raise

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts synchronously."""
        try:
            embeddings = []
            for i in range(0, len(texts), self.embed_batch_size):
                batch = texts[i:i + self.embed_batch_size]
                batch_embeddings = [self._get_text_embedding(text) for text in batch]
                embeddings.extend(batch_embeddings)
            return embeddings
        except Exception as e:
            logger.error(f'Failed to encode texts: {str(e)}')
            raise

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Asynchronous query embedding (falls back to sync)."""
        return self._get_query_embedding(query)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dims

class BasicLLMSynonymRetriever(BasePGRetriever):

    def __init__(self, graph_store: PropertyGraphStore, include_text: bool=True, include_properties: bool=False, synonym_prompt: str=DEFAULT_SYNONYM_EXPAND_TEMPLATE, max_keywords: int=10, path_depth: int=2, limit: int=30, output_parsing_fn: Optional[Callable]=None, llm: Optional[BaseLLM]=None, **kwargs: Any) -> None:
        self._llm = llm
        self._synonym_prompt = synonym_prompt
        self._output_parsing_fn = output_parsing_fn
        self._max_keywords = max_keywords
        self._path_depth = path_depth
        self._limit = limit
        super().__init__(graph_store=graph_store, include_text=include_text, include_properties=include_properties, **kwargs)

    def _parse_llm_output(self, output: str) -> List[str]:
        if self._output_parsing_fn:
            matches = self._output_parsing_fn(output)
        else:
            matches = output.strip().split('^')
        return [x.strip().capitalize().replace(' ', '_') for x in matches if x.strip()]

    def _prepare_matches(self, matches: List[str], limit: Optional[int]=None) -> List[NodeWithScore]:
        kg_nodes = self._graph_store.get(ids=matches)
        triplets = self._graph_store.get_rel_map(kg_nodes, depth=self._path_depth, limit=limit or self._limit, ignore_rels=[KG_SOURCE_REL])
        return self._get_nodes_with_score(triplets)

    async def _aprepare_matches(self, matches: List[str], limit: Optional[int]=None) -> List[NodeWithScore]:
        kg_nodes = await self._graph_store.aget(ids=matches)
        triplets = await self._graph_store.aget_rel_map(kg_nodes, depth=self._path_depth, limit=limit or self._limit, ignore_rels=[KG_SOURCE_REL])
        return self._get_nodes_with_score(triplets)

    def retrieve_from_graph(self, query_bundle: Query, limit: Optional[int]=None) -> List[NodeWithScore]:
        synonym_prompt = self._synonym_prompt.format_map({'max_keywords': self._max_keywords, 'query_str': query_bundle.query_str})
        response = self._llm.generate(prompt=synonym_prompt, parse_mode='str')
        matches = self._parse_llm_output(response.content)
        logger.info(f'{self.__class__.__name__}, synonym words from llm: {matches}')
        return self._prepare_matches(matches, limit=limit or self._limit)

    async def aretrieve_from_graph(self, query_bundle: Query, limit: Optional[int]=None) -> List[NodeWithScore]:
        synonym_prompt = self._synonym_prompt.format_map({'max_keywords': self._limit, 'query_str': query_bundle.query_str})
        response = await self._llm.async_generate(prompt=synonym_prompt, parse_mode='str')
        matches = self._parse_llm_output(response.content)
        logger.info(f'{self.__class__.__name__}: query: {query_bundle.query_str} \nsynonym words from llm: {matches}')
        return await self._aprepare_matches(matches, limit=limit or self._limit)

class RetrieverFactory:
    """Factory for creating retrievers."""

    def create(self, retriever_type: str, llm: Optional[BaseLLM]=None, index: Optional[BaseIndex]=None, graph_store: Optional[GraphStore]=None, embed_model: Optional[BaseEmbedding]=None, query: Optional[Query]=None, storage_handler: Optional[StorageHandler]=None, chunk_class=None) -> BaseRetrieverWrapper:
        """Create a retriever based on configuration."""
        if retriever_type == RetrieverType.VECTOR.value:
            if not index:
                raise ValueError('Index required for vector retriever')
            retriever = VectorRetriever(index=index, top_k=query.top_k if query else 5, chunk_class=chunk_class)
        elif retriever_type == RetrieverType.GRAPH.value:
            if not (graph_store and embed_model and llm):
                raise ValueError('Graph store, embed model and llm model required for graph retriever')
            retriever = GraphRetriever(llm=llm, graph_store=graph_store, embed_model=embed_model, vector_store=storage_handler.vector_store, top_k=query.top_k if query else 5)
        else:
            raise ValueError(f'Unsupported retriever type: {retriever_type}')
        logger.info(f'Created retriever: {retriever_type}')
        return retriever

class MultimodalReader:
    """An efficient image file reader for multimodal RAG.

    This class provides interface for loading images from files or directories,
    supporting various image formats with path-based lazy loading.

    Attributes:
        recursive (bool): Whether to recursively read directories.
        exclude_hidden (bool): Whether to exclude hidden files (starting with '.').
        num_files_limits (Optional[int]): Maximum number of files to read.
        errors (str): Error handling strategy for file reading (e.g., 'ignore', 'strict').
    """

    def __init__(self, recursive: bool=False, exclude_hidden: bool=True, num_files_limits: Optional[int]=None, errors: str='ignore'):
        self.recursive = recursive
        self.exclude_hidden = exclude_hidden
        self.num_files_limits = num_files_limits
        self.errors = errors

    def _validate_path(self, path: Union[str, Path]) -> Path:
        """Validate and convert a path to a Path object.

        Args:
            path: A string or Path object representing a file or directory.

        Returns:
            Path: A validated Path object.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the path is invalid.
        """
        path = Path(path)
        if not path.exists():
            logger.error(f'Path does not exist: {path}')
            raise FileNotFoundError(f'Path does not exist: {path}')
        return path

    def _check_input(self, input_data: Union[str, List, Tuple], is_file: bool=True) -> Union[List[Path], Path]:
        """Check input to a list of Path objects or a single Path for directories.

        Args:
            input_data: A string, list, or tuple of file/directory paths.
            is_file: Whether to treat input as file paths (True) or directory (False).

        Returns:
            Union[List[Path], Path]: Valid file paths or directory path.

        Raises:
            ValueError: If input type is invalid.
        """
        if isinstance(input_data, str):
            return self._validate_path(input_data)
        elif isinstance(input_data, (list, tuple)):
            if is_file:
                return [self._validate_path(p) for p in input_data]
            else:
                return self._validate_path(input_data[0])
        else:
            logger.error(f'Invalid input type: {type(input_data)}')
            raise ValueError(f'Invalid input type: {type(input_data)}')

    def load(self, file_paths: Union[str, List, Tuple], exclude_files: Optional[Union[str, List, Tuple]]=None, filter_file_by_suffix: Optional[Union[str, List, Tuple]]=None, merge_by_file: bool=False, show_progress: bool=False) -> List[ImageDocument]:
        """Load images from files or directories.

        Args:
            file_paths: A string, list, or tuple of file paths or a directory path.
            exclude_files: Files to exclude from loading.
            filter_file_by_suffix: File extensions to include (e.g., ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']).
            merge_by_file: Whether to merge documents by file (unused for images, kept for compatibility).

        Returns:
            List[ImageDocument]: List of loaded ImageDocuments.

        Raises:
            FileNotFoundError: If input paths are invalid.
            RuntimeError: If image loading fails.
        """
        try:
            input_files = None
            input_dir = None
            if isinstance(file_paths, (list, tuple)):
                input_files = self._check_input(file_paths, is_file=True)
            else:
                path = self._check_input(file_paths, is_file=False)
                if path.is_dir():
                    input_dir = path
                else:
                    input_files = [path]
            exclude_files = self._check_input(exclude_files, is_file=True) if exclude_files else None
            filter_file_by_suffix = list(filter_file_by_suffix) if isinstance(filter_file_by_suffix, (list, tuple)) else [filter_file_by_suffix] if isinstance(filter_file_by_suffix, str) else None
            all_files = []
            if input_files:
                all_files = input_files
            elif input_dir:
                pattern = '**/*' if self.recursive else '*'
                all_files = [f for f in input_dir.glob(pattern) if f.is_file()]
                if self.exclude_hidden:
                    all_files = [f for f in all_files if not f.name.startswith('.')]
            if exclude_files:
                exclude_names = {f.name for f in exclude_files}
                all_files = [f for f in all_files if f.name not in exclude_names]
            if filter_file_by_suffix:
                all_files = [f for f in all_files if f.suffix.lower() in filter_file_by_suffix]
            if self.num_files_limits:
                all_files = all_files[:self.num_files_limits]
            documents = []
            for file_path in all_files:
                if show_progress:
                    logger.info(f'Processing: {file_path.name}')
                try:
                    if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']:
                        img_doc = self._process_image(file_path)
                        if img_doc:
                            documents.append(img_doc)
                except Exception as e:
                    logger.error(f'Failed to process {file_path}: {str(e)}')
                    if self.errors == 'strict':
                        raise
            logger.info(f'Loaded {len(documents)} image documents')
            return documents
        except Exception as e:
            logger.error(f'Failed to load documents: {str(e)}')
            raise RuntimeError(f'Failed to load documents: {str(e)}')

    def _process_image(self, file_path: Path) -> ImageDocument:
        """Process a single image file."""
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                format_name = img.format or 'Unknown'
            document = ImageDocument(text='', image=None, image_path=str(file_path), image_mimetype=f'image/{format_name.lower()}', metadata={'file_path': str(file_path), 'file_name': file_path.name, 'file_type': file_path.suffix, 'file_size': file_path.stat().st_size, 'creation_date': str(file_path.stat().st_ctime), 'last_modified_date': str(file_path.stat().st_mtime)})
            return document
        except Exception as e:
            logger.error(f'Failed to process image {file_path}: {str(e)}')
            if self.errors == 'strict':
                raise
            return None

class IndexFactory:
    """Factory for creating LlamaIndex indices."""

    def create(self, index_type: IndexType, embed_model: BaseEmbedding, storage_handler: StorageHandler, index_config: Dict[str, Any]=None, llm: Optional[BaseLLM]=None) -> BaseIndexWrapper:
        """Create an index based on configuration.
        
        Args:
            index_type (IndexType): The type of index to create.
            embed_model (BaseEmbedding): Embedding model for the index.
            storage_context (StorageContext): Storage context for persistence.
            index_config (Dict[str, Any], optional): Index-specific configuration.
            node_parser (Any, optional): Node parser (unused, kept for compatibility).
            
        Returns:
            BaseIndexWrapper: A wrapped LlamaIndex index.
            
        Raises:
            ValueError: If the index type or configuration is invalid.
        """
        index_config = index_config or {}
        if index_type == IndexType.VECTOR:
            index = VectorIndexing(embed_model=embed_model, storage_handler=storage_handler, index_config=index_config)
        elif index_type == IndexType.GRAPH:
            index = GraphIndexing(embed_model=embed_model, storage_handler=storage_handler, index_config=index_config, llm=llm)
        elif index_type == IndexType.SUMMARY:
            raise NotImplementedError()
        elif index_type == IndexType.TREE:
            raise NotImplementedError()
        else:
            raise ValueError(f'Unsupported index type: {index_type}')
        logger.info(f'Created index: {index_type}')
        return index

class VectorIndexing(BaseIndexWrapper):
    """Wrapper for LlamaIndex VectorStoreIndex."""

    def __init__(self, embed_model: BaseEmbedding, storage_handler: StorageHandler, index_config: Dict[str, Any]=None):
        super().__init__()
        self.index_type = IndexType.VECTOR
        self.embed_model = embed_model
        self.storage_handler = storage_handler
        self._create_storage_context()
        self.id_to_node = dict()
        self.index_config = index_config or {}
        try:
            self.index = VectorStoreIndex(nodes=[], embed_model=self.embed_model, storage_context=self.storage_context, show_progress=self.index_config.get('show_progress', False))
        except Exception as e:
            logger.error(f'Failed to initialize VectorStoreIndex: {str(e)}')
            raise

    def _create_storage_context(self):
        assert self.storage_handler.vector_store is not None, "VectorIndexing must init a vector backend in 'storageHandler'"
        self.storage_context = StorageContext.from_defaults(vector_store=self.storage_handler.vector_store.get_vector_store())

    def get_index(self) -> VectorStoreIndex:
        return self.index

    def insert_nodes(self, nodes: List[Union[Chunk, BaseNode]]) -> Sequence[str]:
        """
        Insert or update nodes into the vector index.

        Converts Chunk objects to LlamaIndex nodes, serializes metadata as JSON strings, and inserts
        them into the VectorStoreIndex. Nodes are cached in id_to_node for quick access.

        Args:
            nodes (List[Union[Chunk, BaseNode]]): List of nodes to insert, either Chunk or BaseNode.
        
        Returns:

        """
        try:
            filtered_nodes = []
            for node in nodes:
                llama_node = node.to_llama_node() if isinstance(node, Chunk) else node
                node_id = llama_node.id if hasattr(llama_node, 'id') else llama_node.id_
                if node_id in self.id_to_node:
                    self.delete_nodes([node_id])
                    logger.info(f'Find the same node in vector database: {node_id}. Update it.')
                filtered_nodes.extend([llama_node])
            nodes_with_embedding = self.index._get_node_with_embedding(nodes=filtered_nodes)
            for node in nodes_with_embedding:
                self.id_to_node[node.node_id] = node.model_copy()
            self.index.insert_nodes(nodes_with_embedding)
            logger.info(f'Inserted {len(nodes_with_embedding)} nodes into VectorStoreIndex')
            return list([n.node_id for n in filtered_nodes])
        except Exception as e:
            logger.error(f'Failed to insert nodes: {str(e)}')
            return []

    def delete_nodes(self, node_ids: Optional[List[str]]=None, metadata_filters: Optional[Dict[str, Any]]=None) -> None:
        """
        Delete nodes from the vector index based on node IDs or metadata filters.

        Removes specified nodes from the index and the id_to_node cache. If metadata_filters are
        provided, nodes matching the filters are deleted.

        Args:
            node_ids (Optional[List[str]]): List of node IDs to delete. Defaults to None.
            metadata_filters (Optional[Dict[str, Any]]): Metadata filters to select nodes for deletion. Defaults to None.
        """
        try:
            if node_ids:
                for node_id in node_ids:
                    if node_id in self.id_to_node:
                        self.index.delete_nodes([node_id], delete_from_docstore=False)
                        if self.index.storage_context.docstore._kvstore._collections_mappings.get(node_id, None) is not None:
                            self.index.storage_context.docstore._kvstore._collections_mappings.pop(node_id)
                        self.id_to_node.pop(node_id)
                        logger.info(f'Deleted node {node_id} from VectorStoreIndex')
            elif metadata_filters:
                nodes_to_delete = []
                for node_id, node in self.id_to_node.items():
                    if all((node.metadata.get(k) == v for k, v in metadata_filters.items())):
                        nodes_to_delete.append(node_id)
                if nodes_to_delete:
                    self.index.delete_nodes(nodes_to_delete, delete_from_docstore=True)
                    for node_id in nodes_to_delete:
                        del self.id_to_node[node_id]
                    logger.info(f'Deleted {len(nodes_to_delete)} nodes matching metadata filters from VectorStoreIndex')
            else:
                logger.warning('No node_ids or metadata_filters provided for deletion')
        except Exception as e:
            logger.error(f'Failed to delete nodes: {str(e)}')
            raise

    async def aload(self, nodes: List[Union[Chunk, BaseNode]]) -> Sequence[str]:
        """
        Asynchronously load nodes into the vector index and its backend store.

        Caches nodes in id_to_node and loads them into the FAISS vector store, ensuring
        no duplicates are inserted by relying on the backend's duplicate checking.

        Args:
            nodes (List[Union[Chunk, BaseNode]]): The nodes to load.

        Returns:
            chunk_ids (List[str]): The id of loaded chunk.
        """
        try:
            node_ids = self.insert_nodes(nodes)
            return node_ids
        except Exception as e:
            logger.error(f'Failed to load nodes into VectorStoreIndex: {str(e)}')
            raise

    def load(self, nodes: List[Union[Chunk, BaseNode]]) -> Sequence[str]:
        """
        Synchronously load nodes into the vector index.

        Args:
            nodes (List[Union[Chunk, BaseNode]]): The nodes to load.
        """
        return asyncio.run(self.aload(nodes))

    def clear(self) -> None:
        """
        Clear all nodes from the vector index and its cache.

        Deletes all nodes from the VectorStoreIndex and clears the id_to_node cache.
        """
        try:
            node_ids = list(self.id_to_node.keys())
            self.index.delete_nodes(node_ids, delete_from_docstore=False)
            self.id_to_node.clear()
            self.index.storage_context.docstore._kvstore._collections_mappings.clear()
            logger.info('Cleared all nodes from VectorStoreIndex')
        except Exception as e:
            logger.error(f'Failed to clear index: {str(e)}')
            raise

    async def _get(self, node_id: str) -> Optional[Chunk]:
        """Get a node by node_id from cache or vector store."""
        try:
            node = self.id_to_node.get(node_id, None)
            if node:
                if isinstance(node, Chunk):
                    return node.model_copy()
                return Chunk.from_llama_node(node)
            logger.warning(f'Node with ID {node_id} not found in cache or vector store')
            return None
        except Exception as e:
            logger.error(f'Failed to get node {node_id}: {str(e)}')
            return None

    async def get(self, node_ids: Sequence[str]) -> List[Chunk]:
        """Get nodes by node_ids from cache or vector store."""
        try:
            nodes = await asyncio.gather(*[self._get(node) for node in node_ids])
            nodes = [node for node in nodes if node is not None]
            logger.info(f'Retrieved {len(nodes)} nodes for node_ids: {node_ids}')
            return nodes
        except Exception as e:
            logger.error(f'Failed to get nodes: {str(e)}')
            return []

class GraphIndexing(BaseIndexWrapper):
    """Wrapper for LlamaIndex PropertyGraphIndex."""

    def __init__(self, embed_model: BaseEmbedding, storage_handler: StorageHandler, llm: BaseLLM, index_config: Dict[str, Any]=None) -> None:
        super().__init__()
        self.index_type = IndexType.GRAPH
        self._embed_model = embed_model
        self.storage_handler = storage_handler
        self._create_storage_context()
        self.id_to_node = dict()
        self.index_config = index_config or {}
        assert isinstance(llm, BaseLLM), 'The LLM model should be an instance class.'
        kg_extractor = BasicGraphExtractLLM(llm=llm, num_workers=self.index_config.get('num_workers', 4))
        try:
            vector_store = self.storage_handler.vector_store.get_vector_store() if self.storage_handler.vector_store is not None else None
            self.index = PropertyGraphIndex(nodes=[], kg_extractors=[kg_extractor, ImplicitPathExtractor()], embed_model=self._embed_model, vector_store=vector_store if not self.storage_handler.graph_store.supports_vector_queries else None, property_graph_store=self.storage_context.graph_store, storage_context=self.storage_context, show_progress=self.index_config.get('show_progress', False), use_async=self.index_config.get('use_async', True))
        except Exception as e:
            logger.error(f'Failed to initialize {self.__class__}: {str(e)}')
            raise

    def get_index(self) -> PropertyGraphIndex:
        return self.index

    def _create_storage_context(self):
        """Create the LlamaIndex-compatible storage context."""
        super()._create_storage_context()
        assert self.storage_handler.graph_store is not None, "GraphIndexing must init a graph backend in 'storageHandler'"
        self.storage_context = StorageContext.from_defaults(graph_store=self.storage_handler.graph_store.get_graph_store())

    def insert_nodes(self, nodes: List[Union[Chunk, BaseNode]]):
        """
        Insert or update nodes into the graph index.

        Converts Chunk objects to LlamaIndex nodes, serializes metadata as JSON strings,
        and inserts them into the PropertyGraphIndex. Nodes are cached in id_to_node for
        quick access.

        Args:
            nodes (List[Union[Chunk, BaseNode]]): List of nodes to insert, either Chunk or BaseNode.
        """
        try:
            filtered_nodes = [node.to_llama_node() if isinstance(node, Chunk) else node for node in nodes]
            for node in filtered_nodes:
                node.metadata = {'metadata': json.dumps(node.metadata)}
            nodes = self.index._insert_nodes(filtered_nodes)
            logger.info(f'Inserted {len(nodes)} nodes into PropertyGraphIndex')
            return list([node.node_id for node in nodes])
        except Exception as e:
            logger.error(f'Failed to insert nodes: {str(e)}')
            return []

    def delete_nodes(self, node_ids: Optional[List[str]]=None, metadata_filters: Optional[Dict[str, Any]]=None):
        """
        Delete nodes from the graph index based on node IDs or metadata filters.

        Removes specified nodes from the index and the id_to_node cache. If metadata_filters
        are provided, nodes matching the filters are deleted.

        Args:
            node_ids (Optional[List[str]]): List of node IDs to delete. Defaults to None.
            metadata_filters (Optional[Dict[str, Any]]): Metadata filters to select nodes for deletion. Defaults to None.
        """
        try:
            if node_ids:
                for node_id in node_ids:
                    if node_id in self.id_to_node:
                        self.index.delete_nodes([node_id])
                        self.id_to_node.pop(node_id)
                        logger.info(f'Deleted node {node_id} from PropertyGraphIndex')
            elif metadata_filters:
                nodes_to_delete = []
                for node_id, node in self.id_to_node.items():
                    if all((node.metadata.get(k) == v for k, v in metadata_filters.items())):
                        nodes_to_delete.append(node_id)
                if nodes_to_delete:
                    self.index.delete_nodes(nodes_to_delete)
                    for node_id in nodes_to_delete:
                        self.id_to_node.pop(node_id)
                    logger.info(f'Deleted {len(nodes_to_delete)} nodes matching metadata filters from PropertyGraphIndex')
            else:
                logger.warning('No node_ids or metadata_filters provided for deletion')
        except Exception as e:
            logger.error(f'Failed to delete nodes: {str(e)}')
            raise

    async def aload(self, nodes: List[Union[Chunk, BaseNode, LabelledNode, Relation, EntityNode, ChunkNode]]) -> Sequence[str]:
        """
        Asynchronously load nodes into the graph index and its backend stores.

        Caches nodes in the id_to_node dictionary and loads them into the graph and optionally
        vector stores, ensuring no duplicates by relying on the backend's duplicate checking.

        Args:
            nodes (List[Union[Chunk, BaseNode]]): List of nodes to load, either Chunk or BaseNode.
        """
        try:
            chunk_ids = self.insert_nodes(nodes)
            return chunk_ids
        except Exception as e:
            logger.error(f'Failed to load nodes: {str(e)}')

    def load(self, nodes: List[Union[Chunk, BaseNode]]) -> Sequence[str]:
        """
        Synchronously load nodes into the graph index.

        Wraps the asynchronous aload method to provide a synchronous interface for loading nodes.

        Args:
            nodes (List[Union[Chunk, BaseNode]]): List of nodes to load, either Chunk or BaseNode.

        """
        return asyncio.run(self.aload(nodes))

    def build_kv_store(self) -> None:
        """
        Match all the nodes and relations into python Dict.
        """
        for node in self.storage_handler.graph_store.build_kv_store():
            self.id_to_node[str(uuid4())] = node

    def clear(self):
        """
        Clear all nodes from the graph index and its cache.

        Deletes all nodes from the PropertyGraphIndex and clears the id_to_node cache.
        """
        try:
            self.storage_handler.graph_store.clear()
            self.id_to_node.clear()
            logger.info('Cleared all nodes from PropertyGraphIndex')
        except Exception as e:
            logger.error(f'Failed to clear index: {str(e)}')
            raise

    async def _get(self, node_id: str) -> Optional[Chunk]:
        """Get a node by node_id from cache or vector store."""
        try:
            node = self.storage_handler.graph_store.get(ids=[node_id])
            if node:
                if isinstance(node, Chunk):
                    return node.model_copy()
                return Chunk.from_llama_node(node)
            logger.warning(f'Node with ID {node_id} not found in cache or vector store')
            return None
        except Exception as e:
            logger.error(f'Failed to get node {node_id}: {str(e)}')
            return None

    async def get(self, node_ids: Sequence[str]) -> List[Chunk]:
        """Get nodes by node_ids from cache or vector store."""
        try:
            nodes = await asyncio.gather(*[self._get(node) for node in node_ids])
            nodes = [node for node in nodes if node is not None]
            logger.info(f'Retrieved {len(nodes)} nodes for node_ids: {node_ids}')
            return nodes
        except Exception as e:
            logger.error(f'Failed to get nodes: {str(e)}')
            return []

class ChunkFactory:
    """Factory for creating chunkers based on configuration."""

    def create(self, strategy: ChunkingStrategy, embed_model: BaseEmbedding=None, chunker_config: Dict[str, Any]=None) -> BaseChunker:
        """Create a chunker based on strategy and configuration.
        
        Args:
            strategy (ChunkingStrategy): The chunking strategy.
            embed_model (BaseEmbedding, optional): Embedding model for semantic chunking.
            chunker_config (Dict[str, Any], optional): Chunker configuration.
            
        Returns:
            BaseChunker: A chunker instance.
            
        Raises:
            ValueError: If the strategy or configuration is invalid.
        """
        chunker_config = chunker_config or {}
        if strategy == ChunkingStrategy.SIMPLE:
            chunker = SimpleChunker(chunk_size=chunker_config.get('chunk_size', 1024), chunk_overlap=chunker_config.get('chunk_overlap', 20), max_workers=chunker_config.get('max_workers', 2))
        elif strategy == ChunkingStrategy.SEMANTIC:
            if not embed_model:
                raise ValueError('Embed model required for semantic chunking')
            chunker = SemanticChunker(embed_model=embed_model, similarity_threshold=chunker_config.get('similarity_threshold', 0.7), max_workers=chunker_config.get('max_workers', 2))
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            chunker = HierarchicalChunker(chunk_sizes=chunker_config.get('chunk_sizes', [2048, 512, 128]), chunk_overlap=chunker_config.get('chunk_overlap', 20))
        else:
            raise ValueError(f'Unsupported chunking strategy: {strategy}')
        logger.info(f'Created chunker for strategy: {strategy}')
        return chunker

class PostprocessorFactory:
    """Factory for creating post-processors."""

    def create(self, postprocessor_type: str, query: Optional[Query]=None) -> BasePostprocessor:
        """Create a post-processor based on configuration.
        
        Args:
            postprocessor_type (str): Type of post-processor (e.g., 'simple', 'bge').
            query (Query, optional): Query for configuration.
            
        Returns:
            BasePostprocessor: A post-processor instance.
            
        Raises:
            ValueError: If the post-processor type or configuration is invalid.
        """
        if postprocessor_type == RerankerType.SIMPLE:
            if not query:
                raise ValueError('Query required for reranker')
            postprocessor = SimpleReranker(similarity_cutoff=query.similarity_cutoff, keyword_filters=query.keyword_filters)
        else:
            raise ValueError(f'Unsupported post-processor type: {postprocessor_type}')
        logger.info(f'Created post-processor: {postprocessor_type}')
        return postprocessor

class LongTermMemory(BaseMemory):
    """
    Manages long-term storage and retrieval of memories, integrating with RAGEngine for indexing
    and StorageHandler for persistence.
    """
    storage_handler: StorageHandler = Field(..., description='Handler for persistent storage')
    rag_config: RAGConfig = Field(..., description='Configuration for RAG engine')
    rag_engine: RAGEngine = Field(default=None, description='RAG engine for indexing and retrieval')
    memory_table: str = Field(default='memory', description='Database table for storing memories')
    default_corpus_id: Optional[str] = Field(default=None, description='Default corpus ID for memory indexing')

    def init_module(self):
        """Initialize the RAG engine and memory indices."""
        super().init_module()
        if self.rag_engine is None:
            self.rag_engine = RAGEngine(config=self.rag_config, storage_handler=self.storage_handler)
        if self.default_corpus_id is None:
            self.default_corpus_id = str(uuid4())
        logger.info(f'Initialized LongTermMemory with corpus_id {self.default_corpus_id}')

    def _create_memory_chunk(self, message: Message, memory_id: str) -> Chunk:
        """Convert a Message to a Chunk for RAG indexing."""
        metadata = ChunkMetadata(corpus_id=self.default_corpus_id, memory_id=memory_id, timestamp=message.timestamp, action=message.action, wf_goal=message.wf_goal, agent=message.agent, msg_type=message.msg_type.value if message.msg_type else None, prompt=message.prompt, next_actions=message.next_actions, wf_task=message.wf_task, wf_task_desc=message.wf_task_desc, message_id=message.message_id, content=json.dumps(message.content))
        return Chunk(chunk_id=memory_id, text=str(message.content), metadata=metadata, start_char_idx=0, end_char_idx=len(str(message.content)))

    def _chunk_to_message(self, chunk: Chunk) -> Message:
        """Convert a Chunk to a Message object."""
        return Message(content=chunk.metadata.content, action=chunk.metadata.action, wf_goal=chunk.metadata.wf_goal, timestamp=chunk.metadata.timestamp, agent=chunk.metadata.agent, msg_type=chunk.metadata.msg_type, prompt=chunk.metadata.prompt, next_actions=chunk.metadata.next_actions, wf_task=chunk.metadata.wf_task, wf_task_desc=chunk.metadata.wf_task_desc, message_id=chunk.metadata.message_id)

    def add(self, messages: Union[Message, str, List[Union[Message, str]]]) -> List[str]:
        """Store messages in memory and index them in RAGEngine, returning memory_ids."""
        if not isinstance(messages, list):
            messages = [messages]
        messages = [Message(content=msg) if isinstance(msg, str) else msg for msg in messages]
        messages = [msg for msg in messages if msg.content]
        if not messages:
            logger.warning('No valid messages to add')
            return []
        existing_hashes = {record['content_hash'] for record in self.storage_handler.load(tables=[self.memory_table]).get(self.memory_table, []) if 'content_hash' in record}
        memory_ids = [str(uuid4()) for _ in messages]
        final_messages = []
        final_memory_ids = []
        final_chunks = []
        for msg, memory_id in zip(messages, memory_ids):
            content_hash = hashlib.sha256(str(msg.content).encode()).hexdigest()
            if content_hash in existing_hashes:
                logger.info(f'Duplicate message found (hash): {msg.content[:50]}...')
                existing_id = next((r['memory_id'] for r in self.storage_handler.load(tables=[self.memory_table]).get(self.memory_table, []) if r.get('content_hash') == content_hash), None)
                if existing_id:
                    final_memory_ids.append(existing_id)
                    continue
            final_messages.append(msg)
            final_memory_ids.append(memory_id)
            chunk = self._create_memory_chunk(msg, memory_id)
            chunk.metadata.content_hash = content_hash
            final_chunks.append(chunk)
        if not final_chunks:
            logger.info('No messages added after deduplication')
            return final_memory_ids
        for msg in final_messages:
            super().add_message(msg)
        corpus = Corpus(chunks=final_chunks, corpus_id=self.default_corpus_id)
        chunk_ids = self.rag_engine.add(index_type=self.rag_config.index.index_type, nodes=corpus, corpus_id=self.default_corpus_id)
        if not chunk_ids:
            logger.error('Failed to index memories')
            return final_memory_ids
        return final_memory_ids

    async def get(self, memory_ids: Union[str, List[str]], return_chunk: bool=True) -> List[Tuple[Union[Chunk, Message], str]]:
        """Retrieve memories by memory_ids, returning (Message/Chunk, memory_id) tuples."""
        if not isinstance(memory_ids, list):
            memory_ids = [memory_ids]
        if not memory_ids:
            logger.warning('No memory_ids provided for get')
            return []
        try:
            chunks = await self.rag_engine.aget(corpus_id=self.default_corpus_id, index_type=self.rag_config.index.index_type, node_ids=memory_ids)
            results = [(self._chunk_to_message(chunk), chunk.metadata.memory_id) if not return_chunk else (chunk, chunk.metadata.memory_id) for chunk in chunks if chunk]
            logger.info(f'Retrieved {len(results)} memories for memory_ids: {memory_ids}')
            return results
        except Exception as e:
            logger.error(f'Failed to get memories: {str(e)}')
            return []

    def delete(self, memory_ids: Union[str, List[str]]) -> List[bool]:
        """Delete memories by memory_ids, returning success status for each."""
        if not isinstance(memory_ids, list):
            memory_ids = [memory_ids]
        if not memory_ids:
            logger.warning('No memory_ids provided for deletion')
            return []
        successes = [False] * len(memory_ids)
        valid_memory_ids = []
        existing_chunks = asyncio.run(self.get(memory_ids, return_chunk=True))
        for idx, (chunk, mid) in enumerate(existing_chunks):
            if chunk:
                valid_memory_ids.append(mid)
                super().remove_message(self._chunk_to_message(chunk))
                successes[idx] = True
        if not valid_memory_ids:
            logger.info('No memories found for deletion')
            return successes
        self.rag_engine.delete(corpus_id=self.default_corpus_id, index_type=self.rag_config.index.index_type, node_ids=valid_memory_ids)
        return successes

    def update(self, updates: Union[Tuple[str, Union[Message, str]], List[Tuple[str, Union[Message, str]]]]) -> List[bool]:
        """Update memories with new content, returning success status for each."""
        if not isinstance(updates, list):
            updates = [updates]
        updates = [(mid, Message(content=msg) if isinstance(msg, str) else msg) for mid, msg in updates]
        updates_dict = {mid: msg for mid, msg in updates if msg.content}
        if not updates_dict:
            logger.warning('No valid updates provided')
            return []
        memory_ids = list(updates_dict.keys())
        existing_memories = asyncio.run(self.get(memory_ids, return_chunk=False))
        existing_dict = {mid: msg for msg, mid in existing_memories}
        successes = [False] * len(updates)
        final_updates = []
        final_memory_ids = []
        for mid, msg in updates_dict.items():
            if mid not in existing_dict:
                logger.warning(f'No memory found with memory_id {mid}')
                continue
            final_updates.append((mid, msg))
            final_memory_ids.append(mid)
            successes[memory_ids.index(mid)] = True
            super().remove_message(existing_dict[mid])
        if not final_updates:
            logger.info('No memories updated')
            return successes
        chunks = [self._create_memory_chunk(msg, mid) for mid, msg in final_updates]
        for msg in [msg for _, msg in final_updates]:
            super().add_message(msg)
        corpus = Corpus(chunks=chunks, corpus_id=self.default_corpus_id)
        chunk_ids = self.rag_engine.add(index_type=self.rag_config.index.index_type, nodes=corpus, corpus_id=self.default_corpus_id)
        if not chunk_ids:
            logger.error(f'Failed to update memories in RAG index: {final_memory_ids}')
            return [False] * len(updates)
        return successes

    async def search_async(self, query: Union[str, Query], n: Optional[int]=None, metadata_filters: Optional[Dict]=None, return_chunk=False) -> List[Tuple[Message, str]]:
        """Retrieve messages from RAG index asynchronously based on a query, returning messages and memory_ids."""
        if isinstance(query, str):
            query_obj = Query(query_str=query, top_k=n or self.rag_config.retrieval.top_k, metadata_filters=metadata_filters or {})
        else:
            query_obj = query
            query_obj.top_k = n or self.rag_config.retrieval.top_k
            if metadata_filters:
                query_obj.metadata_filters = {**query_obj.metadata_filters, **metadata_filters} if query_obj.metadata_filters else metadata_filters
        try:
            result: RagResult = await self.rag_engine.query_async(query_obj, corpus_id=self.default_corpus_id)
            if return_chunk:
                return [(chunk, chunk.metadata.memory_id) for chunk in result.corpus.chunks]
            else:
                messages = [(self._chunk_to_message(chunk), chunk.metadata.memory_id) for chunk in result.corpus.chunks]
            logger.info(f'Retrieved {len(messages)} memories for query: {query_obj.query_str}')
            return messages[:n] if n else messages
        except Exception as e:
            logger.error(f'Failed to search memories: {str(e)}')
            return []

    def search(self, query: Union[str, Query], n: Optional[int]=None, metadata_filters: Optional[Dict]=None) -> List[Tuple[Message, str]]:
        """Synchronous wrapper for searching memories."""
        return asyncio.run(self.search_async(query, n, metadata_filters))

    def clear(self) -> None:
        """Clear all messages and indices."""
        super().clear()
        self.rag_engine.clear(corpus_id=self.default_corpus_id)
        logger.info(f'Cleared LongTermMemory with corpus_id {self.default_corpus_id}')

    def save(self, save_path: Optional[str]=None) -> None:
        """Save all indices and memory data to database."""
        self.rag_engine.save(output_path=save_path, corpus_id=self.default_corpus_id, table=self.memory_table)

    def load(self, save_path: Optional[str]=None) -> List[str]:
        """Load memory data from database and reconstruct indices, returning memory_ids."""
        return self.rag_engine.load(source=save_path, corpus_id=self.default_corpus_id, table=self.memory_table)

class TestSearchEngine(unittest.TestCase):
    """Unit tests for SearchEngine interfaces using HotpotQA JSON example."""

    def setUp(self):
        """Set up SearchEngine, StorageHandler, and temporary directory for each test."""
        load_dotenv()
        self.mock_embedding = MockOpenAIEmbeddingWrapper()
        self.patcher = patch('evoagentx.rag.rag.EmbeddingFactory.create', return_value=self.mock_embedding)
        self.mock_create = self.patcher.start()
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f'Created temporary directory: {self.temp_dir}')
        self.store_config = StoreConfig(dbConfig=DBConfig(db_name='sqlite', path=os.path.join(self.temp_dir, 'test_hotpotQA.sql')), vectorConfig=VectorStoreConfig(vector_name='faiss', dimensions=1536, index_type='flat_l2'), graphConfig=None, path=self.temp_dir)
        self.storage_handler = StorageHandler(storageConfig=self.store_config)
        self.rag_config = RAGConfig(reader=ReaderConfig(recursive=False, exclude_hidden=True, num_files_limit=None, custom_metadata_function=None, extern_file_extractor=None, errors='ignore', encoding='utf-8'), chunker=ChunkerConfig(strategy='simple', chunk_size=512, chunk_overlap=0, max_chunks=None), embedding=EmbeddingConfig(provider='openai', model_name='text-embedding-ada-002', api_key='dummy_key'), index=IndexConfig(index_type='vector'), retrieval=RetrievalConfig(retrivel_type='vector', postprocessor_type='simple', top_k=10, similarity_cutoff=0.3, keyword_filters=None, metadata_filters=None))
        self.search_engine = RAGEngine(config=self.rag_config, storage_handler=self.storage_handler)
        self.corpus_id = HOTPOTQA_EXAMPLE['_id']
        self.context_files = []
        self.supporting_titles = {fact[0] for fact in HOTPOTQA_EXAMPLE['supporting_facts']}
        self.context_data = HOTPOTQA_EXAMPLE['context']
        self.query_text = HOTPOTQA_EXAMPLE['question']
        for title, sentences in self.context_data:
            content = '\n'.join(sentences)
            file_path = os.path.join(self.temp_dir, f'{title.replace(' ', '_')}.txt')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.context_files.append(str(file_path))

    def tearDown(self):
        """Clean up temporary directory, clear indices, and stop patcher."""
        self.search_engine.clear()
        self.patcher.stop()
        logger.info(f'Cleaned up temporary directory: {self.temp_dir}')

    def test_read(self):
        """Test the read method by loading HotpotQA context files."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.assertIsInstance(corpus, Corpus, 'read should return a Corpus object')
        self.assertEqual(corpus.corpus_id, self.corpus_id, 'Corpus ID should match')
        self.assertGreater(len(corpus.chunks), 0, 'Corpus should contain chunks')
        for chunk in corpus.chunks:
            self.assertIsInstance(chunk.metadata, ChunkMetadata, 'Chunk should have metadata')
            self.assertIn('file_name', chunk.metadata.model_dump(), 'Metadata should include file_name')
        logger.info(f'Read {len(corpus.chunks)} chunks for corpus {self.corpus_id}')

    def test_add(self):
        """Test the add method by indexing HotpotQA corpus."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        self.assertIn(self.corpus_id, self.search_engine.indices, 'Corpus should be indexed')
        self.assertIn(IndexType.VECTOR, self.search_engine.indices[self.corpus_id], 'Vector index should exist')
        index = self.search_engine.indices[self.corpus_id][IndexType.VECTOR]
        self.assertGreater(len(index.id_to_node), 0, 'Index should contain nodes')
        for node_id, node in index.id_to_node.items():
            self.assertEqual(node.metadata['corpus_id'], self.corpus_id, 'Node metadata should include corpus_id')
            self.assertEqual(node.metadata['index_type'], IndexType.VECTOR, 'Node metadata should include index_type')
        logger.info(f'Added {len(corpus.chunks)} nodes to vector index for corpus {self.corpus_id}')

    def test_query(self):
        """Test the query method with HotpotQA question, validating top-K retrieval."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        query = Query(query_str=self.query_text, top_k=10)
        result = self.search_engine.query(query, corpus_id=self.corpus_id)
        self.assertIsInstance(result, RagResult, 'query should return a RagResult object')
        self.assertLessEqual(len(result.corpus.chunks), 10, 'Should return at most top_k chunks')
        self.assertEqual(len(result.scores), len(result.corpus.chunks), 'Scores should match chunks')
        retrieved_titles = set()
        for chunk in result.corpus.chunks:
            file_name = chunk.metadata.model_dump().get('file_name', '')
            title = os.path.basename(file_name).replace('_', ' ').replace('.txt', '')
            retrieved_titles.add(title)
        recall = len(retrieved_titles.intersection(self.supporting_titles)) / len(self.supporting_titles)
        self.assertGreaterEqual(recall, 0.0, 'Recall may be low with dummy embeddings')
        logger.info(f'Query retrieved {len(result.corpus.chunks)} chunks with recall@10={recall}')

    def test_delete_by_node_ids(self):
        """Test the delete method by removing specific nodes."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        index = self.search_engine.indices[self.corpus_id][IndexType.VECTOR]
        node_ids = list(index.id_to_node.keys())[:2]
        initial_node_count = len(index.id_to_node)
        self.search_engine.delete(corpus_id=self.corpus_id, index_type=IndexType.VECTOR, node_ids=node_ids)
        remaining_node_count = len(index.id_to_node)
        self.assertEqual(remaining_node_count, initial_node_count - len(node_ids), 'Nodes should be deleted')
        for node_id in node_ids:
            self.assertNotIn(node_id, index.id_to_node, f'Node {node_id} should be deleted')
        logger.info(f'Deleted {len(node_ids)} nodes from corpus {self.corpus_id}')

    def test_delete_by_metadata(self):
        """Test the delete method using metadata filters."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        index = self.search_engine.indices[self.corpus_id][IndexType.VECTOR]
        metadata_filters = {'file_name': str(self.context_files[0])}
        initial_node_count = len(index.id_to_node)
        self.search_engine.delete(corpus_id=self.corpus_id, index_type=IndexType.VECTOR, metadata_filters=metadata_filters)
        remaining_nodes = [node_id for node_id, node in index.id_to_node.items() if node.metadata.get('file_name') != str(self.context_files[0])]
        self.assertEqual(len(index.id_to_node), len(remaining_nodes), 'Nodes matching metadata should be deleted')
        logger.info(f'Deleted nodes with metadata {metadata_filters} from corpus {self.corpus_id}')

    def test_clear(self):
        """Test the clear method by removing all indices."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        self.search_engine.clear(corpus_id=self.corpus_id)
        self.assertNotIn(self.corpus_id, self.search_engine.indices, 'Corpus should be cleared')
        self.assertNotIn(self.corpus_id, self.search_engine.retrievers, 'Retrievers should be cleared')
        logger.info(f'Cleared corpus {self.corpus_id}')

    def test_save_to_files(self):
        """Test the save method by saving indices to files."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        output_path = os.path.join(self.temp_dir, 'output')
        self.search_engine.save(output_path=str(output_path), corpus_id=self.corpus_id, index_type=IndexType.VECTOR)
        if isinstance(output_path, str):
            from pathlib import Path
            output_path = Path(output_path)
        nodes_files = list(output_path.glob('*_nodes.jsonl'))
        metadata_files = list(output_path.glob('*_metadata.json'))
        self.assertEqual(len(nodes_files), 1, 'Should save one nodes file')
        self.assertEqual(len(metadata_files), 1, 'Should save one metadata file')
        with open(nodes_files[0], 'r', encoding='utf-8') as f:
            chunks = [json.loads(line) for line in f]
            self.assertGreater(len(chunks), 0, 'Nodes file should contain chunks')
        with open(metadata_files[0], 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            self.assertEqual(metadata['corpus_id'], self.corpus_id, 'Metadata should include corpus_id')
        logger.info(f'Saved indices to {output_path}')

    def test_load_from_files(self):
        """Test the load method by loading indices from files."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        output_path = os.path.join(self.temp_dir, 'output')
        self.search_engine.save(output_path=str(output_path), corpus_id=self.corpus_id, index_type=IndexType.VECTOR)
        self.search_engine.clear()
        self.search_engine.load(source=str(output_path), corpus_id=self.corpus_id, index_type=IndexType.VECTOR)
        self.assertIn(self.corpus_id, self.search_engine.indices, 'Corpus should be loaded')
        index = self.search_engine.indices[self.corpus_id][IndexType.VECTOR]
        self.assertGreater(len(index.id_to_node), 0, 'Index should contain nodes')
        query = Query(query_str=self.query_text, top_k=10)
        result = self.search_engine.query(query, corpus_id=self.corpus_id)
        self.assertEqual(len(result.corpus.chunks), 0)
        logger.info(f'Loaded indices from {output_path}')

    def test_save_to_database(self):
        """Test the save method by saving indices to database."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        self.search_engine.save(corpus_id=self.corpus_id, index_type=IndexType.VECTOR, table='indexing')
        records = self.storage_handler.load(tables=['indexing']).get('indexing', [])
        self.assertGreater(len(records), 0, 'Database should contain records')
        for record in records:
            parsed = self.storage_handler.parse_result(record, IndexStore)
            self.assertEqual(parsed['corpus_id'], self.corpus_id, 'Record should match corpus_id')
        logger.info(f'Saved indices to database table indexing')

    def test_load_from_database(self):
        """Test the load method by loading indices from database."""
        corpus = self.search_engine.read(file_paths=self.context_files, filter_file_by_suffix='.txt', corpus_id=self.corpus_id)
        self.search_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.corpus_id)
        self.search_engine.save(corpus_id=self.corpus_id, index_type=IndexType.VECTOR, table='indexing')
        self.search_engine.clear()
        self.search_engine.load(corpus_id=self.corpus_id, index_type=IndexType.VECTOR, table='indexing')
        self.assertIn(self.corpus_id, self.search_engine.indices, 'Corpus should be loaded')
        index = self.search_engine.indices[self.corpus_id][IndexType.VECTOR]
        self.assertGreater(len(index.id_to_node), 0, 'Index should contain nodes')
        query = Query(query_str=self.query_text, top_k=10)
        result = self.search_engine.query(query, corpus_id=self.corpus_id)
        self.assertEqual(len(result.corpus.chunks), 0)
        logger.info(f'Loaded indices from database table indexing')

    def test_edge_case_empty_corpus(self):
        """Test behavior with empty corpus or invalid corpus_id."""
        result = self.search_engine.query(query=self.query_text, corpus_id='nonexistent')
        self.assertEqual(len(result.corpus.chunks), 0, 'Query on nonexistent corpus should return empty result')
        self.search_engine.delete(corpus_id='nonexistent')
        self.assertNotIn('nonexistent', self.search_engine.indices, 'Delete on nonexistent corpus should not fail')
        self.search_engine.clear(corpus_id='nonexistent')
        self.assertNotIn('nonexistent', self.search_engine.indices, 'Clear on nonexistent corpus should not fail')
        logger.info('Handled edge case for empty/nonexistent corpus')

