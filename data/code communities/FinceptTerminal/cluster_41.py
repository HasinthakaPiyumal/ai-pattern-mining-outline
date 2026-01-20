# Cluster 41

class SessionManager:
    """Session manager with first-time user detection and centralized config"""

    def __init__(self):
        self.config_dir = Path.home() / config.CONFIG_DIR_NAME
        self.credentials_file = self.config_dir / config.CREDENTIALS_FILE
        self.ensure_config_dir()
        info('Session Manager initialized', module='session')
        info('API URL configured', module='session', context={'api_url': config.get_api_url()})
        info('Strict Mode status', module='session', context={'strict_mode': is_strict_mode()})
        info('Credentials file location', module='session', context={'credentials_file': str(self.credentials_file)})

    @monitor_performance
    def ensure_config_dir(self):
        """Ensure config directory exists"""
        with operation('ensure_config_directory', module='session'):
            try:
                self.config_dir.mkdir(exist_ok=True)
                debug('Config directory ensured', module='session', context={'config_dir': str(self.config_dir)})
            except Exception as e:
                error('Could not create config directory', module='session', context={'error': str(e), 'config_dir': str(self.config_dir)}, exc_info=True)

    def is_first_time_user(self) -> bool:
        """Check if this is a first-time user (no credentials file exists)"""
        exists = self.credentials_file.exists()
        info(f'First-time user check: {('No' if exists else 'Yes')}', module='session', context={'credentials_file_exists': exists})
        return not exists

    @monitor_performance
    def save_credentials_only(self, credentials_data: Dict[str, Any]) -> bool:
        """Save only essential credentials (API key, user type) - not full session"""
        with operation('save_credentials', module='session'):
            try:
                essential_data = {'api_key': credentials_data.get('api_key'), 'user_type': credentials_data.get('user_type'), 'device_id': credentials_data.get('device_id'), 'saved_at': datetime.now().isoformat(), 'api_version': config.API_VERSION, 'api_url': config.get_api_url()}
                with open(self.credentials_file, 'w', encoding='utf-8') as f:
                    json.dump(essential_data, f, indent=2)
                user_type = credentials_data.get('user_type', 'unknown')
                info('Credentials saved successfully', module='session', context={'user_type': user_type, 'is_first_time': self.is_first_time_user()})
                return True
            except Exception as e:
                error('Failed to save credentials', module='session', context={'error': str(e)}, exc_info=True)
                return False

    @monitor_performance
    def load_credentials(self) -> Optional[Dict[str, Any]]:
        """Load only stored credentials"""
        with operation('load_credentials', module='session'):
            try:
                if not self.credentials_file.exists():
                    debug('No credentials file found - first-time user', module='session')
                    return None
                with open(self.credentials_file, 'r', encoding='utf-8') as f:
                    credentials = json.load(f)
                saved_api_url = credentials.get('api_url')
                current_api_url = config.get_api_url()
                if saved_api_url and saved_api_url != current_api_url:
                    warning('API URL mismatch detected', module='session', context={'saved_url': saved_api_url, 'current_url': current_api_url})
                    info('Clearing credentials due to API URL change', module='session')
                    self.clear_credentials()
                    return None
                user_type = credentials.get('user_type', 'unknown')
                info('Credentials loaded successfully', module='session', context={'user_type': user_type})
                return credentials
            except json.JSONDecodeError as e:
                error('Invalid JSON in credentials file', module='session', context={'error': str(e)}, exc_info=True)
                return None
            except Exception as e:
                error('Failed to load credentials', module='session', context={'error': str(e)}, exc_info=True)
                return None

    def clear_credentials(self) -> bool:
        """Clear saved credentials"""
        with operation('clear_credentials', module='session'):
            try:
                if self.credentials_file.exists():
                    self.credentials_file.unlink()
                    info('Credentials cleared successfully', module='session')
                    return True
                else:
                    debug('No credentials file to clear', module='session')
                    return True
            except Exception as e:
                error('Failed to clear credentials', module='session', context={'error': str(e)}, exc_info=True)
                return False

    @monitor_performance
    def check_api_connectivity(self) -> bool:
        """Check if API server is available"""
        with operation('check_api_connectivity', module='session'):
            debug('Checking API connectivity', module='session')
            try:
                response = requests.get(get_api_endpoint('/health'), timeout=config.CONNECTION_TIMEOUT)
                if response.status_code == 200:
                    info('API server is available', module='session', context={'api_url': config.get_api_url()})
                    return True
                else:
                    warning('API server returned non-200 status', module='session', context={'status_code': response.status_code, 'api_url': config.get_api_url()})
                    return False
            except requests.exceptions.ConnectionError:
                warning('Cannot connect to API server', module='session', context={'api_url': config.get_api_url()})
                return False
            except requests.exceptions.Timeout:
                warning('API server timeout', module='session', context={'api_url': config.get_api_url(), 'timeout': config.CONNECTION_TIMEOUT})
                return False
            except Exception as e:
                error('API connectivity error', module='session', context={'error': str(e), 'api_url': config.get_api_url()}, exc_info=True)
                return False

    @monitor_performance
    def fetch_session_from_api(self, api_key: Optional[str]=None, user_type: Optional[str]=None) -> Optional[Dict[str, Any]]:
        """Always fetch fresh session data from API"""
        with operation('fetch_session_from_api', module='session'):
            debug('Fetching fresh session data from API', module='session')
            try:
                headers = {'Content-Type': 'application/json'}
                if api_key:
                    headers['X-API-Key'] = api_key
                if user_type == 'guest' and (not api_key):
                    credentials = self.load_credentials()
                    if credentials and credentials.get('device_id'):
                        headers['X-Device-ID'] = credentials.get('device_id')
                        guest_response = requests.get(get_api_endpoint('/guest/status'), headers=headers, timeout=config.REQUEST_TIMEOUT)
                        if guest_response.status_code == 200:
                            guest_data = guest_response.json()
                            if guest_data.get('success'):
                                guest_info = guest_data.get('data', {})
                                session_data = {'authenticated': True, 'api_key': guest_info.get('api_key'), 'user_type': 'guest', 'device_id': guest_info.get('device_id'), 'expires_at': guest_info.get('expires_at'), 'daily_limit': guest_info.get('daily_limit', 50), 'requests_today': guest_info.get('requests_today', 0), 'fetched_at': datetime.now().isoformat(), 'api_version': config.API_VERSION, 'api_url': config.get_api_url()}
                                info('Guest session data fetched successfully', module='session')
                                return session_data
                response = requests.get(get_api_endpoint('/auth/status'), headers=headers, timeout=config.REQUEST_TIMEOUT)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        auth_data = data.get('data', {})
                        if auth_data.get('authenticated'):
                            session_data = {'authenticated': True, 'api_key': api_key, 'user_type': auth_data.get('user_type'), 'fetched_at': datetime.now().isoformat(), 'api_version': config.API_VERSION, 'api_url': config.get_api_url()}
                            if auth_data.get('user_type') == 'registered':
                                session_data['user_info'] = auth_data.get('user', {})
                            elif auth_data.get('user_type') == 'guest':
                                guest_info = auth_data.get('guest', {})
                                session_data.update({'device_id': guest_info.get('device_id'), 'expires_at': guest_info.get('expires_at'), 'daily_limit': guest_info.get('daily_limit', 50), 'requests_today': guest_info.get('requests_today', 0)})
                            session_user_type = session_data.get('user_type', 'unknown')
                            info('Fresh session data fetched successfully', module='session', context={'user_type': session_user_type})
                            return session_data
                        else:
                            warning('API reports not authenticated', module='session')
                            return None
                    else:
                        error('API request failed', module='session', context={'response_data': data})
                        return None
                else:
                    error('API validation failed', module='session', context={'status_code': response.status_code})
                    return None
            except requests.exceptions.ConnectionError:
                warning('Cannot connect to API server', module='session', context={'api_url': config.get_api_url()})
                return None
            except requests.exceptions.Timeout:
                warning('API request timeout', module='session', context={'timeout': config.REQUEST_TIMEOUT})
                return None
            except Exception as e:
                error('API fetch error', module='session', context={'error': str(e)}, exc_info=True)
                return None

    @monitor_performance
    def get_fresh_session(self) -> Optional[Dict[str, Any]]:
        """Get fresh session - either from saved credentials or force new auth"""
        with operation('get_fresh_session', module='session'):
            debug('Getting fresh session data', module='session')
            if is_strict_mode() and (not self.check_api_connectivity()):
                warning('API not available in strict mode', module='session')
                return None
            credentials = self.load_credentials()
            if credentials and credentials.get('api_key'):
                api_key = credentials.get('api_key')
                user_type = credentials.get('user_type')
                debug('Found saved credentials', module='session', context={'user_type': user_type})
                session_data = self.fetch_session_from_api(api_key, user_type)
                if session_data:
                    if not session_data.get('device_id') and credentials.get('device_id'):
                        session_data['device_id'] = credentials.get('device_id')
                    return session_data
                else:
                    warning('Saved credentials are invalid, clearing', module='session')
                    self.clear_credentials()
                    return None
            else:
                debug('No saved credentials found', module='session')
                return None

    def save_session_credentials(self, session_data: Dict[str, Any]) -> bool:
        """Save session credentials after successful authentication"""
        if session_data and session_data.get('authenticated'):
            return self.save_credentials_only(session_data)
        else:
            warning('Cannot save credentials - session not authenticated', module='session')
            return False

    def is_api_available(self) -> bool:
        """Check if API server is available"""
        return self.check_api_connectivity()

    @monitor_performance
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session info (always fresh from API)"""
        with operation('get_session_info', module='session'):
            credentials = self.load_credentials()
            if not credentials:
                return {'status': 'No credentials found', 'first_time_user': self.is_first_time_user()}
            if not self.check_api_connectivity():
                return {'status': 'API not available', 'api_url': config.get_api_url(), 'strict_mode': is_strict_mode(), 'first_time_user': False}
            session_data = self.fetch_session_from_api(credentials.get('api_key'), credentials.get('user_type'))
            if not session_data:
                return {'status': 'Credentials invalid or API error', 'api_url': config.get_api_url(), 'strict_mode': is_strict_mode(), 'first_time_user': False}
            info_data = {'user_type': session_data.get('user_type', 'unknown'), 'authenticated': session_data.get('authenticated', False), 'has_api_key': bool(session_data.get('api_key')), 'fetched_at': session_data.get('fetched_at', 'unknown'), 'api_version': session_data.get('api_version', 'unknown'), 'api_url': config.get_api_url(), 'strict_mode': is_strict_mode(), 'api_available': True, 'first_time_user': False}
            if session_data.get('user_type') == 'guest':
                info_data.update({'expires_at': session_data.get('expires_at', 'unknown'), 'daily_limit': session_data.get('daily_limit', 'unknown'), 'requests_today': session_data.get('requests_today', 0)})
            if session_data.get('user_type') == 'registered':
                user_info = session_data.get('user_info', {})
                info_data.update({'username': user_info.get('username', 'unknown'), 'credit_balance': user_info.get('credit_balance', 0)})
            debug('Session info compiled', module='session', context={'user_type': info_data.get('user_type'), 'authenticated': info_data.get('authenticated')})
            return info_data

    def clear_session(self) -> bool:
        """Clear all session data (credentials)"""
        with operation('clear_session', module='session'):
            result = self.clear_credentials()
            info('Session cleared', module='session', context={'success': result})
            return result

    def get_api_headers(self, session_data: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Get API headers for authenticated requests"""
        headers = {'Content-Type': 'application/json'}
        if session_data and session_data.get('api_key'):
            headers['X-API-Key'] = session_data['api_key']
        debug('Generated API headers', module='session', context={'has_api_key': bool(session_data and session_data.get('api_key'))})
        return headers

    @monitor_performance
    def make_authenticated_request(self, session_data: Dict[str, Any], method: str, endpoint: str, **kwargs) -> Optional[requests.Response]:
        """Make authenticated API request"""
        with operation(f'authenticated_request_{method.lower()}', module='session'):
            try:
                headers = kwargs.get('headers', {})
                headers.update(self.get_api_headers(session_data))
                kwargs['headers'] = headers
                url = get_api_endpoint(endpoint)
                response = getattr(requests, method.lower())(url, **kwargs)
                debug('Authenticated request completed', module='session', context={'method': method, 'endpoint': endpoint, 'status_code': response.status_code})
                return response
            except Exception as e:
                error('Authenticated request error', module='session', context={'method': method, 'endpoint': endpoint, 'error': str(e)}, exc_info=True)
                return None

def is_strict_mode() -> bool:
    """Check if strict API mode is enabled"""
    strict = config.REQUIRE_API_CONNECTION
    debug('Checked strict mode', module='config', context={'strict_mode': strict})
    return strict

class ConnectionPool:
    """Simple connection pool for HTTP requests - optimized"""

    def __init__(self, max_connections=5):
        self.max_connections = max_connections
        self._session = None
        self._lock = threading.RLock()
        self._creation_time = None

    @monitor_performance
    def get_session(self):
        if self._session is None:
            with self._lock:
                if self._session is None:
                    with operation('create_session'):
                        requests = _lazy_imports.get_requests()
                        self._session = requests.Session()
                        self._creation_time = datetime.now()
                        adapter = requests.adapters.HTTPAdapter(pool_connections=self.max_connections, pool_maxsize=self.max_connections, max_retries=0)
                        self._session.mount('http://', adapter)
                        self._session.mount('https://', adapter)
                        logger.debug('HTTP session created with connection pooling')
        return self._session

    def close(self):
        if self._session:
            try:
                self._session.close()
                if self._creation_time:
                    duration = (datetime.now() - self._creation_time).total_seconds()
                    logger.debug(f'HTTP session closed after {duration:.2f} seconds')
                self._session = None
                self._creation_time = None
            except Exception as e:
                logger.error(f'Error closing HTTP session: {e}')

def operation(name: str, module: Optional[str]=None, **kwargs):
    return logger.operation(name, module, **kwargs)

class SplashAuth:
    """Splash screen with optimized performance and preserved security"""

    def __init__(self, is_first_time_user=False):
        self.current_screen = 'welcome'
        self.is_first_time_user = is_first_time_user
        self.session_data = {'user_type': None, 'api_key': None, 'device_id': None, 'user_info': {}, 'authenticated': False, 'expires_at': None}
        self.context_created = False
        self.pending_email = None
        self._connection_pool = ConnectionPool()
        self._ui_cache = UICache()
        self._device_id_cache = None
        self._hardware_info_cache = None
        self._api_status_cache = {'status': None, 'expires': None}
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix='SplashAuth')
        self._auth_lock = threading.RLock()
        self._precompute_device_info()
        logger.info('Splash Auth initialized with optimizations', context={'api_url': config.get_api_url(), 'strict_mode': is_strict_mode(), 'first_time_user': is_first_time_user})

    def _precompute_device_info(self):
        """Pre-compute device information in background - optimized"""

        def _compute():
            try:
                with operation('precompute_device_info'):
                    self._device_id_cache = self._generate_device_id_internal()
                    self._hardware_info_cache = self._get_hardware_info_internal()
                    logger.debug('Device information precomputed successfully')
            except Exception as e:
                logger.error('Failed to precompute device info', context={'error': str(e)}, exc_info=True)
        self._executor.submit(_compute)

    @lru_cache(maxsize=1)
    def _generate_device_id_internal(self) -> str:
        """Generate unique device ID - cached and optimized"""
        try:
            with operation('generate_device_id'):
                mac_address = ':'.join(['{:02x}'.format(uuid.getnode() >> elements & 255) for elements in range(0, 2 * 6, 2)][::-1])
                system_info = f'{platform.system()}-{platform.node()}-{mac_address}'
                device_hash = hashlib.sha256(system_info.encode()).hexdigest()
                device_id = f'desktop_{device_hash[:16]}'
                logger.debug(f'Device ID generated: {device_id}')
                return device_id
        except Exception as e:
            logger.warning(f'Error generating device ID, using fallback: {e}')
            fallback_id = f'desktop_{uuid.uuid4().hex[:16]}'
            logger.debug(f'Fallback device ID: {fallback_id}')
            return fallback_id

    @lru_cache(maxsize=1)
    def _get_hardware_info_internal(self) -> Dict[str, Any]:
        """Get hardware fingerprint - cached and optimized"""
        try:
            with operation('get_hardware_info'):
                hardware_info = {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine(), 'processor': platform.processor(), 'node': platform.node(), 'timestamp': datetime.now().isoformat()}
                logger.debug('Hardware information collected', context={'system': hardware_info['system']})
                return hardware_info
        except Exception as e:
            logger.error(f'Error collecting hardware info: {e}', exc_info=True)
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def generate_device_id(self) -> str:
        """Get cached device ID - optimized"""
        if self._device_id_cache is None:
            self._device_id_cache = self._generate_device_id_internal()
        return self._device_id_cache

    def get_hardware_info(self) -> Dict[str, Any]:
        """Get cached hardware info - optimized"""
        if self._hardware_info_cache is None:
            self._hardware_info_cache = self._get_hardware_info_internal()
        return self._hardware_info_cache

    def _is_api_cache_valid(self) -> bool:
        """Check if API status cache is still valid"""
        if self._api_status_cache['expires'] is None:
            return False
        return datetime.now() < self._api_status_cache['expires']

    @monitor_performance
    def check_api_connectivity(self) -> bool:
        """Check API connectivity with caching - optimized"""
        if self._is_api_cache_valid():
            return self._api_status_cache['status']
        try:
            with operation('api_connectivity_check'):
                logger.info('Checking API connectivity...')
                session = self._connection_pool.get_session()
                response = session.get(get_api_endpoint('/health'), timeout=config.CONNECTION_TIMEOUT)
                status = response.status_code == 200
                self._api_status_cache = {'status': status, 'expires': datetime.now() + timedelta(seconds=30)}
                if status:
                    logger.info('API server is available', context={'api_url': config.get_api_url()})
                else:
                    logger.warning(f'API server returned status {response.status_code}')
                return status
        except Exception as e:
            logger.error('API connectivity error', context={'error': str(e)}, exc_info=True)
            self._api_status_cache = {'status': False, 'expires': datetime.now() + timedelta(seconds=10)}
            return False

    def _get_dpg(self):
        """Get DearPyGui with lazy loading"""
        return _lazy_imports.get_dpg()

    def _create_ui_component(self, component_type: str, **kwargs) -> Any:
        """Create UI component with caching - optimized"""
        cache_key = f'{component_type}_{hash(str(sorted(kwargs.items())))}'
        if cache_key in self._ui_cache.components:
            return self._ui_cache.components[cache_key]
        try:
            dpg = self._get_dpg()
            component = getattr(dpg, component_type)(**kwargs)
            self._ui_cache.components[cache_key] = component
            return component
        except Exception as e:
            logger.error(f'Error creating UI component {component_type}: {e}')
            return None

    @monitor_performance
    def show_api_error_screen(self):
        """Show API connection error screen - optimized"""
        try:
            with operation('show_api_error_screen'):
                self.clear_content()
                dpg = self._get_dpg()
                parent = 'content_container'
                self.safe_add_spacer(30, parent)
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_spacer(width=100)
                    dpg.add_text('🚫 API Connection Error', color=[255, 100, 100])
                self.safe_add_spacer(30, parent)
                with dpg.child_window(width=460, height=350, border=True, parent=parent):
                    dpg.add_spacer(height=30)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_text('Cannot connect to Fincept API server', color=[255, 150, 150])
                    dpg.add_spacer(height=20)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_text(f'API URL: {config.get_api_url()}', color=[200, 200, 200])
                    dpg.add_spacer(height=15)
                    error_messages = ['• Check if the API server is running', '• Verify the API URL is correct', '• Check your internet connection', '• Ensure firewall is not blocking the connection']
                    for msg in error_messages:
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(width=50)
                            dpg.add_text(msg, color=[200, 200, 200])
                        dpg.add_spacer(height=5)
                    dpg.add_spacer(height=30)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_button(label='🔄 Retry Connection', width=150, callback=self.retry_api_connection)
                        dpg.add_spacer(width=20)
                        dpg.add_button(label='❌ Exit', width=100, callback=self.close_splash_error)
                logger.debug('API error screen displayed')
        except Exception as e:
            logger.error(f'Error showing API error screen: {e}', exc_info=True)

    def retry_api_connection(self, *args, **kwargs):
        """Retry API connection with cache invalidation - optimized"""
        try:
            with operation('retry_api_connection'):
                self._api_status_cache = {'status': None, 'expires': None}
                logger.info('Retrying API connection...')
                if self.check_api_connectivity():
                    logger.info('API connection successful on retry')
                    self.create_welcome_screen()
                else:
                    logger.warning('API connection failed on retry')
                    self.show_api_error_screen()
        except Exception as e:
            logger.error(f'Error during API retry: {e}', exc_info=True)

    def close_splash_error(self, *args, **kwargs):
        """Close splash with error"""
        try:
            logger.info('Closing splash due to API error')
            dpg = self._get_dpg()
            dpg.stop_dearpygui()
        except Exception as e:
            logger.error(f'Error closing splash: {e}', exc_info=True)

    @monitor_performance
    def show_splash(self) -> Dict[str, Any]:
        """Show splash screen with performance optimizations"""
        try:
            with operation('show_splash'):
                logger.info('Creating splash screen with optimizations', context={'first_time_user': self.is_first_time_user})
                dpg = self._get_dpg()
                if not self.context_created:
                    dpg.create_context()
                    self.context_created = True
                    logger.debug('DearPyGui context created')
                api_future = None
                if is_strict_mode():
                    api_future = self._executor.submit(self.check_api_connectivity)
                    logger.debug('API connectivity check started in background')
                with dpg.window(tag='splash_window', label='Fincept Authentication', width=500, height=600, no_resize=True, no_move=True, no_collapse=True, no_close=True):
                    with dpg.group(tag='content_container'):
                        if is_strict_mode() and api_future:
                            try:
                                api_available = api_future.result(timeout=config.CONNECTION_TIMEOUT)
                                if not api_available:
                                    logger.warning('API not available in strict mode')
                                    self.show_api_error_screen()
                                else:
                                    logger.info('API available, showing welcome screen')
                                    self.create_welcome_screen()
                            except Exception as e:
                                logger.error(f'Error checking API availability: {e}', exc_info=True)
                                self.show_api_error_screen()
                        else:
                            self.create_welcome_screen()
                title = 'Fincept Terminal - Welcome!' if self.is_first_time_user else 'Fincept Terminal - Authentication'
                dpg.create_viewport(title=title, width=520, height=640, resizable=False)
                dpg.setup_dearpygui()
                dpg.set_primary_window('splash_window', True)
                logger.info('Splash screen created successfully')
                dpg.show_viewport()
                dpg.start_dearpygui()
        except Exception as e:
            logger.error('Splash screen error', context={'error': str(e)}, exc_info=True)
            if is_strict_mode():
                return {'authenticated': False, 'error': f'Splash initialization failed: {str(e)}'}
            else:
                logger.warning('Using secure fallback for guest access')
                return {'user_type': 'guest', 'authenticated': True, 'device_id': self.generate_device_id(), 'api_key': None, 'user_info': {}, 'expires_at': None}
        return self.session_data

    @monitor_performance
    def clear_content(self):
        """Safely clear content with batching - optimized"""
        try:
            with operation('clear_content'):
                dpg = self._get_dpg()
                if dpg.does_item_exist('content_container'):
                    children = dpg.get_item_children('content_container', 1)
                    delete_count = 0
                    for child in children:
                        if dpg.does_item_exist(child):
                            dpg.delete_item(child)
                            delete_count += 1
                    logger.debug(f'Cleared {delete_count} UI elements from content container')
        except Exception as e:
            logger.error('Error clearing content', context={'error': str(e)}, exc_info=True)

    def safe_add_spacer(self, height=10, parent='content_container'):
        """Safely add spacer - optimized with error handling"""
        try:
            dpg = self._get_dpg()
            if dpg.does_item_exist(parent):
                dpg.add_spacer(height=height, parent=parent)
        except Exception as e:
            logger.debug('Could not add spacer', context={'height': height, 'error': str(e)})

    @monitor_performance
    def create_welcome_screen(self):
        """Create welcome screen with optimized rendering"""
        try:
            with operation('create_welcome_screen'):
                self.clear_content()
                dpg = self._get_dpg()
                parent = 'content_container'
                self.safe_add_spacer(20, parent)
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_spacer(width=80)
                    dpg.add_text('🚀 FINCEPT', color=[255, 215, 0])
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_spacer(width=120)
                    dpg.add_text('FINANCIAL TERMINAL', color=[200, 200, 200])
                self.safe_add_spacer(10, parent)
                if self.is_first_time_user:
                    with dpg.group(horizontal=True, parent=parent):
                        dpg.add_spacer(width=140)
                        dpg.add_text('👋 Welcome to Fincept!', color=[100, 255, 100])
                else:
                    with dpg.group(horizontal=True, parent=parent):
                        dpg.add_spacer(width=120)
                        dpg.add_text('🔄 Session Expired - Please Sign In', color=[255, 255, 100])
                self.safe_add_spacer(20, parent)
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_spacer(width=60)
                    dpg.add_text(f'🌐 API: {config.get_api_url()}', color=[100, 255, 100])
                self.safe_add_spacer(30, parent)
                self.create_auth_cards(parent)
                self.safe_add_spacer(30, parent)
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_spacer(width=150)
                    mode_text = 'First Time' if self.is_first_time_user else 'Returning User'
                    dpg.add_text(f'API v{config.API_VERSION} - {mode_text}', color=[100, 100, 100])
                logger.debug('Welcome screen created successfully')
        except Exception as e:
            logger.error(f'Error creating welcome screen: {e}', exc_info=True)

    def create_auth_cards(self, parent):
        """Create authentication cards with optimized layout"""
        try:
            if self.is_first_time_user:
                self.create_guest_card(parent, emphasized=True)
                self.safe_add_spacer(15, parent)
                self.create_signin_card(parent, emphasized=False)
                self.safe_add_spacer(15, parent)
                self.create_signup_card(parent, emphasized=False)
            else:
                self.create_signin_card(parent, emphasized=True)
                self.safe_add_spacer(15, parent)
                self.create_guest_card(parent, emphasized=False)
                self.safe_add_spacer(15, parent)
                self.create_signup_card(parent, emphasized=False)
            logger.debug('Authentication cards created')
        except Exception as e:
            logger.error(f'Error creating auth cards: {e}', exc_info=True)

    def create_signin_card(self, parent, emphasized=False):
        """Create sign in card - optimized"""
        try:
            dpg = self._get_dpg()
            with dpg.child_window(width=460, height=100, border=True, parent=parent):
                if emphasized:
                    dpg.add_spacer(height=5)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=15)
                        dpg.add_text('🔑 RECOMMENDED', color=[100, 255, 100])
                dpg.add_spacer(height=10)
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=15)
                    dpg.add_text('🔐 Sign In', color=[100, 255, 100])
                dpg.add_spacer(height=5)
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=15)
                    text = 'Welcome back! Access your account' if not self.is_first_time_user else 'Access your account with permanent API key'
                    dpg.add_text(text, color=[200, 200, 200])
                dpg.add_spacer(height=10)
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=350)
                    dpg.add_button(label='Sign In', width=100, callback=self.go_to_login)
        except Exception as e:
            logger.error(f'Error creating signin card: {e}', exc_info=True)

    def create_guest_card(self, parent, emphasized=False):
        """Create guest card - optimized"""
        try:
            dpg = self._get_dpg()
            with dpg.child_window(width=460, height=100, border=True, parent=parent):
                if emphasized:
                    dpg.add_spacer(height=5)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=15)
                        dpg.add_text('⭐ QUICK START', color=[255, 255, 100])
                dpg.add_spacer(height=10)
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=15)
                    dpg.add_text('🎯 Guest Access', color=[255, 255, 100])
                dpg.add_spacer(height=5)
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=15)
                    text = '🚀 Try Fincept instantly! No signup required' if self.is_first_time_user else '50 requests/day with temporary API key'
                    dpg.add_text(text, color=[200, 200, 200])
                dpg.add_spacer(height=10)
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=280)
                    button_text = '🚀 Try Now!' if self.is_first_time_user else 'Continue as Guest'
                    button_width = 170 if self.is_first_time_user else 150
                    dpg.add_button(label=button_text, width=button_width, callback=self.setup_guest_access)
        except Exception as e:
            logger.error(f'Error creating guest card: {e}', exc_info=True)

    def create_signup_card(self, parent, emphasized=False):
        """Create signup card - optimized"""
        try:
            dpg = self._get_dpg()
            with dpg.child_window(width=460, height=100, border=True, parent=parent):
                dpg.add_spacer(height=10)
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=15)
                    dpg.add_text('✨ Create Account', color=[100, 150, 255])
                dpg.add_spacer(height=5)
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=15)
                    text = '🎁 Join Fincept for unlimited access'
                    dpg.add_text(text, color=[200, 200, 200])
                dpg.add_spacer(height=10)
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=340)
                    dpg.add_button(label='Sign Up', width=110, callback=self.go_to_signup)
        except Exception as e:
            logger.error(f'Error creating signup card: {e}', exc_info=True)

    @monitor_performance
    def create_login_screen(self):
        """Create login screen with optimized layout"""
        try:
            with operation('create_login_screen'):
                self.clear_content()
                dpg = self._get_dpg()
                parent = 'content_container'
                self.safe_add_spacer(30, parent)
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_spacer(width=180)
                    dpg.add_text('🔐 Sign In', color=[100, 255, 100])
                self.safe_add_spacer(30, parent)
                with dpg.child_window(width=460, height=350, border=True, parent=parent):
                    dpg.add_spacer(height=30)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_text('Email Address:')
                    dpg.add_spacer(height=5)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_input_text(tag='login_email', width=400, hint='Enter your email')
                    dpg.add_spacer(height=20)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_text('Password:')
                    dpg.add_spacer(height=5)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_input_text(tag='login_password', width=400, password=True, hint='Enter password')
                    dpg.add_spacer(height=20)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_text('', tag='login_status', color=[255, 100, 100])
                    dpg.add_spacer(height=20)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_button(label='🔐 Sign In', width=120, callback=self.attempt_login)
                        dpg.add_spacer(width=20)
                        dpg.add_button(label='⬅️ Back', width=120, callback=self.go_to_welcome)
                logger.debug('Login screen created successfully')
        except Exception as e:
            logger.error(f'Error creating login screen: {e}', exc_info=True)

    @monitor_performance
    def create_signup_screen(self):
        """Create signup screen - optimized"""
        try:
            with operation('create_signup_screen'):
                self.clear_content()
                dpg = self._get_dpg()
                parent = 'content_container'
                self.safe_add_spacer(20, parent)
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_spacer(width=170)
                    dpg.add_text('✨ Create Account', color=[100, 150, 255])
                self.safe_add_spacer(20, parent)
                with dpg.child_window(width=460, height=450, border=True, parent=parent):
                    dpg.add_spacer(height=20)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_text('Username:')
                    dpg.add_spacer(height=5)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_input_text(tag='signup_username', width=400, hint='Choose username')
                    dpg.add_spacer(height=15)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_text('Email Address:')
                    dpg.add_spacer(height=5)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_input_text(tag='signup_email', width=400, hint='Enter email')
                    dpg.add_spacer(height=15)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_text('Password:')
                    dpg.add_spacer(height=5)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_input_text(tag='signup_password', width=400, password=True, hint='Create password')
                    dpg.add_spacer(height=15)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_text('Confirm Password:')
                    dpg.add_spacer(height=5)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_input_text(tag='signup_confirm_password', width=400, password=True, hint='Confirm password')
                    dpg.add_spacer(height=20)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_text('', tag='signup_status', color=[255, 100, 100])
                    dpg.add_spacer(height=20)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_button(label='✨ Create Account', width=140, callback=self.attempt_signup)
                        dpg.add_spacer(width=20)
                        dpg.add_button(label='⬅️ Back', width=120, callback=self.go_to_welcome)
                logger.debug('Signup screen created successfully')
        except Exception as e:
            logger.error(f'Error creating signup screen: {e}', exc_info=True)

    @monitor_performance
    def create_otp_screen(self):
        """Create OTP verification screen - optimized"""
        try:
            with operation('create_otp_screen'):
                self.clear_content()
                dpg = self._get_dpg()
                parent = 'content_container'
                self.safe_add_spacer(50, parent)
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_spacer(width=160)
                    dpg.add_text('📧 Email Verification', color=[255, 255, 100])
                self.safe_add_spacer(30, parent)
                with dpg.child_window(width=460, height=300, border=True, parent=parent):
                    dpg.add_spacer(height=30)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_text('Enter the 6-digit code sent to your email:', color=[200, 200, 200])
                    dpg.add_spacer(height=20)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_text('Verification Code:')
                    dpg.add_spacer(height=10)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=130)
                        dpg.add_input_text(tag='otp_code', width=200, hint='6-digit code')
                    dpg.add_spacer(height=20)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_text('', tag='otp_status', color=[255, 100, 100])
                    dpg.add_spacer(height=20)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=80)
                        dpg.add_button(label='✅ Verify Code', width=120, callback=self.verify_otp)
                        dpg.add_spacer(width=20)
                        dpg.add_button(label='⬅️ Back', width=120, callback=self.go_to_signup)
                logger.debug('OTP screen created successfully')
        except Exception as e:
            logger.error(f'Error creating OTP screen: {e}', exc_info=True)

    @monitor_performance
    def create_guest_setup_screen(self):
        """Create guest setup screen - optimized"""
        try:
            with operation('create_guest_setup_screen'):
                self.clear_content()
                dpg = self._get_dpg()
                parent = 'content_container'
                self.safe_add_spacer(40, parent)
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_spacer(width=130)
                    dpg.add_text('🎯 Setting up Guest Access', color=[255, 255, 100])
                self.safe_add_spacer(30, parent)
                with dpg.child_window(width=460, height=350, border=True, parent=parent):
                    dpg.add_spacer(height=30)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_text('Guest Features:', color=[100, 255, 100])
                    dpg.add_spacer(height=15)
                    features = ['📈 Financial market data access', '💹 Real-time stock prices & forex', '🔢 50 API requests per day', '⏰ 24-hour access period', '🔑 Temporary API key authentication']
                    for feature in features:
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(width=50)
                            dpg.add_text(feature, color=[200, 255, 200])
                        dpg.add_spacer(height=5)
                    dpg.add_spacer(height=20)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=30)
                        dpg.add_text('Status: Creating guest API key...', tag='guest_status', color=[255, 255, 100])
                    dpg.add_spacer(height=20)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=130)
                        dpg.add_button(label='🚀 Continue to Terminal', width=200, callback=self.complete_guest_setup, show=False, tag='guest_continue_btn')
                self._executor.submit(self.create_guest_session)
                logger.debug('Guest setup screen created successfully')
        except Exception as e:
            logger.error(f'Error creating guest setup screen: {e}', exc_info=True)

    def go_to_welcome(self, *args, **kwargs):
        logger.debug('Navigating to welcome screen')
        self.current_screen = 'welcome'
        self.create_welcome_screen()

    def go_to_login(self, *args, **kwargs):
        logger.debug('Navigating to login screen')
        self.current_screen = 'login'
        self.create_login_screen()

    def go_to_signup(self, *args, **kwargs):
        logger.debug('Navigating to signup screen')
        self.current_screen = 'signup'
        self.create_signup_screen()

    @monitor_performance
    def _make_api_request(self, method: str, endpoint: str, data: Optional[Dict]=None, headers: Optional[Dict]=None, timeout: Optional[int]=None) -> Tuple[bool, Dict]:
        """Optimized API request with connection pooling"""
        try:
            with operation('api_request', context={'method': method, 'endpoint': endpoint}):
                session = self._connection_pool.get_session()
                timeout = timeout or config.REQUEST_TIMEOUT
                request_start = time.time()
                if method.upper() == 'GET':
                    response = session.get(get_api_endpoint(endpoint), headers=headers, timeout=timeout)
                elif method.upper() == 'POST':
                    response = session.post(get_api_endpoint(endpoint), json=data, headers=headers, timeout=timeout)
                else:
                    logger.error(f'Unsupported HTTP method: {method}')
                    return (False, {'error': 'Unsupported HTTP method'})
                request_duration = time.time() - request_start
                if response.status_code == 200:
                    logger.debug(f'API request successful', context={'method': method, 'endpoint': endpoint, 'duration_ms': f'{request_duration * 1000:.2f}', 'status_code': response.status_code})
                    return (True, response.json())
                else:
                    logger.warning(f'API request failed', context={'method': method, 'endpoint': endpoint, 'status_code': response.status_code, 'duration_ms': f'{request_duration * 1000:.2f}'})
                    return (False, {'error': f'HTTP {response.status_code}', 'status_code': response.status_code})
        except Exception as e:
            logger.error(f'API request exception', context={'method': method, 'endpoint': endpoint, 'error': str(e)}, exc_info=True)
            return (False, {'error': str(e)})

    @monitor_performance
    def attempt_login(self, *args, **kwargs):
        """Attempt user login with optimized API calls"""
        with self._auth_lock:
            try:
                with operation('login_attempt'):
                    dpg = self._get_dpg()
                    email = dpg.get_value('login_email') if dpg.does_item_exist('login_email') else ''
                    password = dpg.get_value('login_password') if dpg.does_item_exist('login_password') else ''
                    if not email or not password:
                        self.update_status('login_status', 'Please fill in all fields')
                        return
                    logger.info('Attempting user login', context={'email': email})
                    self.update_status('login_status', '🔐 Signing in...')
                    success, response_data = self._make_api_request('POST', '/auth/login', {'email': email, 'password': password})
                    if success and response_data.get('success'):
                        data = response_data.get('data', {})
                        self.session_data.update({'user_type': 'registered', 'api_key': data.get('api_key'), 'authenticated': True, 'device_id': self.generate_device_id()})
                        self._executor.submit(self.fetch_user_profile)
                        self.update_status('login_status', '✅ Login successful! Opening terminal...')
                        logger.info('User login successful', context={'user_type': 'registered'})
                        threading.Timer(1.0, self.close_splash_success).start()
                    else:
                        error_msg = response_data.get('message', 'Login failed')
                        self.update_status('login_status', f'❌ {error_msg}')
                        logger.warning(f'Login failed: {error_msg}')
            except Exception as e:
                error_msg = f'❌ Error: {str(e)}'
                self.update_status('login_status', error_msg)
                logger.error('Login error', context={'error': str(e)}, exc_info=True)

    @monitor_performance
    def attempt_signup(self, *args, **kwargs):
        """Attempt user registration with optimized validation"""
        with self._auth_lock:
            try:
                with operation('signup_attempt'):
                    dpg = self._get_dpg()
                    username = dpg.get_value('signup_username') if dpg.does_item_exist('signup_username') else ''
                    email = dpg.get_value('signup_email') if dpg.does_item_exist('signup_email') else ''
                    password = dpg.get_value('signup_password') if dpg.does_item_exist('signup_password') else ''
                    confirm_password = dpg.get_value('signup_confirm_password') if dpg.does_item_exist('signup_confirm_password') else ''
                    if not all([username, email, password, confirm_password]):
                        self.update_status('signup_status', 'Please fill in all fields')
                        return
                    if password != confirm_password:
                        self.update_status('signup_status', 'Passwords do not match')
                        return
                    if len(password) < 6:
                        self.update_status('signup_status', 'Password must be at least 6 characters')
                        return
                    logger.info('Attempting user registration', context={'username': username, 'email': email})
                    self.update_status('signup_status', '✨ Creating account...')
                    success, response_data = self._make_api_request('POST', '/auth/register', {'username': username, 'email': email, 'password': password})
                    if success and response_data.get('success'):
                        self.pending_email = email
                        self.current_screen = 'otp_verify'
                        self.create_otp_screen()
                        logger.info('Registration successful, OTP sent', context={'email': email})
                    else:
                        error_msg = response_data.get('message', 'Registration failed')
                        self.update_status('signup_status', f'❌ {error_msg}')
                        logger.warning(f'Registration failed: {error_msg}')
            except Exception as e:
                error_msg = f'❌ Error: {str(e)}'
                self.update_status('signup_status', error_msg)
                logger.error('Signup error', context={'error': str(e)}, exc_info=True)

    @monitor_performance
    def verify_otp(self, *args, **kwargs):
        """Verify OTP code with optimized validation"""
        with self._auth_lock:
            try:
                with operation('otp_verification'):
                    dpg = self._get_dpg()
                    otp_code = dpg.get_value('otp_code') if dpg.does_item_exist('otp_code') else ''
                    if not otp_code or len(otp_code) != 6 or (not otp_code.isdigit()):
                        self.update_status('otp_status', 'Please enter valid 6-digit code')
                        return
                    logger.info('Verifying OTP code', context={'email': self.pending_email})
                    self.update_status('otp_status', '📧 Verifying...')
                    success, response_data = self._make_api_request('POST', '/auth/verify-otp', {'email': self.pending_email, 'otp': otp_code})
                    if success and response_data.get('success'):
                        data = response_data.get('data', {})
                        self.session_data.update({'user_type': 'registered', 'api_key': data.get('api_key'), 'authenticated': True, 'device_id': self.generate_device_id()})
                        self._executor.submit(self.fetch_user_profile)
                        self.update_status('otp_status', '✅ Success! Opening terminal...')
                        logger.info('OTP verification successful')
                        threading.Timer(1.0, self.close_splash_success).start()
                    else:
                        error_msg = response_data.get('message', 'Verification failed')
                        self.update_status('otp_status', f'❌ {error_msg}')
                        logger.warning(f'OTP verification failed: {error_msg}')
            except Exception as e:
                error_msg = f'❌ Error: {str(e)}'
                self.update_status('otp_status', error_msg)
                logger.error('OTP verification error', context={'error': str(e)}, exc_info=True)

    @monitor_performance
    def setup_guest_access(self, *args, **kwargs):
        """Setup guest access with background processing"""
        try:
            with operation('setup_guest_access'):
                logger.info('Setting up guest access')
                self.current_screen = 'guest_setup'
                self.create_guest_setup_screen()
        except Exception as e:
            logger.error('Error setting up guest access', context={'error': str(e)}, exc_info=True)
            if is_strict_mode():
                self.update_status('guest_status', f'❌ Guest setup failed: {str(e)}')
            else:
                logger.warning('Using secure fallback for guest access')
                self.session_data.update({'user_type': 'guest', 'device_id': self.generate_device_id(), 'authenticated': True, 'api_key': None})
                self.close_splash_success()

    @monitor_performance
    def create_guest_session(self):
        """Create guest session with optimized API integration"""
        try:
            with operation('create_guest_session'):
                device_id = self.generate_device_id()
                hardware_info = self.get_hardware_info()
                logger.info('Creating guest session', context={'device_id': device_id})

                def update_ui_safe(message):
                    try:
                        self.update_status('guest_status', message)
                    except:
                        logger.debug('Could not update UI status (UI may be destroyed)')
                update_ui_safe('🌐 Checking for existing session...')
                from fincept_terminal.utils.APIClient.api_client import FinceptAPIClient
                temp_session = {'user_type': 'guest', 'device_id': device_id}
                api_client = FinceptAPIClient(temp_session)
                result = api_client.get_or_create_guest_session(device_id=device_id, device_name=f'Fincept Terminal - {platform.system()}', platform='desktop', hardware_info=hardware_info)
                if result['success']:
                    guest_data = result.get('data', {})
                    message = result.get('message', 'Session ready')
                    with self._auth_lock:
                        self.session_data.update({'user_type': 'guest', 'device_id': device_id, 'api_key': guest_data.get('api_key') or guest_data.get('temp_api_key'), 'authenticated': True, 'expires_at': guest_data.get('expires_at'), 'daily_limit': guest_data.get('daily_limit', 50), 'requests_today': guest_data.get('requests_today', 0)})
                    update_ui_safe(f'✅ {message}!')
                    logger.info('Guest session created successfully', context={'api_key_present': bool(self.session_data.get('api_key')), 'daily_limit': guest_data.get('daily_limit', 50)})
                    try:
                        dpg = self._get_dpg()
                        if dpg.does_item_exist('guest_continue_btn'):
                            dpg.show_item('guest_continue_btn')
                    except:
                        logger.debug('Could not show continue button')
                else:
                    error_msg = result.get('error', 'Unknown error')
                    update_ui_safe(f'❌ Session setup failed: {error_msg}')
                    logger.error('Guest session setup failed', context={'error': error_msg, 'device_id': device_id})
        except Exception as e:
            try:
                self.update_status('guest_status', f'❌ Guest creation failed: {str(e)}')
            except:
                pass
            logger.error('Guest session creation error', context={'error': str(e)}, exc_info=True)

    def complete_guest_setup(self, *args, **kwargs):
        """Complete guest setup"""
        logger.info('Completing guest setup')
        self.close_splash_success()

    @monitor_performance
    def fetch_user_profile(self):
        """Fetch user profile with optimized API call"""
        try:
            with operation('fetch_user_profile'):
                if not self.session_data.get('api_key'):
                    logger.warning('No API key available for profile fetch')
                    return
                success, response_data = self._make_api_request('GET', '/user/profile', headers={'X-API-Key': self.session_data['api_key']})
                if success and response_data.get('success'):
                    with self._auth_lock:
                        self.session_data['user_info'] = response_data.get('data', {})
                    logger.info('User profile fetched from API')
                else:
                    logger.warning('Failed to fetch user profile from API')
        except Exception as e:
            logger.error('Failed to fetch profile from API', context={'error': str(e)}, exc_info=True)

    def update_status(self, tag: str, message: str):
        """Thread-safe status update - optimized"""
        try:
            dpg = self._get_dpg()
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, message)
        except Exception as e:
            logger.debug('Could not update status', context={'tag': tag, 'message': message, 'error': str(e)})

    def close_splash_success(self):
        """Close splash successfully with cleanup"""
        try:
            logger.info('Closing splash screen successfully', context={'user_type': self.session_data.get('user_type')})
            dpg = self._get_dpg()
            dpg.stop_dearpygui()
        except Exception as e:
            logger.error('Error closing splash', context={'error': str(e)}, exc_info=True)

    @monitor_performance
    def cleanup(self):
        """Enhanced cleanup with resource management - optimized"""
        try:
            with operation('splash_cleanup'):
                logger.info('🧹 Cleaning up splash screen...')
                if hasattr(self, '_executor'):
                    try:
                        self._executor.shutdown(wait=True, timeout=5.0)
                        logger.debug('Thread pool shutdown completed')
                    except Exception as e:
                        logger.warning(f'Thread pool shutdown error: {e}')
                if hasattr(self, '_connection_pool'):
                    try:
                        self._connection_pool.close()
                        logger.debug('Connection pool closed')
                    except Exception as e:
                        logger.warning(f'Connection pool cleanup error: {e}')
                if hasattr(self, '_ui_cache'):
                    self._ui_cache.components.clear()
                    logger.debug('UI cache cleared')
                try:
                    self._generate_device_id_internal.cache_clear()
                    self._get_hardware_info_internal.cache_clear()
                    logger.debug('LRU caches cleared')
                except:
                    pass
                if self.context_created:
                    try:
                        dpg = self._get_dpg()
                        dpg.destroy_context()
                        self.context_created = False
                        logger.debug('DearPyGui context destroyed')
                    except Exception as e:
                        logger.warning(f'DPG context cleanup error: {e}')
                logger.info('Splash screen cleanup completed')
        except Exception as e:
            logger.error('Cleanup error', context={'error': str(e)}, exc_info=True)

    def __del__(self):
        """Destructor with resource cleanup"""
        try:
            self.cleanup()
        except:
            pass

@monitor_performance
def show_authentication_splash(is_first_time_user=False) -> Dict[str, Any]:
    """Show splash with optimized performance and preserved security"""
    splash = None
    try:
        with operation('show_authentication_splash', context={'first_time_user': is_first_time_user}):
            splash = SplashAuth(is_first_time_user=is_first_time_user)
            result = splash.show_splash()
            if is_strict_mode() and (not result.get('authenticated')):
                logger.error('Authentication failed in strict mode')
                return {'authenticated': False, 'error': 'API authentication required but failed'}
            logger.info('Authentication splash completed', context={'authenticated': result.get('authenticated'), 'user_type': result.get('user_type')})
            return result
    except Exception as e:
        logger.error('Splash error', context={'error': str(e)}, exc_info=True)
        if is_strict_mode():
            return {'authenticated': False, 'error': f'Splash failed: {str(e)}'}
        else:
            logger.warning('Using secure fallback authentication')
            device_id = splash.generate_device_id() if splash else f'desktop_{uuid.uuid4().hex[:16]}'
            return {'user_type': 'guest', 'authenticated': True, 'device_id': device_id, 'api_key': None, 'user_info': {}, 'expires_at': None}
    finally:
        if splash:
            splash.cleanup()

class HelpTab(BaseTab):
    """Bloomberg Terminal style Help and About tab"""

    def __init__(self, main_app=None):
        super().__init__(main_app)
        self.main_app = main_app
        self.scroll_position = 0
        self.BLOOMBERG_ORANGE = [255, 165, 0]
        self.BLOOMBERG_WHITE = [255, 255, 255]
        self.BLOOMBERG_RED = [255, 0, 0]
        self.BLOOMBERG_GREEN = [0, 200, 0]
        self.BLOOMBERG_YELLOW = [255, 255, 0]
        self.BLOOMBERG_GRAY = [120, 120, 120]
        self.BLOOMBERG_BLUE = [100, 150, 250]
        self.BLOOMBERG_BLACK = [0, 0, 0]
        self._cached_datetime = None
        self._datetime_cache_time = 0
        debug('HelpTab initialized', module='help', context={'main_app_available': bool(main_app)})

    def get_label(self):
        return ' Help & About'

    def _get_current_time_cached(self):
        """Get current time with caching for performance"""
        import time
        current_time = time.time()
        if current_time - self._datetime_cache_time > 5:
            self._cached_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self._datetime_cache_time = current_time
        return self._cached_datetime

    @monitor_performance
    def create_content(self):
        """Create Bloomberg-style help terminal layout"""
        with operation('create_help_content', module='help'):
            try:
                with dpg.group(horizontal=True):
                    dpg.add_text('FINCEPT', color=self.BLOOMBERG_ORANGE)
                    dpg.add_text('HELP TERMINAL', color=self.BLOOMBERG_WHITE)
                    dpg.add_text(' | ', color=self.BLOOMBERG_GRAY)
                    dpg.add_input_text(label='', default_value='Search Help Topics', width=300)
                    dpg.add_button(label='SEARCH', width=80, callback=self.search_help)
                    dpg.add_text(' | ', color=self.BLOOMBERG_GRAY)
                    dpg.add_text(self._get_current_time_cached())
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    help_functions = ['F1:ABOUT', 'F2:FEATURES', 'F3:SUPPORT', 'F4:CONTACT', 'F5:FEEDBACK', 'F6:DOCS']
                    for key in help_functions:
                        dpg.add_button(label=key, width=100, height=25, callback=lambda s, a, u, k=key: self.navigate_section(k))
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    self.create_left_help_panel()
                    self.create_center_help_panel()
                    self.create_right_help_panel()
                dpg.add_separator()
                self.create_help_status_bar()
                info('Help content created successfully', module='help')
            except Exception as e:
                error('Error creating help content', module='help', context={'error': str(e)}, exc_info=True)
                dpg.add_text('HELP TERMINAL', color=self.BLOOMBERG_ORANGE)
                dpg.add_text('Error loading help content. Please try again.')

    @monitor_performance
    def create_left_help_panel(self):
        """Create left help navigation panel"""
        with operation('create_left_help_panel', module='help'):
            with dpg.child_window(width=350, height=650, border=True):
                dpg.add_text('HELP NAVIGATOR', color=self.BLOOMBERG_ORANGE)
                dpg.add_separator()
                with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True, scrollY=True, height=300):
                    dpg.add_table_column(label='Section', width_fixed=True, init_width_or_weight=120)
                    dpg.add_table_column(label='Status', width_fixed=True, init_width_or_weight=80)
                    dpg.add_table_column(label='Action', width_fixed=True, init_width_or_weight=100)
                    help_sections = [('ABOUT FINCEPT', 'AVAILABLE', 'VIEW'), ('FEATURES', 'AVAILABLE', 'VIEW'), ('MARKET DATA', 'AVAILABLE', 'VIEW'), ('PORTFOLIO', 'AVAILABLE', 'VIEW'), ('ANALYTICS', 'AVAILABLE', 'VIEW'), ('SUPPORT', 'AVAILABLE', 'CONTACT'), ('TUTORIALS', 'COMING SOON', 'NOTIFY'), ('API DOCS', 'AVAILABLE', 'OPEN'), ('COMMUNITY', 'AVAILABLE', 'JOIN'), ('FEEDBACK', 'AVAILABLE', 'SEND')]
                    for section, status, action in help_sections:
                        with dpg.table_row():
                            dpg.add_text(section, color=self.BLOOMBERG_WHITE)
                            status_color = self.BLOOMBERG_GREEN if status == 'AVAILABLE' else self.BLOOMBERG_YELLOW
                            dpg.add_text(status, color=status_color)
                            action_color = self.BLOOMBERG_BLUE if action in ['VIEW', 'OPEN'] else self.BLOOMBERG_ORANGE
                            dpg.add_text(action, color=action_color)
                dpg.add_separator()
                dpg.add_text('HELP STATISTICS', color=self.BLOOMBERG_YELLOW)
                with dpg.table(header_row=False, borders_innerH=False, borders_outerH=False):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    stats = [('Total Help Topics:', '47'), ('Video Tutorials:', '12'), ('FAQ Articles:', '25'), ('API Endpoints:', '156')]
                    for label, value in stats:
                        with dpg.table_row():
                            dpg.add_text(label)
                            dpg.add_text(value, color=self.BLOOMBERG_WHITE)
                dpg.add_separator()
                dpg.add_text('SYSTEM STATUS', color=self.BLOOMBERG_YELLOW)
                with dpg.group(horizontal=True):
                    dpg.add_text('●', color=self.BLOOMBERG_GREEN)
                    dpg.add_text('ALL SYSTEMS OPERATIONAL', color=self.BLOOMBERG_GREEN)
                debug('Left help panel created', module='help')

    @monitor_performance
    def create_center_help_panel(self):
        """Create center help content panel"""
        with operation('create_center_help_panel', module='help'):
            with dpg.child_window(width=900, height=650, border=True):
                with dpg.tab_bar():
                    with dpg.tab(label='About'):
                        self._create_about_tab()
                    with dpg.tab(label='Features'):
                        self._create_features_tab()
                    with dpg.tab(label='Support'):
                        self._create_support_tab()
                    with dpg.tab(label='API Docs'):
                        self._create_api_docs_tab()
                debug('Center help panel created', module='help')

    def _create_about_tab(self):
        """Create about tab content"""
        dpg.add_text('ABOUT FINCEPT TERMINAL', color=self.BLOOMBERG_ORANGE)
        dpg.add_separator()
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text('Fincept Financial Terminal', color=self.BLOOMBERG_ORANGE)
                dpg.add_text('Professional Trading & Analytics Platform')
                dpg.add_spacer(height=10)
                with dpg.table(header_row=False, borders_innerH=False, borders_outerH=False):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    version_info = [('Version:', '4.2.1 Professional'), ('Build:', '20250115.1'), ('License:', 'Enterprise'), ('Data Sources:', 'Real-time'), ('API Status:', 'Connected')]
                    for label, value in version_info:
                        with dpg.table_row():
                            dpg.add_text(label)
                            value_color = self.BLOOMBERG_GREEN if value in ['Enterprise', 'Real-time', 'Connected'] else self.BLOOMBERG_WHITE
                            dpg.add_text(value, color=value_color)
            with dpg.group():
                dpg.add_text('Core Features', color=self.BLOOMBERG_YELLOW)
                features = ['• Real-time market data & analytics', '• Portfolio management & tracking', '• Advanced charting & technical analysis', '• Financial news & sentiment analysis', '• Risk management tools', '• Algorithmic trading support', '• Multi-asset class coverage', '• Professional-grade security']
                for feature in features:
                    dpg.add_text(feature)
        dpg.add_spacer(height=20)
        about_text = 'Fincept Terminal is a cutting-edge financial analysis platform designed to provide real-time market data, portfolio management, and actionable insights to investors, traders, and financial professionals. Our platform integrates advanced analytics, AI-driven sentiment analysis, and the latest market trends to help you make well-informed investment decisions.'
        dpg.add_text(about_text, wrap=850, color=self.BLOOMBERG_WHITE)

    def _create_features_tab(self):
        """Create features tab content"""
        dpg.add_text('TERMINAL FEATURES & CAPABILITIES', color=self.BLOOMBERG_ORANGE)
        dpg.add_separator()
        with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True, scrollY=True, height=400):
            dpg.add_table_column(label='Feature Category', width_fixed=True, init_width_or_weight=200)
            dpg.add_table_column(label='Description', width_fixed=True, init_width_or_weight=400)
            dpg.add_table_column(label='Status', width_fixed=True, init_width_or_weight=100)
            dpg.add_table_column(label='Access Level', width_fixed=True, init_width_or_weight=120)
            features = [('Market Data', 'Real-time quotes, indices, forex, commodities', 'ACTIVE', 'ALL USERS'), ('Portfolio Mgmt', 'Track holdings, P&L, asset allocation', 'ACTIVE', 'ALL USERS'), ('Technical Analysis', 'Advanced charting, indicators, overlays', 'ACTIVE', 'PRO'), ('News & Sentiment', 'Financial news aggregation, sentiment scoring', 'ACTIVE', 'PRO'), ('Risk Analytics', 'VaR, stress testing, correlation analysis', 'ACTIVE', 'ENTERPRISE'), ('Algo Trading', 'Strategy backtesting, execution algorithms', 'BETA', 'ENTERPRISE'), ('Options Analytics', 'Greeks, volatility surface, strategies', 'ACTIVE', 'PRO'), ('Fixed Income', 'Bond analytics, yield curves, duration', 'ACTIVE', 'ENTERPRISE'), ('ESG Analytics', 'Sustainability metrics, ESG scoring', 'COMING SOON', 'PRO'), ('AI Insights', 'Machine learning predictions, pattern recognition', 'BETA', 'ENTERPRISE')]
            for feature, description, status, access in features:
                with dpg.table_row():
                    dpg.add_text(feature, color=self.BLOOMBERG_YELLOW)
                    dpg.add_text(description, color=self.BLOOMBERG_WHITE)
                    status_color = self.BLOOMBERG_GREEN if status == 'ACTIVE' else self.BLOOMBERG_YELLOW if status == 'BETA' else self.BLOOMBERG_ORANGE
                    dpg.add_text(status, color=status_color)
                    access_color = self.BLOOMBERG_GREEN if access == 'ALL USERS' else self.BLOOMBERG_BLUE if access == 'PRO' else self.BLOOMBERG_ORANGE
                    dpg.add_text(access, color=access_color)

    def _create_support_tab(self):
        """Create support tab content"""
        dpg.add_text('CUSTOMER SUPPORT & ASSISTANCE', color=self.BLOOMBERG_ORANGE)
        dpg.add_separator()
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text('Contact Information', color=self.BLOOMBERG_YELLOW)
                dpg.add_spacer(height=10)
                with dpg.table(header_row=False, borders_innerH=False, borders_outerH=False):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    contact_info = [('Email Support:', 'support@fincept.in'), ('Phone Support:', '+1 (555) 123-4567'), ('Live Chat:', 'Available 24/7'), ('Response Time:', '< 2 hours'), ('Support Hours:', '24/7/365')]
                    for label, value in contact_info:
                        with dpg.table_row():
                            dpg.add_text(label)
                            value_color = self.BLOOMBERG_BLUE if 'support@' in value else self.BLOOMBERG_GREEN if 'Available' in value or '< 2' in value or '24/7' in value else self.BLOOMBERG_WHITE
                            dpg.add_text(value, color=value_color)
            with dpg.group():
                dpg.add_text('Support Channels', color=self.BLOOMBERG_YELLOW)
                dpg.add_spacer(height=10)
                support_buttons = [('📧 Email Support', self.contact_email_support), ('💬 Live Chat', self.open_live_chat), ('📞 Phone Support', self.contact_phone_support), ('📖 Documentation', self.open_documentation), ('🎥 Video Tutorials', self.open_tutorials), ('👥 Community Forum', self.open_community), ('🐛 Report Bug', self.report_bug), ('💡 Feature Request', self.request_feature)]
                for label, callback in support_buttons:
                    dpg.add_button(label=label, callback=callback, width=200)
                    dpg.add_spacer(height=5)

    def _create_api_docs_tab(self):
        """Create API documentation tab content"""
        dpg.add_text('API DOCUMENTATION & ENDPOINTS', color=self.BLOOMBERG_ORANGE)
        dpg.add_separator()
        with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True, scrollY=True, height=500):
            dpg.add_table_column(label='Endpoint', width_fixed=True, init_width_or_weight=200)
            dpg.add_table_column(label='Method', width_fixed=True, init_width_or_weight=80)
            dpg.add_table_column(label='Description', width_fixed=True, init_width_or_weight=300)
            dpg.add_table_column(label='Rate Limit', width_fixed=True, init_width_or_weight=100)
            dpg.add_table_column(label='Auth Required', width_fixed=True, init_width_or_weight=120)
            api_endpoints = [('/api/v1/market/quotes', 'GET', 'Real-time market quotes', '1000/min', 'YES'), ('/api/v1/portfolio/holdings', 'GET', 'Portfolio holdings data', '100/min', 'YES'), ('/api/v1/news/latest', 'GET', 'Latest financial news', '500/min', 'NO'), ('/api/v1/analytics/technical', 'POST', 'Technical analysis calculations', '50/min', 'YES'), ('/api/v1/market/history', 'GET', 'Historical market data', '200/min', 'YES'), ('/api/v1/user/profile', 'GET', 'User profile information', '10/min', 'YES'), ('/api/v1/orders/submit', 'POST', 'Submit trading orders', '100/min', 'YES'), ('/api/v1/market/screener', 'POST', 'Stock screening criteria', '50/min', 'YES'), ('/api/v1/research/reports', 'GET', 'Research reports access', '20/min', 'YES'), ('/api/v1/alerts/manage', 'POST', 'Manage price alerts', '100/min', 'YES')]
            for endpoint, method, description, rate_limit, auth in api_endpoints:
                with dpg.table_row():
                    dpg.add_text(endpoint, color=self.BLOOMBERG_BLUE)
                    method_color = self.BLOOMBERG_GREEN if method == 'GET' else self.BLOOMBERG_ORANGE
                    dpg.add_text(method, color=method_color)
                    dpg.add_text(description, color=self.BLOOMBERG_WHITE)
                    dpg.add_text(rate_limit, color=self.BLOOMBERG_YELLOW)
                    auth_color = self.BLOOMBERG_RED if auth == 'YES' else self.BLOOMBERG_GREEN
                    dpg.add_text(auth, color=auth_color)

    @monitor_performance
    def create_right_help_panel(self):
        """Create right quick actions panel"""
        with operation('create_right_help_panel', module='help'):
            with dpg.child_window(width=350, height=650, border=True):
                dpg.add_text('QUICK ACTIONS', color=self.BLOOMBERG_ORANGE)
                dpg.add_separator()
                quick_actions = [('📞 Contact Support', self.contact_support), ('📝 Send Feedback', self.send_feedback), ('📚 User Manual', self.open_manual), ('🎥 Watch Tutorials', self.open_tutorials), ('👥 Join Community', self.open_community), ('🔄 Check Updates', self.check_updates), ('⚙️ System Settings', self.open_settings), ('🐛 Report Issue', self.report_bug)]
                for label, callback in quick_actions:
                    dpg.add_button(label=label, callback=callback, width=-1, height=35)
                    dpg.add_spacer(height=5)
                dpg.add_separator()
                dpg.add_text('SYSTEM INFORMATION', color=self.BLOOMBERG_YELLOW)
                with dpg.table(header_row=False, borders_innerH=False, borders_outerH=False):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    system_info = [('Terminal Version:', '4.2.1'), ('Build Date:', '2025-01-15'), ('Platform:', 'Windows 11'), ('Memory Usage:', '2.4 GB'), ('CPU Usage:', '12%'), ('Network Status:', 'Connected'), ('Data Feed:', 'Live'), ('Session Time:', '02:34:12')]
                    for label, value in system_info:
                        with dpg.table_row():
                            dpg.add_text(label, color=self.BLOOMBERG_GRAY)
                            value_color = self.BLOOMBERG_GREEN if 'Connected' in value or 'Live' in value else self.BLOOMBERG_WHITE
                            dpg.add_text(value, color=value_color)
                dpg.add_separator()
                dpg.add_text('RECENT HELP TOPICS', color=self.BLOOMBERG_YELLOW)
                recent_topics = ['How to create portfolios', 'Setting up price alerts', 'Understanding P&L calculations', 'Using technical indicators', 'Exporting data to Excel']
                for topic in recent_topics:
                    with dpg.group(horizontal=True):
                        dpg.add_text('•', color=self.BLOOMBERG_ORANGE)
                        dpg.add_text(topic, color=self.BLOOMBERG_WHITE, wrap=300)
                debug('Right help panel created', module='help')

    def create_help_status_bar(self):
        """Create help status bar"""
        with dpg.group(horizontal=True):
            status_items = [('HELP STATUS:', 'ONLINE', self.BLOOMBERG_GRAY, self.BLOOMBERG_GREEN), ('SUPPORT AVAILABLE:', '24/7', self.BLOOMBERG_GRAY, self.BLOOMBERG_GREEN), ('LAST UPDATED:', '2025-01-15', self.BLOOMBERG_GRAY, self.BLOOMBERG_WHITE), ('HELP VERSION:', '4.2.1', self.BLOOMBERG_GRAY, self.BLOOMBERG_WHITE)]
            for i, (label, value, label_color, value_color) in enumerate(status_items):
                if i > 0:
                    dpg.add_text(' | ', color=self.BLOOMBERG_GRAY)
                dpg.add_text(label, color=label_color)
                dpg.add_text(value, color=value_color)

    def navigate_section(self, section_key):
        """Navigate to help section"""
        section_map = {'F1:ABOUT': 'About', 'F2:FEATURES': 'Features', 'F3:SUPPORT': 'Support', 'F4:CONTACT': 'Support', 'F5:FEEDBACK': 'Support', 'F6:DOCS': 'API Docs'}
        target_tab = section_map.get(section_key, 'About')
        info('Navigating to help section', module='help', context={'section_key': section_key, 'target_tab': target_tab})

    def search_help(self):
        """Search help topics"""
        with operation('search_help', module='help'):
            info('Help search functionality activated', module='help')

    def contact_support(self):
        """Contact support"""
        info('Contacting support team', module='help', context={'email': 'support@fincept.in', 'phone': '+1 (555) 123-4567'})

    def contact_email_support(self):
        """Contact email support"""
        info('Opening email support', module='help', context={'email': 'support@fincept.in'})

    def open_live_chat(self):
        """Open live chat"""
        info('Opening live chat support', module='help')

    def contact_phone_support(self):
        """Contact phone support"""
        info('Initiating phone support', module='help', context={'phone': '+1 (555) 123-4567'})

    def send_feedback(self):
        """Send feedback"""
        info('Opening feedback form', module='help')

    def open_manual(self):
        """Open user manual"""
        info('Opening user manual', module='help')

    def open_documentation(self):
        """Open documentation"""
        info('Opening documentation', module='help')

    def open_tutorials(self):
        """Open video tutorials"""
        info('Opening video tutorials', module='help')

    def open_community(self):
        """Open community forum"""
        info('Opening community forum', module='help')

    def check_updates(self):
        """Check for updates"""
        info('Checking for updates', module='help')

    def open_settings(self):
        """Open settings"""
        info('Opening system settings', module='help')

    def report_bug(self):
        """Report a bug"""
        info('Opening bug report form', module='help')

    def request_feature(self):
        """Request a feature"""
        info('Opening feature request form', module='help')

    @monitor_performance
    def back_to_dashboard(self):
        """Navigate back to dashboard"""
        with operation('back_to_dashboard', module='help'):
            try:
                if hasattr(self.main_app, 'tabs') and 'dashboard' in self.main_app.tabs:
                    info('Returning to Dashboard', module='help')
                    dpg.set_value('main_tab_bar', 'tab_dashboard')
                else:
                    warning('Dashboard not available', module='help')
            except Exception as e:
                error('Error navigating to dashboard', module='help', context={'error': str(e)}, exc_info=True)

    def resize_components(self, left_width, center_width, right_width, top_height, bottom_height, cell_height):
        """Handle component resizing"""
        debug('Component resize requested - using fixed Bloomberg layout', module='help', context={'left_width': left_width, 'center_width': center_width})

    @monitor_performance
    def cleanup(self):
        """Clean up help tab resources"""
        with operation('help_tab_cleanup', module='help'):
            try:
                info('Cleaning up help tab resources', module='help')
                self.scroll_position = 0
                self._cached_datetime = None
                self._datetime_cache_time = 0
                info('Help tab cleanup complete', module='help')
            except Exception as e:
                error('Error in help cleanup', module='help', context={'error': str(e)}, exc_info=True)

class FyersTab(BaseTab):
    """Optimized Fyers Trading Tab for stock data streaming and API integration"""

    def __init__(self, app):
        super().__init__(app)
        self.tag_prefix = f'fyers_{id(self)}_'
        self.config_dir = self._get_config_directory()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.credentials = {'client_id': '', 'pin': '', 'app_id': '', 'app_type': '', 'app_secret': '', 'totp_secret_key': '', 'redirect_uri': 'https://trade.fyers.in/api-login/redirect-uri/index.html'}
        self.BASE_URL = 'https://api-t2.fyers.in/vagator/v2'
        self.BASE_URL_2 = 'https://api-t1.fyers.in/api/v3'
        self._lock = threading.RLock()
        self.access_token = None
        self.is_connected = False
        self.websocket_client = None
        self.streaming_data = []
        self.max_streaming_rows = 1000
        self.previous_prices = {}
        self.is_paused = False
        self.session_start_time = None
        self.message_count = 0
        self.last_message_time = None
        self.current_symbols = ['NSE:SBIN-EQ', 'NSE:ADANIENT-EQ']
        self.current_data_type = 'DepthUpdate'
        self._last_table_update = None
        self._last_stats_update = None
        self.update_throttle_interval = 0.5
        self.load_access_token_on_startup()
        info('FyersTab initialized', context={'config_dir': str(self.config_dir)})

    def _get_config_directory(self) -> Path:
        """Get platform-specific config directory - uses .fincept folder"""
        config_dir = Path.home() / '.fincept' / 'fyers'
        return config_dir

    def get_label(self):
        return ' Fyers Trading'

    def get_tag(self, tag_name: str) -> str:
        """Generate unique tag with prefix"""
        return f'{self.tag_prefix}{tag_name}'

    def safe_add_item(self, add_func, *args, **kwargs):
        """Safely add DearPyGUI item with tag checking"""
        if 'tag' in kwargs:
            tag = kwargs['tag']
            if dpg.does_item_exist(tag):
                try:
                    dpg.delete_item(tag)
                except:
                    pass
        return add_func(*args, **kwargs)

    def safe_set_value(self, tag: str, value: Any):
        """Safely set value with existence check"""
        try:
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, value)
        except Exception as e:
            warning(f'Error setting value for {tag}: {e}')

    def safe_configure_item(self, tag: str, **kwargs):
        """Safely configure item with existence check"""
        try:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, **kwargs)
        except Exception as e:
            warning(f'Error configuring item {tag}: {e}')

    @monitor_performance
    def create_content(self):
        """Create the Fyers trading interface with error handling"""
        with operation('create_fyers_content'):
            try:
                self.cleanup_existing_items()
                self.add_section_header(' Fyers Trading Platform')
                if not FYERS_AVAILABLE:
                    dpg.add_text(' Fyers API not available!')
                    dpg.add_text('Install with: pip install fyers-apiv3')
                    dpg.add_text('Command: pip install fyers-apiv3')
                    return
                self.create_auth_panel()
                dpg.add_spacer(height=10)
                self.create_streaming_panel()
                dpg.add_spacer(height=10)
                self.create_data_viewer()
                info('Fyers tab content created successfully')
            except Exception as e:
                error('Error creating Fyers tab content', context={'error': str(e)}, exc_info=True)
                try:
                    dpg.add_text(f' Error creating interface: {str(e)}')
                    dpg.add_text('Please restart the application or check logs.')
                except:
                    pass

    def cleanup_existing_items(self):
        """Clean up any existing items with our tag prefix"""
        try:
            all_items = dpg.get_all_items()
            for item in all_items:
                try:
                    alias = dpg.get_item_alias(item)
                    if alias and alias.startswith(self.tag_prefix):
                        dpg.delete_item(item)
                except:
                    continue
        except Exception as e:
            warning(f'Warning during cleanup: {e}')

    def create_auth_panel(self):
        """Create authentication and token management panel"""
        with dpg.collapsing_header(label=' Authentication & Token Management', default_open=True):
            with dpg.group(horizontal=True):
                with self.create_child_window('credentials_panel', width=400, height=320):
                    dpg.add_text('Fyers API Credentials')
                    dpg.add_separator()
                    self.safe_add_item(dpg.add_input_text, label='Client ID', default_value=self.credentials['client_id'], tag=self.get_tag('fyers_client_id'), width=200)
                    self.safe_add_item(dpg.add_input_text, label='PIN', default_value=self.credentials['pin'], tag=self.get_tag('fyers_pin'), password=True, width=200)
                    self.safe_add_item(dpg.add_input_text, label='App ID', default_value=self.credentials['app_id'], tag=self.get_tag('fyers_app_id'), width=200)
                    self.safe_add_item(dpg.add_input_text, label='App Type', default_value=self.credentials['app_type'], tag=self.get_tag('fyers_app_type'), width=200)
                    self.safe_add_item(dpg.add_input_text, label='App Secret', default_value=self.credentials['app_secret'], tag=self.get_tag('fyers_app_secret'), password=True, width=200)
                    self.safe_add_item(dpg.add_input_text, label='TOTP Secret', default_value=self.credentials['totp_secret_key'], tag=self.get_tag('fyers_totp_secret'), password=True, width=200)
                    dpg.add_spacer(height=10)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label=' Generate Token', callback=self.generate_access_token, width=120)
                        dpg.add_button(label=' Load Token', callback=self.load_access_token, width=100)
                        dpg.add_button(label=' Save Config', callback=self.save_credentials, width=100)
                with self.create_child_window('auth_status_panel', width=390, height=320):
                    dpg.add_text('Authentication Status')
                    dpg.add_separator()
                    self.safe_add_item(dpg.add_text, 'Status: Not Authenticated', tag=self.get_tag('auth_status_text'), color=(255, 100, 100))
                    self.safe_add_item(dpg.add_text, 'Token: None', tag=self.get_tag('token_status'))
                    self.safe_add_item(dpg.add_text, 'Generated: Never', tag=self.get_tag('token_generated_time'))
                    self.safe_add_item(dpg.add_text, 'Valid Until: Unknown', tag=self.get_tag('token_validity'))
                    dpg.add_spacer(height=10)
                    dpg.add_text('Token File Status:')
                    self.safe_add_item(dpg.add_text, 'access_token.log: Not Found', tag=self.get_tag('token_file_status'))
                    dpg.add_spacer(height=10)
                    with dpg.child_window(height=120, tag=self.get_tag('auth_log')):
                        self.safe_add_item(dpg.add_text, 'Ready for authentication...', tag=self.get_tag('auth_log_text'), wrap=370)

    def create_streaming_panel(self):
        """Create WebSocket streaming control panel"""
        with dpg.collapsing_header(label=' Real-time Data Streaming', default_open=True):
            with dpg.group(horizontal=True):
                with self.create_child_window('connection_controls', width=300, height=280):
                    dpg.add_text('WebSocket Connection')
                    dpg.add_separator()
                    self.safe_add_item(dpg.add_text, 'Status: Disconnected', tag=self.get_tag('ws_status_text'), color=(255, 100, 100))
                    self.safe_add_item(dpg.add_text, 'Data Type: None', tag=self.get_tag('ws_data_type'))
                    self.safe_add_item(dpg.add_text, 'Symbols: None', tag=self.get_tag('ws_symbols'))
                    self.safe_add_item(dpg.add_text, 'Messages Received: 0', tag=self.get_tag('ws_message_count'))
                    dpg.add_spacer(height=10)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label=' Connect', callback=self.connect_websocket, width=90)
                        dpg.add_button(label=' Disconnect', callback=self.disconnect_websocket, width=90)
                    dpg.add_spacer(height=10)
                    dpg.add_text('Connection Health:')
                    self.safe_add_item(dpg.add_text, 'Ping: Unknown', tag=self.get_tag('ws_ping'))
                    self.safe_add_item(dpg.add_text, 'Reconnects: 0', tag=self.get_tag('ws_reconnects'))
                with self.create_child_window('streaming_settings', width=250, height=280):
                    dpg.add_text('Streaming Settings')
                    dpg.add_separator()
                    dpg.add_text('Data Type:')
                    self.safe_add_item(dpg.add_combo, ['SymbolUpdate', 'DepthUpdate'], default_value=self.current_data_type, tag=self.get_tag('stream_data_type'), width=-1)
                    dpg.add_spacer(height=10)
                    dpg.add_text('Stock Symbols:')
                    self.safe_add_item(dpg.add_input_text, hint='NSE:SBIN-EQ,NSE:ADANIENT-EQ', default_value=','.join(self.current_symbols), tag=self.get_tag('stream_symbols'), width=-1, multiline=True, height=80)
                    dpg.add_spacer(height=10)
                    dpg.add_button(label=' Update Subscription', callback=self.update_subscription, width=-1)
                    dpg.add_spacer(height=10)
                    dpg.add_text('Quick Symbols:')
                    with dpg.group(horizontal=True):
                        dpg.add_button(label='NIFTY50', callback=lambda: self.set_quick_symbols('nifty50'), width=60)
                        dpg.add_button(label='BANKNIFTY', callback=lambda: self.set_quick_symbols('banknifty'), width=80)
                with self.create_child_window('streaming_stats', width=240, height=280):
                    dpg.add_text('Streaming Statistics')
                    dpg.add_separator()
                    self.safe_add_item(dpg.add_text, 'Session Time: 00:00:00', tag=self.get_tag('session_time'))
                    self.safe_add_item(dpg.add_text, 'Data Points: 0', tag=self.get_tag('data_points_count'))
                    self.safe_add_item(dpg.add_text, 'Last Update: Never', tag=self.get_tag('last_update_time'))
                    self.safe_add_item(dpg.add_text, 'Data Rate: 0 msg/sec', tag=self.get_tag('data_rate'))
                    dpg.add_spacer(height=10)
                    dpg.add_text('Max Display Rows:')
                    self.safe_add_item(dpg.add_combo, [100, 500, 1000, 2000], default_value=1000, tag=self.get_tag('max_display_rows'), callback=self.on_max_rows_changed, width=-1)
                    dpg.add_spacer(height=10)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label='Clear', callback=self.clear_streaming_data, width=60)
                        dpg.add_button(label='Stats', callback=self.show_detailed_stats, width=60)

    def create_data_viewer(self):
        """Create real-time data viewer"""
        with dpg.collapsing_header(label='Live Data Feed', default_open=True):
            with dpg.group(horizontal=True):
                with dpg.group():
                    with dpg.group(horizontal=True):
                        self.safe_add_item(dpg.add_button, label='Pause', tag=self.get_tag('pause_button'), callback=self.toggle_pause, width=80)
                        dpg.add_button(label='Export', callback=self.export_data, width=80)
                        dpg.add_button(label='Refresh', callback=self.force_refresh_table, width=80)
                        dpg.add_text('Auto-scroll:')
                        self.safe_add_item(dpg.add_checkbox, tag=self.get_tag('auto_scroll'), default_value=True)
                    with dpg.group(horizontal=True):
                        dpg.add_text('Filter Symbol:')
                        self.safe_add_item(dpg.add_input_text, tag=self.get_tag('symbol_filter'), width=120, callback=self.on_symbol_filter_changed)
                        dpg.add_text('Update Rate:')
                        self.safe_add_item(dpg.add_combo, ['Real-time', '1 sec', '2 sec', '5 sec'], default_value='Real-time', tag=self.get_tag('update_rate'), callback=self.on_update_rate_changed, width=100)
            dpg.add_spacer(height=5)
            with self.create_child_window('live_data_viewer', width=-1, height=450):
                self.safe_add_item(dpg.add_text, 'Connect to WebSocket to see live data...', tag=self.get_tag('data_viewer_status'))
                with dpg.group(tag=self.get_tag('live_data_table_container')):
                    pass

    def load_access_token_on_startup(self):
        """Load access token on startup if available"""
        with operation('load_token_on_startup'):
            try:
                token_path = self.config_dir / 'access_token.log'
                if token_path.exists():
                    with open(token_path, 'r', encoding='utf-8') as f:
                        tokens = [line.strip() for line in f if line.strip()]
                    if tokens:
                        self.access_token = tokens[-1]
                        info('Access token loaded on startup', context={'token_file': str(token_path)})
            except Exception as e:
                warning('Could not load token on startup', context={'error': str(e)})

    @monitor_performance
    def save_credentials(self):
        """Save credentials to config file"""
        with operation('save_credentials'):
            try:
                config = {'client_id': dpg.get_value(self.get_tag('fyers_client_id')), 'app_id': dpg.get_value(self.get_tag('fyers_app_id')), 'app_type': dpg.get_value(self.get_tag('fyers_app_type')), 'redirect_uri': self.credentials['redirect_uri']}
                config_path = self.config_dir / 'fyers_config.json'
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                self.update_auth_log('Credentials saved to fyers_config.json')
                info('Fyers credentials saved', context={'config_path': str(config_path)})
            except Exception as e:
                error_msg = f'Error saving credentials: {str(e)}'
                self.update_auth_log(error_msg)
                error('Failed to save credentials', context={'error': str(e)}, exc_info=True)

    @monitor_performance
    def generate_access_token(self):
        """Generate new access token using Fyers authentication flow"""

        def auth_thread():
            with operation('generate_access_token'):
                try:
                    with self._lock:
                        self.update_auth_log(' Starting authentication process...')
                        credentials = {'client_id': dpg.get_value(self.get_tag('fyers_client_id')), 'pin': dpg.get_value(self.get_tag('fyers_pin')), 'app_id': dpg.get_value(self.get_tag('fyers_app_id')), 'app_type': dpg.get_value(self.get_tag('fyers_app_type')), 'app_secret': dpg.get_value(self.get_tag('fyers_app_secret')), 'totp_secret': dpg.get_value(self.get_tag('fyers_totp_secret'))}
                        missing_fields = [k for k, v in credentials.items() if not v.strip()]
                        if missing_fields:
                            self.update_auth_log(f' Missing fields: {', '.join(missing_fields)}')
                            return
                        request_key = None
                        totp = None
                        status, request_key = self.verify_client_id(credentials['client_id'])
                        if status != 1:
                            self.update_auth_log(f'Client ID verification failed: {request_key}')
                            return
                        self.update_auth_log(' Client ID verified')
                        status, totp = self.generate_totp(credentials['totp_secret'])
                        if status != 1:
                            self.update_auth_log(f'TOTP generation failed: {totp}')
                            return
                        self.update_auth_log(' TOTP generated')
                        status, request_key = self.verify_totp(request_key, totp)
                        if status != 1:
                            self.update_auth_log(f'TOTP verification failed: {request_key}')
                            return
                        self.update_auth_log('TOTP verified')
                        status, fyers_access_token = self.verify_pin(request_key, credentials['pin'])
                        if status != 1:
                            self.update_auth_log(f'PIN verification failed: {fyers_access_token}')
                            return
                        self.update_auth_log('PIN verified')
                        status, auth_code = self.get_token(credentials['client_id'], credentials['app_id'], self.credentials['redirect_uri'], credentials['app_type'], fyers_access_token)
                        if status != 1:
                            self.update_auth_log(f'Token generation failed: {auth_code}')
                            return
                        self.update_auth_log('Auth code received')
                        status, v3_access = self.validate_authcode(auth_code, credentials['app_id'], credentials['app_type'], credentials['app_secret'])
                        if status != 1:
                            self.update_auth_log(f'Auth code validation failed: {v3_access}')
                            return
                        self.update_auth_log('Access token validated')
                        self.access_token = f'{credentials['app_id']}-{credentials['app_type']}:{v3_access}'
                        self.save_access_token()
                        self.update_auth_status()
                        self.update_auth_log(' Authentication completed successfully!')
                        info('Fyers authentication completed successfully')
                except Exception as e:
                    error_msg = f'Authentication error: {str(e)}'
                    self.update_auth_log(error_msg)
                    error('Authentication failed', context={'error': str(e)}, exc_info=True)
        threading.Thread(target=auth_thread, daemon=True).start()

    def load_access_token(self):
        """Load access token from file"""
        with operation('load_access_token'):
            try:
                token_path = self.config_dir / 'access_token.log'
                if not token_path.exists():
                    self.update_auth_log('access_token.log file not found')
                    return
                with open(token_path, 'r', encoding='utf-8') as f:
                    tokens = [line.strip() for line in f if line.strip()]
                if not tokens:
                    self.update_auth_log('No tokens found in access_token.log')
                    return
                with self._lock:
                    self.access_token = tokens[-1]
                self.update_auth_status()
                self.update_auth_log('Access token loaded from file')
                info('Access token loaded from file', context={'token_file': str(token_path)})
            except Exception as e:
                error_msg = f' Error loading token: {str(e)}'
                self.update_auth_log(error_msg)
                error('Failed to load token', context={'error': str(e)}, exc_info=True)

    def save_access_token(self):
        """Save access token to file"""
        try:
            token_path = self.config_dir / 'access_token.log'
            with open(token_path, 'a', encoding='utf-8') as f:
                f.write(f'{self.access_token}\n')
            self.update_auth_log(' Token saved to access_token.log')
            info('Token saved', context={'token_file': str(token_path)})
        except Exception as e:
            error_msg = f' Error saving token: {str(e)}'
            self.update_auth_log(error_msg)
            error('Failed to save token', context={'error': str(e)}, exc_info=True)

    def verify_client_id(self, client_id: str) -> Tuple[int, str]:
        """Verify client ID with Fyers API"""
        try:
            payload = {'fy_id': client_id, 'app_id': '2'}
            resp = requests.post(url=f'{self.BASE_URL}/send_login_otp', json=payload, timeout=30)
            if resp.status_code != 200:
                return [-1, f'HTTP {resp.status_code}: {resp.text}']
            data = resp.json()
            debug('Client ID verified successfully', context={'client_id': client_id})
            return [1, data['request_key']]
        except requests.exceptions.Timeout:
            return [-1, 'Request timeout']
        except requests.exceptions.RequestException as e:
            return [-1, f'Network error: {str(e)}']
        except Exception as e:
            return [-1, str(e)]

    def generate_totp(self, secret: str) -> Tuple[int, str]:
        """Generate TOTP code"""
        try:
            if not secret.strip():
                return [-1, 'TOTP secret is empty']
            totp = pyotp.TOTP(secret).now()
            debug('TOTP generated successfully')
            return [1, totp]
        except Exception as e:
            return [-1, f'TOTP generation error: {str(e)}']

    def verify_totp(self, request_key: str, totp: str) -> Tuple[int, str]:
        """Verify TOTP with Fyers API"""
        try:
            payload = {'request_key': request_key, 'otp': totp}
            resp = requests.post(url=f'{self.BASE_URL}/verify_otp', json=payload, timeout=30)
            if resp.status_code != 200:
                return [-1, f'HTTP {resp.status_code}: {resp.text}']
            data = resp.json()
            debug('TOTP verified successfully')
            return [1, data['request_key']]
        except requests.exceptions.Timeout:
            return [-1, 'Request timeout']
        except requests.exceptions.RequestException as e:
            return [-1, f'Network error: {str(e)}']
        except Exception as e:
            return [-1, str(e)]

    def verify_pin(self, request_key: str, pin: str) -> Tuple[int, str]:
        """Verify PIN with Fyers API"""
        try:
            payload = {'request_key': request_key, 'identity_type': 'pin', 'identifier': pin}
            resp = requests.post(url=f'{self.BASE_URL}/verify_pin', json=payload, timeout=30)
            if resp.status_code != 200:
                return [-1, f'HTTP {resp.status_code}: {resp.text}']
            data = resp.json()
            debug('PIN verified successfully')
            return [1, data['data']['access_token']]
        except requests.exceptions.Timeout:
            return [-1, 'Request timeout']
        except requests.exceptions.RequestException as e:
            return [-1, f'Network error: {str(e)}']
        except Exception as e:
            return [-1, str(e)]

    def get_token(self, client_id: str, app_id: str, redirect_uri: str, app_type: str, access_token: str) -> Tuple[int, str]:
        """Get authorization token"""
        try:
            payload = {'fyers_id': client_id, 'app_id': app_id, 'redirect_uri': redirect_uri, 'appType': app_type, 'code_challenge': '', 'state': 'sample_state', 'scope': '', 'nonce': '', 'response_type': 'code', 'create_cookie': True}
            headers = {'Authorization': f'Bearer {access_token}'}
            resp = requests.post(url=f'{self.BASE_URL_2}/token', json=payload, headers=headers, timeout=30)
            if resp.status_code != 308:
                return [-1, f'HTTP {resp.status_code}: {resp.text}']
            data = resp.json()
            url = data['Url']
            auth_code = parse.parse_qs(parse.urlparse(url).query)['auth_code'][0]
            debug('Authorization token received successfully')
            return [1, auth_code]
        except requests.exceptions.Timeout:
            return [-1, 'Request timeout']
        except requests.exceptions.RequestException as e:
            return [-1, f'Network error: {str(e)}']
        except Exception as e:
            return [-1, str(e)]

    def validate_authcode(self, auth_code: str, app_id: str, app_type: str, app_secret: str) -> Tuple[int, str]:
        """Validate authorization code"""
        try:
            app_id_hash = hashlib.sha256(f'{app_id}-{app_type}:{app_secret}'.encode()).hexdigest()
            payload = {'grant_type': 'authorization_code', 'appIdHash': app_id_hash, 'code': auth_code}
            resp = requests.post(url=f'{self.BASE_URL_2}/validate-authcode', json=payload, timeout=30)
            if resp.status_code != 200:
                return [-1, f'HTTP {resp.status_code}: {resp.text}']
            data = resp.json()
            debug('Authorization code validated successfully')
            return [1, data['access_token']]
        except requests.exceptions.Timeout:
            return [-1, 'Request timeout']
        except requests.exceptions.RequestException as e:
            return [-1, f'Network error: {str(e)}']
        except Exception as e:
            return [-1, str(e)]

    @monitor_performance
    def connect_websocket(self):
        """Connect to Fyers WebSocket with enhanced error handling"""
        if not self.access_token:
            self.update_auth_log(' No access token available. Generate token first.')
            return

        def connect_thread():
            with operation('connect_websocket'):
                try:
                    with self._lock:
                        if self.is_connected:
                            self.update_auth_log(' Already connected to WebSocket')
                            return
                    self.update_auth_log(' Connecting to WebSocket...')
                    self.websocket_client = data_ws.FyersDataSocket(access_token=self.access_token, log_path='', litemode=False, write_to_file=False, reconnect=True, on_connect=self.on_websocket_open, on_close=self.on_websocket_close, on_error=self.on_websocket_error, on_message=self.on_websocket_message)
                    with self._lock:
                        self.session_start_time = datetime.datetime.now()
                        self.message_count = 0
                    self.websocket_client.connect()
                    info('WebSocket connection initiated')
                except Exception as e:
                    error_msg = f' WebSocket connection failed: {str(e)}'
                    self.update_auth_log(error_msg)
                    error('WebSocket connection failed', context={'error': str(e)}, exc_info=True)
        threading.Thread(target=connect_thread, daemon=True).start()

    def disconnect_websocket(self):
        """Enhanced WebSocket disconnection"""
        with operation('disconnect_websocket'):
            try:
                self.update_auth_log(' Disconnecting WebSocket...')
                with self._lock:
                    self.is_connected = False
                    self.is_paused = True
                if self.websocket_client:
                    disconnect_methods = [('disconnect', lambda: self.websocket_client.disconnect()), ('close', lambda: self.websocket_client.close()), ('stop', lambda: self.websocket_client.stop())]
                    for method_name, method_func in disconnect_methods:
                        try:
                            if hasattr(self.websocket_client, method_name):
                                method_func()
                                self.update_auth_log(f' Called {method_name}() method')
                        except Exception as e:
                            warning(f'Warning calling {method_name}: {e}')
                    self.websocket_client = None
                    self.update_auth_log(' WebSocket client reference cleared')
                self.safe_set_value(self.get_tag('ws_status_text'), 'Status: Disconnected')
                self.safe_configure_item(self.get_tag('ws_status_text'), color=(255, 100, 100))
                self.safe_set_value(self.get_tag('ws_data_type'), 'Data Type: None')
                self.safe_set_value(self.get_tag('ws_symbols'), 'Symbols: None')
                self.safe_set_value(self.get_tag('ws_message_count'), 'Messages Received: 0')
                self.safe_set_value(self.get_tag('pause_button'), ' Paused')
                self.update_auth_log(' WebSocket disconnected successfully')
                info('WebSocket disconnected')
            except Exception as e:
                error_msg = f' Disconnect error: {str(e)}'
                self.update_auth_log(error_msg)
                error('WebSocket disconnect error', context={'error': str(e)}, exc_info=True)
                with self._lock:
                    self.is_connected = False
                    self.is_paused = True
                self.safe_set_value(self.get_tag('ws_status_text'), 'Status: Force Disconnected')
                self.safe_configure_item(self.get_tag('ws_status_text'), color=(255, 100, 100))

    def on_websocket_open(self):
        """WebSocket open callback with enhanced setup"""
        try:
            with self._lock:
                self.is_connected = True
                self.is_paused = False
                self.session_start_time = datetime.datetime.now()
            self.safe_set_value(self.get_tag('ws_status_text'), 'Status: Connected')
            self.safe_configure_item(self.get_tag('ws_status_text'), color=(100, 255, 100))
            self.safe_set_value(self.get_tag('pause_button'), ' Pause')
            data_type = dpg.get_value(self.get_tag('stream_data_type'))
            symbols_text = dpg.get_value(self.get_tag('stream_symbols'))
            symbols = [s.strip() for s in symbols_text.split(',') if s.strip()]
            if symbols and self.websocket_client:
                self.websocket_client.subscribe(symbols=symbols, data_type=data_type)
                self.safe_set_value(self.get_tag('ws_data_type'), f'Data Type: {data_type}')
                self.safe_set_value(self.get_tag('ws_symbols'), f'Symbols: {', '.join(symbols)}')
                with self._lock:
                    self.current_symbols = symbols
                    self.current_data_type = data_type
                self.websocket_client.keep_running()
            self.update_auth_log(' WebSocket connected and subscribed')
            info('WebSocket connected successfully', context={'symbols': len(symbols), 'data_type': data_type})
        except Exception as e:
            error_msg = f' WebSocket open callback error: {str(e)}'
            self.update_auth_log(error_msg)
            error('WebSocket open callback error', context={'error': str(e)}, exc_info=True)

    def on_websocket_close(self, code):
        """WebSocket close callback"""
        try:
            with self._lock:
                self.is_connected = False
                self.is_paused = True
            self.safe_set_value(self.get_tag('ws_status_text'), 'Status: Disconnected')
            self.safe_configure_item(self.get_tag('ws_status_text'), color=(255, 100, 100))
            self.safe_set_value(self.get_tag('pause_button'), ' Paused')
            error_msg = f' WebSocket closed (code: {code})'
            self.update_auth_log(error_msg)
            warning('WebSocket closed', context={'code': code})
        except Exception as e:
            error('Error in WebSocket close callback', context={'error': str(e)}, exc_info=True)

    def on_websocket_error(self, error):
        """WebSocket error callback"""
        error_msg = f' WebSocket error: {str(error)}'
        self.update_auth_log(error_msg)
        error('WebSocket error', context={'error': str(error)})

    def on_websocket_message(self, message):
        """Enhanced WebSocket message callback with filtering and throttling"""
        try:
            with self._lock:
                if self.is_paused or not self.is_connected:
                    return
                self.message_count += 1
                self.last_message_time = datetime.datetime.now()
            symbol_filter = dpg.get_value(self.get_tag('symbol_filter'))
            if symbol_filter and symbol_filter.strip():
                message_symbol = message.get('symbol', '').upper()
                if symbol_filter.upper() not in message_symbol:
                    return
            timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            data_row = {'timestamp': timestamp, 'symbol': self.safe_encode_text(message.get('symbol', 'Unknown')), 'type': self.safe_encode_text(message.get('type', 'Unknown')), 'data': message}
            with self._lock:
                self.streaming_data.append(data_row)
                try:
                    max_rows = dpg.get_value(self.get_tag('max_display_rows'))
                    if isinstance(max_rows, str):
                        max_rows = int(max_rows)
                    elif max_rows is None:
                        max_rows = 1000
                except (ValueError, TypeError):
                    max_rows = 1000
                if len(self.streaming_data) > max_rows:
                    self.streaming_data = self.streaming_data[-max_rows:]
            should_update = self.should_update_ui()
            if should_update and dpg.get_value(self.get_tag('auto_scroll')) and (not self.is_paused):
                self.update_streaming_stats()
                self.update_data_table()
        except Exception as e:
            error_msg = f' Message processing error: {str(e)}'
            self.update_auth_log(error_msg)
            error('Message processing error', context={'error': str(e)}, exc_info=True)

    def safe_encode_text(self, text: Any) -> str:
        """Safely encode text with proper handling"""
        try:
            if isinstance(text, bytes):
                return text.decode('utf-8', errors='ignore')
            elif isinstance(text, (int, float)):
                return str(text)
            elif text is None:
                return ''
            else:
                return str(text).encode('ascii', errors='ignore').decode('ascii')
        except Exception:
            return 'N/A'

    def should_update_ui(self) -> bool:
        """Determine if UI should be updated based on throttling settings"""
        current_time = datetime.datetime.now()
        try:
            update_rate = dpg.get_value(self.get_tag('update_rate'))
            if update_rate == 'Real-time':
                throttle_seconds = 0.1
            elif update_rate == '1 sec':
                throttle_seconds = 1.0
            elif update_rate == '2 sec':
                throttle_seconds = 2.0
            elif update_rate == '5 sec':
                throttle_seconds = 5.0
            else:
                throttle_seconds = 0.5
        except:
            throttle_seconds = 0.5
        if self._last_table_update is None:
            self._last_table_update = current_time
            return True
        time_diff = (current_time - self._last_table_update).total_seconds()
        if time_diff >= throttle_seconds:
            self._last_table_update = current_time
            return True
        return False

    def update_auth_status(self):
        """Update authentication status in UI with safe operations"""
        try:
            if self.access_token:
                self.safe_set_value(self.get_tag('auth_status_text'), 'Status: Authenticated')
                self.safe_configure_item(self.get_tag('auth_status_text'), color=(100, 255, 100))
                token_display = f'Token: {self.access_token[:20]}...' if len(self.access_token) > 20 else f'Token: {self.access_token}'
                self.safe_set_value(self.get_tag('token_status'), token_display)
                self.safe_set_value(self.get_tag('token_generated_time'), f'Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')
                self.safe_set_value(self.get_tag('token_validity'), 'Valid Until: End of trading day')
            token_file_path = self.config_dir / 'access_token.log'
            token_file_status = f'access_token.log: Found' if token_file_path.exists() else f'access_token.log: Not Found'
            self.safe_set_value(self.get_tag('token_file_status'), token_file_status)
        except Exception as e:
            error('Error updating auth status', context={'error': str(e)}, exc_info=True)

    def update_streaming_stats(self):
        """Update streaming statistics with enhanced calculations"""
        try:
            with self._lock:
                data_count = len(self.streaming_data)
                current_time = datetime.datetime.now()
                self.safe_set_value(self.get_tag('data_points_count'), f'Data Points: {data_count}')
                self.safe_set_value(self.get_tag('last_update_time'), f'Last Update: {current_time.strftime('%H:%M:%S')}')
                if self.session_start_time:
                    session_duration = current_time - self.session_start_time
                    hours, remainder = divmod(session_duration.total_seconds(), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    session_time_str = f'Session Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}'
                    self.safe_set_value(self.get_tag('session_time'), session_time_str)
                if self.session_start_time and self.message_count > 0:
                    elapsed_seconds = (current_time - self.session_start_time).total_seconds()
                    if elapsed_seconds > 0:
                        rate = self.message_count / elapsed_seconds
                        self.safe_set_value(self.get_tag('data_rate'), f'Data Rate: {rate:.1f} msg/sec')
                self.safe_set_value(self.get_tag('ws_message_count'), f'Messages Received: {self.message_count}')
        except Exception as e:
            error('Error updating streaming stats', context={'error': str(e)}, exc_info=True)

    @monitor_performance
    def update_data_table(self):
        """Enhanced data table update with better performance and error handling"""
        with operation('update_data_table'):
            try:
                if not dpg.get_value(self.get_tag('auto_scroll')) or self.is_paused:
                    return
                with self._lock:
                    if not self.streaming_data:
                        return
                    recent_data = self.streaming_data[-100:] if len(self.streaming_data) > 100 else self.streaming_data
                if not recent_data:
                    return
                container_tag = self.get_tag('live_data_table_container')
                if dpg.does_item_exist(container_tag):
                    dpg.delete_item(container_tag, children_only=True)
                    all_keys = set()
                    for row in recent_data[-10:]:
                        if isinstance(row['data'], dict):
                            all_keys.update(row['data'].keys())
                    sorted_keys = sorted(list(all_keys))
                    table_tag = self.get_tag('live_data_table')
                    if dpg.does_item_exist(table_tag):
                        dpg.delete_item(table_tag)
                    with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True, parent=container_tag, scrollY=True, scrollX=True, height=400, tag=table_tag):
                        dpg.add_table_column(label='Time', width_fixed=True, init_width_or_weight=80)
                        dpg.add_table_column(label='Symbol', width_fixed=True, init_width_or_weight=120)
                        dpg.add_table_column(label='Type', width_fixed=True, init_width_or_weight=60)
                        priority_fields = ['ltp', 'volume', 'bid_price1', 'ask_price1', 'high_price', 'low_price']
                        added_fields = set()
                        for field in priority_fields:
                            if field in sorted_keys:
                                dpg.add_table_column(label=field, width_fixed=True, init_width_or_weight=100)
                                added_fields.add(field)
                        for key in sorted_keys:
                            if key not in ['symbol', 'type'] and key not in added_fields:
                                dpg.add_table_column(label=key, width_fixed=True, init_width_or_weight=100)
                                added_fields.add(key)
                        display_data = list(reversed(recent_data[-50:]))
                        for row in display_data:
                            with dpg.table_row():
                                dpg.add_text(row['timestamp'])
                                dpg.add_text(row['symbol'])
                                dpg.add_text(row['type'])
                                if isinstance(row['data'], dict):
                                    for field in priority_fields:
                                        if field in added_fields:
                                            self.add_table_cell(row, field)
                                    for key in sorted_keys:
                                        if key not in ['symbol', 'type'] and key not in priority_fields:
                                            if key in added_fields:
                                                self.add_table_cell(row, key)
                                else:
                                    for _ in added_fields:
                                        dpg.add_text('')
            except Exception as e:
                error('Error updating data table', context={'error': str(e)}, exc_info=True)
                try:
                    container_tag = self.get_tag('live_data_table_container')
                    if dpg.does_item_exist(container_tag):
                        dpg.delete_item(container_tag, children_only=True)
                        dpg.add_text(f'Table update error: {str(e)}', parent=container_tag)
                except:
                    pass

    def add_table_cell(self, row: Dict[str, Any], field: str):
        """Add a table cell with proper formatting and color coding"""
        try:
            value = row['data'].get(field, '')
            if value is None:
                dpg.add_text('NULL')
            else:
                try:
                    if isinstance(value, float):
                        display_value = f'{value:.2f}'
                        color = self.get_price_color(row['symbol'], field, value)
                    elif isinstance(value, int):
                        display_value = f'{value:,}'
                        color = self.get_price_color(row['symbol'], field, value)
                    else:
                        display_value = str(value)
                        color = None
                    if color:
                        dpg.add_text(display_value, color=color)
                    else:
                        dpg.add_text(display_value)
                except Exception:
                    dpg.add_text(str(value))
        except Exception:
            dpg.add_text('')

    def update_auth_log(self, message: str):
        """Update authentication log with safe operations"""
        try:
            log_tag = self.get_tag('auth_log_text')
            if dpg.does_item_exist(log_tag):
                current_log = dpg.get_value(log_tag)
                timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                new_message = f'[{timestamp}] {message}'
                new_log = f'{new_message}\n{current_log}'
                lines = new_log.split('\n')[:15]
                dpg.set_value(log_tag, '\n'.join(lines))
        except Exception as e:
            warning(f'Error updating auth log: {e}')

    def get_price_color(self, symbol: str, field: str, current_value: Any) -> Optional[Tuple[int, int, int]]:
        """Get color for price fields based on movement with enhanced field detection"""
        price_fields = {'ltp', 'ask_price1', 'ask_price2', 'ask_price3', 'ask_price4', 'ask_price5', 'bid_price1', 'bid_price2', 'bid_price3', 'bid_price4', 'bid_price5', 'high_price', 'low_price', 'open_price', 'prev_close_price', 'avg_trade_price', 'last_traded_price', 'close_price', 'price'}
        if field not in price_fields or not isinstance(current_value, (int, float)):
            return None
        key = f'{symbol}_{field}'
        with self._lock:
            previous_value = self.previous_prices.get(key)
            self.previous_prices[key] = current_value
        if previous_value is None:
            return None
        try:
            if current_value > previous_value:
                return (100, 255, 100)
            elif current_value < previous_value:
                return (255, 100, 100)
            else:
                return None
        except:
            return None

    def update_subscription(self):
        """Enhanced WebSocket subscription update"""
        if not self.is_connected or not self.websocket_client:
            self.update_auth_log(' Not connected to WebSocket')
            return
        with operation('update_subscription'):
            try:
                data_type = dpg.get_value(self.get_tag('stream_data_type'))
                symbols_text = dpg.get_value(self.get_tag('stream_symbols'))
                symbols = [s.strip().upper() for s in symbols_text.split(',') if s.strip()]
                if not symbols:
                    self.update_auth_log(' No symbols provided')
                    return
                self.update_auth_log(f' Updating subscription to {len(symbols)} symbols...')
                with self._lock:
                    if hasattr(self.websocket_client, 'unsubscribe') and self.current_symbols:
                        try:
                            self.websocket_client.unsubscribe(symbols=self.current_symbols, data_type=self.current_data_type)
                            self.update_auth_log(' Unsubscribed from previous symbols')
                        except Exception as e:
                            self.update_auth_log(f' Unsubscribe warning: {str(e)}')
                    self.websocket_client.subscribe(symbols=symbols, data_type=data_type)
                    self.current_symbols = symbols
                    self.current_data_type = data_type
                self.safe_set_value(self.get_tag('ws_data_type'), f'Data Type: {data_type}')
                self.safe_set_value(self.get_tag('ws_symbols'), f'Symbols: {', '.join(symbols)}')
                self.update_auth_log(f' Subscription updated: {data_type} for {len(symbols)} symbols')
                info('Subscription updated', context={'symbols_count': len(symbols), 'data_type': data_type})
            except Exception as e:
                error_msg = f' Subscription update failed: {str(e)}'
                self.update_auth_log(error_msg)
                error('Subscription update failed', context={'error': str(e)}, exc_info=True)

    def set_quick_symbols(self, symbol_set: str):
        """Set predefined symbol sets for quick access"""
        try:
            if symbol_set == 'nifty50':
                symbols = ['NSE:RELIANCE-EQ', 'NSE:TCS-EQ', 'NSE:HDFCBANK-EQ', 'NSE:INFY-EQ', 'NSE:HINDUNILVR-EQ', 'NSE:ICICIBANK-EQ', 'NSE:KOTAKBANK-EQ', 'NSE:SBIN-EQ', 'NSE:BHARTIARTL-EQ', 'NSE:ITC-EQ']
            elif symbol_set == 'banknifty':
                symbols = ['NSE:HDFCBANK-EQ', 'NSE:ICICIBANK-EQ', 'NSE:KOTAKBANK-EQ', 'NSE:SBIN-EQ', 'NSE:AXISBANK-EQ', 'NSE:INDUSINDBK-EQ', 'NSE:PNB-EQ', 'NSE:BANKBARODA-EQ']
            else:
                symbols = self.current_symbols
            symbols_text = ','.join(symbols)
            self.safe_set_value(self.get_tag('stream_symbols'), symbols_text)
            self.update_auth_log(f' Set {symbol_set.upper()} symbols')
            info('Quick symbols set', context={'symbol_set': symbol_set, 'count': len(symbols)})
        except Exception as e:
            error_msg = f' Error setting quick symbols: {str(e)}'
            self.update_auth_log(error_msg)
            error('Failed to set quick symbols', context={'error': str(e)}, exc_info=True)

    def on_max_rows_changed(self, sender, app_data):
        """Handle max rows change with safe conversion"""
        try:
            if isinstance(app_data, str):
                max_rows = int(app_data)
            else:
                max_rows = app_data
            with self._lock:
                self.max_streaming_rows = max_rows
                if len(self.streaming_data) > max_rows:
                    self.streaming_data = self.streaming_data[-max_rows:]
            self.update_data_table()
            info('Max rows changed', context={'max_rows': max_rows})
        except (ValueError, TypeError):
            self.max_streaming_rows = 1000
            warning('Invalid max rows value, using default 1000')

    def on_symbol_filter_changed(self, sender, app_data):
        """Handle symbol filter changes"""
        try:
            filter_value = app_data.strip().upper() if app_data else ''
            if filter_value:
                self.update_auth_log(f' Symbol filter set to: {filter_value}')
            else:
                self.update_auth_log(' Symbol filter cleared')
            if dpg.get_value(self.get_tag('auto_scroll')):
                self.update_data_table()
            debug('Symbol filter changed', context={'filter': filter_value})
        except Exception as e:
            error('Error in symbol filter change', context={'error': str(e)}, exc_info=True)

    def on_update_rate_changed(self, sender, app_data):
        """Handle update rate changes"""
        try:
            self.update_auth_log(f' Update rate changed to: {app_data}')
            info('Update rate changed', context={'rate': app_data})
        except Exception as e:
            error('Error in update rate change', context={'error': str(e)}, exc_info=True)

    def clear_streaming_data(self):
        """Clear all streaming data with enhanced cleanup"""
        with operation('clear_streaming_data'):
            try:
                with self._lock:
                    self.streaming_data.clear()
                    self.previous_prices.clear()
                    self.message_count = 0
                container_tag = self.get_tag('live_data_table_container')
                if dpg.does_item_exist(container_tag):
                    dpg.delete_item(container_tag, children_only=True)
                    dpg.add_text('Data cleared. Waiting for new messages...', parent=container_tag)
                self.update_auth_log(' All streaming data cleared')
                info('Streaming data cleared')
            except Exception as e:
                error_msg = f' Error clearing data: {str(e)}'
                self.update_auth_log(error_msg)
                error('Failed to clear streaming data', context={'error': str(e)}, exc_info=True)

    def toggle_pause(self):
        """Enhanced pause/resume functionality"""
        try:
            with self._lock:
                self.is_paused = not self.is_paused
            if self.is_paused:
                self.safe_set_value(self.get_tag('pause_button'), ' Resume')
                self.update_auth_log(' Data streaming paused')
            else:
                self.safe_set_value(self.get_tag('pause_button'), ' Pause')
                self.update_auth_log(' Data streaming resumed')
                if self.streaming_data:
                    self.update_data_table()
            info(f'Streaming {('paused' if self.is_paused else 'resumed')}')
        except Exception as e:
            error_msg = f' Error toggling pause: {str(e)}'
            self.update_auth_log(error_msg)
            error('Failed to toggle pause', context={'error': str(e)}, exc_info=True)

    def force_refresh_table(self):
        """Force refresh the data table"""
        try:
            self.update_data_table()
            self.update_auth_log(' Table refreshed manually')
            info('Table manually refreshed')
        except Exception as e:
            error_msg = f' Error refreshing table: {str(e)}'
            self.update_auth_log(error_msg)
            error('Failed to refresh table', context={'error': str(e)}, exc_info=True)

    def show_detailed_stats(self):
        """Show detailed streaming statistics"""
        try:
            with self._lock:
                total_messages = self.message_count
                data_count = len(self.streaming_data)
                unique_symbols = len(set((row['symbol'] for row in self.streaming_data)))
            stats_message = f' Detailed Statistics:\n• Total Messages: {total_messages:,}\n• Data Points Stored: {data_count:,}\n• Unique Symbols: {unique_symbols}\n• Memory Usage: ~{len(str(self.streaming_data)) / 1024:.1f} KB'
            self.update_auth_log(stats_message)
            info('Detailed stats shown', context={'messages': total_messages, 'data_points': data_count, 'symbols': unique_symbols})
        except Exception as e:
            error_msg = f' Error showing stats: {str(e)}'
            self.update_auth_log(error_msg)
            error('Failed to show stats', context={'error': str(e)}, exc_info=True)

    @monitor_performance
    def export_data(self):
        """Enhanced export functionality with multiple formats"""
        with operation('export_data'):
            try:
                with self._lock:
                    if not self.streaming_data:
                        self.update_auth_log(' No data to export')
                        return
                    data_to_export = self.streaming_data.copy()
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_filename = f'fyers_stream_{timestamp}.csv'
                csv_path = self.config_dir / csv_filename
                self.export_to_csv(data_to_export, csv_path)
                json_filename = f'fyers_stream_{timestamp}.json'
                json_path = self.config_dir / json_filename
                self.export_to_json(data_to_export, json_path)
                export_msg = f' Exported {len(data_to_export)} records to {csv_filename} and {json_filename}'
                self.update_auth_log(export_msg)
                info('Data exported successfully', context={'records': len(data_to_export), 'csv_file': str(csv_path), 'json_file': str(json_path)})
            except Exception as e:
                error_msg = f' Export failed: {str(e)}'
                self.update_auth_log(error_msg)
                error('Data export failed', context={'error': str(e)}, exc_info=True)

    def export_to_csv(self, data: List[Dict], filepath: Path):
        """Export data to CSV format"""
        try:
            all_keys = set(['timestamp', 'symbol', 'type'])
            for row in data:
                if isinstance(row['data'], dict):
                    all_keys.update(row['data'].keys())
            sorted_keys = sorted(list(all_keys))
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=sorted_keys)
                writer.writeheader()
                for row in data:
                    export_row = {'timestamp': row['timestamp'], 'symbol': row['symbol'], 'type': row['type']}
                    if isinstance(row['data'], dict):
                        for key, value in row['data'].items():
                            if key not in ['symbol', 'type']:
                                export_row[key] = value
                    writer.writerow(export_row)
            debug('CSV export completed', context={'filepath': str(filepath), 'rows': len(data)})
        except Exception as e:
            raise Exception(f'CSV export error: {str(e)}')

    def export_to_json(self, data: List[Dict], filepath: Path):
        """Export data to JSON format"""
        try:
            export_data = {'metadata': {'export_time': datetime.datetime.now().isoformat(), 'total_records': len(data), 'symbols': list(set((row['symbol'] for row in data))), 'data_types': list(set((row['type'] for row in data))), 'time_range': {'start': data[0]['timestamp'] if data else None, 'end': data[-1]['timestamp'] if data else None}}, 'data': data}
            with open(filepath, 'w', encoding='utf-8') as jsonfile:
                json.dump(export_data, jsonfile, indent=2, ensure_ascii=False)
            debug('JSON export completed', context={'filepath': str(filepath), 'records': len(data)})
        except Exception as e:
            raise Exception(f'JSON export error: {str(e)}')

    def get_connection_health(self) -> Dict[str, Any]:
        """Get current connection health status"""
        try:
            with self._lock:
                current_time = datetime.datetime.now()
                health_status = {'is_connected': self.is_connected, 'is_paused': self.is_paused, 'has_token': bool(self.access_token), 'session_duration': None, 'message_count': self.message_count, 'data_points': len(self.streaming_data), 'last_message': None, 'symbols_count': len(self.current_symbols), 'websocket_client': self.websocket_client is not None}
                if self.session_start_time:
                    duration = current_time - self.session_start_time
                    health_status['session_duration'] = duration.total_seconds()
                if self.last_message_time:
                    time_since_last = current_time - self.last_message_time
                    health_status['last_message'] = time_since_last.total_seconds()
                return health_status
        except Exception as e:
            error('Error getting connection health', context={'error': str(e)}, exc_info=True)
            return {'error': str(e)}

    def reconnect_websocket(self):
        """Reconnect WebSocket with enhanced logic"""
        with operation('reconnect_websocket'):
            try:
                self.update_auth_log(' Attempting to reconnect WebSocket...')
                if self.is_connected:
                    self.disconnect_websocket()
                    time.sleep(2)
                self.connect_websocket()
                info('WebSocket reconnection attempted')
            except Exception as e:
                error_msg = f' Reconnection failed: {str(e)}'
                self.update_auth_log(error_msg)
                error('WebSocket reconnection failed', context={'error': str(e)}, exc_info=True)

    def validate_symbols(self, symbols: List[str]) -> Tuple[List[str], List[str]]:
        """Validate symbol format and return valid/invalid lists"""
        valid_symbols = []
        invalid_symbols = []
        for symbol in symbols:
            symbol = symbol.strip().upper()
            if not symbol:
                continue
            if ':' in symbol and '-EQ' in symbol:
                parts = symbol.split(':')
                if len(parts) == 2 and parts[0] in ['NSE', 'BSE']:
                    valid_symbols.append(symbol)
                else:
                    invalid_symbols.append(symbol)
            else:
                invalid_symbols.append(symbol)
        return (valid_symbols, invalid_symbols)

    def auto_save_config(self):
        """Auto-save current configuration"""
        try:
            config = {'last_symbols': self.current_symbols, 'last_data_type': self.current_data_type, 'max_rows': self.max_streaming_rows, 'auto_scroll': dpg.get_value(self.get_tag('auto_scroll')) if dpg.does_item_exist(self.get_tag('auto_scroll')) else True, 'update_rate': dpg.get_value(self.get_tag('update_rate')) if dpg.does_item_exist(self.get_tag('update_rate')) else 'Real-time'}
            config_path = self.config_dir / 'fyers_session_config.json'
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            info('Session configuration auto-saved', context={'config_path': str(config_path)})
        except Exception as e:
            warning('Could not auto-save config', context={'error': str(e)})

    def load_session_config(self):
        """Load saved session configuration"""
        try:
            config_path = self.config_dir / 'fyers_session_config.json'
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                if 'last_symbols' in config:
                    self.current_symbols = config['last_symbols']
                if 'last_data_type' in config:
                    self.current_data_type = config['last_data_type']
                if 'max_rows' in config:
                    self.max_streaming_rows = config['max_rows']
                info('Session configuration loaded', context={'config_path': str(config_path)})
                return True
        except Exception as e:
            warning('Could not load session config', context={'error': str(e)})
        return False

    @monitor_performance
    def cleanup(self):
        """Enhanced cleanup with auto-save"""
        with operation('fyers_tab_cleanup'):
            try:
                self.auto_save_config()
                if self.websocket_client:
                    try:
                        self.disconnect_websocket()
                    except Exception as e:
                        warning('Warning during WebSocket cleanup', context={'error': str(e)})
                with self._lock:
                    self.streaming_data.clear()
                    self.previous_prices.clear()
                self.cleanup_existing_items()
                info('Fyers tab cleaned up successfully')
            except Exception as e:
                error('Error during cleanup', context={'error': str(e)}, exc_info=True)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        try:
            with self._lock:
                current_time = datetime.datetime.now()
                metrics = {'memory_usage_kb': len(str(self.streaming_data)) / 1024, 'total_messages': self.message_count, 'stored_data_points': len(self.streaming_data), 'unique_symbols': len(set((row['symbol'] for row in self.streaming_data))), 'avg_message_size': len(str(self.streaming_data)) / max(len(self.streaming_data), 1), 'is_healthy': self.is_connected and (not self.is_paused), 'uptime_seconds': None}
                if self.session_start_time:
                    uptime = current_time - self.session_start_time
                    metrics['uptime_seconds'] = uptime.total_seconds()
                    if uptime.total_seconds() > 0:
                        metrics['messages_per_second'] = self.message_count / uptime.total_seconds()
                return metrics
        except Exception as e:
            error('Error getting performance metrics', context={'error': str(e)}, exc_info=True)
            return {'error': str(e)}

    def format_number(self, value: Any) -> str:
        """Format numbers for display"""
        try:
            if isinstance(value, float):
                if value >= 1000000:
                    return f'{value / 1000000:.2f}M'
                elif value >= 1000:
                    return f'{value / 1000:.2f}K'
                else:
                    return f'{value:.2f}'
            elif isinstance(value, int):
                if value >= 1000000:
                    return f'{value / 1000000:.1f}M'
                elif value >= 1000:
                    return f'{value / 1000:.1f}K'
                else:
                    return f'{value:,}'
            else:
                return str(value)
        except:
            return str(value)

    def get_symbol_stats(self, symbol: str) -> Dict[str, Any]:
        """Get statistics for a specific symbol"""
        try:
            with self._lock:
                symbol_data = [row for row in self.streaming_data if row['symbol'] == symbol]
            if not symbol_data:
                return {'error': 'No data found for symbol'}
            stats = {'total_updates': len(symbol_data), 'first_seen': symbol_data[0]['timestamp'], 'last_seen': symbol_data[-1]['timestamp'], 'data_types': list(set((row['type'] for row in symbol_data)))}
            prices = []
            for row in symbol_data:
                if isinstance(row['data'], dict) and 'ltp' in row['data']:
                    try:
                        prices.append(float(row['data']['ltp']))
                    except:
                        continue
            if prices:
                stats.update({'price_high': max(prices), 'price_low': min(prices), 'price_avg': sum(prices) / len(prices), 'price_current': prices[-1], 'price_change': prices[-1] - prices[0] if len(prices) > 1 else 0})
            return stats
        except Exception as e:
            return {'error': str(e)}

    def emergency_stop(self):
        """Emergency stop all operations"""
        try:
            warning('Emergency stop initiated')
            with self._lock:
                self.is_connected = False
                self.is_paused = True
            if self.websocket_client:
                self.websocket_client = None
            self.safe_set_value(self.get_tag('ws_status_text'), 'Status: Emergency Stopped')
            self.safe_configure_item(self.get_tag('ws_status_text'), color=(255, 0, 0))
            self.update_auth_log('Emergency stop - All operations halted')
            error('Emergency stop executed')
        except Exception as e:
            error('Error during emergency stop', context={'error': str(e)}, exc_info=True)

class DataSourceManager:
    """
    Universal Data Source Manager - The backbone of all data in the terminal
    All tabs query this manager instead of directly calling APIs
    """

    def __init__(self, app):
        logger.info('Initializing DataSourceManager')
        try:
            with operation('DataSourceManager initialization'):
                self.app = app
                self.config_file = Path.home() / '.fincept' / 'data_sources.json'
                self.ensure_config_dir()
                self.cache = {}
                self.cache_expiry = {}
                self.cache_duration = 300
                self._cache_lock = threading.RLock()
                self._provider_cache = {}
                self._cache_hits = 0
                self._cache_misses = 0
                self._api_calls = 0
                self._errors = 0
                self.default_sources = {'stocks': 'yfinance', 'forex': 'fincept_api', 'crypto': 'fincept_api', 'news': 'dummy_news', 'economic': 'fincept_api', 'portfolio': 'local_storage', 'options': 'yfinance', 'indices': 'yfinance'}
                self.available_sources = {'yfinance': {'name': 'Yahoo Finance', 'type': 'api', 'supports': ['stocks', 'indices', 'options', 'forex'], 'requires_auth': False, 'real_time': False}, 'fincept_api': {'name': 'Fincept Premium API', 'type': 'api', 'supports': ['stocks', 'forex', 'crypto', 'economic', 'news'], 'requires_auth': True, 'real_time': True}, 'alpha_vantage_data': {'name': 'Alpha Vantage', 'type': 'api', 'supports': ['stocks', 'forex', 'crypto'], 'requires_auth': True, 'real_time': False}, 'dummy_news': {'name': 'Sample News Feed', 'type': 'dummy', 'supports': ['news'], 'requires_auth': False, 'real_time': False}, 'csv_import': {'name': 'CSV File Import', 'type': 'file', 'supports': ['stocks', 'portfolio', 'custom'], 'requires_auth': False, 'real_time': False}, 'websocket_feed': {'name': 'WebSocket Data Feed', 'type': 'websocket', 'supports': ['stocks', 'crypto', 'forex'], 'requires_auth': True, 'real_time': True}}
                self.config = self.load_configuration()
                logger.info('DataSourceManager initialized successfully', context={'default_sources': list(self.default_sources.keys()), 'available_sources': len(self.available_sources)})
        except Exception as e:
            logger.error('DataSourceManager initialization failed', context={'error': str(e)}, exc_info=True)
            raise

    def get_settings_manager(self):
        """Get settings manager from the settings tab"""
        try:
            if hasattr(self.app, 'tabs'):
                for tab_key in self.app.tabs.keys():
                    if 'settings' in tab_key.lower() or 'Settings' in tab_key:
                        settings_tab = self.app.tabs[tab_key]
                        if hasattr(settings_tab, 'settings_manager'):
                            debug(f'Found settings manager in tab: {tab_key}', module='DataSourceManager')
                            return settings_tab.settings_manager
            possible_names = ['Settings', '⚙️ Settings', 'settings', 'SETTINGS']
            for name in possible_names:
                if hasattr(self.app, 'tabs') and name in self.app.tabs:
                    settings_tab = self.app.tabs[name]
                    if hasattr(settings_tab, 'settings_manager'):
                        debug(f'Found settings manager in tab: {name}', module='DataSourceManager')
                        return settings_tab.settings_manager
            debug('Settings manager not found', module='DataSourceManager')
            return None
        except Exception as e:
            debug(f'Error getting settings manager: {str(e)}', module='DataSourceManager')
            return None

    def _get_provider_instance(self, provider_name: str, credentials: Dict[str, str]=None):
        """Get or create provider instance"""
        cache_key = f'{provider_name}_{hash(str(credentials))}'
        if cache_key in self._provider_cache:
            return self._provider_cache[cache_key]
        if provider_name == 'alpha_vantage_data':
            api_key = ''
            if credentials and 'alpha_vantage_api_key' in credentials:
                api_key = credentials['alpha_vantage_api_key']
            if not api_key:
                settings_manager = self.get_settings_manager()
                if settings_manager:
                    api_key = settings_manager.get_api_key('alpha_vantage_data')
                    debug(f'Got API key from settings: {len(api_key)} chars', module='DataSourceManager')
            if api_key and len(api_key) > 5:
                try:
                    from fincept_terminal.DatabaseConnector.DataSources.alpha_vantage_data.alpha_vantage_provider import AlphaVantageProvider
                    provider = AlphaVantageProvider(api_key)
                    self._provider_cache[cache_key] = provider
                    info(f'Alpha Vantage provider created successfully', module='DataSourceManager')
                    return provider
                except ImportError as e:
                    error(f'Alpha Vantage provider import failed: {str(e)}', module='DataSourceManager')
                    return None
            else:
                warning(f'No valid Alpha Vantage API key found (length: {len(api_key)})', module='DataSourceManager')
                return None
        return None

    def ensure_config_dir(self):
        """Ensure configuration directory exists"""
        try:
            self.config_file.parent.mkdir(exist_ok=True, parents=True)
            logger.debug('Configuration directory ensured', context={'config_dir': str(self.config_file.parent)})
        except Exception as e:
            logger.error('Failed to create config directory', context={'error': str(e)}, exc_info=True)

    def load_configuration(self) -> Dict[str, Any]:
        """Load data source configuration"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info('Data source configuration loaded', context={'config_file': str(self.config_file)})
                return config
            else:
                logger.info('No configuration found, using defaults')
                return {'data_mappings': self.default_sources.copy(), 'source_configs': {}}
        except Exception as e:
            logger.error('Error loading configuration', context={'error': str(e), 'config_file': str(self.config_file)}, exc_info=True)
            return {'data_mappings': self.default_sources.copy(), 'source_configs': {}}

    def save_configuration(self):
        """Save current configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info('Configuration saved successfully', context={'config_file': str(self.config_file)})
            return True
        except Exception as e:
            logger.error('Error saving configuration', context={'error': str(e)}, exc_info=True)
            return False

    def set_data_source(self, data_type: str, source_name: str, source_config: Dict[str, Any]=None):
        """Set data source for a specific data type"""
        try:
            if source_name not in self.available_sources:
                raise ValueError(f'Unknown data source: {source_name}')
            if data_type not in self.available_sources[source_name]['supports']:
                raise ValueError(f"Source {source_name} doesn't support {data_type}")
            self.config['data_mappings'][data_type] = source_name
            if source_config:
                if 'source_configs' not in self.config:
                    self.config['source_configs'] = {}
                self.config['source_configs'][source_name] = source_config
            self.save_configuration()
            logger.info('Data source updated', context={'data_type': data_type, 'source': source_name})
        except Exception as e:
            logger.error('Failed to set data source', context={'data_type': data_type, 'source_name': source_name, 'error': str(e)}, exc_info=True)
            raise

    def get_data_source(self, data_type: str) -> str:
        """Get configured data source for a data type"""
        settings_manager = self.get_settings_manager()
        if settings_manager:
            try:
                preferences = settings_manager.settings.get('preferences', {})
                default_provider = preferences.get('default_provider', 'yfinance')
                if default_provider in self.available_sources and data_type in self.available_sources[default_provider]['supports'] and settings_manager.is_provider_enabled(default_provider):
                    debug(f'Using provider from settings: {default_provider} for {data_type}', module='DataSourceManager')
                    return default_provider
                if settings_manager.is_provider_enabled('alpha_vantage_data') and data_type in self.available_sources['alpha_vantage_data']['supports'] and settings_manager.get_api_key('alpha_vantage_data'):
                    debug(f'Using Alpha Vantage for {data_type} (enabled with API key)', module='DataSourceManager')
                    return 'alpha_vantage_data'
            except Exception as e:
                debug(f'Error checking settings for data source: {str(e)}', module='DataSourceManager')
        source = self.config['data_mappings'].get(data_type, self.default_sources.get(data_type, 'yfinance'))
        debug(f'Using fallback source: {source} for {data_type}', module='DataSourceManager')
        return source

    def is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        with self._cache_lock:
            if cache_key not in self.cache:
                return False
            if cache_key not in self.cache_expiry:
                return False
            return datetime.now() < self.cache_expiry[cache_key]

    def set_cache(self, cache_key: str, data: Any, duration: int=None):
        """Set data in cache"""
        settings_manager = self.get_settings_manager()
        if settings_manager:
            preferences = settings_manager.settings.get('preferences', {})
            if not preferences.get('cache_enabled', True):
                return
            duration = duration or preferences.get('cache_duration', self.cache_duration)
        duration = duration or self.cache_duration
        with self._cache_lock:
            self.cache[cache_key] = data
            self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=duration)
        logger.debug('Data cached', context={'cache_key': cache_key, 'duration': duration})

    def get_cache(self, cache_key: str) -> Any:
        """Get data from cache if valid"""
        with self._cache_lock:
            if self.is_cache_valid(cache_key):
                self._cache_hits += 1
                logger.debug('Cache hit', context={'cache_key': cache_key})
                return self.cache[cache_key]
            else:
                self._cache_misses += 1
                logger.debug('Cache miss', context={'cache_key': cache_key})
                return None

    @monitor_performance
    def get_stock_data(self, symbol: str, period: str='1d', interval: str='1m') -> Dict[str, Any]:
        """Universal stock data retrieval"""
        try:
            with operation(f'Get stock data for {symbol}'):
                cache_key = f'stock_{symbol}_{period}_{interval}'
                cached_data = self.get_cache(cache_key)
                if cached_data:
                    return cached_data
                source = self.get_data_source('stocks')
                self._api_calls += 1
                logger.debug('Fetching stock data', context={'symbol': symbol, 'period': period, 'interval': interval, 'source': source})
                if source == 'yfinance':
                    data = self._get_yfinance_stock_data(symbol, period, interval)
                elif source == 'fincept_api':
                    data = self._get_fincept_stock_data(symbol, period, interval)
                elif source == 'alpha_vantage_data':
                    data = asyncio.run(self._get_alpha_vantage_stock_data(symbol, period, interval))
                else:
                    data = self._get_fallback_stock_data(symbol, period, interval)
                if data.get('success'):
                    self.set_cache(cache_key, data, 60)
                    logger.info('Stock data retrieved successfully', context={'symbol': symbol, 'source': source})
                else:
                    self._errors += 1
                    logger.warning('Stock data retrieval failed', context={'symbol': symbol, 'error': data.get('error')})
                return data
        except Exception as e:
            self._errors += 1
            logger.error('Stock data retrieval error', context={'symbol': symbol, 'error': str(e)}, exc_info=True)
            return {'success': False, 'error': f'Error fetching stock data: {str(e)}', 'source': 'error', 'symbol': symbol}

    async def _get_alpha_vantage_stock_data(self, symbol: str, period: str, interval: str) -> Dict[str, Any]:
        """Get stock data from Alpha Vantage provider"""
        try:
            provider = self._get_provider_instance('alpha_vantage_data')
            if not provider:
                return {'success': False, 'error': 'Alpha Vantage provider not configured', 'source': 'alpha_vantage_data'}
            return await provider.get_stock_data(symbol, period, interval)
        except Exception as e:
            logger.error('Alpha Vantage stock data error', context={'symbol': symbol, 'error': str(e)}, exc_info=True)
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    async def get_weekly_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get weekly stock data from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_stock_data(symbol, interval='W')

    async def get_monthly_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get monthly stock data from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_stock_data(symbol, interval='M')

    async def get_daily_adjusted(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get daily adjusted stock data from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_daily_adjusted(symbol)

    async def get_weekly_adjusted(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get weekly adjusted stock data from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_weekly_adjusted(symbol)

    async def get_monthly_adjusted(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get monthly adjusted stock data from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_monthly_adjusted(symbol)

    async def get_global_quote(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get global quote from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_global_quote(symbol)

    async def search_symbols(self, keywords: str, **kwargs) -> Dict[str, Any]:
        """Search symbols from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.search_symbols(keywords)

    async def get_company_overview(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get company overview from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_company_overview(symbol)

    async def get_income_statement(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get income statement from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_income_statement(symbol)

    async def get_balance_sheet(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get balance sheet from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_balance_sheet(symbol)

    async def get_cash_flow(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get cash flow from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_cash_flow(symbol)

    async def get_earnings(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get earnings from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_earnings(symbol)

    async def get_earnings_estimates(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get earnings estimates from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_earnings_estimates(symbol)

    async def get_dividends(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get dividends from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_dividends(symbol)

    async def get_splits(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get splits from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_splits(symbol)

    async def get_sma(self, symbol: str, interval: str='daily', time_period: int=14, series_type: str='close', **kwargs) -> Dict[str, Any]:
        """Get Simple Moving Average from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_sma(symbol, interval, time_period, series_type)

    async def get_ema(self, symbol: str, interval: str='daily', time_period: int=14, series_type: str='close', **kwargs) -> Dict[str, Any]:
        """Get Exponential Moving Average from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_ema(symbol, interval, time_period, series_type)

    async def get_rsi(self, symbol: str, interval: str='daily', time_period: int=14, series_type: str='close', **kwargs) -> Dict[str, Any]:
        """Get RSI from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_rsi(symbol, interval, time_period, series_type)

    async def get_macd(self, symbol: str, interval: str='daily', series_type: str='close', **kwargs) -> Dict[str, Any]:
        """Get MACD from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_macd(symbol, interval, series_type)

    async def get_bbands(self, symbol: str, interval: str='daily', time_period: int=20, series_type: str='close', **kwargs) -> Dict[str, Any]:
        """Get Bollinger Bands from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_bbands(symbol, interval, time_period, series_type)

    async def get_stoch(self, symbol: str, interval: str='daily', **kwargs) -> Dict[str, Any]:
        """Get Stochastic from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_stoch(symbol, interval)

    async def get_adx(self, symbol: str, interval: str='daily', time_period: int=14, **kwargs) -> Dict[str, Any]:
        """Get ADX from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_adx(symbol, interval, time_period)

    async def get_vwap(self, symbol: str, interval: str='15min', **kwargs) -> Dict[str, Any]:
        """Get VWAP from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_vwap(symbol, interval)

    async def get_currency_exchange_rate(self, from_currency: str='USD', to_currency: str='EUR', **kwargs) -> Dict[str, Any]:
        """Get currency exchange rate from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_currency_exchange_rate(from_currency, to_currency)

    async def get_fx_intraday(self, from_symbol: str='USD', to_symbol: str='EUR', interval: str='5min', **kwargs) -> Dict[str, Any]:
        """Get FX intraday from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_fx_intraday(from_symbol, to_symbol, interval)

    async def get_fx_weekly(self, from_symbol: str='USD', to_symbol: str='EUR', **kwargs) -> Dict[str, Any]:
        """Get FX weekly from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_fx_weekly(from_symbol, to_symbol)

    async def get_fx_monthly(self, from_symbol: str='USD', to_symbol: str='EUR', **kwargs) -> Dict[str, Any]:
        """Get FX monthly from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_fx_monthly(from_symbol, to_symbol)

    async def get_crypto_intraday(self, symbol: str, market: str='USD', interval: str='5min', **kwargs) -> Dict[str, Any]:
        """Get crypto intraday from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_crypto_intraday(symbol, market, interval)

    async def get_digital_currency_weekly(self, symbol: str, market: str='USD', **kwargs) -> Dict[str, Any]:
        """Get digital currency weekly from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_digital_currency_weekly(symbol, market)

    async def get_digital_currency_monthly(self, symbol: str, market: str='USD', **kwargs) -> Dict[str, Any]:
        """Get digital currency monthly from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_digital_currency_monthly(symbol, market)

    async def get_wti_oil(self, interval: str='monthly', **kwargs) -> Dict[str, Any]:
        """Get WTI oil from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_wti_oil(interval)

    async def get_brent_oil(self, interval: str='monthly', **kwargs) -> Dict[str, Any]:
        """Get Brent oil from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_brent_oil(interval)

    async def get_natural_gas(self, interval: str='monthly', **kwargs) -> Dict[str, Any]:
        """Get Natural gas from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_natural_gas(interval)

    async def get_copper(self, interval: str='monthly', **kwargs) -> Dict[str, Any]:
        """Get Copper from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_copper(interval)

    async def get_aluminum(self, interval: str='monthly', **kwargs) -> Dict[str, Any]:
        """Get Aluminum from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_aluminum(interval)

    async def get_real_gdp(self, interval: str='annual', **kwargs) -> Dict[str, Any]:
        """Get Real GDP from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_real_gdp(interval)

    async def get_unemployment(self, **kwargs) -> Dict[str, Any]:
        """Get Unemployment from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_unemployment()

    async def get_cpi(self, interval: str='monthly', **kwargs) -> Dict[str, Any]:
        """Get CPI from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_cpi(interval)

    async def get_treasury_yield(self, interval: str='monthly', maturity: str='10year', **kwargs) -> Dict[str, Any]:
        """Get Treasury yield from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_treasury_yield(interval, maturity)

    async def get_federal_funds_rate(self, interval: str='monthly', **kwargs) -> Dict[str, Any]:
        """Get Federal funds rate from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_federal_funds_rate(interval)

    async def get_news_sentiment(self, tickers: str=None, topics: str=None, **kwargs) -> Dict[str, Any]:
        """Get news sentiment from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_news_sentiment(tickers, topics)

    async def get_top_gainers_losers(self, **kwargs) -> Dict[str, Any]:
        """Get top gainers/losers from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_top_gainers_losers()

    async def get_insider_transactions(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get insider transactions from Alpha Vantage"""
        provider = self._get_provider_instance('alpha_vantage_data')
        if not provider:
            return {'success': False, 'error': 'Alpha Vantage provider not configured'}
        return await provider.get_insider_transactions(symbol)

    @monitor_performance
    def get_forex_data(self, pair: str, period: str='1d') -> Dict[str, Any]:
        """Universal forex data retrieval"""
        try:
            with operation(f'Get forex data for {pair}'):
                cache_key = f'forex_{pair}_{period}'
                cached_data = self.get_cache(cache_key)
                if cached_data:
                    return cached_data
                source = self.get_data_source('forex')
                self._api_calls += 1
                logger.debug('Fetching forex data', context={'pair': pair, 'period': period, 'source': source})
                if source == 'yfinance':
                    data = self._get_yfinance_forex_data(pair, period)
                elif source == 'fincept_api':
                    data = self._get_fincept_forex_data(pair, period)
                elif source == 'alpha_vantage_data':
                    data = asyncio.run(self._get_alpha_vantage_forex_data(pair, period))
                else:
                    data = self._get_fallback_forex_data(pair, period)
                if data.get('success'):
                    self.set_cache(cache_key, data, 300)
                    logger.info('Forex data retrieved successfully', context={'pair': pair, 'source': source})
                else:
                    self._errors += 1
                    logger.warning('Forex data retrieval failed', context={'pair': pair, 'error': data.get('error')})
                return data
        except Exception as e:
            self._errors += 1
            logger.error('Forex data retrieval error', context={'pair': pair, 'error': str(e)}, exc_info=True)
            return {'success': False, 'error': f'Error fetching forex data: {str(e)}', 'source': 'error', 'pair': pair}

    async def _get_alpha_vantage_forex_data(self, pair: str, period: str) -> Dict[str, Any]:
        """Get forex data from Alpha Vantage provider"""
        try:
            provider = self._get_provider_instance('alpha_vantage_data')
            if not provider:
                return {'success': False, 'error': 'Alpha Vantage provider not configured', 'source': 'alpha_vantage_data'}
            return await provider.get_forex_data(pair, period)
        except Exception as e:
            logger.error('Alpha Vantage forex data error', context={'pair': pair, 'error': str(e)}, exc_info=True)
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    @monitor_performance
    def get_news_data(self, category: str='financial', limit: int=20) -> Dict[str, Any]:
        """Universal news data retrieval"""
        try:
            with operation(f'Get news data for {category}'):
                cache_key = f'news_{category}_{limit}'
                cached_data = self.get_cache(cache_key)
                if cached_data:
                    return cached_data
                source = self.get_data_source('news')
                self._api_calls += 1
                logger.debug('Fetching news data', context={'category': category, 'limit': limit, 'source': source})
                if source == 'fincept_api':
                    data = self._get_fincept_news_data(category, limit)
                elif source == 'dummy_news':
                    data = self._get_dummy_news_data(category, limit)
                else:
                    data = self._get_fallback_news_data(category, limit)
                if data.get('success'):
                    self.set_cache(cache_key, data, 600)
                    logger.info('News data retrieved successfully', context={'category': category, 'articles': len(data.get('articles', []))})
                else:
                    self._errors += 1
                    logger.warning('News data retrieval failed', context={'category': category, 'error': data.get('error')})
                return data
        except Exception as e:
            self._errors += 1
            logger.error('News data retrieval error', context={'category': category, 'error': str(e)}, exc_info=True)
            return {'success': False, 'error': f'Error fetching news data: {str(e)}', 'source': 'error', 'category': category}

    @monitor_performance
    def get_crypto_data(self, symbol: str, period: str='1d') -> Dict[str, Any]:
        """Universal crypto data retrieval"""
        try:
            with operation(f'Get crypto data for {symbol}'):
                cache_key = f'crypto_{symbol}_{period}'
                cached_data = self.get_cache(cache_key)
                if cached_data:
                    return cached_data
                source = self.get_data_source('crypto')
                self._api_calls += 1
                logger.debug('Fetching crypto data', context={'symbol': symbol, 'period': period, 'source': source})
                if source == 'fincept_api':
                    data = self._get_fincept_crypto_data(symbol, period)
                elif source == 'alpha_vantage_data':
                    data = asyncio.run(self._get_alpha_vantage_crypto_data(symbol, period))
                else:
                    data = self._get_fallback_crypto_data(symbol, period)
                if data.get('success'):
                    self.set_cache(cache_key, data, 120)
                    logger.info('Crypto data retrieved successfully', context={'symbol': symbol, 'source': source})
                else:
                    self._errors += 1
                    logger.warning('Crypto data retrieval failed', context={'symbol': symbol, 'error': data.get('error')})
                return data
        except Exception as e:
            self._errors += 1
            logger.error('Crypto data retrieval error', context={'symbol': symbol, 'error': str(e)}, exc_info=True)
            return {'success': False, 'error': f'Error fetching crypto data: {str(e)}', 'source': 'error', 'symbol': symbol}

    async def _get_alpha_vantage_crypto_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Get crypto data from Alpha Vantage provider"""
        try:
            provider = self._get_provider_instance('alpha_vantage_data')
            if not provider:
                return {'success': False, 'error': 'Alpha Vantage provider not configured', 'source': 'alpha_vantage_data'}
            return await provider.get_crypto_data(symbol, period)
        except Exception as e:
            logger.error('Alpha Vantage crypto data error', context={'symbol': symbol, 'error': str(e)}, exc_info=True)
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    @monitor_performance
    def get_economic_data(self, indicator: str, country: str='US') -> Dict[str, Any]:
        """Universal economic data retrieval"""
        try:
            with operation(f'Get economic data for {indicator}'):
                cache_key = f'economic_{indicator}_{country}'
                cached_data = self.get_cache(cache_key)
                if cached_data:
                    return cached_data
                source = self.get_data_source('economic')
                self._api_calls += 1
                logger.debug('Fetching economic data', context={'indicator': indicator, 'country': country, 'source': source})
                if source == 'fincept_api':
                    data = self._get_fincept_economic_data(indicator, country)
                else:
                    data = self._get_fallback_economic_data(indicator, country)
                if data.get('success'):
                    self.set_cache(cache_key, data, 3600)
                    logger.info('Economic data retrieved successfully', context={'indicator': indicator, 'country': country})
                else:
                    self._errors += 1
                    logger.warning('Economic data retrieval failed', context={'indicator': indicator, 'error': data.get('error')})
                return data
        except Exception as e:
            self._errors += 1
            logger.error('Economic data retrieval error', context={'indicator': indicator, 'error': str(e)}, exc_info=True)
            return {'success': False, 'error': f'Error fetching economic data: {str(e)}', 'source': 'error', 'indicator': indicator}

    def _get_yfinance_stock_data(self, symbol: str, period: str, interval: str) -> Dict[str, Any]:
        """Get stock data from Yahoo Finance"""
        try:
            logger.debug('Calling yfinance API', context={'symbol': symbol})
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            if hist.empty:
                logger.warning('No data found in yfinance', context={'symbol': symbol})
                return {'success': False, 'error': f'No data found for symbol {symbol}', 'source': 'yfinance'}
            data = {'success': True, 'source': 'yfinance', 'symbol': symbol, 'data': {'timestamps': [t.isoformat() for t in hist.index], 'open': hist['Open'].tolist(), 'high': hist['High'].tolist(), 'low': hist['Low'].tolist(), 'close': hist['Close'].tolist(), 'volume': hist['Volume'].tolist()}, 'current_price': float(hist['Close'][-1]) if len(hist) > 0 else None, 'fetched_at': datetime.now().isoformat()}
            logger.debug('yfinance data parsed successfully', context={'symbol': symbol, 'data_points': len(hist)})
            return data
        except Exception as e:
            logger.error('yfinance API error', context={'symbol': symbol, 'error': str(e)}, exc_info=True)
            return {'success': False, 'error': f'YFinance error: {str(e)}', 'source': 'yfinance', 'symbol': symbol}

    def _get_yfinance_forex_data(self, pair: str, period: str) -> Dict[str, Any]:
        """Get forex data from Yahoo Finance"""
        try:
            yahoo_pair = f'{pair}=X' if not pair.endswith('=X') else pair
            logger.debug('Calling yfinance forex API', context={'pair': pair, 'yahoo_pair': yahoo_pair})
            ticker = yf.Ticker(yahoo_pair)
            hist = ticker.history(period=period)
            if hist.empty:
                logger.warning('No forex data found in yfinance', context={'pair': pair})
                return {'success': False, 'error': f'No forex data found for pair {pair}', 'source': 'yfinance'}
            data = {'success': True, 'source': 'yfinance', 'pair': pair, 'data': {'timestamps': [t.isoformat() for t in hist.index], 'rates': hist['Close'].tolist()}, 'current_rate': float(hist['Close'][-1]) if len(hist) > 0 else None, 'fetched_at': datetime.now().isoformat()}
            logger.debug('yfinance forex data parsed successfully', context={'pair': pair, 'data_points': len(hist)})
            return data
        except Exception as e:
            logger.error('yfinance forex API error', context={'pair': pair, 'error': str(e)}, exc_info=True)
            return {'success': False, 'error': f'YFinance forex error: {str(e)}', 'source': 'yfinance', 'pair': pair}

    def _get_fincept_stock_data(self, symbol: str, period: str, interval: str) -> Dict[str, Any]:
        """Get stock data from Fincept API (dummy implementation)"""
        logger.debug('Using dummy Fincept API for stock data', context={'symbol': symbol})
        return {'success': True, 'source': 'fincept_api', 'symbol': symbol, 'data': {'timestamps': [datetime.now().isoformat()], 'open': [100.0], 'high': [105.0], 'low': [98.0], 'close': [102.0], 'volume': [1000000]}, 'current_price': 102.0, 'fetched_at': datetime.now().isoformat(), 'note': 'This is dummy Fincept API data'}

    def _get_fincept_forex_data(self, pair: str, period: str) -> Dict[str, Any]:
        """Get forex data from Fincept API (dummy implementation)"""
        logger.debug('Using dummy Fincept API for forex data', context={'pair': pair})
        return {'success': True, 'source': 'fincept_api', 'pair': pair, 'data': {'timestamps': [datetime.now().isoformat()], 'rates': [1.2345]}, 'current_rate': 1.2345, 'fetched_at': datetime.now().isoformat(), 'note': 'This is dummy Fincept forex data'}

    def _get_fincept_news_data(self, category: str, limit: int) -> Dict[str, Any]:
        """Get news data from Fincept API (dummy implementation)"""
        logger.debug('Using dummy Fincept API for news data', context={'category': category, 'limit': limit})
        return {'success': True, 'source': 'fincept_api', 'category': category, 'articles': [{'title': 'Market Update: Tech Stocks Rally', 'summary': "Technology stocks gained momentum in today's trading session.", 'url': 'https://example.com/news/1', 'published_at': datetime.now().isoformat(), 'source': 'Fincept News'}], 'total': limit, 'fetched_at': datetime.now().isoformat(), 'note': 'This is dummy Fincept news data'}

    def _get_dummy_news_data(self, category: str, limit: int) -> Dict[str, Any]:
        """Get dummy news data"""
        logger.debug('Generating dummy news data', context={'category': category, 'limit': limit})
        dummy_articles = []
        for i in range(min(limit, 5)):
            dummy_articles.append({'title': f'Sample Financial News Article {i + 1}', 'summary': f'This is a sample news summary for {category} category.', 'url': f'https://example.com/news/{i + 1}', 'published_at': (datetime.now() - timedelta(hours=i)).isoformat(), 'source': 'Sample News'})
        return {'success': True, 'source': 'dummy_news', 'category': category, 'articles': dummy_articles, 'total': len(dummy_articles), 'fetched_at': datetime.now().isoformat()}

    def _get_fincept_crypto_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Get crypto data from Fincept API (dummy implementation)"""
        logger.debug('Using dummy Fincept API for crypto data', context={'symbol': symbol})
        return {'success': True, 'source': 'fincept_api', 'symbol': symbol, 'data': {'timestamps': [datetime.now().isoformat()], 'prices': [50000.0]}, 'current_price': 50000.0, 'fetched_at': datetime.now().isoformat(), 'note': 'This is dummy Fincept crypto data'}

    def _get_fincept_economic_data(self, indicator: str, country: str) -> Dict[str, Any]:
        """Get economic data from Fincept API (dummy implementation)"""
        logger.debug('Using dummy Fincept API for economic data', context={'indicator': indicator, 'country': country})
        return {'success': True, 'source': 'fincept_api', 'indicator': indicator, 'country': country, 'data': {'timestamps': [datetime.now().isoformat()], 'values': [2.5]}, 'current_value': 2.5, 'fetched_at': datetime.now().isoformat(), 'note': 'This is dummy Fincept economic data'}

    def _get_fallback_stock_data(self, symbol: str, period: str, interval: str) -> Dict[str, Any]:
        """Fallback to YFinance for stock data"""
        logger.info('Using fallback stock data source', context={'symbol': symbol})
        return self._get_yfinance_stock_data(symbol, period, interval)

    def _get_fallback_forex_data(self, pair: str, period: str) -> Dict[str, Any]:
        """Fallback to YFinance for forex data"""
        logger.info('Using fallback forex data source', context={'pair': pair})
        return self._get_yfinance_forex_data(pair, period)

    def _get_fallback_news_data(self, category: str, limit: int) -> Dict[str, Any]:
        """Fallback to dummy news"""
        logger.info('Using fallback news data source', context={'category': category})
        return self._get_dummy_news_data(category, limit)

    def _get_fallback_crypto_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Fallback crypto data"""
        logger.warning('No fallback available for crypto data', context={'symbol': symbol})
        return {'success': False, 'error': 'No fallback available for crypto data', 'source': 'fallback'}

    def _get_fallback_economic_data(self, indicator: str, country: str) -> Dict[str, Any]:
        """Fallback economic data"""
        logger.warning('No fallback available for economic data', context={'indicator': indicator, 'country': country})
        return {'success': False, 'error': 'No fallback available for economic data', 'source': 'fallback'}

    @monitor_performance
    def test_data_source(self, source_name: str, config: Dict[str, Any]=None) -> Dict[str, Any]:
        """Test if a data source is working"""
        try:
            with operation(f'Test data source {source_name}'):
                logger.info('Testing data source', context={'source': source_name})
                if source_name == 'yfinance':
                    result = self._get_yfinance_stock_data('AAPL', '1d', '1d')
                    success = result.get('success', False)
                    logger.info('yfinance test completed', context={'success': success})
                    return {'success': success, 'message': 'YFinance connection successful' if success else result.get('error'), 'response_time': '< 1s'}
                elif source_name == 'alpha_vantage_data':
                    provider = self._get_provider_instance('alpha_vantage_data')
                    if provider:
                        result = asyncio.run(provider.verify_api_key())
                        logger.info('alpha_vantage_data test completed', context={'success': result.get('valid', False)})
                        return {'success': result.get('valid', False), 'message': result.get('message', result.get('error', 'Unknown')), 'response_time': '< 2s'}
                    else:
                        return {'success': False, 'message': 'Alpha Vantage provider not configured', 'response_time': 'immediate'}
                elif source_name == 'fincept_api':
                    logger.info('fincept_api test completed', context={'success': True})
                    return {'success': True, 'message': 'Fincept API connection successful (dummy)', 'response_time': '< 1s'}
                else:
                    logger.info('Generic source test completed', context={'source': source_name})
                    return {'success': True, 'message': f'{source_name} test successful (dummy)', 'response_time': '< 1s'}
        except Exception as e:
            logger.error('Data source test failed', context={'source': source_name, 'error': str(e)}, exc_info=True)
            return {'success': False, 'message': f'Test failed: {str(e)}', 'response_time': 'timeout'}

    def get_available_sources(self) -> Dict[str, Any]:
        """Get all available data sources"""
        logger.debug('Retrieved available sources', context={'count': len(self.available_sources)})
        return self.available_sources

    def get_source_config(self, source_name: str) -> Dict[str, Any]:
        """Get configuration for a specific source"""
        config = self.config.get('source_configs', {}).get(source_name, {})
        logger.debug('Retrieved source config', context={'source': source_name, 'has_config': bool(config)})
        return config

    def get_current_mappings(self) -> Dict[str, str]:
        """Get current data type to source mappings"""
        mappings = {}
        for data_type in ['stocks', 'forex', 'crypto', 'news', 'economic']:
            mappings[data_type] = self.get_data_source(data_type)
        logger.debug('Retrieved current mappings', context={'mappings_count': len(mappings)})
        return mappings

    @monitor_performance
    def import_csv_data(self, file_path: str, data_type: str, column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Import data from CSV file"""
        try:
            with operation(f'Import CSV data from {file_path}'):
                logger.info('Starting CSV import', context={'file_path': file_path, 'data_type': data_type})
                df = pd.read_csv(file_path)
                mapped_data = {}
                for standard_col, csv_col in column_mapping.items():
                    if csv_col in df.columns:
                        mapped_data[standard_col] = df[csv_col].tolist()
                    else:
                        logger.warning('Column not found in CSV', context={'expected_column': csv_col, 'available_columns': list(df.columns)})
                result = {'success': True, 'source': 'csv_import', 'data_type': data_type, 'data': mapped_data, 'row_count': len(df), 'imported_at': datetime.now().isoformat()}
                logger.info('CSV import completed successfully', context={'rows_imported': len(df), 'columns_mapped': len(mapped_data)})
                return result
        except Exception as e:
            logger.error('CSV import failed', context={'file_path': file_path, 'error': str(e)}, exc_info=True)
            return {'success': False, 'error': f'CSV import error: {str(e)}', 'source': 'csv_import'}

    def clear_cache(self, data_type: str=None):
        """Clear cache for specific data type or all"""
        try:
            with self._cache_lock:
                if data_type:
                    keys_to_remove = [k for k in self.cache.keys() if k.startswith(data_type)]
                    for key in keys_to_remove:
                        del self.cache[key]
                        if key in self.cache_expiry:
                            del self.cache_expiry[key]
                    logger.info('Cache cleared for data type', context={'data_type': data_type, 'keys_removed': len(keys_to_remove)})
                else:
                    cache_size = len(self.cache)
                    self.cache.clear()
                    self.cache_expiry.clear()
                    logger.info('All cache cleared', context={'items_removed': cache_size})
        except Exception as e:
            logger.error('Cache clear failed', context={'data_type': data_type, 'error': str(e)}, exc_info=True)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            with self._cache_lock:
                total_items = len(self.cache)
                expired_items = sum((1 for k in self.cache.keys() if not self.is_cache_valid(k)))
                valid_items = total_items - expired_items
                total_requests = self._cache_hits + self._cache_misses
                hit_rate = self._cache_hits / total_requests * 100 if total_requests > 0 else 0
                stats = {'total_items': total_items, 'valid_items': valid_items, 'expired_items': expired_items, 'cache_hits': self._cache_hits, 'cache_misses': self._cache_misses, 'hit_rate_percent': round(hit_rate, 2), 'api_calls': self._api_calls, 'error_count': self._errors, 'memory_usage_estimate': f'{len(str(self.cache))} bytes'}
                logger.debug('Cache statistics calculated', context=stats)
                return stats
        except Exception as e:
            logger.error('Failed to calculate cache stats', context={'error': str(e)}, exc_info=True)
            return {'error': str(e)}

    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        try:
            with operation('Reset to defaults'):
                logger.info('Resetting configuration to defaults')
                self.config = {'data_mappings': self.default_sources.copy(), 'source_configs': {}}
                self.save_configuration()
                self.clear_cache()
                self._cache_hits = 0
                self._cache_misses = 0
                self._api_calls = 0
                self._errors = 0
                logger.info('Configuration reset completed')
        except Exception as e:
            logger.error('Failed to reset configuration', context={'error': str(e)}, exc_info=True)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            cache_stats = self.get_cache_stats()
            stats = {'data_source_manager': {'total_api_calls': self._api_calls, 'total_errors': self._errors, 'error_rate_percent': self._errors / self._api_calls * 100 if self._api_calls > 0 else 0, 'uptime_seconds': time.time() - getattr(self, '_start_time', time.time())}, 'cache_performance': cache_stats, 'active_sources': list(self.config.get('data_mappings', {}).values()), 'available_sources': len(self.available_sources)}
            logger.debug('Performance statistics generated', context={'total_api_calls': self._api_calls, 'error_rate': stats['data_source_manager']['error_rate_percent']})
            return stats
        except Exception as e:
            logger.error('Failed to generate performance stats', context={'error': str(e)}, exc_info=True)
            return {'error': str(e)}

    @lru_cache(maxsize=100)
    def get_supported_data_types(self, source_name: str) -> List[str]:
        """Get supported data types for a source - cached"""
        supported = self.available_sources.get(source_name, {}).get('supports', [])
        logger.debug('Retrieved supported data types', context={'source': source_name, 'types': supported})
        return supported

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration"""
        try:
            with operation('Validate configuration'):
                logger.info('Starting configuration validation')
                issues = []
                warnings = []
                for data_type, source_name in self.config.get('data_mappings', {}).items():
                    if source_name not in self.available_sources:
                        issues.append(f"Unknown source '{source_name}' mapped to '{data_type}'")
                    elif data_type not in self.available_sources[source_name]['supports']:
                        issues.append(f"Source '{source_name}' doesn't support data type '{data_type}'")
                essential_types = ['stocks', 'forex', 'news']
                for data_type in essential_types:
                    if data_type not in self.config.get('data_mappings', {}):
                        warnings.append(f"No source configured for essential data type '{data_type}'")
                for source_name, config in self.config.get('source_configs', {}).items():
                    if source_name not in self.available_sources:
                        warnings.append(f"Configuration exists for unknown source '{source_name}'")
                    elif self.available_sources[source_name]['requires_auth'] and (not config):
                        warnings.append(f"Source '{source_name}' requires authentication but no config provided")
                validation_result = {'valid': len(issues) == 0, 'issues': issues, 'warnings': warnings, 'total_issues': len(issues), 'total_warnings': len(warnings), 'validated_at': datetime.now().isoformat()}
                if issues:
                    logger.warning('Configuration validation found issues', context={'issues': len(issues), 'warnings': len(warnings)})
                else:
                    logger.info('Configuration validation passed', context={'warnings': len(warnings)})
                return validation_result
        except Exception as e:
            logger.error('Configuration validation failed', context={'error': str(e)}, exc_info=True)
            return {'valid': False, 'error': str(e), 'validated_at': datetime.now().isoformat()}

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on data source manager"""
        try:
            with operation('Health check'):
                logger.debug('Starting health check')
                health_status = {'status': 'healthy', 'timestamp': datetime.now().isoformat(), 'cache_functional': False, 'configuration_valid': False, 'primary_sources_available': [], 'issues': []}
                try:
                    test_key = 'health_check_test'
                    test_data = {'test': True}
                    self.set_cache(test_key, test_data, 1)
                    retrieved = self.get_cache(test_key)
                    health_status['cache_functional'] = retrieved == test_data
                except Exception as cache_error:
                    health_status['issues'].append(f'Cache test failed: {str(cache_error)}')
                validation = self.validate_configuration()
                health_status['configuration_valid'] = validation['valid']
                if not validation['valid']:
                    health_status['issues'].extend(validation['issues'])
                primary_sources = ['yfinance', 'alpha_vantage_data']
                for source in primary_sources:
                    try:
                        test_result = self.test_data_source(source)
                        if test_result['success']:
                            health_status['primary_sources_available'].append(source)
                    except Exception as source_error:
                        health_status['issues'].append(f'Source {source} test failed: {str(source_error)}')
                if health_status['issues']:
                    health_status['status'] = 'degraded' if health_status['cache_functional'] else 'unhealthy'
                logger.info('Health check completed', context={'status': health_status['status'], 'issues': len(health_status['issues'])})
                return health_status
        except Exception as e:
            logger.error('Health check failed', context={'error': str(e)}, exc_info=True)
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def __repr__(self) -> str:
        """String representation for debugging"""
        return f'DataSourceManager(sources={len(self.available_sources)}, cache_items={len(self.cache)}, api_calls={self._api_calls}, errors={self._errors})'

    def cleanup(self):
        """Clean up resources"""
        try:
            with operation('DataSourceManager cleanup'):
                logger.info('Starting DataSourceManager cleanup')
                self.clear_cache()
                for provider in self._provider_cache.values():
                    if hasattr(provider, 'close'):
                        try:
                            if asyncio.iscoroutinefunction(provider.close):
                                asyncio.run(provider.close())
                            else:
                                provider.close()
                        except Exception as e:
                            logger.debug('Error closing provider', context={'error': str(e)})
                self._provider_cache.clear()
                self.get_supported_data_types.cache_clear()
                self.save_configuration()
                logger.info('DataSourceManager cleanup completed', context={'final_api_calls': self._api_calls, 'final_errors': self._errors})
        except Exception as e:
            logger.error('DataSourceManager cleanup failed', context={'error': str(e)}, exc_info=True)

class DataGovIndiaTab:
    """DearPyGUI tab for DataGovIndia economic data."""

    def __init__(self, app=None):
        self.app = app
        self.api_key = None
        self.resources = []
        self.current_page = 0
        self.items_per_page = 5
        self.current_resource_id = None
        self._resource_cache = {}
        info('DataGovIndia tab initialized', context={'db_path': DB_PATH})

    def get_label(self):
        return 'Economic Data'

    def create_content(self):
        dpg.add_text('DataGovIndia Economic Data', tag='dg_header')
        with dpg.group(horizontal=True):
            dpg.add_input_text(label='API Key', tag='dg_api_key_input', width=400, hint='Enter your API key here')
            dpg.add_button(label='Validate Key', callback=self._on_validate)
        dpg.add_separator()
        dpg.add_button(label='Initialize/Sync Metadata', callback=self._on_sync)
        dpg.add_separator()
        with dpg.collapsing_header(label='Select Resource', default_open=True):
            dpg.add_combo(items=[], tag='resource_selector', width=450, callback=self._on_select)
            with dpg.group(horizontal=True):
                dpg.add_button(label='Previous', callback=lambda s, a: self._on_page(-1))
                dpg.add_button(label='Next', callback=lambda s, a: self._on_page(1))
        dpg.add_separator()
        dpg.add_text('Resource Info')
        with dpg.table(tag='resource_info_table', header_row=True, row_background=True, height=150, width=-1):
            dpg.add_table_column(label='Attribute')
            dpg.add_table_column(label='Value')
        dpg.add_text('Resource Data')
        with dpg.table(tag='resource_data_table', header_row=True, row_background=True, height=150, width=-1):
            dpg.add_table_column(label='Message')
        dpg.add_separator()
        dpg.add_button(label='Download Selected Data', callback=self._on_download)
        dpg.add_text('', tag='dg_status')

    def _on_validate(self, sender, app_data):
        key = dpg.get_value('dg_api_key_input').strip()
        threading.Thread(target=self._validate_sync, args=(key,), daemon=True).start()

    def _validate_sync(self, key):
        if not key:
            dpg.set_value('dg_status', ' API key is empty.')
            return
        with operation('api_key_validation'):
            try:
                valid = check_api_key(key)
                if valid:
                    self.api_key = key
                    info('API key validated successfully')
                    dpg.set_value('dg_status', ' API key validated.')
                else:
                    warning('API key validation failed - invalid key')
                    dpg.set_value('dg_status', ' Invalid API key.')
            except Exception as e:
                error('API key validation error', context={'error': str(e)}, exc_info=True)
                dpg.set_value('dg_status', ' Invalid API key.')

    def _on_sync(self, sender, app_data):
        if not self.api_key:
            dpg.set_value('dg_status', ' Please validate API key first.')
            warning('Metadata sync attempted without API key')
            return
        threading.Thread(target=self._sync_metadata, daemon=True).start()

    @monitor_performance
    def _sync_metadata(self):
        dpg.set_value('dg_status', ' Syncing metadata... This may take a while.')
        with operation('metadata_sync', context={'db_path': DB_PATH}):
            datagovin = DataGovIndia(api_key=self.api_key, db_path=DB_PATH)
            try:
                datagovin.sync_metadata()
                info('Metadata sync completed', context={'db_path': DB_PATH})
                dpg.set_value('dg_status', f' Metadata synced to {DB_PATH}.')
            except Exception as e:
                error('Metadata sync failed', context={'error': str(e)}, exc_info=True)
                dpg.set_value('dg_status', f' Sync error: {e}')
                return
        self._load_resources_from_db()

    def _load_resources_from_db(self):
        """Load resources from database with error handling and caching"""
        try:
            with operation('load_resources_from_db'):
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute('SELECT title, resource_id FROM resources')
                rows = cursor.fetchall()
                conn.close()
                self.resources = [{'title': r[0], 'resource_id': r[1]} for r in rows]
                info('Resources loaded from database', context={'count': len(self.resources)})
        except Exception as e:
            error('Failed to load resources from database', context={'error': str(e)}, exc_info=True)
            dpg.set_value('dg_status', f' DB load error: {e}')
            self.resources = []
        self._update_resources()

    def _update_resources(self):
        """Update resource selector with pagination"""
        start = self.current_page * self.items_per_page
        end = start + self.items_per_page
        items = [f'{r['title']} - {r['resource_id']}' for r in self.resources[start:end]]
        dpg.configure_item('resource_selector', items=items)
        if items:
            dpg.set_value('resource_selector', items[0])
            self._on_select(None, items[0])
        total = len(self.resources)
        dpg.set_value('dg_status', f'Showing {start + 1}-{min(end, total)} of {total}')

    def _on_page(self, delta):
        """Handle pagination with bounds checking"""
        max_page = max((len(self.resources) - 1) // self.items_per_page, 0)
        new_page = self.current_page + delta
        self.current_page = min(max(new_page, 0), max_page)
        self._update_resources()

    def _on_select(self, sender, value):
        """Handle resource selection with caching"""
        if not value:
            return
        self.current_resource_id = value.rsplit(' - ', 1)[-1]
        threading.Thread(target=self._fetch_info_sync, args=(self.current_resource_id,), daemon=True).start()
        threading.Thread(target=self._fetch_data_sync, args=(self.current_resource_id,), daemon=True).start()

    def _fetch_info_sync(self, res_id):
        """Fetch resource info with caching"""
        table = 'resource_info_table'
        dpg.delete_item(table, children_only=True)
        if res_id in self._resource_cache:
            info_data = self._resource_cache[res_id]
        else:
            try:
                with operation('fetch_resource_info', context={'resource_id': res_id}):
                    info_data = DataGovIndia(api_key=self.api_key, db_path=DB_PATH).get_resource_info(res_id)
                    self._resource_cache[res_id] = info_data
            except Exception as e:
                error('Failed to fetch resource info', context={'resource_id': res_id, 'error': str(e)}, exc_info=True)
                info_data = {'Error': str(e)}
        dpg.add_table_column(label='Attribute', parent=table)
        dpg.add_table_column(label='Value', parent=table)
        for k, v in info_data.items():
            if k == 'field':
                continue
            val = ', '.join(v) if isinstance(v, list) else str(v or 'N/A')
            dpg.add_table_row(parent=table, children=[dpg.add_text(k), dpg.add_text(val)])

    def _fetch_data_sync(self, res_id):
        """Fetch resource data with error handling"""
        table = 'resource_data_table'
        dpg.clear_table(table)
        try:
            with operation('fetch_resource_data', context={'resource_id': res_id, 'limit': 10}):
                df = DataGovIndia(api_key=self.api_key, db_path=DB_PATH).get_data(res_id, 10)
        except Exception as e:
            error('Failed to fetch resource data', context={'resource_id': res_id, 'error': str(e)}, exc_info=True)
            df = None
            err = str(e)
        if df is None or df.empty:
            dpg.add_table_column(label='Message', parent=table)
            error_msg = err if df is None else 'No data available.'
            dpg.add_table_row(parent=table, children=[dpg.add_text(error_msg)])
            return
        for col in df.columns:
            dpg.add_table_column(label=str(col), parent=table)
        for _, row in df.head(10).iterrows():
            cells = [dpg.add_text(str(x)) for x in row]
            dpg.add_table_row(parent=table, children=cells)

    def _on_download(self, sender, app_data):
        """Initiate data download"""
        if not self.current_resource_id:
            dpg.set_value('dg_status', ' Select a resource.')
            warning('Download attempted without resource selection')
            return
        filename = f'resource_{self.current_resource_id}.csv'
        threading.Thread(target=self._download_sync, args=(self.current_resource_id, filename), daemon=True).start()

    @monitor_performance
    def _download_sync(self, res_id, filename):
        """Download resource data to CSV file"""
        try:
            with operation('download_resource_data', context={'resource_id': res_id, 'filename': filename}):
                df = DataGovIndia(api_key=self.api_key).get_data(res_id)
                if df is None or df.empty:
                    warning('No data available for download', context={'resource_id': res_id})
                    dpg.set_value('dg_status', ' No data to download.')
                    return
                path = os.path.abspath(filename)
                df.to_csv(path, index=False)
                info('Data downloaded successfully', context={'resource_id': res_id, 'filename': filename, 'path': path, 'rows': len(df)})
                dpg.set_value('dg_status', f' Downloaded: {path}')
        except Exception as e:
            error('Data download failed', context={'resource_id': res_id, 'filename': filename, 'error': str(e)}, exc_info=True)
            dpg.set_value('dg_status', f' Download error: {e}')

    def cleanup(self):
        """Cleanup resources when tab is closed"""
        try:
            self._resource_cache.clear()
            info('DataGovIndia tab cleanup completed')
        except Exception as e:
            error('Error during tab cleanup', context={'error': str(e)}, exc_info=True)

