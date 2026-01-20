# Cluster 39

class FinanceNotificationSystem:
    """Main notification system for finance terminal"""
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.config = NotificationConfig()
        self.rate_limiter = NotificationRateLimiter(self.config)
        self.metrics = NotificationMetrics()
        self.available = NOTIFYPY_AVAILABLE and self.config.enabled and (not self.config.silent_mode)
        if LOGGER_AVAILABLE:
            if self.available:
                info('Notification system initialized', module='notifications')
            else:
                warning('Notification system disabled or unavailable', module='notifications')

    def _create_notification(self, title: str, message: str, level: NotificationLevel) -> Optional[Notify]:
        """Create a notification object"""
        if not self.available:
            return None
        try:
            notification = Notify()
            notification.title = title
            notification.message = message
            notification.application_name = self.config.app_name
            if self.config.app_icon:
                notification.icon = self.config.app_icon
            return notification
        except Exception as e:
            if LOGGER_AVAILABLE:
                error(f'Failed to create notification: {e}', module='notifications')
            self.metrics.record_failed()
            return None

    def _send_notification(self, title: str, message: str, level: NotificationLevel, module: Optional[str]=None, **kwargs) -> bool:
        """Core notification sending method"""
        if level.value not in self.config.enabled_levels:
            return False
        if not self.rate_limiter.should_allow(title, message, level):
            self.metrics.record_rate_limited()
            if LOGGER_AVAILABLE and self.config.debug_notifications:
                debug(f'Rate limited notification: {title}', module='notifications')
            return False
        if module:
            tab_prefix = self.config.get_tab_prefix(module)
            title = f'{tab_prefix} {title}'
        if LOGGER_AVAILABLE:
            info(f'Sending notification: {title}', module='notifications', context={'level': level.value, 'source_module': module})
        notification = self._create_notification(title, message, level)
        if notification:
            try:
                notification.send()
                self.metrics.record_sent(level)
                return True
            except Exception as e:
                if LOGGER_AVAILABLE:
                    error(f'Failed to send notification: {e}', module='notifications')
                self.metrics.record_failed()
                return False
        return False

    def debug(self, title: str, message: str, module: Optional[str]=None, **kwargs) -> bool:
        """Send debug notification"""
        if not self.config.debug_notifications:
            return False
        return self._send_notification(title, message, NotificationLevel.DEBUG, module, **kwargs)

    def info(self, title: str, message: str, module: Optional[str]=None, **kwargs) -> bool:
        """Send info notification"""
        return self._send_notification(title, message, NotificationLevel.INFO, module, **kwargs)

    def success(self, title: str, message: str, module: Optional[str]=None, **kwargs) -> bool:
        """Send success notification"""
        return self._send_notification(title, message, NotificationLevel.SUCCESS, module, **kwargs)

    def warning(self, title: str, message: str, module: Optional[str]=None, **kwargs) -> bool:
        """Send warning notification"""
        return self._send_notification(title, message, NotificationLevel.WARNING, module, **kwargs)

    def error(self, title: str, message: str, module: Optional[str]=None, **kwargs) -> bool:
        """Send error notification"""
        return self._send_notification(title, message, NotificationLevel.ERROR, module, **kwargs)

    def critical(self, title: str, message: str, module: Optional[str]=None, **kwargs) -> bool:
        """Send critical notification"""
        return self._send_notification(title, message, NotificationLevel.CRITICAL, module, **kwargs)

    def trade_executed(self, symbol: str, action: str, quantity: int, price: float, module: Optional[str]='trading') -> bool:
        """Template for trade execution notifications"""
        title = 'Trade Executed'
        message = f'{action.upper()} {quantity} {symbol} @ ${price:.2f}'
        return self.success(title, message, module)

    def price_alert(self, symbol: str, current_price: float, target_price: float, condition: str, module: Optional[str]='alerts') -> bool:
        """Template for price alert notifications"""
        title = f'Price Alert: {symbol}'
        message = f'Price ${current_price:.2f} {condition} target ${target_price:.2f}'
        return self.warning(title, message, module)

    def connection_status(self, service: str, status: str, module: Optional[str]='api') -> bool:
        """Template for connection status notifications"""
        title = f'Connection {status.title()}'
        message = f'{service} connection is now {status.lower()}'
        if status.lower() in ['connected', 'restored']:
            return self.success(title, message, module)
        else:
            return self.error(title, message, module)

    def data_update(self, data_type: str, count: int, module: Optional[str]='market') -> bool:
        """Template for data update notifications"""
        title = 'Data Updated'
        message = f'{data_type}: {count} items updated'
        return self.info(title, message, module)

    def system_status(self, component: str, status: str, details: str='', module: Optional[str]='main') -> bool:
        """Template for system status notifications"""
        title = f'System {status.title()}'
        message = f'{component}: {details}' if details else component
        if status.lower() in ['started', 'ready', 'healthy']:
            return self.success(title, message, module)
        elif status.lower() in ['warning', 'degraded']:
            return self.warning(title, message, module)
        else:
            return self.error(title, message, module)

    def enable(self, enabled: bool=True):
        """Enable or disable notifications"""
        self.config.enabled = enabled
        self.available = NOTIFYPY_AVAILABLE and enabled and (not self.config.silent_mode)
        if LOGGER_AVAILABLE:
            status = 'enabled' if enabled else 'disabled'
            info(f'Notifications {status}', module='notifications')

    def set_silent_mode(self, silent: bool=True):
        """Enable or disable silent mode"""
        self.config.silent_mode = silent
        self.available = NOTIFYPY_AVAILABLE and self.config.enabled and (not silent)
        if LOGGER_AVAILABLE:
            mode = 'silent' if silent else 'normal'
            info(f'Notification mode: {mode}', module='notifications')

    def set_debug_notifications(self, enabled: bool=True):
        """Enable or disable debug notifications"""
        self.config.debug_notifications = enabled

    def get_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        stats = self.metrics.get_stats()
        stats.update({'config': {'enabled': self.config.enabled, 'silent_mode': self.config.silent_mode, 'available': self.available, 'rate_limiting': self.config.rate_limit_enabled, 'enabled_levels': list(self.config.enabled_levels)}})
        return stats

    def health_check(self) -> Dict[str, Any]:
        """Check notification system health"""
        try:
            if not NOTIFYPY_AVAILABLE:
                return {'status': 'unavailable', 'reason': 'notifypy not installed'}
            if not self.config.enabled:
                return {'status': 'disabled', 'reason': 'notifications disabled in config'}
            if self.config.silent_mode:
                return {'status': 'silent', 'reason': 'silent mode enabled'}
            test_title = 'Health Check'
            test_message = f'Notification system test at {datetime.now().strftime('%H:%M:%S')}'
            return {'status': 'healthy', 'available': self.available, 'stats': self.get_stats()}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

def debug(msg, **kwargs):
    print(f'DEBUG: {msg}')

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

def get_api_endpoint(endpoint: str) -> str:
    """Get full API endpoint URL"""
    full_url = config.get_full_url(endpoint)
    debug('Retrieved API endpoint URL', module='config', context={'endpoint': endpoint, 'full_url': full_url})
    return full_url

class AutomaticThemeManager:
    """OPTIMIZED: Theme manager with lazy loading and minimal overhead"""

    def __init__(self):
        self.current_theme = 'bloomberg_terminal'
        self.themes = {}
        self.theme_applied = False
        self.cleanup_performed = False
        self.themes_initialized = False
        self._initialization_attempted = False
        self.themes_available = True
        self.terminal_font = None
        info('Bloomberg Terminal theme manager initialized', module='theme')

    def _lazy_initialize(self) -> bool:
        """PERFORMANCE: Only initialize when first used"""
        if self._initialization_attempted:
            return self.themes_initialized
        self._initialization_attempted = True
        try:
            try:
                dpg.get_viewport_width()
            except Exception:
                warning('DearPyGUI context not ready', module='theme')
                return False
            self._setup_font_registry()
            info('Creating authentic Bloomberg Terminal themes', module='theme')
            self._create_bloomberg_terminal_theme()
            self._create_dark_gold_theme()
            self._create_green_terminal_theme()
            self._create_default_theme()
            self.themes_initialized = True
            theme_count = len(self.themes)
            info('Bloomberg themes creation completed', module='theme', context={'themes_created': theme_count})
            return True
        except Exception as e:
            error('Error creating themes', module='theme', context={'error': str(e)}, exc_info=True)
            return False

    def _setup_font_registry(self):
        """Setup font registry and load custom fonts - SAFE VERSION"""
        try:
            import os
            font_path = os.path.join(os.path.dirname(__file__), 'oswald2.ttf')
            print(f'[FONT DEBUG] Looking for font at: {font_path}')
            print(f'[FONT DEBUG] Font exists: {os.path.exists(font_path)}')
            if os.path.exists(font_path):
                try:
                    with dpg.font_registry():
                        self.terminal_font = dpg.add_font(font_path, 18)
                    print(f'[FONT DEBUG] Font created with ID: {self.terminal_font}')
                    info('Oswald2 font loaded successfully', module='theme')
                except Exception as font_error:
                    print(f'[FONT DEBUG] Font creation failed: {font_error}')
                    self.terminal_font = None
            else:
                print('[FONT DEBUG] Font file not found')
                self.terminal_font = None
        except Exception as e:
            print(f'[FONT DEBUG] Font setup failed: {str(e)}')
            self.terminal_font = None

    def _ensure_themes_created(self) -> bool:
        """Create themes only when DearPyGUI context is ready"""
        return self._lazy_initialize()

    def setup_fonts(self):
        """Setup Oswald2 font for terminal - DEPRECATED, use _setup_font_registry instead"""
        if hasattr(self, 'terminal_font') and self.terminal_font:
            try:
                dpg.bind_font(self.terminal_font)
                info('Font re-applied via setup_fonts', module='theme')
                return True
            except Exception as e:
                warning(f'setup_fonts failed: {str(e)}', module='theme')
                return False
        return False

    def _create_green_terminal_theme(self):
        """Modern Green Terminal theme with #48f050 primary color"""
        try:
            if dpg.does_item_exist('green_terminal_theme'):
                dpg.delete_item('green_terminal_theme')
            with dpg.theme(tag='green_terminal_theme') as theme:
                with dpg.theme_component(dpg.mvAll):
                    TERMINAL_BLACK = [10, 10, 10, 255]
                    TERMINAL_DARK_GRAY = [25, 30, 25, 255]
                    TERMINAL_MEDIUM_GRAY = [40, 45, 40, 255]
                    GREEN_PRIMARY = [72, 240, 80, 255]
                    GREEN_HOVER = [92, 255, 100, 255]
                    GREEN_ACTIVE = [52, 220, 60, 255]
                    GREEN_BRIGHT = [100, 255, 110, 255]
                    TERMINAL_WHITE = [240, 255, 245, 255]
                    TERMINAL_GRAY_TEXT = [180, 220, 185, 255]
                    TERMINAL_DISABLED = [120, 140, 125, 255]
                    TERMINAL_RED = [255, 100, 100, 255]
                    TERMINAL_YELLOW = [255, 255, 120, 255]
                    TERMINAL_BLUE = [120, 200, 255, 255]
                    TERMINAL_BORDER = [60, 80, 60, 255]
                    TERMINAL_SEPARATOR = [80, 120, 85, 255]
                    dpg.add_theme_color(dpg.mvThemeCol_WindowBg, TERMINAL_BLACK, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ChildBg, TERMINAL_BLACK, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_PopupBg, TERMINAL_DARK_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, TERMINAL_DARK_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ModalWindowDimBg, [0, 0, 0, 180], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_Text, TERMINAL_WHITE, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, TERMINAL_DISABLED, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg, [72, 240, 80, 100], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_Button, TERMINAL_MEDIUM_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [72, 240, 80, 120], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [72, 240, 80, 180], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBg, TERMINAL_DARK_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, [72, 240, 80, 60], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, [72, 240, 80, 100], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_Header, [72, 240, 80, 150], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, [72, 240, 80, 180], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, [72, 240, 80, 220], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, TERMINAL_MEDIUM_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TableBorderStrong, TERMINAL_SEPARATOR, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight, TERMINAL_BORDER, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, [0, 0, 0, 0], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, [15, 25, 15, 255], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_NavHighlight, [72, 240, 80, 200], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_NavWindowingHighlight, [72, 240, 80, 150], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_NavWindowingDimBg, [60, 80, 60, 100], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_Tab, TERMINAL_DARK_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TabHovered, [72, 240, 80, 120], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TabActive, [72, 240, 80, 180], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, [20, 30, 20, 255], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, [35, 50, 35, 255], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_Border, TERMINAL_BORDER, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, [0, 0, 0, 0], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_Separator, TERMINAL_SEPARATOR, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered, [72, 240, 80, 150], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive, [72, 240, 80, 200], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, TERMINAL_BLACK, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, [72, 240, 80, 120], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, [72, 240, 80, 160], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, [72, 240, 80, 200], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_CheckMark, GREEN_PRIMARY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, [72, 240, 80, 180], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, [72, 240, 80, 220], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ResizeGrip, [72, 240, 80, 80], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ResizeGripHovered, [72, 240, 80, 120], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ResizeGripActive, [72, 240, 80, 160], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TitleBg, TERMINAL_DARK_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, TERMINAL_MEDIUM_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, TERMINAL_DARK_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_DockingPreview, [72, 240, 80, 100], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_DockingEmptyBg, TERMINAL_BLACK, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_PlotLines, GREEN_PRIMARY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_PlotLinesHovered, GREEN_HOVER, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, GREEN_PRIMARY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_PlotHistogramHovered, GREEN_HOVER, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 0, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 0, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 0, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 0, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 0, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 1, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_PopupBorderSize, 1, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_TabBorderSize, 1, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 8, 8, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 4, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 6, 4, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 4, 4, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 4, 2, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_IndentSpacing, 20, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize, 14, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 12, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_WindowTitleAlign, 0.0, 0.5, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.5, 0.5, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_SelectableTextAlign, 0.0, 0.0, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_Alpha, 1.0, category=dpg.mvThemeCat_Core)
            self.themes['green_terminal'] = theme
            info('Green Terminal theme created with primary color', module='theme', context={'primary_color': '#48f050'})
        except Exception as e:
            error('Error creating Green Terminal theme', module='theme', context={'error': str(e)}, exc_info=True)

    def _create_bloomberg_terminal_theme(self):
        """Authentic Bloomberg Terminal theme - Precise color matching"""
        try:
            if dpg.does_item_exist('bloomberg_terminal_theme'):
                dpg.delete_item('bloomberg_terminal_theme')
            with dpg.theme(tag='bloomberg_terminal_theme') as theme:
                with dpg.theme_component(dpg.mvAll):
                    BLOOMBERG_BLACK = [0, 0, 0, 255]
                    BLOOMBERG_DARK_GRAY = [40, 40, 40, 255]
                    BLOOMBERG_MEDIUM_GRAY = [60, 60, 60, 255]
                    BLOOMBERG_ORANGE = [255, 140, 0, 255]
                    BLOOMBERG_ORANGE_HOVER = [255, 165, 0, 255]
                    BLOOMBERG_ORANGE_ACTIVE = [255, 120, 0, 255]
                    BLOOMBERG_WHITE = [255, 255, 255, 255]
                    BLOOMBERG_GRAY_TEXT = [192, 192, 192, 255]
                    BLOOMBERG_DISABLED = [128, 128, 128, 255]
                    BLOOMBERG_RED = [255, 80, 80, 255]
                    BLOOMBERG_GREEN = [0, 255, 100, 255]
                    BLOOMBERG_YELLOW = [255, 255, 100, 255]
                    BLOOMBERG_BLUE = [100, 180, 255, 255]
                    BLOOMBERG_BORDER = [80, 80, 80, 255]
                    BLOOMBERG_SEPARATOR = [100, 100, 100, 255]
                    dpg.add_theme_color(dpg.mvThemeCol_WindowBg, BLOOMBERG_BLACK, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ChildBg, BLOOMBERG_BLACK, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_PopupBg, BLOOMBERG_DARK_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, BLOOMBERG_DARK_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ModalWindowDimBg, [0, 0, 0, 180], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_Text, BLOOMBERG_WHITE, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, BLOOMBERG_DISABLED, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg, [255, 140, 0, 100], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_Button, BLOOMBERG_MEDIUM_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [255, 140, 0, 120], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [255, 140, 0, 180], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBg, BLOOMBERG_DARK_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, [255, 140, 0, 60], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, [255, 140, 0, 100], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_Header, [255, 140, 0, 150], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, [255, 140, 0, 180], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, [255, 140, 0, 220], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, BLOOMBERG_MEDIUM_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TableBorderStrong, BLOOMBERG_SEPARATOR, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight, BLOOMBERG_BORDER, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, [0, 0, 0, 0], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, [20, 20, 20, 255], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_NavHighlight, [255, 140, 0, 200], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_NavWindowingHighlight, [255, 140, 0, 150], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_NavWindowingDimBg, [80, 80, 80, 100], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_Tab, BLOOMBERG_DARK_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TabHovered, [255, 140, 0, 120], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TabActive, [255, 140, 0, 180], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, [30, 30, 30, 255], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, [50, 50, 50, 255], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_Border, BLOOMBERG_BORDER, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, [0, 0, 0, 0], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_Separator, BLOOMBERG_SEPARATOR, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered, [255, 140, 0, 150], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive, [255, 140, 0, 200], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, BLOOMBERG_BLACK, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, [255, 140, 0, 120], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, [255, 140, 0, 160], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, [255, 140, 0, 200], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_CheckMark, BLOOMBERG_ORANGE, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, [255, 140, 0, 180], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, [255, 140, 0, 220], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ResizeGrip, [255, 140, 0, 80], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ResizeGripHovered, [255, 140, 0, 120], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ResizeGripActive, [255, 140, 0, 160], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TitleBg, BLOOMBERG_DARK_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, BLOOMBERG_MEDIUM_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, BLOOMBERG_DARK_GRAY, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_DockingPreview, [255, 140, 0, 100], category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_DockingEmptyBg, BLOOMBERG_BLACK, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_PlotLines, BLOOMBERG_ORANGE, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_PlotLinesHovered, BLOOMBERG_ORANGE_HOVER, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, BLOOMBERG_ORANGE, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_PlotHistogramHovered, BLOOMBERG_ORANGE_HOVER, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 0, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 0, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 0, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 0, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 0, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 1, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_PopupBorderSize, 1, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_TabBorderSize, 1, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 8, 8, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 4, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 6, 4, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 4, 4, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 4, 2, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_IndentSpacing, 20, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize, 14, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 12, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_WindowTitleAlign, 0.0, 0.5, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.5, 0.5, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_SelectableTextAlign, 0.0, 0.0, category=dpg.mvThemeCat_Core)
                    dpg.add_theme_style(dpg.mvStyleVar_Alpha, 1.0, category=dpg.mvThemeCat_Core)
            self.themes['bloomberg_terminal'] = theme
            info('Authentic Bloomberg Terminal theme created', module='theme')
        except Exception as e:
            error('Error creating Bloomberg Terminal theme', module='theme', context={'error': str(e)}, exc_info=True)

    def _create_dark_gold_theme(self):
        """Enhanced dark theme with premium gold accents"""
        try:
            if dpg.does_item_exist('dark_gold_theme'):
                dpg.delete_item('dark_gold_theme')
            with dpg.theme(tag='dark_gold_theme') as theme:
                with dpg.theme_component(dpg.mvAll):
                    DARK_BG = [18, 18, 18, 255]
                    DARK_PANEL = [28, 28, 28, 255]
                    DARK_ELEMENT = [38, 38, 38, 255]
                    GOLD_PRIMARY = [255, 215, 0, 255]
                    GOLD_HOVER = [255, 235, 59, 255]
                    GOLD_ACTIVE = [255, 193, 7, 255]
                    WHITE_TEXT = [255, 255, 255, 255]
                    GRAY_TEXT = [180, 180, 180, 255]
                    DISABLED_TEXT = [120, 120, 120, 255]
                    dpg.add_theme_color(dpg.mvThemeCol_WindowBg, DARK_BG)
                    dpg.add_theme_color(dpg.mvThemeCol_ChildBg, DARK_BG)
                    dpg.add_theme_color(dpg.mvThemeCol_PopupBg, DARK_PANEL)
                    dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, DARK_PANEL)
                    dpg.add_theme_color(dpg.mvThemeCol_Text, WHITE_TEXT)
                    dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, DISABLED_TEXT)
                    dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg, [255, 215, 0, 100])
                    dpg.add_theme_color(dpg.mvThemeCol_Button, DARK_ELEMENT)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [255, 215, 0, 120])
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [255, 215, 0, 180])
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBg, DARK_ELEMENT)
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, [255, 215, 0, 60])
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, [255, 215, 0, 100])
                    dpg.add_theme_color(dpg.mvThemeCol_Header, [255, 215, 0, 150])
                    dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, [255, 215, 0, 180])
                    dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, [255, 215, 0, 220])
                    dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, DARK_ELEMENT)
                    dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, [0, 0, 0, 0])
                    dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, [25, 25, 25, 255])
                    dpg.add_theme_color(dpg.mvThemeCol_Tab, DARK_ELEMENT)
                    dpg.add_theme_color(dpg.mvThemeCol_TabHovered, [255, 215, 0, 120])
                    dpg.add_theme_color(dpg.mvThemeCol_TabActive, [255, 215, 0, 180])
                    dpg.add_theme_color(dpg.mvThemeCol_Border, [70, 70, 70, 255])
                    dpg.add_theme_color(dpg.mvThemeCol_Separator, [100, 100, 100, 255])
                    dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, DARK_BG)
                    dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, [255, 215, 0, 120])
                    dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, [255, 215, 0, 160])
                    dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, [255, 215, 0, 200])
                    dpg.add_theme_color(dpg.mvThemeCol_CheckMark, GOLD_PRIMARY)
                    dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, [255, 215, 0, 180])
                    dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, [255, 215, 0, 220])
                    dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 3)
                    dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 3)
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
                    dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 3)
                    dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 4)
                    dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 6, 4)
            self.themes['dark_gold'] = theme
            info('Enhanced Dark Gold theme created', module='theme')
        except Exception as e:
            error('Error creating Dark Gold theme', module='theme', context={'error': str(e)}, exc_info=True)

    def _create_default_theme(self):
        """Improved default theme"""
        try:
            if dpg.does_item_exist('default_theme'):
                dpg.delete_item('default_theme')
            with dpg.theme(tag='default_theme') as theme:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvThemeCol_WindowBg, [15, 15, 15, 240])
                    dpg.add_theme_color(dpg.mvThemeCol_ChildBg, [20, 20, 20, 255])
                    dpg.add_theme_color(dpg.mvThemeCol_Text, [255, 255, 255, 255])
                    dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, [128, 128, 128, 255])
                    dpg.add_theme_color(dpg.mvThemeCol_Button, [60, 60, 60, 255])
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [80, 80, 80, 255])
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [100, 100, 100, 255])
                    dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
            self.themes['default'] = theme
            info('Improved Default theme created', module='theme')
        except Exception as e:
            error('Error creating Default theme', module='theme', context={'error': str(e)}, exc_info=True)

    @monitor_performance
    def apply_theme_globally(self, theme_name: str) -> bool:
        """Apply theme with enhanced error handling and feedback - NOW WITH LAZY LOADING"""
        try:
            theme_map = {'finance_terminal': 'bloomberg_terminal', 'bloomberg_terminal': 'bloomberg_terminal', 'bloomberg': 'bloomberg_terminal', 'terminal': 'bloomberg_terminal', 'green_terminal': 'green_terminal', 'green': 'green_terminal', 'matrix': 'green_terminal', 'dark_gold': 'dark_gold', 'gold': 'dark_gold', 'default': 'default', 'standard': 'default'}
            actual_theme = theme_map.get(theme_name.lower(), 'bloomberg_terminal')
            if not self._lazy_initialize():
                warning('Cannot apply theme - DearPyGUI context not ready', module='theme', context={'requested_theme': theme_name})
                return False
            if actual_theme not in self.themes:
                available_themes = list(self.themes.keys())
                warning('Theme not found in available themes', module='theme', context={'requested_theme': actual_theme, 'available_themes': available_themes})
                return False
            if self.theme_applied:
                try:
                    dpg.bind_theme(0)
                    debug('Unbound previous theme', module='theme')
                except Exception as e:
                    warning('Warning unbinding previous theme', module='theme', context={'error': str(e)})
            dpg.bind_theme(self.themes[actual_theme])
            self.current_theme = actual_theme
            self.theme_applied = True
            if hasattr(self, 'terminal_font') and self.terminal_font:
                try:
                    dpg.bind_font(self.terminal_font)
                    print(f'[FONT DEBUG] Applied custom font after theme: {self.terminal_font}')
                except Exception as e:
                    print(f'[FONT DEBUG] Font application after theme failed: {e}')
            theme_info = self.get_theme_info()
            info('Successfully applied theme', module='theme', context={'theme_name': theme_info['name'], 'description': theme_info['description']})
            return True
        except Exception as e:
            error('Critical error applying theme', module='theme', context={'theme_name': theme_name, 'error': str(e)}, exc_info=True)
            return False

    def ensure_font_applied(self):
        """Ensure custom font is applied"""
        try:
            if hasattr(self, 'terminal_font') and self.terminal_font:
                dpg.bind_font(self.terminal_font)
                print(f'[FONT DEBUG] Applied font: {self.terminal_font}')
                return True
            else:
                print('[FONT DEBUG] No font to apply')
                return False
        except Exception as e:
            print(f'[FONT DEBUG] Font application failed: {e}')
            return False

    def get_available_themes(self) -> Dict[str, str]:
        """Get comprehensive list of available themes"""
        return {'bloomberg_terminal': 'Bloomberg Terminal - Authentic black/orange professional theme', 'green_terminal': 'Green Terminal - Modern terminal with bright green (#48f050) accents', 'dark_gold': 'Dark Gold - Premium dark theme with gold accents', 'default': 'Default - Clean standard interface theme'}

    def get_current_theme(self) -> Dict[str, Any]:
        """Get current theme name with status"""
        return {'theme': self.current_theme, 'applied': self.theme_applied, 'initialized': self.themes_initialized}

    def create_theme_selector_callback(self, sender, app_data):
        """Enhanced callback for theme selector with error handling"""
        try:
            success = self.apply_theme_globally(app_data)
            if not success:
                error('Failed to apply theme from selector', module='theme', context={'theme': app_data})
        except Exception as e:
            error('Theme selector callback error', module='theme', context={'error': str(e)}, exc_info=True)

    def get_theme_info(self) -> Dict[str, Any]:
        """Get comprehensive information about current theme"""
        theme_info = {'bloomberg_terminal': {'name': 'Bloomberg Terminal', 'description': 'Authentic Bloomberg Terminal theme with precise black background and orange accents', 'style': 'Professional financial terminal', 'colors': {'primary': 'Bloomberg Orange (#FF8C00)', 'background': 'Pure Black (#000000)', 'text': 'White (#FFFFFF)', 'accent': 'Orange variations'}}, 'green_terminal': {'name': 'Green Terminal', 'description': 'Modern terminal theme with bright green primary color and dark background', 'style': 'Matrix-style financial terminal', 'colors': {'primary': 'Bright Green (#48f050)', 'background': 'Deep Black (#0A0A0A)', 'text': 'Green-tinted White (#F0FFF5)', 'accent': 'Green variations'}}, 'dark_gold': {'name': 'Dark Gold', 'description': 'Premium dark theme with luxurious gold accents and enhanced readability', 'style': 'Luxury financial interface', 'colors': {'primary': 'Gold (#FFD700)', 'background': 'Dark Gray (#121212)', 'text': 'White (#FFFFFF)', 'accent': 'Gold variations'}}, 'default': {'name': 'Default', 'description': 'Clean and professional standard interface theme', 'style': 'Standard modern interface', 'colors': {'primary': 'Gray (#606060)', 'background': 'Dark Gray (#0F0F0F)', 'text': 'White (#FFFFFF)', 'accent': 'Gray variations'}}}
        return theme_info.get(self.current_theme, theme_info['bloomberg_terminal'])

    def get_theme_colors(self) -> Dict[str, list]:
        """Get current theme color palette for external use"""
        if self.current_theme == 'bloomberg_terminal':
            return {'background': [0, 0, 0, 255], 'primary': [255, 140, 0, 255], 'text': [255, 255, 255, 255], 'secondary': [192, 192, 192, 255], 'accent': [255, 165, 0, 255], 'success': [0, 255, 100, 255], 'warning': [255, 255, 100, 255], 'error': [255, 80, 80, 255]}
        elif self.current_theme == 'green_terminal':
            return {'background': [10, 10, 10, 255], 'primary': [72, 240, 80, 255], 'text': [240, 255, 245, 255], 'secondary': [180, 220, 185, 255], 'accent': [100, 255, 110, 255], 'success': [72, 240, 80, 255], 'warning': [255, 255, 120, 255], 'error': [255, 100, 100, 255]}
        elif self.current_theme == 'dark_gold':
            return {'background': [18, 18, 18, 255], 'primary': [255, 215, 0, 255], 'text': [255, 255, 255, 255], 'secondary': [180, 180, 180, 255], 'accent': [255, 235, 59, 255]}
        else:
            return {'background': [15, 15, 15, 255], 'primary': [60, 60, 60, 255], 'text': [255, 255, 255, 255], 'secondary': [128, 128, 128, 255]}

    def cleanup(self):
        """Enhanced cleanup with better error handling"""
        if self.cleanup_performed:
            return
        try:
            info('Cleaning up Bloomberg Terminal themes', module='theme')
            self.cleanup_performed = True
            if self.theme_applied:
                try:
                    dpg.bind_theme(0)
                    self.theme_applied = False
                    info('Theme unbound successfully', module='theme')
                except Exception as e:
                    warning('Warning unbinding theme', module='theme', context={'error': str(e)})
            themes_deleted = 0
            for theme_name, theme in self.themes.items():
                try:
                    if dpg.does_item_exist(theme):
                        dpg.delete_item(theme)
                        themes_deleted += 1
                except Exception as e:
                    warning('Warning deleting theme', module='theme', context={'theme_name': theme_name, 'error': str(e)})
            self.themes.clear()
            self.themes_initialized = False
            info('Themes cleaned up successfully', module='theme', context={'themes_deleted': themes_deleted})
        except Exception as e:
            error('Theme cleanup error', module='theme', context={'error': str(e)}, exc_info=True)

    def __del__(self):
        """Enhanced destructor with error handling"""
        try:
            if not self.cleanup_performed:
                self.cleanup()
        except Exception as e:
            warning('Warning in theme manager destructor', module='theme', context={'error': str(e)})

    def reset_to_default(self) -> bool:
        """Reset to default theme safely"""
        try:
            return self.apply_theme_globally('default')
        except Exception as e:
            error('Error resetting to default theme', module='theme', context={'error': str(e)}, exc_info=True)
            return False

    def validate_theme_integrity(self) -> Tuple[bool, str]:
        """Validate that themes are properly configured"""
        try:
            if not self.themes_initialized:
                return (False, 'Themes not initialized')
            required_themes = ['bloomberg_terminal', 'green_terminal', 'dark_gold', 'default']
            missing_themes = [t for t in required_themes if t not in self.themes]
            if missing_themes:
                return (False, f'Missing themes: {missing_themes}')
            return (True, 'All themes validated successfully')
        except Exception as e:
            return (False, f'Validation error: {e}')

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

class AlphaVantageProvider:
    """Alpha Vantage data provider with complete API integration"""
    INTERVALS_DICT = {'m': 'TIME_SERIES_INTRADAY', 'd': 'TIME_SERIES_DAILY', 'W': 'TIME_SERIES_WEEKLY', 'M': 'TIME_SERIES_MONTHLY'}

    def __init__(self, api_key: str, rate_limit: int=5):
        self.api_key = api_key
        self.base_url = 'https://www.alphavantage.co/query'
        self.rate_limit = rate_limit
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), connector=aiohttp.TCPConnector(limit=10))
        return self._session

    def get_interval(self, value: str) -> str:
        """Get the intervals for the Alpha Vantage API"""
        try:
            intervals = {'m': 'min', 'd': 'day', 'W': 'week', 'M': 'month'}
            if len(value) < 2:
                return '1min'
            period_num = value[:-1]
            period_type = value[-1]
            if period_type in intervals:
                return f'{period_num}{intervals[period_type]}'
            else:
                return '1min'
        except Exception as e:
            warning(f'Error parsing interval {value}: {str(e)}', module='AlphaVantageProvider')
            return '1min'

    async def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with common error handling"""
        try:
            session = await self._get_session()
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error(f'Alpha Vantage API HTTP error: {response.status}', module='AlphaVantageProvider')
                    return {'success': False, 'error': f'HTTP {response.status}', 'source': 'alpha_vantage_data'}
        except Exception as e:
            error(f'Alpha Vantage API request error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _check_api_errors(self, data: Dict) -> Optional[Dict[str, Any]]:
        """Check for common API errors"""
        if 'Error Message' in data:
            return {'success': False, 'error': data['Error Message'], 'source': 'alpha_vantage_data'}
        if 'Note' in data:
            return {'success': False, 'error': 'API rate limit exceeded', 'source': 'alpha_vantage_data'}
        if 'Information' in data:
            warn(data['Information'])
            return {'success': False, 'error': data['Information'], 'source': 'alpha_vantage_data'}
        return None

    @monitor_performance
    async def get_stock_data(self, symbol: str, period: str='1d', interval: str='1d') -> Dict[str, Any]:
        """Get stock historical data from Alpha Vantage"""
        try:
            with operation(f'AlphaVantage stock data for {symbol}'):
                function = self.INTERVALS_DICT.get(interval[-1], 'TIME_SERIES_DAILY')
                params = {'function': function, 'symbol': symbol, 'apikey': self.api_key, 'outputsize': 'full', 'datatype': 'json'}
                if 'INTRADAY' in function:
                    av_interval = self.get_interval(interval)
                    params['interval'] = av_interval
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_stock_data(data, symbol, interval)
        except Exception as e:
            error(f'Alpha Vantage stock data error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_exchange_rate(self, data: Dict, from_currency: str, to_currency: str) -> Dict[str, Any]:
        """Transform currency exchange rate response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            exchange_data = data.get('Realtime Currency Exchange Rate', {})
            if not exchange_data:
                return {'success': False, 'error': 'No exchange rate data found', 'source': 'alpha_vantage_data'}
            return {'success': True, 'source': 'alpha_vantage_data', 'from_currency': from_currency, 'to_currency': to_currency, 'data': {'from_currency_code': exchange_data.get('1. From_Currency Code', from_currency), 'from_currency_name': exchange_data.get('2. From_Currency Name', ''), 'to_currency_code': exchange_data.get('3. To_Currency Code', to_currency), 'to_currency_name': exchange_data.get('4. To_Currency Name', ''), 'exchange_rate': float(exchange_data.get('5. Exchange Rate', 0)), 'last_refreshed': exchange_data.get('6. Last Refreshed', ''), 'time_zone': exchange_data.get('7. Time Zone', '')}, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming exchange rate: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Exchange rate transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_fx_intraday(self, from_symbol: str, to_symbol: str, interval: str='5min', outputsize: str='compact') -> Dict[str, Any]:
        """Get intraday FX data"""
        try:
            with operation(f'AlphaVantage FX intraday {from_symbol}/{to_symbol}'):
                params = {'function': 'FX_INTRADAY', 'from_symbol': from_symbol, 'to_symbol': to_symbol, 'interval': interval, 'outputsize': outputsize, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_fx_intraday(data, from_symbol, to_symbol, interval)
        except Exception as e:
            error(f'FX intraday error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_fx_intraday(self, data: Dict, from_symbol: str, to_symbol: str, interval: str) -> Dict[str, Any]:
        """Transform FX intraday response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            time_series_key = f'Time Series FX ({interval})'
            time_series = data.get(time_series_key, {})
            if not time_series:
                return {'success': False, 'error': 'No FX intraday data found', 'source': 'alpha_vantage_data'}
            timestamps = []
            opens = []
            highs = []
            lows = []
            closes = []
            for timestamp in sorted(time_series.keys()):
                tick_data = time_series[timestamp]
                timestamps.append(timestamp)
                opens.append(float(tick_data.get('1. open', 0)))
                highs.append(float(tick_data.get('2. high', 0)))
                lows.append(float(tick_data.get('3. low', 0)))
                closes.append(float(tick_data.get('4. close', 0)))
            return {'success': True, 'source': 'alpha_vantage_data', 'from_symbol': from_symbol, 'to_symbol': to_symbol, 'interval': interval, 'data': {'timestamps': timestamps, 'open': opens, 'high': highs, 'low': lows, 'close': closes}, 'current_rate': closes[-1] if closes else None, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming FX intraday: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'FX intraday transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_fx_weekly(self, from_symbol: str, to_symbol: str) -> Dict[str, Any]:
        """Get weekly FX data"""
        try:
            with operation(f'AlphaVantage FX weekly {from_symbol}/{to_symbol}'):
                params = {'function': 'FX_WEEKLY', 'from_symbol': from_symbol, 'to_symbol': to_symbol, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_fx_weekly_monthly(data, from_symbol, to_symbol, 'weekly')
        except Exception as e:
            error(f'FX weekly error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_fx_monthly(self, from_symbol: str, to_symbol: str) -> Dict[str, Any]:
        """Get monthly FX data"""
        try:
            with operation(f'AlphaVantage FX monthly {from_symbol}/{to_symbol}'):
                params = {'function': 'FX_MONTHLY', 'from_symbol': from_symbol, 'to_symbol': to_symbol, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_fx_weekly_monthly(data, from_symbol, to_symbol, 'monthly')
        except Exception as e:
            error(f'FX monthly error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_fx_weekly_monthly(self, data: Dict, from_symbol: str, to_symbol: str, interval: str) -> Dict[str, Any]:
        """Transform FX weekly/monthly response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            time_series_key = f'Time Series FX ({interval.capitalize()})'
            time_series = data.get(time_series_key, {})
            if not time_series:
                return {'success': False, 'error': f'No FX {interval} data found', 'source': 'alpha_vantage_data'}
            timestamps = []
            opens = []
            highs = []
            lows = []
            closes = []
            for date_str in sorted(time_series.keys()):
                period_data = time_series[date_str]
                timestamps.append(date_str)
                opens.append(float(period_data.get('1. open', 0)))
                highs.append(float(period_data.get('2. high', 0)))
                lows.append(float(period_data.get('3. low', 0)))
                closes.append(float(period_data.get('4. close', 0)))
            return {'success': True, 'source': 'alpha_vantage_data', 'from_symbol': from_symbol, 'to_symbol': to_symbol, 'interval': interval, 'data': {'timestamps': timestamps, 'open': opens, 'high': highs, 'low': lows, 'close': closes}, 'current_rate': closes[-1] if closes else None, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming FX {interval}: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'FX {interval} transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_crypto_intraday(self, symbol: str, market: str='USD', interval: str='5min', outputsize: str='compact') -> Dict[str, Any]:
        """Get intraday crypto data"""
        try:
            with operation(f'AlphaVantage crypto intraday {symbol}/{market}'):
                params = {'function': 'CRYPTO_INTRADAY', 'symbol': symbol, 'market': market, 'interval': interval, 'outputsize': outputsize, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_crypto_intraday(data, symbol, market, interval)
        except Exception as e:
            error(f'Crypto intraday error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_crypto_intraday(self, data: Dict, symbol: str, market: str, interval: str) -> Dict[str, Any]:
        """Transform crypto intraday response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            time_series_key = f'Time Series Crypto ({interval})'
            time_series = data.get(time_series_key, {})
            if not time_series:
                return {'success': False, 'error': 'No crypto intraday data found', 'source': 'alpha_vantage_data'}
            timestamps = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            for timestamp in sorted(time_series.keys()):
                tick_data = time_series[timestamp]
                timestamps.append(timestamp)
                opens.append(float(tick_data.get('1. open', 0)))
                highs.append(float(tick_data.get('2. high', 0)))
                lows.append(float(tick_data.get('3. low', 0)))
                closes.append(float(tick_data.get('4. close', 0)))
                volumes.append(float(tick_data.get('5. volume', 0)))
            return {'success': True, 'source': 'alpha_vantage_data', 'symbol': symbol, 'market': market, 'interval': interval, 'data': {'timestamps': timestamps, 'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volumes}, 'current_price': closes[-1] if closes else None, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming crypto intraday: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Crypto intraday transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_digital_currency_weekly(self, symbol: str, market: str='USD') -> Dict[str, Any]:
        """Get weekly digital currency data"""
        try:
            with operation(f'AlphaVantage digital currency weekly {symbol}/{market}'):
                params = {'function': 'DIGITAL_CURRENCY_WEEKLY', 'symbol': symbol, 'market': market, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_digital_currency_weekly_monthly(data, symbol, market, 'weekly')
        except Exception as e:
            error(f'Digital currency weekly error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_digital_currency_monthly(self, symbol: str, market: str='USD') -> Dict[str, Any]:
        """Get monthly digital currency data"""
        try:
            with operation(f'AlphaVantage digital currency monthly {symbol}/{market}'):
                params = {'function': 'DIGITAL_CURRENCY_MONTHLY', 'symbol': symbol, 'market': market, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_digital_currency_weekly_monthly(data, symbol, market, 'monthly')
        except Exception as e:
            error(f'Digital currency monthly error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_digital_currency_weekly_monthly(self, data: Dict, symbol: str, market: str, interval: str) -> Dict[str, Any]:
        """Transform digital currency weekly/monthly response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            time_series_key = f'Time Series (Digital Currency {interval.capitalize()})'
            time_series = data.get(time_series_key, {})
            if not time_series:
                return {'success': False, 'error': f'No digital currency {interval} data found', 'source': 'alpha_vantage_data'}
            timestamps = []
            opens_market = []
            highs_market = []
            lows_market = []
            closes_market = []
            opens_usd = []
            highs_usd = []
            lows_usd = []
            closes_usd = []
            volumes = []
            for date_str in sorted(time_series.keys()):
                period_data = time_series[date_str]
                timestamps.append(date_str)
                opens_market.append(float(period_data.get(f'1a. open ({market})', 0)))
                highs_market.append(float(period_data.get(f'2a. high ({market})', 0)))
                lows_market.append(float(period_data.get(f'3a. low ({market})', 0)))
                closes_market.append(float(period_data.get(f'4a. close ({market})', 0)))
                opens_usd.append(float(period_data.get('1b. open (USD)', 0)))
                highs_usd.append(float(period_data.get('2b. high (USD)', 0)))
                lows_usd.append(float(period_data.get('3b. low (USD)', 0)))
                closes_usd.append(float(period_data.get('4b. close (USD)', 0)))
                volumes.append(float(period_data.get('5. volume', 0)))
            return {'success': True, 'source': 'alpha_vantage_data', 'symbol': symbol, 'market': market, 'interval': interval, 'data': {'timestamps': timestamps, f'open_{market.lower()}': opens_market, f'high_{market.lower()}': highs_market, f'low_{market.lower()}': lows_market, f'close_{market.lower()}': closes_market, 'open_usd': opens_usd, 'high_usd': highs_usd, 'low_usd': lows_usd, 'close_usd': closes_usd, 'volume': volumes}, 'current_price_usd': closes_usd[-1] if closes_usd else None, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming digital currency {interval}: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Digital currency {interval} transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_commodity_data(self, function: str, interval: str='monthly') -> Dict[str, Any]:
        """Generic method for commodity data"""
        try:
            with operation(f'AlphaVantage commodity {function}'):
                params = {'function': function, 'interval': interval, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_commodity_data(data, function, interval)
        except Exception as e:
            error(f'Commodity {function} error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_commodity_data(self, data: Dict, function: str, interval: str) -> Dict[str, Any]:
        """Transform commodity data response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            data_key = None
            for key in data.keys():
                if 'data' in key.lower() or function.lower() in key.lower():
                    data_key = key
                    break
            if not data_key:
                return {'success': False, 'error': 'No commodity data found', 'source': 'alpha_vantage_data'}
            commodity_data = data[data_key]
            if not commodity_data:
                return {'success': False, 'error': 'Empty commodity data', 'source': 'alpha_vantage_data'}
            timestamps = []
            values = []
            for entry in commodity_data:
                if isinstance(entry, dict):
                    date_key = next((k for k in entry.keys() if 'date' in k.lower()), None)
                    value_key = next((k for k in entry.keys() if 'value' in k.lower() or 'price' in k.lower()), None)
                    if date_key and value_key:
                        timestamps.append(entry[date_key])
                        values.append(float(entry[value_key]) if entry[value_key] != '.' else 0)
            return {'success': True, 'source': 'alpha_vantage_data', 'commodity': function, 'interval': interval, 'data': {'timestamps': timestamps, 'values': values}, 'current_value': values[-1] if values else None, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming commodity data: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Commodity transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    async def get_wti_oil(self, interval: str='monthly') -> Dict[str, Any]:
        """Get WTI crude oil prices"""
        return await self.get_commodity_data('WTI', interval)

    async def get_brent_oil(self, interval: str='monthly') -> Dict[str, Any]:
        """Get Brent crude oil prices"""
        return await self.get_commodity_data('BRENT', interval)

    async def get_natural_gas(self, interval: str='monthly') -> Dict[str, Any]:
        """Get natural gas prices"""
        return await self.get_commodity_data('NATURAL_GAS', interval)

    async def get_copper(self, interval: str='monthly') -> Dict[str, Any]:
        """Get copper prices"""
        return await self.get_commodity_data('COPPER', interval)

    async def get_aluminum(self, interval: str='monthly') -> Dict[str, Any]:
        """Get aluminum prices"""
        return await self.get_commodity_data('ALUMINUM', interval)

    async def get_wheat(self, interval: str='monthly') -> Dict[str, Any]:
        """Get wheat prices"""
        return await self.get_commodity_data('WHEAT', interval)

    async def get_corn(self, interval: str='monthly') -> Dict[str, Any]:
        """Get corn prices"""
        return await self.get_commodity_data('CORN', interval)

    async def get_cotton(self, interval: str='monthly') -> Dict[str, Any]:
        """Get cotton prices"""
        return await self.get_commodity_data('COTTON', interval)

    async def get_sugar(self, interval: str='monthly') -> Dict[str, Any]:
        """Get sugar prices"""
        return await self.get_commodity_data('SUGAR', interval)

    async def get_coffee(self, interval: str='monthly') -> Dict[str, Any]:
        """Get coffee prices"""
        return await self.get_commodity_data('COFFEE', interval)

    async def get_all_commodities(self, interval: str='monthly') -> Dict[str, Any]:
        """Get all commodities index"""
        return await self.get_commodity_data('ALL_COMMODITIES', interval)

    @monitor_performance
    async def get_economic_indicator(self, function: str, interval: str=None, maturity: str=None) -> Dict[str, Any]:
        """Generic method for economic indicators"""
        try:
            with operation(f'AlphaVantage economic indicator {function}'):
                params = {'function': function, 'apikey': self.api_key}
                if interval:
                    params['interval'] = interval
                if maturity:
                    params['maturity'] = maturity
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_economic_data(data, function, interval)
        except Exception as e:
            error(f'Economic indicator {function} error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_economic_data(self, data: Dict, function: str, interval: str) -> Dict[str, Any]:
        """Transform economic indicator response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            data_key = None
            for key in data.keys():
                if 'data' in key.lower() or function.lower() in key.lower():
                    data_key = key
                    break
            if not data_key:
                return {'success': False, 'error': 'No economic data found', 'source': 'alpha_vantage_data'}
            economic_data = data[data_key]
            if not economic_data:
                return {'success': False, 'error': 'Empty economic data', 'source': 'alpha_vantage_data'}
            timestamps = []
            values = []
            for entry in economic_data:
                if isinstance(entry, dict):
                    date_key = next((k for k in entry.keys() if 'date' in k.lower()), None)
                    value_key = next((k for k in entry.keys() if 'value' in k.lower()), None)
                    if date_key and value_key:
                        timestamps.append(entry[date_key])
                        values.append(float(entry[value_key]) if entry[value_key] != '.' else 0)
            return {'success': True, 'source': 'alpha_vantage_data', 'indicator': function, 'interval': interval, 'data': {'timestamps': timestamps, 'values': values}, 'latest_value': values[-1] if values else None, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming economic data: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Economic data transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    async def get_real_gdp(self, interval: str='annual') -> Dict[str, Any]:
        """Get Real GDP data"""
        return await self.get_economic_indicator('REAL_GDP', interval)

    async def get_real_gdp_per_capita(self) -> Dict[str, Any]:
        """Get Real GDP per capita data"""
        return await self.get_economic_indicator('REAL_GDP_PER_CAPITA')

    async def get_treasury_yield(self, interval: str='monthly', maturity: str='10year') -> Dict[str, Any]:
        """Get Treasury yield data"""
        return await self.get_economic_indicator('TREASURY_YIELD', interval, maturity)

    async def get_federal_funds_rate(self, interval: str='monthly') -> Dict[str, Any]:
        """Get Federal funds rate data"""
        return await self.get_economic_indicator('FEDERAL_FUNDS_RATE', interval)

    async def get_cpi(self, interval: str='monthly') -> Dict[str, Any]:
        """Get Consumer Price Index data"""
        return await self.get_economic_indicator('CPI', interval)

    async def get_inflation(self) -> Dict[str, Any]:
        """Get inflation data"""
        return await self.get_economic_indicator('INFLATION')

    async def get_retail_sales(self) -> Dict[str, Any]:
        """Get retail sales data"""
        return await self.get_economic_indicator('RETAIL_SALES')

    async def get_durables(self) -> Dict[str, Any]:
        """Get durable goods data"""
        return await self.get_economic_indicator('DURABLES')

    async def get_unemployment(self) -> Dict[str, Any]:
        """Get unemployment rate data"""
        return await self.get_economic_indicator('UNEMPLOYMENT')

    async def get_nonfarm_payroll(self) -> Dict[str, Any]:
        """Get nonfarm payroll data"""
        return await self.get_economic_indicator('NONFARM_PAYROLL')

    @monitor_performance
    async def get_technical_indicator(self, function: str, symbol: str, interval: str, time_period: int=None, series_type: str='close', **kwargs) -> Dict[str, Any]:
        """Generic method for technical indicators"""
        try:
            with operation(f'AlphaVantage {function} for {symbol}'):
                params = {'function': function, 'symbol': symbol, 'interval': interval, 'series_type': series_type, 'apikey': self.api_key}
                if time_period:
                    params['time_period'] = time_period
                params.update(kwargs)
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_technical_indicator(data, function, symbol, interval)
        except Exception as e:
            error(f'Technical indicator {function} error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_technical_indicator(self, data: Dict, function: str, symbol: str, interval: str) -> Dict[str, Any]:
        """Transform technical indicator response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            tech_key = None
            for key in data.keys():
                if 'Technical Analysis' in key or function in key:
                    tech_key = key
                    break
            if not tech_key:
                return {'success': False, 'error': 'No technical indicator data found', 'source': 'alpha_vantage_data'}
            tech_data = data[tech_key]
            if not tech_data:
                return {'success': False, 'error': 'Empty technical indicator data', 'source': 'alpha_vantage_data'}
            timestamps = []
            values = {}
            for timestamp in sorted(tech_data.keys()):
                timestamps.append(timestamp)
                for key, value in tech_data[timestamp].items():
                    indicator_name = key.split('. ')[-1] if '. ' in key else key
                    if indicator_name not in values:
                        values[indicator_name] = []
                    values[indicator_name].append(float(value))
            return {'success': True, 'source': 'alpha_vantage_data', 'function': function, 'symbol': symbol, 'interval': interval, 'data': {'timestamps': timestamps, **values}, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming technical indicator: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Technical indicator transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    async def get_sma(self, symbol: str, interval: str, time_period: int, series_type: str='close') -> Dict[str, Any]:
        """Get Simple Moving Average"""
        return await self.get_technical_indicator('SMA', symbol, interval, time_period, series_type)

    async def get_ema(self, symbol: str, interval: str, time_period: int, series_type: str='close') -> Dict[str, Any]:
        """Get Exponential Moving Average"""
        return await self.get_technical_indicator('EMA', symbol, interval, time_period, series_type)

    async def get_wma(self, symbol: str, interval: str, time_period: int, series_type: str='close') -> Dict[str, Any]:
        """Get Weighted Moving Average"""
        return await self.get_technical_indicator('WMA', symbol, interval, time_period, series_type)

    async def get_dema(self, symbol: str, interval: str, time_period: int, series_type: str='close') -> Dict[str, Any]:
        """Get Double Exponential Moving Average"""
        return await self.get_technical_indicator('DEMA', symbol, interval, time_period, series_type)

    async def get_tema(self, symbol: str, interval: str, time_period: int, series_type: str='close') -> Dict[str, Any]:
        """Get Triple Exponential Moving Average"""
        return await self.get_technical_indicator('TEMA', symbol, interval, time_period, series_type)

    async def get_trima(self, symbol: str, interval: str, time_period: int, series_type: str='close') -> Dict[str, Any]:
        """Get Triangular Moving Average"""
        return await self.get_technical_indicator('TRIMA', symbol, interval, time_period, series_type)

    async def get_kama(self, symbol: str, interval: str, time_period: int, series_type: str='close') -> Dict[str, Any]:
        """Get Kaufman Adaptive Moving Average"""
        return await self.get_technical_indicator('KAMA', symbol, interval, time_period, series_type)

    async def get_mama(self, symbol: str, interval: str, series_type: str='close', fastlimit: float=0.01, slowlimit: float=0.01) -> Dict[str, Any]:
        """Get MESA Adaptive Moving Average"""
        return await self.get_technical_indicator('MAMA', symbol, interval, None, series_type, fastlimit=fastlimit, slowlimit=slowlimit)

    async def get_vwap(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Get Volume Weighted Average Price"""
        return await self.get_technical_indicator('VWAP', symbol, interval)

    async def get_t3(self, symbol: str, interval: str, time_period: int, series_type: str='close') -> Dict[str, Any]:
        """Get T3 Moving Average"""
        return await self.get_technical_indicator('T3', symbol, interval, time_period, series_type)

    async def get_macd(self, symbol: str, interval: str, series_type: str='close', fastperiod: int=12, slowperiod: int=26, signalperiod: int=9) -> Dict[str, Any]:
        """Get Moving Average Convergence Divergence"""
        return await self.get_technical_indicator('MACD', symbol, interval, None, series_type, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)

    async def get_macdext(self, symbol: str, interval: str, series_type: str='close', fastperiod: int=12, slowperiod: int=26, signalperiod: int=9, fastmatype: int=0, slowmatype: int=0, signalmatype: int=0) -> Dict[str, Any]:
        """Get MACD with controllable MA type"""
        return await self.get_technical_indicator('MACDEXT', symbol, interval, None, series_type, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod, fastmatype=fastmatype, slowmatype=slowmatype, signalmatype=signalmatype)

    async def get_stoch(self, symbol: str, interval: str, fastkperiod: int=5, slowkperiod: int=3, slowdperiod: int=3, slowkmatype: int=0, slowdmatype: int=0) -> Dict[str, Any]:
        """Get Stochastic Oscillator"""
        return await self.get_technical_indicator('STOCH', symbol, interval, None, None, fastkperiod=fastkperiod, slowkperiod=slowkperiod, slowdperiod=slowdperiod, slowkmatype=slowkmatype, slowdmatype=slowdmatype)

    async def get_stochf(self, symbol: str, interval: str, fastkperiod: int=5, fastdperiod: int=3, fastdmatype: int=0) -> Dict[str, Any]:
        """Get Stochastic Fast"""
        return await self.get_technical_indicator('STOCHF', symbol, interval, None, None, fastkperiod=fastkperiod, fastdperiod=fastdperiod, fastdmatype=fastdmatype)

    async def get_rsi(self, symbol: str, interval: str, time_period: int, series_type: str='close') -> Dict[str, Any]:
        """Get Relative Strength Index"""
        return await self.get_technical_indicator('RSI', symbol, interval, time_period, series_type)

    async def get_stochrsi(self, symbol: str, interval: str, time_period: int, series_type: str='close', fastkperiod: int=5, fastdperiod: int=3, fastdmatype: int=0) -> Dict[str, Any]:
        """Get Stochastic RSI"""
        return await self.get_technical_indicator('STOCHRSI', symbol, interval, time_period, series_type, fastkperiod=fastkperiod, fastdperiod=fastdperiod, fastdmatype=fastdmatype)

    async def get_willr(self, symbol: str, interval: str, time_period: int) -> Dict[str, Any]:
        """Get Williams %R"""
        return await self.get_technical_indicator('WILLR', symbol, interval, time_period)

    async def get_adx(self, symbol: str, interval: str, time_period: int) -> Dict[str, Any]:
        """Get Average Directional Movement Index"""
        return await self.get_technical_indicator('ADX', symbol, interval, time_period)

    async def get_adxr(self, symbol: str, interval: str, time_period: int) -> Dict[str, Any]:
        """Get Average Directional Movement Index Rating"""
        return await self.get_technical_indicator('ADXR', symbol, interval, time_period)

    async def get_apo(self, symbol: str, interval: str, series_type: str='close', fastperiod: int=12, slowperiod: int=26, matype: int=0) -> Dict[str, Any]:
        """Get Absolute Price Oscillator"""
        return await self.get_technical_indicator('APO', symbol, interval, None, series_type, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)

    async def get_ppo(self, symbol: str, interval: str, series_type: str='close', fastperiod: int=12, slowperiod: int=26, matype: int=0) -> Dict[str, Any]:
        """Get Percentage Price Oscillator"""
        return await self.get_technical_indicator('PPO', symbol, interval, None, series_type, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)

    async def get_mom(self, symbol: str, interval: str, time_period: int, series_type: str='close') -> Dict[str, Any]:
        """Get Momentum"""
        return await self.get_technical_indicator('MOM', symbol, interval, time_period, series_type)

    async def get_bop(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Get Balance of Power"""
        return await self.get_technical_indicator('BOP', symbol, interval)

    async def get_cci(self, symbol: str, interval: str, time_period: int) -> Dict[str, Any]:
        """Get Commodity Channel Index"""
        return await self.get_technical_indicator('CCI', symbol, interval, time_period)

    async def get_cmo(self, symbol: str, interval: str, time_period: int, series_type: str='close') -> Dict[str, Any]:
        """Get Chande Momentum Oscillator"""
        return await self.get_technical_indicator('CMO', symbol, interval, time_period, series_type)

    async def get_roc(self, symbol: str, interval: str, time_period: int, series_type: str='close') -> Dict[str, Any]:
        """Get Rate of Change"""
        return await self.get_technical_indicator('ROC', symbol, interval, time_period, series_type)

    async def get_rocr(self, symbol: str, interval: str, time_period: int, series_type: str='close') -> Dict[str, Any]:
        """Get Rate of Change Ratio"""
        return await self.get_technical_indicator('ROCR', symbol, interval, time_period, series_type)

    async def get_aroon(self, symbol: str, interval: str, time_period: int) -> Dict[str, Any]:
        """Get Aroon"""
        return await self.get_technical_indicator('AROON', symbol, interval, time_period)

    async def get_aroonosc(self, symbol: str, interval: str, time_period: int) -> Dict[str, Any]:
        """Get Aroon Oscillator"""
        return await self.get_technical_indicator('AROONOSC', symbol, interval, time_period)

    async def get_mfi(self, symbol: str, interval: str, time_period: int) -> Dict[str, Any]:
        """Get Money Flow Index"""
        return await self.get_technical_indicator('MFI', symbol, interval, time_period)

    async def get_trix(self, symbol: str, interval: str, time_period: int, series_type: str='close') -> Dict[str, Any]:
        """Get TRIX"""
        return await self.get_technical_indicator('TRIX', symbol, interval, time_period, series_type)

    async def get_ultosc(self, symbol: str, interval: str, timeperiod1: int=7, timeperiod2: int=14, timeperiod3: int=28) -> Dict[str, Any]:
        """Get Ultimate Oscillator"""
        return await self.get_technical_indicator('ULTOSC', symbol, interval, None, None, timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)

    async def get_dx(self, symbol: str, interval: str, time_period: int) -> Dict[str, Any]:
        """Get Directional Movement Index"""
        return await self.get_technical_indicator('DX', symbol, interval, time_period)

    async def get_minus_di(self, symbol: str, interval: str, time_period: int) -> Dict[str, Any]:
        """Get Minus Directional Indicator"""
        return await self.get_technical_indicator('MINUS_DI', symbol, interval, time_period)

    async def get_plus_di(self, symbol: str, interval: str, time_period: int) -> Dict[str, Any]:
        """Get Plus Directional Indicator"""
        return await self.get_technical_indicator('PLUS_DI', symbol, interval, time_period)

    async def get_minus_dm(self, symbol: str, interval: str, time_period: int) -> Dict[str, Any]:
        """Get Minus Directional Movement"""
        return await self.get_technical_indicator('MINUS_DM', symbol, interval, time_period)

    async def get_plus_dm(self, symbol: str, interval: str, time_period: int) -> Dict[str, Any]:
        """Get Plus Directional Movement"""
        return await self.get_technical_indicator('PLUS_DM', symbol, interval, time_period)

    async def get_bbands(self, symbol: str, interval: str, time_period: int, series_type: str='close', nbdevup: int=2, nbdevdn: int=2, matype: int=0) -> Dict[str, Any]:
        """Get Bollinger Bands"""
        return await self.get_technical_indicator('BBANDS', symbol, interval, time_period, series_type, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)

    async def get_midpoint(self, symbol: str, interval: str, time_period: int, series_type: str='close') -> Dict[str, Any]:
        """Get MidPoint"""
        return await self.get_technical_indicator('MIDPOINT', symbol, interval, time_period, series_type)

    async def get_midprice(self, symbol: str, interval: str, time_period: int) -> Dict[str, Any]:
        """Get MidPrice"""
        return await self.get_technical_indicator('MIDPRICE', symbol, interval, time_period)

    async def get_sar(self, symbol: str, interval: str, acceleration: float=0.01, maximum: float=0.2) -> Dict[str, Any]:
        """Get Parabolic SAR"""
        return await self.get_technical_indicator('SAR', symbol, interval, None, None, acceleration=acceleration, maximum=maximum)

    async def get_trange(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Get True Range"""
        return await self.get_technical_indicator('TRANGE', symbol, interval)

    async def get_atr(self, symbol: str, interval: str, time_period: int) -> Dict[str, Any]:
        """Get Average True Range"""
        return await self.get_technical_indicator('ATR', symbol, interval, time_period)

    async def get_natr(self, symbol: str, interval: str, time_period: int) -> Dict[str, Any]:
        """Get Normalized Average True Range"""
        return await self.get_technical_indicator('NATR', symbol, interval, time_period)

    async def get_ad(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Get Chaikin A/D Line"""
        return await self.get_technical_indicator('AD', symbol, interval)

    async def get_adosc(self, symbol: str, interval: str, fastperiod: int=3, slowperiod: int=10) -> Dict[str, Any]:
        """Get Chaikin A/D Oscillator"""
        return await self.get_technical_indicator('ADOSC', symbol, interval, None, None, fastperiod=fastperiod, slowperiod=slowperiod)

    async def get_obv(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Get On Balance Volume"""
        return await self.get_technical_indicator('OBV', symbol, interval)

    async def get_ht_trendline(self, symbol: str, interval: str, series_type: str='close') -> Dict[str, Any]:
        """Get Hilbert Transform - Instantaneous Trendline"""
        return await self.get_technical_indicator('HT_TRENDLINE', symbol, interval, None, series_type)

    async def get_ht_sine(self, symbol: str, interval: str, series_type: str='close') -> Dict[str, Any]:
        """Get Hilbert Transform - Sine Wave"""
        return await self.get_technical_indicator('HT_SINE', symbol, interval, None, series_type)

    async def get_ht_trendmode(self, symbol: str, interval: str, series_type: str='close') -> Dict[str, Any]:
        """Get Hilbert Transform - Trend vs Cycle Mode"""
        return await self.get_technical_indicator('HT_TRENDMODE', symbol, interval, None, series_type)

    async def get_ht_dcperiod(self, symbol: str, interval: str, series_type: str='close') -> Dict[str, Any]:
        """Get Hilbert Transform - Dominant Cycle Period"""
        return await self.get_technical_indicator('HT_DCPERIOD', symbol, interval, None, series_type)

    async def get_ht_dcphase(self, symbol: str, interval: str, series_type: str='close') -> Dict[str, Any]:
        """Get Hilbert Transform - Dominant Cycle Phase"""
        return await self.get_technical_indicator('HT_DCPHASE', symbol, interval, None, series_type)

    async def get_ht_phasor(self, symbol: str, interval: str, series_type: str='close') -> Dict[str, Any]:
        """Get Hilbert Transform - Phasor Components"""
        return await self.get_technical_indicator('HT_PHASOR', symbol, interval, None, series_type)

    async def close(self):
        """Close the aiohttp session"""
        if self._session and (not self._session.closed):
            await self._session.close()
            debug('Alpha Vantage session closed', module='AlphaVantageProvider')

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, '_session') and self._session and (not self._session.closed):
            try:
                asyncio.create_task(self.close())
            except Exception:
                pass

    def _transform_stock_data(self, data: Dict, symbol: str, interval: str) -> Dict[str, Any]:
        """Transform Alpha Vantage response to standard format"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            if not time_series_key:
                return {'success': False, 'error': 'No time series data found', 'source': 'alpha_vantage_data'}
            time_series = data[time_series_key]
            if not time_series:
                return {'success': False, 'error': 'Empty time series data', 'source': 'alpha_vantage_data'}
            timestamps = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            for date_str in sorted(time_series.keys()):
                day_data = time_series[date_str]
                timestamps.append(date_str)
                opens.append(float(day_data.get('1. open', 0)))
                highs.append(float(day_data.get('2. high', 0)))
                lows.append(float(day_data.get('3. low', 0)))
                closes.append(float(day_data.get('4. close', 0)))
                volumes.append(int(float(day_data.get('5. volume', 0))))
            debug(f'Transformed {len(timestamps)} data points for {symbol}', module='AlphaVantageProvider')
            return {'success': True, 'source': 'alpha_vantage_data', 'symbol': symbol, 'data': {'timestamps': timestamps, 'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volumes}, 'current_price': closes[-1] if closes else None, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming Alpha Vantage data: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Data transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_forex_data(self, pair: str, period: str='1d') -> Dict[str, Any]:
        """Get forex data from Alpha Vantage"""
        try:
            with operation(f'AlphaVantage forex data for {pair}'):
                if len(pair) == 6:
                    from_symbol = pair[:3]
                    to_symbol = pair[3:]
                else:
                    parts = pair.replace('/', '').replace('-', '').replace('_', '')
                    if len(parts) == 6:
                        from_symbol = parts[:3]
                        to_symbol = parts[3:]
                    else:
                        return {'success': False, 'error': f'Invalid forex pair format: {pair}', 'source': 'alpha_vantage_data'}
                params = {'function': 'FX_DAILY', 'from_symbol': from_symbol, 'to_symbol': to_symbol, 'apikey': self.api_key, 'outputsize': 'full'}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_forex_data(data, pair)
        except Exception as e:
            error(f'Alpha Vantage forex data error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_forex_data(self, data: Dict, pair: str) -> Dict[str, Any]:
        """Transform forex data response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            time_series = data.get('Time Series FX (Daily)', {})
            if not time_series:
                return {'success': False, 'error': 'No forex data found', 'source': 'alpha_vantage_data'}
            timestamps = []
            rates = []
            for date_str in sorted(time_series.keys()):
                day_data = time_series[date_str]
                timestamps.append(date_str)
                rates.append(float(day_data.get('4. close', 0)))
            return {'success': True, 'source': 'alpha_vantage_data', 'pair': pair, 'data': {'timestamps': timestamps, 'rates': rates}, 'current_rate': rates[-1] if rates else None, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming forex data: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Forex data transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_crypto_data(self, symbol: str, period: str='1d') -> Dict[str, Any]:
        """Get crypto data from Alpha Vantage"""
        try:
            with operation(f'AlphaVantage crypto data for {symbol}'):
                params = {'function': 'DIGITAL_CURRENCY_DAILY', 'symbol': symbol, 'market': 'USD', 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_crypto_data(data, symbol)
        except Exception as e:
            error(f'Alpha Vantage crypto data error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_crypto_data(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Transform crypto data response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            time_series = data.get('Time Series (Digital Currency Daily)', {})
            if not time_series:
                return {'success': False, 'error': 'No crypto data found', 'source': 'alpha_vantage_data'}
            timestamps = []
            prices = []
            for date_str in sorted(time_series.keys()):
                day_data = time_series[date_str]
                timestamps.append(date_str)
                prices.append(float(day_data.get('4a. close (USD)', 0)))
            return {'success': True, 'source': 'alpha_vantage_data', 'symbol': symbol, 'data': {'timestamps': timestamps, 'prices': prices}, 'current_price': prices[-1] if prices else None, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming crypto data: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Crypto data transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def verify_api_key(self) -> Dict[str, Any]:
        """Verify Alpha Vantage API key"""
        try:
            with operation('Verify Alpha Vantage API key'):
                params = {'function': 'TIME_SERIES_DAILY', 'symbol': 'AAPL', 'apikey': self.api_key, 'outputsize': 'compact'}
                session = await self._get_session()
                async with session.get(self.base_url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'Error Message' in data:
                            return {'valid': False, 'error': data['Error Message']}
                        elif 'Note' in data:
                            return {'valid': False, 'error': 'API rate limit exceeded'}
                        elif 'Information' in data:
                            return {'valid': False, 'error': data['Information']}
                        elif 'Time Series (Daily)' in data:
                            info('Alpha Vantage API key verified successfully', module='AlphaVantageProvider')
                            return {'valid': True, 'message': 'API key verified successfully'}
                        else:
                            return {'valid': False, 'error': 'Unexpected API response'}
                    else:
                        return {'valid': False, 'error': f'HTTP {response.status}'}
        except asyncio.TimeoutError:
            return {'valid': False, 'error': 'API request timeout'}
        except Exception as e:
            error(f'API key verification error: {str(e)}', module='AlphaVantageProvider')
            return {'valid': False, 'error': str(e)}

    @monitor_performance
    async def get_daily_adjusted(self, symbol: str, outputsize: str='compact') -> Dict[str, Any]:
        """Get daily adjusted stock data"""
        try:
            with operation(f'AlphaVantage daily adjusted data for {symbol}'):
                params = {'function': 'TIME_SERIES_DAILY_ADJUSTED', 'symbol': symbol, 'outputsize': outputsize, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_adjusted_data(data, symbol, 'daily')
        except Exception as e:
            error(f'Daily adjusted data error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_weekly_adjusted(self, symbol: str) -> Dict[str, Any]:
        """Get weekly adjusted stock data"""
        try:
            with operation(f'AlphaVantage weekly adjusted data for {symbol}'):
                params = {'function': 'TIME_SERIES_WEEKLY_ADJUSTED', 'symbol': symbol, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_adjusted_data(data, symbol, 'weekly')
        except Exception as e:
            error(f'Weekly adjusted data error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_monthly_adjusted(self, symbol: str) -> Dict[str, Any]:
        """Get monthly adjusted stock data"""
        try:
            with operation(f'AlphaVantage monthly adjusted data for {symbol}'):
                params = {'function': 'TIME_SERIES_MONTHLY_ADJUSTED', 'symbol': symbol, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_adjusted_data(data, symbol, 'monthly')
        except Exception as e:
            error(f'Monthly adjusted data error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_adjusted_data(self, data: Dict, symbol: str, interval: str) -> Dict[str, Any]:
        """Transform adjusted time series data"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            if not time_series_key:
                return {'success': False, 'error': 'No time series data found', 'source': 'alpha_vantage_data'}
            time_series = data[time_series_key]
            if not time_series:
                return {'success': False, 'error': 'Empty time series data', 'source': 'alpha_vantage_data'}
            timestamps = []
            opens = []
            highs = []
            lows = []
            closes = []
            adjusted_closes = []
            volumes = []
            dividends = []
            for date_str in sorted(time_series.keys()):
                day_data = time_series[date_str]
                timestamps.append(date_str)
                opens.append(float(day_data.get('1. open', 0)))
                highs.append(float(day_data.get('2. high', 0)))
                lows.append(float(day_data.get('3. low', 0)))
                closes.append(float(day_data.get('4. close', 0)))
                adjusted_closes.append(float(day_data.get('5. adjusted close', 0)))
                volumes.append(int(float(day_data.get('6. volume', 0))))
                dividends.append(float(day_data.get('7. dividend amount', 0)))
            return {'success': True, 'source': 'alpha_vantage_data', 'symbol': symbol, 'interval': interval, 'data': {'timestamps': timestamps, 'open': opens, 'high': highs, 'low': lows, 'close': closes, 'adjusted_close': adjusted_closes, 'volume': volumes, 'dividend_amount': dividends}, 'current_price': closes[-1] if closes else None, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming adjusted data: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Data transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_global_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time global quote"""
        try:
            with operation(f'AlphaVantage global quote for {symbol}'):
                params = {'function': 'GLOBAL_QUOTE', 'symbol': symbol, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_global_quote(data, symbol)
        except Exception as e:
            error(f'Global quote error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_global_quote(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Transform global quote response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            quote_data = data.get('Global Quote', {})
            if not quote_data:
                return {'success': False, 'error': 'No quote data found', 'source': 'alpha_vantage_data'}
            return {'success': True, 'source': 'alpha_vantage_data', 'symbol': symbol, 'data': {'symbol': quote_data.get('01. symbol', symbol), 'open': float(quote_data.get('02. open', 0)), 'high': float(quote_data.get('03. high', 0)), 'low': float(quote_data.get('04. low', 0)), 'price': float(quote_data.get('05. price', 0)), 'volume': int(quote_data.get('06. volume', 0)), 'latest_trading_day': quote_data.get('07. latest trading day', ''), 'previous_close': float(quote_data.get('08. previous close', 0)), 'change': float(quote_data.get('09. change', 0)), 'change_percent': quote_data.get('10. change percent', '0%')}, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming global quote: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Quote transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_realtime_bulk_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time bulk quotes (up to 100 symbols)"""
        try:
            if len(symbols) > 100:
                symbols = symbols[:100]
                warning('Truncated symbols list to 100 items', module='AlphaVantageProvider')
            symbol_string = ','.join(symbols)
            with operation(f'AlphaVantage bulk quotes for {len(symbols)} symbols'):
                params = {'function': 'REALTIME_BULK_QUOTES', 'symbol': symbol_string, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_bulk_quotes(data, symbols)
        except Exception as e:
            error(f'Bulk quotes error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_bulk_quotes(self, data: Dict, symbols: List[str]) -> Dict[str, Any]:
        """Transform bulk quotes response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            quotes = data.get('quotes', [])
            if not quotes:
                return {'success': False, 'error': 'No quotes data found', 'source': 'alpha_vantage_data'}
            transformed_quotes = []
            for quote in quotes:
                transformed_quotes.append({'symbol': quote.get('symbol', ''), 'open': float(quote.get('open', 0)), 'high': float(quote.get('high', 0)), 'low': float(quote.get('low', 0)), 'price': float(quote.get('price', 0)), 'volume': int(quote.get('volume', 0)), 'change': float(quote.get('change', 0)), 'change_percent': quote.get('change_percent', '0%')})
            return {'success': True, 'source': 'alpha_vantage_data', 'symbols': symbols, 'data': transformed_quotes, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming bulk quotes: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Bulk quotes transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def search_symbols(self, keywords: str) -> Dict[str, Any]:
        """Search for symbols by keywords"""
        try:
            with operation(f'AlphaVantage symbol search for: {keywords}'):
                params = {'function': 'SYMBOL_SEARCH', 'keywords': keywords, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_search_results(data, keywords)
        except Exception as e:
            error(f'Symbol search error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_search_results(self, data: Dict, keywords: str) -> Dict[str, Any]:
        """Transform symbol search results"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            matches = data.get('bestMatches', [])
            results = []
            for match in matches:
                results.append({'symbol': match.get('1. symbol', ''), 'name': match.get('2. name', ''), 'type': match.get('3. type', ''), 'region': match.get('4. region', ''), 'market_open': match.get('5. marketOpen', ''), 'market_close': match.get('6. marketClose', ''), 'timezone': match.get('7. timezone', ''), 'currency': match.get('8. currency', ''), 'match_score': float(match.get('9. matchScore', 0))})
            return {'success': True, 'source': 'alpha_vantage_data', 'keywords': keywords, 'data': results, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming search results: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Search transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_market_status(self) -> Dict[str, Any]:
        """Get global market status"""
        try:
            with operation('AlphaVantage market status'):
                params = {'function': 'MARKET_STATUS', 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_market_status(data)
        except Exception as e:
            error(f'Market status error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_market_status(self, data: Dict) -> Dict[str, Any]:
        """Transform market status response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            markets = data.get('markets', [])
            return {'success': True, 'source': 'alpha_vantage_data', 'data': markets, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming market status: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Market status transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_realtime_options(self, symbol: str, require_greeks: bool=False, contract: str=None) -> Dict[str, Any]:
        """Get real-time options data"""
        try:
            with operation(f'AlphaVantage realtime options for {symbol}'):
                params = {'function': 'REALTIME_OPTIONS', 'symbol': symbol, 'require_greeks': str(require_greeks).lower(), 'apikey': self.api_key}
                if contract:
                    params['contract'] = contract
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_options_data(data, symbol)
        except Exception as e:
            error(f'Realtime options error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_historical_options(self, symbol: str, date: str=None) -> Dict[str, Any]:
        """Get historical options data"""
        try:
            with operation(f'AlphaVantage historical options for {symbol}'):
                params = {'function': 'HISTORICAL_OPTIONS', 'symbol': symbol, 'apikey': self.api_key}
                if date:
                    params['date'] = date
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_options_data(data, symbol)
        except Exception as e:
            error(f'Historical options error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_options_data(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Transform options data response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            options_data = data.get('data', [])
            return {'success': True, 'source': 'alpha_vantage_data', 'symbol': symbol, 'data': options_data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming options data: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Options transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_news_sentiment(self, tickers: str=None, topics: str=None, time_from: str=None, time_to: str=None, sort: str='LATEST', limit: int=50) -> Dict[str, Any]:
        """Get market news and sentiment data"""
        try:
            with operation(f'AlphaVantage news sentiment'):
                params = {'function': 'NEWS_SENTIMENT', 'sort': sort, 'limit': limit, 'apikey': self.api_key}
                if tickers:
                    params['tickers'] = tickers
                if topics:
                    params['topics'] = topics
                if time_from:
                    params['time_from'] = time_from
                if time_to:
                    params['time_to'] = time_to
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_news_sentiment(data)
        except Exception as e:
            error(f'News sentiment error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_news_sentiment(self, data: Dict) -> Dict[str, Any]:
        """Transform news sentiment response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            return {'success': True, 'source': 'alpha_vantage_data', 'data': data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming news sentiment: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'News sentiment transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_earnings_call_transcript(self, symbol: str, quarter: str) -> Dict[str, Any]:
        """Get earnings call transcript"""
        try:
            with operation(f'AlphaVantage earnings call transcript for {symbol}'):
                params = {'function': 'EARNINGS_CALL_TRANSCRIPT', 'symbol': symbol, 'quarter': quarter, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_earnings_transcript(data, symbol, quarter)
        except Exception as e:
            error(f'Earnings transcript error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_earnings_transcript(self, data: Dict, symbol: str, quarter: str) -> Dict[str, Any]:
        """Transform earnings transcript response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            return {'success': True, 'source': 'alpha_vantage_data', 'symbol': symbol, 'quarter': quarter, 'data': data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming earnings transcript: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Earnings transcript transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_top_gainers_losers(self) -> Dict[str, Any]:
        """Get top gainers, losers, and most active stocks"""
        try:
            with operation('AlphaVantage top gainers losers'):
                params = {'function': 'TOP_GAINERS_LOSERS', 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_top_gainers_losers(data)
        except Exception as e:
            error(f'Top gainers losers error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_top_gainers_losers(self, data: Dict) -> Dict[str, Any]:
        """Transform top gainers/losers response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            return {'success': True, 'source': 'alpha_vantage_data', 'data': {'metadata': data.get('metadata', ''), 'last_updated': data.get('last_updated', ''), 'top_gainers': data.get('top_gainers', []), 'top_losers': data.get('top_losers', []), 'most_actively_traded': data.get('most_actively_traded', [])}, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming top gainers losers: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Top gainers losers transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_insider_transactions(self, symbol: str) -> Dict[str, Any]:
        """Get insider transactions"""
        try:
            with operation(f'AlphaVantage insider transactions for {symbol}'):
                params = {'function': 'INSIDER_TRANSACTIONS', 'symbol': symbol, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_insider_transactions(data, symbol)
        except Exception as e:
            error(f'Insider transactions error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_insider_transactions(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Transform insider transactions response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            return {'success': True, 'source': 'alpha_vantage_data', 'symbol': symbol, 'data': data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming insider transactions: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Insider transactions transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_analytics_fixed_window(self, symbols: str, range_param: str, interval: str, ohlc: str='close', calculations: str=None) -> Dict[str, Any]:
        """Get advanced analytics for fixed window"""
        try:
            with operation(f'AlphaVantage analytics fixed window'):
                params = {'function': 'ANALYTICS_FIXED_WINDOW', 'SYMBOLS': symbols, 'RANGE': range_param, 'INTERVAL': interval, 'OHLC': ohlc, 'apikey': self.api_key}
                if calculations:
                    params['CALCULATIONS'] = calculations
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_analytics_data(data, 'fixed_window')
        except Exception as e:
            error(f'Analytics fixed window error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_analytics_sliding_window(self, symbols: str, range_param: str, interval: str, window_size: int, ohlc: str='close', calculations: str=None) -> Dict[str, Any]:
        """Get advanced analytics for sliding window"""
        try:
            with operation(f'AlphaVantage analytics sliding window'):
                params = {'function': 'ANALYTICS_SLIDING_WINDOW', 'SYMBOLS': symbols, 'RANGE': range_param, 'INTERVAL': interval, 'WINDOW_SIZE': window_size, 'OHLC': ohlc, 'apikey': self.api_key}
                if calculations:
                    params['CALCULATIONS'] = calculations
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_analytics_data(data, 'sliding_window')
        except Exception as e:
            error(f'Analytics sliding window error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_analytics_data(self, data: Dict, window_type: str) -> Dict[str, Any]:
        """Transform analytics data response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            return {'success': True, 'source': 'alpha_vantage_data', 'window_type': window_type, 'data': data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming analytics data: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Analytics transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """Get company overview and fundamentals"""
        try:
            with operation(f'AlphaVantage company overview for {symbol}'):
                params = {'function': 'OVERVIEW', 'symbol': symbol, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_company_overview(data, symbol)
        except Exception as e:
            error(f'Company overview error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_company_overview(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Transform company overview response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            return {'success': True, 'source': 'alpha_vantage_data', 'symbol': symbol, 'data': data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming company overview: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Company overview transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_etf_profile(self, symbol: str) -> Dict[str, Any]:
        """Get ETF profile and holdings"""
        try:
            with operation(f'AlphaVantage ETF profile for {symbol}'):
                params = {'function': 'ETF_PROFILE', 'symbol': symbol, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_etf_profile(data, symbol)
        except Exception as e:
            error(f'ETF profile error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_etf_profile(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Transform ETF profile response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            return {'success': True, 'source': 'alpha_vantage_data', 'symbol': symbol, 'data': data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming ETF profile: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'ETF profile transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_dividends(self, symbol: str) -> Dict[str, Any]:
        """Get dividend history"""
        try:
            with operation(f'AlphaVantage dividends for {symbol}'):
                params = {'function': 'DIVIDENDS', 'symbol': symbol, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_dividends(data, symbol)
        except Exception as e:
            error(f'Dividends error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_dividends(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Transform dividends response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            return {'success': True, 'source': 'alpha_vantage_data', 'symbol': symbol, 'data': data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming dividends: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Dividends transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_splits(self, symbol: str) -> Dict[str, Any]:
        """Get stock splits history"""
        try:
            with operation(f'AlphaVantage splits for {symbol}'):
                params = {'function': 'SPLITS', 'symbol': symbol, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_splits(data, symbol)
        except Exception as e:
            error(f'Splits error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_splits(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Transform splits response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            return {'success': True, 'source': 'alpha_vantage_data', 'symbol': symbol, 'data': data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming splits: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Splits transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_income_statement(self, symbol: str) -> Dict[str, Any]:
        """Get income statement"""
        try:
            with operation(f'AlphaVantage income statement for {symbol}'):
                params = {'function': 'INCOME_STATEMENT', 'symbol': symbol, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_financial_statement(data, symbol, 'income_statement')
        except Exception as e:
            error(f'Income statement error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_balance_sheet(self, symbol: str) -> Dict[str, Any]:
        """Get balance sheet"""
        try:
            with operation(f'AlphaVantage balance sheet for {symbol}'):
                params = {'function': 'BALANCE_SHEET', 'symbol': symbol, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_financial_statement(data, symbol, 'balance_sheet')
        except Exception as e:
            error(f'Balance sheet error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """Get cash flow statement"""
        try:
            with operation(f'AlphaVantage cash flow for {symbol}'):
                params = {'function': 'CASH_FLOW', 'symbol': symbol, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_financial_statement(data, symbol, 'cash_flow')
        except Exception as e:
            error(f'Cash flow error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_financial_statement(self, data: Dict, symbol: str, statement_type: str) -> Dict[str, Any]:
        """Transform financial statement response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            return {'success': True, 'source': 'alpha_vantage_data', 'symbol': symbol, 'statement_type': statement_type, 'data': data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming financial statement: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Financial statement transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """Get earnings history"""
        try:
            with operation(f'AlphaVantage earnings for {symbol}'):
                params = {'function': 'EARNINGS', 'symbol': symbol, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_earnings(data, symbol)
        except Exception as e:
            error(f'Earnings error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_earnings(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Transform earnings response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            return {'success': True, 'source': 'alpha_vantage_data', 'symbol': symbol, 'data': data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming earnings: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Earnings transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_earnings_estimates(self, symbol: str) -> Dict[str, Any]:
        """Get earnings estimates"""
        try:
            with operation(f'AlphaVantage earnings estimates for {symbol}'):
                params = {'function': 'EARNINGS_ESTIMATES', 'symbol': symbol, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_earnings_estimates(data, symbol)
        except Exception as e:
            error(f'Earnings estimates error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_earnings_estimates(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Transform earnings estimates response"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            return {'success': True, 'source': 'alpha_vantage_data', 'symbol': symbol, 'data': data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming earnings estimates: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Earnings estimates transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_listing_status(self, date: str=None, state: str='active') -> Dict[str, Any]:
        """Get listing and delisting status"""
        try:
            with operation('AlphaVantage listing status'):
                params = {'function': 'LISTING_STATUS', 'state': state, 'apikey': self.api_key}
                if date:
                    params['date'] = date
                session = await self._get_session()
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        csv_data = await response.text()
                        return self._transform_listing_status(csv_data, date, state)
                    else:
                        return {'success': False, 'error': f'HTTP {response.status}', 'source': 'alpha_vantage_data'}
        except Exception as e:
            error(f'Listing status error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_listing_status(self, csv_data: str, date: str, state: str) -> Dict[str, Any]:
        """Transform listing status CSV response"""
        try:
            lines = csv_data.strip().split('\n')
            if len(lines) < 2:
                return {'success': False, 'error': 'No listing data found', 'source': 'alpha_vantage_data'}
            headers = lines[0].split(',')
            data_rows = []
            for line in lines[1:]:
                values = line.split(',')
                if len(values) == len(headers):
                    row_data = dict(zip(headers, values))
                    data_rows.append(row_data)
            return {'success': True, 'source': 'alpha_vantage_data', 'date': date, 'state': state, 'data': data_rows, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming listing status: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Listing status transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_earnings_calendar(self, symbol: str=None, horizon: str='3month') -> Dict[str, Any]:
        """Get earnings calendar"""
        try:
            with operation('AlphaVantage earnings calendar'):
                params = {'function': 'EARNINGS_CALENDAR', 'horizon': horizon, 'apikey': self.api_key}
                if symbol:
                    params['symbol'] = symbol
                session = await self._get_session()
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        csv_data = await response.text()
                        return self._transform_earnings_calendar(csv_data, symbol, horizon)
                    else:
                        return {'success': False, 'error': f'HTTP {response.status}', 'source': 'alpha_vantage_data'}
        except Exception as e:
            error(f'Earnings calendar error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_earnings_calendar(self, csv_data: str, symbol: str, horizon: str) -> Dict[str, Any]:
        """Transform earnings calendar CSV response"""
        try:
            lines = csv_data.strip().split('\n')
            if len(lines) < 2:
                return {'success': False, 'error': 'No earnings calendar data found', 'source': 'alpha_vantage_data'}
            headers = lines[0].split(',')
            data_rows = []
            for line in lines[1:]:
                values = line.split(',')
                if len(values) == len(headers):
                    row_data = dict(zip(headers, values))
                    data_rows.append(row_data)
            return {'success': True, 'source': 'alpha_vantage_data', 'symbol': symbol, 'horizon': horizon, 'data': data_rows, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming earnings calendar: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'Earnings calendar transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_ipo_calendar(self) -> Dict[str, Any]:
        """Get IPO calendar"""
        try:
            with operation('AlphaVantage IPO calendar'):
                params = {'function': 'IPO_CALENDAR', 'apikey': self.api_key}
                session = await self._get_session()
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        csv_data = await response.text()
                        return self._transform_ipo_calendar(csv_data)
                    else:
                        return {'success': False, 'error': f'HTTP {response.status}', 'source': 'alpha_vantage_data'}
        except Exception as e:
            error(f'IPO calendar error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

    def _transform_ipo_calendar(self, csv_data: str) -> Dict[str, Any]:
        """Transform IPO calendar CSV response"""
        try:
            lines = csv_data.strip().split('\n')
            if len(lines) < 2:
                return {'success': False, 'error': 'No IPO calendar data found', 'source': 'alpha_vantage_data'}
            headers = lines[0].split(',')
            data_rows = []
            for line in lines[1:]:
                values = line.split(',')
                if len(values) == len(headers):
                    row_data = dict(zip(headers, values))
                    data_rows.append(row_data)
            return {'success': True, 'source': 'alpha_vantage_data', 'data': data_rows, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming IPO calendar: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': f'IPO calendar transformation error: {str(e)}', 'source': 'alpha_vantage_data'}

    @monitor_performance
    async def get_currency_exchange_rate(self, from_currency: str, to_currency: str) -> Dict[str, Any]:
        """Get real-time currency exchange rate"""
        try:
            with operation(f'AlphaVantage exchange rate {from_currency}/{to_currency}'):
                params = {'function': 'CURRENCY_EXCHANGE_RATE', 'from_currency': from_currency, 'to_currency': to_currency, 'apikey': self.api_key}
                data = await self._make_request(params)
                if not data.get('success', True):
                    return data
                return self._transform_exchange_rate(data, from_currency, to_currency)
        except Exception as e:
            error(f'Exchange rate error: {str(e)}', module='AlphaVantageProvider')
            return {'success': False, 'error': str(e), 'source': 'alpha_vantage_data'}

