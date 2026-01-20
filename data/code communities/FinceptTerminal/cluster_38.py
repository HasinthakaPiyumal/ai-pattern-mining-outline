# Cluster 38

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

def error(msg, **kwargs):
    print(f'ERROR: {msg}')

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

def get_provider_status(self, provider_name: str) -> Dict[str, Any]:
    """Get comprehensive status of a provider"""
    try:
        settings_manager = self.get_settings_manager()
        status = {'name': provider_name, 'enabled': False, 'configured': False, 'tested': False, 'supports': [], 'last_used': None, 'error': None}
        if provider_name not in self.available_sources:
            status['error'] = f'Provider {provider_name} not found'
            return status
        provider_info = self.available_sources[provider_name]
        status['supports'] = provider_info.get('supports', [])
        if settings_manager:
            status['enabled'] = settings_manager.is_provider_enabled(provider_name)
            if provider_info.get('requires_auth', False):
                api_key = settings_manager.get_api_key(provider_name)
                status['configured'] = bool(api_key and len(api_key) > 5)
            else:
                status['configured'] = True
        return status
    except Exception as e:
        error(f'Error getting provider status: {str(e)}', module='DataSourceManager')
        return {'name': provider_name, 'error': str(e)}

class OECDProvider:
    """OECD data provider with complete API integration using OpenBB constants"""

    def __init__(self, rate_limit: int=5):
        self.base_url = 'https://sdmx.oecd.org/public/rest/data'
        self.rate_limit = rate_limit
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with SSL configuration"""
        if self._session is None or self._session.closed:
            ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            ssl_context.options |= 4
            connector = aiohttp.TCPConnector(limit=10, ssl=ssl_context)
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), connector=connector)
        return self._session

    async def _make_request(self, url: str, params: Dict[str, Any]=None) -> Dict[str, Any]:
        """Make API request with common error handling"""
        try:
            session = await self._get_session()
            headers = {'Accept': 'application/vnd.sdmx.data+csv; charset=utf-8'}
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    text = await response.text()
                    return {'success': True, 'data': text}
                else:
                    error(f'OECD API HTTP error: {response.status}', module='OECDProvider')
                    return {'success': False, 'error': f'HTTP {response.status}', 'source': 'oecd'}
        except Exception as e:
            error(f'OECD API request error: {str(e)}', module='OECDProvider')
            return {'success': False, 'error': str(e), 'source': 'oecd'}

    def _parse_csv_response(self, csv_text: str) -> pd.DataFrame:
        """Parse CSV response from OECD API"""
        try:
            df = pd.read_csv(StringIO(csv_text))
            if df.empty:
                return pd.DataFrame()
            column_mapping = {'REF_AREA': 'country', 'TIME_PERIOD': 'date', 'OBS_VALUE': 'value'}
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            return df
        except Exception as e:
            error(f'Error parsing CSV response: {str(e)}', module='OECDProvider')
            return pd.DataFrame()

    def _parse_date(self, date_str: str) -> str:
        """Parse OECD date format to standard format"""
        try:
            if 'Q' in date_str:
                year, quarter = date_str.split('-')
                quarter_num = int(quarter[1])
                month = quarter_num * 3
                return f'{year}-{month:02d}-01'
            elif len(date_str) == 4:
                return f'{date_str}-12-31'
            elif len(date_str) == 7:
                return f'{date_str}-01'
            else:
                return date_str
        except:
            return date_str

    def _validate_countries(self, countries: str, country_mapping: Dict[str, str]) -> str:
        """Validate and convert countries using OpenBB constants"""
        if countries == 'all':
            return ''
        country_list = [c.strip().lower() for c in countries.split(',')]
        valid_codes = []
        for country in country_list:
            if country in country_mapping:
                valid_codes.append(country_mapping[country])
            else:
                warning(f"Country '{country}' not supported for this indicator. Skipping...")
        if not valid_codes:
            raise ValueError(f'No valid countries found in: {countries}')
        return '+'.join(valid_codes)

    def get_available_countries(self, indicator: str) -> List[str]:
        """Get list of available countries for a specific indicator"""
        mapping = {'gdp_nominal': COUNTRY_TO_CODE_GDP, 'gdp_real': COUNTRY_TO_CODE_GDP, 'gdp_forecast': COUNTRY_TO_CODE_GDP_FORECAST, 'cpi': COUNTRY_TO_CODE_CPI, 'unemployment': COUNTRY_TO_CODE_UNEMPLOYMENT, 'interest_rates': COUNTRY_TO_CODE_IR, 'composite_leading_indicator': COUNTRY_TO_CODE_CLI, 'house_price_index': COUNTRY_TO_CODE_RGDP, 'share_price_index': COUNTRY_TO_CODE_SHARES}
        if indicator not in mapping:
            return []
        return list(mapping[indicator].keys())

    @monitor_performance
    async def get_gdp_nominal(self, countries: str='united_states', frequency: str='quarter', start_date: str=None, end_date: str=None, units: str='level', price_base: str='current_prices') -> Dict[str, Any]:
        """Get nominal GDP data"""
        try:
            with operation(f'OECD GDP nominal for {countries}'):
                freq = 'Q' if frequency == 'quarter' else 'A'
                unit = 'USD' if units == 'level' else 'INDICES' if units == 'index' else 'CAPITA'
                price = 'V' if price_base == 'current_prices' else 'LR'
                if unit == 'INDICES' and price == 'V':
                    price = 'DR'
                country_codes = self._validate_countries(countries, COUNTRY_TO_CODE_GDP)
                url = f'{self.base_url}/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_{unit},1.1/{freq}..{country_codes}.S1..B1GQ.....{price}..?'
                if start_date:
                    url += f'&startPeriod={start_date}'
                if end_date:
                    url += f'&endPeriod={end_date}'
                url += '&dimensionAtObservation=TIME_PERIOD&detail=dataonly&format=csvfile'
                if units == 'capita':
                    url = url.replace('B1GQ', 'B1GQ_POP')
                response = await self._make_request(url)
                if not response['success']:
                    return response
                df = self._parse_csv_response(response['data'])
                if df.empty:
                    return {'success': False, 'error': 'No data found', 'source': 'oecd'}
                df['country'] = df['country'].map(CODE_TO_COUNTRY_GDP)
                df['date'] = df['date'].apply(self._parse_date)
                if units == 'level':
                    df['value'] = (df['value'].astype(float) * 1000000).astype('int64')
                return {'success': True, 'source': 'oecd', 'indicator': 'gdp_nominal', 'countries': countries, 'frequency': frequency, 'units': units, 'data': df.to_dict('records'), 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'GDP nominal error: {str(e)}', module='OECDProvider')
            return {'success': False, 'error': str(e), 'source': 'oecd'}

    @monitor_performance
    async def get_gdp_real(self, countries: str='united_states', frequency: str='quarter', start_date: str=None, end_date: str=None) -> Dict[str, Any]:
        """Get real GDP data (PPP-adjusted)"""
        try:
            with operation(f'OECD GDP real for {countries}'):
                freq = 'Q' if frequency == 'quarter' else 'A'
                country_codes = self._validate_countries(countries, COUNTRY_TO_CODE_GDP)
                url = f'{self.base_url}/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,1.1/{freq}..{country_codes}.S1..B1GQ._Z...USD_PPP.LR.LA.T0102?'
                if start_date:
                    url += f'&startPeriod={start_date}'
                if end_date:
                    url += f'&endPeriod={end_date}'
                url += '&dimensionAtObservation=TIME_PERIOD&detail=dataonly&format=csvfile'
                response = await self._make_request(url)
                if not response['success']:
                    return response
                df = self._parse_csv_response(response['data'])
                if df.empty:
                    return {'success': False, 'error': 'No data found', 'source': 'oecd'}
                df['country'] = df['country'].map(CODE_TO_COUNTRY_GDP)
                df['date'] = df['date'].apply(self._parse_date)
                df['value'] = (df['value'].astype(float) * 1000000).astype('int64')
                return {'success': True, 'source': 'oecd', 'indicator': 'gdp_real', 'countries': countries, 'frequency': frequency, 'data': df.to_dict('records'), 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'GDP real error: {str(e)}', module='OECDProvider')
            return {'success': False, 'error': str(e), 'source': 'oecd'}

    @monitor_performance
    async def get_gdp_forecast(self, countries: str='all', frequency: str='annual', start_date: str=None, end_date: str=None, units: str='volume') -> Dict[str, Any]:
        """Get GDP forecast data"""
        try:
            with operation(f'OECD GDP forecast for {countries}'):
                freq = 'Q' if frequency == 'quarter' else 'A'
                measure_dict = {'current_prices': 'GDP_USD', 'volume': 'GDPV_USD', 'capita': 'GDPVD_CAP', 'growth': 'GDPV_ANNPCT', 'deflator': 'PGDP'}
                measure = measure_dict.get(units, 'GDPV_USD')
                country_codes = self._validate_countries(countries, COUNTRY_TO_CODE_GDP_FORECAST)
                url = f'{self.base_url}/OECD.ECO.MAD,DSD_EO@DF_EO,1.1/{country_codes}.{measure}.{freq}?'
                if start_date:
                    url += f'startPeriod={start_date}'
                if end_date:
                    url += f'&endPeriod={end_date}' if start_date else f'endPeriod={end_date}'
                url += '&dimensionAtObservation=TIME_PERIOD&detail=dataonly&format=csvfile'
                response = await self._make_request(url)
                if not response['success']:
                    return response
                df = self._parse_csv_response(response['data'])
                if df.empty:
                    return {'success': False, 'error': 'No data found', 'source': 'oecd'}
                df['country'] = df['country'].map(CODE_TO_COUNTRY_GDP_FORECAST)
                df['date'] = df['date'].apply(self._parse_date)
                if units == 'growth':
                    df['value'] = df['value'].astype(float) / 100
                else:
                    df['value'] = df['value'].astype('int64')
                return {'success': True, 'source': 'oecd', 'indicator': 'gdp_forecast', 'countries': countries, 'frequency': frequency, 'units': units, 'data': df.to_dict('records'), 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'GDP forecast error: {str(e)}', module='OECDProvider')
            return {'success': False, 'error': str(e), 'source': 'oecd'}

    @monitor_performance
    async def get_cpi(self, countries: str='united_states', frequency: str='monthly', start_date: str=None, end_date: str=None, transform: str='index', harmonized: bool=False, expenditure: str='total') -> Dict[str, Any]:
        """Get Consumer Price Index data"""
        try:
            with operation(f'OECD CPI for {countries}'):
                methodology = 'HICP' if harmonized else 'N'
                freq = frequency[0].upper()
                units_map = {'index': 'IX', 'yoy': 'PA', 'mom': 'PC', 'period': 'PC'}
                units = units_map.get(transform, 'IX')
                expenditure_map = {'total': '_T', 'food_non_alcoholic_beverages': 'CP01', 'alcoholic_beverages_tobacco_narcotics': 'CP02', 'clothing_footwear': 'CP03', 'housing_water_electricity_gas': 'CP04', 'furniture_household_equipment': 'CP05', 'health': 'CP06', 'transport': 'CP07', 'communication': 'CP08', 'recreation_culture': 'CP09', 'education': 'CP10', 'restaurants_hotels': 'CP11', 'miscellaneous_goods_services': 'CP12', 'energy': 'CP045_0722', 'goods': 'GD', 'housing': 'CP041T043', 'housing_excluding_rentals': 'CP041T043X042', 'all_non_food_non_energy': '_TXCP01_NRG', 'services_less_housing': 'SERVXCP041_042_0432', 'services': 'SERV'}
                exp_code = expenditure_map.get(expenditure, '_T')
                country_codes = self._validate_countries(countries, COUNTRY_TO_CODE_CPI)
                url = f'{self.base_url}/OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0/{country_codes}.{freq}.{methodology}.CPI.{units}.{exp_code}.N.'
                if start_date or end_date:
                    url += '?'
                    if start_date:
                        url += f'startPeriod={start_date}'
                    if end_date:
                        url += f'&endPeriod={end_date}' if start_date else f'endPeriod={end_date}'
                response = await self._make_request(url)
                if not response['success']:
                    return response
                df = self._parse_csv_response(response['data'])
                if df.empty:
                    return {'success': False, 'error': 'No data found', 'source': 'oecd'}
                df['country'] = df['country'].map(CODE_TO_COUNTRY_CPI)
                df['date'] = df['date'].apply(self._parse_date)
                if transform in ('yoy', 'mom', 'period'):
                    df['value'] = df['value'].astype(float) / 100
                return {'success': True, 'source': 'oecd', 'indicator': 'cpi', 'countries': countries, 'frequency': frequency, 'transform': transform, 'expenditure': expenditure, 'harmonized': harmonized, 'data': df.to_dict('records'), 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'CPI error: {str(e)}', module='OECDProvider')
            return {'success': False, 'error': str(e), 'source': 'oecd'}

    @monitor_performance
    async def get_unemployment(self, countries: str='united_states', frequency: str='monthly', start_date: str=None, end_date: str=None, sex: str='total', age: str='total', seasonal_adjustment: bool=False) -> Dict[str, Any]:
        """Get unemployment rate data"""
        try:
            with operation(f'OECD unemployment for {countries}'):
                sex_map = {'total': '_T', 'male': 'M', 'female': 'F'}
                sex_code = sex_map.get(sex, '_T')
                age_map = {'total': 'Y_GE15', '15-24': 'Y15T24', '25+': 'Y_GE25'}
                age_code = age_map.get(age, 'Y_GE15')
                freq = frequency[0].upper()
                seasonal = 'Y' if seasonal_adjustment else 'N'
                country_codes = self._validate_countries(countries, COUNTRY_TO_CODE_UNEMPLOYMENT)
                url = f'{self.base_url}/OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0/{country_codes}..._Z.{seasonal}.{sex_code}.{age_code}..{freq}'
                if start_date or end_date:
                    url += '?'
                    if start_date:
                        url += f'startPeriod={start_date}'
                    if end_date:
                        url += f'&endPeriod={end_date}' if start_date else f'endPeriod={end_date}'
                url += '&dimensionAtObservation=TIME_PERIOD&detail=dataonly'
                response = await self._make_request(url)
                if not response['success']:
                    return response
                df = self._parse_csv_response(response['data'])
                if df.empty:
                    return {'success': False, 'error': 'No data found', 'source': 'oecd'}
                df['country'] = df['country'].map(CODE_TO_COUNTRY_UNEMPLOYMENT)
                df['date'] = df['date'].apply(self._parse_date)
                df['value'] = df['value'].astype(float) / 100
                return {'success': True, 'source': 'oecd', 'indicator': 'unemployment', 'countries': countries, 'frequency': frequency, 'sex': sex, 'age': age, 'seasonal_adjustment': seasonal_adjustment, 'data': df.to_dict('records'), 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Unemployment error: {str(e)}', module='OECDProvider')
            return {'success': False, 'error': str(e), 'source': 'oecd'}

    @monitor_performance
    async def get_interest_rates(self, countries: str='united_states', duration: str='short', frequency: str='monthly', start_date: str=None, end_date: str=None) -> Dict[str, Any]:
        """Get interest rates data"""
        try:
            with operation(f'OECD interest rates for {countries}'):
                duration_map = {'immediate': 'IRSTCI', 'short': 'IR3TIB', 'long': 'IRLT'}
                duration_code = duration_map.get(duration, 'IR3TIB')
                freq = frequency[0].upper()
                country_codes = self._validate_countries(countries, COUNTRY_TO_CODE_IR)
                url = f'{self.base_url}/OECD.SDD.STES,DSD_KEI@DF_KEI,4.0/{country_codes}.{freq}.{duration_code}....?'
                if start_date:
                    url += f'startPeriod={start_date}'
                if end_date:
                    url += f'&endPeriod={end_date}' if start_date else f'endPeriod={end_date}'
                url += '&dimensionAtObservation=TIME_PERIOD&detail=dataonly'
                response = await self._make_request(url)
                if not response['success']:
                    return response
                df = self._parse_csv_response(response['data'])
                if df.empty:
                    return {'success': False, 'error': 'No data found', 'source': 'oecd'}
                df['country'] = df['country'].map(CODE_TO_COUNTRY_IR)
                df['date'] = df['date'].apply(self._parse_date)
                df['value'] = df['value'].astype(float) / 100
                return {'success': True, 'source': 'oecd', 'indicator': 'interest_rates', 'countries': countries, 'frequency': frequency, 'duration': duration, 'data': df.to_dict('records'), 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Interest rates error: {str(e)}', module='OECDProvider')
            return {'success': False, 'error': str(e), 'source': 'oecd'}

    @monitor_performance
    async def get_composite_leading_indicator(self, countries: str='g20', start_date: str=None, end_date: str=None, adjustment: str='amplitude', growth_rate: bool=False) -> Dict[str, Any]:
        """Get Composite Leading Indicator data"""
        try:
            with operation(f'OECD CLI for {countries}'):
                growth = 'GY' if growth_rate else 'IX'
                adjust = 'AA' if adjustment == 'amplitude' else 'NOR'
                if growth == 'GY':
                    adjust = ''
                country_codes = self._validate_countries(countries, COUNTRY_TO_CODE_CLI)
                url = f'{self.base_url}/OECD.SDD.STES,DSD_STES@DF_CLI,4.1/{country_codes}.M.LI...{adjust}.{growth}..H'
                if start_date or end_date:
                    url += '?'
                    if start_date:
                        url += f'startPeriod={start_date}'
                    if end_date:
                        url += f'&endPeriod={end_date}' if start_date else f'endPeriod={end_date}'
                url += '&dimensionAtObservation=TIME_PERIOD&detail=dataonly&format=csvfile'
                response = await self._make_request(url)
                if not response['success']:
                    return response
                df = self._parse_csv_response(response['data'])
                if df.empty:
                    return {'success': False, 'error': 'No data found', 'source': 'oecd'}
                df['country'] = df['country'].map(CODE_TO_COUNTRY_CLI)
                df['date'] = df['date'].apply(self._parse_date)
                if growth_rate:
                    df['value'] = df['value'].astype(float) / 100
                return {'success': True, 'source': 'oecd', 'indicator': 'composite_leading_indicator', 'countries': countries, 'adjustment': adjustment, 'growth_rate': growth_rate, 'data': df.to_dict('records'), 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'CLI error: {str(e)}', module='OECDProvider')
            return {'success': False, 'error': str(e), 'source': 'oecd'}

    @monitor_performance
    async def get_house_price_index(self, countries: str='united_states', frequency: str='quarter', start_date: str=None, end_date: str=None, transform: str='yoy') -> Dict[str, Any]:
        """Get House Price Index data"""
        try:
            with operation(f'OECD house price index for {countries}'):
                freq_map = {'monthly': 'M', 'quarter': 'Q', 'annual': 'A'}
                freq = freq_map.get(frequency, 'Q')
                transform_map = {'yoy': 'PA', 'period': 'PC', 'index': 'IX'}
                trans = transform_map.get(transform, 'PA')
                country_codes = self._validate_countries(countries, COUNTRY_TO_CODE_RGDP)
                url = f'{self.base_url}/OECD.SDD.TPS,DSD_RHPI_TARGET@DF_RHPI_TARGET,1.0/COU.{country_codes}.{freq}.RHPI.{trans}....?'
                if start_date:
                    url += f'startPeriod={start_date}'
                if end_date:
                    url += f'&endPeriod={end_date}' if start_date else f'endPeriod={end_date}'
                url += '&dimensionAtObservation=TIME_PERIOD&detail=dataonly'
                response = await self._make_request(url)
                if not response['success']:
                    return response
                df = self._parse_csv_response(response['data'])
                if df.empty:
                    return {'success': False, 'error': 'No data found', 'source': 'oecd'}
                df['country'] = df['country'].map(CODE_TO_COUNTRY_RGDP)
                df['date'] = df['date'].apply(self._parse_date)
                return {'success': True, 'source': 'oecd', 'indicator': 'house_price_index', 'countries': countries, 'frequency': frequency, 'transform': transform, 'data': df.to_dict('records'), 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'House price index error: {str(e)}', module='OECDProvider')
            return {'success': False, 'error': str(e), 'source': 'oecd'}

    @monitor_performance
    async def get_share_price_index(self, countries: str='united_states', frequency: str='monthly', start_date: str=None, end_date: str=None) -> Dict[str, Any]:
        """Get Share Price Index data"""
        try:
            with operation(f'OECD share price index for {countries}'):
                freq_map = {'monthly': 'M', 'quarter': 'Q', 'annual': 'A'}
                freq = freq_map.get(frequency, 'M')
                country_codes = self._validate_countries(countries, COUNTRY_TO_CODE_SHARES)
                url = f'{self.base_url}/OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/{country_codes}.{freq}.SHARE......?'
                if start_date:
                    url += f'startPeriod={start_date}'
                if end_date:
                    url += f'&endPeriod={end_date}' if start_date else f'endPeriod={end_date}'
                url += '&dimensionAtObservation=TIME_PERIOD&detail=dataonly'
                response = await self._make_request(url)
                if not response['success']:
                    return response
                df = self._parse_csv_response(response['data'])
                if df.empty:
                    return {'success': False, 'error': 'No data found', 'source': 'oecd'}
                df['country'] = df['country'].map(CODE_TO_COUNTRY_SHARES)
                df['date'] = df['date'].apply(self._parse_date)
                return {'success': True, 'source': 'oecd', 'indicator': 'share_price_index', 'countries': countries, 'frequency': frequency, 'data': df.to_dict('records'), 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Share price index error: {str(e)}', module='OECDProvider')
            return {'success': False, 'error': str(e), 'source': 'oecd'}

    async def close(self):
        """Close the aiohttp session"""
        if self._session and (not self._session.closed):
            await self._session.close()

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, '_session') and self._session and (not self._session.closed):
            asyncio.create_task(self._session.close())

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

class IMFProvider:
    """IMF data provider with complete API integration"""

    def __init__(self, rate_limit: int=5):
        self.base_url = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/'
        self.portwatch_base = 'https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services'
        self.rate_limit = rate_limit
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), connector=aiohttp.TCPConnector(limit=10))
        return self._session

    async def _make_request(self, url: str, params: Optional[Dict]=None) -> Dict[str, Any]:
        """Make API request with error handling"""
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error(f'IMF API HTTP error: {response.status}', module='IMFProvider')
                    return {'success': False, 'error': f'HTTP {response.status}', 'source': 'imf'}
        except Exception as e:
            error(f'IMF API request error: {str(e)}', module='IMFProvider')
            return {'success': False, 'error': str(e), 'source': 'imf'}

    def _check_api_errors(self, data: Dict) -> Optional[Dict[str, Any]]:
        """Check for common API errors"""
        if 'ErrorDetails' in data:
            return {'success': False, 'error': data['ErrorDetails'].get('Message', 'API Error'), 'source': 'imf'}
        return None

    @monitor_performance
    async def get_available_indicators(self, query: Optional[str]=None) -> Dict[str, Any]:
        """Get available IMF indicators"""
        try:
            with operation('IMF available indicators'):
                indicators = {'IRFCL': 'International Reserves and Foreign Currency Liquidity', 'FSI': 'Financial Soundness Indicators', 'DOT': 'Direction of Trade Statistics'}
                if query:
                    indicators = {k: v for k, v in indicators.items() if query.lower() in v.lower()}
                return {'success': True, 'source': 'imf', 'data': indicators, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'IMF available indicators error: {str(e)}', module='IMFProvider')
            return {'success': False, 'error': str(e), 'source': 'imf'}

    @monitor_performance
    async def get_direction_of_trade(self, country: str, counterpart: str='all', direction: str='exports', frequency: str='quarter', start_date: Optional[str]=None, end_date: Optional[str]=None) -> Dict[str, Any]:
        """Get bilateral trade data"""
        try:
            with operation(f'IMF direction of trade {country}-{counterpart}'):
                indicators = {'exports': 'TXG_FOB_USD', 'imports': 'TMG_CIF_USD', 'balance': 'TBG_USD'}
                freq = {'annual': 'A', 'quarter': 'Q', 'month': 'M'}
                f = freq.get(frequency, 'Q')
                indicator = indicators.get(direction, 'TXG_FOB_USD')
                date_range = ''
                if start_date and end_date:
                    date_range = f'?startPeriod={start_date}&endPeriod={end_date}'
                url = f'{self.base_url}CompactData/DOT/{f}.{country}.{indicator}.{counterpart}{date_range}'
                print(f'Trade URL: {url}')
                data = await self._make_request(url)
                print(f'Trade raw data: {data}')
                if not data.get('success', True):
                    return data
                return self._transform_dot_data(data, country, counterpart, direction)
        except Exception as e:
            error(f'IMF direction of trade error: {str(e)}', module='IMFProvider')
            return {'success': False, 'error': str(e), 'source': 'imf'}

    def _transform_dot_data(self, data: Dict, country: str, counterpart: str, direction: str) -> Dict[str, Any]:
        """Transform direction of trade data"""
        try:
            if data.get('success') is False:
                return data
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            compact_data = data.get('CompactData', {})
            dataset = compact_data.get('DataSet', {})
            series = dataset.get('Series', {})
            if not series:
                return {'success': False, 'error': 'No trade data found in response', 'source': 'imf'}
            obs = series.get('Obs', [])
            if isinstance(obs, dict):
                obs = [obs]
            result_data = []
            for o in obs:
                try:
                    value = o.get('@OBS_VALUE')
                    if value is not None:
                        result_data.append({'date': o.get('@TIME_PERIOD'), 'value': float(value), 'country': country, 'counterpart': counterpart, 'direction': direction})
                except (ValueError, TypeError):
                    continue
            if not result_data:
                return {'success': False, 'error': f'No valid observations found for {country}-{counterpart}', 'source': 'imf'}
            return {'success': True, 'source': 'imf', 'data': result_data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming DOT data: {str(e)}', module='IMFProvider')
            return {'success': False, 'error': str(e), 'source': 'imf'}

    @monitor_performance
    async def get_economic_indicators(self, symbol: str='RAF_USD', country: str='US', frequency: str='quarter', start_date: Optional[str]=None, end_date: Optional[str]=None) -> Dict[str, Any]:
        """Get economic indicators (IRFCL/FSI)"""
        try:
            with operation(f'IMF economic indicators {symbol}'):
                freq = {'annual': 'A', 'quarter': 'Q', 'month': 'M'}
                f = freq.get(frequency, 'Q')
                dataset = 'IRFCL' if any((x in symbol for x in ['RAF', 'RAO', 'RAC', 'RAM'])) else 'FSI'
                date_range = ''
                if start_date and end_date:
                    date_range = f'?startPeriod={start_date}&endPeriod={end_date}'
                sector = '' if dataset == 'IRFCL' else ''
                url = f'{self.base_url}CompactData/{dataset}/{f}.{country}.{symbol}.{sector}{date_range}'
                data = await self._make_request(url)
                if not data.get('success', True):
                    return data
                return self._transform_economic_data(data, symbol, country)
        except Exception as e:
            error(f'IMF economic indicators error: {str(e)}', module='IMFProvider')
            return {'success': False, 'error': str(e), 'source': 'imf'}

    def _transform_economic_data(self, data: Dict, symbol: str, country: str) -> Dict[str, Any]:
        """Transform economic indicators data"""
        try:
            error_response = self._check_api_errors(data)
            if error_response:
                return error_response
            series = data.get('CompactData', {}).get('DataSet', {}).get('Series', [])
            if not series:
                return {'success': False, 'error': 'No economic data found', 'source': 'imf'}
            if isinstance(series, dict):
                series = [series]
            result_data = []
            for s in series:
                obs = s.get('Obs', [])
                if isinstance(obs, dict):
                    obs = [obs]
                for o in obs:
                    result_data.append({'date': o.get('@TIME_PERIOD'), 'value': float(o.get('@OBS_VALUE', 0)), 'symbol': s.get('@INDICATOR'), 'country': country})
            return {'success': True, 'source': 'imf', 'data': result_data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming economic data: {str(e)}', module='IMFProvider')
            return {'success': False, 'error': str(e), 'source': 'imf'}

    @monitor_performance
    async def get_maritime_chokepoint_info(self) -> Dict[str, Any]:
        """Get static maritime chokepoint information"""
        try:
            with operation('IMF maritime chokepoint info'):
                url = f'{self.portwatch_base}/PortWatch_chokepoints_database/FeatureServer/0/query'
                params = {'outFields': '*', 'where': '1=1', 'f': 'geojson'}
                data = await self._make_request(url, params)
                if not data.get('success', True):
                    return data
                return self._transform_chokepoint_info(data)
        except Exception as e:
            error(f'IMF chokepoint info error: {str(e)}', module='IMFProvider')
            return {'success': False, 'error': str(e), 'source': 'imf'}

    def _transform_chokepoint_info(self, data: Dict) -> Dict[str, Any]:
        """Transform chokepoint info data"""
        try:
            features = data.get('features', [])
            if not features:
                return {'success': False, 'error': 'No chokepoint data found', 'source': 'imf'}
            result_data = []
            for feature in features:
                props = feature.get('properties', {})
                result_data.append({'chokepoint_code': props.get('portid'), 'name': props.get('portname'), 'latitude': props.get('lat'), 'longitude': props.get('lon'), 'vessel_count_total': props.get('vessel_count_total', 0), 'vessel_count_tanker': props.get('vessel_count_tanker', 0), 'vessel_count_container': props.get('vessel_count_container', 0)})
            return {'success': True, 'source': 'imf', 'data': result_data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming chokepoint info: {str(e)}', module='IMFProvider')
            return {'success': False, 'error': str(e), 'source': 'imf'}

    @monitor_performance
    async def get_maritime_chokepoint_volume(self, chokepoint: Optional[str]=None, start_date: Optional[str]=None, end_date: Optional[str]=None) -> Dict[str, Any]:
        """Get daily chokepoint volume data"""
        try:
            with operation(f'IMF chokepoint volume {chokepoint or 'all'}'):
                url = f'{self.portwatch_base}/Daily_Chokepoints_Data/FeatureServer/0/query'
                where_clause = '1=1'
                if chokepoint:
                    where_clause += f" AND portid = '{chokepoint}'"
                if start_date and end_date:
                    where_clause += f" AND date >= TIMESTAMP '{start_date} 00:00:00' AND date <= TIMESTAMP '{end_date} 00:00:00'"
                params = {'where': where_clause, 'outFields': '*', 'orderByFields': 'date', 'f': 'json'}
                data = await self._make_request(url, params)
                if not data.get('success', True):
                    return data
                return self._transform_chokepoint_volume(data)
        except Exception as e:
            error(f'IMF chokepoint volume error: {str(e)}', module='IMFProvider')
            return {'success': False, 'error': str(e), 'source': 'imf'}

    def _transform_chokepoint_volume(self, data: Dict) -> Dict[str, Any]:
        """Transform chokepoint volume data"""
        try:
            features = data.get('features', [])
            if not features:
                return {'success': False, 'error': 'No chokepoint volume data found', 'source': 'imf'}
            result_data = []
            for feature in features:
                attrs = feature.get('attributes', {})
                year = attrs.get('year')
                month = attrs.get('month')
                day = attrs.get('day')
                date_str = f'{year}-{month:02d}-{day:02d}' if all([year, month, day]) else None
                result_data.append({'date': date_str, 'chokepoint': attrs.get('portname'), 'vessels_total': attrs.get('n_total', 0), 'vessels_cargo': attrs.get('n_cargo', 0), 'vessels_tanker': attrs.get('n_tanker', 0), 'capacity_total': attrs.get('capacity', 0)})
            return {'success': True, 'source': 'imf', 'data': result_data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming chokepoint volume: {str(e)}', module='IMFProvider')
            return {'success': False, 'error': str(e), 'source': 'imf'}

    @monitor_performance
    async def get_port_info(self, continent: Optional[str]=None, country: Optional[str]=None, limit: Optional[int]=None) -> Dict[str, Any]:
        """Get static port information"""
        try:
            with operation(f'IMF port info {country or continent or 'all'}'):
                url = f'{self.portwatch_base}/PortWatch_ports_database/FeatureServer/0/query'
                params = {'where': '1=1', 'outFields': '*', 'returnGeometry': 'false', 'f': 'json'}
                data = await self._make_request(url, params)
                if not data.get('success', True):
                    return data
                return self._transform_port_info(data, continent, country, limit)
        except Exception as e:
            error(f'IMF port info error: {str(e)}', module='IMFProvider')
            return {'success': False, 'error': str(e), 'source': 'imf'}

    def _transform_port_info(self, data: Dict, continent: Optional[str]=None, country: Optional[str]=None, limit: Optional[int]=None) -> Dict[str, Any]:
        """Transform port info data"""
        try:
            features = data.get('features', [])
            if not features:
                return {'success': False, 'error': 'No port data found', 'source': 'imf'}
            result_data = []
            for feature in features:
                attrs = feature.get('attributes', {})
                if country and attrs.get('ISO3') != country.upper():
                    continue
                if continent and attrs.get('continent') != continent:
                    continue
                result_data.append({'port_code': attrs.get('portid'), 'port_name': attrs.get('portname'), 'country': attrs.get('countrynoaccents'), 'country_code': attrs.get('ISO3'), 'continent': attrs.get('continent'), 'latitude': attrs.get('lat'), 'longitude': attrs.get('lon'), 'vessel_count_total': attrs.get('vessel_count_total', 0)})
            result_data.sort(key=lambda x: x['vessel_count_total'], reverse=True)
            if limit:
                result_data = result_data[:limit]
            return {'success': True, 'source': 'imf', 'data': result_data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming port info: {str(e)}', module='IMFProvider')
            return {'success': False, 'error': str(e), 'source': 'imf'}

    @monitor_performance
    async def get_port_volume(self, port_code: Optional[str]=None, country: Optional[str]=None, start_date: Optional[str]=None, end_date: Optional[str]=None) -> Dict[str, Any]:
        """Get daily port volume data"""
        try:
            with operation(f'IMF port volume {port_code or country or 'all'}'):
                url = f'{self.portwatch_base}/Daily_Trade_Data/FeatureServer/0/query'
                where_clause = '1=1'
                if port_code:
                    where_clause += f" AND portid = '{port_code}'"
                if country:
                    where_clause += f" AND ISO3 = '{country.upper()}'"
                if start_date and end_date:
                    where_clause += f" AND date >= TIMESTAMP '{start_date} 00:00:00' AND date <= TIMESTAMP '{end_date} 00:00:00'"
                params = {'where': where_clause, 'outFields': '*', 'orderByFields': 'date', 'f': 'json'}
                data = await self._make_request(url, params)
                if not data.get('success', True):
                    return data
                return self._transform_port_volume(data)
        except Exception as e:
            error(f'IMF port volume error: {str(e)}', module='IMFProvider')
            return {'success': False, 'error': str(e), 'source': 'imf'}

    def _transform_port_volume(self, data: Dict) -> Dict[str, Any]:
        """Transform port volume data"""
        try:
            features = data.get('features', [])
            if not features:
                return {'success': False, 'error': 'No port volume data found', 'source': 'imf'}
            result_data = []
            for feature in features:
                attrs = feature.get('attributes', {})
                year = attrs.get('year')
                month = attrs.get('month')
                day = attrs.get('day')
                date_str = f'{year}-{month:02d}-{day:02d}' if all([year, month, day]) else None
                result_data.append({'date': date_str, 'port_code': attrs.get('portid'), 'port_name': attrs.get('portname'), 'country_code': attrs.get('ISO3'), 'portcalls': attrs.get('portcalls', 0), 'imports': attrs.get('import', 0), 'exports': attrs.get('export', 0), 'imports_tanker': attrs.get('import_tanker', 0), 'exports_tanker': attrs.get('export_tanker', 0)})
            return {'success': True, 'source': 'imf', 'data': result_data, 'fetched_at': datetime.now().isoformat()}
        except Exception as e:
            error(f'Error transforming port volume: {str(e)}', module='IMFProvider')
            return {'success': False, 'error': str(e), 'source': 'imf'}

    async def close(self):
        """Close the aiohttp session"""
        if self._session and (not self._session.closed):
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

