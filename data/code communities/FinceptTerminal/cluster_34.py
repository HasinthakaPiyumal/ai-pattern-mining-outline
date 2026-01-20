# Cluster 34

class ProfileTab(BaseTab):
    """Enhanced profile tab - refactored and optimized"""

    def __init__(self, app):
        super().__init__(app)
        self.constants = ProfileConstants()
        self.last_refresh = None
        self.usage_stats = {}
        self.request_count = 0
        self.logout_in_progress = False
        self.api_client = create_api_client(self._get_initial_session_data())
        self.data_manager = ProfileDataManager(app, self.api_client)
        self.ui_builder = ProfileUIBuilder(self)
        logger.info('ProfileTab initialized', context={'api_url': config.get_api_url()})

    def _get_initial_session_data(self):
        """Get initial session data safely"""
        if hasattr(self.app, 'get_session_data'):
            return self.app.get_session_data()
        elif hasattr(self.app, 'session_data'):
            return self.app.session_data
        return {self.constants.USER_TYPE_KEY: self.constants.UNKNOWN_USER_TYPE}

    def get_label(self):
        return 'Profile'

    @handle_errors('create_profile_content')
    def create_content(self):
        """Create profile content based on user type"""
        self.refresh_data()
        session_data = self.data_manager.get_session_data()
        user_type = session_data.get(self.constants.USER_TYPE_KEY, self.constants.UNKNOWN_USER_TYPE)
        content_creators = {self.constants.GUEST_USER_TYPE: self._create_guest_profile, self.constants.REGISTERED_USER_TYPE: self._create_user_profile, self.constants.UNKNOWN_USER_TYPE: self._create_unknown_profile}
        creator = content_creators.get(user_type, self._create_unknown_profile)
        creator()

    @handle_errors('refresh_profile_data')
    def refresh_data(self):
        """Refresh all profile data"""
        self.last_refresh = datetime.now()
        self.data_manager.invalidate_cache()
        session_data = self.data_manager.get_session_data()
        self.api_client = create_api_client(session_data)
        if session_data.get(self.constants.AUTHENTICATED_KEY) and self.api_client:
            self._fetch_authenticated_data()
        self._update_request_count()

    def _fetch_authenticated_data(self):
        """Fetch data for authenticated users"""
        try:
            if self.api_client.is_registered():
                profile_result = self.api_client.get_user_profile()
                if profile_result.get(self.constants.SUCCESS_KEY):
                    self.data_manager.update_session_data({'user_info': profile_result['profile']})
                usage_result = self.api_client.get_user_usage()
                if usage_result.get(self.constants.SUCCESS_KEY):
                    self.usage_stats = usage_result['usage']
            elif self.api_client.is_guest():
                status_result = self.api_client.get_guest_status()
                if status_result.get(self.constants.SUCCESS_KEY):
                    self.data_manager.update_session_data(status_result['status'])
        except Exception as e:
            logger.warning('Failed to fetch authenticated data', context={'error': str(e)})

    def _update_request_count(self):
        """Update request count from various sources"""
        if self.api_client:
            self.request_count = self.api_client.get_request_count()
        elif hasattr(self.app, 'api_request_count'):
            self.request_count = self.app.api_request_count
        else:
            session_data = self.data_manager.get_session_data()
            self.request_count = session_data.get('requests_today', 0)

    def _create_guest_profile(self):
        """Create guest user profile"""
        session_data = self.data_manager.get_session_data()
        api_key = session_data.get(self.constants.API_KEY_KEY)
        self.ui_builder.create_header('👤 Guest Profile', self.last_refresh)
        self.ui_builder.create_two_column_layout(lambda: self._create_guest_status_info(session_data, api_key), lambda: self._create_guest_upgrade_info(session_data))
        dpg.add_spacer(height=20)
        self._create_session_stats(session_data)

    def _create_user_profile(self):
        """Create registered user profile"""
        session_data = self.data_manager.get_session_data()
        user_info = session_data.get('user_info', {})
        username = user_info.get('username', 'User')
        self.ui_builder.create_header(f"🔑 {username}'s Profile", self.last_refresh)
        self.ui_builder.create_two_column_layout(lambda: self._create_user_account_info(user_info, session_data), lambda: self._create_user_usage_info(user_info, session_data))
        dpg.add_spacer(height=20)
        self._create_user_stats()

    def _create_unknown_profile(self):
        """Create unknown state profile"""
        self.ui_builder.create_header('❓ Unknown Session State', self.last_refresh)
        info_items = ['Unable to determine authentication status', 'This may indicate a configuration issue.', None, {'text': 'Try refreshing or restarting the application', 'color': self.constants.COLORS['warning']}]
        self.ui_builder.create_info_widget('Session Status', info_items, width=500, height=200)
        buttons = [{'label': '🔄 Refresh Profile', 'callback': self.manual_refresh}, {'label': 'Clear Session & Restart', 'callback': self.logout_user}]
        self.ui_builder.create_button_group(buttons)

    def _create_guest_status_info(self, session_data, api_key):
        """Create guest status information widget"""
        device_id = session_data.get(self.constants.DEVICE_ID_KEY, 'Unknown')
        display_device_id = device_id[:20] + '...' if len(device_id) > 20 else device_id
        daily_limit = session_data.get('daily_limit', self.constants.GUEST_DAILY_LIMIT)
        requests_today = session_data.get('requests_today', 0)
        remaining = max(0, daily_limit - requests_today)
        info_items = ['Account Type: Guest User', f'Device ID: {display_device_id}', None, self._get_api_key_info(api_key), None, f'Session Requests: {self.request_count}', f"Today's Requests: {requests_today}/{daily_limit}", {'text': f'Remaining Today: {remaining}', 'color': self.constants.COLORS['success'] if remaining > 10 else self.constants.COLORS['error']}, None, '✓ Basic market data', '✓ Real-time quotes', '✓ Public databases']
        self.ui_builder.create_info_widget('Current Session Status', info_items)

    def _create_guest_upgrade_info(self, session_data):
        """Create guest upgrade information widget"""
        api_key = session_data.get(self.constants.API_KEY_KEY)
        if api_key and api_key.startswith('fk_guest_'):
            current_status = '🔄 Current: Guest API Key'
            status_items = ['• Temporary access (24 hours)', '• 50 requests per day']
        else:
            current_status = '🔄 Current: Offline Mode'
            status_items = ['• No API access']
        info_items = [{'text': current_status, 'color': self.constants.COLORS['warning']}, None, *status_items, None, {'text': '🔑 Create Account', 'color': self.constants.COLORS['info']}, 'Get unlimited access:', '• Permanent API key', '• Unlimited requests', '• All databases access', '• Premium features']
        self.ui_builder.create_info_widget('Upgrade Your Access', info_items)
        buttons = [{'label': 'Create Free Account', 'callback': self.show_signup_info}, {'label': 'Sign In to Account', 'callback': self.show_login_info}]
        self.ui_builder.create_button_group(buttons)

    def _create_user_account_info(self, user_info, session_data):
        """Create user account information widget"""
        api_key = session_data.get(self.constants.API_KEY_KEY)
        info_items = [f'Username: {user_info.get('username', 'N/A')}', f'Email: {user_info.get('email', 'N/A')}', f'Account Type: {user_info.get('account_type', 'free').title()}', f'Member Since: {self._format_date(user_info.get('created_at'))}', None, {'text': 'Authentication:', 'color': self.constants.COLORS['info']}, self._get_api_key_info(api_key, is_user=True), None, '✓ Unlimited API requests', '✓ All database access', '✓ Premium features']
        self.ui_builder.create_info_widget('Account Details', info_items)
        buttons = [{'label': 'Regenerate API Key', 'callback': self.regenerate_api_key}, {'label': 'Switch Account', 'callback': self.logout_user}]
        self.ui_builder.create_button_group(buttons)

    def _create_user_usage_info(self, user_info, session_data):
        """Create user usage information widget"""
        credit_balance = user_info.get('credit_balance', 0)
        if credit_balance > 1000:
            balance_color, status = (self.constants.COLORS['success'], 'Excellent')
        elif credit_balance > 100:
            balance_color, status = (self.constants.COLORS['warning'], 'Good')
        else:
            balance_color, status = (self.constants.COLORS['error'], 'Low Credits')
        info_items = [f'Current Balance: {credit_balance} credits', {'text': f'Status: {status}', 'color': balance_color}, None, {'text': 'Live Usage Stats:', 'color': self.constants.COLORS['info']}, f'Total Requests: {self.usage_stats.get('total_requests', 'Loading...')}', f'Credits Used: {self.usage_stats.get('total_credits_used', 'Loading...')}', f'This Session: {self.request_count}', None, 'Quick Actions:']
        self.ui_builder.create_info_widget('Credits & Usage', info_items)
        buttons = [{'label': 'View Usage Details', 'callback': self.view_usage_stats}, {'label': 'API Documentation', 'callback': self.show_api_docs}, {'label': 'Subscription Info', 'callback': self.show_subscription_info}]
        self.ui_builder.create_button_group(buttons)

    def _create_session_stats(self, session_data):
        """Create session statistics for guest users"""
        dpg.add_text('📊 Live Session Statistics', color=self.constants.COLORS['info'])
        dpg.add_separator()
        dpg.add_spacer(height=10)
        api_key = session_data.get(self.constants.API_KEY_KEY)
        daily_limit = session_data.get('daily_limit', self.constants.GUEST_DAILY_LIMIT)
        requests_today = session_data.get('requests_today', 0)
        stats_text = [f'Session Requests: {self.request_count}', f'Daily Progress: {requests_today}/{daily_limit}', f'Authentication: {('API Key' if api_key else 'Offline')}', f'Server: {config.get_api_url()}']
        for stat in stats_text:
            dpg.add_text(stat)

    def _create_user_stats(self):
        """Create user statistics for registered users"""
        dpg.add_text('📊 Live Account Overview', color=self.constants.COLORS['info'])
        dpg.add_separator()
        dpg.add_spacer(height=10)
        stats_text = [f'Session Requests: {self.request_count}', f'Total Requests: {self.usage_stats.get('total_requests', 'Loading...')}', f'Success Rate: 100%', f'Server: {config.get_api_url()}', f'Last Update: {(self.last_refresh.strftime('%H:%M:%S') if self.last_refresh else 'Never')}']
        for stat in stats_text:
            dpg.add_text(stat)

    def _get_api_key_info(self, api_key, is_user=False):
        """Get API key information text"""
        if not api_key:
            return {'text': 'Method: No API Key', 'color': self.constants.COLORS['error']}
        if api_key.startswith('fk_user_'):
            return {'text': f'Method: Permanent API Key\nAPI Key: {api_key[:25]}...', 'color': self.constants.COLORS['success']}
        elif api_key.startswith('fk_guest_'):
            return {'text': f'Method: Temporary API Key\nAPI Key: {api_key[:20]}...', 'color': self.constants.COLORS['warning']}
        else:
            return {'text': f'Method: Legacy API Key\nAPI Key: {api_key[:20]}...', 'color': self.constants.COLORS['warning']}

    @lru_cache(maxsize=32)
    def _format_date(self, date_str):
        """Format date string for display"""
        if not date_str:
            return 'Never'
        try:
            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return date_obj.strftime('%Y-%m-%d %H:%M')
        except:
            return date_str

    @handle_errors('manual_refresh')
    def manual_refresh(self):
        """Manual refresh with error handling"""
        self.refresh_data()
        self._recreate_content()
        self.show_message('Profile refreshed successfully', 'success')

    @handle_errors('logout_user')
    def logout_user(self):
        """Complete logout process"""
        if self.logout_in_progress:
            return
        self.logout_in_progress = True
        try:
            self._update_logout_button_state(True)
            logger.info('Starting logout process')
            self._perform_api_logout()
            self.data_manager.clear_session()
            self._clear_saved_credentials()
            self._complete_logout()
        finally:
            self.logout_in_progress = False

    def _perform_api_logout(self):
        """Perform API logout with fallbacks"""
        if not self.api_client or not self.data_manager.get_session_data().get(self.constants.AUTHENTICATED_KEY):
            return True
        try:
            result = self.api_client.make_request('POST', '/auth/logout')
            if result.get(self.constants.SUCCESS_KEY):
                logger.info('API logout successful')
                return True
        except Exception as e:
            logger.warning('API logout failed, performing local cleanup', context={'error': str(e)})
        return True

    def _clear_saved_credentials(self):
        """Clear saved credentials"""
        try:
            from fincept_terminal.utils.Managers.session_manager import session_manager
            session_manager.clear_credentials()
            logger.info('Saved credentials cleared')
        except ImportError:
            logger.debug('Session manager not available')
        except Exception as e:
            logger.warning('Could not clear credentials', context={'error': str(e)})

    def _complete_logout(self):
        """Complete logout and exit"""
        logger.info('Logout completed successfully')
        print('\n✅ Logout completed successfully!\n🚪 Closing Fincept Terminal...\n\nTo access Fincept again:\n1. 🔄 Run the application\n2. 🔑 Choose authentication method\n3. 👤 Sign in or continue as guest\n\n👋 Thank you for using Fincept!\n        '.strip())
        threading.Timer(self.constants.LOGOUT_TIMER_DELAY, self._exit_application).start()

    def _update_logout_button_state(self, logging_out=False):
        """Update logout button state"""
        try:
            if dpg.does_item_exist('logout_btn'):
                if logging_out:
                    dpg.set_item_label('logout_btn', 'Logging out...')
                    dpg.disable_item('logout_btn')
                else:
                    dpg.set_item_label('logout_btn', '🚪 Logout')
                    dpg.enable_item('logout_btn')
        except Exception as e:
            logger.debug('Could not update logout button', context={'error': str(e)})

    def _exit_application(self):
        """Exit application with fallbacks"""
        exit_methods = [lambda: self.app.close_application(), lambda: self.app.shutdown(), lambda: dpg.stop_dearpygui(), lambda: __import__('sys').exit(0)]
        for exit_method in exit_methods:
            try:
                exit_method()
                return
            except:
                continue

    @handle_errors('regenerate_api_key')
    def regenerate_api_key(self):
        """Regenerate API key for authenticated users"""
        if not self.api_client or not self.api_client.is_registered():
            self.show_message('API key regeneration requires authenticated user', 'error')
            return
        result = self.api_client.regenerate_api_key()
        if result.get(self.constants.SUCCESS_KEY):
            new_api_key = result.get(self.constants.API_KEY_KEY)
            if new_api_key:
                self.data_manager.update_session_data({self.constants.API_KEY_KEY: new_api_key})
                threading.Timer(1.0, self.manual_refresh).start()
                self.show_message('API key regenerated successfully!', 'success')
            else:
                self.show_message('No new API key received', 'error')
        else:
            self.show_message('API key regeneration failed', 'error')

    def view_usage_stats(self):
        """Display detailed usage statistics"""
        stats = [f'📊 Detailed Usage Statistics:', f'Total Requests: {self.usage_stats.get('total_requests', 0)}', f'Credits Used: {self.usage_stats.get('total_credits_used', 0)}', f'Session Requests: {self.request_count}', f'Success Rate: {self.usage_stats.get('success_rate', 100)}%']
        for stat in stats:
            print(stat)

    def show_api_docs(self):
        """Open API documentation"""
        try:
            api_docs_url = f'{config.get_api_url()}/docs'
            webbrowser.open(api_docs_url)
            print(f'✅ Opened API docs: {api_docs_url}')
        except Exception as e:
            print(f'📖 Manual URL: {config.get_api_url()}/docs')

    def show_subscription_info(self):
        """Display subscription information"""
        session_data = self.data_manager.get_session_data()
        user_type = session_data.get(self.constants.USER_TYPE_KEY)
        if user_type == self.constants.REGISTERED_USER_TYPE:
            print('💳 Registered Account - Full access to all features')
        else:
            print('💳 Guest Account - Limited access. Create account for full features')

    def show_signup_info(self):
        """Display signup information"""
        print('📝 Create Account: Use logout button to return to authentication screen')

    def show_login_info(self):
        """Display login information"""
        print('🔑 Sign In: Use logout button to return to authentication screen')

    def show_message(self, message: str, msg_type: str='info'):
        """Display message with appropriate styling"""
        icons = {'success': '✅', 'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}
        icon = icons.get(msg_type, 'ℹ️')
        print(f'{icon} {message}')
        if msg_type == 'error':
            logger.error(message)
        elif msg_type == 'warning':
            logger.warning(message)
        else:
            logger.info(message)

    def _recreate_content(self):
        """Safely recreate tab content"""
        try:
            if hasattr(self, 'content_tag') and dpg.does_item_exist(self.content_tag):
                children = dpg.get_item_children(self.content_tag, 1)
                for child in children:
                    if dpg.does_item_exist(child):
                        dpg.delete_item(child)
            self.create_content()
        except Exception as e:
            logger.warning('Could not recreate content', context={'error': str(e)})

    @handle_errors('cleanup')
    def cleanup(self):
        """Cleanup resources"""
        self.api_client = None
        self.usage_stats = {}
        self.request_count = 0
        self.data_manager.invalidate_cache()
        self._format_date.cache_clear()
        logger.info('ProfileTab cleanup completed')

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass

def handle_errors(operation_name: str):
    """Decorator for standardized error handling"""

    def decorator(func):

        def wrapper(self, *args, **kwargs):
            try:
                with operation(operation_name):
                    return func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f'{operation_name} failed', exc_info=True)
                self.show_message(f'{operation_name} error: {e}', 'error')
                return None
        return wrapper
    return decorator

