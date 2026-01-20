# Cluster 17

def check_saved_credentials() -> Optional[Dict[str, Any]]:
    """OPTIMIZED: Fast credential check with minimal API calls"""
    try:
        if is_strict_mode():
            if not check_api_availability():
                warning('API unavailable in strict mode', module='main')
                return None
        fresh_session = session_manager.get_fresh_session()
        if fresh_session:
            user_type = fresh_session.get('user_type', 'unknown')
            authenticated = fresh_session.get('authenticated', False)
            info(f'Fresh session obtained - User: {user_type}, Auth: {authenticated}', module='main')
        else:
            warning('No fresh session available', module='main')
        return fresh_session
    except Exception as e:
        error(f'Credential check failed: {str(e)}', module='main')
        return None

def check_api_availability() -> bool:
    """PERFORMANCE: Fast API availability check with caching"""
    try:
        cache_key = '_api_availability_cache'
        cache_time_key = '_api_availability_time'
        if hasattr(check_api_availability, cache_key):
            cached_time = getattr(check_api_availability, cache_time_key, 0)
            if time.time() - cached_time < 30:
                return getattr(check_api_availability, cache_key)
        available = session_manager.is_api_available()
        setattr(check_api_availability, cache_key, available)
        setattr(check_api_availability, cache_time_key, time.time())
        info(f'API availability: {available}', module='main')
        return available
    except Exception as e:
        error(f'API availability check failed: {str(e)}', module='main')
        return False

@monitor_performance
def main():
    """OPTIMIZED: High-performance main entry point"""
    try:
        setup_for_gui()
        set_debug_mode(False)
        info('Fincept Terminal starting up', module='main')
        try:
            health = health_check()
            if health['status'] != 'healthy':
                warning(f'Logging system health: {health['status']}', module='logger')
        except Exception:
            pass
        fresh_session = check_saved_credentials()
        if fresh_session:
            session_data = fresh_session
            info('Using fresh session data', module='main')
        else:
            try:
                from fincept_terminal.utils.Authentication.splash_auth import show_authentication_splash
                is_first_time = session_manager.is_first_time_user()
                info(f'Showing authentication splash - First time: {is_first_time}', module='main')
                session_data = show_authentication_splash(is_first_time_user=is_first_time)
            except ImportError as e:
                critical(f'Authentication module unavailable: {str(e)}', module='main')
                return
            except Exception as e:
                error(f'Authentication splash failed: {str(e)}', module='main')
                return
        if not session_data or not session_data.get('authenticated'):
            warning('Authentication failed or cancelled', module='main')
            return
        try:
            session_manager.save_session_credentials(session_data)
            debug('Session credentials saved', module='session')
        except Exception as e:
            warning(f'Session save failed: {str(e)}', module='session')
        info('Authentication completed - Launching application', module='main')
        app = HighPerformanceMainApplication(session_data)
        app.run()
    except KeyboardInterrupt:
        info('Application interrupted by user (Ctrl+C)', module='main')
    except Exception as e:
        critical(f'Application failed to start: {str(e)}', module='main')
        raise
    finally:
        try:
            info('Application shutdown initiated', module='main')
        except Exception as e:
            print(f'[CRITICAL] Final cleanup failed: {e}')

