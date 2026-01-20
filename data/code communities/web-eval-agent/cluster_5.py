# Cluster 5

def open_log_dashboard(url='http://127.0.0.1:5009'):
    """Opens or refreshes the dashboard in the browser."""
    if refresh_dashboard():
        try:
            send_log('Refreshed existing dashboard tab.', '🔄', log_type='status')
        except Exception:
            pass
        return
    try:
        webbrowser.open_new_tab(url)
        try:
            send_log(f'Opened new dashboard in browser at {url}.', '🌐', log_type='status')
        except Exception:
            pass
    except Exception as e:
        try:
            send_log(f'Could not open browser automatically: {e}', '⚠️', log_type='status')
        except Exception:
            pass

def refresh_dashboard():
    """Send refresh signal to all connected dashboard tabs."""
    if active_dashboard_tabs:
        socketio.emit('refresh_dashboard', {})
        return True
    return False

