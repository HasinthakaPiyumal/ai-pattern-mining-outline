# Cluster 8

@socketio.on('browser_input')
def handle_browser_input_event(data):
    """Handles browser interaction events received from the frontend."""
    event_type = data.get('type')
    details = data.get('details')
    if event_type != 'scroll':
        send_log(f'Received browser input: {event_type}', '🖱️', log_type='status')
    try:
        from .browser_utils import handle_browser_input, active_cdp_session, get_browser_task_loop
    except ImportError:
        error_msg = 'Could not import handle_browser_input from browser_utils'
        send_log(f'Input error: {error_msg}', '❌', log_type='status')
        return
    if not active_cdp_session:
        error_msg = 'No active CDP session for input handling'
        send_log(f'Input error: {error_msg}', '❌', log_type='status')
        return
    try:
        loop = get_browser_task_loop()
        if loop is None:
            send_log('Input error: Browser task loop not available', '❌', log_type='status')
            return
        asyncio.run_coroutine_threadsafe(handle_browser_input(event_type, details), loop)
        if event_type == 'scroll':
            return
        send_log(f'Input {event_type} scheduled for processing', '✅', log_type='status')
    except RuntimeError as e:
        error_msg = f'No running asyncio event loop found: {e}'
        send_log(f'Input error: {error_msg}', '❌', log_type='status')
    except Exception as e:
        error_msg = f'Error scheduling browser input handler: {e}'
        send_log(f'Input error: {error_msg}', '❌', log_type='status')

def get_browser_task_loop():
    """Get the asyncio loop used by run_browser_task."""
    global browser_task_loop
    return browser_task_loop

