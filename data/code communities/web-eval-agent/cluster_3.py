# Cluster 3

def start_log_server(host='127.0.0.1', port=5009):
    """Starts the Flask-SocketIO server in a background thread."""

    def run_server():
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        socketio.run(app, host=host, port=port, log_output=False, use_reloader=False, allow_unsafe_werkzeug=True)
    template_dir = os.path.join(os.path.dirname(__file__), '../templates')
    static_dir = os.path.join(template_dir, 'static')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    os.path.join(template_dir, 'index.html')
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    send_log('Log server thread started.', '🚀', log_type='status')

