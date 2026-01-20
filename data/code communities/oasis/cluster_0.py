# Cluster 0

def check_port_open(host, port):
    while True:
        url = f'http://{host}:{port}/health'
        try:
            response = requests.get(url)
            if response.status_code == 200:
                break
            else:
                time.sleep(0.3)
        except Exception:
            time.sleep(0.3)

