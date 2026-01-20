# Cluster 25

def create_model_urls(server_config):
    urls = []
    for server in server_config:
        host = server['host']
        for port in server['ports']:
            url = f'http://{host}:{port}/v1'
            urls.append(url)
    return urls

