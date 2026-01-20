# Cluster 13

@app.route('/health', methods=['GET'])
def health_check():
    api_health = get_api_health()
    current_time = datetime.now(timezone.utc).isoformat()
    response = {'service': 'chipper-web', 'version': APP_VERSION, 'build': BUILD_NUMBER, 'status': 'healthy', 'timestamp': current_time, 'api': api_health}
    if api_health.get('status') == 'unhealthy':
        response['status'] = 'degraded'
    return jsonify(response)

def get_api_health() -> Dict[str, Any]:
    try:
        api_url = os.getenv('API_URL', 'http://localhost:8000')
        headers = {'X-API-Key': os.getenv('API_KEY', 'EXAMPLE_API_KEY')}
        response = requests.get(f'{api_url}/health', headers=headers, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f'API health check failed: {str(e)}')
        return {'status': 'unhealthy', 'error': 'An internal error has occurred.'}

