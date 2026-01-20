# Cluster 0

@wraps(f)
def decorated_function(*args, **kwargs):
    require_api_key = os.getenv('REQUIRE_API_KEY', 'true')
    require_api_key = require_api_key.lower() == 'true'
    if not require_api_key:
        return f(*args, **kwargs)
    api_key = request.headers.get('X-API-Key')
    bearer_token = get_token_from_header()
    if not (api_key or bearer_token) or (api_key and api_key != API_KEY) or (bearer_token and bearer_token != API_KEY):
        logger.warning(f'Invalid authentication attempt from {request.remote_addr}')
        abort(401, description='Invalid or missing authentication')
    return f(*args, **kwargs)

def get_token_from_header():
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return None
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        return None
    return parts[1]

