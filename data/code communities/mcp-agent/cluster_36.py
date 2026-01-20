# Cluster 36

def simple_jwt_encode(payload, secret):
    """
    Simple JWT encoder without external libraries.

    Args:
        payload: Dict containing the JWT claims
        secret: Secret key for signing

    Returns:
        JWT token string
    """
    if isinstance(secret, str):
        secret = secret.encode('utf-8')
    header = {'alg': 'HS256', 'typ': 'JWT'}
    header_encoded = base64url_encode(json.dumps(header, separators=(',', ':')))
    payload_encoded = base64url_encode(json.dumps(payload, separators=(',', ':')))
    signing_input = f'{header_encoded}.{payload_encoded}'.encode('utf-8')
    signature = hmac.new(secret, signing_input, hashlib.sha256).digest()
    signature_encoded = base64url_encode(signature)
    return f'{header_encoded}.{payload_encoded}.{signature_encoded}'

def base64url_encode(data):
    """
    Base64url encoding as specified in RFC 7515.
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    encoded = base64.urlsafe_b64encode(data).rstrip(b'=')
    return encoded.decode('utf-8')

def generate_jwt(user_id: str, email: str=None, name: str=None, api_token: bool=True, prefix: bool=False, nextauth_secret: str=None, expiry_days: int=365):
    """
    Generate a JWT token compatible with validateApiToken in the web app.

    Args:
        user_id: The user ID to include in the token
        email: Optional email to include in the token
        name: Optional name to include in the token
        api_token: Whether this is an API token (vs a session token)
        prefix: Whether to add the API_TOKEN_PREFIX to the token
        nextauth_secret: The secret used to sign the token (if not provided, will look for env var)
        expiry_days: Number of days until token expiry

    Returns:
        The generated JWT token as a string
    """
    if not nextauth_secret:
        nextauth_secret = os.environ.get('NEXTAUTH_SECRET')
        if not nextauth_secret:
            env_path = '/home/ubuntu/lmai/mcp-agent-cloud/www/.env'
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.startswith('NEXTAUTH_SECRET='):
                            parts = line.strip().split('=', 1)
                            if len(parts) == 2:
                                secret = parts[1].strip()
                                if secret.startswith('"') and secret.endswith('"') or (secret.startswith("'") and secret.endswith("'")):
                                    secret = secret[1:-1]
                                nextauth_secret = secret
                                break
        if not nextauth_secret:
            nextauth_secret = '3Jk0h98K1KKB7Jyh3/Kgp0bAKM0DSMcx1Jk7FJ6boNw'
            print('Warning: Using hardcoded NEXTAUTH_SECRET for testing.', file=sys.stderr)
    now = int(time.time())
    expiry = now + 60 * 60 * 24 * expiry_days
    payload = {'iat': now, 'exp': expiry, 'jti': str(uuid.uuid4()), 'id': user_id}
    if email:
        payload['email'] = email
    if name:
        payload['name'] = name
    if api_token:
        payload['apiToken'] = True
    token = simple_jwt_encode(payload, nextauth_secret)
    if prefix and api_token:
        return f'{API_TOKEN_PREFIX}{token}'
    else:
        return token

def main():
    parser = argparse.ArgumentParser(description='Generate JWT tokens for testing the secrets service API')
    parser.add_argument('--user-id', default=str(uuid.uuid4()), help='User ID to include in the token')
    parser.add_argument('--email', help='Email to include in the token')
    parser.add_argument('--name', help='Name to include in the token')
    parser.add_argument('--api-token', action='store_true', help='Include apiToken: true in the payload')
    parser.add_argument('--prefix', action='store_true', help='Add the API_TOKEN_PREFIX to the token')
    parser.add_argument('--nextauth-secret', help='Secret to use for signing (defaults to NEXTAUTH_SECRET env var)')
    parser.add_argument('--expiry-days', type=int, default=365, help='Number of days until token expiry')
    args = parser.parse_args()
    token = generate_jwt(user_id=args.user_id, email=args.email, name=args.name, api_token=args.api_token, prefix=args.prefix, nextauth_secret=args.nextauth_secret, expiry_days=args.expiry_days)
    print(token)

def generate_test_token():
    return generate_jwt(user_id='user_id', email='email', name='name', api_token=True, prefix=True, nextauth_secret='nextauthsecret', expiry_days=365)

