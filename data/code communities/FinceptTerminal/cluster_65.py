# Cluster 65

def init_db():
    conn = duckdb.connect('fincept.db')
    conn.execute("\n        CREATE TABLE IF NOT EXISTS users (\n            id INTEGER PRIMARY KEY,\n            email VARCHAR UNIQUE NOT NULL,\n            password_hash VARCHAR NOT NULL,\n            full_name VARCHAR NOT NULL,\n            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n            subscription_plan VARCHAR DEFAULT NULL,\n            subscription_status VARCHAR DEFAULT 'inactive'\n        )\n    ")
    conn.execute("\n        CREATE TABLE IF NOT EXISTS payments (\n            id INTEGER PRIMARY KEY,\n            user_id INTEGER,\n            plan_name VARCHAR NOT NULL,\n            amount DECIMAL(10,2) NOT NULL,\n            payment_id VARCHAR UNIQUE,\n            status VARCHAR DEFAULT 'pending',\n            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n            FOREIGN KEY (user_id) REFERENCES users(id)\n        )\n    ")
    conn.close()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def get_db():
    return duckdb.connect('fincept.db')

def create_token(user_id: int) -> str:
    payload = {'user_id': user_id, 'exp': datetime.utcnow() + timedelta(hours=24)}
    return jwt.encode(payload, 'your-secret-key', algorithm='HS256')

def create_dodo_payment(amount: float, plan_name: str, user_email: str):
    """Create payment with Dodo (mock implementation)"""
    payment_id = f'dodo_{secrets.token_hex(8)}'
    dodo_payload = {'amount': amount, 'currency': 'USD', 'description': f'Fincept {plan_name} Plan', 'customer_email': user_email, 'return_url': 'http://localhost:8000/payment/success', 'cancel_url': 'http://localhost:8000/payment/cancel'}
    return {'payment_id': payment_id, 'payment_url': f'https://pay.dodo.com/checkout/{payment_id}', 'status': 'pending'}

