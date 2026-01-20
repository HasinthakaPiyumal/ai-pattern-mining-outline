# Cluster 15

def decrypt_value(encrypted_value: str) -> Optional[str]:
    """Decrypt a value using AES-256-CBC.

    Args:
        encrypted_value (str): Base64-encoded encrypted value (IV + ciphertext)

    Returns:
        Optional[str]: Decrypted string value, or None if decryption fails
    """
    if not encrypted_value:
        return None
    secrets_to_try = ['LOCAL']
    for secret in secrets_to_try:
        try:
            key = get_key(secret)
            data = base64.b64decode(encrypted_value)
            iv = data[:16]
            encrypted = data[16:]
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            decrypted = decryptor.update(encrypted) + decryptor.finalize()
            pad_len = decrypted[-1]
            if pad_len < 1 or pad_len > 16:
                raise ValueError(f'Invalid padding length: {pad_len}')
            padding_bytes = decrypted[-pad_len:]
            if not all((b == pad_len for b in padding_bytes)):
                raise ValueError('Invalid PKCS7 padding')
            unpadded = decrypted[:-pad_len]
            result = unpadded.decode('utf-8')
            return result
        except Exception:
            continue
    return None

def get_key(secret: str) -> bytes:
    """Generate AES-256 key from secret using SHA256.

    Args:
        secret (str): The secret key string

    Returns:
        bytes: 32-byte key for AES-256
    """
    return hashlib.sha256(secret.encode()).digest()

def encrypt_value(value: str) -> Optional[str]:
    """Encrypt a value using AES-256-CBC.

    Args:
        value (str): Plain text value to encrypt

    Returns:
        Optional[str]: Base64-encoded encrypted value, or None if encryption fails
    """
    if not value:
        return None
    try:
        secret = get_encryption_secret()
        key = get_key(secret)
        iv = os.urandom(16)
        pad_len = 16 - len(value) % 16
        padded = value + chr(pad_len) * pad_len
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(padded.encode()) + encryptor.finalize()
        return base64.b64encode(iv + encrypted).decode('utf-8')
    except Exception as e:
        print(f'Error encrypting value: {e}')
        return None

def get_encryption_secret() -> str:
    """Get encryption secret from environment variable.

    Returns:
        str: The encryption secret key
    """
    secret = 'LOCAL'
    return secret

