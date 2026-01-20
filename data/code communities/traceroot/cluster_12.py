# Cluster 12

def get_user_credentials(request: Request) -> tuple[str, str, str]:
    fake_email = 'user@example.com'
    fake_secret = 'fake_secret'
    fake_sub = 'fake_sub'
    return (fake_email, fake_secret, fake_sub)

