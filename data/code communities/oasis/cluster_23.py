# Cluster 23

@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    db_path = test_db_filepath
    mock_channel = MockChannel()
    platform_instance = Platform(db_path, mock_channel)
    return platform_instance

@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    db_path = test_db_filepath
    mock_channel = MockChannel()
    instance = Platform(db_path, mock_channel)
    return instance

@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    db_path = test_db_filepath
    mock_channel = MockChannel()
    instance = Platform(db_path, mock_channel)
    return instance

@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    db_path = test_db_filepath
    mock_channel = MockChannel()
    instance = Platform(db_path, mock_channel, allow_self_rating=False)
    return instance

@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    db_path = test_db_filepath
    mock_channel = MockChannel()
    instance = Platform(db_path, mock_channel)
    return instance

@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    db_path = test_db_filepath
    mock_channel = MockChannel()
    platform_instance = Platform(db_path, mock_channel, show_score=True)
    return platform_instance

@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    db_path = test_db_filepath
    mock_channel = MockChannel()
    instance = Platform(db_path, mock_channel)
    return instance

@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    mock_channel = MockChannel()
    instance = Platform(test_db_filepath, channel=mock_channel)
    return instance

@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    db_path = test_db_filepath
    mock_channel = MockChannel()
    instance = Platform(db_path=db_path, channel=mock_channel, refresh_rec_post_count=10, max_rec_post_len=10)
    return instance

@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    db_path = test_db_filepath
    mock_channel = MockChannel()
    platform_instance = Platform(db_path, mock_channel, allow_self_rating=False)
    return platform_instance

@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    db_path = test_db_filepath
    mock_channel = MockChannel()
    instance = Platform(db_path, mock_channel)
    return instance

@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    db_path = test_db_filepath
    mock_channel = MockChannel()
    instance = Platform(db_path, mock_channel)
    return instance

@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    db_path = test_db_filepath
    mock_channel = MockChannel()
    platform_instance = Platform(db_path, mock_channel)
    return platform_instance

@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    db_path = test_db_filepath
    mock_channel = MockChannel()
    instance = Platform(db_path, mock_channel)
    return instance

@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    db_path = test_db_filepath
    mock_channel = MockChannel()
    instance = Platform(db_path, mock_channel)
    return instance

@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    db_path = test_db_filepath
    mock_channel = MockChannel()
    instance = Platform(db_path=db_path, channel=mock_channel)
    return instance

