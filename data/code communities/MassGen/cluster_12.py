# Cluster 12

def test_setup_api_keys_copies_gemini_key():
    """Test that GEMINI_API_KEY is copied to GOOGLE_GEMINI_API_KEY."""
    os.environ['GEMINI_API_KEY'] = 'test_key'
    if 'GOOGLE_GEMINI_API_KEY' in os.environ:
        del os.environ['GOOGLE_GEMINI_API_KEY']
    setup_api_keys()
    assert os.environ['GOOGLE_GEMINI_API_KEY'] == 'test_key'
    del os.environ['GEMINI_API_KEY']
    del os.environ['GOOGLE_GEMINI_API_KEY']

def setup_api_keys() -> None:
    """Set up API keys for AG2 compatibility."""
    if 'GEMINI_API_KEY' in os.environ and 'GOOGLE_GEMINI_API_KEY' not in os.environ:
        os.environ['GOOGLE_GEMINI_API_KEY'] = os.environ['GEMINI_API_KEY']

