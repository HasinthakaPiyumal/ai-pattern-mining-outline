# Cluster 18

def get_config():
    import httpx
    API_KEY = None
    ssl_verify = server_config.get('ssl_verify', True)
    ssl_cert_path = server_config.get('ssl_cert_path', '')
    if not ssl_verify:
        logger.warning('SSL certificate verification is DISABLED. This is insecure and should only be used for development.')
        http_client = httpx.Client(verify=False)
    elif ssl_cert_path:
        logger.info(f'Using custom CA certificate bundle: {ssl_cert_path}')
        http_client = httpx.Client(verify=ssl_cert_path)
    else:
        http_client = httpx.Client(verify=True)
    if os.environ.get('OPTILLM_API_KEY'):
        from optillm.inference import create_inference_client
        API_KEY = os.environ.get('OPTILLM_API_KEY')
        default_client = create_inference_client()
    elif os.environ.get('CEREBRAS_API_KEY'):
        API_KEY = os.environ.get('CEREBRAS_API_KEY')
        base_url = server_config['base_url']
        if base_url != '':
            default_client = Cerebras(api_key=API_KEY, base_url=base_url, http_client=http_client)
        else:
            default_client = Cerebras(api_key=API_KEY, http_client=http_client)
    elif os.environ.get('OPENAI_API_KEY'):
        API_KEY = os.environ.get('OPENAI_API_KEY')
        base_url = server_config['base_url']
        if base_url != '':
            default_client = OpenAI(api_key=API_KEY, base_url=base_url)
            logger.info(f'Created OpenAI client with base_url: {base_url}')
        else:
            default_client = OpenAI(api_key=API_KEY)
            logger.info('Created OpenAI client without base_url')
    elif os.environ.get('AZURE_OPENAI_API_KEY'):
        API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
        API_VERSION = os.environ.get('AZURE_API_VERSION')
        AZURE_ENDPOINT = os.environ.get('AZURE_API_BASE')
        if API_KEY is not None:
            default_client = AzureOpenAI(api_key=API_KEY, api_version=API_VERSION, azure_endpoint=AZURE_ENDPOINT, http_client=http_client)
        else:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            azure_credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(azure_credential, 'https://cognitiveservices.azure.com/.default')
            default_client = AzureOpenAI(api_version=API_VERSION, azure_endpoint=AZURE_ENDPOINT, azure_ad_token_provider=token_provider, http_client=http_client)
    else:
        from optillm.litellm_wrapper import LiteLLMWrapper
        default_client = LiteLLMWrapper()
        logger.info('Created LiteLLMWrapper as fallback')
    logger.info(f'Client type: {type(default_client)}')
    return (default_client, API_KEY)

def create_inference_client() -> InferenceClient:
    """Factory function to create an inference client"""
    return InferenceClient()

@app.route('/v1/models', methods=['GET'])
def proxy_models():
    logger.info('Received request to /v1/models')
    default_client, API_KEY = get_config()
    try:
        if server_config['base_url']:
            client = OpenAI(api_key=API_KEY, base_url=server_config['base_url'])
            models_response = client.models.list()
            models_data = {'object': 'list', 'data': [model.dict() for model in models_response.data]}
        else:
            current_model = server_config.get('model', 'gpt-3.5-turbo')
            models_data = {'object': 'list', 'data': [{'id': current_model, 'object': 'model', 'created': 1677610602, 'owned_by': 'optillm'}]}
        logger.debug('Models retrieved successfully')
        return (jsonify(models_data), 200)
    except Exception as e:
        logger.error(f'Error fetching models: {str(e)}')
        return (jsonify({'error': f'Error fetching models: {str(e)}'}), 500)

class TestHTTPClientSSLConfiguration(unittest.TestCase):
    """Test that SSL configuration is properly applied to HTTP clients."""

    def setUp(self):
        """Set up test environment."""
        self.original_config = server_config.copy()

    def tearDown(self):
        """Restore original server_config."""
        server_config.clear()
        server_config.update(self.original_config)

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_httpx_client_ssl_verify_disabled(self):
        """Test httpx.Client created with verify=False when SSL disabled."""
        from optillm.server import get_config
        server_config['ssl_verify'] = False
        server_config['ssl_cert_path'] = ''
        with patch('httpx.Client') as mock_httpx_client, patch('optillm.server.OpenAI') as mock_openai:
            get_config()
            mock_httpx_client.assert_called_once_with(verify=False)

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_httpx_client_ssl_verify_enabled(self):
        """Test httpx.Client created with verify=True by default."""
        from optillm.server import get_config
        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = ''
        with patch('httpx.Client') as mock_httpx_client, patch('optillm.server.OpenAI') as mock_openai:
            get_config()
            mock_httpx_client.assert_called_once_with(verify=True)

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_httpx_client_custom_cert_path(self):
        """Test httpx.Client created with custom certificate path."""
        from optillm.server import get_config
        test_cert_path = '/path/to/custom-ca.pem'
        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = test_cert_path
        with patch('httpx.Client') as mock_httpx_client, patch('optillm.server.OpenAI') as mock_openai:
            get_config()
            mock_httpx_client.assert_called_once_with(verify=test_cert_path)

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_openai_client_receives_http_client(self):
        """Test that OpenAI client receives the configured httpx client."""
        from optillm.server import get_config
        server_config['ssl_verify'] = False
        server_config['ssl_cert_path'] = ''
        server_config['base_url'] = ''
        mock_http_client_instance = MagicMock()
        with patch('httpx.Client', return_value=mock_http_client_instance) as mock_httpx_client, patch('optillm.server.OpenAI') as mock_openai:
            get_config()
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            self.assertIn('http_client', call_kwargs)
            self.assertEqual(call_kwargs['http_client'], mock_http_client_instance)

    @patch.dict(os.environ, {'CEREBRAS_API_KEY': 'test-key'})
    def test_cerebras_client_receives_http_client(self):
        """Test that Cerebras client receives the configured httpx client."""
        from optillm.server import get_config
        server_config['ssl_verify'] = False
        server_config['ssl_cert_path'] = ''
        server_config['base_url'] = ''
        mock_http_client_instance = MagicMock()
        with patch('httpx.Client', return_value=mock_http_client_instance) as mock_httpx_client, patch('optillm.server.Cerebras') as mock_cerebras:
            get_config()
            mock_cerebras.assert_called_once()
            call_kwargs = mock_cerebras.call_args[1]
            self.assertIn('http_client', call_kwargs)
            self.assertEqual(call_kwargs['http_client'], mock_http_client_instance)

    @patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': 'test-key', 'AZURE_API_VERSION': '2024-02-15-preview', 'AZURE_API_BASE': 'https://test.openai.azure.com'})
    def test_azure_client_receives_http_client(self):
        """Test that AzureOpenAI client receives the configured httpx client."""
        from optillm.server import get_config
        server_config['ssl_verify'] = False
        server_config['ssl_cert_path'] = ''
        mock_http_client_instance = MagicMock()
        with patch('httpx.Client', return_value=mock_http_client_instance) as mock_httpx_client, patch('optillm.server.AzureOpenAI') as mock_azure:
            get_config()
            mock_azure.assert_called_once()
            call_kwargs = mock_azure.call_args[1]
            self.assertIn('http_client', call_kwargs)
            self.assertEqual(call_kwargs['http_client'], mock_http_client_instance)

class TestSSLWarnings(unittest.TestCase):
    """Test that appropriate warnings are shown when SSL is disabled."""

    def setUp(self):
        """Set up test environment."""
        self.original_config = server_config.copy()

    def tearDown(self):
        """Restore original server_config."""
        server_config.clear()
        server_config.update(self.original_config)

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_warning_when_ssl_disabled(self):
        """Test that a warning is logged when SSL verification is disabled."""
        from optillm.server import get_config
        server_config['ssl_verify'] = False
        server_config['ssl_cert_path'] = ''
        with patch('httpx.Client') as mock_httpx_client, patch('optillm.server.OpenAI') as mock_openai, patch('optillm.server.logger.warning') as mock_logger_warning:
            get_config()
            mock_logger_warning.assert_called()
            warning_message = mock_logger_warning.call_args[0][0]
            self.assertIn('SSL certificate verification is DISABLED', warning_message)
            self.assertIn('insecure', warning_message.lower())

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_info_when_custom_cert_used(self):
        """Test that an info message is logged when using custom certificate."""
        from optillm.server import get_config
        test_cert_path = '/path/to/custom-ca.pem'
        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = test_cert_path
        with patch('httpx.Client') as mock_httpx_client, patch('optillm.server.OpenAI') as mock_openai, patch('optillm.server.logger.info') as mock_logger_info:
            get_config()
            mock_logger_info.assert_called()
            info_message = mock_logger_info.call_args[0][0]
            self.assertIn('custom CA certificate bundle', info_message)
            self.assertIn(test_cert_path, info_message)

