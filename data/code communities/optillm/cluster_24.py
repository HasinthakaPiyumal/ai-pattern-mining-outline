# Cluster 24

def fetch_webpage_content(url: str, max_length: int=100000, verify_ssl: Optional[bool]=None, cert_path: Optional[str]=None) -> str:
    try:
        headers = {'User-Agent': f'optillm/{__version__} (https://github.com/codelion/optillm)'}
        if verify_ssl is None:
            verify_ssl = server_config.get('ssl_verify', True)
        if cert_path is None:
            cert_path = server_config.get('ssl_cert_path', '')
        if not verify_ssl:
            verify = False
        elif cert_path:
            verify = cert_path
        else:
            verify = True
        response = requests.get(url, headers=headers, timeout=10, verify=verify)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        for script in soup(['script', 'style']):
            script.decompose()
        text_elements = []
        for tag in ['article', 'main', 'div[role="main"]', '.main-content']:
            content = soup.select_one(tag)
            if content:
                text_elements.extend(content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table']))
                break
        if not text_elements:
            text_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table'])
        content_parts = []
        for element in text_elements:
            if element.name == 'table':
                table_content = []
                headers = element.find_all('th')
                if headers:
                    header_text = ' | '.join((header.get_text(strip=True) for header in headers))
                    table_content.append(header_text)
                for row in element.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        row_text = ' | '.join((cell.get_text(strip=True) for cell in cells))
                        table_content.append(row_text)
                content_parts.append('\n' + '\n'.join(table_content) + '\n')
            else:
                content_parts.append(element.get_text(strip=False))
        text = ' '.join(content_parts)
        text = re.sub('\\s+', ' ', text).strip()
        text = re.sub('\\[.*?\\]+', '', text)
        if len(text) > max_length:
            text = text[:max_length] + '...'
        return text
    except Exception as e:
        return f'Error fetching content: {str(e)}'

def run(system_prompt, initial_query: str, client=None, model=None) -> Tuple[str, int]:
    urls = extract_urls(initial_query)
    modified_query = initial_query
    for url in urls:
        content = fetch_webpage_content(url)
        domain = urlparse(url).netloc
        modified_query = modified_query.replace(url, f'{url} [Content from {domain}: {content}]')
    return (modified_query, 0)

def extract_urls(text: str) -> List[str]:
    url_pattern = re.compile('https?://[^\\s\\\'"]+')
    urls = url_pattern.findall(text)
    cleaned_urls = []
    for url in urls:
        url = re.sub('[,\\\'\\"\\)\\]]+$', '', url)
        cleaned_urls.append(url)
    return cleaned_urls

class TestSSLConfiguration(unittest.TestCase):
    """Test SSL configuration via CLI arguments and environment variables."""

    def setUp(self):
        """Reset server_config before each test."""
        self.original_config = server_config.copy()
        for key in ['OPTILLM_SSL_VERIFY', 'OPTILLM_SSL_CERT_PATH']:
            if key in os.environ:
                del os.environ[key]

    def tearDown(self):
        """Restore original server_config after each test."""
        server_config.clear()
        server_config.update(self.original_config)

    def test_default_ssl_verify_enabled(self):
        """Test that SSL verification is enabled by default."""
        self.assertTrue(server_config.get('ssl_verify', True))
        self.assertEqual(server_config.get('ssl_cert_path', ''), '')

    def test_cli_no_ssl_verify_flag(self):
        """Test --no-ssl-verify CLI flag disables SSL verification."""
        with patch('sys.argv', ['optillm', '--no-ssl-verify']):
            args = parse_args()
            self.assertFalse(args.ssl_verify)

    def test_cli_ssl_cert_path(self):
        """Test --ssl-cert-path CLI argument."""
        test_cert_path = '/path/to/ca-bundle.crt'
        with patch('sys.argv', ['optillm', '--ssl-cert-path', test_cert_path]):
            args = parse_args()
            self.assertEqual(args.ssl_cert_path, test_cert_path)

    def test_env_ssl_verify_false(self):
        """Test OPTILLM_SSL_VERIFY=false environment variable."""
        os.environ['OPTILLM_SSL_VERIFY'] = 'false'
        with patch('sys.argv', ['optillm']):
            args = parse_args()
            self.assertFalse(args.ssl_verify)

    def test_env_ssl_verify_true(self):
        """Test OPTILLM_SSL_VERIFY=true environment variable."""
        os.environ['OPTILLM_SSL_VERIFY'] = 'true'
        with patch('sys.argv', ['optillm']):
            args = parse_args()
            self.assertTrue(args.ssl_verify)

    def test_env_ssl_cert_path(self):
        """Test OPTILLM_SSL_CERT_PATH environment variable."""
        test_cert_path = '/etc/ssl/certs/custom-ca.pem'
        os.environ['OPTILLM_SSL_CERT_PATH'] = test_cert_path
        with patch('sys.argv', ['optillm']):
            args = parse_args()
            self.assertEqual(args.ssl_cert_path, test_cert_path)

    def test_cli_overrides_env(self):
        """Test that CLI arguments override environment variables."""
        os.environ['OPTILLM_SSL_VERIFY'] = 'true'
        with patch('sys.argv', ['optillm', '--no-ssl-verify']):
            args = parse_args()
            self.assertFalse(args.ssl_verify)

class TestPluginSSLConfiguration(unittest.TestCase):
    """Test that plugins properly use SSL configuration."""

    def setUp(self):
        """Set up test environment."""
        self.original_config = server_config.copy()

    def tearDown(self):
        """Restore original server_config."""
        server_config.clear()
        server_config.update(self.original_config)

    @patch('optillm.plugins.readurls_plugin.requests.get')
    def test_readurls_plugin_ssl_verify_disabled(self, mock_requests_get):
        """Test readurls plugin respects SSL verification disabled."""
        from optillm.plugins.readurls_plugin import fetch_webpage_content
        server_config['ssl_verify'] = False
        server_config['ssl_cert_path'] = ''
        mock_response = MagicMock()
        mock_response.content = b'<html><body><p>Test content</p></body></html>'
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response
        fetch_webpage_content('https://example.com')
        mock_requests_get.assert_called_once()
        call_kwargs = mock_requests_get.call_args[1]
        self.assertIn('verify', call_kwargs)
        self.assertFalse(call_kwargs['verify'])

    @patch('optillm.plugins.readurls_plugin.requests.get')
    def test_readurls_plugin_ssl_verify_enabled(self, mock_requests_get):
        """Test readurls plugin respects SSL verification enabled."""
        from optillm.plugins.readurls_plugin import fetch_webpage_content
        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = ''
        mock_response = MagicMock()
        mock_response.content = b'<html><body><p>Test content</p></body></html>'
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response
        fetch_webpage_content('https://example.com')
        mock_requests_get.assert_called_once()
        call_kwargs = mock_requests_get.call_args[1]
        self.assertIn('verify', call_kwargs)
        self.assertTrue(call_kwargs['verify'])

    @patch('optillm.plugins.readurls_plugin.requests.get')
    def test_readurls_plugin_custom_cert_path(self, mock_requests_get):
        """Test readurls plugin uses custom certificate path."""
        from optillm.plugins.readurls_plugin import fetch_webpage_content
        test_cert_path = '/path/to/custom-ca.pem'
        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = test_cert_path
        mock_response = MagicMock()
        mock_response.content = b'<html><body><p>Test content</p></body></html>'
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response
        fetch_webpage_content('https://example.com')
        mock_requests_get.assert_called_once()
        call_kwargs = mock_requests_get.call_args[1]
        self.assertIn('verify', call_kwargs)
        self.assertEqual(call_kwargs['verify'], test_cert_path)

