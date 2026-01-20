# Cluster 16

class AutoEncoder:
    type: EncoderType
    name: Optional[str]
    model: DenseEncoder | SparseEncoder

    def __init__(self, type: str, name: Optional[str]):
        self.type = EncoderType(type)
        self.name = name
        if self.type == EncoderType.AZURE:
            self.model = AzureOpenAIEncoder(name=name)
        elif self.type == EncoderType.COHERE:
            self.model = CohereEncoder(name=name)
        elif self.type == EncoderType.OPENAI:
            self.model = OpenAIEncoder(name=name)
        elif self.type == EncoderType.AURELIO:
            self.model = AurelioSparseEncoder(name=name)
        elif self.type == EncoderType.BM25:
            if name is None:
                name = 'bm25'
            self.model = BM25Encoder(name=name)
        elif self.type == EncoderType.TFIDF:
            if name is None:
                name = 'tfidf'
            self.model = TfidfEncoder(name=name)
        elif self.type == EncoderType.FASTEMBED:
            self.model = FastEmbedEncoder(name=name)
        elif self.type == EncoderType.HUGGINGFACE:
            self.model = HuggingFaceEncoder(name=name)
        elif self.type == EncoderType.MISTRAL:
            self.model = MistralEncoder(name=name)
        elif self.type == EncoderType.VOYAGE:
            self.model = VoyageEncoder(name=name)
        elif self.type == EncoderType.JINA:
            self.model = JinaEncoder(name=name)
        elif self.type == EncoderType.NIM:
            self.model = NimEncoder(name=name)
        elif self.type == EncoderType.VIT:
            self.model = VitEncoder(name=name)
        elif self.type == EncoderType.CLIP:
            self.model = CLIPEncoder(name=name)
        elif self.type == EncoderType.GOOGLE:
            self.model = GoogleEncoder(name=name)
        elif self.type == EncoderType.BEDROCK:
            self.model = BedrockEncoder(name=name)
        elif self.type == EncoderType.LITELLM:
            self.model = LiteLLMEncoder(name=name)
        elif self.type == EncoderType.OLLAMA:
            self.model = OllamaEncoder(name=name)
        elif self.type == EncoderType.LOCAL:
            self.model = LocalEncoder(name=name)
        else:
            raise ValueError(f"Encoder type '{type}' not supported")

    def __call__(self, texts: List[str]) -> List[List[float]] | List[SparseEmbedding]:
        return self.model(texts)

class TestOllamaEncoder:

    def test_ollama_encoder_init_success(self, mocker):
        mocker.patch('ollama.Client', return_value=Mock())
        encoder = OllamaEncoder(base_url='http://localhost:11434')
        assert encoder.client is not None
        assert encoder.type == 'ollama'

    def test_ollama_encoder_init_import_error(self, mocker):
        mocker.patch.dict('sys.modules', {'ollama': None})
        with patch('builtins.__import__', side_effect=ImportError("No module named 'ollama'")):
            with pytest.raises(ImportError):
                OllamaEncoder(base_url='http://localhost:11434')

    def test_ollama_encoder_call_success(self, mocker):
        mock_client = Mock()
        mock_embed_result = Mock()
        mock_embed_result.embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embed.return_value = mock_embed_result
        mocker.patch('ollama.Client', return_value=mock_client)
        encoder = OllamaEncoder(base_url='http://localhost:11434')
        encoder.client = mock_client
        docs = ['doc1', 'doc2']
        result = encoder(docs)
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embed.assert_called_once_with(model=encoder.name, input=docs)

    def test_ollama_encoder_call_client_not_initialized(self, mocker):
        encoder = OllamaEncoder(base_url='http://localhost:11434')
        encoder.client = None
        with pytest.raises(ValueError) as e:
            encoder(['doc1'])
        assert 'OLLAMA Platform client is not initialized.' in str(e.value)

    def test_ollama_encoder_call_api_error(self, mocker):
        mock_client = Mock()
        mock_client.embed.side_effect = Exception('API error')
        mocker.patch('ollama.Client', return_value=mock_client)
        encoder = OllamaEncoder(base_url='http://localhost:11434')
        encoder.client = mock_client
        with pytest.raises(ValueError) as e:
            encoder(['doc1'])
        assert 'OLLAMA API call failed. Error: API error' in str(e.value)

    def test_ollama_encoder_uses_env_base_url(self, mocker):
        test_url = 'http://env-ollama:1234'
        mock_client = Mock()
        mock_client.host = test_url
        mocker.patch('ollama.Client', return_value=mock_client)
        with patch.dict(os.environ, {'OLLAMA_BASE_URL': test_url}):
            encoder = OllamaEncoder()
            assert encoder.client is not None
            assert encoder.client.host == test_url

@pytest.fixture
def openai_encoder():
    if not has_valid_openai_api_key():
        return DenseEncoder()
    else:
        return OpenAIEncoder()

@pytest.fixture
@pytest.mark.skipif(os.environ.get('RUN_HF_TESTS') is None, reason='Set RUN_HF_TESTS=1 to run. This test downloads models from Hugging Face which can time out in CI.')
def bm25_encoder():
    sparse_encoder = BM25Encoder(use_default_params=True)
    sparse_encoder.fit([Route(name='test_route', utterances=UTTERANCES)])
    return sparse_encoder

