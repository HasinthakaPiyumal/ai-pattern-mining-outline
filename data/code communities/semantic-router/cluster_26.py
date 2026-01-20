# Cluster 26

@pytest.fixture
def cohere_encoder(mocker):
    mocker.patch.object(CohereEncoder, '__call__', side_effect=mock_encoder_call)

    async def async_mock_encoder_call(docs=None, utterances=None):
        texts = docs if docs is not None else utterances
        return mock_encoder_call(texts)
    mocker.patch.object(CohereEncoder, 'acall', side_effect=async_mock_encoder_call)
    return CohereEncoder(name='test-cohere-encoder', cohere_api_key='test_api_key')

def mock_encoder_call(utterances):
    mock_responses = {'Hello': [0.1, 0.2, 0.3], 'Hi': [0.4, 0.5, 0.6], 'Goodbye': [0.7, 0.8, 0.9], 'Bye': [1.0, 1.1, 1.2], 'Au revoir': [1.3, 1.4, 1.5], 'Asparagus': [-2.0, 1.0, 0.0]}
    return [mock_responses.get(u, [0.0, 0.0, 0.0]) for u in utterances]

@pytest.fixture
def openai_encoder(mocker):
    mocker.patch('openai.OpenAI')
    mocker.patch.object(OpenAIEncoder, '__call__', side_effect=mock_encoder_call)

    async def async_mock_encoder_call(docs=None, utterances=None):
        texts = docs if docs is not None else utterances
        return mock_encoder_call(texts)
    mocker.patch.object(OpenAIEncoder, 'acall', side_effect=async_mock_encoder_call)
    encoder = OpenAIEncoder(name='text-embedding-3-small')
    return encoder

@pytest.fixture
def cohere_encoder(mocker):
    mocker.patch.object(CohereEncoder, '__call__', side_effect=mock_encoder_call)

    async def async_mock_encoder_call(docs=None, utterances=None):
        texts = docs if docs is not None else utterances
        return mock_encoder_call(texts)
    mocker.patch.object(CohereEncoder, 'acall', side_effect=async_mock_encoder_call)
    return CohereEncoder(name='test-cohere-encoder', cohere_api_key='test_api_key')

@pytest.fixture
def openai_encoder(mocker):
    mocker.patch.object(OpenAIEncoder, '__call__', side_effect=mock_encoder_call)

    async def async_mock_encoder_call(docs=None, utterances=None):
        texts = docs if docs is not None else utterances
        return mock_encoder_call(texts)
    mocker.patch.object(OpenAIEncoder, 'acall', side_effect=async_mock_encoder_call)
    return OpenAIEncoder(name='text-embedding-3-small', openai_api_key='test_api_key')

class TestOpenAIEncoder:

    def test_openai_encoder_init_success(self, mocker):
        side_effect = ['fake-model-name', 'fake-api-key', 'fake-org-id']
        mocker.patch('os.getenv', side_effect=side_effect)
        encoder = OpenAIEncoder()
        assert encoder._client is not None

    def test_openai_encoder_init_no_api_key(self, mocker):
        mocker.patch('os.getenv', return_value=None)
        with pytest.raises(ValueError) as _:
            OpenAIEncoder()

    def test_openai_encoder_call_uninitialized_client(self, openai_encoder):
        openai_encoder._client = None
        with pytest.raises(ValueError) as e:
            openai_encoder(['test document'])
        assert 'OpenAI client is not initialized.' in str(e.value)

    def test_openai_encoder_init_exception(self, mocker):
        mocker.patch('os.getenv', return_value='fake-api-key')
        mocker.patch('openai.Client', side_effect=Exception('Initialization error'))
        with pytest.raises(ValueError) as e:
            OpenAIEncoder()
        assert 'OpenAI API client failed to initialize. Error: Initialization error' in str(e.value)

    def test_openai_encoder_call_success(self, openai_encoder, mocker):
        mock_embeddings = mocker.Mock()
        mock_embeddings.data = [Embedding(embedding=[0.1, 0.2], index=0, object='embedding')]
        mocker.patch('os.getenv', return_value='fake-api-key')
        mocker.patch('time.sleep', return_value=None)
        mock_embedding = Embedding(index=0, object='embedding', embedding=[0.1, 0.2])
        mock_response = CreateEmbeddingResponse(model='text-embedding-ada-002', object='list', usage=Usage(prompt_tokens=0, total_tokens=20), data=[mock_embedding])
        responses = [OpenAIError('OpenAI error'), mock_response]
        mocker.patch.object(openai_encoder._client.embeddings, 'create', side_effect=responses)
        with patch('semantic_router.encoders.openai.sleep', return_value=None):
            embeddings = openai_encoder(['test document'])
        assert embeddings == [[0.1, 0.2]]

    def test_openai_encoder_call_failure_non_openai_error(self, openai_encoder, mocker):
        mocker.patch('os.getenv', return_value='fake-api-key')
        mocker.patch('time.sleep', return_value=None)
        mocker.patch.object(openai_encoder._client.embeddings, 'create', side_effect=Exception('Non-OpenAIError'))
        with patch('semantic_router.encoders.openai.sleep', return_value=None):
            with pytest.raises(ValueError) as e:
                openai_encoder(['test document'])
        assert 'OpenAI API call failed. Error: Non-OpenAIError' in str(e.value)

    def test_openai_encoder_call_successful_retry(self, openai_encoder, mocker):
        mock_embeddings = mocker.Mock()
        mock_embeddings.data = [Embedding(embedding=[0.1, 0.2], index=0, object='embedding')]
        mocker.patch('os.getenv', return_value='fake-api-key')
        mocker.patch('time.sleep', return_value=None)
        mock_embedding = Embedding(index=0, object='embedding', embedding=[0.1, 0.2])
        mock_response = CreateEmbeddingResponse(model='text-embedding-ada-002', object='list', usage=Usage(prompt_tokens=0, total_tokens=20), data=[mock_embedding])
        responses = [OpenAIError('OpenAI error'), mock_response]
        mocker.patch.object(openai_encoder._client.embeddings, 'create', side_effect=responses)
        with patch('semantic_router.encoders.openai.sleep', return_value=None):
            embeddings = openai_encoder(['test document'])
        assert embeddings == [[0.1, 0.2]]

    def test_retry_logic_sync(self, openai_encoder, mock_openai_client, mocker):
        mock_create = Mock(side_effect=[OpenAIError('API error'), OpenAIError('API error'), CreateEmbeddingResponse(data=[Embedding(embedding=[0.1, 0.2, 0.3], index=0, object='embedding')], model='text-embedding-3-small', object='list', usage={'prompt_tokens': 5, 'total_tokens': 5})])
        mock_openai_client.return_value.embeddings.create = mock_create
        mocker.patch('time.sleep', return_value=None)
        with patch('semantic_router.encoders.openai.sleep', return_value=None):
            result = openai_encoder(['test document'])
        assert result == [[0.1, 0.2, 0.3]]
        assert mock_create.call_count == 3

    def test_no_retry_on_max_retries_zero(self, openai_encoder, mock_openai_client):
        openai_encoder.max_retries = 0
        mock_create = Mock(side_effect=OpenAIError('API error'))
        mock_openai_client.return_value.embeddings.create = mock_create
        with pytest.raises(OpenAIError):
            openai_encoder(['test document'])
        assert mock_create.call_count == 1

    def test_retry_logic_sync_max_retries_exceeded(self, openai_encoder, mock_openai_client, mocker):
        mock_create = Mock(side_effect=OpenAIError('API error'))
        mock_openai_client.return_value.embeddings.create = mock_create
        mocker.patch('time.sleep', return_value=None)
        with patch('semantic_router.encoders.openai.sleep', return_value=None):
            with pytest.raises(OpenAIError):
                openai_encoder(['test document'])
        assert mock_create.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_logic_async(self, openai_encoder, mock_openai_async_client, mocker):
        mock_create = AsyncMock(side_effect=[OpenAIError('API error'), OpenAIError('API error'), CreateEmbeddingResponse(data=[Embedding(embedding=[0.1, 0.2, 0.3], index=0, object='embedding')], model='text-embedding-3-small', object='list', usage={'prompt_tokens': 5, 'total_tokens': 5})])
        mock_openai_async_client.return_value.embeddings.create = mock_create
        mocker.patch('asyncio.sleep', return_value=None)
        with patch('semantic_router.encoders.openai.asleep', return_value=None):
            result = await openai_encoder.acall(['test document'])
        assert result == [[0.1, 0.2, 0.3]]
        assert mock_create.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_logic_async_max_retries_exceeded(self, openai_encoder, mock_openai_async_client, mocker):

        async def raise_error(*args, **kwargs):
            raise OpenAIError('API error')
        mock_create = Mock(side_effect=raise_error)
        mock_openai_async_client.return_value.embeddings.create = mock_create
        mocker.patch('asyncio.sleep', return_value=None)
        with patch('semantic_router.encoders.openai.asleep', return_value=None):
            with pytest.raises(OpenAIError):
                await openai_encoder.acall(['test document'])
        assert mock_create.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_max_retries_zero_async(self, openai_encoder, mock_openai_async_client):
        openai_encoder.max_retries = 0

        async def raise_error(*args, **kwargs):
            raise OpenAIError('API error')
        mock_create = AsyncMock(side_effect=raise_error)
        mock_openai_async_client.return_value.embeddings.create = mock_create
        with pytest.raises(OpenAIError):
            await openai_encoder.acall(['test document'])
        assert mock_create.call_count == 1

class TestAzureOpenAIEncoder:

    def test_openai_encoder_init_success(self, mocker):
        mocker.patch('os.getenv', return_value='fake-api-key')
        encoder = AzureOpenAIEncoder()
        assert encoder.client is not None

    def test_openai_encoder_init_no_api_key(self, mocker):
        mocker.patch('os.getenv', return_value=None)
        with pytest.raises(ValueError) as _:
            AzureOpenAIEncoder()

    def test_openai_encoder_call_uninitialized_client(self, openai_encoder):
        openai_encoder.client = None
        with pytest.raises(ValueError) as e:
            openai_encoder(['test document'])
        assert 'OpenAI client is not initialized.' in str(e.value)

    def test_openai_encoder_init_exception(self, mocker):
        mocker.patch('os.getenv', return_value='fake-api-stuff')
        mocker.patch('openai.AzureOpenAI', side_effect=Exception('Initialization error'))
        with pytest.raises(ValueError) as e:
            AzureOpenAIEncoder()
        assert 'OpenAI API client failed to initialize. Error: Initialization error' in str(e.value)

    def test_openai_encoder_call_success(self, openai_encoder, mocker):
        mock_embeddings = mocker.Mock()
        mock_embeddings.data = [Embedding(embedding=[0.1, 0.2], index=0, object='embedding')]
        mocker.patch('os.getenv', return_value='fake-api-key')
        mocker.patch('time.sleep', return_value=None)
        mock_embedding = Embedding(index=0, object='embedding', embedding=[0.1, 0.2])
        mock_response = CreateEmbeddingResponse(model='text-embedding-ada-002', object='list', usage=Usage(prompt_tokens=0, total_tokens=20), data=[mock_embedding])
        responses = [OpenAIError('OpenAI error'), mock_response]
        mocker.patch.object(openai_encoder.client.embeddings, 'create', side_effect=responses)
        with patch('semantic_router.encoders.azure_openai.sleep', return_value=None):
            embeddings = openai_encoder(['test document'])
        assert embeddings == [[0.1, 0.2]]

    def test_openai_encoder_call_failure_non_openai_error(self, openai_encoder, mocker):
        mocker.patch('os.getenv', return_value='fake-api-key')
        mocker.patch('time.sleep', return_value=None)
        mocker.patch.object(openai_encoder.client.embeddings, 'create', side_effect=Exception('Non-OpenAIError'))
        with patch('semantic_router.encoders.azure_openai.sleep', return_value=None):
            with pytest.raises(ValueError) as e:
                openai_encoder(['test document'])
        assert 'OpenAI API call failed. Error: Non-OpenAIError' in str(e.value)

    def test_openai_encoder_call_successful_retry(self, openai_encoder, mocker):
        mock_embeddings = mocker.Mock()
        mock_embeddings.data = [Embedding(embedding=[0.1, 0.2], index=0, object='embedding')]
        mocker.patch('os.getenv', return_value='fake-api-key')
        mocker.patch('time.sleep', return_value=None)
        mock_embedding = Embedding(index=0, object='embedding', embedding=[0.1, 0.2])
        mock_response = CreateEmbeddingResponse(model='text-embedding-ada-002', object='list', usage=Usage(prompt_tokens=0, total_tokens=20), data=[mock_embedding])
        responses = [OpenAIError('OpenAI error'), mock_response]
        mocker.patch.object(openai_encoder.client.embeddings, 'create', side_effect=responses)
        with patch('semantic_router.encoders.azure_openai.sleep', return_value=None):
            embeddings = openai_encoder(['test document'])
        assert embeddings == [[0.1, 0.2]]

    def test_retry_logic_sync(self, openai_encoder, mock_openai_client, mocker):
        mock_create = Mock(side_effect=[OpenAIError('API error'), OpenAIError('API error'), CreateEmbeddingResponse(data=[Embedding(embedding=[0.1, 0.2, 0.3], index=0, object='embedding')], model='text-embedding-3-small', object='list', usage={'prompt_tokens': 5, 'total_tokens': 5})])
        mock_openai_client.return_value.embeddings.create = mock_create
        mocker.patch('time.sleep', return_value=None)
        with patch('semantic_router.encoders.azure_openai.sleep', return_value=None):
            result = openai_encoder(['test document'])
        assert result == [[0.1, 0.2, 0.3]]
        assert mock_create.call_count == 3

    def test_no_retry_on_max_retries_zero(self, openai_encoder, mock_openai_client):
        openai_encoder.max_retries = 0
        mock_create = Mock(side_effect=OpenAIError('API error'))
        mock_openai_client.return_value.embeddings.create = mock_create
        with pytest.raises(OpenAIError):
            openai_encoder(['test document'])
        assert mock_create.call_count == 1

    def test_retry_logic_sync_max_retries_exceeded(self, openai_encoder, mock_openai_client, mocker):
        mock_create = Mock(side_effect=OpenAIError('API error'))
        mock_openai_client.return_value.embeddings.create = mock_create
        mocker.patch('time.sleep', return_value=None)
        with patch('semantic_router.encoders.azure_openai.sleep', return_value=None):
            with pytest.raises(OpenAIError):
                openai_encoder(['test document'])
        assert mock_create.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_logic_async(self, openai_encoder, mock_openai_async_client, mocker):
        mock_create = AsyncMock(side_effect=[OpenAIError('API error'), OpenAIError('API error'), CreateEmbeddingResponse(data=[Embedding(embedding=[0.1, 0.2, 0.3], index=0, object='embedding')], model='text-embedding-3-small', object='list', usage={'prompt_tokens': 5, 'total_tokens': 5})])
        mock_openai_async_client.return_value.embeddings.create = mock_create
        mocker.patch('asyncio.sleep', return_value=None)
        with patch('semantic_router.encoders.azure_openai.asleep', return_value=None):
            result = await openai_encoder.acall(['test document'])
        assert result == [[0.1, 0.2, 0.3]]
        assert mock_create.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_logic_async_max_retries_exceeded(self, openai_encoder, mock_openai_async_client, mocker):

        async def raise_error(*args, **kwargs):
            raise OpenAIError('API error')
        mock_create = Mock(side_effect=raise_error)
        mock_openai_async_client.return_value.embeddings.create = mock_create
        mocker.patch('asyncio.sleep', return_value=None)
        with patch('semantic_router.encoders.azure_openai.asleep', return_value=None):
            with pytest.raises(OpenAIError):
                await openai_encoder.acall(['test document'])
        assert mock_create.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_max_retries_zero_async(self, openai_encoder, mock_openai_async_client):
        openai_encoder.max_retries = 0

        async def raise_error(*args, **kwargs):
            raise OpenAIError('API error')
        mock_create = AsyncMock(side_effect=raise_error)
        mock_openai_async_client.return_value.embeddings.create = mock_create
        with pytest.raises(OpenAIError):
            await openai_encoder.acall(['test document'])
        assert mock_create.call_count == 1

def has_valid_openai_api_key():
    """Check if a valid OpenAI API key is available."""
    api_key = os.environ.get('OPENAI_API_KEY')
    return api_key is not None and api_key.strip() != ''

class TestOpenAIEncoder:

    @pytest.mark.skipif(not has_valid_openai_api_key(), reason='OpenAI API key required')
    def test_openai_encoder_init_success(self, openai_encoder):
        assert openai_encoder._client is not None

    @pytest.mark.skipif(not has_valid_openai_api_key(), reason='OpenAI API key required')
    def test_openai_encoder_dims(self, openai_encoder):
        embeddings = openai_encoder(['test document'])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536

    @pytest.mark.skipif(not has_valid_openai_api_key(), reason='OpenAI API key required')
    def test_openai_encoder_call_truncation(self, openai_encoder):
        openai_encoder([long_doc])

    @pytest.mark.skipif(not has_valid_openai_api_key(), reason='OpenAI API key required')
    def test_openai_encoder_call_no_truncation(self, openai_encoder):
        with pytest.raises(OpenAIError) as _:
            openai_encoder([long_doc], truncate=False)

    @pytest.mark.skipif(not has_valid_openai_api_key(), reason='OpenAI API key required')
    def test_openai_encoder_call_uninitialized_client(self, openai_encoder):
        openai_encoder._client = None
        with pytest.raises(ValueError) as e:
            openai_encoder(['test document'])
        assert 'OpenAI client is not initialized.' in str(e.value)

