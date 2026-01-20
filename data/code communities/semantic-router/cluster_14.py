# Cluster 14

class NimEncoder(LiteLLMEncoder):
    """Class to encode text using Nvidia NIM. Requires a Nim API key from
    https://nim.ai/api-keys/"""
    type: str = 'nvidia_nim'

    def __init__(self, name: str | None=None, api_key: str | None=None, score_threshold: float=0.4):
        """Initialize the NimEncoder.

        :param name: The name of the embedding model to use such as "nv-embedqa-e5-v5".
        :type name: str
        :param nim_api_key: The Nim API key.
        :type nim_api_key: str
        """
        if name is None:
            name = f'nvidia_nim/{EncoderDefault.NVIDIA_NIM.value['embedding_model']}'
        elif not name.startswith('nvidia_nim/'):
            name = f'nvidia_nim/{name}'
        super().__init__(name=name, score_threshold=score_threshold, api_key=api_key)

    def encode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(input=docs, input_type='passage', model=f'{self.type}/{self.name}', **kwargs)
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f'Nim API call failed. Error: {e}') from e

    def encode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(input=docs, input_type='passage', model=f'{self.type}/{self.name}', **kwargs)
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f'Nim API call failed. Error: {e}') from e

    async def aencode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(input=docs, input_type='passage', model=f'{self.type}/{self.name}', **kwargs)
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f'Nim API call failed. Error: {e}') from e

    async def aencode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(input=docs, input_type='passage', model=f'{self.type}/{self.name}', **kwargs)
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f'Nim API call failed. Error: {e}') from e

def litellm_to_list(embeds: litellm.EmbeddingResponse) -> list[list[float]]:
    """Convert a LiteLLM embedding response to a list of embeddings.

    :param embeds: The LiteLLM embedding response.
    :return: A list of embeddings.
    """
    if not embeds or not isinstance(embeds, litellm.EmbeddingResponse) or (not embeds.data):
        raise ValueError('No embeddings found in LiteLLM embedding response.')
    return [x['embedding'] for x in embeds.data]

class VoyageEncoder(LiteLLMEncoder):
    """Class to encode text using Voyage. Requires a Voyage API key from
    https://voyageai.com/api-keys/"""
    type: str = 'voyage'

    def __init__(self, name: str | None=None, api_key: str | None=None, score_threshold: float=0.4):
        """Initialize the VoyageEncoder.

        :param name: The name of the embedding model to use such as "voyage-embed".
        :type name: str
        :param voyage_api_key: The Voyage API key.
        :type voyage_api_key: str
        """
        if name is None:
            name = f'voyage/{EncoderDefault.VOYAGE.value['embedding_model']}'
        elif not name.startswith('voyage/'):
            name = f'voyage/{name}'
        super().__init__(name=name, score_threshold=score_threshold, api_key=api_key)

    def encode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(input=docs, input_type='query', model=f'{self.type}/{self.name}', **kwargs)
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f'Voyage API call failed. Error: {e}') from e

    def encode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(input=docs, input_type='document', model=f'{self.type}/{self.name}', **kwargs)
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f'Voyage API call failed. Error: {e}') from e

    async def aencode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(input=docs, input_type='query', model=f'{self.type}/{self.name}', **kwargs)
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f'Voyage API call failed. Error: {e}') from e

    async def aencode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(input=docs, input_type='document', model=f'{self.type}/{self.name}', **kwargs)
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f'Voyage API call failed. Error: {e}') from e

class LiteLLMEncoder(DenseEncoder, AsymmetricDenseMixin):
    """LiteLLM encoder class for generating embeddings using LiteLLM.

    The LiteLLMEncoder class is a subclass of DenseEncoder and utilizes the LiteLLM SDK
    to generate embeddings for given documents. It supports all encoders supported by LiteLLM
    and supports customization of the score threshold for filtering or processing the embeddings.
    """
    type: str = 'litellm'

    def __init__(self, name: str | None=None, score_threshold: float | None=None, api_key: str | None=None):
        """Initialize the LiteLLMEncoder.

        :param name: The name of the embedding model to use. Must use LiteLLM naming
            convention (e.g. "openai/text-embedding-3-small" or "mistral/mistral-embed").
        :type name: str
        :param score_threshold: The score threshold for the embeddings.
        :type score_threshold: float
        """
        if name is None:
            name = 'openai/' + EncoderDefault.OPENAI.value['embedding_model']
        super().__init__(name=name, score_threshold=score_threshold if score_threshold is not None else 0.3)
        self.type, self.name = self.name.split('/', 1)
        if api_key is None:
            api_key = os.getenv(self.type.upper() + '_API_KEY')
        if api_key is None:
            raise ValueError('Expected API key via `api_key` parameter or `{self.type.upper()}_API_KEY` environment variable.')
        os.environ[self.type.upper() + '_API_KEY'] = api_key

    def __call__(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of text documents into embeddings using LiteLLM.

        :param docs: List of text documents to encode.
        :return: List of embeddings for each document."""
        return self.encode_queries(docs, **kwargs)

    async def acall(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of documents into embeddings using LiteLLM asynchronously.

        :param docs: List of documents to encode.
        :return: List of embeddings for each document."""
        return await self.aencode_queries(docs, **kwargs)

    def encode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(input=docs, model=f'{self.type}/{self.name}', **kwargs)
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f'{self.type.capitalize()} API call failed. Error: {e}') from e

    def encode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(input=docs, model=f'{self.type}/{self.name}', **kwargs)
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f'{self.type.capitalize()} API call failed. Error: {e}') from e

    async def aencode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(input=docs, model=f'{self.type}/{self.name}', **kwargs)
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f'{self.type.capitalize()} API call failed. Error: {e}') from e

    async def aencode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(input=docs, model=f'{self.type}/{self.name}', **kwargs)
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f'{self.type.capitalize()} API call failed. Error: {e}') from e

class CohereEncoder(LiteLLMEncoder):
    """Dense encoder that uses Cohere API to embed documents. Supports text only. Requires
    a Cohere API key from https://dashboard.cohere.com/api-keys.
    """
    _client: Any = PrivateAttr()
    _async_client: Any = PrivateAttr()
    _embed_type: Any = PrivateAttr()
    type: str = 'cohere'

    def __init__(self, name: str | None=None, cohere_api_key: str | None=None, score_threshold: float=0.3):
        """Initialize the Cohere encoder.

        :param name: The name of the embedding model to use such as "embed-english-v3.0" or
            "embed-multilingual-v3.0".
        :type name: str
        :param cohere_api_key: The API key for the Cohere client, can also
            be set via the COHERE_API_KEY environment variable.
        :type cohere_api_key: str
        :param score_threshold: The threshold for the score of the embedding.
        :type score_threshold: float
        """
        if name is None:
            name = f'cohere/{EncoderDefault.COHERE.value['embedding_model']}'
        elif not name.startswith('cohere/'):
            name = f'cohere/{name}'
        super().__init__(name=name, score_threshold=score_threshold, api_key=cohere_api_key)
        self._client = None
        self._async_client = None

    @deprecated('_initialize_client method no longer required')
    def _initialize_client(self, cohere_api_key: str | None=None):
        """Initializes the Cohere client.

        :param cohere_api_key: The API key for the Cohere client, can also
            be set via the COHERE_API_KEY environment variable.
        :type cohere_api_key: str
        :return: An instance of the Cohere client.
        :rtype: cohere.Client
        """
        cohere_api_key = cohere_api_key or os.getenv('COHERE_API_KEY')
        if cohere_api_key is None:
            raise ValueError("Cohere API key cannot be 'None'.")
        return (None, None)

    def encode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(input=docs, input_type='search_query', model=f'{self.type}/{self.name}', **kwargs)
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f'Cohere API call failed. Error: {e}') from e

    def encode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = litellm.embedding(input=docs, input_type='search_document', model=f'{self.type}/{self.name}', **kwargs)
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f'Cohere API call failed. Error: {e}') from e

    async def aencode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(input=docs, input_type='search_query', model=f'{self.type}/{self.name}', **kwargs)
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f'Cohere API call failed. Error: {e}') from e

    async def aencode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        try:
            embeds = await litellm.aembedding(input=docs, input_type='search_document', model=f'{self.type}/{self.name}', **kwargs)
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(f'Cohere API call failed. Error: {e}') from e

