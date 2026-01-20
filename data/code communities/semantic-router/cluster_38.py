# Cluster 38

class TestBedrockEncoderWithCohere:

    def test_cohere_embedding_single_chunk(self, bedrock_encoder_with_cohere):
        response_content = json.dumps({'embeddings': [[0.1, 0.2, 0.3]]})
        response_body = BytesIO(response_content.encode('utf-8'))
        mock_response = {'body': response_body}
        bedrock_encoder_with_cohere.client.invoke_model.return_value = mock_response
        result = bedrock_encoder_with_cohere(['short test'])
        assert isinstance(result, list), 'Result should be a list'
        assert all((isinstance(item, list) for item in result)), 'Each item should be a list'
        assert result == [[0.1, 0.2, 0.3]], 'Expected embedding [0.1, 0.2, 0.3]'

    def test_cohere_input_type(self, bedrock_encoder_with_cohere):
        bedrock_encoder_with_cohere.input_type = 'different_type'
        response_content = json.dumps({'embeddings': [[0.1, 0.2, 0.3]]})
        response_body = BytesIO(response_content.encode('utf-8'))
        mock_response = {'body': response_body}
        bedrock_encoder_with_cohere.client.invoke_model.return_value = mock_response
        result = bedrock_encoder_with_cohere(['test with different input type'])
        assert isinstance(result, list), 'Result should be a list'
        assert result == [[0.1, 0.2, 0.3]], 'Expected specific embeddings'

@pytest.fixture
def bedrock_encoder_with_cohere(mocker):
    mocker.patch('semantic_router.encoders.bedrock.BedrockEncoder._initialize_client')
    return BedrockEncoder(name='cohere_model', access_key_id='fake_id', secret_access_key='fake_secret', session_token='fake_token', region='us-west-2')

