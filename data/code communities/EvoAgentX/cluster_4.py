# Cluster 4

def default_llm_config():
    """
    Create default LLM configuration. Uses MockLLM in testing environments 
    or when OPENAI_API_KEY is not available.
    """
    is_testing = os.getenv('PYTEST_CURRENT_TEST') is not None or os.getenv('CI') is not None or OPENAI_API_KEY is None or (OPENAI_API_KEY.strip() == '')
    if is_testing:
        mock_config = MockLLMConfig(llm_type='MockLLM', model='mock-model', output_response=True)
        return MockLLM(mock_config)
    else:
        llm_config = OpenAILLMConfig(model='gpt-4o', openai_key=OPENAI_API_KEY, stream=True, output_response=True)
        return OpenAILLM(llm_config)

