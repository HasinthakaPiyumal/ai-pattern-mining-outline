# Cluster 4

def get_prompt_for_model(model_name: str) -> PromptTemplate:
    if model_name.startswith('gpt-') or model_name.startswith('text-'):
        return OPENAI_PROMPT
    elif model_name.startswith('gemini-'):
        return GEMINI_PROMPT
    elif model_name.startswith('ollama:'):
        return OLLAMA_PROMPT
    else:
        raise ValueError(f'Unsupported model: {model_name}')

