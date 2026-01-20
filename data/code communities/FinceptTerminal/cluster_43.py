# Cluster 43

def call_llm_openrouter(prompt, pydantic_model, agent_name, state, default_factory, model='anthropic/claude-3.5-sonnet'):
    """Modified LLM call for OpenRouter"""
    client = setup_openrouter_client()
    try:
        response = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': prompt.messages[0].content}, {'role': 'user', 'content': prompt.messages[1].content}], temperature=0.1, max_tokens=2000)
        import json
        content = response.choices[0].message.content
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        json_str = content[start_idx:end_idx]
        parsed_data = json.loads(json_str)
        return pydantic_model(**parsed_data)
    except Exception as e:
        print(f'OpenRouter API error: {e}')
        return default_factory()

def setup_openrouter_client():
    """Setup OpenRouter client"""
    return openai.OpenAI(base_url='https://openrouter.ai/api/v1', api_key='YOUR_OPENROUTER_API_KEY')

class HedgeFundLLMAdapter:
    """Adapter to use different LLM providers with hedge fund agents"""

    def __init__(self, provider='openai', **kwargs):
        self.provider = provider
        self.config = kwargs

    def call_llm(self, prompt, pydantic_model, agent_name, state, default_factory):
        """Route to appropriate LLM provider"""
        if self.provider == 'openrouter':
            return call_llm_openrouter(prompt, pydantic_model, agent_name, state, default_factory, model=self.config.get('model', 'anthropic/claude-3.5-sonnet'))
        elif self.provider == 'ollama':
            return call_llm_ollama(prompt, pydantic_model, agent_name, state, default_factory, model=self.config.get('model', 'llama3.1:70b'))
        elif self.provider == 'openai':
            return self.call_openai_llm(prompt, pydantic_model, agent_name, state, default_factory)
        else:
            raise ValueError(f'Unsupported provider: {self.provider}')

    def call_openai_llm(self, prompt, pydantic_model, agent_name, state, default_factory):
        """OpenAI integration"""
        import openai
        client = openai.OpenAI(api_key=self.config.get('api_key'))
        try:
            response = client.chat.completions.create(model=self.config.get('model', 'gpt-4'), messages=[{'role': 'system', 'content': prompt.messages[0].content}, {'role': 'user', 'content': prompt.messages[1].content}], temperature=0.1, max_tokens=2000)
            content = response.choices[0].message.content
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            parsed_data = json.loads(json_str)
            return pydantic_model(**parsed_data)
        except Exception as e:
            print(f'OpenAI API error: {e}')
            return default_factory()

def call_llm_ollama(prompt, pydantic_model, agent_name, state, default_factory, model='llama3.1:70b'):
    """Modified LLM call for Ollama local"""
    full_prompt = f'System: {prompt.messages[0].content}\n\nHuman: {prompt.messages[1].content}\n\nAssistant:'
    try:
        response = requests.post('http://localhost:11434/api/generate', json={'model': model, 'prompt': full_prompt, 'stream': False, 'options': {'temperature': 0.1, 'top_p': 0.9}}, timeout=120)
        if response.status_code == 200:
            content = response.json()['response']
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                parsed_data = json.loads(json_str)
                return pydantic_model(**parsed_data)
            else:
                raise ValueError('No valid JSON found in response')
        else:
            raise Exception(f'HTTP {response.status_code}: {response.text}')
    except Exception as e:
        print(f'Ollama API error: {e}')
        return default_factory()

