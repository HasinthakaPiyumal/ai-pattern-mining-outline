# Cluster 52

def register_model(config_cls, alias: List[str]=None):

    def decorator(cls):
        class_name = cls.__name__
        MODEL_REGISTRY.register(class_name, cls, config_cls)
        if alias is not None:
            for alia in alias:
                MODEL_REGISTRY.register(alia, cls, config_cls)
        return cls
    return decorator

@register_model(config_cls=LiteLLMConfig, alias=['litellm'])
class LiteLLM(OpenAILLM):

    def init_model(self):
        """
        Initialize the model based on the configuration.
        """
        if self.config.llm_type != 'LiteLLM':
            raise ValueError("llm_type must be 'LiteLLM'")
        self.model = self.config.model
        self.api_base = self.config.api_base
        self.api_key = self.config.api_key
        company = infer_litellm_company_from_model(self.model)
        if self.config.is_local or company == 'local':
            if not self.api_base:
                raise ValueError('api_base is required for local models in LiteLLMConfig')
            litellm.api_base = self.api_base
            litellm.api_key = self.api_key
        elif company == 'openai':
            if not self.config.openai_key:
                raise ValueError('OpenAI API key is required for OpenAI models. You should set `openai_key` in LiteLLMConfig')
            os.environ['OPENAI_API_KEY'] = self.config.openai_key
        elif company == 'azure':
            if not self.config.azure_key or not self.config.azure_endpoint:
                raise ValueError('Azure OpenAI key and endpoint are required for Azure models. You should set `azure_key` and `azure_endpoint` in LiteLLMConfig')
            os.environ['AZURE_API_KEY'] = self.config.azure_key
            os.environ['AZURE_API_BASE'] = self.config.azure_endpoint
            if self.config.api_version:
                os.environ['AZURE_API_VERSION'] = self.config.api_version
        elif company == 'deepseek':
            if not self.config.deepseek_key:
                raise ValueError('DeepSeek API key is required for DeepSeek models. You should set `deepseek_key` in LiteLLMConfig')
            os.environ['DEEPSEEK_API_KEY'] = self.config.deepseek_key
        elif company == 'anthropic':
            if not self.config.anthropic_key:
                raise ValueError('Anthropic API key is required for Anthropic models. You should set `anthropic_key` in LiteLLMConfig')
            os.environ['ANTHROPIC_API_KEY'] = self.config.anthropic_key
        elif company == 'gemini':
            if not self.config.gemini_key:
                raise ValueError('Gemini API key is required for Gemini models. You should set `gemini_key` in LiteLLMConfig')
            os.environ['GEMINI_API_KEY'] = self.config.gemini_key
        elif company == 'meta_llama':
            if not self.config.meta_llama_key:
                raise ValueError('Meta Llama API key is required for Meta Llama models. You should set `meta_llama_key` in LiteLLMConfig')
            os.environ['LLAMA_API_KEY'] = self.config.meta_llama_key
        elif company == 'openrouter':
            if not self.config.openrouter_key:
                raise ValueError('OpenRouter API key is required for OpenRouter models. You should set `openrouter_key` in LiteLLMConfig. You can also set `openrouter_base` in LiteLLMConfig to use a custom base URL [optional]')
            os.environ['OPENROUTER_API_KEY'] = self.config.openrouter_key
            os.environ['OPENROUTER_API_BASE'] = self.config.openrouter_base
        elif company == 'perplexity':
            if not self.config.perplexity_key:
                raise ValueError('Perplexity API key is required for Perplexity models. You should set `perplexity_key` in LiteLLMConfig')
            os.environ['PERPLEXITYAI_API_KEY'] = self.config.perplexity_key
        elif company == 'groq':
            if not self.config.groq_key:
                raise ValueError('Groq API key is required for Groq models. You should set `groq_key` in LiteLLMConfig')
            os.environ['GROQ_API_KEY'] = self.config.groq_key
        else:
            raise ValueError(f'Unsupported company: {company}')
        self._default_ignore_fields = ['llm_type', 'output_response', 'openai_key', 'deepseek_key', 'anthropic_key', 'gemini_key', 'meta_llama_key', 'openrouter_key', 'openrouter_base', 'perplexity_key', 'groq_key', 'api_base', 'is_local', 'azure_endpoint', 'azure_key', 'api_version', 'api_key']

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> Cost:
        if self.config.is_local:
            return Cost(input_tokens=input_tokens, output_tokens=output_tokens, input_cost=0.0, output_cost=0.0)
        return super()._compute_cost(input_tokens, output_tokens)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def single_generate(self, messages: List[dict], **kwargs) -> str:
        """
        Generate a single response using the completion function.

        Args: 
            messages (List[dict]): A list of dictionaries representing the conversation history.
            **kwargs (Any): Additional parameters to be passed to the `completion` function.
        
        Returns: 
            str: A string containing the model's response.
        """
        stream = kwargs['stream'] if 'stream' in kwargs else self.config.stream
        output_response = kwargs['output_response'] if 'output_response' in kwargs else self.config.output_response
        try:
            completion_params = self.get_completion_params(**kwargs)
            company = infer_litellm_company_from_model(self.model)
            if self.config.is_local or company == 'local':
                completion_params['api_base'] = self.api_base
            elif company == 'azure':
                completion_params['api_base'] = self.config.azure_endpoint
                completion_params['api_version'] = self.config.api_version
                completion_params['api_key'] = self.config.azure_key
            response = completion(messages=messages, **completion_params)
            if stream:
                output = self.get_stream_output(response, output_response=output_response)
                cost = self._stream_cost(messages=messages, output=output)
            else:
                output: str = self.get_completion_output(response=response, output_response=output_response)
                cost = self._completion_cost(response=response)
            self._update_cost(cost=cost)
        except Exception as e:
            raise RuntimeError(f'Error during single_generate: {str(e)}')
        return output

    def batch_generate(self, batch_messages: List[List[dict]], **kwargs) -> List[str]:
        """
        Generate responses for a batch of messages.

        Args: 
            batch_messages (List[List[dict]]): A list of message lists, where each sublist represents a conversation.
            **kwargs (Any): Additional parameters to be passed to the `completion` function.
        
        Returns: 
            List[str]: A list of responses for each conversation.
        """
        results = []
        for messages in batch_messages:
            response = self.single_generate(messages, **kwargs)
            results.append(response)
        return results

    async def single_generate_async(self, messages: List[dict], **kwargs) -> str:
        """
        Generate a single response using the async completion function.

        Args: 
            messages (List[dict]): A list of dictionaries representing the conversation history.
            **kwargs (Any): Additional parameters to be passed to the `completion` function.
        
        Returns: 
            str: A string containing the model's response.
        """
        stream = kwargs['stream'] if 'stream' in kwargs else self.config.stream
        output_response = kwargs['output_response'] if 'output_response' in kwargs else self.config.output_response
        try:
            completion_params = self.get_completion_params(**kwargs)
            company = infer_litellm_company_from_model(self.model)
            if self.config.is_local or company == 'local':
                completion_params['api_base'] = self.api_base
            elif company == 'azure':
                completion_params['api_base'] = self.config.azure_endpoint
                completion_params['api_version'] = self.config.api_version
                completion_params['api_key'] = self.config.azure_key
            response = await acompletion(messages=messages, **completion_params)
            if stream:
                if hasattr(response, '__aiter__'):
                    output = await self.get_stream_output_async(response, output_response=output_response)
                else:
                    output = self.get_stream_output(response, output_response=output_response)
                cost = self._stream_cost(messages=messages, output=output)
            else:
                output: str = self.get_completion_output(response=response, output_response=output_response)
                cost = self._completion_cost(response=response)
            self._update_cost(cost=cost)
        except Exception as e:
            raise RuntimeError(f'Error during single_generate_async: {str(e)}')
        return output

def infer_litellm_company_from_model(model: str) -> str:
    if '/' in model:
        company = model.split('/')[0]
    elif 'claude' in model or 'anthropic' in model:
        company = 'anthropic'
    elif 'gemini' in model:
        company = 'gemini'
    elif 'deepseek' in model:
        company = 'deepseek'
    elif 'openrouter' in model:
        company = 'openrouter'
    elif 'azure' in model.lower():
        company = 'azure'
    else:
        company = 'openai'
    return company

