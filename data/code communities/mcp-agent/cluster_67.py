# Cluster 67

def _select_provider_and_model(*, model: str | ModelPreferences | None=None, provider: SupportedLLMProviders | None=None, context: Context | None=None) -> Tuple[str, str | None]:
    """
    Return (provider, model_name) using a string model id or ModelSelector.

    - If model is a str, treat it as model id; allow 'provider:model' pattern.
    - If it's a ModelPreferences, use ModelSelector.
    - Otherwise, return default provider and no model.
    """
    prov = (provider or 'openai').lower()
    if isinstance(model, str):
        inferred_provider, model_name = _parse_model_identifier(model)
        return (inferred_provider or prov, model_name)
    if isinstance(model, ModelPreferences):
        selector = ModelSelector(context=context)
        model_info = selector.select_best_model(model_preferences=model, provider=prov)
        return (model_info.provider.lower(), model_info.name)
    return (prov, None)

def _parse_model_identifier(model_id: str) -> Tuple[str | None, str]:
    """Parse a model identifier that may be prefixed with provider (e.g., 'openai:gpt-4o')."""
    if ':' in model_id:
        prov, name = model_id.split(':', 1)
        return (prov.strip().lower() or None, name.strip())
    return (None, model_id)

def _merge_model_preferences(provider: str | None=None, model: str | ModelPreferences | None=None, request_params: RequestParams | None=None, context: Context | None=None) -> RequestParams:
    """
    Merge model preferences from provider, model, and request params.
    Explicitly specified model takes precedence over request_params.
    """
    _, model_name = _select_provider_and_model(provider=provider, model=model or getattr(request_params, 'model', None), context=context)
    if request_params is not None:
        if model_name and isinstance(model, ModelPreferences):
            request_params.model = model_name
            request_params.modelPreferences = model
        elif model_name and isinstance(model, str):
            request_params.model = model_name
        elif isinstance(model, ModelPreferences):
            request_params.modelPreferences = model
    else:
        request_params = RequestParams(model=model_name)
        if isinstance(model, ModelPreferences):
            request_params.modelPreferences = model
    return request_params

def _llm_factory(*, provider: SupportedLLMProviders | None=None, model: str | ModelPreferences | None=None, request_params: RequestParams | None=None, context: Context | None=None) -> Callable[[Agent], AugmentedLLM]:
    model_selector_input = model or getattr(request_params, 'model', None) or getattr(request_params, 'modelPreferences', None)
    prov, model_name = _select_provider_and_model(provider=provider, model=model_selector_input, context=context)
    provider_cls = _get_provider_class(prov)

    def _default_params() -> RequestParams | None:
        if model_name and isinstance(model, ModelPreferences):
            return RequestParams(model=model_name, modelPreferences=model)
        if model_name and isinstance(model, str):
            return RequestParams(model=model_name)
        if isinstance(model, ModelPreferences):
            return RequestParams(modelPreferences=model)
        return None
    effective_params: RequestParams | None = request_params
    if effective_params is not None:
        chosen_model: str | None = model_name
        if not chosen_model:
            cfg_obj = None
            try:
                cfg_obj = provider_cls.get_provider_config(context)
            except Exception:
                cfg_obj = None
            if cfg_obj is not None:
                chosen_model = getattr(cfg_obj, 'default_model', None)
        if getattr(effective_params, 'model', None) is None and chosen_model:
            effective_params.model = chosen_model
    return lambda agent: provider_cls(agent=agent, default_request_params=effective_params or _default_params(), context=context)

def _get_provider_class(provider: SupportedLLMProviders):
    p = provider.lower()
    if p == 'openai':
        from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
        return OpenAIAugmentedLLM
    if p == 'anthropic':
        from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
        return AnthropicAugmentedLLM
    if p == 'azure':
        from mcp_agent.workflows.llm.augmented_llm_azure import AzureAugmentedLLM
        return AzureAugmentedLLM
    if p == 'google':
        from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
        return GoogleAugmentedLLM
    if p == 'bedrock':
        from mcp_agent.workflows.llm.augmented_llm_bedrock import BedrockAugmentedLLM
        return BedrockAugmentedLLM
    if p == 'ollama':
        from mcp_agent.workflows.llm.augmented_llm_ollama import OllamaAugmentedLLM
        return OllamaAugmentedLLM
    raise ValueError(f"mcp-agent doesn't support provider: {provider}. To request support, please create an issue at https://github.com/lastmile-ai/mcp-agent/issues")

def _default_params() -> RequestParams | None:
    if model_name and isinstance(model, ModelPreferences):
        return RequestParams(model=model_name, modelPreferences=model)
    if model_name and isinstance(model, str):
        return RequestParams(model=model_name)
    if isinstance(model, ModelPreferences):
        return RequestParams(modelPreferences=model)
    return None

