# Cluster 78

def get_logit_bias(model_name: str) -> Dict[str, float]:
    """Get the appropriate logit bias dictionary for the given model."""
    family = get_model_family(model_name)
    return MODEL_LOGIT_BIASES.get(family, {})

def get_model_family(model_name: str) -> ModelFamily:
    """Determine the model family from the model name."""
    model_name = model_name.lower()
    if any((name in model_name for name in ['gpt-', 'ft:gpt-'])):
        return ModelFamily.GPT
    elif 'llama' in model_name:
        return ModelFamily.LLAMA
    elif 'gemini' in model_name:
        return ModelFamily.GEMINI
    elif 'claude' in model_name:
        return ModelFamily.CLAUDE
    elif 'qwen' in model_name:
        return ModelFamily.QWEN
    elif 'deepseek' in model_name:
        return ModelFamily.DEEPSEEK
    return ModelFamily.UNKNOWN

@dataclass
class ParallelBeamConfig:
    """Configuration for parallel beam search"""
    beam_width: int
    expansion_factor: int
    system_prompt: str
    initial_state: GameState
    beam_kwargs: Dict[str, Any]
    model: Optional[str] = None

    def __post_init__(self):
        if self.model:
            if self.beam_kwargs is None:
                self.beam_kwargs = {}
            logit_bias = get_logit_bias(self.model)
            if logit_bias:
                current_bias = self.beam_kwargs.get('logit_bias', {})
                current_bias.update(logit_bias)
                self.beam_kwargs['logit_bias'] = current_bias

def get_logit_bias(model_name: str) -> Dict[str, float]:
    """Get the appropriate logit bias dictionary for the given model."""
    family = get_model_family(model_name)
    return MODEL_LOGIT_BIASES.get(family, {})

@dataclass
class ChunkedConfig(BaseConfig):
    max_conversation_length: int = 50
    _logit_bias: Dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self.model:
            self._logit_bias = get_logit_bias(self.model)

    @property
    def logit_bias(self) -> Dict[str, float]:
        return self._logit_bias

