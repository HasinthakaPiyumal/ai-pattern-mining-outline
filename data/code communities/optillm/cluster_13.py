# Cluster 13

def should_use_mlx(model_id: str) -> bool:
    """Determine if a model should use MLX instead of PyTorch"""
    if not MLX_AVAILABLE or not is_apple_silicon():
        return False
    mlx_patterns = ['mlx-community/', 'mlx-', '-mlx-']
    problematic_models = ['Qwen/Qwen3-', 'google/gemma-3-', 'google/gemma3-']
    model_lower = model_id.lower()
    for pattern in mlx_patterns:
        if pattern.lower() in model_lower:
            return True
    for pattern in problematic_models:
        if pattern.lower() in model_lower:
            logger.warning(f'Model {model_id} detected as potentially problematic with MPS backend')
            suggested_mlx = suggest_mlx_alternative(model_id)
            logger.warning(f'Consider using MLX model: {suggested_mlx}')
            return False
    return False

def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon"""
    return platform.system() == 'Darwin' and platform.machine() == 'arm64'

class MLXInferencePipeline:
    """MLX-based inference pipeline that mirrors PyTorch pipeline interface"""

    def __init__(self, model_config: MLXModelConfig, cache_manager):
        self.model_config = model_config
        self.cache_manager = cache_manager
        self.last_used = time.time()
        if not MLX_AVAILABLE:
            raise RuntimeError('MLX framework not available. Install with: pip install mlx-lm')
        if not is_apple_silicon():
            raise RuntimeError('MLX framework is only supported on Apple Silicon')
        try:
            logger.info(f'Loading MLX model: {model_config.model_id}')
            self.model, self.tokenizer = self._load_mlx_model(model_config.model_id)
            logger.info('MLX model loaded successfully')
        except Exception as e:
            logger.error(f'Failed to load MLX model: {str(e)}')
            raise

    def _load_mlx_model(self, model_id: str):
        """Load MLX model and tokenizer with caching"""

        def _load_model():
            start_time = time.time()
            logger.info(f'Loading MLX model: {model_id}')
            try:
                model, tokenizer = mlx_load(model_id)
                load_time = time.time() - start_time
                logger.info(f'MLX model loaded in {load_time:.2f}s')
                return (model, tokenizer)
            except Exception as e:
                logger.error(f'Error loading MLX model {model_id}: {str(e)}')
                raise
        return self.cache_manager.get_or_load_model(f'mlx_{model_id}', _load_model)

    def generate(self, prompt: str, generation_params: Optional[Dict[str, Any]]=None) -> Tuple[List[str], List[int], List[Optional[Dict]]]:
        """Generate text using MLX"""
        start_time = time.time()
        if generation_params is None:
            generation_params = {}
        max_tokens = generation_params.get('max_new_tokens', self.model_config.max_new_tokens)
        temperature = generation_params.get('temperature', self.model_config.temperature)
        top_p = generation_params.get('top_p', self.model_config.top_p)
        repetition_penalty = generation_params.get('repetition_penalty', self.model_config.repetition_penalty)
        num_return_sequences = generation_params.get('num_return_sequences', 1)
        if generation_params.get('seed') is not None:
            mx.random.seed(generation_params['seed'])
        responses = []
        token_counts = []
        logprobs_results = []
        for _ in range(num_return_sequences):
            try:
                logger.debug(f'Generating with MLX: max_tokens={max_tokens}, temp={temperature}')
                response = self._robust_mlx_generate(prompt, max_tokens, temperature, top_p, repetition_penalty)
                responses.append(response)
                if isinstance(response, str):
                    token_count = len(self.tokenizer.encode(response))
                else:
                    token_count = len(response) if hasattr(response, '__len__') else 0
                token_counts.append(token_count)
                logprobs_results.append(None)
            except Exception as e:
                logger.error(f'Error during MLX generation: {str(e)}')
                logger.error(f'MLX generation parameters: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}')
                responses.append('')
                token_counts.append(0)
                logprobs_results.append(None)
        generation_time = time.time() - start_time
        logger.info(f'MLX generation completed in {generation_time:.2f}s')
        return (responses, token_counts, logprobs_results)

    def _robust_mlx_generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float, repetition_penalty: float) -> str:
        """Robust MLX generation using sampler approach"""
        try:
            sampler = make_sampler(temp=temperature, top_p=top_p, min_p=0.0, min_tokens_to_keep=1)
            response = mlx_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)
            return response
        except Exception as e:
            logger.error(f'MLX generation with sampler failed: {str(e)}')
            try:
                logger.debug('Attempting MLX generation without sampler')
                response = mlx_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens, verbose=False)
                return response
            except Exception as fallback_e:
                logger.error(f'MLX fallback generation also failed: {str(fallback_e)}')
                raise

    def format_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format the prompt according to model's chat template"""
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                logger.warning(f'Failed to apply chat template: {e}, using fallback')
                return f'System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:'
        else:
            return f'System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:'

    def process_batch(self, system_prompts: List[str], user_prompts: List[str], generation_params: Optional[Dict[str, Any]]=None, active_adapter: str=None, return_token_count: bool=True) -> Tuple[List[str], List[int]]:
        """
        Process a batch of prompts with MLX-based batch inference
        
        This method provides true batch processing for MLX models, processing multiple
        prompts simultaneously for improved throughput.
        
        Args:
            system_prompts: List of system prompts
            user_prompts: List of user prompts
            generation_params: Generation parameters (temperature, max_tokens, etc.)
            active_adapter: Active adapter (not used in MLX)
            return_token_count: Whether to return token counts
            
        Returns:
            Tuple of (responses, token_counts)
        """
        import time
        if generation_params is None:
            generation_params = {}
        if len(system_prompts) != len(user_prompts):
            raise ValueError(f'Number of system prompts ({len(system_prompts)}) must match user prompts ({len(user_prompts)})')
        if not system_prompts:
            return ([], [])
        batch_size = len(system_prompts)
        logger.info(f'MLX batch processing {batch_size} prompts')
        start_time = time.time()
        formatted_prompts = [self.format_chat_prompt(system_prompt, user_prompt) for system_prompt, user_prompt in zip(system_prompts, user_prompts)]
        max_tokens = generation_params.get('max_new_tokens', self.model_config.max_new_tokens)
        temperature = generation_params.get('temperature', self.model_config.temperature)
        top_p = generation_params.get('top_p', self.model_config.top_p)
        repetition_penalty = generation_params.get('repetition_penalty', self.model_config.repetition_penalty)
        n = generation_params.get('num_return_sequences', 1)
        if generation_params.get('seed') is not None:
            mx.random.seed(generation_params['seed'])
        all_responses = []
        token_counts = []
        try:
            for i, prompt in enumerate(formatted_prompts):
                logger.debug(f'Processing MLX batch item {i + 1}/{batch_size}')
                for _ in range(n):
                    try:
                        response = self._robust_mlx_generate(prompt, max_tokens, temperature, top_p, repetition_penalty)
                        all_responses.append(response)
                        if isinstance(response, str):
                            token_count = len(self.tokenizer.encode(response))
                        else:
                            token_count = len(response) if hasattr(response, '__len__') else 0
                        token_counts.append(token_count)
                    except Exception as e:
                        logger.error(f'Error generating response for batch item {i + 1}: {e}')
                        all_responses.append('')
                        token_counts.append(0)
            processing_time = time.time() - start_time
            logger.info(f'MLX batch processing completed in {processing_time:.2f}s')
            if return_token_count:
                return (all_responses, token_counts)
            return (all_responses, [0] * len(all_responses))
        except Exception as e:
            logger.error(f'MLX batch processing failed: {e}')
            raise

    def _batch_tokenize(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Tokenize a batch of prompts with padding
        
        Args:
            prompts: List of text prompts
            
        Returns:
            Dictionary with tokenized inputs suitable for MLX
        """
        pass

    def _batch_generate(self, input_ids, attention_mask, generation_params: Dict) -> List[str]:
        """
        Perform batch generation using MLX model
        
        Args:
            input_ids: Batched input token IDs
            attention_mask: Attention mask for padded sequences
            generation_params: Generation parameters
            
        Returns:
            List of generated responses
        """
        pass

class MLXManager:
    """Manager for MLX models and operations"""

    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.available = MLX_AVAILABLE and is_apple_silicon()
        if self.available:
            logger.info('MLX manager initialized - Apple Silicon detected')
        else:
            logger.debug('MLX manager not available - requires Apple Silicon and mlx-lm')

    def create_pipeline(self, model_id: str, **kwargs) -> MLXInferencePipeline:
        """Create an MLX inference pipeline"""
        if not self.available:
            raise RuntimeError('MLX not available on this platform')
        config = MLXModelConfig(model_id=model_id, **kwargs)
        return MLXInferencePipeline(config, self.cache_manager)

    def is_mlx_model(self, model_id: str) -> bool:
        """Check if model should use MLX"""
        return should_use_mlx(model_id)

class InferenceClient:
    """OpenAI SDK Compatible client for local inference with dynamic model support"""

    def __init__(self):
        self.cache_manager = CacheManager.get_instance(max_size=4)
        self.device_manager = DeviceManager()
        self.model_manager = ModelManager(self.cache_manager, self.device_manager)
        self.lora_manager = LoRAManager(self.cache_manager)
        self.mlx_manager = MLXManager(self.cache_manager)
        self.chat = self.Chat(self)
        self.models = self.Models()

    def get_pipeline(self, model: str):
        """Get inference pipeline - automatically chooses MLX or PyTorch based on model"""
        if self.mlx_manager.available and should_use_mlx(model):
            logger.info(f'Using MLX pipeline for model: {model}')
            return self.mlx_manager.create_pipeline(model)
        else:
            logger.info(f'Using PyTorch pipeline for model: {model}')
            model_config = parse_model_string(model)
            return InferencePipeline(model_config, self.cache_manager, self.device_manager, self.model_manager, self.lora_manager)

    class Chat:
        """OpenAI-compatible chat interface"""

        def __init__(self, client: 'InferenceClient'):
            self.client = client
            self.completions = self.Completions(client)

        class Completions:

            def __init__(self, client: 'InferenceClient'):
                self.client = client

            def create(self, messages: List[Dict[str, str]], model: str, temperature: float=1.0, top_p: float=1.0, n: int=1, stream: bool=False, stop: Optional[Union[str, List[str]]]=None, max_tokens: Optional[int]=None, presence_penalty: float=0, frequency_penalty: float=0, logit_bias: Optional[Dict[str, float]]=None, seed: Optional[int]=None, logprobs: Optional[bool]=None, top_logprobs: Optional[int]=None, active_adapter: Optional[Dict[str, Any]]=None, decoding: Optional[str]=None, k: int=10, num_beams: int=1, length_penalty: float=1.0, no_repeat_ngram_size: int=0, early_stopping: bool=False, aggregate_paths: bool=True, top_k: int=27, min_p: float=0.03, reasoning_effort: str='low', thought_switch_tokens: List[str]=[], min_thinking_tokens: Optional[int]=None, max_thinking_tokens: Optional[int]=None, max_thoughts: Optional[int]=None, prefill: str='', start_think_token: str='<think>', end_think_token: str='</think>', **kwargs) -> ChatCompletion:
                """Create a chat completion with OpenAI-compatible parameters"""
                logger.info('Starting chat completion creation')
                if stream:
                    raise NotImplementedError('Streaming is not yet supported')
                logger.info(f'Getting pipeline for model: {model}')
                pipeline = self.client.get_pipeline(model)
                logger.info('Pipeline acquired')
                if active_adapter is not None:
                    logger.info(f'Setting active adapter to: {active_adapter}')
                    pipeline.lora_manager.set_active_adapter(pipeline.current_model, active_adapter)
                responses = []
                logprobs_results = []
                prompt_tokens = 0
                completion_tokens = 0
                try:
                    if decoding:
                        logger.info(f'Using specialized decoding approach: {decoding}')
                        mlx_unsupported_decodings = ['cot_decoding', 'entropy_decoding', 'autothink', 'deepconf']
                        if isinstance(pipeline, MLXInferencePipeline) and decoding in mlx_unsupported_decodings:
                            logger.warning(f'{decoding} is not supported for MLX models. Falling back to standard generation.')
                            decoding = None
                    if decoding:
                        if not isinstance(pipeline, MLXInferencePipeline):
                            pipeline.current_model.eval()
                            device = pipeline.current_model.device
                        else:
                            device = None
                        if decoding == 'cot_decoding':
                            cot_params = {'k': k, 'num_beams': num_beams, 'max_new_tokens': max_tokens if max_tokens is not None else 512, 'temperature': temperature, 'top_p': top_p, 'repetition_penalty': 1.0, 'length_penalty': length_penalty, 'no_repeat_ngram_size': no_repeat_ngram_size, 'early_stopping': early_stopping, 'aggregate_paths': aggregate_paths}
                            result, confidence = cot_decode(pipeline.current_model, pipeline.tokenizer, messages, **cot_params)
                            responses = [result]
                            logprobs_results = [{'confidence_score': confidence} if confidence is not None else None]
                            completion_tokens = len(pipeline.tokenizer.encode(result))
                        elif decoding == 'entropy_decoding':
                            original_dtype = pipeline.current_model.dtype
                            pipeline.current_model = pipeline.current_model.to(torch.float32)
                            try:
                                generator = None
                                if seed is not None:
                                    generator = torch.Generator(device=device)
                                    generator.manual_seed(seed)
                                else:
                                    generator = torch.Generator(device=device)
                                    generator.manual_seed(1337)
                                entropy_params = {'max_new_tokens': max_tokens if max_tokens is not None else 4096, 'temperature': temperature, 'top_p': top_p, 'top_k': top_k, 'min_p': min_p, 'generator': generator}
                                with torch.amp.autocast('cuda', enabled=False), torch.inference_mode():
                                    result = entropy_decode(pipeline.current_model, pipeline.tokenizer, messages, **entropy_params)
                                responses = [result]
                                logprobs_results = [None]
                                completion_tokens = len(pipeline.tokenizer.encode(result))
                            finally:
                                pipeline.current_model = pipeline.current_model.to(original_dtype)
                        elif decoding == 'thinkdeeper':
                            thinkdeeper_config = get_effort_profile(reasoning_effort, max_tokens)
                            custom_config = {'min_thinking_tokens': min_thinking_tokens if min_thinking_tokens is not None else thinkdeeper_config['min_thinking_tokens'], 'max_thinking_tokens': max_thinking_tokens if max_thinking_tokens is not None else thinkdeeper_config['max_thinking_tokens'], 'max_thoughts': max_thoughts if max_thoughts is not None else thinkdeeper_config['max_thoughts'], 'thought_switch_tokens': thought_switch_tokens if thought_switch_tokens else thinkdeeper_config['thought_switch_tokens'], 'prefill': prefill if prefill else thinkdeeper_config['prefill'], 'start_think_token': start_think_token, 'end_think_token': end_think_token}
                            thinkdeeper_config.update(custom_config)
                            if isinstance(pipeline, MLXInferencePipeline):
                                logger.info('Using MLX ThinkDeeper implementation')
                                user_max_tokens = max_tokens if max_tokens is not None else 512
                                total_tokens_needed = max_thinking_tokens + 512
                                adjusted_max_tokens = max(user_max_tokens, total_tokens_needed)
                                thinkdeeper_config_with_tokens = thinkdeeper_config.copy()
                                thinkdeeper_config_with_tokens['max_tokens'] = adjusted_max_tokens
                                logger.debug(f'ThinkDeeper tokens: user={user_max_tokens}, thinking={max_thinking_tokens}, adjusted={adjusted_max_tokens}')
                                result, reasoning_tokens = thinkdeeper_decode_mlx(pipeline.model, pipeline.tokenizer, messages, thinkdeeper_config_with_tokens)
                            else:
                                logger.info('Using PyTorch ThinkDeeper implementation')
                                result, reasoning_tokens = thinkdeeper_decode(pipeline.current_model, pipeline.tokenizer, messages, thinkdeeper_config)
                            responses = [result]
                            logprobs_results = [None]
                            completion_tokens = len(pipeline.tokenizer.encode(result))
                        elif decoding == 'autothink':
                            steering_dataset = kwargs.get('steering_dataset', 'codelion/Qwen3-0.6B-pts-steering-vectors')
                            target_layer = kwargs.get('target_layer', 19)
                            autothink_config = {'steering_dataset': steering_dataset, 'target_layer': target_layer, 'pattern_strengths': kwargs.get('pattern_strengths', {'depth_and_thoroughness': 2.5, 'numerical_accuracy': 2.0, 'self_correction': 3.0, 'exploration': 2.0, 'organization': 1.5})}
                            result = autothink_decode(pipeline.current_model, pipeline.tokenizer, messages, autothink_config)
                            responses = [result]
                            logprobs_results = [None]
                            completion_tokens = len(pipeline.tokenizer.encode(result))
                        elif decoding == 'deepconf':
                            deepconf_config = {'variant': kwargs.get('variant', 'low'), 'warmup_samples': kwargs.get('warmup_samples', 16), 'consensus_threshold': kwargs.get('consensus_threshold', 0.95), 'max_traces': kwargs.get('max_traces', 128), 'window_size': kwargs.get('window_size', 2048), 'top_k': kwargs.get('top_k', 5), 'min_trace_length': kwargs.get('min_trace_length', 100), 'max_tokens_per_trace': kwargs.get('max_tokens_per_trace', 4096), 'temperature': temperature, 'confidence_metric': kwargs.get('confidence_metric', 'average_confidence'), 'include_stats': kwargs.get('include_stats', False)}
                            result, tokens_used = deepconf_decode(pipeline.current_model, pipeline.tokenizer, messages, deepconf_config)
                            responses = [result]
                            logprobs_results = [None]
                            completion_tokens = tokens_used
                        else:
                            raise ValueError(f'Unknown specialized decoding approach: {decoding}')
                        prompt_text = pipeline.tokenizer.apply_chat_template(messages, tokenize=False)
                        prompt_tokens = len(pipeline.tokenizer.encode(prompt_text))
                    else:
                        prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        generation_params = {'temperature': temperature, 'top_p': top_p, 'num_return_sequences': n, 'max_new_tokens': max_tokens if max_tokens is not None else 4096, 'presence_penalty': presence_penalty, 'frequency_penalty': frequency_penalty, 'stop_sequences': [stop] if isinstance(stop, str) else stop, 'seed': seed, 'logit_bias': logit_bias, 'logprobs': logprobs, 'top_logprobs': top_logprobs}
                        responses, token_counts, logprobs_results = pipeline.generate(prompt, generation_params=generation_params)
                        prompt_tokens = len(pipeline.tokenizer.encode(prompt))
                        completion_tokens = sum(token_counts)
                    total_reasoning_tokens = 0
                    for response in responses:
                        total_reasoning_tokens += count_reasoning_tokens(response, pipeline.tokenizer)
                    response_dict = {'id': f'chatcmpl-{int(time.time() * 1000)}', 'object': 'chat.completion', 'created': int(time.time()), 'model': model, 'choices': [{'index': idx, 'message': {'role': 'assistant', 'content': response, **({'logprobs': logprob_result} if logprob_result else {})}, 'finish_reason': 'stop'} for idx, (response, logprob_result) in enumerate(zip(responses, logprobs_results))], 'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': completion_tokens + prompt_tokens, 'reasoning_tokens': total_reasoning_tokens}}
                    logger.debug(f'Response : {response_dict}')
                    return ChatCompletion(response_dict)
                except Exception as e:
                    logger.error(f'Error in chat completion: {str(e)}')
                    raise

    class Models:
        """OpenAI-compatible models interface"""

        def list(self):
            """Return list of supported models"""
            try:
                import requests
                response = requests.get('https://huggingface.co/api/models?sort=downloads&direction=-1&filter=text-generation&limit=20')
                models = response.json()
                model_list = []
                for model in models:
                    if 'pipeline_tag' in model and model['pipeline_tag'] == 'text-generation':
                        model_list.append({'id': model['id'], 'object': 'model', 'created': int(time.time()), 'owned_by': 'huggingface'})
                return {'data': model_list, 'object': 'list'}
            except Exception as e:
                logger.warning(f'Failed to fetch models: {e}')
                return {'data': [{'id': 'HuggingFaceTB/SmolLM-135M-Instruct', 'object': 'model', 'created': int(time.time()), 'owned_by': 'huggingface'}], 'object': 'list'}

