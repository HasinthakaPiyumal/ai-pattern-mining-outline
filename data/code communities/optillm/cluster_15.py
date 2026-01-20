# Cluster 15

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

def cot_decode(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, messages: List[Dict[str, str]], k: int=10, num_beams: int=1, max_new_tokens: int=512, temperature: float=1.0, top_p: float=1.0, repetition_penalty: float=1.0, length_penalty: float=1.0, no_repeat_ngram_size: int=0, early_stopping: bool=False, aggregate_paths: bool=False) -> Tuple[str, float]:
    """
    Implement CoT-decoding for a given chat input.
    
    Args:
        model: The Hugging Face transformer model.
        tokenizer: The associated tokenizer.
        messages: List of chat messages in the format [{"role": "user", "content": "..."}]
        k: The number of alternative tokens to consider at the first step.
        num_beams: Number of beams for beam search.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        repetition_penalty: Repetition penalty factor.
        length_penalty: Length penalty factor.
        no_repeat_ngram_size: Size of n-grams to avoid repeating.
        early_stopping: Whether to stop generation when all beams are finished.
        aggregate_paths: Whether to aggregate multiple paths.

    Returns:
        A tuple containing the best path (or aggregated result) and its confidence score.
    """
    device = get_device()
    model.to(device)
    if tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = '\n'.join([f'{msg['role']}: {msg['content']}' for msg in messages])
        input_text += '\nassistant:'
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        first_token_logits = outputs.logits[0, -1, :]
        top_k_logits, top_k_indices = torch.topk(first_token_logits, k)
    paths = []
    for idx in top_k_indices:
        start_ids = torch.cat([input_ids, idx.unsqueeze(0).unsqueeze(0)], dim=-1)
        start_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=-1)
        output = model.generate(start_ids, attention_mask=start_mask, max_new_tokens=max_new_tokens, num_beams=num_beams, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, length_penalty=length_penalty, no_repeat_ngram_size=no_repeat_ngram_size, early_stopping=early_stopping, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=True)
        generated_sequence = output.sequences[0]
        answer_ids = generated_sequence[len(input_ids[0]):]
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
        confidence = calculate_confidence(output.scores, answer_ids)
        paths.append((answer_text, confidence))
    if aggregate_paths:
        return aggregate_paths_based_on_scores(paths)
    else:
        return max(paths, key=lambda x: x[1])

def entropy_decode(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, messages: List[Dict[str, str]], max_new_tokens: int=512, temperature: float=0.666, top_p: float=0.9, top_k: int=27, min_p: float=0.03, generator: torch.Generator=torch.Generator(device=device).manual_seed(1337)) -> str:
    model.to(device)
    logging.info('Starting entropy decoding')
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = '\n'.join([f'{msg['role']}: {msg['content']}' for msg in messages])
        input_text += '\nassistant:'
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    generated_tokens = []
    gen_tokens = input_ids
    past_key_values = None
    stop = torch.tensor([tokenizer.eos_token_id], device=device, dtype=torch.int32)
    for step in range(max_new_tokens):
        logging.debug(f'Generation step: {step + 1}')
        with torch.no_grad():
            outputs = model(input_ids if past_key_values is None else input_ids[:, -1:], attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True, output_attentions=True)
        logits = outputs.logits[:, -1:, :]
        attention_scores = outputs.attentions[-1]
        past_key_values = outputs.past_key_values
        entropy, varentropy = calculate_varentropy_logsoftmax(logits)
        attention_metrics = calculate_attention_metrics(attention_scores)
        metrics = {'logits_entropy': entropy, 'logits_varentropy': varentropy, **attention_metrics}
        logging.debug(f'Metrics: entropy={entropy.item():.3f}, varentropy={varentropy.item():.3f}')
        if entropy < 0.1 and varentropy < 0.1:
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
            logging.debug('Using greedy sampling')
        elif entropy > 3.0 and varentropy < 0.1:
            if not torch.isin(gen_tokens[:, -1], torch.tensor([2564], device=device)).any():
                next_token = torch.tensor([[2564]], dtype=torch.int32, device=device)
                logging.debug('Inserting clarification token')
            else:
                temp_adj = 1.3 + 0.2 * attention_metrics['attn_entropy'].item()
                next_token = _sample(logits, temperature=min(1.5, temperature * temp_adj), top_p=top_p, top_k=top_k, min_p=min_p, generator=generator)
                logging.debug(f'Using adjusted temperature sampling: {temp_adj:.3f}')
        elif entropy < 5.0 and varentropy > 5.0:
            temp_adj = 1.2 + 0.3 * attention_metrics['interaction_strength'].item()
            top_k_adj = max(5, int(top_k * (1 + 0.5 * (1 - attention_metrics['agreement'].item()))))
            next_token = _sample(logits, temperature=min(1.5, temperature * temp_adj), top_p=top_p, top_k=top_k_adj, min_p=min_p, generator=generator)
            logging.debug(f'Using exploration sampling: temp={temp_adj:.3f}, top_k={top_k_adj}')
        elif entropy > 5.0 and varentropy > 5.0:
            temp_adj = 2.0 + 0.5 * attention_metrics['attn_varentropy'].item()
            top_p_adj = max(0.5, top_p - 0.2 * attention_metrics['attn_entropy'].item())
            next_token = _sample(logits, temperature=max(2.0, temperature * temp_adj), top_p=top_p_adj, top_k=top_k, min_p=min_p, generator=generator)
            logging.debug(f'Using high uncertainty sampling: temp={temp_adj:.3f}, top_p={top_p_adj:.3f}')
        else:
            next_token = adaptive_sample(logits, metrics, gen_tokens, n_samples=5, base_temp=temperature, base_top_p=top_p, base_top_k=top_k, base_min_p=min_p, generator=generator)
            logging.debug('Using adaptive sampling')
        generated_tokens.append(next_token.item())
        gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device, dtype=torch.long)], dim=-1)
        logging.debug(f'Generated token: {tokenizer.decode([next_token.item()])}')
        if torch.isin(next_token, stop).any():
            logging.info('Reached stop token. Ending generation.')
            break
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    logging.info('Finished entropy decoding')
    logging.info(f'Generated text: {generated_text}')
    return generated_text

def thinkdeeper_decode_mlx(model, tokenizer, messages: List[Dict[str, str]], request_config: Dict[str, Any]=None) -> str:
    """MLX-compatible ThinkDeeper processing function"""
    logger.info('Starting MLX ThinkDeeper processing')
    if not MLX_AVAILABLE:
        raise RuntimeError('MLX framework not available for ThinkDeeper processing')
    config = DEFAULT_CONFIG.copy()
    if request_config:
        for key in DEFAULT_CONFIG:
            if key in request_config:
                config[key] = request_config[key]
        if 'max_tokens' in request_config:
            config['max_tokens'] = request_config['max_tokens']
    logger.info(f'MLX ThinkDeeper using config: {config}')
    try:
        processor = MLXThinkDeeperProcessor(config, tokenizer, model)
        response, reasoning_tokens = processor.reasoning_effort(messages)
        return (response, reasoning_tokens)
    except Exception as e:
        logger.error(f'Error in MLX ThinkDeeper processing: {str(e)}')
        raise

def thinkdeeper_decode(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, messages: List[Dict[str, str]], request_config: Dict[str, Any]=None) -> str:
    """Main plugin execution function with ThinkDeeper's controlled thinking process"""
    logger.info('Starting ThinkDeeper processing')
    config = DEFAULT_CONFIG.copy()
    if request_config:
        for key in DEFAULT_CONFIG:
            if key in request_config:
                config[key] = request_config[key]
    logger.info(f'Using config: {config}')
    try:
        processor = ThinkDeeperProcessor(config, tokenizer, model)
        response, reasoning_tokens = processor.reasoning_effort(messages)
        return (response, reasoning_tokens)
    except Exception as e:
        logger.error(f'Error in ThinkDeeper processing: {str(e)}')
        raise

def count_reasoning_tokens(text: str, tokenizer=None) -> int:
    """
    Count tokens within <think>...</think> tags in the given text.
    
    Args:
        text: The text to analyze
        tokenizer: Optional tokenizer instance for precise counting
        
    Returns:
        Number of reasoning tokens (0 if no think tags found)
    """
    if not text or not isinstance(text, str):
        return 0
    complete_pattern = '<think>(.*?)</think>'
    complete_matches = re.findall(complete_pattern, text, re.DOTALL)
    truncated_pattern = '<think>(?!.*</think>)(.*)$'
    truncated_match = re.search(truncated_pattern, text, re.DOTALL)
    thinking_content = ''.join(complete_matches)
    if truncated_match:
        thinking_content += truncated_match.group(1)
    if not thinking_content:
        return 0
    if tokenizer and hasattr(tokenizer, 'encode'):
        try:
            tokens = tokenizer.encode(thinking_content)
            return len(tokens)
        except Exception as e:
            logger.warning(f'Failed to count tokens with tokenizer: {e}')
    content_length = len(thinking_content.strip())
    return max(1, content_length // 4) if content_length > 0 else 0

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def calculate_confidence(logits: List[torch.Tensor], answer_ids: torch.Tensor) -> float:
    """
    Calculate the confidence score (Δ) as specified in the paper.
    
    Args:
        logits: List of logits for each decoding step
        answer_ids: Tensor of token ids for the answer
    
    Returns:
        Confidence score (Δ)
    """
    confidence_sum = 0.0
    valid_tokens = 0
    for t, token_id in enumerate(answer_ids):
        if t >= len(logits):
            break
        token_logits = logits[t]
        probs = torch.softmax(token_logits, dim=-1)
        if probs.size(-1) > 1:
            top_2_probs, _ = torch.topk(probs, min(2, probs.size(-1)))
            if top_2_probs.size(-1) > 1:
                confidence_sum += (top_2_probs[-1][0] - top_2_probs[-1][1]).item()
            else:
                confidence_sum += 1.0
        else:
            confidence_sum += 1.0
        valid_tokens += 1
    return confidence_sum / valid_tokens if valid_tokens > 0 else 0.0

def aggregate_paths_based_on_scores(paths: List[Tuple[str, float]]) -> Tuple[str, float]:
    """Aggregate multiple paths based on their confidence scores."""
    answer_scores = {}
    for answer, delta in paths:
        answer_scores[answer] = answer_scores.get(answer, 0) + delta
    best_answer = max(answer_scores, key=answer_scores.get)
    return (best_answer, answer_scores[best_answer])

def adaptive_sample(logits: torch.Tensor, metrics: Dict[str, torch.Tensor], gen_tokens: torch.Tensor, n_samples: int, base_temp: float=0.666, base_top_p: float=0.9, base_top_k: int=40, base_min_p: float=0.03, generator: torch.Generator=None) -> torch.Tensor:
    logits_uncertainty = metrics['logits_entropy'] + metrics['logits_varentropy']
    attn_uncertainty = metrics['attn_entropy'] + metrics['attn_varentropy']
    temperature = base_temp * (1 + 0.3 * logits_uncertainty + 0.2 * attn_uncertainty - 0.2 * metrics['agreement'])
    top_p = torch.clamp(base_top_p * (1 + 0.1 * metrics['attn_varentropy']), 0.1, 1.0)
    top_k = int(torch.clamp(torch.round(torch.tensor(base_top_k) * (1 + 0.3 * metrics['interaction_strength'].item() - 0.2 * metrics['agreement'].item())), min=1, max=100).item())
    min_p = torch.clamp(base_min_p * (1 - 0.5 * logits_uncertainty), 0.01, 0.5)
    logging.debug(f'Adaptive sampling params: temp={temperature.item():.3f}, top_p={top_p.item():.3f}, top_k={top_k}, min_p={min_p.item():.3f}')
    samples = []
    for _ in range(n_samples):
        sample = _sample(logits, temperature=temperature.item(), top_p=top_p.item(), top_k=top_k, min_p=min_p.item(), generator=generator)
        samples.append(sample)

    def score_sample(sample):
        sample_flat = sample.flatten().to(torch.long)
        one_hot = F.one_hot(sample_flat, logits.shape[-1])
        log_probs = F.log_softmax(logits, dim=-1).view(-1, logits.shape[-1])
        log_prob = torch.sum(log_probs * one_hot)
        confidence_score = (1 - metrics['logits_entropy']) * 0.1 + (1 - metrics['attn_entropy']) * 0.2 + (1 - metrics['logits_varentropy']) * 0.3 + (1 - metrics['attn_varentropy']) * 0.4 + metrics['agreement'] * 0.5 + metrics['interaction_strength'] * 0.6
        return log_prob + confidence_score
    sample_scores = torch.stack([score_sample(sample) for sample in samples])
    best_sample_idx = torch.argmax(sample_scores)
    return samples[best_sample_idx]

def _sample(logits: torch.Tensor, temperature=0.666, top_p=0.9, top_k=27, min_p: float=0.0, generator: torch.Generator=None) -> torch.Tensor:
    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = F.softmax(logit / temperature, dim=-1)
    if min_p > 0.0:
        p_max = torch.max(probs, dim=-1, keepdim=True).values
        indices_to_remove = probs < min_p * p_max
        logit = torch.where(indices_to_remove, torch.full_like(logit, float('-inf')), logit)
    top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]))
    probs_sort = torch.flip(top_k_probs, dims=[-1])
    probs_idx = torch.flip(top_k_indices, dims=[-1])
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = torch.where(probs_sum - probs_sort > top_p, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    probs_sort = probs_sort * (1 - mask)
    probs_sort = probs_sort / torch.sum(probs_sort, dim=-1, keepdim=True)
    next_token = torch.multinomial(probs_sort, 1, generator=generator)
    next_token_g = torch.gather(probs_idx, -1, next_token.reshape(bsz, 1).to(torch.int64))
    return next_token_g.to(torch.int32)

def score_sample(sample):
    sample_flat = sample.flatten().to(torch.long)
    one_hot = F.one_hot(sample_flat, logits.shape[-1])
    log_probs = F.log_softmax(logits, dim=-1).view(-1, logits.shape[-1])
    log_prob = torch.sum(log_probs * one_hot)
    confidence_score = (1 - metrics['logits_entropy']) * 0.1 + (1 - metrics['attn_entropy']) * 0.2 + (1 - metrics['logits_varentropy']) * 0.3 + (1 - metrics['attn_varentropy']) * 0.4 + metrics['agreement'] * 0.5 + metrics['interaction_strength'] * 0.6
    return log_prob + confidence_score

def calculate_varentropy_logsoftmax(logits: torch.Tensor, axis: int=-1) -> Tuple[torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits, dim=axis)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1)) ** 2, dim=axis)
    return (entropy, varentropy)

def calculate_attention_metrics(attention_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
    attention_probs = attention_weights
    attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, 1e-10, 1.0)), dim=-1)
    if attn_entropy.size(-1) > 1:
        attn_varentropy = torch.var(attn_entropy, dim=-1, unbiased=False)
    else:
        attn_varentropy = torch.zeros_like(attn_entropy)
    attn_varentropy = torch.where(torch.isnan(attn_varentropy), torch.zeros_like(attn_varentropy), attn_varentropy)
    mean_attention = torch.mean(attention_probs, dim=1)
    agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2))
    attention_scores_proxy = torch.log(torch.clamp(attention_probs, 1e-10, 1.0))
    interaction_strength = torch.mean(torch.abs(attention_scores_proxy), dim=(1, 2, 3))
    return {'attn_entropy': torch.mean(attn_entropy), 'attn_varentropy': torch.mean(attn_varentropy), 'agreement': torch.mean(agreement), 'interaction_strength': interaction_strength}

