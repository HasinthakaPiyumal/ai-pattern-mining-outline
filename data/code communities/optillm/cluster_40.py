# Cluster 40

def main():
    parser = argparse.ArgumentParser(description='Run AutoThink demo')
    parser.add_argument('--model', type=str, default='deepseek-ai/deepseek-r1-llama-8b', help='Model name or path')
    parser.add_argument('--steering-dataset', type=str, default='codelion/Qwen3-0.6B-pts-steering-vectors', help='Steering vectors dataset')
    parser.add_argument('--target-layer', type=int, default=19, help='Target layer for steering')
    parser.add_argument('--query', type=str, default='Explain quantum computing to me in detail', help='Query to process')
    args = parser.parse_args()
    try:
        logger.info(f'Loading model: {args.model}')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Using device: {device}')
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model_kwargs = {'trust_remote_code': True}
        if device == 'cuda':
            model_kwargs['torch_dtype'] = torch.float16
            model_kwargs['device_map'] = 'auto'
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info('Model and tokenizer loaded successfully')
        config = {'steering_dataset': args.steering_dataset, 'target_layer': args.target_layer, 'pattern_strengths': {'depth_and_thoroughness': 2.5, 'numerical_accuracy': 2.0, 'self_correction': 3.0, 'exploration': 2.0, 'organization': 1.5}}
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': args.query}]
        logger.info('Running AutoThink processing...')
        response = autothink_decode(model, tokenizer, messages, config)
        print('\n' + '=' * 80)
        print('QUERY:', args.query)
        print('-' * 80)
        print(response)
        print('=' * 80 + '\n')
    except Exception as e:
        logger.error(f'Error in AutoThink demo: {str(e)}')
        raise

def autothink_decode(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, messages: List[Dict[str, str]], request_config: Optional[Dict[str, Any]]=None) -> str:
    """
    Main plugin execution function with AutoThink's controlled thinking process.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer
        messages: List of message dictionaries
        request_config: Optional configuration dictionary
        
    Returns:
        Generated response with thinking process
    """
    logger.info('Starting AutoThink processing')
    config = {}
    if request_config:
        config.update(request_config)
    try:
        processor = AutoThinkProcessor(model, tokenizer, config)
        response = processor.process(messages)
        return response
    except Exception as e:
        logger.error(f'Error in AutoThink processing: {str(e)}')
        raise

