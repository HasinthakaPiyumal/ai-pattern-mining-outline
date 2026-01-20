# Cluster 2

def run_pts(args):
    """Run the Pivotal Token Search algorithm."""
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info(f'Running PTS with model {args.model}')
    examples = load_dataset(dataset_name=args.dataset, split=args.split, config=args.config, sample_size=args.sample_size, seed=args.seed, query_key=args.query_key, answer_key=args.answer_key)
    if not examples:
        logger.error(f'No examples loaded from dataset {args.dataset}')
        return
    logger.info(f'Loaded {len(examples)} examples from {args.dataset}')
    oracle = create_oracle_from_dataset(examples, debug_mode=args.debug)
    storage = TokenStorage(filepath=args.output_path)
    searcher = PivotalTokenSearcher(model_name=args.model, oracle=oracle, device=args.device, prob_threshold=args.prob_threshold, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p, max_new_tokens=args.max_new_tokens, num_samples=args.num_samples, batch_size=args.batch_size, token_storage=storage, log_level=getattr(logging, args.log_level.upper()), debug_mode=args.debug)
    successful_searches = 0
    total_pivotal_tokens = 0
    for i, example in enumerate(tqdm(examples[:args.max_examples], desc='Processing examples')):
        if i >= args.max_examples:
            break
        logger.info(f'Processing example {i + 1}/{min(len(examples), args.max_examples)}')
        query = example['query']
        if not query.strip():
            logger.warning(f'Skipping empty query in example {i}')
            continue
        query_pivotal_tokens = list(searcher.search_pivotal_tokens(query=query, system_prompt=args.system_prompt, task_type='generic', dataset_id=args.dataset, item_id=example.get('item_id', str(i)), max_generations=args.max_generations, min_prob=args.min_prob, max_prob=args.max_prob, category=example.get('metadata', {}).get('category', None)))
        if query_pivotal_tokens:
            for token in query_pivotal_tokens:
                storage.add_token(token)
            successful_searches += 1
            total_pivotal_tokens += len(query_pivotal_tokens)
            logger.info(f'Found {len(query_pivotal_tokens)} pivotal tokens for example {i}')
        else:
            logger.info(f'No pivotal tokens found for example {i}')
    logger.info(f'Found pivotal tokens in {successful_searches}/{args.max_examples} examples')
    logger.info(f'Total pivotal tokens found: {total_pivotal_tokens}')
    if total_pivotal_tokens > 0:
        storage.save()
        logger.info(f'Saved tokens to {args.output_path}')
    else:
        logger.info(f'No tokens found, no file saved')
    logger.info(f'Finished processing {args.max_examples} examples')

def load_dataset(dataset_name: str='codelion/optillmbench', split: str='train', config: Optional[str]=None, sample_size: Optional[int]=None, filter_fn: Optional[Callable[[Dict[str, Any]], bool]]=None, seed: int=42, query_key: Optional[str]=None, answer_key: Optional[str]=None) -> List[Dict[str, Any]]:
    """
    Load a dataset from Hugging Face or a local path.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face or local path
        split: Dataset split to use
        sample_size: Number of examples to sample (None for all)
        filter_fn: Optional function to filter examples
        seed: Random seed for sampling
        query_key: Key for query/question in the dataset (None for auto-detect)
        answer_key: Key for answer/output in the dataset (None for auto-detect)
        
    Returns:
        List of dataset examples
    """
    if config:
        logger.info(f'Loading dataset {dataset_name} (config: {config}, split: {split})')
    else:
        logger.info(f'Loading dataset {dataset_name} (split: {split})')
    try:
        if config:
            dataset = hf_load_dataset(dataset_name, config, split=split)
        else:
            dataset = hf_load_dataset(dataset_name, split=split)
        logger.info(f'Loaded {len(dataset)} examples')
        examples = [dict(example) for example in dataset]
        if filter_fn:
            examples = [example for example in examples if filter_fn(example)]
            logger.info(f'Filtered to {len(examples)} examples')
        if sample_size and sample_size < len(examples):
            random.seed(seed)
            examples = random.sample(examples, sample_size)
            logger.info(f'Sampled {len(examples)} examples')
        if not query_key or not answer_key:
            if examples:
                available_keys = list(examples[0].keys())
                logger.info(f'Available keys in dataset: {available_keys}')
                if dataset_name == 'codelion/optillmbench':
                    default_query_key = 'question'
                    default_answer_key = 'answer'
                else:
                    possible_query_keys = ['question', 'query', 'instruction', 'problem', 'prompt']
                    possible_answer_keys = ['answer', 'output', 'solution', 'response', 'canonical_solution']
                    default_query_key = next((k for k in possible_query_keys if k in available_keys), 'question')
                    default_answer_key = next((k for k in possible_answer_keys if k in available_keys), 'answer')
                query_key = query_key or default_query_key
                answer_key = answer_key or default_answer_key
                logger.info(f"Using keys: query_key='{query_key}', answer_key='{answer_key}'")
        formatted_examples = []
        for i, example in enumerate(examples):
            if query_key not in example:
                logger.warning(f"Example {i} missing '{query_key}' key, skipping")
                continue
            if answer_key and answer_key not in example:
                logger.warning(f"Example {i} missing '{answer_key}' key")
            formatted = {'query': example[query_key], 'answer': example.get(answer_key) if answer_key else None, 'dataset_id': dataset_name, 'item_id': str(i)}
            formatted['metadata'] = example
            formatted_examples.append(formatted)
        return formatted_examples
    except Exception as e:
        logger.error(f'Error loading dataset {dataset_name}: {e}')
        return []

def create_oracle_from_dataset(examples: List[Dict[str, Any]], debug_mode: bool=False) -> Any:
    """
    Create an appropriate oracle from dataset examples.
    
    Args:
        examples: List of dataset examples
        debug_mode: Whether to enable debug mode
        
    Returns:
        Oracle instance for the dataset
    """
    try:
        is_optillmbench = False
        is_gsm8k = False
        has_categories = False
        if examples and 'dataset_id' in examples[0]:
            dataset_id = examples[0]['dataset_id']
            if dataset_id == 'codelion/optillmbench':
                is_optillmbench = True
                if examples[0]['metadata'] and 'category' in examples[0]['metadata']:
                    has_categories = True
            elif 'gsm8k' in dataset_id.lower():
                is_gsm8k = True
        if is_optillmbench and has_categories:
            from .oracle import OptiBenchOracle
            examples_with_categories = {}
            for example in examples:
                if example.get('query') and example.get('answer'):
                    category = example['metadata'].get('category', 'general')
                    if category not in examples_with_categories:
                        examples_with_categories[category] = {}
                    examples_with_categories[category][example['query']] = example['answer']
            return OptiBenchOracle(examples_with_categories=examples_with_categories, debug_mode=debug_mode)
        elif is_gsm8k:
            from .oracle import MathOracle
            answers = {}
            for example in examples:
                if example.get('query') and example.get('answer'):
                    if isinstance(example['answer'], str):
                        gsm8k_match = re.search('####\\s*(-?[\\d,]+(?:\\.\\d+)?)', example['answer'])
                        if gsm8k_match:
                            answers[example['query']] = gsm8k_match.group(1).replace(',', '')
                        else:
                            answers[example['query']] = example['answer']
                    else:
                        answers[example['query']] = example['answer']
            return MathOracle(answers=answers, dataset_format='gsm8k', debug_mode=debug_mode)
        else:
            from .oracle import QAOracle
            answers = {}
            for example in examples:
                if example.get('query') and example.get('answer'):
                    answers[example['query']] = example['answer']
            return QAOracle(answers=answers, debug_mode=debug_mode)
    except Exception as e:
        logger.error(f'Error creating oracle: {e}')
        from .oracle import DummyOracle
        return DummyOracle()

def run_thought_anchors(args):
    """Run the Thought Anchor Search algorithm."""
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info(f'Running Thought Anchor Search with model {args.model}')
    examples = load_dataset(dataset_name=args.dataset, split=args.split, config=args.config, sample_size=args.sample_size, seed=args.seed, query_key=args.query_key, answer_key=args.answer_key)
    if not examples:
        logger.error(f'No examples loaded from dataset {args.dataset}')
        return
    logger.info(f'Loaded {len(examples)} examples from {args.dataset}')
    oracle = create_oracle_from_dataset(examples, debug_mode=args.debug)
    if hasattr(args, 'skip_embeddings') and args.skip_embeddings:
        oracle.skip_embeddings = True
        logger.info('Embeddings will be skipped for faster processing')
    storage = ThoughtAnchorStorage(filepath=args.output_path)
    searcher = ThoughtAnchorSearcher(model_name=args.model, oracle=oracle, device=args.device, prob_threshold=args.prob_threshold, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p, max_new_tokens=args.max_new_tokens, num_samples=args.num_samples, batch_size=args.batch_size, log_level=getattr(logging, args.log_level.upper()), debug_mode=args.debug)
    searcher.storage = storage
    successful_searches = 0
    total_thought_anchors = 0
    for i, example in enumerate(tqdm(examples[:args.max_examples], desc='Processing examples')):
        if i >= args.max_examples:
            break
        logger.info(f'Processing example {i + 1}/{min(len(examples), args.max_examples)}')
        query = example['query']
        if not query.strip():
            logger.warning(f'Skipping empty query in example {i}')
            continue
        logger.info(f'Generating reasoning trace for analysis...')
        if args.system_prompt:
            messages = [{'role': 'system', 'content': args.system_prompt}, {'role': 'user', 'content': query}]
            prompt = searcher.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            category = example.get('metadata', {}).get('category', None)
            if category and hasattr(oracle, 'get_prompt_for_category'):
                prompt = oracle.get_prompt_for_category(query, category)
            else:
                prompt = query
        tokenized = searcher.tokenizer(prompt, return_tensors='pt', padding=True)
        input_ids = tokenized.input_ids.to(searcher.device)
        attention_mask = tokenized.attention_mask.to(searcher.device) if 'attention_mask' in tokenized else None
        try:
            with torch.no_grad():
                outputs = searcher.model.generate(input_ids, attention_mask=attention_mask, do_sample=True, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p, pad_token_id=searcher.tokenizer.pad_token_id, return_dict_in_generate=True)
            reasoning_trace = searcher.tokenizer.decode(outputs.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
            if not reasoning_trace.strip():
                logger.warning(f'Empty reasoning trace for example {i}, skipping')
                continue
            logger.info(f'Generated reasoning trace with {len(reasoning_trace)} characters')
            query_thought_anchors = list(searcher.search_thought_anchors(query=query, reasoning_trace=reasoning_trace, system_prompt=args.system_prompt, task_type='generic', dataset_id=args.dataset, item_id=example.get('item_id', str(i)), min_prob=args.min_prob, max_prob=args.max_prob, category=example.get('metadata', {}).get('category', None)))
            if query_thought_anchors:
                for anchor in query_thought_anchors:
                    storage.add_thought_anchor(anchor)
                successful_searches += 1
                total_thought_anchors += len(query_thought_anchors)
                logger.info(f'Found {len(query_thought_anchors)} thought anchors for example {i}')
            else:
                logger.info(f'No thought anchors found for example {i}')
        except Exception as e:
            logger.error(f'Error processing example {i}: {e}')
            continue
    logger.info(f'Found thought anchors in {successful_searches}/{args.max_examples} examples')
    logger.info(f'Total thought anchors found: {total_thought_anchors}')
    if total_thought_anchors > 0:
        storage.save()
        logger.info(f'Saved thought anchors to {args.output_path}')
        summary = storage.get_anchor_summary()
        logger.info(f'Thought Anchor Summary:')
        logger.info(f'  Total anchors: {summary['total_anchors']}')
        logger.info(f'  Positive anchors: {summary['positive_anchors']}')
        logger.info(f'  Negative anchors: {summary['negative_anchors']}')
        logger.info(f'  Average importance: {summary['average_importance']:.3f}')
        logger.info(f'  Category distribution: {summary['category_distribution']}')
    else:
        logger.info(f'No thought anchors found, no file saved')
    logger.info(f'Finished processing {args.max_examples} examples')

