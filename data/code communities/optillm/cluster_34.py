# Cluster 34

def augment_system_prompt(system_prompt: str, strategies: List[Any]) -> str:
    """
    Augment the system prompt with selected strategies and reasoning examples.
    Instructs the LLM to apply the strategies in its solution.
    
    Args:
        system_prompt: The original system prompt
        strategies: A list of strategies to add to the prompt
    
    Returns:
        str: The augmented system prompt
    """
    if not strategies:
        return system_prompt
    strategies_section = ''
    for i, strategy in enumerate(strategies, 1):
        strategies_section += f'Strategy {i} for {strategy.problem_type} problems:\n{strategy.strategy_text}\n\n'
        if strategy.reasoning_examples:
            reasoning = strategy.reasoning_examples[-1]
            if reasoning:
                strategies_section += f'Example reasoning process:\n<think>\n{reasoning}\n</think>\n\n'
    strategy_prompt = STRATEGY_APPLICATION_PROMPT.format(strategies_section=strategies_section)
    augmented_prompt = system_prompt + '\n\n' + strategy_prompt
    return augmented_prompt

def evaluate_strategy_effectiveness(response: str, thinking: Optional[str], selected_strategies: List[Strategy], client, model: str) -> Dict[str, bool]:
    """
    Evaluate how effective each strategy was in generating the response.
    
    Args:
        response: The LLM's final response to the query
        thinking: The LLM's reasoning process (if any)
        selected_strategies: The strategies that were used
        client: LLM client for making API calls
        model: Model identifier
    
    Returns:
        Dict[str, bool]: Mapping from strategy ID to effectiveness (True/False)
    """
    if not selected_strategies:
        return {}
    results = {}
    try:
        for strategy in selected_strategies:
            full_response = thinking + '\n\n' + response if thinking else response
            messages = [{'role': 'system', 'content': STRATEGY_EVALUATION_PROMPT}, {'role': 'user', 'content': f'Strategy:\n{strategy.strategy_text}\n\nResponse (including reasoning):\n{full_response}\n\nDoes the response show clear evidence that the strategy was effectively applied? Answer with ONLY YES or NO.'}]
            eval_response = client.chat.completions.create(model=model, messages=messages, temperature=0.1, max_tokens=DEFAULT_MAX_TOKENS)
            result_text = eval_response.choices[0].message.content
            final_result, eval_thinking = extract_thinking(result_text)
            final_result = final_result.strip().upper()
            logger.debug(f"Strategy evaluation - raw response: '{result_text}'")
            logger.debug(f"Strategy evaluation - final result after removing thinking: '{final_result}'")
            is_effective = 'YES' in final_result
            results[strategy.strategy_id] = is_effective
            logger.info(f'Strategy {strategy.strategy_id} evaluation: {final_result} -> {is_effective}')
    except Exception as e:
        logger.error(f'Error evaluating strategy effectiveness: {str(e)}')
        for strategy in selected_strategies:
            results[strategy.strategy_id] = False
    return results

def extract_thinking(response: str) -> Tuple[str, Optional[str]]:
    """
    Extract thinking content from <think>...</think> tags and the response after.
    
    Args:
        response: The model's response
    
    Returns:
        Tuple[str, Optional[str]]: The cleaned response and the thinking content (if any)
    """
    thinking_content = None
    final_response = response
    think_pattern = '<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, response, re.DOTALL)
    if think_matches:
        thinking_content = '\n'.join(think_matches)
        final_parts = response.split('</think>')
        if len(final_parts) > 1:
            final_response = final_parts[-1].strip()
    return (final_response, thinking_content)

def refine_strategy(strategy: Strategy, problem: str, response: str, thinking: Optional[str], client, model: str) -> Strategy:
    """
    Refine a strategy based on its application to a specific problem.
    
    Args:
        strategy: The strategy to refine
        problem: The problem that was solved
        response: The LLM's final response to the problem
        thinking: The LLM's reasoning process (if any)
        client: LLM client for making API calls
        model: Model identifier
    
    Returns:
        Strategy: The refined strategy
    """
    try:
        full_response = thinking + '\n\n' + response if thinking else response
        messages = [{'role': 'system', 'content': STRATEGY_REFINEMENT_PROMPT}, {'role': 'user', 'content': f'Original strategy for {strategy.problem_type} problems:\n{strategy.strategy_text}\n\nNew problem:\n{problem}\n\nSolution process (including reasoning):\n{full_response}\n\nProvide a refined version of the original strategy that incorporates any insights from this new example.'}]
        refine_response = client.chat.completions.create(model=model, messages=messages, temperature=0.5, max_tokens=DEFAULT_MAX_TOKENS)
        response_text = refine_response.choices[0].message.content
        refined_text, refinement_thinking = extract_thinking(response_text)
        if not refined_text.strip():
            refined_text = response_text
        logger.debug(f"Strategy refinement - raw response: '{response_text}'")
        logger.debug(f"Strategy refinement - final text after removing thinking: '{refined_text}'")
        refined_strategy = Strategy(strategy_id=strategy.strategy_id, problem_type=strategy.problem_type, strategy_text=refined_text.strip(), examples=strategy.examples + [problem], success_count=strategy.success_count, total_attempts=strategy.total_attempts, created_at=strategy.created_at, last_used=datetime.now().isoformat(), last_updated=datetime.now().isoformat(), confidence=strategy.confidence, tags=strategy.tags, reasoning_examples=strategy.reasoning_examples.copy())
        if refinement_thinking:
            refined_strategy.add_reasoning_example(refinement_thinking)
        return refined_strategy
    except Exception as e:
        logger.error(f'Error refining strategy: {str(e)}')
        return strategy

def run_spl(system_prompt: str, initial_query: str, client, model: str, request_config: dict=None) -> Tuple[str, int]:
    """
    Main plugin function that implements system prompt learning.
    
    By default, the plugin runs in inference-only mode, which uses existing strategies without modifying them.
    Setting request_config['spl_learning'] = True enables learning mode to create and refine strategies.
    
    Args:
        system_prompt: The system prompt
        initial_query: The user's query
        client: The LLM client
        model: The model identifier
        request_config: Optional request configuration
                       Can include {'spl_learning': True} to enable learning mode
    
    Returns:
        Tuple[str, int]: The LLM response and token count
    """
    start_time = time.time()
    logger.info(f'Starting SPL plugin execution for query: {initial_query[:100]}...')
    learning_mode = False
    if request_config and 'spl_learning' in request_config:
        learning_mode = request_config['spl_learning']
        logger.info(f'Running in learning mode: {learning_mode}')
    db = StrategyDatabase()
    logger.info(f'Current strategy count: {len(db.strategies)}')
    logger.info(f'Last strategy ID: {db.metrics.get('last_strategy_id', 0)}')
    if learning_mode:
        db.increment_query_count()
        db._save()
    problem_type = classify_problem(initial_query, client, model)
    logger.info(f'Classified problem as: {problem_type}')
    existing_strategies = db.get_strategies_for_problem(problem_type)
    logger.info(f'Found {len(existing_strategies)} existing strategies for {problem_type}')
    similar_strategy = None
    if learning_mode:
        should_create, similar_strategy = should_create_new_strategy(problem_type, initial_query, existing_strategies, db)
        if should_create:
            logger.info(f'Creating new strategy for {problem_type}')
            new_strategy = generate_strategy(initial_query, problem_type, client, model, db)
            db.add_strategy(new_strategy)
            logger.info(f'Added new strategy with ID: {new_strategy.strategy_id}')
        elif similar_strategy:
            logger.info(f'Updating existing strategy {similar_strategy.strategy_id} with new example')
            db.add_example_to_strategy(similar_strategy.strategy_id, initial_query)
    if learning_mode and db.metrics['total_queries'] % MAINTENANCE_INTERVAL == 0:
        merged_count = db.merge_similar_strategies(similarity_threshold=STRATEGY_MERGING_THRESHOLD)
        logger.info(f'Merged {merged_count} similar strategies')
        limited_count = db.limit_strategies_per_type(max_per_type=MAX_STRATEGIES_PER_TYPE)
        pruned_count = db.prune_strategies()
        logger.info(f'Pruned {pruned_count} low-performing strategies')
    existing_strategies = db.get_strategies_for_problem(problem_type)
    selected_strategies = select_relevant_strategies(initial_query, problem_type, db, learning_mode, MAX_STRATEGIES_FOR_INFERENCE)
    for i, strategy in enumerate(selected_strategies, 1):
        logger.info(f'Selected strategy {i}/{MAX_STRATEGIES_FOR_INFERENCE} for inference: {strategy.strategy_id} (success rate: {strategy.success_rate:.2f})')
    if not selected_strategies:
        if not existing_strategies:
            logger.info(f"No strategies exist for problem type '{problem_type}'. Enable learning mode with 'spl_learning=True' to create strategies.")
        else:
            logger.info(f"Strategies exist for problem type '{problem_type}' but none meet the minimum success rate threshold of {MIN_SUCCESS_RATE_FOR_INFERENCE:.2f}.")
            logger.info(f"Enable learning mode with 'spl_learning=True' to improve strategies.")
        logger.info('Running without strategy augmentation - using base system prompt only.')
        augmented_prompt = system_prompt
    else:
        augmented_prompt = augment_system_prompt(system_prompt, selected_strategies)
        logger.info(f'Augmented system prompt with {len(selected_strategies)} strategies (inference limit: {MAX_STRATEGIES_FOR_INFERENCE})')
    try:
        request_params = {}
        if request_config:
            request_params = {k: v for k, v in request_config.items() if k != 'spl_learning'}
        if 'max_tokens' not in request_params:
            request_params['max_tokens'] = DEFAULT_MAX_TOKENS
        elif request_params['max_tokens'] < DEFAULT_MAX_TOKENS:
            request_params['max_tokens'] = DEFAULT_MAX_TOKENS
        response = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': augmented_prompt}, {'role': 'user', 'content': initial_query}], **request_params)
        completion_tokens = response.usage.completion_tokens
        response_text = response.choices[0].message.content
        final_response, thinking = extract_thinking(response_text)
        logger.debug(f"Main response - raw: '{response_text}'")
        if thinking:
            logger.debug(f"Main response - thinking extracted: '{thinking}'")
            logger.debug(f"Main response - final answer after removing thinking: '{final_response}'")
        if learning_mode:
            if selected_strategies:
                strategy_effectiveness = evaluate_strategy_effectiveness(final_response, thinking, selected_strategies, client, model)
                for strategy_id, effective in strategy_effectiveness.items():
                    if strategy_id != 'fallback_temporary':
                        db.update_strategy_performance(strategy_id, effective)
                        logger.info(f'Strategy {strategy_id} effectiveness: {effective}')
                        if effective and thinking and (strategy_id != 'fallback_temporary'):
                            db.add_reasoning_example(strategy_id, thinking)
                            logger.info(f'Added reasoning example to strategy {strategy_id}')
                for strategy in selected_strategies:
                    if strategy.strategy_id != 'fallback_temporary' and strategy.total_attempts % 10 == 0 and (strategy.total_attempts > 0):
                        logger.info(f'Refining strategy {strategy.strategy_id} after {strategy.total_attempts} attempts')
                        refined_strategy = refine_strategy(strategy, initial_query, final_response, thinking, client, model)
                        db.refine_strategy(strategy.strategy_id, refined_strategy.strategy_text)
            else:
                logger.info('No strategies to evaluate or refine - consider adding strategies for this problem type')
        else:
            logger.info('Strategy evaluation and refinement skipped (not in learning mode)')
        execution_time = time.time() - start_time
        logger.info(f'SPL plugin execution completed in {execution_time:.2f} seconds')
        logger.info(f'Final strategy count: {len(db.strategies)}')
        logger.info(f'Final last strategy ID: {db.metrics.get('last_strategy_id', 0)}')
        return (response_text, completion_tokens)
    except Exception as e:
        logger.error(f'Error in SPL plugin: {str(e)}')
        try:
            response = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': initial_query}], max_tokens=DEFAULT_MAX_TOKENS)
            return (response.choices[0].message.content, response.usage.completion_tokens)
        except Exception as inner_e:
            logger.error(f'Error in fallback completion: {str(inner_e)}')
            return (f'Error processing request: {str(e)}', 0)

def select_relevant_strategies(query: str, problem_type: str, db: Any, learning_mode: bool=False, max_strategies: int=MAX_STRATEGIES_FOR_INFERENCE) -> List[Strategy]:
    """
    Select the most relevant strategies for a given problem to be used during inference.
    This controls how many strategies are included in the system prompt augmentation.
    
    When in inference mode (not learning_mode), only strategies with:
     - A matching problem type 
     - Success rate >= MIN_SUCCESS_RATE_FOR_INFERENCE
     - At least 5 attempts
    are selected.
    
    In learning mode, strategies with fewer attempts are also considered.
    
    Args:
        query: The problem/query text
        problem_type: The type of problem
        db: Strategy database
        learning_mode: Whether we're in learning mode (affects filtering criteria)
        max_strategies: Maximum number of strategies to return
    
    Returns:
        List[Strategy]: The selected strategies (may be empty if none meet criteria)
    """
    type_specific = db.get_strategies_for_problem(problem_type)
    logger.info(f"Found {len(type_specific)} strategies for problem type '{problem_type}'")
    qualified_strategies = []
    for strategy in type_specific:
        if learning_mode and strategy.total_attempts < 5:
            logger.info(f'Strategy {strategy.strategy_id} included (learning mode - only {strategy.total_attempts} attempts so far)')
            qualified_strategies.append(strategy)
        elif strategy.success_rate >= MIN_SUCCESS_RATE_FOR_INFERENCE and strategy.total_attempts >= 5:
            logger.info(f'Strategy {strategy.strategy_id} qualified - success rate {strategy.success_rate:.2f} >= minimum {MIN_SUCCESS_RATE_FOR_INFERENCE:.2f} with {strategy.total_attempts} attempts')
            qualified_strategies.append(strategy)
        elif strategy.total_attempts < 5:
            logger.info(f'Strategy {strategy.strategy_id} skipped - insufficient attempts ({strategy.total_attempts} < 5) in inference mode')
        else:
            logger.info(f'Strategy {strategy.strategy_id} skipped - success rate {strategy.success_rate:.2f} < minimum {MIN_SUCCESS_RATE_FOR_INFERENCE:.2f}')
    if not qualified_strategies:
        logger.info(f"No strategies meet the minimum success rate threshold ({MIN_SUCCESS_RATE_FOR_INFERENCE:.2f}) for problem type '{problem_type}'")
        return []
    logger.info(f'Found {len(qualified_strategies)} strategies that meet minimum success rate requirement')
    if len(qualified_strategies) > max_strategies:
        scored_strategies = []
        for strategy in qualified_strategies:
            recency_score = 0
            if strategy.last_used:
                last_used = datetime.fromisoformat(strategy.last_used)
                days_since = (datetime.now() - last_used).days
                recency_score = max(0, 1.0 - min(1.0, days_since / 30.0))
            score = 0.7 * strategy.success_rate + 0.3 * recency_score
            scored_strategies.append((strategy, score))
        scored_strategies.sort(key=lambda x: x[1], reverse=True)
        selected = [s[0] for s in scored_strategies[:max_strategies]]
        for i, strategy in enumerate(selected, 1):
            logger.info(f'Selected strategy {i}/{max_strategies} for inference: {strategy.strategy_id} (success rate: {strategy.success_rate:.2f})')
        return selected
    for i, strategy in enumerate(qualified_strategies, 1):
        logger.info(f'Selected strategy {i}/{len(qualified_strategies)} for inference: {strategy.strategy_id} (success rate: {strategy.success_rate:.2f})')
    return qualified_strategies

def classify_problem(content: str, client, model: str) -> str:
    """
    Use the LLM to classify the problem type, ensuring the result is one of the valid types.
    
    Args:
        content: The query/problem to classify
        client: LLM client for making API calls
        model: Model identifier
    
    Returns:
        str: The problem type classification (always a valid type)
    """
    problem_types_str = ', '.join(VALID_PROBLEM_TYPES[:-1])
    try:
        messages = [{'role': 'system', 'content': PROBLEM_CLASSIFICATION_PROMPT.format(problem_types=problem_types_str)}, {'role': 'user', 'content': f'Classify the following problem into ONE of these types: {problem_types_str}\n\nProblem: {content}'}]
        response = client.chat.completions.create(model=model, messages=messages, temperature=0.1, max_tokens=DEFAULT_MAX_TOKENS)
        raw_response = response.choices[0].message.content
        final_response, thinking = extract_thinking(raw_response)
        final_response = final_response.strip().lower()
        logger.debug(f"Problem classification - raw response: '{raw_response}'")
        logger.debug(f"Problem classification - final response after removing thinking: '{final_response}'")
        for valid_type in VALID_PROBLEM_TYPES:
            if valid_type.lower() == final_response:
                logger.info(f"Classified problem as '{valid_type}' (exact match)")
                return valid_type
        for valid_type in VALID_PROBLEM_TYPES:
            if valid_type.lower() in final_response:
                logger.info(f"Classified problem as '{valid_type}' (partial match from '{final_response}')")
                return valid_type
        logger.warning(f"Could not match '{final_response}' to any valid problem type, using 'general_problem'")
        return 'general_problem'
    except Exception as e:
        logger.error(f'Error classifying problem: {str(e)}')
        return 'general_problem'

def generate_strategy(problem: str, problem_type: str, client, model: str, db: StrategyDatabase) -> Strategy:
    """
    Generate a new problem-solving strategy using the LLM.
    
    Args:
        problem: The problem that needs a strategy
        problem_type: The type of problem
        client: LLM client for making API calls
        model: Model identifier
        db: The strategy database to use for generating IDs
    
    Returns:
        Strategy: A new strategy for solving this type of problem
    """
    try:
        messages = [{'role': 'system', 'content': STRATEGY_GENERATION_PROMPT}, {'role': 'user', 'content': f'Create a problem-solving strategy for the following {problem_type} problem:\n\n{problem}\n\nThis strategy should help solve not just this specific problem, but any {problem_type} problem.'}]
        response = client.chat.completions.create(model=model, messages=messages, temperature=0.7, max_tokens=DEFAULT_MAX_TOKENS)
        response_text = response.choices[0].message.content
        strategy_text, thinking = extract_thinking(response_text)
        if not strategy_text.strip():
            strategy_text = response_text
        logger.debug(f"Generated strategy - raw response: '{response_text}'")
        logger.debug(f"Generated strategy - final text after removing thinking: '{strategy_text}'")
        strategy = Strategy(strategy_id=db.get_next_strategy_id(), problem_type=problem_type, strategy_text=strategy_text.strip(), examples=[problem], created_at=None, reasoning_examples=[thinking] if thinking else [])
        logger.info(f'Generated new strategy for {problem_type}: ID {strategy.strategy_id}')
        return strategy
    except Exception as e:
        logger.error(f'Error generating strategy: {str(e)}')
        fallback_id = f'fallback_{uuid.uuid4().hex[:8]}'
        logger.info(f'Using fallback strategy with ID: {fallback_id}')
        return Strategy(strategy_id=fallback_id, problem_type=problem_type, strategy_text=f'When solving {problem_type} problems:\n1. Break down the problem into smaller parts\n2. Solve each part systematically\n3. Combine the solutions', examples=[problem])

