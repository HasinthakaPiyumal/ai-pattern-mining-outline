# Cluster 30

def create_comparison_prompt(candidates: List[str], query: str, comparison_mode: str='batch') -> str:
    """
    Create a prompt for comparing candidate solutions.
    
    Args:
        candidates: List of candidate responses
        query: The original user query
        comparison_mode: "batch" for all at once, "tournament" for pairwise
        
    Returns:
        The comparison prompt
    """
    if comparison_mode == 'batch':
        prompt = f'You are an expert evaluator tasked with selecting the best response to the following query:\n\nQuery: {query}\n\nI will provide you with {len(candidates)} different candidate responses. Please analyze each one carefully and select the best response based on the following criteria:\n\n1. **Correctness and Accuracy**: Is the response factually correct and accurate?\n2. **Completeness**: Does it fully address all aspects of the query?\n3. **Clarity**: Is the explanation clear and easy to understand?\n4. **Logical Coherence**: Is the reasoning sound and well-structured?\n5. **Practical Value**: Does it provide useful, actionable information?\n\nFor coding problems, also consider:\n- Code correctness and efficiency\n- Best practices and style\n- Error handling\n\nHere are the candidate responses:\n\n'
        for i, candidate in enumerate(candidates, 1):
            prompt += f'=== Candidate {i} ===\n{candidate}\n\n'
        prompt += 'Please analyze all candidates and provide:\n1. A brief comparison highlighting the strengths and weaknesses of each candidate\n2. Your selection of the best candidate (specify the number)\n3. A clear explanation of why you selected that candidate\n\nFormat your response as:\nCOMPARISON:\n[Your comparison analysis]\n\nBEST CANDIDATE: [number]\n\nREASONING:\n[Your explanation for the selection]'
    else:
        return create_comparison_prompt(candidates, query, 'batch')
    return prompt

def run(system_prompt: str, initial_query: str, client, model: str, request_config: Dict[str, Any]=None) -> Tuple[str, int]:
    """
    Main entry point for the GenSelect plugin.
    
    Generates multiple candidate solutions and uses LLM comparison to select the best one.
    
    Args:
        system_prompt: System prompt for the model
        initial_query: User's query
        client: OpenAI-compatible client instance
        model: Model identifier
        request_config: Additional configuration parameters
        
    Returns:
        Tuple of (response_text, completion_tokens_used)
    """
    logger.info('Starting GenSelect process')
    config = request_config or {}
    num_candidates = config.get('num_candidates', DEFAULT_NUM_CANDIDATES)
    temperature = config.get('temperature', DEFAULT_TEMPERATURE)
    comparison_temperature = config.get('comparison_temperature', DEFAULT_COMPARISON_TEMPERATURE)
    comparison_mode = config.get('comparison_mode', DEFAULT_COMPARISON_MODE)
    include_reasoning = config.get('include_reasoning', DEFAULT_INCLUDE_REASONING)
    max_tokens = config.get('max_tokens', 4096)
    num_candidates = max(2, num_candidates)
    logger.info(f'Generating {num_candidates} candidates with temperature={temperature}')
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': initial_query}]
    candidates = []
    total_tokens = 0
    try:
        response = client.chat.completions.create(model=model, messages=messages, n=num_candidates, temperature=temperature, max_tokens=max_tokens)
        candidates = [choice.message.content for choice in response.choices]
        total_tokens += response.usage.completion_tokens
        logger.info(f'Generated {len(candidates)} candidates using n parameter. Tokens: {total_tokens}')
    except Exception as e:
        logger.warning(f'n parameter not supported: {str(e)}')
        logger.info('Falling back to sequential generation')
        for i in range(num_candidates):
            try:
                response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                candidates.append(response.choices[0].message.content)
                total_tokens += response.usage.completion_tokens
                logger.debug(f'Generated candidate {i + 1}/{num_candidates}')
            except Exception as gen_error:
                logger.error(f'Error generating candidate {i + 1}: {str(gen_error)}')
                continue
    if len(candidates) < 2:
        logger.error(f'Insufficient candidates generated ({len(candidates)})')
        if candidates:
            return (candidates[0], total_tokens)
        return ('Error: Could not generate sufficient candidates for selection', total_tokens)
    comparison_prompt = create_comparison_prompt(candidates, initial_query, comparison_mode)
    logger.info('Comparing candidates for selection')
    try:
        comparison_messages = [{'role': 'system', 'content': 'You are an expert evaluator skilled at comparing and selecting high-quality responses.'}, {'role': 'user', 'content': comparison_prompt}]
        comparison_response = client.chat.completions.create(model=model, messages=comparison_messages, temperature=comparison_temperature, max_tokens=2048)
        selection_response = comparison_response.choices[0].message.content
        total_tokens += comparison_response.usage.completion_tokens
        selected_index, reasoning = parse_selection_response(selection_response, len(candidates))
        selected_candidate = candidates[selected_index]
        logger.info(f'GenSelect Summary:')
        logger.info(f'  - Generated {len(candidates)} candidates')
        logger.info(f'  - Selected candidate {selected_index + 1}')
        logger.info(f'  - Total tokens used: {total_tokens}')
        if include_reasoning:
            final_response = f'{selected_candidate}\n\n---\n**GenSelect Reasoning**: {reasoning}'
        else:
            final_response = selected_candidate
        return (final_response, total_tokens)
    except Exception as e:
        logger.error(f'Error during comparison: {str(e)}')
        logger.warning('Falling back to first candidate due to comparison error')
        return (candidates[0], total_tokens)

def parse_selection_response(response: str, num_candidates: int) -> Tuple[int, str]:
    """
    Parse the selection response to extract the chosen candidate number and reasoning.
    
    Args:
        response: The LLM's comparison response
        num_candidates: Total number of candidates
        
    Returns:
        Tuple of (selected_index, reasoning)
    """
    import re
    match = re.search('BEST CANDIDATE:\\s*(\\d+)', response, re.IGNORECASE)
    if match:
        candidate_num = int(match.group(1))
        selected_index = candidate_num - 1
        if 0 <= selected_index < num_candidates:
            reasoning_match = re.search('REASONING:\\s*(.+)', response, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else 'No explicit reasoning provided'
            logger.info(f'Selected candidate {candidate_num} based on comparison')
            return (selected_index, reasoning)
    patterns = ['[Cc]andidate\\s+(\\d+)\\s+is\\s+(?:the\\s+)?best', '[Ii]\\s+(?:would\\s+)?select\\s+[Cc]andidate\\s+(\\d+)', '[Tt]he\\s+best\\s+(?:response|candidate)\\s+is\\s+(?:number\\s+)?(\\d+)']
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            candidate_num = int(match.group(1))
            selected_index = candidate_num - 1
            if 0 <= selected_index < num_candidates:
                logger.info(f'Selected candidate {candidate_num} using fallback pattern')
                return (selected_index, 'Selection extracted from response pattern')
    logger.warning('Could not parse selection from comparison response, defaulting to first candidate')
    return (0, 'Failed to parse selection, defaulted to first candidate')

