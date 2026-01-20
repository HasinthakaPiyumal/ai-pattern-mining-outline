# Cluster 16

def best_of_n_sampling(system_prompt: str, initial_query: str, client, model: str, n: int=3, request_id: str=None) -> str:
    bon_completion_tokens = 0
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': initial_query}]
    completions = []
    try:
        provider_request = {'model': model, 'messages': messages, 'max_tokens': 4096, 'n': n, 'temperature': 1}
        response = client.chat.completions.create(**provider_request)
        if request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            conversation_logger.log_provider_call(request_id, provider_request, response_dict)
        completions = [choice.message.content for choice in response.choices]
        logger.info(f'Generated {len(completions)} initial completions using n parameter. Tokens used: {response.usage.completion_tokens}')
        bon_completion_tokens += response.usage.completion_tokens
    except Exception as e:
        logger.warning(f'n parameter not supported by provider: {str(e)}')
        logger.info(f'Falling back to generating {n} completions one by one')
        for i in range(n):
            try:
                provider_request = {'model': model, 'messages': messages, 'max_tokens': 4096, 'temperature': 1}
                response = client.chat.completions.create(**provider_request)
                if request_id:
                    response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
                    conversation_logger.log_provider_call(request_id, provider_request, response_dict)
                completions.append(response.choices[0].message.content)
                bon_completion_tokens += response.usage.completion_tokens
                logger.debug(f'Generated completion {i + 1}/{n}')
            except Exception as fallback_error:
                logger.error(f'Error generating completion {i + 1}: {str(fallback_error)}')
                continue
        if not completions:
            logger.error('Failed to generate any completions')
            return ('Error: Could not generate any completions', 0)
        logger.info(f'Generated {len(completions)} completions using fallback method. Total tokens used: {bon_completion_tokens}')
    rating_messages = messages.copy()
    rating_messages.append({'role': 'system', 'content': 'Rate the following responses on a scale from 0 to 10, where 0 is poor and 10 is excellent. Consider factors such as relevance, coherence, and helpfulness. Respond with only a number.'})
    ratings = []
    for completion in completions:
        rating_messages.append({'role': 'assistant', 'content': completion})
        rating_messages.append({'role': 'user', 'content': 'Rate the above response:'})
        provider_request = {'model': model, 'messages': rating_messages, 'max_tokens': 256, 'n': 1, 'temperature': 0.1}
        rating_response = client.chat.completions.create(**provider_request)
        if request_id:
            response_dict = rating_response.model_dump() if hasattr(rating_response, 'model_dump') else rating_response
            conversation_logger.log_provider_call(request_id, provider_request, response_dict)
        bon_completion_tokens += rating_response.usage.completion_tokens
        try:
            rating = float(rating_response.choices[0].message.content.strip())
            ratings.append(rating)
        except ValueError:
            ratings.append(0)
        rating_messages = rating_messages[:-2]
    best_index = ratings.index(max(ratings))
    return (completions[best_index], bon_completion_tokens)

def round_trip_optimization(system_prompt: str, initial_query: str, client, model: str, request_id: str=None) -> str:
    rto_completion_tokens = 0
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': initial_query}]
    provider_request = {'model': model, 'messages': messages, 'max_tokens': 4096, 'n': 1, 'temperature': 0.1}
    response_c1 = client.chat.completions.create(**provider_request)
    if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
        response_dict = response_c1.model_dump() if hasattr(response_c1, 'model_dump') else response_c1
        optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)
    c1 = response_c1.choices[0].message.content
    rto_completion_tokens += response_c1.usage.completion_tokens
    messages.append({'role': 'assistant', 'content': c1})
    messages.append({'role': 'user', 'content': 'Summarize or describe the code you just created. The summary should be in form of an instruction such that, given the instruction you can create the code yourself.'})
    provider_request = {'model': model, 'messages': messages, 'max_tokens': 1024, 'n': 1, 'temperature': 0.1}
    response_q2 = client.chat.completions.create(**provider_request)
    if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
        response_dict = response_q2.model_dump() if hasattr(response_q2, 'model_dump') else response_q2
        optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)
    q2 = response_q2.choices[0].message.content
    rto_completion_tokens += response_q2.usage.completion_tokens
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': q2}]
    provider_request = {'model': model, 'messages': messages, 'max_tokens': 4096, 'n': 1, 'temperature': 0.1}
    response_c2 = client.chat.completions.create(**provider_request)
    if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
        response_dict = response_c2.model_dump() if hasattr(response_c2, 'model_dump') else response_c2
        optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)
    c2 = response_c2.choices[0].message.content
    rto_completion_tokens += response_c2.usage.completion_tokens
    c1 = extract_code_from_prompt(c1)
    c2 = extract_code_from_prompt(c2)
    if c1.strip() == c2.strip():
        return (c1, rto_completion_tokens)
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': f'Initial query: {initial_query}\n\nFirst generated code (C1):\n{c1}\n\nSecond generated code (C2):\n{c2}\n\nBased on the initial query and these two different code implementations, generate a final, optimized version of the code. Only respond with the final code, do not return anything else.'}]
    provider_request = {'model': model, 'messages': messages, 'max_tokens': 4096, 'n': 1, 'temperature': 0.1}
    response_c3 = client.chat.completions.create(**provider_request)
    if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
        response_dict = response_c3.model_dump() if hasattr(response_c3, 'model_dump') else response_c3
        optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)
    c3 = response_c3.choices[0].message.content
    rto_completion_tokens += response_c3.usage.completion_tokens
    return (c3, rto_completion_tokens)

def extract_code_from_prompt(text):
    pattern = '```(?:[\\w-]+)?\\n(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        logger.warning('Could not extract code from prompt. Returning original text.')
        return text

def execute_single_approach(approach, system_prompt, initial_query, client, model, request_config: dict=None, request_id: str=None):
    if approach in known_approaches:
        if approach == 'none':
            kwargs = request_config.copy() if request_config else {}
            kwargs.pop('n', None)
            kwargs.pop('stream', None)
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            if initial_query:
                messages.append({'role': 'user', 'content': initial_query})
            response = none_approach(original_messages=messages, client=client, model=model, request_id=request_id, **kwargs)
            return (response, 0)
        elif approach == 'mcts':
            return chat_with_mcts(system_prompt, initial_query, client, model, server_config['mcts_simulations'], server_config['mcts_exploration'], server_config['mcts_depth'], request_id)
        elif approach == 'bon':
            return best_of_n_sampling(system_prompt, initial_query, client, model, server_config['best_of_n'], request_id)
        elif approach == 'moa':
            return mixture_of_agents(system_prompt, initial_query, client, model, request_id)
        elif approach == 'rto':
            return round_trip_optimization(system_prompt, initial_query, client, model, request_id)
        elif approach == 'z3':
            z3_solver = Z3SymPySolverSystem(system_prompt, client, model, request_id=request_id)
            return z3_solver.process_query(initial_query)
        elif approach == 'self_consistency':
            return advanced_self_consistency_approach(system_prompt, initial_query, client, model, request_id)
        elif approach == 'pvg':
            return inference_time_pv_game(system_prompt, initial_query, client, model, request_id)
        elif approach == 'rstar':
            rstar = RStar(system_prompt, client, model, max_depth=server_config['rstar_max_depth'], num_rollouts=server_config['rstar_num_rollouts'], c=server_config['rstar_c'], request_id=request_id)
            return rstar.solve(initial_query)
        elif approach == 'cot_reflection':
            return cot_reflection(system_prompt, initial_query, client, model, return_full_response=server_config['return_full_response'], request_config=request_config, request_id=request_id)
        elif approach == 'plansearch':
            return plansearch(system_prompt, initial_query, client, model, n=server_config['n'], request_id=request_id)
        elif approach == 'leap':
            return leap(system_prompt, initial_query, client, model, request_id)
        elif approach == 're2':
            return re2_approach(system_prompt, initial_query, client, model, n=server_config['n'], request_id=request_id)
        elif approach == 'cepo':
            return cepo(system_prompt, initial_query, client, model, cepo_config, request_id)
        elif approach == 'mars':
            return multi_agent_reasoning_system(system_prompt, initial_query, client, model, request_config=request_config, request_id=request_id)
    elif approach in plugin_approaches:
        plugin_func = plugin_approaches[approach]
        import inspect
        sig = inspect.signature(plugin_func)
        is_async = inspect.iscoroutinefunction(plugin_func)
        if is_async:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                if 'request_config' in sig.parameters:
                    result = loop.run_until_complete(plugin_func(system_prompt, initial_query, client, model, request_config=request_config))
                else:
                    result = loop.run_until_complete(plugin_func(system_prompt, initial_query, client, model))
                return result
            finally:
                loop.close()
        elif 'request_config' in sig.parameters:
            return plugin_func(system_prompt, initial_query, client, model, request_config=request_config)
        else:
            return plugin_func(system_prompt, initial_query, client, model)
    else:
        raise ValueError(f'Unknown approach: {approach}')

def chat_with_mcts(system_prompt: str, initial_query: str, client, model: str, num_simulations: int=2, exploration_weight: float=0.2, simulation_depth: int=1, request_id: str=None) -> str:
    logger.info('Starting chat with MCTS')
    logger.info(f'Parameters: num_simulations={num_simulations}, exploration_weight={exploration_weight}, simulation_depth={simulation_depth}')
    mcts = MCTS(simulation_depth=simulation_depth, exploration_weight=exploration_weight, client=client, model=model, request_id=request_id)
    initial_state = DialogueState(system_prompt, [], initial_query)
    logger.info(f'Initial query: {initial_query}')
    final_state = mcts.search(initial_state, num_simulations)
    response = final_state.conversation_history[-1]['content'] if final_state.conversation_history else ''
    logger.info(f'MCTS chat complete. Final response: {response[:100]}...')
    return (response, mcts.completion_tokens)

def advanced_self_consistency_approach(system_prompt: str, initial_query: str, client, model: str, request_id: str=None) -> str:
    self_consistency = AdvancedSelfConsistency(client, model, request_id=request_id)
    result = self_consistency.evaluate(system_prompt, initial_query)
    logger.info('Advanced Self-Consistency Results:')
    logger.info(f'Total responses: {result['aggregated_result']['total_responses']}')
    logger.info(f'Number of unique clusters: {result['aggregated_result']['num_unique_clusters']}')
    for i, cluster in enumerate(result['aggregated_result']['clusters'], 1):
        logger.debug(f'\nCluster {i}:')
        logger.debug(f'  Representative answer: {cluster['answer']}')
        logger.debug(f'  Frequency: {cluster['frequency']}')
        logger.debug(f'  Variants: {cluster['variants']}')
    if result['aggregated_result']['clusters']:
        return (result['aggregated_result']['clusters'][0]['answer'], self_consistency.self_consistency_completion_tokens)
    else:
        return ('No consistent answer found.', self_consistency.self_consistency_completion_tokens)

def inference_time_pv_game(system_prompt: str, initial_query: str, client, model: str, num_rounds: int=2, num_solutions: int=3, request_id: str=None) -> str:
    global pvg_completion_tokens
    logger.info(f'Starting inference-time PV game with {num_rounds} rounds and {num_solutions} solutions per round')
    best_solution = ''
    best_score = -1
    for round in range(num_rounds):
        logger.info(f'Starting round {round + 1}')
        temperature = max(0.2, 0.7 - round * 0.1)
        helpful_solutions = generate_solutions(client, system_prompt, initial_query, model, num_solutions, temperature=temperature, request_id=request_id)
        sneaky_solutions = generate_solutions(client, system_prompt, initial_query, model, num_solutions, is_sneaky=True, temperature=temperature, request_id=request_id)
        all_solutions = helpful_solutions + sneaky_solutions
        scores = verify_solutions(client, system_prompt, initial_query, all_solutions, model, request_id=request_id)
        round_best_solution = max(zip(all_solutions, scores), key=lambda x: x[1])
        if round_best_solution[1] > best_score:
            best_solution = round_best_solution[0]
            best_score = round_best_solution[1]
            logger.info(f'New best solution found in round {round + 1} with score {best_score}')
        else:
            logger.debug(f'No improvement in round {round + 1}. Best score remains {best_score}')
        if round < num_rounds - 1:
            logger.debug('Refining query for next round')
            refine_prompt = f'\n            Based on the original query and the best solution so far, suggest a refined query that might lead to an even better solution.\n            Focus on aspects of the problem that were challenging or not fully addressed in the best solution.\n            Maintain the core intent of the original query while adding specificity or context that could improve the solution.\n            \n            Original query: {initial_query}\n            \n            Best solution so far: {best_solution}\n            \n            Refined query:\n            '
            messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': refine_prompt}]
            provider_request = {'model': model, 'messages': messages, 'max_tokens': 1024, 'temperature': 0.5}
            response = client.chat.completions.create(**provider_request)
            if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
                optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)
            pvg_completion_tokens += response.usage.completion_tokens
            initial_query = response.choices[0].message.content
            logger.debug(f'Refined query: {initial_query}')
    logger.info(f'Inference-time PV game completed. Best solution score: {best_score}')
    return (best_solution, pvg_completion_tokens)

def cot_reflection(system_prompt, initial_query, client, model: str, return_full_response: bool=False, request_config: dict=None, request_id: str=None):
    cot_completion_tokens = 0
    temperature = 0.6
    max_tokens = 4096
    if request_config:
        temperature = request_config.get('temperature', temperature)
        max_tokens = request_config.get('max_tokens', max_tokens)
    cot_prompt = f'\n        {system_prompt}\n\n        You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries. Follow these steps:\n\n        1. Think through the problem step by step within the <thinking> tags.\n        2. Reflect on your thinking to check for any errors or improvements within the <reflection> tags.\n        3. Make any necessary adjustments based on your reflection.\n        4. Provide your final, concise answer within the <output> tags.\n\n        Important: The <thinking> and <reflection> sections are for your internal reasoning process only. \n        Do not include any part of the final answer in these sections. \n        The actual response to the query must be entirely contained within the <output> tags.\n\n        Use the following format for your response:\n        <thinking>\n        [Your step-by-step reasoning goes here. This is your internal thought process, not the final answer.]\n        <reflection>\n        [Your reflection on your reasoning, checking for errors or improvements]\n        </reflection>\n        [Any adjustments to your thinking based on your reflection]\n        </thinking>\n        <output>\n        [Your final, concise answer to the query. This is the only part that will be shown to the user.]\n        </output>\n        '
    provider_request = {'model': model, 'messages': [{'role': 'system', 'content': cot_prompt}, {'role': 'user', 'content': initial_query}], 'temperature': temperature, 'max_tokens': max_tokens}
    response = client.chat.completions.create(**provider_request)
    if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
        optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)
    full_response = response.choices[0].message.content
    cot_completion_tokens += response.usage.completion_tokens
    logger.info(f'CoT with Reflection :\n{full_response}')
    thinking_match = re.search('<thinking>(.*?)</thinking>', full_response, re.DOTALL)
    output_match = re.search('<output>(.*?)(?:</output>|$)', full_response, re.DOTALL)
    thinking = thinking_match.group(1).strip() if thinking_match else 'No thinking process provided.'
    output = output_match.group(1).strip() if output_match else full_response
    logger.info(f'Final output :\n{output}')
    if return_full_response:
        return (full_response, cot_completion_tokens)
    else:
        return (output, cot_completion_tokens)

def plansearch(system_prompt: str, initial_query: str, client, model: str, n: int=1, request_id: str=None) -> List[str]:
    planner = PlanSearch(system_prompt, client, model, request_id)
    return (planner.solve_multiple(initial_query, n), planner.plansearch_completion_tokens)

def leap(system_prompt: str, initial_query: str, client, model: str, request_id: str=None) -> str:
    leap_solver = LEAP(system_prompt, client, model, request_id)
    return (leap_solver.solve(initial_query), leap_solver.leap_completion_tokens)

def re2_approach(system_prompt, initial_query, client, model, n=1, request_id: str=None):
    """
    Implement the RE2 (Re-Reading) approach for improved reasoning in LLMs.
    
    Args:
    system_prompt (str): The system prompt to be used.
    initial_query (str): The initial user query.
    client: The OpenAI client object.
    model (str): The name of the model to use.
    n (int): Number of completions to generate.
    
    Returns:
    str or list: The generated response(s) from the model.
    """
    logger.info('Using RE2 approach for query processing')
    re2_completion_tokens = 0
    re2_prompt = f'{initial_query}\nRead the question again: {initial_query}'
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': re2_prompt}]
    try:
        provider_request = {'model': model, 'messages': messages, 'n': n}
        response = client.chat.completions.create(**provider_request)
        if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)
        re2_completion_tokens += response.usage.completion_tokens
        if n == 1:
            return (response.choices[0].message.content.strip(), re2_completion_tokens)
        else:
            return ([choice.message.content.strip() for choice in response.choices], re2_completion_tokens)
    except Exception as e:
        logger.error(f'Error in RE2 approach: {str(e)}')
        raise

def execute_combined_approaches(approaches, system_prompt, initial_query, client, model, request_config: dict=None):
    final_response = initial_query
    total_tokens = 0
    for approach in approaches:
        response, tokens = execute_single_approach(approach, system_prompt, final_response, client, model, request_config)
        final_response = response
        total_tokens += tokens
    return (final_response, total_tokens)

def execute_n_times(n: int, approaches, operation: str, system_prompt: str, initial_query: str, client: Any, model: str, request_config: dict=None, request_id: str=None) -> Tuple[Union[str, List[str]], int]:
    """
    Execute the pipeline n times and return n responses.
    
    Args:
        n (int): Number of times to run the pipeline
        approaches (list): List of approaches to execute
        operation (str): Operation type ('SINGLE', 'AND', or 'OR')
        system_prompt (str): System prompt
        initial_query (str): Initial query
        client: OpenAI client instance
        model (str): Model identifier
        
    Returns:
        Tuple[Union[str, List[str]], int]: List of responses and total token count
    """
    responses = []
    total_tokens = 0
    for _ in range(n):
        if operation == 'SINGLE':
            response, tokens = execute_single_approach(approaches[0], system_prompt, initial_query, client, model, request_config, request_id)
        elif operation == 'AND':
            response, tokens = execute_combined_approaches(approaches, system_prompt, initial_query, client, model, request_config)
        elif operation == 'OR':
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response, tokens = loop.run_until_complete(execute_parallel_approaches(approaches, system_prompt, initial_query, client, model, request_config))
            loop.close()
        else:
            raise ValueError(f'Unknown operation: {operation}')
        if isinstance(response, list):
            responses.extend(response)
        else:
            responses.append(response)
        total_tokens += tokens
    if n == 1 and len(responses) == 1:
        return (responses[0], total_tokens)
    return (responses, total_tokens)

def process_batch_requests(batch_requests):
    """
            Process a batch of requests using true batching when possible
            
            Args:
                batch_requests: List of request data dictionaries
                
            Returns:
                List of response dictionaries
            """
    import time
    from optillm.batching import BatchingError
    if not batch_requests:
        return []
    logger.info(f'Processing batch of {len(batch_requests)} requests')
    can_use_true_batching = True
    first_req = batch_requests[0]
    for req_data in batch_requests:
        if req_data['stream'] or req_data['approaches'] != first_req['approaches'] or req_data['operation'] != first_req['operation'] or (req_data['model'] != first_req['model']):
            can_use_true_batching = False
            break
    responses = []
    for i, req_data in enumerate(batch_requests):
        try:
            logger.debug(f'Processing batch request {i + 1}/{len(batch_requests)}')
            system_prompt = req_data['system_prompt']
            initial_query = req_data['initial_query']
            client = req_data['client']
            model = req_data['model']
            request_config = req_data['request_config']
            approaches = req_data['approaches']
            operation = req_data['operation']
            n = req_data['n']
            stream = req_data['stream']
            if stream:
                raise BatchingError('Streaming requests cannot be batched')
            contains_none = any((approach == 'none' for approach in approaches))
            if operation == 'SINGLE' and approaches[0] == 'none':
                result, completion_tokens = execute_single_approach(approaches[0], system_prompt, initial_query, client, model, request_config)
            elif operation == 'AND' or operation == 'OR':
                if contains_none:
                    raise ValueError("'none' approach cannot be combined with other approaches")
                result, completion_tokens = execute_n_times(n, approaches, operation, system_prompt, initial_query, client, model, request_config)
            else:
                result, completion_tokens = execute_n_times(n, approaches, operation, system_prompt, initial_query, client, model, request_config)
            if isinstance(result, list):
                processed_response = tagged_conversation_to_messages(result)
                if processed_response != result:
                    result = [msg[-1]['content'] if isinstance(msg, list) and msg else msg for msg in processed_response]
            else:
                messages = tagged_conversation_to_messages(result)
                if isinstance(messages, list) and messages:
                    result = messages[-1]['content']
            if isinstance(result, list):
                choices = []
                for j, res in enumerate(result):
                    choices.append({'index': j, 'message': {'role': 'assistant', 'content': res}, 'finish_reason': 'stop'})
            else:
                choices = [{'index': 0, 'message': {'role': 'assistant', 'content': result}, 'finish_reason': 'stop'}]
            response_dict = {'id': f'chatcmpl-{int(time.time() * 1000)}-{i}', 'object': 'chat.completion', 'created': int(time.time()), 'model': model, 'choices': choices, 'usage': {'prompt_tokens': 0, 'completion_tokens': completion_tokens if isinstance(completion_tokens, int) else 0, 'total_tokens': completion_tokens if isinstance(completion_tokens, int) else 0}}
            responses.append(response_dict)
        except Exception as e:
            logger.error(f'Error processing batch request {i + 1}: {e}')
            raise BatchingError(f'Failed to process request {i + 1}: {str(e)}')
    logger.info(f'Completed batch processing of {len(responses)} requests')
    return responses

def generate_solutions(client, system_prompt: str, query: str, model: str, num_solutions: int, is_sneaky: bool=False, temperature: float=0.7, request_id: str=None) -> List[str]:
    global pvg_completion_tokens
    role = 'sneaky' if is_sneaky else 'helpful'
    logger.info(f'Generating {num_solutions} {role} solutions')
    role_instruction = "\n    You are a problem solver tasked with generating solutions to the given problem. \n    If you are in 'helpful' mode:\n    - Provide a correct, detailed, and step-by-step solution to the problem.\n    - Explain your reasoning clearly and logically.\n    - Ensure your final answer or conclusion is accurate and well-supported.\n    - Use domain-specific terminology or notation if appropriate.\n\n    If you are in 'sneaky' mode:\n    - Provide a solution that appears correct but contains subtle errors or flaws.\n    - Make your explanation seem logical and convincing.\n    - Ensure your final answer or conclusion is incorrect, but in a way that's not immediately obvious.\n    - Maintain the appearance of expertise in the relevant domain.\n    "
    messages = [{'role': 'system', 'content': f'{system_prompt}\n{role_instruction}\nYou are in {role} mode.'}, {'role': 'user', 'content': query}]
    provider_request = {'model': model, 'messages': messages, 'n': num_solutions, 'max_tokens': 4096, 'temperature': temperature}
    response = client.chat.completions.create(**provider_request)
    if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
        optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)
    pvg_completion_tokens += response.usage.completion_tokens
    solutions = [choice.message.content for choice in response.choices]
    logger.debug(f'Generated {role} solutions: {solutions}')
    return solutions

def verify_solutions(client, system_prompt: str, initial_query: str, solutions: List[str], model: str, request_id: str=None) -> List[float]:
    global pvg_completion_tokens
    logger.info(f'Verifying {len(solutions)} solutions')
    verify_prompt = f'{system_prompt}\nYou are a verifier tasked with evaluating the correctness and clarity of solutions to the given problem.\nRate the following solution on a scale from 0 to 10, where:\n- 0 is completely incorrect or incomprehensible\n- 5 is partially correct or lacks clarity\n- 10 is perfectly correct, clear, and well-explained\n\nConsider the following criteria:\n1. Accuracy of the final answer or conclusion\n2. Correctness of each step or argument in the solution\n3. Clarity and logical flow of the explanation\n4. Appropriate use of domain-specific concepts or terminology\n5. Completeness of the solution\n\nBe especially vigilant for subtle errors or flaws that might not be immediately obvious.\n\nProvide your response in the following format:\n\nScore: [Your numerical score from 0 to 10]\nExplanation: [Your detailed explanation for the score, highlighting specific strengths or weaknesses]\n\nEnsure that the Score is a single number between 0 and 10, and the Explanation is on a new line.'
    scores = []
    for i, solution in enumerate(solutions):
        messages = [{'role': 'system', 'content': verify_prompt}, {'role': 'user', 'content': f'Problem: {initial_query}\n\nSolution: {solution}'}]
        provider_request = {'model': model, 'messages': messages, 'max_tokens': 1024, 'temperature': 0.2}
        response = client.chat.completions.create(**provider_request)
        if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)
        pvg_completion_tokens += response.usage.completion_tokens
        rating = response.choices[0].message.content
        logger.debug(f'Raw rating for solution {i + 1}: {rating}')
        score_match = re.search('Score:\\s*(\\d+(\\.\\d+)?)', rating)
        explanation_match = re.search('Explanation:\\s*(.*)', rating, re.DOTALL)
        if score_match:
            try:
                score = float(score_match.group(1))
                scores.append(score)
                logger.debug(f'Solution {i + 1} score: {score}')
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                    logger.debug(f'Explanation: {explanation}')
                else:
                    logger.warning(f'No explanation found for solution {i + 1}')
            except ValueError:
                scores.append(0)
                logger.warning(f'Failed to parse score for solution {i + 1}. Setting score to 0.')
        else:
            scores.append(0)
            logger.warning(f'No score found for solution {i + 1}. Setting score to 0.')
    return scores

def run(system_prompt, initial_query, client, model, **kwargs):
    try:
        router_model, tokenizer, device = load_optillm_model()
        input_ids, attention_mask = preprocess_input(tokenizer, system_prompt, initial_query)
        predicted_approach, _ = predict_approach(router_model, input_ids, attention_mask, device)
        print(f'Router predicted approach: {predicted_approach}')
        if predicted_approach == 'none':
            response = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': initial_query}])
            return (response.choices[0].message.content, response.usage.completion_tokens)
        elif predicted_approach == 'mcts':
            return chat_with_mcts(system_prompt, initial_query, client, model, **kwargs)
        elif predicted_approach == 'bon':
            return best_of_n_sampling(system_prompt, initial_query, client, model, **kwargs)
        elif predicted_approach == 'moa':
            return mixture_of_agents(system_prompt, initial_query, client, model)
        elif predicted_approach == 'rto':
            return round_trip_optimization(system_prompt, initial_query, client, model)
        elif predicted_approach == 'z3':
            z3_solver = Z3SymPySolverSystem(system_prompt, client, model)
            return z3_solver.process_query(initial_query)
        elif predicted_approach == 'self_consistency':
            return advanced_self_consistency_approach(system_prompt, initial_query, client, model)
        elif predicted_approach == 'pvg':
            return inference_time_pv_game(system_prompt, initial_query, client, model)
        elif predicted_approach == 'rstar':
            rstar = RStar(system_prompt, client, model, **kwargs)
            return rstar.solve(initial_query)
        elif predicted_approach == 'cot_reflection':
            return cot_reflection(system_prompt, initial_query, client, model, **kwargs)
        elif predicted_approach == 'plansearch':
            return plansearch(system_prompt, initial_query, client, model, **kwargs)
        elif predicted_approach == 'leap':
            return leap(system_prompt, initial_query, client, model)
        elif predicted_approach == 're2':
            return re2_approach(system_prompt, initial_query, client, model, **kwargs)
        else:
            raise ValueError(f'Unknown approach: {predicted_approach}')
    except Exception as e:
        print(f'Error in router plugin: {str(e)}. Falling back to direct model usage.')
        response = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': initial_query}])
        return (response.choices[0].message.content, response.usage.completion_tokens)

def load_optillm_model():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    base_model = AutoModel.from_pretrained(BASE_MODEL)
    model = OptILMClassifier(base_model, num_labels=len(APPROACHES))
    model.to(device)
    safetensors_path = hf_hub_download(repo_id=OPTILLM_MODEL_NAME, filename='model.safetensors')
    load_model(model, safetensors_path)
    tokenizer = AutoTokenizer.from_pretrained(OPTILLM_MODEL_NAME)
    return (model, tokenizer, device)

def preprocess_input(tokenizer, system_prompt, initial_query):
    combined_input = f'{system_prompt}\n\nUser: {initial_query}'
    encoding = tokenizer.encode_plus(combined_input, add_special_tokens=True, max_length=MAX_LENGTH, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
    return (encoding['input_ids'], encoding['attention_mask'])

def predict_approach(model, input_ids, attention_mask, device, effort=0.7):
    model.eval()
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        effort_tensor = torch.tensor([effort], dtype=torch.float).to(device)
        logits = model(input_ids, attention_mask=attention_mask, effort=effort_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_approach_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_approach_index].item()
    return (APPROACHES[predicted_approach_index], confidence)

class TestConversationLoggingApproaches(unittest.TestCase):
    """Test conversation logging across all approaches"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / 'conversations'
        self.logger = ConversationLogger(self.log_dir, enabled=True)
        optillm.conversation_logger = self.logger
        self.system_prompt = 'You are a helpful assistant.'
        self.initial_query = 'What is 2 + 2?'
        self.model = 'test-model'
        self.request_id = 'test-request-123'
        self.client = MockOpenAIClient()

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        optillm.conversation_logger = None

    def test_multi_call_approaches_logging(self):
        """Test BON, MCTS, and RTO approaches log API calls correctly"""
        self.logger.start_conversation({'model': self.model, 'messages': []}, 'bon', self.model)
        result, tokens = best_of_n_sampling(self.system_prompt, self.initial_query, self.client, self.model, n=2, request_id=self.request_id)
        bon_calls = self.client.call_count
        self.assertGreaterEqual(bon_calls, 2)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
        self.client.call_count = 0
        mcts_request_id = self.request_id + '_mcts'
        self.logger.start_conversation({'model': self.model, 'messages': []}, 'mcts', self.model)
        result, tokens = chat_with_mcts(self.system_prompt, self.initial_query, self.client, self.model, num_simulations=2, exploration_weight=0.2, simulation_depth=1, request_id=mcts_request_id)
        mcts_calls = self.client.call_count
        self.assertGreaterEqual(mcts_calls, 1)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
        self.client.call_count = 0
        rto_request_id = self.request_id + '_rto'
        self.logger.start_conversation({'model': self.model, 'messages': []}, 'rto', self.model)
        result, tokens = round_trip_optimization(self.system_prompt, self.initial_query, self.client, self.model, request_id=rto_request_id)
        rto_calls = self.client.call_count
        self.assertGreaterEqual(rto_calls, 3)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)

    def test_single_call_approaches_logging(self):
        """Test CoT Reflection and RE2 approaches log single API calls correctly"""
        self.logger.start_conversation({'model': self.model, 'messages': []}, 'cot_reflection', self.model)
        result, tokens = cot_reflection(self.system_prompt, self.initial_query, self.client, self.model, request_id=self.request_id)
        self.assertEqual(self.client.call_count, 1)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
        self.client.call_count = 0
        re2_request_id = self.request_id + '_re2'
        self.logger.start_conversation({'model': self.model, 'messages': []}, 're2', self.model)
        result, tokens = re2_approach(self.system_prompt, self.initial_query, self.client, self.model, n=1, request_id=re2_request_id)
        self.assertEqual(self.client.call_count, 1)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)

    def test_sampling_approaches_logging(self):
        """Test PVG and Self Consistency approaches log multiple sampling calls"""
        self.logger.start_conversation({'model': self.model, 'messages': []}, 'pvg', self.model)
        result, tokens = inference_time_pv_game(self.system_prompt, self.initial_query, self.client, self.model, num_rounds=1, num_solutions=2, request_id=self.request_id)
        pvg_calls = self.client.call_count
        self.assertGreaterEqual(pvg_calls, 3)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
        self.client.call_count = 0
        sc_request_id = self.request_id + '_sc'
        self.logger.start_conversation({'model': self.model, 'messages': []}, 'self_consistency', self.model)
        result, tokens = advanced_self_consistency_approach(self.system_prompt, self.initial_query, self.client, self.model, request_id=sc_request_id)
        self.assertEqual(self.client.call_count, 5)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)

    @patch('optillm.z3_solver.multiprocessing.get_context')
    def test_complex_class_based_approaches_logging(self, mock_mp_context):
        """Test RStar and Z3 Solver class-based approaches log API calls correctly"""
        self.logger.start_conversation({'model': self.model, 'messages': []}, 'rstar', self.model)
        rstar = RStar(self.system_prompt, self.client, self.model, max_depth=2, num_rollouts=2, request_id=self.request_id)
        result, tokens = rstar.solve(self.initial_query)
        rstar_calls = self.client.call_count
        self.assertGreaterEqual(rstar_calls, 1)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
        self.client.call_count = 0
        z3_request_id = self.request_id + '_z3'
        self.logger.start_conversation({'model': self.model, 'messages': []}, 'z3', self.model)
        mock_pool = Mock()
        mock_result = Mock()
        mock_result.get.return_value = ('success', 'Test solver output')
        mock_pool.apply_async.return_value = mock_result
        mock_context = Mock()
        mock_context.Pool.return_value = MagicMock()
        mock_context.Pool.return_value.__enter__.return_value = mock_pool
        mock_context.Pool.return_value.__exit__.return_value = None
        mock_mp_context.return_value = mock_context
        z3_solver = Z3SymPySolverSystem(self.system_prompt, self.client, self.model, request_id=z3_request_id)
        result, tokens = z3_solver.process_query(self.initial_query)
        self.assertGreaterEqual(self.client.call_count, 1)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)

    def test_logging_edge_cases(self):
        """Test approaches work with logging disabled, no request_id, and API errors"""
        optillm.conversation_logger = None
        result, tokens = best_of_n_sampling(self.system_prompt, self.initial_query, self.client, self.model, n=2, request_id=self.request_id)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
        optillm.conversation_logger = self.logger
        self.client.call_count = 0
        result, tokens = cot_reflection(self.system_prompt, self.initial_query, self.client, self.model, request_id=None)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
        error_client = Mock()
        error_client.chat.completions.create.side_effect = Exception('API Error')
        with self.assertRaises(Exception):
            cot_reflection(self.system_prompt, self.initial_query, error_client, self.model, request_id=self.request_id)

    def test_full_integration_with_file_logging(self):
        """Test complete integration from approach execution to file logging"""
        request_id = self.logger.start_conversation({'model': 'test-model', 'messages': []}, 'bon', 'test-model')
        result, tokens = best_of_n_sampling('You are a helpful assistant.', 'What is 2 + 2?', self.client, 'test-model', n=2, request_id=request_id)
        self.logger.finalize_conversation(request_id)
        log_files = list(self.log_dir.glob('*.jsonl'))
        self.assertGreater(len(log_files), 0)
        with open(log_files[0], 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
            log_entry = json.loads(lines[0].strip())
            self.assertEqual(log_entry['approach'], 'bon')
            self.assertIn('provider_calls', log_entry)
            self.assertGreater(len(log_entry['provider_calls']), 0)

class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing functionality"""

    def test_existing_approaches_still_work(self):
        """Test that existing approaches work without reasoning token changes"""
        from optillm.bon import best_of_n_sampling
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = 'Regular response'
        mock_response.usage.completion_tokens = 10
        mock_client.chat.completions.create.return_value = mock_response
        try:
            result, tokens = best_of_n_sampling(system_prompt='You are a helpful assistant.', initial_query='test', client=mock_client, model='test-model', n=3)
            self.assertIsInstance(result, str)
            self.assertIsInstance(tokens, int)
        except Exception as e:
            self.fail(f'Existing approach failed: {e}')

    def test_api_without_auth_header(self):
        """Test API still returns proper errors without auth"""
        import optillm
        app = optillm.app
        app.config['TESTING'] = True
        client = app.test_client()
        response = client.post('/v1/chat/completions', json={'model': TEST_MODEL, 'messages': []})
        self.assertIn(response.status_code, [401, 403, 500])

