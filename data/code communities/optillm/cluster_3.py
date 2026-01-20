# Cluster 3

def get_llm_response(problem: str, model: str, extra_body: dict=None, timeout: int=600) -> Dict[str, any]:
    """
    Get response from the LLM for an IMO problem with extended timeout for complex reasoning
    """
    try:
        kwargs = {}
        if extra_body:
            kwargs['extra_body'] = extra_body
        response = client.with_options(timeout=timeout).chat.completions.create(model=model, messages=[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': problem}], max_tokens=64000, **kwargs)
        solution_text = response.choices[0].message.content.strip()
        reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0)
        total_tokens = response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
        return {'solution': solution_text, 'reasoning_tokens': reasoning_tokens, 'total_tokens': total_tokens, 'success': True}
    except Exception as e:
        logger.error(f'Error getting LLM response: {e}')
        return {'solution': f'Error generating solution: {str(e)}', 'reasoning_tokens': 0, 'total_tokens': 0, 'success': False}

def save_result(filename: str, result: Dict):
    """Save a single result to the results file."""
    results = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            results = []
    results.append(result)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def load_existing_results(filename: str) -> List[Dict]:
    """Load existing results from file if it exists."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def analyze_results(results: List[Dict], approach_name: str=None):
    """Analyze and print comprehensive statistics of IMO evaluation results"""
    if not results:
        print('No results to analyze')
        return
    total_problems = len(results)
    likely_correct = sum((1 for r in results if r['evaluation']['is_correct']))
    high_confidence = sum((1 for r in results if r['evaluation']['confidence'] == 'high'))
    avg_correctness = sum((r['evaluation']['correctness_score'] for r in results)) / total_problems
    avg_completeness = sum((r['evaluation']['quality_analysis']['completeness_score'] for r in results)) / total_problems
    total_reasoning_tokens = sum((r['response']['reasoning_tokens'] for r in results))
    avg_reasoning_tokens = total_reasoning_tokens / total_problems
    print('\n' + '=' * 80)
    print(f'IMO 2025 Evaluation Results - {approach_name or 'Baseline'}')
    print('=' * 80)
    print(f'Total problems attempted: {total_problems}')
    print(f'Likely correct solutions: {likely_correct} ({likely_correct / total_problems:.1%})')
    print(f'High confidence solutions: {high_confidence} ({high_confidence / total_problems:.1%})')
    print(f'Average correctness score: {avg_correctness:.3f}')
    print(f'Average completeness score: {avg_completeness:.3f}')
    print(f'Total reasoning tokens used: {total_reasoning_tokens:,}')
    print(f'Average reasoning tokens per problem: {avg_reasoning_tokens:.0f}')
    print(f'\nProblem Type Breakdown:')
    type_stats = {}
    for result in results:
        prob_type = result['problem_data']['type']
        if prob_type not in type_stats:
            type_stats[prob_type] = {'total': 0, 'correct': 0, 'scores': []}
        type_stats[prob_type]['total'] += 1
        if result['evaluation']['is_correct']:
            type_stats[prob_type]['correct'] += 1
        type_stats[prob_type]['scores'].append(result['evaluation']['correctness_score'])
    for prob_type, stats in type_stats.items():
        accuracy = stats['correct'] / stats['total']
        avg_score = sum(stats['scores']) / len(stats['scores'])
        print(f'  {prob_type}: {stats['correct']}/{stats['total']} ({accuracy:.1%}) - Avg score: {avg_score:.3f}')
    print(f'\nDetailed Results:')
    print('-' * 80)
    for result in results:
        prob_id = result['problem_data']['id']
        prob_type = result['problem_data']['type']
        tokens = result['response']['reasoning_tokens']
        is_correct = result['evaluation']['is_correct']
        verdict = result['evaluation']['verdict']
        status = '✓' if is_correct else '✗'
        print(f'Problem {prob_id} ({prob_type}): {status} {verdict} - {tokens:,} tokens')
    print(f'\nSolution Quality Analysis:')
    print('-' * 40)
    quality_metrics = ['has_proof_structure', 'uses_mathematical_notation', 'has_logical_steps', 'addresses_all_cases', 'has_conclusion']
    for metric in quality_metrics:
        count = sum((1 for r in results if r['evaluation']['quality_analysis'][metric]))
        percentage = count / total_problems
        print(f'{metric.replace('_', ' ').title()}: {count}/{total_problems} ({percentage:.1%})')

def save_result(filename: str, result: Dict):
    """Save a single result to the results file."""
    results = load_existing_results(filename)
    results.append(result)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def main(model: str):
    """Main evaluation function."""
    os.makedirs('results', exist_ok=True)
    results_file = f'evaluation_results_math500_{model.replace('/', '_')}.json'
    dataset = load_math500_dataset()
    existing_results = load_existing_results(results_file)
    processed_indexes = {result['index'] for result in existing_results}
    for idx, item in enumerate(tqdm(dataset, desc='Evaluating problems')):
        if idx in processed_indexes:
            continue
        problem_text = item['problem']
        correct_answer = item['answer']
        response = get_llm_response(problem_text, model)
        predicted_answer = extract_answer(response)
        is_correct = compare_answers(correct_answer, predicted_answer)
        result = {'index': idx, 'problem': problem_text, 'response': response, 'correct_answer': correct_answer, 'predicted_answer': predicted_answer, 'is_correct': is_correct}
        save_result(results_file, result)
    final_results = load_existing_results(results_file)
    analyze_results(final_results)

def load_math500_dataset() -> list[dict]:
    """
    Load the MATH-500 dataset.
    Returns:
        list[dict]: The dataset of problems.
    """
    dataset = load_dataset('HuggingFaceH4/MATH-500')
    dataset = dataset['test']
    logging.debug(f'Dataset size: {len(dataset)}.')
    return dataset

def extract_answer(response: str) -> Optional[str]:
    """Extract the answer from a math solution response."""
    if not response:
        logger.debug('Empty response received')
        return None
    start_idx = response.rfind('\\boxed{')
    if start_idx == -1:
        logger.debug('No \\boxed{} found in response')
        return None
    brace_count = 1
    pos = start_idx + 7
    while pos < len(response) and brace_count > 0:
        if response[pos] == '{':
            brace_count += 1
        elif response[pos] == '}':
            brace_count -= 1
        pos += 1
    if brace_count == 0:
        answer = response[start_idx + 7:pos - 1]
        logger.debug(f'Extracted answer: {answer}')
        return answer.strip()
    logger.debug('No matching closing brace found')
    return None

def perform_rtc_evaluation(query: str, model: str) -> Tuple[bool, float, Dict]:
    """
    Perform Round-Trip Correctness evaluation.
    
    Args:
        query: Original query
        model: Model name to use
        
    Returns:
        Tuple of (passed_rtc, similarity_score, evaluation_details)
    """
    response_1 = get_llm_response([{'role': 'user', 'content': query}], model)
    if not response_1:
        return (False, 0.0, {'error': 'Failed to get initial response'})
    inverse_prompt = f'Given this query and response pair, generate a new query that would lead to a similar response. Focus on the key aspects that would generate equivalent content:\n\nOriginal Query: {query}\nResponse: {response_1}\n\nGenerate a new query that would elicit a similar response:'
    alternate_query = get_llm_response([{'role': 'user', 'content': inverse_prompt}], model)
    if not alternate_query:
        return (False, 0.0, {'error': 'Failed to generate alternate query'})
    response_2 = get_llm_response([{'role': 'user', 'content': alternate_query}], model)
    if not response_2:
        return (False, 0.0, {'error': 'Failed to get second response'})
    similarity_score = compute_similarity(response_1, response_2)
    evaluation_details = {'original_query': query, 'response_1': response_1, 'alternate_query': alternate_query, 'response_2': response_2, 'similarity_score': similarity_score}
    return (similarity_score >= RTCConfig.similarity_threshold, similarity_score, evaluation_details)

def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two texts using TF-IDF vectorization.
    This is a local implementation that doesn't require any external API.
    """
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except Exception as e:
        logger.error(f'Error computing similarity: {e}')
        return 0.0

def evaluate_dataset(model: str, output_file: str):
    """Evaluate the dataset using RTC methodology."""
    dataset = load_dataset('lmarena-ai/arena-hard-auto-v0.1')
    results = []
    passed_rtc_count = 0
    total_examples = 0
    for item in tqdm(dataset['train'], desc='Evaluating examples'):
        query = extract_first_turn_content(item['turns'])
        if not query:
            continue
        passed_rtc, similarity_score, details = perform_rtc_evaluation(query, model)
        result = {'id': total_examples, 'query': query, 'passed_rtc': passed_rtc, 'similarity_score': similarity_score, 'evaluation_details': details}
        results.append(result)
        if passed_rtc:
            passed_rtc_count += 1
        total_examples += 1
        with open(output_file, 'w') as f:
            json.dump({'model': model, 'total_examples': total_examples, 'passed_rtc': passed_rtc_count, 'rtc_pass_rate': passed_rtc_count / total_examples if total_examples > 0 else 0, 'results': results}, f, indent=2)
    logger.info(f'\nEvaluation Summary for {model}:')
    logger.info(f'Total examples evaluated: {total_examples}')
    logger.info(f'Examples passing RTC: {passed_rtc_count}')
    logger.info(f'RTC pass rate: {passed_rtc_count / total_examples * 100:.2f}%')

def extract_first_turn_content(turns: List[Dict]) -> str:
    """Extract the content from the first turn in the conversation."""
    if not turns or not isinstance(turns, list):
        return ''
    return turns[0].get('content', '')

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLMs on arena-hard-auto dataset using RTC')
    parser.add_argument('--model', type=str, required=True, help='OpenAI model to use')
    parser.add_argument('--output', type=str, default='rtc_eval_results.json', help='Output file for results')
    args = parser.parse_args()
    os.makedirs('results', exist_ok=True)
    output_file = os.path.join('results', args.output)
    evaluate_dataset(args.model, output_file)

def load_dataset_by_year(year: int) -> list[dict]:
    """
    Load dataset by year (2024 or 2025).
    Returns:
        list[dict]: The dataset of problems.
    """
    if year == 2024:
        return load_2024_dataset()
    elif year == 2025:
        return load_2025_dataset()
    else:
        raise ValueError(f'Unsupported year: {year}. Only 2024 and 2025 are supported.')

def load_2024_dataset() -> list[dict]:
    """
    Load the 2024 dataset of problems.
    Returns:
        list[dict]: The dataset of problems.
    """
    dataset_original = load_dataset('AI-MO/aimo-validation-aime')
    dataset = dataset_original['train'].filter(lambda example: '2024' in example['url'])
    logging.debug(f'Filtered dataset size: {len(dataset)}.')
    assert len(dataset) == 30, f'Expected 30 problems after filtering by 2024, but found {len(dataset)}'
    return dataset

def load_2025_dataset() -> list[dict]:
    """
    Load the 2025 dataset of problems from math-ai/aime25.
    Returns:
        list[dict]: The dataset of problems.
    """
    dataset = load_dataset('math-ai/aime25')
    dataset = dataset['test']
    logging.debug(f'Loaded AIME 2025 dataset size: {len(dataset)}.')
    assert len(dataset) == 30, f'Expected 30 problems in AIME 2025, but found {len(dataset)}'
    return dataset

def get_llm_response(problem: str, model: str, analyze_logits: bool=False, extra_body: dict=None) -> Union[str, List[Dict]]:
    """
    Get response from the LLM for a given problem.
    If multiple choices are returned, formats them as attempt dictionaries.
    
    Args:
        problem (str): The problem text
        model (str): The model identifier
        analyze_logits (bool): Whether to request logprobs
        
    Returns:
        Union[str, List[Dict]]: Either a string response or list of attempt dictionaries
    """
    try:
        kwargs = {}
        if analyze_logits:
            kwargs['logprobs'] = True
            kwargs['top_logprobs'] = 3
        if extra_body:
            kwargs['extra_body'] = extra_body
        response = client.with_options(timeout=6000.0).chat.completions.create(model=model, messages=[{'role': 'user', 'content': SYSTEM_PROMPT + problem}], max_tokens=64000, **kwargs)
        if analyze_logits:
            raw_filename = f'results/raw_responses_{model.replace('/', '_')}.json'
            problem_id = hash(problem) % 10000
            save_raw_response(raw_filename, problem_id, response.model_dump())
        if len(response.choices) > 1:
            attempts = []
            for i, choice in enumerate(response.choices):
                response_text = choice.message.content.strip()
                predicted_answer = extract_answer(response_text)
                attempt_data = {'attempt_number': i + 1, 'response': response_text, 'predicted_answer': predicted_answer}
                if analyze_logits and hasattr(choice.message, 'logprobs') and choice.message.logprobs:
                    attempt_data['logprobs'] = choice.message.logprobs
                attempts.append(attempt_data)
            return attempts
        response_text = response.choices[0].message.content.strip()
        if analyze_logits and hasattr(response.choices[0].message, 'logprobs') and response.choices[0].message.logprobs:
            return {'response': response_text, 'logprobs': response.choices[0].message.logprobs}
        return response_text
    except Exception as e:
        logger.error(f'Error getting LLM response: {e}')
        logger.error(f'Error type: {type(e).__name__}')
        if 'timeout' in str(e).lower():
            logger.error('API call timed out - consider increasing timeout for complex approaches like MARS')
        raise e

def make_n_attempts(problem: str, model: str, n: int, analyze_thoughts: bool=False, analyze_logits: bool=False, extra_body: dict=None) -> List[Dict]:
    """
    Make n attempts to solve a problem and return all responses and predictions.
    
    Args:
        problem (str): The problem text
        model (str): The model identifier
        n (int): Number of attempts to make
        analyze_thoughts (bool): Whether to analyze thinking patterns
        analyze_logits (bool): Whether to analyze token probabilities
        
    Returns:
        List[Dict]: List of dictionaries containing response and predicted answer for each attempt
    """
    attempts = []
    remaining_attempts = n
    while remaining_attempts > 0:
        try:
            response = get_llm_response(problem, model, analyze_logits, extra_body)
        except Exception as e:
            logger.error(f'Failed to get response for attempt {n - remaining_attempts + 1}: {e}')
            attempt_data = {'attempt_number': len(attempts) + 1, 'response': f'ERROR: {str(e)}', 'predicted_answer': None, 'error': str(e)}
            attempts.append(attempt_data)
            remaining_attempts -= 1
            continue
        if isinstance(response, list):
            for attempt in response:
                if analyze_thoughts:
                    attempt['thought_analysis'] = analyze_thinking(attempt['response'])
                if analyze_logits and 'logprobs' in attempt:
                    attempt['logit_analysis'] = analyze_logits_probs(attempt['logprobs']['content'])
            attempts.extend(response)
            remaining_attempts = n - len(attempts)
        elif isinstance(response, dict) and 'response' in response:
            response_text = response['response']
            predicted_answer = extract_answer(response_text)
            attempt_data = {'attempt_number': len(attempts) + 1, 'response': response_text, 'predicted_answer': predicted_answer}
            if analyze_thoughts:
                attempt_data['thought_analysis'] = analyze_thinking(response_text)
            if analyze_logits and 'logprobs' in response:
                attempt_data['logit_analysis'] = analyze_logits_probs(response['logprobs']['content'])
            attempts.append(attempt_data)
            remaining_attempts -= 1
        else:
            predicted_answer = extract_answer(response)
            attempt_data = {'attempt_number': len(attempts) + 1, 'response': response, 'predicted_answer': predicted_answer}
            if analyze_thoughts:
                attempt_data['thought_analysis'] = analyze_thinking(response)
            attempts.append(attempt_data)
            remaining_attempts -= 1
    return attempts

def analyze_thinking(response: str) -> Dict:
    """
    Analyze thinking patterns in the response.
    Extract tokens between <think> and </think> tags and count thought transitions.
    
    Args:
        response (str): The model's response text
        
    Returns:
        Dict: Analysis metrics including thinking tokens and thought transitions
    """
    result = {'has_think_tags': False, 'thinking_tokens': 0, 'thinking_tokens_text': '', 'total_tokens': len(response.split()), 'thought_transitions': 0, 'transition_counts': {phrase: 0 for phrase in THOUGHT_TRANSITIONS}, 'transition_positions': []}
    think_pattern = re.compile('<think>(.*?)</think>', re.DOTALL)
    think_match = think_pattern.search(response)
    if think_match:
        thinking_text = think_match.group(1)
        result['has_think_tags'] = True
        result['thinking_tokens'] = len(thinking_text.split())
        result['thinking_tokens_text'] = thinking_text
        position = 0
        for phrase in THOUGHT_TRANSITIONS:
            for match in re.finditer('\\b' + re.escape(phrase) + '\\b', thinking_text):
                result['transition_counts'][phrase] += 1
                token_position = len(thinking_text[:match.start()].split())
                result['transition_positions'].append((phrase, token_position))
        result['transition_positions'].sort(key=lambda x: x[1])
        result['thought_transitions'] = sum(result['transition_counts'].values())
    return result

def analyze_logits_probs(logprobs_data: List[Dict]) -> Dict:
    """
    Analyze token probability distributions and entropy patterns.
    
    Args:
        logprobs_data: List of dictionaries containing token and logprob information
        
    Returns:
        Dict: Analysis metrics including entropy statistics
    """
    if not logprobs_data:
        return {'entropy_stats': None, 'transition_entropy': None, 'token_count': 0}
    token_entropies = []
    token_probs = []
    token_texts = []
    for token_info in logprobs_data:
        if not token_info.get('top_logprobs'):
            continue
        probs = []
        for token, logprob in token_info['top_logprobs'].items():
            probs.append(math.exp(logprob))
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        entropy = -sum((p * math.log2(p) if p > 0 else 0 for p in probs))
        token_entropies.append(entropy)
        token_probs.append(probs[0] if probs else 0)
        token_texts.append(token_info['token'])
    transition_entropy = {}
    for phrase in THOUGHT_TRANSITIONS:
        transition_indices = []
        for i, token in enumerate(token_texts):
            if phrase.startswith(token) and i < len(token_texts) - 1:
                transition_indices.append(i)
        if transition_indices:
            before_entropy = []
            after_entropy = []
            for idx in transition_indices:
                before_window = max(0, idx - 5)
                after_window = min(len(token_entropies), idx + 5)
                if idx > before_window:
                    before_entropy.extend(token_entropies[before_window:idx])
                if after_window > idx:
                    after_entropy.extend(token_entropies[idx:after_window])
            transition_entropy[phrase] = {'before_mean': statistics.mean(before_entropy) if before_entropy else 0, 'after_mean': statistics.mean(after_entropy) if after_entropy else 0, 'count': len(transition_indices)}
    entropy_stats = {'mean': statistics.mean(token_entropies) if token_entropies else 0, 'median': statistics.median(token_entropies) if token_entropies else 0, 'max': max(token_entropies) if token_entropies else 0, 'min': min(token_entropies) if token_entropies else 0, 'std': statistics.stdev(token_entropies) if len(token_entropies) > 1 else 0}
    if token_entropies:
        quartile_size = max(1, len(token_entropies) // 4)
        entropy_stats['quartiles'] = [statistics.mean(token_entropies[i:i + quartile_size]) for i in range(0, len(token_entropies), quartile_size) if i < len(token_entropies)]
    else:
        entropy_stats['quartiles'] = []
    return {'entropy_stats': entropy_stats, 'transition_entropy': transition_entropy, 'token_count': len(token_entropies)}

def save_result(filename: str, result: Dict):
    """Save a single result to the results file."""
    results = load_existing_results(filename)
    results.append(result)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def main(model: str, n_attempts: int, year: int=2024, analyze_thoughts: bool=False, analyze_logits: bool=False, test_time_compute: bool=False, approach_name: str=None, extra_body: dict=None):
    """Main evaluation function that handles gaps in processed indexes."""
    os.makedirs('results', exist_ok=True)
    suffix_parts = []
    if year != 2024:
        suffix_parts.append(f'aime{year}')
    if analyze_thoughts:
        suffix_parts.append('thought_analysis')
    if analyze_logits:
        suffix_parts.append('logit_analysis')
    if approach_name:
        suffix_parts.append(approach_name)
    suffix = '_' + '_'.join(suffix_parts) if suffix_parts else ''
    results_file = f'results/evaluation_results_{model.replace('/', '_')}_pass_at_{n_attempts}{suffix}.json'
    dataset = load_dataset_by_year(year)
    existing_results = load_existing_results(results_file)
    processed_indexes = {result['index'] for result in existing_results}
    for _, item in enumerate(tqdm(dataset, desc='Evaluating problems')):
        id = int(item['id'])
        if id in processed_indexes:
            continue
        problem_text = item['problem']
        correct_answer = int(item['answer'])
        print(f'\n🔬 Processing Problem {id}: {problem_text[:100]}...')
        print(f'   Expected answer: {correct_answer}')
        if extra_body and 'optillm_approach' in extra_body:
            print(f'   Using approach: {extra_body['optillm_approach']}')
        attempts = make_n_attempts(problem_text, model, n_attempts, analyze_thoughts, analyze_logits, extra_body)
        is_correct, first_correct = evaluate_pass_at_n(attempts, correct_answer)
        predicted_answers = [attempt.get('predicted_answer') for attempt in attempts]
        print(f'   Predicted: {predicted_answers}')
        if is_correct:
            print(f'   ✅ CORRECT!')
        else:
            print(f'   ❌ Incorrect')
        result = {'index': id, 'problem': problem_text, 'attempts': attempts, 'correct_answer': correct_answer, 'is_correct': is_correct, 'first_correct_attempt': first_correct}
        save_result(results_file, result)
    final_results = load_existing_results(results_file)
    analyze_results(final_results, n_attempts, analyze_thoughts, analyze_logits)

def evaluate_pass_at_n(attempts: List[Dict], correct_answer: int) -> Tuple[bool, Optional[int]]:
    """
    Evaluate if any of the n attempts got the correct answer.
    
    Args:
        attempts (List[Dict]): List of attempt results
        correct_answer (int): The correct answer
        
    Returns:
        Tuple[bool, Optional[int]]: (whether any attempt was correct, first correct attempt number)
    """
    for attempt in attempts:
        if attempt['predicted_answer'] == correct_answer:
            return (True, attempt['attempt_number'])
    return (False, None)

def save_result(filename: str, result: Dict):
    results = load_existing_results(filename)
    results.append(result)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def main(model: str):
    dataset = load_dataset('google/frames-benchmark', split='test')
    filename = f'evaluation_results_{model.replace('/', '_')}.json'
    existing_results = load_existing_results(filename)
    last_processed_index = get_last_processed_index(existing_results)
    for item in tqdm(dataset, desc='Processing samples'):
        index = int(item['Unnamed: 0'])
        if index <= last_processed_index:
            continue
        prompt = generate_llm_prompt(item['Prompt'], item['wiki_links'])
        llm_response = get_llm_response(prompt, model)
        evaluation = evaluate_response(item['Prompt'], llm_response, item['Answer'], model)
        result = {'index': index, 'prompt': item['Prompt'], 'ground_truth': item['Answer'], 'llm_response': llm_response, 'evaluation_decision': evaluation['decision'], 'evaluation_explanation': evaluation['explanation'], 'reasoning_type': item['reasoning_types']}
        save_result(filename, result)
    results = load_existing_results(filename)
    total_samples = len(results)
    correct_answers = sum((1 for r in results if r['evaluation_decision'] == 'TRUE'))
    accuracy = correct_answers / total_samples
    print(f'Model: {model}')
    print(f'Total samples: {total_samples}')
    print(f'Correct answers: {correct_answers}')
    print(f'Accuracy: {accuracy:.2%}')
    reasoning_types = set((r['reasoning_type'] for r in results))
    for rt in reasoning_types:
        rt_samples = [r for r in results if r['reasoning_type'] == rt]
        rt_correct = sum((1 for r in rt_samples if r['evaluation_decision'] == 'TRUE'))
        rt_accuracy = rt_correct / len(rt_samples)
        print(f'Accuracy for {rt}: {rt_accuracy:.2%}')

def get_last_processed_index(results: List[Dict]) -> int:
    """Get the index of the last processed problem."""
    if not results:
        return -1
    return max((int(r.get('index', -1)) for r in results))

def generate_llm_prompt(prompt: str, wiki_links: List[str]) -> str:
    return f'Here are the relevant Wikipedia articles:\n{wiki_links}\n\nBased on all the information, answer the query. \n\nQuery: {prompt}\n\n'

