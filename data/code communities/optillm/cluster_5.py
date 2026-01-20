# Cluster 5

def evaluate_response(response: str, ground_truth: str, category: str, question: str=None) -> bool:
    """
    Evaluate if the response matches the ground truth based on category.
    
    Args:
        response: Model's response
        ground_truth: Correct answer
        category: Problem category (gsm8k, mmlu_math, boolq, aqua_rat)
        question: Original question text, needed for MMLU evaluation
    
    Returns:
        bool: Whether the response is correct
    """
    if not response or not ground_truth:
        return False
    response = remove_thinking_blocks(response)
    if category == 'gsm8k':
        response_num = extract_gsm8k_answer(response)
        ground_truth_num = extract_gsm8k_answer(ground_truth)
        if response_num is None or ground_truth_num is None:
            return False
        return abs(response_num - ground_truth_num) < 1e-06
    elif category == 'mmlu_math':
        response_clean = response.strip().lower()
        ground_truth_clean = ground_truth.strip().lower()
        if response_clean == ground_truth_clean:
            logger.debug('Exact text match')
            return True
        if question:
            correct_index = extract_choice_index_from_question(question, ground_truth)
            if correct_index >= 0:
                is_numeric, value = is_numeric_only_response(response)
                if is_numeric and value == correct_index:
                    logger.debug(f"Numeric match: response '{response}' -> {value} matches index {correct_index}")
                    return True
                if re.search(f'{correct_index}\\s*\\.\\s*{re.escape(ground_truth_clean)}', response_clean):
                    logger.debug("Pattern match for 'index. answer'")
                    return True
                if str(correct_index) in response_clean and ground_truth_clean in response_clean:
                    logger.debug('Contains both index and answer')
                    return True
        return False
    else:
        response_clean = response.strip().lower()
        ground_truth_clean = ground_truth.strip().lower()
        return response_clean == ground_truth_clean

def remove_thinking_blocks(text: str) -> str:
    """
    Remove <think>...</think> blocks from the response.
    If there's a </think> tag, only keep the content after it.
    """
    if not text:
        return text
    if '</think>' in text:
        parts = text.split('</think>')
        return parts[-1].strip()
    return text

def extract_gsm8k_answer(text: str) -> float:
    """Extract numerical answer after ### from GSM8K responses."""
    match = re.search('###\\s*(-?\\d*\\.?\\d+)', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def extract_choice_index_from_question(question: str, answer: str) -> int:
    """
    Extract the index of the correct answer from a multiple-choice question.
    
    Args:
        question: The question text containing choices
        answer: The correct answer (just the text, no index)
    
    Returns:
        int: The index of the correct answer, or -1 if not found
    """
    answer_clean = answer.strip().lower()
    logger.debug(f"Looking for answer: '{answer_clean}' in question")
    if 'choices:' in question.lower():
        choices_section = question.lower().split('choices:')[1].strip()
        logger.debug(f"Choices section: '{choices_section}'")
        if '\n' not in choices_section:
            all_choices = re.findall('(\\d+)\\s*\\.\\s*([^0-9.]+?)(?=\\s*\\d+\\s*\\.|$)', choices_section)
            logger.debug(f'Single line choices found: {all_choices}')
            for idx, choice_text in all_choices:
                choice_text_clean = choice_text.strip()
                if choice_text_clean.lower() == answer_clean:
                    logger.debug(f"Found match at index {idx}: '{choice_text_clean}'")
                    return int(idx)
        choices = choices_section.split('\n')
        for i, choice in enumerate(choices):
            choice = choice.strip()
            if not choice:
                continue
            logger.debug(f"Checking choice {i}: '{choice}'")
            match = re.match('\\s*(\\d+)\\s*\\.\\s*(.*)', choice)
            if match:
                idx = int(match.group(1))
                choice_text = match.group(2).strip()
                logger.debug(f"Parsed choice: index={idx}, text='{choice_text}'")
                if choice_text.lower() == answer_clean:
                    logger.debug(f'Found exact match at index {idx}')
                    return idx
        pattern = '(\\d+)\\s*\\.\\s*' + re.escape(answer_clean)
        match = re.search(pattern, choices_section)
        if match:
            logger.debug(f'Fallback match found at index {match.group(1)}')
            return int(match.group(1))
    logger.debug('No match found for answer in choices')
    return -1

def is_numeric_only_response(response: str) -> Tuple[bool, int]:
    """
    Check if the response is just a numeric value, possibly with whitespace and newlines.
    
    Args:
        response: The response text to check
        
    Returns:
        Tuple of (is_numeric, value)
    """
    clean_response = re.sub('\\s', '', response)
    if clean_response.isdigit():
        return (True, int(clean_response))
    return (False, -1)

def evaluate_model(client: OpenAI, model: str, dataset: datasets.Dataset, approach: str, approach_extra_body: Dict[str, Any]=None, max_samples: int=None) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Evaluate a model on the dataset using a specific approach.
    Returns metrics and detailed results.
    """
    metrics = {'total_correct': 0, 'total_time': 0, 'samples': 0}
    category_metrics = {}
    detailed_results = []
    examples = dataset if max_samples is None else dataset.select(range(max_samples))
    num_runs = approach_extra_body.get('num_runs', 1) if approach_extra_body else 1
    n_param = approach_extra_body.get('n', 1) if approach_extra_body else 1
    if approach.startswith('avg@') or approach.startswith('pass@'):
        full_model_name = model
    elif approach.startswith('maj@'):
        full_model_name = f'majority_voting-{model}'
    elif approach.startswith('genselect@'):
        full_model_name = f'genselect-{model}'
    elif approach.startswith('thinkdeeper_'):
        full_model_name = model
    elif approach.startswith('majority_voting'):
        full_model_name = f'majority_voting-{model}'
    elif approach == 'none':
        full_model_name = model
    else:
        full_model_name = f'{approach}-{model}'
    for example in tqdm(examples, desc=f'Evaluating {approach}'):
        if n_param > 1 and (approach.startswith('avg@') or approach.startswith('pass@')):
            try:
                prompt = get_prompt_for_category(example['question'], example['category'])
                start_time = time.time()
                extra_body = {'spl_learning': False}
                if approach_extra_body:
                    extra_body_clean = {k: v for k, v in approach_extra_body.items() if k not in ['n', 'approach']}
                    extra_body.update(extra_body_clean)
                responses = []
                try:
                    response = client.chat.completions.create(model=full_model_name, messages=[{'role': 'system', 'content': 'You are a helpful AI assistant focused on providing precise answers in the requested format.'}, {'role': 'user', 'content': prompt}], n=n_param, temperature=0.6, max_tokens=4096, extra_body=extra_body)
                    responses = [(choice.message.content, time.time() - start_time) for choice in response.choices]
                    logger.debug(f'Generated {len(responses)} responses using n={n_param}')
                except Exception as e:
                    logger.warning(f'Parallel generation failed: {type(e).__name__}: {str(e)}')
                    logger.info('Falling back to sequential generation')
                    for i in range(n_param):
                        try:
                            single_start = time.time()
                            response = client.chat.completions.create(model=full_model_name, messages=[{'role': 'system', 'content': 'You are a helpful AI assistant focused on providing precise answers in the requested format.'}, {'role': 'user', 'content': prompt}], temperature=0.6, max_tokens=4096, extra_body=extra_body)
                            response_text = response.choices[0].message.content
                            responses.append((response_text, time.time() - single_start))
                        except Exception as seq_error:
                            logger.error(f'Sequential generation {i + 1}/{n_param} failed: {seq_error}')
                            responses.append((None, 0))
                time_taken = time.time() - start_time
                run_results = []
                for response_text, _ in responses:
                    if response_text is not None:
                        processed_response = remove_thinking_blocks(response_text)
                        is_correct = evaluate_response(processed_response, example['answer'], example['category'], example['question'])
                        run_results.append(is_correct)
                    else:
                        run_results.append(False)
                if approach.startswith('avg@'):
                    success_rate = sum(run_results) / len(run_results) if run_results else 0
                elif approach.startswith('pass@'):
                    success_rate = 1.0 if any(run_results) else 0.0
                else:
                    success_rate = sum(run_results) / len(run_results) if run_results else 0
                metrics['total_correct'] += success_rate
                metrics['total_time'] += time_taken
                metrics['samples'] += 1
                if example['category'] not in category_metrics:
                    category_metrics[example['category']] = {'correct': 0, 'total': 0, 'time': 0}
                category_metrics[example['category']]['correct'] += success_rate
                category_metrics[example['category']]['total'] += 1
                category_metrics[example['category']]['time'] += time_taken
                detailed_results.append({'id': example['id'], 'category': example['category'], 'correct': success_rate, 'n_param': n_param, 'successes': sum(run_results), 'time_taken': time_taken, 'ground_truth': example['answer']})
            except Exception as e:
                logger.error(f'Error processing example {example['id']}: {e}')
                metrics['total_correct'] += 0
                metrics['total_time'] += 0
                metrics['samples'] += 1
                if example['category'] not in category_metrics:
                    category_metrics[example['category']] = {'correct': 0, 'total': 0, 'time': 0}
                category_metrics[example['category']]['correct'] += 0
                category_metrics[example['category']]['total'] += 1
                category_metrics[example['category']]['time'] += 0
                detailed_results.append({'id': example['id'], 'category': example['category'], 'correct': False, 'time_taken': 0, 'raw_response': f'ERROR: {str(e)}', 'processed_response': None, 'has_thinking': False, 'ground_truth': example['answer'], 'error': str(e)})
                continue
        elif num_runs > 1:
            run_results = []
            total_run_time = 0
            for run_idx in range(num_runs):
                try:
                    prompt = get_prompt_for_category(example['question'], example['category'])
                    start_time = time.time()
                    extra_body = {'spl_learning': False}
                    if approach_extra_body:
                        extra_body_clean = {k: v for k, v in approach_extra_body.items() if k not in ['num_runs', 'approach']}
                        extra_body.update(extra_body_clean)
                    response = client.chat.completions.create(model=full_model_name, messages=[{'role': 'system', 'content': 'You are a helpful AI assistant focused on providing precise answers in the requested format.'}, {'role': 'user', 'content': prompt}], temperature=0.6, max_tokens=4096, extra_body=extra_body)
                    time_taken = time.time() - start_time
                    total_run_time += time_taken
                    response_text = response.choices[0].message.content
                    processed_response = remove_thinking_blocks(response_text)
                    is_correct = evaluate_response(processed_response, example['answer'], example['category'], example['question'])
                    run_results.append(is_correct)
                except Exception as e:
                    logger.error(f'Error in run {run_idx + 1} for example {example['id']}: {e}')
                    run_results.append(False)
            success_rate = sum(run_results) / len(run_results) if run_results else 0
            avg_time = total_run_time / len(run_results) if run_results else 0
            metrics['total_correct'] += success_rate
            metrics['total_time'] += avg_time
            metrics['samples'] += 1
            if example['category'] not in category_metrics:
                category_metrics[example['category']] = {'correct': 0, 'total': 0, 'time': 0}
            category_metrics[example['category']]['correct'] += success_rate
            category_metrics[example['category']]['total'] += 1
            category_metrics[example['category']]['time'] += avg_time
            detailed_results.append({'id': example['id'], 'category': example['category'], 'correct': success_rate, 'num_runs': num_runs, 'successes': sum(run_results), 'time_taken': avg_time, 'ground_truth': example['answer']})
        else:
            try:
                prompt = get_prompt_for_category(example['question'], example['category'])
                start_time = time.time()
                extra_body = {'spl_learning': False}
                if approach_extra_body:
                    extra_body_clean = {k: v for k, v in approach_extra_body.items() if k != 'approach'}
                    extra_body.update(extra_body_clean)
                response = client.chat.completions.create(model=full_model_name, messages=[{'role': 'system', 'content': 'You are a helpful AI assistant focused on providing precise answers in the requested format.'}, {'role': 'user', 'content': prompt}], temperature=0.6, max_tokens=4096, extra_body=extra_body)
                time_taken = time.time() - start_time
                response_text = response.choices[0].message.content
                raw_response = response_text
                processed_response = remove_thinking_blocks(response_text)
                is_correct = evaluate_response(processed_response, example['answer'], example['category'], example['question'])
                metrics['total_correct'] += int(is_correct)
                metrics['total_time'] += time_taken
                metrics['samples'] += 1
                if example['category'] not in category_metrics:
                    category_metrics[example['category']] = {'correct': 0, 'total': 0, 'time': 0}
                category_metrics[example['category']]['correct'] += int(is_correct)
                category_metrics[example['category']]['total'] += 1
                category_metrics[example['category']]['time'] += time_taken
                has_thinking = '</think>' in raw_response
                detailed_results.append({'id': example['id'], 'category': example['category'], 'correct': is_correct, 'time_taken': time_taken, 'raw_response': raw_response, 'processed_response': processed_response if has_thinking else None, 'has_thinking': has_thinking, 'ground_truth': example['answer']})
            except Exception as e:
                logger.error(f'Error processing example {example['id']}: {e}')
                metrics['total_correct'] += 0
                metrics['total_time'] += 0
                metrics['samples'] += 1
                if example['category'] not in category_metrics:
                    category_metrics[example['category']] = {'correct': 0, 'total': 0, 'time': 0}
                category_metrics[example['category']]['correct'] += 0
                category_metrics[example['category']]['total'] += 1
                category_metrics[example['category']]['time'] += 0
                detailed_results.append({'id': example['id'], 'category': example['category'], 'correct': False, 'time_taken': 0, 'raw_response': f'ERROR: {str(e)}', 'processed_response': None, 'has_thinking': False, 'ground_truth': example['answer'], 'error': str(e)})
                continue
    final_metrics = {'accuracy': metrics['total_correct'] / metrics['samples'] if metrics['samples'] > 0 else 0, 'average_time': metrics['total_time'] / metrics['samples'] if metrics['samples'] > 0 else 0, 'total_time': metrics['total_time'], 'total_samples': metrics['samples']}
    total_expected = len(examples)
    failures = len([r for r in detailed_results if 'error' in r])
    if failures > 0:
        logger.warning(f'Approach {approach}: {failures}/{total_expected} examples failed due to errors')
        logger.warning(f'Failed examples are counted as incorrect in accuracy calculation')
    for category, cat_metrics in category_metrics.items():
        final_metrics[f'{category}_accuracy'] = cat_metrics['correct'] / cat_metrics['total']
        final_metrics[f'{category}_average_time'] = cat_metrics['time'] / cat_metrics['total']
    return (final_metrics, detailed_results)

def get_prompt_for_category(question: str, category: str) -> str:
    """
    Generate appropriate prompt based on category.
    """
    if category == 'gsm8k':
        return f"Solve this math problem step by step. After solving, provide the final numerical answer after '### ' (three hash symbols and a space).\n\nQuestion: {question}\n\nShow your work, then give the final answer after '### '."
    elif category == 'mmlu_math':
        return f'Solve this math problem. Provide only the answer with no explanation.\n\nQuestion: {question}'
    elif category == 'boolq':
        return f"Answer this yes/no question with only 'yes' or 'no'.\n\nQuestion: {question}"
    elif category == 'aqua_rat':
        return f'Choose the correct answer. Provide only the letter choice with no explanation.\n\nQuestion: {question}'
    else:
        return f'Question: {question}'

def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on OptiLLM Bench. By default, runs test-time compute evaluation with pass@1, maj@64, and genselect@64.')
    parser.add_argument('--model', required=True, help='Model identifier')
    parser.add_argument('--base-url', default='http://localhost:8000/v1', help='Base URL for API endpoint')
    parser.add_argument('--max-samples', type=int, help='Maximum number of samples to evaluate')
    parser.add_argument('--output-dir', default='results', help='Directory to save results')
    parser.add_argument('--approaches', nargs='+', help='Specific approaches to evaluate (overrides default test-time compute)')
    parser.add_argument('--test-time-compute', action='store_true', help='Evaluate full test-time compute scaling approaches (ThinkDeeper and various k values)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    os.makedirs(args.output_dir, exist_ok=True)
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY environment variable must be set')
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    dataset = load_optillm_bench()
    if args.test_time_compute:
        approaches_config = TEST_TIME_COMPUTE_APPROACHES
        if args.approaches:
            approaches_config = [a for a in TEST_TIME_COMPUTE_APPROACHES if a[0] in args.approaches]
    elif args.approaches:
        all_available_approaches = APPROACHES + TEST_TIME_COMPUTE_APPROACHES + DEFAULT_TEST_TIME_COMPUTE
        approaches_config = []
        for requested_approach in args.approaches:
            found = False
            for approach_tuple in all_available_approaches:
                if approach_tuple[0] == requested_approach:
                    if approach_tuple not in approaches_config:
                        approaches_config.append(approach_tuple)
                    found = True
                    break
            if not found:
                logger.warning(f"Approach '{requested_approach}' not found in any configuration")
        if not approaches_config:
            raise ValueError(f'No valid approaches found. Requested: {args.approaches}')
    else:
        approaches_config = DEFAULT_TEST_TIME_COMPUTE
        logger.info('Using default test-time compute evaluation (avg@5, pass@5, maj@5, genselect@5)')
    all_metrics = {}
    for approach_name, description, extra_body_params in approaches_config:
        logger.info(f'Evaluating approach: {approach_name} - {description}')
        if extra_body_params:
            logger.info(f'Extra parameters: {extra_body_params}')
        try:
            metrics, detailed_results = evaluate_model(client, args.model, dataset, approach_name, extra_body_params, args.max_samples)
            all_metrics[approach_name] = metrics
            save_results(metrics, detailed_results, args.model, approach_name, args.output_dir)
            logger.info(f'Completed evaluation for {approach_name}')
            logger.info(f'Accuracy: {metrics['accuracy'] * 100:.2f}%')
            logger.info(f'Average time per sample: {metrics['average_time']:.2f}s')
        except Exception as e:
            logger.error(f'Error evaluating approach {approach_name}: {e}')
            continue
    is_test_time = args.test_time_compute or (not args.approaches and approaches_config == DEFAULT_TEST_TIME_COMPUTE)
    generate_report(all_metrics, args.output_dir, is_test_time)

def load_optillm_bench() -> datasets.Dataset:
    """Load the OptiLLM Bench dataset."""
    try:
        dataset = load_dataset('codelion/optillmbench')
        return dataset['test']
    except Exception as e:
        logger.error(f'Error loading dataset: {e}')
        raise

def save_results(metrics: Dict[str, float], detailed_results: List[Dict[str, Any]], model: str, approach: str, output_dir: str):
    """Save evaluation results to files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(output_dir, model.replace('/', '_'))
    os.makedirs(model_dir, exist_ok=True)
    base_filename = os.path.join(model_dir, f'{approach}_{timestamp}')
    with open(f'{base_filename}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(f'{base_filename}_detailed.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    df = pd.DataFrame([{k: v for k, v in result.items() if k != 'raw_response' and k != 'processed_response'} for result in detailed_results])
    df.to_csv(f'{base_filename}_summary.csv', index=False)
    logger.info(f'Results saved to {base_filename}_*')

def generate_report(all_metrics: Dict[str, Dict[str, float]], output_dir: str, is_test_time_compute: bool=False):
    """Generate a comprehensive report comparing all approaches."""
    report = []
    is_default_test_time = set(all_metrics.keys()) == {'avg@5', 'pass@5', 'maj@5', 'genselect@5'}
    if is_default_test_time:
        report_title = 'OptiLLM Bench Test-Time Compute Evaluation Report'
    elif is_test_time_compute:
        report_title = 'OptiLLM Bench Test-Time Compute Scaling Report'
    else:
        report_title = 'OptiLLM Bench Evaluation Report'
    report.append(f'# {report_title}')
    report.append(f'Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n')
    if is_default_test_time:
        report.append('## Test-Time Compute Evaluation Results\n')
        report.append('This report evaluates the potential of test-time compute with:')
        report.append('- **avg@5**: Average success rate of 5 parallel responses')
        report.append('- **pass@5**: Success if ANY of 5 responses is correct')
        report.append('- **maj@5**: Majority voting with 5 candidates')
        report.append('- **genselect@5**: Quality-based selection from 5 candidates\n')
        report.append('All approaches use n=5 parallel generation (with sequential fallback) for fair comparison.\n')
    elif is_test_time_compute:
        report.append('This report evaluates test-time compute scaling approaches:')
        report.append('- **Sequential scaling**: ThinkDeeper with varying thinking token budgets')
        report.append('- **Parallel scaling**: Majority voting with varying k values\n')
    report.append('## Overall Results')
    headers = ['Approach', 'Accuracy', 'Avg Time (s)', 'Total Time (s)']
    rows = []
    for approach, metrics in all_metrics.items():
        rows.append([approach, f'{metrics['accuracy'] * 100:.2f}%', f'{metrics['average_time']:.2f}', f'{metrics['total_time']:.2f}'])
    df = pd.DataFrame(rows, columns=headers)
    report.append(df.to_markdown())
    report.append('\n## Results by Category')
    categories = ['gsm8k', 'mmlu_math', 'boolq', 'aqua_rat']
    for category in categories:
        report.append(f'\n### {category.upper()}')
        headers = ['Approach', 'Accuracy', 'Avg Time (s)']
        rows = []
        for approach, metrics in all_metrics.items():
            if f'{category}_accuracy' in metrics:
                rows.append([approach, f'{metrics[f'{category}_accuracy'] * 100:.2f}%', f'{metrics[f'{category}_average_time']:.2f}'])
        df = pd.DataFrame(rows, columns=headers)
        report.append(df.to_markdown())
    if is_default_test_time:
        report.append('\n## Summary')
        if all((metric in all_metrics for metric in ['avg@5', 'pass@5', 'maj@5', 'genselect@5'])):
            avg5_acc = all_metrics['avg@5']['accuracy'] * 100
            pass5_acc = all_metrics['pass@5']['accuracy'] * 100
            maj5_acc = all_metrics['maj@5']['accuracy'] * 100
            genselect5_acc = all_metrics['genselect@5']['accuracy'] * 100
            report.append(f'\n**Key Metrics:**')
            report.append(f'- **avg@5** (average of 5 responses): {avg5_acc:.2f}%')
            report.append(f'- **pass@5** (success if any correct): {pass5_acc:.2f}%')
            report.append(f'- **maj@5** (majority voting): {maj5_acc:.2f}%')
            report.append(f'- **genselect@5** (quality-based selection): {genselect5_acc:.2f}%')
            if avg5_acc > 0:
                pass_improvement = (pass5_acc - avg5_acc) / avg5_acc * 100
                maj_improvement = (maj5_acc - avg5_acc) / avg5_acc * 100
                genselect_improvement = (genselect5_acc - avg5_acc) / avg5_acc * 100
                report.append(f'\n**Improvements over avg@5 baseline:**')
                report.append(f'- pass@5: {('+' if pass_improvement > 0 else '')}{pass_improvement:.1f}%')
                report.append(f'- maj@5: {('+' if maj_improvement > 0 else '')}{maj_improvement:.1f}%')
                report.append(f'- genselect@5: {('+' if genselect_improvement > 0 else '')}{genselect_improvement:.1f}%')
            if pass5_acc > avg5_acc:
                variance_ratio = (pass5_acc - avg5_acc) / avg5_acc * 100
                report.append(f'\n**Response Variance Indicator:**')
                report.append(f'- Gap between pass@5 and avg@5: {variance_ratio:.1f}%')
                report.append(f'- This indicates {('high' if variance_ratio > 50 else 'moderate' if variance_ratio > 20 else 'low')} variance in response quality')
    report_path = f'{output_dir}/evaluation_report.md'
    with open(report_path, 'w') as f:
        f.write('\n\n'.join(report))
    logger.info(f'Report saved to {report_path}')

class SimpleQAEvaluator:
    """Main evaluator class for SimpleQA benchmark"""

    def __init__(self, model: str, approach: str, base_url: str=DEFAULT_BASE_URL, grader_model: str=DEFAULT_GRADER_MODEL, timeout: int=DEFAULT_TIMEOUT, cache_dir: str='cache', output_dir: str='results', use_verified: bool=False):
        self.model = model
        self.approach = approach
        self.base_url = base_url
        self.grader_model = grader_model
        self.timeout = timeout
        self.use_verified = use_verified
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.optillm_client = OpenAI(api_key='optillm', base_url=base_url, timeout=httpx.Timeout(timeout, connect=5.0), max_retries=0)
        try:
            self.grader_client = OpenAI(api_key='optillm', base_url=base_url, timeout=httpx.Timeout(timeout, connect=5.0), max_retries=0)
            logger.info('Using OptILLM for grading responses')
        except Exception as e:
            logger.warning(f'Could not initialize grader client: {e}')
            logger.warning('Grading will be skipped.')
            self.grader_client = None
        self.results = []
        self.metrics = {'correct': 0, 'incorrect': 0, 'not_attempted': 0, 'errors': 0, 'total_processed': 0}

    def download_dataset(self) -> str:
        """Download SimpleQA dataset if not cached"""
        if self.use_verified:
            cache_file = self.cache_dir / 'simpleqa_verified.csv'
            url = SIMPLEQA_VERIFIED_CSV_URL
            dataset_name = 'SimpleQA-Verified'
        else:
            cache_file = self.cache_dir / 'simple_qa_test_set.csv'
            url = SIMPLEQA_CSV_URL
            dataset_name = 'SimpleQA'
        if cache_file.exists():
            logger.info(f'Using cached {dataset_name} dataset: {cache_file}')
            return str(cache_file)
        logger.info(f'Downloading {dataset_name} dataset from {url}')
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(cache_file, 'wb') as f:
                f.write(response.content)
            logger.info(f'Dataset downloaded to {cache_file}')
            return str(cache_file)
        except Exception as e:
            logger.error(f'Failed to download dataset: {e}')
            raise

    def load_dataset(self, num_samples: Optional[int]=None, start_index: int=0) -> List[Dict]:
        """Load and parse SimpleQA dataset"""
        dataset_file = self.download_dataset()
        questions = []
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i < start_index:
                        continue
                    if num_samples and len(questions) >= num_samples:
                        break
                    if self.use_verified:
                        metadata = {'original_index': row.get('original_index', i), 'topic': row.get('topic', ''), 'answer_type': row.get('answer_type', ''), 'multi_step': row.get('multi_step', ''), 'requires_reasoning': row.get('requires_reasoning', ''), 'urls': row.get('urls', '')}
                        question_id = row.get('original_index', i)
                    else:
                        try:
                            metadata = json.loads(row['metadata']) if row.get('metadata') else {}
                        except:
                            metadata = {}
                        question_id = i
                    question_data = {'id': question_id, 'metadata': metadata, 'question': row['problem'], 'gold_answer': row['answer']}
                    questions.append(question_data)
            dataset_type = 'SimpleQA-Verified' if self.use_verified else 'SimpleQA'
            logger.info(f'Loaded {len(questions)} questions from {dataset_type} dataset')
            return questions
        except Exception as e:
            logger.error(f'Failed to load dataset: {e}')
            raise

    def get_approach_config(self) -> Dict:
        """Get configuration for specific approach"""
        if self.approach == 'none':
            return {}
        elif self.approach == 'web_search':
            return {'num_results': 10, 'headless': True, 'timeout': 30}
        elif self.approach == 'deep_research':
            return {'max_iterations': 1, 'max_sources': 10}
        else:
            return {}

    def query_optillm(self, question: str) -> Tuple[str, bool]:
        """Query OptILLM with the specified approach"""
        try:
            if self.approach == 'none':
                model_name = self.model
            else:
                model_name = f'{self.approach}-{self.model}'
            messages = [{'role': 'system', 'content': 'You are a helpful assistant that provides accurate, factual answers to questions. Be direct and concise.'}, {'role': 'user', 'content': question}]
            extra_body = {}
            approach_config = self.get_approach_config()
            if approach_config:
                extra_body.update(approach_config)
            logger.debug(f'Querying model: {model_name}')
            logger.debug(f'Question: {question}')
            response = self.optillm_client.chat.completions.create(model=model_name, messages=messages, extra_body=extra_body if extra_body else None, max_tokens=4096, temperature=0.6)
            answer = response.choices[0].message.content
            answer = remove_thinking_blocks(answer)
            logger.debug(f'Response: {answer}')
            return (answer, True)
        except Exception as e:
            logger.error(f'Error querying OptILLM: {e}')
            return (f'Error: {str(e)}', False)

    def grade_response(self, question: str, gold_answer: str, response: str) -> str:
        """Grade response using SimpleQA methodology"""
        if not self.grader_client:
            return 'NOT_GRADED'
        try:
            grading_prompt = GRADING_PROMPT.format(question=question, gold_answer=gold_answer, response=response)
            grader_response = self.grader_client.chat.completions.create(model=self.grader_model, messages=[{'role': 'user', 'content': grading_prompt}], temperature=0.6, max_tokens=4096)
            grade_text = grader_response.choices[0].message.content.strip()
            grade_text = re.sub('<think>.*?</think>', '', grade_text, flags=re.DOTALL).strip()
            if grade_text.startswith('A'):
                return 'CORRECT'
            elif grade_text.startswith('B'):
                return 'INCORRECT'
            elif grade_text.startswith('C'):
                return 'NOT_ATTEMPTED'
            else:
                logger.warning(f'Unexpected grade format: {grade_text}')
                return 'NOT_GRADED'
        except Exception as e:
            logger.error(f'Error grading response: {e}')
            return 'ERROR_GRADING'

    def evaluate_question(self, question_data: Dict) -> Dict:
        """Evaluate a single question"""
        question = question_data['question']
        gold_answer = question_data['gold_answer']
        response, success = self.query_optillm(question)
        result = {'id': question_data['id'], 'metadata': question_data['metadata'], 'question': question, 'gold_answer': gold_answer, 'response': response, 'success': success, 'timestamp': datetime.now().isoformat()}
        if success:
            grade = self.grade_response(question, gold_answer, response)
            result['grade'] = grade
            if grade == 'CORRECT':
                self.metrics['correct'] += 1
            elif grade == 'INCORRECT':
                self.metrics['incorrect'] += 1
            elif grade == 'NOT_ATTEMPTED':
                self.metrics['not_attempted'] += 1
        else:
            result['grade'] = 'ERROR'
            self.metrics['errors'] += 1
        self.metrics['total_processed'] += 1
        return result

    def calculate_metrics(self) -> Dict:
        """Calculate final evaluation metrics"""
        total = self.metrics['total_processed']
        correct = self.metrics['correct']
        incorrect = self.metrics['incorrect']
        not_attempted = self.metrics['not_attempted']
        errors = self.metrics['errors']
        if total == 0:
            return {'error': 'No questions processed'}
        accuracy = correct / total * 100 if total > 0 else 0
        attempted = correct + incorrect
        correct_given_attempted = correct / attempted * 100 if attempted > 0 else 0
        precision = correct / (correct + incorrect) if correct + incorrect > 0 else 0
        recall = correct / (correct + not_attempted) if correct + not_attempted > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        return {'total_questions': total, 'correct': correct, 'incorrect': incorrect, 'not_attempted': not_attempted, 'errors': errors, 'accuracy': accuracy, 'correct_given_attempted': correct_given_attempted, 'precision': precision, 'recall': recall, 'f1_score': f1_score, 'attempted_rate': attempted / total * 100 if total > 0 else 0}

    def save_results(self, timestamp: str) -> Tuple[str, str, str]:
        """Save evaluation results to files"""
        dataset_suffix = '_verified' if self.use_verified else ''
        run_dir = self.output_dir / f'simpleqa{dataset_suffix}_{self.model}_{self.approach}'
        run_dir.mkdir(parents=True, exist_ok=True)
        detailed_file = run_dir / f'{timestamp}_detailed.json'
        metrics_file = run_dir / f'{timestamp}_metrics.json'
        summary_file = run_dir / f'{timestamp}_summary.csv'
        with open(detailed_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        final_metrics = self.calculate_metrics()
        final_metrics.update({'model': self.model, 'approach': self.approach, 'timestamp': timestamp, 'base_url': self.base_url, 'grader_model': self.grader_model})
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        df = pd.DataFrame(self.results)
        df.to_csv(summary_file, index=False)
        logger.info(f'Results saved to {run_dir}')
        return (str(detailed_file), str(metrics_file), str(summary_file))

    def run_evaluation(self, num_samples: Optional[int]=None, start_index: int=0) -> Dict:
        """Run the complete evaluation"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_type = 'SimpleQA-Verified' if self.use_verified else 'SimpleQA'
        logger.info(f'Starting {dataset_type} evaluation')
        logger.info(f'Model: {self.model}')
        logger.info(f'Approach: {self.approach}')
        logger.info(f'Dataset: {dataset_type} ({('1k verified questions' if self.use_verified else '4.3k questions')})')
        logger.info(f'Base URL: {self.base_url}')
        logger.info(f'Timeout: {self.timeout}s')
        questions = self.load_dataset(num_samples, start_index)
        for question_data in tqdm(questions, desc='Evaluating questions'):
            try:
                result = self.evaluate_question(question_data)
                self.results.append(result)
                if len(self.results) % 10 == 0:
                    metrics = self.calculate_metrics()
                    logger.info(f'Progress: {len(self.results)}/{len(questions)} - Accuracy: {metrics['accuracy']:.1f}%')
            except KeyboardInterrupt:
                logger.info('Evaluation interrupted by user')
                break
            except Exception as e:
                logger.error(f'Error evaluating question {question_data['id']}: {e}')
                continue
        detailed_file, metrics_file, summary_file = self.save_results(timestamp)
        final_metrics = self.calculate_metrics()
        logger.info('Evaluation completed!')
        logger.info(f'Total questions: {final_metrics['total_questions']}')
        logger.info(f'Accuracy: {final_metrics['accuracy']:.1f}%')
        logger.info(f'F1 Score: {final_metrics['f1_score']:.3f}')
        logger.info(f'Correct: {final_metrics['correct']}')
        logger.info(f'Incorrect: {final_metrics['incorrect']}')
        logger.info(f'Not Attempted: {final_metrics['not_attempted']}')
        return final_metrics

