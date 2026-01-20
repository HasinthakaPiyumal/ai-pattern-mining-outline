# Cluster 2

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM performance on IMO 2025 problems')
    parser.add_argument('--model', type=str, required=True, help='Model to use (e.g., google/gemma-2.5-flash-lite)')
    parser.add_argument('--approach', type=str, default='none', help='OptiLLM approach to use (none, mars, moa, bon, etc.)')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds for each problem (default: 600)')
    parser.add_argument('--problems', type=str, help="Comma-separated list of problem IDs to evaluate (e.g., '1,3,5')")
    args = parser.parse_args()
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'results/imo25_{args.model.replace('/', '_')}_{args.approach}_{timestamp}.json'
    if args.problems:
        problem_ids = [int(x.strip()) for x in args.problems.split(',')]
        problems_to_evaluate = [p for p in IMO_2025_PROBLEMS if p['id'] in problem_ids]
    else:
        problems_to_evaluate = IMO_2025_PROBLEMS
    print(f'Evaluating {len(problems_to_evaluate)} IMO 2025 problems')
    print(f'Model: {args.model}')
    print(f'Approach: {args.approach}')
    print(f'Results will be saved to: {results_file}')
    if args.approach == 'mars':
        extra_body = {'optillm_approach': 'mars', 'mars_config': {'use_thinking_tags': False, 'answer_extraction_mode': 'none'}}
    elif args.approach != 'none':
        extra_body = {'optillm_approach': args.approach}
    else:
        extra_body = None
    for problem_data in tqdm(problems_to_evaluate, desc='Solving IMO problems'):
        logger.info(f'Evaluating Problem {problem_data['id']}: {problem_data['type']}')
        start_time = time.time()
        response = get_llm_response(problem_data['problem'], args.model, extra_body, args.timeout)
        solve_time = time.time() - start_time
        evaluation = evaluate_solution(problem_data, response['solution'], args.model)
        result = {'timestamp': datetime.now().isoformat(), 'model': args.model, 'approach': args.approach, 'problem_data': problem_data, 'response': response, 'evaluation': evaluation, 'solve_time_seconds': solve_time}
        save_result(results_file, result)
        logger.info(f'Problem {problem_data['id']} completed - Score: {evaluation['correctness_score']:.3f}')
    final_results = load_existing_results(results_file)
    analyze_results(final_results, args.approach)
    print(f'\nEvaluation complete! Results saved to: {results_file}')

def rank_responses(responses: List[Dict[str, Any]], targets: List[str]) -> List[int]:
    """Rank responses based on correctness and token efficiency."""
    ranked_data = []
    for i, response in enumerate(responses):
        is_correct = is_correct_response(response['content'], targets)
        ranked_data.append((i, is_correct, response['tokens']))
    ranked_data.sort(key=lambda x: (-int(x[1]), x[2]))
    return [idx for idx, _, _ in ranked_data]

def is_correct_response(response: str, targets: List[str]) -> bool:
    """Check if response matches any of the target answers."""
    response = response.strip().lower()
    return any((target.strip().lower() == response for target in targets))

def construct_prompt(sample: Dict[str, Any], split_type: str) -> str:
    """Construct prompt based on split type."""
    context = sample.get('context', '')
    prompt = sample['prompt']
    if split_type == 'multiple_choice':
        options = sample['options']
        options_text = '\nOptions:\n' + '\n'.join([f'{i + 1}. {opt}' for i, opt in enumerate(options)])
        return f'Context: {context}\n\nQuestion: {prompt}{options_text}\n\nProvide the correct answer from the options above.'
    else:
        return f'Context: {context}\n\nQuestion: {prompt}\n\nProvide your answer.'

