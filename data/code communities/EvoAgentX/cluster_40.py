# Cluster 40

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    import itertools
    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)
    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def estimator(n: int, c: int, k: int) -> float:
    """Calculates 1 - comb(n - c, k) / comb(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def compute_metrics_from_results(results, k_list=[1, 5]):
    total = []
    correct = []
    task_ids = []
    for task_id, res in results.items():
        all_correct = []
        for generation in res:
            gen = np.array(generation)
            all_correct.append(np.all(gen > 0))
        task_ids.append(task_id)
        total.append(len(all_correct))
        correct.append(sum(all_correct))
    total = np.array(total)
    correct = np.array(correct)
    ks = k_list
    detail_pass_at_k = {f'pass@{k}': estimate_pass_at_k(total, correct, k).tolist() for k in ks if (total >= k).all()}
    pass_at_k = {f'pass@{k}': estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}
    detail_metrics = {k: dict(zip(task_ids, v)) for k, v in detail_pass_at_k.items()}
    pass_at_k['detail'] = detail_metrics
    return pass_at_k

def codegen_metrics(samples_list, generations_list, k_list=[1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000], num_process_evaluate=16, timeout=6, debug=False):
    samples_linear = []
    generations_linear = []
    remap_index = []
    results = defaultdict(list)
    metadatas = defaultdict(list)
    for idx, (sample, generation_list) in enumerate(zip(samples_list, generations_list)):
        assert isinstance(generation_list, list), generations_list[0]
        for generation in generation_list:
            assert isinstance(generation, str), generations_list[0]
            samples_linear.append(sample)
            generations_linear.append([generation])
            remap_index.append(idx)
    results_linear, metadatas_linear = evaluate_generations(samples_linear, generations_linear, debug=debug, num_process_evaluate=num_process_evaluate, timeout=timeout)
    for idx, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
        results[remap_index[idx]].append(sub_results[0])
    for idx, sub_metadatas in sorted(metadatas_linear.items(), key=lambda x: x[0]):
        metadatas[remap_index[idx]].append(sub_metadatas[0])
    metrics = compute_metrics_from_results(results, k_list=k_list)
    final_metadata = []
    for key in sorted(list(metadatas.keys())):
        final_metadata.append(metadatas[key])
    for i in range(len(final_metadata)):
        if type(final_metadata[i]) is not list:
            final_metadata[i] = [json.dumps(final_metadata[i])]
        else:
            final_metadata[i] = [json.dumps(x) for x in final_metadata[i]]
        assert len(final_metadata[i]) == len(generations_list[0]), f'len(final_metadata[i])={len(final_metadata[i])!r}'
    return [metrics, results, final_metadata]

def evaluate_generations(samples_list: list, generations_list: list[list[str]], debug: bool=False, num_process_evaluate: int=16, timeout=6):
    """We take the list of code generations and try to compile them
     and the run their corresponding unit tests which are retrieved from the APPS dataset.

    Args:
        generations: list of code generations (same order as samples in APPS dataset)
        level: difficulty level used in the generation, can be "all", "introductory", "interview" or "competition"

    Returns:
        results: dictionary of results, key is the problem index, value is a list of results for each generation
    """
    inputs = [[(generations_list[index], samples_list[index], debug, timeout), index] for index in range(len(generations_list))]
    with ProcessPoolExecutor(max_workers=1 if debug else num_process_evaluate) as executor:
        futures = {executor.submit(evaluate_generations_by_problem, arg): index for arg, index in inputs}
        results = {}
        metadata = {}
        for future in as_completed(futures):
            index = futures[future]
            results[index], metadata[index] = future.result()
    assert len(results) == len(inputs), f'results = {len(results)} inputs = {len(inputs)} results={results!r}'
    return (results, metadata)

def check_testcase_output(testcase_str, expected_output):
    if len(testcase_str.splitlines()) > 1:
        for line in testcase_str.splitlines():
            if line.startswith('#'):
                continue
            if 'assert' in line:
                testcase_str = line
                break
    testcase_str = testcase_str.strip()
    if 'assert' in testcase_str:
        testcase_output_str = str(parse_assert_statement(testcase_str))
    else:
        testcase_output_str = testcase_str
    global_result = None
    try:
        testcase_output_eval = eval(testcase_output_str)
    except Exception:
        global_result = False
    try:
        expected_output_eval = json.loads(expected_output)
    except Exception:
        global_result = False
        print('Failed to eval expected testcase output', expected_output)
    if global_result is None:
        global_result = testcase_output_eval == expected_output_eval
    return global_result

def parse_assert_statement(statement):
    """
    Parse a Python assert statement and extract the expected output
    from the right side of the '==' operator as a string.

    :param statement: A string containing the assert statement.
    :return: The expected output from the assert statement as a string.
    """
    try:
        parsed = ast.parse(statement, mode='exec')
    except SyntaxError:
        return 'Invalid syntax'
    if len(parsed.body) == 0:
        return 'Empty statement'
    if not isinstance(parsed.body[0], ast.Assert):
        return 'Not an assert statement'
    comparison = parsed.body[0].test
    if not isinstance(comparison, ast.Compare) or not isinstance(comparison.ops[0], ast.Eq):
        return 'Not an equality assertion'
    return ast.get_source_segment(statement, comparison.comparators[0])

def test_output_metrics(samples, generations, k_list=[1, 5]):
    num_samples = len(samples)
    results = []
    for idx in range(num_samples):
        idx_results = []
        sample = samples[idx]
        extracted_generation_list = generations[idx]
        for extracted_generation in extracted_generation_list:
            global_result = check_testcase_output(extracted_generation, sample['output'])
            idx_results.append([global_result])
        results.append(idx_results)
    results = {result_idx: results[result_idx] for result_idx in range(len(results))}
    metrics = compute_metrics_from_results(results, k_list=k_list)
    return [metrics, results]

