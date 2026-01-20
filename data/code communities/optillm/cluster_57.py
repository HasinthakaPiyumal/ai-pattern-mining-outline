# Cluster 57

def run_tests(test_cases: List[Dict], approaches: List[str], client, model: str, single_test_name: str=None) -> List[Dict]:
    results = []
    for test_case in test_cases:
        if single_test_name is None or test_case['name'] == single_test_name:
            result = run_test_case(test_case, approaches, client, model)
            results.append(result)
            logger.info(f'Completed test case: {test_case['name']}')
        if single_test_name and test_case['name'] == single_test_name:
            break
    return results

def run_test_case(test_case: Dict, approaches: List[str], client, model: str) -> Dict:
    system_prompt = test_case['system_prompt']
    query = test_case['query']
    results = []
    with ThreadPoolExecutor() as executor:
        future_to_approach = {executor.submit(run_approach, approach, system_prompt, query, client, model): approach for approach in approaches}
        for future in as_completed(future_to_approach):
            results.append(future.result())
    return {'test_case': test_case, 'results': results}

