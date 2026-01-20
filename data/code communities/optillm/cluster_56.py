# Cluster 56

def test_mars_agent_temperatures():
    """Test that MARS uses different temperatures for agents"""
    from optillm.mars.mars import DEFAULT_CONFIG
    from optillm.mars.agent import MARSAgent
    client = MockOpenAIClient()
    model = 'mock-model'
    config = DEFAULT_CONFIG.copy()
    agents = []
    for i in range(config['num_agents']):
        agent = MARSAgent(i, client, model, config)
        agents.append(agent)
    temperatures = [agent.temperature for agent in agents]
    unique_temps = set(temperatures)
    assert len(unique_temps) == len(agents), 'Agents should have different temperatures'
    assert 0.3 in temperatures, 'Should have conservative agent (temp 0.3)'
    assert 1.0 in temperatures, 'Should have creative agent (temp 1.0)'
    print(f'✅ Agent temperatures test passed: {temperatures}')

def run_tests():
    """Run all MARS tests"""
    print('Running MARS comprehensive tests...')
    print('=' * 80)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMARSParallel)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    try:
        test_mars_agent_temperatures()
    except Exception as e:
        print(f'❌ Agent temperatures test failed: {e}')
    print('=' * 60)
    if result.wasSuccessful():
        print('🎉 All MARS tests passed!')
        return True
    else:
        print('❌ Some MARS tests failed')
        return False

def main():
    parser = argparse.ArgumentParser(description='Test different LLM inference approaches.')
    parser.add_argument('--test_cases', type=str, default=None, help='Path to test cases JSON file')
    parser.add_argument('--approaches', nargs='+', default=list(APPROACHES.keys()), help='Approaches to test')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model to use for testing')
    parser.add_argument('--base-url', type=str, default=None, help='The base_url for the OpenAI API compatible endpoint')
    parser.add_argument('--single-test', type=str, default=None, help='Name of a single test case to run')
    args = parser.parse_args()
    if args.test_cases is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.test_cases = os.path.join(script_dir, 'test_cases.json')
    if API_KEY == 'optillm' and args.model == 'gpt-4o-mini':
        args.model = TEST_MODEL
        logger.info(f'Using local model: {args.model}')
    if API_KEY == 'optillm':
        os.environ['OPTILLM_API_KEY'] = 'optillm'
    test_cases = load_test_cases(args.test_cases)
    if args.base_url:
        client = OpenAI(api_key=API_KEY, base_url=args.base_url)
    elif API_KEY == 'optillm':
        client = OpenAI(api_key=API_KEY, base_url='http://localhost:8000/v1')
        logger.info('Using local inference endpoint: http://localhost:8000/v1')
    else:
        client = OpenAI(api_key=API_KEY)
    results = run_tests(test_cases, args.approaches, client, args.model, args.single_test)
    print_summary(results)
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

def load_test_cases(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

def print_summary(results: List[Dict]):
    print('\n=== Test Results Summary ===')
    for test_result in results:
        print(f'\nTest Case: {test_result['test_case']['name']}')
        for approach_result in test_result['results']:
            status = '✅' if approach_result['status'] == 'success' else '❌'
            print(f'  {status} {approach_result['approach']}: {approach_result['time']:.2f}s')
            if approach_result['status'] == 'error':
                print(f'     Error: {approach_result['result']}')

