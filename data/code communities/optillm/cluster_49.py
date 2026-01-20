# Cluster 49

class TestMARSIMO25(unittest.TestCase):
    """Test MARS on specific IMO25 problems"""

    def setUp(self):
        """Set up test fixtures with logging capture"""
        self.system_prompt = 'You are a mathematical problem solver capable of handling complex olympiad-level problems.'
        self.model = 'mock-model'
        self.log_capture = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_capture)
        self.log_handler.setLevel(logging.INFO)
        mars_logger = logging.getLogger('optillm.mars')
        mars_logger.addHandler(self.log_handler)
        mars_logger.setLevel(logging.INFO)
        self.original_level = mars_logger.level

    def tearDown(self):
        """Clean up test fixtures"""
        mars_logger = logging.getLogger('optillm.mars')
        mars_logger.removeHandler(self.log_handler)
        mars_logger.setLevel(self.original_level)
        self.log_handler.close()

    def get_captured_logs(self):
        """Get the captured log output"""
        return self.log_capture.getvalue()

    def test_imo25_problem3_functional_equation(self):
        """Test MARS on IMO25 Problem 3 - Functional Equation (Expected: c = 4)"""
        problem3 = 'Let ℕ denote the set of positive integers. A function f:ℕ→ℕ is said to be bonza if f(a) divides b^a-f(b)^{f(a)} for all positive integers a and b.\n\nDetermine the smallest real constant c such that f(n)≤cn for all bonza functions f and all positive integers n.'
        print(f'\n🧮 Testing MARS on IMO25 Problem 3 (Expected answer: c = 4)...')
        client = MockOpenAIClient(response_delay=0.05, reasoning_tokens=3000)
        start_time = time.time()
        result = multi_agent_reasoning_system(self.system_prompt, problem3, client, self.model)
        execution_time = time.time() - start_time
        self.assertIsInstance(result, tuple)
        response, tokens = result
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 100, 'Response should be substantial for IMO problem')
        self.assertGreater(tokens, 0)
        has_answer_4 = '4' in response
        has_constant_c = 'c' in response.lower()
        print(f'  📊 Execution time: {execution_time:.2f}s')
        print(f'  📊 Response length: {len(response):,} characters')
        print(f'  📊 Total tokens: {tokens:,}')
        print(f'  📊 API calls made: {client.call_count}')
        print(f"  🎯 Contains answer '4': {has_answer_4}")
        print(f"  🎯 Contains 'constant c': {has_constant_c}")
        logs = self.get_captured_logs()
        voting_logs = [line for line in logs.split('\n') if '🗳️  VOTING' in line]
        synthesis_logs = [line for line in logs.split('\n') if '🤝 SYNTHESIS' in line]
        print(f'  📋 Voting log entries: {len(voting_logs)}')
        print(f'  📋 Synthesis log entries: {len(synthesis_logs)}')
        if voting_logs:
            print(f'  📋 Sample voting log: {voting_logs[0][:100]}...')
        answer_extraction_logs = [line for line in logs.split('\n') if 'extracted answer' in line.lower()]
        if answer_extraction_logs:
            print(f'  🔍 Answer extraction logs found: {len(answer_extraction_logs)}')
            for log in answer_extraction_logs[:3]:
                print(f'    {log}')
        response_lines = response.split('\n')
        key_lines = [line for line in response_lines if any((keyword in line.lower() for keyword in ['constant', 'c =', 'answer', '= 4', 'therefore']))]
        if key_lines:
            print(f'  🔑 Key response lines:')
            for line in key_lines[:5]:
                print(f'    {line.strip()}')
        print(f'✅ IMO25 Problem 3 test completed')

    def test_imo25_problem4_number_theory(self):
        """Test MARS on IMO25 Problem 4 - Number Theory (Expected: 6J·12^K formula)"""
        problem4 = 'A proper divisor of a positive integer N is a positive divisor of N other than N itself.\n\nThe infinite sequence a_1,a_2,… consists of positive integers, each of which has at least three proper divisors. For each n≥1, the integer a_{n+1} is the sum of three largest proper divisors of a_n.\n\nDetermine all possible values of a_1.'
        print(f'\n🔢 Testing MARS on IMO25 Problem 4 (Expected: 6J·12^K formula)...')
        client = MockOpenAIClient(response_delay=0.05, reasoning_tokens=3000)
        start_time = time.time()
        result = multi_agent_reasoning_system(self.system_prompt, problem4, client, self.model)
        execution_time = time.time() - start_time
        self.assertIsInstance(result, tuple)
        response, tokens = result
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 100, 'Response should be substantial for IMO problem')
        has_formula_6J = '6J' in response or '6j' in response.lower()
        has_formula_12K = '12^K' in response or '12^k' in response.lower()
        has_gcd_condition = 'gcd' in response.lower()
        print(f'  📊 Execution time: {execution_time:.2f}s')
        print(f'  📊 Response length: {len(response):,} characters')
        print(f"  🎯 Contains '6J': {has_formula_6J}")
        print(f"  🎯 Contains '12^K': {has_formula_12K}")
        print(f"  🎯 Contains 'gcd': {has_gcd_condition}")
        print(f'✅ IMO25 Problem 4 test completed')

    def test_answer_extraction_analysis(self):
        """Test answer extraction specifically with controlled responses"""
        print(f'\n🔍 Testing answer extraction with controlled responses...')

        class ControlledMockClient(MockOpenAIClient):

            def __init__(self):
                super().__init__(response_delay=0.01, reasoning_tokens=1000)
                self.response_index = 0
                self.controlled_responses = ['After careful analysis, I determine that the smallest constant c = 4. This can be proven by construction and bounds analysis.', 'The minimum value is c = 4. Therefore, the answer is 4.', 'Through systematic analysis, the constant c must equal 4. The final answer is c = 4.']

            def chat_completions_create(self, **kwargs):
                result = super().chat_completions_create(**kwargs)
                if self.response_index < len(self.controlled_responses):
                    result.choices[0].message.content = self.controlled_responses[self.response_index]
                    self.response_index += 1
                return result
        simple_problem = 'Find the smallest constant c such that f(n) ≤ cn for all valid functions f.'
        client = ControlledMockClient()
        result = multi_agent_reasoning_system(self.system_prompt, simple_problem, client, self.model)
        response, tokens = result
        logs = self.get_captured_logs()
        voting_logs = [line for line in logs.split('\n') if 'VOTING' in line and 'extracted answer' in line.lower()]
        print(f"  📊 Response contains '4': {'4' in response}")
        print(f"  📊 Response contains 'c = 4': {'c = 4' in response}")
        print(f'  📋 Voting logs with extraction: {len(voting_logs)}')
        if voting_logs:
            for i, log in enumerate(voting_logs[:3]):
                print(f'    Vote {i + 1}: {log}')
        print(f'✅ Answer extraction analysis completed')

def multi_agent_reasoning_system(system_prompt: str, initial_query: str, client, model: str, request_config: dict=None, request_id: str=None) -> Tuple[str, int]:
    """
    Main MARS function implementing multi-agent reasoning with parallel execution

    Args:
        system_prompt: System-level instructions
        initial_query: The problem or task to solve
        client: OpenAI-compatible client for API calls
        model: Model identifier (should support OpenRouter reasoning API)
        request_id: Optional request ID for conversation logging

    Returns:
        Tuple of (final_solution, total_reasoning_tokens)
    """
    return asyncio.run(_run_mars_parallel(system_prompt, initial_query, client, model, request_config, request_id))

class TestMARSParallel(unittest.TestCase):
    """Test MARS parallel execution functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.system_prompt = 'You are a mathematical problem solver.'
        self.test_query = 'What is the value of x if 2x + 5 = 15?'
        self.model = 'mock-model'
        self.log_capture = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_capture)
        self.log_handler.setLevel(logging.INFO)
        mars_logger = logging.getLogger('optillm.mars')
        mars_logger.addHandler(self.log_handler)
        mars_logger.setLevel(logging.INFO)
        self.original_level = mars_logger.level

    def tearDown(self):
        """Clean up test fixtures"""
        mars_logger = logging.getLogger('optillm.mars')
        mars_logger.removeHandler(self.log_handler)
        mars_logger.setLevel(self.original_level)
        self.log_handler.close()

    def get_captured_logs(self):
        """Get the captured log output"""
        return self.log_capture.getvalue()

    def test_mars_import(self):
        """Test that MARS can be imported correctly"""
        from optillm.mars import multi_agent_reasoning_system
        self.assertTrue(callable(multi_agent_reasoning_system))

    def test_mars_basic_call(self):
        """Test basic MARS functionality with mock client"""
        client = MockOpenAIClient(response_delay=0.01)
        try:
            result = multi_agent_reasoning_system(self.system_prompt, self.test_query, client, self.model)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            response, tokens = result
            self.assertIsInstance(response, str)
            self.assertIsInstance(tokens, int)
            self.assertGreater(len(response), 0)
            self.assertGreater(tokens, 0)
            print('✅ MARS basic call test passed')
        except Exception as e:
            self.fail(f'MARS basic call failed: {e}')

    def test_mars_parallel_execution_performance(self):
        """Test that parallel execution shows improvement over theoretical sequential"""
        client = MockOpenAIClient(response_delay=0.05, reasoning_tokens=2000)
        start_time = time.time()
        result = multi_agent_reasoning_system(self.system_prompt, self.test_query, client, self.model)
        end_time = time.time()
        execution_time = end_time - start_time
        self.assertLess(execution_time, 30, f'Execution took {execution_time:.2f}s, too long for test')
        self.assertIsInstance(result, tuple)
        response, tokens = result
        self.assertGreater(len(response), 0)
        self.assertGreater(tokens, 0)
        call_times = client.call_times
        if len(call_times) >= 3:
            first_three = call_times[:3]
            time_spread = max(first_three) - min(first_three)
            self.assertLess(time_spread, 0.5, f'First 3 calls spread over {time_spread:.2f}s, not parallel enough')
        logs = self.get_captured_logs()
        self.assertIn('🚀 MARS', logs, 'Should contain main orchestration logs')
        print(f'✅ MARS parallel execution completed in {execution_time:.2f}s with {client.call_count} API calls')
        print(f'📋 Captured {len(logs.split('🚀'))} main log entries')

    def test_mars_worker_pool_calculation(self):
        """Test that worker pool size is calculated correctly"""
        from optillm.mars.mars import DEFAULT_CONFIG
        num_agents = DEFAULT_CONFIG['num_agents']
        verification_passes = DEFAULT_CONFIG['verification_passes_required']
        expected_workers = max(num_agents, num_agents * min(2, verification_passes))
        self.assertEqual(expected_workers, 6)
        print(f'✅ Worker pool size calculation correct: {expected_workers} workers')

    def test_mars_error_handling(self):
        """Test error handling in parallel execution"""

        class FailingMockClient(MockOpenAIClient):

            def __init__(self):
                super().__init__(response_delay=0.01)
                self.failure_count = 0

            def chat_completions_create(self, **kwargs):
                self.failure_count += 1
                if self.failure_count % 3 == 0:
                    raise Exception('Mock API failure')
                return super().chat_completions_create(**kwargs)
        failing_client = FailingMockClient()
        try:
            result = multi_agent_reasoning_system(self.system_prompt, self.test_query, failing_client, self.model)
            self.assertIsInstance(result, tuple)
            response, tokens = result
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            print('✅ MARS error handling test passed')
        except Exception as e:
            self.assertIn('MARS system encountered an error', str(e))
            print('✅ MARS fallback error handling works')

    @patch('optillm.mars.mars.ThreadPoolExecutor')
    def test_mars_uses_thread_pool(self, mock_thread_pool):
        """Test that MARS actually uses ThreadPoolExecutor for parallel execution"""
        mock_executor = Mock()
        mock_thread_pool.return_value.__enter__.return_value = mock_executor
        client = MockOpenAIClient(response_delay=0.01)
        multi_agent_reasoning_system(self.system_prompt, self.test_query, client, self.model)
        mock_thread_pool.assert_called_once()
        call_args = mock_thread_pool.call_args
        self.assertIn('max_workers', call_args.kwargs)
        self.assertEqual(call_args.kwargs['max_workers'], 6)
        print('✅ MARS ThreadPoolExecutor usage test passed')

    def test_mars_hard_problems(self):
        """Test MARS on challenging problems that require deep reasoning"""
        hard_problems = [{'name': 'Advanced Algebra', 'problem': 'Find all positive integer solutions to x^3 + y^3 = z^3 - 1 where x, y, z are all less than 100.', 'expected_features': ['systematic', 'case', 'analysis']}, {'name': 'Number Theory', 'problem': 'Prove that there are infinitely many primes of the form 4k+3.', 'expected_features': ['proof', 'contradiction', 'infinite']}, {'name': 'Combinatorics', 'problem': 'In how many ways can 20 identical balls be distributed into 5 distinct boxes such that each box contains at least 2 balls?', 'expected_features': ['stars', 'bars', 'constraint']}, {'name': 'Geometry', 'problem': 'Given a triangle ABC with sides a, b, c, prove that a^2 + b^2 + c^2 ≥ 4√3 * Area.', 'expected_features': ['inequality', 'area', 'geometric']}]

        class EnhancedMockClient(MockOpenAIClient):

            def __init__(self):
                super().__init__(response_delay=0.1, reasoning_tokens=3000)
                self.problem_responses = {'Advanced Algebra': 'This requires systematic case analysis. Let me examine small values systematically. After checking cases x,y,z < 100, the equation x³ + y³ = z³ - 1 has solutions like (x,y,z) = (1,1,1) since 1³ + 1³ = 2 = 2³ - 6... Actually, let me recalculate: 1³ + 1³ = 2, and z³ - 1 = 2 means z³ = 3, so z ≈ 1.44. Let me check (2,2,2): 8 + 8 = 16 = 8 - 1 = 7? No. This is a difficult Diophantine equation requiring advanced techniques.', 'Number Theory': "I'll prove this by contradiction using Euclid's method. Assume there are only finitely many primes of the form 4k+3: p₁, p₂, ..., pₙ. Consider N = 4(p₁p₂...pₙ) + 3. Since N ≡ 3 (mod 4), at least one prime factor of N must be ≡ 3 (mod 4). But N is not divisible by any of p₁, p₂, ..., pₙ, so there must be another prime of the form 4k+3, contradicting our assumption. Therefore, there are infinitely many such primes.", 'Combinatorics': 'This is a stars and bars problem with constraints. We need to distribute 20 balls into 5 boxes with each box having at least 2 balls. First, place 2 balls in each box (using 10 balls). Now we need to distribute the remaining 10 balls into 5 boxes with no constraints. Using stars and bars: C(10+5-1, 5-1) = C(14,4) = 1001 ways.', 'Geometry': "This is a form of Weitzenböck's inequality. We can prove this using the relationship between area and sides. For a triangle with area S and sides a,b,c, we have S = √[s(s-a)(s-b)(s-c)] where s = (a+b+c)/2. We want to show a² + b² + c² ≥ 4√3 · S. This can be proven using the isoperimetric inequality and Jensen's inequality applied to the convex function f(x) = x²."}

            def chat_completions_create(self, **kwargs):
                result = super().chat_completions_create(**kwargs)
                messages = kwargs.get('messages', [])
                for message in messages:
                    content = message.get('content', '')
                    for prob_type, response in self.problem_responses.items():
                        if any((keyword in content for keyword in prob_type.lower().split())):
                            result.choices[0].message.content = response
                            return result
                result.choices[0].message.content = 'This is a complex problem requiring careful analysis. Let me work through it step by step with rigorous reasoning.'
                return result
        client = EnhancedMockClient()
        for problem_data in hard_problems:
            with self.subTest(problem=problem_data['name']):
                print(f'\n🧠 Testing MARS on {problem_data['name']} problem...')
                start_time = time.time()
                result = multi_agent_reasoning_system(self.system_prompt, problem_data['problem'], client, self.model)
                execution_time = time.time() - start_time
                self.assertIsInstance(result, tuple)
                response, tokens = result
                self.assertIsInstance(response, str)
                self.assertGreater(len(response), 50, 'Response should be substantial for hard problems')
                self.assertGreater(tokens, 0)
                response_lower = response.lower()
                found_features = []
                for feature in problem_data['expected_features']:
                    if feature.lower() in response_lower:
                        found_features.append(feature)
                self.assertGreater(len(found_features), 0, f'Response should contain reasoning features like {problem_data['expected_features']}')
                print(f'  ✅ {problem_data['name']}: {execution_time:.2f}s, {len(response):,} chars, features: {found_features}')
        logs = self.get_captured_logs()
        log_checks = [('🚀 MARS', 'Main orchestration logs'), ('🤖 AGENT', 'Agent generation logs'), ('🗳️  VOTING', 'Voting mechanism logs'), ('🤝 SYNTHESIS', 'Synthesis phase logs')]
        for emoji, description in log_checks:
            if emoji in logs:
                count = logs.count(emoji)
                print(f'  📊 Found {count} {description}')
            else:
                print(f'  ⚠️  No {description} found (expected with enhanced logging)')
        print(f'\n✅ MARS hard problems test completed - verified reasoning on {len(hard_problems)} challenging problems')

    def test_mars_logging_and_monitoring(self):
        """Test that MARS logging provides useful monitoring information"""
        print('\n📊 Testing MARS logging and monitoring capabilities...')

        class MonitoringMockClient(MockOpenAIClient):

            def __init__(self):
                super().__init__(response_delay=0.05, reasoning_tokens=2500)
                self.detailed_responses = True

            def chat_completions_create(self, **kwargs):
                result = super().chat_completions_create(**kwargs)
                if 'verifying' in str(kwargs.get('messages', [])):
                    result.choices[0].message.content = 'VERIFICATION: The solution appears CORRECT with high confidence. The reasoning is sound and the final answer is properly justified. Confidence: 9/10.'
                elif 'improving' in str(kwargs.get('messages', [])):
                    result.choices[0].message.content = "IMPROVEMENT: The original solution can be enhanced by adding more rigorous justification. Here's the improved version with stronger mathematical foundations..."
                else:
                    result.choices[0].message.content = "Let me solve this step by step. First, I'll analyze the problem structure. Then I'll apply appropriate mathematical techniques. The solution involves careful reasoning and verification. \\boxed{42}"
                return result
        client = MonitoringMockClient()
        complex_problem = 'Solve the system: x² + y² = 25, x + y = 7. Find all real solutions and verify your answer.'
        start_time = time.time()
        result = multi_agent_reasoning_system(self.system_prompt, complex_problem, client, self.model)
        execution_time = time.time() - start_time
        logs = self.get_captured_logs()
        log_lines = logs.split('\n')
        log_stats = {'🚀 MARS': 0, '🤖 AGENT': 0, '🔍 VERIFIER': 0, '🗳️  VOTING': 0, '🤝 SYNTHESIS': 0, '⏱️  TIMING': 0}
        for line in log_lines:
            for emoji_prefix in log_stats.keys():
                if emoji_prefix in line:
                    log_stats[emoji_prefix] += 1
        total_logs = sum(log_stats.values())
        self.assertGreater(total_logs, 10, 'Should have substantial logging for monitoring')
        monitoring_checks = [('MARS', log_stats['🚀 MARS'], 'Main orchestration phases'), ('AGENT', log_stats['🤖 AGENT'], 'Agent operations'), ('VOTING', log_stats['🗳️  VOTING'], 'Consensus mechanism'), ('SYNTHESIS', log_stats['🤝 SYNTHESIS'], 'Final synthesis')]
        print(f'\n📈 Monitoring Statistics (from {execution_time:.2f}s execution):')
        for name, count, description in monitoring_checks:
            status = '✅' if count > 0 else '⚠️ '
            print(f'  {status} {name}: {count} {description}')
        response, tokens = result
        self.assertGreater(len(response), 100, 'Complex problems should generate substantial responses')
        self.assertGreater(tokens, 1000, 'Should use significant reasoning tokens')
        quality_indicators = ['confidence', 'reasoning', 'verification', 'solution', 'answer']
        found_indicators = []
        logs_lower = logs.lower()
        for indicator in quality_indicators:
            if indicator in logs_lower:
                found_indicators.append(indicator)
        print(f'\n🎯 Quality indicators found in logs: {found_indicators}')
        self.assertGreater(len(found_indicators), 2, 'Should track multiple quality indicators')
        print(f'✅ MARS logging and monitoring test passed - captured {total_logs} log entries')

    def test_mars_consensus_mechanism(self):
        """Test MARS consensus and verification mechanism"""

        class ConsistentMockClient(MockOpenAIClient):

            def chat_completions_create(self, **kwargs):
                result = super().chat_completions_create(**kwargs)
                result.choices[0].message.content = 'The solution is x = 5. Final answer: 5'
                return result
        client = ConsistentMockClient(response_delay=0.01)
        result = multi_agent_reasoning_system(self.system_prompt, self.test_query, client, self.model)
        self.assertIsInstance(result, tuple)
        response, tokens = result
        self.assertIn('5', response)
        logs = self.get_captured_logs()
        if '🗳️  VOTING' in logs:
            print('✅ MARS consensus mechanism test passed with voting logs')
        else:
            print('✅ MARS consensus mechanism test passed')

