# Cluster 10

def main():
    """Main entry point"""
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    evaluator = SimpleQAEvaluator(model=args.model, approach=args.approach, base_url=args.base_url, grader_model=args.grader_model, timeout=args.timeout, cache_dir=args.cache_dir, output_dir=args.output_dir, use_verified=args.verified)
    try:
        metrics = evaluator.run_evaluation(num_samples=args.num_samples, start_index=args.start_index)
        print('\n' + '=' * 50)
        print('EVALUATION SUMMARY')
        print('=' * 50)
        print(f'Model: {args.model}')
        print(f'Approach: {args.approach}')
        print(f'Questions: {metrics['total_questions']}')
        print(f'Accuracy: {metrics['accuracy']:.1f}%')
        print(f'F1 Score: {metrics['f1_score']:.3f}')
        print(f'Correct: {metrics['correct']}')
        print(f'Incorrect: {metrics['incorrect']}')
        print(f'Not Attempted: {metrics['not_attempted']}')
        if metrics['errors'] > 0:
            print(f'Errors: {metrics['errors']}')
    except KeyboardInterrupt:
        print('\nEvaluation interrupted by user')
    except Exception as e:
        logger.error(f'Evaluation failed: {e}')
        raise

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate OptILLM on SimpleQA factuality benchmark')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model to evaluate (default: gpt-4o-mini)')
    parser.add_argument('--approach', type=str, default='none', choices=['none', 'web_search', 'deep_research'], help='Approach to use (default: none)')
    parser.add_argument('--base-url', type=str, default=DEFAULT_BASE_URL, help=f'OptILLM base URL (default: {DEFAULT_BASE_URL})')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help=f'Request timeout in seconds (default: {DEFAULT_TIMEOUT})')
    parser.add_argument('--grader-model', type=str, default=DEFAULT_GRADER_MODEL, help=f'Model for grading responses (default: {DEFAULT_GRADER_MODEL})')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of questions to evaluate (default: all)')
    parser.add_argument('--start-index', type=int, default=0, help='Start from specific question index (default: 0)')
    parser.add_argument('--num-search-results', type=int, default=10, help='Number of search results per query (default: 10)')
    parser.add_argument('--headless', action='store_true', help='Run browser in headless mode for web search')
    parser.add_argument('--cache-dir', type=str, default='cache', help='Directory for caching dataset (default: cache)')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory for saving results (default: results)')
    parser.add_argument('--verified', action='store_true', help='Use SimpleQA-Verified dataset (1k verified questions) instead of original SimpleQA')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()

def main():
    global server_config
    global cepo_config
    global request_batcher
    global conversation_logger
    load_plugins()
    args = parse_args()
    server_config.update(vars(args))
    port = server_config['port']
    if server_config.get('batch_mode', False):
        logger.info(f'Batch mode enabled: size={server_config['batch_size']}, wait={server_config['batch_wait_ms']}ms')
        request_batcher = RequestBatcher(max_batch_size=server_config['batch_size'], max_wait_ms=server_config['batch_wait_ms'], enable_logging=True)

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
        request_batcher.set_processor(process_batch_requests)
    logging_level = server_config['log']
    if logging_level in logging_levels.keys():
        logger.setLevel(logging_levels[logging_level])
    global conversation_logger
    conversation_logger = ConversationLogger(log_dir=Path(server_config['conversation_log_dir']), enabled=server_config['log_conversations'])
    optillm.conversation_logger.set_global_logger(conversation_logger)
    if server_config['log_conversations']:
        logger.info(f'Conversation logging enabled. Logs will be saved to: {server_config['conversation_log_dir']}')
    cepo_config = init_cepo_config(server_config)
    if args.approach == 'cepo':
        logger.info(f'CePO Config: {cepo_config}')
    logger.info(f'Starting server with approach: {server_config['approach']}')
    server_config_clean = server_config.copy()
    if server_config_clean['optillm_api_key']:
        server_config_clean['optillm_api_key'] = '[REDACTED]'
    logger.info(f'Server configuration: {server_config_clean}')
    if server_config.get('launch_gui'):
        try:
            import gradio as gr
            import threading
            server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': port})
            server_thread.daemon = True
            server_thread.start()
            base_url = f'http://localhost:{port}/v1'
            logger.info(f'Launching Gradio interface connected to {base_url}')

            def chat_with_optillm(message, history):
                import httpx
                from openai import OpenAI
                custom_client = OpenAI(api_key='optillm', base_url=base_url, timeout=httpx.Timeout(1800.0, connect=5.0), max_retries=0)
                messages = []
                for h in history:
                    if h[0]:
                        messages.append({'role': 'user', 'content': h[0]})
                    if h[1]:
                        messages.append({'role': 'assistant', 'content': h[1]})
                messages.append({'role': 'user', 'content': message})
                try:
                    response = custom_client.chat.completions.create(model=server_config['model'], messages=messages)
                    return response.choices[0].message.content
                except Exception as e:
                    return f'Error: {str(e)}'
            demo = gr.ChatInterface(chat_with_optillm, title='OptILLM Chat Interface', description=f'Connected to OptILLM proxy at {base_url}')
            demo.queue()
            demo.launch(server_name='0.0.0.0', share=False)
        except ImportError:
            logger.error('Gradio is required for GUI. Install it with: pip install gradio')
            return
    app.run(host='0.0.0.0', port=port)

def load_plugins():
    plugin_approaches.clear()
    import optillm
    package_plugin_dir = os.path.join(os.path.dirname(optillm.__file__), 'plugins')
    current_dir = os.getcwd() if server_config.get('plugins_dir', '') == '' else server_config['plugins_dir']
    local_plugin_dir = os.path.join(current_dir, 'optillm', 'plugins')
    plugin_dirs = []
    plugin_dirs.append((package_plugin_dir, 'package'))
    if local_plugin_dir != package_plugin_dir:
        plugin_dirs.append((local_plugin_dir, 'local'))
    for plugin_dir, source in plugin_dirs:
        logger.info(f'Looking for {source} plugins in: {plugin_dir}')
        if not os.path.exists(plugin_dir):
            logger.debug(f'{source.capitalize()} plugin directory not found: {plugin_dir}')
            continue
        plugin_files = glob.glob(os.path.join(plugin_dir, '*.py'))
        if not plugin_files:
            logger.debug(f'No plugin files found in {source} directory: {plugin_dir}')
            continue
        logger.info(f'Found {source} plugin files: {plugin_files}')
        for plugin_file in plugin_files:
            try:
                module_name = os.path.basename(plugin_file)[:-3]
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, 'SLUG') and hasattr(module, 'run'):
                    if module.SLUG in plugin_approaches:
                        logger.info(f'Overriding {source} plugin: {module.SLUG}')
                    plugin_approaches[module.SLUG] = module.run
                    logger.info(f'Loaded {source} plugin: {module.SLUG}')
                else:
                    logger.warning(f'Plugin {module_name} from {source} missing required attributes (SLUG and run)')
            except Exception as e:
                logger.error(f'Error loading {source} plugin {plugin_file}: {str(e)}')
    if not plugin_approaches:
        logger.warning('No plugins loaded from any location')

def init_cepo_config(cmd_line_args: dict) -> CepoConfig:
    cepo_args = {key.split('cepo_')[1]: value for key, value in cmd_line_args.items() if 'cepo' in key and 'cepo_config_file' != key and (value is not None)}
    cepo_config_yaml = {}
    if cmd_line_args.get('cepo_config_file', None):
        with open(cmd_line_args['cepo_config_file'], 'r') as yaml_file:
            cepo_config_yaml = yaml.safe_load(yaml_file)
    cepo_args = {**cepo_config_yaml, **cepo_args}
    return CepoConfig(**cepo_args)

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

class TestConversationLogger(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.logger_enabled = ConversationLogger(self.temp_dir, enabled=True)
        self.logger_disabled = ConversationLogger(self.temp_dir, enabled=False)

    def tearDown(self):
        """Clean up test fixtures"""
        for file in self.temp_dir.glob('*'):
            file.unlink()
        self.temp_dir.rmdir()

    def test_logger_initialization_and_disabled_state(self):
        """Test logger initialization and disabled logger behavior"""
        self.assertTrue(self.logger_enabled.enabled)
        self.assertEqual(self.logger_enabled.log_dir, self.temp_dir)
        self.assertTrue(self.temp_dir.exists())
        self.assertFalse(self.logger_disabled.enabled)
        request_id = self.logger_disabled.start_conversation({}, 'test', 'model')
        self.assertEqual(request_id, '')
        self.logger_disabled.log_provider_call('req1', {}, {})
        self.logger_disabled.log_final_response('req1', {})
        self.logger_disabled.log_error('req1', 'error')
        self.logger_disabled.finalize_conversation('req1')

    def test_conversation_lifecycle(self):
        """Test complete conversation lifecycle: start, log calls, errors, finalize"""
        client_request = {'messages': [{'role': 'user', 'content': 'Hello'}], 'model': 'gpt-4o-mini', 'temperature': 0.7}
        request_id = self.logger_enabled.start_conversation(client_request=client_request, approach='moa', model='gpt-4o-mini')
        self.assertIsInstance(request_id, str)
        self.assertTrue(request_id.startswith('req_'))
        self.assertEqual(len(request_id), 12)
        self.assertIn(request_id, self.logger_enabled.active_entries)
        entry = self.logger_enabled.active_entries[request_id]
        self.assertEqual(entry.request_id, request_id)
        self.assertEqual(entry.approach, 'moa')
        self.assertEqual(entry.model, 'gpt-4o-mini')
        provider_request = {'model': 'test', 'messages': []}
        provider_response = {'choices': [{'message': {'content': 'response'}}]}
        self.logger_enabled.log_provider_call(request_id, provider_request, provider_response)
        self.logger_enabled.log_provider_call(request_id, provider_request, provider_response)
        entry = self.logger_enabled.active_entries[request_id]
        self.assertEqual(len(entry.provider_calls), 2)
        self.assertEqual(entry.provider_calls[0]['call_number'], 1)
        self.assertEqual(entry.provider_calls[1]['call_number'], 2)
        final_response = {'choices': [{'message': {'content': 'final'}}]}
        self.logger_enabled.log_final_response(request_id, final_response)
        error_msg = 'Test error message'
        self.logger_enabled.log_error(request_id, error_msg)
        entry = self.logger_enabled.active_entries[request_id]
        self.assertEqual(entry.error, error_msg)
        self.logger_enabled.finalize_conversation(request_id)
        self.assertNotIn(request_id, self.logger_enabled.active_entries)
        log_files = list(self.temp_dir.glob('conversations_*.jsonl'))
        self.assertEqual(len(log_files), 1)
        with open(log_files[0], 'r', encoding='utf-8') as f:
            log_line = f.read().strip()
        log_entry = json.loads(log_line)
        self.assertEqual(log_entry['request_id'], request_id)
        self.assertEqual(log_entry['approach'], 'moa')
        self.assertEqual(log_entry['model'], 'gpt-4o-mini')
        self.assertEqual(log_entry['client_request'], client_request)
        self.assertEqual(len(log_entry['provider_calls']), 2)
        self.assertEqual(log_entry['final_response']['choices'][0]['message']['content'], 'final')
        self.assertIsInstance(log_entry['total_duration_ms'], int)
        self.assertEqual(log_entry['error'], error_msg)

    def test_multiple_conversations_and_log_files(self):
        """Test handling multiple concurrent conversations and log file naming"""
        with patch('optillm.conversation_logger.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 1, 27, 12, 0, 0, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            log_path = self.logger_enabled._get_log_file_path()
            expected_filename = 'conversations_2025-01-27.jsonl'
            self.assertEqual(log_path.name, expected_filename)
            self.assertEqual(log_path.parent, self.temp_dir)
        request_id1 = self.logger_enabled.start_conversation({}, 'moa', 'model1')
        request_id2 = self.logger_enabled.start_conversation({}, 'none', 'model2')
        self.assertNotEqual(request_id1, request_id2)
        self.assertIn(request_id1, self.logger_enabled.active_entries)
        self.assertIn(request_id2, self.logger_enabled.active_entries)
        self.logger_enabled.log_provider_call(request_id1, {'req': '1'}, {'resp': '1'})
        self.logger_enabled.log_provider_call(request_id2, {'req': '2'}, {'resp': '2'})
        self.logger_enabled.finalize_conversation(request_id1)
        self.logger_enabled.finalize_conversation(request_id2)
        log_files = list(self.temp_dir.glob('conversations_*.jsonl'))
        self.assertEqual(len(log_files), 1)
        with open(log_files[0], 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        self.assertEqual(len(lines), 2)
        entry1 = json.loads(lines[0])
        entry2 = json.loads(lines[1])
        self.assertEqual(entry1['approach'], 'moa')
        self.assertEqual(entry2['approach'], 'none')

    def test_invalid_request_id_and_stats(self):
        """Test handling of invalid request IDs and logger statistics"""
        self.logger_enabled.log_provider_call('invalid_id', {}, {})
        self.logger_enabled.log_final_response('invalid_id', {})
        self.logger_enabled.log_error('invalid_id', 'error')
        self.logger_enabled.finalize_conversation('invalid_id')
        stats = self.logger_disabled.get_stats()
        expected_disabled_stats = {'enabled': False, 'log_dir': str(self.temp_dir), 'active_conversations': 0}
        self.assertEqual(stats, expected_disabled_stats)
        request_id1 = self.logger_enabled.start_conversation({}, 'test', 'model')
        request_id2 = self.logger_enabled.start_conversation({}, 'test', 'model')
        stats = self.logger_enabled.get_stats()
        self.assertTrue(stats['enabled'])
        self.assertEqual(stats['log_dir'], str(self.temp_dir))
        self.assertEqual(stats['active_conversations'], 2)
        self.assertEqual(stats['log_files_count'], 0)
        self.assertEqual(stats['total_entries_approximate'], 0)
        self.logger_enabled.finalize_conversation(request_id1)
        stats = self.logger_enabled.get_stats()
        self.assertEqual(stats['active_conversations'], 1)
        self.assertEqual(stats['log_files_count'], 1)
        self.assertEqual(stats['total_entries_approximate'], 1)

class TestRequestBatcher(unittest.TestCase):
    """Test the core RequestBatcher functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.batcher = RequestBatcher(max_batch_size=4, max_wait_ms=100)
        self.test_responses = []

        def mock_processor(requests):
            """Mock batch processor that returns simple responses"""
            responses = []
            for i, req in enumerate(requests):
                responses.append({'id': f'test-{i}', 'object': 'chat.completion', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': f'Response to request {i}'}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 10, 'total_tokens': 20}})
            return responses
        self.batcher.set_processor(mock_processor)

    def tearDown(self):
        """Clean up after tests"""
        self.batcher.shutdown()

    def test_single_request(self):
        """Test that single requests work correctly"""
        request_data = {'model': 'test-model', 'prompt': 'Hello'}
        response = self.batcher.add_request(request_data)
        self.assertIsInstance(response, dict)
        self.assertEqual(response['object'], 'chat.completion')
        self.assertEqual(response['choices'][0]['message']['content'], 'Response to request 0')

    def test_batch_formation(self):
        """Test that multiple requests form a batch"""

        def send_request(request_id):
            request_data = {'model': 'test-model', 'prompt': f'Request {request_id}'}
            return self.batcher.add_request(request_data)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(send_request, i) for i in range(3)]
            responses = [future.result() for future in futures]
        self.assertEqual(len(responses), 3)
        for i, response in enumerate(responses):
            self.assertIsInstance(response, dict)
            self.assertEqual(response['object'], 'chat.completion')

    def test_batch_timeout(self):
        """Test that partial batches process after timeout"""
        start_time = time.time()
        request_data = {'model': 'test-model', 'prompt': 'Single request'}
        response = self.batcher.add_request(request_data)
        elapsed_time = time.time() - start_time
        self.assertGreater(elapsed_time, 0.09)
        self.assertIsInstance(response, dict)

    def test_incompatible_requests(self):
        """Test that incompatible requests are properly handled"""
        request_data = {'model': 'test-model', 'stream': True}
        with self.assertRaises(BatchingError):
            self.batcher.add_request(request_data)

    def test_processor_error_handling(self):
        """Test that processor errors are handled correctly"""

        def failing_processor(requests):
            raise Exception('Processor failed')
        batcher = RequestBatcher(max_batch_size=2, max_wait_ms=50)
        batcher.set_processor(failing_processor)
        try:
            request_data = {'model': 'test-model', 'prompt': 'Test'}
            with self.assertRaises(BatchingError):
                batcher.add_request(request_data)
        finally:
            batcher.shutdown()

    def test_batch_stats(self):
        """Test that batch statistics are collected correctly"""
        for i in range(5):
            request_data = {'model': 'test-model', 'prompt': f'Request {i}'}
            self.batcher.add_request(request_data)
        stats = self.batcher.get_stats()
        self.assertGreater(stats['total_requests'], 0)
        self.assertGreater(stats['total_batches'], 0)
        self.assertGreater(stats['avg_batch_size'], 0)

class TestPerformanceBenches(unittest.TestCase):
    """Performance comparison tests"""

    def setUp(self):
        """Set up performance test fixtures"""
        self.test_prompts = [('System prompt 1', 'What is AI?'), ('System prompt 2', 'Explain machine learning'), ('System prompt 3', 'Define neural networks'), ('System prompt 4', 'Describe deep learning')]

    def measure_sequential_processing(self, prompts):
        """Measure time for sequential processing"""
        start_time = time.time()
        responses = []
        for sys_prompt, user_prompt in prompts:
            time.sleep(0.1)
            responses.append(f'Response to: {user_prompt}')
        end_time = time.time()
        return (responses, end_time - start_time)

    def measure_batch_processing(self, prompts):
        """Measure time for batch processing"""
        batcher = RequestBatcher(max_batch_size=len(prompts), max_wait_ms=10)

        def mock_batch_processor(requests):
            time.sleep(0.15)
            return [{'response': f'Batched response {i}'} for i in range(len(requests))]
        batcher.set_processor(mock_batch_processor)
        try:
            start_time = time.time()

            def send_request(prompt_data):
                sys_prompt, user_prompt = prompt_data
                return batcher.add_request({'model': 'test-model', 'system_prompt': sys_prompt, 'user_prompt': user_prompt})
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
                futures = [executor.submit(send_request, prompt) for prompt in prompts]
                responses = [future.result() for future in futures]
            end_time = time.time()
            return (responses, end_time - start_time)
        finally:
            batcher.shutdown()

    def test_batching_performance_improvement(self):
        """Test that batching provides performance improvement"""
        seq_responses, seq_time = self.measure_sequential_processing(self.test_prompts)
        batch_responses, batch_time = self.measure_batch_processing(self.test_prompts)
        improvement_ratio = seq_time / batch_time
        self.assertGreater(improvement_ratio, 1.5, f'Batching should be >1.5x faster, got {improvement_ratio:.2f}x')
        self.assertEqual(len(seq_responses), len(batch_responses))

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def test_batch_mode_errors(self):
        """Test error conditions in batch mode"""
        batcher = RequestBatcher(max_batch_size=2, max_wait_ms=50)
        with self.assertRaises(BatchingError):
            batcher.add_request({'model': 'test'})
        batcher.shutdown()

    def test_mixed_model_requests(self):
        """Test that requests with different models are properly separated"""
        batcher = RequestBatcher(max_batch_size=4, max_wait_ms=50)

        def mock_processor(requests):
            models = set((req.get('model') for req in requests))
            self.assertEqual(len(models), 1, 'Batch should have requests from single model')
            return [{'response': 'ok'}] * len(requests)
        batcher.set_processor(mock_processor)
        try:
            req1 = {'model': 'model-a', 'prompt': 'test1'}
            req2 = {'model': 'model-b', 'prompt': 'test2'}
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future1 = executor.submit(batcher.add_request, req1)
                future2 = executor.submit(batcher.add_request, req2)
                response1 = future1.result()
                response2 = future2.result()
                self.assertIsInstance(response1, dict)
                self.assertIsInstance(response2, dict)
        finally:
            batcher.shutdown()

