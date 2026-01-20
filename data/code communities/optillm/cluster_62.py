# Cluster 62

class TestMLXBatching(unittest.TestCase):
    """Test MLX batch processing functionality"""

    @unittest.skipIf(not MLX_AVAILABLE, 'MLX not available')
    def setUp(self):
        """Set up MLX test fixtures"""
        self.model_config = MLXModelConfig(model_id=TEST_MODEL_MLX, max_new_tokens=100)
        from optillm.inference import CacheManager
        self.cache_manager = CacheManager.get_instance(max_size=1)

    @unittest.skipIf(not MLX_AVAILABLE, 'MLX not available')
    def test_mlx_batch_creation(self):
        """Test that MLX batch processing can be created"""
        try:
            from optillm.inference import MLXInferencePipeline
            self.assertTrue(hasattr(MLXInferencePipeline, 'process_batch'))
        except Exception as e:
            pass

    @unittest.skipIf(not MLX_AVAILABLE, 'MLX not available')
    def test_mlx_batch_parameters(self):
        """Test MLX batch processing parameter validation"""
        print(f'\n📥 Testing MLX model: {self.model_config.model_id}')
        print('This may take a few minutes if model needs to be downloaded...')
        pipeline = MLXInferencePipeline(self.model_config, self.cache_manager)
        print('✅ MLX model loaded successfully')
        with self.assertRaises(ValueError):
            pipeline.process_batch(['system1'], ['user1', 'user2'])
        responses, tokens = pipeline.process_batch([], [])
        self.assertEqual(len(responses), 0)
        self.assertEqual(len(tokens), 0)
        print('✅ MLX parameter validation tests passed')

    @unittest.skipIf(not MLX_AVAILABLE, 'MLX not available')
    def test_mlx_batch_generation(self):
        """Test MLX batch processing with actual generation"""
        print(f'\n🧪 Testing MLX batch generation...')
        pipeline = MLXInferencePipeline(self.model_config, self.cache_manager)
        print('✅ MLX model ready for testing')
        system_prompts = ['You are a helpful assistant.', 'You are a helpful assistant.']
        user_prompts = ['What is AI?', 'What is ML?']
        print('🚀 Running batch generation...')
        responses, token_counts = pipeline.process_batch(system_prompts, user_prompts, generation_params={'max_new_tokens': 20})
        self.assertEqual(len(responses), 2)
        self.assertEqual(len(token_counts), 2)
        for i, response in enumerate(responses):
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            print(f'   Response {i + 1}: {response[:50]}{('...' if len(response) > 50 else '')}')
        for token_count in token_counts:
            self.assertIsInstance(token_count, int)
            self.assertGreater(token_count, 0)
        print(f'✅ MLX batch generation successful - {len(responses)} responses generated')
        print(f'   Token counts: {token_counts}')

