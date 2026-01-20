# Cluster 45

class TestAFlowHotPotQA(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = [{'_id': 'test_id_1', 'question': 'What is the capital of France?', 'answer': 'Paris', 'context': [['France', ['Paris is the capital of France.', 'It is a beautiful city.']]], 'supporting_facts': [['France', 0]], 'type': 'comparison', 'level': 'medium'}]

    def tearDown(self):
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)

    @patch('evoagentx.benchmark.hotpotqa.download_aflow_benchmark_data')
    def test_aflow_load_data(self, mock_download):
        filepath = os.path.join(self.temp_dir, 'hotpotqa_test.jsonl')
        with open(filepath, 'w') as f:
            for item in self.sample_data:
                f.write(json.dumps(item) + '\n')
        benchmark = AFlowHotPotQA(path=self.temp_dir, mode='test')
        self.assertEqual(mock_download.call_count, 0)

