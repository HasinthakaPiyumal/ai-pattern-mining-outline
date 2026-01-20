# Cluster 79

class TestNQ(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = [{'question': 'What is the capital of France?', 'answers': ['Paris']}, {'question': 'Who wrote Romeo and Juliet?', 'answers': ['William Shakespeare']}]

    def tearDown(self):
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)

    def create_test_file(self, filename, data):
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'w', newline='') as f:
            for example in data:
                f.write('{}\t{}\n'.format(example['question'], example['answers']))
        return filepath

    @patch('evoagentx.benchmark.nq.download_raw_nq_data')
    def test_load_data(self, mock_download):
        self.create_test_file('nq-train.qa.csv', self.sample_data)
        self.create_test_file('nq-dev.qa.csv', self.sample_data)
        self.create_test_file('nq-test.qa.csv', self.sample_data)
        benchmark = NQ(path=self.temp_dir)
        self.assertEqual(len(benchmark.get_train_data()), 2)
        self.assertEqual(len(benchmark.get_dev_data()), 2)
        self.assertEqual(len(benchmark.get_test_data()), 2)
        self.assertEqual(mock_download.call_count, 0)

    def test_get_label(self):
        self.create_test_file('nq-train.qa.csv', self.sample_data)
        benchmark = NQ(path=self.temp_dir, mode='train')
        example = benchmark.get_train_data()[0]
        self.assertEqual(benchmark.get_label(example), ['Paris'])
        self.assertEqual(benchmark.get_id(example), 'train-1')

    def test_evaluate(self):
        self.create_test_file('nq-train.qa.csv', self.sample_data)
        benchmark = NQ(path=self.temp_dir, mode='train')
        result = benchmark.evaluate(prediction='Paris', label=['Paris'])
        self.assertEqual(result['em'], 1.0)
        self.assertEqual(result['f1'], 1.0)
        self.assertEqual(result['acc'], 1.0)
        result = benchmark.evaluate(prediction='in Paris, France', label=['Paris'])
        self.assertEqual(result['em'], 0.0)
        self.assertTrue(abs(result['f1'] - 0.5) < 1e-05)
        self.assertEqual(result['acc'], 1.0)
        result = benchmark.evaluate(prediction='London', label=['Paris'])
        self.assertEqual(result['em'], 0.0)
        self.assertEqual(result['f1'], 0.0)
        self.assertEqual(result['acc'], 0.0)

