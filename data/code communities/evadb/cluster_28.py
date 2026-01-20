# Cluster 28

class CropTests(unittest.TestCase):

    def setUp(self):
        self.array_count = ArrayCount()

    def test_array_count_name_exists(self):
        assert hasattr(self.array_count, 'name')

