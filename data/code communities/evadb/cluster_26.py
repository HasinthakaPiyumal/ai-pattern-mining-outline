# Cluster 26

class GaussianBlurTests(unittest.TestCase):

    def setUp(self):
        self.gb_instance = GaussianBlur()
        self.tmp_file = f'{EvaDB_ROOT_DIR}/test/unit_tests/functions/data/tmp.jpeg'

    def test_gb_name_exists(self):
        assert hasattr(self.gb_instance, 'name')

    def test_should_blur_image(self):
        try_to_import_cv2()
        import cv2
        arr = cv2.imread(f'{EvaDB_ROOT_DIR}/test/unit_tests/functions/data/dog.jpeg')
        df = pd.DataFrame([[arr]])
        modified_arr = self.gb_instance(df)['blurred_frame_array']
        cv2.imwrite(self.tmp_file, cv2.cvtColor(modified_arr[0], cv2.COLOR_RGB2BGR))
        actual_array = cv2.imread(self.tmp_file)
        expected_array = cv2.imread(f'{EvaDB_ROOT_DIR}/test/unit_tests/functions/data/blurred_dog.jpeg')
        self.assertEqual(np.sum(actual_array - expected_array), 0)
        file_remove(Path(self.tmp_file))

