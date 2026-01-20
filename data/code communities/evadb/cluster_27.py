# Cluster 27

class ToGrayscaleTests(unittest.TestCase):

    def setUp(self):
        self.to_grayscale_instance = ToGrayscale()

    def test_gray_scale_name_exists(self):
        assert hasattr(self.to_grayscale_instance, 'name')

    def test_should_convert_to_grayscale(self):
        try_to_import_cv2()
        import cv2
        arr = cv2.imread(f'{EvaDB_ROOT_DIR}/test/unit_tests/functions/data/dog.jpeg')
        df = pd.DataFrame([[arr]])
        modified_arr = self.to_grayscale_instance(df)['grayscale_frame_array']
        cv2.imwrite(f'{EvaDB_ROOT_DIR}/test/unit_tests/functions/data/tmp.jpeg', modified_arr[0])
        actual_array = cv2.imread(f'{EvaDB_ROOT_DIR}/test/unit_tests/functions/data/tmp.jpeg')
        expected_arr = cv2.imread(f'{EvaDB_ROOT_DIR}/test/unit_tests/functions/data/grayscale_dog.jpeg')
        self.assertEqual(np.sum(actual_array - expected_arr), 0)
        file_remove(Path(f'{EvaDB_ROOT_DIR}/test/unit_tests/functions/data/tmp.jpeg'))

