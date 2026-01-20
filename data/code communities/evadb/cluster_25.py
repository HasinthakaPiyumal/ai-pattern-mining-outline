# Cluster 25

class FlipTests(unittest.TestCase):

    def setUp(self):
        self.horizontal_flip_instance = HorizontalFlip()
        self.vertical_flip_instance = VerticalFlip()

    def test_flip_name_exists(self):
        assert hasattr(self.horizontal_flip_instance, 'name')
        assert hasattr(self.vertical_flip_instance, 'name')

    def test_should_flip_horizontally(self):
        try_to_import_pillow()
        from PIL import Image
        img = Image.open(f'{EvaDB_ROOT_DIR}/test/data/uadetrac/small-data/MVI_20011/img00001.jpg')
        arr = asarray(img)
        df = pd.DataFrame([[arr]])
        flipped_arr = self.horizontal_flip_instance(df)['horizontally_flipped_frame_array']
        self.assertEqual(np.sum(arr[:, 0] - np.flip(flipped_arr[0][:, -1], 1)), 0)

    def test_should_flip_vertically(self):
        try_to_import_pillow()
        from PIL import Image
        img = Image.open(f'{EvaDB_ROOT_DIR}/test/data/uadetrac/small-data/MVI_20011/img00001.jpg')
        arr = asarray(img)
        df = pd.DataFrame([[arr]])
        flipped_arr = self.vertical_flip_instance(df)['vertically_flipped_frame_array']
        self.assertEqual(np.sum(arr[0, :] - np.flip(flipped_arr[0][-1, :], 1)), 0)

