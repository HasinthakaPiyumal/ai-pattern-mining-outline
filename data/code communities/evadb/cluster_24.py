# Cluster 24

def try_to_import_cv2():
    try:
        import cv2
    except ImportError:
        raise ValueError('Could not import cv2 python package.\n                Please install it with `pip install opencv-python`.')

def create_random_image(i, path):
    img = np.random.random_sample([400, 400, 3]).astype(np.uint8)
    try_to_import_cv2()
    import cv2
    cv2.imwrite(os.path.join(path, f'img{i}.jpg'), img)

@pytest.mark.notparallel
class PytorchTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        os.environ['ray'] = str(cls.evadb.catalog().get_configuration_catalog_value('ray'))
        ua_detrac = f'{EvaDB_ROOT_DIR}/data/ua_detrac/ua_detrac.mp4'
        mnist = f'{EvaDB_ROOT_DIR}/data/mnist/mnist.mp4'
        actions = f'{EvaDB_ROOT_DIR}/data/actions/actions.mp4'
        asl_actions = f'{EvaDB_ROOT_DIR}/data/actions/computer_asl.mp4'
        meme1 = f'{EvaDB_ROOT_DIR}/data/detoxify/meme1.jpg'
        meme2 = f'{EvaDB_ROOT_DIR}/data/detoxify/meme2.jpg'
        execute_query_fetch_all(cls.evadb, f"LOAD VIDEO '{ua_detrac}' INTO MyVideo;")
        execute_query_fetch_all(cls.evadb, f"LOAD VIDEO '{mnist}' INTO MNIST;")
        execute_query_fetch_all(cls.evadb, f"LOAD VIDEO '{actions}' INTO Actions;")
        execute_query_fetch_all(cls.evadb, f"LOAD VIDEO '{asl_actions}' INTO Asl_actions;")
        execute_query_fetch_all(cls.evadb, f"LOAD IMAGE '{meme1}' INTO MemeImages;")
        execute_query_fetch_all(cls.evadb, f"LOAD IMAGE '{meme2}' INTO MemeImages;")
        load_functions_for_testing(cls.evadb)

    @classmethod
    def tearDownClass(cls):
        file_remove('ua_detrac.mp4')
        file_remove('mnist.mp4')
        file_remove('actions.mp4')
        file_remove('computer_asl.mp4')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS Actions;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS MNIST;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS MyVideo;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS Asl_actions;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS MemeImages;')

    def assertBatchEqual(self, a: Batch, b: Batch, msg: str):
        try:
            pd_testing.assert_frame_equal(a.frames, b.frames)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        self.addTypeEqualityFunc(Batch, self.assertBatchEqual)

    def tearDown(self) -> None:
        shutdown_ray()

    @ray_skip_marker
    def test_should_apply_parallel_match_sequential(self):
        select_query = 'SELECT id, obj.labels\n                          FROM MyVideo JOIN LATERAL\n                          FastRCNNObjectDetector(data)\n                          AS obj(labels, bboxes, scores)\n                         WHERE id < 20;'
        par_batch = execute_query_fetch_all(self.evadb, select_query)
        self.evadb.config.update_value('experimental', 'ray', False)
        select_query = 'SELECT id, obj.labels\n                          FROM MyVideo JOIN LATERAL\n                          FastRCNNObjectDetector(data)\n                          AS obj(labels, bboxes, scores)\n                         WHERE id < 20;'
        seq_batch = execute_query_fetch_all(self.evadb, select_query)
        self.evadb.config.update_value('experimental', 'ray', True)
        self.assertEqual(len(par_batch), len(seq_batch))
        self.assertEqual(par_batch, seq_batch)

    @ray_skip_marker
    def test_should_project_parallel_match_sequential(self):
        create_function_query = "CREATE FUNCTION IF NOT EXISTS FaceDetector\n                  INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))\n                  OUTPUT (bboxes NDARRAY FLOAT32(ANYDIM, 4),\n                          scores NDARRAY FLOAT32(ANYDIM))\n                  TYPE  FaceDetection\n                  IMPL  'evadb/functions/face_detector.py';\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        select_query = 'SELECT FaceDetector(data) FROM MyVideo WHERE id < 5;'
        par_batch = execute_query_fetch_all(self.evadb, select_query)
        self.evadb.config.update_value('experimental', 'ray', False)
        seq_batch = execute_query_fetch_all(self.evadb, select_query)
        self.evadb.config.update_value('experimental', 'ray', True)
        self.assertEqual(len(par_batch), len(seq_batch))
        self.assertEqual(par_batch, seq_batch)

    def test_should_raise_exception_with_parallel(self):
        video_path = create_sample_video(100)
        load_query = f"LOAD VIDEO '{video_path}' INTO parallelErrorVideo;"
        execute_query_fetch_all(self.evadb, load_query)
        file_remove('dummy.avi')
        select_query = 'SELECT id, obj.labels\n                          FROM parallelErrorVideo JOIN LATERAL\n                          FastRCNNObjectDetector(data)\n                          AS obj(labels, bboxes, scores)\n                         WHERE id < 2;'
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, select_query, do_not_print_exceptions=True)

    @pytest.mark.torchtest
    def test_should_run_pytorch_and_fastrcnn_with_lateral_join(self):
        select_query = 'SELECT id, obj.labels\n                          FROM MyVideo JOIN LATERAL\n                          FastRCNNObjectDetector(data)\n                          AS obj(labels, bboxes, scores)\n                         WHERE id < 2;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(actual_batch), 2)

    @pytest.mark.torchtest
    def test_should_run_pytorch_and_yolo_and_mvit(self):
        execute_query_fetch_all(self.evadb, Mvit_function_query)
        select_query = "SELECT FIRST(id),\n                            Yolo(FIRST(data)),\n                            MVITActionRecognition(SEGMENT(data))\n                            FROM Actions\n                            WHERE id < 32\n                            GROUP BY '16 frames'; "
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(actual_batch), 2)
        res = actual_batch.frames
        for idx in res.index:
            self.assertTrue('person' in res['yolo.labels'][idx] and 'yoga' in res['mvitactionrecognition.labels'][idx])

    @pytest.mark.torchtest
    def test_should_run_pytorch_and_asl(self):
        execute_query_fetch_all(self.evadb, Asl_function_query)
        select_query = "SELECT FIRST(id), ASLActionRecognition(SEGMENT(data))\n                        FROM Asl_actions\n                        SAMPLE 5\n                        GROUP BY '16 frames';"
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        res = actual_batch.frames
        self.assertEqual(len(res), 1)
        for idx in res.index:
            self.assertTrue('computer' in res['aslactionrecognition.labels'][idx])

    @pytest.mark.torchtest
    def test_should_run_pytorch_and_facenet(self):
        create_function_query = "CREATE FUNCTION IF NOT EXISTS FaceDetector\n                  INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))\n                  OUTPUT (bboxes NDARRAY FLOAT32(ANYDIM, 4),\n                          scores NDARRAY FLOAT32(ANYDIM))\n                  TYPE  FaceDetection\n                  IMPL  'evadb/functions/face_detector.py';\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        select_query = 'SELECT FaceDetector(data) FROM MyVideo\n                        WHERE id < 5 order by scores;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(actual_batch), 5)

    @pytest.mark.torchtest
    @windows_skip_marker
    @ocr_skip_marker
    def test_should_run_pytorch_and_ocr(self):
        create_function_query = "CREATE FUNCTION IF NOT EXISTS OCRExtractor\n                  INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))\n                  OUTPUT (labels NDARRAY STR(10),\n                          bboxes NDARRAY FLOAT32(ANYDIM, 4),\n                          scores NDARRAY FLOAT32(ANYDIM))\n                  TYPE  OCRExtraction\n                  IMPL  'evadb/functions/ocr_extractor.py';\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        select_query = 'SELECT OCRExtractor(data) FROM MNIST\n                        WHERE id >= 150 AND id < 155;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(actual_batch), 5)
        res = actual_batch.frames
        self.assertTrue(res['ocrextractor.labels'][0][0] == '4')
        self.assertTrue(res['ocrextractor.scores'][2][0] > 0.9)

    @pytest.mark.torchtest
    def test_should_run_pytorch_and_resnet50(self):
        create_function_query = "CREATE FUNCTION IF NOT EXISTS FeatureExtractor\n                  INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))\n                  OUTPUT (features NDARRAY FLOAT32(ANYDIM))\n                  TYPE  Classification\n                  IMPL  'evadb/functions/feature_extractor.py';\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        select_query = 'SELECT FeatureExtractor(data) FROM MyVideo\n                        WHERE id < 5;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(actual_batch), 5)
        res = actual_batch.frames
        self.assertEqual(res['featureextractor.features'][0].shape, (1, 2048))

    @pytest.mark.torchtest
    def test_should_run_pytorch_and_similarity(self):
        create_open_function_query = 'CREATE FUNCTION IF NOT EXISTS Open\n                INPUT (img_path TEXT(1000))\n                OUTPUT (data NDARRAY UINT8(3, ANYDIM, ANYDIM))\n                TYPE NdarrayFUNCTION\n                IMPL "evadb/functions/ndarray/open.py";\n        '
        execute_query_fetch_all(self.evadb, create_open_function_query)
        create_similarity_function_query = 'CREATE FUNCTION IF NOT EXISTS Similarity\n                    INPUT (Frame_Array_Open NDARRAY UINT8(3, ANYDIM, ANYDIM),\n                           Frame_Array_Base NDARRAY UINT8(3, ANYDIM, ANYDIM),\n                           Feature_Extractor_Name TEXT(100))\n                    OUTPUT (distance FLOAT(32, 7))\n                    TYPE NdarrayFUNCTION\n                    IMPL "evadb/functions/ndarray/similarity.py";\n        '
        execute_query_fetch_all(self.evadb, create_similarity_function_query)
        create_feat_function_query = 'CREATE FUNCTION IF NOT EXISTS FeatureExtractor\n                  INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))\n                  OUTPUT (features NDARRAY FLOAT32(ANYDIM))\n                  TYPE  Classification\n                  IMPL  "evadb/functions/feature_extractor.py";\n        '
        execute_query_fetch_all(self.evadb, create_feat_function_query)
        select_query = 'SELECT data FROM MyVideo WHERE id = 1;'
        batch_res = execute_query_fetch_all(self.evadb, select_query)
        img = batch_res.frames['myvideo.data'][0]
        tmp_dir_from_config = self.evadb.catalog().get_configuration_catalog_value('tmp_dir')
        img_save_path = os.path.join(tmp_dir_from_config, 'dummy.jpg')
        try:
            os.remove(img_save_path)
        except FileNotFoundError:
            pass
        try_to_import_cv2()
        import cv2
        cv2.imwrite(img_save_path, img)
        similarity_query = 'SELECT data FROM MyVideo WHERE id < 5\n                    ORDER BY Similarity(FeatureExtractor(Open("{}")),\n                                        FeatureExtractor(data))\n                    LIMIT 1;'.format(img_save_path)
        actual_batch = execute_query_fetch_all(self.evadb, similarity_query)
        similar_data = actual_batch.frames['myvideo.data'][0]
        self.assertTrue(np.array_equal(img, similar_data))

    @pytest.mark.torchtest
    @windows_skip_marker
    @ocr_skip_marker
    def test_should_run_ocr_on_cropped_data(self):
        create_function_query = "CREATE FUNCTION IF NOT EXISTS OCRExtractor\n                  INPUT  (text NDARRAY STR(100))\n                  OUTPUT (labels NDARRAY STR(10),\n                          bboxes NDARRAY FLOAT32(ANYDIM, 4),\n                          scores NDARRAY FLOAT32(ANYDIM))\n                  TYPE  OCRExtraction\n                  IMPL  'evadb/functions/ocr_extractor.py';\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        select_query = 'SELECT OCRExtractor(Crop(data, [2, 2, 24, 24])) FROM MNIST\n                        WHERE id >= 150 AND id < 155;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(actual_batch), 5)
        res = actual_batch.frames
        self.assertTrue(res['ocrextractor.labels'][0][0] == '4')
        self.assertTrue(res['ocrextractor.scores'][2][0] > 0.9)

    @pytest.mark.torchtest
    @gpu_skip_marker
    def test_should_run_extract_object(self):
        select_query = '\n            SELECT id, T.iids, T.bboxes, T.scores, T.labels\n            FROM MyVideo JOIN LATERAL EXTRACT_OBJECT(data, Yolo, NorFairTracker)\n                AS T(iids, labels, bboxes, scores)\n            WHERE id < 30;\n            '
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(actual_batch), 30)
        num_of_entries = actual_batch.frames['T.iids'].apply(lambda x: len(x)).sum()
        select_query = '\n            SELECT id, T.iid, T.bbox, T.score, T.label\n            FROM MyVideo JOIN LATERAL\n                UNNEST(EXTRACT_OBJECT(data, Yolo, NorFairTracker)) AS T(iid, label, bbox, score)\n            WHERE id < 30;\n            '
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(actual_batch), num_of_entries)

    def test_check_unnest_with_predicate_on_yolo(self):
        query = "SELECT id, Yolo.label, Yolo.bbox, Yolo.score\n                  FROM MyVideo\n                  JOIN LATERAL UNNEST(Yolo(data)) AS Yolo(label, bbox, score)\n                  WHERE Yolo.label = 'car' AND id < 2;"
        actual_batch = execute_query_fetch_all(self.evadb, query)
        self.assertTrue(len(actual_batch) > 2)

class EmotionDetector(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_path = Path(EvaDB_TEST_DATA_DIR) / 'data' / 'emotion_detector'

    def _load_image(self, path):
        try_to_import_cv2()
        import cv2
        assert path.exists(), f'File does not exist at the path {str(path)}'
        img = cv2.imread(str(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @unittest.skip('disable test due to model downloading time')
    def test_should_return_correct_emotion(self):
        from evadb.functions.emotion_detector import EmotionDetector
        happy_img = self.base_path / 'happy.jpg'
        sad_img = self.base_path / 'sad.jpg'
        angry_img = self.base_path / 'angry.jpg'
        frame_happy = {'id': 1, 'data': self._load_image(happy_img)}
        frame_sad = {'id': 2, 'data': self._load_image(sad_img)}
        frame_angry = {'id': 3, 'data': self._load_image(angry_img)}
        frame_batch = Batch(pd.DataFrame([frame_happy, frame_sad, frame_angry]))
        detector = EmotionDetector()
        result = detector.classify(frame_batch.project(['data']).frames)
        self.assertEqual('happy', result.iloc[0]['labels'])
        self.assertEqual('sad', result.iloc[1]['labels'])
        self.assertEqual('angry', result.iloc[2]['labels'])

class FastRCNNObjectDetectorTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def _load_image(self, path):
        try_to_import_cv2()
        import cv2
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @unittest.skip('disable test due to model downloading time')
    def test_should_return_batches_equivalent_to_number_of_frames(self):
        from evadb.functions.fastrcnn_object_detector import FastRCNNObjectDetector
        frame_dog = {'id': 1, 'data': self._load_image(os.path.join(self.base_path, 'data', 'dog.jpeg'))}
        frame_dog_cat = {'id': 2, 'data': self._load_image(os.path.join(self.base_path, 'data', 'dog_cat.jpg'))}
        frame_batch = Batch(pd.DataFrame([frame_dog, frame_dog_cat]))
        detector = FastRCNNObjectDetector()
        result = detector.classify(frame_batch)
        self.assertEqual(['dog'], result[0].labels)
        self.assertEqual(['cat', 'dog'], result[1].labels)

def numpy_to_yolo_format(numpy_image):
    numpy_image = numpy_image.astype(np.float64)
    numpy_image = numpy_image / 255
    try_to_import_torch()
    import torch
    r = torch.tensor(numpy_image[:, :, 0])
    g = torch.tensor(numpy_image[:, :, 1])
    b = torch.tensor(numpy_image[:, :, 2])
    rgb = torch.stack((r, g, b), dim=0)
    rgb = rgb.unsqueeze(0)
    return rgb

def try_to_import_torch():
    try:
        import torch
    except ImportError:
        raise ValueError('Could not import torch python package.\n                Please install them with `pip install torch`.')

class YoloTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def _load_image(self, path):
        try_to_import_cv2()
        import cv2
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def test_should_raise_import_error_with_missing_torch(self):
        with self.assertRaises(ImportError):
            with mock.patch.dict(sys.modules, {'torch': None}):
                from evadb.functions.decorators.yolo_object_detection_decorators import Yolo
                pass

    @unittest.skip('disable test due to model downloading time')
    def test_should_return_batches_equivalent_to_number_of_frames(self):
        from evadb.functions.decorators.yolo_object_detection_decorators import Yolo
        frame_dog = {'id': 1, 'data': self._load_image(os.path.join(self.base_path, 'data', 'dog.jpeg'))}
        frame_dog_cat = {'id': 2, 'data': self._load_image(os.path.join(self.base_path, 'data', 'dog_cat.jpg'))}
        test_df_dog = pd.DataFrame([frame_dog])
        test_df_cat = pd.DataFrame([frame_dog_cat])
        frame_dog = numpy_to_yolo_format(test_df_dog['data'].values[0])
        frame_cat = numpy_to_yolo_format(test_df_cat['data'].values[0])
        detector = Yolo()
        result = []
        result.append(detector.forward(frame_dog))
        result.append(detector.forward(frame_cat))
        self.assertEqual(['dog'], result[0]['labels'].tolist()[0])
        self.assertEqual(['cat', 'dog'], result[1]['labels'].tolist()[0])

class FaceNet(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_path = Path(EvaDB_TEST_DATA_DIR) / 'data' / 'facenet'

    def _load_image(self, path):
        assert path.exists(), f'File does not exist at the path {str(path)}'
        try_to_import_cv2()
        import cv2
        img = cv2.imread(str(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @windows_skip_marker
    def test_should_return_batches_equivalent_to_number_of_frames(self):
        from evadb.functions.face_detector import FaceDetector
        single_face_img = Path('data/facenet/one.jpg')
        multi_face_img = Path('data/facenet/multiface.jpg')
        frame_single_face = {'id': 1, 'data': self._load_image(single_face_img)}
        frame_multifaces = {'id': 2, 'data': self._load_image(multi_face_img)}
        frame_batch = Batch(pd.DataFrame([frame_single_face, frame_single_face]))
        detector = FaceDetector()
        result = detector(frame_batch.project(['data']).frames)
        self.assertEqual(1, len(result.iloc[0]['bboxes']))
        self.assertEqual(1, len(result.iloc[1]['bboxes']))
        frame_batch = Batch(pd.DataFrame([frame_multifaces]))
        detector = FaceDetector()
        result = detector(frame_batch.project(['data']).frames)
        self.assertEqual(6, len(result.iloc[0]['bboxes']))

    @unittest.skip('Needs GPU')
    def test_should_run_on_gpu(self):
        from evadb.functions.face_detector import FaceDetector
        single_face_img = Path('data/facenet/one.jpg')
        frame_single_face = {'id': 1, 'data': self._load_image(single_face_img)}
        frame_batch = Batch(pd.DataFrame([frame_single_face, frame_single_face]))
        detector = FaceDetector().to_device(0)
        result = detector(frame_batch.project(['data']).frames)
        self.assertEqual(6, len(result.iloc[0]['bboxes']))

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

def try_to_import_pillow():
    try:
        import PIL
    except ImportError:
        raise ValueError('Could not import pillow python package.\n                Please install it with `pip install pillow`.')

class AnnotateTests(unittest.TestCase):

    def setUp(self):
        try_to_import_pillow()
        self.annotate_instance = Annotate()

    def test_annotate_name_exists(self):
        assert hasattr(self.annotate_instance, 'name')

    def test_should_annotate(self):
        from PIL import Image
        img = Image.open(f'{EvaDB_ROOT_DIR}/test/data/uadetrac/small-data/MVI_20011/img00001.jpg')
        arr = asarray(img)
        arr_copy = 0 + arr
        object_type = np.array(['object'])
        bbox = np.array([[50, 50, 70, 70]])
        df = pd.DataFrame([[arr, object_type, bbox]])
        modified_arr = self.annotate_instance(df)['annotated_frame_array']
        self.assertNotEqual(np.sum(arr_copy - modified_arr[0]), 0)
        self.assertEqual(np.sum(modified_arr[0][50][50] - np.array([207, 248, 64])), 0)
        self.assertEqual(np.sum(modified_arr[0][70][70] - np.array([207, 248, 64])), 0)
        self.assertEqual(np.sum(modified_arr[0][50][70] - np.array([207, 248, 64])), 0)
        self.assertEqual(np.sum(modified_arr[0][70][50] - np.array([207, 248, 64])), 0)

@pytest.mark.notparallel
class OpenTests(unittest.TestCase):

    def setUp(self):
        self.open_instance = Open()
        self.image_file_path = create_sample_image()
        try_to_import_cv2()

    def test_open_name_exists(self):
        assert hasattr(self.open_instance, 'name')

    def test_should_open_image(self):
        df = self.open_instance(pd.DataFrame([self.image_file_path]))
        actual_img = df['data'].to_numpy()[0]
        expected_img = np.array(np.ones((3, 3, 3)), dtype=np.uint8)
        expected_img[0] -= 1
        expected_img[2] += 1
        self.assertEqual(actual_img.shape, expected_img.shape)
        self.assertEqual(np.sum(actual_img[0]), np.sum(expected_img[0]))
        self.assertEqual(np.sum(actual_img[1]), np.sum(expected_img[1]))
        self.assertEqual(np.sum(actual_img[2]), np.sum(expected_img[2]))

    def test_open_same_path_should_use_cache(self):
        import cv2
        with patch('cv2.imread') as mock_cv2_imread:
            self.open_instance(pd.DataFrame([self.image_file_path]))
            mock_cv2_imread.assert_called_once_with(self.image_file_path)
        with patch('cv2.imread') as mock_cv2_imread:
            self.open_instance(pd.DataFrame([self.image_file_path]))
            mock_cv2_imread.assert_not_called()

    def test_open_path_should_raise_error(self):
        with self.assertRaises((AssertionError, FileNotFoundError)):
            self.open_instance(pd.DataFrame(['incorrect_path']))

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

@pytest.mark.notparallel
class DecordLoaderTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        try_to_import_decord()
        self.video_file_url = create_sample_video()
        self.video_with_audio_file_url = f'{EvaDB_ROOT_DIR}/data/sample_videos/touchdown.mp4'
        self.frame_size = FRAME_SIZE[0] * FRAME_SIZE[1] * 3
        self.audio_frames = []
        for line in open(f'{EvaDB_ROOT_DIR}/test/data/touchdown_audio_frames.csv').readlines():
            self.audio_frames.append(np.fromstring(line, sep=','))

    @classmethod
    def tearDownClass(self):
        file_remove('dummy.avi')

    def _batches_to_reader_convertor(self, batches):
        new_batches = []
        for batch in batches:
            batch.drop_column_alias()
            new_batches.append(batch.project(['id', 'data', 'seconds', '_row_number']))
        return new_batches

    def test_should_sample_only_iframe(self):
        for k in range(1, 10):
            video_loader = DecordReader(file_url=self.video_file_url, sampling_type=IFRAMES, sampling_rate=k)
            batches = list(video_loader.read())
            expected = self._batches_to_reader_convertor(create_dummy_batches(filters=[i for i in range(0, NUM_FRAMES, k)], is_from_storage=True))
            self.assertEqual(batches, expected)

    def test_should_sample_every_k_frame_with_predicate(self):
        col = TupleValueExpression('id')
        val = ConstantValueExpression(NUM_FRAMES // 2)
        predicate = ComparisonExpression(ExpressionType.COMPARE_GEQ, left=col, right=val)
        for k in range(2, 4):
            video_loader = DecordReader(file_url=self.video_file_url, sampling_rate=k, predicate=predicate)
            batches = list(video_loader.read())
            value = NUM_FRAMES // 2
            start = value + k - value % k if value % k else value
            expected = self._batches_to_reader_convertor(create_dummy_batches(filters=[i for i in range(start, NUM_FRAMES, k)], is_from_storage=True))
        self.assertEqual(batches, expected)
        value = 2
        predicate_1 = ComparisonExpression(ExpressionType.COMPARE_GEQ, left=TupleValueExpression('id'), right=ConstantValueExpression(value))
        predicate_2 = ComparisonExpression(ExpressionType.COMPARE_LEQ, left=TupleValueExpression('id'), right=ConstantValueExpression(8))
        predicate = LogicalExpression(ExpressionType.LOGICAL_AND, predicate_1, predicate_2)
        for k in range(2, 4):
            video_loader = DecordReader(file_url=self.video_file_url, sampling_rate=k, predicate=predicate)
            batches = list(video_loader.read())
            start = value + k - value % k if value % k else value
            expected = self._batches_to_reader_convertor(create_dummy_batches(filters=[i for i in range(start, 8, k)], is_from_storage=True))
        self.assertEqual(batches, expected)

    def test_should_return_one_batch(self):
        video_loader = DecordReader(file_url=self.video_file_url)
        batches = list(video_loader.read())
        expected = self._batches_to_reader_convertor(create_dummy_batches(is_from_storage=True))
        self.assertEqual(batches, expected)

    def test_should_return_batches_equivalent_to_number_of_frames(self):
        video_loader = DecordReader(file_url=self.video_file_url, batch_mem_size=self.frame_size)
        batches = list(video_loader.read())
        expected = self._batches_to_reader_convertor(create_dummy_batches(batch_size=1, is_from_storage=True))
        self.assertEqual(batches, expected)

    def test_should_sample_every_k_frame(self):
        for k in range(1, 10):
            video_loader = DecordReader(file_url=self.video_file_url, sampling_rate=k)
            batches = list(video_loader.read())
            expected = self._batches_to_reader_convertor(create_dummy_batches(filters=[i for i in range(0, NUM_FRAMES, k)], is_from_storage=True))
            self.assertEqual(batches, expected)

    def test_should_throw_error_for_audioless_video(self):
        with self.assertRaises(AssertionError) as error_context:
            video_loader = DecordReader(file_url=self.video_file_url, read_audio=True, read_video=True)
            list(video_loader.read())
        self.assertIn("Can't find audio stream", error_context.exception.args[0].args[0])

    def test_should_throw_error_when_sampling_iframes_for_audio(self):
        with self.assertRaises(AssertionError) as error_context:
            video_loader = DecordReader(file_url=self.video_with_audio_file_url, sampling_type=IFRAMES, read_audio=True, read_video=False)
            list(video_loader.read())
        self.assertEquals('Cannot use IFRAMES with audio streams', error_context.exception.args[0])

    def test_should_throw_error_when_sampling_audio_for_video(self):
        with self.assertRaises(AssertionError) as error_context:
            video_loader = DecordReader(file_url=self.video_file_url, sampling_type=AUDIORATE, read_audio=False, read_video=True)
            list(video_loader.read())
        self.assertEquals('Cannot use AUDIORATE with video streams', error_context.exception.args[0])

    def test_should_return_audio_frames(self):
        video_loader = DecordReader(file_url=self.video_with_audio_file_url, sampling_type=AUDIORATE, sampling_rate=16000, read_audio=True, read_video=False)
        batches = list(video_loader.read())
        batches = batches[0].frames[batches[0].frames.index.isin([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])].reset_index()
        for i, frame in enumerate(self.audio_frames):
            self.assertTrue(np.array_equiv(self.audio_frames[i], batches.iloc[i]['audio']))
        self.assertEqual(batches.iloc[0]['data'].shape, (0,))

def try_to_import_decord():
    try:
        import decord
    except ImportError:
        raise ValueError('Could not import decord python package.\n                Please install it with `pip install eva-decord`.')

class DecoratorTests(unittest.TestCase):

    def test_setup_flags_are_updated(self):

        @setup(cacheable=True, function_type='classification', batchable=True)
        def setup_func():
            pass
        setup_func()
        self.assertTrue(setup_func.tags['cacheable'])
        self.assertTrue(setup_func.tags['batchable'])
        self.assertEqual(setup_func.tags['function_type'], 'classification')

    def test_setup_flags_are_updated_with_default_values(self):

        @setup()
        def setup_func():
            pass
        setup_func()
        self.assertFalse(setup_func.tags['cacheable'])
        self.assertTrue(setup_func.tags['batchable'])
        self.assertEqual(setup_func.tags['function_type'], 'Abstract')

    def test_forward_flags_are_updated(self):
        input_type = PandasDataframe(columns=['Frame_Array'], column_types=[NdArrayType.UINT8], column_shapes=[(3, 256, 256)])
        output_type = NumpyArray(name='label', type=NdArrayType.STR)

        @forward(input_signatures=[input_type], output_signatures=[output_type])
        def forward_func():
            pass
        forward_func()
        self.assertEqual(forward_func.tags['input'], [input_type])
        self.assertEqual(forward_func.tags['output'], [output_type])

def setup(cacheable: bool=False, function_type: str='Abstract', batchable: bool=True):
    """decorator for the setup function. It will be used to set the cache, batching and
    function_type parameters in the catalog

    Args:
        use_cache (bool): True if the function should be cached
        function_type (str): Type of the function
        batch (bool): True if the function should be batched
    """

    def inner_fn(arg_fn):

        def wrapper(*args, **kwargs):
            arg_fn(*args, **kwargs)
        tags = {}
        tags['cacheable'] = cacheable
        tags['function_type'] = function_type
        tags['batchable'] = batchable
        wrapper.tags = tags
        return wrapper
    return inner_fn

class LoadMultimediaExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: LoadDataPlan):
        super().__init__(db, node)
        self.media_type = self.node.file_options['file_format']
        if self.media_type == FileFormatType.IMAGE:
            try_to_import_cv2()
        elif self.media_type == FileFormatType.VIDEO:
            try_to_import_decord()
            try_to_import_cv2()

    def exec(self, *args, **kwargs):
        storage_engine = None
        table_obj = None
        try:
            video_files = []
            valid_files = []
            if self.node.file_path.as_posix().startswith('s3:/'):
                s3_dir = Path(self.catalog().get_configuration_catalog_value('s3_download_dir'))
                dst_path = s3_dir / self.node.table_info.table_name
                dst_path.mkdir(parents=True, exist_ok=True)
                video_files = download_from_s3(self.node.file_path, dst_path)
            else:
                video_files = list(iter_path_regex(self.node.file_path))
            valid_files, invalid_files = ([], [])
            if len(video_files) < mp.cpu_count() * 2:
                valid_bitmap = [self._is_media_valid(path) for path in video_files]
            else:
                pool = Pool(mp.cpu_count())
                valid_bitmap = pool.map(self._is_media_valid, video_files)
            if False in valid_bitmap:
                invalid_files = [str(path) for path, is_valid in zip(video_files, valid_bitmap) if not is_valid]
                invalid_files_str = '\n'.join(invalid_files)
                err_msg = f"no valid file found at -- '{invalid_files_str}'."
                logger.error(err_msg)
            valid_files = [str(path) for path, is_valid in zip(video_files, valid_bitmap) if is_valid]
            if not valid_files:
                raise DatasetFileNotFoundError(f"no file found at -- '{str(self.node.file_path)}'.")
            table_info = self.node.table_info
            database_name = table_info.database_name
            table_name = table_info.table_name
            do_create = False
            table_obj = self.catalog().get_table_catalog_entry(table_name, database_name)
            if table_obj:
                msg = f'Adding to an existing table {table_name}.'
                logger.info(msg)
            else:
                table_obj = self.catalog().create_and_insert_multimedia_table_catalog_entry(table_name, self.media_type)
                do_create = True
            storage_engine = StorageEngine.factory(self.db, table_obj)
            if do_create:
                storage_engine.create(table_obj)
            storage_engine.write(table_obj, Batch(pd.DataFrame({'file_path': valid_files})))
        except Exception as e:
            if storage_engine and table_obj:
                self._rollback_load(storage_engine, table_obj, do_create)
            err_msg = f'Load {self.media_type.name} failed: {str(e)}'
            raise ExecutorError(err_msg)
        else:
            yield Batch(pd.DataFrame([f'Number of loaded {self.media_type.name}: {str(len(valid_files))}']))

    def _rollback_load(self, storage_engine: AbstractStorageEngine, table_obj: TableCatalogEntry, do_create: bool):
        if do_create:
            storage_engine.drop(table_obj)

    def _is_media_valid(self, file_path: Path):
        file_path = Path(file_path)
        if validate_media(file_path, self.media_type):
            return True
        return False

class AudioHFModel(AbstractHFFunction):
    """
    Base Model for all HF Models that take in audio as input
    """

    def input_formatter(self, inputs: Any):
        if inputs.columns.str.contains('audio').any():
            return np.concatenate(inputs.iloc[:, 0].values)
        audio = []
        files = inputs.iloc[:, 0].tolist()
        try_to_import_decord()
        import decord
        for file in files:
            reader = decord.AudioReader(file, mono=True, sample_rate=16000)
            audio.append(reader[0:].asnumpy()[0])
        return audio

class DecordReader(AbstractReader):

    def __init__(self, *args, predicate: AbstractExpression=None, sampling_rate: int=None, sampling_type: str=None, read_audio: bool=False, read_video: bool=True, **kwargs):
        """Read frames from the disk

        Args:
            predicate (AbstractExpression, optional): If only subset of frames
            need to be read. The predicate should be only on single column and
            can be converted to ranges. Defaults to None.
            sampling_rate (int, optional): Set if the caller wants one frame
            every `sampling_rate` number of frames. For example, if `sampling_rate = 10`, it returns every 10th frame. If both `predicate` and `sampling_rate` are specified, `sampling_rate` is given precedence.
            sampling_type (str, optional): Set as IFRAMES if caller want to sample on top on iframes only. e.g if the IFRAME frame numbers are [10,20,30,40,50] then 'SAMPLE IFRAMES 2' will return [10,30,50]
            read_audio (bool, optional): Whether to read audio stream from the video. Defaults to False
            read_video (bool, optional): Whether to read video stream from the video. Defaults to True
        """
        self._predicate = predicate
        self._sampling_rate = sampling_rate or 1
        self._sampling_type = sampling_type
        self._read_audio = read_audio
        self._read_video = read_video
        self._reader = None
        self._get_frame = None
        super().__init__(*args, **kwargs)
        self.initialize_reader()

    def _read(self) -> Iterator[Dict]:
        num_frames = int(len(self._reader))
        if self._predicate:
            range_list = extract_range_list_from_predicate(self._predicate, 0, num_frames - 1)
        else:
            range_list = [(0, num_frames - 1)]
        logger.debug('Reading frames')
        if self._sampling_type == IFRAMES:
            iframes = self._reader.get_key_indices()
            idx = 0
            for begin, end in range_list:
                while idx < len(iframes) and iframes[idx] < begin:
                    idx += self._sampling_rate
                while idx < len(iframes) and iframes[idx] <= end:
                    frame_id = iframes[idx]
                    idx += self._sampling_rate
                    yield self._get_frame(frame_id)
        elif self._sampling_rate == 1 or self._read_audio:
            for begin, end in range_list:
                frame_id = begin
                while frame_id <= end:
                    yield self._get_frame(frame_id)
                    frame_id += 1
        else:
            for begin, end in range_list:
                if begin % self._sampling_rate:
                    begin += self._sampling_rate - begin % self._sampling_rate
                for frame_id in range(begin, end + 1, self._sampling_rate):
                    yield self._get_frame(frame_id)

    def initialize_reader(self):
        try_to_import_decord()
        import decord
        if self._read_audio:
            assert self._sampling_type != IFRAMES, 'Cannot use IFRAMES with audio streams'
            sample_rate = 16000
            if self._sampling_type == AUDIORATE and self._sampling_rate != 1:
                sample_rate = self._sampling_rate
            try:
                self._reader = decord.AVReader(self.file_url, mono=True, sample_rate=sample_rate)
                self._get_frame = self.__get_audio_frame
            except decord._ffi.base.DECORDError as error_msg:
                assert "Can't find audio stream" not in str(error_msg), error_msg
        else:
            assert self._sampling_type != AUDIORATE, 'Cannot use AUDIORATE with video streams'
            self._reader = decord.VideoReader(self.file_url)
            self._get_frame = self.__get_video_frame

    def __get_video_frame(self, frame_id):
        frame_video = self._reader[frame_id]
        frame_video = frame_video.asnumpy()
        timestamp = self._reader.get_frame_timestamp(frame_id)[0]
        return {VideoColumnName.id.name: frame_id, ROW_NUM_COLUMN: frame_id, VideoColumnName.data.name: frame_video, VideoColumnName.seconds.name: round(timestamp, 2)}

    def __get_audio_frame(self, frame_id):
        frame_audio, _ = self._reader[frame_id]
        frame_audio = frame_audio.asnumpy()[0]
        return {VideoColumnName.id.name: frame_id, ROW_NUM_COLUMN: frame_id, VideoColumnName.data.name: np.empty(0), VideoColumnName.seconds.name: 0.0, VideoColumnName.audio.name: frame_audio}

class CVImageReader(AbstractReader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _read(self) -> Iterator[Dict]:
        try_to_import_cv2()
        import cv2
        im_bgr = cv2.imread(str(self.file_url))
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        assert im_rgb is not None, f'Failed to read image file {self.file_url}'
        yield {'data': im_rgb}

class ChatGPT(AbstractFunction):
    """
    Arguments:
        model (str) : ID of the OpenAI model to use. Refer to '_VALID_CHAT_COMPLETION_MODEL' for a list of supported models.
        temperature (float) : Sampling temperature to use in the model. Higher value results in a more random output.

    Input Signatures:
        query (str)   : The task / question that the user wants the model to accomplish / respond.
        content (str) : Any relevant context that the model can use to complete its tasks and generate the response.
        prompt (str)  : An optional prompt that can be passed to the model. It can contain instructions to the model,
                        or a set of examples to help the model generate a better response.
                        If not provided, the system prompt defaults to that of an helpful assistant that accomplishes user tasks.

    Output Signatures:
        response (str) : Contains the response generated by the model based on user input. Any errors encountered
                         will also be passed in the response.

    Example Usage:
        Assume we have the transcripts for a few videos stored in a table 'video_transcripts' in a column named 'text'.
        If the user wants to retrieve the summary of each video, the ChatGPT function can be used as:

            query = "Generate the summary of the video"
            cursor.table("video_transcripts").select(f"ChatGPT({query}, text)")

        In the above function invocation, the 'query' passed would be the user task to generate video summaries, and the
        'content' passed would be the video transcripts that need to be used in order to generate the summary. Since
        no prompt is passed, the default system prompt will be used.

        Now assume the user wants to create the video summary in 50 words and in French. Instead of passing these instructions
        along with each query, a prompt can be set as such:

            prompt = "Generate your responses in 50 words or less. Also, generate the response in French."
            cursor.table("video_transcripts").select(f"ChatGPT({query}, text, {prompt})")

        In the above invocation, an additional argument is passed as prompt. While the query and content arguments remain
        the same, the 'prompt' argument will be set as a system message in model params.

        Both of the above cases would generate a summary for each row / video transcript of the table in the response.
    """

    @property
    def name(self) -> str:
        return 'ChatGPT'

    @setup(cacheable=True, function_type='chat-completion', batchable=True)
    def setup(self, model='gpt-3.5-turbo', temperature: float=0, openai_api_key='') -> None:
        assert model in _VALID_CHAT_COMPLETION_MODEL, f'Unsupported ChatGPT {model}'
        self.model = model
        self.temperature = temperature
        self.openai_api_key = openai_api_key

    @forward(input_signatures=[PandasDataframe(columns=['query', 'content', 'prompt'], column_types=[NdArrayType.STR, NdArrayType.STR, NdArrayType.STR], column_shapes=[(1,), (1,), (None,)])], output_signatures=[PandasDataframe(columns=['response'], column_types=[NdArrayType.STR], column_shapes=[(1,)])])
    def forward(self, text_df):
        try_to_import_openai()
        from openai import OpenAI
        api_key = self.openai_api_key
        if len(self.openai_api_key) == 0:
            api_key = os.environ.get('OPENAI_API_KEY', '')
        assert len(api_key) != 0, "Please set your OpenAI API key using SET OPENAI_API_KEY = 'sk-' or environment variable (OPENAI_API_KEY)"
        client = OpenAI(api_key=api_key)

        @retry(tries=6, delay=20)
        def completion_with_backoff(**kwargs):
            return client.chat.completions.create(**kwargs)
        queries = text_df[text_df.columns[0]]
        content = text_df[text_df.columns[0]]
        if len(text_df.columns) > 1:
            queries = text_df.iloc[:, 0]
            content = text_df.iloc[:, 1]
        prompt = None
        if len(text_df.columns) > 2:
            prompt = text_df.iloc[0, 2]
        results = []
        for query, content in zip(queries, content):
            params = {'model': self.model, 'temperature': self.temperature, 'messages': []}
            def_sys_prompt_message = {'role': 'system', 'content': prompt if prompt is not None else 'You are a helpful assistant that accomplishes user tasks.'}
            params['messages'].append(def_sys_prompt_message)
            params['messages'].extend([{'role': 'user', 'content': f'Here is some context : {content}'}, {'role': 'user', 'content': f'Complete the following task: {query}'}])
            response = completion_with_backoff(**params)
            answer = response.choices[0].message.content
            results.append(answer)
        df = pd.DataFrame({'response': results})
        return df

class MVITActionRecognition(PytorchAbstractClassifierFunction):

    @property
    def name(self) -> str:
        return 'MVITActionRecognition'

    def setup(self):
        try_to_import_torchvision()
        try_to_import_torch()
        from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s
        self.weights = MViT_V2_S_Weights.DEFAULT
        self.model = mvit_v2_s(weights=self.weights)
        self.preprocess = self.weights.transforms()
        self.model.eval()

    @property
    def labels(self) -> np.array([str]):
        return np.array(self.weights.meta['categories'])

    def forward(self, segments):
        return self.classify(segments)

    def transform(self, segments):
        import torch
        segments = torch.Tensor(segments)
        segments = segments.permute(0, 3, 1, 2)
        return self.preprocess(segments).unsqueeze(0)

    def classify(self, segments) -> pd.DataFrame:
        import torch
        with torch.no_grad():
            preds = self.model(segments).softmax(1)
        label_indices = preds.argmax(axis=1)
        actions = self.labels[label_indices]
        if np.isscalar(actions) == 1:
            outcome = pd.DataFrame({'labels': np.array([actions])})
        return outcome

def try_to_import_torchvision():
    try:
        import torchvision
    except ImportError:
        raise ValueError('Could not import torchvision python package.\n                Please install them with `pip install torchvision`.')

class SiftFeatureExtractor(AbstractFunction, GPUCompatible):

    @setup(cacheable=False, function_type='FeatureExtraction', batchable=False)
    def setup(self):
        try_to_import_kornia()
        import kornia
        self.model = kornia.feature.SIFTDescriptor(100)

    def to_device(self, device: str) -> GPUCompatible:
        self.model = self.model.to(device)
        return self

    @property
    def name(self) -> str:
        return 'SiftFeatureExtractor'

    @forward(input_signatures=[PandasDataframe(columns=['data'], column_types=[NdArrayType.UINT8], column_shapes=[(None, None, 3)])], output_signatures=[PandasDataframe(columns=['features'], column_types=[NdArrayType.FLOAT32], column_shapes=[(1, 128)])])
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:

        def _forward(row: pd.Series) -> np.ndarray:
            rgb_img = row[0]
            try_to_import_cv2()
            import cv2
            gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            resized_gray_img = cv2.resize(gray_img, (100, 100), interpolation=cv2.INTER_AREA)
            resized_gray_img = np.moveaxis(resized_gray_img, -1, 0)
            batch_resized_gray_img = np.expand_dims(resized_gray_img, axis=0)
            batch_resized_gray_img = np.expand_dims(batch_resized_gray_img, axis=0)
            batch_resized_gray_img = batch_resized_gray_img.astype(np.float32)
            try_to_import_torch()
            import torch
            with torch.no_grad():
                torch_feat = self.model(torch.from_numpy(batch_resized_gray_img))
                feat = torch_feat.numpy()
            feat = feat.reshape(1, -1)
            return feat
        ret = pd.DataFrame()
        ret['features'] = df.apply(_forward, axis=1)
        return ret

def try_to_import_kornia():
    try:
        import kornia
    except ImportError:
        raise ValueError('Could not import kornia python package.\n                Please install it with `pip install kornia`.')

class FuzzDistance(AbstractFunction):

    @setup(cacheable=False, function_type='FeatureExtraction', batchable=False)
    def setup(self):
        pass

    @property
    def name(self) -> str:
        return 'FuzzDistance'

    @forward(input_signatures=[PandasDataframe(columns=['data1', 'data2'], column_types=[NdArrayType.STR, NdArrayType.STR], column_shapes=[1, 1])], output_signatures=[PandasDataframe(columns=['distance'], column_types=[NdArrayType.FLOAT32], column_shapes=[1])])
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:

        def _forward(row: pd.Series) -> np.ndarray:
            data1 = row.iloc[0]
            data2 = row.iloc[1]
            distance = fuzz.ratio(data1, data2)
            return distance
        ret = pd.DataFrame()
        ret['distance'] = df.apply(_forward, axis=1)
        return ret

class FastRCNNObjectDetector(PytorchAbstractClassifierFunction):
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score

    """

    @property
    def name(self) -> str:
        return 'fastrcnn'

    @setup(cacheable=True, function_type='object_detection', batchable=True)
    def setup(self, threshold=0.85):
        try_to_import_torch()
        try_to_import_torchvision()
        import torchvision
        self.threshold = threshold
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1', progress=False)
        self.model.eval()

    @property
    def labels(self) -> List[str]:
        return ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    @forward(input_signatures=[PyTorchTensor(name='input_col', is_nullable=False, type=NdArrayType.FLOAT32, dimensions=(1, 3, 540, 960))], output_signatures=[PandasDataframe(columns=['labels', 'bboxes', 'scores'], column_types=[NdArrayType.STR, NdArrayType.FLOAT32, NdArrayType.FLOAT32], column_shapes=[(None,), (None,), (None,)])])
    def forward(self, frames) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (np.ndarray): Frames on which predictions need
            to be performed

        Returns:
            tuple containing predicted_classes (List[List[str]]),
            predicted_boxes (List[List[BoundingBox]]),
            predicted_scores (List[List[float]])

        """
        predictions = self.model(frames)
        outcome = []
        for prediction in predictions:
            pred_class = [str(self.labels[i]) for i in list(self.as_numpy(prediction['labels']))]
            pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(self.as_numpy(prediction['boxes']))]
            pred_score = list(self.as_numpy(prediction['scores']))
            valid_pred = [pred_score.index(x) for x in pred_score if x > self.threshold]
            if valid_pred:
                pred_t = valid_pred[-1]
            else:
                pred_t = -1
            pred_boxes = np.array(pred_boxes[:pred_t + 1])
            pred_class = np.array(pred_class[:pred_t + 1])
            pred_score = np.array(pred_score[:pred_t + 1])
            outcome.append({'labels': pred_class, 'scores': pred_score, 'bboxes': pred_boxes})
        return pd.DataFrame(outcome, columns=['labels', 'scores', 'bboxes'])

class ForecastModel(AbstractFunction):

    @property
    def name(self) -> str:
        return 'ForecastModel'

    @setup(cacheable=False, function_type='Forecasting', batchable=True)
    def setup(self, model_name: str, model_path: str, predict_column_rename: str, time_column_rename: str, id_column_rename: str, horizon: int, library: str, conf: int):
        self.library = library
        if 'neuralforecast' in self.library:
            from neuralforecast import NeuralForecast
            loaded_model = NeuralForecast.load(path=model_path)
            self.model_name = model_name[4:] if 'Auto' in model_name else model_name
        else:
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            self.model_name = model_name
        self.model = loaded_model
        self.predict_column_rename = predict_column_rename
        self.time_column_rename = time_column_rename
        self.id_column_rename = id_column_rename
        self.horizon = int(horizon)
        self.library = library
        self.suggestion_dict = {1: "Predictions are flat. Consider using LIBRARY 'neuralforecast' for more accrate predictions."}
        self.conf = conf
        self.hypers = None
        self.rmse = None
        if os.path.isfile(model_path + '_rmse'):
            with open(model_path + '_rmse', 'r') as f:
                self.rmse = float(f.readline())
                if 'arima' in model_name.lower():
                    self.hypers = 'p,d,q: ' + f.readline()

    def forward(self, data) -> pd.DataFrame:
        log_str = ''
        if self.library == 'statsforecast':
            forecast_df = self.model.predict(h=self.horizon, level=[self.conf]).reset_index()
        else:
            forecast_df = self.model.predict().reset_index()
        if len(data) == 0 or list(list(data.iloc[0]))[0] is True:
            suggestion_list = []
            if self.library == 'statsforecast':
                for type_here in forecast_df['unique_id'].unique():
                    if forecast_df.loc[forecast_df['unique_id'] == type_here][self.model_name].nunique() == 1:
                        suggestion_list.append(1)
            for suggestion in set(suggestion_list):
                log_str += '\nSUGGESTION: ' + self.suggestion_dict[suggestion]
            if self.rmse is not None:
                log_str += '\nMean normalized RMSE: ' + str(self.rmse)
            if self.hypers is not None:
                log_str += '\nHyperparameters: ' + self.hypers
            print(log_str)
        forecast_df = forecast_df.rename(columns={'unique_id': self.id_column_rename, 'ds': self.time_column_rename, self.model_name if self.library == 'statsforecast' else self.model_name + '-median': self.predict_column_rename, self.model_name + '-lo-' + str(self.conf): self.predict_column_rename + '-lo', self.model_name + '-hi-' + str(self.conf): self.predict_column_rename + '-hi'})[:self.horizon * forecast_df['unique_id'].nunique()]
        return forecast_df

class SaliencyFeatureExtractor(AbstractFunction, GPUCompatible):

    @setup(cacheable=False, function_type='FeatureExtraction', batchable=False)
    def setup(self):
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.eval()

    def to_device(self, device: str) -> GPUCompatible:
        self.model = self.model.to(device)
        return self

    @property
    def name(self) -> str:
        return 'SaliencyFeatureExtractor'

    @forward(input_signatures=[PandasDataframe(columns=['data'], column_types=[NdArrayType.UINT8], column_shapes=[(None, None, 3)])], output_signatures=[PandasDataframe(columns=['saliency'], column_types=[NdArrayType.FLOAT32], column_shapes=[(1, 224, 224)])])
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:

        def _forward(row: pd.Series) -> np.ndarray:
            rgb_img = row[0]
            composed = Compose([Resize((224, 224)), ToTensor()])
            transformed_img = composed(Image.fromarray(rgb_img[:, :, ::-1])).unsqueeze(0)
            transformed_img.requires_grad_()
            outputs = self.model(transformed_img)
            score_max_index = outputs.argmax()
            score_max = outputs[0, score_max_index]
            score_max.backward()
            saliency, _ = torch.max(transformed_img.grad.data.abs(), dim=1)
            return saliency
        ret = pd.DataFrame()
        ret['saliency'] = df.apply(_forward, axis=1)
        return ret

class TextFilterKeyword(AbstractFunction):

    @setup(cacheable=False, function_type='TextProcessing', batchable=False)
    def setup(self):
        pass

    @property
    def name(self) -> str:
        return 'TextFilterKeyword'

    @forward(input_signatures=[PandasDataframe(columns=['data', 'keyword'], column_types=[NdArrayType.STR, NdArrayType.STR], column_shapes=[1, 1])], output_signatures=[PandasDataframe(columns=['filtered'], column_types=[NdArrayType.STR], column_shapes=[1])])
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:

        def _forward(row: pd.Series) -> np.ndarray:
            import re
            data = row.iloc[0]
            keywords = row.iloc[1]
            flag = False
            for i in keywords:
                pattern = f'^(.*?({i})[^$]*)$'
                match_check = re.search(pattern, data, re.IGNORECASE)
                if match_check:
                    flag = True
            if flag is False:
                return data
            flag = False
        ret = pd.DataFrame()
        ret['filtered'] = df.apply(_forward, axis=1)
        return ret

class ASLActionRecognition(PytorchAbstractClassifierFunction):

    @property
    def name(self) -> str:
        return 'ASLActionRecognition'

    def download_weights(self):
        try_to_import_torch()
        import torch
        if not os.path.exists(self.asl_weights_path):
            torch.hub.download_url_to_file(self.asl_weights_url, self.asl_weights_path, hash_prefix=None, progress=True)

    def setup(self):
        self.asl_weights_url = 'https://www.dropbox.com/s/s9l1mezuplc6ttl/asl_top20_resnet_wts.pth?raw=1'
        import torch
        self.asl_weights_path = torch.hub.get_dir() + '/asl_weights.pth'
        self.download_weights()
        try_to_import_torchvision()
        from torchvision.models.video import R3D_18_Weights, r3d_18
        self.weights = R3D_18_Weights.DEFAULT
        self.model = r3d_18(weights=self.weights)
        in_feats = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_feats, 20)
        self.model.load_state_dict(torch.load(self.asl_weights_path, map_location='cpu'))
        self.model.eval()
        self.preprocess = self.weights.transforms()

    @property
    def labels(self) -> np.array([str]):
        current_file_path = os.path.dirname(os.path.realpath(__file__))
        pkl_file_path = os.path.join(current_file_path, 'asl_20_actions_map.pkl')
        with open(pkl_file_path, 'rb') as f:
            action_to_index_map = pkl.load(f)
        actions_arr = [''] * len(action_to_index_map)
        for action, index in action_to_index_map.items():
            actions_arr[index] = action
        return np.asarray(actions_arr)

    def forward(self, segments):
        return self.classify(segments)

    def transform(self, segments):
        import torch
        segments = torch.Tensor(segments)
        permute_order = [2, 1, 0]
        segments = segments[:, :, :, permute_order]
        segments = segments.permute(0, 3, 1, 2).to(torch.uint8)
        return self.preprocess(segments).unsqueeze(0)

    def classify(self, segments) -> pd.DataFrame:
        import torch
        with torch.no_grad():
            preds = self.model(segments).softmax(1)
        label_indices = preds.argmax(axis=1)
        actions = self.labels[label_indices]
        if np.isscalar(actions) == 1:
            outcome = pd.DataFrame({'labels': np.array([actions])})
        return outcome

class SentenceTransformerFeatureExtractor(AbstractFunction, GPUCompatible):

    @setup(cacheable=False, function_type='FeatureExtraction', batchable=False)
    def setup(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def to_device(self, device: str) -> GPUCompatible:
        self.model = self.model.to(device)
        return self

    @property
    def name(self) -> str:
        return 'SentenceTransformerFeatureExtractor'

    @forward(input_signatures=[PandasDataframe(columns=['data'], column_types=[NdArrayType.STR], column_shapes=[1])], output_signatures=[PandasDataframe(columns=['features'], column_types=[NdArrayType.FLOAT32], column_shapes=[(1, 384)])])
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:

        def _forward(row: pd.Series) -> np.ndarray:
            data = row
            embedded_list = self.model.encode(data)
            return embedded_list
        ret = pd.DataFrame()
        ret['features'] = df.apply(_forward, axis=1)
        return ret

class MnistImageClassifier(PytorchAbstractClassifierFunction):

    @property
    def name(self) -> str:
        return 'MnistImageClassifier'

    def setup(self):
        try_to_import_torch()
        try_to_import_torchvision()
        import torch
        import torch.nn as nn
        model_urls = {'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth'}

        class MLP(nn.Module):

            def __init__(self, input_dims, n_hiddens, n_class):
                super(MLP, self).__init__()
                assert isinstance(input_dims, int), 'Please provide int for input_dims'
                self.input_dims = input_dims
                current_dims = input_dims
                layers = OrderedDict()
                if isinstance(n_hiddens, int):
                    n_hiddens = [n_hiddens]
                else:
                    n_hiddens = list(n_hiddens)
                for i, n_hidden in enumerate(n_hiddens):
                    layers['fc{}'.format(i + 1)] = nn.Linear(current_dims, n_hidden)
                    layers['relu{}'.format(i + 1)] = nn.ReLU()
                    layers['drop{}'.format(i + 1)] = nn.Dropout(0.2)
                    current_dims = n_hidden
                layers['out'] = nn.Linear(current_dims, n_class)
                self.model = nn.Sequential(layers)

            def forward(self, input):
                input = input.view(input.size(0), -1)
                assert input.size(1) == self.input_dims
                return self.model.forward(input)

        def mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=None):
            model = MLP(input_dims, n_hiddens, n_class)
            import torch.utils.model_zoo as model_zoo
            if pretrained is not None:
                m = model_zoo.load_url(model_urls['mnist'], map_location=torch.device('cpu'))
                state_dict = m.state_dict() if isinstance(m, nn.Module) else m
                assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
                model.load_state_dict(state_dict)
            return model
        self.model = mnist(pretrained=True)
        self.model.eval()

    @property
    def labels(self):
        return list([str(num) for num in range(10)])

    def transform(self, images):
        from PIL import Image
        from torchvision.transforms import Compose, Grayscale, Normalize, ToTensor
        composed = Compose([Grayscale(num_output_channels=1), ToTensor(), Normalize((0.1307,), (0.3081,))])
        return composed(Image.fromarray(images[:, :, ::-1])).unsqueeze(0)

    def forward(self, frames) -> pd.DataFrame:
        outcome = []
        predictions = self.model(frames)
        for prediction in predictions:
            label = self.as_numpy(prediction.data.argmax())
            outcome.append({'label': str(label)})
        return pd.DataFrame(outcome, columns=['label'])

class EmotionDetector(PytorchAbstractClassifierFunction):
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score
    """

    @property
    def name(self) -> str:
        return 'EmotionDetector'

    def _download_weights(self, weights_url, weights_path):
        import torch
        if not os.path.exists(weights_path):
            torch.hub.download_url_to_file(weights_url, weights_path, hash_prefix=None, progress=True)

    def setup(self, threshold=0.85):
        self.threshold = threshold
        try_to_import_pillow()
        try_to_import_torch()
        try_to_import_torchvision()
        import torch
        import torch.nn.functional as F
        model_url = 'https://www.dropbox.com/s/85b63eahka5r439/emotion_detector.t7?raw=1'
        model_weights_path = torch.hub.get_dir() + '/emotion_detector.t7'
        self._download_weights(model_url, model_weights_path)

        class VGG(torch.nn.Module):

            def __init__(self, vgg_name):
                super(VGG, self).__init__()
                self.features = self._make_layers(cfg[vgg_name])
                self.classifier = torch.nn.Linear(512, 7)

            def forward(self, x):
                out = self.features(x)
                out = out.view(out.size(0), -1)
                out = F.dropout(out, p=0.5, training=self.training)
                out = self.classifier(out)
                return out

            def _make_layers(self, cfg):
                layers = []
                in_channels = 3
                for x in cfg:
                    if x == 'M':
                        layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
                    else:
                        layers += [torch.nn.Conv2d(in_channels, x, kernel_size=3, padding=1), torch.nn.BatchNorm2d(x), torch.nn.ReLU(inplace=True)]
                        in_channels = x
                layers += [torch.nn.AvgPool2d(kernel_size=1, stride=1)]
                return torch.nn.Sequential(*layers)
        self.model = VGG('VGG19')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_state = torch.load(model_weights_path, map_location=device)
        self.model.load_state_dict(model_state['net'])
        self.model.eval()
        self.cut_size = 44

    def transforms_ed(self, frame):
        """
        Performs augmentation on input frame
        Arguments:
            frame (Tensor): Frame on which augmentation needs
            to be performed
        Returns:
            frame (Tensor): Augmented frame
        """
        from torchvision import transforms
        frame = frame.convert('L')
        frame = transforms.functional.resize(frame, (48, 48))
        frame = transforms.functional.to_tensor(frame)
        return frame

    def transform(self, images: np.ndarray):
        from PIL import Image
        return self.transforms_ed(Image.fromarray(images[:, :, ::-1]))

    @property
    def labels(self) -> List[str]:
        return ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def forward(self, frames) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (Tensor): Frames on which predictions need
            to be performed
        Returns:
            outcome (pd.DataFrame): Emotion Predictions for input frames
        """
        outcome = []
        import torch
        import torch.nn.functional as F
        from torchvision import transforms
        frames = frames.repeat(3, 1, 1)
        frames = transforms.functional.ten_crop(frames, self.cut_size)
        frames = torch.stack([crop for crop in frames])
        predictions = self.model(frames)
        predictions = torch.mean(predictions, dim=0)
        score = F.softmax(predictions, dim=0)
        _, predicted = torch.max(predictions.data, 0)
        outcome.append({'labels': self.labels[predicted.item()], 'scores': score.cpu().detach().numpy()[predicted.item()]})
        return pd.DataFrame(outcome, columns=['labels', 'scores'])

class FaceDetector(AbstractClassifierFunction, GPUCompatible):
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score
    """

    def setup(self, threshold=0.85):
        self.threshold = threshold
        try_to_import_torch()
        try_to_import_torchvision()
        try_to_import_facenet_pytorch()
        from facenet_pytorch import MTCNN
        self.model = MTCNN()

    @property
    def name(self) -> str:
        return 'FaceDetector'

    def to_device(self, device: str):
        try_to_import_facenet_pytorch()
        import torch
        from facenet_pytorch import MTCNN
        gpu = 'cuda:{}'.format(device)
        self.model = MTCNN(device=torch.device(gpu))
        return self

    @property
    def labels(self) -> List[str]:
        return []

    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (np.ndarray): Frames on which predictions need
            to be performed
        Returns:
            face boxes (List[List[BoundingBox]])
        """
        frames_list = frames.transpose().values.tolist()[0]
        frames = np.asarray(frames_list)
        detections = self.model.detect(frames)
        boxes, scores = detections
        outcome = []
        for frame_boxes, frame_scores in zip(boxes, scores):
            pred_boxes = []
            pred_scores = []
            if frame_boxes is not None and frame_scores is not None:
                if not np.isnan(pred_boxes):
                    pred_boxes = np.asarray(frame_boxes, dtype='int')
                    pred_scores = frame_scores
                else:
                    logger.warn(f'Nan entry in box {frame_boxes}')
            outcome.append({'bboxes': pred_boxes, 'scores': pred_scores})
        return pd.DataFrame(outcome, columns=['bboxes', 'scores'])

def try_to_import_facenet_pytorch():
    try:
        import facenet_pytorch
    except ImportError:
        raise ValueError('Could not import facenet_pytorch python package.\n                Please install it with `pip install facenet-pytorch`.')

class GaussianBlur(AbstractFunction):

    @setup(cacheable=False, function_type='cv2-transformation', batchable=True)
    def setup(self):
        pass

    @property
    def name(self):
        return 'GaussianBlur'

    @forward(input_signatures=[PandasDataframe(columns=['data'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])], output_signatures=[PandasDataframe(columns=['blurred_frame_array'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])])
    def forward(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Gaussian Blur to the frame

         Returns:
             ret (pd.DataFrame): The modified frame.
        """

        def gaussianBlur(row: pd.Series) -> np.ndarray:
            row = row.to_list()
            frame = row[0]
            try_to_import_cv2()
            import cv2
            frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        ret = pd.DataFrame()
        ret['blurred_frame_array'] = frame.apply(gaussianBlur, axis=1)
        return ret

class Annotate(AbstractFunction):

    @setup(cacheable=False, function_type='cv2-transformation', batchable=True)
    def setup(self):
        pass

    @property
    def name(self):
        return 'Annotate'

    @forward(input_signatures=[PandasDataframe(columns=['data', 'labels', 'bboxes'], column_types=[NdArrayType.FLOAT32, NdArrayType.STR, NdArrayType.FLOAT32], column_shapes=[(None, None, 3), (None,), (None,)])], output_signatures=[PandasDataframe(columns=['annotated_frame_array'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])])
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Modify the frame to annotate the bbox on it.

         Returns:
             ret (pd.DataFrame): The modified frame.
        """

        def annotate(row: pd.Series) -> np.ndarray:
            row = row.to_list()
            frame = row[0]
            bboxes = row[2]
            try_to_import_cv2()
            import cv2
            for bbox in bboxes:
                x1, y1, x2, y2 = np.asarray(bbox, dtype='int')
                x1, y1, x2, y2 = (int(x1), int(y1), int(x2), int(y2))
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            return frame
        ret = pd.DataFrame()
        ret['annotated_frame_array'] = df.apply(annotate, axis=1)
        return ret

class VerticalFlip(AbstractFunction):

    @setup(cacheable=False, function_type='cv2-transformation', batchable=True)
    def setup(self):
        try_to_import_cv2()

    @property
    def name(self):
        return 'VerticalFlip'

    @forward(input_signatures=[PandasDataframe(columns=['data'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])], output_signatures=[PandasDataframe(columns=['vertically_flipped_frame_array'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])])
    def forward(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Apply vertical flip to the frame

         Returns:
             ret (pd.DataFrame): The modified frame.
        """

        def verticalFlip(row: pd.Series) -> np.ndarray:
            row = row.to_list()
            frame = row[0]
            try_to_import_cv2()
            import cv2
            frame = cv2.flip(frame, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        ret = pd.DataFrame()
        ret['vertically_flipped_frame_array'] = frame.apply(verticalFlip, axis=1)
        return ret

class HorizontalFlip(AbstractFunction):

    @setup(cacheable=False, function_type='cv2-transformation', batchable=True)
    def setup(self):
        try_to_import_cv2()

    @property
    def name(self):
        return 'HorizontalFlip'

    @forward(input_signatures=[PandasDataframe(columns=['data'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])], output_signatures=[PandasDataframe(columns=['horizontally_flipped_frame_array'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])])
    def forward(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Apply horizontal flip to the frame

         Returns:
             ret (pd.DataFrame): The modified frame.
        """

        def horizontalFlip(row: pd.Series) -> np.ndarray:
            row = row.to_list()
            frame = row[0]
            try_to_import_cv2()
            import cv2
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        ret = pd.DataFrame()
        ret['horizontally_flipped_frame_array'] = frame.apply(horizontalFlip, axis=1)
        return ret

class Open(AbstractFunction):

    def setup(self):
        self._data_cache = dict()
        try_to_import_cv2()

    @property
    def name(self):
        return 'Open'

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Open image from server-side path.

        Returns:
            (pd.DataFrame): The opened image.
        """

        def _open(row: pd.Series) -> np.ndarray:
            path_str = row[0]
            if path_str in self._data_cache:
                data = self._data_cache[path_str]
            else:
                import cv2
                data = cv2.imread(path_str)
                assert data is not None, f'Failed to open file {path_str}'
            self._data_cache[path_str] = data
            return data
        ret = pd.DataFrame()
        ret['data'] = df.apply(_open, axis=1)
        return ret

class ToGrayscale(AbstractFunction):

    @setup(cacheable=False, function_type='cv2-transformation', batchable=True)
    def setup(self):
        try_to_import_cv2()

    @property
    def name(self):
        return 'ToGrayscale'

    @forward(input_signatures=[PandasDataframe(columns=['data'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])], output_signatures=[PandasDataframe(columns=['grayscale_frame_array'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])])
    def forward(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the frame from BGR to grayscale

         Returns:
             ret (pd.DataFrame): The modified frame.
        """

        def toGrayscale(row: pd.Series) -> np.ndarray:
            row = row.to_list()
            frame = row[0]
            import cv2
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            return frame
        ret = pd.DataFrame()
        ret['grayscale_frame_array'] = frame.apply(toGrayscale, axis=1)
        return ret

class EvaDBTrackerAbstractFunction(AbstractFunction):
    """
    An abstract class for all EvaDB object trackers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @setup(cacheable=False, function_type='object_tracker', batchable=False)
    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

    @forward(input_signatures=[PandasDataframe(columns=['frame_id', 'frame', 'bboxes', 'scores', 'labels'], column_types=[NdArrayType.INT32, NdArrayType.FLOAT32, NdArrayType.FLOAT32, NdArrayType.FLOAT32, NdArrayType.STR], column_shapes=[(1,), (None, None, 3), (None, 4), (None,), (None,)])], output_signatures=[PandasDataframe(columns=['track_ids', 'track_labels', 'track_bboxes', 'track_scores'], column_types=[NdArrayType.INT32, NdArrayType.INT32, NdArrayType.FLOAT32, NdArrayType.FLOAT32], column_shapes=[(None,), (None,), (None, 4), (None,)])])
    def forward(self, frame_id: numpy.ndarray, frame: numpy.ndarray, labels: numpy.ndarray, bboxes: numpy.ndarray, scores: numpy.ndarray) -> typing.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """
        Args:
            frame_id (numpy.ndarray): the frame id of current frame
            frame (numpy.ndarray): the input frame with shape (C, H, W)
            labels (numpy.ndarray): Corresponding labels for each box
            bboxes (numpy.ndarray): Array of shape `(n, 4)` or of shape `(4,)` where
            each row contains `(xmin, ymin, width, height)`.
            scores (numpy.ndarray): Corresponding scores for each box
        Returns:
            track_ids (numpy.ndarray): Corresponding track id for each box
            track_labels (numpy.ndarray): Corresponding labels for each box
            track_bboxes (numpy.ndarray):  Array of shape `(n, 4)` of tracked objects
            track_scores (numpy.ndarray): Corresponding scores for each box
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        assert isinstance(args[0], pd.DataFrame), f'Expecting pd.DataFrame, got {type(args[0])}'
        results = []
        for _, row in args[0].iterrows():
            tuple = (numpy.array(row[0]), numpy.array(row[1]), numpy.stack(row[2]), numpy.stack(row[3]), numpy.stack(row[4]))
            results.append(self.forward(*tuple))
        return pd.DataFrame(results, columns=['track_ids', 'track_labels', 'track_bboxes', 'track_scores'])

