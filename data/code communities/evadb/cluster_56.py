# Cluster 56

def create_dataframe_same(times=1):
    base_df = create_dataframe()
    for i in range(1, times):
        base_df = pd.concat([base_df, create_dataframe()], ignore_index=True)
    return base_df

def create_dataframe(num_frames=1) -> pd.DataFrame:
    frames = []
    for i in range(1, num_frames + 1):
        frames.append({'id': i, 'data': i * np.ones((1, 1))})
    return pd.DataFrame(frames)

@pytest.mark.notparallel
class LoadExecutorTests(unittest.TestCase):

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        self.video_file_path = create_sample_video()
        self.image_files_path = Path(f'{EvaDB_ROOT_DIR}/test/data/uadetrac/small-data/MVI_20011/*.jpg')
        self.csv_file_path = create_sample_csv()

    def tearDown(self):
        shutdown_ray()
        file_remove('dummy.avi')
        file_remove('dummy.csv')
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyVideos;')

    def test_should_load_video_in_table(self):
        query = f"LOAD VIDEO '{self.video_file_path}' INTO MyVideo;"
        execute_query_fetch_all(self.evadb, query)
        select_query = 'SELECT * FROM MyVideo;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches())[0]
        self.assertEqual(actual_batch, expected_batch)
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyVideo;')

    def test_should_form_symlink_to_individual_video(self):
        catalog_manager = self.evadb.catalog()
        query = f"LOAD VIDEO '{self.video_file_path}' INTO MyVideo;"
        execute_query_fetch_all(self.evadb, query)
        table_catalog_entry = catalog_manager.get_table_catalog_entry('MyVideo')
        video_dir = table_catalog_entry.file_url
        self.assertEqual(len(os.listdir(video_dir)), 1)
        video_file = os.listdir(video_dir)[0]
        video_file_path = os.path.join(video_dir, video_file)
        self.assertTrue(os.path.islink(video_file_path))
        self.assertEqual(os.readlink(video_file_path), str(Path(self.video_file_path).resolve()))
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyVideo;')

    def test_should_raise_error_on_removing_symlinked_file(self):
        query = f"LOAD VIDEO '{self.video_file_path}' INTO MyVideo;"
        execute_query_fetch_all(self.evadb, query)
        select_query = 'SELECT * FROM MyVideo;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches())[0]
        self.assertEqual(actual_batch, expected_batch)
        file_remove('dummy.avi')
        with self.assertRaises(ExecutorError) as e:
            execute_query_fetch_all(self.evadb, select_query, do_not_print_exceptions=True)
        self.assertEqual(str(e.exception), 'The dataset file could not be found. Please verify that the file exists in the specified path.')
        create_sample_video()

    def test_should_form_symlink_to_multiple_videos(self):
        catalog_manager = self.evadb.catalog()
        path = f'{EvaDB_ROOT_DIR}/data/sample_videos/1/*.mp4'
        query = f'LOAD VIDEO "{path}" INTO MyVideos;'
        execute_query_fetch_all(self.evadb, query)
        table_catalog_entry = catalog_manager.get_table_catalog_entry('MyVideos')
        video_dir = table_catalog_entry.file_url
        self.assertEqual(len(os.listdir(video_dir)), 2)
        video_files = os.listdir(video_dir)
        for video_file in video_files:
            video_file_path = os.path.join(video_dir, video_file)
            self.assertTrue(os.path.islink(video_file_path))
            self.assertTrue(os.readlink(video_file_path).startswith(f'{EvaDB_ROOT_DIR}/data/sample_videos/1'))

    def test_should_load_videos_with_same_name_but_different_path(self):
        path = f'{EvaDB_ROOT_DIR}/data/sample_videos/**/*.mp4'
        query = f'LOAD VIDEO "{path}" INTO MyVideos;'
        result = execute_query_fetch_all(self.evadb, query)
        expected = Batch(pd.DataFrame([f'Number of loaded {FileFormatType.VIDEO.name}: 4']))
        self.assertEqual(result, expected)

    def test_should_fail_to_load_videos_with_same_path(self):
        path = f'{EvaDB_ROOT_DIR}/data/sample_videos/2/*.mp4'
        query = f'LOAD VIDEO "{path}" INTO MyVideos;'
        result = execute_query_fetch_all(self.evadb, query)
        expected = Batch(pd.DataFrame([f'Number of loaded {FileFormatType.VIDEO.name}: 1']))
        self.assertEqual(result, expected)
        expected_output = execute_query_fetch_all(self.evadb, 'SELECT id FROM MyVideos;')
        path = f'{EvaDB_ROOT_DIR}/data/sample_videos/**/*.mp4'
        query = f'LOAD VIDEO "{path}" INTO MyVideos;'
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, query, do_not_print_exceptions=True)
        after_load_fail = execute_query_fetch_all(self.evadb, 'SELECT id FROM MyVideos;')
        self.assertEqual(expected_output, after_load_fail)

    def test_should_fail_to_load_missing_video(self):
        path = f'{EvaDB_ROOT_DIR}/data/sample_videos/missing.mp4'
        query = f'LOAD VIDEO "{path}" INTO MyVideos;'
        with self.assertRaises(ExecutorError) as exc_info:
            execute_query_fetch_all(self.evadb, query, do_not_print_exceptions=True)
        self.assertIn('Load VIDEO failed', str(exc_info.exception))

    def test_should_fail_to_load_corrupt_video(self):
        tempfile_name = os.urandom(24).hex()
        tempfile_path = os.path.join(tempfile.gettempdir(), tempfile_name)
        with open(tempfile_path, 'wb') as tmp:
            query = f'LOAD VIDEO "{tmp.name}" INTO MyVideos;'
            with self.assertRaises(Exception):
                execute_query_fetch_all(self.evadb, query, do_not_print_exceptions=True)

    def test_should_fail_to_load_invalid_files_as_video(self):
        path = f'{EvaDB_ROOT_DIR}/data/README.md'
        query = f'LOAD VIDEO "{path}" INTO MyVideos;'
        with self.assertRaises(Exception):
            execute_query_fetch_all(self.evadb, query, do_not_print_exceptions=True)
        with self.assertRaises(BinderError):
            execute_query_fetch_all(self.evadb, 'SELECT name FROM MyVideos', do_not_print_exceptions=True)

    def test_should_rollback_or_skip_if_video_load_fails(self):
        path_regex = Path(f'{EvaDB_ROOT_DIR}/data/sample_videos/1/*.mp4')
        valid_videos = glob.glob(str(path_regex.expanduser()), recursive=True)
        tempfile_name = os.urandom(24).hex()
        tempfile_path = os.path.join(tempfile.gettempdir(), tempfile_name)
        with open(tempfile_path, 'wb') as empty_file:
            with tempfile.TemporaryDirectory() as tmp_dir:
                shutil.copy2(str(empty_file.name), tmp_dir)
                path = Path(tmp_dir) / '*'
                query = f'LOAD VIDEO "{path}" INTO MyVideos;'
                with self.assertRaises(Exception):
                    execute_query_fetch_all(self.evadb, query, do_not_print_exceptions=True)
                with self.assertRaises(BinderError):
                    execute_query_fetch_all(self.evadb, 'SELECT name FROM MyVideos', do_not_print_exceptions=True)
            with tempfile.TemporaryDirectory() as tmp_dir:
                shutil.copy2(str(valid_videos[0]), tmp_dir)
                shutil.copy2(str(empty_file.name), tmp_dir)
                path = Path(tmp_dir) / '*'
                query = f'LOAD VIDEO "{path}" INTO MyVideos;'
                result = execute_query_fetch_all(self.evadb, query)
                expected = Batch(pd.DataFrame([f'Number of loaded {FileFormatType.VIDEO.name}: 1']))
                self.assertEqual(result, expected)
            with tempfile.TemporaryDirectory() as tmp_dir:
                shutil.copy2(str(valid_videos[0]), tmp_dir)
                shutil.copy2(str(valid_videos[1]), tmp_dir)
                shutil.copy2(str(empty_file.name), tmp_dir)
                path = Path(tmp_dir) / '*'
                query = f'LOAD VIDEO "{path}" INTO MyVideos;'
                result = execute_query_fetch_all(self.evadb, query)
                expected = Batch(pd.DataFrame([f'Number of loaded {FileFormatType.VIDEO.name}: 2']))
                self.assertEqual(result, expected)

    def test_should_rollback_and_preserve_previous_state(self):
        path_regex = Path(f'{EvaDB_ROOT_DIR}/data/sample_videos/1/*.mp4')
        valid_videos = glob.glob(str(path_regex.expanduser()), recursive=True)
        load_file = f'{EvaDB_ROOT_DIR}/data/sample_videos/1/1.mp4'
        execute_query_fetch_all(self.evadb, f'LOAD VIDEO "{load_file}" INTO MyVideos;')
        tempfile_name = os.urandom(24).hex()
        tempfile_path = os.path.join(tempfile.gettempdir(), tempfile_name)
        with open(tempfile_path, 'wb') as empty_file:
            with tempfile.TemporaryDirectory() as tmp_dir:
                shutil.copy2(str(empty_file.name), tmp_dir)
                path = Path(tmp_dir) / '*'
                query = f'LOAD VIDEO "{path}" INTO MyVideos;'
                with self.assertRaises(Exception):
                    execute_query_fetch_all(self.evadb, query, do_not_print_exceptions=True)
                result = execute_query_fetch_all(self.evadb, 'SELECT name FROM MyVideos')
                file_names = np.unique(result.frames)
                self.assertEqual(len(file_names), 1)
            with tempfile.TemporaryDirectory() as tmp_dir:
                shutil.copy2(str(valid_videos[1]), tmp_dir)
                shutil.copy2(str(empty_file.name), tmp_dir)
                path = Path(tmp_dir) / '*'
                query = f'LOAD VIDEO "{path}" INTO MyVideos;'
                result = execute_query_fetch_all(self.evadb, query)
                expected = Batch(pd.DataFrame([f'Number of loaded {FileFormatType.VIDEO.name}: 1']))
                self.assertEqual(result, expected)
                result = execute_query_fetch_all(self.evadb, 'SELECT name FROM MyVideos')
                file_names = np.unique(result.frames)
                self.assertEqual(len(file_names), 2)

    def test_should_fail_to_load_missing_image(self):
        path = f'{EvaDB_ROOT_DIR}/data/sample_images/missing.jpg'
        query = f'LOAD IMAGE "{path}" INTO MyImages;'
        with self.assertRaises(ExecutorError) as exc_info:
            execute_query_fetch_all(self.evadb, query, do_not_print_exceptions=True)
        self.assertIn('Load IMAGE failed', str(exc_info.exception))

    def test_should_fail_to_load_images_with_same_path(self):
        image_files = glob.glob(os.path.expanduser(self.image_files_path), recursive=True)
        query = f'LOAD IMAGE "{image_files[0]}" INTO MyImages;'
        result = execute_query_fetch_all(self.evadb, query)
        expected = Batch(pd.DataFrame([f'Number of loaded {FileFormatType.IMAGE.name}: 1']))
        self.assertEqual(result, expected)
        expected_output = execute_query_fetch_all(self.evadb, 'SELECT name FROM MyImages;')
        query = f'LOAD IMAGE "{image_files[0]}" INTO MyImages;'
        with self.assertRaises(Exception):
            execute_query_fetch_all(self.evadb, query, do_not_print_exceptions=True)
        after_load_fail = execute_query_fetch_all(self.evadb, 'SELECT name FROM MyImages;')
        self.assertEqual(expected_output, after_load_fail)

    def test_should_fail_to_load_corrupt_image(self):
        tempfile_name = os.urandom(24).hex()
        tempfile_path = os.path.join(tempfile.gettempdir(), tempfile_name)
        with open(tempfile_path, 'wb') as tmp:
            query = f'LOAD IMAGE "{tmp.name}" INTO MyImages;'
            with self.assertRaises(Exception):
                execute_query_fetch_all(self.evadb, query, do_not_print_exceptions=True)

    def test_should_fail_to_load_invalid_files_as_image(self):
        path = f'{EvaDB_ROOT_DIR}/data/README.md'
        query = f'LOAD IMAGE "{path}" INTO MyImages;'
        with self.assertRaises(Exception):
            execute_query_fetch_all(self.evadb, query, do_not_print_exceptions=True)
        with self.assertRaises(BinderError):
            execute_query_fetch_all(self.evadb, 'SELECT name FROM MyImages;', do_not_print_exceptions=True)

    def test_should_rollback_or_pass_if_image_load_fails(self):
        valid_images = glob.glob(str(self.image_files_path.expanduser()), recursive=True)
        tempfile_name = os.urandom(24).hex()
        tempfile_path = os.path.join(tempfile.gettempdir(), tempfile_name)
        with open(tempfile_path, 'wb') as empty_file:
            with tempfile.TemporaryDirectory() as tmp_dir:
                shutil.copy2(str(empty_file.name), tmp_dir)
                path = Path(tmp_dir) / '*'
                query = f'LOAD IMAGE "{path}" INTO MyImages;'
                with self.assertRaises(Exception):
                    execute_query_fetch_all(self.evadb, query, do_not_print_exceptions=True)
                with self.assertRaises(BinderError):
                    execute_query_fetch_all(self.evadb, 'SELECT name FROM MyImages;', do_not_print_exceptions=True)
            with tempfile.TemporaryDirectory() as tmp_dir:
                shutil.copy2(str(valid_images[0]), tmp_dir)
                shutil.copy2(str(empty_file.name), tmp_dir)
                path = Path(tmp_dir) / '*'
                query = f'LOAD IMAGE "{path}" INTO MyImages;'
                result = execute_query_fetch_all(self.evadb, query)
                expected = Batch(pd.DataFrame([f'Number of loaded {FileFormatType.IMAGE.name}: 1']))
                self.assertEqual(result, expected)
            with tempfile.TemporaryDirectory() as tmp_dir:
                shutil.copy2(str(valid_images[0]), tmp_dir)
                shutil.copy2(str(valid_images[1]), tmp_dir)
                shutil.copy2(str(empty_file.name), tmp_dir)
                path = Path(tmp_dir) / '*'
                query = f'LOAD IMAGE "{path}" INTO MyImages;'
                result = execute_query_fetch_all(self.evadb, query)
                expected = Batch(pd.DataFrame([f'Number of loaded {FileFormatType.IMAGE.name}: 2']))
                self.assertEqual(result, expected)

    def test_should_rollback_or_pass_and_preserve_previous_state_for_load_images(self):
        valid_images = glob.glob(str(self.image_files_path.expanduser()), recursive=True)
        execute_query_fetch_all(self.evadb, f'LOAD IMAGE "{valid_images[0]}" INTO MyImages;')
        tempfile_name = os.urandom(24).hex()
        tempfile_path = os.path.join(tempfile.gettempdir(), tempfile_name)
        with open(tempfile_path, 'wb') as empty_file:
            with tempfile.TemporaryDirectory() as tmp_dir:
                shutil.copy2(str(empty_file.name), tmp_dir)
                path = Path(tmp_dir) / '*'
                query = f'LOAD IMAGE "{path}" INTO MyImages;'
                with self.assertRaises(Exception):
                    execute_query_fetch_all(self.evadb, query, do_not_print_exceptions=True)
                result = execute_query_fetch_all(self.evadb, 'SELECT name FROM MyImages')
                self.assertEqual(len(result), 1)
                expected = Batch(pd.DataFrame([{'myimages.name': valid_images[0]}]))
                self.assertEqual(expected, result)
            with tempfile.TemporaryDirectory() as tmp_dir:
                shutil.copy2(str(valid_images[1]), tmp_dir)
                shutil.copy2(str(empty_file.name), tmp_dir)
                path = Path(tmp_dir) / '*'
                query = f'LOAD IMAGE "{path}" INTO MyImages;'
                result = execute_query_fetch_all(self.evadb, query)
                expected = Batch(pd.DataFrame([f'Number of loaded {FileFormatType.IMAGE.name}: 1']))
                self.assertEqual(result, expected)
                result = execute_query_fetch_all(self.evadb, 'SELECT name FROM MyImages')
                self.assertEqual(len(result), 2)
                expected = Batch(pd.DataFrame([{'myimages.name': valid_images[0]}, {'myimages.name': os.path.join(tmp_dir, os.path.basename(valid_images[1]))}]))
                self.assertEqual(expected, result)

    def test_should_load_csv_with_columns_in_table(self):
        create_table_query = '\n\n            CREATE TABLE IF NOT EXISTS MyVideoCSV (\n                id INTEGER UNIQUE,\n                frame_id INTEGER NOT NULL,\n                video_id INTEGER NOT NULL,\n                dataset_name TEXT(30) NOT NULL\n            );\n            '
        execute_query_fetch_all(self.evadb, create_table_query)
        load_query = "LOAD CSV '{}' INTO MyVideoCSV (id, frame_id, video_id, dataset_name);".format(self.csv_file_path)
        execute_query_fetch_all(self.evadb, load_query)
        select_query = 'SELECT id, frame_id, video_id, dataset_name\n                          FROM MyVideoCSV;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        select_columns = ['id', 'frame_id', 'video_id', 'dataset_name']
        expected_batch = next(create_dummy_csv_batches(target_columns=select_columns))
        expected_batch.modify_column_alias('myvideocsv')
        self.assertEqual(actual_batch, expected_batch)
        drop_query = 'DROP TABLE IF EXISTS MyVideoCSV;'
        execute_query_fetch_all(self.evadb, drop_query)

    def test_should_use_parallel_load(self):
        large_scale_image_files_path = create_large_scale_image_dataset(mp.cpu_count() * 10)
        load_query = f"LOAD IMAGE '{large_scale_image_files_path}/**/*.jpg' INTO MyLargeScaleImages;"
        execute_query_fetch_all(self.evadb, load_query)
        drop_query = 'DROP TABLE IF EXISTS MyLargeScaleImages;'
        execute_query_fetch_all(self.evadb, drop_query)
        shutil.rmtree(large_scale_image_files_path)

    def test_parallel_load_should_raise_exception_or_pass(self):
        large_scale_image_files_path = create_large_scale_image_dataset(mp.cpu_count() * 10)
        with open(os.path.join(large_scale_image_files_path, 'img0.jpg'), 'w') as f:
            f.write('aa')
        load_query = f"LOAD IMAGE '{large_scale_image_files_path}/**/*.jpg' INTO MyLargeScaleImages;"
        result = execute_query_fetch_all(self.evadb, load_query)
        file_count = len([entry for entry in os.listdir(large_scale_image_files_path) if os.path.isfile(os.path.join(large_scale_image_files_path, entry))])
        expected = Batch(pd.DataFrame([f'Number of loaded {FileFormatType.IMAGE.name}: {file_count - 1}']))
        self.assertEqual(result, expected)
        drop_query = 'DROP TABLE IF EXISTS MyLargeScaleImages;'
        execute_query_fetch_all(self.evadb, drop_query)
        shutil.rmtree(large_scale_image_files_path)

    def test_load_pdfs(self):
        execute_query_fetch_all(self.evadb, f"LOAD DOCUMENT '{EvaDB_ROOT_DIR}/data/documents/*.pdf' INTO pdfs;")
        result = execute_query_fetch_all(self.evadb, 'SELECT * from pdfs;')
        self.assertEqual(len(result.columns), 4)
        self.assertEqual(len(result), 26)

    def test_load_query_incorrect_fileFormat(self):
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, f"LOAD document '{EvaDB_ROOT_DIR}/data/documents/*.pdf' INTO pdfs;")

@pytest.mark.notparallel
class OpenTests(unittest.TestCase):

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        load_functions_for_testing(self.evadb, mode='debug')
        self.img_path = create_sample_image()
        create_table_query = 'CREATE TABLE IF NOT EXISTS testOpenTable (num INTEGER);'
        execute_query_fetch_all(self.evadb, create_table_query)
        table_catalog_entry = self.evadb.catalog().get_table_catalog_entry('testOpenTable')
        storage_engine = StorageEngine.factory(self.evadb, table_catalog_entry)
        storage_engine.write(table_catalog_entry, Batch(pd.DataFrame([{'num': 1}, {'num': 2}])))

    def tearDown(self):
        shutdown_ray()
        file_remove('dummy.jpg')
        drop_table_query = 'DROP TABLE testOpenTable;'
        execute_query_fetch_all(self.evadb, drop_table_query)

    def test_open_should_open_image(self):
        select_query = 'SELECT num, Open("{}") FROM testOpenTable;'.format(self.img_path)
        batch_res = execute_query_fetch_all(self.evadb, select_query)
        expected_img = np.array(np.ones((3, 3, 3)), dtype=np.float32)
        expected_img[0] -= 1
        expected_img[2] += 1
        expected_batch = Batch(pd.DataFrame({'testopentable.num': [1, 2], 'open.data': [expected_img, expected_img]}))
        batch_res.sort_orderby(by=['testopentable.num'])
        expected_batch.sort_orderby(by=['testopentable.num'])
        self.assertEqual(expected_batch, batch_res)

@pytest.mark.notparallel
class SelectExecutorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        video_file_path = create_sample_video(NUM_FRAMES)
        load_query = f"LOAD VIDEO '{video_file_path}' INTO MyVideo;"
        execute_query_fetch_all(cls.evadb, load_query)
        ua_detrac = f'{EvaDB_ROOT_DIR}/data/ua_detrac/ua_detrac.mp4'
        load_query = f"LOAD VIDEO '{ua_detrac}' INTO DETRAC;"
        execute_query_fetch_all(cls.evadb, load_query)
        load_functions_for_testing(cls.evadb)
        cls.table1 = create_table(cls.evadb, 'table1', 100, 3)
        cls.table2 = create_table(cls.evadb, 'table2', 500, 3)
        cls.table3 = create_table(cls.evadb, 'table3', 1000, 3)
        cls.meme1 = f'{EvaDB_ROOT_DIR}/data/detoxify/meme1.jpg'
        cls.meme2 = f'{EvaDB_ROOT_DIR}/data/detoxify/meme2.jpg'
        execute_query_fetch_all(cls.evadb, f"LOAD IMAGE '{cls.meme1}' INTO MemeImages;")
        execute_query_fetch_all(cls.evadb, f"LOAD IMAGE '{cls.meme2}' INTO MemeImages;")

    @classmethod
    def tearDownClass(cls):
        shutdown_ray()
        file_remove('dummy.avi')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS table1;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS table2;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS table3;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS MyVideo;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS MemeImages;')

    def test_should_load_and_select_real_audio_in_table(self):
        query = "LOAD VIDEO 'data/sample_videos/touchdown.mp4'\n                   INTO TOUCHDOWN;"
        execute_query_fetch_all(self.evadb, query)
        select_query = 'SELECT id, audio FROM TOUCHDOWN;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort('touchdown.id')
        video_reader = DecordReader('data/sample_videos/touchdown.mp4', read_audio=True)
        expected_batch = Batch(frames=pd.DataFrame())
        for batch in video_reader.read():
            batch.frames['name'] = 'touchdown.mp4'
            expected_batch += batch
        expected_batch.modify_column_alias('touchdown')
        expected_batch = expected_batch.project(['touchdown.id', 'touchdown.audio'])
        self.assertEqual(actual_batch, expected_batch)

    def test_chunk_param_should_fail(self):
        with self.assertRaises(AssertionError):
            execute_query_fetch_all(self.evadb, 'SELECT data from MyVideo chunk_size 4000 chunk_overlap 200;')
        with self.assertRaises(AssertionError):
            execute_query_fetch_all(self.evadb, 'SELECT data from MemeImages chunk_size 4000 chunk_overlap 200;')

    @pytest.mark.torchtest
    def test_lateral_join(self):
        select_query = 'SELECT id, a FROM DETRAC JOIN LATERAL\n                        Yolo(data) AS T(a,b,c) WHERE id < 5;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(list(actual_batch.columns), ['detrac.id', 'T.a'])
        self.assertEqual(len(actual_batch), 5)

    def test_complex_logical_expressions(self):
        query = "SELECT id FROM MyVideo\n            WHERE DummyObjectDetector(data).label = ['{}']  ORDER BY id;"
        persons = execute_query_fetch_all(self.evadb, query.format('person')).frames.to_numpy()
        bicycles = execute_query_fetch_all(self.evadb, query.format('bicycle')).frames.to_numpy()
        import numpy as np
        self.assertTrue(len(np.intersect1d(persons, bicycles)) == 0)
        query_or = "SELECT id FROM MyVideo             WHERE DummyObjectDetector(data).label = ['person']\n                OR DummyObjectDetector(data).label = ['bicycle']\n            ORDER BY id;"
        actual = execute_query_fetch_all(self.evadb, query_or)
        expected = execute_query_fetch_all(self.evadb, 'SELECT id FROM MyVideo ORDER BY id')
        self.assertEqual(expected, actual)
        query_and = "SELECT id FROM MyVideo             WHERE DummyObjectDetector(data).label = ['person']\n                AND DummyObjectDetector(data).label = ['bicycle']\n            ORDER BY id;"
        expected = execute_query_fetch_all(self.evadb, query_and)
        self.assertEqual(len(expected), 0)

    def test_select_and_union_video_in_table(self):
        select_query = 'SELECT * FROM MyVideo WHERE id < 3\n            UNION ALL SELECT * FROM MyVideo WHERE id > 7;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort('myvideo.id')
        expected_batch = list(create_dummy_batches(filters=[i for i in range(NUM_FRAMES) if i < 3 or i > 7]))[0]
        self.assertEqual(actual_batch, expected_batch)
        select_query = 'SELECT * FROM MyVideo WHERE id < 2\n            UNION ALL SELECT * FROM MyVideo WHERE id > 4 AND id < 6\n            UNION ALL SELECT * FROM MyVideo WHERE id > 7;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort('myvideo.id')
        expected_batch = list(create_dummy_batches(filters=[i for i in range(NUM_FRAMES) if i < 2 or i == 5 or i > 7]))[0]
        self.assertEqual(actual_batch, expected_batch)

    def test_sort_on_nonprojected_column(self):
        """This tests doing an order by on a column
        that is not projected. The orderby_executor currently
        catches the KeyError, passes, and returns the untouched
        data
        """
        select_query = 'SELECT data FROM MyVideo ORDER BY id;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        select_query = 'SELECT data FROM MyVideo'
        expected_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(actual_batch), len(expected_batch))

    def test_should_load_and_select_real_video_in_table(self):
        query = "LOAD VIDEO 'data/mnist/mnist.mp4'\n                   INTO MNIST;"
        execute_query_fetch_all(self.evadb, query)
        select_query = 'SELECT id, data FROM MNIST;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort('mnist.id')
        video_reader = DecordReader('data/mnist/mnist.mp4')
        expected_batch = Batch(frames=pd.DataFrame())
        for batch in video_reader.read():
            batch.frames['name'] = 'mnist.mp4'
            expected_batch += batch
        expected_batch.modify_column_alias('mnist')
        expected_batch = expected_batch.project(['mnist.id', 'mnist.data'])
        self.assertEqual(actual_batch, expected_batch)

    def test_project_identifier_column(self):
        batch = execute_query_fetch_all(self.evadb, 'SELECT _row_id, id FROM MyVideo;')
        expected = Batch(pd.DataFrame({'myvideo._row_id': [1] * NUM_FRAMES, 'myvideo.id': range(NUM_FRAMES)}))
        self.assertEqual(batch, expected)
        batch = execute_query_fetch_all(self.evadb, 'SELECT * FROM MyVideo;')
        self.assertTrue('myvideo._row_id' in batch.columns)
        batch = execute_query_fetch_all(self.evadb, 'SELECT _row_id, name FROM MemeImages;')
        expected = Batch(pd.DataFrame({'memeimages._row_id': [1, 2], 'memeimages.name': [self.meme1, self.meme2]}))
        self.assertEqual(batch, expected)
        batch = execute_query_fetch_all(self.evadb, 'SELECT * FROM MemeImages;')
        self.assertTrue('memeimages._row_id' in batch.columns)
        batch = execute_query_fetch_all(self.evadb, 'SELECT _row_id FROM table1;')
        expected = Batch(pd.DataFrame({'table1._row_id': range(1, 101)}))
        self.assertEqual(batch, expected)
        batch = execute_query_fetch_all(self.evadb, 'SELECT * FROM table1;')
        self.assertTrue('table1._row_id' in batch.columns)

    def test_select_and_sample(self):
        select_query = 'SELECT id FROM MyVideo SAMPLE 7 ORDER BY id;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches(filters=range(0, NUM_FRAMES, 7)))
        expected_batch[0] = expected_batch[0].project(['myvideo.id'])
        self.assertEqual(len(actual_batch), len(expected_batch[0]))
        self.assertEqual(actual_batch, expected_batch[0])

    def test_select_and_where_video_in_table(self):
        select_query = 'SELECT * FROM MyVideo WHERE id = 5;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected_batch = list(create_dummy_batches(filters=[5]))[0]
        self.assertEqual(actual_batch, expected_batch)
        select_query = 'SELECT data FROM MyVideo WHERE id = 5;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(actual_batch, expected_batch.project(['myvideo.data']))
        select_query = 'SELECT id, data FROM MyVideo WHERE id >= 2;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches(filters=range(2, NUM_FRAMES)))[0]
        self.assertEqual(actual_batch, expected_batch.project(['myvideo.id', 'myvideo.data']))
        select_query = 'SELECT * FROM MyVideo WHERE id >= 2 AND id < 5;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches(filters=range(2, 5)))[0]
        self.assertEqual(actual_batch, expected_batch)

    def test_hash_join_with_multiple_tables(self):
        select_query = 'SELECT * FROM table1 JOIN table2\n                          ON table2.a0 = table1.a0 JOIN table3\n                          ON table3.a1 = table1.a1 WHERE table1.a2 > 50;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        tmp = pd.merge(self.table1, self.table2, left_on=['table1.a0'], right_on=['table2.a0'], how='inner')
        expected = pd.merge(tmp, self.table3, left_on=['table1.a1'], right_on=['table3.a1'], how='inner')
        expected = expected.where(expected['table1.a2'] > 50)
        if len(expected):
            expected_batch = Batch(expected)
            self.assertEqual(expected_batch.sort_orderby(['table1.a0']), actual_batch.sort_orderby(['table1.a0']))

    def test_nested_select_video_in_table(self):
        nested_select_query = 'SELECT * FROM\n            (SELECT * FROM MyVideo WHERE id >= 2 AND id < 5) AS T\n            WHERE id >= 3;'
        actual_batch = execute_query_fetch_all(self.evadb, nested_select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches(filters=range(3, 5)))[0]
        expected_batch.modify_column_alias('T')
        self.assertEqual(actual_batch, expected_batch)
        nested_select_query = 'SELECT * FROM\n            (SELECT * FROM MyVideo WHERE id >= 2 AND id < 5) AS T\n            WHERE id >= 3;'
        actual_batch = execute_query_fetch_all(self.evadb, nested_select_query)
        actual_batch.sort('T.id')
        expected_batch = list(create_dummy_batches(filters=range(3, 5)))[0]
        expected_batch.modify_column_alias('T')
        self.assertEqual(actual_batch, expected_batch)

    def test_select_and_sample_with_predicate(self):
        select_query = 'SELECT id FROM MyVideo SAMPLE 2 WHERE id > 5 ORDER BY id;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected_batch = list(create_dummy_batches(filters=range(6, NUM_FRAMES, 2)))
        self.assertEqual(actual_batch, expected_batch[0].project(['myvideo.id']))
        select_query = 'SELECT id FROM MyVideo SAMPLE 4 WHERE id > 2 ORDER BY id;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected_batch = list(create_dummy_batches(filters=range(4, NUM_FRAMES, 4)))
        self.assertEqual(actual_batch, expected_batch[0].project(['myvideo.id']))
        select_query = 'SELECT id FROM MyVideo SAMPLE 2 WHERE id > 2 AND id < 8 ORDER BY id;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected_batch = list(create_dummy_batches(filters=range(4, 8, 2)))
        self.assertEqual(actual_batch, expected_batch[0].project(['myvideo.id']))

    def test_lateral_join_with_unnest_on_subset_of_outputs(self):
        query = 'SELECT id, label\n                  FROM MyVideo JOIN LATERAL\n                    UNNEST(DummyMultiObjectDetector(data).labels) AS T(label)\n                  WHERE id < 2 ORDER BY id;'
        unnest_batch = execute_query_fetch_all(self.evadb, query)
        expected = Batch(pd.DataFrame({'myvideo.id': np.array([0, 0, 1, 1], np.intp), 'T.label': np.array(['person', 'person', 'bicycle', 'bicycle'])}))
        self.assertEqual(unnest_batch, expected)
        query = 'SELECT id, label\n                FROM MyVideo JOIN LATERAL\n                UNNEST(DummyMultiObjectDetector(data).labels) AS T(label)\n                WHERE id < 2 AND T.label = "person" ORDER BY id;'
        unnest_batch = execute_query_fetch_all(self.evadb, query)
        expected = Batch(pd.DataFrame({'myvideo.id': np.array([0, 0], np.intp), 'T.label': np.array(['person', 'person'])}))
        self.assertEqual(unnest_batch, expected)

    def test_lateral_join_with_unnest(self):
        query = 'SELECT id, label\n                  FROM MyVideo JOIN LATERAL\n                    UNNEST(DummyObjectDetector(data)) AS T(label)\n                  WHERE id < 2 ORDER BY id;'
        unnest_batch = execute_query_fetch_all(self.evadb, query)
        expected = Batch(pd.DataFrame({'myvideo.id': np.array([0, 1], dtype=np.intp), 'T.label': np.array(['person', 'bicycle'])}))
        self.assertEqual(unnest_batch, expected)
        query = 'SELECT id, label\n                  FROM MyVideo JOIN LATERAL\n                    UNNEST(DummyObjectDetector(data)) AS T\n                  WHERE id < 2 ORDER BY id;'
        unnest_batch = execute_query_fetch_all(self.evadb, query)
        expected = Batch(pd.DataFrame({'myvideo.id': np.array([0, 1], dtype=np.intp), 'T.label': np.array(['person', 'bicycle'])}))
        self.assertEqual(unnest_batch, expected)

    @pytest.mark.torchtest
    def test_lateral_join_with_multiple_projects(self):
        select_query = 'SELECT id, T.labels FROM DETRAC JOIN LATERAL\n                        Yolo(data) AS T WHERE id < 5;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertTrue(all(actual_batch.frames.columns == ['detrac.id', 'T.labels']))
        self.assertEqual(len(actual_batch), 5)

    def test_should_select_star_in_table(self):
        select_query = 'SELECT * FROM MyVideo;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches())[0]
        self.assertEqual(actual_batch, expected_batch)
        select_query = 'SELECT * FROM MyVideo WHERE id = 5;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected_batch = list(create_dummy_batches(filters=[5]))[0]
        self.assertEqual(actual_batch, expected_batch)

class CreateTableTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        video_file_path = create_sample_video()
        load_query = f"LOAD VIDEO '{video_file_path}' INTO MyVideo;"
        execute_query_fetch_all(cls.evadb, load_query)
        ua_detrac = f'{EvaDB_ROOT_DIR}/data/ua_detrac/ua_detrac.mp4'
        execute_query_fetch_all(cls.evadb, f"LOAD VIDEO '{ua_detrac}' INTO UATRAC;")
        load_functions_for_testing(cls.evadb)

    @classmethod
    def tearDownClass(cls):
        file_remove('dummy.avi')
        file_remove('ua_detrac.mp4')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS MyVideo;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS UATRAC;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS uadtrac_fastRCNN;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS dummy_table;')
        shutdown_ray()

    def _test_currently_cannot_create_boolean_table(self):
        query = ' CREATE TABLE BooleanTable( A BOOLEAN);'
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, query)

    def test_should_create_table_from_select(self):
        create_query = 'CREATE TABLE dummy_table\n            AS SELECT id, DummyObjectDetector(data).label FROM MyVideo;\n        '
        execute_query_fetch_all(self.evadb, create_query)
        select_query = 'SELECT id, label FROM dummy_table;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        labels = DummyObjectDetector().labels
        expected = [{'dummy_table.id': i, 'dummy_table.label': [labels[1 + i % 2]]} for i in range(NUM_FRAMES)]
        expected_batch = Batch(frames=pd.DataFrame(expected))
        self.assertEqual(actual_batch, expected_batch)
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS dummy_table;')
        execute_query_fetch_all(self.evadb, create_query)

    @macos_skip_marker
    @pytest.mark.torchtest
    def test_should_create_table_from_select_lateral_join(self):
        select_query = 'SELECT id, label, bbox FROM UATRAC JOIN LATERAL Yolo(data) AS T(label, bbox, score) WHERE id < 5;'
        query = f'CREATE TABLE IF NOT EXISTS uadtrac_fastRCNN AS {select_query};'
        execute_query_fetch_all(self.evadb, query)
        select_view_query = 'SELECT id, label, bbox FROM uadtrac_fastRCNN'
        actual_batch = execute_query_fetch_all(self.evadb, select_view_query)
        actual_batch.sort()
        self.assertEqual(len(actual_batch), 5)
        res = actual_batch.frames
        for idx in res.index:
            self.assertTrue('car' in res['uadtrac_fastrcnn.label'][idx])
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS uadtrac_fastRCNN;')
        query = f'CREATE TABLE IF NOT EXISTS uadtrac_fastRCNN AS {select_query};'
        execute_query_fetch_all(self.evadb, query)

    def test_create_table_with_incorrect_info(self):
        create_table = 'CREATE TABLE SlackCSV(metadata TEXT(1000));'
        with self.assertRaises(Exception):
            execute_query_fetch_all(self.evadb, create_table)
        create_table = 'CREATE TABLE SlackCSV(user_profile TEXT(1000));'
        execute_query_fetch_all(self.evadb, create_table)
        execute_query_fetch_all(self.evadb, 'DROP TABLE SlackCSV;')

    def test_create_table_with_restricted_keywords(self):
        create_table = 'CREATE TABLE hello (_row_id INTEGER, price TEXT);'
        with self.assertRaises(AssertionError):
            execute_query_fetch_all(self.evadb, create_table)
        create_table = 'CREATE TABLE hello2 (_ROW_id INTEGER, price TEXT);'
        with self.assertRaises(AssertionError):
            execute_query_fetch_all(self.evadb, create_table)

@pytest.mark.notparallel
class FunctionExecutorTest(unittest.TestCase):

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        video_file_path = create_sample_video(NUM_FRAMES)
        load_query = f"LOAD VIDEO '{video_file_path}' INTO MyVideo;"
        execute_query_fetch_all(self.evadb, load_query)
        create_function_query = "CREATE FUNCTION DummyObjectDetector\n                  INPUT  (Frame_Array NDARRAY UINT8(3, 256, 256))\n                  OUTPUT (label NDARRAY STR(10))\n                  TYPE  Classification\n                  IMPL  'test/util.py';\n        "
        execute_query_fetch_all(self.evadb, create_function_query)

    def tearDown(self):
        shutdown_ray()
        file_remove('dummy.avi')
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyVideo;')

    def test_should_load_and_select_using_function_video_in_table(self):
        select_query = 'SELECT id,DummyObjectDetector(data) FROM MyVideo             ORDER BY id;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        labels = DummyObjectDetector().labels
        expected = [{'myvideo.id': i, 'dummyobjectdetector.label': np.array([labels[1 + i % 2]])} for i in range(NUM_FRAMES)]
        expected_batch = Batch(frames=pd.DataFrame(expected))
        self.assertEqual(actual_batch, expected_batch)

    def test_should_load_and_select_using_function_video(self):
        select_query = "SELECT id,DummyObjectDetector(data) FROM MyVideo             WHERE DummyObjectDetector(data).label = ['person'] ORDER BY id;"
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected = [{'myvideo.id': i * 2, 'dummyobjectdetector.label': np.array(['person'])} for i in range(NUM_FRAMES // 2)]
        expected_batch = Batch(frames=pd.DataFrame(expected))
        self.assertEqual(actual_batch, expected_batch)
        select_query = "SELECT id,DummyObjectDetector(data) FROM MyVideo             WHERE DummyObjectDetector(data).label @> ['person'] ORDER BY id;"
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(actual_batch, expected_batch)
        select_query = "SELECT id,DummyObjectDetector(data) FROM MyVideo             WHERE DummyObjectDetector(data).label <@ ['person', 'bicycle']             ORDER BY id;"
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected = [{'myvideo.id': i * 2, 'dummyobjectdetector.label': np.array(['person'])} for i in range(NUM_FRAMES // 2)]
        expected += [{'myvideo.id': i, 'dummyobjectdetector.label': np.array(['bicycle'])} for i in range(NUM_FRAMES) if i % 2 + 1 == 2]
        expected_batch = Batch(frames=pd.DataFrame(expected))
        expected_batch.sort()
        self.assertEqual(actual_batch, expected_batch)
        nested_select_query = "SELECT name, id, data FROM\n            (SELECT name, id, data, DummyObjectDetector(data) FROM MyVideo\n                WHERE id >= 2\n            ) AS T\n            WHERE ['person'] <@ label;\n            "
        actual_batch = execute_query_fetch_all(self.evadb, nested_select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches(filters=[i for i in range(2, NUM_FRAMES) if i % 2 == 0]))[0]
        expected_batch = expected_batch.project(['myvideo.name', 'myvideo.id', 'myvideo.data'])
        expected_batch.modify_column_alias('T')
        self.assertEqual(actual_batch, expected_batch)

    def test_create_function(self):
        function_name = 'DummyObjectDetector'
        create_function_query = "CREATE FUNCTION {}\n                  INPUT  (Frame_Array NDARRAY UINT8(3, 256, 256))\n                  OUTPUT (label NDARRAY STR(10))\n                  TYPE  Classification\n                  IMPL  'test/util.py';\n        "
        with self.assertRaises(ExecutorError):
            actual = execute_query_fetch_all(self.evadb, create_function_query.format(function_name))
            expected = Batch(pd.DataFrame([f'Function {function_name} already exists.']))
            self.assertEqual(actual, expected)
        actual = execute_query_fetch_all(self.evadb, create_function_query.format('IF NOT EXISTS ' + function_name))
        expected = Batch(pd.DataFrame([f'Function {function_name} already exists, nothing added.']))
        self.assertEqual(actual, expected)

    def test_create_or_replace(self):
        function_name = 'DummyObjectDetector'
        execute_query_fetch_all(self.evadb, f'DROP FUNCTION IF EXISTS {function_name};')
        create_function_query = "CREATE OR REPLACE FUNCTION {}\n                  INPUT  (Frame_Array NDARRAY UINT8(3, 256, 256))\n                  OUTPUT (label NDARRAY STR(10))\n                  TYPE  Classification\n                  IMPL  'test/util.py';\n        "
        actual = execute_query_fetch_all(self.evadb, create_function_query.format(function_name))
        expected = Batch(pd.DataFrame([f'Function {function_name} added to the database.']))
        self.assertEqual(actual, expected)
        actual = execute_query_fetch_all(self.evadb, create_function_query.format(function_name))
        expected = Batch(pd.DataFrame([f'Function {function_name} overwritten.']))
        self.assertEqual(actual, expected)

    def test_should_create_function_with_metadata(self):
        function_name = 'DummyObjectDetector'
        execute_query_fetch_all(self.evadb, f'DROP FUNCTION {function_name};')
        create_function_query = 'CREATE FUNCTION {}\n                  INPUT  (Frame_Array NDARRAY UINT8(3, 256, 256))\n                  OUTPUT (label NDARRAY STR(10))\n                  TYPE  Classification\n                  IMPL  \'test/util.py\'\n                  CACHE TRUE\n                  BATCH FALSE\n                  INT_VAL 1\n                  FLOAT_VAL 1.5\n                  STR_VAL "gg";\n        '
        execute_query_fetch_all(self.evadb, create_function_query.format(function_name))
        entries = self.evadb.catalog().get_function_metadata_entries_by_function_name(function_name)
        self.assertEqual(len(entries), 5)
        metadata = [(entry.key, entry.value) for entry in entries]
        expected_metadata = [('cache', True), ('batch', False), ('int_val', 1), ('float_val', 1.5), ('str_val', 'gg')]
        self.assertEqual(set(metadata), set(expected_metadata))

    def test_should_return_empty_metadata_list_for_missing_function(self):
        entries = self.evadb.catalog().get_function_metadata_entries_by_function_name('randomFunction')
        self.assertEqual(len(entries), 0)

    def test_should_return_empty_metadata_list_if_function_is_removed(self):
        function_name = 'DummyObjectDetector'
        execute_query_fetch_all(self.evadb, f'DROP FUNCTION {function_name};')
        create_function_query = "CREATE FUNCTION {}\n                  INPUT  (Frame_Array NDARRAY UINT8(3, 256, 256))\n                  OUTPUT (label NDARRAY STR(10))\n                  TYPE  Classification\n                  IMPL  'test/util.py'\n                  CACHE 'TRUE'\n                  BATCH 'FALSE';\n        "
        execute_query_fetch_all(self.evadb, create_function_query.format(function_name))
        entries = self.evadb.catalog().get_function_metadata_entries_by_function_name(function_name)
        self.assertEqual(len(entries), 2)
        execute_query_fetch_all(self.evadb, f'DROP FUNCTION {function_name};')
        entries = self.evadb.catalog().get_function_metadata_entries_by_function_name(function_name)
        self.assertEqual(len(entries), 0)

    def test_should_raise_using_missing_function(self):
        select_query = 'SELECT id,DummyObjectDetector1(data) FROM MyVideo             ORDER BY id;'
        with self.assertRaises(BinderError) as cm:
            execute_query_fetch_all(self.evadb, select_query, do_not_print_exceptions=True)
        err_msg = "Function 'DummyObjectDetector1' does not exist in the catalog. Please create the function using CREATE FUNCTION command."
        self.assertEqual(str(cm.exception), err_msg)

    def test_should_raise_for_function_name_mismatch(self):
        create_function_query = "CREATE FUNCTION TestFUNCTION\n                  INPUT  (Frame_Array NDARRAY UINT8(3, 256, 256))\n                  OUTPUT (label NDARRAY STR(10))\n                  TYPE  Classification\n                  IMPL  'test/util.py';\n        "
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, create_function_query, do_not_print_exceptions=True)

    def test_should_raise_if_function_file_is_modified(self):
        execute_query_fetch_all(self.evadb, 'DROP FUNCTION DummyObjectDetector;')
        execute_query_fetch_all(self.evadb, 'DROP FUNCTION IF EXISTS DummyObjectDetector;')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as tmp_file:
            with open('test/util.py', 'r') as file:
                tmp_file.write(file.read())
            tmp_file.seek(0)
            function_name = 'DummyObjectDetector'
            create_function_query = "CREATE FUNCTION {}\n                    INPUT  (Frame_Array NDARRAY UINT8(3, 256, 256))\n                    OUTPUT (label NDARRAY STR(10))\n                    TYPE  Classification\n                    IMPL  '{}';\n            "
            execute_query_fetch_all(self.evadb, create_function_query.format(function_name, tmp_file.name))
            tmp_file.seek(0, 2)
            tmp_file.write('#comment')
            tmp_file.seek(0)
            select_query = 'SELECT id,DummyObjectDetector(data) FROM MyVideo ORDER BY id;'
            execute_query_fetch_all(self.evadb, select_query)

    def test_create_function_with_decorators(self):
        execute_query_fetch_all(self.evadb, 'DROP FUNCTION IF EXISTS DummyObjectDetectorDecorators;')
        create_function_query = "CREATE FUNCTION DummyObjectDetectorDecorators\n                  IMPL  'test/util.py';\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        catalog_manager = self.evadb.catalog()
        function_obj = catalog_manager.get_function_catalog_entry_by_name('DummyObjectDetectorDecorators')
        function_inputs = catalog_manager.get_function_io_catalog_input_entries(function_obj)
        self.assertEquals(len(function_inputs), 1)
        function_input = function_inputs[0]
        expected_input_attributes = {'name': 'Frame_Array', 'type': ColumnType.NDARRAY, 'is_nullable': False, 'array_type': NdArrayType.UINT8, 'array_dimensions': (3, 256, 256), 'is_input': True}
        for attr in expected_input_attributes:
            self.assertEquals(getattr(function_input, attr), expected_input_attributes[attr])
        function_outputs = catalog_manager.get_function_io_catalog_output_entries(function_obj)
        self.assertEquals(len(function_outputs), 1)
        function_output = function_outputs[0]
        expected_output_attributes = {'name': 'label', 'type': ColumnType.NDARRAY, 'is_nullable': False, 'array_type': NdArrayType.STR, 'array_dimensions': (), 'is_input': False}
        for attr in expected_output_attributes:
            self.assertEquals(getattr(function_output, attr), expected_output_attributes[attr])

    def test_function_cost_entry_created(self):
        execute_query_fetch_all(self.evadb, 'SELECT DummyObjectDetector(data) FROM MyVideo')
        entry = self.evadb.catalog().get_function_cost_catalog_entry('DummyObjectDetector')
        self.assertIsNotNone(entry)

@pytest.mark.notparallel
class ArrayCountTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        video_file_path = create_sample_video(NUM_FRAMES)
        load_query = f"LOAD VIDEO '{video_file_path}' INTO MyVideo;"
        execute_query_fetch_all(cls.evadb, load_query)
        load_functions_for_testing(cls.evadb, mode='debug')

    @classmethod
    def tearDownClass(cls):
        shutdown_ray()
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS MyVideo;')
        file_remove('dummy.avi')

    def test_should_load_and_select_using_function_video(self):
        select_query = "SELECT id,DummyObjectDetector(data) FROM MyVideo             WHERE DummyObjectDetector(data).label = ['person'] ORDER BY id;"
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected = [{'myvideo.id': i * 2, 'dummyobjectdetector.label': ['person']} for i in range(NUM_FRAMES // 2)]
        expected_batch = Batch(frames=pd.DataFrame(expected))
        self.assertEqual(actual_batch, expected_batch)
        select_query = "SELECT id, DummyObjectDetector(data) FROM MyVideo             WHERE DummyObjectDetector(data).label <@ ['person'] ORDER BY id;"
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(actual_batch, expected_batch)
        select_query = "SELECT id FROM MyVideo WHERE             DummyMultiObjectDetector(data).labels @> ['person'] ORDER BY id;"
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected = [{'myvideo.id': i} for i in range(0, NUM_FRAMES, 3)]
        expected_batch = Batch(frames=pd.DataFrame(expected))
        self.assertEqual(actual_batch, expected_batch)

    def test_array_count_integration_test(self):
        select_query = "SELECT id FROM MyVideo WHERE\n            ArrayCount(DummyMultiObjectDetector(data).labels, 'person') = 2\n            ORDER BY id;"
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected = [{'myvideo.id': i} for i in range(0, NUM_FRAMES, 3)]
        expected_batch = Batch(frames=pd.DataFrame(expected))
        self.assertEqual(actual_batch, expected_batch)
        select_query = "SELECT id FROM MyVideo\n            WHERE ArrayCount(DummyObjectDetector(data).label, 'bicycle') = 1\n            ORDER BY id;"
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected = [{'myvideo.id': i} for i in range(1, NUM_FRAMES, 2)]
        expected_batch = Batch(frames=pd.DataFrame(expected))
        self.assertEqual(actual_batch, expected_batch)

@pytest.mark.notparallel
class S3LoadExecutorTest(unittest.TestCase):

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        self.video_file_path = create_sample_video()
        self.multiple_video_file_path = f'{EvaDB_ROOT_DIR}/data/sample_videos/1'
        self.s3_download_dir = self.evadb.catalog().get_configuration_catalog_value('s3_download_dir')
        'Mocked AWS Credentials for moto.'
        os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
        os.environ['AWS_SECURITY_TOKEN'] = 'testing'
        os.environ['AWS_SESSION_TOKEN'] = 'testing'
        os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
        try_to_import_moto()
        from moto import mock_s3
        self.mock_s3 = mock_s3()
        self.mock_s3.start()
        import boto3
        self.s3_client = boto3.client('s3')

    def upload_single_file(self, bucket_name='test-bucket'):
        self.s3_client.create_bucket(Bucket=bucket_name)
        self.s3_client.upload_file(self.video_file_path, bucket_name, 'dummy.avi')

    def upload_multiple_files(self, bucket_name='test-bucket'):
        self.s3_client.create_bucket(Bucket=bucket_name)
        video_path = self.multiple_video_file_path
        for file in os.listdir(video_path):
            self.s3_client.upload_file(f'{video_path}/{file}', bucket_name, file)

    def tearDown(self):
        shutdown_ray()
        file_remove('MyVideo/dummy.avi', parent_dir=self.s3_download_dir)
        for file in os.listdir(self.multiple_video_file_path):
            file_remove(f'MyVideos/{file}', parent_dir=self.s3_download_dir)
        self.mock_s3.stop()

    def test_s3_single_file_load_executor(self):
        bucket_name = 'single-file-bucket'
        self.upload_single_file(bucket_name)
        query = f"LOAD VIDEO 's3://{bucket_name}/dummy.avi' INTO MyVideo;"
        execute_query_fetch_all(self.evadb, query)
        select_query = 'SELECT * FROM MyVideo;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches(video_dir=os.path.join(self.s3_download_dir, 'MyVideo')))[0]
        self.assertEqual(actual_batch, expected_batch)
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyVideo;')

    def test_s3_multiple_file_load_executor(self):
        bucket_name = 'multiple-file-bucket'
        self.upload_multiple_files(bucket_name)
        query = f'LOAD VIDEO "s3://{bucket_name}/*.mp4" INTO MyVideos;'
        result = execute_query_fetch_all(self.evadb, query)
        expected = Batch(pd.DataFrame([f'Number of loaded {FileFormatType.VIDEO.name}: 2']))
        self.assertEqual(result, expected)
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyVideos;')

    def test_s3_multiple_file_multiple_load_executor(self):
        bucket_name = 'multiple-file-multiple-load-bucket'
        self.upload_single_file(bucket_name)
        self.upload_multiple_files(bucket_name)
        insert_query_one = f'LOAD VIDEO "s3://{bucket_name}/1.mp4" INTO MyVideos;'
        execute_query_fetch_all(self.evadb, insert_query_one)
        insert_query_two = f'LOAD VIDEO "s3://{bucket_name}/2.mp4" INTO MyVideos;'
        execute_query_fetch_all(self.evadb, insert_query_two)
        insert_query_three = f"LOAD VIDEO '{self.video_file_path}' INTO MyVideos;"
        execute_query_fetch_all(self.evadb, insert_query_three)
        select_query = 'SELECT * FROM MyVideos;'
        result = execute_query_fetch_all(self.evadb, select_query)
        result_videos = [Path(video).as_posix() for video in result.frames['myvideos.name'].unique()]
        s3_dir_path = Path(self.s3_download_dir)
        expected_videos = [(s3_dir_path / 'MyVideos/1.mp4').as_posix(), (s3_dir_path / 'MyVideos/2.mp4').as_posix(), Path(self.video_file_path).as_posix()]
        self.assertEqual(result_videos, expected_videos)
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyVideos;')

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

class RelationalAPI(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.db_dir = suffix_pytest_xdist_worker_id_to_dir(EvaDB_DATABASE_DIR)
        cls.conn = connect(cls.db_dir)
        cls.evadb = cls.conn._evadb

    def setUp(self):
        self.evadb.catalog().reset()
        self.mnist_path = f'{EvaDB_ROOT_DIR}/data/mnist/mnist.mp4'
        load_functions_for_testing(self.evadb)
        self.images = f'{EvaDB_ROOT_DIR}/data/detoxify/*.jpg'

    def tearDown(self):
        shutdown_ray()
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS mnist_video;')
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS meme_images;')
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS dummy_table;')

    def test_relation_apis(self):
        cursor = self.conn.cursor()
        rel = cursor.load(self.mnist_path, table_name='mnist_video', format='video')
        rel.execute()
        rel = cursor.table('mnist_video')
        assert_frame_equal(rel.df(), cursor.query('select * from mnist_video;').df())
        rel = rel.select('_row_id, id, data')
        assert_frame_equal(rel.df(), cursor.query('select _row_id, id, data from mnist_video;').df())
        rel = rel.filter('id < 10')
        assert_frame_equal(rel.df(), cursor.query('select _row_id, id, data from mnist_video where id < 10;').df())
        rel = rel.cross_apply('unnest(MnistImageClassifier(data))', 'mnist(label)').filter('mnist.label = 1').select('_row_id, id')
        query = ' select _row_id, id\n                    from mnist_video\n                        join lateral unnest(MnistImageClassifier(data)) AS mnist(label)\n                    where id < 10 AND mnist.label = 1;'
        assert_frame_equal(rel.df(), cursor.query(query).df())
        rel = cursor.load(self.images, table_name='meme_images', format='image')
        rel.execute()
        rel = cursor.table('meme_images').select('_row_id, name')
        assert_frame_equal(rel.df(), cursor.query('select _row_id, name from meme_images;').df())
        rel = rel.filter('_row_id < 3')
        assert_frame_equal(rel.df(), cursor.query('select _row_id, name from meme_images where _row_id < 3;').df())

    def test_relation_api_chaining(self):
        cursor = self.conn.cursor()
        rel = cursor.load(self.mnist_path, table_name='mnist_video', format='video')
        rel.execute()
        rel = cursor.table('mnist_video').select('id, data').filter('id > 10').filter('id < 20')
        assert_frame_equal(rel.df(), cursor.query('select id, data from mnist_video where id > 10 AND id < 20;').df())

    def test_interleaving_calls(self):
        cursor = self.conn.cursor()
        rel = cursor.load(self.mnist_path, table_name='mnist_video', format='video')
        rel.execute()
        rel = cursor.table('mnist_video')
        filtered_rel = rel.filter('id > 10')
        assert_frame_equal(rel.filter('id > 10').df(), cursor.query('select * from mnist_video where id > 10;').df())
        assert_frame_equal(filtered_rel.select('_row_id, id').df(), cursor.query('select _row_id, id from mnist_video where id > 10;').df())

    @qdrant_skip_marker
    def test_create_index(self):
        cursor = self.conn.cursor()
        rel = cursor.load(self.images, table_name='meme_images', format='image')
        rel.execute()
        cursor.query(f"CREATE FUNCTION IF NOT EXISTS SiftFeatureExtractor\n                IMPL  '{EvaDB_ROOT_DIR}/evadb/functions/sift_feature_extractor.py'").df()
        cursor.create_vector_index('faiss_index', table_name='meme_images', expr='SiftFeatureExtractor(data)', using='QDRANT').df()
        base_image = f'{EvaDB_ROOT_DIR}/data/detoxify/meme1.jpg'
        rel = cursor.table('meme_images').order(f"Similarity(SiftFeatureExtractor(Open('{base_image}')), SiftFeatureExtractor(data))").limit(1).select('name')
        similarity_sql = 'SELECT name FROM meme_images\n                            ORDER BY\n                                Similarity(SiftFeatureExtractor(Open("{}")), SiftFeatureExtractor(data))\n                            LIMIT 1;'.format(base_image)
        assert_frame_equal(rel.df(), cursor.query(similarity_sql).df())

    def test_create_function_with_relational_api(self):
        video_file_path = create_sample_video(10)
        cursor = self.conn.cursor()
        rel = cursor.load(video_file_path, table_name='dummy_video', format='video')
        rel.execute()
        create_dummy_object_detector_function = cursor.create_function('DummyObjectDetector', if_not_exists=True, impl_path='test/util.py')
        create_dummy_object_detector_function.execute()
        args = {'task': 'automatic-speech-recognition', 'model': 'openai/whisper-base'}
        create_speech_recognizer_function_if_not_exists = cursor.create_function('SpeechRecognizer', if_not_exists=True, type='HuggingFace', **args)
        query = create_speech_recognizer_function_if_not_exists.sql_query()
        self.assertEqual(query, "CREATE FUNCTION IF NOT EXISTS SpeechRecognizer TYPE HuggingFace TASK 'automatic-speech-recognition' MODEL 'openai/whisper-base'")
        create_speech_recognizer_function_if_not_exists.execute()
        create_speech_recognizer_function = cursor.create_function('SpeechRecognizer', if_not_exists=False, type='HuggingFace', **args)
        query = create_speech_recognizer_function.sql_query()
        self.assertEqual(query, "CREATE FUNCTION SpeechRecognizer TYPE HuggingFace TASK 'automatic-speech-recognition' MODEL 'openai/whisper-base'")
        with self.assertRaises(ExecutorError):
            create_speech_recognizer_function.execute()
        select_query_sql = 'SELECT id, DummyObjectDetector(data) FROM dummy_video ORDER BY id;'
        actual_batch = cursor.query(select_query_sql).execute()
        labels = DummyObjectDetector().labels
        expected = [{'id': i, 'label': np.array([labels[1 + i % 2]])} for i in range(10)]
        expected_batch = Batch(frames=pd.DataFrame(expected))
        self.assertEqual(actual_batch, expected_batch)
        actual_batch = cursor.query(select_query_sql).execute(drop_alias=False)
        expected = [{'dummy_video.id': i, 'dummyobjectdetector.label': np.array([labels[1 + i % 2]])} for i in range(10)]
        expected_batch = Batch(frames=pd.DataFrame(expected))
        self.assertEqual(actual_batch, expected_batch)

    def test_drop_with_relational_api(self):
        video_file_path = create_sample_video(10)
        cursor = self.conn.cursor()
        rel = cursor.load(video_file_path, table_name='dummy_video', format='video')
        rel.execute()
        create_dummy_object_detector_function = cursor.create_function('DummyObjectDetector', if_not_exists=True, impl_path='test/util.py')
        create_dummy_object_detector_function.execute()
        drop_dummy_object_detector_function = cursor.drop_function('DummyObjectDetector', if_exists=True)
        drop_dummy_object_detector_function.execute()
        select_query_sql = 'SELECT id, DummyObjectDetector(data) FROM dummy_video ORDER BY id;'
        with self.assertRaises(BinderError):
            cursor.query(select_query_sql).execute()
        drop_dummy_object_detector_function = cursor.drop_function('DummyObjectDetector', if_exists=True)
        drop_dummy_object_detector_function.execute()
        drop_dummy_object_detector_function = cursor.drop_function('DummyObjectDetector', if_exists=False)
        with self.assertRaises(ExecutorError):
            drop_dummy_object_detector_function.execute()
        drop_table = cursor.drop_table('dummy_video', if_exists=True)
        drop_table.execute()
        select_query_sql = 'SELECT id, data FROM dummy_video ORDER BY id;'
        with self.assertRaises(BinderError):
            cursor.query(select_query_sql).execute()
        drop_table = cursor.drop_table('dummy_video', if_exists=True)
        drop_table.execute()
        drop_table = cursor.drop_table('dummy_video', if_exists=False)
        with self.assertRaises(ExecutorError):
            drop_table.execute()

    def test_pdf_similarity_search(self):
        conn = connect()
        cursor = conn.cursor()
        pdf_path = f'{EvaDB_ROOT_DIR}/data/documents/state_of_the_union.pdf'
        load_pdf = cursor.load(file_regex=pdf_path, format='PDF', table_name='PDFs')
        load_pdf.execute()
        function_check = cursor.drop_function('SentenceFeatureExtractor')
        function_check.df()
        function = cursor.create_function('SentenceFeatureExtractor', True, f'{EvaDB_ROOT_DIR}/evadb/functions/sentence_feature_extractor.py')
        function.execute()
        cursor.create_vector_index('faiss_index', table_name='PDFs', expr='SentenceFeatureExtractor(data)', using='FAISS').df()
        query = cursor.table('PDFs').order("Similarity(\n                    SentenceFeatureExtractor('When was the NATO created?'), SentenceFeatureExtractor(data)\n                ) DESC").limit(3).select('data')
        output = query.df()
        self.assertEqual(len(output), 3)
        self.assertTrue('data' in output.columns)
        output = query.df(drop_alias=False)
        self.assertTrue('pdfs.data' in output.columns)
        cursor.drop_index('faiss_index').df()

    def test_langchain_split_doc(self):
        conn = connect()
        cursor = conn.cursor()
        pdf_path1 = f'{EvaDB_ROOT_DIR}/data/documents/state_of_the_union.pdf'
        load_pdf = cursor.load(file_regex=pdf_path1, format='DOCUMENT', table_name='docs')
        load_pdf.execute()
        result1 = cursor.table('docs', chunk_size=2000, chunk_overlap=DEFAULT_DOCUMENT_CHUNK_OVERLAP).select('data').df()
        result2 = cursor.table('docs', chunk_size=DEFAULT_DOCUMENT_CHUNK_SIZE, chunk_overlap=2000).select('data').df()
        result3 = cursor.table('docs', chunk_size=DEFAULT_DOCUMENT_CHUNK_SIZE, chunk_overlap=0).select('data').df()
        self.assertGreater(len(result1), len(result2))
        self.assertGreater(len(result2), len(result3))
        result5 = cursor.table('docs', chunk_size=2000).select('data').df()
        self.assertEqual(len(result5), len(result1))
        result4 = cursor.table('docs', chunk_overlap=0).select('data').df()
        self.assertEqual(len(result3), len(result4))
        result1 = cursor.table('docs').select('data').df()
        result2 = cursor.query(f'SELECT data from docs chunk_size {DEFAULT_DOCUMENT_CHUNK_SIZE} chunk_overlap {DEFAULT_DOCUMENT_CHUNK_OVERLAP}').df()
        self.assertEqual(len(result1), len(result2))

    def test_show_relational(self):
        video_file_path = create_sample_video(10)
        cursor = self.conn.cursor()
        rel = cursor.load(video_file_path, table_name='dummy_video', format='video')
        rel.execute()
        result = cursor.show('tables').df()
        self.assertEqual(len(result), 1)
        self.assertEqual(result['name'][0], 'dummy_video')

    def test_explain_relational(self):
        video_file_path = create_sample_video(10)
        cursor = self.conn.cursor()
        rel = cursor.load(video_file_path, table_name='dummy_video', format='video')
        rel.execute()
        result = cursor.explain('SELECT * FROM dummy_video').df()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], '|__ ProjectPlan\n    |__ SeqScanPlan\n        |__ StoragePlan\n')

    def test_rename_relational(self):
        video_file_path = create_sample_video(10)
        cursor = self.conn.cursor()
        rel = cursor.load(video_file_path, table_name='dummy_video', format='video')
        rel.execute()
        cursor.rename('dummy_video', 'dummy_video_renamed').df()
        result = cursor.show('tables').df()
        self.assertEqual(len(result), 1)
        self.assertEqual(result['name'][0], 'dummy_video_renamed')

    def test_create_table_relational(self):
        cursor = self.conn.cursor()
        cursor.create_table(table_name='dummy_table', if_not_exists=True, columns='id INTEGER, name text(30)').df()
        result = cursor.show('tables').df()
        self.assertEqual(len(result), 1)
        self.assertEqual(result['name'][0], 'dummy_table')
        rel = cursor.create_table(table_name='dummy_table', if_not_exists=False, columns='id INTEGER, name text(30)')
        with self.assertRaises(ExecutorError):
            rel.execute()

@pytest.mark.notparallel
class LoadExecutorTests(unittest.TestCase):

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        self.video_file_path = create_sample_video()
        self.image_files_path = Path(f'{EvaDB_ROOT_DIR}/test/data/uadetrac/small-data/MVI_20011/*.jpg')
        self.csv_file_path = create_sample_csv()

    def tearDown(self):
        shutdown_ray()
        file_remove('dummy.avi')
        file_remove('dummy.csv')
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyVideos;')

    def test_should_load_videos_in_table(self):
        path = f'{EvaDB_ROOT_DIR}/data/sample_videos/1/*.mp4'
        query = f'LOAD VIDEO "{path}" INTO MyVideos;'
        result = execute_query_fetch_all(self.evadb, query)
        expected = Batch(pd.DataFrame([f'Number of loaded {FileFormatType.VIDEO.name}: 2']))
        self.assertEqual(result, expected)

    def test_should_load_images_in_table(self):
        num_files = len(glob.glob(os.path.expanduser(self.image_files_path), recursive=True))
        query = f'LOAD IMAGE "{self.image_files_path}" INTO MyImages;'
        result = execute_query_fetch_all(self.evadb, query)
        expected = Batch(pd.DataFrame([f'Number of loaded {FileFormatType.IMAGE.name}: {num_files}']))
        self.assertEqual(result, expected)

    def test_should_load_csv_in_table(self):
        create_table_query = '\n\n            CREATE TABLE IF NOT EXISTS MyVideoCSV (\n                id INTEGER UNIQUE,\n                frame_id INTEGER,\n                video_id INTEGER,\n                dataset_name TEXT(30),\n                label TEXT(30),\n                bbox NDARRAY FLOAT32(4),\n                object_id INTEGER\n            );\n\n            '
        execute_query_fetch_all(self.evadb, create_table_query)
        load_query = f"LOAD CSV '{self.csv_file_path}' INTO MyVideoCSV;"
        execute_query_fetch_all(self.evadb, load_query)
        select_query = 'SELECT id, frame_id, video_id,\n                          dataset_name, label, bbox,\n                          object_id\n                          FROM MyVideoCSV;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = next(create_dummy_csv_batches())
        expected_batch.modify_column_alias('myvideocsv')
        self.assertEqual(actual_batch, expected_batch)
        drop_query = 'DROP TABLE IF EXISTS MyVideoCSV;'
        execute_query_fetch_all(self.evadb, drop_query)

    def test_should_load_csv_in_table_with_spaces_in_column_name(self):
        create_table_query = '\n\n            CREATE TABLE IF NOT EXISTS MyVideoCSV (\n                id INTEGER UNIQUE,\n                `frame id` INTEGER,\n                `video id` INTEGER,\n                `dataset name` TEXT(30),\n                label TEXT(30),\n                bbox NDARRAY FLOAT32(4),\n                `object id` INTEGER\n            );\n\n            '
        execute_query_fetch_all(self.evadb, create_table_query)
        load_query = f"LOAD CSV '{create_csv_with_comlumn_name_spaces()}' INTO MyVideoCSV;"
        execute_query_fetch_all(self.evadb, load_query)
        select_query = 'SELECT id, `frame id`, `video id`,\n                          `dataset name`, label, bbox,\n                          `object id`\n                          FROM MyVideoCSV;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = next(create_dummy_csv_batches())
        expected_batch.modify_column_alias('myvideocsv')
        self.assertEqual(actual_batch, expected_batch)
        drop_query = 'DROP TABLE IF EXISTS MyVideoCSV;'
        execute_query_fetch_all(self.evadb, drop_query)

@pytest.mark.notparallel
class ShowExecutorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        queries = [Fastrcnn_function_query, ArrayCount_function_query]
        for query in queries:
            execute_query_fetch_all(cls.evadb, query)
        ua_detrac = f'{EvaDB_ROOT_DIR}/data/ua_detrac/ua_detrac.mp4'
        mnist = f'{EvaDB_ROOT_DIR}/data/mnist/mnist.mp4'
        actions = f'{EvaDB_ROOT_DIR}/data/actions/actions.mp4'
        execute_query_fetch_all(cls.evadb, f"LOAD VIDEO '{ua_detrac}' INTO MyVideo;")
        execute_query_fetch_all(cls.evadb, f"LOAD VIDEO '{mnist}' INTO MNIST;")
        execute_query_fetch_all(cls.evadb, f"LOAD VIDEO '{actions}' INTO Actions;")
        import os
        cls.current_file_dir = os.path.dirname(os.path.abspath(__file__))
        for i in range(NUM_DATABASES):
            database_path = f'{cls.current_file_dir}/testing_{i}.db'
            params = {'database': database_path}
            query = 'CREATE DATABASE test_data_source_{}\n                        WITH ENGINE = "sqlite",\n                        PARAMETERS = {};'.format(i, params)
            execute_query_fetch_all(cls.evadb, query)

    @classmethod
    def tearDownClass(cls):
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS Actions;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS MNIST;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS MyVideo;')
        for i in range(NUM_DATABASES):
            execute_query_fetch_all(cls.evadb, f'DROP DATABASE IF EXISTS test_data_source_{i};')
            database_path = f'{cls.current_file_dir}/testing_{i}.db'
            import contextlib
            with contextlib.suppress(FileNotFoundError):
                os.remove(database_path)

    def test_show_functions(self):
        result = execute_query_fetch_all(self.evadb, 'SHOW FUNCTIONS;')
        self.assertEqual(len(result.columns), 6)
        expected = {'name': ['FastRCNNObjectDetector', 'ArrayCount'], 'type': ['Classification', 'NdarrayFunction']}
        expected_df = pd.DataFrame(expected)
        self.assertTrue(all(expected_df.name == result.frames.name))
        self.assertTrue(all(expected_df.type == result.frames.type))

    @windows_skip_marker
    def test_show_tables(self):
        result = execute_query_fetch_all(self.evadb, 'SHOW TABLES;')
        self.assertEqual(len(result), 3)
        expected = {'name': ['MyVideo', 'MNIST', 'Actions']}
        expected_df = pd.DataFrame(expected)
        self.assertEqual(result, Batch(expected_df))
        os.system('nohup evadb_server --stop')
        os.system('nohup evadb_server --start &')
        result = execute_query_fetch_all(self.evadb, 'SHOW TABLES;')
        self.assertEqual(len(result), 3)
        expected = {'name': ['MyVideo', 'MNIST', 'Actions']}
        expected_df = pd.DataFrame(expected)
        self.assertEqual(result, Batch(expected_df))
        os.system('nohup evadb_server --stop')

    def test_show_config_execution(self):
        execute_query_fetch_all(self.evadb, "SET OPENAIKEY = 'ABCD';")
        expected_output = Batch(pd.DataFrame({'OPENAIKEY': ['ABCD']}))
        show_config_value = execute_query_fetch_all(self.evadb, 'SHOW OPENAIKEY')
        self.assertEqual(show_config_value, expected_output)
        with self.assertRaises(Exception):
            execute_query_fetch_all(self.evadb, 'SHOW BADCONFIG')

    def test_show_all_configs(self):
        show_all_config_value = execute_query_fetch_all(self.evadb, 'SHOW CONFIGS')
        columns = show_all_config_value.columns
        self.assertEqual(columns == list(BASE_EVADB_CONFIG.keys()), True)

    def test_show_databases(self):
        result = execute_query_fetch_all(self.evadb, 'SHOW DATABASES;')
        self.assertEqual(len(result.columns), 3)
        self.assertEqual(len(result), 6)
        expected = {'name': [f'test_data_source_{i}' for i in range(NUM_DATABASES)], 'engine': ['sqlite' for _ in range(NUM_DATABASES)]}
        expected_df = pd.DataFrame(expected)
        self.assertTrue(all(expected_df.name == result.frames.name))
        self.assertTrue(all(expected_df.engine == result.frames.engine))

@pytest.mark.notparallel
class SelectExecutorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        video_file_path = create_sample_video(NUM_FRAMES)
        load_query = f"LOAD VIDEO '{video_file_path}' INTO MyVideo;"
        execute_query_fetch_all(cls.evadb, load_query)
        load_functions_for_testing(cls.evadb)
        cls.table1 = create_table(cls.evadb, 'table1', 100, 3)
        cls.table2 = create_table(cls.evadb, 'table2', 500, 3)
        cls.table3 = create_table(cls.evadb, 'table3', 1000, 3)

    @classmethod
    def tearDownClass(cls):
        shutdown_ray()
        file_remove('dummy.avi')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS table1;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS table2;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS table3;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS MyVideo;')

    def test_should_load_and_sort_in_table(self):
        select_query = 'SELECT data, id FROM MyVideo ORDER BY id;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected_rows = [{'myvideo.id': i, 'myvideo.data': np.array(np.ones((32, 32, 3)) * i, dtype=np.uint8)} for i in range(NUM_FRAMES)]
        expected_batch = Batch(frames=pd.DataFrame(expected_rows))
        self.assertEqual(actual_batch, expected_batch)
        select_query = 'SELECT data, id FROM MyVideo ORDER BY id DESC;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected_batch.reverse()
        self.assertEqual(actual_batch, expected_batch)

    def test_should_load_and_select_in_table(self):
        select_query = 'SELECT id FROM MyVideo;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_rows = [{'myvideo.id': i} for i in range(NUM_FRAMES)]
        expected_batch = Batch(frames=pd.DataFrame(expected_rows))
        self.assertEqual(actual_batch, expected_batch)
        select_query = 'SELECT * FROM MyVideo;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches())
        self.assertEqual([actual_batch], expected_batch)

    def test_should_raise_binder_error_on_native_datasource(self):
        select_query = 'SELECT * FROM test.MyVideo'
        self.assertRaises(BinderError, execute_query_fetch_all, self.evadb, select_query)

    def test_should_raise_binder_error_on_non_existent_column(self):
        select_query = 'SELECT b1 FROM table1;'
        with self.assertRaises(BinderError) as ctx:
            execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual("Cannot find column b1. Did you mean a1? The feasible columns are ['_row_id', 'a0', 'a1', 'a2'].", str(ctx.exception))

    def test_should_select_star_in_nested_query(self):
        select_query = 'SELECT * FROM (SELECT * FROM MyVideo) AS T;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches())[0]
        expected_batch.modify_column_alias('T')
        self.assertEqual(actual_batch, expected_batch)
        select_query = 'SELECT * FROM (SELECT id FROM MyVideo) AS T;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_rows = [{'T.id': i} for i in range(NUM_FRAMES)]
        expected_batch = Batch(frames=pd.DataFrame(expected_rows))
        self.assertEqual(actual_batch, expected_batch)

    @unittest.skip('Not supported in current version')
    def test_select_star_in_lateral_join(self):
        select_query = 'SELECT * FROM MyVideo JOIN LATERAL\n                          Yolo(data);'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(actual_batch.frames.columns, ['myvideo.id'])

    def test_should_throw_error_when_both_audio_and_video_selected(self):
        query = "LOAD VIDEO 'data/sample_videos/touchdown.mp4'\n                   INTO TOUCHDOWN1;"
        execute_query_fetch_all(self.evadb, query)
        select_query = 'SELECT id, audio, data FROM TOUCHDOWN1;'
        try:
            execute_query_fetch_all(self.evadb, select_query)
            self.fail("Didn't raise AssertionError")
        except AssertionError as e:
            self.assertEquals('Cannot query over both audio and video streams', e.args[0])

    def test_select_and_limit(self):
        select_query = 'SELECT * FROM MyVideo ORDER BY id LIMIT 5;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches(num_frames=10, batch_size=5))
        self.assertEqual(len(actual_batch), len(expected_batch[0]))
        self.assertEqual(actual_batch, expected_batch[0])

    def test_select_and_aggregate(self):
        simple_aggregate_query = 'SELECT COUNT(*), AVG(id) FROM MyVideo;'
        actual_batch = execute_query_fetch_all(self.evadb, simple_aggregate_query)
        self.assertEqual(actual_batch.frames.iat[0, 0], 10)
        self.assertEqual(actual_batch.frames.iat[0, 1], 4.5)

    def test_select_and_iframe_sample(self):
        select_query = 'SELECT id FROM MyVideo SAMPLE IFRAMES 7 ORDER BY id;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches(filters=range(0, NUM_FRAMES, 7)))
        expected_batch[0] = expected_batch[0].project(['myvideo.id'])
        self.assertEqual(len(actual_batch), len(expected_batch[0]))
        self.assertEqual(actual_batch, expected_batch[0])

    def test_select_and_iframe_sample_without_sampling_rate(self):
        select_query = 'SELECT id FROM MyVideo SAMPLE IFRAMES ORDER BY id;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected_batch = list(create_dummy_batches(filters=range(0, NUM_FRAMES, 1)))
        expected_batch[0] = expected_batch[0].project(['myvideo.id'])
        self.assertEqual(len(actual_batch), len(expected_batch[0]))
        self.assertEqual(actual_batch, expected_batch[0])

    def test_select_and_groupby_first(self):
        segment_size = 3
        select_query = "SELECT FIRST(id), SEGMENT(data) FROM MyVideo GROUP BY '{} frames';".format(segment_size)
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        ids = np.arange(NUM_FRAMES)
        segments = [ids[i:i + segment_size] for i in range(0, len(ids), segment_size)]
        segments = [i for i in segments if len(i) == segment_size]
        expected_batch = list(create_dummy_4d_batches(filters=segments))[0]
        self.assertEqual(len(actual_batch), len(expected_batch))
        expected_batch.rename(columns={'myvideo.id': 'FIRST.id', 'myvideo.data': 'SEGMENT.data'})
        self.assertEqual(actual_batch, expected_batch.project(['FIRST.id', 'SEGMENT.data']))

    def test_select_and_groupby_with_last(self):
        segment_size = 3
        select_query = "SELECT LAST(id), SEGMENT(data) FROM MyVideo GROUP BY '{}frames';".format(segment_size)
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        ids = np.arange(NUM_FRAMES)
        segments = [ids[i:i + segment_size] for i in range(0, len(ids), segment_size)]
        segments = [i for i in segments if len(i) == segment_size]
        expected_batch = list(create_dummy_4d_batches(filters=segments, start_id=segment_size - 1))[0]
        self.assertEqual(len(actual_batch), len(expected_batch))
        expected_batch.rename(columns={'myvideo.id': 'LAST.id', 'myvideo.data': 'SEGMENT.data'})
        self.assertEqual(actual_batch, expected_batch.project(['LAST.id', 'SEGMENT.data']))

    def test_select_and_groupby_should_fail_with_incorrect_pattern(self):
        segment_size = '4a'
        select_query = "SELECT FIRST(id), SEGMENT(data) FROM MyVideo GROUP BY '{} frames';".format(segment_size)
        self.assertRaises(BinderError, execute_query_fetch_all, self.evadb, select_query)

    def test_select_and_groupby_should_fail_with_seconds(self):
        segment_size = 4
        select_query = "SELECT FIRST(id), SEGMENT(data) FROM MyVideo GROUP BY '{} seconds';".format(segment_size)
        self.assertRaises(BinderError, execute_query_fetch_all, self.evadb, select_query)

    def test_select_and_groupby_should_fail_with_non_video_table(self):
        segment_size = 4
        select_query = "SELECT FIRST(a1) FROM table1 GROUP BY '{} frames';".format(segment_size)
        self.assertRaises(BinderError, execute_query_fetch_all, self.evadb, select_query)

    def test_select_and_groupby_with_sample(self):
        segment_size = 2
        sampling_rate = 2
        select_query = "SELECT FIRST(id), SEGMENT(data) FROM MyVideo SAMPLE {} GROUP BY '{} frames';".format(sampling_rate, segment_size)
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        ids = np.arange(0, NUM_FRAMES, sampling_rate)
        segments = [ids[i:i + segment_size] for i in range(0, len(ids), segment_size)]
        segments = [i for i in segments if len(i) == segment_size]
        expected_batch = list(create_dummy_4d_batches(filters=segments))[0]
        self.assertEqual(len(actual_batch), len(expected_batch))
        expected_batch.rename(columns={'myvideo.id': 'FIRST.id', 'myvideo.data': 'SEGMENT.data'})
        self.assertEqual(actual_batch, expected_batch.project(['FIRST.id', 'SEGMENT.data']))

    def test_select_and_groupby_and_aggregate_with_pdf(self):
        GROUPBY_SIZE = 8
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyPDFs;')
        pdf_path = 'test/data/uadetrac/small-data/pdf_data/fall_2023_orientation_document.pdf'
        load_query = f"LOAD PDF '{pdf_path}' INTO MyPDFs;"
        execute_query_fetch_all(self.evadb, load_query)
        select_all_query = 'SELECT * FROM MyPDFs;'
        all_pdf_batch = execute_query_fetch_all(self.evadb, select_all_query)
        select_query = f"SELECT COUNT(*) FROM MyPDFs GROUP BY '{GROUPBY_SIZE} paragraphs';"
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertAlmostEqual(len(all_pdf_batch), len(actual_batch) * actual_batch.frames.iloc[0, 0], None, None, GROUPBY_SIZE)
        self.assertEqual(len(actual_batch), 99)
        n = len(actual_batch)
        for i in range(n):
            self.assertEqual(actual_batch.frames.iloc[i, 0], GROUPBY_SIZE)
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyPDFs;')

    def test_lateral_join_with_unnest_and_sample(self):
        query = 'SELECT id, label\n                  FROM MyVideo SAMPLE 2 JOIN LATERAL\n                    UNNEST(DummyMultiObjectDetector(data).labels) AS T(label)\n                  WHERE id < 10 ORDER BY id;'
        unnest_batch = execute_query_fetch_all(self.evadb, query)
        expected = Batch(pd.DataFrame({'myvideo.id': np.array([0, 0, 2, 2, 4, 4, 6, 6, 8, 8], dtype=np.intp), 'T.label': np.array(['person', 'person', 'car', 'car', 'bicycle', 'bicycle', 'person', 'person', 'car', 'car'])}))
        self.assertEqual(len(unnest_batch), 10)
        self.assertEqual(unnest_batch, expected)

    def test_select_without_from(self):
        query = 'SELECT 1;'
        batch = execute_query_fetch_all(self.evadb, query)
        expected = Batch(pd.DataFrame([{0: 1}]))
        self.assertEqual(batch, expected)
        query = 'SELECT 1>2;'
        batch = execute_query_fetch_all(self.evadb, query)
        expected = Batch(pd.DataFrame([{0: False}]))
        self.assertEqual(batch, expected)

    def test_should_raise_error_with_missing_alias_in_lateral_join(self):
        function_name = 'DummyMultiObjectDetector'
        query = 'SELECT id, labels\n                  FROM MyVideo JOIN LATERAL DummyMultiObjectDetector(data).labels;'
        with self.assertRaises(SyntaxError) as cm:
            execute_query_fetch_all(self.evadb, query, do_not_print_exceptions=True)
        self.assertEqual(str(cm.exception), f'TableValuedFunction {function_name} should have alias.')
        query = 'SELECT id, labels\n                  FROM MyVideo JOIN LATERAL\n                    UNNEST(DummyMultiObjectDetector(data).labels);'
        with self.assertRaises(SyntaxError) as cm:
            execute_query_fetch_all(self.evadb, query)
        self.assertEqual(str(cm.exception), f'TableValuedFunction {function_name} should have alias.')
        query = 'SELECT id, labels\n                  FROM MyVideo JOIN LATERAL DummyMultiObjectDetector(data);'
        with self.assertRaises(SyntaxError) as cm:
            execute_query_fetch_all(self.evadb, query)
        self.assertEqual(str(cm.exception), f'TableValuedFunction {function_name} should have alias.')

    def test_should_raise_error_with_invalid_number_of_aliases(self):
        function_name = 'DummyMultiObjectDetector'
        query = 'SELECT id, labels\n                  FROM MyVideo JOIN LATERAL\n                    DummyMultiObjectDetector(data).bboxes AS T;'
        with self.assertRaises(BinderError) as cm:
            execute_query_fetch_all(self.evadb, query)
        self.assertEqual(str(cm.exception), f'Output bboxes does not exist for {function_name}.')

    def test_should_raise_error_with_invalid_output_lateral_join(self):
        query = 'SELECT id, a\n                  FROM MyVideo JOIN LATERAL\n                    DummyMultiObjectDetector(data) AS T(a, b);\n                '
        with self.assertRaises(AssertionError) as cm:
            execute_query_fetch_all(self.evadb, query)
        self.assertEqual(str(cm.exception), 'Expected 1 output columns for T, got 2.')

    def test_hash_join_with_one_on(self):
        select_query = 'SELECT * FROM table1 JOIN\n                        table2 ON table1.a1 = table2.a1;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected = pd.merge(self.table1, self.table2, left_on=['table1.a1'], right_on=['table2.a1'], how='inner')
        if len(expected):
            expected_batch = Batch(expected)
            self.assertEqual(expected_batch.sort_orderby(['table1.a2']), actual_batch.sort_orderby(['table1.a2']))

    def test_hash_join_with_multiple_on(self):
        select_query = 'SELECT * FROM table1 JOIN\n                        table1 AS table2 ON table1.a1 = table2.a1 AND\n                        table1.a0 = table2.a0;'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected = pd.merge(self.table1, self.table1, left_on=['table1.a1', 'table1.a0'], right_on=['table1.a1', 'table1.a0'], how='inner')
        if len(expected):
            expected_batch = Batch(expected)
            self.assertEqual(expected_batch.sort_orderby(['table1.a1']), actual_batch.sort_orderby(['table1.a1']))

    def test_expression_tree_signature(self):
        plan = get_logical_query_plan(self.evadb, "SELECT id FROM MyVideo WHERE DummyMultiObjectDetector(data).labels @> ['person'];")
        signature = next(plan.find_all(LogicalFilter)).predicate.children[0].signature()
        function_id = self.evadb.catalog().get_function_catalog_entry_by_name('DummyMultiObjectDetector').row_id
        table_entry = self.evadb.catalog().get_table_catalog_entry('MyVideo')
        col_id = self.evadb.catalog().get_column_catalog_entry(table_entry, 'data').row_id
        self.assertEqual(signature, f'DummyMultiObjectDetector[{function_id}](MyVideo.data[{col_id}])')

    def test_function_with_no_input_arguments(self):
        select_query = 'SELECT DummyNoInputFunction();'
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        expected = Batch(pd.DataFrame([{'dummynoinputfunction.label': 'DummyNoInputFunction'}]))
        self.assertEqual(actual_batch, expected)

@pytest.mark.notparallel
class CascadeOptimizer(unittest.TestCase):

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        self.video_file_path = create_sample_video(NUM_FRAMES)

    def tearDown(self):
        shutdown_ray()
        file_remove('dummy.avi')

    def test_logical_to_physical_function(self):
        load_query = f"LOAD VIDEO '{self.video_file_path}' INTO MyVideo;"
        execute_query_fetch_all(self.evadb, load_query)
        create_function_query = "CREATE FUNCTION DummyObjectDetector\n                  INPUT  (Frame_Array NDARRAY UINT8(3, 256, 256))\n                  OUTPUT (label NDARRAY STR(10))\n                  TYPE  Classification\n                  IMPL  'test/util.py';\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        select_query = "SELECT id, DummyObjectDetector(data)\n                    FROM MyVideo\n                    WHERE DummyObjectDetector(data).label = ['person'];\n        "
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_batch.sort()
        expected = [{'myvideo.id': i * 2, 'dummyobjectdetector.label': ['person']} for i in range(NUM_FRAMES // 2)]
        expected_batch = Batch(frames=pd.DataFrame(expected))
        self.assertEqual(actual_batch, expected_batch)
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyVideo;')

class OrderByExecutorTest(unittest.TestCase):

    def test_should_return_sorted_frames(self):
        """
        data (3 batches):
        'A' 'B' 'C'
        [1, 1, 1]
        ----------
        [1, 5, 6]
        [4, 7, 10]
        ----------
        [2, 9, 7]
        [4, 1, 2]
        [4, 2, 4]
        """
        df1 = pd.DataFrame(np.array([[1, 1, 1]]), columns=['A', 'B', 'C'])
        df2 = pd.DataFrame(np.array([[1, 5, 6], [4, 7, 10]]), columns=['A', 'B', 'C'])
        df3 = pd.DataFrame(np.array([[2, 9, 7], [4, 1, 2], [4, 2, 4]]), columns=['A', 'B', 'C'])
        batches = [Batch(frames=df) for df in [df1, df2, df3]]
        'query: .... ORDER BY A ASC, B DESC '
        plan = OrderByPlan([(TupleValueExpression(col_alias='A'), ParserOrderBySortType.ASC), (TupleValueExpression(col_alias='B'), ParserOrderBySortType.DESC)])
        orderby_executor = OrderByExecutor(MagicMock(), plan)
        orderby_executor.append_child(DummyExecutor(batches))
        sorted_batches = list(orderby_executor.exec())
        '\n           A  B   C\n        0  1  5   6\n        1  1  1   1\n        2  2  9   7\n        3  4  7  10\n        4  4  2   4\n        5  4  1   2\n        '
        expected_df1 = pd.DataFrame(np.array([[1, 5, 6]]), columns=['A', 'B', 'C'])
        expected_df2 = pd.DataFrame(np.array([[1, 1, 1], [2, 9, 7]]), columns=['A', 'B', 'C'])
        expected_df3 = pd.DataFrame(np.array([[4, 7, 10], [4, 2, 4], [4, 1, 2]]), columns=['A', 'B', 'C'])
        expected_batches = [Batch(frames=df) for df in [expected_df1, expected_df2, expected_df3]]
        self.assertEqual(expected_batches[0], sorted_batches[0])
        self.assertEqual(expected_batches[1], sorted_batches[1])
        self.assertEqual(expected_batches[2], sorted_batches[2])

class LimitExecutorTest(unittest.TestCase):

    def test_should_return_smaller_num_rows(self):
        dfs = [pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD')) for _ in range(4)]
        batches = [Batch(frames=df) for df in dfs]
        limit_value = 125
        plan = LimitPlan(ConstantValueExpression(limit_value))
        limit_executor = LimitExecutor(MagicMock(), plan)
        limit_executor.append_child(DummyExecutor(batches))
        reduced_batches = list(limit_executor.exec())
        total_size = 0
        for batch in reduced_batches:
            total_size += len(batch)
        self.assertEqual(total_size, limit_value)

    def test_should_return_limit_greater_than_size(self):
        """This should return the exact same data
        if the limit value is greater than what is present.
        This will also leave a warning"""
        dfs = [pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD')) for _ in range(4)]
        batches = [Batch(frames=df) for df in dfs]
        previous_total_size = 0
        for batch in batches:
            previous_total_size += len(batch)
        limit_value = 500
        plan = LimitPlan(ConstantValueExpression(limit_value))
        limit_executor = LimitExecutor(MagicMock(), plan)
        limit_executor.append_child(DummyExecutor(batches))
        reduced_batches = list(limit_executor.exec())
        after_total_size = 0
        for batch in reduced_batches:
            after_total_size += len(batch)
        self.assertEqual(previous_total_size, after_total_size)

    def test_should_return_top_frames_after_sorting(self):
        """
        Checks if limit returns the top 2 rows from the data
        after sorting

        data (3 batches):
        'A' 'B' 'C'
        [1, 1, 1]
        ----------
        [1, 5, 6]
        [4, 7, 10]
        ----------
        [2, 9, 7]
        [4, 1, 2]
        [4, 2, 4]
        """
        df1 = pd.DataFrame(np.array([[1, 1, 1]]), columns=['A', 'B', 'C'])
        df2 = pd.DataFrame(np.array([[1, 5, 6], [4, 7, 10]]), columns=['A', 'B', 'C'])
        df3 = pd.DataFrame(np.array([[2, 9, 7], [4, 1, 2], [4, 2, 4]]), columns=['A', 'B', 'C'])
        batches = [Batch(frames=df) for df in [df1, df2, df3]]
        'query: .... ORDER BY A ASC, B DESC limit 2'
        plan = OrderByPlan([(TupleValueExpression(col_alias='A'), ParserOrderBySortType.ASC), (TupleValueExpression(col_alias='B'), ParserOrderBySortType.DESC)])
        orderby_executor = OrderByExecutor(MagicMock(), plan)
        orderby_executor.append_child(DummyExecutor(batches))
        sorted_batches = list(orderby_executor.exec())
        limit_value = 2
        plan = LimitPlan(ConstantValueExpression(limit_value))
        limit_executor = LimitExecutor(MagicMock(), plan)
        limit_executor.append_child(DummyExecutor(sorted_batches))
        reduced_batches = list(limit_executor.exec())
        aggregated_batch = Batch.concat(reduced_batches, copy=False)
        '\n           A  B   C\n        0  1  5   6\n        1  1  1   1\n        '
        expected_df1 = pd.DataFrame(np.array([[1, 5, 6], [1, 1, 1]]), columns=['A', 'B', 'C'])
        expected_batches = [Batch(frames=df) for df in [expected_df1]]
        self.assertEqual(expected_batches[0], aggregated_batch)

class SampleExecutorTest(unittest.TestCase):

    def test_should_return_smaller_num_rows(self):
        dfs = [pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD')) for _ in range(4)]
        batches = [Batch(frames=df) for df in dfs]
        sample_value = 3
        plan = SamplePlan(ConstantValueExpression(sample_value))
        sample_executor = SampleExecutor(MagicMock(), plan)
        sample_executor.append_child(DummyExecutor(batches))
        reduced_batches = list(sample_executor.exec())
        original = Batch.concat(batches)
        filter = range(0, len(original), sample_value)
        original = original._get_frames_from_indices(filter)
        original = Batch.concat([original])
        reduced = Batch.concat(reduced_batches)
        self.assertEqual(len(original), len(reduced))
        self.assertEqual(original, reduced)

class SeqScanExecutorTest(unittest.TestCase):

    def test_should_return_only_frames_satisfy_predicate(self):
        dataframe = create_dataframe(3)
        batch = Batch(frames=dataframe)
        expression = type('AbstractExpression', (), {'evaluate': lambda x: Batch(pd.DataFrame([False, False, True])), 'find_all': lambda expr: []})
        plan = type('ScanPlan', (), {'predicate': expression, 'columns': None, 'alias': None})
        predicate_executor = SequentialScanExecutor(MagicMock(), plan)
        predicate_executor.append_child(DummyExecutor([batch]))
        expected = Batch(batch.frames[batch.frames.index == 2].reset_index(drop=True))
        filtered = list(predicate_executor.exec())[0]
        self.assertEqual(expected, filtered)

    def test_should_return_all_frames_when_no_predicate_is_applied(self):
        dataframe = create_dataframe(3)
        batch = Batch(frames=dataframe)
        plan = type('ScanPlan', (), {'predicate': None, 'columns': None, 'alias': None})
        predicate_executor = SequentialScanExecutor(MagicMock(), plan)
        predicate_executor.append_child(DummyExecutor([batch]))
        filtered = list(predicate_executor.exec())[0]
        self.assertEqual(batch, filtered)

    def test_should_return_projected_columns(self):
        dataframe = create_dataframe(3)
        batch = Batch(frames=dataframe)
        proj_batch = Batch(frames=pd.DataFrame(dataframe['data']))
        expression = [type('AbstractExpression', (), {'evaluate': lambda x: Batch(pd.DataFrame(x.frames['data'])), 'find_all': lambda expr: []})]
        plan = type('ScanPlan', (), {'predicate': None, 'columns': expression, 'alias': None})
        proj_executor = SequentialScanExecutor(MagicMock(), plan)
        proj_executor.append_child(DummyExecutor([batch]))
        actual = list(proj_executor.exec())[0]
        self.assertEqual(proj_batch, actual)

class LogicalExpressionsTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch = Batch(pd.DataFrame([0]))

    def test_logical_and(self):
        const_exp1 = ConstantValueExpression(1)
        const_exp2 = ConstantValueExpression(1)
        comparison_expression_left = ComparisonExpression(ExpressionType.COMPARE_EQUAL, const_exp1, const_exp2)
        const_exp1 = ConstantValueExpression(2)
        const_exp2 = ConstantValueExpression(1)
        comparison_expression_right = ComparisonExpression(ExpressionType.COMPARE_GREATER, const_exp1, const_exp2)
        logical_expr = LogicalExpression(ExpressionType.LOGICAL_AND, comparison_expression_left, comparison_expression_right)
        self.assertEqual([True], logical_expr.evaluate(self.batch).frames[0].tolist())

    def test_logical_or(self):
        const_exp1 = ConstantValueExpression(1)
        const_exp2 = ConstantValueExpression(1)
        comparison_expression_left = ComparisonExpression(ExpressionType.COMPARE_EQUAL, const_exp1, const_exp2)
        const_exp1 = ConstantValueExpression(1)
        const_exp2 = ConstantValueExpression(2)
        comparison_expression_right = ComparisonExpression(ExpressionType.COMPARE_GREATER, const_exp1, const_exp2)
        logical_expr = LogicalExpression(ExpressionType.LOGICAL_OR, comparison_expression_left, comparison_expression_right)
        self.assertEqual([True], logical_expr.evaluate(self.batch).frames[0].tolist())

    def test_logical_not(self):
        const_exp1 = ConstantValueExpression(0)
        const_exp2 = ConstantValueExpression(1)
        comparison_expression_right = ComparisonExpression(ExpressionType.COMPARE_GREATER, const_exp1, const_exp2)
        logical_expr = LogicalExpression(ExpressionType.LOGICAL_NOT, None, comparison_expression_right)
        self.assertEqual([True], logical_expr.evaluate(self.batch).frames[0].tolist())

    def test_short_circuiting_and_complete(self):
        tup_val_exp_l = TupleValueExpression(name=0)
        tup_val_exp_l.col_alias = 0
        tup_val_exp_r = TupleValueExpression(name=1)
        tup_val_exp_r.col_alias = 1
        comp_exp_l = ComparisonExpression(ExpressionType.COMPARE_EQUAL, tup_val_exp_l, tup_val_exp_r)
        comp_exp_r = Mock(spec=ComparisonExpression)
        logical_exp = LogicalExpression(ExpressionType.LOGICAL_AND, comp_exp_l, comp_exp_r)
        tuples = Batch(pd.DataFrame({0: [1, 2, 3], 1: [4, 5, 6]}))
        self.assertEqual([False, False, False], logical_exp.evaluate(tuples).frames[0].tolist())
        comp_exp_r.evaluate.assert_not_called()

    def test_short_circuiting_or_complete(self):
        tup_val_exp_l = TupleValueExpression(name=0)
        tup_val_exp_l.col_alias = 0
        tup_val_exp_r = TupleValueExpression(name=1)
        tup_val_exp_r.col_alias = 1
        comp_exp_l = ComparisonExpression(ExpressionType.COMPARE_EQUAL, tup_val_exp_l, tup_val_exp_r)
        comp_exp_r = Mock(spec=ComparisonExpression)
        logical_exp = LogicalExpression(ExpressionType.LOGICAL_OR, comp_exp_l, comp_exp_r)
        tuples = Batch(pd.DataFrame({0: [1, 2, 3], 1: [1, 2, 3]}))
        self.assertEqual([True, True, True], logical_exp.evaluate(tuples).frames[0].tolist())
        comp_exp_r.evaluate.assert_not_called()

    def test_short_circuiting_and_partial(self):
        tup_val_exp_l = TupleValueExpression(name=0)
        tup_val_exp_l.col_alias = 0
        tup_val_exp_r = TupleValueExpression(name=1)
        tup_val_exp_r.col_alias = 1
        comp_exp_l = ComparisonExpression(ExpressionType.COMPARE_EQUAL, tup_val_exp_l, tup_val_exp_r)
        comp_exp_r = Mock(spec=ComparisonExpression)
        comp_exp_r.evaluate = Mock(return_value=Mock(_frames=[[True], [False]]))
        logical_exp = LogicalExpression(ExpressionType.LOGICAL_AND, comp_exp_l, comp_exp_r)
        tuples = Batch(pd.DataFrame({0: [1, 2, 3, 4], 1: [1, 2, 5, 6]}))
        self.assertEqual([True, False, False, False], logical_exp.evaluate(tuples).frames[0].tolist())
        comp_exp_r.evaluate.assert_called_once_with(tuples[[0, 1]])

    def test_short_circuiting_or_partial(self):
        tup_val_exp_l = TupleValueExpression(name=0)
        tup_val_exp_l.col_alias = 0
        tup_val_exp_r = TupleValueExpression(name=1)
        tup_val_exp_r.col_alias = 1
        comp_exp_l = ComparisonExpression(ExpressionType.COMPARE_EQUAL, tup_val_exp_l, tup_val_exp_r)
        comp_exp_r = Mock(spec=ComparisonExpression)
        comp_exp_r.evaluate = Mock(return_value=Mock(_frames=[[True], [False]]))
        logical_exp = LogicalExpression(ExpressionType.LOGICAL_OR, comp_exp_l, comp_exp_r)
        tuples = Batch(pd.DataFrame({0: [1, 2, 3, 4], 1: [5, 6, 3, 4]}))
        self.assertEqual([True, False, True, True], logical_exp.evaluate(tuples).frames[0].tolist())
        comp_exp_r.evaluate.assert_called_once_with(tuples[[0, 1]])

    def test_multiple_logical(self):
        batch = Batch(pd.DataFrame({'col': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}))
        comp_left = ComparisonExpression(ExpressionType.COMPARE_GREATER, TupleValueExpression(col_alias='col'), ConstantValueExpression(1))
        batch_copy = Batch(pd.DataFrame({'col': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}))
        expected = batch[list(range(2, 10))]
        batch_copy.drop_zero(comp_left.evaluate(batch))
        self.assertEqual(batch_copy, expected)
        comp_right = ComparisonExpression(ExpressionType.COMPARE_LESSER, TupleValueExpression(col_alias='col'), ConstantValueExpression(8))
        batch_copy = Batch(pd.DataFrame({'col': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}))
        expected = batch[list(range(0, 8))]
        batch_copy.drop_zero(comp_right.evaluate(batch))
        self.assertEqual(batch_copy, expected)
        comp_expr = ComparisonExpression(ExpressionType.COMPARE_GEQ, TupleValueExpression(col_alias='col'), ConstantValueExpression(5))
        batch_copy = Batch(pd.DataFrame({'col': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}))
        expected = batch[list(range(5, 10))]
        batch_copy.drop_zero(comp_expr.evaluate(batch))
        self.assertEqual(batch_copy, expected)
        l_expr = LogicalExpression(ExpressionType.LOGICAL_AND, comp_left, comp_right)
        root_l_expr = LogicalExpression(ExpressionType.LOGICAL_AND, comp_expr, l_expr)
        batch_copy = Batch(pd.DataFrame({'col': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}))
        expected = batch[[5, 6, 7]]
        batch_copy.drop_zero(root_l_expr.evaluate(batch))
        self.assertEqual(batch_copy, expected)
        root_l_expr = LogicalExpression(ExpressionType.LOGICAL_AND, l_expr, comp_expr)
        batch_copy = Batch(pd.DataFrame({'col': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}))
        expected = batch[[5, 6, 7]]
        batch_copy.drop_zero(root_l_expr.evaluate(batch))
        self.assertEqual(batch_copy, expected)
        between_4_7 = LogicalExpression(ExpressionType.LOGICAL_AND, ComparisonExpression(ExpressionType.COMPARE_GEQ, TupleValueExpression(col_alias='col'), ConstantValueExpression(4)), ComparisonExpression(ExpressionType.COMPARE_LEQ, TupleValueExpression(col_alias='col'), ConstantValueExpression(7)))
        test_expr = LogicalExpression(ExpressionType.LOGICAL_AND, between_4_7, l_expr)
        batch_copy = Batch(pd.DataFrame({'col': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}))
        expected = batch[[4, 5, 6, 7]]
        batch_copy.drop_zero(test_expr.evaluate(batch))
        self.assertEqual(batch_copy, expected)
        test_expr = LogicalExpression(ExpressionType.LOGICAL_OR, between_4_7, l_expr)
        batch_copy = Batch(pd.DataFrame({'col': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}))
        expected = batch[[2, 3, 4, 5, 6, 7]]
        batch_copy.drop_zero(test_expr.evaluate(batch))
        self.assertEqual(batch_copy, expected)

class ComparisonExpressionsTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch = Batch(pd.DataFrame([0]))

    def test_comparison_compare_equal(self):
        const_exp1 = ConstantValueExpression(1)
        const_exp2 = ConstantValueExpression(1)
        cmpr_exp = ComparisonExpression(ExpressionType.COMPARE_EQUAL, const_exp1, const_exp2)
        self.assertEqual([True], cmpr_exp.evaluate(self.batch).frames[0].tolist())
        self.assertNotEqual(str(cmpr_exp), None)

    def test_comparison_compare_greater(self):
        const_exp1 = ConstantValueExpression(1)
        const_exp2 = ConstantValueExpression(0)
        cmpr_exp = ComparisonExpression(ExpressionType.COMPARE_GREATER, const_exp1, const_exp2)
        self.assertEqual([True], cmpr_exp.evaluate(self.batch).frames[0].tolist())
        self.assertNotEqual(str(cmpr_exp), None)

    def test_comparison_compare_lesser(self):
        const_exp1 = ConstantValueExpression(0)
        const_exp2 = ConstantValueExpression(2)
        cmpr_exp = ComparisonExpression(ExpressionType.COMPARE_LESSER, const_exp1, const_exp2)
        self.assertEqual([True], cmpr_exp.evaluate(self.batch).frames[0].tolist())
        self.assertNotEqual(str(cmpr_exp), None)

    def test_comparison_compare_geq(self):
        const_exp1 = ConstantValueExpression(1)
        const_exp2 = ConstantValueExpression(1)
        const_exp3 = ConstantValueExpression(0)
        cmpr_exp1 = ComparisonExpression(ExpressionType.COMPARE_GEQ, const_exp1, const_exp2)
        cmpr_exp2 = ComparisonExpression(ExpressionType.COMPARE_GEQ, const_exp1, const_exp3)
        self.assertEqual([True], cmpr_exp1.evaluate(self.batch).frames[0].tolist())
        self.assertEqual([True], cmpr_exp2.evaluate(self.batch).frames[0].tolist())
        self.assertNotEqual(str(cmpr_exp1), None)

    def test_comparison_compare_leq(self):
        const_exp1 = ConstantValueExpression(0)
        const_exp2 = ConstantValueExpression(2)
        const_exp3 = ConstantValueExpression(2)
        cmpr_exp1 = ComparisonExpression(ExpressionType.COMPARE_LEQ, const_exp1, const_exp2)
        cmpr_exp2 = ComparisonExpression(ExpressionType.COMPARE_LEQ, const_exp2, const_exp3)
        self.assertEqual([True], cmpr_exp1.evaluate(self.batch).frames[0].tolist())
        self.assertEqual([True], cmpr_exp2.evaluate(self.batch).frames[0].tolist())
        self.assertNotEqual(str(cmpr_exp1), None)

    def test_comparison_compare_neq(self):
        const_exp1 = ConstantValueExpression(0)
        const_exp2 = ConstantValueExpression(1)
        cmpr_exp = ComparisonExpression(ExpressionType.COMPARE_NEQ, const_exp1, const_exp2)
        self.assertEqual([True], cmpr_exp.evaluate(self.batch).frames[0].tolist())
        self.assertNotEqual(str(cmpr_exp), None)

    def test_comparison_compare_contains(self):
        const_exp1 = ConstantValueExpression([1, 2], ColumnType.NDARRAY)
        const_exp2 = ConstantValueExpression([1, 5], ColumnType.NDARRAY)
        const_exp3 = ConstantValueExpression([1, 2, 3, 4], ColumnType.NDARRAY)
        cmpr_exp1 = ComparisonExpression(ExpressionType.COMPARE_CONTAINS, const_exp3, const_exp1)
        self.assertEqual([True], cmpr_exp1.evaluate(self.batch).frames[0].tolist())
        cmpr_exp2 = ComparisonExpression(ExpressionType.COMPARE_CONTAINS, const_exp3, const_exp2)
        self.assertEqual([False], cmpr_exp2.evaluate(self.batch).frames[0].tolist())
        self.assertNotEqual(str(cmpr_exp1), None)

    def test_comparison_compare_is_contained(self):
        const_exp1 = ConstantValueExpression([1, 2], ColumnType.NDARRAY)
        const_exp2 = ConstantValueExpression([1, 5], ColumnType.NDARRAY)
        const_exp3 = ConstantValueExpression([1, 2, 3, 4], ColumnType.NDARRAY)
        cmpr_exp1 = ComparisonExpression(ExpressionType.COMPARE_IS_CONTAINED, const_exp1, const_exp3)
        self.assertEqual([True], cmpr_exp1.evaluate(self.batch).frames[0].tolist())
        cmpr_exp2 = ComparisonExpression(ExpressionType.COMPARE_IS_CONTAINED, const_exp2, const_exp3)
        self.assertEqual([False], cmpr_exp2.evaluate(self.batch).frames[0].tolist())
        self.assertNotEqual(str(cmpr_exp1), None)

class ArithmeticExpressionsTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch = Batch(pd.DataFrame([None]))

    def test_addition(self):
        const_exp1 = ConstantValueExpression(2)
        const_exp2 = ConstantValueExpression(5)
        cmpr_exp = ArithmeticExpression(ExpressionType.ARITHMETIC_ADD, const_exp1, const_exp2)
        self.assertEqual([7], cmpr_exp.evaluate(self.batch).frames[0].tolist())

    def test_subtraction(self):
        const_exp1 = ConstantValueExpression(5)
        const_exp2 = ConstantValueExpression(2)
        cmpr_exp = ArithmeticExpression(ExpressionType.ARITHMETIC_SUBTRACT, const_exp1, const_exp2)
        self.assertEqual([3], cmpr_exp.evaluate(self.batch).frames[0].tolist())

    def test_multiply(self):
        const_exp1 = ConstantValueExpression(3)
        const_exp2 = ConstantValueExpression(5)
        cmpr_exp = ArithmeticExpression(ExpressionType.ARITHMETIC_MULTIPLY, const_exp1, const_exp2)
        self.assertEqual([15], cmpr_exp.evaluate(self.batch).frames[0].tolist())

    def test_divide(self):
        const_exp1 = ConstantValueExpression(5)
        const_exp2 = ConstantValueExpression(5)
        cmpr_exp = ArithmeticExpression(ExpressionType.ARITHMETIC_DIVIDE, const_exp1, const_exp2)
        self.assertEqual([1], cmpr_exp.evaluate(self.batch).frames[0].tolist())

    def test_aaequality(self):
        const_exp1 = ConstantValueExpression(5)
        const_exp2 = ConstantValueExpression(15)
        cmpr_exp = ArithmeticExpression(ExpressionType.ARITHMETIC_DIVIDE, const_exp1, const_exp2)
        cmpr_exp2 = ArithmeticExpression(ExpressionType.ARITHMETIC_MULTIPLY, const_exp1, const_exp2)
        cmpr_exp3 = ArithmeticExpression(ExpressionType.ARITHMETIC_MULTIPLY, const_exp2, const_exp1)
        self.assertNotEqual(cmpr_exp, cmpr_exp2)
        self.assertNotEqual(cmpr_exp2, cmpr_exp3)

class BatchTest(unittest.TestCase):

    def test_batch_serialize_deserialize(self):
        batch = Batch(frames=create_dataframe())
        batch2 = Batch.deserialize(batch.serialize())
        self.assertEqual(batch, batch2)

    def test_frames_as_numpy_array_should_frames_as_numpy_array(self):
        batch = Batch(frames=create_dataframe_same(2))
        expected = list(np.ones((2, 1, 1)))
        actual = list(batch.column_as_numpy_array(batch.columns[0]))
        self.assertEqual(expected, actual)

    def test_return_only_frames_specified_in_the_indices(self):
        batch = Batch(frames=create_dataframe(2))
        expected = Batch(frames=create_dataframe())
        output = batch[[0]]
        self.assertEqual(expected, output)

    def test_fetching_frames_by_index(self):
        batch = Batch(frames=create_dataframe_same(2))
        expected = Batch(frames=create_dataframe())
        self.assertEqual(expected, batch[0])

    def test_fetching_frames_by_index_should_raise(self):
        batch = Batch(frames=create_dataframe_same(2))
        with self.assertRaises(TypeError):
            batch[1.0]

    def test_slicing_on_batched_should_return_new_batch_frame(self):
        batch = Batch(frames=create_dataframe(2))
        expected = Batch(frames=create_dataframe())
        self.assertEqual(batch, batch[:])
        self.assertEqual(expected, batch[:-1])

    def test_slicing_should_word_for_negative_stop_value(self):
        batch = Batch(frames=create_dataframe(2))
        expected = Batch(frames=create_dataframe())
        self.assertEqual(expected, batch[:-1])

    def test_slicing_should_work_with_skip_value(self):
        batch = Batch(frames=create_dataframe(3))
        expected = Batch(frames=create_dataframe(3).iloc[[0, 2], :])
        self.assertEqual(expected, batch[::2])

    def test_add_should_raise_error_for_incompatible_type(self):
        batch = Batch(frames=create_dataframe())
        with self.assertRaises(TypeError):
            batch + 1

    def test_adding_to_empty_frame_batch_returns_itself(self):
        batch_1 = Batch(frames=pd.DataFrame())
        batch_2 = Batch(frames=create_dataframe())
        self.assertEqual(batch_2, batch_1 + batch_2)
        self.assertEqual(batch_2, batch_2 + batch_1)

    def test_adding_batch_frame_with_outcomes_returns_new_batch_frame(self):
        batch_1 = Batch(frames=create_dataframe())
        batch_2 = Batch(frames=create_dataframe())
        batch_3 = Batch(frames=create_dataframe_same(2))
        self.assertEqual(batch_3, batch_1 + batch_2)

    def test_concat_batch(self):
        batch_1 = Batch(frames=create_dataframe())
        batch_2 = Batch(frames=create_dataframe())
        batch_3 = Batch(frames=create_dataframe_same(2))
        self.assertEqual(batch_3, Batch.concat([batch_1, batch_2], copy=False))

    def test_concat_empty_batch_list_raise_exception(self):
        self.assertEqual(Batch(), Batch.concat([]))

    def test_project_batch_frame(self):
        batch_1 = Batch(frames=pd.DataFrame([{'id': 1, 'data': 2, 'info': 3}]))
        batch_2 = batch_1.project(['id', 'info'])
        batch_3 = Batch(frames=pd.DataFrame([{'id': 1, 'info': 3}]))
        self.assertEqual(batch_2, batch_3)

    def test_merge_column_wise_batch_frame(self):
        batch_1 = Batch(frames=pd.DataFrame([{'id': 0}]))
        batch_2 = Batch(frames=pd.DataFrame([{'data': 1}]))
        batch_3 = Batch.merge_column_wise([batch_1, batch_2])
        batch_4 = Batch(frames=pd.DataFrame([{'id': 0, 'data': 1}]))
        self.assertEqual(batch_3, batch_4)
        self.assertEqual(Batch.merge_column_wise([]), Batch())
        batch_1 = Batch(frames=pd.DataFrame({'id': [0, None, 1]}))
        batch_2 = Batch(frames=pd.DataFrame({'data': [None, 0, None]}))
        batch_res = Batch(frames=pd.DataFrame({'id': [0, None, 1], 'data': [None, 0, None]}))
        self.assertEqual(Batch.merge_column_wise([batch_1, batch_2]), batch_res)
        df_1 = pd.DataFrame({'id': [-10, 1, 2]})
        df_2 = pd.DataFrame({'data': [-20, 2, 3]})
        df_1 = df_1[df_1 < 0].dropna()
        df_1.reset_index(drop=True, inplace=True)
        df_2 = df_2[df_2 < 0].dropna()
        df_2.reset_index(drop=True, inplace=True)
        batch_1 = Batch(frames=df_1)
        batch_2 = Batch(frames=df_2)
        df_res = pd.DataFrame({'id': [-10, 1, 2], 'data': [-20, 2, 3]})
        df_res = df_res[df_res < 0].dropna()
        df_res.reset_index(drop=True, inplace=True)
        batch_res = Batch(frames=df_res)
        self.assertEqual(Batch.merge_column_wise([batch_1, batch_2]), batch_res)

    def test_should_fail_for_list(self):
        frames = [{'id': 0, 'data': [1, 2]}, {'id': 1, 'data': [1, 2]}]
        self.assertRaises(ValueError, Batch, frames)

    def test_should_fail_for_dict(self):
        frames = {'id': 0, 'data': [1, 2]}
        self.assertRaises(ValueError, Batch, frames)

    def test_should_return_correct_length(self):
        batch = Batch(create_dataframe(5))
        self.assertEqual(5, len(batch))

    def test_should_return_empty_dataframe(self):
        batch = Batch()
        self.assertEqual(batch, Batch(create_dataframe(0)))

    def test_stack_batch_more_than_one_column_should_raise_exception(self):
        batch = Batch(create_dataframe_same(2))
        self.assertRaises(ValueError, Batch.stack, batch)

    def test_modify_column_alias_should_raise_exception(self):
        batch = Batch(create_dataframe(5))
        dummy_alias = Alias('dummy', col_names=['t1'])
        with self.assertRaises(RuntimeError):
            batch.modify_column_alias(dummy_alias)

    def test_drop_column_alias_should_work_on_frame_without_alias(self):
        batch = Batch(create_dataframe(5))
        batch.drop_column_alias()

    def test_sort_orderby_should_raise_exception_on_missing_column(self):
        batch = Batch(create_dataframe(5))
        with self.assertRaises(AssertionError):
            batch.sort_orderby(by=['foo'])

class LogicalOrderByToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALORDERBY)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_ORDERBY_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_ORDERBY_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalOrderBy, context: OptimizerContext):
        after = OrderByPlan(before.orderby_list)
        for child in before.children:
            after.append_child(child)
        yield after

class LogicalLimitToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALLIMIT)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_LIMIT_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_LIMIT_TO_PHYSICAL

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalLimit, context: OptimizerContext):
        after = LimitPlan(before.limit_count)
        for child in before.children:
            after.append_child(child)
        yield after

class GroupByExecutor(AbstractExecutor):
    """
    Group inputs into 4d segments of length provided in the query
    E.g., "GROUP BY '8 frames'" groups every 8 frames into one segment

    Arguments:
        node (AbstractPlan): The GroupBy Plan

    """

    def __init__(self, db: EvaDBDatabase, node: GroupByPlan):
        super().__init__(db, node)
        numbers_only = re.sub('\\D', '', node.groupby_clause.value)
        self._segment_length = int(numbers_only)

    def exec(self, *args, **kwargs) -> Iterator[Batch]:
        child_executor = self.children[0]
        buffer = Batch(pd.DataFrame())
        for batch in child_executor.exec(**kwargs):
            new_batch = buffer + batch
            while len(new_batch) >= self._segment_length:
                yield new_batch[:self._segment_length]
                new_batch = new_batch[self._segment_length:]
            buffer = new_batch

class VectorIndexScanExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: VectorIndexScanPlan):
        super().__init__(db, node)
        self.index_name = node.index.name
        self.vector_store_type = node.index.type
        self.feat_column = node.index.feat_column
        self.limit_count = node.limit_count
        self.search_query_expr = node.search_query_expr

    def exec(self, *args, **kwargs) -> Iterator[Batch]:
        if self.vector_store_type == VectorStoreType.PGVECTOR:
            return self._native_vector_index_scan()
        else:
            return self._evadb_vector_index_scan(*args, **kwargs)

    def _get_search_query_results(self):
        dummy_batch = Batch(frames=pd.DataFrame({'0': [0]}))
        search_batch = self.search_query_expr.evaluate(dummy_batch)
        feature_col_name = self.search_query_expr.output_objs[0].name
        search_batch.drop_column_alias()
        search_feat = search_batch.column_as_numpy_array(feature_col_name)[0]
        search_feat = search_feat.reshape(1, -1)
        return search_feat

    def _native_vector_index_scan(self):
        search_feat = self._get_search_query_results()
        search_feat = search_feat.reshape(-1).tolist()
        tb_catalog_entry = list(self.node.find_all(StoragePlan))[0].table
        db_catalog_entry = self.db.catalog().get_database_catalog_entry(tb_catalog_entry.database_name)
        with get_database_handler(db_catalog_entry.engine, **db_catalog_entry.params) as handler:
            resp = handler.execute_native_query(f"SELECT * FROM {tb_catalog_entry.name}\n                                                ORDER BY {self.feat_column.name} <-> '{search_feat}'\n                                                LIMIT {self.limit_count}")
            if resp.error is not None:
                raise ExecutorError(f'Native index can encounters {resp.error}')
            res = Batch(frames=resp.data)
            res.modify_column_alias(tb_catalog_entry.name)
            yield res

    def _evadb_vector_index_scan(self, *args, **kwargs):
        index_catalog_entry = self.catalog().get_index_catalog_entry_by_name(self.index_name)
        self.index_path = index_catalog_entry.save_file_path
        self.index = VectorStoreFactory.init_vector_store(self.vector_store_type, self.index_name, **handle_vector_store_params(self.vector_store_type, self.index_path, self.db.catalog))
        search_feat = self._get_search_query_results()
        index_result = self.index.query(VectorIndexQuery(search_feat, self.limit_count.value))
        row_num_np = index_result.ids
        row_num_col_name = None
        num_required_results = self.limit_count.value
        if len(index_result.ids) < self.limit_count.value:
            num_required_results = len(index_result.ids)
            logger.warning(f'The index {self.index_name} returned only {num_required_results} results, which is fewer than the required {self.limit_count.value}.')
        final_df = pd.DataFrame()
        res_data_list = []
        row_num_df = pd.DataFrame({'row_num_np': row_num_np})
        for batch in self.children[0].exec(**kwargs):
            if not row_num_col_name:
                column_list = batch.columns
                row_num_alias = get_row_num_column_alias(column_list)
                row_num_col_name = '{}.{}'.format(row_num_alias, ROW_NUM_COLUMN)
            if not batch.frames[row_num_col_name].isin(row_num_df['row_num_np']).any():
                continue
            for index, row in batch.frames.iterrows():
                row_dict = row.to_dict()
                res_data_list.append(row_dict)
        result_df = pd.DataFrame(res_data_list)
        final_df = pd.merge(row_num_df, result_df, left_on='row_num_np', right_on=row_num_col_name, how='inner')
        if 'row_num_np' in final_df:
            del final_df['row_num_np']
        yield Batch(final_df)

class ShowInfoExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: ShowInfoPlan):
        super().__init__(db, node)

    def exec(self, *args, **kwargs):
        show_entries = []
        assert self.node.show_type is ShowType.FUNCTIONS or ShowType.TABLES or ShowType.DATABASES or ShowType.CONFIGS, f'Show command does not support type {self.node.show_type}'
        if self.node.show_type is ShowType.FUNCTIONS:
            functions = self.catalog().get_all_function_catalog_entries()
            for function in functions:
                show_entries.append(function.display_format())
        elif self.node.show_type is ShowType.TABLES:
            tables = self.catalog().get_all_table_catalog_entries()
            for table in tables:
                if table.table_type != TableType.SYSTEM_STRUCTURED_DATA:
                    show_entries.append(table.name)
            show_entries = {'name': show_entries}
        elif self.node.show_type is ShowType.DATABASES:
            databases = self.catalog().get_all_database_catalog_entries()
            for db in databases:
                show_entries.append(db.display_format())
        elif self.node.show_type is ShowType.CONFIGS:
            show_entries = {}
            if self.node.show_val.upper() == ShowType.CONFIGS.name:
                configs = self.catalog().get_all_configuration_catalog_entries()
                for config in configs:
                    show_entries[config.key] = config.value
            else:
                value = self.catalog().get_configuration_catalog_value(key=self.node.show_val.upper())
                show_entries = {}
                if value is not None:
                    show_entries = {self.node.show_val: [value]}
                else:
                    raise Exception('No configuration found with key {}'.format(self.node.show_val))
        yield Batch(pd.DataFrame(show_entries))

class ExplainExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: ExplainPlan):
        super().__init__(db, node)

    def exec(self, *args, **kwargs):
        plan_str = self._exec(self._node.children[0], 0)
        yield Batch(pd.DataFrame([plan_str]))

    def _exec(self, node: AbstractPlan, depth: int):
        cur_str = ' ' * depth * 4 + '|__ ' + str(node.__class__.__name__) + '\n'
        for child in node.children:
            cur_str += self._exec(child, depth + 1)
        return cur_str

class CreateIndexExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: CreateIndexPlan):
        super().__init__(db, node)
        self.name = self.node.name
        self.if_not_exists = self.node.if_not_exists
        self.table_ref = self.node.table_ref
        self.col_list = self.node.col_list
        self.vector_store_type = self.node.vector_store_type
        self.project_expr_list = self.node.project_expr_list
        self.index_def = self.node.index_def

    def exec(self, *args, **kwargs):
        if self.vector_store_type == VectorStoreType.PGVECTOR:
            self._create_native_index()
        else:
            self._create_evadb_index()
        yield Batch(pd.DataFrame([f'Index {self.name} successfully added to the database.']))

    def _create_native_index(self):
        table = self.table_ref.table
        db_catalog_entry = self.catalog().get_database_catalog_entry(table.database_name)
        with get_database_handler(db_catalog_entry.engine, **db_catalog_entry.params) as handler:
            resp = handler.execute_native_query(f'CREATE INDEX {self.name} ON {table.table_name}\n                    USING hnsw ({self.col_list[0].name} vector_l2_ops)')
            if resp.error is not None:
                raise ExecutorError(f'Native engine create index encounters error: {resp.error}')

    def _get_evadb_index_save_path(self) -> Path:
        index_dir = Path(self.db.catalog().get_configuration_catalog_value('index_dir'))
        if not index_dir.exists():
            index_dir.mkdir(parents=True, exist_ok=True)
        return str(index_dir / Path('{}_{}.index'.format(self.vector_store_type, self.name)))

    def _create_evadb_index(self):
        function_expression, function_expression_signature = (None, None)
        for project_expr in self.project_expr_list:
            if isinstance(project_expr, FunctionExpression):
                function_expression = project_expr
                function_expression_signature = project_expr.signature()
        feat_tb_catalog_entry = self.table_ref.table.table_obj
        feat_col_name = self.col_list[0].name
        feat_col_catalog_entry = [col for col in feat_tb_catalog_entry.columns if col.name == feat_col_name][0]
        if function_expression is not None:
            feat_col_name = function_expression.output_objs[0].name
        index_catalog_entry = self.catalog().get_index_catalog_entry_by_name(self.name)
        index_path = self._get_evadb_index_save_path()
        if index_catalog_entry is not None:
            msg = f'Index {self.name} already exists.'
            if self.if_not_exists:
                if index_catalog_entry.feat_column == feat_col_catalog_entry and index_catalog_entry.function_signature == function_expression_signature and (index_catalog_entry.type == self.node.vector_store_type):
                    logger.warn(msg + ' It will be updated on existing table.')
                    index = VectorStoreFactory.init_vector_store(self.vector_store_type, self.name, **handle_vector_store_params(self.vector_store_type, index_path, self.catalog))
                else:
                    logger.warn(msg)
                    return
            else:
                logger.error(msg)
                raise ExecutorError(msg)
        else:
            index = None
        try:
            for input_batch in self.children[0].exec():
                input_batch.drop_column_alias()
                feat = input_batch.column_as_numpy_array(feat_col_name)
                row_num = input_batch.column_as_numpy_array(ROW_NUM_COLUMN)
                for i in range(len(input_batch)):
                    row_feat = feat[i].reshape(1, -1)
                    if index is None:
                        input_dim = row_feat.shape[1]
                        index = VectorStoreFactory.init_vector_store(self.vector_store_type, self.name, **handle_vector_store_params(self.vector_store_type, index_path, self.catalog))
                        index.create(input_dim)
                    index.add([FeaturePayload(row_num[i], row_feat)])
            index.persist()
            if index_catalog_entry is None:
                self.catalog().insert_index_catalog_entry(self.name, index_path, self.vector_store_type, feat_col_catalog_entry, function_expression_signature, self.index_def)
        except Exception as e:
            if index:
                index.delete()
            raise ExecutorError(str(e))

class ConstantValueExpression(AbstractExpression):

    def __init__(self, value: Any, v_type: ColumnType=ColumnType.INTEGER):
        super().__init__(ExpressionType.CONSTANT_VALUE)
        self._value = value
        self._v_type = v_type

    def evaluate(self, batch: Batch, **kwargs):
        batch = Batch(pd.DataFrame({0: [self._value] * len(batch)}))
        return batch

    def signature(self) -> str:
        return str(self)

    @property
    def value(self):
        return self._value

    @property
    def v_type(self):
        return self._v_type

    def __eq__(self, other):
        is_subtree_equal = super().__eq__(other)
        if not isinstance(other, ConstantValueExpression):
            return False
        is_equal = is_subtree_equal and self.v_type == other.v_type
        if self.v_type == ColumnType.NDARRAY:
            return is_equal and all(self.value == other.value)
        else:
            return is_equal and self.value == other.value

    def __str__(self) -> str:
        expr_str = ''
        if not isinstance(self._value, np.ndarray):
            expr_str = f'{str(self._value)}'
        else:
            expr_str = f'{np.array_str(self._value)}'
        return expr_str

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.v_type, str(self.value)))

