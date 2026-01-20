# Cluster 9

def shutdown_ray():
    is_ray_enabled = is_ray_available()
    if is_ray_enabled:
        import ray
        ray.shutdown()

def is_ray_available() -> bool:
    try:
        try_to_import_ray()
        return True
    except ValueError:
        return False

def file_remove(path, parent_dir=None):
    parent_dir = parent_dir or get_tmp_dir()
    try:
        os.remove(os.path.join(parent_dir, path))
    except FileNotFoundError:
        pass

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

class JobSchedulerIntegrationTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        cls.job_name_1 = 'test_async_job_1'
        cls.job_name_2 = 'test_async_job_2'

    def setUp(self):
        execute_query_fetch_all(self.evadb, f'DROP JOB IF EXISTS {self.job_name_1};')
        execute_query_fetch_all(self.evadb, f'DROP JOB IF EXISTS {self.job_name_2};')

    @classmethod
    def tearDownClass(cls):
        shutdown_ray()
        execute_query_fetch_all(cls.evadb, f'DROP JOB IF EXISTS {cls.job_name_1};')
        execute_query_fetch_all(cls.evadb, f'DROP JOB IF EXISTS {cls.job_name_2};')

    def create_jobs(self):
        datetime_format = '%Y-%m-%d %H:%M:%S'
        start_time = (datetime.now() - timedelta(seconds=10)).strftime(datetime_format)
        end_time = (datetime.now() + timedelta(seconds=60)).strftime(datetime_format)
        create_csv_query = 'CREATE TABLE IF NOT EXISTS MyCSV (\n                                    id INTEGER UNIQUE,\n                                    frame_id INTEGER,\n                                    video_id INTEGER\n                                );\n                            '
        job_1_query = f"CREATE JOB IF NOT EXISTS {self.job_name_1} AS {{\n                                SELECT * FROM MyCSV;\n                            }}\n                            START '{start_time}'\n                            END '{end_time}'\n                            EVERY 4 seconds;\n                        "
        job_2_query = f"CREATE JOB IF NOT EXISTS {self.job_name_2} AS {{\n                            SHOW FUNCTIONS;\n                        }}\n                        START '{start_time}'\n                        END '{end_time}'\n                        EVERY 2 seconds;\n                    "
        execute_query_fetch_all(self.evadb, create_csv_query)
        execute_query_fetch_all(self.evadb, job_1_query)
        execute_query_fetch_all(self.evadb, job_2_query)

    def test_should_execute_the_scheduled_jobs(self):
        self.create_jobs()
        connection = EvaDBConnection(self.evadb, MagicMock(), MagicMock())
        connection.start_jobs()
        time.sleep(15)
        connection.stop_jobs()
        job_1_execution_count = len(self.evadb.catalog().get_job_history_by_job_id(1))
        job_2_execution_count = len(self.evadb.catalog().get_job_history_by_job_id(2))
        self.assertGreater(job_2_execution_count, job_1_execution_count)
        self.assertGreater(job_2_execution_count, 2)
        self.assertGreater(job_1_execution_count, 2)

@pytest.mark.notparallel
class DeleteExecutorTest(unittest.TestCase):

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        load_functions_for_testing(self.evadb, mode='debug')
        create_table_query = '\n                CREATE TABLE IF NOT EXISTS testDeleteOne\n                (\n                 id INTEGER,\n                 dummyfloat FLOAT(5, 3),\n                 feat   NDARRAY FLOAT32(1, 3),\n                 input  NDARRAY UINT8(1, 3)\n                 );\n                '
        execute_query_fetch_all(self.evadb, create_table_query)
        insert_query1 = '\n                INSERT INTO testDeleteOne (id, dummyfloat, feat, input)\n                VALUES (5, 1.5, [[0, 0, 0]], [[0, 0, 0]]);\n        '
        execute_query_fetch_all(self.evadb, insert_query1)
        insert_query2 = '\n                INSERT INTO testDeleteOne (id, dummyfloat,feat, input)\n                VALUES (15, 2.5, [[100, 100, 100]], [[100, 100, 100]]);\n        '
        execute_query_fetch_all(self.evadb, insert_query2)
        insert_query3 = '\n                INSERT INTO testDeleteOne (id, dummyfloat,feat, input)\n                VALUES (25, 3.5, [[200, 200, 200]], [[200, 200, 200]]);\n        '
        execute_query_fetch_all(self.evadb, insert_query3)
        path = f'{EvaDB_ROOT_DIR}/data/sample_videos/1/*.mp4'
        query = f'LOAD VIDEO "{path}" INTO TestDeleteVideos;'
        _ = execute_query_fetch_all(self.evadb, query)

    def tearDown(self):
        shutdown_ray()
        file_remove('dummy.avi')

    @unittest.skip('Not supported in current version')
    def test_should_delete_single_video_in_table(self):
        path = f'{EvaDB_ROOT_DIR}/data/sample_videos/1/2.mp4'
        delete_query = f'DELETE FROM TestDeleteVideos WHERE name="{path}";'
        batch = execute_query_fetch_all(self.evadb, delete_query)
        query = 'SELECT name FROM MyVideo'
        batch = execute_query_fetch_all(self.evadb, query)
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['data'][0], np.array([[[40, 40, 40], [40, 40, 40]], [[40, 40, 40], [40, 40, 40]]])))
        query = 'SELECT id, data FROM MyVideo WHERE id = 41;'
        batch = execute_query_fetch_all(self.evadb, query)
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['data'][0], np.array([[[41, 41, 41], [41, 41, 41]], [[41, 41, 41], [41, 41, 41]]])))

    @unittest.skip('Not supported in current version')
    def test_should_delete_single_image_in_table(self):
        path = f'{EvaDB_ROOT_DIR}/data/sample_videos/1/2.mp4'
        delete_query = f'DELETE FROM TestDeleteVideos WHERE name="{path}";'
        batch = execute_query_fetch_all(self.evadb, delete_query)
        query = 'SELECT name FROM MyVideo'
        batch = execute_query_fetch_all(self.evadb, query)
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['data'][0], np.array([[[40, 40, 40], [40, 40, 40]], [[40, 40, 40], [40, 40, 40]]])))
        query = 'SELECT id, data FROM MyVideo WHERE id = 41;'
        batch = execute_query_fetch_all(self.evadb, query)
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['data'][0], np.array([[[41, 41, 41], [41, 41, 41]], [[41, 41, 41], [41, 41, 41]]])))

    def test_should_delete_tuple_in_table(self):
        delete_query = 'DELETE FROM testDeleteOne WHERE\n               id < 20 OR dummyfloat < 2 AND id < 5 AND 20 > id\n               AND id <= 20 AND id >= 5 OR id != 15 OR id = 15;'
        batch = execute_query_fetch_all(self.evadb, delete_query)
        query = 'SELECT * FROM testDeleteOne;'
        batch = execute_query_fetch_all(self.evadb, query)
        np.testing.assert_array_equal(batch.frames['testdeleteone.id'].array, np.array([25], dtype=np.int64))

@pytest.mark.notparallel
class SimilarityTests(unittest.TestCase):

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        load_functions_for_testing(self.evadb, mode='debug')
        self.img_path = create_sample_image()
        create_table_query = 'CREATE TABLE IF NOT EXISTS testSimilarityTable\n                                  (data_col NDARRAY UINT8(3, ANYDIM, ANYDIM),\n                                   dummy INTEGER);'
        execute_query_fetch_all(self.evadb, create_table_query)
        create_table_query = 'CREATE TABLE IF NOT EXISTS testSimilarityFeatureTable\n                                  (feature_col NDARRAY FLOAT32(1, ANYDIM),\n                                   dummy INTEGER);'
        execute_query_fetch_all(self.evadb, create_table_query)
        base_img = np.array(np.ones((3, 3, 3)), dtype=np.uint8)
        base_img[0] -= 1
        base_img[2] += 1
        base_img += 4
        base_table_catalog_entry = self.evadb.catalog().get_table_catalog_entry('testSimilarityTable')
        feature_table_catalog_entry = self.evadb.catalog().get_table_catalog_entry('testSimilarityFeatureTable')
        self.img_path_list = []
        storage_engine = StorageEngine.factory(self.evadb, base_table_catalog_entry)
        for i in range(5):
            storage_engine.write(base_table_catalog_entry, Batch(pd.DataFrame([{'data_col': base_img, 'dummy': i}])))
            storage_engine.write(feature_table_catalog_entry, Batch(pd.DataFrame([{'feature_col': base_img.astype(np.float32).reshape(1, -1), 'dummy': i}])))
            img_save_path = os.path.join(self.evadb.catalog().get_configuration_catalog_value('tmp_dir'), f'test_similar_img{i}.jpg')
            try_to_import_cv2()
            import cv2
            cv2.imwrite(img_save_path, base_img)
            load_image_query = f"LOAD IMAGE '{img_save_path}' INTO testSimilarityImageDataset;"
            execute_query_fetch_all(self.evadb, load_image_query)
            self.img_path_list.append(img_save_path)
            base_img -= 1
        self.original_pinecone_key = os.environ.get('PINECONE_API_KEY')
        self.original_pinecone_env = os.environ.get('PINECONE_ENV')
        os.environ['PINECONE_API_KEY'] = '657e4fae-7208-4555-b0f2-9847dfa5b818'
        os.environ['PINECONE_ENV'] = 'gcp-starter'
        self.original_milvus_uri = os.environ.get('MILVUS_URI')
        self.original_milvus_db_name = os.environ.get('MILVUS_DB_NAME')
        os.environ['MILVUS_URI'] = 'http://localhost:19530'
        os.environ['MILVUS_DB_NAME'] = 'default'
        self.original_weaviate_key = os.environ.get('WEAVIATE_API_KEY')
        self.original_weaviate_env = os.environ.get('WEAVIATE_API_URL')
        os.environ['WEAVIATE_API_KEY'] = 'NM4adxLmhtJDF1dPXDiNhEGTN7hhGDpymmO0'
        os.environ['WEAVIATE_API_URL'] = 'https://cs6422-test2-zn83syib.weaviate.network'

    def tearDown(self):
        shutdown_ray()
        drop_table_query = 'DROP TABLE testSimilarityTable;'
        execute_query_fetch_all(self.evadb, drop_table_query)
        drop_table_query = 'DROP TABLE testSimilarityFeatureTable;'
        execute_query_fetch_all(self.evadb, drop_table_query)
        drop_table_query = 'DROP TABLE IF EXISTS testSimilarityImageDataset;'
        execute_query_fetch_all(self.evadb, drop_table_query)
        if self.original_pinecone_key:
            os.environ['PINECONE_API_KEY'] = self.original_pinecone_key
        else:
            del os.environ['PINECONE_API_KEY']
        if self.original_pinecone_env:
            os.environ['PINECONE_ENV'] = self.original_pinecone_env
        else:
            del os.environ['PINECONE_ENV']
        if self.original_milvus_uri:
            os.environ['MILVUS_URI'] = self.original_milvus_uri
        else:
            del os.environ['MILVUS_URI']
        if self.original_milvus_db_name:
            os.environ['MILVUS_DB_NAME'] = self.original_milvus_db_name
        else:
            del os.environ['MILVUS_DB_NAME']

    def test_similarity_should_work_in_order(self):
        select_query = 'SELECT data_col FROM testSimilarityTable\n                            ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data_col))\n                            LIMIT 1;'.format(self.img_path)
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        base_img = np.array(np.ones((3, 3, 3)), dtype=np.uint8)
        base_img[0] -= 1
        base_img[2] += 1
        actual_open = actual_batch.frames['testsimilaritytable.data_col'].to_numpy()[0]
        self.assertTrue(np.array_equal(actual_open, base_img))
        select_query = 'SELECT data_col FROM testSimilarityTable\n                            ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data_col))\n                            LIMIT 2;'.format(self.img_path)
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_open = actual_batch.frames['testsimilaritytable.data_col'].to_numpy()[0]
        self.assertTrue(np.array_equal(actual_open, base_img))
        actual_open = actual_batch.frames['testsimilaritytable.data_col'].to_numpy()[1]
        self.assertTrue(np.array_equal(actual_open, base_img + 1))
        select_query = 'SELECT data_col FROM testSimilarityTable\n                            ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data_col)) DESC\n                            LIMIT 2;'.format(self.img_path)
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_open = actual_batch.frames['testsimilaritytable.data_col'].to_numpy()[0]
        self.assertTrue(np.array_equal(actual_open, base_img + 4))
        actual_open = actual_batch.frames['testsimilaritytable.data_col'].to_numpy()[1]
        self.assertTrue(np.array_equal(actual_open, base_img + 3))
        select_query = 'SELECT feature_col FROM testSimilarityFeatureTable\n                            ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), feature_col)\n                            LIMIT 1;'.format(self.img_path)
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        base_img = np.array(np.ones((3, 3, 3)), dtype=np.uint8)
        base_img[0] -= 1
        base_img[2] += 1
        base_img = base_img.astype(np.float32).reshape(1, -1)
        actual_open = actual_batch.frames['testsimilarityfeaturetable.feature_col'].to_numpy()[0]
        self.assertTrue(np.array_equal(actual_open, base_img))
        select_query = 'SELECT feature_col FROM testSimilarityFeatureTable\n                            ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), feature_col)\n                            LIMIT 2;'.format(self.img_path)
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_open = actual_batch.frames['testsimilarityfeaturetable.feature_col'].to_numpy()[0]
        self.assertTrue(np.array_equal(actual_open, base_img))
        actual_open = actual_batch.frames['testsimilarityfeaturetable.feature_col'].to_numpy()[1]
        self.assertTrue(np.array_equal(actual_open, base_img + 1))

    def test_should_do_vector_index_scan(self):
        select_query = 'SELECT feature_col FROM testSimilarityFeatureTable\n                            ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), feature_col)\n                            LIMIT 3;'.format(self.img_path)
        expected_batch = execute_query_fetch_all(self.evadb, select_query)
        create_index_query = 'CREATE INDEX testFaissIndexScanRewrite1\n                                    ON testSimilarityFeatureTable (feature_col)\n                                    USING FAISS;'
        execute_query_fetch_all(self.evadb, create_index_query)
        select_query = 'SELECT feature_col FROM testSimilarityFeatureTable\n                            ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), feature_col)\n                            LIMIT 3;'.format(self.img_path)
        explain_query = 'EXPLAIN {}'.format(select_query)
        explain_batch = execute_query_fetch_all(self.evadb, explain_query)
        self.assertTrue('VectorIndexScan' in explain_batch.frames[0][0])
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(actual_batch), 3)
        for i in range(3):
            self.assertTrue(np.array_equal(expected_batch.frames['testsimilarityfeaturetable.feature_col'].to_numpy()[i], actual_batch.frames['testsimilarityfeaturetable.feature_col'].to_numpy()[i]))
        select_query = 'SELECT data_col FROM testSimilarityTable\n                            ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data_col))\n                            LIMIT 3;'.format(self.img_path)
        expected_batch = execute_query_fetch_all(self.evadb, select_query)
        create_index_query = 'CREATE INDEX testFaissIndexScanRewrite2\n                                    ON testSimilarityTable (DummyFeatureExtractor(data_col))\n                                    USING FAISS;'
        execute_query_fetch_all(self.evadb, create_index_query)
        select_query = 'SELECT data_col FROM testSimilarityTable\n                            ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data_col))\n                            LIMIT 3;'.format(self.img_path)
        explain_query = 'EXPLAIN {}'.format(select_query)
        explain_batch = execute_query_fetch_all(self.evadb, explain_query)
        self.assertTrue('VectorIndexScan' in explain_batch.frames[0][0])
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(actual_batch), 3)
        for i in range(3):
            self.assertTrue(np.array_equal(expected_batch.frames['testsimilaritytable.data_col'].to_numpy()[i], actual_batch.frames['testsimilaritytable.data_col'].to_numpy()[i]))
        drop_query = 'DROP INDEX testFaissIndexScanRewrite1'
        execute_query_fetch_all(self.evadb, drop_query)
        drop_query = 'DROP INDEX testFaissIndexScanRewrite2'
        execute_query_fetch_all(self.evadb, drop_query)

    def test_should_not_do_vector_index_scan_with_desc_order(self):
        create_index_query = 'CREATE INDEX testFaissIndexScanRewrite\n                                    ON testSimilarityTable (DummyFeatureExtractor(data_col))\n                                    USING FAISS;'
        execute_query_fetch_all(self.evadb, create_index_query)
        explain_query = '\n            EXPLAIN\n                SELECT data_col FROM testSimilarityTable WHERE dummy = 0\n                  ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data_col))\n                  LIMIT 3;\n        '.format('dummypath')
        batch = execute_query_fetch_all(self.evadb, explain_query)
        self.assertFalse('FaissIndexScan' in batch.frames[0][0])
        base_img = np.array(np.ones((3, 3, 3)), dtype=np.uint8)
        base_img[0] -= 1
        base_img[2] += 1
        select_query = 'SELECT data_col FROM testSimilarityTable\n                            ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data_col)) DESC\n                            LIMIT 2;'.format(self.img_path)
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        actual_open = actual_batch.frames['testsimilaritytable.data_col'].to_numpy()[0]
        self.assertTrue(np.array_equal(actual_open, base_img + 4))
        actual_open = actual_batch.frames['testsimilaritytable.data_col'].to_numpy()[1]
        self.assertTrue(np.array_equal(actual_open, base_img + 3))
        drop_query = 'DROP INDEX testFaissIndexScanRewrite'
        execute_query_fetch_all(self.evadb, drop_query)

    def test_should_not_do_vector_index_scan_with_predicate(self):
        create_index_query = 'CREATE INDEX testFaissIndexScanRewrite\n                                    ON testSimilarityTable (DummyFeatureExtractor(data_col))\n                                    USING FAISS;'
        execute_query_fetch_all(self.evadb, create_index_query)
        explain_query = '\n            EXPLAIN\n                SELECT data_col FROM testSimilarityTable WHERE dummy = 0\n                  ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data_col))\n                  LIMIT 3;\n        '.format('dummypath')
        batch = execute_query_fetch_all(self.evadb, explain_query)
        self.assertFalse('FaissIndexScan' in batch.frames[0][0])
        drop_query = 'DROP INDEX testFaissIndexScanRewrite'
        execute_query_fetch_all(self.evadb, drop_query)

    def test_end_to_end_index_scan_should_work_correctly_on_image_dataset_faiss(self):
        for _ in range(2):
            create_index_query = 'CREATE INDEX testFaissIndexImageDataset\n                                        ON testSimilarityImageDataset (DummyFeatureExtractor(data))\n                                        USING FAISS;'
            execute_query_fetch_all(self.evadb, create_index_query)
            select_query = 'SELECT _row_id FROM testSimilarityImageDataset\n                                ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data))\n                                LIMIT 1;'.format(self.img_path)
            explain_batch = execute_query_fetch_all(self.evadb, f'EXPLAIN {select_query}')
            self.assertTrue('VectorIndexScan' in explain_batch.frames[0][0])
            res_batch = execute_query_fetch_all(self.evadb, select_query)
            self.assertEqual(res_batch.frames['testsimilarityimagedataset._row_id'][0], 5)
            drop_query = 'DROP INDEX testFaissIndexImageDataset'
            execute_query_fetch_all(self.evadb, drop_query)

    def _helper_for_auto_update_during_insertion_with_faiss(self, if_exists: bool):
        for i, img_path in enumerate(self.img_path_list):
            insert_query = f"INSERT INTO testIndexAutoUpdate (img_path) VALUES ('{img_path}')"
            execute_query_fetch_all(self.evadb, insert_query)
            if i == 0:
                if_exists_str = 'IF NOT EXISTS ' if if_exists else ''
                create_index_query = f'CREATE INDEX {if_exists_str}testIndex ON testIndexAutoUpdate(DummyFeatureExtractor(Open(img_path))) USING FAISS'
                execute_query_fetch_all(self.evadb, create_index_query)
        select_query = 'SELECT _row_id FROM testIndexAutoUpdate\n                                ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(Open(img_path)))\n                                LIMIT 1;'.format(self.img_path)
        explain_batch = execute_query_fetch_all(self.evadb, f'EXPLAIN {select_query}')
        self.assertTrue('VectorIndexScan' in explain_batch.frames[0][0])
        res_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(res_batch.frames['testindexautoupdate._row_id'][0], 5)

    def test_index_auto_update_on_structured_table_during_insertion_with_faiss(self):
        create_query = 'CREATE TABLE testIndexAutoUpdate (img_path TEXT(100))'
        drop_query = 'DROP TABLE testIndexAutoUpdate'
        execute_query_fetch_all(self.evadb, create_query)
        self._helper_for_auto_update_during_insertion_with_faiss(False)
        execute_query_fetch_all(self.evadb, drop_query)
        execute_query_fetch_all(self.evadb, create_query)
        self._helper_for_auto_update_during_insertion_with_faiss(True)

    @qdrant_skip_marker
    def test_end_to_end_index_scan_should_work_correctly_on_image_dataset_qdrant(self):
        for _ in range(2):
            create_index_query = 'CREATE INDEX testQdrantIndexImageDataset\n                                        ON testSimilarityImageDataset (DummyFeatureExtractor(data))\n                                        USING QDRANT;'
            execute_query_fetch_all(self.evadb, create_index_query)
            select_query = 'SELECT _row_id FROM testSimilarityImageDataset\n                                ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data))\n                                LIMIT 1;'.format(self.img_path)
            explain_batch = execute_query_fetch_all(self.evadb, f'EXPLAIN {select_query}')
            self.assertTrue('VectorIndexScan' in explain_batch.frames[0][0])
            '|__ ProjectPlan\n                |__ VectorIndexScanPlan\n                    |__ SeqScanPlan\n                        |__ StoragePlan'
            res_batch = execute_query_fetch_all(self.evadb, select_query)
            self.assertEqual(res_batch.frames['testsimilarityimagedataset._row_id'][0], 5)
            drop_query = 'DROP INDEX testQdrantIndexImageDataset'
            execute_query_fetch_all(self.evadb, drop_query)

    @chromadb_skip_marker
    def test_end_to_end_index_scan_should_work_correctly_on_image_dataset_chromadb(self):
        for _ in range(2):
            create_index_query = 'CREATE INDEX testChromaDBIndexImageDataset\n                                    ON testSimilarityImageDataset (DummyFeatureExtractor(data))\n                                    USING CHROMADB;'
            execute_query_fetch_all(self.evadb, create_index_query)
            select_query = 'SELECT _row_id FROM testSimilarityImageDataset\n                                ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data))\n                                LIMIT 1;'.format(self.img_path)
            explain_batch = execute_query_fetch_all(self.evadb, f'EXPLAIN {select_query}')
            self.assertTrue('VectorIndexScan' in explain_batch.frames[0][0])
            res_batch = execute_query_fetch_all(self.evadb, select_query)
            self.assertEqual(res_batch.frames['testsimilarityimagedataset._row_id'][0], 5)
            drop_query = 'DROP INDEX testChromaDBIndexImageDataset'
            execute_query_fetch_all(self.evadb, drop_query)

    @pytest.mark.skip(reason='Flaky testcase due to `bad request` error message')
    @pinecone_skip_marker
    def test_end_to_end_index_scan_should_work_correctly_on_image_dataset_pinecone(self):
        for _ in range(2):
            create_index_query = 'CREATE INDEX testpineconeindeximagedataset\n                                    ON testSimilarityImageDataset (DummyFeatureExtractor(data))\n                                    USING PINECONE;'
            execute_query_fetch_all(self.evadb, create_index_query)
            time.sleep(20)
            select_query = 'SELECT _row_id FROM testSimilarityImageDataset\n                                ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data))\n                                LIMIT 1;'.format(self.img_path)
            explain_batch = execute_query_fetch_all(self.evadb, f'EXPLAIN {select_query}')
            self.assertTrue('VectorIndexScan' in explain_batch.frames[0][0])
            res_batch = execute_query_fetch_all(self.evadb, select_query)
            self.assertEqual(res_batch.frames['testsimilarityimagedataset._row_id'][0], 5)
            drop_index_query = 'DROP INDEX testpineconeindeximagedataset;'
            execute_query_fetch_all(self.evadb, drop_index_query)

    @pytest.mark.skip(reason='Requires running local Milvus instance')
    @milvus_skip_marker
    def test_end_to_end_index_scan_should_work_correctly_on_image_dataset_milvus(self):
        for _ in range(2):
            create_index_query = 'CREATE INDEX testMilvusIndexImageDataset\n                                    ON testSimilarityImageDataset (DummyFeatureExtractor(data))\n                                    USING MILVUS;'
            execute_query_fetch_all(self.evadb, create_index_query)
            select_query = 'SELECT _row_id FROM testSimilarityImageDataset\n                                ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data))\n                                LIMIT 1;'.format(self.img_path)
            explain_batch = execute_query_fetch_all(self.evadb, f'EXPLAIN {select_query}')
            self.assertTrue('VectorIndexScan' in explain_batch.frames[0][0])
            res_batch = execute_query_fetch_all(self.evadb, select_query)
            self.assertEqual(res_batch.frames['testsimilarityimagedataset._row_id'][0], 5)
            drop_query = 'DROP INDEX testMilvusIndexImageDataset'
            execute_query_fetch_all(self.evadb, drop_query)

    @pytest.mark.skip(reason='Requires running Weaviate instance')
    @weaviate_skip_marker
    def test_end_to_end_index_scan_should_work_correctly_on_image_dataset_weaviate(self):
        for _ in range(2):
            create_index_query = 'CREATE INDEX testWeaviateIndexImageDataset\n                                    ON testSimilarityImageDataset (DummyFeatureExtractor(data))\n                                    USING WEAVIATE;'
            execute_query_fetch_all(self.evadb, create_index_query)
            select_query = 'SELECT _row_id FROM testSimilarityImageDataset\n                                ORDER BY Similarity(DummyFeatureExtractor(Open("{}")), DummyFeatureExtractor(data))\n                                LIMIT 1;'.format(self.img_path)
            explain_batch = execute_query_fetch_all(self.evadb, f'EXPLAIN {select_query}')
            self.assertTrue('VectorIndexScan' in explain_batch.frames[0][0])
            res_batch = execute_query_fetch_all(self.evadb, select_query)
            self.assertEqual(res_batch.frames['testsimilarityimagedataset._row_id'][0], 5)
            drop_query = 'DROP INDEX testWeaviateIndexImageDataset'
            execute_query_fetch_all(self.evadb, drop_query)

@pytest.mark.notparallel
@gpu_skip_marker
class OptimizerRulesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        ua_detrac = f'{EvaDB_ROOT_DIR}/data/ua_detrac/ua_detrac.mp4'
        execute_query_fetch_all(cls.evadb, f"LOAD VIDEO '{ua_detrac}' INTO MyVideo;")
        execute_query_fetch_all(cls.evadb, f"LOAD VIDEO '{ua_detrac}' INTO MyVideo2;")
        load_functions_for_testing(cls.evadb, mode='debug')

    @classmethod
    def tearDownClass(cls):
        shutdown_ray()
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS MyVideo;')

    @patch('evadb.expression.function_expression.FunctionExpression.evaluate')
    @patch('evadb.models.storage.batch.Batch.merge_column_wise')
    def test_should_benefit_from_pushdown(self, merge_mock, evaluate_mock):
        evaluate_mock.return_value = Batch(pd.DataFrame({'obj.labels': ['car'], 'obj.bboxes': [np.array([1, 2, 3, 4])], 'obj.scores': [0.8]}))
        query = 'SELECT id, obj.labels\n                  FROM MyVideo JOIN LATERAL\n                    FastRCNNObjectDetector(data) AS obj(labels, bboxes, scores)\n                  WHERE id < 2;'
        time_with_rule = Timer()
        result_with_rule = None
        with time_with_rule:
            result_with_rule = execute_query_fetch_all(self.evadb, query)
        evaluate_count_with_rule = evaluate_mock.call_count
        time_without_rule = Timer()
        result_without_pushdown_rules = None
        with time_without_rule:
            rules_manager = RulesManager()
            with disable_rules(rules_manager, [PushDownFilterThroughApplyAndMerge(), PushDownFilterThroughJoin()]):
                custom_plan_generator = PlanGenerator(self.evadb, rules_manager)
                result_without_pushdown_rules = execute_query_fetch_all(self.evadb, query, plan_generator=custom_plan_generator)
        self.assertEqual(result_without_pushdown_rules, result_with_rule)
        evaluate_count_without_rule = evaluate_mock.call_count - evaluate_count_with_rule
        self.assertGreater(evaluate_count_without_rule, 3 * evaluate_count_with_rule)

    def test_should_pushdown_without_pushdown_join_rule(self):
        query = 'SELECT id, obj.labels\n                    FROM MyVideo JOIN LATERAL\n                    FastRCNNObjectDetector(data) AS obj(labels, bboxes, scores)\n                    WHERE id < 2;'
        time_with_rule = Timer()
        result_with_rule = None
        with time_with_rule:
            result_with_rule = execute_query_fetch_all(self.evadb, query)
            query_plan = execute_query_fetch_all(self.evadb, f'EXPLAIN {query}')
        time_without_rule = Timer()
        result_without_pushdown_join_rule = None
        with time_without_rule:
            rules_manager = RulesManager()
            with disable_rules(rules_manager, [PushDownFilterThroughJoin()]):
                custom_plan_generator = PlanGenerator(self.evadb, rules_manager)
                result_without_pushdown_join_rule = execute_query_fetch_all(self.evadb, query, plan_generator=custom_plan_generator)
                query_plan_without_pushdown_join_rule = execute_query_fetch_all(self.evadb, f'EXPLAIN {query}', plan_generator=custom_plan_generator)
        self.assertEqual(result_without_pushdown_join_rule, result_with_rule)
        self.assertEqual(query_plan, query_plan_without_pushdown_join_rule)

    @patch('evadb.catalog.catalog_manager.CatalogManager.get_function_cost_catalog_entry')
    def test_should_reorder_predicates(self, mock):

        def _check_reorder(cost_func):
            mock.side_effect = cost_func
            pred_1 = "DummyObjectDetector(data).label = ['person']"
            pred_2 = "DummyMultiObjectDetector(data).labels @> ['person']"
            query = f'SELECT id FROM MyVideo WHERE {pred_2} AND {pred_1};'
            plan = get_physical_query_plan(self.evadb, query)
            predicate_plans = list(plan.find_all(PredicatePlan))
            self.assertEqual(len(predicate_plans), 1)
            left: ComparisonExpression = predicate_plans[0].predicate.children[0]
            right: ComparisonExpression = predicate_plans[0].predicate.children[1]
            self.assertEqual(left.children[0].name, 'DummyObjectDetector')
            self.assertEqual(right.children[0].name, 'DummyMultiObjectDetector')
        _check_reorder(lambda name: MagicMock(cost=10) if name == 'DummyMultiObjectDetector' else MagicMock(cost=5))
        _check_reorder(lambda name: MagicMock(cost=5) if name == 'DummyObjectDetector' else None)

    @patch('evadb.catalog.catalog_manager.CatalogManager.get_function_cost_catalog_entry')
    def test_should_not_reorder_predicates(self, mock):

        def _check_no_reorder(cost_func):
            mock.side_effect = cost_func
            cheap_pred = "DummyObjectDetector(data).label = ['person']"
            costly_pred = "DummyMultiObjectDetector(data).labels @> ['person']"
            query = f'SELECT id FROM MyVideo WHERE {cheap_pred} AND {costly_pred};'
            plan = get_physical_query_plan(self.evadb, query)
            predicate_plans = list(plan.find_all(PredicatePlan))
            self.assertEqual(len(predicate_plans), 1)
            left: ComparisonExpression = predicate_plans[0].predicate.children[0]
            right: ComparisonExpression = predicate_plans[0].predicate.children[1]
            self.assertEqual(left.children[0].name, 'DummyObjectDetector')
            self.assertEqual(right.children[0].name, 'DummyMultiObjectDetector')
        _check_no_reorder(lambda name: MagicMock(cost=10) if name == 'DummyMultiObjectDetector' else MagicMock(cost=5))
        _check_no_reorder(lambda name: MagicMock(cost=5) if name == 'DummyMultiObjectDetector' else MagicMock(cost=5))
        _check_no_reorder(lambda name: MagicMock(cost=5) if name == 'DummyObjectDetector' else None)
        _check_no_reorder(lambda name: None)

    @patch('evadb.catalog.catalog_manager.CatalogManager.get_function_cost_catalog_entry')
    def test_should_reorder_multiple_predicates(self, mock):

        def side_effect_func(name):
            if name == 'DummyMultiObjectDetector':
                return MagicMock(cost=10)
            else:
                return MagicMock(cost=5)
        mock.side_effect = side_effect_func
        cheapest_pred = 'id<10'
        cheap_pred = "DummyObjectDetector(data).label = ['person']"
        costly_pred = "DummyMultiObjectDetector(data).labels @> ['person']"
        query = f'SELECT id FROM MyVideo WHERE {costly_pred} AND {cheap_pred} AND {cheapest_pred};'
        plan = get_physical_query_plan(self.evadb, query)
        predicate_plans = list(plan.find_all(PredicatePlan))
        self.assertEqual(len(predicate_plans), 1)
        left = predicate_plans[0].predicate.children[0]
        right = predicate_plans[0].predicate.children[1]
        self.assertIsInstance(left, ComparisonExpression)
        self.assertIsInstance(right, ComparisonExpression)
        self.assertEqual(left.children[0].name, 'DummyObjectDetector')
        self.assertEqual(right.children[0].name, 'DummyMultiObjectDetector')

    def test_reorder_rule_should_not_have_side_effects(self):
        query = 'SELECT id FROM MyVideo WHERE id < 20 AND id > 10;'
        result = execute_query_fetch_all(self.evadb, query)
        rules_manager = RulesManager()
        with disable_rules(rules_manager, [ReorderPredicates()]):
            custom_plan_generator = PlanGenerator(self.evadb, rules_manager)
            expected = execute_query_fetch_all(self.evadb, query, plan_generator=custom_plan_generator)
            self.assertEqual(result, expected)

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

class HuggingFaceTests(unittest.TestCase):
    """
    The tests below essentially check for the output format returned by HF.
    We need to ensure that it is in the format that we expect.
    """

    def setUp(self) -> None:
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        query = "LOAD VIDEO 'data/ua_detrac/ua_detrac.mp4' INTO DETRAC;"
        execute_query_fetch_all(self.evadb, query)
        query = "LOAD VIDEO 'data/sample_videos/touchdown.mp4' INTO VIDEOS"
        execute_query_fetch_all(self.evadb, query)
        query = "LOAD PDF 'data/documents/pdf_sample1.pdf' INTO MyPDFs;"
        execute_query_fetch_all(self.evadb, query)
        self.csv_file_path = create_text_csv()

    def tearDown(self) -> None:
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS DETRAC;')
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS VIDEOS;')
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyCSV;')
        file_remove(self.csv_file_path)

    def test_io_catalog_entries_populated(self):
        function_name, task = ('HFObjectDetector', 'image-classification')
        create_function_query = f"CREATE FUNCTION {function_name}\n            TYPE HuggingFace\n            TASK '{task}'\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        catalog = self.evadb.catalog()
        function = catalog.get_function_catalog_entry_by_name(function_name)
        input_entries = catalog.get_function_io_catalog_input_entries(function)
        output_entries = catalog.get_function_io_catalog_output_entries(function)
        self.assertEqual(len(input_entries), 1)
        self.assertEqual(input_entries[0].name, f'{function_name}_IMAGE')
        self.assertEqual(len(output_entries), 2)
        self.assertEqual(output_entries[0].name, 'score')
        self.assertEqual(output_entries[1].name, 'label')

    def test_raise_error_on_unsupported_task(self):
        function_name = 'HFUnsupportedTask'
        task = 'zero-shot-object-detection'
        create_function_query = f"CREATE FUNCTION {function_name}\n            TYPE HuggingFace\n            TASK '{task}'\n        "
        with self.assertRaises(ExecutorError) as exc_info:
            execute_query_fetch_all(self.evadb, create_function_query, do_not_print_exceptions=True)
        self.assertIn(f'Task {task} not supported in EvaDB currently', str(exc_info.exception))

    def test_object_detection(self):
        function_name = 'HFObjectDetector'
        create_function_query = f"CREATE FUNCTION {function_name}\n            TYPE HuggingFace\n            TASK 'object-detection'\n            MODEL 'facebook/detr-resnet-50';\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        select_query = f'SELECT {function_name}(data) FROM DETRAC WHERE id < 4;'
        output = execute_query_fetch_all(self.evadb, select_query)
        output_frames = output.frames
        self.assertEqual(len(output_frames.columns), 3)
        self.assertEqual(len(output.frames), 4)
        self.assertTrue(function_name.lower() + '.score' in output_frames.columns)
        self.assertTrue(all((isinstance(x, list) for x in output.frames[function_name.lower() + '.score'])))
        self.assertTrue(function_name.lower() + '.label' in output_frames.columns)
        self.assertTrue(all((isinstance(x, list) for x in output.frames[function_name.lower() + '.label'])))
        self.assertTrue(function_name.lower() + '.box' in output_frames.columns)
        for bbox in output.frames[function_name.lower() + '.box']:
            self.assertTrue(isinstance(bbox, list))
            bbox = bbox[0]
            self.assertTrue(isinstance(bbox, dict))
            self.assertTrue(len(bbox) == 4)
            self.assertTrue('xmin' in bbox)
            self.assertTrue('ymin' in bbox)
            self.assertTrue('xmax' in bbox)
            self.assertTrue('ymax' in bbox)
        drop_function_query = f'DROP FUNCTION {function_name};'
        execute_query_fetch_all(self.evadb, drop_function_query)

    def test_image_classification(self):
        function_name = 'HFImageClassifier'
        create_function_query = f"CREATE FUNCTION {function_name}\n            TYPE HuggingFace\n            TASK 'image-classification'\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        select_query = f'SELECT {function_name}(data) FROM DETRAC WHERE id < 3;'
        output = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(output.frames.columns), 2)
        self.assertTrue(function_name.lower() + '.score' in output.frames.columns)
        self.assertTrue(all((isinstance(x, list) for x in output.frames[function_name.lower() + '.score'])))
        self.assertTrue(function_name.lower() + '.label' in output.frames.columns)
        self.assertTrue(all((isinstance(x, list) for x in output.frames[function_name.lower() + '.label'])))
        drop_function_query = f'DROP FUNCTION {function_name};'
        execute_query_fetch_all(self.evadb, drop_function_query)

    @pytest.mark.benchmark
    def test_text_classification(self):
        create_table_query = 'CREATE TABLE IF NOT EXISTS MyCSV (\n                id INTEGER UNIQUE,\n                comment TEXT(30)\n            );'
        execute_query_fetch_all(self.evadb, create_table_query)
        load_table_query = f"LOAD CSV '{self.csv_file_path}' INTO MyCSV;"
        execute_query_fetch_all(self.evadb, load_table_query)
        function_name = 'HFTextClassifier'
        create_function_query = f"CREATE FUNCTION {function_name}\n            TYPE HuggingFace\n            TASK 'text-classification'\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        select_query = f'SELECT {function_name}(comment) FROM MyCSV;'
        output = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(output.frames.columns), 2)
        self.assertTrue(function_name.lower() + '.label' in output.frames.columns)
        self.assertTrue(all((x in ['POSITIVE', 'NEGATIVE'] for x in output.frames[function_name.lower() + '.label'])))
        self.assertTrue(function_name.lower() + '.score' in output.frames.columns)
        self.assertTrue(all((isinstance(x, float) for x in output.frames[function_name.lower() + '.score'])))
        drop_function_query = f'DROP FUNCTION {function_name};'
        execute_query_fetch_all(self.evadb, drop_function_query)
        execute_query_fetch_all(self.evadb, 'DROP TABLE MyCSV;')

    @pytest.mark.benchmark
    def test_automatic_speech_recognition(self):
        function_name = 'SpeechRecognizer'
        create_function = f"CREATE FUNCTION {function_name} TYPE HuggingFace TASK 'automatic-speech-recognition' MODEL 'openai/whisper-base';"
        execute_query_fetch_all(self.evadb, create_function)
        select_query = f'SELECT {function_name}(audio) FROM VIDEOS;'
        output = execute_query_fetch_all(self.evadb, select_query)
        self.assertTrue(output.frames.shape == (1, 1))
        self.assertTrue(output.frames.iloc[0][0].count('touchdown') == 2)
        select_query_with_group_by = f"SELECT {function_name}(SEGMENT(audio)) FROM VIDEOS GROUP BY '240 samples';"
        output = execute_query_fetch_all(self.evadb, select_query_with_group_by)
        self.assertEquals(output.frames.shape, (4, 1))
        self.assertEquals(output.frames.iloc[0][0].count('touchdown'), 1)
        drop_function_query = f'DROP FUNCTION {function_name};'
        execute_query_fetch_all(self.evadb, drop_function_query)

    @pytest.mark.benchmark
    def test_summarization_from_video(self):
        asr_function = 'SpeechRecognizer'
        create_function = f"CREATE FUNCTION {asr_function} TYPE HuggingFace TASK 'automatic-speech-recognition' MODEL 'openai/whisper-base';"
        execute_query_fetch_all(self.evadb, create_function)
        summary_function = 'Summarizer'
        create_function = f"CREATE FUNCTION {summary_function} TYPE HuggingFace TASK 'summarization' MODEL 'philschmid/bart-large-cnn-samsum' MIN_LENGTH 10 MAX_NEW_TOKENS 100;"
        execute_query_fetch_all(self.evadb, create_function)
        select_query = f'SELECT {summary_function}({asr_function}(audio)) FROM VIDEOS;'
        output = execute_query_fetch_all(self.evadb, select_query)
        self.assertTrue(output.frames.shape == (1, 1))
        self.assertTrue(output.frames.iloc[0][0] == 'Jalen Hurts has scored his second rushing touchdown of the game.')
        drop_function_query = f'DROP FUNCTION {asr_function};'
        execute_query_fetch_all(self.evadb, drop_function_query)
        drop_function_query = f'DROP FUNCTION {summary_function};'
        execute_query_fetch_all(self.evadb, drop_function_query)

    def test_toxicity_classification(self):
        function_name = 'HFToxicityClassifier'
        create_function_query = f"CREATE FUNCTION {function_name}\n            TYPE HuggingFace\n            TASK 'text-classification'\n            MODEL 'martin-ha/toxic-comment-model'\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        drop_table_query = 'DROP TABLE IF EXISTS MyCSV;'
        execute_query_fetch_all(self.evadb, drop_table_query)
        create_table_query = 'CREATE TABLE IF NOT EXISTS MyCSV (\n                id INTEGER UNIQUE,\n                comment TEXT(30)\n            );'
        execute_query_fetch_all(self.evadb, create_table_query)
        load_table_query = f"LOAD CSV '{self.csv_file_path}' INTO MyCSV;"
        execute_query_fetch_all(self.evadb, load_table_query)
        select_query = f'SELECT {function_name}(comment) FROM MyCSV;'
        output = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(output.frames.columns), 2)
        self.assertTrue(function_name.lower() + '.label' in output.frames.columns)
        self.assertTrue(all((x in ['non-toxic', 'toxic'] for x in output.frames[function_name.lower() + '.label'])))
        self.assertTrue(function_name.lower() + '.score' in output.frames.columns)
        self.assertTrue(all((isinstance(x, float) for x in output.frames[function_name.lower() + '.score'])))
        drop_function_query = f'DROP FUNCTION {function_name};'
        execute_query_fetch_all(self.evadb, drop_function_query)

    @pytest.mark.benchmark
    def test_multilingual_toxicity_classification(self):
        function_name = 'HFMultToxicityClassifier'
        create_function_query = f"CREATE FUNCTION {function_name}\n            TYPE HuggingFace\n            TASK 'text-classification'\n            MODEL 'EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus'\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        drop_table_query = 'DROP TABLE IF EXISTS MyCSV;'
        execute_query_fetch_all(self.evadb, drop_table_query)
        create_table_query = 'CREATE TABLE MyCSV (\n                id INTEGER UNIQUE,\n                comment TEXT(30)\n            );'
        execute_query_fetch_all(self.evadb, create_table_query)
        load_table_query = f"LOAD CSV '{self.csv_file_path}' INTO MyCSV;"
        execute_query_fetch_all(self.evadb, load_table_query)
        select_query = f'SELECT {function_name}(comment) FROM MyCSV;'
        output = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(output.frames.columns), 2)
        self.assertTrue(function_name.lower() + '.label' in output.frames.columns)
        self.assertTrue(all((x in ['LABEL_1', 'LABEL_0'] for x in output.frames[function_name.lower() + '.label'])))
        self.assertTrue(function_name.lower() + '.score' in output.frames.columns)
        self.assertTrue(all((isinstance(x, float) for x in output.frames[function_name.lower() + '.score'])))
        drop_function_query = f'DROP FUNCTION {function_name};'
        execute_query_fetch_all(self.evadb, drop_function_query)

    @pytest.mark.benchmark
    def test_named_entity_recognition_model_all_pdf_data(self):
        function_name = 'HFNERModel'
        create_function_query = f"CREATE FUNCTION {function_name}\n            TYPE HuggingFace\n            TASK 'ner'\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        select_query = f'SELECT data, {function_name}(data) FROM MyPDFs;'
        output = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(output.frames.columns), 7)
        self.assertTrue(function_name.lower() + '.entity' in output.frames.columns)
        self.assertTrue(function_name.lower() + '.score' in output.frames.columns)
        drop_function_query = f'DROP FUNCTION {function_name};'
        execute_query_fetch_all(self.evadb, drop_function_query)

    def test_select_and_groupby_with_paragraphs(self):
        segment_size = 10
        select_query = "SELECT SEGMENT(data) FROM MyPDFs GROUP BY '{}paragraphs';".format(segment_size)
        output = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(output.frames), 3)

    @pytest.mark.benchmark
    def test_named_entity_recognition_model_no_ner_data_exists(self):
        function_name = 'HFNERModel'
        create_function_query = f"CREATE FUNCTION {function_name}\n            TYPE HuggingFace\n            TASK 'ner'\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        select_query = f'SELECT data, {function_name}(data)\n                  FROM MyPDFs\n                  WHERE page = 3\n                  AND paragraph >= 1 AND paragraph <= 3;'
        output = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(output.frames.columns), 1)
        self.assertFalse(function_name.lower() + '.entity' in output.frames.columns)
        drop_function_query = f'DROP FUNCTION {function_name};'
        execute_query_fetch_all(self.evadb, drop_function_query)

class ReuseTest(unittest.TestCase):

    def _load_hf_model(self):
        function_name = 'HFObjectDetector'
        create_function_query = f"CREATE FUNCTION {function_name}\n            TYPE HuggingFace\n            TASK 'object-detection'\n            MODEL 'facebook/detr-resnet-50';\n        "
        execute_query_fetch_all(self.evadb, create_function_query)

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        ua_detrac = f'{EvaDB_ROOT_DIR}/data/ua_detrac/ua_detrac.mp4'
        execute_query_fetch_all(self.evadb, f"LOAD VIDEO '{ua_detrac}' INTO DETRAC;")
        execute_query_fetch_all(self.evadb, 'CREATE TABLE fruitTable (data TEXT(100))')
        data_list = ['The color of apple is red', 'The color of banana is yellow']
        for data in data_list:
            execute_query_fetch_all(self.evadb, f"INSERT INTO fruitTable (data) VALUES ('{data}')")
        load_functions_for_testing(self.evadb)
        self._load_hf_model()

    def tearDown(self):
        shutdown_ray()
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS DETRAC;')
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS fruitTable;')

    def _verify_reuse_correctness(self, query, reuse_batch):
        gc.collect()
        rules_manager = RulesManager()
        with disable_rules(rules_manager, [CacheFunctionExpressionInApply(), CacheFunctionExpressionInFilter(), CacheFunctionExpressionInProject()]):
            custom_plan_generator = PlanGenerator(self.evadb, rules_manager)
            without_reuse_batch = execute_query_fetch_all(self.evadb, query, plan_generator=custom_plan_generator)
        self.assertEqual(reuse_batch.columns, reuse_batch.columns)
        reuse_batch.sort_orderby(by=[reuse_batch.columns[0]])
        without_reuse_batch.sort_orderby(by=[reuse_batch.columns[0]])
        self.assertEqual(without_reuse_batch, reuse_batch, msg=f'Without reuse {without_reuse_batch} \n With reuse{reuse_batch}')

    def _reuse_experiment(self, queries):
        exec_times = []
        batches = []
        for query in queries:
            timer = Timer()
            with timer:
                batches.append(execute_query_fetch_all(self.evadb, query))
            exec_times.append(timer.total_elapsed_time)
        return (batches, exec_times)

    def _strict_reuse_experiment(self, queries):
        exec_times = []
        batches = []
        for i, query in enumerate(queries):
            timer = Timer()
            if i != 0:
                with timer, patch.object(Batch, 'apply_function_expression') as mock_batch_func:
                    mock_batch_func.side_effect = Exception('Results are not reused')
                    batches.append(execute_query_fetch_all(self.evadb, query))
            else:
                with timer:
                    batches.append(execute_query_fetch_all(self.evadb, query))
            exec_times.append(timer.total_elapsed_time)
        return (batches, exec_times)

    def test_reuse_chatgpt(self):
        from evadb.constants import CACHEABLE_FUNCTIONS
        CACHEABLE_FUNCTIONS += ['DummyLLM']
        select_query = "SELECT DummyLLM('What is the fruit described in this sentence', data)\n            FROM fruitTable"
        batches, exec_times = self._strict_reuse_experiment([select_query, select_query])
        self._verify_reuse_correctness(select_query, batches[1])
        self.assertTrue(exec_times[0] > exec_times[1])

    def test_reuse_when_query_is_duplicate(self):
        select_query = 'SELECT id, label FROM DETRAC JOIN\n            LATERAL HFObjectDetector(data) AS Obj(score, label, bbox) WHERE id < 15;'
        batches, exec_times = self._strict_reuse_experiment([select_query, select_query])
        self._verify_reuse_correctness(select_query, batches[1])
        self.assertTrue(exec_times[0] > exec_times[1])

    @gpu_skip_marker
    def test_reuse_partial(self):
        select_query1 = 'SELECT id, label FROM DETRAC JOIN\n            LATERAL HFObjectDetector(data) AS Obj(score, label, bbox) WHERE id < 5;'
        select_query2 = 'SELECT id, label FROM DETRAC JOIN\n            LATERAL HFObjectDetector(data) AS Obj(score, label, bbox) WHERE id < 15;'
        batches, exec_times = self._reuse_experiment([select_query1, select_query2])
        self._verify_reuse_correctness(select_query2, batches[1])

    @gpu_skip_marker
    def test_reuse_in_with_multiple_occurrences(self):
        select_query1 = 'SELECT id, label FROM DETRAC JOIN\n            LATERAL HFObjectDetector(data) AS Obj(score, label, bbox) WHERE id < 10;'
        select_query2 = 'SELECT id, HFObjectDetector(data).label FROM DETRAC JOIN\n            LATERAL HFObjectDetector(data) AS Obj(score, label, bbox) WHERE id < 5;'
        batches, exec_times = self._reuse_experiment([select_query1, select_query2])
        self._verify_reuse_correctness(select_query2, batches[1])
        select_query = 'SELECT id, HFObjectDetector(data).label FROM DETRAC WHERE id < 15;'
        reuse_batch = execute_query_fetch_all(self.evadb, select_query)
        self._verify_reuse_correctness(select_query, reuse_batch)
        select_query = "SELECT id, HFObjectDetector(data).label FROM DETRAC WHERE ['car'] <@ HFObjectDetector(data).label AND id < 20"
        reuse_batch = execute_query_fetch_all(self.evadb, select_query)
        self._verify_reuse_correctness(select_query, reuse_batch)

    @gpu_skip_marker
    def test_reuse_logical_project_with_duplicate_query(self):
        project_query = 'SELECT id, HFObjectDetector(data).label FROM DETRAC WHERE id < 10;'
        batches, exec_times = self._reuse_experiment([project_query, project_query])
        self._verify_reuse_correctness(project_query, batches[1])
        self.assertGreater(exec_times[0], exec_times[1])

    @gpu_skip_marker
    def test_reuse_with_function_in_predicate(self):
        select_query = "SELECT id FROM DETRAC WHERE ['car'] <@ HFObjectDetector(data).label AND id < 4"
        batches, exec_times = self._reuse_experiment([select_query, select_query])
        self._verify_reuse_correctness(select_query, batches[1])
        self.assertGreater(exec_times[0], exec_times[1])

    @gpu_skip_marker
    def test_reuse_across_different_predicate_using_same_function(self):
        query1 = "SELECT id FROM DETRAC WHERE ['car'] <@ HFObjectDetector(data).label AND id < 15"
        query2 = "SELECT id FROM DETRAC WHERE ArrayCount(HFObjectDetector(data).label, 'car') > 3 AND id < 12;"
        batches, exec_times = self._reuse_experiment([query1, query2])
        self._verify_reuse_correctness(query2, batches[1])
        self.assertGreater(exec_times[0], exec_times[1])

    @gpu_skip_marker
    def test_reuse_filter_with_project(self):
        project_query = '\n            SELECT id, Yolo(data).labels FROM DETRAC WHERE id < 5;'
        select_query = "\n            SELECT id FROM DETRAC\n            WHERE ArrayCount(Yolo(data).labels, 'car') > 3 AND id < 5;"
        batches, exec_times = self._reuse_experiment([project_query, select_query])
        self._verify_reuse_correctness(select_query, batches[1])
        self.assertGreater(exec_times[0], exec_times[1])

    @gpu_skip_marker
    def test_reuse_in_extract_object(self):
        select_query = '\n            SELECT id, T.iids, T.bboxes, T.scores, T.labels\n            FROM DETRAC JOIN LATERAL EXTRACT_OBJECT(data, Yolo, NorFairTracker)\n                AS T(iids, labels, bboxes, scores)\n            WHERE id < 30;\n            '
        batches, exec_times = self._reuse_experiment([select_query, select_query])
        self._verify_reuse_correctness(select_query, batches[1])
        self.assertGreater(exec_times[0], exec_times[1])

    @windows_skip_marker
    def test_reuse_after_server_shutdown(self):
        select_query = 'SELECT id, label FROM DETRAC JOIN\n            LATERAL Yolo(data) AS Obj(label, bbox, conf) WHERE id < 4;'
        execute_query_fetch_all(self.evadb, select_query)
        os.system('nohup evadb_server --stop')
        os.system('nohup evadb_server --start &')
        select_query = 'SELECT id, label FROM DETRAC JOIN\n            LATERAL Yolo(data) AS Obj(label, bbox, conf) WHERE id < 6;'
        reuse_batch = execute_query_fetch_all(self.evadb, select_query)
        self._verify_reuse_correctness(select_query, reuse_batch)
        os.system('nohup evadb_server --stop')

    def test_drop_function_should_remove_cache(self):
        select_query = 'SELECT id, label FROM DETRAC JOIN\n            LATERAL Yolo(data) AS Obj(label, bbox, conf) WHERE id < 5;'
        execute_query_fetch_all(self.evadb, select_query)
        plan = next(get_logical_query_plan(self.evadb, select_query).find_all(LogicalFunctionScan))
        cache_name = plan.func_expr.signature()
        function_cache = self.evadb.catalog().get_function_cache_catalog_entry_by_name(cache_name)
        cache_dir = Path(function_cache.cache_path)
        self.assertIsNotNone(function_cache)
        self.assertTrue(cache_dir.exists())
        execute_query_fetch_all(self.evadb, 'DROP FUNCTION Yolo;')
        function_cache = self.evadb.catalog().get_function_cache_catalog_entry_by_name(cache_name)
        self.assertIsNone(function_cache)
        self.assertFalse(cache_dir.exists())

    def test_drop_table_should_remove_cache(self):
        select_query = 'SELECT id, label FROM DETRAC JOIN\n            LATERAL Yolo(data) AS Obj(label, bbox, conf) WHERE id < 5;'
        execute_query_fetch_all(self.evadb, select_query)
        plan = next(get_logical_query_plan(self.evadb, select_query).find_all(LogicalFunctionScan))
        cache_name = plan.func_expr.signature()
        function_cache = self.evadb.catalog().get_function_cache_catalog_entry_by_name(cache_name)
        cache_dir = Path(function_cache.cache_path)
        self.assertIsNotNone(function_cache)
        self.assertTrue(cache_dir.exists())
        execute_query_fetch_all(self.evadb, 'DROP TABLE DETRAC;')
        function_cache = self.evadb.catalog().get_function_cache_catalog_entry_by_name(cache_name)
        self.assertIsNone(function_cache)
        self.assertFalse(cache_dir.exists())

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

@pytest.mark.notparallel
class SaliencyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_dir = suffix_pytest_xdist_worker_id_to_dir(EvaDB_DATABASE_DIR)
        cls.conn = connect(cls.db_dir)
        cls.evadb = cls.conn._evadb

    @classmethod
    def tearDownClass(cls):
        shutdown_ray()
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS SALIENCY;')

    @unittest.skip('Not supported in current version')
    def test_saliency(self):
        Saliency1 = f'{EvaDB_ROOT_DIR}/data/saliency/test1.jpeg'
        create_function_query = f"LOAD IMAGE '{Saliency1}' INTO SALIENCY;"
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS SALIENCY;')
        execute_query_fetch_all(self.evadb, create_function_query)
        execute_query_fetch_all(self.evadb, 'DROP FUNCTION IF EXISTS SaliencyFeatureExtractor')
        create_function_query = f"CREATE FUNCTION IF NOT EXISTS SaliencyFeatureExtractor\n                    IMPL  '{EvaDB_ROOT_DIR}/evadb/functions/saliency_feature_extractor.py';\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        select_query_saliency = 'SELECT data, SaliencyFeatureExtractor(data)\n                  FROM SALIENCY\n        '
        actual_batch_saliency = execute_query_fetch_all(self.evadb, select_query_saliency)
        self.assertEqual(len(actual_batch_saliency.columns), 2)

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

@pytest.mark.notparallel
class ModelTrainTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        create_table_query = '\n           CREATE TABLE IF NOT EXISTS HomeRentals (\n               number_of_rooms INTEGER,\n               number_of_bathrooms INTEGER,\n               sqft INTEGER,\n               location TEXT(128),\n               days_on_market INTEGER,\n               initial_price INTEGER,\n               neighborhood TEXT(128),\n               rental_price FLOAT(64,64)\n           );'
        execute_query_fetch_all(cls.evadb, create_table_query)
        path = f'{EvaDB_ROOT_DIR}/data/ludwig/home_rentals.csv'
        load_query = f"LOAD CSV '{path}' INTO HomeRentals;"
        execute_query_fetch_all(cls.evadb, load_query)
        create_table_query = '\n           CREATE TABLE IF NOT EXISTS Employee (\n               education TEXT(128),\n               joining_year INTEGER,\n               city TEXT(128),\n               payment_tier INTEGER,\n               age INTEGER,\n               gender TEXT(128),\n               ever_benched TEXT(128),\n               experience_in_current_domain INTEGER,\n               leave_or_not INTEGER\n           );'
        execute_query_fetch_all(cls.evadb, create_table_query)
        path = f'{EvaDB_ROOT_DIR}/data/classification/Employee.csv'
        load_query = f"LOAD CSV '{path}' INTO Employee;"
        execute_query_fetch_all(cls.evadb, load_query)

    @classmethod
    def tearDownClass(cls):
        shutdown_ray()
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS HomeRentals;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS Employee;')
        execute_query_fetch_all(cls.evadb, 'DROP FUNCTION IF EXISTS PredictHouseRentLudwig;')
        execute_query_fetch_all(cls.evadb, 'DROP FUNCTION IF EXISTS PredictHouseRentSklearn;')
        execute_query_fetch_all(cls.evadb, 'DROP FUNCTION IF EXISTS PredictRentXgboost;')
        execute_query_fetch_all(cls.evadb, 'DROP FUNCTION IF EXISTS PredictEmployeeXgboost;')

    @pytest.mark.skip(reason='Model training intergration test takes too long to complete.')
    @ludwig_skip_marker
    def test_ludwig_automl(self):
        create_predict_function = "\n            CREATE OR REPLACE FUNCTION PredictHouseRentLudwig FROM\n            ( SELECT * FROM HomeRentals )\n            TYPE Ludwig\n            PREDICT 'rental_price'\n            TIME_LIMIT 120;\n        "
        execute_query_fetch_all(self.evadb, create_predict_function)
        predict_query = '\n            SELECT PredictHouseRentLudwig(*) FROM HomeRentals LIMIT 10;\n        '
        result = execute_query_fetch_all(self.evadb, predict_query)
        self.assertEqual(len(result.columns), 1)
        self.assertEqual(len(result), 10)

    @pytest.mark.skip(reason='Model training intergration test takes too long to complete.')
    @sklearn_skip_marker
    def test_sklearn_regression(self):
        create_predict_function = "\n            CREATE OR REPLACE FUNCTION PredictHouseRentSklearn FROM\n            ( SELECT number_of_rooms, number_of_bathrooms, days_on_market, rental_price FROM HomeRentals )\n            TYPE Sklearn\n            PREDICT 'rental_price'\n            MODEL 'extra_tree'\n            METRIC 'r2';\n        "
        execute_query_fetch_all(self.evadb, create_predict_function)
        predict_query = '\n            SELECT PredictHouseRentSklearn(number_of_rooms, number_of_bathrooms, days_on_market, rental_price) FROM HomeRentals LIMIT 10;\n        '
        result = execute_query_fetch_all(self.evadb, predict_query)
        self.assertEqual(len(result.columns), 1)
        self.assertEqual(len(result), 10)

    @xgboost_skip_marker
    def test_xgboost_regression(self):
        create_predict_function = "\n            CREATE OR REPLACE FUNCTION PredictRentXgboost FROM\n            ( SELECT number_of_rooms, number_of_bathrooms, days_on_market, rental_price FROM HomeRentals )\n            TYPE XGBoost\n            PREDICT 'rental_price'\n            TIME_LIMIT 180\n            METRIC 'r2'\n            TASK 'regression';\n        "
        result = execute_query_fetch_all(self.evadb, create_predict_function)
        self.assertEqual(len(result.columns), 1)
        self.assertEqual(len(result), 3)
        predict_query = '\n            SELECT PredictRentXgboost(number_of_rooms, number_of_bathrooms, days_on_market, rental_price) FROM HomeRentals LIMIT 10;\n        '
        result = execute_query_fetch_all(self.evadb, predict_query)
        self.assertEqual(len(result.columns), 1)
        self.assertEqual(len(result), 10)

    @xgboost_skip_marker
    def test_xgboost_classification(self):
        create_predict_function = "\n            CREATE OR REPLACE FUNCTION PredictEmployeeXgboost FROM\n            ( SELECT payment_tier, age, gender, experience_in_current_domain, leave_or_not FROM Employee )\n            TYPE XGBoost\n            PREDICT 'leave_or_not'\n            TIME_LIMIT 180\n            METRIC 'accuracy'\n            TASK 'classification';\n        "
        result = execute_query_fetch_all(self.evadb, create_predict_function)
        self.assertEqual(len(result.columns), 1)
        self.assertEqual(len(result), 3)
        predict_query = '\n            SELECT PredictEmployeeXgboost(payment_tier, age, gender, experience_in_current_domain, leave_or_not) FROM Employee LIMIT 10;\n        '
        result = execute_query_fetch_all(self.evadb, predict_query)
        self.assertEqual(len(result.columns), 1)
        self.assertEqual(len(result), 10)

@pytest.mark.notparallel
class FuzzyJoinTests(unittest.TestCase):

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        self.video_file_path = create_sample_video()
        self.image_files_path = Path(f'{EvaDB_ROOT_DIR}/test/data/uadetrac/small-data/MVI_20011/*.jpg')
        self.csv_file_path = create_sample_csv()
        create_table_query = '\n            CREATE TABLE IF NOT EXISTS MyVideoCSV (\n                id INTEGER UNIQUE,\n                frame_id INTEGER,\n                video_id INTEGER,\n                dataset_name TEXT(30),\n                label TEXT(30),\n                bbox NDARRAY FLOAT32(4),\n                object_id INTEGER\n            );\n            '
        execute_query_fetch_all(self.evadb, create_table_query)
        load_query = f"LOAD CSV '{self.csv_file_path}' INTO MyVideoCSV;"
        execute_query_fetch_all(self.evadb, load_query)
        query = f"LOAD VIDEO '{self.video_file_path}' INTO MyVideo;"
        execute_query_fetch_all(self.evadb, query)

    def tearDown(self):
        shutdown_ray()
        file_remove('dummy.avi')
        file_remove('dummy.csv')
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyVideo;')
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyVideoCSV;')

    @pytest.mark.xfail(reason='https://github.com/georgia-tech-db/evadb/issues/1242')
    def test_fuzzyjoin(self):
        execute_query_fetch_all(self.evadb, fuzzy_function_query)
        fuzzy_join_query = 'SELECT * FROM MyVideo a JOIN MyVideoCSV b\n                      ON FuzzDistance(a.id, b.id) = 100;'
        actual_batch = execute_query_fetch_all(self.evadb, fuzzy_join_query)
        self.assertEqual(len(actual_batch), 10)

@pytest.mark.notparallel
class ExplainExecutorTest(unittest.TestCase):

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
        file_remove('dummy.avi')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS MyVideo;')

    def test_explain_simple_select(self):
        select_query = 'EXPLAIN SELECT id, data FROM MyVideo'
        batch = execute_query_fetch_all(self.evadb, select_query)
        expected_output = '|__ ProjectPlan\n    |__ SeqScanPlan\n        |__ StoragePlan\n'
        self.assertEqual(batch.frames[0][0], expected_output)
        rules_manager = RulesManager()
        with disable_rules(rules_manager, [XformLateralJoinToLinearFlow()]):
            custom_plan_generator = PlanGenerator(self.evadb, rules_manager)
            select_query = 'EXPLAIN SELECT id, data FROM MyVideo JOIN LATERAL DummyObjectDetector(data) AS T ;'
            batch = execute_query_fetch_all(self.evadb, select_query, plan_generator=custom_plan_generator)
            expected_output = '|__ ProjectPlan\n    |__ LateralJoinPlan\n        |__ SeqScanPlan\n            |__ StoragePlan\n        |__ FunctionScanPlan\n'
            self.assertEqual(batch.frames[0][0], expected_output)
        rules_manager = RulesManager()
        with disable_rules(rules_manager, [XformLateralJoinToLinearFlow(), EmbedFilterIntoGet(), LogicalInnerJoinCommutativity()]):
            custom_plan_generator = PlanGenerator(self.evadb, rules_manager)
            select_query = 'EXPLAIN SELECT id, data FROM MyVideo JOIN LATERAL DummyObjectDetector(data) AS T ;'
            batch = execute_query_fetch_all(self.evadb, select_query, plan_generator=custom_plan_generator)
            expected_output = '|__ ProjectPlan\n    |__ LateralJoinPlan\n        |__ SeqScanPlan\n            |__ StoragePlan\n        |__ FunctionScanPlan\n'
            self.assertEqual(batch.frames[0][0], expected_output)

class ErrorHandlingRayTests(unittest.TestCase):

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        os.environ['ray'] = str(self.evadb.catalog().get_configuration_catalog_value('ray'))
        self.evadb.catalog().reset()
        load_functions_for_testing(self.evadb, mode='debug')
        img_path = create_sample_image()
        execute_query_fetch_all(self.evadb, f"LOAD IMAGE '{img_path}' INTO testRayErrorHandling;")
        Path(img_path).unlink()

    def tearDown(self):
        shutdown_ray()
        drop_table_query = 'DROP TABLE testRayErrorHandling;'
        execute_query_fetch_all(self.evadb, drop_table_query)

    @ray_skip_marker
    def test_ray_error_populate_to_all_stages(self):
        function_name, task = ('HFObjectDetector', 'image-classification')
        create_function_query = f"CREATE FUNCTION {function_name}\n            TYPE HuggingFace\n            TASK '{task}'\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        select_query = 'SELECT HFObjectDetector(data) FROM testRayErrorHandling;'
        with self.assertRaises(ExecutorError):
            _ = execute_query_fetch_all(self.evadb, select_query)
        time.sleep(3)
        self.assertFalse(is_ray_stage_running())

class LikeTest(unittest.TestCase):

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        meme1 = f'{EvaDB_ROOT_DIR}/data/detoxify/meme1.jpg'
        meme2 = f'{EvaDB_ROOT_DIR}/data/detoxify/meme2.jpg'
        execute_query_fetch_all(self.evadb, f"LOAD IMAGE '{meme1}' INTO MemeImages;")
        execute_query_fetch_all(self.evadb, f"LOAD IMAGE '{meme2}' INTO MemeImages;")

    def tearDown(self):
        shutdown_ray()
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MemeImages;')

    @ocr_skip_marker
    def test_like_with_ocr(self):
        create_function_query = "CREATE FUNCTION IF NOT EXISTS OCRExtractor\n                  INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))\n                  OUTPUT (labels NDARRAY STR(10),\n                          bboxes NDARRAY FLOAT32(ANYDIM, 4),\n                          scores NDARRAY FLOAT32(ANYDIM))\n                  TYPE  OCRExtraction\n                  IMPL  'evadb/functions/ocr_extractor.py';\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        select_query = 'SELECT X.label, X.x, X.y FROM MemeImages JOIN LATERAL UNNEST(OCRExtractor(data)) AS X(label, x, y) WHERE label LIKE {};'.format("'.*SWAG.*'")
        actual_batch = execute_query_fetch_all(self.evadb, select_query)
        self.assertEqual(len(actual_batch), 1)

    @ocr_skip_marker
    def test_like_fails_on_non_string_col(self):
        create_function_query = "CREATE FUNCTION IF NOT EXISTS OCRExtractor\n                  INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))\n                  OUTPUT (labels NDARRAY STR(10),\n                          bboxes NDARRAY FLOAT32(ANYDIM, 4),\n                          scores NDARRAY FLOAT32(ANYDIM))\n                  TYPE  OCRExtraction\n                  IMPL  'evadb/functions/ocr_extractor.py';\n        "
        execute_query_fetch_all(self.evadb, create_function_query)
        select_query = 'SELECT * FROM MemeImages JOIN LATERAL UNNEST(OCRExtractor(data)) AS X(label, x, y) WHERE x LIKE "[A-Za-z]*CANT";'
        with self.assertRaises(Exception):
            execute_query_fetch_all(self.evadb, select_query)

@pytest.mark.notparallel
class ModelTrainTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        create_table_query = '\n            CREATE TABLE AirData (            unique_id TEXT(30),            ds TEXT(30),            y INTEGER);'
        execute_query_fetch_all(cls.evadb, create_table_query)
        create_table_query = '\n            CREATE TABLE AirDataPanel (            unique_id TEXT(30),            ds TEXT(30),            y INTEGER,            trend INTEGER,            ylagged INTEGER);'
        execute_query_fetch_all(cls.evadb, create_table_query)
        create_table_query = '\n            CREATE TABLE HomeData (            saledate TEXT(30),            ma INTEGER,\n            type TEXT(30),            bedrooms INTEGER);'
        execute_query_fetch_all(cls.evadb, create_table_query)
        path = f'{EvaDB_ROOT_DIR}/data/forecasting/air-passengers.csv'
        load_query = f"LOAD CSV '{path}' INTO AirData;"
        execute_query_fetch_all(cls.evadb, load_query)
        path = f'{EvaDB_ROOT_DIR}/data/forecasting/AirPassengersPanel.csv'
        load_query = f"LOAD CSV '{path}' INTO AirDataPanel;"
        execute_query_fetch_all(cls.evadb, load_query)
        path = f'{EvaDB_ROOT_DIR}/data/forecasting/home_sales.csv'
        load_query = f"LOAD CSV '{path}' INTO HomeData;"
        execute_query_fetch_all(cls.evadb, load_query)

    @classmethod
    def tearDownClass(cls):
        shutdown_ray()
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS AirData;')
        execute_query_fetch_all(cls.evadb, 'DROP TABLE IF EXISTS HomeData;')
        execute_query_fetch_all(cls.evadb, 'DROP FUNCTION IF EXISTS AirForecast;')
        execute_query_fetch_all(cls.evadb, 'DROP FUNCTION IF EXISTS HomeForecast;')

    @forecast_skip_marker
    def test_forecast(self):
        create_predict_udf = "\n            CREATE FUNCTION AirForecast FROM\n            (SELECT unique_id, ds, y FROM AirData)\n            TYPE Forecasting\n            HORIZON 12\n            PREDICT 'y';\n        "
        execute_query_fetch_all(self.evadb, create_predict_udf)
        predict_query = '\n            SELECT AirForecast() order by y;\n        '
        result = execute_query_fetch_all(self.evadb, predict_query)
        self.assertEqual(len(result), 12)
        self.assertEqual(result.columns, ['airforecast.unique_id', 'airforecast.ds', 'airforecast.y', 'airforecast.y-lo', 'airforecast.y-hi'])

    @pytest.mark.skip(reason='Neuralforecast intergration test takes too long to complete without GPU.')
    @forecast_skip_marker
    def test_forecast_neuralforecast(self):
        create_predict_udf = "\n            CREATE FUNCTION AirPanelForecast FROM\n            (SELECT unique_id, ds, y, trend FROM AirDataPanel)\n            TYPE Forecasting\n            HORIZON 12\n            PREDICT 'y'\n            LIBRARY 'neuralforecast'\n            AUTO 'false'\n            FREQUENCY 'M';\n        "
        execute_query_fetch_all(self.evadb, create_predict_udf)
        predict_query = '\n            SELECT AirPanelForecast() order by y;\n        '
        result = execute_query_fetch_all(self.evadb, predict_query)
        self.assertEqual(len(result), 24)
        self.assertEqual(result.columns, ['airpanelforecast.unique_id', 'airpanelforecast.ds', 'airpanelforecast.y', 'airpanelforecast.y-lo', 'airpanelforecast.y-hi'])

    @forecast_skip_marker
    def test_forecast_with_column_rename(self):
        create_predict_udf = "\n            CREATE FUNCTION HomeForecast FROM\n            (\n                SELECT type, saledate, ma FROM HomeData\n                WHERE bedrooms = 2\n            )\n            TYPE Forecasting\n            HORIZON 12\n            PREDICT 'ma'\n            ID 'type'\n            TIME 'saledate'\n            FREQUENCY 'M';\n        "
        execute_query_fetch_all(self.evadb, create_predict_udf)
        predict_query = '\n            SELECT HomeForecast();\n        '
        result = execute_query_fetch_all(self.evadb, predict_query)
        self.assertEqual(len(result), 24)
        self.assertEqual(result.columns, ['homeforecast.type', 'homeforecast.saledate', 'homeforecast.ma', 'homeforecast.ma-lo', 'homeforecast.ma-hi'])

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

class CreateJobTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        cls.job_name = 'test_async_job'

    def setUp(self):
        execute_query_fetch_all(self.evadb, f'DROP JOB IF EXISTS {self.job_name};')

    @classmethod
    def tearDownClass(cls):
        shutdown_ray()
        execute_query_fetch_all(cls.evadb, f'DROP JOB IF EXISTS {cls.job_name};')

    def test_invalid_query_in_job_should_raise_exception(self):
        query = f"CREATE JOB {self.job_name} AS {{\n                    CREATE OR REPLACE FUNCTION HomeSalesForecast FROM\n                        ( SELECT * FROM postgres_data.home_sales\n                    TYPE Forecasting\n                    PREDICT 'price';\n                }}\n                START '2023-04-01 01:10:00'\n                END '2023-05-01'\n                EVERY 2 week;\n            "
        with self.assertRaisesRegex(Exception, 'Failed to parse the job query'):
            execute_query_fetch_all(self.evadb, query)

    def test_create_job_should_add_the_entry(self):
        queries = ["CREATE OR REPLACE FUNCTION HomeSalesForecast FROM\n                ( SELECT * FROM postgres_data.home_sales )\n                TYPE Forecasting\n                PREDICT 'price';", 'Select HomeSalesForecast(10);']
        start = '2023-04-01 01:10:00'
        end = '2023-05-01'
        repeat_interval = 2
        repeat_period = 'week'
        all_queries = ''.join(queries)
        query = f"CREATE JOB {self.job_name} AS {{\n                   {all_queries}\n                }}\n                START '{start}'\n                END '{end}'\n                EVERY {repeat_interval} {repeat_period};"
        execute_query_fetch_all(self.evadb, query)
        datetime_format = '%Y-%m-%d %H:%M:%S'
        date_format = '%Y-%m-%d'
        job_entry = self.evadb.catalog().get_job_catalog_entry(self.job_name)
        self.assertEqual(job_entry.name, self.job_name)
        self.assertEqual(job_entry.start_time, datetime.strptime(start, datetime_format))
        self.assertEqual(job_entry.end_time, datetime.strptime(end, date_format))
        self.assertEqual(job_entry.repeat_interval, 2 * 7 * 24 * 60 * 60)
        self.assertEqual(job_entry.active, True)
        self.assertEqual(len(job_entry.queries), len(queries))

    def test_should_create_job_with_if_not_exists(self):
        if_not_exists = 'IF NOT EXISTS'
        queries = ["CREATE OR REPLACE FUNCTION HomeSalesForecast FROM\n                ( SELECT * FROM postgres_data.home_sales )\n                TYPE Forecasting\n                PREDICT 'price';", 'Select HomeSalesForecast(10);']
        query = "CREATE JOB {} {} AS {{\n                    {}\n                }}\n                START '2023-04-01'\n                END '2023-05-01'\n                EVERY 2 week;\n            "
        execute_query_fetch_all(self.evadb, query.format(if_not_exists, self.job_name, ''.join(queries)))
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, query.format('', self.job_name, ''.join(queries)))
        execute_query_fetch_all(self.evadb, query.format(if_not_exists, self.job_name, ''.join(queries)))

class CreateDatabaseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        cls.db_path = f'{os.path.dirname(os.path.abspath(__file__))}/testing.db'

    @classmethod
    def tearDownClass(cls):
        shutdown_ray()
        execute_query_fetch_all(cls.evadb, 'DROP DATABASE IF EXISTS test_data_source;')
        execute_query_fetch_all(cls.evadb, 'DROP DATABASE IF EXISTS demo;')
        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)

    def test_create_database_should_add_the_entry(self):
        params = {'user': 'user', 'password': 'password', 'host': '127.0.0.1', 'port': '5432', 'database': 'demo'}
        query = 'CREATE DATABASE demo_db\n                    WITH ENGINE = "postgres",\n                    PARAMETERS = {};'.format(params)
        with patch('evadb.executor.create_database_executor.get_database_handler'):
            execute_query_fetch_all(self.evadb, query)
        db_entry = self.evadb.catalog().get_database_catalog_entry('demo_db')
        self.assertEqual(db_entry.name, 'demo_db')
        self.assertEqual(db_entry.engine, 'postgres')
        self.assertEqual(db_entry.params, params)

    def test_should_create_sqlite_database(self):
        import os
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        database_path = f'{current_file_dir}/testing.db'
        if_not_exists = 'IF NOT EXISTS'
        params = {'database': database_path}
        query = 'CREATE DATABASE {} test_data_source\n                    WITH ENGINE = "sqlite",\n                    PARAMETERS = {};'
        execute_query_fetch_all(self.evadb, query.format(if_not_exists, params))
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, query.format('', params))
        execute_query_fetch_all(self.evadb, query.format(if_not_exists, params))

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
class DropObjectExecutorTest(unittest.TestCase):

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        self.video_file_path = create_sample_video()

    def tearDown(self):
        file_remove('dummy.avi')

    def _create_index(self, index_name):
        import numpy as np
        feat1 = np.array([[0, 0, 0]]).astype(np.float32)
        feat2 = np.array([[100, 100, 100]]).astype(np.float32)
        feat3 = np.array([[200, 200, 200]]).astype(np.float32)
        execute_query_fetch_all(self.evadb, 'create table if not exists testCreateIndexFeatTable (\n                feat NDARRAY FLOAT32(1,3)\n            );')
        feat_batch_data = Batch(pd.DataFrame(data={'feat': [feat1, feat2, feat3]}))
        feat_tb_entry = self.evadb.catalog().get_table_catalog_entry('testCreateIndexFeatTable')
        storage_engine = StorageEngine.factory(self.evadb, feat_tb_entry)
        storage_engine.write(feat_tb_entry, feat_batch_data)
        query = f'CREATE INDEX {index_name} ON testCreateIndexFeatTable (feat) USING FAISS;'
        execute_query_fetch_all(self.evadb, query)

    def test_should_drop_table(self):
        query = f"LOAD VIDEO '{self.video_file_path}' INTO MyVideo;"
        execute_query_fetch_all(self.evadb, query)
        table_catalog_entry = self.evadb.catalog().get_table_catalog_entry('MyVideo')
        video_dir = table_catalog_entry.file_url
        self.assertFalse(table_catalog_entry is None)
        column_objects = self.evadb.catalog().get_column_catalog_entries_by_table(table_catalog_entry)
        self.assertEqual(len(column_objects), len(get_video_table_column_definitions()) + 1)
        self.assertTrue(Path(video_dir).exists())
        video_metadata_table = self.evadb.catalog().get_multimedia_metadata_table_catalog_entry(table_catalog_entry)
        self.assertTrue(video_metadata_table is not None)
        drop_query = 'DROP TABLE IF EXISTS MyVideo;'
        execute_query_fetch_all(self.evadb, drop_query)
        self.assertTrue(self.evadb.catalog().get_table_catalog_entry('MyVideo') is None)
        column_objects = self.evadb.catalog().get_column_catalog_entries_by_table(table_catalog_entry)
        self.assertEqual(len(column_objects), 0)
        self.assertFalse(Path(video_dir).exists())
        drop_query = 'DROP TABLE MyVideo;'
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, drop_query, do_not_print_exceptions=True)
        execute_query_fetch_all(self.evadb, query)
        execute_query_fetch_all(self.evadb, drop_query)

    def run_create_function_query(self):
        create_function_query = "CREATE FUNCTION DummyObjectDetector\n            INPUT  (Frame_Array NDARRAY UINT8(3, 256, 256))\n            OUTPUT (label NDARRAY STR(10))\n            TYPE  Classification\n            IMPL  'test/util.py';"
        execute_query_fetch_all(self.evadb, create_function_query)

    def test_should_drop_function(self):
        self.run_create_function_query()
        function_name = 'DummyObjectDetector'
        function = self.evadb.catalog().get_function_catalog_entry_by_name(function_name)
        self.assertTrue(function is not None)
        drop_query = 'DROP FUNCTION IF EXISTS {};'.format(function_name)
        execute_query_fetch_all(self.evadb, drop_query)
        function = self.evadb.catalog().get_function_catalog_entry_by_name(function_name)
        self.assertTrue(function is None)
        self.run_create_function_query()
        execute_query_fetch_all(self.evadb, drop_query)

    def test_drop_wrong_function_name(self):
        self.run_create_function_query()
        right_function_name = 'DummyObjectDetector'
        wrong_function_name = 'FakeDummyObjectDetector'
        function = self.evadb.catalog().get_function_catalog_entry_by_name(right_function_name)
        self.assertTrue(function is not None)
        drop_query = 'DROP FUNCTION {};'.format(wrong_function_name)
        try:
            execute_query_fetch_all(self.evadb, drop_query, do_not_print_exceptions=True)
        except Exception as e:
            err_msg = 'Function {} does not exist, therefore cannot be dropped.'.format(wrong_function_name)
            self.assertTrue(str(e) == err_msg)
        function = self.evadb.catalog().get_function_catalog_entry_by_name(right_function_name)
        self.assertTrue(function is not None)

    def test_should_drop_index(self):
        index_name = 'index_name'
        self._create_index(index_name)
        index_obj = self.evadb.catalog().get_index_catalog_entry_by_name(index_name)
        self.assertTrue(index_obj is not None)
        wrong_function_name = 'wrong_function_name'
        drop_query = f'DROP INDEX {wrong_function_name};'
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, drop_query, do_not_print_exceptions=True)
        obj = self.evadb.catalog().get_index_catalog_entry_by_name(index_name)
        self.assertTrue(obj is not None)
        drop_query = f'DROP INDEX IF EXISTS {index_name};'
        execute_query_fetch_all(self.evadb, drop_query)
        index_obj = self.evadb.catalog().get_index_catalog_entry_by_name(index_name)
        self.assertTrue(index_obj is None)

    def test_should_drop_database(self):
        database_name = 'test_data_source'
        params = {'database': 'evadb.db'}
        query = f'CREATE DATABASE {database_name}\n                    WITH ENGINE = "sqlite",\n                    PARAMETERS = {params};'
        execute_query_fetch_all(self.evadb, query)
        self.assertIsNotNone(self.evadb.catalog().get_database_catalog_entry(database_name))
        execute_query_fetch_all(self.evadb, f'DROP DATABASE {database_name}')
        self.assertIsNone(self.evadb.catalog().get_database_catalog_entry(database_name))
        result = execute_query_fetch_all(self.evadb, f'DROP DATABASE IF EXISTS {database_name}')
        self.assertTrue('does not exist' in result.frames.to_string())
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, f'DROP DATABASE {database_name}', do_not_print_exceptions=True)
        execute_query_fetch_all(self.evadb, query)
        result = execute_query_fetch_all(self.evadb, f'DROP DATABASE IF EXISTS {database_name}')

    def test_should_drop_job(self):
        job_name = 'test_async_job'
        query = f"CREATE JOB {job_name} AS {{\n            SELECT * from job_catalog;\n        }}\n        START '2023-04-01'\n        END '2023-05-01'\n        EVERY 2 week;"
        execute_query_fetch_all(self.evadb, query)
        self.assertIsNotNone(self.evadb.catalog().get_job_catalog_entry(job_name))
        execute_query_fetch_all(self.evadb, f'DROP JOB {job_name}')
        self.assertIsNone(self.evadb.catalog().get_job_catalog_entry(job_name))
        result = execute_query_fetch_all(self.evadb, f'DROP JOB IF EXISTS {job_name}')
        self.assertTrue('does not exist' in result.frames.to_string())
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, f'DROP JOB {job_name}', do_not_print_exceptions=True)
        execute_query_fetch_all(self.evadb, query)
        result = execute_query_fetch_all(self.evadb, f'DROP JOB IF EXISTS {job_name}')

@pytest.mark.notparallel
class InsertExecutorTest(unittest.TestCase):

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        self.video_file_path = create_sample_video()
        query = 'CREATE TABLE IF NOT EXISTS CSVTable\n            (\n                name TEXT(100)\n            );\n        '
        execute_query_fetch_all(self.evadb, query)
        query = 'CREATE TABLE IF NOT EXISTS books\n            (\n                name    TEXT(100),\n                author  TEXT(100),\n                year    INTEGER\n            );\n        '
        execute_query_fetch_all(self.evadb, query)

    def tearDown(self):
        shutdown_ray()
        file_remove('dummy.avi')
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS books;')

    @unittest.skip('Not supported in current version')
    def test_should_load_video_in_table(self):
        query = f"LOAD VIDEO '{self.video_file_path}' INTO MyVideo;"
        execute_query_fetch_all(self.evadb, query)
        insert_query = ' INSERT INTO MyVideo (id, data) VALUES\n            (40, [[40, 40, 40], [40, 40, 40]],\n                 [[40, 40, 40], [40, 40, 40]]);'
        execute_query_fetch_all(self.evadb, insert_query)
        insert_query_2 = ' INSERT INTO MyVideo (id, data) VALUES\n        ( 41, [[41, 41, 41] , [41, 41, 41]],\n                [[41, 41, 41], [41, 41, 41]]);'
        execute_query_fetch_all(self.evadb, insert_query_2)
        query = 'SELECT id, data FROM MyVideo WHERE id = 40'
        batch = execute_query_fetch_all(self.evadb, query)
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['data'][0], np.array([[[40, 40, 40], [40, 40, 40]], [[40, 40, 40], [40, 40, 40]]])))
        query = 'SELECT id, data FROM MyVideo WHERE id = 41;'
        batch = execute_query_fetch_all(self.evadb, query)
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['data'][0], np.array([[[41, 41, 41], [41, 41, 41]], [[41, 41, 41], [41, 41, 41]]])))

    def test_should_insert_tuples_in_table(self):
        data = pd.read_csv('./test/data/features.csv')
        for i in data.iterrows():
            logger.info(i[1][1])
            query = f"INSERT INTO CSVTable (name) VALUES (\n                            '{i[1][1]}'\n                        );"
            logger.info(query)
            batch = execute_query_fetch_all(self.evadb, query)
        query = 'SELECT name FROM CSVTable;'
        batch = execute_query_fetch_all(self.evadb, query)
        logger.info(batch)
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['csvtable.name'].array, np.array(['test_evadb/similarity/data/sad.jpg', 'test_evadb/similarity/data/happy.jpg', 'test_evadb/similarity/data/angry.jpg'])))
        query = "SELECT name FROM CSVTable WHERE name LIKE '.*(sad|happy)';"
        batch = execute_query_fetch_all(self.evadb, query)
        self.assertEqual(len(batch._frames), 2)

    def test_insert_one_tuple_in_table(self):
        query = "\n            INSERT INTO books (name, author, year) VALUES (\n                'Harry Potter', 'JK Rowling', 1997\n            );\n        "
        execute_query_fetch_all(self.evadb, query)
        query = 'SELECT * FROM books;'
        batch = execute_query_fetch_all(self.evadb, query)
        logger.info(batch)
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['books.name'].array, np.array(['Harry Potter'])))
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['books.author'].array, np.array(['JK Rowling'])))
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['books.year'].array, np.array([1997])))

    def test_insert_multiple_tuples_in_table(self):
        query = "\n            INSERT INTO books (name, author, year) VALUES\n            ('Fantastic Beasts Collection', 'JK Rowling', 2001),\n            ('Magic Tree House Collection', 'Mary Pope Osborne', 1992),\n            ('Sherlock Holmes', 'Arthur Conan Doyle', 1887);\n        "
        execute_query_fetch_all(self.evadb, query)
        query = 'SELECT * FROM books;'
        batch = execute_query_fetch_all(self.evadb, query)
        logger.info(batch)
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['books.name'].array, np.array(['Fantastic Beasts Collection', 'Magic Tree House Collection', 'Sherlock Holmes'])))
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['books.author'].array, np.array(['JK Rowling', 'Mary Pope Osborne', 'Arthur Conan Doyle'])))
        self.assertIsNone(np.testing.assert_array_equal(batch.frames['books.year'].array, np.array([2001, 1992, 1887])))

@pytest.mark.notparallel
class RenameExecutorTest(unittest.TestCase):

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        self.video_file_path = create_sample_video()
        self.csv_file_path = create_sample_csv()

    def tearDown(self):
        file_remove('dummy.avi')
        file_remove('dummy.csv')

    def test_should_rename_table(self):
        catalog_manager = self.evadb.catalog()
        query = f"LOAD VIDEO '{self.video_file_path}' INTO MyVideo;"
        execute_query_fetch_all(self.evadb, query)
        self.assertTrue(catalog_manager.get_table_catalog_entry('MyVideo') is not None)
        self.assertTrue(catalog_manager.get_table_catalog_entry('MyVideo1') is None)
        rename_query = 'RENAME TABLE MyVideo TO MyVideo1;'
        execute_query_fetch_all(self.evadb, rename_query)
        self.assertTrue(catalog_manager.get_table_catalog_entry('MyVideo') is None)
        self.assertTrue(catalog_manager.get_table_catalog_entry('MyVideo1') is not None)
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyVideo;')

    def test_should_fail_on_rename_structured_table(self):
        create_table_query = '\n\n            CREATE TABLE IF NOT EXISTS MyVideoCSV (\n                id INTEGER UNIQUE,\n                frame_id INTEGER NOT NULL,\n                video_id INTEGER NOT NULL,\n                dataset_name TEXT(30) NOT NULL\n            );\n            '
        execute_query_fetch_all(self.evadb, create_table_query)
        load_query = f"LOAD CSV '{self.csv_file_path}' INTO MyVideoCSV (id, frame_id, video_id, dataset_name);"
        execute_query_fetch_all(self.evadb, load_query)
        with self.assertRaises(Exception) as cm:
            rename_query = 'RENAME TABLE MyVideoCSV TO MyVideoCSV1;'
            execute_query_fetch_all(self.evadb, rename_query)
        self.assertEqual(str(cm.exception), 'Rename not yet supported on structured data')
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS MyVideoCSV;')

class CreateDatabaseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()

    @classmethod
    def tearDownClass(cls):
        shutdown_ray()

    def test_use_should_raise_executor_error(self):
        query = 'USE not_available_ds {\n            SELECT * FROM table\n        }'
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, query)

class YoutubeChannelQATest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        os.environ['ray'] = str(cls.evadb.catalog().get_configuration_catalog_value('ray'))

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self) -> None:
        shutdown_ray()

    def test_should_run_youtube_channel_qa_app(self):
        app_path = Path('apps', 'youtube_channel_qa', 'youtube_channel_qa.py')
        input1 = '\n\n\n'
        input2 = 'What is this video about?\n'
        input3 = 'exit\n'
        inputs = input1 + input2 + input3
        command = ['python', app_path]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(inputs.encode())
        decoded_stdout = stdout.decode()
        assert 'keyboards' or 'AliExpress' or 'Rate limit' in decoded_stdout
        print(decoded_stdout)
        print(stderr.decode())

@pytest.mark.notparallel
class PrivateGPTTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        os.environ['ray'] = str(cls.evadb.catalog().get_configuration_catalog_value('ray'))

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self) -> None:
        shutdown_ray()
        source_directory = 'source_documents'
        if shutil.os.path.exists(source_directory):
            shutil.rmtree(source_directory)

    @unittest.skip('disable test due to inference time')
    def test_should_run_privategpt(self):
        print('INGEST')
        source_directory = 'source_documents'
        if shutil.os.path.exists(source_directory):
            shutil.rmtree(source_directory)
        url = 'https://www.eisenhowerlibrary.gov/sites/default/files/file/1953_state_of_the_union.pdf'
        pdf_destination = 'state_of_the_union.pdf'
        response = requests.get(url)
        if response.status_code == 200:
            with open(pdf_destination, 'wb') as file:
                file.write(response.content)
        os.makedirs(source_directory, exist_ok=True)
        shutil.move(pdf_destination, source_directory)
        app_path = Path('apps', 'privategpt', 'ingest.py')
        inputs = ''
        command = ['python', app_path, '--directory', 'source_documents']
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(inputs.encode())
        decoded_stdout = stdout.decode()
        assert 'Data ingestion complete' in decoded_stdout
        decoded_stderr = stderr.decode()
        assert 'Ray' in decoded_stderr
        inputs = 'When was NATO created?\nexit\n'
        app_path = Path('apps', 'privategpt', 'privateGPT.py')
        command = ['python', app_path]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(inputs.encode())
        decoded_stdout = stdout.decode()
        assert 'April 4, 1949' in decoded_stdout

class YoutubeQATest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        os.environ['ray'] = str(cls.evadb.catalog().get_configuration_catalog_value('ray'))

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self) -> None:
        shutdown_ray()

    @chatgpt_skip_marker
    def test_should_run_youtube_qa_app(self):
        app_path = Path('apps', 'youtube_qa', 'youtube_qa.py')
        input1 = 'yes\n\n'
        input2 = 'What is this video on?\n'
        input3 = 'exit\nexit\n'
        inputs = input1 + input2 + input3
        command = ['python', app_path]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(inputs.encode())
        decoded_stdout = stdout.decode()
        assert 'Julia' or 'Rate limit' in decoded_stdout
        print(decoded_stdout)
        print(stderr.decode())

class PandasQATest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evadb = get_evadb_for_testing()
        cls.evadb.catalog().reset()
        os.environ['ray'] = str(cls.evadb.catalog().get_configuration_catalog_value('ray'))

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self) -> None:
        shutdown_ray()

    @chatgpt_skip_marker
    def test_should_run_pandas_qa_app(self):
        app_path = Path('apps', 'pandas_qa', 'pandas_qa.py')
        input1 = '\n'
        input2 = 'Print country with highest gdp\n\n'
        input3 = 'yes\n\n'
        inputs = input1 + input2 + input3
        command = ['python', app_path]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(inputs.encode())
        decoded_stdout = stdout.decode()
        assert 'Country' or 'Rate' in decoded_stdout

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

class CSVLoaderTest(unittest.TestCase):

    def setUp(self):
        self.csv_file_path = create_sample_csv()

    def tearDown(self):
        file_remove('dummy.csv')

    def test_should_return_one_batch(self):
        column_list = [TupleValueExpression(name='id', table_alias='dummy'), TupleValueExpression(name='frame_id', table_alias='dummy'), TupleValueExpression(name='video_id', table_alias='dummy')]
        csv_loader = CSVReader(file_url=self.csv_file_path, column_list=column_list)
        batches = list(csv_loader.read())
        expected = list(create_dummy_csv_batches(target_columns=['id', 'frame_id', 'video_id']))
        self.assertEqual(batches, expected)

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

@pytest.mark.notparallel
class NativeExecutorTest(unittest.TestCase):

    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()

    def tearDown(self):
        shutdown_ray()
        self._drop_table_in_native_database()
        self._drop_table_in_evadb_database()

    def _create_table_in_native_database(self):
        execute_query_fetch_all(self.evadb, 'USE test_data_source {\n                CREATE TABLE test_table (\n                    name VARCHAR(10),\n                    Age INT,\n                    comment VARCHAR (100)\n                )\n            }')

    def _insert_value_into_native_database(self, col1, col2, col3):
        execute_query_fetch_all(self.evadb, f"USE test_data_source {{\n                INSERT INTO test_table (\n                    name, Age, comment\n                ) VALUES (\n                    '{col1}', {col2}, '{col3}'\n                )\n            }}")

    def _drop_table_in_native_database(self):
        execute_query_fetch_all(self.evadb, 'USE test_data_source {\n                DROP TABLE IF EXISTS test_table\n            }')
        execute_query_fetch_all(self.evadb, 'USE test_data_source {\n                DROP TABLE IF EXISTS derived_table\n            }')

    def _drop_table_in_evadb_database(self):
        execute_query_fetch_all(self.evadb, 'DROP TABLE IF EXISTS eva_table;')

    def _create_evadb_table_using_select_query(self):
        execute_query_fetch_all(self.evadb, 'CREATE TABLE eva_table AS SELECT name, Age FROM test_data_source.test_table;')
        res_batch = execute_query_fetch_all(self.evadb, 'Select * from eva_table')
        self.assertEqual(len(res_batch), 2)
        self.assertEqual(res_batch.frames['eva_table.name'][0], 'aa')
        self.assertEqual(res_batch.frames['eva_table.age'][0], 1)
        self.assertEqual(res_batch.frames['eva_table.name'][1], 'bb')
        self.assertEqual(res_batch.frames['eva_table.age'][1], 2)

    def _create_native_table_using_select_query(self):
        execute_query_fetch_all(self.evadb, 'CREATE TABLE test_data_source.derived_table AS SELECT name, age FROM test_data_source.test_table;')
        res_batch = execute_query_fetch_all(self.evadb, 'SELECT * FROM test_data_source.derived_table')
        self.assertEqual(len(res_batch), 2)
        self.assertEqual(res_batch.frames['derived_table.name'][0], 'aa')
        self.assertEqual(res_batch.frames['derived_table.age'][0], 1)
        self.assertEqual(res_batch.frames['derived_table.name'][1], 'bb')
        self.assertEqual(res_batch.frames['derived_table.age'][1], 2)

    def _execute_evadb_query(self):
        self._create_table_in_native_database()
        self._insert_value_into_native_database('aa', 1, 'aaaa')
        self._insert_value_into_native_database('bb', 2, 'bbbb')
        res_batch = execute_query_fetch_all(self.evadb, 'SELECT * FROM test_data_source.test_table')
        self.assertEqual(len(res_batch), 2)
        self.assertEqual(res_batch.frames['test_table.name'][0], 'aa')
        self.assertEqual(res_batch.frames['test_table.age'][0], 1)
        self.assertEqual(res_batch.frames['test_table.name'][1], 'bb')
        self.assertEqual(res_batch.frames['test_table.age'][1], 2)
        self._create_evadb_table_using_select_query()
        self._create_native_table_using_select_query()
        self._drop_table_in_native_database()
        self._drop_table_in_evadb_database()

    def _execute_native_query(self):
        self._create_table_in_native_database()
        self._insert_value_into_native_database('aa', 1, 'aaaa')
        res_batch = execute_query_fetch_all(self.evadb, 'USE test_data_source {\n                SELECT * FROM test_table\n            }')
        self.assertEqual(len(res_batch), 1)
        self.assertEqual(res_batch.frames['name'][0], 'aa')
        self.assertEqual(res_batch.frames['age'][0], 1)
        self.assertEqual(res_batch.frames['comment'][0], 'aaaa')
        self._drop_table_in_native_database()

    def _raise_error_on_multiple_creation(self):
        params = {'user': 'eva', 'password': 'password', 'host': 'localhost', 'port': '5432', 'database': 'evadb'}
        query = f'CREATE DATABASE test_data_source\n                    WITH ENGINE = "postgres",\n                    PARAMETERS = {params};'
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, query)

    def _raise_error_on_invalid_connection(self):
        params = {'user': 'xxxxxx', 'password': 'xxxxxx', 'host': 'localhost', 'port': '5432', 'database': 'evadb'}
        query = f'CREATE DATABASE invaid\n                    WITH ENGINE = "postgres",\n                    PARAMETERS = {params};'
        with self.assertRaises(ExecutorError):
            execute_query_fetch_all(self.evadb, query)

    def test_should_run_query_in_postgres(self):
        params = {'user': 'eva', 'password': 'password', 'host': 'localhost', 'port': '5432', 'database': 'evadb'}
        query = f'CREATE DATABASE test_data_source\n                    WITH ENGINE = "postgres",\n                    PARAMETERS = {params};'
        execute_query_fetch_all(self.evadb, query)
        self._execute_native_query()
        self._execute_evadb_query()
        self._raise_error_on_multiple_creation()
        self._raise_error_on_invalid_connection()

    def test_should_run_query_in_mariadb(self):
        params = {'user': 'eva', 'password': 'password', 'database': 'evadb'}
        query = f'CREATE DATABASE test_data_source\n                    WITH ENGINE = "mariadb",\n                    PARAMETERS = {params};'
        execute_query_fetch_all(self.evadb, query)
        self._execute_native_query()
        self._execute_evadb_query()

    def test_should_run_query_in_clickhouse(self):
        params = {'user': 'eva', 'password': 'password', 'host': 'localhost', 'port': '9000', 'database': 'evadb'}
        query = f'CREATE DATABASE test_data_source\n                    WITH ENGINE = "clickhouse",\n                    PARAMETERS = {params};'
        execute_query_fetch_all(self.evadb, query)
        self._execute_native_query()
        self._execute_evadb_query()

    @pytest.mark.skip(reason='Snowflake does not come with a free version of account, so integration test is not feasible')
    def test_should_run_query_in_snowflake(self):
        params = {'user': 'eva', 'password': 'password', 'account': 'account_number', 'database': 'EVADB', 'schema': 'SAMPLE_DATA', 'warehouse': 'warehouse'}
        query = f'CREATE DATABASE test_data_source\n                    WITH ENGINE = "snowflake",\n                    PARAMETERS = {params};'
        execute_query_fetch_all(self.evadb, query)
        self._execute_native_query()
        self._execute_evadb_query()

    def test_should_run_query_in_sqlite(self):
        import os
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        params = {'database': f'{current_file_dir}/evadb.db'}
        query = f'CREATE DATABASE test_data_source\n                    WITH ENGINE = "sqlite",\n                    PARAMETERS = {params};'
        execute_query_fetch_all(self.evadb, query)
        self._execute_native_query()
        self._execute_evadb_query()

    def test_should_run_query_in_mysql(self):
        params = {'user': 'eva', 'password': 'password', 'host': 'localhost', 'port': '3306', 'database': 'evadb'}
        query = f'CREATE DATABASE test_data_source\n                    WITH ENGINE = "mysql",\n                    PARAMETERS = {params};'
        execute_query_fetch_all(self.evadb, query)
        self._execute_native_query()
        self._execute_evadb_query()

def try_to_import_ray():
    try:
        import ray
        from ray.util.queue import Queue
    except ImportError:
        raise ValueError('Could not import ray python package.\n                Please install it with `pip install ray`.')

