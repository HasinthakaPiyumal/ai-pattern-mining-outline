# Cluster 59

class TestFileBackedBarrier(unittest.TestCase):
    """
    A class to test that the file backed barrier works properly.
    """

    def test_file_backed_barrier_functions_normal_case_local(self) -> None:
        """
        Tests that the file backed barrier functions properly locally.
        """
        sleep_interval_sec = 10
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            current_time = 0.0

            def patch_time_sleep(sleep_time: float) -> None:
                """
                A patch for the time.sleep method.
                :param sleep_time: The time to sleep.
                """
                nonlocal current_time
                if current_time == sleep_interval_sec + 30:
                    current_time += SLEEP_MULTIPLIER_BEFORE_CLEANUP * sleep_interval_sec
                    self.assertEqual(sleep_interval_sec * SLEEP_MULTIPLIER_BEFORE_CLEANUP, sleep_time)
                else:
                    current_time += sleep_interval_sec
                    self.assertEqual(sleep_interval_sec, sleep_time)
                if current_time == 40:
                    FileBackedBarrier(tmp_dir)._register_activity_id_complete('2')
                if current_time == 70 + SLEEP_MULTIPLIER_BEFORE_CLEANUP * sleep_interval_sec:
                    FileBackedBarrier(tmp_dir)._remove_activity_after_processing('2')

            def patch_time_time() -> float:
                """
                A patch for the time.time method.
                :return: The current time.
                """
                nonlocal current_time
                self.assertLess(current_time, 80 + SLEEP_MULTIPLIER_BEFORE_CLEANUP * sleep_interval_sec)
                return current_time
            with unittest.mock.patch('nuplan.common.utils.file_backed_barrier.time.sleep', patch_time_sleep), unittest.mock.patch('nuplan.common.utils.file_backed_barrier.time.time', patch_time_time):
                barrier = FileBackedBarrier(tmp_dir)
                barrier.wait_barrier('1', {'1', '2'}, timeout_s=None, poll_interval_s=sleep_interval_sec)

    def test_file_backed_barrier_functions_normal_case_s3(self) -> None:
        """
        Tests that the file backed barrier functions properly in s3.
        """
        sleep_interval_sec = 10
        sample_s3_path = 's3://ml-caches/mitchell.spryn/barrier'
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            current_time = 0.0

            def patch_time_sleep(sleep_time: float) -> None:
                """
                A patch for the time.sleep method.
                :param sleep_time: The time to sleep.
                """
                nonlocal current_time
                if current_time == sleep_interval_sec + 30:
                    current_time += SLEEP_MULTIPLIER_BEFORE_CLEANUP * sleep_interval_sec
                    self.assertEqual(sleep_interval_sec * SLEEP_MULTIPLIER_BEFORE_CLEANUP, sleep_time)
                else:
                    current_time += sleep_interval_sec
                    self.assertEqual(sleep_interval_sec, sleep_time)
                if current_time == 40:
                    FileBackedBarrier(tmp_dir)._register_activity_id_complete('2')
                if current_time == 70 + SLEEP_MULTIPLIER_BEFORE_CLEANUP * sleep_interval_sec:
                    FileBackedBarrier(tmp_dir)._remove_activity_after_processing('2')

            def patch_time_time() -> float:
                """
                A patch for the time.time method.
                :return: The current time.
                """
                nonlocal current_time
                self.assertLess(current_time, 80 + SLEEP_MULTIPLIER_BEFORE_CLEANUP * sleep_interval_sec)
                return current_time

            def patch_get_s3_client() -> unittest.mock.Mock:
                """
                Mocks the get_s3_client method.
                """

                def patch_s3_client_put_object(Body: bytes, Bucket: str, Key: str) -> None:
                    """
                    A patch for the s3 client put object method.
                    :param body: The body passed to the method.
                    :param bucket: The bucket passed to the method.
                    :param key: The key passed to the method.
                    """
                    self.assertTrue(len(Body) > 0)
                    self.assertEqual('ml-caches', Bucket)
                    self.assertEqual('mitchell.spryn/barrier/1', Key)
                    check_file = tmp_dir / '1'
                    self.assertFalse(check_file.exists())
                    with open(check_file, 'w') as f:
                        f.write('x')

                def patch_s3_client_delete_object(Bucket: str, Key: str) -> None:
                    """
                    A patch for the s3 client put object method.
                    :param bucket: The bucket passed to the method.
                    :param key: The key passed to the method.
                    """
                    self.assertEqual('ml-caches', Bucket)
                    self.assertEqual('mitchell.spryn/barrier/1', Key)
                    check_file = tmp_dir / '1'
                    self.assertTrue(check_file.exists())
                    check_file.unlink()

                def patch_s3_client_list_objects_v2(Bucket: str, Prefix: str) -> Dict[str, List[Dict[str, str]]]:
                    """
                    A patch for the s3 client list objects v2 method.
                    :param Bucket: The bucket passed.
                    :param Prefix: The prefix passed.
                    """
                    self.assertEqual('ml-caches', Bucket)
                    self.assertEqual('mitchell.spryn/barrier/', Prefix)
                    return {'Contents': [{'Key': f's3://ml-caches/mitchell.spryn/barrier/{p.stem}'} for p in tmp_dir.glob('**/*') if p.is_file()]}
                mock_client = unittest.mock.Mock()
                mock_client.put_object = patch_s3_client_put_object
                mock_client.list_objects_v2 = patch_s3_client_list_objects_v2
                mock_client.delete_object = patch_s3_client_delete_object
                return mock_client
            with unittest.mock.patch('nuplan.common.utils.file_backed_barrier.time.sleep', patch_time_sleep), unittest.mock.patch('nuplan.common.utils.file_backed_barrier.time.time', patch_time_time), unittest.mock.patch('nuplan.common.utils.file_backed_barrier.get_s3_client', patch_get_s3_client):
                barrier = FileBackedBarrier(Path(sample_s3_path))
                barrier.wait_barrier('1', {'1', '2'}, timeout_s=None, poll_interval_s=sleep_interval_sec)

    def test_file_backed_barrier_timeout(self) -> None:
        """
        Tests that the timeout feature works properly.
        """
        sleep_interval_sec = 10
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            current_time = 0.0

            def patch_time_sleep(sleep_time: float) -> None:
                """
                A patch for the time.sleep method.
                :param sleep_time: The time to sleep.
                """
                nonlocal current_time
                self.assertEqual(sleep_interval_sec, sleep_time)
                current_time += sleep_interval_sec

            def patch_time_time() -> float:
                """
                A patch for the time.time method.
                :return: The current time.
                """
                nonlocal current_time
                return current_time
            with unittest.mock.patch('nuplan.common.utils.file_backed_barrier.time.sleep', patch_time_sleep), unittest.mock.patch('nuplan.common.utils.file_backed_barrier.time.time', patch_time_time):
                barrier = FileBackedBarrier(tmp_dir)
                with self.assertRaises(TimeoutError):
                    barrier.wait_barrier('1', {'1', '2'}, timeout_s=40, poll_interval_s=sleep_interval_sec)

