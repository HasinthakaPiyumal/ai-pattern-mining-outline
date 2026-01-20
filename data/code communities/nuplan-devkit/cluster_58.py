# Cluster 58

class TestIoUtils(unittest.TestCase):
    """
    A class to test that the I/O utilities in nuplan_devkit function properly.
    """

    def test_nupath(self) -> None:
        """
        Tests that converting NuPath to strings works properly.
        """
        example_s3_path = NuPath('s3://test-bucket/foo/bar/baz.txt')
        expected_s3_str = 's3://test-bucket/foo/bar/baz.txt'
        actual_s3_str = str(example_s3_path)
        self.assertEqual(expected_s3_str, actual_s3_str)
        example_local_path = NuPath('/foo/bar/baz')
        expected_local_str = '/foo/bar/baz'
        actual_local_str = str(example_local_path)
        self.assertEqual(expected_local_str, actual_local_str)

    def test_safe_path_to_string(self) -> None:
        """
        Tests that converting paths to strings safely works properly.
        """
        example_s3_path = Path('s3://test-bucket/foo/bar/baz.txt')
        expected_s3_str = 's3://test-bucket/foo/bar/baz.txt'
        actual_s3_str = safe_path_to_string(example_s3_path)
        self.assertEqual(expected_s3_str, actual_s3_str)
        example_local_path = Path('/foo/bar/baz')
        expected_local_str = '/foo/bar/baz'
        actual_local_str = safe_path_to_string(example_local_path)
        self.assertEqual(expected_local_str, actual_local_str)
        example_s3_str_path = 's3://test-bucket/foo/bar/baz.txt'
        expected_s3_str = 's3://test-bucket/foo/bar/baz.txt'
        actual_s3_str = safe_path_to_string(example_s3_str_path)
        self.assertEqual(expected_s3_str, actual_s3_str)
        example_local_str_path = '/foo/bar/baz'
        expected_local_str = '/foo/bar/baz'
        actual_local_str = safe_path_to_string(example_local_str_path)
        self.assertEqual(expected_local_str, actual_local_str)

    def test_save_buffer_locally(self) -> None:
        """
        Tests that saving a buffer locally works properly.
        """
        expected_buffer = b'test'
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / 'local_buffer.bin'
            save_buffer(output_file, expected_buffer)
            with open(output_file, 'rb') as f:
                reconstructed_buffer = f.read()
            self.assertEqual(expected_buffer, reconstructed_buffer)

    def test_save_buffer_s3(self) -> None:
        """
        Tests that saving a buffer to s3 works properly.
        """
        upload_bucket_name = 'ml-caches'
        upload_path = Path('foo/bar/baz.bin')
        uploaded_file_contents: Optional[bytes] = None

        async def patch_upload_file_to_s3_async(local_path: Path, s3_key: Path, s3_bucket: str) -> None:
            """
            Patch for upload_file_to_s3_async method.
            :param local_path: The passed local_path.
            :param s3_key: The passed s3_key.
            :param s3_bucket: The passed s3_bucket.
            """
            nonlocal uploaded_file_contents
            self.assertEqual(upload_bucket_name, s3_bucket)
            self.assertEqual(upload_path, s3_key)
            with open(local_path, 'rb') as f:
                uploaded_file_contents = f.read()
        expected_buffer = b'test'
        with patch_with_validation('nuplan.common.utils.io_utils.upload_file_to_s3_async', patch_upload_file_to_s3_async):
            output_file = Path(f's3://{upload_bucket_name}') / f'{upload_path}'
            save_buffer(output_file, expected_buffer)
            self.assertIsNotNone(uploaded_file_contents)
            assert uploaded_file_contents is not None
            self.assertEqual(expected_buffer, uploaded_file_contents)

    def test_save_object_as_pickle_locally(self) -> None:
        """
        Tests that saving a pickled object locally works properly.
        """
        expected_object = {'a': 1, 'b': 2}
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / 'local.pkl'
            save_object_as_pickle(output_file, expected_object)
            with open(output_file, 'rb') as f:
                reconstructed_object = pickle.load(f)
            self.assertEqual(expected_object, reconstructed_object)

    def test_save_object_as_pickle_s3(self) -> None:
        """
        Tests that saving a pickled object to s3 works properly.
        """
        upload_bucket_name = 'ml-caches'
        upload_path = Path('foo/bar/baz.pkl')
        uploaded_file_contents: Optional[bytes] = None

        async def patch_upload_file_to_s3_async(local_path: Path, s3_key: Path, s3_bucket: str) -> None:
            """
            Patch for upload_file_to_s3_async method.
            :param local_path: The passed local_path.
            :param s3_key: The passed s3_key.
            :param s3_bucket: The passed s3_bucket.
            """
            nonlocal uploaded_file_contents
            self.assertEqual(upload_bucket_name, s3_bucket)
            self.assertEqual(upload_path, s3_key)
            with open(local_path, 'rb') as f:
                uploaded_file_contents = f.read()
        expected_object = {'a': 1, 'b': 2}
        with patch_with_validation('nuplan.common.utils.io_utils.upload_file_to_s3_async', patch_upload_file_to_s3_async):
            output_file = Path(f's3://{upload_bucket_name}') / f'{upload_path}'
            save_object_as_pickle(output_file, expected_object)
            self.assertIsNotNone(uploaded_file_contents)
            assert uploaded_file_contents is not None
            reconstructed_object: Dict[str, int] = pickle.loads(uploaded_file_contents)
            self.assertEqual(expected_object, reconstructed_object)

    def test_save_text_locally(self) -> None:
        """
        Tests that saving a text file locally works properly.
        """
        expected_text = 'test_save_text_locally.'
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / 'local.txt'
            save_text(output_file, expected_text)
            with open(output_file, 'r') as f:
                reconstructed_text = f.read()
            self.assertEqual(expected_text, reconstructed_text)

    def test_save_text_s3(self) -> None:
        """
        Tests that saving a text file to s3 works properly.
        """
        upload_bucket_name = 'ml-caches'
        upload_path = Path('foo/bar/baz.pkl')
        uploaded_file_contents: Optional[str] = None

        async def patch_upload_file_to_s3_async(local_path: Path, s3_key: Path, s3_bucket: str) -> None:
            """
            Patch for upload_file_to_s3_async method.
            :param local_path: The passed local_path.
            :param s3_key: The passed s3_key.
            :param s3_bucket: The passed s3_bucket.
            """
            nonlocal uploaded_file_contents
            self.assertEqual(upload_bucket_name, s3_bucket)
            self.assertEqual(upload_path, s3_key)
            with open(local_path, 'r') as f:
                uploaded_file_contents = f.read()
        expected_text = 'test_save_text_s3.'
        with patch_with_validation('nuplan.common.utils.io_utils.upload_file_to_s3_async', patch_upload_file_to_s3_async):
            output_file = Path(f's3://{upload_bucket_name}') / f'{upload_path}'
            save_text(output_file, expected_text)
            self.assertIsNotNone(uploaded_file_contents)
            self.assertEqual(expected_text, uploaded_file_contents)

    def test_read_text_locally(self) -> None:
        """
        Tests that reading a text file locally works properly.
        """
        expected_text = 'some expected text.'
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / 'read_text_locally.txt'
            with open(output_file, 'w') as f:
                f.write(expected_text)
            reconstructed_text = read_text(output_file)
            self.assertEqual(expected_text, reconstructed_text)

    def test_read_text_from_s3(self) -> None:
        """
        Tests that reading a text file from S3 works properly.
        """
        download_bucket = 'ml-caches'
        download_key = 'my/file/path.txt'
        expected_text = 'some expected text.'
        full_filepath = Path(f's3://{download_bucket}') / download_key

        async def patch_read_binary_file_contents_from_s3_async(s3_key: Path, s3_bucket: str) -> bytes:
            """
            A patch for the read_binary_file_contents_from_s3_async method.
            :param s3_key: The passed key
            :param s3_bucket: The passed bucket.
            """
            self.assertEqual(Path(download_key), s3_key)
            self.assertEqual(download_bucket, s3_bucket)
            return expected_text.encode('utf-8')
        with patch_with_validation('nuplan.common.utils.io_utils.read_binary_file_contents_from_s3_async', patch_read_binary_file_contents_from_s3_async):
            reconstructed_text = read_text(full_filepath)
            self.assertEqual(expected_text, reconstructed_text)

    def test_read_pickle_locally(self) -> None:
        """
        Tests that reading a pickle file locally works properly.
        """
        expected_obj = {'foo': 'bar'}
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / 'read_text_locally.txt'
            with open(output_file, 'wb') as f:
                f.write(pickle.dumps(expected_obj))
            reconstructed_obj = read_pickle(output_file)
            self.assertEqual(expected_obj, reconstructed_obj)

    def test_read_pickle_from_s3(self) -> None:
        """
        Tests that reading a pickle file from S3 works properly.
        """
        download_bucket = 'ml-caches'
        download_key = 'my/file/path.txt'
        expected_obj = {'foo': 'bar'}
        full_filepath = Path(f's3://{download_bucket}') / download_key

        async def patch_read_binary_file_contents_from_s3_async(s3_key: Path, s3_bucket: str) -> bytes:
            """
            A patch for the read_binary_file_contents_from_s3_async method.
            :param s3_key: The passed key
            :param s3_bucket: The passed bucket.
            """
            self.assertEqual(Path(download_key), s3_key)
            self.assertEqual(download_bucket, s3_bucket)
            return pickle.dumps(expected_obj)
        with patch_with_validation('nuplan.common.utils.io_utils.read_binary_file_contents_from_s3_async', patch_read_binary_file_contents_from_s3_async):
            reconstructed_obj = read_pickle(full_filepath)
            self.assertEqual(expected_obj, reconstructed_obj)

    def test_read_binary_locally(self) -> None:
        """
        Tests that reading a binary file locally works properly.
        """
        expected_data = bytes([1, 2, 3, 4, 5])
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / 'read_text_locally.txt'
            with open(output_file, 'wb') as f:
                f.write(expected_data)
            reconstructed_data = read_binary(output_file)
            self.assertEqual(expected_data, reconstructed_data)

    def test_read_binary_from_s3(self) -> None:
        """
        Tests that reading a binary file from S3 works properly.
        """
        download_bucket = 'ml-caches'
        download_key = 'my/file/path.data'
        expected_data = bytes([1, 2, 3, 4, 5])
        full_filepath = Path(f's3://{download_bucket}') / download_key

        async def patch_read_binary_file_contents_from_s3_async(s3_key: Path, s3_bucket: str) -> bytes:
            """
            A patch for the read_binary_file_contents_from_s3_async method.
            :param s3_key: The passed key
            :param s3_bucket: The passed bucket.
            """
            self.assertEqual(Path(download_key), s3_key)
            self.assertEqual(download_bucket, s3_bucket)
            return expected_data
        with patch_with_validation('nuplan.common.utils.io_utils.read_binary_file_contents_from_s3_async', patch_read_binary_file_contents_from_s3_async):
            reconstructed_data = read_binary(full_filepath)
            self.assertEqual(expected_data, reconstructed_data)

    def test_path_exists_locally(self) -> None:
        """
        Tests that path_exists works for local files.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            file_to_create = tmp_dir_path / 'existing.txt'
            file_to_not_create = tmp_dir_path / 'not_existing.txt'
            with open(file_to_create, 'w') as f:
                f.write('some irrelevant text.')
            self.assertTrue(path_exists(file_to_create))
            self.assertFalse(path_exists(file_to_not_create))
            self.assertTrue(path_exists(tmp_dir_path, include_directories=True))
            self.assertFalse(path_exists(tmp_dir_path, include_directories=False))

    def test_path_exists_s3(self) -> None:
        """
        Tests that path_exists works for s3 files.
        """
        test_bucket = 'ml-caches'
        test_parent_dir = 'my/file/that'
        test_existing_file = f'{test_parent_dir}/exists.txt'
        test_non_existing_file = f'{test_parent_dir}/does_not_exist.txt'
        test_dir_path = Path(f's3://{test_bucket}') / test_parent_dir
        test_existing_path = Path(f's3://{test_bucket}') / test_existing_file
        test_non_existing_path = Path(f's3://{test_bucket}') / test_non_existing_file

        async def patch_check_s3_object_exists_async(s3_key: Path, s3_bucket: str) -> bool:
            """
            Patches the check_s3_object_exists_async method.
            :param key: The s3 key to check.
            :param bucket: The s3 bucket to check.
            :return: The mocked return value.
            """
            self.assertEqual(test_bucket, s3_bucket)
            if str(s3_key) == test_existing_file:
                return True
            elif str(s3_key) in [test_non_existing_file, test_parent_dir]:
                return False
            self.fail(f'Unexpected path passed to check_s3_object_exists patch: {s3_key}')

        async def patch_check_s3_path_exists_async(s3_path: str) -> bool:
            """
            Patches the check_s3_object_exists_async method.
            :param s3_path: The s3 path to check.
            :return: The mocked return value.
            """
            if s3_path in [safe_path_to_string(test_existing_path), safe_path_to_string(test_dir_path)]:
                return True
            elif s3_path == safe_path_to_string(test_non_existing_path):
                return False
            self.fail(f'Unexpected path passed to check_s3_path_exists patch: {s3_path}')
        with patch_with_validation('nuplan.common.utils.io_utils.check_s3_object_exists_async', patch_check_s3_object_exists_async), patch_with_validation('nuplan.common.utils.io_utils.check_s3_path_exists_async', patch_check_s3_path_exists_async):
            self.assertTrue(path_exists(test_existing_path))
            self.assertFalse(path_exists(test_non_existing_path))
            self.assertTrue(path_exists(test_dir_path, include_directories=True))
            self.assertFalse(path_exists(test_dir_path, include_directories=False))

    def test_list_files_in_directory_locally(self) -> None:
        """
        Tests that list_files_in_directory works for local files.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            self.assertEqual(list_files_in_directory(tmp_dir_path), [])
            test_file_contents = {'a.txt': 'test file a.', 'b.txt': 'test file b.'}
            for filename, contents in test_file_contents.items():
                with open(tmp_dir_path / filename, 'w') as f:
                    f.write(contents)
            output_files_in_directory = list_files_in_directory(tmp_dir_path)
            self.assertEqual(len(output_files_in_directory), len(test_file_contents))
            for output_filepath in output_files_in_directory:
                self.assertIn(output_filepath.name, test_file_contents)

    def test_list_files_in_directory_s3(self) -> None:
        """
        Tests that list_files_in_directory works for s3.
        """
        test_bucket = 'ml-caches'
        test_directory_key = Path('test_dir')
        test_directory_s3_path = Path(f's3://{test_bucket}/{test_directory_key}')
        test_files_in_s3 = ['a.txt', 'b.txt']
        expected_files = [Path(f'{test_directory_key}/{filename}') for filename in test_files_in_s3]
        expected_s3_paths = [Path(f's3://{test_bucket}') / filename for filename in expected_files]

        async def patch_list_files_in_s3_directory_async(s3_key: Path, s3_bucket: str, filter_suffix: str='') -> List[Path]:
            """
            Patches the list_files_in_s3_directory_async method.
            :param key: The s3 key of the directory.
            :param bucket: The s3 bucket of the directory.
            :param filter_suffix: Unused.
            :return: The mocked return value.
            """
            self.assertEqual(test_bucket, s3_bucket)
            self.assertEqual(test_directory_key, s3_key)
            return expected_files
        with patch_with_validation('nuplan.common.utils.io_utils.list_files_in_s3_directory_async', patch_list_files_in_s3_directory_async):
            output_filepaths = list_files_in_directory(test_directory_s3_path)
            self.assertEqual(output_filepaths, expected_s3_paths)

    def test_delete_file_locally(self) -> None:
        """
        Tests that delete_file works for local files.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            test_file_contents = {'a.txt': 'test file a.', 'b.txt': 'test file b.'}
            test_file_paths = [tmp_dir_path / filename for filename in test_file_contents]
            for filename, contents in test_file_contents.items():
                with open(tmp_dir_path / filename, 'w') as f:
                    f.write(contents)
            self.assertEqual(set(tmp_dir_path.iterdir()), set(test_file_paths))
            for filename in test_file_contents:
                filepath = tmp_dir_path / filename
                delete_file(filepath)
                self.assertNotIn(filepath, tmp_dir_path.iterdir())
            self.assertEqual(len(list(tmp_dir_path.iterdir())), 0)
            with self.assertRaises(ValueError):
                delete_file(tmp_dir_path)

    def test_delete_file_s3(self) -> None:
        """
        Tests that delete_file works for s3.
        """
        test_bucket = 'ml-caches'
        test_directory_key = Path('test_dir')
        test_directory_s3_path = Path(f's3://{test_bucket}/{test_directory_key}')
        test_files_in_s3 = {'a.txt', 'b.txt'}

        def get_s3_key(filename: str) -> Path:
            """
            Turns a filename into an s3 key.
            """
            return Path(f'{test_directory_key}/{filename}')

        def list_s3_keys() -> List[Path]:
            """
            Lists the keys in s3.
            :return: S3 keys in the mocked test directory.
            """
            return [get_s3_key(filename) for filename in test_files_in_s3]

        async def patch_list_files_in_s3_directory_async(s3_key: Path, s3_bucket: str, filter_suffix: str='') -> List[Path]:
            """
            Patches the list_files_in_s3_directory_async method.
            :param key: The s3 key of the directory.
            :param bucket: The s3 bucket of the directory.
            :param filter_suffix: Unused.
            :return: The mocked return value.
            """
            self.assertEqual(test_bucket, s3_bucket)
            self.assertEqual(test_directory_key, s3_key)
            return list_s3_keys()

        async def patch_delete_file_from_s3_async(s3_key: Path, s3_bucket: str) -> None:
            """
            Patches the delete_file_from_s3_async method.
            :param s3_key: The s3 key to delete.
            :param s3_bucket: The s3 bucket.
            """
            nonlocal test_files_in_s3
            self.assertEqual(test_bucket, s3_bucket)
            self.assertEqual(test_directory_key, s3_key.parent)
            self.assertIn(s3_key.name, test_files_in_s3)
            test_files_in_s3.remove(s3_key.name)
        with patch_with_validation('nuplan.common.utils.io_utils.list_files_in_s3_directory_async', patch_list_files_in_s3_directory_async), patch_with_validation('nuplan.common.utils.io_utils.delete_file_from_s3_async', patch_delete_file_from_s3_async):
            initial_s3_keys = list_s3_keys()
            for filename in test_files_in_s3:
                self.assertIn(get_s3_key(filename), initial_s3_keys)
            for filename in set(test_files_in_s3):
                s3_path = test_directory_s3_path / filename
                delete_file(s3_path)
                self.assertNotIn(get_s3_key(filename), list_s3_keys())

def save_buffer(output_path: Path, buf: bytes) -> None:
    """
    Saves a buffer to file synchronously.
    The path can either be local or S3.
    :param output_path: The output path to which to save.
    :param buf: The byte buffer to save.
    """
    asyncio.run(_save_buffer_async(output_path, buf))

@contextlib.contextmanager
def patch_with_validation(method_to_patch: str, patch_function: Callable[..., Any], override_function: Optional[Callable[..., Any]]=None, **kwargs: Any) -> Generator[Callable[..., Any], None, None]:
    """
    Wraps unittest.mock.patch, injecting the function signature validation.
    :param method_to_patch: The dot-string method to patch (e.g. "my.python.file.mymethod")
    :param patch_function: The function to use for the patch.
    :param override_function: The function to use for validation. If not provided, `method_to_patch` will be imported and used.
      The intent is to provide an escape hatch in the instance automatic lookup via dot-string for method_to_patch does not work.
    :param kwargs: The additional keyword arguments passed to unittest.mock.patch.
    """
    if override_function is None:
        override_function = _get_method_from_import(method_to_patch)
    assert_functions_swappable(override_function, patch_function)
    with unittest.mock.patch(method_to_patch, patch_function, **kwargs) as mock_obj:
        yield mock_obj

def save_object_as_pickle(output_path: Path, obj: Any) -> None:
    """
    Pickles the output object and saves it to the provided path.
    The path can be a local path or an S3 path.
    :param output_path: The output path to which to save.
    :param obj: The object to save. Must be picklable.
    """
    asyncio.run(save_object_as_pickle_async(output_path, obj))

def save_text(output_path: Path, text: str) -> None:
    """
    Saves the provided text string to the given output path.
    The path can be a local path or an S3 path.
    :param output_path: The output path to which to save.
    :param obj: The text to save.
    """
    asyncio.run(save_text_async(output_path, text))

def read_text(path: Path) -> str:
    """
    Reads a text file from the provided path.
    The path can be a local path or an S3 path.
    :param path: The path to read.
    :return: The text of the file.
    """
    result: str = asyncio.run(read_binary_async(path)).decode('utf-8')
    return result

def read_binary(path: Path) -> bytes:
    """
    Reads binary data from the provided path into memory.
    The path can be a local path or an S3 path.
    :param path: The path to read.
    :return: The contents of the file, in binary format.
    """
    result: bytes = asyncio.run(read_binary_async(path))
    return result

def delete_file(path: Path) -> None:
    """
    Deletes a single file.
    The path can be a local path or an S3 path.
    :param path: Path of file to delete.
    """
    asyncio.run(delete_file_async(path))

class TestPatch(unittest.TestCase):
    """
    A class to test the patch utils.
    """

    def test_patch_with_validation_correct_patch(self) -> None:
        """
        Tests that the patch works with a correct patch.
        """

        def correct_patch(x: int) -> int:
            """
            A correct patch for the base_method.
            :param x: The input.
            :return: The output.
            """
            return x + 1
        with patch_with_validation('nuplan.common.utils.test.patch_test_methods.base_method', correct_patch):
            result = complex_method(x=1, y=2)
            self.assertEqual(4, result)

    def test_patch_with_validation_correct_patch_direct(self) -> None:
        """
        Tests that the patch works with a correct patch that is directly provided.
        """

        def correct_patch(x: int) -> int:
            """
            A correct patch for the base_method.
            :param x: The input.
            :return: The output.
            """
            return x + 1
        with patch_with_validation('nuplan.common.utils.test.patch_test_methods.base_method', correct_patch, override_function=swappable_with_base_method):
            result = complex_method(x=1, y=2)
            self.assertEqual(4, result)

    def test_patch_raises_with_incorrect_patch(self) -> None:
        """
        Tests that an incorrect patch causes an error to be rasied.
        """

        def incorrect_patch(x: float) -> int:
            """
            An incorrect patch for the _base_method.
            :param x: The input.
            :return: The output.
            """
            return int(x) + 1
        with self.assertRaises(TypeError):
            with patch_with_validation('nuplan.common.utils.test.patch_test_methods.base_method', incorrect_patch):
                _ = complex_method(x=1, y=2)

    def test_patch_raises_with_incorrect_patch_direct(self) -> None:
        """
        Tests that an incorrect patch causes an error to be rasied.
        """

        def incorrect_patch(x: float) -> int:
            """
            An incorrect patch for the _base_method.
            :param x: The input.
            :return: The output.
            """
            return int(x) + 1
        with self.assertRaises(TypeError):
            with patch_with_validation('nuplan.common.utils.test.patch_test_methods.base_method', incorrect_patch, override_function=swappable_with_base_method):
                _ = complex_method(x=1, y=2)

def complex_method(x: int, y: int) -> int:
    """
    A mock complex method to use with the patch tests.
    :param x: One input parameter.
    :param y: The other input parameter.
    :return: The output.
    """
    xx = base_method(x)
    return xx * y

def base_method(x: int) -> int:
    """
    A base method that should be patched.
    :param x: The input.
    :return: The output.
    """
    raise RuntimeError('Should be patched.')

def _get_method_from_import(import_str: str) -> Callable[..., Any]:
    """
    Gets the method referenced by an import in the form that `unittest.mock.patch` expects.
    That is, given the string "foo.bar.baz.qux", imports "qux" from "foo.bar.baz"
    This is not a general purpose utility, so other import mechanisms (e.g. "from x import y")
      will not work.
    :param import_str: The import str.
    :return: The method.
    """
    import_path, method_name = import_str.rsplit('.', 1)
    module = importlib.import_module(import_path)
    method = cast(Callable[..., Any], getattr(module, method_name))
    return method

@dataclass
class SimulationLog:
    """Simulation log."""
    file_path: Path
    scenario: AbstractScenario
    planner: AbstractPlanner
    simulation_history: SimulationHistory

    def _dump_to_pickle(self) -> None:
        """
        Dump file into compressed pickle.
        """
        pickle_object = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        save_buffer(self.file_path, lzma.compress(pickle_object, preset=0))

    def _dump_to_msgpack(self) -> None:
        """
        Dump file into compressed msgpack.
        """
        pickle_object = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        msg_packed_bytes = msgpack.packb(pickle_object)
        save_buffer(self.file_path, lzma.compress(msg_packed_bytes, preset=0))

    def save_to_file(self) -> None:
        """
        Dump simulation log into file.
        """
        serialization_type = self.simulation_log_type(self.file_path)
        if serialization_type == 'pickle':
            self._dump_to_pickle()
        elif serialization_type == 'msgpack':
            self._dump_to_msgpack()
        else:
            raise ValueError(f'Unknown option: {serialization_type}')

    @staticmethod
    def simulation_log_type(file_path: Path) -> str:
        """
        Deduce the simulation log type based on the last two portions of the suffix.
        The last suffix must be .xz, since we always dump/load to/from an xz container.
        If the second to last suffix is ".msgpack", assumes the log is of type "msgpack".
        If the second to last suffix is ".pkl", assumes the log is of type "pickle."
        If it's neither, raises a ValueError.
        Examples:
        - "/foo/bar/baz.1.2.pkl.xz" -> "pickle"
        - "/foo/bar/baz/1.2.msgpack.xz" -> "msgpack"
        - "/foo/bar/baz/1.2.msgpack.pkl.xz" -> "pickle"
        - "/foo/bar/baz/1.2.msgpack" -> Error
        :param file_path: File path.
        :return: one from ["msgpack", "pickle"].
        """
        if len(file_path.suffixes) < 2:
            raise ValueError(f'Inconclusive file type: {file_path}')
        last_suffix = file_path.suffixes[-1]
        if last_suffix != '.xz':
            raise ValueError(f'Inconclusive file type: {file_path}')
        second_to_last_suffix = file_path.suffixes[-2]
        log_type_mapping = {'.msgpack': 'msgpack', '.pkl': 'pickle'}
        if second_to_last_suffix not in log_type_mapping:
            raise ValueError(f'Inconclusive file type: {file_path}')
        return log_type_mapping[second_to_last_suffix]

    @classmethod
    def load_data(cls, file_path: Path) -> Any:
        """Load simulation log."""
        simulation_log_type = SimulationLog.simulation_log_type(file_path=file_path)
        if simulation_log_type == 'msgpack':
            with lzma.open(str(file_path), 'rb') as f:
                data = msgpack.unpackb(f.read())
                data = pickle.loads(data)
        elif simulation_log_type == 'pickle':
            with lzma.open(str(file_path), 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f'Unknown serialization type: {simulation_log_type}!')
        return data

def _dump_to_json(file: pathlib.Path, scene_to_save: Any) -> None:
    """Dump file into json"""
    scene_json = json.dumps(scene_to_save)
    save_text(file.with_suffix('.json'), scene_json)

def _dump_to_pickle(file: pathlib.Path, scene_to_save: Any) -> None:
    """Dump file into compressed pickle"""
    pickle_object = pickle.dumps(scene_to_save, protocol=pickle.HIGHEST_PROTOCOL)
    save_buffer(file.with_suffix('.pkl.xz'), lzma.compress(pickle_object, preset=0))

def _dump_to_msgpack(file: pathlib.Path, scene_to_save: Any) -> None:
    """Dump file into compressed msgpack"""
    msg_packed_bytes = msgpack.packb(scene_to_save)
    save_buffer(file.with_suffix('.msgpack.xz'), lzma.compress(msg_packed_bytes, preset=0))

def _dump_to_file(file: pathlib.Path, scene_to_save: Any, serialization_type: str) -> None:
    """
    Dump scene into file
    :param serialization_type: type of serialization ["json", "pickle", "msgpack"]
    :param file: file name
    :param scene_to_save: what to store
    """
    if serialization_type == 'json':
        _dump_to_json(file, scene_to_save)
    elif serialization_type == 'pickle':
        _dump_to_pickle(file, scene_to_save)
    elif serialization_type == 'msgpack':
        _dump_to_msgpack(file, scene_to_save)
    else:
        raise ValueError(f'Unknown option: {serialization_type}')

class SerializationCallback(AbstractCallback):
    """Callback for serializing scenes at the end of the simulation."""

    def __init__(self, output_directory: Union[str, pathlib.Path], folder_name: Union[str, pathlib.Path], serialization_type: str, serialize_into_single_file: bool):
        """
        Construct serialization callback
        :param output_directory: where scenes should be serialized
        :param folder_name: folder where output should be serialized
        :param serialization_type: A way to serialize output, options: ["json", "pickle", "msgpack"]
        :param serialize_into_single_file: if true all data will be in single file, if false, each time step will
                be serialized into a separate file
        """
        available_formats = ['json', 'pickle', 'msgpack']
        if serialization_type not in available_formats:
            raise ValueError(f'The serialization callback will not store files anywhere!Choose at least one format from {available_formats} instead of {serialization_type}!')
        self._output_directory = pathlib.Path(output_directory) / folder_name
        self._serialization_type = serialization_type
        self._serialize_into_single_file = serialize_into_single_file

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """
        Create directory at initialization
        :param setup: simulation setup
        :param planner: planner before initialization
        """
        scenario_directory = self._get_scenario_folder(planner.name(), setup.scenario)
        scenario_directory.mkdir(exist_ok=True, parents=True)

    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_end(self, setup: SimulationSetup, planner: AbstractPlanner, trajectory: AbstractTrajectory) -> None:
        """Inherited, see superclass."""
        pass

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """Inherited, see superclass."""
        pass

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """
        On reached_end validate that all steps were correctly serialized
        :param setup: simulation setup
        :param planner: planner when simulation ends
        :param history: resulting from simulation
        """
        number_of_scenes = len(history)
        if number_of_scenes == 0:
            raise RuntimeError('Number of scenes has to be greater than 0')
        scenario_directory = self._get_scenario_folder(planner.name(), setup.scenario)
        scenario = setup.scenario
        expert_trajectory = list(scenario.get_expert_ego_trajectory())
        scenes = [convert_sample_to_scene(map_name=scenario.map_api.map_name, database_interval=scenario.database_interval, traffic_light_status=scenario.get_traffic_light_status_at_iteration(index), expert_trajectory=expert_trajectory, mission_goal=scenario.get_mission_goal(), data=sample, colors=TrajectoryColors()) for index, sample in enumerate(history.data)]
        self._serialize_scenes(scenes, scenario_directory)

    def _serialize_scenes(self, scenes: List[Dict[str, Any]], scenario_directory: pathlib.Path) -> None:
        """
        Serialize scenes based on callback setup to json/pickle or other
        :param scenes: scenes to be serialized
        :param scenario_directory: directory where they should be serialized
        """
        if not self._serialize_into_single_file:
            for scene in scenes:
                file_name = scenario_directory / str(scene['ego']['timestamp_us'])
                _dump_to_file(file_name, scene, self._serialization_type)
        else:
            file_name = scenario_directory / scenario_directory.name
            _dump_to_file(file_name, scenes, self._serialization_type)

    def _get_scenario_folder(self, planner_name: str, scenario: AbstractScenario) -> pathlib.Path:
        """
        Compute scenario folder directory where all files will be stored
        :param planner_name: planner name
        :param scenario: for which to compute directory name
        :return directory path
        """
        return self._output_directory / planner_name / scenario.scenario_type / scenario.log_name / scenario.scenario_name

@dataclass
class NuBoardFile:
    """Data class to save nuBoard file info."""
    simulation_main_path: str
    metric_main_path: str
    metric_folder: str
    aggregator_metric_folder: str
    simulation_folder: Optional[str] = None
    current_path: Optional[pathlib.Path] = None

    @classmethod
    def extension(cls) -> str:
        """Return nuboard file extension."""
        return '.nuboard'

    def __eq__(self, other: object) -> bool:
        """
        Comparison between two NuBoardFile.
        :param other: Other object.
        :return True if both objects are same.
        """
        if not isinstance(other, NuBoardFile):
            return NotImplemented
        return other.simulation_main_path == self.simulation_main_path and other.simulation_folder == self.simulation_folder and (other.metric_main_path == self.metric_main_path) and (other.metric_folder == self.metric_folder) and (other.aggregator_metric_folder == self.aggregator_metric_folder) and (other.current_path == self.current_path)

    def save_nuboard_file(self, filename: pathlib.Path) -> None:
        """
        Save NuBoardFile data class to a file.
        :param filename: The saved file path.
        """
        save_object_as_pickle(filename, self.serialize())

    @classmethod
    def load_nuboard_file(cls, filename: pathlib.Path) -> NuBoardFile:
        """
        Read a NuBoard file to NuBoardFile data class.
        :file: NuBoard file path.
        """
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return cls.deserialize(data=data)

    def serialize(self) -> Dict[str, str]:
        """
        Serialization of NuBoardFile data class to dictionary.
        :return A serialized dictionary class.
        """
        as_dict = {'simulation_main_path': self.simulation_main_path, 'metric_main_path': self.metric_main_path, 'metric_folder': self.metric_folder, 'aggregator_metric_folder': self.aggregator_metric_folder}
        if self.simulation_folder is not None:
            as_dict['simulation_folder'] = self.simulation_folder
        return as_dict

    @classmethod
    def deserialize(cls, data: Dict[str, str]) -> NuBoardFile:
        """
        Deserialization of a NuBoard file into NuBoardFile data class.
        :param data: A serialized nuboard file data.
        :return A NuBoard file data class.
        """
        simulation_main_path = data['simulation_main_path'].replace('//', '/')
        metric_main_path = data['metric_main_path'].replace('//', '/')
        return NuBoardFile(simulation_main_path=simulation_main_path, simulation_folder=data.get('simulation_folder', None), metric_main_path=metric_main_path, metric_folder=data['metric_folder'], aggregator_metric_folder=data['aggregator_metric_folder'])

class MetricsEngine:
    """The metrics engine aggregates and manages the instantiated metrics for a scenario."""

    def __init__(self, main_save_path: Path, metrics: Optional[List[AbstractMetricBuilder]]=None) -> None:
        """
        Initializer for MetricsEngine class
        :param metrics: Metric objects.
        """
        self._main_save_path = main_save_path
        if not is_s3_path(self._main_save_path):
            self._main_save_path.mkdir(parents=True, exist_ok=True)
        if metrics is None:
            self._metrics: List[AbstractMetricBuilder] = []
        else:
            self._metrics = metrics

    @property
    def metrics(self) -> List[AbstractMetricBuilder]:
        """Retrieve a list of metric results."""
        return self._metrics

    def add_metric(self, metric_builder: AbstractMetricBuilder) -> None:
        """TODO: Create the list of types needed from the history"""
        self._metrics.append(metric_builder)

    def write_to_files(self, metric_files: Dict[str, List[MetricFile]]) -> None:
        """
        Write to a file by constructing a dataframe
        :param metric_files: A dictionary of scenario names and a list of their metric files.
        """
        for scenario_name, metric_files in metric_files.items():
            file_name = scenario_name + JSON_FILE_EXTENSION
            save_path = self._main_save_path / file_name
            dataframes = []
            for metric_file in metric_files:
                metric_file_key = metric_file.key
                for metric_statistic in metric_file.metric_statistics:
                    dataframe = construct_dataframe(log_name=metric_file_key.log_name, scenario_name=metric_file_key.scenario_name, scenario_type=metric_file_key.scenario_type, planner_name=metric_file_key.planner_name, metric_statistics=metric_statistic)
                    dataframes.append(dataframe)
            if len(dataframes):
                save_object_as_pickle(save_path, dataframes)

    def compute_metric_results(self, history: SimulationHistory, scenario: AbstractScenario) -> Dict[str, List[MetricStatistics]]:
        """
        Compute metrics in the engine
        :param history: History from simulation
        :param scenario: Scenario running this metric engine
        :return A list of metric statistics.
        """
        metric_results = {}
        for metric in self._metrics:
            try:
                start_time = time.perf_counter()
                metric_results[metric.name] = metric.compute(history, scenario=scenario)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                logger.debug(f'Metric: {metric.name} running time: {elapsed_time:.2f} seconds.')
            except (NotImplementedError, Exception) as e:
                logger.error(f'Running {metric.name} with error: {e}')
                raise RuntimeError(f'Metric Engine failed with: {e}')
        return metric_results

    def compute(self, history: SimulationHistory, scenario: AbstractScenario, planner_name: str) -> Dict[str, List[MetricFile]]:
        """
        Compute metrics and return in a format of MetricStorageResult for each metric computation
        :param history: History from simulation
        :param scenario: Scenario running this metric engine
        :param planner_name: name of the planner
        :return A dictionary of scenario name and list of MetricStorageResult.
        """
        all_metrics_results = self.compute_metric_results(history=history, scenario=scenario)
        metric_files = defaultdict(list)
        for metric_name, metric_statistics_results in all_metrics_results.items():
            metric_file_key = MetricFileKey(metric_name=metric_name, log_name=scenario.log_name, scenario_name=scenario.scenario_name, scenario_type=scenario.scenario_type, planner_name=planner_name)
            metric_file = MetricFile(key=metric_file_key, metric_statistics=metric_statistics_results)
            metric_file_name = scenario.scenario_type + '_' + scenario.scenario_name + '_' + planner_name
            metric_files[metric_file_name].append(metric_file)
        return metric_files

def construct_dataframe(log_name: str, scenario_name: str, scenario_type: str, planner_name: str, metric_statistics: MetricStatistics) -> Dict[str, Any]:
    """
    Construct a metric dataframe for metric results.
    :param log_name: A log name.
    :param scenario_name: Scenario name.
    :param scenario_type: Scenario type.
    :param planner_name: Planner name.
    :param metric_statistics: Metric statistics.
    :return A pandas dataframe for metric statistics.
    """
    statistic_columns = {'log_name': log_name, 'scenario_name': scenario_name, 'scenario_type': scenario_type, 'planner_name': planner_name, 'metric_computator': metric_statistics.metric_computator, 'metric_statistics_name': metric_statistics.name}
    statistic_columns.update(metric_statistics.serialize_dataframe())
    return statistic_columns

def update_config_for_training(cfg: DictConfig) -> None:
    """
    Updates the config based on some conditions.
    :param cfg: omegaconf dictionary that is used to run the experiment.
    """
    OmegaConf.set_struct(cfg, False)
    if cfg.cache.cache_path is None:
        logger.warning('Parameter cache_path is not set, caching is disabled')
    elif not str(cfg.cache.cache_path).startswith('s3://'):
        if cfg.cache.cleanup_cache and Path(cfg.cache.cache_path).exists():
            rmtree(cfg.cache.cache_path)
        Path(cfg.cache.cache_path).mkdir(parents=True, exist_ok=True)
    if cfg.lightning.trainer.overfitting.enable:
        cfg.data_loader.params.num_workers = 0
    if cfg.gpu and torch.cuda.is_available():
        cfg.lightning.trainer.params.gpus = -1
    else:
        cfg.lightning.trainer.params.gpus = None
        cfg.lightning.trainer.params.accelerator = None
        cfg.lightning.trainer.params.precision = 32
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)
    if cfg.log_config:
        logger.info(f'Creating experiment name [{cfg.experiment}] in group [{cfg.group}] with config...')
        logger.info('\n' + OmegaConf.to_yaml(cfg))

class TestUtilsConfig(unittest.TestCase):
    """Tests for the non-distributed training functions in utils_config.py."""
    specific_world_size = 4

    @staticmethod
    def _generate_mock_training_config() -> DictConfig:
        """
        Returns a mock training configuration with sensible default values.
        :return: DictConfig representing the training configuration.
        """
        return DictConfig({'log_config': True, 'experiment': 'mock_experiment_name', 'group': 'mock_group_name', 'cache': {'cleanup_cache': False, 'cache_path': None}, 'data_loader': {'params': {'num_workers': None}}, 'lightning': {'trainer': {'params': {'gpus': None, 'accelerator': None, 'precision': None}, 'overfitting': {'enable': False}}}, 'gpu': False})

    @staticmethod
    def _generate_mock_simulation_config() -> DictConfig:
        """
        Returns a mock simulation configuration with sensible default values.
        :return: DictConfig representing the simulation configuration.
        """
        return DictConfig({'log_config': True, 'experiment': 'mock_experiment_name', 'group': 'mock_group_name', 'callback': {'timing_callback': {'_target_': 'nuplan.planning.simulation.callback.timing_callback.TimingCallback'}, 'simulation_log_callback': {'_target_': 'nuplan.planning.simulation.callback.simulation_log_callback.SimulationLogCallback'}, 'metric_callback': {'_target_': 'nuplan.planning.simulation.callback.metric_callback.MetricCallback'}}})

    @staticmethod
    def _patch_return_false() -> bool:
        """A patch function that will always return False."""
        return False

    @staticmethod
    def _patch_return_true() -> bool:
        """A patch function that will always return True."""
        return True

    @given(cache_path=st.one_of(st.none(), st.just('s3://bucket/key')))
    @settings(deadline=None)
    def test_update_config_for_training_cache_path_none_or_s3(self, cache_path: Optional[str]) -> None:
        """
        Tests the behavior of update_config_for_training when the supplied cfg.cache.cache_path is either
        None or an S3 path.
        """
        mock_config = TestUtilsConfig._generate_mock_training_config()
        mock_config.cache.cache_path = cache_path
        with TemporaryDirectory() as tmp_dir:
            with patch_with_validation('torch.cuda.is_available', TestUtilsConfig._patch_return_false):
                update_config_for_training(mock_config)
            self.assertTrue(Path(tmp_dir).exists())

    def test_update_config_for_training_cache_path_local_non_existing(self) -> None:
        """
        Tests the behavior of update_config_for_training when the supplied cfg.cache.cache_path doesn't exist yet.
        """
        mock_config = TestUtilsConfig._generate_mock_training_config()
        with TemporaryDirectory() as tmp_dir:
            mock_config.cache.cache_path = tmp_dir
            rmtree(tmp_dir)
            with patch_with_validation('torch.cuda.is_available', TestUtilsConfig._patch_return_false):
                update_config_for_training(mock_config)
            self.assertTrue(Path(tmp_dir).exists())

    def test_update_config_for_training_cache_path_local_cleanup(self) -> None:
        """
        Tests the behavior of update_config_for_training when the supplied cfg.cache.cache_path exists and
        cleanup_cache is requested.
        """
        mock_config = TestUtilsConfig._generate_mock_training_config()
        with TemporaryDirectory() as tmp_dir:
            _, tmp_file = mkstemp(dir=tmp_dir)
            mock_config.cache.cache_path = tmp_dir
            mock_config.cache.cleanup_cache = True
            self.assertTrue(Path(tmp_file).exists())
            with patch_with_validation('torch.cuda.is_available', TestUtilsConfig._patch_return_false):
                update_config_for_training(mock_config)
            self.assertFalse(Path(tmp_file).exists())
            self.assertTrue(Path(tmp_dir).exists())
        self.assertFalse(Path(tmp_dir).exists())

    def test_update_config_for_training_overfitting(self) -> None:
        """
        Tests the behavior of update_config_for_training in regard to overfitting configurations.
        """
        mock_config = TestUtilsConfig._generate_mock_training_config()
        num_workers = 32
        mock_config.data_loader.params.num_workers = num_workers
        mock_config.lightning.trainer.overfitting.enable = False
        with patch_with_validation('torch.cuda.is_available', TestUtilsConfig._patch_return_false):
            update_config_for_training(mock_config)
        self.assertEqual(num_workers, mock_config.data_loader.params.num_workers)
        mock_config.lightning.trainer.overfitting.enable = True
        with patch_with_validation('torch.cuda.is_available', TestUtilsConfig._patch_return_false):
            update_config_for_training(mock_config)
        self.assertEqual(0, mock_config.data_loader.params.num_workers)

    @given(is_gpu_enabled=st.booleans(), is_cuda_available=st.booleans())
    @settings(deadline=None)
    def test_update_config_for_training_gpu(self, is_gpu_enabled: bool, is_cuda_available: bool) -> None:
        """
        Tests the behavior of update_config_for_training in regard to gpu configurations.
        """
        invalid_value = -99
        cuda_patch = TestUtilsConfig._patch_return_true if is_cuda_available else TestUtilsConfig._patch_return_false

        def get_expected_gpu_config(gpu_enabled: bool, cuda_available: bool) -> Optional[int]:
            return -1 if gpu_enabled and cuda_available else None

        def get_expected_accelerator_config(gpu_enabled: bool, cuda_available: bool) -> Optional[int]:
            return invalid_value if gpu_enabled and cuda_available else None

        def get_expected_precision_config(gpu_enabled: bool, cuda_available: bool) -> Optional[int]:
            return invalid_value if gpu_enabled and cuda_available else 32
        mock_config = TestUtilsConfig._generate_mock_training_config()
        mock_config.gpu = is_gpu_enabled
        mock_config.lightning.trainer.params.gpus = invalid_value
        mock_config.lightning.trainer.params.accelerator = invalid_value
        mock_config.lightning.trainer.params.precision = invalid_value
        with patch_with_validation('torch.cuda.is_available', cuda_patch):
            update_config_for_training(mock_config)
        self.assertEqual(get_expected_gpu_config(is_gpu_enabled, is_cuda_available), mock_config.lightning.trainer.params.gpus)
        self.assertEqual(get_expected_accelerator_config(is_gpu_enabled, is_cuda_available), mock_config.lightning.trainer.params.accelerator)
        self.assertEqual(get_expected_precision_config(is_gpu_enabled, is_cuda_available), mock_config.lightning.trainer.params.precision)

    @given(max_number_of_workers=st.one_of(st.none(), st.just(0)))
    @settings(deadline=None)
    def test_update_config_for_simulation_falsy_max_number_of_workers(self, max_number_of_workers: int) -> None:
        """
        Tests that update_config_for_simulation works as expected.
        When max number of workers is falsy, timing_callback won't be removed
        """
        mock_config = TestUtilsConfig._generate_mock_simulation_config()
        mock_config.max_number_of_workers = max_number_of_workers
        update_config_for_simulation(mock_config)
        self.assertEqual(3, len(mock_config.callback))

    @given(max_number_of_workers=st.integers(min_value=1))
    @settings(deadline=None)
    def test_update_config_for_simulation_truthy_max_number_of_workers(self, max_number_of_workers: int) -> None:
        """
        Tests that update_config_for_simulation works as expected. When max number of workers is truthy, a new
        `callbacks` entry will be added. The values are taken from `callback` with timing_callback target removed.
        """
        mock_config = TestUtilsConfig._generate_mock_simulation_config()
        mock_config.max_number_of_workers = max_number_of_workers
        update_config_for_simulation(mock_config)
        self.assertEqual(3, len(mock_config.callback))
        self.assertEqual(2, len(mock_config.callbacks))
        callbacks_targets = [callback['_target_'] for callback in mock_config.callbacks]
        self.assertNotIn('nuplan.planning.simulation.callback.timing_callback.TimingCallback', callbacks_targets)

    def test_update_config_for_nuboard(self) -> None:
        """Tests that update_config_for_nuboard works as expected."""
        mock_config = DictConfig({'log_config': True})
        mock_config.simulation_path = None
        update_config_for_nuboard(mock_config)
        self.assertIsNotNone(mock_config.simulation_path)
        self.assertEqual(0, len(mock_config.simulation_path))
        simulation_path_list = ['/mock/path', '/to/somewhere']
        mock_config.simulation_path = simulation_path_list
        update_config_for_nuboard(mock_config)
        self.assertEqual(simulation_path_list, mock_config.simulation_path)
        simulation_path_list_config = ListConfig(element_type=str, content=['/mock/path', '/to/somewhere'])
        mock_config.simulation_path = simulation_path_list_config
        update_config_for_nuboard(mock_config)
        self.assertEqual(simulation_path_list_config, mock_config.simulation_path)
        simulation_path = '/mock/path'
        mock_config.simulation_path = simulation_path
        update_config_for_nuboard(mock_config)
        expected_simulation_path_list = [simulation_path]
        self.assertEqual(expected_simulation_path_list, mock_config.simulation_path)

    @patch.dict(os.environ, {'WORLD_SIZE': str(specific_world_size)}, clear=True)
    def test_get_num_gpus_used_from_world_size(self) -> None:
        """
        Tests that that get_num_gpus_used works as expected. When WORLD_SIZE is set to a specific value, the function
        will simply return that value.
        """
        mock_config = DictConfig({})
        num_gpus = get_num_gpus_used(mock_config)
        self.assertEqual(self.specific_world_size, num_gpus)

    @given(num_gpus_config=st.integers(min_value=-1), cuda_device_count=st.integers(min_value=0), num_nodes=st.integers(min_value=1))
    @example(num_gpus_config=-1, cuda_device_count=2, num_nodes=2)
    @settings(deadline=None)
    def test_get_num_gpus_used_from_config(self, num_gpus_config: int, cuda_device_count: int, num_nodes: int) -> None:
        """
        Tests that that get_num_gpus_used works as expected when WORLD_SIZE environment variable is not set.
        """

        def patch_get_cuda_device_count() -> int:
            return cuda_device_count
        with patch.dict(os.environ, {'NUM_NODES': str(num_nodes)}, clear=True), patch_with_validation('torch.cuda.device_count', patch_get_cuda_device_count):
            mock_config = TestUtilsConfig._generate_mock_training_config()
            mock_config.lightning.trainer.params.gpus = num_gpus_config
            num_gpus = get_num_gpus_used(mock_config)
            expected_num_gpus = num_gpus_config if num_gpus_config != -1 else cuda_device_count * num_nodes
            self.assertEqual(expected_num_gpus, num_gpus)

    def test_get_num_gpus_used_invalid_config(self) -> None:
        """
        Tests that that get_num_gpus_used raises a RuntimeError when WORLD_SIZE environment variable is not set and
        a string is passed as the value of mock_config.lightning.trainer.params.gpus.
        """
        mock_config = TestUtilsConfig._generate_mock_training_config()
        mock_config.lightning.trainer.params.gpus = '1'
        with self.assertRaises(RuntimeError):
            get_num_gpus_used(mock_config)

def get_num_gpus_used(cfg: DictConfig) -> int:
    """
    Gets the number of gpus used in ddp by searching through the environment variable WORLD_SIZE, PytorchLightning Trainer specified number of GPUs, and torch.cuda.device_count() in that order.
    :param cfg: Config with experiment parameters.
    :return num_gpus: Number of gpus used in ddp.
    """
    num_gpus = os.getenv('WORLD_SIZE', -1)
    if num_gpus == -1:
        logger.info('WORLD_SIZE was not set.')
        trainer_num_gpus = cfg.lightning.trainer.params.gpus
        if isinstance(trainer_num_gpus, str):
            raise RuntimeError('Error, please specify gpus as integer. Received string.')
        trainer_num_gpus = cast(int, trainer_num_gpus)
        if trainer_num_gpus == -1:
            logger.info('PytorchLightning Trainer gpus was set to -1, finding number of GPUs used from torch.cuda.device_count().')
            cuda_num_gpus = torch.cuda.device_count() * int(os.getenv('NUM_NODES', 1))
            num_gpus = cuda_num_gpus
        else:
            logger.info(f'Trainer gpus was set to {trainer_num_gpus}, using this as the number of gpus.')
            num_gpus = trainer_num_gpus
    num_gpus = int(num_gpus)
    logger.info(f'Number of gpus found to be in use: {num_gpus}')
    return num_gpus

class TestDataLoader(unittest.TestCase):
    """
    Tests data loading functionality
    """

    def setUp(self) -> None:
        """Setup hydra config."""
        seed = 10
        pl.seed_everything(seed, workers=True)
        main_path = os.path.dirname(os.path.realpath(__file__))
        self.config_path = os.path.join(main_path, '../config/training/')
        self.group = tempfile.TemporaryDirectory()
        self.cache_path = os.path.join(self.group.name, 'cache_path')

    def tearDown(self) -> None:
        """Remove temporary folder."""
        self.group.cleanup()

    @staticmethod
    def validate_cfg(cfg: DictConfig) -> None:
        """Validate hydra config."""
        update_config_for_training(cfg)
        OmegaConf.set_struct(cfg, False)
        cfg.scenario_filter.limit_total_scenarios = 0.001
        cfg.data_loader.datamodule.train_fraction = 1.0
        cfg.data_loader.datamodule.val_fraction = 1.0
        cfg.data_loader.datamodule.test_fraction = 1.0
        cfg.data_loader.params.batch_size = 2
        cfg.data_loader.params.num_workers = 2
        cfg.data_loader.params.pin_memory = False
        OmegaConf.set_struct(cfg, True)

    @staticmethod
    def _iterate_dataloader(dataloader: torch.utils.data.DataLoader) -> None:
        """
        Iterate a fixed number of batches of the dataloader.
        :param dataloader: Data loader to iterate.
        """
        num_batches = 5
        dataloader_iter = iter(dataloader)
        iterations = min(len(dataloader), num_batches)
        for _ in range(iterations):
            next(dataloader_iter)

    def _run_dataloader(self, cfg: DictConfig) -> None:
        """
        Test that the training dataloader can be iterated without errors.
        :param cfg: Hydra config.
        """
        worker = build_worker(cfg)
        lightning_module_wrapper = build_torch_module_wrapper(cfg.model)
        datamodule = build_lightning_datamodule(cfg, worker, lightning_module_wrapper)
        datamodule.setup('fit')
        datamodule.setup('test')
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        test_dataloader = datamodule.test_dataloader()
        for dataloader in [train_dataloader, val_dataloader]:
            assert len(dataloader) > 0
            self._iterate_dataloader(dataloader)
        self._iterate_dataloader(test_dataloader)

    def test_dataloader(self) -> None:
        """Test dataloader on nuPlan DB."""
        log_names = ['2021.07.16.20.45.29_veh-35_01095_01486', '2021.08.17.18.54.02_veh-45_00665_01065', '2021.06.08.12.54.54_veh-26_04262_04732', '2021.10.06.07.26.10_veh-52_00006_00398']
        overrides = ['scenario_builder=nuplan_mini', 'worker=sequential', 'splitter=nuplan', f'scenario_filter.log_names={log_names}', f'group={self.group.name}', f'cache.cache_path={self.cache_path}', 'output_dir=${group}/${experiment}', 'scenario_type_weights=default_scenario_type_weights']
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*overrides, '+training=training_raster_model'])
            self.validate_cfg(cfg)
            self._run_dataloader(cfg)

