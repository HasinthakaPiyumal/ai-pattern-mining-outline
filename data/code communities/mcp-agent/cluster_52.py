# Cluster 52

def load_resource_content(resource_path: str, prompt_files: List[Path]) -> ResourceContent:
    """
    Load a resource's content and determine its mime type

    Args:
        resource_path: Path to the resource file
        prompt_files: List of prompt files (to find relative paths)

    Returns:
        Tuple of (content, mime_type, is_binary)
        - content: String content for text files, base64-encoded string for binary files
        - mime_type: The MIME type of the resource
        - is_binary: Whether the content is binary (and base64-encoded)

    Raises:
        FileNotFoundError: If the resource cannot be found
    """
    resource_file = find_resource_file(resource_path, prompt_files)
    if resource_file is None:
        raise FileNotFoundError(f'Resource not found: {resource_path}')
    mime_type = mime_utils.guess_mime_type(str(resource_file))
    is_binary = mime_utils.is_binary_content(mime_type)
    if is_binary:
        with open(resource_file, 'rb') as f:
            content = base64.b64encode(f.read()).decode('utf-8')
    else:
        with open(resource_file, 'r', encoding='utf-8') as f:
            content = f.read()
    return (content, mime_type, is_binary)

def find_resource_file(resource_path: str, prompt_files: List[Path]) -> Optional[Path]:
    """Find a resource file relative to one of the prompt files"""
    for prompt_file in prompt_files:
        potential_path = prompt_file.parent / resource_path
        if potential_path.exists():
            return potential_path
    return None

class TestFindResourceFile:

    def test_find_resource_file_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            prompt_file = tmppath / 'prompt.txt'
            prompt_file.write_text('test prompt')
            resource_file = tmppath / 'resource.txt'
            resource_file.write_text('test resource')
            found = find_resource_file('resource.txt', [prompt_file])
            assert found == resource_file

    def test_find_resource_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            prompt_file = tmppath / 'prompt.txt'
            prompt_file.write_text('test prompt')
            found = find_resource_file('nonexistent.txt', [prompt_file])
            assert found is None

    def test_find_resource_file_multiple_prompt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            subdir1 = tmppath / 'sub1'
            subdir2 = tmppath / 'sub2'
            subdir1.mkdir()
            subdir2.mkdir()
            prompt1 = subdir1 / 'prompt1.txt'
            prompt2 = subdir2 / 'prompt2.txt'
            prompt1.write_text('prompt 1')
            prompt2.write_text('prompt 2')
            resource_file = subdir2 / 'resource.txt'
            resource_file.write_text('test resource')
            found = find_resource_file('resource.txt', [prompt1, prompt2])
            assert found == resource_file

class TestLoadResourceContent:

    def test_load_resource_content_text_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            prompt_file = tmppath / 'prompt.txt'
            prompt_file.write_text('test')
            resource_file = tmppath / 'resource.txt'
            resource_file.write_text('Hello, world!', encoding='utf-8')
            content, mime_type, is_binary = load_resource_content('resource.txt', [prompt_file])
            assert content == 'Hello, world!'
            assert mime_type == 'text/plain'
            assert is_binary is False

    def test_load_resource_content_binary_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            prompt_file = tmppath / 'prompt.txt'
            prompt_file.write_text('test')
            resource_file = tmppath / 'image.png'
            binary_data = b'\x89PNG\r\n\x1a\n'
            resource_file.write_bytes(binary_data)
            content, mime_type, is_binary = load_resource_content('image.png', [prompt_file])
            expected_content = base64.b64encode(binary_data).decode('utf-8')
            assert content == expected_content
            assert mime_type == 'image/png'
            assert is_binary is True

    def test_load_resource_content_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            prompt_file = tmppath / 'prompt.txt'
            prompt_file.write_text('test')
            with pytest.raises(FileNotFoundError, match='Resource not found: nonexistent.txt'):
                load_resource_content('nonexistent.txt', [prompt_file])

