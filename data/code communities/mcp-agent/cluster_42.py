# Cluster 42

def validate_entrypoint(entrypoint_path: Path):
    """
    Validates the entrypoint file for the project.
    Raises an exception if the contents are not valid.
    """
    if not entrypoint_path.exists():
        raise FileNotFoundError(f'Entrypoint file {entrypoint_path} does not exist.')
    with open(entrypoint_path, 'r', encoding='utf-8') as f:
        content = f.read()
        has_app_def = re.search('^(\\w+)\\s*=\\s*MCPApp\\s*\\(', content, re.MULTILINE)
        if not has_app_def:
            raise ValueError('No MCPApp definition found in main.py.')
        has_main = re.search('(?m)^if\\s+__name__\\s*==\\s*[\\\'"]__main__[\\\'"]\\s*:\\n(?:[ \\t]+.*\\n?)*', content)
        if has_main:
            print_warning('Found a __main__ entrypoint in main.py. This will be ignored in the deployment.')

class TestValidateEntrypoint:
    """Tests for validate_entrypoint function."""

    def test_validate_entrypoint_success_simple(self):
        """Test validation of a simple valid entrypoint."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("app = MCPApp(name='test-app')")
            f.flush()
            validate_entrypoint(Path(f.name))

    def test_validate_entrypoint_success_multiline(self):
        """Test validation of a multiline MCPApp definition."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('\nfrom mcp_agent.cloud import MCPApp\n\nmy_app = MCPApp(\n    name="test-app",\n    description="My test app"\n)\n')
            f.flush()
            validate_entrypoint(Path(f.name))

    def test_validate_entrypoint_success_with_variable_name(self):
        """Test validation with different variable names for MCPApp."""
        test_cases = ['app = MCPApp()', 'my_app = MCPApp()', 'agent = MCPApp()', '_private_app = MCPApp()', 'app123 = MCPApp()']
        for content in test_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                f.flush()
                validate_entrypoint(Path(f.name))

    def test_validate_entrypoint_file_not_exists(self):
        """Test validation fails when entrypoint file doesn't exist."""
        non_existent_file = Path('/non/existent/file.py')
        with pytest.raises(FileNotFoundError, match='Entrypoint file .* does not exist'):
            validate_entrypoint(non_existent_file)

    def test_validate_entrypoint_no_mcpapp_definition(self):
        """Test validation fails when no MCPApp definition is found."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('\nimport os\nprint("Hello world")\n\ndef main():\n    pass\n')
            f.flush()
            with pytest.raises(ValueError, match='No MCPApp definition found in main.py'):
                validate_entrypoint(Path(f.name))

    def test_validate_entrypoint_invalid_mcpapp_patterns(self):
        """Test validation fails for invalid MCPApp patterns."""
        invalid_patterns = ['# app = MCPApp()', 'MCPApp()', "print('app = MCPApp()')", 'def create_app(): return MCPApp()']
        for content in invalid_patterns:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                f.flush()
                with pytest.raises(ValueError, match='No MCPApp definition found in main.py'):
                    validate_entrypoint(Path(f.name))

    @patch('mcp_agent.cli.cloud.commands.deploy.validation.print_warning')
    def test_validate_entrypoint_warns_about_main_block(self, mock_print_warning):
        """Test that validation warns about __main__ entrypoint."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('\napp = MCPApp()\n\nif __name__ == "__main__":\n    app.run()\n')
            f.flush()
            validate_entrypoint(Path(f.name))
            mock_print_warning.assert_called_once_with('Found a __main__ entrypoint in main.py. This will be ignored in the deployment.')

    @patch('mcp_agent.cli.cloud.commands.deploy.validation.print_warning')
    def test_validate_entrypoint_warns_about_main_block_variations(self, mock_print_warning):
        """Test warning for different __main__ block variations."""
        main_block_variations = ['if __name__ == "__main__":\n    app.run()', "if __name__ == '__main__':\n    app.run()", 'if __name__ == "__main__":\n    # comment\n    app.run()', 'if __name__ == "__main__":\n    pass\n    app.run()\n    print("done")']
        for i, main_block in enumerate(main_block_variations):
            mock_print_warning.reset_mock()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(f'app = MCPApp()\n\n{main_block}')
                f.flush()
                validate_entrypoint(Path(f.name))
                mock_print_warning.assert_called_once()

    @patch('mcp_agent.cli.cloud.commands.deploy.validation.print_warning')
    def test_validate_entrypoint_no_warning_without_main_block(self, mock_print_warning):
        """Test that no warning is issued when there's no __main__ block."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('app = MCPApp()')
            f.flush()
            validate_entrypoint(Path(f.name))
            mock_print_warning.assert_not_called()

    def test_validate_entrypoint_with_complex_content(self):
        """Test validation with more complex but valid Python content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('\nimport os\nfrom pathlib import Path\nfrom mcp_agent.cloud import MCPApp\n\n# Configuration\nCONFIG_PATH = Path(__file__).parent / "config.yaml"\n\ndef load_config():\n    \'\'\'Load configuration from file.\'\'\'\n    pass\n\n# Create the MCP application\napplication = MCPApp(\n    name="complex-app",\n    config_path=CONFIG_PATH,\n    debug=os.getenv("DEBUG", False)\n)\n\nclass Helper:\n    def __init__(self):\n        pass\n')
            f.flush()
            validate_entrypoint(Path(f.name))

    def test_validate_entrypoint_handles_encoding(self):
        """Test that validation handles different file encodings properly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write('# -*- coding: utf-8 -*-\n# This file contains unicode characters: test\napp = MCPApp()\n')
            f.flush()
            validate_entrypoint(Path(f.name))

    def test_validate_entrypoint_empty_file(self):
        """Test validation fails for empty files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('')
            f.flush()
            with pytest.raises(ValueError, match='No MCPApp definition found in main.py'):
                validate_entrypoint(Path(f.name))

def test_validate_entrypoint_success(valid_project_dir):
    """Test validate_entrypoint with valid MCPApp definition."""
    entrypoint_path = valid_project_dir / 'main.py'
    validate_entrypoint(entrypoint_path)

def test_validate_entrypoint_missing_file():
    """Test validate_entrypoint with non-existent file."""
    with pytest.raises(FileNotFoundError, match='Entrypoint file .* does not exist'):
        validate_entrypoint(Path('/non/existent/main.py'))

def test_validate_entrypoint_no_mcp_app():
    """Test validate_entrypoint without MCPApp definition."""
    with tempfile.TemporaryDirectory() as temp_dir:
        main_py_path = Path(temp_dir) / 'main.py'
        main_py_content = '\ndef main():\n    print("Hello, world!")\n\nif __name__ == "__main__":\n    main()\n'
        main_py_path.write_text(main_py_content)
        with pytest.raises(ValueError, match='No MCPApp definition found in main.py'):
            validate_entrypoint(main_py_path)

def test_validate_entrypoint_with_main_block_warning(capsys):
    """Test validate_entrypoint with __main__ block shows warning."""
    with tempfile.TemporaryDirectory() as temp_dir:
        main_py_path = Path(temp_dir) / 'main.py'
        main_py_content = 'from mcp_agent_cloud import MCPApp\n\napp = MCPApp(name="test-app")\n\nif __name__ == "__main__":\n    print("This will be ignored")\n'
        main_py_path.write_text(main_py_content)
        validate_entrypoint(main_py_path)
        captured = capsys.readouterr()
        assert 'Found a __main__ entrypoint in main.py. This will be ignored' in captured.err or 'Found a __main__ entrypoint in main.py. This will be ignored' in captured.out

def test_validate_entrypoint_multiline_mcp_app():
    """Test validate_entrypoint with multiline MCPApp definition."""
    with tempfile.TemporaryDirectory() as temp_dir:
        main_py_path = Path(temp_dir) / 'main.py'
        main_py_content = 'from mcp_agent_cloud import MCPApp\n\nmy_app = MCPApp(\n    name="test-app",\n    description="A test application",\n    version="1.0.0"\n)\n'
        main_py_path.write_text(main_py_content)
        validate_entrypoint(main_py_path)

def test_validate_entrypoint_different_variable_names():
    """Test validate_entrypoint with different variable names for MCPApp."""
    with tempfile.TemporaryDirectory() as temp_dir:
        main_py_path = Path(temp_dir) / 'main.py'
        for var_name in ['app', 'my_app', 'application', 'mcp_app']:
            main_py_content = f'from mcp_agent_cloud import MCPApp\n\n{var_name} = MCPApp(name="test-app")\n'
            main_py_path.write_text(main_py_content)
            validate_entrypoint(main_py_path)

