# Cluster 43

def validate_project(project_dir: Path):
    """
    Validates the project directory structure and required files.
    Raises an exception if validation fails.
    Logs warnings for non-critical issues.
    """
    if not project_dir.exists():
        raise FileNotFoundError(f'Project directory {project_dir} does not exist.')
    required_files = ['main.py']
    for file in required_files:
        if not (project_dir / file).exists():
            raise FileNotFoundError(f'Required file {file} is missing in the project directory.')
    validate_entrypoint(project_dir / 'main.py')
    has_requirements = os.path.exists(os.path.join(project_dir, 'requirements.txt'))
    has_poetry_lock = os.path.exists(os.path.join(project_dir, 'poetry.lock'))
    has_uv_lock = os.path.exists(os.path.join(project_dir, 'uv.lock'))
    if sum([has_requirements, has_poetry_lock, has_uv_lock]) > 1:
        raise ValueError('Multiple Python project dependency management files found. Expected only one of: requirements.txt, poetry.lock, uv.lock')
    has_pyproject = os.path.exists(os.path.join(project_dir, 'pyproject.toml'))
    if has_uv_lock and (not has_pyproject):
        raise ValueError('Invalid uv project: uv.lock found without corresponding pyproject.toml')
    if has_poetry_lock and (not has_pyproject):
        raise ValueError('Invalid poetry project: poetry.lock found without corresponding pyproject.toml')
    if sum([has_pyproject, has_requirements, has_poetry_lock, has_uv_lock]) == 0:
        raise ValueError('No Python project dependency management files found. Expected one of: pyproject.toml, requirements.txt, poetry.lock, uv.lock in the project directory.')

class TestValidateProject:
    """Tests for validate_project function."""

    def test_validate_project_success(self):
        """Test validation of a valid project directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            main_py = project_dir / 'main.py'
            main_py.write_text('\nfrom mcp_agent.cloud import MCPApp\n\napp = MCPApp(name="test-app")\n')
            (project_dir / 'requirements.txt').write_text('mcp-agent')
            validate_project(project_dir)

    def test_validate_project_directory_not_exists(self):
        """Test validation fails when project directory doesn't exist."""
        non_existent_dir = Path('/non/existent/directory')
        with pytest.raises(FileNotFoundError, match='Project directory .* does not exist'):
            validate_project(non_existent_dir)

    def test_validate_project_missing_main_py(self):
        """Test validation fails when main.py is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            with pytest.raises(FileNotFoundError, match='Required file main.py is missing'):
                validate_project(project_dir)

    def test_validate_project_calls_validate_entrypoint(self):
        """Test that validate_project calls validate_entrypoint for main.py."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            main_py = project_dir / 'main.py'
            main_py.write_text('app = MCPApp()')
            (project_dir / 'requirements.txt').write_text('mcp-agent')
            with patch('mcp_agent.cli.cloud.commands.deploy.validation.validate_entrypoint') as mock_validate:
                validate_project(project_dir)
                mock_validate.assert_called_once_with(main_py)

def test_validate_project_success(valid_project_dir):
    """Test validate_project with a valid project structure."""
    validate_project(valid_project_dir)

def test_validate_project_missing_directory():
    """Test validate_project with non-existent directory."""
    with pytest.raises(FileNotFoundError, match='Project directory .* does not exist'):
        validate_project(Path('/non/existent/path'))

def test_validate_project_missing_main_py():
    """Test validate_project with missing main.py."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        with pytest.raises(FileNotFoundError, match='Required file main.py is missing'):
            validate_project(project_path)

def test_validate_project_with_requirements_txt(project_with_requirements):
    """Test validate_project with requirements.txt dependency management."""
    validate_project(project_with_requirements)

def test_validate_project_with_poetry(project_with_poetry):
    """Test validate_project with poetry dependency management."""
    validate_project(project_with_poetry)

def test_validate_project_with_uv(project_with_uv):
    """Test validate_project with uv dependency management."""
    validate_project(project_with_uv)

def test_validate_project_multiple_dependency_managers():
    """Test validate_project with multiple dependency management files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        main_py_content = 'from mcp_agent_cloud import MCPApp\n\napp = MCPApp(name="test-app")\n'
        (project_path / 'main.py').write_text(main_py_content)
        (project_path / 'requirements.txt').write_text('requests==2.31.0')
        (project_path / 'poetry.lock').write_text('# Poetry lock')
        with pytest.raises(ValueError, match='Multiple Python project dependency management files found'):
            validate_project(project_path)

def test_validate_project_uv_without_pyproject():
    """Test validate_project with uv.lock but no pyproject.toml."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        main_py_content = 'from mcp_agent_cloud import MCPApp\n\napp = MCPApp(name="test-app")\n'
        (project_path / 'main.py').write_text(main_py_content)
        (project_path / 'uv.lock').write_text('# UV lock file')
        with pytest.raises(ValueError, match='Invalid uv project: uv.lock found without corresponding pyproject.toml'):
            validate_project(project_path)

def test_validate_project_poetry_without_pyproject():
    """Test validate_project with poetry.lock but no pyproject.toml."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        main_py_content = 'from mcp_agent_cloud import MCPApp\n\napp = MCPApp(name="test-app")\n'
        (project_path / 'main.py').write_text(main_py_content)
        (project_path / 'poetry.lock').write_text('# Poetry lock file')
        with pytest.raises(ValueError, match='Invalid poetry project: poetry.lock found without corresponding pyproject.toml'):
            validate_project(project_path)

def test_validate_project_no_dependency_files():
    """Test validate_project when no dependency management files exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        main_py_content = 'from mcp_agent_cloud import MCPApp\n\napp = MCPApp(name="test-app")\n'
        (project_path / 'main.py').write_text(main_py_content)
        with pytest.raises(ValueError, match='No Python project dependency management files found. Expected one of: pyproject.toml, requirements.txt, poetry.lock, uv.lock in the project directory.'):
            validate_project(project_path)

