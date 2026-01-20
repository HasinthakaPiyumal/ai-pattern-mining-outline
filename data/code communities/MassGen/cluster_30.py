# Cluster 30

class TestCodeExecutionBasics:
    """Test basic command execution functionality."""

    def test_simple_python_command(self, tmp_path):
        """Test executing a simple Python command."""
        exit_code, stdout, stderr = run_command_directly(f'{sys.executable} -c "print(\\"Hello, World!\\")"', cwd=str(tmp_path))
        assert exit_code == 0
        assert 'Hello, World!' in stdout

    def test_python_script_execution(self, tmp_path):
        """Test executing a Python script."""
        script_path = tmp_path / 'test_script.py'
        script_path.write_text("print('Script executed')\nprint('Success')")
        exit_code, stdout, stderr = run_command_directly(f'{sys.executable} test_script.py', cwd=str(tmp_path))
        assert exit_code == 0
        assert 'Script executed' in stdout
        assert 'Success' in stdout

    def test_command_with_error(self, tmp_path):
        """Test that command errors are captured."""
        exit_code, stdout, stderr = run_command_directly(f'{sys.executable} -c "import sys; sys.exit(1)"', cwd=str(tmp_path))
        assert exit_code == 1

    def test_command_timeout(self, tmp_path):
        """Test that commands can timeout."""
        with pytest.raises(subprocess.TimeoutExpired):
            run_command_directly(f'{sys.executable} -c "import time; time.sleep(10)"', cwd=str(tmp_path), timeout=1)

    def test_working_directory(self, tmp_path):
        """Test that working directory is respected."""
        test_file = tmp_path / 'test.txt'
        test_file.write_text('test content')
        exit_code, stdout, stderr = run_command_directly(f'{sys.executable} -c "import os; print(os.listdir())"', cwd=str(tmp_path))
        assert exit_code == 0
        assert 'test.txt' in stdout

def run_command_directly(command: str, cwd: str=None, timeout: int=10) -> tuple:
    """Helper to run commands directly for testing."""
    result = subprocess.run(command, shell=True, cwd=cwd, timeout=timeout, capture_output=True, text=True)
    return (result.returncode, result.stdout, result.stderr)

class TestPathValidation:
    """Test path validation and security."""

    def test_path_exists_validation(self, tmp_path):
        """Test that non-existent paths are rejected."""
        non_existent = tmp_path / 'does_not_exist'
        with pytest.raises(FileNotFoundError):
            run_command_directly('echo "test"', cwd=str(non_existent))

    def test_relative_path_resolution(self, tmp_path):
        """Test that relative paths are resolved correctly."""
        subdir = tmp_path / 'subdir'
        subdir.mkdir()
        test_file = subdir / 'test.txt'
        test_file.write_text('content')
        exit_code, stdout, stderr = run_command_directly(f'''{sys.executable} -c "import os; print(os.path.exists('subdir/test.txt'))"''', cwd=str(tmp_path))
        assert exit_code == 0
        assert 'True' in stdout

class TestOutputHandling:
    """Test output capture and size limits."""

    def test_stdout_capture(self, tmp_path):
        """Test that stdout is captured correctly."""
        exit_code, stdout, stderr = run_command_directly(f'{sys.executable} -c "print(\\"line1\\"); print(\\"line2\\")"', cwd=str(tmp_path))
        assert exit_code == 0
        assert 'line1' in stdout
        assert 'line2' in stdout

    def test_stderr_capture(self, tmp_path):
        """Test that stderr is captured correctly."""
        exit_code, stdout, stderr = run_command_directly(f'{sys.executable} -c "import sys; sys.stderr.write(\\"error message\\\\n\\")"', cwd=str(tmp_path))
        assert 'error message' in stderr

    def test_large_output_handling(self, tmp_path):
        """Test handling of large output."""
        exit_code, stdout, stderr = run_command_directly(f'{sys.executable} -c "for i in range(1000): print(i)"', cwd=str(tmp_path))
        assert exit_code == 0
        assert len(stdout) > 0

class TestCrossPlatform:
    """Test cross-platform compatibility."""

    def test_python_version_check(self, tmp_path):
        """Test that Python version can be checked."""
        exit_code, stdout, stderr = run_command_directly(f'{sys.executable} --version', cwd=str(tmp_path))
        assert exit_code == 0
        assert 'Python' in stdout or 'Python' in stderr

    def test_pip_install(self, tmp_path):
        """Test that pip commands work."""
        exit_code, stdout, stderr = run_command_directly(f'{sys.executable} -m pip --version', cwd=str(tmp_path))
        assert exit_code == 0
        assert 'pip' in stdout or 'pip' in stderr

