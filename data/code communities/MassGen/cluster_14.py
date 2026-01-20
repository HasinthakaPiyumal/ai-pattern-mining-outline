# Cluster 14

@mcp.tool()
def execute_command(command: str, timeout: Optional[int]=None, work_dir: Optional[str]=None) -> Dict[str, Any]:
    """
        Execute a command line command.

        This tool allows executing any command line program including:
        - Python: execute_command("python script.py")
        - Node.js: execute_command("node app.js")
        - Tests: execute_command("pytest tests/")
        - Build tools: execute_command("npm run build")
        - Shell commands: execute_command("ls -la")

        The command is executed in a shell environment, so you can use shell features
        like pipes, redirection, and environment variables. On Windows, this uses
        cmd.exe; on Unix/Mac, this uses the default shell (typically bash).

        Args:
            command: The command to execute (required)
            timeout: Maximum execution time in seconds (default: 60)
                    Set to None for no timeout (use with caution)
            work_dir: Working directory for execution (relative to workspace)
                     If not specified, uses the current workspace directory

        Returns:
            Dictionary containing:
            - success: bool - True if exit code was 0
            - exit_code: int - Process exit code
            - stdout: str - Standard output from the command
            - stderr: str - Standard error from the command
            - execution_time: float - Time taken to execute in seconds
            - command: str - The command that was executed
            - work_dir: str - The working directory used

        Security:
            - Execution is confined to allowed paths
            - Timeout enforced to prevent infinite loops
            - Output size limited to prevent memory exhaustion
            - Basic sanitization against dangerous commands

        Examples:
            # Run Python script
            execute_command("python test.py")

            # Run tests with pytest
            execute_command("pytest tests/ -v")

            # Install package and run script
            execute_command("pip install requests && python scraper.py")

            # Check Python version
            execute_command("python --version")

            # List files
            execute_command("ls -la")  # Unix/Mac
            execute_command("dir")      # Windows
        """
    try:
        try:
            _sanitize_command(command)
        except ValueError as e:
            return {'success': False, 'exit_code': -1, 'stdout': '', 'stderr': str(e), 'execution_time': 0.0, 'command': command, 'work_dir': work_dir or str(Path.cwd())}
        try:
            _check_command_filters(command, mcp.allowed_commands, mcp.blocked_commands)
        except ValueError as e:
            return {'success': False, 'exit_code': -1, 'stdout': '', 'stderr': str(e), 'execution_time': 0.0, 'command': command, 'work_dir': work_dir or str(Path.cwd())}
        if timeout is None:
            timeout = mcp.default_timeout
        if work_dir:
            if Path(work_dir).is_absolute():
                work_path = Path(work_dir).resolve()
            else:
                work_path = (Path.cwd() / work_dir).resolve()
        else:
            work_path = Path.cwd()
        _validate_path_access(work_path, mcp.allowed_paths)
        if not work_path.exists():
            return {'success': False, 'exit_code': -1, 'stdout': '', 'stderr': f'Working directory does not exist: {work_path}', 'execution_time': 0.0, 'command': command, 'work_dir': str(work_path)}
        if not work_path.is_dir():
            return {'success': False, 'exit_code': -1, 'stdout': '', 'stderr': f'Working directory is not a directory: {work_path}', 'execution_time': 0.0, 'command': command, 'work_dir': str(work_path)}
        if mcp.execution_mode == 'docker':
            if not mcp.docker_client:
                return {'success': False, 'exit_code': -1, 'stdout': '', 'stderr': 'Docker mode enabled but docker_client not initialized', 'execution_time': 0.0, 'command': command, 'work_dir': str(work_path)}
            if not mcp.agent_id:
                return {'success': False, 'exit_code': -1, 'stdout': '', 'stderr': 'Docker mode requires agent_id to be set. This should be configured by the orchestrator.', 'execution_time': 0.0, 'command': command, 'work_dir': str(work_path)}
            try:
                container_name = f'massgen-{mcp.agent_id}'
                container = mcp.docker_client.containers.get(container_name)
                exec_config = {'cmd': ['/bin/sh', '-c', command], 'workdir': str(work_path), 'stdout': True, 'stderr': True}
                start_time = time.time()
                exit_code, output = container.exec_run(**exec_config)
                execution_time = time.time() - start_time
                output_str = output.decode('utf-8') if isinstance(output, bytes) else output
                if len(output_str) > mcp.max_output_size:
                    output_str = output_str[:mcp.max_output_size] + f'\n... (truncated, exceeded {mcp.max_output_size} bytes)'
                return {'success': exit_code == 0, 'exit_code': exit_code, 'stdout': output_str, 'stderr': '', 'execution_time': execution_time, 'command': command, 'work_dir': str(work_path)}
            except DockerException as e:
                return {'success': False, 'exit_code': -1, 'stdout': '', 'stderr': f'Docker container error: {str(e)}', 'execution_time': 0.0, 'command': command, 'work_dir': str(work_path)}
            except Exception as e:
                return {'success': False, 'exit_code': -1, 'stdout': '', 'stderr': f'Docker execution error: {str(e)}', 'execution_time': 0.0, 'command': command, 'work_dir': str(work_path)}
        else:
            env = _prepare_environment(work_path)
            start_time = time.time()
            try:
                result = subprocess.run(command, shell=True, cwd=str(work_path), timeout=timeout, capture_output=True, text=True, env=env)
                execution_time = time.time() - start_time
                stdout = result.stdout
                stderr = result.stderr
                if len(stdout) > mcp.max_output_size:
                    stdout = stdout[:mcp.max_output_size] + f'\n... (truncated, exceeded {mcp.max_output_size} bytes)'
                if len(stderr) > mcp.max_output_size:
                    stderr = stderr[:mcp.max_output_size] + f'\n... (truncated, exceeded {mcp.max_output_size} bytes)'
                return {'success': result.returncode == 0, 'exit_code': result.returncode, 'stdout': stdout, 'stderr': stderr, 'execution_time': execution_time, 'command': command, 'work_dir': str(work_path)}
            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time
                return {'success': False, 'exit_code': -1, 'stdout': '', 'stderr': f'Command timed out after {timeout} seconds', 'execution_time': execution_time, 'command': command, 'work_dir': str(work_path)}
            except Exception as e:
                execution_time = time.time() - start_time
                return {'success': False, 'exit_code': -1, 'stdout': '', 'stderr': f'Execution error: {str(e)}', 'execution_time': execution_time, 'command': command, 'work_dir': str(work_path)}
    except ValueError as e:
        return {'success': False, 'exit_code': -1, 'stdout': '', 'stderr': f'Path validation error: {str(e)}', 'execution_time': 0.0, 'command': command, 'work_dir': work_dir or str(Path.cwd())}
    except Exception as e:
        return {'success': False, 'exit_code': -1, 'stdout': '', 'stderr': f'Unexpected error: {str(e)}', 'execution_time': 0.0, 'command': command, 'work_dir': work_dir or str(Path.cwd())}

def _sanitize_command(command: str) -> None:
    """
    Sanitize the command to prevent dangerous operations.

    Adapted from AG2's LocalCommandLineCodeExecutor.sanitize_command().
    This provides basic protection for users running commands outside Docker.

    Args:
        command: The command to sanitize

    Raises:
        ValueError: If dangerous command is detected
    """
    dangerous_patterns = [('\\brm\\s+-rf\\s+/', "Use of 'rm -rf /' is not allowed"), ('\\bmv\\b.*?\\s+/dev/null', 'Moving files to /dev/null is not allowed'), ('\\bdd\\b', "Use of 'dd' command is not allowed"), ('>\\s*/dev/sd[a-z][1-9]?', 'Overwriting disk blocks directly is not allowed'), (':\\(\\)\\{\\s*:\\|\\:&\\s*\\};:', 'Fork bombs are not allowed'), ('\\bsudo\\b', "Use of 'sudo' is not allowed"), ('\\bsu\\b', "Use of 'su' is not allowed"), ('\\bchown\\b', "Use of 'chown' is not allowed"), ('\\bchmod\\b', "Use of 'chmod' is not allowed")]
    for pattern, message in dangerous_patterns:
        if re.search(pattern, command):
            raise ValueError(f'Potentially dangerous command detected: {message}')

def _check_command_filters(command: str, allowed_patterns: Optional[List[str]], blocked_patterns: Optional[List[str]]) -> None:
    """
    Check command against whitelist/blacklist filters.

    Args:
        command: The command to check
        allowed_patterns: Whitelist regex patterns (if provided, command MUST match one)
        blocked_patterns: Blacklist regex patterns (command must NOT match any)

    Raises:
        ValueError: If command doesn't match whitelist or matches blacklist
    """
    if allowed_patterns:
        if not any((re.match(pattern, command) for pattern in allowed_patterns)):
            raise ValueError(f'Command not in allowed list. Allowed patterns: {', '.join(allowed_patterns)}')
    if blocked_patterns:
        for pattern in blocked_patterns:
            if re.match(pattern, command):
                raise ValueError(f"Command matches blocked pattern: '{pattern}'")

def _prepare_environment(work_dir: Path) -> Dict[str, str]:
    """
    Prepare environment by auto-detecting .venv in work_dir.

    This function checks for a .venv directory in the working directory and
    automatically modifies PATH to use it if found. Each workspace manages
    its own virtual environment independently.

    Args:
        work_dir: Working directory to check for .venv

    Returns:
        Environment variables dict with PATH modified if .venv exists
    """
    env = os.environ.copy()
    venv_dir = work_dir / '.venv'
    if venv_dir.exists():
        venv_bin = venv_dir / ('Scripts' if WIN32 else 'bin')
        if venv_bin.exists():
            env['PATH'] = f'{venv_bin}{os.pathsep}{env['PATH']}'
            env['VIRTUAL_ENV'] = str(venv_dir)
    return env

class TestCommandSanitization:
    """Test command sanitization patterns."""

    def test_dangerous_command_patterns(self):
        """Test that dangerous patterns are identified."""
        from massgen.filesystem_manager._code_execution_server import _sanitize_command
        dangerous_commands = ['rm -rf /', 'dd if=/dev/zero of=/dev/sda', ':(){ :|:& };:', 'mv file /dev/null', 'sudo apt install something', 'su root', 'chown root file.txt', 'chmod 777 file.txt']
        for cmd in dangerous_commands:
            with pytest.raises(ValueError, match='dangerous|not allowed'):
                _sanitize_command(cmd)

    def test_safe_commands_pass(self):
        """Test that safe commands pass sanitization."""
        from massgen.filesystem_manager._code_execution_server import _sanitize_command
        safe_commands = ['python script.py', 'pytest tests/', 'npm run build', 'ls -la', 'rm file.txt', 'git submodule update', "echo 'summary'", 'python -m pip install --user requests']
        for cmd in safe_commands:
            _sanitize_command(cmd)

class TestVirtualEnvironment:
    """Test virtual environment handling."""

    def test_auto_detect_venv(self, tmp_path):
        """Test auto-detection of .venv directory."""
        from massgen.filesystem_manager._code_execution_server import _prepare_environment
        venv_dir = tmp_path / '.venv'
        venv_bin = venv_dir / 'bin'
        venv_bin.mkdir(parents=True, exist_ok=True)
        env = _prepare_environment(tmp_path)
        assert 'PATH' in env
        assert str(venv_bin) in env['PATH']
        assert 'VIRTUAL_ENV' in env
        assert str(venv_dir) in env['VIRTUAL_ENV']

    def test_no_venv_fallback(self, tmp_path):
        """Test fallback to system environment when no venv."""
        import os
        from massgen.filesystem_manager._code_execution_server import _prepare_environment
        env = _prepare_environment(tmp_path)
        assert env['PATH'] == os.environ['PATH']

