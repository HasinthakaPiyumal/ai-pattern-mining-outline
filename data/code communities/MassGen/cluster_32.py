# Cluster 32

class TestDockerExecution:
    """Test Docker-based command execution."""

    @pytest.fixture(autouse=True)
    def check_docker(self):
        """Skip tests if Docker is not available."""
        try:
            import docker
            client = docker.from_env()
            client.ping()
            try:
                client.images.get('massgen/mcp-runtime:latest')
            except docker.errors.ImageNotFound:
                pytest.skip("Docker image 'massgen/mcp-runtime:latest' not found. Run: bash massgen/docker/build.sh")
        except ImportError:
            pytest.skip('Docker library not installed. Install with: pip install docker')
        except Exception as e:
            pytest.skip(f'Docker not available: {e}')

    def test_docker_manager_initialization(self):
        """Test that DockerManager can be initialized."""
        from massgen.filesystem_manager._docker_manager import DockerManager
        manager = DockerManager(image='massgen/mcp-runtime:latest', network_mode='none')
        assert manager.image == 'massgen/mcp-runtime:latest'
        assert manager.network_mode == 'none'
        assert manager.containers == {}

    def test_docker_container_creation(self, tmp_path):
        """Test creating a Docker container."""
        from massgen.filesystem_manager._docker_manager import DockerManager
        manager = DockerManager()
        workspace = tmp_path / 'workspace'
        workspace.mkdir()
        container_id = manager.create_container(agent_id='test_agent', workspace_path=workspace)
        assert container_id is not None
        assert 'test_agent' in manager.containers
        manager.cleanup('test_agent')

    def test_docker_command_execution(self, tmp_path):
        """Test executing commands in Docker container."""
        from massgen.filesystem_manager._docker_manager import DockerManager
        manager = DockerManager()
        workspace = tmp_path / 'workspace'
        workspace.mkdir()
        manager.create_container(agent_id='test_exec', workspace_path=workspace)
        result = manager.exec_command(agent_id='test_exec', command="echo 'Hello from Docker'")
        assert result['success'] is True
        assert result['exit_code'] == 0
        assert 'Hello from Docker' in result['stdout']
        manager.cleanup('test_exec')

    def test_docker_container_persistence(self, tmp_path):
        """Test that container state persists across commands."""
        from massgen.filesystem_manager._docker_manager import DockerManager
        manager = DockerManager()
        workspace = tmp_path / 'workspace'
        workspace.mkdir()
        manager.create_container(agent_id='test_persist', workspace_path=workspace)
        result1 = manager.exec_command(agent_id='test_persist', command='pip install --quiet click')
        assert result1['success'] is True
        result2 = manager.exec_command(agent_id='test_persist', command="python -c 'import click; print(click.__version__)'")
        assert result2['success'] is True
        assert len(result2['stdout'].strip()) > 0
        manager.cleanup('test_persist')

    def test_docker_workspace_mounting(self, tmp_path):
        """Test that workspace is mounted correctly (with path transparency)."""
        from massgen.filesystem_manager._docker_manager import DockerManager
        manager = DockerManager()
        workspace = tmp_path / 'workspace'
        workspace.mkdir()
        test_file = workspace / 'test.txt'
        test_file.write_text('Hello from host')
        manager.create_container(agent_id='test_mount', workspace_path=workspace)
        result = manager.exec_command(agent_id='test_mount', command=f'cat {workspace}/test.txt')
        assert result['success'] is True
        assert 'Hello from host' in result['stdout']
        result2 = manager.exec_command(agent_id='test_mount', command=f"echo 'Hello from container' > {workspace}/from_container.txt")
        assert result2['success'] is True
        from_container = workspace / 'from_container.txt'
        assert from_container.exists()
        assert 'Hello from container' in from_container.read_text()
        manager.cleanup('test_mount')

    def test_docker_container_isolation(self, tmp_path):
        """Test that containers are isolated from each other."""
        from massgen.filesystem_manager._docker_manager import DockerManager
        manager = DockerManager()
        workspace1 = tmp_path / 'workspace1'
        workspace1.mkdir()
        workspace2 = tmp_path / 'workspace2'
        workspace2.mkdir()
        manager.create_container(agent_id='agent1', workspace_path=workspace1)
        manager.create_container(agent_id='agent2', workspace_path=workspace2)
        result1 = manager.exec_command(agent_id='agent1', command=f"echo 'agent1 data' > {workspace1}/data.txt")
        assert result1['success'] is True
        result2 = manager.exec_command(agent_id='agent2', command=f'ls {workspace2}/')
        assert result2['success'] is True
        assert 'data.txt' not in result2['stdout']
        manager.cleanup('agent1')
        manager.cleanup('agent2')

    def test_docker_resource_limits(self, tmp_path):
        """Test that resource limits are applied."""
        from massgen.filesystem_manager._docker_manager import DockerManager
        manager = DockerManager(memory_limit='512m', cpu_limit=1.0)
        workspace = tmp_path / 'workspace'
        workspace.mkdir()
        container_id = manager.create_container(agent_id='test_limits', workspace_path=workspace)
        assert container_id is not None
        container = manager.get_container('test_limits')
        assert container is not None
        manager.cleanup('test_limits')

    def test_docker_network_isolation(self, tmp_path):
        """Test that network isolation works."""
        from massgen.filesystem_manager._docker_manager import DockerManager
        manager = DockerManager(network_mode='none')
        workspace = tmp_path / 'workspace'
        workspace.mkdir()
        manager.create_container(agent_id='test_network', workspace_path=workspace)
        result = manager.exec_command(agent_id='test_network', command='ping -c 1 google.com')
        assert result['success'] is False or 'Network is unreachable' in result['stdout']
        manager.cleanup('test_network')

    def test_docker_command_timeout(self, tmp_path):
        """Test that Docker commands can timeout."""
        from massgen.filesystem_manager._docker_manager import DockerManager
        manager = DockerManager()
        workspace = tmp_path / 'workspace'
        workspace.mkdir()
        manager.create_container(agent_id='test_timeout', workspace_path=workspace)
        result = manager.exec_command(agent_id='test_timeout', command='sleep 10', timeout=1)
        assert result['success'] is False
        assert result['exit_code'] == -1
        assert 'timed out' in result['stderr'].lower()
        assert result['execution_time'] >= 1.0
        manager.cleanup('test_timeout')

    def test_docker_context_path_mounting(self, tmp_path):
        """Test that context paths are mounted correctly with proper read-only enforcement."""
        from massgen.filesystem_manager._docker_manager import DockerManager
        manager = DockerManager()
        workspace = tmp_path / 'workspace'
        workspace.mkdir()
        context_dir = tmp_path / 'context'
        context_dir.mkdir()
        context_file = context_dir / 'context.txt'
        context_file.write_text('Context data')
        context_paths = [{'path': str(context_dir), 'permission': 'read', 'name': 'my_context'}]
        manager.create_container(agent_id='test_context', workspace_path=workspace, context_paths=context_paths)
        result = manager.exec_command(agent_id='test_context', command=f'cat {context_dir}/context.txt')
        assert result['success'] is True
        assert 'Context data' in result['stdout']
        result_write = manager.exec_command(agent_id='test_context', command=f"echo 'should fail' > {context_dir}/new_file.txt")
        assert result_write['success'] is False
        assert 'Read-only file system' in result_write['stdout']
        new_file = context_dir / 'new_file.txt'
        assert not new_file.exists()
        manager.cleanup('test_context')

