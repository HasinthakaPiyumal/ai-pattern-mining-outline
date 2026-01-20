# Cluster 24

class MockCLIBackend(CLIBackend):
    """Mock CLI backend for testing purposes."""

    def __init__(self, cli_command: str, mock_output: str='Mock response', **kwargs):
        self.mock_output = mock_output
        self.cli_command = cli_command
        self.working_dir = kwargs.get('working_dir', Path.cwd())
        self.timeout = kwargs.get('timeout', 300)
        self.config = kwargs
        from massgen.backend.base import TokenUsage
        self.token_usage = TokenUsage()

    def _build_command(self, messages, tools, **kwargs):
        return ['echo', 'mock command']

    def _parse_output(self, output):
        return {'content': self.mock_output, 'tool_calls': [], 'raw_response': output}

    async def _execute_cli_command(self, command):
        """Mock command execution."""
        await asyncio.sleep(0.1)
        return self.mock_output

    def get_cost_per_token(self):
        """Mock cost per token."""
        return {'input': 0.001, 'output': 0.002}

