# Cluster 89

@pytest.fixture
def mock_config(multi_instance):
    """Create a mock configuration"""
    task = DefaultTask(trajectory_length=2, goal_description='Test task for agent messaging', task_key='test_task')
    agents = [BasicAgent(model='test-model-1', system_prompt='You are Agent 1', task=task), BasicAgent(model='test-model-2', system_prompt='You are Agent 2', task=task)]
    config = MagicMock()
    config.agents = agents
    config.version = 0
    config.version_description = 'test'
    config.task = task
    return config

