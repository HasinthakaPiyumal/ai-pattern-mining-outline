# Cluster 90

@pytest.fixture
def mock_workflow_class():
    """Mock workflow class for testing"""

    class MockWorkflow:

        def __init__(self):
            self.name = None
            self.context = None
            self.run_async = AsyncMock()

        @classmethod
        async def create(cls, name=None, context=None):
            instance = cls()
            instance.name = name
            instance.context = context
            return instance
    MockWorkflow.create = AsyncMock()
    return MockWorkflow

