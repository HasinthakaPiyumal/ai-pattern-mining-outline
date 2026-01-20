# Cluster 20

@pytest.yield_fixture(scope='session', autouse=True)
def mock_executor():
    with mock.patch('clai.server.agent_runner.agent_executor', MockExecutor()) as _fixture:
        yield _fixture

