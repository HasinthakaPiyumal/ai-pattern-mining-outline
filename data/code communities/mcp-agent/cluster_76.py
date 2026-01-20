# Cluster 76

def create_agent_resource(agent: 'Agent') -> AgentResource:
    return AgentResource(type='resource', agent=agent, resource=TextResourceContents(text=f"You are now Agent '{agent.name}'. Please review the messages and continue execution", uri=AnyUrl('http://fake.url')))

def create_agent_function_result_resource(result: 'AgentFunctionResult') -> AgentFunctionResultResource:
    return AgentFunctionResultResource(type='resource', result=result, resource=TextResourceContents(text=result.value or result.agent.name or 'AgentFunctionResult', uri=AnyUrl('http://fake.url')))

class TestUtilityFunctions:
    """Tests for utility functions in the swarm module."""

    def test_create_agent_resource(self, mock_agent):
        """Test create_agent_resource function."""
        resource = create_agent_resource(mock_agent)
        assert resource.type == 'resource'
        assert resource.agent == mock_agent
        assert 'You are now Agent' in resource.resource.text
        assert mock_agent.name in resource.resource.text

    def test_create_agent_function_result_resource(self):
        """Test create_agent_function_result_resource function."""
        result = AgentFunctionResult(value='test value')
        resource = create_agent_function_result_resource(result)
        assert resource.type == 'resource'
        assert resource.result == result
        assert resource.resource.text == result.value

