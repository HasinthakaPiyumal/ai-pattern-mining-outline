# Cluster 9

def create_mock_agent(llm, name, description):
    """Create a simple mock agent for demonstration purposes."""
    meta = AgentMeta(name=name, description=description, tools=[])
    agent = ReActAgent(llm=llm, tools=[], memory=TokenMemory(llm), meta=meta)
    return agent

