# Cluster 11

def connect_platform_channel(channel: Channel, agent_graph: AgentGraph | None=None) -> AgentGraph:
    for _, agent in agent_graph.get_agents():
        agent.channel = channel
        agent.env.action.channel = channel
    return agent_graph

