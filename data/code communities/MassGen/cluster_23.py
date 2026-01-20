# Cluster 23

def create_simple_agent(backend: LLMBackend, system_message: str=None, agent_id: str=None) -> SingleAgent:
    """Create a simple single agent."""
    if system_message is None:
        from .message_templates import MessageTemplates
        templates = MessageTemplates()
        system_message = templates.evaluation_system_message()
    return SingleAgent(backend=backend, agent_id=agent_id, system_message=system_message)

