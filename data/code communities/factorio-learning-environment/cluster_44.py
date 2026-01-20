# Cluster 44

def create_default_agent_card(name: str) -> AgentCard:
    """Create a default A2A agent card describing a Factorio agent's capabilities"""
    return AgentCard(name=name, version='1.0', description='An AI agent specialized in Factorio game automation and assistance', url='https://github.com/JackHopkins/factorio-learning-environment', capabilities=AgentCapabilities(pushNotifications=False, stateTransitionHistory=False, streaming=False), defaultInputModes=['text/plain', 'application/json'], defaultOutputModes=['text/plain', 'application/json'], skills=[AgentSkill(id='factorio_automation', name='Factorio Automation', description='Automate and optimize Factorio gameplay', tags=['automation', 'optimization', 'gameplay'], examples=['Automate resource gathering', 'Optimize production lines', 'Design efficient layouts'])], provider=AgentProvider(organization='FLE team', url='https://github.com/JackHopkins/factorio-learning-environment'))

