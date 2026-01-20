# Cluster 10

def build_case1_conversation(task: str) -> Dict[str, Any]:
    """Build Case 1 conversation (no summaries exist)."""
    return get_templates().build_initial_conversation(task)

def get_templates() -> MessageTemplates:
    """Get global message templates instance."""
    return _templates

def build_case2_conversation(task: str, agent_summaries: Dict[str, str], valid_agent_ids: Optional[List[str]]=None) -> Dict[str, Any]:
    """Build Case 2 conversation (summaries exist)."""
    return get_templates().build_initial_conversation(task, agent_summaries, valid_agent_ids)

def get_standard_tools(valid_agent_ids: Optional[List[str]]=None) -> List[Dict[str, Any]]:
    """Get standard MassGen tools."""
    return get_templates().get_standard_tools(valid_agent_ids)

def get_enforcement_message() -> str:
    """Get enforcement message for Case 3."""
    return get_templates().enforcement_message()

