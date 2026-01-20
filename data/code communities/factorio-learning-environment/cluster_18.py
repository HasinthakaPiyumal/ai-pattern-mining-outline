# Cluster 18

def __getattr__(name):
    if name in ('Policy', 'PolicyMeta'):
        Policy, PolicyMeta = _get_policy_classes()
        globals()[name] = globals().get('Policy' if name == 'Policy' else 'PolicyMeta')
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def _get_policy_classes():
    """Lazy import for Policy classes to avoid circular imports."""
    from fle.agents.llm.parsing import Policy, PolicyMeta
    return (Policy, PolicyMeta)

