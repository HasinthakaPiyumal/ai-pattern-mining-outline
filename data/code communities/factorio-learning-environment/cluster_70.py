# Cluster 70

@mcp.prompt(name='tutorial', description='Comprehensive guide to using the Factorio Learning Environment')
def tutorial() -> str:
    """Load and return the tutorial content."""
    tutorial_content = load_tutorial_md()
    return 'You are an expert at at the Factorio Learning Environment, ready to write Python code (with the API) and introspect the existing implementations, to build factories in the game.\n\n' + tutorial_content

def load_tutorial_md():
    try:
        with open(TUTORIAL_MD_PATH, 'r') as f:
            return f.read()
    except Exception as e:
        print(f'Error loading agent.md: {e}')
        return 'Error loading tutorial content. Please check if agent.md exists.'

