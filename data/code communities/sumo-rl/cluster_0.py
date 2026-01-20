# Cluster 0

def get_version():
    """Gets the mo-gymnasium version."""
    path = CWD / 'sumo_rl' / '__init__.py'
    content = path.read_text()
    for line in content.splitlines():
        if line.startswith('__version__'):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError('bad version data in __init__.py')

