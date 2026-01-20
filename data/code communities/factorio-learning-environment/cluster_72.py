# Cluster 72

def log_info(message):
    """Log a message to stderr to avoid MCP protocol corruption"""
    print(message, file=sys.stderr)

