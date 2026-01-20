# Cluster 25

def log_mcp_message(direction: str, method: str, params: Any=None, result: Any=None, error: Any=None):
    """Log MCP communication in detail"""
    message_parts = [f'MCP {direction} - Method: {method}']
    if params:
        try:
            params_str = json.dumps(params, indent=2)
            message_parts.append(f'Params: {params_str}')
        except:
            message_parts.append(f'Params: {params}')
    if result:
        try:
            result_str = json.dumps(result, indent=2)
            message_parts.append(f'Result: {result_str}')
        except:
            message_parts.append(f'Result: {result}')
    if error:
        message_parts.append(f'Error: {error}')
    logger.debug('\n'.join(message_parts))

def find_executable(cmd: str) -> Optional[str]:
    """
    Find the full path to an executable command.
    
    Args:
        cmd: The command to find
        
    Returns:
        Full path to the executable if found, None otherwise
    """
    if os.path.isfile(cmd) and os.access(cmd, os.X_OK):
        return cmd
    cmd_path = shutil.which(cmd)
    if cmd_path:
        logger.info(f'Found {cmd} in PATH at {cmd_path}')
        return cmd_path
    common_paths = ['/usr/local/bin', '/usr/bin', '/bin', '/opt/homebrew/bin', os.path.expanduser('~/.npm-global/bin'), os.path.expanduser('~/.nvm/current/bin')]
    for path in common_paths:
        full_path = os.path.join(path, cmd)
        if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
            logger.info(f'Found {cmd} at {full_path}')
            return full_path
    logger.error(f'Could not find executable: {cmd}')
    return None

