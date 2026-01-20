# Cluster 11

def prepare_command(command: str, max_length: int=MAX_COMMAND_LENGTH, *, security_level: str='strict', allowed_executables: Optional[Set[str]]=None) -> List[str]:
    """
    Sanitize a command and split it into parts before using it to run an MCP server.

    Returns:
        List of command parts

    Raises:
        ValueError: If command contains dangerous characters or uses disallowed executables
    """
    if not command or not command.strip():
        raise ValueError('MCP command cannot be empty')
    if len(command) > max_length:
        raise ValueError(f'MCP command too long: {len(command)} > {max_length} characters')
    dangerous_chars = ['&', '|', ';', '`', '$', '(', ')', '<', '>']
    for char in dangerous_chars:
        if char in command:
            raise ValueError(f'MCP command cannot contain shell metacharacters: {char}')
    dangerous_patterns = ['\\$\\{.*\\}', '\\$\\(.*\\)', '`.*`', '\\.\\./', '\\\\\\.\\\\']
    for pattern in dangerous_patterns:
        if re.search(pattern, command):
            raise ValueError(f'MCP command contains dangerous pattern: {pattern}')
    try:
        parts = shlex.split(command)
    except ValueError as e:
        raise ValueError(f'Invalid command syntax: {e}')
    if not parts:
        raise ValueError('MCP command cannot be empty after parsing')
    if len(parts) > MAX_ARGS_COUNT:
        raise ValueError(f'Too many command arguments: {len(parts)} > {MAX_ARGS_COUNT}')
    for i, part in enumerate(parts):
        if len(part) > MAX_ARG_LENGTH:
            raise ValueError(f'Command argument {i} too long: {len(part)} > {MAX_ARG_LENGTH} characters')
    normalized_level = _normalize_security_level(security_level)
    allowed = {name.lower() for name in allowed_executables or _get_default_allowed_executables(normalized_level)}
    executable_path = Path(parts[0])
    if any((part == '..' for part in executable_path.parts)):
        raise ValueError("MCP command path cannot contain parent directory components ('..')")
    base_name = executable_path.name
    lower_name = base_name.lower()
    for ext in ('.exe', '.bat', '.cmd', '.ps1'):
        if lower_name.endswith(ext):
            base_name = base_name[:-len(ext)]
            lower_name = lower_name[:-len(ext)]
            break
    if lower_name not in allowed:
        raise ValueError(f"MCP command executable '{base_name}' is not allowed (level={security_level}). Allowed executables: {sorted(allowed)}")
    return parts

def _normalize_security_level(level: str) -> str:
    """
    Normalize security level to a valid value.

    Args:
        level: Security level string

    Returns:
        Normalized security level, defaults to "strict" for unknown values
    """
    return level if level in {'strict', 'moderate', 'permissive'} else 'strict'

def _get_default_allowed_executables(level: str) -> Set[str]:
    """Get default allowed executables based on security level.

    Args:
        level: Security level string

    Returns:
        Set of allowed executable names (lowercase)
    """
    base_strict: Set[str] = {'python', 'python3', 'python3.8', 'python3.9', 'python3.10', 'python3.11', 'python3.12', 'python3.13', 'python3.14', 'py', 'uv', 'uvx', 'pipx', 'pip', 'pip3', 'node', 'npm', 'npx', 'yarn', 'pnpm', 'bun', 'deno', 'java', 'ruby', 'go', 'rust', 'cargo', 'fastmcp', 'sh', 'bash', 'zsh', 'fish', 'powershell', 'pwsh', 'cmd'}
    if level == 'strict':
        return base_strict
    if level == 'moderate':
        return base_strict | {'git', 'nodejs'}
    if level == 'permissive':
        return base_strict | {'git', 'curl', 'wget', 'nodejs'}
    return base_strict

def validate_url(url: str, *, resolve_dns: bool=False, allow_private_ips: bool=False, allow_localhost: bool=False, allowed_hostnames: Optional[Set[str]]=None) -> bool:
    """
    Validate URL for security and correctness.

    Args:
        url: URL to validate
        resolve_dns: If True, resolve hostnames and validate the resulting IPs
        allow_private_ips: If True, do not block private/link-local/reserved ranges
        allow_localhost: If True, allow localhost/loopback addresses
        allowed_hostnames: Optional explicit allowlist for hostnames

    Returns:
        True if URL is valid and safe

    Raises:
        ValueError: If URL is invalid or potentially dangerous
    """
    if not url or not isinstance(url, str):
        raise ValueError('URL must be a non-empty string')
    if len(url) > MAX_URL_LENGTH:
        raise ValueError(f'URL too long: {len(url)} > {MAX_URL_LENGTH} characters')
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as e:
        raise ValueError(f'Invalid URL format: {e}')
    if parsed.scheme not in ('http', 'https'):
        raise ValueError(f'Unsupported URL scheme: {parsed.scheme}. Only http and https are allowed.')
    if not parsed.hostname:
        raise ValueError('URL must include a hostname')
    hostname = parsed.hostname.lower()
    if allowed_hostnames and hostname in {h.lower() for h in allowed_hostnames}:
        pass
    else:
        if not allow_localhost and hostname in {'localhost', 'ip6-localhost'}:
            raise ValueError(f'Hostname not allowed for security reasons: {hostname}')
        ip_obj: Optional[Union[ipaddress.IPv4Address, ipaddress.IPv6Address]]
        try:
            ip_obj = ipaddress.ip_address(hostname)
        except ValueError:
            ip_obj = None

        def _is_forbidden_ip(ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]) -> bool:
            if allow_private_ips:
                return False
            return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified
        if ip_obj is not None:
            if _is_forbidden_ip(ip_obj) and (not (allow_localhost and ip_obj.is_loopback)):
                raise ValueError(f'IP address not allowed for security reasons: {hostname}')
        elif resolve_dns:
            try:
                port_for_resolution = parsed.port if parsed.port is not None else 443 if parsed.scheme == 'https' else 80
                addrinfos = socket.getaddrinfo(hostname, port_for_resolution, proto=socket.IPPROTO_TCP)
                for ai in addrinfos:
                    sockaddr = ai[4]
                    ip_literal = sockaddr[0]
                    try:
                        resolved_ip = ipaddress.ip_address(ip_literal)
                        if _is_forbidden_ip(resolved_ip) and (not (allow_localhost and resolved_ip.is_loopback)):
                            raise ValueError(f'Resolved IP not allowed for security reasons: {hostname} -> {resolved_ip}')
                    except ValueError:
                        continue
            except socket.gaierror as e:
                raise ValueError(f"Failed to resolve hostname '{hostname}': {e}")
    if parsed.port is not None:
        if not 1 <= parsed.port <= 65535:
            raise ValueError(f'Invalid port number: {parsed.port}')
        dangerous_ports = {22, 23, 25, 53, 135, 139, 445, 1433, 1521, 3306, 3389, 5432, 6379}
        if parsed.port in dangerous_ports:
            raise ValueError(f'Port {parsed.port} is not allowed for security reasons')
    return True

def _is_forbidden_ip(ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]) -> bool:
    if allow_private_ips:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified

def validate_environment_variables(env: Dict[str, str], *, level: str='strict', mode: str='denylist', allowed_vars: Optional[Set[str]]=None, denied_vars: Optional[Set[str]]=None, max_key_length: int=MAX_ENV_KEY_LENGTH, max_value_length: int=MAX_ENV_VALUE_LENGTH) -> Dict[str, str]:
    """
    Validate environment variables for security.

    Args:
        env: Environment variables dictionary
        level: Security level {"strict", "moderate", "permissive"}
        mode: Validation mode {"denylist", "allowlist"}
        allowed_vars: Optional explicit allowlist (case-insensitive) when mode is allowlist
        denied_vars: Optional explicit denylist (case-insensitive) when mode is denylist
        max_key_length: Maximum allowed environment variable name length
        max_value_length: Maximum allowed environment variable value length

    Returns:
        Validated environment variables

    Raises:
        ValueError: If environment variables contain dangerous values
    """
    if not isinstance(env, dict):
        raise ValueError('Environment variables must be a dictionary')
    validated_env: Dict[str, str] = {}
    normalized_level = _normalize_security_level(level)
    default_deny: Set[str] = {'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'PYTHONPATH', 'PWD', 'OLDPWD'}
    if normalized_level == 'strict':
        default_deny |= {'PATH', 'HOME', 'USER', 'USERNAME', 'SHELL'}
    elif normalized_level == 'moderate':
        default_deny |= set()
    elif normalized_level == 'permissive':
        default_deny |= set()
    denylist_active = {v.upper() for v in (denied_vars if denied_vars is not None else default_deny)}
    allowlist_active = {v.upper() for v in allowed_vars or set()}
    for key, value in env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f'Environment variable key and value must be strings: {key}={value}')
        if len(key) > max_key_length:
            raise ValueError(f'Environment variable name too long: {len(key)} > {max_key_length}')
        if len(value) > max_value_length:
            raise ValueError(f'Environment variable value too long: {len(value)} > {max_value_length}')
        upper_key = key.upper()
        if mode == 'allowlist':
            if allowlist_active and upper_key not in allowlist_active:
                raise ValueError(f"Environment variable '{key}' is not permitted by allowlist policy")
        elif upper_key in denylist_active:
            raise ValueError(f"Environment variable '{key}' is not allowed for security reasons")
        dangerous_patterns = ['$(', '`', '&', ';', '|']
        for pattern in dangerous_patterns:
            if pattern in value:
                raise ValueError(f"Environment variable '{key}' contains dangerous pattern: {pattern}")
        if '${' in value:
            if not re.match('^[^$]*\\$\\{[A-Z_][A-Z0-9_]*\\}[^$]*$', value):
                raise ValueError(f"Environment variable '{key}' contains dangerous pattern: ${{")
        validated_env[key] = value
    return validated_env

def validate_server_security(config: dict) -> dict:
    """
    Validate and sanitize MCP server configuration with comprehensive security checks.

    Args:
        config: Server configuration dictionary

    Returns:
        Validated configuration dictionary

    Raises:
        ValueError: If configuration is invalid or insecure
    """
    if not isinstance(config, dict):
        raise ValueError('Server configuration must be a dictionary')
    validated_config = config.copy()
    if 'name' not in validated_config:
        raise ValueError("Server configuration must include 'name'")
    server_name = validated_config['name']
    _validate_non_empty_string(server_name, 'Server name')
    _validate_string_length(server_name, MAX_SERVER_NAME_LENGTH, 'Server name')
    if not re.match('^[a-zA-Z0-9_-]+$', server_name):
        raise ValueError('Server name can only contain alphanumeric characters, underscores, and hyphens')
    transport_type = validated_config.get('type', 'stdio')
    security_cfg = _get_dict_from_config(validated_config, 'security')
    security_level = security_cfg.get('level', 'strict')
    if transport_type == 'stdio':
        if 'command' not in validated_config and 'args' not in validated_config:
            raise ValueError("Stdio server configuration must include 'command' or 'args'")
        if 'command' in validated_config:
            if isinstance(validated_config['command'], str):
                validated_config['command'] = prepare_command(validated_config['command'], security_level=security_level, allowed_executables=_get_set_from_config(security_cfg, 'allowed_executables'))
            elif isinstance(validated_config['command'], list):
                if not validated_config['command']:
                    raise ValueError('Command list cannot be empty')
                command_str = ' '.join((shlex.quote(arg) for arg in validated_config['command']))
                validated_config['command'] = prepare_command(command_str, security_level=security_level, allowed_executables=_get_set_from_config(security_cfg, 'allowed_executables'))
            else:
                raise ValueError('Command must be a string or list')
        if 'args' in validated_config:
            args = validated_config['args']
            if not isinstance(args, list):
                raise ValueError('Arguments must be a list')
            for i, arg in enumerate(args):
                if not isinstance(arg, str):
                    raise ValueError(f'Argument {i} must be a string')
                if len(arg) > MAX_ARG_LENGTH:
                    raise ValueError(f'Argument {i} too long: {len(arg)} > {MAX_ARG_LENGTH} characters')
        if 'env' in validated_config:
            env_policy = _get_dict_from_config(security_cfg, 'env')
            validated_config['env'] = validate_environment_variables(validated_config['env'], level=env_policy.get('level', security_level), mode=env_policy.get('mode', 'denylist'), allowed_vars=_get_set_from_config(env_policy, 'allowed_vars') or set(), denied_vars=_get_set_from_config(env_policy, 'denied_vars'))
        if 'cwd' in validated_config:
            cwd = validated_config['cwd']
            if not isinstance(cwd, str):
                raise ValueError('Working directory must be a string')
            _validate_string_length(cwd, MAX_CWD_LENGTH, 'Working directory path')
            cwd_path = Path(cwd)
            if any((part == '..' for part in cwd_path.parts)):
                raise ValueError("Working directory cannot contain parent directory components ('..')")
    elif transport_type == 'streamable-http':
        if 'url' not in validated_config:
            raise ValueError(f"{transport_type} server configuration must include 'url'")
        allowed_hostnames_cfg = security_cfg.get('allowed_hostnames')
        allowed_hostnames = None
        if isinstance(allowed_hostnames_cfg, (list, set, tuple)):
            allowed_hostnames = {str(h) for h in allowed_hostnames_cfg if isinstance(h, (str, bytes))}
        validate_url(validated_config['url'], resolve_dns=bool(security_cfg.get('resolve_dns', False)), allow_private_ips=bool(security_cfg.get('allow_private_ips', False)), allow_localhost=bool(security_cfg.get('allow_localhost', False)), allowed_hostnames=allowed_hostnames)
        if 'headers' in validated_config:
            headers = validated_config['headers']
            if not isinstance(headers, dict):
                raise ValueError('Headers must be a dictionary')
            for key, value in headers.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError('Header keys and values must be strings')
                _validate_string_length(key, MAX_HEADER_KEY_LENGTH, 'Header name')
                _validate_string_length(value, MAX_HEADER_VALUE_LENGTH, 'Header value')
        if 'timeout' in validated_config:
            timeout = validated_config['timeout']
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ValueError('Timeout must be a positive number')
            if timeout > MAX_TIMEOUT_SECONDS:
                raise ValueError(f'Timeout too large: {timeout} > {MAX_TIMEOUT_SECONDS} seconds')
        if 'http_read_timeout' in validated_config:
            http_read_timeout = validated_config['http_read_timeout']
            if not isinstance(http_read_timeout, (int, float)) or http_read_timeout <= 0:
                raise ValueError('http_read_timeout must be a positive number')
            if http_read_timeout > MAX_TIMEOUT_SECONDS:
                raise ValueError(f'http_read_timeout too large: {http_read_timeout} > {MAX_TIMEOUT_SECONDS} seconds')
    else:
        supported_types = ['stdio', 'streamable-http']
        raise ValueError(f"Unsupported transport type: {transport_type}. Supported types: {supported_types}. Note: 'sse' transport was deprecated in MCP v2025-03-26, use 'streamable-http' instead.")
    return validated_config

def _validate_non_empty_string(value: Any, field_name: str) -> None:
    """Validate that value is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field_name} must be a non-empty string')

def _validate_string_length(value: str, max_length: int, field_name: str) -> None:
    """Validate string length."""
    if len(value) > max_length:
        raise ValueError(f'{field_name} too long: {len(value)} > {max_length} characters')

def _get_dict_from_config(config: dict, key: str, default: Optional[dict]=None) -> dict:
    """Safely extract dict from config with type checking."""
    value = config.get(key, default or {})
    return value if isinstance(value, dict) else {}

def _get_set_from_config(config: dict, key: str, default: Optional[List]=None) -> Optional[Set[str]]:
    """Extract a set from config, handling empty lists and None."""
    value = config.get(key, default or [])
    if not value:
        return None
    return set(value) if isinstance(value, (list, set, tuple)) else None

def sanitize_tool_name(tool_name: str, server_name: str) -> str:
    """
    Create a sanitized tool name with server prefix and comprehensive validation.

    Args:
        tool_name: Original tool name
        server_name: Server name for prefixing

    Returns:
        Sanitized tool name with prefix

    Raises:
        ValueError: If tool name or server name is invalid
    """
    _validate_non_empty_string(tool_name, 'Tool name')
    _validate_non_empty_string(server_name, 'Server name')
    _validate_string_length(tool_name, MAX_TOOL_NAME_LENGTH, 'Tool name')
    _validate_string_length(server_name, MAX_SERVER_NAME_FOR_TOOL_LENGTH, 'Server name')
    if tool_name.startswith('mcp__'):
        tool_name = tool_name[5:]
        if '__' in tool_name:
            parts = tool_name.split('__', 1)
            if len(parts) == 2:
                tool_name = parts[1]
    reserved_names = {'connect', 'disconnect', 'list', 'help', 'version', 'status', 'health', 'ping', 'debug', 'admin', 'system', 'config', 'settings', 'auth', 'login', 'logout', 'exit', 'quit'}
    if tool_name.lower() in reserved_names:
        raise ValueError(f"Tool name '{tool_name}' is reserved and cannot be used")
    if not re.match('^[a-zA-Z0-9_.-]+$', tool_name):
        raise ValueError(f"Tool name '{tool_name}' contains invalid characters. Only alphanumeric, underscore, hyphen, and dot are allowed.")
    if not re.match('^[a-zA-Z0-9_-]+$', server_name):
        raise ValueError(f"Server name '{server_name}' contains invalid characters. Only alphanumeric, underscore, and hyphen are allowed.")
    safe_server_name = server_name.strip('_-')
    safe_tool_name = tool_name.strip('_.-')
    if not safe_server_name:
        raise ValueError(f"Server name '{server_name}' becomes empty after sanitization")
    if not safe_tool_name:
        raise ValueError(f"Tool name '{tool_name}' becomes empty after sanitization")
    final_name = f'mcp__{safe_server_name}__{safe_tool_name}'
    _validate_string_length(final_name, MAX_FINAL_TOOL_NAME_LENGTH, 'Final tool name')
    return final_name

