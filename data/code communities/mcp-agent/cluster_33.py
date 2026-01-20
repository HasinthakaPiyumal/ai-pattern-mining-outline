# Cluster 33

def _is_outdated(current: str, latest: str) -> bool:
    try:
        return _parse_version(latest) > _parse_version(current)
    except Exception:
        return latest != current

def _parse_version(s: str):
    try:
        from packaging.version import parse as _vparse
        return _vparse(s)
    except Exception:
        return _simple_version_tuple(s)

def _run_version_check() -> None:
    """Worker that performs the HTTP lookup and captures the message if needed."""
    global _version_check_message
    try:
        current = _get_installed_version()
        if not current:
            return
        latest = _fetch_latest_version(timeout_seconds=5.0)
        if not latest:
            return
        if _is_outdated(current, latest):
            _version_check_message = f"A new version of mcp-agent is available: {current} -> {latest}. Update with: 'uv tool upgrade mcp-agent'"
    finally:
        _version_check_event.set()

def _get_installed_version() -> Optional[str]:
    try:
        import importlib.metadata as _im
        return _im.version('mcp-agent')
    except Exception:
        return None

def _fetch_latest_version(timeout_seconds: float=5.0) -> Optional[str]:
    try:
        import httpx
        url = 'https://pypi.org/pypi/mcp-agent/json'
        timeout = httpx.Timeout(timeout_seconds)
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                version = (data or {}).get('info', {}).get('version')
                if isinstance(version, str) and version:
                    return version
    except Exception:
        pass
    return None

