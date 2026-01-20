# Cluster 78

def _tool_result_to_json(tool_result: CallToolResult):
    if tool_result.content and len(tool_result.content) > 0:
        text = tool_result.content[0].text
        try:
            import json
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None

def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in (overlay or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in (overlay or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in (overlay or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

