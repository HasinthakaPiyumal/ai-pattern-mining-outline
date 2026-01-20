# Cluster 98

def _create_test_secrets_file(file_path, processed_secrets):
    """Helper to create a test secrets file with proper structure."""
    nested_secrets = nest_keys(processed_secrets)
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(nested_secrets, f, default_flow_style=False, sort_keys=False)
    return processed_secrets

def nest_keys(flat_dict: dict[str, str]) -> dict:
    """Convert flat dict with dot-notation keys to nested dict."""
    nested: Dict[str, Any] = {}
    for flat_key, value in flat_dict.items():
        parts = flat_key.split('.')
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return nested

