# Cluster 4

def _remove_title_recursively(data, parent_key=None):
    """Recursively removes the 'title' key from all levels of a nested
    dictionary, except when 'title' is an argument name in the schema.
    """
    if isinstance(data, dict):
        if parent_key not in ['properties', '$defs', 'items', 'allOf', 'oneOf', 'anyOf']:
            data.pop('title', None)
        for key, value in data.items():
            _remove_title_recursively(value, parent_key=key)
    elif isinstance(data, list):
        for item in data:
            _remove_title_recursively(item, parent_key=parent_key)

