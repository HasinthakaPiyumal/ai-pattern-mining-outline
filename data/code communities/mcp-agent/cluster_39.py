# Cluster 39

def serialize_for_debug(obj):
    if isinstance(obj, (DeveloperSecret, UserSecret)):
        return f'{obj.__class__.__name__}({obj.value})'
    elif isinstance(obj, dict):
        return {k: serialize_for_debug(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_debug(item) for item in obj]
    else:
        return obj

