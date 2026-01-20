# Cluster 37

def is_serializable(obj: Any) -> bool:
    """Test if an object can be serialized with pickle"""
    try:
        if obj == True or obj == False:
            return True
        if isinstance(obj, type):
            return False
        if obj.__module__ == 'builtins':
            return False
        if isinstance(obj, Enum):
            return True
        if isinstance(obj, (list, dict)):
            return all((is_serializable(item) for item in obj))
        if isinstance(obj, (int, float, str, bool, list, dict, tuple, set)):
            return True
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError, AttributeError):
        return False

