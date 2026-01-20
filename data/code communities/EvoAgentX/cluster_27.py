# Cluster 27

def safe_deepcopy(obj):
    """
    Safely attempt to deep copy any Python object, with graceful fallback behavior.

    This function performs a standard `copy.deepcopy` when possible. If that fails
    (e.g., due to the presence of uncopyable components such as file handles, threads,
    or custom classes that don't support deep copying), it falls back to a more resilient strategy:

    1. Attempts to create a blank instance of the object's class using `__new__`.
    2. Recursively copies all attributes found in the object's `__dict__`, using:
    - `safe_deepcopy` for deep recursive copy,
    - `copy.copy` as a shallow fallback,
    - or the original reference as a last resort.
    3. If the object has no `__dict__` or cannot be instantiated, returns the original object.

    Parameters:
        obj (Any): The object to be deep copied.

    Returns:
        Any: A deep copy of the input object if possible, or a best-effort fallback copy.
    
    Warnings:
        Issues a `warnings.warn()` message whenever:
        - The deep copy fails and fallback mechanisms are used.
        - An attribute copy fails and falls back to a shallower or direct reference.
        - The class cannot be re-instantiated and the original reference is returned.

    Notes:
        - This function is intended for robust copying in systems where user-defined objects,
        templates, or agents may not support strict deep copying.
        - It is not guaranteed to preserve identity semantics or copy objects with `__slots__`.
        - For critical correctness or mutation isolation, ensure your objects are deepcopy-compatible.

    Example:
        >>> obj = CustomObject()
        >>> obj_copy = safe_deepcopy(obj)
    """
    try:
        return copy.deepcopy(obj)
    except Exception:
        warnings.warn(f'Failed to deepcopy {obj.__class__.__name__}. Falling back to advanced handling.')
        pass
    try:
        new_instance = obj.__class__.__new__(obj.__class__)
    except Exception:
        warnings.warn(f'Failed to create a blank instance of {obj.__class__.__name__}. Falling back to reference.')
        return obj
    for attr, value in getattr(obj, '__dict__', {}).items():
        try:
            setattr(new_instance, attr, safe_deepcopy(value))
        except Exception:
            try:
                warnings.warn(f'Failed to copy {attr} of {obj.__class__.__name__}. Falling back to shallow copy.')
                setattr(new_instance, attr, copy.copy(value))
            except Exception:
                warnings.warn(f'Failed to copy {attr} of {obj.__class__.__name__}. Falling back to reference.')
                setattr(new_instance, attr, value)
    return new_instance

