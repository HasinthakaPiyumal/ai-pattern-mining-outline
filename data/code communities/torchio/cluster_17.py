# Cluster 17

def _stack(x):
    """Determine the type of the input and stack it accordingly.

    Args:
        x: List of elements to stack.
    Returns:
        Stacked elements, as either a torch.Tensor, np.ndarray, dict or list.
    """
    first_element = x[0]
    if isinstance(first_element, torch.Tensor):
        return torch.stack(x, dim=0)
    elif isinstance(first_element, np.ndarray):
        return np.stack(x, axis=0)
    elif isinstance(first_element, dict):
        collated_dict = {}
        for key in first_element.keys():
            collated_dict[key] = _stack([element[key] for element in x])
        return collated_dict
    else:
        return x

