# Cluster 16

def check_contain_nan(x):
    if isinstance(x, dict):
        return any((check_contain_nan(v) for k, v in x.items()))
    if isinstance(x, list):
        return any((check_contain_nan(itm) for itm in x))
    if isinstance(x, int) or isinstance(x, float):
        return False
    if isinstance(x, np.ndarray):
        return np.any(np.isnan(x))
    return torch.any(x.isnan()).detach().cpu().item()

