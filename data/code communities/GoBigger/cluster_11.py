# Cluster 11

def flatten_data(data, start_dim=0, end_dim=1):
    if isinstance(data, dict):
        return {k: flatten_data(v, start_dim=start_dim, end_dim=end_dim) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return torch.flatten(data, start_dim=start_dim, end_dim=end_dim)

