# Cluster 21

def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)

def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

