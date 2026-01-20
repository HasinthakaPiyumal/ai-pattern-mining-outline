# Cluster 28

def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device(f'cuda:{gpu_id}' if _use_gpu else 'cpu')

