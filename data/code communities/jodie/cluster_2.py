# Cluster 2

def select_free_gpu():
    mem = []
    gpus = list(set(range(torch.cuda.device_count())))
    for i in gpus:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        mem.append(gpu_stats.jsonify()['gpus'][i]['memory.used'])
    return str(gpus[np.argmin(mem)])

