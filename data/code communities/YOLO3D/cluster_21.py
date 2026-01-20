# Cluster 21

def prune(model, amount=0.3):
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)
            prune.remove(m, 'weight')
    print(' %.3g global sparsity' % sparsity(model))

def sparsity(model):
    a, b = (0, 0)
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a

