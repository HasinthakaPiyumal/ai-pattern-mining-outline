# Cluster 3

@torch.no_grad()
def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert name in src_tensors or not require_all
        if name in src_tensors:
            srct = src_tensors[name]
            if tensor.shape == srct.shape:
                tensor.copy_(src_tensors[name])
            else:
                for i in range(len(tensor) // len(srct)):
                    tensor[i * len(srct):i * len(srct) + len(srct)] = srct
    dst_dic = dict(dst_module.named_modules())
    for src_submodule_name, src_submodule in src_module.named_modules():
        if hasattr(src_submodule, 'sdd'):
            if src_submodule_name in dst_dic:
                dst_submodule = dst_dic[src_submodule_name]
                if dst_submodule.sdd.k == src_submodule.sdd.k:
                    setattr(dst_submodule, 'sdd', getattr(src_submodule, 'sdd'))
                else:
                    dst_submodule.sdd.split_start = dst_submodule.sdd.k * 100

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def check_ddp_consistency(module, ignore_regex=None):
    assert isinstance(module, torch.nn.Module)
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        assert (tensor == other).all(), fullname

def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None):
    assert isinstance(input, torch.Tensor)
    if posinf is None:
        posinf = torch.finfo(input.dtype).max
    if neginf is None:
        neginf = torch.finfo(input.dtype).min
    assert nan == 0
    return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)

