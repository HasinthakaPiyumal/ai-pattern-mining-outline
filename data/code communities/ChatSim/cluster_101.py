# Cluster 101

def gather_map(outputs):
    out = outputs[0]
    if torch.is_tensor(out):
        if out.dim() == 0:
            outputs = [o.unsqueeze(0) for o in outputs]
        return Gather.apply(target_device, dim, *outputs)
    elif out is None:
        return None
    elif isinstance(out, collections.Mapping):
        return {k: gather_map([o[k] for o in outputs]) for k in out}
    elif isinstance(out, collections.Sequence):
        return type(out)(map(gather_map, zip(*outputs)))

def dict_gather(outputs, target_device, dim=0):
    """
    Gathers variables from different GPUs on a specified device
      (-1 means the CPU), with dictionary support.
    """

    def gather_map(outputs):
        out = outputs[0]
        if torch.is_tensor(out):
            if out.dim() == 0:
                outputs = [o.unsqueeze(0) for o in outputs]
            return Gather.apply(target_device, dim, *outputs)
        elif out is None:
            return None
        elif isinstance(out, collections.Mapping):
            return {k: gather_map([o[k] for o in outputs]) for k in out}
        elif isinstance(out, collections.Sequence):
            return type(out)(map(gather_map, zip(*outputs)))
    return gather_map(outputs)

class DictGatherDataParallel(nn.DataParallel):

    def gather(self, outputs, output_device):
        return dict_gather(outputs, output_device, dim=self.dim)

def gather_map(outputs):
    out = outputs[0]
    if torch.is_tensor(out):
        if out.dim() == 0:
            outputs = [o.unsqueeze(0) for o in outputs]
        return Gather.apply(target_device, dim, *outputs)
    elif out is None:
        return None
    elif isinstance(out, collections.Mapping):
        return {k: gather_map([o[k] for o in outputs]) for k in out}
    elif isinstance(out, collections.Sequence):
        return type(out)(map(gather_map, zip(*outputs)))

def dict_gather(outputs, target_device, dim=0):
    """
    Gathers variables from different GPUs on a specified device
      (-1 means the CPU), with dictionary support.
    """

    def gather_map(outputs):
        out = outputs[0]
        if torch.is_tensor(out):
            if out.dim() == 0:
                outputs = [o.unsqueeze(0) for o in outputs]
            return Gather.apply(target_device, dim, *outputs)
        elif out is None:
            return None
        elif isinstance(out, collections.Mapping):
            return {k: gather_map([o[k] for o in outputs]) for k in out}
        elif isinstance(out, collections.Sequence):
            return type(out)(map(gather_map, zip(*outputs)))
    return gather_map(outputs)

class DictGatherDataParallel(nn.DataParallel):

    def gather(self, outputs, output_device):
        return dict_gather(outputs, output_device, dim=self.dim)

