# Cluster 127

def label_function(sz: torch.Tensor, sigma: torch.Tensor):
    return gauss_fourier(sz[0].item(), sigma[0].item()).reshape(1, 1, -1, 1) * gauss_fourier(sz[1].item(), sigma[1].item(), True).reshape(1, 1, 1, -1)

def gauss_fourier(sz: int, sigma: float, half: bool=False) -> torch.Tensor:
    if half:
        k = torch.arange(0, int(sz / 2 + 1))
    else:
        k = torch.arange(-int((sz - 1) / 2), int(sz / 2 + 1))
    return math.sqrt(2 * math.pi) * sigma / sz * torch.exp(-2 * (math.pi * sigma * k.float() / sz) ** 2)

