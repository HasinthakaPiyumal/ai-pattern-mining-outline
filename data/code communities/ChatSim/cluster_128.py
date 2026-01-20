# Cluster 128

def label_function_spatial(sz: torch.Tensor, sigma: torch.Tensor, center: torch.Tensor=torch.zeros(2), end_pad: torch.Tensor=torch.zeros(2)):
    """The origin is in the middle of the image."""
    return gauss_spatial(sz[0].item(), sigma[0].item(), center[0], end_pad[0].item()).reshape(1, 1, -1, 1) * gauss_spatial(sz[1].item(), sigma[1].item(), center[1], end_pad[1].item()).reshape(1, 1, 1, -1)

def gauss_spatial(sz, sigma, center=0, end_pad=0):
    k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad)
    return torch.exp(-1.0 / (2 * sigma ** 2) * (k - center) ** 2)

