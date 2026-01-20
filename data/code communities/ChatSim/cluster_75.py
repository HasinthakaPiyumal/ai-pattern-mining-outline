# Cluster 75

class BlurMask(nn.Module):

    def __init__(self, kernel_size=5, width_factor=1):
        super().__init__()
        self.filter = nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, padding_mode='replicate', bias=False)
        self.filter.weight.data.copy_(get_gauss_kernel(kernel_size, width_factor=width_factor))

    def forward(self, real_img, pred_img, mask):
        with torch.no_grad():
            result = self.filter(mask) * mask
            return result

def get_gauss_kernel(kernel_size, width_factor=1):
    coords = torch.stack(torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size)), dim=0).float()
    diff = torch.exp(-((coords - kernel_size // 2) ** 2).sum(0) / kernel_size / width_factor)
    diff /= diff.sum()
    return diff

class EmulatedEDTMask(nn.Module):

    def __init__(self, dilate_kernel_size=5, blur_kernel_size=5, width_factor=1):
        super().__init__()
        self.dilate_filter = nn.Conv2d(1, 1, dilate_kernel_size, padding=dilate_kernel_size // 2, padding_mode='replicate', bias=False)
        self.dilate_filter.weight.data.copy_(torch.ones(1, 1, dilate_kernel_size, dilate_kernel_size, dtype=torch.float))
        self.blur_filter = nn.Conv2d(1, 1, blur_kernel_size, padding=blur_kernel_size // 2, padding_mode='replicate', bias=False)
        self.blur_filter.weight.data.copy_(get_gauss_kernel(blur_kernel_size, width_factor=width_factor))

    def forward(self, real_img, pred_img, mask):
        with torch.no_grad():
            known_mask = 1 - mask
            dilated_known_mask = (self.dilate_filter(known_mask) > 1).float()
            result = self.blur_filter(1 - dilated_known_mask) * mask
            return result

class BlurMask(nn.Module):

    def __init__(self, kernel_size=5, width_factor=1):
        super().__init__()
        self.filter = nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, padding_mode='replicate', bias=False)
        self.filter.weight.data.copy_(get_gauss_kernel(kernel_size, width_factor=width_factor))

    def forward(self, real_img, pred_img, mask):
        with torch.no_grad():
            result = self.filter(mask) * mask
            return result

class EmulatedEDTMask(nn.Module):

    def __init__(self, dilate_kernel_size=5, blur_kernel_size=5, width_factor=1):
        super().__init__()
        self.dilate_filter = nn.Conv2d(1, 1, dilate_kernel_size, padding=dilate_kernel_size // 2, padding_mode='replicate', bias=False)
        self.dilate_filter.weight.data.copy_(torch.ones(1, 1, dilate_kernel_size, dilate_kernel_size, dtype=torch.float))
        self.blur_filter = nn.Conv2d(1, 1, blur_kernel_size, padding=blur_kernel_size // 2, padding_mode='replicate', bias=False)
        self.blur_filter.weight.data.copy_(get_gauss_kernel(blur_kernel_size, width_factor=width_factor))

    def forward(self, real_img, pred_img, mask):
        with torch.no_grad():
            known_mask = 1 - mask
            dilated_known_mask = (self.dilate_filter(known_mask) > 1).float()
            result = self.blur_filter(1 - dilated_known_mask) * mask
            return result

