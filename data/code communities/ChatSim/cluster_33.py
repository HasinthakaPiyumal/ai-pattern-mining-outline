# Cluster 33

class BasicBlock(nn.Module):

    def __init__(self, inplanes: int, planes: int, kernel_size=3, stride: int=1, act: str='relu', use_bn: bool=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'selu':
            self.act = nn.SELU(inplace=True)
        elif act == 'elu':
            self.act = nn.ELU(inplace=True)
        if inplanes != planes or stride != 1:
            self.downsample = nn.Sequential(conv1x1(inplanes, planes, stride), self.bn3)
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act(out)
        return out

def conv1x1(in_planes: int, out_planes: int, stride: int=1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

