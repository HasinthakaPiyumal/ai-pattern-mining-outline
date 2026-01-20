# Cluster 49

class Discriminator(BaseNetwork):

    def __init__(self, in_channels=3, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 64
        self.conv = nn.Sequential(spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=nf * 1, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=1, bias=not use_spectral_norm), use_spectral_norm), nn.LeakyReLU(0.2, inplace=True), spectral_norm(nn.Conv3d(nf * 1, nf * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm), nn.LeakyReLU(0.2, inplace=True), spectral_norm(nn.Conv3d(nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm), nn.LeakyReLU(0.2, inplace=True), spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm), nn.LeakyReLU(0.2, inplace=True), spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm), nn.LeakyReLU(0.2, inplace=True), nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)))
        if init_weights:
            self.init_weights()

    def forward(self, xs):
        xs_t = torch.transpose(xs, 0, 1)
        xs_t = xs_t.unsqueeze(0)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)
        return out

def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module

def use_spectral_norm(module, use_sn=False):
    if use_sn:
        return spectral_norm(module)
    return module

