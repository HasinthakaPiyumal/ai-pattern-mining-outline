# Cluster 14

class DenseMotionModule(nn.Module):
    """
    Module that predicting a dense optical flow only from the displacement of a keypoints
    and the appearance of the first frame
    """

    def __init__(self, block_expansion, num_blocks, max_features, mask_embedding_params, num_kp, num_channels, kp_variance, use_correction, use_mask, bg_init=2, num_group_blocks=0, scale_factor=1):
        super(DenseMotionModule, self).__init__()
        self.mask_embedding = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance, num_channels=num_channels, add_bg_feature_map=True, **mask_embedding_params)
        self.difference_embedding = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance, num_channels=num_channels, add_bg_feature_map=True, use_difference=True, use_heatmap=False, use_deformed_source_image=False)
        group_blocks = []
        for i in range(num_group_blocks):
            group_blocks.append(SameBlock3D(self.mask_embedding.out_channels, self.mask_embedding.out_channels, groups=num_kp + 1, kernel_size=(1, 1, 1), padding=(0, 0, 0)))
        self.group_blocks = nn.ModuleList(group_blocks)
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=self.mask_embedding.out_channels, out_features=(num_kp + 1) * use_mask + 2 * use_correction, max_features=max_features, num_blocks=num_blocks)
        self.hourglass.decoder.conv.weight.data.zero_()
        bias_init = ([bg_init] + [0] * num_kp) * use_mask + [0, 0] * use_correction
        self.hourglass.decoder.conv.bias.data.copy_(torch.tensor(bias_init, dtype=torch.float))
        self.num_kp = num_kp
        self.use_correction = use_correction
        self.use_mask = use_mask
        self.scale_factor = scale_factor

    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1:
            source_image = F.interpolate(source_image, scale_factor=(1, self.scale_factor, self.scale_factor))
        prediction = self.mask_embedding(source_image, kp_driving, kp_source)
        for block in self.group_blocks:
            prediction = block(prediction)
            prediction = F.leaky_relu(prediction, 0.2)
        prediction = self.hourglass(prediction)
        bs, _, d, h, w = prediction.shape
        if self.use_mask:
            mask = prediction[:, :self.num_kp + 1]
            mask = F.softmax(mask, dim=1)
            mask = mask.unsqueeze(2)
            difference_embedding = self.difference_embedding(source_image, kp_driving, kp_source)
            difference_embedding = difference_embedding.view(bs, self.num_kp + 1, 2, d, h, w)
            deformations_relative = (difference_embedding * mask).sum(dim=1)
        else:
            deformations_relative = 0
        if self.use_correction:
            correction = prediction[:, -2:]
        else:
            correction = 0
        deformations_relative = deformations_relative + correction
        deformations_relative = deformations_relative.permute(0, 2, 3, 4, 1)
        coordinate_grid = make_coordinate_grid((h, w), type=deformations_relative.type())
        coordinate_grid = coordinate_grid.view(1, 1, h, w, 2)
        deformation = deformations_relative + coordinate_grid
        z_coordinate = torch.zeros(deformation.shape[:-1] + (1,)).type(deformation.type())
        return torch.cat([deformation, z_coordinate], dim=-1)

class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and variance.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features, num_blocks, temperature, kp_variance, scale_factor=1, clip_variance=None):
        super(KPDetector, self).__init__()
        self.predictor = Hourglass(block_expansion, in_features=num_channels, out_features=num_kp, max_features=max_features, num_blocks=num_blocks)
        self.temperature = temperature
        self.kp_variance = kp_variance
        self.scale_factor = scale_factor
        self.clip_variance = clip_variance

    def forward(self, x):
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=(1, self.scale_factor, self.scale_factor))
        heatmap = self.predictor(x)
        final_shape = heatmap.shape
        heatmap = heatmap.view(final_shape[0], final_shape[1], final_shape[2], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=3)
        heatmap = heatmap.view(*final_shape)
        out = gaussian2kp(heatmap, self.kp_variance, self.clip_variance)
        return out

class Discriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(self, num_channels=3, num_kp=10, kp_variance=0.01, scale_factor=1, block_expansion=64, num_blocks=4, max_features=512, kp_embedding_params=None):
        super(Discriminator, self).__init__()
        if kp_embedding_params is not None:
            self.kp_embedding = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance, num_channels=num_channels, **kp_embedding_params)
            embedding_channels = self.kp_embedding.out_channels
        else:
            self.kp_embedding = None
            embedding_channels = 0
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3D(num_channels + embedding_channels if i == 0 else min(max_features, block_expansion * 2 ** i), min(max_features, block_expansion * 2 ** (i + 1)), norm=i != 0, kernel_size=4))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv3d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x, kp_driving, kp_source):
        out_maps = [x]
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=(1, self.scale_factor, self.scale_factor))
        if self.kp_embedding:
            heatmap = self.kp_embedding(x, kp_driving, kp_source)
            out = torch.cat([x, heatmap], dim=1)
        else:
            out = x
        for down_block in self.down_blocks:
            out_maps.append(down_block(out))
            out = out_maps[-1]
        out = self.conv(out)
        out_maps.append(out)
        return out_maps

