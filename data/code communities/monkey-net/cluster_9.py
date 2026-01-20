# Cluster 9

def matrix_inverse(batch_of_matrix, eps=0):
    if eps != 0:
        init_shape = batch_of_matrix.shape
        a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
        b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
        c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
        d = batch_of_matrix[..., 1, 1].unsqueeze(-1)
        det = a * d - b * c
        out = torch.cat([d, -b, -c, a], dim=-1)
        eps = torch.tensor(eps).type(out.type())
        out /= det.max(eps)
        return out.view(init_shape)
    else:
        b_mat = batch_of_matrix
        eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
        b_inv, _ = torch.gesv(eye, b_mat)
        return b_inv

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

def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    x = 2 * (x / (w - 1)) - 1
    y = 2 * (y / (h - 1)) - 1
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)
    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    return meshed

class IdentityDeformation(nn.Module):

    def forward(self, appearance_frame, kp_video, kp_appearance):
        bs, _, _, h, w = appearance_frame.shape
        _, d, num_kp, _ = kp_video['mean'].shape
        coordinate_grid = make_coordinate_grid((h, w), type=appearance_frame.type())
        coordinate_grid = coordinate_grid.view(1, 1, h, w, 2).repeat(bs, d, 1, 1, 1)
        z_coordinate = torch.zeros(coordinate_grid.shape[:-1] + (1,)).type(coordinate_grid.type())
        return torch.cat([coordinate_grid, z_coordinate], dim=-1)

def kp2gaussian(kp, spatial_size, kp_variance='matrix'):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['mean']
    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)
    mean_sub = coordinate_grid - mean
    if kp_variance == 'matrix':
        var = kp['var']
        inv_var = matrix_inverse(var)
        shape = inv_var.shape[:number_of_leading_dimensions] + (1, 1, 2, 2)
        inv_var = inv_var.view(*shape)
        under_exp = torch.matmul(torch.matmul(mean_sub.unsqueeze(-2), inv_var), mean_sub.unsqueeze(-1))
        under_exp = under_exp.squeeze(-1).squeeze(-1)
        out = torch.exp(-0.5 * under_exp)
    elif kp_variance == 'single':
        out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp['var'])
    else:
        out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    return out

def gaussian2kp(heatmap, kp_variance='matrix', clip_variance=None):
    """
    Extract the mean and the variance from a heatmap
    """
    shape = heatmap.shape
    heatmap = heatmap.unsqueeze(-1) + 1e-07
    grid = make_coordinate_grid(shape[3:], heatmap.type()).unsqueeze_(0).unsqueeze_(0).unsqueeze_(0)
    mean = (heatmap * grid).sum(dim=(3, 4))
    kp = {'mean': mean.permute(0, 2, 1, 3)}
    if kp_variance == 'matrix':
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        var = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))
        var = var * heatmap.unsqueeze(-1)
        var = var.sum(dim=(3, 4))
        var = var.permute(0, 2, 1, 3, 4)
        if clip_variance:
            min_norm = torch.tensor(clip_variance).type(var.type())
            sg = smallest_singular(var).unsqueeze(-1)
            var = torch.max(min_norm, sg) * var / sg
        kp['var'] = var
    elif kp_variance == 'single':
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        var = mean_sub ** 2
        var = var * heatmap
        var = var.sum(dim=(3, 4))
        var = var.mean(dim=-1, keepdim=True)
        var = var.unsqueeze(-1)
        var = var.permute(0, 2, 1, 3, 4)
        kp['var'] = var
    return kp

def smallest_singular(batch_of_matrix):
    a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
    b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
    c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
    d = batch_of_matrix[..., 1, 1].unsqueeze(-1)
    s1 = a ** 2 + b ** 2 + c ** 2 + d ** 2
    s2 = (a ** 2 + b ** 2 - c ** 2 - d ** 2) ** 2
    s2 = torch.sqrt(s2 + 4 * (a * c + b * d) ** 2)
    norm = torch.sqrt((s1 - s2) / 2)
    return norm

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

class MovementEmbeddingModule(nn.Module):
    """
    Produce a keypoint representation that will be further used by other modules
    """

    def __init__(self, num_kp, kp_variance, num_channels, use_deformed_source_image=False, use_difference=False, use_heatmap=True, add_bg_feature_map=False, heatmap_type='gaussian', norm_const='sum', scale_factor=1):
        super(MovementEmbeddingModule, self).__init__()
        assert heatmap_type in ['gaussian', 'difference']
        assert int(use_heatmap) + int(use_deformed_source_image) + int(use_difference) >= 1
        self.out_channels = (1 * use_heatmap + 2 * use_difference + num_channels * use_deformed_source_image) * (num_kp + add_bg_feature_map)
        self.kp_variance = kp_variance
        self.heatmap_type = heatmap_type
        self.use_difference = use_difference
        self.use_deformed_source_image = use_deformed_source_image
        self.use_heatmap = use_heatmap
        self.add_bg_feature_map = add_bg_feature_map
        self.norm_const = norm_const
        self.scale_factor = scale_factor

    def normalize_heatmap(self, heatmap):
        if self.norm_const == 'sum':
            heatmap_shape = heatmap.shape
            heatmap = heatmap.view(heatmap_shape[0], heatmap_shape[1], heatmap_shape[2], -1)
            heatmap = heatmap / heatmap.sum(dim=3, keepdim=True)
            return heatmap.view(*heatmap_shape)
        else:
            return heatmap / self.norm_const

    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1:
            source_image = F.interpolate(source_image, scale_factor=(1, self.scale_factor, self.scale_factor))
        spatial_size = source_image.shape[3:]
        bs, _, _, h, w = source_image.shape
        _, d, num_kp, _ = kp_driving['mean'].shape
        inputs = []
        if self.use_heatmap:
            heatmap = self.normalize_heatmap(kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance))
            if self.heatmap_type == 'difference':
                heatmap_appearance = self.normalize_heatmap(kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance))
                heatmap = heatmap - heatmap_appearance
            if self.add_bg_feature_map:
                zeros = torch.zeros(bs, d, 1, h, w).type(heatmap.type())
                heatmap = torch.cat([zeros, heatmap], dim=2)
            heatmap = heatmap.unsqueeze(3)
            inputs.append(heatmap)
        num_kp += self.add_bg_feature_map
        if self.use_difference or self.use_deformed_source_image:
            kp_video_diff = kp_source['mean'] - kp_driving['mean']
            if self.add_bg_feature_map:
                zeros = torch.zeros(bs, d, 1, 2).type(kp_video_diff.type())
                kp_video_diff = torch.cat([zeros, kp_video_diff], dim=2)
            kp_video_diff = kp_video_diff.view((bs, d, num_kp, 2, 1, 1)).repeat(1, 1, 1, 1, h, w)
        if self.use_difference:
            inputs.append(kp_video_diff)
        if self.use_deformed_source_image:
            appearance_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, d, num_kp, 1, 1, 1, 1)
            appearance_repeat = appearance_repeat.view(bs * d * num_kp, -1, h, w)
            deformation_approx = kp_video_diff.view((bs * d * num_kp, -1, h, w)).permute(0, 2, 3, 1)
            coordinate_grid = make_coordinate_grid((h, w), type=deformation_approx.type())
            coordinate_grid = coordinate_grid.view(1, h, w, 2)
            deformation_approx = coordinate_grid + deformation_approx
            appearance_approx_deform = F.grid_sample(appearance_repeat, deformation_approx)
            appearance_approx_deform = appearance_approx_deform.view((bs, d, num_kp, -1, h, w))
            inputs.append(appearance_approx_deform)
        movement_encoding = torch.cat(inputs, dim=3)
        movement_encoding = movement_encoding.view(bs, d, -1, h, w)
        return movement_encoding.permute(0, 2, 1, 3, 4)

