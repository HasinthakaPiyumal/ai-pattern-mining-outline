# Cluster 25

class NaiveSingleView(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        view_args = args['view_setting']
        self.view_num = view_args['view_num']
        self.view_dis_deg = view_args['view_dis']
        self.view_dis_rad = [np.radians(x) for x in view_args['view_dis']]
        self.center_view = view_args['view_dis'].index(0)
        self.img_encoder = build_module(args['img_encoder'])
        self.shared_mlp = build_module(args['shared_mlp'])
        self.latent_mlp = build_module(args['latent_mlp'])
        self.peak_dir_mlp = build_module(args['peak_dir_mlp'])
        self.peak_int_mlp = build_module(args['peak_int_mlp'])

    def forward(self, x):
        """
        x: B, N_view, C, H, W
        """
        x = x[:, self.center_view, ...]
        x = self.img_encoder(x).permute(0, 2, 3, 1)
        x_flatten = x.flatten(1)
        deep_vector = self.shared_mlp(x_flatten)
        latent_vector = self.latent_mlp(deep_vector)
        peak_dir_vector = self.peak_dir_mlp(deep_vector)
        peak_int_vector = self.peak_int_mlp(deep_vector)
        peak_dir_vector = peak_dir_vector / peak_dir_vector.norm(dim=1, keepdim=True)
        peak_vector = torch.cat([peak_dir_vector, peak_int_vector], dim=-1)
        return (peak_vector, latent_vector)

def build_module(args):
    module_type = args['type']
    module_args = args['args']
    module_cls = eval(module_type)
    return module_cls(module_args)

class CatMultiView(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        view_args = args['view_setting']
        self.view_num = view_args['view_num']
        self.view_dis_deg = view_args['view_dis']
        self.sort_index = np.argsort(self.view_dis_deg)
        self.view_dis_rad = [np.radians(x) for x in view_args['view_dis']]
        self.center_view = view_args['view_dis'].index(0)
        self.img_encoder = build_module(args['img_encoder'])
        self.shared_mlp = build_module(args['shared_mlp'])
        self.latent_mlp = build_module(args['latent_mlp'])
        self.peak_dir_mlp = build_module(args['peak_dir_mlp'])
        self.peak_int_mlp = build_module(args['peak_int_mlp'])

    def forward(self, x):
        """
        x: B, N_view, C, H, W, suppose center view is the first 
        """
        B, N_view, C, H, W = x.shape
        x = x[:, self.sort_index, ...]
        x = x.permute(0, 2, 3, 1, 4).flatten(3)
        x = self.img_encoder(x).permute(0, 2, 3, 1)
        x_flatten = x.flatten(1)
        deep_vector = self.shared_mlp(x_flatten)
        latent_vector = self.latent_mlp(deep_vector)
        peak_dir_vector = self.peak_dir_mlp(deep_vector)
        peak_int_vector = self.peak_int_mlp(deep_vector)
        peak_dir_vector = peak_dir_vector / peak_dir_vector.norm(dim=1, keepdim=True)
        peak_vector = torch.cat([peak_dir_vector, peak_int_vector], dim=-1)
        return (peak_vector, latent_vector)

class AvgMultiView(nn.Module):
    """
    Avg is not suitable.

    use self.attention to fuse latent vector
    use max to fuse peak intensity
    use avg to fuse peak direction
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        view_args = args['view_setting']
        self.view_num = view_args['view_num']
        self.view_dis_deg = view_args['view_dis']
        self.view_dis_rad = [np.radians(x) for x in view_args['view_dis']]
        self.center_view = view_args['view_dis'].index(0)
        self.crop_H = view_args['camera_H'] // view_args['downsample_for_crop']
        self.crop_W = view_args['camera_W'] // view_args['downsample_for_crop']
        self.camera_vfov = np.degrees(np.arctan2(view_args['camera_H'] / 2, view_args['focal'])) * 2
        self.aspect_ratio = view_args['camera_W'] / view_args['camera_H']
        self.img_encoder = build_module(args['img_encoder'])
        self.shared_mlp = build_module(args['shared_mlp'])
        self.latent_mlp = build_module(args['latent_mlp'])
        int_channel = args['latent_mlp']['args']['layer_channels'][-1]
        self.att = ScaledDotProductAttention(int_channel)
        self.peak_dir_mlp = build_module(args['peak_dir_mlp'])
        self.peak_int_mlp = build_module(args['peak_int_mlp'])

    def forward(self, x):
        """
        x: B, N_view, C, H, W
        """
        B, N_view, _, _, _ = x.shape
        x = x.flatten(0, 1)
        x = self.img_encoder(x).permute(0, 2, 3, 1)
        x_flatten = x.flatten(1)
        deep_vector = self.shared_mlp(x_flatten)
        deep_vector = deep_vector.view(B, N_view, -1)
        latent_vector = self.latent_mlp(deep_vector)
        latent_vector = self.att(latent_vector[:, self.center_view:self.center_view + 1], latent_vector, latent_vector)
        peak_dir_vector = self.peak_dir_mlp(deep_vector)
        for i in range(self.view_num):
            azimuth_rad_i = self.view_dis_rad[i]
            rotation_mat_i = rotation_matrix(azimuth=azimuth_rad_i, elevation=0)
            inv_rotation_mat_i = rotation_matrix(azimuth=-azimuth_rad_i, elevation=0)
            inv_rotation_mat_i = torch.from_numpy(inv_rotation_mat_i).to(x.device).float()
            peak_dir_vector[:, i] = (inv_rotation_mat_i @ peak_dir_vector[:, i].T).T
        peak_dir_vector = peak_dir_vector / peak_dir_vector.norm(dim=-1, keepdim=True)
        peak_dir_vector_sum = peak_dir_vector.mean(dim=1)
        peak_dir_vector_avg = peak_dir_vector_sum / peak_dir_vector_sum.norm(dim=-1, keepdim=True)
        peak_int_vector = self.peak_int_mlp(deep_vector).mean(dim=1)
        peak_vector = torch.cat([peak_dir_vector_avg, peak_int_vector], dim=-1)
        return (peak_vector, latent_vector)

