# Cluster 3

class V2XTEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        cav_att_config = args['cav_att_config']
        pwindow_att_config = args['pwindow_att_config']
        feed_config = args['feed_forward']
        num_blocks = args['num_blocks']
        depth = args['depth']
        mlp_dim = feed_config['mlp_dim']
        dropout = feed_config['dropout']
        self.downsample_rate = args['sttf']['downsample_rate']
        self.discrete_ratio = args['sttf']['voxel_size'][0]
        self.use_roi_mask = args['use_roi_mask']
        self.use_RTE = cav_att_config['use_RTE']
        self.RTE_ratio = cav_att_config['RTE_ratio']
        self.sttf = STTF(args['sttf'])
        self.prior_feed = nn.Linear(cav_att_config['dim'] + 3, cav_att_config['dim'])
        self.layers = nn.ModuleList([])
        if self.use_RTE:
            self.rte = RTE(cav_att_config['dim'], self.RTE_ratio)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([V2XFusionBlock(num_blocks, cav_att_config, pwindow_att_config), PreNorm(cav_att_config['dim'], FeedForward(cav_att_config['dim'], mlp_dim, dropout=dropout))]))

    def forward(self, x, mask, spatial_correction_matrix):
        prior_encoding = x[..., -3:]
        x = x[..., :-3]
        if self.use_RTE:
            dt = prior_encoding[:, :, 0, 0, 1].to(torch.int)
            x = self.rte(x, dt)
        x = self.sttf(x, mask, spatial_correction_matrix)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3) if not self.use_roi_mask else get_roi_and_cav_mask(x.shape, mask, spatial_correction_matrix, self.discrete_ratio, self.downsample_rate)
        for attn, ff in self.layers:
            x = attn(x, mask=com_mask, prior_encoding=prior_encoding)
            x = ff(x) + x
        return x

def get_roi_and_cav_mask(shape, cav_mask, spatial_correction_matrix, discrete_ratio, downsample_rate):
    """
    Get mask for the combination of cav_mask and rorated ROI mask.
    Parameters
    ----------
    shape : tuple
        Shape of (B, L, H, W, C).
    cav_mask : torch.Tensor
        Shape of (B, L).
    spatial_correction_matrix : torch.Tensor
        Shape of (B, L, 4, 4)
    discrete_ratio : float
        Discrete ratio.
    downsample_rate : float
        Downsample rate.

    Returns
    -------
    com_mask : torch.Tensor
        Combined mask with shape (B, H, W, L, 1).

    """
    B, L, H, W, C = shape
    C = 1
    dist_correction_matrix = get_discretized_transformation_matrix(spatial_correction_matrix, discrete_ratio, downsample_rate)
    T = get_transformation_matrix(dist_correction_matrix.reshape(-1, 2, 3), (H, W))
    roi_mask = get_rotated_roi((B, L, C, H, W), T)
    com_mask = combine_roi_and_cav_mask(roi_mask, cav_mask)
    com_mask = com_mask.permute(0, 3, 4, 2, 1)
    return com_mask

class Test:
    """
    Test the transformation in this file.
    The methods in this class are not supposed to be used outside of this file.
    """

    def __init__(self):
        pass

    @staticmethod
    def load_img():
        torch.manual_seed(0)
        x = torch.randn(1, 5, 16, 400, 200) * 100
        return x

    @staticmethod
    def load_raw_transformation_matrix(N):
        a = 90 / 180 * np.pi
        matrix = torch.Tensor([[np.cos(a), -np.sin(a), 10], [np.sin(a), np.cos(a), 10]])
        matrix = torch.repeat_interleave(matrix.unsqueeze(0).unsqueeze(0), N, dim=1)
        return matrix

    @staticmethod
    def load_raw_transformation_matrix2(N, alpha):
        a = alpha / 180 * np.pi
        matrix = torch.Tensor([[np.cos(a), -np.sin(a), 0, 0], [np.sin(a), np.cos(a), 0, 0]])
        matrix = torch.repeat_interleave(matrix.unsqueeze(0).unsqueeze(0), N, dim=1)
        return matrix

    @staticmethod
    def test():
        img = Test.load_img()
        B, L, C, H, W = img.shape
        raw_T = Test.load_raw_transformation_matrix(5)
        T = get_transformation_matrix(raw_T.reshape(-1, 2, 3), (H, W))
        img_rot = warp_affine(img.reshape(-1, C, H, W), T, (H, W))
        print(img_rot[0, 0, :, :])
        plt.matshow(img_rot[0, 0, :, :])
        plt.show()

    @staticmethod
    def test_combine_roi_and_cav_mask():
        B = 2
        L = 5
        C = 16
        H = 300
        W = 400
        cav_mask = torch.Tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])
        x = torch.zeros(B, L, C, H, W)
        correction_matrix = Test.load_raw_transformation_matrix2(5, 10)
        correction_matrix = torch.cat([correction_matrix, correction_matrix], dim=0)
        mask = get_roi_and_cav_mask((B, L, H, W, C), cav_mask, correction_matrix, 0.4, 4)
        plt.matshow(mask[0, :, :, 0, 0])
        plt.show()

