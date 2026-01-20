# Cluster 27

def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)
    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)

def torch_tensor_to_numpy(torch_tensor):
    """
    Convert a torch tensor to numpy.

    Parameters
    ----------
    torch_tensor : torch.Tensor

    Returns
    -------
    A numpy array.
    """
    return torch_tensor.numpy() if not torch_tensor.is_cuda else torch_tensor.cpu().detach().numpy()

class PointPillarTransformer(nn.Module):

    def __init__(self, args):
        super(PointPillarTransformer, self).__init__()
        self.max_cav = args['max_cav']
        self.pillar_vfe = PillarVFE(args['pillar_vfe'], num_point_features=4, voxel_size=args['voxel_size'], point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False
        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        self.fusion_net = V2XTransformer(args['transformer'])
        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'], kernel_size=1)
        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False
        for p in self.scatter.parameters():
            p.requires_grad = False
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False
        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']
        prior_encoding = data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)
        batch_dict = {'voxel_features': voxel_features, 'voxel_coords': voxel_coords, 'voxel_num_points': voxel_num_points, 'record_len': record_len}
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        regroup_feature, mask = regroup(spatial_features_2d, record_len, self.max_cav)
        prior_encoding = prior_encoding.repeat(1, 1, 1, regroup_feature.shape[3], regroup_feature.shape[4])
        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        fused_feature = fused_feature.permute(0, 3, 1, 2)
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        output_dict = {'psm': psm, 'rm': rm}
        return output_dict

def regroup(dense_feature, record_len, max_len):
    """
    Regroup the data based on the record_len.

    Parameters
    ----------
    dense_feature : torch.Tensor
        N, C, H, W
    record_len : list
        [sample1_len, sample2_len, ...]
    max_len : int
        Maximum cav number

    Returns
    -------
    regroup_feature : torch.Tensor
        B, L, C, H, W
    """
    cum_sum_len = list(np.cumsum(torch_tensor_to_numpy(record_len)))
    split_features = torch.tensor_split(dense_feature, cum_sum_len[:-1])
    regroup_features = []
    mask = []
    for split_feature in split_features:
        feature_shape = split_feature.shape
        padding_len = max_len - feature_shape[0]
        mask.append([1] * feature_shape[0] + [0] * padding_len)
        padding_tensor = torch.zeros(padding_len, feature_shape[1], feature_shape[2], feature_shape[3])
        padding_tensor = padding_tensor.to(split_feature.device)
        split_feature = torch.cat([split_feature, padding_tensor], dim=0)
        split_feature = split_feature.view(-1, feature_shape[2], feature_shape[3]).unsqueeze(0)
        regroup_features.append(split_feature)
    regroup_features = torch.cat(regroup_features, dim=0)
    regroup_features = rearrange(regroup_features, 'b (l c) h w -> b l c h w', l=max_len)
    mask = torch.from_numpy(np.array(mask)).to(regroup_features.device)
    return (regroup_features, mask)

