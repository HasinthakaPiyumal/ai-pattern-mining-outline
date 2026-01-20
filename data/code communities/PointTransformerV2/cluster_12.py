# Cluster 12

def knn_query_and_group(feat, xyz, offset=None, new_xyz=None, new_offset=None, idx=None, nsample=None, with_xyz=False):
    if idx is None:
        assert nsample is not None
        idx, _ = knn_query(nsample, xyz, offset, new_xyz, new_offset)
    return (grouping(idx, feat, xyz, new_xyz, with_xyz), idx)

def grouping(idx, feat, xyz, new_xyz=None, with_xyz=False):
    if new_xyz is None:
        new_xyz = xyz
    assert xyz.is_contiguous() and feat.is_contiguous()
    m, nsample, c = (idx.shape[0], idx.shape[1], feat.shape[1])
    xyz = torch.cat([xyz, torch.zeros([1, 3]).to(xyz.device)], dim=0)
    feat = torch.cat([feat, torch.zeros([1, c]).to(feat.device)], dim=0)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c)
    if with_xyz:
        assert new_xyz.is_contiguous()
        mask = torch.sign(idx + 1)
        grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3) - new_xyz.unsqueeze(1)
        grouped_xyz = torch.einsum('n s c, n s -> n s c', grouped_xyz, mask)
        return torch.cat((grouped_xyz, grouped_feat), -1)
    else:
        return grouped_feat

def ball_query_and_group(feat, xyz, offset=None, new_xyz=None, new_offset=None, idx=None, max_radio=None, min_radio=0, nsample=None, with_xyz=False):
    if idx is None:
        assert nsample is not None and offset is not None
        assert max_radio is not None and min_radio is not None
        idx, _ = ball_query(nsample, max_radio, min_radio, xyz, offset, new_xyz, new_offset)
    return (grouping(idx, feat, xyz, new_xyz, with_xyz), idx)

