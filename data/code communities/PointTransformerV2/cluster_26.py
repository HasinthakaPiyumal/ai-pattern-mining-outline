# Cluster 26

class BasicLayer(nn.Module):

    def __init__(self, downsample_scale, depth, channel, num_heads, window_size, grid_size, quant_size, rel_query=True, rel_key=False, rel_value=False, drop_path=0.0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, downsample=None, ratio=0.25, k=16, out_channels=None):
        super().__init__()
        self.depth = depth
        self.grid_size = grid_size
        self.max_window_counts = 64
        self.window_size = window_size
        self.downsample_scale = downsample_scale
        self.blocks = nn.ModuleList([SwinTransformerBlock(channel, num_heads, window_size, quant_size, rel_query=rel_query, rel_key=rel_key, rel_value=rel_value, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer) for i in range(depth)])
        self.downsample = downsample(channel, out_channels, ratio, k) if downsample else None

    def forward(self, feats, xyz, offset):
        window_size = torch.tensor([self.window_size] * 3).type_as(xyz).to(xyz.device)
        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0).long().cuda()
        v2p_map, p2v_map, counts = grid_sample(xyz, batch, window_size, start=None)
        shift_size = 1 / 2 * window_size
        shift_v2p_map, shift_p2v_map, shift_counts = grid_sample(xyz + shift_size, batch, window_size, start=xyz.min(0)[0])
        downsample_scale = self.downsample_scale
        new_offset, count = ([offset[0].item() // downsample_scale + 1], offset[0].item() // downsample_scale + 1)
        for i in range(1, offset.shape[0]):
            count += (offset[i].item() - offset[i - 1].item()) // downsample_scale + 1
            new_offset.append(count)
        new_offset = torch.cuda.IntTensor(new_offset)
        downsample_idx = pointops.furthestsampling(xyz, offset.int(), new_offset.int())
        new_window_size = 2 * torch.tensor([self.window_size] * 3).type_as(xyz).to(xyz.device)
        new_v2p_map, new_p2v_map, new_counts = grid_sample(xyz, batch, new_window_size, start=None)
        shift_size = 1 / 2 * new_window_size
        shift_new_v2p_map, shift_new_p2v_map, shift_new_counts = grid_sample(xyz + shift_size, batch, new_window_size, start=xyz.min(0)[0])
        for i, blk in enumerate(self.blocks):
            p2v_map_blk = p2v_map if i % 2 == 0 else shift_p2v_map
            counts_blk = counts if i % 2 == 0 else shift_counts
            new_p2v_map_blk = new_p2v_map if i % 2 == 0 else shift_new_p2v_map
            new_counts_blk = new_counts if i % 2 == 0 else shift_new_counts
            index_0, index_1 = get_indice_pairs(p2v_map_blk, counts_blk, new_p2v_map_blk, new_counts_blk, downsample_idx, batch, xyz, window_size, i)
            index_0, indices = torch.sort(index_0)
            index_1 = index_1[indices]
            index_0_counts = index_0.bincount()
            n_max = index_0_counts.max()
            index_0_offsets = index_0_counts.cumsum(dim=-1)
            index_0_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), index_0_offsets], 0)
            feats = blk(feats, xyz, index_0, index_1, index_0_offsets, n_max)
        if self.downsample:
            feats_down, xyz_down, offset_down = self.downsample(feats, xyz, offset)
        else:
            feats_down, xyz_down, offset_down = (None, None, None)
        return (feats, xyz, offset, feats_down, xyz_down, offset_down)

def grid_sample(pos, batch, size, start, return_p2v=True):
    cluster = voxel_grid(pos, batch, size, start=start)
    if return_p2v == False:
        unique, cluster = torch.unique(cluster, sorted=True, return_inverse=True)
        return cluster
    unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
    n = unique.shape[0]
    k = counts.max().item()
    p2v_map = cluster.new_zeros(n, k)
    mask = torch.arange(k).cuda().unsqueeze(0) < counts.unsqueeze(-1)
    p2v_map[mask] = torch.argsort(cluster)
    return (cluster, p2v_map, counts)

def get_indice_pairs(p2v_map, counts, new_p2v_map, new_counts, downsample_idx, batch, xyz, window_size, i):
    n, k = p2v_map.shape
    mask = torch.arange(k).unsqueeze(0).cuda() < counts.unsqueeze(-1)
    mask_mat = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    index_0 = p2v_map.unsqueeze(-1).expand(-1, -1, k)[mask_mat]
    index_1 = p2v_map.unsqueeze(1).expand(-1, k, -1)[mask_mat]
    downsample_mask = torch.zeros_like(batch).bool()
    downsample_mask[downsample_idx.long()] = True
    downsample_mask = downsample_mask[new_p2v_map]
    n, k = new_p2v_map.shape
    mask = torch.arange(k).unsqueeze(0).cuda() < new_counts.unsqueeze(-1)
    downsample_mask = downsample_mask & mask
    mask_mat = mask.unsqueeze(-1) & downsample_mask.unsqueeze(-2)
    xyz_min = xyz.min(0)[0]
    if i % 2 == 0:
        window_coord = (xyz[new_p2v_map] - xyz_min) // window_size
    else:
        window_coord = (xyz[new_p2v_map] + 1 / 2 * window_size - xyz_min) // window_size
    mask_mat_prev = (window_coord.unsqueeze(2) != window_coord.unsqueeze(1)).any(-1)
    mask_mat = mask_mat & mask_mat_prev
    new_index_0 = new_p2v_map.unsqueeze(-1).expand(-1, -1, k)[mask_mat]
    new_index_1 = new_p2v_map.unsqueeze(1).expand(-1, k, -1)[mask_mat]
    index_0 = torch.cat([index_0, new_index_0], 0)
    index_1 = torch.cat([index_1, new_index_1], 0)
    return (index_0, index_1)

class BasicLayer(nn.Module):

    def __init__(self, embed_channels, out_channels, depth, num_heads, window_size, quant_size, mlp_expend_ratio=4.0, down_ratio=0.25, down_num_sample=16, drop_path=None, qk_scale=None, down=True, rel_query=True, rel_key=True, rel_value=True, qkv_bias=True):
        super().__init__()
        self.depth = depth
        self.window_size = window_size
        self.quant_size = quant_size
        self.down_ratio = down_ratio
        if isinstance(drop_path, list):
            drop_path = drop_path
            assert len(drop_path) == depth
        elif isinstance(drop_path, float):
            drop_path = [deepcopy(drop_path) for _ in range(depth)]
        else:
            drop_path = [0.0 for _ in range(depth)]
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(embed_channels, num_heads, window_size, quant_size, mlp_expend_ratio=mlp_expend_ratio, drop_path=drop_path[i], qk_scale=qk_scale, rel_query=rel_query, rel_key=rel_key, rel_value=rel_value, qkv_bias=qkv_bias)
            self.blocks.append(block)
        self.down = TransitionDown(embed_channels, out_channels, down_ratio, down_num_sample) if down else None

    def forward(self, feats, coords, offset):
        window_size = torch.tensor([self.window_size] * 3, dtype=coords.dtype, device=coords.device)
        new_window_size = 2 * torch.tensor([self.window_size] * 3, dtype=coords.dtype, device=coords.device)
        batch = offset2batch(offset)
        new_offset = [int(offset[0].item() * self.down_ratio) + 1]
        count = int(offset[0].item() * self.down_ratio) + 1
        for i in range(1, offset.shape[0]):
            count += int((offset[i].item() - offset[i - 1].item()) * self.down_ratio) + 1
            new_offset.append(count)
        new_offset = torch.cuda.IntTensor(new_offset)
        down_idx = pointops.furthestsampling(coords, offset.int(), new_offset.int())
        coords_min = coords.min(0).values
        v2p_map, p2v_map, counts = grid_sample(coords, batch, window_size, start=None)
        shift_size = window_size * 1 / 2
        shift_v2p_map, shift_p2v_map, shift_counts = grid_sample(coords + shift_size, batch, window_size, start=coords_min)
        new_v2p_map, new_p2v_map, new_counts = grid_sample(coords, batch, new_window_size, start=None)
        shift_size = new_window_size * 1 / 2
        shift_new_v2p_map, shift_new_p2v_map, shift_new_counts = grid_sample(coords + shift_size, batch, new_window_size, start=coords_min)
        for i, blk in enumerate(self.blocks):
            p2v_map_blk = p2v_map if i % 2 == 0 else shift_p2v_map
            counts_blk = counts if i % 2 == 0 else shift_counts
            new_p2v_map_blk = new_p2v_map if i % 2 == 0 else shift_new_p2v_map
            new_counts_blk = new_counts if i % 2 == 0 else shift_new_counts
            n, k = p2v_map_blk.shape
            mask = torch.arange(k).unsqueeze(0).cuda() < counts_blk.unsqueeze(-1)
            mask_mat = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            index_0 = p2v_map_blk.unsqueeze(-1).expand(-1, -1, k)[mask_mat]
            index_1 = p2v_map_blk.unsqueeze(1).expand(-1, k, -1)[mask_mat]
            down_mask = torch.zeros_like(batch).bool()
            down_mask[down_idx.long()] = True
            down_mask = down_mask[new_p2v_map_blk]
            n, k = new_p2v_map_blk.shape
            mask = torch.arange(k).unsqueeze(0).cuda() < new_counts_blk.unsqueeze(-1)
            down_mask = down_mask & mask
            mask_mat = mask.unsqueeze(-1) & down_mask.unsqueeze(-2)
            if i % 2 == 0:
                window_coord = torch.div(coords[new_p2v_map_blk] - coords_min, window_size, rounding_mode='trunc')
            else:
                window_coord = torch.div(coords[new_p2v_map_blk] - coords_min + 1 / 2 * window_size, window_size, rounding_mode='trunc')
            mask_mat_prev = (window_coord.unsqueeze(2) != window_coord.unsqueeze(1)).any(-1)
            mask_mat = mask_mat & mask_mat_prev
            new_index_0 = new_p2v_map_blk.unsqueeze(-1).expand(-1, -1, k)[mask_mat]
            new_index_1 = new_p2v_map_blk.unsqueeze(1).expand(-1, k, -1)[mask_mat]
            index_0 = torch.cat([index_0, new_index_0], 0)
            index_1 = torch.cat([index_1, new_index_1], 0)
            index_0, indices = torch.sort(index_0)
            index_1 = index_1[indices]
            index_0_counts = index_0.bincount()
            n_max = index_0_counts.max()
            index_0_offsets = index_0_counts.cumsum(dim=-1)
            index_0_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), index_0_offsets], 0)
            feats = blk(feats, coords, index_0, index_1, index_0_offsets, n_max)
        if self.down:
            feats_down, coords_down, offset_down = self.down(feats, coords, offset)
        else:
            feats_down, coords_down, offset_down = (None, None, None)
        return (feats, coords, offset, feats_down, coords_down, offset_down)

