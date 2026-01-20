# Cluster 23

def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == 'truncated_normal':
        trunc_normal_(tensor, std=math.sqrt(variance) / 0.8796256610342398)
    elif distribution == 'normal':
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == 'uniform':
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f'invalid distribution {distribution}')

def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

@MODELS.register_module()
class SpUNetBase(nn.Module):

    def __init__(self, in_channels, out_channels, base_channels=32, channels=(32, 64, 128, 256, 256, 128, 96, 96), layers=(2, 3, 4, 6, 2, 2, 2, 2)):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        block = BasicBlock
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(in_channels, base_channels, kernel_size=5, padding=1, bias=False, indice_key='stem'), norm_fn(base_channels), nn.ReLU())
        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        for s in range(self.num_stages):
            self.down.append(spconv.SparseSequential(spconv.SparseConv3d(enc_channels, channels[s], kernel_size=2, stride=2, bias=False, indice_key=f'spconv{s + 1}'), norm_fn(channels[s]), nn.ReLU()))
            self.enc.append(spconv.SparseSequential(OrderedDict([(f'block{i}', block(channels[s], channels[s], norm_fn=norm_fn, indice_key=f'subm{s + 1}')) for i in range(layers[s])])))
            self.up.append(spconv.SparseSequential(spconv.SparseInverseConv3d(channels[len(channels) - s - 2], dec_channels, kernel_size=2, bias=False, indice_key=f'spconv{s + 1}'), norm_fn(dec_channels), nn.ReLU()))
            self.dec.append(spconv.SparseSequential(OrderedDict([(f'block{i}', block(dec_channels + enc_channels, dec_channels, norm_fn=norm_fn, indice_key=f'subm{s}')) if i == 0 else (f'block{i}', block(dec_channels, dec_channels, norm_fn=norm_fn, indice_key=f'subm{s}')) for i in range(layers[len(channels) - s - 1])])))
            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]
        self.final = spconv.SubMConv3d(channels[-1], out_channels, kernel_size=1, padding=1, bias=True) if out_channels > 0 else spconv.Identity()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_dict):
        discrete_coord = input_dict['discrete_coord']
        feat = input_dict['feat']
        offset = input_dict['offset']
        batch = offset2batch(offset)
        sparse_shape = torch.add(torch.max(discrete_coord, dim=0).values, 1).tolist()
        x = spconv.SparseConvTensor(features=feat, indices=torch.cat([batch.unsqueeze(-1).int(), discrete_coord.int()], dim=1).contiguous(), spatial_shape=sparse_shape, batch_size=batch[-1].tolist() + 1)
        x = self.conv_input(x)
        skips = [x]
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)
        x = skips.pop(-1)
        for s in reversed(range(self.num_stages)):
            x = self.up[s](x)
            skip = skips.pop(-1)
            x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
            x = self.dec[s](x)
        x = self.final(x)
        return x.features

@MODELS.register_module()
class SpUNetNoSkipBase(nn.Module):

    def __init__(self, in_channels, out_channels, base_channels=32, channels=(32, 64, 128, 256, 256, 128, 96, 96), layers=(2, 3, 4, 6, 2, 2, 2, 2)):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        block = BasicBlock
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(in_channels, base_channels, kernel_size=5, padding=1, bias=False, indice_key='stem'), norm_fn(base_channels), nn.ReLU())
        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        for s in range(self.num_stages):
            self.down.append(spconv.SparseSequential(spconv.SparseConv3d(enc_channels, channels[s], kernel_size=2, stride=2, bias=False, indice_key=f'spconv{s + 1}'), norm_fn(channels[s]), nn.ReLU()))
            self.enc.append(spconv.SparseSequential(OrderedDict([(f'block{i}', block(channels[s], channels[s], norm_fn=norm_fn, indice_key=f'subm{s + 1}')) for i in range(layers[s])])))
            self.up.append(spconv.SparseSequential(spconv.SparseInverseConv3d(channels[len(channels) - s - 2], dec_channels, kernel_size=2, bias=False, indice_key=f'spconv{s + 1}'), norm_fn(dec_channels), nn.ReLU()))
            self.dec.append(spconv.SparseSequential(OrderedDict([(f'block{i}', block(dec_channels, dec_channels, norm_fn=norm_fn, indice_key=f'subm{s}')) if i == 0 else (f'block{i}', block(dec_channels, dec_channels, norm_fn=norm_fn, indice_key=f'subm{s}')) for i in range(layers[len(channels) - s - 1])])))
            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]
        self.final = spconv.SubMConv3d(channels[-1], out_channels, kernel_size=1, padding=1, bias=True) if out_channels > 0 else spconv.Identity()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_dict):
        discrete_coord = input_dict['discrete_coord']
        feat = input_dict['feat']
        offset = input_dict['offset']
        batch = offset2batch(offset)
        sparse_shape = torch.add(torch.max(discrete_coord, dim=0).values, 1).tolist()
        x = spconv.SparseConvTensor(features=feat, indices=torch.cat([batch.unsqueeze(-1).int(), discrete_coord.int()], dim=1).contiguous(), spatial_shape=sparse_shape, batch_size=batch[-1].tolist() + 1)
        x = self.conv_input(x)
        skips = [x]
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)
        x = skips.pop(-1)
        for s in reversed(range(self.num_stages)):
            x = self.up[s](x)
            x = self.dec[s](x)
        x = self.final(x)
        return x.features

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, quant_size, rel_query=True, rel_key=False, rel_value=False, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.window_size = window_size
        self.quant_size = quant_size
        self.rel_query = rel_query
        self.rel_key = rel_key
        self.rel_value = rel_value
        quant_grid_length = int((2 * window_size + 0.0001) // quant_size)
        if rel_query:
            self.relative_pos_query_table = nn.Parameter(torch.zeros(2 * quant_grid_length, num_heads, head_dim, 3))
            trunc_normal_(self.relative_pos_query_table, std=0.02)
        if rel_key:
            self.relative_pos_key_table = nn.Parameter(torch.zeros(2 * quant_grid_length, num_heads, head_dim, 3))
            trunc_normal_(self.relative_pos_key_table, std=0.02)
        if rel_value:
            self.relative_pos_value_table = nn.Parameter(torch.zeros(2 * quant_grid_length, num_heads, head_dim, 3))
            trunc_normal_(self.relative_pos_value_table, std=0.02)
        self.quant_grid_length = quant_grid_length
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feats, xyz, index_0, index_1, index_0_offsets, n_max):
        """ Forward function.

        Args:
            feats: N, C
            xyz: N, 3
            index_0: M,
            index_1: M,
        """
        N, C = feats.shape
        M = index_0.shape[0]
        assert index_0.shape[0] == index_1.shape[0]
        qkv = self.qkv(feats).reshape(N, 3, self.num_heads, C // self.num_heads).permute(1, 0, 2, 3).contiguous()
        query, key, value = (qkv[0], qkv[1], qkv[2])
        query = query * self.scale
        attn_flat = pointops.attention_step1_v2(query.float(), key.float(), index_1.int(), index_0_offsets.int(), n_max)
        relative_position = xyz[index_0] - xyz[index_1]
        relative_position = torch.round(relative_position * 100000) / 100000
        relative_position_index = (relative_position + 2 * self.window_size - 0.0001) // self.quant_size
        assert (relative_position_index >= 0).all()
        assert (relative_position_index <= 2 * self.quant_grid_length - 1).all()
        assert self.rel_query and self.rel_key
        if self.rel_query and self.rel_key:
            relative_position_bias = pointops.dot_prod_with_idx_v3(query.float(), index_0_offsets.int(), n_max, key.float(), index_1.int(), self.relative_pos_query_table.float(), self.relative_pos_key_table.float(), relative_position_index.int())
        elif self.rel_query:
            relative_position_bias = pointops.dot_prod_with_idx(query.float(), index_0.int(), self.relative_pos_query_table.float(), relative_position_index.int())
        elif self.rel_key:
            relative_position_bias = pointops.dot_prod_with_idx(key.float(), index_1.int(), self.relative_pos_key_table.float(), relative_position_index.int())
        else:
            relative_position_bias = 0.0
        attn_flat = attn_flat + relative_position_bias
        softmax_attn_flat = scatter_softmax(src=attn_flat, index=index_0, dim=0)
        if self.rel_value:
            x = pointops.attention_step2_with_rel_pos_value_v2(softmax_attn_flat.float(), value.float(), index_0_offsets.int(), n_max, index_1.int(), self.relative_pos_value_table.float(), relative_position_index.int())
        else:
            x = pointops.attention_step2(softmax_attn_flat.float(), value.float(), index_0.int(), index_1.int())
        x = x.view(N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

@MODELS.register_module('stv1m1')
class StratifiedTransformer(nn.Module):

    def __init__(self, downsample_scale, depths, channels, num_heads, window_size, up_k, grid_sizes, quant_sizes, rel_query=True, rel_key=False, rel_value=False, drop_path_rate=0.2, num_layers=4, concat_xyz=False, num_classes=13, ratio=0.25, k=16, prev_grid_size=0.04, sigma=1.0, stem_transformer=False, kp_ball_radius=0.02 * 2.5, kp_max_neighbor=34):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.kp_ball_radius = kp_ball_radius
        self.kp_max_neighbor = kp_max_neighbor
        if stem_transformer:
            self.stem_layer = nn.ModuleList([KPConvSimpleBlock(3 if not concat_xyz else 6, channels[0], prev_grid_size, sigma=sigma)])
            self.layer_start = 0
        else:
            self.stem_layer = nn.ModuleList([KPConvSimpleBlock(3 if not concat_xyz else 6, channels[0], prev_grid_size, sigma=sigma), KPConvResBlock(channels[0], channels[0], prev_grid_size, sigma=sigma)])
            self.downsample = TransitionDown(channels[0], channels[1], ratio, k)
            self.layer_start = 1
        self.layers = nn.ModuleList([BasicLayer(downsample_scale, depths[i], channels[i], num_heads[i], window_size[i], grid_sizes[i], quant_sizes[i], rel_query=rel_query, rel_key=rel_key, rel_value=rel_value, drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], downsample=TransitionDown if i < num_layers - 1 else None, ratio=ratio, k=k, out_channels=channels[i + 1] if i < num_layers - 1 else None) for i in range(self.layer_start, num_layers)])
        self.upsamples = nn.ModuleList([Upsample(up_k, channels[i], channels[i - 1]) for i in range(num_layers - 1, 0, -1)])
        self.classifier = nn.Sequential(nn.Linear(channels[0], channels[0]), nn.BatchNorm1d(channels[0]), nn.ReLU(inplace=True), nn.Linear(channels[0], num_classes))
        self.init_weights()

    def forward(self, input_dict):
        feats = input_dict['feat']
        xyz = input_dict['coord']
        offset = input_dict['offset'].int()
        batch = offset2batch(offset)
        neighbor_idx = tp.ball_query(self.kp_ball_radius, self.kp_max_neighbor, xyz, xyz, mode='partial_dense', batch_x=batch, batch_y=batch)[0]
        feats_stack = []
        xyz_stack = []
        offset_stack = []
        for i, layer in enumerate(self.stem_layer):
            feats = layer(feats, xyz, batch, neighbor_idx)
        feats = feats.contiguous()
        if self.layer_start == 1:
            feats_stack.append(feats)
            xyz_stack.append(xyz)
            offset_stack.append(offset)
            feats, xyz, offset = self.downsample(feats, xyz, offset)
        for i, layer in enumerate(self.layers):
            feats, xyz, offset, feats_down, xyz_down, offset_down = layer(feats, xyz, offset)
            feats_stack.append(feats)
            xyz_stack.append(xyz)
            offset_stack.append(offset)
            feats = feats_down
            xyz = xyz_down
            offset = offset_down
        feats = feats_stack.pop()
        xyz = xyz_stack.pop()
        offset = offset_stack.pop()
        for i, upsample in enumerate(self.upsamples):
            feats, xyz, offset = upsample(feats, xyz, xyz_stack.pop(), offset, offset_stack.pop(), support_feats=feats_stack.pop())
        out = self.classifier(feats)
        return out

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    """

    def __init__(self, embed_channels, num_heads, window_size, quant_size, attn_drop=0.0, proj_drop=0.0, scale=None, rel_query=True, rel_key=True, rel_value=True, qkv_bias=True):
        super().__init__()
        self.embed_channels = embed_channels
        self.head_channels = embed_channels // num_heads
        self.num_heads = num_heads
        self.scale = scale or self.head_channels ** (-0.5)
        self.window_size = window_size
        self.quant_size = quant_size
        self.rel_query = rel_query
        self.rel_key = rel_key
        self.rel_value = rel_value
        self.quant_grid_length = int((2 * window_size + 0.0001) // quant_size)
        assert self.rel_query and self.rel_key
        if rel_query:
            self.relative_pos_query_table = nn.Parameter(torch.zeros(2 * self.quant_grid_length, self.num_heads, self.head_channels, 3))
            trunc_normal_(self.relative_pos_query_table, std=0.02)
        if rel_key:
            self.relative_pos_key_table = nn.Parameter(torch.zeros(2 * self.quant_grid_length, self.num_heads, self.head_channels, 3))
            trunc_normal_(self.relative_pos_query_table, std=0.02)
        if rel_value:
            self.relative_pos_value_table = nn.Parameter(torch.zeros(2 * self.quant_grid_length, self.num_heads, self.head_channels, 3))
            trunc_normal_(self.relative_pos_query_table, std=0.02)
        self.qkv = nn.Linear(embed_channels, embed_channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(embed_channels, embed_channels)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feats, coords, index_0, index_1, index_0_offsets, n_max):
        n, c = feats.shape
        m = index_0.shape[0]
        assert index_0.shape[0] == index_1.shape[0]
        qkv = self.qkv(feats).reshape(n, 3, self.num_heads, c // self.num_heads).permute(1, 0, 2, 3).contiguous()
        query, key, value = (qkv[0], qkv[1], qkv[2])
        query = query * self.scale
        attn_flat = pointops.attention_step1_v2(query.float(), key.float(), index_1.int(), index_0_offsets.int(), n_max)
        relative_position = coords[index_0] - coords[index_1]
        relative_position = torch.round(relative_position * 100000) / 100000
        relative_position_index = torch.div(relative_position + 2 * self.window_size - 0.0001, self.quant_size, rounding_mode='trunc')
        assert (relative_position_index >= 0).all()
        assert (relative_position_index <= 2 * self.quant_grid_length - 1).all()
        if self.rel_query and self.rel_key:
            relative_position_bias = pointops.dot_prod_with_idx_v3(query.float(), index_0_offsets.int(), n_max, key.float(), index_1.int(), self.relative_pos_query_table.float(), self.relative_pos_key_table.float(), relative_position_index.int())
        elif self.rel_query:
            relative_position_bias = pointops.dot_prod_with_idx(query.float(), index_0.int(), self.relative_pos_query_table.float(), relative_position_index.int())
        elif self.rel_key:
            relative_position_bias = pointops.dot_prod_with_idx(key.float(), index_1.int(), self.relative_pos_key_table.float(), relative_position_index.int())
        else:
            relative_position_bias = 0.0
        attn_flat += relative_position_bias
        softmax_attn_flat = scatter_softmax(src=attn_flat, index=index_0, dim=0)
        if self.rel_value:
            x = pointops.attention_step2_with_rel_pos_value_v2(softmax_attn_flat.float(), value.float(), index_0_offsets.int(), n_max, index_1.int(), self.relative_pos_value_table.float(), relative_position_index.int())
        else:
            x = pointops.attention_step2(softmax_attn_flat.float(), value.float(), index_0.int(), index_1.int())
        x = x.view(n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

@MODELS.register_module('stv1m2')
class StratifiedTransformer(nn.Module):

    def __init__(self, in_channels, num_classes, channels=(48, 96, 192, 384, 384), num_heads=(6, 12, 24, 24), depths=(3, 9, 3, 3), window_size=(0.2, 0.4, 0.8, 1.6), quant_size=(0.01, 0.02, 0.04, 0.08), mlp_expend_ratio=4.0, down_ratio=0.25, down_num_sample=16, kp_ball_radius=2.5 * 0.02, kp_max_neighbor=34, kp_grid_size=0.02, kp_sigma=1.0, drop_path_rate=0.2, rel_query=True, rel_key=True, rel_value=True, qkv_bias=True, stem=True):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.kp_ball_radius = kp_ball_radius
        self.kp_max_neighbor = kp_max_neighbor
        self.stem = stem
        if stem:
            self.point_embed = nn.ModuleList([KPConvSimpleBlock(in_channels, channels[0], kp_grid_size, sigma=kp_sigma), KPConvResBlock(channels[0], channels[0], kp_grid_size, sigma=kp_sigma)])
            self.down = TransitionDown(channels[0], channels[1], down_ratio, down_num_sample)
        else:
            assert channels[0] == channels[1]
            self.point_embed = nn.ModuleList([KPConvSimpleBlock(in_channels, channels[1], kp_grid_size, sigma=kp_sigma)])
        num_layers = len(depths)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = BasicLayer(embed_channels=channels[i + 1], out_channels=channels[i + 2] if i < num_layers - 1 else channels[i + 1], depth=depths[i], num_heads=num_heads[i], window_size=window_size[i], quant_size=quant_size[i], mlp_expend_ratio=mlp_expend_ratio, down_ratio=down_ratio, down_num_sample=down_num_sample, drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], rel_query=rel_query, rel_key=rel_key, rel_value=rel_value, qkv_bias=qkv_bias, down=True if i < num_layers - 1 else False)
            self.layers.append(layer)
        self.up = nn.ModuleList([TransitionUp(channels[i + 1], channels[i]) for i in reversed(range(1, num_layers))])
        if self.stem:
            self.up.append(TransitionUp(channels[1], channels[0]))
        self.classifier = nn.Sequential(nn.Linear(channels[0], channels[0]), nn.BatchNorm1d(channels[0]), nn.ReLU(inplace=True), nn.Linear(channels[0], num_classes))
        self.init_weights()

    def forward(self, input_dict):
        feats = input_dict['feat']
        coords = input_dict['coord']
        offset = input_dict['offset'].int()
        batch = offset2batch(offset)
        neighbor_idx = tp.ball_query(self.kp_ball_radius, self.kp_max_neighbor, coords, coords, mode='partial_dense', batch_x=batch, batch_y=batch)[0]
        feats_stack = []
        coords_stack = []
        offset_stack = []
        for i, layer in enumerate(self.point_embed):
            feats = layer(feats, coords, batch, neighbor_idx)
        feats = feats.contiguous()
        if self.stem:
            feats_stack.append(feats)
            coords_stack.append(coords)
            offset_stack.append(offset)
            feats, coords, offset = self.down(feats, coords, offset)
        for i, layer in enumerate(self.layers):
            feats, coords, offset, feats_down, coords_down, offset_down = layer(feats, coords, offset)
            feats_stack.append(feats)
            coords_stack.append(coords)
            offset_stack.append(offset)
            feats = feats_down
            coords = coords_down
            offset = offset_down
        feats = feats_stack.pop()
        coords = coords_stack.pop()
        offset = offset_stack.pop()
        for i, up in enumerate(self.up):
            feats, coords, offset = up(feats, coords, offset, feats_stack.pop(), coords_stack.pop(), offset_stack.pop())
        out = self.classifier(feats)
        return out

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

