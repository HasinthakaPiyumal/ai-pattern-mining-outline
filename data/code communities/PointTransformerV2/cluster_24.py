# Cluster 24

class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = segment_csr(coord, torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]), reduce='min') if start is None else start
        cluster = voxel_grid(pos=coord - start[batch], size=self.grid_size, batch=batch, start=0)
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce='mean')
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce='max')
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return ([coord, feat, offset], cluster)

def offset2batch(offset):
    return torch.cat([torch.tensor([i] * (o - offset[i - 1])) if i > 0 else torch.tensor([i] * o) for i, o in enumerate(offset)], dim=0).long().to(offset.device)

def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()

class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = segment_csr(coord, torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]), reduce='min') if start is None else start
        cluster = voxel_grid(pos=coord - start[batch], size=self.grid_size, batch=batch, start=0)
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce='mean')
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce='max')
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return ([coord, feat, offset], cluster)

class MinkUNetBase(nn.Module):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    def __init__(self, in_channels, out_channels, dimension=3):
        super().__init__()
        self.D = dimension
        assert self.BLOCK is not None
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=5, dimension=self.D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)
        self.conv1p1s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=self.D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])
        self.conv2p2s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=self.D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])
        self.conv3p4s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=self.D)
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])
        self.conv4p8s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=self.D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])
        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=self.D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])
        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=self.D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])
        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=self.D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])
        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=self.D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])
        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])
        self.final = ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, out_channels, kernel_size=1, bias=True, dimension=self.D)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(ME.MinkowskiConvolution(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, dimension=self.D), ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, dimension=self.D))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D))
        return nn.Sequential(*layers)

    def forward(self, input_dict):
        discrete_coord = input_dict['discrete_coord']
        feat = input_dict['feat']
        offset = input_dict['offset']
        batch = offset2batch(offset)
        in_field = ME.TensorField(feat, coordinates=torch.cat([batch.unsqueeze(-1).int(), discrete_coord.int()], dim=1), quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE, minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, device=feat.device)
        x = in_field.sparse()
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)
        out = ME.cat(out, out_b3p8)
        out = self.block5(out)
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)
        out = ME.cat(out, out_b2p4)
        out = self.block6(out)
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)
        out = ME.cat(out, out_b1p2)
        out = self.block7(out)
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)
        out = ME.cat(out, out_p1)
        out = self.block8(out)
        return self.final(out).slice(in_field).F

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

@MODELS.register_module()
class SPVCNN(nn.Module):

    def __init__(self, in_channels, out_channels, base_channels=32, channels=(32, 64, 128, 256, 256, 128, 96, 96), layers=(2, 2, 2, 2, 2, 2, 2, 2)):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2
        self.stem = nn.Sequential(spnn.Conv3d(in_channels, base_channels, kernel_size=3, stride=1), spnn.BatchNorm(base_channels), spnn.ReLU(True), spnn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1), spnn.BatchNorm(base_channels), spnn.ReLU(True))
        self.stage1 = nn.Sequential(*[BasicConvolutionBlock(base_channels, base_channels, ks=2, stride=2, dilation=1), ResidualBlock(base_channels, channels[0], ks=3, stride=1, dilation=1)] + [ResidualBlock(channels[0], channels[0], ks=3, stride=1, dilation=1) for _ in range(layers[0] - 1)])
        self.stage2 = nn.Sequential(*[BasicConvolutionBlock(channels[0], channels[0], ks=2, stride=2, dilation=1), ResidualBlock(channels[0], channels[1], ks=3, stride=1, dilation=1)] + [ResidualBlock(channels[1], channels[1], ks=3, stride=1, dilation=1) for _ in range(layers[1] - 1)])
        self.stage3 = nn.Sequential(*[BasicConvolutionBlock(channels[1], channels[1], ks=2, stride=2, dilation=1), ResidualBlock(channels[1], channels[2], ks=3, stride=1, dilation=1)] + [ResidualBlock(channels[2], channels[2], ks=3, stride=1, dilation=1) for _ in range(layers[2] - 1)])
        self.stage4 = nn.Sequential(*[BasicConvolutionBlock(channels[2], channels[2], ks=2, stride=2, dilation=1), ResidualBlock(channels[2], channels[3], ks=3, stride=1, dilation=1)] + [ResidualBlock(channels[3], channels[3], ks=3, stride=1, dilation=1) for _ in range(layers[3] - 1)])
        self.up1 = nn.ModuleList([BasicDeconvolutionBlock(channels[3], channels[4], ks=2, stride=2), nn.Sequential(*[ResidualBlock(channels[4] + channels[2], channels[4], ks=3, stride=1, dilation=1)] + [ResidualBlock(channels[4], channels[4], ks=3, stride=1, dilation=1) for _ in range(layers[4] - 1)])])
        self.up2 = nn.ModuleList([BasicDeconvolutionBlock(channels[4], channels[5], ks=2, stride=2), nn.Sequential(*[ResidualBlock(channels[5] + channels[1], channels[5], ks=3, stride=1, dilation=1)] + [ResidualBlock(channels[5], channels[5], ks=3, stride=1, dilation=1) for _ in range(layers[5] - 1)])])
        self.up3 = nn.ModuleList([BasicDeconvolutionBlock(channels[5], channels[6], ks=2, stride=2), nn.Sequential(*[ResidualBlock(channels[6] + channels[0], channels[6], ks=3, stride=1, dilation=1)] + [ResidualBlock(channels[6], channels[6], ks=3, stride=1, dilation=1) for _ in range(layers[6] - 1)])])
        self.up4 = nn.ModuleList([BasicDeconvolutionBlock(channels[6], channels[7], ks=2, stride=2), nn.Sequential(*[ResidualBlock(channels[7] + base_channels, channels[7], ks=3, stride=1, dilation=1)] + [ResidualBlock(channels[7], channels[7], ks=3, stride=1, dilation=1) for _ in range(layers[7] - 1)])])
        self.classifier = nn.Sequential(nn.Linear(channels[7], out_channels))
        self.point_transforms = nn.ModuleList([nn.Sequential(nn.Linear(base_channels, channels[3]), nn.BatchNorm1d(channels[3]), nn.ReLU(True)), nn.Sequential(nn.Linear(channels[3], channels[5]), nn.BatchNorm1d(channels[5]), nn.ReLU(True)), nn.Sequential(nn.Linear(channels[5], channels[7]), nn.BatchNorm1d(channels[7]), nn.ReLU(True))])
        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_dict):
        discrete_coord = input_dict['discrete_coord']
        feat = input_dict['feat']
        offset = input_dict['offset']
        batch = offset2batch(offset)
        z = PointTensor(feat, torch.cat([discrete_coord.float(), batch.unsqueeze(-1).float()], dim=1).contiguous())
        x0 = initial_voxelize(z)
        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F
        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)
        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)
        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)
        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)
        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)
        out = self.classifier(z3.F)
        return out

def initial_voxelize(z):
    pc_hash = F.sphash(torch.floor(z.C).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    counts = F.spcount(idx_query.int(), len(sparse_hash))
    inserted_coords = F.spvoxelize(torch.floor(z.C), idx_query, counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    return new_tensor

def voxel_to_point(x, z, nearest=False):
    if z.idx_query is None or z.weights is None or z.idx_query.get(x.s) is None or (z.weights.get(x.s) is None):
        off = spnn.utils.get_kernel_offsets(2, x.s, 1, device=z.F.device)
        old_hash = F.sphash(torch.cat([torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0], z.C[:, -1].int().view(-1, 1)], 1), off)
        pc_hash = F.sphash(x.C.to(z.F.device))
        idx_query = F.sphashquery(old_hash, pc_hash)
        weights = F.calc_ti_weights(z.C, idx_query, scale=x.s[0]).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.0
            idx_query[:, 1:] = -1
        new_feat = F.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat, z.C, idx_query=z.idx_query, weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights
    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat, z.C, idx_query=z.idx_query, weights=z.weights)
        new_tensor.additional_features = z.additional_features
    return new_tensor

def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get('idx_query') is None or z.additional_features['idx_query'].get(x.s) is None:
        pc_hash = F.sphash(torch.cat([torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0], z.C[:, -1].int().view(-1, 1)], 1))
        sparse_hash = F.sphash(x.C)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]
    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps
    return new_tensor

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

