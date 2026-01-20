# Cluster 66

@BACKBONES.register_module()
class TridentResNet(ResNet):
    """The stem layer, stage 1 and stage 2 in Trident ResNet are identical to
    ResNet, while in stage 3, Trident BottleBlock is utilized to replace the
    normal BottleBlock to yield trident output. Different branch shares the
    convolution weight but uses different dilations to achieve multi-scale
    output.

                               / stage3(b0)     x - stem - stage1 - stage2 - stage3(b1) - output
                               \\ stage3(b2) /

    Args:
        depth (int): Depth of resnet, from {50, 101, 152}.
        num_branch (int): Number of branches in TridentNet.
        test_branch_idx (int): In inference, all 3 branches will be used
            if `test_branch_idx==-1`, otherwise only branch with index
            `test_branch_idx` will be used.
        trident_dilations (tuple[int]): Dilations of different trident branch.
            len(trident_dilations) should be equal to num_branch.
    """

    def __init__(self, depth, num_branch, test_branch_idx, trident_dilations, **kwargs):
        assert num_branch == len(trident_dilations)
        assert depth in (50, 101, 152)
        super(TridentResNet, self).__init__(depth, **kwargs)
        assert self.num_stages == 3
        self.test_branch_idx = test_branch_idx
        self.num_branch = num_branch
        last_stage_idx = self.num_stages - 1
        stride = self.strides[last_stage_idx]
        dilation = trident_dilations
        dcn = self.dcn if self.stage_with_dcn[last_stage_idx] else None
        if self.plugins is not None:
            stage_plugins = self.make_stage_plugins(self.plugins, last_stage_idx)
        else:
            stage_plugins = None
        planes = self.base_channels * 2 ** last_stage_idx
        res_layer = make_trident_res_layer(TridentBottleneck, inplanes=self.block.expansion * self.base_channels * 2 ** (last_stage_idx - 1), planes=planes, num_blocks=self.stage_blocks[last_stage_idx], stride=stride, trident_dilations=dilation, style=self.style, with_cp=self.with_cp, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, dcn=dcn, plugins=stage_plugins, test_branch_idx=self.test_branch_idx)
        layer_name = f'layer{last_stage_idx + 1}'
        self.__setattr__(layer_name, res_layer)
        self.res_layers.pop(last_stage_idx)
        self.res_layers.insert(last_stage_idx, layer_name)
        self._freeze_stages()

def make_trident_res_layer(block, inplanes, planes, num_blocks, stride=1, trident_dilations=(1, 2, 3), style='pytorch', with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, plugins=None, test_branch_idx=-1):
    """Build Trident Res Layers."""
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = []
        conv_stride = stride
        downsample.extend([build_conv_layer(conv_cfg, inplanes, planes * block.expansion, kernel_size=1, stride=conv_stride, bias=False), build_norm_layer(norm_cfg, planes * block.expansion)[1]])
        downsample = nn.Sequential(*downsample)
    layers = []
    for i in range(num_blocks):
        layers.append(block(inplanes=inplanes, planes=planes, stride=stride if i == 0 else 1, trident_dilations=trident_dilations, downsample=downsample if i == 0 else None, style=style, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, dcn=dcn, plugins=plugins, test_branch_idx=test_branch_idx, concat_output=True if i == num_blocks - 1 else False))
        inplanes = planes * block.expansion
    return nn.Sequential(*layers)

