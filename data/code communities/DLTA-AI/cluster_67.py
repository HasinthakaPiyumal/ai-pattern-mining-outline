# Cluster 67

def model_scaling(layer_setting, arch_setting):
    """Scaling operation to the layer's parameters according to the
    arch_setting."""
    new_layer_setting = copy.deepcopy(layer_setting)
    for layer_cfg in new_layer_setting:
        for block_cfg in layer_cfg:
            block_cfg[1] = make_divisible(block_cfg[1] * arch_setting[0], 8)
    split_layer_setting = [new_layer_setting[0]]
    for layer_cfg in new_layer_setting[1:-1]:
        tmp_index = [0]
        for i in range(len(layer_cfg) - 1):
            if layer_cfg[i + 1][1] != layer_cfg[i][1]:
                tmp_index.append(i + 1)
        tmp_index.append(len(layer_cfg))
        for i in range(len(tmp_index) - 1):
            split_layer_setting.append(layer_cfg[tmp_index[i]:tmp_index[i + 1]])
    split_layer_setting.append(new_layer_setting[-1])
    num_of_layers = [len(layer_cfg) for layer_cfg in split_layer_setting[1:-1]]
    new_layers = [int(math.ceil(arch_setting[1] * num)) for num in num_of_layers]
    merge_layer_setting = [split_layer_setting[0]]
    for i, layer_cfg in enumerate(split_layer_setting[1:-1]):
        if new_layers[i] <= num_of_layers[i]:
            tmp_layer_cfg = layer_cfg[:new_layers[i]]
        else:
            tmp_layer_cfg = copy.deepcopy(layer_cfg) + [layer_cfg[-1]] * (new_layers[i] - num_of_layers[i])
        if tmp_layer_cfg[0][3] == 1 and i != 0:
            merge_layer_setting[-1] += tmp_layer_cfg.copy()
        else:
            merge_layer_setting.append(tmp_layer_cfg.copy())
    merge_layer_setting.append(split_layer_setting[-1])
    return merge_layer_setting

def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number to the nearest value that can be
    divisible by the divisor. It is taken from the original tf repo. It ensures
    that all layers have a channel number that is divisible by divisor. It can
    be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py  # noqa

    Args:
        value (int): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel number to
            the original channel number. Default: 0.9.

    Returns:
        int: The modified output channel number.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value

@BACKBONES.register_module()
class EfficientNet(BaseModule):
    """EfficientNet backbone.

    Args:
        arch (str): Architecture of efficientnet. Defaults to b0.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (6, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        conv_cfg (dict): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='Swish').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
    """
    layer_settings = {'b': [[[3, 32, 0, 2, 0, -1]], [[3, 16, 4, 1, 1, 0]], [[3, 24, 4, 2, 6, 0], [3, 24, 4, 1, 6, 0]], [[5, 40, 4, 2, 6, 0], [5, 40, 4, 1, 6, 0]], [[3, 80, 4, 2, 6, 0], [3, 80, 4, 1, 6, 0], [3, 80, 4, 1, 6, 0], [5, 112, 4, 1, 6, 0], [5, 112, 4, 1, 6, 0], [5, 112, 4, 1, 6, 0]], [[5, 192, 4, 2, 6, 0], [5, 192, 4, 1, 6, 0], [5, 192, 4, 1, 6, 0], [5, 192, 4, 1, 6, 0], [3, 320, 4, 1, 6, 0]], [[1, 1280, 0, 1, 0, -1]]], 'e': [[[3, 32, 0, 2, 0, -1]], [[3, 24, 0, 1, 3, 1]], [[3, 32, 0, 2, 8, 1], [3, 32, 0, 1, 8, 1]], [[3, 48, 0, 2, 8, 1], [3, 48, 0, 1, 8, 1], [3, 48, 0, 1, 8, 1], [3, 48, 0, 1, 8, 1]], [[5, 96, 0, 2, 8, 0], [5, 96, 0, 1, 8, 0], [5, 96, 0, 1, 8, 0], [5, 96, 0, 1, 8, 0], [5, 96, 0, 1, 8, 0], [5, 144, 0, 1, 8, 0], [5, 144, 0, 1, 8, 0], [5, 144, 0, 1, 8, 0], [5, 144, 0, 1, 8, 0]], [[5, 192, 0, 2, 8, 0], [5, 192, 0, 1, 8, 0]], [[1, 1280, 0, 1, 0, -1]]]}
    arch_settings = {'b0': (1.0, 1.0, 224), 'b1': (1.0, 1.1, 240), 'b2': (1.1, 1.2, 260), 'b3': (1.2, 1.4, 300), 'b4': (1.4, 1.8, 380), 'b5': (1.6, 2.2, 456), 'b6': (1.8, 2.6, 528), 'b7': (2.0, 3.1, 600), 'b8': (2.2, 3.6, 672), 'es': (1.0, 1.0, 224), 'em': (1.0, 1.1, 240), 'el': (1.2, 1.4, 300)}

    def __init__(self, arch='b0', drop_path_rate=0.0, out_indices=(6,), frozen_stages=0, conv_cfg=dict(type='Conv2dAdaptivePadding'), norm_cfg=dict(type='BN', eps=0.001), act_cfg=dict(type='Swish'), norm_eval=False, with_cp=False, init_cfg=[dict(type='Kaiming', layer='Conv2d'), dict(type='Constant', layer=['_BatchNorm', 'GroupNorm'], val=1)]):
        super(EfficientNet, self).__init__(init_cfg)
        assert arch in self.arch_settings, f'"{arch}" is not one of the arch_settings ({', '.join(self.arch_settings.keys())})'
        self.arch_setting = self.arch_settings[arch]
        self.layer_setting = self.layer_settings[arch[:1]]
        for index in out_indices:
            if index not in range(0, len(self.layer_setting)):
                raise ValueError(f'the item in out_indices must in range(0, {len(self.layer_setting)}). But received {index}')
        if frozen_stages not in range(len(self.layer_setting) + 1):
            raise ValueError(f'frozen_stages must be in range(0, {len(self.layer_setting) + 1}). But received {frozen_stages}')
        self.drop_path_rate = drop_path_rate
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.layer_setting = model_scaling(self.layer_setting, self.arch_setting)
        block_cfg_0 = self.layer_setting[0][0]
        block_cfg_last = self.layer_setting[-1][0]
        self.in_channels = make_divisible(block_cfg_0[1], 8)
        self.out_channels = block_cfg_last[1]
        self.layers = nn.ModuleList()
        self.layers.append(ConvModule(in_channels=3, out_channels=self.in_channels, kernel_size=block_cfg_0[0], stride=block_cfg_0[3], padding=block_cfg_0[0] // 2, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
        self.make_layer()
        if len(self.layers) < max(self.out_indices) + 1:
            self.layers.append(ConvModule(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=block_cfg_last[0], stride=block_cfg_last[3], padding=block_cfg_last[0] // 2, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))

    def make_layer(self):
        layer_setting = self.layer_setting[1:-1]
        total_num_blocks = sum([len(x) for x in layer_setting])
        block_idx = 0
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, total_num_blocks)]
        for i, layer_cfg in enumerate(layer_setting):
            if i > max(self.out_indices) - 1:
                break
            layer = []
            for i, block_cfg in enumerate(layer_cfg):
                kernel_size, out_channels, se_ratio, stride, expand_ratio, block_type = block_cfg
                mid_channels = int(self.in_channels * expand_ratio)
                out_channels = make_divisible(out_channels, 8)
                if se_ratio <= 0:
                    se_cfg = None
                else:
                    se_cfg = dict(channels=mid_channels, ratio=expand_ratio * se_ratio, act_cfg=(self.act_cfg, dict(type='Sigmoid')))
                if block_type == 1:
                    if i > 0 and expand_ratio == 3:
                        with_residual = False
                        expand_ratio = 4
                    else:
                        with_residual = True
                    mid_channels = int(self.in_channels * expand_ratio)
                    if se_cfg is not None:
                        se_cfg = dict(channels=mid_channels, ratio=se_ratio * expand_ratio, act_cfg=(self.act_cfg, dict(type='Sigmoid')))
                    block = partial(EdgeResidual, with_residual=with_residual)
                else:
                    block = InvertedResidual
                layer.append(block(in_channels=self.in_channels, out_channels=out_channels, mid_channels=mid_channels, kernel_size=kernel_size, stride=stride, se_cfg=se_cfg, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, drop_path_rate=dpr[block_idx], with_cp=self.with_cp, with_expand_conv=mid_channels != self.in_channels))
                self.in_channels = out_channels
                block_idx += 1
            self.layers.append(Sequential(*layer))

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(EfficientNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

@BACKBONES.register_module()
class MobileNetV2(BaseModule):
    """MobileNetV2 backbone.

    Args:
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int], optional): Output from which stages.
            Default: (1, 2, 4, 7).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    arch_settings = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]

    def __init__(self, widen_factor=1.0, out_indices=(1, 2, 4, 7), frozen_stages=-1, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU6'), norm_eval=False, with_cp=False, pretrained=None, init_cfg=None):
        super(MobileNetV2, self).__init__(init_cfg)
        self.pretrained = pretrained
        assert not (init_cfg and pretrained), 'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [dict(type='Kaiming', layer='Conv2d'), dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])]
        else:
            raise TypeError('pretrained must be a str or None')
        self.widen_factor = widen_factor
        self.out_indices = out_indices
        if not set(out_indices).issubset(set(range(0, 8))):
            raise ValueError(f'out_indices must be a subset of range(0, 8). But received {out_indices}')
        if frozen_stages not in range(-1, 8):
            raise ValueError(f'frozen_stages must be in range(-1, 8). But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.in_channels = make_divisible(32 * widen_factor, 8)
        self.conv1 = ConvModule(in_channels=3, out_channels=self.in_channels, kernel_size=3, stride=2, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.layers = []
        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks, stride = layer_cfg
            out_channels = make_divisible(channel * widen_factor, 8)
            inverted_res_layer = self.make_layer(out_channels=out_channels, num_blocks=num_blocks, stride=stride, expand_ratio=expand_ratio)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)
        if widen_factor > 1.0:
            self.out_channel = int(1280 * widen_factor)
        else:
            self.out_channel = 1280
        layer = ConvModule(in_channels=self.in_channels, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.add_module('conv2', layer)
        self.layers.append('conv2')

    def make_layer(self, out_channels, num_blocks, stride, expand_ratio):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): number of blocks.
            stride (int): stride of the first block. Default: 1
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio. Default: 6.
        """
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            layers.append(InvertedResidual(self.in_channels, out_channels, mid_channels=int(round(self.in_channels * expand_ratio)), stride=stride, with_expand_conv=expand_ratio != 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, with_cp=self.with_cp))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super(MobileNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

