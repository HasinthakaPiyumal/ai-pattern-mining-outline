# Cluster 35

class EncoderNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        layer_num = len(args['layer_channels'])
        layer_channels = args['layer_channels']
        kernel_size = args['kernel_size']
        strides = args['strides']
        block_nums = args['block_nums']
        use_bn = args['use_bn']
        act = args['act']
        inplanes = args['in_ch']
        module_list = []
        for i in range(layer_num):
            module_list.append(build_layer(inplanes, layer_channels[i], kernel_size, strides[i], block_nums[i], act, use_bn))
            inplanes = layer_channels[i]
        self.model = nn.Sequential(*module_list)

    def forward(self, x):
        return self.model(x)

def build_layer(inplane, plane, kernel_size, stride, block_num, act='relu', use_bn=True):
    module_list = []
    module_list.append(BasicBlock(inplane, plane, kernel_size=kernel_size, stride=stride, act=act, use_bn=use_bn))
    for _ in range(1, block_num):
        module_list.append(BasicBlock(plane, plane, kernel_size=kernel_size, act=act, use_bn=use_bn))
    return nn.Sequential(*module_list)

class DecoderNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        layer_num = len(args['layer_channels'])
        layer_channels = args['layer_channels']
        kernel_size = args['kernel_size']
        upstrides = args['upstrides']
        block_nums = args['block_nums']
        use_bn = args['use_bn']
        act = args['act']
        inplanes = args['in_ch']
        module_list = []
        for i in range(layer_num):
            module_list.append(build_up_layer(inplanes, layer_channels[i], kernel_size, upstrides[i], block_nums[i], act, use_bn))
            inplanes = layer_channels[i]
        self.model = nn.Sequential(*module_list)

    def forward(self, x):
        return self.model(x)

def build_up_layer(inplane, plane, kernel_size, stride, block_num, act='relu', use_bn=True):
    """
    Here stride refers to upsampling stride
    """
    module_list = []
    module_list.append(UpBasicBlock(inplane, plane, kernel_size=kernel_size, stride=stride, act=act, use_bn=use_bn))
    for _ in range(1, block_num):
        module_list.append(BasicBlock(plane, plane, kernel_size=kernel_size, act=act, use_bn=use_bn))
    return nn.Sequential(*module_list)

class UNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        layer_num = len(args['layer_channels'])
        self.layer_num = layer_num
        layer_channels = args['layer_channels']
        strides = args['strides']
        block_nums = args['block_nums']
        up_in_channels = args['up_in_channels']
        up_layer_channels = args['up_layer_channels']
        up_strides = args['up_strides']
        up_block_nums = args['up_block_nums']
        kernel_size = args['kernel_size']
        use_bn = args['use_bn']
        act = args['act']
        self.inject_latent = args['inject_latent']
        inplanes = args['in_ch']
        self.final_conv_to_RGB = args['final_conv_to_RGB']
        final_act = args.get('final_conv_to_RGB_act', 'relu')
        if final_act == 'relu':
            self.final_act = nn.ReLU()
        elif final_act == 'sigmoid':
            self.final_act = nn.Sigmoid()
        self.module_dict = nn.ModuleDict()
        for i in range(layer_num):
            self.module_dict[f'down{i}'] = build_layer(inplanes, layer_channels[i], kernel_size, strides[i], block_nums[i], act, use_bn)
            inplanes = layer_channels[i]
            self.module_dict[f'up{i}'] = build_up_layer(up_in_channels[i], up_layer_channels[i], kernel_size, up_strides[i], up_block_nums[i], act, use_bn)
        if self.final_conv_to_RGB:
            self.final_conv = nn.Sequential(nn.Conv2d(up_layer_channels[-1], 3, kernel_size=3, stride=1, padding=1), self.final_act)

    def forward(self, x, latent_feature=None):
        """
        Args:
            x : tensor 
                shape [N, C, H, W], C = 7
            latent_feature : tensor
                shape [N, C2, h, w], the small feature map from LDR encoder
        """
        x_downs = []
        for i in range(self.layer_num):
            x = self.module_dict[f'down{i}'](x)
            x_downs.append(x)
        if self.inject_latent:
            x = torch.cat([x, latent_feature], dim=1)
        x = self.module_dict[f'up0'](x)
        for i in range(1, self.layer_num):
            x = torch.cat([x, x_downs[self.layer_num - 1 - i]], dim=1)
            x = self.module_dict[f'up{i}'](x)
        if self.final_conv_to_RGB:
            x = self.final_conv(x)
        return x

