# Cluster 97

class PPMDeepsup(nn.Module):

    def __init__(self, num_class=NUM_CLASS, fc_dim=4096, use_softmax=False, pool_scales=(1, 2, 3, 6), drop_last_conv=False):
        super().__init__()
        self.use_softmax = use_softmax
        self.drop_last_conv = drop_last_conv
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False), BatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last = nn.Sequential(nn.Conv2d(fc_dim + len(pool_scales) * 512, 512, kernel_size=3, padding=1, bias=False), BatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(512, num_class, kernel_size=1))
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(pool_scale(conv5), (input_size[2], input_size[3]), mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        if self.drop_last_conv:
            return ppm_out
        else:
            x = self.conv_last(ppm_out)
            if self.use_softmax:
                x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)
                x = nn.functional.softmax(x, dim=1)
                return x
            conv4 = conv_out[-2]
            _ = self.cbr_deepsup(conv4)
            _ = self.dropout_deepsup(_)
            _ = self.conv_last_deepsup(_)
            x = nn.functional.log_softmax(x, dim=1)
            _ = nn.functional.log_softmax(_, dim=1)
            return (x, _)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False), BatchNorm2d(out_planes), nn.ReLU(inplace=True))

class C1DeepSup(nn.Module):

    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False, drop_last_conv=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax
        self.drop_last_conv = drop_last_conv
        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        if self.drop_last_conv:
            return x
        else:
            x = self.conv_last(x)
            if self.use_softmax:
                x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)
                x = nn.functional.softmax(x, dim=1)
                return x
            conv4 = conv_out[-2]
            _ = self.cbr_deepsup(conv4)
            _ = self.conv_last_deepsup(_)
            x = nn.functional.log_softmax(x, dim=1)
            _ = nn.functional.log_softmax(_, dim=1)
            return (x, _)

class C1(nn.Module):

    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax
        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x

class PPMDeepsup(nn.Module):

    def __init__(self, num_class=NUM_CLASS, fc_dim=4096, use_softmax=False, pool_scales=(1, 2, 3, 6), drop_last_conv=False):
        super().__init__()
        self.use_softmax = use_softmax
        self.drop_last_conv = drop_last_conv
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False), BatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last = nn.Sequential(nn.Conv2d(fc_dim + len(pool_scales) * 512, 512, kernel_size=3, padding=1, bias=False), BatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(512, num_class, kernel_size=1))
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(pool_scale(conv5), (input_size[2], input_size[3]), mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        if self.drop_last_conv:
            return ppm_out
        else:
            x = self.conv_last(ppm_out)
            if self.use_softmax:
                x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)
                x = nn.functional.softmax(x, dim=1)
                return x
            conv4 = conv_out[-2]
            _ = self.cbr_deepsup(conv4)
            _ = self.dropout_deepsup(_)
            _ = self.conv_last_deepsup(_)
            x = nn.functional.log_softmax(x, dim=1)
            _ = nn.functional.log_softmax(_, dim=1)
            return (x, _)

class C1DeepSup(nn.Module):

    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False, drop_last_conv=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax
        self.drop_last_conv = drop_last_conv
        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        if self.drop_last_conv:
            return x
        else:
            x = self.conv_last(x)
            if self.use_softmax:
                x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)
                x = nn.functional.softmax(x, dim=1)
                return x
            conv4 = conv_out[-2]
            _ = self.cbr_deepsup(conv4)
            _ = self.conv_last_deepsup(_)
            x = nn.functional.log_softmax(x, dim=1)
            _ = nn.functional.log_softmax(_, dim=1)
            return (x, _)

class C1(nn.Module):

    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax
        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if self.use_softmax:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x

