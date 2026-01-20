# Cluster 117

def resnet50_ibn_b(num_classes, loss='softmax', pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

def nasnetamobile(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = NASNetAMobile(num_classes, loss, **kwargs)
    if pretrained:
        model_url = pretrained_settings['nasnetamobile']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model

def densenet121(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(num_classes=num_classes, loss=loss, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet121'])
    return model

def densenet169(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(num_classes=num_classes, loss=loss, num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet169'])
    return model

def densenet201(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(num_classes=num_classes, loss=loss, num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet201'])
    return model

def densenet161(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(num_classes=num_classes, loss=loss, num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet161'])
    return model

def densenet121_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(num_classes=num_classes, loss=loss, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), fc_dims=[512], dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet121'])
    return model

def senet154(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(num_classes=num_classes, loss=loss, block=SEBottleneck, layers=[3, 8, 36, 3], groups=64, reduction=16, dropout_p=0.2, last_stride=2, fc_dims=None, **kwargs)
    if pretrained:
        model_url = pretrained_settings['senet154']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model

def se_resnet50(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(num_classes=num_classes, loss=loss, block=SEResNetBottleneck, layers=[3, 4, 6, 3], groups=1, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, last_stride=2, fc_dims=None, **kwargs)
    if pretrained:
        model_url = pretrained_settings['se_resnet50']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model

def se_resnet50_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(num_classes=num_classes, loss=loss, block=SEResNetBottleneck, layers=[3, 4, 6, 3], groups=1, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, last_stride=1, fc_dims=[512], **kwargs)
    if pretrained:
        model_url = pretrained_settings['se_resnet50']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model

def se_resnet101(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(num_classes=num_classes, loss=loss, block=SEResNetBottleneck, layers=[3, 4, 23, 3], groups=1, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, last_stride=2, fc_dims=None, **kwargs)
    if pretrained:
        model_url = pretrained_settings['se_resnet101']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model

def se_resnet152(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(num_classes=num_classes, loss=loss, block=SEResNetBottleneck, layers=[3, 8, 36, 3], groups=1, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, last_stride=2, fc_dims=None, **kwargs)
    if pretrained:
        model_url = pretrained_settings['se_resnet152']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model

def se_resnext50_32x4d(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(num_classes=num_classes, loss=loss, block=SEResNeXtBottleneck, layers=[3, 4, 6, 3], groups=32, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, last_stride=2, fc_dims=None, **kwargs)
    if pretrained:
        model_url = pretrained_settings['se_resnext50_32x4d']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model

def se_resnext101_32x4d(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(num_classes=num_classes, loss=loss, block=SEResNeXtBottleneck, layers=[3, 4, 23, 3], groups=32, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, last_stride=2, fc_dims=None, **kwargs)
    if pretrained:
        model_url = pretrained_settings['se_resnext101_32x4d']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model

def squeezenet1_0(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SqueezeNet(num_classes, loss, version=1.0, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['squeezenet1_0'])
    return model

def squeezenet1_0_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SqueezeNet(num_classes, loss, version=1.0, fc_dims=[512], dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['squeezenet1_0'])
    return model

def squeezenet1_1(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SqueezeNet(num_classes, loss, version=1.1, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['squeezenet1_1'])
    return model

def resnet18(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(num_classes=num_classes, loss=loss, block=BasicBlock, layers=[2, 2, 2, 2], last_stride=2, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model

def resnet34(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(num_classes=num_classes, loss=loss, block=BasicBlock, layers=[3, 4, 6, 3], last_stride=2, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet34'])
    return model

def resnet50(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 4, 6, 3], last_stride=2, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def resnet101(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 4, 23, 3], last_stride=2, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])
    return model

def resnet152(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 8, 36, 3], last_stride=2, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet152'])
    return model

def resnext50_32x4d(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 4, 6, 3], last_stride=2, fc_dims=None, dropout_p=None, groups=32, width_per_group=4, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnext50_32x4d'])
    return model

def resnext101_32x8d(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 4, 23, 3], last_stride=2, fc_dims=None, dropout_p=None, groups=32, width_per_group=8, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnext101_32x8d'])
    return model

def resnet50_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 4, 6, 3], last_stride=1, fc_dims=[512], dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def osnet_x1_0(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2], channels=[64, 256, 384, 512], loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_x1_0')
    return model

def osnet_x0_75(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2], channels=[48, 192, 288, 384], loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_x0_75')
    return model

def osnet_x0_5(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2], channels=[32, 128, 192, 256], loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_x0_5')
    return model

def osnet_x0_25(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2], channels=[16, 64, 96, 128], loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_x0_25')
    return model

def osnet_ibn_x1_0(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2], channels=[64, 256, 384, 512], loss=loss, IN=True, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_ibn_x1_0')
    return model

def resnet50_ibn_a(num_classes, loss='softmax', pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def shufflenet_v2_x0_5(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ShuffleNetV2(num_classes, loss, [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['shufflenetv2_x0.5'])
    return model

def shufflenet_v2_x1_0(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ShuffleNetV2(num_classes, loss, [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['shufflenetv2_x1.0'])
    return model

def shufflenet_v2_x1_5(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ShuffleNetV2(num_classes, loss, [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['shufflenetv2_x1.5'])
    return model

def shufflenet_v2_x2_0(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ShuffleNetV2(num_classes, loss, [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['shufflenetv2_x2.0'])
    return model

def resnet50mid(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNetMid(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 4, 6, 3], last_stride=2, fc_dims=[1024], **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def inceptionv4(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = InceptionV4(num_classes, loss, **kwargs)
    if pretrained:
        model_url = pretrained_settings['inceptionv4']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model

def xception(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = Xception(num_classes, loss, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        model_url = pretrained_settings['xception']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model

def pcb_p6(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = PCB(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 4, 6, 3], last_stride=1, parts=6, reduced_dim=256, nonlinear='relu', **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def pcb_p4(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = PCB(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 4, 6, 3], last_stride=1, parts=4, reduced_dim=256, nonlinear='relu', **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def osnet_ain_x1_0(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[[OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin], [OSBlockINin, OSBlock]], layers=[2, 2, 2], channels=[64, 256, 384, 512], loss=loss, conv1_IN=True, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_ain_x1_0')
    return model

def osnet_ain_x0_75(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[[OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin], [OSBlockINin, OSBlock]], layers=[2, 2, 2], channels=[48, 192, 288, 384], loss=loss, conv1_IN=True, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_ain_x0_75')
    return model

def osnet_ain_x0_5(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[[OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin], [OSBlockINin, OSBlock]], layers=[2, 2, 2], channels=[32, 128, 192, 256], loss=loss, conv1_IN=True, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_ain_x0_5')
    return model

def osnet_ain_x0_25(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[[OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin], [OSBlockINin, OSBlock]], layers=[2, 2, 2], channels=[16, 64, 96, 128], loss=loss, conv1_IN=True, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_ain_x0_25')
    return model

