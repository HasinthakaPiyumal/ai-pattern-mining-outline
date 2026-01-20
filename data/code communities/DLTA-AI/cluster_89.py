# Cluster 89

def test_resnext_backbone():
    with pytest.raises(KeyError):
        ResNeXt(depth=18)
    model = ResNeXt(depth=50, groups=32, base_width=4)
    for m in model.modules():
        if is_block(m):
            assert m.conv2.groups == 32
    model.train()
    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 8, 8])
    assert feat[1].shape == torch.Size([1, 512, 4, 4])
    assert feat[2].shape == torch.Size([1, 1024, 2, 2])
    assert feat[3].shape == torch.Size([1, 2048, 1, 1])

def is_block(modules):
    """Check if is ResNet building block."""
    if isinstance(modules, (BasicBlock, Bottleneck, BottleneckX, Bottle2neck, SimplifiedBasicBlock)):
        return True
    return False

def test_mobilenetv2_backbone():
    with pytest.raises(ValueError):
        MobileNetV2(frozen_stages=8)
    with pytest.raises(ValueError):
        MobileNetV2(out_indices=[8])
    frozen_stages = 1
    model = MobileNetV2(frozen_stages=frozen_stages)
    model.train()
    for mod in model.conv1.modules():
        for param in mod.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False
    model = MobileNetV2(norm_eval=True)
    model.train()
    assert check_norm_state(model.modules(), False)
    model = MobileNetV2(widen_factor=1.0, out_indices=range(0, 8))
    model.train()
    assert check_norm_state(model.modules(), True)
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 8
    assert feat[0].shape == torch.Size((1, 16, 112, 112))
    assert feat[1].shape == torch.Size((1, 24, 56, 56))
    assert feat[2].shape == torch.Size((1, 32, 28, 28))
    assert feat[3].shape == torch.Size((1, 64, 14, 14))
    assert feat[4].shape == torch.Size((1, 96, 14, 14))
    assert feat[5].shape == torch.Size((1, 160, 7, 7))
    assert feat[6].shape == torch.Size((1, 320, 7, 7))
    assert feat[7].shape == torch.Size((1, 1280, 7, 7))
    model = MobileNetV2(widen_factor=0.5, out_indices=range(0, 7))
    model.train()
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 7
    assert feat[0].shape == torch.Size((1, 8, 112, 112))
    assert feat[1].shape == torch.Size((1, 16, 56, 56))
    assert feat[2].shape == torch.Size((1, 16, 28, 28))
    assert feat[3].shape == torch.Size((1, 32, 14, 14))
    assert feat[4].shape == torch.Size((1, 48, 14, 14))
    assert feat[5].shape == torch.Size((1, 80, 7, 7))
    assert feat[6].shape == torch.Size((1, 160, 7, 7))
    model = MobileNetV2(widen_factor=2.0, out_indices=range(0, 8))
    model.train()
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat[0].shape == torch.Size((1, 32, 112, 112))
    assert feat[1].shape == torch.Size((1, 48, 56, 56))
    assert feat[2].shape == torch.Size((1, 64, 28, 28))
    assert feat[3].shape == torch.Size((1, 128, 14, 14))
    assert feat[4].shape == torch.Size((1, 192, 14, 14))
    assert feat[5].shape == torch.Size((1, 320, 7, 7))
    assert feat[6].shape == torch.Size((1, 640, 7, 7))
    assert feat[7].shape == torch.Size((1, 2560, 7, 7))
    model = MobileNetV2(widen_factor=1.0, act_cfg=dict(type='ReLU'), out_indices=range(0, 7))
    model.train()
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 7
    assert feat[0].shape == torch.Size((1, 16, 112, 112))
    assert feat[1].shape == torch.Size((1, 24, 56, 56))
    assert feat[2].shape == torch.Size((1, 32, 28, 28))
    assert feat[3].shape == torch.Size((1, 64, 14, 14))
    assert feat[4].shape == torch.Size((1, 96, 14, 14))
    assert feat[5].shape == torch.Size((1, 160, 7, 7))
    assert feat[6].shape == torch.Size((1, 320, 7, 7))
    model = MobileNetV2(widen_factor=1.0, out_indices=range(0, 7))
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)
    model.train()
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 7
    assert feat[0].shape == torch.Size((1, 16, 112, 112))
    assert feat[1].shape == torch.Size((1, 24, 56, 56))
    assert feat[2].shape == torch.Size((1, 32, 28, 28))
    assert feat[3].shape == torch.Size((1, 64, 14, 14))
    assert feat[4].shape == torch.Size((1, 96, 14, 14))
    assert feat[5].shape == torch.Size((1, 160, 7, 7))
    assert feat[6].shape == torch.Size((1, 320, 7, 7))
    model = MobileNetV2(widen_factor=1.0, norm_cfg=dict(type='GN', num_groups=2, requires_grad=True), out_indices=range(0, 7))
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, GroupNorm)
    model.train()
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 7
    assert feat[0].shape == torch.Size((1, 16, 112, 112))
    assert feat[1].shape == torch.Size((1, 24, 56, 56))
    assert feat[2].shape == torch.Size((1, 32, 28, 28))
    assert feat[3].shape == torch.Size((1, 64, 14, 14))
    assert feat[4].shape == torch.Size((1, 96, 14, 14))
    assert feat[5].shape == torch.Size((1, 160, 7, 7))
    assert feat[6].shape == torch.Size((1, 320, 7, 7))
    model = MobileNetV2(widen_factor=1.0, out_indices=(0, 2, 4))
    model.train()
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size((1, 16, 112, 112))
    assert feat[1].shape == torch.Size((1, 32, 28, 28))
    assert feat[2].shape == torch.Size((1, 96, 14, 14))
    model = MobileNetV2(widen_factor=1.0, with_cp=True, out_indices=range(0, 7))
    for m in model.modules():
        if is_block(m):
            assert m.with_cp
    model.train()
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 7
    assert feat[0].shape == torch.Size((1, 16, 112, 112))
    assert feat[1].shape == torch.Size((1, 24, 56, 56))
    assert feat[2].shape == torch.Size((1, 32, 28, 28))
    assert feat[3].shape == torch.Size((1, 64, 14, 14))
    assert feat[4].shape == torch.Size((1, 96, 14, 14))
    assert feat[5].shape == torch.Size((1, 160, 7, 7))
    assert feat[6].shape == torch.Size((1, 320, 7, 7))

def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True

def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False

def test_res2net_backbone():
    with pytest.raises(KeyError):
        Res2Net(depth=18)
    model = Res2Net(depth=50, scales=4, base_width=26)
    for m in model.modules():
        if is_block(m):
            assert m.scales == 4
    model.train()
    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 8, 8])
    assert feat[1].shape == torch.Size([1, 512, 4, 4])
    assert feat[2].shape == torch.Size([1, 1024, 2, 2])
    assert feat[3].shape == torch.Size([1, 2048, 1, 1])

def test_resnet_backbone():
    """Test resnet backbone."""
    with pytest.raises(KeyError):
        ResNet(20)
    with pytest.raises(AssertionError):
        ResNet(50, num_stages=0)
    with pytest.raises(AssertionError):
        dcn = dict(type='DCN', deform_groups=1, fallback_on_stride=False)
        ResNet(50, dcn=dcn, stage_with_dcn=(True,))
    with pytest.raises(AssertionError):
        plugins = [dict(cfg=dict(type='ContextBlock', ratio=1.0 / 16), stages=(False, True, True), position='after_conv3')]
        ResNet(50, plugins=plugins)
    with pytest.raises(AssertionError):
        ResNet(50, num_stages=5)
    with pytest.raises(AssertionError):
        ResNet(50, strides=(1,), dilations=(1, 1), num_stages=3)
    with pytest.raises(TypeError):
        model = ResNet(50, pretrained=0)
    with pytest.raises(AssertionError):
        ResNet(50, style='tensorflow')
    model = ResNet(50, norm_eval=True, base_channels=1)
    model.train()
    assert check_norm_state(model.modules(), False)
    model = ResNet(depth=50, norm_eval=True, pretrained='torchvision://resnet50')
    model.train()
    assert check_norm_state(model.modules(), False)
    frozen_stages = 1
    model = ResNet(50, frozen_stages=frozen_stages, base_channels=1)
    model.train()
    assert model.norm1.training is False
    for layer in [model.conv1, model.norm1]:
        for param in layer.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False
    model = ResNetV1d(depth=50, frozen_stages=frozen_stages, base_channels=2)
    assert len(model.stem) == 9
    model.train()
    assert check_norm_state(model.stem, False)
    for param in model.stem.parameters():
        assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False
    model = ResNet(18)
    model.train()
    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 64, 8, 8])
    assert feat[1].shape == torch.Size([1, 128, 4, 4])
    assert feat[2].shape == torch.Size([1, 256, 2, 2])
    assert feat[3].shape == torch.Size([1, 512, 1, 1])
    model = ResNet(18, with_cp=True)
    for m in model.modules():
        if is_block(m):
            assert m.with_cp
    model = ResNet(50, base_channels=1)
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)
    model.train()
    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 4, 8, 8])
    assert feat[1].shape == torch.Size([1, 8, 4, 4])
    assert feat[2].shape == torch.Size([1, 16, 2, 2])
    assert feat[3].shape == torch.Size([1, 32, 1, 1])
    model = ResNet(50, out_indices=(0, 1, 2), base_channels=1)
    model.train()
    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size([1, 4, 8, 8])
    assert feat[1].shape == torch.Size([1, 8, 4, 4])
    assert feat[2].shape == torch.Size([1, 16, 2, 2])
    model = ResNet(50, with_cp=True, base_channels=1)
    for m in model.modules():
        if is_block(m):
            assert m.with_cp
    model.train()
    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 4, 8, 8])
    assert feat[1].shape == torch.Size([1, 8, 4, 4])
    assert feat[2].shape == torch.Size([1, 16, 2, 2])
    assert feat[3].shape == torch.Size([1, 32, 1, 1])
    model = ResNet(50, base_channels=4, norm_cfg=dict(type='GN', num_groups=2, requires_grad=True))
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, GroupNorm)
    model.train()
    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 16, 8, 8])
    assert feat[1].shape == torch.Size([1, 32, 4, 4])
    assert feat[2].shape == torch.Size([1, 64, 2, 2])
    assert feat[3].shape == torch.Size([1, 128, 1, 1])
    plugins = [dict(cfg=dict(type='GeneralizedAttention', spatial_range=-1, num_heads=8, attention_type='0010', kv_stride=2), stages=(False, True, True, True), position='after_conv2'), dict(cfg=dict(type='NonLocal2d'), position='after_conv2'), dict(cfg=dict(type='ContextBlock', ratio=1.0 / 16), stages=(False, True, True, False), position='after_conv3')]
    model = ResNet(50, plugins=plugins, base_channels=8)
    for m in model.layer1.modules():
        if is_block(m):
            assert not hasattr(m, 'context_block')
            assert not hasattr(m, 'gen_attention_block')
            assert m.nonlocal_block.in_channels == 8
    for m in model.layer2.modules():
        if is_block(m):
            assert m.nonlocal_block.in_channels == 16
            assert m.gen_attention_block.in_channels == 16
            assert m.context_block.in_channels == 64
    for m in model.layer3.modules():
        if is_block(m):
            assert m.nonlocal_block.in_channels == 32
            assert m.gen_attention_block.in_channels == 32
            assert m.context_block.in_channels == 128
    for m in model.layer4.modules():
        if is_block(m):
            assert m.nonlocal_block.in_channels == 64
            assert m.gen_attention_block.in_channels == 64
            assert not hasattr(m, 'context_block')
    model.train()
    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 32, 8, 8])
    assert feat[1].shape == torch.Size([1, 64, 4, 4])
    assert feat[2].shape == torch.Size([1, 128, 2, 2])
    assert feat[3].shape == torch.Size([1, 256, 1, 1])
    plugins = [dict(cfg=dict(type='ContextBlock', ratio=1.0 / 16, postfix=1), stages=(False, True, True, False), position='after_conv3'), dict(cfg=dict(type='ContextBlock', ratio=1.0 / 16, postfix=2), stages=(False, True, True, False), position='after_conv3')]
    model = ResNet(50, plugins=plugins, base_channels=8)
    for m in model.layer1.modules():
        if is_block(m):
            assert not hasattr(m, 'context_block')
            assert not hasattr(m, 'context_block1')
            assert not hasattr(m, 'context_block2')
    for m in model.layer2.modules():
        if is_block(m):
            assert not hasattr(m, 'context_block')
            assert m.context_block1.in_channels == 64
            assert m.context_block2.in_channels == 64
    for m in model.layer3.modules():
        if is_block(m):
            assert not hasattr(m, 'context_block')
            assert m.context_block1.in_channels == 128
            assert m.context_block2.in_channels == 128
    for m in model.layer4.modules():
        if is_block(m):
            assert not hasattr(m, 'context_block')
            assert not hasattr(m, 'context_block1')
            assert not hasattr(m, 'context_block2')
    model.train()
    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 32, 8, 8])
    assert feat[1].shape == torch.Size([1, 64, 4, 4])
    assert feat[2].shape == torch.Size([1, 128, 2, 2])
    assert feat[3].shape == torch.Size([1, 256, 1, 1])
    model = ResNet(50, zero_init_residual=True, base_channels=1)
    model.init_weights()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            assert assert_params_all_zeros(m.norm3)
        elif isinstance(m, BasicBlock):
            assert assert_params_all_zeros(m.norm2)
    model.train()
    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 4, 8, 8])
    assert feat[1].shape == torch.Size([1, 8, 4, 4])
    assert feat[2].shape == torch.Size([1, 16, 2, 2])
    assert feat[3].shape == torch.Size([1, 32, 1, 1])
    model = ResNetV1d(depth=50, base_channels=2)
    model.train()
    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 8, 8, 8])
    assert feat[1].shape == torch.Size([1, 16, 4, 4])
    assert feat[2].shape == torch.Size([1, 32, 2, 2])
    assert feat[3].shape == torch.Size([1, 64, 1, 1])

