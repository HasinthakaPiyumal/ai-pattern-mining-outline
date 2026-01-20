# Cluster 95

def test_learning_rate_decay_optimizer_constructor():
    backbone = ToyConvNeXt()
    model = PseudoDataParallel(ToyDetector(backbone))
    optimizer_cfg = dict(type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05)
    stagewise_paramwise_cfg = dict(decay_rate=decay_rate, decay_type='stage_wise', num_layers=6)
    optim_constructor = LearningRateDecayOptimizerConstructor(optimizer_cfg, stagewise_paramwise_cfg)
    optimizer = optim_constructor(model)
    check_optimizer_lr_wd(optimizer, expected_stage_wise_lr_wd_convnext)
    layerwise_paramwise_cfg = dict(decay_rate=decay_rate, decay_type='layer_wise', num_layers=6)
    optim_constructor = LearningRateDecayOptimizerConstructor(optimizer_cfg, layerwise_paramwise_cfg)
    optimizer = optim_constructor(model)
    check_optimizer_lr_wd(optimizer, expected_layer_wise_lr_wd_convnext)

def check_optimizer_lr_wd(optimizer, gt_lr_wd):
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['weight_decay'] == base_wd
    param_groups = optimizer.param_groups
    print(param_groups)
    assert len(param_groups) == len(gt_lr_wd)
    for i, param_dict in enumerate(param_groups):
        assert param_dict['weight_decay'] == gt_lr_wd[i]['weight_decay']
        assert param_dict['lr_scale'] == gt_lr_wd[i]['lr_scale']
        assert param_dict['lr_scale'] == param_dict['lr']

