# Cluster 136

def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    model = VisionTransformer(**kwargs)
    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location='cpu')
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
            print('Load pretrained model from: ' + pretrained)
    return model

def vit_base_patch16_224(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model

def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model

def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model

def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and 'OSTrack' not in cfg.MODEL.PRETRAIN_FILE and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''
    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, ce_loc=cfg.MODEL.BACKBONE.CE_LOC, ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, ce_loc=cfg.MODEL.BACKBONE.CE_LOC, ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    else:
        raise NotImplementedError
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    box_head = build_box_head(cfg, hidden_dim)
    model = OSTrack(backbone, box_head, aux_loss=False, head_type=cfg.MODEL.HEAD.TYPE)
    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['net'], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
    return model

