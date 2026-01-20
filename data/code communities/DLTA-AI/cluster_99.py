# Cluster 99

def _check_mask_head(mask_cfg, mask_head):
    import torch.nn as nn
    if isinstance(mask_cfg, list):
        for single_mask_cfg, single_mask_head in zip(mask_cfg, mask_head):
            _check_mask_head(single_mask_cfg, single_mask_head)
    elif isinstance(mask_head, nn.ModuleList):
        for single_mask_head in mask_head:
            _check_mask_head(mask_cfg, single_mask_head)
    else:
        assert mask_cfg['type'] == mask_head.__class__.__name__
        assert mask_cfg.in_channels == mask_head.in_channels
        class_agnostic = mask_cfg.get('class_agnostic', False)
        out_dim = 1 if class_agnostic else mask_cfg.num_classes
        if hasattr(mask_head, 'conv_logits'):
            assert mask_cfg.conv_out_channels == mask_head.conv_logits.in_channels
            assert mask_head.conv_logits.out_channels == out_dim
        else:
            assert mask_cfg.fc_out_channels == mask_head.fc_logits.in_features
            assert mask_head.fc_logits.out_features == out_dim * mask_head.output_area

