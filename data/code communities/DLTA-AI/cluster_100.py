# Cluster 100

def _check_bbox_head(bbox_cfg, bbox_head):
    import torch.nn as nn
    if isinstance(bbox_cfg, list):
        for single_bbox_cfg, single_bbox_head in zip(bbox_cfg, bbox_head):
            _check_bbox_head(single_bbox_cfg, single_bbox_head)
    elif isinstance(bbox_head, nn.ModuleList):
        for single_bbox_head in bbox_head:
            _check_bbox_head(bbox_cfg, single_bbox_head)
    else:
        assert bbox_cfg['type'] == bbox_head.__class__.__name__
        if bbox_cfg['type'] == 'SABLHead':
            assert bbox_cfg.cls_in_channels == bbox_head.cls_in_channels
            assert bbox_cfg.reg_in_channels == bbox_head.reg_in_channels
            cls_out_channels = bbox_cfg.get('cls_out_channels', 1024)
            assert cls_out_channels == bbox_head.fc_cls.in_features
            assert bbox_cfg.num_classes + 1 == bbox_head.fc_cls.out_features
        elif bbox_cfg['type'] == 'DIIHead':
            assert bbox_cfg['num_ffn_fcs'] == bbox_head.ffn.num_fcs
            assert bbox_cfg['num_cls_fcs'] == len(bbox_head.cls_fcs) // 3
            assert bbox_cfg['num_reg_fcs'] == len(bbox_head.reg_fcs) // 3
            assert bbox_cfg['in_channels'] == bbox_head.in_channels
            assert bbox_cfg['in_channels'] == bbox_head.fc_cls.in_features
            assert bbox_cfg['in_channels'] == bbox_head.fc_reg.in_features
            assert bbox_cfg['in_channels'] == bbox_head.attention.embed_dims
            assert bbox_cfg['feedforward_channels'] == bbox_head.ffn.feedforward_channels
        else:
            assert bbox_cfg.in_channels == bbox_head.in_channels
            with_cls = bbox_cfg.get('with_cls', True)
            if with_cls:
                fc_out_channels = bbox_cfg.get('fc_out_channels', 2048)
                assert fc_out_channels == bbox_head.fc_cls.in_features
                if bbox_head.custom_cls_channels:
                    assert bbox_head.loss_cls.get_cls_channels(bbox_head.num_classes) == bbox_head.fc_cls.out_features
                else:
                    assert bbox_cfg.num_classes + 1 == bbox_head.fc_cls.out_features
            with_reg = bbox_cfg.get('with_reg', True)
            if with_reg:
                out_dim = 4 if bbox_cfg.reg_class_agnostic else 4 * bbox_cfg.num_classes
                assert bbox_head.fc_reg.out_features == out_dim

