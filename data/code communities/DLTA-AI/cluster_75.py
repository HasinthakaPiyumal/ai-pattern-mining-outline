# Cluster 75

@HEADS.register_module()
class SOLOHead(BaseMaskHead):
    """SOLO mask head used in `SOLO: Segmenting Objects by Locations.

    <https://arxiv.org/abs/1912.04488>`_

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
            Default: 256.
        stacked_convs (int): Number of stacking convs of the head.
            Default: 4.
        strides (tuple): Downsample factor of each feature map.
        scale_ranges (tuple[tuple[int, int]]): Area range of multiple
            level masks, in the format [(min1, max1), (min2, max2), ...].
            A range of (16, 64) means the area range between (16, 64).
        pos_scale (float): Constant scale factor to control the center region.
        num_grids (list[int]): Divided image into a uniform grids, each
            feature map has a different grid value. The number of output
            channels is grid ** 2. Default: [40, 36, 24, 16, 12].
        cls_down_index (int): The index of downsample operation in
            classification branch. Default: 0.
        loss_mask (dict): Config of mask loss.
        loss_cls (dict): Config of classification loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32,
                                   requires_grad=True).
        train_cfg (dict): Training config of head.
        test_cfg (dict): Testing config of head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, num_classes, in_channels, feat_channels=256, stacked_convs=4, strides=(4, 8, 16, 32, 64), scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)), pos_scale=0.2, num_grids=[40, 36, 24, 16, 12], cls_down_index=0, loss_mask=None, loss_cls=None, norm_cfg=dict(type='GN', num_groups=32, requires_grad=True), train_cfg=None, test_cfg=None, init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01), dict(type='Normal', std=0.01, bias_prob=0.01, override=dict(name='conv_mask_list')), dict(type='Normal', std=0.01, bias_prob=0.01, override=dict(name='conv_cls'))]):
        super(SOLOHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = self.num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.num_grids = num_grids
        self.num_levels = len(strides)
        assert self.num_levels == len(scale_ranges) == len(num_grids)
        self.scale_ranges = scale_ranges
        self.pos_scale = pos_scale
        self.cls_down_index = cls_down_index
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers()

    def _init_layers(self):
        self.mask_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.feat_channels
            self.mask_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, norm_cfg=self.norm_cfg))
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, norm_cfg=self.norm_cfg))
        self.conv_mask_list = nn.ModuleList()
        for num_grid in self.num_grids:
            self.conv_mask_list.append(nn.Conv2d(self.feat_channels, num_grid ** 2, 1))
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)

    def resize_feats(self, feats):
        """Downsample the first feat and upsample last feat in feats."""
        out = []
        for i in range(len(feats)):
            if i == 0:
                out.append(F.interpolate(feats[0], size=feats[i + 1].shape[-2:], mode='bilinear', align_corners=False))
            elif i == len(feats) - 1:
                out.append(F.interpolate(feats[i], size=feats[i - 1].shape[-2:], mode='bilinear', align_corners=False))
            else:
                out.append(feats[i])
        return out

    def forward(self, feats):
        assert len(feats) == self.num_levels
        feats = self.resize_feats(feats)
        mlvl_mask_preds = []
        mlvl_cls_preds = []
        for i in range(self.num_levels):
            x = feats[i]
            mask_feat = x
            cls_feat = x
            coord_feat = generate_coordinate(mask_feat.size(), mask_feat.device)
            mask_feat = torch.cat([mask_feat, coord_feat], 1)
            for mask_layer in self.mask_convs:
                mask_feat = mask_layer(mask_feat)
            mask_feat = F.interpolate(mask_feat, scale_factor=2, mode='bilinear')
            mask_pred = self.conv_mask_list[i](mask_feat)
            for j, cls_layer in enumerate(self.cls_convs):
                if j == self.cls_down_index:
                    num_grid = self.num_grids[i]
                    cls_feat = F.interpolate(cls_feat, size=num_grid, mode='bilinear')
                cls_feat = cls_layer(cls_feat)
            cls_pred = self.conv_cls(cls_feat)
            if not self.training:
                feat_wh = feats[0].size()[-2:]
                upsampled_size = (feat_wh[0] * 2, feat_wh[1] * 2)
                mask_pred = F.interpolate(mask_pred.sigmoid(), size=upsampled_size, mode='bilinear')
                cls_pred = cls_pred.sigmoid()
                local_max = F.max_pool2d(cls_pred, 2, stride=1, padding=1)
                keep_mask = local_max[:, :, :-1, :-1] == cls_pred
                cls_pred = cls_pred * keep_mask
            mlvl_mask_preds.append(mask_pred)
            mlvl_cls_preds.append(cls_pred)
        return (mlvl_mask_preds, mlvl_cls_preds)

    def loss(self, mlvl_mask_preds, mlvl_cls_preds, gt_labels, gt_masks, img_metas, gt_bboxes=None, **kwargs):
        """Calculate the loss of total batch.

        Args:
            mlvl_mask_preds (list[Tensor]): Multi-level mask prediction.
                Each element in the list has shape
                (batch_size, num_grids**2 ,h ,w).
            mlvl_cls_preds (list[Tensor]): Multi-level scores. Each element
                in the list has shape
                (batch_size, num_classes, num_grids ,num_grids).
            gt_labels (list[Tensor]): Labels of multiple images.
            gt_masks (list[Tensor]): Ground truth masks of multiple images.
                Each has shape (num_instances, h, w).
            img_metas (list[dict]): Meta information of multiple images.
            gt_bboxes (list[Tensor]): Ground truth bboxes of multiple
                images. Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_levels = self.num_levels
        num_imgs = len(gt_labels)
        featmap_sizes = [featmap.size()[-2:] for featmap in mlvl_mask_preds]
        pos_mask_targets, labels, pos_masks = multi_apply(self._get_targets_single, gt_bboxes, gt_labels, gt_masks, featmap_sizes=featmap_sizes)
        mlvl_pos_mask_targets = [[] for _ in range(num_levels)]
        mlvl_pos_mask_preds = [[] for _ in range(num_levels)]
        mlvl_pos_masks = [[] for _ in range(num_levels)]
        mlvl_labels = [[] for _ in range(num_levels)]
        for img_id in range(num_imgs):
            assert num_levels == len(pos_mask_targets[img_id])
            for lvl in range(num_levels):
                mlvl_pos_mask_targets[lvl].append(pos_mask_targets[img_id][lvl])
                mlvl_pos_mask_preds[lvl].append(mlvl_mask_preds[lvl][img_id, pos_masks[img_id][lvl], ...])
                mlvl_pos_masks[lvl].append(pos_masks[img_id][lvl].flatten())
                mlvl_labels[lvl].append(labels[img_id][lvl].flatten())
        temp_mlvl_cls_preds = []
        for lvl in range(num_levels):
            mlvl_pos_mask_targets[lvl] = torch.cat(mlvl_pos_mask_targets[lvl], dim=0)
            mlvl_pos_mask_preds[lvl] = torch.cat(mlvl_pos_mask_preds[lvl], dim=0)
            mlvl_pos_masks[lvl] = torch.cat(mlvl_pos_masks[lvl], dim=0)
            mlvl_labels[lvl] = torch.cat(mlvl_labels[lvl], dim=0)
            temp_mlvl_cls_preds.append(mlvl_cls_preds[lvl].permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels))
        num_pos = sum((item.sum() for item in mlvl_pos_masks))
        loss_mask = []
        for pred, target in zip(mlvl_pos_mask_preds, mlvl_pos_mask_targets):
            if pred.size()[0] == 0:
                loss_mask.append(pred.sum().unsqueeze(0))
                continue
            loss_mask.append(self.loss_mask(pred, target, reduction_override='none'))
        if num_pos > 0:
            loss_mask = torch.cat(loss_mask).sum() / num_pos
        else:
            loss_mask = torch.cat(loss_mask).mean()
        flatten_labels = torch.cat(mlvl_labels)
        flatten_cls_preds = torch.cat(temp_mlvl_cls_preds)
        loss_cls = self.loss_cls(flatten_cls_preds, flatten_labels, avg_factor=num_pos + 1)
        return dict(loss_mask=loss_mask, loss_cls=loss_cls)

    def _get_targets_single(self, gt_bboxes, gt_labels, gt_masks, featmap_sizes=None):
        """Compute targets for predictions of single image.

        Args:
            gt_bboxes (Tensor): Ground truth bbox of each instance,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth label of each instance,
                shape (num_gts,).
            gt_masks (Tensor): Ground truth mask of each instance,
                shape (num_gts, h, w).
            featmap_sizes (list[:obj:`torch.size`]): Size of each
                feature map from feature pyramid, each element
                means (feat_h, feat_w). Default: None.

        Returns:
            Tuple: Usually returns a tuple containing targets for predictions.

                - mlvl_pos_mask_targets (list[Tensor]): Each element represent
                  the binary mask targets for positive points in this
                  level, has shape (num_pos, out_h, out_w).
                - mlvl_labels (list[Tensor]): Each element is
                  classification labels for all
                  points in this level, has shape
                  (num_grid, num_grid).
                - mlvl_pos_masks (list[Tensor]): Each element is
                  a `BoolTensor` to represent whether the
                  corresponding point in single level
                  is positive, has shape (num_grid **2).
        """
        device = gt_labels.device
        gt_areas = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1]))
        mlvl_pos_mask_targets = []
        mlvl_labels = []
        mlvl_pos_masks = []
        for (lower_bound, upper_bound), stride, featmap_size, num_grid in zip(self.scale_ranges, self.strides, featmap_sizes, self.num_grids):
            mask_target = torch.zeros([num_grid ** 2, featmap_size[0], featmap_size[1]], dtype=torch.uint8, device=device)
            labels = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device) + self.num_classes
            pos_mask = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)
            gt_inds = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(gt_inds) == 0:
                mlvl_pos_mask_targets.append(mask_target.new_zeros(0, featmap_size[0], featmap_size[1]))
                mlvl_labels.append(labels)
                mlvl_pos_masks.append(pos_mask)
                continue
            hit_gt_bboxes = gt_bboxes[gt_inds]
            hit_gt_labels = gt_labels[gt_inds]
            hit_gt_masks = gt_masks[gt_inds, ...]
            pos_w_ranges = 0.5 * (hit_gt_bboxes[:, 2] - hit_gt_bboxes[:, 0]) * self.pos_scale
            pos_h_ranges = 0.5 * (hit_gt_bboxes[:, 3] - hit_gt_bboxes[:, 1]) * self.pos_scale
            valid_mask_flags = hit_gt_masks.sum(dim=-1).sum(dim=-1) > 0
            output_stride = stride / 2
            for gt_mask, gt_label, pos_h_range, pos_w_range, valid_mask_flag in zip(hit_gt_masks, hit_gt_labels, pos_h_ranges, pos_w_ranges, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4)
                center_h, center_w = center_of_mass(gt_mask)
                coord_w = int(floordiv(center_w / upsampled_size[1], 1.0 / num_grid, rounding_mode='trunc'))
                coord_h = int(floordiv(center_h / upsampled_size[0], 1.0 / num_grid, rounding_mode='trunc'))
                top_box = max(0, int(floordiv((center_h - pos_h_range) / upsampled_size[0], 1.0 / num_grid, rounding_mode='trunc')))
                down_box = min(num_grid - 1, int(floordiv((center_h + pos_h_range) / upsampled_size[0], 1.0 / num_grid, rounding_mode='trunc')))
                left_box = max(0, int(floordiv((center_w - pos_w_range) / upsampled_size[1], 1.0 / num_grid, rounding_mode='trunc')))
                right_box = min(num_grid - 1, int(floordiv((center_w + pos_w_range) / upsampled_size[1], 1.0 / num_grid, rounding_mode='trunc')))
                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)
                labels[top:down + 1, left:right + 1] = gt_label
                gt_mask = np.uint8(gt_mask.cpu().numpy())
                gt_mask = mmcv.imrescale(gt_mask, scale=1.0 / output_stride)
                gt_mask = torch.from_numpy(gt_mask).to(device=device)
                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        index = int(i * num_grid + j)
                        mask_target[index, :gt_mask.shape[0], :gt_mask.shape[1]] = gt_mask
                        pos_mask[index] = True
            mlvl_pos_mask_targets.append(mask_target[pos_mask])
            mlvl_labels.append(labels)
            mlvl_pos_masks.append(pos_mask)
        return (mlvl_pos_mask_targets, mlvl_labels, mlvl_pos_masks)

    def get_results(self, mlvl_mask_preds, mlvl_cls_scores, img_metas, **kwargs):
        """Get multi-image mask results.

        Args:
            mlvl_mask_preds (list[Tensor]): Multi-level mask prediction.
                Each element in the list has shape
                (batch_size, num_grids**2 ,h ,w).
            mlvl_cls_scores (list[Tensor]): Multi-level scores. Each element
                in the list has shape
                (batch_size, num_classes, num_grids ,num_grids).
            img_metas (list[dict]): Meta information of all images.

        Returns:
            list[:obj:`InstanceData`]: Processed results of multiple
            images.Each :obj:`InstanceData` usually contains
            following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """
        mlvl_cls_scores = [item.permute(0, 2, 3, 1) for item in mlvl_cls_scores]
        assert len(mlvl_mask_preds) == len(mlvl_cls_scores)
        num_levels = len(mlvl_cls_scores)
        results_list = []
        for img_id in range(len(img_metas)):
            cls_pred_list = [mlvl_cls_scores[lvl][img_id].view(-1, self.cls_out_channels) for lvl in range(num_levels)]
            mask_pred_list = [mlvl_mask_preds[lvl][img_id] for lvl in range(num_levels)]
            cls_pred_list = torch.cat(cls_pred_list, dim=0)
            mask_pred_list = torch.cat(mask_pred_list, dim=0)
            results = self._get_results_single(cls_pred_list, mask_pred_list, img_meta=img_metas[img_id])
            results_list.append(results)
        return results_list

    def _get_results_single(self, cls_scores, mask_preds, img_meta, cfg=None):
        """Get processed mask related results of single image.

        Args:
            cls_scores (Tensor): Classification score of all points
                in single image, has shape (num_points, num_classes).
            mask_preds (Tensor): Mask prediction of all points in
                single image, has shape (num_points, feat_h, feat_w).
            img_meta (dict): Meta information of corresponding image.
            cfg (dict, optional): Config used in test phase.
                Default: None.

        Returns:
            :obj:`InstanceData`: Processed results of single image.
             it usually contains following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """

        def empty_results(results, cls_scores):
            """Generate a empty results."""
            results.scores = cls_scores.new_ones(0)
            results.masks = cls_scores.new_zeros(0, *results.ori_shape[:2])
            results.labels = cls_scores.new_ones(0)
            return results
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(mask_preds)
        results = InstanceData(img_meta)
        featmap_size = mask_preds.size()[-2:]
        img_shape = results.img_shape
        ori_shape = results.ori_shape
        h, w, _ = img_shape
        upsampled_size = (featmap_size[0] * 4, featmap_size[1] * 4)
        score_mask = cls_scores > cfg.score_thr
        cls_scores = cls_scores[score_mask]
        if len(cls_scores) == 0:
            return empty_results(results, cls_scores)
        inds = score_mask.nonzero()
        cls_labels = inds[:, 1]
        lvl_interval = cls_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = cls_scores.new_ones(lvl_interval[-1])
        strides[:lvl_interval[0]] *= self.strides[0]
        for lvl in range(1, self.num_levels):
            strides[lvl_interval[lvl - 1]:lvl_interval[lvl]] *= self.strides[lvl]
        strides = strides[inds[:, 0]]
        mask_preds = mask_preds[inds[:, 0]]
        masks = mask_preds > cfg.mask_thr
        sum_masks = masks.sum((1, 2)).float()
        keep = sum_masks > strides
        if keep.sum() == 0:
            return empty_results(results, cls_scores)
        masks = masks[keep]
        mask_preds = mask_preds[keep]
        sum_masks = sum_masks[keep]
        cls_scores = cls_scores[keep]
        cls_labels = cls_labels[keep]
        mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
        cls_scores *= mask_scores
        scores, labels, _, keep_inds = mask_matrix_nms(masks, cls_labels, cls_scores, mask_area=sum_masks, nms_pre=cfg.nms_pre, max_num=cfg.max_per_img, kernel=cfg.kernel, sigma=cfg.sigma, filter_thr=cfg.filter_thr)
        mask_preds = mask_preds[keep_inds]
        mask_preds = F.interpolate(mask_preds.unsqueeze(0), size=upsampled_size, mode='bilinear')[:, :, :h, :w]
        mask_preds = F.interpolate(mask_preds, size=ori_shape[:2], mode='bilinear').squeeze(0)
        masks = mask_preds > cfg.mask_thr
        results.masks = masks
        results.labels = labels
        results.scores = scores
        return results

def generate_coordinate(featmap_sizes, device='cuda'):
    """Generate the coordinate.

    Args:
        featmap_sizes (tuple): The feature to be calculated,
            of shape (N, C, W, H).
        device (str): The device where the feature will be put on.
    Returns:
        coord_feat (Tensor): The coordinate feature, of shape (N, 2, W, H).
    """
    x_range = torch.linspace(-1, 1, featmap_sizes[-1], device=device)
    y_range = torch.linspace(-1, 1, featmap_sizes[-2], device=device)
    y, x = torch.meshgrid(y_range, x_range)
    y = y.expand([featmap_sizes[0], 1, -1, -1])
    x = x.expand([featmap_sizes[0], 1, -1, -1])
    coord_feat = torch.cat([x, y], 1)
    return coord_feat

@HEADS.register_module()
class DecoupledSOLOHead(SOLOHead):
    """Decoupled SOLO mask head used in `SOLO: Segmenting Objects by Locations.

    <https://arxiv.org/abs/1912.04488>`_

    Args:
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, *args, init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01), dict(type='Normal', std=0.01, bias_prob=0.01, override=dict(name='conv_mask_list_x')), dict(type='Normal', std=0.01, bias_prob=0.01, override=dict(name='conv_mask_list_y')), dict(type='Normal', std=0.01, bias_prob=0.01, override=dict(name='conv_cls'))], **kwargs):
        super(DecoupledSOLOHead, self).__init__(*args, init_cfg=init_cfg, **kwargs)

    def _init_layers(self):
        self.mask_convs_x = nn.ModuleList()
        self.mask_convs_y = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels + 1 if i == 0 else self.feat_channels
            self.mask_convs_x.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, norm_cfg=self.norm_cfg))
            self.mask_convs_y.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, norm_cfg=self.norm_cfg))
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, norm_cfg=self.norm_cfg))
        self.conv_mask_list_x = nn.ModuleList()
        self.conv_mask_list_y = nn.ModuleList()
        for num_grid in self.num_grids:
            self.conv_mask_list_x.append(nn.Conv2d(self.feat_channels, num_grid, 3, padding=1))
            self.conv_mask_list_y.append(nn.Conv2d(self.feat_channels, num_grid, 3, padding=1))
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)

    def forward(self, feats):
        assert len(feats) == self.num_levels
        feats = self.resize_feats(feats)
        mask_preds_x = []
        mask_preds_y = []
        cls_preds = []
        for i in range(self.num_levels):
            x = feats[i]
            mask_feat = x
            cls_feat = x
            coord_feat = generate_coordinate(mask_feat.size(), mask_feat.device)
            mask_feat_x = torch.cat([mask_feat, coord_feat[:, 0:1, ...]], 1)
            mask_feat_y = torch.cat([mask_feat, coord_feat[:, 1:2, ...]], 1)
            for mask_layer_x, mask_layer_y in zip(self.mask_convs_x, self.mask_convs_y):
                mask_feat_x = mask_layer_x(mask_feat_x)
                mask_feat_y = mask_layer_y(mask_feat_y)
            mask_feat_x = F.interpolate(mask_feat_x, scale_factor=2, mode='bilinear')
            mask_feat_y = F.interpolate(mask_feat_y, scale_factor=2, mode='bilinear')
            mask_pred_x = self.conv_mask_list_x[i](mask_feat_x)
            mask_pred_y = self.conv_mask_list_y[i](mask_feat_y)
            for j, cls_layer in enumerate(self.cls_convs):
                if j == self.cls_down_index:
                    num_grid = self.num_grids[i]
                    cls_feat = F.interpolate(cls_feat, size=num_grid, mode='bilinear')
                cls_feat = cls_layer(cls_feat)
            cls_pred = self.conv_cls(cls_feat)
            if not self.training:
                feat_wh = feats[0].size()[-2:]
                upsampled_size = (feat_wh[0] * 2, feat_wh[1] * 2)
                mask_pred_x = F.interpolate(mask_pred_x.sigmoid(), size=upsampled_size, mode='bilinear')
                mask_pred_y = F.interpolate(mask_pred_y.sigmoid(), size=upsampled_size, mode='bilinear')
                cls_pred = cls_pred.sigmoid()
                local_max = F.max_pool2d(cls_pred, 2, stride=1, padding=1)
                keep_mask = local_max[:, :, :-1, :-1] == cls_pred
                cls_pred = cls_pred * keep_mask
            mask_preds_x.append(mask_pred_x)
            mask_preds_y.append(mask_pred_y)
            cls_preds.append(cls_pred)
        return (mask_preds_x, mask_preds_y, cls_preds)

    def loss(self, mlvl_mask_preds_x, mlvl_mask_preds_y, mlvl_cls_preds, gt_labels, gt_masks, img_metas, gt_bboxes=None, **kwargs):
        """Calculate the loss of total batch.

        Args:
            mlvl_mask_preds_x (list[Tensor]): Multi-level mask prediction
                from x branch. Each element in the list has shape
                (batch_size, num_grids ,h ,w).
            mlvl_mask_preds_x (list[Tensor]): Multi-level mask prediction
                from y branch. Each element in the list has shape
                (batch_size, num_grids ,h ,w).
            mlvl_cls_preds (list[Tensor]): Multi-level scores. Each element
                in the list has shape
                (batch_size, num_classes, num_grids ,num_grids).
            gt_labels (list[Tensor]): Labels of multiple images.
            gt_masks (list[Tensor]): Ground truth masks of multiple images.
                Each has shape (num_instances, h, w).
            img_metas (list[dict]): Meta information of multiple images.
            gt_bboxes (list[Tensor]): Ground truth bboxes of multiple
                images. Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_levels = self.num_levels
        num_imgs = len(gt_labels)
        featmap_sizes = [featmap.size()[-2:] for featmap in mlvl_mask_preds_x]
        pos_mask_targets, labels, xy_pos_indexes = multi_apply(self._get_targets_single, gt_bboxes, gt_labels, gt_masks, featmap_sizes=featmap_sizes)
        mlvl_pos_mask_targets = [[] for _ in range(num_levels)]
        mlvl_pos_mask_preds_x = [[] for _ in range(num_levels)]
        mlvl_pos_mask_preds_y = [[] for _ in range(num_levels)]
        mlvl_labels = [[] for _ in range(num_levels)]
        for img_id in range(num_imgs):
            for lvl in range(num_levels):
                mlvl_pos_mask_targets[lvl].append(pos_mask_targets[img_id][lvl])
                mlvl_pos_mask_preds_x[lvl].append(mlvl_mask_preds_x[lvl][img_id, xy_pos_indexes[img_id][lvl][:, 1]])
                mlvl_pos_mask_preds_y[lvl].append(mlvl_mask_preds_y[lvl][img_id, xy_pos_indexes[img_id][lvl][:, 0]])
                mlvl_labels[lvl].append(labels[img_id][lvl].flatten())
        temp_mlvl_cls_preds = []
        for lvl in range(num_levels):
            mlvl_pos_mask_targets[lvl] = torch.cat(mlvl_pos_mask_targets[lvl], dim=0)
            mlvl_pos_mask_preds_x[lvl] = torch.cat(mlvl_pos_mask_preds_x[lvl], dim=0)
            mlvl_pos_mask_preds_y[lvl] = torch.cat(mlvl_pos_mask_preds_y[lvl], dim=0)
            mlvl_labels[lvl] = torch.cat(mlvl_labels[lvl], dim=0)
            temp_mlvl_cls_preds.append(mlvl_cls_preds[lvl].permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels))
        num_pos = 0.0
        loss_mask = []
        for pred_x, pred_y, target in zip(mlvl_pos_mask_preds_x, mlvl_pos_mask_preds_y, mlvl_pos_mask_targets):
            num_masks = pred_x.size(0)
            if num_masks == 0:
                loss_mask.append((pred_x.sum() + pred_y.sum()).unsqueeze(0))
                continue
            num_pos += num_masks
            pred_mask = pred_y.sigmoid() * pred_x.sigmoid()
            loss_mask.append(self.loss_mask(pred_mask, target, reduction_override='none'))
        if num_pos > 0:
            loss_mask = torch.cat(loss_mask).sum() / num_pos
        else:
            loss_mask = torch.cat(loss_mask).mean()
        flatten_labels = torch.cat(mlvl_labels)
        flatten_cls_preds = torch.cat(temp_mlvl_cls_preds)
        loss_cls = self.loss_cls(flatten_cls_preds, flatten_labels, avg_factor=num_pos + 1)
        return dict(loss_mask=loss_mask, loss_cls=loss_cls)

    def _get_targets_single(self, gt_bboxes, gt_labels, gt_masks, featmap_sizes=None):
        """Compute targets for predictions of single image.

        Args:
            gt_bboxes (Tensor): Ground truth bbox of each instance,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth label of each instance,
                shape (num_gts,).
            gt_masks (Tensor): Ground truth mask of each instance,
                shape (num_gts, h, w).
            featmap_sizes (list[:obj:`torch.size`]): Size of each
                feature map from feature pyramid, each element
                means (feat_h, feat_w). Default: None.

        Returns:
            Tuple: Usually returns a tuple containing targets for predictions.

                - mlvl_pos_mask_targets (list[Tensor]): Each element represent
                  the binary mask targets for positive points in this
                  level, has shape (num_pos, out_h, out_w).
                - mlvl_labels (list[Tensor]): Each element is
                  classification labels for all
                  points in this level, has shape
                  (num_grid, num_grid).
                - mlvl_xy_pos_indexes (list[Tensor]): Each element
                  in the list contains the index of positive samples in
                  corresponding level, has shape (num_pos, 2), last
                  dimension 2 present (index_x, index_y).
        """
        mlvl_pos_mask_targets, mlvl_labels, mlvl_pos_masks = super()._get_targets_single(gt_bboxes, gt_labels, gt_masks, featmap_sizes=featmap_sizes)
        mlvl_xy_pos_indexes = [(item - self.num_classes).nonzero() for item in mlvl_labels]
        return (mlvl_pos_mask_targets, mlvl_labels, mlvl_xy_pos_indexes)

    def get_results(self, mlvl_mask_preds_x, mlvl_mask_preds_y, mlvl_cls_scores, img_metas, rescale=None, **kwargs):
        """Get multi-image mask results.

        Args:
            mlvl_mask_preds_x (list[Tensor]): Multi-level mask prediction
                from x branch. Each element in the list has shape
                (batch_size, num_grids ,h ,w).
            mlvl_mask_preds_y (list[Tensor]): Multi-level mask prediction
                from y branch. Each element in the list has shape
                (batch_size, num_grids ,h ,w).
            mlvl_cls_scores (list[Tensor]): Multi-level scores. Each element
                in the list has shape
                (batch_size, num_classes ,num_grids ,num_grids).
            img_metas (list[dict]): Meta information of all images.

        Returns:
            list[:obj:`InstanceData`]: Processed results of multiple
            images.Each :obj:`InstanceData` usually contains
            following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """
        mlvl_cls_scores = [item.permute(0, 2, 3, 1) for item in mlvl_cls_scores]
        assert len(mlvl_mask_preds_x) == len(mlvl_cls_scores)
        num_levels = len(mlvl_cls_scores)
        results_list = []
        for img_id in range(len(img_metas)):
            cls_pred_list = [mlvl_cls_scores[i][img_id].view(-1, self.cls_out_channels).detach() for i in range(num_levels)]
            mask_pred_list_x = [mlvl_mask_preds_x[i][img_id] for i in range(num_levels)]
            mask_pred_list_y = [mlvl_mask_preds_y[i][img_id] for i in range(num_levels)]
            cls_pred_list = torch.cat(cls_pred_list, dim=0)
            mask_pred_list_x = torch.cat(mask_pred_list_x, dim=0)
            mask_pred_list_y = torch.cat(mask_pred_list_y, dim=0)
            results = self._get_results_single(cls_pred_list, mask_pred_list_x, mask_pred_list_y, img_meta=img_metas[img_id], cfg=self.test_cfg)
            results_list.append(results)
        return results_list

    def _get_results_single(self, cls_scores, mask_preds_x, mask_preds_y, img_meta, cfg):
        """Get processed mask related results of single image.

        Args:
            cls_scores (Tensor): Classification score of all points
                in single image, has shape (num_points, num_classes).
            mask_preds_x (Tensor): Mask prediction of x branch of
                all points in single image, has shape
                (sum_num_grids, feat_h, feat_w).
            mask_preds_y (Tensor): Mask prediction of y branch of
                all points in single image, has shape
                (sum_num_grids, feat_h, feat_w).
            img_meta (dict): Meta information of corresponding image.
            cfg (dict): Config used in test phase.

        Returns:
            :obj:`InstanceData`: Processed results of single image.
             it usually contains following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """

        def empty_results(results, cls_scores):
            """Generate a empty results."""
            results.scores = cls_scores.new_ones(0)
            results.masks = cls_scores.new_zeros(0, *results.ori_shape[:2])
            results.labels = cls_scores.new_ones(0)
            return results
        cfg = self.test_cfg if cfg is None else cfg
        results = InstanceData(img_meta)
        img_shape = results.img_shape
        ori_shape = results.ori_shape
        h, w, _ = img_shape
        featmap_size = mask_preds_x.size()[-2:]
        upsampled_size = (featmap_size[0] * 4, featmap_size[1] * 4)
        score_mask = cls_scores > cfg.score_thr
        cls_scores = cls_scores[score_mask]
        inds = score_mask.nonzero()
        lvl_interval = inds.new_tensor(self.num_grids).pow(2).cumsum(0)
        num_all_points = lvl_interval[-1]
        lvl_start_index = inds.new_ones(num_all_points)
        num_grids = inds.new_ones(num_all_points)
        seg_size = inds.new_tensor(self.num_grids).cumsum(0)
        mask_lvl_start_index = inds.new_ones(num_all_points)
        strides = inds.new_ones(num_all_points)
        lvl_start_index[:lvl_interval[0]] *= 0
        mask_lvl_start_index[:lvl_interval[0]] *= 0
        num_grids[:lvl_interval[0]] *= self.num_grids[0]
        strides[:lvl_interval[0]] *= self.strides[0]
        for lvl in range(1, self.num_levels):
            lvl_start_index[lvl_interval[lvl - 1]:lvl_interval[lvl]] *= lvl_interval[lvl - 1]
            mask_lvl_start_index[lvl_interval[lvl - 1]:lvl_interval[lvl]] *= seg_size[lvl - 1]
            num_grids[lvl_interval[lvl - 1]:lvl_interval[lvl]] *= self.num_grids[lvl]
            strides[lvl_interval[lvl - 1]:lvl_interval[lvl]] *= self.strides[lvl]
        lvl_start_index = lvl_start_index[inds[:, 0]]
        mask_lvl_start_index = mask_lvl_start_index[inds[:, 0]]
        num_grids = num_grids[inds[:, 0]]
        strides = strides[inds[:, 0]]
        y_lvl_offset = (inds[:, 0] - lvl_start_index) // num_grids
        x_lvl_offset = (inds[:, 0] - lvl_start_index) % num_grids
        y_inds = mask_lvl_start_index + y_lvl_offset
        x_inds = mask_lvl_start_index + x_lvl_offset
        cls_labels = inds[:, 1]
        mask_preds = mask_preds_x[x_inds, ...] * mask_preds_y[y_inds, ...]
        masks = mask_preds > cfg.mask_thr
        sum_masks = masks.sum((1, 2)).float()
        keep = sum_masks > strides
        if keep.sum() == 0:
            return empty_results(results, cls_scores)
        masks = masks[keep]
        mask_preds = mask_preds[keep]
        sum_masks = sum_masks[keep]
        cls_scores = cls_scores[keep]
        cls_labels = cls_labels[keep]
        mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
        cls_scores *= mask_scores
        scores, labels, _, keep_inds = mask_matrix_nms(masks, cls_labels, cls_scores, mask_area=sum_masks, nms_pre=cfg.nms_pre, max_num=cfg.max_per_img, kernel=cfg.kernel, sigma=cfg.sigma, filter_thr=cfg.filter_thr)
        mask_preds = mask_preds[keep_inds]
        mask_preds = F.interpolate(mask_preds.unsqueeze(0), size=upsampled_size, mode='bilinear')[:, :, :h, :w]
        mask_preds = F.interpolate(mask_preds, size=ori_shape[:2], mode='bilinear').squeeze(0)
        masks = mask_preds > cfg.mask_thr
        results.masks = masks
        results.labels = labels
        results.scores = scores
        return results

@HEADS.register_module()
class DecoupledSOLOLightHead(DecoupledSOLOHead):
    """Decoupled Light SOLO mask head used in `SOLO: Segmenting Objects by
    Locations <https://arxiv.org/abs/1912.04488>`_

    Args:
        with_dcn (bool): Whether use dcn in mask_convs and cls_convs,
            default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, *args, dcn_cfg=None, init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01), dict(type='Normal', std=0.01, bias_prob=0.01, override=dict(name='conv_mask_list_x')), dict(type='Normal', std=0.01, bias_prob=0.01, override=dict(name='conv_mask_list_y')), dict(type='Normal', std=0.01, bias_prob=0.01, override=dict(name='conv_cls'))], **kwargs):
        assert dcn_cfg is None or isinstance(dcn_cfg, dict)
        self.dcn_cfg = dcn_cfg
        super(DecoupledSOLOLightHead, self).__init__(*args, init_cfg=init_cfg, **kwargs)

    def _init_layers(self):
        self.mask_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            if self.dcn_cfg is not None and i == self.stacked_convs - 1:
                conv_cfg = self.dcn_cfg
            else:
                conv_cfg = None
            chn = self.in_channels + 2 if i == 0 else self.feat_channels
            self.mask_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=self.norm_cfg))
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=self.norm_cfg))
        self.conv_mask_list_x = nn.ModuleList()
        self.conv_mask_list_y = nn.ModuleList()
        for num_grid in self.num_grids:
            self.conv_mask_list_x.append(nn.Conv2d(self.feat_channels, num_grid, 3, padding=1))
            self.conv_mask_list_y.append(nn.Conv2d(self.feat_channels, num_grid, 3, padding=1))
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)

    def forward(self, feats):
        assert len(feats) == self.num_levels
        feats = self.resize_feats(feats)
        mask_preds_x = []
        mask_preds_y = []
        cls_preds = []
        for i in range(self.num_levels):
            x = feats[i]
            mask_feat = x
            cls_feat = x
            coord_feat = generate_coordinate(mask_feat.size(), mask_feat.device)
            mask_feat = torch.cat([mask_feat, coord_feat], 1)
            for mask_layer in self.mask_convs:
                mask_feat = mask_layer(mask_feat)
            mask_feat = F.interpolate(mask_feat, scale_factor=2, mode='bilinear')
            mask_pred_x = self.conv_mask_list_x[i](mask_feat)
            mask_pred_y = self.conv_mask_list_y[i](mask_feat)
            for j, cls_layer in enumerate(self.cls_convs):
                if j == self.cls_down_index:
                    num_grid = self.num_grids[i]
                    cls_feat = F.interpolate(cls_feat, size=num_grid, mode='bilinear')
                cls_feat = cls_layer(cls_feat)
            cls_pred = self.conv_cls(cls_feat)
            if not self.training:
                feat_wh = feats[0].size()[-2:]
                upsampled_size = (feat_wh[0] * 2, feat_wh[1] * 2)
                mask_pred_x = F.interpolate(mask_pred_x.sigmoid(), size=upsampled_size, mode='bilinear')
                mask_pred_y = F.interpolate(mask_pred_y.sigmoid(), size=upsampled_size, mode='bilinear')
                cls_pred = cls_pred.sigmoid()
                local_max = F.max_pool2d(cls_pred, 2, stride=1, padding=1)
                keep_mask = local_max[:, :, :-1, :-1] == cls_pred
                cls_pred = cls_pred * keep_mask
            mask_preds_x.append(mask_pred_x)
            mask_preds_y.append(mask_pred_y)
            cls_preds.append(cls_pred)
        return (mask_preds_x, mask_preds_y, cls_preds)

class MaskFeatModule(BaseModule):
    """SOLOv2 mask feature map branch used in `SOLOv2: Dynamic and Fast
    Instance Segmentation. <https://arxiv.org/pdf/2003.10152>`_

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels of the mask feature
             map branch.
        start_level (int): The starting feature map level from RPN that
             will be used to predict the mask feature map.
        end_level (int): The ending feature map level from rpn that
             will be used to predict the mask feature map.
        out_channels (int): Number of output channels of the mask feature
             map branch. This is the channel count of the mask
             feature map that to be dynamically convolved with the predicted
             kernel.
        mask_stride (int): Downsample factor of the mask feature map output.
            Default: 4.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, in_channels, feat_channels, start_level, end_level, out_channels, mask_stride=4, conv_cfg=None, norm_cfg=None, init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01)]):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.start_level = start_level
        self.end_level = end_level
        self.mask_stride = mask_stride
        assert start_level >= 0 and end_level >= start_level
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()
        self.fp16_enabled = False

    def _init_layers(self):
        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            if i == 0:
                convs_per_level.add_module(f'conv{i}', ConvModule(self.in_channels, self.feat_channels, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, inplace=False))
                self.convs_all_levels.append(convs_per_level)
                continue
            for j in range(i):
                if j == 0:
                    if i == self.end_level:
                        chn = self.in_channels + 2
                    else:
                        chn = self.in_channels
                    convs_per_level.add_module(f'conv{j}', ConvModule(chn, self.feat_channels, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, inplace=False))
                    convs_per_level.add_module(f'upsample{j}', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
                    continue
                convs_per_level.add_module(f'conv{j}', ConvModule(self.feat_channels, self.feat_channels, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, inplace=False))
                convs_per_level.add_module(f'upsample{j}', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            self.convs_all_levels.append(convs_per_level)
        self.conv_pred = ConvModule(self.feat_channels, self.out_channels, 1, padding=0, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)

    @auto_fp16()
    def forward(self, feats):
        inputs = feats[self.start_level:self.end_level + 1]
        assert len(inputs) == self.end_level - self.start_level + 1
        feature_add_all_level = self.convs_all_levels[0](inputs[0])
        for i in range(1, len(inputs)):
            input_p = inputs[i]
            if i == len(inputs) - 1:
                coord_feat = generate_coordinate(input_p.size(), input_p.device)
                input_p = torch.cat([input_p, coord_feat], 1)
            feature_add_all_level = feature_add_all_level + self.convs_all_levels[i](input_p)
        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred

@HEADS.register_module()
class SOLOV2Head(SOLOHead):
    """SOLOv2 mask head used in `SOLOv2: Dynamic and Fast Instance
    Segmentation. <https://arxiv.org/pdf/2003.10152>`_

    Args:
        mask_feature_head (dict): Config of SOLOv2MaskFeatHead.
        dynamic_conv_size (int): Dynamic Conv kernel size. Default: 1.
        dcn_cfg (dict): Dcn conv configurations in kernel_convs and cls_conv.
            default: None.
        dcn_apply_to_all_conv (bool): Whether to use dcn in every layer of
            kernel_convs and cls_convs, or only the last layer. It shall be set
            `True` for the normal version of SOLOv2 and `False` for the
            light-weight version. default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, *args, mask_feature_head, dynamic_conv_size=1, dcn_cfg=None, dcn_apply_to_all_conv=True, init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01), dict(type='Normal', std=0.01, bias_prob=0.01, override=dict(name='conv_cls'))], **kwargs):
        assert dcn_cfg is None or isinstance(dcn_cfg, dict)
        self.dcn_cfg = dcn_cfg
        self.with_dcn = dcn_cfg is not None
        self.dcn_apply_to_all_conv = dcn_apply_to_all_conv
        self.dynamic_conv_size = dynamic_conv_size
        mask_out_channels = mask_feature_head.get('out_channels')
        self.kernel_out_channels = mask_out_channels * self.dynamic_conv_size * self.dynamic_conv_size
        super().__init__(*args, init_cfg=init_cfg, **kwargs)
        if mask_feature_head.get('in_channels', None) is not None:
            if mask_feature_head.in_channels != self.in_channels:
                warnings.warn(f'The `in_channels` of SOLOv2MaskFeatHead and SOLOv2Head should be same, changing mask_feature_head.in_channels to {self.in_channels}')
                mask_feature_head.update(in_channels=self.in_channels)
        else:
            mask_feature_head.update(in_channels=self.in_channels)
        self.mask_feature_head = MaskFeatModule(**mask_feature_head)
        self.mask_stride = self.mask_feature_head.mask_stride
        self.fp16_enabled = False

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        conv_cfg = None
        for i in range(self.stacked_convs):
            if self.with_dcn:
                if self.dcn_apply_to_all_conv:
                    conv_cfg = self.dcn_cfg
                elif i == self.stacked_convs - 1:
                    conv_cfg = self.dcn_cfg
            chn = self.in_channels + 2 if i == 0 else self.feat_channels
            self.kernel_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None))
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None))
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_kernel = nn.Conv2d(self.feat_channels, self.kernel_out_channels, 3, padding=1)

    @auto_fp16()
    def forward(self, feats):
        assert len(feats) == self.num_levels
        mask_feats = self.mask_feature_head(feats)
        feats = self.resize_feats(feats)
        mlvl_kernel_preds = []
        mlvl_cls_preds = []
        for i in range(self.num_levels):
            ins_kernel_feat = feats[i]
            coord_feat = generate_coordinate(ins_kernel_feat.size(), ins_kernel_feat.device)
            ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)
            kernel_feat = ins_kernel_feat
            kernel_feat = F.interpolate(kernel_feat, size=self.num_grids[i], mode='bilinear', align_corners=False)
            cate_feat = kernel_feat[:, :-2, :, :]
            kernel_feat = kernel_feat.contiguous()
            for i, kernel_conv in enumerate(self.kernel_convs):
                kernel_feat = kernel_conv(kernel_feat)
            kernel_pred = self.conv_kernel(kernel_feat)
            cate_feat = cate_feat.contiguous()
            for i, cls_conv in enumerate(self.cls_convs):
                cate_feat = cls_conv(cate_feat)
            cate_pred = self.conv_cls(cate_feat)
            mlvl_kernel_preds.append(kernel_pred)
            mlvl_cls_preds.append(cate_pred)
        return (mlvl_kernel_preds, mlvl_cls_preds, mask_feats)

    def _get_targets_single(self, gt_bboxes, gt_labels, gt_masks, featmap_size=None):
        """Compute targets for predictions of single image.

        Args:
            gt_bboxes (Tensor): Ground truth bbox of each instance,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth label of each instance,
                shape (num_gts,).
            gt_masks (Tensor): Ground truth mask of each instance,
                shape (num_gts, h, w).
            featmap_sizes (:obj:`torch.size`): Size of UNified mask
                feature map used to generate instance segmentation
                masks by dynamic convolution, each element means
                (feat_h, feat_w). Default: None.

        Returns:
            Tuple: Usually returns a tuple containing targets for predictions.

                - mlvl_pos_mask_targets (list[Tensor]): Each element represent
                  the binary mask targets for positive points in this
                  level, has shape (num_pos, out_h, out_w).
                - mlvl_labels (list[Tensor]): Each element is
                  classification labels for all
                  points in this level, has shape
                  (num_grid, num_grid).
                - mlvl_pos_masks  (list[Tensor]): Each element is
                  a `BoolTensor` to represent whether the
                  corresponding point in single level
                  is positive, has shape (num_grid **2).
                - mlvl_pos_indexes  (list[list]): Each element
                  in the list contains the positive index in
                  corresponding level, has shape (num_pos).
        """
        device = gt_labels.device
        gt_areas = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1]))
        mlvl_pos_mask_targets = []
        mlvl_pos_indexes = []
        mlvl_labels = []
        mlvl_pos_masks = []
        for (lower_bound, upper_bound), num_grid in zip(self.scale_ranges, self.num_grids):
            mask_target = []
            pos_index = []
            labels = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device) + self.num_classes
            pos_mask = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)
            gt_inds = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(gt_inds) == 0:
                mlvl_pos_mask_targets.append(torch.zeros([0, featmap_size[0], featmap_size[1]], dtype=torch.uint8, device=device))
                mlvl_labels.append(labels)
                mlvl_pos_masks.append(pos_mask)
                mlvl_pos_indexes.append([])
                continue
            hit_gt_bboxes = gt_bboxes[gt_inds]
            hit_gt_labels = gt_labels[gt_inds]
            hit_gt_masks = gt_masks[gt_inds, ...]
            pos_w_ranges = 0.5 * (hit_gt_bboxes[:, 2] - hit_gt_bboxes[:, 0]) * self.pos_scale
            pos_h_ranges = 0.5 * (hit_gt_bboxes[:, 3] - hit_gt_bboxes[:, 1]) * self.pos_scale
            valid_mask_flags = hit_gt_masks.sum(dim=-1).sum(dim=-1) > 0
            for gt_mask, gt_label, pos_h_range, pos_w_range, valid_mask_flag in zip(hit_gt_masks, hit_gt_labels, pos_h_ranges, pos_w_ranges, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (featmap_size[0] * self.mask_stride, featmap_size[1] * self.mask_stride)
                center_h, center_w = center_of_mass(gt_mask)
                coord_w = int(floordiv(center_w / upsampled_size[1], 1.0 / num_grid, rounding_mode='trunc'))
                coord_h = int(floordiv(center_h / upsampled_size[0], 1.0 / num_grid, rounding_mode='trunc'))
                top_box = max(0, int(floordiv((center_h - pos_h_range) / upsampled_size[0], 1.0 / num_grid, rounding_mode='trunc')))
                down_box = min(num_grid - 1, int(floordiv((center_h + pos_h_range) / upsampled_size[0], 1.0 / num_grid, rounding_mode='trunc')))
                left_box = max(0, int(floordiv((center_w - pos_w_range) / upsampled_size[1], 1.0 / num_grid, rounding_mode='trunc')))
                right_box = min(num_grid - 1, int(floordiv((center_w + pos_w_range) / upsampled_size[1], 1.0 / num_grid, rounding_mode='trunc')))
                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)
                labels[top:down + 1, left:right + 1] = gt_label
                gt_mask = np.uint8(gt_mask.cpu().numpy())
                gt_mask = mmcv.imrescale(gt_mask, scale=1.0 / self.mask_stride)
                gt_mask = torch.from_numpy(gt_mask).to(device=device)
                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        index = int(i * num_grid + j)
                        this_mask_target = torch.zeros([featmap_size[0], featmap_size[1]], dtype=torch.uint8, device=device)
                        this_mask_target[:gt_mask.shape[0], :gt_mask.shape[1]] = gt_mask
                        mask_target.append(this_mask_target)
                        pos_mask[index] = True
                        pos_index.append(index)
            if len(mask_target) == 0:
                mask_target = torch.zeros([0, featmap_size[0], featmap_size[1]], dtype=torch.uint8, device=device)
            else:
                mask_target = torch.stack(mask_target, 0)
            mlvl_pos_mask_targets.append(mask_target)
            mlvl_labels.append(labels)
            mlvl_pos_masks.append(pos_mask)
            mlvl_pos_indexes.append(pos_index)
        return (mlvl_pos_mask_targets, mlvl_labels, mlvl_pos_masks, mlvl_pos_indexes)

    @force_fp32(apply_to=('mlvl_kernel_preds', 'mlvl_cls_preds', 'mask_feats'))
    def loss(self, mlvl_kernel_preds, mlvl_cls_preds, mask_feats, gt_labels, gt_masks, img_metas, gt_bboxes=None, **kwargs):
        """Calculate the loss of total batch.

        Args:
            mlvl_kernel_preds (list[Tensor]): Multi-level dynamic kernel
                prediction. The kernel is used to generate instance
                segmentation masks by dynamic convolution. Each element in the
                list has shape
                (batch_size, kernel_out_channels, num_grids, num_grids).
            mlvl_cls_preds (list[Tensor]): Multi-level scores. Each element
                in the list has shape
                (batch_size, num_classes, num_grids, num_grids).
            mask_feats (Tensor): Unified mask feature map used to generate
                instance segmentation masks by dynamic convolution. Has shape
                (batch_size, mask_out_channels, h, w).
            gt_labels (list[Tensor]): Labels of multiple images.
            gt_masks (list[Tensor]): Ground truth masks of multiple images.
                Each has shape (num_instances, h, w).
            img_metas (list[dict]): Meta information of multiple images.
            gt_bboxes (list[Tensor]): Ground truth bboxes of multiple
                images. Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_size = mask_feats.size()[-2:]
        pos_mask_targets, labels, pos_masks, pos_indexes = multi_apply(self._get_targets_single, gt_bboxes, gt_labels, gt_masks, featmap_size=featmap_size)
        mlvl_mask_targets = [torch.cat(lvl_mask_targets, 0) for lvl_mask_targets in zip(*pos_mask_targets)]
        mlvl_pos_kernel_preds = []
        for lvl_kernel_preds, lvl_pos_indexes in zip(mlvl_kernel_preds, zip(*pos_indexes)):
            lvl_pos_kernel_preds = []
            for img_lvl_kernel_preds, img_lvl_pos_indexes in zip(lvl_kernel_preds, lvl_pos_indexes):
                img_lvl_pos_kernel_preds = img_lvl_kernel_preds.view(img_lvl_kernel_preds.shape[0], -1)[:, img_lvl_pos_indexes]
                lvl_pos_kernel_preds.append(img_lvl_pos_kernel_preds)
            mlvl_pos_kernel_preds.append(lvl_pos_kernel_preds)
        mlvl_mask_preds = []
        for lvl_pos_kernel_preds in mlvl_pos_kernel_preds:
            lvl_mask_preds = []
            for img_id, img_lvl_pos_kernel_pred in enumerate(lvl_pos_kernel_preds):
                if img_lvl_pos_kernel_pred.size()[-1] == 0:
                    continue
                img_mask_feats = mask_feats[[img_id]]
                h, w = img_mask_feats.shape[-2:]
                num_kernel = img_lvl_pos_kernel_pred.shape[1]
                img_lvl_mask_pred = F.conv2d(img_mask_feats, img_lvl_pos_kernel_pred.permute(1, 0).view(num_kernel, -1, self.dynamic_conv_size, self.dynamic_conv_size), stride=1).view(-1, h, w)
                lvl_mask_preds.append(img_lvl_mask_pred)
            if len(lvl_mask_preds) == 0:
                lvl_mask_preds = None
            else:
                lvl_mask_preds = torch.cat(lvl_mask_preds, 0)
            mlvl_mask_preds.append(lvl_mask_preds)
        num_pos = 0
        for img_pos_masks in pos_masks:
            for lvl_img_pos_masks in img_pos_masks:
                num_pos += lvl_img_pos_masks.count_nonzero()
        loss_mask = []
        for lvl_mask_preds, lvl_mask_targets in zip(mlvl_mask_preds, mlvl_mask_targets):
            if lvl_mask_preds is None:
                continue
            loss_mask.append(self.loss_mask(lvl_mask_preds, lvl_mask_targets, reduction_override='none'))
        if num_pos > 0:
            loss_mask = torch.cat(loss_mask).sum() / num_pos
        else:
            loss_mask = mask_feats.sum() * 0
        flatten_labels = [torch.cat([img_lvl_labels.flatten() for img_lvl_labels in lvl_labels]) for lvl_labels in zip(*labels)]
        flatten_labels = torch.cat(flatten_labels)
        flatten_cls_preds = [lvl_cls_preds.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for lvl_cls_preds in mlvl_cls_preds]
        flatten_cls_preds = torch.cat(flatten_cls_preds)
        loss_cls = self.loss_cls(flatten_cls_preds, flatten_labels, avg_factor=num_pos + 1)
        return dict(loss_mask=loss_mask, loss_cls=loss_cls)

    @force_fp32(apply_to=('mlvl_kernel_preds', 'mlvl_cls_scores', 'mask_feats'))
    def get_results(self, mlvl_kernel_preds, mlvl_cls_scores, mask_feats, img_metas, **kwargs):
        """Get multi-image mask results.

        Args:
            mlvl_kernel_preds (list[Tensor]): Multi-level dynamic kernel
                prediction. The kernel is used to generate instance
                segmentation masks by dynamic convolution. Each element in the
                list has shape
                (batch_size, kernel_out_channels, num_grids, num_grids).
            mlvl_cls_scores (list[Tensor]): Multi-level scores. Each element
                in the list has shape
                (batch_size, num_classes, num_grids, num_grids).
            mask_feats (Tensor): Unified mask feature map used to generate
                instance segmentation masks by dynamic convolution. Has shape
                (batch_size, mask_out_channels, h, w).
            img_metas (list[dict]): Meta information of all images.

        Returns:
            list[:obj:`InstanceData`]: Processed results of multiple
            images.Each :obj:`InstanceData` usually contains
            following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """
        num_levels = len(mlvl_cls_scores)
        assert len(mlvl_kernel_preds) == len(mlvl_cls_scores)
        for lvl in range(num_levels):
            cls_scores = mlvl_cls_scores[lvl]
            cls_scores = cls_scores.sigmoid()
            local_max = F.max_pool2d(cls_scores, 2, stride=1, padding=1)
            keep_mask = local_max[:, :, :-1, :-1] == cls_scores
            cls_scores = cls_scores * keep_mask
            mlvl_cls_scores[lvl] = cls_scores.permute(0, 2, 3, 1)
        result_list = []
        for img_id in range(len(img_metas)):
            img_cls_pred = [mlvl_cls_scores[lvl][img_id].view(-1, self.cls_out_channels) for lvl in range(num_levels)]
            img_mask_feats = mask_feats[[img_id]]
            img_kernel_pred = [mlvl_kernel_preds[lvl][img_id].permute(1, 2, 0).view(-1, self.kernel_out_channels) for lvl in range(num_levels)]
            img_cls_pred = torch.cat(img_cls_pred, dim=0)
            img_kernel_pred = torch.cat(img_kernel_pred, dim=0)
            result = self._get_results_single(img_kernel_pred, img_cls_pred, img_mask_feats, img_meta=img_metas[img_id])
            result_list.append(result)
        return result_list

    def _get_results_single(self, kernel_preds, cls_scores, mask_feats, img_meta, cfg=None):
        """Get processed mask related results of single image.

        Args:
            kernel_preds (Tensor): Dynamic kernel prediction of all points
                in single image, has shape
                (num_points, kernel_out_channels).
            cls_scores (Tensor): Classification score of all points
                in single image, has shape (num_points, num_classes).
            mask_preds (Tensor): Mask prediction of all points in
                single image, has shape (num_points, feat_h, feat_w).
            img_meta (dict): Meta information of corresponding image.
            cfg (dict, optional): Config used in test phase.
                Default: None.

        Returns:
            :obj:`InstanceData`: Processed results of single image.
             it usually contains following keys.
                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """

        def empty_results(results, cls_scores):
            """Generate a empty results."""
            results.scores = cls_scores.new_ones(0)
            results.masks = cls_scores.new_zeros(0, *results.ori_shape[:2])
            results.labels = cls_scores.new_ones(0)
            return results
        cfg = self.test_cfg if cfg is None else cfg
        assert len(kernel_preds) == len(cls_scores)
        results = InstanceData(img_meta)
        featmap_size = mask_feats.size()[-2:]
        img_shape = results.img_shape
        ori_shape = results.ori_shape
        h, w, _ = img_shape
        upsampled_size = (featmap_size[0] * self.mask_stride, featmap_size[1] * self.mask_stride)
        score_mask = cls_scores > cfg.score_thr
        cls_scores = cls_scores[score_mask]
        if len(cls_scores) == 0:
            return empty_results(results, cls_scores)
        inds = score_mask.nonzero()
        cls_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]
        lvl_interval = cls_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(lvl_interval[-1])
        strides[:lvl_interval[0]] *= self.strides[0]
        for lvl in range(1, self.num_levels):
            strides[lvl_interval[lvl - 1]:lvl_interval[lvl]] *= self.strides[lvl]
        strides = strides[inds[:, 0]]
        kernel_preds = kernel_preds.view(kernel_preds.size(0), -1, self.dynamic_conv_size, self.dynamic_conv_size)
        mask_preds = F.conv2d(mask_feats, kernel_preds, stride=1).squeeze(0).sigmoid()
        masks = mask_preds > cfg.mask_thr
        sum_masks = masks.sum((1, 2)).float()
        keep = sum_masks > strides
        if keep.sum() == 0:
            return empty_results(results, cls_scores)
        masks = masks[keep]
        mask_preds = mask_preds[keep]
        sum_masks = sum_masks[keep]
        cls_scores = cls_scores[keep]
        cls_labels = cls_labels[keep]
        mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
        cls_scores *= mask_scores
        scores, labels, _, keep_inds = mask_matrix_nms(masks, cls_labels, cls_scores, mask_area=sum_masks, nms_pre=cfg.nms_pre, max_num=cfg.max_per_img, kernel=cfg.kernel, sigma=cfg.sigma, filter_thr=cfg.filter_thr)
        mask_preds = mask_preds[keep_inds]
        mask_preds = F.interpolate(mask_preds.unsqueeze(0), size=upsampled_size, mode='bilinear', align_corners=False)[:, :, :h, :w]
        mask_preds = F.interpolate(mask_preds, size=ori_shape[:2], mode='bilinear', align_corners=False).squeeze(0)
        masks = mask_preds > cfg.mask_thr
        results.masks = masks
        results.labels = labels
        results.scores = scores
        return results

