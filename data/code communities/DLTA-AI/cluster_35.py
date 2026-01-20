# Cluster 35

@BBOX_ASSIGNERS.register_module()
class RegionAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        center_ratio: ratio of the region in the center of the bbox to
            define positive sample.
        ignore_ratio: ratio of the region to define ignore samples.
    """

    def __init__(self, center_ratio=0.2, ignore_ratio=0.5):
        self.center_ratio = center_ratio
        self.ignore_ratio = ignore_ratio

    def assign(self, mlvl_anchors, mlvl_valid_flags, gt_bboxes, img_meta, featmap_sizes, anchor_scale, anchor_strides, gt_bboxes_ignore=None, gt_labels=None, allowed_border=0):
        """Assign gt to anchors.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.

        The assignment is done in following steps, and the order matters.

        1. Assign every anchor to 0 (negative)
        2. (For each gt_bboxes) Compute ignore flags based on ignore_region
           then assign -1 to anchors w.r.t. ignore flags
        3. (For each gt_bboxes) Compute pos flags based on center_region then
           assign gt_bboxes to anchors w.r.t. pos flags
        4. (For each gt_bboxes) Compute ignore flags based on adjacent anchor
           level then assign -1 to anchors w.r.t. ignore flags
        5. Assign anchor outside of image to -1

        Args:
            mlvl_anchors (list[Tensor]): Multi level anchors.
            mlvl_valid_flags (list[Tensor]): Multi level valid flags.
            gt_bboxes (Tensor): Ground truth bboxes of image
            img_meta (dict): Meta info of image.
            featmap_sizes (list[Tensor]): Feature mapsize each level
            anchor_scale (int): Scale of the anchor.
            anchor_strides (list[int]): Stride of the anchor.
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).
            allowed_border (int, optional): The border to allow the valid
                anchor. Defaults to 0.

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if gt_bboxes_ignore is not None:
            raise NotImplementedError
        num_gts = gt_bboxes.shape[0]
        num_bboxes = sum((x.shape[0] for x in mlvl_anchors))
        if num_gts == 0 or num_bboxes == 0:
            max_overlaps = gt_bboxes.new_zeros((num_bboxes,))
            assigned_gt_inds = gt_bboxes.new_zeros((num_bboxes,), dtype=torch.long)
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = gt_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        num_lvls = len(mlvl_anchors)
        r1 = (1 - self.center_ratio) / 2
        r2 = (1 - self.ignore_ratio) / 2
        scale = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1]))
        min_anchor_size = scale.new_full((1,), float(anchor_scale * anchor_strides[0]))
        target_lvls = torch.floor(torch.log2(scale) - torch.log2(min_anchor_size) + 0.5)
        target_lvls = target_lvls.clamp(min=0, max=num_lvls - 1).long()
        mlvl_assigned_gt_inds = []
        mlvl_ignore_flags = []
        for lvl in range(num_lvls):
            h, w = featmap_sizes[lvl]
            assert h * w == mlvl_anchors[lvl].shape[0]
            assigned_gt_inds = gt_bboxes.new_full((h * w,), 0, dtype=torch.long)
            ignore_flags = torch.zeros_like(assigned_gt_inds)
            mlvl_assigned_gt_inds.append(assigned_gt_inds)
            mlvl_ignore_flags.append(ignore_flags)
        for gt_id in range(num_gts):
            lvl = target_lvls[gt_id].item()
            featmap_size = featmap_sizes[lvl]
            stride = anchor_strides[lvl]
            anchors = mlvl_anchors[lvl]
            gt_bbox = gt_bboxes[gt_id, :4]
            ignore_region = calc_region(gt_bbox, r2, stride, featmap_size)
            ctr_region = calc_region(gt_bbox, r1, stride, featmap_size)
            ignore_flags = anchor_ctr_inside_region_flags(anchors, stride, ignore_region)
            mlvl_assigned_gt_inds[lvl][ignore_flags] = -1
            pos_flags = anchor_ctr_inside_region_flags(anchors, stride, ctr_region)
            mlvl_assigned_gt_inds[lvl][pos_flags] = gt_id + 1
            if lvl > 0:
                d_lvl = lvl - 1
                d_anchors = mlvl_anchors[d_lvl]
                d_featmap_size = featmap_sizes[d_lvl]
                d_stride = anchor_strides[d_lvl]
                d_ignore_region = calc_region(gt_bbox, r2, d_stride, d_featmap_size)
                ignore_flags = anchor_ctr_inside_region_flags(d_anchors, d_stride, d_ignore_region)
                mlvl_ignore_flags[d_lvl][ignore_flags] = 1
            if lvl < num_lvls - 1:
                u_lvl = lvl + 1
                u_anchors = mlvl_anchors[u_lvl]
                u_featmap_size = featmap_sizes[u_lvl]
                u_stride = anchor_strides[u_lvl]
                u_ignore_region = calc_region(gt_bbox, r2, u_stride, u_featmap_size)
                ignore_flags = anchor_ctr_inside_region_flags(u_anchors, u_stride, u_ignore_region)
                mlvl_ignore_flags[u_lvl][ignore_flags] = 1
        for lvl in range(num_lvls):
            ignore_flags = mlvl_ignore_flags[lvl]
            mlvl_assigned_gt_inds[lvl][ignore_flags] = -1
        flat_assigned_gt_inds = torch.cat(mlvl_assigned_gt_inds)
        flat_anchors = torch.cat(mlvl_anchors)
        flat_valid_flags = torch.cat(mlvl_valid_flags)
        assert flat_assigned_gt_inds.shape[0] == flat_anchors.shape[0] == flat_valid_flags.shape[0]
        inside_flags = anchor_inside_flags(flat_anchors, flat_valid_flags, img_meta['img_shape'], allowed_border)
        outside_flags = ~inside_flags
        flat_assigned_gt_inds[outside_flags] = -1
        if gt_labels is not None:
            assigned_labels = torch.zeros_like(flat_assigned_gt_inds)
            pos_flags = assigned_gt_inds > 0
            assigned_labels[pos_flags] = gt_labels[flat_assigned_gt_inds[pos_flags] - 1]
        else:
            assigned_labels = None
        return AssignResult(num_gts, flat_assigned_gt_inds, None, labels=assigned_labels)

def calc_region(bbox, ratio, featmap_size=None):
    """Calculate a proportional bbox region.

    The bbox center are fixed and the new h' and w' is h * ratio and w * ratio.

    Args:
        bbox (Tensor): Bboxes to calculate regions, shape (n, 4).
        ratio (float): Ratio of the output region.
        featmap_size (tuple): Feature map size used for clipping the boundary.

    Returns:
        tuple: x1, y1, x2, y2
    """
    x1 = torch.round((1 - ratio) * bbox[0] + ratio * bbox[2]).long()
    y1 = torch.round((1 - ratio) * bbox[1] + ratio * bbox[3]).long()
    x2 = torch.round(ratio * bbox[0] + (1 - ratio) * bbox[2]).long()
    y2 = torch.round(ratio * bbox[1] + (1 - ratio) * bbox[3]).long()
    if featmap_size is not None:
        x1 = x1.clamp(min=0, max=featmap_size[1])
        y1 = y1.clamp(min=0, max=featmap_size[0])
        x2 = x2.clamp(min=0, max=featmap_size[1])
        y2 = y2.clamp(min=0, max=featmap_size[0])
    return (x1, y1, x2, y2)

def anchor_ctr_inside_region_flags(anchors, stride, region):
    """Get the flag indicate whether anchor centers are inside regions."""
    x1, y1, x2, y2 = region
    f_anchors = anchors / stride
    x = (f_anchors[:, 0] + f_anchors[:, 2]) * 0.5
    y = (f_anchors[:, 1] + f_anchors[:, 3]) * 0.5
    flags = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    return flags

def anchor_inside_flags(flat_anchors, valid_flags, img_shape, allowed_border=0):
    """Check whether the anchors are inside the border.

    Args:
        flat_anchors (torch.Tensor): Flatten anchors, shape (n, 4).
        valid_flags (torch.Tensor): An existing valid flags of anchors.
        img_shape (tuple(int)): Shape of current image.
        allowed_border (int, optional): The border to allow the valid anchor.
            Defaults to 0.

    Returns:
        torch.Tensor: Flags indicating whether the anchors are inside a             valid range.
    """
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & (flat_anchors[:, 0] >= -allowed_border) & (flat_anchors[:, 1] >= -allowed_border) & (flat_anchors[:, 2] < img_w + allowed_border) & (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags

@HEADS.register_module()
class SABLRetinaHead(BaseDenseHead, BBoxTestMixin):
    """Side-Aware Boundary Localization (SABL) for RetinaNet.

    The anchor generation, assigning and sampling in SABLRetinaHead
    are the same as GuidedAnchorHead for guided anchoring.

    Please refer to https://arxiv.org/abs/1912.04260 for more details.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of Convs for classification             and regression branches. Defaults to 4.
        feat_channels (int): Number of hidden channels.             Defaults to 256.
        approx_anchor_generator (dict): Config dict for approx generator.
        square_anchor_generator (dict): Config dict for square generator.
        conv_cfg (dict): Config dict for ConvModule. Defaults to None.
        norm_cfg (dict): Config dict for Norm Layer. Defaults to None.
        bbox_coder (dict): Config dict for bbox coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        train_cfg (dict): Training config of SABLRetinaHead.
        test_cfg (dict): Testing config of SABLRetinaHead.
        loss_cls (dict): Config of classification loss.
        loss_bbox_cls (dict): Config of classification loss for bbox branch.
        loss_bbox_reg (dict): Config of regression loss for bbox branch.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, num_classes, in_channels, stacked_convs=4, feat_channels=256, approx_anchor_generator=dict(type='AnchorGenerator', octave_base_scale=4, scales_per_octave=3, ratios=[0.5, 1.0, 2.0], strides=[8, 16, 32, 64, 128]), square_anchor_generator=dict(type='AnchorGenerator', ratios=[1.0], scales=[4], strides=[8, 16, 32, 64, 128]), conv_cfg=None, norm_cfg=None, bbox_coder=dict(type='BucketingBBoxCoder', num_buckets=14, scale_factor=3.0), reg_decoded_bbox=False, train_cfg=None, test_cfg=None, loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0), loss_bbox_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.5), loss_bbox_reg=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.5), init_cfg=dict(type='Normal', layer='Conv2d', std=0.01, override=dict(type='Normal', name='retina_cls', std=0.01, bias_prob=0.01))):
        super(SABLRetinaHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_buckets = bbox_coder['num_buckets']
        self.side_num = int(np.ceil(self.num_buckets / 2))
        assert approx_anchor_generator['octave_base_scale'] == square_anchor_generator['scales'][0]
        assert approx_anchor_generator['strides'] == square_anchor_generator['strides']
        self.approx_anchor_generator = build_prior_generator(approx_anchor_generator)
        self.square_anchor_generator = build_prior_generator(square_anchor_generator)
        self.approxs_per_octave = self.approx_anchor_generator.num_base_priors[0]
        self.num_base_priors = self.square_anchor_generator.num_base_priors[0]
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC', 'QualityFocalLoss']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_cls = build_loss(loss_bbox_cls)
        self.loss_bbox_reg = build_loss(loss_bbox_reg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False
        self._init_layers()

    @property
    def num_anchors(self):
        warnings.warn('DeprecationWarning: `num_anchors` is deprecated, please use "num_base_priors" instead')
        return self.square_anchor_generator.num_base_priors[0]

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.retina_bbox_reg = nn.Conv2d(self.feat_channels, self.side_num * 4, 3, padding=1)
        self.retina_bbox_cls = nn.Conv2d(self.feat_channels, self.side_num * 4, 3, padding=1)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_cls_pred = self.retina_bbox_cls(reg_feat)
        bbox_reg_pred = self.retina_bbox_reg(reg_feat)
        bbox_pred = (bbox_cls_pred, bbox_reg_pred)
        return (cls_score, bbox_pred)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get squares according to feature map sizes and guided anchors.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): device for returned tensors

        Returns:
            tuple: square approxs of each image
        """
        num_imgs = len(img_metas)
        multi_level_squares = self.square_anchor_generator.grid_priors(featmap_sizes, device=device)
        squares_list = [multi_level_squares for _ in range(num_imgs)]
        return squares_list

    def get_target(self, approx_list, inside_flag_list, square_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=None, sampling=True, unmap_outputs=True):
        """Compute bucketing targets.
        Args:
            approx_list (list[list]): Multi level approxs of each image.
            inside_flag_list (list[list]): Multi level inside flags of each
                image.
            square_list (list[list]): Multi level squares of each image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): ignore list of gt bboxes.
            gt_bboxes_list (list[Tensor]): Gt bboxes of each image.
            label_channels (int): Channel of label.
            sampling (bool): Sample Anchors or not.
            unmap_outputs (bool): unmap outputs or not.

        Returns:
            tuple: Returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each                     level.
                - bbox_cls_targets_list (list[Tensor]): BBox cls targets of                     each level.
                - bbox_cls_weights_list (list[Tensor]): BBox cls weights of                     each level.
                - bbox_reg_targets_list (list[Tensor]): BBox reg targets of                     each level.
                - bbox_reg_weights_list (list[Tensor]): BBox reg weights of                     each level.
                - num_total_pos (int): Number of positive samples in all                     images.
                - num_total_neg (int): Number of negative samples in all                     images.
        """
        num_imgs = len(img_metas)
        assert len(approx_list) == len(inside_flag_list) == len(square_list) == num_imgs
        num_level_squares = [squares.size(0) for squares in square_list[0]]
        inside_flag_flat_list = []
        approx_flat_list = []
        square_flat_list = []
        for i in range(num_imgs):
            assert len(square_list[i]) == len(inside_flag_list[i])
            inside_flag_flat_list.append(torch.cat(inside_flag_list[i]))
            approx_flat_list.append(torch.cat(approx_list[i]))
            square_flat_list.append(torch.cat(square_list[i]))
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_labels, all_label_weights, all_bbox_cls_targets, all_bbox_cls_weights, all_bbox_reg_targets, all_bbox_reg_weights, pos_inds_list, neg_inds_list = multi_apply(self._get_target_single, approx_flat_list, inside_flag_flat_list, square_flat_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, label_channels=label_channels, sampling=sampling, unmap_outputs=unmap_outputs)
        if any([labels is None for labels in all_labels]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_squares)
        label_weights_list = images_to_levels(all_label_weights, num_level_squares)
        bbox_cls_targets_list = images_to_levels(all_bbox_cls_targets, num_level_squares)
        bbox_cls_weights_list = images_to_levels(all_bbox_cls_weights, num_level_squares)
        bbox_reg_targets_list = images_to_levels(all_bbox_reg_targets, num_level_squares)
        bbox_reg_weights_list = images_to_levels(all_bbox_reg_weights, num_level_squares)
        return (labels_list, label_weights_list, bbox_cls_targets_list, bbox_cls_weights_list, bbox_reg_targets_list, bbox_reg_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, flat_approxs, inside_flags, flat_squares, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, label_channels=None, sampling=True, unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_approxs (Tensor): flat approxs of a single image,
                shape (n, 4)
            inside_flags (Tensor): inside flags of a single image,
                shape (n, ).
            flat_squares (Tensor): flat squares of a single image,
                shape (approxs_per_octave * n, 4)
            gt_bboxes (Tensor): Ground truth bboxes of a single image,                 shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            sampling (bool): Sample Anchors or not.
            unmap_outputs (bool): unmap outputs or not.

        Returns:
            tuple:

                - labels_list (Tensor): Labels in a single image
                - label_weights (Tensor): Label weights in a single image
                - bbox_cls_targets (Tensor): BBox cls targets in a single image
                - bbox_cls_weights (Tensor): BBox cls weights in a single image
                - bbox_reg_targets (Tensor): BBox reg targets in a single image
                - bbox_reg_weights (Tensor): BBox reg weights in a single image
                - num_total_pos (int): Number of positive samples                     in a single image
                - num_total_neg (int): Number of negative samples                     in a single image
        """
        if not inside_flags.any():
            return (None,) * 8
        expand_inside_flags = inside_flags[:, None].expand(-1, self.approxs_per_octave).reshape(-1)
        approxs = flat_approxs[expand_inside_flags, :]
        squares = flat_squares[inside_flags, :]
        assign_result = self.assigner.assign(approxs, squares, self.approxs_per_octave, gt_bboxes, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, squares, gt_bboxes)
        num_valid_squares = squares.shape[0]
        bbox_cls_targets = squares.new_zeros((num_valid_squares, self.side_num * 4))
        bbox_cls_weights = squares.new_zeros((num_valid_squares, self.side_num * 4))
        bbox_reg_targets = squares.new_zeros((num_valid_squares, self.side_num * 4))
        bbox_reg_weights = squares.new_zeros((num_valid_squares, self.side_num * 4))
        labels = squares.new_full((num_valid_squares,), self.num_classes, dtype=torch.long)
        label_weights = squares.new_zeros(num_valid_squares, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_reg_targets, pos_bbox_reg_weights, pos_bbox_cls_targets, pos_bbox_cls_weights = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            bbox_cls_targets[pos_inds, :] = pos_bbox_cls_targets
            bbox_reg_targets[pos_inds, :] = pos_bbox_reg_targets
            bbox_cls_weights[pos_inds, :] = pos_bbox_cls_weights
            bbox_reg_weights[pos_inds, :] = pos_bbox_reg_weights
            if gt_labels is None:
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        if unmap_outputs:
            num_total_anchors = flat_squares.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_cls_targets = unmap(bbox_cls_targets, num_total_anchors, inside_flags)
            bbox_cls_weights = unmap(bbox_cls_weights, num_total_anchors, inside_flags)
            bbox_reg_targets = unmap(bbox_reg_targets, num_total_anchors, inside_flags)
            bbox_reg_weights = unmap(bbox_reg_weights, num_total_anchors, inside_flags)
        return (labels, label_weights, bbox_cls_targets, bbox_cls_weights, bbox_reg_targets, bbox_reg_weights, pos_inds, neg_inds)

    def loss_single(self, cls_score, bbox_pred, labels, label_weights, bbox_cls_targets, bbox_cls_weights, bbox_reg_targets, bbox_reg_weights, num_total_samples):
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        bbox_cls_targets = bbox_cls_targets.reshape(-1, self.side_num * 4)
        bbox_cls_weights = bbox_cls_weights.reshape(-1, self.side_num * 4)
        bbox_reg_targets = bbox_reg_targets.reshape(-1, self.side_num * 4)
        bbox_reg_weights = bbox_reg_weights.reshape(-1, self.side_num * 4)
        bbox_cls_pred, bbox_reg_pred = bbox_pred
        bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, self.side_num * 4)
        bbox_reg_pred = bbox_reg_pred.permute(0, 2, 3, 1).reshape(-1, self.side_num * 4)
        loss_bbox_cls = self.loss_bbox_cls(bbox_cls_pred, bbox_cls_targets.long(), bbox_cls_weights, avg_factor=num_total_samples * 4 * self.side_num)
        loss_bbox_reg = self.loss_bbox_reg(bbox_reg_pred, bbox_reg_targets, bbox_reg_weights, avg_factor=num_total_samples * 4 * self.bbox_coder.offset_topk)
        return (loss_cls, loss_bbox_cls, loss_bbox_reg)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.approx_anchor_generator.num_levels
        device = cls_scores[0].device
        approxs_list, inside_flag_list = GuidedAnchorHead.get_sampled_approxs(self, featmap_sizes, img_metas, device=device)
        square_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_target(approxs_list, inside_flag_list, square_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels, sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_cls_targets_list, bbox_cls_weights_list, bbox_reg_targets_list, bbox_reg_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos
        losses_cls, losses_bbox_cls, losses_bbox_reg = multi_apply(self.loss_single, cls_scores, bbox_preds, labels_list, label_weights_list, bbox_cls_targets_list, bbox_cls_weights_list, bbox_reg_targets_list, bbox_reg_weights_list, num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox_cls=losses_bbox_cls, loss_bbox_reg=losses_bbox_reg)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg=None, rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        mlvl_anchors = self.get_anchors(featmap_sizes, img_metas, device=device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_cls_pred_list = [bbox_preds[i][0][img_id].detach() for i in range(num_levels)]
            bbox_reg_pred_list = [bbox_preds[i][1][img_id].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_cls_pred_list, bbox_reg_pred_list, mlvl_anchors[img_id], img_shape, scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self, cls_scores, bbox_cls_preds, bbox_reg_preds, mlvl_anchors, img_shape, scale_factor, cfg, rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        nms_pre = cfg.get('nms_pre', -1)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_confids = []
        mlvl_labels = []
        assert len(cls_scores) == len(bbox_cls_preds) == len(bbox_reg_preds) == len(mlvl_anchors)
        for cls_score, bbox_cls_pred, bbox_reg_pred, anchors in zip(cls_scores, bbox_cls_preds, bbox_reg_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_cls_pred.size()[-2:] == bbox_reg_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]
            bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.side_num * 4)
            bbox_reg_pred = bbox_reg_pred.permute(1, 2, 0).reshape(-1, self.side_num * 4)
            results = filter_scores_and_topk(scores, cfg.score_thr, nms_pre, dict(anchors=anchors, bbox_cls_pred=bbox_cls_pred, bbox_reg_pred=bbox_reg_pred))
            scores, labels, _, filtered_results = results
            anchors = filtered_results['anchors']
            bbox_cls_pred = filtered_results['bbox_cls_pred']
            bbox_reg_pred = filtered_results['bbox_reg_pred']
            bbox_preds = [bbox_cls_pred.contiguous(), bbox_reg_pred.contiguous()]
            bboxes, confids = self.bbox_coder.decode(anchors.contiguous(), bbox_preds, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_confids.append(confids)
            mlvl_labels.append(labels)
        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes, scale_factor, cfg, rescale, True, mlvl_confids)

def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret

@HEADS.register_module()
class TOODHead(ATSSHead):
    """TOODHead used in `TOOD: Task-aligned One-stage Object Detection.

    <https://arxiv.org/abs/2108.07755>`_.

    TOOD uses Task-aligned head (T-head) and is optimized by Task Alignment
    Learning (TAL).

    Args:
        num_dcn (int): Number of deformable convolution in the head.
            Default: 0.
        anchor_type (str): If set to `anchor_free`, the head will use centers
            to regress bboxes. If set to `anchor_based`, the head will
            regress bboxes based on anchors. Default: `anchor_free`.
        initial_loss_cls (dict): Config of initial loss.

    Example:
        >>> self = TOODHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self, num_classes, in_channels, num_dcn=0, anchor_type='anchor_free', initial_loss_cls=dict(type='FocalLoss', use_sigmoid=True, activated=True, gamma=2.0, alpha=0.25, loss_weight=1.0), **kwargs):
        assert anchor_type in ['anchor_free', 'anchor_based']
        self.num_dcn = num_dcn
        self.anchor_type = anchor_type
        self.epoch = 0
        super(TOODHead, self).__init__(num_classes, in_channels, **kwargs)
        if self.train_cfg:
            self.initial_epoch = self.train_cfg.initial_epoch
            self.initial_assigner = build_assigner(self.train_cfg.initial_assigner)
            self.initial_loss_cls = build_loss(initial_loss_cls)
            self.assigner = self.initial_assigner
            self.alignment_assigner = build_assigner(self.train_cfg.assigner)
            self.alpha = self.train_cfg.alpha
            self.beta = self.train_cfg.beta

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.inter_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            if i < self.num_dcn:
                conv_cfg = dict(type='DCNv2', deform_groups=4)
            else:
                conv_cfg = self.conv_cfg
            chn = self.in_channels if i == 0 else self.feat_channels
            self.inter_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=self.norm_cfg))
        self.cls_decomp = TaskDecomposition(self.feat_channels, self.stacked_convs, self.stacked_convs * 8, self.conv_cfg, self.norm_cfg)
        self.reg_decomp = TaskDecomposition(self.feat_channels, self.stacked_convs, self.stacked_convs * 8, self.conv_cfg, self.norm_cfg)
        self.tood_cls = nn.Conv2d(self.feat_channels, self.num_base_priors * self.cls_out_channels, 3, padding=1)
        self.tood_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4, 3, padding=1)
        self.cls_prob_module = nn.Sequential(nn.Conv2d(self.feat_channels * self.stacked_convs, self.feat_channels // 4, 1), nn.ReLU(inplace=True), nn.Conv2d(self.feat_channels // 4, 1, 3, padding=1))
        self.reg_offset_module = nn.Sequential(nn.Conv2d(self.feat_channels * self.stacked_convs, self.feat_channels // 4, 1), nn.ReLU(inplace=True), nn.Conv2d(self.feat_channels // 4, 4 * 2, 3, padding=1))
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.prior_generator.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        bias_cls = bias_init_with_prob(0.01)
        for m in self.inter_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cls_prob_module:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_offset_module:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.cls_prob_module[-1], std=0.01, bias=bias_cls)
        self.cls_decomp.init_weights()
        self.reg_decomp.init_weights()
        normal_init(self.tood_cls, std=0.01, bias=bias_cls)
        normal_init(self.tood_reg, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Decoded box for all scale levels,
                    each is a 4D-tensor, the channels number is
                    num_anchors * 4. In [tl_x, tl_y, br_x, br_y] format.
        """
        cls_scores = []
        bbox_preds = []
        for idx, (x, scale, stride) in enumerate(zip(feats, self.scales, self.prior_generator.strides)):
            b, c, h, w = x.shape
            anchor = self.prior_generator.single_level_grid_priors((h, w), idx, device=x.device)
            anchor = torch.cat([anchor for _ in range(b)])
            inter_feats = []
            for inter_conv in self.inter_convs:
                x = inter_conv(x)
                inter_feats.append(x)
            feat = torch.cat(inter_feats, 1)
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat = self.cls_decomp(feat, avg_feat)
            reg_feat = self.reg_decomp(feat, avg_feat)
            cls_logits = self.tood_cls(cls_feat)
            cls_prob = self.cls_prob_module(feat)
            cls_score = sigmoid_geometric_mean(cls_logits, cls_prob)
            if self.anchor_type == 'anchor_free':
                reg_dist = scale(self.tood_reg(reg_feat).exp()).float()
                reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, 4)
                reg_bbox = distance2bbox(self.anchor_center(anchor) / stride[0], reg_dist).reshape(b, h, w, 4).permute(0, 3, 1, 2)
            elif self.anchor_type == 'anchor_based':
                reg_dist = scale(self.tood_reg(reg_feat)).float()
                reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, 4)
                reg_bbox = self.bbox_coder.decode(anchor, reg_dist).reshape(b, h, w, 4).permute(0, 3, 1, 2) / stride[0]
            else:
                raise NotImplementedError(f'Unknown anchor type: {self.anchor_type}.Please use `anchor_free` or `anchor_based`.')
            reg_offset = self.reg_offset_module(feat)
            bbox_pred = self.deform_sampling(reg_bbox.contiguous(), reg_offset.contiguous())
            invalid_bbox_idx = (bbox_pred[:, [0]] > bbox_pred[:, [2]]) | (bbox_pred[:, [1]] > bbox_pred[:, [3]])
            invalid_bbox_idx = invalid_bbox_idx.expand_as(bbox_pred)
            bbox_pred = torch.where(invalid_bbox_idx, reg_bbox, bbox_pred)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return (tuple(cls_scores), tuple(bbox_preds))

    def deform_sampling(self, feat, offset):
        """Sampling the feature x according to offset.

        Args:
            feat (Tensor): Feature
            offset (Tensor): Spatial offset for feature sampling
        """
        b, c, h, w = feat.shape
        weight = feat.new_ones(c, 1, 1, 1)
        y = deform_conv2d(feat, offset, weight, 1, 0, 1, c, c)
        return y

    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def loss_single(self, anchors, cls_score, bbox_pred, labels, label_weights, bbox_targets, alignment_metrics, stride):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            alignment_metrics (Tensor): Alignment metrics with shape
                (N, num_total_anchors).
            stride (tuple[int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        alignment_metrics = alignment_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = labels if self.epoch < self.initial_epoch else (labels, alignment_metrics)
        cls_loss_func = self.initial_loss_cls if self.epoch < self.initial_epoch else self.loss_cls
        loss_cls = cls_loss_func(cls_score, targets, label_weights, avg_factor=1.0)
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            pos_bbox_weight = self.centerness_target(pos_anchors, pos_bbox_targets) if self.epoch < self.initial_epoch else alignment_metrics[pos_inds]
            loss_bbox = self.loss_bbox(pos_decode_bbox_pred, pos_decode_bbox_targets, weight=pos_bbox_weight, avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.0)
        return (loss_cls, loss_bbox, alignment_metrics.sum(), pos_bbox_weight.sum())

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Decoded box for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        flatten_cls_scores = torch.cat([cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels) for cls_score in cls_scores], 1)
        flatten_bbox_preds = torch.cat([bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) * stride[0] for bbox_pred, stride in zip(bbox_preds, self.prior_generator.strides)], 1)
        cls_reg_targets = self.get_targets(flatten_cls_scores, flatten_bbox_preds, anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels)
        anchor_list, labels_list, label_weights_list, bbox_targets_list, alignment_metrics_list = cls_reg_targets
        losses_cls, losses_bbox, cls_avg_factors, bbox_avg_factors = multi_apply(self.loss_single, anchor_list, cls_scores, bbox_preds, labels_list, label_weights_list, bbox_targets_list, alignment_metrics_list, self.prior_generator.strides)
        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))
        bbox_avg_factor = reduce_mean(sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def _get_bboxes_single(self, cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors, img_meta, cfg, rescale=False, with_nms=True, **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape                     [num_bboxes, 5], where the first 4 columns are bounding                     box positions (tl_x, tl_y, br_x, br_y) and the 5-th                     column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding                     box with shape [num_bboxes].
        """
        cfg = self.test_cfg if cfg is None else cfg
        nms_pre = cfg.get('nms_pre', -1)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for cls_score, bbox_pred, priors, stride in zip(cls_score_list, bbox_pred_list, mlvl_priors, self.prior_generator.strides):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4) * stride[0]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            results = filter_scores_and_topk(scores, cfg.score_thr, nms_pre, dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results
            bboxes = filtered_results['bbox_pred']
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes, img_meta['scale_factor'], cfg, rescale, with_nms, None, **kwargs)

    def get_targets(self, cls_scores, bbox_preds, anchor_list, valid_flag_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, unmap_outputs=True):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            cls_scores (Tensor): Classification predictions of images,
                a 3D-Tensor with shape [num_imgs, num_priors, num_classes].
            bbox_preds (Tensor): Decoded bboxes predictions of one image,
                a 3D-Tensor with shape [num_imgs, num_priors, 4] in [tl_x,
                tl_y, br_x, br_y] format.
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: a tuple containing learning targets.

                - anchors_list (list[list[Tensor]]): Anchors of each level.
                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - norm_alignment_metrics_list (list[Tensor]): Normalized
                  alignment metrics of each level.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        if self.epoch < self.initial_epoch:
            all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, pos_inds_list, neg_inds_list = multi_apply(super()._get_target_single, anchor_list, valid_flag_list, num_level_anchors_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, label_channels=label_channels, unmap_outputs=unmap_outputs)
            all_assign_metrics = [weight[..., 0] for weight in all_bbox_weights]
        else:
            all_anchors, all_labels, all_label_weights, all_bbox_targets, all_assign_metrics = multi_apply(self._get_target_single, cls_scores, bbox_preds, anchor_list, valid_flag_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, label_channels=label_channels, unmap_outputs=unmap_outputs)
        if any([labels is None for labels in all_labels]):
            return None
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        norm_alignment_metrics_list = images_to_levels(all_assign_metrics, num_level_anchors)
        return (anchors_list, labels_list, label_weights_list, bbox_targets_list, norm_alignment_metrics_list)

    def _get_target_single(self, cls_scores, bbox_preds, flat_anchors, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, label_channels=1, unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            cls_scores (list(Tensor)): Box scores for each image.
            bbox_preds (list(Tensor)): Box energies / deltas for each image.
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                norm_alignment_metrics (Tensor): Normalized alignment metrics
                    of all priors in the image with shape (N,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7
        anchors = flat_anchors[inside_flags, :]
        assign_result = self.alignment_assigner.assign(cls_scores[inside_flags, :], bbox_preds[inside_flags, :], anchors, gt_bboxes, gt_bboxes_ignore, gt_labels, self.alpha, self.beta)
        assign_ious = assign_result.max_overlaps
        assign_metrics = assign_result.assign_metrics
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        norm_alignment_metrics = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            if gt_labels is None:
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        class_assigned_gt_inds = torch.unique(sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds == gt_inds]
            pos_alignment_metrics = assign_metrics[gt_class_inds]
            pos_ious = assign_ious[gt_class_inds]
            pos_norm_alignment_metrics = pos_alignment_metrics / (pos_alignment_metrics.max() + 1e-07) * pos_ious.max()
            norm_alignment_metrics[gt_class_inds] = pos_norm_alignment_metrics
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            norm_alignment_metrics = unmap(norm_alignment_metrics, num_total_anchors, inside_flags)
        return (anchors, labels, label_weights, bbox_targets, norm_alignment_metrics)

@HEADS.register_module()
class ATSSHead(AnchorHead):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection.

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    https://arxiv.org/abs/1912.02424
    """

    def __init__(self, num_classes, in_channels, pred_kernel_size=3, stacked_convs=4, conv_cfg=None, norm_cfg=dict(type='GN', num_groups=32, requires_grad=True), reg_decoded_bbox=True, loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), init_cfg=dict(type='Normal', layer='Conv2d', std=0.01, override=dict(type='Normal', name='atss_cls', std=0.01, bias_prob=0.01)), **kwargs):
        self.pred_kernel_size = pred_kernel_size
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(ATSSHead, self).__init__(num_classes, in_channels, reg_decoded_bbox=reg_decoded_bbox, init_cfg=init_cfg, **kwargs)
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.loss_centerness = build_loss(loss_centerness)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        pred_pad_size = self.pred_kernel_size // 2
        self.atss_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, self.pred_kernel_size, padding=pred_pad_size)
        self.atss_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4, self.pred_kernel_size, padding=pred_pad_size)
        self.atss_centerness = nn.Conv2d(self.feat_channels, self.num_base_priors * 1, self.pred_kernel_size, padding=pred_pad_size)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.prior_generator.strides])

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W).
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        centerness = self.atss_centerness(reg_feat)
        return (cls_score, bbox_pred, centerness)

    def loss_single(self, anchors, cls_score, bbox_pred, centerness, labels, label_weights, bbox_targets, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]
            centerness_targets = self.centerness_target(pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = self.bbox_coder.decode(pos_anchors, pos_bbox_pred)
            loss_bbox = self.loss_bbox(pos_decode_bbox_pred, pos_bbox_targets, weight=centerness_targets, avg_factor=1.0)
            loss_centerness = self.loss_centerness(pos_centerness, centerness_targets, avg_factor=num_total_samples)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.0)
        return (loss_cls, loss_bbox, loss_centerness, centerness_targets.sum())

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self, cls_scores, bbox_preds, centernesses, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        num_total_samples = reduce_mean(torch.tensor(num_total_pos, dtype=torch.float, device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)
        losses_cls, losses_bbox, loss_centerness, bbox_avg_factor = multi_apply(self.loss_single, anchor_list, cls_scores, bbox_preds, centernesses, labels_list, label_weights_list, bbox_targets_list, num_total_samples=num_total_samples)
        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_centerness=loss_centerness)

    def centerness_target(self, anchors, gts):
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy
        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0] * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    def get_targets(self, anchor_list, valid_flag_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, unmap_outputs=True):
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, pos_inds_list, neg_inds_list = multi_apply(self._get_target_single, anchor_list, valid_flag_list, num_level_anchors_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, label_channels=label_channels, unmap_outputs=unmap_outputs)
        if any([labels is None for labels in all_labels]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        return (anchors_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, flat_anchors, valid_flags, num_level_anchors, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, label_channels=1, unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7
        anchors = flat_anchors[inside_flags, :]
        num_level_anchors_inside = self.get_num_level_anchors_inside(num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore, gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if self.reg_decoded_bbox:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            else:
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        return (anchors, labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [int(flags.sum()) for flags in split_inside_flags]
        return num_level_anchors_inside

@HEADS.register_module()
class RepPointsHead(AnchorFreeHead):
    """RepPoint head.

    Args:
        point_feat_channels (int): Number of channels of points features.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        loss_cls (dict): Config of classification loss.
        loss_bbox_init (dict): Config of initial points loss.
        loss_bbox_refine (dict): Config of points loss in refinement.
        use_grid_points (bool): If we use bounding box representation, the
        reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
        transform_method (str): The methods to transform RepPoints to bbox.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, num_classes, in_channels, point_feat_channels=256, num_points=9, gradient_mul=0.1, point_strides=[8, 16, 32, 64, 128], point_base_scale=4, loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0), loss_bbox_init=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5), loss_bbox_refine=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0), use_grid_points=False, center_init=True, transform_method='moment', moment_mul=0.01, init_cfg=dict(type='Normal', layer='Conv2d', std=0.01, override=dict(type='Normal', name='reppoints_cls_out', std=0.01, bias_prob=0.01)), **kwargs):
        self.num_points = num_points
        self.point_feat_channels = point_feat_channels
        self.use_grid_points = use_grid_points
        self.center_init = center_init
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, 'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, 'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad, self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(-1)
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, init_cfg=init_cfg, **kwargs)
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.prior_generator = MlvlPointGenerator(self.point_strides, offset=0.0)
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        if self.train_cfg:
            self.init_assigner = build_assigner(self.train_cfg.init.assigner)
            self.refine_assigner = build_assigner(self.train_cfg.refine.assigner)
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.transform_method = transform_method
        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(data=torch.zeros(2), requires_grad=True)
            self.moment_mul = moment_mul
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        pts_out_dim = 4 if self.use_grid_points else 2 * self.num_points
        self.reppoints_cls_conv = DeformConv2d(self.feat_channels, self.point_feat_channels, self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels, self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv2d(self.feat_channels, self.point_feat_channels, self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)

    def points2bbox(self, pts, y_first=True):
        """Converting the points set into bounding box.

        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_first=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :return: each points set is converting to a bbox [x1, y1, x2, y2].
        """
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1, ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0, ...]
        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)
        elif self.transform_method == 'partial_minmax':
            pts_y = pts_y[:, :4, ...]
            pts_x = pts_x[:, :4, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)
        elif self.transform_method == 'moment':
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = self.moment_transfer * self.moment_mul + self.moment_transfer.detach() * (1 - self.moment_mul)
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)
            bbox = torch.cat([pts_x_mean - half_width, pts_y_mean - half_height, pts_x_mean + half_width, pts_y_mean + half_height], dim=1)
        else:
            raise NotImplementedError
        return bbox

    def gen_grid_from_reg(self, reg, previous_boxes):
        """Base on the previous bboxes and regression values, we compute the
        regressed bboxes and generate the grids on the bboxes.

        :param reg: the regression value to previous bboxes.
        :param previous_boxes: previous bboxes.
        :return: generate grids on the regressed bboxes.
        """
        b, _, h, w = reg.shape
        bxy = (previous_boxes[:, :2, ...] + previous_boxes[:, 2:, ...]) / 2.0
        bwh = (previous_boxes[:, 2:, ...] - previous_boxes[:, :2, ...]).clamp(min=1e-06)
        grid_topleft = bxy + bwh * reg[:, :2, ...] - 0.5 * bwh * torch.exp(reg[:, 2:, ...])
        grid_wh = bwh * torch.exp(reg[:, 2:, ...])
        grid_left = grid_topleft[:, [0], ...]
        grid_top = grid_topleft[:, [1], ...]
        grid_width = grid_wh[:, [0], ...]
        grid_height = grid_wh[:, [1], ...]
        intervel = torch.linspace(0.0, 1.0, self.dcn_kernel).view(1, self.dcn_kernel, 1, 1).type_as(reg)
        grid_x = grid_left + grid_width * intervel
        grid_x = grid_x.unsqueeze(1).repeat(1, self.dcn_kernel, 1, 1, 1)
        grid_x = grid_x.view(b, -1, h, w)
        grid_y = grid_top + grid_height * intervel
        grid_y = grid_y.unsqueeze(2).repeat(1, 1, self.dcn_kernel, 1, 1)
        grid_y = grid_y.view(b, -1, h, w)
        grid_yx = torch.stack([grid_y, grid_x], dim=2)
        grid_yx = grid_yx.view(b, -1, h, w)
        regressed_bbox = torch.cat([grid_left, grid_top, grid_left + grid_width, grid_top + grid_height], 1)
        return (grid_yx, regressed_bbox)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        """Forward feature map of a single FPN level."""
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        if self.use_grid_points or not self.center_init:
            scale = self.point_base_scale / 2
            points_init = dcn_base_offset / dcn_base_offset.max() * scale
            bbox_init = x.new_tensor([-scale, -scale, scale, scale]).view(1, 4, 1, 1)
        else:
            points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        pts_out_init = self.reppoints_pts_init_out(self.relu(self.reppoints_pts_init_conv(pts_feat)))
        if self.use_grid_points:
            pts_out_init, bbox_out_init = self.gen_grid_from_reg(pts_out_init, bbox_init.detach())
        else:
            pts_out_init = pts_out_init + points_init
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        cls_out = self.reppoints_cls_out(self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))
        pts_out_refine = self.reppoints_pts_refine_out(self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
        if self.use_grid_points:
            pts_out_refine, bbox_out_refine = self.gen_grid_from_reg(pts_out_refine, bbox_out_init.detach())
        else:
            pts_out_refine = pts_out_refine + pts_out_init.detach()
        if self.training:
            return (cls_out, pts_out_init, pts_out_refine)
        else:
            return (cls_out, self.points2bbox(pts_out_refine))

    def get_points(self, featmap_sizes, img_metas, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        multi_level_points = self.prior_generator.grid_priors(featmap_sizes, device=device, with_stride=True)
        points_list = [[point.clone() for point in multi_level_points] for _ in range(num_imgs)]
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(featmap_sizes, img_meta['pad_shape'])
            valid_flag_list.append(multi_level_flags)
        return (points_list, valid_flag_list)

    def centers_to_bboxes(self, point_list):
        """Get bboxes according to center points.

        Only used in :class:`MaxIoUAssigner`.
        """
        bbox_list = []
        for i_img, point in enumerate(point_list):
            bbox = []
            for i_lvl in range(len(self.point_strides)):
                scale = self.point_base_scale * self.point_strides[i_lvl] * 0.5
                bbox_shift = torch.Tensor([-scale, -scale, scale, scale]).view(1, 4).type_as(point[0])
                bbox_center = torch.cat([point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift)
            bbox_list.append(bbox)
        return bbox_list

    def offset_to_pts(self, center_list, pred_list):
        """Change from point offset to point coordinate."""
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(-1, 2 * self.num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def _point_target_single(self, flat_proposals, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels, stage='init', unmap_outputs=True):
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None,) * 7
        proposals = flat_proposals[inside_flags, :]
        if stage == 'init':
            assigner = self.init_assigner
            pos_weight = self.train_cfg.init.pos_weight
        else:
            assigner = self.refine_assigner
            pos_weight = self.train_cfg.refine.pos_weight
        assign_result = assigner.assign(proposals, gt_bboxes, gt_bboxes_ignore, None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, proposals, gt_bboxes)
        num_valid_proposals = proposals.shape[0]
        bbox_gt = proposals.new_zeros([num_valid_proposals, 4])
        pos_proposals = torch.zeros_like(proposals)
        proposals_weights = proposals.new_zeros([num_valid_proposals, 4])
        labels = proposals.new_full((num_valid_proposals,), self.num_classes, dtype=torch.long)
        label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            bbox_gt[pos_inds, :] = pos_gt_bboxes
            pos_proposals[pos_inds, :] = proposals[pos_inds, :]
            proposals_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(labels, num_total_proposals, inside_flags)
            label_weights = unmap(label_weights, num_total_proposals, inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
            pos_proposals = unmap(pos_proposals, num_total_proposals, inside_flags)
            proposals_weights = unmap(proposals_weights, num_total_proposals, inside_flags)
        return (labels, label_weights, bbox_gt, pos_proposals, proposals_weights, pos_inds, neg_inds)

    def get_targets(self, proposals_list, valid_flag_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, stage='init', label_channels=1, unmap_outputs=True):
        """Compute corresponding GT box and classification targets for
        proposals.

        Args:
            proposals_list (list[list]): Multi level points/bboxes of each
                image.
            valid_flag_list (list[list]): Multi level valid flags of each
                image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_bboxes_list (list[Tensor]): Ground truth labels of each box.
            stage (str): `init` or `refine`. Generate target for init stage or
                refine stage
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each level.  # noqa: E501
                - bbox_gt_list (list[Tensor]): Ground truth bbox of each level.
                - proposal_list (list[Tensor]): Proposals(points/bboxes) of each level.  # noqa: E501
                - proposal_weights_list (list[Tensor]): Proposal weights of each level.  # noqa: E501
                - num_total_pos (int): Number of positive samples in all images.  # noqa: E501
                - num_total_neg (int): Number of negative samples in all images.  # noqa: E501
        """
        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs
        num_level_proposals = [points.size(0) for points in proposals_list[0]]
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_labels, all_label_weights, all_bbox_gt, all_proposals, all_proposal_weights, pos_inds_list, neg_inds_list = multi_apply(self._point_target_single, proposals_list, valid_flag_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, stage=stage, unmap_outputs=unmap_outputs)
        if any([labels is None for labels in all_labels]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights, num_level_proposals)
        bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
        proposals_list = images_to_levels(all_proposals, num_level_proposals)
        proposal_weights_list = images_to_levels(all_proposal_weights, num_level_proposals)
        return (labels_list, label_weights_list, bbox_gt_list, proposals_list, proposal_weights_list, num_total_pos, num_total_neg)

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine, labels, label_weights, bbox_gt_init, bbox_weights_init, bbox_gt_refine, bbox_weights_refine, stride, num_total_samples_init, num_total_samples_refine):
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        cls_score = cls_score.contiguous()
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples_refine)
        bbox_gt_init = bbox_gt_init.reshape(-1, 4)
        bbox_weights_init = bbox_weights_init.reshape(-1, 4)
        bbox_pred_init = self.points2bbox(pts_pred_init.reshape(-1, 2 * self.num_points), y_first=False)
        bbox_gt_refine = bbox_gt_refine.reshape(-1, 4)
        bbox_weights_refine = bbox_weights_refine.reshape(-1, 4)
        bbox_pred_refine = self.points2bbox(pts_pred_refine.reshape(-1, 2 * self.num_points), y_first=False)
        normalize_term = self.point_base_scale * stride
        loss_pts_init = self.loss_bbox_init(bbox_pred_init / normalize_term, bbox_gt_init / normalize_term, bbox_weights_init, avg_factor=num_total_samples_init)
        loss_pts_refine = self.loss_bbox_refine(bbox_pred_refine / normalize_term, bbox_gt_refine / normalize_term, bbox_weights_refine, avg_factor=num_total_samples_refine)
        return (loss_cls, loss_pts_init, loss_pts_refine)

    def loss(self, cls_scores, pts_preds_init, pts_preds_refine, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas, device)
        pts_coordinate_preds_init = self.offset_to_pts(center_list, pts_preds_init)
        if self.train_cfg.init.assigner['type'] == 'PointAssigner':
            candidate_list = center_list
        else:
            bbox_list = self.centers_to_bboxes(center_list)
            candidate_list = bbox_list
        cls_reg_targets_init = self.get_targets(candidate_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, stage='init', label_channels=label_channels)
        *_, bbox_gt_list_init, candidate_list_init, bbox_weights_list_init, num_total_pos_init, num_total_neg_init = cls_reg_targets_init
        num_total_samples_init = num_total_pos_init + num_total_neg_init if self.sampling else num_total_pos_init
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas, device)
        pts_coordinate_preds_refine = self.offset_to_pts(center_list, pts_preds_refine)
        bbox_list = []
        for i_img, center in enumerate(center_list):
            bbox = []
            for i_lvl in range(len(pts_preds_refine)):
                bbox_preds_init = self.points2bbox(pts_preds_init[i_lvl].detach())
                bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
                bbox_center = torch.cat([center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))
            bbox_list.append(bbox)
        cls_reg_targets_refine = self.get_targets(bbox_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, stage='refine', label_channels=label_channels)
        labels_list, label_weights_list, bbox_gt_list_refine, candidate_list_refine, bbox_weights_list_refine, num_total_pos_refine, num_total_neg_refine = cls_reg_targets_refine
        num_total_samples_refine = num_total_pos_refine + num_total_neg_refine if self.sampling else num_total_pos_refine
        losses_cls, losses_pts_init, losses_pts_refine = multi_apply(self.loss_single, cls_scores, pts_coordinate_preds_init, pts_coordinate_preds_refine, labels_list, label_weights_list, bbox_gt_list_init, bbox_weights_list_init, bbox_gt_list_refine, bbox_weights_list_refine, self.point_strides, num_total_samples_init=num_total_samples_init, num_total_samples_refine=num_total_samples_refine)
        loss_dict_all = {'loss_cls': losses_cls, 'loss_pts_init': losses_pts_init, 'loss_pts_refine': losses_pts_refine}
        return loss_dict_all

    def _get_bboxes_single(self, cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors, img_meta, cfg, rescale=False, with_nms=True, **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RepPoints head does not need
                this value.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 2).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape                     [num_bboxes, 5], where the first 4 columns are bounding                     box positions (tl_x, tl_y, br_x, br_y) and the 5-th                     column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding                     box with shape [num_bboxes].
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, priors) in enumerate(zip(cls_score_list, bbox_pred_list, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]
            results = filter_scores_and_topk(scores, cfg.score_thr, nms_pre, dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, _, filtered_results = results
            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            bboxes = self._bbox_decode(priors, bbox_pred, self.point_strides[level_idx], img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes, img_meta['scale_factor'], cfg, rescale=rescale, with_nms=with_nms)

    def _bbox_decode(self, points, bbox_pred, stride, max_shape):
        bbox_pos_center = torch.cat([points[:, :2], points[:, :2]], dim=1)
        bboxes = bbox_pred * stride + bbox_pos_center
        x1 = bboxes[:, 0].clamp(min=0, max=max_shape[1])
        y1 = bboxes[:, 1].clamp(min=0, max=max_shape[0])
        x2 = bboxes[:, 2].clamp(min=0, max=max_shape[1])
        y2 = bboxes[:, 3].clamp(min=0, max=max_shape[0])
        decoded_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return decoded_bboxes

@HEADS.register_module()
class AnchorHead(BaseDenseHead, BBoxTestMixin):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, num_classes, in_channels, feat_channels=256, anchor_generator=dict(type='AnchorGenerator', scales=[8, 16, 32], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]), bbox_coder=dict(type='DeltaXYWHBBoxCoder', clip_border=True, target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(1.0, 1.0, 1.0, 1.0)), reg_decoded_bbox=False, loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0), train_cfg=None, test_cfg=None, init_cfg=dict(type='Normal', layer='Conv2d', std=0.01)):
        super(AnchorHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.reg_decoded_bbox = reg_decoded_bbox
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if hasattr(self.train_cfg, 'sampler') and self.train_cfg.sampler.type.split('.')[-1] != 'PseudoSampler':
                self.sampling = True
                sampler_cfg = self.train_cfg.sampler
                if loss_cls['type'] in ['FocalLoss', 'GHMC', 'QualityFocalLoss']:
                    warnings.warn('DeprecationWarning: Determining whether to samplingby loss type is deprecated, please delete sampler inyour config when using `FocalLoss`, `GHMC`, `QualityFocalLoss` or other FocalLoss variant.')
                    self.sampling = False
                    sampler_cfg = dict(type='PseudoSampler')
            else:
                self.sampling = False
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False
        self.prior_generator = build_prior_generator(anchor_generator)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self._init_layers()

    @property
    def num_anchors(self):
        warnings.warn('DeprecationWarning: `num_anchors` is deprecated, for consistency or also use `num_base_priors` instead')
        return self.prior_generator.num_base_priors[0]

    @property
    def anchor_generator(self):
        warnings.warn('DeprecationWarning: anchor_generator is deprecated, please use "prior_generator" instead')
        return self.prior_generator

    def _init_layers(self):
        """Initialize layers of the head."""
        self.conv_cls = nn.Conv2d(self.in_channels, self.num_base_priors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_base_priors * 4, 1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level                     the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale                     level, the channels number is num_base_priors * 4.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return (cls_score, bbox_pred)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all                     scale levels, each is a 4D-tensor, the channels number                     is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all                     scale levels, each is a 4D-tensor, the channels number                     is num_base_priors * 4.
        """
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)
        multi_level_anchors = self.prior_generator.grid_priors(featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)
        return (anchor_list, valid_flag_list)

    def _get_targets_single(self, flat_anchors, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, label_channels=1, unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7
        anchors = flat_anchors[inside_flags, :]
        assign_result = self.assigner.assign(anchors, gt_bboxes, gt_bboxes_ignore, None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds, sampling_result)

    def get_targets(self, anchor_list, valid_flag_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, unmap_outputs=True, return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all
                  images.
                - num_total_neg (int): Number of negative samples in all
                  images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(self._get_targets_single, concat_anchor_list, concat_valid_flag_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, label_channels=label_channels, unmap_outputs=unmap_outputs)
        all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, pos_inds_list, neg_inds_list, sampling_results_list = results[:7]
        rest_results = list(results[7:])
        if any([labels is None for labels in all_labels]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list,)
        for i, r in enumerate(rest_results):
            rest_results[i] = images_to_levels(r, num_level_anchors)
        return res + tuple(rest_results)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights, bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples)
        return (loss_cls, loss_bbox)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)
        losses_cls, losses_bbox = multi_apply(self.loss_single, cls_scores, bbox_preds, all_anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5), where
                5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,), The length of list should always be 1.
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)

@HEADS.register_module()
class YOLOFHead(AnchorHead):
    """YOLOFHead Paper link: https://arxiv.org/abs/2103.09460.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): The number of input channels per scale.
        cls_num_convs (int): The number of convolutions of cls branch.
           Default 2.
        reg_num_convs (int): The number of convolutions of reg branch.
           Default 4.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self, num_classes, in_channels, num_cls_convs=2, num_reg_convs=4, norm_cfg=dict(type='BN', requires_grad=True), **kwargs):
        self.num_cls_convs = num_cls_convs
        self.num_reg_convs = num_reg_convs
        self.norm_cfg = norm_cfg
        super(YOLOFHead, self).__init__(num_classes, in_channels, **kwargs)

    def _init_layers(self):
        cls_subnet = []
        bbox_subnet = []
        for i in range(self.num_cls_convs):
            cls_subnet.append(ConvModule(self.in_channels, self.in_channels, kernel_size=3, padding=1, norm_cfg=self.norm_cfg))
        for i in range(self.num_reg_convs):
            bbox_subnet.append(ConvModule(self.in_channels, self.in_channels, kernel_size=3, padding=1, norm_cfg=self.norm_cfg))
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(self.in_channels, self.num_base_priors * self.num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(self.in_channels, self.num_base_priors * 4, kernel_size=3, stride=1, padding=1)
        self.object_pred = nn.Conv2d(self.in_channels, self.num_base_priors, kernel_size=3, stride=1, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
        bias_cls = bias_init_with_prob(0.01)
        torch.nn.init.constant_(self.cls_score.bias, bias_cls)

    def forward_single(self, feature):
        cls_score = self.cls_score(self.cls_subnet(feature))
        N, _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, self.num_classes, H, W)
        reg_feat = self.bbox_subnet(feature)
        bbox_reg = self.bbox_pred(reg_feat)
        objectness = self.object_pred(reg_feat)
        objectness = objectness.view(N, -1, 1, H, W)
        normalized_cls_score = cls_score + objectness - torch.log(1.0 + torch.clamp(cls_score.exp(), max=INF) + torch.clamp(objectness.exp(), max=INF))
        normalized_cls_score = normalized_cls_score.view(N, -1, H, W)
        return (normalized_cls_score, bbox_reg)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (batch, num_anchors * num_classes, h, w)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (batch, num_anchors * 4, h, w)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == 1
        assert self.prior_generator.num_levels == 1
        device = cls_scores[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        anchor_list = [anchors[0] for anchors in anchor_list]
        valid_flag_list = [valid_flags[0] for valid_flags in valid_flag_list]
        cls_scores_list = levels_to_images(cls_scores)
        bbox_preds_list = levels_to_images(bbox_preds)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        batch_labels, batch_label_weights, num_total_pos, num_total_neg, batch_bbox_weights, batch_pos_predicted_boxes, batch_target_boxes = cls_reg_targets
        flatten_labels = batch_labels.reshape(-1)
        batch_label_weights = batch_label_weights.reshape(-1)
        cls_score = cls_scores[0].permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos
        num_total_samples = reduce_mean(cls_score.new_tensor(num_total_samples)).clamp_(1.0).item()
        loss_cls = self.loss_cls(cls_score, flatten_labels, batch_label_weights, avg_factor=num_total_samples)
        if batch_pos_predicted_boxes.shape[0] == 0:
            loss_bbox = batch_pos_predicted_boxes.sum() * 0
        else:
            loss_bbox = self.loss_bbox(batch_pos_predicted_boxes, batch_target_boxes, batch_bbox_weights.float(), avg_factor=num_total_samples)
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    def get_targets(self, cls_scores_list, bbox_preds_list, anchor_list, valid_flag_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, unmap_outputs=True):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            cls_scores_list (list[Tensor])： Classification scores of
                each image. each is a 4D-tensor, the shape is
                (h * w, num_anchors * num_classes).
            bbox_preds_list (list[Tensor])： Bbox preds of each image.
                each is a 4D-tensor, the shape is (h * w, num_anchors * 4).
            anchor_list (list[Tensor]): Anchors of each image. Each element of
                is a tensor of shape (h * w * num_anchors, 4).
            valid_flag_list (list[Tensor]): Valid flags of each image. Each
               element of is a tensor of shape (h * w * num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - batch_labels (Tensor): Label of all images. Each element                     of is a tensor of shape (batch, h * w * num_anchors)
                - batch_label_weights (Tensor): Label weights of all images                     of is a tensor of shape (batch, h * w * num_anchors)
                - num_total_pos (int): Number of positive samples in all                     images.
                - num_total_neg (int): Number of negative samples in all                     images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(self._get_targets_single, bbox_preds_list, anchor_list, valid_flag_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, label_channels=label_channels, unmap_outputs=unmap_outputs)
        all_labels, all_label_weights, pos_inds_list, neg_inds_list, sampling_results_list = results[:5]
        rest_results = list(results[5:])
        if any([labels is None for labels in all_labels]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        batch_labels = torch.stack(all_labels, 0)
        batch_label_weights = torch.stack(all_label_weights, 0)
        res = (batch_labels, batch_label_weights, num_total_pos, num_total_neg)
        for i, rests in enumerate(rest_results):
            rest_results[i] = torch.cat(rests, 0)
        return res + tuple(rest_results)

    def _get_targets_single(self, bbox_preds, flat_anchors, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, label_channels=1, unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            bbox_preds (Tensor): Bbox prediction of the image, which
                shape is (h * w ,4)
            flat_anchors (Tensor): Anchors of the image, which shape is
                (h * w * num_anchors ,4)
            valid_flags (Tensor): Valid flags of the image, which shape is
                (h * w * num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels (Tensor): Labels of image, which shape is
                    (h * w * num_anchors, ).
                label_weights (Tensor): Label weights of image, which shape is
                    (h * w * num_anchors, ).
                pos_inds (Tensor): Pos index of image.
                neg_inds (Tensor): Neg index of image.
                sampling_result (obj:`SamplingResult`): Sampling result.
                pos_bbox_weights (Tensor): The Weight of using to calculate
                    the bbox branch loss, which shape is (num, ).
                pos_predicted_boxes (Tensor): boxes predicted value of
                    using to calculate the bbox branch loss, which shape is
                    (num, 4).
                pos_target_boxes (Tensor): boxes target value of
                    using to calculate the bbox branch loss, which shape is
                    (num, 4).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 8
        anchors = flat_anchors[inside_flags, :]
        bbox_preds = bbox_preds.reshape(-1, 4)
        bbox_preds = bbox_preds[inside_flags, :]
        decoder_bbox_preds = self.bbox_coder.decode(anchors, bbox_preds)
        assign_result = self.assigner.assign(decoder_bbox_preds, anchors, gt_bboxes, gt_bboxes_ignore, None if self.sampling else gt_labels)
        pos_bbox_weights = assign_result.get_extra_property('pos_idx')
        pos_predicted_boxes = assign_result.get_extra_property('pos_predicted_boxes')
        pos_target_boxes = assign_result.get_extra_property('target_boxes')
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if gt_labels is None:
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        return (labels, label_weights, pos_inds, neg_inds, sampling_result, pos_bbox_weights, pos_predicted_boxes, pos_target_boxes)

@HEADS.register_module()
class DDODHead(AnchorHead):
    """DDOD head decomposes conjunctions lying in most current one-stage
    detectors via label assignment disentanglement, spatial feature
    disentanglement, and pyramid supervision disentanglement.

    https://arxiv.org/abs/2107.02963

    Args:
        num_classes (int): Number of categories excluding the
            background category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): The number of stacked Conv. Default: 4.
        conv_cfg (dict): Conv config of ddod head. Default: None.
        use_dcn (bool): Use dcn, Same as ATSS when False. Default: True.
        norm_cfg (dict): Normal config of ddod head. Default:
            dict(type='GN', num_groups=32, requires_grad=True).
        loss_iou (dict): Config of IoU loss. Default:
            dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0).
    """

    def __init__(self, num_classes, in_channels, stacked_convs=4, conv_cfg=None, use_dcn=True, norm_cfg=dict(type='GN', num_groups=32, requires_grad=True), loss_iou=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_dcn = use_dcn
        super(DDODHead, self).__init__(num_classes, in_channels, **kwargs)
        self.sampling = False
        if self.train_cfg:
            self.cls_assigner = build_assigner(self.train_cfg.assigner)
            self.reg_assigner = build_assigner(self.train_cfg.reg_assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.loss_iou = build_loss(loss_iou)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=dict(type='DCN', deform_groups=1) if i == 0 and self.use_dcn else self.conv_cfg, norm_cfg=self.norm_cfg))
            self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=dict(type='DCN', deform_groups=1) if i == 0 and self.use_dcn else self.conv_cfg, norm_cfg=self.norm_cfg))
        self.atss_cls = nn.Conv2d(self.feat_channels, self.num_base_priors * self.cls_out_channels, 3, padding=1)
        self.atss_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4, 3, padding=1)
        self.atss_iou = nn.Conv2d(self.feat_channels, self.num_base_priors * 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.prior_generator.strides])
        self.cls_num_pos_samples_per_level = [0.0 for _ in range(len(self.prior_generator.strides))]
        self.reg_num_pos_samples_per_level = [0.0 for _ in range(len(self.prior_generator.strides))]

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        normal_init(self.atss_reg, std=0.01)
        normal_init(self.atss_iou, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.atss_cls, std=0.01, bias=bias_cls)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_base_priors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_base_priors * 4.
                iou_preds (list[Tensor]): IoU scores for all scale levels,
                    each is a 4D-tensor, the channels number is
                    num_base_priors * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                - cls_score (Tensor): Cls scores for a single scale level                     the channels number is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for a single                     scale level, the channels number is num_base_priors * 4.
                - iou_pred (Tensor): Iou for a single scale level, the                     channel number is (N, num_base_priors * 1, H, W).
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        iou_pred = self.atss_iou(reg_feat)
        return (cls_score, bbox_pred, iou_pred)

    def loss_cls_single(self, cls_score, labels, label_weights, reweight_factor, num_total_samples):
        """Compute cls loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_base_priors * num_classes, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            reweight_factor (list[int]): Reweight factor for cls and reg
                loss.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            tuple[Tensor]: A tuple of loss components.
        """
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous()
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        return (reweight_factor * loss_cls,)

    def loss_reg_single(self, anchors, bbox_pred, iou_pred, labels, label_weights, bbox_targets, bbox_weights, reweight_factor, num_total_samples):
        """Compute reg loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_base_priors * 4, H, W).
            iou_pred (Tensor): Iou for a single scale level, the
                channel number is (N, num_base_priors * 1, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox weights of all anchors in the
                image with shape (N, 4)
            reweight_factor (list[int]): Reweight factor for cls and reg
                loss.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        anchors = anchors.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        iou_pred = iou_pred.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        iou_targets = label_weights.new_zeros(labels.shape)
        iou_weights = label_weights.new_zeros(labels.shape)
        iou_weights[(bbox_weights.sum(axis=1) > 0).nonzero(as_tuple=False)] = 1.0
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_decode_bbox_pred = self.bbox_coder.decode(pos_anchors, pos_bbox_pred)
            pos_decode_bbox_targets = self.bbox_coder.decode(pos_anchors, pos_bbox_targets)
            loss_bbox = self.loss_bbox(pos_decode_bbox_pred, pos_decode_bbox_targets, avg_factor=num_total_samples)
            iou_targets[pos_inds] = bbox_overlaps(pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True)
            loss_iou = self.loss_iou(iou_pred, iou_targets, iou_weights, avg_factor=num_total_samples)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_iou = iou_pred.sum() * 0
        return (reweight_factor * loss_bbox, reweight_factor * loss_iou)

    def calc_reweight_factor(self, labels_list):
        """Compute reweight_factor for regression and classification loss."""
        bg_class_ind = self.num_classes
        for ii, each_level_label in enumerate(labels_list):
            pos_inds = ((each_level_label >= 0) & (each_level_label < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)
            self.cls_num_pos_samples_per_level[ii] += len(pos_inds)
        min_pos_samples = min(self.cls_num_pos_samples_per_level)
        max_pos_samples = max(self.cls_num_pos_samples_per_level)
        interval = 1.0 / (max_pos_samples - min_pos_samples + 1e-10)
        reweight_factor_per_level = []
        for pos_samples in self.cls_num_pos_samples_per_level:
            factor = 2.0 - (pos_samples - min_pos_samples) * interval
            reweight_factor_per_level.append(factor)
        return reweight_factor_per_level

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def loss(self, cls_scores, bbox_preds, iou_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_base_priors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_base_priors * 4, H, W)
            iou_preds (list[Tensor]): Score factor for all scale level,
                each is a 4D-tensor, has shape (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        targets_com = self.process_predictions_and_anchors(anchor_list, valid_flag_list, cls_scores, bbox_preds, img_metas, gt_bboxes_ignore)
        anchor_list, valid_flag_list, num_level_anchors_list, cls_score_list, bbox_pred_list, gt_bboxes_ignore_list = targets_com
        cls_targets = self.get_cls_targets(anchor_list, valid_flag_list, num_level_anchors_list, cls_score_list, bbox_pred_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore_list, gt_labels_list=gt_labels, label_channels=label_channels)
        if cls_targets is None:
            return None
        cls_anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_targets
        num_total_samples = reduce_mean(torch.tensor(num_total_pos, dtype=torch.float, device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)
        reweight_factor_per_level = self.calc_reweight_factor(labels_list)
        cls_losses_cls, = multi_apply(self.loss_cls_single, cls_scores, labels_list, label_weights_list, reweight_factor_per_level, num_total_samples=num_total_samples)
        reg_targets = self.get_reg_targets(anchor_list, valid_flag_list, num_level_anchors_list, cls_score_list, bbox_pred_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore_list, gt_labels_list=gt_labels, label_channels=label_channels)
        if reg_targets is None:
            return None
        reg_anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = reg_targets
        num_total_samples = reduce_mean(torch.tensor(num_total_pos, dtype=torch.float, device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)
        reweight_factor_per_level = self.calc_reweight_factor(labels_list)
        reg_losses_bbox, reg_losses_iou = multi_apply(self.loss_reg_single, reg_anchor_list, bbox_preds, iou_preds, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, reweight_factor_per_level, num_total_samples=num_total_samples)
        return dict(loss_cls=cls_losses_cls, loss_bbox=reg_losses_bbox, loss_iou=reg_losses_iou)

    def process_predictions_and_anchors(self, anchor_list, valid_flag_list, cls_scores, bbox_preds, img_metas, gt_bboxes_ignore_list):
        """Compute common vars for regression and classification targets.

        Args:
            anchor_list (list[Tensor]): anchors of each image.
            valid_flag_list (list[Tensor]): Valid flags of each image.
            cls_scores (list[Tensor]): Classification scores for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * 4.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore_list (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Return:
            tuple[Tensor]: A tuple of common loss vars.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        anchor_list_ = []
        valid_flag_list_ = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list_.append(torch.cat(anchor_list[i]))
            valid_flag_list_.append(torch.cat(valid_flag_list[i]))
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        num_levels = len(cls_scores)
        cls_score_list = []
        bbox_pred_list = []
        mlvl_cls_score_list = [cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_base_priors * self.cls_out_channels) for cls_score in cls_scores]
        mlvl_bbox_pred_list = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_base_priors * 4) for bbox_pred in bbox_preds]
        for i in range(num_imgs):
            mlvl_cls_tensor_list = [mlvl_cls_score_list[j][i] for j in range(num_levels)]
            mlvl_bbox_tensor_list = [mlvl_bbox_pred_list[j][i] for j in range(num_levels)]
            cat_mlvl_cls_score = torch.cat(mlvl_cls_tensor_list, dim=0)
            cat_mlvl_bbox_pred = torch.cat(mlvl_bbox_tensor_list, dim=0)
            cls_score_list.append(cat_mlvl_cls_score)
            bbox_pred_list.append(cat_mlvl_bbox_pred)
        return (anchor_list_, valid_flag_list_, num_level_anchors_list, cls_score_list, bbox_pred_list, gt_bboxes_ignore_list)

    def get_cls_targets(self, anchor_list, valid_flag_list, num_level_anchors_list, cls_score_list, bbox_pred_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, unmap_outputs=True):
        """Get cls targets for DDOD head.

        This method is almost the same as `AnchorHead.get_targets()`.
        Besides returning the targets as the parent  method does,
        it also returns the anchors as the first element of the
        returned tuple.

        Args:
            anchor_list (list[Tensor]): anchors of each image.
            valid_flag_list (list[Tensor]): Valid flags of each image.
            num_level_anchors_list (list[Tensor]): Number of anchors of each
                scale level of all image.
            cls_score_list (list[Tensor]): Classification scores for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * num_classes.
            bbox_pred_list (list[Tensor]): Box energies / deltas for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * 4.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore_list (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.
            gt_labels_list (list[Tensor]): class indices corresponding to
                each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Return:
            tuple[Tensor]: A tuple of cls targets components.
        """
        all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, pos_inds_list, neg_inds_list = multi_apply(self._get_target_single, anchor_list, valid_flag_list, cls_score_list, bbox_pred_list, num_level_anchors_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, label_channels=label_channels, unmap_outputs=unmap_outputs, is_cls_assigner=True)
        if any([labels is None for labels in all_labels]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        anchors_list = images_to_levels(all_anchors, num_level_anchors_list[0])
        labels_list = images_to_levels(all_labels, num_level_anchors_list[0])
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors_list[0])
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors_list[0])
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors_list[0])
        return (anchors_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)

    def get_reg_targets(self, anchor_list, valid_flag_list, num_level_anchors_list, cls_score_list, bbox_pred_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, unmap_outputs=True):
        """Get reg targets for DDOD head.

        This method is almost the same as `AnchorHead.get_targets()` when
        is_cls_assigner is False. Besides returning the targets as the parent
        method does, it also returns the anchors as the first element of the
        returned tuple.

        Args:
            anchor_list (list[Tensor]): anchors of each image.
            valid_flag_list (list[Tensor]): Valid flags of each image.
            num_level_anchors (int): Number of anchors of each scale level.
            cls_scores (list[Tensor]): Classification scores for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * 4.
            gt_labels_list (list[Tensor]): class indices corresponding to
                each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore_list (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Return:
            tuple[Tensor]: A tuple of reg targets components.
        """
        all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, pos_inds_list, neg_inds_list = multi_apply(self._get_target_single, anchor_list, valid_flag_list, cls_score_list, bbox_pred_list, num_level_anchors_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, label_channels=label_channels, unmap_outputs=unmap_outputs, is_cls_assigner=False)
        if any([labels is None for labels in all_labels]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        anchors_list = images_to_levels(all_anchors, num_level_anchors_list[0])
        labels_list = images_to_levels(all_labels, num_level_anchors_list[0])
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors_list[0])
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors_list[0])
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors_list[0])
        return (anchors_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, flat_anchors, valid_flags, cls_scores, bbox_preds, num_level_anchors, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, label_channels=1, unmap_outputs=True, is_cls_assigner=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image,
                which are concatenated into a single tensor of shape
                (num_base_priors, 4).
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                shape (num_base_priors,).
            cls_scores (Tensor): Classification scores for all scale
                levels of the image.
            bbox_preds (Tensor): Box energies / deltas for all scale
                levels of the image.
            num_level_anchors (list[int]): Number of anchors of each
                scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, ).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts, ).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label. Default: 1.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Default: True.
            is_cls_assigner (bool): Classification or regression.
                Default: True.

        Returns:
            tuple: N is the number of total anchors in the image.
                - labels (Tensor): Labels of all anchors in the image with                     shape (N, ).
                - label_weights (Tensor): Label weights of all anchor in the                     image with shape (N, ).
                - bbox_targets (Tensor): BBox targets of all anchors in the                     image with shape (N, 4).
                - bbox_weights (Tensor): BBox weights of all anchors in the                     image with shape (N, 4)
                - pos_inds (Tensor): Indices of positive anchor with shape                     (num_pos, ).
                - neg_inds (Tensor): Indices of negative anchor with shape                     (num_neg, ).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7
        anchors = flat_anchors[inside_flags, :]
        num_level_anchors_inside = self.get_num_level_anchors_inside(num_level_anchors, inside_flags)
        bbox_preds_valid = bbox_preds[inside_flags, :]
        cls_scores_valid = cls_scores[inside_flags, :]
        assigner = self.cls_assigner if is_cls_assigner else self.reg_assigner
        bbox_preds_valid = self.bbox_coder.decode(anchors, bbox_preds_valid)
        assign_result = assigner.assign(anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore, gt_labels, cls_scores_valid, bbox_preds_valid)
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if hasattr(self, 'bbox_coder'):
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        return (anchors, labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        """Get the anchors of each scale level inside.

        Args:
            num_level_anchors (list[int]): Number of anchors of each
                scale level.
            inside_flags (Tensor): Multi level inside flags of the image,
                which are concatenated into a single tensor of
                shape (num_base_priors,).

        Returns:
            list[int]: Number of anchors of each scale level inside.
        """
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [int(flags.sum()) for flags in split_inside_flags]
        return num_level_anchors_inside

@HEADS.register_module()
class GuidedAnchorHead(AnchorHead):
    """Guided-Anchor-based head (GA-RPN, GA-RetinaNet, etc.).

    This GuidedAnchorHead will predict high-quality feature guided
    anchors and locations where anchors will be kept in inference.
    There are mainly 3 categories of bounding-boxes.

    - Sampled 9 pairs for target assignment. (approxes)
    - The square boxes where the predicted anchors are based on. (squares)
    - Guided anchors.

    Please refer to https://arxiv.org/abs/1901.03278 for more details.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels.
        approx_anchor_generator (dict): Config dict for approx generator
        square_anchor_generator (dict): Config dict for square generator
        anchor_coder (dict): Config dict for anchor coder
        bbox_coder (dict): Config dict for bbox coder
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        deform_groups: (int): Group number of DCN in
            FeatureAdaption module.
        loc_filter_thr (float): Threshold to filter out unconcerned regions.
        loss_loc (dict): Config of location loss.
        loss_shape (dict): Config of anchor shape loss.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of bbox regression loss.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, num_classes, in_channels, feat_channels=256, approx_anchor_generator=dict(type='AnchorGenerator', octave_base_scale=8, scales_per_octave=3, ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]), square_anchor_generator=dict(type='AnchorGenerator', ratios=[1.0], scales=[8], strides=[4, 8, 16, 32, 64]), anchor_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]), bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]), reg_decoded_bbox=False, deform_groups=4, loc_filter_thr=0.01, train_cfg=None, test_cfg=None, loss_loc=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0), loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0), loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0), init_cfg=dict(type='Normal', layer='Conv2d', std=0.01, override=dict(type='Normal', name='conv_loc', std=0.01, bias_prob=0.01))):
        super(AnchorHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.deform_groups = deform_groups
        self.loc_filter_thr = loc_filter_thr
        assert approx_anchor_generator['octave_base_scale'] == square_anchor_generator['scales'][0]
        assert approx_anchor_generator['strides'] == square_anchor_generator['strides']
        self.approx_anchor_generator = build_prior_generator(approx_anchor_generator)
        self.square_anchor_generator = build_prior_generator(square_anchor_generator)
        self.approxs_per_octave = self.approx_anchor_generator.num_base_priors[0]
        self.reg_decoded_bbox = reg_decoded_bbox
        self.num_base_priors = self.square_anchor_generator.num_base_priors[0]
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.loc_focal_loss = loss_loc['type'] in ['FocalLoss']
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        self.ga_sampling = train_cfg is not None and hasattr(train_cfg, 'ga_sampler')
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        self.anchor_coder = build_bbox_coder(anchor_coder)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_loc = build_loss(loss_loc)
        self.loss_shape = build_loss(loss_shape)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
            self.ga_assigner = build_assigner(self.train_cfg.ga_assigner)
            if self.ga_sampling:
                ga_sampler_cfg = self.train_cfg.ga_sampler
            else:
                ga_sampler_cfg = dict(type='PseudoSampler')
            self.ga_sampler = build_sampler(ga_sampler_cfg, context=self)
        self.fp16_enabled = False
        self._init_layers()

    @property
    def num_anchors(self):
        warnings.warn('DeprecationWarning: `num_anchors` is deprecated, please use "num_base_priors" instead')
        return self.square_anchor_generator.num_base_priors[0]

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.conv_loc = nn.Conv2d(self.in_channels, 1, 1)
        self.conv_shape = nn.Conv2d(self.in_channels, self.num_base_priors * 2, 1)
        self.feature_adaption = FeatureAdaption(self.in_channels, self.feat_channels, kernel_size=3, deform_groups=self.deform_groups)
        self.conv_cls = MaskedConv2d(self.feat_channels, self.num_base_priors * self.cls_out_channels, 1)
        self.conv_reg = MaskedConv2d(self.feat_channels, self.num_base_priors * 4, 1)

    def forward_single(self, x):
        loc_pred = self.conv_loc(x)
        shape_pred = self.conv_shape(x)
        x = self.feature_adaption(x, shape_pred)
        if not self.training:
            mask = loc_pred.sigmoid()[0] >= self.loc_filter_thr
        else:
            mask = None
        cls_score = self.conv_cls(x, mask)
        bbox_pred = self.conv_reg(x, mask)
        return (cls_score, bbox_pred, shape_pred, loc_pred)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_sampled_approxs(self, featmap_sizes, img_metas, device='cuda'):
        """Get sampled approxs and inside flags according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): device for returned tensors

        Returns:
            tuple: approxes of each image, inside flags of each image
        """
        num_imgs = len(img_metas)
        multi_level_approxs = self.approx_anchor_generator.grid_priors(featmap_sizes, device=device)
        approxs_list = [multi_level_approxs for _ in range(num_imgs)]
        inside_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            multi_level_approxs = approxs_list[img_id]
            multi_level_approx_flags = self.approx_anchor_generator.valid_flags(featmap_sizes, img_meta['pad_shape'], device=device)
            for i, flags in enumerate(multi_level_approx_flags):
                approxs = multi_level_approxs[i]
                inside_flags_list = []
                for i in range(self.approxs_per_octave):
                    split_valid_flags = flags[i::self.approxs_per_octave]
                    split_approxs = approxs[i::self.approxs_per_octave, :]
                    inside_flags = anchor_inside_flags(split_approxs, split_valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
                    inside_flags_list.append(inside_flags)
                inside_flags = torch.stack(inside_flags_list, 0).sum(dim=0) > 0
                multi_level_flags.append(inside_flags)
            inside_flag_list.append(multi_level_flags)
        return (approxs_list, inside_flag_list)

    def get_anchors(self, featmap_sizes, shape_preds, loc_preds, img_metas, use_loc_filter=False, device='cuda'):
        """Get squares according to feature map sizes and guided anchors.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            shape_preds (list[tensor]): Multi-level shape predictions.
            loc_preds (list[tensor]): Multi-level location predictions.
            img_metas (list[dict]): Image meta info.
            use_loc_filter (bool): Use loc filter or not.
            device (torch.device | str): device for returned tensors

        Returns:
            tuple: square approxs of each image, guided anchors of each image,
                loc masks of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)
        multi_level_squares = self.square_anchor_generator.grid_priors(featmap_sizes, device=device)
        squares_list = [multi_level_squares for _ in range(num_imgs)]
        guided_anchors_list = []
        loc_mask_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_guided_anchors = []
            multi_level_loc_mask = []
            for i in range(num_levels):
                squares = squares_list[img_id][i]
                shape_pred = shape_preds[i][img_id]
                loc_pred = loc_preds[i][img_id]
                guided_anchors, loc_mask = self._get_guided_anchors_single(squares, shape_pred, loc_pred, use_loc_filter=use_loc_filter)
                multi_level_guided_anchors.append(guided_anchors)
                multi_level_loc_mask.append(loc_mask)
            guided_anchors_list.append(multi_level_guided_anchors)
            loc_mask_list.append(multi_level_loc_mask)
        return (squares_list, guided_anchors_list, loc_mask_list)

    def _get_guided_anchors_single(self, squares, shape_pred, loc_pred, use_loc_filter=False):
        """Get guided anchors and loc masks for a single level.

        Args:
            square (tensor): Squares of a single level.
            shape_pred (tensor): Shape predictions of a single level.
            loc_pred (tensor): Loc predictions of a single level.
            use_loc_filter (list[tensor]): Use loc filter or not.

        Returns:
            tuple: guided anchors, location masks
        """
        loc_pred = loc_pred.sigmoid().detach()
        if use_loc_filter:
            loc_mask = loc_pred >= self.loc_filter_thr
        else:
            loc_mask = loc_pred >= 0.0
        mask = loc_mask.permute(1, 2, 0).expand(-1, -1, self.num_base_priors)
        mask = mask.contiguous().view(-1)
        squares = squares[mask]
        anchor_deltas = shape_pred.permute(1, 2, 0).contiguous().view(-1, 2).detach()[mask]
        bbox_deltas = anchor_deltas.new_full(squares.size(), 0)
        bbox_deltas[:, 2:] = anchor_deltas
        guided_anchors = self.anchor_coder.decode(squares, bbox_deltas, wh_ratio_clip=1e-06)
        return (guided_anchors, mask)

    def ga_loc_targets(self, gt_bboxes_list, featmap_sizes):
        """Compute location targets for guided anchoring.

        Each feature map is divided into positive, negative and ignore regions.
        - positive regions: target 1, weight 1
        - ignore regions: target 0, weight 0
        - negative regions: target 0, weight 0.1

        Args:
            gt_bboxes_list (list[Tensor]): Gt bboxes of each image.
            featmap_sizes (list[tuple]): Multi level sizes of each feature
                maps.

        Returns:
            tuple
        """
        anchor_scale = self.approx_anchor_generator.octave_base_scale
        anchor_strides = self.approx_anchor_generator.strides
        for stride in anchor_strides:
            assert stride[0] == stride[1]
        anchor_strides = [stride[0] for stride in anchor_strides]
        center_ratio = self.train_cfg.center_ratio
        ignore_ratio = self.train_cfg.ignore_ratio
        img_per_gpu = len(gt_bboxes_list)
        num_lvls = len(featmap_sizes)
        r1 = (1 - center_ratio) / 2
        r2 = (1 - ignore_ratio) / 2
        all_loc_targets = []
        all_loc_weights = []
        all_ignore_map = []
        for lvl_id in range(num_lvls):
            h, w = featmap_sizes[lvl_id]
            loc_targets = torch.zeros(img_per_gpu, 1, h, w, device=gt_bboxes_list[0].device, dtype=torch.float32)
            loc_weights = torch.full_like(loc_targets, -1)
            ignore_map = torch.zeros_like(loc_targets)
            all_loc_targets.append(loc_targets)
            all_loc_weights.append(loc_weights)
            all_ignore_map.append(ignore_map)
        for img_id in range(img_per_gpu):
            gt_bboxes = gt_bboxes_list[img_id]
            scale = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1]))
            min_anchor_size = scale.new_full((1,), float(anchor_scale * anchor_strides[0]))
            target_lvls = torch.floor(torch.log2(scale) - torch.log2(min_anchor_size) + 0.5)
            target_lvls = target_lvls.clamp(min=0, max=num_lvls - 1).long()
            for gt_id in range(gt_bboxes.size(0)):
                lvl = target_lvls[gt_id].item()
                gt_ = gt_bboxes[gt_id, :4] / anchor_strides[lvl]
                ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(gt_, r2, featmap_sizes[lvl])
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = calc_region(gt_, r1, featmap_sizes[lvl])
                all_loc_targets[lvl][img_id, 0, ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1
                all_loc_weights[lvl][img_id, 0, ignore_y1:ignore_y2 + 1, ignore_x1:ignore_x2 + 1] = 0
                all_loc_weights[lvl][img_id, 0, ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1
                if lvl > 0:
                    d_lvl = lvl - 1
                    gt_ = gt_bboxes[gt_id, :4] / anchor_strides[d_lvl]
                    ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(gt_, r2, featmap_sizes[d_lvl])
                    all_ignore_map[d_lvl][img_id, 0, ignore_y1:ignore_y2 + 1, ignore_x1:ignore_x2 + 1] = 1
                if lvl < num_lvls - 1:
                    u_lvl = lvl + 1
                    gt_ = gt_bboxes[gt_id, :4] / anchor_strides[u_lvl]
                    ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(gt_, r2, featmap_sizes[u_lvl])
                    all_ignore_map[u_lvl][img_id, 0, ignore_y1:ignore_y2 + 1, ignore_x1:ignore_x2 + 1] = 1
        for lvl_id in range(num_lvls):
            all_loc_weights[lvl_id][(all_loc_weights[lvl_id] < 0) & (all_ignore_map[lvl_id] > 0)] = 0
            all_loc_weights[lvl_id][all_loc_weights[lvl_id] < 0] = 0.1
        loc_avg_factor = sum([t.size(0) * t.size(-1) * t.size(-2) for t in all_loc_targets]) / 200
        return (all_loc_targets, all_loc_weights, loc_avg_factor)

    def _ga_shape_target_single(self, flat_approxs, inside_flags, flat_squares, gt_bboxes, gt_bboxes_ignore, img_meta, unmap_outputs=True):
        """Compute guided anchoring targets.

        This function returns sampled anchors and gt bboxes directly
        rather than calculates regression targets.

        Args:
            flat_approxs (Tensor): flat approxs of a single image,
                shape (n, 4)
            inside_flags (Tensor): inside flags of a single image,
                shape (n, ).
            flat_squares (Tensor): flat squares of a single image,
                shape (approxs_per_octave * n, 4)
            gt_bboxes (Tensor): Ground truth bboxes of a single image.
            img_meta (dict): Meta info of a single image.
            approxs_per_octave (int): number of approxs per octave
            cfg (dict): RPN train configs.
            unmap_outputs (bool): unmap outputs or not.

        Returns:
            tuple
        """
        if not inside_flags.any():
            return (None,) * 5
        expand_inside_flags = inside_flags[:, None].expand(-1, self.approxs_per_octave).reshape(-1)
        approxs = flat_approxs[expand_inside_flags, :]
        squares = flat_squares[inside_flags, :]
        assign_result = self.ga_assigner.assign(approxs, squares, self.approxs_per_octave, gt_bboxes, gt_bboxes_ignore)
        sampling_result = self.ga_sampler.sample(assign_result, squares, gt_bboxes)
        bbox_anchors = torch.zeros_like(squares)
        bbox_gts = torch.zeros_like(squares)
        bbox_weights = torch.zeros_like(squares)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            bbox_anchors[pos_inds, :] = sampling_result.pos_bboxes
            bbox_gts[pos_inds, :] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds, :] = 1.0
        if unmap_outputs:
            num_total_anchors = flat_squares.size(0)
            bbox_anchors = unmap(bbox_anchors, num_total_anchors, inside_flags)
            bbox_gts = unmap(bbox_gts, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        return (bbox_anchors, bbox_gts, bbox_weights, pos_inds, neg_inds)

    def ga_shape_targets(self, approx_list, inside_flag_list, square_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, unmap_outputs=True):
        """Compute guided anchoring targets.

        Args:
            approx_list (list[list]): Multi level approxs of each image.
            inside_flag_list (list[list]): Multi level inside flags of each
                image.
            square_list (list[list]): Multi level squares of each image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): ignore list of gt bboxes.
            unmap_outputs (bool): unmap outputs or not.

        Returns:
            tuple
        """
        num_imgs = len(img_metas)
        assert len(approx_list) == len(inside_flag_list) == len(square_list) == num_imgs
        num_level_squares = [squares.size(0) for squares in square_list[0]]
        inside_flag_flat_list = []
        approx_flat_list = []
        square_flat_list = []
        for i in range(num_imgs):
            assert len(square_list[i]) == len(inside_flag_list[i])
            inside_flag_flat_list.append(torch.cat(inside_flag_list[i]))
            approx_flat_list.append(torch.cat(approx_list[i]))
            square_flat_list.append(torch.cat(square_list[i]))
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        all_bbox_anchors, all_bbox_gts, all_bbox_weights, pos_inds_list, neg_inds_list = multi_apply(self._ga_shape_target_single, approx_flat_list, inside_flag_flat_list, square_flat_list, gt_bboxes_list, gt_bboxes_ignore_list, img_metas, unmap_outputs=unmap_outputs)
        if any([bbox_anchors is None for bbox_anchors in all_bbox_anchors]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        bbox_anchors_list = images_to_levels(all_bbox_anchors, num_level_squares)
        bbox_gts_list = images_to_levels(all_bbox_gts, num_level_squares)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_squares)
        return (bbox_anchors_list, bbox_gts_list, bbox_weights_list, num_total_pos, num_total_neg)

    def loss_shape_single(self, shape_pred, bbox_anchors, bbox_gts, anchor_weights, anchor_total_num):
        shape_pred = shape_pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        bbox_anchors = bbox_anchors.contiguous().view(-1, 4)
        bbox_gts = bbox_gts.contiguous().view(-1, 4)
        anchor_weights = anchor_weights.contiguous().view(-1, 4)
        bbox_deltas = bbox_anchors.new_full(bbox_anchors.size(), 0)
        bbox_deltas[:, 2:] += shape_pred
        inds = torch.nonzero(anchor_weights[:, 0] > 0, as_tuple=False).squeeze(1)
        bbox_deltas_ = bbox_deltas[inds]
        bbox_anchors_ = bbox_anchors[inds]
        bbox_gts_ = bbox_gts[inds]
        anchor_weights_ = anchor_weights[inds]
        pred_anchors_ = self.anchor_coder.decode(bbox_anchors_, bbox_deltas_, wh_ratio_clip=1e-06)
        loss_shape = self.loss_shape(pred_anchors_, bbox_gts_, anchor_weights_, avg_factor=anchor_total_num)
        return loss_shape

    def loss_loc_single(self, loc_pred, loc_target, loc_weight, loc_avg_factor):
        loss_loc = self.loss_loc(loc_pred.reshape(-1, 1), loc_target.reshape(-1).long(), loc_weight.reshape(-1), avg_factor=loc_avg_factor)
        return loss_loc

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'shape_preds', 'loc_preds'))
    def loss(self, cls_scores, bbox_preds, shape_preds, loc_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.approx_anchor_generator.num_levels
        device = cls_scores[0].device
        loc_targets, loc_weights, loc_avg_factor = self.ga_loc_targets(gt_bboxes, featmap_sizes)
        approxs_list, inside_flag_list = self.get_sampled_approxs(featmap_sizes, img_metas, device=device)
        squares_list, guided_anchors_list, _ = self.get_anchors(featmap_sizes, shape_preds, loc_preds, img_metas, device=device)
        shape_targets = self.ga_shape_targets(approxs_list, inside_flag_list, squares_list, gt_bboxes, img_metas)
        if shape_targets is None:
            return None
        bbox_anchors_list, bbox_gts_list, anchor_weights_list, anchor_fg_num, anchor_bg_num = shape_targets
        anchor_total_num = anchor_fg_num if not self.ga_sampling else anchor_fg_num + anchor_bg_num
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(guided_anchors_list, inside_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos
        num_level_anchors = [anchors.size(0) for anchors in guided_anchors_list[0]]
        concat_anchor_list = []
        for i in range(len(guided_anchors_list)):
            concat_anchor_list.append(torch.cat(guided_anchors_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)
        losses_cls, losses_bbox = multi_apply(self.loss_single, cls_scores, bbox_preds, all_anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_samples=num_total_samples)
        losses_loc = []
        for i in range(len(loc_preds)):
            loss_loc = self.loss_loc_single(loc_preds[i], loc_targets[i], loc_weights[i], loc_avg_factor=loc_avg_factor)
            losses_loc.append(loss_loc)
        losses_shape = []
        for i in range(len(shape_preds)):
            loss_shape = self.loss_shape_single(shape_preds[i], bbox_anchors_list[i], bbox_gts_list[i], anchor_weights_list[i], anchor_total_num=anchor_total_num)
            losses_shape.append(loss_shape)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_shape=losses_shape, loss_loc=losses_loc)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'shape_preds', 'loc_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, shape_preds, loc_preds, img_metas, cfg=None, rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(shape_preds) == len(loc_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        _, guided_anchors, loc_masks = self.get_anchors(featmap_sizes, shape_preds, loc_preds, img_metas, use_loc_filter=not self.training, device=device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            guided_anchor_list = [guided_anchors[img_id][i].detach() for i in range(num_levels)]
            loc_mask_list = [loc_masks[img_id][i].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list, guided_anchor_list, loc_mask_list, img_shape, scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self, cls_scores, bbox_preds, mlvl_anchors, mlvl_masks, img_shape, scale_factor, cfg, rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors, mask in zip(cls_scores, bbox_preds, mlvl_anchors, mlvl_masks):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            if mask.sum() == 0:
                continue
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            scores = scores[mask, :]
            bbox_pred = bbox_pred[mask, :]
            if scores.dim() == 0:
                anchors = anchors.unsqueeze(0)
                scores = scores.unsqueeze(0)
                bbox_pred = bbox_pred.unsqueeze(0)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return (det_bboxes, det_labels)

@HEADS.register_module()
class GFLHead(AnchorHead):
    """Generalized Focal Loss: Learning Qualified and Distributed Bounding
    Boxes for Dense Object Detection.

    GFL head structure is similar with ATSS, however GFL uses
    1) joint representation for classification and localization quality, and
    2) flexible General distribution for bounding box locations,
    which are supervised by
    Quality Focal Loss (QFL) and Distribution Focal Loss (DFL), respectively

    https://arxiv.org/abs/2006.04388

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 4.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='GN', num_groups=32, requires_grad=True).
        loss_qfl (dict): Config of Quality Focal Loss (QFL).
        bbox_coder (dict): Config of bbox coder. Defaults
            'DistancePointBBoxCoder'.
        reg_max (int): Max value of integral set :math: `{0, ..., reg_max}`
            in QFL setting. Default: 16.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    Example:
        >>> self = GFLHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_quality_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_quality_score) == len(self.scales)
    """

    def __init__(self, num_classes, in_channels, stacked_convs=4, conv_cfg=None, norm_cfg=dict(type='GN', num_groups=32, requires_grad=True), loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25), bbox_coder=dict(type='DistancePointBBoxCoder'), reg_max=16, init_cfg=dict(type='Normal', layer='Conv2d', std=0.01, override=dict(type='Normal', name='gfl_cls', std=0.01, bias_prob=0.01)), **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        super(GFLHead, self).__init__(num_classes, in_channels, bbox_coder=bbox_coder, init_cfg=init_cfg, **kwargs)
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.integral = Integral(self.reg_max)
        self.loss_dfl = build_loss(loss_dfl)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        assert self.num_anchors == 1, 'anchor free version'
        self.gfl_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.gfl_reg = nn.Conv2d(self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.prior_generator.strides])

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification and quality (IoU)
                    joint scores for all scale levels, each is a 4D-tensor,
                    the channel number is num_classes.
                bbox_preds (list[Tensor]): Box distribution logits for all
                    scale levels, each is a 4D-tensor, the channel number is
                    4*(n+1), n is max value of integral set.
        """
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls and quality joint scores for a single
                    scale level the channel number is num_classes.
                bbox_pred (Tensor): Box distribution logits for a single scale
                    level, the channel number is 4*(n+1), n is max value of
                    integral set.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.gfl_cls(cls_feat)
        bbox_pred = scale(self.gfl_reg(reg_feat)).float()
        return (cls_score, bbox_pred)

    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[..., 2] + anchors[..., 0]) / 2
        anchors_cy = (anchors[..., 3] + anchors[..., 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def loss_single(self, anchors, cls_score, bbox_pred, labels, label_weights, bbox_targets, stride, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]
            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = self.bbox_coder.decode(pos_anchor_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = self.bbox_coder.encode(pos_anchor_centers, pos_decode_bbox_targets, self.reg_max).reshape(-1)
            loss_bbox = self.loss_bbox(pos_decode_bbox_pred, pos_decode_bbox_targets, weight=weight_targets, avg_factor=1.0)
            loss_dfl = self.loss_dfl(pred_corners, target_corners, weight=weight_targets[:, None].expand(-1, 4).reshape(-1), avg_factor=4.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)
        loss_cls = self.loss_cls(cls_score, (labels, score), weight=label_weights, avg_factor=num_total_samples)
        return (loss_cls, loss_bbox, loss_dfl, weight_targets.sum())

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        num_total_samples = reduce_mean(torch.tensor(num_total_pos, dtype=torch.float, device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)
        losses_cls, losses_bbox, losses_dfl, avg_factor = multi_apply(self.loss_single, anchor_list, cls_scores, bbox_preds, labels_list, label_weights_list, bbox_targets_list, self.prior_generator.strides, num_total_samples=num_total_samples)
        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

    def _get_bboxes_single(self, cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors, img_meta, cfg, rescale=False, with_nms=True, **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. GFL head does not need this value.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape                     [num_bboxes, 5], where the first 4 columns are bounding                     box positions (tl_x, tl_y, br_x, br_y) and the 5-th                     column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding                     box with shape [num_bboxes].
        """
        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, stride, priors) in enumerate(zip(cls_score_list, bbox_pred_list, self.prior_generator.strides, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert stride[0] == stride[1]
            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.integral(bbox_pred) * stride[0]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            results = filter_scores_and_topk(scores, cfg.score_thr, nms_pre, dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, _, filtered_results = results
            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            bboxes = self.bbox_coder.decode(self.anchor_center(priors), bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes, img_meta['scale_factor'], cfg, rescale=rescale, with_nms=with_nms)

    def get_targets(self, anchor_list, valid_flag_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, unmap_outputs=True):
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, pos_inds_list, neg_inds_list = multi_apply(self._get_target_single, anchor_list, valid_flag_list, num_level_anchors_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, label_channels=label_channels, unmap_outputs=unmap_outputs)
        if any([labels is None for labels in all_labels]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        return (anchors_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, flat_anchors, valid_flags, num_level_anchors, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, label_channels=1, unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4).
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7
        anchors = flat_anchors[inside_flags, :]
        num_level_anchors_inside = self.get_num_level_anchors_inside(num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore, gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        return (anchors, labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [int(flags.sum()) for flags in split_inside_flags]
        return num_level_anchors_inside

@HEADS.register_module()
class FSAFHead(RetinaHead):
    """Anchor-free head used in `FSAF <https://arxiv.org/abs/1903.00621>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors (num_anchors is 1 for anchor-
    free methods)

    Args:
        *args: Same as its base class in :class:`RetinaHead`
        score_threshold (float, optional): The score_threshold to calculate
            positive recall. If given, prediction scores lower than this value
            is counted as incorrect prediction. Default to None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
        **kwargs: Same as its base class in :class:`RetinaHead`

    Example:
        >>> import torch
        >>> self = FSAFHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == self.num_classes
        >>> assert box_per_anchor == 4
    """

    def __init__(self, *args, score_threshold=None, init_cfg=None, **kwargs):
        if init_cfg is None:
            init_cfg = dict(type='Normal', layer='Conv2d', std=0.01, override=[dict(type='Normal', name='retina_cls', std=0.01, bias_prob=0.01), dict(type='Normal', name='retina_reg', std=0.01, bias=0.25)])
        super().__init__(*args, init_cfg=init_cfg, **kwargs)
        self.score_threshold = score_threshold

    def forward_single(self, x):
        """Forward feature map of a single scale level.

        Args:
            x (Tensor): Feature map of a single scale level.

        Returns:
            tuple (Tensor):
                cls_score (Tensor): Box scores for each scale level
                    Has shape (N, num_points * num_classes, H, W).
                bbox_pred (Tensor): Box energies / deltas for each scale
                    level with shape (N, num_points * 4, H, W).
        """
        cls_score, bbox_pred = super().forward_single(x)
        return (cls_score, self.relu(bbox_pred))

    def _get_targets_single(self, flat_anchors, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, label_channels=1, unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Most of the codes are the same with the base class
          :obj: `AnchorHead`, except that it also collects and returns
          the matched gt index in the image (from 0 to num_gt-1). If the
          anchor bbox is not matched to any gt, the corresponding value in
          pos_gt_inds is -1.
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7
        anchors = flat_anchors[inside_flags.type(torch.bool), :]
        assign_result = self.assigner.assign(anchors, gt_bboxes, gt_bboxes_ignore, None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros((num_valid_anchors, label_channels), dtype=torch.float)
        pos_gt_inds = anchors.new_full((num_valid_anchors,), -1, dtype=torch.long)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            pos_gt_inds[pos_inds] = sampling_result.pos_assigned_gt_inds
            if gt_labels is None:
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        shadowed_labels = assign_result.get_extra_property('shadowed_labels')
        if shadowed_labels is not None and shadowed_labels.numel():
            if len(shadowed_labels.shape) == 2:
                idx_, label_ = (shadowed_labels[:, 0], shadowed_labels[:, 1])
                assert (labels[idx_] != label_).all(), 'One label cannot be both positive and ignored'
                label_weights[idx_, label_] = 0
            else:
                label_weights[shadowed_labels] = 0
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            pos_gt_inds = unmap(pos_gt_inds, num_total_anchors, inside_flags, fill=-1)
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds, sampling_result, pos_gt_inds)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        for i in range(len(bbox_preds)):
            bbox_preds[i] = bbox_preds[i].clamp(min=0.0001)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        batch_size = len(gt_bboxes)
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg, pos_assigned_gt_inds_list = cls_reg_targets
        num_gts = np.array(list(map(len, gt_labels)))
        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)
        losses_cls, losses_bbox = multi_apply(self.loss_single, cls_scores, bbox_preds, all_anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_samples=num_total_samples)
        cum_num_gts = list(np.cumsum(num_gts))
        for i, assign in enumerate(pos_assigned_gt_inds_list):
            for j in range(1, batch_size):
                assign[j][assign[j] >= 0] += int(cum_num_gts[j - 1])
            pos_assigned_gt_inds_list[i] = assign.flatten()
            labels_list[i] = labels_list[i].flatten()
        num_gts = sum(map(len, gt_labels))
        label_sequence = torch.arange(num_gts, device=device)
        with torch.no_grad():
            loss_levels, = multi_apply(self.collect_loss_level_single, losses_cls, losses_bbox, pos_assigned_gt_inds_list, labels_seq=label_sequence)
            loss_levels = torch.stack(loss_levels, dim=0)
            if loss_levels.numel() == 0:
                argmin = loss_levels.new_empty((num_gts,), dtype=torch.long)
            else:
                _, argmin = loss_levels.min(dim=0)
        losses_cls, losses_bbox, pos_inds = multi_apply(self.reweight_loss_single, losses_cls, losses_bbox, pos_assigned_gt_inds_list, labels_list, list(range(len(losses_cls))), min_levels=argmin)
        num_pos = torch.cat(pos_inds, 0).sum().float()
        pos_recall = self.calculate_pos_recall(cls_scores, labels_list, pos_inds)
        if num_pos == 0:
            avg_factor = num_pos + float(num_total_neg)
        else:
            avg_factor = num_pos
        for i in range(len(losses_cls)):
            losses_cls[i] /= avg_factor
            losses_bbox[i] /= avg_factor
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, num_pos=num_pos / batch_size, pos_recall=pos_recall)

    def calculate_pos_recall(self, cls_scores, labels_list, pos_inds):
        """Calculate positive recall with score threshold.

        Args:
            cls_scores (list[Tensor]): Classification scores at all fpn levels.
                Each tensor is in shape (N, num_classes * num_anchors, H, W)
            labels_list (list[Tensor]): The label that each anchor is assigned
                to. Shape (N * H * W * num_anchors, )
            pos_inds (list[Tensor]): List of bool tensors indicating whether
                the anchor is assigned to a positive label.
                Shape (N * H * W * num_anchors, )

        Returns:
            Tensor: A single float number indicating the positive recall.
        """
        with torch.no_grad():
            num_class = self.num_classes
            scores = [cls.permute(0, 2, 3, 1).reshape(-1, num_class)[pos] for cls, pos in zip(cls_scores, pos_inds)]
            labels = [label.reshape(-1)[pos] for label, pos in zip(labels_list, pos_inds)]
            scores = torch.cat(scores, dim=0)
            labels = torch.cat(labels, dim=0)
            if self.use_sigmoid_cls:
                scores = scores.sigmoid()
            else:
                scores = scores.softmax(dim=1)
            return accuracy(scores, labels, thresh=self.score_threshold)

    def collect_loss_level_single(self, cls_loss, reg_loss, assigned_gt_inds, labels_seq):
        """Get the average loss in each FPN level w.r.t. each gt label.

        Args:
            cls_loss (Tensor): Classification loss of each feature map pixel,
              shape (num_anchor, num_class)
            reg_loss (Tensor): Regression loss of each feature map pixel,
              shape (num_anchor, 4)
            assigned_gt_inds (Tensor): It indicates which gt the prior is
              assigned to (0-based, -1: no assignment). shape (num_anchor),
            labels_seq: The rank of labels. shape (num_gt)

        Returns:
            shape: (num_gt), average loss of each gt in this level
        """
        if len(reg_loss.shape) == 2:
            reg_loss = reg_loss.sum(dim=-1)
        if len(cls_loss.shape) == 2:
            cls_loss = cls_loss.sum(dim=-1)
        loss = cls_loss + reg_loss
        assert loss.size(0) == assigned_gt_inds.size(0)
        losses_ = loss.new_full(labels_seq.shape, 1000000.0)
        for i, l in enumerate(labels_seq):
            match = assigned_gt_inds == l
            if match.any():
                losses_[i] = loss[match].mean()
        return (losses_,)

    def reweight_loss_single(self, cls_loss, reg_loss, assigned_gt_inds, labels, level, min_levels):
        """Reweight loss values at each level.

        Reassign loss values at each level by masking those where the
        pre-calculated loss is too large. Then return the reduced losses.

        Args:
            cls_loss (Tensor): Element-wise classification loss.
              Shape: (num_anchors, num_classes)
            reg_loss (Tensor): Element-wise regression loss.
              Shape: (num_anchors, 4)
            assigned_gt_inds (Tensor): The gt indices that each anchor bbox
              is assigned to. -1 denotes a negative anchor, otherwise it is the
              gt index (0-based). Shape: (num_anchors, ),
            labels (Tensor): Label assigned to anchors. Shape: (num_anchors, ).
            level (int): The current level index in the pyramid
              (0-4 for RetinaNet)
            min_levels (Tensor): The best-matching level for each gt.
              Shape: (num_gts, ),

        Returns:
            tuple:
                - cls_loss: Reduced corrected classification loss. Scalar.
                - reg_loss: Reduced corrected regression loss. Scalar.
                - pos_flags (Tensor): Corrected bool tensor indicating the
                  final positive anchors. Shape: (num_anchors, ).
        """
        loc_weight = torch.ones_like(reg_loss)
        cls_weight = torch.ones_like(cls_loss)
        pos_flags = assigned_gt_inds >= 0
        pos_indices = torch.nonzero(pos_flags, as_tuple=False).flatten()
        if pos_flags.any():
            pos_assigned_gt_inds = assigned_gt_inds[pos_flags]
            zeroing_indices = min_levels[pos_assigned_gt_inds] != level
            neg_indices = pos_indices[zeroing_indices]
            if neg_indices.numel():
                pos_flags[neg_indices] = 0
                loc_weight[neg_indices] = 0
                zeroing_labels = labels[neg_indices]
                assert (zeroing_labels >= 0).all()
                cls_weight[neg_indices, zeroing_labels] = 0
        cls_loss = weight_reduce_loss(cls_loss, cls_weight, reduction='sum')
        reg_loss = weight_reduce_loss(reg_loss, loc_weight, reduction='sum')
        return (cls_loss, reg_loss, pos_flags)

