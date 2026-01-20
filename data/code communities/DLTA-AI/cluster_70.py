# Cluster 70

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

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains             a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        level_targets.append(target[:, start:end])
        start = end
    return level_targets

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

def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor

@HEADS.register_module()
class CenterNetHead(BaseDenseHead, BBoxTestMixin):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self, in_channel, feat_channel, num_classes, loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0), loss_wh=dict(type='L1Loss', loss_weight=0.1), loss_offset=dict(type='L1Loss', loss_weight=1.0), train_cfg=None, test_cfg=None, init_cfg=None):
        super(CenterNetHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel, num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)
        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        return (center_heatmap_pred, wh_pred, offset_pred)

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def loss(self, center_heatmap_preds, wh_preds, offset_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
               shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 2, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1
        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]
        target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels, center_heatmap_pred.shape, img_metas[0]['pad_shape'])
        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']
        loss_center_heatmap = self.loss_center_heatmap(center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)
        loss_wh = self.loss_wh(wh_pred, wh_target, wh_offset_target_weight, avg_factor=avg_factor * 2)
        loss_offset = self.loss_offset(offset_pred, offset_target, wh_offset_target_weight, avg_factor=avg_factor * 2)
        return dict(loss_center_heatmap=loss_center_heatmap, loss_wh=loss_wh, loss_offset=loss_offset)

    def get_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap,                    shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape                    (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape                    (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset                    predict, shape (B, 2, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape
        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)
        center_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_classes, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)
            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind], [ctx_int, cty_int], radius)
                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h
                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int
                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1
        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(center_heatmap_target=center_heatmap_target, wh_target=wh_target, offset_target=offset_target, wh_offset_target_weight=wh_offset_target_weight)
        return (target_result, avg_factor)

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def get_bboxes(self, center_heatmap_preds, wh_preds, offset_preds, img_metas, rescale=True, with_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): Center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): WH predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): Offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(self._get_bboxes_single(center_heatmap_preds[0][img_id:img_id + 1, ...], wh_preds[0][img_id:img_id + 1, ...], offset_preds[0][img_id:img_id + 1, ...], img_metas[img_id], rescale=rescale, with_nms=with_nms))
        return result_list

    def _get_bboxes_single(self, center_heatmap_pred, wh_pred, offset_pred, img_meta, rescale=False, with_nms=True):
        """Transform outputs of a single image into bbox results.

        Args:
            center_heatmap_pred (Tensor): Center heatmap for current level with
                shape (1, num_classes, H, W).
            wh_pred (Tensor): WH heatmap for current level with shape
                (1, num_classes, H, W).
            offset_pred (Tensor): Offset for current level with shape
                (1, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor, Tensor]: The first item is an (n, 5) tensor, where
                5 represent (tl_x, tl_y, br_x, br_y, score) and the score
                between 0 and 1. The shape of the second tensor in the tuple
                is (n,), and each element represents the class label of the
                corresponding box.
        """
        batch_det_bboxes, batch_labels = self.decode_heatmap(center_heatmap_pred, wh_pred, offset_pred, img_meta['batch_input_shape'], k=self.test_cfg.topk, kernel=self.test_cfg.local_maximum_kernel)
        det_bboxes = batch_det_bboxes.view([-1, 5])
        det_labels = batch_labels.view(-1)
        batch_border = det_bboxes.new_tensor(img_meta['border'])[..., [2, 0, 2, 0]]
        det_bboxes[..., :4] -= batch_border
        if rescale:
            det_bboxes[..., :4] /= det_bboxes.new_tensor(img_meta['scale_factor'])
        if with_nms:
            det_bboxes, det_labels = self._bboxes_nms(det_bboxes, det_labels, self.test_cfg)
        return (det_bboxes, det_labels)

    def decode_heatmap(self, center_heatmap_pred, wh_pred, offset_pred, img_shape, k=100, kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with                   shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape
        center_heatmap_pred = get_local_maximum(center_heatmap_pred, kernel=kernel)
        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets
        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)
        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]), dim=-1)
        return (batch_bboxes, batch_topk_labels)

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() > 0:
            max_num = cfg.max_per_img
            bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1].contiguous(), labels, cfg.nms)
            if max_num > 0:
                bboxes = bboxes[:max_num]
                labels = labels[keep][:max_num]
        return (bboxes, labels)

@HEADS.register_module()
class CentripetalHead(CornerHead):
    """Head of CentripetalNet: Pursuing High-quality Keypoint Pairs for Object
    Detection.

    CentripetalHead inherits from :class:`CornerHead`. It removes the
    embedding branch and adds guiding shift and centripetal shift branches.
    More details can be found in the `paper
    <https://arxiv.org/abs/2003.09119>`_ .

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        num_feat_levels (int): Levels of feature from the previous module. 2
            for HourglassNet-104 and 1 for HourglassNet-52. HourglassNet-104
            outputs the final feature and intermediate supervision feature and
            HourglassNet-52 only outputs the final feature. Default: 2.
        corner_emb_channels (int): Channel of embedding vector. Default: 1.
        train_cfg (dict | None): Training config. Useless in CornerHead,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CornerHead. Default: None.
        loss_heatmap (dict | None): Config of corner heatmap loss. Default:
            GaussianFocalLoss.
        loss_embedding (dict | None): Config of corner embedding loss. Default:
            AssociativeEmbeddingLoss.
        loss_offset (dict | None): Config of corner offset loss. Default:
            SmoothL1Loss.
        loss_guiding_shift (dict): Config of guiding shift loss. Default:
            SmoothL1Loss.
        loss_centripetal_shift (dict): Config of centripetal shift loss.
            Default: SmoothL1Loss.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self, *args, centripetal_shift_channels=2, guiding_shift_channels=2, feat_adaption_conv_kernel=3, loss_guiding_shift=dict(type='SmoothL1Loss', beta=1.0, loss_weight=0.05), loss_centripetal_shift=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1), init_cfg=None, **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization behavior, init_cfg is not allowed to be set'
        assert centripetal_shift_channels == 2, 'CentripetalHead only support centripetal_shift_channels == 2'
        self.centripetal_shift_channels = centripetal_shift_channels
        assert guiding_shift_channels == 2, 'CentripetalHead only support guiding_shift_channels == 2'
        self.guiding_shift_channels = guiding_shift_channels
        self.feat_adaption_conv_kernel = feat_adaption_conv_kernel
        super(CentripetalHead, self).__init__(*args, init_cfg=init_cfg, **kwargs)
        self.loss_guiding_shift = build_loss(loss_guiding_shift)
        self.loss_centripetal_shift = build_loss(loss_centripetal_shift)

    def _init_centripetal_layers(self):
        """Initialize centripetal layers.

        Including feature adaption deform convs (feat_adaption), deform offset
        prediction convs (dcn_off), guiding shift (guiding_shift) and
        centripetal shift ( centripetal_shift). Each branch has two parts:
        prefix `tl_` for top-left and `br_` for bottom-right.
        """
        self.tl_feat_adaption = nn.ModuleList()
        self.br_feat_adaption = nn.ModuleList()
        self.tl_dcn_offset = nn.ModuleList()
        self.br_dcn_offset = nn.ModuleList()
        self.tl_guiding_shift = nn.ModuleList()
        self.br_guiding_shift = nn.ModuleList()
        self.tl_centripetal_shift = nn.ModuleList()
        self.br_centripetal_shift = nn.ModuleList()
        for _ in range(self.num_feat_levels):
            self.tl_feat_adaption.append(DeformConv2d(self.in_channels, self.in_channels, self.feat_adaption_conv_kernel, 1, 1))
            self.br_feat_adaption.append(DeformConv2d(self.in_channels, self.in_channels, self.feat_adaption_conv_kernel, 1, 1))
            self.tl_guiding_shift.append(self._make_layers(out_channels=self.guiding_shift_channels, in_channels=self.in_channels))
            self.br_guiding_shift.append(self._make_layers(out_channels=self.guiding_shift_channels, in_channels=self.in_channels))
            self.tl_dcn_offset.append(ConvModule(self.guiding_shift_channels, self.feat_adaption_conv_kernel ** 2 * self.guiding_shift_channels, 1, bias=False, act_cfg=None))
            self.br_dcn_offset.append(ConvModule(self.guiding_shift_channels, self.feat_adaption_conv_kernel ** 2 * self.guiding_shift_channels, 1, bias=False, act_cfg=None))
            self.tl_centripetal_shift.append(self._make_layers(out_channels=self.centripetal_shift_channels, in_channels=self.in_channels))
            self.br_centripetal_shift.append(self._make_layers(out_channels=self.centripetal_shift_channels, in_channels=self.in_channels))

    def _init_layers(self):
        """Initialize layers for CentripetalHead.

        Including two parts: CornerHead layers and CentripetalHead layers
        """
        super()._init_layers()
        self._init_centripetal_layers()

    def init_weights(self):
        super(CentripetalHead, self).init_weights()
        for i in range(self.num_feat_levels):
            normal_init(self.tl_feat_adaption[i], std=0.01)
            normal_init(self.br_feat_adaption[i], std=0.01)
            normal_init(self.tl_dcn_offset[i].conv, std=0.1)
            normal_init(self.br_dcn_offset[i].conv, std=0.1)
            _ = [x.conv.reset_parameters() for x in self.tl_guiding_shift[i]]
            _ = [x.conv.reset_parameters() for x in self.br_guiding_shift[i]]
            _ = [x.conv.reset_parameters() for x in self.tl_centripetal_shift[i]]
            _ = [x.conv.reset_parameters() for x in self.br_centripetal_shift[i]]

    def forward_single(self, x, lvl_ind):
        """Forward feature of a single level.

        Args:
            x (Tensor): Feature of a single level.
            lvl_ind (int): Level index of current feature.

        Returns:
            tuple[Tensor]: A tuple of CentripetalHead's output for current
            feature level. Containing the following Tensors:

                - tl_heat (Tensor): Predicted top-left corner heatmap.
                - br_heat (Tensor): Predicted bottom-right corner heatmap.
                - tl_off (Tensor): Predicted top-left offset heatmap.
                - br_off (Tensor): Predicted bottom-right offset heatmap.
                - tl_guiding_shift (Tensor): Predicted top-left guiding shift
                  heatmap.
                - br_guiding_shift (Tensor): Predicted bottom-right guiding
                  shift heatmap.
                - tl_centripetal_shift (Tensor): Predicted top-left centripetal
                  shift heatmap.
                - br_centripetal_shift (Tensor): Predicted bottom-right
                  centripetal shift heatmap.
        """
        tl_heat, br_heat, _, _, tl_off, br_off, tl_pool, br_pool = super().forward_single(x, lvl_ind, return_pool=True)
        tl_guiding_shift = self.tl_guiding_shift[lvl_ind](tl_pool)
        br_guiding_shift = self.br_guiding_shift[lvl_ind](br_pool)
        tl_dcn_offset = self.tl_dcn_offset[lvl_ind](tl_guiding_shift.detach())
        br_dcn_offset = self.br_dcn_offset[lvl_ind](br_guiding_shift.detach())
        tl_feat_adaption = self.tl_feat_adaption[lvl_ind](tl_pool, tl_dcn_offset)
        br_feat_adaption = self.br_feat_adaption[lvl_ind](br_pool, br_dcn_offset)
        tl_centripetal_shift = self.tl_centripetal_shift[lvl_ind](tl_feat_adaption)
        br_centripetal_shift = self.br_centripetal_shift[lvl_ind](br_feat_adaption)
        result_list = [tl_heat, br_heat, tl_off, br_off, tl_guiding_shift, br_guiding_shift, tl_centripetal_shift, br_centripetal_shift]
        return result_list

    @force_fp32()
    def loss(self, tl_heats, br_heats, tl_offs, br_offs, tl_guiding_shifts, br_guiding_shifts, tl_centripetal_shifts, br_centripetal_shifts, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            tl_heats (list[Tensor]): Top-left corner heatmaps for each level
                with shape (N, num_classes, H, W).
            br_heats (list[Tensor]): Bottom-right corner heatmaps for each
                level with shape (N, num_classes, H, W).
            tl_offs (list[Tensor]): Top-left corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            br_offs (list[Tensor]): Bottom-right corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            tl_guiding_shifts (list[Tensor]): Top-left guiding shifts for each
                level with shape (N, guiding_shift_channels, H, W).
            br_guiding_shifts (list[Tensor]): Bottom-right guiding shifts for
                each level with shape (N, guiding_shift_channels, H, W).
            tl_centripetal_shifts (list[Tensor]): Top-left centripetal shifts
                for each level with shape (N, centripetal_shift_channels, H,
                W).
            br_centripetal_shifts (list[Tensor]): Bottom-right centripetal
                shifts for each level with shape (N,
                centripetal_shift_channels, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [left, top, right, bottom] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components. Containing the
            following losses:

                - det_loss (list[Tensor]): Corner keypoint losses of all
                  feature levels.
                - off_loss (list[Tensor]): Corner offset losses of all feature
                  levels.
                - guiding_loss (list[Tensor]): Guiding shift losses of all
                  feature levels.
                - centripetal_loss (list[Tensor]): Centripetal shift losses of
                  all feature levels.
        """
        targets = self.get_targets(gt_bboxes, gt_labels, tl_heats[-1].shape, img_metas[0]['pad_shape'], with_corner_emb=self.with_corner_emb, with_guiding_shift=True, with_centripetal_shift=True)
        mlvl_targets = [targets for _ in range(self.num_feat_levels)]
        [det_losses, off_losses, guiding_losses, centripetal_losses] = multi_apply(self.loss_single, tl_heats, br_heats, tl_offs, br_offs, tl_guiding_shifts, br_guiding_shifts, tl_centripetal_shifts, br_centripetal_shifts, mlvl_targets)
        loss_dict = dict(det_loss=det_losses, off_loss=off_losses, guiding_loss=guiding_losses, centripetal_loss=centripetal_losses)
        return loss_dict

    def loss_single(self, tl_hmp, br_hmp, tl_off, br_off, tl_guiding_shift, br_guiding_shift, tl_centripetal_shift, br_centripetal_shift, targets):
        """Compute losses for single level.

        Args:
            tl_hmp (Tensor): Top-left corner heatmap for current level with
                shape (N, num_classes, H, W).
            br_hmp (Tensor): Bottom-right corner heatmap for current level with
                shape (N, num_classes, H, W).
            tl_off (Tensor): Top-left corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            br_off (Tensor): Bottom-right corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            tl_guiding_shift (Tensor): Top-left guiding shift for current level
                with shape (N, guiding_shift_channels, H, W).
            br_guiding_shift (Tensor): Bottom-right guiding shift for current
                level with shape (N, guiding_shift_channels, H, W).
            tl_centripetal_shift (Tensor): Top-left centripetal shift for
                current level with shape (N, centripetal_shift_channels, H, W).
            br_centripetal_shift (Tensor): Bottom-right centripetal shift for
                current level with shape (N, centripetal_shift_channels, H, W).
            targets (dict): Corner target generated by `get_targets`.

        Returns:
            tuple[torch.Tensor]: Losses of the head's different branches
            containing the following losses:

                - det_loss (Tensor): Corner keypoint loss.
                - off_loss (Tensor): Corner offset loss.
                - guiding_loss (Tensor): Guiding shift loss.
                - centripetal_loss (Tensor): Centripetal shift loss.
        """
        targets['corner_embedding'] = None
        det_loss, _, _, off_loss = super().loss_single(tl_hmp, br_hmp, None, None, tl_off, br_off, targets)
        gt_tl_guiding_shift = targets['topleft_guiding_shift']
        gt_br_guiding_shift = targets['bottomright_guiding_shift']
        gt_tl_centripetal_shift = targets['topleft_centripetal_shift']
        gt_br_centripetal_shift = targets['bottomright_centripetal_shift']
        gt_tl_heatmap = targets['topleft_heatmap']
        gt_br_heatmap = targets['bottomright_heatmap']
        tl_mask = gt_tl_heatmap.eq(1).sum(1).gt(0).unsqueeze(1).type_as(gt_tl_heatmap)
        br_mask = gt_br_heatmap.eq(1).sum(1).gt(0).unsqueeze(1).type_as(gt_br_heatmap)
        tl_guiding_loss = self.loss_guiding_shift(tl_guiding_shift, gt_tl_guiding_shift, tl_mask, avg_factor=tl_mask.sum())
        br_guiding_loss = self.loss_guiding_shift(br_guiding_shift, gt_br_guiding_shift, br_mask, avg_factor=br_mask.sum())
        guiding_loss = (tl_guiding_loss + br_guiding_loss) / 2.0
        tl_centripetal_loss = self.loss_centripetal_shift(tl_centripetal_shift, gt_tl_centripetal_shift, tl_mask, avg_factor=tl_mask.sum())
        br_centripetal_loss = self.loss_centripetal_shift(br_centripetal_shift, gt_br_centripetal_shift, br_mask, avg_factor=br_mask.sum())
        centripetal_loss = (tl_centripetal_loss + br_centripetal_loss) / 2.0
        return (det_loss, off_loss, guiding_loss, centripetal_loss)

    @force_fp32()
    def get_bboxes(self, tl_heats, br_heats, tl_offs, br_offs, tl_guiding_shifts, br_guiding_shifts, tl_centripetal_shifts, br_centripetal_shifts, img_metas, rescale=False, with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            tl_heats (list[Tensor]): Top-left corner heatmaps for each level
                with shape (N, num_classes, H, W).
            br_heats (list[Tensor]): Bottom-right corner heatmaps for each
                level with shape (N, num_classes, H, W).
            tl_offs (list[Tensor]): Top-left corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            br_offs (list[Tensor]): Bottom-right corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            tl_guiding_shifts (list[Tensor]): Top-left guiding shifts for each
                level with shape (N, guiding_shift_channels, H, W). Useless in
                this function, we keep this arg because it's the raw output
                from CentripetalHead.
            br_guiding_shifts (list[Tensor]): Bottom-right guiding shifts for
                each level with shape (N, guiding_shift_channels, H, W).
                Useless in this function, we keep this arg because it's the
                raw output from CentripetalHead.
            tl_centripetal_shifts (list[Tensor]): Top-left centripetal shifts
                for each level with shape (N, centripetal_shift_channels, H,
                W).
            br_centripetal_shifts (list[Tensor]): Bottom-right centripetal
                shifts for each level with shape (N,
                centripetal_shift_channels, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        """
        assert tl_heats[-1].shape[0] == br_heats[-1].shape[0] == len(img_metas)
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(self._get_bboxes_single(tl_heats[-1][img_id:img_id + 1, :], br_heats[-1][img_id:img_id + 1, :], tl_offs[-1][img_id:img_id + 1, :], br_offs[-1][img_id:img_id + 1, :], img_metas[img_id], tl_emb=None, br_emb=None, tl_centripetal_shift=tl_centripetal_shifts[-1][img_id:img_id + 1, :], br_centripetal_shift=br_centripetal_shifts[-1][img_id:img_id + 1, :], rescale=rescale, with_nms=with_nms))
        return result_list

@HEADS.register_module()
class YOLOXHead(BaseDenseHead, BBoxTestMixin):
    """YOLOXHead head used in `YOLOX <https://arxiv.org/abs/2107.08430>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Default: 256
        stacked_convs (int): Number of stacking convs of the head.
            Default: 2.
        strides (tuple): Downsample factor of each feature map.
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_obj (dict): Config of objectness loss.
        loss_l1 (dict): Config of L1 loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, num_classes, in_channels, feat_channels=256, stacked_convs=2, strides=[8, 16, 32], use_depthwise=False, dcn_on_last_conv=False, conv_bias='auto', conv_cfg=None, norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=dict(type='Swish'), loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='sum', loss_weight=1.0), loss_bbox=dict(type='IoULoss', mode='square', eps=1e-16, reduction='sum', loss_weight=5.0), loss_obj=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='sum', loss_weight=1.0), loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0), train_cfg=None, test_cfg=None, init_cfg=dict(type='Kaiming', layer='Conv2d', a=math.sqrt(5), distribution='uniform', mode='fan_in', nonlinearity='leaky_relu')):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_sigmoid_cls = True
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_obj = build_loss(loss_obj)
        self.use_l1 = False
        self.loss_l1 = build_loss(loss_l1)
        self.prior_generator = MlvlPointGenerator(strides, offset=0)
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False
        self._init_layers()

    def _init_layers(self):
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            conv_cls, conv_reg, conv_obj = self._build_predictor()
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_obj.append(conv_obj)

    def _build_stacked_convs(self):
        """Initialize conv layers of a single level head."""
        conv = DepthwiseSeparableConvModule if self.use_depthwise else ConvModule
        stacked_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            stacked_convs.append(conv(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, bias=self.conv_bias))
        return nn.Sequential(*stacked_convs)

    def _build_predictor(self):
        """Initialize predictor layers of a single level head."""
        conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        conv_reg = nn.Conv2d(self.feat_channels, 4, 1)
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
        return (conv_cls, conv_reg, conv_obj)

    def init_weights(self):
        super(YOLOXHead, self).init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(self.multi_level_conv_cls, self.multi_level_conv_obj):
            conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)

    def forward_single(self, x, cls_convs, reg_convs, conv_cls, conv_reg, conv_obj):
        """Forward feature of a single scale level."""
        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)
        cls_score = conv_cls(cls_feat)
        bbox_pred = conv_reg(reg_feat)
        objectness = conv_obj(reg_feat)
        return (cls_score, bbox_pred, objectness)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """
        return multi_apply(self.forward_single, feats, self.multi_level_cls_convs, self.multi_level_reg_convs, self.multi_level_conv_cls, self.multi_level_conv_reg, self.multi_level_conv_obj)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def get_bboxes(self, cls_scores, bbox_preds, objectnesses, img_metas=None, cfg=None, rescale=False, with_nms=True):
        """Transform network outputs of a batch into bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.
        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg
        scale_factors = np.array([img_meta['scale_factor'] for img_meta in img_metas])
        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device, with_stride=True)
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels) for cls_score in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds]
        flatten_objectness = [objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1) for objectness in objectnesses]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)
        if rescale:
            flatten_bboxes[..., :4] /= flatten_bboxes.new_tensor(scale_factors).unsqueeze(1)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_scores = flatten_cls_scores[img_id]
            score_factor = flatten_objectness[img_id]
            bboxes = flatten_bboxes[img_id]
            result_list.append(self._bboxes_nms(cls_scores, bboxes, score_factor, cfg))
        return result_list

    def _bbox_decode(self, priors, bbox_preds):
        xys = bbox_preds[..., :2] * priors[:, 2:] + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]
        tl_x = xys[..., 0] - whs[..., 0] / 2
        tl_y = xys[..., 1] - whs[..., 1] / 2
        br_x = xys[..., 0] + whs[..., 0] / 2
        br_y = xys[..., 1] + whs[..., 1] / 2
        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes

    def _bboxes_nms(self, cls_scores, bboxes, score_factor, cfg):
        max_scores, labels = torch.max(cls_scores, 1)
        valid_mask = score_factor * max_scores >= cfg.score_thr
        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        labels = labels[valid_mask]
        if labels.numel() == 0:
            return (bboxes, labels)
        else:
            dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
            return (dets, labels[keep])

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss(self, cls_scores, bbox_preds, objectnesses, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device, with_stride=True)
        flatten_cls_preds = [cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels) for cls_pred in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds]
        flatten_objectness = [objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1) for objectness in objectnesses]
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)
        pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets, num_fg_imgs = multi_apply(self._get_target_single, flatten_cls_preds.detach(), flatten_objectness.detach(), flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1), flatten_bboxes.detach(), gt_bboxes, gt_labels)
        num_pos = torch.tensor(sum(num_fg_imgs), dtype=torch.float, device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)
        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        loss_bbox = self.loss_bbox(flatten_bboxes.view(-1, 4)[pos_masks], bbox_targets) / num_total_samples
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1), obj_targets) / num_total_samples
        loss_cls = self.loss_cls(flatten_cls_preds.view(-1, self.num_classes)[pos_masks], cls_targets) / num_total_samples
        loss_dict = dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)
        if self.use_l1:
            loss_l1 = self.loss_l1(flatten_bbox_preds.view(-1, 4)[pos_masks], l1_targets) / num_total_samples
            loss_dict.update(loss_l1=loss_l1)
        return loss_dict

    @torch.no_grad()
    def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes, gt_bboxes, gt_labels):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """
        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target, l1_target, 0)
        offset_priors = torch.cat([priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)
        assign_result = self.assigner.assign(cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(), offset_priors, decoded_bboxes, gt_bboxes, gt_labels)
        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)
        pos_ious = assign_result.max_overlaps[pos_inds]
        cls_target = F.one_hot(sampling_result.pos_gt_labels, self.num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target, priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, obj_target, bbox_target, l1_target, num_pos_per_img)

    def _get_l1_target(self, l1_target, gt_bboxes, priors, eps=1e-08):
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return l1_target

@HEADS.register_module()
class LDHead(GFLHead):
    """Localization distillation Head. (Short description)

    It utilizes the learned bbox distributions to transfer the localization
    dark knowledge from teacher to student. Original paper: `Localization
    Distillation for Object Detection. <https://arxiv.org/abs/2102.12252>`_

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss_ld (dict): Config of Localization Distillation Loss (LD),
            T is the temperature for distillation.
    """

    def __init__(self, num_classes, in_channels, loss_ld=dict(type='LocalizationDistillationLoss', loss_weight=0.25, T=10), **kwargs):
        super(LDHead, self).__init__(num_classes, in_channels, **kwargs)
        self.loss_ld = build_loss(loss_ld)

    def loss_single(self, anchors, cls_score, bbox_pred, labels, label_weights, bbox_targets, stride, soft_targets, num_total_samples):
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
            dict[tuple, Tensor]: Loss components and weight targets.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
        soft_targets = soft_targets.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
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
            pos_soft_targets = soft_targets[pos_inds]
            soft_corners = pos_soft_targets.reshape(-1, self.reg_max + 1)
            target_corners = self.bbox_coder.encode(pos_anchor_centers, pos_decode_bbox_targets, self.reg_max).reshape(-1)
            loss_bbox = self.loss_bbox(pos_decode_bbox_pred, pos_decode_bbox_targets, weight=weight_targets, avg_factor=1.0)
            loss_dfl = self.loss_dfl(pred_corners, target_corners, weight=weight_targets[:, None].expand(-1, 4).reshape(-1), avg_factor=4.0)
            loss_ld = self.loss_ld(pred_corners, soft_corners, weight=weight_targets[:, None].expand(-1, 4).reshape(-1), avg_factor=4.0)
        else:
            loss_ld = bbox_pred.sum() * 0
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)
        loss_cls = self.loss_cls(cls_score, (labels, score), weight=label_weights, avg_factor=num_total_samples)
        return (loss_cls, loss_bbox, loss_dfl, loss_ld, weight_targets.sum())

    def forward_train(self, x, out_teacher, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=None, proposal_cfg=None, **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple[dict, list]: The loss components and proposals of each image.

            - losses (dict[str, Tensor]): A dictionary of loss components.
            - proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        soft_target = out_teacher[1]
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, soft_target, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, soft_target, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return (losses, proposal_list)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, soft_target, img_metas, gt_bboxes_ignore=None):
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
        losses_cls, losses_bbox, losses_dfl, losses_ld, avg_factor = multi_apply(self.loss_single, anchor_list, cls_scores, bbox_preds, labels_list, label_weights_list, bbox_targets_list, self.prior_generator.strides, soft_target, num_total_samples=num_total_samples)
        avg_factor = sum(avg_factor) + 1e-06
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox = [x / avg_factor for x in losses_bbox]
        losses_dfl = [x / avg_factor for x in losses_dfl]
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl, loss_ld=losses_ld)

@HEADS.register_module()
class VFNetHead(ATSSHead, FCOSHead):
    """Head of `VarifocalNet (VFNet): An IoU-aware Dense Object
    Detector.<https://arxiv.org/abs/2008.13367>`_.

    The VFNet predicts IoU-aware classification scores which mix the
    object presence confidence and object localization accuracy as the
    detection score. It is built on the FCOS architecture and uses ATSS
    for defining positive/negative training examples. The VFNet is trained
    with Varifocal Loss and empolys star-shaped deformable convolution to
    extract features for a bbox.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        sync_num_pos (bool): If true, synchronize the number of positive
            examples across GPUs. Default: True
        gradient_mul (float): The multiplier to gradients from bbox refinement
            and recognition. Default: 0.1.
        bbox_norm_type (str): The bbox normalization type, 'reg_denom' or
            'stride'. Default: reg_denom
        loss_cls_fl (dict): Config of focal loss.
        use_vfl (bool): If true, use varifocal loss for training.
            Default: True.
        loss_cls (dict): Config of varifocal loss.
        loss_bbox (dict): Config of localization loss, GIoU Loss.
        loss_bbox (dict): Config of localization refinement loss, GIoU Loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32,
            requires_grad=True).
        use_atss (bool): If true, use ATSS to define positive/negative
            examples. Default: True.
        anchor_generator (dict): Config of anchor generator for ATSS.
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = VFNetHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, bbox_pred_refine= self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self, num_classes, in_channels, regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)), center_sampling=False, center_sample_radius=1.5, sync_num_pos=True, gradient_mul=0.1, bbox_norm_type='reg_denom', loss_cls_fl=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0), use_vfl=True, loss_cls=dict(type='VarifocalLoss', use_sigmoid=True, alpha=0.75, gamma=2.0, iou_weighted=True, loss_weight=1.0), loss_bbox=dict(type='GIoULoss', loss_weight=1.5), loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0), norm_cfg=dict(type='GN', num_groups=32, requires_grad=True), use_atss=True, reg_decoded_bbox=True, anchor_generator=dict(type='AnchorGenerator', ratios=[1.0], octave_base_scale=8, scales_per_octave=1, center_offset=0.0, strides=[8, 16, 32, 64, 128]), init_cfg=dict(type='Normal', layer='Conv2d', std=0.01, override=dict(type='Normal', name='vfnet_cls', std=0.01, bias_prob=0.01)), **kwargs):
        self.num_dconv_points = 9
        self.dcn_kernel = int(np.sqrt(self.num_dconv_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        dcn_base = np.arange(-self.dcn_pad, self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(-1)
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        super(FCOSHead, self).__init__(num_classes, in_channels, norm_cfg=norm_cfg, init_cfg=init_cfg, **kwargs)
        self.regress_ranges = regress_ranges
        self.reg_denoms = [regress_range[-1] for regress_range in regress_ranges]
        self.reg_denoms[-1] = self.reg_denoms[-2] * 2
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.sync_num_pos = sync_num_pos
        self.bbox_norm_type = bbox_norm_type
        self.gradient_mul = gradient_mul
        self.use_vfl = use_vfl
        if self.use_vfl:
            self.loss_cls = build_loss(loss_cls)
        else:
            self.loss_cls = build_loss(loss_cls_fl)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.use_atss = use_atss
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.anchor_center_offset = anchor_generator['center_offset']
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.atss_prior_generator = build_prior_generator(anchor_generator)
        self.fcos_prior_generator = MlvlPointGenerator(anchor_generator['strides'], self.anchor_center_offset if self.use_atss else 0.5)
        self.prior_generator = self.fcos_prior_generator

    @property
    def num_anchors(self):
        """
        Returns:
            int: Number of anchors on each point of feature map.
        """
        warnings.warn('DeprecationWarning: `num_anchors` is deprecated, please use "num_base_priors" instead')
        return self.num_base_priors

    @property
    def anchor_generator(self):
        warnings.warn('DeprecationWarning: anchor_generator is deprecated, please use "atss_prior_generator" instead')
        return self.prior_generator

    def _init_layers(self):
        """Initialize layers of the head."""
        super(FCOSHead, self)._init_cls_convs()
        super(FCOSHead, self)._init_reg_convs()
        self.relu = nn.ReLU(inplace=True)
        self.vfnet_reg_conv = ConvModule(self.feat_channels, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.conv_bias)
        self.vfnet_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.vfnet_reg_refine_dconv = DeformConv2d(self.feat_channels, self.feat_channels, self.dcn_kernel, 1, padding=self.dcn_pad)
        self.vfnet_reg_refine = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales_refine = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.vfnet_cls_dconv = DeformConv2d(self.feat_channels, self.feat_channels, self.dcn_kernel, 1, padding=self.dcn_pad)
        self.vfnet_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box iou-aware scores for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box offsets for each
                    scale level, each is a 4D-tensor, the channel number is
                    num_points * 4.
                bbox_preds_refine (list[Tensor]): Refined Box offsets for
                    each scale level, each is a 4D-tensor, the channel
                    number is num_points * 4.
        """
        return multi_apply(self.forward_single, feats, self.scales, self.scales_refine, self.strides, self.reg_denoms)

    def forward_single(self, x, scale, scale_refine, stride, reg_denom):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            scale_refine (:obj: `mmcv.cnn.Scale`): Learnable scale module to
                resize the refined bbox prediction.
            stride (int): The corresponding stride for feature maps,
                used to normalize the bbox prediction when
                bbox_norm_type = 'stride'.
            reg_denom (int): The corresponding regression range for feature
                maps, only used to normalize the bbox prediction when
                bbox_norm_type = 'reg_denom'.

        Returns:
            tuple: iou-aware cls scores for each box, bbox predictions and
                refined bbox predictions of input feature maps.
        """
        cls_feat = x
        reg_feat = x
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        reg_feat_init = self.vfnet_reg_conv(reg_feat)
        if self.bbox_norm_type == 'reg_denom':
            bbox_pred = scale(self.vfnet_reg(reg_feat_init)).float().exp() * reg_denom
        elif self.bbox_norm_type == 'stride':
            bbox_pred = scale(self.vfnet_reg(reg_feat_init)).float().exp() * stride
        else:
            raise NotImplementedError
        dcn_offset = self.star_dcn_offset(bbox_pred, self.gradient_mul, stride).to(reg_feat.dtype)
        reg_feat = self.relu(self.vfnet_reg_refine_dconv(reg_feat, dcn_offset))
        bbox_pred_refine = scale_refine(self.vfnet_reg_refine(reg_feat)).float().exp()
        bbox_pred_refine = bbox_pred_refine * bbox_pred.detach()
        cls_feat = self.relu(self.vfnet_cls_dconv(cls_feat, dcn_offset))
        cls_score = self.vfnet_cls(cls_feat)
        if self.training:
            return (cls_score, bbox_pred, bbox_pred_refine)
        else:
            return (cls_score, bbox_pred_refine)

    def star_dcn_offset(self, bbox_pred, gradient_mul, stride):
        """Compute the star deformable conv offsets.

        Args:
            bbox_pred (Tensor): Predicted bbox distance offsets (l, r, t, b).
            gradient_mul (float): Gradient multiplier.
            stride (int): The corresponding stride for feature maps,
                used to project the bbox onto the feature map.

        Returns:
            dcn_offsets (Tensor): The offsets for deformable convolution.
        """
        dcn_base_offset = self.dcn_base_offset.type_as(bbox_pred)
        bbox_pred_grad_mul = (1 - gradient_mul) * bbox_pred.detach() + gradient_mul * bbox_pred
        bbox_pred_grad_mul = bbox_pred_grad_mul / stride
        N, C, H, W = bbox_pred.size()
        x1 = bbox_pred_grad_mul[:, 0, :, :]
        y1 = bbox_pred_grad_mul[:, 1, :, :]
        x2 = bbox_pred_grad_mul[:, 2, :, :]
        y2 = bbox_pred_grad_mul[:, 3, :, :]
        bbox_pred_grad_mul_offset = bbox_pred.new_zeros(N, 2 * self.num_dconv_points, H, W)
        bbox_pred_grad_mul_offset[:, 0, :, :] = -1.0 * y1
        bbox_pred_grad_mul_offset[:, 1, :, :] = -1.0 * x1
        bbox_pred_grad_mul_offset[:, 2, :, :] = -1.0 * y1
        bbox_pred_grad_mul_offset[:, 4, :, :] = -1.0 * y1
        bbox_pred_grad_mul_offset[:, 5, :, :] = x2
        bbox_pred_grad_mul_offset[:, 7, :, :] = -1.0 * x1
        bbox_pred_grad_mul_offset[:, 11, :, :] = x2
        bbox_pred_grad_mul_offset[:, 12, :, :] = y2
        bbox_pred_grad_mul_offset[:, 13, :, :] = -1.0 * x1
        bbox_pred_grad_mul_offset[:, 14, :, :] = y2
        bbox_pred_grad_mul_offset[:, 16, :, :] = y2
        bbox_pred_grad_mul_offset[:, 17, :, :] = x2
        dcn_offset = bbox_pred_grad_mul_offset - dcn_base_offset
        return dcn_offset

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_refine'))
    def loss(self, cls_scores, bbox_preds, bbox_preds_refine, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box offsets for each
                scale level, each is a 4D-tensor, the channel number is
                num_points * 4.
            bbox_preds_refine (list[Tensor]): Refined Box offsets for
                each scale level, each is a 4D-tensor, the channel
                number is num_points * 4.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
                Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(bbox_preds_refine)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.fcos_prior_generator.grid_priors(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        labels, label_weights, bbox_targets, bbox_weights = self.get_targets(cls_scores, all_level_points, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous() for cls_score in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4).contiguous() for bbox_pred in bbox_preds]
        flatten_bbox_preds_refine = [bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4).contiguous() for bbox_pred_refine in bbox_preds_refine]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_bbox_preds_refine = torch.cat(flatten_bbox_preds_refine)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])
        bg_class_ind = self.num_classes
        pos_inds = torch.where((flatten_labels >= 0) & (flatten_labels < bg_class_ind) > 0)[0]
        num_pos = len(pos_inds)
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_preds_refine = flatten_bbox_preds_refine[pos_inds]
        pos_labels = flatten_labels[pos_inds]
        if self.sync_num_pos:
            num_pos_avg_per_gpu = reduce_mean(pos_inds.new_tensor(num_pos).float()).item()
            num_pos_avg_per_gpu = max(num_pos_avg_per_gpu, 1.0)
        else:
            num_pos_avg_per_gpu = num_pos
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_points = flatten_points[pos_inds]
        pos_decoded_bbox_preds = self.bbox_coder.decode(pos_points, pos_bbox_preds)
        pos_decoded_target_preds = self.bbox_coder.decode(pos_points, pos_bbox_targets)
        iou_targets_ini = bbox_overlaps(pos_decoded_bbox_preds, pos_decoded_target_preds.detach(), is_aligned=True).clamp(min=1e-06)
        bbox_weights_ini = iou_targets_ini.clone().detach()
        bbox_avg_factor_ini = reduce_mean(bbox_weights_ini.sum()).clamp_(min=1).item()
        pos_decoded_bbox_preds_refine = self.bbox_coder.decode(pos_points, pos_bbox_preds_refine)
        iou_targets_rf = bbox_overlaps(pos_decoded_bbox_preds_refine, pos_decoded_target_preds.detach(), is_aligned=True).clamp(min=1e-06)
        bbox_weights_rf = iou_targets_rf.clone().detach()
        bbox_avg_factor_rf = reduce_mean(bbox_weights_rf.sum()).clamp_(min=1).item()
        if num_pos > 0:
            loss_bbox = self.loss_bbox(pos_decoded_bbox_preds, pos_decoded_target_preds.detach(), weight=bbox_weights_ini, avg_factor=bbox_avg_factor_ini)
            loss_bbox_refine = self.loss_bbox_refine(pos_decoded_bbox_preds_refine, pos_decoded_target_preds.detach(), weight=bbox_weights_rf, avg_factor=bbox_avg_factor_rf)
            if self.use_vfl:
                pos_ious = iou_targets_rf.clone().detach()
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)
                cls_iou_targets[pos_inds, pos_labels] = pos_ious
        else:
            loss_bbox = pos_bbox_preds.sum() * 0
            loss_bbox_refine = pos_bbox_preds_refine.sum() * 0
            if self.use_vfl:
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)
        if self.use_vfl:
            loss_cls = self.loss_cls(flatten_cls_scores, cls_iou_targets, avg_factor=num_pos_avg_per_gpu)
        else:
            loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, weight=label_weights, avg_factor=num_pos_avg_per_gpu)
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_bbox_rf=loss_bbox_refine)

    def get_targets(self, cls_scores, mlvl_points, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore):
        """A wrapper for computing ATSS and FCOS targets for points in multiple
        images.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level.
                label_weights (Tensor/None): Label weights of all levels.
                bbox_targets_list (list[Tensor]): Regression targets of each
                    level, (l, t, r, b).
                bbox_weights (Tensor/None): Bbox weights of all levels.
        """
        if self.use_atss:
            return self.get_atss_targets(cls_scores, mlvl_points, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
        else:
            self.norm_on_bbox = False
            return self.get_fcos_targets(mlvl_points, gt_bboxes, gt_labels)

    def _get_target_single(self, *args, **kwargs):
        """Avoid ambiguity in multiple inheritance."""
        if self.use_atss:
            return ATSSHead._get_target_single(self, *args, **kwargs)
        else:
            return FCOSHead._get_target_single(self, *args, **kwargs)

    def get_fcos_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute FCOS regression and classification targets for points in
        multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                labels (list[Tensor]): Labels of each level.
                label_weights: None, to be compatible with ATSS targets.
                bbox_targets (list[Tensor]): BBox targets of each level.
                bbox_weights: None, to be compatible with ATSS targets.
        """
        labels, bbox_targets = FCOSHead.get_targets(self, points, gt_bboxes_list, gt_labels_list)
        label_weights = None
        bbox_weights = None
        return (labels, label_weights, bbox_targets, bbox_weights)

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
        multi_level_anchors = self.atss_prior_generator.grid_priors(featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.atss_prior_generator.valid_flags(featmap_sizes, img_meta['pad_shape'], device=device)
            valid_flag_list.append(multi_level_flags)
        return (anchor_list, valid_flag_list)

    def get_atss_targets(self, cls_scores, mlvl_points, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """A wrapper for computing ATSS targets for points in multiple images.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4). Default: None.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level.
                label_weights (Tensor): Label weights of all levels.
                bbox_targets_list (list[Tensor]): Regression targets of each
                    level, (l, t, r, b).
                bbox_weights (Tensor): Bbox weights of all levels.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.atss_prior_generator.num_levels == self.fcos_prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = ATSSHead.get_targets(self, anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels, unmap_outputs=True)
        if cls_reg_targets is None:
            return None
        anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        bbox_targets_list = [bbox_targets.reshape(-1, 4) for bbox_targets in bbox_targets_list]
        num_imgs = len(img_metas)
        bbox_targets_list = self.transform_bbox_targets(bbox_targets_list, mlvl_points, num_imgs)
        labels_list = [labels.reshape(-1) for labels in labels_list]
        label_weights_list = [label_weights.reshape(-1) for label_weights in label_weights_list]
        bbox_weights_list = [bbox_weights.reshape(-1) for bbox_weights in bbox_weights_list]
        label_weights = torch.cat(label_weights_list)
        bbox_weights = torch.cat(bbox_weights_list)
        return (labels_list, label_weights, bbox_targets_list, bbox_weights)

    def transform_bbox_targets(self, decoded_bboxes, mlvl_points, num_imgs):
        """Transform bbox_targets (x1, y1, x2, y2) into (l, t, r, b) format.

        Args:
            decoded_bboxes (list[Tensor]): Regression targets of each level,
                in the form of (x1, y1, x2, y2).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            num_imgs (int): the number of images in a batch.

        Returns:
            bbox_targets (list[Tensor]): Regression targets of each level in
                the form of (l, t, r, b).
        """
        assert len(decoded_bboxes) == len(mlvl_points)
        num_levels = len(decoded_bboxes)
        mlvl_points = [points.repeat(num_imgs, 1) for points in mlvl_points]
        bbox_targets = []
        for i in range(num_levels):
            bbox_target = self.bbox_coder.encode(mlvl_points[i], decoded_bboxes[i])
            bbox_targets.append(bbox_target)
        return bbox_targets

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Override the method in the parent class to avoid changing para's
        name."""
        pass

    def _get_points_single(self, featmap_size, stride, dtype, device, flatten=False):
        """Get points according to feature map size.

        This function will be deprecated soon.
        """
        warnings.warn('`_get_points_single` in `VFNetHead` will be deprecated soon, we support a multi level point generator nowyou can get points of a single level feature mapwith `self.fcos_prior_generator.single_level_grid_priors` ')
        h, w = featmap_size
        x_range = torch.arange(0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        if self.use_atss:
            points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1) + stride * self.anchor_center_offset
        else:
            points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

@HEADS.register_module()
class YOLACTHead(AnchorHead):
    """YOLACT box head used in https://arxiv.org/abs/1904.02689.

    Note that YOLACT head is a light version of RetinaNet head.
    Four differences are described as follows:

    1. YOLACT box head has three-times fewer anchors.
    2. YOLACT box head shares the convs for box and cls branches.
    3. YOLACT box head uses OHEM instead of Focal loss.
    4. YOLACT box head predicts a set of mask coefficients for each box.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        anchor_generator (dict): Config dict for anchor generator
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        num_head_convs (int): Number of the conv layers shared by
            box and cls branches.
        num_protos (int): Number of the mask coefficients.
        use_ohem (bool): If true, ``loss_single_OHEM`` will be used for
            cls loss calculation. If false, ``loss_single`` will be used.
        conv_cfg (dict): Dictionary to construct and config conv layer.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, num_classes, in_channels, anchor_generator=dict(type='AnchorGenerator', octave_base_scale=3, scales_per_octave=1, ratios=[0.5, 1.0, 2.0], strides=[8, 16, 32, 64, 128]), loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, reduction='none', loss_weight=1.0), loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.5), num_head_convs=1, num_protos=32, use_ohem=True, conv_cfg=None, norm_cfg=None, init_cfg=dict(type='Xavier', distribution='uniform', bias=0, layer='Conv2d'), **kwargs):
        self.num_head_convs = num_head_convs
        self.num_protos = num_protos
        self.use_ohem = use_ohem
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(YOLACTHead, self).__init__(num_classes, in_channels, loss_cls=loss_cls, loss_bbox=loss_bbox, anchor_generator=anchor_generator, init_cfg=init_cfg, **kwargs)
        if self.use_ohem:
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
            self.sampling = False

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.head_convs = ModuleList()
        for i in range(self.num_head_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.head_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        self.conv_cls = nn.Conv2d(self.feat_channels, self.num_base_priors * self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4, 3, padding=1)
        self.conv_coeff = nn.Conv2d(self.feat_channels, self.num_base_priors * self.num_protos, 3, padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level                     the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale                     level, the channels number is num_anchors * 4.
                coeff_pred (Tensor): Mask coefficients for a single scale                     level, the channels number is num_anchors * num_protos.
        """
        for head_conv in self.head_convs:
            x = head_conv(x)
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        coeff_pred = self.conv_coeff(x).tanh()
        return (cls_score, bbox_pred, coeff_pred)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """A combination of the func:``AnchorHead.loss`` and
        func:``SSDHead.loss``.

        When ``self.use_ohem == True``, it functions like ``SSDHead.loss``,
        otherwise, it follows ``AnchorHead.loss``. Besides, it additionally
        returns ``sampling_results``.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            tuple:
                dict[str, Tensor]: A dictionary of loss components.
                List[:obj:``SamplingResult``]: Sampler results for each image.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels, unmap_outputs=not self.use_ohem, return_sampling_results=True)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg, sampling_results = cls_reg_targets
        if self.use_ohem:
            num_images = len(img_metas)
            all_cls_scores = torch.cat([s.permute(0, 2, 3, 1).reshape(num_images, -1, self.cls_out_channels) for s in cls_scores], 1)
            all_labels = torch.cat(labels_list, -1).view(num_images, -1)
            all_label_weights = torch.cat(label_weights_list, -1).view(num_images, -1)
            all_bbox_preds = torch.cat([b.permute(0, 2, 3, 1).reshape(num_images, -1, 4) for b in bbox_preds], -2)
            all_bbox_targets = torch.cat(bbox_targets_list, -2).view(num_images, -1, 4)
            all_bbox_weights = torch.cat(bbox_weights_list, -2).view(num_images, -1, 4)
            all_anchors = []
            for i in range(num_images):
                all_anchors.append(torch.cat(anchor_list[i]))
            assert torch.isfinite(all_cls_scores).all().item(), 'classification scores become infinite or NaN!'
            assert torch.isfinite(all_bbox_preds).all().item(), 'bbox predications become infinite or NaN!'
            losses_cls, losses_bbox = multi_apply(self.loss_single_OHEM, all_cls_scores, all_bbox_preds, all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, num_total_samples=num_total_pos)
        else:
            num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos
            num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
            concat_anchor_list = []
            for i in range(len(anchor_list)):
                concat_anchor_list.append(torch.cat(anchor_list[i]))
            all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)
            losses_cls, losses_bbox = multi_apply(self.loss_single, cls_scores, bbox_preds, all_anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_samples=num_total_samples)
        return (dict(loss_cls=losses_cls, loss_bbox=losses_bbox), sampling_results)

    def loss_single_OHEM(self, cls_score, bbox_pred, anchors, labels, label_weights, bbox_targets, bbox_weights, num_total_samples):
        """"See func:``SSDHead.loss``."""
        loss_cls_all = self.loss_cls(cls_score, labels, label_weights)
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(as_tuple=False).reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero(as_tuple=False).view(-1)
        num_pos_samples = pos_inds.size(0)
        if num_pos_samples == 0:
            num_neg_samples = neg_inds.size(0)
        else:
            num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
            if num_neg_samples > neg_inds.size(0):
                num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples
        if self.reg_decoded_bbox:
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples)
        return (loss_cls[None], loss_bbox)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'coeff_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, coeff_preds, img_metas, cfg=None, rescale=False):
        """"Similar to func:``AnchorHead.get_bboxes``, but additionally
        processes coeff_preds.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            coeff_preds (list[Tensor]): Mask coefficients for each scale
                level with shape (N, num_anchors * num_protos, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor, Tensor]]: Each item in result_list is
                a 3-tuple. The first item is an (n, 5) tensor, where the
                first 4 columns are bounding box positions
                (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                between 0 and 1. The second item is an (n,) tensor where each
                item is the predicted class label of the corresponding box.
                The third item is an (n, num_protos) tensor where each item
                is the predicted mask coefficients of instance inside the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.prior_generator.grid_priors(featmap_sizes, device=device)
        det_bboxes = []
        det_labels = []
        det_coeffs = []
        for img_id in range(len(img_metas)):
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            coeff_pred_list = select_single_mlvl(coeff_preds, img_id)
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            bbox_res = self._get_bboxes_single(cls_score_list, bbox_pred_list, coeff_pred_list, mlvl_anchors, img_shape, scale_factor, cfg, rescale)
            det_bboxes.append(bbox_res[0])
            det_labels.append(bbox_res[1])
            det_coeffs.append(bbox_res[2])
        return (det_bboxes, det_labels, det_coeffs)

    def _get_bboxes_single(self, cls_score_list, bbox_pred_list, coeff_preds_list, mlvl_anchors, img_shape, scale_factor, cfg, rescale=False):
        """"Similar to func:``AnchorHead._get_bboxes_single``, but additionally
        processes coeff_preds_list and uses fast NMS instead of traditional
        NMS.

        Args:
            cls_score_list (list[Tensor]): Box scores for a single scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            coeff_preds_list (list[Tensor]): Mask coefficients for a single
                scale level with shape (num_anchors * num_protos, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            tuple[Tensor, Tensor, Tensor]: The first item is an (n, 5) tensor,
                where the first 4 columns are bounding box positions
                (tl_x, tl_y, br_x, br_y) and the 5-th column is a score between
                0 and 1. The second item is an (n,) tensor where each item is
                the predicted class label of the corresponding box. The third
                item is an (n, num_protos) tensor where each item is the
                predicted mask coefficients of instance inside the
                corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        nms_pre = cfg.get('nms_pre', -1)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_coeffs = []
        for cls_score, bbox_pred, coeff_pred, anchors in zip(cls_score_list, bbox_pred_list, coeff_preds_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            coeff_pred = coeff_pred.permute(1, 2, 0).reshape(-1, self.num_protos)
            if 0 < nms_pre < scores.shape[0]:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                coeff_pred = coeff_pred[topk_inds, :]
            bboxes = self.bbox_coder.decode(anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_coeffs.append(coeff_pred)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_coeffs = torch.cat(mlvl_coeffs)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels, det_coeffs = fast_nms(mlvl_bboxes, mlvl_scores, mlvl_coeffs, cfg.score_thr, cfg.iou_thr, cfg.top_k, cfg.max_per_img)
        return (det_bboxes, det_labels, det_coeffs)

@HEADS.register_module()
class FCOSHead(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self, num_classes, in_channels, regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)), center_sampling=False, center_sample_radius=1.5, norm_on_bbox=False, centerness_on_reg=False, loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0), loss_bbox=dict(type='IoULoss', loss_weight=1.0), loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), norm_cfg=dict(type='GN', num_groups=32, requires_grad=True), init_cfg=dict(type='Normal', layer='Conv2d', std=0.01, override=dict(type='Normal', name='conv_cls', std=0.01, bias_prob=0.01)), **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, loss_bbox=loss_bbox, norm_cfg=norm_cfg, init_cfg=init_cfg, **kwargs)
        self.loss_centerness = build_loss(loss_centerness)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level,                     each is a 4D-tensor, the channel number is                     num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each                     scale level, each is a 4D-tensor, the channel number is                     num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level,                     each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales, self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness                 predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return (cls_score, bbox_pred, centerness)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self, cls_scores, bbox_preds, centernesses, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes, gt_labels)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds]
        flatten_centerness = [centerness.permute(0, 2, 3, 1).reshape(-1) for centerness in centernesses]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos)
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        centerness_denorm = max(reduce_mean(pos_centerness_targets.sum().detach()), 1e-06)
        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(pos_decoded_bbox_preds, pos_decoded_target_preds, weight=pos_centerness_targets, avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_centerness=loss_centerness)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level.                 concat_lvl_bbox_targets (list[Tensor]): BBox targets of each                     level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        expanded_regress_ranges = [points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i]) for i in range(num_levels)]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        num_points = [center.size(0) for center in points]
        labels_list, bbox_targets_list = multi_apply(self._get_target_single, gt_bboxes_list, gt_labels_list, points=concat_points, regress_ranges=concat_regress_ranges, num_points_per_lvl=num_points)
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [bbox_targets.split(num_points, 0) for bbox_targets in bbox_targets_list]
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets)

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges, num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return (gt_labels.new_full((num_points,), self.num_classes), gt_bboxes.new_zeros((num_points, 4)))
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = (points[:, 0], points[:, 1])
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        if self.center_sampling:
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end
            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0], x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1], y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2], gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3], gt_bboxes[..., 3], y_maxs)
            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack((cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (max_regress_distance >= regress_ranges[..., 0]) & (max_regress_distance <= regress_ranges[..., 1])
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        return (labels, bbox_targets)

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0] * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def _get_points_single(self, featmap_size, stride, dtype, device, flatten=False):
        """Get points according to feature map size.

        This function will be deprecated soon.
        """
        warnings.warn('`_get_points_single` in `FCOSHead` will be deprecated soon, we support a multi level point generator nowyou can get points of a single level feature map with `self.prior_generator.single_level_grid_priors` ')
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride), dim=-1) + stride // 2
        return points

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
class DeformableDETRHead(DETRHead):
    """Head of DeformDETR: Deformable DETR: Deformable Transformers for End-to-
    End Object Detection.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(self, *args, with_box_refine=False, as_two_stage=False, transformer=None, **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        super(DeformableDETRHead, self).__init__(*args, transformer=transformer, **kwargs)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        num_pred = self.transformer.decoder.num_layers + 1 if self.as_two_stage else self.transformer.decoder.num_layers
        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,                 shape [nb_dec, bs, num_query, cls_out_channels]. Note                 cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression                 head with normalized coordinate format (cx, cy, w, h).                 Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode                 feature map, has shape (N, h*w, num_class). Only when                 as_two_stage is True it would be returned, otherwise                 `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the                 encode feature map, has shape (N, h*w, 4). Only when                 as_two_stage is True it would be returned, otherwise                 `None` would be returned.
        """
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(self.positional_encoding(mlvl_masks[-1]))
        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord = self.transformer(mlvl_feats, mlvl_masks, query_embeds, mlvl_positional_encodings, reg_branches=self.reg_branches if self.with_box_refine else None, cls_branches=self.cls_branches if self.as_two_stage else None)
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        if self.as_two_stage:
            return (outputs_classes, outputs_coords, enc_outputs_class, enc_outputs_coord.sigmoid())
        else:
            return (outputs_classes, outputs_coords, None, None)

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self, all_cls_scores, all_bbox_preds, enc_cls_scores, enc_bbox_preds, gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore=None):
        """"Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, f'{self.__class__.__name__} only supports for gt_bboxes_ignore setting to None.'
        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_bbox, losses_iou = multi_apply(self.loss_single, all_cls_scores, all_bbox_preds, all_gt_bboxes_list, all_gt_labels_list, img_metas_list, all_gt_bboxes_ignore_list)
        loss_dict = dict()
        if enc_cls_scores is not None:
            binary_labels_list = [torch.zeros_like(gt_labels_list[i]) for i in range(len(img_metas))]
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = self.loss_single(enc_cls_scores, enc_bbox_preds, gt_bboxes_list, binary_labels_list, img_metas, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self, all_cls_scores, all_bbox_preds, enc_cls_scores, enc_bbox_preds, img_metas, rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.                 The first item is an (n, 5) tensor, where the first 4 columns                 are bounding box positions (tl_x, tl_y, br_x, br_y) and the                 5-th column is a score between 0 and 1. The second item is a                 (n,) tensor where each item is the predicted class label of                 the corresponding box.
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred, img_shape, scale_factor, rescale)
            result_list.append(proposals)
        return result_list

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
class AnchorFreeHead(BaseDenseHead, BBoxTestMixin):
    """Anchor-free head (FCOS, Fovea, RepPoints, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        stacked_convs (int): Number of stacking convs of the head.
        strides (tuple): Downsample factor of each feature map.
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        bbox_coder (dict): Config of bbox coder. Defaults
            'DistancePointBBoxCoder'.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    _version = 1

    def __init__(self, num_classes, in_channels, feat_channels=256, stacked_convs=4, strides=(4, 8, 16, 32, 64), dcn_on_last_conv=False, conv_bias='auto', loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0), loss_bbox=dict(type='IoULoss', loss_weight=1.0), bbox_coder=dict(type='DistancePointBBoxCoder'), conv_cfg=None, norm_cfg=None, train_cfg=None, test_cfg=None, init_cfg=dict(type='Normal', layer='Conv2d', std=0.01, override=dict(type='Normal', name='conv_cls', std=0.01, bias_prob=0.01))):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.prior_generator = MlvlPointGenerator(strides)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=self.norm_cfg, bias=self.conv_bias))

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=self.norm_cfg, bias=self.conv_bias))

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Hack some keys of the model state dict so that can load checkpoints
        of previous version."""
        version = local_metadata.get('version', None)
        if version is None:
            bbox_head_keys = [k for k in state_dict.keys() if k.startswith(prefix)]
            ori_predictor_keys = []
            new_predictor_keys = []
            for key in bbox_head_keys:
                ori_predictor_keys.append(key)
                key = key.split('.')
                conv_name = None
                if key[1].endswith('cls'):
                    conv_name = 'conv_cls'
                elif key[1].endswith('reg'):
                    conv_name = 'conv_reg'
                elif key[1].endswith('centerness'):
                    conv_name = 'conv_centerness'
                else:
                    assert NotImplementedError
                if conv_name is not None:
                    key[1] = conv_name
                    new_predictor_keys.append('.'.join(key))
                else:
                    ori_predictor_keys.pop(-1)
            for i in range(len(new_predictor_keys)):
                state_dict[new_predictor_keys[i]] = state_dict.pop(ori_predictor_keys[i])
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually contain classification scores and bbox predictions.
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * 4.
        """
        return multi_apply(self.forward_single, feats)[:2]

    def forward_single(self, x):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.

        Returns:
            tuple: Scores for each class, bbox predictions, features
                after classification and regression conv layers, some
                models needs these features like FCOS.
        """
        cls_feat = x
        reg_feat = x
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        return (cls_score, bbox_pred, cls_feat, reg_feat)

    @abstractmethod
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        raise NotImplementedError

    @abstractmethod
    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
        """
        raise NotImplementedError

    def _get_points_single(self, featmap_size, stride, dtype, device, flatten=False):
        """Get points of a single scale level.

        This function will be deprecated soon.
        """
        warnings.warn('`_get_points_single` in `AnchorFreeHead` will be deprecated soon, we support a multi level point generator nowyou can get points of a single level feature map with `self.prior_generator.single_level_grid_priors` ')
        h, w = featmap_size
        x_range = torch.arange(w, device=device).to(dtype)
        y_range = torch.arange(h, device=device).to(dtype)
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        return (y, x)

    def get_points(self, featmap_sizes, dtype, device, flatten=False):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        warnings.warn('`get_points` in `AnchorFreeHead` will be deprecated soon, we support a multi level point generator nowyou can get points of all levels with `self.prior_generator.grid_priors` ')
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(self._get_points_single(featmap_sizes[i], self.strides[i], dtype, device, flatten))
        return mlvl_points

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
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)

@HEADS.register_module()
class AutoAssignHead(FCOSHead):
    """AutoAssignHead head used in AutoAssign.

    More details can be found in the `paper
    <https://arxiv.org/abs/2007.03496>`_ .

    Args:
        force_topk (bool): Used in center prior initialization to
            handle extremely small gt. Default is False.
        topk (int): The number of points used to calculate the
            center prior when no point falls in gt_bbox. Only work when
            force_topk if True. Defaults to 9.
        pos_loss_weight (float): The loss weight of positive loss
            and with default value 0.25.
        neg_loss_weight (float): The loss weight of negative loss
            and with default value 0.75.
        center_loss_weight (float): The loss weight of center prior
            loss and with default value 0.75.
    """

    def __init__(self, *args, force_topk=False, topk=9, pos_loss_weight=0.25, neg_loss_weight=0.75, center_loss_weight=0.75, **kwargs):
        super().__init__(*args, conv_bias=True, **kwargs)
        self.center_prior = CenterPrior(force_topk=force_topk, topk=topk, num_classes=self.num_classes, strides=self.strides)
        self.pos_loss_weight = pos_loss_weight
        self.neg_loss_weight = neg_loss_weight
        self.center_loss_weight = center_loss_weight
        self.prior_generator = MlvlPointGenerator(self.strides, offset=0)

    def init_weights(self):
        """Initialize weights of the head.

        In particular, we have special initialization for classified conv's and
        regression conv's bias
        """
        super(AutoAssignHead, self).init_weights()
        bias_cls = bias_init_with_prob(0.02)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01, bias=4.0)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness                 predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super(FCOSHead, self).forward_single(x)
        centerness = self.conv_centerness(reg_feat)
        bbox_pred = scale(bbox_pred).float()
        bbox_pred = bbox_pred.clamp(min=0)
        bbox_pred *= stride
        return (cls_score, bbox_pred, centerness)

    def get_pos_loss_single(self, cls_score, objectness, reg_loss, gt_labels, center_prior_weights):
        """Calculate the positive loss of all points in gt_bboxes.

        Args:
            cls_score (Tensor): All category scores for each point on
                the feature map. The shape is (num_points, num_class).
            objectness (Tensor): Foreground probability of all points,
                has shape (num_points, 1).
            reg_loss (Tensor): The regression loss of each gt_bbox and each
                prediction box, has shape of (num_points, num_gt).
            gt_labels (Tensor): The zeros based gt_labels of all gt
                with shape of (num_gt,).
            center_prior_weights (Tensor): Float tensor with shape
                of (num_points, num_gt). Each value represents
                the center weighting coefficient.

        Returns:
            tuple[Tensor]:

                - pos_loss (Tensor): The positive loss of all points
                  in the gt_bboxes.
        """
        p_loc = torch.exp(-reg_loss)
        p_cls = (cls_score * objectness)[:, gt_labels]
        p_pos = p_cls * p_loc
        confidence_weight = torch.exp(p_pos * 3)
        p_pos_weight = confidence_weight * center_prior_weights / (confidence_weight * center_prior_weights).sum(0, keepdim=True).clamp(min=EPS)
        reweighted_p_pos = (p_pos * p_pos_weight).sum(0)
        pos_loss = F.binary_cross_entropy(reweighted_p_pos, torch.ones_like(reweighted_p_pos), reduction='none')
        pos_loss = pos_loss.sum() * self.pos_loss_weight
        return (pos_loss,)

    def get_neg_loss_single(self, cls_score, objectness, gt_labels, ious, inside_gt_bbox_mask):
        """Calculate the negative loss of all points in feature map.

        Args:
            cls_score (Tensor): All category scores for each point on
                the feature map. The shape is (num_points, num_class).
            objectness (Tensor): Foreground probability of all points
                and is shape of (num_points, 1).
            gt_labels (Tensor): The zeros based label of all gt with shape of
                (num_gt).
            ious (Tensor): Float tensor with shape of (num_points, num_gt).
                Each value represent the iou of pred_bbox and gt_bboxes.
            inside_gt_bbox_mask (Tensor): Tensor of bool type,
                with shape of (num_points, num_gt), each
                value is used to mark whether this point falls
                within a certain gt.

        Returns:
            tuple[Tensor]:

                - neg_loss (Tensor): The negative loss of all points
                  in the feature map.
        """
        num_gts = len(gt_labels)
        joint_conf = cls_score * objectness
        p_neg_weight = torch.ones_like(joint_conf)
        if num_gts > 0:
            inside_gt_bbox_mask = inside_gt_bbox_mask.permute(1, 0)
            ious = ious.permute(1, 0)
            foreground_idxs = torch.nonzero(inside_gt_bbox_mask, as_tuple=True)
            temp_weight = 1 / (1 - ious[foreground_idxs]).clamp_(EPS)

            def normalize(x):
                return (x - x.min() + EPS) / (x.max() - x.min() + EPS)
            for instance_idx in range(num_gts):
                idxs = foreground_idxs[0] == instance_idx
                if idxs.any():
                    temp_weight[idxs] = normalize(temp_weight[idxs])
            p_neg_weight[foreground_idxs[1], gt_labels[foreground_idxs[0]]] = 1 - temp_weight
        logits = joint_conf * p_neg_weight
        neg_loss = logits ** 2 * F.binary_cross_entropy(logits, torch.zeros_like(logits), reduction='none')
        neg_loss = neg_loss.sum() * self.neg_loss_weight
        return (neg_loss,)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss(self, cls_scores, bbox_preds, objectnesses, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            objectnesses (list[Tensor]): objectness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        all_num_gt = sum([len(item) for item in gt_bboxes])
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)
        inside_gt_bbox_mask_list, bbox_targets_list = self.get_targets(all_level_points, gt_bboxes)
        center_prior_weight_list = []
        temp_inside_gt_bbox_mask_list = []
        for gt_bboxe, gt_label, inside_gt_bbox_mask in zip(gt_bboxes, gt_labels, inside_gt_bbox_mask_list):
            center_prior_weight, inside_gt_bbox_mask = self.center_prior(all_level_points, gt_bboxe, gt_label, inside_gt_bbox_mask)
            center_prior_weight_list.append(center_prior_weight)
            temp_inside_gt_bbox_mask_list.append(inside_gt_bbox_mask)
        inside_gt_bbox_mask_list = temp_inside_gt_bbox_mask_list
        mlvl_points = torch.cat(all_level_points, dim=0)
        bbox_preds = levels_to_images(bbox_preds)
        cls_scores = levels_to_images(cls_scores)
        objectnesses = levels_to_images(objectnesses)
        reg_loss_list = []
        ious_list = []
        num_points = len(mlvl_points)
        for bbox_pred, encoded_targets, inside_gt_bbox_mask in zip(bbox_preds, bbox_targets_list, inside_gt_bbox_mask_list):
            temp_num_gt = encoded_targets.size(1)
            expand_mlvl_points = mlvl_points[:, None, :].expand(num_points, temp_num_gt, 2).reshape(-1, 2)
            encoded_targets = encoded_targets.reshape(-1, 4)
            expand_bbox_pred = bbox_pred[:, None, :].expand(num_points, temp_num_gt, 4).reshape(-1, 4)
            decoded_bbox_preds = self.bbox_coder.decode(expand_mlvl_points, expand_bbox_pred)
            decoded_target_preds = self.bbox_coder.decode(expand_mlvl_points, encoded_targets)
            with torch.no_grad():
                ious = bbox_overlaps(decoded_bbox_preds, decoded_target_preds, is_aligned=True)
                ious = ious.reshape(num_points, temp_num_gt)
                if temp_num_gt:
                    ious = ious.max(dim=-1, keepdim=True).values.repeat(1, temp_num_gt)
                else:
                    ious = ious.new_zeros(num_points, temp_num_gt)
                ious[~inside_gt_bbox_mask] = 0
                ious_list.append(ious)
            loss_bbox = self.loss_bbox(decoded_bbox_preds, decoded_target_preds, weight=None, reduction_override='none')
            reg_loss_list.append(loss_bbox.reshape(num_points, temp_num_gt))
        cls_scores = [item.sigmoid() for item in cls_scores]
        objectnesses = [item.sigmoid() for item in objectnesses]
        pos_loss_list, = multi_apply(self.get_pos_loss_single, cls_scores, objectnesses, reg_loss_list, gt_labels, center_prior_weight_list)
        pos_avg_factor = reduce_mean(bbox_pred.new_tensor(all_num_gt)).clamp_(min=1)
        pos_loss = sum(pos_loss_list) / pos_avg_factor
        neg_loss_list, = multi_apply(self.get_neg_loss_single, cls_scores, objectnesses, gt_labels, ious_list, inside_gt_bbox_mask_list)
        neg_avg_factor = sum((item.data.sum() for item in center_prior_weight_list))
        neg_avg_factor = reduce_mean(neg_avg_factor).clamp_(min=1)
        neg_loss = sum(neg_loss_list) / neg_avg_factor
        center_loss = []
        for i in range(len(img_metas)):
            if inside_gt_bbox_mask_list[i].any():
                center_loss.append(len(gt_bboxes[i]) / center_prior_weight_list[i].sum().clamp_(min=EPS))
            else:
                center_loss.append(center_prior_weight_list[i].sum() * 0)
        center_loss = torch.stack(center_loss).mean() * self.center_loss_weight
        if all_num_gt == 0:
            pos_loss = bbox_preds[0].sum() * 0
            dummy_center_prior_loss = self.center_prior.mean.sum() * 0 + self.center_prior.sigma.sum() * 0
            center_loss = objectnesses[0].sum() * 0 + dummy_center_prior_loss
        loss = dict(loss_pos=pos_loss, loss_neg=neg_loss, loss_center=center_loss)
        return loss

    def get_targets(self, points, gt_bboxes_list):
        """Compute regression targets and each point inside or outside gt_bbox
        in multiple images.

        Args:
            points (list[Tensor]): Points of all fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).

        Returns:
            tuple(list[Tensor]):

                - inside_gt_bbox_mask_list (list[Tensor]): Each
                  Tensor is with bool type and shape of
                  (num_points, num_gt), each value
                  is used to mark whether this point falls
                  within a certain gt.
                - concat_lvl_bbox_targets (list[Tensor]): BBox
                  targets of each level. Each tensor has shape
                  (num_points, num_gt, 4).
        """
        concat_points = torch.cat(points, dim=0)
        inside_gt_bbox_mask_list, bbox_targets_list = multi_apply(self._get_target_single, gt_bboxes_list, points=concat_points)
        return (inside_gt_bbox_mask_list, bbox_targets_list)

    def _get_target_single(self, gt_bboxes, points):
        """Compute regression targets and each point inside or outside gt_bbox
        for a single image.

        Args:
            gt_bboxes (Tensor): gt_bbox of single image, has shape
                (num_gt, 4).
            points (Tensor): Points of all fpn level, has shape
                (num_points, 2).

        Returns:
            tuple[Tensor]: Containing the following Tensors:

                - inside_gt_bbox_mask (Tensor): Bool tensor with shape
                  (num_points, num_gt), each value is used to mark
                  whether this point falls within a certain gt.
                - bbox_targets (Tensor): BBox targets of each points with
                  each gt_bboxes, has shape (num_points, num_gt, 4).
        """
        num_points = points.size(0)
        num_gts = gt_bboxes.size(0)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = (points[:, 0], points[:, 1])
        xs = xs[:, None]
        ys = ys[:, None]
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        if num_gts:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        else:
            inside_gt_bbox_mask = bbox_targets.new_zeros((num_points, num_gts), dtype=torch.bool)
        return (inside_gt_bbox_mask, bbox_targets)

    def _get_points_single(self, featmap_size, stride, dtype, device, flatten=False):
        """Almost the same as the implementation in fcos, we remove half stride
        offset to align with the original implementation.

        This function will be deprecated soon.
        """
        warnings.warn('`_get_points_single` in `AutoAssignHead` will be deprecated soon, we support a multi level point generator nowyou can get points of a single level feature map with `self.prior_generator.single_level_grid_priors` ')
        y, x = super(FCOSHead, self)._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride), dim=-1)
        return points

@HEADS.register_module()
class YOLOV3Head(BaseDenseHead, BBoxTestMixin):
    """YOLOV3Head Paper link: https://arxiv.org/abs/1804.02767.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): Number of input channels per scale.
        out_channels (List[int]): The number of output channels per scale
            before the final 1x1 layer. Default: (1024, 512, 256).
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        featmap_strides (List[int]): The stride of each scale.
            Should be in descending order. Default: (32, 16, 8).
        one_hot_smoother (float): Set a non-zero value to enable label-smooth
            Default: 0.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        loss_cls (dict): Config of classification loss.
        loss_conf (dict): Config of confidence loss.
        loss_xy (dict): Config of xy coordinate loss.
        loss_wh (dict): Config of wh coordinate loss.
        train_cfg (dict): Training config of YOLOV3 head. Default: None.
        test_cfg (dict): Testing config of YOLOV3 head. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, num_classes, in_channels, out_channels=(1024, 512, 256), anchor_generator=dict(type='YOLOAnchorGenerator', base_sizes=[[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)], [(10, 13), (16, 30), (33, 23)]], strides=[32, 16, 8]), bbox_coder=dict(type='YOLOBBoxCoder'), featmap_strides=[32, 16, 8], one_hot_smoother=0.0, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='LeakyReLU', negative_slope=0.1), loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_conf=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_xy=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_wh=dict(type='MSELoss', loss_weight=1.0), train_cfg=None, test_cfg=None, init_cfg=dict(type='Normal', std=0.01, override=dict(name='convs_pred'))):
        super(YOLOV3Head, self).__init__(init_cfg)
        assert len(in_channels) == len(out_channels) == len(featmap_strides)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False
        self.one_hot_smoother = one_hot_smoother
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.prior_generator = build_prior_generator(anchor_generator)
        self.loss_cls = build_loss(loss_cls)
        self.loss_conf = build_loss(loss_conf)
        self.loss_xy = build_loss(loss_xy)
        self.loss_wh = build_loss(loss_wh)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        assert len(self.prior_generator.num_base_priors) == len(featmap_strides)
        self._init_layers()

    @property
    def anchor_generator(self):
        warnings.warn('DeprecationWarning: `anchor_generator` is deprecated, please use "prior_generator" instead')
        return self.prior_generator

    @property
    def num_anchors(self):
        """
        Returns:
            int: Number of anchors on each point of feature map.
        """
        warnings.warn('DeprecationWarning: `num_anchors` is deprecated, please use "num_base_priors" instead')
        return self.num_base_priors

    @property
    def num_levels(self):
        return len(self.featmap_strides)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""
        return 5 + self.num_classes

    def _init_layers(self):
        self.convs_bridge = nn.ModuleList()
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_bridge = ConvModule(self.in_channels[i], self.out_channels[i], 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            conv_pred = nn.Conv2d(self.out_channels[i], self.num_base_priors * self.num_attrib, 1)
            self.convs_bridge.append(conv_bridge)
            self.convs_pred.append(conv_pred)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
        for conv_pred, stride in zip(self.convs_pred, self.featmap_strides):
            bias = conv_pred.bias.reshape(self.num_base_priors, -1)
            nn.init.constant_(bias.data[:, 4], bias_init_with_prob(8 / (608 / stride) ** 2))
            nn.init.constant_(bias.data[:, 5:], bias_init_with_prob(0.01))

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """
        assert len(feats) == self.num_levels
        pred_maps = []
        for i in range(self.num_levels):
            x = feats[i]
            x = self.convs_bridge[i](x)
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)
        return (tuple(pred_maps),)

    @force_fp32(apply_to=('pred_maps',))
    def get_bboxes(self, pred_maps, img_metas, cfg=None, rescale=False, with_nms=True):
        """Transform network output for a batch into bbox predictions. It has
        been accelerated since PR #5991.

        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(pred_maps) == self.num_levels
        cfg = self.test_cfg if cfg is None else cfg
        scale_factors = np.array([img_meta['scale_factor'] for img_meta in img_metas])
        num_imgs = len(img_metas)
        featmap_sizes = [pred_map.shape[-2:] for pred_map in pred_maps]
        mlvl_anchors = self.prior_generator.grid_priors(featmap_sizes, device=pred_maps[0].device)
        flatten_preds = []
        flatten_strides = []
        for pred, stride in zip(pred_maps, self.featmap_strides):
            pred = pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_attrib)
            pred[..., :2].sigmoid_()
            flatten_preds.append(pred)
            flatten_strides.append(pred.new_tensor(stride).expand(pred.size(1)))
        flatten_preds = torch.cat(flatten_preds, dim=1)
        flatten_bbox_preds = flatten_preds[..., :4]
        flatten_objectness = flatten_preds[..., 4].sigmoid()
        flatten_cls_scores = flatten_preds[..., 5:].sigmoid()
        flatten_anchors = torch.cat(mlvl_anchors)
        flatten_strides = torch.cat(flatten_strides)
        flatten_bboxes = self.bbox_coder.decode(flatten_anchors, flatten_bbox_preds, flatten_strides.unsqueeze(-1))
        if with_nms and flatten_objectness.size(0) == 0:
            return (torch.zeros((0, 5)), torch.zeros((0,)))
        if rescale:
            flatten_bboxes /= flatten_bboxes.new_tensor(scale_factors).unsqueeze(1)
        padding = flatten_bboxes.new_zeros(num_imgs, flatten_bboxes.shape[1], 1)
        flatten_cls_scores = torch.cat([flatten_cls_scores, padding], dim=-1)
        det_results = []
        for bboxes, scores, objectness in zip(flatten_bboxes, flatten_cls_scores, flatten_objectness):
            conf_thr = cfg.get('conf_thr', -1)
            if conf_thr > 0:
                conf_inds = objectness >= conf_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]
            det_bboxes, det_labels = multiclass_nms(bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img, score_factors=objectness)
            det_results.append(tuple([det_bboxes, det_labels]))
        return det_results

    @force_fp32(apply_to=('pred_maps',))
    def loss(self, pred_maps, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            pred_maps (list[Tensor]): Prediction map for each scale level,
                shape (N, num_anchors * num_attrib, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(img_metas)
        device = pred_maps[0][0].device
        featmap_sizes = [pred_maps[i].shape[-2:] for i in range(self.num_levels)]
        mlvl_anchors = self.prior_generator.grid_priors(featmap_sizes, device=device)
        anchor_list = [mlvl_anchors for _ in range(num_imgs)]
        responsible_flag_list = []
        for img_id in range(len(img_metas)):
            responsible_flag_list.append(self.prior_generator.responsible_flags(featmap_sizes, gt_bboxes[img_id], device))
        target_maps_list, neg_maps_list = self.get_targets(anchor_list, responsible_flag_list, gt_bboxes, gt_labels)
        losses_cls, losses_conf, losses_xy, losses_wh = multi_apply(self.loss_single, pred_maps, target_maps_list, neg_maps_list)
        return dict(loss_cls=losses_cls, loss_conf=losses_conf, loss_xy=losses_xy, loss_wh=losses_wh)

    def loss_single(self, pred_map, target_map, neg_map):
        """Compute loss of a single image from a batch.

        Args:
            pred_map (Tensor): Raw predictions for a single level.
            target_map (Tensor): The Ground-Truth target for a single level.
            neg_map (Tensor): The negative masks for a single level.

        Returns:
            tuple:
                loss_cls (Tensor): Classification loss.
                loss_conf (Tensor): Confidence loss.
                loss_xy (Tensor): Regression loss of x, y coordinate.
                loss_wh (Tensor): Regression loss of w, h coordinate.
        """
        num_imgs = len(pred_map)
        pred_map = pred_map.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_attrib)
        neg_mask = neg_map.float()
        pos_mask = target_map[..., 4]
        pos_and_neg_mask = neg_mask + pos_mask
        pos_mask = pos_mask.unsqueeze(dim=-1)
        if torch.max(pos_and_neg_mask) > 1.0:
            warnings.warn('There is overlap between pos and neg sample.')
            pos_and_neg_mask = pos_and_neg_mask.clamp(min=0.0, max=1.0)
        pred_xy = pred_map[..., :2]
        pred_wh = pred_map[..., 2:4]
        pred_conf = pred_map[..., 4]
        pred_label = pred_map[..., 5:]
        target_xy = target_map[..., :2]
        target_wh = target_map[..., 2:4]
        target_conf = target_map[..., 4]
        target_label = target_map[..., 5:]
        loss_cls = self.loss_cls(pred_label, target_label, weight=pos_mask)
        loss_conf = self.loss_conf(pred_conf, target_conf, weight=pos_and_neg_mask)
        loss_xy = self.loss_xy(pred_xy, target_xy, weight=pos_mask)
        loss_wh = self.loss_wh(pred_wh, target_wh, weight=pos_mask)
        return (loss_cls, loss_conf, loss_xy, loss_wh)

    def get_targets(self, anchor_list, responsible_flag_list, gt_bboxes_list, gt_labels_list):
        """Compute target maps for anchors in multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_total_anchors, 4).
            responsible_flag_list (list[list[Tensor]]): Multi level responsible
                flags of each image. Each element is a tensor of shape
                (num_total_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.

        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - target_map_list (list[Tensor]): Target map of each level.
                - neg_map_list (list[Tensor]): Negative map of each level.
        """
        num_imgs = len(anchor_list)
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        results = multi_apply(self._get_targets_single, anchor_list, responsible_flag_list, gt_bboxes_list, gt_labels_list)
        all_target_maps, all_neg_maps = results
        assert num_imgs == len(all_target_maps) == len(all_neg_maps)
        target_maps_list = images_to_levels(all_target_maps, num_level_anchors)
        neg_maps_list = images_to_levels(all_neg_maps, num_level_anchors)
        return (target_maps_list, neg_maps_list)

    def _get_targets_single(self, anchors, responsible_flags, gt_bboxes, gt_labels):
        """Generate matching bounding box prior and converted GT.

        Args:
            anchors (list[Tensor]): Multi-level anchors of the image.
            responsible_flags (list[Tensor]): Multi-level responsible flags of
                anchors
            gt_bboxes (Tensor): Ground truth bboxes of single image.
            gt_labels (Tensor): Ground truth labels of single image.

        Returns:
            tuple:
                target_map (Tensor): Predication target map of each
                    scale level, shape (num_total_anchors,
                    5+num_classes)
                neg_map (Tensor): Negative map of each scale level,
                    shape (num_total_anchors,)
        """
        anchor_strides = []
        for i in range(len(anchors)):
            anchor_strides.append(torch.tensor(self.featmap_strides[i], device=gt_bboxes.device).repeat(len(anchors[i])))
        concat_anchors = torch.cat(anchors)
        concat_responsible_flags = torch.cat(responsible_flags)
        anchor_strides = torch.cat(anchor_strides)
        assert len(anchor_strides) == len(concat_anchors) == len(concat_responsible_flags)
        assign_result = self.assigner.assign(concat_anchors, concat_responsible_flags, gt_bboxes)
        sampling_result = self.sampler.sample(assign_result, concat_anchors, gt_bboxes)
        target_map = concat_anchors.new_zeros(concat_anchors.size(0), self.num_attrib)
        target_map[sampling_result.pos_inds, :4] = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes, anchor_strides[sampling_result.pos_inds])
        target_map[sampling_result.pos_inds, 4] = 1
        gt_labels_one_hot = F.one_hot(gt_labels, num_classes=self.num_classes).float()
        if self.one_hot_smoother != 0:
            gt_labels_one_hot = gt_labels_one_hot * (1 - self.one_hot_smoother) + self.one_hot_smoother / self.num_classes
        target_map[sampling_result.pos_inds, 5:] = gt_labels_one_hot[sampling_result.pos_assigned_gt_inds]
        neg_map = concat_anchors.new_zeros(concat_anchors.size(0), dtype=torch.uint8)
        neg_map[sampling_result.neg_inds] = 1
        return (target_map, neg_map)

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
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)

    @force_fp32(apply_to='pred_maps')
    def onnx_export(self, pred_maps, img_metas, with_nms=True):
        num_levels = len(pred_maps)
        pred_maps_list = [pred_maps[i].detach() for i in range(num_levels)]
        cfg = self.test_cfg
        assert len(pred_maps_list) == self.num_levels
        device = pred_maps_list[0].device
        batch_size = pred_maps_list[0].shape[0]
        featmap_sizes = [pred_maps_list[i].shape[-2:] for i in range(self.num_levels)]
        mlvl_anchors = self.prior_generator.grid_priors(featmap_sizes, device=device)
        nms_pre_tensor = torch.tensor(cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        multi_lvl_bboxes = []
        multi_lvl_cls_scores = []
        multi_lvl_conf_scores = []
        for i in range(self.num_levels):
            pred_map = pred_maps_list[i]
            stride = self.featmap_strides[i]
            pred_map = pred_map.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_attrib)
            pred_map_conf = torch.sigmoid(pred_map[..., :2])
            pred_map_rest = pred_map[..., 2:]
            pred_map = torch.cat([pred_map_conf, pred_map_rest], dim=-1)
            pred_map_boxes = pred_map[..., :4]
            multi_lvl_anchor = mlvl_anchors[i]
            multi_lvl_anchor = multi_lvl_anchor.expand_as(pred_map_boxes)
            bbox_pred = self.bbox_coder.decode(multi_lvl_anchor, pred_map_boxes, stride)
            conf_pred = torch.sigmoid(pred_map[..., 4])
            cls_pred = torch.sigmoid(pred_map[..., 5:]).view(batch_size, -1, self.num_classes)
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                _, topk_inds = conf_pred.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(-1, 1).expand_as(topk_inds).long()
                transformed_inds = bbox_pred.shape[1] * batch_inds + topk_inds
                bbox_pred = bbox_pred.reshape(-1, 4)[transformed_inds, :].reshape(batch_size, -1, 4)
                cls_pred = cls_pred.reshape(-1, self.num_classes)[transformed_inds, :].reshape(batch_size, -1, self.num_classes)
                conf_pred = conf_pred.reshape(-1, 1)[transformed_inds].reshape(batch_size, -1)
            multi_lvl_bboxes.append(bbox_pred)
            multi_lvl_cls_scores.append(cls_pred)
            multi_lvl_conf_scores.append(conf_pred)
        batch_mlvl_bboxes = torch.cat(multi_lvl_bboxes, dim=1)
        batch_mlvl_scores = torch.cat(multi_lvl_cls_scores, dim=1)
        batch_mlvl_conf_scores = torch.cat(multi_lvl_conf_scores, dim=1)
        from mmdet.core.export import add_dummy_nms_for_onnx
        conf_thr = cfg.get('conf_thr', -1)
        score_thr = cfg.get('score_thr', -1)
        if conf_thr > 0:
            mask = (batch_mlvl_conf_scores >= conf_thr).float()
            batch_mlvl_conf_scores *= mask
        if score_thr > 0:
            mask = (batch_mlvl_scores > score_thr).float()
            batch_mlvl_scores *= mask
        batch_mlvl_conf_scores = batch_mlvl_conf_scores.unsqueeze(2).expand_as(batch_mlvl_scores)
        batch_mlvl_scores = batch_mlvl_scores * batch_mlvl_conf_scores
        if with_nms:
            max_output_boxes_per_class = cfg.nms.get('max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = 0
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores, max_output_boxes_per_class, iou_threshold, score_threshold, nms_pre, cfg.max_per_img)
        else:
            return (batch_mlvl_bboxes, batch_mlvl_scores)

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
class StageCascadeRPNHead(RPNHead):
    """Stage of CascadeRPNHead.

    Args:
        in_channels (int): Number of channels in the input feature map.
        anchor_generator (dict): anchor generator config.
        adapt_cfg (dict): adaptation config.
        bridged_feature (bool, optional): whether update rpn feature.
            Default: False.
        with_cls (bool, optional): whether use classification branch.
            Default: True.
        sampling (bool, optional): whether use sampling. Default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self, in_channels, anchor_generator=dict(type='AnchorGenerator', scales=[8], ratios=[1.0], strides=[4, 8, 16, 32, 64]), adapt_cfg=dict(type='dilation', dilation=3), bridged_feature=False, with_cls=True, sampling=True, init_cfg=None, **kwargs):
        self.with_cls = with_cls
        self.anchor_strides = anchor_generator['strides']
        self.anchor_scales = anchor_generator['scales']
        self.bridged_feature = bridged_feature
        self.adapt_cfg = adapt_cfg
        super(StageCascadeRPNHead, self).__init__(in_channels, anchor_generator=anchor_generator, init_cfg=init_cfg, **kwargs)
        self.sampling = sampling
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        if init_cfg is None:
            self.init_cfg = dict(type='Normal', std=0.01, override=[dict(name='rpn_reg')])
            if self.with_cls:
                self.init_cfg['override'].append(dict(name='rpn_cls'))

    def _init_layers(self):
        """Init layers of a CascadeRPN stage."""
        self.rpn_conv = AdaptiveConv(self.in_channels, self.feat_channels, **self.adapt_cfg)
        if self.with_cls:
            self.rpn_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward_single(self, x, offset):
        """Forward function of single scale."""
        bridged_x = x
        x = self.relu(self.rpn_conv(x, offset))
        if self.bridged_feature:
            bridged_x = x
        cls_score = self.rpn_cls(x) if self.with_cls else None
        bbox_pred = self.rpn_reg(x)
        return (bridged_x, cls_score, bbox_pred)

    def forward(self, feats, offset_list=None):
        """Forward function."""
        if offset_list is None:
            offset_list = [None for _ in range(len(feats))]
        return multi_apply(self.forward_single, feats, offset_list)

    def _region_targets_single(self, anchors, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, featmap_sizes, label_channels=1):
        """Get anchor targets based on region for single level."""
        assign_result = self.assigner.assign(anchors, valid_flags, gt_bboxes, img_meta, featmap_sizes, self.anchor_scales[0], self.anchor_strides, gt_bboxes_ignore=gt_bboxes_ignore, gt_labels=None, allowed_border=self.train_cfg.allowed_border)
        flat_anchors = torch.cat(anchors)
        sampling_result = self.sampler.sample(assign_result, flat_anchors, gt_bboxes)
        num_anchors = flat_anchors.shape[0]
        bbox_targets = torch.zeros_like(flat_anchors)
        bbox_weights = torch.zeros_like(flat_anchors)
        labels = flat_anchors.new_zeros(num_anchors, dtype=torch.long)
        label_weights = flat_anchors.new_zeros(num_anchors, dtype=torch.float)
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
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def region_targets(self, anchor_list, valid_flag_list, gt_bboxes_list, img_metas, featmap_sizes, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, unmap_outputs=True):
        """See :func:`StageCascadeRPNHead.get_targets`."""
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, pos_inds_list, neg_inds_list = multi_apply(self._region_targets_single, anchor_list, valid_flag_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, featmap_sizes=featmap_sizes, label_channels=label_channels)
        if any([labels is None for labels in all_labels]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)

    def get_targets(self, anchor_list, valid_flag_list, gt_bboxes, img_metas, featmap_sizes, gt_bboxes_ignore=None, label_channels=1):
        """Compute regression and classification targets for anchors.

        Args:
            anchor_list (list[list]): Multi level anchors of each image.
            valid_flag_list (list[list]): Multi level valid flags of each
                image.
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            featmap_sizes (list[Tensor]): Feature mapsize each level
            gt_bboxes_ignore (list[Tensor]): Ignore bboxes of each images
            label_channels (int): Channel of label.

        Returns:
            cls_reg_targets (tuple)
        """
        if isinstance(self.assigner, RegionAssigner):
            cls_reg_targets = self.region_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas, featmap_sizes, gt_bboxes_ignore_list=gt_bboxes_ignore, label_channels=label_channels)
        else:
            cls_reg_targets = super(StageCascadeRPNHead, self).get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, label_channels=label_channels)
        return cls_reg_targets

    def anchor_offset(self, anchor_list, anchor_strides, featmap_sizes):
        """ Get offset for deformable conv based on anchor shape
        NOTE: currently support deformable kernel_size=3 and dilation=1

        Args:
            anchor_list (list[list[tensor])): [NI, NLVL, NA, 4] list of
                multi-level anchors
            anchor_strides (list[int]): anchor stride of each level

        Returns:
            offset_list (list[tensor]): [NLVL, NA, 2, 18]: offset of DeformConv
                kernel.
        """

        def _shape_offset(anchors, stride, ks=3, dilation=1):
            assert ks == 3 and dilation == 1
            pad = (ks - 1) // 2
            idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
            yy, xx = torch.meshgrid(idx, idx)
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)
            w = (anchors[:, 2] - anchors[:, 0]) / stride
            h = (anchors[:, 3] - anchors[:, 1]) / stride
            w = w / (ks - 1) - dilation
            h = h / (ks - 1) - dilation
            offset_x = w[:, None] * xx
            offset_y = h[:, None] * yy
            return (offset_x, offset_y)

        def _ctr_offset(anchors, stride, featmap_size):
            feat_h, feat_w = featmap_size
            assert len(anchors) == feat_h * feat_w
            x = (anchors[:, 0] + anchors[:, 2]) * 0.5
            y = (anchors[:, 1] + anchors[:, 3]) * 0.5
            x = x / stride
            y = y / stride
            xx = torch.arange(0, feat_w, device=anchors.device)
            yy = torch.arange(0, feat_h, device=anchors.device)
            yy, xx = torch.meshgrid(yy, xx)
            xx = xx.reshape(-1).type_as(x)
            yy = yy.reshape(-1).type_as(y)
            offset_x = x - xx
            offset_y = y - yy
            return (offset_x, offset_y)
        num_imgs = len(anchor_list)
        num_lvls = len(anchor_list[0])
        dtype = anchor_list[0][0].dtype
        device = anchor_list[0][0].device
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        offset_list = []
        for i in range(num_imgs):
            mlvl_offset = []
            for lvl in range(num_lvls):
                c_offset_x, c_offset_y = _ctr_offset(anchor_list[i][lvl], anchor_strides[lvl], featmap_sizes[lvl])
                s_offset_x, s_offset_y = _shape_offset(anchor_list[i][lvl], anchor_strides[lvl])
                offset_x = s_offset_x + c_offset_x[:, None]
                offset_y = s_offset_y + c_offset_y[:, None]
                offset = torch.stack([offset_y, offset_x], dim=-1)
                offset = offset.reshape(offset.size(0), -1)
                mlvl_offset.append(offset)
            offset_list.append(torch.cat(mlvl_offset))
        offset_list = images_to_levels(offset_list, num_level_anchors)
        return offset_list

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights, bbox_targets, bbox_weights, num_total_samples):
        """Loss function on single scale."""
        if self.with_cls:
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
        loss_reg = self.loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples)
        if self.with_cls:
            return (loss_cls, loss_reg)
        return (None, loss_reg)

    def loss(self, anchor_list, valid_flag_list, cls_scores, bbox_preds, gt_bboxes, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            anchor_list (list[list]): Multi level anchors of each image.
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas, featmap_sizes, gt_bboxes_ignore=gt_bboxes_ignore, label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        if self.sampling:
            num_total_samples = num_total_pos + num_total_neg
        else:
            num_total_samples = sum([label.numel() for label in labels_list]) / 200.0
        mlvl_anchor_list = list(zip(*anchor_list))
        mlvl_anchor_list = [torch.cat(anchors, dim=0) for anchors in mlvl_anchor_list]
        losses = multi_apply(self.loss_single, cls_scores, bbox_preds, mlvl_anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_samples=num_total_samples)
        if self.with_cls:
            return dict(loss_rpn_cls=losses[0], loss_rpn_reg=losses[1])
        return dict(loss_rpn_reg=losses[1])

    def get_bboxes(self, anchor_list, cls_scores, bbox_preds, img_metas, cfg, rescale=False):
        """Get proposal predict.

        Args:
            anchor_list (list[list]): Multi level anchors of each image.
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        assert len(cls_scores) == len(bbox_preds)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list, anchor_list[img_id], img_shape, scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self, cls_scores, bbox_preds, mlvl_anchors, img_shape, scale_factor, cfg, rescale=False):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has
                shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference from all scale
                levels of a single image, each item has shape
                (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default False.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get('nms_pre', -1)
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if 0 < nms_pre < scores.shape[0]:
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(scores.new_full((scores.size(0),), idx, dtype=torch.long))
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)
        if cfg.min_bbox_size >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]
        if 'nms' not in cfg or 'max_num' in cfg or 'nms_thr' in cfg:
            warnings.warn('In rpn_proposal or test_cfg, nms_thr has been moved to a dict named nms as iou_threshold, max_num has been renamed as max_per_img, name of original arguments and the way to specify iou_threshold of NMS will be deprecated.')
        if 'nms' not in cfg:
            cfg.nms = ConfigDict(dict(type='nms', iou_threshold=cfg.nms_thr))
        if 'max_num' in cfg:
            if 'max_per_img' in cfg:
                assert cfg.max_num == cfg.max_per_img, f'You set max_num and max_per_img at the same time, but get {cfg.max_num} and {cfg.max_per_img} respectivelyPlease delete max_num which will be deprecated.'
            else:
                cfg.max_per_img = cfg.max_num
        if 'nms_thr' in cfg:
            assert cfg.nms.iou_threshold == cfg.nms_thr, f'You set iou_threshold in nms and nms_thr at the same time, but get {cfg.nms.iou_threshold} and {cfg.nms_thr} respectively. Please delete the nms_thr which will be deprecated.'
        if proposals.numel() > 0:
            dets, _ = batched_nms(proposals, scores, ids, cfg.nms)
        else:
            return proposals.new_zeros(0, 5)
        return dets[:cfg.max_per_img]

    def refine_bboxes(self, anchor_list, bbox_preds, img_metas):
        """Refine bboxes through stages."""
        num_levels = len(bbox_preds)
        new_anchor_list = []
        for img_id in range(len(img_metas)):
            mlvl_anchors = []
            for i in range(num_levels):
                bbox_pred = bbox_preds[i][img_id].detach()
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
                img_shape = img_metas[img_id]['img_shape']
                bboxes = self.bbox_coder.decode(anchor_list[img_id][i], bbox_pred, img_shape)
                mlvl_anchors.append(bboxes)
            new_anchor_list.append(mlvl_anchors)
        return new_anchor_list

@HEADS.register_module()
class SSDHead(AnchorHead):
    """SSD head used in https://arxiv.org/abs/1512.02325.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 0.
        feat_channels (int): Number of hidden channels when stacked_convs
            > 0. Default: 256.
        use_depthwise (bool): Whether to use DepthwiseSeparableConv.
            Default: False.
        conv_cfg (dict): Dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: None.
        act_cfg (dict): Dictionary to construct and config activation layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, num_classes=80, in_channels=(512, 1024, 512, 256, 256, 256), stacked_convs=0, feat_channels=256, use_depthwise=False, conv_cfg=None, norm_cfg=None, act_cfg=None, anchor_generator=dict(type='SSDAnchorGenerator', scale_major=False, input_size=300, strides=[8, 16, 32, 64, 100, 300], ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]), basesize_ratio_range=(0.1, 0.9)), bbox_coder=dict(type='DeltaXYWHBBoxCoder', clip_border=True, target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]), reg_decoded_bbox=False, train_cfg=None, test_cfg=None, init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform', bias=0)):
        super(AnchorHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.use_depthwise = use_depthwise
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.cls_out_channels = num_classes + 1
        self.prior_generator = build_prior_generator(anchor_generator)
        self.num_base_priors = self.prior_generator.num_base_priors
        self._init_layers()
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

    @property
    def num_anchors(self):
        """
        Returns:
            list[int]: Number of base_anchors on each point of each level.
        """
        warnings.warn('DeprecationWarning: `num_anchors` is deprecated, please use "num_base_priors" instead')
        return self.num_base_priors

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        conv = DepthwiseSeparableConvModule if self.use_depthwise else ConvModule
        for channel, num_base_priors in zip(self.in_channels, self.num_base_priors):
            cls_layers = []
            reg_layers = []
            in_channel = channel
            for i in range(self.stacked_convs):
                cls_layers.append(conv(in_channel, self.feat_channels, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
                reg_layers.append(conv(in_channel, self.feat_channels, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
                in_channel = self.feat_channels
            if self.use_depthwise:
                cls_layers.append(ConvModule(in_channel, in_channel, 3, padding=1, groups=in_channel, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
                reg_layers.append(ConvModule(in_channel, in_channel, 3, padding=1, groups=in_channel, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
            cls_layers.append(nn.Conv2d(in_channel, num_base_priors * self.cls_out_channels, kernel_size=1 if self.use_depthwise else 3, padding=0 if self.use_depthwise else 1))
            reg_layers.append(nn.Conv2d(in_channel, num_base_priors * 4, kernel_size=1 if self.use_depthwise else 3, padding=0 if self.use_depthwise else 1))
            self.cls_convs.append(nn.Sequential(*cls_layers))
            self.reg_convs.append(nn.Sequential(*reg_layers))

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs, self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return (cls_scores, bbox_preds)

    def loss_single(self, cls_score, bbox_pred, anchor, labels, label_weights, bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single image.

        Args:
            cls_score (Tensor): Box scores for eachimage
                Has shape (num_total_anchors, num_classes).
            bbox_pred (Tensor): Box energies / deltas for each image
                level with shape (num_total_anchors, 4).
            anchors (Tensor): Box reference for each scale level with shape
                (num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (num_total_anchors,).
            label_weights (Tensor): Label weights of each anchor with shape
                (num_total_anchors,)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_cls_all = F.cross_entropy(cls_score, labels, reduction='none') * label_weights
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(as_tuple=False).reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero(as_tuple=False).view(-1)
        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples
        if self.reg_decoded_bbox:
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)
        loss_bbox = smooth_l1_loss(bbox_pred, bbox_targets, bbox_weights, beta=self.train_cfg.smoothl1_beta, avg_factor=num_total_samples)
        return (loss_cls[None], loss_bbox)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
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
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=1, unmap_outputs=True)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        num_images = len(img_metas)
        all_cls_scores = torch.cat([s.permute(0, 2, 3, 1).reshape(num_images, -1, self.cls_out_channels) for s in cls_scores], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list, -1).view(num_images, -1)
        all_bbox_preds = torch.cat([b.permute(0, 2, 3, 1).reshape(num_images, -1, 4) for b in bbox_preds], -2)
        all_bbox_targets = torch.cat(bbox_targets_list, -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list, -2).view(num_images, -1, 4)
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))
        losses_cls, losses_bbox = multi_apply(self.loss_single, all_cls_scores, all_bbox_preds, all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, num_total_samples=num_total_pos)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

@HEADS.register_module()
class MaskFormerHead(AnchorFreeHead):
    """Implements the MaskFormer head.

    See `Per-Pixel Classification is Not All You Need for Semantic
    Segmentation <https://arxiv.org/pdf/2107.06278>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for feature.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add a layer
            to change the embed_dim of tranformer encoder in pixel decoder to
            the embed_dim of transformer decoder. Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (:obj:`mmcv.ConfigDict` | dict): Config of the classification
            loss. Defaults to `CrossEntropyLoss`.
        loss_mask (:obj:`mmcv.ConfigDict` | dict): Config of the mask loss.
            Defaults to `FocalLoss`.
        loss_dice (:obj:`mmcv.ConfigDict` | dict): Config of the dice loss.
            Defaults to `DiceLoss`.
        train_cfg (:obj:`mmcv.ConfigDict` | dict): Training config of
            Maskformer head.
        test_cfg (:obj:`mmcv.ConfigDict` | dict): Testing config of Maskformer
            head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self, in_channels, feat_channels, out_channels, num_things_classes=80, num_stuff_classes=53, num_queries=100, pixel_decoder=None, enforce_decoder_input_project=False, transformer_decoder=None, positional_encoding=None, loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0] * 133 + [0.1]), loss_mask=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=20.0), loss_dice=dict(type='DiceLoss', use_sigmoid=True, activate=True, naive_dice=True, loss_weight=1.0), train_cfg=None, test_cfg=None, init_cfg=None, **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        pixel_decoder.update(in_channels=in_channels, feat_channels=feat_channels, out_channels=out_channels)
        self.pixel_decoder = build_plugin_layer(pixel_decoder)[1]
        self.transformer_decoder = build_transformer_layer_sequence(transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        pixel_decoder_type = pixel_decoder.get('type')
        if pixel_decoder_type == 'PixelDecoder' and (self.decoder_embed_dims != in_channels[-1] or enforce_decoder_input_project):
            self.decoder_input_proj = Conv2d(in_channels[-1], self.decoder_embed_dims, kernel_size=1)
        else:
            self.decoder_input_proj = nn.Identity()
        self.decoder_pe = build_positional_encoding(positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, out_channels)
        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True), nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True), nn.Linear(feat_channels, out_channels))
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(train_cfg.get('assigner', None))
            self.sampler = build_sampler(train_cfg.get('sampler', None), context=self)
        self.class_weight = loss_cls.get('class_weight', None)
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

    def init_weights(self):
        if isinstance(self.decoder_input_proj, Conv2d):
            caffe2_xavier_init(self.decoder_input_proj, bias=0)
        self.pixel_decoder.init_weights()
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def preprocess_gt(self, gt_labels_list, gt_masks_list, gt_semantic_segs, img_metas):
        """Preprocess the ground truth for all images.

        Args:
            gt_labels_list (list[Tensor]): Each is ground truth
                labels of each bbox, with shape (num_gts, ).
            gt_masks_list (list[BitmapMasks]): Each is ground truth
                masks of each instances of a image, shape
                (num_gts, h, w).
            gt_semantic_seg (Tensor | None): Ground truth of semantic
                segmentation with the shape (batch_size, n, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple: a tuple containing the following targets.
                - labels (list[Tensor]): Ground truth class indices                    for all images. Each with shape (n, ), n is the sum of                    number of stuff type and number of instance in a image.
                - masks (list[Tensor]): Ground truth mask for each                    image, each with shape (n, h, w).
        """
        num_things_list = [self.num_things_classes] * len(gt_labels_list)
        num_stuff_list = [self.num_stuff_classes] * len(gt_labels_list)
        if gt_semantic_segs is None:
            gt_semantic_segs = [None] * len(gt_labels_list)
        targets = multi_apply(preprocess_panoptic_gt, gt_labels_list, gt_masks_list, gt_semantic_segs, num_things_list, num_stuff_list, img_metas)
        labels, masks = targets
        return (labels, masks)

    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list, gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels of all images.                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of                    all images. Each with shape (num_queries, ).
                - num_total_pos (int): Number of positive samples in                    all images.
                - num_total_neg (int): Number of negative samples in                    all images.
        """
        labels_list, label_weights_list, mask_targets_list, mask_weights_list, pos_inds_list, neg_inds_list = multi_apply(self._get_target_single, cls_scores_list, mask_preds_list, gt_labels_list, gt_masks_list, img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list, mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks, img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (n, ). n is the sum of number of stuff type and number
                of instance in a image.
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (n, h, w).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image.
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image.
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image.
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        target_shape = mask_pred.shape[-2:]
        if gt_masks.shape[0] > 0:
            gt_masks_downsampled = F.interpolate(gt_masks.unsqueeze(1).float(), target_shape, mode='nearest').squeeze(1).long()
        else:
            gt_masks_downsampled = gt_masks
        assign_result = self.assigner.assign(cls_score, mask_pred, gt_labels, gt_masks_downsampled, img_metas)
        sampling_result = self.sampler.sample(assign_result, mask_pred, gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        labels = gt_labels.new_full((self.num_queries,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones(self.num_queries)
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries,))
        mask_weights[pos_inds] = 1.0
        return (labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds)

    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list, gt_masks_list, img_metas):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice = multi_apply(self.loss_single, all_cls_scores, all_mask_preds, all_gt_labels_list, all_gt_masks_list, img_metas_list)
        loss_dict = dict()
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self, cls_scores, mask_preds, gt_labels_list, gt_masks_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (n, ). n is the sum of number of stuff
                types and number of instances in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single decoder                layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        labels_list, label_weights_list, mask_targets_list, mask_weights_list, num_total_pos, num_total_neg = self.get_targets(cls_scores_list, mask_preds_list, gt_labels_list, gt_masks_list, img_metas)
        labels = torch.stack(labels_list, dim=0)
        label_weights = torch.stack(label_weights_list, dim=0)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        mask_weights = torch.stack(mask_weights_list, dim=0)
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)
        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=class_weight[labels].sum())
        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)
        mask_preds = mask_preds[mask_weights > 0]
        target_shape = mask_targets.shape[-2:]
        if mask_targets.shape[0] == 0:
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return (loss_cls, loss_mask, loss_dice)
        mask_preds = F.interpolate(mask_preds.unsqueeze(1), target_shape, mode='bilinear', align_corners=False).squeeze(1)
        loss_dice = self.loss_dice(mask_preds, mask_targets, avg_factor=num_total_masks)
        h, w = mask_preds.shape[-2:]
        mask_preds = mask_preds.reshape(-1, 1)
        mask_targets = mask_targets.reshape(-1)
        loss_mask = self.loss_mask(mask_preds, 1 - mask_targets, avg_factor=num_total_masks * h * w)
        return (loss_cls, loss_mask, loss_dice)

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (list[Tensor]): Features from the upstream network, each
                is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: a tuple contains two elements.
                - all_cls_scores (Tensor): Classification scores for each                    scale level. Each is a 4D-tensor with shape                    (num_decoder, batch_size, num_queries, cls_out_channels).                    Note `cls_out_channels` should includes background.
                - all_mask_preds (Tensor): Mask scores for each decoder                    layer. Each with shape (num_decoder, batch_size,                    num_queries, h, w).
        """
        batch_size = len(img_metas)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        padding_mask = feats[-1].new_ones((batch_size, input_img_h, input_img_w), dtype=torch.float32)
        for i in range(batch_size):
            img_h, img_w, _ = img_metas[i]['img_shape']
            padding_mask[i, :img_h, :img_w] = 0
        padding_mask = F.interpolate(padding_mask.unsqueeze(1), size=feats[-1].shape[-2:], mode='nearest').to(torch.bool).squeeze(1)
        mask_features, memory = self.pixel_decoder(feats, img_metas)
        pos_embed = self.decoder_pe(padding_mask)
        memory = self.decoder_input_proj(memory)
        memory = memory.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        padding_mask = padding_mask.flatten(1)
        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        target = torch.zeros_like(query_embed)
        out_dec = self.transformer_decoder(query=target, key=memory, value=memory, key_pos=pos_embed, query_pos=query_embed, key_padding_mask=padding_mask)
        out_dec = out_dec.transpose(1, 2)
        all_cls_scores = self.cls_embed(out_dec)
        mask_embed = self.mask_embed(out_dec)
        all_mask_preds = torch.einsum('lbqc,bchw->lbqhw', mask_embed, mask_features)
        return (all_cls_scores, all_mask_preds)

    def forward_train(self, feats, img_metas, gt_bboxes, gt_labels, gt_masks, gt_semantic_seg, gt_bboxes_ignore=None):
        """Forward function for training mode.

        Args:
            feats (list[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_bboxes (list[Tensor]): Each element is ground truth bboxes of
                the image, shape (num_gts, 4). Not used here.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[tensor] | None): Each element is the ground
                truth of semantic segmentation with the shape (N, H, W).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored. Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        assert gt_bboxes_ignore is None
        all_cls_scores, all_mask_preds = self(feats, img_metas)
        gt_labels, gt_masks = self.preprocess_gt(gt_labels, gt_masks, gt_semantic_seg, img_metas)
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks, img_metas)
        return losses

    def simple_test(self, feats, img_metas, **kwargs):
        """Test without augmentaton.

        Args:
            feats (list[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two tensors.

            - mask_cls_results (Tensor): Mask classification logits,                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            - mask_pred_results (Tensor): Mask logits, shape                 (batch_size, num_queries, h, w).
        """
        all_cls_scores, all_mask_preds = self(feats, img_metas)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        img_shape = img_metas[0]['batch_input_shape']
        mask_pred_results = F.interpolate(mask_pred_results, size=(img_shape[0], img_shape[1]), mode='bilinear', align_corners=False)
        return (mask_cls_results, mask_pred_results)

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
class PAAHead(ATSSHead):
    """Head of PAAAssignment: Probabilistic Anchor Assignment with IoU
    Prediction for Object Detection.

    Code is modified from the `official github repo
    <https://github.com/kkhoot/PAA/blob/master/paa_core
    /modeling/rpn/paa/loss.py>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2007.08103>`_ .

    Args:
        topk (int): Select topk samples with smallest loss in
            each level.
        score_voting (bool): Whether to use score voting in post-process.
        covariance_type : String describing the type of covariance parameters
            to be used in :class:`sklearn.mixture.GaussianMixture`.
            It must be one of:

            - 'full': each component has its own general covariance matrix
            - 'tied': all components share the same general covariance matrix
            - 'diag': each component has its own diagonal covariance matrix
            - 'spherical': each component has its own single variance
            Default: 'diag'. From 'full' to 'spherical', the gmm fitting
            process is faster yet the performance could be influenced. For most
            cases, 'diag' should be a good choice.
    """

    def __init__(self, *args, topk=9, score_voting=True, covariance_type='diag', **kwargs):
        self.topk = topk
        self.with_score_voting = score_voting
        self.covariance_type = covariance_type
        super(PAAHead, self).__init__(*args, **kwargs)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def loss(self, cls_scores, bbox_preds, iou_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            iou_preds (list[Tensor]): iou_preds for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when are computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss gmm_assignment.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels)
        labels, labels_weight, bboxes_target, bboxes_weight, pos_inds, pos_gt_index = cls_reg_targets
        cls_scores = levels_to_images(cls_scores)
        cls_scores = [item.reshape(-1, self.cls_out_channels) for item in cls_scores]
        bbox_preds = levels_to_images(bbox_preds)
        bbox_preds = [item.reshape(-1, 4) for item in bbox_preds]
        iou_preds = levels_to_images(iou_preds)
        iou_preds = [item.reshape(-1, 1) for item in iou_preds]
        pos_losses_list, = multi_apply(self.get_pos_loss, anchor_list, cls_scores, bbox_preds, labels, labels_weight, bboxes_target, bboxes_weight, pos_inds)
        with torch.no_grad():
            reassign_labels, reassign_label_weight, reassign_bbox_weights, num_pos = multi_apply(self.paa_reassign, pos_losses_list, labels, labels_weight, bboxes_weight, pos_inds, pos_gt_index, anchor_list)
            num_pos = sum(num_pos)
        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        bbox_preds = torch.cat(bbox_preds, 0).view(-1, bbox_preds[0].size(-1))
        iou_preds = torch.cat(iou_preds, 0).view(-1, iou_preds[0].size(-1))
        labels = torch.cat(reassign_labels, 0).view(-1)
        flatten_anchors = torch.cat([torch.cat(item, 0) for item in anchor_list])
        labels_weight = torch.cat(reassign_label_weight, 0).view(-1)
        bboxes_target = torch.cat(bboxes_target, 0).view(-1, bboxes_target[0].size(-1))
        pos_inds_flatten = ((labels >= 0) & (labels < self.num_classes)).nonzero().reshape(-1)
        losses_cls = self.loss_cls(cls_scores, labels, labels_weight, avg_factor=max(num_pos, len(img_metas)))
        if num_pos:
            pos_bbox_pred = self.bbox_coder.decode(flatten_anchors[pos_inds_flatten], bbox_preds[pos_inds_flatten])
            pos_bbox_target = bboxes_target[pos_inds_flatten]
            iou_target = bbox_overlaps(pos_bbox_pred.detach(), pos_bbox_target, is_aligned=True)
            losses_iou = self.loss_centerness(iou_preds[pos_inds_flatten], iou_target.unsqueeze(-1), avg_factor=num_pos)
            losses_bbox = self.loss_bbox(pos_bbox_pred, pos_bbox_target, iou_target.clamp(min=EPS), avg_factor=iou_target.sum())
        else:
            losses_iou = iou_preds.sum() * 0
            losses_bbox = bbox_preds.sum() * 0
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_iou=losses_iou)

    def get_pos_loss(self, anchors, cls_score, bbox_pred, label, label_weight, bbox_target, bbox_weight, pos_inds):
        """Calculate loss of all potential positive samples obtained from first
        match process.

        Args:
            anchors (list[Tensor]): Anchors of each scale.
            cls_score (Tensor): Box scores of single image with shape
                (num_anchors, num_classes)
            bbox_pred (Tensor): Box energies / deltas of single image
                with shape (num_anchors, 4)
            label (Tensor): classification target of each anchor with
                shape (num_anchors,)
            label_weight (Tensor): Classification loss weight of each
                anchor with shape (num_anchors).
            bbox_target (dict): Regression target of each anchor with
                shape (num_anchors, 4).
            bbox_weight (Tensor): Bbox weight of each anchor with shape
                (num_anchors, 4).
            pos_inds (Tensor): Index of all positive samples got from
                first assign process.

        Returns:
            Tensor: Losses of all positive samples in single image.
        """
        if not len(pos_inds):
            return (cls_score.new([]),)
        anchors_all_level = torch.cat(anchors, 0)
        pos_scores = cls_score[pos_inds]
        pos_bbox_pred = bbox_pred[pos_inds]
        pos_label = label[pos_inds]
        pos_label_weight = label_weight[pos_inds]
        pos_bbox_target = bbox_target[pos_inds]
        pos_bbox_weight = bbox_weight[pos_inds]
        pos_anchors = anchors_all_level[pos_inds]
        pos_bbox_pred = self.bbox_coder.decode(pos_anchors, pos_bbox_pred)
        loss_cls = self.loss_cls(pos_scores, pos_label, pos_label_weight, avg_factor=1.0, reduction_override='none')
        loss_bbox = self.loss_bbox(pos_bbox_pred, pos_bbox_target, pos_bbox_weight, avg_factor=1.0, reduction_override='none')
        loss_cls = loss_cls.sum(-1)
        pos_loss = loss_bbox + loss_cls
        return (pos_loss,)

    def paa_reassign(self, pos_losses, label, label_weight, bbox_weight, pos_inds, pos_gt_inds, anchors):
        """Fit loss to GMM distribution and separate positive, ignore, negative
        samples again with GMM model.

        Args:
            pos_losses (Tensor): Losses of all positive samples in
                single image.
            label (Tensor): classification target of each anchor with
                shape (num_anchors,)
            label_weight (Tensor): Classification loss weight of each
                anchor with shape (num_anchors).
            bbox_weight (Tensor): Bbox weight of each anchor with shape
                (num_anchors, 4).
            pos_inds (Tensor): Index of all positive samples got from
                first assign process.
            pos_gt_inds (Tensor): Gt_index of all positive samples got
                from first assign process.
            anchors (list[Tensor]): Anchors of each scale.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - label (Tensor): classification target of each anchor after
                  paa assign, with shape (num_anchors,)
                - label_weight (Tensor): Classification loss weight of each
                  anchor after paa assign, with shape (num_anchors).
                - bbox_weight (Tensor): Bbox weight of each anchor with shape
                  (num_anchors, 4).
                - num_pos (int): The number of positive samples after paa
                  assign.
        """
        if not len(pos_inds):
            return (label, label_weight, bbox_weight, 0)
        label = label.clone()
        label_weight = label_weight.clone()
        bbox_weight = bbox_weight.clone()
        num_gt = pos_gt_inds.max() + 1
        num_level = len(anchors)
        num_anchors_each_level = [item.size(0) for item in anchors]
        num_anchors_each_level.insert(0, 0)
        inds_level_interval = np.cumsum(num_anchors_each_level)
        pos_level_mask = []
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)
        pos_inds_after_paa = [label.new_tensor([])]
        ignore_inds_after_paa = [label.new_tensor([])]
        for gt_ind in range(num_gt):
            pos_inds_gmm = []
            pos_loss_gmm = []
            gt_mask = pos_gt_inds == gt_ind
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                level_gt_mask = level_mask & gt_mask
                value, topk_inds = pos_losses[level_gt_mask].topk(min(level_gt_mask.sum(), self.topk), largest=False)
                pos_inds_gmm.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_gmm.append(value)
            pos_inds_gmm = torch.cat(pos_inds_gmm)
            pos_loss_gmm = torch.cat(pos_loss_gmm)
            if len(pos_inds_gmm) < 2:
                continue
            device = pos_inds_gmm.device
            pos_loss_gmm, sort_inds = pos_loss_gmm.sort()
            pos_inds_gmm = pos_inds_gmm[sort_inds]
            pos_loss_gmm = pos_loss_gmm.view(-1, 1).cpu().numpy()
            min_loss, max_loss = (pos_loss_gmm.min(), pos_loss_gmm.max())
            means_init = np.array([min_loss, max_loss]).reshape(2, 1)
            weights_init = np.array([0.5, 0.5])
            precisions_init = np.array([1.0, 1.0]).reshape(2, 1, 1)
            if self.covariance_type == 'spherical':
                precisions_init = precisions_init.reshape(2)
            elif self.covariance_type == 'diag':
                precisions_init = precisions_init.reshape(2, 1)
            elif self.covariance_type == 'tied':
                precisions_init = np.array([[1.0]])
            if skm is None:
                raise ImportError('Please run "pip install sklearn" to install sklearn first.')
            gmm = skm.GaussianMixture(2, weights_init=weights_init, means_init=means_init, precisions_init=precisions_init, covariance_type=self.covariance_type)
            gmm.fit(pos_loss_gmm)
            gmm_assignment = gmm.predict(pos_loss_gmm)
            scores = gmm.score_samples(pos_loss_gmm)
            gmm_assignment = torch.from_numpy(gmm_assignment).to(device)
            scores = torch.from_numpy(scores).to(device)
            pos_inds_temp, ignore_inds_temp = self.gmm_separation_scheme(gmm_assignment, scores, pos_inds_gmm)
            pos_inds_after_paa.append(pos_inds_temp)
            ignore_inds_after_paa.append(ignore_inds_temp)
        pos_inds_after_paa = torch.cat(pos_inds_after_paa)
        ignore_inds_after_paa = torch.cat(ignore_inds_after_paa)
        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_paa).all(1)
        reassign_ids = pos_inds[reassign_mask]
        label[reassign_ids] = self.num_classes
        label_weight[ignore_inds_after_paa] = 0
        bbox_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_paa)
        return (label, label_weight, bbox_weight, num_pos)

    def gmm_separation_scheme(self, gmm_assignment, scores, pos_inds_gmm):
        """A general separation scheme for gmm model.

        It separates a GMM distribution of candidate samples into three
        parts, 0 1 and uncertain areas, and you can implement other
        separation schemes by rewriting this function.

        Args:
            gmm_assignment (Tensor): The prediction of GMM which is of shape
                (num_samples,). The 0/1 value indicates the distribution
                that each sample comes from.
            scores (Tensor): The probability of sample coming from the
                fit GMM distribution. The tensor is of shape (num_samples,).
            pos_inds_gmm (Tensor): All the indexes of samples which are used
                to fit GMM model. The tensor is of shape (num_samples,)

        Returns:
            tuple[Tensor]: The indices of positive and ignored samples.

                - pos_inds_temp (Tensor): Indices of positive samples.
                - ignore_inds_temp (Tensor): Indices of ignore samples.
        """
        fgs = gmm_assignment == 0
        pos_inds_temp = fgs.new_tensor([], dtype=torch.long)
        ignore_inds_temp = fgs.new_tensor([], dtype=torch.long)
        if fgs.nonzero().numel():
            _, pos_thr_ind = scores[fgs].topk(1)
            pos_inds_temp = pos_inds_gmm[fgs][:pos_thr_ind + 1]
            ignore_inds_temp = pos_inds_gmm.new_tensor([])
        return (pos_inds_temp, ignore_inds_temp)

    def get_targets(self, anchor_list, valid_flag_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, unmap_outputs=True):
        """Get targets for PAA head.

        This method is almost the same as `AnchorHead.get_targets()`. We direct
        return the results from _get_targets_single instead map it to levels
        by images_to_levels function.

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

                - labels (list[Tensor]): Labels of all anchors, each with
                    shape (num_anchors,).
                - label_weights (list[Tensor]): Label weights of all anchor.
                    each with shape (num_anchors,).
                - bbox_targets (list[Tensor]): BBox targets of all anchors.
                    each with shape (num_anchors, 4).
                - bbox_weights (list[Tensor]): BBox weights of all anchors.
                    each with shape (num_anchors, 4).
                - pos_inds (list[Tensor]): Contains all index of positive
                    sample in all anchor.
                - gt_inds (list[Tensor]): Contains all gt_index of positive
                    sample in all anchor.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
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
        labels, label_weights, bbox_targets, bbox_weights, valid_pos_inds, valid_neg_inds, sampling_result = results
        pos_inds = []
        for i, single_labels in enumerate(labels):
            pos_mask = (0 <= single_labels) & (single_labels < self.num_classes)
            pos_inds.append(pos_mask.nonzero().view(-1))
        gt_inds = [item.pos_assigned_gt_inds for item in sampling_result]
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, gt_inds)

    def _get_targets_single(self, flat_anchors, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, label_channels=1, unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        This method is same as `AnchorHead._get_targets_single()`.
        """
        assert unmap_outputs, 'We must map outputs back to the originalset of anchors in PAAhead'
        return super(ATSSHead, self)._get_targets_single(flat_anchors, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, label_channels=1, unmap_outputs=True)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, score_factors=None, img_metas=None, cfg=None, rescale=False, with_nms=True, **kwargs):
        assert with_nms, 'PAA only supports "with_nms=True" now and it means PAAHead does not support test-time augmentation'
        return super(ATSSHead, self).get_bboxes(cls_scores, bbox_preds, score_factors, img_metas, cfg, rescale, with_nms, **kwargs)

    def _get_bboxes_single(self, cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors, img_meta, cfg, rescale=False, with_nms=True, **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factors from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
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
        mlvl_score_factors = []
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in enumerate(zip(cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            if 0 < nms_pre < scores.shape[0]:
                max_scores, _ = (scores * score_factor[:, None]).sqrt().max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                priors = priors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                score_factor = score_factor[topk_inds]
            bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_score_factors.append(score_factor)
        return self._bbox_post_process(mlvl_scores, mlvl_bboxes, img_meta['scale_factor'], cfg, rescale, with_nms, mlvl_score_factors, **kwargs)

    def _bbox_post_process(self, mlvl_scores, mlvl_bboxes, scale_factor, cfg, rescale=False, with_nms=True, mlvl_score_factors=None, **kwargs):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually with_nms is False is used for aug test.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, num_class).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
            mlvl_score_factors (list[Tensor], optional): Score factor from
                all scale levels of a single image, each item has shape
                (num_bboxes, ). Default: None.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape                     [num_bboxes, 5], where the first 4 columns are bounding                     box positions (tl_x, tl_y, br_x, br_y) and the 5-th                     column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding                     box with shape [num_bboxes].
        """
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_iou_preds = torch.cat(mlvl_score_factors)
        mlvl_nms_scores = (mlvl_scores * mlvl_iou_preds[:, None]).sqrt()
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_nms_scores, cfg.score_thr, cfg.nms, cfg.max_per_img, score_factors=None)
        if self.with_score_voting and len(det_bboxes) > 0:
            det_bboxes, det_labels = self.score_voting(det_bboxes, det_labels, mlvl_bboxes, mlvl_nms_scores, cfg.score_thr)
        return (det_bboxes, det_labels)

    def score_voting(self, det_bboxes, det_labels, mlvl_bboxes, mlvl_nms_scores, score_thr):
        """Implementation of score voting method works on each remaining boxes
        after NMS procedure.

        Args:
            det_bboxes (Tensor): Remaining boxes after NMS procedure,
                with shape (k, 5), each dimension means
                (x1, y1, x2, y2, score).
            det_labels (Tensor): The label of remaining boxes, with shape
                (k, 1),Labels are 0-based.
            mlvl_bboxes (Tensor): All boxes before the NMS procedure,
                with shape (num_anchors,4).
            mlvl_nms_scores (Tensor): The scores of all boxes which is used
                in the NMS procedure, with shape (num_anchors, num_class)
            score_thr (float): The score threshold of bboxes.

        Returns:
            tuple: Usually returns a tuple containing voting results.

                - det_bboxes_voted (Tensor): Remaining boxes after
                    score voting procedure, with shape (k, 5), each
                    dimension means (x1, y1, x2, y2, score).
                - det_labels_voted (Tensor): Label of remaining bboxes
                    after voting, with shape (num_anchors,).
        """
        candidate_mask = mlvl_nms_scores > score_thr
        candidate_mask_nonzeros = candidate_mask.nonzero(as_tuple=False)
        candidate_inds = candidate_mask_nonzeros[:, 0]
        candidate_labels = candidate_mask_nonzeros[:, 1]
        candidate_bboxes = mlvl_bboxes[candidate_inds]
        candidate_scores = mlvl_nms_scores[candidate_mask]
        det_bboxes_voted = []
        det_labels_voted = []
        for cls in range(self.cls_out_channels):
            candidate_cls_mask = candidate_labels == cls
            if not candidate_cls_mask.any():
                continue
            candidate_cls_scores = candidate_scores[candidate_cls_mask]
            candidate_cls_bboxes = candidate_bboxes[candidate_cls_mask]
            det_cls_mask = det_labels == cls
            det_cls_bboxes = det_bboxes[det_cls_mask].view(-1, det_bboxes.size(-1))
            det_candidate_ious = bbox_overlaps(det_cls_bboxes[:, :4], candidate_cls_bboxes)
            for det_ind in range(len(det_cls_bboxes)):
                single_det_ious = det_candidate_ious[det_ind]
                pos_ious_mask = single_det_ious > 0.01
                pos_ious = single_det_ious[pos_ious_mask]
                pos_bboxes = candidate_cls_bboxes[pos_ious_mask]
                pos_scores = candidate_cls_scores[pos_ious_mask]
                pis = (torch.exp(-(1 - pos_ious) ** 2 / 0.025) * pos_scores)[:, None]
                voted_box = torch.sum(pis * pos_bboxes, dim=0) / torch.sum(pis, dim=0)
                voted_score = det_cls_bboxes[det_ind][-1:][None, :]
                det_bboxes_voted.append(torch.cat((voted_box[None, :], voted_score), dim=1))
                det_labels_voted.append(cls)
        det_bboxes_voted = torch.cat(det_bboxes_voted, dim=0)
        det_labels_voted = det_labels.new_tensor(det_labels_voted)
        return (det_bboxes_voted, det_labels_voted)

@HEADS.register_module()
class DETRHead(AnchorFreeHead):
    """Implements the DETR transformer head.

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2

    def __init__(self, num_classes, in_channels, num_query=100, num_reg_fcs=2, transformer=None, sync_cls_avg_factor=False, positional_encoding=dict(type='SinePositionalEncoding', num_feats=128, normalize=True), loss_cls=dict(type='CrossEntropyLoss', bg_cls_weight=0.1, use_sigmoid=False, loss_weight=1.0, class_weight=1.0), loss_bbox=dict(type='L1Loss', loss_weight=5.0), loss_iou=dict(type='GIoULoss', loss_weight=2.0), train_cfg=dict(assigner=dict(type='HungarianAssigner', cls_cost=dict(type='ClassificationCost', weight=1.0), reg_cost=dict(type='BBoxL1Cost', weight=5.0), iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))), test_cfg=dict(max_per_img=100), init_cfg=None, **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and self.__class__ is DETRHead:
            assert isinstance(class_weight, float), f'Expected class_weight to have type float. Found {type(class_weight)}.'
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), f'Expected bg_cls_weight to have type float. Found {type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], 'The classification weight for loss and matcher should beexactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost']['weight'], 'The regression L1 weight for loss and matcher should be exactly the same.'
            assert loss_iou['loss_weight'] == assigner['iou_cost']['weight'], 'The regression iou weight for loss and matcher should beexactly the same.'
            self.assigner = build_assigner(assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.act_cfg = transformer.get('act_cfg', dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, f'embed_dims should be exactly 2 times of num_feats. Found {self.embed_dims} and {num_feats}.'
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        self.input_proj = Conv2d(self.in_channels, self.embed_dims, kernel_size=1)
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.reg_ffn = FFN(self.embed_dims, self.embed_dims, self.num_reg_fcs, self.act_cfg, dropout=0.0, add_residual=False)
        self.fc_reg = Linear(self.embed_dims, 4)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        self.transformer.init_weights()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is DETRHead:
            convert_dict = {'.self_attn.': '.attentions.0.', '.ffn.': '.ffns.0.', '.multihead_attn.': '.attentions.1.', '.decoder.norm.': '.decoder.post_norm.'}
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]
        super(AnchorFreeHead, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores_list (list[Tensor]): Classification scores                     for each scale level. Each is a 4D-tensor with shape                     [nb_dec, bs, num_query, cls_out_channels]. Note                     `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression                     outputs for each scale level. Each is a 4D-tensor with                     normalized coordinate format (cx, cy, w, h) and shape                     [nb_dec, bs, num_query, 4].
        """
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single, feats, img_metas_list)

    def forward_single(self, x, img_metas):
        """"Forward function for a single feature level.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0
        x = self.input_proj(x)
        masks = F.interpolate(masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        pos_embed = self.positional_encoding(masks)
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight, pos_embed)
        all_cls_scores = self.fc_cls(outs_dec)
        all_bbox_preds = self.fc_reg(self.activate(self.reg_ffn(outs_dec))).sigmoid()
        return (all_cls_scores, all_bbox_preds)

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self, all_cls_scores_list, all_bbox_preds_list, gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore=None):
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        all_cls_scores = all_cls_scores_list[-1]
        all_bbox_preds = all_bbox_preds_list[-1]
        assert gt_bboxes_ignore is None, 'Only supports for gt_bboxes_ignore setting to None.'
        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_bbox, losses_iou = multi_apply(self.loss_single, all_cls_scores, all_bbox_preds, all_gt_bboxes_list, all_gt_labels_list, img_metas_list, all_gt_bboxes_ignore_list)
        loss_dict = dict()
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self, cls_scores, bbox_preds, gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors
        loss_iou = self.loss_iou(bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)
        loss_bbox = self.loss_bbox(bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return (loss_cls, loss_bbox, loss_iou)

    def get_targets(self, cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all                     images.
                - bbox_targets_list (list[Tensor]): BBox targets for all                     images.
                - bbox_weights_list (list[Tensor]): BBox weights for all                     images.
                - num_total_pos (int): Number of positive samples in all                     images.
                - num_total_neg (int): Number of negative samples in all                     images.
        """
        assert gt_bboxes_ignore_list is None, 'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, pos_inds_list, neg_inds_list = multi_apply(self._get_target_single, cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, bbox_pred, gt_bboxes, gt_labels, img_meta, gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes, gt_labels, img_meta, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=None, proposal_cfg=None, **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self, all_cls_scores_list, all_bbox_preds_list, img_metas, rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.                 The first item is an (n, 5) tensor, where the first 4 columns                 are bounding box positions (tl_x, tl_y, br_x, br_y) and the                 5-th column is a score between 0 and 1. The second item is a                 (n,) tensor where each item is the predicted class label of                 the corresponding box.
        """
        cls_scores = all_cls_scores_list[-1][-1]
        bbox_preds = all_bbox_preds_list[-1][-1]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred, img_shape, scale_factor, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self, cls_score, bbox_pred, img_shape, scale_factor, rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5],                     where the first 4 columns are bounding box positions                     (tl_x, tl_y, br_x, br_y) and the 5-th column are scores                     between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with                     shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]
        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)
        return (det_bboxes, det_labels)

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list

    def forward_onnx(self, feats, img_metas):
        """Forward function for exporting to ONNX.

        Over-write `forward` because: `masks` is directly created with
        zero (valid position tag) and has the same spatial size as `x`.
        Thus the construction of `masks` is different from that in `forward`.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores_list (list[Tensor]): Classification scores                     for each scale level. Each is a 4D-tensor with shape                     [nb_dec, bs, num_query, cls_out_channels]. Note                     `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression                     outputs for each scale level. Each is a 4D-tensor with                     normalized coordinate format (cx, cy, w, h) and shape                     [nb_dec, bs, num_query, 4].
        """
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single_onnx, feats, img_metas_list)

    def forward_single_onnx(self, x, img_metas):
        """"Forward function for a single feature level with ONNX exportation.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        batch_size = x.size(0)
        h, w = x.size()[-2:]
        masks = x.new_zeros((batch_size, h, w))
        x = self.input_proj(x)
        masks = F.interpolate(masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        pos_embed = self.positional_encoding(masks)
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight, pos_embed)
        all_cls_scores = self.fc_cls(outs_dec)
        all_bbox_preds = self.fc_reg(self.activate(self.reg_ffn(outs_dec))).sigmoid()
        return (all_cls_scores, all_bbox_preds)

    def onnx_export(self, all_cls_scores_list, all_bbox_preds_list, img_metas):
        """Transform network outputs into bbox predictions, with ONNX
        exportation.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        assert len(img_metas) == 1, 'Only support one input image while in exporting to ONNX'
        cls_scores = all_cls_scores_list[-1][-1]
        bbox_preds = all_bbox_preds_list[-1][-1]
        img_shape = img_metas[0]['img_shape_for_onnx']
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        batch_size = cls_scores.size(0)
        batch_index_offset = torch.arange(batch_size).to(cls_scores.device) * max_per_img
        batch_index_offset = batch_index_offset.unsqueeze(1).expand(batch_size, max_per_img)
        if self.loss_cls.use_sigmoid:
            cls_scores = cls_scores.sigmoid()
            scores, indexes = cls_scores.view(batch_size, -1).topk(max_per_img, dim=1)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_index = (bbox_index + batch_index_offset).view(-1)
            bbox_preds = bbox_preds.view(-1, 4)[bbox_index]
            bbox_preds = bbox_preds.view(batch_size, -1, 4)
        else:
            scores, det_labels = F.softmax(cls_scores, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img, dim=1)
            bbox_index = (bbox_index + batch_index_offset).view(-1)
            bbox_preds = bbox_preds.view(-1, 4)[bbox_index]
            det_labels = det_labels.view(-1)[bbox_index]
            bbox_preds = bbox_preds.view(batch_size, -1, 4)
            det_labels = det_labels.view(batch_size, -1)
        det_bboxes = bbox_cxcywh_to_xyxy(bbox_preds)
        img_shape_tensor = img_shape.flip(0).repeat(2)
        img_shape_tensor = img_shape_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, det_bboxes.size(1), 4)
        det_bboxes = det_bboxes * img_shape_tensor
        x1, y1, x2, y2 = det_bboxes.split((1, 1, 1, 1), dim=-1)
        from mmdet.core.export import dynamic_clip_for_onnx
        x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, img_shape)
        det_bboxes = torch.cat([x1, y1, x2, y2], dim=-1)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(-1)), -1)
        return (det_bboxes, det_labels)

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
class CornerHead(BaseDenseHead, BBoxTestMixin):
    """Head of CornerNet: Detecting Objects as Paired Keypoints.

    Code is modified from the `official github repo
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/
    kp.py#L73>`_ .

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_ .

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        num_feat_levels (int): Levels of feature from the previous module. 2
            for HourglassNet-104 and 1 for HourglassNet-52. Because
            HourglassNet-104 outputs the final feature and intermediate
            supervision feature and HourglassNet-52 only outputs the final
            feature. Default: 2.
        corner_emb_channels (int): Channel of embedding vector. Default: 1.
        train_cfg (dict | None): Training config. Useless in CornerHead,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CornerHead. Default: None.
        loss_heatmap (dict | None): Config of corner heatmap loss. Default:
            GaussianFocalLoss.
        loss_embedding (dict | None): Config of corner embedding loss. Default:
            AssociativeEmbeddingLoss.
        loss_offset (dict | None): Config of corner offset loss. Default:
            SmoothL1Loss.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self, num_classes, in_channels, num_feat_levels=2, corner_emb_channels=1, train_cfg=None, test_cfg=None, loss_heatmap=dict(type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1), loss_embedding=dict(type='AssociativeEmbeddingLoss', pull_weight=0.25, push_weight=0.25), loss_offset=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1), init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization behavior, init_cfg is not allowed to be set'
        super(CornerHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.corner_emb_channels = corner_emb_channels
        self.with_corner_emb = self.corner_emb_channels > 0
        self.corner_offset_channels = 2
        self.num_feat_levels = num_feat_levels
        self.loss_heatmap = build_loss(loss_heatmap) if loss_heatmap is not None else None
        self.loss_embedding = build_loss(loss_embedding) if loss_embedding is not None else None
        self.loss_offset = build_loss(loss_offset) if loss_offset is not None else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self._init_layers()

    def _make_layers(self, out_channels, in_channels=256, feat_channels=256):
        """Initialize conv sequential for CornerHead."""
        return nn.Sequential(ConvModule(in_channels, feat_channels, 3, padding=1), ConvModule(feat_channels, out_channels, 1, norm_cfg=None, act_cfg=None))

    def _init_corner_kpt_layers(self):
        """Initialize corner keypoint layers.

        Including corner heatmap branch and corner offset branch. Each branch
        has two parts: prefix `tl_` for top-left and `br_` for bottom-right.
        """
        self.tl_pool, self.br_pool = (nn.ModuleList(), nn.ModuleList())
        self.tl_heat, self.br_heat = (nn.ModuleList(), nn.ModuleList())
        self.tl_off, self.br_off = (nn.ModuleList(), nn.ModuleList())
        for _ in range(self.num_feat_levels):
            self.tl_pool.append(BiCornerPool(self.in_channels, ['top', 'left'], out_channels=self.in_channels))
            self.br_pool.append(BiCornerPool(self.in_channels, ['bottom', 'right'], out_channels=self.in_channels))
            self.tl_heat.append(self._make_layers(out_channels=self.num_classes, in_channels=self.in_channels))
            self.br_heat.append(self._make_layers(out_channels=self.num_classes, in_channels=self.in_channels))
            self.tl_off.append(self._make_layers(out_channels=self.corner_offset_channels, in_channels=self.in_channels))
            self.br_off.append(self._make_layers(out_channels=self.corner_offset_channels, in_channels=self.in_channels))

    def _init_corner_emb_layers(self):
        """Initialize corner embedding layers.

        Only include corner embedding branch with two parts: prefix `tl_` for
        top-left and `br_` for bottom-right.
        """
        self.tl_emb, self.br_emb = (nn.ModuleList(), nn.ModuleList())
        for _ in range(self.num_feat_levels):
            self.tl_emb.append(self._make_layers(out_channels=self.corner_emb_channels, in_channels=self.in_channels))
            self.br_emb.append(self._make_layers(out_channels=self.corner_emb_channels, in_channels=self.in_channels))

    def _init_layers(self):
        """Initialize layers for CornerHead.

        Including two parts: corner keypoint layers and corner embedding layers
        """
        self._init_corner_kpt_layers()
        if self.with_corner_emb:
            self._init_corner_emb_layers()

    def init_weights(self):
        super(CornerHead, self).init_weights()
        bias_init = bias_init_with_prob(0.1)
        for i in range(self.num_feat_levels):
            self.tl_heat[i][-1].conv.reset_parameters()
            self.tl_heat[i][-1].conv.bias.data.fill_(bias_init)
            self.br_heat[i][-1].conv.reset_parameters()
            self.br_heat[i][-1].conv.bias.data.fill_(bias_init)
            self.tl_off[i][-1].conv.reset_parameters()
            self.br_off[i][-1].conv.reset_parameters()
            if self.with_corner_emb:
                self.tl_emb[i][-1].conv.reset_parameters()
                self.br_emb[i][-1].conv.reset_parameters()

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of corner heatmaps, offset heatmaps and
            embedding heatmaps.
                - tl_heats (list[Tensor]): Top-left corner heatmaps for all
                  levels, each is a 4D-tensor, the channels number is
                  num_classes.
                - br_heats (list[Tensor]): Bottom-right corner heatmaps for all
                  levels, each is a 4D-tensor, the channels number is
                  num_classes.
                - tl_embs (list[Tensor] | list[None]): Top-left embedding
                  heatmaps for all levels, each is a 4D-tensor or None.
                  If not None, the channels number is corner_emb_channels.
                - br_embs (list[Tensor] | list[None]): Bottom-right embedding
                  heatmaps for all levels, each is a 4D-tensor or None.
                  If not None, the channels number is corner_emb_channels.
                - tl_offs (list[Tensor]): Top-left offset heatmaps for all
                  levels, each is a 4D-tensor. The channels number is
                  corner_offset_channels.
                - br_offs (list[Tensor]): Bottom-right offset heatmaps for all
                  levels, each is a 4D-tensor. The channels number is
                  corner_offset_channels.
        """
        lvl_ind = list(range(self.num_feat_levels))
        return multi_apply(self.forward_single, feats, lvl_ind)

    def forward_single(self, x, lvl_ind, return_pool=False):
        """Forward feature of a single level.

        Args:
            x (Tensor): Feature of a single level.
            lvl_ind (int): Level index of current feature.
            return_pool (bool): Return corner pool feature or not.

        Returns:
            tuple[Tensor]: A tuple of CornerHead's output for current feature
            level. Containing the following Tensors:

                - tl_heat (Tensor): Predicted top-left corner heatmap.
                - br_heat (Tensor): Predicted bottom-right corner heatmap.
                - tl_emb (Tensor | None): Predicted top-left embedding heatmap.
                  None for `self.with_corner_emb == False`.
                - br_emb (Tensor | None): Predicted bottom-right embedding
                  heatmap. None for `self.with_corner_emb == False`.
                - tl_off (Tensor): Predicted top-left offset heatmap.
                - br_off (Tensor): Predicted bottom-right offset heatmap.
                - tl_pool (Tensor): Top-left corner pool feature. Not must
                  have.
                - br_pool (Tensor): Bottom-right corner pool feature. Not must
                  have.
        """
        tl_pool = self.tl_pool[lvl_ind](x)
        tl_heat = self.tl_heat[lvl_ind](tl_pool)
        br_pool = self.br_pool[lvl_ind](x)
        br_heat = self.br_heat[lvl_ind](br_pool)
        tl_emb, br_emb = (None, None)
        if self.with_corner_emb:
            tl_emb = self.tl_emb[lvl_ind](tl_pool)
            br_emb = self.br_emb[lvl_ind](br_pool)
        tl_off = self.tl_off[lvl_ind](tl_pool)
        br_off = self.br_off[lvl_ind](br_pool)
        result_list = [tl_heat, br_heat, tl_emb, br_emb, tl_off, br_off]
        if return_pool:
            result_list.append(tl_pool)
            result_list.append(br_pool)
        return result_list

    def get_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape, with_corner_emb=False, with_guiding_shift=False, with_centripetal_shift=False):
        """Generate corner targets.

        Including corner heatmap, corner offset.

        Optional: corner embedding, corner guiding shift, centripetal shift.

        For CornerNet, we generate corner heatmap, corner offset and corner
        embedding from this function.

        For CentripetalNet, we generate corner heatmap, corner offset, guiding
        shift and centripetal shift from this function.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image, each
                has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box, each has
                shape (num_gt,).
            feat_shape (list[int]): Shape of output feature,
                [batch, channel, height, width].
            img_shape (list[int]): Shape of input image,
                [height, width, channel].
            with_corner_emb (bool): Generate corner embedding target or not.
                Default: False.
            with_guiding_shift (bool): Generate guiding shift target or not.
                Default: False.
            with_centripetal_shift (bool): Generate centripetal shift target or
                not. Default: False.

        Returns:
            dict: Ground truth of corner heatmap, corner offset, corner
            embedding, guiding shift and centripetal shift. Containing the
            following keys:

                - topleft_heatmap (Tensor): Ground truth top-left corner
                  heatmap.
                - bottomright_heatmap (Tensor): Ground truth bottom-right
                  corner heatmap.
                - topleft_offset (Tensor): Ground truth top-left corner offset.
                - bottomright_offset (Tensor): Ground truth bottom-right corner
                  offset.
                - corner_embedding (list[list[list[int]]]): Ground truth corner
                  embedding. Not must have.
                - topleft_guiding_shift (Tensor): Ground truth top-left corner
                  guiding shift. Not must have.
                - bottomright_guiding_shift (Tensor): Ground truth bottom-right
                  corner guiding shift. Not must have.
                - topleft_centripetal_shift (Tensor): Ground truth top-left
                  corner centripetal shift. Not must have.
                - bottomright_centripetal_shift (Tensor): Ground truth
                  bottom-right corner centripetal shift. Not must have.
        """
        batch_size, _, height, width = feat_shape
        img_h, img_w = img_shape[:2]
        width_ratio = float(width / img_w)
        height_ratio = float(height / img_h)
        gt_tl_heatmap = gt_bboxes[-1].new_zeros([batch_size, self.num_classes, height, width])
        gt_br_heatmap = gt_bboxes[-1].new_zeros([batch_size, self.num_classes, height, width])
        gt_tl_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
        gt_br_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
        if with_corner_emb:
            match = []
        if with_guiding_shift:
            gt_tl_guiding_shift = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
            gt_br_guiding_shift = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
        if with_centripetal_shift:
            gt_tl_centripetal_shift = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
            gt_br_centripetal_shift = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
        for batch_id in range(batch_size):
            corner_match = []
            for box_id in range(len(gt_labels[batch_id])):
                left, top, right, bottom = gt_bboxes[batch_id][box_id]
                center_x = (left + right) / 2.0
                center_y = (top + bottom) / 2.0
                label = gt_labels[batch_id][box_id]
                scale_left = left * width_ratio
                scale_right = right * width_ratio
                scale_top = top * height_ratio
                scale_bottom = bottom * height_ratio
                scale_center_x = center_x * width_ratio
                scale_center_y = center_y * height_ratio
                left_idx = int(min(scale_left, width - 1))
                right_idx = int(min(scale_right, width - 1))
                top_idx = int(min(scale_top, height - 1))
                bottom_idx = int(min(scale_bottom, height - 1))
                scale_box_width = ceil(scale_right - scale_left)
                scale_box_height = ceil(scale_bottom - scale_top)
                radius = gaussian_radius((scale_box_height, scale_box_width), min_overlap=0.3)
                radius = max(0, int(radius))
                gt_tl_heatmap[batch_id, label] = gen_gaussian_target(gt_tl_heatmap[batch_id, label], [left_idx, top_idx], radius)
                gt_br_heatmap[batch_id, label] = gen_gaussian_target(gt_br_heatmap[batch_id, label], [right_idx, bottom_idx], radius)
                left_offset = scale_left - left_idx
                top_offset = scale_top - top_idx
                right_offset = scale_right - right_idx
                bottom_offset = scale_bottom - bottom_idx
                gt_tl_offset[batch_id, 0, top_idx, left_idx] = left_offset
                gt_tl_offset[batch_id, 1, top_idx, left_idx] = top_offset
                gt_br_offset[batch_id, 0, bottom_idx, right_idx] = right_offset
                gt_br_offset[batch_id, 1, bottom_idx, right_idx] = bottom_offset
                if with_corner_emb:
                    corner_match.append([[top_idx, left_idx], [bottom_idx, right_idx]])
                if with_guiding_shift:
                    gt_tl_guiding_shift[batch_id, 0, top_idx, left_idx] = scale_center_x - left_idx
                    gt_tl_guiding_shift[batch_id, 1, top_idx, left_idx] = scale_center_y - top_idx
                    gt_br_guiding_shift[batch_id, 0, bottom_idx, right_idx] = right_idx - scale_center_x
                    gt_br_guiding_shift[batch_id, 1, bottom_idx, right_idx] = bottom_idx - scale_center_y
                if with_centripetal_shift:
                    gt_tl_centripetal_shift[batch_id, 0, top_idx, left_idx] = log(scale_center_x - scale_left)
                    gt_tl_centripetal_shift[batch_id, 1, top_idx, left_idx] = log(scale_center_y - scale_top)
                    gt_br_centripetal_shift[batch_id, 0, bottom_idx, right_idx] = log(scale_right - scale_center_x)
                    gt_br_centripetal_shift[batch_id, 1, bottom_idx, right_idx] = log(scale_bottom - scale_center_y)
            if with_corner_emb:
                match.append(corner_match)
        target_result = dict(topleft_heatmap=gt_tl_heatmap, topleft_offset=gt_tl_offset, bottomright_heatmap=gt_br_heatmap, bottomright_offset=gt_br_offset)
        if with_corner_emb:
            target_result.update(corner_embedding=match)
        if with_guiding_shift:
            target_result.update(topleft_guiding_shift=gt_tl_guiding_shift, bottomright_guiding_shift=gt_br_guiding_shift)
        if with_centripetal_shift:
            target_result.update(topleft_centripetal_shift=gt_tl_centripetal_shift, bottomright_centripetal_shift=gt_br_centripetal_shift)
        return target_result

    @force_fp32()
    def loss(self, tl_heats, br_heats, tl_embs, br_embs, tl_offs, br_offs, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            tl_heats (list[Tensor]): Top-left corner heatmaps for each level
                with shape (N, num_classes, H, W).
            br_heats (list[Tensor]): Bottom-right corner heatmaps for each
                level with shape (N, num_classes, H, W).
            tl_embs (list[Tensor]): Top-left corner embeddings for each level
                with shape (N, corner_emb_channels, H, W).
            br_embs (list[Tensor]): Bottom-right corner embeddings for each
                level with shape (N, corner_emb_channels, H, W).
            tl_offs (list[Tensor]): Top-left corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            br_offs (list[Tensor]): Bottom-right corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [left, top, right, bottom] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components. Containing the
            following losses:

                - det_loss (list[Tensor]): Corner keypoint losses of all
                  feature levels.
                - pull_loss (list[Tensor]): Part one of AssociativeEmbedding
                  losses of all feature levels.
                - push_loss (list[Tensor]): Part two of AssociativeEmbedding
                  losses of all feature levels.
                - off_loss (list[Tensor]): Corner offset losses of all feature
                  levels.
        """
        targets = self.get_targets(gt_bboxes, gt_labels, tl_heats[-1].shape, img_metas[0]['pad_shape'], with_corner_emb=self.with_corner_emb)
        mlvl_targets = [targets for _ in range(self.num_feat_levels)]
        det_losses, pull_losses, push_losses, off_losses = multi_apply(self.loss_single, tl_heats, br_heats, tl_embs, br_embs, tl_offs, br_offs, mlvl_targets)
        loss_dict = dict(det_loss=det_losses, off_loss=off_losses)
        if self.with_corner_emb:
            loss_dict.update(pull_loss=pull_losses, push_loss=push_losses)
        return loss_dict

    def loss_single(self, tl_hmp, br_hmp, tl_emb, br_emb, tl_off, br_off, targets):
        """Compute losses for single level.

        Args:
            tl_hmp (Tensor): Top-left corner heatmap for current level with
                shape (N, num_classes, H, W).
            br_hmp (Tensor): Bottom-right corner heatmap for current level with
                shape (N, num_classes, H, W).
            tl_emb (Tensor): Top-left corner embedding for current level with
                shape (N, corner_emb_channels, H, W).
            br_emb (Tensor): Bottom-right corner embedding for current level
                with shape (N, corner_emb_channels, H, W).
            tl_off (Tensor): Top-left corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            br_off (Tensor): Bottom-right corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            targets (dict): Corner target generated by `get_targets`.

        Returns:
            tuple[torch.Tensor]: Losses of the head's different branches
            containing the following losses:

                - det_loss (Tensor): Corner keypoint loss.
                - pull_loss (Tensor): Part one of AssociativeEmbedding loss.
                - push_loss (Tensor): Part two of AssociativeEmbedding loss.
                - off_loss (Tensor): Corner offset loss.
        """
        gt_tl_hmp = targets['topleft_heatmap']
        gt_br_hmp = targets['bottomright_heatmap']
        gt_tl_off = targets['topleft_offset']
        gt_br_off = targets['bottomright_offset']
        gt_embedding = targets['corner_embedding']
        tl_det_loss = self.loss_heatmap(tl_hmp.sigmoid(), gt_tl_hmp, avg_factor=max(1, gt_tl_hmp.eq(1).sum()))
        br_det_loss = self.loss_heatmap(br_hmp.sigmoid(), gt_br_hmp, avg_factor=max(1, gt_br_hmp.eq(1).sum()))
        det_loss = (tl_det_loss + br_det_loss) / 2.0
        if self.with_corner_emb and self.loss_embedding is not None:
            pull_loss, push_loss = self.loss_embedding(tl_emb, br_emb, gt_embedding)
        else:
            pull_loss, push_loss = (None, None)
        tl_off_mask = gt_tl_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(gt_tl_hmp)
        br_off_mask = gt_br_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(gt_br_hmp)
        tl_off_loss = self.loss_offset(tl_off, gt_tl_off, tl_off_mask, avg_factor=max(1, tl_off_mask.sum()))
        br_off_loss = self.loss_offset(br_off, gt_br_off, br_off_mask, avg_factor=max(1, br_off_mask.sum()))
        off_loss = (tl_off_loss + br_off_loss) / 2.0
        return (det_loss, pull_loss, push_loss, off_loss)

    @force_fp32()
    def get_bboxes(self, tl_heats, br_heats, tl_embs, br_embs, tl_offs, br_offs, img_metas, rescale=False, with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            tl_heats (list[Tensor]): Top-left corner heatmaps for each level
                with shape (N, num_classes, H, W).
            br_heats (list[Tensor]): Bottom-right corner heatmaps for each
                level with shape (N, num_classes, H, W).
            tl_embs (list[Tensor]): Top-left corner embeddings for each level
                with shape (N, corner_emb_channels, H, W).
            br_embs (list[Tensor]): Bottom-right corner embeddings for each
                level with shape (N, corner_emb_channels, H, W).
            tl_offs (list[Tensor]): Top-left corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            br_offs (list[Tensor]): Bottom-right corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        """
        assert tl_heats[-1].shape[0] == br_heats[-1].shape[0] == len(img_metas)
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(self._get_bboxes_single(tl_heats[-1][img_id:img_id + 1, :], br_heats[-1][img_id:img_id + 1, :], tl_offs[-1][img_id:img_id + 1, :], br_offs[-1][img_id:img_id + 1, :], img_metas[img_id], tl_emb=tl_embs[-1][img_id:img_id + 1, :], br_emb=br_embs[-1][img_id:img_id + 1, :], rescale=rescale, with_nms=with_nms))
        return result_list

    def _get_bboxes_single(self, tl_heat, br_heat, tl_off, br_off, img_meta, tl_emb=None, br_emb=None, tl_centripetal_shift=None, br_centripetal_shift=None, rescale=False, with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            tl_heat (Tensor): Top-left corner heatmap for current level with
                shape (N, num_classes, H, W).
            br_heat (Tensor): Bottom-right corner heatmap for current level
                with shape (N, num_classes, H, W).
            tl_off (Tensor): Top-left corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            br_off (Tensor): Bottom-right corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            tl_emb (Tensor): Top-left corner embedding for current level with
                shape (N, corner_emb_channels, H, W).
            br_emb (Tensor): Bottom-right corner embedding for current level
                with shape (N, corner_emb_channels, H, W).
            tl_centripetal_shift: Top-left corner's centripetal shift for
                current level with shape (N, 2, H, W).
            br_centripetal_shift: Bottom-right corner's centripetal shift for
                current level with shape (N, 2, H, W).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        """
        if isinstance(img_meta, (list, tuple)):
            img_meta = img_meta[0]
        batch_bboxes, batch_scores, batch_clses = self.decode_heatmap(tl_heat=tl_heat.sigmoid(), br_heat=br_heat.sigmoid(), tl_off=tl_off, br_off=br_off, tl_emb=tl_emb, br_emb=br_emb, tl_centripetal_shift=tl_centripetal_shift, br_centripetal_shift=br_centripetal_shift, img_meta=img_meta, k=self.test_cfg.corner_topk, kernel=self.test_cfg.local_maximum_kernel, distance_threshold=self.test_cfg.distance_threshold)
        if rescale:
            batch_bboxes /= batch_bboxes.new_tensor(img_meta['scale_factor'])
        bboxes = batch_bboxes.view([-1, 4])
        scores = batch_scores.view(-1)
        clses = batch_clses.view(-1)
        detections = torch.cat([bboxes, scores.unsqueeze(-1)], -1)
        keepinds = detections[:, -1] > -0.1
        detections = detections[keepinds]
        labels = clses[keepinds]
        if with_nms:
            detections, labels = self._bboxes_nms(detections, labels, self.test_cfg)
        return (detections, labels)

    def _bboxes_nms(self, bboxes, labels, cfg):
        if 'nms_cfg' in cfg:
            warning.warn('nms_cfg in test_cfg will be deprecated. Please rename it as nms')
        if 'nms' not in cfg:
            cfg.nms = cfg.nms_cfg
        if labels.numel() > 0:
            max_num = cfg.max_per_img
            bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1].contiguous(), labels, cfg.nms)
            if max_num > 0:
                bboxes = bboxes[:max_num]
                labels = labels[keep][:max_num]
        return (bboxes, labels)

    def decode_heatmap(self, tl_heat, br_heat, tl_off, br_off, tl_emb=None, br_emb=None, tl_centripetal_shift=None, br_centripetal_shift=None, img_meta=None, k=100, kernel=3, distance_threshold=0.5, num_dets=1000):
        """Transform outputs for a single batch item into raw bbox predictions.

        Args:
            tl_heat (Tensor): Top-left corner heatmap for current level with
                shape (N, num_classes, H, W).
            br_heat (Tensor): Bottom-right corner heatmap for current level
                with shape (N, num_classes, H, W).
            tl_off (Tensor): Top-left corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            br_off (Tensor): Bottom-right corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            tl_emb (Tensor | None): Top-left corner embedding for current
                level with shape (N, corner_emb_channels, H, W).
            br_emb (Tensor | None): Bottom-right corner embedding for current
                level with shape (N, corner_emb_channels, H, W).
            tl_centripetal_shift (Tensor | None): Top-left centripetal shift
                for current level with shape (N, 2, H, W).
            br_centripetal_shift (Tensor | None): Bottom-right centripetal
                shift for current level with shape (N, 2, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            k (int): Get top k corner keypoints from heatmap.
            kernel (int): Max pooling kernel for extract local maximum pixels.
            distance_threshold (float): Distance threshold. Top-left and
                bottom-right corner keypoints with feature distance less than
                the threshold will be regarded as keypoints from same object.
            num_dets (int): Num of raw boxes before doing nms.

        Returns:
            tuple[torch.Tensor]: Decoded output of CornerHead, containing the
            following Tensors:

            - bboxes (Tensor): Coords of each box.
            - scores (Tensor): Scores of each box.
            - clses (Tensor): Categories of each box.
        """
        with_embedding = tl_emb is not None and br_emb is not None
        with_centripetal_shift = tl_centripetal_shift is not None and br_centripetal_shift is not None
        assert with_embedding + with_centripetal_shift == 1
        batch, _, height, width = tl_heat.size()
        if torch.onnx.is_in_onnx_export():
            inp_h, inp_w = img_meta['pad_shape_for_onnx'][:2]
        else:
            inp_h, inp_w, _ = img_meta['pad_shape']
        tl_heat = get_local_maximum(tl_heat, kernel=kernel)
        br_heat = get_local_maximum(br_heat, kernel=kernel)
        tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = get_topk_from_heatmap(tl_heat, k=k)
        br_scores, br_inds, br_clses, br_ys, br_xs = get_topk_from_heatmap(br_heat, k=k)
        tl_ys = tl_ys.view(batch, k, 1).repeat(1, 1, k)
        tl_xs = tl_xs.view(batch, k, 1).repeat(1, 1, k)
        br_ys = br_ys.view(batch, 1, k).repeat(1, k, 1)
        br_xs = br_xs.view(batch, 1, k).repeat(1, k, 1)
        tl_off = transpose_and_gather_feat(tl_off, tl_inds)
        tl_off = tl_off.view(batch, k, 1, 2)
        br_off = transpose_and_gather_feat(br_off, br_inds)
        br_off = br_off.view(batch, 1, k, 2)
        tl_xs = tl_xs + tl_off[..., 0]
        tl_ys = tl_ys + tl_off[..., 1]
        br_xs = br_xs + br_off[..., 0]
        br_ys = br_ys + br_off[..., 1]
        if with_centripetal_shift:
            tl_centripetal_shift = transpose_and_gather_feat(tl_centripetal_shift, tl_inds).view(batch, k, 1, 2).exp()
            br_centripetal_shift = transpose_and_gather_feat(br_centripetal_shift, br_inds).view(batch, 1, k, 2).exp()
            tl_ctxs = tl_xs + tl_centripetal_shift[..., 0]
            tl_ctys = tl_ys + tl_centripetal_shift[..., 1]
            br_ctxs = br_xs - br_centripetal_shift[..., 0]
            br_ctys = br_ys - br_centripetal_shift[..., 1]
        tl_xs *= inp_w / width
        tl_ys *= inp_h / height
        br_xs *= inp_w / width
        br_ys *= inp_h / height
        if with_centripetal_shift:
            tl_ctxs *= inp_w / width
            tl_ctys *= inp_h / height
            br_ctxs *= inp_w / width
            br_ctys *= inp_h / height
        x_off, y_off = (0, 0)
        if not torch.onnx.is_in_onnx_export():
            if 'border' in img_meta:
                x_off = img_meta['border'][2]
                y_off = img_meta['border'][0]
        tl_xs -= x_off
        tl_ys -= y_off
        br_xs -= x_off
        br_ys -= y_off
        zeros = tl_xs.new_zeros(*tl_xs.size())
        tl_xs = torch.where(tl_xs > 0.0, tl_xs, zeros)
        tl_ys = torch.where(tl_ys > 0.0, tl_ys, zeros)
        br_xs = torch.where(br_xs > 0.0, br_xs, zeros)
        br_ys = torch.where(br_ys > 0.0, br_ys, zeros)
        bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)
        area_bboxes = ((br_xs - tl_xs) * (br_ys - tl_ys)).abs()
        if with_centripetal_shift:
            tl_ctxs -= x_off
            tl_ctys -= y_off
            br_ctxs -= x_off
            br_ctys -= y_off
            tl_ctxs *= tl_ctxs.gt(0.0).type_as(tl_ctxs)
            tl_ctys *= tl_ctys.gt(0.0).type_as(tl_ctys)
            br_ctxs *= br_ctxs.gt(0.0).type_as(br_ctxs)
            br_ctys *= br_ctys.gt(0.0).type_as(br_ctys)
            ct_bboxes = torch.stack((tl_ctxs, tl_ctys, br_ctxs, br_ctys), dim=3)
            area_ct_bboxes = ((br_ctxs - tl_ctxs) * (br_ctys - tl_ctys)).abs()
            rcentral = torch.zeros_like(ct_bboxes)
            mu = torch.ones_like(area_bboxes) / 2.4
            mu[area_bboxes > 3500] = 1 / 2.1
            bboxes_center_x = (bboxes[..., 0] + bboxes[..., 2]) / 2
            bboxes_center_y = (bboxes[..., 1] + bboxes[..., 3]) / 2
            rcentral[..., 0] = bboxes_center_x - mu * (bboxes[..., 2] - bboxes[..., 0]) / 2
            rcentral[..., 1] = bboxes_center_y - mu * (bboxes[..., 3] - bboxes[..., 1]) / 2
            rcentral[..., 2] = bboxes_center_x + mu * (bboxes[..., 2] - bboxes[..., 0]) / 2
            rcentral[..., 3] = bboxes_center_y + mu * (bboxes[..., 3] - bboxes[..., 1]) / 2
            area_rcentral = ((rcentral[..., 2] - rcentral[..., 0]) * (rcentral[..., 3] - rcentral[..., 1])).abs()
            dists = area_ct_bboxes / area_rcentral
            tl_ctx_inds = (ct_bboxes[..., 0] <= rcentral[..., 0]) | (ct_bboxes[..., 0] >= rcentral[..., 2])
            tl_cty_inds = (ct_bboxes[..., 1] <= rcentral[..., 1]) | (ct_bboxes[..., 1] >= rcentral[..., 3])
            br_ctx_inds = (ct_bboxes[..., 2] <= rcentral[..., 0]) | (ct_bboxes[..., 2] >= rcentral[..., 2])
            br_cty_inds = (ct_bboxes[..., 3] <= rcentral[..., 1]) | (ct_bboxes[..., 3] >= rcentral[..., 3])
        if with_embedding:
            tl_emb = transpose_and_gather_feat(tl_emb, tl_inds)
            tl_emb = tl_emb.view(batch, k, 1)
            br_emb = transpose_and_gather_feat(br_emb, br_inds)
            br_emb = br_emb.view(batch, 1, k)
            dists = torch.abs(tl_emb - br_emb)
        tl_scores = tl_scores.view(batch, k, 1).repeat(1, 1, k)
        br_scores = br_scores.view(batch, 1, k).repeat(1, k, 1)
        scores = (tl_scores + br_scores) / 2
        tl_clses = tl_clses.view(batch, k, 1).repeat(1, 1, k)
        br_clses = br_clses.view(batch, 1, k).repeat(1, k, 1)
        cls_inds = tl_clses != br_clses
        dist_inds = dists > distance_threshold
        width_inds = br_xs <= tl_xs
        height_inds = br_ys <= tl_ys
        negative_scores = -1 * torch.ones_like(scores)
        scores = torch.where(cls_inds, negative_scores, scores)
        scores = torch.where(width_inds, negative_scores, scores)
        scores = torch.where(height_inds, negative_scores, scores)
        scores = torch.where(dist_inds, negative_scores, scores)
        if with_centripetal_shift:
            scores[tl_ctx_inds] = -1
            scores[tl_cty_inds] = -1
            scores[br_ctx_inds] = -1
            scores[br_cty_inds] = -1
        scores = scores.view(batch, -1)
        scores, inds = torch.topk(scores, num_dets)
        scores = scores.unsqueeze(2)
        bboxes = bboxes.view(batch, -1, 4)
        bboxes = gather_feat(bboxes, inds)
        clses = tl_clses.contiguous().view(batch, -1, 1)
        clses = gather_feat(clses, inds).float()
        return (bboxes, scores, clses)

    def onnx_export(self, tl_heats, br_heats, tl_embs, br_embs, tl_offs, br_offs, img_metas, rescale=False, with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            tl_heats (list[Tensor]): Top-left corner heatmaps for each level
                with shape (N, num_classes, H, W).
            br_heats (list[Tensor]): Bottom-right corner heatmaps for each
                level with shape (N, num_classes, H, W).
            tl_embs (list[Tensor]): Top-left corner embeddings for each level
                with shape (N, corner_emb_channels, H, W).
            br_embs (list[Tensor]): Bottom-right corner embeddings for each
                level with shape (N, corner_emb_channels, H, W).
            tl_offs (list[Tensor]): Top-left corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            br_offs (list[Tensor]): Bottom-right corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor, Tensor]: First tensor bboxes with shape
            [N, num_det, 5], 5 arrange as (x1, y1, x2, y2, score)
            and second element is class labels of shape [N, num_det].
        """
        assert tl_heats[-1].shape[0] == br_heats[-1].shape[0] == len(img_metas) == 1
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(self._get_bboxes_single(tl_heats[-1][img_id:img_id + 1, :], br_heats[-1][img_id:img_id + 1, :], tl_offs[-1][img_id:img_id + 1, :], br_offs[-1][img_id:img_id + 1, :], img_metas[img_id], tl_emb=tl_embs[-1][img_id:img_id + 1, :], br_emb=br_embs[-1][img_id:img_id + 1, :], rescale=rescale, with_nms=with_nms))
        detections, labels = result_list[0]
        return (detections.unsqueeze(0), labels.unsqueeze(0))

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

@HEADS.register_module()
class FoveaHead(AnchorFreeHead):
    """FoveaBox: Beyond Anchor-based Object Detector
    https://arxiv.org/abs/1904.03797
    """

    def __init__(self, num_classes, in_channels, base_edge_list=(16, 32, 64, 128, 256), scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)), sigma=0.4, with_deform=False, deform_groups=4, init_cfg=dict(type='Normal', layer='Conv2d', std=0.01, override=dict(type='Normal', name='conv_cls', std=0.01, bias_prob=0.01)), **kwargs):
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.sigma = sigma
        self.with_deform = with_deform
        self.deform_groups = deform_groups
        super().__init__(num_classes, in_channels, init_cfg=init_cfg, **kwargs)

    def _init_layers(self):
        super()._init_reg_convs()
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        if not self.with_deform:
            super()._init_cls_convs()
            self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        else:
            self.cls_convs = nn.ModuleList()
            self.cls_convs.append(ConvModule(self.feat_channels, self.feat_channels * 4, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None))
            self.cls_convs.append(ConvModule(self.feat_channels * 4, self.feat_channels * 4, 1, stride=1, padding=0, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None))
            self.feature_adaption = FeatureAlign(self.feat_channels, self.feat_channels, kernel_size=3, deform_groups=self.deform_groups)
            self.conv_cls = nn.Conv2d(int(self.feat_channels * 4), self.cls_out_channels, 3, padding=1)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        if self.with_deform:
            cls_feat = self.feature_adaption(cls_feat, bbox_pred.exp())
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)
        return (cls_score, bbox_pred)

    def loss(self, cls_scores, bbox_preds, gt_bbox_list, gt_label_list, img_metas, gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        points = self.prior_generator.grid_priors(featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_labels, flatten_bbox_targets = self.get_targets(gt_bbox_list, gt_label_list, featmap_sizes, points)
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < self.num_classes)).nonzero().view(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos + num_imgs)
        if num_pos > 0:
            pos_bbox_preds = flatten_bbox_preds[pos_inds]
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_weights = pos_bbox_targets.new_zeros(pos_bbox_targets.size()) + 1.0
            loss_bbox = self.loss_bbox(pos_bbox_preds, pos_bbox_targets, pos_weights, avg_factor=num_pos)
        else:
            loss_bbox = torch.tensor(0, dtype=flatten_bbox_preds.dtype, device=flatten_bbox_preds.device)
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    def get_targets(self, gt_bbox_list, gt_label_list, featmap_sizes, points):
        label_list, bbox_target_list = multi_apply(self._get_target_single, gt_bbox_list, gt_label_list, featmap_size_list=featmap_sizes, point_list=points)
        flatten_labels = [torch.cat([labels_level_img.flatten() for labels_level_img in labels_level]) for labels_level in zip(*label_list)]
        flatten_bbox_targets = [torch.cat([bbox_targets_level_img.reshape(-1, 4) for bbox_targets_level_img in bbox_targets_level]) for bbox_targets_level in zip(*bbox_target_list)]
        flatten_labels = torch.cat(flatten_labels)
        flatten_bbox_targets = torch.cat(flatten_bbox_targets)
        return (flatten_labels, flatten_bbox_targets)

    def _get_target_single(self, gt_bboxes_raw, gt_labels_raw, featmap_size_list=None, point_list=None):
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
        label_list = []
        bbox_target_list = []
        for base_len, (lower_bound, upper_bound), stride, featmap_size, points in zip(self.base_edge_list, self.scale_ranges, self.strides, featmap_size_list, point_list):
            points = points.view(*featmap_size, 2)
            x, y = (points[..., 0], points[..., 1])
            labels = gt_labels_raw.new_zeros(featmap_size) + self.num_classes
            bbox_targets = gt_bboxes_raw.new(featmap_size[0], featmap_size[1], 4) + 1
            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                label_list.append(labels)
                bbox_target_list.append(torch.log(bbox_targets))
                continue
            _, hit_index_order = torch.sort(-gt_areas[hit_indices])
            hit_indices = hit_indices[hit_index_order]
            gt_bboxes = gt_bboxes_raw[hit_indices, :] / stride
            gt_labels = gt_labels_raw[hit_indices]
            half_w = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0])
            half_h = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
            pos_left = torch.ceil(gt_bboxes[:, 0] + (1 - self.sigma) * half_w - 0.5).long().clamp(0, featmap_size[1] - 1)
            pos_right = torch.floor(gt_bboxes[:, 0] + (1 + self.sigma) * half_w - 0.5).long().clamp(0, featmap_size[1] - 1)
            pos_top = torch.ceil(gt_bboxes[:, 1] + (1 - self.sigma) * half_h - 0.5).long().clamp(0, featmap_size[0] - 1)
            pos_down = torch.floor(gt_bboxes[:, 1] + (1 + self.sigma) * half_h - 0.5).long().clamp(0, featmap_size[0] - 1)
            for px1, py1, px2, py2, label, (gt_x1, gt_y1, gt_x2, gt_y2) in zip(pos_left, pos_top, pos_right, pos_down, gt_labels, gt_bboxes_raw[hit_indices, :]):
                labels[py1:py2 + 1, px1:px2 + 1] = label
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 0] = (x[py1:py2 + 1, px1:px2 + 1] - gt_x1) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 1] = (y[py1:py2 + 1, px1:px2 + 1] - gt_y1) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 2] = (gt_x2 - x[py1:py2 + 1, px1:px2 + 1]) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 3] = (gt_y2 - y[py1:py2 + 1, px1:px2 + 1]) / base_len
            bbox_targets = bbox_targets.clamp(min=1.0 / 16, max=16.0)
            label_list.append(labels)
            bbox_target_list.append(torch.log(bbox_targets))
        return (label_list, bbox_target_list)

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
                levels of a single image. Fovea head does not need this value.
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
        for level_idx, (cls_score, bbox_pred, stride, base_len, priors) in enumerate(zip(cls_score_list, bbox_pred_list, self.strides, self.base_edge_list, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            results = filter_scores_and_topk(scores, cfg.score_thr, nms_pre, dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, _, filtered_results = results
            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            bboxes = self._bbox_decode(priors, bbox_pred, base_len, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes, img_meta['scale_factor'], cfg, rescale, with_nms)

    def _bbox_decode(self, priors, bbox_pred, base_len, max_shape):
        bbox_pred = bbox_pred.exp()
        y = priors[:, 1]
        x = priors[:, 0]
        x1 = (x - base_len * bbox_pred[:, 0]).clamp(min=0, max=max_shape[1] - 1)
        y1 = (y - base_len * bbox_pred[:, 1]).clamp(min=0, max=max_shape[0] - 1)
        x2 = (x + base_len * bbox_pred[:, 2]).clamp(min=0, max=max_shape[1] - 1)
        y2 = (y + base_len * bbox_pred[:, 3]).clamp(min=0, max=max_shape[0] - 1)
        decoded_bboxes = torch.stack([x1, y1, x2, y2], -1)
        return decoded_bboxes

    def _get_points_single(self, *args, **kwargs):
        """Get points according to feature map size.

        This function will be deprecated soon.
        """
        warnings.warn('`_get_points_single` in `FoveaHead` will be deprecated soon, we support a multi level point generator nowyou can get points of a single level feature map with `self.prior_generator.single_level_grid_priors` ')
        y, x = super()._get_points_single(*args, **kwargs)
        return (y + 0.5, x + 0.5)

@HEADS.register_module()
class BBoxHead(BaseModule):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self, with_avg_pool=False, with_cls=True, with_reg=True, roi_feat_size=7, in_channels=256, num_classes=80, bbox_coder=dict(type='DeltaXYWHBBoxCoder', clip_border=True, target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2]), reg_class_agnostic=False, reg_decoded_bbox=False, reg_predictor_cfg=dict(type='Linear'), cls_predictor_cfg=dict(type='Linear'), loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0), loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0), init_cfg=None):
        super(BBoxHead, self).__init__(init_cfg)
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.reg_predictor_cfg = reg_predictor_cfg
        self.cls_predictor_cfg = cls_predictor_cfg
        self.fp16_enabled = False
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = num_classes + 1
            self.fc_cls = build_linear_layer(self.cls_predictor_cfg, in_features=in_channels, out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = build_linear_layer(self.reg_predictor_cfg, in_features=in_channels, out_features=out_dim_reg)
        self.debug_imgs = None
        if init_cfg is None:
            self.init_cfg = []
            if self.with_cls:
                self.init_cfg += [dict(type='Normal', std=0.01, override=dict(name='fc_cls'))]
            if self.with_reg:
                self.init_cfg += [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]

    @property
    def custom_cls_channels(self):
        return getattr(self.loss_cls, 'custom_cls_channels', False)

    @property
    def custom_activation(self):
        return getattr(self.loss_cls, 'custom_activation', False)

    @property
    def custom_accuracy(self):
        return getattr(self.loss_cls, 'custom_accuracy', False)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
            else:
                x = torch.mean(x, dim=(-1, -2))
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return (cls_score, bbox_pred)

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg
        labels = pos_bboxes.new_full((num_samples,), self.num_classes, dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0
        return (labels, label_weights, bbox_targets, bbox_weights)

    def get_targets(self, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg, concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(self._get_target_single, pos_bboxes_list, neg_bboxes_list, pos_gt_bboxes_list, pos_gt_labels_list, cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return (labels, label_weights, bbox_targets, bbox_weights)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_weights, reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(cls_score, labels, label_weights, avg_factor=avg_factor, reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(pos_bbox_pred, bbox_targets[pos_inds.type(torch.bool)], bbox_weights[pos_inds.type(torch.bool)], avg_factor=bbox_targets.size(0), reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(cls_score, dim=-1) if cls_score is not None else None
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])
        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(bboxes.size()[0], -1)
        if cfg is None:
            return (bboxes, scores)
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return (det_bboxes, det_labels)

    @force_fp32(apply_to=('bbox_preds',))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image. The first column is
                the image id and the next 4 columns are x1, y1, x2, y2.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import kwarray
            >>> import numpy as np
            >>> from mmdet.core.bbox.demodata import random_boxes
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            >>> img_metas = [{'img_shape': (scale, scale)}
            ...              for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torch.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torch.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torch.randint(0, 2, (n_roi,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> # For each image, pretend random positive boxes are gts
            >>> is_label_pos = (labels.numpy() > 0).astype(np.int)
            >>> lbl_per_img = kwarray.group_items(is_label_pos,
            ...                                   img_ids.numpy())
            >>> pos_per_img = [sum(lbl_per_img.get(gid, []))
            ...                for gid in range(n_img)]
            >>> pos_is_gts = [
            >>>     torch.randint(0, 2, (npos,)).byte().sort(
            >>>         descending=True)[0]
            >>>     for npos in pos_per_img
            >>> ]
            >>> bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
            >>>                    pos_is_gts, img_metas)
            >>> print(bboxes_list)
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)
        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()
            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]
            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_, img_meta_)
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep
            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])
        return bboxes_list

    @force_fp32(apply_to=('bbox_pred',))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): Rois from `rpn_head` or last stage
                `bbox_head`, has shape (num_proposals, 4) or
                (num_proposals, 5).
            label (Tensor): Only used when `self.reg_class_agnostic`
                is False, has shape (num_proposals, ).
            bbox_pred (Tensor): Regression prediction of
                current stage `bbox_head`. When `self.reg_class_agnostic`
                is False, it has shape (n, num_classes * 4), otherwise
                it has shape (n, 4).
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)
        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4
        max_shape = img_meta['img_shape']
        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(rois, bbox_pred, max_shape=max_shape)
        else:
            bboxes = self.bbox_coder.decode(rois[:, 1:], bbox_pred, max_shape=max_shape)
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)
        return new_rois

    def onnx_export(self, rois, cls_score, bbox_pred, img_shape, cfg=None, **kwargs):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed.
                Has shape (B, num_boxes, 5)
            cls_score (Tensor): Box scores. has shape
                (B, num_boxes, num_classes + 1), 1 represent the background.
            bbox_pred (Tensor, optional): Box energies / deltas for,
                has shape (B, num_boxes, num_classes * 4) when.
            img_shape (torch.Tensor): Shape of image.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        assert rois.ndim == 3, 'Only support export two stage model to ONNX with batch dimension. '
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(cls_score, dim=-1) if cls_score is not None else None
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[..., 1:].clone()
            if img_shape is not None:
                max_shape = bboxes.new_tensor(img_shape)[..., :2]
                min_xy = bboxes.new_tensor(0)
                max_xy = torch.cat([max_shape] * 2, dim=-1).flip(-1).unsqueeze(-2)
                bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
                bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)
        from mmdet.core.export import add_dummy_nms_for_onnx
        max_output_boxes_per_class = cfg.nms.get('max_output_boxes_per_class', cfg.max_per_img)
        iou_threshold = cfg.nms.get('iou_threshold', 0.5)
        score_threshold = cfg.score_thr
        nms_pre = cfg.get('deploy_nms_pre', -1)
        scores = scores[..., :self.num_classes]
        if self.reg_class_agnostic:
            return add_dummy_nms_for_onnx(bboxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, pre_top_k=nms_pre, after_top_k=cfg.max_per_img)
        else:
            batch_size = scores.shape[0]
            labels = torch.arange(self.num_classes, dtype=torch.long).to(scores.device)
            labels = labels.view(1, 1, -1).expand_as(scores)
            labels = labels.reshape(batch_size, -1)
            scores = scores.reshape(batch_size, -1)
            bboxes = bboxes.reshape(batch_size, -1, 4)
            max_size = torch.max(img_shape)
            offsets = (labels * max_size + 1).unsqueeze(2)
            bboxes_for_nms = bboxes + offsets
            batch_dets, labels = add_dummy_nms_for_onnx(bboxes_for_nms, scores.unsqueeze(2), max_output_boxes_per_class, iou_threshold, score_threshold, pre_top_k=nms_pre, after_top_k=cfg.max_per_img, labels=labels)
            offsets = (labels * max_size + 1).unsqueeze(2)
            bboxes, scores = (batch_dets[..., 0:4], batch_dets[..., 4:5])
            bboxes -= offsets
            batch_dets = torch.cat([bboxes, scores], dim=2)
            return (batch_dets, labels)

@HEADS.register_module()
class SABLHead(BaseModule):
    """Side-Aware Boundary Localization (SABL) for RoI-Head.

    Side-Aware features are extracted by conv layers
    with an attention mechanism.
    Boundary Localization with Bucketing and Bucketing Guided Rescoring
    are implemented in BucketingBBoxCoder.

    Please refer to https://arxiv.org/abs/1912.04260 for more details.

    Args:
        cls_in_channels (int): Input channels of cls RoI feature.             Defaults to 256.
        reg_in_channels (int): Input channels of reg RoI feature.             Defaults to 256.
        roi_feat_size (int): Size of RoI features. Defaults to 7.
        reg_feat_up_ratio (int): Upsample ratio of reg features.             Defaults to 2.
        reg_pre_kernel (int): Kernel of 2D conv layers before             attention pooling. Defaults to 3.
        reg_post_kernel (int): Kernel of 1D conv layers after             attention pooling. Defaults to 3.
        reg_pre_num (int): Number of pre convs. Defaults to 2.
        reg_post_num (int): Number of post convs. Defaults to 1.
        num_classes (int): Number of classes in dataset. Defaults to 80.
        cls_out_channels (int): Hidden channels in cls fcs. Defaults to 1024.
        reg_offset_out_channels (int): Hidden and output channel             of reg offset branch. Defaults to 256.
        reg_cls_out_channels (int): Hidden and output channel             of reg cls branch. Defaults to 256.
        num_cls_fcs (int): Number of fcs for cls branch. Defaults to 1.
        num_reg_fcs (int): Number of fcs for reg branch.. Defaults to 0.
        reg_class_agnostic (bool): Class agnostic regression or not.             Defaults to True.
        norm_cfg (dict): Config of norm layers. Defaults to None.
        bbox_coder (dict): Config of bbox coder. Defaults 'BucketingBBoxCoder'.
        loss_cls (dict): Config of classification loss.
        loss_bbox_cls (dict): Config of classification loss for bbox branch.
        loss_bbox_reg (dict): Config of regression loss for bbox branch.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self, num_classes, cls_in_channels=256, reg_in_channels=256, roi_feat_size=7, reg_feat_up_ratio=2, reg_pre_kernel=3, reg_post_kernel=3, reg_pre_num=2, reg_post_num=1, cls_out_channels=1024, reg_offset_out_channels=256, reg_cls_out_channels=256, num_cls_fcs=1, num_reg_fcs=0, reg_class_agnostic=True, norm_cfg=None, bbox_coder=dict(type='BucketingBBoxCoder', num_buckets=14, scale_factor=1.7), loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0), loss_bbox_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_bbox_reg=dict(type='SmoothL1Loss', beta=0.1, loss_weight=1.0), init_cfg=None):
        super(SABLHead, self).__init__(init_cfg)
        self.cls_in_channels = cls_in_channels
        self.reg_in_channels = reg_in_channels
        self.roi_feat_size = roi_feat_size
        self.reg_feat_up_ratio = int(reg_feat_up_ratio)
        self.num_buckets = bbox_coder['num_buckets']
        assert self.reg_feat_up_ratio // 2 >= 1
        self.up_reg_feat_size = roi_feat_size * self.reg_feat_up_ratio
        assert self.up_reg_feat_size == bbox_coder['num_buckets']
        self.reg_pre_kernel = reg_pre_kernel
        self.reg_post_kernel = reg_post_kernel
        self.reg_pre_num = reg_pre_num
        self.reg_post_num = reg_post_num
        self.num_classes = num_classes
        self.cls_out_channels = cls_out_channels
        self.reg_offset_out_channels = reg_offset_out_channels
        self.reg_cls_out_channels = reg_cls_out_channels
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_fcs = num_reg_fcs
        self.reg_class_agnostic = reg_class_agnostic
        assert self.reg_class_agnostic
        self.norm_cfg = norm_cfg
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_cls = build_loss(loss_bbox_cls)
        self.loss_bbox_reg = build_loss(loss_bbox_reg)
        self.cls_fcs = self._add_fc_branch(self.num_cls_fcs, self.cls_in_channels, self.roi_feat_size, self.cls_out_channels)
        self.side_num = int(np.ceil(self.num_buckets / 2))
        if self.reg_feat_up_ratio > 1:
            self.upsample_x = nn.ConvTranspose1d(reg_in_channels, reg_in_channels, self.reg_feat_up_ratio, stride=self.reg_feat_up_ratio)
            self.upsample_y = nn.ConvTranspose1d(reg_in_channels, reg_in_channels, self.reg_feat_up_ratio, stride=self.reg_feat_up_ratio)
        self.reg_pre_convs = nn.ModuleList()
        for i in range(self.reg_pre_num):
            reg_pre_conv = ConvModule(reg_in_channels, reg_in_channels, kernel_size=reg_pre_kernel, padding=reg_pre_kernel // 2, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))
            self.reg_pre_convs.append(reg_pre_conv)
        self.reg_post_conv_xs = nn.ModuleList()
        for i in range(self.reg_post_num):
            reg_post_conv_x = ConvModule(reg_in_channels, reg_in_channels, kernel_size=(1, reg_post_kernel), padding=(0, reg_post_kernel // 2), norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))
            self.reg_post_conv_xs.append(reg_post_conv_x)
        self.reg_post_conv_ys = nn.ModuleList()
        for i in range(self.reg_post_num):
            reg_post_conv_y = ConvModule(reg_in_channels, reg_in_channels, kernel_size=(reg_post_kernel, 1), padding=(reg_post_kernel // 2, 0), norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))
            self.reg_post_conv_ys.append(reg_post_conv_y)
        self.reg_conv_att_x = nn.Conv2d(reg_in_channels, 1, 1)
        self.reg_conv_att_y = nn.Conv2d(reg_in_channels, 1, 1)
        self.fc_cls = nn.Linear(self.cls_out_channels, self.num_classes + 1)
        self.relu = nn.ReLU(inplace=True)
        self.reg_cls_fcs = self._add_fc_branch(self.num_reg_fcs, self.reg_in_channels, 1, self.reg_cls_out_channels)
        self.reg_offset_fcs = self._add_fc_branch(self.num_reg_fcs, self.reg_in_channels, 1, self.reg_offset_out_channels)
        self.fc_reg_cls = nn.Linear(self.reg_cls_out_channels, 1)
        self.fc_reg_offset = nn.Linear(self.reg_offset_out_channels, 1)
        if init_cfg is None:
            self.init_cfg = [dict(type='Xavier', layer='Linear', distribution='uniform', override=[dict(type='Normal', name='reg_conv_att_x', std=0.01), dict(type='Normal', name='reg_conv_att_y', std=0.01), dict(type='Normal', name='fc_reg_cls', std=0.01), dict(type='Normal', name='fc_cls', std=0.01), dict(type='Normal', name='fc_reg_offset', std=0.001)])]
            if self.reg_feat_up_ratio > 1:
                self.init_cfg += [dict(type='Kaiming', distribution='normal', override=[dict(name='upsample_x'), dict(name='upsample_y')])]

    @property
    def custom_cls_channels(self):
        return getattr(self.loss_cls, 'custom_cls_channels', False)

    @property
    def custom_activation(self):
        return getattr(self.loss_cls, 'custom_activation', False)

    @property
    def custom_accuracy(self):
        return getattr(self.loss_cls, 'custom_accuracy', False)

    def _add_fc_branch(self, num_branch_fcs, in_channels, roi_feat_size, fc_out_channels):
        in_channels = in_channels * roi_feat_size * roi_feat_size
        branch_fcs = nn.ModuleList()
        for i in range(num_branch_fcs):
            fc_in_channels = in_channels if i == 0 else fc_out_channels
            branch_fcs.append(nn.Linear(fc_in_channels, fc_out_channels))
        return branch_fcs

    def cls_forward(self, cls_x):
        cls_x = cls_x.view(cls_x.size(0), -1)
        for fc in self.cls_fcs:
            cls_x = self.relu(fc(cls_x))
        cls_score = self.fc_cls(cls_x)
        return cls_score

    def attention_pool(self, reg_x):
        """Extract direction-specific features fx and fy with attention
        methanism."""
        reg_fx = reg_x
        reg_fy = reg_x
        reg_fx_att = self.reg_conv_att_x(reg_fx).sigmoid()
        reg_fy_att = self.reg_conv_att_y(reg_fy).sigmoid()
        reg_fx_att = reg_fx_att / reg_fx_att.sum(dim=2).unsqueeze(2)
        reg_fy_att = reg_fy_att / reg_fy_att.sum(dim=3).unsqueeze(3)
        reg_fx = (reg_fx * reg_fx_att).sum(dim=2)
        reg_fy = (reg_fy * reg_fy_att).sum(dim=3)
        return (reg_fx, reg_fy)

    def side_aware_feature_extractor(self, reg_x):
        """Refine and extract side-aware features without split them."""
        for reg_pre_conv in self.reg_pre_convs:
            reg_x = reg_pre_conv(reg_x)
        reg_fx, reg_fy = self.attention_pool(reg_x)
        if self.reg_post_num > 0:
            reg_fx = reg_fx.unsqueeze(2)
            reg_fy = reg_fy.unsqueeze(3)
            for i in range(self.reg_post_num):
                reg_fx = self.reg_post_conv_xs[i](reg_fx)
                reg_fy = self.reg_post_conv_ys[i](reg_fy)
            reg_fx = reg_fx.squeeze(2)
            reg_fy = reg_fy.squeeze(3)
        if self.reg_feat_up_ratio > 1:
            reg_fx = self.relu(self.upsample_x(reg_fx))
            reg_fy = self.relu(self.upsample_y(reg_fy))
        reg_fx = torch.transpose(reg_fx, 1, 2)
        reg_fy = torch.transpose(reg_fy, 1, 2)
        return (reg_fx.contiguous(), reg_fy.contiguous())

    def reg_pred(self, x, offset_fcs, cls_fcs):
        """Predict bucketing estimation (cls_pred) and fine regression (offset
        pred) with side-aware features."""
        x_offset = x.view(-1, self.reg_in_channels)
        x_cls = x.view(-1, self.reg_in_channels)
        for fc in offset_fcs:
            x_offset = self.relu(fc(x_offset))
        for fc in cls_fcs:
            x_cls = self.relu(fc(x_cls))
        offset_pred = self.fc_reg_offset(x_offset)
        cls_pred = self.fc_reg_cls(x_cls)
        offset_pred = offset_pred.view(x.size(0), -1)
        cls_pred = cls_pred.view(x.size(0), -1)
        return (offset_pred, cls_pred)

    def side_aware_split(self, feat):
        """Split side-aware features aligned with orders of bucketing
        targets."""
        l_end = int(np.ceil(self.up_reg_feat_size / 2))
        r_start = int(np.floor(self.up_reg_feat_size / 2))
        feat_fl = feat[:, :l_end]
        feat_fr = feat[:, r_start:].flip(dims=(1,))
        feat_fl = feat_fl.contiguous()
        feat_fr = feat_fr.contiguous()
        feat = torch.cat([feat_fl, feat_fr], dim=-1)
        return feat

    def bbox_pred_split(self, bbox_pred, num_proposals_per_img):
        """Split batch bbox prediction back to each image."""
        bucket_cls_preds, bucket_offset_preds = bbox_pred
        bucket_cls_preds = bucket_cls_preds.split(num_proposals_per_img, 0)
        bucket_offset_preds = bucket_offset_preds.split(num_proposals_per_img, 0)
        bbox_pred = tuple(zip(bucket_cls_preds, bucket_offset_preds))
        return bbox_pred

    def reg_forward(self, reg_x):
        outs = self.side_aware_feature_extractor(reg_x)
        edge_offset_preds = []
        edge_cls_preds = []
        reg_fx = outs[0]
        reg_fy = outs[1]
        offset_pred_x, cls_pred_x = self.reg_pred(reg_fx, self.reg_offset_fcs, self.reg_cls_fcs)
        offset_pred_y, cls_pred_y = self.reg_pred(reg_fy, self.reg_offset_fcs, self.reg_cls_fcs)
        offset_pred_x = self.side_aware_split(offset_pred_x)
        offset_pred_y = self.side_aware_split(offset_pred_y)
        cls_pred_x = self.side_aware_split(cls_pred_x)
        cls_pred_y = self.side_aware_split(cls_pred_y)
        edge_offset_preds = torch.cat([offset_pred_x, offset_pred_y], dim=-1)
        edge_cls_preds = torch.cat([cls_pred_x, cls_pred_y], dim=-1)
        return (edge_cls_preds, edge_offset_preds)

    def forward(self, x):
        bbox_pred = self.reg_forward(x)
        cls_score = self.cls_forward(x)
        return (cls_score, bbox_pred)

    def get_targets(self, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        cls_reg_targets = self.bucket_target(pos_proposals, neg_proposals, pos_gt_bboxes, pos_gt_labels, rcnn_train_cfg)
        labels, label_weights, bucket_cls_targets, bucket_cls_weights, bucket_offset_targets, bucket_offset_weights = cls_reg_targets
        return (labels, label_weights, (bucket_cls_targets, bucket_offset_targets), (bucket_cls_weights, bucket_offset_weights))

    def bucket_target(self, pos_proposals_list, neg_proposals_list, pos_gt_bboxes_list, pos_gt_labels_list, rcnn_train_cfg, concat=True):
        labels, label_weights, bucket_cls_targets, bucket_cls_weights, bucket_offset_targets, bucket_offset_weights = multi_apply(self._bucket_target_single, pos_proposals_list, neg_proposals_list, pos_gt_bboxes_list, pos_gt_labels_list, cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bucket_cls_targets = torch.cat(bucket_cls_targets, 0)
            bucket_cls_weights = torch.cat(bucket_cls_weights, 0)
            bucket_offset_targets = torch.cat(bucket_offset_targets, 0)
            bucket_offset_weights = torch.cat(bucket_offset_weights, 0)
        return (labels, label_weights, bucket_cls_targets, bucket_cls_weights, bucket_offset_targets, bucket_offset_weights)

    def _bucket_target_single(self, pos_proposals, neg_proposals, pos_gt_bboxes, pos_gt_labels, cfg):
        """Compute bucketing estimation targets and fine regression targets for
        a single image.

        Args:
            pos_proposals (Tensor): positive proposals of a single image,
                 Shape (n_pos, 4)
            neg_proposals (Tensor): negative proposals of a single image,
                 Shape (n_neg, 4).
            pos_gt_bboxes (Tensor): gt bboxes assigned to positive proposals
                 of a single image, Shape (n_pos, 4).
            pos_gt_labels (Tensor): gt labels assigned to positive proposals
                 of a single image, Shape (n_pos, ).
            cfg (dict): Config of calculating targets

        Returns:
            tuple:

                - labels (Tensor): Labels in a single image.                     Shape (n,).
                - label_weights (Tensor): Label weights in a single image.                    Shape (n,)
                - bucket_cls_targets (Tensor): Bucket cls targets in                     a single image. Shape (n, num_buckets*2).
                - bucket_cls_weights (Tensor): Bucket cls weights in                     a single image. Shape (n, num_buckets*2).
                - bucket_offset_targets (Tensor): Bucket offset targets                     in a single image. Shape (n, num_buckets*2).
                - bucket_offset_targets (Tensor): Bucket offset weights                     in a single image. Shape (n, num_buckets*2).
        """
        num_pos = pos_proposals.size(0)
        num_neg = neg_proposals.size(0)
        num_samples = num_pos + num_neg
        labels = pos_gt_bboxes.new_full((num_samples,), self.num_classes, dtype=torch.long)
        label_weights = pos_proposals.new_zeros(num_samples)
        bucket_cls_targets = pos_proposals.new_zeros(num_samples, 4 * self.side_num)
        bucket_cls_weights = pos_proposals.new_zeros(num_samples, 4 * self.side_num)
        bucket_offset_targets = pos_proposals.new_zeros(num_samples, 4 * self.side_num)
        bucket_offset_weights = pos_proposals.new_zeros(num_samples, 4 * self.side_num)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            label_weights[:num_pos] = 1.0
            pos_bucket_offset_targets, pos_bucket_offset_weights, pos_bucket_cls_targets, pos_bucket_cls_weights = self.bbox_coder.encode(pos_proposals, pos_gt_bboxes)
            bucket_cls_targets[:num_pos, :] = pos_bucket_cls_targets
            bucket_cls_weights[:num_pos, :] = pos_bucket_cls_weights
            bucket_offset_targets[:num_pos, :] = pos_bucket_offset_targets
            bucket_offset_weights[:num_pos, :] = pos_bucket_offset_weights
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0
        return (labels, label_weights, bucket_cls_targets, bucket_cls_weights, bucket_offset_targets, bucket_offset_weights)

    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_weights, reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)
            losses['loss_cls'] = self.loss_cls(cls_score, labels, label_weights, avg_factor=avg_factor, reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bucket_cls_preds, bucket_offset_preds = bbox_pred
            bucket_cls_targets, bucket_offset_targets = bbox_targets
            bucket_cls_weights, bucket_offset_weights = bbox_weights
            bucket_cls_preds = bucket_cls_preds.view(-1, self.side_num)
            bucket_cls_targets = bucket_cls_targets.view(-1, self.side_num)
            bucket_cls_weights = bucket_cls_weights.view(-1, self.side_num)
            losses['loss_bbox_cls'] = self.loss_bbox_cls(bucket_cls_preds, bucket_cls_targets, bucket_cls_weights, avg_factor=bucket_cls_targets.size(0), reduction_override=reduction_override)
            losses['loss_bbox_reg'] = self.loss_bbox_reg(bucket_offset_preds, bucket_offset_targets, bucket_offset_weights, avg_factor=bucket_offset_targets.size(0), reduction_override=reduction_override)
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if bbox_pred is not None:
            bboxes, confidences = self.bbox_coder.decode(rois[:, 1:], bbox_pred, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            confidences = None
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)
        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)
        if cfg is None:
            return (bboxes, scores)
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img, score_factors=confidences)
            return (det_bboxes, det_labels)

    @force_fp32(apply_to=('bbox_preds',))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (list[Tensor]): Shape [(n*bs, num_buckets*2),                 (n*bs, num_buckets*2)].
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)
        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()
            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            edge_cls_preds, edge_offset_preds = bbox_preds
            edge_cls_preds_ = edge_cls_preds[inds]
            edge_offset_preds_ = edge_offset_preds[inds]
            bbox_pred_ = [edge_cls_preds_, edge_offset_preds_]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]
            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_, img_meta_)
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep
            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])
        return bboxes_list

    @force_fp32(apply_to=('bbox_pred',))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (list[Tensor]): shape [(n, num_buckets *2),                 (n, num_buckets *2)]
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5
        if rois.size(1) == 4:
            new_rois, _ = self.bbox_coder.decode(rois, bbox_pred, img_meta['img_shape'])
        else:
            bboxes, _ = self.bbox_coder.decode(rois[:, 1:], bbox_pred, img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)
        return new_rois

@HEADS.register_module()
class DIIHead(BBoxHead):
    """Dynamic Instance Interactive Head for `Sparse R-CNN: End-to-End Object
    Detection with Learnable Proposals <https://arxiv.org/abs/2011.12450>`_

    Args:
        num_classes (int): Number of class in dataset.
            Defaults to 80.
        num_ffn_fcs (int): The number of fully-connected
            layers in FFNs. Defaults to 2.
        num_heads (int): The hidden dimension of FFNs.
            Defaults to 8.
        num_cls_fcs (int): The number of fully-connected
            layers in classification subnet. Defaults to 1.
        num_reg_fcs (int): The number of fully-connected
            layers in regression subnet. Defaults to 3.
        feedforward_channels (int): The hidden dimension
            of FFNs. Defaults to 2048
        in_channels (int): Hidden_channels of MultiheadAttention.
            Defaults to 256.
        dropout (float): Probability of drop the channel.
            Defaults to 0.0
        ffn_act_cfg (dict): The activation config for FFNs.
        dynamic_conv_cfg (dict): The convolution config
            for DynamicConv.
        loss_iou (dict): The config for iou or giou loss.

    """

    def __init__(self, num_classes=80, num_ffn_fcs=2, num_heads=8, num_cls_fcs=1, num_reg_fcs=3, feedforward_channels=2048, in_channels=256, dropout=0.0, ffn_act_cfg=dict(type='ReLU', inplace=True), dynamic_conv_cfg=dict(type='DynamicConv', in_channels=256, feat_channels=64, out_channels=256, input_feat_shape=7, act_cfg=dict(type='ReLU', inplace=True), norm_cfg=dict(type='LN')), loss_iou=dict(type='GIoULoss', loss_weight=2.0), init_cfg=None, **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization behavior, init_cfg is not allowed to be set'
        super(DIIHead, self).__init__(num_classes=num_classes, reg_decoded_bbox=True, reg_class_agnostic=True, init_cfg=init_cfg, **kwargs)
        self.loss_iou = build_loss(loss_iou)
        self.in_channels = in_channels
        self.fp16_enabled = False
        self.attention = MultiheadAttention(in_channels, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), in_channels)[1]
        self.instance_interactive_conv = build_transformer(dynamic_conv_cfg)
        self.instance_interactive_conv_dropout = nn.Dropout(dropout)
        self.instance_interactive_conv_norm = build_norm_layer(dict(type='LN'), in_channels)[1]
        self.ffn = FFN(in_channels, feedforward_channels, num_ffn_fcs, act_cfg=ffn_act_cfg, dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]
        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(nn.Linear(in_channels, in_channels, bias=False))
            self.cls_fcs.append(build_norm_layer(dict(type='LN'), in_channels)[1])
            self.cls_fcs.append(build_activation_layer(dict(type='ReLU', inplace=True)))
        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(in_channels, self.num_classes)
        else:
            self.fc_cls = nn.Linear(in_channels, self.num_classes + 1)
        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(nn.Linear(in_channels, in_channels, bias=False))
            self.reg_fcs.append(build_norm_layer(dict(type='LN'), in_channels)[1])
            self.reg_fcs.append(build_activation_layer(dict(type='ReLU', inplace=True)))
        self.fc_reg = nn.Linear(in_channels, 4)
        assert self.reg_class_agnostic, 'DIIHead only suppport `reg_class_agnostic=True` '
        assert self.reg_decoded_bbox, 'DIIHead only suppport `reg_decoded_bbox=True`'

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        super(DIIHead, self).init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                pass
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)

    @auto_fp16()
    def forward(self, roi_feat, proposal_feat):
        """Forward function of Dynamic Instance Interactive Head.

        Args:
            roi_feat (Tensor): Roi-pooling features with shape
                (batch_size*num_proposals, feature_dimensions,
                pooling_h , pooling_w).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size, num_proposals, feature_dimensions)

          Returns:
                tuple[Tensor]: Usually a tuple of classification scores
                and bbox prediction and a intermediate feature.

                    - cls_scores (Tensor): Classification scores for
                      all proposals, has shape
                      (batch_size, num_proposals, num_classes).
                    - bbox_preds (Tensor): Box energies / deltas for
                      all proposals, has shape
                      (batch_size, num_proposals, 4).
                    - obj_feat (Tensor): Object feature before classification
                      and regression subnet, has shape
                      (batch_size, num_proposal, feature_dimensions).
        """
        N, num_proposals = proposal_feat.shape[:2]
        proposal_feat = proposal_feat.permute(1, 0, 2)
        proposal_feat = self.attention_norm(self.attention(proposal_feat))
        attn_feats = proposal_feat.permute(1, 0, 2)
        proposal_feat = attn_feats.reshape(-1, self.in_channels)
        proposal_feat_iic = self.instance_interactive_conv(proposal_feat, roi_feat)
        proposal_feat = proposal_feat + self.instance_interactive_conv_dropout(proposal_feat_iic)
        obj_feat = self.instance_interactive_conv_norm(proposal_feat)
        obj_feat = self.ffn_norm(self.ffn(obj_feat))
        cls_feat = obj_feat
        reg_feat = obj_feat
        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)
        cls_score = self.fc_cls(cls_feat).view(N, num_proposals, self.num_classes if self.loss_cls.use_sigmoid else self.num_classes + 1)
        bbox_delta = self.fc_reg(reg_feat).view(N, num_proposals, 4)
        return (cls_score, bbox_delta, obj_feat.view(N, num_proposals, self.in_channels), attn_feats)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self, cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights, imgs_whwh=None, reduction_override=None, **kwargs):
        """"Loss function of DIIHead, get loss of all images.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            labels (Tensor): Label of each proposals, has shape
                (batch_size * num_proposals_single_image
            label_weights (Tensor): Classification loss
                weight of each proposals, has shape
                (batch_size * num_proposals_single_image
            bbox_targets (Tensor): Regression targets of each
                proposals, has shape
                (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents
                [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression loss weight of each
                proposals's coordinate, has shape
                (batch_size * num_proposals_single_image, 4),
            imgs_whwh (Tensor): imgs_whwh (Tensor): Tensor with                shape (batch_size, num_proposals, 4), the last
                dimension means
                [img_width,img_height, img_width, img_height].
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

            Returns:
                dict[str, Tensor]: Dictionary of loss components
        """
        losses = dict()
        bg_class_ind = self.num_classes
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos)
        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(cls_score, labels, label_weights, avg_factor=avg_factor, reduction_override=reduction_override)
                losses['pos_acc'] = accuracy(cls_score[pos_inds], labels[pos_inds])
        if bbox_pred is not None:
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                imgs_whwh = imgs_whwh.reshape(bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                losses['loss_bbox'] = self.loss_bbox(pos_bbox_pred / imgs_whwh, bbox_targets[pos_inds.type(torch.bool)] / imgs_whwh, bbox_weights[pos_inds.type(torch.bool)], avg_factor=avg_factor)
                losses['loss_iou'] = self.loss_iou(pos_bbox_pred, bbox_targets[pos_inds.type(torch.bool)], bbox_weights[pos_inds.type(torch.bool)], avg_factor=avg_factor)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                losses['loss_iou'] = bbox_pred.sum() * 0
        return losses

    def _get_target_single(self, pos_inds, neg_inds, pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Almost the same as the implementation in `bbox_head`,
        we add pos_inds and neg_inds to select positive and
        negative samples instead of selecting the first num_pos
        as positive samples.

        Args:
            pos_inds (Tensor): The length is equal to the
                positive sample numbers contain all index
                of the positive sample in the origin proposal set.
            neg_inds (Tensor): The length is equal to the
                negative sample numbers contain all index
                of the negative sample in the origin proposal set.
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all proposals, has
                  shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all proposals, has
                  shape (num_proposals, 4), the last dimension 4
                  represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all proposals,
                  has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg
        labels = pos_bboxes.new_full((num_samples,), self.num_classes, dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1
        if num_neg > 0:
            label_weights[neg_inds] = 1.0
        return (labels, label_weights, bbox_targets, bbox_weights)

    def get_targets(self, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg, concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:`ConfigDict`): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise just
                  a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals,) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list has
                  shape (num_proposals, 4) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals, 4),
                  the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(self._get_target_single, pos_inds_list, neg_inds_list, pos_bboxes_list, neg_bboxes_list, pos_gt_bboxes_list, pos_gt_labels_list, cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return (labels, label_weights, bbox_targets, bbox_weights)

@HEADS.register_module()
class DynamicMaskHead(FCNMaskHead):
    """Dynamic Mask Head for
    `Instances as Queries <http://arxiv.org/abs/2105.01928>`_

    Args:
        num_convs (int): Number of convolution layer.
            Defaults to 4.
        roi_feat_size (int): The output size of RoI extractor,
            Defaults to 14.
        in_channels (int): Input feature channels.
            Defaults to 256.
        conv_kernel_size (int): Kernel size of convolution layers.
            Defaults to 3.
        conv_out_channels (int): Output channels of convolution layers.
            Defaults to 256.
        num_classes (int): Number of classes.
            Defaults to 80
        class_agnostic (int): Whether generate class agnostic prediction.
            Defaults to False.
        dropout (float): Probability of drop the channel.
            Defaults to 0.0
        upsample_cfg (dict): The config for upsample layer.
        conv_cfg (dict): The convolution layer config.
        norm_cfg (dict): The norm layer config.
        dynamic_conv_cfg (dict): The dynamic convolution layer config.
        loss_mask (dict): The config for mask loss.
    """

    def __init__(self, num_convs=4, roi_feat_size=14, in_channels=256, conv_kernel_size=3, conv_out_channels=256, num_classes=80, class_agnostic=False, upsample_cfg=dict(type='deconv', scale_factor=2), conv_cfg=None, norm_cfg=None, dynamic_conv_cfg=dict(type='DynamicConv', in_channels=256, feat_channels=64, out_channels=256, input_feat_shape=14, with_proj=False, act_cfg=dict(type='ReLU', inplace=True), norm_cfg=dict(type='LN')), loss_mask=dict(type='DiceLoss', loss_weight=8.0), **kwargs):
        super(DynamicMaskHead, self).__init__(num_convs=num_convs, roi_feat_size=roi_feat_size, in_channels=in_channels, conv_kernel_size=conv_kernel_size, conv_out_channels=conv_out_channels, num_classes=num_classes, class_agnostic=class_agnostic, upsample_cfg=upsample_cfg, conv_cfg=conv_cfg, norm_cfg=norm_cfg, loss_mask=loss_mask, **kwargs)
        assert class_agnostic is False, 'DynamicMaskHead only support class_agnostic=False'
        self.fp16_enabled = False
        self.instance_interactive_conv = build_transformer(dynamic_conv_cfg)

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            nn.init.constant_(self.conv_logits.bias, 0.0)

    @auto_fp16()
    def forward(self, roi_feat, proposal_feat):
        """Forward function of DynamicMaskHead.

        Args:
            roi_feat (Tensor): Roi-pooling features with shape
                (batch_size*num_proposals, feature_dimensions,
                pooling_h , pooling_w).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size*num_proposals, feature_dimensions)

          Returns:
            mask_pred (Tensor): Predicted foreground masks with shape
                (batch_size*num_proposals, num_classes,
                                        pooling_h*2, pooling_w*2).
        """
        proposal_feat = proposal_feat.reshape(-1, self.in_channels)
        proposal_feat_iic = self.instance_interactive_conv(proposal_feat, roi_feat)
        x = proposal_feat_iic.permute(0, 2, 1).reshape(roi_feat.size())
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    @force_fp32(apply_to=('mask_pred',))
    def loss(self, mask_pred, mask_targets, labels):
        num_pos = labels.new_ones(labels.size()).float().sum()
        avg_factor = torch.clamp(reduce_mean(num_pos), min=1.0).item()
        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
        else:
            loss_mask = self.loss_mask(mask_pred[torch.arange(num_pos).long(), labels, ...].sigmoid(), mask_targets, avg_factor=avg_factor)
        loss['loss_mask'] = loss_mask
        return loss

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds, gt_masks, rcnn_train_cfg)
        return mask_targets

