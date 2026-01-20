# Cluster 71

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

def filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter results using score threshold and topk candidates.

    Args:
        scores (Tensor): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.
        results (dict or list or Tensor, Optional): The results to
           which the filtering rule is to be applied. The shape
           of each item is (num_bboxes, N).

    Returns:
        tuple: Filtered results

            - scores (Tensor): The scores after being filtered,                 shape (num_bboxes_filtered, ).
            - labels (Tensor): The class labels, shape                 (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): The anchor indexes, shape                 (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional):                 The filtered results. The shape of each item is                 (num_bboxes_filtered, N).
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)
    num_topk = min(topk, valid_idxs.size(0))
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)
    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, but get {type(results)}.')
    return (scores, labels, keep_idxs, filtered_results)

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

class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg=None):
        super(BaseDenseHead, self).__init__(init_cfg)

    def init_weights(self):
        super(BaseDenseHead, self).init_weights()
        for m in self.modules():
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, score_factors=None, img_metas=None, cfg=None, rescale=False, with_nms=True, **kwargs):
        """Transform network outputs of a batch into bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Default None.
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
        assert len(cls_scores) == len(bbox_preds)
        if score_factors is None:
            with_score_factors = False
        else:
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]
            results = self._get_bboxes_single(cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors, img_meta, cfg, rescale, with_nms, **kwargs)
            result_list.append(results)
        return result_list

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
        if score_factor_list[0] is None:
            with_score_factors = False
        else:
            with_score_factors = True
        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in enumerate(zip(cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]
            results = filter_scores_and_topk(scores, cfg.score_thr, nms_pre, dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results
            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            if with_score_factors:
                score_factor = score_factor[keep_idxs]
            bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)
        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes, img_meta['scale_factor'], cfg, rescale, with_nms, mlvl_score_factors, **kwargs)

    def _bbox_post_process(self, mlvl_scores, mlvl_labels, mlvl_bboxes, scale_factor, cfg, rescale=False, with_nms=True, mlvl_score_factors=None, **kwargs):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_labels (list[Tensor]): Box class labels from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
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
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)
        if mlvl_score_factors is not None:
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors
        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                return (det_bboxes, mlvl_labels)
            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores, mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
            return (det_bboxes, det_labels)
        else:
            return (mlvl_bboxes, mlvl_scores, mlvl_labels)

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=None, proposal_cfg=None, **kwargs):
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
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas=img_metas, cfg=proposal_cfg)
            return (losses, proposal_list)

    def simple_test(self, feats, img_metas, rescale=False):
        """Test function without test-time augmentation.

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
                with shape (n, ).
        """
        return self.simple_test_bboxes(feats, img_metas, rescale=rescale)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def onnx_export(self, cls_scores, bbox_preds, score_factors=None, img_metas=None, with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            score_factors (list[Tensor]): score_factors for each s
                cale level with shape (N, num_points * 1, H, W).
                Default: None.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc. Default: None.
            with_nms (bool): Whether apply nms to the bboxes. Default: True.

        Returns:
            tuple[Tensor, Tensor] | list[tuple]: When `with_nms` is True,
            it is tuple[Tensor, Tensor], first tensor bboxes with shape
            [N, num_det, 5], 5 arrange as (x1, y1, x2, y2, score)
            and second element is class labels of shape [N, num_det].
            When `with_nms` is False, first tensor is bboxes with
            shape [N, num_det, 4], second tensor is raw score has
            shape  [N, num_det, num_classes].
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)
        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]
        assert len(img_metas) == 1, 'Only support one input image while in exporting to ONNX'
        img_shape = img_metas[0]['img_shape_for_onnx']
        cfg = self.test_cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_priors)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        nms_pre_tensor = torch.tensor(cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        if score_factors is None:
            with_score_factors = False
            mlvl_score_factor = [None for _ in range(num_levels)]
        else:
            with_score_factors = True
            mlvl_score_factor = [score_factors[i].detach() for i in range(num_levels)]
            mlvl_score_factors = []
        mlvl_batch_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, score_factors, priors in zip(mlvl_cls_scores, mlvl_bbox_preds, mlvl_score_factor, mlvl_priors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = scores.sigmoid()
                nms_pre_score = scores
            else:
                scores = scores.softmax(-1)
                nms_pre_score = scores
            if with_score_factors:
                score_factors = score_factors.permute(0, 2, 3, 1).reshape(batch_size, -1).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            priors = priors.expand(batch_size, -1, priors.size(-1))
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                if with_score_factors:
                    nms_pre_score = nms_pre_score * score_factors[..., None]
                else:
                    nms_pre_score = nms_pre_score
                if self.use_sigmoid_cls:
                    max_scores, _ = nms_pre_score.max(-1)
                else:
                    max_scores, _ = nms_pre_score[..., :-1].max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size, device=bbox_pred.device).view(-1, 1).expand_as(topk_inds).long()
                transformed_inds = bbox_pred.shape[1] * batch_inds + topk_inds
                priors = priors.reshape(-1, priors.size(-1))[transformed_inds, :].reshape(batch_size, -1, priors.size(-1))
                bbox_pred = bbox_pred.reshape(-1, 4)[transformed_inds, :].reshape(batch_size, -1, 4)
                scores = scores.reshape(-1, self.cls_out_channels)[transformed_inds, :].reshape(batch_size, -1, self.cls_out_channels)
                if with_score_factors:
                    score_factors = score_factors.reshape(-1, 1)[transformed_inds].reshape(batch_size, -1)
            bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)
            mlvl_batch_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            if with_score_factors:
                mlvl_score_factors.append(score_factors)
        batch_bboxes = torch.cat(mlvl_batch_bboxes, dim=1)
        batch_scores = torch.cat(mlvl_scores, dim=1)
        if with_score_factors:
            batch_score_factors = torch.cat(mlvl_score_factors, dim=1)
        from mmdet.core.export import add_dummy_nms_for_onnx
        if not self.use_sigmoid_cls:
            batch_scores = batch_scores[..., :self.num_classes]
        if with_score_factors:
            batch_scores = batch_scores * batch_score_factors.unsqueeze(2)
        if with_nms:
            max_output_boxes_per_class = cfg.nms.get('max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_bboxes, batch_scores, max_output_boxes_per_class, iou_threshold, score_threshold, nms_pre, cfg.max_per_img)
        else:
            return (batch_bboxes, batch_scores)

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

def test_filter_scores_and_topk():
    score = torch.tensor([[0.1, 0.3, 0.2], [0.12, 0.7, 0.9], [0.02, 0.8, 0.08], [0.4, 0.1, 0.08]])
    bbox_pred = torch.tensor([[0.2, 0.3], [0.4, 0.7], [0.1, 0.1], [0.5, 0.1]])
    score_thr = 0.15
    nms_pre = 4
    with pytest.raises(NotImplementedError):
        filter_scores_and_topk(score, score_thr, nms_pre, (score,))
    filtered_results = filter_scores_and_topk(score, score_thr, nms_pre, results=dict(bbox_pred=bbox_pred))
    filtered_score, labels, keep_idxs, results = filtered_results
    assert filtered_score.allclose(torch.tensor([0.9, 0.8, 0.7, 0.4]))
    assert labels.allclose(torch.tensor([2, 1, 1, 0]))
    assert keep_idxs.allclose(torch.tensor([1, 2, 1, 3]))
    assert results['bbox_pred'].allclose(torch.tensor([[0.4, 0.7], [0.1, 0.1], [0.4, 0.7], [0.5, 0.1]]))

