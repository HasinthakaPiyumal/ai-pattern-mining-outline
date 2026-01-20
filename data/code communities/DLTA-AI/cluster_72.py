# Cluster 72

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

def select_single_mlvl(mlvl_tensors, batch_id, detach=True):
    """Extract a multi-scale single image tensor from a multi-scale batch
    tensor based on batch index.

    Note: The default value of detach is True, because the proposal gradient
    needs to be detached during the training of the two-stage model. E.g
    Cascade Mask R-CNN.

    Args:
        mlvl_tensors (list[Tensor]): Batch tensor for all scale levels,
           each is a 4D-tensor.
        batch_id (int): Batch index.
        detach (bool): Whether detach gradient. Default True.

    Returns:
        list[Tensor]: Multi-scale single image tensor.
    """
    assert isinstance(mlvl_tensors, (list, tuple))
    num_levels = len(mlvl_tensors)
    if detach:
        mlvl_tensor_list = [mlvl_tensors[i][batch_id].detach() for i in range(num_levels)]
    else:
        mlvl_tensor_list = [mlvl_tensors[i][batch_id] for i in range(num_levels)]
    return mlvl_tensor_list

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

def test_select_single_mlvl():
    mlvl_tensors = [torch.rand(2, 1, 10, 10)] * 5
    mlvl_tensor_list = select_single_mlvl(mlvl_tensors, 1)
    assert len(mlvl_tensor_list) == 5 and mlvl_tensor_list[0].ndim == 3

