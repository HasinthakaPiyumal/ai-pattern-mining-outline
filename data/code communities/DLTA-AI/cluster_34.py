# Cluster 34

@BBOX_ASSIGNERS.register_module()
class UniformAssigner(BaseAssigner):
    """Uniform Matching between the anchors and gt boxes, which can achieve
    balance in positive anchors, and gt_bboxes_ignore was not considered for
    now.

    Args:
        pos_ignore_thr (float): the threshold to ignore positive anchors
        neg_ignore_thr (float): the threshold to ignore negative anchors
        match_times(int): Number of positive anchors for each gt box.
           Default 4.
        iou_calculator (dict): iou_calculator config
    """

    def __init__(self, pos_ignore_thr, neg_ignore_thr, match_times=4, iou_calculator=dict(type='BboxOverlaps2D')):
        self.match_times = match_times
        self.pos_ignore_thr = pos_ignore_thr
        self.neg_ignore_thr = neg_ignore_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, bbox_pred, anchor, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        num_gts, num_bboxes = (gt_bboxes.size(0), bbox_pred.size(0))
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), 0, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            assign_result = AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
            assign_result.set_extra_property('pos_idx', bbox_pred.new_empty(0, dtype=torch.bool))
            assign_result.set_extra_property('pos_predicted_boxes', bbox_pred.new_empty((0, 4)))
            assign_result.set_extra_property('target_boxes', bbox_pred.new_empty((0, 4)))
            return assign_result
        cost_bbox = torch.cdist(bbox_xyxy_to_cxcywh(bbox_pred), bbox_xyxy_to_cxcywh(gt_bboxes), p=1)
        cost_bbox_anchors = torch.cdist(bbox_xyxy_to_cxcywh(anchor), bbox_xyxy_to_cxcywh(gt_bboxes), p=1)
        C = cost_bbox.cpu()
        C1 = cost_bbox_anchors.cpu()
        index = torch.topk(C, k=self.match_times, dim=0, largest=False)[1]
        index1 = torch.topk(C1, k=self.match_times, dim=0, largest=False)[1]
        indexes = torch.cat((index, index1), dim=1).reshape(-1).to(bbox_pred.device)
        pred_overlaps = self.iou_calculator(bbox_pred, gt_bboxes)
        anchor_overlaps = self.iou_calculator(anchor, gt_bboxes)
        pred_max_overlaps, _ = pred_overlaps.max(dim=1)
        anchor_max_overlaps, _ = anchor_overlaps.max(dim=0)
        ignore_idx = pred_max_overlaps > self.neg_ignore_thr
        assigned_gt_inds[ignore_idx] = -1
        pos_gt_index = torch.arange(0, C1.size(1), device=bbox_pred.device).repeat(self.match_times * 2)
        pos_ious = anchor_overlaps[indexes, pos_gt_index]
        pos_ignore_idx = pos_ious < self.pos_ignore_thr
        pos_gt_index_with_ignore = pos_gt_index + 1
        pos_gt_index_with_ignore[pos_ignore_idx] = -1
        assigned_gt_inds[indexes] = pos_gt_index_with_ignore
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        assign_result = AssignResult(num_gts, assigned_gt_inds, anchor_max_overlaps, labels=assigned_labels)
        assign_result.set_extra_property('pos_idx', ~pos_ignore_idx)
        assign_result.set_extra_property('pos_predicted_boxes', bbox_pred[indexes])
        assign_result.set_extra_property('target_boxes', gt_bboxes[pos_gt_index])
        return assign_result

def bbox_xyxy_to_cxcywh(bbox):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    return torch.cat(bbox_new, dim=-1)

@BBOX_ASSIGNERS.register_module()
class HungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self, cls_cost=dict(type='ClassificationCost', weight=1.0), reg_cost=dict(type='BBoxL1Cost', weight=1.0), iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0)):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)

    def assign(self, bbox_pred, cls_pred, gt_bboxes, gt_labels, img_meta, gt_bboxes_ignore=None, eps=1e-07):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, 'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = (gt_bboxes.size(0), bbox_pred.size(0))
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
        img_h, img_w, _ = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        normalize_gt_bboxes = gt_bboxes / factor
        reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        iou_cost = self.iou_cost(bboxes, gt_bboxes)
        cost = cls_cost + reg_cost + iou_cost
        cost = cost.detach().cpu()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)
        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]
    return torch.cat(bbox_new, dim=-1)

@MATCH_COST.register_module()
class BBoxL1Cost:
    """BBoxL1Cost.

     Args:
         weight (int | float, optional): loss_weight
         box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost()
         >>> bbox_pred = torch.rand(1, 4)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self, weight=1.0, box_format='xyxy'):
        self.weight = weight
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                (num_query, 4).
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        if self.box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        elif self.box_format == 'xyxy':
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

@HEADS.register_module()
class EmbeddingRPNHead(BaseModule):
    """RPNHead in the `Sparse R-CNN <https://arxiv.org/abs/2011.12450>`_ .

    Unlike traditional RPNHead, this module does not need FPN input, but just
    decode `init_proposal_bboxes` and expand the first dimension of
    `init_proposal_bboxes` and `init_proposal_features` to the batch_size.

    Args:
        num_proposals (int): Number of init_proposals. Default 100.
        proposal_feature_channel (int): Channel number of
            init_proposal_feature. Defaults to 256.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self, num_proposals=100, proposal_feature_channel=256, init_cfg=None, **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization behavior, init_cfg is not allowed to be set'
        super(EmbeddingRPNHead, self).__init__(init_cfg)
        self.num_proposals = num_proposals
        self.proposal_feature_channel = proposal_feature_channel
        self._init_layers()

    def _init_layers(self):
        """Initialize a sparse set of proposal boxes and proposal features."""
        self.init_proposal_bboxes = nn.Embedding(self.num_proposals, 4)
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.proposal_feature_channel)

    def init_weights(self):
        """Initialize the init_proposal_bboxes as normalized.

        [c_x, c_y, w, h], and we initialize it to the size of  the entire
        image.
        """
        super(EmbeddingRPNHead, self).init_weights()
        nn.init.constant_(self.init_proposal_bboxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_bboxes.weight[:, 2:], 1)

    def _decode_init_proposals(self, imgs, img_metas):
        """Decode init_proposal_bboxes according to the size of images and
        expand dimension of init_proposal_features to batch_size.

        Args:
            imgs (list[Tensor]): List of FPN features.
            img_metas (list[dict]): List of meta-information of
                images. Need the img_shape to decode the init_proposals.

        Returns:
            Tuple(Tensor):

                - proposals (Tensor): Decoded proposal bboxes,
                  has shape (batch_size, num_proposals, 4).
                - init_proposal_features (Tensor): Expanded proposal
                  features, has shape
                  (batch_size, num_proposals, proposal_feature_channel).
                - imgs_whwh (Tensor): Tensor with shape
                  (batch_size, 4), the dimension means
                  [img_width, img_height, img_width, img_height].
        """
        proposals = self.init_proposal_bboxes.weight.clone()
        proposals = bbox_cxcywh_to_xyxy(proposals)
        num_imgs = len(imgs[0])
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(imgs[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]
        proposals = proposals * imgs_whwh
        init_proposal_features = self.init_proposal_features.weight.clone()
        init_proposal_features = init_proposal_features[None].expand(num_imgs, *init_proposal_features.size())
        return (proposals, init_proposal_features, imgs_whwh)

    def forward_dummy(self, img, img_metas):
        """Dummy forward function.

        Used in flops calculation.
        """
        return self._decode_init_proposals(img, img_metas)

    def forward_train(self, img, img_metas):
        """Forward function in training stage."""
        return self._decode_init_proposals(img, img_metas)

    def simple_test_rpn(self, img, img_metas):
        """Forward function in testing stage."""
        return self._decode_init_proposals(img, img_metas)

    def simple_test(self, img, img_metas):
        """Forward function in testing stage."""
        raise NotImplementedError

    def aug_test_rpn(self, feats, img_metas):
        raise NotImplementedError('EmbeddingRPNHead does not support test-time augmentation')

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

def dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape):
    """Clip boxes dynamically for onnx.

    Since torch.clamp cannot have dynamic `min` and `max`, we scale the
      boxes by 1/max_shape and clamp in the range [0, 1].

    Args:
        x1 (Tensor): The x1 for bounding boxes.
        y1 (Tensor): The y1 for bounding boxes.
        x2 (Tensor): The x2 for bounding boxes.
        y2 (Tensor): The y2 for bounding boxes.
        max_shape (Tensor or torch.Size): The (H,W) of original image.
    Returns:
        tuple(Tensor): The clipped x1, y1, x2, y2.
    """
    assert isinstance(max_shape, torch.Tensor), '`max_shape` should be tensor of (h,w) for onnx'
    x1 = x1 / max_shape[1]
    y1 = y1 / max_shape[0]
    x2 = x2 / max_shape[1]
    y2 = y2 / max_shape[0]
    x1 = torch.clamp(x1, 0, 1)
    y1 = torch.clamp(y1, 0, 1)
    x2 = torch.clamp(x2, 0, 1)
    y2 = torch.clamp(y2, 0, 1)
    x1 = x1 * max_shape[1]
    y1 = y1 * max_shape[0]
    x2 = x2 * max_shape[1]
    y2 = y2 * max_shape[0]
    return (x1, y1, x2, y2)

@HEADS.register_module()
class SparseRoIHead(CascadeRoIHead):
    """The RoIHead for `Sparse R-CNN: End-to-End Object Detection with
    Learnable Proposals <https://arxiv.org/abs/2011.12450>`_
    and `Instances as Queries <http://arxiv.org/abs/2105.01928>`_

    Args:
        num_stages (int): Number of stage whole iterative process.
            Defaults to 6.
        stage_loss_weights (Tuple[float]): The loss
            weight of each stage. By default all stages have
            the same weight 1.
        bbox_roi_extractor (dict): Config of box roi extractor.
        mask_roi_extractor (dict): Config of mask roi extractor.
        bbox_head (dict): Config of box head.
        mask_head (dict): Config of mask head.
        train_cfg (dict, optional): Configuration information in train stage.
            Defaults to None.
        test_cfg (dict, optional): Configuration information in test stage.
            Defaults to None.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    """

    def __init__(self, num_stages=6, stage_loss_weights=(1, 1, 1, 1, 1, 1), proposal_feature_channel=256, bbox_roi_extractor=dict(type='SingleRoIExtractor', roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2), out_channels=256, featmap_strides=[4, 8, 16, 32]), mask_roi_extractor=None, bbox_head=dict(type='DIIHead', num_classes=80, num_fcs=2, num_heads=8, num_cls_fcs=1, num_reg_fcs=3, feedforward_channels=2048, hidden_channels=256, dropout=0.0, roi_feat_size=7, ffn_act_cfg=dict(type='ReLU', inplace=True)), mask_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.proposal_feature_channel = proposal_feature_channel
        super(SparseRoIHead, self).__init__(num_stages, stage_loss_weights, bbox_roi_extractor=bbox_roi_extractor, mask_roi_extractor=mask_roi_extractor, bbox_head=bbox_head, mask_head=mask_head, train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained, init_cfg=init_cfg)
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler), 'Sparse R-CNN and QueryInst only support `PseudoSampler`'

    def _bbox_forward(self, stage, x, rois, object_feats, img_metas):
        """Box head forward function used in both training and testing. Returns
        all regression, classification results and a intermediate feature.

        Args:
            stage (int): The index of current stage in
                iterative process.
            x (List[Tensor]): List of FPN features
            rois (Tensor): Rois in total batch. With shape (num_proposal, 5).
                the last dimension 5 represents (img_index, x1, y1, x2, y2).
            object_feats (Tensor): The object feature extracted from
                the previous stage.
            img_metas (dict): meta information of images.

        Returns:
            dict[str, Tensor]: a dictionary of bbox head outputs,
                Containing the following results:

                    - cls_score (Tensor): The score of each class, has
                      shape (batch_size, num_proposals, num_classes)
                      when use focal loss or
                      (batch_size, num_proposals, num_classes+1)
                      otherwise.
                    - decode_bbox_pred (Tensor): The regression results
                      with shape (batch_size, num_proposal, 4).
                      The last dimension 4 represents
                      [tl_x, tl_y, br_x, br_y].
                    - object_feats (Tensor): The object feature extracted
                      from current stage
                    - detach_cls_score_list (list[Tensor]): The detached
                      classification results, length is batch_size, and
                      each tensor has shape (num_proposal, num_classes).
                    - detach_proposal_list (list[tensor]): The detached
                      regression results, length is batch_size, and each
                      tensor has shape (num_proposal, 4). The last
                      dimension 4 represents [tl_x, tl_y, br_x, br_y].
        """
        num_imgs = len(img_metas)
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred, object_feats, attn_feats = bbox_head(bbox_feats, object_feats)
        proposal_list = self.bbox_head[stage].refine_bboxes(rois, rois.new_zeros(len(rois)), bbox_pred.view(-1, bbox_pred.size(-1)), [rois.new_zeros(object_feats.size(1)) for _ in range(num_imgs)], img_metas)
        bbox_results = dict(cls_score=cls_score, decode_bbox_pred=torch.cat(proposal_list), object_feats=object_feats, attn_feats=attn_feats, detach_cls_score_list=[cls_score[i].detach() for i in range(num_imgs)], detach_proposal_list=[item.detach() for item in proposal_list])
        return bbox_results

    def _mask_forward(self, stage, x, rois, attn_feats):
        """Mask head forward function used in both training and testing."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs], rois)
        mask_pred = mask_head(mask_feats, attn_feats)
        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _mask_forward_train(self, stage, x, attn_feats, sampling_results, gt_masks, rcnn_train_cfg):
        """Run forward function and calculate loss for mask head in
        training."""
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        attn_feats = torch.cat([feats[res.pos_inds] for feats, res in zip(attn_feats, sampling_results)])
        mask_results = self._mask_forward(stage, x, pos_rois, attn_feats)
        mask_targets = self.mask_head[stage].get_targets(sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'], mask_targets, pos_labels)
        mask_results.update(loss_mask)
        return mask_results

    def forward_train(self, x, proposal_boxes, proposal_features, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, imgs_whwh=None, gt_masks=None):
        """Forward function in training stage.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposals (Tensor): Decoded proposal bboxes, has shape
                (batch_size, num_proposals, 4)
            proposal_features (Tensor): Expanded proposal
                features, has shape
                (batch_size, num_proposals, proposal_feature_channel)
            img_metas (list[dict]): list of image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape',
                'pad_shape', and 'img_norm_cfg'. For details on the
                values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                    the dimension means
                    [img_width,img_height, img_width, img_height].
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components of all stage.
        """
        num_imgs = len(img_metas)
        num_proposals = proposal_boxes.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features
        all_stage_loss = {}
        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats, img_metas)
            all_stage_bbox_results.append(bbox_results)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            cls_pred_list = bbox_results['detach_cls_score_list']
            proposal_list = bbox_results['detach_proposal_list']
            for i in range(num_imgs):
                normalize_bbox_ccwh = bbox_xyxy_to_cxcywh(proposal_list[i] / imgs_whwh[i])
                assign_result = self.bbox_assigner[stage].assign(normalize_bbox_ccwh, cls_pred_list[i], gt_bboxes[i], gt_labels[i], img_metas[i])
                sampling_result = self.bbox_sampler[stage].sample(assign_result, proposal_list[i], gt_bboxes[i])
                sampling_results.append(sampling_result)
            bbox_targets = self.bbox_head[stage].get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg[stage], True)
            cls_score = bbox_results['cls_score']
            decode_bbox_pred = bbox_results['decode_bbox_pred']
            single_stage_loss = self.bbox_head[stage].loss(cls_score.view(-1, cls_score.size(-1)), decode_bbox_pred.view(-1, 4), *bbox_targets, imgs_whwh=imgs_whwh)
            if self.with_mask:
                mask_results = self._mask_forward_train(stage, x, bbox_results['attn_feats'], sampling_results, gt_masks, self.train_cfg[stage])
                single_stage_loss['loss_mask'] = mask_results['loss_mask']
            for key, value in single_stage_loss.items():
                all_stage_loss[f'stage{stage}_{key}'] = value * self.stage_loss_weights[stage]
            object_feats = bbox_results['object_feats']
        return all_stage_loss

    def simple_test(self, x, proposal_boxes, proposal_features, img_metas, imgs_whwh, rescale=False):
        """Test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_boxes (Tensor): Decoded proposal bboxes, has shape
                (batch_size, num_proposals, 4)
            proposal_features (Tensor): Expanded proposal
                features, has shape
                (batch_size, num_proposals, proposal_feature_channel)
            img_metas (dict): meta information of images.
            imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                    the dimension means
                    [img_width,img_height, img_width, img_height].
            rescale (bool): If True, return boxes in original image
                space. Defaults to False.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has a mask branch,
            it is a list[tuple] that contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(img_metas)
        proposal_list = [proposal_boxes[i] for i in range(num_imgs)]
        ori_shapes = tuple((meta['ori_shape'] for meta in img_metas))
        scale_factors = tuple((meta['scale_factor'] for meta in img_metas))
        object_feats = proposal_features
        if all([proposal.shape[0] == 0 for proposal in proposal_list]):
            bbox_results = [[np.zeros((0, 5), dtype=np.float32) for i in range(self.bbox_head[-1].num_classes)]] * num_imgs
            return bbox_results
        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats, img_metas)
            object_feats = bbox_results['object_feats']
            cls_score = bbox_results['cls_score']
            proposal_list = bbox_results['detach_proposal_list']
        if self.with_mask:
            rois = bbox2roi(proposal_list)
            mask_results = self._mask_forward(stage, x, rois, bbox_results['attn_feats'])
            mask_results['mask_pred'] = mask_results['mask_pred'].reshape(num_imgs, -1, *mask_results['mask_pred'].size()[1:])
        num_classes = self.bbox_head[-1].num_classes
        det_bboxes = []
        det_labels = []
        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]
        for img_id in range(num_imgs):
            cls_score_per_img = cls_score[img_id]
            scores_per_img, topk_indices = cls_score_per_img.flatten(0, 1).topk(self.test_cfg.max_per_img, sorted=False)
            labels_per_img = topk_indices % num_classes
            bbox_pred_per_img = proposal_list[img_id][topk_indices // num_classes]
            if rescale:
                scale_factor = img_metas[img_id]['scale_factor']
                bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
            det_bboxes.append(torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))
            det_labels.append(labels_per_img)
        bbox_results = [bbox2result(det_bboxes[i], det_labels[i], num_classes) for i in range(num_imgs)]
        if self.with_mask:
            if rescale and (not isinstance(scale_factors[0], float)):
                scale_factors = [torch.from_numpy(scale_factor).to(det_bboxes[0].device) for scale_factor in scale_factors]
            _bboxes = [det_bboxes[i][:, :4] * scale_factors[i] if rescale else det_bboxes[i][:, :4] for i in range(len(det_bboxes))]
            segm_results = []
            mask_pred = mask_results['mask_pred']
            for img_id in range(num_imgs):
                mask_pred_per_img = mask_pred[img_id].flatten(0, 1)[topk_indices]
                mask_pred_per_img = mask_pred_per_img[:, None, ...].repeat(1, num_classes, 1, 1)
                segm_result = self.mask_head[-1].get_seg_masks(mask_pred_per_img, _bboxes[img_id], det_labels[img_id], self.test_cfg, ori_shapes[img_id], scale_factors[img_id], rescale)
                segm_results.append(segm_result)
        if self.with_mask:
            results = list(zip(bbox_results, segm_results))
        else:
            results = bbox_results
        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        raise NotImplementedError('Sparse R-CNN and QueryInst does not support `aug_test`')

    def forward_dummy(self, x, proposal_boxes, proposal_features, img_metas):
        """Dummy forward function when do the flops computing."""
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features
        if self.with_bbox:
            for stage in range(self.num_stages):
                rois = bbox2roi(proposal_list)
                bbox_results = self._bbox_forward(stage, x, rois, object_feats, img_metas)
                all_stage_bbox_results.append((bbox_results,))
                proposal_list = bbox_results['detach_proposal_list']
                object_feats = bbox_results['object_feats']
                if self.with_mask:
                    rois = bbox2roi(proposal_list)
                    mask_results = self._mask_forward(stage, x, rois, bbox_results['attn_feats'])
                    all_stage_bbox_results[-1] += (mask_results,)
        return all_stage_bbox_results

class BaseAnchorOptimizer:
    """Base class for anchor optimizer.

    Args:
        dataset (obj:`Dataset`): Dataset object.
        input_shape (list[int]): Input image shape of the model.
            Format in [width, height].
        logger (obj:`logging.Logger`): The logger for logging.
        device (str, optional): Device used for calculating.
            Default: 'cuda:0'
        out_dir (str, optional): Path to save anchor optimize result.
            Default: None
    """

    def __init__(self, dataset, input_shape, logger, device='cuda:0', out_dir=None):
        self.dataset = dataset
        self.input_shape = input_shape
        self.logger = logger
        self.device = device
        self.out_dir = out_dir
        bbox_whs, img_shapes = self.get_whs_and_shapes()
        ratios = img_shapes.max(1, keepdims=True) / np.array([input_shape])
        self.bbox_whs = bbox_whs / ratios

    def get_whs_and_shapes(self):
        """Get widths and heights of bboxes and shapes of images.

        Returns:
            tuple[np.ndarray]: Array of bbox shapes and array of image
            shapes with shape (num_bboxes, 2) in [width, height] format.
        """
        self.logger.info('Collecting bboxes from annotation...')
        bbox_whs = []
        img_shapes = []
        prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(len(self.dataset)):
            ann = self.dataset.get_ann_info(idx)
            data_info = self.dataset.data_infos[idx]
            img_shape = np.array([data_info['width'], data_info['height']])
            gt_bboxes = ann['bboxes']
            for bbox in gt_bboxes:
                wh = bbox[2:4] - bbox[0:2]
                img_shapes.append(img_shape)
                bbox_whs.append(wh)
            prog_bar.update()
        print('\n')
        bbox_whs = np.array(bbox_whs)
        img_shapes = np.array(img_shapes)
        self.logger.info(f'Collected {bbox_whs.shape[0]} bboxes.')
        return (bbox_whs, img_shapes)

    def get_zero_center_bbox_tensor(self):
        """Get a tensor of bboxes centered at (0, 0).

        Returns:
            Tensor: Tensor of bboxes with shape (num_bboxes, 4)
            in [xmin, ymin, xmax, ymax] format.
        """
        whs = torch.from_numpy(self.bbox_whs).to(self.device, dtype=torch.float32)
        bboxes = bbox_cxcywh_to_xyxy(torch.cat([torch.zeros_like(whs), whs], dim=1))
        return bboxes

    def optimize(self):
        raise NotImplementedError

    def save_result(self, anchors, path=None):
        anchor_results = []
        for w, h in anchors:
            anchor_results.append([round(w), round(h)])
        self.logger.info(f'Anchor optimize result:{anchor_results}')
        if path:
            json_path = osp.join(path, 'anchor_optimize_result.json')
            mmcv.dump(anchor_results, json_path)
            self.logger.info(f'Result saved in {json_path}')

class YOLOKMeansAnchorOptimizer(BaseAnchorOptimizer):
    """YOLO anchor optimizer using k-means. Code refer to `AlexeyAB/darknet.
    <https://github.com/AlexeyAB/darknet/blob/master/src/detector.c>`_.

    Args:
        num_anchors (int) : Number of anchors.
        iters (int): Maximum iterations for k-means.
    """

    def __init__(self, num_anchors, iters, **kwargs):
        super(YOLOKMeansAnchorOptimizer, self).__init__(**kwargs)
        self.num_anchors = num_anchors
        self.iters = iters

    def optimize(self):
        anchors = self.kmeans_anchors()
        self.save_result(anchors, self.out_dir)

    def kmeans_anchors(self):
        self.logger.info(f'Start cluster {self.num_anchors} YOLO anchors with K-means...')
        bboxes = self.get_zero_center_bbox_tensor()
        cluster_center_idx = torch.randint(0, bboxes.shape[0], (self.num_anchors,)).to(self.device)
        assignments = torch.zeros((bboxes.shape[0],)).to(self.device)
        cluster_centers = bboxes[cluster_center_idx]
        if self.num_anchors == 1:
            cluster_centers = self.kmeans_maximization(bboxes, assignments, cluster_centers)
            anchors = bbox_xyxy_to_cxcywh(cluster_centers)[:, 2:].cpu().numpy()
            anchors = sorted(anchors, key=lambda x: x[0] * x[1])
            return anchors
        prog_bar = mmcv.ProgressBar(self.iters)
        for i in range(self.iters):
            converged, assignments = self.kmeans_expectation(bboxes, assignments, cluster_centers)
            if converged:
                self.logger.info(f'K-means process has converged at iter {i}.')
                break
            cluster_centers = self.kmeans_maximization(bboxes, assignments, cluster_centers)
            prog_bar.update()
        print('\n')
        avg_iou = bbox_overlaps(bboxes, cluster_centers).max(1)[0].mean().item()
        anchors = bbox_xyxy_to_cxcywh(cluster_centers)[:, 2:].cpu().numpy()
        anchors = sorted(anchors, key=lambda x: x[0] * x[1])
        self.logger.info(f'Anchor cluster finish. Average IOU: {avg_iou}')
        return anchors

    def kmeans_maximization(self, bboxes, assignments, centers):
        """Maximization part of EM algorithm(Expectation-Maximization)"""
        new_centers = torch.zeros_like(centers)
        for i in range(centers.shape[0]):
            mask = assignments == i
            if mask.sum():
                new_centers[i, :] = bboxes[mask].mean(0)
        return new_centers

    def kmeans_expectation(self, bboxes, assignments, centers):
        """Expectation part of EM algorithm(Expectation-Maximization)"""
        ious = bbox_overlaps(bboxes, centers)
        closest = ious.argmax(1)
        converged = (closest == assignments).all()
        return (converged, closest)

class YOLODEAnchorOptimizer(BaseAnchorOptimizer):
    """YOLO anchor optimizer using differential evolution algorithm.

    Args:
        num_anchors (int) : Number of anchors.
        iters (int): Maximum iterations for k-means.
        strategy (str): The differential evolution strategy to use.
            Should be one of:

                - 'best1bin'
                - 'best1exp'
                - 'rand1exp'
                - 'randtobest1exp'
                - 'currenttobest1exp'
                - 'best2exp'
                - 'rand2exp'
                - 'randtobest1bin'
                - 'currenttobest1bin'
                - 'best2bin'
                - 'rand2bin'
                - 'rand1bin'

            Default: 'best1bin'.
        population_size (int): Total population size of evolution algorithm.
            Default: 15.
        convergence_thr (float): Tolerance for convergence, the
            optimizing stops when ``np.std(pop) <= abs(convergence_thr)
            + convergence_thr * np.abs(np.mean(population_energies))``,
            respectively. Default: 0.0001.
        mutation (tuple[float]): Range of dithering randomly changes the
            mutation constant. Default: (0.5, 1).
        recombination (float): Recombination constant of crossover probability.
            Default: 0.7.
    """

    def __init__(self, num_anchors, iters, strategy='best1bin', population_size=15, convergence_thr=0.0001, mutation=(0.5, 1), recombination=0.7, **kwargs):
        super(YOLODEAnchorOptimizer, self).__init__(**kwargs)
        self.num_anchors = num_anchors
        self.iters = iters
        self.strategy = strategy
        self.population_size = population_size
        self.convergence_thr = convergence_thr
        self.mutation = mutation
        self.recombination = recombination

    def optimize(self):
        anchors = self.differential_evolution()
        self.save_result(anchors, self.out_dir)

    def differential_evolution(self):
        bboxes = self.get_zero_center_bbox_tensor()
        bounds = []
        for i in range(self.num_anchors):
            bounds.extend([(0, self.input_shape[0]), (0, self.input_shape[1])])
        result = differential_evolution(func=self.avg_iou_cost, bounds=bounds, args=(bboxes,), strategy=self.strategy, maxiter=self.iters, popsize=self.population_size, tol=self.convergence_thr, mutation=self.mutation, recombination=self.recombination, updating='immediate', disp=True)
        self.logger.info(f'Anchor evolution finish. Average IOU: {1 - result.fun}')
        anchors = [(w, h) for w, h in zip(result.x[::2], result.x[1::2])]
        anchors = sorted(anchors, key=lambda x: x[0] * x[1])
        return anchors

    @staticmethod
    def avg_iou_cost(anchor_params, bboxes):
        assert len(anchor_params) % 2 == 0
        anchor_whs = torch.tensor([[w, h] for w, h in zip(anchor_params[::2], anchor_params[1::2])]).to(bboxes.device, dtype=bboxes.dtype)
        anchor_boxes = bbox_cxcywh_to_xyxy(torch.cat([torch.zeros_like(anchor_whs), anchor_whs], dim=1))
        ious = bbox_overlaps(bboxes, anchor_boxes)
        max_ious, _ = ious.max(1)
        cost = 1 - max_ious.mean().item()
        return cost

