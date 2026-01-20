# Cluster 48

@LOSSES.register_module()
class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    `Gradient Harmonized Single-stage Detector
    <https://arxiv.org/abs/1811.05181>`_.

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
        reduction (str): Options are "none", "mean" and "sum".
            Defaults to "mean"
    """

    def __init__(self, bins=10, momentum=0, use_sigmoid=True, loss_weight=1.0, reduction='mean'):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-06
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, label_weight, reduction_override=None, **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            The gradient harmonized loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if pred.dim() != target.dim():
            target, label_weight = _expand_onehot_labels(target, label_weight, pred.size(-1))
        target, label_weight = (target.float(), label_weight.float())
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)
        g = torch.abs(pred.sigmoid().detach() - target)
        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        loss = weight_reduce_loss(loss, weights, reduction=reduction, avg_factor=tot)
        return loss * self.loss_weight

def _expand_onehot_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero((labels >= 0) & (labels < label_channels), as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0), label_channels)
    return (bin_labels, bin_label_weights)

@mmcv.jit(derivate=True, coderize=True)
def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        eps = torch.finfo(torch.float32).eps
        loss = loss.sum() / (avg_factor + eps)
    elif reduction != 'none':
        raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

@functools.wraps(loss_func)
def wrapper(pred, target, weight=None, reduction='mean', avg_factor=None, **kwargs):
    loss = loss_func(pred, target, **kwargs)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

def seesaw_ce_loss(cls_score, labels, label_weights, cum_samples, num_classes, p, q, eps, reduction='mean', avg_factor=None):
    """Calculate the Seesaw CrossEntropy loss.

    Args:
        cls_score (torch.Tensor): The prediction with shape (N, C),
             C is the number of classes.
        labels (torch.Tensor): The learning label of the prediction.
        label_weights (torch.Tensor): Sample-wise loss weight.
        cum_samples (torch.Tensor): Cumulative samples for each category.
        num_classes (int): The number of classes.
        p (float): The ``p`` in the mitigation factor.
        q (float): The ``q`` in the compenstation factor.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert cls_score.size(-1) == num_classes
    assert len(cum_samples) == num_classes
    onehot_labels = F.one_hot(labels, num_classes)
    seesaw_weights = cls_score.new_ones(onehot_labels.size())
    if p > 0:
        sample_ratio_matrix = cum_samples[None, :].clamp(min=1) / cum_samples[:, None].clamp(min=1)
        index = (sample_ratio_matrix < 1.0).float()
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index)
        mitigation_factor = sample_weights[labels.long(), :]
        seesaw_weights = seesaw_weights * mitigation_factor
    if q > 0:
        scores = F.softmax(cls_score.detach(), dim=1)
        self_scores = scores[torch.arange(0, len(scores)).to(scores.device).long(), labels.long()]
        score_matrix = scores / self_scores[:, None].clamp(min=eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)
        seesaw_weights = seesaw_weights * compensation_factor
    cls_score = cls_score + seesaw_weights.log() * (1 - onehot_labels)
    loss = F.cross_entropy(cls_score, labels, weight=None, reduction='none')
    if label_weights is not None:
        label_weights = label_weights.float()
    loss = weight_reduce_loss(loss, weight=label_weights, reduction=reduction, avg_factor=avg_factor)
    return loss

def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None, ignore_index=-100, avg_non_ignore=False):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        torch.Tensor: The calculated loss
    """
    ignore_index = -100 if ignore_index is None else ignore_index
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none', ignore_index=ignore_index)
    if avg_factor is None and avg_non_ignore and (reduction == 'mean'):
        avg_factor = label.numel() - (label == ignore_index).sum().item()
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss

def binary_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None, ignore_index=-100, avg_non_ignore=False):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1) or (N, ).
            When the shape of pred is (N, 1), label will be expanded to
            one-hot format, and when the shape of pred is (N, ), label
            will not be expanded to one-hot format.
        label (torch.Tensor): The learning label of the prediction,
            with shape (N, ).
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        torch.Tensor: The calculated loss.
    """
    ignore_index = -100 if ignore_index is None else ignore_index
    if pred.dim() != label.dim():
        label, weight, valid_mask = _expand_onehot_labels(label, weight, pred.size(-1), ignore_index)
    else:
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            weight = weight * valid_mask
        else:
            weight = valid_mask
    if avg_factor is None and avg_non_ignore and (reduction == 'mean'):
        avg_factor = valid_mask.sum().item()
    weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(pred, label.float(), pos_weight=class_weight, reduction='none')
    loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)
    return loss

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

@patch('torch.__version__', torch_version)
def test_AdaptiveAvgPool2d():
    x_empty = torch.randn(0, 3, 4, 5)
    wrapper = AdaptiveAvgPool2d((2, 2))
    wrapper_out = wrapper(x_empty)
    assert wrapper_out.shape == (0, 3, 2, 2)
    wrapper = AdaptiveAvgPool2d(2)
    wrapper_out = wrapper(x_empty)
    assert wrapper_out.shape == (0, 3, 2, 2)
    wrapper = AdaptiveAvgPool2d((None, 2))
    wrapper_out = wrapper(x_empty)
    assert wrapper_out.shape == (0, 3, 4, 2)
    wrapper = AdaptiveAvgPool2d((2, None))
    wrapper_out = wrapper(x_empty)
    assert wrapper_out.shape == (0, 3, 2, 5)
    x_normal = torch.randn(3, 3, 4, 5)
    wrapper = AdaptiveAvgPool2d((2, 2))
    ref = nn.AdaptiveAvgPool2d((2, 2))
    wrapper_out = wrapper(x_normal)
    ref_out = ref(x_normal)
    assert wrapper_out.shape == (3, 3, 2, 2)
    assert torch.equal(wrapper_out, ref_out)
    wrapper = AdaptiveAvgPool2d(2)
    ref = nn.AdaptiveAvgPool2d(2)
    wrapper_out = wrapper(x_normal)
    ref_out = ref(x_normal)
    assert wrapper_out.shape == (3, 3, 2, 2)
    assert torch.equal(wrapper_out, ref_out)
    wrapper = AdaptiveAvgPool2d((None, 2))
    ref = nn.AdaptiveAvgPool2d((None, 2))
    wrapper_out = wrapper(x_normal)
    ref_out = ref(x_normal)
    assert wrapper_out.shape == (3, 3, 4, 2)
    assert torch.equal(wrapper_out, ref_out)
    wrapper = AdaptiveAvgPool2d((2, None))
    ref = nn.AdaptiveAvgPool2d((2, None))
    wrapper_out = wrapper(x_normal)
    ref_out = ref(x_normal)
    assert wrapper_out.shape == (3, 3, 2, 5)
    assert torch.equal(wrapper_out, ref_out)

