# Cluster 33

def bbox_mapping(bboxes, img_shape, scale_factor, flip, flip_direction='horizontal'):
    """Map bboxes from the original image scale to testing scale."""
    new_bboxes = bboxes * bboxes.new_tensor(scale_factor)
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape, flip_direction)
    return new_bboxes

def bbox_flip(bboxes, img_shape, direction='horizontal'):
    """Flip bboxes horizontally or vertically.

    Args:
        bboxes (Tensor): Shape (..., 4*k)
        img_shape (tuple): Image shape.
        direction (str): Flip direction, options are "horizontal", "vertical",
            "diagonal". Default: "horizontal"

    Returns:
        Tensor: Flipped bboxes.
    """
    assert bboxes.shape[-1] % 4 == 0
    assert direction in ['horizontal', 'vertical', 'diagonal']
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[..., 0::4] = img_shape[1] - bboxes[..., 2::4]
        flipped[..., 2::4] = img_shape[1] - bboxes[..., 0::4]
    elif direction == 'vertical':
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    else:
        flipped[..., 0::4] = img_shape[1] - bboxes[..., 2::4]
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 2::4] = img_shape[1] - bboxes[..., 0::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    return flipped

def bbox_mapping_back(bboxes, img_shape, scale_factor, flip, flip_direction='horizontal'):
    """Map bboxes from testing scale to original image scale."""
    new_bboxes = bbox_flip(bboxes, img_shape, flip_direction) if flip else bboxes
    new_bboxes = new_bboxes.view(-1, 4) / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)

@BBOX_SAMPLERS.register_module()
class OHEMSampler(BaseSampler):
    """Online Hard Example Mining Sampler described in `Training Region-based
    Object Detectors with Online Hard Example Mining
    <https://arxiv.org/abs/1604.03540>`_.
    """

    def __init__(self, num, pos_fraction, context, neg_pos_ub=-1, add_gt_as_proposals=True, loss_key='loss_cls', **kwargs):
        super(OHEMSampler, self).__init__(num, pos_fraction, neg_pos_ub, add_gt_as_proposals)
        self.context = context
        if not hasattr(self.context, 'num_stages'):
            self.bbox_head = self.context.bbox_head
        else:
            self.bbox_head = self.context.bbox_head[self.context.current_stage]
        self.loss_key = loss_key

    def hard_mining(self, inds, num_expected, bboxes, labels, feats):
        with torch.no_grad():
            rois = bbox2roi([bboxes])
            if not hasattr(self.context, 'num_stages'):
                bbox_results = self.context._bbox_forward(feats, rois)
            else:
                bbox_results = self.context._bbox_forward(self.context.current_stage, feats, rois)
            cls_score = bbox_results['cls_score']
            loss = self.bbox_head.loss(cls_score=cls_score, bbox_pred=None, rois=rois, labels=labels, label_weights=cls_score.new_ones(cls_score.size(0)), bbox_targets=None, bbox_weights=None, reduction_override='none')[self.loss_key]
            _, topk_loss_inds = loss.topk(num_expected)
        return inds[topk_loss_inds]

    def _sample_pos(self, assign_result, num_expected, bboxes=None, feats=None, **kwargs):
        """Sample positive boxes.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            num_expected (int): Number of expected positive samples
            bboxes (torch.Tensor, optional): Boxes. Defaults to None.
            feats (list[torch.Tensor], optional): Multi-level features.
                Defaults to None.

        Returns:
            torch.Tensor: Indices  of positive samples
        """
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.hard_mining(pos_inds, num_expected, bboxes[pos_inds], assign_result.labels[pos_inds], feats)

    def _sample_neg(self, assign_result, num_expected, bboxes=None, feats=None, **kwargs):
        """Sample negative boxes.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            num_expected (int): Number of expected negative samples
            bboxes (torch.Tensor, optional): Boxes. Defaults to None.
            feats (list[torch.Tensor], optional): Multi-level features.
                Defaults to None.

        Returns:
            torch.Tensor: Indices  of negative samples
        """
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            neg_labels = assign_result.labels.new_empty(neg_inds.size(0)).fill_(self.bbox_head.num_classes)
            return self.hard_mining(neg_inds, num_expected, bboxes[neg_inds], neg_labels, feats)

def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois

@BBOX_SAMPLERS.register_module()
class ScoreHLRSampler(BaseSampler):
    """Importance-based Sample Reweighting (ISR_N), described in `Prime Sample
    Attention in Object Detection <https://arxiv.org/abs/1904.04821>`_.

    Score hierarchical local rank (HLR) differentiates with RandomSampler in
    negative part. It firstly computes Score-HLR in a two-step way,
    then linearly maps score hlr to the loss weights.

    Args:
        num (int): Total number of sampled RoIs.
        pos_fraction (float): Fraction of positive samples.
        context (:class:`BaseRoIHead`): RoI head that the sampler belongs to.
        neg_pos_ub (int): Upper bound of the ratio of num negative to num
            positive, -1 means no upper bound.
        add_gt_as_proposals (bool): Whether to add ground truth as proposals.
        k (float): Power of the non-linear mapping.
        bias (float): Shift of the non-linear mapping.
        score_thr (float): Minimum score that a negative sample is to be
            considered as valid bbox.
    """

    def __init__(self, num, pos_fraction, context, neg_pos_ub=-1, add_gt_as_proposals=True, k=0.5, bias=0, score_thr=0.05, iou_thr=0.5, **kwargs):
        super().__init__(num, pos_fraction, neg_pos_ub, add_gt_as_proposals)
        self.k = k
        self.bias = bias
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.context = context
        if not hasattr(context, 'num_stages'):
            self.bbox_roi_extractor = context.bbox_roi_extractor
            self.bbox_head = context.bbox_head
            self.with_shared_head = context.with_shared_head
            if self.with_shared_head:
                self.shared_head = context.shared_head
        else:
            self.bbox_roi_extractor = context.bbox_roi_extractor[context.current_stage]
            self.bbox_head = context.bbox_head[context.current_stage]

    @staticmethod
    def random_choice(gallery, num):
        """Randomly select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num
        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0).flatten()
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, bboxes, feats=None, img_meta=None, **kwargs):
        """Sample negative samples.

        Score-HLR sampler is done in the following steps:
        1. Take the maximum positive score prediction of each negative samples
            as s_i.
        2. Filter out negative samples whose s_i <= score_thr, the left samples
            are called valid samples.
        3. Use NMS-Match to divide valid samples into different groups,
            samples in the same group will greatly overlap with each other
        4. Rank the matched samples in two-steps to get Score-HLR.
            (1) In the same group, rank samples with their scores.
            (2) In the same score rank across different groups,
                rank samples with their scores again.
        5. Linearly map Score-HLR to the final label weights.

        Args:
            assign_result (:obj:`AssignResult`): result of assigner.
            num_expected (int): Expected number of samples.
            bboxes (Tensor): bbox to be sampled.
            feats (Tensor): Features come from FPN.
            img_meta (dict): Meta information dictionary.
        """
        neg_inds = torch.nonzero(assign_result.gt_inds == 0).flatten()
        num_neg = neg_inds.size(0)
        if num_neg == 0:
            return (neg_inds, None)
        with torch.no_grad():
            neg_bboxes = bboxes[neg_inds]
            neg_rois = bbox2roi([neg_bboxes])
            bbox_result = self.context._bbox_forward(feats, neg_rois)
            cls_score, bbox_pred = (bbox_result['cls_score'], bbox_result['bbox_pred'])
            ori_loss = self.bbox_head.loss(cls_score=cls_score, bbox_pred=None, rois=None, labels=neg_inds.new_full((num_neg,), self.bbox_head.num_classes), label_weights=cls_score.new_ones(num_neg), bbox_targets=None, bbox_weights=None, reduction_override='none')['loss_cls']
            max_score, argmax_score = cls_score.softmax(-1)[:, :-1].max(-1)
            valid_inds = (max_score > self.score_thr).nonzero().view(-1)
            invalid_inds = (max_score <= self.score_thr).nonzero().view(-1)
            num_valid = valid_inds.size(0)
            num_invalid = invalid_inds.size(0)
            num_expected = min(num_neg, num_expected)
            num_hlr = min(num_valid, num_expected)
            num_rand = num_expected - num_hlr
            if num_valid > 0:
                valid_rois = neg_rois[valid_inds]
                valid_max_score = max_score[valid_inds]
                valid_argmax_score = argmax_score[valid_inds]
                valid_bbox_pred = bbox_pred[valid_inds]
                valid_bbox_pred = valid_bbox_pred.view(valid_bbox_pred.size(0), -1, 4)
                selected_bbox_pred = valid_bbox_pred[range(num_valid), valid_argmax_score]
                pred_bboxes = self.bbox_head.bbox_coder.decode(valid_rois[:, 1:], selected_bbox_pred)
                pred_bboxes_with_score = torch.cat([pred_bboxes, valid_max_score[:, None]], -1)
                group = nms_match(pred_bboxes_with_score, self.iou_thr)
                imp = cls_score.new_zeros(num_valid)
                for g in group:
                    g_score = valid_max_score[g]
                    rank = g_score.new_tensor(range(g_score.size(0)))
                    imp[g] = num_valid - rank + g_score
                _, imp_rank_inds = imp.sort(descending=True)
                _, imp_rank = imp_rank_inds.sort()
                hlr_inds = imp_rank_inds[:num_expected]
                if num_rand > 0:
                    rand_inds = torch.randperm(num_invalid)[:num_rand]
                    select_inds = torch.cat([valid_inds[hlr_inds], invalid_inds[rand_inds]])
                else:
                    select_inds = valid_inds[hlr_inds]
                neg_label_weights = cls_score.new_ones(num_expected)
                up_bound = max(num_expected, num_valid)
                imp_weights = (up_bound - imp_rank[hlr_inds].float()) / up_bound
                neg_label_weights[:num_hlr] = imp_weights
                neg_label_weights[num_hlr:] = imp_weights.min()
                neg_label_weights = (self.bias + (1 - self.bias) * neg_label_weights).pow(self.k)
                ori_selected_loss = ori_loss[select_inds]
                new_loss = ori_selected_loss * neg_label_weights
                norm_ratio = ori_selected_loss.sum() / new_loss.sum()
                neg_label_weights *= norm_ratio
            else:
                neg_label_weights = cls_score.new_ones(num_expected)
                select_inds = torch.randperm(num_neg)[:num_expected]
            return (neg_inds[select_inds], neg_label_weights)

    def sample(self, assign_result, bboxes, gt_bboxes, gt_labels=None, img_meta=None, **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            tuple[:obj:`SamplingResult`, Tensor]: Sampling result and negative
                label weights.
        """
        bboxes = bboxes[:, :4]
        gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
        if self.add_gt_as_proposals:
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])
        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds, neg_label_weights = self.neg_sampler._sample_neg(assign_result, num_expected_neg, bboxes, img_meta=img_meta, **kwargs)
        return (SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags), neg_label_weights)

class DeployBaseDetector(BaseDetector):
    """DeployBaseDetector."""

    def __init__(self, class_names, device_id):
        super(DeployBaseDetector, self).__init__()
        self.CLASSES = class_names
        self.device_id = device_id

    def simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def forward_train(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def val_step(self, data, optimizer):
        raise NotImplementedError('This method is not implemented.')

    def train_step(self, data, optimizer):
        raise NotImplementedError('This method is not implemented.')

    def forward_test(self, *, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        outputs = self.forward_test(img, img_metas, **kwargs)
        batch_dets, batch_labels = outputs[:2]
        batch_masks = outputs[2] if len(outputs) == 3 else None
        batch_size = img[0].shape[0]
        img_metas = img_metas[0]
        results = []
        rescale = kwargs.get('rescale', True)
        for i in range(batch_size):
            dets, labels = (batch_dets[i], batch_labels[i])
            if rescale:
                scale_factor = img_metas[i]['scale_factor']
                if isinstance(scale_factor, (list, tuple, np.ndarray)):
                    assert len(scale_factor) == 4
                    scale_factor = np.array(scale_factor)[None, :]
                dets[:, :4] /= scale_factor
            if 'border' in img_metas[i]:
                x_off = img_metas[i]['border'][2]
                y_off = img_metas[i]['border'][0]
                dets[:, [0, 2]] -= x_off
                dets[:, [1, 3]] -= y_off
                dets[:, :4] *= (dets[:, :4] > 0).astype(dets.dtype)
            dets_results = bbox2result(dets, labels, len(self.CLASSES))
            if batch_masks is not None:
                masks = batch_masks[i]
                img_h, img_w = img_metas[i]['img_shape'][:2]
                ori_h, ori_w = img_metas[i]['ori_shape'][:2]
                masks = masks[:, :img_h, :img_w]
                if rescale:
                    masks = masks.astype(np.float32)
                    masks = torch.from_numpy(masks)
                    masks = torch.nn.functional.interpolate(masks.unsqueeze(0), size=(ori_h, ori_w))
                    masks = masks.squeeze(0).detach().numpy()
                if masks.dtype != np.bool:
                    masks = masks >= 0.5
                segms_results = [[] for _ in range(len(self.CLASSES))]
                for j in range(len(dets)):
                    segms_results[labels[j]].append(masks[j])
                results.append((dets_results, segms_results))
            else:
                results.append(dets_results)
        return results

def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]

class BBoxTestMixin(object):
    """Mixin class for testing det bboxes via DenseHead."""

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

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
        outs = self.forward(feats)
        results_list = self.get_bboxes(*outs, img_metas=img_metas, rescale=rescale)
        return results_list

    def aug_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes with test time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

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
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,). The length of list should always be 1.
        """
        gb_sig = signature(self.get_bboxes)
        gb_args = [p.name for p in gb_sig.parameters.values()]
        gbs_sig = signature(self._get_bboxes_single)
        gbs_args = [p.name for p in gbs_sig.parameters.values()]
        assert 'with_nms' in gb_args and 'with_nms' in gbs_args, f'{self.__class__.__name__} does not support test-time augmentation'
        aug_bboxes = []
        aug_scores = []
        aug_labels = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.forward(x)
            bbox_outputs = self.get_bboxes(*outs, img_metas=img_meta, cfg=self.test_cfg, rescale=False, with_nms=False)[0]
            aug_bboxes.append(bbox_outputs[0])
            aug_scores.append(bbox_outputs[1])
            if len(bbox_outputs) >= 3:
                aug_labels.append(bbox_outputs[2])
        merged_bboxes, merged_scores = self.merge_aug_bboxes(aug_bboxes, aug_scores, img_metas)
        merged_labels = torch.cat(aug_labels, dim=0) if aug_labels else None
        if merged_bboxes.numel() == 0:
            det_bboxes = torch.cat([merged_bboxes, merged_scores[:, None]], -1)
            return [(det_bboxes, merged_labels)]
        det_bboxes, keep_idxs = batched_nms(merged_bboxes, merged_scores, merged_labels, self.test_cfg.nms)
        det_bboxes = det_bboxes[:self.test_cfg.max_per_img]
        det_labels = merged_labels[keep_idxs][:self.test_cfg.max_per_img]
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(img_metas[0][0]['scale_factor'])
        return [(_det_bboxes, det_labels)]

    def simple_test_rpn(self, x, img_metas):
        """Test without augmentation, only for ``RPNHead`` and its variants,
        e.g., ``GARPNHead``, etc.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        rpn_outs = self(x)
        proposal_list = self.get_bboxes(*rpn_outs, img_metas=img_metas)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas):
        """Test with augmentation for only for ``RPNHead`` and its variants,
        e.g., ``GARPNHead``, etc.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                        a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        samples_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(samples_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        aug_img_metas = []
        for i in range(samples_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)
        merged_proposals = [merge_aug_proposals(proposals, aug_img_meta, self.test_cfg) for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)]
        return merged_proposals
    if sys.version_info >= (3, 7):

        async def async_simple_test_rpn(self, x, img_metas):
            sleep_interval = self.test_cfg.pop('async_sleep_interval', 0.025)
            async with completed(__name__, 'rpn_head_forward', sleep_interval=sleep_interval):
                rpn_outs = self(x)
            proposal_list = self.get_bboxes(*rpn_outs, img_metas=img_metas)
            return proposal_list

    def merge_aug_bboxes(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).

        Returns:
            tuple[Tensor]: ``bboxes`` with shape (n,4), where
            4 represent (tl_x, tl_y, br_x, br_y)
            and ``scores`` with shape (n,).
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip, flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return (bboxes, scores)

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

def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None, return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (dict): a dict that contains the arguments of nms operations
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]
    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)
    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    if not torch.onnx.is_in_onnx_export():
        valid_mask = scores > score_thr
    if score_factors is not None:
        score_factors = score_factors.view(-1, 1).expand(multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors
    if not torch.onnx.is_in_onnx_export():
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = (bboxes[inds], scores[inds], labels[inds])
    else:
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)
    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS as it has not been executed this time')
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return (dets, labels, inds)
        else:
            return (dets, labels)
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
    if return_inds:
        return (dets, labels[keep], inds[keep])
    else:
        return (dets, labels[keep])

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

@DETECTORS.register_module()
class CornerNet(SingleStageDetector):
    """CornerNet.

    This detector is the implementation of the paper `CornerNet: Detecting
    Objects as Paired Keypoints <https://arxiv.org/abs/1808.01244>`_ .
    """

    def __init__(self, backbone, neck, bbox_head, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        super(CornerNet, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)

    def merge_aug_results(self, aug_results, img_metas):
        """Merge augmented detection bboxes and score.

        Args:
            aug_results (list[list[Tensor]]): Det_bboxes and det_labels of each
                image.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: (bboxes, labels)
        """
        recovered_bboxes, aug_labels = ([], [])
        for bboxes_labels, img_info in zip(aug_results, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            bboxes, labels = bboxes_labels
            bboxes, scores = (bboxes[:, :4], bboxes[:, -1:])
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
            recovered_bboxes.append(torch.cat([bboxes, scores], dim=-1))
            aug_labels.append(labels)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        labels = torch.cat(aug_labels)
        if bboxes.shape[0] > 0:
            out_bboxes, out_labels = self.bbox_head._bboxes_nms(bboxes, labels, self.bbox_head.test_cfg)
        else:
            out_bboxes, out_labels = (bboxes, labels)
        return (out_bboxes, out_labels)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Augment testing of CornerNet.

        Args:
            imgs (list[Tensor]): Augmented images.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Note:
            ``imgs`` must including flipped image pairs.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        img_inds = list(range(len(imgs)))
        assert img_metas[0][0]['flip'] + img_metas[1][0]['flip'], 'aug test must have flipped image pair'
        aug_results = []
        for ind, flip_ind in zip(img_inds[0::2], img_inds[1::2]):
            img_pair = torch.cat([imgs[ind], imgs[flip_ind]])
            x = self.extract_feat(img_pair)
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(*outs, [img_metas[ind], img_metas[flip_ind]], False, False)
            aug_results.append(bbox_list[0])
            aug_results.append(bbox_list[1])
        bboxes, labels = self.merge_aug_results(aug_results, img_metas)
        bbox_results = bbox2result(bboxes, labels, self.bbox_head.num_classes)
        return [bbox_results]

@DETECTORS.register_module()
class MaskFormer(SingleStageDetector):
    """Implementation of `Per-Pixel Classification is
    NOT All You Need for Semantic Segmentation
    <https://arxiv.org/pdf/2107.06278>`_."""

    def __init__(self, backbone, neck=None, panoptic_head=None, panoptic_fusion_head=None, train_cfg=None, test_cfg=None, init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        panoptic_head_ = copy.deepcopy(panoptic_head)
        panoptic_head_.update(train_cfg=train_cfg)
        panoptic_head_.update(test_cfg=test_cfg)
        self.panoptic_head = build_head(panoptic_head_)
        panoptic_fusion_head_ = copy.deepcopy(panoptic_fusion_head)
        panoptic_fusion_head_.update(test_cfg=test_cfg)
        self.panoptic_fusion_head = build_head(panoptic_fusion_head_)
        self.num_things_classes = self.panoptic_head.num_things_classes
        self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.panoptic_head.num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.num_stuff_classes > 0:
            self.show_result = self._show_pan_result

    def forward_dummy(self, img, img_metas):
        """Used for computing network flops. See
        `mmdetection/tools/analysis_tools/get_flops.py`

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[Dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        outs = self.panoptic_head(x, img_metas)
        return outs

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_masks, gt_semantic_seg=None, gt_bboxes_ignore=None, **kargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[Dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            gt_masks (list[BitmapMasks]): true segmentation masks for each box
                used if the architecture supports a segmentation task.
            gt_semantic_seg (list[tensor]): semantic segmentation mask for
                images for panoptic segmentation.
                Defaults to None for instance segmentation.
            gt_bboxes_ignore (list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
                Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.panoptic_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_masks, gt_semantic_seg, gt_bboxes_ignore)
        return losses

    def simple_test(self, imgs, img_metas, **kwargs):
        """Test without augmentation.

        Args:
            imgs (Tensor): A batch of images.
            img_metas (list[dict]): List of image information.

        Returns:
            list[dict[str, np.array | tuple[list]] | tuple[list]]:
                Semantic segmentation results and panoptic segmentation                 results of each image for panoptic segmentation, or formatted                 bbox and mask results of each image for instance segmentation.

            .. code-block:: none

                [
                    # panoptic segmentation
                    {
                        'pan_results': np.array, # shape = [h, w]
                        'ins_results': tuple[list],
                        # semantic segmentation results are not supported yet
                        'sem_results': np.array
                    },
                    ...
                ]

            or

            .. code-block:: none

                [
                    # instance segmentation
                    (
                        bboxes, # list[np.array]
                        masks # list[list[np.array]]
                    ),
                    ...
                ]
        """
        feats = self.extract_feat(imgs)
        mask_cls_results, mask_pred_results = self.panoptic_head.simple_test(feats, img_metas, **kwargs)
        results = self.panoptic_fusion_head.simple_test(mask_cls_results, mask_pred_results, img_metas, **kwargs)
        for i in range(len(results)):
            if 'pan_results' in results[i]:
                results[i]['pan_results'] = results[i]['pan_results'].detach().cpu().numpy()
            if 'ins_results' in results[i]:
                labels_per_image, bboxes, mask_pred_binary = results[i]['ins_results']
                bbox_results = bbox2result(bboxes, labels_per_image, self.num_things_classes)
                mask_results = [[] for _ in range(self.num_things_classes)]
                for j, label in enumerate(labels_per_image):
                    mask = mask_pred_binary[j].detach().cpu().numpy()
                    mask_results[label].append(mask)
                results[i]['ins_results'] = (bbox_results, mask_results)
            assert 'sem_results' not in results[i], 'segmantic segmentation results are not supported yet.'
        if self.num_stuff_classes == 0:
            results = [res['ins_results'] for res in results]
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError

    def onnx_export(self, img, img_metas):
        raise NotImplementedError

    def _show_pan_result(self, img, result, score_thr=0.3, bbox_color=(72, 101, 241), text_color=(72, 101, 241), mask_color=None, thickness=2, font_size=13, win_name='', show=False, wait_time=0, out_file=None):
        """Draw `panoptic result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (dict): The results.

            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'.
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'.
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None.
            thickness (int): Thickness of lines. Default: 2.
            font_size (int): Font size of texts. Default: 13.
            win_name (str): The window name. Default: ''.
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()
        pan_results = result['pan_results']
        ids = np.unique(pan_results)[::-1]
        legal_indices = ids != self.num_classes
        ids = ids[legal_indices]
        labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
        segms = pan_results[None] == ids[:, None, None]
        if out_file is not None:
            show = False
        img = imshow_det_bboxes(img, segms=segms, labels=labels, class_names=self.CLASSES, bbox_color=bbox_color, text_color=text_color, mask_color=mask_color, thickness=thickness, font_size=font_size, win_name=win_name, show=show, wait_time=wait_time, out_file=out_file)
        if not (show or out_file):
            return img

@DETECTORS.register_module()
class TwoStagePanopticSegmentor(TwoStageDetector):
    """Base class of Two-stage Panoptic Segmentor.

    As well as the components in TwoStageDetector, Panoptic Segmentor has extra
    semantic_head and panoptic_fusion_head.
    """

    def __init__(self, backbone, neck=None, rpn_head=None, roi_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None, semantic_head=None, panoptic_fusion_head=None):
        super(TwoStagePanopticSegmentor, self).__init__(backbone, neck, rpn_head, roi_head, train_cfg, test_cfg, pretrained, init_cfg)
        if semantic_head is not None:
            self.semantic_head = build_head(semantic_head)
        if panoptic_fusion_head is not None:
            panoptic_cfg = test_cfg.panoptic if test_cfg is not None else None
            panoptic_fusion_head_ = panoptic_fusion_head.deepcopy()
            panoptic_fusion_head_.update(test_cfg=panoptic_cfg)
            self.panoptic_fusion_head = build_head(panoptic_fusion_head_)
            self.num_things_classes = self.panoptic_fusion_head.num_things_classes
            self.num_stuff_classes = self.panoptic_fusion_head.num_stuff_classes
            self.num_classes = self.panoptic_fusion_head.num_classes

    @property
    def with_semantic_head(self):
        return hasattr(self, 'semantic_head') and self.semantic_head is not None

    @property
    def with_panoptic_fusion_head(self):
        return hasattr(self, 'panoptic_fusion_heads') and self.panoptic_fusion_head is not None

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        raise NotImplementedError(f'`forward_dummy` is not implemented in {self.__class__.__name__}')

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, gt_semantic_seg=None, proposals=None, **kwargs):
        x = self.extract_feat(img)
        losses = dict()
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=gt_bboxes_ignore, proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)
        semantic_loss = self.semantic_head.forward_train(x, gt_semantic_seg)
        losses.update(semantic_loss)
        return losses

    def simple_test_mask(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        """Simple test for mask head without augmentation."""
        img_shapes = tuple((meta['ori_shape'] for meta in img_metas)) if rescale else tuple((meta['pad_shape'] for meta in img_metas))
        scale_factors = tuple((meta['scale_factor'] for meta in img_metas))
        if all((det_bbox.shape[0] == 0 for det_bbox in det_bboxes)):
            masks = []
            for img_shape in img_shapes:
                out_shape = (0, self.roi_head.bbox_head.num_classes) + img_shape[:2]
                masks.append(det_bboxes[0].new_zeros(out_shape))
            mask_pred = det_bboxes[0].new_zeros((0, 80, 28, 28))
            mask_results = dict(masks=masks, mask_pred=mask_pred, mask_feats=None)
            return mask_results
        _bboxes = [det_bboxes[i][:, :4] for i in range(len(det_bboxes))]
        if rescale:
            if not isinstance(scale_factors[0], float):
                scale_factors = [det_bboxes[0].new_tensor(scale_factor) for scale_factor in scale_factors]
            _bboxes = [_bboxes[i] * scale_factors[i] for i in range(len(_bboxes))]
        mask_rois = bbox2roi(_bboxes)
        mask_results = self.roi_head._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
        mask_preds = mask_pred.split(num_mask_roi_per_img, 0)
        masks = []
        for i in range(len(_bboxes)):
            det_bbox = det_bboxes[i][:, :4]
            det_label = det_labels[i]
            mask_pred = mask_preds[i].sigmoid()
            box_inds = torch.arange(mask_pred.shape[0])
            mask_pred = mask_pred[box_inds, det_label][:, None]
            img_h, img_w, _ = img_shapes[i]
            mask_pred, _ = _do_paste_mask(mask_pred, det_bbox, img_h, img_w, skip_empty=False)
            masks.append(mask_pred)
        mask_results['masks'] = masks
        return mask_results

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without Augmentation."""
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        bboxes, scores = self.roi_head.simple_test_bboxes(x, img_metas, proposal_list, None, rescale=rescale)
        pan_cfg = self.test_cfg.panoptic
        det_bboxes = []
        det_labels = []
        for bboxe, score in zip(bboxes, scores):
            det_bbox, det_label = multiclass_nms(bboxe, score, pan_cfg.score_thr, pan_cfg.nms, pan_cfg.max_per_img)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        mask_results = self.simple_test_mask(x, img_metas, det_bboxes, det_labels, rescale=rescale)
        masks = mask_results['masks']
        seg_preds = self.semantic_head.simple_test(x, img_metas, rescale)
        results = []
        for i in range(len(det_bboxes)):
            pan_results = self.panoptic_fusion_head.simple_test(det_bboxes[i], det_labels[i], masks[i], seg_preds[i])
            pan_results = pan_results.int().detach().cpu().numpy()
            result = dict(pan_results=pan_results)
            results.append(result)
        return results

    def show_result(self, img, result, score_thr=0.3, bbox_color=(72, 101, 241), text_color=(72, 101, 241), mask_color=None, thickness=2, font_size=13, win_name='', show=False, wait_time=0, out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (dict): The results.

            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'.
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'.
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None.
            thickness (int): Thickness of lines. Default: 2.
            font_size (int): Font size of texts. Default: 13.
            win_name (str): The window name. Default: ''.
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()
        pan_results = result['pan_results']
        ids = np.unique(pan_results)[::-1]
        legal_indices = ids != self.num_classes
        ids = ids[legal_indices]
        labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
        segms = pan_results[None] == ids[:, None, None]
        if out_file is not None:
            show = False
        img = imshow_det_bboxes(img, segms=segms, labels=labels, class_names=self.CLASSES, bbox_color=bbox_color, text_color=text_color, mask_color=mask_color, thickness=thickness, font_size=font_size, win_name=win_name, show=show, wait_time=wait_time, out_file=out_file)
        if not (show or out_file):
            return img

@DETECTORS.register_module()
class YOLACT(SingleStageDetector):
    """Implementation of `YOLACT <https://arxiv.org/abs/1904.02689>`_"""

    def __init__(self, backbone, neck, bbox_head, segm_head, mask_head, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        super(YOLACT, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)
        self.segm_head = build_head(segm_head)
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        feat = self.extract_feat(img)
        bbox_outs = self.bbox_head(feat)
        prototypes = self.mask_head.forward_dummy(feat[0])
        return (bbox_outs, prototypes)

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        gt_masks = [gt_mask.to_tensor(dtype=torch.uint8, device=img.device) for gt_mask in gt_masks]
        x = self.extract_feat(img)
        cls_score, bbox_pred, coeff_pred = self.bbox_head(x)
        bbox_head_loss_inputs = (cls_score, bbox_pred) + (gt_bboxes, gt_labels, img_metas)
        losses, sampling_results = self.bbox_head.loss(*bbox_head_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        segm_head_outs = self.segm_head(x[0])
        loss_segm = self.segm_head.loss(segm_head_outs, gt_masks, gt_labels)
        losses.update(loss_segm)
        mask_pred = self.mask_head(x[0], coeff_pred, gt_bboxes, img_metas, sampling_results)
        loss_mask = self.mask_head.loss(mask_pred, gt_masks, gt_bboxes, img_metas, sampling_results)
        losses.update(loss_mask)
        for loss_name in losses.keys():
            assert torch.isfinite(torch.stack(losses[loss_name])).all().item(), '{} becomes infinite or NaN!'.format(loss_name)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation."""
        feat = self.extract_feat(img)
        det_bboxes, det_labels, det_coeffs = self.bbox_head.simple_test(feat, img_metas, rescale=rescale)
        bbox_results = [bbox2result(det_bbox, det_label, self.bbox_head.num_classes) for det_bbox, det_label in zip(det_bboxes, det_labels)]
        segm_results = self.mask_head.simple_test(feat, det_bboxes, det_labels, det_coeffs, img_metas, rescale=rescale)
        return list(zip(bbox_results, segm_results))

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations."""
        raise NotImplementedError('YOLACT does not support test-time augmentation')

@DETECTORS.register_module()
class CenterNet(SingleStageDetector):
    """Implementation of CenterNet(Objects as Points)

    <https://arxiv.org/abs/1904.07850>.
    """

    def __init__(self, backbone, neck, bbox_head, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        super(CenterNet, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)

    def merge_aug_results(self, aug_results, with_nms):
        """Merge augmented detection bboxes and score.

        Args:
            aug_results (list[list[Tensor]]): Det_bboxes and det_labels of each
                image.
            with_nms (bool): If True, do nms before return boxes.

        Returns:
            tuple: (out_bboxes, out_labels)
        """
        recovered_bboxes, aug_labels = ([], [])
        for single_result in aug_results:
            recovered_bboxes.append(single_result[0][0])
            aug_labels.append(single_result[0][1])
        bboxes = torch.cat(recovered_bboxes, dim=0).contiguous()
        labels = torch.cat(aug_labels).contiguous()
        if with_nms:
            out_bboxes, out_labels = self.bbox_head._bboxes_nms(bboxes, labels, self.bbox_head.test_cfg)
        else:
            out_bboxes, out_labels = (bboxes, labels)
        return (out_bboxes, out_labels)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Augment testing of CenterNet. Aug test must have flipped image pair,
        and unlike CornerNet, it will perform an averaging operation on the
        feature map instead of detecting bbox.

        Args:
            imgs (list[Tensor]): Augmented images.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.

        Note:
            ``imgs`` must including flipped image pairs.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        img_inds = list(range(len(imgs)))
        assert img_metas[0][0]['flip'] + img_metas[1][0]['flip'], 'aug test must have flipped image pair'
        aug_results = []
        for ind, flip_ind in zip(img_inds[0::2], img_inds[1::2]):
            flip_direction = img_metas[flip_ind][0]['flip_direction']
            img_pair = torch.cat([imgs[ind], imgs[flip_ind]])
            x = self.extract_feat(img_pair)
            center_heatmap_preds, wh_preds, offset_preds = self.bbox_head(x)
            assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1
            center_heatmap_preds[0] = (center_heatmap_preds[0][0:1] + flip_tensor(center_heatmap_preds[0][1:2], flip_direction)) / 2
            wh_preds[0] = (wh_preds[0][0:1] + flip_tensor(wh_preds[0][1:2], flip_direction)) / 2
            bbox_list = self.bbox_head.get_bboxes(center_heatmap_preds, wh_preds, [offset_preds[0][0:1]], img_metas[ind], rescale=rescale, with_nms=False)
            aug_results.append(bbox_list)
        nms_cfg = self.bbox_head.test_cfg.get('nms_cfg', None)
        if nms_cfg is None:
            with_nms = False
        else:
            with_nms = True
        bbox_list = [self.merge_aug_results(aug_results, with_nms)]
        bbox_results = [bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes) for det_bboxes, det_labels in bbox_list]
        return bbox_results

def flip_tensor(src_tensor, flip_direction):
    """flip tensor base on flip_direction.

    Args:
        src_tensor (Tensor): input feature map, shape (B, C, H, W).
        flip_direction (str): The flipping direction. Options are
          'horizontal', 'vertical', 'diagonal'.

    Returns:
        out_tensor (Tensor): Flipped tensor.
    """
    assert src_tensor.ndim == 4
    valid_directions = ['horizontal', 'vertical', 'diagonal']
    assert flip_direction in valid_directions
    if flip_direction == 'horizontal':
        out_tensor = torch.flip(src_tensor, [3])
    elif flip_direction == 'vertical':
        out_tensor = torch.flip(src_tensor, [2])
    else:
        out_tensor = torch.flip(src_tensor, [2, 3])
    return out_tensor

@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self, backbone, neck=None, bbox_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(feat, img_metas, rescale=rescale)
        bbox_results = [bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes) for det_bboxes, det_labels in results_list]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), f'{self.bbox_head.__class__.__name__} does not support test-time augmentation'
        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(feats, img_metas, rescale=rescale)
        bbox_results = [bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes) for det_bboxes, det_labels in results_list]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        if len(outs) == 2:
            outs = (*outs, None)
        det_bboxes, det_labels = self.bbox_head.onnx_export(*outs, img_metas, with_nms=with_nms)
        return (det_bboxes, det_labels)

@DETECTORS.register_module()
class RPN(BaseDetector):
    """Implementation of Region Proposal Network."""

    def __init__(self, backbone, neck, rpn_head, train_cfg, test_cfg, pretrained=None, init_cfg=None):
        super(RPN, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck) if neck is not None else None
        rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
        rpn_head.update(train_cfg=rpn_train_cfg)
        rpn_head.update(test_cfg=test_cfg.rpn)
        self.rpn_head = build_head(rpn_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Extract features.

        Args:
            img (torch.Tensor): Image tensor with shape (n, c, h ,w).

        Returns:
            list[torch.Tensor]: Multi-level features that may have
                different resolutions.
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Dummy forward function."""
        x = self.extract_feat(img)
        rpn_outs = self.rpn_head(x)
        return rpn_outs

    def forward_train(self, img, img_metas, gt_bboxes=None, gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if isinstance(self.train_cfg.rpn, dict) and self.train_cfg.rpn.get('debug', False):
            self.rpn_head.debug_imgs = tensor2imgs(img)
        x = self.extract_feat(img)
        losses = self.rpn_head.forward_train(x, img_metas, gt_bboxes, None, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[np.ndarray]: proposals
        """
        x = self.extract_feat(img)
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape
        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        if rescale:
            for proposals, meta in zip(proposal_list, img_metas):
                proposals[:, :4] /= proposals.new_tensor(meta['scale_factor'])
        if torch.onnx.is_in_onnx_export():
            return proposal_list
        return [proposal.cpu().numpy() for proposal in proposal_list]

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[np.ndarray]: proposals
        """
        proposal_list = self.rpn_head.aug_test_rpn(self.extract_feats(imgs), img_metas)
        if not rescale:
            for proposals, img_meta in zip(proposal_list, img_metas[0]):
                img_shape = img_meta['img_shape']
                scale_factor = img_meta['scale_factor']
                flip = img_meta['flip']
                flip_direction = img_meta['flip_direction']
                proposals[:, :4] = bbox_mapping(proposals[:, :4], img_shape, scale_factor, flip, flip_direction)
        return [proposal.cpu().numpy() for proposal in proposal_list]

    def show_result(self, data, result, top_k=20, **kwargs):
        """Show RPN proposals on the image.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            top_k (int): Plot the first k bboxes only
               if set positive. Default: 20

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        if kwargs is not None:
            kwargs['colors'] = 'green'
            sig = signature(mmcv.imshow_bboxes)
            for k in list(kwargs.keys()):
                if k not in sig.parameters:
                    kwargs.pop(k)
        mmcv.imshow_bboxes(data, result, top_k=top_k, **kwargs)

@HEADS.register_module()
class DynamicRoIHead(StandardRoIHead):
    """RoI head for `Dynamic R-CNN <https://arxiv.org/abs/2004.06002>`_."""

    def __init__(self, **kwargs):
        super(DynamicRoIHead, self).__init__(**kwargs)
        assert isinstance(self.bbox_head.loss_bbox, SmoothL1Loss)
        self.iou_history = []
        self.beta_history = []

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None):
        """Forward function for training.

        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposals (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            cur_iou = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
                sampling_result = self.bbox_sampler.sample(assign_result, proposal_list[i], gt_bboxes[i], gt_labels[i], feats=[lvl_feat[i][None] for lvl_feat in x])
                iou_topk = min(self.train_cfg.dynamic_rcnn.iou_topk, len(assign_result.max_overlaps))
                ious, _ = torch.topk(assign_result.max_overlaps, iou_topk)
                cur_iou.append(ious[-1].item())
                sampling_results.append(sampling_result)
            cur_iou = np.mean(cur_iou)
            self.iou_history.append(cur_iou)
        losses = dict()
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results, gt_bboxes, gt_labels, img_metas)
            losses.update(bbox_results['loss_bbox'])
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results, bbox_results['bbox_feats'], gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])
        update_iter_interval = self.train_cfg.dynamic_rcnn.update_iter_interval
        if len(self.iou_history) % update_iter_interval == 0:
            new_iou_thr, new_beta = self.update_hyperparameters()
        return losses

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        num_imgs = len(img_metas)
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg)
        pos_inds = bbox_targets[3][:, 0].nonzero().squeeze(1)
        num_pos = len(pos_inds)
        cur_target = bbox_targets[2][pos_inds, :2].abs().mean(dim=1)
        beta_topk = min(self.train_cfg.dynamic_rcnn.beta_topk * num_imgs, num_pos)
        cur_target = torch.kthvalue(cur_target, beta_topk)[0].item()
        self.beta_history.append(cur_target)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'], bbox_results['bbox_pred'], rois, *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def update_hyperparameters(self):
        """Update hyperparameters like IoU thresholds for assigner and beta for
        SmoothL1 loss based on the training statistics.

        Returns:
            tuple[float]: the updated ``iou_thr`` and ``beta``.
        """
        new_iou_thr = max(self.train_cfg.dynamic_rcnn.initial_iou, np.mean(self.iou_history))
        self.iou_history = []
        self.bbox_assigner.pos_iou_thr = new_iou_thr
        self.bbox_assigner.neg_iou_thr = new_iou_thr
        self.bbox_assigner.min_pos_iou = new_iou_thr
        if np.median(self.beta_history) < EPS:
            new_beta = self.bbox_head.loss_bbox.beta
        else:
            new_beta = min(self.train_cfg.dynamic_rcnn.initial_beta, np.median(self.beta_history))
        self.beta_history = []
        self.bbox_head.loss_bbox.beta = new_beta
        return (new_iou_thr, new_beta)

@HEADS.register_module()
class GridRoIHead(StandardRoIHead):
    """Grid roi head for Grid R-CNN.

    https://arxiv.org/abs/1811.12030
    """

    def __init__(self, grid_roi_extractor, grid_head, **kwargs):
        assert grid_head is not None
        super(GridRoIHead, self).__init__(**kwargs)
        if grid_roi_extractor is not None:
            self.grid_roi_extractor = build_roi_extractor(grid_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.grid_roi_extractor = self.bbox_roi_extractor
        self.grid_head = build_head(grid_head)

    def _random_jitter(self, sampling_results, img_metas, amplitude=0.15):
        """Ramdom jitter positive proposals for training."""
        for sampling_result, img_meta in zip(sampling_results, img_metas):
            bboxes = sampling_result.pos_bboxes
            random_offsets = bboxes.new_empty(bboxes.shape[0], 4).uniform_(-amplitude, amplitude)
            cxcy = (bboxes[:, 2:4] + bboxes[:, :2]) / 2
            wh = (bboxes[:, 2:4] - bboxes[:, :2]).abs()
            new_cxcy = cxcy + wh * random_offsets[:, :2]
            new_wh = wh * (1 + random_offsets[:, 2:])
            new_x1y1 = new_cxcy - new_wh / 2
            new_x2y2 = new_cxcy + new_wh / 2
            new_bboxes = torch.cat([new_x1y1, new_x2y2], dim=1)
            max_shape = img_meta['img_shape']
            if max_shape is not None:
                new_bboxes[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bboxes[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)
            sampling_result.pos_bboxes = new_bboxes
        return sampling_results

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'], bbox_results['bbox_pred'])
        grid_rois = rois[:100]
        grid_feats = self.grid_roi_extractor(x[:self.grid_roi_extractor.num_inputs], grid_rois)
        if self.with_shared_head:
            grid_feats = self.shared_head(grid_feats)
        grid_pred = self.grid_head(grid_feats)
        outs = outs + (grid_pred,)
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'],)
        return outs

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        """Run forward function and calculate loss for box head in training."""
        bbox_results = super(GridRoIHead, self)._bbox_forward_train(x, sampling_results, gt_bboxes, gt_labels, img_metas)
        sampling_results = self._random_jitter(sampling_results, img_metas)
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        if pos_rois.shape[0] == 0:
            return bbox_results
        grid_feats = self.grid_roi_extractor(x[:self.grid_roi_extractor.num_inputs], pos_rois)
        if self.with_shared_head:
            grid_feats = self.shared_head(grid_feats)
        max_sample_num_grid = self.train_cfg.get('max_num_grid', 192)
        sample_idx = torch.randperm(grid_feats.shape[0])[:min(grid_feats.shape[0], max_sample_num_grid)]
        grid_feats = grid_feats[sample_idx]
        grid_pred = self.grid_head(grid_feats)
        grid_targets = self.grid_head.get_targets(sampling_results, self.train_cfg)
        grid_targets = grid_targets[sample_idx]
        loss_grid = self.grid_head.loss(grid_pred, grid_targets)
        bbox_results['loss_bbox'].update(loss_grid)
        return bbox_results

    def simple_test(self, x, proposal_list, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.simple_test_bboxes(x, img_metas, proposal_list, self.test_cfg, rescale=False)
        grid_rois = bbox2roi([det_bbox[:, :4] for det_bbox in det_bboxes])
        if grid_rois.shape[0] != 0:
            grid_feats = self.grid_roi_extractor(x[:len(self.grid_roi_extractor.featmap_strides)], grid_rois)
            self.grid_head.test_mode = True
            grid_pred = self.grid_head(grid_feats)
            num_roi_per_img = tuple((len(det_bbox) for det_bbox in det_bboxes))
            grid_pred = {k: v.split(num_roi_per_img, 0) for k, v in grid_pred.items()}
            bbox_results = []
            num_imgs = len(det_bboxes)
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    bbox_results.append([np.zeros((0, 5), dtype=np.float32) for _ in range(self.bbox_head.num_classes)])
                else:
                    det_bbox = self.grid_head.get_bboxes(det_bboxes[i], grid_pred['fused'][i], [img_metas[i]])
                    if rescale:
                        det_bbox[:, :4] /= img_metas[i]['scale_factor']
                    bbox_results.append(bbox2result(det_bbox, det_labels[i], self.bbox_head.num_classes))
        else:
            bbox_results = [[np.zeros((0, 5), dtype=np.float32) for _ in range(self.bbox_head.num_classes)] for _ in range(len(det_bboxes))]
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

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

@HEADS.register_module()
class StandardRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'], bbox_results['bbox_pred'])
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'],)
        return outs

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
                sampling_result = self.bbox_sampler.sample(assign_result, proposal_list[i], gt_bboxes[i], gt_labels[i], feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
        losses = dict()
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results, gt_bboxes, gt_labels, img_metas)
            losses.update(bbox_results['loss_bbox'])
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results, bbox_results['bbox_feats'], gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])
        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'], bbox_results['bbox_pred'], rois, *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks, img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(torch.ones(res.pos_bboxes.shape[0], device=device, dtype=torch.uint8))
                pos_inds.append(torch.zeros(res.neg_bboxes.shape[0], device=device, dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            mask_results = self._mask_forward(x, pos_inds=pos_inds, bbox_feats=bbox_feats)
        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks, self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'], mask_targets, pos_labels)
        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert (rois is not None) ^ (pos_inds is not None and bbox_feats is not None)
        if rois is not None:
            mask_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]
        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self, x, proposal_list, img_metas, proposals=None, rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = await self.async_test_bboxes(x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(x, img_metas, det_bboxes, det_labels, rescale=rescale, mask_test_cfg=self.test_cfg.get('mask'))
            return (bbox_results, segm_results)

    def simple_test(self, x, proposal_list, img_metas, proposals=None, rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.simple_test_bboxes(x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = [bbox2result(det_bboxes[i], det_labels[i], self.bbox_head.num_classes) for i in range(len(det_bboxes))]
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas, proposal_list, self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels, self.bbox_head.num_classes)
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes, det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(x, img_metas, proposals, self.test_cfg, rescale=rescale)
        if not self.with_mask:
            return (det_bboxes, det_labels)
        else:
            segm_results = self.mask_onnx_export(x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return (det_bboxes, det_labels, segm_results)

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            Tensor: The segmentation results of shape [N, num_bboxes,
                image_height, image_width].
        """
        if all((det_bbox.shape[0] == 0 for det_bbox in det_bboxes)):
            raise RuntimeError('[ONNX Error] Can not record MaskHead as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(det_bboxes.size(0), device=det_bboxes.device).float().view(-1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes, det_labels, self.test_cfg, max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0], max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg, **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        assert len(img_metas) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']
        rois = proposals
        batch_index = torch.arange(rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(rois.size(0), rois.size(1), 1)
        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img, cls_score.size(-1))
        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img, bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)
        return (det_bboxes, det_labels)

@HEADS.register_module()
class PointRendRoIHead(StandardRoIHead):
    """`PointRend <https://arxiv.org/abs/1912.08193>`_."""

    def __init__(self, point_head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.with_bbox and self.with_mask
        self.init_point_head(point_head)

    def init_point_head(self, point_head):
        """Initialize ``point_head``"""
        self.point_head = builder.build_head(point_head)

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks, img_metas):
        """Run forward function and calculate loss for mask head and point head
        in training."""
        mask_results = super()._mask_forward_train(x, sampling_results, bbox_feats, gt_masks, img_metas)
        if mask_results['loss_mask'] is not None:
            loss_point = self._mask_point_forward_train(x, sampling_results, mask_results['mask_pred'], gt_masks, img_metas)
            mask_results['loss_mask'].update(loss_point)
        return mask_results

    def _mask_point_forward_train(self, x, sampling_results, mask_pred, gt_masks, img_metas):
        """Run forward function and calculate loss for point head in
        training."""
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        rel_roi_points = self.point_head.get_roi_rel_points_train(mask_pred, pos_labels, cfg=self.train_cfg)
        rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        fine_grained_point_feats = self._get_fine_grained_point_feats(x, rois, rel_roi_points, img_metas)
        coarse_point_feats = point_sample(mask_pred, rel_roi_points)
        mask_point_pred = self.point_head(fine_grained_point_feats, coarse_point_feats)
        mask_point_target = self.point_head.get_targets(rois, rel_roi_points, sampling_results, gt_masks, self.train_cfg)
        loss_mask_point = self.point_head.loss(mask_point_pred, mask_point_target, pos_labels)
        return loss_mask_point

    def _get_fine_grained_point_feats(self, x, rois, rel_roi_points, img_metas):
        """Sample fine grained feats from each level feature map and
        concatenate them together.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            rois (Tensor): shape (num_rois, 5).
            rel_roi_points (Tensor): A tensor of shape (num_rois, num_points,
                2) that contains [0, 1] x [0, 1] normalized coordinates of the
                most uncertain points from the [mask_height, mask_width] grid.
            img_metas (list[dict]): Image meta info.

        Returns:
            Tensor: The fine grained features for each points,
                has shape (num_rois, feats_channels, num_points).
        """
        num_imgs = len(img_metas)
        fine_grained_feats = []
        for idx in range(self.mask_roi_extractor.num_inputs):
            feats = x[idx]
            spatial_scale = 1.0 / float(self.mask_roi_extractor.featmap_strides[idx])
            point_feats = []
            for batch_ind in range(num_imgs):
                feat = feats[batch_ind].unsqueeze(0)
                inds = rois[:, 0].long() == batch_ind
                if inds.any():
                    rel_img_points = rel_roi_point_to_rel_img_point(rois[inds], rel_roi_points[inds], feat.shape[2:], spatial_scale).unsqueeze(0)
                    point_feat = point_sample(feat, rel_img_points)
                    point_feat = point_feat.squeeze(0).transpose(0, 1)
                    point_feats.append(point_feat)
            fine_grained_feats.append(torch.cat(point_feats, dim=0))
        return torch.cat(fine_grained_feats, dim=1)

    def _mask_point_forward_test(self, x, rois, label_pred, mask_pred, img_metas):
        """Mask refining process with point head in testing.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            rois (Tensor): shape (num_rois, 5).
            label_pred (Tensor): The predication class for each rois.
            mask_pred (Tensor): The predication coarse masks of
                shape (num_rois, num_classes, small_size, small_size).
            img_metas (list[dict]): Image meta info.

        Returns:
            Tensor: The refined masks of shape (num_rois, num_classes,
                large_size, large_size).
        """
        refined_mask_pred = mask_pred.clone()
        for subdivision_step in range(self.test_cfg.subdivision_steps):
            refined_mask_pred = F.interpolate(refined_mask_pred, scale_factor=self.test_cfg.scale_factor, mode='bilinear', align_corners=False)
            num_rois, channels, mask_height, mask_width = refined_mask_pred.shape
            if self.test_cfg.subdivision_num_points >= self.test_cfg.scale_factor ** 2 * mask_height * mask_width and subdivision_step < self.test_cfg.subdivision_steps - 1:
                continue
            point_indices, rel_roi_points = self.point_head.get_roi_rel_points_test(refined_mask_pred, label_pred, cfg=self.test_cfg)
            fine_grained_point_feats = self._get_fine_grained_point_feats(x, rois, rel_roi_points, img_metas)
            coarse_point_feats = point_sample(mask_pred, rel_roi_points)
            mask_point_pred = self.point_head(fine_grained_point_feats, coarse_point_feats)
            point_indices = point_indices.unsqueeze(1).expand(-1, channels, -1)
            refined_mask_pred = refined_mask_pred.reshape(num_rois, channels, mask_height * mask_width)
            refined_mask_pred = refined_mask_pred.scatter_(2, point_indices, mask_point_pred)
            refined_mask_pred = refined_mask_pred.view(num_rois, channels, mask_height, mask_width)
        return refined_mask_pred

    def simple_test_mask(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        """Obtain mask prediction without augmentation."""
        ori_shapes = tuple((meta['ori_shape'] for meta in img_metas))
        scale_factors = tuple((meta['scale_factor'] for meta in img_metas))
        if isinstance(scale_factors[0], float):
            warnings.warn('Scale factor in img_metas should be a ndarray with shape (4,) arrange as (factor_w, factor_h, factor_w, factor_h), The scale_factor with float type has been deprecated. ')
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)
        num_imgs = len(det_bboxes)
        if all((det_bbox.shape[0] == 0 for det_bbox in det_bboxes)):
            segm_results = [[[] for _ in range(self.mask_head.num_classes)] for _ in range(num_imgs)]
        else:
            _bboxes = [det_bboxes[i][:, :4] for i in range(len(det_bboxes))]
            if rescale:
                scale_factors = [torch.from_numpy(scale_factor).to(det_bboxes[0].device) for scale_factor in scale_factors]
                _bboxes = [_bboxes[i] * scale_factors[i] for i in range(len(_bboxes))]
            mask_rois = bbox2roi(_bboxes)
            mask_results = self._mask_forward(x, mask_rois)
            mask_pred = mask_results['mask_pred']
            num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
            mask_preds = mask_pred.split(num_mask_roi_per_img, 0)
            mask_rois = mask_rois.split(num_mask_roi_per_img, 0)
            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append([[] for _ in range(self.mask_head.num_classes)])
                else:
                    x_i = [xx[[i]] for xx in x]
                    mask_rois_i = mask_rois[i]
                    mask_rois_i[:, 0] = 0
                    mask_pred_i = self._mask_point_forward_test(x_i, mask_rois_i, det_labels[i], mask_preds[i], [img_metas])
                    segm_result = self.mask_head.get_seg_masks(mask_pred_i, _bboxes[i], det_labels[i], self.test_cfg, ori_shapes[i], scale_factors[i], rescale)
                    segm_results.append(segm_result)
        return segm_results

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        """Test for mask head with test time augmentation."""
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape, scale_factor, flip)
                mask_rois = bbox2roi([_bboxes])
                mask_results = self._mask_forward(x, mask_rois)
                mask_results['mask_pred'] = self._mask_point_forward_test(x, mask_rois, det_labels, mask_results['mask_pred'], img_meta)
                aug_masks.append(mask_results['mask_pred'].sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas, self.test_cfg)
            ori_shape = img_metas[0][0]['ori_shape']
            segm_result = self.mask_head.get_seg_masks(merged_masks, det_bboxes, det_labels, self.test_cfg, ori_shape, scale_factor=1.0, rescale=False)
        return segm_result

    def _onnx_get_fine_grained_point_feats(self, x, rois, rel_roi_points):
        """Export the process of sampling fine grained feats to onnx.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            rois (Tensor): shape (num_rois, 5).
            rel_roi_points (Tensor): A tensor of shape (num_rois, num_points,
                2) that contains [0, 1] x [0, 1] normalized coordinates of the
                most uncertain points from the [mask_height, mask_width] grid.

        Returns:
            Tensor: The fine grained features for each points,
                has shape (num_rois, feats_channels, num_points).
        """
        batch_size = x[0].shape[0]
        num_rois = rois.shape[0]
        fine_grained_feats = []
        for idx in range(self.mask_roi_extractor.num_inputs):
            feats = x[idx]
            spatial_scale = 1.0 / float(self.mask_roi_extractor.featmap_strides[idx])
            rel_img_points = rel_roi_point_to_rel_img_point(rois, rel_roi_points, feats, spatial_scale)
            channels = feats.shape[1]
            num_points = rel_img_points.shape[1]
            rel_img_points = rel_img_points.reshape(batch_size, -1, num_points, 2)
            point_feats = point_sample(feats, rel_img_points)
            point_feats = point_feats.transpose(1, 2).reshape(num_rois, channels, num_points)
            fine_grained_feats.append(point_feats)
        return torch.cat(fine_grained_feats, dim=1)

    def _mask_point_onnx_export(self, x, rois, label_pred, mask_pred):
        """Export mask refining process with point head to onnx.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            rois (Tensor): shape (num_rois, 5).
            label_pred (Tensor): The predication class for each rois.
            mask_pred (Tensor): The predication coarse masks of
                shape (num_rois, num_classes, small_size, small_size).

        Returns:
            Tensor: The refined masks of shape (num_rois, num_classes,
                large_size, large_size).
        """
        refined_mask_pred = mask_pred.clone()
        for subdivision_step in range(self.test_cfg.subdivision_steps):
            refined_mask_pred = F.interpolate(refined_mask_pred, scale_factor=self.test_cfg.scale_factor, mode='bilinear', align_corners=False)
            num_rois, channels, mask_height, mask_width = refined_mask_pred.shape
            if self.test_cfg.subdivision_num_points >= self.test_cfg.scale_factor ** 2 * mask_height * mask_width and subdivision_step < self.test_cfg.subdivision_steps - 1:
                continue
            point_indices, rel_roi_points = self.point_head.get_roi_rel_points_test(refined_mask_pred, label_pred, cfg=self.test_cfg)
            fine_grained_point_feats = self._onnx_get_fine_grained_point_feats(x, rois, rel_roi_points)
            coarse_point_feats = point_sample(mask_pred, rel_roi_points)
            mask_point_pred = self.point_head(fine_grained_point_feats, coarse_point_feats)
            point_indices = point_indices.unsqueeze(1).expand(-1, channels, -1)
            refined_mask_pred = refined_mask_pred.reshape(num_rois, channels, mask_height * mask_width)
            is_trt_backend = os.environ.get('ONNX_BACKEND') == 'MMCVTensorRT'
            if is_trt_backend:
                mask_shape = refined_mask_pred.shape
                point_shape = point_indices.shape
                inds_dim0 = torch.arange(point_shape[0]).reshape(point_shape[0], 1, 1).expand_as(point_indices)
                inds_dim1 = torch.arange(point_shape[1]).reshape(1, point_shape[1], 1).expand_as(point_indices)
                inds_1d = inds_dim0.reshape(-1) * mask_shape[1] * mask_shape[2] + inds_dim1.reshape(-1) * mask_shape[2] + point_indices.reshape(-1)
                refined_mask_pred = refined_mask_pred.reshape(-1)
                refined_mask_pred[inds_1d] = mask_point_pred.reshape(-1)
                refined_mask_pred = refined_mask_pred.reshape(*mask_shape)
            else:
                refined_mask_pred = refined_mask_pred.scatter_(2, point_indices, mask_point_pred)
            refined_mask_pred = refined_mask_pred.view(num_rois, channels, mask_height, mask_width)
        return refined_mask_pred

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            Tensor: The segmentation results of shape [N, num_bboxes,
                image_height, image_width].
        """
        if all((det_bbox.shape[0] == 0 for det_bbox in det_bboxes)):
            raise RuntimeError('[ONNX Error] Can not record MaskHead as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(det_bboxes.size(0), device=det_bboxes.device).float().view(-1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        mask_pred = self._mask_point_onnx_export(x, mask_rois, det_labels, mask_pred)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes, det_labels, self.test_cfg, max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0], max_shape[1])
        return segm_results

def merge_aug_masks(aug_masks, img_metas, rcnn_test_cfg, weights=None):
    """Merge augmented mask prediction.

    Args:
        aug_masks (list[ndarray]): shape (n, #class, h, w)
        img_shapes (list[ndarray]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_masks = []
    for mask, img_info in zip(aug_masks, img_metas):
        flip = img_info[0]['flip']
        if flip:
            flip_direction = img_info[0]['flip_direction']
            if flip_direction == 'horizontal':
                mask = mask[:, :, :, ::-1]
            elif flip_direction == 'vertical':
                mask = mask[:, :, ::-1, :]
            elif flip_direction == 'diagonal':
                mask = mask[:, :, :, ::-1]
                mask = mask[:, :, ::-1, :]
            else:
                raise ValueError(f"Invalid flipping direction '{flip_direction}'")
        recovered_masks.append(mask)
    if weights is None:
        merged_masks = np.mean(recovered_masks, axis=0)
    else:
        merged_masks = np.average(np.array(recovered_masks), axis=0, weights=np.array(weights))
    return merged_masks

@HEADS.register_module()
class TridentRoIHead(StandardRoIHead):
    """Trident roi head.

    Args:
        num_branch (int): Number of branches in TridentNet.
        test_branch_idx (int): In inference, all 3 branches will be used
            if `test_branch_idx==-1`, otherwise only branch with index
            `test_branch_idx` will be used.
    """

    def __init__(self, num_branch, test_branch_idx, **kwargs):
        self.num_branch = num_branch
        self.test_branch_idx = test_branch_idx
        super(TridentRoIHead, self).__init__(**kwargs)

    def merge_trident_bboxes(self, trident_det_bboxes, trident_det_labels):
        """Merge bbox predictions of each branch."""
        if trident_det_bboxes.numel() == 0:
            det_bboxes = trident_det_bboxes.new_zeros((0, 5))
            det_labels = trident_det_bboxes.new_zeros((0,), dtype=torch.long)
        else:
            nms_bboxes = trident_det_bboxes[:, :4]
            nms_scores = trident_det_bboxes[:, 4].contiguous()
            nms_inds = trident_det_labels
            nms_cfg = self.test_cfg['nms']
            det_bboxes, keep = batched_nms(nms_bboxes, nms_scores, nms_inds, nms_cfg)
            det_labels = trident_det_labels[keep]
            if self.test_cfg['max_per_img'] > 0:
                det_labels = det_labels[:self.test_cfg['max_per_img']]
                det_bboxes = det_bboxes[:self.test_cfg['max_per_img']]
        return (det_bboxes, det_labels)

    def simple_test(self, x, proposal_list, img_metas, proposals=None, rescale=False):
        """Test without augmentation as follows:

        1. Compute prediction bbox and label per branch.
        2. Merge predictions of each branch according to scores of
           bboxes, i.e., bboxes with higher score are kept to give
           top-k prediction.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes_list, det_labels_list = self.simple_test_bboxes(x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        num_branch = self.num_branch if self.test_branch_idx == -1 else 1
        for _ in range(len(det_bboxes_list)):
            if det_bboxes_list[_].shape[0] == 0:
                det_bboxes_list[_] = det_bboxes_list[_].new_empty((0, 5))
        det_bboxes, det_labels = ([], [])
        for i in range(len(img_metas) // num_branch):
            det_result = self.merge_trident_bboxes(torch.cat(det_bboxes_list[i * num_branch:(i + 1) * num_branch]), torch.cat(det_labels_list[i * num_branch:(i + 1) * num_branch]))
            det_bboxes.append(det_result[0])
            det_labels.append(det_result[1])
        bbox_results = [bbox2result(det_bboxes[i], det_labels[i], self.bbox_head.num_classes) for i in range(len(det_bboxes))]
        return bbox_results

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            trident_bboxes, trident_scores = ([], [])
            for branch_idx in range(len(proposal_list)):
                proposals = bbox_mapping(proposal_list[0][:, :4], img_shape, scale_factor, flip, flip_direction)
                rois = bbox2roi([proposals])
                bbox_results = self._bbox_forward(x, rois)
                bboxes, scores = self.bbox_head.get_bboxes(rois, bbox_results['cls_score'], bbox_results['bbox_pred'], img_shape, scale_factor, rescale=False, cfg=None)
                trident_bboxes.append(bboxes)
                trident_scores.append(scores)
            aug_bboxes.append(torch.cat(trident_bboxes, 0))
            aug_scores.append(torch.cat(trident_scores, 0))
        merged_bboxes, merged_scores = merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores, rcnn_test_cfg.score_thr, rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
        return (det_bboxes, det_labels)

def merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg):
    """Merge augmented detection bboxes and scores.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 4*#class)
        aug_scores (list[Tensor] or None): shape (n, #class)
        img_shapes (list[Tensor]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        flip_direction = img_info[0]['flip_direction']
        bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip, flip_direction)
        recovered_bboxes.append(bboxes)
    bboxes = torch.stack(recovered_bboxes).mean(dim=0)
    if aug_scores is None:
        return bboxes
    else:
        scores = torch.stack(aug_scores).mean(dim=0)
        return (bboxes, scores)

class BBoxTestMixin:
    if sys.version_info >= (3, 7):

        async def async_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False, **kwargs):
            """Asynchronized test for box head without augmentation."""
            rois = bbox2roi(proposals)
            roi_feats = self.bbox_roi_extractor(x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            sleep_interval = rcnn_test_cfg.get('async_sleep_interval', 0.017)
            async with completed(__name__, 'bbox_head_forward', sleep_interval=sleep_interval):
                cls_score, bbox_pred = self.bbox_head(roi_feats)
            img_shape = img_metas[0]['img_shape']
            scale_factor = img_metas[0]['scale_factor']
            det_bboxes, det_labels = self.bbox_head.get_bboxes(rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=rescale, cfg=rcnn_test_cfg)
            return (det_bboxes, det_labels)

    def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        rois = bbox2roi(proposals)
        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0,), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros((0, self.bbox_head.fc_cls.out_features))
            return ([det_bbox] * batch_size, [det_label] * batch_size)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple((meta['img_shape'] for meta in img_metas))
        scale_factors = tuple((meta['scale_factor'] for meta in img_metas))
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple((len(p) for p in proposals))
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        if bbox_pred is not None:
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0,), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros((0, self.bbox_head.fc_cls.out_features))
            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(rois[i], cls_score[i], bbox_pred[i], img_shapes[i], scale_factors[i], rescale=rescale, cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return (det_bboxes, det_labels)

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape, scale_factor, flip, flip_direction)
            rois = bbox2roi([proposals])
            bbox_results = self._bbox_forward(x, rois)
            bboxes, scores = self.bbox_head.get_bboxes(rois, bbox_results['cls_score'], bbox_results['bbox_pred'], img_shape, scale_factor, rescale=False, cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        merged_bboxes, merged_scores = merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        if merged_bboxes.shape[0] == 0:
            det_bboxes = merged_bboxes.new_zeros(0, 5)
            det_labels = merged_bboxes.new_zeros((0,), dtype=torch.long)
        else:
            det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores, rcnn_test_cfg.score_thr, rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
        return (det_bboxes, det_labels)

class MaskTestMixin:
    if sys.version_info >= (3, 7):

        async def async_test_mask(self, x, img_metas, det_bboxes, det_labels, rescale=False, mask_test_cfg=None):
            """Asynchronized test for mask head without augmentation."""
            ori_shape = img_metas[0]['ori_shape']
            scale_factor = img_metas[0]['scale_factor']
            if det_bboxes.shape[0] == 0:
                segm_result = [[] for _ in range(self.mask_head.num_classes)]
            else:
                if rescale and (not isinstance(scale_factor, (float, torch.Tensor))):
                    scale_factor = det_bboxes.new_tensor(scale_factor)
                _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                if mask_test_cfg and mask_test_cfg.get('async_sleep_interval'):
                    sleep_interval = mask_test_cfg['async_sleep_interval']
                else:
                    sleep_interval = 0.035
                async with completed(__name__, 'mask_head_forward', sleep_interval=sleep_interval):
                    mask_pred = self.mask_head(mask_feats)
                segm_result = self.mask_head.get_seg_masks(mask_pred, _bboxes, det_labels, self.test_cfg, ori_shape, scale_factor, rescale)
            return segm_result

    def simple_test_mask(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        """Simple test for mask head without augmentation."""
        ori_shapes = tuple((meta['ori_shape'] for meta in img_metas))
        scale_factors = tuple((meta['scale_factor'] for meta in img_metas))
        if isinstance(scale_factors[0], float):
            warnings.warn('Scale factor in img_metas should be a ndarray with shape (4,) arrange as (factor_w, factor_h, factor_w, factor_h), The scale_factor with float type has been deprecated. ')
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)
        num_imgs = len(det_bboxes)
        if all((det_bbox.shape[0] == 0 for det_bbox in det_bboxes)):
            segm_results = [[[] for _ in range(self.mask_head.num_classes)] for _ in range(num_imgs)]
        else:
            if rescale:
                scale_factors = [torch.from_numpy(scale_factor).to(det_bboxes[0].device) for scale_factor in scale_factors]
            _bboxes = [det_bboxes[i][:, :4] * scale_factors[i] if rescale else det_bboxes[i][:, :4] for i in range(len(det_bboxes))]
            mask_rois = bbox2roi(_bboxes)
            mask_results = self._mask_forward(x, mask_rois)
            mask_pred = mask_results['mask_pred']
            num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
            mask_preds = mask_pred.split(num_mask_roi_per_img, 0)
            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append([[] for _ in range(self.mask_head.num_classes)])
                else:
                    segm_result = self.mask_head.get_seg_masks(mask_preds[i], _bboxes[i], det_labels[i], self.test_cfg, ori_shapes[i], scale_factors[i], rescale)
                    segm_results.append(segm_result)
        return segm_results

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        """Test for mask head with test time augmentation."""
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                flip_direction = img_meta[0]['flip_direction']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape, scale_factor, flip, flip_direction)
                mask_rois = bbox2roi([_bboxes])
                mask_results = self._mask_forward(x, mask_rois)
                aug_masks.append(mask_results['mask_pred'].sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas, self.test_cfg)
            ori_shape = img_metas[0][0]['ori_shape']
            scale_factor = det_bboxes.new_ones(4)
            segm_result = self.mask_head.get_seg_masks(merged_masks, det_bboxes, det_labels, self.test_cfg, ori_shape, scale_factor=scale_factor, rescale=False)
        return segm_result

@HEADS.register_module()
class MaskScoringRoIHead(StandardRoIHead):
    """Mask Scoring RoIHead for Mask Scoring RCNN.

    https://arxiv.org/abs/1903.00241
    """

    def __init__(self, mask_iou_head, **kwargs):
        assert mask_iou_head is not None
        super(MaskScoringRoIHead, self).__init__(**kwargs)
        self.mask_iou_head = build_head(mask_iou_head)

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks, img_metas):
        """Run forward function and calculate loss for Mask head in
        training."""
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        mask_results = super(MaskScoringRoIHead, self)._mask_forward_train(x, sampling_results, bbox_feats, gt_masks, img_metas)
        if mask_results['loss_mask'] is None:
            return mask_results
        pos_mask_pred = mask_results['mask_pred'][range(mask_results['mask_pred'].size(0)), pos_labels]
        mask_iou_pred = self.mask_iou_head(mask_results['mask_feats'], pos_mask_pred)
        pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)), pos_labels]
        mask_iou_targets = self.mask_iou_head.get_targets(sampling_results, gt_masks, pos_mask_pred, mask_results['mask_targets'], self.train_cfg)
        loss_mask_iou = self.mask_iou_head.loss(pos_mask_iou_pred, mask_iou_targets)
        mask_results['loss_mask'].update(loss_mask_iou)
        return mask_results

    def simple_test_mask(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        """Obtain mask prediction without augmentation."""
        ori_shapes = tuple((meta['ori_shape'] for meta in img_metas))
        scale_factors = tuple((meta['scale_factor'] for meta in img_metas))
        num_imgs = len(det_bboxes)
        if all((det_bbox.shape[0] == 0 for det_bbox in det_bboxes)):
            num_classes = self.mask_head.num_classes
            segm_results = [[[] for _ in range(num_classes)] for _ in range(num_imgs)]
            mask_scores = [[[] for _ in range(num_classes)] for _ in range(num_imgs)]
        else:
            if rescale and (not isinstance(scale_factors[0], float)):
                scale_factors = [torch.from_numpy(scale_factor).to(det_bboxes[0].device) for scale_factor in scale_factors]
            _bboxes = [det_bboxes[i][:, :4] * scale_factors[i] if rescale else det_bboxes[i] for i in range(num_imgs)]
            mask_rois = bbox2roi(_bboxes)
            mask_results = self._mask_forward(x, mask_rois)
            concat_det_labels = torch.cat(det_labels)
            mask_feats = mask_results['mask_feats']
            mask_pred = mask_results['mask_pred']
            mask_iou_pred = self.mask_iou_head(mask_feats, mask_pred[range(concat_det_labels.size(0)), concat_det_labels])
            num_bboxes_per_img = tuple((len(_bbox) for _bbox in _bboxes))
            mask_preds = mask_pred.split(num_bboxes_per_img, 0)
            mask_iou_preds = mask_iou_pred.split(num_bboxes_per_img, 0)
            segm_results = []
            mask_scores = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append([[] for _ in range(self.mask_head.num_classes)])
                    mask_scores.append([[] for _ in range(self.mask_head.num_classes)])
                else:
                    segm_result = self.mask_head.get_seg_masks(mask_preds[i], _bboxes[i], det_labels[i], self.test_cfg, ori_shapes[i], scale_factors[i], rescale)
                    mask_score = self.mask_iou_head.get_mask_scores(mask_iou_preds[i], det_bboxes[i], det_labels[i])
                    segm_results.append(segm_result)
                    mask_scores.append(mask_score)
        return list(zip(segm_results, mask_scores))

@HEADS.register_module()
class HybridTaskCascadeRoIHead(CascadeRoIHead):
    """Hybrid task cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1901.07518
    """

    def __init__(self, num_stages, stage_loss_weights, semantic_roi_extractor=None, semantic_head=None, semantic_fusion=('bbox', 'mask'), interleaved=True, mask_info_flow=True, **kwargs):
        super(HybridTaskCascadeRoIHead, self).__init__(num_stages, stage_loss_weights, **kwargs)
        assert self.with_bbox
        assert not self.with_shared_head
        if semantic_head is not None:
            self.semantic_roi_extractor = build_roi_extractor(semantic_roi_extractor)
            self.semantic_head = build_head(semantic_head)
        self.semantic_fusion = semantic_fusion
        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow

    @property
    def with_semantic(self):
        """bool: whether the head has semantic head"""
        if hasattr(self, 'semantic_head') and self.semantic_head is not None:
            return True
        else:
            return False

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        outs = ()
        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None
        rois = bbox2roi([proposals])
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois, semantic_feat=semantic_feat)
            outs = outs + (bbox_results['cls_score'], bbox_results['bbox_pred'])
        if self.with_mask:
            mask_rois = rois[:100]
            mask_roi_extractor = self.mask_roi_extractor[-1]
            mask_feats = mask_roi_extractor(x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_semantic and 'mask' in self.semantic_fusion:
                mask_semantic_feat = self.semantic_roi_extractor([semantic_feat], mask_rois)
                mask_feats = mask_feats + mask_semantic_feat
            last_feat = None
            for i in range(self.num_stages):
                mask_head = self.mask_head[i]
                if self.mask_info_flow:
                    mask_pred, last_feat = mask_head(mask_feats, last_feat)
                else:
                    mask_pred = mask_head(mask_feats)
                outs = outs + (mask_pred,)
        return outs

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg, semantic_feat=None):
        """Run forward function and calculate loss for box head in training."""
        bbox_head = self.bbox_head[stage]
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois, semantic_feat=semantic_feat)
        bbox_targets = bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        loss_bbox = bbox_head.loss(bbox_results['cls_score'], bbox_results['bbox_pred'], rois, *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward_train(self, stage, x, sampling_results, gt_masks, rcnn_train_cfg, semantic_feat=None):
        """Run forward function and calculate loss for mask head in
        training."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs], pos_rois)
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat], pos_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats = mask_feats + mask_semantic_feat
        if self.mask_info_flow:
            last_feat = None
            for i in range(stage):
                last_feat = self.mask_head[i](mask_feats, last_feat, return_logits=False)
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
        else:
            mask_pred = mask_head(mask_feats, return_feat=False)
        mask_targets = mask_head.get_targets(sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)
        mask_results = dict(loss_mask=loss_mask)
        return mask_results

    def _bbox_forward(self, stage, x, rois, semantic_feat=None):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:len(bbox_roi_extractor.featmap_strides)], rois)
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat], rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = adaptive_avg_pool2d(bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats = bbox_feats + bbox_semantic_feat
        cls_score, bbox_pred = bbox_head(bbox_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def _mask_forward_test(self, stage, x, bboxes, semantic_feat=None):
        """Mask head forward function for testing."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_rois = bbox2roi([bboxes])
        mask_feats = mask_roi_extractor(x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat], mask_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats = mask_feats + mask_semantic_feat
        if self.mask_info_flow:
            last_feat = None
            last_pred = None
            for i in range(stage):
                mask_pred, last_feat = self.mask_head[i](mask_feats, last_feat)
                if last_pred is not None:
                    mask_pred = mask_pred + last_pred
                last_pred = mask_pred
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
            if last_pred is not None:
                mask_pred = mask_pred + last_pred
        else:
            mask_pred = mask_head(mask_feats)
        return mask_pred

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, gt_semantic_seg=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposal_list (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None, list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None, Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            gt_semantic_seg (None, list[Tensor]): semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
            losses['loss_semantic_seg'] = loss_seg
        else:
            semantic_feat = None
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]
            sampling_results = []
            bbox_assigner = self.bbox_assigner[i]
            bbox_sampler = self.bbox_sampler[i]
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j], gt_labels[j])
                sampling_result = bbox_sampler.sample(assign_result, proposal_list[j], gt_bboxes[j], gt_labels[j], feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
            bbox_results = self._bbox_forward_train(i, x, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_results['bbox_targets'][0]
            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = value * lw if 'loss' in name else value
            if self.with_mask:
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_bboxes(bbox_results['rois'], roi_labels, bbox_results['bbox_pred'], pos_is_gts, img_metas)
                        sampling_results = []
                        for j in range(num_imgs):
                            assign_result = bbox_assigner.assign(proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j], gt_labels[j])
                            sampling_result = bbox_sampler.sample(assign_result, proposal_list[j], gt_bboxes[j], gt_labels[j], feats=[lvl_feat[j][None] for lvl_feat in x])
                            sampling_results.append(sampling_result)
                mask_results = self._mask_forward_train(i, x, sampling_results, gt_masks, rcnn_train_cfg, semantic_feat)
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = value * lw if 'loss' in name else value
            if i < self.num_stages - 1 and (not self.interleaved):
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(bbox_results['rois'], roi_labels, bbox_results['bbox_pred'], pos_is_gts, img_metas)
        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None
        num_imgs = len(proposal_list)
        img_shapes = tuple((meta['img_shape'] for meta in img_metas))
        ori_shapes = tuple((meta['ori_shape'] for meta in img_metas))
        scale_factors = tuple((meta['scale_factor'] for meta in img_metas))
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg
        rois = bbox2roi(proposal_list)
        if rois.shape[0] == 0:
            bbox_results = [[np.zeros((0, 5), dtype=np.float32) for _ in range(self.bbox_head[-1].num_classes)]] * num_imgs
            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)] for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results
            return results
        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            bbox_results = self._bbox_forward(i, x, rois, semantic_feat=semantic_feat)
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple((len(p) for p in proposal_list))
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            ms_scores.append(cls_score)
            if i < self.num_stages - 1:
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refine_rois = bbox_head.regress_by_class(rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refine_rois)
                rois = torch.cat(refine_rois_list)
        cls_score = [sum([score[i] for score in ms_scores]) / float(len(ms_scores)) for i in range(num_imgs)]
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(rois[i], cls_score[i], bbox_pred[i], img_shapes[i], scale_factors[i], rescale=rescale, cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        bbox_result = [bbox2result(det_bboxes[i], det_labels[i], self.bbox_head[-1].num_classes) for i in range(num_imgs)]
        ms_bbox_result['ensemble'] = bbox_result
        if self.with_mask:
            if all((det_bbox.shape[0] == 0 for det_bbox in det_bboxes)):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)] for _ in range(num_imgs)]
            else:
                if rescale and (not isinstance(scale_factors[0], float)):
                    scale_factors = [torch.from_numpy(scale_factor).to(det_bboxes[0].device) for scale_factor in scale_factors]
                _bboxes = [det_bboxes[i][:, :4] * scale_factors[i] if rescale else det_bboxes[i] for i in range(num_imgs)]
                mask_rois = bbox2roi(_bboxes)
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor([semantic_feat], mask_rois)
                    mask_feats = mask_feats + mask_semantic_feat
                last_feat = None
                num_bbox_per_img = tuple((len(_bbox) for _bbox in _bboxes))
                for i in range(self.num_stages):
                    mask_head = self.mask_head[i]
                    if self.mask_info_flow:
                        mask_pred, last_feat = mask_head(mask_feats, last_feat)
                    else:
                        mask_pred = mask_head(mask_feats)
                    mask_pred = mask_pred.split(num_bbox_per_img, 0)
                    aug_masks.append([mask.sigmoid().cpu().numpy() for mask in mask_pred])
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append([[] for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_mask = merge_aug_masks(aug_mask, [[img_metas[i]]] * self.num_stages, rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(merged_mask, _bboxes[i], det_labels[i], rcnn_test_cfg, ori_shapes[i], scale_factors[i], rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results
        if self.with_mask:
            results = list(zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']
        return results

    def aug_test(self, img_feats, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        if self.with_semantic:
            semantic_feats = [self.semantic_head(feat)[1] for feat in img_feats]
        else:
            semantic_feats = [None] * len(img_metas)
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta, semantic in zip(img_feats, img_metas, semantic_feats):
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape, scale_factor, flip, flip_direction)
            ms_scores = []
            rois = bbox2roi([proposals])
            if rois.shape[0] == 0:
                aug_bboxes.append(rois.new_zeros(0, 4))
                aug_scores.append(rois.new_zeros(0, 1))
                continue
            for i in range(self.num_stages):
                bbox_head = self.bbox_head[i]
                bbox_results = self._bbox_forward(i, x, rois, semantic_feat=semantic)
                ms_scores.append(bbox_results['cls_score'])
                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'].argmax(dim=1)
                    rois = bbox_head.regress_by_class(rois, bbox_label, bbox_results['bbox_pred'], img_meta[0])
            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(rois, cls_score, bbox_results['bbox_pred'], img_shape, scale_factor, rescale=False, cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        merged_bboxes, merged_scores = merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores, rcnn_test_cfg.score_thr, rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
        bbox_result = bbox2result(det_bboxes, det_labels, self.bbox_head[-1].num_classes)
        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[] for _ in range(self.mask_head[-1].num_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta, semantic in zip(img_feats, img_metas, semantic_feats):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape, scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    mask_feats = self.mask_roi_extractor[-1](x[:len(self.mask_roi_extractor[-1].featmap_strides)], mask_rois)
                    if self.with_semantic:
                        semantic_feat = semantic
                        mask_semantic_feat = self.semantic_roi_extractor([semantic_feat], mask_rois)
                        if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                            mask_semantic_feat = F.adaptive_avg_pool2d(mask_semantic_feat, mask_feats.shape[-2:])
                        mask_feats = mask_feats + mask_semantic_feat
                    last_feat = None
                    for i in range(self.num_stages):
                        mask_head = self.mask_head[i]
                        if self.mask_info_flow:
                            mask_pred, last_feat = mask_head(mask_feats, last_feat)
                        else:
                            mask_pred = mask_head(mask_feats)
                        aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas, self.test_cfg)
                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(merged_masks, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor=1.0, rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]

@HEADS.register_module()
class CascadeRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(self, num_stages, stage_loss_weights, bbox_roi_extractor=None, bbox_head=None, mask_roi_extractor=None, mask_head=None, shared_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, 'Shared head is not supported in Cascade RCNN anymore'
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        super(CascadeRoIHead, self).__init__(bbox_roi_extractor=bbox_roi_extractor, bbox_head=bbox_head, mask_roi_extractor=mask_roi_extractor, mask_head=mask_head, shared_head=shared_head, train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained, init_cfg=init_cfg)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = ModuleList()
        self.bbox_head = ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [bbox_roi_extractor for _ in range(self.num_stages)]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [mask_roi_extractor for _ in range(self.num_stages)]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(build_sampler(rcnn_train_cfg.sampler, context=self))

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                outs = outs + (bbox_results['cls_score'], bbox_results['bbox_pred'])
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                outs = outs + (mask_results['mask_pred'],)
        return outs

    def _bbox_forward(self, stage, x, rois):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred = bbox_head(bbox_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_targets = self.bbox_head[stage].get_targets(sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'], bbox_results['bbox_pred'], rois, *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward(self, stage, x, rois):
        """Mask head forward function used in both training and testing."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs], rois)
        mask_pred = mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _mask_forward_train(self, stage, x, sampling_results, gt_masks, rcnn_train_cfg, bbox_feats=None):
        """Run forward function and calculate loss for mask head in
        training."""
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_results = self._mask_forward(stage, x, pos_rois)
        mask_targets = self.mask_head[stage].get_targets(sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'], mask_targets, pos_labels)
        mask_results.update(loss_mask=loss_mask)
        return mask_results

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j], gt_labels[j])
                    sampling_result = bbox_sampler.sample(assign_result, proposal_list[j], gt_bboxes[j], gt_labels[j], feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)
            bbox_results = self._bbox_forward_train(i, x, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = value * lw if 'loss' in name else value
            if self.with_mask:
                mask_results = self._mask_forward_train(i, x, sampling_results, gt_masks, rcnn_train_cfg, bbox_results['bbox_feats'])
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = value * lw if 'loss' in name else value
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(cls_score)
                    if cls_score.numel() == 0:
                        break
                    roi_labels = torch.where(roi_labels == self.bbox_head[i].num_classes, cls_score[:, :-1].argmax(1), roi_labels)
                    proposal_list = self.bbox_head[i].refine_bboxes(bbox_results['rois'], roi_labels, bbox_results['bbox_pred'], pos_is_gts, img_metas)
        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple((meta['img_shape'] for meta in img_metas))
        ori_shapes = tuple((meta['ori_shape'] for meta in img_metas))
        scale_factors = tuple((meta['scale_factor'] for meta in img_metas))
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg
        rois = bbox2roi(proposal_list)
        if rois.shape[0] == 0:
            bbox_results = [[np.zeros((0, 5), dtype=np.float32) for _ in range(self.bbox_head[-1].num_classes)]] * num_imgs
            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)] for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results
            return results
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple((len(proposals) for proposals in proposal_list))
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)
            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [self.bbox_head[i].loss_cls.get_activation(s) for s in cls_score]
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refined_rois = self.bbox_head[i].regress_by_class(rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)
        cls_score = [sum([score[i] for score in ms_scores]) / float(len(ms_scores)) for i in range(num_imgs)]
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(rois[i], cls_score[i], bbox_pred[i], img_shapes[i], scale_factors[i], rescale=rescale, cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        bbox_results = [bbox2result(det_bboxes[i], det_labels[i], self.bbox_head[-1].num_classes) for i in range(num_imgs)]
        ms_bbox_result['ensemble'] = bbox_results
        if self.with_mask:
            if all((det_bbox.shape[0] == 0 for det_bbox in det_bboxes)):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)] for _ in range(num_imgs)]
            else:
                if rescale and (not isinstance(scale_factors[0], float)):
                    scale_factors = [torch.from_numpy(scale_factor).to(det_bboxes[0].device) for scale_factor in scale_factors]
                _bboxes = [det_bboxes[i][:, :4] * scale_factors[i] if rescale else det_bboxes[i][:, :4] for i in range(len(det_bboxes))]
                mask_rois = bbox2roi(_bboxes)
                num_mask_rois_per_img = tuple((_bbox.size(0) for _bbox in _bboxes))
                aug_masks = []
                for i in range(self.num_stages):
                    mask_results = self._mask_forward(i, x, mask_rois)
                    mask_pred = mask_results['mask_pred']
                    mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                    aug_masks.append([m.sigmoid().cpu().detach().numpy() for m in mask_pred])
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append([[] for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_masks = merge_aug_masks(aug_mask, [[img_metas[i]]] * self.num_stages, rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(merged_masks, _bboxes[i], det_labels[i], rcnn_test_cfg, ori_shapes[i], scale_factors[i], rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results
        if self.with_mask:
            results = list(zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']
        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(features, img_metas):
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape, scale_factor, flip, flip_direction)
            ms_scores = []
            rois = bbox2roi([proposals])
            if rois.shape[0] == 0:
                aug_bboxes.append(rois.new_zeros(0, 4))
                aug_scores.append(rois.new_zeros(0, 1))
                continue
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                ms_scores.append(bbox_results['cls_score'])
                if i < self.num_stages - 1:
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(cls_score)
                    bbox_label = cls_score[:, :-1].argmax(dim=1)
                    rois = self.bbox_head[i].regress_by_class(rois, bbox_label, bbox_results['bbox_pred'], img_meta[0])
            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(rois, cls_score, bbox_results['bbox_pred'], img_shape, scale_factor, rescale=False, cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        merged_bboxes, merged_scores = merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores, rcnn_test_cfg.score_thr, rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
        bbox_result = bbox2result(det_bboxes, det_labels, self.bbox_head[-1].num_classes)
        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[] for _ in range(self.mask_head[-1].num_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta in zip(features, img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape, scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    for i in range(self.num_stages):
                        mask_results = self._mask_forward(i, x, mask_rois)
                        aug_masks.append(mask_results['mask_pred'].sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas, self.test_cfg)
                ori_shape = img_metas[0][0]['ori_shape']
                dummy_scale_factor = np.ones(4)
                segm_result = self.mask_head[-1].get_seg_masks(merged_masks, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor=dummy_scale_factor, rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]

    def onnx_export(self, x, proposals, img_metas):
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert proposals.shape[0] == 1, 'Only support one input image while in exporting to ONNX'
        rois = proposals[..., :-1]
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]
        rois = rois.view(-1, 4)
        rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois], dim=-1)
        max_shape = img_metas[0]['img_shape_for_onnx']
        ms_scores = []
        rcnn_test_cfg = self.test_cfg
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
            cls_score = cls_score.reshape(batch_size, num_proposals_per_img, cls_score.size(-1))
            bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img, 4)
            ms_scores.append(cls_score)
            if i < self.num_stages - 1:
                assert self.bbox_head[i].reg_class_agnostic
                new_rois = self.bbox_head[i].bbox_coder.decode(rois[..., 1:], bbox_pred, max_shape=max_shape)
                rois = new_rois.reshape(-1, new_rois.shape[-1])
                rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois], dim=-1)
        cls_score = sum(ms_scores) / float(len(ms_scores))
        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img, 4)
        rois = rois.reshape(batch_size, num_proposals_per_img, -1)
        det_bboxes, det_labels = self.bbox_head[-1].onnx_export(rois, cls_score, bbox_pred, max_shape, cfg=rcnn_test_cfg)
        if not self.with_mask:
            return (det_bboxes, det_labels)
        else:
            batch_index = torch.arange(det_bboxes.size(0), device=det_bboxes.device).float().view(-1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
            rois = det_bboxes[..., :4]
            mask_rois = torch.cat([batch_index, rois], dim=-1)
            mask_rois = mask_rois.view(-1, 5)
            aug_masks = []
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                mask_pred = mask_results['mask_pred']
                aug_masks.append(mask_pred)
            max_shape = img_metas[0]['img_shape_for_onnx']
            mask_pred = sum(aug_masks) / len(aug_masks)
            segm_results = self.mask_head[-1].onnx_export(mask_pred, rois.reshape(-1, 4), det_labels.reshape(-1), self.test_cfg, max_shape)
            segm_results = segm_results.reshape(batch_size, det_bboxes.shape[1], max_shape[0], max_shape[1])
            return (det_bboxes, det_labels, segm_results)

@HEADS.register_module()
class SCNetRoIHead(CascadeRoIHead):
    """RoIHead for `SCNet <https://arxiv.org/abs/2012.10150>`_.

    Args:
        num_stages (int): number of cascade stages.
        stage_loss_weights (list): loss weight of cascade stages.
        semantic_roi_extractor (dict): config to init semantic roi extractor.
        semantic_head (dict): config to init semantic head.
        feat_relay_head (dict): config to init feature_relay_head.
        glbctx_head (dict): config to init global context head.
    """

    def __init__(self, num_stages, stage_loss_weights, semantic_roi_extractor=None, semantic_head=None, feat_relay_head=None, glbctx_head=None, **kwargs):
        super(SCNetRoIHead, self).__init__(num_stages, stage_loss_weights, **kwargs)
        assert self.with_bbox and self.with_mask
        assert not self.with_shared_head
        if semantic_head is not None:
            self.semantic_roi_extractor = build_roi_extractor(semantic_roi_extractor)
            self.semantic_head = build_head(semantic_head)
        if feat_relay_head is not None:
            self.feat_relay_head = build_head(feat_relay_head)
        if glbctx_head is not None:
            self.glbctx_head = build_head(glbctx_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.mask_head = build_head(mask_head)

    @property
    def with_semantic(self):
        """bool: whether the head has semantic head"""
        return hasattr(self, 'semantic_head') and self.semantic_head is not None

    @property
    def with_feat_relay(self):
        """bool: whether the head has feature relay head"""
        return hasattr(self, 'feat_relay_head') and self.feat_relay_head is not None

    @property
    def with_glbctx(self):
        """bool: whether the head has global context head"""
        return hasattr(self, 'glbctx_head') and self.glbctx_head is not None

    def _fuse_glbctx(self, roi_feats, glbctx_feat, rois):
        """Fuse global context feats with roi feats."""
        assert roi_feats.size(0) == rois.size(0)
        img_inds = torch.unique(rois[:, 0].cpu(), sorted=True).long()
        fused_feats = torch.zeros_like(roi_feats)
        for img_id in img_inds:
            inds = rois[:, 0] == img_id.item()
            fused_feats[inds] = roi_feats[inds] + glbctx_feat[img_id]
        return fused_feats

    def _slice_pos_feats(self, feats, sampling_results):
        """Get features from pos rois."""
        num_rois = [res.bboxes.size(0) for res in sampling_results]
        num_pos_rois = [res.pos_bboxes.size(0) for res in sampling_results]
        inds = torch.zeros(sum(num_rois), dtype=torch.bool)
        start = 0
        for i in range(len(num_rois)):
            start = 0 if i == 0 else start + num_rois[i - 1]
            stop = start + num_pos_rois[i]
            inds[start:stop] = 1
        sliced_feats = feats[inds]
        return sliced_feats

    def _bbox_forward(self, stage, x, rois, semantic_feat=None, glbctx_feat=None):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:len(bbox_roi_extractor.featmap_strides)], rois)
        if self.with_semantic and semantic_feat is not None:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat], rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = adaptive_avg_pool2d(bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats = bbox_feats + bbox_semantic_feat
        if self.with_glbctx and glbctx_feat is not None:
            bbox_feats = self._fuse_glbctx(bbox_feats, glbctx_feat, rois)
        cls_score, bbox_pred, relayed_feat = bbox_head(bbox_feats, return_shared_feat=True)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred, relayed_feat=relayed_feat)
        return bbox_results

    def _mask_forward(self, x, rois, semantic_feat=None, glbctx_feat=None, relayed_feat=None):
        """Mask head forward function used in both training and testing."""
        mask_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois)
        if self.with_semantic and semantic_feat is not None:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat], rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats = mask_feats + mask_semantic_feat
        if self.with_glbctx and glbctx_feat is not None:
            mask_feats = self._fuse_glbctx(mask_feats, glbctx_feat, rois)
        if self.with_feat_relay and relayed_feat is not None:
            mask_feats = mask_feats + relayed_feat
        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg, semantic_feat=None, glbctx_feat=None):
        """Run forward function and calculate loss for box head in training."""
        bbox_head = self.bbox_head[stage]
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois, semantic_feat=semantic_feat, glbctx_feat=glbctx_feat)
        bbox_targets = bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        loss_bbox = bbox_head.loss(bbox_results['cls_score'], bbox_results['bbox_pred'], rois, *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, gt_masks, rcnn_train_cfg, semantic_feat=None, glbctx_feat=None, relayed_feat=None):
        """Run forward function and calculate loss for mask head in
        training."""
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_results = self._mask_forward(x, pos_rois, semantic_feat=semantic_feat, glbctx_feat=glbctx_feat, relayed_feat=relayed_feat)
        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'], mask_targets, pos_labels)
        mask_results = loss_mask
        return mask_results

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, gt_semantic_seg=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposal_list (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None, list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None, Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
            gt_semantic_seg (None, list[Tensor]): semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
            losses['loss_semantic_seg'] = loss_seg
        else:
            semantic_feat = None
        if self.with_glbctx:
            mc_pred, glbctx_feat = self.glbctx_head(x)
            loss_glbctx = self.glbctx_head.loss(mc_pred, gt_labels)
            losses['loss_glbctx'] = loss_glbctx
        else:
            glbctx_feat = None
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]
            sampling_results = []
            bbox_assigner = self.bbox_assigner[i]
            bbox_sampler = self.bbox_sampler[i]
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j], gt_labels[j])
                sampling_result = bbox_sampler.sample(assign_result, proposal_list[j], gt_bboxes[j], gt_labels[j], feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
            bbox_results = self._bbox_forward_train(i, x, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg, semantic_feat, glbctx_feat)
            roi_labels = bbox_results['bbox_targets'][0]
            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = value * lw if 'loss' in name else value
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(bbox_results['rois'], roi_labels, bbox_results['bbox_pred'], pos_is_gts, img_metas)
        if self.with_feat_relay:
            relayed_feat = self._slice_pos_feats(bbox_results['relayed_feat'], sampling_results)
            relayed_feat = self.feat_relay_head(relayed_feat)
        else:
            relayed_feat = None
        mask_results = self._mask_forward_train(x, sampling_results, gt_masks, rcnn_train_cfg, semantic_feat, glbctx_feat, relayed_feat)
        mask_lw = sum(self.stage_loss_weights)
        losses['loss_mask'] = mask_lw * mask_results['loss_mask']
        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None
        if self.with_glbctx:
            mc_pred, glbctx_feat = self.glbctx_head(x)
        else:
            glbctx_feat = None
        num_imgs = len(proposal_list)
        img_shapes = tuple((meta['img_shape'] for meta in img_metas))
        ori_shapes = tuple((meta['ori_shape'] for meta in img_metas))
        scale_factors = tuple((meta['scale_factor'] for meta in img_metas))
        ms_scores = []
        rcnn_test_cfg = self.test_cfg
        rois = bbox2roi(proposal_list)
        if rois.shape[0] == 0:
            bbox_results = [[np.zeros((0, 5), dtype=np.float32) for _ in range(self.bbox_head[-1].num_classes)]] * num_imgs
            if self.with_mask:
                mask_classes = self.mask_head.num_classes
                segm_results = [[[] for _ in range(mask_classes)] for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results
            return results
        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            bbox_results = self._bbox_forward(i, x, rois, semantic_feat=semantic_feat, glbctx_feat=glbctx_feat)
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple((len(p) for p in proposal_list))
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            ms_scores.append(cls_score)
            if i < self.num_stages - 1:
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refine_rois = bbox_head.regress_by_class(rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refine_rois)
                rois = torch.cat(refine_rois_list)
        cls_score = [sum([score[i] for score in ms_scores]) / float(len(ms_scores)) for i in range(num_imgs)]
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(rois[i], cls_score[i], bbox_pred[i], img_shapes[i], scale_factors[i], rescale=rescale, cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        det_bbox_results = [bbox2result(det_bboxes[i], det_labels[i], self.bbox_head[-1].num_classes) for i in range(num_imgs)]
        if self.with_mask:
            if all((det_bbox.shape[0] == 0 for det_bbox in det_bboxes)):
                mask_classes = self.mask_head.num_classes
                det_segm_results = [[[] for _ in range(mask_classes)] for _ in range(num_imgs)]
            else:
                if rescale and (not isinstance(scale_factors[0], float)):
                    scale_factors = [torch.from_numpy(scale_factor).to(det_bboxes[0].device) for scale_factor in scale_factors]
                _bboxes = [det_bboxes[i][:, :4] * scale_factors[i] if rescale else det_bboxes[i] for i in range(num_imgs)]
                mask_rois = bbox2roi(_bboxes)
                bbox_results = self._bbox_forward(-1, x, mask_rois, semantic_feat=semantic_feat, glbctx_feat=glbctx_feat)
                relayed_feat = bbox_results['relayed_feat']
                relayed_feat = self.feat_relay_head(relayed_feat)
                mask_results = self._mask_forward(x, mask_rois, semantic_feat=semantic_feat, glbctx_feat=glbctx_feat, relayed_feat=relayed_feat)
                mask_pred = mask_results['mask_pred']
                num_bbox_per_img = tuple((len(_bbox) for _bbox in _bboxes))
                mask_preds = mask_pred.split(num_bbox_per_img, 0)
                det_segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        det_segm_results.append([[] for _ in range(self.mask_head.num_classes)])
                    else:
                        segm_result = self.mask_head.get_seg_masks(mask_preds[i], _bboxes[i], det_labels[i], self.test_cfg, ori_shapes[i], scale_factors[i], rescale)
                        det_segm_results.append(segm_result)
        if self.with_mask:
            return list(zip(det_bbox_results, det_segm_results))
        else:
            return det_bbox_results

    def aug_test(self, img_feats, proposal_list, img_metas, rescale=False):
        if self.with_semantic:
            semantic_feats = [self.semantic_head(feat)[1] for feat in img_feats]
        else:
            semantic_feats = [None] * len(img_metas)
        if self.with_glbctx:
            glbctx_feats = [self.glbctx_head(feat)[1] for feat in img_feats]
        else:
            glbctx_feats = [None] * len(img_metas)
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta, semantic_feat, glbctx_feat in zip(img_feats, img_metas, semantic_feats, glbctx_feats):
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape, scale_factor, flip)
            ms_scores = []
            rois = bbox2roi([proposals])
            if rois.shape[0] == 0:
                aug_bboxes.append(rois.new_zeros(0, 4))
                aug_scores.append(rois.new_zeros(0, 1))
                continue
            for i in range(self.num_stages):
                bbox_head = self.bbox_head[i]
                bbox_results = self._bbox_forward(i, x, rois, semantic_feat=semantic_feat, glbctx_feat=glbctx_feat)
                ms_scores.append(bbox_results['cls_score'])
                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'].argmax(dim=1)
                    rois = bbox_head.regress_by_class(rois, bbox_label, bbox_results['bbox_pred'], img_meta[0])
            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(rois, cls_score, bbox_results['bbox_pred'], img_shape, scale_factor, rescale=False, cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        merged_bboxes, merged_scores = merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores, rcnn_test_cfg.score_thr, rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
        det_bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head[-1].num_classes)
        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                det_segm_results = [[] for _ in range(self.mask_head.num_classes)]
            else:
                aug_masks = []
                for x, img_meta, semantic_feat, glbctx_feat in zip(img_feats, img_metas, semantic_feats, glbctx_feats):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape, scale_factor, flip)
                    mask_rois = bbox2roi([_bboxes])
                    bbox_results = self._bbox_forward(-1, x, mask_rois, semantic_feat=semantic_feat, glbctx_feat=glbctx_feat)
                    relayed_feat = bbox_results['relayed_feat']
                    relayed_feat = self.feat_relay_head(relayed_feat)
                    mask_results = self._mask_forward(x, mask_rois, semantic_feat=semantic_feat, glbctx_feat=glbctx_feat, relayed_feat=relayed_feat)
                    mask_pred = mask_results['mask_pred']
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks, img_metas, self.test_cfg)
                ori_shape = img_metas[0][0]['ori_shape']
                det_segm_results = self.mask_head.get_seg_masks(merged_masks, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor=1.0, rescale=False)
            return [(det_bbox_results, det_segm_results)]
        else:
            return [det_bbox_results]

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

def test_flip_tensor():
    img = np.random.random((1, 3, 10, 10))
    src_tensor = torch.from_numpy(img)
    with pytest.raises(AssertionError):
        flip_tensor(src_tensor, 'flip')
    with pytest.raises(AssertionError):
        flip_tensor(src_tensor[0], 'vertical')
    hfilp_tensor = flip_tensor(src_tensor, 'horizontal')
    expected_hflip_tensor = torch.from_numpy(img[..., ::-1, :].copy())
    expected_hflip_tensor.allclose(hfilp_tensor)
    vfilp_tensor = flip_tensor(src_tensor, 'vertical')
    expected_vflip_tensor = torch.from_numpy(img[..., ::-1].copy())
    expected_vflip_tensor.allclose(vfilp_tensor)
    diag_filp_tensor = flip_tensor(src_tensor, 'diagonal')
    expected_diag_filp_tensor = torch.from_numpy(img[..., ::-1, ::-1].copy())
    expected_diag_filp_tensor.allclose(diag_filp_tensor)

