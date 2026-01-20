# Cluster 74

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

def merge_aug_proposals(aug_proposals, img_metas, cfg):
    """Merge augmented proposals (multiscale, flip, etc.)

    Args:
        aug_proposals (list[Tensor]): proposals from different testing
            schemes, shape (n, 5). Note that they are not rescaled to the
            original image size.

        img_metas (list[dict]): list of image info dict where each dict has:
            'img_shape', 'scale_factor', 'flip', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `mmdet/datasets/pipelines/formatting.py:Collect`.

        cfg (dict): rpn test config.

    Returns:
        Tensor: shape (n, 4), proposals corresponding to original image scale.
    """
    cfg = copy.deepcopy(cfg)
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
    recovered_proposals = []
    for proposals, img_info in zip(aug_proposals, img_metas):
        img_shape = img_info['img_shape']
        scale_factor = img_info['scale_factor']
        flip = img_info['flip']
        flip_direction = img_info['flip_direction']
        _proposals = proposals.clone()
        _proposals[:, :4] = bbox_mapping_back(_proposals[:, :4], img_shape, scale_factor, flip, flip_direction)
        recovered_proposals.append(_proposals)
    aug_proposals = torch.cat(recovered_proposals, dim=0)
    merged_proposals, _ = nms(aug_proposals[:, :4].contiguous(), aug_proposals[:, -1].contiguous(), cfg.nms.iou_threshold)
    scores = merged_proposals[:, 4]
    _, order = scores.sort(0, descending=True)
    num = min(cfg.max_per_img, merged_proposals.shape[0])
    order = order[:num]
    merged_proposals = merged_proposals[order, :]
    return merged_proposals

