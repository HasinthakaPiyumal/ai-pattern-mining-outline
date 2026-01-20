# Cluster 78

@HEADS.register_module()
class CascadeRPNHead(BaseDenseHead):
    """The CascadeRPNHead will predict more accurate region proposals, which is
    required for two-stage detectors (such as Fast/Faster R-CNN). CascadeRPN
    consists of a sequence of RPNStage to progressively improve the accuracy of
    the detected proposals.

    More details can be found in ``https://arxiv.org/abs/1909.06720``.

    Args:
        num_stages (int): number of CascadeRPN stages.
        stages (list[dict]): list of configs to build the stages.
        train_cfg (list[dict]): list of configs at training time each stage.
        test_cfg (dict): config at testing time.
    """

    def __init__(self, num_stages, stages, train_cfg, test_cfg, init_cfg=None):
        super(CascadeRPNHead, self).__init__(init_cfg)
        assert num_stages == len(stages)
        self.num_stages = num_stages
        self.stages = ModuleList()
        for i in range(len(stages)):
            train_cfg_i = train_cfg[i] if train_cfg is not None else None
            stages[i].update(train_cfg=train_cfg_i)
            stages[i].update(test_cfg=test_cfg)
            self.stages.append(build_head(stages[i]))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(self):
        """loss() is implemented in StageCascadeRPNHead."""
        pass

    def get_bboxes(self):
        """get_bboxes() is implemented in StageCascadeRPNHead."""
        pass

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=None, proposal_cfg=None):
        """Forward train function."""
        assert gt_labels is None, 'RPN does not require gt_labels'
        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        device = x[0].device
        anchor_list, valid_flag_list = self.stages[0].get_anchors(featmap_sizes, img_metas, device=device)
        losses = dict()
        for i in range(self.num_stages):
            stage = self.stages[i]
            if stage.adapt_cfg['type'] == 'offset':
                offset_list = stage.anchor_offset(anchor_list, stage.anchor_strides, featmap_sizes)
            else:
                offset_list = None
            x, cls_score, bbox_pred = stage(x, offset_list)
            rpn_loss_inputs = (anchor_list, valid_flag_list, cls_score, bbox_pred, gt_bboxes, img_metas)
            stage_loss = stage.loss(*rpn_loss_inputs)
            for name, value in stage_loss.items():
                losses['s{}.{}'.format(i, name)] = value
            if i < self.num_stages - 1:
                anchor_list = stage.refine_bboxes(anchor_list, bbox_pred, img_metas)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.stages[-1].get_bboxes(anchor_list, cls_score, bbox_pred, img_metas, self.test_cfg)
            return (losses, proposal_list)

    def simple_test_rpn(self, x, img_metas):
        """Simple forward test function."""
        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        device = x[0].device
        anchor_list, _ = self.stages[0].get_anchors(featmap_sizes, img_metas, device=device)
        for i in range(self.num_stages):
            stage = self.stages[i]
            if stage.adapt_cfg['type'] == 'offset':
                offset_list = stage.anchor_offset(anchor_list, stage.anchor_strides, featmap_sizes)
            else:
                offset_list = None
            x, cls_score, bbox_pred = stage(x, offset_list)
            if i < self.num_stages - 1:
                anchor_list = stage.refine_bboxes(anchor_list, bbox_pred, img_metas)
        proposal_list = self.stages[-1].get_bboxes(anchor_list, cls_score, bbox_pred, img_metas, self.test_cfg)
        return proposal_list

    def aug_test_rpn(self, x, img_metas):
        """Augmented forward test function."""
        raise NotImplementedError('CascadeRPNHead does not support test-time augmentation')

def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)

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

def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)

def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)

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
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self, backbone, neck=None, rpn_head=None, roi_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)
        if roi_head is not None:
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

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
        outs = ()
        x = self.extract_feat(img)
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None, **kwargs):
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

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)
        losses = dict()
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=gt_bboxes_ignore, proposal_cfg=proposal_cfg, **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)
        return losses

    async def async_simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(x, img_meta)
        else:
            proposal_list = proposals
        return await self.roi_head.async_simple_test(x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        return self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(f'{self.__class__.__name__} can not be exported to ONNX. Please refer to the list of supported models,https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx')

@DETECTORS.register_module()
class SingleStageInstanceSegmentor(BaseDetector):
    """Base class for single-stage instance segmentors."""

    def __init__(self, backbone, neck=None, bbox_head=None, mask_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead')
            backbone.pretrained = pretrained
        super(SingleStageInstanceSegmentor, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck = None
        if bbox_head is not None:
            bbox_head.update(train_cfg=copy.deepcopy(train_cfg))
            bbox_head.update(test_cfg=copy.deepcopy(test_cfg))
            self.bbox_head = build_head(bbox_head)
        else:
            self.bbox_head = None
        assert mask_head, f'`mask_head` must be implemented in {self.__class__.__name__}'
        mask_head.update(train_cfg=copy.deepcopy(train_cfg))
        mask_head.update(test_cfg=copy.deepcopy(test_cfg))
        self.mask_head = build_head(mask_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone and neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        raise NotImplementedError(f'`forward_dummy` is not implemented in {self.__class__.__name__}')

    def forward_train(self, img, img_metas, gt_masks, gt_labels, gt_bboxes=None, gt_bboxes_ignore=None, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (B, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_masks (list[:obj:`BitmapMasks`] | None) : The segmentation
                masks for each box.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes (list[Tensor]): Each item is the truth boxes
                of each image in [tl_x, tl_y, br_x, br_y] format.
                Default: None.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        gt_masks = [gt_mask.to_tensor(dtype=torch.bool, device=img.device) for gt_mask in gt_masks]
        x = self.extract_feat(img)
        losses = dict()
        if self.bbox_head:
            bbox_head_preds = self.bbox_head(x)
            det_losses, positive_infos = self.bbox_head.loss(*bbox_head_preds, gt_bboxes=gt_bboxes, gt_labels=gt_labels, gt_masks=gt_masks, img_metas=img_metas, gt_bboxes_ignore=gt_bboxes_ignore, **kwargs)
            losses.update(det_losses)
        else:
            positive_infos = None
        mask_loss = self.mask_head.forward_train(x, gt_labels, gt_masks, img_metas, positive_infos=positive_infos, gt_bboxes=gt_bboxes, gt_bboxes_ignore=gt_bboxes_ignore, **kwargs)
        assert not set(mask_loss.keys()) & set(losses.keys())
        losses.update(mask_loss)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (B, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list(tuple): Formatted bbox and mask results of multiple                 images. The outer list corresponds to each image.                 Each tuple contains two type of results of single image:

                - bbox_results (list[np.ndarray]): BBox results of
                  single image. The list corresponds to each class.
                  each ndarray has a shape (N, 5), N is the number of
                  bboxes with this category, and last dimension
                  5 arrange as (x1, y1, x2, y2, scores).
                - mask_results (list[np.ndarray]): Mask results of
                  single image. The list corresponds to each class.
                  each ndarray has a shape (N, img_h, img_w), N
                  is the number of masks with this category.
        """
        feat = self.extract_feat(img)
        if self.bbox_head:
            outs = self.bbox_head(feat)
            results_list = self.bbox_head.get_results(*outs, img_metas=img_metas, cfg=self.test_cfg, rescale=rescale)
        else:
            results_list = None
        results_list = self.mask_head.simple_test(feat, img_metas, rescale=rescale, instances_list=results_list)
        format_results_list = []
        for results in results_list:
            format_results_list.append(self.format_results(results))
        return format_results_list

    def format_results(self, results):
        """Format the model predictions according to the interface with
        dataset.

        Args:
            results (:obj:`InstanceData`): Processed
                results of single images. Usually contains
                following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,)
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).

        Returns:
            tuple: Formatted bbox and mask results.. It contains two items:

                - bbox_results (list[np.ndarray]): BBox results of
                  single image. The list corresponds to each class.
                  each ndarray has a shape (N, 5), N is the number of
                  bboxes with this category, and last dimension
                  5 arrange as (x1, y1, x2, y2, scores).
                - mask_results (list[np.ndarray]): Mask results of
                  single image. The list corresponds to each class.
                  each ndarray has shape (N, img_h, img_w), N
                  is the number of masks with this category.
        """
        data_keys = results.keys()
        assert 'scores' in data_keys
        assert 'labels' in data_keys
        assert 'masks' in data_keys, 'results should contain masks when format the results '
        mask_results = [[] for _ in range(self.mask_head.num_classes)]
        num_masks = len(results)
        if num_masks == 0:
            bbox_results = [np.zeros((0, 5), dtype=np.float32) for _ in range(self.mask_head.num_classes)]
            return (bbox_results, mask_results)
        labels = results.labels.detach().cpu().numpy()
        if 'bboxes' not in results:
            results.bboxes = results.scores.new_zeros(len(results), 4)
        det_bboxes = torch.cat([results.bboxes, results.scores[:, None]], dim=-1)
        det_bboxes = det_bboxes.detach().cpu().numpy()
        bbox_results = [det_bboxes[labels == i, :] for i in range(self.mask_head.num_classes)]
        masks = results.masks.detach().cpu().numpy()
        for idx in range(num_masks):
            mask = masks[idx]
            mask_results[labels[idx]].append(mask)
        return (bbox_results, mask_results)

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    def show_result(self, img, result, score_thr=0.3, bbox_color=(72, 101, 241), text_color=(72, 101, 241), mask_color=None, thickness=2, font_size=13, win_name='', show=False, wait_time=0, out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (tuple): Format bbox and mask results.
                It contains two items:

                - bbox_results (list[np.ndarray]): BBox results of
                  single image. The list corresponds to each class.
                  each ndarray has a shape (N, 5), N is the number of
                  bboxes with this category, and last dimension
                  5 arrange as (x1, y1, x2, y2, scores).
                - mask_results (list[np.ndarray]): Mask results of
                  single image. The list corresponds to each class.
                  each ndarray has shape (N, img_h, img_w), N
                  is the number of masks with this category.

            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        assert isinstance(result, tuple)
        bbox_result, mask_result = result
        bboxes = np.vstack(bbox_result)
        img = mmcv.imread(img)
        img = img.copy()
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)
        if len(labels) == 0:
            bboxes = np.zeros([0, 5])
            masks = np.zeros([0, 0, 0])
        else:
            masks = mmcv.concat_list(mask_result)
            if isinstance(masks[0], torch.Tensor):
                masks = torch.stack(masks, dim=0).detach().cpu().numpy()
            else:
                masks = np.stack(masks, axis=0)
            if bboxes[:, :4].sum() == 0:
                num_masks = len(bboxes)
                x_any = masks.any(axis=1)
                y_any = masks.any(axis=2)
                for idx in range(num_masks):
                    x = np.where(x_any[idx, :])[0]
                    y = np.where(y_any[idx, :])[0]
                    if len(x) > 0 and len(y) > 0:
                        bboxes[idx, :4] = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float32)
        if out_file is not None:
            show = False
        img = imshow_det_bboxes(img, bboxes, labels, masks, class_names=self.CLASSES, score_thr=score_thr, bbox_color=bbox_color, text_color=text_color, mask_color=mask_color, thickness=thickness, font_size=font_size, win_name=win_name, show=show, wait_time=wait_time, out_file=out_file)
        if not (show or out_file):
            return img

@DETECTORS.register_module()
class LAD(KnowledgeDistillationSingleStageDetector):
    """Implementation of `LAD <https://arxiv.org/pdf/2108.10520.pdf>`_."""

    def __init__(self, backbone, neck, bbox_head, teacher_backbone, teacher_neck, teacher_bbox_head, teacher_ckpt, eval_teacher=True, train_cfg=None, test_cfg=None, pretrained=None):
        super(KnowledgeDistillationSingleStageDetector, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)
        self.eval_teacher = eval_teacher
        self.teacher_model = nn.Module()
        self.teacher_model.backbone = build_backbone(teacher_backbone)
        if teacher_neck is not None:
            self.teacher_model.neck = build_neck(teacher_neck)
        teacher_bbox_head.update(train_cfg=train_cfg)
        teacher_bbox_head.update(test_cfg=test_cfg)
        self.teacher_model.bbox_head = build_head(teacher_bbox_head)
        if teacher_ckpt is not None:
            load_checkpoint(self.teacher_model, teacher_ckpt, map_location='cpu')

    @property
    def with_teacher_neck(self):
        """bool: whether the detector has a teacher_neck"""
        return hasattr(self.teacher_model, 'neck') and self.teacher_model.neck is not None

    def extract_teacher_feat(self, img):
        """Directly extract teacher features from the backbone+neck."""
        x = self.teacher_model.backbone(img)
        if self.with_teacher_neck:
            x = self.teacher_model.neck(x)
        return x

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
        with torch.no_grad():
            x_teacher = self.extract_teacher_feat(img)
            outs_teacher = self.teacher_model.bbox_head(x_teacher)
            label_assignment_results = self.teacher_model.bbox_head.get_label_assignment(*outs_teacher, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, label_assignment_results, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        return losses

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

def build_roi_extractor(cfg):
    """Build roi extractor."""
    return ROI_EXTRACTORS.build(cfg)

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

@NECKS.register_module()
class RFP(FPN):
    """RFP (Recursive Feature Pyramid)

    This is an implementation of RFP in `DetectoRS
    <https://arxiv.org/pdf/2006.02334.pdf>`_. Different from standard FPN, the
    input of RFP should be multi level features along with origin input image
    of backbone.

    Args:
        rfp_steps (int): Number of unrolled steps of RFP.
        rfp_backbone (dict): Configuration of the backbone for RFP.
        aspp_out_channels (int): Number of output channels of ASPP module.
        aspp_dilations (tuple[int]): Dilation rates of four branches.
            Default: (1, 3, 6, 1)
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self, rfp_steps, rfp_backbone, aspp_out_channels, aspp_dilations=(1, 3, 6, 1), init_cfg=None, **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.rfp_steps = rfp_steps
        self.rfp_modules = ModuleList()
        for rfp_idx in range(1, rfp_steps):
            rfp_module = build_backbone(rfp_backbone)
            self.rfp_modules.append(rfp_module)
        self.rfp_aspp = ASPP(self.out_channels, aspp_out_channels, aspp_dilations)
        self.rfp_weight = nn.Conv2d(self.out_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def init_weights(self):
        for convs in [self.lateral_convs, self.fpn_convs]:
            for m in convs.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
        for rfp_idx in range(self.rfp_steps - 1):
            self.rfp_modules[rfp_idx].init_weights()
        constant_init(self.rfp_weight, 0)

    def forward(self, inputs):
        inputs = list(inputs)
        assert len(inputs) == len(self.in_channels) + 1
        img = inputs.pop(0)
        x = super().forward(tuple(inputs))
        for rfp_idx in range(self.rfp_steps - 1):
            rfp_feats = [x[0]] + list((self.rfp_aspp(x[i]) for i in range(1, len(x))))
            x_idx = self.rfp_modules[rfp_idx].rfp_forward(img, rfp_feats)
            x_idx = super().forward(x_idx)
            x_new = []
            for ft_idx in range(len(x_idx)):
                add_weight = torch.sigmoid(self.rfp_weight(x_idx[ft_idx]))
                x_new.append(add_weight * x_idx[ft_idx] + (1 - add_weight) * x[ft_idx])
            x = x_new
        return x

def test_retina_anchor():
    from mmdet.models import build_head
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    bbox_head = dict(type='RetinaSepBNHead', num_classes=4, num_ins=5, in_channels=4, stacked_convs=1, feat_channels=4, anchor_generator=dict(type='AnchorGenerator', octave_base_scale=4, scales_per_octave=3, ratios=[0.5, 1.0, 2.0], strides=[8, 16, 32, 64, 128]), bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]))
    retina_head = build_head(bbox_head)
    assert retina_head.anchor_generator is not None
    featmap_sizes = [(80, 80), (40, 40), (20, 20), (10, 10), (5, 5)]
    expected_base_anchors = [torch.Tensor([[-22.6274, -11.3137, 22.6274, 11.3137], [-28.5088, -14.2544, 28.5088, 14.2544], [-35.9188, -17.9594, 35.9188, 17.9594], [-16.0, -16.0, 16.0, 16.0], [-20.1587, -20.1587, 20.1587, 20.1587], [-25.3984, -25.3984, 25.3984, 25.3984], [-11.3137, -22.6274, 11.3137, 22.6274], [-14.2544, -28.5088, 14.2544, 28.5088], [-17.9594, -35.9188, 17.9594, 35.9188]]), torch.Tensor([[-45.2548, -22.6274, 45.2548, 22.6274], [-57.0175, -28.5088, 57.0175, 28.5088], [-71.8376, -35.9188, 71.8376, 35.9188], [-32.0, -32.0, 32.0, 32.0], [-40.3175, -40.3175, 40.3175, 40.3175], [-50.7968, -50.7968, 50.7968, 50.7968], [-22.6274, -45.2548, 22.6274, 45.2548], [-28.5088, -57.0175, 28.5088, 57.0175], [-35.9188, -71.8376, 35.9188, 71.8376]]), torch.Tensor([[-90.5097, -45.2548, 90.5097, 45.2548], [-114.035, -57.0175, 114.035, 57.0175], [-143.6751, -71.8376, 143.6751, 71.8376], [-64.0, -64.0, 64.0, 64.0], [-80.6349, -80.6349, 80.6349, 80.6349], [-101.5937, -101.5937, 101.5937, 101.5937], [-45.2548, -90.5097, 45.2548, 90.5097], [-57.0175, -114.035, 57.0175, 114.035], [-71.8376, -143.6751, 71.8376, 143.6751]]), torch.Tensor([[-181.0193, -90.5097, 181.0193, 90.5097], [-228.0701, -114.035, 228.0701, 114.035], [-287.3503, -143.6751, 287.3503, 143.6751], [-128.0, -128.0, 128.0, 128.0], [-161.2699, -161.2699, 161.2699, 161.2699], [-203.1873, -203.1873, 203.1873, 203.1873], [-90.5097, -181.0193, 90.5097, 181.0193], [-114.035, -228.0701, 114.035, 228.0701], [-143.6751, -287.3503, 143.6751, 287.3503]]), torch.Tensor([[-362.0387, -181.0193, 362.0387, 181.0193], [-456.1401, -228.0701, 456.1401, 228.0701], [-574.7006, -287.3503, 574.7006, 287.3503], [-256.0, -256.0, 256.0, 256.0], [-322.5398, -322.5398, 322.5398, 322.5398], [-406.3747, -406.3747, 406.3747, 406.3747], [-181.0193, -362.0387, 181.0193, 362.0387], [-228.0701, -456.1401, 228.0701, 456.1401], [-287.3503, -574.7006, 287.3503, 574.7006]])]
    base_anchors = retina_head.anchor_generator.base_anchors
    for i, base_anchor in enumerate(base_anchors):
        assert base_anchor.allclose(expected_base_anchors[i])
    expected_valid_pixels = [57600, 14400, 3600, 900, 225]
    multi_level_valid_flags = retina_head.anchor_generator.valid_flags(featmap_sizes, (640, 640), device)
    for i, single_level_valid_flag in enumerate(multi_level_valid_flags):
        assert single_level_valid_flag.sum() == expected_valid_pixels[i]
    assert retina_head.anchor_generator.num_base_anchors == [9, 9, 9, 9, 9]
    anchors = retina_head.anchor_generator.grid_anchors(featmap_sizes, device)
    assert len(anchors) == 5

def test_guided_anchor():
    from mmdet.models import build_head
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    bbox_head = dict(type='GARetinaHead', num_classes=8, in_channels=4, stacked_convs=1, feat_channels=4, approx_anchor_generator=dict(type='AnchorGenerator', octave_base_scale=4, scales_per_octave=3, ratios=[0.5, 1.0, 2.0], strides=[8, 16, 32, 64, 128]), square_anchor_generator=dict(type='AnchorGenerator', ratios=[1.0], scales=[4], strides=[8, 16, 32, 64, 128]))
    ga_retina_head = build_head(bbox_head)
    assert ga_retina_head.approx_anchor_generator is not None
    featmap_sizes = [(100, 152), (50, 76), (25, 38), (13, 19), (7, 10)]
    expected_approxs = [torch.Tensor([[-22.6274, -11.3137, 22.6274, 11.3137], [-28.5088, -14.2544, 28.5088, 14.2544], [-35.9188, -17.9594, 35.9188, 17.9594], [-16.0, -16.0, 16.0, 16.0], [-20.1587, -20.1587, 20.1587, 20.1587], [-25.3984, -25.3984, 25.3984, 25.3984], [-11.3137, -22.6274, 11.3137, 22.6274], [-14.2544, -28.5088, 14.2544, 28.5088], [-17.9594, -35.9188, 17.9594, 35.9188]]), torch.Tensor([[-45.2548, -22.6274, 45.2548, 22.6274], [-57.0175, -28.5088, 57.0175, 28.5088], [-71.8376, -35.9188, 71.8376, 35.9188], [-32.0, -32.0, 32.0, 32.0], [-40.3175, -40.3175, 40.3175, 40.3175], [-50.7968, -50.7968, 50.7968, 50.7968], [-22.6274, -45.2548, 22.6274, 45.2548], [-28.5088, -57.0175, 28.5088, 57.0175], [-35.9188, -71.8376, 35.9188, 71.8376]]), torch.Tensor([[-90.5097, -45.2548, 90.5097, 45.2548], [-114.035, -57.0175, 114.035, 57.0175], [-143.6751, -71.8376, 143.6751, 71.8376], [-64.0, -64.0, 64.0, 64.0], [-80.6349, -80.6349, 80.6349, 80.6349], [-101.5937, -101.5937, 101.5937, 101.5937], [-45.2548, -90.5097, 45.2548, 90.5097], [-57.0175, -114.035, 57.0175, 114.035], [-71.8376, -143.6751, 71.8376, 143.6751]]), torch.Tensor([[-181.0193, -90.5097, 181.0193, 90.5097], [-228.0701, -114.035, 228.0701, 114.035], [-287.3503, -143.6751, 287.3503, 143.6751], [-128.0, -128.0, 128.0, 128.0], [-161.2699, -161.2699, 161.2699, 161.2699], [-203.1873, -203.1873, 203.1873, 203.1873], [-90.5097, -181.0193, 90.5097, 181.0193], [-114.035, -228.0701, 114.035, 228.0701], [-143.6751, -287.3503, 143.6751, 287.3503]]), torch.Tensor([[-362.0387, -181.0193, 362.0387, 181.0193], [-456.1401, -228.0701, 456.1401, 228.0701], [-574.7006, -287.3503, 574.7006, 287.3503], [-256.0, -256.0, 256.0, 256.0], [-322.5398, -322.5398, 322.5398, 322.5398], [-406.3747, -406.3747, 406.3747, 406.3747], [-181.0193, -362.0387, 181.0193, 362.0387], [-228.0701, -456.1401, 228.0701, 456.1401], [-287.3503, -574.7006, 287.3503, 574.7006]])]
    approxs = ga_retina_head.approx_anchor_generator.base_anchors
    for i, base_anchor in enumerate(approxs):
        assert base_anchor.allclose(expected_approxs[i])
    expected_valid_pixels = [136800, 34200, 8550, 2223, 630]
    multi_level_valid_flags = ga_retina_head.approx_anchor_generator.valid_flags(featmap_sizes, (800, 1216), device)
    for i, single_level_valid_flag in enumerate(multi_level_valid_flags):
        assert single_level_valid_flag.sum() == expected_valid_pixels[i]
    assert ga_retina_head.approx_anchor_generator.num_base_anchors == [9, 9, 9, 9, 9]
    squares = ga_retina_head.square_anchor_generator.grid_anchors(featmap_sizes, device)
    assert len(squares) == 5
    expected_squares = [torch.Tensor([[-16.0, -16.0, 16.0, 16.0]]), torch.Tensor([[-32.0, -32.0, 32.0, 32]]), torch.Tensor([[-64.0, -64.0, 64.0, 64.0]]), torch.Tensor([[-128.0, -128.0, 128.0, 128.0]]), torch.Tensor([[-256.0, -256.0, 256.0, 256.0]])]
    squares = ga_retina_head.square_anchor_generator.base_anchors
    for i, base_anchor in enumerate(squares):
        assert base_anchor.allclose(expected_squares[i])
    assert ga_retina_head.square_anchor_generator.num_base_anchors == [1, 1, 1, 1, 1]
    anchors = ga_retina_head.square_anchor_generator.grid_anchors(featmap_sizes, device)
    assert len(anchors) == 5

