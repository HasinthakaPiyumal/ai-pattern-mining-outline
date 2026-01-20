# Cluster 18

def _get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model

@DETECTORS.register_module()
class KnowledgeDistillationSingleStageDetector(SingleStageDetector):
    """Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.

    Args:
        teacher_config (str | dict): Config file path
            or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
    """

    def __init__(self, backbone, neck, bbox_head, teacher_config, teacher_ckpt=None, eval_teacher=True, train_cfg=None, test_cfg=None, pretrained=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)
        self.eval_teacher = eval_teacher
        if isinstance(teacher_config, (str, Path)):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_detector(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(self.teacher_model, teacher_ckpt, map_location='cpu')

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
        x = self.extract_feat(img)
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(img)
            out_teacher = self.teacher_model.bbox_head(teacher_x)
        losses = self.bbox_head.forward_train(x, out_teacher, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        return losses

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn('train_cfg and test_cfg is deprecated, please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, 'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, 'test_cfg specified in both outer field and model field '
    return DETECTORS.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))

@pytest.mark.parametrize('loss_bbox', [dict(type='L1Loss', loss_weight=1.0), dict(type='GHMR', mu=0.02, bins=10, momentum=0.7, loss_weight=10.0), dict(type='IoULoss', loss_weight=1.0), dict(type='BoundedIoULoss', loss_weight=1.0), dict(type='GIoULoss', loss_weight=1.0), dict(type='DIoULoss', loss_weight=1.0), dict(type='CIoULoss', loss_weight=1.0), dict(type='MSELoss', loss_weight=1.0), dict(type='SmoothL1Loss', loss_weight=1.0), dict(type='BalancedL1Loss', loss_weight=1.0)])
def test_bbox_loss_compatibility(loss_bbox):
    """Test loss_bbox compatibility.

    Using Faster R-CNN as a sample, modifying the loss function in the config
    file to verify the compatibility of Loss APIS
    """
    config_path = '_base_/models/faster_rcnn_r50_fpn.py'
    cfg_model = _get_detector_cfg(config_path)
    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    if 'IoULoss' in loss_bbox['type']:
        cfg_model.roi_head.bbox_head.reg_decoded_bbox = True
    cfg_model.roi_head.bbox_head.loss_bbox = loss_bbox
    from mmdet.models import build_detector
    detector = build_detector(cfg_model)
    loss = detector.forward(imgs, img_metas, return_loss=True, **mm_inputs)
    assert isinstance(loss, dict)
    loss, _ = detector._parse_losses(loss)
    assert float(loss.item()) > 0

@pytest.mark.parametrize('loss_cls', [dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0), dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0), dict(type='GHMC', bins=30, momentum=0.75, use_sigmoid=True, loss_weight=1.0)])
def test_cls_loss_compatibility(loss_cls):
    """Test loss_cls compatibility.

    Using Faster R-CNN as a sample, modifying the loss function in the config
    file to verify the compatibility of Loss APIS
    """
    config_path = '_base_/models/faster_rcnn_r50_fpn.py'
    cfg_model = _get_detector_cfg(config_path)
    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    cfg_model.roi_head.bbox_head.loss_cls = loss_cls
    from mmdet.models import build_detector
    detector = build_detector(cfg_model)
    loss = detector.forward(imgs, img_metas, return_loss=True, **mm_inputs)
    assert isinstance(loss, dict)
    loss, _ = detector._parse_losses(loss)
    assert float(loss.item()) > 0

def test_sparse_rcnn_forward():
    config_path = 'sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py'
    model = _get_detector_cfg(config_path)
    model = _replace_r50_with_r18(model)
    model.backbone.init_cfg = None
    from mmdet.models import build_detector
    detector = build_detector(model)
    detector.init_weights()
    input_shape = (1, 3, 100, 100)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[5])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    detector.train()
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_bboxes = [item for item in gt_bboxes]
    gt_labels = mm_inputs['gt_labels']
    gt_labels = [item for item in gt_labels]
    losses = detector.forward(imgs, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0
    detector.forward_dummy(imgs)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[0])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_bboxes = [item for item in gt_bboxes]
    gt_labels = mm_inputs['gt_labels']
    gt_labels = [item for item in gt_labels]
    losses = detector.forward(imgs, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0
    detector.eval()
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]], rescale=True, return_loss=False)
            batch_results.append(result)
    with torch.no_grad():
        detector.roi_head.simple_test([imgs[0][None, :]], torch.empty((1, 0, 4)), torch.empty((1, 100, 4)), [img_metas[0]], torch.ones((1, 4)))

def _replace_r50_with_r18(model):
    """Replace ResNet50 with ResNet18 in config."""
    model = copy.deepcopy(model)
    if model.backbone.type == 'ResNet':
        model.backbone.depth = 18
        model.backbone.base_channels = 2
        model.neck.in_channels = [2, 4, 8, 16]
    return model

def _demo_mm_inputs(input_shape=(1, 3, 300, 300), num_items=None, num_classes=10, with_semantic=False):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions

        num_items (None | List[int]):
            specifies the number of boxes in each batch item

        num_classes (int):
            number of different labels a box might have
    """
    from mmdet.core import BitmapMasks
    N, C, H, W = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    img_metas = [{'img_shape': (H, W, C), 'ori_shape': (H, W, C), 'pad_shape': (H, W, C), 'filename': '<demo>.png', 'scale_factor': np.array([1.1, 1.2, 1.1, 1.2]), 'flip': False, 'flip_direction': None} for _ in range(N)]
    gt_bboxes = []
    gt_labels = []
    gt_masks = []
    for batch_idx in range(N):
        if num_items is None:
            num_boxes = rng.randint(1, 10)
        else:
            num_boxes = num_items[batch_idx]
        cx, cy, bw, bh = rng.rand(num_boxes, 4).T
        tl_x = (cx * W - W * bw / 2).clip(0, W)
        tl_y = (cy * H - H * bh / 2).clip(0, H)
        br_x = (cx * W + W * bw / 2).clip(0, W)
        br_y = (cy * H + H * bh / 2).clip(0, H)
        boxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
        class_idxs = rng.randint(1, num_classes, size=num_boxes)
        gt_bboxes.append(torch.FloatTensor(boxes))
        gt_labels.append(torch.LongTensor(class_idxs))
    mask = np.random.randint(0, 2, (len(boxes), H, W), dtype=np.uint8)
    gt_masks.append(BitmapMasks(mask, H, W))
    mm_inputs = {'imgs': torch.FloatTensor(imgs).requires_grad_(True), 'img_metas': img_metas, 'gt_bboxes': gt_bboxes, 'gt_labels': gt_labels, 'gt_bboxes_ignore': None, 'gt_masks': gt_masks}
    if with_semantic:
        gt_semantic_seg = np.random.randint(0, num_classes, (1, 1, H // 8, W // 8), dtype=np.uint8)
        mm_inputs.update({'gt_semantic_seg': torch.ByteTensor(gt_semantic_seg)})
    return mm_inputs

def test_rpn_forward():
    model = _get_detector_cfg('rpn/rpn_r50_fpn_1x_coco.py')
    model = _replace_r50_with_r18(model)
    model.backbone.init_cfg = None
    from mmdet.models import build_detector
    detector = build_detector(model)
    input_shape = (1, 3, 100, 100)
    mm_inputs = _demo_mm_inputs(input_shape)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    losses = detector.forward(imgs, img_metas, gt_bboxes=gt_bboxes, return_loss=True)
    assert isinstance(losses, dict)
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]], return_loss=False)
            batch_results.append(result)

@pytest.mark.parametrize('cfg_file', ['reppoints/reppoints_moment_r50_fpn_1x_coco.py', 'retinanet/retinanet_r50_fpn_1x_coco.py', 'guided_anchoring/ga_retinanet_r50_fpn_1x_coco.py', 'ghm/retinanet_ghm_r50_fpn_1x_coco.py', 'fcos/fcos_center_r50_caffe_fpn_gn-head_1x_coco.py', 'foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco.py', 'yolo/yolov3_mobilenetv2_320_300e_coco.py', 'yolox/yolox_tiny_8x8_300e_coco.py'])
def test_single_stage_forward_gpu(cfg_file):
    if not torch.cuda.is_available():
        import pytest
        pytest.skip('test requires GPU and torch+cuda')
    model = _get_detector_cfg(cfg_file)
    model = _replace_r50_with_r18(model)
    model.backbone.init_cfg = None
    from mmdet.models import build_detector
    detector = build_detector(model)
    input_shape = (2, 3, 128, 128)
    mm_inputs = _demo_mm_inputs(input_shape)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    detector = detector.cuda()
    imgs = imgs.cuda()
    gt_bboxes = [b.cuda() for b in mm_inputs['gt_bboxes']]
    gt_labels = [g.cuda() for g in mm_inputs['gt_labels']]
    losses = detector.forward(imgs, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, return_loss=True)
    assert isinstance(losses, dict)
    detector.eval()
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]], return_loss=False)
            batch_results.append(result)

def test_faster_rcnn_ohem_forward():
    model = _get_detector_cfg('faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py')
    model = _replace_r50_with_r18(model)
    model.backbone.init_cfg = None
    from mmdet.models import build_detector
    detector = build_detector(model)
    input_shape = (1, 3, 100, 100)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    losses = detector.forward(imgs, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[0])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    losses = detector.forward(imgs, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0
    feature = detector.extract_feat(imgs[0][None, :])
    losses = detector.roi_head.forward_train(feature, img_metas, [torch.empty((0, 5))], gt_bboxes=gt_bboxes, gt_labels=gt_labels)
    assert isinstance(losses, dict)

@pytest.mark.parametrize('cfg_file', ['mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'])
def test_two_stage_forward(cfg_file):
    models_with_semantic = ['htc/htc_r50_fpn_1x_coco.py', 'panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco.py', 'scnet/scnet_r50_fpn_20e_coco.py']
    if cfg_file in models_with_semantic:
        with_semantic = True
    else:
        with_semantic = False
    model = _get_detector_cfg(cfg_file)
    model = _replace_r50_with_r18(model)
    model.backbone.init_cfg = None
    if cfg_file in ['seesaw_loss/mask_rcnn_r50_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py']:
        model.roi_head.bbox_head.num_classes = 80
        model.roi_head.bbox_head.loss_cls.num_classes = 80
        model.roi_head.mask_head.num_classes = 80
        model.test_cfg.rcnn.score_thr = 0.05
        model.test_cfg.rcnn.max_per_img = 100
    from mmdet.models import build_detector
    detector = build_detector(model)
    input_shape = (1, 3, 128, 128)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10], with_semantic=with_semantic)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    losses = detector.forward(imgs, img_metas, return_loss=True, **mm_inputs)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[0], with_semantic=with_semantic)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    losses = detector.forward(imgs, img_metas, return_loss=True, **mm_inputs)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()
    if cfg_file in ['panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco.py']:
        mm_inputs.pop('gt_semantic_seg')
    feature = detector.extract_feat(imgs[0][None, :])
    losses = detector.roi_head.forward_train(feature, img_metas, [torch.empty((0, 5))], **mm_inputs)
    assert isinstance(losses, dict)
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]], return_loss=False)
            batch_results.append(result)
    cascade_models = ['cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py', 'htc/htc_r50_fpn_1x_coco.py', 'scnet/scnet_r50_fpn_20e_coco.py']
    with torch.no_grad():
        detector.simple_test(imgs[0][None, :], [img_metas[0]], proposals=[torch.empty((0, 4))])
        features = detector.extract_feats([imgs[0][None, :]] * 2)
        detector.roi_head.aug_test(features, [torch.empty((0, 4))] * 2, [[img_metas[0]]] * 2)
        if cfg_file not in cascade_models:
            feature = detector.extract_feat(imgs[0][None, :])
            bboxes, scores = detector.roi_head.simple_test_bboxes(feature, [img_metas[0]], [torch.empty((0, 4))], None)
            assert all([bbox.shape == torch.Size((0, 4)) for bbox in bboxes])
            assert all([score.shape == torch.Size((0, detector.roi_head.bbox_head.fc_cls.out_features)) for score in scores])
        x1y1 = torch.randint(1, 100, (10, 2)).float()
        x2y2 = x1y1 + torch.randint(1, 100, (10, 2))
        detector.simple_test(imgs[0][None, :].repeat(2, 1, 1, 1), [img_metas[0]] * 2, proposals=[torch.empty((0, 4)), torch.cat([x1y1, x2y2], dim=-1)])
        detector.roi_head.aug_test(features, [torch.cat([x1y1, x2y2], dim=-1), torch.empty((0, 4))], [[img_metas[0]]] * 2)
        if cfg_file not in cascade_models:
            feature = detector.extract_feat(imgs[0][None, :].repeat(2, 1, 1, 1))
            bboxes, scores = detector.roi_head.simple_test_bboxes(feature, [img_metas[0]] * 2, [torch.empty((0, 4)), torch.cat([x1y1, x2y2], dim=-1)], None)
            assert bboxes[0].shape == torch.Size((0, 4))
            assert scores[0].shape == torch.Size((0, detector.roi_head.bbox_head.fc_cls.out_features))

@pytest.mark.parametrize('cfg_file', ['ghm/retinanet_ghm_r50_fpn_1x_coco.py', 'ssd/ssd300_coco.py'])
def test_single_stage_forward_cpu(cfg_file):
    model = _get_detector_cfg(cfg_file)
    model = _replace_r50_with_r18(model)
    model.backbone.init_cfg = None
    from mmdet.models import build_detector
    detector = build_detector(model)
    input_shape = (1, 3, 300, 300)
    mm_inputs = _demo_mm_inputs(input_shape)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    losses = detector.forward(imgs, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, return_loss=True)
    assert isinstance(losses, dict)
    detector.eval()
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]], return_loss=False)
            batch_results.append(result)

def test_yolact_forward():
    model = _get_detector_cfg('yolact/yolact_r50_1x8_coco.py')
    model = _replace_r50_with_r18(model)
    model.backbone.init_cfg = None
    from mmdet.models import build_detector
    detector = build_detector(model)
    input_shape = (1, 3, 100, 100)
    mm_inputs = _demo_mm_inputs(input_shape)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    detector.train()
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_masks = mm_inputs['gt_masks']
    losses = detector.forward(imgs, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, gt_masks=gt_masks, return_loss=True)
    assert isinstance(losses, dict)
    detector.forward_dummy(imgs)
    detector.eval()
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]], rescale=True, return_loss=False)
            batch_results.append(result)

def test_detr_forward():
    model = _get_detector_cfg('detr/detr_r50_8x2_150e_coco.py')
    model.backbone.depth = 18
    model.bbox_head.in_channels = 512
    model.backbone.init_cfg = None
    from mmdet.models import build_detector
    detector = build_detector(model)
    input_shape = (1, 3, 100, 100)
    mm_inputs = _demo_mm_inputs(input_shape)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    detector.train()
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    losses = detector.forward(imgs, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[0])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    losses = detector.forward(imgs, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0
    detector.eval()
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]], rescale=True, return_loss=False)
            batch_results.append(result)

def test_yolox_random_size():
    from mmdet.models import build_detector
    model = _get_detector_cfg('yolox/yolox_tiny_8x8_300e_coco.py')
    model.random_size_range = (2, 2)
    model.input_size = (64, 96)
    model.random_size_interval = 1
    detector = build_detector(model)
    input_shape = (1, 3, 64, 64)
    mm_inputs = _demo_mm_inputs(input_shape)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    detector.train()
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    detector.forward(imgs, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, return_loss=True)
    assert detector._input_size == (64, 96)

def test_maskformer_forward():
    model_cfg = _get_detector_cfg('maskformer/maskformer_r50_mstrain_16x1_75e_coco.py')
    base_channels = 32
    model_cfg.backbone.depth = 18
    model_cfg.backbone.init_cfg = None
    model_cfg.backbone.base_channels = base_channels
    model_cfg.panoptic_head.in_channels = [base_channels * 2 ** i for i in range(4)]
    model_cfg.panoptic_head.feat_channels = base_channels
    model_cfg.panoptic_head.out_channels = base_channels
    model_cfg.panoptic_head.pixel_decoder.encoder.transformerlayers.attn_cfgs.embed_dims = base_channels
    model_cfg.panoptic_head.pixel_decoder.encoder.transformerlayers.ffn_cfgs.embed_dims = base_channels
    model_cfg.panoptic_head.pixel_decoder.encoder.transformerlayers.ffn_cfgs.feedforward_channels = base_channels * 8
    model_cfg.panoptic_head.pixel_decoder.positional_encoding.num_feats = base_channels // 2
    model_cfg.panoptic_head.positional_encoding.num_feats = base_channels // 2
    model_cfg.panoptic_head.transformer_decoder.transformerlayers.attn_cfgs.embed_dims = base_channels
    model_cfg.panoptic_head.transformer_decoder.transformerlayers.ffn_cfgs.embed_dims = base_channels
    model_cfg.panoptic_head.transformer_decoder.transformerlayers.ffn_cfgs.feedforward_channels = base_channels * 8
    model_cfg.panoptic_head.transformer_decoder.transformerlayers.feedforward_channels = base_channels * 8
    from mmdet.core import BitmapMasks
    from mmdet.models import build_detector
    detector = build_detector(model_cfg)
    detector.train()
    img_metas = [{'batch_input_shape': (128, 160), 'img_shape': (126, 160, 3), 'ori_shape': (63, 80, 3), 'pad_shape': (128, 160, 3)}]
    img = torch.rand((1, 3, 128, 160))
    gt_bboxes = None
    gt_labels = [torch.tensor([10]).long()]
    thing_mask1 = np.zeros((1, 128, 160), dtype=np.int32)
    thing_mask1[0, :50] = 1
    gt_masks = [BitmapMasks(thing_mask1, 128, 160)]
    stuff_mask1 = torch.zeros((1, 128, 160)).long()
    stuff_mask1[0, :50] = 10
    stuff_mask1[0, 50:] = 100
    gt_semantic_seg = [stuff_mask1]
    losses = detector.forward(img=img, img_metas=img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, gt_masks=gt_masks, gt_semantic_seg=gt_semantic_seg, return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0
    gt_bboxes = [torch.empty((0, 4)).float()]
    gt_labels = [torch.empty((0,)).long()]
    mask = np.zeros((0, 128, 160), dtype=np.uint8)
    gt_masks = [BitmapMasks(mask, 128, 160)]
    gt_semantic_seg = [torch.randint(0, 133, (0, 128, 160))]
    losses = detector.forward(img, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, gt_masks=gt_masks, gt_semantic_seg=gt_semantic_seg, return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0
    detector.eval()
    with torch.no_grad():
        img_list = [g[None, :] for g in img]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]], rescale=True, return_loss=False)
        batch_results.append(result)

@pytest.mark.parametrize('cfg_file', ['mask2former/mask2former_r50_lsj_8x2_50e_coco.py', 'mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic.py'])
def test_mask2former_forward(cfg_file):
    model_cfg = _get_detector_cfg(cfg_file)
    base_channels = 32
    model_cfg.backbone.depth = 18
    model_cfg.backbone.init_cfg = None
    model_cfg.backbone.base_channels = base_channels
    model_cfg.panoptic_head.in_channels = [base_channels * 2 ** i for i in range(4)]
    model_cfg.panoptic_head.feat_channels = base_channels
    model_cfg.panoptic_head.out_channels = base_channels
    model_cfg.panoptic_head.pixel_decoder.encoder.transformerlayers.attn_cfgs.embed_dims = base_channels
    model_cfg.panoptic_head.pixel_decoder.encoder.transformerlayers.ffn_cfgs.embed_dims = base_channels
    model_cfg.panoptic_head.pixel_decoder.encoder.transformerlayers.ffn_cfgs.feedforward_channels = base_channels * 4
    model_cfg.panoptic_head.pixel_decoder.positional_encoding.num_feats = base_channels // 2
    model_cfg.panoptic_head.positional_encoding.num_feats = base_channels // 2
    model_cfg.panoptic_head.transformer_decoder.transformerlayers.attn_cfgs.embed_dims = base_channels
    model_cfg.panoptic_head.transformer_decoder.transformerlayers.ffn_cfgs.embed_dims = base_channels
    model_cfg.panoptic_head.transformer_decoder.transformerlayers.ffn_cfgs.feedforward_channels = base_channels * 8
    model_cfg.panoptic_head.transformer_decoder.transformerlayers.feedforward_channels = base_channels * 8
    num_stuff_classes = model_cfg.panoptic_head.num_stuff_classes
    from mmdet.core import BitmapMasks
    from mmdet.models import build_detector
    detector = build_detector(model_cfg)

    def _forward_train():
        losses = detector.forward(img, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, gt_masks=gt_masks, gt_semantic_seg=gt_semantic_seg, return_loss=True)
        assert isinstance(losses, dict)
        loss, _ = detector._parse_losses(losses)
        assert float(loss.item()) > 0
    detector.train()
    img_metas = [{'batch_input_shape': (128, 160), 'img_shape': (126, 160, 3), 'ori_shape': (63, 80, 3), 'pad_shape': (128, 160, 3)}]
    img = torch.rand((1, 3, 128, 160))
    gt_bboxes = None
    gt_labels = [torch.tensor([10]).long()]
    thing_mask1 = np.zeros((1, 128, 160), dtype=np.int32)
    thing_mask1[0, :50] = 1
    gt_masks = [BitmapMasks(thing_mask1, 128, 160)]
    stuff_mask1 = torch.zeros((1, 128, 160)).long()
    stuff_mask1[0, :50] = 10
    stuff_mask1[0, 50:] = 100
    gt_semantic_seg = [stuff_mask1]
    _forward_train()
    gt_semantic_seg = None
    _forward_train()
    gt_bboxes = [torch.empty((0, 4)).float()]
    gt_labels = [torch.empty((0,)).long()]
    mask = np.zeros((0, 128, 160), dtype=np.uint8)
    gt_masks = [BitmapMasks(mask, 128, 160)]
    gt_semantic_seg = [torch.randint(0, 133, (0, 128, 160))]
    _forward_train()
    gt_semantic_seg = None
    _forward_train()
    detector.eval()
    with torch.no_grad():
        img_list = [g[None, :] for g in img]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]], rescale=True, return_loss=False)
            if num_stuff_classes > 0:
                assert isinstance(result[0], dict)
            else:
                assert isinstance(result[0], tuple)
        batch_results.append(result)

def _forward_train():
    losses = detector.forward(img, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, gt_masks=gt_masks, gt_semantic_seg=gt_semantic_seg, return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0

@pytest.mark.parametrize('cfg_file', ['./tests/data/configs_mmtrack/selsa_faster_rcnn_r101_dc5_1x.py'])
def test_vid_fgfa_style_forward(cfg_file):
    config = Config.fromfile(cfg_file)
    model = copy.deepcopy(config.model)
    model.pretrains = None
    model.detector.pretrained = None
    from mmtrack.models import build_model
    detector = build_model(model)
    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    img_metas[0]['is_video_data'] = True
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_masks = mm_inputs['gt_masks']
    ref_input_shape = (2, 3, 256, 256)
    ref_mm_inputs = _demo_mm_inputs(ref_input_shape, num_items=[9, 11])
    ref_img = ref_mm_inputs.pop('imgs')[None]
    ref_img_metas = ref_mm_inputs.pop('img_metas')
    ref_img_metas[0]['is_video_data'] = True
    ref_img_metas[1]['is_video_data'] = True
    ref_gt_bboxes = ref_mm_inputs['gt_bboxes']
    ref_gt_labels = ref_mm_inputs['gt_labels']
    ref_gt_masks = ref_mm_inputs['gt_masks']
    losses = detector.forward(img=imgs, img_metas=img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, ref_img=ref_img, ref_img_metas=[ref_img_metas], ref_gt_bboxes=ref_gt_bboxes, ref_gt_labels=ref_gt_labels, gt_masks=gt_masks, ref_gt_masks=ref_gt_masks, return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[0])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    img_metas[0]['is_video_data'] = True
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_masks = mm_inputs['gt_masks']
    ref_mm_inputs = _demo_mm_inputs(ref_input_shape, num_items=[0, 0])
    ref_imgs = ref_mm_inputs.pop('imgs')[None]
    ref_img_metas = ref_mm_inputs.pop('img_metas')
    ref_img_metas[0]['is_video_data'] = True
    ref_img_metas[1]['is_video_data'] = True
    ref_gt_bboxes = ref_mm_inputs['gt_bboxes']
    ref_gt_labels = ref_mm_inputs['gt_labels']
    ref_gt_masks = ref_mm_inputs['gt_masks']
    losses = detector.forward(img=imgs, img_metas=img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, ref_img=ref_imgs, ref_img_metas=[ref_img_metas], ref_gt_bboxes=ref_gt_bboxes, ref_gt_labels=ref_gt_labels, gt_masks=gt_masks, ref_gt_masks=ref_gt_masks, return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()
    with torch.no_grad():
        imgs = torch.cat([imgs, imgs.clone()], dim=0)
        img_list = [g[None, :] for g in imgs]
        img_metas.extend(copy.deepcopy(img_metas))
        for i in range(len(img_metas)):
            img_metas[i]['frame_id'] = i
            img_metas[i]['num_left_ref_imgs'] = 1
            img_metas[i]['frame_stride'] = 1
        ref_imgs = [ref_imgs.clone(), imgs[[0]][None].clone()]
        ref_img_metas = [copy.deepcopy(ref_img_metas), copy.deepcopy([img_metas[0]])]
        results = defaultdict(list)
        for one_img, one_meta, ref_img, ref_img_meta in zip(img_list, img_metas, ref_imgs, ref_img_metas):
            result = detector.forward([one_img], [[one_meta]], ref_img=[ref_img], ref_img_metas=[[ref_img_meta]], return_loss=False)
            for k, v in result.items():
                results[k].append(v)

@pytest.mark.parametrize('cfg_file', ['./tests/data/configs_mmtrack/tracktor_faster-rcnn_r50_fpn_4e.py'])
def test_tracktor_forward(cfg_file):
    config = Config.fromfile(cfg_file)
    model = copy.deepcopy(config.model)
    model.pretrains = None
    model.detector.pretrained = None
    from mmtrack.models import build_model
    mot = build_model(model)
    mot.eval()
    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10], with_track=True)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    with torch.no_grad():
        imgs = torch.cat([imgs, imgs.clone()], dim=0)
        img_list = [g[None, :] for g in imgs]
        img2_metas = copy.deepcopy(img_metas)
        img2_metas[0]['frame_id'] = 1
        img_metas.extend(img2_metas)
        results = defaultdict(list)
        for one_img, one_meta in zip(img_list, img_metas):
            result = mot.forward([one_img], [[one_meta]], return_loss=False)
            for k, v in result.items():
                results[k].append(v)

def test_cascade_onnx_export():
    config_path = './configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
    cfg = mmcv.Config.fromfile(config_path)
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    with torch.no_grad():
        model.forward = partial(model.forward, img_metas=[[dict()]])
        dynamic_axes = {'input_img': {0: 'batch', 2: 'width', 3: 'height'}, 'dets': {0: 'batch', 1: 'num_dets'}, 'labels': {0: 'batch', 1: 'num_dets'}}
        torch.onnx.export(model, [torch.rand(1, 3, 400, 500)], 'tmp.onnx', output_names=['dets', 'labels'], input_names=['input_img'], keep_initializers_as_inputs=True, do_constant_folding=True, verbose=False, opset_version=11, dynamic_axes=dynamic_axes)

def test_faster_onnx_export():
    config_path = './configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    cfg = mmcv.Config.fromfile(config_path)
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    with torch.no_grad():
        model.forward = partial(model.forward, img_metas=[[dict()]])
        dynamic_axes = {'input_img': {0: 'batch', 2: 'width', 3: 'height'}, 'dets': {0: 'batch', 1: 'num_dets'}, 'labels': {0: 'batch', 1: 'num_dets'}}
        torch.onnx.export(model, [torch.rand(1, 3, 400, 500)], 'tmp.onnx', output_names=['dets', 'labels'], input_names=['input_img'], keep_initializers_as_inputs=True, do_constant_folding=True, verbose=False, opset_version=11, dynamic_axes=dynamic_axes)

def _context_for_ohem():
    import sys
    from os.path import dirname
    sys.path.insert(0, dirname(dirname(dirname(__file__))))
    from test_models.test_forward import _get_detector_cfg
    model = _get_detector_cfg('faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py')
    model['pretrained'] = None
    from mmdet.models import build_detector
    context = build_detector(model).roi_head
    return context

def test_ohem_sampler():
    assigner = MaxIoUAssigner(pos_iou_thr=0.5, neg_iou_thr=0.5, ignore_iof_thr=0.5, ignore_wrt_candidates=False)
    bboxes = torch.FloatTensor([[0, 0, 10, 10], [10, 10, 20, 20], [5, 5, 15, 15], [32, 32, 38, 42]])
    gt_bboxes = torch.FloatTensor([[0, 0, 10, 9], [0, 10, 10, 19]])
    gt_labels = torch.LongTensor([1, 2])
    gt_bboxes_ignore = torch.Tensor([[30, 30, 40, 40]])
    assign_result = assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore=gt_bboxes_ignore, gt_labels=gt_labels)
    context = _context_for_ohem()
    sampler = OHEMSampler(num=10, pos_fraction=0.5, context=context, neg_pos_ub=-1, add_gt_as_proposals=True)
    feats = [torch.rand(1, 256, int(2 ** i), int(2 ** i)) for i in [6, 5, 4, 3, 2]]
    sample_result = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels, feats=feats)
    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)

def test_ohem_sampler_empty_gt():
    assigner = MaxIoUAssigner(pos_iou_thr=0.5, neg_iou_thr=0.5, ignore_iof_thr=0.5, ignore_wrt_candidates=False)
    bboxes = torch.FloatTensor([[0, 0, 10, 10], [10, 10, 20, 20], [5, 5, 15, 15], [32, 32, 38, 42]])
    gt_bboxes = torch.empty(0, 4)
    gt_labels = torch.LongTensor([])
    gt_bboxes_ignore = torch.Tensor([])
    assign_result = assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore=gt_bboxes_ignore, gt_labels=gt_labels)
    context = _context_for_ohem()
    sampler = OHEMSampler(num=10, pos_fraction=0.5, context=context, neg_pos_ub=-1, add_gt_as_proposals=True)
    feats = [torch.rand(1, 256, int(2 ** i), int(2 ** i)) for i in [6, 5, 4, 3, 2]]
    sample_result = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels, feats=feats)
    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)

def test_ohem_sampler_empty_pred():
    assigner = MaxIoUAssigner(pos_iou_thr=0.5, neg_iou_thr=0.5, ignore_iof_thr=0.5, ignore_wrt_candidates=False)
    bboxes = torch.empty(0, 4)
    gt_bboxes = torch.FloatTensor([[0, 0, 10, 10], [10, 10, 20, 20], [5, 5, 15, 15], [32, 32, 38, 42]])
    gt_labels = torch.LongTensor([1, 2, 2, 3])
    gt_bboxes_ignore = torch.Tensor([])
    assign_result = assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore=gt_bboxes_ignore, gt_labels=gt_labels)
    context = _context_for_ohem()
    sampler = OHEMSampler(num=10, pos_fraction=0.5, context=context, neg_pos_ub=-1, add_gt_as_proposals=True)
    feats = [torch.rand(1, 256, int(2 ** i), int(2 ** i)) for i in [6, 5, 4, 3, 2]]
    sample_result = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels, feats=feats)
    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)

def test_score_hlr_sampler_empty_pred():
    assigner = MaxIoUAssigner(pos_iou_thr=0.5, neg_iou_thr=0.5, ignore_iof_thr=0.5, ignore_wrt_candidates=False)
    context = _context_for_ohem()
    sampler = ScoreHLRSampler(num=10, pos_fraction=0.5, context=context, neg_pos_ub=-1, add_gt_as_proposals=True)
    gt_bboxes_ignore = torch.Tensor([])
    feats = [torch.rand(1, 256, int(2 ** i), int(2 ** i)) for i in [6, 5, 4, 3, 2]]
    bboxes = torch.empty(0, 4)
    gt_bboxes = torch.FloatTensor([[0, 0, 10, 10], [10, 10, 20, 20], [5, 5, 15, 15], [32, 32, 38, 42]])
    gt_labels = torch.LongTensor([1, 2, 2, 3])
    assign_result = assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore=gt_bboxes_ignore, gt_labels=gt_labels)
    sample_result, _ = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels, feats=feats)
    assert len(sample_result.neg_inds) == 0
    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)
    bboxes = torch.FloatTensor([[0, 0, 10, 10], [10, 10, 20, 20], [5, 5, 15, 15], [32, 32, 38, 42]])
    gt_bboxes = torch.empty(0, 4)
    gt_labels = torch.LongTensor([])
    assign_result = assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore=gt_bboxes_ignore, gt_labels=gt_labels)
    sample_result, _ = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels, feats=feats)
    assert len(sample_result.pos_inds) == 0
    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)
    bboxes = torch.FloatTensor([[0, 0, 10, 10], [10, 10, 20, 20], [5, 5, 15, 15], [32, 32, 38, 42]])
    gt_bboxes = torch.FloatTensor([[0, 0, 10, 10], [10, 10, 20, 20], [5, 5, 15, 15], [32, 32, 38, 42]])
    gt_labels = torch.LongTensor([1, 2, 2, 3])
    assign_result = assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore=gt_bboxes_ignore, gt_labels=gt_labels)
    sample_result, _ = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels, feats=feats)
    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)

def model_aug_test_template(cfg_file):
    cfg = mmcv.Config.fromfile(cfg_file)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_detector(cfg.model)
    load_cfg, multi_scale_cfg = cfg.test_pipeline
    multi_scale_cfg['flip'] = True
    multi_scale_cfg['flip_direction'] = ['horizontal', 'vertical', 'diagonal']
    multi_scale_cfg['img_scale'] = [(1333, 800), (800, 600), (640, 480)]
    load = build_from_cfg(load_cfg, PIPELINES)
    transform = build_from_cfg(multi_scale_cfg, PIPELINES)
    results = dict(img_prefix=osp.join(osp.dirname(__file__), '../../../data'), img_info=dict(filename='color.jpg'))
    results = transform(load(results))
    assert len(results['img']) == 12
    assert len(results['img_metas']) == 12
    results['img'] = [collate([x]) for x in results['img']]
    results['img_metas'] = [collate([x]).data[0] for x in results['img_metas']]
    model.eval()
    with torch.no_grad():
        aug_result = model(return_loss=False, rescale=True, **results)
    return aug_result

def test_cascade_rcnn_aug_test():
    aug_result = model_aug_test_template('configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py')
    assert len(aug_result[0]) == 80

def test_mask_rcnn_aug_test():
    aug_result = model_aug_test_template('configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py')
    assert len(aug_result[0]) == 2
    assert len(aug_result[0][0]) == 80
    assert len(aug_result[0][1]) == 80

def test_htc_aug_test():
    aug_result = model_aug_test_template('configs/htc/htc_r50_fpn_1x_coco.py')
    assert len(aug_result[0]) == 2
    assert len(aug_result[0][0]) == 80
    assert len(aug_result[0][1]) == 80

def test_scnet_aug_test():
    aug_result = model_aug_test_template('configs/scnet/scnet_r50_fpn_1x_coco.py')
    assert len(aug_result[0]) == 2
    assert len(aug_result[0][0]) == 80
    assert len(aug_result[0][1]) == 80

def test_cornernet_aug_test():
    cfg = mmcv.Config.fromfile('configs/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco.py')
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_detector(cfg.model)
    load_cfg, multi_scale_cfg = cfg.test_pipeline
    multi_scale_cfg['flip'] = True
    multi_scale_cfg['flip_direction'] = ['horizontal', 'vertical', 'diagonal']
    multi_scale_cfg['scale_factor'] = [0.5, 1.0, 2.0]
    load = build_from_cfg(load_cfg, PIPELINES)
    transform = build_from_cfg(multi_scale_cfg, PIPELINES)
    results = dict(img_prefix=osp.join(osp.dirname(__file__), '../../../data'), img_info=dict(filename='color.jpg'))
    results = transform(load(results))
    assert len(results['img']) == 12
    assert len(results['img_metas']) == 12
    results['img'] = [collate([x]) for x in results['img']]
    results['img_metas'] = [collate([x]).data[0] for x in results['img_metas']]
    model.eval()
    with torch.no_grad():
        aug_result = model(return_loss=False, rescale=True, **results)
    assert len(aug_result[0]) == 80

