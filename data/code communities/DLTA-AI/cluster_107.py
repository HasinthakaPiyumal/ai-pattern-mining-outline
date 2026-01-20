# Cluster 107

def test_random_crop():
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCrop', crop_size=(-1, 0))
        build_from_cfg(transform, PIPELINES)
    results = dict()
    img = mmcv.imread(osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='RandomCrop', crop_size=(h - 20, w - 20))
    crop_module = build_from_cfg(transform, PIPELINES)
    results = crop_module(results)
    assert results['img'].shape[:2] == (h - 20, w - 20)
    assert results['img_shape'][:2] == (h - 20, w - 20)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes'].shape[0] == 8
    assert results['gt_bboxes_ignore'].shape[0] == 2

    def area(bboxes):
        return np.prod(bboxes[:, 2:4] - bboxes[:, 0:2], axis=1)
    assert (area(results['gt_bboxes']) <= area(gt_bboxes)).all()
    assert (area(results['gt_bboxes_ignore']) <= area(gt_bboxes_ignore)).all()
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32
    with pytest.raises(ValueError):
        transform = dict(type='RandomCrop', crop_size=(1, 1), crop_type='unknown')
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCrop', crop_type='relative', crop_size=(0, 0))
        build_from_cfg(transform, PIPELINES)

    def _construct_toy_data():
        img = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8)
        img = np.stack([img, img, img], axis=-1)
        results = dict()
        results['img'] = img
        results['img_shape'] = img.shape
        results['img_fields'] = ['img']
        results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
        results['gt_bboxes'] = np.array([[0.0, 0.0, 2.0, 1.0]], dtype=np.float32)
        results['gt_bboxes_ignore'] = np.array([[2.0, 0.0, 3.0, 1.0]], dtype=np.float32)
        results['gt_labels'] = np.array([1], dtype=np.int64)
        return results
    results = _construct_toy_data()
    transform = dict(type='RandomCrop', crop_type='relative_range', crop_size=(0.3, 0.7), allow_negative_crop=True)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    h, w = results_transformed['img_shape'][:2]
    assert int(2 * 0.3 + 0.5) <= h <= int(2 * 1 + 0.5)
    assert int(4 * 0.7 + 0.5) <= w <= int(4 * 1 + 0.5)
    assert results_transformed['gt_bboxes'].dtype == np.float32
    assert results_transformed['gt_bboxes_ignore'].dtype == np.float32
    transform = dict(type='RandomCrop', crop_type='relative', crop_size=(0.3, 0.7), allow_negative_crop=True)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    h, w = results_transformed['img_shape'][:2]
    assert h == int(2 * 0.3 + 0.5) and w == int(4 * 0.7 + 0.5)
    assert results_transformed['gt_bboxes'].dtype == np.float32
    assert results_transformed['gt_bboxes_ignore'].dtype == np.float32
    transform = dict(type='RandomCrop', crop_type='absolute', crop_size=(1, 2), allow_negative_crop=True)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    h, w = results_transformed['img_shape'][:2]
    assert h == 1 and w == 2
    assert results_transformed['gt_bboxes'].dtype == np.float32
    assert results_transformed['gt_bboxes_ignore'].dtype == np.float32
    transform = dict(type='RandomCrop', crop_type='absolute_range', crop_size=(1, 20), allow_negative_crop=True)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    h, w = results_transformed['img_shape'][:2]
    assert 1 <= h <= 2 and 1 <= w <= 4
    assert results_transformed['gt_bboxes'].dtype == np.float32
    assert results_transformed['gt_bboxes_ignore'].dtype == np.float32

def create_random_bboxes(num_bboxes, img_w, img_h):
    bboxes_left_top = np.random.uniform(0, 0.5, size=(num_bboxes, 2))
    bboxes_right_bottom = np.random.uniform(0.5, 1, size=(num_bboxes, 2))
    bboxes = np.concatenate((bboxes_left_top, bboxes_right_bottom), 1)
    bboxes = (bboxes * np.array([img_w, img_h, img_w, img_h])).astype(np.float32)
    return bboxes

def area(bboxes):
    return np.prod(bboxes[:, 2:4] - bboxes[:, 0:2], axis=1)

def _construct_toy_data():
    img = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8)
    img = np.stack([img, img, img], axis=-1)
    results = dict()
    results['img'] = img
    results['img_shape'] = img.shape
    results['img_fields'] = ['img']
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
    results['gt_bboxes'] = np.array([[0.0, 0.0, 2.0, 1.0]], dtype=np.float32)
    results['gt_bboxes_ignore'] = np.array([[2.0, 0.0, 3.0, 1.0]], dtype=np.float32)
    results['gt_labels'] = np.array([1], dtype=np.int64)
    return results

def test_min_iou_random_crop():
    results = dict()
    img = mmcv.imread(osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(1, w, h)
    gt_bboxes_ignore = create_random_bboxes(1, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='MinIoURandomCrop')
    crop_module = build_from_cfg(transform, PIPELINES)
    results_test = copy.deepcopy(results)
    results_test['img1'] = results_test['img']
    results_test['img_fields'] = ['img', 'img1']
    with pytest.raises(AssertionError):
        crop_module(results_test)
    results = crop_module(results)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32
    patch = np.array([0, 0, results['img_shape'][1], results['img_shape'][0]])
    ious = bbox_overlaps(patch.reshape(-1, 4), results['gt_bboxes']).reshape(-1)
    ious_ignore = bbox_overlaps(patch.reshape(-1, 4), results['gt_bboxes_ignore']).reshape(-1)
    mode = crop_module.mode
    if mode == 1:
        assert np.equal(results['gt_bboxes'], gt_bboxes).all()
        assert np.equal(results['gt_bboxes_ignore'], gt_bboxes_ignore).all()
    else:
        assert (ious >= mode).all()
        assert (ious_ignore >= mode).all()

def test_random_center_crop_pad():
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCenterCropPad', crop_size=(-1, 0), test_mode=False, test_pad_mode=None)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCenterCropPad', crop_size=(511, 511), ratios=1.0, test_mode=False, test_pad_mode=None)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCenterCropPad', crop_size=(511, 511), mean=None, std=None, to_rgb=None, test_mode=False, test_pad_mode=None)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCenterCropPad', crop_size=(511, 511), ratios=None, border=None, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True, test_mode=True, test_pad_mode=('logical_or', 127))
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCenterCropPad', crop_size=None, ratios=(0.9, 1.0, 1.1), border=None, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True, test_mode=True, test_pad_mode=('logical_or', 127))
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCenterCropPad', crop_size=None, ratios=None, border=128, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True, test_mode=True, test_pad_mode=('logical_or', 127))
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCenterCropPad', crop_size=None, ratios=None, border=None, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True, test_mode=True, test_pad_mode=('do_nothing', 100))
        build_from_cfg(transform, PIPELINES)
    results = dict(img_prefix=osp.join(osp.dirname(__file__), '../../../data'), img_info=dict(filename='color.jpg'))
    load = dict(type='LoadImageFromFile', to_float32=True)
    load = build_from_cfg(load, PIPELINES)
    results = load(results)
    test_results = copy.deepcopy(results)
    h, w, _ = results['img_shape']
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    train_transform = dict(type='RandomCenterCropPad', crop_size=(h - 20, w - 20), ratios=(1.0,), border=128, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True, test_mode=False, test_pad_mode=None)
    crop_module = build_from_cfg(train_transform, PIPELINES)
    train_results = crop_module(results)
    assert train_results['img'].shape[:2] == (h - 20, w - 20)
    assert train_results['pad_shape'][:2] == (h - 20, w - 20)
    assert train_results['gt_bboxes'].shape[0] == 8
    assert train_results['gt_bboxes_ignore'].shape[0] == 2
    assert train_results['gt_bboxes'].dtype == np.float32
    assert train_results['gt_bboxes_ignore'].dtype == np.float32
    test_transform = dict(type='RandomCenterCropPad', crop_size=None, ratios=None, border=None, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True, test_mode=True, test_pad_mode=('logical_or', 127))
    crop_module = build_from_cfg(test_transform, PIPELINES)
    test_results = crop_module(test_results)
    assert test_results['img'].shape[:2] == (h | 127, w | 127)
    assert test_results['pad_shape'][:2] == (h | 127, w | 127)
    assert 'border' in test_results

def test_random_shift():
    with pytest.raises(AssertionError):
        transform = dict(type='RandomShift', shift_ratio=1.5)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='RandomShift', max_shift_px=-1)
        build_from_cfg(transform, PIPELINES)
    results = dict()
    img = mmcv.imread(osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='RandomShift', shift_ratio=1.0)
    random_shift_module = build_from_cfg(transform, PIPELINES)
    results = random_shift_module(results)
    assert results['img'].shape[:2] == (h, w)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32

def test_random_affine():
    with pytest.raises(AssertionError):
        transform = dict(type='RandomAffine', max_translate_ratio=1.5)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='RandomAffine', scaling_ratio_range=(1.5, 0.5))
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='RandomAffine', scaling_ratio_range=(0, 0.5))
        build_from_cfg(transform, PIPELINES)
    results = dict()
    img = mmcv.imread(osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='RandomAffine')
    random_affine_module = build_from_cfg(transform, PIPELINES)
    results = random_affine_module(results)
    assert results['img'].shape[:2] == (h, w)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32
    gt_bboxes = np.array([[0, 0, 1, 1], [0, 0, 3, 100]], dtype=np.float32)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    transform = dict(type='RandomAffine', max_rotate_degree=0.0, max_translate_ratio=0.0, scaling_ratio_range=(1.0, 1.0), max_shear_degree=0.0, border=(0, 0), min_bbox_size=2, max_aspect_ratio=20, skip_filter=False)
    random_affine_module = build_from_cfg(transform, PIPELINES)
    results = random_affine_module(results)
    assert results['gt_bboxes'].shape[0] == 0
    assert results['gt_labels'].shape[0] == 0
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32

def test_mosaic():
    with pytest.raises(AssertionError):
        transform = dict(type='Mosaic', img_scale=640)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='Mosaic', prob=1.5)
        build_from_cfg(transform, PIPELINES)
    results = dict()
    img = mmcv.imread(osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='Mosaic', img_scale=(10, 12))
    mosaic_module = build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        mosaic_module(results)
    results['mix_results'] = [copy.deepcopy(results)] * 3
    results = mosaic_module(results)
    assert results['img'].shape[:2] == (20, 24)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32

def test_mixup():
    with pytest.raises(AssertionError):
        transform = dict(type='MixUp', img_scale=640)
        build_from_cfg(transform, PIPELINES)
    results = dict()
    img = mmcv.imread(osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='MixUp', img_scale=(10, 12))
    mixup_module = build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        mixup_module(results)
    with pytest.raises(AssertionError):
        results['mix_results'] = [copy.deepcopy(results)] * 2
        mixup_module(results)
    results['mix_results'] = [copy.deepcopy(results)]
    results = mixup_module(results)
    assert results['img'].shape[:2] == (288, 512)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32
    gt_bboxes = np.array([[0, 0, 1, 1], [0, 0, 3, 3]], dtype=np.float32)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = np.array([], dtype=np.float32)
    mixresults = results['mix_results'][0]
    mixresults['gt_labels'] = copy.deepcopy(results['gt_labels'])
    mixresults['gt_bboxes'] = copy.deepcopy(results['gt_bboxes'])
    mixresults['gt_bboxes_ignore'] = copy.deepcopy(results['gt_bboxes_ignore'])
    transform = dict(type='MixUp', img_scale=(10, 12), ratio_range=(1.5, 1.5), min_bbox_size=5, skip_filter=False)
    mixup_module = build_from_cfg(transform, PIPELINES)
    results = mixup_module(results)
    assert results['gt_bboxes'].shape[0] == 2
    assert results['gt_labels'].shape[0] == 2
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32

