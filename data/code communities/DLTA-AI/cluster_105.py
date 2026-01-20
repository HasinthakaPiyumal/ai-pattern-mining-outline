# Cluster 105

def check_result_same(results, pipeline_results):
    """Check whether the `pipeline_results` is the same with the predefined
    `results`.

    Args:
        results (dict): Predefined results which should be the standard output
            of the transform pipeline.
        pipeline_results (dict): Results processed by the transform pipeline.
    """
    _check_fields(results, pipeline_results, results.get('img_fields', ['img']))
    _check_fields(results, pipeline_results, results.get('bbox_fields', []))
    _check_fields(results, pipeline_results, results.get('mask_fields', []))
    _check_fields(results, pipeline_results, results.get('seg_fields', []))
    if 'gt_labels' in results:
        assert np.equal(results['gt_labels'], pipeline_results['gt_labels']).all()

def _check_fields(results, pipeline_results, keys):
    """Check data in fields from two results are same."""
    for key in keys:
        if isinstance(results[key], (BitmapMasks, PolygonMasks)):
            assert np.equal(results[key].to_ndarray(), pipeline_results[key].to_ndarray()).all()
        else:
            assert np.equal(results[key], pipeline_results[key]).all()
            assert results[key].dtype == pipeline_results[key].dtype

def test_translate():
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', level=-1)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', level=[1])
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', level=1, prob=-0.5)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', level=1, img_fill_val=(128, 128, 128, 128))
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(ValueError):
        transform = dict(type='Translate', level=1, img_fill_val=[128, 128, 128])
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', level=1, img_fill_val=(128, -1, 256))
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', level=1, img_fill_val=128, direction='diagonal')
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', level=1, img_fill_val=128, max_translate_offset=(250.0,))
        build_from_cfg(transform, PIPELINES)
    results = construct_toy_data()

    def _check_bbox_mask(results, results_translated, offset, direction, min_size=0.0):
        bbox2label = {'gt_bboxes': 'gt_labels', 'gt_bboxes_ignore': 'gt_labels_ignore'}
        bbox2mask = {'gt_bboxes': 'gt_masks', 'gt_bboxes_ignore': 'gt_masks_ignore'}

        def _translate_bbox(bboxes, offset, direction, max_h, max_w):
            if direction == 'horizontal':
                bboxes[:, 0::2] = bboxes[:, 0::2] + offset
            elif direction == 'vertical':
                bboxes[:, 1::2] = bboxes[:, 1::2] + offset
            else:
                raise ValueError
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, max_w)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, max_h)
            return bboxes
        h, w, c = results_translated['img'].shape
        for key in results_translated.get('bbox_fields', []):
            label_key, mask_key = (bbox2label[key], bbox2mask[key])
            if label_key in results:
                assert len(results_translated[key]) == len(results_translated[label_key])
            if mask_key in results:
                assert len(results_translated[key]) == len(results_translated[mask_key])
            gt_bboxes = _translate_bbox(copy.deepcopy(results[key]), offset, direction, h, w)
            valid_inds = (gt_bboxes[:, 2] - gt_bboxes[:, 0] > min_size) & (gt_bboxes[:, 3] - gt_bboxes[:, 1] > min_size)
            gt_bboxes = gt_bboxes[valid_inds]
            assert np.equal(gt_bboxes, results_translated[key]).all()
            if mask_key not in results:
                continue
            masks, masks_translated = (results[mask_key].to_ndarray(), results_translated[mask_key].to_ndarray())
            assert masks.dtype == masks_translated.dtype
            if direction == 'horizontal':
                masks_pad = _pad(h, abs(offset), masks.shape[0], 0, axis=0, dtype=masks.dtype)
                if offset <= 0:
                    gt_masks = np.concatenate((masks[:, :, -offset:], masks_pad), axis=-1)
                else:
                    gt_masks = np.concatenate((masks_pad, masks[:, :, :-offset]), axis=-1)
            else:
                masks_pad = _pad(abs(offset), w, masks.shape[0], 0, axis=0, dtype=masks.dtype)
                if offset <= 0:
                    gt_masks = np.concatenate((masks[:, -offset:, :], masks_pad), axis=1)
                else:
                    gt_masks = np.concatenate((masks_pad, masks[:, :-offset, :]), axis=1)
            gt_masks = gt_masks[valid_inds]
            assert np.equal(gt_masks, masks_translated).all()

    def _check_img_seg(results, results_translated, keys, offset, fill_val, direction):
        for key in keys:
            assert isinstance(results_translated[key], type(results[key]))
            data, data_translated = (results[key], results_translated[key])
            if 'mask' in key:
                data, data_translated = (data.to_ndarray(), data_translated.to_ndarray())
            assert data.dtype == data_translated.dtype
            if 'img' in key:
                data, data_translated = (data.transpose((2, 0, 1)), data_translated.transpose((2, 0, 1)))
            elif 'seg' in key:
                data, data_translated = (data[None, :, :], data_translated[None, :, :])
            c, h, w = data.shape
            if direction == 'horizontal':
                data_pad = _pad(h, abs(offset), c, fill_val, axis=0, dtype=data.dtype)
                if offset <= 0:
                    data_gt = np.concatenate((data[:, :, -offset:], data_pad), axis=-1)
                else:
                    data_gt = np.concatenate((data_pad, data[:, :, :-offset]), axis=-1)
            else:
                data_pad = _pad(abs(offset), w, c, fill_val, axis=0, dtype=data.dtype)
                if offset <= 0:
                    data_gt = np.concatenate((data[:, -offset:, :], data_pad), axis=1)
                else:
                    data_gt = np.concatenate((data_pad, data[:, :-offset, :]), axis=1)
            if 'mask' in key:
                pass
            else:
                assert np.equal(data_gt, data_translated).all()

    def check_translate(results, results_translated, offset, img_fill_val, seg_ignore_label, direction, min_size=0):
        _check_keys(results, results_translated)
        _check_img_seg(results, results_translated, results.get('img_fields', ['img']), offset, img_fill_val, direction)
        _check_img_seg(results, results_translated, results.get('seg_fields', []), offset, seg_ignore_label, direction)
        _check_bbox_mask(results, results_translated, offset, direction, min_size)
    img_fill_val = (104, 116, 124)
    seg_ignore_label = 255
    transform = dict(type='Translate', level=0, prob=1.0, img_fill_val=img_fill_val, seg_ignore_label=seg_ignore_label)
    translate_module = build_from_cfg(transform, PIPELINES)
    results_wo_translate = translate_module(copy.deepcopy(results))
    check_translate(copy.deepcopy(results), results_wo_translate, 0, img_fill_val, seg_ignore_label, 'horizontal')
    transform = dict(type='Translate', level=8, prob=1.0, img_fill_val=img_fill_val, random_negative_prob=1.0, seg_ignore_label=seg_ignore_label)
    translate_module = build_from_cfg(transform, PIPELINES)
    offset = translate_module.offset
    results_translated = translate_module(copy.deepcopy(results))
    check_translate(copy.deepcopy(results), results_translated, -offset, img_fill_val, seg_ignore_label, 'horizontal')
    translate_module.random_negative_prob = 0.0
    results_translated = translate_module(copy.deepcopy(results))
    check_translate(copy.deepcopy(results), results_translated, offset, img_fill_val, seg_ignore_label, 'horizontal')
    transform = dict(type='Translate', level=10, prob=1.0, img_fill_val=img_fill_val, seg_ignore_label=seg_ignore_label, random_negative_prob=1.0, direction='vertical')
    translate_module = build_from_cfg(transform, PIPELINES)
    offset = translate_module.offset
    results_translated = translate_module(copy.deepcopy(results))
    check_translate(copy.deepcopy(results), results_translated, -offset, img_fill_val, seg_ignore_label, 'vertical')
    translate_module.random_negative_prob = 0.0
    results_translated = translate_module(copy.deepcopy(results))
    check_translate(copy.deepcopy(results), results_translated, offset, img_fill_val, seg_ignore_label, 'vertical')
    transform = dict(type='Translate', level=8, prob=0.0, img_fill_val=img_fill_val, random_negative_prob=0.0, seg_ignore_label=seg_ignore_label)
    translate_module = build_from_cfg(transform, PIPELINES)
    results_translated = translate_module(copy.deepcopy(results))
    results = construct_toy_data(False)
    transform = dict(type='Translate', level=10, prob=1.0, img_fill_val=img_fill_val, seg_ignore_label=seg_ignore_label, direction='vertical')
    translate_module = build_from_cfg(transform, PIPELINES)
    offset = translate_module.offset
    translate_module.random_negative_prob = 1.0
    results_translated = translate_module(copy.deepcopy(results))

    def _translated_gt(masks, direction, offset, out_shape):
        translated_masks = []
        for poly_per_obj in masks:
            translated_poly_per_obj = []
            for p in poly_per_obj:
                p = p.copy()
                if direction == 'horizontal':
                    p[0::2] = np.clip(p[0::2] + offset, 0, out_shape[1])
                elif direction == 'vertical':
                    p[1::2] = np.clip(p[1::2] + offset, 0, out_shape[0])
                if PolygonMasks([[p]], *out_shape).areas[0] > 0:
                    translated_poly_per_obj.append(p)
            if len(translated_poly_per_obj):
                translated_masks.append(translated_poly_per_obj)
        translated_masks = PolygonMasks(translated_masks, *out_shape)
        return translated_masks
    h, w = results['img_shape'][:2]
    for key in results.get('mask_fields', []):
        masks = results[key]
        translated_gt = _translated_gt(masks, 'vertical', -offset, (h, w))
        assert np.equal(results_translated[key].to_ndarray(), translated_gt.to_ndarray()).all()
    results = construct_toy_data(False)
    transform = dict(type='Translate', level=8, prob=1.0, img_fill_val=img_fill_val, random_negative_prob=0.0, seg_ignore_label=seg_ignore_label)
    translate_module = build_from_cfg(transform, PIPELINES)
    offset = translate_module.offset
    results_translated = translate_module(copy.deepcopy(results))
    h, w = results['img_shape'][:2]
    for key in results.get('mask_fields', []):
        masks = results[key]
        translated_gt = _translated_gt(masks, 'horizontal', offset, (h, w))
        assert np.equal(results_translated[key].to_ndarray(), translated_gt.to_ndarray()).all()
    policies = [[dict(type='Translate', level=10, prob=1.0)]]
    autoaug = dict(type='AutoAugment', policies=policies)
    autoaug_module = build_from_cfg(autoaug, PIPELINES)
    autoaug_module(copy.deepcopy(results))
    policies = [[dict(type='Translate', level=10, prob=1.0), dict(type='Translate', level=8, img_fill_val=img_fill_val, direction='vertical')]]
    autoaug = dict(type='AutoAugment', policies=policies)
    autoaug_module = build_from_cfg(autoaug, PIPELINES)
    autoaug_module(copy.deepcopy(results))

def construct_toy_data(poly2mask=True):
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
    results['mask_fields'] = ['gt_masks']
    if poly2mask:
        gt_masks = np.array([[0, 1, 1, 0], [0, 1, 0, 0]], dtype=np.uint8)[None, :, :]
        results['gt_masks'] = BitmapMasks(gt_masks, 2, 4)
    else:
        raw_masks = [[np.array([0, 0, 2, 0, 2, 1, 0, 1], dtype=np.float)]]
        results['gt_masks'] = PolygonMasks(raw_masks, 2, 4)
    results['seg_fields'] = ['gt_semantic_seg']
    results['gt_semantic_seg'] = img[..., 0]
    return results

def _check_bbox_mask(results, results_translated, offset, direction, min_size=0.0):
    bbox2label = {'gt_bboxes': 'gt_labels', 'gt_bboxes_ignore': 'gt_labels_ignore'}
    bbox2mask = {'gt_bboxes': 'gt_masks', 'gt_bboxes_ignore': 'gt_masks_ignore'}

    def _translate_bbox(bboxes, offset, direction, max_h, max_w):
        if direction == 'horizontal':
            bboxes[:, 0::2] = bboxes[:, 0::2] + offset
        elif direction == 'vertical':
            bboxes[:, 1::2] = bboxes[:, 1::2] + offset
        else:
            raise ValueError
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, max_w)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, max_h)
        return bboxes
    h, w, c = results_translated['img'].shape
    for key in results_translated.get('bbox_fields', []):
        label_key, mask_key = (bbox2label[key], bbox2mask[key])
        if label_key in results:
            assert len(results_translated[key]) == len(results_translated[label_key])
        if mask_key in results:
            assert len(results_translated[key]) == len(results_translated[mask_key])
        gt_bboxes = _translate_bbox(copy.deepcopy(results[key]), offset, direction, h, w)
        valid_inds = (gt_bboxes[:, 2] - gt_bboxes[:, 0] > min_size) & (gt_bboxes[:, 3] - gt_bboxes[:, 1] > min_size)
        gt_bboxes = gt_bboxes[valid_inds]
        assert np.equal(gt_bboxes, results_translated[key]).all()
        if mask_key not in results:
            continue
        masks, masks_translated = (results[mask_key].to_ndarray(), results_translated[mask_key].to_ndarray())
        assert masks.dtype == masks_translated.dtype
        if direction == 'horizontal':
            masks_pad = _pad(h, abs(offset), masks.shape[0], 0, axis=0, dtype=masks.dtype)
            if offset <= 0:
                gt_masks = np.concatenate((masks[:, :, -offset:], masks_pad), axis=-1)
            else:
                gt_masks = np.concatenate((masks_pad, masks[:, :, :-offset]), axis=-1)
        else:
            masks_pad = _pad(abs(offset), w, masks.shape[0], 0, axis=0, dtype=masks.dtype)
            if offset <= 0:
                gt_masks = np.concatenate((masks[:, -offset:, :], masks_pad), axis=1)
            else:
                gt_masks = np.concatenate((masks_pad, masks[:, :-offset, :]), axis=1)
        gt_masks = gt_masks[valid_inds]
        assert np.equal(gt_masks, masks_translated).all()

def _translate_bbox(bboxes, offset, direction, max_h, max_w):
    if direction == 'horizontal':
        bboxes[:, 0::2] = bboxes[:, 0::2] + offset
    elif direction == 'vertical':
        bboxes[:, 1::2] = bboxes[:, 1::2] + offset
    else:
        raise ValueError
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, max_w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, max_h)
    return bboxes

def _pad(h, w, c, pad_val, axis=-1, dtype=np.float32):
    assert isinstance(pad_val, (int, float, tuple))
    if isinstance(pad_val, (int, float)):
        pad_val = tuple([pad_val] * c)
    assert len(pad_val) == c
    pad_data = np.stack([np.ones((h, w)) * pad_val[i] for i in range(c)], axis=axis).astype(dtype)
    return pad_data

def _check_img_seg(results, results_translated, keys, offset, fill_val, direction):
    for key in keys:
        assert isinstance(results_translated[key], type(results[key]))
        data, data_translated = (results[key], results_translated[key])
        if 'mask' in key:
            data, data_translated = (data.to_ndarray(), data_translated.to_ndarray())
        assert data.dtype == data_translated.dtype
        if 'img' in key:
            data, data_translated = (data.transpose((2, 0, 1)), data_translated.transpose((2, 0, 1)))
        elif 'seg' in key:
            data, data_translated = (data[None, :, :], data_translated[None, :, :])
        c, h, w = data.shape
        if direction == 'horizontal':
            data_pad = _pad(h, abs(offset), c, fill_val, axis=0, dtype=data.dtype)
            if offset <= 0:
                data_gt = np.concatenate((data[:, :, -offset:], data_pad), axis=-1)
            else:
                data_gt = np.concatenate((data_pad, data[:, :, :-offset]), axis=-1)
        else:
            data_pad = _pad(abs(offset), w, c, fill_val, axis=0, dtype=data.dtype)
            if offset <= 0:
                data_gt = np.concatenate((data[:, -offset:, :], data_pad), axis=1)
            else:
                data_gt = np.concatenate((data_pad, data[:, :-offset, :]), axis=1)
        if 'mask' in key:
            pass
        else:
            assert np.equal(data_gt, data_translated).all()

def check_translate(results, results_translated, offset, img_fill_val, seg_ignore_label, direction, min_size=0):
    _check_keys(results, results_translated)
    _check_img_seg(results, results_translated, results.get('img_fields', ['img']), offset, img_fill_val, direction)
    _check_img_seg(results, results_translated, results.get('seg_fields', []), offset, seg_ignore_label, direction)
    _check_bbox_mask(results, results_translated, offset, direction, min_size)

def _check_keys(results, results_translated):
    assert len(set(results.keys()).difference(set(results_translated.keys()))) == 0
    assert len(set(results_translated.keys()).difference(set(results.keys()))) == 0

def _translated_gt(masks, direction, offset, out_shape):
    translated_masks = []
    for poly_per_obj in masks:
        translated_poly_per_obj = []
        for p in poly_per_obj:
            p = p.copy()
            if direction == 'horizontal':
                p[0::2] = np.clip(p[0::2] + offset, 0, out_shape[1])
            elif direction == 'vertical':
                p[1::2] = np.clip(p[1::2] + offset, 0, out_shape[0])
            if PolygonMasks([[p]], *out_shape).areas[0] > 0:
                translated_poly_per_obj.append(p)
        if len(translated_poly_per_obj):
            translated_masks.append(translated_poly_per_obj)
    translated_masks = PolygonMasks(translated_masks, *out_shape)
    return translated_masks

def test_shear():
    with pytest.raises(AssertionError):
        transform = dict(type='Shear', level=1, max_shear_magnitude=(0.5,))
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='Shear', level=2, max_shear_magnitude=1.2)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(ValueError):
        transform = dict(type='Shear', level=2, img_fill_val=[128])
        build_from_cfg(transform, PIPELINES)
    results = construct_toy_data()
    img_fill_val = (104, 116, 124)
    seg_ignore_label = 255
    transform = dict(type='Shear', level=0, prob=1.0, img_fill_val=img_fill_val, seg_ignore_label=seg_ignore_label, direction='horizontal')
    shear_module = build_from_cfg(transform, PIPELINES)
    results_wo_shear = shear_module(copy.deepcopy(results))
    check_result_same(results, results_wo_shear)
    transform = dict(type='Shear', level=0, prob=1.0, img_fill_val=img_fill_val, seg_ignore_label=seg_ignore_label, direction='vertical')
    shear_module = build_from_cfg(transform, PIPELINES)
    results_wo_shear = shear_module(copy.deepcopy(results))
    check_result_same(results, results_wo_shear)
    transform = dict(type='Shear', level=10, prob=0.0, img_fill_val=img_fill_val, direction='vertical')
    shear_module = build_from_cfg(transform, PIPELINES)
    results_wo_shear = shear_module(copy.deepcopy(results))
    check_result_same(results, results_wo_shear)
    transform = dict(type='Shear', level=10, prob=1.0, img_fill_val=img_fill_val, direction='horizontal', max_shear_magnitude=1.0, random_negative_prob=0.0)
    shear_module = build_from_cfg(transform, PIPELINES)
    results_sheared = shear_module(copy.deepcopy(results))
    results_gt = copy.deepcopy(results)
    img_s = np.array([[1, 2, 3, 4], [0, 5, 6, 7]], dtype=np.uint8)
    img_s = np.stack([img_s, img_s, img_s], axis=-1)
    img_s[1, 0, :] = np.array(img_fill_val)
    results_gt['img'] = img_s
    results_gt['gt_bboxes'] = np.array([[0.0, 0.0, 3.0, 1.0]], dtype=np.float32)
    results_gt['gt_bboxes_ignore'] = np.array([[2.0, 0.0, 4.0, 1.0]], dtype=np.float32)
    gt_masks = np.array([[0, 1, 1, 0], [0, 0, 1, 0]], dtype=np.uint8)[None, :, :]
    results_gt['gt_masks'] = BitmapMasks(gt_masks, 2, 4)
    results_gt['gt_semantic_seg'] = np.array([[1, 2, 3, 4], [255, 5, 6, 7]], dtype=results['gt_semantic_seg'].dtype)
    check_result_same(results_gt, results_sheared)
    results = construct_toy_data(poly2mask=False)
    results_sheared = shear_module(copy.deepcopy(results))
    print(results_sheared['gt_masks'])
    gt_masks = [[np.array([0, 0, 2, 0, 3, 1, 1, 1], dtype=np.float)]]
    results_gt['gt_masks'] = PolygonMasks(gt_masks, 2, 4)
    check_result_same(results_gt, results_sheared)
    img_fill_val = 128
    results = construct_toy_data()
    transform = dict(type='Shear', level=10, prob=1.0, img_fill_val=img_fill_val, direction='vertical', max_shear_magnitude=1.0, random_negative_prob=1.0)
    shear_module = build_from_cfg(transform, PIPELINES)
    results_sheared = shear_module(copy.deepcopy(results))
    results_gt = copy.deepcopy(results)
    img_s = np.array([[1, 6, img_fill_val, img_fill_val], [5, img_fill_val, img_fill_val, img_fill_val]], dtype=np.uint8)
    img_s = np.stack([img_s, img_s, img_s], axis=-1)
    results_gt['img'] = img_s
    results_gt['gt_bboxes'] = np.empty((0, 4), dtype=np.float32)
    results_gt['gt_labels'] = np.empty((0,), dtype=np.int64)
    results_gt['gt_bboxes_ignore'] = np.empty((0, 4), dtype=np.float32)
    gt_masks = np.array([[0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)[None, :, :]
    results_gt['gt_masks'] = BitmapMasks(gt_masks, 2, 4)
    results_gt['gt_semantic_seg'] = np.array([[1, 6, 255, 255], [5, 255, 255, 255]], dtype=results['gt_semantic_seg'].dtype)
    check_result_same(results_gt, results_sheared)
    results = construct_toy_data(poly2mask=False)
    results_sheared = shear_module(copy.deepcopy(results))
    gt_masks = [[np.array([0, 0, 2, 0, 2, 0, 0, 1], dtype=np.float)]]
    results_gt['gt_masks'] = PolygonMasks(gt_masks, 2, 4)
    check_result_same(results_gt, results_sheared)
    results = construct_toy_data()
    results['gt_masks'] = BitmapMasks(np.array([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=np.uint8)[None, :, :], 2, 4)
    results['gt_bboxes'] = np.array([[1.0, 0.0, 2.0, 1.0]], dtype=np.float32)
    results_sheared_bitmap = shear_module(copy.deepcopy(results))
    check_result_same(results_sheared_bitmap, results_sheared)
    policies = [[dict(type='Shear', level=10, prob=1.0)]]
    autoaug = dict(type='AutoAugment', policies=policies)
    autoaug_module = build_from_cfg(autoaug, PIPELINES)
    autoaug_module(copy.deepcopy(results))
    policies = [[dict(type='Shear', level=10, prob=1.0), dict(type='Shear', level=8, img_fill_val=img_fill_val, direction='vertical', max_shear_magnitude=1.0)]]
    autoaug = dict(type='AutoAugment', policies=policies)
    autoaug_module = build_from_cfg(autoaug, PIPELINES)
    autoaug_module(copy.deepcopy(results))

def test_adjust_color():
    results = construct_toy_data()
    transform = dict(type='ColorTransform', prob=0, level=10)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], results['img'])
    img = results['img']
    transform = dict(type='ColorTransform', prob=1, level=10)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], img)
    transform_module.factor = 0
    img_gray = mmcv.bgr2gray(img.copy())
    img_r = np.stack([img_gray, img_gray, img_gray], axis=-1)
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], img_r)
    transform_module.factor = 0.5
    results_transformed = transform_module(copy.deepcopy(results))
    img = results['img']
    assert_array_equal(results_transformed['img'], np.round(np.clip(img * 0.5 + img_r * 0.5, 0, 255)).astype(img.dtype))

def test_imequalize(nb_rand_test=100):

    def _imequalize(img):
        from PIL import Image, ImageOps
        img = Image.fromarray(img)
        equalized_img = np.asarray(ImageOps.equalize(img))
        return equalized_img
    results = construct_toy_data()
    transform = dict(type='EqualizeTransform', prob=0)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], results['img'])
    transform = dict(type='EqualizeTransform', prob=1.0)
    transform_module = build_from_cfg(transform, PIPELINES)
    img = np.array([[0, 0, 0], [120, 120, 120], [255, 255, 255]], dtype=np.uint8)
    img = np.stack([img, img, img], axis=-1)
    results['img'] = img
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], img)
    for _ in range(nb_rand_test):
        img = np.clip(np.random.uniform(0, 1, (1000, 1200, 3)) * 260, 0, 255).astype(np.uint8)
        results['img'] = img
        results_transformed = transform_module(copy.deepcopy(results))
        assert_array_equal(results_transformed['img'], _imequalize(img))

def _imequalize(img):
    from PIL import Image, ImageOps
    img = Image.fromarray(img)
    equalized_img = np.asarray(ImageOps.equalize(img))
    return equalized_img

def test_adjust_brightness(nb_rand_test=100):

    def _adjust_brightness(img, factor):
        from PIL import Image
        from PIL.ImageEnhance import Brightness
        img = Image.fromarray(img)
        brightened_img = Brightness(img).enhance(factor)
        return np.asarray(brightened_img)
    results = construct_toy_data()
    transform = dict(type='BrightnessTransform', level=10, prob=0)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], results['img'])
    transform = dict(type='BrightnessTransform', level=10, prob=1.0)
    transform_module = build_from_cfg(transform, PIPELINES)
    transform_module.factor = 1.0
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], results['img'])
    transform_module.factor = 0.0
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], np.zeros_like(results['img']))
    for _ in range(nb_rand_test):
        img = np.clip(np.random.uniform(0, 1, (1000, 1200, 3)) * 260, 0, 255).astype(np.uint8)
        factor = np.random.uniform()
        transform_module.factor = factor
        results['img'] = img
        np.testing.assert_allclose(transform_module(copy.deepcopy(results))['img'].astype(np.int32), _adjust_brightness(img, factor).astype(np.int32), rtol=0, atol=1)

def _adjust_brightness(img, factor):
    from PIL import Image
    from PIL.ImageEnhance import Brightness
    img = Image.fromarray(img)
    brightened_img = Brightness(img).enhance(factor)
    return np.asarray(brightened_img)

def test_adjust_contrast(nb_rand_test=100):

    def _adjust_contrast(img, factor):
        from PIL import Image
        from PIL.ImageEnhance import Contrast
        img = Image.fromarray(img[..., ::-1], mode='RGB')
        contrasted_img = Contrast(img).enhance(factor)
        return np.asarray(contrasted_img)[..., ::-1]
    results = construct_toy_data()
    transform = dict(type='ContrastTransform', level=10, prob=0)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], results['img'])
    transform = dict(type='ContrastTransform', level=10, prob=1.0)
    transform_module = build_from_cfg(transform, PIPELINES)
    transform_module.factor = 1.0
    results_transformed = transform_module(copy.deepcopy(results))
    assert_array_equal(results_transformed['img'], results['img'])
    transform_module.factor = 0.0
    results_transformed = transform_module(copy.deepcopy(results))
    np.testing.assert_allclose(results_transformed['img'], _adjust_contrast(results['img'], 0.0), rtol=0, atol=1)
    for _ in range(nb_rand_test):
        img = np.clip(np.random.uniform(0, 1, (1200, 1000, 3)) * 260, 0, 255).astype(np.uint8)
        factor = np.random.uniform()
        transform_module.factor = factor
        results['img'] = img
        results_transformed = transform_module(copy.deepcopy(results))
        np.testing.assert_allclose(transform_module(copy.deepcopy(results))['img'].astype(np.int32), _adjust_contrast(results['img'], factor).astype(np.int32), rtol=0, atol=1)

def _adjust_contrast(img, factor):
    from PIL import Image
    from PIL.ImageEnhance import Contrast
    img = Image.fromarray(img[..., ::-1], mode='RGB')
    contrasted_img = Contrast(img).enhance(factor)
    return np.asarray(contrasted_img)[..., ::-1]

def test_rotate():
    with pytest.raises(AssertionError):
        transform = dict(type='Rotate', level=1, max_rotate_angle=(30,))
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='Rotate', level=2, scale=(1.2,))
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(ValueError):
        transform = dict(type='Rotate', level=2, img_fill_val=[128])
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='Rotate', level=2, center=(0.5,))
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='Rotate', level=2, center=[0, 0])
        build_from_cfg(transform, PIPELINES)
    results = construct_toy_data()
    img_fill_val = (104, 116, 124)
    seg_ignore_label = 255
    transform = dict(type='Rotate', level=0, prob=1.0, img_fill_val=img_fill_val, seg_ignore_label=seg_ignore_label)
    rotate_module = build_from_cfg(transform, PIPELINES)
    results_wo_rotate = rotate_module(copy.deepcopy(results))
    check_result_same(results, results_wo_rotate)
    transform = dict(type='Rotate', level=10, prob=0.0, img_fill_val=img_fill_val, scale=0.6)
    rotate_module = build_from_cfg(transform, PIPELINES)
    results_wo_rotate = rotate_module(copy.deepcopy(results))
    check_result_same(results, results_wo_rotate)
    results = construct_toy_data()
    img_fill_val = 128
    transform = dict(type='Rotate', level=10, max_rotate_angle=90, img_fill_val=img_fill_val, random_negative_prob=0.0, prob=1.0)
    rotate_module = build_from_cfg(transform, PIPELINES)
    results_rotated = rotate_module(copy.deepcopy(results))
    img_r = np.array([[img_fill_val, 6, 2, img_fill_val], [img_fill_val, 7, 3, img_fill_val]]).astype(np.uint8)
    img_r = np.stack([img_r, img_r, img_r], axis=-1)
    results_gt = copy.deepcopy(results)
    results_gt['img'] = img_r
    results_gt['gt_bboxes'] = np.array([[1.0, 0.0, 2.0, 1.0]], dtype=np.float32)
    results_gt['gt_bboxes_ignore'] = np.empty((0, 4), dtype=np.float32)
    gt_masks = np.array([[0, 1, 1, 0], [0, 0, 1, 0]], dtype=np.uint8)[None, :, :]
    results_gt['gt_masks'] = BitmapMasks(gt_masks, 2, 4)
    results_gt['gt_semantic_seg'] = np.array([[255, 6, 2, 255], [255, 7, 3, 255]]).astype(results['gt_semantic_seg'].dtype)
    check_result_same(results_gt, results_rotated)
    results = construct_toy_data(poly2mask=False)
    results_rotated = rotate_module(copy.deepcopy(results))
    gt_masks = [[np.array([2, 0, 2, 1, 1, 1, 1, 0], dtype=np.float)]]
    results_gt['gt_masks'] = PolygonMasks(gt_masks, 2, 4)
    check_result_same(results_gt, results_rotated)
    img_fill_val = (104, 116, 124)
    transform = dict(type='Rotate', level=10, max_rotate_angle=90, center=(0, 0), img_fill_val=img_fill_val, random_negative_prob=1.0, prob=1.0)
    results = construct_toy_data()
    rotate_module = build_from_cfg(transform, PIPELINES)
    results_rotated = rotate_module(copy.deepcopy(results))
    results_gt = copy.deepcopy(results)
    h, w = results['img'].shape[:2]
    img_r = np.stack([np.ones((h, w)) * img_fill_val[0], np.ones((h, w)) * img_fill_val[1], np.ones((h, w)) * img_fill_val[2]], axis=-1).astype(np.uint8)
    img_r[0, 0, :] = 1
    img_r[0, 1, :] = 5
    results_gt['img'] = img_r
    results_gt['gt_bboxes'] = np.empty((0, 4), dtype=np.float32)
    results_gt['gt_bboxes_ignore'] = np.empty((0, 4), dtype=np.float32)
    results_gt['gt_labels'] = np.empty((0,), dtype=np.int64)
    gt_masks = np.empty((0, h, w), dtype=np.uint8)
    results_gt['gt_masks'] = BitmapMasks(gt_masks, h, w)
    gt_seg = (np.ones((h, w)) * 255).astype(results['gt_semantic_seg'].dtype)
    gt_seg[0, 0], gt_seg[0, 1] = (1, 5)
    results_gt['gt_semantic_seg'] = gt_seg
    check_result_same(results_gt, results_rotated)
    transform = dict(type='Rotate', level=10, max_rotate_angle=90, center=0, img_fill_val=img_fill_val, random_negative_prob=1.0, prob=1.0)
    rotate_module = build_from_cfg(transform, PIPELINES)
    results_rotated = rotate_module(copy.deepcopy(results))
    check_result_same(results_gt, results_rotated)
    results = construct_toy_data(poly2mask=False)
    results_rotated = rotate_module(copy.deepcopy(results))
    gt_masks = [[np.array([0, 0, 0, 0, 1, 0, 1, 0], dtype=np.float)]]
    results_gt['gt_masks'] = PolygonMasks(gt_masks, 2, 4)
    check_result_same(results_gt, results_rotated)
    policies = [[dict(type='Rotate', level=10, prob=1.0)]]
    autoaug = dict(type='AutoAugment', policies=policies)
    autoaug_module = build_from_cfg(autoaug, PIPELINES)
    autoaug_module(copy.deepcopy(results))
    policies = [[dict(type='Rotate', level=10, prob=1.0), dict(type='Rotate', level=8, max_rotate_angle=90, center=0, img_fill_val=img_fill_val)]]
    autoaug = dict(type='AutoAugment', policies=policies)
    autoaug_module = build_from_cfg(autoaug, PIPELINES)
    autoaug_module(copy.deepcopy(results))

