# Cluster 108

def test_copypaste():
    dst_results, src_results = (dict(), dict())
    img = mmcv.imread(osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    dst_results['img'] = img.copy()
    src_results['img'] = img.copy()
    h, w, _ = img.shape
    dst_bboxes = np.array([[0.2 * w, 0.2 * h, 0.4 * w, 0.4 * h], [0.5 * w, 0.5 * h, 0.6 * w, 0.6 * h]], dtype=np.float32)
    src_bboxes = np.array([[0.1 * w, 0.1 * h, 0.3 * w, 0.5 * h], [0.4 * w, 0.4 * h, 0.7 * w, 0.7 * h], [0.8 * w, 0.8 * h, 0.9 * w, 0.9 * h]], dtype=np.float32)
    dst_labels = np.ones(dst_bboxes.shape[0], dtype=np.int64)
    src_labels = np.ones(src_bboxes.shape[0], dtype=np.int64) * 2
    dst_masks = create_full_masks(dst_bboxes, w, h)
    src_masks = create_full_masks(src_bboxes, w, h)
    dst_results['gt_bboxes'] = dst_bboxes.copy()
    src_results['gt_bboxes'] = src_bboxes.copy()
    dst_results['gt_labels'] = dst_labels.copy()
    src_results['gt_labels'] = src_labels.copy()
    dst_results['gt_masks'] = copy.deepcopy(dst_masks)
    src_results['gt_masks'] = copy.deepcopy(src_masks)
    results = copy.deepcopy(dst_results)
    transform = dict(type='CopyPaste', selected=False)
    copypaste_module = build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        copypaste_module(results)
    results['mix_results'] = [copy.deepcopy(src_results)]
    results = copypaste_module(results)
    assert results['img'].shape[:2] == (h, w)
    assert results['gt_bboxes'].shape[0] == dst_bboxes.shape[0] + src_bboxes.shape[0] - 1
    assert results['gt_labels'].shape[0] == dst_labels.shape[0] + src_labels.shape[0] - 1
    assert results['gt_masks'].masks.shape[0] == dst_masks.masks.shape[0] + src_masks.masks.shape[0] - 1
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    ori_bbox = dst_bboxes[0]
    occ_bbox = results['gt_bboxes'][0]
    ori_mask = dst_masks.masks[0]
    occ_mask = results['gt_masks'].masks[0]
    assert ori_mask.sum() > occ_mask.sum()
    assert np.all(np.abs(occ_bbox - ori_bbox) <= copypaste_module.bbox_occluded_thr) or occ_mask.sum() > copypaste_module.mask_occluded_thr
    transform = dict(type='CopyPaste')
    copypaste_module = build_from_cfg(transform, PIPELINES)
    results = copy.deepcopy(dst_results)
    results['mix_results'] = [copy.deepcopy(src_results)]
    copypaste_module(results)
    results = copy.deepcopy(dst_results)
    valid_inds = [False] * src_bboxes.shape[0]
    src_results['gt_bboxes'] = src_bboxes[valid_inds]
    src_results['gt_labels'] = src_labels[valid_inds]
    src_results['gt_masks'] = src_masks[valid_inds]
    results['mix_results'] = [copy.deepcopy(src_results)]
    copypaste_module(results)
    dst_results.pop('gt_masks')
    src_results.pop('gt_masks')
    dst_bboxes = dst_results['gt_bboxes']
    src_bboxes = src_results['gt_bboxes']
    dst_masks = create_full_masks(dst_bboxes, w, h)
    src_masks = create_full_masks(src_bboxes, w, h)
    results = copy.deepcopy(dst_results)
    results['mix_results'] = [copy.deepcopy(src_results)]
    results = copypaste_module(results)
    result_masks = create_full_masks(results['gt_bboxes'], w, h)
    result_masks_np = np.where(result_masks.to_ndarray().sum(0) > 0, 1, 0)
    masks_np = np.where(src_masks.to_ndarray().sum(0) + dst_masks.to_ndarray().sum(0) > 0, 1, 0)
    assert np.all(result_masks_np == masks_np)
    assert 'gt_masks' not in results

def create_full_masks(gt_bboxes, img_w, img_h):
    xmin, ymin = (gt_bboxes[:, 0:1], gt_bboxes[:, 1:2])
    xmax, ymax = (gt_bboxes[:, 2:3], gt_bboxes[:, 3:4])
    gt_masks = np.zeros((len(gt_bboxes), img_h, img_w), dtype=np.uint8)
    for i in range(len(gt_bboxes)):
        gt_masks[i, int(ymin[i]):int(ymax[i]), int(xmin[i]):int(xmax[i])] = 1
    gt_masks = BitmapMasks(gt_masks, img_h, img_w)
    return gt_masks

