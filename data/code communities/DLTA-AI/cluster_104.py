# Cluster 104

@pytest.mark.parametrize('target, kwargs', _build_filter_annotations_args())
def test_filter_annotations(target, kwargs):
    filter_ann = FilterAnnotations(**kwargs)
    bboxes = np.array([[2.0, 10.0, 4.0, 14.0], [2.0, 10.0, 2.1, 10.1]])
    raw_masks = np.zeros((2, 24, 24))
    raw_masks[0, 10:14, 2:4] = 1
    bitmap_masks = BitmapMasks(raw_masks, 24, 24)
    results = dict(gt_bboxes=bboxes, gt_masks=bitmap_masks)
    results = filter_ann(results)
    if results is not None:
        results = results['gt_bboxes'].shape[0]
    assert results == target
    polygons = [[np.array([2.0, 10.0, 4.0, 10.0, 4.0, 14.0, 2.0, 14.0])], [np.array([2.0, 10.0, 2.1, 10.0, 2.1, 10.1, 2.0, 10.1])]]
    polygon_masks = PolygonMasks(polygons, 24, 24)
    results = dict(gt_bboxes=bboxes, gt_masks=polygon_masks)
    results = filter_ann(results)
    if results is not None:
        results = len(results.get('gt_masks').masks)
    assert results == target

def _build_filter_annotations_args():
    kwargs = (dict(min_gt_bbox_wh=(100, 100)), dict(min_gt_bbox_wh=(100, 100), keep_empty=False), dict(min_gt_bbox_wh=(1, 1)), dict(min_gt_bbox_wh=(0.01, 0.01)), dict(min_gt_bbox_wh=(0.01, 0.01), by_mask=True), dict(by_mask=True), dict(by_box=False, by_mask=True))
    targets = (None, 0, 1, 2, 1, 1, 1)
    return list(zip(targets, kwargs))

