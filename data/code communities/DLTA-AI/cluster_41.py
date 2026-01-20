# Cluster 41

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-06):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]
    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols
    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])
        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        return x.float().clamp(min, max).half()
    return x.clamp(min, max)

def eval_recalls(gts, proposals, proposal_nums=None, iou_thrs=0.5, logger=None, use_legacy_coordinate=False):
    """Calculate recalls.

    Args:
        gts (list[ndarray]): a list of arrays of shape (n, 4)
        proposals (list[ndarray]): a list of arrays of shape (k, 4) or (k, 5)
        proposal_nums (int | Sequence[int]): Top N proposals to be evaluated.
        iou_thrs (float | Sequence[float]): IoU thresholds. Default: 0.5.
        logger (logging.Logger | str | None): The way to print the recall
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        use_legacy_coordinate (bool): Whether use coordinate system
            in mmdet v1.x. "1" was added to both height and width
            which means w, h should be
            computed as 'x2 - x1 + 1` and 'y2 - y1 + 1'. Default: False.


    Returns:
        ndarray: recalls of different ious and proposal nums
    """
    img_num = len(gts)
    assert img_num == len(proposals)
    proposal_nums, iou_thrs = set_recall_param(proposal_nums, iou_thrs)
    all_ious = []
    for i in range(img_num):
        if proposals[i].ndim == 2 and proposals[i].shape[1] == 5:
            scores = proposals[i][:, 4]
            sort_idx = np.argsort(scores)[::-1]
            img_proposal = proposals[i][sort_idx, :]
        else:
            img_proposal = proposals[i]
        prop_num = min(img_proposal.shape[0], proposal_nums[-1])
        if gts[i] is None or gts[i].shape[0] == 0:
            ious = np.zeros((0, img_proposal.shape[0]), dtype=np.float32)
        else:
            ious = bbox_overlaps(gts[i], img_proposal[:prop_num, :4], use_legacy_coordinate=use_legacy_coordinate)
        all_ious.append(ious)
    all_ious = np.array(all_ious)
    recalls = _recalls(all_ious, proposal_nums, iou_thrs)
    print_recall_summary(recalls, proposal_nums, iou_thrs, logger=logger)
    return recalls

def set_recall_param(proposal_nums, iou_thrs):
    """Check proposal_nums and iou_thrs and set correct format."""
    if isinstance(proposal_nums, Sequence):
        _proposal_nums = np.array(proposal_nums)
    elif isinstance(proposal_nums, int):
        _proposal_nums = np.array([proposal_nums])
    else:
        _proposal_nums = proposal_nums
    if iou_thrs is None:
        _iou_thrs = np.array([0.5])
    elif isinstance(iou_thrs, Sequence):
        _iou_thrs = np.array(iou_thrs)
    elif isinstance(iou_thrs, float):
        _iou_thrs = np.array([iou_thrs])
    else:
        _iou_thrs = iou_thrs
    return (_proposal_nums, _iou_thrs)

def _recalls(all_ious, proposal_nums, thrs):
    img_num = all_ious.shape[0]
    total_gt_num = sum([ious.shape[0] for ious in all_ious])
    _ious = np.zeros((proposal_nums.size, total_gt_num), dtype=np.float32)
    for k, proposal_num in enumerate(proposal_nums):
        tmp_ious = np.zeros(0)
        for i in range(img_num):
            ious = all_ious[i][:, :proposal_num].copy()
            gt_ious = np.zeros(ious.shape[0])
            if ious.size == 0:
                tmp_ious = np.hstack((tmp_ious, gt_ious))
                continue
            for j in range(ious.shape[0]):
                gt_max_overlaps = ious.argmax(axis=1)
                max_ious = ious[np.arange(0, ious.shape[0]), gt_max_overlaps]
                gt_idx = max_ious.argmax()
                gt_ious[j] = max_ious[gt_idx]
                box_idx = gt_max_overlaps[gt_idx]
                ious[gt_idx, :] = -1
                ious[:, box_idx] = -1
            tmp_ious = np.hstack((tmp_ious, gt_ious))
        _ious[k, :] = tmp_ious
    _ious = np.fliplr(np.sort(_ious, axis=1))
    recalls = np.zeros((proposal_nums.size, thrs.size))
    for i, thr in enumerate(thrs):
        recalls[:, i] = (_ious >= thr).sum(axis=1) / float(total_gt_num)
    return recalls

def tpfp_imagenet(det_bboxes, gt_bboxes, gt_bboxes_ignore=None, default_iou_thr=0.5, area_ranges=None, use_legacy_coordinate=False, **kwargs):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        default_iou_thr (float): IoU threshold to be considered as matched for
            medium and large bboxes (small ones have special rules).
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """
    if not use_legacy_coordinate:
        extra_length = 0.0
    else:
        extra_length = 1.0
    gt_ignore_inds = np.concatenate((np.zeros(gt_bboxes.shape[0], dtype=np.bool), np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + extra_length) * (det_bboxes[:, 3] - det_bboxes[:, 1] + extra_length)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return (tp, fp)
    ious = bbox_overlaps(det_bboxes, gt_bboxes - 1, use_legacy_coordinate=use_legacy_coordinate)
    gt_w = gt_bboxes[:, 2] - gt_bboxes[:, 0] + extra_length
    gt_h = gt_bboxes[:, 3] - gt_bboxes[:, 1] + extra_length
    iou_thrs = np.minimum(gt_w * gt_h / ((gt_w + 10.0) * (gt_h + 10.0)), default_iou_thr)
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = gt_w * gt_h
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            max_iou = -1
            matched_gt = -1
            for j in range(num_gts):
                if gt_covered[j]:
                    continue
                elif ious[i, j] >= iou_thrs[j] and ious[i, j] > max_iou:
                    max_iou = ious[i, j]
                    matched_gt = j
            if matched_gt >= 0:
                gt_covered[matched_gt] = 1
                if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
                    tp[k, i] = 1
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0] + extra_length) * (bbox[3] - bbox[1] + extra_length)
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return (tp, fp)

def tpfp_default(det_bboxes, gt_bboxes, gt_bboxes_ignore=None, iou_thr=0.5, area_ranges=None, use_legacy_coordinate=False, **kwargs):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be
            evaluated, in the format [(min1, max1), (min2, max2), ...].
            Default: None.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """
    if not use_legacy_coordinate:
        extra_length = 0.0
    else:
        extra_length = 1.0
    gt_ignore_inds = np.concatenate((np.zeros(gt_bboxes.shape[0], dtype=np.bool), np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + extra_length) * (det_bboxes[:, 3] - det_bboxes[:, 1] + extra_length)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return (tp, fp)
    ious = bbox_overlaps(det_bboxes, gt_bboxes, use_legacy_coordinate=use_legacy_coordinate)
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + extra_length) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + extra_length)
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0] + extra_length) * (bbox[3] - bbox[1] + extra_length)
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return (tp, fp)

def tpfp_openimages(det_bboxes, gt_bboxes, gt_bboxes_ignore=None, iou_thr=0.5, area_ranges=None, use_legacy_coordinate=False, gt_bboxes_group_of=None, use_group_of=True, ioa_thr=0.5, **kwargs):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be
            evaluated, in the format [(min1, max1), (min2, max2), ...].
            Default: None.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.
        gt_bboxes_group_of (ndarray): GT group_of of this image, of shape
            (k, 1). Default: None
        use_group_of (bool): Whether to use group of when calculate TP and FP,
            which only used in OpenImages evaluation. Default: True.
        ioa_thr (float | None): IoA threshold to be considered as matched,
            which only used in OpenImages evaluation. Default: 0.5.

    Returns:
        tuple[np.ndarray]: Returns a tuple (tp, fp, det_bboxes), where
        (tp, fp) whose elements are 0 and 1. The shape of each array is
        (num_scales, m). (det_bboxes) whose will filter those are not
        matched by group of gts when processing Open Images evaluation.
        The shape is (num_scales, m).
    """
    if not use_legacy_coordinate:
        extra_length = 0.0
    else:
        extra_length = 1.0
    gt_ignore_inds = np.concatenate((np.zeros(gt_bboxes.shape[0], dtype=np.bool), np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + extra_length) * (det_bboxes[:, 3] - det_bboxes[:, 1] + extra_length)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return (tp, fp, det_bboxes)
    if gt_bboxes_group_of is not None and use_group_of:
        assert gt_bboxes_group_of.shape[0] == gt_bboxes.shape[0]
        non_group_gt_bboxes = gt_bboxes[~gt_bboxes_group_of]
        group_gt_bboxes = gt_bboxes[gt_bboxes_group_of]
        num_gts_group = group_gt_bboxes.shape[0]
        ious = bbox_overlaps(det_bboxes, non_group_gt_bboxes)
        ioas = bbox_overlaps(det_bboxes, group_gt_bboxes, mode='iof')
    else:
        ious = bbox_overlaps(det_bboxes, gt_bboxes, use_legacy_coordinate=use_legacy_coordinate)
        ioas = None
    if ious.shape[1] > 0:
        ious_max = ious.max(axis=1)
        ious_argmax = ious.argmax(axis=1)
        sort_inds = np.argsort(-det_bboxes[:, -1])
        for k, (min_area, max_area) in enumerate(area_ranges):
            gt_covered = np.zeros(num_gts, dtype=bool)
            if min_area is None:
                gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
            else:
                gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + extra_length) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + extra_length)
                gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
            for i in sort_inds:
                if ious_max[i] >= iou_thr:
                    matched_gt = ious_argmax[i]
                    if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
                        if not gt_covered[matched_gt]:
                            gt_covered[matched_gt] = True
                            tp[k, i] = 1
                        else:
                            fp[k, i] = 1
                elif min_area is None:
                    fp[k, i] = 1
                else:
                    bbox = det_bboxes[i, :4]
                    area = (bbox[2] - bbox[0] + extra_length) * (bbox[3] - bbox[1] + extra_length)
                    if area >= min_area and area < max_area:
                        fp[k, i] = 1
    elif area_ranges == [(None, None)]:
        fp[...] = 1
    else:
        det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + extra_length) * (det_bboxes[:, 3] - det_bboxes[:, 1] + extra_length)
        for i, (min_area, max_area) in enumerate(area_ranges):
            fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
    if ioas is None or ioas.shape[1] <= 0:
        return (tp, fp, det_bboxes)
    else:
        det_bboxes_group = np.zeros((num_scales, ioas.shape[1], det_bboxes.shape[1]), dtype=float)
        match_group_of = np.zeros((num_scales, num_dets), dtype=bool)
        tp_group = np.zeros((num_scales, num_gts_group), dtype=np.float32)
        ioas_max = ioas.max(axis=1)
        ioas_argmax = ioas.argmax(axis=1)
        sort_inds = np.argsort(-det_bboxes[:, -1])
        for k, (min_area, max_area) in enumerate(area_ranges):
            box_is_covered = tp[k]
            if min_area is None:
                gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
            else:
                gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
                gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
            for i in sort_inds:
                matched_gt = ioas_argmax[i]
                if not box_is_covered[i]:
                    if ioas_max[i] >= ioa_thr:
                        if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
                            if not tp_group[k, matched_gt]:
                                tp_group[k, matched_gt] = 1
                                match_group_of[k, i] = True
                            else:
                                match_group_of[k, i] = True
                            if det_bboxes_group[k, matched_gt, -1] < det_bboxes[i, -1]:
                                det_bboxes_group[k, matched_gt] = det_bboxes[i]
        fp_group = (tp_group <= 0).astype(float)
        tps = []
        fps = []
        for i in range(num_scales):
            tps.append(np.concatenate((tp[i][~match_group_of[i]], tp_group[i])))
            fps.append(np.concatenate((fp[i][~match_group_of[i]], fp_group[i])))
            det_bboxes = np.concatenate((det_bboxes[~match_group_of[i]], det_bboxes_group[i]))
        tp = np.vstack(tps)
        fp = np.vstack(fps)
        return (tp, fp, det_bboxes)

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def iou_loss(pred, target, linear=False, mode='log', eps=1e-06):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Eps to avoid log(0).

    Return:
        torch.Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn('DeprecationWarning: Setting "linear=True" in iou_loss is deprecated, please use "mode=`linear`" instead.')
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'square':
        loss = 1 - ious ** 2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def giou_loss(pred, target, eps=1e-07):
    """`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss

@LOSSES.register_module()
class IoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self, linear=False, eps=1e-06, reduction='mean', loss_weight=1.0, mode='log'):
        super(IoULoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            warnings.warn('DeprecationWarning: Setting "linear=True" in IOULoss is deprecated, please use "mode=`linear`" instead.')
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and (not torch.any(weight > 0)) and (reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * iou_loss(pred, target, weight, mode=self.mode, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss

@LOSSES.register_module()
class GIoULoss(nn.Module):

    def __init__(self, eps=1e-06, reduction='mean', loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if weight is not None and (not torch.any(weight > 0)):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(pred, target, weight, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss

@mmcv.jit(derivate=True, coderize=True)
def isr_p(cls_score, bbox_pred, bbox_targets, rois, sampling_results, loss_cls, bbox_coder, k=2, bias=0, num_class=80):
    """Importance-based Sample Reweighting (ISR_P), positive part.

    Args:
        cls_score (Tensor): Predicted classification scores.
        bbox_pred (Tensor): Predicted bbox deltas.
        bbox_targets (tuple[Tensor]): A tuple of bbox targets, the are
            labels, label_weights, bbox_targets, bbox_weights, respectively.
        rois (Tensor): Anchors (single_stage) in shape (n, 4) or RoIs
            (two_stage) in shape (n, 5).
        sampling_results (obj): Sampling results.
        loss_cls (func): Classification loss func of the head.
        bbox_coder (obj): BBox coder of the head.
        k (float): Power of the non-linear mapping.
        bias (float): Shift of the non-linear mapping.
        num_class (int): Number of classes, default: 80.

    Return:
        tuple([Tensor]): labels, imp_based_label_weights, bbox_targets,
            bbox_target_weights
    """
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets
    pos_label_inds = ((labels >= 0) & (labels < num_class)).nonzero().reshape(-1)
    pos_labels = labels[pos_label_inds]
    num_pos = float(pos_label_inds.size(0))
    if num_pos == 0:
        return (labels, label_weights, bbox_targets, bbox_weights)
    gts = list()
    last_max_gt = 0
    for i in range(len(sampling_results)):
        gt_i = sampling_results[i].pos_assigned_gt_inds
        gts.append(gt_i + last_max_gt)
        if len(gt_i) != 0:
            last_max_gt = gt_i.max() + 1
    gts = torch.cat(gts)
    assert len(gts) == num_pos
    cls_score = cls_score.detach()
    bbox_pred = bbox_pred.detach()
    if rois.size(-1) == 5:
        pos_rois = rois[pos_label_inds][:, 1:]
    else:
        pos_rois = rois[pos_label_inds]
    if bbox_pred.size(-1) > 4:
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)
        pos_delta_pred = bbox_pred[pos_label_inds, pos_labels].view(-1, 4)
    else:
        pos_delta_pred = bbox_pred[pos_label_inds].view(-1, 4)
    pos_delta_target = bbox_targets[pos_label_inds].view(-1, 4)
    pos_bbox_pred = bbox_coder.decode(pos_rois, pos_delta_pred)
    target_bbox_pred = bbox_coder.decode(pos_rois, pos_delta_target)
    ious = bbox_overlaps(pos_bbox_pred, target_bbox_pred, is_aligned=True)
    pos_imp_weights = label_weights[pos_label_inds]
    max_l_num = pos_labels.bincount().max()
    for label in pos_labels.unique():
        l_inds = (pos_labels == label).nonzero().view(-1)
        l_gts = gts[l_inds]
        for t in l_gts.unique():
            t_inds = l_inds[l_gts == t]
            t_ious = ious[t_inds]
            _, t_iou_rank_idx = t_ious.sort(descending=True)
            _, t_iou_rank = t_iou_rank_idx.sort()
            ious[t_inds] += max_l_num - t_iou_rank.float()
        l_ious = ious[l_inds]
        _, l_iou_rank_idx = l_ious.sort(descending=True)
        _, l_iou_rank = l_iou_rank_idx.sort()
        pos_imp_weights[l_inds] *= (max_l_num - l_iou_rank.float()) / max_l_num
    pos_imp_weights = (bias + pos_imp_weights * (1 - bias)).pow(k)
    pos_loss_cls = loss_cls(cls_score[pos_label_inds], pos_labels, reduction_override='none')
    if pos_loss_cls.dim() > 1:
        ori_pos_loss_cls = pos_loss_cls * label_weights[pos_label_inds][:, None]
        new_pos_loss_cls = pos_loss_cls * pos_imp_weights[:, None]
    else:
        ori_pos_loss_cls = pos_loss_cls * label_weights[pos_label_inds]
        new_pos_loss_cls = pos_loss_cls * pos_imp_weights
    pos_loss_cls_ratio = ori_pos_loss_cls.sum() / new_pos_loss_cls.sum()
    pos_imp_weights = pos_imp_weights * pos_loss_cls_ratio
    label_weights[pos_label_inds] = pos_imp_weights
    bbox_targets = (labels, label_weights, bbox_targets, bbox_weights)
    return bbox_targets

@HEADS.register_module()
class LDHead(GFLHead):
    """Localization distillation Head. (Short description)

    It utilizes the learned bbox distributions to transfer the localization
    dark knowledge from teacher to student. Original paper: `Localization
    Distillation for Object Detection. <https://arxiv.org/abs/2102.12252>`_

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss_ld (dict): Config of Localization Distillation Loss (LD),
            T is the temperature for distillation.
    """

    def __init__(self, num_classes, in_channels, loss_ld=dict(type='LocalizationDistillationLoss', loss_weight=0.25, T=10), **kwargs):
        super(LDHead, self).__init__(num_classes, in_channels, **kwargs)
        self.loss_ld = build_loss(loss_ld)

    def loss_single(self, anchors, cls_score, bbox_pred, labels, label_weights, bbox_targets, stride, soft_targets, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[tuple, Tensor]: Loss components and weight targets.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
        soft_targets = soft_targets.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]
            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = self.bbox_coder.decode(pos_anchor_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            pos_soft_targets = soft_targets[pos_inds]
            soft_corners = pos_soft_targets.reshape(-1, self.reg_max + 1)
            target_corners = self.bbox_coder.encode(pos_anchor_centers, pos_decode_bbox_targets, self.reg_max).reshape(-1)
            loss_bbox = self.loss_bbox(pos_decode_bbox_pred, pos_decode_bbox_targets, weight=weight_targets, avg_factor=1.0)
            loss_dfl = self.loss_dfl(pred_corners, target_corners, weight=weight_targets[:, None].expand(-1, 4).reshape(-1), avg_factor=4.0)
            loss_ld = self.loss_ld(pred_corners, soft_corners, weight=weight_targets[:, None].expand(-1, 4).reshape(-1), avg_factor=4.0)
        else:
            loss_ld = bbox_pred.sum() * 0
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)
        loss_cls = self.loss_cls(cls_score, (labels, score), weight=label_weights, avg_factor=num_total_samples)
        return (loss_cls, loss_bbox, loss_dfl, loss_ld, weight_targets.sum())

    def forward_train(self, x, out_teacher, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=None, proposal_cfg=None, **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple[dict, list]: The loss components and proposals of each image.

            - losses (dict[str, Tensor]): A dictionary of loss components.
            - proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        soft_target = out_teacher[1]
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, soft_target, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, soft_target, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return (losses, proposal_list)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, soft_target, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        num_total_samples = reduce_mean(torch.tensor(num_total_pos, dtype=torch.float, device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)
        losses_cls, losses_bbox, losses_dfl, losses_ld, avg_factor = multi_apply(self.loss_single, anchor_list, cls_scores, bbox_preds, labels_list, label_weights_list, bbox_targets_list, self.prior_generator.strides, soft_target, num_total_samples=num_total_samples)
        avg_factor = sum(avg_factor) + 1e-06
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox = [x / avg_factor for x in losses_bbox]
        losses_dfl = [x / avg_factor for x in losses_dfl]
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl, loss_ld=losses_ld)

@HEADS.register_module()
class FreeAnchorRetinaHead(RetinaHead):
    """FreeAnchor RetinaHead used in https://arxiv.org/abs/1909.02466.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 4.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32,
            requires_grad=True).
        pre_anchor_topk (int): Number of boxes that be token in each bag.
        bbox_thr (float): The threshold of the saturated linear function. It is
            usually the same with the IoU threshold used in NMS.
        gamma (float): Gamma parameter in focal loss.
        alpha (float): Alpha parameter in focal loss.
    """

    def __init__(self, num_classes, in_channels, stacked_convs=4, conv_cfg=None, norm_cfg=None, pre_anchor_topk=50, bbox_thr=0.6, gamma=2.0, alpha=0.5, **kwargs):
        super(FreeAnchorRetinaHead, self).__init__(num_classes, in_channels, stacked_convs, conv_cfg, norm_cfg, **kwargs)
        self.pre_anchor_topk = pre_anchor_topk
        self.bbox_thr = bbox_thr
        self.gamma = gamma
        self.alpha = alpha

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
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
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list, _ = self.get_anchors(featmap_sizes, img_metas, device=device)
        anchors = [torch.cat(anchor) for anchor in anchor_list]
        cls_scores = [cls.permute(0, 2, 3, 1).reshape(cls.size(0), -1, self.cls_out_channels) for cls in cls_scores]
        bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(bbox_pred.size(0), -1, 4) for bbox_pred in bbox_preds]
        cls_scores = torch.cat(cls_scores, dim=1)
        bbox_preds = torch.cat(bbox_preds, dim=1)
        cls_prob = torch.sigmoid(cls_scores)
        box_prob = []
        num_pos = 0
        positive_losses = []
        for _, (anchors_, gt_labels_, gt_bboxes_, cls_prob_, bbox_preds_) in enumerate(zip(anchors, gt_labels, gt_bboxes, cls_prob, bbox_preds)):
            with torch.no_grad():
                if len(gt_bboxes_) == 0:
                    image_box_prob = torch.zeros(anchors_.size(0), self.cls_out_channels).type_as(bbox_preds_)
                else:
                    pred_boxes = self.bbox_coder.decode(anchors_, bbox_preds_)
                    object_box_iou = bbox_overlaps(gt_bboxes_, pred_boxes)
                    t1 = self.bbox_thr
                    t2 = object_box_iou.max(dim=1, keepdim=True).values.clamp(min=t1 + 1e-12)
                    object_box_prob = ((object_box_iou - t1) / (t2 - t1)).clamp(min=0, max=1)
                    num_obj = gt_labels_.size(0)
                    indices = torch.stack([torch.arange(num_obj).type_as(gt_labels_), gt_labels_], dim=0)
                    object_cls_box_prob = torch.sparse_coo_tensor(indices, object_box_prob)
                    '\n                    from "start" to "end" implement:\n                    image_box_iou = torch.sparse.max(object_cls_box_prob,\n                                                     dim=0).t()\n\n                    '
                    box_cls_prob = torch.sparse.sum(object_cls_box_prob, dim=0).to_dense()
                    indices = torch.nonzero(box_cls_prob, as_tuple=False).t_()
                    if indices.numel() == 0:
                        image_box_prob = torch.zeros(anchors_.size(0), self.cls_out_channels).type_as(object_box_prob)
                    else:
                        nonzero_box_prob = torch.where(gt_labels_.unsqueeze(dim=-1) == indices[0], object_box_prob[:, indices[1]], torch.tensor([0]).type_as(object_box_prob)).max(dim=0).values
                        image_box_prob = torch.sparse_coo_tensor(indices.flip([0]), nonzero_box_prob, size=(anchors_.size(0), self.cls_out_channels)).to_dense()
                box_prob.append(image_box_prob)
            match_quality_matrix = bbox_overlaps(gt_bboxes_, anchors_)
            _, matched = torch.topk(match_quality_matrix, self.pre_anchor_topk, dim=1, sorted=False)
            del match_quality_matrix
            matched_cls_prob = torch.gather(cls_prob_[matched], 2, gt_labels_.view(-1, 1, 1).repeat(1, self.pre_anchor_topk, 1)).squeeze(2)
            matched_anchors = anchors_[matched]
            matched_object_targets = self.bbox_coder.encode(matched_anchors, gt_bboxes_.unsqueeze(dim=1).expand_as(matched_anchors))
            loss_bbox = self.loss_bbox(bbox_preds_[matched], matched_object_targets, reduction_override='none').sum(-1)
            matched_box_prob = torch.exp(-loss_bbox)
            num_pos += len(gt_bboxes_)
            positive_losses.append(self.positive_bag_loss(matched_cls_prob, matched_box_prob))
        positive_loss = torch.cat(positive_losses).sum() / max(1, num_pos)
        box_prob = torch.stack(box_prob, dim=0)
        negative_loss = self.negative_bag_loss(cls_prob, box_prob).sum() / max(1, num_pos * self.pre_anchor_topk)
        if num_pos == 0:
            positive_loss = bbox_preds.sum() * 0
        losses = {'positive_bag_loss': positive_loss, 'negative_bag_loss': negative_loss}
        return losses

    def positive_bag_loss(self, matched_cls_prob, matched_box_prob):
        """Compute positive bag loss.

        :math:`-log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) )`.

        :math:`P_{ij}^{cls}`: matched_cls_prob, classification probability of matched samples.

        :math:`P_{ij}^{loc}`: matched_box_prob, box probability of matched samples.

        Args:
            matched_cls_prob (Tensor): Classification probability of matched
                samples in shape (num_gt, pre_anchor_topk).
            matched_box_prob (Tensor): BBox probability of matched samples,
                in shape (num_gt, pre_anchor_topk).

        Returns:
            Tensor: Positive bag loss in shape (num_gt,).
        """
        matched_prob = matched_cls_prob * matched_box_prob
        weight = 1 / torch.clamp(1 - matched_prob, 1e-12, None)
        weight /= weight.sum(dim=1).unsqueeze(dim=-1)
        bag_prob = (weight * matched_prob).sum(dim=1)
        return self.alpha * F.binary_cross_entropy(bag_prob, torch.ones_like(bag_prob), reduction='none')

    def negative_bag_loss(self, cls_prob, box_prob):
        """Compute negative bag loss.

        :math:`FL((1 - P_{a_{j} \\in A_{+}}) * (1 - P_{j}^{bg}))`.

        :math:`P_{a_{j} \\in A_{+}}`: Box_probability of matched samples.

        :math:`P_{j}^{bg}`: Classification probability of negative samples.

        Args:
            cls_prob (Tensor): Classification probability, in shape
                (num_img, num_anchors, num_classes).
            box_prob (Tensor): Box probability, in shape
                (num_img, num_anchors, num_classes).

        Returns:
            Tensor: Negative bag loss in shape (num_img, num_anchors, num_classes).
        """
        prob = cls_prob * (1 - box_prob)
        prob = prob.clamp(min=EPS, max=1 - EPS)
        negative_bag_loss = prob ** self.gamma * F.binary_cross_entropy(prob, torch.zeros_like(prob), reduction='none')
        return (1 - self.alpha) * negative_bag_loss

@HEADS.register_module()
class YOLOFHead(AnchorHead):
    """YOLOFHead Paper link: https://arxiv.org/abs/2103.09460.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): The number of input channels per scale.
        cls_num_convs (int): The number of convolutions of cls branch.
           Default 2.
        reg_num_convs (int): The number of convolutions of reg branch.
           Default 4.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self, num_classes, in_channels, num_cls_convs=2, num_reg_convs=4, norm_cfg=dict(type='BN', requires_grad=True), **kwargs):
        self.num_cls_convs = num_cls_convs
        self.num_reg_convs = num_reg_convs
        self.norm_cfg = norm_cfg
        super(YOLOFHead, self).__init__(num_classes, in_channels, **kwargs)

    def _init_layers(self):
        cls_subnet = []
        bbox_subnet = []
        for i in range(self.num_cls_convs):
            cls_subnet.append(ConvModule(self.in_channels, self.in_channels, kernel_size=3, padding=1, norm_cfg=self.norm_cfg))
        for i in range(self.num_reg_convs):
            bbox_subnet.append(ConvModule(self.in_channels, self.in_channels, kernel_size=3, padding=1, norm_cfg=self.norm_cfg))
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(self.in_channels, self.num_base_priors * self.num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(self.in_channels, self.num_base_priors * 4, kernel_size=3, stride=1, padding=1)
        self.object_pred = nn.Conv2d(self.in_channels, self.num_base_priors, kernel_size=3, stride=1, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
        bias_cls = bias_init_with_prob(0.01)
        torch.nn.init.constant_(self.cls_score.bias, bias_cls)

    def forward_single(self, feature):
        cls_score = self.cls_score(self.cls_subnet(feature))
        N, _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, self.num_classes, H, W)
        reg_feat = self.bbox_subnet(feature)
        bbox_reg = self.bbox_pred(reg_feat)
        objectness = self.object_pred(reg_feat)
        objectness = objectness.view(N, -1, 1, H, W)
        normalized_cls_score = cls_score + objectness - torch.log(1.0 + torch.clamp(cls_score.exp(), max=INF) + torch.clamp(objectness.exp(), max=INF))
        normalized_cls_score = normalized_cls_score.view(N, -1, H, W)
        return (normalized_cls_score, bbox_reg)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (batch, num_anchors * num_classes, h, w)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (batch, num_anchors * 4, h, w)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == 1
        assert self.prior_generator.num_levels == 1
        device = cls_scores[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        anchor_list = [anchors[0] for anchors in anchor_list]
        valid_flag_list = [valid_flags[0] for valid_flags in valid_flag_list]
        cls_scores_list = levels_to_images(cls_scores)
        bbox_preds_list = levels_to_images(bbox_preds)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        batch_labels, batch_label_weights, num_total_pos, num_total_neg, batch_bbox_weights, batch_pos_predicted_boxes, batch_target_boxes = cls_reg_targets
        flatten_labels = batch_labels.reshape(-1)
        batch_label_weights = batch_label_weights.reshape(-1)
        cls_score = cls_scores[0].permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos
        num_total_samples = reduce_mean(cls_score.new_tensor(num_total_samples)).clamp_(1.0).item()
        loss_cls = self.loss_cls(cls_score, flatten_labels, batch_label_weights, avg_factor=num_total_samples)
        if batch_pos_predicted_boxes.shape[0] == 0:
            loss_bbox = batch_pos_predicted_boxes.sum() * 0
        else:
            loss_bbox = self.loss_bbox(batch_pos_predicted_boxes, batch_target_boxes, batch_bbox_weights.float(), avg_factor=num_total_samples)
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    def get_targets(self, cls_scores_list, bbox_preds_list, anchor_list, valid_flag_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, unmap_outputs=True):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            cls_scores_list (list[Tensor])： Classification scores of
                each image. each is a 4D-tensor, the shape is
                (h * w, num_anchors * num_classes).
            bbox_preds_list (list[Tensor])： Bbox preds of each image.
                each is a 4D-tensor, the shape is (h * w, num_anchors * 4).
            anchor_list (list[Tensor]): Anchors of each image. Each element of
                is a tensor of shape (h * w * num_anchors, 4).
            valid_flag_list (list[Tensor]): Valid flags of each image. Each
               element of is a tensor of shape (h * w * num_anchors, )
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

                - batch_labels (Tensor): Label of all images. Each element                     of is a tensor of shape (batch, h * w * num_anchors)
                - batch_label_weights (Tensor): Label weights of all images                     of is a tensor of shape (batch, h * w * num_anchors)
                - num_total_pos (int): Number of positive samples in all                     images.
                - num_total_neg (int): Number of negative samples in all                     images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(self._get_targets_single, bbox_preds_list, anchor_list, valid_flag_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, label_channels=label_channels, unmap_outputs=unmap_outputs)
        all_labels, all_label_weights, pos_inds_list, neg_inds_list, sampling_results_list = results[:5]
        rest_results = list(results[5:])
        if any([labels is None for labels in all_labels]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        batch_labels = torch.stack(all_labels, 0)
        batch_label_weights = torch.stack(all_label_weights, 0)
        res = (batch_labels, batch_label_weights, num_total_pos, num_total_neg)
        for i, rests in enumerate(rest_results):
            rest_results[i] = torch.cat(rests, 0)
        return res + tuple(rest_results)

    def _get_targets_single(self, bbox_preds, flat_anchors, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, label_channels=1, unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            bbox_preds (Tensor): Bbox prediction of the image, which
                shape is (h * w ,4)
            flat_anchors (Tensor): Anchors of the image, which shape is
                (h * w * num_anchors ,4)
            valid_flags (Tensor): Valid flags of the image, which shape is
                (h * w * num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels (Tensor): Labels of image, which shape is
                    (h * w * num_anchors, ).
                label_weights (Tensor): Label weights of image, which shape is
                    (h * w * num_anchors, ).
                pos_inds (Tensor): Pos index of image.
                neg_inds (Tensor): Neg index of image.
                sampling_result (obj:`SamplingResult`): Sampling result.
                pos_bbox_weights (Tensor): The Weight of using to calculate
                    the bbox branch loss, which shape is (num, ).
                pos_predicted_boxes (Tensor): boxes predicted value of
                    using to calculate the bbox branch loss, which shape is
                    (num, 4).
                pos_target_boxes (Tensor): boxes target value of
                    using to calculate the bbox branch loss, which shape is
                    (num, 4).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 8
        anchors = flat_anchors[inside_flags, :]
        bbox_preds = bbox_preds.reshape(-1, 4)
        bbox_preds = bbox_preds[inside_flags, :]
        decoder_bbox_preds = self.bbox_coder.decode(anchors, bbox_preds)
        assign_result = self.assigner.assign(decoder_bbox_preds, anchors, gt_bboxes, gt_bboxes_ignore, None if self.sampling else gt_labels)
        pos_bbox_weights = assign_result.get_extra_property('pos_idx')
        pos_predicted_boxes = assign_result.get_extra_property('pos_predicted_boxes')
        pos_target_boxes = assign_result.get_extra_property('target_boxes')
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
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
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        return (labels, label_weights, pos_inds, neg_inds, sampling_result, pos_bbox_weights, pos_predicted_boxes, pos_target_boxes)

def levels_to_images(mlvl_tensor):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]

@HEADS.register_module()
class PISASSDHead(SSDHead):

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image
                with shape (num_obj, 4).
            gt_labels (list[Tensor]): Ground truth labels of each image
                with shape (num_obj, 4).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor]): Ignored gt bboxes of each image.
                Default: None.

        Returns:
            dict: Loss dict, comprise classification loss regression loss and
                carl loss.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=1, unmap_outputs=False, return_sampling_results=True)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg, sampling_results_list = cls_reg_targets
        num_images = len(img_metas)
        all_cls_scores = torch.cat([s.permute(0, 2, 3, 1).reshape(num_images, -1, self.cls_out_channels) for s in cls_scores], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list, -1).view(num_images, -1)
        all_bbox_preds = torch.cat([b.permute(0, 2, 3, 1).reshape(num_images, -1, 4) for b in bbox_preds], -2)
        all_bbox_targets = torch.cat(bbox_targets_list, -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list, -2).view(num_images, -1, 4)
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))
        isr_cfg = self.train_cfg.get('isr', None)
        all_targets = (all_labels.view(-1), all_label_weights.view(-1), all_bbox_targets.view(-1, 4), all_bbox_weights.view(-1, 4))
        if isr_cfg is not None:
            all_targets = isr_p(all_cls_scores.view(-1, all_cls_scores.size(-1)), all_bbox_preds.view(-1, 4), all_targets, torch.cat(all_anchors), sampling_results_list, loss_cls=CrossEntropyLoss(), bbox_coder=self.bbox_coder, **self.train_cfg.isr, num_class=self.num_classes)
            new_labels, new_label_weights, new_bbox_targets, new_bbox_weights = all_targets
            all_labels = new_labels.view(all_labels.shape)
            all_label_weights = new_label_weights.view(all_label_weights.shape)
            all_bbox_targets = new_bbox_targets.view(all_bbox_targets.shape)
            all_bbox_weights = new_bbox_weights.view(all_bbox_weights.shape)
        carl_loss_cfg = self.train_cfg.get('carl', None)
        if carl_loss_cfg is not None:
            loss_carl = carl_loss(all_cls_scores.view(-1, all_cls_scores.size(-1)), all_targets[0], all_bbox_preds.view(-1, 4), all_targets[2], SmoothL1Loss(beta=1.0), **self.train_cfg.carl, avg_factor=num_total_pos, num_class=self.num_classes)
        assert torch.isfinite(all_cls_scores).all().item(), 'classification scores become infinite or NaN!'
        assert torch.isfinite(all_bbox_preds).all().item(), 'bbox predications become infinite or NaN!'
        losses_cls, losses_bbox = multi_apply(self.loss_single, all_cls_scores, all_bbox_preds, all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, num_total_samples=num_total_pos)
        loss_dict = dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        if carl_loss_cfg is not None:
            loss_dict.update(loss_carl)
        return loss_dict

@mmcv.jit(derivate=True, coderize=True)
def carl_loss(cls_score, labels, bbox_pred, bbox_targets, loss_bbox, k=1, bias=0.2, avg_factor=None, sigmoid=False, num_class=80):
    """Classification-Aware Regression Loss (CARL).

    Args:
        cls_score (Tensor): Predicted classification scores.
        labels (Tensor): Targets of classification.
        bbox_pred (Tensor): Predicted bbox deltas.
        bbox_targets (Tensor): Target of bbox regression.
        loss_bbox (func): Regression loss func of the head.
        bbox_coder (obj): BBox coder of the head.
        k (float): Power of the non-linear mapping.
        bias (float): Shift of the non-linear mapping.
        avg_factor (int): Average factor used in regression loss.
        sigmoid (bool): Activation of the classification score.
        num_class (int): Number of classes, default: 80.

    Return:
        dict: CARL loss dict.
    """
    pos_label_inds = ((labels >= 0) & (labels < num_class)).nonzero().reshape(-1)
    if pos_label_inds.numel() == 0:
        return dict(loss_carl=cls_score.sum()[None] * 0.0)
    pos_labels = labels[pos_label_inds]
    if sigmoid:
        pos_cls_score = cls_score.sigmoid()[pos_label_inds, pos_labels]
    else:
        pos_cls_score = cls_score.softmax(-1)[pos_label_inds, pos_labels]
    carl_loss_weights = (bias + (1 - bias) * pos_cls_score).pow(k)
    num_pos = float(pos_cls_score.size(0))
    weight_ratio = num_pos / carl_loss_weights.sum()
    carl_loss_weights *= weight_ratio
    if avg_factor is None:
        avg_factor = bbox_targets.size(0)
    if bbox_pred.size(-1) > 4:
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)
        pos_bbox_preds = bbox_pred[pos_label_inds, pos_labels]
    else:
        pos_bbox_preds = bbox_pred[pos_label_inds]
    ori_loss_reg = loss_bbox(pos_bbox_preds, bbox_targets[pos_label_inds], reduction_override='none') / avg_factor
    loss_carl = (ori_loss_reg * carl_loss_weights[:, None]).sum()
    return dict(loss_carl=loss_carl[None])

@HEADS.register_module()
class LADHead(PAAHead):
    """Label Assignment Head from the paper: `Improving Object Detection by
    Label Assignment Distillation <https://arxiv.org/pdf/2108.10520.pdf>`_"""

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def get_label_assignment(self, cls_scores, bbox_preds, iou_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Get label assignment (from teacher).

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level.
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
            tuple: Returns a tuple containing label assignment variables.

                - labels (Tensor): Labels of all anchors, each with
                    shape (num_anchors,).
                - labels_weight (Tensor): Label weights of all anchor.
                    each with shape (num_anchors,).
                - bboxes_target (Tensor): BBox targets of all anchors.
                    each with shape (num_anchors, 4).
                - bboxes_weight (Tensor): BBox weights of all anchors.
                    each with shape (num_anchors, 4).
                - pos_inds_flatten (Tensor): Contains all index of positive
                    sample in all anchor.
                - pos_anchors (Tensor): Positive anchors.
                - num_pos (int): Number of positive anchors.
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
        pos_losses_list, = multi_apply(self.get_pos_loss, anchor_list, cls_scores, bbox_preds, labels, labels_weight, bboxes_target, bboxes_weight, pos_inds)
        with torch.no_grad():
            reassign_labels, reassign_label_weight, reassign_bbox_weights, num_pos = multi_apply(self.paa_reassign, pos_losses_list, labels, labels_weight, bboxes_weight, pos_inds, pos_gt_index, anchor_list)
            num_pos = sum(num_pos)
        labels = torch.cat(reassign_labels, 0).view(-1)
        flatten_anchors = torch.cat([torch.cat(item, 0) for item in anchor_list])
        labels_weight = torch.cat(reassign_label_weight, 0).view(-1)
        bboxes_target = torch.cat(bboxes_target, 0).view(-1, bboxes_target[0].size(-1))
        pos_inds_flatten = ((labels >= 0) & (labels < self.num_classes)).nonzero().reshape(-1)
        if num_pos:
            pos_anchors = flatten_anchors[pos_inds_flatten]
        else:
            pos_anchors = None
        label_assignment_results = (labels, labels_weight, bboxes_target, bboxes_weight, pos_inds_flatten, pos_anchors, num_pos)
        return label_assignment_results

    def forward_train(self, x, label_assignment_results, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=None, **kwargs):
        """Forward train with the available label assignment (student receives
        from teacher).

        Args:
            x (list[Tensor]): Features from FPN.
            label_assignment_results (tuple): As the outputs defined in the
                function `self.get_label_assignment`.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).

        Returns:
            losses: (dict[str, Tensor]): A dictionary of loss components.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, label_assignment_results=label_assignment_results)
        return losses

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def loss(self, cls_scores, bbox_preds, iou_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None, label_assignment_results=None):
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
            label_assignment_results (tuple): As the outputs defined in the
                function `self.get_label_assignment`.

        Returns:
            dict[str, Tensor]: A dictionary of loss gmm_assignment.
        """
        labels, labels_weight, bboxes_target, bboxes_weight, pos_inds_flatten, pos_anchors, num_pos = label_assignment_results
        cls_scores = levels_to_images(cls_scores)
        cls_scores = [item.reshape(-1, self.cls_out_channels) for item in cls_scores]
        bbox_preds = levels_to_images(bbox_preds)
        bbox_preds = [item.reshape(-1, 4) for item in bbox_preds]
        iou_preds = levels_to_images(iou_preds)
        iou_preds = [item.reshape(-1, 1) for item in iou_preds]
        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        bbox_preds = torch.cat(bbox_preds, 0).view(-1, bbox_preds[0].size(-1))
        iou_preds = torch.cat(iou_preds, 0).view(-1, iou_preds[0].size(-1))
        losses_cls = self.loss_cls(cls_scores, labels, labels_weight, avg_factor=max(num_pos, len(img_metas)))
        if num_pos:
            pos_bbox_pred = self.bbox_coder.decode(pos_anchors, bbox_preds[pos_inds_flatten])
            pos_bbox_target = bboxes_target[pos_inds_flatten]
            iou_target = bbox_overlaps(pos_bbox_pred.detach(), pos_bbox_target, is_aligned=True)
            losses_iou = self.loss_centerness(iou_preds[pos_inds_flatten], iou_target.unsqueeze(-1), avg_factor=num_pos)
            losses_bbox = self.loss_bbox(pos_bbox_pred, pos_bbox_target, avg_factor=num_pos)
        else:
            losses_iou = iou_preds.sum() * 0
            losses_bbox = bbox_preds.sum() * 0
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_iou=losses_iou)

@HEADS.register_module()
class DDODHead(AnchorHead):
    """DDOD head decomposes conjunctions lying in most current one-stage
    detectors via label assignment disentanglement, spatial feature
    disentanglement, and pyramid supervision disentanglement.

    https://arxiv.org/abs/2107.02963

    Args:
        num_classes (int): Number of categories excluding the
            background category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): The number of stacked Conv. Default: 4.
        conv_cfg (dict): Conv config of ddod head. Default: None.
        use_dcn (bool): Use dcn, Same as ATSS when False. Default: True.
        norm_cfg (dict): Normal config of ddod head. Default:
            dict(type='GN', num_groups=32, requires_grad=True).
        loss_iou (dict): Config of IoU loss. Default:
            dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0).
    """

    def __init__(self, num_classes, in_channels, stacked_convs=4, conv_cfg=None, use_dcn=True, norm_cfg=dict(type='GN', num_groups=32, requires_grad=True), loss_iou=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_dcn = use_dcn
        super(DDODHead, self).__init__(num_classes, in_channels, **kwargs)
        self.sampling = False
        if self.train_cfg:
            self.cls_assigner = build_assigner(self.train_cfg.assigner)
            self.reg_assigner = build_assigner(self.train_cfg.reg_assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.loss_iou = build_loss(loss_iou)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=dict(type='DCN', deform_groups=1) if i == 0 and self.use_dcn else self.conv_cfg, norm_cfg=self.norm_cfg))
            self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=dict(type='DCN', deform_groups=1) if i == 0 and self.use_dcn else self.conv_cfg, norm_cfg=self.norm_cfg))
        self.atss_cls = nn.Conv2d(self.feat_channels, self.num_base_priors * self.cls_out_channels, 3, padding=1)
        self.atss_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4, 3, padding=1)
        self.atss_iou = nn.Conv2d(self.feat_channels, self.num_base_priors * 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.prior_generator.strides])
        self.cls_num_pos_samples_per_level = [0.0 for _ in range(len(self.prior_generator.strides))]
        self.reg_num_pos_samples_per_level = [0.0 for _ in range(len(self.prior_generator.strides))]

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        normal_init(self.atss_reg, std=0.01)
        normal_init(self.atss_iou, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.atss_cls, std=0.01, bias=bias_cls)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_base_priors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_base_priors * 4.
                iou_preds (list[Tensor]): IoU scores for all scale levels,
                    each is a 4D-tensor, the channels number is
                    num_base_priors * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                - cls_score (Tensor): Cls scores for a single scale level                     the channels number is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for a single                     scale level, the channels number is num_base_priors * 4.
                - iou_pred (Tensor): Iou for a single scale level, the                     channel number is (N, num_base_priors * 1, H, W).
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        iou_pred = self.atss_iou(reg_feat)
        return (cls_score, bbox_pred, iou_pred)

    def loss_cls_single(self, cls_score, labels, label_weights, reweight_factor, num_total_samples):
        """Compute cls loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_base_priors * num_classes, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            reweight_factor (list[int]): Reweight factor for cls and reg
                loss.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            tuple[Tensor]: A tuple of loss components.
        """
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous()
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        return (reweight_factor * loss_cls,)

    def loss_reg_single(self, anchors, bbox_pred, iou_pred, labels, label_weights, bbox_targets, bbox_weights, reweight_factor, num_total_samples):
        """Compute reg loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_base_priors * 4, H, W).
            iou_pred (Tensor): Iou for a single scale level, the
                channel number is (N, num_base_priors * 1, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox weights of all anchors in the
                image with shape (N, 4)
            reweight_factor (list[int]): Reweight factor for cls and reg
                loss.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        anchors = anchors.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        iou_pred = iou_pred.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        iou_targets = label_weights.new_zeros(labels.shape)
        iou_weights = label_weights.new_zeros(labels.shape)
        iou_weights[(bbox_weights.sum(axis=1) > 0).nonzero(as_tuple=False)] = 1.0
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_decode_bbox_pred = self.bbox_coder.decode(pos_anchors, pos_bbox_pred)
            pos_decode_bbox_targets = self.bbox_coder.decode(pos_anchors, pos_bbox_targets)
            loss_bbox = self.loss_bbox(pos_decode_bbox_pred, pos_decode_bbox_targets, avg_factor=num_total_samples)
            iou_targets[pos_inds] = bbox_overlaps(pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True)
            loss_iou = self.loss_iou(iou_pred, iou_targets, iou_weights, avg_factor=num_total_samples)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_iou = iou_pred.sum() * 0
        return (reweight_factor * loss_bbox, reweight_factor * loss_iou)

    def calc_reweight_factor(self, labels_list):
        """Compute reweight_factor for regression and classification loss."""
        bg_class_ind = self.num_classes
        for ii, each_level_label in enumerate(labels_list):
            pos_inds = ((each_level_label >= 0) & (each_level_label < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)
            self.cls_num_pos_samples_per_level[ii] += len(pos_inds)
        min_pos_samples = min(self.cls_num_pos_samples_per_level)
        max_pos_samples = max(self.cls_num_pos_samples_per_level)
        interval = 1.0 / (max_pos_samples - min_pos_samples + 1e-10)
        reweight_factor_per_level = []
        for pos_samples in self.cls_num_pos_samples_per_level:
            factor = 2.0 - (pos_samples - min_pos_samples) * interval
            reweight_factor_per_level.append(factor)
        return reweight_factor_per_level

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def loss(self, cls_scores, bbox_preds, iou_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_base_priors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_base_priors * 4, H, W)
            iou_preds (list[Tensor]): Score factor for all scale level,
                each is a 4D-tensor, has shape (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        targets_com = self.process_predictions_and_anchors(anchor_list, valid_flag_list, cls_scores, bbox_preds, img_metas, gt_bboxes_ignore)
        anchor_list, valid_flag_list, num_level_anchors_list, cls_score_list, bbox_pred_list, gt_bboxes_ignore_list = targets_com
        cls_targets = self.get_cls_targets(anchor_list, valid_flag_list, num_level_anchors_list, cls_score_list, bbox_pred_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore_list, gt_labels_list=gt_labels, label_channels=label_channels)
        if cls_targets is None:
            return None
        cls_anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_targets
        num_total_samples = reduce_mean(torch.tensor(num_total_pos, dtype=torch.float, device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)
        reweight_factor_per_level = self.calc_reweight_factor(labels_list)
        cls_losses_cls, = multi_apply(self.loss_cls_single, cls_scores, labels_list, label_weights_list, reweight_factor_per_level, num_total_samples=num_total_samples)
        reg_targets = self.get_reg_targets(anchor_list, valid_flag_list, num_level_anchors_list, cls_score_list, bbox_pred_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore_list, gt_labels_list=gt_labels, label_channels=label_channels)
        if reg_targets is None:
            return None
        reg_anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = reg_targets
        num_total_samples = reduce_mean(torch.tensor(num_total_pos, dtype=torch.float, device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)
        reweight_factor_per_level = self.calc_reweight_factor(labels_list)
        reg_losses_bbox, reg_losses_iou = multi_apply(self.loss_reg_single, reg_anchor_list, bbox_preds, iou_preds, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, reweight_factor_per_level, num_total_samples=num_total_samples)
        return dict(loss_cls=cls_losses_cls, loss_bbox=reg_losses_bbox, loss_iou=reg_losses_iou)

    def process_predictions_and_anchors(self, anchor_list, valid_flag_list, cls_scores, bbox_preds, img_metas, gt_bboxes_ignore_list):
        """Compute common vars for regression and classification targets.

        Args:
            anchor_list (list[Tensor]): anchors of each image.
            valid_flag_list (list[Tensor]): Valid flags of each image.
            cls_scores (list[Tensor]): Classification scores for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * 4.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore_list (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Return:
            tuple[Tensor]: A tuple of common loss vars.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        anchor_list_ = []
        valid_flag_list_ = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list_.append(torch.cat(anchor_list[i]))
            valid_flag_list_.append(torch.cat(valid_flag_list[i]))
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        num_levels = len(cls_scores)
        cls_score_list = []
        bbox_pred_list = []
        mlvl_cls_score_list = [cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_base_priors * self.cls_out_channels) for cls_score in cls_scores]
        mlvl_bbox_pred_list = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_base_priors * 4) for bbox_pred in bbox_preds]
        for i in range(num_imgs):
            mlvl_cls_tensor_list = [mlvl_cls_score_list[j][i] for j in range(num_levels)]
            mlvl_bbox_tensor_list = [mlvl_bbox_pred_list[j][i] for j in range(num_levels)]
            cat_mlvl_cls_score = torch.cat(mlvl_cls_tensor_list, dim=0)
            cat_mlvl_bbox_pred = torch.cat(mlvl_bbox_tensor_list, dim=0)
            cls_score_list.append(cat_mlvl_cls_score)
            bbox_pred_list.append(cat_mlvl_bbox_pred)
        return (anchor_list_, valid_flag_list_, num_level_anchors_list, cls_score_list, bbox_pred_list, gt_bboxes_ignore_list)

    def get_cls_targets(self, anchor_list, valid_flag_list, num_level_anchors_list, cls_score_list, bbox_pred_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, unmap_outputs=True):
        """Get cls targets for DDOD head.

        This method is almost the same as `AnchorHead.get_targets()`.
        Besides returning the targets as the parent  method does,
        it also returns the anchors as the first element of the
        returned tuple.

        Args:
            anchor_list (list[Tensor]): anchors of each image.
            valid_flag_list (list[Tensor]): Valid flags of each image.
            num_level_anchors_list (list[Tensor]): Number of anchors of each
                scale level of all image.
            cls_score_list (list[Tensor]): Classification scores for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * num_classes.
            bbox_pred_list (list[Tensor]): Box energies / deltas for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * 4.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore_list (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.
            gt_labels_list (list[Tensor]): class indices corresponding to
                each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Return:
            tuple[Tensor]: A tuple of cls targets components.
        """
        all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, pos_inds_list, neg_inds_list = multi_apply(self._get_target_single, anchor_list, valid_flag_list, cls_score_list, bbox_pred_list, num_level_anchors_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, label_channels=label_channels, unmap_outputs=unmap_outputs, is_cls_assigner=True)
        if any([labels is None for labels in all_labels]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        anchors_list = images_to_levels(all_anchors, num_level_anchors_list[0])
        labels_list = images_to_levels(all_labels, num_level_anchors_list[0])
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors_list[0])
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors_list[0])
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors_list[0])
        return (anchors_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)

    def get_reg_targets(self, anchor_list, valid_flag_list, num_level_anchors_list, cls_score_list, bbox_pred_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, unmap_outputs=True):
        """Get reg targets for DDOD head.

        This method is almost the same as `AnchorHead.get_targets()` when
        is_cls_assigner is False. Besides returning the targets as the parent
        method does, it also returns the anchors as the first element of the
        returned tuple.

        Args:
            anchor_list (list[Tensor]): anchors of each image.
            valid_flag_list (list[Tensor]): Valid flags of each image.
            num_level_anchors (int): Number of anchors of each scale level.
            cls_scores (list[Tensor]): Classification scores for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * 4.
            gt_labels_list (list[Tensor]): class indices corresponding to
                each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore_list (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Return:
            tuple[Tensor]: A tuple of reg targets components.
        """
        all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, pos_inds_list, neg_inds_list = multi_apply(self._get_target_single, anchor_list, valid_flag_list, cls_score_list, bbox_pred_list, num_level_anchors_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, label_channels=label_channels, unmap_outputs=unmap_outputs, is_cls_assigner=False)
        if any([labels is None for labels in all_labels]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        anchors_list = images_to_levels(all_anchors, num_level_anchors_list[0])
        labels_list = images_to_levels(all_labels, num_level_anchors_list[0])
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors_list[0])
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors_list[0])
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors_list[0])
        return (anchors_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, flat_anchors, valid_flags, cls_scores, bbox_preds, num_level_anchors, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, label_channels=1, unmap_outputs=True, is_cls_assigner=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image,
                which are concatenated into a single tensor of shape
                (num_base_priors, 4).
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                shape (num_base_priors,).
            cls_scores (Tensor): Classification scores for all scale
                levels of the image.
            bbox_preds (Tensor): Box energies / deltas for all scale
                levels of the image.
            num_level_anchors (list[int]): Number of anchors of each
                scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, ).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts, ).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label. Default: 1.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Default: True.
            is_cls_assigner (bool): Classification or regression.
                Default: True.

        Returns:
            tuple: N is the number of total anchors in the image.
                - labels (Tensor): Labels of all anchors in the image with                     shape (N, ).
                - label_weights (Tensor): Label weights of all anchor in the                     image with shape (N, ).
                - bbox_targets (Tensor): BBox targets of all anchors in the                     image with shape (N, 4).
                - bbox_weights (Tensor): BBox weights of all anchors in the                     image with shape (N, 4)
                - pos_inds (Tensor): Indices of positive anchor with shape                     (num_pos, ).
                - neg_inds (Tensor): Indices of negative anchor with shape                     (num_neg, ).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7
        anchors = flat_anchors[inside_flags, :]
        num_level_anchors_inside = self.get_num_level_anchors_inside(num_level_anchors, inside_flags)
        bbox_preds_valid = bbox_preds[inside_flags, :]
        cls_scores_valid = cls_scores[inside_flags, :]
        assigner = self.cls_assigner if is_cls_assigner else self.reg_assigner
        bbox_preds_valid = self.bbox_coder.decode(anchors, bbox_preds_valid)
        assign_result = assigner.assign(anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore, gt_labels, cls_scores_valid, bbox_preds_valid)
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if hasattr(self, 'bbox_coder'):
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
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
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        return (anchors, labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        """Get the anchors of each scale level inside.

        Args:
            num_level_anchors (list[int]): Number of anchors of each
                scale level.
            inside_flags (Tensor): Multi level inside flags of the image,
                which are concatenated into a single tensor of
                shape (num_base_priors,).

        Returns:
            list[int]: Number of anchors of each scale level inside.
        """
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [int(flags.sum()) for flags in split_inside_flags]
        return num_level_anchors_inside

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
class GFLHead(AnchorHead):
    """Generalized Focal Loss: Learning Qualified and Distributed Bounding
    Boxes for Dense Object Detection.

    GFL head structure is similar with ATSS, however GFL uses
    1) joint representation for classification and localization quality, and
    2) flexible General distribution for bounding box locations,
    which are supervised by
    Quality Focal Loss (QFL) and Distribution Focal Loss (DFL), respectively

    https://arxiv.org/abs/2006.04388

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 4.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='GN', num_groups=32, requires_grad=True).
        loss_qfl (dict): Config of Quality Focal Loss (QFL).
        bbox_coder (dict): Config of bbox coder. Defaults
            'DistancePointBBoxCoder'.
        reg_max (int): Max value of integral set :math: `{0, ..., reg_max}`
            in QFL setting. Default: 16.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    Example:
        >>> self = GFLHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_quality_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_quality_score) == len(self.scales)
    """

    def __init__(self, num_classes, in_channels, stacked_convs=4, conv_cfg=None, norm_cfg=dict(type='GN', num_groups=32, requires_grad=True), loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25), bbox_coder=dict(type='DistancePointBBoxCoder'), reg_max=16, init_cfg=dict(type='Normal', layer='Conv2d', std=0.01, override=dict(type='Normal', name='gfl_cls', std=0.01, bias_prob=0.01)), **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        super(GFLHead, self).__init__(num_classes, in_channels, bbox_coder=bbox_coder, init_cfg=init_cfg, **kwargs)
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.integral = Integral(self.reg_max)
        self.loss_dfl = build_loss(loss_dfl)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        assert self.num_anchors == 1, 'anchor free version'
        self.gfl_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.gfl_reg = nn.Conv2d(self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.prior_generator.strides])

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification and quality (IoU)
                    joint scores for all scale levels, each is a 4D-tensor,
                    the channel number is num_classes.
                bbox_preds (list[Tensor]): Box distribution logits for all
                    scale levels, each is a 4D-tensor, the channel number is
                    4*(n+1), n is max value of integral set.
        """
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls and quality joint scores for a single
                    scale level the channel number is num_classes.
                bbox_pred (Tensor): Box distribution logits for a single scale
                    level, the channel number is 4*(n+1), n is max value of
                    integral set.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.gfl_cls(cls_feat)
        bbox_pred = scale(self.gfl_reg(reg_feat)).float()
        return (cls_score, bbox_pred)

    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[..., 2] + anchors[..., 0]) / 2
        anchors_cy = (anchors[..., 3] + anchors[..., 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def loss_single(self, anchors, cls_score, bbox_pred, labels, label_weights, bbox_targets, stride, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]
            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = self.bbox_coder.decode(pos_anchor_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = self.bbox_coder.encode(pos_anchor_centers, pos_decode_bbox_targets, self.reg_max).reshape(-1)
            loss_bbox = self.loss_bbox(pos_decode_bbox_pred, pos_decode_bbox_targets, weight=weight_targets, avg_factor=1.0)
            loss_dfl = self.loss_dfl(pred_corners, target_corners, weight=weight_targets[:, None].expand(-1, 4).reshape(-1), avg_factor=4.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)
        loss_cls = self.loss_cls(cls_score, (labels, score), weight=label_weights, avg_factor=num_total_samples)
        return (loss_cls, loss_bbox, loss_dfl, weight_targets.sum())

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        num_total_samples = reduce_mean(torch.tensor(num_total_pos, dtype=torch.float, device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)
        losses_cls, losses_bbox, losses_dfl, avg_factor = multi_apply(self.loss_single, anchor_list, cls_scores, bbox_preds, labels_list, label_weights_list, bbox_targets_list, self.prior_generator.strides, num_total_samples=num_total_samples)
        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

    def _get_bboxes_single(self, cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors, img_meta, cfg, rescale=False, with_nms=True, **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. GFL head does not need this value.
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
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, stride, priors) in enumerate(zip(cls_score_list, bbox_pred_list, self.prior_generator.strides, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert stride[0] == stride[1]
            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.integral(bbox_pred) * stride[0]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            results = filter_scores_and_topk(scores, cfg.score_thr, nms_pre, dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, _, filtered_results = results
            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            bboxes = self.bbox_coder.decode(self.anchor_center(priors), bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes, img_meta['scale_factor'], cfg, rescale=rescale, with_nms=with_nms)

    def get_targets(self, anchor_list, valid_flag_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, unmap_outputs=True):
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, pos_inds_list, neg_inds_list = multi_apply(self._get_target_single, anchor_list, valid_flag_list, num_level_anchors_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, label_channels=label_channels, unmap_outputs=unmap_outputs)
        if any([labels is None for labels in all_labels]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        return (anchors_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, flat_anchors, valid_flags, num_level_anchors, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, label_channels=1, unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4).
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7
        anchors = flat_anchors[inside_flags, :]
        num_level_anchors_inside = self.get_num_level_anchors_inside(num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore, gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
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
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        return (anchors, labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [int(flags.sum()) for flags in split_inside_flags]
        return num_level_anchors_inside

@HEADS.register_module()
class PISARetinaHead(RetinaHead):
    """PISA Retinanet Head.

    The head owns the same structure with Retinanet Head, but differs in two
        aspects:
        1. Importance-based Sample Reweighting Positive (ISR-P) is applied to
            change the positive loss weights.
        2. Classification-aware regression loss is adopted as a third loss.
    """

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image
                with shape (num_obj, 4).
            gt_labels (list[Tensor]): Ground truth labels of each image
                with shape (num_obj, 4).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor]): Ignored gt bboxes of each image.
                Default: None.

        Returns:
            dict: Loss dict, comprise classification loss, regression loss and
                carl loss.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels, return_sampling_results=True)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg, sampling_results_list = cls_reg_targets
        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)
        num_imgs = len(img_metas)
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, label_channels) for cls_score in cls_scores]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).reshape(-1, flatten_cls_scores[0].size(-1))
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1).view(-1, flatten_bbox_preds[0].size(-1))
        flatten_labels = torch.cat(labels_list, dim=1).reshape(-1)
        flatten_label_weights = torch.cat(label_weights_list, dim=1).reshape(-1)
        flatten_anchors = torch.cat(all_anchor_list, dim=1).reshape(-1, 4)
        flatten_bbox_targets = torch.cat(bbox_targets_list, dim=1).reshape(-1, 4)
        flatten_bbox_weights = torch.cat(bbox_weights_list, dim=1).reshape(-1, 4)
        isr_cfg = self.train_cfg.get('isr', None)
        if isr_cfg is not None:
            all_targets = (flatten_labels, flatten_label_weights, flatten_bbox_targets, flatten_bbox_weights)
            with torch.no_grad():
                all_targets = isr_p(flatten_cls_scores, flatten_bbox_preds, all_targets, flatten_anchors, sampling_results_list, bbox_coder=self.bbox_coder, loss_cls=self.loss_cls, num_class=self.num_classes, **self.train_cfg.isr)
            flatten_labels, flatten_label_weights, flatten_bbox_targets, flatten_bbox_weights = all_targets
        losses_cls = self.loss_cls(flatten_cls_scores, flatten_labels, flatten_label_weights, avg_factor=num_total_samples)
        losses_bbox = self.loss_bbox(flatten_bbox_preds, flatten_bbox_targets, flatten_bbox_weights, avg_factor=num_total_samples)
        loss_dict = dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        carl_cfg = self.train_cfg.get('carl', None)
        if carl_cfg is not None:
            loss_carl = carl_loss(flatten_cls_scores, flatten_labels, flatten_bbox_preds, flatten_bbox_targets, self.loss_bbox, **self.train_cfg.carl, avg_factor=num_total_pos, sigmoid=True, num_class=self.num_classes)
            loss_dict.update(loss_carl)
        return loss_dict

@HEADS.register_module()
class PISARoIHead(StandardRoIHead):
    """The RoI head for `Prime Sample Attention in Object Detection
    <https://arxiv.org/abs/1904.04821>`_."""

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None):
        """Forward function for training.

        Args:
            x (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): List of region proposals.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (list[Tensor], optional): Specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : True segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            neg_label_weights = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
                sampling_result = self.bbox_sampler.sample(assign_result, proposal_list[i], gt_bboxes[i], gt_labels[i], feats=[lvl_feat[i][None] for lvl_feat in x])
                neg_label_weight = None
                if isinstance(sampling_result, tuple):
                    sampling_result, neg_label_weight = sampling_result
                sampling_results.append(sampling_result)
                neg_label_weights.append(neg_label_weight)
        losses = dict()
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results, gt_bboxes, gt_labels, img_metas, neg_label_weights=neg_label_weights)
            losses.update(bbox_results['loss_bbox'])
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results, bbox_results['bbox_feats'], gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])
        return losses

    def _bbox_forward(self, x, rois):
        """Box forward function used in both training and testing."""
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas, neg_label_weights=None):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg)
        if neg_label_weights[0] is not None:
            label_weights = bbox_targets[1]
            cur_num_rois = 0
            for i in range(len(sampling_results)):
                num_pos = sampling_results[i].pos_inds.size(0)
                num_neg = sampling_results[i].neg_inds.size(0)
                label_weights[cur_num_rois + num_pos:cur_num_rois + num_pos + num_neg] = neg_label_weights[i]
                cur_num_rois += num_pos + num_neg
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        isr_cfg = self.train_cfg.get('isr', None)
        if isr_cfg is not None:
            bbox_targets = isr_p(cls_score, bbox_pred, bbox_targets, rois, sampling_results, self.bbox_head.loss_cls, self.bbox_head.bbox_coder, **isr_cfg, num_class=self.bbox_head.num_classes)
        loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, rois, *bbox_targets)
        carl_cfg = self.train_cfg.get('carl', None)
        if carl_cfg is not None:
            loss_carl = carl_loss(cls_score, bbox_targets[0], bbox_pred, bbox_targets[2], self.bbox_head.loss_bbox, **carl_cfg, num_class=self.bbox_head.num_classes)
            loss_bbox.update(loss_carl)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

def analyze_per_img_dets(confusion_matrix, gt_bboxes, gt_labels, result, score_thr=0, tp_iou_thr=0.5, nms_iou_thr=None):
    """Analyze detection results on each image.

    Args:
        confusion_matrix (ndarray): The confusion matrix,
            has shape (num_classes + 1, num_classes + 1).
        gt_bboxes (ndarray): Ground truth bboxes, has shape (num_gt, 4).
        gt_labels (ndarray): Ground truth labels, has shape (num_gt).
        result (ndarray): Detection results, has shape
            (num_classes, num_bboxes, 5).
        score_thr (float): Score threshold to filter bboxes.
            Default: 0.
        tp_iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
    """
    true_positives = np.zeros_like(gt_labels)
    for det_label, det_bboxes in enumerate(result):
        if nms_iou_thr:
            det_bboxes, _ = nms(det_bboxes[:, :4], det_bboxes[:, -1], nms_iou_thr, score_threshold=score_thr)
        ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
        for i, det_bbox in enumerate(det_bboxes):
            score = det_bbox[4]
            det_match = 0
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr:
                        det_match += 1
                        if gt_label == det_label:
                            true_positives[j] += 1
                        confusion_matrix[gt_label, det_label] += 1
                if det_match == 0:
                    confusion_matrix[-1, det_label] += 1
    for num_tp, gt_label in zip(true_positives, gt_labels):
        if num_tp == 0:
            confusion_matrix[gt_label, -1] += 1

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

def test_corner_head_encode_and_decode_heatmap():
    """Tests corner head generating and decoding the heatmap."""
    s = 256
    img_metas = [{'img_shape': (s, s, 3), 'scale_factor': 1, 'pad_shape': (s, s, 3), 'border': (0, 0, 0, 0)}]
    gt_bboxes = [torch.Tensor([[10, 20, 200, 240], [40, 50, 100, 200], [10, 20, 200, 240]])]
    gt_labels = [torch.LongTensor([1, 1, 2])]
    self = CornerHead(num_classes=4, in_channels=1, corner_emb_channels=1)
    feat = [torch.rand(1, 1, s // 4, s // 4) for _ in range(self.num_feat_levels)]
    targets = self.get_targets(gt_bboxes, gt_labels, feat[0].shape, img_metas[0]['pad_shape'], with_corner_emb=self.with_corner_emb)
    gt_tl_heatmap = targets['topleft_heatmap']
    gt_br_heatmap = targets['bottomright_heatmap']
    gt_tl_offset = targets['topleft_offset']
    gt_br_offset = targets['bottomright_offset']
    embedding = targets['corner_embedding']
    [top, left], [bottom, right] = embedding[0][0]
    gt_tl_embedding_heatmap = torch.zeros([1, 1, s // 4, s // 4])
    gt_br_embedding_heatmap = torch.zeros([1, 1, s // 4, s // 4])
    gt_tl_embedding_heatmap[0, 0, top, left] = 1
    gt_br_embedding_heatmap[0, 0, bottom, right] = 1
    batch_bboxes, batch_scores, batch_clses = self.decode_heatmap(tl_heat=gt_tl_heatmap, br_heat=gt_br_heatmap, tl_off=gt_tl_offset, br_off=gt_br_offset, tl_emb=gt_tl_embedding_heatmap, br_emb=gt_br_embedding_heatmap, img_meta=img_metas[0], k=100, kernel=3, distance_threshold=0.5)
    bboxes = batch_bboxes.view(-1, 4)
    scores = batch_scores.view(-1, 1)
    clses = batch_clses.view(-1, 1)
    idx = scores.argsort(dim=0, descending=True)
    bboxes = bboxes[idx].view(-1, 4)
    scores = scores[idx].view(-1)
    clses = clses[idx].view(-1)
    valid_bboxes = bboxes[torch.where(scores > 0.05)]
    valid_labels = clses[torch.where(scores > 0.05)]
    max_coordinate = valid_bboxes.max()
    offsets = valid_labels.to(valid_bboxes) * (max_coordinate + 1)
    gt_offsets = gt_labels[0].to(gt_bboxes[0]) * (max_coordinate + 1)
    offset_bboxes = valid_bboxes + offsets[:, None]
    offset_gtbboxes = gt_bboxes[0] + gt_offsets[:, None]
    iou_matrix = bbox_overlaps(offset_bboxes.numpy(), offset_gtbboxes.numpy())
    assert (iou_matrix == 1).sum() == 3

def test_lad_head_loss():
    """Tests lad head loss when truth is empty and non-empty."""

    class mock_skm:

        def GaussianMixture(self, *args, **kwargs):
            return self

        def fit(self, loss):
            pass

        def predict(self, loss):
            components = np.zeros_like(loss, dtype=np.long)
            return components.reshape(-1)

        def score_samples(self, loss):
            scores = np.random.random(len(loss))
            return scores
    lad_head.skm = mock_skm()
    s = 256
    img_metas = [{'img_shape': (s, s, 3), 'scale_factor': 1, 'pad_shape': (s, s, 3)}]
    train_cfg = mmcv.Config(dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.1, neg_iou_thr=0.1, min_pos_iou=0, ignore_iof_thr=-1), allowed_border=-1, pos_weight=-1, debug=False))
    self = LADHead(num_classes=4, in_channels=1, train_cfg=train_cfg, loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_bbox=dict(type='GIoULoss', loss_weight=1.3), loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5))
    teacher_model = LADHead(num_classes=4, in_channels=1, train_cfg=train_cfg, loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_bbox=dict(type='GIoULoss', loss_weight=1.3), loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5))
    feat = [torch.rand(1, 1, s // feat_size, s // feat_size) for feat_size in [4, 8, 16, 32, 64]]
    self.init_weights()
    teacher_model.init_weights()
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_bboxes_ignore = None
    outs_teacher = teacher_model(feat)
    label_assignment_results = teacher_model.get_label_assignment(*outs_teacher, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
    outs = teacher_model(feat)
    empty_gt_losses = self.loss(*outs, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore, label_assignment_results)
    empty_cls_loss = empty_gt_losses['loss_cls']
    empty_box_loss = empty_gt_losses['loss_bbox']
    empty_iou_loss = empty_gt_losses['loss_iou']
    assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_box_loss.item() == 0, 'there should be no box loss when there are no true boxes'
    assert empty_iou_loss.item() == 0, 'there should be no box loss when there are no true boxes'
    gt_bboxes = [torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]])]
    gt_labels = [torch.LongTensor([2])]
    label_assignment_results = teacher_model.get_label_assignment(*outs_teacher, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
    one_gt_losses = self.loss(*outs, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore, label_assignment_results)
    onegt_cls_loss = one_gt_losses['loss_cls']
    onegt_box_loss = one_gt_losses['loss_bbox']
    onegt_iou_loss = one_gt_losses['loss_iou']
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'
    assert onegt_iou_loss.item() > 0, 'box loss should be non-zero'
    n, c, h, w = (10, 4, 20, 20)
    mlvl_tensor = [torch.ones(n, c, h, w) for i in range(5)]
    results = levels_to_images(mlvl_tensor)
    assert len(results) == n
    assert results[0].size() == (h * w * 5, c)
    assert self.with_score_voting
    self = LADHead(num_classes=4, in_channels=1, train_cfg=train_cfg, anchor_generator=dict(type='AnchorGenerator', ratios=[1.0], octave_base_scale=8, scales_per_octave=1, strides=[8]), loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_bbox=dict(type='GIoULoss', loss_weight=1.3), loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5))
    cls_scores = [torch.ones(2, 4, 5, 5)]
    bbox_preds = [torch.ones(2, 4, 5, 5)]
    iou_preds = [torch.ones(2, 1, 5, 5)]
    cfg = mmcv.Config(dict(nms_pre=1000, min_bbox_size=0, score_thr=0.05, nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))
    rescale = False
    self.get_bboxes(cls_scores, bbox_preds, iou_preds, img_metas, cfg, rescale=rescale)

def test_autoassign_head_loss():
    """Tests autoassign head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{'img_shape': (s, s, 3), 'scale_factor': 1, 'pad_shape': (s, s, 3)}]
    train_cfg = mmcv.Config(dict(assigner=None, allowed_border=-1, pos_weight=-1, debug=False))
    self = AutoAssignHead(num_classes=4, in_channels=1, train_cfg=train_cfg, loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_bbox=dict(type='GIoULoss', loss_weight=1.3))
    feat = [torch.rand(1, 1, s // feat_size, s // feat_size) for feat_size in [4, 8, 16, 32, 64]]
    self.init_weights()
    cls_scores, bbox_preds, objectnesses = self(feat)
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, bbox_preds, objectnesses, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
    empty_pos_loss = empty_gt_losses['loss_pos']
    empty_neg_loss = empty_gt_losses['loss_neg']
    empty_center_loss = empty_gt_losses['loss_center']
    assert empty_neg_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_pos_loss.item() == 0, 'there should be no box loss when there are no true boxes'
    assert empty_center_loss.item() == 0, 'there should be no box loss when there are no true boxes'
    gt_bboxes = [torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]])]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, objectnesses, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
    onegt_pos_loss = one_gt_losses['loss_pos']
    onegt_neg_loss = one_gt_losses['loss_neg']
    onegt_center_loss = one_gt_losses['loss_center']
    assert onegt_pos_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_neg_loss.item() > 0, 'box loss should be non-zero'
    assert onegt_center_loss.item() > 0, 'box loss should be non-zero'
    n, c, h, w = (10, 4, 20, 20)
    mlvl_tensor = [torch.ones(n, c, h, w) for i in range(5)]
    results = levels_to_images(mlvl_tensor)
    assert len(results) == n
    assert results[0].size() == (h * w * 5, c)
    self = AutoAssignHead(num_classes=4, in_channels=1, train_cfg=train_cfg, loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_bbox=dict(type='GIoULoss', loss_weight=1.3), strides=(4,))
    cls_scores = [torch.ones(2, 4, 5, 5)]
    bbox_preds = [torch.ones(2, 4, 5, 5)]
    iou_preds = [torch.ones(2, 1, 5, 5)]
    cfg = mmcv.Config(dict(nms_pre=1000, min_bbox_size=0, score_thr=0.05, nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))
    rescale = False
    self.get_bboxes(cls_scores, bbox_preds, iou_preds, img_metas, cfg, rescale=rescale)

def test_paa_head_loss():
    """Tests paa head loss when truth is empty and non-empty."""

    class mock_skm:

        def GaussianMixture(self, *args, **kwargs):
            return self

        def fit(self, loss):
            pass

        def predict(self, loss):
            components = np.zeros_like(loss, dtype=np.long)
            return components.reshape(-1)

        def score_samples(self, loss):
            scores = np.random.random(len(loss))
            return scores
    paa_head.skm = mock_skm()
    s = 256
    img_metas = [{'img_shape': (s, s, 3), 'scale_factor': 1, 'pad_shape': (s, s, 3)}]
    train_cfg = mmcv.Config(dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.1, neg_iou_thr=0.1, min_pos_iou=0, ignore_iof_thr=-1), allowed_border=-1, pos_weight=-1, debug=False))
    self = PAAHead(num_classes=4, in_channels=1, train_cfg=train_cfg, anchor_generator=dict(type='AnchorGenerator', ratios=[1.0], octave_base_scale=8, scales_per_octave=1, strides=[8, 16, 32, 64, 128]), loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_bbox=dict(type='GIoULoss', loss_weight=1.3), loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5))
    feat = [torch.rand(1, 1, s // feat_size, s // feat_size) for feat_size in [4, 8, 16, 32, 64]]
    self.init_weights()
    cls_scores, bbox_preds, iou_preds = self(feat)
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, bbox_preds, iou_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
    empty_cls_loss = empty_gt_losses['loss_cls']
    empty_box_loss = empty_gt_losses['loss_bbox']
    empty_iou_loss = empty_gt_losses['loss_iou']
    assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_box_loss.item() == 0, 'there should be no box loss when there are no true boxes'
    assert empty_iou_loss.item() == 0, 'there should be no box loss when there are no true boxes'
    gt_bboxes = [torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]])]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, iou_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
    onegt_cls_loss = one_gt_losses['loss_cls']
    onegt_box_loss = one_gt_losses['loss_bbox']
    onegt_iou_loss = one_gt_losses['loss_iou']
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'
    assert onegt_iou_loss.item() > 0, 'box loss should be non-zero'
    n, c, h, w = (10, 4, 20, 20)
    mlvl_tensor = [torch.ones(n, c, h, w) for i in range(5)]
    results = levels_to_images(mlvl_tensor)
    assert len(results) == n
    assert results[0].size() == (h * w * 5, c)
    assert self.with_score_voting
    self = PAAHead(num_classes=4, in_channels=1, train_cfg=train_cfg, anchor_generator=dict(type='AnchorGenerator', ratios=[1.0], octave_base_scale=8, scales_per_octave=1, strides=[8]), loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_bbox=dict(type='GIoULoss', loss_weight=1.3), loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5))
    cls_scores = [torch.ones(2, 4, 5, 5)]
    bbox_preds = [torch.ones(2, 4, 5, 5)]
    iou_preds = [torch.ones(2, 1, 5, 5)]
    cfg = mmcv.Config(dict(nms_pre=1000, min_bbox_size=0, score_thr=0.05, nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))
    rescale = False
    self.get_bboxes(cls_scores, bbox_preds, iou_preds, img_metas, cfg, rescale=rescale)

def test_tpfp_imagenet():
    result = tpfp_imagenet(det_bboxes, gt_bboxes, gt_bboxes_ignore=gt_ignore, use_legacy_coordinate=True)
    tp = result[0]
    fp = result[1]
    assert tp.shape == (1, 3)
    assert fp.shape == (1, 3)
    assert (tp == np.array([[1, 1, 0]])).all()
    assert (fp == np.array([[0, 0, 1]])).all()
    result = tpfp_imagenet(det_bboxes, gt_bboxes, gt_bboxes_ignore=gt_ignore, use_legacy_coordinate=False)
    tp = result[0]
    fp = result[1]
    assert tp.shape == (1, 3)
    assert fp.shape == (1, 3)
    assert (tp == np.array([[1, 1, 0]])).all()
    assert (fp == np.array([[0, 0, 1]])).all()

def test_tpfp_default():
    result = tpfp_default(det_bboxes, gt_bboxes, gt_bboxes_ignore=gt_ignore, use_legacy_coordinate=True)
    tp = result[0]
    fp = result[1]
    assert tp.shape == (1, 3)
    assert fp.shape == (1, 3)
    assert (tp == np.array([[1, 1, 0]])).all()
    assert (fp == np.array([[0, 0, 1]])).all()
    result = tpfp_default(det_bboxes, gt_bboxes, gt_bboxes_ignore=gt_ignore, use_legacy_coordinate=False)
    tp = result[0]
    fp = result[1]
    assert tp.shape == (1, 3)
    assert fp.shape == (1, 3)
    assert (tp == np.array([[1, 1, 0]])).all()
    assert (fp == np.array([[0, 0, 1]])).all()

def test_tpfp_openimages():
    det_bboxes = np.array([[10, 10, 15, 15, 1.0], [15, 15, 30, 30, 0.98], [10, 10, 25, 25, 0.98], [28, 28, 35, 35, 0.97], [30, 30, 51, 51, 0.96], [100, 110, 120, 130, 0.15]])
    gt_bboxes = np.array([[10.0, 10.0, 30.0, 30.0], [30.0, 30.0, 50.0, 50.0]])
    gt_groups_of = np.array([True, False], dtype=np.bool)
    gt_ignore = np.zeros((0, 4))
    result = tpfp_openimages(det_bboxes, gt_bboxes, gt_bboxes_ignore=gt_ignore, gt_bboxes_group_of=gt_groups_of, use_group_of=True, ioa_thr=0.5)
    tp = result[0]
    fp = result[1]
    cls_dets = result[2]
    assert tp.shape == (1, 4)
    assert fp.shape == (1, 4)
    assert cls_dets.shape == (4, 5)
    assert (tp == np.array([[0, 1, 0, 1]])).all()
    assert (fp == np.array([[1, 0, 1, 0]])).all()
    cls_dets_gt = np.array([[28.0, 28.0, 35.0, 35.0, 0.97], [30.0, 30.0, 51.0, 51.0, 0.96], [100.0, 110.0, 120.0, 130.0, 0.15], [10.0, 10.0, 15.0, 15.0, 1.0]])
    assert (cls_dets == cls_dets_gt).all()
    result = tpfp_openimages(det_bboxes, gt_bboxes, gt_bboxes_ignore=gt_ignore, gt_bboxes_group_of=gt_groups_of, use_group_of=False, ioa_thr=0.5)
    tp = result[0]
    fp = result[1]
    cls_dets = result[2]
    assert tp.shape == (1, 6)
    assert fp.shape == (1, 6)
    assert cls_dets.shape == (6, 5)
    gt_groups_of = np.array([True, True], dtype=np.bool)
    result = tpfp_openimages(det_bboxes, gt_bboxes, gt_bboxes_ignore=gt_ignore, gt_bboxes_group_of=gt_groups_of, use_group_of=True, ioa_thr=0.5)
    tp = result[0]
    fp = result[1]
    cls_dets = result[2]
    assert tp.shape == (1, 3)
    assert fp.shape == (1, 3)
    assert cls_dets.shape == (3, 5)
    gt_bboxes = np.zeros((0, 4))
    gt_groups_of = np.empty(0)
    result = tpfp_openimages(det_bboxes, gt_bboxes, gt_bboxes_ignore=gt_ignore, gt_bboxes_group_of=gt_groups_of, use_group_of=True, ioa_thr=0.5)
    fp = result[1]
    assert (fp == np.array([[1, 1, 1, 1, 1, 1]])).all()

def test_eval_recalls():
    gts = [gt_bboxes, gt_bboxes, gt_bboxes]
    proposals = [det_bboxes, det_bboxes, det_bboxes]
    recall = eval_recalls(gts, proposals, proposal_nums=2, use_legacy_coordinate=True)
    assert recall.shape == (1, 1)
    assert 0.66 < recall[0][0] < 0.667
    recall = eval_recalls(gts, proposals, proposal_nums=2, use_legacy_coordinate=False)
    assert recall.shape == (1, 1)
    assert 0.66 < recall[0][0] < 0.667
    recall = eval_recalls(gts, proposals, proposal_nums=2, use_legacy_coordinate=True)
    assert recall.shape == (1, 1)
    assert 0.66 < recall[0][0] < 0.667
    recall = eval_recalls(gts, proposals, iou_thrs=[0.1, 0.9], proposal_nums=2, use_legacy_coordinate=False)
    assert recall.shape == (1, 2)
    assert recall[0][1] <= recall[0][0]
    recall = eval_recalls(gts, proposals, iou_thrs=[0.1, 0.9], proposal_nums=2, use_legacy_coordinate=True)
    assert recall.shape == (1, 2)
    assert recall[0][1] <= recall[0][0]

def test_bbox_overlaps_2d(eps=1e-07):

    def _construct_bbox(num_bbox=None):
        img_h = int(np.random.randint(3, 1000))
        img_w = int(np.random.randint(3, 1000))
        if num_bbox is None:
            num_bbox = np.random.randint(1, 10)
        x1y1 = torch.rand((num_bbox, 2))
        x2y2 = torch.max(torch.rand((num_bbox, 2)), x1y1)
        bboxes = torch.cat((x1y1, x2y2), -1)
        bboxes[:, 0::2] *= img_w
        bboxes[:, 1::2] *= img_h
        return (bboxes, num_bbox)
    self = BboxOverlaps2D()
    bboxes1, num_bbox = _construct_bbox()
    bboxes2, _ = _construct_bbox(num_bbox)
    bboxes1 = torch.cat((bboxes1, torch.rand((num_bbox, 1))), 1)
    bboxes2 = torch.cat((bboxes2, torch.rand((num_bbox, 1))), 1)
    gious = self(bboxes1, bboxes2, 'giou', True)
    assert gious.size() == (num_bbox,), gious.size()
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    bboxes1 = torch.empty((0, 4))
    bboxes2 = torch.empty((0, 4))
    gious = self(bboxes1, bboxes2, 'giou', True)
    assert gious.size() == (0,), gious.size()
    assert torch.all(gious == torch.empty((0,)))
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    bboxes1, num_bbox = _construct_bbox()
    bboxes2, _ = _construct_bbox(num_bbox)
    bboxes1 = bboxes1.unsqueeze(0).repeat(2, 1, 1)
    with pytest.raises(AssertionError):
        self(bboxes1, bboxes2.unsqueeze(0).repeat(3, 1, 1), 'giou', True)
    bboxes2 = bboxes2.unsqueeze(0).repeat(2, 1, 1)
    gious = self(bboxes1, bboxes2, 'giou', True)
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    assert gious.size() == (2, num_bbox)
    bboxes1 = bboxes1.unsqueeze(0).repeat(2, 1, 1, 1)
    bboxes2 = bboxes2.unsqueeze(0).repeat(2, 1, 1, 1)
    gious = self(bboxes1, bboxes2, 'giou', True)
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    assert gious.size() == (2, 2, num_bbox)
    bboxes1, num_bbox1 = _construct_bbox()
    bboxes2, num_bbox2 = _construct_bbox()
    gious = self(bboxes1, bboxes2, 'giou')
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    assert gious.size() == (num_bbox1, num_bbox2)
    bboxes1 = bboxes1.unsqueeze(0).repeat(2, 1, 1)
    bboxes2 = bboxes2.unsqueeze(0).repeat(2, 1, 1)
    gious = self(bboxes1, bboxes2, 'giou')
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    assert gious.size() == (2, num_bbox1, num_bbox2)
    bboxes1 = bboxes1.unsqueeze(0)
    bboxes2 = bboxes2.unsqueeze(0)
    gious = self(bboxes1, bboxes2, 'giou')
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    assert gious.size() == (1, 2, num_bbox1, num_bbox2)
    gious = self(torch.empty(1, 2, 0, 4), bboxes2, 'giou')
    assert torch.all(gious == torch.empty(1, 2, 0, bboxes2.size(-2)))
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    bboxes1 = torch.FloatTensor([[0, 0, 10, 10], [10, 10, 20, 20], [32, 32, 38, 42]])
    bboxes2 = torch.FloatTensor([[0, 0, 10, 20], [0, 10, 10, 19], [10, 10, 20, 20]])
    gious = bbox_overlaps(bboxes1, bboxes2, 'giou', is_aligned=True, eps=eps)
    gious = gious.numpy().round(4)
    expected_gious = np.array([0.5, -0.05, -0.8214])
    assert np.allclose(gious, expected_gious, rtol=0, atol=eps)
    ious = bbox_overlaps(bboxes1, bboxes2, 'iof', is_aligned=True, eps=eps)
    assert torch.all(ious >= -1) and torch.all(ious <= 1)
    assert ious.size() == (bboxes1.size(0),)
    ious = bbox_overlaps(bboxes1, bboxes2, 'iof', eps=eps)
    assert torch.all(ious >= -1) and torch.all(ious <= 1)
    assert ious.size() == (bboxes1.size(0), bboxes2.size(0))

def _construct_bbox(num_bbox=None):
    img_h = int(np.random.randint(3, 1000))
    img_w = int(np.random.randint(3, 1000))
    if num_bbox is None:
        num_bbox = np.random.randint(1, 10)
    x1y1 = torch.rand((num_bbox, 2))
    x2y2 = torch.max(torch.rand((num_bbox, 2)), x1y1)
    bboxes = torch.cat((x1y1, x2y2), -1)
    bboxes[:, 0::2] *= img_w
    bboxes[:, 1::2] *= img_h
    return (bboxes.numpy(), num_bbox)

def test_voc_recall_overlaps():

    def _construct_bbox(num_bbox=None):
        img_h = int(np.random.randint(3, 1000))
        img_w = int(np.random.randint(3, 1000))
        if num_bbox is None:
            num_bbox = np.random.randint(1, 10)
        x1y1 = torch.rand((num_bbox, 2))
        x2y2 = torch.max(torch.rand((num_bbox, 2)), x1y1)
        bboxes = torch.cat((x1y1, x2y2), -1)
        bboxes[:, 0::2] *= img_w
        bboxes[:, 1::2] *= img_h
        return (bboxes.numpy(), num_bbox)
    bboxes1, num_bbox = _construct_bbox()
    bboxes2, _ = _construct_bbox(num_bbox)
    ious = recall_overlaps(bboxes1, bboxes2, 'iou', use_legacy_coordinate=False)
    assert ious.shape == (num_bbox, num_bbox)
    assert np.all(ious >= -1) and np.all(ious <= 1)
    ious = recall_overlaps(bboxes1, bboxes2, 'iou', use_legacy_coordinate=True)
    assert ious.shape == (num_bbox, num_bbox)
    assert np.all(ious >= -1) and np.all(ious <= 1)

