# Cluster 37

@BBOX_ASSIGNERS.register_module()
class CenterRegionAssigner(BaseAssigner):
    """Assign pixels at the center region of a bbox as positive.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.
    - -1: negative samples
    - semi-positive numbers: positive sample, index (0-based) of assigned gt

    Args:
        pos_scale (float): Threshold within which pixels are
          labelled as positive.
        neg_scale (float): Threshold above which pixels are
          labelled as positive.
        min_pos_iof (float): Minimum iof of a pixel with a gt to be
          labelled as positive. Default: 1e-2
        ignore_gt_scale (float): Threshold within which the pixels
          are ignored when the gt is labelled as shadowed. Default: 0.5
        foreground_dominate (bool): If True, the bbox will be assigned as
          positive when a gt's kernel region overlaps with another's shadowed
          (ignored) region, otherwise it is set as ignored. Default to False.
    """

    def __init__(self, pos_scale, neg_scale, min_pos_iof=0.01, ignore_gt_scale=0.5, foreground_dominate=False, iou_calculator=dict(type='BboxOverlaps2D')):
        self.pos_scale = pos_scale
        self.neg_scale = neg_scale
        self.min_pos_iof = min_pos_iof
        self.ignore_gt_scale = ignore_gt_scale
        self.foreground_dominate = foreground_dominate
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def get_gt_priorities(self, gt_bboxes):
        """Get gt priorities according to their areas.

        Smaller gt has higher priority.

        Args:
            gt_bboxes (Tensor): Ground truth boxes, shape (k, 4).

        Returns:
            Tensor: The priority of gts so that gts with larger priority is               more likely to be assigned. Shape (k, )
        """
        gt_areas = bboxes_area(gt_bboxes)
        _, sort_idx = gt_areas.sort(descending=True)
        sort_idx = sort_idx.argsort()
        return sort_idx

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assigns gts to every bbox (proposal/anchor), each bbox         will be assigned with -1, or a semi-positive number. -1 means         negative sample, semi-positive number is the index (0-based) of         assigned gt.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (tensor, optional): Ground truth bboxes that are
              labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (tensor, optional): Label of gt_bboxes, shape (num_gts,).

        Returns:
            :obj:`AssignResult`: The assigned result. Note that               shadowed_labels of shape (N, 2) is also added as an               `assign_result` attribute. `shadowed_labels` is a tensor               composed of N pairs of anchor_ind, class_label], where N               is the number of anchors that lie in the outer region of a               gt, anchor_ind is the shadowed anchor index and class_label               is the shadowed class label.

        Example:
            >>> self = CenterRegionAssigner(0.2, 0.2)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 10]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        assert bboxes.size(1) == 4, 'bboxes must have size of 4'
        gt_core = scale_boxes(gt_bboxes, self.pos_scale)
        gt_shadow = scale_boxes(gt_bboxes, self.neg_scale)
        bbox_centers = (bboxes[:, 2:4] + bboxes[:, 0:2]) / 2
        is_bbox_in_gt = is_located_in(bbox_centers, gt_bboxes)
        bbox_and_gt_core_overlaps = self.iou_calculator(bboxes, gt_core, mode='iof')
        is_bbox_in_gt_core = is_bbox_in_gt & (bbox_and_gt_core_overlaps > self.min_pos_iof)
        is_bbox_in_gt_shadow = self.iou_calculator(bboxes, gt_shadow, mode='iof') > self.min_pos_iof
        is_bbox_in_gt_shadow &= ~is_bbox_in_gt_core
        num_gts, num_bboxes = (gt_bboxes.size(0), bboxes.size(0))
        if num_gts == 0 or num_bboxes == 0:
            assigned_gt_ids = is_bbox_in_gt_core.new_zeros((num_bboxes,), dtype=torch.long)
            pixels_in_gt_shadow = assigned_gt_ids.new_empty((0, 2))
        else:
            sort_idx = self.get_gt_priorities(gt_bboxes)
            assigned_gt_ids, pixels_in_gt_shadow = self.assign_one_hot_gt_indices(is_bbox_in_gt_core, is_bbox_in_gt_shadow, gt_priority=sort_idx)
        if gt_bboxes_ignore is not None and gt_bboxes_ignore.numel() > 0:
            gt_bboxes_ignore = scale_boxes(gt_bboxes_ignore, scale=self.ignore_gt_scale)
            is_bbox_in_ignored_gts = is_located_in(bbox_centers, gt_bboxes_ignore)
            is_bbox_in_ignored_gts = is_bbox_in_ignored_gts.any(dim=1)
            assigned_gt_ids[is_bbox_in_ignored_gts] = -1
        assigned_labels = None
        shadowed_pixel_labels = None
        if gt_labels is not None:
            assigned_labels = assigned_gt_ids.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(assigned_gt_ids > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_ids[pos_inds] - 1]
            shadowed_pixel_labels = pixels_in_gt_shadow.clone()
            if pixels_in_gt_shadow.numel() > 0:
                pixel_idx, gt_idx = (pixels_in_gt_shadow[:, 0], pixels_in_gt_shadow[:, 1])
                assert (assigned_gt_ids[pixel_idx] != gt_idx).all(), 'Some pixels are dually assigned to ignore and gt!'
                shadowed_pixel_labels[:, 1] = gt_labels[gt_idx - 1]
                override = assigned_labels[pixel_idx] == shadowed_pixel_labels[:, 1]
                if self.foreground_dominate:
                    shadowed_pixel_labels = shadowed_pixel_labels[~override]
                else:
                    assigned_labels[pixel_idx[override]] = -1
                    assigned_gt_ids[pixel_idx[override]] = 0
        assign_result = AssignResult(num_gts, assigned_gt_ids, None, labels=assigned_labels)
        assign_result.set_extra_property('shadowed_labels', shadowed_pixel_labels)
        return assign_result

    def assign_one_hot_gt_indices(self, is_bbox_in_gt_core, is_bbox_in_gt_shadow, gt_priority=None):
        """Assign only one gt index to each prior box.

        Gts with large gt_priority are more likely to be assigned.

        Args:
            is_bbox_in_gt_core (Tensor): Bool tensor indicating the bbox center
              is in the core area of a gt (e.g. 0-0.2).
              Shape: (num_prior, num_gt).
            is_bbox_in_gt_shadow (Tensor): Bool tensor indicating the bbox
              center is in the shadowed area of a gt (e.g. 0.2-0.5).
              Shape: (num_prior, num_gt).
            gt_priority (Tensor): Priorities of gts. The gt with a higher
              priority is more likely to be assigned to the bbox when the bbox
              match with multiple gts. Shape: (num_gt, ).

        Returns:
            tuple: Returns (assigned_gt_inds, shadowed_gt_inds).

                - assigned_gt_inds: The assigned gt index of each prior bbox                     (i.e. index from 1 to num_gts). Shape: (num_prior, ).
                - shadowed_gt_inds: shadowed gt indices. It is a tensor of                     shape (num_ignore, 2) with first column being the                     shadowed prior bbox indices and the second column the                     shadowed gt indices (1-based).
        """
        num_bboxes, num_gts = is_bbox_in_gt_core.shape
        if gt_priority is None:
            gt_priority = torch.arange(num_gts, device=is_bbox_in_gt_core.device)
        assert gt_priority.size(0) == num_gts
        assigned_gt_inds = is_bbox_in_gt_core.new_zeros((num_bboxes,), dtype=torch.long)
        shadowed_gt_inds = torch.nonzero(is_bbox_in_gt_shadow, as_tuple=False)
        if is_bbox_in_gt_core.sum() == 0:
            shadowed_gt_inds[:, 1] += 1
            return (assigned_gt_inds, shadowed_gt_inds)
        pair_priority = is_bbox_in_gt_core.new_full((num_bboxes, num_gts), -1, dtype=torch.long)
        inds_of_match = torch.any(is_bbox_in_gt_core, dim=1)
        matched_bbox_gt_inds = torch.nonzero(is_bbox_in_gt_core, as_tuple=False)[:, 1]
        pair_priority[is_bbox_in_gt_core] = gt_priority[matched_bbox_gt_inds]
        _, argmax_priority = pair_priority[inds_of_match].max(dim=1)
        assigned_gt_inds[inds_of_match] = argmax_priority + 1
        is_bbox_in_gt_core[inds_of_match, argmax_priority] = 0
        shadowed_gt_inds = torch.cat((shadowed_gt_inds, torch.nonzero(is_bbox_in_gt_core, as_tuple=False)), dim=0)
        is_bbox_in_gt_core[inds_of_match, argmax_priority] = 1
        if shadowed_gt_inds.numel() > 0:
            shadowed_gt_inds[:, 1] += 1
        return (assigned_gt_inds, shadowed_gt_inds)

def scale_boxes(bboxes, scale):
    """Expand an array of boxes by a given scale.

    Args:
        bboxes (Tensor): Shape (m, 4)
        scale (float): The scale factor of bboxes

    Returns:
        (Tensor): Shape (m, 4). Scaled bboxes
    """
    assert bboxes.size(1) == 4
    w_half = (bboxes[:, 2] - bboxes[:, 0]) * 0.5
    h_half = (bboxes[:, 3] - bboxes[:, 1]) * 0.5
    x_c = (bboxes[:, 2] + bboxes[:, 0]) * 0.5
    y_c = (bboxes[:, 3] + bboxes[:, 1]) * 0.5
    w_half *= scale
    h_half *= scale
    boxes_scaled = torch.zeros_like(bboxes)
    boxes_scaled[:, 0] = x_c - w_half
    boxes_scaled[:, 2] = x_c + w_half
    boxes_scaled[:, 1] = y_c - h_half
    boxes_scaled[:, 3] = y_c + h_half
    return boxes_scaled

def is_located_in(points, bboxes):
    """Are points located in bboxes.

    Args:
      points (Tensor): Points, shape: (m, 2).
      bboxes (Tensor): Bounding boxes, shape: (n, 4).

    Return:
      Tensor: Flags indicating if points are located in bboxes, shape: (m, n).
    """
    assert points.size(1) == 2
    assert bboxes.size(1) == 4
    return (points[:, 0].unsqueeze(1) > bboxes[:, 0].unsqueeze(0)) & (points[:, 0].unsqueeze(1) < bboxes[:, 2].unsqueeze(0)) & (points[:, 1].unsqueeze(1) > bboxes[:, 1].unsqueeze(0)) & (points[:, 1].unsqueeze(1) < bboxes[:, 3].unsqueeze(0))

