# Cluster 38

@BBOX_CODERS.register_module()
class DistancePointBBoxCoder(BaseBBoxCoder):
    """Distance Point BBox coder.

    This coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left,
    right) and decode it back to the original.

    Args:
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    def __init__(self, clip_border=True):
        super(BaseBBoxCoder, self).__init__()
        self.clip_border = clip_border

    def encode(self, points, gt_bboxes, max_dis=None, eps=0.1):
        """Encode bounding box to distances.

        Args:
            points (Tensor): Shape (N, 2), The format is [x, y].
            gt_bboxes (Tensor): Shape (N, 4), The format is "xyxy"
            max_dis (float): Upper bound of the distance. Default None.
            eps (float): a small value to ensure target < max_dis, instead <=.
                Default 0.1.

        Returns:
            Tensor: Box transformation deltas. The shape is (N, 4).
        """
        assert points.size(0) == gt_bboxes.size(0)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 4
        return bbox2distance(points, gt_bboxes, max_dis, eps)

    def decode(self, points, pred_bboxes, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (B, N, 2) or (N, 2).
            pred_bboxes (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom). Shape (B, N, 4)
                or (N, 4)
            max_shape (Sequence[int] or torch.Tensor or Sequence[
                Sequence[int]],optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W). If priors shape is (B, N, 4), then
                the max_shape should be a Sequence[Sequence[int]],
                and the length of max_shape should also be B.
                Default None.
        Returns:
            Tensor: Boxes with shape (N, 4) or (B, N, 4)
        """
        assert points.size(0) == pred_bboxes.size(0)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 4
        if self.clip_border is False:
            max_shape = None
        return distance2bbox(points, pred_bboxes, max_shape)

def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)

