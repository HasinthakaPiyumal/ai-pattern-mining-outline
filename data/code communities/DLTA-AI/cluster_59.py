# Cluster 59

@LOSSES.register_module()
class CIoULoss(nn.Module):

    def __init__(self, eps=1e-06, reduction='mean', loss_weight=1.0):
        super(CIoULoss, self).__init__()
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
        loss = self.loss_weight * ciou_loss(pred, target, weight, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def ciou_loss(pred, target, eps=1e-07):
    """`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps
    ious = overlap / union
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]
    c2 = cw ** 2 + ch ** 2 + eps
    b1_x1, b1_y1 = (pred[:, 0], pred[:, 1])
    b1_x2, b1_y2 = (pred[:, 2], pred[:, 3])
    b2_x1, b2_y1 = (target[:, 0], target[:, 1])
    b2_x2, b2_y2 = (target[:, 2], target[:, 3])
    w1, h1 = (b1_x2 - b1_x1, b1_y2 - b1_y1 + eps)
    w2, h2 = (b2_x2 - b2_x1, b2_y2 - b2_y1 + eps)
    left = (b2_x1 + b2_x2 - (b1_x1 + b1_x2)) ** 2 / 4
    right = (b2_y1 + b2_y2 - (b1_y1 + b1_y2)) ** 2 / 4
    rho2 = left + right
    factor = 4 / math.pi ** 2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = (ious > 0.5).float() * v / (1 - ious + v)
    cious = ious - (rho2 / c2 + alpha * v)
    loss = 1 - cious.clamp(min=-1.0, max=1.0)
    return loss

