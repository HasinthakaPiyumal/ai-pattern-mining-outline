# Cluster 57

@LOSSES.register_module()
class BoundedIoULoss(nn.Module):

    def __init__(self, beta=0.2, eps=0.001, reduction='mean', loss_weight=1.0):
        super(BoundedIoULoss, self).__init__()
        self.beta = beta
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
        loss = self.loss_weight * bounded_iou_loss(pred, target, weight, beta=self.beta, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def bounded_iou_loss(pred, target, beta=0.2, eps=0.001):
    """BIoULoss.

    This is an implementation of paper
    `Improving Object Localization with Fitness NMS and Bounded IoU Loss.
    <https://arxiv.org/abs/1711.00164>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Target bboxes.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]
    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]
    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry
    loss_dx = 1 - torch.max((target_w - 2 * dx.abs()) / (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max((target_h - 2 * dy.abs()) / (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w / (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h / (target_h + eps))
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh], dim=-1).flatten(1)
    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta, loss_comb - 0.5 * beta)
    return loss

