# Cluster 65

def interpolate_as(source, target, mode='bilinear', align_corners=False):
    """Interpolate the `source` to the shape of the `target`.

    The `source` must be a Tensor, but the `target` can be a Tensor or a
    np.ndarray with the shape (..., target_h, target_w).

    Args:
        source (Tensor): A 3D/4D Tensor with the shape (N, H, W) or
            (N, C, H, W).
        target (Tensor | np.ndarray): The interpolation target with the shape
            (..., target_h, target_w).
        mode (str): Algorithm used for interpolation. The options are the
            same as those in F.interpolate(). Default: ``'bilinear'``.
        align_corners (bool): The same as the argument in F.interpolate().

    Returns:
        Tensor: The interpolated source Tensor.
    """
    assert len(target.shape) >= 2

    def _interpolate_as(source, target, mode='bilinear', align_corners=False):
        """Interpolate the `source` (4D) to the shape of the `target`."""
        target_h, target_w = target.shape[-2:]
        source_h, source_w = source.shape[-2:]
        if target_h != source_h or target_w != source_w:
            source = F.interpolate(source, size=(target_h, target_w), mode=mode, align_corners=align_corners)
        return source
    if len(source.shape) == 3:
        source = source[:, None, :, :]
        source = _interpolate_as(source, target, mode, align_corners)
        return source[:, 0, :, :]
    else:
        return _interpolate_as(source, target, mode, align_corners)

def _interpolate_as(source, target, mode='bilinear', align_corners=False):
    """Interpolate the `source` (4D) to the shape of the `target`."""
    target_h, target_w = target.shape[-2:]
    source_h, source_w = source.shape[-2:]
    if target_h != source_h or target_w != source_w:
        source = F.interpolate(source, size=(target_h, target_w), mode=mode, align_corners=align_corners)
    return source

class BaseSemanticHead(BaseModule, metaclass=ABCMeta):
    """Base module of Semantic Head.

    Args:
        num_classes (int): the number of classes.
        init_cfg (dict): the initialization config.
        loss_seg (dict): the loss of the semantic head.
    """

    def __init__(self, num_classes, init_cfg=None, loss_seg=dict(type='CrossEntropyLoss', ignore_index=255, loss_weight=1.0)):
        super(BaseSemanticHead, self).__init__(init_cfg)
        self.loss_seg = build_loss(loss_seg)
        self.num_classes = num_classes

    @force_fp32(apply_to=('seg_preds',))
    def loss(self, seg_preds, gt_semantic_seg):
        """Get the loss of semantic head.

        Args:
            seg_preds (Tensor): The input logits with the shape (N, C, H, W).
            gt_semantic_seg: The ground truth of semantic segmentation with
                the shape (N, H, W).
            label_bias: The starting number of the semantic label.
                Default: 1.

        Returns:
            dict: the loss of semantic head.
        """
        if seg_preds.shape[-2:] != gt_semantic_seg.shape[-2:]:
            seg_preds = interpolate_as(seg_preds, gt_semantic_seg)
        seg_preds = seg_preds.permute((0, 2, 3, 1))
        loss_seg = self.loss_seg(seg_preds.reshape(-1, self.num_classes), gt_semantic_seg.reshape(-1).long())
        return dict(loss_seg=loss_seg)

    @abstractmethod
    def forward(self, x):
        """Placeholder of forward function.

        Returns:
            dict[str, Tensor]: A dictionary, including features
                and predicted scores. Required keys: 'seg_preds'
                and 'feats'.
        """
        pass

    def forward_train(self, x, gt_semantic_seg):
        output = self.forward(x)
        seg_preds = output['seg_preds']
        return self.loss(seg_preds, gt_semantic_seg)

    def simple_test(self, x, img_metas, rescale=False):
        output = self.forward(x)
        seg_preds = output['seg_preds']
        seg_preds = F.interpolate(seg_preds, size=img_metas[0]['pad_shape'][:2], mode='bilinear', align_corners=False)
        if rescale:
            h, w, _ = img_metas[0]['img_shape']
            seg_preds = seg_preds[:, :, :h, :w]
            h, w, _ = img_metas[0]['ori_shape']
            seg_preds = F.interpolate(seg_preds, size=(h, w), mode='bilinear', align_corners=False)
        return seg_preds

def test_interpolate_as():
    source = torch.rand((1, 5, 4, 4))
    target = torch.rand((1, 1, 16, 16))
    result = interpolate_as(source, target)
    assert result.shape == torch.Size((1, 5, 16, 16))
    result = interpolate_as(source, target.squeeze(0))
    assert result.shape == torch.Size((1, 5, 16, 16))
    result = interpolate_as(source.squeeze(0), target)
    assert result.shape == torch.Size((5, 16, 16))
    target = np.random.rand(16, 16)
    result = interpolate_as(source.squeeze(0), target)
    assert result.shape == torch.Size((5, 16, 16))

