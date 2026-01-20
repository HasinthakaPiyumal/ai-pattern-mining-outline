# Cluster 81

@HEADS.register_module()
class FCNMaskHead(BaseModule):

    def __init__(self, num_convs=4, roi_feat_size=14, in_channels=256, conv_kernel_size=3, conv_out_channels=256, num_classes=80, class_agnostic=False, upsample_cfg=dict(type='deconv', scale_factor=2), conv_cfg=None, norm_cfg=None, predictor_cfg=dict(type='Conv'), loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0), init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization behavior, init_cfg is not allowed to be set'
        super(FCNMaskHead, self).__init__(init_cfg)
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [None, 'deconv', 'nearest', 'bilinear', 'carafe']:
            raise ValueError(f'Invalid upsample method {self.upsample_cfg['type']}, accepted methods are "deconv", "nearest", "bilinear", "carafe"')
        self.num_convs = num_convs
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.predictor_cfg = predictor_cfg
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)
        self.convs = ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else self.conv_out_channels
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(ConvModule(in_channels, self.conv_out_channels, self.conv_kernel_size, padding=padding, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
        upsample_in_channels = self.conv_out_channels if self.num_convs > 0 else in_channels
        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.update(in_channels=upsample_in_channels, out_channels=self.conv_out_channels, kernel_size=self.scale_factor, stride=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        elif self.upsample_method == 'carafe':
            upsample_cfg_.update(channels=upsample_in_channels, scale_factor=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        else:
            align_corners = None if self.upsample_method == 'nearest' else False
            upsample_cfg_.update(scale_factor=self.scale_factor, mode=self.upsample_method, align_corners=align_corners)
            self.upsample = build_upsample_layer(upsample_cfg_)
        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = self.conv_out_channels if self.upsample_method == 'deconv' else upsample_in_channels
        self.conv_logits = build_conv_layer(self.predictor_cfg, logits_in_channel, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        super(FCNMaskHead, self).init_weights()
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            elif isinstance(m, CARAFEPack):
                m.init_weights()
            elif hasattr(m, 'weight') and hasattr(m, 'bias'):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds, gt_masks, rcnn_train_cfg)
        return mask_targets

    @force_fp32(apply_to=('mask_pred',))
    def loss(self, mask_pred, mask_targets, labels):
        """
        Example:
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> # There are lots of variations depending on the configuration
            >>> self = FCNMaskHead(num_classes=C, num_convs=1)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_pred = self.forward(inputs)
            >>> sf = self.scale_factor
            >>> labels = torch.randint(0, C, size=(N,))
            >>> # With the default properties the mask targets should indicate
            >>> # a (potentially soft) single-class label
            >>> mask_targets = torch.rand(N, H * sf, W * sf)
            >>> loss = self.loss(mask_pred, mask_targets, labels)
            >>> print('loss = {!r}'.format(loss))
        """
        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
        elif self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets, torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape (Tuple): original image height and width, shape (2,)
            scale_factor(ndarray | Tensor): If ``rescale is True``, box
                coordinates are divided by this scale factor to fit
                ``ori_shape``.
            rescale (bool): If True, the resulting masks will be rescaled to
                ``ori_shape``.

        Returns:
            list[list]: encoded masks. The c-th item in the outer list
                corresponds to the c-th class. Given the c-th outer list, the
                i-th item in that inner list is the mask for the i-th box with
                class label c.

        Example:
            >>> import mmcv
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> self = FCNMaskHead(num_classes=C, num_convs=0)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_pred = self.forward(inputs)
            >>> # Each input is associated with some bounding box
            >>> det_bboxes = torch.Tensor([[1, 1, 42, 42 ]] * N)
            >>> det_labels = torch.randint(0, C, size=(N,))
            >>> rcnn_test_cfg = mmcv.Config({'mask_thr_binary': 0, })
            >>> ori_shape = (H * 4, W * 4)
            >>> scale_factor = torch.FloatTensor((1, 1))
            >>> rescale = False
            >>> # Encoded masks are a list for each category.
            >>> encoded_masks = self.get_seg_masks(
            >>>     mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape,
            >>>     scale_factor, rescale
            >>> )
            >>> assert len(encoded_masks) == C
            >>> assert sum(list(map(len, encoded_masks))) == N
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            mask_pred = det_bboxes.new_tensor(mask_pred)
        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)]
        bboxes = det_bboxes[:, :4]
        labels = det_labels
        if not isinstance(scale_factor, torch.Tensor):
            if isinstance(scale_factor, float):
                scale_factor = np.array([scale_factor] * 4)
                warn('Scale_factor should be a Tensor or ndarray with shape (4,), float would be deprecated. ')
            assert isinstance(scale_factor, np.ndarray)
            scale_factor = torch.Tensor(scale_factor)
        if rescale:
            img_h, img_w = ori_shape[:2]
            bboxes = bboxes / scale_factor.to(bboxes)
        else:
            w_scale, h_scale = (scale_factor[0], scale_factor[1])
            img_h = np.round(ori_shape[0] * h_scale.item()).astype(np.int32)
            img_w = np.round(ori_shape[1] * w_scale.item()).astype(np.int32)
        N = len(mask_pred)
        if device.type == 'cpu':
            num_chunks = N
        else:
            num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert num_chunks <= N, 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)
        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(N, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8)
        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]
        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(mask_pred[inds], bboxes[inds], img_h, img_w, skip_empty=device.type == 'cpu')
            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)
            im_mask[(inds,) + spatial_inds] = masks_chunk
        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy())
        return cls_segms

    def onnx_export(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, **kwargs):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor): shape (n, #class, h, w).
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape (Tuple): original image height and width, shape (2,)

        Returns:
            Tensor: a mask of shape (N, img_h, img_w).
        """
        mask_pred = mask_pred.sigmoid()
        bboxes = det_bboxes[:, :4]
        labels = det_labels
        img_h, img_w = ori_shape[:2]
        threshold = rcnn_test_cfg.mask_thr_binary
        if not self.class_agnostic:
            box_inds = torch.arange(mask_pred.shape[0])
            mask_pred = mask_pred[box_inds, labels][:, None]
        masks, _ = _do_paste_mask(mask_pred, bboxes, img_h, img_w, skip_empty=False)
        if threshold >= 0:
            masks = (masks >= threshold).to(dtype=torch.float)
        return masks

def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, cfg):
    """Compute mask target for positive proposals in multiple images.

    Args:
        pos_proposals_list (list[Tensor]): Positive proposals in multiple
            images.
        pos_assigned_gt_inds_list (list[Tensor]): Assigned GT indices for each
            positive proposals.
        gt_masks_list (list[:obj:`BaseInstanceMasks`]): Ground truth masks of
            each image.
        cfg (dict): Config dict that specifies the mask size.

    Returns:
        list[Tensor]: Mask target of each image.

    Example:
        >>> import mmcv
        >>> import mmdet
        >>> from mmdet.core.mask import BitmapMasks
        >>> from mmdet.core.mask.mask_target import *
        >>> H, W = 17, 18
        >>> cfg = mmcv.Config({'mask_size': (13, 14)})
        >>> rng = np.random.RandomState(0)
        >>> # Positive proposals (tl_x, tl_y, br_x, br_y) for each image
        >>> pos_proposals_list = [
        >>>     torch.Tensor([
        >>>         [ 7.2425,  5.5929, 13.9414, 14.9541],
        >>>         [ 7.3241,  3.6170, 16.3850, 15.3102],
        >>>     ]),
        >>>     torch.Tensor([
        >>>         [ 4.8448, 6.4010, 7.0314, 9.7681],
        >>>         [ 5.9790, 2.6989, 7.4416, 4.8580],
        >>>         [ 0.0000, 0.0000, 0.1398, 9.8232],
        >>>     ]),
        >>> ]
        >>> # Corresponding class index for each proposal for each image
        >>> pos_assigned_gt_inds_list = [
        >>>     torch.LongTensor([7, 0]),
        >>>     torch.LongTensor([5, 4, 1]),
        >>> ]
        >>> # Ground truth mask for each true object for each image
        >>> gt_masks_list = [
        >>>     BitmapMasks(rng.rand(8, H, W), height=H, width=W),
        >>>     BitmapMasks(rng.rand(6, H, W), height=H, width=W),
        >>> ]
        >>> mask_targets = mask_target(
        >>>     pos_proposals_list, pos_assigned_gt_inds_list,
        >>>     gt_masks_list, cfg)
        >>> assert mask_targets.shape == (5,) + cfg['mask_size']
    """
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = list(mask_targets)
    if len(mask_targets) > 0:
        mask_targets = torch.cat(mask_targets)
    return mask_targets

@HEADS.register_module()
class DynamicMaskHead(FCNMaskHead):
    """Dynamic Mask Head for
    `Instances as Queries <http://arxiv.org/abs/2105.01928>`_

    Args:
        num_convs (int): Number of convolution layer.
            Defaults to 4.
        roi_feat_size (int): The output size of RoI extractor,
            Defaults to 14.
        in_channels (int): Input feature channels.
            Defaults to 256.
        conv_kernel_size (int): Kernel size of convolution layers.
            Defaults to 3.
        conv_out_channels (int): Output channels of convolution layers.
            Defaults to 256.
        num_classes (int): Number of classes.
            Defaults to 80
        class_agnostic (int): Whether generate class agnostic prediction.
            Defaults to False.
        dropout (float): Probability of drop the channel.
            Defaults to 0.0
        upsample_cfg (dict): The config for upsample layer.
        conv_cfg (dict): The convolution layer config.
        norm_cfg (dict): The norm layer config.
        dynamic_conv_cfg (dict): The dynamic convolution layer config.
        loss_mask (dict): The config for mask loss.
    """

    def __init__(self, num_convs=4, roi_feat_size=14, in_channels=256, conv_kernel_size=3, conv_out_channels=256, num_classes=80, class_agnostic=False, upsample_cfg=dict(type='deconv', scale_factor=2), conv_cfg=None, norm_cfg=None, dynamic_conv_cfg=dict(type='DynamicConv', in_channels=256, feat_channels=64, out_channels=256, input_feat_shape=14, with_proj=False, act_cfg=dict(type='ReLU', inplace=True), norm_cfg=dict(type='LN')), loss_mask=dict(type='DiceLoss', loss_weight=8.0), **kwargs):
        super(DynamicMaskHead, self).__init__(num_convs=num_convs, roi_feat_size=roi_feat_size, in_channels=in_channels, conv_kernel_size=conv_kernel_size, conv_out_channels=conv_out_channels, num_classes=num_classes, class_agnostic=class_agnostic, upsample_cfg=upsample_cfg, conv_cfg=conv_cfg, norm_cfg=norm_cfg, loss_mask=loss_mask, **kwargs)
        assert class_agnostic is False, 'DynamicMaskHead only support class_agnostic=False'
        self.fp16_enabled = False
        self.instance_interactive_conv = build_transformer(dynamic_conv_cfg)

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            nn.init.constant_(self.conv_logits.bias, 0.0)

    @auto_fp16()
    def forward(self, roi_feat, proposal_feat):
        """Forward function of DynamicMaskHead.

        Args:
            roi_feat (Tensor): Roi-pooling features with shape
                (batch_size*num_proposals, feature_dimensions,
                pooling_h , pooling_w).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size*num_proposals, feature_dimensions)

          Returns:
            mask_pred (Tensor): Predicted foreground masks with shape
                (batch_size*num_proposals, num_classes,
                                        pooling_h*2, pooling_w*2).
        """
        proposal_feat = proposal_feat.reshape(-1, self.in_channels)
        proposal_feat_iic = self.instance_interactive_conv(proposal_feat, roi_feat)
        x = proposal_feat_iic.permute(0, 2, 1).reshape(roi_feat.size())
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    @force_fp32(apply_to=('mask_pred',))
    def loss(self, mask_pred, mask_targets, labels):
        num_pos = labels.new_ones(labels.size()).float().sum()
        avg_factor = torch.clamp(reduce_mean(num_pos), min=1.0).item()
        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
        else:
            loss_mask = self.loss_mask(mask_pred[torch.arange(num_pos).long(), labels, ...].sigmoid(), mask_targets, avg_factor=avg_factor)
        loss['loss_mask'] = loss_mask
        return loss

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds, gt_masks, rcnn_train_cfg)
        return mask_targets

