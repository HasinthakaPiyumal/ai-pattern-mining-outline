# Cluster 64

def transpose_and_gather_feat(feat, ind):
    """Transpose and gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.

    Returns:
        feat (Tensor): Transposed and gathered feature.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat

def gather_feat(feat, ind, mask=None):
    """Gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.
        mask (Tensor | None): Mask of feature map. Default: None.

    Returns:
        feat (Tensor): Gathered feature.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).repeat(1, 1, dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

@HEADS.register_module()
class CenterNetHead(BaseDenseHead, BBoxTestMixin):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self, in_channel, feat_channel, num_classes, loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0), loss_wh=dict(type='L1Loss', loss_weight=0.1), loss_offset=dict(type='L1Loss', loss_weight=1.0), train_cfg=None, test_cfg=None, init_cfg=None):
        super(CenterNetHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel, num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)
        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        return (center_heatmap_pred, wh_pred, offset_pred)

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def loss(self, center_heatmap_preds, wh_preds, offset_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
               shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 2, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1
        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]
        target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels, center_heatmap_pred.shape, img_metas[0]['pad_shape'])
        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']
        loss_center_heatmap = self.loss_center_heatmap(center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)
        loss_wh = self.loss_wh(wh_pred, wh_target, wh_offset_target_weight, avg_factor=avg_factor * 2)
        loss_offset = self.loss_offset(offset_pred, offset_target, wh_offset_target_weight, avg_factor=avg_factor * 2)
        return dict(loss_center_heatmap=loss_center_heatmap, loss_wh=loss_wh, loss_offset=loss_offset)

    def get_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap,                    shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape                    (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape                    (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset                    predict, shape (B, 2, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape
        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)
        center_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_classes, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)
            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind], [ctx_int, cty_int], radius)
                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h
                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int
                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1
        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(center_heatmap_target=center_heatmap_target, wh_target=wh_target, offset_target=offset_target, wh_offset_target_weight=wh_offset_target_weight)
        return (target_result, avg_factor)

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def get_bboxes(self, center_heatmap_preds, wh_preds, offset_preds, img_metas, rescale=True, with_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): Center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): WH predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): Offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(self._get_bboxes_single(center_heatmap_preds[0][img_id:img_id + 1, ...], wh_preds[0][img_id:img_id + 1, ...], offset_preds[0][img_id:img_id + 1, ...], img_metas[img_id], rescale=rescale, with_nms=with_nms))
        return result_list

    def _get_bboxes_single(self, center_heatmap_pred, wh_pred, offset_pred, img_meta, rescale=False, with_nms=True):
        """Transform outputs of a single image into bbox results.

        Args:
            center_heatmap_pred (Tensor): Center heatmap for current level with
                shape (1, num_classes, H, W).
            wh_pred (Tensor): WH heatmap for current level with shape
                (1, num_classes, H, W).
            offset_pred (Tensor): Offset for current level with shape
                (1, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor, Tensor]: The first item is an (n, 5) tensor, where
                5 represent (tl_x, tl_y, br_x, br_y, score) and the score
                between 0 and 1. The shape of the second tensor in the tuple
                is (n,), and each element represents the class label of the
                corresponding box.
        """
        batch_det_bboxes, batch_labels = self.decode_heatmap(center_heatmap_pred, wh_pred, offset_pred, img_meta['batch_input_shape'], k=self.test_cfg.topk, kernel=self.test_cfg.local_maximum_kernel)
        det_bboxes = batch_det_bboxes.view([-1, 5])
        det_labels = batch_labels.view(-1)
        batch_border = det_bboxes.new_tensor(img_meta['border'])[..., [2, 0, 2, 0]]
        det_bboxes[..., :4] -= batch_border
        if rescale:
            det_bboxes[..., :4] /= det_bboxes.new_tensor(img_meta['scale_factor'])
        if with_nms:
            det_bboxes, det_labels = self._bboxes_nms(det_bboxes, det_labels, self.test_cfg)
        return (det_bboxes, det_labels)

    def decode_heatmap(self, center_heatmap_pred, wh_pred, offset_pred, img_shape, k=100, kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with                   shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape
        center_heatmap_pred = get_local_maximum(center_heatmap_pred, kernel=kernel)
        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets
        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)
        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]), dim=-1)
        return (batch_bboxes, batch_topk_labels)

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() > 0:
            max_num = cfg.max_per_img
            bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1].contiguous(), labels, cfg.nms)
            if max_num > 0:
                bboxes = bboxes[:max_num]
                labels = labels[keep][:max_num]
        return (bboxes, labels)

def get_local_maximum(heat, kernel=3):
    """Extract local maximum pixel with given kernel.

    Args:
        heat (Tensor): Target heatmap.
        kernel (int): Kernel size of max pooling. Default: 3.

    Returns:
        heat (Tensor): A heatmap where local maximum pixels maintain its
            own value and other positions are 0.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def get_topk_from_heatmap(scores, k=20):
    """Get top k positions from heatmap.

    Args:
        scores (Tensor): Target heatmap with shape
            [batch, num_classes, height, width].
        k (int): Target number. Default: 20.

    Returns:
        tuple[torch.Tensor]: Scores, indexes, categories and coords of
            topk keypoint. Containing following Tensors:

        - topk_scores (Tensor): Max scores of each topk keypoint.
        - topk_inds (Tensor): Indexes of each topk keypoint.
        - topk_clses (Tensor): Categories of each topk keypoint.
        - topk_ys (Tensor): Y-coord of each topk keypoint.
        - topk_xs (Tensor): X-coord of each topk keypoint.
    """
    batch, _, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
    topk_clses = topk_inds // (height * width)
    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds // width
    topk_xs = (topk_inds % width).int().float()
    return (topk_scores, topk_inds, topk_clses, topk_ys, topk_xs)

@HEADS.register_module()
class CornerHead(BaseDenseHead, BBoxTestMixin):
    """Head of CornerNet: Detecting Objects as Paired Keypoints.

    Code is modified from the `official github repo
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/
    kp.py#L73>`_ .

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_ .

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        num_feat_levels (int): Levels of feature from the previous module. 2
            for HourglassNet-104 and 1 for HourglassNet-52. Because
            HourglassNet-104 outputs the final feature and intermediate
            supervision feature and HourglassNet-52 only outputs the final
            feature. Default: 2.
        corner_emb_channels (int): Channel of embedding vector. Default: 1.
        train_cfg (dict | None): Training config. Useless in CornerHead,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CornerHead. Default: None.
        loss_heatmap (dict | None): Config of corner heatmap loss. Default:
            GaussianFocalLoss.
        loss_embedding (dict | None): Config of corner embedding loss. Default:
            AssociativeEmbeddingLoss.
        loss_offset (dict | None): Config of corner offset loss. Default:
            SmoothL1Loss.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self, num_classes, in_channels, num_feat_levels=2, corner_emb_channels=1, train_cfg=None, test_cfg=None, loss_heatmap=dict(type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1), loss_embedding=dict(type='AssociativeEmbeddingLoss', pull_weight=0.25, push_weight=0.25), loss_offset=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1), init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization behavior, init_cfg is not allowed to be set'
        super(CornerHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.corner_emb_channels = corner_emb_channels
        self.with_corner_emb = self.corner_emb_channels > 0
        self.corner_offset_channels = 2
        self.num_feat_levels = num_feat_levels
        self.loss_heatmap = build_loss(loss_heatmap) if loss_heatmap is not None else None
        self.loss_embedding = build_loss(loss_embedding) if loss_embedding is not None else None
        self.loss_offset = build_loss(loss_offset) if loss_offset is not None else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self._init_layers()

    def _make_layers(self, out_channels, in_channels=256, feat_channels=256):
        """Initialize conv sequential for CornerHead."""
        return nn.Sequential(ConvModule(in_channels, feat_channels, 3, padding=1), ConvModule(feat_channels, out_channels, 1, norm_cfg=None, act_cfg=None))

    def _init_corner_kpt_layers(self):
        """Initialize corner keypoint layers.

        Including corner heatmap branch and corner offset branch. Each branch
        has two parts: prefix `tl_` for top-left and `br_` for bottom-right.
        """
        self.tl_pool, self.br_pool = (nn.ModuleList(), nn.ModuleList())
        self.tl_heat, self.br_heat = (nn.ModuleList(), nn.ModuleList())
        self.tl_off, self.br_off = (nn.ModuleList(), nn.ModuleList())
        for _ in range(self.num_feat_levels):
            self.tl_pool.append(BiCornerPool(self.in_channels, ['top', 'left'], out_channels=self.in_channels))
            self.br_pool.append(BiCornerPool(self.in_channels, ['bottom', 'right'], out_channels=self.in_channels))
            self.tl_heat.append(self._make_layers(out_channels=self.num_classes, in_channels=self.in_channels))
            self.br_heat.append(self._make_layers(out_channels=self.num_classes, in_channels=self.in_channels))
            self.tl_off.append(self._make_layers(out_channels=self.corner_offset_channels, in_channels=self.in_channels))
            self.br_off.append(self._make_layers(out_channels=self.corner_offset_channels, in_channels=self.in_channels))

    def _init_corner_emb_layers(self):
        """Initialize corner embedding layers.

        Only include corner embedding branch with two parts: prefix `tl_` for
        top-left and `br_` for bottom-right.
        """
        self.tl_emb, self.br_emb = (nn.ModuleList(), nn.ModuleList())
        for _ in range(self.num_feat_levels):
            self.tl_emb.append(self._make_layers(out_channels=self.corner_emb_channels, in_channels=self.in_channels))
            self.br_emb.append(self._make_layers(out_channels=self.corner_emb_channels, in_channels=self.in_channels))

    def _init_layers(self):
        """Initialize layers for CornerHead.

        Including two parts: corner keypoint layers and corner embedding layers
        """
        self._init_corner_kpt_layers()
        if self.with_corner_emb:
            self._init_corner_emb_layers()

    def init_weights(self):
        super(CornerHead, self).init_weights()
        bias_init = bias_init_with_prob(0.1)
        for i in range(self.num_feat_levels):
            self.tl_heat[i][-1].conv.reset_parameters()
            self.tl_heat[i][-1].conv.bias.data.fill_(bias_init)
            self.br_heat[i][-1].conv.reset_parameters()
            self.br_heat[i][-1].conv.bias.data.fill_(bias_init)
            self.tl_off[i][-1].conv.reset_parameters()
            self.br_off[i][-1].conv.reset_parameters()
            if self.with_corner_emb:
                self.tl_emb[i][-1].conv.reset_parameters()
                self.br_emb[i][-1].conv.reset_parameters()

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of corner heatmaps, offset heatmaps and
            embedding heatmaps.
                - tl_heats (list[Tensor]): Top-left corner heatmaps for all
                  levels, each is a 4D-tensor, the channels number is
                  num_classes.
                - br_heats (list[Tensor]): Bottom-right corner heatmaps for all
                  levels, each is a 4D-tensor, the channels number is
                  num_classes.
                - tl_embs (list[Tensor] | list[None]): Top-left embedding
                  heatmaps for all levels, each is a 4D-tensor or None.
                  If not None, the channels number is corner_emb_channels.
                - br_embs (list[Tensor] | list[None]): Bottom-right embedding
                  heatmaps for all levels, each is a 4D-tensor or None.
                  If not None, the channels number is corner_emb_channels.
                - tl_offs (list[Tensor]): Top-left offset heatmaps for all
                  levels, each is a 4D-tensor. The channels number is
                  corner_offset_channels.
                - br_offs (list[Tensor]): Bottom-right offset heatmaps for all
                  levels, each is a 4D-tensor. The channels number is
                  corner_offset_channels.
        """
        lvl_ind = list(range(self.num_feat_levels))
        return multi_apply(self.forward_single, feats, lvl_ind)

    def forward_single(self, x, lvl_ind, return_pool=False):
        """Forward feature of a single level.

        Args:
            x (Tensor): Feature of a single level.
            lvl_ind (int): Level index of current feature.
            return_pool (bool): Return corner pool feature or not.

        Returns:
            tuple[Tensor]: A tuple of CornerHead's output for current feature
            level. Containing the following Tensors:

                - tl_heat (Tensor): Predicted top-left corner heatmap.
                - br_heat (Tensor): Predicted bottom-right corner heatmap.
                - tl_emb (Tensor | None): Predicted top-left embedding heatmap.
                  None for `self.with_corner_emb == False`.
                - br_emb (Tensor | None): Predicted bottom-right embedding
                  heatmap. None for `self.with_corner_emb == False`.
                - tl_off (Tensor): Predicted top-left offset heatmap.
                - br_off (Tensor): Predicted bottom-right offset heatmap.
                - tl_pool (Tensor): Top-left corner pool feature. Not must
                  have.
                - br_pool (Tensor): Bottom-right corner pool feature. Not must
                  have.
        """
        tl_pool = self.tl_pool[lvl_ind](x)
        tl_heat = self.tl_heat[lvl_ind](tl_pool)
        br_pool = self.br_pool[lvl_ind](x)
        br_heat = self.br_heat[lvl_ind](br_pool)
        tl_emb, br_emb = (None, None)
        if self.with_corner_emb:
            tl_emb = self.tl_emb[lvl_ind](tl_pool)
            br_emb = self.br_emb[lvl_ind](br_pool)
        tl_off = self.tl_off[lvl_ind](tl_pool)
        br_off = self.br_off[lvl_ind](br_pool)
        result_list = [tl_heat, br_heat, tl_emb, br_emb, tl_off, br_off]
        if return_pool:
            result_list.append(tl_pool)
            result_list.append(br_pool)
        return result_list

    def get_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape, with_corner_emb=False, with_guiding_shift=False, with_centripetal_shift=False):
        """Generate corner targets.

        Including corner heatmap, corner offset.

        Optional: corner embedding, corner guiding shift, centripetal shift.

        For CornerNet, we generate corner heatmap, corner offset and corner
        embedding from this function.

        For CentripetalNet, we generate corner heatmap, corner offset, guiding
        shift and centripetal shift from this function.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image, each
                has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box, each has
                shape (num_gt,).
            feat_shape (list[int]): Shape of output feature,
                [batch, channel, height, width].
            img_shape (list[int]): Shape of input image,
                [height, width, channel].
            with_corner_emb (bool): Generate corner embedding target or not.
                Default: False.
            with_guiding_shift (bool): Generate guiding shift target or not.
                Default: False.
            with_centripetal_shift (bool): Generate centripetal shift target or
                not. Default: False.

        Returns:
            dict: Ground truth of corner heatmap, corner offset, corner
            embedding, guiding shift and centripetal shift. Containing the
            following keys:

                - topleft_heatmap (Tensor): Ground truth top-left corner
                  heatmap.
                - bottomright_heatmap (Tensor): Ground truth bottom-right
                  corner heatmap.
                - topleft_offset (Tensor): Ground truth top-left corner offset.
                - bottomright_offset (Tensor): Ground truth bottom-right corner
                  offset.
                - corner_embedding (list[list[list[int]]]): Ground truth corner
                  embedding. Not must have.
                - topleft_guiding_shift (Tensor): Ground truth top-left corner
                  guiding shift. Not must have.
                - bottomright_guiding_shift (Tensor): Ground truth bottom-right
                  corner guiding shift. Not must have.
                - topleft_centripetal_shift (Tensor): Ground truth top-left
                  corner centripetal shift. Not must have.
                - bottomright_centripetal_shift (Tensor): Ground truth
                  bottom-right corner centripetal shift. Not must have.
        """
        batch_size, _, height, width = feat_shape
        img_h, img_w = img_shape[:2]
        width_ratio = float(width / img_w)
        height_ratio = float(height / img_h)
        gt_tl_heatmap = gt_bboxes[-1].new_zeros([batch_size, self.num_classes, height, width])
        gt_br_heatmap = gt_bboxes[-1].new_zeros([batch_size, self.num_classes, height, width])
        gt_tl_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
        gt_br_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
        if with_corner_emb:
            match = []
        if with_guiding_shift:
            gt_tl_guiding_shift = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
            gt_br_guiding_shift = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
        if with_centripetal_shift:
            gt_tl_centripetal_shift = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
            gt_br_centripetal_shift = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
        for batch_id in range(batch_size):
            corner_match = []
            for box_id in range(len(gt_labels[batch_id])):
                left, top, right, bottom = gt_bboxes[batch_id][box_id]
                center_x = (left + right) / 2.0
                center_y = (top + bottom) / 2.0
                label = gt_labels[batch_id][box_id]
                scale_left = left * width_ratio
                scale_right = right * width_ratio
                scale_top = top * height_ratio
                scale_bottom = bottom * height_ratio
                scale_center_x = center_x * width_ratio
                scale_center_y = center_y * height_ratio
                left_idx = int(min(scale_left, width - 1))
                right_idx = int(min(scale_right, width - 1))
                top_idx = int(min(scale_top, height - 1))
                bottom_idx = int(min(scale_bottom, height - 1))
                scale_box_width = ceil(scale_right - scale_left)
                scale_box_height = ceil(scale_bottom - scale_top)
                radius = gaussian_radius((scale_box_height, scale_box_width), min_overlap=0.3)
                radius = max(0, int(radius))
                gt_tl_heatmap[batch_id, label] = gen_gaussian_target(gt_tl_heatmap[batch_id, label], [left_idx, top_idx], radius)
                gt_br_heatmap[batch_id, label] = gen_gaussian_target(gt_br_heatmap[batch_id, label], [right_idx, bottom_idx], radius)
                left_offset = scale_left - left_idx
                top_offset = scale_top - top_idx
                right_offset = scale_right - right_idx
                bottom_offset = scale_bottom - bottom_idx
                gt_tl_offset[batch_id, 0, top_idx, left_idx] = left_offset
                gt_tl_offset[batch_id, 1, top_idx, left_idx] = top_offset
                gt_br_offset[batch_id, 0, bottom_idx, right_idx] = right_offset
                gt_br_offset[batch_id, 1, bottom_idx, right_idx] = bottom_offset
                if with_corner_emb:
                    corner_match.append([[top_idx, left_idx], [bottom_idx, right_idx]])
                if with_guiding_shift:
                    gt_tl_guiding_shift[batch_id, 0, top_idx, left_idx] = scale_center_x - left_idx
                    gt_tl_guiding_shift[batch_id, 1, top_idx, left_idx] = scale_center_y - top_idx
                    gt_br_guiding_shift[batch_id, 0, bottom_idx, right_idx] = right_idx - scale_center_x
                    gt_br_guiding_shift[batch_id, 1, bottom_idx, right_idx] = bottom_idx - scale_center_y
                if with_centripetal_shift:
                    gt_tl_centripetal_shift[batch_id, 0, top_idx, left_idx] = log(scale_center_x - scale_left)
                    gt_tl_centripetal_shift[batch_id, 1, top_idx, left_idx] = log(scale_center_y - scale_top)
                    gt_br_centripetal_shift[batch_id, 0, bottom_idx, right_idx] = log(scale_right - scale_center_x)
                    gt_br_centripetal_shift[batch_id, 1, bottom_idx, right_idx] = log(scale_bottom - scale_center_y)
            if with_corner_emb:
                match.append(corner_match)
        target_result = dict(topleft_heatmap=gt_tl_heatmap, topleft_offset=gt_tl_offset, bottomright_heatmap=gt_br_heatmap, bottomright_offset=gt_br_offset)
        if with_corner_emb:
            target_result.update(corner_embedding=match)
        if with_guiding_shift:
            target_result.update(topleft_guiding_shift=gt_tl_guiding_shift, bottomright_guiding_shift=gt_br_guiding_shift)
        if with_centripetal_shift:
            target_result.update(topleft_centripetal_shift=gt_tl_centripetal_shift, bottomright_centripetal_shift=gt_br_centripetal_shift)
        return target_result

    @force_fp32()
    def loss(self, tl_heats, br_heats, tl_embs, br_embs, tl_offs, br_offs, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            tl_heats (list[Tensor]): Top-left corner heatmaps for each level
                with shape (N, num_classes, H, W).
            br_heats (list[Tensor]): Bottom-right corner heatmaps for each
                level with shape (N, num_classes, H, W).
            tl_embs (list[Tensor]): Top-left corner embeddings for each level
                with shape (N, corner_emb_channels, H, W).
            br_embs (list[Tensor]): Bottom-right corner embeddings for each
                level with shape (N, corner_emb_channels, H, W).
            tl_offs (list[Tensor]): Top-left corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            br_offs (list[Tensor]): Bottom-right corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [left, top, right, bottom] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components. Containing the
            following losses:

                - det_loss (list[Tensor]): Corner keypoint losses of all
                  feature levels.
                - pull_loss (list[Tensor]): Part one of AssociativeEmbedding
                  losses of all feature levels.
                - push_loss (list[Tensor]): Part two of AssociativeEmbedding
                  losses of all feature levels.
                - off_loss (list[Tensor]): Corner offset losses of all feature
                  levels.
        """
        targets = self.get_targets(gt_bboxes, gt_labels, tl_heats[-1].shape, img_metas[0]['pad_shape'], with_corner_emb=self.with_corner_emb)
        mlvl_targets = [targets for _ in range(self.num_feat_levels)]
        det_losses, pull_losses, push_losses, off_losses = multi_apply(self.loss_single, tl_heats, br_heats, tl_embs, br_embs, tl_offs, br_offs, mlvl_targets)
        loss_dict = dict(det_loss=det_losses, off_loss=off_losses)
        if self.with_corner_emb:
            loss_dict.update(pull_loss=pull_losses, push_loss=push_losses)
        return loss_dict

    def loss_single(self, tl_hmp, br_hmp, tl_emb, br_emb, tl_off, br_off, targets):
        """Compute losses for single level.

        Args:
            tl_hmp (Tensor): Top-left corner heatmap for current level with
                shape (N, num_classes, H, W).
            br_hmp (Tensor): Bottom-right corner heatmap for current level with
                shape (N, num_classes, H, W).
            tl_emb (Tensor): Top-left corner embedding for current level with
                shape (N, corner_emb_channels, H, W).
            br_emb (Tensor): Bottom-right corner embedding for current level
                with shape (N, corner_emb_channels, H, W).
            tl_off (Tensor): Top-left corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            br_off (Tensor): Bottom-right corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            targets (dict): Corner target generated by `get_targets`.

        Returns:
            tuple[torch.Tensor]: Losses of the head's different branches
            containing the following losses:

                - det_loss (Tensor): Corner keypoint loss.
                - pull_loss (Tensor): Part one of AssociativeEmbedding loss.
                - push_loss (Tensor): Part two of AssociativeEmbedding loss.
                - off_loss (Tensor): Corner offset loss.
        """
        gt_tl_hmp = targets['topleft_heatmap']
        gt_br_hmp = targets['bottomright_heatmap']
        gt_tl_off = targets['topleft_offset']
        gt_br_off = targets['bottomright_offset']
        gt_embedding = targets['corner_embedding']
        tl_det_loss = self.loss_heatmap(tl_hmp.sigmoid(), gt_tl_hmp, avg_factor=max(1, gt_tl_hmp.eq(1).sum()))
        br_det_loss = self.loss_heatmap(br_hmp.sigmoid(), gt_br_hmp, avg_factor=max(1, gt_br_hmp.eq(1).sum()))
        det_loss = (tl_det_loss + br_det_loss) / 2.0
        if self.with_corner_emb and self.loss_embedding is not None:
            pull_loss, push_loss = self.loss_embedding(tl_emb, br_emb, gt_embedding)
        else:
            pull_loss, push_loss = (None, None)
        tl_off_mask = gt_tl_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(gt_tl_hmp)
        br_off_mask = gt_br_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(gt_br_hmp)
        tl_off_loss = self.loss_offset(tl_off, gt_tl_off, tl_off_mask, avg_factor=max(1, tl_off_mask.sum()))
        br_off_loss = self.loss_offset(br_off, gt_br_off, br_off_mask, avg_factor=max(1, br_off_mask.sum()))
        off_loss = (tl_off_loss + br_off_loss) / 2.0
        return (det_loss, pull_loss, push_loss, off_loss)

    @force_fp32()
    def get_bboxes(self, tl_heats, br_heats, tl_embs, br_embs, tl_offs, br_offs, img_metas, rescale=False, with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            tl_heats (list[Tensor]): Top-left corner heatmaps for each level
                with shape (N, num_classes, H, W).
            br_heats (list[Tensor]): Bottom-right corner heatmaps for each
                level with shape (N, num_classes, H, W).
            tl_embs (list[Tensor]): Top-left corner embeddings for each level
                with shape (N, corner_emb_channels, H, W).
            br_embs (list[Tensor]): Bottom-right corner embeddings for each
                level with shape (N, corner_emb_channels, H, W).
            tl_offs (list[Tensor]): Top-left corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            br_offs (list[Tensor]): Bottom-right corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        """
        assert tl_heats[-1].shape[0] == br_heats[-1].shape[0] == len(img_metas)
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(self._get_bboxes_single(tl_heats[-1][img_id:img_id + 1, :], br_heats[-1][img_id:img_id + 1, :], tl_offs[-1][img_id:img_id + 1, :], br_offs[-1][img_id:img_id + 1, :], img_metas[img_id], tl_emb=tl_embs[-1][img_id:img_id + 1, :], br_emb=br_embs[-1][img_id:img_id + 1, :], rescale=rescale, with_nms=with_nms))
        return result_list

    def _get_bboxes_single(self, tl_heat, br_heat, tl_off, br_off, img_meta, tl_emb=None, br_emb=None, tl_centripetal_shift=None, br_centripetal_shift=None, rescale=False, with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            tl_heat (Tensor): Top-left corner heatmap for current level with
                shape (N, num_classes, H, W).
            br_heat (Tensor): Bottom-right corner heatmap for current level
                with shape (N, num_classes, H, W).
            tl_off (Tensor): Top-left corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            br_off (Tensor): Bottom-right corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            tl_emb (Tensor): Top-left corner embedding for current level with
                shape (N, corner_emb_channels, H, W).
            br_emb (Tensor): Bottom-right corner embedding for current level
                with shape (N, corner_emb_channels, H, W).
            tl_centripetal_shift: Top-left corner's centripetal shift for
                current level with shape (N, 2, H, W).
            br_centripetal_shift: Bottom-right corner's centripetal shift for
                current level with shape (N, 2, H, W).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        """
        if isinstance(img_meta, (list, tuple)):
            img_meta = img_meta[0]
        batch_bboxes, batch_scores, batch_clses = self.decode_heatmap(tl_heat=tl_heat.sigmoid(), br_heat=br_heat.sigmoid(), tl_off=tl_off, br_off=br_off, tl_emb=tl_emb, br_emb=br_emb, tl_centripetal_shift=tl_centripetal_shift, br_centripetal_shift=br_centripetal_shift, img_meta=img_meta, k=self.test_cfg.corner_topk, kernel=self.test_cfg.local_maximum_kernel, distance_threshold=self.test_cfg.distance_threshold)
        if rescale:
            batch_bboxes /= batch_bboxes.new_tensor(img_meta['scale_factor'])
        bboxes = batch_bboxes.view([-1, 4])
        scores = batch_scores.view(-1)
        clses = batch_clses.view(-1)
        detections = torch.cat([bboxes, scores.unsqueeze(-1)], -1)
        keepinds = detections[:, -1] > -0.1
        detections = detections[keepinds]
        labels = clses[keepinds]
        if with_nms:
            detections, labels = self._bboxes_nms(detections, labels, self.test_cfg)
        return (detections, labels)

    def _bboxes_nms(self, bboxes, labels, cfg):
        if 'nms_cfg' in cfg:
            warning.warn('nms_cfg in test_cfg will be deprecated. Please rename it as nms')
        if 'nms' not in cfg:
            cfg.nms = cfg.nms_cfg
        if labels.numel() > 0:
            max_num = cfg.max_per_img
            bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1].contiguous(), labels, cfg.nms)
            if max_num > 0:
                bboxes = bboxes[:max_num]
                labels = labels[keep][:max_num]
        return (bboxes, labels)

    def decode_heatmap(self, tl_heat, br_heat, tl_off, br_off, tl_emb=None, br_emb=None, tl_centripetal_shift=None, br_centripetal_shift=None, img_meta=None, k=100, kernel=3, distance_threshold=0.5, num_dets=1000):
        """Transform outputs for a single batch item into raw bbox predictions.

        Args:
            tl_heat (Tensor): Top-left corner heatmap for current level with
                shape (N, num_classes, H, W).
            br_heat (Tensor): Bottom-right corner heatmap for current level
                with shape (N, num_classes, H, W).
            tl_off (Tensor): Top-left corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            br_off (Tensor): Bottom-right corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            tl_emb (Tensor | None): Top-left corner embedding for current
                level with shape (N, corner_emb_channels, H, W).
            br_emb (Tensor | None): Bottom-right corner embedding for current
                level with shape (N, corner_emb_channels, H, W).
            tl_centripetal_shift (Tensor | None): Top-left centripetal shift
                for current level with shape (N, 2, H, W).
            br_centripetal_shift (Tensor | None): Bottom-right centripetal
                shift for current level with shape (N, 2, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            k (int): Get top k corner keypoints from heatmap.
            kernel (int): Max pooling kernel for extract local maximum pixels.
            distance_threshold (float): Distance threshold. Top-left and
                bottom-right corner keypoints with feature distance less than
                the threshold will be regarded as keypoints from same object.
            num_dets (int): Num of raw boxes before doing nms.

        Returns:
            tuple[torch.Tensor]: Decoded output of CornerHead, containing the
            following Tensors:

            - bboxes (Tensor): Coords of each box.
            - scores (Tensor): Scores of each box.
            - clses (Tensor): Categories of each box.
        """
        with_embedding = tl_emb is not None and br_emb is not None
        with_centripetal_shift = tl_centripetal_shift is not None and br_centripetal_shift is not None
        assert with_embedding + with_centripetal_shift == 1
        batch, _, height, width = tl_heat.size()
        if torch.onnx.is_in_onnx_export():
            inp_h, inp_w = img_meta['pad_shape_for_onnx'][:2]
        else:
            inp_h, inp_w, _ = img_meta['pad_shape']
        tl_heat = get_local_maximum(tl_heat, kernel=kernel)
        br_heat = get_local_maximum(br_heat, kernel=kernel)
        tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = get_topk_from_heatmap(tl_heat, k=k)
        br_scores, br_inds, br_clses, br_ys, br_xs = get_topk_from_heatmap(br_heat, k=k)
        tl_ys = tl_ys.view(batch, k, 1).repeat(1, 1, k)
        tl_xs = tl_xs.view(batch, k, 1).repeat(1, 1, k)
        br_ys = br_ys.view(batch, 1, k).repeat(1, k, 1)
        br_xs = br_xs.view(batch, 1, k).repeat(1, k, 1)
        tl_off = transpose_and_gather_feat(tl_off, tl_inds)
        tl_off = tl_off.view(batch, k, 1, 2)
        br_off = transpose_and_gather_feat(br_off, br_inds)
        br_off = br_off.view(batch, 1, k, 2)
        tl_xs = tl_xs + tl_off[..., 0]
        tl_ys = tl_ys + tl_off[..., 1]
        br_xs = br_xs + br_off[..., 0]
        br_ys = br_ys + br_off[..., 1]
        if with_centripetal_shift:
            tl_centripetal_shift = transpose_and_gather_feat(tl_centripetal_shift, tl_inds).view(batch, k, 1, 2).exp()
            br_centripetal_shift = transpose_and_gather_feat(br_centripetal_shift, br_inds).view(batch, 1, k, 2).exp()
            tl_ctxs = tl_xs + tl_centripetal_shift[..., 0]
            tl_ctys = tl_ys + tl_centripetal_shift[..., 1]
            br_ctxs = br_xs - br_centripetal_shift[..., 0]
            br_ctys = br_ys - br_centripetal_shift[..., 1]
        tl_xs *= inp_w / width
        tl_ys *= inp_h / height
        br_xs *= inp_w / width
        br_ys *= inp_h / height
        if with_centripetal_shift:
            tl_ctxs *= inp_w / width
            tl_ctys *= inp_h / height
            br_ctxs *= inp_w / width
            br_ctys *= inp_h / height
        x_off, y_off = (0, 0)
        if not torch.onnx.is_in_onnx_export():
            if 'border' in img_meta:
                x_off = img_meta['border'][2]
                y_off = img_meta['border'][0]
        tl_xs -= x_off
        tl_ys -= y_off
        br_xs -= x_off
        br_ys -= y_off
        zeros = tl_xs.new_zeros(*tl_xs.size())
        tl_xs = torch.where(tl_xs > 0.0, tl_xs, zeros)
        tl_ys = torch.where(tl_ys > 0.0, tl_ys, zeros)
        br_xs = torch.where(br_xs > 0.0, br_xs, zeros)
        br_ys = torch.where(br_ys > 0.0, br_ys, zeros)
        bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)
        area_bboxes = ((br_xs - tl_xs) * (br_ys - tl_ys)).abs()
        if with_centripetal_shift:
            tl_ctxs -= x_off
            tl_ctys -= y_off
            br_ctxs -= x_off
            br_ctys -= y_off
            tl_ctxs *= tl_ctxs.gt(0.0).type_as(tl_ctxs)
            tl_ctys *= tl_ctys.gt(0.0).type_as(tl_ctys)
            br_ctxs *= br_ctxs.gt(0.0).type_as(br_ctxs)
            br_ctys *= br_ctys.gt(0.0).type_as(br_ctys)
            ct_bboxes = torch.stack((tl_ctxs, tl_ctys, br_ctxs, br_ctys), dim=3)
            area_ct_bboxes = ((br_ctxs - tl_ctxs) * (br_ctys - tl_ctys)).abs()
            rcentral = torch.zeros_like(ct_bboxes)
            mu = torch.ones_like(area_bboxes) / 2.4
            mu[area_bboxes > 3500] = 1 / 2.1
            bboxes_center_x = (bboxes[..., 0] + bboxes[..., 2]) / 2
            bboxes_center_y = (bboxes[..., 1] + bboxes[..., 3]) / 2
            rcentral[..., 0] = bboxes_center_x - mu * (bboxes[..., 2] - bboxes[..., 0]) / 2
            rcentral[..., 1] = bboxes_center_y - mu * (bboxes[..., 3] - bboxes[..., 1]) / 2
            rcentral[..., 2] = bboxes_center_x + mu * (bboxes[..., 2] - bboxes[..., 0]) / 2
            rcentral[..., 3] = bboxes_center_y + mu * (bboxes[..., 3] - bboxes[..., 1]) / 2
            area_rcentral = ((rcentral[..., 2] - rcentral[..., 0]) * (rcentral[..., 3] - rcentral[..., 1])).abs()
            dists = area_ct_bboxes / area_rcentral
            tl_ctx_inds = (ct_bboxes[..., 0] <= rcentral[..., 0]) | (ct_bboxes[..., 0] >= rcentral[..., 2])
            tl_cty_inds = (ct_bboxes[..., 1] <= rcentral[..., 1]) | (ct_bboxes[..., 1] >= rcentral[..., 3])
            br_ctx_inds = (ct_bboxes[..., 2] <= rcentral[..., 0]) | (ct_bboxes[..., 2] >= rcentral[..., 2])
            br_cty_inds = (ct_bboxes[..., 3] <= rcentral[..., 1]) | (ct_bboxes[..., 3] >= rcentral[..., 3])
        if with_embedding:
            tl_emb = transpose_and_gather_feat(tl_emb, tl_inds)
            tl_emb = tl_emb.view(batch, k, 1)
            br_emb = transpose_and_gather_feat(br_emb, br_inds)
            br_emb = br_emb.view(batch, 1, k)
            dists = torch.abs(tl_emb - br_emb)
        tl_scores = tl_scores.view(batch, k, 1).repeat(1, 1, k)
        br_scores = br_scores.view(batch, 1, k).repeat(1, k, 1)
        scores = (tl_scores + br_scores) / 2
        tl_clses = tl_clses.view(batch, k, 1).repeat(1, 1, k)
        br_clses = br_clses.view(batch, 1, k).repeat(1, k, 1)
        cls_inds = tl_clses != br_clses
        dist_inds = dists > distance_threshold
        width_inds = br_xs <= tl_xs
        height_inds = br_ys <= tl_ys
        negative_scores = -1 * torch.ones_like(scores)
        scores = torch.where(cls_inds, negative_scores, scores)
        scores = torch.where(width_inds, negative_scores, scores)
        scores = torch.where(height_inds, negative_scores, scores)
        scores = torch.where(dist_inds, negative_scores, scores)
        if with_centripetal_shift:
            scores[tl_ctx_inds] = -1
            scores[tl_cty_inds] = -1
            scores[br_ctx_inds] = -1
            scores[br_cty_inds] = -1
        scores = scores.view(batch, -1)
        scores, inds = torch.topk(scores, num_dets)
        scores = scores.unsqueeze(2)
        bboxes = bboxes.view(batch, -1, 4)
        bboxes = gather_feat(bboxes, inds)
        clses = tl_clses.contiguous().view(batch, -1, 1)
        clses = gather_feat(clses, inds).float()
        return (bboxes, scores, clses)

    def onnx_export(self, tl_heats, br_heats, tl_embs, br_embs, tl_offs, br_offs, img_metas, rescale=False, with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            tl_heats (list[Tensor]): Top-left corner heatmaps for each level
                with shape (N, num_classes, H, W).
            br_heats (list[Tensor]): Bottom-right corner heatmaps for each
                level with shape (N, num_classes, H, W).
            tl_embs (list[Tensor]): Top-left corner embeddings for each level
                with shape (N, corner_emb_channels, H, W).
            br_embs (list[Tensor]): Bottom-right corner embeddings for each
                level with shape (N, corner_emb_channels, H, W).
            tl_offs (list[Tensor]): Top-left corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            br_offs (list[Tensor]): Bottom-right corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor, Tensor]: First tensor bboxes with shape
            [N, num_det, 5], 5 arrange as (x1, y1, x2, y2, score)
            and second element is class labels of shape [N, num_det].
        """
        assert tl_heats[-1].shape[0] == br_heats[-1].shape[0] == len(img_metas) == 1
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(self._get_bboxes_single(tl_heats[-1][img_id:img_id + 1, :], br_heats[-1][img_id:img_id + 1, :], tl_offs[-1][img_id:img_id + 1, :], br_offs[-1][img_id:img_id + 1, :], img_metas[img_id], tl_emb=tl_embs[-1][img_id:img_id + 1, :], br_emb=br_embs[-1][img_id:img_id + 1, :], rescale=rescale, with_nms=with_nms))
        detections, labels = result_list[0]
        return (detections.unsqueeze(0), labels.unsqueeze(0))

