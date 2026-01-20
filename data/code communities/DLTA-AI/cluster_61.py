# Cluster 61

def get_uncertain_point_coords_with_randomness(mask_pred, labels, num_points, oversample_ratio, importance_sample_ratio):
    """Get ``num_points`` most uncertain points with random points during
    train.

    Sample points in [0, 1] x [0, 1] coordinate space based on their
    uncertainty. The uncertainties are calculated for each point using
    'get_uncertainty()' function that takes point's logit prediction as
    input.

    Args:
        mask_pred (Tensor): A tensor of shape (num_rois, num_classes,
            mask_height, mask_width) for class-specific or class-agnostic
            prediction.
        labels (list): The ground truth class for each instance.
        num_points (int): The number of points to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled
            via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (num_rois, num_points, 2)
            that contains the coordinates sampled points.
    """
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    batch_size = mask_pred.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(batch_size, num_sampled, 2, device=mask_pred.device)
    point_logits = point_sample(mask_pred, point_coords)
    point_uncertainties = get_uncertainty(point_logits, labels)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(batch_size, dtype=torch.long, device=mask_pred.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(batch_size, num_uncertain_points, 2)
    if num_random_points > 0:
        rand_roi_coords = torch.rand(batch_size, num_random_points, 2, device=mask_pred.device)
        point_coords = torch.cat((point_coords, rand_roi_coords), dim=1)
    return point_coords

def get_uncertainty(mask_pred, labels):
    """Estimate uncertainty based on pred logits.

    We estimate uncertainty as L1 distance between 0.0 and the logits
    prediction in 'mask_pred' for the foreground class in `classes`.

    Args:
        mask_pred (Tensor): mask predication logits, shape (num_rois,
            num_classes, mask_height, mask_width).

        labels (list[Tensor]): Either predicted or ground truth label for
            each predicted mask, of length num_rois.

    Returns:
        scores (Tensor): Uncertainty scores with the most uncertain
            locations having the highest uncertainty score,
            shape (num_rois, 1, mask_height, mask_width)
    """
    if mask_pred.shape[1] == 1:
        gt_class_logits = mask_pred.clone()
    else:
        inds = torch.arange(mask_pred.shape[0], device=mask_pred.device)
        gt_class_logits = mask_pred[inds, labels].unsqueeze(1)
    return -torch.abs(gt_class_logits)

@HEADS.register_module()
class Mask2FormerHead(MaskFormerHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (:obj:`mmcv.ConfigDict` | dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`mmcv.ConfigDict` | dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`mmcv.ConfigDict` | dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`mmcv.ConfigDict` | dict): Training config of
            Mask2Former head.
        test_cfg (:obj:`mmcv.ConfigDict` | dict): Testing config of
            Mask2Former head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self, in_channels, feat_channels, out_channels, num_things_classes=80, num_stuff_classes=53, num_queries=100, num_transformer_feat_level=3, pixel_decoder=None, enforce_decoder_input_project=False, transformer_decoder=None, positional_encoding=None, loss_cls=None, loss_mask=None, loss_dice=None, train_cfg=None, test_cfg=None, init_cfg=None, **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.transformerlayers.attn_cfgs.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(in_channels=in_channels, feat_channels=feat_channels, out_channels=out_channels)
        self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]
        self.transformer_decoder = build_transformer_layer_sequence(transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        self.decoder_input_projs = ModuleList()
        for _ in range(num_transformer_feat_level):
            if self.decoder_embed_dims != feat_channels or enforce_decoder_input_project:
                self.decoder_input_projs.append(Conv2d(feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = build_positional_encoding(positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)
        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True), nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True), nn.Linear(feat_channels, out_channels))
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get('importance_sample_ratio', 0.75)
        self.class_weight = loss_cls.class_weight
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)
        self.pixel_decoder.init_weights()
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks, img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (num_gts, ).
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (num_gts, h, w).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image.                     shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image.                     shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image.                     shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image.                     shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each                     image.
                - neg_inds (Tensor): Sampled negative indices for each                     image.
        """
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]
        point_coords = torch.rand((1, self.num_points, 2), device=cls_score.device)
        mask_points_pred = point_sample(mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1, 1)).squeeze(1)
        gt_points_masks = point_sample(gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1, 1)).squeeze(1)
        assign_result = self.assigner.assign(cls_score, mask_points_pred, gt_labels, gt_points_masks, img_metas)
        sampling_result = self.sampler.sample(assign_result, mask_pred, gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        labels = gt_labels.new_full((self.num_queries,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries,))
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries,))
        mask_weights[pos_inds] = 1.0
        return (labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds)

    def loss_single(self, cls_scores, mask_preds, gt_labels_list, gt_masks_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (num_gts, ).
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (num_gts, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single                 decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        labels_list, label_weights_list, mask_targets_list, mask_weights_list, num_total_pos, num_total_neg = self.get_targets(cls_scores_list, mask_preds_list, gt_labels_list, gt_masks_list, img_metas)
        labels = torch.stack(labels_list, dim=0)
        label_weights = torch.stack(label_weights_list, dim=0)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        mask_weights = torch.stack(mask_weights_list, dim=0)
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)
        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=class_weight[labels].sum())
        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)
        mask_preds = mask_preds[mask_weights > 0]
        if mask_targets.shape[0] == 0:
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return (loss_cls, loss_mask, loss_dice)
        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(mask_preds.unsqueeze(1), None, self.num_points, self.oversample_ratio, self.importance_sample_ratio)
            mask_point_targets = point_sample(mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        mask_point_preds = point_sample(mask_preds.unsqueeze(1), points_coords).squeeze(1)
        loss_dice = self.loss_dice(mask_point_preds, mask_point_targets, avg_factor=num_total_masks)
        mask_point_preds = mask_point_preds.reshape(-1)
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(mask_point_preds, mask_point_targets, avg_factor=num_total_masks * self.num_points)
        return (loss_cls, loss_mask, loss_dice)

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape                 (batch_size, num_queries, cls_out_channels).                 Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape                 (batch_size, num_queries,h, w).
            - attn_mask (Tensor): Attention mask in shape                 (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        cls_pred = self.cls_embed(decoder_out)
        mask_embed = self.mask_embed(decoder_out)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(mask_pred, attn_mask_target_size, mode='bilinear', align_corners=False)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()
        return (cls_pred, mask_pred, attn_mask)

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits                 for each decoder layer. Each is a 3D-tensor with shape                 (batch_size, num_queries, cls_out_channels).                 Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each                 decoder layer. Each with shape (batch_size, num_queries,                  h, w).
        """
        batch_size = len(img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            mask = decoder_input.new_zeros((batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat((1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))
        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self.forward_head(query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(query=query_feat, key=decoder_inputs[level_idx], value=decoder_inputs[level_idx], query_pos=query_embed, key_pos=decoder_positional_encodings[level_idx], attn_masks=attn_masks, query_key_padding_mask=None, key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self.forward_head(query_feat, mask_features, multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:])
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
        return (cls_pred_list, mask_pred_list)

@HEADS.register_module()
class MaskPointHead(BaseModule):
    """A mask point head use in PointRend.

    ``MaskPointHead`` use shared multi-layer perceptron (equivalent to
    nn.Conv1d) to predict the logit of input points. The fine-grained feature
    and coarse feature will be concatenate together for predication.

    Args:
        num_fcs (int): Number of fc layers in the head. Default: 3.
        in_channels (int): Number of input channels. Default: 256.
        fc_channels (int): Number of fc channels. Default: 256.
        num_classes (int): Number of classes for logits. Default: 80.
        class_agnostic (bool): Whether use class agnostic classification.
            If so, the output channels of logits will be 1. Default: False.
        coarse_pred_each_layer (bool): Whether concatenate coarse feature with
            the output of each fc layer. Default: True.
        conv_cfg (dict | None): Dictionary to construct and config conv layer.
            Default: dict(type='Conv1d'))
        norm_cfg (dict | None): Dictionary to construct and config norm layer.
            Default: None.
        loss_point (dict): Dictionary to construct and config loss layer of
            point head. Default: dict(type='CrossEntropyLoss', use_mask=True,
            loss_weight=1.0).
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, num_classes, num_fcs=3, in_channels=256, fc_channels=256, class_agnostic=False, coarse_pred_each_layer=True, conv_cfg=dict(type='Conv1d'), norm_cfg=None, act_cfg=dict(type='ReLU'), loss_point=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0), init_cfg=dict(type='Normal', std=0.001, override=dict(name='fc_logits'))):
        super().__init__(init_cfg)
        self.num_fcs = num_fcs
        self.in_channels = in_channels
        self.fc_channels = fc_channels
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.coarse_pred_each_layer = coarse_pred_each_layer
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_point = build_loss(loss_point)
        fc_in_channels = in_channels + num_classes
        self.fcs = nn.ModuleList()
        for _ in range(num_fcs):
            fc = ConvModule(fc_in_channels, fc_channels, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.fcs.append(fc)
            fc_in_channels = fc_channels
            fc_in_channels += num_classes if self.coarse_pred_each_layer else 0
        out_channels = 1 if self.class_agnostic else self.num_classes
        self.fc_logits = nn.Conv1d(fc_in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, fine_grained_feats, coarse_feats):
        """Classify each point base on fine grained and coarse feats.

        Args:
            fine_grained_feats (Tensor): Fine grained feature sampled from FPN,
                shape (num_rois, in_channels, num_points).
            coarse_feats (Tensor): Coarse feature sampled from CoarseMaskHead,
                shape (num_rois, num_classes, num_points).

        Returns:
            Tensor: Point classification results,
                shape (num_rois, num_class, num_points).
        """
        x = torch.cat([fine_grained_feats, coarse_feats], dim=1)
        for fc in self.fcs:
            x = fc(x)
            if self.coarse_pred_each_layer:
                x = torch.cat((x, coarse_feats), dim=1)
        return self.fc_logits(x)

    def get_targets(self, rois, rel_roi_points, sampling_results, gt_masks, cfg):
        """Get training targets of MaskPointHead for all images.

        Args:
            rois (Tensor): Region of Interest, shape (num_rois, 5).
            rel_roi_points: Points coordinates relative to RoI, shape
                (num_rois, num_points, 2).
            sampling_results (:obj:`SamplingResult`): Sampling result after
                sampling and assignment.
            gt_masks (Tensor) : Ground truth segmentation masks of
                corresponding boxes, shape (num_rois, height, width).
            cfg (dict): Training cfg.

        Returns:
            Tensor: Point target, shape (num_rois, num_points).
        """
        num_imgs = len(sampling_results)
        rois_list = []
        rel_roi_points_list = []
        for batch_ind in range(num_imgs):
            inds = rois[:, 0] == batch_ind
            rois_list.append(rois[inds])
            rel_roi_points_list.append(rel_roi_points[inds])
        pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]
        cfg_list = [cfg for _ in range(num_imgs)]
        point_targets = map(self._get_target_single, rois_list, rel_roi_points_list, pos_assigned_gt_inds_list, gt_masks, cfg_list)
        point_targets = list(point_targets)
        if len(point_targets) > 0:
            point_targets = torch.cat(point_targets)
        return point_targets

    def _get_target_single(self, rois, rel_roi_points, pos_assigned_gt_inds, gt_masks, cfg):
        """Get training target of MaskPointHead for each image."""
        num_pos = rois.size(0)
        num_points = cfg.num_points
        if num_pos > 0:
            gt_masks_th = gt_masks.to_tensor(rois.dtype, rois.device).index_select(0, pos_assigned_gt_inds)
            gt_masks_th = gt_masks_th.unsqueeze(1)
            rel_img_points = rel_roi_point_to_rel_img_point(rois, rel_roi_points, gt_masks_th)
            point_targets = point_sample(gt_masks_th, rel_img_points).squeeze(1)
        else:
            point_targets = rois.new_zeros((0, num_points))
        return point_targets

    def loss(self, point_pred, point_targets, labels):
        """Calculate loss for MaskPointHead.

        Args:
            point_pred (Tensor): Point predication result, shape
                (num_rois, num_classes, num_points).
            point_targets (Tensor): Point targets, shape (num_roi, num_points).
            labels (Tensor): Class label of corresponding boxes,
                shape (num_rois, )

        Returns:
            dict[str, Tensor]: a dictionary of point loss components
        """
        loss = dict()
        if self.class_agnostic:
            loss_point = self.loss_point(point_pred, point_targets, torch.zeros_like(labels))
        else:
            loss_point = self.loss_point(point_pred, point_targets, labels)
        loss['loss_point'] = loss_point
        return loss

    def get_roi_rel_points_train(self, mask_pred, labels, cfg):
        """Get ``num_points`` most uncertain points with random points during
        train.

        Sample points in [0, 1] x [0, 1] coordinate space based on their
        uncertainty. The uncertainties are calculated for each point using
        '_get_uncertainty()' function that takes point's logit prediction as
        input.

        Args:
            mask_pred (Tensor): A tensor of shape (num_rois, num_classes,
                mask_height, mask_width) for class-specific or class-agnostic
                prediction.
            labels (list): The ground truth class for each instance.
            cfg (dict): Training config of point head.

        Returns:
            point_coords (Tensor): A tensor of shape (num_rois, num_points, 2)
                that contains the coordinates sampled points.
        """
        point_coords = get_uncertain_point_coords_with_randomness(mask_pred, labels, cfg.num_points, cfg.oversample_ratio, cfg.importance_sample_ratio)
        return point_coords

    def get_roi_rel_points_test(self, mask_pred, pred_label, cfg):
        """Get ``num_points`` most uncertain points during test.

        Args:
            mask_pred (Tensor): A tensor of shape (num_rois, num_classes,
                mask_height, mask_width) for class-specific or class-agnostic
                prediction.
            pred_label (list): The predication class for each instance.
            cfg (dict): Testing config of point head.

        Returns:
            point_indices (Tensor): A tensor of shape (num_rois, num_points)
                that contains indices from [0, mask_height x mask_width) of the
                most uncertain points.
            point_coords (Tensor): A tensor of shape (num_rois, num_points, 2)
                that contains [0, 1] x [0, 1] normalized coordinates of the
                most uncertain points from the [mask_height, mask_width] grid .
        """
        num_points = cfg.subdivision_num_points
        uncertainty_map = get_uncertainty(mask_pred, pred_label)
        num_rois, _, mask_height, mask_width = uncertainty_map.shape
        if isinstance(mask_height, torch.Tensor):
            h_step = 1.0 / mask_height.float()
            w_step = 1.0 / mask_width.float()
        else:
            h_step = 1.0 / mask_height
            w_step = 1.0 / mask_width
        mask_size = int(mask_height * mask_width)
        uncertainty_map = uncertainty_map.view(num_rois, mask_size)
        num_points = min(mask_size, num_points)
        point_indices = uncertainty_map.topk(num_points, dim=1)[1]
        xs = w_step / 2.0 + (point_indices % mask_width).float() * w_step
        ys = h_step / 2.0 + (point_indices // mask_width).float() * h_step
        point_coords = torch.stack([xs, ys], dim=2)
        return (point_indices, point_coords)

