# Cluster 12

class LateFusionBackbone(nn.Module):
    """
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """

    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=0):
        super().__init__()
        self.config = config
        if config.use_point_pillars == True:
            in_channels = config.num_features[-1]
        else:
            in_channels = 2 * config.lidar_seq_len
        if self.config.use_target_point_image == True:
            in_channels += 1
        self.image_encoder = ImageCNN(architecture=image_architecture, normalize=True)
        self.lidar_encoder = LidarEncoder(architecture=lidar_architecture, in_channels=in_channels)
        if image_architecture.startswith('convnext'):
            self.norm_after_pool_img = nn.LayerNorm((self.config.perception_output_features,), eps=1e-06)
        else:
            self.norm_after_pool_img = nn.Sequential()
        if lidar_architecture.startswith('convnext'):
            self.norm_after_pool_lidar = nn.LayerNorm((self.config.perception_output_features,), eps=1e-06)
        else:
            self.norm_after_pool_lidar = nn.Sequential()
        self.use_velocity = use_velocity
        if use_velocity:
            self.vel_emb = nn.Linear(1, self.config.perception_output_features)
        channel = self.config.bev_features_chanels
        self.relu = nn.ReLU(inplace=True)
        if self.image_encoder.features.num_features != self.config.perception_output_features:
            self.reduce_channels_conv_image = nn.Conv2d(self.image_encoder.features.num_features, self.config.perception_output_features, (1, 1))
        else:
            self.reduce_channels_conv_image = nn.Sequential()
        if self.image_encoder.features.num_features != self.config.perception_output_features:
            self.reduce_channels_conv_lidar = nn.Conv2d(self.lidar_encoder._model.num_features, self.config.perception_output_features, (1, 1))
        else:
            self.reduce_channels_conv_lidar = nn.Sequential()
        self.upsample = nn.Upsample(scale_factor=self.config.bev_upsample_factor, mode='bilinear', align_corners=False)
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))
        self.c5_conv = nn.Conv2d(self.config.perception_output_features, channel, (1, 1))

    def top_down(self, c5):
        p5 = self.relu(self.c5_conv(c5))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample(p4)))
        p2 = self.relu(self.up_conv3(self.upsample(p3)))
        return (p2, p3, p4, p5)

    def forward(self, image, lidar, velocity):
        """
        Image + LiDAR feature fusion
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
        """
        if self.image_encoder.normalize:
            image_tensor = normalize_imagenet(image)
        else:
            image_tensor = image
        output_features_image = self.image_encoder.features(image_tensor)
        output_features_image = self.reduce_channels_conv_image(output_features_image)
        image_features_grid = output_features_image
        image_features = torch.nn.AdaptiveAvgPool2d((1, 1))(output_features_image)
        image_features = torch.flatten(image_features, 1)
        image_features = self.norm_after_pool_img(image_features)
        output_features_lidar = self.lidar_encoder._model(lidar)
        output_features_lidar = self.reduce_channels_conv_lidar(output_features_lidar)
        lidar_features_grid = output_features_lidar
        features = self.top_down(lidar_features_grid)
        lidar_features = torch.nn.AdaptiveAvgPool2d((1, 1))(output_features_lidar)
        lidar_features = torch.flatten(lidar_features, 1)
        lidar_features = self.norm_after_pool_lidar(lidar_features)
        fused_features = torch.add(image_features, lidar_features)
        if self.use_velocity:
            velocity_embeddings = self.vel_emb(velocity)
            fused_features = torch.add(fused_features, velocity_embeddings)
        return (features, image_features_grid, fused_features)

def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0] / 255.0 - 0.485) / 0.229
    x[:, 1] = (x[:, 1] / 255.0 - 0.456) / 0.224
    x[:, 2] = (x[:, 2] / 255.0 - 0.406) / 0.225
    return x

class latentTFBackbone(nn.Module):
    """
    Multi-scale Fusion Transformer for image + pos_embedding feature fusion
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """

    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        super().__init__()
        self.config = config
        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((self.config.lidar_vert_anchors, self.config.lidar_horz_anchors))
        if config.use_point_pillars == True:
            in_channels = config.num_features[-1]
        else:
            in_channels = 2 * config.lidar_seq_len
        if self.config.use_target_point_image == True:
            in_channels += 1
        self.image_encoder = ImageCNN(architecture=image_architecture, normalize=True)
        self.lidar_encoder = LidarEncoder(architecture=lidar_architecture, in_channels=in_channels)
        self.transformer1 = GPT(n_embd=self.image_encoder.features.feature_info[1]['num_chs'], n_head=config.n_head, block_exp=config.block_exp, n_layer=config.n_layer, img_vert_anchors=config.img_vert_anchors, img_horz_anchors=config.img_horz_anchors, lidar_vert_anchors=config.lidar_vert_anchors, lidar_horz_anchors=config.lidar_horz_anchors, seq_len=config.seq_len, embd_pdrop=config.embd_pdrop, attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop, config=config, use_velocity=use_velocity)
        self.transformer2 = GPT(n_embd=self.image_encoder.features.feature_info[2]['num_chs'], n_head=config.n_head, block_exp=config.block_exp, n_layer=config.n_layer, img_vert_anchors=config.img_vert_anchors, img_horz_anchors=config.img_horz_anchors, lidar_vert_anchors=config.lidar_vert_anchors, lidar_horz_anchors=config.lidar_horz_anchors, seq_len=config.seq_len, embd_pdrop=config.embd_pdrop, attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop, config=config, use_velocity=use_velocity)
        self.transformer3 = GPT(n_embd=self.image_encoder.features.feature_info[3]['num_chs'], n_head=config.n_head, block_exp=config.block_exp, n_layer=config.n_layer, img_vert_anchors=config.img_vert_anchors, img_horz_anchors=config.img_horz_anchors, lidar_vert_anchors=config.lidar_vert_anchors, lidar_horz_anchors=config.lidar_horz_anchors, seq_len=config.seq_len, embd_pdrop=config.embd_pdrop, attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop, config=config, use_velocity=use_velocity)
        self.transformer4 = GPT(n_embd=self.image_encoder.features.feature_info[4]['num_chs'], n_head=config.n_head, block_exp=config.block_exp, n_layer=config.n_layer, img_vert_anchors=config.img_vert_anchors, img_horz_anchors=config.img_horz_anchors, lidar_vert_anchors=config.lidar_vert_anchors, lidar_horz_anchors=config.lidar_horz_anchors, seq_len=config.seq_len, embd_pdrop=config.embd_pdrop, attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop, config=config, use_velocity=use_velocity)
        if self.image_encoder.features.feature_info[4]['num_chs'] != self.config.perception_output_features:
            self.change_channel_conv_image = nn.Conv2d(self.image_encoder.features.feature_info[4]['num_chs'], self.config.perception_output_features, (1, 1))
            self.change_channel_conv_lidar = nn.Conv2d(self.image_encoder.features.feature_info[4]['num_chs'], self.config.perception_output_features, (1, 1))
        else:
            self.change_channel_conv_image = nn.Sequential()
            self.change_channel_conv_lidar = nn.Sequential()
        channel = self.config.bev_features_chanels
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=self.config.bev_upsample_factor, mode='bilinear', align_corners=False)
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))
        self.c5_conv = nn.Conv2d(self.config.perception_output_features, channel, (1, 1))

    def top_down(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample(p4)))
        p2 = self.relu(self.up_conv3(self.upsample(p3)))
        return (p2, p3, p4, p5)

    def forward(self, image, lidar, velocity):
        """
        Image + LiDAR feature fusion using transformers
        Args:
            image: input rgb image
            lidar: LiDAR input will be replaced by positional encoding. Third channel may contain target point.
            velocity (tensor): input velocity from speedometer
        """
        if self.image_encoder.normalize:
            image_tensor = normalize_imagenet(image)
        else:
            image_tensor = image
        x = torch.linspace(-1, 1, self.config.lidar_resolution_width)
        y = torch.linspace(-1, 1, self.config.lidar_resolution_height)
        y_grid, x_grid = torch.meshgrid(x, y, indexing='ij')
        lidar[:, 0] = y_grid.unsqueeze(0)
        lidar[:, 1] = x_grid.unsqueeze(0)
        lidar_tensor = lidar
        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.act1(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)
        lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
        lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        lidar_features = self.lidar_encoder._model.act1(lidar_features)
        lidar_features = self.lidar_encoder._model.maxpool(lidar_features)
        image_features = self.image_encoder.features.layer1(image_features)
        lidar_features = self.lidar_encoder._model.layer1(lidar_features)
        image_embd_layer1 = self.avgpool_img(image_features)
        lidar_embd_layer1 = self.avgpool_lidar(lidar_features)
        image_features_layer1, lidar_features_layer1 = self.transformer1(image_embd_layer1, lidar_embd_layer1, velocity)
        image_features_layer1 = F.interpolate(image_features_layer1, size=(image_features.shape[2], image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_layer1 = F.interpolate(lidar_features_layer1, size=(lidar_features.shape[2], lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer1
        lidar_features = lidar_features + lidar_features_layer1
        image_features = self.image_encoder.features.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        image_embd_layer2 = self.avgpool_img(image_features)
        lidar_embd_layer2 = self.avgpool_lidar(lidar_features)
        image_features_layer2, lidar_features_layer2 = self.transformer2(image_embd_layer2, lidar_embd_layer2, velocity)
        image_features_layer2 = F.interpolate(image_features_layer2, size=(image_features.shape[2], image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_layer2 = F.interpolate(lidar_features_layer2, size=(lidar_features.shape[2], lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer2
        lidar_features = lidar_features + lidar_features_layer2
        image_features = self.image_encoder.features.layer3(image_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        image_embd_layer3 = self.avgpool_img(image_features)
        lidar_embd_layer3 = self.avgpool_lidar(lidar_features)
        image_features_layer3, lidar_features_layer3 = self.transformer3(image_embd_layer3, lidar_embd_layer3, velocity)
        image_features_layer3 = F.interpolate(image_features_layer3, size=(image_features.shape[2], image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_layer3 = F.interpolate(lidar_features_layer3, size=(lidar_features.shape[2], lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer3
        lidar_features = lidar_features + lidar_features_layer3
        image_features = self.image_encoder.features.layer4(image_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        image_embd_layer4 = self.avgpool_img(image_features)
        lidar_embd_layer4 = self.avgpool_lidar(lidar_features)
        image_features_layer4, lidar_features_layer4 = self.transformer4(image_embd_layer4, lidar_embd_layer4, velocity)
        image_features_layer4 = F.interpolate(image_features_layer4, size=(image_features.shape[2], image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_layer4 = F.interpolate(lidar_features_layer4, size=(lidar_features.shape[2], lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer4
        lidar_features = lidar_features + lidar_features_layer4
        image_features = self.change_channel_conv_image(image_features)
        lidar_features = self.change_channel_conv_lidar(lidar_features)
        x4 = lidar_features
        image_features_grid = image_features
        image_features = self.image_encoder.features.global_pool(image_features)
        image_features = torch.flatten(image_features, 1)
        lidar_features = self.lidar_encoder._model.global_pool(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)
        fused_features = image_features + lidar_features
        features = self.top_down(x4)
        return (features, image_features_grid, fused_features)

class GeometricFusionBackbone(nn.Module):
    """
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """

    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=0):
        super().__init__()
        self.config = config
        self.use_velocity = use_velocity
        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((self.config.lidar_vert_anchors, self.config.lidar_horz_anchors))
        if config.use_point_pillars == True:
            in_channels = config.num_features[-1]
        else:
            in_channels = 2 * config.lidar_seq_len
        if self.config.use_target_point_image == True:
            in_channels += 1
        self.image_encoder = ImageCNN(architecture=image_architecture, normalize=True)
        self.lidar_encoder = LidarEncoder(architecture=lidar_architecture, in_channels=in_channels)
        self.image_conv1 = nn.Conv2d(self.image_encoder.features.feature_info[1]['num_chs'], config.n_embd, 1)
        self.image_conv2 = nn.Conv2d(self.image_encoder.features.feature_info[2]['num_chs'], config.n_embd, 1)
        self.image_conv3 = nn.Conv2d(self.image_encoder.features.feature_info[3]['num_chs'], config.n_embd, 1)
        self.image_conv4 = nn.Conv2d(self.image_encoder.features.feature_info[4]['num_chs'], config.n_embd, 1)
        self.image_deconv1 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[1]['num_chs'], 1)
        self.image_deconv2 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[2]['num_chs'], 1)
        self.image_deconv3 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[3]['num_chs'], 1)
        self.image_deconv4 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[4]['num_chs'], 1)
        if use_velocity:
            self.vel_emb1 = nn.Linear(1, self.image_encoder.features.feature_info[1]['num_chs'])
            self.vel_emb2 = nn.Linear(1, self.image_encoder.features.feature_info[2]['num_chs'])
            self.vel_emb3 = nn.Linear(1, self.image_encoder.features.feature_info[3]['num_chs'])
            self.vel_emb4 = nn.Linear(1, self.image_encoder.features.feature_info[4]['num_chs'])
        self.lidar_conv1 = nn.Conv2d(self.image_encoder.features.feature_info[1]['num_chs'], config.n_embd, 1)
        self.lidar_conv2 = nn.Conv2d(self.image_encoder.features.feature_info[2]['num_chs'], config.n_embd, 1)
        self.lidar_conv3 = nn.Conv2d(self.image_encoder.features.feature_info[3]['num_chs'], config.n_embd, 1)
        self.lidar_conv4 = nn.Conv2d(self.image_encoder.features.feature_info[4]['num_chs'], config.n_embd, 1)
        self.lidar_deconv1 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[1]['num_chs'], 1)
        self.lidar_deconv2 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[2]['num_chs'], 1)
        self.lidar_deconv3 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[3]['num_chs'], 1)
        self.lidar_deconv4 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[4]['num_chs'], 1)
        hid_dim = config.n_embd
        self.image_projection1 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.image_projection2 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.image_projection3 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.image_projection4 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.lidar_projection1 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.lidar_projection2 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.lidar_projection3 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.lidar_projection4 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        if self.image_encoder.features.feature_info[4]['num_chs'] != self.config.perception_output_features:
            self.change_channel_conv_image = nn.Conv2d(self.image_encoder.features.feature_info[4]['num_chs'], self.config.perception_output_features, (1, 1))
            self.change_channel_conv_lidar = nn.Conv2d(self.image_encoder.features.feature_info[4]['num_chs'], self.config.perception_output_features, (1, 1))
        else:
            self.change_channel_conv_image = nn.Sequential()
            self.change_channel_conv_lidar = nn.Sequential()
        channel = self.config.bev_features_chanels
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=self.config.bev_upsample_factor, mode='bilinear', align_corners=False)
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))
        self.c5_conv = nn.Conv2d(self.config.perception_output_features, channel, (1, 1))

    def top_down(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample(p4)))
        p2 = self.relu(self.up_conv3(self.upsample(p3)))
        return (p2, p3, p4, p5)

    def forward(self, image, lidar, velocity, bev_points, img_points):
        """
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
            bev_points (tensor): projected image pixels onto the BEV grid
            cam_points (tensor): projected LiDAR point cloud onto the image space
        """
        if self.image_encoder.normalize:
            image_tensor = normalize_imagenet(image)
        else:
            image_tensor = image
        lidar_tensor = lidar
        bz = lidar_tensor.shape[0]
        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.act1(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)
        lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
        lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        lidar_features = self.lidar_encoder._model.act1(lidar_features)
        lidar_features = self.lidar_encoder._model.maxpool(lidar_features)
        image_features = self.image_encoder.features.layer1(image_features)
        lidar_features = self.lidar_encoder._model.layer1(lidar_features)
        if self.config.n_scale >= 4:
            image_embd_layer1 = self.image_conv1(image_features)
            image_embd_layer1 = self.avgpool_img(image_embd_layer1)
            lidar_embd_layer1 = self.lidar_conv1(lidar_features)
            lidar_embd_layer1 = self.avgpool_lidar(lidar_embd_layer1)
            curr_h_image, curr_w_image = image_embd_layer1.shape[-2:]
            curr_h_lidar, curr_w_lidar = lidar_embd_layer1.shape[-2:]
            bev_points_layer1 = bev_points.view(bz * curr_h_lidar * curr_w_lidar * 5, 2)
            bev_encoding_layer1 = image_embd_layer1.permute(0, 2, 3, 1).contiguous()[:, bev_points_layer1[:, 1], bev_points_layer1[:, 0]].view(bz, bz, curr_h_lidar, curr_w_lidar, 5, -1)
            bev_encoding_layer1 = torch.diagonal(bev_encoding_layer1, 0).permute(4, 3, 0, 1, 2).contiguous()
            bev_encoding_layer1 = torch.sum(bev_encoding_layer1, -1)
            bev_encoding_layer1 = self.image_projection1(bev_encoding_layer1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
            lidar_features_layer1 = F.interpolate(bev_encoding_layer1, scale_factor=8, mode='bilinear', align_corners=False)
            lidar_features_layer1 = self.lidar_deconv1(lidar_features_layer1)
            lidar_features = lidar_features + lidar_features_layer1
            if self.use_velocity:
                vel_embedding1 = self.vel_emb1(velocity).unsqueeze(-1).unsqueeze(-1)
                lidar_features = lidar_features + vel_embedding1
            img_points_layer1 = img_points.view(bz * curr_h_image * curr_w_image * 5, 2)
            img_encoding_layer1 = lidar_embd_layer1.permute(0, 2, 3, 1).contiguous()[:, img_points_layer1[:, 1], img_points_layer1[:, 0]].view(bz, bz, curr_h_image, curr_w_image, 5, -1)
            img_encoding_layer1 = torch.diagonal(img_encoding_layer1, 0).permute(4, 3, 0, 1, 2).contiguous()
            img_encoding_layer1 = torch.sum(img_encoding_layer1, -1)
            img_encoding_layer1 = self.lidar_projection1(img_encoding_layer1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
            image_features_layer1 = F.interpolate(img_encoding_layer1, scale_factor=8, mode='bilinear', align_corners=False)
            image_features_layer1 = self.image_deconv1(image_features_layer1)
            image_features = image_features + image_features_layer1
            if self.use_velocity:
                image_features = image_features + vel_embedding1
        image_features = self.image_encoder.features.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        if self.config.n_scale >= 3:
            image_embd_layer2 = self.image_conv2(image_features)
            image_embd_layer2 = self.avgpool_img(image_embd_layer2)
            lidar_embd_layer2 = self.lidar_conv2(lidar_features)
            lidar_embd_layer2 = self.avgpool_lidar(lidar_embd_layer2)
            curr_h_image, curr_w_image = image_embd_layer2.shape[-2:]
            curr_h_lidar, curr_w_lidar = lidar_embd_layer2.shape[-2:]
            bev_points_layer2 = bev_points.view(bz * curr_h_lidar * curr_w_lidar * 5, 2)
            bev_encoding_layer2 = image_embd_layer2.permute(0, 2, 3, 1).contiguous()[:, bev_points_layer2[:, 1], bev_points_layer2[:, 0]].view(bz, bz, curr_h_lidar, curr_w_lidar, 5, -1)
            bev_encoding_layer2 = torch.diagonal(bev_encoding_layer2, 0).permute(4, 3, 0, 1, 2).contiguous()
            bev_encoding_layer2 = torch.sum(bev_encoding_layer2, -1)
            bev_encoding_layer2 = self.image_projection2(bev_encoding_layer2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
            lidar_features_layer2 = F.interpolate(bev_encoding_layer2, scale_factor=4, mode='bilinear', align_corners=False)
            lidar_features_layer2 = self.lidar_deconv2(lidar_features_layer2)
            lidar_features = lidar_features + lidar_features_layer2
            if self.use_velocity:
                vel_embedding2 = self.vel_emb2(velocity).unsqueeze(-1).unsqueeze(-1)
                lidar_features = lidar_features + vel_embedding2
            img_points_layer2 = img_points.view(bz * curr_h_image * curr_w_image * 5, 2)
            img_encoding_layer2 = lidar_embd_layer2.permute(0, 2, 3, 1).contiguous()[:, img_points_layer2[:, 1], img_points_layer2[:, 0]].view(bz, bz, curr_h_image, curr_w_image, 5, -1)
            img_encoding_layer2 = torch.diagonal(img_encoding_layer2, 0).permute(4, 3, 0, 1, 2).contiguous()
            img_encoding_layer2 = torch.sum(img_encoding_layer2, -1)
            img_encoding_layer2 = self.lidar_projection2(img_encoding_layer2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
            image_features_layer2 = F.interpolate(img_encoding_layer2, scale_factor=4, mode='bilinear', align_corners=False)
            image_features_layer2 = self.image_deconv2(image_features_layer2)
            image_features = image_features + image_features_layer2
            if self.use_velocity:
                image_features = image_features + vel_embedding2
        image_features = self.image_encoder.features.layer3(image_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        if self.config.n_scale >= 2:
            image_embd_layer3 = self.image_conv3(image_features)
            image_embd_layer3 = self.avgpool_img(image_embd_layer3)
            lidar_embd_layer3 = self.lidar_conv3(lidar_features)
            lidar_embd_layer3 = self.avgpool_lidar(lidar_embd_layer3)
            curr_h_image, curr_w_image = image_embd_layer3.shape[-2:]
            curr_h_lidar, curr_w_lidar = lidar_embd_layer3.shape[-2:]
            bev_points_layer3 = bev_points.view(bz * curr_h_lidar * curr_w_lidar * 5, 2)
            bev_encoding_layer3 = image_embd_layer3.permute(0, 2, 3, 1).contiguous()[:, bev_points_layer3[:, 1], bev_points_layer3[:, 0]].view(bz, bz, curr_h_lidar, curr_w_lidar, 5, -1)
            bev_encoding_layer3 = torch.diagonal(bev_encoding_layer3, 0).permute(4, 3, 0, 1, 2).contiguous()
            bev_encoding_layer3 = torch.sum(bev_encoding_layer3, -1)
            bev_encoding_layer3 = self.image_projection3(bev_encoding_layer3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
            lidar_features_layer3 = F.interpolate(bev_encoding_layer3, scale_factor=2, mode='bilinear', align_corners=False)
            lidar_features_layer3 = self.lidar_deconv3(lidar_features_layer3)
            lidar_features = lidar_features + lidar_features_layer3
            if self.use_velocity:
                vel_embedding3 = self.vel_emb3(velocity).unsqueeze(-1).unsqueeze(-1)
                lidar_features = lidar_features + vel_embedding3
            img_points_layer3 = img_points.view(bz * curr_h_image * curr_w_image * 5, 2)
            img_encoding_layer3 = lidar_embd_layer3.permute(0, 2, 3, 1).contiguous()[:, img_points_layer3[:, 1], img_points_layer3[:, 0]].view(bz, bz, curr_h_image, curr_w_image, 5, -1)
            img_encoding_layer3 = torch.diagonal(img_encoding_layer3, 0).permute(4, 3, 0, 1, 2).contiguous()
            img_encoding_layer3 = torch.sum(img_encoding_layer3, -1)
            img_encoding_layer3 = self.lidar_projection3(img_encoding_layer3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
            image_features_layer3 = F.interpolate(img_encoding_layer3, scale_factor=2, mode='bilinear', align_corners=False)
            image_features_layer3 = self.image_deconv3(image_features_layer3)
            image_features = image_features + image_features_layer3
            if self.use_velocity:
                image_features = image_features + vel_embedding3
        image_features = self.image_encoder.features.layer4(image_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        if self.config.n_scale >= 1:
            image_embd_layer4 = self.image_conv4(image_features)
            image_embd_layer4 = self.avgpool_img(image_embd_layer4)
            lidar_embd_layer4 = self.lidar_conv4(lidar_features)
            lidar_embd_layer4 = self.avgpool_lidar(lidar_embd_layer4)
            curr_h_image, curr_w_image = image_embd_layer4.shape[-2:]
            curr_h_lidar, curr_w_lidar = lidar_embd_layer4.shape[-2:]
            bev_points_layer4 = bev_points.view(bz * curr_h_lidar * curr_w_lidar * 5, 2)
            bev_encoding_layer4 = image_embd_layer4.permute(0, 2, 3, 1).contiguous()[:, bev_points_layer4[:, 1], bev_points_layer4[:, 0]].view(bz, bz, curr_h_lidar, curr_w_lidar, 5, -1)
            bev_encoding_layer4 = torch.diagonal(bev_encoding_layer4, 0).permute(4, 3, 0, 1, 2).contiguous()
            bev_encoding_layer4 = torch.sum(bev_encoding_layer4, -1)
            bev_encoding_layer4 = self.image_projection4(bev_encoding_layer4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
            lidar_features_layer4 = self.lidar_deconv4(bev_encoding_layer4)
            lidar_features = lidar_features + lidar_features_layer4
            if self.use_velocity:
                vel_embedding4 = self.vel_emb4(velocity).unsqueeze(-1).unsqueeze(-1)
                lidar_features = lidar_features + vel_embedding4
            img_points_layer4 = img_points.view(bz * curr_h_image * curr_w_image * 5, 2)
            img_encoding_layer4 = lidar_embd_layer3.permute(0, 2, 3, 1).contiguous()[:, img_points_layer4[:, 1], img_points_layer4[:, 0]].view(bz, bz, curr_h_image, curr_w_image, 5, -1)
            img_encoding_layer4 = torch.diagonal(img_encoding_layer4, 0).permute(4, 3, 0, 1, 2).contiguous()
            img_encoding_layer4 = torch.sum(img_encoding_layer4, -1)
            img_encoding_layer4 = self.lidar_projection4(img_encoding_layer4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
            image_features_layer4 = self.image_deconv4(img_encoding_layer4)
            image_features = image_features + image_features_layer4
            if self.use_velocity:
                image_features = image_features + vel_embedding4
        image_features = self.change_channel_conv_image(image_features)
        lidar_features = self.change_channel_conv_lidar(lidar_features)
        x4 = lidar_features
        image_features_grid = image_features
        image_features = self.image_encoder.features.global_pool(image_features)
        image_features = torch.flatten(image_features, 1)
        lidar_features = self.lidar_encoder._model.global_pool(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)
        fused_features = image_features + lidar_features
        features = self.top_down(x4)
        return (features, image_features_grid, fused_features)

