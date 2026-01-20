# Cluster 19

def sext_cpn_backbone(input_image, istraining, data_format, net_depth=50, group=32):
    bn_axis = -1 if data_format == 'channels_last' else 1
    if data_format == 'channels_last':
        image_channels = tf.unstack(input_image, axis=-1)
        swaped_input_image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1)
    else:
        image_channels = tf.unstack(input_image, axis=1)
        swaped_input_image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=1)
    if net_depth not in [50, 101]:
        raise TypeError('Only ResNeXt50 or ResNeXt101 is supprted now.')
    input_depth = [256, 512, 1024, 2048]
    num_units = [3, 4, 6, 3] if net_depth == 50 else [3, 4, 23, 3]
    block_name_prefix = ['conv2_{}', 'conv3_{}', 'conv4_{}', 'conv5_{}']
    if data_format == 'channels_first':
        swaped_input_image = tf.pad(swaped_input_image, paddings=[[0, 0], [0, 0], [3, 3], [3, 3]])
    else:
        swaped_input_image = tf.pad(swaped_input_image, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]])
    inputs_features = tf.layers.conv2d(swaped_input_image, input_depth[0] // 4, (7, 7), use_bias=False, name='conv1/7x7_s2', strides=(2, 2), padding='valid', data_format=data_format, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
    inputs_features = tf.layers.batch_normalization(inputs_features, momentum=_BATCH_NORM_DECAY, name='conv1/7x7_s2/bn', axis=bn_axis, epsilon=_BATCH_NORM_EPSILON, training=istraining, reuse=None, fused=_USE_FUSED_BN)
    inputs_features = tf.nn.relu(inputs_features, name='conv1/relu_7x7_s2')
    inputs_features = tf.layers.max_pooling2d(inputs_features, [3, 3], [2, 2], padding='same', data_format=data_format, name='pool1/3x3_s2')
    end_points = []
    is_root = True
    for ind, num_unit in enumerate(num_units):
        need_reduce = True
        for unit_index in range(1, num_unit + 1):
            inputs_features = se_next_bottleneck_block(inputs_features, input_depth[ind], block_name_prefix[ind].format(unit_index), is_training=istraining, group=group, data_format=data_format, need_reduce=need_reduce, is_root=is_root)
            need_reduce = False
            is_root = False
        end_points.append(inputs_features)
    return end_points

def se_cpn_backbone(input_image, istraining, data_format):
    bn_axis = -1 if data_format == 'channels_last' else 1
    if data_format == 'channels_last':
        image_channels = tf.unstack(input_image, axis=-1)
        swaped_input_image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1)
    else:
        image_channels = tf.unstack(input_image, axis=1)
        swaped_input_image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=1)
    input_depth = [128, 256, 512, 1024]
    num_units = [3, 4, 6, 3]
    block_name_prefix = ['conv2_{}', 'conv3_{}', 'conv4_{}', 'conv5_{}']
    if data_format == 'channels_first':
        swaped_input_image = tf.pad(swaped_input_image, paddings=[[0, 0], [0, 0], [3, 3], [3, 3]])
    else:
        swaped_input_image = tf.pad(swaped_input_image, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]])
    inputs_features = tf.layers.conv2d(swaped_input_image, input_depth[0] // 2, (7, 7), use_bias=False, name='conv1/7x7_s2', strides=(2, 2), padding='valid', data_format=data_format, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
    inputs_features = tf.layers.batch_normalization(inputs_features, momentum=_BATCH_NORM_DECAY, name='conv1/7x7_s2/bn', axis=bn_axis, epsilon=_BATCH_NORM_EPSILON, training=istraining, reuse=None, fused=_USE_FUSED_BN)
    inputs_features = tf.nn.relu(inputs_features, name='conv1/relu_7x7_s2')
    inputs_features = tf.layers.max_pooling2d(inputs_features, [3, 3], [2, 2], padding='same', data_format=data_format, name='pool1/3x3_s2')
    end_points = []
    is_root = True
    for ind, num_unit in enumerate(num_units):
        need_reduce = True
        for unit_index in range(1, num_unit + 1):
            inputs_features = se_bottleneck_block(inputs_features, input_depth[ind], block_name_prefix[ind].format(unit_index), is_training=istraining, data_format=data_format, need_reduce=need_reduce, is_root=is_root)
            need_reduce = False
            is_root = False
        end_points.append(inputs_features)
    return end_points

def se_bottleneck_block(inputs, input_filters, name_prefix, is_training, data_format='channels_last', need_reduce=True, is_root=False, reduced_scale=16):
    bn_axis = -1 if data_format == 'channels_last' else 1
    strides_to_use = 1
    residuals = inputs
    if need_reduce:
        strides_to_use = 1 if is_root else 2
        proj_mapping = tf.layers.conv2d(inputs, input_filters * 2, (1, 1), use_bias=False, name=name_prefix + '_1x1_proj', strides=(strides_to_use, strides_to_use), padding='valid', data_format=data_format, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
        residuals = tf.layers.batch_normalization(proj_mapping, momentum=_BATCH_NORM_DECAY, name=name_prefix + '_1x1_proj/bn', axis=bn_axis, epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)
    reduced_inputs = tf.layers.conv2d(inputs, input_filters / 2, (1, 1), use_bias=False, name=name_prefix + '_1x1_reduce', strides=(strides_to_use, strides_to_use), padding='valid', data_format=data_format, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
    reduced_inputs_bn = tf.layers.batch_normalization(reduced_inputs, momentum=_BATCH_NORM_DECAY, name=name_prefix + '_1x1_reduce/bn', axis=bn_axis, epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)
    reduced_inputs_relu = tf.nn.relu(reduced_inputs_bn, name=name_prefix + '_1x1_reduce/relu')
    conv3_inputs = tf.layers.conv2d(reduced_inputs_relu, input_filters / 2, (3, 3), use_bias=False, name=name_prefix + '_3x3', strides=(1, 1), padding='same', data_format=data_format, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
    conv3_inputs_bn = tf.layers.batch_normalization(conv3_inputs, momentum=_BATCH_NORM_DECAY, name=name_prefix + '_3x3/bn', axis=bn_axis, epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)
    conv3_inputs_relu = tf.nn.relu(conv3_inputs_bn, name=name_prefix + '_3x3/relu')
    increase_inputs = tf.layers.conv2d(conv3_inputs_relu, input_filters * 2, (1, 1), use_bias=False, name=name_prefix + '_1x1_increase', strides=(1, 1), padding='valid', data_format=data_format, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
    increase_inputs_bn = tf.layers.batch_normalization(increase_inputs, momentum=_BATCH_NORM_DECAY, name=name_prefix + '_1x1_increase/bn', axis=bn_axis, epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)
    if data_format == 'channels_first':
        pooled_inputs = tf.reduce_mean(increase_inputs_bn, [2, 3], name=name_prefix + '_global_pool', keep_dims=True)
    else:
        pooled_inputs = tf.reduce_mean(increase_inputs_bn, [1, 2], name=name_prefix + '_global_pool', keep_dims=True)
    down_inputs = tf.layers.conv2d(pooled_inputs, input_filters * 2 // reduced_scale, (1, 1), use_bias=True, name=name_prefix + '_1x1_down', strides=(1, 1), padding='valid', data_format=data_format, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
    down_inputs_relu = tf.nn.relu(down_inputs, name=name_prefix + '_1x1_down/relu')
    up_inputs = tf.layers.conv2d(down_inputs_relu, input_filters * 2, (1, 1), use_bias=True, name=name_prefix + '_1x1_up', strides=(1, 1), padding='valid', data_format=data_format, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
    prob_outputs = tf.nn.sigmoid(up_inputs, name=name_prefix + '_prob')
    rescaled_feat = tf.multiply(prob_outputs, increase_inputs_bn, name=name_prefix + '_mul')
    pre_act = tf.add(residuals, rescaled_feat, name=name_prefix + '_add')
    return tf.nn.relu(pre_act, name=name_prefix + '/relu')

def global_net_bottleneck_block(inputs, filters, istraining, data_format, projection_shortcut=None, name=None):
    with tf.variable_scope(name, 'global_net_bottleneck', values=[inputs]):
        shortcut = inputs
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)
            shortcut = batch_norm(inputs=shortcut, training=istraining, data_format=data_format, name='batch_normalization_shortcut')
        inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=1, data_format=data_format, name='1x1_down')
        inputs = batch_norm(inputs, istraining, data_format, name='batch_normalization_1')
        inputs = tf.nn.relu(inputs, name='relu1')
        inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1, data_format=data_format, name='3x3_conv')
        inputs = batch_norm(inputs, istraining, data_format, name='batch_normalization_2')
        inputs = tf.nn.relu(inputs, name='relu2')
        inputs = conv2d_fixed_padding(inputs=inputs, filters=2 * filters, kernel_size=1, strides=1, data_format=data_format, name='1x1_up')
        inputs = batch_norm(inputs, istraining, data_format, name='batch_normalization_3')
        inputs += shortcut
        inputs = tf.nn.relu(inputs, name='relu3')
        return inputs

def global_net_sext_bottleneck_block(inputs, input_filters, is_training, data_format, need_reduce=False, name_prefix=None, group=32, reduced_scale=16):
    with tf.variable_scope(name_prefix, 'global_net_sext_bottleneck_block', values=[inputs]):
        bn_axis = -1 if data_format == 'channels_last' else 1
        residuals = inputs
        if need_reduce:
            proj_mapping = tf.layers.conv2d(inputs, input_filters * 2, (1, 1), use_bias=False, name=name_prefix + '_1x1_proj', strides=(1, 1), padding='valid', data_format=data_format, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
            residuals = tf.layers.batch_normalization(proj_mapping, momentum=_BATCH_NORM_DECAY, name=name_prefix + '_1x1_proj/bn', axis=bn_axis, epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)
        reduced_inputs = tf.layers.conv2d(inputs, input_filters, (1, 1), use_bias=False, name=name_prefix + '_1x1_reduce', strides=(1, 1), padding='valid', data_format=data_format, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
        reduced_inputs_bn = tf.layers.batch_normalization(reduced_inputs, momentum=_BATCH_NORM_DECAY, name=name_prefix + '_1x1_reduce/bn', axis=bn_axis, epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)
        reduced_inputs_relu = tf.nn.relu(reduced_inputs_bn, name=name_prefix + '_1x1_reduce/relu')
        if data_format == 'channels_first':
            reduced_inputs_relu = tf.pad(reduced_inputs_relu, paddings=[[0, 0], [0, 0], [1, 1], [1, 1]])
            weight_shape = [3, 3, reduced_inputs_relu.get_shape().as_list()[1] // group, input_filters]
            if is_training:
                weight_ = tf.Variable(constant_xavier_initializer(weight_shape, group=group, dtype=tf.float32), trainable=is_training, name=name_prefix + '_3x3/kernel')
            else:
                weight_ = tf.get_variable(name_prefix + '_3x3/kernel', shape=weight_shape, initializer=wrapper_initlizer, trainable=is_training)
            weight_groups = tf.split(weight_, num_or_size_splits=group, axis=-1, name=name_prefix + '_weight_split')
            xs = tf.split(reduced_inputs_relu, num_or_size_splits=group, axis=1, name=name_prefix + '_inputs_split')
        else:
            reduced_inputs_relu = tf.pad(reduced_inputs_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            weight_shape = [3, 3, reduced_inputs_relu.get_shape().as_list()[-1] // group, input_filters]
            if is_training:
                weight_ = tf.Variable(constant_xavier_initializer(weight_shape, group=group, dtype=tf.float32), trainable=is_training, name=name_prefix + '_3x3/kernel')
            else:
                weight_ = tf.get_variable(name_prefix + '_3x3/kernel', shape=weight_shape, initializer=wrapper_initlizer, trainable=is_training)
            weight_groups = tf.split(weight_, num_or_size_splits=group, axis=-1, name=name_prefix + '_weight_split')
            xs = tf.split(reduced_inputs_relu, num_or_size_splits=group, axis=-1, name=name_prefix + '_inputs_split')
        convolved = [tf.nn.convolution(x, weight, padding='VALID', strides=[1, 1], name=name_prefix + '_group_conv', data_format='NCHW' if data_format == 'channels_first' else 'NHWC') for x, weight in zip(xs, weight_groups)]
        if data_format == 'channels_first':
            conv3_inputs = tf.concat(convolved, axis=1, name=name_prefix + '_concat')
        else:
            conv3_inputs = tf.concat(convolved, axis=-1, name=name_prefix + '_concat')
        conv3_inputs_bn = tf.layers.batch_normalization(conv3_inputs, momentum=_BATCH_NORM_DECAY, name=name_prefix + '_3x3/bn', axis=bn_axis, epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)
        conv3_inputs_relu = tf.nn.relu(conv3_inputs_bn, name=name_prefix + '_3x3/relu')
        increase_inputs = tf.layers.conv2d(conv3_inputs_relu, input_filters * 2, (1, 1), use_bias=False, name=name_prefix + '_1x1_increase', strides=(1, 1), padding='valid', data_format=data_format, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
        increase_inputs_bn = tf.layers.batch_normalization(increase_inputs, momentum=_BATCH_NORM_DECAY, name=name_prefix + '_1x1_increase/bn', axis=bn_axis, epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)
        if data_format == 'channels_first':
            pooled_inputs = tf.reduce_mean(increase_inputs_bn, [2, 3], name=name_prefix + '_global_pool', keep_dims=True)
        else:
            pooled_inputs = tf.reduce_mean(increase_inputs_bn, [1, 2], name=name_prefix + '_global_pool', keep_dims=True)
        down_inputs = tf.layers.conv2d(pooled_inputs, input_filters * 2 // reduced_scale, (1, 1), use_bias=True, name=name_prefix + '_1x1_down', strides=(1, 1), padding='valid', data_format=data_format, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
        down_inputs_relu = tf.nn.relu(down_inputs, name=name_prefix + '_1x1_down/relu')
        up_inputs = tf.layers.conv2d(down_inputs_relu, input_filters * 2, (1, 1), use_bias=True, name=name_prefix + '_1x1_up', strides=(1, 1), padding='valid', data_format=data_format, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
        prob_outputs = tf.nn.sigmoid(up_inputs, name=name_prefix + '_prob')
        rescaled_feat = tf.multiply(prob_outputs, increase_inputs_bn, name=name_prefix + '_mul')
        pre_act = tf.add(residuals, rescaled_feat, name=name_prefix + '_add')
        return tf.nn.relu(pre_act, name=name_prefix + '/relu')

def cascaded_pyramid_net(inputs, output_channals, heatmap_size, istraining, data_format):
    end_points = se_cpn_backbone(inputs, istraining, data_format)
    pyramid_len = len(end_points)
    up_sampling = None
    pyramid_heatmaps = []
    pyramid_laterals = []
    with tf.variable_scope('feature_pyramid', 'feature_pyramid', values=end_points):
        for ind, pyramid in enumerate(reversed(end_points)):
            inputs = conv2d_fixed_padding(inputs=pyramid, filters=256, kernel_size=1, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv1_p{}'.format(pyramid_len - ind))
            lateral = tf.nn.relu(inputs, name='relu1_p{}'.format(pyramid_len - ind))
            if up_sampling is not None:
                if data_format == 'channels_first':
                    up_sampling = tf.transpose(up_sampling, [0, 2, 3, 1], name='trans_p{}'.format(pyramid_len - ind))
                up_sampling = tf.image.resize_bilinear(up_sampling, tf.shape(up_sampling)[-3:-1] * 2, name='upsample_p{}'.format(pyramid_len - ind))
                if data_format == 'channels_first':
                    up_sampling = tf.transpose(up_sampling, [0, 3, 1, 2], name='trans_inv_p{}'.format(pyramid_len - ind))
                up_sampling = conv2d_fixed_padding(inputs=up_sampling, filters=256, kernel_size=1, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='up_conv_p{}'.format(pyramid_len - ind))
                up_sampling = lateral + up_sampling
                lateral = up_sampling
            else:
                up_sampling = lateral
            pyramid_laterals.append(lateral)
            lateral = conv2d_fixed_padding(inputs=lateral, filters=256, kernel_size=1, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv2_p{}'.format(pyramid_len - ind))
            lateral = tf.nn.relu(lateral, name='relu2_p{}'.format(pyramid_len - ind))
            outputs = conv2d_fixed_padding(inputs=lateral, filters=output_channals, kernel_size=3, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='conv_heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 2, 3, 1], name='output_trans_p{}'.format(pyramid_len - ind))
            outputs = tf.image.resize_bilinear(outputs, [heatmap_size, heatmap_size], name='heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2], name='heatmap_trans_inv_p{}'.format(pyramid_len - ind))
            pyramid_heatmaps.append(outputs)
    with tf.variable_scope('global_net', 'global_net', values=pyramid_laterals):
        global_pyramids = []
        for ind, lateral in enumerate(pyramid_laterals):
            inputs = lateral
            for bottleneck_ind in range(pyramid_len - ind - 1):
                inputs = global_net_bottleneck_block(inputs, 128, istraining, data_format, name='global_net_bottleneck_{}_p{}'.format(bottleneck_ind, pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(inputs, [0, 2, 3, 1], name='global_output_trans_p{}'.format(pyramid_len - ind))
            else:
                outputs = inputs
            outputs = tf.image.resize_bilinear(outputs, [heatmap_size, heatmap_size], name='global_heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2], name='global_heatmap_trans_inv_p{}'.format(pyramid_len - ind))
            global_pyramids.append(outputs)
        concat_pyramids = tf.concat(global_pyramids, 1 if data_format == 'channels_first' else 3, name='concat')

        def projection_shortcut(inputs):
            return conv2d_fixed_padding(inputs=inputs, filters=256, kernel_size=1, strides=1, data_format=data_format, name='shortcut')
        outputs = global_net_bottleneck_block(concat_pyramids, 128, istraining, data_format, projection_shortcut=projection_shortcut, name='global_concat_bottleneck')
        outputs = conv2d_fixed_padding(inputs=outputs, filters=output_channals, kernel_size=3, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='conv_heatmap')
    return pyramid_heatmaps + [outputs]

def xt_cascaded_pyramid_net(inputs, output_channals, heatmap_size, istraining, data_format, net_depth=50):
    end_points = sext_cpn_backbone(inputs, istraining, data_format, net_depth=net_depth)
    pyramid_len = len(end_points)
    up_sampling = None
    pyramid_heatmaps = []
    pyramid_laterals = []
    with tf.variable_scope('feature_pyramid', 'feature_pyramid', values=end_points):
        for ind, pyramid in enumerate(reversed(end_points)):
            inputs = conv2d_fixed_padding(inputs=pyramid, filters=256, kernel_size=1, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv1_p{}'.format(pyramid_len - ind))
            lateral = tf.nn.relu(inputs, name='relu1_p{}'.format(pyramid_len - ind))
            if up_sampling is not None:
                if data_format == 'channels_first':
                    up_sampling = tf.transpose(up_sampling, [0, 2, 3, 1], name='trans_p{}'.format(pyramid_len - ind))
                up_sampling = tf.image.resize_bilinear(up_sampling, tf.shape(up_sampling)[-3:-1] * 2, name='upsample_p{}'.format(pyramid_len - ind))
                if data_format == 'channels_first':
                    up_sampling = tf.transpose(up_sampling, [0, 3, 1, 2], name='trans_inv_p{}'.format(pyramid_len - ind))
                up_sampling = conv2d_fixed_padding(inputs=up_sampling, filters=256, kernel_size=1, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='up_conv_p{}'.format(pyramid_len - ind))
                up_sampling = lateral + up_sampling
                lateral = up_sampling
            else:
                up_sampling = lateral
            pyramid_laterals.append(lateral)
            lateral = conv2d_fixed_padding(inputs=lateral, filters=256, kernel_size=1, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv2_p{}'.format(pyramid_len - ind))
            lateral = tf.nn.relu(lateral, name='relu2_p{}'.format(pyramid_len - ind))
            outputs = conv2d_fixed_padding(inputs=lateral, filters=output_channals, kernel_size=3, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='conv_heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 2, 3, 1], name='output_trans_p{}'.format(pyramid_len - ind))
            outputs = tf.image.resize_bilinear(outputs, [heatmap_size, heatmap_size], name='heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2], name='heatmap_trans_inv_p{}'.format(pyramid_len - ind))
            pyramid_heatmaps.append(outputs)
    with tf.variable_scope('global_net', 'global_net', values=pyramid_laterals):
        global_pyramids = []
        for ind, lateral in enumerate(pyramid_laterals):
            inputs = lateral
            for bottleneck_ind in range(pyramid_len - ind - 1):
                inputs = global_net_bottleneck_block(inputs, 128, istraining, data_format, name='global_net_bottleneck_{}_p{}'.format(bottleneck_ind, pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(inputs, [0, 2, 3, 1], name='global_output_trans_p{}'.format(pyramid_len - ind))
            else:
                outputs = inputs
            outputs = tf.image.resize_bilinear(outputs, [heatmap_size, heatmap_size], name='global_heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2], name='global_heatmap_trans_inv_p{}'.format(pyramid_len - ind))
            global_pyramids.append(outputs)
        concat_pyramids = tf.concat(global_pyramids, 1 if data_format == 'channels_first' else 3, name='concat')

        def projection_shortcut(inputs):
            return conv2d_fixed_padding(inputs=inputs, filters=256, kernel_size=1, strides=1, data_format=data_format, name='shortcut')
        outputs = global_net_bottleneck_block(concat_pyramids, 128, istraining, data_format, projection_shortcut=projection_shortcut, name='global_concat_bottleneck')
        outputs = conv2d_fixed_padding(inputs=outputs, filters=output_channals, kernel_size=3, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='conv_heatmap')
    return pyramid_heatmaps + [outputs]

def head_xt_cascaded_pyramid_net(inputs, output_channals, heatmap_size, istraining, data_format):
    end_points = sext_cpn_backbone(inputs, istraining, data_format)
    pyramid_len = len(end_points)
    up_sampling = None
    pyramid_heatmaps = []
    pyramid_laterals = []
    with tf.variable_scope('feature_pyramid', 'feature_pyramid', values=end_points):
        for ind, pyramid in enumerate(reversed(end_points)):
            inputs = conv2d_fixed_padding(inputs=pyramid, filters=256, kernel_size=1, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv1_p{}'.format(pyramid_len - ind))
            lateral = tf.nn.relu(inputs, name='relu1_p{}'.format(pyramid_len - ind))
            if up_sampling is not None:
                if data_format == 'channels_first':
                    up_sampling = tf.transpose(up_sampling, [0, 2, 3, 1], name='trans_p{}'.format(pyramid_len - ind))
                up_sampling = tf.image.resize_bilinear(up_sampling, tf.shape(up_sampling)[-3:-1] * 2, name='upsample_p{}'.format(pyramid_len - ind))
                if data_format == 'channels_first':
                    up_sampling = tf.transpose(up_sampling, [0, 3, 1, 2], name='trans_inv_p{}'.format(pyramid_len - ind))
                up_sampling = conv2d_fixed_padding(inputs=up_sampling, filters=256, kernel_size=1, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='up_conv_p{}'.format(pyramid_len - ind))
                up_sampling = lateral + up_sampling
                lateral = up_sampling
            else:
                up_sampling = lateral
            pyramid_laterals.append(lateral)
            lateral = conv2d_fixed_padding(inputs=lateral, filters=256, kernel_size=1, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv2_p{}'.format(pyramid_len - ind))
            lateral = tf.nn.relu(lateral, name='relu2_p{}'.format(pyramid_len - ind))
            outputs = conv2d_fixed_padding(inputs=lateral, filters=output_channals, kernel_size=3, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='conv_heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 2, 3, 1], name='output_trans_p{}'.format(pyramid_len - ind))
            outputs = tf.image.resize_bilinear(outputs, [heatmap_size, heatmap_size], name='heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2], name='heatmap_trans_inv_p{}'.format(pyramid_len - ind))
            pyramid_heatmaps.append(outputs)
    with tf.variable_scope('global_net', 'global_net', values=pyramid_laterals):
        global_pyramids = []
        for ind, lateral in enumerate(pyramid_laterals):
            inputs = lateral
            for bottleneck_ind in range(pyramid_len - ind - 1):
                inputs = global_net_sext_bottleneck_block(inputs, 128, istraining, data_format, name_prefix='global_net_bottleneck_{}_p{}'.format(bottleneck_ind, pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(inputs, [0, 2, 3, 1], name='global_output_trans_p{}'.format(pyramid_len - ind))
            else:
                outputs = inputs
            outputs = tf.image.resize_bilinear(outputs, [heatmap_size, heatmap_size], name='global_heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2], name='global_heatmap_trans_inv_p{}'.format(pyramid_len - ind))
            global_pyramids.append(outputs)
        concat_pyramids = tf.concat(global_pyramids, 1 if data_format == 'channels_first' else 3, name='concat')
        outputs = global_net_sext_bottleneck_block(concat_pyramids, 128, istraining, data_format, need_reduce=True, name_prefix='global_concat_bottleneck')
        outputs = conv2d_fixed_padding(inputs=outputs, filters=output_channals, kernel_size=3, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='conv_heatmap')
    return pyramid_heatmaps + [outputs]

def cascaded_pyramid_net(inputs, output_channals, heatmap_size, istraining, data_format, net_depth=50):
    end_points = sext_cpn_backbone(inputs, istraining, data_format, net_depth=net_depth)
    pyramid_len = len(end_points)
    up_sampling = None
    pyramid_heatmaps = []
    pyramid_laterals = []
    with tf.variable_scope('feature_pyramid', 'feature_pyramid', values=end_points):
        for ind, pyramid in enumerate(reversed(end_points)):
            inputs = conv2d_fixed_padding(inputs=pyramid, filters=256, kernel_size=1, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv1_p{}'.format(pyramid_len - ind + 1))
            lateral = tf.nn.relu(inputs, name='relu1_p{}'.format(pyramid_len - ind + 1))
            if up_sampling is not None:
                if ind > pyramid_len - 2:
                    if data_format == 'channels_first':
                        up_sampling = tf.transpose(up_sampling, [0, 2, 3, 1], name='trans_p{}'.format(pyramid_len - ind + 1))
                    up_sampling = tf.image.resize_bilinear(up_sampling, tf.shape(up_sampling)[-3:-1] * 2, name='upsample_p{}'.format(pyramid_len - ind + 1))
                    if data_format == 'channels_first':
                        up_sampling = tf.transpose(up_sampling, [0, 3, 1, 2], name='trans_inv_p{}'.format(pyramid_len - ind + 1))
                    up_sampling = conv2d_fixed_padding(inputs=up_sampling, filters=256, kernel_size=1, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='up_conv_p{}'.format(pyramid_len - ind + 1))
                up_sampling = lateral + up_sampling
                lateral = up_sampling
            else:
                up_sampling = lateral
            pyramid_laterals.append(lateral)
            lateral = conv2d_fixed_padding(inputs=lateral, filters=256, kernel_size=1, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv2_p{}'.format(pyramid_len - ind + 1))
            lateral = tf.nn.relu(lateral, name='relu2_p{}'.format(pyramid_len - ind + 1))
            outputs = conv2d_fixed_padding(inputs=lateral, filters=output_channals, kernel_size=3, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='conv_heatmap_p{}'.format(pyramid_len - ind + 1))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 2, 3, 1], name='output_trans_p{}'.format(pyramid_len - ind + 1))
            outputs = tf.image.resize_bilinear(outputs, [heatmap_size, heatmap_size], name='heatmap_p{}'.format(pyramid_len - ind + 1))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2], name='heatmap_trans_inv_p{}'.format(pyramid_len - ind + 1))
            pyramid_heatmaps.append(outputs)
    with tf.variable_scope('global_net', 'global_net', values=pyramid_laterals):
        global_pyramids = []
        for ind, lateral in enumerate(pyramid_laterals):
            inputs = lateral
            for bottleneck_ind in range(pyramid_len - ind - 1):
                inputs = global_net_bottleneck_block(inputs, 128, istraining, data_format, name='global_net_bottleneck_{}_p{}'.format(bottleneck_ind, pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(inputs, [0, 2, 3, 1], name='global_output_trans_p{}'.format(pyramid_len - ind))
            else:
                outputs = inputs
            outputs = tf.image.resize_bilinear(outputs, [heatmap_size, heatmap_size], name='global_heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2], name='global_heatmap_trans_inv_p{}'.format(pyramid_len - ind))
            global_pyramids.append(outputs)
        concat_pyramids = tf.concat(global_pyramids, 1 if data_format == 'channels_first' else 3, name='concat')

        def projection_shortcut(inputs):
            return conv2d_fixed_padding(inputs=inputs, filters=256, kernel_size=1, strides=1, data_format=data_format, name='shortcut')
        outputs = global_net_bottleneck_block(concat_pyramids, 128, istraining, data_format, projection_shortcut=projection_shortcut, name='global_concat_bottleneck')
        outputs = conv2d_fixed_padding(inputs=outputs, filters=output_channals, kernel_size=3, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='conv_heatmap')
    return pyramid_heatmaps + [outputs]

def head_xt_cascaded_pyramid_net(inputs, output_channals, heatmap_size, istraining, data_format):
    end_points = sext_cpn_backbone(inputs, istraining, data_format)
    pyramid_len = len(end_points)
    up_sampling = None
    pyramid_heatmaps = []
    pyramid_laterals = []
    with tf.variable_scope('feature_pyramid', 'feature_pyramid', values=end_points):
        for ind, pyramid in enumerate(reversed(end_points)):
            inputs = conv2d_fixed_padding(inputs=pyramid, filters=256, kernel_size=1, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv1_p{}'.format(pyramid_len - ind + 1))
            lateral = tf.nn.relu(inputs, name='relu1_p{}'.format(pyramid_len - ind + 1))
            if up_sampling is not None:
                if ind > pyramid_len - 2:
                    if data_format == 'channels_first':
                        up_sampling = tf.transpose(up_sampling, [0, 2, 3, 1], name='trans_p{}'.format(pyramid_len - ind + 1))
                    up_sampling = tf.image.resize_bilinear(up_sampling, tf.shape(up_sampling)[-3:-1] * 2, name='upsample_p{}'.format(pyramid_len - ind + 1))
                    if data_format == 'channels_first':
                        up_sampling = tf.transpose(up_sampling, [0, 3, 1, 2], name='trans_inv_p{}'.format(pyramid_len - ind + 1))
                    up_sampling = conv2d_fixed_padding(inputs=up_sampling, filters=256, kernel_size=1, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='up_conv_p{}'.format(pyramid_len - ind + 1))
                up_sampling = lateral + up_sampling
                lateral = up_sampling
            else:
                up_sampling = lateral
            pyramid_laterals.append(lateral)
            lateral = conv2d_fixed_padding(inputs=lateral, filters=256, kernel_size=1, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv2_p{}'.format(pyramid_len - ind + 1))
            lateral = tf.nn.relu(lateral, name='relu2_p{}'.format(pyramid_len - ind + 1))
            outputs = conv2d_fixed_padding(inputs=lateral, filters=output_channals, kernel_size=3, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='conv_heatmap_p{}'.format(pyramid_len - ind + 1))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 2, 3, 1], name='output_trans_p{}'.format(pyramid_len - ind + 1))
            outputs = tf.image.resize_bilinear(outputs, [heatmap_size, heatmap_size], name='heatmap_p{}'.format(pyramid_len - ind + 1))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2], name='heatmap_trans_inv_p{}'.format(pyramid_len - ind + 1))
            pyramid_heatmaps.append(outputs)
    with tf.variable_scope('global_net', 'global_net', values=pyramid_laterals):
        global_pyramids = []
        for ind, lateral in enumerate(pyramid_laterals):
            inputs = lateral
            for bottleneck_ind in range(pyramid_len - ind - 1):
                inputs = global_net_sext_bottleneck_block(inputs, 128, istraining, data_format, name_prefix='global_net_bottleneck_{}_p{}'.format(bottleneck_ind, pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(inputs, [0, 2, 3, 1], name='global_output_trans_p{}'.format(pyramid_len - ind))
            else:
                outputs = inputs
            outputs = tf.image.resize_bilinear(outputs, [heatmap_size, heatmap_size], name='global_heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2], name='global_heatmap_trans_inv_p{}'.format(pyramid_len - ind))
            global_pyramids.append(outputs)
        concat_pyramids = tf.concat(global_pyramids, 1 if data_format == 'channels_first' else 3, name='concat')
        outputs = global_net_sext_bottleneck_block(concat_pyramids, 128, istraining, data_format, need_reduce=True, name_prefix='global_concat_bottleneck')
        outputs = conv2d_fixed_padding(inputs=outputs, filters=output_channals, kernel_size=3, strides=1, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='conv_heatmap')
    return pyramid_heatmaps + [outputs]

