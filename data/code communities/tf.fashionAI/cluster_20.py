# Cluster 20

def detnet_cpn_backbone(inputs, istraining, data_format):
    block_strides = [1, 2, 2]
    inputs = conv2d_fixed_padding(inputs=inputs, filters=64, kernel_size=7, strides=2, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=3, strides=2, padding='SAME', data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')
    end_points = []
    for i, num_blocks in enumerate([3, 4, 6]):
        num_filters = 64 * 2 ** i
        inputs = block_layer(inputs=inputs, filters=num_filters, bottleneck=True, block_fn=_bottleneck_block_v1, blocks=num_blocks, strides=block_strides[i], training=istraining, name='block_layer{}'.format(i + 1), data_format=data_format)
        end_points.append(inputs)
    with tf.variable_scope('additional_layer', 'additional_layer', values=[inputs]):
        inputs = dilated_block_layer(inputs=inputs, filters=256, bottleneck=True, block_fn=_dilated_bottleneck_block_v1, blocks=3, training=istraining, name='block_layer{}'.format(4), data_format=data_format)
        end_points.append(inputs)
        inputs = dilated_block_layer(inputs=inputs, filters=256, bottleneck=True, block_fn=_dilated_bottleneck_block_v1, blocks=3, training=istraining, name='block_layer{}'.format(5), data_format=data_format)
        end_points.append(inputs)
    return end_points[1:]

def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides, training, name, data_format):
    """Creates one layer of blocks for the ResNet model.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the first convolution of the layer.
      bottleneck: Is the block created a bottleneck block.
      block_fn: The block to use within the model, either `building_block` or
        `bottleneck_block`.
      blocks: The number of blocks contained in the layer.
      strides: The stride to use for the first convolution of the layer. If
        greater than 1, this layer will ultimately downsample the input.
      training: Either True or False, whether we are currently training the
        model. Needed for batch norm.
      name: A string name for the tensor output of the block layer.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      The output tensor of the block layer.
    """
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, strides=strides, data_format=data_format)
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides, data_format)
    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, data_format)
    return tf.identity(inputs, name)

def dilated_block_layer(inputs, filters, bottleneck, block_fn, blocks, training, name, data_format):
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, strides=1, data_format=data_format)
    inputs = block_fn(inputs, filters, training, projection_shortcut, data_format)
    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, data_format)
    return tf.identity(inputs, name)

def cascaded_pyramid_net(inputs, output_channals, heatmap_size, istraining, data_format):
    end_points = detnet_cpn_backbone(inputs, istraining, data_format)
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

def cpn_backbone(inputs, istraining, data_format):
    block_strides = [1, 2, 2, 2]
    inputs = conv2d_fixed_padding(inputs=inputs, filters=64, kernel_size=7, strides=2, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=3, strides=2, padding='SAME', data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')
    end_points = []
    for i, num_blocks in enumerate([3, 4, 6, 3]):
        num_filters = 64 * 2 ** i
        inputs = block_layer(inputs=inputs, filters=num_filters, bottleneck=True, block_fn=_bottleneck_block_v1, blocks=num_blocks, strides=block_strides[i], training=istraining, name='block_layer{}'.format(i + 1), data_format=data_format)
        end_points.append(inputs)
    return end_points

def cascaded_pyramid_net(inputs, output_channals, heatmap_size, istraining, data_format):
    end_points = cpn_backbone(inputs, istraining, data_format)
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

