# Cluster 21

def bottleneck_block_v2(inputs, in_filters, out_filters, is_training, data_format, name=None):
    with tf.variable_scope(name, 'bottleneck_block', values=[inputs]):
        shortcut = inputs
        inputs = batch_norm_relu(inputs, is_training, data_format, name='bn_relu_1')
        if in_filters != out_filters:
            shortcut = conv2d_fixed_padding(inputs=inputs, filters=out_filters, kernel_size=1, strides=1, data_format=data_format, name='skip')
        inputs = conv2d_fixed_padding(inputs=inputs, filters=out_filters // 2, kernel_size=1, strides=1, data_format=data_format, name='1x1_down')
        inputs = batch_norm_relu(inputs, is_training, data_format, name='bn_relu_2')
        inputs = conv2d_fixed_padding(inputs=inputs, filters=out_filters // 2, kernel_size=3, strides=1, data_format=data_format, name='3x3_conv')
        inputs = batch_norm_relu(inputs, is_training, data_format, name='bn_relu_3')
        inputs = conv2d_fixed_padding(inputs=inputs, filters=out_filters, kernel_size=1, strides=1, data_format=data_format, name='1x1_up')
        return tf.add(inputs, shortcut, name='elem_add')

def batch_norm_relu(inputs, is_training, data_format, name=None):
    """Performs a batch normalization followed by a ReLU."""
    with tf.variable_scope(name, 'batch_norm_relu', values=[inputs]):
        inputs = tf.layers.batch_normalization(inputs=inputs, axis=1 if data_format == 'channels_first' else 3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=is_training, fused=_USE_FUSED_BN, name='batch_normalization')
        inputs = tf.nn.relu(inputs, name='relu')
        return inputs

def bottleneck_block_v1(inputs, in_filters, out_filters, is_training, data_format, name=None):
    with tf.variable_scope(name, 'bottleneck_block_v1', values=[inputs]):
        shortcut = inputs
        if in_filters != out_filters:
            shortcut = conv2d_fixed_padding(inputs=shortcut, filters=out_filters, kernel_size=1, strides=1, data_format=data_format, name='skip')
            shortcut = batch_norm(shortcut, is_training, data_format, name='skip_bn')
        inputs = conv2d_fixed_padding(inputs=inputs, filters=out_filters // 2, kernel_size=1, strides=1, data_format=data_format, name='1x1_down')
        inputs = batch_norm_relu(inputs, is_training, data_format, name='bn_relu_1')
        inputs = conv2d_fixed_padding(inputs=inputs, filters=out_filters // 2, kernel_size=3, strides=1, data_format=data_format, name='3x3_conv')
        inputs = batch_norm_relu(inputs, is_training, data_format, name='bn_relu_2')
        inputs = conv2d_fixed_padding(inputs=inputs, filters=out_filters, kernel_size=1, strides=1, data_format=data_format, name='1x1_up')
        inputs = batch_norm(inputs, is_training, data_format, name='up_bn')
        return tf.nn.relu(tf.add(inputs, shortcut, name='elem_add'), name='relu')

def hourglass(inputs, filters, is_training, data_format, deep_index=1, num_modules=1, name=None):
    with tf.variable_scope(name, 'hourglass_unit', values=[inputs]):
        upchannal1 = dozen_bottleneck_blocks(inputs, filters, filters, num_modules, is_training, data_format, name='up_{}')
        downchannal1 = tf.layers.max_pooling2d(inputs=inputs, pool_size=2, strides=2, padding='valid', data_format=data_format, name='down_pool')
        downchannal1 = dozen_bottleneck_blocks(downchannal1, filters, filters, num_modules, is_training, data_format, name='down1_{}')
        if deep_index > 1:
            downchannal2 = hourglass(downchannal1, filters, is_training, data_format, deep_index=deep_index - 1, num_modules=num_modules, name='inner_{}'.format(deep_index))
        else:
            downchannal2 = dozen_bottleneck_blocks(downchannal1, filters, filters, num_modules, is_training, data_format, name='down2_{}')
        downchannal3 = dozen_bottleneck_blocks(downchannal2, filters, filters, num_modules, is_training, data_format, name='down3_{}')
        if data_format == 'channels_first':
            downchannal3 = tf.transpose(downchannal3, [0, 2, 3, 1], name='trans')
        input_shape = tf.shape(downchannal3)[-3:-1] * 2
        upchannal2 = tf.image.resize_bilinear(downchannal3, input_shape, name='resize')
        if data_format == 'channels_first':
            upchannal2 = tf.transpose(upchannal2, [0, 3, 1, 2], name='trans_inv')
        return tf.add(upchannal1, upchannal2, name='elem_add')

def dozen_bottleneck_blocks(inputs, in_filters, out_filters, num_modules, is_training, data_format, name=None):
    for m in range(num_modules):
        inputs = bottleneck_block(inputs, in_filters, out_filters, is_training, data_format, name=None if name is None else name.format(m))
    return inputs

def create_model(inputs, num_stack, feat_channals, output_channals, num_modules, is_training, data_format):
    with tf.variable_scope('precede', values=[inputs]):
        inputs = conv2d_fixed_padding(inputs=inputs, filters=64, kernel_size=7, strides=2, data_format=data_format, kernel_initializer=conv_bn_initializer_to_use, name='conv_7x7')
        inputs = batch_norm_relu(inputs, is_training, data_format, name='inputs_bn')
        inputs = bottleneck_block(inputs, 64, 128, is_training, data_format, name='residual1')
        inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=2, strides=2, padding='valid', data_format=data_format, name='pool')
        inputs = bottleneck_block(inputs, 128, 128, is_training, data_format, name='residual2')
        inputs = bottleneck_block(inputs, 128, feat_channals, is_training, data_format, name='residual3')
    hg_inputs = inputs
    outputs_list = []
    for stack_index in range(num_stack):
        hg = hourglass(hg_inputs, feat_channals, is_training, data_format, deep_index=4, num_modules=num_modules, name='stack_{}/hg'.format(stack_index))
        hg = dozen_bottleneck_blocks(hg, feat_channals, feat_channals, num_modules, is_training, data_format, name='stack_{}/'.format(stack_index) + 'output_{}')
        output_scores = conv2d_fixed_padding(inputs=hg, filters=feat_channals, kernel_size=1, strides=1, data_format=data_format, name='stack_{}/output_1x1'.format(stack_index))
        output_scores = batch_norm_relu(output_scores, is_training, data_format, name='stack_{}/output_bn'.format(stack_index))
        heatmap = tf.layers.conv2d(inputs=output_scores, filters=output_channals, kernel_size=1, strides=1, padding='same', use_bias=True, activation=None, kernel_initializer=initializer_to_use(), bias_initializer=tf.zeros_initializer(), data_format=data_format, name='hg_heatmap/stack_{}/heatmap_1x1'.format(stack_index))
        outputs_list.append(heatmap)
        if stack_index < num_stack - 1:
            output_scores_ = tf.layers.conv2d(inputs=output_scores, filters=feat_channals, kernel_size=1, strides=1, padding='same', use_bias=True, activation=None, kernel_initializer=initializer_to_use(), bias_initializer=tf.zeros_initializer(), data_format=data_format, name='stack_{}/remap_outputs'.format(stack_index))
            heatmap_ = tf.layers.conv2d(inputs=heatmap, filters=feat_channals, kernel_size=1, strides=1, padding='same', use_bias=True, activation=None, kernel_initializer=initializer_to_use(), bias_initializer=tf.zeros_initializer(), data_format=data_format, name='hg_heatmap/stack_{}/remap_heatmap'.format(stack_index))
            fused_heatmap = tf.add(output_scores_, heatmap_, 'stack_{}/fused_heatmap'.format(stack_index))
            hg_inputs = tf.add(hg_inputs, fused_heatmap, 'stack_{}/next_inputs'.format(stack_index))
    return outputs_list

