# Cluster 16

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, kernel_initializer=tf.glorot_uniform_initializer, name=None):
    """Strided 2-D convolution with explicit padding."""
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='SAME' if strides == 1 else 'VALID', use_bias=False, kernel_initializer=kernel_initializer(), data_format=data_format, name=name)

def _bottleneck_block_v1(inputs, filters, training, projection_shortcut, strides, data_format):
    """A single block for ResNet v1, with a bottleneck.

    Similar to _building_block_v1(), except using the "bottleneck" blocks
    described in:
      Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training, data_format=data_format)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)
    return inputs

def batch_norm(inputs, training, data_format, name=None):
    """Performs a batch normalization using a standard set of parameters."""
    return tf.layers.batch_normalization(inputs=inputs, axis=1 if data_format == 'channels_first' else 3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=training, name=name, fused=_USE_FUSED_BN)

def projection_shortcut(inputs):
    return conv2d_fixed_padding(inputs=inputs, filters=256, kernel_size=1, strides=1, data_format=data_format, name='shortcut')

def _bottleneck_block_v1(inputs, filters, training, projection_shortcut, strides, data_format):
    """A single block for ResNet v1, with a bottleneck.

    Similar to _building_block_v1(), except using the "bottleneck" blocks
    described in:
      Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training, data_format=data_format)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)
    return inputs

def projection_shortcut(inputs):
    return conv2d_fixed_padding(inputs=inputs, filters=256, kernel_size=1, strides=1, data_format=data_format, name='shortcut')

def _dilated_bottleneck_block_v1(inputs, filters, training, projection_shortcut, data_format):
    shortcut = inputs
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training, data_format=data_format)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=3, strides=1, dilation_rate=(2, 2), padding='SAME', use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(), data_format=data_format, name=None)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)
    return inputs

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

def _bottleneck_block_v1(inputs, filters, training, projection_shortcut, strides, data_format):
    """A single block for ResNet v1, with a bottleneck.

    Similar to _building_block_v1(), except using the "bottleneck" blocks
    described in:
      Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training, data_format=data_format)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)
    return inputs

def projection_shortcut(inputs):
    return conv2d_fixed_padding(inputs=inputs, filters=256, kernel_size=1, strides=1, data_format=data_format, name='shortcut')

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

def projection_shortcut(inputs):
    return conv2d_fixed_padding(inputs=inputs, filters=256, kernel_size=1, strides=1, data_format=data_format, name='shortcut')

