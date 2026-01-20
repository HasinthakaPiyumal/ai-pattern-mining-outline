# Cluster 17

def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, kernel_initializer=tf.glorot_uniform_initializer, name=None):
    """Strided 2-D convolution with explicit padding."""
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='SAME' if strides == 1 else 'VALID', use_bias=False, kernel_initializer=kernel_initializer(), data_format=data_format, name=name)

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, kernel_initializer=tf.glorot_uniform_initializer, name=None):
    """Strided 2-D convolution with explicit padding."""
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='SAME' if strides == 1 else 'VALID', use_bias=False, kernel_initializer=kernel_initializer(), data_format=data_format, name=name)

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, kernel_initializer=conv_bn_initializer_to_use, name=None):
    """Strided 2-D convolution with explicit padding."""
    with tf.variable_scope(name, 'fix_padding_conv', values=[inputs]):
        if strides > 1:
            inputs = fixed_padding(inputs, kernel_size, data_format)
        return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='same' if strides == 1 else 'valid', use_bias=False, kernel_initializer=kernel_initializer(), data_format=data_format, name='conv2d')

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, kernel_initializer=tf.glorot_uniform_initializer, name=None):
    """Strided 2-D convolution with explicit padding."""
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='SAME' if strides == 1 else 'VALID', use_bias=False, kernel_initializer=kernel_initializer(), data_format=data_format, name=name)

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, kernel_initializer=tf.glorot_uniform_initializer, name=None):
    """Strided 2-D convolution with explicit padding."""
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='SAME' if strides == 1 else 'VALID', use_bias=False, kernel_initializer=kernel_initializer(), data_format=data_format, name=name)

