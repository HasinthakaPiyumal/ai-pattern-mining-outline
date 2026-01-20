# Cluster 8

def depth_conv2d(inputs, kernel_size, stride=1, channel_multiplier=1, padding='SAME', data_format=DATA_FORMAT_NHWC, rate=1, activation_fn=nn.relu, normalizer_fn=None, normalizer_params=None, weights_initializer=initializers.xavier_initializer(), weights_regularizer=None, biases_initializer=init_ops.zeros_initializer(), biases_regularizer=None, reuse=None, variables_collections=None, outputs_collections=None, trainable=True, scope=None):
    if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
        raise ValueError('data_format has to be either NCHW or NHWC.')
    layer_variable_getter = _build_variable_getter({'bias': 'biases', 'depthwise_kernel': 'depthwise_weights'})
    with variable_scope.variable_scope(scope, 'SeparableConv2d', [inputs], reuse=reuse, custom_getter=layer_variable_getter) as sc:
        inputs = ops.convert_to_tensor(inputs)
        df = 'channels_first' if data_format and data_format.startswith('NC') else 'channels_last'
        dtype = inputs.dtype.base_dtype
        kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
        stride_h, stride_w = utils.two_element_tuple(stride)
        num_filters_in = utils.channel_dimension(inputs.get_shape(), df, min_rank=4)
        weights_collections = utils.get_variable_collections(variables_collections, 'weights')
        depthwise_shape = [kernel_h, kernel_w, num_filters_in, channel_multiplier]
        depthwise_weights = variables.model_variable('depthwise_weights', shape=depthwise_shape, dtype=dtype, initializer=weights_initializer, regularizer=weights_regularizer, trainable=trainable, collections=weights_collections)
        strides = [1, 1, stride_h, stride_w] if data_format.startswith('NC') else [1, stride_h, stride_w, 1]
        outputs = nn.depthwise_conv2d(inputs, depthwise_weights, strides, padding, rate=utils.two_element_tuple(rate), data_format=data_format)
        num_outputs = num_filters_in
        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        elif biases_initializer is not None:
            biases_collections = utils.get_variable_collections(variables_collections, 'biases')
            biases = variables.model_variable('biases', shape=[num_outputs], dtype=dtype, initializer=biases_initializer, regularizer=biases_regularizer, trainable=trainable, collections=biases_collections)
            outputs = nn.bias_add(outputs, biases, data_format=data_format)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

def _build_variable_getter(rename=None):
    """Build a model variable getter that respects scope getter and renames."""

    def layer_variable_getter(getter, *args, **kwargs):
        kwargs['rename'] = rename
        return _model_variable_getter(getter, *args, **kwargs)
    return layer_variable_getter

