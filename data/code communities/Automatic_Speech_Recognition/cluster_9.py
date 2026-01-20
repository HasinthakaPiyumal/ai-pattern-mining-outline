# Cluster 9

def routing(u, next_num_channels, next_num_capsules, next_output_vector_len, num_iter, scope=None):
    """ Routing algorithm for capsules of two adjacent layers
    size of u: [batch_size, channels, num_capsules, output_vector_len]
    size of w: [batch_size, channels, num_capsules, next_channels, next_num_capsules, vec_len, next_vec_len]
    """
    scope = scope or 'routing'
    shape = u.get_shape()
    u = tf.reshape(u, [shape[0], shape[1], shape[2], 1, 1, shape[3], 1])
    u_ij = tf.tile(u, [1, 1, 1, next_num_channels, next_num_capsules, 1, 1])
    with tf.variable_scope(scope):
        w_shape = [1, shape[1], shape[2], next_num_channels, next_num_capsules, shape[3], next_output_vector_len]
        w = tf.get_variable('w', shape=w_shape, dtype=tf.float32)
        w = tf.tile(w, [shape[0], 1, 1, 1, 1, 1, 1])
        u_hat = tf.matmul(w, u_ij, transpose_a=True)
        u_hat = tf.reshape(u_hat, [shape[0], shape[1] * shape[2], -1, next_output_vector_len, 1])
        u_hat_without_backprop = tf.stop_gradient(u_hat, 'u_hat_without_backprop')
        b_ij = tf.constant(np.zeros([shape[0], shape[1] * shape[2], next_num_channels * next_num_capsules, 1, 1]), dtype=tf.float32)
        c_ij = tf.nn.softmax(b_ij, dim=2)
        for r in range(num_iter):
            if r != num_iter - 1:
                s_j = tf.reduce_sum(tf.multiply(c_ij, u_hat_without_backprop), axis=1, keep_dims=True)
                v_j = squashing(s_j)
                v_j = tf.tile(v_j, [1, shape[1] * shape[2], 1, 1, 1])
                b_ij = b_ij + tf.matmul(u_hat, v_j, transpose_a=True)
            else:
                s_j = tf.reduce_sum(tf.multiply(c_ij, u_hat), axis=1, keep_dims=True)
                v_j = squashing(s_j)
    return v_j

def squashing(s):
    """ Squashing function for normalization
    size of s: [batch_size, 1, next_channels*next_num_capsules, next_output_vector_len, 1]
    size of v: [batch_size, 1, next_channels*next_num_capsules, next_output_vector_len, 1]
    """
    assert s.dtype == tf.float32
    squared_s = tf.reduce_sum(tf.square(s), axis=2, keep_dims=True)
    normed_s = tf.norm(s, axis=2, keep_dims=True)
    v = squared_s / (1 + squared_s) / normed_s * s
    assert v.get_shape() == s.get_shape()
    return v

class CapsuleLayer(object):
    """ Capsule layer based on convolutional neural network
    """

    def __init__(self, num_capsules, num_channels, output_vector_len, layer_type='conv', vars_scope=None):
        self._num_capsules = num_capsules
        self._num_channels = num_channels
        self._output_vector_len = output_vector_len
        self._layer_type = layer_type
        self._vars_scope = vars_scope or 'capsule_layer'

    @property
    def num_capsules(self):
        return self._num_capsules

    @property
    def output_vector_len(self):
        return self._output_vector_len

    def __call__(self, inputX, kernel_size, strides, num_iter, with_routing=True, padding='VALID'):
        input_shape = inputX.get_shape()
        with tf.variable_scope(self._vars_scope) as scope:
            if self._layer_type == 'conv':
                kernel = tf.get_variable('conv_kernel', shape=[kernel_size[0], kernel_size[1], input_shape[-1], self._num_channels * self._num_capsules * self._output_vector_len], dtype=tf.float32)
                conv_output = tf.nn.conv2d(inputX, kernel, strides, padding)
                shape1 = conv_output.get_shape()
                capsule_output = tf.reshape(conv_output, [shape1[0], 1, -1, self._output_vector_len, 1])
                if with_routing:
                    capsule_output = routing(capsule_output, self._num_channels, self._num_capsules, self._output_vector_len, num_iter, scope)
                capsule_output = squashing(capsule_output)
                capsule_output = tf.reshape(capsule_output, [input_shape[0], self._num_capsules, self._output_vector_len, self._num_channels])
            elif self._layer_type == 'dnn':
                inputX = tf.reshape(inputX, [input_shape[0], 1, input_shape[1] * input_shape[3], input_shape[2], 1])
                capsule_output = routing(inputX, self._num_channels, self._num_capsules, self._output_vector_len, num_iter, scope)
                capsule_output = squashing(capsule_output)
                capsule_output = tf.squeeze(capsule_output, axis=[1, 4])
            else:
                capsule_output = None
        return capsule_output

