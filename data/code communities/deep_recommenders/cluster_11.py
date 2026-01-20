# Cluster 11

class HardNegativeMining(tf.keras.layers.Layer):
    """Hard Negative"""

    def __init__(self, num_hard_negatives: int, **kwargs):
        super(HardNegativeMining, self).__init__(**kwargs)
        self._num_hard_negatives = num_hard_negatives

    def call(self, logits: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        num_sampled = tf.minimum(self._num_hard_negatives + 1, tf.shape(logits)[1])
        _, indices = tf.nn.top_k(logits + labels * MAX_FLOAT, k=num_sampled, sorted=False)
        logits = _gather_elements_along_row(logits, indices)
        labels = _gather_elements_along_row(labels, indices)
        return (logits, labels)

def _gather_elements_along_row(data: tf.Tensor, column_indices: tf.Tensor) -> tf.Tensor:
    """与factorized_top_k中_take_long_axis相同"""
    with tf.control_dependencies([tf.assert_equal(tf.shape(data)[0], tf.shape(column_indices)[0])]):
        num_row = tf.shape(data)[0]
        num_column = tf.shape(data)[1]
        num_gathered = tf.shape(column_indices)[1]
        row_indices = tf.tile(tf.expand_dims(tf.range(num_row), -1), [1, num_gathered])
        flat_data = tf.reshape(data, [-1])
        flat_indices = tf.reshape(row_indices * num_column + column_indices, [-1])
        return tf.reshape(tf.gather(flat_data, flat_indices), [num_row, num_gathered])

