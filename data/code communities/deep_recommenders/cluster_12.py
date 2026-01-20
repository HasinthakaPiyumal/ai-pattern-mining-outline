# Cluster 12

class FM(object):
    """
    Factorization Machine
    """

    def __init__(self, indicator_columns, embedding_columns):
        self._indicator_columns = indicator_columns
        self._embedding_columns = embedding_columns

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, features):
        with tf.variable_scope('linear'):
            linear_outputs = tf.feature_column.linear_model(features, self._indicator_columns)
        with tf.variable_scope('factorized'):
            self.embeddings = []
            for embedding_column in self._embedding_columns:
                feature_name = embedding_column.name.replace('_embedding', '')
                feature = {feature_name: features.get(feature_name)}
                embedding = tf.feature_column.input_layer(feature, embedding_column)
                self.embeddings.append(embedding)
            stack_embeddings = tf.stack(self.embeddings, axis=1)
            factorized_outputs = fm(stack_embeddings)
        return linear_outputs + factorized_outputs

def fm(x):
    """
    Second order interaction in Factorization Machine
    :param x:
        type: tf.Tensor
        shape: (batch_size, num_features, embedding_dim)
    :return: tf.Tensor
    """
    if x.shape.rank != 3:
        raise ValueError('The rank of `x` should be 3. Got rank = {}.'.format(x.shape.rank))
    sum_square = tf.square(tf.reduce_sum(x, axis=1))
    square_sum = tf.reduce_sum(tf.square(x), axis=1)
    return 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), axis=1, keepdims=True)

class FNN(object):

    def __init__(self, indicator_columns, embedding_columns, warmup_from_fm, dnn_units, dnn_activation=tf.nn.relu, dnn_batch_normalization=False, dnn_dropout=None, **dnn_kwargs):
        self._indicator_columns = indicator_columns
        self._embedding_columns = embedding_columns
        self._warmup_from_fm = warmup_from_fm
        self._dnn_hidden_units = dnn_units
        self._dnn_activation = dnn_activation
        self._dnn_batch_norm = dnn_batch_normalization
        self._dnn_dropout = dnn_dropout
        self._dnn_kwargs = dnn_kwargs

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def warm_up(self):
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.load(sess, ['serve'], self._warmup_from_fm)
            linear_variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, 'linear')
            linear_variables = {var.name.split('/')[2].replace('_indicator', '') if 'bias' not in var.name else 'bias': sess.run(var) for var in linear_variables}
            factorized_variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, 'factorized')
            factorized_variables = {var.name.split('/')[2].replace('_embedding', ''): sess.run(var) for var in factorized_variables}
            return (linear_variables, factorized_variables)

    def call(self, features):
        linear_variables, factorized_variables = self.warm_up()
        weights = []
        for indicator_column in self._indicator_columns:
            feature_name = indicator_column.categorical_column.key
            feature = {feature_name: features.get(feature_name)}
            sparse = tf.feature_column.input_layer(feature, indicator_column)
            weights_initializer = tf.constant_initializer(linear_variables.get(feature_name))
            weight = tf.layers.dense(sparse, units=1, use_bias=False, kernel_initializer=weights_initializer)
            weights.append(weight)
        concat_weights = tf.concat(weights, axis=1)
        embeddings = []
        for embedding_column in self._embedding_columns:
            feature_name = embedding_column.categorical_column.key
            feature = {feature_name: features.get(feature_name)}
            embedding_column = tf.feature_column.embedding_column(embedding_column.categorical_column, embedding_column.dimension, initializer=tf.constant_initializer(factorized_variables.get(feature_name)))
            embedding = tf.feature_column.input_layer(feature, embedding_column)
            embeddings.append(embedding)
        concat_embeddings = tf.concat(embeddings, axis=1)
        bias = tf.expand_dims(linear_variables.get('bias'), axis=0)
        bias = tf.tile(bias, [tf.shape(concat_weights)[0], 1])
        dnn_inputs = tf.concat([bias, concat_weights, concat_embeddings], axis=1)
        outputs = dnn(dnn_inputs, self._dnn_hidden_units + [1], activation=self._dnn_activation, batch_normalization=self._dnn_batch_norm, dropout=self._dnn_dropout, **self._dnn_kwargs)
        return tf.nn.sigmoid(outputs)

def dnn(inputs, hidden_units, activation=tf.nn.relu, batch_normalization=False, dropout=None, **kwargs):
    x = inputs
    for units in hidden_units[:-1]:
        x = tf.layers.dense(x, units, activation, **kwargs)
        if batch_normalization is True:
            x = tf.nn.batch_normalization(x)
        if dropout is not None:
            x = tf.nn.dropout(x, rate=dropout)
    outputs = tf.layers.dense(x, hidden_units[-1], **kwargs)
    return outputs

class WDL(object):

    def __init__(self, indicator_columns, embedding_columns, dnn_units, dnn_activation=tf.nn.relu, dnn_batch_normalization=False, dnn_dropout=None, **dnn_kwargs):
        self._indicator_columns = indicator_columns
        self._embedding_columns = embedding_columns
        self._dnn_hidden_units = dnn_units
        self._dnn_activation = dnn_activation
        self._dnn_batch_norm = dnn_batch_normalization
        self._dnn_dropout = dnn_dropout
        self._dnn_kwargs = dnn_kwargs

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, features):
        with tf.variable_scope('wide'):
            linear_outputs = tf.feature_column.linear_model(features, self._indicator_columns)
        with tf.variable_scope('deep'):
            embeddings = []
            for embedding_column in self._embedding_columns:
                feature_name = embedding_column.name.replace('_embedding', '')
                feature = {feature_name: features.get(feature_name)}
                embedding = tf.feature_column.input_layer(feature, embedding_column)
                embeddings.append(embedding)
            concat_embeddings = tf.concat(embeddings, axis=1)
            dnn_outputs = dnn(concat_embeddings, self._dnn_hidden_units + [1], activation=self._dnn_activation, batch_normalization=self._dnn_batch_norm, dropout=self._dnn_dropout, **self._dnn_kwargs)
        return tf.nn.sigmoid(linear_outputs + dnn_outputs)

class DeepFM(object):

    def __init__(self, indicator_columns, embedding_columns, dnn_units, dnn_activation=tf.nn.relu, dnn_batch_normalization=False, dnn_dropout=None, **dnn_kwargs):
        self._indicator_columns = indicator_columns
        self._embedding_columns = embedding_columns
        self._dnn_hidden_units = dnn_units
        self._dnn_activation = dnn_activation
        self._dnn_batch_norm = dnn_batch_normalization
        self._dnn_dropout = dnn_dropout
        self._dnn_kwargs = dnn_kwargs

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, features):
        fm = FM(self._indicator_columns, self._embedding_columns)
        fm_outputs = fm(features)
        concat_embeddings = tf.concat(fm.embeddings, axis=1)
        dnn_outputs = dnn(concat_embeddings, self._dnn_hidden_units + [1], activation=self._dnn_activation, batch_normalization=self._dnn_batch_norm, dropout=self._dnn_dropout, **self._dnn_kwargs)
        return tf.nn.sigmoid(fm_outputs + dnn_outputs)

