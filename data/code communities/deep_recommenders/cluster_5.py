# Cluster 5

def model_fn(features, labels, mode):
    indicator_columns, embedding_columns = build_columns()
    crossed_product_columns = cross_product_transformation()
    outputs = WDL(indicator_columns + crossed_product_columns, embedding_columns, [64, 16])(features)
    predictions = {'predictions': outputs}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    loss = tf.losses.log_loss(labels, outputs)
    metrics = {'auc': tf.metrics.auc(labels, outputs)}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    wide_variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, 'wide')
    wide_optimizer = tf.train.FtrlOptimizer(0.01, l1_regularization_strength=0.5)
    wide_train_op = wide_optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), var_list=wide_variables)
    deep_variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, 'deep')
    deep_optimizer = tf.train.AdamOptimizer(0.01)
    deep_train_op = deep_optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), var_list=deep_variables)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(update_ops, wide_train_op, deep_train_op)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def cross_product_transformation():
    crossed_columns = [tf.feature_column.crossed_column(['user_gender', 'user_age'], 14), tf.feature_column.crossed_column(['user_gender', 'user_occupation'], 40), tf.feature_column.crossed_column(['user_age', 'user_occupation'], 140)]
    crossed_product_columns = [tf.feature_column.indicator_column(c) for c in crossed_columns]
    return crossed_product_columns

