# Cluster 5

def build_model(_indices, _values, _values2, _shape, _y, _ind, keep_probs, feature_cnt, field_cnt, params):
    eta = tf.constant(params['eta'])
    _x = tf.SparseTensor(_indices, _values, _shape)
    _xx = tf.SparseTensor(_indices, _values2, _shape)
    model_params = []
    tmp = []
    init_value = params['init_value']
    dim = params['dim']
    layer_sizes = params['layer_sizes']
    w_linear = tf.Variable(tf.truncated_normal([feature_cnt, 1], stddev=init_value, mean=0), name='w_linear', dtype=tf.float32)
    bias = tf.Variable(tf.truncated_normal([1], stddev=init_value, mean=0), name='bias')
    model_params.append(bias)
    model_params.append(w_linear)
    preds = bias
    preds += tf.sparse_tensor_dense_matmul(_x, w_linear, name='contr_from_linear')
    w_fm = tf.Variable(tf.truncated_normal([feature_cnt, dim], stddev=init_value / math.sqrt(float(dim)), mean=0), name='w_fm', dtype=tf.float32)
    model_params.append(w_fm)
    if params['is_use_fm_part']:
        preds = preds + 0.5 * tf.reduce_sum(tf.pow(tf.sparse_tensor_dense_matmul(_x, w_fm), 2) - tf.sparse_tensor_dense_matmul(_xx, tf.pow(w_fm, 2)), 1, keep_dims=True)
    if params['is_use_dnn_part']:
        w_fm_nn_input = tf.reshape(tf.gather(w_fm, _ind) * tf.expand_dims(_values, 1), [-1, field_cnt * dim])
        print(w_fm_nn_input.shape)
        hidden_nn_layers = []
        hidden_nn_layers.append(w_fm_nn_input)
        last_layer_size = field_cnt * dim
        layer_idx = 0
        w_nn_params = []
        b_nn_params = []
        for layer_size in layer_sizes:
            cur_w_nn_layer = tf.Variable(tf.truncated_normal([last_layer_size, layer_size], stddev=init_value / math.sqrt(float(10)), mean=0), name='w_nn_layer' + str(layer_idx), dtype=tf.float32)
            cur_b_nn_layer = tf.Variable(tf.truncated_normal([layer_size], stddev=init_value, mean=0), name='b_nn_layer' + str(layer_idx))
            cur_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[layer_idx], cur_w_nn_layer, cur_b_nn_layer)
            cur_hidden_nn_layer = tf.nn.dropout(cur_hidden_nn_layer, keep_probs[layer_idx])
            if params['activations'][layer_idx] == 'tanh':
                cur_hidden_nn_layer = tf.nn.tanh(cur_hidden_nn_layer)
            elif params['activations'][layer_idx] == 'sigmoid':
                cur_hidden_nn_layer = tf.nn.sigmoid(cur_hidden_nn_layer)
            elif params['activations'][layer_idx] == 'relu':
                cur_hidden_nn_layer = tf.nn.relu(cur_hidden_nn_layer)
            hidden_nn_layers.append(cur_hidden_nn_layer)
            layer_idx += 1
            last_layer_size = layer_size
            model_params.append(cur_w_nn_layer)
            model_params.append(cur_b_nn_layer)
            w_nn_params.append(cur_w_nn_layer)
            b_nn_params.append(cur_b_nn_layer)
        w_nn_output = tf.Variable(tf.truncated_normal([last_layer_size, 1], stddev=init_value, mean=0), name='w_nn_output', dtype=tf.float32)
        nn_output = tf.matmul(hidden_nn_layers[-1], w_nn_output)
        model_params.append(w_nn_output)
        w_nn_params.append(w_nn_output)
        preds += nn_output
    if params['loss'] == 'cross_entropy_loss':
        error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(preds, [-1]), labels=tf.reshape(_y, [-1])))
    elif params['loss'] == 'square_loss':
        preds = tf.sigmoid(preds)
        error = tf.reduce_mean(tf.squared_difference(preds, _y))
    elif params['loss'] == 'log_loss':
        preds = tf.sigmoid(preds)
        error = tf.reduce_mean(tf.losses.log_loss(predictions=preds, labels=_y))
    lambda_w_linear = tf.constant(params['reg_w_linear'], name='lambda_w_linear')
    lambda_w_fm = tf.constant(params['reg_w_fm'], name='lambda_w_fm')
    lambda_w_nn = tf.constant(params['reg_w_nn'], name='lambda_nn_fm')
    lambda_w_l1 = tf.constant(params['reg_w_l1'], name='lambda_w_l1')
    l2_norm = tf.multiply(lambda_w_linear, tf.reduce_sum(tf.pow(w_linear, 2)))
    l2_norm += tf.multiply(lambda_w_l1, tf.reduce_sum(tf.abs(w_linear)))
    if params['is_use_fm_part'] or params['is_use_dnn_part']:
        l2_norm += tf.multiply(lambda_w_fm, tf.reduce_sum(tf.pow(w_fm, 2)))
    if params['is_use_dnn_part']:
        for i in range(len(w_nn_params)):
            l2_norm += tf.multiply(lambda_w_nn, tf.reduce_sum(tf.pow(w_nn_params[i], 2)))
        for i in range(len(b_nn_params)):
            l2_norm += tf.multiply(lambda_w_nn, tf.reduce_sum(tf.pow(b_nn_params[i], 2)))
    loss = tf.add(error, l2_norm)
    if params['optimizer'] == 'adadelta':
        train_step = tf.train.AdadeltaOptimizer(eta).minimize(loss, var_list=model_params)
    elif params['optimizer'] == 'sgd':
        train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss, var_list=model_params)
    elif params['optimizer'] == 'adam':
        train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss, var_list=model_params)
    elif params['optimizer'] == 'ftrl':
        train_step = tf.train.FtrlOptimizer(params['learning_rate']).minimize(loss, var_list=model_params)
    else:
        train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss, var_list=model_params)
    tf.summary.scalar('square_error', error)
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('linear_weights_hist', w_linear)
    if params['is_use_fm_part']:
        tf.summary.histogram('fm_weights_hist', w_fm)
    if params['is_use_dnn_part']:
        for idx in range(len(w_nn_params)):
            tf.summary.histogram('nn_layer' + str(idx) + '_weights', w_nn_params[idx])
    merged_summary = tf.summary.merge_all()
    return (train_step, loss, error, preds, merged_summary, tmp)

def single_run(dataset, rank, user_cnt, item_cnt, lr, lamb, mu, n_eopch, batch_size, is_eval_on, init_value):
    user_indices = tf.placeholder(tf.int32, [None])
    item_indices = tf.placeholder(tf.int32, [None])
    ratings = tf.placeholder(tf.float32, [None])
    train_step, square_error, loss, merged_summary = build_model(user_indices, item_indices, rank, ratings, user_cnt, item_cnt, lr, lamb, mu, init_value)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    train_writer = tf.summary.FileWriter('logs', sess.graph)
    n_instances = len(dataset.training_ratings_user)
    best_train_rmse, best_test_rmse, best_eval_rmse = (-1, -1, -1)
    best_eopch_idx = -1
    for ite in range(n_eopch):
        start = clock()
        for i in range(n_instances // batch_size):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            cur_user_indices, cur_item_indices, cur_label = (dataset.training_ratings_user[start_idx:end_idx], dataset.training_ratings_item[start_idx:end_idx], dataset.training_ratings_score[start_idx:end_idx])
            sess.run(train_step, {user_indices: cur_user_indices, item_indices: cur_item_indices, ratings: cur_label})
        error_traing = sess.run(square_error, {user_indices: dataset.training_ratings_user, item_indices: dataset.training_ratings_item, ratings: dataset.training_ratings_score})
        error_test = sess.run(square_error, {user_indices: dataset.test_ratings_user, item_indices: dataset.test_ratings_item, ratings: dataset.test_ratings_score})
        if is_eval_on:
            error_eval = sess.run(square_error, {user_indices: dataset.eval_ratings_user, item_indices: dataset.eval_ratings_item, ratings: dataset.eval_ratings_score})
        else:
            error_eval = -1
        if best_test_rmse < 0 or best_test_rmse > error_test:
            best_train_rmse, best_test_rmse, best_eval_rmse = (error_traing, error_test, error_eval)
            best_eopch_idx = ite
        elif ite - best_eopch_idx > 10:
            break
        loss_traing = sess.run(loss, {user_indices: dataset.training_ratings_user, item_indices: dataset.training_ratings_item, ratings: dataset.training_ratings_score})
        summary = sess.run(merged_summary, {user_indices: dataset.training_ratings_user, item_indices: dataset.training_ratings_item, ratings: dataset.training_ratings_score})
        train_writer.add_summary(summary, ite)
        end = clock()
        print('Iteration %d  RMSE(train): %f  RMSE(test): %f   RMSE(eval): %f   LOSS(train): %f  minutes: %f' % (ite, error_traing, error_test, error_eval, loss_traing, (end - start) / 60))
    train_writer.close()
    return (best_train_rmse, best_test_rmse, best_eval_rmse, best_eopch_idx)

def single_run(dataset, params):
    cf_dim, user_attr_rank, item_attr_rank, layer_sizes, lr, lamb, mu, n_eopch, batch_size, init_value = (params['cf_dim'], params['user_attr_rank'], params['item_attr_rank'], params['layer_sizes'], params['lr'], params['lamb'], params['mu'], params['n_eopch'], params['batch_size'], params['init_value'])
    user_cnt, user_attr_cnt = (dataset.n_user, dataset.n_user_attr)
    item_cnt, item_attr_cnt = (dataset.n_item, dataset.n_item_attr)
    W_user = tf.Variable(tf.truncated_normal([user_cnt, cf_dim], stddev=init_value / math.sqrt(float(cf_dim)), mean=0), name='user_cf_embedding', dtype=tf.float32)
    W_item = tf.Variable(tf.truncated_normal([item_cnt, cf_dim], stddev=init_value / math.sqrt(float(cf_dim)), mean=0), name='item_cf_embedding', dtype=tf.float32)
    W_user_bias = tf.concat([W_user, tf.ones((user_cnt, 1), dtype=tf.float32)], 1, name='user_cf_embedding_bias')
    W_item_bias = tf.concat([tf.ones((item_cnt, 1), dtype=tf.float32), W_item], 1, name='item_cf_embedding_bias')
    user_attr_indices, user_attr_indices_values, user_attr_indices_weights = compose_vector_for_sparse_tensor(dataset.user_attr)
    item_attr_indices, item_attr_indices_values, item_attr_indices_weights = compose_vector_for_sparse_tensor(dataset.item_attr)
    user_sp_ids = tf.SparseTensor(indices=user_attr_indices, values=user_attr_indices_values, dense_shape=[user_cnt, user_attr_cnt])
    user_sp_weights = tf.SparseTensor(indices=user_attr_indices, values=user_attr_indices_weights, dense_shape=[user_cnt, user_attr_cnt])
    item_sp_ids = tf.SparseTensor(indices=item_attr_indices, values=item_attr_indices_values, dense_shape=[item_cnt, item_attr_cnt])
    item_sp_weights = tf.SparseTensor(indices=item_attr_indices, values=item_attr_indices_weights, dense_shape=[item_cnt, item_attr_cnt])
    W_user_attr = tf.Variable(tf.truncated_normal([user_attr_cnt, user_attr_rank], stddev=init_value / math.sqrt(float(user_attr_rank)), mean=0), name='user_attr_embedding', dtype=tf.float32)
    W_item_attr = tf.Variable(tf.truncated_normal([item_attr_cnt, item_attr_rank], stddev=init_value / math.sqrt(float(item_attr_rank)), mean=0), name='item_attr_embedding', dtype=tf.float32)
    user_embeddings = tf.nn.embedding_lookup_sparse(W_user_attr, user_sp_ids, user_sp_weights, name='user_embeddings', combiner='sum')
    item_embeddings = tf.nn.embedding_lookup_sparse(W_item_attr, item_sp_ids, item_sp_weights, name='item_embeddings', combiner='sum')
    user_indices = tf.placeholder(tf.int32, [None])
    item_indices = tf.placeholder(tf.int32, [None])
    ratings = tf.placeholder(tf.float32, [None])
    user_cf_feature = tf.nn.embedding_lookup(W_user_bias, user_indices, name='user_feature')
    item_cf_feature = tf.nn.embedding_lookup(W_item_bias, item_indices, name='item_feature')
    user_attr_feature = tf.nn.embedding_lookup(user_embeddings, user_indices, name='user_feature')
    item_attr_feature = tf.nn.embedding_lookup(item_embeddings, item_indices, name='item_feature')
    train_step, square_error, loss, merged_summary = build_model(user_cf_feature, user_attr_feature, user_attr_rank, item_cf_feature, item_attr_feature, item_attr_rank, ratings, layer_sizes, W_user, W_item, W_user_attr, W_item_attr, lamb, lr, mu)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    train_writer = tf.summary.FileWriter('\\\\mlsdata\\e$\\Users\\v-lianji\\DeepRecsys\\Test\\logs', sess.graph)
    n_instances = len(dataset.training_ratings_user)
    best_train_rmse, best_test_rmse, best_eval_rmse = (-1, -1, -1)
    best_eopch_idx = -1
    for ite in range(n_eopch):
        start = clock()
        for i in range(n_instances // batch_size):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            cur_user_indices, cur_item_indices, cur_label = (dataset.training_ratings_user[start_idx:end_idx], dataset.training_ratings_item[start_idx:end_idx], dataset.training_ratings_score[start_idx:end_idx])
            sess.run(train_step, {user_indices: cur_user_indices, item_indices: cur_item_indices, ratings: cur_label})
        error_traing = sess.run(square_error, {user_indices: dataset.training_ratings_user, item_indices: dataset.training_ratings_item, ratings: dataset.training_ratings_score})
        error_test = sess.run(square_error, {user_indices: dataset.test_ratings_user, item_indices: dataset.test_ratings_item, ratings: dataset.test_ratings_score})
        error_eval = sess.run(square_error, {user_indices: dataset.eval_ratings_user, item_indices: dataset.eval_ratings_item, ratings: dataset.eval_ratings_score})
        loss_traing = sess.run(loss, {user_indices: dataset.training_ratings_user, item_indices: dataset.training_ratings_item, ratings: dataset.training_ratings_score})
        summary = sess.run(merged_summary, {user_indices: dataset.training_ratings_user, item_indices: dataset.training_ratings_item, ratings: dataset.training_ratings_score})
        train_writer.add_summary(summary, ite)
        end = clock()
        print('Iteration %d  RMSE(train): %f  RMSE(test): %f   RMSE(eval): %f   LOSS(train): %f  minutes: %f' % (ite, error_traing, error_test, error_eval, loss_traing, (end - start) / 60))
        if best_test_rmse < 0 or best_test_rmse > error_test:
            best_train_rmse, best_test_rmse, best_eval_rmse = (error_traing, error_test, error_eval)
            best_eopch_idx = ite
        elif ite - best_eopch_idx > 10:
            break
    train_writer.close()
    return (best_train_rmse, best_test_rmse, best_eval_rmse, best_eopch_idx)

