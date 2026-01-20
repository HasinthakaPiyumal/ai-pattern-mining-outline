# Cluster 2

def single_run(feature_cnt, field_cnt, params):
    print(params)
    pre_build_data_cache_if_need(params['train_file'], feature_cnt, params['batch_size'])
    pre_build_data_cache_if_need(params['test_file'], feature_cnt, params['batch_size'])
    params['train_file'] = params['train_file'].replace('.csv', '.pkl').replace('.txt', '.pkl')
    params['test_file'] = params['test_file'].replace('.csv', '.pkl').replace('.txt', '.pkl')
    print('start single_run')
    tf.reset_default_graph()
    n_epoch = params['n_epoch']
    batch_size = params['batch_size']
    _indices = tf.placeholder(tf.int64, shape=[None, 2], name='raw_indices')
    _values = tf.placeholder(tf.float32, shape=[None], name='raw_values')
    _values2 = tf.placeholder(tf.float32, shape=[None], name='raw_values_square')
    _shape = tf.placeholder(tf.int64, shape=[2], name='raw_shape')
    _y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')
    _ind = tf.placeholder(tf.int64, shape=[None])
    _keep_probs = tf.placeholder(tf.float32, shape=[len(params['keep_probs'])], name='dropout_keep_probability')
    train_step, loss, error, preds, merged_summary, tmp = build_model(_indices, _values, _values2, _shape, _y, _ind, _keep_probs, feature_cnt, field_cnt, params)
    saver = tf.train.Saver()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    log_writer = tf.summary.FileWriter(params['log_path'], graph=sess.graph)
    glo_ite = 0
    for eopch in range(n_epoch):
        iteration = -1
        start = clock()
        time_load_data, time_sess = (0, 0)
        time_cp02 = clock()
        train_loss_per_epoch = 0
        for training_input_in_sp in load_data_cache(params['train_file']):
            time_cp01 = clock()
            time_load_data += time_cp01 - time_cp02
            iteration += 1
            glo_ite += 1
            _, cur_loss, summary, _tmp = sess.run([train_step, loss, merged_summary, tmp], feed_dict={_indices: training_input_in_sp['indices'], _values: training_input_in_sp['values'], _shape: training_input_in_sp['shape'], _y: training_input_in_sp['labels'], _values2: training_input_in_sp['values2'], _ind: training_input_in_sp['feature_indices'], _keep_probs: np.asarray(params['keep_probs'])})
            time_cp02 = clock()
            time_sess += time_cp02 - time_cp01
            train_loss_per_epoch += cur_loss
            log_writer.add_summary(summary, glo_ite)
        end = clock()
        if eopch % 5 == 0:
            model_path = params['model_path'] + '/' + str(params['layer_sizes']).replace(':', '_') + str(params['reg_w_linear']).replace(':', '_')
            os.makedirs(model_path, exist_ok=True)
            saver.save(sess, model_path, global_step=eopch)
            auc = predict_test_file(preds, sess, params['test_file'], feature_cnt, _indices, _values, _shape, _y, _values2, _ind, _keep_probs, eopch, batch_size, 'test', model_path, params['output_predictions'], params)
            print('auc is ', auc, ', at epoch  ', eopch, ', time is {0:.4f} min'.format((end - start) / 60.0), ', train_loss is {0:.2f}'.format(train_loss_per_epoch))
    log_writer.close()

def run_with_parameter(dataset, rank, lr, lamb, mu, n_eopch, batch_size, wt, init_value):
    start = clock()
    tf.reset_default_graph()
    best_train_rmse, best_test_rmse, best_eval_rmse, best_eopch_idx = single_run(dataset, rank, dataset.n_user, dataset.n_item, lr, lamb, mu, n_eopch, batch_size, True, init_value)
    end = clock()
    wt.write('%d,%f,%f,%f,%d,%d,%f,%f,%f,%d,%f,%f\n' % (rank, lr, lamb, mu, n_eopch, batch_size, best_train_rmse, best_test_rmse, best_eval_rmse, best_eopch_idx, init_value, (end - start) / 60))
    wt.flush()

def run_with_parameters(dataset, params, wt):
    start = clock()
    tf.reset_default_graph()
    best_train_rmse, best_test_rmse, best_eval_rmse, best_eopch_idx = single_run(dataset, params)
    end = clock()
    wt.write('%f,%f,%f,%d,%f,%s\n' % (best_train_rmse, best_test_rmse, best_eval_rmse, best_eopch_idx, (end - start) / 60, str(params)))
    wt.flush()

