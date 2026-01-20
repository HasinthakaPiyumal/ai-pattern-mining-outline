# Cluster 3

def load_data_cache(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def predict_test_file(preds, sess, test_file, feature_cnt, _indices, _values, _shape, _y, _values2, _ind, _keep_probs, epoch, batch_size, tag, path, output_prediction=True, params=None):
    if output_prediction:
        wt = open(path + '/deepFM_pred_' + tag + str(epoch) + '.txt', 'w')
    gt_scores = []
    pred_scores = []
    for test_input_in_sp in load_data_cache(test_file):
        predictios = sess.run(preds, feed_dict={_indices: test_input_in_sp['indices'], _values: test_input_in_sp['values'], _shape: test_input_in_sp['shape'], _y: test_input_in_sp['labels'], _values2: test_input_in_sp['values2'], _ind: test_input_in_sp['feature_indices'], _keep_probs: np.ones_like(params['keep_probs'])}).reshape(-1).tolist()
        if output_prediction:
            for gt, preded in zip(test_input_in_sp['labels'].reshape(-1).tolist(), predictios):
                wt.write('{0:d},{1:f}\n'.format(int(gt), preded))
                gt_scores.append(gt)
                pred_scores.append(preded)
        else:
            gt_scores.extend(test_input_in_sp['labels'].reshape(-1).tolist())
            pred_scores.extend(predictios)
    auc = roc_auc_score(np.asarray(gt_scores), np.asarray(pred_scores))
    if output_prediction:
        wt.close()
    return auc

def single_run(params):
    logger.info('\n\n')
    logger.info(params)
    logger.info('\n\n')
    pre_build_data_cache_if_need(params['train_file'], params['batch_size'], params['clean_cache'] if 'clean_cache' in params else False)
    pre_build_data_cache_if_need(params['test_file'], params['batch_size'], params['clean_cache'] if 'clean_cache' in params else False)
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
    _field2feature_indices = tf.placeholder(tf.int64, shape=[None, 2], name='field2feature_indices')
    _field2feature_values = tf.placeholder(tf.int64, shape=[None], name='field2feature_values')
    _field2feature_weights = tf.placeholder(tf.float32, shape=[None], name='field2feature_weights')
    _field2feature_shape = tf.placeholder(tf.int64, shape=[2], name='field2feature_shape')
    _y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')
    train_step, loss, error, preds, tmp = build_model(_indices, _values, _values2, _shape, _field2feature_indices, _field2feature_values, _field2feature_weights, _field2feature_shape, _y, params)
    saver = tf.train.Saver()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    glo_ite = 0
    last_best_auc = None
    max_stop_grow_torrelence = 50
    stop_grow_cnt = 0
    start = clock()
    for eopch in range(n_epoch):
        iteration = -1
        time_load_data, time_sess = (0, 0)
        time_cp02 = clock()
        train_loss_per_epoch = 0
        for training_input_in_sp, qids, docids in load_data_cache(params['train_file']):
            time_cp01 = clock()
            time_load_data += time_cp01 - time_cp02
            iteration += 1
            glo_ite += 1
            _, cur_loss = sess.run([train_step, loss], feed_dict={_indices: training_input_in_sp['indices'], _values: training_input_in_sp['values'], _shape: training_input_in_sp['shape'], _y: training_input_in_sp['labels'], _values2: training_input_in_sp['values2'], _field2feature_indices: training_input_in_sp['field2feature_indices'], _field2feature_values: training_input_in_sp['field2feature_values'], _field2feature_weights: training_input_in_sp['field2feature_weights'], _field2feature_shape: training_input_in_sp['filed2feature_shape']})
            time_cp02 = clock()
            time_sess += time_cp02 - time_cp01
            train_loss_per_epoch += cur_loss
        end = clock()
        if eopch % 1 == 0:
            model_path = params['model_path'] + '/' + str(params['layer_sizes']).replace(':', '_') + str(params['reg_w_linear']).replace(':', '_')
            os.makedirs(model_path, exist_ok=True)
            saver.save(sess, model_path, global_step=eopch)
            metrics = predict_test_file(preds, sess, params['test_file'], _indices, _values, _shape, _y, _values2, _field2feature_indices, _field2feature_values, _field2feature_weights, _field2feature_shape, eopch, batch_size, 'test', model_path, params['output_predictions'], params)
            metrics_strs = []
            auc = 0
            for metric_name in metrics:
                metrics_strs.append('{0} is {1:.5f}'.format(metric_name, metrics[metric_name]))
                if metric_name == 'global_auc':
                    auc = metrics[metric_name]
            if last_best_auc is None or auc > last_best_auc:
                last_best_auc = auc
                stop_grow_cnt = 0
            else:
                stop_grow_cnt += 1
            res_str = ' ,'.join(metrics_strs) + ', at epoch {0:d}, time is {1:.4f} min, train_loss is {2:.2f}'.format(eopch, (end - start) / 60.0, train_loss_per_epoch)
            logger.info(res_str)
            start = clock()
            if stop_grow_cnt > max_stop_grow_torrelence:
                break

def predict_test_file(preds, sess, test_file, _indices, _values, _shape, _y, _values2, _field2feature_indices, _field2feature_values, _field2feature_weights, _field2feature_shape, epoch, batch_size, tag, path, output_prediction, params):
    if output_prediction:
        wt = open(path + '/deepFM_pred_' + tag + str(epoch) + '.txt', 'w')
    gt_scores = []
    pred_scores = []
    query2res = {}
    for test_input_in_sp, qids, docids in load_data_cache(test_file):
        predictios = sess.run(preds, feed_dict={_indices: test_input_in_sp['indices'], _values: test_input_in_sp['values'], _shape: test_input_in_sp['shape'], _y: test_input_in_sp['labels'], _values2: test_input_in_sp['values2'], _field2feature_indices: test_input_in_sp['field2feature_indices'], _field2feature_values: test_input_in_sp['field2feature_values'], _field2feature_weights: test_input_in_sp['field2feature_weights'], _field2feature_shape: test_input_in_sp['filed2feature_shape']}).reshape(-1).tolist()
        if output_prediction:
            for gt, preded, qid in zip(test_input_in_sp['labels'].reshape(-1).tolist(), predictios, qids):
                wt.write('{0:d},{1:f}\n'.format(int(gt), preded))
                gt_scores.append(gt)
                pred_scores.append(preded)
        else:
            for gt, preded, qid in zip(test_input_in_sp['labels'].reshape(-1).tolist(), predictios, qids):
                if qid not in query2res:
                    query2res[qid] = []
                query2res[qid].append([gt, preded])
    metrics = compute_metric(query2res, params)
    if output_prediction:
        wt.close()
    return metrics

