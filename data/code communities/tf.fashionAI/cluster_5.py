# Cluster 5

def eval_each(model_fn, model_dir, model_scope, run_config):
    fashionAI = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=run_config, params={'train_image_size': FLAGS.train_image_size, 'heatmap_size': FLAGS.heatmap_size, 'data_format': FLAGS.data_format, 'model_scope': model_scope, 'flip_on_test': FLAGS.flip_on_test})
    tensors_to_log = {'cur_file': 'current_file'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps, formatter=lambda dicts: ', '.join(['%s=%s' % (k, v) for k, v in dicts.items()]))
    tf.logging.info('Starting to predict model {}.'.format(model_scope))
    pred_results = fashionAI.predict(input_fn=lambda: input_pipeline(model_scope), hooks=[logging_hook], checkpoint_path=train_helper.get_latest_checkpoint_for_evaluate_(model_dir, model_dir))
    return list(pred_results)

def main(_):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=FLAGS.num_cpu_threads, inter_op_parallelism_threads=FLAGS.num_cpu_threads, gpu_options=gpu_options)
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=None).replace(save_checkpoints_steps=None).replace(save_summary_steps=FLAGS.save_summary_steps).replace(keep_checkpoint_max=5).replace(tf_random_seed=FLAGS.tf_random_seed).replace(log_step_count_steps=FLAGS.log_every_n_steps).replace(session_config=sess_config)
    model_to_eval = [s.strip() for s in FLAGS.model_to_eval.split(',')]
    full_model_dir = os.path.join(FLAGS.model_dir, all_models[FLAGS.backbone.strip()]['logs_sub_dir'])
    for m in model_to_eval:
        if m == '':
            continue
        pred_results = eval_each(keypoint_model_fn, os.path.join(full_model_dir, m), m, run_config)
        df = pd.DataFrame(columns=['image_id', 'image_category'] + config.all_keys)
        cur_record = 0
        gloabl2local_ind = dict(zip(config.class2global_ind_map[m], list(range(len(config.class2global_ind_map[m])))))
        for pred_item in pred_results:
            temp_list = []
            index = 0
            x = pred_item['pred_x'].tolist()
            y = pred_item['pred_y'].tolist()
            filename = pred_item['file_name'].decode('utf8')
            for ind in list(range(config.class_num_joints['*'])):
                if ind in gloabl2local_ind:
                    temp_list.append('{}_{}_1'.format(round(x[gloabl2local_ind[ind]]), round(y[gloabl2local_ind[ind]])))
                else:
                    temp_list.append('-1_-1_-1')
            df.loc[cur_record] = [filename, m] + temp_list
            cur_record = cur_record + 1
        df.to_csv('./{}_{}.csv'.format(FLAGS.backbone.strip(), m), encoding='utf-8', index=False)
    df_list = [pd.read_csv('./{}_{}.csv'.format(FLAGS.backbone.strip(), model_to_eval[0]), encoding='utf-8')]
    for m in model_to_eval[1:]:
        if m == '':
            continue
        df_list.append(pd.read_csv('./{}_{}.csv'.format(FLAGS.backbone.strip(), m), encoding='utf-8'))
    time_stamps = int(time.time())
    pd.concat(df_list, ignore_index=True).to_csv('./{}_sub_{}.csv'.format(FLAGS.backbone.strip(), time_stamps), encoding='utf-8', index=False)
    if FLAGS.run_on_cloud:
        tf.gfile.Copy('./{}_sub_{}.csv'.format(FLAGS.backbone.strip(), time_stamps), os.path.join(full_model_dir, '{}_sub_{}.csv'.format(FLAGS.backbone.strip(), time_stamps)), overwrite=True)

def main(_):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=FLAGS.num_cpu_threads, inter_op_parallelism_threads=FLAGS.num_cpu_threads, gpu_options=gpu_options)
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=None).replace(save_checkpoints_steps=None).replace(save_summary_steps=FLAGS.save_summary_steps).replace(keep_checkpoint_max=5).replace(tf_random_seed=FLAGS.tf_random_seed).replace(log_step_count_steps=FLAGS.log_every_n_steps).replace(session_config=sess_config)
    model_to_eval = [s.strip() for s in FLAGS.model_to_eval.split(',')]
    for m in model_to_eval:
        if m == '':
            continue
        pred_results = eval_each(keypoint_model_fn, os.path.join(FLAGS.model_dir, m), m, run_config)
        df = pd.DataFrame(columns=['image_id', 'image_category'] + config.all_keys)
        cur_record = 0
        gloabl2local_ind = dict(zip(config.class2global_ind_map[m], list(range(len(config.class2global_ind_map[m])))))
        for pred_item in pred_results:
            temp_list = []
            index = 0
            x = pred_item['pred_x'].tolist()
            y = pred_item['pred_y'].tolist()
            filename = pred_item['file_name'].decode('utf8')
            for ind in list(range(config.class_num_joints['*'])):
                if ind in gloabl2local_ind:
                    temp_list.append('{}_{}_1'.format(round(x[gloabl2local_ind[ind]]), round(y[gloabl2local_ind[ind]])))
                else:
                    temp_list.append('-1_-1_-1')
            df.loc[cur_record] = [filename, m] + temp_list
            cur_record = cur_record + 1
        df.to_csv('./{}.csv'.format(m), encoding='utf-8', index=False)
    df_list = [pd.read_csv('./{}.csv'.format(model_to_eval[0]), encoding='utf-8')]
    for m in model_to_eval[1:]:
        if m == '':
            continue
        df_list.append(pd.read_csv('./{}.csv'.format(m), encoding='utf-8'))
    pd.concat(df_list, ignore_index=True).to_csv('./sub.csv', encoding='utf-8', index=False)
    if FLAGS.run_on_cloud:
        tf.gfile.Copy('./sub.csv', os.path.join(FLAGS.model_dir, 'sub.csv'), overwrite=True)

def main(_):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=FLAGS.num_cpu_threads, inter_op_parallelism_threads=FLAGS.num_cpu_threads, gpu_options=gpu_options)
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=None).replace(save_checkpoints_steps=None).replace(save_summary_steps=FLAGS.save_summary_steps).replace(keep_checkpoint_max=5).replace(tf_random_seed=FLAGS.tf_random_seed).replace(log_step_count_steps=FLAGS.log_every_n_steps).replace(session_config=sess_config)
    model_to_eval = [s.strip() for s in FLAGS.model_to_eval.split(',')]
    full_model_dir = os.path.join(FLAGS.model_dir, all_models[FLAGS.backbone.strip()]['logs_sub_dir'])
    for m in model_to_eval:
        if m == '':
            continue
        pred_results = eval_each(keypoint_model_fn, os.path.join(full_model_dir, m), m, run_config)
        df = pd.DataFrame(columns=['image_id', 'image_category'] + config.all_keys)
        cur_record = 0
        gloabl2local_ind = dict(zip(config.class2global_ind_map[m], list(range(len(config.class2global_ind_map[m])))))
        for pred_item in pred_results:
            temp_list = []
            index = 0
            x = pred_item['pred_x'].tolist()
            y = pred_item['pred_y'].tolist()
            filename = pred_item['file_name'].decode('utf8')
            for ind in list(range(config.class_num_joints['*'])):
                if ind in gloabl2local_ind:
                    temp_list.append('{}_{}_1'.format(round(x[gloabl2local_ind[ind]]), round(y[gloabl2local_ind[ind]])))
                else:
                    temp_list.append('-1_-1_-1')
            df.loc[cur_record] = [filename, m] + temp_list
            cur_record = cur_record + 1
        df.to_csv('./{}_{}.csv'.format(FLAGS.backbone.strip(), m), encoding='utf-8', index=False)
    df_list = [pd.read_csv('./{}_{}.csv'.format(FLAGS.backbone.strip(), model_to_eval[0]), encoding='utf-8')]
    for m in model_to_eval[1:]:
        if m == '':
            continue
        df_list.append(pd.read_csv('./{}_{}.csv'.format(FLAGS.backbone.strip(), m), encoding='utf-8'))
    pd.concat(df_list, ignore_index=True).to_csv('./{}_sub.csv'.format(FLAGS.backbone.strip()), encoding='utf-8', index=False)
    if FLAGS.run_on_cloud:
        tf.gfile.Copy('./{}_sub.csv'.format(FLAGS.backbone.strip()), os.path.join(full_model_dir, '{}_sub.csv'.format(FLAGS.backbone.strip())), overwrite=True)

