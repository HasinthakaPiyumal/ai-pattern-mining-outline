# Cluster 4

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def input_pipeline(is_training=True, model_scope=FLAGS.model_scope, num_epochs=FLAGS.epochs_per_eval):
    if 'all' in model_scope:
        lnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.global_norm_key, dtype=tf.int64), tf.constant(config.global_norm_lvalues, dtype=tf.int64)), 0)
        rnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.global_norm_key, dtype=tf.int64), tf.constant(config.global_norm_rvalues, dtype=tf.int64)), 1)
    else:
        lnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.local_norm_key, dtype=tf.int64), tf.constant(config.local_norm_lvalues, dtype=tf.int64)), 0)
        rnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.local_norm_key, dtype=tf.int64), tf.constant(config.local_norm_rvalues, dtype=tf.int64)), 1)
    preprocessing_fn = lambda org_image, classid, shape, key_x, key_y, key_v: preprocessing.preprocess_image(org_image, classid, shape, FLAGS.train_image_size, FLAGS.train_image_size, key_x, key_y, key_v, (lnorm_table, rnorm_table), is_training=is_training, data_format='NCHW' if FLAGS.data_format == 'channels_first' else 'NHWC', category=model_scope if 'all' not in model_scope else '*', bbox_border=FLAGS.bbox_border, heatmap_sigma=FLAGS.heatmap_sigma, heatmap_size=FLAGS.heatmap_size)
    images, shape, classid, targets, key_v, isvalid, norm_value = dataset.slim_get_split(FLAGS.data_dir, preprocessing_fn, FLAGS.batch_size, FLAGS.num_readers, FLAGS.num_preprocessing_threads, num_epochs=num_epochs, is_training=is_training, file_pattern=FLAGS.dataset_name, category=model_scope if 'all' not in model_scope else '*', reader=None)
    return (images, {'targets': targets, 'key_v': key_v, 'shape': shape, 'classid': classid, 'isvalid': isvalid, 'norm_value': norm_value})

def sub_loop(model_fn, model_scope, model_dir, run_config, train_epochs, epochs_per_eval, lr_decay_factors, decay_boundaries, checkpoint_path=None, checkpoint_exclude_scopes='', checkpoint_model_scope='', ignore_missing_vars=True):
    steps_per_epoch = config.split_size[model_scope if 'all' not in model_scope else '*']['train'] // FLAGS.batch_size
    fashionAI = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=run_config, params={'checkpoint_path': checkpoint_path, 'model_dir': model_dir, 'checkpoint_exclude_scopes': checkpoint_exclude_scopes, 'model_scope': model_scope, 'checkpoint_model_scope': checkpoint_model_scope, 'ignore_missing_vars': ignore_missing_vars, 'train_image_size': FLAGS.train_image_size, 'heatmap_size': FLAGS.heatmap_size, 'data_format': FLAGS.data_format, 'steps_per_epoch': steps_per_epoch, 'use_ohkm': FLAGS.use_ohkm, 'batch_size': FLAGS.batch_size, 'weight_decay': FLAGS.weight_decay, 'mse_weight': FLAGS.mse_weight, 'momentum': FLAGS.momentum, 'learning_rate': FLAGS.learning_rate, 'end_learning_rate': FLAGS.end_learning_rate, 'warmup_learning_rate': FLAGS.warmup_learning_rate, 'warmup_steps': FLAGS.warmup_steps, 'decay_boundaries': parse_comma_list(decay_boundaries), 'lr_decay_factors': parse_comma_list(lr_decay_factors)})
    tf.gfile.MakeDirs(model_dir)
    tf.logging.info('Starting to train model {}.'.format(model_scope))
    for _ in range(train_epochs // epochs_per_eval):
        tensors_to_log = {'lr': 'learning_rate', 'loss': 'total_loss', 'mse': 'mse_loss', 'ne': 'ne_mertric'}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps, formatter=lambda dicts: '{}:'.format(model_scope) + ', '.join(['%s=%.6f' % (k, v) for k, v in dicts.items()]))
        tf.logging.info('Starting a training cycle.')
        fashionAI.train(input_fn=lambda: input_pipeline(True, model_scope, epochs_per_eval), hooks=[logging_hook], max_steps=steps_per_epoch * train_epochs)
        tf.logging.info('Starting to evaluate.')
        eval_results = fashionAI.evaluate(input_fn=lambda: input_pipeline(False, model_scope, 1))
        tf.logging.info(eval_results)
    tf.logging.info('Finished model {}.'.format(model_scope))

def sub_loop(model_fn, model_scope, model_dir, run_config, train_epochs, epochs_per_eval, lr_decay_factors, decay_boundaries, checkpoint_path=None, checkpoint_exclude_scopes='', checkpoint_model_scope='', ignore_missing_vars=True):
    steps_per_epoch = config.split_size[model_scope if 'all' not in model_scope else '*']['train'] // FLAGS.batch_size
    fashionAI = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=run_config, params={'checkpoint_path': checkpoint_path, 'model_dir': model_dir, 'checkpoint_exclude_scopes': checkpoint_exclude_scopes, 'model_scope': model_scope, 'checkpoint_model_scope': checkpoint_model_scope, 'ignore_missing_vars': ignore_missing_vars, 'train_image_size': FLAGS.train_image_size, 'heatmap_size': FLAGS.heatmap_size, 'feats_channals': FLAGS.feats_channals, 'num_stacks': FLAGS.num_stacks, 'num_modules': FLAGS.num_modules, 'data_format': FLAGS.data_format, 'steps_per_epoch': steps_per_epoch, 'batch_size': FLAGS.batch_size, 'use_ohkm': FLAGS.use_ohkm, 'weight_decay': FLAGS.weight_decay, 'mse_weight': FLAGS.mse_weight, 'momentum': FLAGS.momentum, 'learning_rate': FLAGS.learning_rate, 'end_learning_rate': FLAGS.end_learning_rate, 'warmup_learning_rate': FLAGS.warmup_learning_rate, 'warmup_steps': FLAGS.warmup_steps, 'decay_boundaries': parse_comma_list(decay_boundaries), 'lr_decay_factors': parse_comma_list(lr_decay_factors)})
    tf.gfile.MakeDirs(model_dir)
    tf.logging.info('Starting to train model {}.'.format(model_scope))
    for _ in range(train_epochs // epochs_per_eval):
        tensors_to_log = {'lr': 'learning_rate', 'loss': 'total_loss', 'mse': 'mse_loss', 'ne': 'ne_mertric'}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps, formatter=lambda dicts: '{}:'.format(model_scope) + ', '.join(['%s=%.6f' % (k, v) for k, v in dicts.items()]))
        tf.logging.info('Starting a training cycle.')
        fashionAI.train(input_fn=lambda: input_pipeline(True, model_scope, epochs_per_eval), hooks=[logging_hook], max_steps=steps_per_epoch * train_epochs)
        tf.logging.info('Starting to evaluate.')
        eval_results = fashionAI.evaluate(input_fn=lambda: input_pipeline(False, model_scope, 1))
        tf.logging.info(eval_results)
    tf.logging.info('Finished model {}.'.format(model_scope))

def sub_loop(model_fn, model_scope, model_dir, run_config, train_epochs, epochs_per_eval, lr_decay_factors, decay_boundaries, checkpoint_path=None, checkpoint_exclude_scopes='', checkpoint_model_scope='', ignore_missing_vars=True):
    steps_per_epoch = config.split_size[model_scope if 'all' not in model_scope else '*']['train'] // FLAGS.batch_size
    fashionAI = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=run_config, params={'checkpoint_path': checkpoint_path, 'model_dir': model_dir, 'checkpoint_exclude_scopes': checkpoint_exclude_scopes, 'model_scope': model_scope, 'checkpoint_model_scope': checkpoint_model_scope, 'ignore_missing_vars': ignore_missing_vars, 'train_image_size': FLAGS.train_image_size, 'heatmap_size': FLAGS.heatmap_size, 'data_format': FLAGS.data_format, 'steps_per_epoch': steps_per_epoch, 'use_ohkm': FLAGS.use_ohkm, 'batch_size': FLAGS.batch_size, 'weight_decay': FLAGS.weight_decay, 'mse_weight': FLAGS.mse_weight, 'momentum': FLAGS.momentum, 'learning_rate': FLAGS.learning_rate, 'end_learning_rate': FLAGS.end_learning_rate, 'warmup_learning_rate': FLAGS.warmup_learning_rate, 'warmup_steps': FLAGS.warmup_steps, 'decay_boundaries': parse_comma_list(decay_boundaries), 'lr_decay_factors': parse_comma_list(lr_decay_factors)})
    tf.gfile.MakeDirs(model_dir)
    tf.logging.info('Starting to train model {}.'.format(model_scope))
    for _ in range(train_epochs // epochs_per_eval):
        tensors_to_log = {'lr': 'learning_rate', 'loss': 'total_loss', 'mse': 'mse_loss', 'ne': 'ne_mertric'}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps, formatter=lambda dicts: '{}:'.format(model_scope) + ', '.join(['%s=%.6f' % (k, v) for k, v in dicts.items()]))
        tf.logging.info('Starting a training cycle.')
        fashionAI.train(input_fn=lambda: input_pipeline(True, model_scope, epochs_per_eval), hooks=[logging_hook], max_steps=steps_per_epoch * train_epochs)
        tf.logging.info('Starting to evaluate.')
        eval_results = fashionAI.evaluate(input_fn=lambda: input_pipeline(False, model_scope, 1))
        tf.logging.info(eval_results)
    tf.logging.info('Finished model {}.'.format(model_scope))

def main(_):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=FLAGS.num_cpu_threads, inter_op_parallelism_threads=FLAGS.num_cpu_threads, gpu_options=gpu_options)
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=FLAGS.save_checkpoints_secs).replace(save_checkpoints_steps=None).replace(save_summary_steps=FLAGS.save_summary_steps).replace(keep_checkpoint_max=5).replace(tf_random_seed=FLAGS.tf_random_seed).replace(log_step_count_steps=FLAGS.log_every_n_steps).replace(session_config=sess_config)
    fashionAI = tf.estimator.Estimator(model_fn=keypoint_model_fn, model_dir=FLAGS.model_dir, config=run_config, params={'train_image_size': FLAGS.train_image_size, 'heatmap_size': FLAGS.heatmap_size, 'feats_channals': FLAGS.feats_channals, 'num_stacks': FLAGS.num_stacks, 'num_modules': FLAGS.num_modules, 'data_format': FLAGS.data_format, 'model_scope': FLAGS.model_scope, 'steps_per_epoch': config.split_size[FLAGS.model_scope if 'all' not in FLAGS.model_scope else '*']['train'] // FLAGS.batch_size, 'batch_size': FLAGS.batch_size, 'weight_decay': FLAGS.weight_decay, 'mse_weight': FLAGS.mse_weight, 'momentum': FLAGS.momentum, 'learning_rate': FLAGS.learning_rate, 'end_learning_rate': FLAGS.end_learning_rate, 'warmup_learning_rate': FLAGS.warmup_learning_rate, 'warmup_steps': FLAGS.warmup_steps, 'decay_boundaries': parse_comma_list(FLAGS.decay_boundaries), 'lr_decay_factors': parse_comma_list(FLAGS.lr_decay_factors)})
    if not FLAGS.run_on_cloud:
        tf.logging.info('params recv: %s', FLAGS.flag_values_dict())
    tf.gfile.MakeDirs(FLAGS.model_dir)
    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tensors_to_log = {'lr': 'learning_rate', 'loss': 'total_loss', 'mse': 'mse_loss', 'ne': 'ne_mertric'}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps, formatter=lambda dicts: ', '.join(['%s=%.7f' % (k, v) for k, v in dicts.items()]))
        tf.logging.info('Starting a training cycle.')
        fashionAI.train(input_fn=lambda: input_pipeline(True), hooks=[logging_hook])
        tf.logging.info('Starting to evaluate.')
        eval_results = fashionAI.evaluate(input_fn=lambda: input_pipeline(False, 1))
        tf.logging.info(eval_results)

def sub_loop(model_fn, model_scope, model_dir, run_config, train_epochs, epochs_per_eval, lr_decay_factors, decay_boundaries, checkpoint_path=None, checkpoint_exclude_scopes='', checkpoint_model_scope='', ignore_missing_vars=True):
    steps_per_epoch = config.split_size[model_scope if 'all' not in model_scope else '*']['train'] // (FLAGS.xt_batch_size if 'seresnext50' in FLAGS.backbone else FLAGS.batch_size)
    fashionAI = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=run_config, params={'checkpoint_path': checkpoint_path, 'model_dir': model_dir, 'checkpoint_exclude_scopes': checkpoint_exclude_scopes, 'model_scope': model_scope, 'checkpoint_model_scope': checkpoint_model_scope, 'ignore_missing_vars': ignore_missing_vars, 'train_image_size': FLAGS.train_image_size, 'heatmap_size': FLAGS.heatmap_size, 'data_format': FLAGS.data_format, 'steps_per_epoch': steps_per_epoch, 'use_ohkm': FLAGS.use_ohkm, 'batch_size': FLAGS.xt_batch_size if 'seresnext50' in FLAGS.backbone else FLAGS.batch_size, 'weight_decay': FLAGS.weight_decay, 'mse_weight': FLAGS.mse_weight, 'momentum': FLAGS.momentum, 'learning_rate': FLAGS.learning_rate, 'end_learning_rate': FLAGS.end_learning_rate, 'warmup_learning_rate': FLAGS.warmup_learning_rate, 'warmup_steps': FLAGS.warmup_steps, 'decay_boundaries': parse_comma_list(decay_boundaries), 'lr_decay_factors': parse_comma_list(lr_decay_factors)})
    tf.gfile.MakeDirs(model_dir)
    tf.logging.info('Starting to train model {}.'.format(model_scope))
    for _ in range(train_epochs // epochs_per_eval):
        tensors_to_log = {'lr': 'learning_rate', 'loss': 'total_loss', 'mse': 'mse_loss', 'ne': 'ne_mertric'}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps, formatter=lambda dicts: '{}:'.format(model_scope) + ', '.join(['%s=%.6f' % (k, v) for k, v in dicts.items()]))
        tf.logging.info('Starting a training cycle.')
        fashionAI.train(input_fn=lambda: input_pipeline(True, model_scope, epochs_per_eval), hooks=[logging_hook], max_steps=steps_per_epoch * train_epochs)
        tf.logging.info('Starting to evaluate.')
        eval_results = fashionAI.evaluate(input_fn=lambda: input_pipeline(False, model_scope, 1))
        tf.logging.info(eval_results)
    tf.logging.info('Finished model {}.'.format(model_scope))

def eval_each(model_fn, model_dir, model_scope, run_config):
    fashionAI = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=run_config, params={'train_image_size': FLAGS.train_image_size, 'heatmap_size': FLAGS.heatmap_size, 'feats_channals': FLAGS.feats_channals, 'num_stacks': FLAGS.num_stacks, 'num_modules': FLAGS.num_modules, 'data_format': FLAGS.data_format, 'model_scope': model_scope, 'flip_on_test': FLAGS.flip_on_test})
    tensors_to_log = {'cur_file': 'current_file'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps, formatter=lambda dicts: ', '.join(['%s=%s' % (k, v) for k, v in dicts.items()]))
    tf.logging.info('Starting to predict model {}.'.format(model_scope))
    pred_results = fashionAI.predict(input_fn=lambda: input_pipeline(model_scope), hooks=[logging_hook], checkpoint_path=train_helper.get_latest_checkpoint_for_evaluate_(model_dir, model_dir))
    return list(pred_results)

def sub_loop(model_fn, model_scope, model_dir, run_config, train_epochs, high_learning_rate, low_learning_rate, checkpoint_path=None):
    steps_per_epoch = config.split_size[model_scope if 'all' not in model_scope else '*']['train'] // FLAGS.batch_size
    fashionAI = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=run_config, params={'checkpoint_path': checkpoint_path, 'model_dir': model_dir, 'model_scope': model_scope, 'train_image_size': FLAGS.train_image_size, 'heatmap_size': FLAGS.heatmap_size, 'data_format': FLAGS.data_format, 'steps_per_epoch': steps_per_epoch, 'use_ohkm': FLAGS.use_ohkm, 'batch_size': FLAGS.batch_size, 'weight_decay': FLAGS.weight_decay, 'mse_weight': FLAGS.mse_weight, 'momentum': FLAGS.momentum, 'dummy_train': FLAGS.dummy_train, 'high_learning_rate': high_learning_rate, 'low_learning_rate': low_learning_rate})
    tf.gfile.MakeDirs(model_dir)
    tf.logging.info('Starting to train model {}.'.format(model_scope))
    tensors_to_log = {'lr': 'learning_rate', 'loss': 'total_loss', 'mse': 'mse_loss', 'ne': 'ne_mertric'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps, formatter=lambda dicts: '{}:'.format(model_scope) + ', '.join(['%s=%.6f' % (k, v) for k, v in dicts.items()]))
    tf.logging.info('Starting a training cycle.')
    fashionAI.train(input_fn=lambda: input_pipeline(True, model_scope, train_epochs), hooks=[logging_hook], max_steps=steps_per_epoch * (train_epochs + 1 if FLAGS.dummy_train else train_epochs))
    tf.logging.info('Finished model {}.'.format(model_scope))

def sub_loop(model_fn, model_scope, model_dir, run_config, train_epochs, epochs_per_eval, lr_decay_factors, decay_boundaries, checkpoint_path=None, checkpoint_exclude_scopes='', checkpoint_model_scope='', ignore_missing_vars=True):
    steps_per_epoch = config.split_size[model_scope if 'all' not in model_scope else '*']['train'] // FLAGS.batch_size
    _replicate_model_fn = tf_replicate_model_fn.replicate_model_fn(model_fn, loss_reduction=tf.losses.Reduction.MEAN)
    fashionAI = tf.estimator.Estimator(model_fn=_replicate_model_fn, model_dir=model_dir, config=run_config.replace(save_checkpoints_steps=2 * steps_per_epoch), params={'checkpoint_path': checkpoint_path, 'model_dir': model_dir, 'checkpoint_exclude_scopes': checkpoint_exclude_scopes, 'model_scope': model_scope, 'checkpoint_model_scope': checkpoint_model_scope, 'ignore_missing_vars': ignore_missing_vars, 'train_image_size': FLAGS.train_image_size, 'heatmap_size': FLAGS.heatmap_size, 'data_format': FLAGS.data_format, 'steps_per_epoch': steps_per_epoch, 'use_ohkm': FLAGS.use_ohkm, 'batch_size': FLAGS.batch_size, 'weight_decay': FLAGS.weight_decay, 'mse_weight': FLAGS.mse_weight, 'momentum': FLAGS.momentum, 'learning_rate': FLAGS.learning_rate, 'end_learning_rate': FLAGS.end_learning_rate, 'warmup_learning_rate': FLAGS.warmup_learning_rate, 'warmup_steps': FLAGS.warmup_steps, 'decay_boundaries': parse_comma_list(decay_boundaries), 'lr_decay_factors': parse_comma_list(lr_decay_factors)})
    tf.gfile.MakeDirs(model_dir)
    tf.logging.info('Starting to train model {}.'.format(model_scope))
    for _ in range(train_epochs // epochs_per_eval):
        tensors_to_log = {'lr': 'learning_rate', 'loss': 'total_loss', 'mse': 'mse_loss', 'ne': 'ne_mertric'}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps, formatter=lambda dicts: '{}:'.format(model_scope) + ', '.join(['%s=%.6f' % (k, v) for k, v in dicts.items()]))
        tf.logging.info('Starting a training cycle.')
        fashionAI.train(input_fn=lambda: input_pipeline(True, model_scope, epochs_per_eval), hooks=[logging_hook], max_steps=steps_per_epoch * train_epochs)
        tf.logging.info('Starting to evaluate.')
        eval_results = fashionAI.evaluate(input_fn=lambda: input_pipeline(False, model_scope, 1))
        tf.logging.info(eval_results)
    tf.logging.info('Finished model {}.'.format(model_scope))

def replicate_model_fn(model_fn, loss_reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS, devices=None):
    """Replicate `Estimator.model_fn` over GPUs.

  The given `model_fn` specifies a single forward pass of a model.  To replicate
  such a model over GPUs, each GPU gets its own instance of the forward pass
  (a.k.a. a tower).  The input features and labels get sharded into the chunks
  that correspond to the number of GPUs.  Each tower computes a loss based
  on its input.  For each such loss, gradients are computed.  After that, the
  available losses are aggregated to form aggregated loss.  Available
  gradients are summed.  Then, they update weights using the specified
  optimizer.

  If `devices` are `None`, then all available GPUs are going to be used for
  replication.  If no GPUs are available, then the model is going to be
  placed on the CPU.

  Two modes of local replication over available GPUs are supported:
    1)  If exactly 1 GPU is detected, then variables and operations are placed
        onto the GPU.
    2)  If more than 1 GPU is detected, then variables are going to be placed on
        the CPU.  Replicas of operations are placed on each individual GPU.

  Here is an example of how one might use their `model_fn` to run over GPUs:
    ```python
       ...
       def model_fn(...):  # See `model_fn` in `Estimator`.
         loss = ...
         optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
         optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
         if mode == tf.estimator.ModeKeys.TRAIN:
           #  See the section below on `EstimatorSpec.train_op`.
           return EstimatorSpec(mode=mode, loss=loss,
                                train_op=optimizer.minimize(loss))

         #  No change for `ModeKeys.EVAL` or `ModeKeys.PREDICT`.
         return EstimatorSpec(...)
       ...
       classifier = tf.estimator.Estimator(
         model_fn=tf.contrib.estimator.replicate_model_fn(model_fn))
    ```

  Please see `DNNClassifierIntegrationTest` for an example with a canned
  Estimator.

  On `EstimatorSpec.train_op`:
  `model_fn` returns `EstimatorSpec.train_op` for
  `tf.estimator.GraphKeys.TRAIN`. It is typically derived using an optimizer.
  Towers are expected to populate it in the same way.  Gradients from all towers
  are reduced and applied in the last tower.  To achieve that in the case of
  multiple towers, `TowerOptimizer` needs to be used.  See `TowerOptimizer`.

  On sharding input features and labels:
  Input features and labels are split for consumption by each tower. They are
  split across the dimension 0.  Features and labels need to be batch major.

  On reduction algorithms:
  Certain algorithms were chosen for aggregating results of computations on
  multiple towers:
    - Losses from all towers are reduced according to `loss_reduction`.
    - Gradients are reduced using sum for each trainable variable.
    - `eval_metrics_ops` are reduced per metric using `reduce_mean`.
    - `EstimatorSpec.predictions` and `EstimatorSpec.export_outputs` are
      reduced using concatenation.
    - For all other fields of `EstimatorSpec` the values of the first tower
      are taken.

  On distribution of variables:
  Variables are not duplicated between towers.  Instead, they are placed on a
  single device as defined above and shared across towers.

  On overhead:
  If only one device is specified, then aggregation of loss and gradients
  doesn't happen. Replication consists of placing `model_fn` onto the
  specified device.

  On current limitations:
    - `predictions` are not supported for `ModeKeys.EVAL`.  They are required
       for `tf.contrib.estimator.add_metrics`.

  Args:
    model_fn: `model_fn` as defined in `Estimator`.  See the section above about
      the train_op argument of `EstimatorSpec`.
    loss_reduction: controls whether losses are summed or averaged.
    devices: Optional list of devices to replicate the model across.  This
      argument can be used to replice only on the subset of available GPUs.
      If `None`, then all available GPUs are going to be used for replication.
      If no GPUs are available, then the model is going to be placed on the CPU.

  Raises:
    ValueError: if there is no `loss_reduction` or if TowerOptimizer is
      mis-used.

  Returns:
    A replicated version of the supplied `model_fn`. Returned function that
      conforms to the requirements of `Estimator`'s `model_fn` and can be used
      instead of the supplied `model_fn`.
  """
    return _replicate_model_fn_with_mode(model_fn, loss_reduction, devices, mode=_VariableDistributionMode.SHARED_LOCAL_PARAMETER_SERVER)

def eval_each(model_fn, model_dir, model_scope, run_config):
    fashionAI = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=run_config, params={'train_image_size': FLAGS.train_image_size, 'heatmap_size': FLAGS.heatmap_size, 'data_format': FLAGS.data_format, 'model_scope': model_scope, 'flip_on_test': FLAGS.flip_on_test})
    tensors_to_log = {'cur_file': 'current_file'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps, formatter=lambda dicts: ', '.join(['%s=%s' % (k, v) for k, v in dicts.items()]))
    tf.logging.info('Starting to predict model {}.'.format(model_scope))
    pred_results = fashionAI.predict(input_fn=lambda: input_pipeline(model_scope), hooks=[logging_hook], checkpoint_path=train_helper.get_latest_checkpoint_for_evaluate_(model_dir, model_dir))
    return list(pred_results)

def sub_loop(model_fn, model_scope, model_dir, run_config, train_epochs, epochs_per_eval, lr_decay_factors, decay_boundaries, checkpoint_path=None, checkpoint_exclude_scopes='', checkpoint_model_scope='', ignore_missing_vars=True):
    steps_per_epoch = config.split_size[model_scope if 'all' not in model_scope else '*']['train'] // FLAGS.batch_size
    _replicate_model_fn = tf_replicate_model_fn.replicate_model_fn(model_fn, loss_reduction=tf.losses.Reduction.MEAN)
    fashionAI = tf.estimator.Estimator(model_fn=_replicate_model_fn, model_dir=model_dir, config=run_config.replace(save_checkpoints_steps=2 * steps_per_epoch), params={'checkpoint_path': checkpoint_path, 'model_dir': model_dir, 'checkpoint_exclude_scopes': checkpoint_exclude_scopes, 'model_scope': model_scope, 'checkpoint_model_scope': checkpoint_model_scope, 'ignore_missing_vars': ignore_missing_vars, 'net_depth': FLAGS.net_depth, 'train_image_size': FLAGS.train_image_size, 'heatmap_size': FLAGS.heatmap_size, 'data_format': FLAGS.data_format, 'steps_per_epoch': steps_per_epoch, 'use_ohkm': FLAGS.use_ohkm, 'batch_size': FLAGS.batch_size, 'weight_decay': FLAGS.weight_decay, 'mse_weight': FLAGS.mse_weight, 'momentum': FLAGS.momentum, 'learning_rate': FLAGS.learning_rate, 'end_learning_rate': FLAGS.end_learning_rate, 'warmup_learning_rate': FLAGS.warmup_learning_rate, 'warmup_steps': FLAGS.warmup_steps, 'decay_boundaries': parse_comma_list(decay_boundaries), 'lr_decay_factors': parse_comma_list(lr_decay_factors)})
    tf.gfile.MakeDirs(model_dir)
    tf.logging.info('Starting to train model {}.'.format(model_scope))
    for _ in range(train_epochs // epochs_per_eval):
        tensors_to_log = {'lr': 'learning_rate', 'loss': 'total_loss', 'mse': 'mse_loss', 'ne': 'ne_mertric'}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps, formatter=lambda dicts: '{}:'.format(model_scope) + ', '.join(['%s=%.6f' % (k, v) for k, v in dicts.items()]))
        tf.logging.info('Starting a training cycle.')
        fashionAI.train(input_fn=lambda: input_pipeline(True, model_scope, epochs_per_eval), hooks=[logging_hook], max_steps=steps_per_epoch * train_epochs)
        tf.logging.info('Starting to evaluate.')
        eval_results = fashionAI.evaluate(input_fn=lambda: input_pipeline(False, model_scope, 1))
        tf.logging.info(eval_results)
    tf.logging.info('Finished model {}.'.format(model_scope))

