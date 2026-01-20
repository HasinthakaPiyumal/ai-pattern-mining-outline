# Cluster 2

def build_estimator(params, model_dir=None, inter_op=8, intra_op=8):
    config_proto = tf.ConfigProto(device_count={'GPU': 0}, inter_op_parallelism_threads=inter_op, intra_op_parallelism_threads=intra_op)
    run_config = tf.estimator.RunConfig().replace(tf_random_seed=42, keep_checkpoint_max=10, save_checkpoints_steps=1000, log_step_count_steps=100, session_config=config_proto)
    return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=run_config, params=params)

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = build_estimator()
    early_stop_hook = tf.estimator.experimental.stop_if_no_decrease_hook(estimator, 'loss', 1000)
    movielens = MovielensRanking()
    train_spec = tf.estimator.TrainSpec(lambda: movielens.training_input_fn, max_steps=None, hooks=[early_stop_hook])
    eval_spec = tf.estimator.EvalSpec(lambda: movielens.testing_input_fn, steps=None, start_delay_secs=0, throttle_secs=0)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = build_estimator()
    early_stop_hook = tf.estimator.experimental.stop_if_no_decrease_hook(estimator, 'loss', 1000)
    movielens = MovielensRanking()
    train_spec = tf.estimator.TrainSpec(lambda: movielens.training_input_fn, max_steps=None, hooks=[early_stop_hook])
    eval_spec = tf.estimator.EvalSpec(lambda: movielens.testing_input_fn, steps=None, start_delay_secs=0, throttle_secs=0)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = build_estimator()
    early_stop_hook = tf.estimator.experimental.stop_if_no_decrease_hook(estimator, 'loss', 1000)
    movielens = MovielensRanking()
    train_spec = tf.estimator.TrainSpec(lambda: movielens.training_input_fn, max_steps=None, hooks=[early_stop_hook])
    eval_spec = tf.estimator.EvalSpec(lambda: movielens.testing_input_fn, steps=None, start_delay_secs=0, throttle_secs=0)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = build_estimator()
    early_stop_hook = tf.estimator.experimental.stop_if_no_decrease_hook(estimator, 'loss', 1000)
    synthetic = SyntheticForMultiTask(512 * 1000, example_dim=EXAMPLE_DIM)
    train_spec = tf.estimator.TrainSpec(lambda: synthetic.input_fn().take(800), max_steps=None, hooks=[early_stop_hook])
    eval_spec = tf.estimator.EvalSpec(lambda: synthetic.input_fn().skip(800).take(200), steps=None, start_delay_secs=60, throttle_secs=60)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

