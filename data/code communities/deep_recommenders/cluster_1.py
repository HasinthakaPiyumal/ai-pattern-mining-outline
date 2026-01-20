# Cluster 1

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = build_estimator({'warm_up_from_fm': 'FM'})
    early_stop_hook = tf.estimator.experimental.stop_if_no_decrease_hook(estimator, 'loss', 1000)
    movielens = MovielensRanking()
    train_spec = tf.estimator.TrainSpec(lambda: movielens.training_input_fn, max_steps=None, hooks=[early_stop_hook])
    eval_spec = tf.estimator.EvalSpec(lambda: movielens.testing_input_fn, steps=None, start_delay_secs=0, throttle_secs=0)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

