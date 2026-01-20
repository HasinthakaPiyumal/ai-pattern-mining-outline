# Cluster 15

def _train_spec(tower_specs, train_op, aggregation_device, aggregated_loss_name='loss'):
    """Populate replicated EstimatorSpec for `GraphKeys.TRAIN`."""
    estimator_spec = _asdict(tower_specs[-1])
    estimator_spec['mode'] = model_fn_lib.ModeKeys.TRAIN
    estimator_spec['train_op'] = train_op
    estimator_spec['loss'] = _compute_sum_on_device([spec.loss for spec in tower_specs], aggregation_device, aggregated_loss_name)
    return model_fn_lib.EstimatorSpec(**estimator_spec)

def _compute_sum_on_device(values, device, name=None):
    with ops_lib.device(device):
        if isinstance(values[0], ops_lib.IndexedSlices):
            if name:
                raise ValueError('The name {} is not expected to be given to IndexedSlices {}'.format(name, values))
            values_concat = array_ops.concat([v.values for v in values], axis=0)
            indices_concat = array_ops.concat([v.indices for v in values], axis=0)
            return ops_lib.IndexedSlices(values_concat, indices_concat, values[0].dense_shape)
        else:
            return math_ops.add_n(values, name=name)

def _eval_spec(tower_specs, aggregation_device, aggregated_loss_name='loss'):
    """Populate replicated EstimatorSpec for `GraphKeys.EVAL`."""
    estimator_spec = _asdict(tower_specs[0])
    estimator_spec['mode'] = model_fn_lib.ModeKeys.EVAL
    estimator_spec['loss'] = _compute_sum_on_device([spec.loss for spec in tower_specs], aggregation_device, aggregated_loss_name)
    update_ops = []
    for tower_spec in tower_specs:
        for name, (_, update_op) in six.iteritems(tower_spec.eval_metric_ops):
            update_ops.append(update_op)
    with ops_lib.control_dependencies(update_ops):
        reduced_update_op = _reduce_metric_variables(len(tower_specs))
    eval_metric_ops = {}
    for name, (metric_tensor, _) in six.iteritems(tower_specs[0].eval_metric_ops):
        eval_metric_ops[name] = (metric_tensor, reduced_update_op)
    estimator_spec['eval_metric_ops'] = eval_metric_ops
    return model_fn_lib.EstimatorSpec(**estimator_spec)

