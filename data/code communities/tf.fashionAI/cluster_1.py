# Cluster 1

def _zero_debias(unbiased_var, value, decay):
    """Compute the delta required for a debiased Variable.

  All exponential moving averages initialized with Tensors are initialized to 0,
  and therefore are biased to 0. Variables initialized to 0 and used as EMAs are
  similarly biased. This function creates the debias updated amount according to
  a scale factor, as in https://arxiv.org/abs/1412.6980.

  To demonstrate the bias the results from 0-initialization, take an EMA that
  was initialized to `0` with decay `b`. After `t` timesteps of seeing the
  constant `c`, the variable have the following value:

  ```
    EMA = 0*b^(t) + c*(1 - b)*b^(t-1) + c*(1 - b)*b^(t-2) + ...
        = c*(1 - b^t)
  ```

  To have the true value `c`, we would divide by the scale factor `1 - b^t`.

  In order to perform debiasing, we use two shadow variables. One keeps track of
  the biased estimate, and the other keeps track of the number of updates that
  have occurred.

  Args:
    unbiased_var: A Variable representing the current value of the unbiased EMA.
    value: A Tensor representing the most recent value.
    decay: A Tensor representing `1-decay` for the EMA.

  Returns:
    The amount that the unbiased variable should be updated. Computing this
    tensor will also update the shadow variables appropriately.
  """
    with variable_scope.variable_scope(unbiased_var.op.name, values=[unbiased_var, value, decay]) as scope:
        with ops.colocate_with(unbiased_var):
            with ops.init_scope():
                biased_initializer = init_ops.zeros_initializer(dtype=unbiased_var.dtype)(unbiased_var.get_shape())
                local_step_initializer = init_ops.zeros_initializer()

            def _maybe_get_unique(name):
                """Get name for a unique variable, if not `reuse=True`."""
                if variable_scope.get_variable_scope().reuse:
                    return name
                vs_vars = [x.op.name for x in variable_scope.get_variable_scope().global_variables()]
                full_name = variable_scope.get_variable_scope().name + '/' + name
                if full_name not in vs_vars:
                    return name
                idx = 1
                while full_name + '_%d' % idx in vs_vars:
                    idx += 1
                return name + '_%d' % idx
            biased_var = variable_scope.get_variable(_maybe_get_unique('biased'), initializer=biased_initializer, trainable=False)
            local_step = variable_scope.get_variable(_maybe_get_unique('local_step'), shape=[], dtype=unbiased_var.dtype, initializer=local_step_initializer, trainable=False)
            update_biased = state_ops.assign_sub(biased_var, (biased_var - value) * decay, name=scope.name)
            update_local_step = local_step.assign_add(1)
            with ops.control_dependencies([update_biased, update_local_step]):
                unbiased_ema_delta = unbiased_var - biased_var.read_value() / (1 - math_ops.pow(1.0 - decay, local_step.read_value()))
            return unbiased_ema_delta

def _maybe_get_unique(name):
    """Get name for a unique variable, if not `reuse=True`."""
    if variable_scope.get_variable_scope().reuse:
        return name
    vs_vars = [x.op.name for x in variable_scope.get_variable_scope().global_variables()]
    full_name = variable_scope.get_variable_scope().name + '/' + name
    if full_name not in vs_vars:
        return name
    idx = 1
    while full_name + '_%d' % idx in vs_vars:
        idx += 1
    return name + '_%d' % idx

