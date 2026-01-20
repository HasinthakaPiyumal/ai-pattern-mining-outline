# Cluster 0

def weighted_moving_average(value, decay, weight, truediv=True, collections=None, name=None):
    """Compute the weighted moving average of `value`.

  Conceptually, the weighted moving average is:
    `moving_average(value * weight) / moving_average(weight)`,
  where a moving average updates by the rule
    `new_value = decay * old_value + (1 - decay) * update`
  Internally, this Op keeps moving average variables of both `value * weight`
  and `weight`.

  Args:
    value: A numeric `Tensor`.
    decay: A float `Tensor` or float value.  The moving average decay.
    weight:  `Tensor` that keeps the current value of a weight.
      Shape should be able to multiply `value`.
    truediv:  Boolean, if `True`, dividing by `moving_average(weight)` is
      floating point division.  If `False`, use division implied by dtypes.
    collections:  List of graph collections keys to add the internal variables
      `value * weight` and `weight` to.
      Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
    name: Optional name of the returned operation.
      Defaults to "WeightedMovingAvg".

  Returns:
    An Operation that updates and returns the weighted moving average.
  """
    if collections is None:
        collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    with variable_scope.variable_scope(name, 'WeightedMovingAvg', [value, weight, decay]) as scope:
        value_x_weight_var = variable_scope.get_variable('value_x_weight', shape=value.get_shape(), dtype=value.dtype, initializer=init_ops.zeros_initializer(), trainable=False, collections=collections)
        weight_var = variable_scope.get_variable('weight', shape=weight.get_shape(), dtype=weight.dtype, initializer=init_ops.zeros_initializer(), trainable=False, collections=collections)
        numerator = assign_moving_average(value_x_weight_var, value * weight, decay, zero_debias=False)
        denominator = assign_moving_average(weight_var, weight, decay, zero_debias=False)
        if truediv:
            return math_ops.truediv(numerator, denominator, name=scope.name)
        else:
            return math_ops.div(numerator, denominator, name=scope.name)

def assign_moving_average(variable, value, decay, zero_debias=True, name=None):
    """Compute the moving average of a variable.

  The moving average of 'variable' updated with 'value' is:
    variable * decay + value * (1 - decay)

  The returned Operation sets 'variable' to the newly computed moving average.

  The new value of 'variable' can be set with the 'AssignSub' op as:
     variable -= (1 - decay) * (variable - value)

  Since variables that are initialized to a `0` value will be `0` biased,
  `zero_debias` optionally enables scaling by the mathematically correct
  debiasing factor of
    1 - decay ** num_updates
  See `ADAM: A Method for Stochastic Optimization` Section 3 for more details
  (https://arxiv.org/abs/1412.6980).

  The names of the debias shadow variables, by default, include both the scope
  they were created in and the scope of the variables they debias. They are also
  given a uniqifying-suffix.

  E.g.:

  ```
    with tf.variable_scope('scope1'):
      with tf.variable_scope('scope2'):
        var = tf.get_variable('foo')
        tf.assign_moving_average(var, 0.0, 1.0)
        tf.assign_moving_average(var, 0.0, 0.9)

    # var.name: 'scope1/scope2/foo'
    # shadow var names: 'scope1/scope2/scope1/scope2/foo/biased'
    #                   'scope1/scope2/scope1/scope2/foo/biased_1'
  ```

  Args:
    variable: A Variable.
    value: A tensor with the same shape as 'variable'.
    decay: A float Tensor or float value.  The moving average decay.
    zero_debias: A python bool. If true, assume the variable is 0-initialized
      and unbias it, as in https://arxiv.org/abs/1412.6980. See docstring in
      `_zero_debias` for more details.
    name: Optional name of the returned operation.

  Returns:
    A reference to the input 'variable' tensor with the newly computed
    moving average.
  """
    with ops.name_scope(name, 'AssignMovingAvg', [variable, value, decay]) as scope:
        with ops.colocate_with(variable):
            decay = ops.convert_to_tensor(1.0 - decay, name='decay')
            if decay.dtype != variable.dtype.base_dtype:
                decay = math_ops.cast(decay, variable.dtype.base_dtype)
            if zero_debias:
                update_delta = _zero_debias(variable, value, decay)
            else:
                update_delta = (variable - value) * decay
            return state_ops.assign_sub(variable, update_delta, name=scope)

class SWAMovingAverage(object):
    """Maintains moving averages of variables by employing an exponential decay.

  When training a model, it is often beneficial to maintain moving averages of
  the trained parameters.  Evaluations that use averaged parameters sometimes
  produce significantly better results than the final trained values.

  The `apply()` method adds shadow copies of trained variables and add ops that
  maintain a moving average of the trained variables in their shadow copies.
  It is used when building the training model.  The ops that maintain moving
  averages are typically run after each training step.
  The `average()` and `average_name()` methods give access to the shadow
  variables and their names.  They are useful when building an evaluation
  model, or when restoring a model from a checkpoint file.  They help use the
  moving averages in place of the last trained values for evaluations.

  The moving averages are computed using exponential decay.  You specify the
  decay value when creating the `ExponentialMovingAverage` object.  The shadow
  variables are initialized with the same initial values as the trained
  variables.  When you run the ops to maintain the moving averages, each
  shadow variable is updated with the formula:

    `shadow_variable -= (1 - decay) * (shadow_variable - variable)`

  This is mathematically equivalent to the classic formula below, but the use
  of an `assign_sub` op (the `"-="` in the formula) allows concurrent lockless
  updates to the variables:

    `shadow_variable = decay * shadow_variable + (1 - decay) * variable`

  Reasonable values for `decay` are close to 1.0, typically in the
  multiple-nines range: 0.999, 0.9999, etc.

  Example usage when creating a training model:

  ```python
  # Create variables.
  var0 = tf.Variable(...)
  var1 = tf.Variable(...)
  # ... use the variables to build a training model...
  ...
  # Create an op that applies the optimizer.  This is what we usually
  # would use as a training op.
  opt_op = opt.minimize(my_loss, [var0, var1])

  # Create an ExponentialMovingAverage object
  ema = tf.train.ExponentialMovingAverage(decay=0.9999)

  with tf.control_dependencies([opt_op]):
      # Create the shadow variables, and add ops to maintain moving averages
      # of var0 and var1. This also creates an op that will update the moving
      # averages after each training step.  This is what we will use in place
      # of the usual training op.
      training_op = ema.apply([var0, var1])

  ...train the model by running training_op...
  ```

  There are two ways to use the moving averages for evaluations:

  *  Build a model that uses the shadow variables instead of the variables.
     For this, use the `average()` method which returns the shadow variable
     for a given variable.
  *  Build a model normally but load the checkpoint files to evaluate by using
     the shadow variable names.  For this use the `average_name()` method.  See
     the @{tf.train.Saver} for more
     information on restoring saved variables.

  Example of restoring the shadow variable values:

  ```python
  # Create a Saver that loads variables from their saved shadow values.
  shadow_var0_name = ema.average_name(var0)
  shadow_var1_name = ema.average_name(var1)
  saver = tf.train.Saver({shadow_var0_name: var0, shadow_var1_name: var1})
  saver.restore(...checkpoint filename...)
  # var0 and var1 now hold the moving average values
  ```
  """

    def __init__(self, num_updates, zero_debias=False, name='SWAMovingAverage'):
        """Creates a new ExponentialMovingAverage object.

    The `apply()` method has to be called to create shadow variables and add
    ops to maintain moving averages.

    The optional `num_updates` parameter allows one to tweak the decay rate
    dynamically. It is typical to pass the count of training steps, usually
    kept in a variable that is incremented at each step, in which case the
    decay rate is lower at the start of training.  This makes moving averages
    move faster.  If passed, the actual decay rate used is:

      `num_updates / (1 + num_updates)`

    Args:
      decay: Float.  The decay to use.
      num_updates: Optional count of number of updates applied to variables.
      zero_debias: If `True`, zero debias moving-averages that are initialized
        with tensors.
      name: String. Optional prefix name to use for the name of ops added in
        `apply()`.
    """
        self._num_updates = num_updates
        self._zero_debias = zero_debias
        self._name = name
        self._averages = {}

    def apply(self, var_list=None):
        """Maintains moving averages of variables.

    `var_list` must be a list of `Variable` or `Tensor` objects.  This method
    creates shadow variables for all elements of `var_list`.  Shadow variables
    for `Variable` objects are initialized to the variable's initial value.
    They will be added to the `GraphKeys.MOVING_AVERAGE_VARIABLES` collection.
    For `Tensor` objects, the shadow variables are initialized to 0 and zero
    debiased (see docstring in `assign_moving_average` for more details).

    shadow variables are created with `trainable=False` and added to the
    `GraphKeys.ALL_VARIABLES` collection.  They will be returned by calls to
    `tf.global_variables()`.

    Returns an op that updates all shadow variables as described above.

    Note that `apply()` can be called multiple times with different lists of
    variables.

    Args:
      var_list: A list of Variable or Tensor objects. The variables
        and Tensors must be of types float16, float32, or float64.

    Returns:
      An Operation that updates the moving averages.

    Raises:
      TypeError: If the arguments are not all float16, float32, or float64.
      ValueError: If the moving average of one of the variables is already
        being computed.
    """
        if var_list is None:
            var_list = variables.trainable_variables()
        zero_debias_true = set()
        for var in var_list:
            if var.dtype.base_dtype not in [dtypes.float16, dtypes.float32, dtypes.float64]:
                raise TypeError('The variables must be half, float, or double: %s' % var.name)
            if var in self._averages:
                raise ValueError('Moving average already computed for: %s' % var.name)
            with ops.init_scope():
                if isinstance(var, variables.Variable):
                    avg = slot_creator.create_slot(var, var.initialized_value(), self._name, colocate_with_primary=True)
                    ops.add_to_collection(ops.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
                else:
                    avg = slot_creator.create_zeros_slot(var, self._name, colocate_with_primary=var.op.type in ['Variable', 'VariableV2', 'VarHandleOp'])
                    if self._zero_debias:
                        zero_debias_true.add(avg)
            self._averages[var] = avg
        with ops.name_scope(self._name) as scope:
            num_updates = math_ops.cast(self._num_updates, dtypes.float32, name='num_updates')
            decay = num_updates / (1.0 + num_updates)
            decay = array_ops.identity(decay, name='decay')
            updates = []
            for var in var_list:
                zero_debias = self._averages[var] in zero_debias_true
                updates.append(assign_moving_average(self._averages[var], var, decay, zero_debias=zero_debias))
            return control_flow_ops.group(*updates, name=scope)

    def average(self, var):
        """Returns the `Variable` holding the average of `var`.

    Args:
      var: A `Variable` object.

    Returns:
      A `Variable` object or `None` if the moving average of `var`
      is not maintained.
    """
        return self._averages.get(var, None)

    def average_name(self, var):
        """Returns the name of the `Variable` holding the average for `var`.

    The typical scenario for `ExponentialMovingAverage` is to compute moving
    averages of variables during training, and restore the variables from the
    computed moving averages during evaluations.

    To restore variables, you have to know the name of the shadow variables.
    That name and the original variable can then be passed to a `Saver()` object
    to restore the variable from the moving average value with:
      `saver = tf.train.Saver({ema.average_name(var): var})`

    `average_name()` can be called whether or not `apply()` has been called.

    Args:
      var: A `Variable` object.

    Returns:
      A string: The name of the variable that will be used or was used
      by the `ExponentialMovingAverage class` to hold the moving average of
      `var`.
    """
        if var in self._averages:
            return self._averages[var].op.name
        return ops.get_default_graph().unique_name(var.op.name + '/' + self._name, mark_as_used=False)

    def variables_to_restore(self, moving_avg_variables=None):
        """Returns a map of names to `Variables` to restore.

    If a variable has a moving average, use the moving average variable name as
    the restore name; otherwise, use the variable name.

    For example,

    ```python
      variables_to_restore = ema.variables_to_restore()
      saver = tf.train.Saver(variables_to_restore)
    ```

    Below is an example of such mapping:

    ```
      conv/batchnorm/gamma/ExponentialMovingAverage: conv/batchnorm/gamma,
      conv_4/conv2d_params/ExponentialMovingAverage: conv_4/conv2d_params,
      global_step: global_step
    ```
    Args:
      moving_avg_variables: a list of variables that require to use of the
        moving variable name to be restored. If None, it will default to
        variables.moving_average_variables() + variables.trainable_variables()

    Returns:
      A map from restore_names to variables. The restore_name can be the
      moving_average version of the variable name if it exist, or the original
      variable name.
    """
        name_map = {}
        if moving_avg_variables is None:
            moving_avg_variables = variables.trainable_variables()
            moving_avg_variables += variables.moving_average_variables()
        moving_avg_variables = set(moving_avg_variables)
        for v in moving_avg_variables:
            name_map[self.average_name(v)] = v
        moving_avg_variable_names = set([v.name for v in moving_avg_variables])
        for v in list(set(variables.global_variables())):
            if v.name not in moving_avg_variable_names and v.op.name not in name_map:
                name_map[v.op.name] = v
        return name_map

