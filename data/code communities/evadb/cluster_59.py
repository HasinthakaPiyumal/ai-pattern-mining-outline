# Cluster 59

class ExecutionContextTest(unittest.TestCase):

    @patch('evadb.executor.execution_context.get_gpu_count')
    @patch('evadb.executor.execution_context.is_gpu_available')
    def test_CUDA_VISIBLE_DEVICES_gets_populated_from_config(self, gpu_check, get_gpu_count):
        gpu_check.return_value = True
        get_gpu_count.return_value = 3
        context = Context([0, 1])
        self.assertEqual(context.gpus, [0, 1])

    @patch('evadb.executor.execution_context.os')
    @patch('evadb.executor.execution_context.get_gpu_count')
    @patch('evadb.executor.execution_context.is_gpu_available')
    def test_CUDA_VISIBLE_DEVICES_gets_populated_from_environment_if_no_config(self, is_gpu, get_gpu_count, os):
        is_gpu.return_value = True
        get_gpu_count.return_value = 3
        os.environ.get.return_value = '0,1'
        context = Context()
        os.environ.get.assert_called_with('CUDA_VISIBLE_DEVICES', '')
        self.assertEqual(context.gpus, [0, 1])

    @patch('evadb.executor.execution_context.os')
    @patch('evadb.executor.execution_context.get_gpu_count')
    @patch('evadb.executor.execution_context.is_gpu_available')
    def test_CUDA_VISIBLE_DEVICES_should_be_empty_if_nothing_provided(self, gpu_check, get_gpu_count, os):
        gpu_check.return_value = True
        get_gpu_count.return_value = 3
        os.environ.get.return_value = ''
        context = Context()
        os.environ.get.assert_called_with('CUDA_VISIBLE_DEVICES', '')
        self.assertEqual(context.gpus, [])

    @patch('evadb.executor.execution_context.os')
    @patch('evadb.executor.execution_context.is_gpu_available')
    def test_gpus_ignores_config_if_no_gpu_available(self, gpu_check, os):
        gpu_check.return_value = False
        os.environ.get.return_value = '0,1,2'
        context = Context([0, 1, 2])
        self.assertEqual(context.gpus, [])

    @patch('evadb.executor.execution_context.os')
    @patch('evadb.executor.execution_context.is_gpu_available')
    def test_gpu_device_should_return_NO_GPU_if_GPU_not_available(self, gpu_check, os):
        gpu_check.return_value = True
        os.environ.get.return_value = ''
        context = Context()
        os.environ.get.assert_called_with('CUDA_VISIBLE_DEVICES', '')
        self.assertEqual(context.gpu_device(), NO_GPU)

    @patch('evadb.executor.execution_context.get_gpu_count')
    @patch('evadb.executor.execution_context.is_gpu_available')
    def test_should_return_random_gpu_ID_if_available(self, gpu_check, get_gpu_count):
        gpu_check.return_value = True
        get_gpu_count.return_value = 1
        context = Context([0, 1, 2])
        selected_device = context.gpu_device()
        self.assertEqual(selected_device, 0)

class LogicalProjectToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALPROJECT)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_PROJECT_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_PROJECT_TO_PHYSICAL

    def check(self, grp_id: int, context: OptimizerContext):
        return True

    def apply(self, before: LogicalProject, context: OptimizerContext):
        after = ProjectPlan(before.target_list)
        for child in before.children:
            after.append_child(child)
        yield after

class LogicalProjectNoTableToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALPROJECT)
        super().__init__(RuleType.LOGICAL_PROJECT_NO_TABLE_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_PROJECT_NO_TABLE_TO_PHYSICAL

    def check(self, grp_id: int, context: OptimizerContext):
        return True

    def apply(self, before: LogicalProject, context: OptimizerContext):
        after = ProjectPlan(before.target_list)
        yield after

class LogicalApplyAndMergeToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICAL_APPLY_AND_MERGE)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_APPLY_AND_MERGE_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_APPLY_AND_MERGE_TO_PHYSICAL

    def check(self, grp_id: int, context: OptimizerContext):
        return True

    def apply(self, before: LogicalApplyAndMerge, context: OptimizerContext):
        after = ApplyAndMergePlan(before.func_expr, before.alias, before.do_unnest)
        for child in before.children:
            after.append_child(child)
        yield after

def get_ray_env_dict():
    if len(Context().gpus) > 0:
        max_gpu_id = max(Context().gpus) + 1
        return {'CUDA_VISIBLE_DEVICES': ','.join([str(n) for n in range(max_gpu_id)])}
    else:
        return {}

class LogicalExchangeToPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALEXCHANGE)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_EXCHANGE_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_EXCHANGE_TO_PHYSICAL

    def check(self, grp_id: int, context: OptimizerContext):
        return True

    def apply(self, before: LogicalExchange, context: OptimizerContext):
        after = ExchangePlan(before.view)
        for child in before.children:
            after.append_child(child)
        yield after

class LogicalApplyAndMergeToRayPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICAL_APPLY_AND_MERGE)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_APPLY_AND_MERGE_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_APPLY_AND_MERGE_TO_PHYSICAL

    def check(self, grp_id: int, context: OptimizerContext):
        return True

    def apply(self, before: LogicalApplyAndMerge, context: OptimizerContext):
        apply_plan = ApplyAndMergePlan(before.func_expr, before.alias, before.do_unnest)
        parallelism = 2
        ray_process_env_dict = get_ray_env_dict()
        ray_parallel_env_conf_dict = [ray_process_env_dict for _ in range(parallelism)]
        exchange_plan = ExchangePlan(inner_plan=apply_plan, parallelism=parallelism, ray_pull_env_conf_dict=ray_process_env_dict, ray_parallel_env_conf_dict=ray_parallel_env_conf_dict)
        for child in before.children:
            exchange_plan.append_child(child)
        yield exchange_plan

class LogicalProjectToRayPhysical(Rule):

    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALPROJECT)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_PROJECT_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_PROJECT_TO_PHYSICAL

    def check(self, before: LogicalProject, context: OptimizerContext):
        return True

    def apply(self, before: LogicalProject, context: OptimizerContext):
        project_plan = ProjectPlan(before.target_list)
        if before.target_list is None or not any([isinstance(expr, FunctionExpression) for expr in before.target_list]):
            for child in before.children:
                project_plan.append_child(child)
            yield project_plan
        else:
            parallelism = 2
            ray_process_env_dict = get_ray_env_dict()
            ray_parallel_env_conf_dict = [ray_process_env_dict for _ in range(parallelism)]
            exchange_plan = ExchangePlan(inner_plan=project_plan, parallelism=parallelism, ray_pull_env_conf_dict=ray_process_env_dict, ray_parallel_env_conf_dict=ray_parallel_env_conf_dict)
            for child in before.children:
                exchange_plan.append_child(child)
            yield exchange_plan

class FunctionExpression(AbstractExpression):
    """
    Consider FunctionExpression: ObjDetector -> (labels, boxes)

    `output`: If the user wants only subset of outputs. Eg,
    ObjDetector.labels the parser with set output to 'labels'

    `output_objs`: It is populated by the binder. In case the
    output is None, the binder sets output_objs to list of all
    output columns of the FunctionExpression. Eg, ['labels',
    'boxes']. Otherwise, only the output columns.

    FunctionExpression also needs to prepend its alias to all the
    projected columns. This is important as other parts of the query
    might be assessing the results using alias. Eg,

    `Select Detector.labels
     FROM Video JOIN LATERAL ObjDetector AS Detector;`
    """

    def __init__(self, func: Callable, name: str, output: str=None, alias: Alias=None, **kwargs):
        super().__init__(ExpressionType.FUNCTION_EXPRESSION, **kwargs)
        self._context = Context()
        self._name = name
        self._function = func
        self._function_instance = None
        self._output = output
        self.alias = alias
        self.function_obj: FunctionCatalogEntry = None
        self.output_objs: List[FunctionIOCatalogEntry] = []
        self.projection_columns: List[str] = []
        self._cache: FunctionExpressionCache = None
        self._stats = FunctionStats()

    @property
    def name(self):
        return self._name

    @property
    def output(self):
        return self._output

    @property
    def col_alias(self):
        col_alias_list = []
        if self.alias is not None:
            for col in self.alias.col_names:
                col_alias_list.append('{}.{}'.format(self.alias.alias_name, col))
        return col_alias_list

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, func: Callable):
        self._function = func

    def enable_cache(self, cache: 'FunctionExpressionCache'):
        self._cache = cache
        return self

    def has_cache(self):
        return self._cache is not None

    def consolidate_stats(self):
        if self.function_obj is None:
            return
        if self.has_cache() and self._stats.cache_misses > 0:
            cost_per_func_call = self._stats.timer.total_elapsed_time / self._stats.cache_misses
        else:
            cost_per_func_call = self._stats.timer.total_elapsed_time / self._stats.num_calls
        if abs(self._stats.prev_cost - cost_per_func_call) > cost_per_func_call / 10:
            self._stats.prev_cost = cost_per_func_call

    def evaluate(self, batch: Batch, **kwargs) -> Batch:
        func = self._gpu_enabled_function()
        with self._stats.timer:
            outcomes = self._apply_function_expression(func, batch, **kwargs)
            if outcomes.frames.empty is False:
                outcomes = outcomes.project(self.projection_columns)
                outcomes.modify_column_alias(self.alias)
        self._stats.num_calls += len(batch)
        try:
            self.consolidate_stats()
        except Exception as e:
            logger.warn(f'Persisting FunctionExpression {str(self)} stats failed with {str(e)}')
        return outcomes

    def signature(self) -> str:
        """It constructs the signature of the function expression.
        It traverses the children (function arguments) and compute signature for each
        child. The output is in the form `function_name[row_id](arg1, arg2, ...)`.

        Returns:
            str: signature string
        """
        child_sigs = []
        for child in self.children:
            child_sigs.append(child.signature())
        func_sig = f'{self.name}[{self.function_obj.row_id}]({','.join(child_sigs)})'
        return func_sig

    def _gpu_enabled_function(self):
        if self._function_instance is None:
            self._function_instance = self.function()
            if isinstance(self._function_instance, GPUCompatible):
                device = self._context.gpu_device()
                if device != NO_GPU:
                    self._function_instance = self._function_instance.to_device(device)
        return self._function_instance

    def _apply_function_expression(self, func: Callable, batch: Batch, **kwargs):
        """
        If cache is not enabled, call the func on the batch and return.
        If cache is enabled:
        (1) iterate over the input batch rows and check if we have the value in the
        cache;
        (2) for all cache miss rows, call the func;
        (3) iterate over each cache miss row and store the results in the cache;
        (4) stitch back the partial cache results with the new func calls.
        """
        func_args = Batch.merge_column_wise([child.evaluate(batch, **kwargs) for child in self.children])
        if not self._cache:
            return func_args.apply_function_expression(func)
        output_cols = [obj.name for obj in self.function_obj.outputs]
        results = np.full([len(batch), len(output_cols)], None)
        cache_keys = func_args
        if self._cache.key:
            cache_keys = Batch.merge_column_wise([child.evaluate(batch, **kwargs) for child in self._cache.key])
            assert len(cache_keys) == len(batch), 'Not all rows have the cache key'
        cache_miss = np.full(len(batch), True)
        for idx, (_, key) in enumerate(cache_keys.iterrows()):
            val = self._cache.store.get(key.to_numpy())
            results[idx] = val
            cache_miss[idx] = val is None
        self._stats.cache_misses += sum(cache_miss)
        if cache_miss.any():
            func_args = func_args[list(cache_miss)]
            cache_miss_results = func_args.apply_function_expression(func)
            missing_keys = cache_keys[list(cache_miss)]
            for key, value in zip(missing_keys.iterrows(), cache_miss_results.iterrows()):
                self._cache.store.set(key[1].to_numpy(), value[1].to_numpy())
            results[cache_miss] = cache_miss_results.to_numpy()
        return Batch(pd.DataFrame(results, columns=output_cols))

    def __str__(self) -> str:
        args = [str(child) for child in self.children]
        expr_str = f'{self.name}({','.join(args)})'
        return expr_str

    def __eq__(self, other):
        is_subtree_equal = super().__eq__(other)
        if not isinstance(other, FunctionExpression):
            return False
        return is_subtree_equal and self.name == other.name and (self.output == other.output) and (self.alias == other.alias) and (self.function == other.function) and (self.output_objs == other.output_objs) and (self._cache == other._cache)

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.name, self.output, self.alias, self.function, tuple(self.output_objs), self._cache))

