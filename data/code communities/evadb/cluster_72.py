# Cluster 72

class BatchTest(unittest.TestCase):

    def test_batch_serialize_deserialize(self):
        batch = Batch(frames=create_dataframe())
        batch2 = Batch.deserialize(batch.serialize())
        self.assertEqual(batch, batch2)

    def test_frames_as_numpy_array_should_frames_as_numpy_array(self):
        batch = Batch(frames=create_dataframe_same(2))
        expected = list(np.ones((2, 1, 1)))
        actual = list(batch.column_as_numpy_array(batch.columns[0]))
        self.assertEqual(expected, actual)

    def test_return_only_frames_specified_in_the_indices(self):
        batch = Batch(frames=create_dataframe(2))
        expected = Batch(frames=create_dataframe())
        output = batch[[0]]
        self.assertEqual(expected, output)

    def test_fetching_frames_by_index(self):
        batch = Batch(frames=create_dataframe_same(2))
        expected = Batch(frames=create_dataframe())
        self.assertEqual(expected, batch[0])

    def test_fetching_frames_by_index_should_raise(self):
        batch = Batch(frames=create_dataframe_same(2))
        with self.assertRaises(TypeError):
            batch[1.0]

    def test_slicing_on_batched_should_return_new_batch_frame(self):
        batch = Batch(frames=create_dataframe(2))
        expected = Batch(frames=create_dataframe())
        self.assertEqual(batch, batch[:])
        self.assertEqual(expected, batch[:-1])

    def test_slicing_should_word_for_negative_stop_value(self):
        batch = Batch(frames=create_dataframe(2))
        expected = Batch(frames=create_dataframe())
        self.assertEqual(expected, batch[:-1])

    def test_slicing_should_work_with_skip_value(self):
        batch = Batch(frames=create_dataframe(3))
        expected = Batch(frames=create_dataframe(3).iloc[[0, 2], :])
        self.assertEqual(expected, batch[::2])

    def test_add_should_raise_error_for_incompatible_type(self):
        batch = Batch(frames=create_dataframe())
        with self.assertRaises(TypeError):
            batch + 1

    def test_adding_to_empty_frame_batch_returns_itself(self):
        batch_1 = Batch(frames=pd.DataFrame())
        batch_2 = Batch(frames=create_dataframe())
        self.assertEqual(batch_2, batch_1 + batch_2)
        self.assertEqual(batch_2, batch_2 + batch_1)

    def test_adding_batch_frame_with_outcomes_returns_new_batch_frame(self):
        batch_1 = Batch(frames=create_dataframe())
        batch_2 = Batch(frames=create_dataframe())
        batch_3 = Batch(frames=create_dataframe_same(2))
        self.assertEqual(batch_3, batch_1 + batch_2)

    def test_concat_batch(self):
        batch_1 = Batch(frames=create_dataframe())
        batch_2 = Batch(frames=create_dataframe())
        batch_3 = Batch(frames=create_dataframe_same(2))
        self.assertEqual(batch_3, Batch.concat([batch_1, batch_2], copy=False))

    def test_concat_empty_batch_list_raise_exception(self):
        self.assertEqual(Batch(), Batch.concat([]))

    def test_project_batch_frame(self):
        batch_1 = Batch(frames=pd.DataFrame([{'id': 1, 'data': 2, 'info': 3}]))
        batch_2 = batch_1.project(['id', 'info'])
        batch_3 = Batch(frames=pd.DataFrame([{'id': 1, 'info': 3}]))
        self.assertEqual(batch_2, batch_3)

    def test_merge_column_wise_batch_frame(self):
        batch_1 = Batch(frames=pd.DataFrame([{'id': 0}]))
        batch_2 = Batch(frames=pd.DataFrame([{'data': 1}]))
        batch_3 = Batch.merge_column_wise([batch_1, batch_2])
        batch_4 = Batch(frames=pd.DataFrame([{'id': 0, 'data': 1}]))
        self.assertEqual(batch_3, batch_4)
        self.assertEqual(Batch.merge_column_wise([]), Batch())
        batch_1 = Batch(frames=pd.DataFrame({'id': [0, None, 1]}))
        batch_2 = Batch(frames=pd.DataFrame({'data': [None, 0, None]}))
        batch_res = Batch(frames=pd.DataFrame({'id': [0, None, 1], 'data': [None, 0, None]}))
        self.assertEqual(Batch.merge_column_wise([batch_1, batch_2]), batch_res)
        df_1 = pd.DataFrame({'id': [-10, 1, 2]})
        df_2 = pd.DataFrame({'data': [-20, 2, 3]})
        df_1 = df_1[df_1 < 0].dropna()
        df_1.reset_index(drop=True, inplace=True)
        df_2 = df_2[df_2 < 0].dropna()
        df_2.reset_index(drop=True, inplace=True)
        batch_1 = Batch(frames=df_1)
        batch_2 = Batch(frames=df_2)
        df_res = pd.DataFrame({'id': [-10, 1, 2], 'data': [-20, 2, 3]})
        df_res = df_res[df_res < 0].dropna()
        df_res.reset_index(drop=True, inplace=True)
        batch_res = Batch(frames=df_res)
        self.assertEqual(Batch.merge_column_wise([batch_1, batch_2]), batch_res)

    def test_should_fail_for_list(self):
        frames = [{'id': 0, 'data': [1, 2]}, {'id': 1, 'data': [1, 2]}]
        self.assertRaises(ValueError, Batch, frames)

    def test_should_fail_for_dict(self):
        frames = {'id': 0, 'data': [1, 2]}
        self.assertRaises(ValueError, Batch, frames)

    def test_should_return_correct_length(self):
        batch = Batch(create_dataframe(5))
        self.assertEqual(5, len(batch))

    def test_should_return_empty_dataframe(self):
        batch = Batch()
        self.assertEqual(batch, Batch(create_dataframe(0)))

    def test_stack_batch_more_than_one_column_should_raise_exception(self):
        batch = Batch(create_dataframe_same(2))
        self.assertRaises(ValueError, Batch.stack, batch)

    def test_modify_column_alias_should_raise_exception(self):
        batch = Batch(create_dataframe(5))
        dummy_alias = Alias('dummy', col_names=['t1'])
        with self.assertRaises(RuntimeError):
            batch.modify_column_alias(dummy_alias)

    def test_drop_column_alias_should_work_on_frame_without_alias(self):
        batch = Batch(create_dataframe(5))
        batch.drop_column_alias()

    def test_sort_orderby_should_raise_exception_on_missing_column(self):
        batch = Batch(create_dataframe(5))
        with self.assertRaises(AssertionError):
            batch.sort_orderby(by=['foo'])

class PredicateExecutor(AbstractExecutor):
    """ """

    def __init__(self, db: EvaDBDatabase, node: PredicatePlan):
        super().__init__(db, node)
        self.predicate = node.predicate

    def exec(self, *args, **kwargs) -> Iterator[Batch]:
        child_executor = self.children[0]
        for batch in child_executor.exec(**kwargs):
            batch = apply_predicate(batch, self.predicate)
            if not batch.empty():
                yield batch
        instrument_function_expression_cost(self.predicate, self.catalog())

def apply_predicate(batch: Batch, predicate: AbstractExpression) -> Batch:
    if not batch.empty() and predicate is not None:
        outcomes = predicate.evaluate(batch)
        batch.drop_zero(outcomes)
        batch.reset_index()
    return batch

def instrument_function_expression_cost(expr: Union[AbstractExpression, List[AbstractExpression]], catalog: 'CatalogManager'):
    """We are expecting an instance of a catalog. An optimization can be to avoid creating a catalog instance if there is no function expression. An easy fix is to pass the function handler and create the catalog instance only if there is a function expression. In the past, this was problematic because of Ray. We can revisit it again."""
    if expr is None:
        return
    list_expr = expr
    if not isinstance(expr, list):
        list_expr = [expr]
    for expr in list_expr:
        for func_expr in expr.find_all(FunctionExpression):
            if func_expr.function_obj and func_expr._stats:
                function_id = func_expr.function_obj.row_id
                catalog.upsert_function_cost_catalog_entry(function_id, func_expr.function_obj.name, func_expr._stats.prev_cost)

class ProjectExecutor(AbstractExecutor):
    """ """

    def __init__(self, db: EvaDBDatabase, node: ProjectPlan):
        super().__init__(db, node)
        self.target_list = node.target_list

    def exec(self, *args, **kwargs) -> Iterator[Batch]:
        if len(self.children) == 0:
            dummy_batch = Batch(pd.DataFrame([0]))
            batch = apply_project(dummy_batch, self.target_list)
            if not batch.empty():
                yield batch
        elif len(self.children) == 1:
            child_executor = self.children[0]
            for batch in child_executor.exec(**kwargs):
                batch = apply_project(batch, self.target_list)
                if not batch.empty():
                    yield batch
        else:
            raise ExecutorError('ProjectExecutor has more than 1 children.')
        instrument_function_expression_cost(self.target_list, self.catalog())

def apply_project(batch: Batch, project_list: List[AbstractExpression]):
    if not batch.empty() and project_list:
        batches = [expr.evaluate(batch) for expr in project_list]
        batch = Batch.merge_column_wise(batches)
    return batch

class OrderByExecutor(AbstractExecutor):
    """
    Sort the frames which satisfy the condition

    Arguments:
        node (AbstractPlan): The OrderBy Plan

    """

    def __init__(self, db: EvaDBDatabase, node: OrderByPlan):
        super().__init__(db, node)
        self._orderby_list = node.orderby_list
        self._columns = node.columns
        self._sort_types = node.sort_types
        self.batch_sizes = []

    def _extract_column_name(self, col):
        col_name = []
        if isinstance(col, TupleValueExpression):
            col_name += [col.col_alias]
        elif isinstance(col, FunctionExpression):
            col_name += col.col_alias
        else:
            raise ExecutorError('Expression type {} is not supported.'.format(type(col)))
        return col_name

    def extract_column_names(self):
        """extracts the string name of the column"""
        col_name_list = []
        for col in self._columns:
            col_name_list += self._extract_column_name(col)
        return col_name_list

    def extract_sort_types(self):
        """extracts the sort type for the column"""
        sort_type_bools = []
        for st in self._sort_types:
            if st is ParserOrderBySortType.ASC:
                sort_type_bools.append(True)
            else:
                sort_type_bools.append(False)
        return sort_type_bools

    def exec(self, *args, **kwargs) -> Iterator[Batch]:
        child_executor = self.children[0]
        aggregated_batch_list = []
        for batch in child_executor.exec(**kwargs):
            self.batch_sizes.append(len(batch))
            aggregated_batch_list.append(batch)
        aggregated_batch = Batch.concat(aggregated_batch_list, copy=False)
        if not len(aggregated_batch):
            return
        merge_batch_list = [aggregated_batch]
        for col in self._columns:
            col_name_list = self._extract_column_name(col)
            for col_name in col_name_list:
                if col_name not in aggregated_batch.columns:
                    batch = col.evaluate(aggregated_batch)
                    merge_batch_list.append(batch)
        if len(merge_batch_list) > 1:
            aggregated_batch = Batch.merge_column_wise(merge_batch_list)
        try:
            aggregated_batch.sort_orderby(by=self.extract_column_names(), sort_type=self.extract_sort_types())
        except KeyError:
            pass
        index = 0
        for i in self.batch_sizes:
            batch = aggregated_batch[index:index + i]
            batch.reset_index()
            index += i
            yield batch

class NestedLoopJoinExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: NestedLoopJoinPlan):
        super().__init__(db, node)
        self.predicate = node.join_predicate

    def exec(self, *args, **kwargs) -> Iterator[Batch]:
        outer = self.children[0]
        inner = self.children[1]
        for row1 in outer.exec(**kwargs):
            for row2 in inner.exec(**kwargs):
                result_batch = Batch.join(row1, row2)
                result_batch.reset_index()
                result_batch = apply_predicate(result_batch, self.predicate)
                if not result_batch.empty():
                    yield result_batch
        if self.predicate:
            instrument_function_expression_cost(self.predicate, self.catalog())

class SequentialScanExecutor(AbstractExecutor):
    """
    Applies predicates to filter the frames which satisfy the condition
    Arguments:
        node (AbstractPlan): The SequentialScanPlan

    """

    def __init__(self, db: EvaDBDatabase, node: SeqScanPlan):
        super().__init__(db, node)
        self.predicate = node.predicate
        self.project_expr = node.columns
        self.alias = node.alias

    def exec(self, *args, **kwargs) -> Iterator[Batch]:
        child_executor = self.children[0]
        for batch in child_executor.exec(**kwargs):
            if self.alias:
                batch.modify_column_alias(self.alias)
            batch = apply_predicate(batch, self.predicate)
            batch = apply_project(batch, self.project_expr)
            if not batch.empty():
                yield batch
        if self.predicate or self.project_expr:
            catalog = self.catalog()
            instrument_function_expression_cost(self.predicate, catalog)
            instrument_function_expression_cost(self.project_expr, catalog)

class ApplyAndMergeExecutor(AbstractExecutor):
    """
    Apply the function expression to the input data, merge the output of the function
    with the input data, and yield the result to the parent. The current implementation
    assumes an inner join while merging. Therefore, if the function does not return any
    output, the input rows are dropped.
    Arguments:
        node (AbstractPlan): ApplyAndMergePlan

    """

    def __init__(self, db: EvaDBDatabase, node: ApplyAndMergePlan):
        super().__init__(db, node)
        self.func_expr = node.func_expr
        self.do_unnest = node.do_unnest
        self.alias = node.alias

    def exec(self, *args, **kwargs) -> Iterator[Batch]:
        child_executor = self.children[0]
        for batch in child_executor.exec(**kwargs):
            func_result = self.func_expr.evaluate(batch)
            output = Batch.merge_column_wise([batch, func_result])
            if self.do_unnest:
                output.unnest(func_result.columns)
                output.reset_index()
            yield output
        instrument_function_expression_cost(self.func_expr, self.catalog())

class HashJoinExecutor(AbstractExecutor):

    def __init__(self, db: EvaDBDatabase, node: HashJoinProbePlan):
        super().__init__(db, node)
        self.predicate = node.join_predicate
        self.join_type = node.join_type
        self.probe_keys = node.probe_keys
        self.join_project = node.join_project

    def exec(self, *args, **kwargs) -> Iterator[Batch]:
        build_table = self.children[0]
        probe_table = self.children[1]
        hash_keys = [key.col_alias for key in self.probe_keys]
        for build_batch in build_table.exec():
            for probe_batch in probe_table.exec():
                probe_batch.reassign_indices_to_hash(hash_keys)
                join_batch = Batch.join(probe_batch, build_batch)
                join_batch.reset_index()
                join_batch = apply_predicate(join_batch, self.predicate)
                join_batch = apply_project(join_batch, self.join_project)
                yield join_batch
        if self.predicate or self.join_project:
            catalog = self.catalog()
            instrument_function_expression_cost(self.predicate, catalog)
            instrument_function_expression_cost(self.join_project, catalog)

class FunctionScanExecutor(AbstractExecutor):
    """
    Executes functional expression which yields a table of rows
    Arguments:
        node (AbstractPlan): FunctionScanPlan

    """

    def __init__(self, db: EvaDBDatabase, node: FunctionScanPlan):
        super().__init__(db, node)
        self.func_expr = node.func_expr
        self.do_unnest = node.do_unnest

    def exec(self, *args, **kwargs) -> Iterator[Batch]:
        assert 'lateral_input' in kwargs, 'Key lateral_input not passed to the FunctionScan'
        lateral_input = kwargs.get('lateral_input')
        if not lateral_input.empty():
            res = self.func_expr.evaluate(lateral_input)
            if not res.empty():
                if self.do_unnest:
                    res.unnest(res.columns)
                yield res
            instrument_function_expression_cost(self.func_expr, self.catalog())

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

