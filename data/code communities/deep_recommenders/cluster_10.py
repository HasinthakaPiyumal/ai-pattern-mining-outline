# Cluster 10

class Streaming(TopK):
    """Retrieves top k scoring items and identifiers from large dataset."""

    def __init__(self, k: int=10, query_model: Optional[tf.keras.Model]=None, handle_incomplete_batches: bool=True, num_parallel_calls: int=tf.data.experimental.AUTOTUNE, sorted_order: bool=True, *args, **kwargs):
        super().__init__(k, *args, **kwargs)
        self._query_model = query_model
        self._handle_incomplete_batches = handle_incomplete_batches
        self._num_parallel_calls = num_parallel_calls
        self._sorted_order = sorted_order
        self._candidates = None
        self._identifiers = None
        self._counter = self.add_weight('counter', dtype=tf.int32, trainable=False)

    def index(self, candidates: tf.data.Dataset, identifiers: Optional[tf.data.Dataset]=None, **kwargs) -> 'Streaming':
        """构建索引
        Args:
            candidates: 候选embeddings的Dataset
            identifiers: 候选 embeddings对应标识的Dataset(Opt)
        Returns:
            Self.
        """
        self._candidates = candidates
        self._identifiers = identifiers
        return self

    def call(self, queries: Union[tf.Tensor, Dict[Text, tf.Tensor]], k: Optional[int]=None, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """检索索引
        args:
            queries: queries embeddings,
            k: 返回候选个数
        returns:
            Tuple(top k candidates scores, top k candidates identifiers)
        """
        k = k if k is not None else self._k
        if self._candidates is None:
            raise ValueError('The `index` method must be called first to create the retrieval index.')
        if self._query_model is not None:
            queries = self._query_model(queries)
        self._counter.assign(0)

        def top_scores(candidate_index: tf.Tensor, candidate_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            """计算一个batch的候选集中的topK的scores和indices"""
            scores = tf.matmul(queries, candidate_batch, transpose_b=True)
            if self._handle_incomplete_batches is True:
                k_ = tf.math.minimum(k, tf.shape(scores)[1])
            else:
                k_ = k
            scores, indices = tf.math.top_k(scores, k=k_, sorted=self._sorted_order)
            return (scores, tf.gather(candidate_index, indices))

        def top_k(state: Tuple[tf.Tensor, tf.Tensor], x: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
            """Reduction function.
            合并现在的topk和新的topk，重新从中选出topk
            """
            state_scores, state_indices = state
            x_scores, x_indices = x
            joined_scores = tf.concat([state_scores, x_scores], axis=1)
            joined_indices = tf.concat([state_indices, x_indices], axis=1)
            if self._handle_incomplete_batches is True:
                k_ = tf.math.minimum(k, tf.shape(joined_scores)[1])
            else:
                k_ = k
            scores, indices = tf.math.top_k(joined_scores, k=k_, sorted=self._sorted_order)
            return (scores, tf.gather(joined_indices, indices, batch_dims=1))
        if self._identifiers is not None:
            index_dtype = self._identifiers.element_spec.dtype
        else:
            index_dtype = tf.int32
        initial_state = (tf.zeros((tf.shape(queries)[0], 0), dtype=tf.float32), tf.zeros((tf.shape(queries)[0], 0), dtype=index_dtype))

        def enumerate_rows(batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            """Enumerates rows in each batch using a total element counter."""
            starting_counter = self._counter.read_value()
            end_counter = self._counter.assign_add(tf.shape(batch)[0])
            return (tf.range(starting_counter, end_counter), batch)
        if self._identifiers is not None:
            dataset = tf.data.Dataset.zip((self._identifiers, self._candidates))
        else:
            dataset = self._candidates.map(enumerate_rows)
        with _wrap_batch_too_small_error(k):
            result = dataset.map(top_scores, num_parallel_calls=self._num_parallel_calls).reduce(initial_state, top_k)
        return result

@contextlib.contextmanager
def _wrap_batch_too_small_error(k: int):
    """ Candidate batch too small error """
    try:
        yield
    except tf.errors.InvalidArgumentError as e:
        error_msg = str(e)
        if 'input must have at least k columns' in error_msg:
            raise ValueError('Tried to retrieve k={k} top items, but candidate batch too small.To resolve this, 1. increase batch-size, 2. set `drop_remainder`=True, 3. set `handle_incomplete_batches`=True in constructor.'.format(k=k))

