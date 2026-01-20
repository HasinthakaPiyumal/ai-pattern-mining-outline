# Cluster 9

def _exclude(scores: tf.Tensor, identifiers: tf.Tensor, exclude: tf.Tensor, k: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """从TopK中的items移除指定的候选item
    Args:
        scores: candidate scores. 2D
        identifiers: candidate identifiers. 2D
        exclude: identifiers to exclude. 2D
        k: 返回候选个数
    Returns:
        Tuple(top k candidates scores, top k candidates indentifiers)   
    """
    indents = tf.expand_dims(identifiers, -1)
    exclude = tf.expand_dims(exclude, 1)
    isin = tf.math.reduce_any(tf.math.equal(indents, exclude), -1)
    adjusted_scores = scores - tf.cast(isin, tf.float32) * 100000.0
    k = tf.math.minimum(k, tf.shape(scores)[1])
    _, indices = tf.math.top_k(adjusted_scores, k=k)
    return (_take_long_axis(scores, indices), _take_long_axis(identifiers, indices))

def _take_long_axis(arr: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
    """从原始数据arr中，根据indices指定的下标，取出元素
    Args:
        arr: 原始数据，2D
        indices: 下标，2D
    Returns:
        根据下标取出的数据，2D
    """
    row_indices = tf.tile(tf.expand_dims(tf.range(tf.shape(indices)[0]), 1), [1, tf.shape(indices)[1]])
    gather_indices = tf.concat([tf.reshape(row_indices, (-1, 1)), tf.reshape(indices, (-1, 1))], axis=1)
    return tf.reshape(tf.gather_nd(arr, gather_indices), tf.shape(indices))

class TopK(tf.keras.Model, abc.ABC):
    """TopK layer 接口
    注意，必须实现两个方法
    1、index: 创建索引
    2、call: 检索索引
    """

    def __init__(self, k: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._k = k

    @abc.abstractmethod
    def index(self, candidates: Union[tf.Tensor, tf.data.Dataset], identifiers: Optional[Union[tf.Tensor, tf.data.Dataset]]=None) -> 'TopK':
        """创建索引 
        args:
            candidates: 候选 embeddings
            identifiers: 候选 embeddings对应标识 (Opt)
        returns:
            Self.
        """
        raise NotImplementedError('Implementers must provide `index` method.')

    @abc.abstractmethod
    def call(self, queries: Union[tf.Tensor, Dict[Text, tf.Tensor]], k: Optional[int]=None, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """检索索引
        args:
            queries: queries embeddings,
            k: 返回候选个数
        returns:
            Tuple(top k candidates scores, top k candidates indentifiers)
        """
        raise NotImplementedError()

    @tf.function
    def query_with_exclusions(self, queries: Union[tf.Tensor, Dict[Text, tf.Tensor]], exclusions: tf.Tensor, k: Optional[int]=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """检索索引并过滤exclusions
        Args:
            queries: queries embeddings,
            exclusions: candidates identifiers. 从TopK的候选集中过滤指定的item.
            k: 返回候选个数
        Returns:
            Tuple(top k candidates scores, top k candidates indetifiers)
        """
        k = k if k is not None else self._k
        adjusted_k = k + exclusions.shape[1]
        scores, identifiers = self(queries=queries, k=adjusted_k)
        return _exclude(scores, identifiers, exclusions, adjusted_k)

    def _reset_tf_function_cache(self):
        """Resets the tf.function cache."""
        if hasattr(self.query_with_exclusions, 'python_function'):
            self.query_with_exclusions = tf.function(self.query_with_exclusions.python_function)

