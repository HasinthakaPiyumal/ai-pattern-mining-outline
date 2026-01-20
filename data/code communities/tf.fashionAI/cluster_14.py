# Cluster 14

def split_dictionary(dictionary):
    """Split a dictionary into shards."""
    shards = [{} for _ in range(number_of_shards)]
    for name, tensor in six.iteritems(dictionary):
        if isinstance(tensor, sparse_tensor.SparseTensor):
            for i, shard in enumerate(sparse_ops.sparse_split(sp_input=tensor, num_split=number_of_shards, axis=0)):
                shards[i][name] = shard
        else:
            ensure_divisible_by_shards(tensor)
            for i, shard in enumerate(array_ops.split(tensor, number_of_shards)):
                shards[i][name] = shard
    return shards

def ensure_divisible_by_shards(sequence):
    batch_size = ops_lib.convert_to_tensor(sequence).get_shape()[0]
    if batch_size % number_of_shards != 0:
        raise ValueError('Batch size {} needs to be divisible by the number of GPUs, which is {}.'.format(batch_size, number_of_shards))

def _split_batch(features, labels, number_of_shards, device):
    """Split input features and labes into batches."""

    def ensure_divisible_by_shards(sequence):
        batch_size = ops_lib.convert_to_tensor(sequence).get_shape()[0]
        if batch_size % number_of_shards != 0:
            raise ValueError('Batch size {} needs to be divisible by the number of GPUs, which is {}.'.format(batch_size, number_of_shards))

    def split_dictionary(dictionary):
        """Split a dictionary into shards."""
        shards = [{} for _ in range(number_of_shards)]
        for name, tensor in six.iteritems(dictionary):
            if isinstance(tensor, sparse_tensor.SparseTensor):
                for i, shard in enumerate(sparse_ops.sparse_split(sp_input=tensor, num_split=number_of_shards, axis=0)):
                    shards[i][name] = shard
            else:
                ensure_divisible_by_shards(tensor)
                for i, shard in enumerate(array_ops.split(tensor, number_of_shards)):
                    shards[i][name] = shard
        return shards
    with ops_lib.name_scope('split_inputs'):
        with ops_lib.device(device):
            if isinstance(features, dict):
                feature_shards = split_dictionary(features)
            else:
                ensure_divisible_by_shards(features)
                feature_shards = array_ops.split(features, number_of_shards)
            if labels is None:
                label_shards = None
            elif isinstance(labels, dict):
                label_shards = split_dictionary(labels)
            else:
                ensure_divisible_by_shards(labels)
                label_shards = array_ops.split(labels, number_of_shards)
    return (feature_shards, label_shards)

