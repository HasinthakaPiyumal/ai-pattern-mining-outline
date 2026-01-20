# Cluster 27

def elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple((elem_or_tuple_to_variable(e) for e in elem_or_tuple))
    return from_numpy(elem_or_tuple)

def np_to_pytorch_batch(np_batch):
    return {k: elem_or_tuple_to_variable(x) for k, x in filter_batch(np_batch) if x.dtype != np.dtype('O')}

def filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield (k, v.astype(int))
        else:
            yield (k, v)

def update(num_updates):
    rl_losses_agg = {}
    for update in range(num_updates):
        batch = ptu.np_to_pytorch_batch(policy_storage.random_episodes(batch_size))
        rl_losses = agent.update(batch)
        for k, v in rl_losses.items():
            if update == 0:
                rl_losses_agg[k] = [v]
            else:
                rl_losses_agg[k].append(v)
    for k in rl_losses_agg:
        rl_losses_agg[k] = np.mean(rl_losses_agg[k])
    return rl_losses_agg

