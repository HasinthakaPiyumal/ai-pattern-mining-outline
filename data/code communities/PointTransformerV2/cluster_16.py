# Cluster 16

def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')
    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if 'offset' in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)

def point_collate_fn(batch, max_batch_points=10000000000.0, mix_prob=0):
    assert isinstance(batch[0], Mapping)
    batch = collate_fn(batch)
    if 'offset' in batch.keys():
        assert batch['offset'][0] <= max_batch_points
        for i in range(len(batch['offset']) - 1):
            if batch['offset'][i + 1] > max_batch_points:
                batch['offset'] = batch['offset'][:i + 1]
                for key in batch.keys():
                    if key != 'offset':
                        batch[key] = batch[key][:batch['offset'][-1]]
                break
        if random.random() < mix_prob:
            batch['offset'] = torch.cat([batch['offset'][1:-1:2], batch['offset'][-1].unsqueeze(0)], dim=0)
    return batch

