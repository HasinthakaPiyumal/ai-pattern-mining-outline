# Cluster 93

def test_compat_loader_args():
    cfg = ConfigDict(dict(data=dict(val=dict(), test=dict(), train=dict())))
    cfg = compat_loader_args(cfg)
    assert 'val_dataloader' in cfg.data
    assert 'train_dataloader' in cfg.data
    assert 'test_dataloader' in cfg.data
    cfg = ConfigDict(dict(data=dict(samples_per_gpu=1, persistent_workers=True, workers_per_gpu=1, val=dict(samples_per_gpu=3), test=dict(samples_per_gpu=2), train=dict())))
    cfg = compat_loader_args(cfg)
    assert cfg.data.train_dataloader.workers_per_gpu == 1
    assert cfg.data.train_dataloader.samples_per_gpu == 1
    assert cfg.data.train_dataloader.persistent_workers
    assert cfg.data.val_dataloader.workers_per_gpu == 1
    assert cfg.data.val_dataloader.samples_per_gpu == 3
    assert cfg.data.test_dataloader.workers_per_gpu == 1
    assert cfg.data.test_dataloader.samples_per_gpu == 2
    cfg = ConfigDict(dict(data=dict(samples_per_gpu=1, persistent_workers=True, workers_per_gpu=1, val=dict(samples_per_gpu=3), test=[dict(samples_per_gpu=2), dict(samples_per_gpu=3)], train=dict())))
    cfg = compat_loader_args(cfg)
    assert cfg.data.test_dataloader.samples_per_gpu == 3
    cfg = ConfigDict(dict(data=dict(samples_per_gpu=1, persistent_workers=True, workers_per_gpu=1, val=dict(samples_per_gpu=3), test=dict(samples_per_gpu=2), train=dict(), train_dataloader=dict(samples_per_gpu=2))))
    with pytest.raises(AssertionError):
        compat_loader_args(cfg)
    cfg = ConfigDict(dict(data=dict(samples_per_gpu=1, persistent_workers=True, workers_per_gpu=1, val=dict(samples_per_gpu=3), test=dict(samples_per_gpu=2), train=dict(), val_dataloader=dict(samples_per_gpu=2))))
    with pytest.raises(AssertionError):
        compat_loader_args(cfg)
    cfg = ConfigDict(dict(data=dict(samples_per_gpu=1, persistent_workers=True, workers_per_gpu=1, val=dict(samples_per_gpu=3), test=dict(samples_per_gpu=2), test_dataloader=dict(samples_per_gpu=2))))
    with pytest.raises(AssertionError):
        compat_loader_args(cfg)

def compat_loader_args(cfg):
    """Deprecated sample_per_gpu in cfg.data."""
    cfg = copy.deepcopy(cfg)
    if 'train_dataloader' not in cfg.data:
        cfg.data['train_dataloader'] = ConfigDict()
    if 'val_dataloader' not in cfg.data:
        cfg.data['val_dataloader'] = ConfigDict()
    if 'test_dataloader' not in cfg.data:
        cfg.data['test_dataloader'] = ConfigDict()
    if 'samples_per_gpu' in cfg.data:
        samples_per_gpu = cfg.data.pop('samples_per_gpu')
        assert 'samples_per_gpu' not in cfg.data.train_dataloader, '`samples_per_gpu` are set in `data` field and ` data.train_dataloader` at the same time. Please only set it in `data.train_dataloader`. '
        cfg.data.train_dataloader['samples_per_gpu'] = samples_per_gpu
    if 'persistent_workers' in cfg.data:
        persistent_workers = cfg.data.pop('persistent_workers')
        assert 'persistent_workers' not in cfg.data.train_dataloader, '`persistent_workers` are set in `data` field and ` data.train_dataloader` at the same time. Please only set it in `data.train_dataloader`. '
        cfg.data.train_dataloader['persistent_workers'] = persistent_workers
    if 'workers_per_gpu' in cfg.data:
        workers_per_gpu = cfg.data.pop('workers_per_gpu')
        cfg.data.train_dataloader['workers_per_gpu'] = workers_per_gpu
        cfg.data.val_dataloader['workers_per_gpu'] = workers_per_gpu
        cfg.data.test_dataloader['workers_per_gpu'] = workers_per_gpu
    if 'samples_per_gpu' in cfg.data.val:
        assert 'samples_per_gpu' not in cfg.data.val_dataloader, '`samples_per_gpu` are set in `data.val` field and ` data.val_dataloader` at the same time. Please only set it in `data.val_dataloader`. '
        cfg.data.val_dataloader['samples_per_gpu'] = cfg.data.val.pop('samples_per_gpu')
    if isinstance(cfg.data.test, dict):
        if 'samples_per_gpu' in cfg.data.test:
            assert 'samples_per_gpu' not in cfg.data.test_dataloader, '`samples_per_gpu` are set in `data.test` field and ` data.test_dataloader` at the same time. Please only set it in `data.test_dataloader`. '
            cfg.data.test_dataloader['samples_per_gpu'] = cfg.data.test.pop('samples_per_gpu')
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            if 'samples_per_gpu' in ds_cfg:
                assert 'samples_per_gpu' not in cfg.data.test_dataloader, '`samples_per_gpu` are set in `data.test` field and ` data.test_dataloader` at the same time. Please only set it in `data.test_dataloader`. '
        samples_per_gpu = max([ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        cfg.data.test_dataloader['samples_per_gpu'] = samples_per_gpu
    return cfg

