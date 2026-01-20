# Cluster 92

def test_compat_runner_args():
    cfg = ConfigDict(dict(total_epochs=12))
    with pytest.warns(None) as record:
        cfg = compat_runner_args(cfg)
    assert len(record) == 1
    assert 'runner' in record.list[0].message.args[0]
    assert 'runner' in cfg
    assert cfg.runner.type == 'EpochBasedRunner'
    assert cfg.runner.max_epochs == cfg.total_epochs

def compat_runner_args(cfg):
    if 'runner' not in cfg:
        cfg.runner = ConfigDict({'type': 'EpochBasedRunner', 'max_epochs': cfg.total_epochs})
        warnings.warn('config is now expected to have a `runner` section, please set `runner` in your config.', UserWarning)
    elif 'total_epochs' in cfg:
        assert cfg.total_epochs == cfg.runner.max_epochs
    return cfg

