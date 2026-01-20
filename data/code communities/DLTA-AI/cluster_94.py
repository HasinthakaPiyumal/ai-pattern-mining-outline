# Cluster 94

def test_compat_imgs_per_gpu():
    cfg = ConfigDict(dict(data=dict(imgs_per_gpu=1, samples_per_gpu=2, val=dict(), test=dict(), train=dict())))
    cfg = compat_imgs_per_gpu(cfg)
    assert cfg.data.samples_per_gpu == cfg.data.imgs_per_gpu

def compat_imgs_per_gpu(cfg):
    cfg = copy.deepcopy(cfg)
    if 'imgs_per_gpu' in cfg.data:
        warnings.warn('"imgs_per_gpu" is deprecated in MMDet V2.0. Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            warnings.warn(f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and "samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            warnings.warn(f'Automatically set "samples_per_gpu"="imgs_per_gpu"={cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
    return cfg

