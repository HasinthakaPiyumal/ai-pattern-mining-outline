# Cluster 45

def build_optimizer(model, cfg):
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor', 'DefaultOptimizerConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = build_optimizer_constructor(dict(type=constructor_type, optimizer_cfg=optimizer_cfg, paramwise_cfg=paramwise_cfg))
    optimizer = optim_constructor(model)
    return optimizer

def build_optimizer_constructor(cfg):
    constructor_type = cfg.get('type')
    if constructor_type in OPTIMIZER_BUILDERS:
        return build_from_cfg(cfg, OPTIMIZER_BUILDERS)
    elif constructor_type in MMCV_OPTIMIZER_BUILDERS:
        return build_from_cfg(cfg, MMCV_OPTIMIZER_BUILDERS)
    else:
        raise KeyError(f'{constructor_type} is not registered in the optimizer builder registry.')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA support')
@patch('mmdet.apis.single_gpu_test', MagicMock)
@patch('mmdet.apis.multi_gpu_test', MagicMock)
@pytest.mark.parametrize('EvalHookCls', (EvalHook, DistEvalHook))
def test_eval_hook(EvalHookCls):
    with pytest.raises(TypeError):
        test_dataset = ExampleDataset()
        data_loader = [DataLoader(test_dataset, batch_size=1, sampler=None, num_worker=0, shuffle=False)]
        EvalHookCls(data_loader)
    with pytest.raises(KeyError):
        test_dataset = ExampleDataset()
        data_loader = DataLoader(test_dataset, batch_size=1, sampler=None, num_workers=0, shuffle=False)
        EvalHookCls(data_loader, save_best='auto', rule='unsupport')
    with pytest.raises(ValueError):
        test_dataset = ExampleDataset()
        data_loader = DataLoader(test_dataset, batch_size=1, sampler=None, num_workers=0, shuffle=False)
        EvalHookCls(data_loader, save_best='unsupport')
    optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
    test_dataset = ExampleDataset()
    loader = DataLoader(test_dataset, batch_size=1)
    model = ExampleModel()
    optimizer = build_optimizer(model, optimizer_cfg)
    data_loader = DataLoader(test_dataset, batch_size=1)
    eval_hook = EvalHookCls(data_loader, save_best=None)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, batch_processor=None, optimizer=optimizer, work_dir=tmpdir, logger=logger)
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 1)
        assert runner.meta is None or 'best_score' not in runner.meta['hook_msgs']
        assert runner.meta is None or 'best_ckpt' not in runner.meta['hook_msgs']
    loader = DataLoader(EvalDataset(), batch_size=1)
    model = ExampleModel()
    data_loader = DataLoader(EvalDataset(), batch_size=1)
    eval_hook = EvalHookCls(data_loader, interval=1, save_best='auto')
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, batch_processor=None, optimizer=optimizer, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)
        real_path = osp.join(tmpdir, 'best_mAP_epoch_4.pth')
        assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(real_path)
        assert runner.meta['hook_msgs']['best_score'] == 0.7
    loader = DataLoader(EvalDataset(), batch_size=1)
    model = ExampleModel()
    data_loader = DataLoader(EvalDataset(), batch_size=1)
    eval_hook = EvalHookCls(data_loader, interval=1, save_best='mAP')
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, batch_processor=None, optimizer=optimizer, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)
        real_path = osp.join(tmpdir, 'best_mAP_epoch_4.pth')
        assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(real_path)
        assert runner.meta['hook_msgs']['best_score'] == 0.7
    data_loader = DataLoader(EvalDataset(), batch_size=1)
    eval_hook = EvalHookCls(data_loader, interval=1, save_best='score', rule='greater')
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, batch_processor=None, optimizer=optimizer, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)
        real_path = osp.join(tmpdir, 'best_score_epoch_4.pth')
        assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(real_path)
        assert runner.meta['hook_msgs']['best_score'] == 0.7
    data_loader = DataLoader(EvalDataset(), batch_size=1)
    eval_hook = EvalHookCls(data_loader, save_best='mAP', rule='less')
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, batch_processor=None, optimizer=optimizer, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)
        real_path = osp.join(tmpdir, 'best_mAP_epoch_6.pth')
        assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(real_path)
        assert runner.meta['hook_msgs']['best_score'] == 0.05
    data_loader = DataLoader(EvalDataset(), batch_size=1)
    eval_hook = EvalHookCls(data_loader, save_best='mAP')
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(model=model, batch_processor=None, optimizer=optimizer, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 2)
        real_path = osp.join(tmpdir, 'best_mAP_epoch_2.pth')
        assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(real_path)
        assert runner.meta['hook_msgs']['best_score'] == 0.4
        resume_from = osp.join(tmpdir, 'latest.pth')
        loader = DataLoader(ExampleDataset(), batch_size=1)
        eval_hook = EvalHookCls(data_loader, save_best='mAP')
        runner = EpochBasedRunner(model=model, batch_processor=None, optimizer=optimizer, work_dir=tmpdir, logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.resume(resume_from)
        runner.run([loader], [('train', 1)], 8)
        real_path = osp.join(tmpdir, 'best_mAP_epoch_4.pth')
        assert runner.meta['hook_msgs']['best_ckpt'] == osp.realpath(real_path)
        assert runner.meta['hook_msgs']['best_score'] == 0.7

