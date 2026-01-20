# Cluster 21

def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDetection'] = mmdet.__version__ + '+' + get_git_hash()[:7]
    return env_info

def get_device():
    """Returns an available device, cpu, cuda or mlu."""
    is_device_available = {'npu': is_npu_available(), 'cuda': torch.cuda.is_available(), 'mlu': is_mlu_available()}
    device_list = [k for k, v in is_device_available.items() if v]
    return device_list[0] if len(device_list) >= 1 else 'cpu'

def is_npu_available():
    """Returns a bool indicating if NPU is currently available."""
    return hasattr(torch, 'npu') and torch.npu.is_available()

def is_mlu_available():
    """Returns a bool indicating if MLU is currently available."""
    return hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()

def train_detector(model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None):
    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner['type']
    train_dataloader_default_args = dict(samples_per_gpu=2, workers_per_gpu=2, num_gpus=len(cfg.gpu_ids), dist=distributed, seed=cfg.seed, runner_type=runner_type, persistent_workers=False)
    train_loader_cfg = {**train_dataloader_default_args, **cfg.data.get('train_dataloader', {})}
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = build_ddp(model, cfg.device, device_ids=[int(os.environ['LOCAL_RANK'])], broadcast_buffers=False, find_unused_parameters=find_unused_parameters)
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    auto_scale_lr(cfg, distributed, logger)
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = build_runner(cfg.runner, default_args=dict(model=model, optimizer=optimizer, work_dir=cfg.work_dir, logger=logger, meta=meta))
    runner.timestamp = timestamp
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is None and cfg.get('device', None) == 'npu':
        fp16_cfg = dict(loss_scale='dynamic')
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config, cfg.checkpoint_config, cfg.log_config, cfg.get('momentum_config', None), custom_hooks_config=cfg.get('custom_hooks', None))
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())
    if validate:
        val_dataloader_default_args = dict(samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False, persistent_workers=False)
        val_dataloader_args = {**val_dataloader_default_args, **cfg.data.get('val_dataloader', {})}
        if val_dataloader_args['samples_per_gpu'] > 1:
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority='LOW')
    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)

def compat_cfg(cfg):
    """This function would modify some filed to keep the compatibility of
    config.

    For example, it will move some args which will be deprecated to the correct
    fields.
    """
    cfg = copy.deepcopy(cfg)
    cfg = compat_imgs_per_gpu(cfg)
    cfg = compat_loader_args(cfg)
    cfg = compat_runner_args(cfg)
    return cfg

def build_ddp(model, device='cuda', *args, **kwargs):
    """Build DistributedDataParallel module by device type.

    If device is cuda, return a MMDistributedDataParallel model;
    if device is mlu, return a MLUDistributedDataParallel model.

    Args:
        model (:class:`nn.Module`): module to be parallelized.
        device (str): device type, mlu or cuda.

    Returns:
        :class:`nn.Module`: the module to be parallelized

    References:
        .. [1] https://pytorch.org/docs/stable/generated/torch.nn.parallel.
                     DistributedDataParallel.html
    """
    assert device in ['cuda', 'mlu', 'npu'], 'Only available for cuda or mlu or npu devices.'
    if device == 'npu':
        from mmcv.device.npu import NPUDistributedDataParallel
        torch.npu.set_compile_mode(jit_compile=False)
        ddp_factory['npu'] = NPUDistributedDataParallel
        model = model.npu()
    elif device == 'cuda':
        model = model.cuda()
    elif device == 'mlu':
        from mmcv.device.mlu import MLUDistributedDataParallel
        ddp_factory['mlu'] = MLUDistributedDataParallel
        model = model.mlu()
    return ddp_factory[device](model, *args, **kwargs)

def build_dp(model, device='cuda', dim=0, *args, **kwargs):
    """build DataParallel module by device type.

    if device is cuda, return a MMDataParallel model; if device is mlu,
    return a MLUDataParallel model.

    Args:
        model (:class:`nn.Module`): model to be parallelized.
        device (str): device type, cuda, cpu or mlu. Defaults to cuda.
        dim (int): Dimension used to scatter the data. Defaults to 0.

    Returns:
        nn.Module: the model to be parallelized.
    """
    if device == 'npu':
        from mmcv.device.npu import NPUDataParallel
        dp_factory['npu'] = NPUDataParallel
        torch.npu.set_device(kwargs['device_ids'][0])
        torch.npu.set_compile_mode(jit_compile=False)
        model = model.npu()
    elif device == 'cuda':
        model = model.cuda(kwargs['device_ids'][0])
    elif device == 'mlu':
        from mmcv.device.mlu import MLUDataParallel
        dp_factory['mlu'] = MLUDataParallel
        model = model.mlu()
    return dp_factory[device](model, *args, dim=dim, **kwargs)

def auto_scale_lr(cfg, distributed, logger):
    """Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg (config): Training config.
        distributed (bool): Using distributed or not.
        logger (logging.Logger): Logger.
    """
    if 'auto_scale_lr' not in cfg or not cfg.auto_scale_lr.get('enable', False):
        logger.info('Automatic scaling of learning rate (LR) has been disabled.')
        return
    base_batch_size = cfg.auto_scale_lr.get('base_batch_size', None)
    if base_batch_size is None:
        return
    if distributed:
        _, world_size = get_dist_info()
        num_gpus = len(range(world_size))
    else:
        num_gpus = len(cfg.gpu_ids)
    samples_per_gpu = cfg.data.train_dataloader.samples_per_gpu
    batch_size = num_gpus * samples_per_gpu
    logger.info(f'Training with {num_gpus} GPU(s) with {samples_per_gpu} samples per GPU. The total batch size is {batch_size}.')
    if batch_size != base_batch_size:
        scaled_lr = batch_size / base_batch_size * cfg.optimizer.lr
        logger.info(f'LR has been automatically scaled from {cfg.optimizer.lr} to {scaled_lr}')
        cfg.optimizer.lr = scaled_lr
    else:
        logger.info(f'The batch size match the base batch size: {base_batch_size}, will not scaling the LR ({cfg.optimizer.lr}).')

def find_latest_checkpoint(path, suffix='pth'):
    """Find the latest checkpoint from the working directory.

    Args:
        path(str): The path to find checkpoints.
        suffix(str): File extension.
            Defaults to pth.

    Returns:
        latest_path(str | None): File path of the latest checkpoint.
    References:
        .. [1] https://github.com/microsoft/SoftTeacher
                  /blob/main/ssod/utils/patch.py
    """
    if not osp.exists(path):
        warnings.warn('The path of checkpoints does not exist.')
        return None
    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')
    checkpoints = glob.glob(osp.join(path, f'*.{suffix}'))
    if len(checkpoints) == 0:
        warnings.warn('There are no checkpoints in the path.')
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split('_')[-1].split('.')[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path

class EvalHook(BaseEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(EvalHook, self).__init__(*args, **kwargs)
        self.latest_results = None
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, progress + 1)
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return
        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.latest_results = results
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)

def single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]
                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None
                model.module.show_result(img_show, result[i], bbox_color=PALETTE, text_color=PALETTE, mask_color=PALETTE, show=show, out_file=out_file, score_thr=show_score_thr)
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results)) for bbox_results, mask_results in result]
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results, encode_mask_results(mask_results))
        results.extend(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results

class DistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(DistEvalHook, self).__init__(*args, **kwargs)
        self.latest_results = None
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, progress + 1)
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)
        if not self._should_evaluate(runner):
            return
        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')
        from mmdet.apis import multi_gpu_test
        results = multi_gpu_test(runner.model, self.dataloader, tmpdir=tmpdir, gpu_collect=self.gpu_collect)
        self.latest_results = results
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)

def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results)) for bbox_results, mask_results in result]
            elif isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (bbox_results, encode_mask_results(mask_results))
        results.extend(result)
        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results

def eval_map(det_results, annotations, scale_ranges=None, iou_thr=0.5, ioa_thr=None, dataset=None, logger=None, tpfp_fn=None, nproc=4, use_legacy_coordinate=False, use_group_of=False):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        ioa_thr (float | None): IoA threshold to be considered as matched,
            which only used in OpenImages evaluation. Default: None.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        tpfp_fn (callable | None): The function used to determine true/
            false positives. If None, :func:`tpfp_default` is used as default
            unless dataset is 'det' or 'vid' (:func:`tpfp_imagenet` in this
            case). If it is given as a function, then this function is used
            to evaluate tp & fp. Default None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.
        use_group_of (bool): Whether to use group of when calculate TP and FP,
            which only used in OpenImages evaluation. Default: False.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)
    if not use_legacy_coordinate:
        extra_length = 0.0
    else:
        extra_length = 1.0
    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])
    area_ranges = [(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges] if scale_ranges is not None else None
    if num_imgs > 1:
        assert nproc > 0, 'nproc must be at least one.'
        nproc = min(nproc, num_imgs)
        pool = Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(det_results, annotations, i)
        if tpfp_fn is None:
            if dataset in ['det', 'vid']:
                tpfp_fn = tpfp_imagenet
            elif dataset in ['oid_challenge', 'oid_v6'] or use_group_of is True:
                tpfp_fn = tpfp_openimages
            else:
                tpfp_fn = tpfp_default
        if not callable(tpfp_fn):
            raise ValueError(f'tpfp_fn has to be a function or None, but got {tpfp_fn}')
        if num_imgs > 1:
            args = []
            if use_group_of:
                gt_group_ofs = get_cls_group_ofs(annotations, i)
                args.append(gt_group_ofs)
                args.append([use_group_of for _ in range(num_imgs)])
            if ioa_thr is not None:
                args.append([ioa_thr for _ in range(num_imgs)])
            tpfp = pool.starmap(tpfp_fn, zip(cls_dets, cls_gts, cls_gts_ignore, [iou_thr for _ in range(num_imgs)], [area_ranges for _ in range(num_imgs)], [use_legacy_coordinate for _ in range(num_imgs)], *args))
        else:
            tpfp = tpfp_fn(cls_dets[0], cls_gts[0], cls_gts_ignore[0], iou_thr, area_ranges, use_legacy_coordinate, gt_bboxes_group_of=get_cls_group_ofs(annotations, i)[0] if use_group_of else None, use_group_of=use_group_of, ioa_thr=ioa_thr)
            tpfp = [tpfp]
        if use_group_of:
            tp, fp, cls_dets = tuple(zip(*tpfp))
        else:
            tp, fp = tuple(zip(*tpfp))
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0] + extra_length) * (bbox[:, 3] - bbox[:, 1] + extra_length)
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area) & (gt_areas < max_area))
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum(tp + fp, eps)
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if dataset != 'voc07' else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({'num_gts': num_gts, 'num_dets': num_dets, 'recall': recalls, 'precision': precisions, 'ap': ap})
    if num_imgs > 1:
        pool.close()
    if scale_ranges is not None:
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack([cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0
    print_map_summary(mean_ap, eval_results, dataset, area_ranges, logger=logger)
    return (mean_ap, eval_results)

def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['bboxes'][gt_inds, :])
        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
        else:
            cls_gts_ignore.append(np.empty((0, 4), dtype=np.float32))
    return (cls_dets, cls_gts, cls_gts_ignore)

def get_cls_group_ofs(annotations, class_id):
    """Get `gt_group_of` of a certain class, which is used in Open Images.

    Args:
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        list[np.ndarray]: `gt_group_of` of a certain class.
    """
    gt_group_ofs = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        if ann.get('gt_is_group_ofs', None) is not None:
            gt_group_ofs.append(ann['gt_is_group_ofs'][gt_inds])
        else:
            gt_group_ofs.append(np.empty((0, 1), dtype=np.bool))
    return gt_group_ofs

def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum((mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 0.001, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    else:
        raise ValueError('Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and 'enable' in cfg.auto_scale_lr and ('base_batch_size' in cfg.auto_scale_lr):
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn('Can not find "auto_scale_lr" or "auto_scale_lr.enable" or "auto_scale_lr.base_batch_size" in your configuration file. Please update all the configuration files to mmdet >= 2.24.1.')
    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support single GPU mode in non-distributed training. Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. Because we only support single GPU mode in non-distributed training. Use the first GPU in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    cfg.device = get_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        assert 'val' in [mode for mode, _ in cfg.workflow]
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.get('pipeline', cfg.data.train.dataset.get('pipeline'))
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(mmdet_version=__version__ + get_git_hash()[:7], CLASSES=datasets[0].CLASSES)
    model.CLASSES = datasets[0].CLASSES
    train_detector(model, datasets, cfg, distributed=distributed, validate=not args.no_validate, timestamp=timestamp, meta=meta)

def setup_multi_processes(cfg):
    """Setup multi-processing environment variables."""
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', 'fork')
        current_method = mp.get_start_method(allow_none=True)
        if current_method is not None and current_method != mp_start_method:
            warnings.warn(f'Multi-processing start method `{mp_start_method}` is different from the previous setting `{current_method}`.It will be force set to `{mp_start_method}`. You can change this behavior by changing `mp_start_method` in your config.')
        mp.set_start_method(mp_start_method, force=True)
    opencv_num_threads = cfg.get('opencv_num_threads', 0)
    cv2.setNumThreads(opencv_num_threads)
    workers_per_gpu = cfg.data.get('workers_per_gpu', 1)
    if 'train_dataloader' in cfg.data:
        workers_per_gpu = max(cfg.data.train_dataloader.get('workers_per_gpu', 1), workers_per_gpu)
    if 'OMP_NUM_THREADS' not in os.environ and workers_per_gpu > 1:
        omp_num_threads = 1
        warnings.warn(f'Setting OMP_NUM_THREADS environment variable for each process to be {omp_num_threads} in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    if 'MKL_NUM_THREADS' not in os.environ and workers_per_gpu > 1:
        mkl_num_threads = 1
        warnings.warn(f'Setting MKL_NUM_THREADS environment variable for each process to be {mkl_num_threads} in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)

def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed
    rank, world_size = get_dist_info()
    seed = np.random.randint(2 ** 31)
    if world_size == 1:
        return seed
    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show or args.show_dir, 'Please specify at least one operation (save/eval/format/show the results / save the results) with the argument "--out", "--eval", "--format-only", "--show" or "--show-dir"'
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')
    if args.out is not None and (not args.out.endswith(('.pkl', '.pickle'))):
        raise ValueError('The output file must be a pkl file.')
    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. Because we only support single GPU mode in non-distributed testing. Use the first GPU in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    test_dataloader_default_args = dict(samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    test_loader_cfg = {**test_dataloader_default_args, **cfg.data.get('test_dataloader', {})}
    rank, _ = get_dist_info()
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is None and cfg.get('device', None) == 'npu':
        fp16_cfg = dict(loss_scale='dynamic')
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir, args.show_score_thr)
    else:
        model = build_ddp(model, cfg.device, device_ids=[int(os.environ['LOCAL_RANK'])], broadcast_buffers=False)
        if cfg.device == 'npu' and args.tmpdir is None:
            args.tmpdir = './npu_tmpdir'
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect or cfg.evaluation.get('gpu_collect', False))
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule', 'dynamic_intervals']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)

def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show or args.show_dir, 'Please specify at least one operation (save/eval/format/show the results / save the results) with the argument "--out", "--eval", "--format-only", "--show" or "--show-dir"'
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')
    if args.out is not None and (not args.out.endswith(('.pkl', '.pickle'))):
        raise ValueError('The output file must be a pkl file.')
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max([ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False)
    if args.backend == 'onnxruntime':
        from mmdet.core.export.model_wrappers import ONNXRuntimeDetector
        model = ONNXRuntimeDetector(args.model, class_names=dataset.CLASSES, device_id=0)
    elif args.backend == 'tensorrt':
        from mmdet.core.export.model_wrappers import TensorRTDetector
        model = TensorRTDetector(args.model, class_names=dataset.CLASSES, device_id=0)
    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir, args.show_score_thr)
    if args.out:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(outputs, args.out)
    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(outputs, **kwargs)
    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        print(dataset.evaluate(outputs, **eval_kwargs))

def print_coco_results(results):

    def _print(result, ap=1, iouThr=None, areaRng='all', maxDets=100):
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '0.50:0.95' if iouThr is None else f'{iouThr:0.2f}'
        iStr = f' {titleStr:<18} {typeStr} @[ IoU={iouStr:<9} | '
        iStr += f'area={areaRng:>6s} | maxDets={maxDets:>3d} ] = {result:0.3f}'
        print(iStr)
    stats = np.zeros((12,))
    stats[0] = _print(results[0], 1)
    stats[1] = _print(results[1], 1, iouThr=0.5)
    stats[2] = _print(results[2], 1, iouThr=0.75)
    stats[3] = _print(results[3], 1, areaRng='small')
    stats[4] = _print(results[4], 1, areaRng='medium')
    stats[5] = _print(results[5], 1, areaRng='large')
    stats[6] = _print(results[6], 0, maxDets=1)
    stats[7] = _print(results[7], 0, maxDets=10)
    stats[8] = _print(results[8], 0)
    stats[9] = _print(results[9], 0, areaRng='small')
    stats[10] = _print(results[10], 0, areaRng='medium')
    stats[11] = _print(results[11], 0, areaRng='large')

def _print(result, ap=1, iouThr=None, areaRng='all', maxDets=100):
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap == 1 else '(AR)'
    iouStr = '0.50:0.95' if iouThr is None else f'{iouThr:0.2f}'
    iStr = f' {titleStr:<18} {typeStr} @[ IoU={iouStr:<9} | '
    iStr += f'area={areaRng:>6s} | maxDets={maxDets:>3d} ] = {result:0.3f}'
    print(iStr)

def get_coco_style_results(filename, task='bbox', metric=None, prints='mPC', aggregate='benchmark'):
    assert aggregate in ['benchmark', 'all']
    if prints == 'all':
        prints = ['P', 'mPC', 'rPC']
    elif isinstance(prints, str):
        prints = [prints]
    for p in prints:
        assert p in ['P', 'mPC', 'rPC']
    if metric is None:
        metrics = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
    elif isinstance(metric, list):
        metrics = metric
    else:
        metrics = [metric]
    for metric_name in metrics:
        assert metric_name in ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
    eval_output = mmcv.load(filename)
    num_distortions = len(list(eval_output.keys()))
    results = np.zeros((num_distortions, 6, len(metrics)), dtype='float32')
    for corr_i, distortion in enumerate(eval_output):
        for severity in eval_output[distortion]:
            for metric_j, metric_name in enumerate(metrics):
                mAP = eval_output[distortion][severity][task][metric_name]
                results[corr_i, severity, metric_j] = mAP
    P = results[0, 0, :]
    if aggregate == 'benchmark':
        mPC = np.mean(results[:15, 1:, :], axis=(0, 1))
    else:
        mPC = np.mean(results[:, 1:, :], axis=(0, 1))
    rPC = mPC / P
    print(f'\nmodel: {osp.basename(filename)}')
    if metric is None:
        if 'P' in prints:
            print(f'Performance on Clean Data [P] ({task})')
            print_coco_results(P)
        if 'mPC' in prints:
            print(f'Mean Performance under Corruption [mPC] ({task})')
            print_coco_results(mPC)
        if 'rPC' in prints:
            print(f'Relative Performance under Corruption [rPC] ({task})')
            print_coco_results(rPC)
    else:
        if 'P' in prints:
            print(f'Performance on Clean Data [P] ({task})')
            for metric_i, metric_name in enumerate(metrics):
                print(f'{metric_name:5} =  {P[metric_i]:0.3f}')
        if 'mPC' in prints:
            print(f'Mean Performance under Corruption [mPC] ({task})')
            for metric_i, metric_name in enumerate(metrics):
                print(f'{metric_name:5} =  {mPC[metric_i]:0.3f}')
        if 'rPC' in prints:
            print(f'Relative Performance under Corruption [rPC] ({task})')
            for metric_i, metric_name in enumerate(metrics):
                print(f'{metric_name:5} => {rPC[metric_i] * 100:0.1f} %')
    return results

def get_results(filename, dataset='coco', task='bbox', metric=None, prints='mPC', aggregate='benchmark'):
    assert dataset in ['coco', 'voc', 'cityscapes']
    if dataset in ['coco', 'cityscapes']:
        results = get_coco_style_results(filename, task=task, metric=metric, prints=prints, aggregate=aggregate)
    elif dataset == 'voc':
        if task != 'bbox':
            print('Only bbox analysis is supported for Pascal VOC')
            print('Will report bbox results\n')
        if metric not in [None, ['AP'], ['AP50']]:
            print('Only the AP50 metric is supported for Pascal VOC')
            print('Will report AP50 metric\n')
        results = get_voc_style_results(filename, prints=prints, aggregate=aggregate)
    return results

def get_voc_style_results(filename, prints='mPC', aggregate='benchmark'):
    assert aggregate in ['benchmark', 'all']
    if prints == 'all':
        prints = ['P', 'mPC', 'rPC']
    elif isinstance(prints, str):
        prints = [prints]
    for p in prints:
        assert p in ['P', 'mPC', 'rPC']
    eval_output = mmcv.load(filename)
    num_distortions = len(list(eval_output.keys()))
    results = np.zeros((num_distortions, 6, 20), dtype='float32')
    for i, distortion in enumerate(eval_output):
        for severity in eval_output[distortion]:
            mAP = [eval_output[distortion][severity][j]['ap'] for j in range(len(eval_output[distortion][severity]))]
            results[i, severity, :] = mAP
    P = results[0, 0, :]
    if aggregate == 'benchmark':
        mPC = np.mean(results[:15, 1:, :], axis=(0, 1))
    else:
        mPC = np.mean(results[:, 1:, :], axis=(0, 1))
    rPC = mPC / P
    print(f'\nmodel: {osp.basename(filename)}')
    if 'P' in prints:
        print(f'Performance on Clean Data [P] in AP50 = {np.mean(P):0.3f}')
    if 'mPC' in prints:
        print(f'Mean Performance under Corruption [mPC] in AP50 = {np.mean(mPC):0.3f}')
    if 'rPC' in prints:
        print(f'Relative Performance under Corruption [rPC] in % = {np.mean(rPC) * 100:0.1f}')
    return np.mean(results, axis=2, keepdims=True)

def main():
    parser = ArgumentParser(description='Corruption Result Analysis')
    parser.add_argument('filename', help='result file path')
    parser.add_argument('--dataset', type=str, choices=['coco', 'voc', 'cityscapes'], default='coco', help='dataset type')
    parser.add_argument('--task', type=str, nargs='+', choices=['bbox', 'segm'], default=['bbox'], help='task to report')
    parser.add_argument('--metric', nargs='+', choices=[None, 'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl'], default=None, help='metric to report')
    parser.add_argument('--prints', type=str, nargs='+', choices=['P', 'mPC', 'rPC'], default='mPC', help='corruption benchmark metric to print')
    parser.add_argument('--aggregate', type=str, choices=['all', 'benchmark'], default='benchmark', help='aggregate all results or only those         for benchmark corruptions')
    args = parser.parse_args()
    for task in args.task:
        get_results(args.filename, dataset=args.dataset, task=task, metric=args.metric, prints=args.prints, aggregate=args.aggregate)

def voc_eval_with_return(result_file, dataset, iou_thr=0.5, logger='print', only_ap=True):
    det_results = mmcv.load(result_file)
    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = dataset.CLASSES
    mean_ap, eval_results = eval_map(det_results, annotations, scale_ranges=None, iou_thr=iou_thr, dataset=dataset_name, logger=logger)
    if only_ap:
        eval_results = [{'ap': eval_results[i]['ap']} for i in range(len(eval_results))]
    return (mean_ap, eval_results)

def main():
    args = parse_args()
    assert args.out or args.show or args.show_dir, 'Please specify at least one operation (save or show the results) with the argument "--out", "--show" or "show-dir"'
    if args.out is not None and (not args.out.endswith(('.pkl', '.pickle'))):
        raise ValueError('The output file must be a pkl file.')
    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.workers == 0:
        args.workers = cfg.data.workers_per_gpu
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    if args.seed is not None:
        set_random_seed(args.seed)
    if 'all' in args.corruptions:
        corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
    elif 'benchmark' in args.corruptions:
        corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    elif 'noise' in args.corruptions:
        corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise']
    elif 'blur' in args.corruptions:
        corruptions = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur']
    elif 'weather' in args.corruptions:
        corruptions = ['snow', 'frost', 'fog', 'brightness']
    elif 'digital' in args.corruptions:
        corruptions = ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    elif 'holdout' in args.corruptions:
        corruptions = ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
    elif 'None' in args.corruptions:
        corruptions = ['None']
        args.severities = [0]
    else:
        corruptions = args.corruptions
    rank, _ = get_dist_info()
    aggregated_results = {}
    for corr_i, corruption in enumerate(corruptions):
        aggregated_results[corruption] = {}
        for sev_i, corruption_severity in enumerate(args.severities):
            if corr_i > 0 and corruption_severity == 0:
                aggregated_results[corruption][0] = aggregated_results[corruptions[0]][0]
                continue
            test_data_cfg = copy.deepcopy(cfg.data.test)
            if corruption_severity > 0:
                corruption_trans = dict(type='Corrupt', corruption=corruption, severity=corruption_severity)
                test_data_cfg['pipeline'].insert(1, corruption_trans)
            print(f'\nTesting {corruption} at severity {corruption_severity}')
            dataset = build_dataset(test_data_cfg)
            data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=args.workers, dist=distributed, shuffle=False)
            cfg.model.train_cfg = None
            model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
            fp16_cfg = cfg.get('fp16', None)
            if fp16_cfg is not None:
                wrap_fp16_model(model)
            checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
            if 'CLASSES' in checkpoint.get('meta', {}):
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                model.CLASSES = dataset.CLASSES
            if not distributed:
                model = MMDataParallel(model, device_ids=[0])
                show_dir = args.show_dir
                if show_dir is not None:
                    show_dir = osp.join(show_dir, corruption)
                    show_dir = osp.join(show_dir, str(corruption_severity))
                    if not osp.exists(show_dir):
                        osp.makedirs(show_dir)
                outputs = single_gpu_test(model, data_loader, args.show, show_dir, args.show_score_thr)
            else:
                model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
                outputs = multi_gpu_test(model, data_loader, args.tmpdir)
            if args.out and rank == 0:
                eval_results_filename = osp.splitext(args.out)[0] + '_results' + osp.splitext(args.out)[1]
                mmcv.dump(outputs, args.out)
                eval_types = args.eval
                if cfg.dataset_type == 'VOCDataset':
                    if eval_types:
                        for eval_type in eval_types:
                            if eval_type == 'bbox':
                                test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
                                logger = 'print' if args.summaries else None
                                mean_ap, eval_results = voc_eval_with_return(args.out, test_dataset, args.iou_thr, logger)
                                aggregated_results[corruption][corruption_severity] = eval_results
                            else:
                                print('\nOnly "bbox" evaluation                                 is supported for pascal voc')
                elif eval_types:
                    print(f'Starting evaluate {' and '.join(eval_types)}')
                    if eval_types == ['proposal_fast']:
                        result_file = args.out
                    elif not isinstance(outputs[0], dict):
                        result_files = dataset.results2json(outputs, args.out)
                    else:
                        for name in outputs[0]:
                            print(f'\nEvaluating {name}')
                            outputs_ = [out[name] for out in outputs]
                            result_file = args.out
                            +f'.{name}'
                            result_files = dataset.results2json(outputs_, result_file)
                    eval_results = coco_eval_with_return(result_files, eval_types, dataset.coco)
                    aggregated_results[corruption][corruption_severity] = eval_results
                else:
                    print('\nNo task was selected for evaluation;\nUse --eval to select a task')
                mmcv.dump(aggregated_results, eval_results_filename)
    if rank == 0:
        print('\nAggregated results:')
        prints = args.final_prints
        aggregate = args.final_prints_aggregate
        if cfg.dataset_type == 'VOCDataset':
            get_results(eval_results_filename, dataset='voc', prints=prints, aggregate=aggregate)
        else:
            get_results(eval_results_filename, dataset='coco', prints=prints, aggregate=aggregate)

def coco_eval_with_return(result_files, result_types, coco, max_dets=(100, 300, 1000)):
    for res_type in result_types:
        assert res_type in ['proposal', 'bbox', 'segm', 'keypoints']
    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)
    eval_results = {}
    for res_type in result_types:
        result_file = result_files[res_type]
        assert result_file.endswith('.json')
        coco_dets = coco.loadRes(result_file)
        img_ids = coco.getImgIds()
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        cocoEval.params.imgIds = img_ids
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        if res_type == 'segm' or res_type == 'bbox':
            metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
            eval_results[res_type] = {metric_names[i]: cocoEval.stats[i] for i in range(len(metric_names))}
        else:
            eval_results[res_type] = cocoEval.stats
    return eval_results

def test_eval_map():
    det_results = [[det_bboxes, det_bboxes], [det_bboxes, det_bboxes]]
    labels = np.array([0, 1, 1])
    labels_ignore = np.array([0, 1])
    gt_info = {'bboxes': gt_bboxes, 'bboxes_ignore': gt_ignore, 'labels': labels, 'labels_ignore': labels_ignore}
    annotations = [gt_info, gt_info]
    mean_ap, eval_results = eval_map(det_results, annotations, use_legacy_coordinate=True)
    assert 0.291 < mean_ap < 0.293
    mean_ap, eval_results = eval_map(det_results, annotations, use_legacy_coordinate=False)
    assert 0.291 < mean_ap < 0.293
    det_results = [[det_bboxes, det_bboxes]]
    labels = np.array([0, 1, 1])
    labels_ignore = np.array([0, 1])
    gt_info = {'bboxes': gt_bboxes, 'bboxes_ignore': gt_ignore, 'labels': labels, 'labels_ignore': labels_ignore}
    annotations = [gt_info]
    mean_ap, eval_results = eval_map(det_results, annotations, use_legacy_coordinate=True)
    assert 0.291 < mean_ap < 0.293
    mean_ap, eval_results = eval_map(det_results, annotations, use_legacy_coordinate=False)
    assert 0.291 < mean_ap < 0.293

def test_setup_multi_processes():
    sys_start_mehod = mp.get_start_method(allow_none=True)
    sys_cv_threads = cv2.getNumThreads()
    sys_omp_threads = os.environ.pop('OMP_NUM_THREADS', default=None)
    sys_mkl_threads = os.environ.pop('MKL_NUM_THREADS', default=None)
    config = dict(data=dict(workers_per_gpu=2))
    cfg = Config(config)
    setup_multi_processes(cfg)
    assert os.getenv('OMP_NUM_THREADS') == '1'
    assert os.getenv('MKL_NUM_THREADS') == '1'
    assert cv2.getNumThreads() == 1
    if platform.system() != 'Windows':
        assert mp.get_start_method() == 'fork'
    os.environ.pop('OMP_NUM_THREADS')
    os.environ.pop('MKL_NUM_THREADS')
    config = dict(data=dict(workers_per_gpu=0))
    cfg = Config(config)
    setup_multi_processes(cfg)
    assert 'OMP_NUM_THREADS' not in os.environ
    assert 'MKL_NUM_THREADS' not in os.environ
    os.environ['OMP_NUM_THREADS'] = '4'
    config = dict(data=dict(workers_per_gpu=2))
    cfg = Config(config)
    setup_multi_processes(cfg)
    assert os.getenv('OMP_NUM_THREADS') == '4'
    config = dict(data=dict(workers_per_gpu=2), opencv_num_threads=4, mp_start_method='spawn')
    cfg = Config(config)
    setup_multi_processes(cfg)
    assert cv2.getNumThreads() == 4
    assert mp.get_start_method() == 'spawn'
    if sys_start_mehod:
        mp.set_start_method(sys_start_mehod, force=True)
    cv2.setNumThreads(sys_cv_threads)
    if sys_omp_threads:
        os.environ['OMP_NUM_THREADS'] = sys_omp_threads
    else:
        os.environ.pop('OMP_NUM_THREADS')
    if sys_mkl_threads:
        os.environ['MKL_NUM_THREADS'] = sys_mkl_threads
    else:
        os.environ.pop('MKL_NUM_THREADS')

def test_find_latest_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir
        latest = find_latest_checkpoint(path)
        assert latest is None
        path = osp.join(tmpdir, 'none')
        latest = find_latest_checkpoint(path)
        assert latest is None
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(osp.join(tmpdir, 'latest.pth'), 'w') as f:
            f.write('latest')
        path = tmpdir
        latest = find_latest_checkpoint(path)
        assert latest == osp.join(tmpdir, 'latest.pth')
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(osp.join(tmpdir, 'iter_4000.pth'), 'w') as f:
            f.write('iter_4000')
        with open(osp.join(tmpdir, 'iter_8000.pth'), 'w') as f:
            f.write('iter_8000')
        path = tmpdir
        latest = find_latest_checkpoint(path)
        assert latest == osp.join(tmpdir, 'iter_8000.pth')
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(osp.join(tmpdir, 'epoch_1.pth'), 'w') as f:
            f.write('epoch_1')
        with open(osp.join(tmpdir, 'epoch_2.pth'), 'w') as f:
            f.write('epoch_2')
        path = tmpdir
        latest = find_latest_checkpoint(path)
        assert latest == osp.join(tmpdir, 'epoch_2.pth')

