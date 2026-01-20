# Cluster 22

def replace_value(cfg):
    if isinstance(cfg, dict):
        return {key: replace_value(value) for key, value in cfg.items()}
    elif isinstance(cfg, list):
        return [replace_value(item) for item in cfg]
    elif isinstance(cfg, tuple):
        return tuple([replace_value(item) for item in cfg])
    elif isinstance(cfg, str):
        keys = pattern_key.findall(cfg)
        values = [get_value(ori_cfg, key[2:-1]) for key in keys]
        if len(keys) == 1 and keys[0] == cfg:
            cfg = values[0]
        else:
            for key, value in zip(keys, values):
                assert not isinstance(value, (dict, list, tuple)), f"for the format of string cfg is 'xxxxx${key}xxxxx' or 'xxx${key}xxx${key}xxx', the type of the value of '${key}' can not be dict, list, or tuplebut you input {type(value)} in {cfg}"
                cfg = cfg.replace(key, str(value))
        return cfg
    else:
        return cfg

def get_value(cfg, key):
    for k in key.split('.'):
        cfg = cfg[k]
    return cfg

def replace_cfg_vals(ori_cfg):
    """Replace the string "${key}" with the corresponding value.

    Replace the "${key}" with the value of ori_cfg.key in the config. And
    support replacing the chained ${key}. Such as, replace "${key0.key1}"
    with the value of cfg.key0.key1. Code is modified from `vars.py
    < https://github.com/microsoft/SoftTeacher/blob/main/ssod/utils/vars.py>`_  # noqa: E501

    Args:
        ori_cfg (mmcv.utils.config.Config):
            The origin config with "${key}" generated from a file.

    Returns:
        updated_cfg [mmcv.utils.config.Config]:
            The config with "${key}" replaced by the corresponding value.
    """

    def get_value(cfg, key):
        for k in key.split('.'):
            cfg = cfg[k]
        return cfg

    def replace_value(cfg):
        if isinstance(cfg, dict):
            return {key: replace_value(value) for key, value in cfg.items()}
        elif isinstance(cfg, list):
            return [replace_value(item) for item in cfg]
        elif isinstance(cfg, tuple):
            return tuple([replace_value(item) for item in cfg])
        elif isinstance(cfg, str):
            keys = pattern_key.findall(cfg)
            values = [get_value(ori_cfg, key[2:-1]) for key in keys]
            if len(keys) == 1 and keys[0] == cfg:
                cfg = values[0]
            else:
                for key, value in zip(keys, values):
                    assert not isinstance(value, (dict, list, tuple)), f"for the format of string cfg is 'xxxxx${key}xxxxx' or 'xxx${key}xxx${key}xxx', the type of the value of '${key}' can not be dict, list, or tuplebut you input {type(value)} in {cfg}"
                    cfg = cfg.replace(key, str(value))
            return cfg
        else:
            return cfg
    pattern_key = re.compile('\\$\\{[a-zA-Z\\d_.]*\\}')
    updated_cfg = Config(replace_value(ori_cfg._cfg_dict), filename=ori_cfg.filename)
    if updated_cfg.get('model_wrapper', None) is not None:
        updated_cfg.model = updated_cfg.model_wrapper
        updated_cfg.pop('model_wrapper')
    return updated_cfg

def update(cfg, src_str, dst_str):
    for k, v in cfg.items():
        if isinstance(v, mmcv.ConfigDict):
            update(cfg[k], src_str, dst_str)
        if isinstance(v, str) and src_str in v:
            cfg[k] = v.replace(src_str, dst_str)

def update_data_root(cfg, logger=None):
    """Update data root according to env MMDET_DATASETS.

    If set env MMDET_DATASETS, update cfg.data_root according to
    MMDET_DATASETS. Otherwise, using cfg.data_root as default.

    Args:
        cfg (mmcv.Config): The model config need to modify
        logger (logging.Logger | str | None): the way to print msg
    """
    assert isinstance(cfg, mmcv.Config), f'cfg got wrong type: {type(cfg)}, expected mmcv.Config'
    if 'MMDET_DATASETS' in os.environ:
        dst_root = os.environ['MMDET_DATASETS']
        print_log(f'MMDET_DATASETS has been set to be {dst_root}.Using {dst_root} as data root.')
    else:
        return
    assert isinstance(cfg, mmcv.Config), f'cfg got wrong type: {type(cfg)}, expected mmcv.Config'

    def update(cfg, src_str, dst_str):
        for k, v in cfg.items():
            if isinstance(v, mmcv.ConfigDict):
                update(cfg[k], src_str, dst_str)
            if isinstance(v, str) and src_str in v:
                cfg[k] = v.replace(src_str, dst_str)
    update(cfg.data, cfg.data_root, dst_root)
    cfg.data_root = dst_root

def retrieve_data_cfg(config_path, skip_type, cfg_options):

    def skip_pipeline_steps(config):
        config['pipeline'] = [x for x in config.pipeline if x['type'] not in skip_type]
    cfg = Config.fromfile(config_path)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg and train_data_cfg['type'] != 'MultiImageMixDataset':
        train_data_cfg = train_data_cfg['dataset']
    if isinstance(train_data_cfg, Sequence):
        [skip_pipeline_steps(c) for c in train_data_cfg]
    else:
        skip_pipeline_steps(train_data_cfg)
    return cfg

def skip_pipeline_steps(config):
    config['pipeline'] = [x for x in config.pipeline if x['type'] not in skip_type]

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    print(f'Config:\n{cfg.pretty_text}')

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    assert args.eval or args.format_only, 'Please specify at least one operation (eval/format the results) with the argument "--eval", "--format-only"'
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.pkl_results)
    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(outputs, **kwargs)
    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        print(dataset.evaluate(outputs, **eval_kwargs))

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    results = mmcv.load(args.prediction_path)
    assert isinstance(results, list)
    if isinstance(results[0], list):
        pass
    elif isinstance(results[0], tuple):
        results = [result[0] for result in results]
    else:
        raise TypeError('invalid type of prediction results')
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
    dataset = build_dataset(cfg.data.test)
    confusion_matrix = calculate_confusion_matrix(dataset, results, args.score_thr, args.nms_iou_thr, args.tp_iou_thr)
    plot_confusion_matrix(confusion_matrix, dataset.CLASSES + ('background',), save_dir=args.save_dir, show=args.show, color_theme=args.color_theme)

def calculate_confusion_matrix(dataset, results, score_thr=0, nms_iou_thr=None, tp_iou_thr=0.5):
    """Calculate the confusion matrix.

    Args:
        dataset (Dataset): Test or val dataset.
        results (list[ndarray]): A list of detection results in each image.
        score_thr (float|optional): Score threshold to filter bboxes.
            Default: 0.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
        tp_iou_thr (float|optional): IoU threshold to be considered as matched.
            Default: 0.5.
    """
    num_classes = len(dataset.CLASSES)
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    assert len(dataset) == len(results)
    prog_bar = mmcv.ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        if isinstance(per_img_res, tuple):
            res_bboxes, _ = per_img_res
        else:
            res_bboxes = per_img_res
        ann = dataset.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        labels = ann['labels']
        analyze_per_img_dets(confusion_matrix, gt_bboxes, labels, res_bboxes, score_thr, tp_iou_thr, nms_iou_thr)
        prog_bar.update()
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, labels, save_dir=None, show=True, title='Normalized Confusion Matrix', color_theme='plasma'):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `plasma`.
    """
    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix = confusion_matrix.astype(np.float32) / per_label_sums * 100
    num_classes = len(labels)
    fig, ax = plt.subplots(figsize=(0.5 * num_classes, 0.5 * num_classes * 0.8), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)
    title_font = {'weight': 'bold', 'size': 12}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 10}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)
    ax.grid(True, which='minor', linestyle='-')
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, '{}%'.format(int(confusion_matrix[i, j]) if not np.isnan(confusion_matrix[i, j]) else -1), ha='center', va='center', color='w', size=7)
    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)
    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), format='png')
    if show:
        plt.show()

def main():
    args = parse_args()
    mmcv.check_file_exist(args.prediction_path)
    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', 0)
    if cfg.data.train.type in ('MultiImageMixDataset', 'ClassBalancedDataset', 'RepeatDataset', 'ConcatDataset'):
        cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.dataset.pipeline)
    else:
        cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.pipeline)
    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.prediction_path)
    result_visualizer = ResultVisualizer(args.show, args.wait_time, args.show_score_thr, args.overlay_gt_pred)
    result_visualizer.evaluate_and_show(dataset, outputs, topk=args.topk, show_dir=args.show_dir)

def measure_inference_speed(cfg, checkpoint, max_iter, log_interval, is_fuse_conv_bn):
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=True, shuffle=False)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    if is_fuse_conv_bn:
        model = fuse_conv_bn(model)
    model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
    model.eval()
    num_warmup = 5
    pure_inf_time = 0
    fps = 0
    for i, data in enumerate(data_loader):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {max_iter}], fps: {fps:.1f} img / s, times per image: {1000 / fps:.1f} ms / img', flush=True)
        if i + 1 == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} img / s, times per image: {1000 / fps:.1f} ms / img', flush=True)
            break
    return fps

def repeat_measure_inference_speed(cfg, checkpoint, max_iter, log_interval, is_fuse_conv_bn, repeat_num=1):
    assert repeat_num >= 1
    fps_list = []
    for _ in range(repeat_num):
        cp_cfg = copy.deepcopy(cfg)
        fps_list.append(measure_inference_speed(cp_cfg, checkpoint, max_iter, log_interval, is_fuse_conv_bn))
    if repeat_num > 1:
        fps_list_ = [round(fps, 1) for fps in fps_list]
        times_pre_image_list_ = [round(1000 / fps, 1) for fps in fps_list]
        mean_fps_ = sum(fps_list_) / len(fps_list_)
        mean_times_pre_image_ = sum(times_pre_image_list_) / len(times_pre_image_list_)
        print(f'Overall fps: {fps_list_}[{mean_fps_:.1f}] img / s, times per image: {times_pre_image_list_}[{mean_times_pre_image_:.1f}] ms / img', flush=True)
        return fps_list
    return fps_list[0]

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.launcher == 'none':
        raise NotImplementedError('Only supports distributed mode')
    else:
        init_dist(args.launcher, **cfg.dist_params)
    repeat_measure_inference_speed(cfg, args.checkpoint, args.max_iter, args.log_interval, args.fuse_conv_bn, args.repeat_num)

def test_replace_cfg_vals():
    temp_file = tempfile.NamedTemporaryFile()
    cfg_path = f'{temp_file.name}.py'
    with open(cfg_path, 'w') as f:
        f.write('configs')
    ori_cfg_dict = dict()
    ori_cfg_dict['cfg_name'] = osp.basename(temp_file.name)
    ori_cfg_dict['work_dir'] = 'work_dirs/${cfg_name}/${percent}/${fold}'
    ori_cfg_dict['percent'] = 5
    ori_cfg_dict['fold'] = 1
    ori_cfg_dict['model_wrapper'] = dict(type='SoftTeacher', detector='${model}')
    ori_cfg_dict['model'] = dict(type='FasterRCNN', backbone=dict(type='ResNet'), neck=dict(type='FPN'), rpn_head=dict(type='RPNHead'), roi_head=dict(type='StandardRoIHead'), train_cfg=dict(rpn=dict(assigner=dict(type='MaxIoUAssigner'), sampler=dict(type='RandomSampler')), rpn_proposal=dict(nms=dict(type='nms', iou_threshold=0.7)), rcnn=dict(assigner=dict(type='MaxIoUAssigner'), sampler=dict(type='RandomSampler'))), test_cfg=dict(rpn=dict(nms=dict(type='nms', iou_threshold=0.7)), rcnn=dict(nms=dict(type='nms', iou_threshold=0.5))))
    ori_cfg_dict['iou_threshold'] = dict(rpn_proposal_nms='${model.train_cfg.rpn_proposal.nms.iou_threshold}', test_rpn_nms='${model.test_cfg.rpn.nms.iou_threshold}', test_rcnn_nms='${model.test_cfg.rcnn.nms.iou_threshold}')
    ori_cfg_dict['str'] = 'Hello, world!'
    ori_cfg_dict['dict'] = {'Hello': 'world!'}
    ori_cfg_dict['list'] = ['Hello, world!']
    ori_cfg_dict['tuple'] = ('Hello, world!',)
    ori_cfg_dict['test_str'] = 'xxx${str}xxx'
    ori_cfg = Config(ori_cfg_dict, filename=cfg_path)
    updated_cfg = replace_cfg_vals(deepcopy(ori_cfg))
    assert updated_cfg.work_dir == f'work_dirs/{osp.basename(temp_file.name)}/5/1'
    assert updated_cfg.model.detector == ori_cfg.model
    assert updated_cfg.iou_threshold.rpn_proposal_nms == ori_cfg.model.train_cfg.rpn_proposal.nms.iou_threshold
    assert updated_cfg.test_str == 'xxxHello, world!xxx'
    ori_cfg_dict['test_dict'] = 'xxx${dict}xxx'
    ori_cfg_dict['test_list'] = 'xxx${list}xxx'
    ori_cfg_dict['test_tuple'] = 'xxx${tuple}xxx'
    with pytest.raises(AssertionError):
        cfg = deepcopy(ori_cfg)
        cfg['test_dict'] = 'xxx${dict}xxx'
        updated_cfg = replace_cfg_vals(cfg)
    with pytest.raises(AssertionError):
        cfg = deepcopy(ori_cfg)
        cfg['test_list'] = 'xxx${list}xxx'
        updated_cfg = replace_cfg_vals(cfg)
    with pytest.raises(AssertionError):
        cfg = deepcopy(ori_cfg)
        cfg['test_tuple'] = 'xxx${tuple}xxx'
        updated_cfg = replace_cfg_vals(cfg)

def batch_filter(x, P, zs, Fs, Qs, Hs, Rs, Bs=None, us=None, update_first=False, saver=None):
    """
    Batch processes a sequences of measurements.
    Parameters
    ----------
    zs : list-like
        list of measurements at each time step. Missing measurements must be
        represented by None.
    Fs : list-like
        list of values to use for the state transition matrix matrix.
    Qs : list-like
        list of values to use for the process error
        covariance.
    Hs : list-like
        list of values to use for the measurement matrix.
    Rs : list-like
        list of values to use for the measurement error
        covariance.
    Bs : list-like, optional
        list of values to use for the control transition matrix;
        a value of None in any position will cause the filter
        to use `self.B` for that time step.
    us : list-like, optional
        list of values to use for the control input vector;
        a value of None in any position will cause the filter to use
        0 for that time step.
    update_first : bool, optional
        controls whether the order of operations is update followed by
        predict, or predict followed by update. Default is predict->update.
        saver : filterpy.common.Saver, optional
            filterpy.common.Saver object. If provided, saver.save() will be
            called after every epoch
    Returns
    -------
    means : np.array((n,dim_x,1))
        array of the state for each time step after the update. Each entry
        is an np.array. In other words `means[k,:]` is the state at step
        `k`.
    covariance : np.array((n,dim_x,dim_x))
        array of the covariances for each time step after the update.
        In other words `covariance[k,:,:]` is the covariance at step `k`.
    means_predictions : np.array((n,dim_x,1))
        array of the state for each time step after the predictions. Each
        entry is an np.array. In other words `means[k,:]` is the state at
        step `k`.
    covariance_predictions : np.array((n,dim_x,dim_x))
        array of the covariances for each time step after the prediction.
        In other words `covariance[k,:,:]` is the covariance at step `k`.
    Examples
    --------
    .. code-block:: Python
        zs = [t + random.randn()*4 for t in range (40)]
        Fs = [kf.F for t in range (40)]
        Hs = [kf.H for t in range (40)]
        (mu, cov, _, _) = kf.batch_filter(zs, Rs=R_list, Fs=Fs, Hs=Hs, Qs=None,
                                          Bs=None, us=None, update_first=False)
        (xs, Ps, Ks, Pps) = kf.rts_smoother(mu, cov, Fs=Fs, Qs=None)
    """
    n = np.size(zs, 0)
    dim_x = x.shape[0]
    if x.ndim == 1:
        means = zeros((n, dim_x))
        means_p = zeros((n, dim_x))
    else:
        means = zeros((n, dim_x, 1))
        means_p = zeros((n, dim_x, 1))
    covariances = zeros((n, dim_x, dim_x))
    covariances_p = zeros((n, dim_x, dim_x))
    if us is None:
        us = [0.0] * n
        Bs = [0.0] * n
    if update_first:
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):
            x, P = update(x, P, z, R=R, H=H)
            means[i, :] = x
            covariances[i, :, :] = P
            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P
            if saver is not None:
                saver.save()
    else:
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):
            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P
            x, P = update(x, P, z, R=R, H=H)
            means[i, :] = x
            covariances[i, :, :] = P
            if saver is not None:
                saver.save()
    return (means, covariances, means_p, covariances_p)

def predict(x, P, F=1, Q=0, u=0, B=1, alpha=1.0):
    """
    Predict next state (prior) using the Kalman filter state propagation
    equations.
    Parameters
    ----------
    x : numpy.array
        State estimate vector
    P : numpy.array
        Covariance matrix
    F : numpy.array()
        State Transition matrix
    Q : numpy.array, Optional
        Process noise matrix
    u : numpy.array, Optional, default 0.
        Control vector. If non-zero, it is multiplied by B
        to create the control input into the system.
    B : numpy.array, optional, default 0.
        Control transition matrix.
    alpha : float, Optional, default=1.0
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon
    Returns
    -------
    x : numpy.array
        Prior state estimate vector
    P : numpy.array
        Prior covariance matrix
    """
    if np.isscalar(F):
        F = np.array(F)
    x = dot(F, x) + dot(B, u)
    P = alpha * alpha * dot(dot(F, P), F.T) + Q
    return (x, P)

def batch_filter(x, P, zs, Fs, Qs, Hs, Rs, Bs=None, us=None, update_first=False, saver=None):
    """
    Batch processes a sequences of measurements.
    Parameters
    ----------
    zs : list-like
        list of measurements at each time step. Missing measurements must be
        represented by None.
    Fs : list-like
        list of values to use for the state transition matrix matrix.
    Qs : list-like
        list of values to use for the process error
        covariance.
    Hs : list-like
        list of values to use for the measurement matrix.
    Rs : list-like
        list of values to use for the measurement error
        covariance.
    Bs : list-like, optional
        list of values to use for the control transition matrix;
        a value of None in any position will cause the filter
        to use `self.B` for that time step.
    us : list-like, optional
        list of values to use for the control input vector;
        a value of None in any position will cause the filter to use
        0 for that time step.
    update_first : bool, optional
        controls whether the order of operations is update followed by
        predict, or predict followed by update. Default is predict->update.
        saver : filterpy.common.Saver, optional
            filterpy.common.Saver object. If provided, saver.save() will be
            called after every epoch
    Returns
    -------
    means : np.array((n,dim_x,1))
        array of the state for each time step after the update. Each entry
        is an np.array. In other words `means[k,:]` is the state at step
        `k`.
    covariance : np.array((n,dim_x,dim_x))
        array of the covariances for each time step after the update.
        In other words `covariance[k,:,:]` is the covariance at step `k`.
    means_predictions : np.array((n,dim_x,1))
        array of the state for each time step after the predictions. Each
        entry is an np.array. In other words `means[k,:]` is the state at
        step `k`.
    covariance_predictions : np.array((n,dim_x,dim_x))
        array of the covariances for each time step after the prediction.
        In other words `covariance[k,:,:]` is the covariance at step `k`.
    Examples
    --------
    .. code-block:: Python
        zs = [t + random.randn()*4 for t in range (40)]
        Fs = [kf.F for t in range (40)]
        Hs = [kf.H for t in range (40)]
        (mu, cov, _, _) = kf.batch_filter(zs, Rs=R_list, Fs=Fs, Hs=Hs, Qs=None,
                                          Bs=None, us=None, update_first=False)
        (xs, Ps, Ks, Pps) = kf.rts_smoother(mu, cov, Fs=Fs, Qs=None)
    """
    n = np.size(zs, 0)
    dim_x = x.shape[0]
    if x.ndim == 1:
        means = zeros((n, dim_x))
        means_p = zeros((n, dim_x))
    else:
        means = zeros((n, dim_x, 1))
        means_p = zeros((n, dim_x, 1))
    covariances = zeros((n, dim_x, dim_x))
    covariances_p = zeros((n, dim_x, dim_x))
    if us is None:
        us = [0.0] * n
        Bs = [0.0] * n
    if update_first:
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):
            x, P = update(x, P, z, R=R, H=H)
            means[i, :] = x
            covariances[i, :, :] = P
            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P
            if saver is not None:
                saver.save()
    else:
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):
            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P
            x, P = update(x, P, z, R=R, H=H)
            means[i, :] = x
            covariances[i, :, :] = P
            if saver is not None:
                saver.save()
    return (means, covariances, means_p, covariances_p)

