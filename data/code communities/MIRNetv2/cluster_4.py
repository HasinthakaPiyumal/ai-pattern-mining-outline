# Cluster 4

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)
    opt['rank'], opt['world_size'] = get_dist_info()
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])
    return opt

def parse(opt_path, is_train=True):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)
    opt['is_train'] = is_train
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])
    for key, val in opt['path'].items():
        if val is not None and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)
    opt['path']['root'] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')
    return opt

def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')

def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return (rank, world_size)

def init_loggers(opt):
    log_file = osp.join(opt['path']['log'], f'train_{opt['name']}_{get_time_str()}.log')
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    if opt['logger'].get('wandb') is not None and opt['logger']['wandb'].get('project') is not None and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, 'should turn on tensorboard when using wandb'
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return (logger, tb_logger)

def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg

def create_train_val_dataloader(opt, logger):
    train_loader, val_loader = (None, None)
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(train_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=train_sampler, seed=opt['manual_seed'])
            num_iter_per_epoch = math.ceil(len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)
            logger.info(f'Training statistics:\n\tNumber of train images: {len(train_set)}\n\tDataset enlarge ratio: {dataset_enlarge_ratio}\n\tBatch size per gpu: {dataset_opt['batch_size_per_gpu']}\n\tWorld size (gpu number): {opt['world_size']}\n\tRequire iter number per epoch: {num_iter_per_epoch}\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt['name']}: {len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')
    return (train_loader, train_sampler, val_loader, total_epochs, total_iters)

def main():
    opt = parse_options(is_train=True)
    torch.backends.cudnn.benchmark = True
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []
    resume_state = None
    if len(states) > 0:
        max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'], map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and (opt['rank'] == 0):
            mkdir_and_rename(osp.join('tb_logger', opt['name']))
    logger, tb_logger = init_loggers(opt)
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result
    if resume_state:
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)
        logger.info(f'Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.')
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0
    msg_logger = MessageLogger(opt, current_iter, tb_logger)
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}.Supported ones are: None, 'cuda', 'cpu'.")
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = (time.time(), time.time())
    start_time = time.time()
    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')
    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])
    logger_j = [True] * len(groups)
    scale = opt['scale']
    epoch = start_epoch
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()
        while train_data is not None:
            data_time = time.time() - data_time
            current_iter += 1
            if current_iter > total_iters:
                break
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            j = ((current_iter > groups) != True).nonzero()[0]
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]
            mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]
            if logger_j[bs_j]:
                logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(mini_gt_size, mini_batch_size * torch.cuda.device_count()))
                logger_j[bs_j] = False
            lq = train_data['lq']
            gt = train_data['gt']
            if mini_batch_size < batch_size:
                indices = random.sample(range(0, batch_size), k=mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]
            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random())
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq = lq[:, :, x0:x1, y0:y1]
                gt = gt[:, :, x0 * scale:x1 * scale, y0 * scale:y1 * scale]
            model.feed_train_data({'lq': lq, 'gt': gt})
            model.optimize_parameters(current_iter)
            iter_time = time.time() - iter_time
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)
            if opt.get('val') is not None and current_iter % opt['val']['val_freq'] == 0:
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                use_image = opt['val'].get('use_image', True)
                model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'], rgb2bgr, use_image)
            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        epoch += 1
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()

def main():
    opt = parse_options(is_train=False)
    torch.backends.cudnn.benchmark = True
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f'test_{opt['name']}_{get_time_str()}.log')
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f'Number of test images in {dataset_opt['name']}: {len(test_set)}')
        test_loaders.append(test_loader)
    model = create_model(opt)
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        use_image = opt['val'].get('use_image', True)
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'], rgb2bgr=rgb2bgr, use_image=use_image)

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))
    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return (Loader, Dumper)

@functools.wraps(func)
def wrapper(*args, **kwargs):
    rank, _ = get_dist_info()
    if rank == 0:
        return func(*args, **kwargs)

