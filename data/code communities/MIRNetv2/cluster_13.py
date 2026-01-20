# Cluster 13

class REDSDataset(data.Dataset):
    """REDS dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, seperated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_flow (str, optional): Data root path for flow.
            meta_info_file (str): Path for meta information file.
            val_partition (str): Validation partition types. 'REDS4' or
                'official'.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(REDSDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = (Path(opt['dataroot_gt']), Path(opt['dataroot_lq']))
        self.flow_root = Path(opt['dataroot_flow']) if opt['dataroot_flow'] is not None else None
        assert opt['num_frame'] % 2 == 1, f'num_frame should be odd number, but got {opt['num_frame']}'
        self.num_frame = opt['num_frame']
        self.num_half_frames = opt['num_frame'] // 2
        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])
        if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f"Wrong validation partition {opt['val_partition']}.Supported ones are ['official', 'REDS4'].")
        self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        interval_str = ','.join((str(x) for x in opt['interval_list']))
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')
        center_frame_idx = int(frame_name)
        interval = random.choice(self.interval_list)
        start_frame_idx = center_frame_idx - self.num_half_frames * interval
        end_frame_idx = center_frame_idx + self.num_half_frames * interval
        while start_frame_idx < 0 or end_frame_idx > 99:
            center_frame_idx = random.randint(0, 99)
            start_frame_idx = center_frame_idx - self.num_half_frames * interval
            end_frame_idx = center_frame_idx + self.num_half_frames * interval
        frame_name = f'{center_frame_idx:08d}'
        neighbor_list = list(range(center_frame_idx - self.num_half_frames * interval, center_frame_idx + self.num_half_frames * interval + 1, interval))
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()
        assert len(neighbor_list) == self.num_frame, f'Wrong length of neighbor list: {len(neighbor_list)}'
        if self.is_lmdb:
            img_gt_path = f'{clip_name}/{frame_name}'
        else:
            img_gt_path = self.gt_root / clip_name / f'{frame_name}.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        img_lqs = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)
        if self.flow_root is not None:
            img_flows = []
            for i in range(self.num_half_frames, 0, -1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_p{i}'
                else:
                    flow_path = self.flow_root / clip_name / f'{frame_name}_p{i}.png'
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(img_bytes, flag='grayscale', float32=False)
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(dx, dy, max_val=20, denorm=False)
                img_flows.append(flow)
            for i in range(1, self.num_half_frames + 1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_n{i}'
                else:
                    flow_path = self.flow_root / clip_name / f'{frame_name}_n{i}.png'
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(img_bytes, flag='grayscale', float32=False)
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(dx, dy, max_val=20, denorm=False)
                img_flows.append(flow)
            img_lqs.extend(img_flows)
        img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, gt_size, scale, img_gt_path)
        if self.flow_root is not None:
            img_lqs, img_flows = (img_lqs[:self.num_frame], img_lqs[self.num_frame:])
        img_lqs.append(img_gt)
        if self.flow_root is not None:
            img_results, img_flows = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'], img_flows)
        else:
            img_results = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'])
        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]
        if self.flow_root is not None:
            img_flows = img2tensor(img_flows)
            img_flows.insert(self.num_half_frames, torch.zeros_like(img_flows[0]))
            img_flows = torch.stack(img_flows, dim=0)
        if self.flow_root is not None:
            return {'lq': img_lqs, 'flow': img_flows, 'gt': img_gt, 'key': key}
        else:
            return {'lq': img_lqs, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)

def get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    if logger_name in initialized_logger:
        return logger
    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger

def create_dataset(dataset_opt):
    """Create dataset.

    Args:
        dataset_opt (dict): Configuration for dataset. It constains:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_type = dataset_opt['type']
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')
    dataset = dataset_cls(dataset_opt)
    logger = get_root_logger()
    logger.info(f'Dataset {dataset.__class__.__name__} - {dataset_opt['name']} is created.')
    return dataset

def create_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """Create dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_opt['phase']
    rank, _ = get_dist_info()
    if phase == 'train':
        if dist:
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=sampler, drop_last=True)
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
    elif phase in ['val', 'test']:
        dataloader_args = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        raise ValueError(f"Wrong dataset phase: {phase}. Supported ones are 'train', 'val' and 'test'.")
    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)
    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        return torch.utils.data.DataLoader(**dataloader_args)

class VideoTestVimeo90KDataset(data.Dataset):
    """Video test dataset for Vimeo90k-Test dataset.

    It only keeps the center frame for testing.
    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoTestVimeo90KDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        if self.cache_data:
            raise NotImplementedError('cache_data in Vimeo90K-Test dataset is not implemented.')
        self.gt_root, self.lq_root = (opt['dataroot_gt'], opt['dataroot_lq'])
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        neighbor_list = [i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])]
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'
        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt['name']}')
        with open(opt['meta_info_file'], 'r') as fin:
            subfolders = [line.split(' ')[0] for line in fin]
        for idx, subfolder in enumerate(subfolders):
            gt_path = osp.join(self.gt_root, subfolder, 'im4.png')
            self.data_info['gt_path'].append(gt_path)
            lq_paths = [osp.join(self.lq_root, subfolder, f'im{i}.png') for i in neighbor_list]
            self.data_info['lq_path'].append(lq_paths)
            self.data_info['folder'].append('vimeo90k')
            self.data_info['idx'].append(f'{idx}/{len(subfolders)}')
            self.data_info['border'].append(0)

    def __getitem__(self, index):
        lq_path = self.data_info['lq_path'][index]
        gt_path = self.data_info['gt_path'][index]
        imgs_lq = read_img_seq(lq_path)
        img_gt = read_img_seq([gt_path])
        img_gt.squeeze_(0)
        return {'lq': imgs_lq, 'gt': img_gt, 'folder': self.data_info['folder'][index], 'idx': self.data_info['idx'][index], 'border': self.data_info['border'][index], 'lq_path': lq_path[self.opt['num_frame'] // 2]}

    def __len__(self):
        return len(self.data_info['gt_path'])

class Vimeo90KDataset(data.Dataset):
    """Vimeo90K dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, seperated by a white space.
    Examples:
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    Key examples: "00001/0001"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    The neighboring frame list for different num_frame:
    num_frame | frame list
             1 | 4
             3 | 3,4,5
             5 | 2,3,4,5,6
             7 | 1,2,3,4,5,6,7

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(Vimeo90KDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = (Path(opt['dataroot_gt']), Path(opt['dataroot_lq']))
        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
        self.neighbor_list = [i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])]
        self.random_reverse = opt['random_reverse']
        logger = get_root_logger()
        logger.info(f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')
        if self.is_lmdb:
            img_gt_path = f'{key}/im4'
        else:
            img_gt_path = self.gt_root / clip / seq / 'im4.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        img_lqs = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)
        img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, gt_size, scale, img_gt_path)
        img_lqs.append(img_gt)
        img_results = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'])
        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]
        return {'lq': img_lqs, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)

def create_model(opt):
    """Create model.

    Args:
        opt (dict): Configuration. It constains:
            model_type (str): Model type.
    """
    model_type = opt['model_type']
    for module in _model_modules:
        model_cls = getattr(module, model_type, None)
        if model_cls is not None:
            break
    if model_cls is None:
        raise ValueError(f'Model {model_type} is not found.')
    model = model_cls(opt)
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model

class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            raise ValueError('pixel loss are None.')
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]
        self.output = preds[-1]
        loss_dict = OrderedDict()
        l_pix = 0.0
        for pred in preds:
            l_pix += self.cri_pix(pred, self.gt)
        loss_dict['l_pix'] = l_pix
        l_pix.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = (0, 0)
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.0

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        window_size = self.opt['val'].get('window_size', 0)
        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test
        cnt = 0
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            test()
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}.png')
                    save_gt_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}_gt.png')
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
                    save_gt_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_gt.png')
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)
            if with_metrics:
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)
            cnt += 1
        current_metric = 0.0
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

