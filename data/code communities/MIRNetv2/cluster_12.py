# Cluster 12

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img
    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def _totensor(img, bgr2rgb, float32):
    if img.shape[2] == 3 and bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose(2, 0, 1))
    if float32:
        img = img.float()
    return img

class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder, self.lq_folder = (opt['dataroot_gt'], opt['dataroot_lq'])
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        index = index % len(self.paths)
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception('gt path {} not working'.format(gt_path))
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception('lq path {} not working'.format(lq_path))
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    if img_np is None:
        raise Exception('None .. !!!')
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.0
    return img

def padding(img_lq, img_gt, gt_size):
    h, w, _ = img_lq.shape
    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    if h_pad == 0 and w_pad == 0:
        return (img_lq, img_gt)
    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    if img_lq.ndim == 2:
        img_lq = np.expand_dims(img_lq, axis=2)
    if img_gt.ndim == 2:
        img_gt = np.expand_dims(img_gt, axis=2)
    return (img_lq, img_gt)

def paired_random_crop(img_gts, img_lqs, lq_patch_size, scale, gt_path):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        lq_patch_size (int): LQ patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]
    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    gt_patch_size = int(lq_patch_size * scale)
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ', f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size ({lq_patch_size}, {lq_patch_size}). Please remove {gt_path}.')
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)
    img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]
    top_gt, left_gt = (int(top * scale), int(left * scale))
    img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return (img_gts, img_lqs)

def random_augmentation(*args):
    out = []
    flag_aug = random.randint(0, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out

class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt
        if self.opt['phase'] == 'train':
            self.sigma_type = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder = opt['dataroot_gt']
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.gt_folder, line.split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))
        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        index = index % len(self.paths)
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception('gt path {} not working'.format(gt_path))
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception('gt path {} not working'.format(gt_path))
            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)
            noise_level = torch.FloatTensor([sigma_value]) / 255.0
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)
        else:
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test / 255.0, img_lq.shape)
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': gt_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):

    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder, self.lqL_folder, self.lqR_folder = (opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR'])
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
        self.paths = paired_DP_paths_from_folder([self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'], self.filename_tmpl)
        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        index = index % len(self.paths)
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception('gt path {} not working'.format(gt_path))
        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception('lqL path {} not working'.format(lqL_path))
        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqR_path, 'lqR')
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception('lqR path {} not working'.format(lqR_path))
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt], bgr2rgb=True, float32=True)
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        img_lq = torch.cat([img_lqL, img_lqR], 0)
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lqL_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

def imfrombytesDP(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    if img_np is None:
        raise Exception('None .. !!!')
    img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
    if float32:
        img = img.astype(np.float32) / 65535.0
    return img

def padding_DP(img_lqL, img_lqR, img_gt, gt_size):
    h, w, _ = img_gt.shape
    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    if h_pad == 0 and w_pad == 0:
        return (img_lqL, img_lqR, img_gt)
    img_lqL = cv2.copyMakeBorder(img_lqL, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_lqR = cv2.copyMakeBorder(img_lqR, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    return (img_lqL, img_lqR, img_gt)

def paired_random_crop_DP(img_lqLs, img_lqRs, img_gts, gt_patch_size, scale, gt_path):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqLs, list):
        img_lqLs = [img_lqLs]
    if not isinstance(img_lqRs, list):
        img_lqRs = [img_lqRs]
    h_lq, w_lq, _ = img_lqLs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ', f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size ({lq_patch_size}, {lq_patch_size}). Please remove {gt_path}.')
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)
    img_lqLs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqLs]
    img_lqRs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqRs]
    top_gt, left_gt = (int(top * scale), int(left * scale))
    img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqLs) == 1:
        img_lqLs = img_lqLs[0]
    if len(img_lqRs) == 1:
        img_lqRs = img_lqRs[0]
    return (img_lqLs, img_lqRs, img_gts)

class SingleImageDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SingleImageDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.lq_folder = opt['dataroot_lq']
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paths_from_lmdb(self.lq_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.lq_folder, line.split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
        return {'lq': img_lq, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)

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

def dequantize_flow(dx, dy, max_val=0.02, denorm=True):
    """Recover from quantized flow.

    Args:
        dx (ndarray): Quantized dx.
        dy (ndarray): Quantized dy.
        max_val (float): Maximum value used when quantizing.
        denorm (bool): Whether to multiply flow values with width/height.

    Returns:
        ndarray: Dequantized flow.
    """
    assert dx.shape == dy.shape
    assert dx.ndim == 2 or (dx.ndim == 3 and dx.shape[-1] == 1)
    dx, dy = [dequantize(d, -max_val, max_val, 255) for d in [dx, dy]]
    if denorm:
        dx *= dx.shape[1]
        dy *= dx.shape[0]
    flow = np.dstack((dx, dy))
    return flow

def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:
            cv2.flip(img, 1, img)
        if vflip:
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow
    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]
    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return (imgs, flows)
    elif return_status:
        return (imgs, (hflip, vflip, rot90))
    else:
        return imgs

class FFHQDataset(data.Dataset):
    """FFHQ dataset for StyleGAN.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.

    """

    def __init__(self, opt):
        super(FFHQDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean']
        self.std = opt['std']
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            self.paths = [osp.join(self.gt_folder, f'{v:08d}.png') for v in range(70000)]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)
        img_gt = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False)
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        normalize(img_gt, self.mean, self.std, inplace=True)
        return {'gt': img_gt, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

def _augment(img):
    if hflip:
        cv2.flip(img, 1, img)
    if vflip:
        cv2.flip(img, 0, img)
    if rot90:
        img = img.transpose(1, 0, 2)
    return img

def _augment_flow(flow):
    if hflip:
        cv2.flip(flow, 1, flow)
        flow[:, :, 0] *= -1
    if vflip:
        cv2.flip(flow, 0, flow)
        flow[:, :, 1] *= -1
    if rot90:
        flow = flow.transpose(1, 0, 2)
        flow = flow[:, :, [1, 0]]
    return flow

def data_augmentation(image, mode):
    """
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    """
    if mode == 0:
        out = image
    elif mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(image)
    elif mode == 3:
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        out = np.rot90(image, k=2)
    elif mode == 5:
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        out = np.rot90(image, k=3)
    elif mode == 7:
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return out

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

