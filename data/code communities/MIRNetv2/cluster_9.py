# Cluster 9

def _scandir(dir_path, keywords, recursive):
    for entry in os.scandir(dir_path):
        if not entry.name.startswith('.') and entry.is_file():
            if full_path:
                return_path = entry.path
            else:
                return_path = osp.relpath(entry.path, root)
            if keywords is None:
                yield return_path
            elif return_path.find(keywords) > 0:
                yield return_path
        elif recursive:
            yield from _scandir(entry.path, keywords=keywords, recursive=recursive)
        else:
            continue

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """
    if suffix is not None and (not isinstance(suffix, (str, tuple))):
        raise TypeError('"suffix" must be a string or tuple of strings')
    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            elif recursive:
                yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
            else:
                continue
    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def scandir_SIDD(dir_path, keywords=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        keywords (str | tuple(str), optional): File keywords that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """
    if keywords is not None and (not isinstance(keywords, (str, tuple))):
        raise TypeError('"keywords" must be a string or tuple of strings')
    root = dir_path

    def _scandir(dir_path, keywords, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)
                if keywords is None:
                    yield return_path
                elif return_path.find(keywords) > 0:
                    yield return_path
            elif recursive:
                yield from _scandir(entry.path, keywords=keywords, recursive=recursive)
            else:
                continue
    return _scandir(dir_path, keywords=keywords, recursive=recursive)

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

def paired_DP_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 3, f'The len of folders should be 3 with [inputL_folder, inputR_folder, gt_folder]. But got {len(folders)}'
    assert len(keys) == 3, f'The len of keys should be 2 with [inputL_key, inputR_key, gt_key]. But got {len(keys)}'
    inputL_folder, inputR_folder, gt_folder = folders
    inputL_key, inputR_key, gt_key = keys
    inputL_paths = list(scandir(inputL_folder))
    inputR_paths = list(scandir(inputR_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(inputL_paths) == len(inputR_paths) == len(gt_paths), f'{inputL_key} and {inputR_key} and {gt_key} datasets have different number of images: {len(inputL_paths)}, {len(inputR_paths)}, {len(gt_paths)}.'
    paths = []
    for idx in range(len(gt_paths)):
        gt_path = gt_paths[idx]
        basename, ext = osp.splitext(osp.basename(gt_path))
        inputL_path = inputL_paths[idx]
        basename_input, ext_input = osp.splitext(osp.basename(inputL_path))
        inputL_name = f'{filename_tmpl.format(basename)}{ext_input}'
        inputL_path = osp.join(inputL_folder, inputL_name)
        assert inputL_name in inputL_paths, f'{inputL_name} is not in {inputL_key}_paths.'
        inputR_path = inputR_paths[idx]
        basename_input, ext_input = osp.splitext(osp.basename(inputR_path))
        inputR_name = f'{filename_tmpl.format(basename)}{ext_input}'
        inputR_path = osp.join(inputR_folder, inputR_name)
        assert inputR_name in inputR_paths, f'{inputR_name} is not in {inputR_key}_paths.'
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(dict([(f'{inputL_key}_path', inputL_path), (f'{inputR_key}_path', inputR_path), (f'{gt_key}_path', gt_path)]))
    return paths

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

def paths_from_lmdb(folder):
    """Generate paths from lmdb.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """
    if not folder.endswith('.lmdb'):
        raise ValueError(f'Folder {folder}folder should in lmdb format.')
    with open(osp.join(folder, 'meta_info.txt')) as fin:
        paths = [line.split('.')[0] for line in fin]
    return paths

def paths_from_folder(folder):
    """Generate paths from folder.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """
    paths = list(scandir(folder))
    paths = [osp.join(folder, path) for path in paths]
    return paths

