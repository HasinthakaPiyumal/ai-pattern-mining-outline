# Cluster 33

def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)

def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()

def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)

def download_url(url: str, root: TypePath, filename: TypePath | None=None, md5: str | None=None) -> None:
    """Download a file from a url and place it in root.

    Args:
        url: URL to download file from
        root: Directory to place downloaded file in
        filename: Name to save the file under.
            If ``None``, use the basename of the URL
        md5: MD5 checksum of the download. If None, do not check
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)
    os.makedirs(root, exist_ok=True)
    if not check_integrity(fpath, md5):
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
        except (urllib.error.URLError, OSError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                message = 'Failed download. Trying https -> http instead. Downloading ' + url + ' to ' + fpath
                print(message)
                urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
            else:
                raise e
        if not check_integrity(fpath, md5):
            raise RuntimeError('File not found or corrupted.')

def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)
    return bar_update

class MedMNIST(SubjectsDataset):
    """3D MedMNIST v2 datasets.

    Datasets from `MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and
    3D Biomedical Image Classification <https://arxiv.org/abs/2110.14795>`_.

    Please check the `MedMNIST website <https://medmnist.com/>`_ for more
    information, inclusing the license.

    Args:
        split: Dataset split. Should be ``'train'``, ``'val'`` or ``'test'``.
    """
    BASE_URL = 'https://zenodo.org/record/5208230/files'
    SPLITS = ('train', 'training', 'val', 'validation', 'test', 'testing')

    def __init__(self, split, **kwargs):
        if split not in self.SPLITS:
            raise ValueError(f'The split must be one of {self.SPLITS}')
        split = 'train' if split == 'training' else split
        split = 'val' if split == 'validation' else split
        split = 'test' if split == 'testing' else split
        url = f'{self.BASE_URL}/{self.filename}?download=1'
        download_root = get_torchio_cache_dir() / 'MedMNIST'
        download_url(url, download_root, filename=self.filename)
        path = download_root / self.filename
        npz_file = np.load(path)
        images = npz_file[f'{split}_images']
        labels = npz_file[f'{split}_labels']
        subjects = []
        for image, label in zip(images, labels):
            image = ScalarImage(tensor=image[np.newaxis])
            subject = Subject(image=image, labels=torch.from_numpy(label))
            subjects.append(subject)
        super().__init__(subjects, **kwargs)

    @property
    def filename(self):
        return f'{self.__class__.__name__.lower()}.npz'

def get_torchio_cache_dir() -> Path:
    return Path('~/.cache/torchio').expanduser()

class Slicer(Subject):
    """Sample data provided by `3D Slicer <https://www.slicer.org/>`_.

    See `the Slicer wiki <https://www.slicer.org/wiki/SampleData>`_
    for more information.

    For information about licensing and permissions, check the `Sample Data
    module <https://github.com/Slicer/Slicer/blob/31c89f230919a953e56f6722718281ce6da49e06/Modules/Scripted/SampleData/SampleData.py#L75-L81>`_.

    Args:
        name: One of the keys in :attr:`torchio.datasets.slicer.URLS_DICT`.
    """

    def __init__(self, name='MRHead'):
        try:
            filenames, url_files = URLS_DICT[name]
        except KeyError as e:
            message = f'Invalid name "{name}". Valid names are: {', '.join(URLS_DICT)}'
            raise ValueError(message) from e
        for filename, url_file in zip(filenames, url_files):
            filename = filename.replace('-', '_')
            url = urllib.parse.urljoin(SLICER_URL, url_file)
            download_root = get_torchio_cache_dir() / 'slicer'
            stem = filename.split('.')[0]
            download_url(url, download_root, filename=filename)
        super().__init__({stem: ScalarImage(download_root / filename)})

class FPG(Subject):
    """3T :math:`T_1`-weighted brain MRI and corresponding parcellation.

    Args:
        load_all: If ``True``, three more images will be loaded: a
            :math:`T_2`-weighted MRI, a diffusion MRI and a functional MRI.
    """

    def __init__(self, load_all: bool=False):
        repo_dir = urllib.parse.urljoin(DATA_REPO, 'fernando/')
        self.filenames = {'t1': 't1.nii.gz', 'seg': 't1_seg_gif.nii.gz', 'rigid': 't1_to_mni.tfm', 'affine': 't1_to_mni_affine.h5'}
        if load_all:
            self.filenames['t2'] = 't2.nii.gz'
            self.filenames['fmri'] = 'fmri.nrrd'
            self.filenames['dmri'] = 'dmri.nrrd'
        download_root = get_torchio_cache_dir() / 'fpg'
        for filename in self.filenames.values():
            download_url(urllib.parse.urljoin(repo_dir, filename), download_root, filename=filename)
        rigid = read_matrix(download_root / self.filenames['rigid'])
        affine = read_matrix(download_root / self.filenames['affine'])
        subject_dict = {'t1': ScalarImage(download_root / self.filenames['t1'], rigid_matrix=rigid, affine_matrix=affine), 'seg': LabelMap(download_root / self.filenames['seg'], rigid_matrix=rigid, affine_matrix=affine)}
        if load_all:
            subject_dict['t2'] = ScalarImage(download_root / self.filenames['t2'])
            subject_dict['fmri'] = ScalarImage(download_root / self.filenames['fmri'])
            subject_dict['dmri'] = ScalarImage(download_root / self.filenames['dmri'])
        super().__init__(subject_dict)
        self.gif_colors = self.GIF_COLORS

    def plot(self, *args, **kwargs):
        super().plot(*args, **kwargs, cmap_dict={'seg': self.gif_colors})
    GIF_COLORS = {0: (0, 0, 0), 1: (0, 0, 0), 5: (127, 255, 212), 12: (240, 230, 140), 16: (176, 48, 96), 24: (48, 176, 96), 31: (48, 176, 96), 32: (103, 255, 255), 33: (103, 255, 255), 35: (238, 186, 243), 36: (119, 159, 176), 37: (122, 186, 220), 38: (122, 186, 220), 39: (96, 204, 96), 40: (96, 204, 96), 41: (220, 247, 164), 42: (220, 247, 164), 43: (205, 62, 78), 44: (205, 62, 78), 45: (225, 225, 225), 46: (225, 225, 225), 47: (60, 60, 60), 48: (220, 216, 20), 49: (220, 216, 20), 50: (196, 58, 250), 51: (196, 58, 250), 52: (120, 18, 134), 53: (120, 18, 134), 54: (255, 165, 0), 55: (255, 165, 0), 56: (12, 48, 255), 57: (12, 48, 225), 58: (236, 13, 176), 59: (236, 13, 176), 60: (0, 118, 14), 61: (0, 118, 14), 62: (165, 42, 42), 63: (165, 42, 42), 64: (160, 32, 240), 65: (160, 32, 240), 66: (56, 192, 255), 67: (56, 192, 255), 70: (255, 225, 225), 72: (184, 237, 194), 73: (180, 231, 250), 74: (225, 183, 231), 76: (180, 180, 180), 77: (180, 180, 180), 81: (245, 255, 200), 82: (255, 230, 255), 83: (245, 245, 245), 84: (220, 255, 220), 85: (220, 220, 220), 86: (200, 255, 255), 87: (250, 220, 200), 89: (245, 255, 200), 90: (255, 230, 255), 91: (245, 245, 245), 92: (220, 255, 220), 93: (220, 220, 220), 94: (200, 255, 255), 96: (140, 125, 255), 97: (140, 125, 255), 101: (255, 62, 150), 102: (255, 62, 150), 103: (160, 82, 45), 104: (160, 82, 45), 105: (165, 42, 42), 106: (165, 42, 42), 107: (205, 91, 69), 108: (205, 91, 69), 109: (100, 149, 237), 110: (100, 149, 237), 113: (135, 206, 235), 114: (135, 206, 235), 115: (250, 128, 114), 116: (250, 128, 114), 117: (255, 255, 0), 118: (255, 255, 0), 119: (221, 160, 221), 120: (221, 160, 221), 121: (0, 238, 0), 122: (0, 238, 0), 123: (205, 92, 92), 124: (205, 92, 92), 125: (176, 48, 96), 126: (176, 48, 96), 129: (152, 251, 152), 130: (152, 251, 152), 133: (50, 205, 50), 134: (50, 205, 50), 135: (0, 100, 0), 136: (0, 100, 0), 137: (173, 216, 230), 138: (173, 216, 230), 139: (153, 50, 204), 140: (153, 50, 204), 141: (160, 32, 240), 142: (160, 32, 240), 143: (0, 206, 208), 144: (0, 206, 208), 145: (51, 50, 135), 146: (51, 50, 135), 147: (135, 50, 74), 148: (135, 50, 74), 149: (218, 112, 214), 150: (218, 112, 214), 151: (240, 230, 140), 152: (240, 230, 140), 153: (255, 255, 0), 154: (255, 255, 0), 155: (255, 110, 180), 156: (255, 110, 180), 157: (0, 255, 255), 158: (0, 255, 255), 161: (100, 50, 100), 162: (100, 50, 100), 163: (178, 34, 34), 164: (178, 34, 34), 165: (255, 0, 255), 166: (255, 0, 255), 167: (39, 64, 139), 168: (39, 64, 139), 169: (255, 99, 71), 170: (255, 99, 71), 171: (255, 69, 0), 172: (255, 69, 0), 173: (210, 180, 140), 174: (210, 180, 140), 175: (0, 255, 127), 176: (0, 255, 127), 177: (74, 155, 60), 178: (74, 155, 60), 179: (255, 215, 0), 180: (255, 215, 0), 181: (238, 0, 0), 182: (238, 0, 0), 183: (46, 139, 87), 184: (46, 139, 87), 185: (238, 201, 0), 186: (238, 201, 0), 187: (102, 205, 170), 188: (102, 205, 170), 191: (255, 218, 185), 192: (255, 218, 185), 193: (238, 130, 238), 194: (238, 130, 238), 195: (255, 165, 0), 196: (255, 165, 0), 197: (255, 192, 203), 198: (255, 192, 203), 199: (244, 222, 179), 200: (244, 222, 179), 201: (208, 32, 144), 202: (208, 32, 144), 203: (34, 139, 34), 204: (34, 139, 34), 205: (125, 255, 212), 206: (127, 255, 212), 207: (0, 0, 128), 208: (0, 0, 128)}

def read_matrix(path: TypePath):
    """Read an affine transform and convert to tensor."""
    path = Path(path)
    suffix = path.suffix
    if suffix in ('.tfm', '.h5'):
        tensor = _read_itk_matrix(path)
    elif suffix in ('.txt', '.trsf'):
        tensor = _read_niftyreg_matrix(path)
    else:
        raise ValueError(f'Unknown suffix for transform file: "{suffix}"')
    return tensor

class SubjectITKSNAP(Subject):
    """ITK-SNAP Image Data Downloads.

    See `the ITK-SNAP website`_ for more information.

    .. _the ITK-SNAP website: http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.Data
    """
    url_base = 'https://www.nitrc.org/frs/download.php/'

    def __init__(self, name, code):
        self.name = name
        self.url_dir = urllib.parse.urljoin(self.url_base, f'{code}/')
        self.filename = f'{self.name}.zip'
        self.url = urllib.parse.urljoin(self.url_dir, self.filename)
        self.download_root = get_torchio_cache_dir() / self.name
        if not self.download_root.is_dir():
            download_and_extract_archive(self.url, download_root=self.download_root, filename=self.filename)
        super().__init__(**self.get_kwargs())

    def get_kwargs(self):
        raise NotImplementedError

class SubjectMNI(Subject):
    """Atlases from the Montreal Neurological Institute (MNI).

    See `the website <https://nist.mni.mcgill.ca/?page_id=714>`_ for more
    information.
    """
    url_base = 'http://packages.bic.mni.mcgill.ca/mni-models/'

    @property
    def download_root(self):
        return get_torchio_cache_dir() / self.name

