# Cluster 1

class IXI(SubjectsDataset):
    """Full IXI dataset.

    Args:
        root: Root directory to which the dataset will be downloaded.
        transform: An instance of
            :class:`~torchio.transforms.transform.Transform`.
        download: If set to ``True``, will download the data into :attr:`root`.
        modalities: List of modalities to be downloaded. They must be in
            ``('T1', 'T2', 'PD', 'MRA', 'DTI')``.

    .. warning:: The size of this dataset is multiple GB.
        If you set :attr:`download` to ``True``, it will take some time
        to be downloaded if it is not already present.

    Example:

        >>> import torchio as tio
        >>> transforms = [
        ...     tio.ToCanonical(),  # to RAS
        ...     tio.Resample((1, 1, 1)),  # to 1 mm iso
        ... ]
        >>> ixi_dataset = tio.datasets.IXI(
        ...     'path/to/ixi_root/',
        ...     modalities=('T1', 'T2'),
        ...     transform=tio.Compose(transforms),
        ...     download=True,
        ... )
        >>> print('Number of subjects in dataset:', len(ixi_dataset))  # 577
        >>> sample_subject = ixi_dataset[0]
        >>> print('Keys in subject:', tuple(sample_subject.keys()))  # ('T1', 'T2')
        >>> print('Shape of T1 data:', sample_subject['T1'].shape)  # [1, 180, 268, 268]
        >>> print('Shape of T2 data:', sample_subject['T2'].shape)  # [1, 241, 257, 188]
    """
    base_url = 'http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-{modality}.tar'
    md5_dict = {'T1': '34901a0593b41dd19c1a1f746eac2d58', 'T2': 'e3140d78730ecdd32ba92da48c0a9aaa', 'PD': '88ecd9d1fa33cb4a2278183b42ffd749', 'MRA': '29be7d2fee3998f978a55a9bdaf3407e', 'DTI': '636573825b1c8b9e8c78f1877df3ee66'}

    def __init__(self, root: TypePath, download: bool=False, modalities: Sequence[str]=('T1', 'T2'), **kwargs):
        root = Path(root)
        for modality in modalities:
            if modality not in self.md5_dict:
                message = f'Modality "{modality}" must be one of {tuple(self.md5_dict.keys())}'
                raise ValueError(message)
        if download:
            self._download(root, modalities)
        if not self._check_exists(root, modalities):
            message = 'Dataset not found. You can use download=True to download it'
            raise RuntimeError(message)
        subjects_list = self._get_subjects_list(root, modalities)
        super().__init__(subjects_list, **kwargs)

    @staticmethod
    def _check_exists(root, modalities):
        for modality in modalities:
            modality_dir = root / modality
            if not modality_dir.is_dir():
                exists = False
                break
        else:
            exists = True
        return exists

    @staticmethod
    def _get_subjects_list(root, modalities):
        one_modality = modalities[0]
        paths = sglob(root / one_modality, '*.nii.gz')
        subjects = []
        for filepath in paths:
            subject_id = get_subject_id(filepath)
            images_dict: dict[str, str | ScalarImage] = {'subject_id': subject_id}
            images_dict[one_modality] = ScalarImage(filepath)
            for modality in modalities[1:]:
                globbed = sglob(root / modality, f'{subject_id}-{modality}.nii.gz')
                if globbed:
                    assert len(globbed) == 1
                    images_dict[modality] = ScalarImage(globbed[0])
                else:
                    skip_subject = True
                    break
            else:
                skip_subject = False
            if skip_subject:
                continue
            subjects.append(Subject(**images_dict))
        return subjects

    def _download(self, root, modalities):
        """Download the IXI data if it does not exist already."""
        for modality in modalities:
            modality_dir = root / modality
            if modality_dir.is_dir():
                continue
            modality_dir.mkdir(exist_ok=True, parents=True)
            url = self.base_url.format(modality=modality)
            md5 = self.md5_dict[modality]
            with NamedTemporaryFile(suffix='.tar', delete=False) as f:
                download_and_extract_archive(url, download_root=modality_dir, filename=f.name, md5=md5)

def download_and_extract_archive(url: str, download_root: TypePath, extract_root: TypePath | None=None, filename: TypePath | None=None, md5: str | None=None, remove_finished: bool=False) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)
    download_url(url, download_root, filename, md5)
    archive = os.path.join(download_root, filename)
    extract_archive(archive, extract_root, remove_finished)

class IXITiny(SubjectsDataset):
    """This is the dataset used in the main `notebook`_. It is a tiny version
    of IXI, containing 566 :math:`T_1`-weighted brain MR images and their
    corresponding brain segmentations, all with size :math:`83 \\times 44 \\times
    55`.

    It can be used as a medical image MNIST.

    Args:
        root: Root directory to which the dataset will be downloaded.
        transform: An instance of
            :class:`~torchio.transforms.transform.Transform`.
        download: If set to ``True``, will download the data into :attr:`root`.

    .. _notebook: https://github.com/TorchIO-project/torchio/blob/main/tutorials/README.md
    """
    url = 'https://www.dropbox.com/s/ogxjwjxdv5mieah/ixi_tiny.zip?dl=1'
    md5 = 'bfb60f4074283d78622760230bfa1f98'

    def __init__(self, root: TypePath, transform: Transform | None=None, download: bool=False, **kwargs):
        root = Path(root)
        if download:
            self._download(root)
        if not root.is_dir():
            message = 'Dataset not found. You can use download=True to download it'
            raise RuntimeError(message)
        subjects_list = self._get_subjects_list(root)
        super().__init__(subjects_list, transform=transform, **kwargs)

    @staticmethod
    def _get_subjects_list(root):
        image_paths = sglob(root / 'image', '*.nii.gz')
        label_paths = sglob(root / 'label', '*.nii.gz')
        if not (image_paths and label_paths):
            message = f'Images not found. Remove the root directory ({root}) and try again'
            raise FileNotFoundError(message)
        subjects = []
        for image_path, label_path in zip(image_paths, label_paths):
            subject_id = get_subject_id(image_path)
            subject_dict = {}
            subject_dict['image'] = ScalarImage(image_path)
            subject_dict['label'] = LabelMap(label_path)
            subject_dict['subject_id'] = subject_id
            subjects.append(Subject(**subject_dict))
        return subjects

    def _download(self, root):
        """Download the tiny IXI data if it doesn't exist already."""
        if root.is_dir():
            return
        with NamedTemporaryFile(suffix='.zip', delete=False) as f:
            download_and_extract_archive(self.url, download_root=root, filename=f.name, md5=self.md5)
        ixi_tiny_dir = root / 'ixi_tiny'
        (ixi_tiny_dir / 'image').rename(root / 'image')
        (ixi_tiny_dir / 'label').rename(root / 'label')
        shutil.rmtree(ixi_tiny_dir)

class EPISURG(SubjectsDataset):
    """
    `EPISURG <https://doi.org/10.5522/04/9996158.v1>`_ is a clinical dataset of
    :math:`T_1`-weighted MRI from 430 epileptic patients who underwent
    resective brain surgery at the National Hospital of Neurology and
    Neurosurgery (Queen Square, London, United Kingdom) between 1990 and 2018.

    The dataset comprises 430 postoperative MRI. The corresponding preoperative
    MRI is present for 268 subjects.

    Three human raters segmented the resection cavity on partially overlapping
    subsets of EPISURG.

    If you use this dataset for your research, you agree with the *Data use
    agreement* presented at the EPISURG entry on the `UCL Research Data
    Repository <https://doi.org/10.5522/04/9996158.v1>`_ and you must cite the
    corresponding publications.

    Args:
        root: Root directory to which the dataset will be downloaded.
        transform: An instance of
            :class:`~torchio.transforms.transform.Transform`.
        download: If set to ``True``, will download the data into :attr:`root`.

    .. warning:: The size of this dataset is multiple GB.
        If you set :attr:`download` to ``True``, it will take some time
        to be downloaded if it is not already present.
    """
    data_url = 'https://s3-eu-west-1.amazonaws.com/pstorage-ucl-2748466690/26153588/EPISURG.zip'
    md5 = '5ec5831a2c6fbfdc8489ba2910a6504b'

    def __init__(self, root: TypePath, transform: Transform | None=None, download: bool=False, **kwargs):
        root = Path(root).expanduser().absolute()
        if download:
            self._download(root)
        subjects_list = self._get_subjects_list(root)
        self.kwargs = kwargs
        super().__init__(subjects_list, transform=transform, **kwargs)

    @staticmethod
    def _check_exists(root, modalities):
        for modality in modalities:
            modality_dir = root / modality
            if not modality_dir.is_dir():
                exists = False
                break
        else:
            exists = True
        return exists

    @staticmethod
    def _get_subjects_list(root):
        episurg_dir = root / 'EPISURG'
        subjects_dir = episurg_dir / 'subjects'
        csv_path = episurg_dir / 'subjects.csv'
        with open(csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            subjects = []
            for row in reader:
                subject_id = row['Subject']
                subject_dir = subjects_dir / subject_id
                subject_dict = {'subject_id': subject_id, 'hemisphere': row['Hemisphere'], 'surgery_type': row['Type']}
                preop_dir = subject_dir / 'preop'
                preop_paths = list(preop_dir.glob('*preop*'))
                assert len(preop_paths) <= 1
                if preop_paths:
                    subject_dict['preop_mri'] = ScalarImage(preop_paths[0])
                postop_dir = subject_dir / 'postop'
                postop_path = list(postop_dir.glob('*postop-t1mri*'))[0]
                subject_dict['postop_mri'] = ScalarImage(postop_path)
                for seg_path in postop_dir.glob('*seg*'):
                    seg_id = seg_path.name[-8]
                    subject_dict[f'seg_{seg_id}'] = LabelMap(seg_path)
                subjects.append(Subject(**subject_dict))
        return subjects

    def _download(self, root: Path):
        """Download the EPISURG data if it does not exist already."""
        if (root / 'EPISURG').is_dir():
            return
        root.mkdir(exist_ok=True, parents=True)
        download_and_extract_archive(self.data_url, download_root=root, md5=self.md5)
        (root / 'EPISURG.zip').unlink()

    def _glob_subjects(self, string):
        subjects = []
        for subject in self._subjects:
            for image_name in subject:
                if string in image_name:
                    subjects.append(subject)
                    break
        return subjects

    def _get_labeled_subjects(self):
        return self._glob_subjects('seg')

    def _get_paired_subjects(self):
        return self._glob_subjects('preop')

    def _get_subset(self, subjects):
        dataset = SubjectsDataset(subjects, transform=self._transform, **self.kwargs)
        return dataset

    def get_labeled(self) -> SubjectsDataset:
        """Get dataset from subjects with manual annotations."""
        return self._get_subset(self._get_labeled_subjects())

    def get_unlabeled(self) -> SubjectsDataset:
        """Get dataset from subjects without manual annotations."""
        subjects = [s for s in self._subjects if s not in self._get_labeled_subjects()]
        return self._get_subset(subjects)

    def get_paired(self) -> SubjectsDataset:
        """Get dataset from subjects with pre- and post-op MRI."""
        return self._get_subset(self._get_paired_subjects())

class BITE3(BITE):
    """Pre- and post-resection MR images in BITE.

    *The goal of BITE is to share in vivo medical images of patients wtith
    brain tumors to facilitate the development and validation of new image
    processing algorithms.*

    Please check the `BITE website`_ for more information and
    acknowledgments instructions.

    .. _BITE website: https://nist.mni.mcgill.ca/bite-brain-images-of-tumors-for-evaluation-database/

    Args:
        root: Root directory to which the dataset will be downloaded.
        transform: An instance of
            :class:`~torchio.transforms.transform.Transform`.
        download: If set to ``True``, will download the data into :attr:`root`.
    """
    dirname = 'group3'

    def _download(self, root: Path):
        if (root / self.dirname).is_dir():
            return
        root.mkdir(exist_ok=True, parents=True)
        filename = f'{self.dirname}.tar.gz'
        url = self.base_url + filename
        download_and_extract_archive(url, download_root=root, md5='e415b63887c40b727c45552614b44634')
        (root / filename).unlink()

    def _get_subjects_list(self, root: Path):
        subjects_dir = root / self.dirname
        subjects = []
        for i in range(1, 15):
            if i == 13:
                continue
            subject_id = f'{i:02d}'
            subject_dir = subjects_dir / subject_id
            preop_path = subject_dir / f'{subject_id}_preop_mri.mnc'
            postop_path = subject_dir / f'{subject_id}_postop_mri.mnc'
            images_dict: dict[str, Image] = {}
            images_dict['preop'] = ScalarImage(preop_path)
            images_dict['postop'] = ScalarImage(postop_path)
            for fp in subject_dir.glob('*tumor*'):
                images_dict[fp.stem[3:]] = LabelMap(fp)
            subject = Subject(images_dict)
            subjects.append(subject)
        return subjects

class Pediatric(SubjectMNI):
    """MNI pediatric atlases.

    See `the MNI website <https://nist.mni.mcgill.ca/pediatric-atlases-4-5-18-5y/>`_
    for more information.

    .. image:: https://nist.mni.mcgill.ca/wp-content/uploads/2016/04/nihpd_asym_all_sm.jpg
        :alt: Pediatric MNI template

    Arguments:
        years: Tuple of 2 ages. Possible values are: ``(4.5, 18.5)``,
            ``(4.5, 8.5)``,
            ``(7, 11)``,
            ``(7.5, 13.5)``,
            ``(10, 14)`` and
            ``(13, 18.5)``.
        symmetric: If ``True``, the left-right symmetric templates will be
            used. Else, the asymmetric (natural) templates will be used.
    """

    def __init__(self, years, symmetric=False):
        self.url_dir = 'http://www.bic.mni.mcgill.ca/~vfonov/nihpd/obj1/'
        sym_string = 'sym' if symmetric else 'asym'
        if not isinstance(years, tuple) or years not in SUPPORTED_YEARS:
            message = f'Years must be a tuple in {SUPPORTED_YEARS}'
            raise ValueError(message)
        a, b = years
        self.file_id = f'{sym_string}_{format_age(a)}-{format_age(b)}'
        self.name = f'nihpd_{self.file_id}_nifti'
        self.filename = f'{self.name}.zip'
        self.url = urllib.parse.urljoin(self.url_dir, self.filename)
        if not self.download_root.is_dir():
            download_and_extract_archive(self.url, download_root=self.download_root, filename=self.filename)
            (self.download_root / self.filename).unlink()
            for path in self.download_root.glob('*.nii'):
                compress(path)
                path.unlink()
        try:
            subject_dict = self.get_subject_dict('.nii.gz')
        except FileNotFoundError:
            subject_dict = self.get_subject_dict('.nii')
        super().__init__(subject_dict)

    def get_subject_dict(self, extension):
        root = self.download_root
        subject_dict = {'t1': ScalarImage(root / f'nihpd_{self.file_id}_t1w{extension}'), 't2': ScalarImage(root / f'nihpd_{self.file_id}_t2w{extension}'), 'pd': ScalarImage(root / f'nihpd_{self.file_id}_pdw{extension}'), 'mask': LabelMap(root / f'nihpd_{self.file_id}_mask{extension}')}
        return subject_dict

def format_age(n):
    integer = int(n)
    decimal = int(10 * (n - integer))
    return f'{integer:02d}.{decimal}'

def compress(input_path: TypePath, output_path: TypePath | None=None) -> Path:
    if output_path is None:
        output_path = Path(input_path).with_suffix('.nii.gz')
    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return Path(output_path)

class ICBM2009CNonlinearSymmetric(SubjectMNI):
    """ICBM template.

    More information can be found in the `website
    <http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009>`_.

    .. image:: http://www.bic.mni.mcgill.ca/uploads/ServicesAtlases/mni_icbm152_sym_09c_small.jpg
        :alt: ICBM 2009c Nonlinear Symmetric

    Args:
        load_4d_tissues: If ``True``, the tissue probability maps will be loaded
            together into a 4D image. Otherwise, they will be loaded into
            independent images.

    Example:
        >>> import torchio as tio
        >>> icbm = tio.datasets.ICBM2009CNonlinearSymmetric()
        >>> icbm
        ICBM2009CNonlinearSymmetric(Keys: ('t1', 'eyes', 'face', 'brain', 't2', 'pd', 'tissues'); images: 7)
        >>> icbm = tio.datasets.ICBM2009CNonlinearSymmetric(load_4d_tissues=False)
        >>> icbm
        ICBM2009CNonlinearSymmetric(Keys: ('t1', 'eyes', 'face', 'brain', 't2', 'pd', 'gm', 'wm', 'csf'); images: 9)
    """

    def __init__(self, load_4d_tissues: bool=True):
        self.name = 'mni_icbm152_nlin_sym_09c_nifti'
        self.url_base = 'http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/'
        self.filename = f'{self.name}.zip'
        self.url = urllib.parse.urljoin(self.url_base, self.filename)
        download_root = get_torchio_cache_dir() / self.name
        if not download_root.is_dir():
            download_and_extract_archive(self.url, download_root=download_root, filename=self.filename, remove_finished=True)
        files_dir = download_root / 'mni_icbm152_nlin_sym_09c'
        p = files_dir / 'mni_icbm152'
        m = 'tal_nlin_sym_09c'
        s = '.nii.gz'
        tissues_path = files_dir / f'{p}_tissues_{m}.nii.gz'
        if not tissues_path.is_file():
            gm = LabelMap(f'{p}_gm_{m}.nii')
            wm = LabelMap(f'{p}_wm_{m}.nii')
            csf = LabelMap(f'{p}_csf_{m}.nii')
            gm.set_data(torch.cat((gm.data, wm.data, csf.data)))
            gm.save(tissues_path)
        for fp in files_dir.glob('*.nii'):
            compress(fp, fp.with_suffix('.nii.gz'))
            fp.unlink()
        subject_dict = {'t1': ScalarImage(f'{p}_t1_{m}{s}'), 'eyes': LabelMap(f'{p}_t1_{m}_eye_mask{s}'), 'face': LabelMap(f'{p}_t1_{m}_face_mask{s}'), 'brain': LabelMap(f'{p}_t1_{m}_mask{s}'), 't2': ScalarImage(f'{p}_t2_{m}{s}'), 'pd': ScalarImage(f'{p}_csf_{m}{s}')}
        if load_4d_tissues:
            subject_dict['tissues'] = LabelMap(tissues_path, channels_last=True)
        else:
            subject_dict['gm'] = LabelMap(f'{p}_gm_{m}{s}')
            subject_dict['wm'] = LabelMap(f'{p}_wm_{m}{s}')
            subject_dict['csf'] = LabelMap(f'{p}_csf_{m}{s}')
        super().__init__(subject_dict)

class Colin27(SubjectMNI):
    NAME_TO_LABEL = {name: label for label, name in TISSUES_2008.items()}
    "Colin27 MNI template.\n\n    More information can be found in the website of the\n    `1998 <https://nist.mni.mcgill.ca/colin-27-average-brain/>`_ and\n    `2008 <http://www.bic.mni.mcgill.ca/ServicesAtlases/Colin27Highres>`_\n    versions.\n\n    .. image:: http://www.bic.mni.mcgill.ca/uploads/ServicesAtlases/mni_colin27_2008.jpg\n        :alt: MNI Colin 27 2008 version\n\n    Arguments:\n        version: Template year. It can be ``1998`` or ``2008``.\n\n    .. warning:: The resolution of the ``2008`` version is quite high. The\n        subject instance will contain four images of size\n        :math:`362 \\times 434 \\times 362`, therefore applying a transform to\n        it might take longer than expected.\n\n    Example:\n        >>> import torchio as tio\n        >>> colin_1998 = tio.datasets.Colin27(version=1998)\n        >>> colin_1998\n        Colin27(Keys: ('t1', 'head', 'brain'); images: 3)\n        >>> colin_1998.load()\n        >>> colin_1998.t1\n        ScalarImage(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; memory: 27.1 MiB; type: intensity)\n        >>>\n        >>> colin_2008 = tio.datasets.Colin27(version=2008)\n        >>> colin_2008\n        Colin27(Keys: ('t1', 't2', 'pd', 'cls'); images: 4)\n        >>> colin_2008.load()\n        >>> colin_2008.t1\n        ScalarImage(shape: (1, 362, 434, 362); spacing: (0.50, 0.50, 0.50); orientation: RAS+; memory: 217.0 MiB; type: intensity)\n    "

    def __init__(self, version=1998):
        if version not in (1998, 2008):
            raise ValueError(f'Version must be 1998 or 2008, not "{version}"')
        self.version = version
        self.name = f'mni_colin27_{version}_nifti'
        self.url_dir = urllib.parse.urljoin(self.url_base, 'colin27/')
        self.filename = f'{self.name}.zip'
        self.url = urllib.parse.urljoin(self.url_dir, self.filename)
        if not self.download_root.is_dir():
            download_and_extract_archive(self.url, download_root=self.download_root, filename=self.filename)
            if version == 2008:
                path = self.download_root / 'colin27_cls_tal_hires.nii'
                cls_image = LabelMap(path)
                cls_image.set_data(cls_image.data.round().byte())
                cls_image.save(path)
            (self.download_root / self.filename).unlink()
            for path in self.download_root.glob('*.nii'):
                compress(path)
                path.unlink()
        try:
            subject_dict = self.get_subject_dict(self.download_root, extension='.nii.gz')
        except FileNotFoundError:
            subject_dict = self.get_subject_dict(self.download_root, extension='.nii')
        super().__init__(subject_dict)

    def get_subject_dict(self, download_root, extension):
        if self.version == 1998:
            subject_dict = Colin1998.get_subject_dict(download_root, extension)
        elif self.version == 2008:
            subject_dict = Colin2008.get_subject_dict(download_root, extension)
        return subject_dict

class Sheep(SubjectMNI):

    def __init__(self):
        self.name = 'NIFTI_ovine_05mm'
        self.url_dir = urllib.parse.urljoin(self.url_base, 'sheep/')
        self.filename = f'{self.name}.zip'
        self.url = urllib.parse.urljoin(self.url_dir, self.filename)
        t1_nii_path = self.download_root / 'ovine_model_05.nii'
        t1_niigz_path = self.download_root / 'ovine_model_05.nii.gz'
        if not self.download_root.is_dir():
            download_and_extract_archive(self.url, download_root=self.download_root, filename=self.filename)
            shutil.rmtree(self.download_root / 'masks')
            for path in self.download_root.iterdir():
                if path == t1_nii_path:
                    compress(t1_nii_path, t1_niigz_path)
                path.unlink()
        try:
            subject_dict = {'t1': ScalarImage(t1_niigz_path)}
        except FileNotFoundError:
            subject_dict = {'t1': ScalarImage(t1_nii_path)}
        super().__init__(subject_dict)

