# Cluster 25

class LoadImagesAndLabels(Dataset):
    cache_version = 0.6

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False, cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and (not self.rect)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations() if augment else None
        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted((x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS))
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')
        self.label_files = img2label_paths(self.img_files)
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = (np.load(cache_path, allow_pickle=True).item(), True)
            assert cache['version'] == self.cache_version
            assert cache['hash'] == get_hash(self.label_files + self.img_files)
        except:
            cache, exists = (self.cache_labels(cache_path, prefix), False)
        nf, nm, ne, nc, n = cache.pop('results')
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())
        self.label_files = img2label_paths(cache.keys())
        n = len(shapes)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)
        nb = bi[-1] + 1
        self.batch = bi
        self.n = n
        self.indices = range(n)
        include_class = []
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0
        if self.rect:
            s = self.shapes
            ar = s[:, 1] / s[:, 0]
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]
            ar = ar[irect]
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = (ari.min(), ari.max())
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride
        self.imgs, self.img_npy = ([None] * n, [None] * n)
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0
            self.img_hw0, self.img_hw = ([None] * n, [None] * n)
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1000000000.0:.1f}GB {cache_images})'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        x = {}
        nm, nf, ne, nc, msgs = (0, 0, 0, 0, [])
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))), desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted'
        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = (nf, nm, ne, nc, len(self.img_files))
        x['msgs'] = msgs
        x['version'] = self.cache_version
        try:
            np.save(path, x)
            path.with_suffix('.cache.npy').rename(path)
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        index = self.indices[index]
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            img, labels = load_mosaic(self, index)
            shapes = None
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))
        else:
            img, (h0, w0), (h, w) = load_image(self, index)
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = ((h0, w0), ((h / h0, w / w0), pad))
            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
            if self.augment:
                img, labels = random_perspective(img, labels, degrees=hyp['degrees'], translate=hyp['translate'], scale=hyp['scale'], shear=hyp['shear'], perspective=hyp['perspective'])
        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=0.001)
        if self.augment:
            img, labels = self.albumentations(img, labels)
            nl = len(labels)
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        return (torch.from_numpy(img), labels_out, self.img_files[index], shapes)

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i
        return (torch.stack(img, 0), torch.cat(label, 0), path, shapes)

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = ([], [], path[:n], shapes[:n])
        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])
        for i in range(n):
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear', align_corners=False)[0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)
        for i, l in enumerate(label4):
            l[:, 0] = i
        return (torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4)

def img2label_paths(img_paths):
    sa, sb = (os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep)
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def get_hash(paths):
    size = sum((os.path.getsize(p) for p in paths if os.path.exists(p)))
    h = hashlib.md5(str(size).encode())
    h.update(''.join(paths).encode())
    return h.hexdigest()

def extract_boxes(path='../datasets/coco128'):
    path = Path(path)
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None
    files = list(path.rglob('*.*'))
    n = len(files)
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            im = cv2.imread(str(im_file))[..., ::-1]
            h, w = im.shape[:2]
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file) as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
                for j, x in enumerate(lb):
                    c = int(x[0])
                    f = path / 'classifier' / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)
                    b = x[1:] * [w, h, w, h]
                    b[2:] = b[2:] * 1.2 + 3
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)
                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'

def autosplit(path='../datasets/coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)
    files = sorted((x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS))
    n = len(files)
    random.seed(0)
    indices = random.choices([0, 1, 2], weights=weights, k=n)
    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']
    [(path.parent / x).unlink(missing_ok=True) for x in txt]
    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')

class WandbLogger:
    """Log training runs, datasets, models, and predictions to Weights & Biases.

    This logger sends information to W&B at wandb.ai. By default, this information
    includes hyperparameters, system configuration and metrics, model metrics,
    and basic data metrics and analyses.

    By providing additional command line arguments to train.py, datasets,
    models and predictions can also be logged.

    For more on how this logger is used, see the Weights & Biases documentation:
    https://docs.wandb.com/guides/integrations/yolov5
    """

    def __init__(self, opt, run_id=None, job_type='Training'):
        """
        - Initialize WandbLogger instance
        - Upload dataset if opt.upload_dataset is True
        - Setup trainig processes if job_type is 'Training'

        arguments:
        opt (namespace) -- Commandline arguments for this run
        run_id (str) -- Run ID of W&B run to be resumed
        job_type (str) -- To set the job_type for this run

       """
        self.job_type = job_type
        self.wandb, self.wandb_run = (wandb, None if not wandb else wandb.run)
        self.val_artifact, self.train_artifact = (None, None)
        self.train_artifact_path, self.val_artifact_path = (None, None)
        self.result_artifact = None
        self.val_table, self.result_table = (None, None)
        self.bbox_media_panel_images = []
        self.val_table_path_map = None
        self.max_imgs_to_log = 16
        self.wandb_artifact_data_dict = None
        self.data_dict = None
        if isinstance(opt.resume, str):
            if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
                entity, project, run_id, model_artifact_name = get_run_info(opt.resume)
                model_artifact_name = WANDB_ARTIFACT_PREFIX + model_artifact_name
                assert wandb, 'install wandb to resume wandb runs'
                self.wandb_run = wandb.init(id=run_id, project=project, entity=entity, resume='allow', allow_val_change=True)
                opt.resume = model_artifact_name
        elif self.wandb:
            self.wandb_run = wandb.init(config=opt, resume='allow', project='YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem, entity=opt.entity, name=opt.name if opt.name != 'exp' else None, job_type=job_type, id=run_id, allow_val_change=True) if not wandb.run else wandb.run
        if self.wandb_run:
            if self.job_type == 'Training':
                if opt.upload_dataset:
                    if not opt.resume:
                        self.wandb_artifact_data_dict = self.check_and_upload_dataset(opt)
                if opt.resume:
                    if isinstance(opt.resume, str) and opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
                        self.data_dict = dict(self.wandb_run.config.data_dict)
                    else:
                        self.data_dict = check_wandb_dataset(opt.data)
                else:
                    self.data_dict = check_wandb_dataset(opt.data)
                    self.wandb_artifact_data_dict = self.wandb_artifact_data_dict or self.data_dict
                    self.wandb_run.config.update({'data_dict': self.wandb_artifact_data_dict}, allow_val_change=True)
                self.setup_training(opt)
            if self.job_type == 'Dataset Creation':
                self.wandb_run.config.update({'upload_dataset': True})
                self.data_dict = self.check_and_upload_dataset(opt)

    def check_and_upload_dataset(self, opt):
        """
        Check if the dataset format is compatible and upload it as W&B artifact

        arguments:
        opt (namespace)-- Commandline arguments for current run

        returns:
        Updated dataset info dictionary where local dataset paths are replaced by WAND_ARFACT_PREFIX links.
        """
        assert wandb, 'Install wandb to upload dataset'
        config_path = self.log_dataset_artifact(opt.data, opt.single_cls, 'YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem)
        with open(config_path, errors='ignore') as f:
            wandb_data_dict = yaml.safe_load(f)
        return wandb_data_dict

    def setup_training(self, opt):
        """
        Setup the necessary processes for training YOLO models:
          - Attempt to download model checkpoint and dataset artifacts if opt.resume stats with WANDB_ARTIFACT_PREFIX
          - Update data_dict, to contain info of previous run if resumed and the paths of dataset artifact if downloaded
          - Setup log_dict, initialize bbox_interval

        arguments:
        opt (namespace) -- commandline arguments for this run

        """
        self.log_dict, self.current_epoch = ({}, 0)
        self.bbox_interval = opt.bbox_interval
        if isinstance(opt.resume, str):
            modeldir, _ = self.download_model_artifact(opt)
            if modeldir:
                self.weights = Path(modeldir) / 'last.pt'
                config = self.wandb_run.config
                opt.weights, opt.save_period, opt.batch_size, opt.bbox_interval, opt.epochs, opt.hyp = (str(self.weights), config.save_period, config.batch_size, config.bbox_interval, config.epochs, config.hyp)
        data_dict = self.data_dict
        if self.val_artifact is None:
            self.train_artifact_path, self.train_artifact = self.download_dataset_artifact(data_dict.get('train'), opt.artifact_alias)
            self.val_artifact_path, self.val_artifact = self.download_dataset_artifact(data_dict.get('val'), opt.artifact_alias)
        if self.train_artifact_path is not None:
            train_path = Path(self.train_artifact_path) / 'data/images/'
            data_dict['train'] = str(train_path)
        if self.val_artifact_path is not None:
            val_path = Path(self.val_artifact_path) / 'data/images/'
            data_dict['val'] = str(val_path)
        if self.val_artifact is not None:
            self.result_artifact = wandb.Artifact('run_' + wandb.run.id + '_progress', 'evaluation')
            columns = ['epoch', 'id', 'ground truth', 'prediction']
            columns.extend(self.data_dict['names'])
            self.result_table = wandb.Table(columns)
            self.val_table = self.val_artifact.get('val')
            if self.val_table_path_map is None:
                self.map_val_table_path()
        if opt.bbox_interval == -1:
            self.bbox_interval = opt.bbox_interval = opt.epochs // 10 if opt.epochs > 10 else 1
        train_from_artifact = self.train_artifact_path is not None and self.val_artifact_path is not None
        if train_from_artifact:
            self.data_dict = data_dict

    def download_dataset_artifact(self, path, alias):
        """
        download the model checkpoint artifact if the path starts with WANDB_ARTIFACT_PREFIX

        arguments:
        path -- path of the dataset to be used for training
        alias (str)-- alias of the artifact to be download/used for training

        returns:
        (str, wandb.Artifact) -- path of the downladed dataset and it's corresponding artifact object if dataset
        is found otherwise returns (None, None)
        """
        if isinstance(path, str) and path.startswith(WANDB_ARTIFACT_PREFIX):
            artifact_path = Path(remove_prefix(path, WANDB_ARTIFACT_PREFIX) + ':' + alias)
            dataset_artifact = wandb.use_artifact(artifact_path.as_posix().replace('\\', '/'))
            assert dataset_artifact is not None, "'Error: W&B dataset artifact doesn't exist'"
            datadir = dataset_artifact.download()
            return (datadir, dataset_artifact)
        return (None, None)

    def download_model_artifact(self, opt):
        """
        download the model checkpoint artifact if the resume path starts with WANDB_ARTIFACT_PREFIX

        arguments:
        opt (namespace) -- Commandline arguments for this run
        """
        if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
            model_artifact = wandb.use_artifact(remove_prefix(opt.resume, WANDB_ARTIFACT_PREFIX) + ':latest')
            assert model_artifact is not None, "Error: W&B model artifact doesn't exist"
            modeldir = model_artifact.download()
            epochs_trained = model_artifact.metadata.get('epochs_trained')
            total_epochs = model_artifact.metadata.get('total_epochs')
            is_finished = total_epochs is None
            assert not is_finished, 'training is finished, can only resume incomplete runs.'
            return (modeldir, model_artifact)
        return (None, None)

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        """
        Log the model checkpoint as W&B artifact

        arguments:
        path (Path)   -- Path of directory containing the checkpoints
        opt (namespace) -- Command line arguments for this run
        epoch (int)  -- Current epoch number
        fitness_score (float) -- fitness score for current epoch
        best_model (boolean) -- Boolean representing if the current checkpoint is the best yet.
        """
        model_artifact = wandb.Artifact('run_' + wandb.run.id + '_model', type='model', metadata={'original_url': str(path), 'epochs_trained': epoch + 1, 'save period': opt.save_period, 'project': opt.project, 'total_epochs': opt.epochs, 'fitness_score': fitness_score})
        model_artifact.add_file(str(path / 'last.pt'), name='last.pt')
        wandb.log_artifact(model_artifact, aliases=['latest', 'last', 'epoch ' + str(self.current_epoch), 'best' if best_model else ''])
        LOGGER.info(f'Saving model artifact on epoch {epoch + 1}')

    def log_dataset_artifact(self, data_file, single_cls, project, overwrite_config=False):
        """
        Log the dataset as W&B artifact and return the new data file with W&B links

        arguments:
        data_file (str) -- the .yaml file with information about the dataset like - path, classes etc.
        single_class (boolean)  -- train multi-class data as single-class
        project (str) -- project name. Used to construct the artifact path
        overwrite_config (boolean) -- overwrites the data.yaml file if set to true otherwise creates a new
        file with _wandb postfix. Eg -> data_wandb.yaml

        returns:
        the new .yaml file with artifact links. it can be used to start training directly from artifacts
        """
        upload_dataset = self.wandb_run.config.upload_dataset
        log_val_only = isinstance(upload_dataset, str) and upload_dataset == 'val'
        self.data_dict = check_dataset(data_file)
        data = dict(self.data_dict)
        nc, names = (1, ['item']) if single_cls else (int(data['nc']), data['names'])
        names = {k: v for k, v in enumerate(names)}
        if not log_val_only:
            self.train_artifact = self.create_dataset_table(LoadImagesAndLabels(data['train'], rect=True, batch_size=1), names, name='train') if data.get('train') else None
            if data.get('train'):
                data['train'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'train')
        self.val_artifact = self.create_dataset_table(LoadImagesAndLabels(data['val'], rect=True, batch_size=1), names, name='val') if data.get('val') else None
        if data.get('val'):
            data['val'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'val')
        path = Path(data_file)
        if not log_val_only:
            path = (path.stem if overwrite_config else path.stem + '_wandb') + '.yaml'
            path = Path('data') / path
            data.pop('download', None)
            data.pop('path', None)
            with open(path, 'w') as f:
                yaml.safe_dump(data, f)
                LOGGER.info(f'Created dataset config file {path}')
        if self.job_type == 'Training':
            if not log_val_only:
                self.wandb_run.log_artifact(self.train_artifact)
            self.wandb_run.use_artifact(self.val_artifact)
            self.val_artifact.wait()
            self.val_table = self.val_artifact.get('val')
            self.map_val_table_path()
        else:
            self.wandb_run.log_artifact(self.train_artifact)
            self.wandb_run.log_artifact(self.val_artifact)
        return path

    def map_val_table_path(self):
        """
        Map the validation dataset Table like name of file -> it's id in the W&B Table.
        Useful for - referencing artifacts for evaluation.
        """
        self.val_table_path_map = {}
        LOGGER.info('Mapping dataset')
        for i, data in enumerate(tqdm(self.val_table.data)):
            self.val_table_path_map[data[3]] = data[0]

    def create_dataset_table(self, dataset: LoadImagesAndLabels, class_to_id: Dict[int, str], name: str='dataset'):
        """
        Create and return W&B artifact containing W&B Table of the dataset.

        arguments:
        dataset -- instance of LoadImagesAndLabels class used to iterate over the data to build Table
        class_to_id -- hash map that maps class ids to labels
        name -- name of the artifact

        returns:
        dataset artifact to be logged or used
        """
        artifact = wandb.Artifact(name=name, type='dataset')
        img_files = tqdm([dataset.path]) if isinstance(dataset.path, str) and Path(dataset.path).is_dir() else None
        img_files = tqdm(dataset.img_files) if not img_files else img_files
        for img_file in img_files:
            if Path(img_file).is_dir():
                artifact.add_dir(img_file, name='data/images')
                labels_path = 'labels'.join(dataset.path.rsplit('images', 1))
                artifact.add_dir(labels_path, name='data/labels')
            else:
                artifact.add_file(img_file, name='data/images/' + Path(img_file).name)
                label_file = Path(img2label_paths([img_file])[0])
                artifact.add_file(str(label_file), name='data/labels/' + label_file.name) if label_file.exists() else None
        table = wandb.Table(columns=['id', 'train_image', 'Classes', 'name'])
        class_set = wandb.Classes([{'id': id, 'name': name} for id, name in class_to_id.items()])
        for si, (img, labels, paths, shapes) in enumerate(tqdm(dataset)):
            box_data, img_classes = ([], {})
            for cls, *xywh in labels[:, 1:].tolist():
                cls = int(cls)
                box_data.append({'position': {'middle': [xywh[0], xywh[1]], 'width': xywh[2], 'height': xywh[3]}, 'class_id': cls, 'box_caption': '%s' % class_to_id[cls]})
                img_classes[cls] = class_to_id[cls]
            boxes = {'ground_truth': {'box_data': box_data, 'class_labels': class_to_id}}
            table.add_data(si, wandb.Image(paths, classes=class_set, boxes=boxes), list(img_classes.values()), Path(paths).name)
        artifact.add(table, name)
        return artifact

    def log_training_progress(self, predn, path, names):
        """
        Build evaluation Table. Uses reference from validation dataset table.

        arguments:
        predn (list): list of predictions in the native space in the format - [xmin, ymin, xmax, ymax, confidence, class]
        path (str): local path of the current evaluation image
        names (dict(int, str)): hash map that maps class ids to labels
        """
        class_set = wandb.Classes([{'id': id, 'name': name} for id, name in names.items()])
        box_data = []
        avg_conf_per_class = [0] * len(self.data_dict['names'])
        pred_class_count = {}
        for *xyxy, conf, cls in predn.tolist():
            if conf >= 0.25:
                cls = int(cls)
                box_data.append({'position': {'minX': xyxy[0], 'minY': xyxy[1], 'maxX': xyxy[2], 'maxY': xyxy[3]}, 'class_id': cls, 'box_caption': f'{names[cls]} {conf:.3f}', 'scores': {'class_score': conf}, 'domain': 'pixel'})
                avg_conf_per_class[cls] += conf
                if cls in pred_class_count:
                    pred_class_count[cls] += 1
                else:
                    pred_class_count[cls] = 1
        for pred_class in pred_class_count.keys():
            avg_conf_per_class[pred_class] = avg_conf_per_class[pred_class] / pred_class_count[pred_class]
        boxes = {'predictions': {'box_data': box_data, 'class_labels': names}}
        id = self.val_table_path_map[Path(path).name]
        self.result_table.add_data(self.current_epoch, id, self.val_table.data[id][1], wandb.Image(self.val_table.data[id][1], boxes=boxes, classes=class_set), *avg_conf_per_class)

    def val_one_image(self, pred, predn, path, names, im):
        """
        Log validation data for one image. updates the result Table if validation dataset is uploaded and log bbox media panel

        arguments:
        pred (list): list of scaled predictions in the format - [xmin, ymin, xmax, ymax, confidence, class]
        predn (list): list of predictions in the native space - [xmin, ymin, xmax, ymax, confidence, class]
        path (str): local path of the current evaluation image
        """
        if self.val_table and self.result_table:
            self.log_training_progress(predn, path, names)
        if len(self.bbox_media_panel_images) < self.max_imgs_to_log and self.current_epoch > 0:
            if self.current_epoch % self.bbox_interval == 0:
                box_data = [{'position': {'minX': xyxy[0], 'minY': xyxy[1], 'maxX': xyxy[2], 'maxY': xyxy[3]}, 'class_id': int(cls), 'box_caption': f'{names[cls]} {conf:.3f}', 'scores': {'class_score': conf}, 'domain': 'pixel'} for *xyxy, conf, cls in pred.tolist()]
                boxes = {'predictions': {'box_data': box_data, 'class_labels': names}}
                self.bbox_media_panel_images.append(wandb.Image(im, boxes=boxes, caption=path.name))

    def log(self, log_dict):
        """
        save the metrics to the logging dictionary

        arguments:
        log_dict (Dict) -- metrics/media to be logged in current step
        """
        if self.wandb_run:
            for key, value in log_dict.items():
                self.log_dict[key] = value

    def end_epoch(self, best_result=False):
        """
        commit the log_dict, model artifacts and Tables to W&B and flush the log_dict.

        arguments:
        best_result (boolean): Boolean representing if the result of this evaluation is best or not
        """
        if self.wandb_run:
            with all_logging_disabled():
                if self.bbox_media_panel_images:
                    self.log_dict['BoundingBoxDebugger'] = self.bbox_media_panel_images
                try:
                    wandb.log(self.log_dict)
                except BaseException as e:
                    LOGGER.info(f'An error occurred in wandb logger. The training will proceed without interruption. More info\n{e}')
                    self.wandb_run.finish()
                    self.wandb_run = None
                self.log_dict = {}
                self.bbox_media_panel_images = []
            if self.result_artifact:
                self.result_artifact.add(self.result_table, 'result')
                wandb.log_artifact(self.result_artifact, aliases=['latest', 'last', 'epoch ' + str(self.current_epoch), 'best' if best_result else ''])
                wandb.log({'evaluation': self.result_table})
                columns = ['epoch', 'id', 'ground truth', 'prediction']
                columns.extend(self.data_dict['names'])
                self.result_table = wandb.Table(columns)
                self.result_artifact = wandb.Artifact('run_' + wandb.run.id + '_progress', 'evaluation')

    def finish_run(self):
        """
        Log metrics if any and finish the current W&B run
        """
        if self.wandb_run:
            if self.log_dict:
                with all_logging_disabled():
                    wandb.log(self.log_dict)
            wandb.run.finish()

