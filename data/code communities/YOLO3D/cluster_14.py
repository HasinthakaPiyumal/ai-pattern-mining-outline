# Cluster 14

def metric(k, wh):
    r = wh[:, None] / k[None]
    x = torch.min(r, 1 / r).min(2)[0]
    return (x, x.max(1)[0])

def anchor_fitness(k):
    _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
    return (best * (best > thr).float()).mean()

def print_results(k, verbose=True):
    k = k[np.argsort(k.prod(1))]
    x, best = metric(k, wh0)
    bpr, aat = ((best > thr).float().mean(), (x > thr).float().mean() * n)
    s = f'{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, past_thr={x[x > thr].mean():.3f}-mean: '
    for i, x in enumerate(k):
        s += '%i,%i, ' % (round(x[0]), round(x[1]))
    if verbose:
        LOGGER.info(s[:-2])
    return k

def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            dataset: path to data.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans
    thr = 1 / thr

    def metric(k, wh):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]
        return (x, x.max(1)[0])

    def anchor_fitness(k):
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()

    def print_results(k, verbose=True):
        k = k[np.argsort(k.prod(1))]
        x, best = metric(k, wh0)
        bpr, aat = ((best > thr).float().mean(), (x > thr).float().mean() * n)
        s = f'{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, past_thr={x[x > thr].mean():.3f}-mean: '
        for i, x in enumerate(k):
            s += '%i,%i, ' % (round(x[0]), round(x[1]))
        if verbose:
            LOGGER.info(s[:-2])
        return k
    if isinstance(dataset, str):
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
        from utils.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f'{PREFIX}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[(wh0 >= 2.0).any(1)]
    LOGGER.info(f'{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...')
    s = wh.std(0)
    k, dist = kmeans(wh / s, n, iter=30)
    assert len(k) == n, f'{PREFIX}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}'
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)
    wh0 = torch.tensor(wh0, dtype=torch.float32)
    k = print_results(k, verbose=False)
    npr = np.random
    f, sh, mp, s = (anchor_fitness(k), k.shape, 0.9, 0.1)
    pbar = tqdm(range(gen), desc=f'{PREFIX}Evolving anchors with Genetic Algorithm:')
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = (fg, kg.copy())
            pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k, verbose)
    return print_results(k)

def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
    """ Return dataset statistics dictionary with images and instances counts per split per class
    To run in parent directory: export PYTHONPATH="$PWD/yolov5"
    Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', autodownload=True)
    Usage2: from utils.datasets import *; dataset_stats('../datasets/coco128_with_yaml.zip')
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def round_labels(labels):
        return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

    def unzip(path):
        if str(path).endswith('.zip'):
            assert Path(path).is_file(), f'Error unzipping {path}, file not found'
            ZipFile(path).extractall(path=path.parent)
            dir = path.with_suffix('')
            return (True, str(dir), next(dir.rglob('*.yaml')))
        else:
            return (False, None, path)

    def hub_ops(f, max_dim=1920):
        f_new = im_dir / Path(f).name
        try:
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)
            if r < 1.0:
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, 'JPEG', quality=75, optimize=True)
        except Exception as e:
            print(f'WARNING: HUB ops PIL failure {f}: {e}')
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)
            if r < 1.0:
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(f_new), im)
    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_yaml(yaml_path), errors='ignore') as f:
        data = yaml.safe_load(f)
        if zipped:
            data['path'] = data_dir
    check_dataset(data, autodownload)
    hub_dir = Path(data['path'] + ('-hub' if hub else ''))
    stats = {'nc': data['nc'], 'names': data['names']}
    for split in ('train', 'val', 'test'):
        if data.get(split) is None:
            stats[split] = None
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))
        x = np.array(x)
        stats[split] = {'instance_stats': {'total': int(x.sum()), 'per_class': x.sum(0).tolist()}, 'image_stats': {'total': dataset.n, 'unlabelled': int(np.all(x == 0, 1).sum()), 'per_class': (x > 0).sum(0).tolist()}, 'labels': [{str(Path(k).name): round_labels(v.tolist())} for k, v in zip(dataset.img_files, dataset.labels)]}
        if hub:
            im_dir = hub_dir / 'images'
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(hub_ops, dataset.img_files), total=dataset.n, desc='HUB Ops'):
                pass
    stats_path = hub_dir / 'stats.json'
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix('.npy')
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f'stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')
            file = stats_path.with_suffix('.json')
            t1 = time.time()
            with open(file, 'w') as f:
                json.dump(stats, f)
            t2 = time.time()
            with open(file) as f:
                x = json.load(f)
            print(f'stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')
    if hub:
        print(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats

def unzip(path):
    if str(path).endswith('.zip'):
        assert Path(path).is_file(), f'Error unzipping {path}, file not found'
        ZipFile(path).extractall(path=path.parent)
        dir = path.with_suffix('')
        return (True, str(dir), next(dir.rglob('*.yaml')))
    else:
        return (False, None, path)

def check_dataset(data, autodownload=True):
    extract_dir = ''
    if isinstance(data, (str, Path)) and str(data).endswith('.zip'):
        download(data, dir='../datasets', unzip=True, delete=False, curl=False, threads=1)
        data = next((Path('../datasets') / Path(data).stem).rglob('*.yaml'))
        extract_dir, autodownload = (data.parent, False)
    if isinstance(data, (str, Path)):
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)
    path = extract_dir or Path(data.get('path') or '')
    for k in ('train', 'val', 'test'):
        if data.get(k):
            data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]
    assert 'nc' in data, "Dataset 'nc' key missing."
    if 'names' not in data:
        data['names'] = [f'class{i}' for i in range(data['nc'])]
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]
        if not all((x.exists() for x in val)):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            if s and autodownload:
                root = path.parent if 'path' in data else '..'
                if s.startswith('http') and s.endswith('.zip'):
                    f = Path(s).name
                    print(f'Downloading {s} to {f}...')
                    torch.hub.download_url_to_file(s, f)
                    Path(root).mkdir(parents=True, exist_ok=True)
                    ZipFile(f).extractall(path=root)
                    Path(f).unlink()
                    r = None
                elif s.startswith('bash '):
                    print(f'Running {s} ...')
                    r = os.system(s)
                else:
                    r = exec(s, {'yaml': data})
                print(f'Dataset autodownload {(f'success, saved to {root}' if r in (0, None) else 'failure')}\n')
            else:
                raise Exception('Dataset not found.')
    return data

def round_labels(labels):
    return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

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

