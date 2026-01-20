# Cluster 6

def check_anchor_order(m):
    a = m.anchors.prod(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da.sign() != ds.sign():
        LOGGER.info(f'{PREFIX}Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)

class EarlyStopping:

    def __init__(self, patience=30):
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.patience = patience or float('inf')
        self.possible_stop = False

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch
        self.possible_stop = delta >= self.patience - 1
        stop = delta >= self.patience
        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\nTo update EarlyStopping(patience={self.patience}) pass a new patience value, i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.')
        return stop

class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        copy_attr(self.ema, model, include, exclude)

def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if len(include) and k not in include or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

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

def create_dataset_artifact(opt):
    logger = WandbLogger(opt, None, job_type='Dataset Creation')
    if not logger.wandb:
        LOGGER.info('install wandb using `pip install wandb` to log the dataset')

class Model(nn.Module):

    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f'Overriding model.yaml nc={self.yaml['nc']} with nc={nc}')
            self.yaml['nc'] = nc
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]
        self.inplace = self.yaml.get('inplace', True)
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)
        return self._forward_once(x, profile, visualize)

    def _forward_augment(self, x):
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return (torch.cat(y, 1), None)

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = ([], [])
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        if self.inplace:
            p[..., :4] /= scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]
        else:
            x, y, wh = (p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale)
            if flips == 2:
                y = img_size[0] - y
            elif flips == 3:
                x = img_size[1] - x
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        nl = self.model[-1].nl
        g = sum((4 ** x for x in range(nl)))
        e = 1
        i = y[0].shape[1] // g * sum((4 ** x for x in range(e)))
        y[0] = y[0][:, :-i]
        i = y[-1].shape[1] // g * sum((4 ** (nl - 1 - x) for x in range(e)))
        y[-1] = y[-1][:, i:]
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1000000000.0 * 2 if thop else 0
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f'{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}')
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f'{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total')

    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]
        for mi in m.m:
            b = mi.bias.detach().view(m.na, -1).T
            LOGGER.info(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        self = super()._apply(fn)
        m = self.model[-1]
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 0.001
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, groups=conv.groups, bias=True).requires_grad_(False).to(conv.weight.device)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv

def parse_model(d, ch):
    LOGGER.info(f'\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}')
    anchors, nc, gd, gw = (d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'])
    na = len(anchors[0]) // 2 if isinstance(anchors, list) else anchors
    no = na * (nc + 5)
    layers, save, c2 = ([], [], ch[-1])
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except NameError:
                pass
        n = n_ = max(round(n * gd), 1) if n > 1 else n
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            c1, c2 = (ch[f], args[0])
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum((ch[x] for x in f))
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum((x.numel() for x in m_.parameters()))
        m_.i, m_.f, m_.type, m_.np = (i, f, t, np)
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')
        save.extend((x % i for x in ([f] if isinstance(f, int) else f) if x != -1))
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return (nn.Sequential(*layers), sorted(save))

def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor

def parse_model(d, ch, model, imgsz):
    LOGGER.info(f'\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}')
    anchors, nc, gd, gw = (d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'])
    na = len(anchors[0]) // 2 if isinstance(anchors, list) else anchors
    no = na * (nc + 5)
    layers, save, c2 = ([], [], ch[-1])
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m_str = m
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except NameError:
                pass
        n = max(round(n * gd), 1) if n > 1 else n
        if m in [nn.Conv2d, Conv, Bottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = (ch[f], args[0])
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2
            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum((ch[-1 if x == -1 else x + 1] for x in f))
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
            args.append(imgsz)
        else:
            c2 = ch[f]
        tf_m = eval('TF' + m_str.replace('nn.', ''))
        m_ = keras.Sequential([tf_m(*args, w=model.model[i][j]) for j in range(n)]) if n > 1 else tf_m(*args, w=model.model[i])
        torch_m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum((x.numel() for x in torch_m_.parameters()))
        m_.i, m_.f, m_.type, m_.np = (i, f, t, np)
        LOGGER.info(f'{i:>3}{str(f):>18}{str(n):>3}{np:>10}  {t:<40}{str(args):<30}')
        save.extend((x % i for x in ([f] if isinstance(f, int) else f) if x != -1))
        layers.append(m_)
        ch.append(c2)
    return (keras.Sequential(layers), sorted(save))

class TFModel:

    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, model=None, imgsz=(640, 640)):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f'Overriding {cfg} nc={self.yaml['nc']} with nc={nc}')
            self.yaml['nc'] = nc
        self.model, self.savelist = parse_model(deepcopy(self.yaml), ch=[ch], model=model, imgsz=imgsz)

    def predict(self, inputs, tf_nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25):
        y = []
        x = inputs
        for i, m in enumerate(self.model.layers):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.savelist else None)
        if tf_nms:
            boxes = self._xywh2xyxy(x[0][..., :4])
            probs = x[0][:, :, 4:5]
            classes = x[0][:, :, 5:]
            scores = probs * classes
            if agnostic_nms:
                nms = AgnosticNMS()((boxes, classes, scores), topk_all, iou_thres, conf_thres)
                return (nms, x[1])
            else:
                boxes = tf.expand_dims(boxes, 2)
                nms = tf.image.combined_non_max_suppression(boxes, scores, topk_per_class, topk_all, iou_thres, conf_thres, clip_boxes=False)
                return (nms, x[1])
        return x[0]

    @staticmethod
    def _xywh2xyxy(xywh):
        x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
        return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)

class AutoShape(nn.Module):
    conf = 0.25
    iou = 0.45
    agnostic = False
    multi_label = False
    classes = None
    max_det = 1000
    amp = False

    def __init__(self, model):
        super().__init__()
        LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())
        self.dmb = isinstance(model, DetectMultiBackend)
        self.pt = not self.dmb or model.pt
        self.model = model.eval()

    def _apply(self, fn):
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        t = [time_sync()]
        p = next(self.model.parameters()) if self.pt else torch.zeros(1)
        autocast = self.amp and p.device.type != 'cpu'
        if isinstance(imgs, torch.Tensor):
            with amp.autocast(enabled=autocast):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])
        shape0, shape1, files = ([], [], [])
        for i, im in enumerate(imgs):
            f = f'image{i}'
            if isinstance(im, (str, Path)):
                im, f = (Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im)
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):
                im, f = (np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f)
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:
                im = im.transpose((1, 2, 0))
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)
            s = im.shape[:2]
            shape0.append(s)
            g = size / max(s)
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)
        shape1 = [make_divisible(x, self.stride) for x in np.stack(shape1, 0).max(0)]
        x = [letterbox(im, new_shape=shape1 if self.pt else size, auto=False)[0] for im in imgs]
        x = np.stack(x, 0) if n > 1 else x[0][None]
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255
        t.append(time_sync())
        with amp.autocast(enabled=autocast):
            y = self.model(x, augment, profile)
            t.append(time_sync())
            y = non_max_suppression(y if self.dmb else y[0], self.conf, iou_thres=self.iou, classes=self.classes, agnostic=self.agnostic, multi_label=self.multi_label, max_det=self.max_det)
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])
            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)

class Detections:

    def __init__(self, imgs, pred, files, times=(0, 0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]
        self.imgs = imgs
        self.pred = pred
        self.names = names
        self.files = files
        self.times = times
        self.xyxy = pred
        self.xywh = [xyxy2xywh(x) for x in pred]
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]
        self.n = len(self.pred)
        self.t = tuple(((times[i + 1] - times[i]) * 1000 / self.n for i in range(3)))
        self.s = shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()
                    s += f'{n} {self.names[int(c)]}{'s' * (n > 1)}, '
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label, 'im': save_one_box(box, im, file=file, save=save)})
                        else:
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'
            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)
                if i == self.n - 1:
                    LOGGER.info(f'Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}')
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)
        self.display(save=True, save_dir=save_dir)

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)

    def render(self):
        self.display(render=True)
        return self.imgs

    def pandas(self):
        new = copy(self)
        ca = ('xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name')
        cb = ('xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name')
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        r = range(self.n)
        x = [Detections([self.imgs[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        return x

    def __len__(self):
        return self.n

