# Cluster 32

@HOOKS.register_module()
class MMDetWandbHook(WandbLoggerHook):
    """Enhanced Wandb logger hook for MMDetection.

    Comparing with the :cls:`mmcv.runner.WandbLoggerHook`, this hook can not
    only automatically log all the metrics but also log the following extra
    information - saves model checkpoints as W&B Artifact, and
    logs model prediction as interactive W&B Tables.

    - Metrics: The MMDetWandbHook will automatically log training
        and validation metrics along with system metrics (CPU/GPU).

    - Checkpointing: If `log_checkpoint` is True, the checkpoint saved at
        every checkpoint interval will be saved as W&B Artifacts.
        This depends on the : class:`mmcv.runner.CheckpointHook` whose priority
        is higher than this hook. Please refer to
        https://docs.wandb.ai/guides/artifacts/model-versioning
        to learn more about model versioning with W&B Artifacts.

    - Checkpoint Metadata: If evaluation results are available for a given
        checkpoint artifact, it will have a metadata associated with it.
        The metadata contains the evaluation metrics computed on validation
        data with that checkpoint along with the current epoch. It depends
        on `EvalHook` whose priority is more than MMDetWandbHook.

    - Evaluation: At every evaluation interval, the `MMDetWandbHook` logs the
        model prediction as interactive W&B Tables. The number of samples
        logged is given by `num_eval_images`. Currently, the `MMDetWandbHook`
        logs the predicted bounding boxes along with the ground truth at every
        evaluation interval. This depends on the `EvalHook` whose priority is
        more than `MMDetWandbHook`. Also note that the data is just logged once
        and subsequent evaluation tables uses reference to the logged data
        to save memory usage. Please refer to
        https://docs.wandb.ai/guides/data-vis to learn more about W&B Tables.

    For more details check out W&B's MMDetection docs:
    https://docs.wandb.ai/guides/integrations/mmdetection

    ```
    Example:
        log_config = dict(
            ...
            hooks=[
                ...,
                dict(type='MMDetWandbHook',
                     init_kwargs={
                         'entity': "YOUR_ENTITY",
                         'project': "YOUR_PROJECT_NAME"
                     },
                     interval=50,
                     log_checkpoint=True,
                     log_checkpoint_metadata=True,
                     num_eval_images=100,
                     bbox_score_thr=0.3)
            ])
    ```

    Args:
        init_kwargs (dict): A dict passed to wandb.init to initialize
            a W&B run. Please refer to https://docs.wandb.ai/ref/python/init
            for possible key-value pairs.
        interval (int): Logging interval (every k iterations). Defaults to 50.
        log_checkpoint (bool): Save the checkpoint at every checkpoint interval
            as W&B Artifacts. Use this for model versioning where each version
            is a checkpoint. Defaults to False.
        log_checkpoint_metadata (bool): Log the evaluation metrics computed
            on the validation data with the checkpoint, along with current
            epoch as a metadata to that checkpoint.
            Defaults to True.
        num_eval_images (int): The number of validation images to be logged.
            If zero, the evaluation won't be logged. Defaults to 100.
        bbox_score_thr (float): Threshold for bounding box scores.
            Defaults to 0.3.
    """

    def __init__(self, init_kwargs=None, interval=50, log_checkpoint=False, log_checkpoint_metadata=False, num_eval_images=100, bbox_score_thr=0.3, **kwargs):
        super(MMDetWandbHook, self).__init__(init_kwargs, interval, **kwargs)
        self.log_checkpoint = log_checkpoint
        self.log_checkpoint_metadata = log_checkpoint and log_checkpoint_metadata
        self.num_eval_images = num_eval_images
        self.bbox_score_thr = bbox_score_thr
        self.log_evaluation = num_eval_images > 0
        self.ckpt_hook: CheckpointHook = None
        self.eval_hook: EvalHook = None

    def import_wandb(self):
        try:
            import wandb
            from wandb import init
            if digit_version(wandb.__version__) < digit_version('0.12.10'):
                warnings.warn(f'The current wandb {wandb.__version__} is lower than v0.12.10 will cause ResourceWarning when calling wandb.log, Please run "pip install --upgrade wandb"')
        except ImportError:
            raise ImportError('Please run "pip install "wandb>=0.12.10"" to install wandb')
        self.wandb = wandb

    @master_only
    def before_run(self, runner):
        super(MMDetWandbHook, self).before_run(runner)
        if runner.meta is not None and runner.meta.get('exp_name', None) is not None:
            src_cfg_path = osp.join(runner.work_dir, runner.meta.get('exp_name', None))
            if osp.exists(src_cfg_path):
                self.wandb.save(src_cfg_path, base_path=runner.work_dir)
                self._update_wandb_config(runner)
        else:
            runner.logger.warning('No meta information found in the runner. ')
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                self.ckpt_hook = hook
            if isinstance(hook, (EvalHook, DistEvalHook)):
                self.eval_hook = hook
        if self.log_checkpoint:
            if self.ckpt_hook is None:
                self.log_checkpoint = False
                self.log_checkpoint_metadata = False
                runner.logger.warning('To log checkpoint in MMDetWandbHook, `CheckpointHook` isrequired, please check hooks in the runner.')
            else:
                self.ckpt_interval = self.ckpt_hook.interval
        if self.log_evaluation or self.log_checkpoint_metadata:
            if self.eval_hook is None:
                self.log_evaluation = False
                self.log_checkpoint_metadata = False
                runner.logger.warning('To log evaluation or checkpoint metadata in MMDetWandbHook, `EvalHook` or `DistEvalHook` in mmdet is required, please check whether the validation is enabled.')
            else:
                self.eval_interval = self.eval_hook.interval
                self.val_dataset = self.eval_hook.dataloader.dataset
                if self.num_eval_images > len(self.val_dataset):
                    self.num_eval_images = len(self.val_dataset)
                    runner.logger.warning(f'The num_eval_images ({self.num_eval_images}) is greater than the total number of validation samples ({len(self.val_dataset)}). The complete validation dataset will be logged.')
        if self.log_checkpoint_metadata:
            assert self.ckpt_interval % self.eval_interval == 0, f'To log checkpoint metadata in MMDetWandbHook, the interval of checkpoint saving ({self.ckpt_interval}) should be divisible by the interval of evaluation ({self.eval_interval}).'
        if self.log_evaluation:
            self._init_data_table()
            self._add_ground_truth(runner)
            self._log_data_table()

    @master_only
    def after_train_epoch(self, runner):
        super(MMDetWandbHook, self).after_train_epoch(runner)
        if not self.by_epoch:
            return
        if self.log_checkpoint and self.every_n_epochs(runner, self.ckpt_interval) or (self.ckpt_hook.save_last and self.is_last_epoch(runner)):
            if self.log_checkpoint_metadata and self.eval_hook:
                metadata = {'epoch': runner.epoch + 1, **self._get_eval_results()}
            else:
                metadata = None
            aliases = [f'epoch_{runner.epoch + 1}', 'latest']
            model_path = osp.join(self.ckpt_hook.out_dir, f'epoch_{runner.epoch + 1}.pth')
            self._log_ckpt_as_artifact(model_path, aliases, metadata)
        if self.log_evaluation and self.eval_hook._should_evaluate(runner):
            results = self.eval_hook.latest_results
            self._init_pred_table()
            self._log_predictions(results)
            self._log_eval_table(runner.epoch + 1)

    @master_only
    def after_train_iter(self, runner):
        if self.get_mode(runner) == 'train':
            return super(MMDetWandbHook, self).after_train_iter(runner)
        else:
            super(MMDetWandbHook, self).after_train_iter(runner)
        if self.by_epoch:
            return
        if self.log_checkpoint and self.every_n_iters(runner, self.ckpt_interval) or (self.ckpt_hook.save_last and self.is_last_iter(runner)):
            if self.log_checkpoint_metadata and self.eval_hook:
                metadata = {'iter': runner.iter + 1, **self._get_eval_results()}
            else:
                metadata = None
            aliases = [f'iter_{runner.iter + 1}', 'latest']
            model_path = osp.join(self.ckpt_hook.out_dir, f'iter_{runner.iter + 1}.pth')
            self._log_ckpt_as_artifact(model_path, aliases, metadata)
        if self.log_evaluation and self.eval_hook._should_evaluate(runner):
            results = self.eval_hook.latest_results
            self._init_pred_table()
            self._log_predictions(results)
            self._log_eval_table(runner.iter + 1)

    @master_only
    def after_run(self, runner):
        self.wandb.finish()

    def _update_wandb_config(self, runner):
        """Update wandb config."""
        sys.path.append(runner.work_dir)
        config_filename = runner.meta['exp_name'][:-3]
        configs = importlib.import_module(config_filename)
        config_keys = [key for key in dir(configs) if not key.startswith('__')]
        config_dict = {key: getattr(configs, key) for key in config_keys}
        self.wandb.config.update(config_dict)

    def _log_ckpt_as_artifact(self, model_path, aliases, metadata=None):
        """Log model checkpoint as  W&B Artifact.

        Args:
            model_path (str): Path of the checkpoint to log.
            aliases (list): List of the aliases associated with this artifact.
            metadata (dict, optional): Metadata associated with this artifact.
        """
        model_artifact = self.wandb.Artifact(f'run_{self.wandb.run.id}_model', type='model', metadata=metadata)
        model_artifact.add_file(model_path)
        self.wandb.log_artifact(model_artifact, aliases=aliases)

    def _get_eval_results(self):
        """Get model evaluation results."""
        results = self.eval_hook.latest_results
        eval_results = self.val_dataset.evaluate(results, logger='silent', **self.eval_hook.eval_kwargs)
        return eval_results

    def _init_data_table(self):
        """Initialize the W&B Tables for validation data."""
        columns = ['image_name', 'image']
        self.data_table = self.wandb.Table(columns=columns)

    def _init_pred_table(self):
        """Initialize the W&B Tables for model evaluation."""
        columns = ['image_name', 'ground_truth', 'prediction']
        self.eval_table = self.wandb.Table(columns=columns)

    def _add_ground_truth(self, runner):
        from mmdet.datasets.pipelines import LoadImageFromFile
        img_loader = None
        for t in self.val_dataset.pipeline.transforms:
            if isinstance(t, LoadImageFromFile):
                img_loader = t
        if img_loader is None:
            self.log_evaluation = False
            runner.logger.warning('LoadImageFromFile is required to add images to W&B Tables.')
            return
        self.eval_image_indexs = np.arange(len(self.val_dataset))
        np.random.seed(42)
        np.random.shuffle(self.eval_image_indexs)
        self.eval_image_indexs = self.eval_image_indexs[:self.num_eval_images]
        CLASSES = self.val_dataset.CLASSES
        self.class_id_to_label = {id + 1: name for id, name in enumerate(CLASSES)}
        self.class_set = self.wandb.Classes([{'id': id, 'name': name} for id, name in self.class_id_to_label.items()])
        img_prefix = self.val_dataset.img_prefix
        for idx in self.eval_image_indexs:
            img_info = self.val_dataset.data_infos[idx]
            image_name = img_info.get('filename', f'img_{idx}')
            img_height, img_width = (img_info['height'], img_info['width'])
            img_meta = img_loader(dict(img_info=img_info, img_prefix=img_prefix))
            image = mmcv.bgr2rgb(img_meta['img'])
            data_ann = self.val_dataset.get_ann_info(idx)
            bboxes = data_ann['bboxes']
            labels = data_ann['labels']
            masks = data_ann.get('masks', None)
            assert len(bboxes) == len(labels)
            wandb_boxes = self._get_wandb_bboxes(bboxes, labels)
            if masks is not None:
                wandb_masks = self._get_wandb_masks(masks, labels, is_poly_mask=True, height=img_height, width=img_width)
            else:
                wandb_masks = None
            self.data_table.add_data(image_name, self.wandb.Image(image, boxes=wandb_boxes, masks=wandb_masks, classes=self.class_set))

    def _log_predictions(self, results):
        table_idxs = self.data_table_ref.get_index()
        assert len(table_idxs) == len(self.eval_image_indexs)
        for ndx, eval_image_index in enumerate(self.eval_image_indexs):
            result = results[eval_image_index]
            if isinstance(result, tuple):
                bbox_result, segm_result = result
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]
            else:
                bbox_result, segm_result = (result, None)
            assert len(bbox_result) == len(self.class_id_to_label)
            bboxes = np.vstack(bbox_result)
            labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
            labels = np.concatenate(labels)
            segms = None
            if segm_result is not None and len(labels) > 0:
                segms = mmcv.concat_list(segm_result)
                segms = mask_util.decode(segms)
                segms = segms.transpose(2, 0, 1)
                assert len(segms) == len(labels)
            if self.bbox_score_thr > 0:
                assert bboxes is not None and bboxes.shape[1] == 5
                scores = bboxes[:, -1]
                inds = scores > self.bbox_score_thr
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                if segms is not None:
                    segms = segms[inds, ...]
            wandb_boxes = self._get_wandb_bboxes(bboxes, labels, log_gt=False)
            if segms is not None:
                wandb_masks = self._get_wandb_masks(segms, labels)
            else:
                wandb_masks = None
            self.eval_table.add_data(self.data_table_ref.data[ndx][0], self.data_table_ref.data[ndx][1], self.wandb.Image(self.data_table_ref.data[ndx][1], boxes=wandb_boxes, masks=wandb_masks, classes=self.class_set))

    def _get_wandb_bboxes(self, bboxes, labels, log_gt=True):
        """Get list of structured dict for logging bounding boxes to W&B.

        Args:
            bboxes (list): List of bounding box coordinates in
                        (minX, minY, maxX, maxY) format.
            labels (int): List of label ids.
            log_gt (bool): Whether to log ground truth or prediction boxes.

        Returns:
            Dictionary of bounding boxes to be logged.
        """
        wandb_boxes = {}
        box_data = []
        for bbox, label in zip(bboxes, labels):
            if not isinstance(label, int):
                label = int(label)
            label = label + 1
            if len(bbox) == 5:
                confidence = float(bbox[4])
                class_name = self.class_id_to_label[label]
                box_caption = f'{class_name} {confidence:.2f}'
            else:
                box_caption = str(self.class_id_to_label[label])
            position = dict(minX=int(bbox[0]), minY=int(bbox[1]), maxX=int(bbox[2]), maxY=int(bbox[3]))
            box_data.append({'position': position, 'class_id': label, 'box_caption': box_caption, 'domain': 'pixel'})
        wandb_bbox_dict = {'box_data': box_data, 'class_labels': self.class_id_to_label}
        if log_gt:
            wandb_boxes['ground_truth'] = wandb_bbox_dict
        else:
            wandb_boxes['predictions'] = wandb_bbox_dict
        return wandb_boxes

    def _get_wandb_masks(self, masks, labels, is_poly_mask=False, height=None, width=None):
        """Get list of structured dict for logging masks to W&B.

        Args:
            masks (list): List of masks.
            labels (int): List of label ids.
            is_poly_mask (bool): Whether the mask is polygonal or not.
                This is true for CocoDataset.
            height (int): Height of the image.
            width (int): Width of the image.

        Returns:
            Dictionary of masks to be logged.
        """
        mask_label_dict = dict()
        for mask, label in zip(masks, labels):
            label = label + 1
            if is_poly_mask:
                if height is not None and width is not None:
                    mask = polygon_to_bitmap(mask, height, width)
            if label not in mask_label_dict.keys():
                mask_label_dict[label] = mask
            else:
                mask_label_dict[label] = np.logical_or(mask_label_dict[label], mask)
        wandb_masks = dict()
        for key, value in mask_label_dict.items():
            value = value.astype(np.uint8)
            value[value > 0] = key
            class_name = self.class_id_to_label[key]
            wandb_masks[class_name] = {'mask_data': value, 'class_labels': self.class_id_to_label}
        return wandb_masks

    def _log_data_table(self):
        """Log the W&B Tables for validation data as artifact and calls
        `use_artifact` on it so that the evaluation table can use the reference
        of already uploaded images.

        This allows the data to be uploaded just once.
        """
        data_artifact = self.wandb.Artifact('val', type='dataset')
        data_artifact.add(self.data_table, 'val_data')
        if not self.wandb.run.offline:
            self.wandb.run.use_artifact(data_artifact)
            data_artifact.wait()
            self.data_table_ref = data_artifact.get('val_data')
        else:
            self.data_table_ref = self.data_table

    def _log_eval_table(self, idx):
        """Log the W&B Tables for model evaluation.

        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.
        """
        pred_artifact = self.wandb.Artifact(f'run_{self.wandb.run.id}_pred', type='evaluation')
        pred_artifact.add(self.eval_table, 'eval_data')
        if self.by_epoch:
            aliases = ['latest', f'epoch_{idx}']
        else:
            aliases = ['latest', f'iter_{idx}']
        self.wandb.run.log_artifact(pred_artifact, aliases=aliases)

def polygon_to_bitmap(polygons, height, width):
    """Convert masks from the form of polygons to bitmaps.

    Args:
        polygons (list[ndarray]): masks in polygon representation
        height (int): mask height
        width (int): mask width

    Return:
        ndarray: the converted masks in bitmap representation
    """
    rles = maskUtils.frPyObjects(polygons, height, width)
    rle = maskUtils.merge(rles)
    bitmap_mask = maskUtils.decode(rle).astype(np.bool)
    return bitmap_mask

