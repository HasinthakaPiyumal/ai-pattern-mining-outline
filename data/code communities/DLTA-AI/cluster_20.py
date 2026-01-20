# Cluster 20

def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version

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

class SpatialReductionAttention(MultiheadAttention):
    """An implementation of Spatial Reduction Attention of PVT.

    This module is modified from MultiheadAttention which is a module from
    mmcv.cnn.bricks.transformer.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
            Attention of PVT. Default: 1.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self, embed_dims, num_heads, attn_drop=0.0, proj_drop=0.0, dropout_layer=None, batch_first=True, qkv_bias=True, norm_cfg=dict(type='LN'), sr_ratio=1, init_cfg=None):
        super().__init__(embed_dims, num_heads, attn_drop, proj_drop, batch_first=batch_first, dropout_layer=dropout_layer, bias=qkv_bias, init_cfg=init_cfg)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        from mmdet import digit_version, mmcv_version
        if mmcv_version < digit_version('1.3.17'):
            warnings.warn('The legacy version of forward function inSpatialReductionAttention is deprecated inmmcv>=1.3.17 and will no longer support in thefuture. Please upgrade your mmcv.')
            self.forward = self.legacy_forward

    def forward(self, x, hw_shape, identity=None):
        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x
        if identity is None:
            identity = x_q
        if self.batch_first:
            x_q = x_q.transpose(0, 1)
            x_kv = x_kv.transpose(0, 1)
        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]
        if self.batch_first:
            out = out.transpose(0, 1)
        return identity + self.dropout_layer(self.proj_drop(out))

    def legacy_forward(self, x, hw_shape, identity=None):
        """multi head attention forward in mmcv version < 1.3.17."""
        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x
        if identity is None:
            identity = x_q
        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]
        return identity + self.dropout_layer(self.proj_drop(out))

def pytorch2onnx(model, input_img, input_shape, normalize_cfg, opset_version=11, show=False, output_file='tmp.onnx', verify=False, test_img=None, do_simplify=False, dynamic_export=None, skip_postprocess=False):
    input_config = {'input_shape': input_shape, 'input_path': input_img, 'normalize_cfg': normalize_cfg}
    one_img, one_meta = preprocess_example_input(input_config)
    img_list, img_meta_list = ([one_img], [[one_meta]])
    if skip_postprocess:
        warnings.warn('Not all models support export onnx without post process, especially two stage detectors!')
        model.forward = model.forward_dummy
        torch.onnx.export(model, one_img, output_file, input_names=['input'], export_params=True, keep_initializers_as_inputs=True, do_constant_folding=True, verbose=show, opset_version=opset_version)
        print(f'Successfully exported ONNX model without post process: {output_file}')
        return
    origin_forward = model.forward
    model.forward = partial(model.forward, img_metas=img_meta_list, return_loss=False, rescale=False)
    output_names = ['dets', 'labels']
    if model.with_mask:
        output_names.append('masks')
    input_name = 'input'
    dynamic_axes = None
    if dynamic_export:
        dynamic_axes = {input_name: {0: 'batch', 2: 'height', 3: 'width'}, 'dets': {0: 'batch', 1: 'num_dets'}, 'labels': {0: 'batch', 1: 'num_dets'}}
        if model.with_mask:
            dynamic_axes['masks'] = {0: 'batch', 1: 'num_dets'}
    torch.onnx.export(model, img_list, output_file, input_names=[input_name], output_names=output_names, export_params=True, keep_initializers_as_inputs=True, do_constant_folding=True, verbose=show, opset_version=opset_version, dynamic_axes=dynamic_axes)
    model.forward = origin_forward
    if do_simplify:
        import onnxsim
        from mmdet import digit_version
        min_required_version = '0.4.0'
        assert digit_version(onnxsim.__version__) >= digit_version(min_required_version), f'Requires to install onnxsim>={min_required_version}'
        model_opt, check_ok = onnxsim.simplify(output_file)
        if check_ok:
            onnx.save(model_opt, output_file)
            print(f'Successfully simplified ONNX model: {output_file}')
        else:
            warnings.warn('Failed to simplify ONNX model.')
    print(f'Successfully exported ONNX model: {output_file}')
    if verify:
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        onnx_model = ONNXRuntimeDetector(output_file, model.CLASSES, 0)
        if dynamic_export:
            h, w = [int(_ * 1.5 // 32 * 32) for _ in input_shape[2:]]
            h, w = (min(1344, h), min(1344, w))
            input_config['input_shape'] = (1, 3, h, w)
        if test_img is None:
            input_config['input_path'] = input_img
        one_img, one_meta = preprocess_example_input(input_config)
        img_list, img_meta_list = ([one_img], [[one_meta]])
        with torch.no_grad():
            pytorch_results = model(img_list, img_metas=img_meta_list, return_loss=False, rescale=True)[0]
        img_list = [_.cuda().contiguous() for _ in img_list]
        if dynamic_export:
            img_list = img_list + [_.flip(-1).contiguous() for _ in img_list]
            img_meta_list = img_meta_list * 2
        onnx_results = onnx_model(img_list, img_metas=img_meta_list, return_loss=False)[0]
        score_thr = 0.3
        if show:
            out_file_ort, out_file_pt = (None, None)
        else:
            out_file_ort, out_file_pt = ('show-ort.png', 'show-pt.png')
        show_img = one_meta['show_img']
        model.show_result(show_img, pytorch_results, score_thr=score_thr, show=True, win_name='PyTorch', out_file=out_file_pt)
        onnx_model.show_result(show_img, onnx_results, score_thr=score_thr, show=True, win_name='ONNXRuntime', out_file=out_file_ort)
        if model.with_mask:
            compare_pairs = list(zip(onnx_results, pytorch_results))
        else:
            compare_pairs = [(onnx_results, pytorch_results)]
        err_msg = 'The numerical values are different between Pytorch' + ' and ONNX, but it does not necessarily mean the' + ' exported ONNX model is problematic.'
        for onnx_res, pytorch_res in compare_pairs:
            for o_res, p_res in zip(onnx_res, pytorch_res):
                np.testing.assert_allclose(o_res, p_res, rtol=0.001, atol=1e-05, err_msg=err_msg)
        print('The numerical values are the same between Pytorch and ONNX')

def preprocess_example_input(input_config):
    """Prepare an example input image for ``generate_inputs_and_wrap_model``.

    Args:
        input_config (dict): customized config describing the example input.

    Returns:
        tuple: (one_img, one_meta), tensor of the example input image and             meta information for the example input image.

    Examples:
        >>> from mmdet.core.export import preprocess_example_input
        >>> input_config = {
        >>>         'input_shape': (1,3,224,224),
        >>>         'input_path': 'demo/demo.jpg',
        >>>         'normalize_cfg': {
        >>>             'mean': (123.675, 116.28, 103.53),
        >>>             'std': (58.395, 57.12, 57.375)
        >>>             }
        >>>         }
        >>> one_img, one_meta = preprocess_example_input(input_config)
        >>> print(one_img.shape)
        torch.Size([1, 3, 224, 224])
        >>> print(one_meta)
        {'img_shape': (224, 224, 3),
        'ori_shape': (224, 224, 3),
        'pad_shape': (224, 224, 3),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False}
    """
    input_path = input_config['input_path']
    input_shape = input_config['input_shape']
    one_img = mmcv.imread(input_path)
    one_img = mmcv.imresize(one_img, input_shape[2:][::-1])
    show_img = one_img.copy()
    if 'normalize_cfg' in input_config.keys():
        normalize_cfg = input_config['normalize_cfg']
        mean = np.array(normalize_cfg['mean'], dtype=np.float32)
        std = np.array(normalize_cfg['std'], dtype=np.float32)
        to_rgb = normalize_cfg.get('to_rgb', True)
        one_img = mmcv.imnormalize(one_img, mean, std, to_rgb=to_rgb)
    one_img = one_img.transpose(2, 0, 1)
    one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(True)
    _, C, H, W = input_shape
    one_meta = {'img_shape': (H, W, C), 'ori_shape': (H, W, C), 'pad_shape': (H, W, C), 'filename': '<demo>.png', 'scale_factor': np.ones(4, dtype=np.float32), 'flip': False, 'show_img': show_img, 'flip_direction': None}
    return (one_img, one_meta)

def build_model_from_cfg(config_path, checkpoint_path, cfg_options=None):
    """Build a model from config and load the given checkpoint.

    Args:
        config_path (str): the OpenMMLab config for the model we want to
            export to ONNX
        checkpoint_path (str): Path to the corresponding checkpoint

    Returns:
        torch.nn.Module: the built model
    """
    from mmdet.models import build_detector
    cfg = mmcv.Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmdet.datasets import DATASETS
        dataset = DATASETS.get(cfg.data.test['type'])
        assert dataset is not None
        model.CLASSES = dataset.CLASSES
    model.cpu().eval()
    return model

def parse_normalize_cfg(test_pipeline):
    transforms = None
    for pipeline in test_pipeline:
        if 'transforms' in pipeline:
            transforms = pipeline['transforms']
            break
    assert transforms is not None, 'Failed to find `transforms`'
    norm_config_li = [_ for _ in transforms if _['type'] == 'Normalize']
    assert len(norm_config_li) == 1, '`norm_config` should only have one'
    norm_config = norm_config_li[0]
    return norm_config

def onnx2tensorrt(onnx_file, trt_file, input_config, verify=False, show=False, workspace_size=1, verbose=False):
    import tensorrt as trt
    onnx_model = onnx.load(onnx_file)
    max_shape = input_config['max_shape']
    min_shape = input_config['min_shape']
    opt_shape = input_config['opt_shape']
    fp16_mode = False
    opt_shape_dict = {'input': [min_shape, opt_shape, max_shape]}
    max_workspace_size = get_GiB(workspace_size)
    trt_engine = onnx2trt(onnx_model, opt_shape_dict, log_level=trt.Logger.VERBOSE if verbose else trt.Logger.ERROR, fp16_mode=fp16_mode, max_workspace_size=max_workspace_size)
    save_dir, _ = osp.split(trt_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    save_trt_engine(trt_engine, trt_file)
    print(f'Successfully created TensorRT engine: {trt_file}')
    if verify:
        one_img, one_meta = preprocess_example_input(input_config)
        img_list, img_meta_list = ([one_img], [[one_meta]])
        img_list = [_.cuda().contiguous() for _ in img_list]
        onnx_model = ONNXRuntimeDetector(onnx_file, CLASSES, device_id=0)
        trt_model = TensorRTDetector(trt_file, CLASSES, device_id=0)
        with torch.no_grad():
            onnx_results = onnx_model(img_list, img_metas=img_meta_list, return_loss=False)[0]
            trt_results = trt_model(img_list, img_metas=img_meta_list, return_loss=False)[0]
        if show:
            out_file_ort, out_file_trt = (None, None)
        else:
            out_file_ort, out_file_trt = ('show-ort.png', 'show-trt.png')
        show_img = one_meta['show_img']
        score_thr = 0.3
        onnx_model.show_result(show_img, onnx_results, score_thr=score_thr, show=True, win_name='ONNXRuntime', out_file=out_file_ort)
        trt_model.show_result(show_img, trt_results, score_thr=score_thr, show=True, win_name='TensorRT', out_file=out_file_trt)
        with_mask = trt_model.with_masks
        if with_mask:
            compare_pairs = list(zip(onnx_results, trt_results))
        else:
            compare_pairs = [(onnx_results, trt_results)]
        err_msg = 'The numerical values are different between Pytorch' + ' and ONNX, but it does not necessarily mean the' + ' exported ONNX model is problematic.'
        for onnx_res, pytorch_res in compare_pairs:
            for o_res, p_res in zip(onnx_res, pytorch_res):
                np.testing.assert_allclose(o_res, p_res, rtol=0.001, atol=1e-05, err_msg=err_msg)
        print('The numerical values are the same between Pytorch and ONNX')

def get_GiB(x: int):
    """return x GiB."""
    return x * (1 << 30)

def parse_shape(shape):
    if len(shape) == 1:
        shape = (1, 3, shape[0], shape[0])
    elif len(args.shape) == 2:
        shape = (1, 3) + tuple(shape)
    else:
        raise ValueError('invalid input shape')
    return shape

@pytest.mark.parametrize('loss_class', [FocalLoss, CrossEntropyLoss])
@pytest.mark.parametrize('input_shape', [(10, 5), (0, 5)])
def test_classification_losses(loss_class, input_shape):
    if input_shape[0] == 0 and digit_version(torch.__version__) < digit_version('1.5.0'):
        pytest.skip(f'CELoss in PyTorch {torch.__version__} does not support emptytensor.')
    pred = torch.rand(input_shape)
    target = torch.randint(0, 5, (input_shape[0],))
    loss = loss_class()(pred, target)
    assert isinstance(loss, torch.Tensor)
    loss = loss_class()(pred, target, reduction_override='mean')
    assert isinstance(loss, torch.Tensor)
    loss = loss_class()(pred, target, avg_factor=10)
    assert isinstance(loss, torch.Tensor)
    with pytest.raises(ValueError):
        reduction_override = 'sum'
        loss_class()(pred, target, avg_factor=10, reduction_override=reduction_override)
    for reduction_override in [None, 'none', 'mean']:
        loss_class()(pred, target, avg_factor=10, reduction_override=reduction_override)
        assert isinstance(loss, torch.Tensor)

def test_version_check():
    assert digit_version('1.0.5') > digit_version('1.0.5rc0')
    assert digit_version('1.0.5') > digit_version('1.0.4rc0')
    assert digit_version('1.0.5') > digit_version('1.0rc0')
    assert digit_version('1.0.0') > digit_version('0.6.2')
    assert digit_version('1.0.0') > digit_version('0.2.16')
    assert digit_version('1.0.5rc0') > digit_version('1.0.0rc0')
    assert digit_version('1.0.0rc1') > digit_version('1.0.0rc0')
    assert digit_version('1.0.0rc2') > digit_version('1.0.0rc0')
    assert digit_version('1.0.0rc2') > digit_version('1.0.0rc1')
    assert digit_version('1.0.1rc1') > digit_version('1.0.0rc1')
    assert digit_version('1.0.0') > digit_version('1.0.0rc1')

def test_general_data():
    meta_info = dict(img_size=[256, 256], path='dadfaff', scale_factor=np.array([1.5, 1.5]), img_shape=torch.rand(4))
    data = dict(bboxes=torch.rand(4, 4), labels=torch.rand(4), masks=np.random.rand(4, 2, 2))
    instance_data = GeneralData(meta_info=meta_info)
    assert 'img_size' in instance_data
    assert instance_data.img_size == [256, 256]
    assert instance_data['img_size'] == [256, 256]
    assert 'path' in instance_data
    assert instance_data.path == 'dadfaff'
    repr_instance_data = instance_data.new(data=data)
    nice_repr = str(repr_instance_data)
    for line in nice_repr.split('\n'):
        if 'masks' in line:
            assert 'shape' in line
            assert '(4, 2, 2)' in line
        if 'bboxes' in line:
            assert 'shape' in line
            assert 'torch.Size([4, 4])' in line
        if 'path' in line:
            assert 'dadfaff' in line
        if 'scale_factor' in line:
            assert '[1.5 1.5]' in line
    instance_data = GeneralData(meta_info=meta_info, data=dict(bboxes=torch.rand(5)))
    assert 'bboxes' in instance_data
    assert len(instance_data.bboxes) == 5
    with pytest.raises(AssertionError):
        GeneralData(data=1)
    instance_data = GeneralData()
    instance_data.set_data(data)
    assert 'bboxes' in instance_data
    assert len(instance_data.bboxes) == 4
    assert 'masks' in instance_data
    assert len(instance_data.masks) == 4
    with pytest.raises(AssertionError):
        instance_data.set_data(data=1)
    instance_data = GeneralData()
    instance_data.set_meta_info(meta_info)
    assert 'img_size' in instance_data
    assert instance_data.img_size == [256, 256]
    assert instance_data['img_size'] == [256, 256]
    assert 'path' in instance_data
    assert instance_data.path == 'dadfaff'
    instance_data.set_meta_info(meta_info)
    with pytest.raises(AssertionError):
        instance_data.set_meta_info(meta_info='fjhka')
    instance_data.set_meta_info(meta_info)
    with pytest.raises(KeyError):
        instance_data.set_meta_info(dict(img_size=[254, 251]))
    with pytest.raises(KeyError):
        duplicate_meta_info = copy.deepcopy(meta_info)
        duplicate_meta_info['path'] = 'dada'
        instance_data.set_meta_info(duplicate_meta_info)
    with pytest.raises(KeyError):
        duplicate_meta_info = copy.deepcopy(meta_info)
        duplicate_meta_info['scale_factor'] = np.array([1.5, 1.6])
        instance_data.set_meta_info(duplicate_meta_info)
    instance_data = GeneralData(meta_info)
    new_instance_data = instance_data.new()
    for k, v in instance_data.meta_info_items():
        assert k in new_instance_data
        _equal(v, new_instance_data[k])
    instance_data = GeneralData(meta_info, data=data)
    temp_meta = copy.deepcopy(meta_info)
    temp_data = copy.deepcopy(data)
    temp_data['time'] = '12212'
    temp_meta['img_norm'] = np.random.random(3)
    new_instance_data = instance_data.new(meta_info=temp_meta, data=temp_data)
    for k, v in new_instance_data.meta_info_items():
        if k in instance_data:
            _equal(v, instance_data[k])
        else:
            assert _equal(v, temp_meta[k])
            assert k == 'img_norm'
    for k, v in new_instance_data.items():
        if k in instance_data:
            _equal(v, instance_data[k])
        else:
            assert k == 'time'
            assert _equal(v, temp_data[k])
    instance_data = GeneralData(meta_info, data=dict(bboxes=10))
    assert 'bboxes' in instance_data.keys()
    instance_data.b = 10
    assert 'b' in instance_data
    instance_data = GeneralData(meta_info, data=dict(bboxes=10))
    assert 'path' in instance_data.meta_info_keys()
    assert len(instance_data.meta_info_keys()) == len(meta_info)
    instance_data.set_meta_info(dict(workdir='fafaf'))
    assert 'workdir' in instance_data
    assert len(instance_data.meta_info_keys()) == len(meta_info) + 1
    instance_data = GeneralData(meta_info, data=dict(bboxes=10))
    assert 10 in instance_data.values()
    assert len(instance_data.values()) == 1
    instance_data = GeneralData(meta_info, data=dict(bboxes=10))
    from mmdet import digit_version
    if digit_version(torch.__version__) >= [1, 4]:
        assert 'dadfaff' in instance_data.meta_info_values()
    assert len(instance_data.meta_info_values()) == len(meta_info)
    instance_data = GeneralData(data=data)
    for k, v in instance_data.items():
        assert k in data
        assert _equal(v, data[k])
    instance_data = GeneralData(meta_info=meta_info)
    for k, v in instance_data.meta_info_items():
        assert k in meta_info
        assert _equal(v, meta_info[k])
    new_instance_data = GeneralData(data=data)
    new_instance_data.mask = torch.rand(3, 4, 5)
    new_instance_data.bboxes = torch.rand(2, 4)
    assert 'mask' in new_instance_data
    assert len(new_instance_data.mask) == 3
    assert len(new_instance_data.bboxes) == 2
    assert 'mask' in new_instance_data._data_fields
    assert 'bboxes' in new_instance_data._data_fields
    for k in data:
        assert k in new_instance_data._data_fields
    with pytest.raises(AttributeError):
        new_instance_data._data_fields = None
    with pytest.raises(AttributeError):
        new_instance_data._meta_info_fields = None
    with pytest.raises(AttributeError):
        del new_instance_data._data_fields
    with pytest.raises(AttributeError):
        del new_instance_data._meta_info_fields
    new_instance_data.set_meta_info(meta_info)
    with pytest.raises(KeyError):
        del new_instance_data.img_size
    with pytest.raises(KeyError):
        del new_instance_data.scale_factor
    for k in new_instance_data.meta_info_keys():
        with pytest.raises(AttributeError):
            new_instance_data[k] = None
    assert 'mask' in new_instance_data._data_fields
    assert 'mask' in new_instance_data.keys()
    assert 'mask' in new_instance_data
    assert hasattr(new_instance_data, 'mask')
    del new_instance_data.mask
    assert 'mask' not in new_instance_data.keys()
    assert 'mask' not in new_instance_data
    assert 'mask' not in new_instance_data._data_fields
    assert not hasattr(new_instance_data, 'mask')
    new_instance_data.mask = torch.rand(1, 2, 3)
    assert 'mask' in new_instance_data._data_fields
    assert 'mask' in new_instance_data
    assert hasattr(new_instance_data, 'mask')
    del new_instance_data['mask']
    assert 'mask' not in new_instance_data
    assert 'mask' not in new_instance_data._data_fields
    assert 'mask' not in new_instance_data
    assert not hasattr(new_instance_data, 'mask')
    new_instance_data['mask'] = torch.rand(1, 2, 3)
    assert 'mask' in new_instance_data._data_fields
    assert 'mask' in new_instance_data.keys()
    assert hasattr(new_instance_data, 'mask')
    assert 'mask' in new_instance_data.keys()
    assert 'mask' in new_instance_data._data_fields
    with pytest.raises(AttributeError):
        del new_instance_data['_data_fields']
    with pytest.raises(AttributeError):
        del new_instance_data['_meta_info_field']
    new_instance_data.mask is new_instance_data['mask']
    assert new_instance_data.get('mask') is new_instance_data.mask
    assert new_instance_data.get('none_attribute', None) is None
    assert new_instance_data.get('none_attribute', 1) == 1
    mask = new_instance_data.mask
    assert new_instance_data.pop('mask') is mask
    assert new_instance_data.pop('mask', None) is None
    assert new_instance_data.pop('mask', 1) == 1
    with pytest.raises(KeyError):
        new_instance_data.pop('_data_fields')
    with pytest.raises(KeyError):
        new_instance_data.pop('_meta_info_field')
    with pytest.raises(KeyError):
        new_instance_data.pop('img_size')
    new_instance_data['mask'] = torch.rand(1, 2, 3)
    new_instance_data.pop('mask')
    assert 'mask' not in new_instance_data
    assert 'mask' not in new_instance_data._data_fields
    assert 'mask' not in new_instance_data
    new_instance_data.mask = torch.ones(1, 2, 3)
    'mask' in new_instance_data.keys()
    has_flag = False
    for key in new_instance_data.keys():
        if key == 'mask':
            has_flag = True
    assert has_flag
    assert len(list(new_instance_data.keys())) == len(list(new_instance_data.values()))
    mask = new_instance_data.mask
    has_flag = False
    for value in new_instance_data.values():
        if value is mask:
            has_flag = True
    assert has_flag
    assert len(list(new_instance_data.keys())) == len(list(new_instance_data.items()))
    mask = new_instance_data.mask
    has_flag = False
    for key, value in new_instance_data.items():
        if value is mask:
            assert key == 'mask'
            has_flag = True
    assert has_flag
    new_instance_data = GeneralData()
    if torch.cuda.is_available():
        newnew_instance_data = new_instance_data.new()
        devices = ('cpu', 'cuda')
        for i in range(10):
            device = devices[i % 2]
            newnew_instance_data[f'{i}'] = torch.rand(1, 2, 3, device=device)
        newnew_instance_data = newnew_instance_data.cpu()
        for value in newnew_instance_data.values():
            assert not value.is_cuda
        newnew_instance_data = new_instance_data.new()
        devices = ('cuda', 'cpu')
        for i in range(10):
            device = devices[i % 2]
            newnew_instance_data[f'{i}'] = torch.rand(1, 2, 3, device=device)
        newnew_instance_data = newnew_instance_data.cuda()
        for value in newnew_instance_data.values():
            assert value.is_cuda
    double_instance_data = instance_data.new()
    double_instance_data.long = torch.LongTensor(1, 2, 3, 4)
    double_instance_data.bool = torch.BoolTensor(1, 2, 3, 4)
    double_instance_data = instance_data.to(torch.double)
    for k, v in double_instance_data.items():
        if isinstance(v, torch.Tensor):
            assert v.dtype is torch.double
    if torch.cuda.is_available():
        cpu_instance_data = double_instance_data.new()
        cpu_instance_data.mask = torch.rand(1)
        cuda_tensor = torch.rand(1, 2, 3).cuda()
        cuda_instance_data = cpu_instance_data.to(cuda_tensor.device)
        for value in cuda_instance_data.values():
            assert value.is_cuda
        cpu_instance_data = cuda_instance_data.cpu()
        for value in cpu_instance_data.values():
            assert not value.is_cuda
        cuda_instance_data = cpu_instance_data.cuda()
        for value in cuda_instance_data.values():
            assert value.is_cuda
    grad_instance_data = double_instance_data.new()
    grad_instance_data.mask = torch.rand(2, requires_grad=True)
    grad_instance_data.mask_1 = torch.rand(2, requires_grad=True)
    detach_instance_data = grad_instance_data.detach()
    for value in detach_instance_data.values():
        assert not value.requires_grad
    tensor_instance_data = double_instance_data.new()
    tensor_instance_data.mask = torch.rand(2, requires_grad=True)
    tensor_instance_data.mask_1 = torch.rand(2, requires_grad=True)
    numpy_instance_data = tensor_instance_data.numpy()
    for value in numpy_instance_data.values():
        assert isinstance(value, np.ndarray)
    if torch.cuda.is_available():
        tensor_instance_data = double_instance_data.new()
        tensor_instance_data.mask = torch.rand(2)
        tensor_instance_data.mask_1 = torch.rand(2)
        tensor_instance_data = tensor_instance_data.cuda()
        numpy_instance_data = tensor_instance_data.numpy()
        for value in numpy_instance_data.values():
            assert isinstance(value, np.ndarray)
    instance_data['_c'] = 10000
    instance_data.get('dad', None) is None
    assert hasattr(instance_data, '_c')
    del instance_data['_c']
    assert not hasattr(instance_data, '_c')
    instance_data.a = 1000
    instance_data['a'] = 2000
    assert instance_data['a'] == 2000
    assert instance_data.a == 2000
    assert instance_data.get('a') == instance_data['a'] == instance_data.a
    instance_data._meta = 1000
    assert '_meta' in instance_data.keys()
    if torch.cuda.is_available():
        instance_data.bbox = torch.ones(2, 3, 4, 5).cuda()
        instance_data.score = torch.ones(2, 3, 4, 4)
    else:
        instance_data.bbox = torch.ones(2, 3, 4, 5)
    assert len(instance_data.new().keys()) == 0
    with pytest.raises(AttributeError):
        instance_data.img_size = 100
    for k, v in instance_data.items():
        if k == 'bbox':
            assert isinstance(v, torch.Tensor)
    assert 'a' in instance_data
    instance_data.pop('a')
    assert 'a' not in instance_data
    cpu_instance_data = instance_data.cpu()
    for k, v in cpu_instance_data.items():
        if isinstance(v, torch.Tensor):
            assert not v.is_cuda
    assert isinstance(cpu_instance_data.numpy().bbox, np.ndarray)
    if torch.cuda.is_available():
        cuda_resutls = instance_data.cuda()
        for k, v in cuda_resutls.items():
            if isinstance(v, torch.Tensor):
                assert v.is_cuda

def _equal(a, b):
    if isinstance(a, (torch.Tensor, np.ndarray)):
        return (a == b).all()
    else:
        return a == b

def test_instance_data():
    meta_info = dict(img_size=(256, 256), path='dadfaff', scale_factor=np.array([1.5, 1.5, 1, 1]))
    data = dict(bboxes=torch.rand(4, 4), masks=torch.rand(4, 2, 2), labels=np.random.rand(4), size=[(i, i) for i in range(4)])
    instance_data = InstanceData(meta_info)
    assert 'path' in instance_data
    instance_data = InstanceData(meta_info, data=data)
    assert len(instance_data) == 4
    instance_data.set_data(data)
    assert len(instance_data) == 4
    meta_info = copy.deepcopy(meta_info)
    meta_info['img_name'] = 'flag'
    new_instance_data = instance_data.new(meta_info=meta_info)
    for k, v in new_instance_data.meta_info_items():
        if k in instance_data:
            _equal(v, instance_data[k])
        else:
            assert _equal(v, meta_info[k])
            assert k == 'img_name'
    with pytest.raises(KeyError):
        meta_info = copy.deepcopy(meta_info)
        meta_info['path'] = 'fdasfdsd'
        instance_data.new(meta_info=meta_info)
    with pytest.raises(AssertionError):
        temp_data = copy.deepcopy(data)
        temp_data['bboxes'] = torch.rand(5, 4)
        instance_data.new(data=temp_data)
    temp_data = copy.deepcopy(data)
    temp_data['scores'] = torch.rand(4)
    new_instance_data = instance_data.new(data=temp_data)
    for k, v in new_instance_data.items():
        if k in instance_data:
            _equal(v, instance_data[k])
        else:
            assert k == 'scores'
            assert _equal(v, temp_data[k])
    instance_data = instance_data.new()
    with pytest.raises(AttributeError):
        instance_data._data_fields = dict()
    with pytest.raises(AttributeError):
        instance_data._data_fields = dict()
    with pytest.raises(AssertionError):
        instance_data.a = 1000
    new_instance_data = instance_data.new()
    new_instance_data.det_bbox = torch.rand(100, 4)
    new_instance_data.det_label = torch.arange(100)
    with pytest.raises(AssertionError):
        new_instance_data.scores = torch.rand(101, 1)
    new_instance_data.none = [None] * 100
    with pytest.raises(AssertionError):
        new_instance_data.scores = [None] * 101
    new_instance_data.numpy_det = np.random.random([100, 1])
    with pytest.raises(AssertionError):
        new_instance_data.scores = np.random.random([101, 1])
    item = torch.Tensor([1, 2, 3, 4])
    with pytest.raises(AssertionError):
        new_instance_data[item]
    len(new_instance_data[item.long()]) == 1
    with pytest.raises(AssertionError):
        new_instance_data[item.bool()]
    for i in range(len(new_instance_data)):
        assert new_instance_data[i].det_label == i
        assert len(new_instance_data[i]) == 1
    with pytest.raises(IndexError):
        new_instance_data[101]
    new_new_instance_data = new_instance_data.new()
    with pytest.raises(AssertionError):
        new_new_instance_data[0]
    with pytest.raises(AssertionError):
        instance_data.img_size_dummmy = meta_info['img_size']
    ten_ressults = new_instance_data[:10]
    len(ten_ressults) == 10
    for v in ten_ressults.values():
        assert len(v) == 10
    long_tensor = torch.randint(100, (50,))
    long_index_instance_data = new_instance_data[long_tensor]
    assert len(long_index_instance_data) == len(long_tensor)
    for key, value in long_index_instance_data.items():
        if not isinstance(value, list):
            assert (long_index_instance_data[key] == new_instance_data[key][long_tensor]).all()
        else:
            len(long_tensor) == len(value)
    bool_tensor = torch.rand(100) > 0.5
    bool_index_instance_data = new_instance_data[bool_tensor]
    assert len(bool_index_instance_data) == bool_tensor.sum()
    for key, value in bool_index_instance_data.items():
        if not isinstance(value, list):
            assert (bool_index_instance_data[key] == new_instance_data[key][bool_tensor]).all()
        else:
            assert len(value) == bool_tensor.sum()
    num_instance = 1000
    instance_data_list = []
    with pytest.raises(AssertionError):
        instance_data.cat(instance_data_list)
    for _ in range(2):
        instance_data['bbox'] = torch.rand(num_instance, 4)
        instance_data['label'] = torch.rand(num_instance, 1)
        instance_data['mask'] = torch.rand(num_instance, 224, 224)
        instance_data['instances_infos'] = [1] * num_instance
        instance_data['cpu_bbox'] = np.random.random((num_instance, 4))
        if torch.cuda.is_available():
            instance_data.cuda_tensor = torch.rand(num_instance).cuda()
            assert instance_data.cuda_tensor.is_cuda
            cuda_instance_data = instance_data.cuda()
            assert cuda_instance_data.cuda_tensor.is_cuda
        assert len(instance_data[0]) == 1
        with pytest.raises(IndexError):
            return instance_data[num_instance + 1]
        with pytest.raises(AssertionError):
            instance_data.centerness = torch.rand(num_instance + 1, 1)
        mask_tensor = torch.rand(num_instance) > 0.5
        length = mask_tensor.sum()
        assert len(instance_data[mask_tensor]) == length
        index_tensor = torch.LongTensor([1, 5, 8, 110, 399])
        length = len(index_tensor)
        assert len(instance_data[index_tensor]) == length
        instance_data_list.append(instance_data)
    cat_resutls = InstanceData.cat(instance_data_list)
    assert len(cat_resutls) == num_instance * 2
    instances = InstanceData(data=dict(bboxes=torch.rand(4, 4)))
    assert len(InstanceData.cat([instances])) == 4

